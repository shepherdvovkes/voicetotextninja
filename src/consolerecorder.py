import sounddevice as sd
import numpy as np
import webrtcvad
import queue
import time
from datetime import datetime
import sys
import openai
import wave
import tempfile
import os
from dotenv import load_dotenv
from select import select
from rich.live import Live
from rich.panel import Panel
from rich.console import Console
from rich.layout import Layout
from rich.text import Text
from rich import box
import shutil

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = 'int16'
CHUNK_SEC = 10
VAD_LEVEL = 2
SILENCE_TIMEOUT = 3

vad = webrtcvad.Vad(VAD_LEVEL)
q = queue.Queue()
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

console = Console()

def audio_callback(indata, frames, time_info, status):
    if status:
        debug_log.append(f"[Status] {status}")
    q.put(indata.copy())

def vad_activity(chunk_bytes, sample_rate):
    frame_size = int(0.03 * sample_rate) * 2
    for i in range(0, len(chunk_bytes), frame_size):
        frame = chunk_bytes[i:i+frame_size]
        if len(frame) == frame_size:
            if vad.is_speech(frame, sample_rate):
                return True
    return False

def recognize_audio(audio_array, sample_rate=16000):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        with wave.open(tf.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_array.tobytes())
        tf.flush()
        with open(tf.name, "rb") as f:
            text = openai.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text"
            )
    return text, tf.name

def save_text(text, basename):
    fname = basename.replace('.wav', '.txt')
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text.strip())
    return fname

def split_sentences(text):
    import re
    return re.split(r'(?<=[.?!])\s+', text.strip())

def make_layout(dictation_text, debug_log, prompt_message):
    layout = Layout()
    layout.split_column(
        Layout(name="upper", ratio=3),
        Layout(name="middle", ratio=1),
        Layout(name="lower", size=4)
    )
    layout["upper"].update(
        Panel(
            Text(dictation_text, style="bold white on blue"),
            title="📝 Dictation",
            box=box.ROUNDED,
            padding=(1,1)
        )
    )
    layout["middle"].update(
        Panel(
            Text("\n".join(debug_log[-14:]), style="yellow"),
            title="⚙️ Debug Log",
            box=box.ROUNDED,
            padding=(0,1)
        )
    )
    layout["lower"].update(
        Panel(
            Text(prompt_message, style="bold magenta"),
            title="📢 Prompt / Status",
            box=box.ROUNDED
        )
    )
    return layout

def record_and_recognize_session():
    global dictation_text, debug_log, prompt_message
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE, callback=audio_callback, blocksize=int(SAMPLE_RATE * 0.5)):
        try:
            while True:
                frames = []
                vad_timeout = None
                silence_sec = 0
                t0 = time.time()
                prompt_message = "[green]Говорите! (до 10 сек или пауза >3 сек)[/green]"
                speech_active = False
                with Live(make_layout(dictation_text, debug_log, prompt_message), refresh_per_second=20, screen=True) as live:
                    while True:
                        now = time.time()
                        elapsed = now - t0
                        if elapsed >= CHUNK_SEC:
                            break
                        try:
                            chunk = q.get(timeout=0.05)
                        except queue.Empty:
                            continue
                        frames.append(chunk)
                        speech = vad_activity(chunk.tobytes(), SAMPLE_RATE)
                        speech_active = bool(speech)
                        if speech:
                            silence_sec = 0
                            vad_timeout = time.time()
                        else:
                            if vad_timeout:
                                silence_sec = time.time() - vad_timeout
                        debug_log.append(
                            f"[{int(elapsed):2d}s] {'🎤' if speech else '  '} | Silence: {silence_sec:.1f}s"
                        )
                        if silence_sec >= SILENCE_TIMEOUT:
                            debug_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Silence {SILENCE_TIMEOUT}s: fragment end.")
                            break
                        prompt_message = "[green]Говорите! (до 10 сек или пауза >3 сек)[/green]"
                        live.update(make_layout(dictation_text, debug_log, prompt_message))
                    audio = np.concatenate(frames, axis=0)
                    debug_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Fragment recorded. Sending to Whisper...")
                    prompt_message = "[yellow]Обработка...[/yellow]"
                    live.update(make_layout(dictation_text, debug_log, prompt_message))
                    try:
                        text, wavfile = recognize_audio(audio)
                    except Exception as e:
                        debug_log.append(f"Recognition error: {e}")
                        text = ""
                        wavfile = ""
                    if text.strip():
                        dictation_text += ("\n" if dictation_text else "") + text.strip()
                        save_text(text, wavfile)
                        debug_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Saved: {wavfile.replace('.wav','.txt')}")
                    else:
                        debug_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] No text recognized.")
                    prompt_message = "[cyan]Готово! Говорите следующий фрагмент или Enter — стоп, Ctrl+C — выход.[/cyan]"
                    live.update(make_layout(dictation_text, debug_log, prompt_message))
                    time.sleep(1.2)
                    debug_log.append("---- Next fragment ----")
                # После выхода из Live (по break) обновим панели с новой подсказкой
                prompt_message = "[magenta]Говорите следующий фрагмент или Enter — стоп, Ctrl+C — выход.[/magenta]"
                with Live(make_layout(dictation_text, debug_log, prompt_message), refresh_per_second=12, screen=True) as live:
                    time.sleep(1)
                # Прервать, если пользователь нажал Enter
                if sys.stdin in select([sys.stdin], [], [], 0)[0]:
                    _ = sys.stdin.readline()
                    debug_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Enter — stop session.")
                    break
        except KeyboardInterrupt:
            debug_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] KeyboardInterrupt (Exit).")
            prompt_message = "[red]Завершено пользователем.[/red]"
            with Live(make_layout(dictation_text, debug_log, prompt_message), refresh_per_second=12, screen=True):
                time.sleep(2)

def main():
    global dictation_text, debug_log, prompt_message
    dictation_text = ""
    debug_log = []
    prompt_message = "[bold green]Готово к диктовке. Говорите фрагментами по 10 секунд. Enter — стоп, Ctrl+C — выход.[/bold green]"
    console.clear()
    with Live(make_layout(dictation_text, debug_log, prompt_message), refresh_per_second=12, screen=True):
        time.sleep(2)
    record_and_recognize_session()
    prompt_message = "[bold blue]Диктовка завершена. Смотрите результат ниже![/bold blue]"
    with Live(make_layout(dictation_text, debug_log, prompt_message), refresh_per_second=12, screen=True):
        time.sleep(5)
    console.print(Panel(Text(dictation_text, style="bold white on blue"), title="📝 Итоговая диктовка"))

if __name__ == "__main__":
    dictation_text = ""
    debug_log = []
    prompt_message = ""
    main()
