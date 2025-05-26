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

def compute_vu(pcm):
    rms = np.sqrt(np.mean(np.square(pcm.astype(np.float32))))
    db = 20 * np.log10(rms + 1e-8)
    return np.clip((db + 50) / 50, 0, 1)

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

# --- RICH LAYOUT ---
def make_layout():
    layout = Layout()
    layout.split_column(
        Layout(name="dictation", ratio=3),
        Layout(name="debug", ratio=1),
    )
    layout["dictation"].update(Panel(Text(dictation_text, style="bold white on blue"), title="📝 Dictation"))
    layout["debug"].update(Panel(Text("\n".join(debug_log[-12:]), style="yellow"), title="⚙️ Debug Log"))
    return layout

def record_and_recognize_session():
    global dictation_text, debug_log
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE, callback=audio_callback, blocksize=int(SAMPLE_RATE * 0.5)):
        try:
            while True:
                frames = []
                vad_timeout = None
                silence_sec = 0
                t0 = time.time()
                debug_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] New fragment started...")
                while True:
                    now = time.time()
                    elapsed = now - t0
                    if elapsed >= CHUNK_SEC:
                        break
                    try:
                        chunk = q.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    frames.append(chunk)
                    pcm = chunk.flatten()
                    vu = compute_vu(pcm)
                    bar = "█" * int(vu * 20) + "-" * (20 - int(vu * 20))
                    speech = vad_activity(chunk.tobytes(), SAMPLE_RATE)
                    if speech:
                        silence_sec = 0
                        vad_timeout = time.time()
                    else:
                        if vad_timeout:
                            silence_sec = time.time() - vad_timeout
                    debug_log.append(
                        f"[{int(elapsed):2d}s] VU:{vu:.2f} {'🎤' if speech else '  '} | Silence: {silence_sec:.1f}s {bar}"
                    )
                    if silence_sec >= SILENCE_TIMEOUT:
                        debug_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Silence {SILENCE_TIMEOUT}s: fragment end.")
                        break
                audio = np.concatenate(frames, axis=0)
                debug_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Fragment recorded. Sending to Whisper...")
                with Live(make_layout(), refresh_per_second=3, screen=False):
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
                with Live(make_layout(), refresh_per_second=3, screen=False):
                    time.sleep(1)
                debug_log.append("---- Next fragment ----")
                print(">>> Speak next fragment or press Enter to stop...\n")
                # Прервать, если пользователь нажал Enter
                if sys.stdin in select([sys.stdin], [], [], 0)[0]:
                    _ = sys.stdin.readline()
                    debug_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Enter — stop session.")
                    break
        except KeyboardInterrupt:
            debug_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] KeyboardInterrupt (Exit).")

def main():
    global dictation_text, debug_log
    dictation_text = ""
    debug_log = []
    console.clear()
    console.print("[bold green]Готово к диктовке. Говорите фрагментами по 10 секунд. Enter — стоп, Ctrl+C — выход.[/bold green]\n")
    record_and_recognize_session()
    console.print(Panel(Text(dictation_text, style="bold white on blue"), title="📝 Итоговая диктовка"))

if __name__ == "__main__":
    dictation_text = ""
    debug_log = []
    main()
