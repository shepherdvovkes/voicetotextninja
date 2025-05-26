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

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = 'int16'
CHUNK_SEC = 10      # длина одного куска (10 секунд)
VAD_LEVEL = 2
SILENCE_TIMEOUT = 3 # автостоп куска после 3 сек тишины

vad = webrtcvad.Vad(VAD_LEVEL)
q = queue.Queue()
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def log_event(event):
    ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f"[{ts}] {event}")

def audio_callback(indata, frames, time_info, status):
    if status:
        log_event(f"Status: {status}")
    q.put(indata.copy())

def compute_vu(pcm):
    rms = np.sqrt(np.mean(np.square(pcm.astype(np.float32))))
    db = 20 * np.log10(rms + 1e-8)
    return np.clip((db + 50) / 50, 0, 1)

def vad_activity(chunk_bytes, sample_rate):
    frame_size = int(0.03 * sample_rate) * 2
    speech_detected = False
    for i in range(0, len(chunk_bytes), frame_size):
        frame = chunk_bytes[i:i+frame_size]
        if len(frame) == frame_size:
            if vad.is_speech(frame, sample_rate):
                speech_detected = True
                break
    return speech_detected

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
                # language="auto"  # можно явно указать
            )
    return text, tf.name

def print_progress(current, vu, speech, new, bar="", silence_sec=0):
    secs = int(current)
    sil = f" | Тишина: {silence_sec:.1f}s" if silence_sec else ""
    speech_ind = "🎤" if speech else "  "
    msg = f"\r[{secs:2d}s] [{bar}] VU:{vu:.2f} {speech_ind}{sil}    "
    if new:
        print()
    sys.stdout.write(msg)
    sys.stdout.flush()

def save_text(text, basename):
    fname = basename.replace('.wav', '.txt')
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text.strip())
    return fname

def split_sentences(text):
    # Простое разбиение на предложения (по точкам, воскл., вопрос.)
    import re
    return re.split(r'(?<=[.?!])\s+', text.strip())

def record_and_recognize_session():
    log_event("Готово к диктовке. Enter — стоп, Ctrl+C — выход.")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE, callback=audio_callback, blocksize=int(SAMPLE_RATE * 0.5)):
        fragments = []
        try:
            while True:
                frames = []
                vad_timeout = None
                silence_sec = 0
                t0 = time.time()
                log_event("Запись фрагмента начата...")
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
                    print_progress(elapsed, vu, speech, False, bar=bar, silence_sec=silence_sec)
                    if silence_sec >= SILENCE_TIMEOUT:
                        log_event(f"Тишина {SILENCE_TIMEOUT} сек, фрагмент завершён.")
                        break
                audio = np.concatenate(frames, axis=0)
                log_event("Фрагмент записан. Отправляем в Whisper...")
                text, wavfile = recognize_audio(audio)
                log_event("Готово. Сохраняем результаты...")
                txtfile = save_text(text, wavfile)
                log_event(f"Сохранено: {wavfile}, {txtfile}")
                print("\n===== Распознанный текст =====\n")
                for sent in split_sentences(text):
                    print(sent.strip())
                print("\n==============================\n")
                fragments.append((wavfile, txtfile, text))
                print(">>> Следующий фрагмент — продолжайте говорить или жмите Enter для остановки...\n")
                # Прервать, если пользователь нажал Enter
                if sys.stdin in select([sys.stdin], [], [], 0)[0]:
                    _ = sys.stdin.readline()
                    log_event("Enter — остановка диктовки.")
                    break
        except KeyboardInterrupt:
            print("\nВыход.")
        return fragments

def main():
    record_and_recognize_session()

if __name__ == "__main__":
    main()
