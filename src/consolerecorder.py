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

# === CONFIG ===
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = 'int16'
CHUNK_SEC = 0.5
RECORD_TIMEOUT = 10   # max sec
SILENCE_TIMEOUT = 3   # auto-stop after 3 sec silence
VAD_LEVEL = 2

# === INIT ===
vad = webrtcvad.Vad(VAD_LEVEL)
q = queue.Queue()
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# === LOGGING ===
def log_event(event):
    ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f"[{ts}] {event}")

# === AUDIO CALLBACK ===
def audio_callback(indata, frames, time_info, status):
    if status:
        log_event(f"Status: {status}")
    q.put(indata.copy())

# === VU METER ===
def compute_vu(pcm):
    rms = np.sqrt(np.mean(np.square(pcm.astype(np.float32))))
    db = 20 * np.log10(rms + 1e-8)
    return np.clip((db + 50) / 50, 0, 1)

# === CORRECT VAD HANDLING ===
def vad_activity(chunk_bytes, sample_rate):
    # 30 ms = 480 samples = 960 bytes for 16kHz, int16
    frame_size = int(0.03 * sample_rate) * 2
    speech_detected = False
    for i in range(0, len(chunk_bytes), frame_size):
        frame = chunk_bytes[i:i+frame_size]
        if len(frame) == frame_size:
            if vad.is_speech(frame, sample_rate):
                speech_detected = True
                break
    return speech_detected

# === AUDIO RECORDER ===
def record_audio():
    frames = []
    vad_timeout = None
    total_sec = 0
    silence_sec = 0
    max_time = RECORD_TIMEOUT

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE, callback=audio_callback, blocksize=int(CHUNK_SEC*SAMPLE_RATE)):
        log_event("Запись начата. Enter — остановить, Ctrl+C — выйти.")
        print_progress(0, max_time, 0, 0, True)
        t0 = time.time()
        while True:
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
            total_sec = time.time() - t0
            print_progress(total_sec, max_time, vu, speech, False, bar=bar, silence_sec=silence_sec)
            if silence_sec >= SILENCE_TIMEOUT:
                log_event(f"Тишина {SILENCE_TIMEOUT} сек, автостоп.")
                break
            if total_sec >= max_time:
                log_event("Достигнут лимит времени, автостоп.")
                break
            # Stop immediately if Enter is pressed
            if sys.stdin in select([sys.stdin], [], [], 0)[0]:
                _ = sys.stdin.readline()
                log_event("Enter — остановка записи.")
                break
    log_event("Запись остановлена.")
    audio = np.concatenate(frames, axis=0)
    return audio

def print_progress(current, total, vu, speech, new, bar="", silence_sec=0):
    secs = int(current)
    sil = f" | Тишина: {silence_sec:.1f}s" if silence_sec else ""
    speech_ind = "🎤" if speech else "  "
    msg = f"\r[{secs:2d}/{int(total)}s] [{bar}] VU:{vu:.2f} {speech_ind}{sil}    "
    if new:
        print()
    sys.stdout.write(msg)
    sys.stdout.flush()

# === OPENAI WHISPER RECOGNITION ===
def recognize_audio(audio_array, sample_rate=16000):
    import tempfile
    import wave
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
    return text

# === MAIN LOOP ===
def main():
    log_event("Готово к записи. Нажмите Enter для старта, Ctrl+C для выхода.")
    try:
        while True:
            input(">>> Enter — начать запись")
            audio = record_audio()
            print("\nАудио записано. Отправляем в OpenAI...")
            try:
                text = recognize_audio(audio)
                print("\n===== Распознанный текст =====\n")
                print(text.strip())
                print("\n==============================\n")
            except Exception as e:
                print("Ошибка распознавания:", e)
    except KeyboardInterrupt:
        print("\nВыход.")

if __name__ == "__main__":
    main()
