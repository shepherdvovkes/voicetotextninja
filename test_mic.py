import sounddevice as sd
import numpy as np

duration = 3  # секунды записи
fs = 16000    # sample rate

print("Говорите в микрофон...")
rec = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
sd.wait()
print("Запись завершена. Максимальное значение:", np.max(np.abs(rec)))
