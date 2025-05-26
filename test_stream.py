from src.capture import record_stream

for i, chunk in enumerate(record_stream()):
    print(f"Chunk {i}, bytes: {len(chunk)}")
    if i > 2:
        break
 