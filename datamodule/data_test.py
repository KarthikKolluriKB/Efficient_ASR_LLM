import json
import whisper

data_path = "data/test.jsonl"

with open(data_path, "r", encoding="utf-8") as f:
    first_line = f.readline()
    file_info = json.loads(first_line)
    file_path = file_info["source"]

audio_raw = whisper.load_audio(file_path)

print(f"Loaded audio from {file_path}, length: {len(audio_raw)} samples")
