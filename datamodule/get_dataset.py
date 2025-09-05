import os
import json
import torchaudio

librispeech_test = torchaudio.datasets.LIBRISPEECH(
    root="data",
    url="test-clean",
    download=True
)

out_path = os.path.join("data", "test.jsonl")
base = os.path.join("data", "LibriSpeech", "test-clean")

with open(out_path, "w", encoding="utf-8") as f:
    for idx, sample in enumerate(librispeech_test):
        waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = sample
        # Construct the correct path
        rel_path = librispeech_test._walker[idx]  # e.g., 121/121726/121-121726-0000.flac
        abs_path = os.path.join(base, rel_path)   # data/LibriSpeech/test-clean/121/121726/121-121726-0000.flac
        entry = {
            "source": abs_path,
            "target": transcript,
            "key": f"{speaker_id}-{chapter_id}-{utterance_id}",
        }
        f.write(json.dumps(entry) + "\n")

print(f"Wrote {len(librispeech_test)} examples to {out_path}")
