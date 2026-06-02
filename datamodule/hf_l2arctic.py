"""
Build an HF dataset for L2-ARCTIC in the project's standard SLAM-ASR schema.

Source (already on disk, fetched by data_external/download_l2arctic.py):
    data/l2arctic/   (HF DatasetDict from KoelLabs/L2Arctic — gated, CC-BY-NC)

Source schema (per the verifier's output):
    audio, ipa, text, g2p, speaker_code, speaker_gender, speaker_native_language
    - audio: WAV bytes at 16 kHz (already correct sample rate, no resample)
    - speaker_gender: 'm' / 'f'
    - speaker_native_language: one of Arabic/Hindi/Korean/Mandarin/Spanish/Vietnamese
    - speaker_code: 3-letter speaker ID (24 speakers total)

Output schema (data/l2arctic_hf/):
    audio_array         List[float32]
    sampling_rate       int (16000)
    raw_transcription   str  (= text, unchanged)
    transcription       str  (lowercased + minimal punctuation strip)
    duration            float (seconds)
    speaker_id          str  (= speaker_code)
    gender              str  ('male' / 'female')
    l1                  str  ('arabic' / ... lowercase)
    native_english      str  ('no' — all L2-ARCTIC speakers are non-native by construction)
    age                 str  ('missing')
    accent              str  (= l1, since L2-ARCTIC accents map 1:1 to L1)

Usage:
    python -m datamodule.hf_l2arctic
    python -m datamodule.hf_l2arctic --input_dir data/l2arctic --output_dir data/l2arctic_hf

By default writes the 'scripted' split (3,599 samples) as 'test' since
it's the larger, more controlled subset. Pass --include_spontaneous to
also include the 22 spontaneous samples.
"""

from __future__ import annotations

import argparse
import io
import re
from pathlib import Path

import numpy as np


def _norm_text(s: str) -> str:
    """Lowercase + collapse whitespace + strip non-word non-space (keeping apostrophes)."""
    s = (s or "").lower()
    s = re.sub(r"[^\w\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _norm_gender(g: str) -> str:
    g = (g or "").strip().lower()
    if g.startswith("m"):
        return "male"
    if g.startswith("f"):
        return "female"
    return "missing"


def _norm_l1(l1: str) -> str:
    return (l1 or "").strip().lower() or "missing"


def _decode_audio(audio_field) -> tuple[np.ndarray, int]:
    """L2-ARCTIC audio is stored either as a dict({path, bytes, array, sampling_rate})
    or as raw bytes. Decode to a float32 np.ndarray at 16 kHz.

    We use soundfile so torchcodec isn't required (matches the rest of the
    project's audio path). Resample to 16k if the source isn't already.
    """
    import soundfile as sf
    if isinstance(audio_field, dict):
        if audio_field.get("array") is not None:
            arr = np.asarray(audio_field["array"], dtype=np.float32)
            sr = int(audio_field.get("sampling_rate") or 16000)
            return arr, sr
        if audio_field.get("bytes") is not None:
            arr, sr = sf.read(io.BytesIO(audio_field["bytes"]), dtype="float32")
            return arr, sr
        if audio_field.get("path"):
            arr, sr = sf.read(audio_field["path"], dtype="float32")
            return arr, sr
    if isinstance(audio_field, (bytes, bytearray)):
        arr, sr = sf.read(io.BytesIO(audio_field), dtype="float32")
        return arr, sr
    raise TypeError(f"Cannot decode audio field of type {type(audio_field)}")


def _resample_if_needed(arr: np.ndarray, sr: int, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    if sr == target_sr:
        return arr, sr
    import librosa
    arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    return arr, target_sr


def _build_split(d, features, datasets_pkg, batch_size: int) -> tuple[object, int, float]:
    """Process one split into a HF Dataset by accumulating rows in batches.

    Returns (Dataset, n_rows, total_duration_s).
    """
    from datasets import concatenate_datasets
    batches = []
    buf: list[dict] = []
    n_rows = 0
    total_dur = 0.0
    import gc

    def _drain():
        nonlocal buf
        if buf:
            batches.append(datasets_pkg.Dataset.from_list(buf, features=features))
            buf = []
            gc.collect()

    for row in d:
        try:
            arr, sr = _decode_audio(row["audio"])
        except Exception as e:
            print(f"  WARN: skipping row (audio decode failed): {e}")
            continue
        arr, sr = _resample_if_needed(arr, sr, target_sr=16000)
        duration = float(len(arr) / sr)
        raw_text = (row.get("text") or "").strip()
        if not raw_text:
            continue
        l1 = _norm_l1(row.get("speaker_native_language"))
        buf.append({
            "audio_array": arr.tolist(),
            "sampling_rate": sr,
            "raw_transcription": raw_text,
            "transcription": _norm_text(raw_text),
            "duration": duration,
            "speaker_id": row.get("speaker_code") or "",
            "gender": _norm_gender(row.get("speaker_gender")),
            "l1": l1,
            "native_english": "no",
            "age": "missing",
            "accent": l1,
        })
        n_rows += 1
        total_dur += duration
        if len(buf) >= batch_size:
            _drain()

    _drain()
    if not batches:
        return None, 0, 0.0
    return concatenate_datasets(batches), n_rows, total_dur


def build(input_dir: Path, output_dir: Path, include_spontaneous: bool = False,
          batch_size: int = 1000) -> None:
    import datasets as ds_pkg
    from datasets import (DatasetDict, Features, Sequence, Value,
                           load_from_disk, Audio)

    print(f"[L2-ARCTIC] Loading source HF dataset from {input_dir}")
    src = load_from_disk(str(input_dir))

    # Cast Audio columns to decode=False so we control decoding via soundfile.
    for split in src:
        for col, feat in list(src[split].features.items()):
            if isinstance(feat, Audio):
                src[split] = src[split].cast_column(col, Audio(decode=False))

    splits_to_build = ["scripted"]
    if include_spontaneous and "spontaneous" in src:
        splits_to_build.append("spontaneous")

    features = Features({
        "audio_array": Sequence(Value("float32")),
        "sampling_rate": Value("int32"),
        "raw_transcription": Value("string"),
        "transcription": Value("string"),
        "duration": Value("float32"),
        "speaker_id": Value("string"),
        "gender": Value("string"),
        "l1": Value("string"),
        "native_english": Value("string"),
        "age": Value("string"),
        "accent": Value("string"),
    })

    out_splits: dict = {}
    total_dur = 0.0
    for split_name in splits_to_build:
        d = src[split_name]
        print(f"[L2-ARCTIC] Building '{split_name}' from {len(d)} source rows ...")
        ds, n, dur = _build_split(d, features, ds_pkg, batch_size=batch_size)
        if ds is None:
            print(f"  WARN: '{split_name}' produced 0 rows after filtering.")
            continue
        out_splits[split_name] = ds
        total_dur += dur
        print(f"  '{split_name}': {n} valid rows.")

    if not out_splits:
        raise SystemExit("No rows survived the build; aborting.")

    out_dict = DatasetDict()
    out_dict["test"] = out_splits["scripted"]
    if "spontaneous" in out_splits:
        out_dict["spontaneous"] = out_splits["spontaneous"]

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[L2-ARCTIC] Saving to {output_dir}")
    out_dict.save_to_disk(str(output_dir))
    print(f"[L2-ARCTIC] Done. test split: {len(out_dict['test'])} rows, "
          f"total audio ~{total_dur/3600.0:.2f} h.")


def parse_args():
    p = argparse.ArgumentParser(description="Build an HF dataset for L2-ARCTIC in SLAM-ASR schema.")
    p.add_argument("--input_dir", type=Path, default=Path("data/l2arctic"),
                   help="HF dataset on disk (output of data_external/download_l2arctic.py).")
    p.add_argument("--output_dir", type=Path, default=Path("data/l2arctic_hf"),
                   help="Where to save the SLAM-ASR-shaped HF dataset.")
    p.add_argument("--include_spontaneous", action="store_true",
                   help="Also save the 22 'spontaneous' rows as a separate split.")
    p.add_argument("--batch_size", type=int, default=1000,
                   help="Rows per HF Dataset.from_list batch.")
    return p.parse_args()


def main():
    args = parse_args()
    build(args.input_dir, args.output_dir, include_spontaneous=args.include_spontaneous,
          batch_size=args.batch_size)


if __name__ == "__main__":
    main()
