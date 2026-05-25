"""
Stage the Meta ASR Fairness Evaluation Dataset (paper name: Fair-Speech).

Source (manual, gated):
    Meta releases the dataset via an "Accept terms" download page:
        asr_fairness_audio.zip       (~4.1 GB, 30,000 utts across 602 speakers)
        asr_fairness_metadata.tsv    (demographics: age, gender, ethnicity,
                                       geographic location, native-English flag)
    Paper: https://arxiv.org/abs/2408.12734  (Veliche et al., 2024)

There is no scriptable download (the page is JS-gated, click-to-accept).
After your browser finishes the download, point this script at the
local files and it will set up `data/fairspeech/` correctly.

License (READ BEFORE PROCEEDING):
    Meta's license FORBIDS redistribution of any part of the dataset,
    including reference transcripts. Concretely:
      * Do NOT commit `asr_fairness_audio.zip`, the metadata TSV, or any
        derived per-utterance CSV that contains the reference text.
      * `data/` is already gitignored, so audio is safe by default.
      * Per-utterance CSVs that include this dataset's text are blocked
        in .gitignore (pattern: `*fairspeech*`, `*fair_speech*`).
      * Summary-level results (per-gender WER, p-values, etc.) ARE
        derivatives we own and may publish (per the license's IP clause).
      * License also forbids using the dataset to train classifiers
        that predict race / ethnicity / gender of individuals. The use
        we are making (evaluating ASR model fairness) is the stated
        Purpose and explicitly allowed.

Usage:
    # After the browser download finishes:
    python experiments/bias_pruning/data_external/download_fairspeech.py \
        --zip_path ~/Downloads/asr_fairness_audio.zip \
        --metadata_path ~/Downloads/asr_fairness_metadata.tsv

    # Skip extraction (faster, you'll extract manually):
    python experiments/bias_pruning/data_external/download_fairspeech.py \
        --zip_path ... --metadata_path ... --no_extract
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

LICENSE_REMINDER = """
[Fair-Speech] LICENSE REMINDER:
    - Do NOT commit the audio zip, the metadata TSV, or any per-utterance
      CSV containing this dataset's reference text. `data/` is gitignored;
      `*fairspeech*` per-utterance CSVs are also gitignored.
    - Allowed use: evaluating ASR/LLM model fairness (this experiment).
    - Forbidden: redistribution, training demographic classifiers,
      re-identifying individuals.
""".strip()


def parse_args():
    p = argparse.ArgumentParser(description="Stage the manually-downloaded Meta ASR Fairness Dataset.")
    p.add_argument("--zip_path", type=Path, required=True,
                   help="Local path to asr_fairness_audio.zip (~4.1 GB).")
    p.add_argument("--metadata_path", type=Path, required=True,
                   help="Local path to asr_fairness_metadata.tsv.")
    p.add_argument("--output_dir", type=Path,
                   default=Path(__file__).resolve().parents[3] / "data" / "fairspeech",
                   help="Where to stage the data. Defaults to data/fairspeech/ (gitignored).")
    p.add_argument("--no_extract", action="store_true",
                   help="Copy/move the zip but skip extraction.")
    p.add_argument("--move", action="store_true",
                   help="Move the downloaded files instead of copying (saves disk).")
    return p.parse_args()


def main():
    args = parse_args()
    print(LICENSE_REMINDER)
    print()

    if not args.zip_path.exists():
        raise SystemExit(f"[Fair-Speech] Zip not found at {args.zip_path}")
    if not args.metadata_path.exists():
        raise SystemExit(f"[Fair-Speech] Metadata TSV not found at {args.metadata_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    # Stage the metadata file
    meta_target = args.output_dir / "metadata.tsv"
    if meta_target.exists():
        print(f"[Fair-Speech] Metadata already at {meta_target}; overwriting.")
        meta_target.unlink()
    op = shutil.move if args.move else shutil.copy2
    print(f"[Fair-Speech] {'Moving' if args.move else 'Copying'} metadata -> {meta_target}")
    op(str(args.metadata_path), str(meta_target))

    # Stage the audio zip
    zip_target = args.output_dir / args.zip_path.name
    if zip_target.exists():
        print(f"[Fair-Speech] Audio zip already at {zip_target}; using in place.")
    else:
        print(f"[Fair-Speech] {'Moving' if args.move else 'Copying'} audio zip -> {zip_target}")
        op(str(args.zip_path), str(zip_target))

    if args.no_extract:
        print(f"[Fair-Speech] Skipping extraction. Zip is at {zip_target}.")
        return

    print(f"[Fair-Speech] Extracting {zip_target.name} -> {audio_dir}")
    with zipfile.ZipFile(zip_target, "r") as zf:
        members = zf.namelist()
        print(f"[Fair-Speech] Archive contains {len(members)} entries.")
        zf.extractall(audio_dir)
    print(f"[Fair-Speech] Audio extracted to {audio_dir}")

    # Quick sanity check
    n_audio_files = sum(1 for _ in audio_dir.rglob("*"))
    n_meta_rows = sum(1 for _ in open(meta_target, "r", encoding="utf-8")) - 1
    print(f"[Fair-Speech] {n_audio_files} extracted entries, {n_meta_rows} metadata rows.")
    print(f"[Fair-Speech] Done. Next: write the HF-format builder for Fair-Speech.")


if __name__ == "__main__":
    main()
