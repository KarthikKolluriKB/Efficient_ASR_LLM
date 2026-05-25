"""
Download the EdAcc (Edinburgh International Accents of English) dataset.

Source:   https://huggingface.co/datasets/edinburghcstr/edacc
License:  CC-BY-SA
Size:     ~6.95 GB on disk, ~40 h of conversational English
Splits:   validation (9,850), test (9,290)
Gating:   None — direct `load_dataset()` works without credentials

Fields:
    speaker, text, accent, raw_accent, gender, l1, audio (32 kHz array)

Usage:
    python experiments/bias_pruning/data_external/download_edacc.py
    python experiments/bias_pruning/data_external/download_edacc.py --output_dir /path/to/edacc

The dataset will be downloaded to the HF cache (set HF_HOME or
HF_DATASETS_CACHE to control where) and then saved to disk in HF format
under {output_dir} so the project's existing dataset loaders can use it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Download the EdAcc dataset to local HF format.")
    p.add_argument("--output_dir", type=Path,
                   default=Path(__file__).resolve().parents[3] / "data" / "edacc",
                   help="Directory to save the dataset (HF save_to_disk format).")
    p.add_argument("--splits", nargs="+", default=["validation", "test"],
                   help="Splits to save. Default: validation + test.")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install `datasets`: pip install datasets")

    print(f"[EdAcc] Loading from edinburghcstr/edacc ...")
    ds = load_dataset("edinburghcstr/edacc")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[EdAcc] Saving to: {args.output_dir}")
    # Save only requested splits to keep the on-disk layout consistent
    # with the project's other datasets.
    keep = {s: ds[s] for s in args.splits if s in ds}
    if not keep:
        raise SystemExit(f"None of the requested splits {args.splits} found. Available: {list(ds.keys())}")
    from datasets import DatasetDict
    DatasetDict(keep).save_to_disk(str(args.output_dir))

    for s, d in keep.items():
        print(f"[EdAcc] {s}: {len(d)} samples")
    print("[EdAcc] Done.")


if __name__ == "__main__":
    main()
