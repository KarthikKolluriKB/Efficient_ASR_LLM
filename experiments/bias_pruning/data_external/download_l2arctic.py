"""
Download the L2-ARCTIC dataset (non-native English speakers, 6 L1 backgrounds).

Source:   https://huggingface.co/datasets/KoelLabs/L2Arctic   (HF mirror)
          https://psi.engr.tamu.edu/l2-arctic-corpus/         (original, form-gated)
License:  CC-BY-NC-4.0
Size:     ~473 MB
Splits:   scripted (3,599 samples), spontaneous (22 samples), ~246 min total
L1s:      Hindi, Korean, Mandarin, Spanish, Arabic, Vietnamese (24 speakers)
Gating:   Requires HF account, TOS acceptance, and `huggingface-cli login`

Prerequisites (one-time):
    1. Visit https://huggingface.co/datasets/KoelLabs/L2Arctic, accept terms.
    2. Create a HF token with "Read access to public gated repos" enabled
       at https://huggingface.co/settings/tokens
    3. On the server, run: `huggingface-cli login` and paste the token.

Usage:
    python experiments/bias_pruning/data_external/download_l2arctic.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Download L2-ARCTIC via the HF mirror.")
    p.add_argument("--output_dir", type=Path,
                   default=Path(__file__).resolve().parents[3] / "data" / "l2arctic",
                   help="Directory to save the dataset (HF save_to_disk format).")
    p.add_argument("--splits", nargs="+", default=["scripted", "spontaneous"],
                   help="Splits to save.")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        from datasets import load_dataset, DatasetDict
    except ImportError:
        raise SystemExit("Install `datasets`: pip install datasets")

    print("[L2-ARCTIC] Loading from KoelLabs/L2Arctic ...")
    print("[L2-ARCTIC] If this fails with 401/403, accept the dataset terms on HF first")
    print("[L2-ARCTIC] and run `huggingface-cli login` with a token that has gated-repo read.")

    try:
        ds = load_dataset("KoelLabs/L2Arctic")
    except Exception as e:
        msg = str(e)
        if "401" in msg or "403" in msg or "gated" in msg.lower():
            print("\n[L2-ARCTIC] ACCESS DENIED. Do this on the server:")
            print("  1. Browse to https://huggingface.co/datasets/KoelLabs/L2Arctic and click 'Agree'")
            print("  2. huggingface-cli login   (paste a token with gated-repo read)")
            print("  3. Re-run this script\n")
        raise

    args.output_dir.mkdir(parents=True, exist_ok=True)
    keep = {s: ds[s] for s in args.splits if s in ds}
    if not keep:
        raise SystemExit(f"None of {args.splits} found. Available: {list(ds.keys())}")
    DatasetDict(keep).save_to_disk(str(args.output_dir))

    for s, d in keep.items():
        print(f"[L2-ARCTIC] {s}: {len(d)} samples")
    print(f"[L2-ARCTIC] Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
