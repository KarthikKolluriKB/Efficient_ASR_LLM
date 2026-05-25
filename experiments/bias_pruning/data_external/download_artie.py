"""
Download the Artie Bias Corpus.

Source:   https://github.com/artie-inc/artie-bias-corpus  (TSV metadata only)
          The audio files come from Mozilla Common Voice June 2019 (en).
License:  CC-0 for transcripts; CV audio is also CC-0 in older releases.
Size:     1,712 clips, ~2.4 h of English audio across 17 accents
Fields:   demographic tags for age, gender, accent (validated)

Two-step process:
    1. Clone the artie-inc/artie-bias-corpus GitHub repo. This gives you
       `artie-bias-corpus.tsv` which lists the 1,712 clip paths and the
       demographic labels.
    2. Obtain the matching MP3 clips. This requires the **June 2019** CV
       English release. Mozilla switched to gated access in October 2025
       (Mozilla Data Collective), so the easiest path now is to use any
       still-mirrored CV release that contains the older `cv-valid-test`
       clip set, or to download the older HF-hosted CV release if
       available. See the printed instructions at the end of this script.

Usage:
    python experiments/bias_pruning/data_external/download_artie.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ARTIE_REPO = "https://github.com/artie-inc/artie-bias-corpus.git"


def parse_args():
    p = argparse.ArgumentParser(description="Download Artie Bias Corpus metadata (step 1 of 2).")
    p.add_argument("--output_dir", type=Path,
                   default=Path(__file__).resolve().parents[3] / "data" / "artie_bias",
                   help="Where to clone the metadata repo and place audio.")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = args.output_dir / "artie-bias-corpus"

    if repo_dir.exists():
        print(f"[Artie] Repo already cloned at {repo_dir}; pulling latest.")
        subprocess.run(["git", "-C", str(repo_dir), "pull"], check=True)
    else:
        print(f"[Artie] Cloning {ARTIE_REPO} into {repo_dir}")
        subprocess.run(["git", "clone", "--depth", "1", ARTIE_REPO, str(repo_dir)], check=True)

    tsv_candidates = list(repo_dir.rglob("artie-bias-corpus*.tsv"))
    if tsv_candidates:
        print(f"[Artie] Metadata TSV(s) found:")
        for t in tsv_candidates:
            print(f"   {t}  ({t.stat().st_size/1024:.1f} KB)")
    else:
        print("[Artie] WARNING: no artie-bias-corpus*.tsv found in the cloned repo.")

    audio_dir = args.output_dir / "clips"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"""
[Artie] STEP 2 IS MANUAL. The corpus references clip paths from Mozilla
        Common Voice (June 2019, en). Put the matching MP3 files under:
            {audio_dir}/
        such that each entry in the TSV's `path` column resolves to:
            {audio_dir}/<path>

Two ways to get those clips:
  (a) If you still have a local copy of CV June 2019 en (cv-valid-test
      or similar), symlink/copy the relevant MP3s into {audio_dir}.
  (b) Otherwise, look for the older HF mirror (e.g. some legacy CV
      release on HuggingFace Hub) and extract the matching `path`
      values listed in the TSV.

After audio is in place, build the HF-format dataset by running
build_hf_artie.py (TBD).
""")


if __name__ == "__main__":
    main()
