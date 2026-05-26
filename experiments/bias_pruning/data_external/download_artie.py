"""
Download the Artie Bias Corpus *metadata*. Audio is NOT recoverable from
the original Artie distribution any longer.

Source:   https://github.com/artie-inc/artie-bias-corpus  (TSV + bias scripts)
License:  CC-0
Size:     1,712 clips, ~2.4 h, 17 distinct English accents

STATUS NOTE (May 2026):
    The DATASHEET in the repo still advertises a tarball at
    http://ml-corpora.artie.com/artie-bias-corpus.tar.gz  — that
    host is DEAD (ECONNREFUSED, DNS no longer resolves). Artie Inc.
    was acquired by Match Group and the corpus CDN was retired with
    no announced successor. No HuggingFace / Zenodo / archive.org
    mirror of the audio surfaced via search as of May 2026.

What this script does:
    1. Clones the GitHub repo to obtain artie-bias-corpus.tsv
       (columns: client_id, path, sentence, up_votes, down_votes,
        age, gender, accent — same schema as CommonVoice).
    2. Prints what you need to do to source the matching audio
       (~1,712 MP3 clips from Mozilla CommonVoice English June 2019).

Practical recommendation:
    Artie has become the hardest of the five external datasets to
    operationalise. If you only have time for some of them, prefer
    EdAcc and Fair-Speech for the bias-pruning experiment. Artie's
    value (~2.4 h, balanced cells) is real but the audio sourcing
    cost has risen substantially since 2024.

Usage:
    python experiments/bias_pruning/data_external/download_artie.py
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ARTIE_REPO = "https://github.com/artie-inc/artie-bias-corpus.git"


def parse_args():
    p = argparse.ArgumentParser(description="Clone Artie Bias Corpus metadata repo (audio is manual).")
    p.add_argument("--output_dir", type=Path,
                   default=Path(__file__).resolve().parents[3] / "data" / "artie_bias",
                   help="Where to clone the repo and place audio later.")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = args.output_dir / "artie-bias-corpus"

    if repo_dir.exists():
        print(f"[Artie] Repo already cloned at {repo_dir}; pulling.")
        subprocess.run(["git", "-C", str(repo_dir), "pull"], check=True)
    else:
        print(f"[Artie] Cloning {ARTIE_REPO} -> {repo_dir}")
        subprocess.run(["git", "clone", "--depth", "1", ARTIE_REPO, str(repo_dir)], check=True)

    tsv_candidates = sorted(repo_dir.rglob("artie-bias-corpus*.tsv"))
    if tsv_candidates:
        print(f"[Artie] Metadata TSV:")
        for t in tsv_candidates:
            print(f"   {t}  ({t.stat().st_size/1024:.1f} KB)")
    else:
        print("[Artie] WARNING: no artie-bias-corpus*.tsv found in the cloned repo.")
        return

    audio_dir = args.output_dir / "clips"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"""
[Artie] METADATA STAGED. AUDIO IS MANUAL.

The original Artie audio CDN (http://ml-corpora.artie.com/) has been
retired with no public successor. To complete the corpus, you need
to provide the 1,712 MP3 clips referenced by the TSV's `path`
column under:

    {audio_dir}/<path>

These clips come from Mozilla CommonVoice English, June 2019 release.
Three ways to get them:

    1. Mozilla Data Collective (new path since Oct 2025):
       https://commonvoice.mozilla.org/en/datasets
       Sign up, accept terms, and look for a historical release
       containing the referenced filenames.

    2. Local mirror: if you, your lab, or a collaborator has a saved
       copy of CV en from 2019-2020, the matching MP3s should be in
       its cv-valid-test or validated set. Symlink them into
       {audio_dir}/.

    3. Skip Artie. Of the five external datasets in data_external/,
       Artie has become the hardest to operationalise. EdAcc and
       Fair-Speech provide comparable demographic stratification
       with less effort. Artie's small size (~2.4 h) means it would
       only ever be a confirmation, not a primary result.

Once the clips are in place, build the HF-format dataset with the
generic adapter (TBD).
""")


if __name__ == "__main__":
    main()
