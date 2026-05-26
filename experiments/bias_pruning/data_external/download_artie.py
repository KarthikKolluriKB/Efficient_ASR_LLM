"""
Download the Artie Bias Corpus (audio + metadata in a single tarball).

Sources (in order of preference):
    1. Direct tarball (audio + TSV in one file, fastest):
       http://ml-corpora.artie.com/artie-bias-corpus.tar.gz
    2. GitHub repo for the TSV + scripts only:
       https://github.com/artie-inc/artie-bias-corpus
       (still requires obtaining the matching CV clips separately)

License:  CC-0 for both audio and metadata
Size:     1,712 clips, ~2.4 h, 17 distinct English accents

The tarball is the official Artie Inc. distribution of the bias-evaluation
subset — it bundles the CV June-2019 MP3 clips referenced by the TSV
so you don't have to wrangle the Mozilla Data Collective.

Usage:
    # Default (tarball-first, recommended):
    python experiments/bias_pruning/data_external/download_artie.py

    # Skip the tarball and only clone the GitHub repo (TSV-only, audio manual):
    python experiments/bias_pruning/data_external/download_artie.py --tsv_only
"""

from __future__ import annotations

import argparse
import shutil
import ssl
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ARTIE_TARBALL = "http://ml-corpora.artie.com/artie-bias-corpus.tar.gz"
ARTIE_REPO = "https://github.com/artie-inc/artie-bias-corpus.git"


def parse_args():
    p = argparse.ArgumentParser(description="Download the Artie Bias Corpus.")
    p.add_argument("--output_dir", type=Path,
                   default=Path(__file__).resolve().parents[3] / "data" / "artie_bias",
                   help="Where to drop the tarball and extract it.")
    p.add_argument("--tsv_only", action="store_true",
                   help="Skip the tarball; clone the GitHub repo for TSV+scripts only.")
    p.add_argument("--no_extract", action="store_true",
                   help="Download the tarball but skip extraction.")
    return p.parse_args()


def _http_get(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[Artie] Already on disk: {dest}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[Artie] Downloading {url}")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (bias_pruning)"})
    with urllib.request.urlopen(req, context=ctx, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", "0"))
        got = 0
        chunk = 1024 * 256
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
            got += len(buf)
            if total:
                pct = 100.0 * got / total
                sys.stdout.write(f"\r   {got/1e6:.1f} / {total/1e6:.1f} MB ({pct:.1f}%)")
                sys.stdout.flush()
        sys.stdout.write("\n")
    tmp.rename(dest)


def _extract_tar(targz: Path, dest: Path) -> None:
    print(f"[Artie] Extracting {targz.name} -> {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(targz, "r:gz") as tar:
        tar.extractall(dest)


def _clone_github(repo: str, dest: Path) -> None:
    if dest.exists():
        print(f"[Artie] Repo already cloned at {dest}; pulling.")
        subprocess.run(["git", "-C", str(dest), "pull"], check=True)
    else:
        print(f"[Artie] Cloning {repo} into {dest}")
        subprocess.run(["git", "clone", "--depth", "1", repo, str(dest)], check=True)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.tsv_only:
        _clone_github(ARTIE_REPO, args.output_dir / "artie-bias-corpus")
        print("""
[Artie] TSV-only mode. To complete the corpus you still need the CV
        June-2019 English MP3s under {clips}/, matching the `path`
        column of artie-bias-corpus.tsv. Mozilla now gates CV downloads
        via the Mozilla Data Collective. Prefer the default mode
        (tarball) unless you have a specific reason to use TSV-only.
""".format(clips=args.output_dir / "clips"))
        return

    # Tarball path (recommended)
    tarball = args.output_dir / "artie-bias-corpus.tar.gz"
    try:
        _http_get(ARTIE_TARBALL, tarball)
    except Exception as e:
        print(f"[Artie] Tarball download failed: {e}")
        print("[Artie] Falling back to GitHub-only mode. You'll still need the CV audio separately.")
        _clone_github(ARTIE_REPO, args.output_dir / "artie-bias-corpus")
        return

    if args.no_extract:
        print(f"[Artie] Tarball at {tarball}. Skipping extraction.")
        return

    _extract_tar(tarball, args.output_dir)

    # Surface what landed.
    tsv_hits = sorted(args.output_dir.rglob("artie-bias-corpus*.tsv"))
    audio_dirs = sorted({p.parent for p in args.output_dir.rglob("*.mp3")})
    if tsv_hits:
        print(f"[Artie] TSV: {tsv_hits[0]}")
    if audio_dirs:
        total_mp3 = sum(1 for _ in args.output_dir.rglob("*.mp3"))
        print(f"[Artie] MP3 audio directories: {audio_dirs}")
        print(f"[Artie] {total_mp3} MP3 clips extracted.")
    else:
        print("[Artie] WARNING: no .mp3 files were extracted from the tarball — investigate.")

    print(f"[Artie] Done. Ready for the HF-format builder step.")


if __name__ == "__main__":
    main()
