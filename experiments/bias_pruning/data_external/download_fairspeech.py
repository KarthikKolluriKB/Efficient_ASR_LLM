"""
Download the Meta Fair-Speech dataset.

Paper:    https://arxiv.org/abs/2408.12734  (Veliche et al., 2024)
Contents: ~26.5K utterances, 593 US-based speakers, voice-command style
Demographics: gender, age, ethnicity, geographic variation, native-English flag
License:  TBD (check the official release page)

STATUS AS OF MAY 2026: No clear public download URL surfaced from a web
search. The paper says the dataset is "publicly released" but does not
link to a Hub repo or download mirror. Three places to check, in order:

    1. https://huggingface.co/datasets?search=fair-speech
       Look under `facebook/...` or `meta-llama/...` for an official drop.
    2. https://github.com/facebookresearch — search for "fair speech".
       Meta usually publishes a small loader repo alongside the paper.
    3. Email the paper's corresponding author (Irina-Elena Veliche).

Once you have a path / URL, fill in the FAIR_SPEECH_SOURCE constant below
and re-run; the script will then handle the rest (extract, save to HF
format under {output_dir}).

Usage (after configuration):
    python experiments/bias_pruning/data_external/download_fairspeech.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Fill in one of:
#   ("hf", "<repo_id>")          -> use datasets.load_dataset(repo_id)
#   ("url", "<tar.gz_or_zip>")   -> download + extract via stdlib
#   ("local", "<path>")          -> use a directory already on disk
FAIR_SPEECH_SOURCE: tuple[str, str] | None = None


def parse_args():
    p = argparse.ArgumentParser(description="Download Meta Fair-Speech (configure source first).")
    p.add_argument("--output_dir", type=Path,
                   default=Path(__file__).resolve().parents[3] / "data" / "fairspeech",
                   help="Directory to save the dataset.")
    p.add_argument("--source_hf", type=str, default=None,
                   help="HF repo_id to download from (overrides FAIR_SPEECH_SOURCE).")
    p.add_argument("--source_url", type=str, default=None,
                   help="Direct URL (tar.gz/zip) to download (overrides FAIR_SPEECH_SOURCE).")
    return p.parse_args()


def main():
    args = parse_args()
    source: tuple[str, str] | None = None
    if args.source_hf:
        source = ("hf", args.source_hf)
    elif args.source_url:
        source = ("url", args.source_url)
    elif FAIR_SPEECH_SOURCE is not None:
        source = FAIR_SPEECH_SOURCE

    if source is None:
        print("""
[Fair-Speech] No source configured. To proceed, do ONE of:

  1. Find the official HF repo, e.g. `facebook/fair-speech-2024`, then:
       python download_fairspeech.py --source_hf facebook/fair-speech-2024

  2. Find a direct download URL (tar.gz / zip):
       python download_fairspeech.py --source_url https://.../fair-speech.tar.gz

  3. Edit FAIR_SPEECH_SOURCE in this script to bake the source in.

The paper to check: https://arxiv.org/abs/2408.12734
""")
        sys.exit(1)

    kind, value = source
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if kind == "hf":
        try:
            from datasets import load_dataset
        except ImportError:
            raise SystemExit("Install `datasets`: pip install datasets")
        print(f"[Fair-Speech] Loading HF dataset: {value}")
        ds = load_dataset(value)
        ds.save_to_disk(str(args.output_dir))
        print(f"[Fair-Speech] Saved to {args.output_dir}")
        return

    if kind == "url":
        import urllib.request
        out_arc = args.output_dir / Path(value).name
        if not out_arc.exists():
            print(f"[Fair-Speech] Downloading {value}")
            urllib.request.urlretrieve(value, out_arc)
        print(f"[Fair-Speech] Extracting {out_arc} -> {args.output_dir}")
        if out_arc.suffix in (".gz", ".tgz") or out_arc.name.endswith(".tar.gz"):
            import tarfile
            with tarfile.open(out_arc, "r:*") as tar:
                tar.extractall(args.output_dir)
        elif out_arc.suffix == ".zip":
            import zipfile
            with zipfile.ZipFile(out_arc, "r") as zf:
                zf.extractall(args.output_dir)
        else:
            print(f"[Fair-Speech] Unknown archive type {out_arc.suffix}; leaving as-is.")
        return

    if kind == "local":
        print(f"[Fair-Speech] Using local path: {value} (no action taken).")
        return

    raise SystemExit(f"Unknown source kind: {kind}")


if __name__ == "__main__":
    main()
