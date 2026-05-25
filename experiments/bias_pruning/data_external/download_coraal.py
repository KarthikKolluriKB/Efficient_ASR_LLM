"""
Download the CORAAL (Corpus of Regional African American Language) components.

Source:   http://lingtools.uoregon.edu/coraal/ (v. 2023.06)
License:  CC BY-NC-SA 4.0
Components (subcorpora): ATL, DCA, DCB, DTA, LES, PRV, ROC, VLD ... etc.
Each component ships separately: audio (.tar.gz of WAV) + metadata + transcripts.

Strategy:
    1. Fetch the components index page.
    2. Parse out the component directories and the tar.gz download URLs.
    3. Download each component's audio and transcript archives (resumable).
    4. Optionally extract them into the output directory.

Audio is uncompressed WAV in long-form interviews (30-60 min files each).
The interview-level audio must later be chunked into <=30 s segments
aligned with transcript timestamps before the ASR pipeline can use it.

Usage:
    # Dry run: list what would be downloaded
    python experiments/bias_pruning/data_external/download_coraal.py --list_only

    # Download a single component first to test (DCA is small-ish)
    python experiments/bias_pruning/data_external/download_coraal.py --components DCA

    # Full corpus
    python experiments/bias_pruning/data_external/download_coraal.py --all
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

INDEX_URL = "http://lingtools.uoregon.edu/coraal/"


def parse_args():
    p = argparse.ArgumentParser(description="Download CORAAL components.")
    p.add_argument("--output_dir", type=Path,
                   default=Path(__file__).resolve().parents[3] / "data" / "coraal",
                   help="Where to drop the .tar.gz archives.")
    p.add_argument("--components", nargs="*", default=None,
                   help="Specific components to download (e.g. ATL DCA DCB). Case-insensitive.")
    p.add_argument("--all", action="store_true", help="Download every component listed in the index.")
    p.add_argument("--list_only", action="store_true", help="Print available components and URLs, do not download.")
    p.add_argument("--no_extract", action="store_true", help="Skip the tar -xzf step after download.")
    return p.parse_args()


def fetch_index_html() -> str:
    print(f"[CORAAL] Fetching index from {INDEX_URL}")
    req = urllib.request.Request(INDEX_URL, headers={"User-Agent": "Mozilla/5.0 (bias_pruning)"})
    # ssl verify off because lingtools.uoregon.edu sometimes has chain issues
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(req, context=ctx, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def parse_targz_links(html: str) -> dict[str, list[str]]:
    """Extract .tar.gz download links from the index page, grouped by component."""
    # Match href="...XXX/YYY.tar.gz" where the component is usually an
    # uppercase 3-4 letter code in the path or filename.
    hrefs = re.findall(r'href=["\']([^"\']+\.tar\.gz)["\']', html, flags=re.IGNORECASE)
    by_comp: dict[str, list[str]] = {}
    for h in hrefs:
        full = urllib.parse.urljoin(INDEX_URL, h)
        # Component code = first 3-4 uppercase letters of the filename (e.g. DCA_audio_part00.tar.gz)
        fname = Path(urllib.parse.urlparse(full).path).name
        m = re.match(r"([A-Z]{3,4})", fname)
        comp = m.group(1) if m else "UNKNOWN"
        by_comp.setdefault(comp, []).append(full)
    return by_comp


def download_one(url: str, out_path: Path) -> None:
    if out_path.exists():
        print(f"[CORAAL] Skipping (exists): {out_path.name}")
        return
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    print(f"[CORAAL] Downloading {url}")
    import ssl
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
    tmp.rename(out_path)


def maybe_extract(targz: Path, dest_dir: Path) -> None:
    import tarfile
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"[CORAAL] Extracting {targz.name} -> {dest_dir}")
    with tarfile.open(targz, "r:gz") as tar:
        tar.extractall(dest_dir)


def main():
    args = parse_args()
    html = fetch_index_html()
    by_comp = parse_targz_links(html)
    if not by_comp:
        print("[CORAAL] No .tar.gz links found in index. The site layout may have changed; "
              "fall back to manual download from http://lingtools.uoregon.edu/coraal/")
        sys.exit(1)

    print("[CORAAL] Components and downloadable files:")
    for comp, urls in sorted(by_comp.items()):
        print(f"   {comp}: {len(urls)} archive(s)")
        for u in urls:
            print(f"      {u}")

    if args.list_only:
        return

    if args.all:
        chosen = sorted(by_comp.keys())
    elif args.components:
        chosen = [c.upper() for c in args.components if c.upper() in by_comp]
        missing = [c.upper() for c in args.components if c.upper() not in by_comp]
        if missing:
            print(f"[CORAAL] Requested components not found: {missing}")
    else:
        print("[CORAAL] Nothing selected. Pass --components ATL DCA ... or --all.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for comp in chosen:
        comp_dir = args.output_dir / comp
        comp_dir.mkdir(parents=True, exist_ok=True)
        for url in by_comp[comp]:
            fname = Path(urllib.parse.urlparse(url).path).name
            out = comp_dir / fname
            download_one(url, out)
            if not args.no_extract:
                maybe_extract(out, comp_dir)

    print("[CORAAL] Done.")


if __name__ == "__main__":
    main()
