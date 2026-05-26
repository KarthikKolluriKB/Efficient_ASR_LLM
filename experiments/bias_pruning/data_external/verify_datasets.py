"""
Verify that the four downloaded external datasets are on disk and readable.

Default paths follow what the download_*.py scripts produce:
    data/edacc/             (HF DatasetDict: validation + test)
    data/l2arctic/          (HF DatasetDict: scripted + spontaneous)
    data/fairspeech/        (audio/ + metadata.tsv)
    data/coraal/<COMPONENT>/ (.wav + transcripts + .TextGrid/.eaf/.txt)

For each present dataset, prints:
    - Path and total size on disk
    - Sample count (rows or files)
    - Schema (column names for HF datasets; file types for CORAAL)
    - One example row, with the audio array truncated for readability

Usage:
    python experiments/bias_pruning/data_external/verify_datasets.py
    python experiments/bias_pruning/data_external/verify_datasets.py --only edacc
    python experiments/bias_pruning/data_external/verify_datasets.py --data_root /scratch/data
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _du(path: Path) -> str:
    if not path.exists():
        return "(missing)"
    total = 0
    if path.is_file():
        total = path.stat().st_size
    else:
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                pass
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if total < 1024 or unit == "TB":
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} TB"


def _truncate_value(v, max_len: int = 80) -> str:
    """Print-friendly truncation. Hide raw audio arrays."""
    if isinstance(v, dict):
        if "array" in v:
            n = len(v["array"]) if v.get("array") is not None else 0
            sr = v.get("sampling_rate", "?")
            return f"<audio: {n} samples @ {sr} Hz>"
        if "bytes" in v or "path" in v:
            # decode=False form: dict with `bytes` and/or `path`
            path = v.get("path")
            nbytes = len(v["bytes"]) if v.get("bytes") is not None else 0
            return f"<audio (undecoded): path={path!r}, bytes={nbytes}>"
    s = repr(v)
    return s if len(s) <= max_len else s[:max_len - 3] + "..."


def _disable_audio_decoding(ds):
    """Cast every Audio column to decode=False so row access doesn't pull torchcodec.

    Works on a single Dataset or a DatasetDict. Returns the (possibly modified) object.
    The verifier doesn't need decoded audio — it just needs to confirm the row schema.
    Downstream HF-dataset builders will decode at preprocessing time with soundfile.
    """
    try:
        from datasets import Audio, DatasetDict
    except ImportError:
        return ds

    def _cast(d):
        for col, feat in list(d.features.items()):
            if isinstance(feat, Audio):
                d = d.cast_column(col, Audio(decode=False))
        return d

    if isinstance(ds, DatasetDict):
        for k in list(ds.keys()):
            ds[k] = _cast(ds[k])
        return ds
    return _cast(ds)


def _print_header(name: str, path: Path) -> None:
    print(f"\n=== {name} ===")
    print(f"path: {path}")
    print(f"size: {_du(path)}")


def _print_sample(row: dict, max_cols: int = 12) -> None:
    keys = list(row.keys())[:max_cols]
    width = max(len(k) for k in keys) if keys else 0
    print("  sample row:")
    for k in keys:
        print(f"    {k.ljust(width)}  {_truncate_value(row[k])}")
    if len(row) > max_cols:
        print(f"    ... ({len(row) - max_cols} more columns)")


# ---------------------------------------------------------------------------
# Per-dataset checks
# ---------------------------------------------------------------------------

def check_edacc(path: Path) -> bool:
    _print_header("EdAcc", path)
    if not path.exists():
        print("  STATUS: missing — run `python data_external/download_edacc.py`")
        return False
    try:
        from datasets import load_from_disk
    except ImportError:
        print("  STATUS: cannot verify — `pip install datasets`")
        return False
    try:
        ds = load_from_disk(str(path))
    except Exception as e:
        print(f"  STATUS: failed to load HF dataset: {e}")
        return False
    ds = _disable_audio_decoding(ds)
    print(f"  splits: {list(ds.keys())}")
    for s in ds:
        d = ds[s]
        print(f"    {s}: {len(d)} rows, columns = {d.column_names}")
    first_split = next(iter(ds.values()))
    _print_sample(first_split[0])
    expected = {"speaker", "text", "accent", "gender", "l1", "audio"}
    missing = expected - set(first_split.column_names)
    if missing:
        print(f"  WARNING: expected columns missing: {missing}")
    print("  STATUS: OK")
    return True


def check_l2arctic(path: Path) -> bool:
    _print_header("L2-ARCTIC", path)
    if not path.exists():
        print("  STATUS: missing — run `python data_external/download_l2arctic.py`")
        return False
    try:
        from datasets import load_from_disk
    except ImportError:
        print("  STATUS: cannot verify — `pip install datasets`")
        return False
    try:
        ds = load_from_disk(str(path))
    except Exception as e:
        print(f"  STATUS: failed to load HF dataset: {e}")
        return False
    ds = _disable_audio_decoding(ds)
    print(f"  splits: {list(ds.keys())}")
    for s in ds:
        d = ds[s]
        print(f"    {s}: {len(d)} rows, columns = {d.column_names}")
    first_split = next(iter(ds.values()))
    _print_sample(first_split[0])
    print("  STATUS: OK")
    return True


def check_fairspeech(path: Path) -> bool:
    _print_header("Fair-Speech (Meta ASR Fairness)", path)
    if not path.exists():
        print("  STATUS: missing — run `python data_external/download_fairspeech.py --zip_path ... --metadata_path ...`")
        return False

    meta_path = path / "metadata.tsv"
    audio_dir = path / "audio"
    zip_path = next(path.glob("asr_fairness_audio.zip*"), None)

    if not meta_path.exists():
        print(f"  STATUS: missing metadata.tsv at {meta_path}")
        return False

    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        cols = reader.fieldnames or []
        first_row = next(reader, None)
        rest_count = sum(1 for _ in reader)
    print(f"  metadata.tsv: {1 + rest_count} rows" + (f" (first row + {rest_count})" if rest_count else ""))
    print(f"  columns: {cols}")
    if first_row:
        _print_sample(first_row)

    expected_any = {"age", "gender", "ethnicity"}
    if not any(c.lower() in {ec for ec in expected_any} for c in cols):
        # The actual Meta TSV column names may differ — flag for inspection rather than fail.
        print(f"  NOTE: none of {expected_any} found in column names. "
              "Check the TSV header — Meta's column names may differ from the README description.")

    if audio_dir.exists():
        n_audio = sum(1 for _ in audio_dir.rglob("*") if _.is_file())
        print(f"  audio files extracted: {n_audio}")
        if n_audio == 0 and zip_path is not None:
            print(f"  zip present but not extracted: {zip_path.name}")
    elif zip_path is not None:
        print(f"  audio not extracted yet; zip present at {zip_path}")
    else:
        print("  WARNING: no audio/ folder and no asr_fairness_audio.zip — re-run the downloader")

    print("  REMINDER: do not commit metadata.tsv or per-utterance CSVs from this dataset.")
    print("  STATUS: OK" if (audio_dir.exists() or zip_path is not None) else "  STATUS: metadata-only")
    return True


def _is_macos_junk(p: Path) -> bool:
    """macOS extended-attribute resource-fork files. Created when tarballs
    are produced on a Mac; not part of the actual CORAAL distribution."""
    return p.name.startswith("._")


def _list_files(dirp: Path, suffix: str) -> list[Path]:
    return [p for p in dirp.rglob(f"*{suffix}") if not _is_macos_junk(p)]


def check_coraal(path: Path) -> bool:
    _print_header("CORAAL", path)
    if not path.exists():
        print("  STATUS: missing — run `python data_external/download_coraal.py --components DCA`")
        return False

    components = [p for p in path.iterdir() if p.is_dir() and not _is_macos_junk(p)]
    if not components:
        print(f"  STATUS: no components found under {path}")
        return False
    print(f"  components on disk: {[c.name for c in components]}")
    total_junk = 0
    for comp in sorted(components, key=lambda p: p.name):
        wavs = _list_files(comp, ".wav")
        txts = _list_files(comp, ".txt")
        tgs = _list_files(comp, ".TextGrid")
        eafs = _list_files(comp, ".eaf")
        tars = _list_files(comp, ".tar.gz")
        # Heuristic for a per-speaker demographics file shipped with each
        # CORAAL component (e.g. CORAAL_DCB_metadata_2020.05.txt).
        demos = [p for p in txts if "metadata" in p.name.lower() or "speaker" in p.name.lower()]
        junk = sum(1 for p in comp.rglob("*") if _is_macos_junk(p))
        total_junk += junk

        print(f"    {comp.name}: {len(wavs)} wav, {len(txts)} txt, "
              f"{len(tgs)} TextGrid, {len(eafs)} eaf, {len(tars)} tar.gz"
              + (f"  [+{junk} macOS junk filtered]" if junk else ""))
        if wavs:
            sample = wavs[0]
            print(f"    sample wav: {sample.name} ({sample.stat().st_size/1e6:.1f} MB)")
        # Prefer a non-metadata txt for the transcript-head preview.
        non_demo = [t for t in txts if t not in demos]
        if non_demo:
            txt = non_demo[0]
            try:
                head = open(txt, "r", encoding="utf-8", errors="replace").read(400)
                print(f"    sample transcript head ({txt.name}):")
                for line in head.splitlines()[:4]:
                    print(f"      {line}")
            except Exception as e:
                print(f"    (could not read {txt.name}: {e})")
        if demos:
            print(f"    demographics file(s): {[p.name for p in demos]}")
        else:
            print(f"    NOTE: no per-speaker metadata file found in {comp.name}; "
                  "may live alongside the component or in a top-level CORAAL_README/.")

    if total_junk:
        print(f"  ({total_junk} macOS resource-fork files filtered across all components)")
    print("  NOTE: CORAAL interviews are long-form. Plan a chunking step "
          "(transcript timestamps -> <=30 s segments) before ASR eval.")
    print("  STATUS: OK")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

CHECKS = {
    "edacc":      ("data/edacc",      check_edacc),
    "l2arctic":   ("data/l2arctic",   check_l2arctic),
    "fairspeech": ("data/fairspeech", check_fairspeech),
    "coraal":     ("data/coraal",     check_coraal),
}


def parse_args():
    p = argparse.ArgumentParser(description="Verify the four downloaded fairness datasets.")
    p.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT,
                   help="Base directory under which the four datasets live (default: ./data).")
    p.add_argument("--only", choices=sorted(CHECKS.keys()), default=None,
                   help="Verify just one dataset.")
    return p.parse_args()


def main():
    args = parse_args()
    targets = [args.only] if args.only else list(CHECKS.keys())

    results = {}
    for name in targets:
        rel, fn = CHECKS[name]
        path = args.data_root / Path(rel).relative_to("data")
        try:
            results[name] = fn(path)
        except Exception as e:
            print(f"\n=== {name} ===\n  STATUS: verifier crashed — {e!r}")
            results[name] = False

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name in targets:
        flag = "OK" if results.get(name) else "MISSING/INCOMPLETE"
        print(f"  {name:<12} {flag}")
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
