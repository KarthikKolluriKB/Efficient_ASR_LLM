"""
Step 2 — Build the demographic descriptive table for the CommonVoice 22 English test split.

Downloads transcript/en/test.tsv from the upstream CommonVoice 22 repository
(no audio download required), aggregates by gender and age, writes
results/descriptive_table.csv, and prints which gender cells pass the
analysability threshold.

Threshold for analysability (must hold both):
    - >= 30 minutes (1800 s) of audio in the cell
    - >= 200 utterances

The `duration` column is not part of the upstream TSV. We estimate per-row
duration in one of two ways:
    1. If an HF dataset is available locally at data/cv22_hf/en, we join on
       (speaker_id_prefix, normalised_sentence) to read true durations.
    2. Otherwise we use the standard CV duration proxy of 0.072 sec per
       character of the sentence (close to the population mean for CV en;
       only used so the threshold check has *some* signal — the final WER
       analysis always uses real durations from the HF dataset).

Usage:
    python experiments/bias_pruning/scripts/build_descriptive_table.py
    python experiments/bias_pruning/scripts/build_descriptive_table.py \
        --tsv_path /custom/path/to/test.tsv \
        --hf_dataset_path data/cv22_hf/en \
        --output_path experiments/bias_pruning/results/descriptive_table.csv \
        --test_split_path experiments/bias_pruning/data/splits/test_subset.tsv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Resolve project root so this script works when invoked from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_TSV_REPO = "fsicoli/common_voice_22_0"
DEFAULT_TSV_FILENAME = "transcript/en/test.tsv"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments/bias_pruning/results/descriptive_table.csv"
DEFAULT_HF_DATASET = PROJECT_ROOT / "data/cv22_hf/en"

GENDER_BUCKETS = ["male", "female", "other", "missing"]
AGE_BUCKETS = [
    "teens", "twenties", "thirties", "fourties", "fifties",
    "sixties", "seventies", "eighties", "nineties", "missing",
]

# CV duration proxy: seconds per character of `sentence`. Used only if no HF
# dataset is available; final analyses use real durations.
SEC_PER_CHAR_FALLBACK = 0.072

ANALYSABLE_MIN_SECONDS = 30 * 60  # 30 minutes
ANALYSABLE_MIN_UTTS = 200


@dataclass
class RowAgg:
    n_utts: int = 0
    n_speakers: int = 0
    duration_s: float = 0.0
    speakers: set = None  # populated lazily

    def __post_init__(self):
        if self.speakers is None:
            self.speakers = set()

    def add(self, client_id: str, duration_s: float):
        self.n_utts += 1
        self.duration_s += duration_s
        self.speakers.add(client_id)

    def finalize(self):
        self.n_speakers = len(self.speakers)


def _norm_gender(raw: str | None) -> str:
    if raw is None:
        return "missing"
    g = raw.strip().lower()
    if g in {"", "nan", "none"}:
        return "missing"
    if g.startswith("male") or g == "m" or g == "male_masculine":
        return "male"
    if g.startswith("female") or g == "f" or g == "female_feminine":
        return "female"
    return "other"


def _norm_age(raw: str | None) -> str:
    if raw is None:
        return "missing"
    a = raw.strip().lower()
    if a in {"", "nan", "none"}:
        return "missing"
    if a in AGE_BUCKETS:
        return a
    # CV has a few alternate spellings.
    aliases = {
        "fourties": "fourties",
        "forties": "fourties",
    }
    return aliases.get(a, "missing")


def _normalise_sentence(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _download_test_tsv() -> Path:
    """Download CV22 transcript/en/test.tsv via huggingface_hub. Cached after first call."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit(
            "huggingface_hub is required to download the CV22 transcript. "
            "Install it with `pip install huggingface_hub` or pass --tsv_path."
        ) from e
    path = hf_hub_download(
        repo_id=DEFAULT_TSV_REPO,
        filename=DEFAULT_TSV_FILENAME,
        repo_type="dataset",
    )
    return Path(path)


def _load_test_subset(split_path: Path | None) -> set[str] | None:
    """Optional restriction to a subset of the upstream test split.

    Expects either a CSV/TSV with a `path` column, or a plain text file
    with one filename per line. Returns the set of CV `path` values to keep,
    or None if no subset file was provided.
    """
    if split_path is None:
        return None
    if not split_path.exists():
        raise FileNotFoundError(f"--test_split_path not found: {split_path}")

    keep: set[str] = set()
    with open(split_path, "r", encoding="utf-8") as f:
        first = f.readline().strip()
        rest = f.readlines()
    has_header = "\t" in first or "," in first
    lines = [first, *[ln.strip() for ln in rest]]
    if has_header:
        delim = "\t" if "\t" in first else ","
        reader = csv.DictReader(lines, delimiter=delim)
        for row in reader:
            p = row.get("path") or row.get("filename") or row.get("clip")
            if p:
                keep.add(p)
    else:
        keep.update(ln for ln in lines if ln)
    return keep


def _try_load_hf_durations(hf_path: Path) -> dict[tuple[str, str], float] | None:
    """If a local HF dataset exists, build a (speaker_id_prefix, norm_sentence) -> duration map.

    The HF dataset stores `speaker_id` truncated to 16 chars and either
    `raw_transcription` or `transcription` for the text. Both forms are
    indexed so the join works regardless of which field a downstream user
    keyed against.
    """
    if not hf_path.exists():
        return None
    try:
        from datasets import load_from_disk
    except ImportError:
        print("[Warning] `datasets` not installed; falling back to char-based duration proxy.")
        return None

    try:
        ds = load_from_disk(str(hf_path))
    except Exception as e:
        print(f"[Warning] Could not load HF dataset at {hf_path}: {e}")
        return None

    if "test" not in ds:
        print(f"[Warning] HF dataset at {hf_path} has no `test` split.")
        return None
    test = ds["test"]
    cols = set(test.column_names)

    speaker_col = "speaker_id" if "speaker_id" in cols else None
    if speaker_col is None:
        return None

    text_cols = [c for c in ("raw_transcription", "transcription") if c in cols]
    if not text_cols or "duration" not in cols:
        return None

    print(f"[Info] Loading durations from local HF dataset: {hf_path}")
    out: dict[tuple[str, str], float] = {}
    for row in test:
        sp = (row[speaker_col] or "")[:16]
        dur = float(row["duration"])
        for tc in text_cols:
            key = (sp, _normalise_sentence(row[tc] or ""))
            out[key] = dur
    return out


def _estimate_duration(sentence: str) -> float:
    return max(1.0, SEC_PER_CHAR_FALLBACK * len(sentence))


def aggregate(
    tsv_path: Path,
    subset_paths: set[str] | None,
    hf_durations: dict[tuple[str, str], float] | None,
) -> tuple[dict[str, RowAgg], dict[str, RowAgg], dict[tuple[str, str], RowAgg], int, int]:
    by_gender: dict[str, RowAgg] = {g: RowAgg() for g in GENDER_BUCKETS}
    by_age: dict[str, RowAgg] = {a: RowAgg() for a in AGE_BUCKETS}
    by_gender_age: dict[tuple[str, str], RowAgg] = defaultdict(RowAgg)

    total_rows = 0
    used_rows = 0
    used_real_duration = 0
    used_proxy = 0

    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            total_rows += 1
            path = row.get("path", "")
            if subset_paths is not None and path not in subset_paths:
                continue

            client_id = row.get("client_id", "")
            gender = _norm_gender(row.get("gender"))
            age = _norm_age(row.get("age"))
            sentence = row.get("sentence", "") or ""

            duration_s = None
            if hf_durations is not None:
                duration_s = hf_durations.get((client_id[:16], _normalise_sentence(sentence)))
            if duration_s is None:
                duration_s = _estimate_duration(sentence)
                used_proxy += 1
            else:
                used_real_duration += 1

            by_gender[gender].add(client_id, duration_s)
            by_age[age].add(client_id, duration_s)
            by_gender_age[(gender, age)].add(client_id, duration_s)
            used_rows += 1

    for agg in by_gender.values():
        agg.finalize()
    for agg in by_age.values():
        agg.finalize()
    for agg in by_gender_age.values():
        agg.finalize()

    print(
        f"[Info] Rows in TSV: {total_rows} | rows used (post subset filter): {used_rows} | "
        f"durations from HF: {used_real_duration} | duration proxy used: {used_proxy}"
    )
    return by_gender, by_age, by_gender_age, total_rows, used_rows


def write_csv(
    out_path: Path,
    by_gender: dict[str, RowAgg],
    by_age: dict[str, RowAgg],
    by_gender_age: dict[tuple[str, str], RowAgg],
    total_used: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "section", "stratum", "n_utts", "duration_hours",
            "n_unique_speakers", "pct_of_utts", "analysable",
        ])

        for g in GENDER_BUCKETS:
            agg = by_gender[g]
            pct = (agg.n_utts / total_used * 100.0) if total_used else 0.0
            analysable = (agg.duration_s >= ANALYSABLE_MIN_SECONDS) and (agg.n_utts >= ANALYSABLE_MIN_UTTS)
            w.writerow([
                "gender", g, agg.n_utts, round(agg.duration_s / 3600.0, 3),
                agg.n_speakers, round(pct, 2), "yes" if analysable else "no",
            ])

        for a in AGE_BUCKETS:
            agg = by_age[a]
            pct = (agg.n_utts / total_used * 100.0) if total_used else 0.0
            analysable = (agg.duration_s >= ANALYSABLE_MIN_SECONDS) and (agg.n_utts >= ANALYSABLE_MIN_UTTS)
            w.writerow([
                "age", a, agg.n_utts, round(agg.duration_s / 3600.0, 3),
                agg.n_speakers, round(pct, 2), "yes" if analysable else "no",
            ])

        for (g, a) in sorted(by_gender_age.keys()):
            agg = by_gender_age[(g, a)]
            pct = (agg.n_utts / total_used * 100.0) if total_used else 0.0
            analysable = (agg.duration_s >= ANALYSABLE_MIN_SECONDS) and (agg.n_utts >= ANALYSABLE_MIN_UTTS)
            w.writerow([
                "gender_x_age", f"{g}|{a}", agg.n_utts, round(agg.duration_s / 3600.0, 3),
                agg.n_speakers, round(pct, 2), "yes" if analysable else "no",
            ])


def print_summary(
    by_gender: dict[str, RowAgg],
    by_age: dict[str, RowAgg],
    total_used: int,
) -> None:
    print("\n=== Gender breakdown (CommonVoice 22 English test split) ===")
    print(f"{'gender':<10} {'n_utts':>8} {'hours':>8} {'speakers':>10} {'pct_utts':>10} {'analysable':>12}")
    for g in GENDER_BUCKETS:
        a = by_gender[g]
        pct = (a.n_utts / total_used * 100.0) if total_used else 0.0
        flag = "yes" if (a.duration_s >= ANALYSABLE_MIN_SECONDS and a.n_utts >= ANALYSABLE_MIN_UTTS) else "no"
        print(f"{g:<10} {a.n_utts:>8d} {a.duration_s/3600.0:>8.2f} {a.n_speakers:>10d} {pct:>9.2f}% {flag:>12}")

    print("\n=== Age breakdown ===")
    print(f"{'age':<10} {'n_utts':>8} {'hours':>8} {'speakers':>10} {'pct_utts':>10} {'analysable':>12}")
    for a_name in AGE_BUCKETS:
        a = by_age[a_name]
        pct = (a.n_utts / total_used * 100.0) if total_used else 0.0
        flag = "yes" if (a.duration_s >= ANALYSABLE_MIN_SECONDS and a.n_utts >= ANALYSABLE_MIN_UTTS) else "no"
        print(f"{a_name:<10} {a.n_utts:>8d} {a.duration_s/3600.0:>8.2f} {a.n_speakers:>10d} {pct:>9.2f}% {flag:>12}")

    missing_pct = (by_gender['missing'].n_utts / total_used * 100.0) if total_used else 0.0
    print(f"\nMissing-gender share of utterances: {missing_pct:.2f}%")

    analysable_genders = [
        g for g in GENDER_BUCKETS
        if by_gender[g].duration_s >= ANALYSABLE_MIN_SECONDS
        and by_gender[g].n_utts >= ANALYSABLE_MIN_UTTS
    ]
    print(f"Analysable gender cells (>= {ANALYSABLE_MIN_SECONDS//60} min and >= {ANALYSABLE_MIN_UTTS} utts): {analysable_genders}")
    if "male" not in analysable_genders or "female" not in analysable_genders:
        print("[WARNING] male or female cell falls below the analysability threshold. "
              "The experiment design needs to change before continuing.")


def parse_args():
    p = argparse.ArgumentParser(description="Build the gender/age descriptive table for CV22 English test.")
    p.add_argument("--tsv_path", type=Path, default=None,
                   help="Local path to transcript/en/test.tsv. If omitted, downloaded from HF Hub.")
    p.add_argument("--hf_dataset_path", type=Path, default=DEFAULT_HF_DATASET,
                   help="Local HF dataset (used only to read real durations). Optional.")
    p.add_argument("--test_split_path", type=Path, default=None,
                   help="Optional file listing utterance `path` values to keep (subset of test.tsv).")
    p.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT,
                   help="Where to write the descriptive table CSV.")
    return p.parse_args()


def main():
    args = parse_args()

    tsv_path = args.tsv_path
    if tsv_path is None:
        print("[Info] Downloading CV22 en/test.tsv from HF Hub (no audio, ~ a few MB)...")
        tsv_path = _download_test_tsv()
    elif not tsv_path.exists():
        raise SystemExit(f"--tsv_path does not exist: {tsv_path}")
    print(f"[Info] Using transcript TSV: {tsv_path}")

    subset_paths = _load_test_subset(args.test_split_path)
    if subset_paths is not None:
        print(f"[Info] Restricting to {len(subset_paths)} utterances from {args.test_split_path}")

    hf_durations = _try_load_hf_durations(args.hf_dataset_path)
    if hf_durations is None:
        print("[Info] No local HF dataset available; using character-based duration proxy.")

    by_gender, by_age, by_gender_age, total_rows, used_rows = aggregate(tsv_path, subset_paths, hf_durations)

    write_csv(args.output_path, by_gender, by_age, by_gender_age, used_rows)
    print(f"[Info] Wrote descriptive table to: {args.output_path}")

    print_summary(by_gender, by_age, used_rows)


if __name__ == "__main__":
    main()
