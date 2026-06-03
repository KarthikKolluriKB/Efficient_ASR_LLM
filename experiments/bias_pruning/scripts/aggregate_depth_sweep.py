"""
Aggregate a whisper depth-sweep of per-gender WER/CER summaries into a
findings table (CSV + Markdown) and a single depth-vs-error figure.

Companion to run_depth_sweep.py: that produces one per-gender summary CSV per
pruning depth (pulled here from wandb into --in_dir). This script collapses the
whole sweep into the headline deliverables described in SMALL_SWEEP.md.

Input: a directory of per-gender summary CSVs named d{NN}*.csv, each with the
schema written by evaluate_subgroup_wer.py:
    gender, n_utts, duration_hours, wer, wer_ci_low, wer_ci_high,
    cer, cer_ci_low, cer_ci_high, analysable
one row per gender bucket (male, female, other, missing, ALL). The depth is
parsed from the filename: d00 = unpruned (12 layers kept), d11 = 1 layer kept.

Outputs (under --out_dir):
    depth_sweep_findings.csv   one row per depth: per-gender WER/CER + male-female gap
    depth_sweep_findings.md    same, rendered as a Markdown table (WER/CER as %)
    depth_sweep_plot.png       depth-vs-WER and depth-vs-CER, one line per gender,
                               full y-range with the collapse regime shaded

Usage:
    python experiments/bias_pruning/scripts/aggregate_depth_sweep.py \
        --in_dir  experiments/bias_pruning/results/small_sweep_en/per_gender \
        --out_dir experiments/bias_pruning/results/small_sweep_en \
        --total_layers 12
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IN = PROJECT_ROOT / "experiments/bias_pruning/results/small_sweep_en/per_gender"
DEFAULT_OUT = PROJECT_ROOT / "experiments/bias_pruning/results/small_sweep_en"

# Buckets reported / plotted. 'other' is dropped — n=11, flagged analysable=no
# in every depth file (its CIs are degenerate point values).
REPORT_GENDERS = ["male", "female", "missing", "ALL"]
_NUMERIC_FIELDS = (
    "wer", "wer_ci_low", "wer_ci_high",
    "cer", "cer_ci_low", "cer_ci_high",
    "duration_hours",
)


def parse_depth(path: Path) -> int:
    m = re.match(r"d0*(\d+)", path.stem)
    if not m:
        raise ValueError(f"Cannot parse depth from filename: {path.name!r}")
    return int(m.group(1))


def load_depth_file(path: Path) -> dict[str, dict]:
    """Return {gender: row-with-numeric-fields-coerced}."""
    with open(path, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out: dict[str, dict] = {}
    for r in rows:
        r["n_utts"] = int(r["n_utts"])
        for k in _NUMERIC_FIELDS:
            v = r.get(k, "")
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                r[k] = float("nan")  # handles "", "null", "nan" (wandb JSON export)
        out[r["gender"]] = r
    return out


def build_records(in_dir: Path, total_layers: int) -> list[dict]:
    files = sorted(in_dir.glob("d*.csv"), key=parse_depth)
    if not files:
        raise SystemExit(f"No d*.csv files found in {in_dir}")
    records: list[dict] = []
    for f in files:
        depth = parse_depth(f)
        g = load_depth_file(f)

        def cell(gender: str, metric: str) -> float:
            return g.get(gender, {}).get(metric, float("nan"))

        rec: dict = {
            "depth": depth,
            "layers_kept": total_layers - depth,
            "n_male": g.get("male", {}).get("n_utts", 0),
            "n_female": g.get("female", {}).get("n_utts", 0),
            "n_missing": g.get("missing", {}).get("n_utts", 0),
        }
        for gender in REPORT_GENDERS:
            rec[f"wer_{gender}"] = cell(gender, "wer")
            rec[f"cer_{gender}"] = cell(gender, "cer")
            rec[f"wer_{gender}_lo"] = cell(gender, "wer_ci_low")
            rec[f"wer_{gender}_hi"] = cell(gender, "wer_ci_high")
            rec[f"cer_{gender}_lo"] = cell(gender, "cer_ci_low")
            rec[f"cer_{gender}_hi"] = cell(gender, "cer_ci_high")
        rec["wer_gap_mf"] = rec["wer_male"] - rec["wer_female"]
        rec["cer_gap_mf"] = rec["cer_male"] - rec["cer_female"]
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "depth", "layers_kept", "n_male", "n_female", "n_missing",
    "wer_male", "wer_female", "wer_missing", "wer_ALL", "wer_gap_mf",
    "cer_male", "cer_female", "cer_missing", "cer_ALL", "cer_gap_mf",
]


def write_findings_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in _CSV_FIELDS})


def write_findings_md(records: list[dict], path: Path) -> None:
    def pct(x: float) -> str:
        return f"{x * 100:.2f}"

    lines = [
        "# Whisper-small depth sweep — CommonVoice 22 English (seed 42)",
        "",
        "One independently-trained checkpoint per pruning depth. WER / CER as %, "
        "`analysable` gender cells only (`other`, n=11, excluded).",
        "",
        "| depth | kept | n(m/f/miss) | WER male | WER female | WER miss | **WER ALL** "
        "| gap m−f | CER male | CER female | CER miss | **CER ALL** | gap m−f |",
        "|---:|---:|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        lines.append(
            f"| {r['depth']} | {r['layers_kept']} "
            f"| {r['n_male']}/{r['n_female']}/{r['n_missing']} "
            f"| {pct(r['wer_male'])} | {pct(r['wer_female'])} | {pct(r['wer_missing'])} "
            f"| **{pct(r['wer_ALL'])}** | {r['wer_gap_mf']*100:+.2f} "
            f"| {pct(r['cer_male'])} | {pct(r['cer_female'])} | {pct(r['cer_missing'])} "
            f"| **{pct(r['cer_ALL'])}** | {r['cer_gap_mf']*100:+.2f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plot(records: list[dict], out_path: Path, total_layers: int,
              collapse_wer: float = 1.0,
              dataset_label: str = "CommonVoice 22 English") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Aggregate] matplotlib not available; skipping plot.")
        return

    depths = [r["depth"] for r in records]
    # First depth whose aggregate WER >= collapse_wer marks the collapse onset.
    collapse_onset = next((r["depth"] for r in records if r["wer_ALL"] >= collapse_wer), None)

    style = {  # color, linestyle, linewidth, z
        "male":    ("#d62728", "-", 1.8),
        "female":  ("#1f77b4", "-", 1.8),
        "missing": ("#7f7f7f", "--", 1.4),
        "ALL":     ("#000000", "-", 2.6),
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, metric, label in zip(axes, ("wer", "cer"), ("WER (%)", "CER (%)")):
        for gender in REPORT_GENDERS:
            color, ls, lw = style[gender]
            y = [100 * r[f"{metric}_{gender}"] for r in records]
            ax.plot(depths, y, marker="o", ms=4, lw=lw, ls=ls, color=color,
                    label=gender, zorder=3)
            if gender in ("male", "female"):  # bias-relevant CI bands only
                lo = [100 * r[f"{metric}_{gender}_lo"] for r in records]
                hi = [100 * r[f"{metric}_{gender}_hi"] for r in records]
                ax.fill_between(depths, lo, hi, color=color, alpha=0.12, zorder=1)

        if metric == "wer":
            ax.axhline(100, color="grey", ls=":", lw=1, zorder=2)
        if collapse_onset is not None:
            ax.axvspan(collapse_onset - 0.5, depths[-1] + 0.5,
                       color="red", alpha=0.06, zorder=0)
            top = ax.get_ylim()[1]
            ax.text(collapse_onset + 0.1, top * 0.6,
                    "collapse\n(WER ≥ 100%)", color="#a00000", fontsize=9, va="top")

        ax.set_xticks(depths)
        ax.set_xticklabels([f"{d}\n{total_layers - d}L" for d in depths], fontsize=8)
        ax.set_xlabel("prune depth (layers removed) / layers kept")
        ax.set_ylabel(label)
        ax.set_title(f"{label.split()[0]} vs pruning depth, by gender")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(title="gender", fontsize=9)

    fig.suptitle(
        f"Whisper-small SLAM-ASR depth sweep — {dataset_label} (seed 42, "
        "one checkpoint per depth)\nCI bands = per-group utterance bootstrap (male/female)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Aggregate] Wrote plot: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate a depth-sweep into a findings table + plot.")
    p.add_argument("--in_dir", type=Path, default=DEFAULT_IN,
                   help="Directory of per-gender summary CSVs named d{NN}*.csv.")
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--total_layers", type=int, default=12,
                   help="Total encoder layers (whisper-small=12). Used for 'layers kept'.")
    p.add_argument("--dataset_label", type=str, default="CommonVoice 22 English",
                   help="Dataset name shown in the plot title (e.g. 'L2-ARCTIC (non-native English)').")
    return p.parse_args()


def main():
    args = parse_args()
    records = build_records(args.in_dir, args.total_layers)
    print(f"[Aggregate] Loaded {len(records)} depths from {args.in_dir}")

    write_findings_csv(records, args.out_dir / "depth_sweep_findings.csv")
    print(f"[Aggregate] Wrote table CSV: {args.out_dir / 'depth_sweep_findings.csv'}")
    write_findings_md(records, args.out_dir / "depth_sweep_findings.md")
    print(f"[Aggregate] Wrote table MD:  {args.out_dir / 'depth_sweep_findings.md'}")
    make_plot(records, args.out_dir / "depth_sweep_plot.png", args.total_layers,
              dataset_label=args.dataset_label)

    # Console summary
    print(f"\n{'depth':>5} {'kept':>4} {'WER_all':>8} {'WER_m':>7} {'WER_f':>7} "
          f"{'gap_mf':>7} {'CER_all':>8}")
    for r in records:
        print(f"{r['depth']:>5d} {r['layers_kept']:>4d} "
              f"{r['wer_ALL']*100:>7.2f}% {r['wer_male']*100:>6.2f}% {r['wer_female']*100:>6.2f}% "
              f"{r['wer_gap_mf']*100:>+6.2f} {r['cer_ALL']*100:>7.2f}%")

    # Flag non-monotonic steps in aggregate WER — the headline methodological caveat.
    inversions = [(records[i - 1]["depth"], records[i]["depth"],
                   (records[i]["wer_ALL"] - records[i - 1]["wer_ALL"]) * 100)
                  for i in range(1, len(records))
                  if records[i]["wer_ALL"] < records[i - 1]["wer_ALL"]]
    if inversions:
        print("\n[Aggregate] Non-monotonic WER steps (more pruning -> lower WER):")
        for a, b, d in inversions:
            print(f"    d{a} -> d{b}: {d:+.2f} pts")


if __name__ == "__main__":
    main()
