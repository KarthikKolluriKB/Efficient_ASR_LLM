"""
Aggregate a depth sweep of MULTI-AXIS per-group summaries (gender + age + SES +
ethnicity + ...) into one findings table and one depth-vs-error plot PER AXIS.

This is the multi-axis sibling of aggregate_depth_sweep.py. That script reads the
legacy gender-only summary files (one `gender` column); this one reads the long
`*_multiaxis.csv` files written by evaluate_subgroup_wer.multiaxis_summary, which
carry an `axis` + `group` pair so any demographic axis can be sliced out.

Input: a directory of per-depth multi-axis CSVs named d{NN}*.csv, each with:
    condition, seed, axis, group, n_utts, duration_hours,
    wer, wer_ci_low, wer_ci_high, cer, cer_ci_low, cer_ci_high, analysable

Outputs (under --out_dir), for each requested --axis:
    findings_{axis}.csv   long: one row per (depth, group)
    findings_{axis}.md    wide WER + CER tables (rows = depth, cols = group)
    plot_{axis}.png       depth-vs-WER and depth-vs-CER, one line per group

Usage:
    python experiments/bias_pruning/scripts/aggregate_multiaxis_sweep.py \
        --in_dir  experiments/bias_pruning/results/fairspeech_sweep/per_axis \
        --out_dir experiments/bias_pruning/results/fairspeech_sweep \
        --total_layers 12
    # restrict to specific axes:
    python .../aggregate_multiaxis_sweep.py --axes gender ses ethnicity
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IN = PROJECT_ROOT / "experiments/bias_pruning/results/fairspeech_sweep/per_axis"
DEFAULT_OUT = PROJECT_ROOT / "experiments/bias_pruning/results/fairspeech_sweep"

_NUMERIC_FIELDS = (
    "wer", "wer_ci_low", "wer_ci_high",
    "cer", "cer_ci_low", "cer_ci_high", "duration_hours",
)
# Plot ALL last and in a distinct style; gender buckets get a stable order.
_GENDER_ORDER = ["male", "female", "other", "missing"]


def parse_depth(path: Path) -> int:
    m = re.match(r"d0*(\d+)", path.stem)
    if not m:
        raise ValueError(f"Cannot parse depth from filename: {path.name!r}")
    return int(m.group(1))


def load_multiaxis_file(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "axis" not in reader.fieldnames:
            raise SystemExit(
                f"{path.name} is not a multi-axis file (no 'axis' column). "
                "Use aggregate_depth_sweep.py for legacy gender-only files."
            )
        rows = list(reader)
    for r in rows:
        r["n_utts"] = int(r["n_utts"])
        for k in _NUMERIC_FIELDS:
            v = r.get(k, "")
            r[k] = float(v) if v not in ("", None) else float("nan")
    return rows


def load_sweep(in_dir: Path) -> tuple[dict[int, list[dict]], list[str]]:
    """Return ({depth: rows}, sorted_axes_present)."""
    files = sorted(in_dir.glob("d*.csv"), key=parse_depth)
    if not files:
        raise SystemExit(f"No d*.csv multi-axis files found in {in_dir}")
    by_depth: dict[int, list[dict]] = {}
    axes: set[str] = set()
    for f in files:
        rows = load_multiaxis_file(f)
        by_depth[parse_depth(f)] = rows
        axes.update(r["axis"] for r in rows)
    return by_depth, sorted(axes)


def _group_order(groups: list[str], axis: str) -> list[str]:
    non_all = [g for g in groups if g != "ALL"]
    if axis == "gender":
        ordered = [g for g in _GENDER_ORDER if g in non_all] + \
                  sorted(g for g in non_all if g not in _GENDER_ORDER)
    else:
        ordered = sorted(non_all)
    if "ALL" in groups:
        ordered.append("ALL")
    return ordered


def axis_records(by_depth: dict[int, list[dict]], axis: str, total_layers: int):
    """Return (depths, ordered_groups, table) where
    table[depth][group] = metric row. Groups never analysable in any depth are
    kept in the long CSV but dropped from the plot/wide-table for legibility.
    """
    depths = sorted(by_depth.keys())
    table: dict[int, dict[str, dict]] = {}
    all_groups: set[str] = set()
    analysable_groups: set[str] = set()
    for d in depths:
        cells = {r["group"]: r for r in by_depth[d] if r["axis"] == axis}
        table[d] = cells
        for g, r in cells.items():
            all_groups.add(g)
            if r.get("analysable") == "yes" or g == "ALL":
                analysable_groups.add(g)
    ordered_all = _group_order(sorted(all_groups), axis)
    ordered_plot = _group_order(sorted(analysable_groups), axis)
    return depths, ordered_all, ordered_plot, table


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

_LONG_FIELDS = ["depth", "layers_kept", "axis", "group", "n_utts",
                "wer", "wer_ci_low", "wer_ci_high",
                "cer", "cer_ci_low", "cer_ci_high", "analysable"]


def write_long_csv(depths, ordered_groups, table, axis, total_layers, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_LONG_FIELDS)
        w.writeheader()
        for d in depths:
            for g in ordered_groups:
                r = table[d].get(g)
                if r is None:
                    continue
                w.writerow({
                    "depth": d, "layers_kept": total_layers - d, "axis": axis,
                    "group": g, "n_utts": r["n_utts"],
                    "wer": r["wer"], "wer_ci_low": r["wer_ci_low"], "wer_ci_high": r["wer_ci_high"],
                    "cer": r["cer"], "cer_ci_low": r["cer_ci_low"], "cer_ci_high": r["cer_ci_high"],
                    "analysable": r.get("analysable", ""),
                })


def write_wide_md(depths, ordered_plot, table, axis, total_layers, path: Path):
    def pct(r, metric):
        if r is None:
            return "—"
        v = r[metric]
        return f"{v * 100:.2f}" if v == v else "—"  # nan check

    def section(metric, label):
        head = "| depth | kept | " + " | ".join(ordered_plot) + " |"
        sep = "|---:|---:|" + "|".join(["---:"] * len(ordered_plot)) + "|"
        lines = [f"### {label} by `{axis}`", "", head, sep]
        for d in depths:
            cells = " | ".join(pct(table[d].get(g), metric) for g in ordered_plot)
            lines.append(f"| {d} | {total_layers - d} | {cells} |")
        return "\n".join(lines)

    # under-powered note: groups present but not in plot set
    md = [f"# Multi-axis depth sweep — `{axis}`", "",
          "WER / CER as %. Columns are analysable groups (≥200 utts and ≥30 min); "
          "under-powered groups are in the long CSV only.", "",
          section("wer", "WER"), "", section("cer", "CER"), ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


def make_plot(depths, ordered_plot, table, axis, total_layers, path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Multiaxis] matplotlib not available; skipping plot.")
        return

    cmap = plt.get_cmap("tab10")
    colors = {g: ("#000000" if g == "ALL" else cmap(i % 10))
              for i, g in enumerate([g for g in ordered_plot if g != "ALL"])}
    colors["ALL"] = "#000000"

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, metric, label in zip(axes, ("wer", "cer"), ("WER (%)", "CER (%)")):
        for g in ordered_plot:
            ys, xs, los, his = [], [], [], []
            for d in depths:
                r = table[d].get(g)
                if r is None or r[metric] != r[metric]:  # missing/nan
                    continue
                xs.append(d)
                ys.append(100 * r[metric])
                los.append(100 * r[f"{metric}_ci_low"])
                his.append(100 * r[f"{metric}_ci_high"])
            if not xs:
                continue
            lw = 2.6 if g == "ALL" else 1.7
            ls = "-" if g != "ALL" else "-"
            ax.plot(xs, ys, marker="o", ms=4, lw=lw, ls=ls, color=colors[g],
                    label=g, zorder=3)
            if g != "ALL":
                ax.fill_between(xs, los, his, color=colors[g], alpha=0.10, zorder=1)
        if metric == "wer":
            ax.axhline(100, color="grey", ls=":", lw=1, zorder=2)
        ax.set_xticks(depths)
        ax.set_xticklabels([f"{d}\n{total_layers - d}L" for d in depths], fontsize=8)
        ax.set_xlabel("prune depth (layers removed) / layers kept")
        ax.set_ylabel(label)
        ax.set_title(f"{label.split()[0]} vs depth by {axis}")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(title=axis, fontsize=8, ncol=2)
    fig.suptitle(f"Fair-Speech depth sweep — WER/CER by {axis} (seed 42, one checkpoint per depth)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Multiaxis] Wrote plot: {path}")


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate a multi-axis depth sweep per axis.")
    p.add_argument("--in_dir", type=Path, default=DEFAULT_IN,
                   help="Directory of per-depth *_multiaxis.csv files (named d{NN}*.csv).")
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--axes", nargs="+", default=None,
                   help="Axes to aggregate (default: all axes present in the files).")
    p.add_argument("--total_layers", type=int, default=12)
    return p.parse_args()


def main():
    args = parse_args()
    by_depth, axes_present = load_sweep(args.in_dir)
    axes = args.axes or axes_present
    unknown = [a for a in axes if a not in axes_present]
    if unknown:
        print(f"[Multiaxis] WARNING: requested axes not in data: {unknown}. "
              f"Available: {axes_present}")
    axes = [a for a in axes if a in axes_present]
    print(f"[Multiaxis] Loaded {len(by_depth)} depths from {args.in_dir}; axes={axes}")

    for axis in axes:
        depths, ordered_all, ordered_plot, table = axis_records(
            by_depth, axis, args.total_layers)
        write_long_csv(depths, ordered_all, table, axis, args.total_layers,
                       args.out_dir / f"findings_{axis}.csv")
        write_wide_md(depths, ordered_plot, table, axis, args.total_layers,
                      args.out_dir / f"findings_{axis}.md")
        make_plot(depths, ordered_plot, table, axis, args.total_layers,
                  args.out_dir / f"plot_{axis}.png")
        dropped = [g for g in ordered_all if g not in ordered_plot]
        print(f"[Multiaxis] axis={axis}: {len(ordered_plot)} plotted group(s) "
              f"{ordered_plot}" + (f"; under-powered (CSV only): {dropped}" if dropped else ""))


if __name__ == "__main__":
    main()
