"""
Step 6 — Compare unpruned vs 2L-pruned across all available seeds.

Inputs:
    - Per-utterance CSVs in results/per_utterance/{condition}_seed{N}.csv
    - Per-seed summary CSVs in results/per_seed_wer/{condition}_seed{N}.csv

Outputs:
    - results/final_comparison.csv       (per-(condition, gender) mean/SD/min/max)
    - results/disparity_metrics.csv      (gap, ratio, worst-group WER per condition)
    - results/paired_bootstrap.csv       (per-gender significance test)
    - results/comparison_plot.png        (per-gender bars with seed-to-seed SD)

Usage:
    python experiments/bias_pruning/scripts/compare_conditions.py
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bootstrap_ci import paired_bootstrap_diff  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "experiments/bias_pruning/results"
PER_SEED_DIR = RESULTS_DIR / "per_seed_wer"
PER_UTT_DIR = RESULTS_DIR / "per_utterance"

CONDITIONS = ["unpruned", "pruned_2L"]
GENDER_BUCKETS = ["male", "female", "other", "missing"]


def _parse_seed_from_name(path: Path) -> int | None:
    m = re.search(r"seed(\d+)", path.stem)
    return int(m.group(1)) if m else None


_NUMERIC_FIELDS = (
    "wer", "wer_ci_low", "wer_ci_high",
    "cer", "cer_ci_low", "cer_ci_high",
    "duration_hours",
)


def load_per_seed_summaries(per_seed_dir: Path) -> dict[str, dict[int, list[dict]]]:
    """Return: {condition: {seed: [rows...]}}"""
    out: dict[str, dict[int, list[dict]]] = {c: {} for c in CONDITIONS}
    if not per_seed_dir.exists():
        return out
    for f in sorted(per_seed_dir.glob("*.csv")):
        seed = _parse_seed_from_name(f)
        if seed is None:
            continue
        cond = None
        for c in CONDITIONS:
            if f.stem.startswith(c + "_"):
                cond = c
                break
        if cond is None:
            continue
        with open(f, "r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            r["n_utts"] = int(r["n_utts"])
            for k in _NUMERIC_FIELDS:
                v = r.get(k, "")
                r[k] = float(v) if v not in ("", None) else float("nan")
        out[cond][seed] = rows
    return out


def _agg_stats(arr: np.ndarray) -> dict:
    return {
        "mean": float(np.nanmean(arr)),
        "sd": float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
    }


def aggregate_across_seeds(summaries: dict[str, dict[int, list[dict]]]) -> list[dict]:
    """For each (condition, gender) compute mean/SD/min/max across seeds for both WER and CER.

    Uses per-seed point estimates and counts utterances from the first seed
    (cell sizes are identical across seeds when the test split is fixed).
    """
    out: list[dict] = []
    for cond in CONDITIONS:
        per_seed = summaries.get(cond, {})
        if not per_seed:
            continue
        seeds = sorted(per_seed.keys())
        wer_by_gender: dict[str, list[float]] = defaultdict(list)
        cer_by_gender: dict[str, list[float]] = defaultdict(list)
        n_utts_by_gender: dict[str, int] = {}
        hours_by_gender: dict[str, float] = {}
        for s in seeds:
            for r in per_seed[s]:
                g = r["gender"]
                wer_by_gender[g].append(r["wer"])
                cer_by_gender[g].append(r["cer"])
                n_utts_by_gender.setdefault(g, r["n_utts"])
                hours_by_gender.setdefault(g, r["duration_hours"])
        for g in wer_by_gender:
            w = _agg_stats(np.array(wer_by_gender[g], dtype=np.float64))
            c = _agg_stats(np.array(cer_by_gender[g], dtype=np.float64))
            out.append({
                "condition": cond, "gender": g,
                "n_seeds": len(wer_by_gender[g]),
                "n_utts": n_utts_by_gender[g],
                "duration_hours": hours_by_gender[g],
                "mean_wer": w["mean"], "sd_wer": w["sd"], "min_wer": w["min"], "max_wer": w["max"],
                "mean_cer": c["mean"], "sd_cer": c["sd"], "min_cer": c["min"], "max_cer": c["max"],
            })
    return out


def disparity_metrics(rows: list[dict]) -> list[dict]:
    """Gap = worst_gender_rate - best_gender_rate, Ratio = worst/best, Worst = worst_gender_rate.
    Computed per seed for both WER and CER, restricted to gender in {male, female}
    (which are the analysable cells in CV22 English; expand if 'other' passes
    the threshold in future). Then mean and SD across seeds.
    """
    # Re-key: condition -> seed -> gender -> (wer, cer)
    raw: dict = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        raw[r["condition"]][r["seed"]][r["gender"]] = (r["wer"], r["cer"])

    def _disp(values):
        best, worst = min(values), max(values)
        return worst - best, (worst / best) if best > 0 else float("nan"), worst

    out = []
    for cond in CONDITIONS:
        per_seed = raw.get(cond, {})
        if not per_seed:
            continue
        w_gaps, w_ratios, w_worst = [], [], []
        c_gaps, c_ratios, c_worst = [], [], []
        for seed, g_to_pair in per_seed.items():
            usable = {g: v for g, v in g_to_pair.items() if g in ("male", "female")
                      and not (np.isnan(v[0]) or np.isnan(v[1]))}
            if len(usable) < 2:
                continue
            wers = [v[0] for v in usable.values()]
            cers = [v[1] for v in usable.values()]
            g, r, w = _disp(wers)
            w_gaps.append(g); w_ratios.append(r); w_worst.append(w)
            g, r, w = _disp(cers)
            c_gaps.append(g); c_ratios.append(r); c_worst.append(w)

        def _ms(xs):
            arr = np.array(xs, dtype=np.float64) if xs else np.array([np.nan])
            return float(np.nanmean(arr)), (float(np.nanstd(arr, ddof=1)) if len(xs) > 1 else 0.0)

        mg_w, sg_w = _ms(w_gaps); mr_w, sr_w = _ms(w_ratios); mw_w, sw_w = _ms(w_worst)
        mg_c, sg_c = _ms(c_gaps); mr_c, sr_c = _ms(c_ratios); mw_c, sw_c = _ms(c_worst)
        out.append({
            "condition": cond, "n_seeds": len(w_gaps),
            "mean_wer_gap": mg_w, "sd_wer_gap": sg_w,
            "mean_wer_ratio": mr_w, "sd_wer_ratio": sr_w,
            "mean_wer_worst": mw_w, "sd_wer_worst": sw_w,
            "mean_cer_gap": mg_c, "sd_cer_gap": sg_c,
            "mean_cer_ratio": mr_c, "sd_cer_ratio": sr_c,
            "mean_cer_worst": mw_c, "sd_cer_worst": sw_c,
        })
    return out


def _per_seed_rows_long(summaries: dict[str, dict[int, list[dict]]]) -> list[dict]:
    """Flatten per-seed summaries into a single list with one row per (cond, seed, gender)."""
    out = []
    for cond in CONDITIONS:
        for seed, rs in summaries.get(cond, {}).items():
            for r in rs:
                out.append({
                    "condition": cond, "seed": seed, "gender": r["gender"],
                    "wer": r["wer"], "cer": r["cer"],
                })
    return out


# ---------------------------------------------------------------------------
# Paired bootstrap (per gender)
# ---------------------------------------------------------------------------

def load_per_utterance(per_utt_dir: Path) -> dict[str, dict[int, list[dict]]]:
    out: dict[str, dict[int, list[dict]]] = {c: {} for c in CONDITIONS}
    if not per_utt_dir.exists():
        return out
    for f in sorted(per_utt_dir.glob("*.csv")):
        seed = _parse_seed_from_name(f)
        if seed is None:
            continue
        cond = None
        for c in CONDITIONS:
            if f.stem.startswith(c + "_"):
                cond = c
                break
        if cond is None:
            continue
        with open(f, "r", encoding="utf-8") as fh:
            out[cond][seed] = list(csv.DictReader(fh))
    return out


def paired_bootstrap_per_gender(
    per_utt: dict[str, dict[int, list[dict]]],
    n_bootstrap: int = 1000,
) -> list[dict]:
    """For each gender that appears in both conditions for at least one shared seed,
    align per-utterance rows by `key` (the dataset's per-utterance identifier),
    then run a paired bootstrap between the two conditions.
    """
    out = []
    if "unpruned" not in per_utt or "pruned_2L" not in per_utt:
        return out
    shared_seeds = set(per_utt["unpruned"]).intersection(per_utt["pruned_2L"])
    if not shared_seeds:
        return out

    for seed in sorted(shared_seeds):
        rows_a = per_utt["unpruned"][seed]
        rows_b = per_utt["pruned_2L"][seed]
        # `key` in the per-utterance CSV is the dataset's speaker_id (non-unique
        # because one speaker contributes many utterances). `row_idx` is the
        # HF dataset position and is unique per utterance, identical across
        # conditions because valid_indices is deterministic.
        index_a = {r["row_idx"]: r for r in rows_a}
        index_b = {r["row_idx"]: r for r in rows_b}
        shared_keys = set(index_a) & set(index_b)
        per_gender_keys = defaultdict(list)
        for k in shared_keys:
            g = index_a[k].get("gender", "missing")
            per_gender_keys[g].append(k)

        for g, keys in per_gender_keys.items():
            if g not in GENDER_BUCKETS:
                continue
            if len(keys) < 30:
                continue
            refs = [index_a[k]["reference"] for k in keys]
            ha = [index_a[k]["hypothesis"] for k in keys]
            hb = [index_b[k]["hypothesis"] for k in keys]
            res_w = paired_bootstrap_diff(refs, ha, hb, unit="word",
                                          n_bootstrap=n_bootstrap, seed=42)
            res_c = paired_bootstrap_diff(refs, ha, hb, unit="char",
                                          n_bootstrap=n_bootstrap, seed=42)
            out.append({
                "seed": seed, "gender": g, "n_utts": len(keys),
                "wer_unpruned": res_w["rate_a"], "wer_pruned_2L": res_w["rate_b"],
                "wer_delta": res_w["delta"],
                "wer_ci_low": res_w["ci_low"], "wer_ci_high": res_w["ci_high"],
                "wer_p_value_two_sided": res_w["p_value"],
                "cer_unpruned": res_c["rate_a"], "cer_pruned_2L": res_c["rate_b"],
                "cer_delta": res_c["delta"],
                "cer_ci_low": res_c["ci_low"], "cer_ci_high": res_c["ci_high"],
                "cer_p_value_two_sided": res_c["p_value"],
            })
    return out


# ---------------------------------------------------------------------------
# CSV / plot
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def make_plot(comparison_rows: list[dict], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Compare] matplotlib not available; skipping plot.")
        return

    genders = [g for g in GENDER_BUCKETS if any(r["gender"] == g for r in comparison_rows)]
    if not genders:
        print("[Compare] No data to plot.")
        return

    by_cond_gender = {(r["condition"], r["gender"]): r for r in comparison_rows}
    width = 0.38
    x = np.arange(len(genders))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, label in zip(axes, ("wer", "cer"), ("WER (%)", "CER (%)")):
        for offset, cond in zip([-width / 2, width / 2], CONDITIONS):
            means = [by_cond_gender.get((cond, g), {}).get(f"mean_{metric}", float("nan")) for g in genders]
            sds = [by_cond_gender.get((cond, g), {}).get(f"sd_{metric}", 0.0) for g in genders]
            ax.bar(x + offset, [100 * m for m in means], width=width,
                   yerr=[100 * s for s in sds], capsize=4, label=cond)
        ax.set_xticks(x)
        ax.set_xticklabels(genders)
        ax.set_ylabel(label)
        ax.set_title(f"Per-gender {label.split()[0]}: unpruned vs 2L pruned")
        ax.legend()
    fig.suptitle("Error bars = SD across seeds (single-seed runs show 0)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Compare] Wrote plot: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Compare unpruned vs 2L-pruned across seeds.")
    p.add_argument("--per_seed_dir", type=Path, default=PER_SEED_DIR)
    p.add_argument("--per_utt_dir", type=Path, default=PER_UTT_DIR)
    p.add_argument("--out_dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--n_bootstrap", type=int, default=1000)
    return p.parse_args()


def main():
    args = parse_args()
    summaries = load_per_seed_summaries(args.per_seed_dir)
    seed_counts = {c: len(v) for c, v in summaries.items()}
    print(f"[Compare] per-seed summaries found: {seed_counts}")
    if not any(seed_counts.values()):
        raise SystemExit(f"No per-seed summaries found in {args.per_seed_dir}. Run Step 5 first.")

    comparison_rows = aggregate_across_seeds(summaries)
    write_csv(
        comparison_rows,
        args.out_dir / "final_comparison.csv",
        ["condition", "gender", "n_seeds", "n_utts", "duration_hours",
         "mean_wer", "sd_wer", "min_wer", "max_wer",
         "mean_cer", "sd_cer", "min_cer", "max_cer"],
    )
    print(f"[Compare] Wrote final comparison: {args.out_dir/'final_comparison.csv'}")

    long_rows = _per_seed_rows_long(summaries)
    disp = disparity_metrics(long_rows)
    write_csv(
        disp,
        args.out_dir / "disparity_metrics.csv",
        ["condition", "n_seeds",
         "mean_wer_gap", "sd_wer_gap", "mean_wer_ratio", "sd_wer_ratio",
         "mean_wer_worst", "sd_wer_worst",
         "mean_cer_gap", "sd_cer_gap", "mean_cer_ratio", "sd_cer_ratio",
         "mean_cer_worst", "sd_cer_worst"],
    )
    print(f"[Compare] Wrote disparity metrics: {args.out_dir/'disparity_metrics.csv'}")

    per_utt = load_per_utterance(args.per_utt_dir)
    paired = paired_bootstrap_per_gender(per_utt, n_bootstrap=args.n_bootstrap)
    write_csv(
        paired,
        args.out_dir / "paired_bootstrap.csv",
        ["seed", "gender", "n_utts",
         "wer_unpruned", "wer_pruned_2L", "wer_delta",
         "wer_ci_low", "wer_ci_high", "wer_p_value_two_sided",
         "cer_unpruned", "cer_pruned_2L", "cer_delta",
         "cer_ci_low", "cer_ci_high", "cer_p_value_two_sided"],
    )
    print(f"[Compare] Wrote paired bootstrap test: {args.out_dir/'paired_bootstrap.csv'}")

    make_plot(comparison_rows, args.out_dir / "comparison_plot.png")

    # Pretty-print
    print("\n=== Per-(condition, gender) WER / CER across seeds ===")
    print(f"{'cond':<10} {'gender':<8} {'n_seeds':>4} {'n_utts':>7} "
          f"{'mean_WER':>9} {'SD':>6} {'mean_CER':>9} {'SD':>6}")
    for r in comparison_rows:
        print(f"{r['condition']:<10} {r['gender']:<8} {r['n_seeds']:>4d} {r['n_utts']:>7d} "
              f"{r['mean_wer']*100:>8.2f}% {r['sd_wer']*100:>5.2f}% "
              f"{r['mean_cer']*100:>8.2f}% {r['sd_cer']*100:>5.2f}%")
    print("\n=== Disparity metrics ===")
    print(f"{'cond':<10} {'WER gap':>15} {'WER worst':>15} {'CER gap':>15} {'CER worst':>15}")
    for r in disp:
        print(f"{r['condition']:<10} "
              f"{r['mean_wer_gap']*100:>9.2f}% +-{r['sd_wer_gap']*100:>4.2f}  "
              f"{r['mean_wer_worst']*100:>9.2f}% +-{r['sd_wer_worst']*100:>4.2f}  "
              f"{r['mean_cer_gap']*100:>9.2f}% +-{r['sd_cer_gap']*100:>4.2f}  "
              f"{r['mean_cer_worst']*100:>9.2f}% +-{r['sd_cer_worst']*100:>4.2f}")


if __name__ == "__main__":
    main()
