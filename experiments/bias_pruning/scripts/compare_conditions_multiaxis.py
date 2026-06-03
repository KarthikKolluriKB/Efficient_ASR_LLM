"""
Multi-axis paired-bootstrap disparity test between two pruning conditions.

The fairness question this answers: when we prune, does each demographic group
degrade, by how much, and is the degradation significant — across gender, age,
SES, ethnicity (any axis carried in the per-utterance CSV)? This is the
generalisation of compare_conditions.py's per-gender paired bootstrap to
arbitrary axes, reading the demographic columns that evaluate_subgroup_wer.py
now persists per utterance.

For each (axis, group) present in BOTH conditions with enough shared utterances,
it aligns hypotheses by `row_idx` (identical reference set, so paired) and runs
bootstrap_ci.paired_bootstrap_diff for WER and CER. It also reports, per axis and
per condition, the between-group disparity gap (worst − best analysable group)
and how that gap shifts under pruning.

Inputs:
    Per-utterance CSVs in --per_utt_dir named {condition}_seed{seed}.csv, with
    columns row_idx, reference, hypothesis, and one column per demographic axis.

Outputs (under --out_dir):
    multiaxis_paired_{baseline}_vs_{pruned}.csv   per-(axis, group) paired delta
    multiaxis_gap_{baseline}_vs_{pruned}.csv       per-axis disparity gap shift

Usage:
    python experiments/bias_pruning/scripts/compare_conditions_multiaxis.py \
        --per_utt_dir experiments/bias_pruning/results/fairspeech_sweep/per_utterance \
        --baseline d00_keep12 --pruned d02_keep10 d04_keep08 --seed 42 \
        --axes gender age ses ethnicity
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bootstrap_ci import paired_bootstrap_diff  # noqa: E402

KNOWN_AXES = ["gender", "age", "accent", "l1", "ses", "ethnicity"]


def _load_per_utt(per_utt_dir: Path, condition: str, seed: int) -> dict[str, dict]:
    path = per_utt_dir / f"{condition}_seed{seed}.csv"
    if not path.exists():
        raise SystemExit(f"Per-utterance CSV not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["row_idx"]: r for r in rows}, (rows[0].keys() if rows else [])


def _axes_present(fieldnames, requested) -> list[str]:
    present = [a for a in KNOWN_AXES if a in fieldnames]
    if requested:
        missing = [a for a in requested if a not in present]
        if missing:
            print(f"[Compare-MA] WARNING: requested axes absent from CSV: {missing}. Present: {present}")
        return [a for a in requested if a in present]
    return present


def compare_pair(per_utt_dir: Path, baseline: str, pruned: str, seed: int,
                 axes: list[str], n_bootstrap: int, min_utts: int):
    base_idx, base_fields = _load_per_utt(per_utt_dir, baseline, seed)
    prun_idx, _ = _load_per_utt(per_utt_dir, pruned, seed)
    shared = set(base_idx) & set(prun_idx)
    if not shared:
        raise SystemExit(f"No shared row_idx between {baseline} and {pruned}.")
    axes = _axes_present(base_fields, axes)

    paired_rows: list[dict] = []
    gap_rows: list[dict] = []

    for axis in axes:
        # Bucket shared utterances by the baseline's demographic value.
        by_group: dict[str, list[str]] = defaultdict(list)
        for k in shared:
            by_group[(base_idx[k].get(axis) or "missing")].append(k)

        # Per-group paired delta (between conditions).
        group_rate_base: dict[str, float] = {}
        group_rate_prun: dict[str, float] = {}
        for group, keys in sorted(by_group.items()):
            if len(keys) < min_utts:
                continue
            refs = [base_idx[k]["reference"] for k in keys]
            ha = [base_idx[k]["hypothesis"] for k in keys]
            hb = [prun_idx[k]["hypothesis"] for k in keys]
            rw = paired_bootstrap_diff(refs, ha, hb, unit="word", n_bootstrap=n_bootstrap, seed=42)
            rc = paired_bootstrap_diff(refs, ha, hb, unit="char", n_bootstrap=n_bootstrap, seed=42)
            group_rate_base[group] = rw["rate_a"]
            group_rate_prun[group] = rw["rate_b"]
            paired_rows.append({
                "axis": axis, "group": group, "n_utts": len(keys),
                "wer_baseline": rw["rate_a"], "wer_pruned": rw["rate_b"], "wer_delta": rw["delta"],
                "wer_ci_low": rw["ci_low"], "wer_ci_high": rw["ci_high"], "wer_p": rw["p_value"],
                "cer_baseline": rc["rate_a"], "cer_pruned": rc["rate_b"], "cer_delta": rc["delta"],
                "cer_ci_low": rc["ci_low"], "cer_ci_high": rc["ci_high"], "cer_p": rc["p_value"],
            })

        # Disparity gap (worst − best analysable group) per condition + shift.
        if len(group_rate_base) >= 2:
            def _gap(rates):
                worst_g = max(rates, key=rates.get)
                best_g = min(rates, key=rates.get)
                return worst_g, best_g, rates[worst_g] - rates[best_g]
            wg_b, bg_b, gap_b = _gap(group_rate_base)
            wg_p, bg_p, gap_p = _gap(group_rate_prun)
            gap_rows.append({
                "axis": axis, "n_groups": len(group_rate_base),
                "worst_group_baseline": wg_b, "best_group_baseline": bg_b,
                "wer_gap_baseline": gap_b,
                "worst_group_pruned": wg_p, "best_group_pruned": bg_p,
                "wer_gap_pruned": gap_p,
                "wer_gap_shift": gap_p - gap_b,
            })

    return paired_rows, gap_rows


def _write_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


_PAIRED_FIELDS = ["axis", "group", "n_utts",
                  "wer_baseline", "wer_pruned", "wer_delta", "wer_ci_low", "wer_ci_high", "wer_p",
                  "cer_baseline", "cer_pruned", "cer_delta", "cer_ci_low", "cer_ci_high", "cer_p"]
_GAP_FIELDS = ["axis", "n_groups",
               "worst_group_baseline", "best_group_baseline", "wer_gap_baseline",
               "worst_group_pruned", "best_group_pruned", "wer_gap_pruned", "wer_gap_shift"]


def parse_args():
    p = argparse.ArgumentParser(description="Multi-axis paired-bootstrap disparity test between two conditions.")
    p.add_argument("--per_utt_dir", type=Path, required=True,
                   help="Directory of per-utterance CSVs named {condition}_seed{seed}.csv.")
    p.add_argument("--baseline", required=True, help="Baseline condition label (e.g. d00_keep12).")
    p.add_argument("--pruned", nargs="+", required=True,
                   help="One or more pruned condition labels to compare against the baseline.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--axes", nargs="+", default=None,
                   help="Axes to test (default: all demographic axes present in the CSV).")
    p.add_argument("--out_dir", type=Path,
                   default=PROJECT_ROOT / "experiments/bias_pruning/results")
    p.add_argument("--n_bootstrap", type=int, default=1000)
    p.add_argument("--min_utts", type=int, default=100,
                   help="Skip a group with fewer than this many shared utterances (power floor).")
    return p.parse_args()


def main():
    args = parse_args()
    for pruned in args.pruned:
        paired_rows, gap_rows = compare_pair(
            args.per_utt_dir, args.baseline, pruned, args.seed,
            args.axes, args.n_bootstrap, args.min_utts,
        )
        tag = f"{args.baseline}_vs_{pruned}"
        _write_csv(paired_rows, args.out_dir / f"multiaxis_paired_{tag}.csv", _PAIRED_FIELDS)
        _write_csv(gap_rows, args.out_dir / f"multiaxis_gap_{tag}.csv", _GAP_FIELDS)

        print(f"\n=== {args.baseline}  ->  {pruned}  (per-group paired delta, pruned-baseline) ===")
        print(f"{'axis':<10} {'group':<12} {'n':>6} {'WER_b':>7} {'WER_p':>7} "
              f"{'dWER':>7} {'95% CI':>18} {'p':>7}")
        for r in paired_rows:
            ci = f"[{r['wer_ci_low']*100:+.2f},{r['wer_ci_high']*100:+.2f}]"
            star = " *" if r["wer_p"] < 0.05 else ""
            print(f"{r['axis']:<10} {r['group']:<12} {r['n_utts']:>6d} "
                  f"{r['wer_baseline']*100:>6.2f}% {r['wer_pruned']*100:>6.2f}% "
                  f"{r['wer_delta']*100:>+6.2f} {ci:>18} {r['wer_p']:>7.3f}{star}")

        print(f"\n=== disparity gap shift (worst-best WER, {args.baseline} -> {pruned}) ===")
        print(f"{'axis':<10} {'gap_base':>9} {'gap_pruned':>11} {'dgap':>8}  worst(base->pruned)")
        for r in gap_rows:
            print(f"{r['axis']:<10} {r['wer_gap_baseline']*100:>8.2f}% "
                  f"{r['wer_gap_pruned']*100:>10.2f}% {r['wer_gap_shift']*100:>+7.2f}  "
                  f"{r['worst_group_baseline']} -> {r['worst_group_pruned']}")


if __name__ == "__main__":
    main()
