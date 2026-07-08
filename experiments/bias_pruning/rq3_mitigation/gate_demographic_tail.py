"""
RQ3 demographic go/no-go gate (zero GPU) — "does the utterance tail
over-represent the worst-served group?"

Batch-CVaR is label-free: it can only help a demographic group if that group
is over-represented among the worst utterances. This script checks that on
REAL per-utterance Fair-Speech results (unpruned vs pruned), two ways:

  A. damage tail  — utterances ranked by NEW errors (pruned - unpruned):
                    who bears the damage the recovery must undo?
  B. level tail   — utterances ranked by pruned per-utterance WER:
                    who does a tail objective see during recovery?

GO  -> the focus group's tail over-representation ratio is > 1 with CI support:
       label-free tail recovery plausibly transfers to the demographic gap.
NO-GO -> damage is demographically diffuse: pivot per the plan (the null is
       itself a paper finding).

Run (server, or laptop after scp'ing the per-utterance CSVs):
    python experiments/bias_pruning/rq3_mitigation/gate_demographic_tail.py \
        --unpruned <fairspeech per-utt CSV, unpruned> \
        --pruned   <fairspeech per-utt CSV, keep-30> \
        --axis ethnicity --focus "black or african american"
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "bias_pruning" / "scripts"))

from bootstrap_ci import _per_utt_stats  # repo-canonical jiwer normalization


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["key"]: r for r in csv.DictReader(f)}


def tail_stats(scores, groups, focus, frac, rng, n_bootstrap):
    """Over-representation of `focus` among the top-`frac` utterances by
    `scores` (higher = worse). Returns (overall_share, tail_share, ratio,
    ci_low, ci_high)."""
    n = len(scores)
    k = max(1, int(np.ceil(frac * n)))
    is_focus = (groups == focus)

    def ratio_of(idx):
        s, g = scores[idx], is_focus[idx]
        order = np.argsort(-s, kind="stable")
        tail = g[order[:k]]
        overall = g.mean()
        return (tail.mean() / overall) if overall > 0 else np.nan

    all_idx = np.arange(n)
    point = ratio_of(all_idx)
    boots = np.array([ratio_of(rng.integers(0, n, size=n)) for _ in range(n_bootstrap)])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    order = np.argsort(-scores, kind="stable")
    return is_focus.mean(), is_focus[order[:k]].mean(), point, lo, hi


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unpruned", required=True)
    ap.add_argument("--pruned", required=True)
    ap.add_argument("--axis", default="ethnicity", help="demographic column (ethnicity, ses, ...)")
    ap.add_argument("--focus", default="black or african american",
                    help="worst-served group value (as it appears in the CSV, lowercased)")
    ap.add_argument("--tail_frac", type=float, default=0.10)
    ap.add_argument("--min_ratio", type=float, default=1.2,
                    help="GO requires point ratio >= this AND CI low > 1.0")
    ap.add_argument("--n_bootstrap", type=int, default=1000)
    args = ap.parse_args()

    base = load_rows(args.unpruned)
    prun = load_rows(args.pruned)
    keys = [k for k in sorted(set(base) & set(prun))
            if base[k]["reference"] == prun[k]["reference"]]
    print(f"Paired utterances: {len(keys)}")

    refs = [base[k]["reference"] for k in keys]
    err_b, n_tok = _per_utt_stats(refs, [base[k]["hypothesis"] for k in keys], unit="word")
    err_p, _ = _per_utt_stats(refs, [prun[k]["hypothesis"] for k in keys], unit="word")
    words = np.maximum(n_tok, 1)

    groups = np.array([(base[k].get(args.axis) or "missing").strip().lower() for k in keys])
    known = groups != "missing"
    print(f"Utterances with known {args.axis}: {known.sum()} ({known.mean():.1%})")
    if known.sum() < 200:
        sys.exit(f"Too few labelled utterances for a reliable gate on axis {args.axis!r}.")

    # restrict to labelled rows: composition ratios need labels on both sides
    damage = ((err_p - err_b) / words)[known]
    level = (err_p / words)[known]
    g = groups[known]

    focus = args.focus.strip().lower()
    if focus not in set(g):
        sys.exit(f"Focus group {focus!r} not found in axis {args.axis!r}. "
                 f"Present: {sorted(set(g))}")

    # context: group WERs both conditions
    print(f"\nGroup corpus WER ({args.axis}):")
    for grp in sorted(set(g)):
        m = g == grp
        wb = err_b[known][m].sum() / max(n_tok[known][m].sum(), 1)
        wp = err_p[known][m].sum() / max(n_tok[known][m].sum(), 1)
        mark = "  <- focus" if grp == focus else ""
        print(f"   {grp:<38} n={m.sum():>5}  unpruned {wb:.4f} -> pruned {wp:.4f} "
              f"({wp-wb:+.4f}){mark}")

    rng = np.random.default_rng(42)
    print(f"\nTail composition (worst {args.tail_frac:.0%} of labelled utterances):")
    verdicts = {}
    for name, scores in [("A. damage tail (new errors)", damage),
                         ("B. level tail (pruned WER)", level)]:
        overall, tail, ratio, lo, hi = tail_stats(
            scores, g, focus, args.tail_frac, rng, args.n_bootstrap)
        verdicts[name] = (ratio, lo)
        print(f"   {name:<30} focus share {overall:.1%} overall -> {tail:.1%} in tail; "
              f"over-representation {ratio:.2f}x [95% CI {lo:.2f}, {hi:.2f}]")

    d_ratio, d_lo = verdicts["A. damage tail (new errors)"]
    l_ratio, l_lo = verdicts["B. level tail (pruned WER)"]
    go = d_ratio >= args.min_ratio and d_lo > 1.0
    weak_go = (not go) and l_ratio >= args.min_ratio and l_lo > 1.0

    print()
    if go:
        print(f"GATE: GO — the damage tail over-represents {focus!r} "
              f"({d_ratio:.2f}x, CI low {d_lo:.2f}). A label-free tail objective")
        print("      sees this group disproportionately; CVaR recovery plausibly transfers.")
    elif weak_go:
        print(f"GATE: WEAK GO — level tail over-represents the group ({l_ratio:.2f}x) but the")
        print("      damage tail does not. CVaR targets who is worst-served rather than who")
        print("      the prune newly harmed; expect a smaller transfer effect.")
    else:
        print(f"GATE: NO-GO — the tail does not over-represent {focus!r} "
              f"(damage {d_ratio:.2f}x, level {l_ratio:.2f}x).")
        print("      Damage is demographically diffuse; per the plan, pivot — and write up")
        print("      the null: label-free reweighting cannot transfer without labels.")


if __name__ == "__main__":
    main()
