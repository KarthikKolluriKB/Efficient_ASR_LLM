"""
RQ3 success-bar checker — did De-Averaged Pruning actually work?

Consumes three Fair-Speech PER-UTTERANCE CSVs (same schema everywhere in the
pipeline) and scores the CVaR arm against the plan's success bar:

  "The prune stays free on the average AND for the worst-served group."

  1. aggregate held:   WER(cvar) <= WER(mean-CE arm) + tol
  2. group restored:   focusWER(cvar) <= focusWER(unpruned) + tol
  3. disparity <= baseline: ratio(cvar) <= ratio(unpruned) + tol_ratio

Also prints the mean-CE arm on the same criteria (expected: fails 2 and 3 —
that contrast IS the paper result).

Run:
    python experiments/bias_pruning/rq3_mitigation/check_rq3_success.py \
        --unpruned <fairspeech per-utt CSV, unpruned 32L> \
        --mean     <fairspeech per-utt CSV, keep-30 mean-CE recovery> \
        --cvar     <fairspeech per-utt CSV, keep-30 CVaR recovery> \
        --axis ethnicity --focus "black or african american" --reference_group "asian"
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


class Condition:
    def __init__(self, name, path, keys, refs):
        rows = load_rows(path)
        self.name = name
        hyps = [rows[k]["hypothesis"] for k in keys]
        self.err, self.n_tok = _per_utt_stats(refs, hyps, unit="word")

    def wer(self, mask=None):
        e = self.err if mask is None else self.err[mask]
        t = self.n_tok if mask is None else self.n_tok[mask]
        return e.sum() / max(t.sum(), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unpruned", required=True)
    ap.add_argument("--mean", required=True, help="keep-30 recovered with mean CE (status quo)")
    ap.add_argument("--cvar", required=True, help="keep-30 recovered with batch-CVaR + tail ckpt")
    ap.add_argument("--axis", default="ethnicity")
    ap.add_argument("--focus", default="black or african american")
    ap.add_argument("--reference_group", default="asian",
                    help="advantaged group for the disparity ratio (substring match allowed)")
    ap.add_argument("--tol", type=float, default=0.005, help="absolute WER tolerance (0.5pp)")
    ap.add_argument("--tol_ratio", type=float, default=0.05)
    args = ap.parse_args()

    base_rows = load_rows(args.unpruned)
    mean_rows = load_rows(args.mean)
    cvar_rows = load_rows(args.cvar)
    keys = [k for k in sorted(set(base_rows) & set(mean_rows) & set(cvar_rows))
            if base_rows[k]["reference"] == mean_rows[k]["reference"]
            == cvar_rows[k]["reference"]]
    refs = [base_rows[k]["reference"] for k in keys]
    print(f"Paired utterances across all three conditions: {len(keys)}")

    groups = np.array([(base_rows[k].get(args.axis) or "missing").strip().lower()
                       for k in keys])
    focus = args.focus.strip().lower()
    ref_grp = args.reference_group.strip().lower()
    focus_mask = groups == focus
    # exact match preferred; substring fallback for long labels like
    # "asian or pacific islander" (never matching the focus group)
    ref_mask = groups == ref_grp
    if ref_mask.sum() == 0:
        ref_mask = (np.array([ref_grp in g for g in groups])
                    & (groups != "missing") & ~focus_mask)
    if focus_mask.sum() == 0:
        sys.exit(f"Focus group {focus!r} not found. Present: {sorted(set(groups))}")
    if ref_mask.sum() == 0:
        sys.exit(f"Reference group ~{ref_grp!r} not found. Present: {sorted(set(groups))}")
    print(f"{args.axis}: focus {focus!r} n={focus_mask.sum()}, "
          f"reference ~{ref_grp!r} n={ref_mask.sum()}")

    conds = {
        "unpruned": Condition("unpruned", args.unpruned, keys, refs),
        "mean": Condition("mean", args.mean, keys, refs),
        "cvar": Condition("cvar", args.cvar, keys, refs),
    }

    print()
    print(f"{'condition':<26} {'aggregate':>10} {'focus WER':>10} {'ref WER':>9} {'ratio':>7}")
    print("-" * 68)
    stats = {}
    for label, c in conds.items():
        agg = c.wer()
        fw = c.wer(focus_mask)
        rw = c.wer(ref_mask)
        ratio = fw / max(rw, 1e-9)
        stats[label] = (agg, fw, rw, ratio)
        print(f"{label:<26} {agg:>10.4f} {fw:>10.4f} {rw:>9.4f} {ratio:>6.2f}x")

    def judge(arm):
        agg, fw, _, ratio = stats[arm]
        agg_u, fw_u, _, ratio_u = stats["unpruned"]
        agg_m = stats["mean"][0]
        c1 = agg <= agg_m + args.tol           # free on the average
        c2 = fw <= fw_u + args.tol             # free for the worst-served group
        c3 = ratio <= ratio_u + args.tol_ratio  # disparity back to baseline
        return c1, c2, c3

    print()
    for arm, title in [("mean", "keep-30 + mean-CE recovery [status quo]"),
                       ("cvar", "keep-30 + CVaR recovery    [de-averaged]")]:
        c1, c2, c3 = judge(arm)
        print(f"{title}")
        print(f"   1. aggregate held vs mean arm (+{args.tol:.3f}):    {'PASS' if c1 else 'FAIL'}")
        print(f"   2. focus-group WER <= unpruned (+{args.tol:.3f}):   {'PASS' if c2 else 'FAIL'}")
        print(f"   3. ratio <= unpruned baseline (+{args.tol_ratio:.2f}):     {'PASS' if c3 else 'FAIL'}")

    c1, c2, c3 = judge("cvar")
    print()
    if c1 and c2 and c3:
        print("RQ3 SUCCESS BAR: MET — the prune is free on the average AND for the")
        print("worst-served group, with a label-free change to the recovery recipe.")
    elif c1 and (c2 or c3):
        print("RQ3 SUCCESS BAR: PARTIAL — aggregate held and disparity moved the right")
        print("way but did not fully reach the bar. Report effect size + CI; consider")
        print("the alpha=0.1 probe or the DAR fallback per the plan.")
    else:
        print("RQ3 SUCCESS BAR: NOT MET — per the plan, the null is still publishable:")
        print("label-free tail recovery does not transfer to demographic repair.")


if __name__ == "__main__":
    main()
