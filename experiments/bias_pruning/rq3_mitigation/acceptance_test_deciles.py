"""
De-averaged acceptance test on REAL per-utterance results — RQ3 demo, part 3.
Zero GPU: runs on the committed large-v2 CV22 seed-42 CSVs.

De-averaging point 3 of the plan: replace "the prune is free if the AGGREGATE
WER holds" with "the prune is free only if no utterance-decile regresses" —
the same tolerance the aggregate test uses, just without the averaging.

Also computes the damage-concentration profile (what share of the new errors
comes from the worst 1/5/10% of utterances), which is the free go/no-go signal
for whether a label-free tail loss (batch-CVaR) can target the harm at all.

Run from repo root:
    python experiments/bias_pruning/rq3_mitigation/acceptance_test_deciles.py
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SCRIPTS_DIR = REPO_ROOT / "experiments" / "bias_pruning" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from bootstrap_ci import _per_utt_stats  # repo-canonical jiwer normalization

RESULTS_DIR = REPO_ROOT / "experiments" / "bias_pruning" / "results"
DEFAULT_UNPRUNED = RESULTS_DIR / "per_utterance" / "unpruned_seed42.csv"
DEFAULT_PRUNED = RESULTS_DIR / "per_utterance" / "pruned_2L_seed42.csv"

DECILES = [10, 20, 30, 40, 50, 60, 70, 80, 90]
TAILS = [0.10, 0.20]  # CVaR-style worst-fraction means


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["key"]: r for r in csv.DictReader(f)}


def corpus_wer(errors, words):
    return errors.sum() / max(words.sum(), 1)


def tail_mean(per_utt_wer, frac):
    """Mean per-utterance WER of the worst `frac` of utterances."""
    k = max(1, int(np.ceil(frac * len(per_utt_wer))))
    return float(np.sort(per_utt_wer)[-k:].mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unpruned", default=str(DEFAULT_UNPRUNED))
    ap.add_argument("--pruned", default=str(DEFAULT_PRUNED))
    ap.add_argument("--pruned_label", default="large-v2 keep-30 (pruned_2L)")
    ap.add_argument("--tol_abs", type=float, default=0.005,
                    help="absolute WER tolerance (default 0.5pp)")
    ap.add_argument("--tol_rel", type=float, default=0.05,
                    help="relative tolerance vs baseline (default 5%%)")
    ap.add_argument("--n_bootstrap", type=int, default=1000)
    args = ap.parse_args()

    base_rows = load_rows(args.unpruned)
    prun_rows = load_rows(args.pruned)
    keys = sorted(set(base_rows) & set(prun_rows))
    # paired: identical reference text required
    keys = [k for k in keys if base_rows[k]["reference"] == prun_rows[k]["reference"]]
    print(f"Paired utterances: {len(keys)} "
          f"(unpruned file: {len(base_rows)}, pruned file: {len(prun_rows)})")

    refs = [base_rows[k]["reference"] for k in keys]
    hyp_base = [base_rows[k]["hypothesis"] for k in keys]
    hyp_prun = [prun_rows[k]["hypothesis"] for k in keys]

    print("Scoring per-utterance WER (repo-canonical jiwer normalization) ...")
    err_b, n_b = _per_utt_stats(refs, hyp_base, unit="word")
    err_p, n_p = _per_utt_stats(refs, hyp_prun, unit="word")
    assert np.array_equal(n_b, n_p), "reference token counts must match (paired data)"
    words = np.maximum(n_b, 1)

    wer_b = err_b / words
    wer_p = err_p / words

    agg_b = corpus_wer(err_b, n_b)
    agg_p = corpus_wer(err_p, n_p)
    tol = max(args.tol_abs, args.tol_rel * agg_b)

    # ----------------------------------------------------------- aggregate test
    print()
    print("=" * 74)
    print(f"1) STATUS-QUO ACCEPTANCE TEST (the average decides)")
    print("=" * 74)
    agg_delta = agg_p - agg_b
    agg_free = agg_delta <= tol
    print(f"   aggregate WER: {agg_b:.4f} -> {agg_p:.4f}  (delta +{agg_delta:.4f}, "
          f"tolerance {tol:.4f})")
    print(f"   verdict: prune is {'FREE (accepted)' if agg_free else 'NOT free (rejected)'}")

    # -------------------------------------------------------- de-averaged test
    print()
    print("=" * 74)
    print("2) DE-AVERAGED ACCEPTANCE TEST (no utterance-decile may regress)")
    print("=" * 74)
    print(f"   same tolerance ({tol:.4f}), applied per decile of the per-utterance")
    print(f"   WER distribution instead of to the average only.")
    print()
    print(f"   {'metric':<28} {'unpruned':>10} {'pruned':>10} {'delta':>9} "
          f"{'95% CI (delta)':>18}  flag")

    rng = np.random.default_rng(42)
    n = len(keys)
    boot_idx = [rng.integers(0, n, size=n) for _ in range(args.n_bootstrap)]

    def report(name, stat_fn):
        s_b, s_p = stat_fn(wer_b), stat_fn(wer_p)
        delta = s_p - s_b
        boots = np.array([stat_fn(wer_p[i]) - stat_fn(wer_b[i]) for i in boot_idx])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        regress = delta > tol and lo > 0
        flag = "REGRESSES" if regress else ("ok" if delta <= tol else "regress? (CI~0)")
        print(f"   {name:<28} {s_b:>9.4f} {s_p:>10.4f} {delta:>+9.4f} "
              f"{'[%+.4f, %+.4f]' % (lo, hi):>18}  {flag}")
        return regress

    regressions = []
    for d in DECILES:
        r = report(f"utterance-WER p{d}", lambda w, d=d: float(np.percentile(w, d)))
        if r:
            regressions.append(f"p{d}")
    for frac in TAILS:
        r = report(f"tail mean (worst {frac:.0%})", lambda w, f=frac: tail_mean(w, f))
        if r:
            regressions.append(f"tail{frac:.0%}")

    deavg_free = not regressions
    print()
    print(f"   verdict: prune is "
          f"{'FREE (no decile regresses)' if deavg_free else 'NOT free (rejected)'}"
          + (f" — regressions at: {', '.join(regressions)}" if regressions else ""))

    # ------------------------------------------------- damage concentration
    print()
    print("=" * 74)
    print("3) DAMAGE CONCENTRATION (go/no-go signal for a label-free tail loss)")
    print("=" * 74)
    new_err = (err_p - err_b).astype(float)
    total_new = new_err[new_err > 0].sum()
    order = np.argsort(-new_err)
    print(f"   utterances made worse: {(new_err > 0).sum()} "
          f"({(new_err > 0).mean():.1%}); improved: {(new_err < 0).sum()} "
          f"({(new_err < 0).mean():.1%}); unchanged: {(new_err == 0).mean():.1%}")
    for frac in (0.01, 0.05, 0.10, 0.20):
        k = int(np.ceil(frac * n))
        share = new_err[order[:k]].sum() / max(total_new, 1)
        print(f"   worst {frac:>4.0%} of utterances carry {share:>6.1%} of all new errors")
    print()
    print("   (Damage concentrated in a small utterance tail -> a label-free")
    print("    tail objective can see and target it. The demographic version of")
    print("    this gate — does the tail over-represent Black speakers? — needs")
    print("    the Fair-Speech per-utterance CSVs on the server.)")

    # ------------------------------------------------------------- summary
    print()
    print("=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"   condition: {args.pruned_label}")
    print(f"   status-quo aggregate test:  {'ACCEPTS the prune' if agg_free else 'rejects the prune'} "
          f"(delta +{agg_delta:.4f} vs tolerance {tol:.4f})")
    print(f"   de-averaged decile test:    {'accepts the prune' if deavg_free else 'REJECTS the prune'}")

    # The tolerance is a free parameter of any acceptance test. The decisive
    # comparison is therefore the BAND between the aggregate's regression and
    # the worst decile's regression: any tolerance inside it means "the average
    # blesses the prune while the tail regresses" — the hidden-harm outcome.
    worst_decile_delta = max(
        [float(np.percentile(wer_p, d) - np.percentile(wer_b, d)) for d in DECILES]
        + [tail_mean(wer_p, f) - tail_mean(wer_b, f) for f in TAILS]
    )
    if worst_decile_delta > agg_delta:
        print()
        print(f"   hidden-harm tolerance band: aggregate regresses +{agg_delta:.4f} "
              f"but the worst decile regresses +{worst_decile_delta:.4f}.")
        print(f"   -> ANY acceptance tolerance in (+{agg_delta:.4f}, +{worst_decile_delta:.4f}) "
              f"accepts this prune on the")
        print("      average while its utterance tail regresses several times harder —")
        print("      exactly the hidden harm RQ1 measured demographically, caught here")
        print("      with ZERO GPU cost and no demographic labels.")

    # write CSV artifact next to the other findings
    out_dir = RESULTS_DIR / "rq3_acceptance"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "acceptance_test_deciles.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "unpruned", "pruned", "delta"])
        w.writerow(["aggregate_wer", f"{agg_b:.6f}", f"{agg_p:.6f}", f"{agg_p-agg_b:+.6f}"])
        for d in DECILES:
            a, b = float(np.percentile(wer_b, d)), float(np.percentile(wer_p, d))
            w.writerow([f"utt_wer_p{d}", f"{a:.6f}", f"{b:.6f}", f"{b-a:+.6f}"])
        for frac in TAILS:
            a, b = tail_mean(wer_b, frac), tail_mean(wer_p, frac)
            w.writerow([f"tail_mean_{int(frac*100)}", f"{a:.6f}", f"{b:.6f}", f"{b-a:+.6f}"])
    print(f"\n   wrote {out_csv}")


if __name__ == "__main__":
    main()
