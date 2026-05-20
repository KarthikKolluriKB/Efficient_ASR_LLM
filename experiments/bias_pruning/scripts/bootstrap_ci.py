"""
Step 3 — Utterance-level bootstrap WER/CER with 95% confidence intervals.

Public API:
    wer_with_ci(refs, hyps, ...)          -> dict   (alias: metric_with_ci(unit="word"))
    cer_with_ci(refs, hyps, ...)          -> dict   (alias: metric_with_ci(unit="char"))
    paired_bootstrap_diff(refs, hyp_a, hyp_b, ..., unit="word"|"char") -> dict

Both functions sample utterance indices with replacement (same N as the
input) and re-aggregate the metric per resample. The point estimate is
computed from the full set, not from the mean of resamples.

Run `python bootstrap_ci.py --self_test` for the acceptance-criterion test:
the bootstrap point estimate must match `jiwer.wer()` / `jiwer.cer()`
exactly on the same input.
"""

from __future__ import annotations

import argparse
from typing import Callable, List, Optional, Sequence

import numpy as np

try:
    import jiwer
except ImportError as e:
    raise SystemExit("jiwer is required. Install it with `pip install jiwer`.") from e


# A WER-friendly word-level normalizer: lowercase, strip punctuation,
# collapse whitespace, split into words.
DEFAULT_WORD_TRANSFORM = jiwer.Compose([
    jiwer.SubstituteRegexes({
        r"[<\[][^>\]]*[>\]]": "",
        r"\(([^)]+?)\)": "",
    }),
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])

# Character-level normalizer for CER: same cleanup as words, but reduce to a
# list-of-list-of-characters instead of words.
DEFAULT_CHAR_TRANSFORM = jiwer.Compose([
    jiwer.SubstituteRegexes({
        r"[<\[][^>\]]*[>\]]": "",
        r"\(([^)]+?)\)": "",
    }),
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfChars(),
])

# Back-compat alias for older imports.
DEFAULT_TRANSFORM = DEFAULT_WORD_TRANSFORM


def _to_lists(seq) -> List[str]:
    if isinstance(seq, np.ndarray):
        return [str(x) for x in seq.tolist()]
    return list(seq)


def _per_utt_stats(
    references: Sequence[str],
    hypotheses: Sequence[str],
    unit: str = "word",
    transform=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-utterance (errors, n_ref_tokens) for either word- or character-level scoring.

    `unit` selects the granularity:
      - "word": uses jiwer.process_words and counts reference words.
      - "char": uses jiwer.process_characters and counts reference chars.

    The bootstrap then re-aggregates these counts cheaply per resample.
    """
    if unit not in {"word", "char"}:
        raise ValueError(f"unit must be 'word' or 'char', got {unit!r}")

    if transform is None:
        transform = DEFAULT_WORD_TRANSFORM if unit == "word" else DEFAULT_CHAR_TRANSFORM
    process = jiwer.process_words if unit == "word" else jiwer.process_characters

    refs = _to_lists(references)
    hyps = _to_lists(hypotheses)
    if len(refs) != len(hyps):
        raise ValueError(f"reference and hypothesis lengths differ: {len(refs)} vs {len(hyps)}")

    errors = np.zeros(len(refs), dtype=np.int64)
    n_tokens = np.zeros(len(refs), dtype=np.int64)
    for i, (r, h) in enumerate(zip(refs, hyps)):
        out = process(
            [r], [h],
            reference_transform=transform,
            hypothesis_transform=transform,
        )
        err = int(out.substitutions + out.deletions + out.insertions)
        n_ref = int(out.hits + out.substitutions + out.deletions)
        errors[i] = err
        n_tokens[i] = n_ref
    return errors, n_tokens


# Back-compat: older code/tests imported _per_utt_word_stats by name.
def _per_utt_word_stats(references, hypotheses, transform=DEFAULT_WORD_TRANSFORM):
    return _per_utt_stats(references, hypotheses, unit="word", transform=transform)


def _corpus_rate(errors: np.ndarray, n_tokens: np.ndarray) -> float:
    total = int(n_tokens.sum())
    if total == 0:
        return 0.0
    return float(errors.sum()) / float(total)


# Back-compat alias.
def _corpus_wer(errors, n_words):
    return _corpus_rate(errors, n_words)


def metric_with_ci(
    references: Sequence[str],
    hypotheses: Sequence[str],
    unit: str = "word",
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    transform=None,
) -> dict:
    """Point-estimate WER (unit='word') or CER (unit='char') plus bootstrap CI.

    Returns dict with: point, ci_low, ci_high, n_utts, n_bootstrap, alpha, unit.
    For backwards compatibility, the dict also has a `wer` (resp. `cer`) key
    aliasing `point`.
    """
    errors, n_tokens = _per_utt_stats(references, hypotheses, unit=unit, transform=transform)
    n = len(errors)
    point = _corpus_rate(errors, n_tokens)
    alias_key = "wer" if unit == "word" else "cer"

    if n == 0 or n_bootstrap <= 0:
        return {
            "point": point, alias_key: point,
            "ci_low": point, "ci_high": point,
            "n_utts": n, "n_bootstrap": 0, "alpha": alpha, "unit": unit,
        }

    rng = np.random.default_rng(seed)
    boots = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boots[i] = _corpus_rate(errors[idx], n_tokens[idx])

    lo = float(np.percentile(boots, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boots, 100.0 * (1.0 - alpha / 2.0)))
    return {
        "point": point, alias_key: point,
        "ci_low": lo, "ci_high": hi,
        "n_utts": n, "n_bootstrap": int(n_bootstrap), "alpha": alpha, "unit": unit,
    }


def wer_with_ci(references, hypotheses, n_bootstrap=1000, alpha=0.05, seed=42, transform=None):
    """Convenience wrapper: word-level metric_with_ci."""
    return metric_with_ci(references, hypotheses, unit="word",
                          n_bootstrap=n_bootstrap, alpha=alpha, seed=seed, transform=transform)


def cer_with_ci(references, hypotheses, n_bootstrap=1000, alpha=0.05, seed=42, transform=None):
    """Convenience wrapper: character-level metric_with_ci."""
    return metric_with_ci(references, hypotheses, unit="char",
                          n_bootstrap=n_bootstrap, alpha=alpha, seed=seed, transform=transform)


def paired_bootstrap_diff(
    references: Sequence[str],
    hypotheses_a: Sequence[str],
    hypotheses_b: Sequence[str],
    unit: str = "word",
    n_bootstrap: int = 1000,
    seed: int = 42,
    alternative: str = "two-sided",
    transform=None,
) -> dict:
    """Paired bootstrap test for the metric difference (B minus A) over the
    same utterance set. `unit` selects WER ('word') or CER ('char').

    Returns dict with: rate_a, rate_b, delta, ci_low, ci_high, p_value,
    n_utts, n_bootstrap, unit. Aliases `wer_a/wer_b` (resp. `cer_a/cer_b`)
    are also set for back-compat.
    """
    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError(f"alternative must be one of two-sided/less/greater, got {alternative!r}")

    err_a, n_a = _per_utt_stats(references, hypotheses_a, unit=unit, transform=transform)
    err_b, n_b = _per_utt_stats(references, hypotheses_b, unit=unit, transform=transform)
    if not np.array_equal(n_a, n_b):
        raise ValueError("n_ref_tokens differs between conditions; references must be identical.")

    n = len(err_a)
    alias_a = ("wer_a" if unit == "word" else "cer_a")
    alias_b = ("wer_b" if unit == "word" else "cer_b")
    if n == 0:
        return {
            "rate_a": 0.0, "rate_b": 0.0, alias_a: 0.0, alias_b: 0.0,
            "delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0,
            "n_utts": 0, "n_bootstrap": 0, "unit": unit,
        }

    point_a = _corpus_rate(err_a, n_a)
    point_b = _corpus_rate(err_b, n_b)
    point_delta = point_b - point_a

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        ra = _corpus_rate(err_a[idx], n_a[idx])
        rb = _corpus_rate(err_b[idx], n_b[idx])
        deltas[i] = rb - ra

    ci_low = float(np.percentile(deltas, 2.5))
    ci_high = float(np.percentile(deltas, 97.5))

    centred = deltas - point_delta
    if alternative == "two-sided":
        p = float(np.mean(np.abs(centred) >= abs(point_delta)))
    elif alternative == "greater":
        p = float(np.mean(centred >= point_delta))
    else:
        p = float(np.mean(centred <= point_delta))

    return {
        "rate_a": point_a, "rate_b": point_b, alias_a: point_a, alias_b: point_b,
        "delta": point_delta, "ci_low": ci_low, "ci_high": ci_high, "p_value": p,
        "n_utts": n, "n_bootstrap": int(n_bootstrap), "unit": unit,
    }


def _self_test() -> int:
    print("[Self-test] Bootstrap point estimates must match jiwer on the same input ...")
    refs = [
        "the quick brown fox jumps over the lazy dog",
        "hello world",
        "she sells seashells by the seashore",
        "the rain in spain falls mainly on the plain",
        "a stitch in time saves nine",
    ]
    hyps = [
        "the quick brown fox jumps over the lazy cat",
        "hello world",
        "she sells sea shells by the sea shore",
        "the rain in spain falls mainly on plain",
        "a stitch in time saves none",
    ]

    # --- WER ---
    out_w = wer_with_ci(refs, hyps, n_bootstrap=500, seed=7)
    jw = jiwer.process_words(refs, hyps,
                              reference_transform=DEFAULT_WORD_TRANSFORM,
                              hypothesis_transform=DEFAULT_WORD_TRANSFORM)
    print(f"   WER  point={out_w['wer']:.6f}  jiwer.process_words.wer={jw.wer:.6f}")
    if not np.isclose(out_w["wer"], jw.wer, atol=1e-8):
        print("   FAIL: WER point estimate does not match jiwer.")
        return 1
    if not (out_w["ci_low"] <= out_w["wer"] <= out_w["ci_high"]):
        print("   FAIL: WER point estimate sits outside its own CI.")
        return 1

    # --- CER ---
    out_c = cer_with_ci(refs, hyps, n_bootstrap=500, seed=7)
    jc = jiwer.process_characters(refs, hyps,
                                   reference_transform=DEFAULT_CHAR_TRANSFORM,
                                   hypothesis_transform=DEFAULT_CHAR_TRANSFORM)
    print(f"   CER  point={out_c['cer']:.6f}  jiwer.process_characters.cer={jc.cer:.6f}")
    if not np.isclose(out_c["cer"], jc.cer, atol=1e-8):
        print("   FAIL: CER point estimate does not match jiwer.")
        return 1
    if not (out_c["ci_low"] <= out_c["cer"] <= out_c["ci_high"]):
        print("   FAIL: CER point estimate sits outside its own CI.")
        return 1

    # --- Paired tests (WER and CER) ---
    p_same_w = paired_bootstrap_diff(refs, hyps, hyps, unit="word", n_bootstrap=200, seed=11)
    p_same_c = paired_bootstrap_diff(refs, hyps, hyps, unit="char", n_bootstrap=200, seed=11)
    if abs(p_same_w["delta"]) > 1e-12 or abs(p_same_c["delta"]) > 1e-12:
        print("   FAIL: identical hypotheses produced non-zero delta.")
        return 1

    worse = ["xxx xxx xxx" for _ in refs]
    p_worse_w = paired_bootstrap_diff(refs, hyps, worse, unit="word", n_bootstrap=200, seed=11)
    p_worse_c = paired_bootstrap_diff(refs, hyps, worse, unit="char", n_bootstrap=200, seed=11)
    if p_worse_w["delta"] <= 0 or p_worse_c["delta"] <= 0:
        print("   FAIL: expected positive delta when B is worse.")
        return 1

    print("[Self-test] OK")
    return 0


def parse_args():
    p = argparse.ArgumentParser(description="Bootstrap WER + CI utility.")
    p.add_argument("--self_test", action="store_true", help="Run the acceptance-criterion self-test.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        raise SystemExit(_self_test())
    print("This module is a utility; import its functions or run with --self_test.")


if __name__ == "__main__":
    main()
