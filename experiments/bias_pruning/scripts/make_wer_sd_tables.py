"""Per-subgroup WER tables with TRUE bootstrap standard deviations.

For a given sweep (encoder variant x language x eval corpus), emit two tables
per bias axis:
  (1) absolute WER +/- SD   -- SD = std of corpus WER over 1000 utterance-level
      bootstrap resamples (with replacement); verified to match the analytic
      cluster-robust SE of the ratio estimator to <1%.
  (2) relative to baseline  -- WER_Lx / WER_L0 per group.

Reads per-utterance ref/hyp files (one per depth) so the bootstrap re-aggregates
error/word COUNTS -- the correct corpus-WER resample, not an average of
per-utterance WERs. Scoring normalization matches evaluate_subgroup_wer.py
(lowercase, strip punctuation). Subgroups below the analysability threshold
(>=200 utts AND >=30 min) and the "missing" label are excluded. Usable range
only: depths where aggregate WER <= 40%.

Currently configured for: Whisper large-v2, English, Common Voice 22.
Outputs Markdown to results/tables/<sweep>_wer_sd.md and prints to stdout.
"""
import csv, glob, re, os, collections
import numpy as np
import jiwer

# ---- sweeps to generate (English / Common Voice 22) ----
# per_globs: one or more glob patterns for per-utterance ref/hyp files (one per
# depth). Medium's baseline lives in a separate per_utt/ folder, so it lists two.
EXT = r"P:\Programming\Bias in pruning exps\all_results"
# All three variants' per-utterance files were pulled into one mixed folder;
# run_sweep() selects each variant's files by depth+keep == total (see guard).
# CAUTION: per-utterance filenames encode only d{depth}_keep{keep}, NOT the
# corpus, so a shared dump folder mixes corpora that collide by name. Two
# guards protect every sweep: depth+keep == layers (variant), and n_utts within
# 3% of expect_n (corpus). Files failing either are skipped.
CV22_DUMP = r"P:\Programming\Bias in pruning exps\small_cv22_eng\*.csv"  # contaminated dump
CV22_N = 16390          # Common Voice 22 English test utterances
CV22_AXES = ["accent", "gender", "age"]
FS_AXES = ["ethnicity", "ses", "gender", "age"]   # ethnicity + SES are headline
SWEEPS = [
    {"name": "largev2_cv22_en", "layers": 32, "corpus": "Common Voice 22 (EN)",
     "axes": CV22_AXES, "expect_n": CV22_N,
     "per_globs": [os.path.join(EXT, "largev2_cv22_sweep", "per_utterance", "*.csv")]},
    # CV22 small+medium re-evaluated clean -> reeval_results (see run_parallel_evals)
    {"name": "medium_cv22_en", "layers": 24, "corpus": "Common Voice 22 (EN)",
     "axes": CV22_AXES, "expect_n": CV22_N,
     "per_globs": [r"P:\Programming\Bias in pruning exps\reeval_results\cv22_medium_base\*.csv"]},
    {"name": "small_cv22_en", "layers": 12, "corpus": "Common Voice 22 (EN)",
     "axes": CV22_AXES, "expect_n": CV22_N,
     "per_globs": [r"P:\Programming\Bias in pruning exps\reeval_results\cv22_small_base\*.csv"]},
    {"name": "largev2_fairspeech", "layers": 32, "corpus": "Fair-Speech",
     "axes": FS_AXES, "expect_n": 26417,
     "per_globs": [os.path.join(EXT, "largev2_fairspeech_sweep", "per_utterance", "*.csv")]},
    {"name": "medium_fairspeech", "layers": 24, "corpus": "Fair-Speech",
     "axes": FS_AXES, "expect_n": 26417,
     "per_globs": [os.path.join(EXT, "medium_fairspeech_sweep", "per_utterance", "*.csv"),
                   os.path.join(EXT, "medium_fairspeech_sweep", "per_utt", "*.csv")]},
    {"name": "small_fairspeech", "layers": 12, "corpus": "Fair-Speech",
     "axes": FS_AXES, "expect_n": 26417,
     "per_globs": [os.path.join(EXT, "fairspeech_sweep", "per_utterance", "*.csv")]},
    # cross-lingual: Common Voice Dutch (accent = Belgian vs Netherlands) + Danish
    {"name": "largev2_cv_nl", "layers": 32, "corpus": "Common Voice (NL)",
     "axes": CV22_AXES, "expect_n": 12033,
     "per_globs": [os.path.join(EXT, "largev2_nl_sweep", "per_utt", "*.csv")]},
    {"name": "medium_cv_nl", "layers": 24, "corpus": "Common Voice (NL)",
     "axes": CV22_AXES, "expect_n": 12033,
     "per_globs": [os.path.join(EXT, "medium_nl_sweep", "per_utt", "*.csv")]},
    {"name": "small_cv_nl", "layers": 12, "corpus": "Common Voice (NL)",
     "axes": CV22_AXES, "expect_n": 12033,
     "per_globs": [os.path.join(EXT, "small_nl_sweep", "per_utt", "*.csv")]},
    {"name": "largev2_cv_da", "layers": 32, "corpus": "Common Voice (DA)",
     "axes": CV22_AXES, "expect_n": 2684,
     "per_globs": [os.path.join(EXT, "largev2_da_sweep", "per_utt", "*.csv")]},
    {"name": "medium_cv_da", "layers": 24, "corpus": "Common Voice (DA)",
     "axes": CV22_AXES, "expect_n": 2684,
     "per_globs": [os.path.join(EXT, "medium_da_sweep", "per_utt", "*.csv")]},
    {"name": "small_cv_da", "layers": 12, "corpus": "Common Voice (DA)",
     "axes": CV22_AXES, "expect_n": 2684,
     "per_globs": [r"P:\Programming\Bias in pruning exps\reeval_results\da_small_base\*.csv"]},
]

# ---- LoRA (RQ2) sweeps: same corpora/scales/axes/counts as base, adapter
# condition. LoRA per-utterance is produced by run_parallel_evals.py and lands
# in reeval_results/{corpus}_{scale}_lora/ after scp back from the server.
REEVAL = r"P:\Programming\Bias in pruning exps\reeval_results"   # scp target
# (name, layers, axes, expect_n, corpus, reeval folder = {corpus}_{scale}_lora)
_LORA = [
    ("largev2_cv22", 32, CV22_AXES, CV22_N, "Common Voice 22 (EN)", "cv22_largev2_lora"),
    ("medium_cv22",  24, CV22_AXES, CV22_N, "Common Voice 22 (EN)", "cv22_medium_lora"),
    ("small_cv22",   12, CV22_AXES, CV22_N, "Common Voice 22 (EN)", "cv22_small_lora"),
    ("largev2_fairspeech", 32, FS_AXES, 26417, "Fair-Speech", "fairspeech_largev2_lora"),
    ("medium_fairspeech",  24, FS_AXES, 26417, "Fair-Speech", "fairspeech_medium_lora"),
    ("small_fairspeech",   12, FS_AXES, 26417, "Fair-Speech", "fairspeech_small_lora"),
    ("largev2_cv_nl", 32, CV22_AXES, 12033, "Common Voice (NL)", "nl_largev2_lora"),
    ("medium_cv_nl",  24, CV22_AXES, 12033, "Common Voice (NL)", "nl_medium_lora"),
    ("small_cv_nl",   12, CV22_AXES, 12033, "Common Voice (NL)", "nl_small_lora"),
    ("largev2_cv_da", 32, CV22_AXES, 2684,  "Common Voice (DA)", "da_largev2_lora"),
    ("medium_cv_da",  24, CV22_AXES, 2684,  "Common Voice (DA)", "da_medium_lora"),
]
for _n, _L, _ax, _en, _corp, _folder in _LORA:
    SWEEPS.append({
        "name": f"{_n}_lora", "layers": _L, "corpus": f"{_corp} +LoRA",
        "axes": _ax, "expect_n": _en,
        "per_globs": [os.path.join(REEVAL, _folder, "*.csv")],
    })
OUT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "results", "tables"))

N_BOOT = 1000
BOOT_SEED = 42
MIN_UTTS = 200
MIN_SECONDS = 30 * 60
USABLE_AGG_MAX = 40.0        # aggregate WER (%) at/below = "usable range" (for claims)
MAX_VALID_WER = 150.0        # above this = corrupt export (e.g. keep-22 ~600%), excluded;
                             # real model collapse (~100-110%) is kept and shown

TX = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
                    jiwer.RemoveMultipleSpaces(), jiwer.Strip(),
                    jiwer.ReduceToListOfListOfWords()])

SHORT = {
    # CV22 accent
    "united states english": "US", "england english": "England",
    "india and south asia (india, pakistan, sri lanka)": "India/S-Asia",
    # gender / age
    "male": "Male", "female": "Female",
    "teens": "Teens", "twenties": "Twenties", "thirties": "Thirties",
    # Fair-Speech ethnicity
    "black or african american": "Black",
    "asian, south asian or asian american": "Asian",
    "hispanic, latino, or spanish": "Hispanic",
    "middle eastern or north african": "MENA",
    "native american, american indian, or alaska native": "Native Am.",
    "native hawaiian or other pacific islander": "Native Haw.",
    "white": "White",
    # Fair-Speech SES
    "affluent": "Affluent", "medium": "Medium", "low": "Low",
    # Common Voice Dutch accent
    "belgisch nederlands": "Belgian", "nederlands nederlands": "Netherlands",
}

def norm(v):
    return (v or "").strip().lower() or "missing"

def gender_norm(g):
    g = norm(g)
    if g.startswith("male") or g in ("m", "male_masculine"):
        return "male"
    if g.startswith("female") or g in ("f", "female_feminine"):
        return "female"
    return g

def utt_counts(ref, hyp):
    """(#errors, #ref_words) for one utterance under the scoring transform."""
    o = jiwer.process_words([ref], [hyp], reference_transform=TX, hypothesis_transform=TX)
    return o.substitutions + o.deletions + o.insertions, o.hits + o.substitutions + o.deletions

def bootstrap_wer_sd(errors, words, rng):
    """point WER (%) and TRUE bootstrap SD (%) via count re-aggregation."""
    e = np.asarray(errors, float); w = np.asarray(words, float)
    point = e.sum() / w.sum() * 100.0
    n = len(e)
    boots = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        boots[b] = e[idx].sum() / w[idx].sum() * 100.0
    return point, float(np.std(boots))     # ddof=0, standard for bootstrap

def load_depth(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:
        r["_e"], r["_n"] = utt_counts(r["reference"], r["hypothesis"])
        r["_dur"] = float(r["duration_s"])
    return rows

def group_key(row, axis):
    return gender_norm(row[axis]) if axis == "gender" else norm(row[axis])

VARIANT = {32: "large-v2", 24: "medium", 12: "small"}

def run_sweep(sweep):
    name, total, per_globs = sweep["name"], sweep["layers"], sweep["per_globs"]
    axes, corpus, expect_n = sweep["axes"], sweep["corpus"], sweep["expect_n"]
    rng = np.random.default_rng(BOOT_SEED)

    # results[axis][group][removed] = (wer, sd)
    results = collections.defaultdict(lambda: collections.defaultdict(dict))
    agg_wer = {}     # removed -> aggregate WER (for usable-range gating)
    agg_sd = {}      # removed -> aggregate bootstrap SD

    paths = sorted(p for g in per_globs for p in glob.glob(g))
    for path in paths:
        fn = os.path.basename(path)
        keep = int(re.search(r"keep(\d+)", fn).group(1))
        removed = int(re.search(r"d(\d+)_", fn).group(1))
        # mixed-folder guard: a file belongs to THIS variant only if its
        # depth + kept layers equals the variant's total encoder depth.
        if removed + keep != total:
            continue
        rows = load_depth(path)
        # corpus guard: filename carries no corpus tag, so verify by utterance
        # count. Rejects a different-corpus run that collided at this depth.
        if abs(len(rows) - expect_n) > 0.03 * expect_n:
            print(f"[warn] {name}: skip {os.path.basename(path)} "
                  f"(n={len(rows)}, expected ~{expect_n} -> different corpus)")
            continue
        # aggregate WER (+ bootstrap SD) over ALL utterances at this depth
        a_point, a_sd = bootstrap_wer_sd([r["_e"] for r in rows],
                                         [r["_n"] for r in rows], rng)
        agg_wer[removed] = a_point
        agg_sd[removed] = a_sd
        for axis in axes:
            buckets = collections.defaultdict(list)
            for r in rows:
                k = group_key(r, axis)
                if k == "missing":
                    continue
                buckets[k].append(r)
            for grp, items in buckets.items():
                if len(items) < MIN_UTTS or sum(x["_dur"] for x in items) < MIN_SECONDS:
                    continue
                point, sd = bootstrap_wer_sd([x["_e"] for x in items],
                                             [x["_n"] for x in items], rng)
                results[axis][grp][removed] = (point, sd)

    # show ALL real depths (corrupt exports > MAX_VALID_WER excluded)
    usable = sorted(d for d, w in agg_wer.items() if w <= MAX_VALID_WER)
    if not usable:
        print(f"[skip] {name}: no per-utterance files found "
              f"(pattern {per_globs}); nothing to write.")
        return
    usable_last = max((d for d in usable if agg_wer[d] <= USABLE_AGG_MAX), default=usable[0])

    lines = [f"# WER +/- bootstrap SD --- {name} "
             f"(Whisper {VARIANT.get(total, total)}, {corpus})",
             f"\nFull pruning sweep L-{usable[0]} ... L-{usable[-1]} "
             f"(all depths shown). Usable range for claims = aggregate WER "
             f"<= {USABLE_AGG_MAX:.0f}% (up to L-{usable_last}); deeper depths "
             f"show degradation/collapse. SD = std of corpus WER over {N_BOOT} "
             f"utterance-level bootstrap resamples. Subgroups need >= {MIN_UTTS} "
             f"utts AND >= {MIN_SECONDS // 60} min; 'missing' excluded.\n"]

    for axis in axes:
        groups = sorted(results[axis])
        header = "| group | " + " | ".join(f"L-{d}" for d in usable) + " |"
        sep = "|---" * (len(usable) + 1) + "|"

        lines.append(f"\n## {axis.upper()} --- absolute WER (%) +/- SD")
        lines += [header, sep]
        for g in groups:
            cells = " | ".join(
                f"{results[axis][g][d][0]:.1f} ± {results[axis][g][d][1]:.1f}"
                if d in results[axis][g] else "---" for d in usable)
            lines.append(f"| {SHORT.get(g, g)} | {cells} |")
        agg_cells = " | ".join(f"**{agg_wer[d]:.1f} ± {agg_sd[d]:.1f}**" for d in usable)
        lines.append(f"| **ALL (aggregate)** | {agg_cells} |")

        lines.append(f"\n## {axis.upper()} --- relative to baseline (WER$_{{Lx}}$/WER$_{{L0}}$)")
        lines += [header, sep]
        for g in groups:
            base = results[axis][g].get(0, (None,))[0]
            cells = " | ".join(
                f"{results[axis][g][d][0] / base:.2f}"
                if (d in results[axis][g] and base) else "---" for d in usable)
            lines.append(f"| {SHORT.get(g, g)} | {cells} |")
        agg_base = agg_wer[0]
        agg_cells = " | ".join(f"**{agg_wer[d] / agg_base:.2f}**" for d in usable)
        lines.append(f"| **ALL (aggregate)** | {agg_cells} |")

    out = os.path.join(OUT_DIR, f"{name}_wer_sd.md")
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")

if __name__ == "__main__":
    import sys
    os.makedirs(OUT_DIR, exist_ok=True)
    # optional substring filters: `python make_wer_sd_tables.py cv_nl cv_da`
    filters = sys.argv[1:]
    for sweep in SWEEPS:
        if filters and not any(f in sweep["name"] for f in filters):
            continue
        run_sweep(sweep)
        print("\n" + "=" * 70 + "\n")
