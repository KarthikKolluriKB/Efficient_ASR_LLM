"""LoRA (RQ2) WER +/- SD tables, hybrid source.

For every (corpus, scale) LoRA cell:
  - baseline coverage from the LoRA sweep's *_multiaxis.csv (per-group WER +
    bootstrap CI). SD is CI-derived: (ci_hi - ci_lo) / 2 / 1.96. Marked '*'.
  - where fresh per-utterance re-eval exists (reeval_results/{corpus}_{scale}_lora/),
    the TRUE resample SD overrides the CI estimate for that depth (unmarked).

So the table is complete immediately (all depths from CI), and upgrades to true
SD depth-by-depth as the re-eval lands. A footnote records that '*' = SD
estimated from the stored bootstrap CI (per-utterance checkpoint unavailable).

Outputs results/tables/{name}_lora_wer_sd.md per cell (feeds make_table_slides).
"""
import csv, glob, os, re, collections
import numpy as np
import make_wer_sd_tables as W          # bootstrap_wer_sd, load_depth, group_key, thresholds

EXT = r"P:\Programming\Bias in pruning exps\all_results"
REEVAL = r"P:\Programming\Bias in pruning exps\reeval_results"
OUT = W.OUT_DIR
MAXW = W.MAX_VALID_WER

# (name, layers, axes, corpus label, multiaxis folder, reeval folder)
CELLS = [
    ("largev2_cv22", 32, W.CV22_AXES, "Common Voice 22 (EN)", "largev2_cv22_LORA_sweep", "cv22_largev2_lora"),
    ("medium_cv22",  24, W.CV22_AXES, "Common Voice 22 (EN)", "medium_cv22_LORA_sweep",  "cv22_medium_lora"),
    ("small_cv22",   12, W.CV22_AXES, "Common Voice 22 (EN)", "small_cv22_LORA_sweep",   "cv22_small_lora"),
    ("largev2_fairspeech", 32, W.FS_AXES, "Fair-Speech", "largev2_fairspeech_LORA_sweep", "fairspeech_largev2_lora"),
    ("medium_fairspeech",  24, W.FS_AXES, "Fair-Speech", "medium_fairspeech_LORA_sweep",  "fairspeech_medium_lora"),
    ("small_fairspeech",   12, W.FS_AXES, "Fair-Speech", "small_fairspeech_LORA_sweep",   "fairspeech_small_lora"),
    ("largev2_cv_nl", 32, W.CV22_AXES, "Common Voice (NL)", "largev2_nl_LORA_sweep", "nl_largev2_lora"),
    ("medium_cv_nl",  24, W.CV22_AXES, "Common Voice (NL)", "medium_nl_LORA_sweep",  "nl_medium_lora"),
    ("small_cv_nl",   12, W.CV22_AXES, "Common Voice (NL)", "small_nl_LORA_sweep",   "nl_small_lora"),
    ("largev2_cv_da", 32, W.CV22_AXES, "Common Voice (DA)", "largev2_da_LORA_sweep", "da_largev2_lora"),
    ("medium_cv_da",  24, W.CV22_AXES, "Common Voice (DA)", "medium_da_LORA_sweep",  "da_medium_lora"),
    ("largev2_l2arctic", 32, ["l1"], "L2-ARCTIC", "largev2_l2arctic_LORA_sweep", "l2arctic_largev2_lora"),
    ("medium_l2arctic",  24, ["l1"], "L2-ARCTIC", "medium_l2arctic_LORA_sweep",  "l2arctic_medium_lora"),
    ("small_l2arctic",   12, ["l1"], "L2-ARCTIC", "small_l2arctic_LORA_sweep",   "l2arctic_small_lora"),
]
# rq_final holds NL/DA LoRA sweeps
ALT_ROOTS = [EXT, os.path.join(os.path.dirname(EXT), "rq_final")]

def find_folder(name):
    for root in ALT_ROOTS:
        p = os.path.join(root, name)
        if os.path.isdir(p):
            return p
    return None

def ci_from_multiaxis(folder, axes):
    """removed -> {axis: {group: (wer, sd_est)}}, agg -> {removed:(wer,sd,used)}."""
    res = collections.defaultdict(lambda: collections.defaultdict(dict))
    agg = {}
    for f in glob.glob(os.path.join(folder, "*_multiaxis.csv")):
        keep = int(re.search(r"keep(\d+)", os.path.basename(f)).group(1))
        removed = int(re.search(r"d(\d+)_", os.path.basename(f)).group(1))
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if r.get("analysable") != "yes":
                continue
            wer = float(r["wer"]) * 100
            sd = (float(r["wer_ci_high"]) - float(r["wer_ci_low"])) * 100 / 2 / 1.96
            if r["group"] == "ALL":
                agg[removed] = (wer, sd)
            elif r["axis"] in axes:
                res[removed][r["axis"]][r["group"]] = (wer, sd)
    return res, agg

def true_from_reeval(folder, total, axes):
    """removed -> {axis:{group:(wer,sd)}}, agg  — computed from per-utterance."""
    res = collections.defaultdict(lambda: collections.defaultdict(dict))
    agg = {}
    rng = np.random.default_rng(W.BOOT_SEED)
    for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        m = re.search(r"d(\d+)_keep(\d+)", os.path.basename(path))
        if not m:
            continue
        removed = int(m.group(1))
        rows = W.load_depth(path)
        agg[removed] = W.bootstrap_wer_sd([r["_e"] for r in rows], [r["_n"] for r in rows], rng)
        for axis in axes:
            buckets = collections.defaultdict(list)
            for r in rows:
                k = W.group_key(r, axis)
                if k != "missing":
                    buckets[k].append(r)
            for g, items in buckets.items():
                if len(items) >= W.MIN_UTTS and sum(x["_dur"] for x in items) >= W.MIN_SECONDS:
                    res[removed][axis][g] = W.bootstrap_wer_sd(
                        [x["_e"] for x in items], [x["_n"] for x in items], rng)
    return res, agg

SHORT = W.SHORT
SHORT.setdefault("arabic", "Arabic"); SHORT.setdefault("chinese", "Chinese")
for k in ("hindi", "korean", "spanish", "vietnamese"):
    SHORT.setdefault(k, k.title())

def build(name, total, axes, corpus, ma_folder, reeval_name):
    folder = find_folder(ma_folder)
    if not folder:
        print(f"[skip] {name}: no multiaxis folder {ma_folder}"); return
    ci, ci_agg = ci_from_multiaxis(folder, axes)
    tr, tr_agg = true_from_reeval(os.path.join(REEVAL, reeval_name), total, axes)

    # merge: prefer true (reeval) SD, else CI estimate; track estimated cells
    depths = sorted(d for d in set(ci_agg) | set(tr_agg)
                    if (tr_agg.get(d, ci_agg.get(d))[0]) <= MAXW)
    def cell(d, axis, g):
        if d in tr and axis in tr[d] and g in tr[d][axis]:
            return tr[d][axis][g], False           # true SD
        if d in ci and axis in ci[d] and g in ci[d][axis]:
            return ci[d][axis][g], True            # CI-estimated
        return None, None
    def aggcell(d):
        if d in tr_agg:
            return tr_agg[d], False
        return ci_agg[d], True

    any_est = False
    lines = [f"# WER +/- SD --- {name}_lora (Whisper {W.VARIANT.get(total,total)}, {corpus} +LoRA)",
             "\n'*' = SD estimated from stored bootstrap CI (per-utterance "
             "re-eval unavailable); unmarked = true resample SD. Full sweep; "
             "aggregate WER <= 40% is the usable range.\n"]
    for axis in axes:
        groups = sorted({g for d in depths if d in ci for g in ci[d].get(axis, {})}
                        | {g for d in depths if d in tr for g in tr[d].get(axis, {})})
        hdr = "| group | " + " | ".join(f"L-{d}" for d in depths) + " |"
        sep = "|---" * (len(depths) + 1) + "|"
        lines += [f"\n## {axis.upper()} --- absolute WER (%) +/- SD", hdr, sep]
        for g in groups:
            cells = []
            for d in depths:
                v, est = cell(d, axis, g)
                if v is None:
                    cells.append("---")
                else:
                    cells.append(f"{v[0]:.1f} ± {v[1]:.1f}{'*' if est else ''}")
                    any_est |= bool(est)
            lines.append(f"| {SHORT.get(g, g)} | " + " | ".join(cells) + " |")
        acells = []
        for d in depths:
            (w, s), est = aggcell(d)
            acells.append(f"**{w:.1f} ± {s:.1f}{'*' if est else ''}**")
        lines.append(f"| **ALL (aggregate)** | " + " | ".join(acells) + " |")
        # relative to baseline (or first available depth if L-0 absent)
        ref = 0 if 0 in depths else depths[0]
        lines += [f"\n## {axis.upper()} --- relative to baseline (WER Lx / WER L{ref})", hdr, sep]
        for g in groups:
            v0, _ = cell(ref, axis, g)
            b = v0[0] if v0 else None
            cells = [f"{cell(d, axis, g)[0][0]/b:.2f}"
                     if (cell(d, axis, g)[0] and b) else "---" for d in depths]
            lines.append(f"| {SHORT.get(g, g)} | " + " | ".join(cells) + " |")
        ab = aggcell(ref)[0][0]
        arow = " | ".join(f"**{aggcell(d)[0][0]/ab:.2f}**" for d in depths)
        lines.append(f"| **ALL (aggregate)** | {arow} |")

    out = os.path.join(OUT, f"{name}_lora_wer_sd.md")
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    ntrue = len(tr_agg)
    print(f"wrote {name}_lora ({len(depths)} depths; {ntrue} true-SD, rest CI-est)")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for c in CELLS:
        build(*c)
