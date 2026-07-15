"""L2-ARCTIC L1-background tables (WER +/- true bootstrap SD), all model scales
in ONE markdown file and ONE slide deck.

L2-ARCTIC is small (6 L1 groups x ~600 utts) and its per-depth checkpoints are
noisy, so it is reported as illustrative, not for quantitative claims. Same
bootstrap-SD method as make_wer_sd_tables.py (imported), same slide style as
make_table_slides.py (imported).

Outputs (results/tables/):
  l2arctic_all_scales_wer_sd.md      - all three scales, L1 axis
  l2arctic_all_scales_slides.pptx    - one slide per scale (absolute + relative)
"""
import os, glob, re, collections
import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt

import make_wer_sd_tables as W          # bootstrap_wer_sd, load_depth, norm, thresholds
import make_table_slides as S           # add_axis_slide, styling

EXT = r"P:\Programming\Bias in pruning exps\all_results"
OUT_DIR = W.OUT_DIR
AXIS = "l1"
EXPECT_N = 3599                          # L2-ARCTIC utterances
SCALES = [
    ("large-v2", 32, [os.path.join(EXT, "largev2_l2arctic_sweep", "per_utterance", "*.csv"),
                      os.path.join(EXT, "largev2_l2arctic_sweep", "per_utt", "*.csv")]),
    ("medium", 24, [os.path.join(EXT, "medium_l2arctic_sweep", "per_utterance", "*.csv"),
                    os.path.join(EXT, "medium_l2arctic_sweep", "per_utt", "*.csv")]),
    ("small", 12, [os.path.join(EXT, "l2arctic_sweep", "per_utterance", "*.csv"),
                   os.path.join(EXT, "l2arctic_sweep", "per_utt", "*.csv")]),
]
LABEL = {"arabic": "Arabic", "chinese": "Chinese", "hindi": "Hindi",
         "korean": "Korean", "spanish": "Spanish", "vietnamese": "Vietnamese"}

def compute(total, globs):
    rng = np.random.default_rng(W.BOOT_SEED)
    res = collections.defaultdict(dict)          # group -> {removed: (wer, sd)}
    agg = {}; asd = {}
    for path in sorted(p for g in globs for p in glob.glob(g)):
        fn = os.path.basename(path)
        keep = int(re.search(r"keep(\d+)", fn).group(1))
        removed = int(re.search(r"d(\d+)_", fn).group(1))
        if removed + keep != total:
            continue
        rows = W.load_depth(path)
        if abs(len(rows) - EXPECT_N) > 0.03 * EXPECT_N:
            continue
        a_p, a_s = W.bootstrap_wer_sd([r["_e"] for r in rows], [r["_n"] for r in rows], rng)
        agg[removed], asd[removed] = a_p, a_s
        buckets = collections.defaultdict(list)
        for r in rows:
            k = W.norm(r[AXIS])
            if k != "missing":
                buckets[k].append(r)
        for grp, items in buckets.items():
            if len(items) < W.MIN_UTTS or sum(x["_dur"] for x in items) < W.MIN_SECONDS:
                continue
            p, s = W.bootstrap_wer_sd([x["_e"] for x in items], [x["_n"] for x in items], rng)
            res[grp][removed] = (p, s)
    usable = sorted(d for d, w in agg.items() if w <= W.MAX_VALID_WER)   # full sweep
    return res, agg, asd, usable

def md_tables(variant, res, agg, asd, usable):
    groups = sorted(res)
    hdr = "| group | " + " | ".join(f"L-{d}" for d in usable) + " |"
    sep = "|---" * (len(usable) + 1) + "|"
    out = [f"\n## L1 --- Whisper {variant} --- absolute WER (%) +/- SD", hdr, sep]
    for g in groups:
        out.append(f"| {LABEL.get(g, g)} | " + " | ".join(
            f"{res[g][d][0]:.1f} ± {res[g][d][1]:.1f}" if d in res[g] else "---"
            for d in usable) + " |")
    out.append("| **ALL (aggregate)** | " + " | ".join(
        f"**{agg[d]:.1f} ± {asd[d]:.1f}**" for d in usable) + " |")
    out += [f"\n## L1 --- Whisper {variant} --- relative to baseline (WER Lx / WER L0)", hdr, sep]
    for g in groups:
        b = res[g].get(0, (None,))[0]
        out.append(f"| {LABEL.get(g, g)} | " + " | ".join(
            f"{res[g][d][0]/b:.2f}" if (d in res[g] and b) else "---"
            for d in usable) + " |")
    ab = agg.get(0)
    out.append("| **ALL (aggregate)** | " + " | ".join(
        f"**{agg[d]/ab:.2f}**" for d in usable) + " |")
    return out, {"absolute": (["group"] + [f"L-{d}" for d in usable],
                              [[LABEL.get(g, g)] + [f"{res[g][d][0]:.1f} ± {res[g][d][1]:.1f}"
                               if d in res[g] else "---" for d in usable] for g in groups]
                              + [["ALL (aggregate)"] + [f"{agg[d]:.1f} ± {asd[d]:.1f}" for d in usable]]),
                 "relative": (["group"] + [f"L-{d}" for d in usable],
                              [[LABEL.get(g, g)] + [f"{res[g][d][0]/res[g][0][0]:.2f}"
                               if (d in res[g] and 0 in res[g]) else "---" for d in usable] for g in groups]
                              + [["ALL (aggregate)"] + [f"{agg[d]/agg[0]:.2f}" for d in usable]])}

def main():
    md = ["# WER +/- bootstrap SD --- L2-ARCTIC (all Whisper scales), L1 background",
          "\n**Illustrative only:** L2-ARCTIC has 6 L1 groups (~600 utts each) and "
          "noisy per-depth checkpoints; usable range (aggregate WER <= 40%) is short "
          "and non-monotonic. SD = std over 1000 utterance-level bootstrap resamples.\n"]
    prs = Presentation()
    prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
    margin = Inches(0.55); width = prs.slide_width - 2 * margin
    for variant, total, globs in SCALES:
        res, agg, asd, usable = compute(total, globs)
        if not usable:
            print(f"[skip] {variant}: no usable depths"); continue
        lines, tables = md_tables(variant, res, agg, asd, usable)
        md += lines
        S.add_axis_slide(prs, variant, "L2-ARCTIC", "L1", tables, margin, width)
        print(f"{variant}: usable L-{usable[0]}..L-{usable[-1]} ({len(usable)} depths)")
    open(os.path.join(OUT_DIR, "l2arctic_all_scales_wer_sd.md"), "w",
         encoding="utf-8").write("\n".join(md) + "\n")
    prs.save(os.path.join(OUT_DIR, "l2arctic_all_scales_slides.pptx"))
    print(f"wrote l2arctic_all_scales_wer_sd.md + l2arctic_all_scales_slides.pptx "
          f"({len(prs.slides._sldIdLst)} slides)")

if __name__ == "__main__":
    main()
