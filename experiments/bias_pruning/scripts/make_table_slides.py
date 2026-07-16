"""Render WER±SD table markdown into styled .pptx decks (one slide per bias
axis), matching the project's teal-header table style.

For each input <name>_wer_sd.md it writes <name>_slides.pptx with one slide per
axis, each carrying the absolute WER±SD table and the relative-to-baseline
table. Title is parsed from the md header: "Whisper <variant> – English –
<corpus> – <Axis>".

Edit TABLES below (or pass .md paths as argv) and run:
    python make_table_slides.py
"""
import re, os, sys, glob
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

HERE = os.path.dirname(os.path.abspath(__file__))
TBL_DIR = os.path.normpath(os.path.join(HERE, "..", "results", "tables"))

# default batch: all Fair-Speech scales
TABLES = [
    os.path.join(TBL_DIR, "largev2_fairspeech_wer_sd.md"),
    os.path.join(TBL_DIR, "medium_fairspeech_wer_sd.md"),
    os.path.join(TBL_DIR, "small_fairspeech_wer_sd.md"),
]

TEAL = RGBColor(0x1F, 0x6E, 0x8C)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x1A, 0x1A, 0x1A)
ROW_ALT = RGBColor(0xF2, 0xF2, 0xF2)
AGG = RGBColor(0xE3, 0xE3, 0xE3)
FONT = "Calibri"
AXIS_TITLE = {"SES": "SES", "L1": "L1"}   # keep acronyms; others Title-cased

def sd_source(md):
    """How the source table says its SD was obtained. Read from the table's own
    footnote -- never assumed -- so a combined file cannot claim a computation
    that was not run for those cells."""
    if "CI-width" in md:
        return "derived from the stored bootstrap 95% CI (CI-width / 3.92)"
    m = re.search(r"std of corpus WER over (\d+) utterance-level bootstrap", md)
    if m:
        return (f"std of corpus WER over {m.group(1)} utterance-level bootstrap "
                f"resamples")
    return "source not stated in the per-scale table"


def parse(md):
    """Return (header_variant, header_corpus, ordered axes,
    {axis: {'absolute': (cols, rows), 'relative': (cols, rows)}})."""
    variant = corpus = None
    m = re.search(r"\(Whisper ([^,]+), (.+)\)", md)   # greedy: corpus may contain ()
    if m:
        variant, corpus = m.group(1).strip(), m.group(2).strip()
    axes = {}; order = []
    cur_axis = cur_kind = cols = None; rows = []
    def flush():
        if cur_axis and cur_kind and cols:
            axes.setdefault(cur_axis, {})[cur_kind] = (cols, list(rows))
    for line in md.splitlines():
        h = re.match(r"## ([\w/ ]+?) --- (absolute|relative)", line)
        if h:
            flush()
            cur_axis, cur_kind = h.group(1).strip(), h.group(2)
            if cur_axis not in order:
                order.append(cur_axis)
            cols = None; rows = []
            continue
        if line.startswith("|") and cur_axis:
            cells = [c.strip().replace("**", "")
                     for c in line.strip().strip("|").split("|")]
            if set("".join(cells)) <= set("-"):
                continue
            if cells[0] == "group":
                cols = cells
            else:
                rows.append(cells)
    flush()
    return variant, corpus, order, axes

ROW_H = Inches(0.28)     # explicit row height so 8-group tables fit two per slide

def fit_params(ndepth):
    """(font_pt, group_col_in) adapted to the number of depth columns so wide
    full-sweep tables (up to ~16 depths) still fit one slide."""
    if ndepth <= 8:
        return 10.5, 2.0
    if ndepth <= 11:
        return 9.0, 1.7
    if ndepth <= 14:
        return 8.0, 1.5
    return 7.0, 1.35

def set_cell(cell, text, *, fill=None, color=INK, bold=False, align=PP_ALIGN.CENTER, size=10.5):
    cell.margin_left = Inches(0.03); cell.margin_right = Inches(0.03)
    cell.margin_top = Inches(0.01); cell.margin_bottom = Inches(0.01)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    if fill is None:
        cell.fill.background()
    else:
        cell.fill.solid(); cell.fill.fore_color.rgb = fill
    p = cell.text_frame.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.bold = bold; r.font.name = FONT; r.font.color.rgb = color

def add_block(slide, title, cols, rows, top, margin, width):
    bar = slide.shapes.add_shape(1, margin, top, width, Inches(0.34))
    bar.fill.solid(); bar.fill.fore_color.rgb = TEAL; bar.line.fill.background()
    bar.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    bp = bar.text_frame.paragraphs[0]; bp.alignment = PP_ALIGN.LEFT
    bar.text_frame.margin_left = Inches(0.12)
    br = bp.add_run(); br.text = title
    br.font.size = Pt(12.5); br.font.bold = True; br.font.name = FONT; br.font.color.rgb = WHITE

    ncol, nrow = len(cols), len(rows) + 1
    fsize, gcol_in = fit_params(ncol - 1)
    gcol = Inches(gcol_in)
    tbl_top = top + Inches(0.36)
    gt = slide.shapes.add_table(nrow, ncol, margin, tbl_top, width, ROW_H * nrow).table
    gt.first_row = False; gt.horz_banding = False
    gt.columns[0].width = gcol
    rest = (width - gcol) // (ncol - 1)
    for c in range(1, ncol):
        gt.columns[c].width = Emu(int(rest))
    for r in gt.rows:
        r.height = ROW_H
    set_cell(gt.cell(0, 0), cols[0], fill=WHITE, color=TEAL, bold=True, size=fsize)
    for c in range(1, ncol):
        set_cell(gt.cell(0, c), cols[c], fill=TEAL, color=WHITE, bold=True, size=fsize)
    for ri, row in enumerate(rows, start=1):
        is_agg = "aggregate" in row[0].lower()
        fill = AGG if is_agg else (ROW_ALT if ri % 2 == 0 else WHITE)
        for ci, val in enumerate(row):
            set_cell(gt.cell(ri, ci), val, fill=fill, bold=is_agg, size=fsize,
                     align=PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER)
    return tbl_top + ROW_H * nrow

def add_axis_slide(prs, variant, corpus, axis, tables, margin, width):
    blank = prs.slide_layouts[6]
    s = prs.slides.add_slide(blank)
    tb = s.shapes.add_textbox(margin, Inches(0.25), width, Inches(0.7)).text_frame
    tp = tb.paragraphs[0]; tp.alignment = PP_ALIGN.CENTER
    axis_disp = AXIS_TITLE.get(axis.upper(), axis.title())
    lang = ("Dutch" if "(NL)" in corpus else "Danish" if "(DA)" in corpus else "English")
    tr = tp.add_run()
    tr.text = f"Whisper {variant} – {lang} – {corpus} – {axis_disp}"
    tr.font.size = Pt(26); tr.font.bold = True; tr.font.name = FONT; tr.font.color.rgb = INK
    y = add_block(s, f"{axis.upper()} – absolute WER (%) ± SD",
                  *tables["absolute"], Inches(1.05), margin, width)
    add_block(s, f"{axis.upper()} – relative to baseline (WER Lx / WER L0)",
              *tables["relative"], y + Inches(0.28), margin, width)

# canonical axis order and scale order for combined decks
AXIS_SEQ = ["ETHNICITY", "SES", "ACCENT", "L1", "GENDER", "AGE"]
SCALE_RANK = {"large-v2": 0, "medium": 1, "small": 2}

def build(md_path):
    """One deck per table file (slides in the file's own axis order)."""
    variant, corpus, order, axes = parse(open(md_path, encoding="utf-8").read())
    prs = Presentation()
    prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
    margin = Inches(0.55); width = prs.slide_width - 2 * margin
    for axis in order:
        a = axes[axis]
        if "absolute" in a and "relative" in a:
            add_axis_slide(prs, variant, corpus, axis, a, margin, width)
    out = md_path.replace("_wer_sd.md", "_slides.pptx")
    prs.save(out)
    print(f"wrote {os.path.basename(out)}  ({len(prs.slides._sldIdLst)} slides)")

def build_combined_md(md_paths, out_md, dataset):
    """One markdown file per dataset: all scales, grouped by axis then scale
    (scale order large-v2 -> medium -> small)."""
    parsed = []
    for p in md_paths:
        text = open(p, encoding="utf-8").read()
        v, c, _o, axes = parse(text)
        parsed.append((v, c, axes, sd_source(text)))
    corpus = parsed[0][1]
    all_axes = {ax.upper() for _, _, axes, _ in parsed for ax in axes}
    axis_order = [a for a in AXIS_SEQ if a in all_axes]

    # Never assert one SD method across scales: a dataset can mix true-resample
    # SD (per-utterance available) with CI-derived SD. Report what each source
    # table actually says.
    srcs = {v: s for v, _, _, s in parsed}
    if len(set(srcs.values())) == 1:
        sd_line = f"SD: {next(iter(srcs.values()))}."
    else:
        per = "; ".join(f"{v} — {srcs[v]}" for v in
                        sorted(srcs, key=lambda x: SCALE_RANK.get(x, 9)))
        sd_line = f"SD source varies by scale — {per}."

    out = [f"# WER ± SD — {dataset} (all Whisper scales)",
           f"\nCorpus: {corpus}. Full pruning sweep shown per scale (aggregate WER "
           f"≤ 40% = usable range for claims; deeper depths show degradation/"
           f"collapse). {sd_line}\n"]
    for axis in axis_order:
        out.append(f"\n# {axis}")
        for variant, corp, axes, _sd in sorted(parsed,
                                               key=lambda t: SCALE_RANK.get(t[0], 9)):
            match = next((axes[a] for a in axes if a.upper() == axis), None)
            if not match or "absolute" not in match or "relative" not in match:
                continue
            for kind, label in (("absolute", "absolute WER (%) ± SD"),
                                 ("relative", "relative to baseline (WER Lx / WER L0)")):
                cols, rows = match[kind]
                out.append(f"\n## Whisper {variant} — {label}")
                out.append("| " + " | ".join(cols) + " |")
                out.append("|" + "---|" * len(cols))
                for r in rows:
                    bold = "aggregate" in r[0].lower()
                    cells = [f"**{x}**" if bold else x for x in r]
                    out.append("| " + " | ".join(cells) + " |")
    open(out_md, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print(f"wrote {os.path.basename(out_md)}")

def build_combined(md_paths, out_path):
    """One deck across several scales, grouped by axis (all scales per axis
    consecutively), scale order large-v2 -> medium -> small."""
    parsed = []   # (variant, corpus, axes-dict)
    for p in md_paths:
        v, c, _order, axes = parse(open(p, encoding="utf-8").read())
        parsed.append((v, c, axes))
    corpus = parsed[0][1]
    all_axes = {ax for _, _, axes in parsed for ax in axes}
    axis_order = [a for a in AXIS_SEQ if a.upper() in {x.upper() for x in all_axes}]
    prs = Presentation()
    prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
    margin = Inches(0.55); width = prs.slide_width - 2 * margin
    n = 0
    for axis in axis_order:
        for variant, corp, axes in sorted(parsed, key=lambda t: SCALE_RANK.get(t[0], 9)):
            match = next((axes[a] for a in axes if a.upper() == axis.upper()), None)
            if match and "absolute" in match and "relative" in match:
                add_axis_slide(prs, variant, corp, axis, match, margin, width)
                n += 1
    prs.save(out_path)
    print(f"wrote {os.path.basename(out_path)}  ({n} slides, grouped by axis)")

# dataset -> (output basename, per-scale md files large->small)
DATASETS = {
    "Common Voice 22 (EN)":  ("cv22_en_all_scales",
        ["largev2_cv22_en", "medium_cv22_en", "small_cv22_en"]),
    "Fair-Speech":           ("fairspeech_all_scales",
        ["largev2_fairspeech", "medium_fairspeech", "small_fairspeech"]),
    "Common Voice (NL)":     ("dutch_all_scales",
        ["largev2_cv_nl", "medium_cv_nl", "small_cv_nl"]),
    "Common Voice (DA)":     ("danish_all_scales",
        ["largev2_cv_da", "medium_cv_da", "small_cv_da"]),
    # --- LoRA (RQ2) combined decks, per dataset ---
    "Common Voice 22 (EN) +LoRA": ("cv22_en_lora_all_scales",
        ["largev2_cv22_lora", "medium_cv22_lora", "small_cv22_lora"]),
    "Fair-Speech +LoRA":          ("fairspeech_lora_all_scales",
        ["largev2_fairspeech_lora", "medium_fairspeech_lora", "small_fairspeech_lora"]),
    "Common Voice (NL) +LoRA":    ("dutch_lora_all_scales",
        ["largev2_cv_nl_lora", "medium_cv_nl_lora", "small_cv_nl_lora"]),
    "Common Voice (DA) +LoRA":    ("danish_lora_all_scales",
        ["largev2_cv_da_lora", "medium_cv_da_lora"]),
}

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--datasets":
        # optional filter: --datasets [all|base|lora]  (default all)
        only = sys.argv[2].lower() if len(sys.argv) > 2 else "all"
        for dataset, (base, names) in DATASETS.items():
            is_lora = "+LoRA" in dataset
            if (only == "base" and is_lora) or (only == "lora" and not is_lora):
                continue
            files = [os.path.join(TBL_DIR, f"{n}_wer_sd.md") for n in names]
            files = [f for f in files if os.path.exists(f)]
            if not files:
                continue      # dataset's per-scale tables not generated yet
            build_combined_md(files, os.path.join(TBL_DIR, f"{base}_wer_sd.md"), dataset)
            build_combined(files, os.path.join(TBL_DIR, f"{base}_slides.pptx"))
        sys.exit(0)
    # --combined <out_basename> <md...>  -> one grouped deck across scales
    # (no extra args -> default Fair-Speech trio)
    if len(sys.argv) > 1 and sys.argv[1] == "--combined":
        if len(sys.argv) >= 4:
            out = os.path.join(TBL_DIR, sys.argv[2] + "_slides.pptx")
            files = [f for pat in sys.argv[3:] for f in glob.glob(pat)]
        else:
            out = os.path.join(TBL_DIR, "fairspeech_all_scales_slides.pptx")
            files = TABLES
        build_combined(files, out)
    else:
        targets = sys.argv[1:] or TABLES
        for t in targets:
            for f in glob.glob(t):
                build(f)
