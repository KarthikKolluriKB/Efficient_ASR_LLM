"""
Research presentation (PPTX) — plain black-on-white, short bullet points.
Run with the Python that has python-pptx:
  "C:/Users/Karthik/AppData/Local/Programs/Python/Python312/python.exe" \
      experiments/bias_pruning/make_presentation.py
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

BLACK = RGBColor(0, 0, 0)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def title_slide(title, subtitle):
    s = prs.slides.add_slide(BLANK)
    box = s.shapes.add_textbox(Inches(1.0), Inches(2.7), Inches(11.3), Inches(2.0))
    tf = box.text_frame; tf.word_wrap = True
    r = tf.paragraphs[0].add_run(); r.text = title
    r.font.size = Pt(32); r.font.bold = True; r.font.color.rgb = BLACK
    p = tf.add_paragraph(); r2 = p.add_run(); r2.text = subtitle
    r2.font.size = Pt(17); r2.font.color.rgb = BLACK
    ln = s.shapes.add_shape(1, Inches(1.0), Inches(4.5), Inches(5.0), Inches(0.02))
    ln.fill.solid(); ln.fill.fore_color.rgb = BLACK; ln.line.fill.background()


def slide(title, lines):
    """lines: (text, level). level 0 = bullet, 1 = sub-bullet."""
    s = prs.slides.add_slide(BLANK)
    tb = s.shapes.add_textbox(Inches(0.7), Inches(0.4), Inches(12.0), Inches(0.8))
    r = tb.text_frame.paragraphs[0].add_run(); r.text = title
    r.font.size = Pt(26); r.font.bold = True; r.font.color.rgb = BLACK
    rule = s.shapes.add_shape(1, Inches(0.7), Inches(1.12), Inches(12.0), Inches(0.02))
    rule.fill.solid(); rule.fill.fore_color.rgb = BLACK; rule.line.fill.background()

    body = s.shapes.add_textbox(Inches(0.85), Inches(1.45), Inches(11.7), Inches(5.6))
    tf = body.text_frame; tf.word_wrap = True
    first = True
    for item in lines:
        text = item[0]; level = item[1] if len(item) > 1 else 0
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = level
        mark = "•  " if level == 0 else "–  "
        run = p.add_run(); run.text = mark + text
        run.font.size = Pt(20 if level == 0 else 17)
        run.font.color.rgb = BLACK
        p.space_after = Pt(9 if level == 0 else 4)


# 1 — Title
title_slide(
    "Fairness of Encoder Pruning in SLAM-ASR",
    "Does WER-preserving pruning hide or amplify demographic bias? — Whisper + Qwen2.5-3B",
)

# 2 — Motivation
slide("Motivation", [
    ("Pruning shrinks Speech LLMs — cheaper, faster deployment.", 0),
    ("A prune is judged \"safe\" by one number: average WER.", 0),
    ("The average can hide harm to specific groups.", 0),
    ("Risk: an unfair model ships, unnoticed.", 0),
])

# 3 — Research Questions
slide("Research Questions", [
    ("RQ1 — Does WER-preserving pruning stay fair, or widen group gaps?", 0),
    ("RQ2 — Is the effect scale-dependent (small vs medium vs large)?", 0),
    ("RQ3 — Which demographic axes are affected?", 0),
    ("RQ4 — Does it generalise across languages?", 0),
    ("RQ5 — Can LoRA / mitigation compensate?", 0),
])

# 4 — Objectives
slide("Objectives", [
    ("Build per-group (disaggregated) evaluation of pruned SLAM-ASR.", 0),
    ("Measure how the group gap shifts across pruning depths.", 0),
    ("Compare across model scales and across languages.", 0),
    ("Expose disparate impact hidden by aggregate WER.", 0),
])

# 5 — Methodology
slide("Methodology", [
    ("SLAM-ASR: Whisper encoder -> projector -> Qwen2.5-3B (encoder + LLM frozen; train projector).", 0),
    ("Prune top encoder layers; retrain projector per kept depth.", 0),
    ("Depth sweep every 2 layers; zero-shot evaluation on each test set.", 0),
    ("Scales: small (12L), medium (24L), large-v2 (32L).", 0),
    ("Metrics: per-group WER; GAP (worst-best) and RATIO (worst/best).", 0),
    ("Significance: paired difference-in-differences bootstrap.", 0),
])

# 6 — Datasets used
slide("Datasets Used", [
    ("Fair-Speech (Meta) — US English, 30k utts.  [Headline]", 0),
    ("Common Voice 22 — English, Danish, Dutch.", 0),
    ("L2-ARCTIC — non-native English (illustration; small cells).", 0),
    ("Analysable threshold: >= 200 utts AND >= 30 min per group.", 0),
])

# 7 — Demographic axes
slide("Demographic Axes", [
    ("Race / ethnicity — Fair-Speech.   [Headline]", 0),
    ("Socioeconomic status (SES) — Fair-Speech.", 0),
    ("Accent — Common Voice; L2-ARCTIC.", 0),
    ("Native language (L1) — L2-ARCTIC.", 0),
    ("Gender, Age — all datasets.", 0),
    ("Danish / Dutch: only gender, age, accent (no race / SES in Common Voice).", 0),
])

# 8 — Findings
slide("Findings", [
    ("Raw gap grows on EVERY scale -> mechanical, not real bias.", 0),
    ("Ratio is the honest measure (controls for overall difficulty).", 0),
    ("Small + medium: ratio SHRINKS under pruning.", 0),
    ("Large-v2: ratio GROWS -> amplification is scale-dependent.", 0),
    ("\"WER-preserving\" prune hides the harm (large-v2 only).", 0),
    ("SES + accent corroborate; gender + age null; cross-lingual null.", 0),
])

# 9 — Results
slide("Results (key numbers)", [
    ("Black/Asian ratio under a light prune:", 0),
    ("small 2.03x -> 1.70x   |   medium 2.16x -> 1.94x   |   large 1.98x -> 2.15x", 1),
    ("Large-v2 keep-30: average 21.6 -> 21.1, but Black 27.2 -> 28.1 (p = 0.018).", 0),
    ("Large-v2 keep-28: Black +7.1 vs Asian +2.6 WER (2.8x faster).", 0),
    ("Cross-lingual relative degradation: English x2.81, Dutch x2.94, Danish x1.88.", 0),
    ("-> Languages degrade proportionally (no cross-lingual amplification).", 1),
])

# 10 — Next plan
slide("Next Plan", [
    ("LoRA: does it compensate or amplify?", 0),
    ("Compare non-LoRA vs LoRA pruned models per group — does it recover the hurt group?", 1),
    ("Mitigation ideas:", 0),
    ("Fairness-aware pruning (keep layers important for under-served groups).", 1),
    ("Group-balanced projector retraining.", 1),
    ("Targeted LoRA for the worst-hit group; distillation from unpruned teacher.", 1),
    ("Multilingual: extend da/nl per-group (gender/age/accent only).", 0),
    ("Strengthen: multi-seed at keep-30; add CORAAL (2nd race set); re-run broken checkpoints.", 0),
])

out = "experiments/bias_pruning/Pruning_Fairness_Research.pptx"
prs.save(out)
print(f"Saved {len(prs.slides._sldIdLst)} slides -> {out}")
