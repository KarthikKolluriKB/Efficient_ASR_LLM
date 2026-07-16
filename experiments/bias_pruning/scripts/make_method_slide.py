"""Two plain slides explaining how WER (%) and its bootstrap SD are computed.

Deliberately unstyled: black text on white, one thin rule under each title.
Worked example is the cell reproduced from the data:
Asian speakers, Fair-Speech, Whisper large-v2 baseline -> 13.7 +/- 0.6.

    python make_method_slide.py
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR

HERE = os.path.dirname(os.path.abspath(__file__))
TBL_DIR = os.path.normpath(os.path.join(HERE, "..", "results", "tables"))
OUT = os.path.join(TBL_DIR, "method_wer_sd_slide.pptx")

INK = RGBColor(0x00, 0x00, 0x00)
MUTED = RGBColor(0x66, 0x66, 0x66)
RULE = RGBColor(0xBF, 0xBF, 0xBF)
FONT = "Calibri"
MONO = "Consolas"

MARGIN = Inches(0.85)


def textbox(slide, x, y, w, h):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    return tf


def para(tf, text, size, bold=False, color=INK, space_after=6, first=False,
         font=FONT, space_before=0):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.space_after = Pt(space_after)
    p.space_before = Pt(space_before)
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = font
    return p


def head(slide, prs, title):
    """Plain title + a thin rule beneath it. Returns y below the rule."""
    tf = textbox(slide, MARGIN, Inches(0.6), prs.slide_width - 2 * MARGIN,
                 Inches(0.6))
    para(tf, title, 30, True, INK, first=True)
    y = Inches(1.28)
    ln = slide.shapes.add_connector(1, MARGIN, y,
                                    prs.slide_width - MARGIN, y)
    ln.line.color.rgb = RULE
    ln.line.width = Pt(0.75)
    return y


def slide_wer(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    y = head(s, prs, "How we compute WER (%)")
    w = prs.slide_width - 2 * MARGIN
    tf = textbox(s, MARGIN, y + Inches(0.35), w, Inches(5.2))

    para(tf, "For each utterance, compare the reference to the model's "
             "transcript and count word errors.", 17, first=True, space_after=14)
    para(tf, "errors  =  substitutions + deletions + insertions", 15,
         color=MUTED, font=MONO, space_after=18)

    para(tf, "Add the counts across every utterance in the group, then divide:",
         17, space_after=14)
    para(tf, "WER  =  100  ×   total errors  /  total words", 22, True,
         font=MONO, space_after=20)

    para(tf, "Example — Asian speakers, Fair-Speech, Whisper large-v2 baseline "
             "(3,854 utterances):", 15, color=MUTED, space_after=10)
    para(tf, "WER  =  100  ×  5,722 / 41,736  =  13.7 %", 22, True,
         font=MONO, space_after=22)

    para(tf, "The totals are summed first, then divided.", 16, True,
         space_after=6)
    para(tf, "This is not the average of per-utterance WERs. Summing counts "
             "weights each utterance by its length, so a 3-word utterance does "
             "not count as much as a 40-word one. This is the standard corpus "
             "WER used by NIST sclite, jiwer and Kaldi.", 14, color=MUTED,
         space_after=0)


def slide_sd(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    y = head(s, prs, "How we compute the ±  (bootstrap SD)")
    w = prs.slide_width - 2 * MARGIN
    tf = textbox(s, MARGIN, y + Inches(0.35), w, Inches(5.4))

    para(tf, "The question: how much would 13.7 move if we had happened to "
             "collect a different sample of utterances?", 17, first=True,
         space_after=14)

    para(tf, "Steps:", 16, True, space_after=7)
    para(tf, "1.   Redraw 3,854 utterances from the same group, with replacement",
         16, space_after=6)
    para(tf, "2.   Recompute the WER on that draw (same formula as before)", 16,
         space_after=6)
    para(tf, "3.   Repeat B = 1,000 times   →   13.66,  13.55,  13.45,  13.01,  …",
         16, space_after=6)
    para(tf, "4.   Take the standard deviation of those 1,000 values", 16,
         space_after=16)

    para(tf, "SD  =  √[ (1/B) · Σ (WERᵇ − mean)² ]  =  0.6",
         20, True, font=MONO, space_after=10)

    para(tf, "where    B       =  number of resamples  =  1,000", 14,
         color=MUTED, font=MONO, space_after=3)
    para(tf, "         WERᵇ    =  the WER of resample b     (b = 1 … B)", 14,
         color=MUTED, font=MONO, space_after=3)
    para(tf, "         mean    =  average of the B resampled WERs  =  13.71",
         14, color=MUTED, font=MONO, space_after=12)

    para(tf, "Reported cell:    13.7 ± 0.6", 22, True, space_after=6)
    para(tf, "≈68% of resamples fall within ±0.6 of 13.7. Black is 27.2 — about "
             "24 SDs away, so that gap is not sampling noise. No data is "
             "invented: only measured utterances are reused, and the reported "
             "WER stays the real one.", 13, color=MUTED, space_after=10)

    para(tf, "Standard practice for ASR error bars (Bisani & Ney, ICASSP 2004); "
             "agrees with the closed-form ratio-estimator SE to 0.08%. Single "
             "seed (42) — bars cover sampling uncertainty, not checkpoint "
             "variance.", 11, color=MUTED, space_after=0)


def build():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    slide_wer(prs)
    slide_sd(prs)
    prs.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
