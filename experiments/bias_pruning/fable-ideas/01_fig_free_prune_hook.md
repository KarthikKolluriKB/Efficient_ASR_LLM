# Fig 1 — The hook: one "free" prune, opposite directions

## Why this figure exists

This is the paper's thesis in a single image. The canonical "free" prune
(large-v2, keep 32→30) **improves the aggregate while worsening exactly one
group**. Prose can state it; only a picture makes the reader *feel* that the
average points the wrong way. It must be Figure 1, placed on page 1, ideally
reachable from the abstract ("see Fig. 1").

**Reader's one sentence:** "The average improved, but Black speakers got worse."

## Why this form: horizontal diverging bar chart

- The data's job is **polarity** (improved vs worsened per group) → the correct
  form is a diverging bar around zero, using the diverging pair (blue↔red with
  a neutral zero line). A grouped before/after bar chart would bury the sign —
  the whole point — in a length *comparison*; the diverging form encodes the
  sign directly as direction.
- Horizontal, because group names are long ("Native Hawaiian / Pac. Isl.") and
  horizontal bars give them a natural left rail without rotated text.

## Data (Fair-Speech ethnicity, large-v2, keep 32 → keep 30)

Source: `largev2_fairspeech_sweep/per_axis/d00_keep32_…_multiaxis.csv` and
`d02_keep30_…_multiaxis.csv`. Signed ΔWER in percentage points:

| group | ΔWER (pp) | direction |
|---|---:|---|
| Native American | −2.48 | improves |
| Hispanic / Latino | −0.84 | improves |
| Asian / South Asian | −0.66 | improves |
| Middle Eastern / N. African | −0.62 | improves |
| White | −0.48 | improves |
| **ALL (aggregate)** | **−0.44** | improves |
| Native Hawaiian / Pac. Isl. | +0.29 | worsens (low-power) |
| **Black / African American** | **+0.91** | worsens |

## Construction directions

1. **Order rows by ΔWER** (most-improved at top, Black at the bottom) so the eye
   travels down into the regression. Zero line = solid hairline, slightly darker
   than grid (`#c3c2b7`), full height.
2. **Color:** improvements blue `#2a78d6`; regressions red `#e34948`. The ALL
   bar gets the same blue (it improves) but is **visually distinguished as the
   aggregate**: bold label + a pale outline or `ALL` set in bold ink — never a
   third hue (it is not a third category of thing).
3. **Direct labels, selectively:** value labels (−0.44, +0.91) ONLY on the two
   bars that carry the story — ALL and Black. Other bars are read from the axis.
   Labels sit outside the bar end (bars are short), in primary ink, never inside.
4. **Annotation (the payload):** a thin callout bracketing ALL and Black:
   *"aggregate improves ↓ while Black speakers regress ↑"* — one line, 7 pt,
   secondary ink. This is the only annotation; resist adding more.
5. x-axis: "ΔWER after pruning 2/32 layers (pp)", range approx −2.8 … +1.2,
   ticks at −2, −1, 0, +1. Gridlines vertical hairlines only.
6. Size: single column, 3.35 × ~2.3 in. Eight thin bars, 2 px surface gaps.

## Caption skeleton

> **Figure 1:** Per-group change in WER when pruning Whisper large-v2 from 32 to
> 30 encoder layers (Fair-Speech). Aggregate WER *improves* (−0.44 pp) while WER
> for Black speakers *worsens* (+0.91 pp; paired bootstrap p < .05): an
> aggregate-only acceptance test reports the opposite sign of the effect on the
> harmed group.

## Anti-pattern checklist

- ✗ No before/after paired bars (buries the sign).
- ✗ No third hue for ALL; no status-green for improvements (blue/red diverging).
- ✗ No value label on every bar — only ALL and Black.
- ✓ Sub-3:1 contrast not an issue here (blue and red both pass on light).

## Fallback / variant

If reviewers want uncertainty: add per-group paired-bootstrap 95 % CI whiskers
on Δ (data exists in `multiaxis_paired_d00_keep32_vs_d02_keep30.csv`). Whiskers
in muted ink, no caps. Only Black's CI excludes zero cleanly — that *strengthens*
the figure; consider including from the start if space allows.
