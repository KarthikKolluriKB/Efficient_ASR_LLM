# Fig 2 — The trend: harm grows monotonically across the usable range

## Why this figure exists

Fig 1 alone is one prune step — a skeptic calls it noise. This figure shows the
Black–Asian separation **widening at five independently trained pruned
models** (keep 32→24), which is what elevates the hook from anecdote to trend.
It also visually defines the "usable range" and the "WER-preserving regime,"
two scoping concepts the text leans on repeatedly.

**Reader's one sentence:** "The gap widens at every prune step while the model
still works."

## Why this form: two-series line chart with a shaded regime band

- The job is **change-over-depth of two entities** → lines. Plotting Black and
  Asian WER as separate lines (rather than plotting the gap Δ directly) shows
  *both* phenomena at once: everyone eventually degrades, AND the lines fan
  apart. A gap-only line would hide that the spread opens at both ends
  (Asian pulls below the aggregate while Black pulls above).
- The aggregate (ALL) rides along as a dashed neutral line — it is context, not
  a protagonist.
- The WER-preserving regime (keep 32–30, where aggregate ≤ baseline) is a shaded
  vertical band: this is where Fig 1 lives, and where the mitigation (RQ3) aims.

## Data (large-v2, Fair-Speech ethnicity; usable range keep ≥ 24)

| keep | Black | Asian | ALL | Δ (pp) |
|---:|---:|---:|---:|---:|
| 32 | 27.21 | 13.71 | 21.58 | 13.5 |
| 30 | 28.12 | 13.05 | 21.14 | 15.1 |
| 28 | 34.34 | 16.26 | 25.16 | 18.1 |
| 26 | 42.80 | 22.18 | 31.98 | 20.6 |
| 24 | 51.00 | 26.49 | 37.57 | 24.5 |

Source: `largev2_fairspeech_sweep` per-depth multiaxis CSVs (or
`results/largev2_fairspeech_sweep/findings_ethnicity.csv`).

## Construction directions

1. **x-axis: layers kept, DESCENDING left→right** (32 → 24), so "more pruning"
   reads left-to-right like a story. Label: "encoder layers kept (pruning →)".
2. **Series:** Black = red `#e34948`, solid, 2 px, round markers ≥8 px at the five
   points; Asian = blue `#2a78d6`, same treatment; ALL = gray `#898781`, dashed,
   no markers (context line).
3. **Direct labels at right line-ends:** "Black", "Asian", "ALL (aggregate)" —
   no floating legend box needed if all three are end-labeled (still include the
   legend if the caption font is small; both is acceptable, box-less).
4. **Shaded band** over keep ∈ [32, 30]: fill `#cde2fb` at ~45 % alpha, label
   inside top: "WER-preserving" in 7 pt secondary ink. This band is where the
   aggregate is flat-or-better yet the red line already rises — the visual echo
   of Fig 1.
5. **Gap annotation, selective:** bracket the vertical distance at keep 32
   ("13.5 pp") and keep 24 ("24.5 pp") only — two brackets, muted ink. Do NOT
   label the gap at every depth.
6. y-axis: "WER (%)", 0–55, ticks every 10. Start at 0 (bars aren't involved,
   but a zero-based axis avoids exaggerating the fan-out — credibility matters
   more than drama here; the fan-out is big enough anyway).
7. Size: single column 3.35 × 2.5 in.

## Caption skeleton

> **Figure 2:** WER for Black and Asian speakers (Fair-Speech) across five
> independently trained pruned large-v2 models. The absolute gap widens
> monotonically (13.5 → 24.5 pp) across the entire range in which the model
> remains usable; in the shaded WER-preserving regime the aggregate (dashed) is
> flat-to-better while Black WER already rises. Relative gap ρ peaks at keep-30
> (2.15) and compresses under global collapse — reported in Table 1.

## Anti-pattern checklist

- ✗ Do NOT overlay ρ on a second y-axis (dual-axis ban). ρ's early-peak story
  goes to Table 1 / text, or a tiny separate panel if truly needed.
- ✗ No markers/labels on the ALL line — it must recede.
- ✓ 2-series + context; legend or end-labels; zero-based y.

## Variant if space is tight

Merge with Fig 1: two-panel single-column stack (a: diverging bars; b: this).
Costs some hook impact; acceptable if page limit forces it.
