# Fig 6 (optional) — Cross-lingual: not every gap behaves like race

## Why this figure exists (and why it's optional)

The workshop explicitly lists cross-lingual/multilingual fairness as a topic;
this figure is the visual proof that the paper engages it. Its scientific value
is as a **contrast**: the Dutch accent gap widens with pruning but is
**scale-INVARIANT** (all three model sizes behave alike), unlike race, which is
scale-dependent — so the paper isn't claiming one universal law but mapping
when the hiding phenomenon appears (needs a capacity buffer + headroom) and
when it doesn't (Danish: model too weak; Dutch accent: harm present at all
scales, hidden at none).

It is OPTIONAL because the same content compresses into 3 sentences + one row
of Table 2 if pages run out. Cut this before cutting anything else.

**Reader's one sentence:** "Dutch accent harm grows the same at every scale —
unlike race, which only the big model amplifies."

## Why this form: small multiples (three mini-panels), one per scale

- The claim is "three curves have the SAME shape" → small multiples are the
  honest form: same axes, same ranges, one panel per scale, letting the eye
  verify sameness. Overlaying all three on one plot (as in Fig 3) would work
  too, but the deliberate contrast WITH Fig 3 is the point: Fig 3 overlays to
  show divergence; Fig 6 facets to show sameness. (Same data-job difference,
  different form.)

## Data (CV22-NL, Belgian vs Netherlands Dutch accent ρ, usable range)

**large-v2:** (0%, 1.19) (6.25%, 1.18) (12.5%, 1.19) (25%, 1.13) (31%, 1.23) (44%, 1.24) (50%, 1.27)
**medium:** (0%, 1.13) (8.3%, 1.20) (16.7%, 1.15) (25%, 1.13) (33%, 1.21) (50%, 1.26) (58%, 1.32)
**small:** (0%, 1.18) (16.7%, 1.16) (33%, 1.21) (50%, 1.27)

x = fraction of layers pruned. Note the Dutch usable range extends deeper than
Fair-Speech's (aggregate stays workable longer), so x runs to ~50 %.

Danish is NOT plotted: gender ρ 1.00–1.02 throughout, accent unanalysable —
Danish appears as one caption/text sentence ("the low-resource null: aggregate
WER ≥ 35 % unpruned leaves no headroom for differential harm").

## Construction directions

1. Three panels in a row, single-column width total (each ~1.05 in wide) OR one
   overlay panel if the row is too cramped at 3.35 in — try faceted first.
2. All panels: identical y-range (1.0–1.4), identical x (0–60 %), y gridlines
   at 1.1/1.2/1.3 only. Panel headers: "small · 12L", "medium · 24L",
   "large-v2 · 32L" in 7 pt bold.
3. One hue for all panels — the entity is the SAME (Belgian-accent gap), so it
   keeps one color across facets: blue `#2a78d6`. Do NOT reuse the Fig 3
   scale-color coding here (that coding meant "which model"; here the panel
   header carries that, and repainting would break color-follows-entity).
4. A dotted horizontal hairline at each panel's own baseline ρ (its 0 % value)
   so "ends above where it started" is readable per panel.
5. End-label the final point of each panel with its ρ value (1.27 / 1.32 / 1.27)
   — three labels total, no other numbers.
6. Size: 3.35 × 1.9 in.

## Caption skeleton

> **Figure 6:** Belgian-vs-Netherlands Dutch accent WER ratio under pruning, per
> model scale (CV22-NL). The gap widens with depth at every scale by a similar
> amount — scale-invariant, in contrast to the scale-dependent race effect of
> Fig. 3 — and is never hidden (aggregate WER rises from the first prune at all
> scales). Danish, the low-resource case, shows no measurable disparity at any
> depth: with aggregate WER ≥ 35 % before pruning, there is no headroom for
> differential harm.

## Anti-pattern checklist

- ✗ Don't color the three panels differently (entity is the same everywhere).
- ✗ Don't put Danish in as a flat-line panel — an unanalysable corpus drawn as
  a curve overstates what was measured; it belongs in text.
- ✓ Identical axes across facets (the sameness claim depends on it).

## Data dependency

If the DA/NL **LoRA** evals land before the freeze (2026-07-15), a companion
row in Table 2 ("does LoRA widen the Dutch accent ratio too?") is cheap to add
and would extend RQ2 cross-lingually — but do not hold this figure for it.
