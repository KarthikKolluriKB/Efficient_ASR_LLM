# Fig 5 — Mechanism schematic: spare capacity is the hiding place

## Why this figure exists

Figures 1–4 establish WHAT happens; this one proposes WHY, and the "why" is the
paper's conceptual contribution: **the capacity that lets a large model absorb
pruning without aggregate loss is the same capacity that lets it relocate the
loss onto a subgroup unnoticed.** Reviewers remember papers whose mechanism they
can redraw on a whiteboard; this is that drawing. It also sets up RQ3: the
mitigation (tail-aware recovery) is legible only once the reader sees that
mean-based recovery abandons the tail.

**Reader's one sentence:** "Big models have tail-only layers; small models'
layers are shared by everyone — so only big models can hide the harm."

This is a SCHEMATIC (drawn shapes, no data axes) — the one non-data figure the
paper gets. Keep it austere so it reads as explanation, not decoration.

## Layout (two columns, mirrored)

```
   LARGE ENCODER (32L)                SMALL ENCODER (12L)
   "has spare capacity"               "no spare capacity"

   ┌──────────────┐ ← pruned          ┌──────────────┐ ← pruned
   │  tail-only   │   (top layers)    │   shared     │   (top layers)
   ├──────────────┤                   ├──────────────┤
   │              │                   │              │
   │  kept: used  │                   │  kept: used  │
   │  by ALL      │                   │  by ALL      │
   │              │                   │              │
   └──────────────┘                   └──────────────┘

   avg WER   −0.4 pp ✓                avg WER   +4.7 pp ✗
   Black     +0.9 pp ✗                worst grp  worse  ✗
   → harm is HIDDEN                   → harm is VISIBLE
```

The single load-bearing word pair: the pruned band is labeled **"tail-only"**
on the large model and **"shared"** on the small one. Everything else is
identical between the two columns — the contrast IS the argument.

## Construction directions

1. **Vector, not raster:** TikZ (preferred, native LaTeX) or hand-built SVG →
   PDF. No screenshots, no 3-D, no gradients, no icons.
2. **Two encoder stacks** drawn as rounded rectangles of proportional heights
   (32 vs 12 units tall is too extreme to draw literally; draw ~5:3 and note
   layer counts as text). Top band (the pruned layers) in a warm tone
   (`#eda100` tint or hatch); kept body in blue tint `#cde2fb` with outline.
   A small "✂ pruned" marker at each top band.
3. **Outcome mini-table** under each stack (two rows: avg WER, worst group),
   with ✓/✗ set in text ink — NOT colored green/red boxes (status color
   inflation). The numbers are real (from Figs 1/3 data) — that's what keeps a
   schematic honest: −0.4/+0.9 pp (large), +4.7 pp (small).
4. **Verdict line** under each column, italic, secondary ink: "gain is real —
   harm is hidden" / "harm shows up in the average".
5. **No arrows between the columns.** Mirroring does the comparison; arrows
   would imply a process flowing left→right that doesn't exist.
6. Size: single column 3.35 in wide; ~2.8 in tall.

## Placement & text hook

Place at the top of the Discussion/Mechanism subsection. The paragraph that
references it should also make the RQ3 bridge explicit: recovery training
optimizes a MEAN, so the surviving capacity is re-allocated to the majority —
which is why the mitigation de-averages the recovery objective.

## Honesty framing (must appear in caption or text)

The "top layers are tail-only" attribution is an *interpretation consistent
with* the measurements (aggregate flat + tail regressing on large; everything
regressing on small), not itself directly measured. Caption should say
"schematic illustration of the proposed mechanism" — reviewers punish schematics
passed off as evidence.

## Caption skeleton

> **Figure 5:** Proposed mechanism (schematic). A large encoder holds capacity
> the majority of speech does not need; pruning removes layers that were
> load-bearing only for atypical ("tail") speech, so aggregate WER — dominated
> by the majority — holds while worst-group WER degrades. A small encoder has no
> such slack: every layer is shared, so pruning visibly costs aggregate WER and
> nothing is hidden. Numbers from Figs. 1 and 3.

## Anti-pattern checklist

- ✗ No decorative icons (people, microphones); shapes + text only.
- ✗ No green "good" / red "bad" fills on the outcome rows — ink ✓/✗ only.
- ✓ Real numbers embedded; interpretation labeled as proposed, not proven.
