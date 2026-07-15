# Reconciling the PPT figures with the paper figure plan (brainstorm, 2026-07-10)

Input: `P:\KTP\Monthly Meetings\Bias pruning project.pptx` (claims on slides 4–8,
tables 9–13, rendered figures 14–15) vs the specs in this directory.

## 1. Claim → artifact map

Every claim in the PPT, and the artifact that should carry it in the paper:

| Claim (PPT wording) | Carrier | Status |
|---|---|---|
| Black baseline 2× Asian; only Black rises under a "safe" prune | **Fig 1** diverging bars (hook) | NOT rendered yet — highest priority |
| Gap widens monotonically with depth | **Fig 2** depth trend (Black/Asian/Agg, large-v2) | partially rendered (buried in 3-panel) |
| Hidden amplification only on large-v2 (scale-dependent) | **Fig 3** ρ-vs-%-pruned overlay | not rendered; PPT 3-panel doesn't carry it (see §2) |
| SES: ratio 1.36, gap 6→9.5 pp, same direction | Table 2 row + one text sentence | no figure needed |
| Dutch accent widens but scale-INVARIANT, never hidden | Fig 6 facets (optional) or Table 2 row | cut first if space |
| Danish: model too weak, no signal | one text sentence | never a figure |
| Gender/age: no claim | Table 2 "not claimed" rows | honesty device |
| LoRA lowers average, widens racial ratio at every depth incl. L-0 | **Fig 4** two-panel (rendered, slide 15) | done, minor fixes |
| LoRA bias race-specific (absent on accent, all 3 languages) | Table 2 column or Fig 4 caption sentence | cheap, add |

## 2. Critique of the two rendered figures

### RQ1 3-panel, 6 groups (slide 14)
- Six lines × three panels = spaghetti; Hispanic/White/Native-Am./MENA carry no
  claim and dilute the red line.
- Each panel has its own y-range AND raw "layers removed" x — the cross-scale
  comparison (the actual novel claim) is visually impossible: 2/12 layers ≠ 2/32.
- The entire "hidden" story lives in one small shaded band + 3-line annotation
  on panel (c). Fails the one-sentence test: a fresh reader says "everything
  gets worse everywhere," not "only the big model hides it."
- Verdict: **keep as the appendix evidence figure** (it is honest and complete);
  do not make it the claim-carrier. For the main text, slim to Black/Asian/Agg
  (= spec Fig 2) and let Fig 3 carry scale.

### RQ2 two-panel (slide 15)
- Matches spec 04; the one-sentence test passes ("LoRA moves the average down
  and the ratio up"). Keep.
- Fixes: (a) ring/annotate the L-0 points in panel (b) — "widens even unpruned"
  preempts the pruning-interaction counter-read; (b) settle the ρ convention:
  fixed Black/Asian everywhere (baseline 1.98) — figure, Table 1, text must
  agree; (c) the slide's handwritten note "(b) relative to their L0 baseline"
  = normalization idea → answered better by the gains-allocation panel (§3B)
  than by re-baselining the ratio.

### PPT tables (slides 9–12)
- Full 7-group × 5-depth grids per scale → appendix only. Main text gets
  Table 1 (large-v2 per-depth, with Δ, ρ, paired-bootstrap p) + Table 2 (scope
  grid with the two-YES "hidden?" column).
- Small-model L-8 column (WER 92–104 %) is beyond the usable range — exclude
  from every paper artifact; state usable-range rule once.
- Slide 9's "Relative increase? – bring it to same scale" note = exactly what
  spec Fig 3's normalized-% axis + ρ metric solve.

## 3. New ideas from this brainstorm (not in specs 01–07)

### A. "Hidden-harm quadrant" scatter — candidate replacement for Fig 3
One panel. x = Δ aggregate WER (pp vs own baseline), y = Δ worst-group (Black)
WER. One dot per (model, depth); shapes/colors by scale. Reference lines x=0,
y=0 and the diagonal y=x.
- Upper-left quadrant (agg improves, Black worsens) = **hidden harm as a
  geometric region**. Only large-v2 keep-30 lands there.
- Above-diagonal = disproportionate harm even when visible.
- Generalizes: Dutch accent dots can join the same plot (they hug the diagonal,
  never enter the quadrant) — could absorb Fig 6 too, saving a figure slot.
- Risk: more abstract; needs a reader to decode axes first. **Prototype both
  this and spec-03; keep whichever passes the one-sentence test with fresh eyes.**

### B. LoRA gains-allocation bars — optional Fig 4 panel (c)
Per-group WER improvement from LoRA (base − LoRA, pp) at the free-prune depth:
Asian gain vs Black gain side by side (memory: advantaged group gains ~2×).
Makes the mechanism ("average-loss adaptation serves the well-served") tangible
and answers the slide-15 normalization note. If it crowds the figure, one text
sentence with the two numbers suffices.

### C. Δρ heatmap (axes × scale grid, cells = amplification)
Graphical Table 2. Rejected for main text: the table version is auditable and
the "hidden?" column is editorial, which a color ramp would oversell. Revisit
only for the talk/poster, not the paper.

## 4. Recommended final lineup (workshop, RQ1+RQ2 only, deadline 2026-07-25)

| Slot | Artifact | Source |
|---|---|---|
| Fig 1 (page 1) | diverging bars, free prune, agg −0.44 vs Black +0.91 | spec 01 — **render next** |
| Fig 2 | Black/Asian/Agg vs depth, large-v2, shaded free-prune band | spec 02 (slim of current 3-panel) |
| Fig 3 | scale contrast: ρ overlay (spec 03) **or** quadrant scatter (§3A) | prototype both |
| Fig 4 | LoRA two-panel | rendered; fixes §2 |
| Fig 5 | mechanism schematic (tail-only vs shared layers) | spec 05, strong keep |
| Table 1 | large-v2 per-depth audit numbers | spec 07 |
| Table 2 | scope grid, two YES cells | spec 07 |
| Appendix | 6-group 3-panel (slide 14), per-scale grids (slides 10–12), Dutch facets | existing |

Cut order if pages run out: Fig 6 already cut → quadrant absorbs cross-lingual;
then Fig 5 schematic → two text sentences; never cut Fig 1 or Fig 4.

## 5. Consistency rules (bind every artifact)

1. ρ = fixed Black/Asian everywhere; large-v2 baseline **1.98**.
2. Usable range only: agg WER ≤ ~40 %; small L-8 etc. excluded everywhere.
3. Color follows entity: Black=red, Asian=blue, Aggregate=gray-dashed in every
   figure; base-vs-LoRA pair keeps its own two colors in both Fig 4 panels.
4. Scale comparisons always on %-of-layers-pruned axis, never raw layers.
5. All figures + tables generated from CSVs by script, plotted values printed
   to stdout (auditable against ANALYSIS_MASTER.md).
