# Fig 4 — RQ2: LoRA improves aggregate accuracy while widening the racial ratio

## Why this figure exists

RQ2's result is the paper's second punch: the field's default repair (LoRA)
improves aggregate WER at every depth **while widening the racial ratio at every
depth — including unpruned**. The finding is intrinsically two-directional
(one metric down, another up), which prose states weakly and a single chart
cannot legally show (dual-axis ban). The two-panel form makes the trade visible:
same two entities, same colors, opposite verticals.

**Reader's one sentence:** "LoRA moves the average down and the ratio up."

## Why this form: two side-by-side panels, shared x, one entity-pair

- Panel A: aggregate WER vs depth — LoRA below base everywhere (the seduction).
- Panel B: ρ (Black/Asian) vs depth — LoRA above base everywhere (the price).
- Two measures of different scale → **two panels, never two y-axes**. Color
  follows the entity across both: base = blue, +LoRA = red. The reader learns
  the pair once and reads both panels instantly.
- Large-v2 is the shown scale (headline); medium replicates and goes to the
  appendix or one caption sentence ("identical pattern on medium, incl.
  unpruned: 2.16→2.40").

## Data (large-v2, Fair-Speech, usable range)

| keep | agg base | agg LoRA | ρ base | ρ LoRA |
|---:|---:|---:|---:|---:|
| 32 | 21.6 | 17.7 | 2.03 | 2.20 |
| 30 | 21.1 | 18.9 | 2.15 | 2.35 |
| 28 | 25.2 | 22.3 | 2.11 | 2.30 |
| 26 | 32.0 | 28.2 | 1.93 | 2.25 |
| 24 | 37.6 | 33.0 | 1.93 | 2.23 |

Source: `largev2_fairspeech_sweep` vs `largev2_fairspeech_LORA_sweep`.
(ρ base uses worst/best analysable; at keep-32 the dynamic best is Native
Hawaiian → 2.03. If the builder prefers strictly-fixed Black/Asian groups,
keep-32 base is 1.98 — EITHER is defensible but the figure and Table 1 must use
the SAME convention. Recommend fixed Black/Asian everywhere: 1.98.)

## Construction directions

1. **Layout:** full-width figure 6.9 × 2.4 in, two panels, shared x-axis
   "encoder layers kept (pruning →)" descending 32→24. Panel titles set as
   small bold headers: "(a) Aggregate WER (%)" / "(b) Black/Asian WER ratio ρ".
2. **Series:** base = blue `#2a78d6` solid; +LoRA = red `#e34948` solid; both
   2 px with ≥8 px markers at each depth. Same colors in BOTH panels. One
   legend, top-right of panel A only; end-labels "base"/"+LoRA" in panel B.
3. **The delta shading (optional but strong):** in each panel, fill the vertical
   region between the two lines at ~12 % alpha of the upper line's hue — in A
   the fill reads "WER saved", in B "fairness cost". If it clutters at this
   size, drop it from A and keep it only in B (the price is the message).
4. **Unpruned marker:** at keep-32 in panel B, a small ring around both points
   with a 7 pt note "worsens even unpruned" — this preempts the strongest
   counter-read ("it's a pruning interaction, not LoRA itself").
5. y-ranges: A: 15–40 %; B: 1.8–2.5 (non-zero baseline is fine for ρ — a ratio
   has a natural reference of its own baseline; add a hairline at base keep-32
   value for anchoring).
6. Grid: horizontal hairlines only, both panels.

## Caption skeleton

> **Figure 4:** The standard repair does not restore parity. Applying LoRA to the LLM on top
> of pruned encoders (large-v2, Fair-Speech) reduces aggregate WER at every
> depth (a) while *increasing* the Black/Asian error ratio at every depth (b) —
> including with no pruning at all (keep-32) — because average-loss adaptation
> allocates its gains preferentially to already-well-served groups. Medium
> replicates the pattern at all seven matched depths (2.16→2.40 unpruned).

## Anti-pattern checklist

- ✗ NEVER one panel with WER on the left axis and ρ on the right (dual-axis ban
  — and this figure is the tempting case; resist).
- ✗ No green/red status coloring of "good/bad" lines — blue/red are entity
  colors here (base vs LoRA), consistent across panels.
- ✓ One legend + selective end labels; markers on every real data point.

## Variant

If the paper needs to save a full-width slot: stack the panels vertically in a
single column (3.35 × 4.2 in). Keep shared x and the entity-color pairing.
