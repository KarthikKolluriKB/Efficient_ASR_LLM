# Fig 3 — The novel claim: only the large model hides the harm

## Why this figure exists

This carries the paper's **most novel finding**: disparity amplification under
"free" pruning is a *large-model* failure mode. Small and medium have no
WER-preserving prune (their first cut visibly costs +4.7/+7.3 pp aggregate), and
their Black/Asian *ratio* tightens under pruning; only large-v2's ratio rises —
precisely inside its free-prune window. Without this figure the scale claim is
a table of numbers; with it, the divergence of one curve from the other two is
instantly visible.

**Reader's one sentence:** "Only the largest model's ratio rises — and only it
has a free prune to hide in."

## Why this form: three-line comparison on a NORMALIZED pruning axis

- The job is **compare one trajectory across three entities (scales)** → three
  lines, one per scale, of ρ = WER_Black / WER_Asian.
- The x-axis must be **fraction of encoder layers pruned (%)**, not raw layers
  kept — 12-, 24- and 32-layer models are only comparable proportionally. This
  normalization is itself an editorial decision to state in the caption.
- Using ρ (not the absolute gap Δ) is the honesty guarantee: Δ grows on every
  scale simply because everyone degrades; ρ cancels the "everyone got worse"
  effect. The figure exists to show the *ratio* diverging — Δ's story is Fig 2.

## Data (Fair-Speech ethnicity, usable range only, agg WER ≤ ~40 %)

ρ = Black/Asian; x = layers pruned / total layers:

**large-v2 (32L):** (0 %, 1.98) (6.25 %, **2.15**) (12.5 %, 2.11) (18.75 %, 1.93) (25 %, 1.93)
**medium (24L):** (0 %, 2.16) (8.3 %, 2.01) (16.7 %, 1.94) (25 %, 1.97) — complete; keep-22 point = 2.01 (Black 29.49 / Asian 14.65)
**small (12L):** (0 %, 2.03) (8.3 %, 2.03) (16.7 %, 1.70)

Aggregate-WER context (for the annotation, not plotted): first prune costs
−0.44 pp (large-v2), +7.3 pp (medium, 24→20 currently), +4.7 pp (small).

## Construction directions

1. **x-axis:** "encoder layers pruned (%)", 0–26 %. **y-axis:** "WER ratio,
   Black / Asian (ρ)", range ~1.5–2.5, reference hairline at the large-v2
   baseline is NOT needed — each curve's own start is its reference.
2. **Series colors (validated set, fixed order):** large-v2 = blue `#2a78d6`;
   medium = yellow `#eda100`; small = aqua `#1baf7a`. Yellow and aqua are
   sub-3:1 on light surface → **direct end-labels on every line are mandatory**
   (relief rule), plus distinct markers (circle / square / triangle) as the
   secondary encoding.
3. **Emphasis:** large-v2 is the protagonist — full saturation, 2.5 px; medium
   and small at 2 px. Do not gray them out entirely (their *falling* is half
   the finding), but the blue line should read first.
4. **The key annotation:** shade large-v2's WER-preserving window (0–6.25 % on
   the x axis... careful: the window is a property of large-v2 only, so use a
   horizontal bracket UNDER the blue curve from 0 to 6.25 % labeled
   "large-v2's 'free' prune — ρ rises here", not a full-height band (a band
   would falsely apply to all three curves).
5. **Second annotation (one line, secondary ink), near small/medium curves:**
   "no WER-preserving prune exists at these scales (first cut: +4.7 to +7.3 pp
   aggregate)". This is the mechanism hook that Fig 5 elaborates.
6. Marker at EVERY plotted point (few points per curve; the sweep depths are
   discrete models, and readers must see the data density honestly).
7. Size: single column 3.35 × 2.5 in.

## Caption skeleton

> **Figure 3:** Black/Asian WER ratio ρ vs. fraction of encoder layers pruned,
> per model scale (Fair-Speech, usable range). Only large-v2 shows ρ rising
> (1.98→2.15), and it does so exactly within its WER-preserving window; small
> and medium — which have no aggregate-neutral prune to hide in — show flat or
> tightening ratios. Pruning fractions are normalized per scale; medium's first
> prune point (keep-22, ρ = 2.01) confirms medium does not amplify.

## Anti-pattern checklist

- ✗ No raw-layers x-axis (incomparable scales) — normalize, and say so.
- ✗ No full-height regime band (the regime belongs to one curve only).
- ✓ Direct labels required on aqua/yellow (validator WARN honored).
- ✓ Markers as secondary encoding; colors in validated fixed order.

## Honest-scope note for the builder

RESOLVED (2026-07-07): the medium keep-22 point landed at ρ = 2.01 (down from
2.16 at baseline) — medium does NOT amplify at its first prune, so the
"only large-v2" phrasing stands as written. The figure script should still read
CSVs, not constants; note the valid keep-22 values live in the server's
`results/per_seed_wer/`, not in `medium_fairspeech_sweep/per_axis/` (which
holds the superseded invalid file).
