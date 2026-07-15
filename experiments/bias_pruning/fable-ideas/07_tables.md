# Tables — the numbers reviewers will check

Figures persuade; tables get audited. Two tables, both compact.

## Table 1 — RQ1 headline: per-depth race numbers (large-v2)

**Why:** every number behind Figs 1–3 in one auditable place, including the
Δ-vs-ρ divergence that the figures deliberately split up (Fig 2 shows Δ's
monotone growth; ρ's early peak is stated here, avoiding the dual-axis trap).
Also carries the paired-bootstrap significance that the figures only allude to.

| kept | agg WER | Black | Asian | Δ (pp) | ρ | agg Δ vs base | Black Δ p |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 21.58 | 27.21 | 13.71 | 13.5 | 1.98 | — | — |
| 30 | 21.14 | 28.12 | 13.05 | 15.1 | 2.15 | **−0.44** | <.05 |
| 28 | 25.16 | 34.34 | 16.26 | 18.1 | 2.11 | +3.58 | <.001 |
| 26 | 31.98 | 42.80 | 22.18 | 20.6 | 1.93 | +10.40 | <.001 |
| 24 | 37.57 | 51.00 | 26.49 | 24.5 | 1.93 | +15.99 | <.001 |

- Bold the −0.44 (the "free" prune) and the 2.15 (ρ peak).
- p-values from `multiaxis_paired_*.csv` (paired bootstrap, per-group Δ) —
  builder must pull exact values; the <.05/<.001 above are placeholders to
  verify, not to print unverified.
- Fixed groups (Black, Asian) throughout — same convention as Fig 4 (see that
  spec's note on the keep-32 1.98-vs-2.03 discrepancy; pick fixed groups and be
  consistent in every artifact).

## Table 2 — Secondary axes & scales: where is harm, where is it hidden?

**Why:** the honesty table. It scopes the claim — which axes/scales show
amplification, which show it *hidden*, and what the evidence grade is. This
single table preempts the three likeliest reviewer objections (overclaiming
axes, overclaiming scales, L2-ARCTIC noise).

| axis (dataset) | scale | inherited ρ | induced Δ (usable) | hidden under avg? | evidence |
|---|---|---:|---:|:--:|---|
| race (Fair-Speech) | large-v2 | 1.98 | +10.7 pp | **YES** | strong (5 depths, monotone) |
| race (Fair-Speech) | medium | 2.16 | +6.3 pp | no | strong |
| race (Fair-Speech) | small | 2.03 | +3.2 pp | no | strong |
| SES (Fair-Speech) | large-v2 | 1.36 | +3.7 pp | **YES** (onset +1 step) | moderate |
| accent (CV22-EN) | large-v2 | 1.43 | +3.8 pp | no | moderate |
| accent (CV22-NL) | all three | 1.13–1.19 | ~+0.1 ρ each | no | moderate, scale-invariant |
| L1 (L2-ARCTIC) | large-v2 | 3.42 | +15.6 pp | no | **illustrative only** |
| gender / age | all | ~1.0–1.3 | inconsistent | — | not claimed |
| any (Danish) | all | ρ ≈ 1.0 | none | — | null (low-resource) |

- The "hidden?" column is the paper's sharpest editorial device: exactly two
  YES cells, both Fair-Speech + large-v2. That restraint is what makes the
  headline believable.
- "induced Δ" = worst-usable-prune gap minus baseline gap (ANALYSIS_MASTER §2/§5
  has the derivations).

## RQ2 numbers live in Fig 4 + one sentence

A third table (base vs LoRA per depth) is NOT needed: Fig 4 plots all ten
numbers and the caption carries medium's replication. If a reviewer wants the
grid, it goes to an appendix table generated from the same CSVs.

## LaTeX notes

- `booktabs`, no vertical rules; `\small` or `\footnotesize`; tabular numbers.
- Generate both tables from CSVs by script (same auditability rule as figures)
  — hand-typed tables drift from data when medium keep-22 lands.
