# Figure plan — overview & shared style

Plans for the paper's figures (IMPACT-SPEECH, deadline 2026-07-20). Each numbered
file specifies ONE figure: the claim it carries, why that visual form, the exact
data, and construction directions. Nothing here is rendered yet — these are specs.

## The narrative arc and the figure that carries each beat

| # | Beat of the argument | Figure | Priority |
|---|---|---|---|
| 1 | "A prune that looks free harms one group" (the hook) | [01 diverging bars](01_fig_free_prune_hook.md) | MUST |
| 2 | "The harm grows with depth — not a fluke" | [02 depth trend](02_fig_depth_trend.md) | MUST |
| 3 | "Only the large model hides it" (the novel claim) | [03 scale contrast](03_fig_scale_contrast.md) | MUST |
| 4 | "LoRA improves the average, widens the racial ratio" (RQ2) | [04 LoRA figure](04_fig_lora_backfire.md) | MUST |
| 5 | "WHY: spare capacity is the hiding place" (mechanism) | [05 mechanism schematic](05_fig_mechanism_diagram.md) | STRONG |
| 6 | "Cross-lingual: not all gaps behave like race" | [06 cross-lingual](06_fig_crosslingual.md) | OPTIONAL |
| — | Exact numbers reviewers will check | [07 tables](07_tables.md) | MUST |

Four data figures + one schematic is the right budget for a 4–8 page workshop
paper. If space forces a cut, drop 06 first, then fold 02 into Table 1.

## One-sentence test

Every figure must let a reviewer say ONE sentence without reading the caption:

1. *"The average improved but Black speakers got worse."*
2. *"The gap widens at every prune step while the model still works."*
3. *"Only the largest model's ratio rises — and only it has a free prune."*
4. *"LoRA moves the average down and the ratio up."*
5. *"Big models have tail-only layers; small models don't — that's the hiding place."*

If a draft figure doesn't produce its sentence at a glance, simplify it.

## Shared style (applies to every figure)

Derived from the dataviz method; palette subsets already machine-validated.

- **Sizing:** single-column 3.35 in wide (≈2.2–2.6 in tall); Fig 04 may go
  full-width 6.9 in two-panel. Export PDF (LaTeX) + PNG 300 dpi (review).
- **Type:** one sans throughout (system/DejaVu), 8 pt base, 9 pt bold titles,
  7 pt tick labels. Numbers on axes use tabular figures.
- **Palette (validated, light surface #fcfcfb):**
  - Black speakers / harmed series → red `#e34948`
  - Asian speakers / reference series → blue `#2a78d6`
  - Aggregate (ALL) → neutral gray `#898781`, dashed
  - Scale curves (Fig 03): large-v2 `#2a78d6`, medium `#eda100`, small `#1baf7a`
    — aqua & yellow are sub-3:1 contrast, so **direct labels are mandatory** on
    those lines (validator WARN, relief rule).
  - Shaded "WER-preserving" region → blue-100 `#cde2fb` at ~45 % alpha.
- **Marks:** 2 px lines, ≥8 px endpoint markers, thin bars with a 2 px surface
  gap; grid = solid hairline `#e1e0d9`; no top/right spines; axis ink `#898781`.
- **Labels:** legend for ≥2 series PLUS selective direct labels (line ends, the
  one bar that matters). Never a number on every point.
- **Hard rules (anti-patterns):** no dual y-axes (split into panels instead);
  color follows the entity across ALL figures (Black is red everywhere, base is
  blue everywhere); no rainbow ramps; ρ and Δ never share an axis.

## Data sources (all local)

- Sweep CSVs: `P:\Programming\Bias in pruning exps\all_results\<sweep>\...` —
  loader must union top-level AND `per_axis/` `*_multiaxis.csv` (medium stores
  d00 at top level; see ANALYSIS_MASTER.md §0).
- Aggregated findings: `experiments/bias_pruning/results/<sweep>/findings_*.csv`.
- Every spec file below embeds the exact numbers so the builder can hard-verify.

## Implementation route (when we do render)

One script `scripts/make_paper_figures.py` reading the CSVs (not hard-coded
numbers), writing `results/figures/fig0N_*.{pdf,png}`; print the plotted values
to stdout so each figure is auditable against ANALYSIS_MASTER.md. Pending data
(medium keep-22, DA/NL LoRA) slots in by re-running the script.
