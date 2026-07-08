# RUNBOOK — RQ3 De-Averaged Pruning, best-scenario run (large-v2 keep-30)

Goal: prove the approach on the single condition where every prior finding
stacks the deck — **large-v2, keep-30, seed 42, Fair-Speech ethnicity axis**:
- RQ1: this is the "WER-preserving" prune that amplifies race/SES disparity.
- RQ5: this is where mean-loss LoRA recovery backfires (ratio worsens).
- Local gate (2026-07-08, CV22): keep-30's damage is tail-concentrated —
  worst 10% of utterances carry 70% of new errors.

Both control arms already exist on the server (unpruned baseline + keep-30
mean-CE recovery), so proving the approach costs **one training run** (the
CVaR arm) + evals.

Success bar (from the plan): aggregate <= ~21.6%, Black WER <= 27.2%
(unpruned level), Black/Asian ratio <= 1.98 (baseline).

All commands run from the repo root on the server. Timeline: results freeze
2026-07-15.

---

## Phase 0 — demographic go/no-go gate (zero GPU if CSVs exist)

The gate asks: does keep-30's utterance tail over-represent Black speakers on
Fair-Speech? (CVaR is label-free — it can only fix the group gap if the group
lives in the tail it targets.)

Check whether large-v2 Fair-Speech per-utterance CSVs already exist:

```bash
ls experiments/bias_pruning/results/largev2_fairspeech_sweep/per_utterance/ 2>/dev/null
ls experiments/bias_pruning/results/per_utterance/ | grep -i fair
```

If missing, regenerate the two needed conditions (~1-3 h each, single GPU):

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/bias_pruning/scripts/evaluate_subgroup_wer.py \
    --config configs/whisper_largev2/english/eval/baseline.yaml \
    --checkpoint_path outputs/english/whisper-largev2/baseline/checkpoint_best_wer.pt \
    --prune_depth 0 --seed 42 --condition fs_unpruned \
    --hf_dataset_path data/fairspeech_hf --demographic_source hf_columns \
    --output_path experiments/bias_pruning/results/largev2_fairspeech_sweep/per_utterance/fs_unpruned_seed42.csv

CUDA_VISIBLE_DEVICES=0 python experiments/bias_pruning/scripts/evaluate_subgroup_wer.py \
    --config configs/whisper_largev2/english/eval/ablation_30L.yaml \
    --checkpoint_path outputs/english/whisper-largev2/ablation_30L/checkpoint_best_wer.pt \
    --prune_depth 2 --seed 42 --condition fs_pruned_2L \
    --hf_dataset_path data/fairspeech_hf --demographic_source hf_columns \
    --output_path experiments/bias_pruning/results/largev2_fairspeech_sweep/per_utterance/fs_pruned_2L_seed42.csv
```

Run the gate (zero GPU):

```bash
python experiments/bias_pruning/rq3_mitigation/gate_demographic_tail.py \
    --unpruned experiments/bias_pruning/results/largev2_fairspeech_sweep/per_utterance/fs_unpruned_seed42.csv \
    --pruned   experiments/bias_pruning/results/largev2_fairspeech_sweep/per_utterance/fs_pruned_2L_seed42.csv \
    --axis ethnicity --focus "black or african american"
# also worth one look at the SES axis:
#   --axis ses --focus <lowest-ses label as printed by the script>
```

Decision:
- **GO** (damage tail over-represents the group, CI-supported) -> Phase 1.
- **WEAK GO** (only the level tail over-represents) -> Phase 1, but expect a
  smaller effect; consider the alpha=0.1 probe up front.
- **NO-GO** -> stop; do NOT spend the GPU-days. Write the null (label-free
  reweighting cannot transfer) — it is a planned, publishable outcome.

---

## Phase 1 — train the CVaR arm (~ same cost as the original 30L run)

Config differs from `ablation_30L.yaml` by exactly two knobs
(`cvar_alpha: 0.2`, `checkpoint_monitor: tail_loss`) + output/log paths:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config configs/whisper_largev2/english/ablation_30L_cvar.yaml
```

Sanity marks in the log:
- startup: `Batch-CVaR loss enabled: alpha=0.2`
- each validation: `... | Val Tail Loss: X.XXXX` and wandb `val/tail_loss`
- best-checkpoint lines say `New best model (tail_loss)!`
- train loss will read HIGHER than the mean-CE run (it is the mean of the
  worst 20% of utterances, not the batch mean) — that is expected, compare
  `val/wer` / `val/tail_loss` trends, not raw train loss.

Output: `outputs/english/whisper-largev2/ablation_30L_cvar/checkpoint_best_wer.pt`
(name kept for pipeline compat; it was selected on tail val loss).

---

## Phase 2 — evaluate the CVaR checkpoint

CV22 aggregate sanity (~1-3 h) — the prune must stay "free on the average":

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/bias_pruning/scripts/evaluate_subgroup_wer.py \
    --config configs/whisper_largev2/english/eval/ablation_30L.yaml \
    --checkpoint_path outputs/english/whisper-largev2/ablation_30L_cvar/checkpoint_best_wer.pt \
    --prune_depth 2 --seed 42 --condition pruned_2L_cvar
```

Fair-Speech demographics (~1-3 h) — the actual RQ3 measurement:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/bias_pruning/scripts/evaluate_subgroup_wer.py \
    --config configs/whisper_largev2/english/eval/ablation_30L.yaml \
    --checkpoint_path outputs/english/whisper-largev2/ablation_30L_cvar/checkpoint_best_wer.pt \
    --prune_depth 2 --seed 42 --condition fs_pruned_2L_cvar \
    --hf_dataset_path data/fairspeech_hf --demographic_source hf_columns \
    --output_path experiments/bias_pruning/results/largev2_fairspeech_sweep/per_utterance/fs_pruned_2L_cvar_seed42.csv
```

---

## Phase 3 — verdict (zero GPU)

Success bar, all three arms side by side:

```bash
python experiments/bias_pruning/rq3_mitigation/check_rq3_success.py \
    --unpruned experiments/bias_pruning/results/largev2_fairspeech_sweep/per_utterance/fs_unpruned_seed42.csv \
    --mean     experiments/bias_pruning/results/largev2_fairspeech_sweep/per_utterance/fs_pruned_2L_seed42.csv \
    --cvar     experiments/bias_pruning/results/largev2_fairspeech_sweep/per_utterance/fs_pruned_2L_cvar_seed42.csv \
    --axis ethnicity --focus "black or african american" --reference_group asian
```

De-averaged acceptance test on the CVaR arm (does it now pass what the mean
arm failed?):

```bash
python experiments/bias_pruning/rq3_mitigation/acceptance_test_deciles.py \
    --unpruned experiments/bias_pruning/results/largev2_fairspeech_sweep/per_utterance/fs_unpruned_seed42.csv \
    --pruned   experiments/bias_pruning/results/largev2_fairspeech_sweep/per_utterance/fs_pruned_2L_cvar_seed42.csv \
    --pruned_label "large-v2 keep-30 + CVaR recovery"
```

Repeat with SES (`--axis ses`) for the paper's SES hook.

Interpretation matrix:
- **Bar met** -> headline: "De-averaging the recovery makes the prune free for
  the worst-served group too." Optionally add keep-28 + the CVaR-LoRA arm.
- **Partial** (direction right, bar missed) -> report effect size + CI; one
  alpha=0.1 probe run is allowed by the plan; DAR remains the fallback arm.
- **Null** -> planned publishable finding; the de-averaged acceptance test
  (Phase 3) and the gate remain contributions regardless.

## Budget

| Step | GPU cost |
|---|---|
| Phase 0 (if CSVs missing) | 2 evals, ~2-6 h |
| Phase 1 train | ~1 GPU-day (same as original ablation_30L) |
| Phase 2 evals | ~2-6 h |
| Phase 3 | 0 |

Well inside the plan's 2-3 GPU-day envelope, leaving room for keep-28 and/or
a CVaR-LoRA variant before the 07-15 freeze.
