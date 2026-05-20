# Experiment 1 — Encoder Pruning and Gender Bias

Does removing the top 2 encoder layers from Whisper-Medium (a depth where
aggregate WER was shown to be preserved in paper 1) also preserve WER
**equally across gender subgroups**, or does bias hide under the aggregate?

Models: Whisper-Medium encoder + ConcatLinear projector + Qwen2.5-3B LLM
(checkpoints from the prior paper; no training in this experiment).
Data: CommonVoice 22, English split.

---

## Layer-count vocabulary (READ THIS FIRST)

Whisper-Medium has 24 encoder layers. There are two naming conventions in
this repo that mean *opposite things*.

| Spec wording (this experiment) | Repo config name        | Layers kept | Layers pruned (top-down) |
|--------------------------------|-------------------------|------------:|-------------------------:|
| `unpruned` baseline            | `baseline`              |          24 |                        0 |
| `2L pruned` (target condition) | `ablation_22L`          |          22 |                        2 |

The repo's `ablation_2L.yaml` is **not** "2L pruned" — it keeps only 2 layers
(prunes 22). Whenever this experiment says "2L pruned" it means the
`ablation_22L` checkpoint, which keeps 22 layers.

---

## Inputs and prerequisites

These items live outside this folder and must be present before the
evaluation step runs. None are required to build the descriptive table.

1. **Paper 1 checkpoints (English, Whisper-Medium):**
   - Unpruned: `outputs/english/whisper-medium/baseline/checkpoint_best_wer.pt`
   - 2L pruned: `outputs/english/whisper-medium/ablation_22L/checkpoint_best_wer.pt`
   - If multiple seeds exist they should follow a parallel directory pattern
     such as `outputs/english/whisper-medium/baseline_seed{N}/checkpoint_best_wer.pt`.
     If only one seed exists (the configs ship with `seed: 42`), the per-seed
     SD across seeds is undefined; this is noted as a limitation.

2. **Preprocessed HuggingFace test split:**
   - Path: `data/cv22_hf/en` (built by `python -m datamodule.hf_data --language en`)
   - The current preprocessing pipeline **truncates `client_id` to 16 chars**
     (stored as `speaker_id`) and **drops `gender`, `age`, `accent`**.
     `evaluate_subgroup_wer.py` therefore joins per-utterance predictions
     back to the upstream CommonVoice 22 `test.tsv` to recover demographics.
     The join key is `(speaker_id_prefix, normalised_sentence)`.

3. **Upstream CommonVoice 22 transcript:**
   - `transcript/en/test.tsv` from `fsicoli/common_voice_22_0`
   - Downloaded automatically by `build_descriptive_table.py` via
     `huggingface_hub.hf_hub_download`. Carries `client_id`, `path`,
     `sentence`, `gender`, `age`, `accents` and other metadata.

4. **Speaker-disjoint test split:**
   - CommonVoice 22 already ships a speaker-disjoint `test.tsv` (its split
     definition guarantees no `client_id` overlap with `train` or `dev`).
     This experiment uses that file as the speaker-disjoint test split.
   - If paper 1 used a *further* subset of `test.tsv`, drop a
     CSV/TSV of utterance IDs (`path` column) into `data/splits/` and pass
     `--test_split_path` to the scripts.

5. **Paper 1 aggregate WER targets (for the sanity check in Step 4):**
   - Fill these in once retrieved from paper 1's results table:
     - Unpruned (24L) English test WER: **TBD**
     - 2L-pruned (22L kept) English test WER: **TBD**

---

## What each script does

- `scripts/build_descriptive_table.py` — downloads CV22 `en/test.tsv`,
  aggregates by gender and age, writes `results/descriptive_table.csv`,
  prints which gender cells pass the (≥30 min audio, ≥200 utterances)
  analysability threshold.
- `scripts/bootstrap_ci.py` — utility module: utterance-level bootstrap
  WER with 95 % CI. Includes a `__main__` self-test.
- `scripts/evaluate_subgroup_wer.py` — loads a checkpoint (reusing the
  inference path from `eval.py`), runs inference over the English test
  split, joins demographics from CV22 `test.tsv`, and writes per-utterance
  results plus a per-gender summary with bootstrap CIs.
- `scripts/compare_conditions.py` — aggregates per-seed CSVs, computes
  gap / ratio / worst-group WER, runs paired bootstrap significance tests
  per gender, writes `results/final_comparison.csv` and a comparison plot.

---

## Run order

```
# Step 2 — descriptive table (no model required)
python experiments/bias_pruning/scripts/build_descriptive_table.py

# Step 3 — self-test for the bootstrap utility
python experiments/bias_pruning/scripts/bootstrap_ci.py --self_test

# Step 5 — per-seed evaluation (one call per checkpoint, per condition)
python experiments/bias_pruning/scripts/evaluate_subgroup_wer.py \
    --config configs/whisper_medium/english/eval/baseline.yaml \
    --checkpoint_path outputs/english/whisper-medium/baseline/checkpoint_best_wer.pt \
    --prune_depth 0 --seed 42 --condition unpruned

python experiments/bias_pruning/scripts/evaluate_subgroup_wer.py \
    --config configs/whisper_medium/english/eval/ablation_22L.yaml \
    --checkpoint_path outputs/english/whisper-medium/ablation_22L/checkpoint_best_wer.pt \
    --prune_depth 2 --seed 42 --condition pruned_2L

# Step 6 — comparison
python experiments/bias_pruning/scripts/compare_conditions.py
```

Note: `configs/whisper_medium/english/eval/ablation_22L.yaml` does not yet
exist in the repo. Copy `baseline.yaml`, set `encoder_num_layers: 22`, and
point `eval.projector_path` to the 22L checkpoint.
