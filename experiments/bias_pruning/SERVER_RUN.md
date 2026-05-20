# Server-side runbook — Experiment 1 (gender bias under encoder pruning)

This guide walks through everything that has to happen on the server.
You'll pull two small per-seed CSVs back to your laptop afterwards; the
comparison + summary steps run locally.

Scope locked: English, CommonVoice 22, single seed = 42, both WER and CER
recorded (final reported metric still TBD).

---

## What to transfer to the server

Either `git pull` the latest branch, or copy these directories from the
laptop to the server:

```
experiments/bias_pruning/                             # all the new scripts + README
configs/whisper_largev2/english/eval/baseline.yaml    # 32 layers (unpruned)
configs/whisper_largev2/english/eval/ablation_30L.yaml # 30 layers (= 2L pruned)
```

Everything else this experiment touches already lives in the project
(`models/`, `datamodule/`, `utils/`, `eval.py`).

---

## Prerequisites on the server

1. The paper-1 checkpoints, at these exact paths (or pass `--checkpoint_path`
   on the command line to override):

   ```
   outputs/english/whisper-largev2/baseline/checkpoint_best_wer.pt
   outputs/english/whisper-largev2/ablation_30L/checkpoint_best_wer.pt
   ```

2. The preprocessed English HuggingFace dataset at `data/cv22_hf/en` with a
   `test` split. If it isn't there yet:

   ```bash
   # ~22 h of audio for test (modest); train is ~100 GB so only run this
   # with --splits test if you only need the bias evaluation:
   python -m datamodule.hf_data --language en --splits test \
     --output-dir data/cv22_hf
   ```

3. Python env that already runs `eval.py`. The new scripts add no new deps
   beyond what `eval.py` uses (`jiwer`, `huggingface_hub`, `numpy`,
   `matplotlib` for the local plot step — server doesn't need matplotlib).

4. Internet access from the server (so `evaluate_subgroup_wer.py` can pull
   `transcript/en/test.tsv` for the demographic join). If the server is
   offline, copy `test.tsv` over manually and pass `--cv_test_tsv`.

---

## One-time sanity check (a few seconds, no GPU)

```bash
cd <project_root>
python experiments/bias_pruning/scripts/bootstrap_ci.py --self_test
```

Expected output: `[Self-test] OK` with WER and CER both matching jiwer
exactly.

---

## Step 5 — run both conditions

Both commands run in inference mode only. Beam=2 by default from the eval
configs; ~1–3 h per condition on a single GPU at the CV22-en test size
(16,396 utterances after filtering).

```bash
# Unpruned baseline (32 layers, seed 42)
CUDA_VISIBLE_DEVICES=1 python experiments/bias_pruning/scripts/evaluate_subgroup_wer.py \
    --config configs/whisper_largev2/english/eval/baseline.yaml \
    --checkpoint_path outputs/english/whisper-largev2/baseline/checkpoint_best_wer.pt \
    --prune_depth 0 --seed 42 --condition unpruned

# 2L pruned (30 of 32 layers kept, seed 42)
CUDA_VISIBLE_DEVICES=1 python experiments/bias_pruning/scripts/evaluate_subgroup_wer.py \
    --config configs/whisper_largev2/english/eval/ablation_30L.yaml \
    --checkpoint_path outputs/english/whisper-largev2/ablation_30L/checkpoint_best_wer.pt \
    --prune_depth 2 --seed 42 --condition pruned_2L
```

Each run writes two files:

```
experiments/bias_pruning/results/per_utterance/{condition}_seed42.csv
experiments/bias_pruning/results/per_seed_wer/{condition}_seed42.csv
```

The first one is one row per utterance with `row_idx`, `speaker_id`,
`gender`, `age`, `accent`, `duration_s`, `reference`, `hypothesis`. The
second is the per-gender summary (n_utts, hours, WER, WER 95 % CI, CER,
CER 95 % CI, analysable yes/no).

At the end of each run the script prints aggregate WER + CER. If paper 1's
reported number is known, add `--target_wer 0.XX` and the script exits
non-zero when aggregate WER deviates by more than 0.5 absolute points —
useful to catch a broken data pipeline before the 2L run starts.

---

## What to pull back to the laptop

Just these four files (a few MB total):

```
experiments/bias_pruning/results/per_utterance/unpruned_seed42.csv
experiments/bias_pruning/results/per_utterance/pruned_2L_seed42.csv
experiments/bias_pruning/results/per_seed_wer/unpruned_seed42.csv
experiments/bias_pruning/results/per_seed_wer/pruned_2L_seed42.csv
```

Quick scp pattern:

```bash
scp -r <user>@<server>:<project>/experiments/bias_pruning/results/per_utterance/ \
       experiments/bias_pruning/results/
scp -r <user>@<server>:<project>/experiments/bias_pruning/results/per_seed_wer/ \
       experiments/bias_pruning/results/
```

---

## Step 6 — local analysis (on the laptop)

```bash
python experiments/bias_pruning/scripts/compare_conditions.py
```

This produces, all in `experiments/bias_pruning/results/`:

- `final_comparison.csv` — per (condition, gender) mean/SD/min/max for WER and CER.
- `disparity_metrics.csv` — gap, ratio, worst-group rate for WER and CER per condition.
- `paired_bootstrap.csv` — paired WER and CER test per gender, p-value two-sided.
- `comparison_plot.png` — twin bar plot of WER and CER by gender.

It also pretty-prints both metrics side-by-side. With seed=42 only,
seed-to-seed SD columns are 0 by definition; the paired bootstrap is the
significance signal that actually matters.

---

## Troubleshooting

- **"Checkpoint not found"** — pass `--checkpoint_path` explicitly; the
  configs hardcode the expected path but the CLI override wins.
- **"WARNING: N unmatched"** — utterances whose `(speaker_id[:16],
  normalised sentence)` didn't match a CV22 `test.tsv` row. Expect a small
  number (truncation collisions or punctuation drift). They get
  `gender=missing` and are excluded from gender-restricted analyses.
- **CUDA OOM at beam=2** — drop `eval.batch_size` to 16 or pass
  `--batch_size 16` (the script forwards the standard config knobs).
- **"`data/cv22_hf/en` not found"** — see Prerequisites step 2.
- **"transcripts/en/test.tsv" download fails** — pre-download once with
  `huggingface-cli download fsicoli/common_voice_22_0 transcript/en/test.tsv
  --repo-type dataset`, then pass that path via `--cv_test_tsv`.

---

## Reminder: naming convention

`whisper-largev2/ablation_30L` = "2L pruned" in this experiment = 30 of 32
encoder layers kept. The repo also has `ablation_2L`/`ablation_4L`/etc.
configs that keep *only* that many layers — those are **not** the same as
"2L/4L pruned" and should not be used for this experiment.
