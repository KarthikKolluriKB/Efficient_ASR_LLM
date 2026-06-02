# Phase 1 — whisper-small bias evaluation across 3 datasets

Goal: replicate (or refute) the disparate-impact pattern from Experiment 1
(whisper-large-v2 × CV22) by evaluating whisper-small at every pruning depth
on three independent datasets:

| dataset | bias dimensions added | n test utts | adapter |
|---|---|---:|---|
| **CV22 English** | gender, age, accent (sparse) | 16,390 | already built |
| **L2-ARCTIC** | gender, **L1**, native-English | 3,599 | `datamodule/hf_l2arctic.py` |
| **Fair-Speech (Meta)** | gender, age, **SES**, **ethnicity**, L1 | ~26,000 | `datamodule/hf_fairspeech.py` |

EdAcc and CORAAL are reserved for Phase 2/3.

---

## Workflow per dataset

```
data/<dataset>/                  ← downloaded raw (verifies green via verify_datasets.py)
    │
    │  python -m datamodule.hf_<dataset>     (one-time, ~5-30 min)
    ▼
data/<dataset>_hf/               ← SLAM-ASR schema, demographics inline
    │
    │  python experiments/bias_pruning/scripts/run_depth_sweep.py \
    │      --dataset <dataset>   (12 wandb runs, ~3-5 h on one GPU)
    ▼
wandb project whisper_small_bias_sweep_<dataset>
```

---

## 1. CV22 — already running / done

Use the same command as before (the new code preserves CV22 behaviour exactly
when `--dataset cv22` is passed, which is also the default):

```bash
CUDA_VISIBLE_DEVICES=1 python experiments/bias_pruning/scripts/run_depth_sweep.py \
    --dataset cv22 \
    --continue_on_error \
    --checkpoint_overrides \
        1=outputs/english/whisper-s_ablation_11L/checkpoint_best_wer.pt \
        2=outputs/english/whisper-s_ablation_10L/checkpoint_best_wer.pt \
        3=outputs/english/whisper-s_ablation_9L/checkpoint_best_wer.pt \
        4=outputs/english/whisper-s_ablation_8L/checkpoint_best_wer.pt \
        5=outputs/english/whisper-s_ablation_7L/checkpoint_best_wer.pt \
        6=outputs/english/whisper-s_ablation_6L/checkpoint_best_wer.pt
```

Wandb project: `whisper_small_bias_sweep_en`.

---

## 2. L2-ARCTIC

### 2a. Build the HF dataset (one-time, ~5 min)

```bash
python -m datamodule.hf_l2arctic
# Output: data/l2arctic_hf/test  (3,599 rows; gender + L1 + native_english)
```

Confirm:
```bash
python -c "from datasets import load_from_disk; d = load_from_disk('data/l2arctic_hf'); print(d); print(d['test'][0].keys())"
```

### 2b. Sweep

```bash
CUDA_VISIBLE_DEVICES=1 python experiments/bias_pruning/scripts/run_depth_sweep.py \
    --dataset l2arctic \
    --continue_on_error \
    --checkpoint_overrides \
        1=outputs/english/whisper-s_ablation_11L/checkpoint_best_wer.pt \
        2=outputs/english/whisper-s_ablation_10L/checkpoint_best_wer.pt \
        3=outputs/english/whisper-s_ablation_9L/checkpoint_best_wer.pt \
        4=outputs/english/whisper-s_ablation_8L/checkpoint_best_wer.pt \
        5=outputs/english/whisper-s_ablation_7L/checkpoint_best_wer.pt \
        6=outputs/english/whisper-s_ablation_6L/checkpoint_best_wer.pt
```

L2-ARCTIC is small (3.6k utterances) so each depth runs in **~3–5 minutes**.
Full sweep: ~30–60 min. Wandb project: `whisper_small_bias_sweep_l2arctic`.

What changes inside `evaluate_subgroup_wer.py` for L2-ARCTIC:
- `--hf_dataset_path data/l2arctic_hf` overrides the CV22 path in the YAML.
- `--demographic_source hf_columns` reads `gender`, `l1`, `native_english`,
  `age`, `accent` directly from each HF row (no CV22 TSV-join).

---

## 3. Fair-Speech

### 3a. Build the HF dataset (one-time, ~15–30 min — 26k audio files to decode)

```bash
python -m datamodule.hf_fairspeech
# Output: data/fairspeech_hf/test  (~26k rows; gender + age + SES + ethnicity + L1)
```

The build is batched (2,000 rows per chunk) so peak memory stays bounded
(~2 GB instead of ~12 GB if loaded in one shot). If you still OOM, drop
the batch size: `python -m datamodule.hf_fairspeech --batch_size 1000`.

Audio files are pre-indexed once at startup, then per-row lookup is O(1).

### 3b. Sweep

```bash
CUDA_VISIBLE_DEVICES=1 python experiments/bias_pruning/scripts/run_depth_sweep.py \
    --dataset fairspeech \
    --continue_on_error \
    --checkpoint_overrides \
        1=outputs/english/whisper-s_ablation_11L/checkpoint_best_wer.pt \
        2=outputs/english/whisper-s_ablation_10L/checkpoint_best_wer.pt \
        3=outputs/english/whisper-s_ablation_9L/checkpoint_best_wer.pt \
        4=outputs/english/whisper-s_ablation_8L/checkpoint_best_wer.pt \
        5=outputs/english/whisper-s_ablation_7L/checkpoint_best_wer.pt \
        6=outputs/english/whisper-s_ablation_6L/checkpoint_best_wer.pt
```

~26k utterances ≈ ~50 min per depth × 12 = **~10 h**. Run it overnight.
Wandb project: `whisper_small_bias_sweep_fairspeech`.

---

## License reminder — Fair-Speech

Meta's terms forbid redistribution of any part of the dataset, including
reference transcripts. Concretely:

- ✅ Running this evaluation (Meta's explicit Purpose: ASR fairness eval).
- ✅ Publishing summary-level metrics (per-gender mean WER, p-values).
- ❌ Committing `data/fairspeech_hf/` (gitignored under `data/`).
- ❌ Committing per-utterance CSVs derived from Fair-Speech (gitignored
   under `experiments/bias_pruning/results/`).
- ❌ Training a model that predicts race/ethnicity/gender of individuals.

The `data/` and `experiments/bias_pruning/results/` gitignores already
block both classes of files from accidentally entering version control.

---

## What to look for after Phase 1 finishes

The three datasets answer different parts of the same question:

| evidence | from where |
|---|---|
| Does the female-vs-male WER gap widen with depth on **read English** speech? | CV22, Fair-Speech |
| Does pruning hurt **non-native English** more than native? | L2-ARCTIC |
| Does pruning interact with **SES**, **ethnicity**, **age**? | Fair-Speech alone |

If all three datasets show the gap widening monotonically, the claim is solid.
If only Fair-Speech shows it but CV22 doesn't, content/recording confounds
remain plausible. If only CV22 shows it, the original finding may be
CV-specific.

After all three sweeps land, paste me the wandb project URLs and I'll write
the cross-dataset comparison + Phase 1 summary section of `summary.md`.
