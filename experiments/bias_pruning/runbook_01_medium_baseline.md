# Runbook 01 — Fill the medium Fair-Speech baseline hole

**Goal:** the medium-scale Fair-Speech race curve is missing its anchor. In the
current results, `medium_fairspeech_sweep` has **no keep-24 (unpruned) row** and
its **keep-22 checkpoint is broken** (produced 500 %+ WER on every dataset). This
runbook fills both depths so the medium scale point has a valid baseline for the
scale-dependence comparison (small vs medium vs large-v2).

**Priority:** #1 (blocks a headline figure).
**Where:** server, GPUs 0 and 1. **Time:** NOT minutes — see reality below.

## What we actually found (2026-07 run)

Pre-flight (`find outputs/english/whisper-medium -name '*.pt'`) and a first eval
revealed the situation is heavier than "re-run two evals":

1. **CORRECTION (later finding):** the keep-24 baseline *result was never actually
   missing* — `medium_fairspeech_sweep`/`medium_cv22_sweep` store `d00` at the
   sweep top level, not in `per_axis/`, and an analysis loader dropped it. Medium
   keep-24 race ρ = 2.16 (clean) existed all along. The baseline retrain done here
   was therefore **unnecessary** (it merely reproduces the existing result). The
   only genuine gap was keep-22 (below). See `results/ANALYSIS_MASTER.md` §0.
2. **keep-22 (`ablation_22L`) is genuinely corrupt** — re-eval gave aggregate WER
   **535 %** (male 554 %, female 519 %), confirming a bad checkpoint, not a bad
   eval. It must be **retrained** (`train/ablation_22L.yaml`), overwriting the
   corrupt file. ⏳ retraining.
3. **`--total_layers 24` is REQUIRED.** The driver defaults to 12 (whisper-small).
   Without it, depth 2 mis-resolves to `ablation_10L` and depth 0 to keep-12.
4. **One Fair-Speech eval ≈ 3 hours** (184.7 min, 26 417 utts). Budget for it.
5. RQ1's scale-dependence figure uses the **base (projector-only)** line, so the
   base baseline is the right thing to train — the LoRA anchor (`baseline_lora`)
   already exists and is not the missing piece.

Layer/keep map for medium (24 encoder layers, 2-layer prune steps):

| depth | layers kept | checkpoint | eval config |
|------:|------------:|---|---|
| 0 | 24 (unpruned) | `outputs/english/whisper-medium/baseline/checkpoint_best_wer.pt` | `configs/whisper_medium/english/eval/baseline.yaml` |
| 2 | 22 (first prune) | `outputs/english/whisper-medium/ablation_22L/checkpoint_best_wer.pt` | `configs/whisper_medium/english/eval/ablation_22L.yaml` |

---

## Step 1 — Pre-flight: do the checkpoints exist?

```bash
cd <repo root>
ls -la outputs/english/whisper-medium/baseline/checkpoint_best_wer.pt      # keep-24
ls -la outputs/english/whisper-medium/ablation_22L/checkpoint_best_wer.pt  # keep-22
```

- **baseline present** → keep-24 is a cheap eval-only fix (the anchor you need).
- **ablation_22L present** → we still eval it, but expect it to reproduce the
  garbage (corruption, not a bad eval). Handled in Step 4.
- **baseline MISSING** → the medium unpruned projector was never saved; it must be
  trained before anything else: `python train.py --config configs/whisper_medium/english/train/baseline.yaml`.

## Step 2 — Dry-run the plan (no GPU)

```bash
python experiments/bias_pruning/scripts/run_depth_sweep.py \
    --model_dir           configs/whisper_medium/english/eval \
    --total_layers        24 \
    --checkpoint_root     outputs/english/whisper-medium \
    --baseline_checkpoint outputs/english/whisper-medium/baseline/checkpoint_best_wer.pt \
    --dataset             fairspeech \
    --depths              0 2 \
    --wandb_project       whisper_medium_bias_sweep_en \
    --dry_run
```

Confirm it prints **depth 0 → keep-24 (baseline)** and **depth 2 → keep-22
(ablation_22L)** against Fair-Speech. Wrong plan → stop here.

## Step 3 — Run the baseline (the anchor)

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/bias_pruning/scripts/run_depth_sweep.py \
    --model_dir           configs/whisper_medium/english/eval \
    --total_layers        24 \
    --checkpoint_root     outputs/english/whisper-medium \
    --baseline_checkpoint outputs/english/whisper-medium/baseline/checkpoint_best_wer.pt \
    --dataset             fairspeech \
    --depths              0 \
    --wandb_project       whisper_medium_bias_sweep_en
```

**Sanity gate:** the keep-24 aggregate WER should land ~20 % with Black and Asian
both finite. If so, the anchor is filled — the core goal of this runbook is done.

## Step 4 — Run keep-22, then decide

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/bias_pruning/scripts/run_depth_sweep.py \
    --model_dir           configs/whisper_medium/english/eval \
    --total_layers        24 \
    --checkpoint_root     outputs/english/whisper-medium \
    --baseline_checkpoint outputs/english/whisper-medium/baseline/checkpoint_best_wer.pt \
    --dataset             fairspeech \
    --depths              2 \
    --wandb_project       whisper_medium_bias_sweep_en
```

Inspect the keep-22 aggregate WER:

- **< 45 %** → checkpoint is fine; earlier failure was transient. Keep it. Done.
- **> 100 % / nonsense** → checkpoint is **corrupt**. Retrain it, then re-eval:

  ```bash
  # retrain the medium first-prune projector
  CUDA_VISIBLE_DEVICES=0 python train.py \
      --config configs/whisper_medium/english/train/ablation_22L.yaml

  # re-eval the fresh checkpoint
  CUDA_VISIBLE_DEVICES=1 python experiments/bias_pruning/scripts/run_depth_sweep.py \
      --model_dir configs/whisper_medium/english/eval \
      --total_layers 24 \
      --checkpoint_root outputs/english/whisper-medium \
      --baseline_checkpoint outputs/english/whisper-medium/baseline/checkpoint_best_wer.pt \
      --dataset fairspeech --depths 2 --wandb_project whisper_medium_bias_sweep_en
  ```

  The keep-24 anchor (Step 3) stands regardless — keep-22 corruption does not
  block it.

## Step 5 — Regenerate the findings tables

```bash
python experiments/bias_pruning/scripts/aggregate_multiaxis_sweep.py \
    --in_dir  experiments/bias_pruning/results/medium_fairspeech_sweep/per_axis \
    --out_dir experiments/bias_pruning/results/medium_fairspeech_sweep \
    --total_layers 24 \
    --axes gender age ses ethnicity
```

## Step 6 — Verify the hole is closed

Open `results/medium_fairspeech_sweep/findings_ethnicity.md` and confirm a
**keep-24 row** now exists with Black, Asian, and ALL populated. Record the
keep-24 Black/Asian ratio — that is the medium baseline anchor the
scale-dependence figure was missing.

Expected shape once complete (small vs medium vs large-v2 baseline ρ):

| scale | baseline keep | Black/Asian ρ |
|---|---:|---:|
| small | 12 | 2.03 |
| medium | 24 | **← this runbook fills it** |
| large-v2 | 32 | 1.98 |

---

## Done criteria

- [ ] keep-24 baseline row present in `findings_ethnicity.md`, aggregate ~20 %.
- [ ] keep-22 either valid (<45 %) or retrained-and-valid.
- [ ] medium baseline Black/Asian ρ recorded for the scale-dependence table.
