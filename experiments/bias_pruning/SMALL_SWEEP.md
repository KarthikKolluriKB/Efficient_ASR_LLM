# Whisper-small depth sweep — bias evaluation runbook

> **✅ COMPLETE — do not re-run.** This CV22 English sweep has already been
> run for all 12 depths (seed 42). Aggregated results live in
> `results/small_sweep_en/` (per-gender WER/CER per depth + findings table +
> plot). Re-running only overwrites identical outputs. The active next step is
> the Fair-Speech multi-axis sweep — see [FAIRSPEECH_SWEEP.md](FAIRSPEECH_SWEEP.md).

Sweeps the bias evaluation across **all 12 encoder depths** of
whisper-small on English CommonVoice 22. One eval per depth, results
streamed to wandb so you can pull them from a browser.

| depth | layers kept | layers pruned | wandb run name |
|---:|---:|---:|---|
| 0 | 12 | 0 | `d00_keep12` (unpruned baseline) |
| 1 | 11 | 1 | `d01_keep11` |
| 2 | 10 | 2 | `d02_keep10` |
| 3 | 9 | 3 | `d03_keep09` |
| 4 | 8 | 4 | `d04_keep08` |
| 5 | 7 | 5 | `d05_keep07` |
| 6 | 6 | 6 | `d06_keep06` |
| 7 | 5 | 7 | `d07_keep05` |
| 8 | 4 | 8 | `d08_keep04` |
| 9 | 3 | 9 | `d09_keep03` |
| 10 | 2 | 10 | `d10_keep02` |
| 11 | 1 | 11 | `d11_keep01` |

---

## One-time setup on the server

```bash
cd ~/Karthik/Efficient_ASR_LLM
git pull

# wandb auth (skip if already done)
wandb login   # paste API key from https://wandb.ai/authorize
```

---

## Confirm the checkpoints are on disk

```bash
ls outputs/english/whisper-s_baseline/checkpoint_best_wer.pt
ls outputs/english/whisper-small/ablation_*L/checkpoint_best_wer.pt
```

If any depth's checkpoint is missing, the sweep will skip that depth and
print a warning — the others still run. Override paths with
`--checkpoint_overrides 5=/scratch/d5.pt 7=/scratch/d7.pt` etc.

---

## Dry run first

This lists every depth, its config, and whether the checkpoint exists.
No GPU used.

```bash
python experiments/bias_pruning/scripts/run_depth_sweep.py --dry_run
```

Expected output: a table with `ckpt exists` column saying `yes` for every
depth you have a checkpoint for, plus the message `DRY RUN — no commands
executed.`

---

## Run the full sweep

12 sequential evaluations, each on a single GPU. Whisper-small is faster
than the large-v2 runs you did before — probably **8–15 min per depth**,
so the whole sweep takes ~2–3 hours.

```bash
CUDA_VISIBLE_DEVICES=1 python experiments/bias_pruning/scripts/run_depth_sweep.py \
    --wandb_project whisper_small_bias_sweep_en
```

Outputs per depth (each handled automatically):
- Local: `experiments/bias_pruning/results/per_utterance/d{NN}_keep{NN}_seed42.csv`
- Local: `experiments/bias_pruning/results/per_seed_wer/d{NN}_keep{NN}_seed42.csv`
- wandb: one run per depth in the `whisper_small_bias_sweep_en` project,
  with the per-utterance CSV and per-gender summary as artifacts plus a
  `per_gender_summary` table viewable in the UI.

---

## Subsets and resumes

Just the mild end of the curve:
```bash
python experiments/bias_pruning/scripts/run_depth_sweep.py --depths 0 1 2 3 4
```

Just the depths you haven't done yet (after a partial failure):
```bash
python experiments/bias_pruning/scripts/run_depth_sweep.py --depths 7 8 9 10 11
```

Keep going past a failure:
```bash
python experiments/bias_pruning/scripts/run_depth_sweep.py --continue_on_error
```

---

## Where to find the results

Three places:

1. **wandb dashboard.** `https://wandb.ai/<your_entity>/whisper_small_bias_sweep_en/`
   - Each depth is one run named `d{NN}_keep{NN}`.
   - Run "Summary" panel shows `aggregate/wer`, `wer/male`, `wer/female`, etc.
   - The "Files" tab on each run has the per-utterance CSV and per-gender
     summary CSV downloadable in one click.
   - To get a depth-vs-WER curve, use the wandb run table, select all 12
     runs, and plot `depth/prune_depth` against `depth/aggregate_wer` /
     `depth/wer_female` / `depth/wer_male`.

2. **On the server (raw outputs).**
   `experiments/bias_pruning/results/per_seed_wer/d*_keep*_seed42.csv`

3. **Pull all artifacts to the laptop at once:**
   ```bash
   wandb artifact get --root ./experiments/bias_pruning/results/wandb_pull \
       <your_entity>/whisper_small_bias_sweep_en/bias_eval_d00_keep12_seed42:latest
   # ... repeat per depth, or scripted via the wandb Python API
   ```

---

## When the sweep finishes — what to look for

The depth-vs-WER curve, broken out by gender, is the headline plot. If
the female-vs-male degradation pattern from Experiment 1 (whisper-
large-v2 at 30L) replicates here, you'll see:
- Female WER rising **faster** with depth than male WER.
- Aggregate WER curve crossing some threshold (e.g. paper 1's "preserved
  up to depth 2" claim) at a different point than the per-gender curves.
- The gender gap narrowing (or flipping) at deep pruning.

If the pattern *doesn't* replicate at the small-model scale, that itself
is a finding — the disparate-impact effect may be size-dependent.
