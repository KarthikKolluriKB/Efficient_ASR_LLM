# Fair-Speech multi-axis depth sweep — fairness evaluation runbook

Sweeps the **whisper-small SLAM-ASR** model across all 12 encoder depths,
evaluated on **Meta Fair-Speech**, breaking WER/CER out by **gender, age,
socioeconomic status (SES), and ethnicity**. Companion to
[SMALL_SWEEP.md](SMALL_SWEEP.md) (which does the same on CV22 English, gender
only).

## Why this run (IMPACT-SPEECH framing)

The research question is a **fairness** one, not an efficiency one: *does
encoder compression of a SpeechLLM introduce or amplify demographic
disparities, and is the effect consistent across axes?* Fair-Speech is the
benchmark because it carries the axis most ASR-fairness work omits —
**socioeconomic background** — alongside gender/age/ethnicity. The headline
result we are hunting: a group (e.g. low-SES) that degrades **faster** under
pruning than others, i.e. compression with a disparate impact.

Same checkpoints as the CV22 sweep — only the evaluation dataset changes.

---

## ⚠️ License — Fair-Speech stays on the server

Meta's terms forbid redistributing the dataset **or derived per-utterance
transcripts**. Therefore:

- **Do NOT pull per-utterance CSVs to the laptop.** Run aggregation and the
  paired bootstrap **on the server**.
- Only **summary-level** outputs may leave the server: `*_multiaxis.csv`
  (per-group WER/CER), `findings_*.csv/.md`, `plot_*.png`,
  `multiaxis_paired_*.csv`, `multiaxis_gap_*.csv`.
- The `*fairspeech*` .gitignore patterns already block per-utterance commits;
  keep everything under `data/` and `results/fairspeech_sweep/`.

---

## Step 0 — Build the HF dataset (one-time)

```bash
cd ~/Karthik/Efficient_ASR_LLM
git pull
python -m datamodule.hf_fairspeech          # data/fairspeech -> data/fairspeech_hf
```

Expect ~26k rows kept and a build-stats block. Confirms the `gender/age/l1/
ses/ethnicity` columns are present.

---

## Step 1 — Dry run (no GPU)

```bash
python experiments/bias_pruning/scripts/run_depth_sweep.py \
    --dataset fairspeech --dry_run
```

Check every depth shows `ckpt exists = yes` (these are the whisper-small
checkpoints under `outputs/english/whisper-small/` + the baseline).

---

## Step 2 — Run the multi-axis sweep (~2–3 GPU h)

Outputs are routed to a Fair-Speech-specific folder so they don't collide with
the CV22 sweep (which reuses the same `d{NN}_keep{NN}` condition names):

```bash
CUDA_VISIBLE_DEVICES=1 python experiments/bias_pruning/scripts/run_depth_sweep.py \
    --dataset fairspeech \
    --wandb_project whisper_small_bias_sweep_fairspeech \
    --per_seed_dir  experiments/bias_pruning/results/fairspeech_sweep/per_axis \
    --per_utt_dir   experiments/bias_pruning/results/fairspeech_sweep/per_utterance
```

Each depth writes (handled by `evaluate_subgroup_wer.py`):
- `per_axis/d{NN}_keep{NN}_seed42.csv` — per-**gender** summary (back-compat)
- `per_axis/d{NN}_keep{NN}_seed42_multiaxis.csv` — **gender+age+SES+ethnicity**
- `per_utterance/d{NN}_keep{NN}_seed42.csv` — per-utterance (server-only)
- one wandb run per depth in `whisper_small_bias_sweep_fairspeech`

Resume after a partial failure with `--depths 7 8 9 10 11` or
`--continue_on_error` (see SMALL_SWEEP.md).

---

## Step 3 — Aggregate per axis (on the server)

```bash
python experiments/bias_pruning/scripts/aggregate_multiaxis_sweep.py \
    --in_dir  experiments/bias_pruning/results/fairspeech_sweep/per_axis \
    --out_dir experiments/bias_pruning/results/fairspeech_sweep \
    --total_layers 12
```

Produces, per axis (gender/age/ses/ethnicity):
`findings_{axis}.csv` (long), `findings_{axis}.md` (wide WER/CER tables),
`plot_{axis}.png` (WER/CER vs depth, one line per group). Under-powered groups
(<200 utts or <30 min) stay in the CSV but are dropped from the plot.

---

## Step 4 — Significance: per-group paired bootstrap (on the server)

Tests, per axis and group, whether each group degrades significantly from the
unpruned baseline — and how the worst−best disparity gap shifts. Run it
against the depths that matter (the "safe" pruning end + a deeper point):

```bash
python experiments/bias_pruning/scripts/compare_conditions_multiaxis.py \
    --per_utt_dir experiments/bias_pruning/results/fairspeech_sweep/per_utterance \
    --baseline d00_keep12 --pruned d02_keep10 d04_keep08 --seed 42 \
    --axes gender age ses ethnicity \
    --out_dir experiments/bias_pruning/results/fairspeech_sweep
```

Outputs per comparison (summary-level — OK to pull):
- `multiaxis_paired_{base}_vs_{pruned}.csv` — per-(axis,group) ΔWER/ΔCER + 95% CI + p
- `multiaxis_gap_{base}_vs_{pruned}.csv` — disparity gap (worst−best) at baseline
  vs pruned and the shift

The console marks `p < 0.05` deltas with `*`.

---

## Step 5 — Pull summary outputs to the laptop

```bash
# from the laptop — summary-level only, never per_utterance/
scp -r server:~/Karthik/Efficient_ASR_LLM/experiments/bias_pruning/results/fairspeech_sweep/{findings_*,plot_*,multiaxis_*} \
    experiments/bias_pruning/results/fairspeech_sweep/
```

---

## What to look for (the fairness claim)

- **SES disparate impact:** does low-SES WER rise faster with depth than
  high-SES, and does `multiaxis_gap` show the SES gap *widening* under pruning
  (positive `wer_gap_shift`)? That's the headline.
- **Gender at scale:** does the small-model gender null hold here too (bigger
  cells = more power), contrasting Experiment 1's large-v2 signal?
- **Cross-axis:** which axis is most sensitive to compression? Report all four;
  null results on age/ethnicity are still on-theme.
- **Rigor caveat:** single seed — frame group effects via the paired bootstrap,
  not the raw depth curve (which is non-monotonic; see SMALL_SWEEP.md).
```
