# RQ3 — "De-Averaged Pruning" mitigation: implementation + demo tests

The compression pipeline's bias enters wherever an **average makes a decision**.
The mitigation replaces each of the three averages with a tail (worst-α%)
measure — utterance-level only, **no demographic labels ever**:

| De-averaging point | Status quo | De-averaged | Where implemented |
|---|---|---|---|
| 1. Recovery loss | batch-mean CE | batch-CVaR (worst α-fraction) | `models/losses.py`, wired in `models/model.py` via `train.cvar_alpha` |
| 2. Checkpoint selection | best aggregate val WER | best tail (CVaR) val loss | `train.py` via `train.checkpoint_monitor: tail_loss` |
| 3. Acceptance test | "free if aggregate holds" | "free only if no utterance-decile regresses" | `acceptance_test_deciles.py` (retroactive, zero GPU) |

## Pipeline changes (server-ready)

- **`models/losses.py`** — `per_utterance_ce`, `cvar`, `sequence_cvar_ce`.
- **`models/model.py`** — `train_config.cvar_alpha` (default 1.0 = unchanged
  mean CE). When < 1.0, the loss is batch-CVaR and per-utterance losses are
  exposed on the model output for tail validation.
- **`train.py`** — validation now computes `val/tail_loss` (CVaR over the val
  set) when CVaR is enabled; `train.checkpoint_monitor: tail_loss` selects the
  best checkpoint on it (file name stays `checkpoint_best_wer.pt` so every
  downstream eval script keeps working).
- **`configs/whisper_largev2/english/ablation_30L_cvar.yaml`** — the real
  experiment arm (keep-30 + α=0.2 + tail checkpoint), identical to
  `ablation_30L.yaml` otherwise.

## Demo tests (all CPU-local, no GPU)

Run from repo root.

### 1. Unit tests — the loss math is right
```
python experiments/bias_pruning/rq3_mitigation/test_cvar_loss.py
```
6/6 pass: per-utterance CE matches manual computation and HF shift semantics;
α=1 reduces to the mean; worst-k selection exact; fully-masked utterances
excluded (no NaN); **gradient flows only through the worst α-fraction**.

### 2. Synthetic mechanism demo — does CVaR mitigate?
```
python experiments/bias_pruning/rq3_mitigation/demo_cvar_recovery.py
```
Toy 2-group problem (2% minority, harder mapping), avg-importance pruning +
tail-concentrated damage, LoRA-style recovery at a fixed budget, mean-CE vs
batch-CVaR — using the exact `sequence_cvar_ce` the pipeline uses. Result
(4 seeds):

- **RQ1 analog reproduces:** damage concentrates ~4x on the minority.
- **RQ2/RQ5 analog reproduces:** mean-CE recovery restores the aggregate
  (16.3% → 1.9%) but leaves the minority harmed (12.7% dense → 40% recovered)
  and *worsens* the disparity ratio (18x → 35x).
- **RQ3 direction confirmed, magnitude modest:** CVaR + tail checkpoint
  improves minority error and ratio in 3/4 seeds at equal budget and equal
  aggregate — but by a few points only, because plain CE magnitude already
  up-weights badly-served samples under mean training. **CVaR's leverage
  depends on how strongly damage concentrates in the loss tail** → which is
  precisely what the free go/no-go gate checks (next).

### 3. Real-data acceptance test + go/no-go gate (zero GPU)
```
python experiments/bias_pruning/rq3_mitigation/acceptance_test_deciles.py
```
Runs on the committed large-v2 CV22 seed-42 per-utterance CSVs
(unpruned vs keep-30). Findings (12,078 paired utterances):

- Aggregate WER: 15.5% → 16.9% (+1.45pp). Utterance-WER p90: **+7.1pp**;
  tail-20% mean: +3.2pp (bootstrap-significant).
- **Hidden-harm tolerance band:** any acceptance tolerance in
  (+1.45pp, +7.1pp) blesses this prune on the average while its tail
  regresses several times harder — the de-averaged decile test rejects it.
- **Go/no-go gate (utterance-level part): GO.** Damage is heavily
  concentrated: the worst 10% of utterances carry **70%** of all new errors
  (worst 20% → 98.7%). A label-free tail objective can see this harm.
  The demographic half of the gate (does the tail over-represent Black
  speakers?) needs the Fair-Speech per-utterance CSVs on the server.

Output CSV: `../results/rq3_acceptance/acceptance_test_deciles.csv`.

## Next (server): the real run

Everything is scripted — see **`RUNBOOK_rq3_cvar.md`** for copy-paste commands:

1. `gate_demographic_tail.py` — demographic go/no-go on Fair-Speech
   per-utterance CSVs (zero GPU). Gates the training run.
2. Train `configs/whisper_largev2/english/ablation_30L_cvar.yaml` (~1 GPU-day).
3. Eval on CV22 + Fair-Speech, then `check_rq3_success.py` scores the three
   arms (unpruned / mean-CE / CVaR) against the success bar: aggregate
   ≤ ~21.6%, Black WER ≤ 27.2%, Black/Asian ρ ≤ 1.98.
