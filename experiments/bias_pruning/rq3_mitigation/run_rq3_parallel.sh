#!/usr/bin/env bash
# =============================================================================
# run_rq3_parallel.sh — train all 4 RQ3 CVaR arms on 2 GPUs (2 runs per GPU).
# =============================================================================
# Lanes run in parallel; runs within a lane run sequentially.
#
#   GPU 0:  A keep-30 CVaR a=0.2   ->  C keep-30 CVaR a=0.1
#   GPU 1:  B keep-28 CVaR a=0.2   ->  D keep-30 LoRA CVaR a=0.2
#
# A (headline) and B (robustness) run first, so both finish ~day 1 and can be
# evaluated while C (alpha probe) and D (LoRA) train.
#
# Usage (from anywhere; must run inside the eff-asr-llm env):
#   # dry run — validate configs + print the plan, no training:
#   bash experiments/bias_pruning/rq3_mitigation/run_rq3_parallel.sh plan
#
#   # real run — detach so it survives disconnects:
#   tmux new -s rq3
#   bash experiments/bias_pruning/rq3_mitigation/run_rq3_parallel.sh
#   # Ctrl-b d to detach;  tmux attach -t rq3  to return
#
# Env knobs:
#   GPU0="0" GPU1="1"      # override device ids (e.g. GPU0=2 GPU1=3)
#   FORCE=1               # retrain even if checkpoint_best_wer.pt already exists
#                         # (default: skip completed runs, so re-running resumes)
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
FORCE="${FORCE:-0}"
# CONCURRENT=1 runs BOTH jobs on a GPU at the same time (2 per card, 4 total).
# Default 0 = sequential within a lane (one job per card at a time).
CONCURRENT="${CONCURRENT:-0}"
# seconds to wait before starting the 2nd job on each card, so the two model
# loads don't peak host RAM / GPU transfer simultaneously.
STAGGER="${STAGGER:-60}"

LOG_DIR="logs/rq3"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/orchestrator.log"

# Lane definitions (config path per run, in execution order).
GPU0_RUNS=(
  "configs/whisper_largev2/english/ablation_30L_cvar.yaml"              # A
  "configs/whisper_largev2/english/ablation_30L_cvar_a01.yaml"          # C
)
GPU1_RUNS=(
  "configs/whisper_largev2/english/ablation_28L_cvar.yaml"             # B
  "configs/whisper_largev2/english/LoRA/train/ablation_30L_cvar.yaml"  # D
)

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*" | tee -a "$MASTER_LOG"; }

outdir_of() {
  python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['train']['output_dir'])" "$1"
}

# Unique short tag from a config path (disambiguates the projector vs LoRA
# runs that share the ablation_30L_cvar.yaml basename).
tag_of() {
  echo "$1" | sed -e 's#configs/whisper_largev2/english/##' -e 's#\.yaml$##' -e 's#/#_#g'
}

# Start one training run in the background. Sets LAST_PID on launch.
# Returns 1 (and starts nothing) if the run is already complete.
LAST_PID=""
launch_one() {
  local gpu="$1" cfg="$2"
  local tag log outdir
  tag="$(tag_of "$cfg")"
  log="$LOG_DIR/gpu${gpu}_${tag}.log"
  outdir="$(outdir_of "$cfg")"
  if [ "$FORCE" != "1" ] && [ -f "${outdir%/}/checkpoint_best_wer.pt" ]; then
    say "GPU$gpu SKIP  $tag (checkpoint exists at $outdir; set FORCE=1 to retrain)"
    return 1
  fi
  say "GPU$gpu START $tag  (cfg=$cfg -> $log)"
  CUDA_VISIBLE_DEVICES="$gpu" python train.py --config "$cfg" > "$log" 2>&1 &
  LAST_PID=$!
  return 0
}

# --- sequential lane: run its configs one at a time --------------------------
run_lane() {
  local gpu="$1"; shift
  local configs=("$@")
  local cfg rc
  for cfg in "${configs[@]}"; do
    if launch_one "$gpu" "$cfg"; then
      wait "$LAST_PID"; rc=$?
      if [ "$rc" -eq 0 ]; then say "GPU$gpu DONE  $(tag_of "$cfg")"
      else say "GPU$gpu FAIL  $(tag_of "$cfg") (exit $rc) — see log; continuing lane"; fi
    fi
  done
  say "GPU$gpu lane complete"
}

# --- concurrent: launch every run at once (2 per card), collect pids ---------
ALL_PIDS=(); ALL_TAGS=()
maybe_launch() {   # gpu cfg
  if launch_one "$1" "$2"; then
    ALL_PIDS+=("$LAST_PID"); ALL_TAGS+=("$(tag_of "$2")")
  fi
}

# --- preflight ----------------------------------------------------------------
ALL_RUNS=("${GPU0_RUNS[@]}" "${GPU1_RUNS[@]}")
missing=0
for c in "${ALL_RUNS[@]}"; do
  [ -f "$c" ] || { echo "MISSING config: $c"; missing=1; }
done
[ "$missing" -eq 0 ] || { echo "Aborting: fix missing configs above."; exit 1; }

# --- clean: kill leftover trainings + remove ONLY the CVaR output dirs --------
# Guarded to paths containing "cvar", so the mean-CE baselines (ablation_30L/,
# ablation_28L/, ablation_30l_lora/) can never be touched.
if [ "${1:-}" = "clean" ]; then
  say "clean: stopping any running RQ3 trainings"
  if pkill -f "train.py --config configs/whisper_largev2/english" 2>/dev/null; then
    say "  sent SIGTERM to train.py process(es); waiting 3s"; sleep 3
  else
    say "  no matching train.py process running"
  fi
  for c in "${ALL_RUNS[@]}"; do
    outdir="$(outdir_of "$c")"
    if [[ "$outdir" != *cvar* ]]; then
      say "  REFUSING to remove non-CVaR dir: $outdir (skipped)"; continue
    fi
    if [ -d "$outdir" ]; then
      echo "  removing $outdir  (contents:)"; ls -1 "$outdir" 2>/dev/null | sed 's/^/      /'
      rm -rf "$outdir"
    else
      echo "  (already absent) $outdir"
    fi
  done
  rm -f "$LOG_DIR"/gpu*cvar*.log
  say "clean complete. Relaunch with:  CONCURRENT=1 bash $0"
  exit 0
fi

echo "============================================================"
echo " RQ3 parallel training plan   (repo: $REPO_ROOT)"
echo "============================================================"
printf " GPU %s lane:\n" "$GPU0"; for c in "${GPU0_RUNS[@]}"; do printf "    - %s  -> %s\n" "$(basename "$c")" "$(outdir_of "$c")"; done
printf " GPU %s lane:\n" "$GPU1"; for c in "${GPU1_RUNS[@]}"; do printf "    - %s  -> %s\n" "$(basename "$c")" "$(outdir_of "$c")"; done
echo " MODE=$([ "$CONCURRENT" = 1 ] && echo 'CONCURRENT (2 runs/GPU at once)' || echo 'SEQUENTIAL (1 run/GPU at a time)')"
echo " FORCE=$FORCE (1=retrain completed runs, 0=skip them)"
echo "============================================================"

if [ "${1:-}" = "plan" ]; then
  # validate the CVaR knobs are actually present, then exit without training
  python - <<'PY'
import yaml
cfgs = [
 "configs/whisper_largev2/english/ablation_30L_cvar.yaml",
 "configs/whisper_largev2/english/ablation_30L_cvar_a01.yaml",
 "configs/whisper_largev2/english/ablation_28L_cvar.yaml",
 "configs/whisper_largev2/english/LoRA/train/ablation_30L_cvar.yaml",
]
for c in cfgs:
    t = yaml.safe_load(open(c))["train"]
    assert 0 < t["cvar_alpha"] < 1.0, (c, t.get("cvar_alpha"))
    assert t["checkpoint_monitor"] == "tail_loss", (c, t.get("checkpoint_monitor"))
    print(f"  ok  {c.split('/')[-1]:<40} alpha={t['cvar_alpha']} monitor={t['checkpoint_monitor']}")
print("Plan valid. Re-run without 'plan' to start training.")
PY
  exit 0
fi

command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# kill every training child we launched, regardless of mode
trap 'echo; say "interrupted — stopping all training"; pkill -f "train.py --config configs/whisper_largev2/english" 2>/dev/null; exit 130' INT TERM

if [ "$CONCURRENT" = "1" ]; then
  say "CONCURRENT mode: 2 runs per GPU at once (stagger ${STAGGER}s, ~11GB each of 48GB)"
  # index 0 = first job on each card, index 1 = second job on each card
  n0=${#GPU0_RUNS[@]}; n1=${#GPU1_RUNS[@]}
  maxlen=$(( n0 > n1 ? n0 : n1 ))
  for idx in $(seq 0 $((maxlen - 1))); do
    if [ "$idx" -gt 0 ]; then say "stagger ${STAGGER}s before next job on each card"; sleep "$STAGGER"; fi
    [ "$idx" -lt "$n0" ] && maybe_launch "$GPU0" "${GPU0_RUNS[$idx]}"
    [ "$idx" -lt "$n1" ] && maybe_launch "$GPU1" "${GPU1_RUNS[$idx]}"
  done
  say "all ${#ALL_PIDS[@]} runs launched; waiting for completion"
  for i in "${!ALL_PIDS[@]}"; do
    wait "${ALL_PIDS[$i]}"; rc=$?
    if [ "$rc" -eq 0 ]; then say "DONE  ${ALL_TAGS[$i]}"
    else say "FAIL  ${ALL_TAGS[$i]} (exit $rc) — see log"; fi
  done
else
  say "SEQUENTIAL mode: 1 run per GPU at a time (set CONCURRENT=1 for 2 per GPU)"
  run_lane "$GPU0" "${GPU0_RUNS[@]}" &
  PID0=$!
  run_lane "$GPU1" "${GPU1_RUNS[@]}" &
  PID1=$!
  wait "$PID0"
  wait "$PID1"
fi

# --- summary ------------------------------------------------------------------
echo
echo "============================================================"
echo " RQ3 training summary"
echo "============================================================"
for c in "${ALL_RUNS[@]}"; do
  outdir="$(outdir_of "$c")"
  if [ -f "${outdir%/}/checkpoint_best_wer.pt" ]; then
    echo "  OK    $(tag_of "$c")  -> ${outdir%/}/checkpoint_best_wer.pt"
  else
    echo "  MISS  $(tag_of "$c")  (no checkpoint_best_wer.pt in $outdir)"
  fi
done
say "all lanes finished"
echo "Next: evaluate each checkpoint on Fair-Speech, then check_rq3_success.py"
echo "(see experiments/bias_pruning/rq3_mitigation/RUNBOOK_rq3_cvar.md)"
