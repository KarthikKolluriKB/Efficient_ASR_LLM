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

# --- run one lane: a GPU id followed by its config list -----------------------
run_lane() {
  local gpu="$1"; shift
  local configs=("$@")
  local cfg tag log outdir rc
  for cfg in "${configs[@]}"; do
    tag="$(tag_of "$cfg")"
    log="$LOG_DIR/gpu${gpu}_${tag}.log"
    outdir="$(outdir_of "$cfg")"

    if [ "$FORCE" != "1" ] && [ -f "${outdir%/}/checkpoint_best_wer.pt" ]; then
      say "GPU$gpu SKIP  $tag (checkpoint exists at $outdir; set FORCE=1 to retrain)"
      continue
    fi

    say "GPU$gpu START $tag  (cfg=$cfg -> $log)"
    CUDA_VISIBLE_DEVICES="$gpu" python train.py --config "$cfg" > "$log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
      say "GPU$gpu DONE  $tag"
    else
      say "GPU$gpu FAIL  $tag (exit $rc) — see $log; continuing lane"
    fi
  done
  say "GPU$gpu lane complete"
}

# --- preflight ----------------------------------------------------------------
ALL_RUNS=("${GPU0_RUNS[@]}" "${GPU1_RUNS[@]}")
missing=0
for c in "${ALL_RUNS[@]}"; do
  [ -f "$c" ] || { echo "MISSING config: $c"; missing=1; }
done
[ "$missing" -eq 0 ] || { echo "Aborting: fix missing configs above."; exit 1; }

echo "============================================================"
echo " RQ3 parallel training plan   (repo: $REPO_ROOT)"
echo "============================================================"
printf " GPU %s lane:\n" "$GPU0"; for c in "${GPU0_RUNS[@]}"; do printf "    - %s  -> %s\n" "$(basename "$c")" "$(outdir_of "$c")"; done
printf " GPU %s lane:\n" "$GPU1"; for c in "${GPU1_RUNS[@]}"; do printf "    - %s  -> %s\n" "$(basename "$c")" "$(outdir_of "$c")"; done
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

# --- launch both lanes in parallel -------------------------------------------
say "launching GPU$GPU0 and GPU$GPU1 lanes"
run_lane "$GPU0" "${GPU0_RUNS[@]}" &
PID0=$!
run_lane "$GPU1" "${GPU1_RUNS[@]}" &
PID1=$!

trap 'echo; say "interrupted — stopping training"; pkill -P $PID0 2>/dev/null; pkill -P $PID1 2>/dev/null; kill $PID0 $PID1 2>/dev/null; exit 130' INT TERM

wait "$PID0"
wait "$PID1"

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
