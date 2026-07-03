"""
Driver: run every LEFTOVER bias evaluation across model scales x datasets.

For each (scale x dataset) sweep it expands the depth list into single-depth
jobs, then runs only the ones that are (a) not already done and (b) whose
checkpoint actually exists on disk. Jobs run 3-per-GPU across 2 GPUs, wave by
wave (6 concurrent), until everything is finished.

Usage (from repo root, or anywhere — it chdirs to the repo root):
    python experiments/bias_pruning/scripts/run_leftover_evals.py            # run
    python experiments/bias_pruning/scripts/run_leftover_evals.py --dry      # just print the plan

Edit GPUS / PER_GPU to change the distribution. Edit MANIFEST/SCALES/DATASETS
to add more sweeps (e.g. cross-lingual da/nl — see the commented example).
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
os.chdir(REPO)  # all paths below are repo-relative

SWEEP = "experiments/bias_pruning/scripts/run_depth_sweep.py"
RESULTS = "experiments/bias_pruning/results"

GPUS = [0, 1]      # available GPUs
PER_GPU = 3        # experiments per GPU per wave  -> wave size = PER_GPU * len(GPUS)
SEED = 42


def small_ckpt(keep: int) -> str:
    """Small English LoRA checkpoints are split across two dirs."""
    if keep >= 6:
        return f"outputs/english/whisper-s_ablation_{keep}L_lora/checkpoint_best_wer.pt"
    return f"outputs/english/whisper-small/ablation_{keep}L_lora/checkpoint_best_wer.pt"


# One entry per model scale (LoRA checkpoints). ckpt(keep) -> checkpoint path.
SCALES = [
    dict(scale="largev2", total_layers=32, depths=[0, 2, 4, 6, 8, 10, 12, 14, 16],
         model_dir="configs/whisper_largev2/english/LoRA/eval", batch=48,
         baseline="outputs/english/whisper-largev2/baseline_lora/checkpoint_best_wer.pt",
         ckpt=lambda k: f"outputs/english/whisper-largev2/ablation_{k}l_lora/checkpoint_best_wer.pt"),
    dict(scale="medium", total_layers=24, depths=[0, 2, 4, 6, 8, 10, 12, 14, 16],
         model_dir="configs/whisper_medium/english/LoRA/eval", batch=64,
         baseline="outputs/english/whisper-medium/baseline_lora/checkpoint_best_wer.pt",
         ckpt=lambda k: f"outputs/english/whisper-medium/ablation_{k}L_lora/checkpoint_best_wer.pt"),
    dict(scale="small", total_layers=12, depths=list(range(0, 12)),
         model_dir="configs/whisper_small/english/LoRA/eval", batch=64,
         baseline="outputs/english/whisper-s_baseline_lora/checkpoint_best_wer.pt",
         ckpt=small_ckpt),
]

# English evaluation datasets (same LoRA checkpoints, different test set = zero-shot).
# fairspeech -> ethnicity/SES ; cv22 -> accent/gender/age ; l2arctic -> L1.
DATASETS = ["fairspeech", "cv22", "l2arctic"]

# Build the full manifest: scale x dataset.
MANIFEST = []
for s in SCALES:
    for ds in DATASETS:
        MANIFEST.append(dict(
            name=f"{s['scale']}_{ds}_LoRA",
            dataset=ds,
            model_dir=s["model_dir"],
            total_layers=s["total_layers"],
            depths=s["depths"],
            batch=s["batch"],
            baseline=s["baseline"],
            ckpt=s["ckpt"],
            per_seed_dir=f"{RESULTS}/{s['scale']}_{ds}_LORA_sweep",
            extra=[],   # e.g. ["--cv_test_tsv", "/path/to/transcript/da/test.tsv"] for da/nl
        ))

# --- Example: add cross-lingual da/nl LoRA sweeps by appending here ---
# MANIFEST.append(dict(name="largev2_cv22_da_LoRA", dataset="cv22_da", ...,
#                      extra=["--cv_test_tsv", "<DA_TSV>"]))


def build_jobs():
    jobs, skipped_done, skipped_missing = [], 0, 0
    for m in MANIFEST:
        for d in m["depths"]:
            keep = m["total_layers"] - d
            cond = f"d{d:02d}_keep{keep:02d}"
            out = os.path.join(m["per_seed_dir"], f"{cond}_seed{SEED}_multiaxis.csv")
            if os.path.exists(out):
                skipped_done += 1
                continue
            ck = m["baseline"] if d == 0 else m["ckpt"](keep)
            if not os.path.exists(ck):
                skipped_missing += 1
                continue
            cmd = ["python", SWEEP,
                   "--dataset", m["dataset"],
                   "--model_dir", m["model_dir"],
                   "--total_layers", str(m["total_layers"]),
                   "--depths", str(d),
                   "--seed", str(SEED),
                   "--baseline_checkpoint", m["baseline"],
                   "--per_seed_dir", m["per_seed_dir"],
                   "--batch_size", str(m["batch"])] + m["extra"]
            if d != 0:
                cmd += ["--checkpoint_overrides", f"{d}={ck}"]
            jobs.append((f"{m['name']}:{cond}", cmd))
    return jobs, skipped_done, skipped_missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="print the plan, run nothing")
    args = ap.parse_args()

    jobs, done, missing = build_jobs()
    wave = PER_GPU * len(GPUS)
    print(f"[leftover] {len(jobs)} to run  |  {done} already done  |  {missing} skipped (no checkpoint)")
    for name, _ in jobs:
        print("   RUN ", name)
    if args.dry or not jobs:
        print("[dry] nothing executed." if args.dry else "[done] nothing left to run.")
        return

    n_waves = (len(jobs) + wave - 1) // wave
    for w in range(0, len(jobs), wave):
        batch = jobs[w:w + wave]
        procs = []
        print(f"\n===== wave {w // wave + 1}/{n_waves}  ({len(batch)} jobs) =====")
        for i, (name, cmd) in enumerate(batch):
            gpu = GPUS[i // PER_GPU]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            print(f"  GPU{gpu} <- {name}")
            procs.append((name, subprocess.Popen(cmd, env=env)))
        for name, p in procs:
            rc = p.wait()
            print(f"  [{'ok' if rc == 0 else 'FAIL rc=%d' % rc}] {name}")
    print("\n[done] all leftover evaluations complete.")


if __name__ == "__main__":
    main()
