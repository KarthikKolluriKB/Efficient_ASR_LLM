"""
Driver: cross-lingual LoRA bias evaluations — Danish + Dutch x {small, medium,
large-v2}. Zero-shot eval of the DA/NL LoRA checkpoints on Common Voice 22
(gender / age / accent; CV never collected race/SES).

Mirrors run_leftover_evals.py: expands each (scale x language) sweep into
single-depth jobs, runs only those whose LoRA checkpoint exists on disk (and
that aren't already done), and schedules them 3-per-GPU across 2 GPUs, wave by
wave (6 concurrent).

    python experiments/bias_pruning/scripts/run_da_nl_lora_evals.py --dry   # print plan
    python experiments/bias_pruning/scripts/run_da_nl_lora_evals.py         # run

BEFORE RUNNING, verify the two CONFIG blocks below:
  1. DA_TSV / NL_TSV  -> the Common Voice transcript/<lang>/test.tsv for the join.
  2. The checkpoint path templates in ckpt_path() -> match your server layout.
A --dry run prints the exact checkpoint + tsv path each job resolves to, so if
everything shows "skipped (no checkpoint)" the templates are wrong, not the data.
"""
from __future__ import annotations
import argparse
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
os.chdir(REPO)  # all paths below are repo-relative

SWEEP = "experiments/bias_pruning/scripts/run_depth_sweep.py"
RESULTS = "experiments/bias_pruning/results"

GPUS = [0, 1]      # available GPUs
PER_GPU = 3        # experiments per GPU per wave -> wave size = 6
SEED = 42

# ============================ CONFIG 1: transcripts ============================
# cv22_da / cv22_nl MUST receive the matching-language test.tsv (the auto-download
# default is English-only). Point these at the transcript you used for the base
# da/nl sweeps. Edit if your server path differs.
DA_TSV = "data/cv22_hf/da/test.tsv"
NL_TSV = "data/cv22_hf/nl/test.tsv"

LANGS = [
    dict(tag="da", dataset="cv22_da", cfg_lang="danish", tsv=DA_TSV,
         wandb="whisper_bias_sweep_da_lora"),
    dict(tag="nl", dataset="cv22_nl", cfg_lang="dutch", tsv=NL_TSV,
         wandb="whisper_bias_sweep_nl_lora"),
]

# One entry per model scale. Depth lists mirror the base da/nl sweeps; the
# existence check trims any depth whose checkpoint wasn't trained.
SCALES = [
    dict(scale="largev2", stub="whisper-largev2", total_layers=32, batch=48,
         depths=[0, 2, 4, 6, 8, 10, 12, 14, 16], acase="l"),   # ablation_{k}l_lora
    dict(scale="medium",  stub="whisper-medium",  total_layers=24, batch=64,
         depths=[0, 2, 4, 6, 8, 10, 12, 14, 16], acase="L"),   # ablation_{k}L_lora
    dict(scale="small",   stub="whisper-small",   total_layers=12, batch=64,
         depths=list(range(0, 12)),               acase="L"),
]

# ========================= CONFIG 2: checkpoint paths =========================
def ckpt_path(langdir: str, sc: dict, keep: int, baseline: bool) -> str:
    """Resolve a DA/NL LoRA checkpoint path. Edit here if the layout differs."""
    root = f"outputs/{langdir}/{sc['stub']}"
    if baseline:
        return f"{root}/baseline_lora/checkpoint_best_wer.pt"
    return f"{root}/ablation_{keep}{sc['acase']}_lora/checkpoint_best_wer.pt"


def build_jobs():
    jobs, skipped_done, skipped_missing, missing_paths = [], 0, 0, []
    for lang in LANGS:
        langdir = lang["cfg_lang"]
        for sc in SCALES:
            model_dir = f"configs/whisper_{sc['scale']}/{langdir}/LoRA/eval"
            per_seed_dir = f"{RESULTS}/{sc['scale']}_{lang['tag']}_LORA_sweep"
            baseline_ck = ckpt_path(langdir, sc, sc["total_layers"], baseline=True)
            for d in sc["depths"]:
                keep = sc["total_layers"] - d
                cond = f"d{d:02d}_keep{keep:02d}"
                out = os.path.join(per_seed_dir, f"{cond}_seed{SEED}_multiaxis.csv")
                if os.path.exists(out):
                    skipped_done += 1
                    continue
                ck = baseline_ck if d == 0 else ckpt_path(langdir, sc, keep, baseline=False)
                if not os.path.exists(ck):
                    skipped_missing += 1
                    missing_paths.append(ck)
                    continue
                cmd = ["python", SWEEP,
                       "--dataset", lang["dataset"],
                       "--model_dir", model_dir,
                       "--total_layers", str(sc["total_layers"]),
                       "--depths", str(d),
                       "--seed", str(SEED),
                       "--baseline_checkpoint", baseline_ck,
                       "--per_seed_dir", per_seed_dir,
                       "--batch_size", str(sc["batch"]),
                       "--wandb_project", lang["wandb"],
                       "--cv_test_tsv", lang["tsv"]]
                if d != 0:
                    cmd += ["--checkpoint_overrides", f"{d}={ck}"]
                jobs.append((f"{sc['scale']}_{lang['tag']}_LoRA:{cond}", cmd))
    return jobs, skipped_done, skipped_missing, missing_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--show-missing", action="store_true",
                    help="also list the checkpoint paths that were not found")
    args = ap.parse_args()

    # Fail fast if a transcript is missing (only matters for a real run).
    if not args.dry:
        for lang in LANGS:
            if not os.path.exists(lang["tsv"]):
                raise SystemExit(
                    f"[error] {lang['tag']} transcript not found: {lang['tsv']}\n"
                    f"        Fix DA_TSV/NL_TSV at the top of this script.")

    jobs, done, missing, missing_paths = build_jobs()
    wave = PER_GPU * len(GPUS)
    print(f"[da/nl lora] {len(jobs)} to run  |  {done} already done  |  "
          f"{missing} skipped (no checkpoint)")
    for name, cmd in jobs:
        # surface the resolved checkpoint so a dry run is verifiable at a glance
        ck = cmd[cmd.index("--baseline_checkpoint") + 1]
        if "--checkpoint_overrides" in cmd:
            ck = cmd[cmd.index("--checkpoint_overrides") + 1].split("=", 1)[1]
        print(f"   RUN  {name:34s} <- {ck}")
    if args.show_missing and missing_paths:
        print("\n[skipped — checkpoint not found]")
        for p in missing_paths:
            print("   MISS", p)
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
    print("\n[done] all da/nl LoRA evaluations complete.")


if __name__ == "__main__":
    main()
