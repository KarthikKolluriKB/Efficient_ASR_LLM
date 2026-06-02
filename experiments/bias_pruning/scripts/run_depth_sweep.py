"""
Run evaluate_subgroup_wer.py across a depth-sweep of checkpoints.

By default targets whisper-small English (12 encoder layers): every 1-layer
pruning depth from 0 (unpruned baseline = 12L kept) through 11 (= 1L kept).
Each call produces a wandb run (one per depth) plus the usual local outputs;
upload-to-wandb is handled inside evaluate_subgroup_wer.py.

The sweep is sequential — only one GPU is needed. Each run cleans up GPU
state before the next starts.

Usage:
    # Standard whisper-small English sweep, all 12 depths, seed 42:
    CUDA_VISIBLE_DEVICES=1 python experiments/bias_pruning/scripts/run_depth_sweep.py \
        --model_dir configs/whisper_small/english/eval \
        --checkpoint_root outputs/english/whisper-small \
        --baseline_checkpoint outputs/english/whisper-s_baseline/checkpoint_best_wer.pt \
        --wandb_project whisper_small_bias_sweep_en

    # Subset (e.g. just the mild-pruning end of the curve):
    python experiments/bias_pruning/scripts/run_depth_sweep.py \
        --depths 0 1 2 3 4 ...

    # Dry run — print what would be executed, don't run:
    python experiments/bias_pruning/scripts/run_depth_sweep.py --dry_run

The default --checkpoint_root assumes the layout used by the project's
training configs:
    {checkpoint_root}/ablation_{N}L/checkpoint_best_wer.pt  for N in 1..11
    {baseline_checkpoint}                                    for N=12 (unpruned)
Override individual files with --checkpoint_overrides if a few live elsewhere.
"""

from __future__ import annotations

import argparse
import gc
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVAL_SCRIPT = PROJECT_ROOT / "experiments/bias_pruning/scripts/evaluate_subgroup_wer.py"


def parse_args():
    p = argparse.ArgumentParser(description="Sweep evaluate_subgroup_wer.py over pruning depths.")
    p.add_argument("--model_dir", type=Path,
                   default=PROJECT_ROOT / "configs/whisper_small/english/eval",
                   help="Folder with baseline.yaml + ablation_NL.yaml configs.")
    p.add_argument("--total_layers", type=int, default=12,
                   help="Total encoder layers for this model (whisper-small=12, medium=24, large-v2=32).")
    p.add_argument("--depths", type=int, nargs="+", default=None,
                   help="Specific prune depths to run (0 = unpruned). Defaults to 0..total_layers-1.")
    p.add_argument("--checkpoint_root", type=Path,
                   default=PROJECT_ROOT / "outputs/english/whisper-small",
                   help="Folder with ablation_NL/checkpoint_best_wer.pt subfolders.")
    p.add_argument("--baseline_checkpoint", type=Path,
                   default=PROJECT_ROOT / "outputs/english/whisper-s_baseline/checkpoint_best_wer.pt",
                   help="Path to the unpruned baseline checkpoint (separate because the project's "
                        "baseline output folder is named differently from the ablation folders).")
    p.add_argument("--checkpoint_overrides", nargs="*", default=[],
                   help="Per-depth overrides, format 'depth=/abs/path/checkpoint.pt'. Example: "
                        "--checkpoint_overrides 2=/scratch/ckpts/depth2.pt 4=/scratch/ckpts/depth4.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_project", type=str, default="whisper_small_bias_sweep_en",
                   help="wandb project name for all runs in this sweep.")
    p.add_argument("--device", default=None,
                   help="Pass to evaluate_subgroup_wer.py. If unset, that script picks its own default. "
                        "Prefer CUDA_VISIBLE_DEVICES=N instead so each child sees just one GPU.")
    p.add_argument("--n_bootstrap", type=int, default=1000)
    p.add_argument("--cv_test_tsv", type=Path, default=None)
    p.add_argument("--per_seed_dir", type=Path, default=None)
    p.add_argument("--continue_on_error", action="store_true",
                   help="Don't stop the sweep when one depth fails.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print the commands that would run without executing them.")
    p.add_argument("--gap_seconds", type=int, default=10,
                   help="Pause between depths to let CUDA settle / wandb flush.")
    return p.parse_args()


def _condition_name(prune_depth: int, layers_kept: int) -> str:
    """Human-readable, sortable condition tag used as the filename + wandb run name."""
    return f"d{prune_depth:02d}_keep{layers_kept:02d}"


def _eval_config_for(depth: int, model_dir: Path, total_layers: int) -> Path:
    if depth == 0:
        return model_dir / "baseline.yaml"
    layers_kept = total_layers - depth
    return model_dir / f"ablation_{layers_kept}L.yaml"


def _checkpoint_for(depth: int, args, overrides: dict[int, Path]) -> Path:
    if depth in overrides:
        return overrides[depth]
    if depth == 0:
        return args.baseline_checkpoint
    layers_kept = args.total_layers - depth
    return args.checkpoint_root / f"ablation_{layers_kept}L" / "checkpoint_best_wer.pt"


def _parse_overrides(items: list[str]) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--checkpoint_overrides item must be 'depth=path', got: {item!r}")
        depth_s, path_s = item.split("=", 1)
        out[int(depth_s)] = Path(path_s)
    return out


def main():
    args = parse_args()
    overrides = _parse_overrides(args.checkpoint_overrides)
    depths = args.depths if args.depths is not None else list(range(0, args.total_layers))

    if not DEFAULT_EVAL_SCRIPT.exists():
        raise SystemExit(f"Missing eval script: {DEFAULT_EVAL_SCRIPT}")

    print(f"[Sweep] depths to evaluate: {depths}")
    print(f"[Sweep] wandb project: {args.wandb_project}")
    print(f"[Sweep] eval script:   {DEFAULT_EVAL_SCRIPT}")

    # Plan first — surface missing checkpoints before any GPU work.
    plan: list[dict] = []
    for d in depths:
        cfg = _eval_config_for(d, args.model_dir, args.total_layers)
        ckpt = _checkpoint_for(d, args, overrides)
        layers_kept = args.total_layers - d
        condition = _condition_name(d, layers_kept)
        plan.append({"depth": d, "layers_kept": layers_kept, "config": cfg,
                     "checkpoint": ckpt, "condition": condition})

    print(f"\n{'depth':>5} {'kept':>5}  {'condition':<14} {'config':<60} {'ckpt exists':<11}")
    for p_ in plan:
        print(f"{p_['depth']:>5d} {p_['layers_kept']:>5d}  {p_['condition']:<14} "
              f"{str(p_['config']):<60} {'yes' if p_['checkpoint'].exists() else 'NO':<11}")

    missing = [p_ for p_ in plan if not p_["checkpoint"].exists()]
    if missing and not args.dry_run:
        print(f"\n[Sweep] WARNING: {len(missing)} checkpoint(s) missing; those depths will be skipped.")

    if args.dry_run:
        print("\n[Sweep] DRY RUN — no commands executed.")
        return

    results: list[dict] = []
    for p_ in plan:
        if not p_["checkpoint"].exists():
            print(f"\n[Sweep] SKIP {p_['condition']} — checkpoint missing: {p_['checkpoint']}")
            results.append({"depth": p_["depth"], "status": "skip-missing-ckpt"})
            continue

        cmd = [
            sys.executable, str(DEFAULT_EVAL_SCRIPT),
            "--config", str(p_["config"]),
            "--checkpoint_path", str(p_["checkpoint"]),
            "--prune_depth", str(p_["depth"]),
            "--seed", str(args.seed),
            "--condition", p_["condition"],
            "--n_bootstrap", str(args.n_bootstrap),
            "--wandb_project", args.wandb_project,
            "--wandb_run_name", p_["condition"],
        ]
        if args.cv_test_tsv:
            cmd += ["--cv_test_tsv", str(args.cv_test_tsv)]
        if args.per_seed_dir:
            cmd += ["--per_seed_dir", str(args.per_seed_dir)]
        if args.device:
            cmd += ["--device", args.device]

        print(f"\n[Sweep] ===== depth={p_['depth']} kept={p_['layers_kept']} =====")
        print("        " + " ".join(cmd))
        start = time.time()
        try:
            subprocess.run(cmd, check=True)
            status = "ok"
        except subprocess.CalledProcessError as e:
            status = f"failed ({e.returncode})"
            print(f"[Sweep] FAILED depth={p_['depth']}: return code {e.returncode}")
            if not args.continue_on_error:
                print("[Sweep] Stopping sweep. Pass --continue_on_error to keep going.")
                results.append({"depth": p_["depth"], "status": status,
                                 "wall_time_s": time.time() - start})
                break

        wall = time.time() - start
        results.append({"depth": p_["depth"], "status": status, "wall_time_s": wall})
        print(f"[Sweep] depth={p_['depth']} done in {wall/60:.1f} min, status={status}")

        # Give CUDA + wandb a moment to settle before the next depth.
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        if args.gap_seconds > 0:
            time.sleep(args.gap_seconds)

    print(f"\n{'='*60}\nSWEEP SUMMARY ({len(results)} runs)\n{'='*60}")
    print(f"{'depth':>5}  {'status':<20}  {'wall_time':>10}")
    for r in results:
        wt = f"{r.get('wall_time_s', 0)/60:.1f} min" if r.get("wall_time_s") else "—"
        print(f"{r['depth']:>5d}  {r['status']:<20}  {wt:>10}")


if __name__ == "__main__":
    main()
