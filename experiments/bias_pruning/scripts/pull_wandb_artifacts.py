"""
Pull bias-eval summary CSVs (per-axis / per-gender WER) from a wandb project to
a local folder, so a depth sweep run on the server can be aggregated anywhere.

The eval (evaluate_subgroup_wer.py) logs, per depth, a `bias_eval_*` artifact
containing:
    per_axis_wer/{cond}_seed{N}_multiaxis.csv   (gender+age+SES+ethnicity / L1...)
    per_gender_wer/{cond}_seed{N}.csv           (gender only)
    per_utterance/{cond}_seed{N}.csv            (one row per utterance)

By DEFAULT this pulls ONLY the summary files and NOT per_utterance, because some
datasets (Fair-Speech) forbid redistributing per-utterance transcripts — and a
wandb cloud download counts as redistribution. Pass --include_per_utterance only
for datasets whose license allows it (e.g. CC-BY-NC L2-ARCTIC for your own use).

Usage:
    python experiments/bias_pruning/scripts/pull_wandb_artifacts.py \
        --project whisper_small_bias_sweep_l2arctic \
        --out_dir experiments/bias_pruning/results/l2arctic_sweep/per_axis

    # explicit entity, and also grab per-utterance (license permitting):
    python .../pull_wandb_artifacts.py --entity my_team \
        --project whisper_small_bias_sweep_l2arctic \
        --out_dir .../per_axis --include_per_utterance
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Pull bias-eval artifact CSVs from a wandb project.")
    p.add_argument("--project", required=True, help="wandb project name.")
    p.add_argument("--entity", default=None,
                   help="wandb entity (team/user). Default: your configured default entity.")
    p.add_argument("--out_dir", type=Path, required=True,
                   help="Local folder to copy the CSVs into (flattened).")
    p.add_argument("--kinds", nargs="+", default=["per_axis_wer", "per_gender_wer"],
                   help="Artifact subfolders to extract (default: summary-level only).")
    p.add_argument("--include_per_utterance", action="store_true",
                   help="ALSO pull per_utterance/. Do NOT use for license-restricted "
                        "datasets (Fair-Speech forbids redistributing transcripts).")
    p.add_argument("--artifact_type", default="bias_eval_results",
                   help="Only download artifacts of this type.")
    return p.parse_args()


def main():
    import wandb

    args = parse_args()
    kinds = list(args.kinds)
    if args.include_per_utterance and "per_utterance" not in kinds:
        kinds.append("per_utterance")
        print("[Pull] WARNING: including per_utterance/. Ensure the dataset license "
              "permits redistributing per-utterance transcripts.")

    api = wandb.Api()
    path = f"{args.entity}/{args.project}" if args.entity else args.project
    runs = list(api.runs(path))
    print(f"[Pull] {len(runs)} run(s) in {path}; extracting {kinds}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_files = 0
    for run in runs:
        try:
            arts = run.logged_artifacts()
        except Exception as e:
            print(f"  {run.name}: could not list artifacts ({e})")
            continue
        for art in arts:
            if art.type != args.artifact_type:
                continue
            local = Path(art.download())
            for kind in kinds:
                for f in (local / kind).glob("*.csv"):
                    dest = args.out_dir / f.name
                    shutil.copyfile(f, dest)
                    n_files += 1
                    print(f"  {run.name}: {kind}/{f.name} -> {dest}")
    print(f"[Pull] Done. Copied {n_files} file(s) to {args.out_dir}")
    if n_files == 0:
        print("[Pull] Nothing copied. Either the runs predate the multi-axis code "
              "(no per_axis_wer artifact) or the entity/project is wrong.")


if __name__ == "__main__":
    main()
