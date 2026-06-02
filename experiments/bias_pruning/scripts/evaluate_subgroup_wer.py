"""
Step 4 — Per-utterance inference + per-gender WER (with bootstrap CIs).

Reuses the project's existing inference pipeline (model_builder,
SpeechDatasetHF) so the model behaviour is identical to `eval.py`. The
extra work here is:

    1. Streaming results to a per-utterance CSV keyed by (speaker_id, key,
       sentence) so demographics can be joined afterwards.
    2. Joining demographics from the upstream CV22 `test.tsv` on
       (speaker_id_prefix, normalised_sentence). The preprocessed HF dataset
       stores `speaker_id` truncated to 16 chars and drops gender/age/accent,
       so the join is the only way to recover them without re-building the
       data.
    3. Computing per-gender WER with bootstrap CIs using bootstrap_ci.

Outputs:
    {output_path}                     -- per-utterance CSV
    results/per_seed_wer/{condition}_{seed}.csv  -- per-gender summary

Usage:
    python experiments/bias_pruning/scripts/evaluate_subgroup_wer.py \
        --config configs/whisper_medium/english/eval/baseline.yaml \
        --checkpoint_path outputs/english/whisper-medium/baseline/checkpoint_best_wer.pt \
        --prune_depth 0 --seed 42 --condition unpruned

The acceptance-criterion sanity check (aggregate WER within +/-0.5 abs
points of paper 1) is printed at the end; pass --target_wer to also exit
with a non-zero status if the gap is too wide.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datamodule.dataset import get_speech_dataset  # noqa: E402
from models.model import model_builder  # noqa: E402
from utils.wand_config import init_wandb  # noqa: E402

# Local imports
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from bootstrap_ci import wer_with_ci, cer_with_ci  # noqa: E402


GENDER_BUCKETS = ["male", "female", "other", "missing"]
ANALYSABLE_MIN_SECONDS = 30 * 60
ANALYSABLE_MIN_UTTS = 200

DEFAULT_PER_SEED_DIR = PROJECT_ROOT / "experiments/bias_pruning/results/per_seed_wer"


# ---------------------------------------------------------------------------
# Demographics join
# ---------------------------------------------------------------------------

def _norm_gender(raw: str | None) -> str:
    if raw is None:
        return "missing"
    g = raw.strip().lower()
    if g in {"", "nan", "none"}:
        return "missing"
    if g.startswith("male") or g == "m" or g == "male_masculine":
        return "male"
    if g.startswith("female") or g == "f" or g == "female_feminine":
        return "female"
    return "other"


def _normalise_sentence(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^\w\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_cv_demographics(tsv_path: Path | None) -> dict[tuple[str, str], dict]:
    """Load CV22 en/test.tsv into a (client_id_prefix, normalised_sentence) -> demographics map.

    If `tsv_path` is None, downloads it via huggingface_hub. The prefix is
    the first 16 chars of `client_id` because the HF preprocessing pipeline
    truncates it to that length.
    """
    if tsv_path is None:
        from huggingface_hub import hf_hub_download
        tsv_path = Path(hf_hub_download(
            repo_id="fsicoli/common_voice_22_0",
            filename="transcript/en/test.tsv",
            repo_type="dataset",
        ))
    out: dict[tuple[str, str], dict] = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            client_id = row.get("client_id", "") or ""
            sent = _normalise_sentence(row.get("sentence", ""))
            key = (client_id[:16], sent)
            out[key] = {
                "client_id_full": client_id,
                "gender": _norm_gender(row.get("gender")),
                "age": (row.get("age") or "").strip().lower() or "missing",
                "accent": (row.get("accents") or row.get("accent") or "").strip().lower() or "missing",
                "path": row.get("path", ""),
            }
    return out


# ---------------------------------------------------------------------------
# Inference (mirrors eval.py)
# ---------------------------------------------------------------------------

def _truncate_for_generation(input_ids, attention_mask, labels, modality_mask, device):
    """Copy of eval.py's truncate_for_generation; unused here (kept for
    parity if labels-driven decoding ever returns)."""
    raise NotImplementedError("Not used: we rely on dataset inference_mode=True.")


@torch.no_grad()
def run_inference(
    cfg,
    checkpoint_path: Path,
    device: torch.device,
    split: str,
) -> list[dict]:
    train_cfg = cfg.train
    model_cfg = cfg.model
    data_cfg = cfg.data
    eval_cfg = cfg.eval if hasattr(cfg, "eval") else None

    # Force inference mode regardless of config.
    if hasattr(data_cfg, "inference_mode"):
        data_cfg.inference_mode = True

    # Generation knobs (mirror eval.py priorities).
    max_new_tokens = getattr(eval_cfg, "max_new_tokens", 128) if eval_cfg else 128
    num_beams = getattr(eval_cfg, "num_beams", 1) if eval_cfg else 1
    do_sample = getattr(eval_cfg, "do_sample", False) if eval_cfg else False
    repetition_penalty = getattr(eval_cfg, "repetition_penalty", 1.0) if eval_cfg else 1.0
    length_penalty = getattr(eval_cfg, "length_penalty", 1.0) if eval_cfg else 1.0
    temperature = getattr(eval_cfg, "temperature", 1.0) if eval_cfg else 1.0
    batch_size = getattr(eval_cfg, "batch_size", 16) if eval_cfg else 16

    print(f"[Eval] gen settings: num_beams={num_beams}, max_new_tokens={max_new_tokens}, "
          f"rep_pen={repetition_penalty}, len_pen={length_penalty}, batch_size={batch_size}")

    # Build model
    print(f"[Eval] Building model from config; loading checkpoint: {checkpoint_path}")
    model, tokenizer = model_builder(train_cfg, model_cfg, data_config=data_cfg)
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    projector_state = ckpt["projector"] if isinstance(ckpt, dict) and "projector" in ckpt else ckpt
    model.projector.load_state_dict(projector_state)
    use_lora = getattr(train_cfg, "use_lora", False)
    if use_lora and isinstance(ckpt, dict) and "lora" in ckpt:
        try:
            model.llm.load_state_dict(ckpt["lora"], strict=False)
            print("[Eval] Loaded LoRA adapter weights.")
        except Exception as e:
            print(f"[Eval] WARNING: could not load LoRA weights: {e}")

    model = model.to(device)
    model.eval()

    test_dataset = get_speech_dataset(data_cfg, tokenizer, split=split)
    print(f"[Eval] Test dataset: {len(test_dataset)} samples (split={split})")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_dataset.collator,
        num_workers=getattr(eval_cfg, "num_workers", 4) if eval_cfg else 4,
        pin_memory=(device.type == "cuda"),
    )

    # Mixed-precision dtype follows training config.
    use_autocast = bool(getattr(train_cfg, "mixed_precision", False)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    enc_dtype = amp_dtype if use_autocast else torch.float32

    rows: list[dict] = []
    # Track row index against the dataset's `valid_indices` so we can join
    # back to the HF dataset for speaker_id and duration.
    valid_indices = test_dataset.valid_indices
    hf_dataset = test_dataset.hf_dataset

    cursor = 0  # position in the (filtered) dataset; advances by batch size
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Inference")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        audio_mel = batch["audio_mel"].to(device).to(enc_dtype)
        modality_mask = batch["modality_mask"].to(device)

        ref_texts = batch.get("targets")
        keys = batch.get("keys")
        if ref_texts is None or keys is None:
            raise RuntimeError("Batch missing `targets`/`keys` — set data.inference_mode=true.")

        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                gen_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    audio_mel=audio_mel,
                    modality_mask=modality_mask,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    temperature=temperature,
                )
        else:
            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_mel=audio_mel,
                modality_mask=modality_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                temperature=temperature,
            )

        hyp_texts = [tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen_ids]

        for i, (hyp, ref, key) in enumerate(zip(hyp_texts, ref_texts, keys)):
            real_idx = valid_indices[cursor + i]
            sample_row = hf_dataset[real_idx]
            rows.append({
                "row_idx": int(real_idx),
                "key": str(key),
                "speaker_id": str(sample_row.get("speaker_id", "")),
                "duration_s": float(sample_row.get("duration", 0.0)),
                "reference": ref,
                "hypothesis": hyp,
            })
        cursor += len(hyp_texts)

        del input_ids, attention_mask, audio_mel, modality_mask, gen_ids
        if batch_idx % 20 == 0 and device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

    return rows


def join_demographics(rows: list[dict], demo_map: dict[tuple[str, str], dict]) -> tuple[list[dict], int]:
    """Mutates each row in-place with gender/age/accent/client_id_full.
    Returns (rows, n_unmatched).
    """
    n_unmatched = 0
    for r in rows:
        key = (r["speaker_id"][:16], _normalise_sentence(r["reference"]))
        demo = demo_map.get(key)
        if demo is None:
            n_unmatched += 1
            r.update({"gender": "missing", "age": "missing", "accent": "missing", "client_id_full": ""})
        else:
            r.update({
                "gender": demo["gender"],
                "age": demo["age"],
                "accent": demo["accent"],
                "client_id_full": demo["client_id_full"],
            })
    return rows, n_unmatched


def write_per_utterance_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["row_idx", "key", "speaker_id", "client_id_full",
              "gender", "age", "accent", "duration_s", "reference", "hypothesis"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _empty_metric() -> dict:
    return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}


def per_gender_summary(
    rows: list[dict],
    condition: str,
    seed: int,
    n_bootstrap: int = 1000,
) -> tuple[list[dict], dict]:
    by_gender: dict[str, list[dict]] = {g: [] for g in GENDER_BUCKETS}
    for r in rows:
        by_gender.setdefault(r["gender"], []).append(r)

    def _summarise(refs, hyps, do_bootstrap):
        nb = n_bootstrap if do_bootstrap else 0
        if not refs:
            return _empty_metric(), _empty_metric()
        w = wer_with_ci(refs, hyps, n_bootstrap=nb, seed=42)
        c = cer_with_ci(refs, hyps, n_bootstrap=nb, seed=42)
        return (
            {"point": w["wer"], "ci_low": w["ci_low"], "ci_high": w["ci_high"]},
            {"point": c["cer"], "ci_low": c["ci_low"], "ci_high": c["ci_high"]},
        )

    summary_rows = []
    for g in GENDER_BUCKETS:
        subset = by_gender.get(g, [])
        n = len(subset)
        dur = sum(float(r["duration_s"]) for r in subset)
        analysable = (dur >= ANALYSABLE_MIN_SECONDS) and (n >= ANALYSABLE_MIN_UTTS)
        wer, cer = _summarise(
            [r["reference"] for r in subset],
            [r["hypothesis"] for r in subset],
            do_bootstrap=analysable,
        )
        summary_rows.append({
            "condition": condition, "seed": seed, "gender": g,
            "n_utts": n, "duration_hours": round(dur / 3600.0, 4),
            "wer": wer["point"], "wer_ci_low": wer["ci_low"], "wer_ci_high": wer["ci_high"],
            "cer": cer["point"], "cer_ci_low": cer["ci_low"], "cer_ci_high": cer["ci_high"],
            "analysable": "yes" if analysable else "no",
        })

    # Aggregate (all utterances)
    agg_wer, agg_cer = _summarise(
        [r["reference"] for r in rows],
        [r["hypothesis"] for r in rows],
        do_bootstrap=True,
    )
    aggregate_row = {
        "condition": condition, "seed": seed, "gender": "ALL",
        "n_utts": len(rows),
        "duration_hours": round(sum(float(r["duration_s"]) for r in rows) / 3600.0, 4),
        "wer": agg_wer["point"], "wer_ci_low": agg_wer["ci_low"], "wer_ci_high": agg_wer["ci_high"],
        "cer": agg_cer["point"], "cer_ci_low": agg_cer["ci_low"], "cer_ci_high": agg_cer["ci_high"],
        "analysable": "yes",
    }
    summary_rows.append(aggregate_row)
    return summary_rows, aggregate_row


def write_summary_csv(summary_rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["condition", "seed", "gender", "n_utts", "duration_hours",
              "wer", "wer_ci_low", "wer_ci_high",
              "cer", "cer_ci_low", "cer_ci_high",
              "analysable"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)


def parse_args():
    p = argparse.ArgumentParser(description="Per-subgroup WER evaluation for the bias-pruning experiment.")
    p.add_argument("--config", type=Path, required=True,
                   help="Path to eval YAML config (e.g. configs/whisper_medium/english/eval/baseline.yaml).")
    p.add_argument("--checkpoint_path", type=Path, required=True,
                   help="Path to the projector checkpoint (.pt).")
    p.add_argument("--prune_depth", type=int, required=True,
                   help="Top-down prune depth. 0 == unpruned. For whisper-small 12L total: "
                        "prune_depth=N means (12-N) layers kept.")
    p.add_argument("--seed", type=int, required=True,
                   help="Training seed of this checkpoint (for filename + bookkeeping only).")
    p.add_argument("--condition", type=str, required=True,
                   help="Free-form condition label used in output filenames (e.g. 'unpruned', "
                        "'pruned_2L', 'depth_05_kept_07').")
    p.add_argument("--language", default="en")
    p.add_argument("--split", default="test")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cv_test_tsv", type=Path, default=None,
                   help="Path to upstream CV22 transcript/en/test.tsv. Downloaded if omitted.")
    p.add_argument("--output_path", type=Path, default=None,
                   help="Per-utterance results CSV. Default: experiments/bias_pruning/results/per_utterance/{condition}_{seed}.csv")
    p.add_argument("--per_seed_dir", type=Path, default=DEFAULT_PER_SEED_DIR)
    p.add_argument("--n_bootstrap", type=int, default=1000)
    p.add_argument("--target_wer", type=float, default=None,
                   help="Paper 1's reported aggregate WER for this condition (0-1 scale). "
                        "If provided, the script exits non-zero when the gap exceeds 0.005 (0.5 abs pts).")
    p.add_argument("--wandb_project", type=str, default=None,
                   help="Override the wandb project name. If unset, uses log.wandb_project_name "
                        "from the YAML config (or skips wandb if log.use_wandb is false there).")
    p.add_argument("--wandb_run_name", type=str, default=None,
                   help="Override the wandb run name. Default: '{condition}_seed{seed}'.")
    p.add_argument("--no_wandb", action="store_true",
                   help="Disable wandb entirely for this run, even if log.use_wandb is true in config.")
    return p.parse_args()


def _resolve_wandb_settings(cfg, args) -> tuple[bool, str, str]:
    """Return (use_wandb, project, run_name) from CLI overrides + config + defaults."""
    if args.no_wandb:
        return False, "", ""
    log_cfg = cfg.log if hasattr(cfg, "log") else None
    config_says_on = bool(getattr(log_cfg, "use_wandb", False)) if log_cfg else False
    use = config_says_on or bool(args.wandb_project)
    project = args.wandb_project or (getattr(log_cfg, "wandb_project_name", None) if log_cfg else None) \
              or "bias_pruning_eval"
    run_name = args.wandb_run_name or f"{args.condition}_seed{args.seed}"
    return use, project, run_name


def _log_artifacts_to_wandb(run, *, per_utt_csv: Path, summary_csv: Path,
                            aggregate_row: dict, summary_rows: list[dict],
                            condition: str, prune_depth: int, seed: int) -> None:
    """Upload outputs as wandb artifacts + log scalar/tabular metrics. No-op if run is None."""
    if run is None:
        return
    try:
        import wandb
    except ImportError:
        return

    # 1. Scalar summaries (top-level metric panel + tagged for sweep grouping)
    run.summary["condition"] = condition
    run.summary["prune_depth"] = prune_depth
    run.summary["seed"] = seed
    run.summary["aggregate/wer"] = aggregate_row["wer"]
    run.summary["aggregate/cer"] = aggregate_row["cer"]
    run.summary["aggregate/n_utts"] = aggregate_row["n_utts"]
    for r in summary_rows:
        g = r["gender"]
        if g == "ALL":
            continue
        run.summary[f"wer/{g}"] = r["wer"]
        run.summary[f"cer/{g}"] = r["cer"]
        run.summary[f"n_utts/{g}"] = r["n_utts"]
    # Step-style logs so the sweep can plot WER vs depth across runs.
    run.log({
        "depth/prune_depth": prune_depth,
        "depth/aggregate_wer": aggregate_row["wer"],
        "depth/aggregate_cer": aggregate_row["cer"],
        **{f"depth/wer_{r['gender']}": r["wer"] for r in summary_rows if r["gender"] != "ALL"},
        **{f"depth/cer_{r['gender']}": r["cer"] for r in summary_rows if r["gender"] != "ALL"},
    })

    # 2. Per-gender table (browseable in the wandb UI)
    table = wandb.Table(columns=["gender", "n_utts", "duration_hours",
                                  "wer", "wer_ci_low", "wer_ci_high",
                                  "cer", "cer_ci_low", "cer_ci_high", "analysable"])
    for r in summary_rows:
        table.add_data(r["gender"], r["n_utts"], r["duration_hours"],
                       r["wer"], r["wer_ci_low"], r["wer_ci_high"],
                       r["cer"], r["cer_ci_low"], r["cer_ci_high"], r["analysable"])
    run.log({"per_gender_summary": table})

    # 3. Artifacts: per-utterance CSV (license-aware — caller decides whether
    # to use this on restricted-license datasets) and per-gender summary CSV.
    art = wandb.Artifact(
        name=f"bias_eval_{condition}_seed{seed}",
        type="bias_eval_results",
        description=f"Per-utterance + per-gender outputs for condition={condition}, seed={seed}.",
        metadata={"condition": condition, "prune_depth": prune_depth, "seed": seed},
    )
    if per_utt_csv.exists():
        art.add_file(str(per_utt_csv), name=per_utt_csv.name)
    if summary_csv.exists():
        art.add_file(str(summary_csv), name=summary_csv.name)
    run.log_artifact(art)


def main():
    args = parse_args()

    if not args.config.exists():
        raise SystemExit(f"Config not found: {args.config}")
    if not args.checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint_path}")

    cfg = OmegaConf.load(str(args.config))
    device = torch.device(args.device)

    use_wandb, wandb_project, wandb_run_name = _resolve_wandb_settings(cfg, args)
    run = None
    if use_wandb:
        run = init_wandb(
            use_wand=True,
            project=wandb_project,
            run_name=wandb_run_name,
            tags=["bias_eval", args.condition, f"depth_{args.prune_depth}", f"seed_{args.seed}"],
            config={
                "condition": args.condition,
                "prune_depth": args.prune_depth,
                "seed": args.seed,
                "config_path": str(args.config),
                "checkpoint_path": str(args.checkpoint_path),
                "language": args.language,
                "split": args.split,
                "n_bootstrap": args.n_bootstrap,
            },
        )

    try:
        rows = run_inference(cfg, args.checkpoint_path, device, split=args.split)
        print(f"[Eval] Inference produced {len(rows)} rows.")

        print("[Eval] Loading CV22 demographics ...")
        demo_map = load_cv_demographics(args.cv_test_tsv)
        rows, n_unmatched = join_demographics(rows, demo_map)
        if n_unmatched:
            print(f"[Eval] WARNING: {n_unmatched} of {len(rows)} utterances did not match a CV22 demo row "
                  "(speaker_id prefix + normalised sentence). These are labelled gender=missing.")

        out_utt = args.output_path or (
            PROJECT_ROOT / "experiments/bias_pruning/results/per_utterance"
            / f"{args.condition}_seed{args.seed}.csv"
        )
        write_per_utterance_csv(rows, out_utt)
        print(f"[Eval] Wrote per-utterance CSV: {out_utt}")

        summary_rows, aggregate_row = per_gender_summary(
            rows, condition=args.condition, seed=args.seed, n_bootstrap=args.n_bootstrap,
        )
        summary_path = args.per_seed_dir / f"{args.condition}_seed{args.seed}.csv"
        write_summary_csv(summary_rows, summary_path)
        print(f"[Eval] Wrote per-gender summary: {summary_path}")

        # Pretty-print summary
        print("\n=== Per-gender WER / CER (this seed) ===")
        print(f"{'gender':<8} {'n':>6} {'hours':>7} "
              f"{'WER':>8} {'WER 95% CI':>20} "
              f"{'CER':>8} {'CER 95% CI':>20} {'analysable':>11}")
        for r in summary_rows:
            wci = f"[{r['wer_ci_low']*100:.2f}, {r['wer_ci_high']*100:.2f}]"
            cci = f"[{r['cer_ci_low']*100:.2f}, {r['cer_ci_high']*100:.2f}]"
            print(f"{r['gender']:<8} {r['n_utts']:>6d} {r['duration_hours']:>7.2f} "
                  f"{r['wer']*100:>7.2f}% {wci:>20} "
                  f"{r['cer']*100:>7.2f}% {cci:>20} {r['analysable']:>11}")

        print(f"\nAggregate WER = {aggregate_row['wer']*100:.2f}% | "
              f"Aggregate CER = {aggregate_row['cer']*100:.2f}% on {aggregate_row['n_utts']} utts.")
        if args.target_wer is not None:
            diff = abs(aggregate_row["wer"] - args.target_wer)
            ok = diff <= 0.005
            print(f"Target (paper 1): {args.target_wer*100:.2f}% | abs diff = {diff*100:.2f} pts | within 0.5 pts: {ok}")
            if not ok:
                print("[Eval] FAIL: aggregate WER deviates from paper 1 by more than 0.5 absolute points.")
                raise SystemExit(2)

        _log_artifacts_to_wandb(
            run,
            per_utt_csv=out_utt,
            summary_csv=summary_path,
            aggregate_row=aggregate_row,
            summary_rows=summary_rows,
            condition=args.condition,
            prune_depth=args.prune_depth,
            seed=args.seed,
        )
    finally:
        if run is not None:
            try:
                run.finish()
            except Exception as e:
                print(f"[Eval] wandb finish failed: {e}")


if __name__ == "__main__":
    main()
