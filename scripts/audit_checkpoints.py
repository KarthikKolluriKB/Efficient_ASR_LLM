"""
Audit which projector checkpoints exist on disk vs which configs expect them.

Walks two trees:
    outputs/<language>/...      -- where trained `checkpoint_best_wer.pt` files live
    configs/<model>/<language>/ -- where the training/eval configs expect them

and produces a coverage matrix per (model_size, language, training recipe).
"Missing" means a config exists but no checkpoint can be found that matches it.

Why this matters: the depth-sweep driver needs a checkpoint for every depth
it tries to evaluate. The bias-pruning experiment has 3 candidate model
sizes (small=12L, medium=24L, large-v2=32L), 3 candidate languages (en,
da, nl), and 2 training recipes (full-projector, LoRA). Without a quick
"what do we have?" view, planning the next sweep is guesswork.

Filename / folder conventions observed in this repo:
    outputs/<lang>/whisper-s_baseline[ _lora ]/                  -> small, baseline
    outputs/<lang>/whisper-s_ablation_<N>L[ _lora ]/             -> small, N layers kept
    outputs/<lang>/whisper-small/ablation_<N>L/                  -> small, N layers (alt)
    outputs/<lang>/whisper-medium/{baseline,ablation_<N>L}/      -> medium
    outputs/<lang>/whisper-m_*                                   -> medium (alt naming)
    outputs/<lang>/whisper-largev2/{baseline,ablation_<N>L}/     -> large-v2
    outputs/<lang>/whisper-l_*                                   -> large-v2 (alt naming)
Any path containing 'lora' (case-insensitive) is classified recipe=lora.

Usage:
    python scripts/audit_checkpoints.py
    python scripts/audit_checkpoints.py --csv_out checkpoint_coverage.csv
    python scripts/audit_checkpoints.py --outputs_dir /scratch/karthik/outputs
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Encoder layer counts for the three model sizes we care about.
TOTAL_LAYERS = {"small": 12, "medium": 24, "large-v2": 32}

# Languages this audit knows how to talk about. Anything else under
# outputs/ is reported as "other languages" but not detailed.
KNOWN_LANGUAGES = ["english", "danish", "dutch"]

# Mapping from the model-size hint found in a path to the canonical size name.
MODEL_SIZE_HINTS = [
    ("whisper-largev2", "large-v2"),
    ("whisper-large-v2", "large-v2"),
    ("whisper-l_",      "large-v2"),
    ("whisper-medium",  "medium"),
    ("whisper-m_",      "medium"),
    ("whisper-small",   "small"),
    ("whisper-s_",      "small"),
]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _detect_model_size(tokens) -> str | None:
    """Find a known model size hint in any of the path tokens. Tolerates
    both 'whisper-largev2' / 'whisper-s_' / 'whisper-m_' style and the
    underscore form 'whisper_largev2' / 'whisper_small' / 'whisper_base'.
    """
    blob = "/".join(t.lower() for t in tokens)
    for hint, canonical in MODEL_SIZE_HINTS:
        if hint in blob:
            return canonical
    # `whisper_base` and `whisper-base` appear in the server outputs but are
    # not size variants we depth-sweep on. Surface them so they're not silent
    # orphans, just classified as 'base' (not in TOTAL_LAYERS).
    if "whisper_base" in blob or "whisper-base" in blob:
        return "base"
    return None


def parse_ckpt_path(p: Path, outputs_root: Path) -> dict | None:
    """Extract (language, model_size, layers_kept, recipe) from a checkpoint path.

    Handles two folder layouts seen in this project:
        Layout 1: outputs/<language>/<encoder_folder>/checkpoint_best_wer.pt
        Layout 2: outputs/<model>/<language>/<config>/checkpoint_best_wer.pt

    Returns None if any of the four pieces couldn't be confidently parsed.
    """
    try:
        rel = p.relative_to(outputs_root)
    except ValueError:
        return None
    parts = rel.parts
    if not parts:
        return None

    # Detect layout by checking whether parts[0] is a known model token
    # (underscore form) or a language.
    first = parts[0].lower()
    is_layout2 = first.startswith("whisper_") or first.startswith("whisper-")

    if is_layout2:
        # outputs/<model>/<lang>/<config>/...
        if len(parts) < 3:
            return None
        language = parts[1].lower()
        size_tokens = [parts[0]]
        config_tokens = list(parts[2:])  # the rest (config folder + filename)
    else:
        # outputs/<lang>/<encoder_folder...>/...
        language = parts[0].lower()
        size_tokens = list(parts[1:])
        config_tokens = list(parts[1:])

    model_size = _detect_model_size(size_tokens)
    if model_size is None:
        return None

    # Recipe — any "lora" token anywhere classifies as lora; otherwise full.
    blob = "/".join(t.lower() for t in config_tokens)
    recipe = "lora" if "lora" in blob else "full"

    # Layers kept: 'baseline'/'baseline_<X>' => total, otherwise look for
    # any '_<N>L' token (handles ablation_NL, baseline_LoRA_NL, etc.).
    layers_kept = None
    if "baseline" in blob and not re.search(r"_(\d+)l(?:[_/]|$)", blob):
        # Pure baseline with no layer suffix → use the full layer count.
        layers_kept = TOTAL_LAYERS.get(model_size)
    else:
        m = re.search(r"_(\d+)l(?:[_/]|$)", blob)
        if m:
            layers_kept = int(m.group(1))
        elif "baseline" in blob:
            layers_kept = TOTAL_LAYERS.get(model_size)
    if layers_kept is None:
        return None

    total = TOTAL_LAYERS.get(model_size, 0)
    return {
        "path": p,
        "language": language,
        "model_size": model_size,
        "layers_kept": layers_kept,
        "prune_depth": (total - layers_kept) if total else None,
        "recipe": recipe,
    }


def parse_config_path(p: Path, configs_root: Path) -> dict | None:
    """Map a YAML config (configs/<model>/<lang>/<train|eval|LoRA>/<x>.yaml)
    to the same (lang, size, layers, recipe) tuple the checkpoint parser uses.
    Returns None for anything that doesn't match the expected layout.
    """
    try:
        rel = p.relative_to(configs_root)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 4:
        return None
    # parts[0] = "whisper_small" | "whisper_medium" | "whisper_largev2"
    # parts[1] = language
    # parts[2] = "train" | "eval" | "LoRA"   (under LoRA there are train/eval subdirs)
    # parts[-1] = "<name>.yaml"
    size_map = {"whisper_small": "small",
                "whisper_medium": "medium",
                "whisper_largev2": "large-v2"}
    if parts[0] not in size_map:
        return None
    model_size = size_map[parts[0]]
    language = parts[1].lower()
    recipe = "lora" if "lora" in [s.lower() for s in parts] else "full"

    name = p.stem.lower()
    # "baseline.yaml" or e.g. "whisper-s_baseline.yaml" -> baseline
    # "ablation_<N>L.yaml" or "whisper-s_ablation_<N>L.yaml" -> N kept
    if "baseline" in name:
        layers_kept = TOTAL_LAYERS[model_size]
    else:
        m = re.search(r"ablation_(\d+)l", name)
        if not m:
            return None
        layers_kept = int(m.group(1))

    # We don't want to double-count train + eval configs as separate rows.
    return {
        "path": p,
        "language": language,
        "model_size": model_size,
        "layers_kept": layers_kept,
        "prune_depth": TOTAL_LAYERS[model_size] - layers_kept,
        "recipe": recipe,
    }


# ---------------------------------------------------------------------------
# Walkers
# ---------------------------------------------------------------------------

def find_checkpoints(outputs_root: Path) -> tuple[list[dict], list[Path]]:
    """Find every checkpoint_best_wer.pt under outputs/. Returns (parsed, orphans)."""
    parsed: list[dict] = []
    orphans: list[Path] = []
    if not outputs_root.exists():
        print(f"[Audit] WARNING: {outputs_root} does not exist.")
        return parsed, orphans
    for p in outputs_root.rglob("checkpoint_best_wer.pt"):
        info = parse_ckpt_path(p, outputs_root)
        if info is None:
            orphans.append(p)
        else:
            parsed.append(info)
    return parsed, orphans


def find_configs(configs_root: Path) -> set[tuple]:
    """Collect (language, model_size, layers_kept, recipe) tuples from configs.
    Returns a SET to dedup train/eval pairs.
    """
    out: set[tuple] = set()
    if not configs_root.exists():
        return out
    for p in configs_root.rglob("*.yaml"):
        info = parse_config_path(p, configs_root)
        if info is None:
            continue
        # Restrict to known model sizes / languages so we don't drown in noise.
        if info["model_size"] not in TOTAL_LAYERS:
            continue
        out.add((info["language"], info["model_size"], info["layers_kept"], info["recipe"]))
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def make_coverage(parsed_ckpts: list[dict], expected_configs: set[tuple]) -> list[dict]:
    """Build a row per (language, model_size, recipe, layers_kept) showing
    whether a checkpoint is present.
    """
    have: dict[tuple, Path] = {}
    for info in parsed_ckpts:
        key = (info["language"], info["model_size"], info["layers_kept"], info["recipe"])
        have.setdefault(key, info["path"])

    rows: list[dict] = []
    all_keys = expected_configs | set(have.keys())
    for (lang, size, kept, recipe) in sorted(all_keys):
        depth = TOTAL_LAYERS.get(size, 0) - kept
        rows.append({
            "language": lang,
            "model_size": size,
            "recipe": recipe,
            "layers_kept": kept,
            "prune_depth": depth,
            "expected_by_config": (lang, size, kept, recipe) in expected_configs,
            "checkpoint_present": (lang, size, kept, recipe) in have,
            "checkpoint_path": str(have.get((lang, size, kept, recipe), "")),
        })
    return rows


def print_summary(rows: list[dict]) -> None:
    """Pretty-print one matrix per (model_size, language, recipe)."""
    by_block: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_block[(r["language"], r["model_size"], r["recipe"])].append(r)

    # Process known languages first for predictable output, then anything else.
    known = [(lang, size, recipe) for (lang, size, recipe) in by_block
             if lang in KNOWN_LANGUAGES and size in TOTAL_LAYERS]
    other = [k for k in by_block if k not in known]

    for key in sorted(known) + sorted(other):
        lang, size, recipe = key
        block = sorted(by_block[key], key=lambda r: r["prune_depth"])
        total = TOTAL_LAYERS.get(size)
        if total is None:
            print(f"\n=== {lang} / {size} / {recipe} (unknown size) ===")
            continue
        print(f"\n=== {lang} / whisper-{size} ({total}L) / {recipe} ===")
        print(f"{'depth':>5} {'kept':>5}  {'config?':<8} {'ckpt?':<8}  {'path'}")
        print(f"{'-'*5} {'-'*5}  {'-'*8} {'-'*8}  {'-'*4}")
        # Build a complete row per expected prune depth (0..total) so missing
        # entries are visible, not just absent.
        existing = {r["prune_depth"]: r for r in block}
        any_expected = any(r["expected_by_config"] for r in block)
        for depth in range(0, total + 1):
            kept = total - depth
            r = existing.get(depth)
            if r is None:
                # Not in this block at all — only show if a config existed at
                # this (size, recipe, lang, depth) — otherwise skip to keep
                # the table compact.
                continue
            cfg_flag = "yes" if r["expected_by_config"] else "—"
            ckpt_flag = "yes" if r["checkpoint_present"] else "MISSING"
            path = r["checkpoint_path"] or ""
            print(f"{depth:>5d} {kept:>5d}  {cfg_flag:<8} {ckpt_flag:<8}  {path}")
        if any_expected:
            missing = [r for r in block if r["expected_by_config"] and not r["checkpoint_present"]]
            if missing:
                gaps = ", ".join(f"depth {r['prune_depth']} (keep {r['layers_kept']})" for r in missing)
                print(f"   GAPS: {gaps}")
            else:
                print("   coverage: complete for every config in this block")

    print(f"\n[Audit] Total checkpoint rows: {len(rows)}")
    n_present = sum(1 for r in rows if r["checkpoint_present"])
    n_expected = sum(1 for r in rows if r["expected_by_config"])
    n_missing = sum(1 for r in rows if r["expected_by_config"] and not r["checkpoint_present"])
    print(f"[Audit] Configs expect: {n_expected}    Checkpoints present: {n_present}    "
          f"Missing (config but no ckpt): {n_missing}")


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["language", "model_size", "recipe", "layers_kept", "prune_depth",
              "expected_by_config", "checkpoint_present", "checkpoint_path"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[Audit] Wrote {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Audit projector checkpoint coverage.")
    p.add_argument("--outputs_dir", type=Path, default=PROJECT_ROOT / "outputs",
                   help="Where trained projectors live.")
    p.add_argument("--configs_dir", type=Path, default=PROJECT_ROOT / "configs",
                   help="Where YAML configs live (used to know what was expected).")
    p.add_argument("--language", nargs="+", default=None,
                   help="Filter to one or more languages (english / danish / dutch). "
                        "Default: all.")
    p.add_argument("--model_size", nargs="+", default=None,
                   choices=sorted(TOTAL_LAYERS.keys()),
                   help="Filter to one or more sizes (small / medium / large-v2). "
                        "Default: all.")
    p.add_argument("--recipe", nargs="+", default=None,
                   choices=["full", "lora"],
                   help="Filter to one or more training recipes. Default: all.")
    p.add_argument("--csv_out", type=Path, default=None,
                   help="Optional: also dump the coverage table to a CSV.")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[Audit] outputs dir: {args.outputs_dir}")
    print(f"[Audit] configs dir: {args.configs_dir}")

    parsed, orphans = find_checkpoints(args.outputs_dir)
    expected = find_configs(args.configs_dir)
    print(f"[Audit] checkpoints parsed: {len(parsed)}  orphans (path not understood): {len(orphans)}")
    print(f"[Audit] (lang, size, kept, recipe) tuples expected by configs: {len(expected)}")

    rows = make_coverage(parsed, expected)

    # Apply filters (keep behaviour identical when no filters are passed).
    lang_filter = {l.lower() for l in args.language} if args.language else None
    size_filter = set(args.model_size) if args.model_size else None
    recipe_filter = set(args.recipe) if args.recipe else None

    def _keep(r):
        if lang_filter is not None and r["language"] not in lang_filter:
            return False
        if size_filter is not None and r["model_size"] not in size_filter:
            return False
        if recipe_filter is not None and r["recipe"] not in recipe_filter:
            return False
        return True

    filtered = [r for r in rows if _keep(r)]
    if lang_filter or size_filter or recipe_filter:
        print(f"[Audit] Filters: language={sorted(lang_filter) if lang_filter else 'any'}, "
              f"model_size={sorted(size_filter) if size_filter else 'any'}, "
              f"recipe={sorted(recipe_filter) if recipe_filter else 'any'}")
        print(f"[Audit] Rows after filter: {len(filtered)} / {len(rows)}")

    print_summary(filtered)

    # Surface every classified checkpoint (so the user sees what was found
    # even when no config expects it — e.g. extra unused ablation depths).
    extras = []
    for info in parsed:
        if lang_filter is not None and info["language"] not in lang_filter:
            continue
        if size_filter is not None and info["model_size"] not in size_filter:
            continue
        if recipe_filter is not None and info["recipe"] not in recipe_filter:
            continue
        extras.append(info)
    if extras and (lang_filter or size_filter or recipe_filter):
        print("\n[Audit] Checkpoints matching filters (every parsed path, including ones with no eval config yet):")
        for info in sorted(extras, key=lambda r: (r["language"], r["model_size"], r["recipe"], r["layers_kept"])):
            print(f"   {info['language']:<8} whisper-{info['model_size']:<8} kept={info['layers_kept']:>3} "
                  f"recipe={info['recipe']:<6} {info['path']}")

    if orphans:
        print(f"\n[Audit] Orphan checkpoints (path layout not understood; {len(orphans)} total):")
        # Limit output to keep it scannable.
        for op in orphans[:25]:
            print(f"   {op}")
        if len(orphans) > 25:
            print(f"   ... and {len(orphans) - 25} more")

    if args.csv_out:
        write_csv(filtered, args.csv_out)


if __name__ == "__main__":
    main()
