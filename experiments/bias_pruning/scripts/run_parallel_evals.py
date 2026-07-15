#!/usr/bin/env python3
"""Parallel subgroup-WER evaluation runner: 2 GPUs x 3 jobs each = 6 concurrent.

Fills the MISSING per-utterance results (everything except LoRA, which is pulled
from the server). Each job re-runs evaluate_subgroup_wer.py, writing per-utterance
output into a CORPUS-TAGGED folder (never a shared one) so runs cannot overwrite
across corpora.

Design
------
- 6 worker threads: workers 0-2 -> GPU 0, workers 3-5 -> GPU 1 (3 procs/GPU).
- Shared job queue; each worker pulls jobs until empty.
- Resumable: a job whose output_path already exists (and is non-trivial) is
  skipped, so you can re-launch after an interruption.
- Each job pins its GPU via CUDA_VISIBLE_DEVICES; the child sees a single device
  as cuda:0.

USAGE
-----
  python run_parallel_evals.py --list            # print the job matrix, no run
  python run_parallel_evals.py --dry-run         # print exact commands, no run
  python run_parallel_evals.py                   # run all missing jobs, 6-wide
  python run_parallel_evals.py --only small_cv22 # substring filter on job id

!!! BEFORE RUNNING: verify CONFIG/CKPT/DATA paths in CELLS below against the
server layout. Placeholders are marked TODO. Run --dry-run first.
"""
import argparse, os, queue, subprocess, threading, time, sys, re, glob
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]     # repo root (…/Efficient_ASR_LLM)
EVAL = "experiments/bias_pruning/scripts/evaluate_subgroup_wer.py"
# Isolated root for fresh re-evaluations — kept ENTIRELY separate from the old
# (possibly contaminated) per_utterance folders. Each cell gets its own subfolder
# `{corpus}_{scale}_{condition}/` so filenames cannot collide across corpora.
OUT_ROOT = PROJECT / "experiments/bias_pruning/results/reeval_results"
CONDITION = "base"        # these re-runs are all base condition (LoRA is pulled)
SEED = 42
GPUS = [0, 1]
JOBS_PER_GPU = 3
# per-process inference batch size (overrides the YAML). Effective GPU load is
# BATCH_SIZE x JOBS_PER_GPU (3 concurrent procs share each card) — raise to go
# faster, lower if you hit CUDA OOM. None = use the config's value.
BATCH_SIZE = 32

# encoder depth per scale (layer counts) and the prune depths to sweep
SCALE_LAYERS = {"largev2": 32, "medium": 24, "small": 12}
def depths_for(scale, step):
    n = SCALE_LAYERS[scale]
    return list(range(0, n, step))            # 0, step, 2*step, ... < n

# ---------------------------------------------------------------------------
# CELLS: one entry per (corpus, scale) to (re)evaluate. Edit paths to match the
# server. `config`/`ckpt` use {scale}/{depth}/{keep}; depth 0 -> baseline.
# `extra` holds dataset-specific flags (hf dataset path, demographic source).
# `step` is the prune-depth increment (small was swept in 1-layer steps).
# ---------------------------------------------------------------------------
def cfg(scale, lang, depth, cond="base"):
    sub = "LoRA/eval" if cond == "lora" else "eval"
    # baseline config name varies (baseline.yaml vs whisper-s_baseline.yaml);
    # ablations are consistently ablation_{N}L.yaml. Return first that exists.
    stems = [f"ablation_{depth}L"] if depth else ["baseline", "whisper-s_baseline"]
    for stem in stems:
        rel = f"configs/whisper_{scale}/{lang}/{sub}/{stem}.yaml"
        if (PROJECT / rel).exists():
            return rel
    return f"configs/whisper_{scale}/{lang}/{sub}/{stems[0]}.yaml"   # for the skip msg

# ---- checkpoint DISCOVERY (naming is inconsistent, so scan instead of guess) ----
# Maps (lang, scale, depth) -> checkpoint path by scanning outputs/. Handles:
#   whisper-s_baseline           (baseline, depth 0)
#   whisper-small/ablation_2L    (nested ablation)
#   whisper-s_ablation_6L        (flat ablation)
# and the medium/large equivalents. Excludes _lora and *big* variants.
def discover_checkpoints():
    """Map (cond, lang, scale, depth) -> checkpoint path. cond in {base, lora}.
    LoRA folders end in '_lora'/'_lora_final' (canonical only — sub-variants like
    _lora_me/_mid/_qv are skipped). Base excludes any _lora/_seed/big/rand_proj."""
    m, rankmap = {}, {}
    for fname, rank in (("checkpoint_best_wer.pt", 0), ("projector_best_wer.pt", 0),
                        ("checkpoint_final.pt", 1), ("projector_final.pt", 1)):
        for p in glob.glob(str(PROJECT / "outputs" / "**" / fname), recursive=True):
            rel = p.replace("\\", "/")
            low = rel.lower()
            if "big" in low or "_seed" in low or "rand_proj" in low:
                continue
            parent = os.path.basename(os.path.dirname(rel)).lower()   # ckpt folder
            if "_lora" in parent:
                cond = "lora"
                base = parent[:-6] if parent.endswith("_final") else parent
                if not base.endswith("_lora"):        # sub-variant -> skip
                    continue
            elif "_lora" in low:                       # lora token elsewhere in path
                continue
            else:
                cond = "base"
            lang = next((L for L in ("danish", "dutch", "english") if L in low), None)
            if lang is None:
                continue
            if "largev2" in low or "whisper-l" in low:
                scale = "largev2"
            elif "whisper-medium" in low or "whisper-m_" in low:
                scale = "medium"
            elif "whisper-small" in low or "whisper-s_" in low:
                scale = "small"
            else:
                continue
            md = re.search(r"ablation_(\d+)L", rel, re.IGNORECASE)
            depth = int(md.group(1)) if md else (0 if "baseline" in low else None)
            if depth is None:
                continue
            key = (cond, lang, scale, depth)
            if key not in m or rank < rankmap[key]:
                m[key], rankmap[key] = rel, rank
    return m

CKPTS = None      # lazily populated on first build_jobs()

def ckpt(scale, lang, depth, cond="base"):
    return CKPTS.get((cond, lang, scale, depth))           # None if not found

# English-trained system is reused zero-shot for Fair-Speech and L2-ARCTIC:
# same checkpoint/config as CV22-EN, only the eval dataset + demographic source
# change via `extra`.
CELLS = [
    # --- Small CV22-EN: full range (L-5..L-11 lost to collision; redo all) ---
    dict(corpus="cv22", scale="small", lang="english", step=1,
         demo="cv22_tsv", extra=[]),
    # --- Medium CV22-EN: full range ---
    dict(corpus="cv22", scale="medium", lang="english", step=2,
         demo="cv22_tsv", extra=[]),
    # --- Large-v2 CV22-EN: full range (uncomment to also redo large) ---
    # dict(corpus="cv22", scale="largev2", lang="english", step=2,
    #      demo="cv22_tsv", extra=[]),
    # --- Small Danish: incomplete (missing baseline); redo all ---
    # demo=hf_columns reads gender/age/accent from the HF dataset rows (cv22_tsv
    # would wrongly join against the ENGLISH CV22 transcript). VERIFY the Danish
    # HF dataset actually carries these columns; else a DA-specific tsv is needed.
    dict(corpus="da", scale="small", lang="danish", step=1,
         demo="hf_columns", extra=[]),
    # --- Medium L2-ARCTIC: only keep-22 (depth 2) corrupt; redo that depth ---
    dict(corpus="l2arctic", scale="medium", lang="english", step=2, depths=[2],
         demo="hf_columns",
         extra=["--hf_dataset_path", "data/l2arctic_hf"]),

    # ===================== LoRA (RQ2) — per-utterance never saved =============
    # English LoRA system (trained on CV22-EN) is reused zero-shot for
    # Fair-Speech / L2-ARCTIC; only the eval dataset + demo source change.
    # --- CV22-EN +LoRA ---
    dict(corpus="cv22", scale="largev2", lang="english", step=2, cond="lora",
         demo="cv22_tsv", extra=[]),
    dict(corpus="cv22", scale="medium", lang="english", step=2, cond="lora",
         demo="cv22_tsv", extra=[]),
    dict(corpus="cv22", scale="small", lang="english", step=1, cond="lora",
         demo="cv22_tsv", extra=[]),
    # --- Fair-Speech +LoRA (headline RQ2) ---
    dict(corpus="fairspeech", scale="largev2", lang="english", step=2, cond="lora",
         demo="hf_columns", extra=["--hf_dataset_path", "data/fairspeech_hf"]),
    dict(corpus="fairspeech", scale="medium", lang="english", step=2, cond="lora",
         demo="hf_columns", extra=["--hf_dataset_path", "data/fairspeech_hf"]),
    dict(corpus="fairspeech", scale="small", lang="english", step=1, cond="lora",
         demo="hf_columns", extra=["--hf_dataset_path", "data/fairspeech_hf"]),
    # --- Dutch +LoRA ---
    dict(corpus="nl", scale="largev2", lang="dutch", step=2, cond="lora",
         demo="hf_columns", extra=[]),
    dict(corpus="nl", scale="medium", lang="dutch", step=2, cond="lora",
         demo="hf_columns", extra=[]),
    dict(corpus="nl", scale="small", lang="dutch", step=1, cond="lora",
         demo="hf_columns", extra=[]),
    # --- Danish +LoRA ---
    dict(corpus="da", scale="largev2", lang="danish", step=2, cond="lora",
         demo="hf_columns", extra=[]),
    dict(corpus="da", scale="medium", lang="danish", step=2, cond="lora",
         demo="hf_columns", extra=[]),
    # --- L2-ARCTIC +LoRA (L1 axis; illustrative) ---
    dict(corpus="l2arctic", scale="largev2", lang="english", step=2, cond="lora",
         demo="hf_columns", extra=["--hf_dataset_path", "data/l2arctic_hf"]),
    dict(corpus="l2arctic", scale="medium", lang="english", step=2, cond="lora",
         demo="hf_columns", extra=["--hf_dataset_path", "data/l2arctic_hf"]),
    dict(corpus="l2arctic", scale="small", lang="english", step=1, cond="lora",
         demo="hf_columns", extra=["--hf_dataset_path", "data/l2arctic_hf"]),
    # ---- OPTIONAL: uncomment to re-evaluate ALL non-LoRA base cells into clean
    #      corpus-tagged folders (guarantees no contamination). Large batch. ----
    # dict(corpus="cv22", scale="largev2", lang="english", step=2, demo="cv22_tsv", extra=[]),
    # dict(corpus="cv22", scale="medium",  lang="english", step=2, demo="cv22_tsv", extra=[]),
    # dict(corpus="fairspeech", scale="largev2", lang="english", step=2,
    #      demo="hf_columns", extra=["--hf_dataset_path", "data/fairspeech_hf"]),
    # dict(corpus="fairspeech", scale="medium", lang="english", step=2,
    #      demo="hf_columns", extra=["--hf_dataset_path", "data/fairspeech_hf"]),
    # dict(corpus="fairspeech", scale="small", lang="english", step=1,
    #      demo="hf_columns", extra=["--hf_dataset_path", "data/fairspeech_hf"]),
    # dict(corpus="l2arctic", scale="largev2", lang="english", step=2,
    #      demo="hf_columns", extra=["--hf_dataset_path", "data/l2arctic_hf"]),
    # dict(corpus="l2arctic", scale="small", lang="english", step=1,
    #      demo="hf_columns", extra=["--hf_dataset_path", "data/l2arctic_hf"]),
    # dict(corpus="nl", scale="largev2", lang="dutch", step=2, demo="cv22_tsv", extra=[]),
    # dict(corpus="nl", scale="medium",  lang="dutch", step=2, demo="cv22_tsv", extra=[]),
    # dict(corpus="nl", scale="small",   lang="dutch", step=1, demo="cv22_tsv", extra=[]),
    # dict(corpus="da", scale="largev2", lang="danish", step=2, demo="cv22_tsv", extra=[]),
    # dict(corpus="da", scale="medium",  lang="danish", step=2, demo="cv22_tsv", extra=[]),
]

def build_jobs():
    global CKPTS
    if CKPTS is None:
        CKPTS = discover_checkpoints()
    jobs, skipped = [], []
    for cell in CELLS:
        scale, lang, corpus, step = cell["scale"], cell["lang"], cell["corpus"], cell["step"]
        cond = cell.get("cond", "base")
        depths = cell.get("depths", depths_for(scale, step))
        outdir = OUT_ROOT / f"{corpus}_{scale}_{cond}"       # isolated per cell
        for d in depths:
            keep = SCALE_LAYERS[scale] - d
            jid = f"{corpus}_{scale}_{cond}_d{d:02d}_keep{keep:02d}"
            out = outdir / f"d{d:02d}_keep{keep:02d}_seed{SEED}.csv"
            ck = ckpt(scale, lang, d, cond)
            cf = PROJECT / cfg(scale, lang, d, cond)
            if ck is None:
                skipped.append((jid, "no checkpoint")); continue
            if not cf.exists():
                skipped.append((jid, f"no config {cfg(scale, lang, d, cond)}")); continue
            cmd = [
                sys.executable, EVAL,
                "--config", cfg(scale, lang, d, cond),
                "--checkpoint_path", ck,
                "--prune_depth", str(d), "--seed", str(SEED),
                "--condition", jid,
                "--output_path", str(out),
                "--language", {"english":"en","dutch":"nl","danish":"da"}[lang],
                "--demographic_source", cell["demo"],
                "--no_wandb",
            ] + (["--batch_size", str(BATCH_SIZE)] if BATCH_SIZE else []) + cell["extra"]
            jobs.append(dict(id=jid, out=out, cmd=cmd, ckpt=ck))
    if skipped:
        print(f"[!] {len(skipped)} job(s) skipped (missing checkpoint/config):")
        for jid, why in skipped:
            print(f"    {jid}: {why}")
    return jobs

# ---------------------------------------------------------------------------
def worker(wid, gpu, q, results, lock):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    while True:
        try:
            job = q.get_nowait()
        except queue.Empty:
            return
        jid, out, cmd = job["id"], job["out"], job["cmd"]
        if out.exists() and out.stat().st_size > 1024:
            with lock: print(f"[w{wid} gpu{gpu}] SKIP {jid} (exists)")
            results.append((jid, "skip")); q.task_done(); continue
        out.parent.mkdir(parents=True, exist_ok=True)
        log = out.with_suffix(".log")
        with lock: print(f"[w{wid} gpu{gpu}] START {jid}")
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as lf:
            rc = subprocess.run(cmd, cwd=PROJECT, env=env, stdout=lf,
                                stderr=subprocess.STDOUT).returncode
        dt = time.time() - t0
        status = "ok" if rc == 0 else f"FAIL(rc={rc})"
        with lock:
            print(f"[w{wid} gpu{gpu}] {status} {jid}  ({dt/60:.1f} min)  log={log.name}")
        results.append((jid, status)); q.task_done()

def main():
    global BATCH_SIZE, JOBS_PER_GPU
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print job matrix and exit")
    ap.add_argument("--dry-run", action="store_true", help="print commands and exit")
    ap.add_argument("--only", default=None, help="substring filter on job id")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                    help=f"inference batch per process (default {BATCH_SIZE}); "
                         f"effective load = this x jobs-per-gpu")
    ap.add_argument("--jobs-per-gpu", type=int, default=JOBS_PER_GPU,
                    help=f"concurrent procs per GPU (default {JOBS_PER_GPU})")
    args = ap.parse_args()
    BATCH_SIZE, JOBS_PER_GPU = args.batch_size, args.jobs_per_gpu

    jobs = build_jobs()
    if args.only:
        jobs = [j for j in jobs if args.only in j["id"]]

    if args.list:
        for j in jobs:
            print(f"  {j['id']:34s}  ckpt={j['ckpt']}")
        print(f"\n{len(jobs)} jobs, {len(GPUS)} GPUs x {JOBS_PER_GPU} = "
              f"{len(GPUS)*JOBS_PER_GPU} concurrent")
        return
    if args.dry_run:
        for j in jobs:
            print(" ".join(str(x) for x in j["cmd"]))
        print(f"\n# {len(jobs)} jobs total", file=sys.stderr)
        return

    q = queue.Queue()
    for j in jobs:
        q.put(j)
    results, lock = [], threading.Lock()
    slots = [(g, s) for g in GPUS for s in range(JOBS_PER_GPU)]   # 6 slots
    threads = []
    for wid, (gpu, _) in enumerate(slots):
        t = threading.Thread(target=worker, args=(wid, gpu, q, results, lock))
        t.start(); threads.append(t)
    for t in threads:
        t.join()

    ok = sum(1 for _, s in results if s == "ok")
    skip = sum(1 for _, s in results if s == "skip")
    fail = [jid for jid, s in results if s.startswith("FAIL")]
    print(f"\n=== done: {ok} ok, {skip} skipped, {len(fail)} failed ===")
    for jid in fail:
        print(f"  FAILED: {jid}")

if __name__ == "__main__":
    main()
