# Project status — Experiment 1 and beyond

Snapshot as of 2026-05-31. Companion docs:
- [README.md](README.md) — scaffold and layer-count vocabulary
- [SERVER_RUN.md](SERVER_RUN.md) — copy-pasteable server runbook
- [summary.md](summary.md) — full Experiment 1 results and interpretation
- [data_external/README.md](data_external/README.md) — external-dataset download guide

---

## What this project is

A follow-up study to the SLAM-ASR encoder-pruning paper. Paper 1 showed
that pruning 1–2 encoder layers from Whisper preserves aggregate WER.
This project asks whether that "safe" depth also preserves performance
**equally across demographic subgroups**, or whether bias hides under
the aggregate number.

The first experiment uses **Whisper-large-v2** + ConcatLinear projector
+ Qwen2.5-3B LLM, evaluated on **CommonVoice 22 English**, with a single
training seed (42). No new training — paper 1's checkpoints are reused.

Layer-count vocabulary (important — the repo has two opposite conventions):

| Spec wording in this project | Repo config name | Layers kept | Layers pruned |
|---|---|---:|---:|
| `unpruned` baseline | `whisper_largev2/eval/baseline.yaml` | 32 | 0 |
| `2L pruned` (target condition) | `whisper_largev2/eval/ablation_30L.yaml` | 30 | 2 |

The repo also has `ablation_2L`/`ablation_4L`/etc. — those *keep* that
many layers and are NOT what this project means by "2L pruned."

---

## What's done

### Scaffold

- [scripts/build_descriptive_table.py](scripts/build_descriptive_table.py) — CV22 demographic descriptive table from `test.tsv`.
- [scripts/bootstrap_ci.py](scripts/bootstrap_ci.py) — utterance-level bootstrap WER/CER with 95 % CI, paired-bootstrap test for between-condition differences. Self-tested against jiwer (exact match).
- [scripts/evaluate_subgroup_wer.py](scripts/evaluate_subgroup_wer.py) — runs inference (reusing `eval.py`'s model loader unchanged), joins CV22 demographics on `(speaker_id[:16], normalised_sentence)`, writes per-utterance CSV + per-gender summary with bootstrap CIs.
- [scripts/compare_conditions.py](scripts/compare_conditions.py) — aggregates per-seed summaries, computes disparity metrics (gap, ratio, worst-group), runs paired bootstrap per gender, writes the final comparison table + twin WER/CER plot.

### Experiment 1 run

Both conditions evaluated on the server (1 × GPU each, ~22 min):

```
Unpruned (32L):  aggregate WER 15.18 %  | CER 8.64 %
Pruned   (30L):  aggregate WER 16.68 %  | CER 9.65 %
Δ aggregate:                +1.50 pts   |     +1.01 pts
```

Per-gender paired bootstrap on the same utterance set:

| gender | n | WER Δ | WER 95 % CI | p (two-sided) | CER Δ | CER p |
|---|---:|---:|:---:|:---:|---:|:---:|
| male | 1,806 | +0.82 | [+0.29, +1.34] | **0.001** | +0.45 | 0.075 |
| female | 491 | +1.96 | [+1.00, +2.95] | **<0.001** | +1.22 | **<0.001** |
| missing | 14,082 | +1.57 | [+1.18, +2.01] | **<0.001** | +1.08 | **<0.001** |

### Three findings from the run

1. **Aggregate WER is not preserved at this depth.** Paper 1's
   preservation claim doesn't replicate for whisper-large-v2 → 30L kept
   in this configuration. Either paper 1 reported on whisper-medium
   specifically, or the 30L checkpoint here uses a different training
   recipe (LoRA?). Worth resolving before citing the experiment as a
   replication failure.
2. **Female numerical degradation is ~2.4× male's** (+1.96 vs +0.82 pts).
   The two delta CIs overlap in [+1.00, +1.34], so "female degraded more
   than male" is true at the point estimate but not statistically
   distinguishable from "equal degradation" at the 95 % level. Power-
   limited by n=491 for the female cell — CV22 English has 85.9 %
   missing gender labels.
3. **CER tells a sharper story for female-only effects.** Male CER
   degradation is not significant (p=0.075); female and missing both
   significantly worse (p<0.001).

### Disparity metrics

| metric | unpruned | pruned | Δ |
|---|---:|---:|---:|
| WER gap (male − female) | 2.57 pts | 1.43 pts | −1.14 pts |
| WER worst-group (= male) | 16.59 % | 17.41 % | +0.82 pts |
| CER gap | 1.71 pts | 0.93 pts | −0.78 pts |

The gap narrows under pruning — but only because the better-performing
group (female) degrades faster than the worse-performing group (male).
Worst-group WER worsens. Nobody gets better.

### Caveats baked into Experiment 1

- **Single seed.** Cross-seed SD is undefined; paired bootstrap is the only significance signal.
- **Content / accent / age confounds uncontrolled.** A WER differential could be content-driven (CV22 female speakers read different sentences on average) rather than acoustic.
- **85.9 % of CV22 English test has no gender label** — gender analysis runs on 14.1 % of utterances.
- **Whisper-large-v2 only.** Paper 1's preservation claim may be size-specific.

---

## External datasets prepared for follow-ups

Five candidates were considered; four are downloaded and verified, one
was de-prioritised after its CDN turned out to be dead.

| dataset | status | size | gender | age | accent | L1 | SES | ethnicity | notes |
|---|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| [EdAcc](data_external/README.md#edacc) | ✅ on server | 7.0 GB | ✅ | — | ✅ (25) | ✅ (29) | — | — | 40 h dyadic conversations; 32 kHz |
| [L2-ARCTIC](data_external/README.md#l2-arctic) | ✅ on server | 451 MB | ✅ (m/f) | — | — | ✅ (6) | — | — | 16 kHz already; small + read speech |
| [Fair-Speech](data_external/README.md#fair-speech) | ✅ on server | 10.6 GB | ✅ | ✅ | — | ✅ | ✅ | ✅ | Meta ASR Fairness; richest demographic schema; **license restricts redistribution** |
| [CORAAL](data_external/README.md#coraal) | ✅ on server (8 components, ~78 GB); metadata fetch pending | ~78 GB | ✅ | ✅ | — | — | ✅ (SEC + Education) | implicit (AAL) | long-form interviews — **chunking step required** before ASR eval |
| [Artie Bias](data_external/README.md#artie-bias-corpus) | ❌ de-prioritised | TSV only | (would be) | (would be) | (would be) | — | — | — | Audio CDN dead since May 2024; manual reconstruction from CV2 not worth the effort |

### Key finding about cross-dataset coverage

**Gender + age + SES** are testable on at least two independent
datasets (Fair-Speech ∧ CORAAL). That's the strongest replication
structure available — and it's what would turn the current
"suggestive" Experiment 1 result into a defensible paper claim.

I earlier told the user "no speech dataset has SES labels." That was
wrong. Fair-Speech has a `socioeconomic_bkgd` column (sample value
`'Low'`); CORAAL has `CORAAL.SEC.Group` and `Education`. Both are
self-reported and bucketed.

### What still has to happen per external dataset before evaluation

For each of the four downloaded datasets we need:

1. **An HF-format builder** — turns raw audio + metadata into the
   standard SLAM-ASR schema (`audio_array`, `sampling_rate`,
   `transcription`, `raw_transcription`, `duration`, `speaker_id`)
   plus a column per demographic axis. ~1 h per dataset.
2. **A small generalisation of `evaluate_subgroup_wer.py`** to read
   demographics directly from the dataset row instead of joining
   from a separate TSV (which is the CV22-only special case).

CORAAL has an extra step: **chunk the 30–60 min interview WAVs into
≤30 s segments** using the transcript timestamps. Budget 3–5 days for
that work alone before CORAAL is usable.

---

## Repo / git workflow

- Pre-existing changes in the working tree (`pyproject.toml`,
  `uv.lock`, deleted `notebooks/data.ipynb`, `.vscode/`) are
  intentionally untouched.
- `data/` is gitignored. The Fair-Speech license additionally forbids
  redistribution of audio / metadata / derived transcripts — the
  whole `data/fairspeech/` tree stays on the server.
- `experiments/bias_pruning/results/` is gitignored (server runs
  produce these files at the same paths and used to conflict on
  `git pull`). They're trivially regeneratable from the checkpoint +
  test split.
- Tracked outputs: scripts, configs, README, SERVER_RUN, summary.md,
  data_external downloaders + README, this STATUS.md.

Recent commits on `main` (most recent first):

```
db09b4d download_coraal: also fetch per-component metadata .txt files
a2e3c2c verify_datasets: filter macOS resource-fork files and surface CORAAL metadata
5e1b6e3 Fix verify_datasets: skip audio decoding so torchcodec isn't required
c1c282d Add verify_datasets.py for the four external fairness datasets
467ffbc Mark Artie audio CDN as dead, revert downloader to GitHub-only
3442cd2 Switch Artie downloader to the official audio-bundled tarball  (later reverted)
f2cbe44 Untrack everything under experiments/bias_pruning/results/
a47b5a8 Stop tracking per-seed and per-utterance CSVs to avoid server merge conflicts
18f4d2b Wire up Fair-Speech manual download flow and block redistribution
a71caa5 Add Experiment 1 results and external-dataset downloaders
a08fa3a Update bias_pruning docs to whisper-large-v2 (32L vs 30L)
d8d658e Add bias_pruning experiment scaffold (Experiment 1)
```

---

## What's next, in priority order

### Layer 1 — strengthen Experiment 1 without new training

1. **Depth sweep** with existing `ablation_*L` checkpoints
   (`ablation_28L`, `ablation_24L`, `ablation_14L`, ...). Run the same
   `evaluate_subgroup_wer.py` for each depth. If the female-vs-male
   degradation rate widens monotonically with depth, the disparate-
   impact story strengthens substantially. Currently the script doesn't
   have a sweep driver — would need a small wrapper. ~3 GPU h per depth.
2. **Per-speaker drill-down on the female cell.** Is the female effect
   driven by a few outlier speakers or distributed evenly? Cheap script
   work, ~30 min, no GPU.
3. **Content-matched subset.** Filter to sentences spoken by both male
   and female speakers; rerun the paired bootstrap on that subset.
   Removes the content confound in one step. ~1 h script work.
4. **Age + accent aggregation** from existing per-utterance CSVs. Same
   data we already have; new aggregator function. ~30 min.

### Layer 2 — replicate on external datasets

5. **Fair-Speech HF builder + adapter**. Highest value: bigger gender
   cell, plus age, SES, ethnicity, native flag in one dataset. ~1 day.
6. **L2-ARCTIC adapter**. Cheapest second dataset. ~half day.
7. **EdAcc adapter**. Adds accent breadth. ~half day after Fair-Speech.
8. **CORAAL adapter** with chunking. Biggest engineering lift. 3–5 days.

### Layer 3 — paper-worthy infrastructure (only if targeting publication)

9. **Multi-seed training of the 30L (or whatever depth) condition.** A
   single-seed disparate-impact finding is routinely rejected by
   reviewers; cross-seed SD makes the result publishable.
10. **Whisper-medium replication.** Directly settles the "is the +1.5 pt
    aggregate degradation model-size-specific?" question that
    Experiment 1 leaves open.

---

## Open items requiring user action

- Server: run the CORAAL metadata fetch (one-line Python invocation in
  `download_coraal.py`'s `download_metadata()`) so the metadata adapter
  can be written.
- Decide whether to commit further (depth-sweep results, follow-up
  scripts) and whether to push them to `main` directly or via a feature
  branch. So far direct-to-main has been the chosen pattern.
- Resolve the paper-1 epistemic question: was the preservation claim on
  whisper-medium, whisper-large-v2, or both? That changes how the
  +1.5 pt aggregate Δ is framed in any write-up.
