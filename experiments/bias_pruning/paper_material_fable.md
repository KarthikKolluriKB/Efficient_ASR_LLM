# Paper material — "Bias in Encoder Pruning of Speech-LLMs"

Writing-ready material for every section of the IMPACT-SPEECH submission
(deadline 2026-07-20). Each section: the content, the numbers, and notes on
emphasis/honesty. Companion docs: `results/ANALYSIS_MASTER.md` (auditable
numbers), `fable-ideas/` (figure specs).

Working titles (pick one register):
- *When "Free" Pruning Isn't Free: Compression Hides Demographic Harm in Speech-LLMs*
- *The Average Is the Bug: Encoder Pruning, Hidden Bias, and Scale in Speech-LLMs*
- *Pruned for Some: How WER-Preserving Compression Redistributes Error Across Speakers*

---

## 1. Research questions (detailed)

### RQ1 — Does encoder pruning damage subgroups unequally, and is that effect stable across model scale and language?

We prune the Whisper encoder of a SLAM-ASR system progressively (top-down) and
measure per-subgroup WER at every depth. The core question is whether
between-group disparities widen even at depths that look "safe" on aggregate
WER. Two deliberate extensions make this more than a single-operating-point
audit:

- **Across scale.** The full pruning sweep is repeated on three encoder sizes
  (Whisper small = 12 layers, medium = 24, large-v2 = 32), making encoder
  capacity an experimental variable. Hypothesis space: amplification could be
  scale-independent (a property of pruning), grow with scale (more capacity →
  more redistributable error), or shrink with scale (bigger models more robust).
- **Across language.** The analysis is repeated for English, Danish and Dutch
  systems (training data availability spans an order of magnitude), testing
  whether the disparity pattern is a property of the architecture or interacts
  with per-language resources.

Framing note: the deployed-model gap. Deployed speech systems are almost always
compressed versions of research models, so fairness numbers reported on a full
model systematically overstate the fairness of what ships. Compression is the
gap between what we evaluate and what we deploy; RQ1 asks whether bias hides in
that gap.

### RQ2 — Does parameter-efficient adaptation (LoRA) repair the damage uniformly?

The field's default answer to compression-induced quality loss is lightweight
adaptation. We apply LoRA to the (otherwise frozen) LLM on top of every pruned
encoder and re-measure per-subgroup WER at matched depths. The question is not
whether LoRA improves aggregate WER (it will) but whether the improvement is
distributed evenly — or preferentially benefits already-well-served groups,
leaving relative disparity unchanged or worse.

### RQ3 — Can harm be mitigated at the compression stage itself?

If RQ1 shows pruning creates hidden harm and RQ2 shows the standard repair does
not fix it, what intervention *at the compression/recovery stage* prevents the
harm — under the realistic constraint that **no demographic labels exist at
training time**? (Our training corpus collects no ethnicity labels; most
industrial pipelines are in the same position; the evaluation corpus must stay
zero-shot.) See §6.

---

## 2. Methodology (how we did it)

### System under study

SLAM-ASR architecture: a **frozen Whisper encoder** → a small **trainable
linear projector** (ConcatLinear) → a **frozen Qwen2.5-3B** LLM that decodes
text. The projector is the only trained component in the base condition; it is
trained independently for every pruning configuration, so each depth is a
separately trained, deployable system rather than a post-hoc ablation.

### Pruning protocol

- **Structured depth pruning, top-down:** remove the top *k* encoder layers,
  keep the bottom *(N − k)*. Depths swept in 2-layer steps for medium/large-v2
  and 1-layer steps for small, from unpruned to model collapse.
- After each prune, the projector is trained with the standard recipe
  (identical data, budget, seed) and the best checkpoint is selected by
  **aggregate dev WER** (`checkpoint_best_wer`) — i.e., exactly the mean-based
  pipeline the field uses. (This detail becomes load-bearing in §6.)
- **LoRA condition (RQ2):** low-rank adapters on the LLM, trained on top of each
  pruned encoder + projector, same data; evaluated at matched depths.

### Measurement

- Per-utterance inference over each evaluation corpus; WER/CER per demographic
  subgroup with **utterance-level bootstrap 95 % CIs** (1 000 resamples).
- Between-condition changes tested with a **paired bootstrap** on the same
  utterance set (two-sided p-values per group).
- Disparity is always reported **two ways**: absolute gap Δ = WER_worst −
  WER_best (pp) and relative gap ρ = WER_worst / WER_best. They diverge under
  heavy pruning (as all groups approach the WER ceiling, Δ keeps growing while
  ρ compresses), so either alone can mislead; reporting both forecloses
  cherry-picking.
- **Usable range:** claims are restricted to depths where the model remains a
  working ASR system (aggregate WER ≤ ~40 %). Beyond it, all groups saturate
  and neither metric is interpretable.
- **Analysability threshold** per subgroup cell (≥ 200 utterances / ≥ 30 min
  audio); cells below it are reported as unanalysable, not plotted.
- Groups for the headline contrast are **fixed** (Black vs Asian — consistently
  the highest- and lowest-error major groups), not re-picked per depth, to
  avoid reference-group cherry-picking. Robustness: the same monotone widening
  holds for Black-vs-White.

### Zero-shot evaluation discipline

All evaluation corpora are used **zero-shot** — no system component ever sees
them in training. Results therefore measure transfer to unseen data, and no
demographic label is ever available to any training stage.

---

## 3. Experimental setup

### Datasets

| corpus | role | axes | notes |
|---|---|---|---|
| Common Voice 22 (EN) | projector + LoRA training; eval | gender, age, accent | training corpus; test split speaker-disjoint |
| Common Voice (DA, NL) | cross-lingual training + eval | gender, age, accent | Danish/Dutch systems trained per-language; Dutch on a reduced subset [verify exact hours before camera-ready] |
| **Fair-Speech** (Meta) | eval, zero-shot | **ethnicity, SES**, gender, age, L1 | 26 417 utts; richest demographic schema; the headline corpus |
| **L2-ARCTIC** | eval, zero-shot | **L1 background** | small, read speech; per-group cells noisy → qualitative only |

CV22 English test set caveat: 85.9 % of utterances lack a gender label; gender
analyses run on the labeled 14.1 %.

### Bias dimensions

- **Ethnicity** (Fair-Speech; 7 analysable groups) — headline axis.
- **Socio-economic status** (Fair-Speech; affluent / medium / low) — secondary.
- **Accent** (CV22: US vs India/S-Asia vs England …; CV-NL: Belgian vs
  Netherlands Dutch) — supporting + cross-lingual.
- **L1 background** (L2-ARCTIC: Hindi, Korean, Vietnamese, …) — illustration.
- **Gender, age** — measured everywhere; direction inconsistent across corpora;
  explicitly NOT claimed.

### Implementation details

- Encoders: Whisper small (12 L), medium (24 L), large-v2 (32 L), frozen.
- LLM: Qwen2.5-3B, frozen (base) / LoRA-adapted (RQ2).
- Projector: ConcatLinear; the sole trained module per configuration (base).
- One seed (42) throughout — a workshop-scope limitation; the paired bootstrap
  on a shared utterance set is the significance instrument.
- Evaluation cost: ~3 h per Fair-Speech sweep point on one GPU; full study
  spans ~2 × 25 sweeps (base + LoRA × scales × corpora × languages).

---

## 4. Results (findings)

### F1 — A WER-preserving prune that degrades one group (the central case)

Large-v2, Fair-Speech, ethnicity, keep 32→30: aggregate WER **improves** by
0.44 pp; six of seven groups improve; **Black speakers regress** (+0.91 pp,
paired bootstrap significant). The utterance-weighted average inherits the
majority's direction and reports a net win — it is not merely imprecise but
**directionally misleading** about the harmed group.

| group | 32 L | 30 L | Δ |
|---|---:|---:|---:|
| Asian | 13.71 | 13.05 | −0.66 |
| … (Hispanic, White, MENA, Nat. Am.) | | | −0.5…−2.5 |
| **Black** | **27.21** | **28.12** | **+0.91** |
| **ALL** | 21.58 | 21.14 | **−0.44** |

### F2 — The harm grows monotonically across the usable range

Black−Asian absolute gap over five independently trained pruned large-v2
models: **13.5 → 15.1 → 18.1 → 20.6 → 24.5 pp** (keep 32→24). The spread opens
at both ends (Asian pulls further below the aggregate, Black further above):
pruning **redistributes** error toward the worst-off, it does not merely reveal
a fixed gap. Same direction for Black-vs-White (6.7 → 19.3 pp).

### F3 — Scale dependence: only the large model hides it (the novel finding)

| scale | free prune exists? | first-prune agg cost | ρ under light pruning |
|---|---|---:|---|
| small (12 L) | no | +4.7 pp | 2.03 → 2.03 → 1.70 (flat/shrinks) |
| medium (24 L) | no | +2.3 pp (24→22) | 2.16 → 2.01 → 1.94 (shrinks) |
| **large-v2 (32 L)** | **yes (keep 30)** | **−0.44 pp** | **1.98 → 2.15 (grows)** |

The pointed statement: *the capacity that lets a large model absorb pruning
without aggregate loss is the same capacity that lets it relocate that loss
onto a subgroup unnoticed.* Small models cannot hide the harm only because they
cannot hide the damage at all.

### F4 — RQ2: LoRA improves the average and worsens the ratio — at every depth

Large-v2, Fair-Speech, base → +LoRA at matched depth: aggregate WER falls 2–4 pp
at all 7 depths while ρ **rises at all 7**, e.g. keep-32 2.03→2.20, keep-30
2.15→2.35, keep-28 2.11→2.30. Medium replicates at all matched depths incl.
unpruned (keep-24: 2.16→**2.40**). Two sharpenings:
- **LoRA worsens ρ even with no pruning at all** → average-loss adaptation is
  itself disparity-amplifying; pruning is not required.
- **Axis-specific:** on CV22 accent ρ is ~flat under LoRA (1.43→1.44); on
  L2-ARCTIC LoRA mostly rescues collapse (noise). The amplification is a
  race-axis phenomenon.

### F5 — Cross-lingual: the effect requires headroom, and not all gaps scale alike

- **Dutch:** Belgian-vs-Netherlands accent gap widens with pruning at every
  scale by a similar amount (ρ ≈ 1.13–1.19 → 1.26–1.32) — **scale-invariant**,
  in contrast to race, and never hidden (aggregate rises from the first prune).
- **Danish (low-resource null):** aggregate WER ≥ 35 % even unpruned; gender
  ρ 1.00–1.02 throughout; accent unanalysable. No headroom → no differential
  harm → no hiding place. Consistent with the mechanism in F3.

### F6 — Decomposition: how much bias is pruning's fault?

At the free prune (large-v2 keep-30) the Black–Asian gap is 15.1 pp, of which
13.5 pp is inherited from the unpruned model and **~1.6 pp (~11 %) is
pruning-induced** — small, hidden, and opposite in sign to the aggregate change.
This scopes the mitigation target honestly: *worst-group-preserving
compression*, not "close the 2× inherited gap."

### Explicit non-claims

Hidden-under-the-average is Fair-Speech-only (race/SES); accent/L1 harm is real
but visible. Gender/age not claimed (inconsistent). L2-ARCTIC qualitative only.
Trend claimed only in the usable range. Single seed.

---

## 5. Proposed mechanism (for Discussion)

Recovery training after a prune minimizes a **mean** over a majority-dominated
distribution. A large encoder holds capacity the majority does not need; the
pruned top layers were load-bearing mainly for atypical ("tail") speech, so the
majority — and therefore the average — barely moves while the tail degrades. A
small encoder has no slack: every layer is shared, pruning visibly costs
aggregate WER, and nothing is hidden. The same logic one level up explains F4:
LoRA's average-loss objective allocates adaptation capacity where data density
is, recovering the majority most and the tail least. (Label as interpretation
consistent with the measurements, not directly measured. Connect to Hooker et
al.'s compression-identified exemplars — the vision analog of the concentration
effect.)

---

## 6. Future ideas — mitigation (RQ3 plan)

### The idea: De-Averaged Pruning

The pipeline's bias enters wherever an **average makes a decision**. Our own
findings identify three such points, so the mitigation is one principle applied
three times — all label-free (utterance-level tail, never demographics):

1. **Recovery loss:** mean CE → **batch-CVaR**: per batch, backprop only the
   worst α-fraction of per-utterance losses (α ≈ 0.2). The gradient can only
   come from the hardest utterances; the majority cannot outvote the tail.
   Single training run — identification and repair are fused per batch.
2. **Checkpoint selection:** best aggregate dev-WER (`checkpoint_best_wer` —
   the current pipeline literally selects for the confound) → best **tail**
   dev loss (CVaR@α).
3. **Acceptance test:** "prune is free if aggregate holds" → "free only if no
   utterance-decile regresses." Computable retroactively on existing sweeps at
   zero GPU cost; it correctly **rejects** large-v2 keep-30, the exact prune the
   aggregate test wrongly blesses.

**Success bar** (large-v2 keep-30, Fair-Speech, zero-shot): aggregate ≤ ~21.6 %
(prune stays free) AND Black WER ≤ 27.2 % (no regression vs unpruned) AND
ρ ≤ 1.98 (baseline). One knob (α), one probe allowed, no grids.

**Free go/no-go gate:** from existing per-utterance CSVs, test whether the
worst utterance-decile under pruning over-represents Black speakers. If damage
is diffuse rather than concentrated/skewed, tail-optimization won't transfer —
pivot before any GPU cost. (Either outcome is a paper figure.)

**Fallback:** damage-aware reweighting (DAR) — weight recovery sampling by each
utterance's loss increase vs the unpruned parent (JTT-style, one extra scoring
pass). Runs only if CVaR misses the bar; otherwise appears as an ablation.

**Positioning / novelty:** fairness-aware pruning exists for text-LLM stereotype
metrics (FASP) and label-dependent vision pruning (FairGRAPE); DRO-without-
demographics exists for training (Hashimoto et al.), group-DRO for ASR with
labels (CTC-DRO). To our knowledge no work makes the **compression recovery
stage** distributionally robust, in any modality — and in speech, where the
sensitive axis is unlabeled by construction, label-free is the only admissible
class of method. A null result is itself publishable: "label-free tail
optimization does not transfer across corpora → fairness-aware compression
requires demographic access."

**Deliberately out of scope (future work paragraph):** fairness-aware layer
*selection* (which layers to cut, not how many); cascade deployment (route
low-confidence/tail utterances to the unpruned parent — never harms the tail at
equal average FLOPs); multi-seed hardening; replication of the race axis on an
independent AAE corpus (CORAAL).

### Paper arc closing sentence

> The average hides the harm (RQ1); the average-trained repair amplifies it
> (RQ2); removing the average from the compression pipeline — loss, checkpoint,
> acceptance test — is the repair (RQ3).

---

## 7. Limitations (draft register)

1. Single seed (42); paired bootstrap is the significance signal.
2. Race/SES amplification demonstrated on one corpus (Fair-Speech); CORAAL
   replication left to future work.
3. "Hidden under the average" claimed only where measured (Fair-Speech ×
   large-v2); accent/L1 harm is visible, not hidden.
4. Content/accent/age confounds within groups uncontrolled beyond corpus design.
5. Cross-lingual LoRA (DA/NL) is on the accent/gender axes only — CV collects no
   race/SES — and Danish is a near-null low-resource case; all three scales are
   covered (large-v2 adapters do exist).
6. Mechanism (§5) is interpretation, consistent with but not proven by the
   sweeps.
