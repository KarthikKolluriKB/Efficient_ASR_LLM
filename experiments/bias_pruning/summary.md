# Experiment 1 — Summary

## What was tested

Whether removing the top 2 encoder layers from Whisper-large-v2 (32 → 30
layers kept), a depth that prior work suggested preserves aggregate WER,
also preserves WER and CER equally across gender subgroups on CommonVoice
22 English. Both checkpoints come from paper 1 (seed=42); no training
was performed in this experiment.

## Descriptive table headline

CommonVoice 22 English test split: 16,390 valid utterances after the
dataset's text-length filter (16,396 in the upstream TSV). After joining
demographics via `(speaker_id[:16], normalised_sentence)`, 119 of 16,390
utterances (0.73 %) could not be paired with a CV22 demo row and were
labelled `gender=missing`.

| gender | n_utts | hours | analysable |
|---|---:|---:|:---:|
| male | 1,806 | 2.90 | yes |
| female | 491 | 0.81 | yes |
| other | 11 | 0.02 | no (<200 utts) |
| missing | 14,082 | 23.30 | yes (by count) but excluded from gap analysis |
| **all** | **16,390** | **27.02** | — |

Gender is unlabelled for **85.9 %** of the English test split. The
gender-stratified analysis runs on the 14.1 % of utterances that carry a
label (`male` + `female`).

## Aggregate WER / CER comparison

| condition | aggregate WER | 95 % CI | aggregate CER | 95 % CI |
|---|---:|:---:|---:|:---:|
| unpruned (32L) | 15.18 % | [14.77, 15.61] | 8.64 % | [8.33, 8.99] |
| pruned (30L)   | 16.68 % | [16.24, 17.15] | 9.65 % | [9.27, 10.05] |
| Δ | **+1.50 pts** | — | **+1.01 pts** | — |

**Reproduces paper 1's preservation claim: no.** The aggregate WER moves
by 1.50 absolute points, well outside the ±0.5 pt threshold typically
used to call WER "preserved." Whether this is because paper 1's
preservation result was on a different Whisper size (e.g. medium) or
because the 30L checkpoint here was trained under a different recipe is
not determined by this experiment.

## Per-gender WER and CER (seed 42)

| condition | gender | n | WER % | 95 % CI | CER % | 95 % CI |
|---|---|---:|---:|:---:|---:|:---:|
| unpruned | male | 1,806 | 16.59 | [15.63, 17.69] | 9.64 | [8.85, 10.63] |
| unpruned | female | 491 | 14.02 | [12.44, 15.71] | 7.93 | [6.86, 9.15] |
| pruned | male | 1,806 | 17.41 | [16.44, 18.54] | 10.09 | [9.40, 10.87] |
| pruned | female | 491 | 15.98 | [14.13, 17.77] | 9.16 | [7.95, 10.43] |

Cross-seed SD is undefined because only seed=42 exists; per-cell CIs
above are utterance-level bootstrap (1,000 resamples).

## Disparity metrics

| metric | unpruned | pruned | Δ |
|---|---:|---:|---:|
| WER gap (male − female) | 2.57 pts | 1.43 pts | **−1.14 pts** |
| WER ratio (worst / best) | 1.18 | 1.09 | −0.09 |
| WER worst-group (=male) | 16.59 % | 17.41 % | **+0.82 pts** |
| CER gap | 1.71 pts | 0.93 pts | −0.78 pts |
| CER worst-group (=male) | 9.64 % | 10.09 % | +0.45 pts |

The gap narrows in both metrics, but only because the better-performing
group (female) degrades faster than the worse-performing group (male).
Worst-group WER worsens by 0.82 pts; nobody got better.

## Statistical significance (paired bootstrap, same utterance set, 1,000 resamples)

| gender | n | WER Δ | WER 95 % CI | WER p (two-sided) | CER Δ | CER 95 % CI | CER p |
|---|---:|---:|:---:|:---:|---:|:---:|:---:|
| male | 1,806 | +0.82 pts | [+0.29, +1.34] | **0.001** | +0.45 pts | [−0.10, +0.89] | 0.075 |
| female | 491 | +1.96 pts | [+1.00, +2.95] | **<0.001** | +1.22 pts | [+0.60, +1.86] | **<0.001** |
| missing | 14,082 | +1.57 pts | [+1.18, +2.01] | **<0.001** | +1.08 pts | [+0.72, +1.46] | **<0.001** |

Both labelled gender cells show statistically significant WER
degradation. Male and female WER-delta CIs overlap in the range
[+1.00, +1.34], so the per-gender degradation rates are numerically
different (+0.82 vs +1.96 pts) but not statistically distinguishable
from each other at the 95 % level given the female cell size. CER tells
a sharper story for the female group (significant at p < 0.001) than
for the male group (p = 0.075).

## Interpretation

Pruning two top encoder layers from Whisper-large-v2 produces a
significant and broadly distributed WER cost on CV22 English: aggregate
WER rises by 1.5 pts and every labelled subgroup degrades significantly.
The point-estimate degradation falls disproportionately on the
better-performing group (female), and the male-female gap narrows
because female speakers lose performance faster than male speakers —
not because male speakers improve. Worst-group WER worsens by 0.82 pts.
The "preserved aggregate, hidden bias" pattern hypothesised at the
outset of this experiment is not what these data show: aggregate is not
preserved, and the per-group cost is not equal. However, the cell sizes
in CV22 English (n=491 for female) leave the differential per-group
cost (female Δ − male Δ = +1.14 pts) statistically inconclusive at the
95 % level; the data are consistent with both "pruning hurts female
more" and "pruning hurts everyone equally and we lack power to tell
them apart."

## Limitations

- **Content / accent / age confounds are not controlled.** The
  per-utterance reference texts are not matched across gender, so any
  WER difference could in principle be driven by sentence difficulty,
  accent distribution, or age skew that correlates with gender in CV22.
- **Single seed.** Only seed=42 exists for these checkpoints. Cross-seed
  variance is undefined and the paired bootstrap is the only available
  significance signal. A multi-seed sweep would tighten the differential
  estimate.
- **Single model size and single language.** Whisper-large-v2 + CV22-en
  only. Whether the pattern transfers to Whisper-medium (the size paper
  1 reported preservation on), to LoRA-tuned variants, or to other
  languages is unknown from this run.

## Next experiment

Sweep prune-depth from 32L through ~16L using the existing
`ablation_*L` checkpoints (no new training) and check whether the
female-vs-male degradation rate diverges further at deeper pruning — if
the gap-narrowing pattern continues at greater depths, the disparate-
impact story strengthens; if it reverses or stabilises, the current
finding is depth-specific noise.
