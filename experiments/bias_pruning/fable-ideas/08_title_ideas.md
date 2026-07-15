# Title ideas — ranked, with reasoning

## What an apt title must do for THIS paper

1. **Name the precise claim**, not the vibe. The finding is not "pruning is
   biased" — it is *"a prune that improves the average worsens one group, and
   the average conceals it."* Titles that only say "bias in compression" undersell
   the directional-misleading result.
2. **Carry the searchable keywords** somewhere (title or subtitle): pruning /
   compression, speech-LLM (or ASR), demographic / fairness / disparities.
3. **Match the paper's centre of gravity** — diagnosis (RQ1+RQ2). RQ3 will be
   the smallest section; a mitigation-flavored title overpromises.
4. **Survive the data.** (Resolved: the medium keep-22 point landed at ρ = 2.01,
   confirming medium does not amplify — the scale claim stands on complete
   curves.) Still prefer titles that don't hard-code a single data point.
5. Venue fit: IMPACT-SPEECH is a fairness workshop — a title in the field's own
   fairness idiom lands better than a compression-engineering idiom.

## ★ Top recommendation

> **Whose WER Is Preserved? Compression Hides Demographic Harm in Speech-LLMs**

Why it wins:
- "WER-preserving pruning" is the literal technical term the paper interrogates;
  the question turns the field's own vocabulary into the critique. Readers who
  know the term see the point instantly; readers who don't still parse it.
- Question + declarative-answer subtitle: the title asks, the subtitle answers,
  and the subtitle alone is fully searchable (compression, demographic,
  speech-LLMs).
- Doesn't hard-code scale or a specific group → robust to pending results and
  accurate for the multi-axis scope (race, SES, accent, L1).
- Fairness-native register ("whose X" is an established fairness-title move)
  without sacrificing precision.

## Runner-up (safest strong choice)

> **When "Free" Pruning Isn't Free: Compression Hides Demographic Harm in Speech-LLMs**

Equally precise, more compression-idiom than fairness-idiom. Pick this if a
question-form title feels risky to any co-author. ("Free prune" is instantly
legible to the efficiency community — good if we also want that audience.)

## Precision alternative (mechanism in the title)

> **Average Gains, Hidden Losses: Encoder Pruning Amplifies Demographic Disparities in Speech-LLMs**

The four-word main clause IS the finding (aggregate improves / a group loses,
concealed). Slightly duller than the top two but the most literally accurate;
zero rhetorical risk.

## Full ranked shortlist

| # | Title | Register | Risk/note |
|---|---|---|---|
| 1 | Whose WER Is Preserved? Compression Hides Demographic Harm in Speech-LLMs | fairness, sharp | none significant |
| 2 | When "Free" Pruning Isn't Free: Compression Hides Demographic Harm in Speech-LLMs | compression, sharp | quote marks in title (fine) |
| 3 | Average Gains, Hidden Losses: Encoder Pruning Amplifies Demographic Disparities in Speech-LLMs | neutral, precise | slightly long |
| 4 | Hidden by the Average, Worsened by the Fix: Pruning and LoRA Bias in Speech-LLMs | two-punch (RQ1+RQ2) | best if LoRA result is co-headline |
| 5 | Better on Average, Worse for Black Speakers: Hidden Harm in Pruned Speech-LLMs | maximally concrete | naming one group narrows the multi-axis scope; strong PNAS-style precedent though |
| 6 | Big Enough to Hide It: Scale-Dependent Bias under Speech-LLM Compression | scale-forward | scale claim now confirmed on complete curves |
| 7 | Compression Bias in Speech-LLMs: Scale-Dependent Amplification of Racial Disparities under Encoder Pruning | fully descriptive | conservative venue insurance |
| 8 | Is That Prune Really Free? Auditing Speech-LLM Compression Across Demographics, Scale, and Language | survey-flavored | undersells the finding as an "audit" |

## Rejected patterns (and why)

- **"The Average Is the Bug/Alibi" style** — great talk-slide, but as a title it
  promises the De-Averaged-Pruning mitigation paper; RQ3 is preliminary here.
  Save it for the follow-up full paper.
- **Anything with "LoRA" as the lead** — RQ2 is a co-finding, not the frame.
- **"Fairness-aware X" phrasing** — signals a methods/mitigation paper.
- **Pure pun titles without a load-bearing subtitle** — workshop proceedings are
  found via search; the subtitle must do the indexing work.

## Consistency checks before final choice

- Title ↔ abstract first sentence must not repeat verbatim; the abstract should
  open with the finding, the title with the question/frame.
- If #5/#6-style specificity is chosen, re-verify against the frozen results
  (2026-07-15) — especially the medium keep-22 point for any scale wording.
- Whatever is chosen, use "WER-preserving" consistently in §1 so the title's
  hook is anchored in the text within the first page.
