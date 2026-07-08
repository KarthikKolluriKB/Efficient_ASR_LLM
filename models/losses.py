"""
De-averaged (tail-focused) sequence losses — RQ3 "De-Averaged Pruning".

The compression pipeline's bias enters wherever an *average* makes a decision.
This module replaces the first of those averages: the recovery loss. Instead of
backpropagating the batch-mean cross-entropy (which lets the majority of
utterances outvote the hardest ones), batch-CVaR backprops only the mean of the
worst alpha-fraction of per-utterance losses. Label-free: the tail is defined
at the utterance level, never by demographics.

Public API:
    per_utterance_ce(logits, labels, ...)  -> (utt_losses, utt_valid)
    cvar(losses, alpha, valid=None)        -> scalar (mean of worst ceil(alpha*n))
    sequence_cvar_ce(logits, labels, alpha, ...) -> (scalar loss, detached utt losses)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def per_utterance_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
    label_smoothing: float = 0.0,
    shift: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-utterance mean cross-entropy.

    Args:
        logits: (B, T, V) unshifted logits.
        labels: (B, T) target ids with `ignore_index` on non-answer positions.
        shift: apply the causal-LM shift (predict token t+1 from position t),
            matching the standard HF `labels=...` loss.

    Returns:
        utt_losses: (B,) mean CE over each utterance's valid tokens
            (0.0 for utterances with no valid tokens).
        utt_valid: (B,) bool mask, True where the utterance has >= 1 valid token.
    """
    if shift:
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

    bsz, seq_len, vocab = logits.shape
    token_losses = F.cross_entropy(
        logits.view(-1, vocab),
        labels.view(-1),
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction="none",
    ).view(bsz, seq_len)

    valid_tokens = (labels != ignore_index)
    n_valid = valid_tokens.sum(dim=1)
    # ignored positions already contribute 0 loss, so a plain row-sum is safe
    utt_losses = token_losses.sum(dim=1) / n_valid.clamp(min=1)
    return utt_losses, n_valid > 0


def cvar(
    losses: torch.Tensor,
    alpha: float,
    valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean of the worst ceil(alpha * n) losses (batch-CVaR at level alpha).

    alpha=1.0 reduces to the plain mean; alpha=0.2 means the gradient can only
    come from the hardest 20% of utterances in the batch.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    if valid is not None:
        losses = losses[valid]
    n = losses.numel()
    if n == 0:
        # keep the graph alive with a zero that carries no gradient signal
        return losses.sum()
    k = max(1, math.ceil(alpha * n))
    if k >= n:
        return losses.mean()
    worst = torch.topk(losses, k, largest=True).values
    return worst.mean()


def sequence_cvar_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    ignore_index: int = IGNORE_INDEX,
    label_smoothing: float = 0.0,
    shift: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch-CVaR sequence cross-entropy.

    Returns (scalar loss for backprop, detached per-utterance losses). The
    per-utterance losses are also useful at validation time for tail-based
    checkpoint selection (de-averaging point 2).
    """
    utt_losses, utt_valid = per_utterance_ce(
        logits, labels,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        shift=shift,
    )
    loss = cvar(utt_losses, alpha, valid=utt_valid)
    return loss, utt_losses.detach()
