"""
Unit tests for the batch-CVaR loss (models/losses.py) — RQ3 demo, part 1.

Run from repo root (no GPU needed):
    python experiments/bias_pruning/rq3_mitigation/test_cvar_loss.py
or with pytest:
    pytest experiments/bias_pruning/rq3_mitigation/test_cvar_loss.py -q
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from models.losses import per_utterance_ce, cvar, sequence_cvar_ce  # noqa: E402

torch.manual_seed(0)


def _manual_per_utt(logits, labels):
    """Reference implementation: loop over utterances, shift, mask, mean."""
    out = []
    for i in range(logits.shape[0]):
        lg = logits[i, :-1, :]
        lb = labels[i, 1:]
        mask = lb != -100
        if mask.sum() == 0:
            out.append(torch.tensor(0.0))
            continue
        out.append(F.cross_entropy(lg[mask], lb[mask]))
    return torch.stack(out)


def test_per_utterance_ce_matches_manual():
    B, T, V = 6, 12, 50
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    # variable-length answers: mask out different prefixes per row
    for i in range(B):
        labels[i, : 2 + i] = -100

    utt_losses, valid = per_utterance_ce(logits, labels)
    manual = _manual_per_utt(logits, labels)
    assert torch.allclose(utt_losses, manual, atol=1e-5), (utt_losses, manual)
    assert valid.all()
    print("PASS: per-utterance CE matches manual per-row computation")


def test_fully_masked_row_is_excluded():
    B, T, V = 4, 8, 20
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    labels[2, :] = -100  # utterance with no valid tokens

    utt_losses, valid = per_utterance_ce(logits, labels)
    assert not torch.isnan(utt_losses).any(), "NaN leaked from empty row"
    assert bool(valid[2]) is False and valid.sum() == 3

    loss = cvar(utt_losses, alpha=1.0, valid=valid)
    expected = utt_losses[valid].mean()
    assert torch.allclose(loss, expected, atol=1e-6)
    print("PASS: fully-masked utterance excluded, no NaN")


def test_cvar_alpha1_equals_mean_and_worst_k_selection():
    losses = torch.tensor([0.1, 5.0, 0.2, 3.0, 0.3, 0.15, 0.25, 0.05, 0.4, 0.35])
    # alpha = 1.0 -> plain mean
    assert torch.allclose(cvar(losses, 1.0), losses.mean())
    # alpha = 0.2, n = 10 -> ceil(2) = worst 2: (5.0 + 3.0) / 2
    assert torch.allclose(cvar(losses, 0.2), torch.tensor(4.0))
    # alpha = 0.25, n = 10 -> ceil(2.5) = 3 worst: (5.0 + 3.0 + 0.4) / 3
    assert torch.allclose(cvar(losses, 0.25), torch.tensor((5.0 + 3.0 + 0.4) / 3))
    # tiny batch: k never drops below 1
    assert torch.allclose(cvar(losses[:2], 0.01), torch.tensor(5.0))
    print("PASS: CVaR selection math (alpha=1 -> mean; worst-k exact)")


def test_gradient_flows_only_through_worst_alpha():
    B, T, V = 10, 6, 30
    logits = torch.randn(B, T, V, requires_grad=True)
    labels = torch.randint(0, V, (B, T))
    labels[:, 0] = -100

    # make rows 3 and 7 catastrophically bad so they are the worst 20%
    with torch.no_grad():
        for bad in (3, 7):
            wrong = (labels[bad, 1:] + 1) % V
            logits[bad, :-1, :] = -10.0
            logits[bad, :-1].scatter_(1, wrong.unsqueeze(1), 10.0)

    loss, utt_losses = sequence_cvar_ce(logits, labels, alpha=0.2)
    loss.backward()

    grad_norms = logits.grad.flatten(1).norm(dim=1)
    assert (grad_norms[[3, 7]] > 0).all(), "worst utterances got no gradient"
    others = [i for i in range(B) if i not in (3, 7)]
    assert torch.allclose(grad_norms[others], torch.zeros(len(others))), (
        "gradient leaked into utterances outside the worst alpha-fraction"
    )
    assert set(torch.topk(utt_losses, 2).indices.tolist()) == {3, 7}
    print("PASS: gradient flows ONLY through the worst alpha-fraction of utterances")


def test_shift_semantics_perfect_prediction():
    """Logits at position t predict label t+1 -> loss ~ 0 (matches HF shift)."""
    B, T, V = 3, 10, 25
    labels = torch.randint(0, V, (B, T))
    labels[:, :3] = -100
    logits = torch.full((B, T, V), -10.0)
    for i in range(B):
        for t in range(T - 1):
            if labels[i, t + 1] != -100:
                logits[i, t, labels[i, t + 1]] = 10.0

    loss, _ = sequence_cvar_ce(logits, labels, alpha=0.5)
    assert loss.item() < 1e-4, f"expected ~0 loss, got {loss.item()}"
    print("PASS: causal shift matches HF labels convention (perfect pred -> ~0 loss)")


def test_cvar_upweights_tail_vs_mean():
    """Sanity: CVaR loss >= mean loss, strictly greater when tail exists."""
    B, T, V = 16, 8, 40
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    utt_losses, valid = per_utterance_ce(logits, labels)
    assert cvar(utt_losses, 0.2, valid) >= cvar(utt_losses, 1.0, valid)
    print("PASS: CVaR(alpha<1) >= mean (tail is upweighted)")


if __name__ == "__main__":
    test_per_utterance_ce_matches_manual()
    test_fully_masked_row_is_excluded()
    test_cvar_alpha1_equals_mean_and_worst_k_selection()
    test_gradient_flows_only_through_worst_alpha()
    test_shift_semantics_perfect_prediction()
    test_cvar_upweights_tail_vs_mean()
    print("\nAll CVaR loss unit tests passed.")
