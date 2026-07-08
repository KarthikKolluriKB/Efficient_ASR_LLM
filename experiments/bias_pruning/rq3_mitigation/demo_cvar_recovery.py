"""
Synthetic mechanism demo for RQ3 "De-Averaged Pruning" — CPU, ~1 min.

Reproduces the paper's causal story end-to-end on a controlled toy problem,
using the EXACT loss code the real pipeline uses (models/losses.py):

  1. A model serves a majority group (90%) and a minority group (10%) whose
     mapping conflicts with the majority's and is disambiguated by a group
     cue (think: accent/dialect marker) — serving the minority needs
     dedicated, cue-gated capacity.
  2. Compression damages the minority circuitry disproportionately while the
     aggregate stays tolerable — the damage profile RQ1 measured on real
     large-v2 (aggregate "WER-preserving" prune, race/SES disparity
     amplified). Here: prune to KEEP units by majority-calibrated average
     importance + degrade the cue circuitry.
  3. Recovery with mean CE under a fixed budget pours gradient into the
     majority (90% of the loss mass) -> aggregate recovers, minority lags.
     (RQ2/LoRA analog)
  4. Recovery with batch-CVaR (backprop only the worst alpha-fraction of
     per-utterance losses; label-free) improves the minority WITHOUT
     demographic labels.  (RQ3 mitigation)
  5. Checkpoint selection: best-aggregate vs best-tail val loss.
     (de-averaging point 2)

Expected outcome (see verdict): stages 1-3 reproduce robustly; the CVaR
mitigation (stage 4) is direction-consistent but MODEST in this toy, because
cross-entropy magnitude already up-weights badly-served samples under plain
mean training. CVaR's leverage grows with how strongly damage concentrates in
the loss tail — which is exactly what the plan's free go/no-go gate checks on
real per-utterance data (acceptance_test_deciles.py) before spending GPU.

Run from repo root:
    python experiments/bias_pruning/rq3_mitigation/demo_cvar_recovery.py
"""

import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from models.losses import sequence_cvar_ce, cvar  # noqa: E402  (the real pipeline loss)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
D_MAJ_FEAT = 10       # features driving the majority's (easy, linear) mapping
D_MIN_FEAT = 4        # features driving the minority's (harder, quadratic) mapping
D = D_MAJ_FEAT + D_MIN_FEAT
N_CLASSES = 5
MINORITY_FRAC = 0.02  # worst-served subgroup share. Small on purpose: mean
                      # training neglects a tail only when its loss MASS is
                      # negligible — 2% of samples cannot outvote the other 98%.
HIDDEN = 64
KEEP = 24             # hidden units kept after pruning
CUE_DAMAGE = 0.25     # compression degrades the minority's circuitry
                      # (the tail-concentrated damage profile observed in RQ1)
RECOVERY_RANK = 8     # LoRA-style recovery: few trainable directions,
                      # so the loss decides WHO gets repaired first (RQ5 analog)
N_TRAIN, N_VAL, N_TEST = 40_000, 8_000, 40_000
PRETRAIN_STEPS = 12_000
RECOVERY_STEPS = 4_000  # fixed recovery budget — same for both arms
BATCH = 512
LR_PRETRAIN, LR_RECOVER = 1e-3, 1e-3
CVAR_ALPHA = 0.05     # tail fraction. The plan's alpha~0.2/0.1 targets a 10-15%
                      # minority (Fair-Speech); this toy's minority is 2%, so the
                      # matched tail fraction is ~2-3x the minority share.
SEEDS = [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Data: each group relies on its own feature block (group is identifiable
# from the input, like accent in audio — no demographic label needed). The
# minority's mapping is HARDER to represent (quadratic vs linear teacher),
# so under tight capacity the two groups genuinely compete for hidden units.
# ---------------------------------------------------------------------------
def make_data(n, teacher_a, teacher_b, rng):
    x = torch.from_numpy(rng.standard_normal((n, D))).float()
    group = torch.from_numpy((rng.random(n) < MINORITY_FRAC).astype(np.int64))
    x[group == 0, D_MAJ_FEAT:] = 0.0   # majority never activates minority dims
    x[group == 1, :D_MAJ_FEAT] = 0.0   # minority relies entirely on its own dims
    scores_maj = x[:, :D_MAJ_FEAT] @ teacher_a.T                    # easy linear
    scores_min = (x[:, D_MAJ_FEAT:] ** 2 - 1.0) @ teacher_b.T       # hard quadratic
    y = torch.where(group == 0, scores_maj.argmax(dim=1), scores_min.argmax(dim=1))
    return x, y, group


class MLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.fc1 = nn.Linear(D, hidden)
        self.fc2 = nn.Linear(hidden, N_CLASSES)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class LoRARecovery(nn.Module):
    """LoRA-style recovery of a pruned model: the backbone (fc1) is frozen and
    only a rank-RECOVERY_RANK update + the readout are trainable — the same
    capacity-constrained recovery the real pipeline uses (projector/LoRA).
    With only a few trainable directions, the LOSS decides who gets repaired."""

    def __init__(self, base, rank=RECOVERY_RANK):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        k, d = base.fc1.weight.shape
        self.lora_a = nn.Parameter(torch.zeros(k, rank))
        self.lora_b = nn.Parameter(torch.randn(rank, d) * 0.05)
        self.fc2 = nn.Linear(k, N_CLASSES)
        self.fc2.load_state_dict(base.fc2.state_dict())

    def forward(self, x):
        w1 = self.base.fc1.weight + self.lora_a @ self.lora_b
        h = F.relu(x @ w1.T + self.base.fc1.bias)
        return self.fc2(h)


def as_sequence(logits, y):
    """Wrap classification logits as a length-2 sequence so we exercise the
    EXACT sequence_cvar_ce path used by models/model.py (position 0 predicts
    the label at position 1 after the causal shift)."""
    b = logits.shape[0]
    seq_logits = torch.stack([logits, torch.zeros_like(logits)], dim=1)  # (B,2,C)
    seq_labels = torch.stack(
        [torch.full_like(y, -100), y], dim=1                              # (B,2)
    )
    return seq_logits, seq_labels


def group_err(model, x, y, group):
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
        err = (pred != y).float()
    overall = err.mean().item()
    maj = err[group == 0].mean().item()
    mino = err[group == 1].mean().item()
    return overall, maj, mino


def val_losses(model, x, y):
    """Per-sample CE on the val set -> (aggregate mean, tail CVaR@alpha)."""
    with torch.no_grad():
        logits = model(x)
        losses = F.cross_entropy(logits, y, reduction="none")
    return losses.mean().item(), cvar(losses, CVAR_ALPHA).item()


def train_steps(model, x, y, steps, lr, alpha, rng, x_val=None, y_val=None):
    """Fixed-budget training. alpha=1.0 -> plain mean CE; alpha<1 -> batch-CVaR.
    Tracks best-aggregate and best-tail checkpoints on the val set."""
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    n = x.shape[0]
    best_agg = (float("inf"), None)
    best_tail = (float("inf"), None)
    for step in range(steps):
        idx = torch.from_numpy(rng.integers(0, n, size=BATCH))
        logits = model(x[idx])
        seq_logits, seq_labels = as_sequence(logits, y[idx])
        loss, _ = sequence_cvar_ce(seq_logits, seq_labels, alpha=alpha)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if x_val is not None and (step + 1) % 20 == 0:
            agg, tail = val_losses(model, x_val, y_val)
            if agg < best_agg[0]:
                best_agg = (agg, copy.deepcopy(model.state_dict()))
            if tail < best_tail[0]:
                best_tail = (tail, copy.deepcopy(model.state_dict()))
    return best_agg[1], best_tail[1]


def prune_by_average_importance(model, x, group):
    """Rank hidden units by mean |activation| over a majority-dominated
    calibration sample — the pipeline's 'average makes the decision' step —
    and keep the top KEEP units. Minority-gated units are nearly silent on
    majority traffic, so the averaged importance ranks them at the bottom."""
    with torch.no_grad():
        calib = x[group == 0]                      # what 'average traffic' looks like
        act = F.relu(model.fc1(calib))             # (N, HIDDEN)
        # standard magnitude importance: average contribution to the output
        importance = act.abs().mean(dim=0) * model.fc2.weight.norm(dim=0)
        keep_idx = torch.topk(importance, KEEP).indices.sort().values

    pruned = MLP(KEEP)
    with torch.no_grad():
        pruned.fc1.weight.copy_(model.fc1.weight[keep_idx])
        pruned.fc1.bias.copy_(model.fc1.bias[keep_idx])
        pruned.fc2.weight.copy_(model.fc2.weight[:, keep_idx])
        pruned.fc2.bias.copy_(model.fc2.bias)
        # Tail-concentrated damage (the RQ1 finding, imposed by construction):
        # compression degrades the minority-serving circuitry much more than
        # the majority circuitry.
        pruned.fc1.weight[:, D_MAJ_FEAT:] *= CUE_DAMAGE
    return pruned, keep_idx


def run_seed(seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    teacher_a = torch.from_numpy(rng.standard_normal((N_CLASSES, D_MAJ_FEAT))).float()
    teacher_b = torch.from_numpy(rng.standard_normal((N_CLASSES, D_MIN_FEAT))).float()

    x_tr, y_tr, g_tr = make_data(N_TRAIN, teacher_a, teacher_b, rng)
    x_va, y_va, g_va = make_data(N_VAL, teacher_a, teacher_b, rng)
    x_te, y_te, g_te = make_data(N_TEST, teacher_a, teacher_b, rng)

    results = {}

    # --- Stage 1: dense model -------------------------------------------------
    dense = MLP(HIDDEN)
    train_steps(dense, x_tr, y_tr, PRETRAIN_STEPS, LR_PRETRAIN, alpha=1.0, rng=rng)
    results["dense"] = group_err(dense, x_te, y_te, g_te)

    # --- Stage 2: prune by average importance ---------------------------------
    pruned, keep_idx = prune_by_average_importance(dense, x_tr, g_tr)
    results["pruned"] = group_err(pruned, x_te, y_te, g_te)

    # --- Stage 3/4: LoRA-style recovery, mean CE vs batch-CVaR, same budget ---
    for name, alpha in [("mean", 1.0), ("cvar", CVAR_ALPHA)]:
        model = LoRARecovery(copy.deepcopy(pruned))
        sd_agg, sd_tail = train_steps(
            model, x_tr, y_tr, RECOVERY_STEPS, LR_RECOVER, alpha=alpha,
            rng=np.random.default_rng(seed + 1000), x_val=x_va, y_val=y_va,
        )
        for sel, sd in [("agg_ckpt", sd_agg), ("tail_ckpt", sd_tail)]:
            m = LoRARecovery(copy.deepcopy(pruned))
            m.load_state_dict(sd)
            results[f"{name}_{sel}"] = group_err(m, x_te, y_te, g_te)

    return results


def main():
    stages = [
        ("dense",          "Dense (pre-prune)"),
        ("pruned",         "Compressed (prune + tail damage, no recovery)"),
        ("mean_agg_ckpt",  "Recovery: mean CE + aggregate ckpt  [status quo]"),
        ("mean_tail_ckpt", "Recovery: mean CE + tail ckpt"),
        ("cvar_agg_ckpt",  f"Recovery: CVaR@{CVAR_ALPHA} + aggregate ckpt"),
        ("cvar_tail_ckpt", f"Recovery: CVaR@{CVAR_ALPHA} + tail ckpt   [de-averaged]"),
    ]

    all_results = [run_seed(s) for s in SEEDS]

    def agg_stat(key):
        vals = np.array([r[key] for r in all_results])  # (seeds, 3)
        mean, std = vals.mean(axis=0), vals.std(axis=0)
        ratio = vals[:, 2] / np.maximum(vals[:, 1], 1e-9)
        return mean, std, ratio.mean(), ratio.std()

    print()
    print(f"Synthetic De-Averaged Pruning demo | {len(SEEDS)} seeds | "
          f"minority share {MINORITY_FRAC:.0%} | keep {KEEP}/{HIDDEN} units | "
          f"LoRA-rank-{RECOVERY_RANK} recovery, {RECOVERY_STEPS} steps "
          f"(identical budget for both arms)")
    print()
    header = (f"{'stage':<52} {'overall':>9} {'majority':>9} "
              f"{'minority':>9} {'min/maj':>8}")
    print(header)
    print("-" * len(header))
    for key, label in stages:
        mean, std, rmean, rstd = agg_stat(key)
        print(f"{label:<52} "
              f"{mean[0]*100:5.1f}%   "
              f"{mean[1]*100:5.1f}%   "
              f"{mean[2]*100:5.1f}%   "
              f"{rmean:5.1f}x")

    # ------------------------------------------------------------------ verdict
    dense_m, _, dense_r, _ = agg_stat("dense")
    prune_m, _, prune_r, _ = agg_stat("pruned")
    mean_m, _, mean_r, _ = agg_stat("mean_agg_ckpt")
    cvar_m, _, cvar_r, _ = agg_stat("cvar_tail_ckpt")

    d_maj = prune_m[1] - dense_m[1]
    d_min = prune_m[2] - dense_m[2]

    print()
    print("Verdict (analog of the paper's claims):")

    # 1. RQ1 analog: compression damage concentrates on the minority, and the
    #    aggregate (~= majority at 2% share) cannot see it.
    conc = d_min > 2.5 * d_maj
    print(f"  1. Damage concentrates on the tail:  majority +{d_maj*100:.1f}pp vs "
          f"minority +{d_min*100:.1f}pp ({d_min/max(d_maj,1e-9):.1f}x)  "
          f"[{'YES' if conc else 'no'}]")

    # 2. RQ2/RQ5 analog: mean-loss recovery restores the aggregate but leaves
    #    the minority harmed and worsens relative disparity vs. dense.
    mean_leaves = mean_m[2] > 2.0 * dense_m[2] and mean_r > 1.4 * dense_r
    print(f"  2. Mean-CE recovery leaves the tail harmed: aggregate "
          f"{prune_m[0]*100:.1f}%->{mean_m[0]*100:.1f}% but minority stuck at "
          f"{mean_m[2]*100:.1f}% (dense {dense_m[2]*100:.1f}%), "
          f"ratio {dense_r:.1f}x->{mean_r:.1f}x  [{'YES' if mean_leaves else 'no'}]")

    # 3. RQ3: batch-CVaR + tail checkpoint, at equal budget, no group labels.
    #    Report both magnitude and per-seed sign consistency.
    agg_ok = cvar_m[0] <= mean_m[0] + 0.01
    per_seed_min = [(r["mean_agg_ckpt"][2], r["cvar_tail_ckpt"][2]) for r in all_results]
    per_seed_ratio = [
        (r["mean_agg_ckpt"][2] / max(r["mean_agg_ckpt"][1], 1e-9),
         r["cvar_tail_ckpt"][2] / max(r["cvar_tail_ckpt"][1], 1e-9))
        for r in all_results
    ]
    min_wins = sum(c < m for m, c in per_seed_min)
    ratio_wins = sum(c < m for m, c in per_seed_ratio)
    print(f"  3. De-averaged recovery (label-free): minority "
          f"{mean_m[2]*100:.1f}%->{cvar_m[2]*100:.1f}%, ratio {mean_r:.1f}x->{cvar_r:.1f}x, "
          f"aggregate {cvar_m[0]*100:.1f}% (mean-CE arm {mean_m[0]*100:.1f}%)")
    print(f"     per-seed minority (mean-CE vs CVaR): "
          + "  ".join(f"{m*100:.0f}->{c*100:.0f}%" for m, c in per_seed_min))
    print(f"     -> aggregate held: {'YES' if agg_ok else 'NO'} | "
          f"minority better in {min_wins}/{len(SEEDS)} seeds | "
          f"ratio better in {ratio_wins}/{len(SEEDS)} seeds")
    print()

    direction_ok = agg_ok and min_wins >= len(SEEDS) - 1 and ratio_wins >= len(SEEDS) - 1
    if conc and mean_leaves and direction_ok:
        print("DEMO RESULT: mechanism validated end-to-end.")
        print("  - Compression damage concentrates on the minority; the aggregate hides it (RQ1).")
        print("  - Mean-CE recovery restores the aggregate but worsens relative disparity (RQ2/RQ5).")
        print("  - Batch-CVaR + tail checkpoint moves the disparity in the mitigating direction")
        print("    at equal budget and aggregate, without group labels (RQ3) — but the effect")
        print("    size in this toy is MODEST. CVaR's leverage depends on how strongly the")
        print("    damage concentrates in the loss tail. That is exactly what the free")
        print("    go/no-go gate must check on the real per-utterance data before any GPU run")
        print("    (see acceptance_test_deciles.py).")
    else:
        print("DEMO RESULT: criteria not all met on this run — inspect the table above.")


if __name__ == "__main__":
    main()
