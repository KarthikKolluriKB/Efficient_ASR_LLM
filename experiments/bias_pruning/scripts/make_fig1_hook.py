"""Fig 1 — the hook, two panels with shared group rows.

(a) baseline per-group WER (Whisper large-v2, 32 layers, Fair-Speech) as
neutral horizontal bars — shows the inherited disparity (Black ~2x Asian).
(b) signed per-group DeltaWER after pruning to 30 layers as diverging bars —
improvements blue, regressions red. Aggregate outlined + bold in both panels.
No annotations or value labels; axes only. Rows sorted by Delta (most improved
top, Black bottom). Reads from CSV; prints values for audit.

Outputs: results/figures/fig1_hook.{pdf,png}
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
OUT = os.path.join(RES, "figures")
os.makedirs(OUT, exist_ok=True)
CSV = os.path.join(RES, "largev2_fairspeech_sweep", "findings_ethnicity.csv")

BLUE, RED, GRAY = "#2a78d6", "#e34948", "#b8b7b0"
INK, GRID, ZERO = "#111111", "#e1e0d9", "#8f8e88"
LABEL = {
    "asian, south asian or asian american": "Asian",
    "black or african american": "Black",
    "hispanic, latino, or spanish": "Hispanic",
    "middle eastern or north african": "MENA",
    "native american, american indian, or alaska native": "Native American",
    "native hawaiian or other pacific islander": "Native Hawaiian",
    "white": "White", "ALL": "ALL (aggregate)",
}

wer = {0: {}, 2: {}}
for r in csv.DictReader(open(CSV, encoding="utf-8")):
    d = int(r["depth"])
    if d in wer and r["group"] in LABEL and r["analysable"] == "yes":
        wer[d][r["group"]] = float(r["wer"]) * 100

delta = {g: wer[2][g] - wer[0][g] for g in wer[0]}
order = sorted(delta, key=lambda g: delta[g])   # most improved first, Black last

plt.rcParams.update({"font.size": 8, "font.family": "sans-serif",
                     "axes.edgecolor": ZERO, "axes.linewidth": 0.8})
fig, (axa, axb) = plt.subplots(1, 2, figsize=(5.6, 2.4), sharey=True,
                               gridspec_kw={"width_ratios": [1, 1]})
fig.subplots_adjust(left=0.235, right=0.98, top=0.88, bottom=0.19, wspace=0.10)

y = list(range(len(order)))[::-1]   # most improved at top, Black at bottom

# (a) baseline absolute WER
for yi, g in zip(y, order):
    is_agg = g == "ALL"
    axa.barh(yi, wer[0][g], height=0.66, color=GRAY, zorder=3,
             edgecolor=INK if is_agg else "none", linewidth=0.9 if is_agg else 0)
axa.set_title("(a) Baseline WER — 32 layers", fontsize=8, fontweight="bold",
              color=INK, pad=5)
axa.set_xlabel("WER (%)", fontsize=8)
axa.set_xlim(0, 30)
axa.set_xticks([0, 10, 20, 30])

# (b) delta after pruning 32 -> 30
for yi, g in zip(y, order):
    v = delta[g]
    is_agg = g == "ALL"
    axb.barh(yi, v, height=0.66, color=RED if v > 0 else BLUE, zorder=3,
             edgecolor=INK if is_agg else "none", linewidth=0.9 if is_agg else 0)
axb.axvline(0, color=ZERO, lw=0.9, zorder=2)
axb.set_title("(b) Change after pruning to 30", fontsize=8, fontweight="bold",
              color=INK, pad=5)
axb.set_xlabel(r"$\Delta$WER (pp)", fontsize=8)
axb.set_xlim(-2.9, 1.35)
axb.set_xticks([-2, -1, 0, 1])

for ax in (axa, axb):
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)

axa.set_yticks(y)
axa.set_yticklabels([r"$\bf{ALL}$" if g == "ALL" else LABEL[g] for g in order],
                    fontsize=7.5)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig1_hook.{ext}"), dpi=300)

print(f"{'group':18s} {'base':>6s} {'pruned':>7s} {'Δ':>6s}")
for g in order:
    print(f"{LABEL[g]:18s} {wer[0][g]:6.2f} {wer[2][g]:7.2f} {delta[g]:+6.2f}")
print("wrote fig1_hook.pdf/png")
