"""RQ2 figure — the LoRA "scissors": adaptation lowers aggregate WER while
widening the Black-Asian gap, at every pruning depth (Whisper large-v2,
Fair-Speech, zero-shot).

(a) Aggregate WER vs layers removed: +LoRA sits BELOW base (better average).
(b) Black-Asian gap vs layers removed: +LoRA sits ABOVE base (wider gap).

Same style as fig_rq1. Outputs results/figures/fig_rq2.{pdf,png}; prints
plotted values for audit.
"""
import csv, os, glob, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
LORA_DIR = r"P:\Programming\Bias in pruning exps\all_results\largev2_fairspeech_LORA_sweep"
OUT = os.path.join(RES, "figures")
os.makedirs(OUT, exist_ok=True)

BASE_C, LORA_C = "#5f5e5a", "#c1392b"     # base gray, +LoRA red
INK, GRID = "#111111", "#e2e1dc"

plt.rcParams.update({
    "font.size": 9, "font.family": "sans-serif",
    "axes.edgecolor": "#8f8e88", "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
})

B = "black or african american"
A = "asian, south asian or asian american"

def load_findings(path):
    out = collections.defaultdict(dict)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["axis"] == "ethnicity" and r["analysable"] == "yes":
            out[int(r["layers_kept"])][r["group"]] = float(r["wer"]) * 100
    return out

def load_raw(folder):
    out = collections.defaultdict(dict)
    for p in glob.glob(os.path.join(folder, "*_multiaxis.csv")):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r["axis"] != "ethnicity" or r["analysable"] != "yes":
                continue
            k = next(int(t[4:]) for t in r["condition"].split("_") if t.startswith("keep"))
            out[k][r["group"]] = float(r["wer"]) * 100
    return out

base = load_findings(os.path.join(RES, "largev2_fairspeech_sweep", "findings_ethnicity.csv"))
lora = load_raw(LORA_DIR)

TOT = 32
keeps = [32, 30, 28, 26, 24]           # usable range, matches fig_rq1 / tables
x = [TOT - k for k in keeps]

agg_b = [base[k]["ALL"] for k in keeps]
agg_l = [lora[k]["ALL"] for k in keeps]
# ratio, not plain gap: LoRA lowers BOTH groups' WER, so the absolute gap is
# ~tied at shallow depths; the unfairness is RELATIVE (Asian helped ~2x more),
# which the Black/Asian ratio shows at every depth.
gap_b = [base[k][B] / base[k][A] for k in keeps]
gap_l = [lora[k][B] / lora[k][A] for k in keeps]

fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.6))
fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.28, wspace=0.28)

def style(ax, title):
    ax.set_title(title, fontsize=9, fontweight="bold", color=INK, pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L-{v}" for v in x])
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=3)

# ---- (a) aggregate WER: LoRA better -------------------------------------
ax = axes[0]
for ys, c, lbl, dy in ((agg_b, BASE_C, "Base", 8), (agg_l, LORA_C, "+LoRA", -15)):
    ax.plot(x, ys, color=c, lw=2.0, marker="o", ms=3.8, markerfacecolor=c,
            markeredgecolor="white", markeredgewidth=0.6, zorder=4)
    ax.annotate(lbl, (x[0], ys[0]), xytext=(2, dy), textcoords="offset points",
                fontsize=8.5, color=c, fontweight="bold", ha="left")
style(ax, "(a) Aggregate WER — LoRA helps")
ax.set_ylabel("WER (%)", fontsize=9)
ax.set_ylim(14, 41)

# ---- (b) Black-Asian gap: LoRA worsens ------------------------------------
ax = axes[1]
for ys, c, lbl, dy in ((gap_b, BASE_C, "Base", -11), (gap_l, LORA_C, "+LoRA", 6)):
    ax.plot(x, ys, color=c, lw=2.0, marker="o", ms=3.8, markerfacecolor=c,
            markeredgecolor="white", markeredgewidth=0.6, zorder=4)
    ax.annotate(lbl, (x[-1], ys[-1]), xytext=(-2, dy), textcoords="offset points",
                fontsize=8.5, color=c, fontweight="bold", ha="right")
style(ax, "(b) Black/Asian WER ratio — LoRA widens")
ax.set_ylabel("WER ratio (Black / Asian)", fontsize=9)
ax.set_ylim(1.80, 2.48)

fig.text(0.53, 0.045, "Encoder layers removed (Whisper large-v2)",
         ha="center", fontsize=9, color=INK)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_rq2.{ext}"), dpi=300)
print("wrote fig_rq2.pdf/png")
print("agg base:", [round(v, 1) for v in agg_b], " agg lora:", [round(v, 1) for v in agg_l])
print("gap base:", [round(v, 1) for v in gap_b], " gap lora:", [round(v, 1) for v in gap_l])
