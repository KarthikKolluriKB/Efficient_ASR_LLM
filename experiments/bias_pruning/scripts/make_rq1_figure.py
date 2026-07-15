"""RQ1 figure — three independent panels, one per model scale.

Each panel has its OWN x-axis (number of encoder layers removed) and its OWN
y-axis fitted to that model's data, so no scale is cramped. Six groups shown;
Black (red) is emphasized. The aggregate (gray, dashed) degrades alongside the
groups at small/medium but is preserved at large-v2's first prune (shaded),
where Black WER already rises (+0.9 pp, p=.018).

Outputs: results/figures/fig_rq1.{pdf,png}; prints plotted values for audit.
"""
import csv, os, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
OUT = os.path.join(RES, "figures")
os.makedirs(OUT, exist_ok=True)

BAND, INK, MUTED, GRID = "#dce8f7", "#111111", "#4f4e4a", "#e2e1dc"

plt.rcParams.update({
    "font.size": 9, "font.family": "sans-serif",
    "axes.edgecolor": "#8f8e88", "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
})

def load_eth(path):
    out = collections.defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["axis"] != "ethnicity" or r["analysable"] != "yes":
                continue
            out[int(r["layers_kept"])][r["group"]] = float(r["wer"]) * 100
    return out

sml = load_eth(os.path.join(RES, "fairspeech_sweep", "findings_ethnicity.csv"))
med = load_eth(os.path.join(RES, "medium_fairspeech_sweep", "findings_ethnicity.csv"))
lv2 = load_eth(os.path.join(RES, "largev2_fairspeech_sweep", "findings_ethnicity.csv"))
# NOTE: medium keep-22 has verified values only for Black/Asian/Aggregate; the
# other groups' keep-22 values are not in the local export, so the medium panel
# uses keep {24,20,18} (all groups real). Black keep-22 = 29.49 lives in Table 1.

# group -> (label, color, linewidth, emphasis)  — fixed colors per group
GROUPS = [
    ("black or african american", "Black", "#d62728", 2.4, True),
    ("hispanic, latino, or spanish", "Hispanic", "#e07b1a", 1.4, False),
    ("native american, american indian, or alaska native", "Native Am.", "#7a5bd0", 1.4, False),
    ("white", "White", "#1f9e63", 1.4, False),
    ("asian, south asian or asian american", "Asian", "#1f6fc0", 1.6, False),
    ("ALL", "Aggregate", "#5f5e5a", 1.6, False),
]

# (title, data, total layers, kept-depths in usable range)
PANELS = [
    ("(a) small · 12 layers", sml, 12, [12, 11, 10]),
    ("(b) medium · 24 layers", med, 24, [24, 20, 18, 16]),
    ("(c) large-v2 · 32 layers", lv2, 32, [32, 30, 28, 26, 24]),
]

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.9))
fig.subplots_adjust(left=0.065, right=0.99, top=0.90, bottom=0.30, wspace=0.28)

for i, (ax, (title, dm, tot, keeps)) in enumerate(zip(axes, PANELS)):
    removed = [tot - k for k in keeps]
    last = i == 2

    if last:  # large-v2's WER-preserving regime (the first 2-layer prune)
        ax.axvspan(0, 2, color=BAND, alpha=0.9, zorder=1)

    ymin, ymax = 1e9, -1e9
    for g, lbl, c, lw, emph in GROUPS:
        vals = [dm[k][g] for k in keeps]
        ymin, ymax = min(ymin, *vals), max(ymax, *vals)
        ls = (0, (4, 2)) if g == "ALL" else "-"
        ax.plot(removed, vals, color=c, lw=lw + (0.3 if emph else 0), ls=ls,
                marker="o", ms=3.6 if emph else 3.0, markerfacecolor=c,
                markeredgecolor="white", markeredgewidth=0.6,
                zorder=5 if emph else 4, label=lbl if i == 0 else None)

    ax.set_title(title, fontsize=9, fontweight="bold", color=INK, pad=6)
    pad = (ymax - ymin) * 0.10
    ax.set_ylim(ymin - pad, ymax + pad * (2.6 if last else 1.4))
    ax.set_xlim(-max(removed) * 0.04, max(removed) * 1.06)
    ax.set_xticks(removed)
    ax.set_ylabel("WER (%)", fontsize=8.5)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=3)

    if last:  # concealment note in the empty upper-left of panel (c)
        ax.text(0.18, ymax + pad * 1.4,
                "shaded = WER-preserving prune:\nAggregate $-$0.4 pp,\nBlack $+$0.9 pp ($p$ = .018)",
                fontsize=6.9, color=INK, ha="left", va="top", linespacing=1.3)

axes[1].set_xlabel("Number of encoder layers removed", fontsize=9)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False,
           fontsize=8.2, handlelength=1.6, columnspacing=1.4,
           bbox_to_anchor=(0.5, 0.005))

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_rq1.{ext}"), dpi=300)
print("wrote fig_rq1.pdf/png")
for title, dm, tot, keeps in PANELS:
    print(title)
    for g, lbl, *_ in GROUPS:
        print(f"   {lbl:11}", [round(dm[k][g], 1) for k in keeps])
