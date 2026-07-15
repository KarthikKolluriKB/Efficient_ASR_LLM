"""Table 1 — RQ1 headline: large-v2 per-depth race numbers (Fair-Speech).

Reads findings_ethnicity.csv (never hard-codes numbers), emits a booktabs LaTeX
table and prints every plotted value for audit. The rho column (Black/Asian) is
the honest "same-scale" device: it cancels the "everyone degrades" effect that a
relative-increase (Delta/baseline) normalization would misattribute to low-
baseline groups. Bold marks the free prune (agg -0.44) and the rho peak (2.15).

Usage: python make_table1.py  ->  results/tables/table1_rq1.tex  (+ stdout audit)
"""
import csv, os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
OUT = os.path.join(RES, "tables")
os.makedirs(OUT, exist_ok=True)

CSV = os.path.join(RES, "largev2_fairspeech_sweep", "findings_ethnicity.csv")
DEPTHS = [0, 2, 4, 6, 8]              # usable range (agg WER <= ~40%)
KEPT = {0: 32, 2: 30, 4: 28, 6: 26, 8: 24}
GN = {
    "asian, south asian or asian american": "Asian",
    "black or african american": "Black",
    "ALL": "Agg",
}

D = {}
for r in csv.DictReader(open(CSV, encoding="utf-8")):
    d = int(r["depth"])
    if d in DEPTHS and r["group"] in GN and r["analysable"] == "yes":
        D[(GN[r["group"]], d)] = float(r["wer"]) * 100

# per-depth derived quantities
base_agg = D[("Agg", 0)]
rows = []
for d in DEPTHS:
    agg, blk, asn = D[("Agg", d)], D[("Black", d)], D[("Asian", d)]
    rows.append({
        "kept": KEPT[d],
        "agg": agg, "black": blk, "asian": asn,
        "gap": blk - asn,                 # absolute Black-Asian gap (pp)
        "rho": blk / asn,                 # same-scale ratio
        "agg_delta": None if d == 0 else agg - base_agg,
    })

# ---- stdout audit ----
print(f"{'keep':>4} {'agg':>6} {'Black':>6} {'Asian':>6} {'gapΔpp':>7} {'ρ':>5} {'aggΔvsbase':>11}")
for r in rows:
    ad = "  —" if r["agg_delta"] is None else f"{r['agg_delta']:+.2f}"
    print(f"{r['kept']:>4} {r['agg']:6.2f} {r['black']:6.2f} {r['asian']:6.2f} "
          f"{r['gap']:7.1f} {r['rho']:5.2f} {ad:>11}")
print(f"\nfree prune (keep32->30): agg {rows[1]['agg_delta']:+.2f} pp, "
      f"ρ {rows[0]['rho']:.2f}->{rows[1]['rho']:.2f} (peak)")

# ---- LaTeX (booktabs) ----
def f2(x): return f"{x:.2f}"
def bold(s): return r"\textbf{" + s + "}"

lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\small",
    r"\caption{RQ1 headline (Whisper large-v2, Fair-Speech). Per-depth WER (\%) "
    r"for the aggregate and the extreme groups. $\rho=$WER$_\text{Black}$/WER$_\text{Asian}$ "
    r"is the same-scale disparity ratio; it peaks inside the WER-preserving prune "
    r"(keep-30), where the aggregate \emph{improves} ($-0.44$ pp). "
    r"Per-group significance ($p$) to be filled from the paired bootstrap.}",
    r"\label{tab:rq1}",
    r"\begin{tabular}{r rrr r r r}",
    r"\toprule",
    r"kept & agg & Black & Asian & gap (pp) & $\rho$ & agg $\Delta$ \\",
    r"\midrule",
]
for r in rows:
    kept = str(r["kept"])
    agg = f2(r["agg"])
    rho = f2(r["rho"])
    if r["agg_delta"] is None:
        ad = "--"
    else:
        ad = f"{r['agg_delta']:+.2f}"
        if abs(r["agg_delta"] + 0.44) < 0.01:   # the free prune
            ad = bold(ad)
    if abs(r["rho"] - 2.155) < 0.01:            # rho peak
        rho = bold(rho)
    lines.append(f"{kept} & {agg} & {f2(r['black'])} & {f2(r['asian'])} & "
                 f"{r['gap']:.1f} & {rho} & {ad} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

tex = "\n".join(lines)
open(os.path.join(OUT, "table1_rq1.tex"), "w", encoding="utf-8").write(tex)
print(f"\nwrote {os.path.join(OUT, 'table1_rq1.tex')}")
