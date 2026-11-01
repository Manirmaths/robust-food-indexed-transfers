"""Summary tables (LaTeX-ready CSVs) and figures for the Phase C results."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB, FIG = os.path.join(ROOT, "results", "tables"), os.path.join(ROOT, "results", "figures")
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", grey="#8a8985", ink="#0b0b0b", ink2="#52514e")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": C["ink2"],
                     "axes.labelcolor": C["ink"], "xtick.color": C["ink2"], "ytick.color": C["ink2"], "axes.grid": True,
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "lines.linewidth": 1.8, "legend.frameon": False})

# ---- frontier table
R = pd.read_csv(os.path.join(TAB, "e3c_frontier_by_month.csv"), parse_dates=["date"])
names = ["flat", "zonal", "plugin", "k50", "k80", "k85", "k90", "k95", "k100"]
rows = []
for tag, x in R.groupby("weights"):
    for n in names:
        rows.append(dict(weights=tag, rule=n, E_mean=x[f"E_{n}"].mean(), E_min=x[f"E_{n}"].min(),
                         cert_mean=x[f"cert_{n}"].mean(), cert_min=x[f"cert_{n}"].min(),
                         holds=(x[f"E_{n}"] >= x[f"cert_{n}"] - 1e-12).mean(), months=len(x), coverage=x.covered.mean(), G_mean=x.G.mean()))
F = pd.DataFrame(rows); F.to_csv(os.path.join(TAB, "summary_frontier.csv"), index=False)
print(F[F.weights == "poor_zonal"].round(3).to_string(index=False))
print(F.pivot_table(index="rule", columns="weights", values="E_mean").reindex(names).round(3).to_string())
print(F.pivot_table(index="rule", columns="weights", values="cert_mean").reindex(names).round(3).to_string())

fig, ax = plt.subplots(figsize=(4.6, 3.2))
x = F[F.weights == "poor_zonal"].set_index("rule")
kap = [n for n in names if n.startswith("k")]
ax.plot(x.loc[kap, "cert_mean"], x.loc[kap, "E_mean"], "-o", color=C["blue"], ms=5, label="guarantee-constrained, $\\kappa$=0.5 to 1")
for n in ("k50", "k90", "k100"):
    ax.annotate(f"$\\kappa$={int(n[1:])/100:g}", (x.loc[n, "cert_mean"], x.loc[n, "E_mean"]), xytext=(5, 4), textcoords="offset points", color=C["ink2"], fontsize=8)
for n, lab, col in (("flat", "flat", C["orange"]), ("zonal", "zonal", C["yellow"]), ("plugin", "plug-in", C["aqua"])):
    ax.plot(x.loc[n, "cert_mean"], x.loc[n, "E_mean"], "s", color=col, ms=6)
    ax.annotate(lab, (x.loc[n, "cert_mean"], x.loc[n, "E_mean"]), xytext=(5, -10), textcoords="offset points", color=C["ink2"], fontsize=8)
ax.set_xlabel("certified worst-case efficiency (mean)"); ax.set_ylabel("realised efficiency (mean)")
ax.legend(loc="lower right", fontsize=8); fig.tight_layout()
for e in ("pdf", "png"): fig.savefig(os.path.join(FIG, f"fig_frontier.{e}"), dpi=200)

# ---- monitoring
M = pd.read_csv(os.path.join(TAB, "e5_monitoring_curve.csv")); t = M[M.split == "test"]
fig, ax = plt.subplots(figsize=(4.6, 3.2))
for (d, des), lab, col, ls in (((0, "decision"), "decision-focused, no delay", C["blue"], "-"),
                                ((0, "accuracy"), "accuracy-focused, no delay", C["orange"], "--"),
                                ((1, "decision"), "decision-focused, 1-month delay", C["aqua"], "-"),
                                ((3, "decision"), "decision-focused, 3-month delay", C["grey"], "-")):
    y = t[(t.delay == d) & (t.design == des)].sort_values("K")
    ax.plot(y.K, y.G_mean, ls, color=col, label=lab)
ax.set_xlabel("states monitored, K"); ax.set_ylabel("mean guarantee G, test months"); ax.legend(fontsize=7.5); fig.tight_layout()
for e in ("pdf", "png"): fig.savefig(os.path.join(FIG, f"fig_monitoring.{e}"), dpi=200)
t.to_csv(os.path.join(TAB, "summary_monitoring_test.csv"), index=False)
print(t[t.K.isin([0, 1, 3, 5, 10, 15])].pivot_table(index=["design", "delay"], columns="K", values="G_mean").round(3).to_string())
print(t[t.K.isin([0, 3, 10, 15])].pivot_table(index=["design", "delay"], columns="K", values="E_mean").round(3).to_string())

# ---- live
L = pd.read_csv(os.path.join(TAB, "e8_live_by_month.csv"), parse_dates=["date"])
fig, ax = plt.subplots(figsize=(6.2, 3.0))
for c, lab, col in (("G", "guarantee G", C["ink"]), ("cert_k90", "$\\kappa$=0.9 rule", C["blue"]), ("cert_plugin", "plug-in", C["aqua"]),
                    ("cert_flat", "flat", C["orange"]), ("cert_zonal", "zonal", C["yellow"])):
    ax.plot(L.date, L[c], color=col, label=lab, lw=2.2 if c == "G" else 1.8)
ax.axvline(pd.Timestamp("2025-01-01"), color=C["grey"], lw=0.8, ls=":")
ax.text(pd.Timestamp("2025-01-15"), 0.83, "26-item basket\npublished from 2025", fontsize=7.5, color=C["ink2"], va="top")
ax.set_ylabel("certified worst-case efficiency"); ax.legend(fontsize=7.5, ncol=3, loc="lower left"); fig.tight_layout()
for e in ("pdf", "png"): fig.savefig(os.path.join(FIG, f"fig_live.{e}"), dpi=200)
print(L.iloc[[0, 3, 8, 15, 16, 24, -1]][["date", "horizon", "G", "cert_k90", "cert_plugin", "cert_flat", "cert_zonal"]].round(3).to_string(index=False))

# ---- price-cell polytope backtest summary
B = pd.read_csv(os.path.join(TAB, "e3_e4_backtest_by_month.csv"), parse_dates=["date"]); b = B[B.weights == "poor_zonal"]
cols = ["E_flat", "E_zonal", "E_plugin", "E_rob_I1_aci_a20", "E_rob_I2_aci_a20", "G_I1_aci_a20", "G_I2_aci_a20", "cov_I1_aci_a20"]
print("price-cell backtest", len(b), b.date.min().date(), b.date.max().date()); print(b[cols].agg(["mean", "min"]).round(3).to_string())
