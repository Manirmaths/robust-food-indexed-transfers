"""Gate 1 with NLSS poor-household weights (zone-specific) and poor-population recipient counts."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "..", "NBS_Research_Datasets_2026-11-01", "data")
DER, TAB, FIG = (os.path.join(ROOT, *p) for p in [("data", "derived"), ("results", "tables"), ("results", "figures")])
s = pd.read_csv(os.path.join(DER, "food_state_clean.csv"), parse_dates=["date"])
zone = s.drop_duplicates("state").set_index("state").zone
P = s.pivot_table(index=["date", "state"], columns="item", values="price")
IW = pd.read_csv(os.path.join(DER, "basket_item_weights.csv"))
R = pd.read_csv(os.path.join(DER, "state_recipients.csv")).set_index("state")
macro = pd.read_csv(os.path.join(DATA, "national_macro_monthly_2015_2026.csv"), parse_dates=["date"]).set_index("date")

def weights(variant):
    pop, drop = {"nlss_poor_zonal": ("poor", None), "nlss_poor_national": ("poor", None),
                 "nlss_poor_no_other_grains": ("poor", "Grains (other)"), "nlss_all_zonal": ("all", None)}[variant]
    w = IW[IW.population == pop]
    if drop: w = w[w.group != drop]
    out = {}
    for st, z in zone.items():
        scope = "National" if variant == "nlss_poor_national" else z
        ww = w[w.scope == scope].set_index("item").weight
        out[st] = ww / ww.sum()
    return pd.DataFrame(out).T.fillna(0.0)          # state x item

def need(Wm, base):
    b = P.xs(pd.Timestamp(base), level="date")
    rel = P.div(b).reindex(columns=Wm.columns)
    st = rel.index.get_level_values("state")
    return (rel.fillna(0).values * Wm.loc[st].values).sum(axis=1)

def eff(T, D, N):
    return float((T / D).min() * (N * D).sum() / (N * T).sum())

rows = []
for variant in ["nlss_poor_zonal", "nlss_poor_national", "nlss_poor_no_other_grains", "nlss_all_zonal"]:
    Wm = weights(variant)
    for base in ["2019-01-01", "2023-05-01"]:
        D = pd.Series(need(Wm, base), index=P.index).unstack("state")
        for Nname, N in [("poor_population", R.poor_population_2019), ("equal", pd.Series(1.0, R.index))]:
            N = N.reindex(D.columns)
            dates = list(D.index)
            for i, t in enumerate(dates):
                if t <= pd.Timestamp(base): continue
                d = D.loc[t]
                Zm = (d * N).groupby(zone).transform("sum") / N.groupby(zone).transform("sum")
                r = dict(weights=variant, base=base[:7], recipients=Nname, date=t, max_min_ratio=d.max() / d.min(),
                         E_flat=eff(pd.Series(1.0, d.index), d, N), E_zonal=eff(Zm, d, N))
                for lag in (1, 3, 6):
                    ok = i - lag >= 0 and dates[i - lag] > pd.Timestamp(base)
                    r[f"E_carry{lag}"] = eff(D.iloc[i - lag], d, N) if ok else np.nan
                mean_need = (N * d).sum() / N.sum()
                r["adequacy_fixed"] = 1 / mean_need
                r["adequacy_cpi_food"] = (macro.cpi_food_index_b2024.loc[t] / macro.cpi_food_index_b2024.loc[pd.Timestamp(base)]) / mean_need
                r["most_exposed_state"] = d.idxmax()
                rows.append(r)
G = pd.DataFrame(rows); G.to_csv(os.path.join(TAB, "gate1_nlss_by_month.csv"), index=False)
G["period"] = np.where(G.date >= "2023-06-01", "post", "pre")
summ = G.groupby(["weights", "recipients", "base", "period"])[["E_flat", "E_zonal", "E_carry1", "E_carry3", "E_carry6", "adequacy_cpi_food"]].agg(["mean", "min"]).round(3)
summ.to_csv(os.path.join(TAB, "gate1_nlss_summary.csv"))
main = summ.xs(("nlss_poor_zonal", "poor_population"), level=("weights", "recipients"))
print(main.to_string())
print("\nrobustness (base 2023-05, post-reform, poor_population): E_flat mean/min, E_zonal mean/min")
print(summ.xs(("2023-05", "post"), level=("base", "period"))[[("E_flat", "mean"), ("E_flat", "min"), ("E_zonal", "mean"), ("E_zonal", "min")]].to_string())
x = G[(G.weights == "nlss_poor_zonal") & (G.recipients == "poor_population") & (G.base == "2023-05")]
print(x[["date", "E_flat", "E_zonal", "E_carry3", "adequacy_cpi_food", "max_min_ratio", "most_exposed_state"]].round(3).to_string(index=False))
y = G[(G.weights == "nlss_poor_zonal") & (G.recipients == "poor_population") & (G.base == "2019-01")].set_index("date")
fig, ax = plt.subplots(figsize=(7.2, 3.6))
for c, lab in [("E_flat", "Flat transfer (status quo)"), ("E_zonal", "Zonal indexation"), ("E_carry3", "State prices, 3-month delay")]:
    ax.plot(y.index, y[c], label=lab, lw=1.6)
ax.axvspan(pd.Timestamp("2023-06-01"), y.index.max(), color="0.9", zorder=0, label="Post-reform")
ax.set_ylabel("Allocative efficiency E"); ax.set_ylim(0, 1.02); ax.legend(frameon=False, fontsize=8, loc="lower left")
ax.set_title("Share of attainable protection delivered (NLSS poor-household basket, recipients = poor population)", fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_gate1_nlss.pdf")); fig.savefig(os.path.join(FIG, "fig_gate1_nlss.png"), dpi=200)
