"""Gate 1 (blueprint E2 and E10): how much protection does a flat transfer lose,
and how does protection split into budget adequacy and allocative efficiency?
Placeholder inputs until NLSS weights and recipient counts are added:
equal recipients per state; equal-weight (43 items) and staples baskets."""
import os, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "..", "NBS_Research_Datasets_2026-11-01", "data")
TAB, FIG = os.path.join(ROOT, "results", "tables"), os.path.join(ROOT, "results", "figures")
s = pd.read_csv(os.path.join(ROOT, "data", "derived", "food_state_clean.csv"), parse_dates=["date"])
zone = s.drop_duplicates("state").set_index("state").zone
P = s.pivot_table(index=["date", "state"], columns="item", values="price")
STAPLES = ["Rice local sold loose", "Rice,imported high quality sold loose", "Beans brown,sold loose",
           "Beans:white black eye. sold loose", "Maize grain white sold loose", "Maize grain yellow sold loose",
           "Gari white,sold loose", "Gari yellow,sold loose", "Yam tuber", "Bread unsliced 500g",
           "Wheat flour: prepacked (golden penny 2kg)", "Vegetable oil:1 bottle,specify bottle",
           "Palm oil: 1 bottle,specify bottle"]
BASKETS = {"equal43": [c for c in P.columns if P[c].notna().mean() > 0.99], "staples13": STAPLES}
macro = pd.read_csv(os.path.join(DATA, "national_macro_monthly_2015_2026.csv"), parse_dates=["date"]).set_index("date")

def need_index(items, base):
    b = P.xs(pd.Timestamp(base), level="date")[items]
    return P[items].div(b).mean(axis=1).unstack("state")      # D_st / T0 (Laspeyres, equal shares)

def eff(T, D):                      # equal recipients: E = min_s T_s/D_s / (sum T / sum D)
    return float((T / D).min() * D.sum() / T.sum())

rows = []
for bname, items in BASKETS.items():
    for base in ["2019-01-01", "2023-05-01"]:
        D = need_index(items, base)
        dates = list(D.index)
        for i, t in enumerate(dates):
            if t <= pd.Timestamp(base): continue
            d = D.loc[t]
            r = dict(basket=bname, base=base[:7], date=t, max_min_ratio=d.max() / d.min(),
                     E_flat=eff(pd.Series(1.0, d.index), d),
                     E_zonal=eff(d.groupby(zone).transform("mean"), d))
            for lag in (1, 3, 6):
                r[f"E_carry{lag}"] = eff(D.iloc[i - lag], d) if i - lag >= 0 and dates[i - lag] > pd.Timestamp(base) else np.nan
            # adequacy: (a) fixed nominal budget; (b) budget indexed to national food CPI
            r["adequacy_fixed"] = 1.0 / d.mean()
            cpi = macro.cpi_food_index_b2024
            r["adequacy_cpi_food"] = (cpi.loc[t] / cpi.loc[pd.Timestamp(base)]) / d.mean()
            rows.append(r)
R = pd.DataFrame(rows)
R.to_csv(os.path.join(TAB, "gate1_efficiency_by_month.csv"), index=False)
summ = (R.assign(period=np.where(R.date >= "2023-06-01", "post-reform (Jun 2023-Apr 2024)", "pre-reform"))
         .groupby(["basket", "base", "period"])[["E_flat", "E_zonal", "E_carry1", "E_carry3", "E_carry6", "max_min_ratio"]]
         .agg(["mean", "min"]).round(3))
summ.to_csv(os.path.join(TAB, "gate1_summary.csv")); print(summ.to_string())
x = R[(R.basket == "equal43") & (R.base == "2023-05")].set_index("date")
print(x[["E_flat", "E_zonal", "E_carry1", "E_carry3", "adequacy_fixed", "adequacy_cpi_food"]].round(3).to_string())
# Figure: efficiency over time, base Jan 2019, equal-weight basket
y = R[(R.basket == "equal43") & (R.base == "2019-01")].set_index("date")
fig, ax = plt.subplots(figsize=(7.2, 3.6))
for c, lab in [("E_flat", "Flat transfer (status quo)"), ("E_zonal", "Zonal indexation"), ("E_carry3", "State prices, 3-month delay")]:
    ax.plot(y.index, y[c], label=lab, lw=1.6)
ax.axvspan(pd.Timestamp("2023-06-01"), y.index.max(), color="0.9", zorder=0, label="Post-reform")
ax.set_ylabel("Allocative efficiency E"); ax.set_ylim(0, 1.02); ax.legend(frameon=False, fontsize=8, loc="lower left")
ax.set_title("Share of attainable protection delivered, equal-weight basket, base Jan 2019", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_gate1_efficiency.pdf")); fig.savefig(os.path.join(FIG, "fig_gate1_efficiency.png"), dpi=200)
