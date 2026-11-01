"""Steps 1.1, 1.2 and 1.6 of the blueprint: freeze, clean and audit the state food panel."""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "..", "NBS_Research_Datasets_2026-11-01", "data")
OUT = os.path.join(ROOT, "data", "derived"); TAB = os.path.join(ROOT, "results", "tables")

s = pd.read_csv(os.path.join(DATA, "food_prices_state_month_2017_2024.csv"), parse_dates=["date"])
s = s[s.state_data_genuine]   # drop zonal averages copied to states (Jun 2023, Sep 2023-Apr 2024)
# 1.1 use the latest NBS vintage where NBS revised a cell
rev = s.nbs_check.eq("differs_from_latest_vintage")
s["price"] = np.where(rev, s.nbs_elibrary_latest_vintage_value, s.price_ngn)
s = s[["date", "state", "zone", "item", "price"]].sort_values(["item", "state", "date"]).reset_index(drop=True)

# national average per item-month (unweighted mean of states, as NBS publishes)
s["A"] = s.groupby(["item", "date"]).price.transform("mean")
s["x"] = np.log(s.price / s.A)            # relative price
g = s.groupby(["item", "state"])
s["dx_prev"] = s.x - g.x.shift(1)
s["dx_next"] = g.x.shift(-1) - s.x
# 1.2 transient spike: relative price jumps by more than 1 log point and reverses by at least half next month
spike = (s.dx_prev.abs() > 1.0) & (np.sign(s.dx_next) == -np.sign(s.dx_prev)) & (s.dx_next.abs() > 0.5 * s.dx_prev.abs())
x_fix = 0.5 * (g.x.shift(1) + g.x.shift(-1))
log = s.loc[spike, ["date", "state", "item", "price", "A"]].copy()
log["replacement"] = np.exp(x_fix[spike]) * s.loc[spike, "A"]
log["jump_log_points"] = s.loc[spike, "dx_prev"].round(3)
s.loc[spike, "price"] = log["replacement"].values
s["outlier_replaced"] = spike
# recompute national averages after cleaning
s["A"] = s.groupby(["item", "date"]).price.transform("mean")
s[["date", "state", "zone", "item", "price", "outlier_replaced"]].to_csv(os.path.join(OUT, "food_state_clean.csv"), index=False)
log.to_csv(os.path.join(TAB, "outlier_log.csv"), index=False)
print("cells", len(s), "revised cells used", int(rev.sum()), "spikes replaced", int(spike.sum()))
print(log.groupby(log.date.dt.year).size().to_dict())

# 1.6 aggregation audit against published national and zonal averages (raw panel, as NBS computed them)
raw = pd.read_csv(os.path.join(DATA, "food_prices_state_month_2017_2024.csv"), parse_dates=["date"])
raw = raw[raw.state_data_genuine]
pub = pd.read_csv(os.path.join(DATA, "food_prices_national_zonal_2017_2026.csv"), parse_dates=["date"])
pub = pub[pub.date <= "2024-04-01"]
rows = []
for level, keys in [("national", ["item", "date"]), ("zone", ["zone", "item", "date"])]:
    m = raw.groupby(keys).price_ngn.mean().rename("state_mean").reset_index()
    p = pub[pub.geo_level == level].rename(columns={"item_harmonized": "item"})
    if level == "zone": p = p.rename(columns={"geo": "zone"})
    j = p.merge(m, on=keys)
    j["gap_pct"] = 100 * (j.value / j.state_mean - 1)
    j["level"] = level
    rows.append(j[["level", "date", "item", "value", "state_mean", "gap_pct"] + (["zone"] if level == "zone" else [])])
A = pd.concat(rows)
A.to_csv(os.path.join(TAB, "aggregation_audit_cells.csv"), index=False)
summ = A.groupby("level").gap_pct.agg(cells="size", exact_share=lambda v: (v.abs() < 0.01).mean(),
        within_1pct=lambda v: (v.abs() < 1).mean(), p95_abs=lambda v: v.abs().quantile(.95),
        p99_abs=lambda v: v.abs().quantile(.99))
print(summ.round(3).to_string())
z = A[A.level == "zone"]
print("zone exact share by month:\n", z.groupby(z.date.dt.strftime("%Y-%m")).gap_pct.apply(lambda v: round((v.abs() < 0.01).mean(), 2)).to_string())
n = A[A.level == "national"]
print("national exact share by year:", n.groupby(n.date.dt.year).gap_pct.apply(lambda v: round((v.abs() < 0.01).mean(), 2)).to_dict())
summ.to_csv(os.path.join(TAB, "aggregation_audit_summary.csv"))
