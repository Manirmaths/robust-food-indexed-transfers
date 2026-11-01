"""Alternative need definition: the same real basket everywhere, D_s = q' p_s (absolute cost level)."""
import os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DER, TAB = os.path.join(ROOT, "data", "derived"), os.path.join(ROOT, "results", "tables")
s = pd.read_csv(os.path.join(DER, "food_state_clean.csv"), parse_dates=["date"])
zone = s.drop_duplicates("state").set_index("state").zone
P = s.pivot_table(index=["date", "state"], columns="item", values="price")
IW = pd.read_csv(os.path.join(DER, "basket_item_weights.csv"))
w = IW[(IW.population == "poor") & (IW.scope == "National")].set_index("item").weight
N = pd.read_csv(os.path.join(DER, "state_recipients.csv")).set_index("state").poor_population_2019
def eff(T, D, N): return float((T / D).min() * (N * D).sum() / (N * T).sum())
rows = []
for base in ["2019-01-01", "2023-05-01"]:
    pbar = P.xs(pd.Timestamp(base), level="date")[w.index].median()     # national base prices
    q = w / pbar                                                         # common quantities
    C = (P[w.index].fillna(P[w.index].groupby(level="date").transform("median")) * q).sum(axis=1).unstack("state")
    dates = list(C.index)
    for i, t in enumerate(dates):
        if t < pd.Timestamp("2018-01-01"): continue
        d = C.loc[t]; Nn = N.reindex(d.index)
        Zm = (d * Nn).groupby(zone).transform("sum") / Nn.groupby(zone).transform("sum")
        r = dict(base=base[:7], date=t, E_flat=eff(pd.Series(1.0, d.index), d, Nn), E_zonal=eff(Zm, d, Nn),
                 E_carry3=eff(C.iloc[i - 3], d, Nn) if i >= 3 else np.nan, max_min=d.max() / d.min(),
                 most_expensive=d.idxmax(), cheapest=d.idxmin())
        rows.append(r)
A = pd.DataFrame(rows).drop_duplicates("date")      # level definition does not depend on base except via q
A.to_csv(os.path.join(TAB, "gate1_absolute_need.csv"), index=False)
A["period"] = np.where(A.date >= "2023-06-01", "post", "pre")
print(A.groupby("period")[["E_flat", "E_zonal", "E_carry3", "max_min"]].agg(["mean", "min", "max"]).round(3).to_string())
print(A[A.date >= "2023-05-01"][["date", "E_flat", "E_zonal", "E_carry3", "max_min", "most_expensive", "cheapest"]].round(3).to_string(index=False))
