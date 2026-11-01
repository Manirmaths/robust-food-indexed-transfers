"""E3 (rule comparison), E4 (coverage of guarantees) and E10 (adequacy vs efficiency) on genuine state data.

Decision month t: the planner sees published zonal averages for month t (and, in I2, the highest and
lowest state per item), plus the full state panel up to the previous genuine month (history for bands).
Bands are on relative prices x_sjt = log(p_sjt / Zbar_{z(s) j t}) with rolling median/MAD centres and a
simultaneous conformal quantile; ACI adapts the level month by month.
"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from setup import Panel, ROOT
from fastlf import pi_max_fast, GroupedPolytope

OUT = os.path.join(ROOT, "results", "tables")
WIN_CENTRE, WIN_CAL = 12, 24
ALPHAS = (0.10, 0.20)
GAMMA = 0.02


def zone_means(P, p):
    S, J = P.S, P.J
    pm = p.reshape(S, J)
    Z = np.zeros_like(pm)
    for z in P.zones:
        rows = [i for i, st in enumerate(P.states) if P.zone_of[st] == z]
        Z[rows] = pm[rows].mean(0)
    return Z.ravel()


def efficiency(P, T, p):
    D = P.need(p)
    return float((T / D).min() * (P.N @ D) / (P.N @ T))


def prop(P, d):
    d = np.asarray(d, float)
    return d / (P.N @ d)          # allocation with budget 1


def fit_polytope(P, p_true, l, u, level, extremes, observed=()):
    G = P.polytope(p_true, l, u, level=level, eps=0.0, extremes=extremes, observed_states=observed)
    return G, P.relaxed


def run(P, tag):
    dates = P.dates
    X = np.array([np.log(P.prices(t) / zone_means(P, P.prices(t))) for t in dates])     # (T, n) relative prices
    rows, scores = [], {}
    aci = {a: a for a in ALPHAS}
    first = WIN_CENTRE + WIN_CAL
    # scores for every month that has a centre window
    centre, scale = {}, {}
    for k in range(WIN_CENTRE, len(dates)):
        hist = X[k - WIN_CENTRE:k]
        m = np.median(hist, 0); sd = 1.4826 * np.median(np.abs(hist - m), 0)
        sd = np.maximum(sd, 0.02)
        centre[k], scale[k] = m, sd
        scores[k] = float(np.max(np.abs(X[k] - m) / sd))
    for k in range(first, len(dates)):
        t = dates[k]; p = P.prices(t); Zt = zone_means(P, p)
        cal = np.array([scores[i] for i in range(k - WIN_CAL, k)])
        m, sd = centre[k], scale[k]
        D = P.need(p); T_or = prop(P, D)
        lag3 = max(i for i in range(k) if dates[i] <= t - pd.DateOffset(months=3))
        r = dict(weights=tag, date=t, score=scores[k], E_oracle=1.0,
                 rho_star_cpi=None,
                 E_flat=efficiency(P, prop(P, np.ones(P.S)), p),
                 E_zonal=efficiency(P, prop(P, P.need(Zt)), p),
                 E_carry3=efficiency(P, prop(P, P.need(P.prices(dates[lag3]))), p),
                 E_plugin=efficiency(P, prop(P, P.need(Zt * np.exp(m))), p))
        for a in ALPHAS:
            for mode in ("split", "aci"):
                level = a if mode == "split" else aci[a]
                qq = np.quantile(cal, min(1.0, np.ceil((len(cal) + 1) * (1 - level)) / len(cal)), method="higher") if level > 1 / (len(cal) + 1) else cal.max() * 2
                l = Zt * np.exp(m - qq * sd); u = Zt * np.exp(m + qq * sd)
                covered = bool(scores[k] <= qq)
                for info, extremes in (("I1", False), ("I2", True)):
                    G, fixes = fit_polytope(P, p, l, u, "zone", extremes)
                    pim, _, it = pi_max_fast(P.A, P.N, G)
                    Tr = pim / (P.N @ pim); g = 1.0 / (P.N @ pim)
                    key = f"{info}_{mode}_a{int(a*100)}"
                    r[f"E_rob_{key}"] = efficiency(P, Tr, p)
                    r[f"G_{key}"] = g
                    r[f"cov_{key}"] = covered
                    r[f"fix_{key}"] = fixes
                    r[f"q_{key}"] = qq
            # ACI update after observing coverage (uses split coverage indicator at the ACI level)
            ql = aci[a]
            qq = np.quantile(cal, min(1.0, np.ceil((len(cal) + 1) * (1 - ql)) / len(cal)), method="higher") if ql > 1 / (len(cal) + 1) else cal.max() * 2
            err = float(scores[k] > qq)
            aci[a] = aci[a] + GAMMA * (a - err)
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = []
    for tag, kw in [("poor_zonal", {}), ("poor_national", {"weights_scope": "national"}),
                    ("all_zonal", {"weights_pop": "all"}), ("poor_no_other_grains", {"drop_group": "Grains (other)"})]:
        P = Panel(**kw)
        df = run(P, tag); out.append(df)
        print(tag, "months", len(df), df.date.min().date(), "to", df.date.max().date())
    R = pd.concat(out); R.to_csv(os.path.join(OUT, "e3_e4_backtest_by_month.csv"), index=False)
