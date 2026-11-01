"""Guarantee-constrained allocation: maximise nominal protection min_s T_s/Dhat_s subject to a worst-case
efficiency of at least kappa*G over the calibrated need box. Solution (water-filling):
   T_s = max(kappa*G*B*pimax_s, theta*Dhat_s), theta set by the budget.
Also certifies the guaranteed efficiency theta(T) = min_s T_s/(B pimax_s) of every rule."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from setup import Panel, ROOT
from e3_e4_backtest import zone_means, efficiency, prop
from e3b_need_space import need_box_pimax, WIN_PREM, WIN_RES, WIN_CAL, GAMMA

def waterfill(N, floor, Dhat, B=1.0):
    """smallest theta with sum N max(floor, theta*Dhat) = B (floor sums to <= B)."""
    lo, hi = 0.0, B / (N @ Dhat) * 10
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if N @ np.maximum(floor, mid * Dhat) > B: hi = mid
        else: lo = mid
    return np.maximum(floor, lo * Dhat)

def certified(N, T, pim):
    return float(np.min(T / ((N @ T) * pim)))

KAPPAS = (0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0)
def run(P, tag, a=0.20):
    dates = P.dates
    prices = np.array([P.prices(t) for t in dates]); Z = np.array([zone_means(P, p) for p in prices])
    X = np.log(prices / Z); D = np.array([P.need(p) for p in prices]); Dhat = np.full_like(D, np.nan)
    for k in range(WIN_PREM, len(dates)):
        Dhat[k] = P.need(Z[k] * np.exp(np.median(X[k - WIN_PREM:k], 0)))
    Rr = np.log(D / Dhat); cen = np.full_like(D, np.nan); sca = cen.copy(); score = np.full(len(dates), np.nan)
    k0 = WIN_PREM + WIN_RES
    for k in range(k0, len(dates)):
        h = Rr[k - WIN_RES:k]; c = np.median(h, 0); s = np.maximum(1.4826 * np.median(np.abs(h - c), 0), 0.005)
        cen[k], sca[k] = c, s; score[k] = np.max(np.abs(Rr[k] - c) / s)
    lev = a; rows = []
    for k in range(k0 + WIN_CAL, len(dates)):
        cal = score[k - WIN_CAL:k]
        q = np.quantile(cal, min(1.0, np.ceil((len(cal) + 1) * (1 - lev)) / len(cal)), method="higher") if lev > 1 / (len(cal) + 1) else 2 * cal.max()
        dc = Dhat[k] * np.exp(cen[k]); L, U = dc * np.exp(-q * sca[k]), dc * np.exp(q * sca[k])
        pim = need_box_pimax(P.N, L, U); G = 1 / (P.N @ pim)
        r = dict(weights=tag, date=dates[k], G=G, covered=bool(score[k] <= q))
        for name, T in [("flat", prop(P, np.ones(P.S))), ("zonal", prop(P, P.need(Z[k]))), ("plugin", prop(P, dc))]:
            r[f"E_{name}"] = efficiency(P, T, prices[k]); r[f"cert_{name}"] = certified(P.N, T, pim)
        for kap in KAPPAS:
            T = waterfill(P.N, kap * G * pim, dc)
            r[f"E_k{int(kap*100)}"] = efficiency(P, T, prices[k]); r[f"cert_k{int(kap*100)}"] = certified(P.N, T, pim)
        rows.append(r)
        lev += GAMMA * (a - float(score[k] > q))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    out = [run(Panel(**kw), tag) for tag, kw in [("poor_zonal", {}), ("poor_national", {"weights_scope": "national"}),
           ("all_zonal", {"weights_pop": "all"}), ("poor_no_other_grains", {"drop_group": "Grains (other)"})]]
    R = pd.concat(out); R.to_csv(os.path.join(ROOT, "results", "tables", "e3c_frontier_by_month.csv"), index=False)
    for tag in R.weights.unique():
        x = R[R.weights == tag]
        names = ["flat", "zonal", "plugin"] + [f"k{int(k*100)}" for k in KAPPAS]
        t = pd.DataFrame({"E_mean": [x[f"E_{n}"].mean() for n in names], "E_min": [x[f"E_{n}"].min() for n in names],
                          "E_p10": [x[f"E_{n}"].quantile(.1) for n in names],
                          "certified_mean": [x[f"cert_{n}"].mean() for n in names],
                          "certified_holds": [(x[f"E_{n}"] >= x[f"cert_{n}"] - 1e-12).mean() for n in names]}, index=names)
        print("==", tag, "months", len(x), "coverage", round(x.covered.mean(), 3), "G mean", round(x.G.mean(), 3)); print(t.round(3).to_string())
