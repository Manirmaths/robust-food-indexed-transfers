"""Robust allocation with uncertainty on state needs (37 numbers) instead of prices (1,554 cells).

Need estimate: Dhat_s = D_s(p_hat), p_hat = zonal average x exp(usual state premium) (plug-in).
Residual r_st = log(D_s(p_t) / Dhat_st). Centre/scale from a rolling window; simultaneous conformal
score over states. Box U_D = {L_s <= D_s <= U_s}; Theorem 1 then gives the closed form
   pi_s^max = U_s / (N_s U_s + sum_{k != s} N_k L_k).
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from setup import Panel, ROOT
from e3_e4_backtest import zone_means, efficiency, prop

OUT = os.path.join(ROOT, "results", "tables")
WIN_PREM, WIN_RES, WIN_CAL = 12, 12, 24
GAMMA = 0.02


def need_box_pimax(N, L, U):
    tot_L = N @ L
    return U / (N * U + (tot_L - N * L))


def run(P, tag, alphas=(0.05, 0.10, 0.20, 0.30)):
    dates = P.dates
    prices = np.array([P.prices(t) for t in dates])
    Z = np.array([zone_means(P, p) for p in prices])
    X = np.log(prices / Z)
    D = np.array([P.need(p) for p in prices])                       # true needs (T, S)
    Dhat = np.full_like(D, np.nan)
    for k in range(WIN_PREM, len(dates)):
        m = np.median(X[k - WIN_PREM:k], 0)
        Dhat[k] = P.need(Z[k] * np.exp(m))
    Rres = np.log(D / Dhat)                                          # (T, S) residuals
    cen = np.full_like(D, np.nan); sca = np.full_like(D, np.nan); score = np.full(len(dates), np.nan)
    k0 = WIN_PREM + WIN_RES
    for k in range(k0, len(dates)):
        h = Rres[k - WIN_RES:k]
        c = np.median(h, 0); s = np.maximum(1.4826 * np.median(np.abs(h - c), 0), 0.005)
        cen[k], sca[k] = c, s
        score[k] = np.max(np.abs(Rres[k] - c) / s)
    first = k0 + WIN_CAL
    aci = {a: a for a in alphas}
    rows = []
    for k in range(first, len(dates)):
        t, d = dates[k], D[k]
        cal = score[k - WIN_CAL:k]
        dcen = Dhat[k] * np.exp(cen[k])
        r = dict(weights=tag, date=t, score=score[k],
                 E_flat=efficiency(P, prop(P, np.ones(P.S)), prices[k]),
                 E_zonal=efficiency(P, prop(P, P.need(Z[k])), prices[k]),
                 E_plugin=efficiency(P, prop(P, Dhat[k]), prices[k]),
                 E_plugin_bc=efficiency(P, prop(P, dcen), prices[k]),
                 rho_star_fixed=1.0 / ((P.N @ d) / P.N.sum()))
        lag3 = max(i for i in range(k) if dates[i] <= t - pd.DateOffset(months=3))
        r["E_carry3"] = efficiency(P, prop(P, D[lag3]), prices[k])
        for a in alphas:
            for mode in ("split", "aci"):
                lev = a if mode == "split" else aci[a]
                if lev <= 1.0 / (len(cal) + 1):
                    q = 2 * cal.max()
                else:
                    q = np.quantile(cal, min(1.0, np.ceil((len(cal) + 1) * (1 - lev)) / len(cal)), method="higher")
                L, U = dcen * np.exp(-q * sca[k]), dcen * np.exp(q * sca[k])
                pim = need_box_pimax(P.N, L, U)
                T = pim / (P.N @ pim)
                key = f"{mode}_a{int(round(a * 100))}"
                r[f"E_rob_{key}"] = efficiency(P, T, prices[k])
                r[f"G_{key}"] = 1.0 / (P.N @ pim)
                r[f"cov_{key}"] = bool(score[k] <= q)
                if mode == "aci":
                    aci[a] += GAMMA * (a - float(score[k] > q))
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = []
    for tag, kw in [("poor_zonal", {}), ("poor_national", {"weights_scope": "national"}),
                    ("all_zonal", {"weights_pop": "all"}), ("poor_no_other_grains", {"drop_group": "Grains (other)"})]:
        out.append(run(Panel(**kw), tag))
    R = pd.concat(out); R.to_csv(os.path.join(OUT, "e3b_need_space_by_month.csv"), index=False)
    x = R[R.weights == "poor_zonal"]
    cols = [c for c in x.columns if c.startswith("E_")]
    s = x[cols].agg(["mean", "min"]).T
    s["p10"] = x[cols].quantile(0.10)
    s["months_beat_plugin"] = [(x[c] > x.E_plugin + 1e-12).sum() for c in cols]
    print("months", len(x), x.date.min().date(), "to", x.date.max().date())
    print(s.round(3).to_string())
    g = [c for c in x.columns if c.startswith("G_")]; cv = [c for c in x.columns if c.startswith("cov_")]
    print(pd.DataFrame({"G_mean": x[g].mean().values, "G_min": x[g].min().values, "coverage": x[cv].mean().values,
                        "realised>=G": [(x["E_rob_" + c[2:]] >= x[c] - 1e-12).mean() for c in g]}, index=[c[2:] for c in g]).round(3).to_string())
