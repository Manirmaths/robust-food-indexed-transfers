"""E8: live application (defaults: growth centre, no drift correction, 3-month premium window; see E8_* env vars)
, September 2023 to May 2026, when NBS publishes only national and zonal averages
(and state extremes). Last genuine state data: August 2023.

Plug-in need (default 'growth'): Dhat_st = D_s(last genuine month) x zonal-need growth since then, both on the
items published in month t. Alternative 'premium': zonal prices x frozen state premia (median of last WIN months).
Band widths are made non-decreasing in h (running maximum), since uncertainty cannot shrink as data age.
Staleness bands: for horizon h (months since the last state data), residuals log(D / Dhat_h) are computed on
history with premia frozen h months earlier; a simultaneous conformal quantile (level 1 - alpha) per h gives the
need box at horizon h. Then: G_t, the kappa = 0.9 allocation, and certified efficiency of flat and zonal rules.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from setup import Panel, ROOT, DATA
from e3_e4_backtest import zone_means
from e3b_need_space import need_box_pimax
from e3c_frontier import waterfill, certified

TAB = os.path.join(ROOT, "results", "tables")
ALPHA = 0.20
WIN = int(os.environ.get('E8_WIN', 3))
BIAS = os.environ.get('E8_BIAS', '0') == '1'   # per-state drift correction at horizon h (off: it extrapolates past episodes)
CENTRE = os.environ.get('E8_CENTRE', 'growth')   # 'premium': frozen state premia; 'growth': last state need x zonal-need growth


def predict(P, Zt, Ze, pe, m, avail):
    if CENTRE == 'premium':
        return need_with(P, Zt, m, avail)
    z0 = np.zeros(P.n)
    return need_with(P, pe, z0, avail) * need_with(P, Zt, z0, avail) / need_with(P, Ze, z0, avail)



def need_with(P, Zfull, m, avail):
    """Need from zonal prices Zfull (n,), premia m (n,), using only items flagged in avail (J,)."""
    S, J = P.S, P.J
    # renormalise weights: a_sj = w_j / p_sj0, so rescale each state's block so that sum_j w_j = 1 over available items
    D = np.zeros(S)
    for i in range(S):
        a = P.A[i, i * J:(i + 1) * J]
        w_tot = (a * P.P.loc[P.base].values[i * J:(i + 1) * J]).sum()        # = 1 for the full basket
        w_av = (a * avail * P.P.loc[P.base].values[i * J:(i + 1) * J]).sum()
        D[i] = (a * avail * Zfull[i * J:(i + 1) * J] * np.exp(m[i * J:(i + 1) * J])).sum() * w_tot / w_av
    return D


def main():
    P = Panel(); S, J = P.S, P.J; dates = P.dates
    prices = np.array([P.prices(t) for t in dates]); Z = np.array([zone_means(P, p) for p in prices])
    X = np.log(prices / Z); D = np.array([P.need(p) for p in prices])
    ones = np.ones(J)
    # ---- staleness calibration on history ----
    H = 33
    qh, sh = {}, {}
    for h in range(1, H + 1):
        res = []
        for k in range(len(dates)):
            last = [i for i in range(k) if dates[i] <= dates[k] - pd.DateOffset(months=h)]
            if not last: continue
            e = max(last)
            if e + 1 < WIN: continue
            m = np.median(X[e + 1 - WIN:e + 1], 0)
            res.append(np.log(D[k] / predict(P, Z[k], Z[e], prices[e], m, ones)))
        res = np.array(res)
        if len(res) < 8: break
        c = np.median(res, 0) if BIAS else np.zeros(res.shape[1])
        s = np.maximum(1.4826 * np.median(np.abs(res - np.median(res, 0)), 0), 0.005)
        sc = np.max(np.abs(res - c) / s, 1)
        n = len(sc)
        q = np.quantile(sc, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n), method="higher")
        qh[h] = (c, s, q, n)
    hmax = max(qh)
    width = np.zeros(S); cen_prev = None
    for h in range(1, hmax + 1):
        c, s_, q, n = qh[h]
        width = np.maximum(width, q * s_)
        qh[h] = (c, width / q, q, n)
    print('mean band half-width (log) at h=1,6,12,24,33:', [round(float(np.mean(qh[h][1] * qh[h][2])), 3) for h in (1, 6, 12, 24, min(33, hmax))])
    pd.DataFrame([dict(h=h, state=st, half_width=float(qh[h][1][i] * qh[h][2])) for h in qh for i, st in enumerate(P.states)]).to_csv(os.path.join(TAB, 'e8_band_halfwidth.csv'), index=False)
    # ---- live months ----
    z = pd.read_csv(os.path.join(DATA, "food_prices_national_zonal_2017_2026.csv"), parse_dates=["date"])
    z = z[(z.geo_level == "zone") & (z.date >= "2023-09-01")]
    zp = z.pivot_table(index="date", columns=["geo", "item_harmonized"], values="value")
    last = len(dates) - 1                                   # August 2023
    m = np.median(X[last + 1 - WIN:last + 1], 0)
    rows, shares = [], []
    for t in zp.index:
        h = (t.year - dates[last].year) * 12 + t.month - dates[last].month
        hh = min(h, hmax)
        c, s, q, n = qh[hh]
        Zfull = np.zeros(P.n); avail = np.zeros(J)
        for j, it in enumerate(P.items):
            ok = all((zz, it) in zp.columns and not np.isnan(zp.loc[t, (zz, it)]) for zz in P.zones)
            avail[j] = ok
            if ok:
                for i, st in enumerate(P.states):
                    Zfull[i * J + j] = zp.loc[t, (P.zone_of[st], it)]
        Dh = predict(P, Zfull, Z[last], prices[last], m, avail)
        ctr = Dh * np.exp(c); L, U = ctr * np.exp(-q * s), ctr * np.exp(q * s)
        pim = need_box_pimax(P.N, L, U); G = 1 / (P.N @ pim)
        T09 = waterfill(P.N, 0.9 * G * pim, ctr)
        Tflat = np.ones(S) / P.N.sum()
        Dz = need_with(P, Zfull, np.zeros(P.n), avail); Tz = Dz / (P.N @ Dz)
        rows.append(dict(date=t, horizon=h, horizon_used=hh, items_used=int(avail.sum()), G=G,
                         cert_k90=certified(P.N, T09, pim), cert_flat=certified(P.N, Tflat, pim),
                         cert_zonal=certified(P.N, Tz, pim), cert_plugin=certified(P.N, ctr / (P.N @ ctr), pim),
                         calib_months=n))
        for i, st in enumerate(P.states):
            shares.append(dict(date=t, state=st, zone=P.zone_of[st], T_k90_rel_flat=T09[i] / Tflat[i],
                               T_plugin_rel_flat=(ctr[i] / (P.N @ ctr)) / Tflat[i], need_centre=ctr[i]))
    R = pd.DataFrame(rows); Sh = pd.DataFrame(shares)
    R.to_csv(os.path.join(TAB, "e8_live_by_month.csv" if (CENTRE, BIAS, WIN) == ("growth", False, 3) else f"e8_live_by_month_{CENTRE}_bias{int(BIAS)}_win{WIN}.csv"), index=False); Sh.to_csv(os.path.join(TAB, "e8_live_state_allocations.csv" if (CENTRE, BIAS, WIN) == ("growth", False, 3) else f"e8_live_state_allocations_{CENTRE}_bias{int(BIAS)}_win{WIN}.csv"), index=False)
    print("horizons calibrated up to", hmax, "| calibration sample at h=1:", qh[1][3], "at hmax:", qh[hmax][3])
    print(R[["date", "horizon", "items_used", "G", "cert_k90", "cert_plugin", "cert_flat", "cert_zonal"]].iloc[[0, 3, 8, 15, 16, 24, -1]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    lt = Sh[Sh.date == Sh.date.max()].sort_values("T_k90_rel_flat", ascending=False)
    print("latest month top/bottom (payment relative to flat):")
    print(pd.concat([lt.head(5), lt.tail(5)])[["state", "zone", "T_k90_rel_flat"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
