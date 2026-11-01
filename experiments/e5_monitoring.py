"""E5-E7: which states to monitor. Monitoring state s reveals its need D_s (with delay d, its need d months
earlier, rolled forward with the zonal-average growth and a calibrated residual band).

Objective (chosen on training months only): Phi(M) = mean_t G(U_t(M)), G = 1 / sum_s N_s pimax_s.
Greedy forward selection (lazy), exhaustive check for K <= 3, and an accuracy-focused rival that picks
states by their recipient-weighted squared forecast error. Evaluation on later months: guarantee G and the
realised efficiency of the kappa = 0.9 guarantee-constrained rule.
"""
import os, sys, itertools, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from setup import Panel, ROOT
from e3_e4_backtest import zone_means, efficiency, prop
from e3b_need_space import need_box_pimax, WIN_PREM, WIN_RES, WIN_CAL, GAMMA
from e3c_frontier import waterfill, certified

TAB, FIG = os.path.join(ROOT, "results", "tables"), os.path.join(ROOT, "results", "figures")


def calibrated_boxes(P, a=0.20, delays=(0, 1, 3)):
    """Per decision month: true needs D, plug-in centre dc, unmonitored box (L, U), and for each delay d
    the box a monitored state would have (Lm[d], Um[d])."""
    dates = P.dates
    prices = np.array([P.prices(t) for t in dates]); Z = np.array([zone_means(P, p) for p in prices])
    X = np.log(prices / Z); D = np.array([P.need(p) for p in prices]); DZ = np.array([P.need(z) for z in Z])
    Dhat = np.full_like(D, np.nan)
    for k in range(WIN_PREM, len(dates)):
        Dhat[k] = P.need(Z[k] * np.exp(np.median(X[k - WIN_PREM:k], 0)))

    def lagged(k, d):
        if d == 0: return k
        c = [i for i in range(k) if dates[i] <= dates[k] - pd.DateOffset(months=d)]
        return max(c) if c else None

    def box_series(pred):
        """Calibrate a simultaneous conformal box (ACI, level a) around predictions pred (T,S)."""
        res = np.log(D / pred); out = {}
        cen = np.full_like(D, np.nan); sca = cen.copy(); sc = np.full(len(dates), np.nan)
        k0 = WIN_PREM + WIN_RES + 3
        for k in range(k0, len(dates)):
            h = res[k - WIN_RES:k]
            if np.isnan(h).any(): continue
            c = np.median(h, 0); s = np.maximum(1.4826 * np.median(np.abs(h - c), 0), 0.002)
            cen[k], sca[k] = c, s; sc[k] = np.max(np.abs(res[k] - c) / s)
        lev = a
        for k in range(k0 + WIN_CAL, len(dates)):
            cal = sc[k - WIN_CAL:k]; cal = cal[~np.isnan(cal)]
            q = np.quantile(cal, min(1.0, np.ceil((len(cal) + 1) * (1 - lev)) / len(cal)), method="higher") if lev > 1 / (len(cal) + 1) else 2 * cal.max()
            ctr = pred[k] * np.exp(cen[k])
            out[k] = (ctr, ctr * np.exp(-q * sca[k]), ctr * np.exp(q * sca[k]), bool(sc[k] <= q))
            lev += GAMMA * (a - float(sc[k] > q))
        return out

    base = box_series(Dhat)
    mon = {}
    for d in delays:
        if d == 0:
            mon[d] = {k: (D[k], D[k], D[k], True) for k in base}
        else:
            pred = np.full_like(D, np.nan)
            for k in range(len(dates)):
                j = lagged(k, d)
                if j is not None: pred[k] = D[j] * DZ[k] / DZ[j]
            mon[d] = box_series(pred)
    ks = sorted(set(base) & set.intersection(*[set(v) for v in mon.values()]))
    return dict(dates=dates, prices=prices, D=D, base=base, mon=mon, ks=ks)


def G_of(P, B, k, M, d):
    ctr, L, U, _ = B["base"][k]
    L, U, ctr = L.copy(), U.copy(), ctr.copy()
    if M:
        mc, mL, mU, _ = B["mon"][d][k]
        idx = list(M); L[idx], U[idx], ctr[idx] = mL[idx], mU[idx], mc[idx]
    pim = need_box_pimax(P.N, L, U)
    return 1.0 / (P.N @ pim), pim, ctr


def phi(P, B, months, M, d):
    return float(np.mean([G_of(P, B, k, M, d)[0] for k in months]))


def greedy(P, B, months, K, d, score=None):
    M, path = [], []
    for _ in range(K):
        best, bv = None, -np.inf
        for s in range(P.S):
            if s in M: continue
            v = phi(P, B, months, M + [s], d) if score is None else score[s]
            if v > bv: best, bv = s, v
        M.append(best); path.append(list(M))
    return path


if __name__ == "__main__":
    P = Panel()
    B = calibrated_boxes(P)
    ks = B["ks"]; dates = B["dates"]
    train = [k for k in ks if dates[k] < pd.Timestamp("2022-07-01")]
    test = [k for k in ks if dates[k] >= pd.Timestamp("2022-07-01")]
    print("train", dates[train[0]].date(), "to", dates[train[-1]].date(), len(train), "| test", dates[test[0]].date(), "to", dates[test[-1]].date(), len(test))
    rows = []
    # accuracy-focused rival: recipient-weighted squared log forecast error on training months
    err = np.zeros(P.S)
    for k in train:
        ctr = B["base"][k][0]; err += P.N * np.log(B["D"][k] / ctr) ** 2
    acc_path = greedy(P, B, train, 15, 0, score=err)
    for d in (0, 1, 3):
        path = greedy(P, B, train, 15, d)
        for K in range(0, 16):
            for name, M in (("decision", path[K - 1] if K else []), ("accuracy", acc_path[K - 1] if K else [])):
                if name == "accuracy" and d != 0: continue
                for split, months in (("train", train), ("test", test)):
                    Gs, Es = [], []
                    for k in months:
                        G, pim, ctr = G_of(P, B, k, M, d)
                        T = waterfill(P.N, 0.9 * G * pim, ctr)
                        Gs.append(G); Es.append(efficiency(P, T, B["prices"][k]))
                    rows.append(dict(delay=d, design=name, K=K, split=split, G_mean=np.mean(Gs), E_mean=np.mean(Es), E_min=np.min(Es),
                                     states="; ".join(P.states[i] for i in M)))
    R = pd.DataFrame(rows); R.to_csv(os.path.join(TAB, "e5_monitoring_curve.csv"), index=False)
    t = R[(R.split == "test")]
    print(t[(t.design == "decision") & (t.delay == 0) & (t.K.isin([0, 1, 2, 3, 5, 8, 10, 15]))][["K", "G_mean", "E_mean", "E_min", "states"]].round(3).to_string(index=False))
    print(t[(t.K.isin([3, 5, 10]))].pivot_table(index=["design", "delay"], columns="K", values="G_mean").round(3).to_string())
    print(t[(t.K.isin([3, 5, 10]))].pivot_table(index=["design", "delay"], columns="K", values="E_mean").round(3).to_string())
    # E7: greedy vs exhaustive for K <= 3, d = 0, on training objective
    ex = []
    path0 = greedy(P, B, train, 3, 0)
    for K in (1, 2, 3):
        best = max(itertools.combinations(range(P.S), K), key=lambda M: phi(P, B, train, list(M), 0))
        ex.append(dict(K=K, greedy=phi(P, B, train, path0[K - 1], 0), exhaustive=phi(P, B, train, list(best), 0),
                       greedy_states="; ".join(P.states[i] for i in path0[K - 1]), best_states="; ".join(P.states[i] for i in best)))
    E = pd.DataFrame(ex); E["gap_pct"] = 100 * (E.exhaustive - E.greedy) / E.exhaustive
    E.to_csv(os.path.join(TAB, "e7_greedy_vs_exhaustive.csv"), index=False); print(E.round(4).to_string(index=False))
