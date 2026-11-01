"""E9: block-bootstrap intervals, sensitivity to calibration settings, and a run with published zonal averages.

Re-implements the need-box frontier of e3c_frontier.py with explicit parameters:
  prem, res, cal : window lengths (months) for premia, residual centre/scale and conformal calibration
  alpha, gamma   : target miss rate and adaptive step of ACI
  zpub           : if True, use published zonal averages instead of state means where NBS publishes all items
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from setup import Panel, ROOT, DATA
from e3_e4_backtest import zone_means, efficiency, prop
from e3b_need_space import need_box_pimax
from e3c_frontier import waterfill, certified

TAB = os.path.join(ROOT, "results", "tables")
RULES = ["flat", "zonal", "plugin", "k90", "k100"]


def published_zones(P):
    z = pd.read_csv(os.path.join(DATA, "food_prices_national_zonal_2017_2026.csv"), parse_dates=["date"])
    z = z[z.geo_level == "zone"].pivot_table(index="date", columns=["geo", "item_harmonized"], values="value")
    out = {}
    for t in P.dates:
        if t not in z.index: continue
        cols = [(P.zone_of[st], it) for st in P.states for it in P.items]
        if all(c in z.columns for c in cols):
            v = z.loc[t, cols].values.astype(float)
            if not np.isnan(v).any(): out[t] = v
    return out


def run(P, prem=12, res=12, cal=24, alpha=0.20, gamma=0.02, zpub=False):
    dates = P.dates
    prices = np.array([P.prices(t) for t in dates]); Z = np.array([zone_means(P, p) for p in prices])
    if zpub:
        for t, v in published_zones(P).items(): Z[dates.index(t)] = v
    X = np.log(prices / Z); D = np.array([P.need(p) for p in prices]); Dhat = np.full_like(D, np.nan)
    for k in range(prem, len(dates)):
        Dhat[k] = P.need(Z[k] * np.exp(np.median(X[k - prem:k], 0)))
    Rr = np.log(D / Dhat); cen = np.full_like(D, np.nan); sca = cen.copy(); score = np.full(len(dates), np.nan)
    k0 = prem + res
    for k in range(k0, len(dates)):
        h = Rr[k - res:k]; c = np.median(h, 0); s = np.maximum(1.4826 * np.median(np.abs(h - c), 0), 0.005)
        cen[k], sca[k] = c, s; score[k] = np.max(np.abs(Rr[k] - c) / s)
    lev = alpha; rows = []
    for k in range(k0 + cal, len(dates)):
        cl = score[k - cal:k]
        q = np.quantile(cl, min(1.0, np.ceil((len(cl) + 1) * (1 - lev)) / len(cl)), method="higher") if lev > 1 / (len(cl) + 1) else 2 * cl.max()
        dc = Dhat[k] * np.exp(cen[k]); L, U = dc * np.exp(-q * sca[k]), dc * np.exp(q * sca[k])
        pim = need_box_pimax(P.N, L, U); G = 1 / (P.N @ pim)
        r = dict(date=dates[k], G=G, covered=bool(score[k] <= q))
        Ts = {"flat": prop(P, np.ones(P.S)), "zonal": prop(P, P.need(Z[k])), "plugin": prop(P, dc),
              "k90": waterfill(P.N, 0.9 * G * pim, dc), "k100": waterfill(P.N, G * pim, dc)}
        for n, T in Ts.items():
            r[f"E_{n}"] = efficiency(P, T, prices[k]); r[f"cert_{n}"] = certified(P.N, T, pim)
        rows.append(r)
        lev += gamma * (alpha - float(score[k] > q))
    return pd.DataFrame(rows)


def block_ci(x, B=4000, block=3, seed=0, level=0.95):
    """Moving-block bootstrap percentile interval for the mean of a monthly series."""
    x = np.asarray(x, float); n = len(x); rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block)); starts = rng.integers(0, n - block + 1, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(B, -1)[:, :n]
    m = x[idx].mean(1); a = (1 - level) / 2
    return np.quantile(m, a), np.quantile(m, 1 - a)


if __name__ == "__main__":
    P = Panel()
    # ---- 1. bootstrap intervals for the frontier table (base settings, all kappa) ----
    F = pd.read_csv(os.path.join(TAB, "e3c_frontier_by_month.csv"), parse_dates=["date"]); F = F[F.weights == "poor_zonal"]
    rows = []
    for n in ["flat", "zonal", "plugin", "k50", "k80", "k85", "k90", "k95", "k100"]:
        for m in ("E", "cert"):
            lo, hi = block_ci(F[f"{m}_{n}"]); rows.append(dict(rule=n, metric=m, mean=F[f"{m}_{n}"].mean(), lo=lo, hi=hi))
    lo, hi = block_ci(F.covered.astype(float)); rows.append(dict(rule="box", metric="coverage", mean=F.covered.mean(), lo=lo, hi=hi))
    for a, b in (("k90", "flat"), ("plugin", "flat"), ("k90", "plugin")):
        for m in ("E", "cert"):
            d = F[f"{m}_{a}"] - F[f"{m}_{b}"]; lo, hi = block_ci(d)
            rows.append(dict(rule=f"{a}-{b}", metric=m, mean=d.mean(), lo=lo, hi=hi))
    CI = pd.DataFrame(rows); CI.to_csv(os.path.join(TAB, "e9_frontier_ci.csv"), index=False)
    print(CI.round(3).to_string(index=False))

    # ---- 2. sensitivity, evaluated on common months ----
    base = dict(prem=12, res=12, cal=24, alpha=0.20, gamma=0.02)
    variants = [("base", {}), ("alpha=0.10", dict(alpha=0.10)), ("alpha=0.30", dict(alpha=0.30)),
                ("gamma=0.005", dict(gamma=0.005)), ("gamma=0.05", dict(gamma=0.05)),
                ("windows 6/6/24", dict(prem=6, res=6)), ("windows 24/12/24", dict(prem=24)),
                ("windows 12/24/24", dict(res=24)), ("windows 12/12/36", dict(cal=36)),
                ("published zonal", dict(zpub=True))]
    out = {}
    for name, kw in variants:
        out[name] = run(P, **{**base, **kw}).set_index("date")
    common = sorted(set.intersection(*[set(v.index) for v in out.values()]))
    rows = []
    for name, R in out.items():
        x = R.loc[common]
        r = dict(variant=name, months=len(x), first=x.index.min().date(), G=x.G.mean(), coverage=x.covered.mean())
        for n in RULES: r[f"E_{n}"] = x[f"E_{n}"].mean(); r[f"cert_{n}"] = x[f"cert_{n}"].mean()
        r["holds_k90"] = (x.E_k90 >= x.cert_k90 - 1e-12).mean()
        rows.append(r)
    S = pd.DataFrame(rows); S.to_csv(os.path.join(TAB, "e9_sensitivity.csv"), index=False)
    print("common months:", len(common), common[0].date(), "to", common[-1].date())
    print(S.round(3).to_string(index=False))

    # ---- 3. published zonal averages: months where they replace state means ----
    pz = sorted(published_zones(P))
    b, z = out["base"], out["published zonal"]
    m = [t for t in pz if t in b.index]
    cmp = pd.DataFrame({f"{c}_{lab}": R.loc[m, c] for lab, R in (("statemean", b), ("published", z)) for c in ["E_zonal", "E_plugin", "E_k90", "cert_k90", "G"]})
    cmp.to_csv(os.path.join(TAB, "e9_published_zonal_months.csv"))
    print("published zonal months:", [t.date() for t in m]); print(cmp.round(3).to_string())
