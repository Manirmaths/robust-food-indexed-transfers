"""E10: computation of the worst-case need shares. HiGHS LP (Charnes-Cooper, Prop. 2) versus sorting + Dinkelbach
(Prop. 5) on grouped price polytopes, and the closed form for need boxes (Prop. 6)."""
import os, sys, time, numpy as np, pandas as pd, scipy.sparse as sp
from scipy.optimize import linprog
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from setup import Panel, ROOT
from fastlf import pi_max_fast
from e3b_need_space import need_box_pimax

TAB = os.path.join(ROOT, "results", "tables")


def lp_all(P, G):
    n = P.n; lb = np.zeros(n); ub = np.zeros(n); idx, mask = G.index, G.mask
    lb[idx[mask]] = G.l[mask]; ub[idx[mask]] = G.u[mask]
    rows, cols, vals, r = [], [], [], []
    for g in range(len(idx)):
        cells = idx[g][idx[g] >= 0]
        rows += [2 * g] * len(cells) + [2 * g + 1] * len(cells); cols += list(cells) * 2
        vals += [1.0] * len(cells) + [-1.0] * len(cells); r += [G.hi_sum[g], -G.lo_sum[g]]
    R = sp.csr_matrix((vals, (rows, cols)), shape=(2 * len(idx), n)); r = np.array(r)
    I = sp.identity(n, format="csr")
    A_ub = sp.vstack([sp.hstack([I, -sp.csr_matrix(ub[:, None])]), sp.hstack([-I, sp.csr_matrix(lb[:, None])]),
                      sp.hstack([R, -sp.csr_matrix(r[:, None])])]).tocsr()
    A_eq = sp.csr_matrix(np.r_[P.N @ P.A, 0.0][None])
    out = np.zeros(P.S); pts = []; t0 = time.perf_counter()
    for s in range(P.S):
        res = linprog(np.r_[-P.A[s], 0.0], A_ub=A_ub, b_ub=np.zeros(A_ub.shape[0]), A_eq=A_eq, b_eq=[1.0],
                      bounds=[(0, None)] * (n + 1), method="highs")
        out[s] = -res.fun; pts.append(res.x[:-1] / res.x[-1])
    return out, time.perf_counter() - t0, (lb, ub, R, r), pts


def violation(x, lb, ub, R, r):
    return max(np.max(lb - x, initial=0), np.max(x - ub, initial=0), np.max(R @ x - r, initial=0))


if __name__ == "__main__":
    P = Panel(); rows = []
    months = P.dates[36::6]                         # 2020-01 onwards, every six months
    for t in months:
        p = P.prices(t); l, u = p * 0.7, p * 1.4
        for name, level, ext in (("Zonal averages (I1)", "zone", False), ("Zonal averages and extremes (I2)", "zone", True),
                                 ("National averages", "national", False)):
            G = P.polytope(p, l, u, level=level, eps=0.0, extremes=ext)
            t0 = time.perf_counter(); pf, x, it = pi_max_fast(P.A, P.N, G); tf = time.perf_counter() - t0
            pl, tl, (lb, ub, R, r), pts = lp_all(P, G); scale = ub.max()
            v = max(violation(x[s], lb, ub, R, r) for s in range(P.S)) / scale; vl = max(violation(q, lb, ub, R, r) for q in pts) / scale
            rows.append(dict(date=t, info=name, cells=P.n, groups=len(G.index), t_dinkelbach=tf, t_lp=tl, iters=it,
                             rel_diff=np.max(np.abs(pf - pl) / pl), dk_ge_lp=bool(np.all(pf >= pl * (1 - 1e-9))), viol=v, viol_lp=vl))
        L = P.need(p) * 0.8; U = P.need(p) * 1.25
        t0 = time.perf_counter()
        for _ in range(1000): need_box_pimax(P.N, L, U)
        rows.append(dict(date=t, info="Need box", cells=P.S, groups=0, t_dinkelbach=(time.perf_counter() - t0) / 1000, t_lp=np.nan, iters=0, rel_diff=np.nan, viol=0.0, viol_lp=np.nan))
    R_ = pd.DataFrame(rows); R_.to_csv(os.path.join(TAB, "e10_computation_by_month.csv"), index=False)
    S = R_.groupby("info", sort=False).agg(months=("date", "size"), groups=("groups", "first"), t_dinkelbach=("t_dinkelbach", "mean"),
                                           t_lp=("t_lp", "mean"), iters_mean=("iters", "mean"), iters_max=("iters", "max"),
                                           rel_diff_max=("rel_diff", "max"), viol_max=("viol", "max"), viol_lp_max=("viol_lp", "max"))
    S["speedup"] = S.t_lp / S.t_dinkelbach
    S.to_csv(os.path.join(TAB, "e10_computation.csv")); print(S.to_string(float_format=lambda v: f"{v:.3g}"))
