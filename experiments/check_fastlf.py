import sys, time, numpy as np
sys.path.insert(0, "src")
from setup import Panel
from fastlf import pi_max_fast
from robust import pi_max as pi_max_lp
from scipy.sparse import lil_matrix
P = Panel()
print("states", P.S, "items", P.J, "filled cells", P.n_filled, "months", len(P.dates))
t = P.dates[60]; p = P.prices(t)
l, u = p * 0.7, p * 1.4     # arbitrary test bands containing the truth
for level, extremes in [("zone", False), ("zone", True), ("national", False)]:
    G = P.polytope(p, l, u, level=level, eps=0.01, extremes=extremes, observed_states=(3,))
    t0 = time.time(); pf, argf, it = pi_max_fast(P.A, P.N, G); tf = time.time() - t0
    # same polytope for the LP: bounds per cell + two rows per group
    n = P.n; lb = np.zeros(n); ub = np.zeros(n)
    idx = G.index; mask = G.mask
    lb[idx[mask]] = G.l[mask]; ub[idx[mask]] = G.u[mask]
    R = lil_matrix((2 * len(idx), n)); r = []
    for g in range(len(idx)):
        cells = idx[g][idx[g] >= 0]
        R[2 * g, cells] = 1.0; R[2 * g + 1, cells] = -1.0
        r += [G.hi_sum[g], -G.lo_sum[g]]
    sel = [0, 5, 17, 24, 36]
    t0 = time.time(); pl = pi_max_lp(P.A[:, :].reshape(P.S, -1)[:, :].reshape(P.S, n), P.N, lb, ub, R.tocsr(), np.array(r)) if False else None
    # robust.pi_max expects a as (S,J) blocks; call a generic LP directly instead
    from scipy.optimize import linprog
    import scipy.sparse as sp
    I = sp.identity(n, format="csr")
    A_ub = sp.vstack([sp.hstack([I, -sp.csr_matrix(ub[:, None])]), sp.hstack([-I, sp.csr_matrix(lb[:, None])]),
                      sp.hstack([R.tocsr(), -sp.csr_matrix(np.array(r)[:, None])])]).tocsr()
    w = P.N @ P.A
    A_eq = sp.csr_matrix(np.r_[w, 0.0][None])
    errs = []
    t0 = time.time()
    for s in sel:
        res = linprog(np.r_[-P.A[s], 0.0], A_ub=A_ub, b_ub=np.zeros(A_ub.shape[0]), A_eq=A_eq, b_eq=[1.0],
                      bounds=[(0, None)] * (n + 1), method="highs")
        errs.append(abs(-res.fun - pf[s]) / pf[s])
    tl = (time.time() - t0) / len(sel) * P.S
    print(f"{level:8s} extremes={extremes!s:5s} max rel err vs LP {max(errs):.2e}  dinkelbach iters {it}  fast {tf:.3f}s  LP (all states, est.) {tl:.1f}s")
