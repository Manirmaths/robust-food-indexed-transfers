"""Fast exact computation of worst-case need shares pi_s^max when the price
polytope is built from cell bounds and group-average constraints.

Structure exploited (Proposition 5 in the paper):
  U = { p : l <= p <= u,  (1-eps) A_g <= mean_{c in g} p_c <= (1+eps) A_g  for every group g },
where the groups (zone x item, or nation x item) partition the price cells.
For fixed lambda, max_{p in U} (a_s - lambda * w)' p separates into one
continuous knapsack per group, solved exactly by sorting. Dinkelbach's method
on lambda then converges to pi_s^max = max_{p in U} a_s'p / w'p in finitely many steps.

All functions are vectorised over the target state s.
"""
from __future__ import annotations
import numpy as np


def group_knapsack(c, l, u, lo_sum, hi_sum):
    """Maximise sum(c*x) s.t. l <= x <= u, lo_sum <= sum(x) <= hi_sum, along the last axis.

    c, l, u: arrays (..., m); lo_sum, hi_sum: arrays (...). Padding cells must have l = u = 0.
    Returns the maximiser x (..., m). Assumes feasibility: sum(l) <= hi_sum and sum(u) >= lo_sum.
    """
    order = np.argsort(-c, axis=-1, kind="stable")
    cs = np.take_along_axis(c, order, -1)
    ls = np.take_along_axis(l, order, -1)
    us = np.take_along_axis(u, order, -1)
    cap = us - ls
    base = ls.sum(-1)
    need = np.clip(lo_sum - base, 0.0, None)
    before = np.cumsum(cap, -1) - cap
    forced = np.clip(need[..., None] - before, 0.0, cap)
    room = np.clip(hi_sum - base - need, 0.0, None)
    rem = np.where(cs > 0, cap - forced, 0.0)
    before2 = np.cumsum(rem, -1) - rem
    vol = np.clip(room[..., None] - before2, 0.0, rem)
    xs = ls + forced + vol
    x = np.empty_like(xs)
    np.put_along_axis(x, order, xs, -1)
    return x


class GroupedPolytope:
    """Cells arranged as (G groups, m slots); index maps slot -> cell (or -1 for padding)."""

    def __init__(self, index, l, u, lo_sum, hi_sum):
        self.index = index                      # (G, m) int, -1 = padding
        self.mask = index >= 0
        self.l = np.where(self.mask, l, 0.0)    # (G, m)
        self.u = np.where(self.mask, u, 0.0)
        self.lo_sum, self.hi_sum = lo_sum, hi_sum
        if np.any(self.l > self.u + 1e-9):
            raise ValueError("empty box")
        if np.any(self.l.sum(1) > hi_sum * (1 + 1e-9)) or np.any(self.u.sum(1) < lo_sum * (1 - 1e-9)):
            raise ValueError("bands inconsistent with published averages")

    def gather(self, v):
        """v: (..., n_cells) -> (..., G, m)."""
        out = v[..., np.where(self.mask, self.index, 0)]
        return np.where(self.mask, out, 0.0)

    def scatter(self, x, n_cells):
        """x: (..., G, m) -> (..., n_cells)."""
        out = np.zeros(x.shape[:-2] + (n_cells,))
        idx = self.index[self.mask]
        out[..., idx] = x[..., self.mask]
        return out


def pi_max_fast(A, N, P: GroupedPolytope, tol=1e-12, max_iter=100):
    """Worst-case need shares for all states.

    A: (S, n) matrix, row s = a_s (zero outside state s's cells). N: (S,).
    Returns pimax (S,), argmax prices (S, n), iterations used.
    """
    S, n = A.shape
    w = N @ A                                   # (n,) denominator weights
    aG = P.gather(A)                            # (S, G, m)
    wG = P.gather(w)                            # (G, m)
    lam = np.zeros(S)
    for it in range(1, max_iter + 1):
        c = aG - lam[:, None, None] * wG[None]
        x = group_knapsack(c, np.broadcast_to(P.l, c.shape), np.broadcast_to(P.u, c.shape),
                           np.broadcast_to(P.lo_sum, c.shape[:-1]), np.broadcast_to(P.hi_sum, c.shape[:-1]))
        num = (aG * x).sum((1, 2)); den = (wG[None] * x).sum((1, 2))
        new = num / den
        if np.all(np.abs(new - lam) <= tol * np.maximum(1.0, np.abs(new))):
            lam = new
            break
        lam = new
    return lam, P.scatter(x, n), it
