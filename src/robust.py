"""Core optimisation for the conformally calibrated linear-fractional robust model.

Notation follows the paper:
  p      : price vector of length n = S*J, ordered state-major (s, j)
  a      : (S, J) array, D_s(p) = sum_j a[s, j] * p[s, j]   (needed entitlement)
  N      : (S,) recipients per state
  B      : budget
The admissible price set is the polytope
  U = { p : lb <= p <= ub,  R p <= r }.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog


def need(a: np.ndarray, p: np.ndarray) -> np.ndarray:
    """D_s(p) for every state."""
    S, J = a.shape
    return (a * p.reshape(S, J)).sum(axis=1)


def protection(T, a, p):
    """rho(T, p) = min_s T_s / D_s(p)."""
    return float(np.min(T / need(a, p)))


def full_info_protection(a, N, B, p):
    """rho*(p) = B / sum_k N_k D_k(p)  (Proposition 1)."""
    return float(B / np.dot(N, need(a, p)))


def efficiency(T, a, N, B, p):
    """E(T, p) = rho(T, p) / rho*(p), in (0, 1]."""
    return protection(T, a, p) / full_info_protection(a, N, B, p)


def proportional_allocation(weights, N, B):
    """T_s = B w_s / sum_k N_k w_k : any rule that pays in proportion to w."""
    w = np.asarray(weights, float)
    return B * w / np.dot(N, w)


def pi_max(a, N, lb, ub, R=None, r=None, return_argmax=False):
    """Worst-case need share pi_s^max(U) for every state (Proposition 2).

    Solves, for each s, the Charnes-Cooper LP
        max a_s'y  s.t.  R y <= r tau,  lb tau <= y <= ub tau,
                         sum_k N_k a_k'y = 1,  y, tau >= 0,
    whose optimum equals max_{p in U} D_s(p) / sum_k N_k D_k(p).
    """
    S, J = a.shape
    n = S * J
    lb = np.asarray(lb, float).ravel(); ub = np.asarray(ub, float).ravel()
    I = sp.identity(n, format="csr")
    blocks = [sp.hstack([I, -sp.csr_matrix(ub[:, None])]),
              sp.hstack([-I, sp.csr_matrix(lb[:, None])])]
    if R is not None and len(r):
        R = sp.csr_matrix(R)
        blocks.append(sp.hstack([R, -sp.csr_matrix(np.asarray(r, float)[:, None])]))
    A_ub = sp.vstack(blocks).tocsr()
    b_ub = np.zeros(A_ub.shape[0])
    total = (np.asarray(N, float)[:, None] * a).ravel()
    A_eq = sp.csr_matrix(np.r_[total, 0.0][None, :])
    out, arg = np.empty(S), []
    for s in range(S):
        c = np.zeros(n + 1)
        c[s * J:(s + 1) * J] = -a[s]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                      bounds=[(0, None)] * (n + 1), method="highs")
        if res.status != 0:
            raise RuntimeError(f"LP for state {s} failed: {res.message}")
        out[s] = -res.fun
        if return_argmax:
            arg.append(res.x[:n] / res.x[-1])
    return (out, arg) if return_argmax else out


def robust_allocation(pimax, N, B):
    """Theorem 1: T_s = B pi_s^max / sum_k N_k pi_k^max and guarantee G."""
    pimax = np.asarray(pimax, float)
    denom = float(np.dot(N, pimax))
    return B * pimax / denom, 1.0 / denom
