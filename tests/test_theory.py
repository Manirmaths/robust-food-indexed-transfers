import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from robust import need, efficiency, full_info_protection, protection, pi_max, robust_allocation, proportional_allocation
from polytope import Polytope


def instance(seed, S=5, J=3, eps=0.02, width=(0.7, 1.4)):
    rng = np.random.default_rng(seed)
    N = rng.integers(1, 10, S).astype(float)
    a = rng.uniform(0.5, 2, (S, J))
    c0 = rng.uniform(50, 150, (S, J))
    U = Polytope(S, J)
    for s in range(S):
        for j in range(J):
            U.bound(s, j, c0[s, j] * width[0], c0[s, j] * width[1])
    for j in range(J):
        U.average(range(S), j, c0[:, j].mean(), eps)
    return rng, N, a, U


def test_proposition1_full_information():
    rng = np.random.default_rng(0)
    for _ in range(200):
        S, J = 6, 4
        N = rng.integers(1, 20, S).astype(float); a = rng.uniform(0.1, 3, (S, J))
        p = rng.uniform(10, 100, S * J); B = 1000.0
        T = proportional_allocation(need(a, p), N, B)
        assert np.isclose(protection(T, a, p), full_info_protection(a, N, B, p))
        Talt = proportional_allocation(rng.uniform(0.1, 1, S), N, B)
        assert protection(Talt, a, p) <= full_info_protection(a, N, B, p) + 1e-12


def test_theorem1_closed_form_is_maximin():
    for seed in range(5):
        rng, N, a, U = instance(seed)
        lb, ub, R, r = U.matrices(); B = 100.0
        pm, arg = pi_max(a, N, lb, ub, R, r, return_argmax=True)
        T, G = robust_allocation(pm, N, B)
        assert 0 < G <= 1 + 1e-12
        # (i) the guarantee is attained at every state's maximiser
        for p in arg:
            assert np.isclose(efficiency(T, a, N, B, p), G, atol=1e-7)
        # (ii) no allocation does better at those maximisers (an upper bound on its worst case)
        for w in rng.dirichlet(np.ones(len(N)), 5000):
            Ta = proportional_allocation(w, N, B)
            assert min(efficiency(Ta, a, N, B, p) for p in arg) <= G + 1e-9
        # (iii) the guarantee holds at random feasible prices
        lo, hi = lb.ravel(), ub.ravel(); kept = 0
        while kept < 2000:
            p = rng.uniform(lo, hi)
            if R is None or np.all(R @ p <= r + 1e-9):
                assert efficiency(T, a, N, B, p) >= G - 1e-9; kept += 1


def test_monotonicity_more_information_never_hurts():
    for seed in range(5):
        rng, N, a, U = instance(seed)
        lb, ub, R, r = U.matrices()
        G0 = robust_allocation(pi_max(a, N, lb, ub, R, r), N, 1.0)[1]
        U.fix(0, 0, (lb[0, 0] + ub[0, 0]) / 2)   # observe one price
        lb, ub, R, r = U.matrices()
        G1 = robust_allocation(pi_max(a, N, lb, ub, R, r), N, 1.0)[1]
        assert G1 >= G0 - 1e-10


def test_full_information_gives_G_equal_one():
    rng, N, a, U = instance(7)
    p = np.random.default_rng(1).uniform(60, 120, a.size)
    G = robust_allocation(pi_max(a, N, p, p), N, 1.0)[1]
    assert np.isclose(G, 1.0)
