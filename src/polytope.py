"""Build the admissible price polytope U_t from an information set (U1-U6)."""
from __future__ import annotations
import numpy as np


class Polytope:
    """U = {p : lb <= p <= ub, R p <= r}, p ordered state-major (S x J)."""

    def __init__(self, S: int, J: int):
        self.S, self.J = S, J
        self.lb = np.zeros((S, J))
        self.ub = np.full((S, J), np.inf)
        self.rows, self.rhs = [], []

    # (U1) monitored prices
    def fix(self, s, j, value):
        self.lb[s, j] = max(self.lb[s, j], value); self.ub[s, j] = min(self.ub[s, j], value)

    # (U4)/(U5)/(U6) box information
    def bound(self, s, j, lo=None, hi=None):
        if lo is not None: self.lb[s, j] = max(self.lb[s, j], lo)
        if hi is not None: self.ub[s, j] = min(self.ub[s, j], hi)

    # (U2)/(U3) published averages over a set of states, with tolerance eps
    def average(self, states, j, value, eps):
        row = np.zeros((self.S, self.J)); row[list(states), j] = 1.0 / len(states)
        self.rows += [row.ravel(), -row.ravel()]
        self.rhs += [value * (1 + eps), -value * (1 - eps)]

    def matrices(self):
        if not np.all(np.isfinite(self.ub)):
            raise ValueError("polytope must be bounded: set an upper bound for every price")
        if np.any(self.lb > self.ub + 1e-9):
            raise ValueError("empty polytope: some lower bound exceeds its upper bound")
        R = np.array(self.rows) if self.rows else None
        r = np.array(self.rhs) if self.rhs else np.array([])
        return self.lb, self.ub, R, r
