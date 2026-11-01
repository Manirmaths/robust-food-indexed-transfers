"""Load the cleaned state panel, basket weights and recipients; build need vectors and information polytopes."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from fastlf import GroupedPolytope

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "derived")
BASE = pd.Timestamp("2019-01-01")


class Panel:
    def __init__(self, weights_pop="poor", weights_scope="zonal", drop_group=None, base=BASE):
        s = pd.read_csv(os.path.join(DATA, "food_state_clean.csv"), parse_dates=["date"])
        IW = pd.read_csv(os.path.join(DATA, "basket_item_weights.csv"))
        R = pd.read_csv(os.path.join(DATA, "state_recipients.csv")).set_index("state")
        IW = IW[IW.population == weights_pop]
        if drop_group:
            IW = IW[IW.group != drop_group]
        self.items = sorted(IW.item.unique())
        self.states = sorted(s.state.unique())
        self.zone_of = s.drop_duplicates("state").set_index("state").zone.reindex(self.states)
        self.zones = sorted(self.zone_of.unique())
        P = s.pivot_table(index="date", columns=["state", "item"], values="price")
        cols = pd.MultiIndex.from_product([self.states, self.items])
        P = P.reindex(columns=cols)
        # fill isolated missing cells with the state's previous value (logged as count)
        self.n_filled = int(P.isna().sum().sum())
        P = P.ffill().bfill()
        self.P = P                                   # (T, S*J) true prices
        self.dates = list(P.index)
        S, J = len(self.states), len(self.items)
        self.S, self.J, self.n = S, J, S * J
        self.N = R.poor_population_2019.reindex(self.states).values.astype(float)
        p0 = P.loc[base].values.reshape(S, J)
        A = np.zeros((S, S * J))
        for i, st in enumerate(self.states):
            scope = "National" if weights_scope == "national" else self.zone_of[st]
            w = IW[IW.scope == scope].set_index("item").weight.reindex(self.items).fillna(0.0).values
            w = w / w.sum()
            A[i, i * J:(i + 1) * J] = w / p0[i]       # D_s(p) = sum_j w_j p_sj / p_sj0  (T0 = 1)
        self.A = A
        self.base = base

    def prices(self, t):
        return self.P.loc[t].values.astype(float)

    def need(self, p):
        return self.A @ p

    # ---- information sets ------------------------------------------------
    def group_index(self, level):
        """(G, m) cell indices for zone x item groups ('zone') or item groups ('national')."""
        S, J = self.S, self.J
        groups = []
        if level == "zone":
            for z in self.zones:
                members = [i for i, st in enumerate(self.states) if self.zone_of[st] == z]
                for j in range(J):
                    groups.append([i * J + j for i in members])
        else:
            for j in range(J):
                groups.append([i * J + j for i in range(S)])
        m = max(len(g) for g in groups)
        return np.array([g + [-1] * (m - len(g)) for g in groups])

    def polytope(self, p_true, l, u, level="zone", eps=0.0, extremes=False, observed_states=()):
        """Polytope from published group averages (computed from true prices), cell bounds l/u,
        optional published extremes (highest and lowest state per item) and monitored states."""
        idx = self.group_index(level)
        mask = idx >= 0
        vals = np.where(mask, p_true[np.where(mask, idx, 0)], 0.0)
        cnt = mask.sum(1)
        mean = vals.sum(1) / cnt
        lo_sum, hi_sum = (1 - eps) * mean * cnt, (1 + eps) * mean * cnt
        l = l.copy(); u = u.copy()
        S, J = self.S, self.J
        if extremes:
            pt = p_true.reshape(S, J)
            hi_s, lo_s = pt.argmax(0), pt.argmin(0)
            H, L = pt.max(0), pt.min(0)
            lm, um = l.reshape(S, J), u.reshape(S, J)
            lm[:] = np.maximum(lm, L[None]); um[:] = np.minimum(um, H[None])
            for j in range(J):
                lm[hi_s[j], j] = um[hi_s[j], j] = H[j]
                lm[lo_s[j], j] = um[lo_s[j], j] = L[j]
        for i in observed_states:
            l[i * J:(i + 1) * J] = p_true[i * J:(i + 1) * J]
            u[i * J:(i + 1) * J] = p_true[i * J:(i + 1) * J]
        # published facts win: where a band contradicts a group's published average, relax the band
        # of that group's free cells to the widest range the published facts allow
        fixed = np.zeros(self.n, bool)
        for i in observed_states: fixed[i * J:(i + 1) * J] = True
        if extremes:
            fx = fixed.reshape(S, J)
            for j in range(J): fx[hi_s[j], j] = fx[lo_s[j], j] = True
        self.relaxed = 0
        for g in range(len(idx)):
            c = idx[g][mask[g]]
            if l[c].sum() > hi_sum[g] * (1 + 1e-12) or u[c].sum() < lo_sum[g] * (1 - 1e-12):
                free = c[~fixed[c]]
                j = c[0] % J
                floor = L[j] if extremes else 0.0
                ceil = H[j] if extremes else hi_sum[g]
                l[free] = floor; u[free] = ceil
                self.relaxed += 1
        lg = np.where(mask, l[np.where(mask, idx, 0)], 0.0); ug = np.where(mask, u[np.where(mask, idx, 0)], 0.0)
        return GroupedPolytope(idx, lg, ug, lo_sum, hi_sum)
