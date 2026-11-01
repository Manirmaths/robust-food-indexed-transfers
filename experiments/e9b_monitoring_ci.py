"""Block-bootstrap intervals for the monitoring curve (test months), from the selections stored by e5_monitoring.py."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from setup import Panel, ROOT
from e5_monitoring import calibrated_boxes, G_of
from e9_robustness import block_ci

TAB = os.path.join(ROOT, "results", "tables")
if __name__ == "__main__":
    P = Panel(); B = calibrated_boxes(P); dates = B["dates"]
    test = [k for k in B["ks"] if dates[k] >= pd.Timestamp("2022-07-01")]
    M = pd.read_csv(os.path.join(TAB, "e5_monitoring_curve.csv")); M = M[M.split == "test"]
    base = np.array([G_of(P, B, k, [], 0)[0] for k in test])
    rows = []
    for _, r in M[M.K.isin([1, 3, 5, 10, 15])].iterrows():
        idx = [P.states.index(s) for s in str(r.states).split("; ")]
        g = np.array([G_of(P, B, k, idx, int(r.delay))[0] for k in test])
        lo, hi = block_ci(g); dlo, dhi = block_ci(g - base)
        rows.append(dict(design=r.design, delay=r.delay, K=r.K, G=g.mean(), lo=lo, hi=hi, gain=(g - base).mean(), gain_lo=dlo, gain_hi=dhi))
    lo, hi = block_ci(base); rows.append(dict(design="none", delay=0, K=0, G=base.mean(), lo=lo, hi=hi, gain=0, gain_lo=0, gain_hi=0))
    # decision minus accuracy at the same K (d = 0)
    for K in (3, 5, 10, 15):
        gs = {}
        for des in ("decision", "accuracy"):
            r = M[(M.design == des) & (M.delay == 0) & (M.K == K)].iloc[0]
            idx = [P.states.index(s) for s in r.states.split("; ")]
            gs[des] = np.array([G_of(P, B, k, idx, 0)[0] for k in test])
        d = gs["decision"] - gs["accuracy"]; lo, hi = block_ci(d)
        rows.append(dict(design="decision-accuracy", delay=0, K=K, G=d.mean(), lo=lo, hi=hi))
    R = pd.DataFrame(rows); R.to_csv(os.path.join(TAB, "e9_monitoring_ci.csv"), index=False)
    print(R.round(3).to_string(index=False))
