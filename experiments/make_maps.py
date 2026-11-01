"""Choropleth maps of Nigeria's states. Boundaries: geoBoundaries gbOpen NGA ADM1 (CC BY 4.0)."""
import os, sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
import matplotlib.patheffects as pe
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from setup import Panel, ROOT

TAB, FIG = os.path.join(ROOT, "results", "tables"), os.path.join(ROOT, "results", "figures")
GEO = os.path.join(ROOT, "data", "geo", "geoBoundaries-NGA-ADM1_simplified.geojson")
INK, INK2, EDGE = "#0b0b0b", "#52514e", "#ffffff"
BLUE = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
ORANGE = LinearSegmentedColormap.from_list("orange", ["#fde3d6", "#f6a782", "#eb6834", "#b8461d", "#7a2c10"])
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a99", "#b52f2f"])
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2})


def shapes():
    g = json.load(open(GEO)); out = {}
    for f in g["features"]:
        name = f["properties"]["shapeName"]; name = "FCT Abuja" if name.startswith("Abuja") else name
        geom = f["geometry"]; polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        out[name] = [np.array(p[0]) for p in polys]
    return out


def draw(ax, shp, values, cmap, norm, title, cbar_label, ticks=None, fmt=None, marks=None):
    patches, cols = [], []
    for st, polys in shp.items():
        for poly in polys:
            patches.append(Polygon(poly, closed=True)); cols.append(values.get(st, np.nan))
    pc = PatchCollection(patches, cmap=cmap, norm=norm, edgecolor=EDGE, linewidth=0.5)
    pc.set_array(np.array(cols)); ax.add_collection(pc)
    ax.autoscale_view(); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    cb = plt.colorbar(pc, ax=ax, fraction=0.035, pad=0.01, ticks=ticks)
    cb.set_label(cbar_label, color=INK2, fontsize=8); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7.5, colors=INK2)
    if fmt: cb.ax.set_yticklabels([fmt(t) for t in ticks])
    if marks:
        for lab, st in marks:
            x, y = np.concatenate(shp[st]).mean(0)
            ax.text(x, y, lab, ha="center", va="center", fontsize=8, fontweight="bold", color=INK,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])


if __name__ == "__main__":
    P = Panel(); shp = shapes()
    assert set(shp) == set(P.states), set(shp) ^ set(P.states)
    N = dict(zip(P.states, P.N / 1e6))
    D = P.need(P.prices(pd.Timestamp("2023-08-01"))); rel = D / (P.N @ D / P.N.sum())
    growth = dict(zip(P.states, rel))
    B = pd.read_csv(os.path.join(TAB, "e8_band_halfwidth.csv")); B = B[B.h == B.h.max()].set_index("state").half_width.to_dict()
    S = pd.read_csv(os.path.join(TAB, "e8_live_state_allocations.csv"), parse_dates=["date"])
    pay = S[S.date == S.date.max()].set_index("state").T_k90_rel_flat.to_dict()
    M = pd.read_csv(os.path.join(TAB, "e5_monitoring_curve.csv"))
    order = M[(M.design == "decision") & (M.delay == 0) & (M.split == "test") & (M.K == 5)].states.iloc[0].split("; ")

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.4))
    draw(axs[0, 0], shp, N, BLUE, Normalize(0, max(N.values())), "(a) Recipients: poor population, 2019", "millions",
         ticks=[0, 2, 4, 6, 8, 10, 12])
    lt = [0.8, 0.9, 1.0, 1.1, 1.2]
    draw(axs[0, 1], shp, {k: np.log(v) for k, v in growth.items()}, DIV, TwoSlopeNorm(0, np.log(0.75), np.log(1 / 0.75)),
         "(b) Growth of food need, Jan 2019 to Aug 2023", "growth relative to national",
         ticks=np.log(lt), fmt=lambda t: f"{np.exp(t):.1f}")
    draw(axs[1, 0], shp, B, ORANGE, Normalize(0, max(B.values())), "(c) Band half-width at a 33-month horizon",
         "half-width (log points)", ticks=[0, 0.2, 0.4, 0.6, 0.8], marks=[(str(i + 1), s) for i, s in enumerate(order)])
    pt = [0.8, 1.0, 1.25, 1.5]
    draw(axs[1, 1], shp, {k: np.log(v) for k, v in pay.items()}, DIV, TwoSlopeNorm(0, np.log(1 / 1.6), np.log(1.6)),
         "(d) Payment relative to flat, May 2026 ($\\kappa$ = 0.9)", "payment / flat payment",
         ticks=np.log(pt), fmt=lambda t: f"{np.exp(t):.2f}")
    fig.tight_layout()
    for e in ("pdf", "png"): fig.savefig(os.path.join(FIG, f"fig_maps.{e}"), dpi=220)
    # ---- heatmap: relative need by state and month ----
    dates = [t for t in P.dates if t >= pd.Timestamp("2019-01-01")]
    R = np.array([P.need(P.prices(t)) for t in dates]); R = R / (R @ P.N / P.N.sum())[:, None]
    order_s = sorted(range(P.S), key=lambda i: (P.zone_of[P.states[i]], P.states[i]))
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    im = ax.imshow(np.log(R[:, order_s].T), aspect="auto", cmap=DIV, norm=TwoSlopeNorm(0, np.log(0.7), np.log(1 / 0.7)), interpolation="nearest")
    ax.set_yticks(range(P.S)); ax.set_yticklabels([P.states[i] for i in order_s], fontsize=6.5, color=INK2)
    xt = [i for i, t in enumerate(dates) if t.month == 1]
    ax.set_xticks(xt); ax.set_xticklabels([dates[i].strftime("%b %Y") for i in xt], fontsize=7.5, color=INK2)
    zs = [P.zone_of[P.states[i]] for i in order_s]
    for k in range(1, P.S):
        if zs[k] != zs[k - 1]: ax.axhline(k - 0.5, color="white", lw=1.5)
    for k, z in enumerate(zs):
        if k == 0 or zs[k - 1] != z: ax.text(len(dates) + 0.8, k + zs.count(z) / 2 - 0.5, z, fontsize=7, va="center", color=INK2)
    ax.set_xlim(-0.5, len(dates) - 0.5)
    for sp_ in ax.spines.values(): sp_.set_visible(False)
    gap = [i for i, t in enumerate(dates) if t == pd.Timestamp("2023-07-01")]
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.17, ticks=np.log([0.7, 0.85, 1.0, 1.2, 1.4]))
    cb.ax.set_yticklabels(["0.70", "0.85", "1.00", "1.20", "1.40"]); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7.5, colors=INK2)
    cb.set_label("need relative to the national mean (Jan 2019 = 1)", fontsize=8, color=INK2)
    fig.tight_layout()
    for e in ("pdf", "png"): fig.savefig(os.path.join(FIG, f"fig_heatmap.{e}"), dpi=220)
    print("monitoring order", order); print("growth range", min(growth.values()), max(growth.values()), "pay range", min(pay.values()), max(pay.values()))
