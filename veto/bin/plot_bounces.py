#!/usr/bin/env python3
"""Where muons bounce in the cavern and rough trajectories, from a --storage shell transport (stored rows, their weights).

    python3 bin/plot_bounces.py <seed dir> --out bounces [--seeds 200]

Bounce point K: closest approach of the line forward from the z = 30.6 m state and the line backward from the z = 82 m state
(muon_types.bounce_points; only credible single bounces). Figure 1: K in top (z-x) and side (z-y) view, weighted, by charge,
with the cavern wall (x = -4.56 m), floor (y = -3.36 m), vessel and shield end. Figure 2: rough paths of sampled muons per
type: source (target) -> state at 30.6 m -> K -> state at 82 m (or stop). The piece through the shield is a straight line
from the target to the 30.6 m state: the real path inside the magnets is curved.
"""
import _bootstrap  # noqa: F401
import argparse
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from veto import geometry as geo
from veto import muon_types as mt

WALL, FLOOR = geo.CAVERN_X_WALL, geo.CAVERN_Y_FLOOR


def outline(ax, view):
    zs = np.array([geo.Z_VESSEL_MIN, geo.Z_VESSEL_MAX])
    if view == "x":
        h = np.array([geo.VESSEL_X_IN, geo.VESSEL_X_OUT])
        ax.axhline(WALL, color="#8a7b63", lw=2); ax.text(1, WALL - 0.45, "cavern wall", color="#8a7b63", fontsize=9)
    else:
        h = np.array([geo.VESSEL_Y_IN, geo.VESSEL_Y_OUT])
        ax.axhline(FLOOR, color="#8a7b63", lw=2); ax.text(1, FLOOR - 0.45, "cavern floor", color="#8a7b63", fontsize=9)
    ax.plot(zs, h, color="#2a78d6", lw=1.2); ax.plot(zs, -h, color="#2a78d6", lw=1.2)
    ax.axvspan(0, 29.89, color="0.85", zorder=0); ax.text(1, (5.5 if view == "x" else 5.5), "muon shield", fontsize=9, color="0.4")
    ax.axvline(geo.Z_SST1, color="#2a78d6", lw=1, ls="--")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir"); ap.add_argument("--out", required=True); ap.add_argument("--seeds", type=int, default=200)
    a = ap.parse_args()
    files = sorted(glob.glob(f"{a.dir}/seed_*.npz"))[:a.seeds]
    S = len(files)
    ZE, XE, YE = np.linspace(28, 84, 113), np.linspace(-8, 11, 77), np.linspace(-5, 8, 53)
    H = {(q, v): 0 for q in (1, -1) for v in ("x", "y")}
    stat = {}
    rng = np.random.default_rng(3)
    paths = {t: [] for t in (0, 1, 3, 4)}
    for f in files:
        d = np.load(f)
        w, t, q = d["weight"].astype(float), d["type"], d["pdg"]
        k = np.isfinite(d["kz"])
        for qq in (1, -1):
            m = k & (q == qq)
            H[(qq, "x")] = H[(qq, "x")] + np.histogram2d(d["kz"][m], d["kx"][m], [ZE, XE], weights=w[m])[0]
            H[(qq, "y")] = H[(qq, "y")] + np.histogram2d(d["kz"][m], d["ky"][m], [ZE, YE], weights=w[m])[0]
        for name, m in (("wall (x < -4.3 m)", k & (d["kx"] < WALL + 0.26)), ("floor (y < -3.1 m)", k & (d["ky"] < FLOOR + 0.26)),
                        ("elsewhere", k & (d["kx"] >= WALL + 0.26) & (d["ky"] >= FLOOR + 0.26))):
            for qq in (1, -1):
                stat[(name, qq)] = stat.get((name, qq), 0.0) + w[m & (q == qq)].sum()
        for tt in paths:
            m = np.nonzero(t == tt)[0]
            if len(m) and len(paths[tt]) < 120:
                take = rng.choice(m, min(3, len(m)), replace=False, p=w[m] / w[m].sum())
                for i in take:
                    pts = [(d["z_start"][i], d["x_start"][i], d["y_start"][i]), (d["z_30"][i], d["x_30"][i], d["y_30"][i])]
                    if np.isfinite(d["kz"][i]):
                        pts.append((d["kz"][i], d["kx"][i], d["ky"][i]))
                    pts.append((d["z_82"][i], d["x_82"][i], d["y_82"][i]))
                    paths[tt].append((np.array(pts), int(q[i])))
    print(f"{S} seeds; bounces per spill (weighted), mu+ / mu-:")
    for name in ("wall (x < -4.3 m)", "floor (y < -3.1 m)", "elsewhere"):
        print(f"   {name:20s} mu+ {stat[(name, 1)] / S:10.4g}   mu- {stat[(name, -1)] / S:10.4g}")
    # figure 1
    fig, axs = plt.subplots(2, 2, figsize=(22, 12), constrained_layout=True)
    for j, qq in enumerate((1, -1)):
        for i, (v, E) in enumerate((("x", XE), ("y", YE))):
            ax = axs[i, j]
            h = H[(qq, v)] / S
            im = ax.pcolormesh(ZE, E, np.where(h > 0, h, np.nan).T, norm=LogNorm(), cmap="viridis")
            outline(ax, v)
            ax.set_xlabel("z [m]"); ax.set_ylabel(f"{v} [m]")
            ax.set_title(f"{'mu+' if qq > 0 else 'mu-'}: where the path bends (bounce point), {'top' if v == 'x' else 'side'} view")
            fig.colorbar(im, ax=ax, label="muons / spill per bin")
    fig.suptitle("Bounce points of the transported muons (credible single bounces), Nov-4 pool", fontsize=15)
    fig.savefig(a.out + "_points.png", dpi=65)
    # figure 2
    names = {0: "SST1 bounced", 1: "SST1 selected", 3: "veto only", 4: "rock bounced"}
    fig, axs = plt.subplots(2, 4, figsize=(30, 12), constrained_layout=True)
    for j, tt in enumerate((0, 4, 3, 1)):
        for i, v in enumerate(("x", "y")):
            ax = axs[i, j]
            outline(ax, v)
            for pts, qq in paths[tt]:
                ax.plot(pts[:2, 0], pts[:2, 1 if v == "x" else 2], color="#c0392b" if qq > 0 else "#2a78d6", lw=0.5, alpha=0.35, ls=":")
                ax.plot(pts[1:, 0], pts[1:, 1 if v == "x" else 2], color="#c0392b" if qq > 0 else "#2a78d6", lw=0.8, alpha=0.6)
            ax.set_xlim(-2, 84); ax.set_ylim(*((-8, 11) if v == "x" else (-5, 8)))
            ax.set_xlabel("z [m]"); ax.set_ylabel(f"{v} [m]")
            ax.set_title(f"{names[tt]} ({len(paths[tt])} sampled by weight): {'top' if v == 'x' else 'side'} view; red mu+, blue mu-")
    fig.suptitle("Rough muon paths: target -> 30.6 m (dotted: straight through the shield, the real path bends) -> bounce point "
                 "-> 82 m", fontsize=15)
    fig.savefig(a.out + "_paths.png", dpi=60)
    print(a.out + "_points.png", a.out + "_paths.png")


if __name__ == "__main__":
    main()
