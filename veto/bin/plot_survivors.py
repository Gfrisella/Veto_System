#!/usr/bin/env python3
"""Where are the SST1 muons the veto misses? One figure per design, from the 200 k single-muon sample.

Runs the same chain as run_reco.py (single muons: own hits + accidental hits, algorithm A) and keeps
the per-muon veto flag. Plots the muons that are NOT vetoed, weighted in muons per spill:
  side (y-z) and top (x-z) views of their true paths with the design's planes, the cavern wall and floor;
  their position at SST1; where the bounced ones bounced (wall x = -4.56 m, floor y = -3.36 m);
  their momentum at SST1 compared with all SST1 muons.

  bin/plot_survivors.py D6_straw
  bin/plot_survivors.py D6_straw --floor          # also a copy with every fin down to the floor
  bin/plot_survivors.py results/optimize_fins/best.json
"""

import _bootstrap  # noqa: F401

import argparse

import matplotlib.pyplot as plt
import numpy as np

from run_reco import TYPE_NAMES, load_inputs, take
from veto import design as dsg
from veto import geometry as geo
from veto import muon_types as mt
from veto import paths, plotting, reco

COLORS = {0: "#eb6834", 1: "#2a78d6", 2: "#1baf7a"}     # bounced, selected, other (validated, CVD-safe)
N_PATHS = 300


def run(design, tr, inp, cfg):
    single = inp["single"]
    rng = np.random.default_rng(cfg.seed)
    m = reco.run_events(take(tr, single), np.arange(len(single)), np.full(len(single), 1e9), design, inp, cfg, rng)
    return single, inp["single_w"], tr["type"][single].astype(int), m["vetoed"]


def draw_frame(ax, design, axis):
    """Vessel, cavern limit and the design's planes in the z-x (axis = 'x') or z-y (axis = 'y') view."""
    z = np.linspace(geo.Z_VESSEL_MIN, geo.Z_VESSEL_MAX, 2)
    hx, hy = geo.vessel_half_widths(z)
    half = hx if axis == "x" else hy
    ax.fill_between(z, -half, half, color="0.85", lw=0, label="decay vessel")
    limit = geo.CAVERN_X_WALL if axis == "x" else geo.CAVERN_Y_FLOOR
    ax.axhline(limit, color="0.3", ls=":", lw=1.5, label="cavern wall x = -4.56 m" if axis == "x" else "floor y = -3.36 m")
    half_sst1 = geo.SST1_HALF_X if axis == "x" else geo.SST1_HALF_Y
    ax.vlines(geo.Z_SST1, -half_sst1, half_sst1, color="0.2", lw=2)
    ax.text(geo.Z_SST1, half_sst1 + 0.2, "SST1", ha="center", fontsize=8)
    for k, p in enumerate(design.planes):
        label = "veto planes" if k == 0 else None
        if p.kind == "bottom":      # panel under the vessel: a slanted line from the side, a box from the top
            if axis == "x":
                ax.add_patch(plt.Rectangle((p.z, p.x_lo), p.z_end - p.z, p.u_max, color="tab:purple", alpha=0.6,
                                           lw=0, label="bottom panel"))
            else:
                ax.plot([p.z, p.z_end], [p.y0, p.y_end], color="tab:purple", lw=3.5, label="bottom panel")
            continue
        z0, z1 = p.z_at(0.0), p.z_at(p.u_max)
        if axis == "x":         # top view: a tilted fin is a slanted segment
            ax.plot([z0, z1], [p.to_global(0.0), p.to_global(p.u_max)], color="0.1", lw=2.5, label=label)
        elif z0 == z1:
            ax.vlines(p.z, p.y_lo, p.y_hi, color="0.1", lw=2.5, label=label)
        else:                   # side view: a tilted fin covers z0 .. z1
            ax.fill_between(sorted([z0, z1]), p.y_lo, p.y_hi, color="0.1", alpha=0.5, lw=0, label=label)


def figure(design, tr, single, w, ty, vetoed, outdir):
    surv = ~vetoed
    idx = single[surv]
    ws, ts = w[surv], ty[surv]
    rate = {k: (w[ty == k].sum(), ws[ts == k].sum()) for k in range(3)}

    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"{design.name}: SST1 muons not vetoed, {ws.sum():.0f} of {w.sum():.0f} per spill\n"
                 + ",   ".join(f"{TYPE_NAMES[k]} {rate[k][1]:.0f} / {rate[k][0]:.0f}" for k in range(3)), fontsize=14)

    # paths of a weighted random sample of the survivors
    rng = np.random.default_rng(0)
    pick = rng.choice(len(idx), min(N_PATHS, len(idx)), replace=False, p=ws / ws.sum()) if len(idx) else []
    for a, axis, j in ((ax[0, 0], "y", 1), (ax[0, 1], "x", 0)):
        draw_frame(a, design, axis)
        for i in pick:
            r = idx[i]
            pts = [(tr["az"][r], (tr["ax"][r], tr["ay"][r])[j])]
            if np.isfinite(tr["kz"][r]):
                pts.append((tr["kz"][r], (tr["kx"][r], tr["ky"][r])[j]))
            pts.append((tr["bz"][r], (tr["bx"][r], tr["by"][r])[j]))
            zz, vv = zip(*pts)
            a.plot(zz, vv, color=COLORS[ts[i]], lw=0.7, alpha=0.6)
        for k in range(3):
            a.plot([], [], color=COLORS[k], lw=2, label=f"{TYPE_NAMES[k]} (not vetoed)")
        a.set_xlabel("z [m]")
        a.set_ylabel(f"{axis} [m]")
        a.set_xlim(29, 84)
        a.set_title(f"{'Side' if axis == 'y' else 'Top'} view, {len(pick)} paths", fontsize=11)
    ax[0, 0].legend(fontsize=7, loc="upper left")
    ax[0, 1].set_ylim(-5, 5)
    ax[0, 0].set_ylim(-3.8, 5)

    # position at SST1
    a = ax[0, 2]
    for k in range(3):
        m = ts == k
        a.scatter(tr["bx"][idx[m]], tr["by"][idx[m]], s=4, color=COLORS[k], alpha=0.5, label=TYPE_NAMES[k])
    a.add_patch(plt.Rectangle((-geo.SST1_HALF_X, -geo.SST1_HALF_Y), 2 * geo.SST1_HALF_X, 2 * geo.SST1_HALF_Y,
                              fill=False, color="0.2"))
    a.set_aspect("equal")
    a.set_xlabel("x at SST1 [m]")
    a.set_ylabel("y at SST1 [m]")
    a.set_title("Position at SST1", fontsize=11)
    a.legend(fontsize=7, markerscale=3)

    # bounce points: wall (z, y) and floor (z, x), survivors vs all bounced
    bounced = (ty == 0) & np.isfinite(tr["kz"][single])
    kx, ky, kz = (tr[c][single] for c in ("kx", "ky", "kz"))
    on_wall = (kx - geo.CAVERN_X_WALL) < (ky - geo.CAVERN_Y_FLOOR)
    for a, sel, coord, lab in ((ax[1, 0], on_wall, ky, "y on the wall x = -4.56 m [m]"),
                               (ax[1, 1], ~on_wall, kx, "x on the floor y = -3.36 m [m]")):
        m_all, m_s = bounced & sel, bounced & sel & ~vetoed
        zb = np.linspace(0, 85, 86)
        cb = np.linspace(-5, 3, 41) if coord is kx else np.linspace(-3.6, 4, 39)
        h, _, _, im = a.hist2d(kz[m_s], coord[m_s], bins=(zb, cb), weights=w[m_s], cmin=1e-9, cmap="Oranges")
        fig.colorbar(im, ax=a, label="missed per spill per bin")
        frac = w[m_s].sum() / max(w[m_all].sum(), 1e-30)
        a.set_title(f"{'Wall' if coord is ky else 'Floor'} bounces: {w[m_s].sum():.0f} / "
                    f"{w[m_all].sum():.0f} per spill missed ({100 * frac:.0f} %)", fontsize=11)
        a.set_xlabel("z of the bounce [m]")
        a.set_ylabel(lab)
        for p in design.planes:
            if p.kind == "bottom":
                a.axvspan(p.z, p.z_end, color="tab:purple", alpha=0.15, lw=0)
            elif p.tilt:
                a.axvspan(*sorted([p.z_at(0.0), p.z_at(p.u_max)]), color="0.3", alpha=0.15, lw=0)
            else:
                a.axvline(p.z, color="0.3", lw=0.8, ls="--")

    # momentum at SST1
    a = ax[1, 2]
    p_all = np.sqrt(tr["bpx"][single] ** 2 + tr["bpy"][single] ** 2 + tr["bpz"][single] ** 2)
    bins = np.logspace(-1, 2.5, 50)
    a.hist(p_all, bins=bins, weights=w, histtype="step", color="0.5", lw=1.5, label="all SST1 muons")
    for k in range(3):
        m = ts == k
        a.hist(p_all[surv][m], bins=bins, weights=ws[m], histtype="step", color=COLORS[k], lw=2,
               label=f"{TYPE_NAMES[k]} not vetoed")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("p at SST1 [GeV]")
    a.set_ylabel("muons per spill per bin")
    a.set_title("Momentum", fontsize=11)
    a.legend(fontsize=7)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = outdir / f"{design.name}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out, rate


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("designs", nargs="+")
    parser.add_argument("--floor", action="store_true", help="also every design with all fins down to the floor")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    plotting.set_style()
    tr, inp = load_inputs()
    cfg = reco.RecoConfig(seed=args.seed)
    outdir = paths.results_dir() / "reco" / "survivors"
    outdir.mkdir(parents=True, exist_ok=True)
    designs = [dsg.load(n) for n in args.designs]
    if args.floor:
        for d in list(designs):
            raw = dict(d.raw, name=d.name + "_floor")
            raw["fins"] = [dict(f, y_lo=mt.FIN_Y_MIN) for f in raw["fins"]]
            designs.append(dsg.from_dict(raw))
    for d in designs:
        single, w, ty, vetoed = run(d, tr, inp, cfg)
        out, rate = figure(d, tr, single, w, ty, vetoed, outdir)
        print(f"{d.name}: " + ", ".join(f"{TYPE_NAMES[k]} {r[1]:.0f}/{r[0]:.0f}" for k, r in rate.items())
              + f" per spill not vetoed -> {out}")


if __name__ == "__main__":
    main()
