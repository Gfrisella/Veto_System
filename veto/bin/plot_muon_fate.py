#!/usr/bin/env python3
"""Figure of bin/muon_fate.py: one row per chosen muon (fate fractions, stopping z, x-y at 30.6 and 82 m, momentum at 82 m,
shell side cells).   python3 bin/plot_muon_fate.py fate.npz fate.png"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

d = np.load(sys.argv[1])
names, k = [str(x) for x in d["names"]], len(d["names"])
N = float(d["copies"]) * float(d["passes"])
TN = ["SST1\nbounced", "SST1\nselected", "SST1\nother", "veto\nonly", "rock\nbounced", "straight\npast", "stops\nbetween", "stops\nin shield"]
p0 = np.sqrt(d["px_start"].astype(float) ** 2 + d["py_start"] ** 2 + d["pz_start"] ** 2)
th0 = np.arctan2(np.hypot(d["px_start"], d["py_start"]), d["pz_start"]) * 1e3
fig, axs = plt.subplots(k, 6, figsize=(30, 4.6 * k), gridspec_kw=dict(width_ratios=[1.3, 1.2, 1, 1, 1.1, 1.4]),
                        constrained_layout=True)
ink = "0.25"
for i in range(k):
    q = "+" if d["pdg"][i] < 0 else "-"
    label = f"{names[i]}\n(chosen as: in one seed)\np = {p0[i]:.1f} GeV, {th0[i]:.1f} mrad, $\\mu^{q}$"
    # 1. fates
    ax = axs[i, 0]
    f = d["types"][i] / N
    bars = ax.bar(range(8), np.maximum(f, 1e-12), color="#2a78d6")
    ax.set_yscale("log"); ax.set_ylim(1 / N / 3, 2); ax.set_xticks(range(8)); ax.set_xticklabels(TN, fontsize=8)
    for j, v in enumerate(f):
        if v > 0:
            ax.text(j, v * 1.4, f"{100 * v:.3g}%" if v >= 1e-4 else f"{v:.1e}", ha="center", fontsize=7, color=ink)
    ax.set_ylabel(label, fontsize=10)
    ax.set_title(f"fate of {N:.0e} transports" if i == 0 else "", fontsize=11)
    # 2. stopping z
    ax = axs[i, 1]
    z = d["ZSTOP"]
    h = d["stop_z"][i] / N
    ax.step(0.5 * (z[1:] + z[:-1]), np.where(h > 0, h, np.nan), where="mid", color="#c77700")
    ax.axvspan(28.89, 29.89, color="#2b8a3e", alpha=0.2, lw=0)
    ax.axvline(29.89, color=ink, lw=0.8, ls=":")
    ax.set_yscale("log"); ax.set_xlim(0, 82); ax.set_xlabel("stopping z [m] (green: last metre of the shield)")
    ax.set_title("where it stops (fraction per 0.5 m)" if i == 0 else "", fontsize=11)
    # 3. x-y at 30.6
    ax = axs[i, 2]
    e = d["XY30"]
    m = ax.pcolormesh(e, e, d["xy30"][i].T / N, norm=LogNorm(1 / N, None), cmap="viridis", shading="flat")
    ax.add_patch(plt.Rectangle((-0.5, -1.35), 1.0, 2.7, fill=False, ec="w", lw=1))
    ax.axhline(-3.36, color="w", lw=0.8, ls="--"); ax.axvline(-4.56, color="w", lw=0.8, ls="--")
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("x-y at z = 30.6 m (window, floor, wall)" if i == 0 else "", fontsize=11)
    fig.colorbar(m, ax=ax, shrink=0.8)
    # 4. x-y at 82
    ax = axs[i, 3]
    e = d["XY82"]
    if d["xy82"][i].sum() > 0:
        m = ax.pcolormesh(e, e, d["xy82"][i].T / N, norm=LogNorm(1 / N, None), cmap="viridis", shading="flat")
        fig.colorbar(m, ax=ax, shrink=0.8)
    ax.add_patch(plt.Rectangle((-2, -3), 4, 6, fill=False, ec="w", lw=1))
    ax.axhline(-3.36, color="w", lw=0.8, ls="--"); ax.axvline(-4.56, color="w", lw=0.8, ls="--")
    ax.set_facecolor("0.15"); ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("x-y at z = 82 m (SST1 aperture)" if i == 0 else "", fontsize=11)
    # 5. momentum at 82
    ax = axs[i, 4]
    pb = d["PB"]
    h = d["p82"][i] / N
    ax.step(np.sqrt(pb[1:] * pb[:-1]), np.where(h > 0, h, np.nan), where="mid", color="#2a78d6")
    ax.axvline(p0[i], color=ink, lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("p at z = 82 m [GeV] (dotted: at the target)")
    ax.set_title("momentum left at 82 m" if i == 0 else "", fontsize=11)
    # 6. shell side cells: z vs sector
    ax = axs[i, 5]
    side = d["shell"][i][:32 * 16].reshape(32, 16) / N
    zc = np.linspace(29.89, 94.5, 33)
    m = ax.pcolormesh(np.arange(17) * 360 / 16 - 180, zc, np.where(side > 0, side, np.nan), norm=LogNorm(1 / N, None),
                      cmap="magma", shading="flat")
    ax.set_ylim(29.89, 82); ax.set_xlabel("angle around the shell [deg] (0 = +x, 90 = top, -90 = floor)"); ax.set_ylabel("z [m]")
    up, dn = d["shell"][i][512:576].sum() / N, d["shell"][i][576:].sum() / N
    ax.set_title(("shell side crossings per transport" if i == 0 else "") + f"\nup cap {up:.3g}, down cap (SST1) {dn:.3g}", fontsize=10)
    fig.colorbar(m, ax=ax, shrink=0.8)
fig.suptitle("Five muons from the Nov-4 pool, each transported many times through the shield and cavern (TRY6, field, "
             "MuonsAndMatter)", fontsize=15)
fig.savefig(sys.argv[2], dpi=70)
print(sys.argv[2])
