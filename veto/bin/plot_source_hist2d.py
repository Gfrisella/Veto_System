#!/usr/bin/env python3
"""Plots of bin/source_hist2d.py: flux and its statistical error in (p, pT) for the old inputs and the new pool.

    python3 bin/plot_source_hist2d.py results/pool_v2/hist2d.npz
Writes source_hist2d_all.png and source_hist2d_heavy.png (min. bias + charm, the rows that limit the statistics).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

SAMPLES = (("old506", "Guglielmo's 506 M file"), ("enr601", "601 M enriched file\n(datasets a, b)"), ("pool", "new pool, training part\n(dataset c)"))


def figure(h, comps, title, out):
    pe, te = h["p_edges"], h["pt_edges"]
    s1 = {k: h[f"{k}_s1"][comps].sum(0) for k, _ in SAMPLES}
    s2 = {k: h[f"{k}_s2"][comps].sum(0) for k, _ in SAMPLES}
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = {k: np.where(s1[k] > 0, 100 * np.sqrt(s2[k]) / s1[k], np.nan) for k in s1}
    fl = {k: np.where(s1[k] > 0, s1[k], np.nan) for k in s1}
    vmax = np.nanmax([np.nanmax(v) for v in fl.values()])
    fig, axs = plt.subplots(2, 4, figsize=(22, 10), constrained_layout=True)
    for j, (k, lab) in enumerate(SAMPLES):
        m0 = axs[0, j].pcolormesh(pe, te, fl[k].T, norm=LogNorm(vmax * 1e-7, vmax), cmap="viridis")
        axs[0, j].set_title(f"{lab}\nflux [muons / spill / bin]")
        m1 = axs[1, j].pcolormesh(pe, te, rel[k].T, norm=LogNorm(0.1, 100), cmap="magma_r")
        axs[1, j].set_title(f"{lab}\nstatistical error of the flux [%]")
    fig.colorbar(m0, ax=axs[0, :3], label="muons per spill per bin", shrink=0.9)
    fig.colorbar(m1, ax=axs[1, :3], label="relative error [%]  (dark = poorly known)", shrink=0.9)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = fl["pool"] / fl["enr601"]
        gain = rel["enr601"] / rel["pool"]
    m2 = axs[0, 3].pcolormesh(pe, te, ratio.T, norm=TwoSlopeNorm(1.0, 0.8, 1.2), cmap="RdBu_r")
    axs[0, 3].set_title("flux: new pool / 601 M file\n(white = same flux)")
    fig.colorbar(m2, ax=axs[0, 3], label="ratio")
    m3 = axs[1, 3].pcolormesh(pe, te, gain.T, norm=LogNorm(0.3, 30), cmap="PuOr")
    axs[1, 3].set_title("error: 601 M file / new pool\n(> 1 = new pool more precise)")
    fig.colorbar(m3, ax=axs[1, 3], label="error ratio (x better)")
    for ax in axs.flat:
        ax.set_xscale("log"); ax.set_xlabel("p at production [GeV]"); ax.set_ylabel("pT [GeV]")
    fig.suptitle(title, fontsize=13)
    fig.savefig(out, dpi=100)
    # one-line summary per p range (pT integrated): relative error of the flux
    print(title)
    for lo, hi in ((10, 30), (30, 60), (60, 100), (100, 150), (150, 250), (250, 400)):
        m = (pe[:-1] >= lo) & (pe[1:] <= hi)
        txt = []
        for k, _ in SAMPLES:
            a, b = s1[k][m].sum(), s2[k][m].sum()
            txt.append(f"{k} {100 * np.sqrt(b) / a:.3f}%" if a > 0 else f"{k} -")
        print(f"  p {lo:3d}-{hi:3d} GeV: " + ", ".join(txt))
    return out


def main():
    f = Path(sys.argv[1])
    h = np.load(f)
    figure(h, [0, 1, 2, 3, 4], "All source muons: flux (top) and its statistical error (bottom) in (p, pT) at production",
           f.with_name("source_hist2d_all.png"))
    figure(h, [0, 1, 3], "Min. bias + tail + charm (the heavy rows that limit the veto statistics)",
           f.with_name("source_hist2d_heavy.png"))


if __name__ == "__main__":
    main()
