#!/usr/bin/env python3
"""(p, pT) maps of the muon rate after the target: target-production-complete-november-4-2026 against the 506 M 2018
sample, with the statistical uncertainty of each cell. Production momentum, 2018 selection (p > 25 GeV, the 2018 veto plane
emulated), muons per spill of 4e13 p.o.t. (= Hz for a 1 s spill), per GeV^2 of (p, pT); histograms of
bin/compare_new_vs_506M.py.

    python3 bin/plot_p_pt_new_vs_506M.py results/target_talk/nov4/new_selected.npz results/target_talk/histograms_506M_4e13.npz OUTDIR
Writes OUTDIR/p_pt_<component>.png: rows new / 506 M / new over 506 M; columns rate, relative error.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

COMPS = {"all": ("minbias", "onia", "charm", "beauty"), "min_bias": ("minbias",), "quarkonia": ("onia",),
         "charm": ("charm",), "beauty": ("beauty",)}


def h(src, comps, lab, k):
    out = 0
    for c in comps:
        for ch in ("mu-", "mu+"):
            key = f"{c}|{lab}_{ch}_2d_{k}"
            if key in src.files:
                out = out + src[key]
    return out


def main():
    new, ref, out = np.load(sys.argv[1]), np.load(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    PE, TE, pmin = new["PE"], new["TE"], float(new["p_min"])
    pc = np.sqrt(PE[1:] * PE[:-1])
    area = np.diff(PE)[:, None] * np.diff(TE)[None, :]
    for name, cs in COMPS.items():
        a1, a2 = h(new, cs, "phys", "s1"), h(new, cs, "phys", "s2")
        b1, b2 = h(ref, cs, "ref", "s1"), h(ref, cs, "ref", "s2")
        with np.errstate(divide="ignore", invalid="ignore"):
            ra, rb = a1 / area, b1 / area                                   # Hz / GeV^2
            ea, eb = np.sqrt(a2) / a1, np.sqrt(b2) / b1                     # relative errors
            rat = a1 / b1
            er = rat * np.sqrt(ea ** 2 + eb ** 2)
        for z in (ra, rb, ea, eb, rat, er):
            z[pc < pmin] = np.nan
        ra[a1 <= 0] = np.nan; rb[b1 <= 0] = np.nan; ea[a1 <= 0] = np.nan; eb[b1 <= 0] = np.nan
        bad = ~((a1 > 0) & (b1 > 0))
        rat[bad] = np.nan; er[bad] = np.nan
        er_all = er.copy()
        rat[er > 0.5 * rat] = np.nan                                       # ratio shown where known to 50 %
        vmax = np.nanmax(np.concatenate([ra.ravel(), rb.ravel()]))
        rnorm = LogNorm(vmax * 1e-8, vmax)
        fig, axs = plt.subplots(3, 2, figsize=(17, 17), constrained_layout=True)
        panels = ((ra, "new sample: muons / spill / GeV$^2$ (Hz)", rnorm, "viridis", "Hz / GeV$^2$"),
                  (100 * ea, "new sample: relative statistical error", LogNorm(0.01, 100), "magma_r", "%"),
                  (rb, "506 M sample (2018): muons / spill / GeV$^2$ (Hz)", rnorm, "viridis", "Hz / GeV$^2$"),
                  (100 * eb, "506 M sample: relative statistical error", LogNorm(0.01, 100), "magma_r", "%"),
                  (rat, "new / 506 M (grey: error > 50 %)", TwoSlopeNorm(1.0, 0.4, 1.6), "RdBu_r", "ratio"),
                  (100 * er_all / np.where(a1 / np.where(b1 > 0, b1, np.nan) > 0, a1 / np.where(b1 > 0, b1, np.nan), np.nan),
                   "relative error of the ratio", LogNorm(0.1, 100), "magma_r", "%"))
        for ax, (z, title, norm, cmap, cl) in zip(axs.ravel(), panels):
            m = ax.pcolormesh(PE, TE, z.T, norm=norm, cmap=cmap, shading="flat")
            ax.set_facecolor("0.88"); ax.set_xlim(pmin, 400); ax.set_ylim(0, 4)
            ax.set_xlabel("production p [GeV]"); ax.set_ylabel("production pT [GeV]"); ax.set_title(title, fontsize=12)
            fig.colorbar(m, ax=ax, label=cl)
        # numbers in the ratio panel where the ratio is known to better than 30 %
        ax = axs[2, 0]
        for i in range(len(PE) - 1):
            for j in range(len(TE) - 1):
                if np.isfinite(rat[i, j]) and er[i, j] < 0.3 * rat[i, j] and i % 2 == 0 and j % 2 == 0:
                    ax.text(pc[i], 0.5 * (TE[j] + TE[j + 1]), f"{rat[i, j]:.2f}", fontsize=5.5, ha="center", va="center")
        fig.suptitle(f"{name.replace('_', ' ')}: muon rate after the target per (p, pT) cell, per spill of 4e13 p.o.t.; "
                     f"totals (p > {pmin:.0f} GeV): new {np.nansum(a1[pc >= pmin]):.3g} Hz, 506 M {np.nansum(b1[pc >= pmin]):.3g} Hz",
                     fontsize=13)
        fig.savefig(out / f"p_pt_{name}.png", dpi=85)
        plt.close(fig)
        print(out / f"p_pt_{name}.png")


if __name__ == "__main__":
    main()
