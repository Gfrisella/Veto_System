#!/usr/bin/env python3
"""2D (p, pT) comparison of a complete-mode production with the 2018 FairShip files (muons per spill, production
momentum, 2018 selection, no open charm): the flux maps, ours / 2018 and the pull (ours - 2018) / sigma, for our full
physics and with FairShip's 2018 weighting applied per muon (rule2018.py on eiger: eta / omega Dalitz muons and cascade
eta / eta' dropped). Linear axes.

    python3 bin/plot_match_2018_2d.py results/complete_validation/rule2018_cv_2018exact_c.npz results/complete_validation/compare_refs.npz [REBIN]
Writes match_2018_2d.png (match_2018_2d_rebinN.png with REBIN = N: N x N cells merged) next to the first file.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

P_MIN, P_MAX, PT_MAX = 25.0, 400.0, 3.0
MAX_REL = 0.3                 # cells where either side has a larger relative error are greyed out in the ratio / pull


def main():
    d, r = dict(np.load(sys.argv[1])), np.load(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    pe, te = d["PE"], d["TE"]
    ref = r["ref_minbias_s1"] + r["ref_dimuon_s1"]
    ref2 = r["ref_minbias_s2"] + r["ref_dimuon_s2"]
    if n > 1:                                       # merge n x n cells (statistics of rare sources, e.g. J/psi at high pT)
        cut = lambda a: a[: a.shape[0] // n * n, : a.shape[1] // n * n]
        merge = lambda a: cut(a).reshape(a.shape[0] // n, n, a.shape[1] // n, n).sum((1, 3))
        ref, ref2 = merge(ref), merge(ref2)
        for k in ("all", "rule2018", "all_w2", "rule2018_w2"):
            d[k] = merge(d[k])
        pe, te = pe[: (len(pe) - 1) // n * n + 1: n], te[: (len(te) - 1) // n * n + 1: n]
    dp, dt = np.diff(pe)[:, None], np.diff(te)[None, :]
    dens = lambda h: np.where(h > 0, h / dp / dt, np.nan)          # muons per spill per GeV per GeV
    rows = (("rule2018", "ours with FairShip's 2018 weighting"), ("all", "ours, complete physics"))
    fig, axs = plt.subplots(2, 4, figsize=(26, 11), constrained_layout=True)
    vmax = np.nanmax(dens(ref))
    norm = LogNorm(vmax * 1e-6, vmax)
    for i, (k, lab) in enumerate(rows):
        o, o2 = d[k], d[f"{k}_w2"]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = o / ref
            sig = np.sqrt(o2 + ref2)
            pull = (o - ref) / sig
            bad = ~((o > 0) & (ref > 0) & (np.sqrt(o2) / o < MAX_REL) & (np.sqrt(ref2) / ref < MAX_REL))
        ratio[bad] = np.nan
        pull[bad] = np.nan
        panels = ((dens(ref), "2018 FairShip files (min. bias + dimuon class)", norm, "viridis", "muons / spill / GeV²"),
                  (dens(o), lab, norm, "viridis", "muons / spill / GeV²"),
                  (ratio, f"{lab} / 2018 files", TwoSlopeNorm(1.0, 0.6, 1.4), "RdBu_r", "ratio"),
                  (pull, f"pull ({lab} - 2018) / sigma", TwoSlopeNorm(0.0, -5, 5), "PuOr_r", "pull"))
        if i == 1:
            panels = panels[1:] + ((dens(o) * 0 + 100 * np.sqrt(o2) / np.where(o > 0, o, np.nan),
                                    "ours: relative statistical error", LogNorm(0.1, 100), "magma_r", "%"),)
        for j, (val, title, nm, cmap, cl) in enumerate(panels):
            ax = axs[i, j]
            m = ax.pcolormesh(pe, te, val.T, norm=nm, cmap=cmap, shading="flat")
            ax.set_facecolor("0.85")
            ax.set_xlim(P_MIN, P_MAX); ax.set_ylim(0, PT_MAX)
            ax.set_xlabel("p at production [GeV]"); ax.set_ylabel("pT at production [GeV]"); ax.set_title(title, fontsize=11)
            fig.colorbar(m, ax=ax, label=cl)
        tot = np.nansum(o[np.sqrt(pe[1:] * pe[:-1]) >= P_MIN]) / np.nansum(ref[np.sqrt(pe[1:] * pe[:-1]) >= P_MIN])
        good = ~np.isnan(pull)
        axs[i, 2 if i == 0 else 1].text(0.97, 0.95, f"total > {P_MIN:.0f} GeV: {tot:.3f}\npulls: mean {np.nanmean(pull):.2f}, "
                                        f"rms {np.nanstd(pull):.2f}\n({good.sum()} cells, grey: error > {MAX_REL:.0%})",
                                        transform=axs[i, 2 if i == 0 else 1].transAxes, ha="right", va="top", fontsize=10,
                                        bbox=dict(facecolor="white", alpha=0.85))
    fig.suptitle(f"Pythia 8.223 + Geant4 10.1.p03 (the 2018 software), {float(d['pot']):.2g} p.o.t., vs the 2018 FairShip files "
                 "(no open charm, 2018 selection)", fontsize=14)
    out = Path(sys.argv[1]).with_name("match_2018_2d.png" if n == 1 else f"match_2018_2d_rebin{n}.png")
    fig.savefig(out, dpi=90)
    print(out)


if __name__ == "__main__":
    main()
