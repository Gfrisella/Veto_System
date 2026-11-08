#!/usr/bin/env python3
"""Maps of expected muon crossings over the experiment's lifetime per region, at every checkpoint plane (sandbox).

    python3 lifetime_maps.py run.npz --out lifetime_maps [--spills 1.5e7 --region 1.0 --title "..."]

Colour: log10 of the expected crossings in `spills` spills (15 years: 6e20 p.o.t. / 4e13) per REGION x REGION m.
Regions without a simulated muon: grey, shaded by their 95 % upper limit (3 x the weight one simulated muon carries at
that plane x spills), hatched. White lines: decay-vessel window (z <= 30.6 m) or SST1 aperture (z = 82 m); dashed: cavern
floor (y = -3.36 m) and wall (x = -4.56 m).
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run"); ap.add_argument("--out", required=True)
    ap.add_argument("--spills", type=float, default=1.5e7); ap.add_argument("--region", type=float, default=1.0)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    d = np.load(a.run)
    Z, XY = d["checkpoints"], d["XY"]
    r = int(round(a.region / (XY[1] - XY[0])))
    e = XY[::r]
    B = d["xy_s1"].shape[1]
    fig, axs = plt.subplots(3, 5, figsize=(30, 17), constrained_layout=True)
    norm = Normalize(0, 16)
    gnorm = Normalize(6, 16)
    for k, z in enumerate(Z):
        s1 = d["xy_s1"][k].reshape(B, len(e) - 1, r, len(e) - 1, r).sum((2, 4))
        n = d["xy_n"][k].reshape(B, len(e) - 1, r, len(e) - 1, r).sum((2, 4)).sum(0)
        meas = n > 0
        N = s1.mean(0) * a.spills
        w_row = np.median(s1.sum(0)[meas] / n[meas]) / B if meas.any() else np.nan
        ul = 3.0 * w_row * a.spills
        ax = axs.ravel()[k]
        im = ax.pcolormesh(e, e, np.where(meas, np.log10(np.maximum(N, 1e-30)), np.nan).T, cmap="viridis", norm=norm)
        gray = np.where(~meas, np.log10(ul), np.nan)
        ax.pcolormesh(e, e, gray.T, cmap="Greys", norm=gnorm, hatch="//", alpha=0.6)
        if z <= 30.6:
            ax.add_patch(plt.Rectangle((-0.5, -1.35), 1.0, 2.7, fill=False, ec="w", lw=1.2))
        else:
            ax.add_patch(plt.Rectangle((-2, -3), 4, 6, fill=False, ec="w", lw=1.2))
        ax.axhline(-3.36, color="w", lw=0.8, ls="--"); ax.axvline(-4.56, color="w", lw=0.8, ls="--")
        ax.set_aspect("equal"); ax.set_title(f"z = {z:g} m: {meas.sum()} regions measured, {(~meas).sum()} empty "
                                             f"(UL {ul:.1g})", fontsize=11)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    for ax in axs.ravel()[len(Z):]:
        ax.axis("off")
    cb = fig.colorbar(im, ax=axs.ravel()[len(Z)], fraction=0.5)
    cb.set_label(f"log10 expected crossings per {a.region:g} m region in {a.spills:g} spills (15 years)", fontsize=12)
    axs.ravel()[len(Z)].text(0.0, 0.05, "grey, hatched: no simulated muon;\nshade = 95 % upper limit\n"
                             "(same log10 scale, 6 .. 16)", fontsize=12, transform=axs.ravel()[len(Z)].transAxes)
    fig.suptitle(f"Expected muon crossings over the lifetime ({a.spills:g} spills), {a.region:g} m regions"
                 + (f": {a.title}" if a.title else ""), fontsize=16)
    fig.savefig(a.out + ".png", dpi=60)
    print(a.out + ".png")


if __name__ == "__main__":
    main()
