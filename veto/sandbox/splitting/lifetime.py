#!/usr/bin/env python3
"""Expected muon crossings over the experiment's lifetime per region, at every checkpoint plane (sandbox).

    python3 lifetime.py run.npz --out lifetime [--spills 1.5e7 --threshold 1e-4 --region 1.0]

lifetime: 6e20 p.o.t. / 4e13 per spill = 1.5e7 spills (15 years). Per region (REGION x REGION m) and plane:
  measured   N = rate x spills > 0, with its error (batch spread);  95 % upper limit N + 1.645 err
  empty      no simulated muon: 95 % upper limit = 3.0 x (weight one simulated muon carries there) x spills; the weight is
             the median row weight of that plane (analog runs: rows carry the source weights)
Writes <out>.png (histograms of log10 lifetime crossings per plane, the threshold) and <out>.txt.
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run"); ap.add_argument("--out", required=True)
    ap.add_argument("--spills", type=float, default=1.5e7); ap.add_argument("--threshold", type=float, default=1e-4)
    ap.add_argument("--region", type=float, default=1.0)
    a = ap.parse_args()
    d = np.load(a.run)
    Z, XY = d["checkpoints"], d["XY"]
    r = int(round(a.region / (XY[1] - XY[0])))
    B = d["xy_s1"].shape[1]
    lines = [f"{a.run}: lifetime {a.spills:g} spills, threshold {a.threshold:g} crossings, {a.region:g} m regions over "
             f"+-{XY[-1]:g} m", f"{'z [m]':>6s} {'measured':>9s} {'empty':>6s} {'lifetime N: median':>19s} {'min':>9s} "
             f"{'empty UL: min':>14s} {'median':>9s}   below threshold: measured / certified-empty"]
    fig, axs = plt.subplots(3, 5, figsize=(26, 13), constrained_layout=True)
    bins = np.arange(-8, 18.5, 0.5)
    for k, z in enumerate(Z):
        s1 = d["xy_s1"][k].reshape(B, -1, r, d["xy_s1"].shape[-1] // r, r).sum((2, 4))
        n = d["xy_n"][k].reshape(B, -1, r, d["xy_n"].shape[-1] // r, r).sum((2, 4)).sum(0)
        mean, err = s1.mean(0), s1.std(0, ddof=1) / np.sqrt(B)
        meas = n > 0
        w_row = np.median((s1.sum(0)[meas] / n[meas])) / B if meas.any() else np.nan   # what one muon contributes
        N = mean * a.spills
        ul = np.where(meas, (mean + 1.645 * err) * a.spills, 3.0 * w_row * a.spills)
        lines.append(f"{z:6.2f} {meas.sum():9d} {(~meas).sum():6d} {np.median(N[meas]):19.3g} {N[meas].min():9.3g} "
                     f"{ul[~meas].min() if (~meas).any() else np.nan:14.3g} {np.median(ul[~meas]) if (~meas).any() else np.nan:9.3g}"
                     f"   {(ul[meas] < a.threshold).sum()} / {(ul[~meas] < a.threshold).sum()}")
        if k < axs.size:
            ax = axs.ravel()[k]
            ax.hist(np.log10(N[meas]), bins, color="#2a78d6", label="measured: expected crossings")
            if (~meas).any():
                ax.hist(np.log10(ul[~meas]), bins, color="#c77700", alpha=0.8, label="empty: 95 % upper limit")
            ax.axvline(np.log10(a.threshold), color="#c0392b", lw=2, ls="--", label=f"threshold {a.threshold:g}")
            ax.set_yscale("log"); ax.set_title(f"z = {z:g} m", fontsize=12)
            ax.set_xlabel(f"log10 crossings per {a.region:g} m region in {a.spills:g} spills")
            if k == 0:
                ax.legend(fontsize=9)
    for ax in axs.ravel()[len(Z):]:
        ax.axis("off")
    fig.suptitle(f"Expected muon crossings over the lifetime ({a.spills:g} spills) per {a.region:g} m region, at each checkpoint",
                 fontsize=15)
    fig.savefig(a.out + ".png", dpi=70)
    open(a.out + ".txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
