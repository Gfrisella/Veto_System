#!/usr/bin/env python3
"""Figures and numbers of restart_test.py (sandbox): per restarted state, survival to SST1, arrival position and class vs the
injected angle; the stochastic spread at zero offset in 'mrad equivalent' (RMS / slope of the mean arrival vs offset).
    python3 plot_restart.py restart.npz restart"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load(sys.argv[1])
A, meta = d["arrivals"], json.loads(str(d["meta"]))
out = sys.argv[2]
NAME = {0: "common", 1: "low occ.", 2: "isolated"}
groups = {}
for i, m in enumerate(meta):
    groups.setdefault((m["traj"], m["z"]), []).append(i)
keys = sorted(groups, key=lambda k: (-[m for m in meta if m["traj"] == k[0]][0]["cls"], k[0], k[1]))
lines = ["state (SST1 class of the original track, checkpoint, p, charge): survival at 0 offset; mean |shift| of the arrival per "
         "mrad (x, y offsets); stochastic RMS at 0 offset; RMS in mrad equivalent = RMS / shift per mrad"]
fig, axs = plt.subplots(len(keys), 4, figsize=(26, 3.6 * len(keys)), constrained_layout=True)
for row, key in enumerate(keys):
    idx = groups[key]
    m0 = meta[idx[0]]
    base = [i for i in idx if meta[i]["offset"] == 0][0]
    a0 = A[base]
    ok0 = np.isfinite(a0[:, 0])
    rms0 = np.hypot(np.nanstd(a0[:, 0]), np.nanstd(a0[:, 1])) if ok0.sum() > 2 else np.nan
    slopes = {}
    for axis in ("x", "y"):
        sel = sorted([i for i in idx if meta[i]["axis"] == axis or meta[i]["offset"] == 0], key=lambda i: meta[i]["offset"])
        off = np.array([meta[i]["offset"] for i in sel], float)
        surv = np.array([np.isfinite(A[i][:, 0]).mean() for i in sel])
        mx = np.array([np.nanmean(A[i][:, 0]) if np.isfinite(A[i][:, 0]).sum() else np.nan for i in sel])
        my = np.array([np.nanmean(A[i][:, 1]) if np.isfinite(A[i][:, 1]).sum() else np.nan for i in sel])
        sx = np.array([np.nanstd(A[i][:, 0]) if np.isfinite(A[i][:, 0]).sum() > 2 else np.nan for i in sel])
        rare = np.array([np.mean(A[i][:, 3] >= 1) for i in sel])
        small = np.abs(off) <= 5
        good = small & np.isfinite(mx) & np.isfinite(my)
        if good.sum() >= 3:
            gx, gy = np.polyfit(off[good], mx[good], 1)[0], np.polyfit(off[good], my[good], 1)[0]
            slopes[axis] = np.hypot(gx, gy)
        c = "#2a78d6" if axis == "x" else "#c77700"
        axs[row, 0].plot(off, surv, "o-", color=c, label=f"{axis} offset")
        axs[row, 1].errorbar(off, mx, yerr=sx, fmt="o-", color=c, capsize=2, label=f"mean x at SST1, {axis} offset")
        axs[row, 1].plot(off, my, "s--", color=c, lw=1, label=f"mean y at SST1, {axis} offset")
        axs[row, 2].plot(off, rare, "o-", color=c, label=f"{axis} offset")
    for i in idx:
        mm = meta[i]
        if mm["offset"] in (0, 10, -10, 20, -20) and (mm["axis"] == "x" or mm["offset"] == 0):
            a = A[i]
            axs[row, 3].scatter(a[:, 0], a[:, 1], s=3, alpha=0.5, label=f"x {mm['offset']:+d} mrad")
    axs[row, 3].add_patch(plt.Rectangle((-2, -3), 4, 6, fill=False, ec="k", lw=1))
    axs[row, 3].set_xlim(-12, 12); axs[row, 3].set_ylim(-12, 12); axs[row, 3].set_aspect("equal")
    axs[row, 3].legend(fontsize=7, markerscale=3)
    lab = f"{NAME[m0['cls']]} track {m0['traj']}\nrestart at {m0['z']} m\np {m0['p']:.0f} GeV, q {m0['q']:+d}"
    axs[row, 0].set_ylabel(lab, fontsize=9)
    for j, t in enumerate(("survival to SST1", "mean arrival (+- RMS) vs offset [m]", "fraction arriving in low/isolated regions",
                           "SST1 arrivals: 0, +-10, +-20 mrad (x)")):
        if row == 0:
            axs[row, j].set_title(t)
    for j in range(3):
        axs[row, j].set_xlabel("injected angle offset [mrad]"); axs[row, j].grid(alpha=0.3)
    axs[row, 0].set_ylim(-0.05, 1.05); axs[row, 0].legend(fontsize=7); axs[row, 1].legend(fontsize=6)
    s = np.nanmean(list(slopes.values())) if slopes else np.nan
    lines.append(f"{NAME[m0['cls']]:9s} track {m0['traj']:8d} at {m0['z']:>5s} m, p {m0['p']:6.1f} GeV, q {m0['q']:+d}: survival "
                 f"{ok0.mean():.2f}; shift {s * 1e3 if np.isfinite(s) else np.nan:7.1f} mm per mrad; RMS {rms0:5.2f} m; "
                 f"RMS = {rms0 / s if s else np.nan:6.1f} mrad equivalent")
fig.suptitle("Restarts from checkpoint states with injected angle offsets (500 continuations each, unchanged physics): "
             "sensitivity tests, not beam probabilities", fontsize=14)
fig.savefig(out + ".png", dpi=55)
open(out + ".txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
