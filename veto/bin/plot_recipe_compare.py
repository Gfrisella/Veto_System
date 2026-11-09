#!/usr/bin/env python3
"""Plots of bin/recipe_compare.py: our recipes A, B, C against the 2018 file's min. bias, per GeV and in pT.

    python3 bin/plot_recipe_compare.py results/pool_v3/recipe_compare.npz
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"file": "#555555", "A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "A18": "#2a78d6", "C18": "#1baf7a"}
STY = {"A18": "--", "C18": "--"}
LAB = {"file": "2018 FairShip file (Pythia 8 + Geant4 10, QGSP_BERT_HP_PEN)", "A": "A: Pythia 6 + Geant4 11 FTFP_BERT (ours)",
       "B": "B: Pythia 8 + Geant4 11 FTFP_BERT", "C": "C: Pythia 8 + Geant4 11 QGSP_BERT",
       "A18": "A': as A, 2018 target (dashed)", "C18": "C': as C, 2018 target (dashed)"}
KEYS = [k for k in ("file", "A", "B", "C", "A18", "C18")]
h = np.load(sys.argv[1])
pe, te = h["p_edges"], h["pt_edges"]
pc, tc = np.sqrt(pe[1:] * pe[:-1]), 0.5 * (te[1:] + te[:-1])
fig, axs = plt.subplots(2, 4, figsize=(22, 9), gridspec_kw=dict(height_ratios=[3, 2]))
ref = h["file_s1"]
# column 0: momentum spectrum (pT integrated)
ax, bx = axs[0, 0], axs[1, 0]
for k in KEYS:
    if f"{k}_s1" not in h.files:
        continue
    s1, s2 = h[f"{k}_s1"].sum(1), h[f"{k}_s2"].sum(1)
    ax.errorbar(pc, s1 / np.diff(pe), yerr=np.sqrt(s2) / np.diff(pe), color=COL[k], lw=2, ls=STY.get(k, "-"), label=LAB[k])
    if k != "file":
        r = s1 / ref.sum(1)
        bx.errorbar(pc, r, yerr=r * np.sqrt(s2 / s1 ** 2 + h["file_s2"].sum(1) / ref.sum(1) ** 2), color=COL[k], lw=2, ls=STY.get(k, "-"), marker="o", ms=4)
ax.set_yscale("log"); ax.set_xscale("log"); ax.set_ylabel("muons per spill per GeV"); ax.set_title("momentum at production (all pT)")
ax.legend(fontsize=8); bx.set_xscale("log"); bx.set_xlabel("p [GeV]"); bx.set_ylabel("ours / 2018 file"); bx.axhline(1, color="k")
# columns 1-3: pT shape in three momentum ranges, each normalised to unit area (shape only)
for c, (lo, hi) in zip((1, 2, 3), ((20, 40), (40, 80), (80, 150))):
    m = (pc >= lo) & (pc < hi)
    ax, bx = axs[0, c], axs[1, c]
    f1 = ref[m].sum(0)
    for k in KEYS:
        if f"{k}_s1" not in h.files:
            continue
        s1, s2 = h[f"{k}_s1"][m].sum(0), h[f"{k}_s2"][m].sum(0)
        ax.errorbar(tc, s1 / s1.sum(), yerr=np.sqrt(s2) / s1.sum(), color=COL[k], lw=2, ls=STY.get(k, "-"), label=LAB[k] if c == 1 else None)
        if k != "file":
            with np.errstate(divide="ignore", invalid="ignore"):
                r = (s1 / s1.sum()) / (f1 / f1.sum())
            bx.plot(tc, r, color=COL[k], lw=2, ls=STY.get(k, "-"), marker="o", ms=4)
    ax.set_yscale("log"); ax.set_title(f"pT shape, p = {lo}-{hi} GeV (area 1)"); ax.set_xlabel("pT [GeV]")
    bx.axhline(1, color="k"); bx.set_ylim(0.3, 2.5); bx.set_xlabel("pT [GeV]"); bx.set_ylabel("shape ratio ours / file")
for ax in axs.flat:
    ax.grid(alpha=0.3)
fig.suptitle("Fossil hunt: can a different generator reproduce the 2018 min. bias muons? (Geant4 11 for A, B, C; same selection as the file)", fontsize=12)
fig.tight_layout()
out = Path(sys.argv[1]).with_name("recipe_compare.png")
fig.savefig(out, dpi=100)
print(out)
