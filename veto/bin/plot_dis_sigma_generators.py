#!/usr/bin/env python3
"""Muon-nucleon cross section per nucleon in iron vs muon energy for the DIS libraries (generator comparison 2026-11-06):
Geant4 11.2.2 / 10.7.3 muonNuclear (Kokoulin), Pythia 6 (W > 2, all Q^2), Pythia 6 with Q^2 > 1, Pythia 8.230 DIS (Q^2 > 1, W > 2).
Iron per nucleon = Z/A sigma_p + (1 - Z/A) sigma_n. -> results/shield_dis/dis_sigma_generators.{png,pdf}"""
import _bootstrap  # noqa: F401
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastveto import materials as mat
sys.path.insert(0, "/disk/users/gfrise/.claude-config/skills/cern-ship-slides/theme")
from plotstyle import save
D = "data/interactions/dis/"
za = mat.MATERIALS["iron"].z_over_a
base, g107 = np.load(D + "mu_library_iron.npz"), np.load(D + "mu_library_iron_g4v1073.npz")
p6q, p8 = np.load(D + "mu_library_iron_p6q2.npz"), np.load(D + "mu_library_iron_p8dis.npz")
E = base["E_grid"]
fe = lambda s: za * s[0] + (1 - za) * s[1]
fig, ax = plt.subplots(figsize=(8, 5.5))
for y, lab, ls in ((base["g4_iron_sigma"], "Geant4 11.2 muonNuclear (all)", "-"), (g107["g4_iron_sigma"], "Geant4 10.7 muonNuclear (all)", "--"),
                   (fe(base["p6_sigma"]), r"Pythia 6, W > 2 GeV, all $Q^2$", "-"), (fe(p6q["p6_sigma"]), r"Pythia 6, $Q^2$ > 1 GeV$^2$", "-"),
                   (fe(p8["p6_sigma"]), r"Pythia 8 DIS, $Q^2$ > 1 GeV$^2$", "-")):
    m = y > 0
    ax.plot(E[m], y[m], ls, label=lab, lw=2)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"$E_\mu$ [GeV]"); ax.set_ylabel(r"$\sigma$ per nucleon in iron [$\mu$b]")
ax.set_xlim(2, 500); ax.legend(fontsize=11); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); save(fig, "results/shield_dis/dis_sigma_generators.png")
for ie in (20, 30, 40, 49):
    print(f"E={E[ie]:6.1f}: G4 {base['g4_iron_sigma'][ie]:.2f}  P6 {fe(base['p6_sigma'])[ie]:.2f}  P6Q2>1 {fe(p6q['p6_sigma'])[ie]:.3f}  P8 {fe(p8['p6_sigma'])[ie]:.3f} ub")
