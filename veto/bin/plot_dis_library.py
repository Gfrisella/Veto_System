"""Plots of the muon interaction library (bin/dis_library.py) and the hybrid against pure Geant4.

    bin/lcg.sh python3 bin/plot_dis_library.py [--lib data/dis/mu_library.npz] [--out results/dis_library]

1. cross sections per nucleon vs E: Geant4 total (Kokoulin), Geant4 W < 2 per material, Pythia 6 p / n, hybrid
2. at 10, 30, 100, 300 GeV (helium): energy transfer nu and W of the hybrid (Pythia + Geant4 W < 2) vs Kokoulin,
   charged particles with p > 1 GeV per interaction, momentum of the leading charged hadron
"""

import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from veto import paths

from fastveto import dis, kokoulin

CHARGED = (11, 13, 211, 321, 2212, 3112, 3222, 3312, 3334)
M_N = 0.9389


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=str(paths.interactions_data() / "dis/mu_library.npz"))
    ap.add_argument("--out", default=str(paths.results_dir() / "dis_library"))
    ap.add_argument("--split", default="crossover")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lib = dis.MuonLibrary(a.lib, split=a.split)
    E = np.geomspace(0.5, 500, 200)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].plot(E, kokoulin.sigma(E, A=4) / 4, "k-", lw=2, label="Geant4 total, helium (Kokoulin, nu > 0.2 GeV)")
    ax[0].plot(E, kokoulin.sigma(E, nu_min=2.0, A=1), "k:", label="Kokoulin nu > 2 GeV, nucleon")
    ax[0].plot(lib.E, lib.p6_sigma[0], "o-", color="crimson", ms=3, label="Pythia 6 part, p")
    ax[0].plot(lib.E, lib.p6_sigma[1], "s-", color="orange", ms=3, label="Pythia 6 part, n")
    for m, c in zip(lib.g4, plt.cm.viridis(np.linspace(0, 0.9, len(lib.g4)))):
        ax[0].plot(lib.E, lib.g4[m], "-", color=c, lw=1, label=f"Geant4 part, {m}")
    ax[0].plot(E, lib.sigma("helium", E), "b-", lw=2.5, alpha=0.6, label="hybrid, helium")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_ylim(1e-3, 50)
    ax[0].set_xlabel("muon energy [GeV]"); ax[0].set_ylabel("cross section per nucleon [microbarn]"); ax[0].legend(fontsize=7)
    ax[1].plot(E, lib.sigma("helium", E) / (kokoulin.sigma(E, A=4) / 4), "b-", label="helium")
    for m in ("pvc", "mylar", "silicon", "pvt"):
        A = {"pvc": 20.8, "mylar": 12.7, "silicon": 28.1, "pvt": 11.7}[m]
        ax[1].plot(E, lib.sigma(m, E) / (kokoulin.sigma(E, A=A) / A), label=m)
    ax[1].axhline(1, color="k", lw=0.8); ax[1].set_xscale("log"); ax[1].set_ylim(0, 1.5)
    ax[1].set_xlabel("muon energy [GeV]"); ax[1].set_ylabel("hybrid / Geant4 total"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out / f"cross_sections_{a.split}.png", dpi=100); plt.close(fig)

    rng = np.random.default_rng(3)
    fig, axs = plt.subplots(4, 4, figsize=(22, 18))
    for row, E0 in enumerate((10.0, 30.0, 100.0, 300.0)):
        n = 200000
        o = lib.sample("helium", np.full(n, E0), np.tile([0, 0, 1.0], (n, 1)), np.zeros((n, 3)), np.full(n, 13), rng)
        beam = o["beam"]
        e_mu = np.zeros(n); e_mu[o["int"][beam]] = o["p"][beam, 3]
        pmu = np.zeros((n, 3)); pmu[o["int"][beam]] = o["p"][beam, :3]
        el = o["E_lib"]
        nu = el - e_mu
        q2 = 2 * (el * e_mu - el * pmu[:, 2])
        w = np.sqrt(np.maximum(M_N ** 2 + 2 * M_N * nu - q2, 0))
        sig = lib.sigma("helium", np.array([E0]))[0]
        comp = o["component"]
        nb = np.geomspace(0.2, E0, 40)
        ax = axs[row, 0]
        for c, lab, col in ((None, "hybrid", "b"), (0, "Pythia part", "crimson"), (1, "Geant4 part", "green")):
            m = np.ones(n, bool) if c is None else comp == c
            h, _ = np.histogram(nu[m], nb)
            ax.stairs(h / n * sig / np.diff(nb), nb, color=col, label=lab)
        kk = np.array([kokoulin.sigma(E0, nu_min=x, A=4)[0] / 4 for x in nb])
        ax.stairs(-np.diff(kk) / np.diff(nb), nb, color="k", ls="--", label="Geant4 total (Kokoulin)")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("nu [GeV]"); ax.set_ylabel("dsigma/dnu [ub/GeV]")
        ax.set_title(f"E = {E0:g} GeV, helium"); ax.legend(fontsize=7)
        ax = axs[row, 1]
        wb = np.linspace(0.9, min(25, np.sqrt(2 * M_N * E0) + 1), 60)
        ax.set_title(f"nu_x = {np.interp(np.log(E0), lib.logE, lib.nu_x):.1f} GeV") if a.split == "crossover" else None
        for c, lab, col in ((0, "Pythia", "crimson"), (1, "Geant4 part", "green")):
            h, _ = np.histogram(w[comp == c], wb)
            ax.stairs(h / n * sig, wb, color=col, label=lab)
        ax.set_xlabel("W [GeV]"); ax.set_ylabel("ub / bin"); ax.legend(fontsize=7)
        ch = np.isin(np.abs(o["pdg"]), CHARGED) & ~beam
        pp = np.linalg.norm(o["p"][:, :3], axis=1)
        nch = np.bincount(o["int"][ch & (pp > 1)], minlength=n)
        ax = axs[row, 2]
        for c, lab, col in ((0, "Pythia", "crimson"), (1, "Geant4 part", "green")):
            h = np.bincount(nch[comp == c], minlength=12)[:12]
            ax.step(np.arange(12), h / n * sig, where="mid", color=col, label=lab)
        ax.set_yscale("log"); ax.set_xlabel("charged hadrons with p > 1 GeV"); ax.set_ylabel("ub"); ax.legend(fontsize=7)
        lead = np.zeros(n); np.maximum.at(lead, o["int"][ch], pp[ch])
        ax = axs[row, 3]
        lb = np.geomspace(0.05, E0, 40)
        for c, lab, col in ((0, "Pythia", "crimson"), (1, "Geant4 part", "green")):
            h, _ = np.histogram(lead[comp == c], lb)
            ax.stairs(h / n * sig, lb, color=col, label=lab)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("leading charged hadron p [GeV]"); ax.set_ylabel("ub / bin")
        ax.legend(fontsize=7)
        print(f"E={E0:g}: sigma {sig:.3f} ub (Pythia {np.mean(comp == 0):.3f}); >=1 charged hadron p>1: "
              f"Pythia part {np.mean(nch[comp == 0] >= 1):.3f}, Geant4 part {np.mean(nch[comp == 1] >= 1):.3f}; "
              f"sigma(>=1 charged p>1 GeV) = {sig * np.mean(nch >= 1):.3f} ub of which Pythia {np.sum((nch >= 1) & (comp == 0)) / max(np.sum(nch >= 1), 1):.3f}")
    fig.tight_layout(); fig.savefig(out / f"final_states_{a.split}.png", dpi=80); plt.close(fig)
    print("plots in", out)


if __name__ == "__main__":
    main()
