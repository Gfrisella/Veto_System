#!/usr/bin/env python3
"""The plain C++ debris model (cpp/emdebris) against the Geant4 library (bin/debris_library.py), iron exit face:
P(e+- > 1 MeV), multiplicities, e+- and photon spectra, e+- angle to the muon and distance at the exit face; time per muon.

    python3 bin/debris_model_compare.py --lib results/debris_library --exe cpp/emdebris/emdebris
Writes results/debris_library/model_vs_geant4.png and prints a table.
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REC = np.dtype([("ev", "<i4"), ("pdg", "<i4"), ("ekin", "<f4"), ("ox", "<f4"), ("oy", "<f4"), ("oz", "<f4"),
                ("px", "<f4"), ("py", "<f4"), ("pz", "<f4"), ("dt", "<f4")])
ENERGIES = (2, 5, 10, 20, 50, 100, 300)
DEPTH = 50.0                    # cm of iron simulated by the model before the exit face (Geant4: 50 cm)


def run_model(exe, E, theta, n, seed):
    with tempfile.NamedTemporaryFile(suffix=".bin") as f:
        r = subprocess.run([exe, str(E), str(theta), str(n), str(seed), f.name, str(DEPTH)], capture_output=True, text=True, check=True)
        raw = open(f.name, "rb").read()
    us = float(r.stderr.split(",")[-1].split()[0])
    rec = np.frombuffer(raw[4:], dtype=REC)
    return dict(ev=rec["ev"], pdg=rec["pdg"], ekin=rec["ekin"], off=np.stack([rec["ox"], rec["oy"], rec["oz"]], 1),
                p=np.stack([rec["px"], rec["py"], rec["pz"]], 1)), us


def from_lib(d, E):
    k = f"E{E:g}_"
    return dict(ev=d[k + "ev"], pdg=d[k + "pdg"], ekin=d[k + "ekin"], off=d[k + "off"], p=d[k + "p"])


def metrics(x, n):
    ap = np.abs(x["pdg"])
    e = (ap == 11) & (x["ekin"] > 1e-3)
    g = (x["pdg"] == 22) & (x["ekin"] > 1e-4)
    ang = np.arccos(np.clip(x["p"][e, 2] / np.maximum(np.linalg.norm(x["p"][e], axis=1), 1e-12), -1, 1))
    r = np.hypot(x["off"][e, 0], x["off"][e, 1])
    return dict(p_e=len(np.unique(x["ev"][e])) / n, n_e=e.sum() / n, n_g=g.sum() / n, e_ekin=x["ekin"][e], g_ekin=x["ekin"][g],
                ang=ang, r=r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="results/debris_library")
    ap.add_argument("--exe", default="cpp/emdebris/emdebris")
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()
    lib = Path(a.lib)
    rows, hist = [], {}
    for theta in (0.0, 0.1):
        d = np.load(lib / f"iron_B0_a{theta:g}.npz")
        n_g4 = int(d["n"])
        for E in ENERGIES:
            g4 = metrics(from_lib(d, E), n_g4)
            mo, us = run_model(a.exe, E, theta, a.n, 1000 + E)
            mm = metrics(mo, a.n)
            rows.append((theta, E, g4, mm, us))
            hist[(theta, E)] = (g4, mm)
    print(f"{'theta':>6}{'E':>6} | {'P(e>1MeV) G4 / model':>22} | {'e+-/mu G4 / model':>19} | {'gamma/mu G4 / model':>21} | "
          f"{'e+- angle med G4/model':>23} | {'r med [mm] G4/model':>20} | {'model us/mu':>11}")
    for th, E, g, m, us in rows:
        print(f"{th:6.2f}{E:6d} | {g['p_e']:9.3f} / {m['p_e']:<9.3f} | {g['n_e']:7.3f} / {m['n_e']:<8.3f} | {g['n_g']:8.2f} / {m['n_g']:<9.2f} | "
              f"{np.median(g['ang']):9.3f} / {np.median(m['ang']):<10.3f} | {1e3 * np.median(g['r']):8.2f} / {1e3 * np.median(m['r']):<8.2f} | {us:11.1f}")
    fig, axs = plt.subplots(2, 4, figsize=(22, 9))
    for row, th in enumerate((0.0, 0.1)):
        Es = [E for E in ENERGIES]
        g4p = [hist[(th, E)][0]["p_e"] for E in Es]
        mop = [hist[(th, E)][1]["p_e"] for E in Es]
        g4g = [hist[(th, E)][0]["n_g"] for E in Es]
        mog = [hist[(th, E)][1]["n_g"] for E in Es]
        ax = axs[row, 0]
        ax.plot(Es, g4p, "o-", color="#555555", lw=2, label="Geant4: P(e+- > 1 MeV)")
        ax.plot(Es, mop, "s--", color="#2a78d6", lw=2, label="C++ model: P(e+- > 1 MeV)")
        ax.plot(Es, np.array(g4g) / 20, "o-", color="#555555", lw=1, alpha=0.5, label="Geant4: photons/mu / 20")
        ax.plot(Es, np.array(mog) / 20, "s--", color="#eb6834", lw=1, label="C++ model: photons/mu / 20")
        ax.set_xscale("log"); ax.set_xlabel("muon energy [GeV]"); ax.set_title(f"iron exit, {1000 * th:.0f} mrad"); ax.legend(fontsize=8)
        for c, (key, bins, xl) in enumerate((("e_ekin", np.geomspace(1e-3, 10, 50), "e+- kinetic energy [GeV]"),
                                             ("g_ekin", np.geomspace(1e-4, 10, 50), "photon energy [GeV]"),
                                             ("ang", np.linspace(0, np.pi / 2, 46), "e+- angle to the muon [rad]")), start=1):
            ax = axs[row, c]
            for E, col in ((5, "#1baf7a"), (50, "#2a78d6"), (300, "#eb6834")):
                g, m = hist[(th, E)]
                ax.hist(g[key], bins=bins, weights=np.full(len(g[key]), 1 / 20000), histtype="step", color=col, lw=2, label=f"Geant4 {E} GeV")
                ax.hist(m[key], bins=bins, weights=np.full(len(m[key]), 1 / a.n), histtype="step", color=col, lw=2, ls="--", label=f"model {E} GeV")
            if key != "ang":
                ax.set_xscale("log")
            ax.set_yscale("log"); ax.set_xlabel(xl); ax.set_ylabel("per muon"); ax.legend(fontsize=7)
    for ax in axs.flat:
        ax.grid(alpha=0.3)
    fig.suptitle("EM debris leaving iron behind a muon: plain C++ model (dashed) vs full Geant4 (solid)", fontsize=12)
    fig.tight_layout()
    fig.savefig(lib / "model_vs_geant4.png", dpi=100)
    print("wrote", lib / "model_vs_geant4.png")


if __name__ == "__main__":
    main()
