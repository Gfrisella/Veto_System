#!/usr/bin/env python3
"""EM debris library: what leaves a slab with a muon, full Geant4 (cpp/g4slab.cc, FTFP_BERT_EMZ), for resampling onto
the muons of a dataset (the shield exit, the stations, walls). One process per configuration:

  iron_B0 / iron_B17p / iron_B17m   0.5 m iron (~28 X0: equilibrium), no field / B_y = 1.7 T with mu- / mu+ (the last
                                    magnet is magnetised: low-energy electrons curl up inside it)
  scint, straw, steel               1 cm plastic scintillator, a straw layer (2 x 36 um Mylar + 1 cm Ar), 3 cm steel
Muon energies ENERGIES, incidence angles (about y) ANGLES. Every particle crossing the exit face (kinetic > 100 keV)
is stored in the MUON FRAME: offset from the muon's exit point and direction, both rotated so the muon's exit direction
is +z (the sampler rotates them onto any muon), with kinetic energy, pdg, delay; per event, so whole events are drawn.

    bin/g4.sh python3 bin/debris_library.py --config iron_B0 --angle 0.05      # one job
    bin/g4.sh python3 bin/debris_library.py --list                             # the job list
Writes results/debris_library/<config>_a<angle>.npz.
"""

import _bootstrap  # noqa: F401

import argparse
import sys
import time

import numpy as np

from veto import paths

ENERGIES = (2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 70.0, 100.0, 150.0, 200.0, 300.0)   # GeV total
CONFIGS = {   # name: (layers, B [T], muon pdg, events per energy, angles)
    "iron_B0": ([("G4_Fe", 0.5)], 0.0, 13, 20000, (0.0, 0.02, 0.05, 0.1, 0.2)),
    "iron_B17m": ([("G4_Fe", 0.5)], 1.7, 13, 20000, (0.0, 0.02, 0.05, 0.1, 0.2)),
    "iron_B17p": ([("G4_Fe", 0.5)], 1.7, -13, 20000, (0.0, 0.02, 0.05, 0.1, 0.2)),
    "scint": ([("G4_PLASTIC_SC_VINYLTOLUENE", 0.01)], 0.0, 13, 50000, (0.0, 0.1)),
    "straw": ([("G4_MYLAR", 36e-6), ("G4_Ar", 0.0099), ("G4_MYLAR", 36e-6)], 0.0, 13, 50000, (0.0, 0.1)),
    "steel": ([("G4_STAINLESS-STEEL", 0.03)], 0.0, 13, 50000, (0.0, 0.1)),
}
KEEP = 1e-4                       # GeV kinetic
OUT = paths.results_dir() / "debris_library"


def to_muon_frame(d):
    """Per event: secondaries relative to the muon at the exit face (offset, direction), muon direction -> +z."""
    mu = d["track"] == 1
    ev_mu = d["ev"][mu]
    idx = {e: i for i, e in enumerate(ev_mu)}
    sec = (~mu) & np.isin(d["ev"], ev_mu) & (d["ekin"] > KEEP)
    j = np.array([idx[e] for e in d["ev"][sec]], dtype=np.int64)
    m = np.stack([d["px"][mu], d["py"][mu], d["pz"][mu]], 1)
    u = m / np.linalg.norm(m, axis=1, keepdims=True)                    # muon direction at the exit
    # rotation taking u to +z (Rodrigues), per event
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(u, z)
    s = np.linalg.norm(v, axis=1)
    c = u @ z
    K = np.zeros((len(u), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0], K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -v[:, 2], v[:, 1], v[:, 2], -v[:, 0], -v[:, 1], v[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(s > 1e-12, (1 - c) / s ** 2, 0.0)
    Rm = np.eye(3)[None] + K + f[:, None, None] * (K @ K)
    R = Rm[j]
    off = np.stack([d["x"][sec] - d["x"][mu][j], d["y"][sec] - d["y"][mu][j], np.zeros(sec.sum())], 1)
    p = np.stack([d["px"][sec], d["py"][sec], d["pz"][sec]], 1)
    return dict(ev=d["ev"][sec].astype(np.int32), pdg=d["pdg"][sec].astype(np.int32), ekin=d["ekin"][sec].astype(np.float32),
                off=np.einsum("nij,nj->ni", R, off).astype(np.float32), p=np.einsum("nij,nj->ni", R, p).astype(np.float32),
                dt=(d["t"][sec] - d["t"][mu][j]).astype(np.float32),
                mu_out=len(ev_mu), mu_p=np.linalg.norm(m, axis=1).astype(np.float32))


def run(name, angle, seed):
    sys.path.insert(0, str(paths.cpp_build()))
    import g4slab
    layers, B, pdg, n, _ = CONFIGS[name]
    g4slab.initialize(seed, layers, [0.0], 0.7, B)
    out = {}
    for E in ENERGIES:
        t0 = time.time()
        d = {k: np.asarray(v) for k, v in g4slab.shoot(pdg, E, n, angle).items()}
        r = to_muon_frame(d)
        for k in ("ev", "pdg", "ekin", "off", "p", "dt"):
            out[f"E{E:g}_{k}"] = r[k]
        out[f"E{E:g}_mu_out"] = r["mu_out"]
        out[f"E{E:g}_mu_p"] = r["mu_p"]
        ch = (np.abs(r["pdg"]) == 11) & (r["ekin"] > 1e-3)
        print(f"{name} angle {angle:g} E {E:5g} GeV: {n} muons ({r['mu_out']} out), {len(np.unique(r['ev'][ch])) / n:.3f} with e+- > 1 MeV, "
              f"{(r['pdg'] == 22).sum() / n:.2f} gamma/mu, {time.time() - t0:.0f} s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{name}_a{angle:g}.npz", energies=np.array(ENERGIES), n=n, field=B, mu_pdg=pdg, angle=angle,
             layers=str(layers), **out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", choices=list(CONFIGS))
    ap.add_argument("--angle", type=float, default=0.0)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--seed", type=int, default=20261031)
    a = ap.parse_args()
    if a.list:
        for name, c in CONFIGS.items():
            for ang in c[4]:
                print(name, ang)
        return
    run(a.config, a.angle, a.seed + 1000 * list(CONFIGS).index(a.config) + int(round(a.angle * 1000)))


if __name__ == "__main__":
    main()
