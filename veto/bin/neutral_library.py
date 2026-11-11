#!/usr/bin/env python3
"""Library of neutral-hadron interactions in the vessel helium (ISSUES D4; docs/STATUS.md 2026-11-10 "rock neutrons"):
one forced inelastic interaction per event (cpp/g4hadlib, Geant4 11.2 FTFP_BERT, G4_He at 1 atm), final states only,
in the frame of the incoming hadron along +z. Species n, anti-n, K_L on a log energy grid; with the unbiased inelastic
cross section per metre of helium at each energy (the forcing weight of bin/evstore_neutral.py).

    bin/g4.sh python3 bin/neutral_library.py --n 20000 --out /disk/lhcb_data/sqasim/neutral_lib/neutral_he.npz
Output: E (GeV, total), species, xs (n_species, n_E) [1/m], ev_species, ev_ie, ev_first, ev_n (one row per interacted
event), x_pdg, x_px, x_py, x_pz, x_E (products, frame of the hadron).
"""
import _bootstrap  # noqa: F401
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from veto import paths

SPECIES = (2112, -2112, 130)
E_GRID = np.geomspace(1.5, 300.0, 24)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20000, help="events per species and energy")
    ap.add_argument("--seed", type=int, default=11); ap.add_argument("--material", default="G4_He")
    ap.add_argument("--species", default=",".join(map(str, SPECIES))); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(paths.cpp_build()))
    import g4hadlib
    g4hadlib.initialize(seed=a.seed, material=a.material)
    species = [int(s) for s in a.species.split(",")]
    xs = np.array([[g4hadlib.cross_section(q, E) for E in E_GRID] for q in species])
    print("inelastic cross section per metre of", a.material, "(interaction length km):")
    for q, row in zip(species, xs):
        print(f"  {q:>6}: " + " ".join(f"{E:.3g}:{1e-3 / x:.2f}" for E, x in zip(E_GRID, row)), flush=True)
    t0 = time.time()
    ev = {k: [] for k in ("species", "ie", "first", "n")}
    X = {k: [] for k in ("pdg", "px", "py", "pz", "E")}
    n_tot = 0
    for js, q in enumerate(species):
        for ie, E in enumerate(E_GRID):
            r = g4hadlib.generate(q, float(E), a.n)
            done = np.asarray(r["events"]["done"]).astype(bool)
            pr = {k: np.asarray(v) for k, v in r["products"].items()}
            ids = np.nonzero(done)[0]
            keep = done[pr["ev"]]
            order = np.argsort(pr["ev"][keep], kind="stable")
            evk = pr["ev"][keep][order]
            cnt = np.bincount(evk, minlength=len(done))[ids]
            first = n_tot + np.concatenate([[0], np.cumsum(cnt)[:-1]])
            ev["species"].append(np.full(len(ids), q)); ev["ie"].append(np.full(len(ids), ie))
            ev["first"].append(first); ev["n"].append(cnt)
            for k in X:
                X[k].append(pr[k][keep][order])
            n_tot += int(cnt.sum())
            print(f"  {q:>6} E {E:7.3g} GeV: {done.mean():.3f} interacted, {cnt.mean():.1f} products, "
                  f"{time.time() - t0:.0f} s", flush=True)
    out = {f"ev_{k}": np.concatenate(v) for k, v in ev.items()} | {f"x_{k}": np.concatenate(v) for k, v in X.items()}
    meta = dict(material=a.material, n=a.n, seed=a.seed, geant4="11.2.2 FTFP_BERT (cpp/g4hadlib)", frame="hadron along +z")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, E=E_GRID, species=np.array(species), xs=xs, meta=json.dumps(meta), **out)
    print(f"-> {a.out}: {len(out['ev_n']):,} events, {n_tot:,} products ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
