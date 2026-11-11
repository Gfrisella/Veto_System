#!/usr/bin/env python3
"""ISSUES D6 (11 Nov 2026): compare muon-DIS libraries before switching (branch veto-d6-energy-rescale).

The same interactions (energies log-uniform 1 - 400 GeV, --n per material, fixed directions) are sampled from each
library, with and without the rescaling to the true energy (fastveto.dis RESCALE). Per energy bin: V0 per interaction
(K_L, K_S, Lambda with p > 1 GeV), charged particles with p > 1 GeV, mean visible energy / E. The new library must agree
with the old one within the statistics (multiplicity, V0 yield); the rescaling should remove the E / E_lib scatter.

    python3 bin/d6_compare_libraries.py --libs data/interactions/dis/mu_library_iron.npz <new library> --material iron
"""
import _bootstrap  # noqa: F401

import argparse
import importlib
import json
import os

import numpy as np

V0 = (130, 310, 3122, -3122)
CHARGED = (11, 13, 211, 321, 2212, 3112, 3222, 3312, 3334)


def run(lib_path, material, E, rescale, seed):
    os.environ["VETO_LIB_RESCALE"] = "1" if rescale else "0"
    from fastveto import dis
    importlib.reload(dis)                                 # RESCALE is read at import
    lib = dis.MuonLibrary(lib_path, split="geant4")
    rng = np.random.default_rng(seed)
    n = len(E)
    u = np.tile([0.0, 0.0, 1.0], (n, 1)); v = np.zeros((n, 3))
    o = lib.sample(material, E, u, v, np.full(n, 13), rng)
    pm = np.linalg.norm(o["p"][:, :3], axis=1)
    a = np.abs(o["pdg"])
    nv0 = np.bincount(o["int"], (np.isin(o["pdg"], V0) | (a == 130)) & (pm > 1), minlength=n)
    nch = np.bincount(o["int"], np.isin(a, CHARGED) & (pm > 1) & ~o["beam"], minlength=n)
    evis = np.bincount(o["int"], np.where(o["beam"], 0.0, o["p"][:, 3]), minlength=n)
    return nv0, nch, evis / E, o["E_lib"] / E


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--libs", nargs="+", required=True)
    ap.add_argument("--material", default="iron")
    ap.add_argument("--n", type=int, default=400_000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(11)
    E = np.exp(rng.uniform(np.log(1.0), np.log(400.0), a.n))
    edges = np.array([1, 3, 10, 30, 100, 400.0])
    ib = np.clip(np.searchsorted(edges, E) - 1, 0, len(edges) - 2)
    rows = []
    for lp in a.libs:
        for resc in (False, True):
            nv0, nch, fvis, elr = run(lp, a.material, E, resc, 5)
            for b in range(len(edges) - 1):
                m = ib == b
                rows.append(dict(lib=lp, rescale=resc, E_lo=edges[b], E_hi=edges[b + 1], n=int(m.sum()),
                                 v0=float(nv0[m].mean()), v0_err=float(nv0[m].std() / np.sqrt(m.sum())),
                                 nch=float(nch[m].mean()), nch_err=float(nch[m].std() / np.sqrt(m.sum())),
                                 evis_over_E=float(fvis[m].mean()), Elib_over_E_rms=float(np.sqrt(((elr[m] - 1) ** 2).mean()))))
    print(f"{'library':<48}{'resc':>5}{'E [GeV]':>12}{'V0 / int (p>1)':>20}{'charged p>1':>18}{'Evis/E':>8}{'Elib/E rms':>11}")
    for r in rows:
        print(f"{os.path.basename(r['lib']):<48}{str(r['rescale'])[0]:>5}{r['E_lo']:>5g}-{r['E_hi']:<6g}"
              f"{r['v0']:>12.4f} ± {r['v0_err']:.4f}{r['nch']:>11.3f} ± {r['nch_err']:.3f}{r['evis_over_E']:>8.3f}{r['Elib_over_E_rms']:>11.3f}")
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
