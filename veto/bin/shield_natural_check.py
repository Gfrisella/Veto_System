#!/usr/bin/env python3
"""Cross-check of the forced MS7 DIS chain with plain (unforced) Geant4: muons of data/shield_dis/muons_iron.npz are
started at the entrance of the last 50 cm of iron (z = 29.39 m, straight line from the stored plane, mean dE/dx) and
transported by g4shield (FTFP_BERT: muonNuclear, photonuclear of radiated photons, everything) to z = 29.90 m.
Counts what leaves the iron per spill (p > 1 GeV) to compare with results/shield_dis/chain_probs.json 'out_per_spill'.

    bin/g4.sh python3 bin/shield_natural_check.py [--n 4000000 --workers 32]   -> results/shield_dis/natural_check.json
"""
import _bootstrap  # noqa: F401
import argparse
import json

import numpy as np

import shield_dis as sd

SPECIES = {"K_L": [130], "K_S": [310], "Lambda": [3122, -3122], "pi+-": [211, -211], "K+-": [321, -321],
           "p": [2212], "n": [2112]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    par = sd.parents("muon")
    N = len(par["p"])
    rows = rng.choice(N, size=min(a.n, N), replace=False)
    s_in = par["s_in"][rows]
    u = par["u"][rows]
    v = np.stack([par["x"][rows], par["y"][rows], par["z"][rows]], 1) + u * (s_in[:, None] + 1e-4)
    E = par["p"][rows] - sd.si.DEDX_MU * s_in
    ok = E > 0.5
    rows, u, v, E = rows[ok], u[ok], v[ok], E[ok]
    scale = N / len(rows)                              # each sampled row stands for N / n rows
    w = par["weight"][rows] * scale
    part = dict(int=np.arange(len(rows)), pdg=par["pdg"][rows], p=np.concatenate([u * E[:, None], E[:, None]], 1), v=v)
    print(f"{len(rows):,} muons of {N:,} rows ({par['weight'].sum():.4g} per spill), scale {scale:.3g}", flush=True)
    g = sd.transport(part, len(rows), a.workers, 20000, a.seed)
    g = {k: np.asarray(x) for k, x in g.items()}
    pm = np.sqrt(g["px"] ** 2 + g["py"] ** 2 + g["pz"] ** 2)
    W = w[g["ev"]]
    chain = json.load(open(sd.OUT / "chain_probs.json"))["species"]
    res = dict(muons_sampled=int(len(rows)), rows=N, species={})
    print(f"{'species':8s} {'natural /spill':>15s} {'± (Σw²)':>10s} {'MC':>7s} {'forced chain':>13s} {'ratio':>6s}")
    for name, codes in SPECIES.items():
        m = np.isin(g["pdg"], codes) & (pm > 1)
        r = float(W[m].sum())
        e = float(np.sqrt((np.bincount(g["ev"][m], W[m]) ** 2).sum()))
        c = chain[name]["out_per_spill"]
        res["species"][name] = dict(natural_per_spill=r, err=e, mc=int(m.sum()), forced_chain_per_spill=c)
        print(f"{name:8s} {r:15.4g} {e:10.3g} {m.sum():7d} {c:13.4g} {r / c if c else 0:6.2f}")
    json.dump(res, open(sd.OUT / "natural_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
