#!/usr/bin/env python3
"""Validation of the forced MS7 muon-DIS chain with NATURAL Geant4 (ISSUES D3, 2026-11-09).

Muons of data/shield_dis/muons_iron.npz (the chain's parents, per spill of 4e13 p.o.t.) are drawn in proportion to
their weight (each then carries W / n), started at the entrance of the last 50 cm of MS7 iron (straight line from the
stored plane, mean dE/dx, as bin/shield_natural_check.py) and transported by cpp/g4shieldnat (same geometry and field
as the chain's g4shield) with the muonNuclear cross section x --mu-bias (products weight 1 / bias) and the physics list
of --physics-list (FTFP_BERT = our chain, QGSP_BERT_HP_PEN = FairShip). What crosses z = 29.90 m with p > 1 GeV is
counted per species and origin (1 = from a muonNuclear interaction: what the chain simulates; 2 = from the muon's
other secondaries, e.g. photonuclear of radiated photons: not in the chain) and compared with the chain runs.

    bin/g4ship.sh /usr/bin/python3 bin/shield_natural.py --physics-list QGSP_BERT_HP_PEN --mu-bias 10 --n 20000000
        -> results/shield_dis/natural_<list>_b<bias>.json
"""
import _bootstrap  # noqa: F401
import argparse
import json
import time

import numpy as np

import shield_dis as sd
from veto.config import POT_PER_SPILL

SPECIES = {"K_L": [130], "K_S": [310], "Lambda": [3122, -3122], "pi+-": [211, -211], "K+-": [321, -321],
           "p": [2212], "n": [2112]}
CHAIN = {"G4 10.7 (FTFP_BERT)": "shield_muon_fs2026_geant4v1073", "G4 11.2 (FTFP_BERT)": "shield_muon_fs2026_geant4only",
         "Pythia 6": "shield_muon_fs2026", "P8 PP+DIS": "shield_muon_fs2026_p8full"}


def chain_exit(tag):
    js = json.load(open(sd.OUT / f"{tag}.json"))
    f = POT_PER_SPILL / js.get("pot_per_spill", 5e13)
    return {k: js["exit"][k]["per_spill"] * f for k in SPECIES if k in js["exit"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20_000_000, help="muons drawn (proportional to weight)")
    ap.add_argument("--physics-list", default="QGSP_BERT_HP_PEN")
    ap.add_argument("--mu-bias", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    par = sd.parents("muon")
    W = par["weight"].astype(np.float64)
    rows = rng.choice(len(W), size=a.n, p=W / W.sum())
    s_in, u = par["s_in"][rows], par["u"][rows]
    v = np.stack([par["x"][rows], par["y"][rows], par["z"][rows]], 1) + u * (s_in[:, None] + 1e-4)
    E = par["p"][rows] - sd.si.DEDX_MU * s_in
    ok = E > 0.5
    w_mu = W.sum() / a.n                                     # per drawn muon, per spill
    rows, u, v, E = rows[ok], u[ok], v[ok], E[ok]
    part = dict(int=np.arange(len(rows)), pdg=par["pdg"][rows], p=np.concatenate([u * E[:, None], E[:, None]], 1), v=v)
    print(f"{len(rows):,} muons drawn ({W.sum():.4g} per spill, {w_mu:.3g} each), list {a.physics_list}, "
          f"muonNuclear x {a.mu_bias:g}", flush=True)
    g = sd.transport(part, len(rows), a.workers, a.chunk, a.seed,
                     natural=dict(physics_list=a.physics_list, mu_bias=a.mu_bias))
    g = {k: np.asarray(x) for k, x in g.items()}
    pm = np.sqrt(g["px"] ** 2 + g["py"] ** 2 + g["pz"] ** 2)
    wt = g["w"] * w_mu
    chains = {k: chain_exit(t) for k, t in CHAIN.items()}
    res = dict(physics_list=a.physics_list, mu_bias=a.mu_bias, muons=int(len(rows)), pot_per_spill=POT_PER_SPILL,
               seconds=round(time.time() - t0), species={})
    head = f"{'species':8s} {'muNuclear /spill':>17s} {'± stat':>9s} {'MC':>7s} {'other origin':>13s}  " + \
           "  ".join(f"{k:>20s}" for k in CHAIN)
    print(head)
    for name, codes in SPECIES.items():
        row = {}
        for origin, lab in ((1, "mu_nuclear"), (2, "other")):
            m = np.isin(g["pdg"], codes) & (pm > 1) & (g["origin"] == origin)
            r = float(wt[m].sum())
            e = float(np.sqrt((np.bincount(g["ev"][m], wt[m]) ** 2).sum())) if m.any() else 0.0
            row[lab] = dict(per_spill=r, err=e, mc=int(m.sum()))
        row["chain"] = {k: c.get(name) for k, c in chains.items()}
        res["species"][name] = row
        mn = row["mu_nuclear"]
        print(f"{name:8s} {mn['per_spill']:17.4g} {mn['err']:9.3g} {mn['mc']:7d} {row['other']['per_spill']:13.4g}  " +
              "  ".join(f"{(c or 0):20.4g}" for c in row["chain"].values()), flush=True)
    out = sd.OUT / f"natural_{a.physics_list}_b{a.mu_bias:g}.json"
    json.dump(res, open(out, "w"), indent=1)
    print(f"wrote {out}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
