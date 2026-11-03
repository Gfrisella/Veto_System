#!/usr/bin/env python3
"""Step-by-step probabilities of the muon-DIS chain in the last 50 cm of MS7, per species, on one consistent sample:

  1  P_int(mu)        a muon crossing the iron interacts there: P = n sigma(E) L  (rate-weighted mean, and at fixed E)
  2  multiplicity     particles of the species made per interaction (library final state, p > 1 GeV)
  3  P_out            they reach the shield end (z = 29.90 m): Geant4 FTFP_BERT through the remaining iron
                      primary = the particle itself unchanged; all = any particle of the species at the plane
                      (includes those made in the hadronic cascade inside the iron)
  4  P_dec            a particle at the plane decays inside the vessel (z = 30.6 - 80.6 m), straight line, c tau
  5  per spill        rate = S x multiplicity x P_out x P_dec

    bin/g4.sh python3 bin/shield_chain_probs.py [--n 40000 --workers 32]   -> results/shield_dis/chain_probs.json
"""
import _bootstrap  # noqa: F401
import argparse
import json

import numpy as np

import shield_dis as sd                       # bin/ is on sys.path; a real module so the spawned workers can import it
si, geo = sd.si, sd.geo

SPECIES = {"K_L": [130], "K_S": [310], "Lambda": [3122, -3122], "pi+-": [211, -211], "K+-": [321, -321],
           "p": [2212], "n": [2112], "mu": [13, -13]}
CTAU = {130: 15.34, 310: 0.026844, 3122: 0.0789, 211: 7.8045, 321: 3.712, 13: 658.64}      # m
MASS = {130: 0.497611, 310: 0.497611, 3122: 1.115683, 211: 0.13957, 321: 0.493677, 13: 0.105658, 2212: 0.938272,
        2112: 0.939565}


def p_decay_vessel(pdg, p3, z0):
    """straight line from z0 along p3: probability to decay between the vessel entrance and exit (0 if stable)"""
    a = np.abs(pdg)
    out = np.zeros(len(a))
    for c, ct in CTAU.items():
        m = a == c
        if not m.any():
            continue
        pm = np.linalg.norm(p3[m], axis=1)
        uz = p3[m, 2] / pm
        lam = pm / MASS[c] * ct
        d1 = np.maximum(geo.Z_VESSEL_MIN - z0[m], 0) / uz
        d2 = np.maximum(geo.Z_VESSEL_MAX - z0[m], 0) / uz
        out[m] = np.where(uz > 0, np.exp(-d1 / lam) - np.exp(-d2 / lam), 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--mu-lib", help="muon DIS library (default: shield_dis.MU_LIB)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    par = sd.parents("muon")
    inter, part, summ = sd.draw("muon", par, a.n, rng, mu_lib=a.mu_lib)
    S = summ["interactions_per_spill"]
    w_int = inter["weight"]                                   # S / N each (0 if dropped)
    # 1. interaction probability
    from fastveto import dis
    lib = dis.MuonLibrary(str(a.mu_lib or sd.MU_LIB))
    P_mean = S / par["weight"].sum()
    P_E = {f"{E:g} GeV": float(si.nucleons_per_m3() * lib.sigma("iron", np.array([E]))[0] * sd.UB * 0.5)
           for E in (5.0, 10.0, 30.0, 100.0, 300.0)}
    # 2.-3. transport
    g = sd.transport(part, len(w_int), a.workers, 1000, a.seed)
    gp = {k: np.asarray(v) for k, v in g.items()}
    pm0 = np.linalg.norm(part["p"][:, :3], axis=1)
    pm1 = np.sqrt(gp["px"] ** 2 + gp["py"] ** 2 + gp["pz"] ** 2)
    w0, w1 = w_int[part["int"]], w_int[gp["ev"]]
    prim = gp["primary"] == 1
    p3 = np.stack([gp["px"], gp["py"], gp["pz"]], 1)
    Pdec = p_decay_vessel(gp["pdg"], p3, np.full(len(p3), si.Z_PLANE))
    res = dict(n_interactions=int((w_int > 0).sum()), interactions_per_spill=S, muons_through_iron_per_spill=float(par["weight"].sum()),
               P_int_mean=P_mean, P_int_50cm_at_E=P_E, species={})
    print(f"muons through the last 50 cm: {par['weight'].sum():.3g} /spill; interactions S = {S:.3g} /spill; "
          f"P_int = {P_mean:.3g} (mean); at E: " + ", ".join(f"{k} {v:.2e}" for k, v in P_E.items()))
    print(f"{'species':8s} {'made/int':>9s} {'made/spill':>11s} {'P_out prim':>10s} {'P_out all':>10s} {'out/spill':>10s} "
          f"{'P_dec ves':>9s} {'dec/spill':>10s}")
    for name, codes in SPECIES.items():
        m0 = np.isin(part["pdg"], codes) & (pm0 > 1)
        m1 = np.isin(gp["pdg"], codes) & (pm1 > 1)
        made = w0[m0].sum()
        out_all = w1[m1].sum()
        out_prim = w1[m1 & prim].sum()
        dec = (w1[m1] * Pdec[m1]).sum()
        r = dict(made_per_interaction=made / S, made_per_spill=made, mc_made=int(m0.sum()),
                 P_out_primary=out_prim / made if made else 0.0, P_out_all=out_all / made if made else 0.0,
                 out_per_spill=out_all, mc_out=int(m1.sum()),
                 P_decay_in_vessel=dec / out_all if out_all else 0.0, decays_in_vessel_per_spill=dec)
        res["species"][name] = r
        print(f"{name:8s} {r['made_per_interaction']:9.3g} {made:11.3g} {r['P_out_primary']:10.3g} {r['P_out_all']:10.3g} "
              f"{out_all:10.3g} {r['P_decay_in_vessel']:9.3g} {dec:10.3g}   (MC {r['mc_made']} made, {r['mc_out']} out)")
    out = sd.OUT / f"chain_probs{a.tag}.json"
    json.dump(res, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
