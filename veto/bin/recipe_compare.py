#!/usr/bin/env python3
"""Which recipe reproduces the 2018 FairShip min. bias muons? Our productions against the min. bias rows (weight
768.75) of the input file, in the file's convention (production momentum; reaching the 2018 plane with E > 10 GeV,
bin/old_convention.py), absolute (muons per spill, 5e13 p.o.t.):
  A  Pythia 6 + Geant4 11 FTFP_BERT   (pool_mb20, first N slices)
  B  Pythia 8 + Geant4 11 FTFP_BERT   (fsm_p8_ftfp)
  C  Pythia 8 + Geant4 11 QGSP_BERT   (fsm_p8_qgsp; the 2018 production: Pythia 8 + Geant4 10 QGSP_BERT_HP_PEN)
  A18, C18  as A, C with the 2018 target (--target SHiP2018: 17 cm less tungsten than ECN3)
  A10, C10  as A18, C18 on Geant4 10.00.p04 (built from source, bin/g4_10.sh) instead of 11.2
Writes <out>/recipe_compare.npz (flux and sum w^2 per (p, pT) for each).

    python3 bin/recipe_compare.py [--slices 20]
"""

import _bootstrap  # noqa: F401

import argparse
import glob
from pathlib import Path

import h5py
import numpy as np

from veto import paths

import build_pool as bp
import old_convention as oc

P_EDGES = np.geomspace(20, 150, 21)
PT_EDGES = np.linspace(0, 4, 21)
RECIPES = {"A": ("/disk/lhcb_data/sqasim/veto_source_pool1/tail", "pool_mb20"),
           "B": ("/disk/lhcb_data/sqasim/fairship_match/tail", "fsm_p8_ftfp"),
           "C": ("/disk/lhcb_data/sqasim/fairship_match/tail", "fsm_p8_qgsp"),
           "A18": ("/disk/lhcb_data/sqasim/fairship_match/tail", "fsm_p6_ftfp_t18"),    # the 2018 target (SHiP2018)
           "C18": ("/disk/lhcb_data/sqasim/fairship_match/tail", "fsm_p8_qgsp_t18"),
           "A10": ("/disk/lhcb_data/sqasim/fairship_match/tail", "fsm_p6_ftfp_t18_g410"),   # Geant4 10.00.p04 (bin/g4_10.sh)
           "C10": ("/disk/lhcb_data/sqasim/fairship_match/tail", "fsm_p8_qgsp_t18_g410"),
           "A18Fe": ("/disk/lhcb_data/sqasim/fairship_match/tail", "fsm_p6_ftfp_t18fe"),     # target inside an iron shield
           "C18Fe": ("/disk/lhcb_data/sqasim/fairship_match/tail", "fsm_p8_qgsp_t18fe")}


def h2(p, pt, w):
    i, j = np.digitize(p, P_EDGES) - 1, np.digitize(pt, PT_EDGES) - 1
    ok = (i >= 0) & (i < len(P_EDGES) - 1) & (j >= 0) & (j < len(PT_EDGES) - 1)
    k = i[ok] * (len(PT_EDGES) - 1) + j[ok]
    n = (len(P_EDGES) - 1) * (len(PT_EDGES) - 1)
    shape = (len(P_EDGES) - 1, len(PT_EDGES) - 1)
    return np.bincount(k, weights=w[ok], minlength=n).reshape(shape), np.bincount(k, weights=w[ok] ** 2, minlength=n).reshape(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", type=int, default=20)
    ap.add_argument("--out", default="/disk/lhcb_data/sqasim/fairship_match")
    a = ap.parse_args()
    table = oc.loss_map()
    out = dict(p_edges=P_EDGES, pt_edges=PT_EDGES)
    with h5py.File(paths.enriched_input_h5(), "r") as f:
        w = f["weight"][:]
        comp = bp.components(w, f["source"][:])
        m = comp == 1
        px, py, pz = (f[k][:][m].astype(np.float64) for k in ("px", "py", "pz"))
    pt = np.hypot(px, py)
    out["file_s1"], out["file_s2"] = h2(np.sqrt(pt ** 2 + pz ** 2), pt, w[m].astype(np.float64))
    del w, comp, m, px, py, pz, pt
    for key, (d, tag) in RECIPES.items():
        files = sorted(glob.glob(f"{d}/{tag}_[0-9]*.npz"))[:a.slices]
        s1 = s2 = 0
        events = 0.0
        for fn in files:
            x = np.load(fn)
            events += float(x["events"])
            p = np.sqrt(x["px"].astype(np.float64) ** 2 + x["py"] ** 2 + x["pz"] ** 2)
            pp = x["p_prod"].astype(np.float64)
            sel = pp - oc.draw_loss(table, pp) > 10.0                # the file's selection
            s = pp / p                                               # momentum at production along the direction
            w = x["w_decay"].astype(np.float64) / float(x["rep"])
            a1, a2 = h2(pp[sel], np.hypot(x["px"][sel], x["py"][sel]) * s[sel], w[sel])
            s1, s2 = s1 + a1, s2 + a2
        norm = 5e13 / events
        out[f"{key}_s1"], out[f"{key}_s2"] = s1 * norm, s2 * norm ** 2
        print(f"{key} ({tag}): {len(files)} slices, {events:.3g} p.o.t., flux 30-150 GeV ratio to file "
              f"{(s1 * norm)[P_EDGES[:-1] >= 30].sum() / out['file_s1'][P_EDGES[:-1] >= 30].sum():.3f}", flush=True)
    np.savez(Path(a.out) / "recipe_compare.npz", **out)
    print("wrote", Path(a.out) / "recipe_compare.npz")


if __name__ == "__main__":
    main()
