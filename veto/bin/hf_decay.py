#!/usr/bin/env python3
"""Charm / beauty hadrons of bin/charm_cascade.py --hadrons, decayed and transported by tailtransport's complete (or
analog) mode, as FairShip's charm / beauty production: each hadron starts at its interaction point, Pythia 8 decays it
(muonic channels forced, the other products tracked by Geant4, chains included), muons recorded at z = -0.0045 m.

    bin/g4.sh python3 bin/hf_decay.py --tag charmc --slice K [--mode complete] [--target SHiP2018] [--physics QGSP_BERT]
Reads data/tail/<tag>_had_<K>.npz, writes data/tail/<tag>_mu_<K>.npz (tail_produce.py layout + hf, ch, pcx, pcy, pcz;
w_decay = hadron weight per proton x muon weight; events = protons of the slice).
"""

import _bootstrap  # noqa: F401

import argparse
import json
import os
import sys
import time

import numpy as np

from veto import paths

sys.path.insert(0, os.environ.get("TAILTRANSPORT_BUILD", str(paths.PROJECT_DIR / "cpp/tailgen/build")))
from tail_produce import OUT  # noqa: E402

EMIN = 20.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="charmc")
    ap.add_argument("--slice", type=int, required=True)
    ap.add_argument("--mode", default="complete", choices=["complete", "analog"])
    ap.add_argument("--target", default="SHiP2018")
    ap.add_argument("--physics", default="QGSP_BERT")
    ap.add_argument("--emin", type=float, default=EMIN, help="threshold [GeV]: particles followed, muons kept")
    ap.add_argument("--split", type=int, default=1, help="muonic decays per prompt decay (weight / split)")
    ap.add_argument("--muon-window", default=None, help="npz (p_edges, pt_edges, w_min): weight window of the recorded muons")
    a = ap.parse_args()
    import tailtransport as T
    t0 = time.time()
    h = np.load(OUT / f"{a.tag}_had_{a.slice}.npz")
    n = len(h["pdg"])
    T.initialize(seed=8000 + a.slice, target=a.target, physics=a.physics, mode=a.mode)
    if a.split > 1:
        T.set_split(a.split)
    if a.muon_window:                                 # weight window of the recorded muons (full weights, as w_decay)
        ww = np.load(a.muon_window)
        T.set_muon_window(ww["p_edges"].tolist(), ww["pt_edges"].tolist(), ww["w_min"].ravel().tolist())
    # the hadron weights go in as primary weights, so the muon weights (and the weight window) are the full ones
    o = T.cascade(list(range(n + 1)), h["pdg"].astype(int).tolist(), h["px"].tolist(), h["py"].tolist(), h["pz"].tolist(),
                  h["x"].tolist(), h["y"].tolist(), h["z"].tolist(), a.emin, a.emin, 0.0, [], h["w"].astype(float).tolist())
    diag = o.pop("diag", {})
    o = {k: np.asarray(v) for k, v in o.items() if np.ndim(v) > 0}
    w = o["w"]
    np.savez(OUT / f"{a.tag}_mu_{a.slice}.npz", events=float(h["events"]), rep=1, emin=a.emin, pmin=a.emin,
             pdg=o["q"].astype(int), x=o["x"], y=o["y"], z=o["z"], px=o["px"], py=o["py"], pz=o["pz"], w_decay=w,
             parent=o["parent"], z_decay=o["z_decay"], gen=o["gen"], event=o["ev"], z_int=h["z"][o["ev"]],
             p_prod=o["p_create"], hf=o["hf"], ch=o["ch"], pcx=o["pcx"], pcy=o["pcy"], pcz=o["pcz"],
             had_pdg=h["pdg"][o["ev"]], mode=a.mode, diag=json.dumps([diag]))
    print(f"{a.tag} slice {a.slice}: {n} hadrons -> {len(w)} muons at the plane, {w.sum() / float(h['events']) * 5e13:.4g} "
          f"per spill; {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
