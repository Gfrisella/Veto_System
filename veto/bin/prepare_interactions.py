#!/usr/bin/env python3
"""Cache the design-independent interaction backgrounds for the objective: selected events, ready for any veto.

Muons: fastveto.dis.run on the muon dataset for the helium, the air and the PVC only (library split crossover),
with --draws interactions per target. Neutrinos: a bin/nu_dis_apply.py file (--nu, its helium / air / PVC rows).
Both through the selection of RecoConfig (measured IP < 0.1 m, at most 2 tracks in SST1); the tracks of the
selected events are stored, so bin/evolve and run_reco only run the veto per design (interactions.veto).

  GPU_HOST=farm-gpu5 bin/remote.sh run bin/prepare_interactions.py            # ~5 min, 40 workers
"""

import _bootstrap  # noqa: F401

import argparse
import time

import numpy as np

from run_interactions import DATASET, load_npz
from veto import design as dsg
from veto import interactions as ix
from veto import io, paths, reco


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=float, nargs=3, default=[5e6, 5e6, 1e6], help="helium pvc air")
    ap.add_argument("--nu", default=str(paths.apply_file("nu", "D6_straw")))
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--dataset", default=DATASET, help="muon dataset (parents of the muon interactions)")
    ap.add_argument("--tag", default="", help="cache names prepared_<kind><tag>.npz")
    args = ap.parse_args()
    t0 = time.time()
    out_dir = paths.results_dir() / "interactions"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = reco.RecoConfig(multiple_scattering=True)
    store = paths.store_dir(args.dataset)
    muons, meta = io.open_store(store)

    from fastveto import dis
    lib = dis.MuonLibrary(str(paths.mu_library()), split="crossover")
    draws = {"helium": args.draws[0], "pvc": args.draws[1], "air": args.draws[2]}
    design = dsg.load("D0_minimal")                  # any design: helium / air / PVC do not depend on it
    inter, part, _ = dis.run(muons, design, lib, draws, np.random.default_rng(args.seed),
                             include=("helium", "air", "pvc"), workers=args.workers)
    print(f"muons: {len(inter['weight']):,} interactions, {len(part['pdg']):,} particles, {time.time() - t0:.0f} s", flush=True)
    prep = ix.prepare(inter, part, cfg, muons)
    np.savez(out_dir / f"prepared_muon{args.tag}.npz", **prep)
    print(f"  {len(prep['event_weight15']):,} selected events, {prep['event_weight15'].sum():.3g} per 15 years before the veto")

    inter, part = load_npz(args.nu)
    prep = ix.prepare(inter, part, cfg, None)
    np.savez(out_dir / f"prepared_nu{args.tag}.npz", **prep)
    print(f"neutrinos: {len(prep['event_weight15']):,} selected events, {prep['event_weight15'].sum():.3g} per 15 years "
          f"before the veto, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
