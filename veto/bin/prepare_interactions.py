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
from pathlib import Path
import time

import numpy as np

from run_interactions import DATASET, load_npz
from veto import design as dsg
from veto import interactions as ix
from veto import io, paths, reco


def merge_prepared(parts):
    """Concatenate prepared dicts (event and track indices shifted; interactions15 summed per code)."""
    if len(parts) == 1:
        return parts[0]
    out = {"codes": parts[0]["codes"], "interactions15": sum(p["interactions15"] for p in parts)}
    n_ev = 0
    for k in parts[0]:
        if k in ("codes", "interactions15"):
            continue
        vals = []
        n_ev = 0
        for p in parts:
            v = p.get(k)
            if v is None:
                continue
            if k in ("track_event", "strip_hit_event"):
                v = v + n_ev
            vals.append(v)
            n_ev += len(p["event_weight15"])
        out[k] = np.concatenate(vals)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=float, nargs=3, default=[5e6, 5e6, 1e6], help="helium pvc air")
    ap.add_argument("--nu", default=str(paths.apply_file("nu", "D6_straw")))
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--dataset", default=DATASET, help="muon dataset (parents of the muon interactions)")
    ap.add_argument("--tag", default="", help="cache names prepared_<kind><tag>.npz")
    ap.add_argument("--ms7", nargs="*", default=[], help="MS7 event stores (muon V0 chain, MS7 neutrinos): prepared_ms7<tag>.npz")
    ap.add_argument("--only", nargs="*", default=["muon", "nu", "ms7"], help="which caches to build")
    ap.add_argument("--muon-npz", default=None, help="a bin/run_interactions.py output (e.g. the re-run S5 file) instead of a new "
                                                    "fastveto draw; its design-free targets are used, parents from --dataset")
    args = ap.parse_args()
    t0 = time.time()
    out_dir = paths.results_dir() / "interactions"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = reco.RecoConfig(multiple_scattering=True)
    store = paths.store_dir(args.dataset)
    muons, meta = io.open_store(store)

    from fastveto import dis
    lib = dis.MuonLibrary(str(paths.mu_library()), split="crossover")
    draws = {"helium": args.draws[0], "wall": args.draws[1], "air": args.draws[2], "air_upstream": args.draws[2]}
    if "muon" in args.only and args.muon_npz:
        inter, part = load_npz(args.muon_npz)
    elif "muon" in args.only:
        design = dsg.load("R1_ubt_only")             # any design: helium / air / wall do not depend on it
        inter, part, _ = dis.run(muons, design, lib, draws, np.random.default_rng(args.seed),
                                 include=("helium", "air", "wall", "air_upstream"), workers=args.workers)
    if "muon" in args.only:
        print(f"muons: {len(inter['weight']):,} interactions, {len(part['pdg']):,} particles, {time.time() - t0:.0f} s", flush=True)
        prep = ix.prepare(inter, part, cfg, muons)
        np.savez(out_dir / f"prepared_muon{args.tag}.npz", **prep)
        print(f"  {len(prep['event_weight15']):,} selected events, {prep['event_weight15'].sum():.3g} per 15 years before the veto")
    if "nu" in args.only:
        inter, part = load_npz(args.nu)
        prep = ix.prepare(inter, part, cfg, None)
        np.savez(out_dir / f"prepared_nu{args.tag}.npz", **prep)
        print(f"neutrinos: {len(prep['event_weight15']):,} selected events, {prep['event_weight15'].sum():.3g} per 15 years "
              f"before the veto, {time.time() - t0:.0f} s")
    if "ms7" in args.only and args.ms7:              # MS7 V0 chain + MS7 neutrinos (event stores), codes 3 / 4
        from veto import evstore as es
        parts = []
        for s in args.ms7:
            _, ev, trs = es.read(s)
            inter, part = es.to_fastveto(ev, trs)
            del ev, trs
            pr = ix.prepare(inter, part, cfg, None, codes=(3, 4))
            if "muon" in Path(s).name and "strip_parent_row" in pr:   # parents = MS7 muons (rows of paths.ms7_parents)
                pr["strip_parent_row"] = pr["event_group"].astype(np.int64)
            parts.append(pr)
            print(f"MS7 {s}: {len(parts[-1]['event_weight15']):,} selected events, {parts[-1]['event_weight15'].sum():.3g} "
                  f"per 15 years before the veto, {time.time() - t0:.0f} s", flush=True)
        np.savez(out_dir / f"prepared_ms7{args.tag}.npz", **merge_prepared(parts))


if __name__ == "__main__":
    main()
