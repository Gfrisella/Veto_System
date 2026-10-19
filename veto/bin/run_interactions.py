#!/usr/bin/env python3
"""Interaction backgrounds of one design: fast-veto-sim's forced interactions through our selection and veto.

  bin/remote.sh run bin/run_interactions.py --design E2_llm_run2            # muons (generated if needed)
  bin/remote.sh run bin/run_interactions.py --design D6_straw --kind nu     # neutrinos (fast-veto-sim npz)

Muons: the interactions come from fast-veto-sim's data/dis/apply_<design>.npz if it exists (--interactions to give
another file), else fastveto.dis.run is called on the muon dataset (library split "crossover", --workers forked
processes). Neutrinos: data/genie/apply_<design>.npz (fast-veto-sim scripts/nu_dis_apply.py). Selection and veto:
lib/veto/interactions.py, with the full reconstruction (multiple scattering with tails, helium and air).
Writes results/interactions/<design>_<kind>.json and prints the table per target.
"""

import _bootstrap  # noqa: F401

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from run_reco import load_inputs
from veto import design as dsg
from veto import interactions as ix
from veto import io, paths, reco

FVS = Path("/disk/users/sqasim/workspace/uzh/fast-veto-sim")
DATASET = "design-optimization-dataset-gamma-2026-10-17"


def load_npz(path):
    f = np.load(path)
    inter = {k[4:]: f[k] for k in f.files if k.startswith("int_")}
    part = {k[5:]: f[k] for k in f.files if k.startswith("part_")}
    return inter, part


def generate_muon_interactions(design, store, args):
    """fastveto.dis.run on the muon dataset for this design (our veto package is already loaded, so fastveto
    uses it, not its own copy)."""
    sys.path.insert(0, str(FVS))
    from fastveto import dis
    lib = dis.MuonLibrary(str(FVS / "data/dis/mu_library.npz"), split=args.split)
    meta = json.load(open(store / "meta.json"))
    cols = {k: np.load(store / f"{k}.npy", mmap_mode="r") for k in meta["columns"]}
    draws = {"helium": args.draws[0], "pvc": args.draws[1], "air": args.draws[2], "planes": args.draws[3]}
    inter, part, _ = dis.run(cols, design, lib, draws, np.random.default_rng(args.seed), workers=args.workers)
    return inter, part


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", required=True, help="designs/<name>.json or a path")
    ap.add_argument("--kind", choices=["muon", "nu"], default="muon")
    ap.add_argument("--interactions", default=None, help="fast-veto-sim npz (default: its apply_<design>.npz)")
    ap.add_argument("--split", default="crossover", help="muon library: crossover | geant4 | w2")
    ap.add_argument("--draws", type=float, nargs=4, default=[1e6, 1e6, 2e5, 2e5], help="helium pvc air planes")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t0 = time.time()
    p = Path(args.design)
    design = dsg.load(p if p.exists() else args.design)
    store = paths.store_dir(args.dataset)
    sub = "dis" if args.kind == "muon" else "genie"
    # muons: generated fresh by default (fast-veto-sim's older apply files predate its exact PVC chords)
    npz = Path(args.interactions) if args.interactions else (None if args.kind == "muon" else FVS / f"data/{sub}/apply_{design.name}.npz")
    if npz is not None and npz.exists():
        inter, part = load_npz(npz)
        source = str(npz)
    elif args.kind == "muon":
        inter, part = generate_muon_interactions(design, store, args)
        source = f"fastveto.dis.run (split {args.split}, draws {args.draws})"
    else:
        raise SystemExit(f"no {npz}: run fast-veto-sim scripts/nu_dis_apply.py --design {design.name}")
    if args.kind == "muon" and not args.interactions:
        save = paths.results_dir() / "interactions" / f"{design.name}_muon_interactions.npz"
        save.parent.mkdir(parents=True, exist_ok=True)
        np.savez(save, **{f"int_{k}": v for k, v in inter.items()}, **{f"part_{k}": v for k, v in part.items()})
    print(f"{len(inter['weight']):,} interactions, {len(part['pdg']):,} particles ({source}), {time.time() - t0:.0f} s", flush=True)

    muons = None
    if args.kind == "muon":
        cols, _ = io.open_store(store)
        muons = cols
    _, maps = load_inputs()
    cfg = reco.RecoConfig(seed=args.seed, multiple_scattering=True)
    res = ix.evaluate(inter, part, design, maps, cfg, np.random.default_rng(args.seed), muons=muons)
    print(f"selection and veto: {time.time() - t0:.0f} s\n")
    print(f"{'target':<20}{'interactions/15y':>17}{'selected':>12}{'after veto a':>14}{'b':>11}{'c (truth)':>11}   (per 15 years)")
    tot = {k: 0.0 for k in ("selected", "survive_a", "survive_b", "survive_c")}
    for name, r in res.items():
        print(f"{name:<20}{r['interactions']:>17.3g}{r['selected']:>12.3g}{r.get('survive_a', 0):>14.3g}"
              f"{r.get('survive_b', 0):>11.3g}{r.get('survive_c', 0):>11.3g}   ({r['n_selected']} selected)")
        for k in tot:
            tot[k] += r.get(k, 0.0)
    print(f"{'total':<20}{'':>17}{tot['selected']:>12.3g}{tot['survive_a']:>14.3g}{tot['survive_b']:>11.3g}{tot['survive_c']:>11.3g}")
    out = Path(args.out) if args.out else paths.results_dir() / "interactions" / f"{design.name}_{args.kind}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(design=design.name, kind=args.kind, source=source, split=args.split, dataset=args.dataset,
                   per_target=res, total=tot, seconds=round(time.time() - t0, 1)), open(out, "w"), indent=1)
    for name, r in res.items():
        if r.get("diag") and r["n_selected"]:
            print(f"  {name}: " + ", ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in r["diag"].items()))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
