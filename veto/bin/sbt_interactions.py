#!/usr/bin/env python3
"""Interactions in the SBT material (ISSUES D19): muons of the Nov-4 store or the 2026 neutrino flux, forced in the SBT
layers (inner Al, LS, ribs, outer Al; lib/veto/sbt_material.py), final states from the muon library with Al / LS tables
(bin/dis_library.py --only-g4 aluminium ls -> mu_library_sbt.npz) or GENIE (nu_library_v2: Al27, C12, H1).

    python3 bin/sbt_interactions.py --kind muon --draws 2e6 --workers 16 --out $R/interactions/sbt_mu.npz
    python3 bin/sbt_interactions.py --kind nu --draws 3e5 --workers 16 --out $R/interactions/sbt_nu.npz
    then: bin/evstore_build.py apply <npz> --out <store>  (+ bin/evstore_redecay.py --natural for the K_L), bin/sbt_eval.py

--no-enrich draws every final state (default: only those that can give a vertex in the helium, weight x their fraction;
sbt_material._flag_final_states). Output: int_* / part_* as bin/run_interactions.py (codes 9101-9104, int_active,
int_w_factor), and a json with the interactions per spill / 15 y per layer.
"""
import _bootstrap  # noqa: F401
import argparse
import json
import time
from pathlib import Path

import numpy as np

from veto import io, paths
from veto import sbt_material as sm
from veto.config import SPILLS_15Y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=("muon", "nu"), required=True)
    ap.add_argument("--dataset", default=paths.MUON_DATASET, help="muon store (default: the Nov-4 store of the environment)")
    ap.add_argument("--types", type=int, nargs="+", default=None, help="muon types to use (default all)")
    ap.add_argument("--mu-lib", default=str(paths.interactions_data() / "dis/mu_library_sbt.npz"))
    ap.add_argument("--nu-lib", default=str(paths.interactions_data() / "genie/nu_library_v2.npz"))
    ap.add_argument("--flavours", default=",".join(map(str, paths.NU_FLAVOURS)))
    ap.add_argument("--draws", type=float, default=1e6, help="draws per layer (per flavour for neutrinos)")
    ap.add_argument("--no-enrich", action="store_true")
    ap.add_argument("--codes", type=int, nargs="*", default=None, help="only these layers (9101 al_inner, 9102 ls, 9103 rib, 9104 al_outer)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=2_000_000)
    ap.add_argument("--max-rows", type=int, default=None, help="test: only the first rows of the dataset")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    sm.register()
    rng = np.random.default_rng(a.seed)
    draws = {c: (int(a.draws) if a.codes is None or c in a.codes else 0) for c in sm.CODES}
    enrich = not a.no_enrich
    summary = dict(kind=a.kind, draws_per_layer=int(a.draws), enrich=enrich, geometry=dict(
        al_inner=sm.T_AL_INNER, ls=sm.T_LS, al_outer=sm.T_AL_OUTER, rib=sm.T_RIB, source="FairShip 26.10 veto_config_helium.yaml"))
    I, P, n_int = [], [], 0
    if a.kind == "muon":
        from fastveto import dis
        store = paths.store_dir(a.dataset)
        cols, meta = io.open_store(store)
        if a.max_rows:
            cols = {k: v[:a.max_rows] for k, v in cols.items()}
        lib = dis.MuonLibrary(a.mu_lib, split="crossover")
        if enrich:
            sm.enrich_muon_library(lib)
        print(f"library {a.mu_lib} ({time.time() - t0:.0f} s); dataset {store} ({len(cols['weight']):,} rows)", flush=True)
        inter, part, totals = sm.run(cols, lib, draws, rng, chunk=a.chunk, workers=a.workers, enrich=enrich, types=a.types)
        summary.update(dataset=a.dataset, library=a.mu_lib, types=a.types, interactions_per_spill=totals,
                       interactions_15y={k: v * SPILLS_15Y for k, v in totals.items()})
        I.append(inter); P.append(part)
    else:
        summary.update(library=a.nu_lib, flavours={})
        for fl in [int(x) for x in a.flavours.split(",")]:
            cols, meta = io.open_store(Path(f"{paths.NU_FLUX_PREFIX}_nu{fl}"))
            cols = {k: np.asarray(v) if isinstance(v, io.ScaledColumn) else v for k, v in cols.items()}
            if a.max_rows:
                cols = {k: v[:a.max_rows] for k, v in cols.items()}
            lib = sm.EnrichedNuLibrary(a.nu_lib, flavour=fl, enrich=enrich)
            inter, part, totals = sm.run(cols, lib, draws, rng, chunk=a.chunk, workers=a.workers, enrich=enrich)
            inter["flavour"] = np.full(len(inter["E"]), fl, np.int32)
            part["int"] = part["int"] + n_int
            n_int += len(inter["E"])
            I.append(inter); P.append(part)
            summary["flavours"][str(fl)] = dict(interactions_per_spill=totals, interactions_15y={k: v * SPILLS_15Y for k, v in totals.items()})
            print(f"nu {fl}: " + ", ".join(f"{k} {v * SPILLS_15Y:.3g}" for k, v in totals.items()) + " per 15 y", flush=True)
    keys = set.intersection(*[set(x) for x in I])
    inter = {k: np.concatenate([x[k] for x in I]) for k in keys}
    part = {k: np.concatenate([x[k] for x in P]) for k in P[0]}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{f"int_{k}": v for k, v in inter.items()}, **{f"part_{k}": v for k, v in part.items()})
    summary.update(n_interactions=int(len(inter["E"])), n_particles=int(len(part["pdg"])), seconds=round(time.time() - t0, 1),
                   enriched_weight_15y={sm.CODES[c]: float(inter["weight"][inter["code"] == c].sum() * SPILLS_15Y) for c in sm.CODES})
    json.dump(summary, open(out.with_suffix(".json"), "w"), indent=1)
    print(f"{len(inter['E']):,} interactions, {len(part['pdg']):,} particles -> {out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
