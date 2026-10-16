#!/usr/bin/env python3
"""Design dataset from our own muon production (bin/generate_muon_study.py): every type from the new seeds.

Each seed file keeps, per type, all muons or a random fraction re-weighted to the seed's total of that type
(kept_types, kept_fraction), so for type t the dataset takes the rows of the seeds that kept t and divides
their weights by the number of those seeds: muons per spill. Works on a production still running: only
complete files (written atomically) of the chosen seeds are read, and the seeds used are recorded.
Same columns as the alpha and beta datasets (plus nothing): start, z = 30.6 and z = 82 states, bounce point,
walls, type, weight, scale, source 3, run = seed.

  bin/remote.sh run bin/make_design_dataset_gamma.py --production data/sim/veto_muons_v1
"""

import _bootstrap  # noqa: F401

import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np

from make_design_dataset import finish
from veto import io, paths
from veto import muon_types as mt

NAME = "design-optimization-dataset-gamma"
N_TYPES = 8


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--production", default=str(Path(__file__).resolve().parents[1] / "data" / "sim" / "veto_muons_v1"))
    ap.add_argument("--max-seed", type=int, default=None, help="only seeds below this (e.g. a fixed snapshot)")
    ap.add_argument("--name", default=f"{NAME}-{datetime.date.today().isoformat()}")
    args = ap.parse_args()
    t0 = time.time()
    files = sorted(Path(args.production).glob("seed_*.npz"), key=lambda f: int(f.stem.split("_")[1]))
    if args.max_seed is not None:
        files = [f for f in files if int(f.stem.split("_")[1]) < args.max_seed]
    if not files:
        raise SystemExit(f"no seed files in {args.production}")

    parts, n_seeds_by_type, n_rows_by_type = [], np.zeros(N_TYPES, int), np.zeros(N_TYPES, int)
    w_all = np.zeros(N_TYPES)                       # per-type totals of ALL muons (kept or not), for a check
    for f in files:
        d = np.load(f)
        kept = d["kept_types"]
        n_seeds_by_type += kept
        w_all += d["w_by_type"]
        rows = {k: d[k] for k in d.files if d[k].ndim == 1 and len(d[k]) == len(d["type"])}
        rows["run"] = np.full(len(rows["type"]), int(d["seed"]), np.int32)
        parts.append(rows)
    data = io.concatenate(parts)
    typ = data["type"].astype(np.int8)
    scale = np.zeros(len(typ))
    for t in range(N_TYPES):
        m = typ == t
        n_rows_by_type[t] = m.sum()
        if n_seeds_by_type[t]:
            scale[m] = 1.0 / n_seeds_by_type[t]
    data["weight"] = (data["weight"] * scale).astype(np.float32)
    data["scale"] = (data["weight"] / np.where(data["weight_raw"] > 0, data["weight_raw"], 1)).astype(np.float32)
    del data["weight_raw"]
    data = finish({k: v for k, v in data.items() if k != "type"}, source=3)
    assert np.array_equal(data["type"], typ), "classification differs from the production's"
    per_type = {}
    print(f"{len(files)} seed files, {len(typ):,} rows, {time.time() - t0:.0f} s\n")
    print(f"{'type':<15}{'seeds':>8}{'rows':>13}{'muons/spill':>14}{'all muons/spill (check)':>26}")
    for t in range(N_TYPES):
        w = float(data["weight"][typ == t].sum())
        check = w_all[t] / len(files)
        per_type[mt.TYPES[t]] = dict(seeds=int(n_seeds_by_type[t]), rows=int(n_rows_by_type[t]), muons_per_spill=w,
                                     all_seeds_muons_per_spill=float(check))
        print(f"{mt.TYPES[t]:<15}{n_seeds_by_type[t]:>8}{n_rows_by_type[t]:>13,}{w:>14.4g}{check:>26.4g}")
    out = paths.store_dir(args.name)
    io.write_store(out, data, meta={
        "name": args.name, "created": datetime.date.today().isoformat(),
        "description": "Muons for veto design optimisation, every type from our production "
                       "(generate_muon_study.py: Guglielmo's simulation, our types, fill-until-budget storage)",
        "weight": "muons per spill: a seed's rows of a type are re-weighted to its total, divided by the number of "
                  "seeds that kept the type",
        "production": str(Path(args.production).resolve()), "seeds": [int(f.stem.split("_")[1]) for f in files],
        "per_type": per_type, "types": {str(k): v for k, v in mt.TYPES.items()}, "wall_bits": mt.WALL_BITS,
    })
    print(f"\nwrote {out}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
