#!/usr/bin/env python3
"""Beta design dataset: alpha plus the veto-only muons of Guglielmo's shah_study seeds where they help.

The shah_study seeds (one spill each; read only, completed runs only) keep up to 1500 muons per category
per seed. Of our types, only the veto-only muons entering through the central |x|, |y| <= 0.5 m of the
vessel entrance are well sampled there: they are all in the category SST1_miss (entered |x|, |y| <= 0.5 at
z = 30.6, misses the SST1 acceptance), which is not swamped by other muons. Alpha has 214 such rows, from
one run; the seeds give ~240 per seed. Everything else is as in alpha:
  SST1 types               the 20 001 old runs (the seeds reuse their seed numbers: not independent)
  veto-only, elsewhere     alpha's full run (walls, outer window: swamped in SBT_miss, ~4 muons / 300 seeds)
  rock-bounced, straight-past, stops-between   alpha's full run (15 M rows capped, enough)
New rows: weight = weight_by_category["SST1_miss"] / n_seeds (muons per spill), source 2, run = seed.

  bin/remote.sh run bin/make_design_dataset_beta.py            # farm-gpu3, ~10 min
"""

import _bootstrap  # noqa: F401

import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np

from make_design_dataset import STATE, finish
from scan_shah_study import SHAH, completed_seeds, load
from veto import io, paths
from veto import muon_types as mt

ALPHA = "design-optimization-dataset-alpha-14-oct-2026"
NAME = "design-optimization-dataset-beta-16-oct-2026"
VETO_ONLY = 3
CENTRAL = 0.5          # m, his SST1_miss window at z = 30.6


def central(x, y):
    return (np.abs(x) <= CENTRAL) & (np.abs(y) <= CENTRAL)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-age", type=float, default=10.0, help="minutes since a seed file was last written")
    ap.add_argument("--max-seeds", type=int, default=None)
    ap.add_argument("--out", default=str(paths.store_dir(NAME)))
    args = ap.parse_args()
    t0 = time.time()

    seeds = completed_seeds(args.min_age)[:args.max_seeds]
    n_seeds = len(seeds)
    print(f"{n_seeds} completed shah_study seeds", flush=True)
    new, n_cat = [], 0
    for k, s in enumerate(seeds):
        d = load(s)
        typ = mt.classify(d)["type"]
        take = (typ == VETO_ONLY) & central(d["x_30"], d["y_30"]) & (d["wcat"]["SST1_miss"] > 0)
        n_cat += int((d["wcat"]["SST1_miss"] > 0).sum())
        n = int(take.sum())
        rows = {f"{c}_{p}": d[f"{c}_{p}"][take].astype(np.float32) for c in STATE for p in ("30", "82")}
        rows.update({f"{c}_start": np.full(n, np.nan, np.float32) for c in STATE})
        rows["weight"] = d["wcat"]["SST1_miss"][take] / n_seeds
        rows["pdg"] = np.sign(d["pdg"][take]).astype(np.int8)
        rows["evt_n"] = d["evt_n"][take].astype(np.int64)
        rows["run"] = np.full(n, s, np.int32)
        new.append(rows)
        if k % 500 == 0:
            print(f"  seed {k} / {n_seeds}, {time.time() - t0:.0f} s", flush=True)
    new = finish(io.concatenate(new), source=2)
    assert np.all(new["type"] == VETO_ONLY)
    new["scale"] = np.full(len(new["weight"]), 1.0 / n_seeds, np.float32)
    print(f"new: {len(new['weight']):,} central veto-only rows ({new['weight'].sum():.4g} / spill), "
          f"{time.time() - t0:.0f} s", flush=True)

    store, meta = io.open_store(paths.store_dir(ALPHA))
    t, x, y = np.asarray(store["type"]), np.asarray(store["x_30"]), np.asarray(store["y_30"])
    drop = (t == VETO_ONLY) & central(x, y)
    keep = np.where(~drop)[0]
    old_w = float(np.asarray(store["weight"])[drop].sum())
    print(f"alpha: {meta['n']:,} rows, {drop.sum()} central veto-only rows replaced ({old_w:.4g} / spill)", flush=True)
    data = {}
    for c in meta["columns"]:
        col = np.asarray(store[c])[keep]
        data[c] = np.concatenate([col, new[c].astype(col.dtype)]) if c in new else None
        assert data[c] is not None, f"column {c} missing in the new rows"
    per_type = {}
    for k_t, name in mt.TYPES.items():
        m = data["type"] == k_t
        per_type[name] = dict(rows=int(m.sum()), muons_per_spill=float(data["weight"][m].sum()))
    io.write_store(Path(args.out), data, meta={
        **{k: v for k, v in meta.items() if k not in ("n", "columns")},
        "name": NAME, "created": datetime.date.today().isoformat(),
        "description": "alpha, with the veto-only muons through the central |x|, |y| <= 0.5 m of the entrance "
                       "replaced by those of the shah_study seeds (category SST1_miss, one spill per seed)",
        "beta": {"shah_study": str(SHAH), "n_seeds": n_seeds, "seeds": seeds, "min_age_min": args.min_age,
                 "rows_new": int(len(new["weight"])), "picked_in_SST1_miss": n_cat,
                 "alpha_rows_replaced": int(drop.sum()), "alpha_weight_replaced": old_w,
                 "new_weight": float(new["weight"].sum())},
        "per_type_beta": per_type,
    })
    print(json.dumps(per_type, indent=1))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
