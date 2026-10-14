#!/usr/bin/env python3
"""Build the design-optimisation dataset from the 20 001 old output files and the new full file.

Sources
  old:  TRY6_UBT_T1_seed_*.pkl (column store sst1_TRY6), 20 001 runs. Only muons ending in the
        SST1 acceptance, with start, z = 30.6 and z = 82 states. Gives the three SST1 types.
  new:  Full_dataset_z_3060_8200.pkl, every muon of ONE run (a rerun of seed 42) with its states
        at z = 30.6 and z = 82 (or the stop point). Gives everything that misses SST1.
        Its SST1 rows duplicate old seed 42 and are not used.

Every type (lib/veto/muon_types.py) is capped at --cap rows by uniform random sampling and
re-weighted, so `weight` is always muons per spill:
  old rows: weight / n_old_files * (n_type / n_kept)
  new rows: weight * (n_type / n_kept)
stops_before (never reaches z = 30.6) is dropped. Muons are paths A -> B, or A -> K -> B when
bounced (K = kx, ky, kz); `walls` says which SBT walls / the UBT window a full veto would see.

  bin/remote.sh run bin/make_design_dataset.py            # ~5 min on farm-gpu3, ~4 GB
"""

import _bootstrap  # noqa: F401

import argparse
import datetime
import time
from pathlib import Path

import numpy as np

from veto import io, paths
from veto import muon_types as mt

NAME = "design-optimization-dataset-alpha-14-oct-2026"
STATE = ("x", "y", "z", "px", "py", "pz")
SST1_TYPES = (0, 1, 2)
NEW_TYPES = (3, 4, 5, 6)


def old_rows(store, index):
    """Rows `index` (slice or sorted indices) of the old store in the dataset's column names."""
    d = {}
    for k in STATE:
        d[f"{k}_start"] = np.asarray(store[f"{k}0"][index], np.float32)
        d[f"{k}_30"] = np.asarray(store[f"{k}1"][index], np.float32)
        d[f"{k}_82"] = np.asarray(store[f"{k}2"][index], np.float32)
    d["weight"] = np.asarray(store["weight"][index], np.float64)
    d["pdg"] = np.asarray(store["pdg"][index]).astype(np.int8)
    d["evt_n"] = np.asarray(store["evt_n"][index]).astype(np.int64)
    d["run"] = np.asarray(store["file_index"][index]).astype(np.int32)
    return d


def new_rows(columns, index):
    """Rows `index` (slice or sorted indices) of the new file in the dataset's column names."""
    n = len(np.arange(*index.indices(len(columns["weight"])))) if isinstance(index, slice) else len(index)
    d = {}
    for k in STATE:
        d[f"{k}_30"] = np.asarray(columns[f"{k}0"][index], np.float32)
        d[f"{k}_82"] = np.asarray(columns[f"{k}1"][index], np.float32)
        d[f"{k}_start"] = np.full(n, np.nan, np.float32)
    d["weight"] = np.asarray(columns["weight"][index], np.float64)
    d["pdg"] = np.asarray(columns["pdg"][index]).astype(np.int8)
    d["evt_n"] = np.asarray(columns["evt_n"][index]).astype(np.int64)
    d["run"] = np.full(n, -1, np.int32)
    return d


def finish(d, source):
    """Add derived columns (type, walls, bounce point) and source."""
    c = mt.classify(d)
    d.update(type=c["type"], walls=c["walls"], kx=c["kx"].astype(np.float32),
             ky=c["ky"].astype(np.float32), kz=c["kz"].astype(np.float32),
             source=np.full(len(c["type"]), source, np.int8))
    return d


def pick(indices_by_type, cap, rng):
    """Uniform random subset of at most `cap` per type; returns {type: (sorted indices, scale)}."""
    out = {}
    for t, idx in indices_by_type.items():
        if len(idx) > cap:
            kept = np.sort(rng.choice(idx, cap, replace=False))
            out[t] = (kept, len(idx) / cap)
        else:
            out[t] = (idx, 1.0)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old-store", default=str(paths.store_dir("sst1_TRY6")))
    parser.add_argument("--new-file", default=str(paths.full_dataset_pickle()))
    parser.add_argument("--cap", type=int, default=15_000_000, help="max rows per type")
    parser.add_argument("--chunk", type=int, default=25_000_000)
    parser.add_argument("--stop", type=int, default=None, help="only the first N rows of each source (testing)")
    parser.add_argument("--out", default=str(paths.store_dir(NAME)))
    parser.add_argument("--seed", type=int, default=14102026)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    start_time = time.time()
    info = {}

    # ---- old files: SST1 types -------------------------------------------------------------
    store, meta = io.open_store(Path(args.old_store))
    n_old = min(meta["n"], args.stop or meta["n"])
    n_files = meta["n_files"]
    types_old = np.empty(n_old, np.int8)
    for a in range(0, n_old, args.chunk):
        types_old[a:a + args.chunk] = mt.classify(old_rows(store, slice(a, min(a + args.chunk, n_old))))["type"]
    other = np.setdiff1d(np.unique(types_old), SST1_TYPES)
    if len(other):
        print(f"warning: old rows of non-SST1 types {other.tolist()} dropped: "
              f"{np.isin(types_old, other).sum():,} rows")
    chosen_old = pick({t: np.where(types_old == t)[0] for t in SST1_TYPES}, args.cap, rng)
    print(f"old: {n_old:,} rows classified, {time.time() - start_time:.0f} s", flush=True)

    parts = []
    for t, (idx, scale) in chosen_old.items():
        rows = finish(old_rows(store, idx), source=0)
        assert np.all(rows["type"] == t)
        raw = rows["weight"].copy()
        rows["weight"] = rows["weight"] * scale / n_files
        rows["scale"] = np.full(len(raw), scale / n_files, np.float32)
        info[mt.TYPES[t]] = dict(source="old", n_source=int((types_old == t).sum()), n_kept=len(idx),
                                 scale=scale, runs=n_files, muons_per_spill=float(rows["weight"].sum()))
        parts.append(rows)
    del types_old

    # ---- new file: everything that misses SST1 ---------------------------------------------
    columns, n_new = io.open_pickle_columns(args.new_file)
    n_new = min(n_new, args.stop or n_new)
    types_new = np.empty(n_new, np.int8)
    for a in range(0, n_new, args.chunk):
        b = min(a + args.chunk, n_new)
        types_new[a:b] = mt.classify(new_rows(columns, slice(a, b)))["type"]
        print(f"  new: classified {b:,} / {n_new:,}, {time.time() - start_time:.0f} s", flush=True)
    for t in range(8):
        info.setdefault(mt.TYPES[t], {})["n_new_file"] = int((types_new == t).sum())
    chosen_new = pick({t: np.where(types_new == t)[0] for t in NEW_TYPES}, args.cap, rng)
    del types_new

    for t, (idx, scale) in chosen_new.items():
        rows = finish(new_rows(columns, idx), source=1)
        assert np.all(rows["type"] == t), (t, np.unique(rows["type"]))
        rows["weight"] = rows["weight"] * scale
        rows["scale"] = np.full(len(idx), scale, np.float32)
        info[mt.TYPES[t]].update(source="new", n_source=info[mt.TYPES[t]]["n_new_file"], n_kept=len(idx),
                                 scale=scale, runs=1, muons_per_spill=float(rows["weight"].sum()))
        parts.append(rows)
        print(f"  new: gathered {mt.TYPES[t]} ({len(idx):,} rows), {time.time() - start_time:.0f} s", flush=True)

    data = io.concatenate(parts)
    data["weight"] = data["weight"].astype(np.float32)
    data["evt_n"] = data["evt_n"].astype(np.int64)

    # ---- write -----------------------------------------------------------------------------
    print(f"\n{'type':<16}{'source':>7}{'in source':>14}{'kept':>12}{'scale':>12}{'muons/spill':>14}")
    for name, v in info.items():
        if "n_kept" in v:
            print(f"{name:<16}{v['source']:>7}{v['n_source']:>14,}{v['n_kept']:>12,}{v['scale']:>12.4g}"
                  f"{v['muons_per_spill']:>14.4g}")
    dropped = {k: v["n_new_file"] for k, v in info.items() if "n_kept" not in v}
    print(f"not used from the new file: {dropped}")
    io.write_store(Path(args.out), data, meta={
        "name": NAME,
        "created": datetime.date.today().isoformat(),
        "description": "Muons for veto design optimisation: SST1-acceptance types from the 20 001 old runs, "
                       "everything else from one full run; each type capped and re-weighted.",
        "weight": "muons per spill (old: / n_old_files; capped types: * n_type / n_kept). The repeated-evt_n "
                  "correction of the original analysis is NOT applied: plain average over runs.",
        "sources": {"old": {"store": str(args.old_store), "n_files": n_files, "rows": n_old},
                    "new": {"file": str(args.new_file), "rows": n_new,
                            "note": "rerun of seed 42; its SST1-acceptance rows duplicate old run 42, not used"}},
        "cap": args.cap, "seed": args.seed,
        "types": {str(k): v for k, v in mt.TYPES.items()},
        "per_type": info,
        "wall_bits": mt.WALL_BITS,
        "columns_doc": {
            "x_start..pz_start": "start state at z = -0.0045 m (old files only, NaN for new rows)",
            "x_30..pz_30": "state at z = 30.6 m (UBT plane)",
            "x_82..pz_82": "state at z = 82 m (SST1), or where the muon stopped (z_82 < 82)",
            "kx, ky, kz": "estimated bounce point (NaN if none): path is A -> K -> B",
            "walls": "bitmask of SBT walls / UBT window the path crosses (wall_bits)",
            "weight": "muons per spill", "scale": "factor applied to the original weight",
            "type": "see types", "run": "old file index (sorted by seed), -1 for the new file",
            "source": "0 old files, 1 new file", "evt_n": "input muon index", "pdg": "muon charge sign",
        },
    })
    print(f"done in {time.time() - start_time:.0f} s")


if __name__ == "__main__":
    main()
