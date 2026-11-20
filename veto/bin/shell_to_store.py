#!/usr/bin/env python3
"""A design-optimisation dataset store from --storage shell seeds (bin/generate_muon_study.py): the first --seeds seed
files of a shell directory concatenated, weight per spill = seed weight / n_seeds (each seed is one spill of the input
pool), run = seed index (pairs of the combinatorial term are formed within a run). Same columns as the other stores
(states at 30.6 m and 82 m, bounce point k, type, evt_n, walls), so muon_dis / make_reco_inputs / end_to_end use it as is.

    python3 bin/shell_to_store.py /disk/lhcb_data/sqasim/nov4_pool/shell/train --seeds 300 \
        --name design-optimization-dataset-nov4-shell-300seeds-2026-11-10
    python3 bin/shell_to_store.py /disk/lhcb_data/sqasim/nov4_pool/shell/train --seeds 3220 --stream --workers 8 \
        --name design-optimization-dataset-nov4-shell-3220seeds-2026-11-11      # ISSUES C5: column by column, low memory
"""
import _bootstrap  # noqa: F401
import argparse
import glob
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from veto import io, paths
from veto import muon_types as mt

COLS = ("x_start", "y_start", "z_start", "px_start", "py_start", "pz_start", "pdg", "x_30", "y_30", "z_30", "px_30", "py_30",
        "pz_30", "x_82", "y_82", "z_82", "px_82", "py_82", "pz_82", "weight", "weight_raw", "type", "evt_n", "kx", "ky", "kz", "walls")


def _fill_column(job):
    """Stream one column of every seed file into its .npy (memory map): pass 2 of --stream."""
    c, files, lengths, path, S = job
    off = np.concatenate([[0], np.cumsum(lengths)])
    out = None
    for k, f in enumerate(files):
        z = np.load(f)
        v = np.full(lengths[k], int(z["seed"]), np.int32) if c == "run" else z[c]
        if c in ("weight", "weight_raw"):
            v = v / S
        if out is None:
            out = np.lib.format.open_memmap(path, mode="w+", dtype=v.dtype, shape=(int(off[-1]),))
        out[off[k]:off[k + 1]] = v
    out.flush()
    return c, str(out.dtype)


def stream(a, files, S, t0):
    """Pass 1: lengths, per-type rates (type + weight only); pass 2: one column at a time into memory-mapped .npy files.
    Same columns, order and metadata as the in-memory path (checked on 20 seeds: identical stores)."""
    lengths, w_type = [], np.zeros(len(mt.TYPES) + 8)
    rows_type = np.zeros(len(w_type), np.int64)
    meta0 = None
    for k, f in enumerate(files):
        z = np.load(f)
        meta0 = meta0 or str(z["meta"])
        t, w = z["type"].astype(int), z["weight"] / S
        lengths.append(len(t))
        np.add.at(w_type, t, w); np.add.at(rows_type, t, 1)
        if k % 200 == 0:
            print(f"pass 1: {k + 1}/{S} seeds, {time.time() - t0:.0f} s", flush=True)
    lengths = np.asarray(lengths, np.int64)
    d = paths.store_dir(a.name)
    d.mkdir(parents=True, exist_ok=True)
    cols = list(COLS) + ["run"]
    jobs = [(c, files, lengths, str(d / f"{c}.npy"), S) for c in cols]
    with ProcessPoolExecutor(a.workers) as ex:
        for c, dt in ex.map(_fill_column, jobs):
            print(f"pass 2: {c} ({dt}) written, {time.time() - t0:.0f} s", flush=True)
    per_type = {mt.TYPES[t]: dict(seeds=S, rows=int(rows_type[t]), per_spill=float(w_type[t])) for t in mt.TYPES}
    meta = dict(source=a.dir, seeds=S, files=[f.split("/")[-1] for f in files], storage="shell", per_type=per_type,
                pot_per_spill=4e13, seed_meta=meta0[:2000], weight="per spill: seed weight / seeds",
                pdg_convention="charge sign: +1 = mu+, -1 = mu- (read with muon_types.charge_from_pdg_column)")
    n = int(lengths.sum())
    with open(d / "meta.json", "w") as f:
        json.dump({"n": n, "columns": cols, **meta}, f, indent=2)
    print(f"{n:,} rows from {S} seeds, {w_type.sum():.4g} muons / spill, {time.time() - t0:.0f} s -> {d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir"); ap.add_argument("--seeds", type=int, default=300); ap.add_argument("--name", required=True)
    ap.add_argument("--stream", action="store_true", help="column by column into memory-mapped files (thousands of seeds)")
    ap.add_argument("--workers", type=int, default=8, help="--stream: columns written in parallel")
    a = ap.parse_args()
    t0 = time.time()
    files = sorted(glob.glob(f"{a.dir}/seed_*.npz"), key=lambda f: int(re.findall(r"seed_(\d+)", f)[-1]))[:a.seeds]
    S = len(files)
    if a.stream:
        return stream(a, files, S, t0)
    out = {k: [] for k in COLS}; out["run"] = []
    meta0 = None
    for k, f in enumerate(files):
        z = np.load(f)
        meta0 = meta0 or str(z["meta"])
        for c in COLS:
            v = z[c]
            out[c].append(v / S if c in ("weight", "weight_raw") else v)
        out["run"].append(np.full(len(z["weight"]), int(z["seed"]), np.int32))
        if k % 50 == 0:
            print(f"{k + 1}/{S} seeds, {time.time() - t0:.0f} s", flush=True)
    cols = {k: np.concatenate(v) for k, v in out.items()}
    per_type = {mt.TYPES[t]: dict(seeds=S, rows=int((cols["type"] == t).sum()), per_spill=float(cols["weight"][cols["type"] == t].sum()))
                for t in mt.TYPES}
    meta = dict(source=a.dir, seeds=S, files=[f.split("/")[-1] for f in files], storage="shell", per_type=per_type,
                pot_per_spill=4e13, seed_meta=meta0[:2000], weight="per spill: seed weight / seeds",
                pdg_convention="charge sign: +1 = mu+, -1 = mu- (read with muon_types.charge_from_pdg_column)")
    io.write_store(paths.store_dir(a.name), cols, meta)
    print(f"{len(cols['weight']):,} rows from {S} seeds, {cols['weight'].sum():.4g} muons / spill, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
