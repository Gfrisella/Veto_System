#!/usr/bin/env python3
"""A design-optimisation dataset store from --storage shell seeds (bin/generate_muon_study.py): the first --seeds seed
files of a shell directory concatenated, weight per spill = seed weight / n_seeds (each seed is one spill of the input
pool), run = seed index (pairs of the combinatorial term are formed within a run). Same columns as the other stores
(states at 30.6 m and 82 m, bounce point k, type, evt_n, walls), so muon_dis / make_reco_inputs / end_to_end use it as is.

    python3 bin/shell_to_store.py /disk/lhcb_data/sqasim/nov4_pool/shell/train --seeds 300 \
        --name design-optimization-dataset-nov4-shell-300seeds-2026-11-10
"""
import _bootstrap  # noqa: F401
import argparse
import glob
import json
import re
import time

import numpy as np

from veto import io, paths
from veto import muon_types as mt

COLS = ("x_start", "y_start", "z_start", "px_start", "py_start", "pz_start", "pdg", "x_30", "y_30", "z_30", "px_30", "py_30",
        "pz_30", "x_82", "y_82", "z_82", "px_82", "py_82", "pz_82", "weight", "weight_raw", "type", "evt_n", "kx", "ky", "kz", "walls")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir"); ap.add_argument("--seeds", type=int, default=300); ap.add_argument("--name", required=True)
    a = ap.parse_args()
    t0 = time.time()
    files = sorted(glob.glob(f"{a.dir}/seed_*.npz"), key=lambda f: int(re.findall(r"seed_(\d+)", f)[-1]))[:a.seeds]
    S = len(files)
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
                pot_per_spill=4e13, seed_meta=meta0[:2000], weight="per spill: seed weight / seeds")
    io.write_store(paths.store_dir(a.name), cols, meta)
    print(f"{len(cols['weight']):,} rows from {S} seeds, {cols['weight'].sum():.4g} muons / spill, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
