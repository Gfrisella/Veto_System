#!/usr/bin/env python3
"""Rate sample for veto planes on the lateral walls of the decay vessel (design "walls", the SBT side walls).

Every muon of the design-optimisation dataset whose path A -> (K) -> B crosses the +x or -x vessel wall
(the "walls" bitmask, bits +x = 1, -x = 2). Any wall plane is then intersected with these paths
exactly (reco.wall_rate_map).

  srun -p express -c 2 --mem 16G bin/make_wall_rates.py      # -> results/reco/wall_rates.npz
"""

import _bootstrap  # noqa: F401

import argparse
import time

import numpy as np

from veto import io, paths

DATASET = "design-optimization-dataset-alpha-14-oct-2026"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunk", type=int, default=4_000_000)
    parser.add_argument("--cap", type=int, default=1_000_000, help="rows kept per muon type, weights scaled up")
    parser.add_argument("--seed", type=int, default=6)
    args = parser.parse_args()
    start = time.time()
    store, meta = io.open_store(paths.store_dir(DATASET))
    n = meta["n"]
    cols = ["x_30", "y_30", "z_30", "x_82", "y_82", "z_82", "kx", "ky", "kz", "weight", "type"]
    keep = {c: [] for c in cols}
    for lo in range(0, n, args.chunk):
        m = (np.asarray(store["walls"][lo:lo + args.chunk]) & 3) != 0
        for c in cols:
            keep[c].append(np.asarray(store[c][lo:lo + args.chunk])[m])
    keep = {c: np.concatenate(v) for c, v in keep.items()}
    w = keep["weight"].astype(np.float64)
    rng = np.random.default_rng(args.seed)
    rows, scale = [], []
    for t in np.unique(keep["type"]):
        idx = np.where(keep["type"] == t)[0]
        sub = np.sort(rng.choice(idx, args.cap, replace=False)) if len(idx) > args.cap else idx
        rows.append(sub)
        scale.append(np.full(len(sub), w[idx].sum() / w[sub].sum()))
    rows, scale = np.concatenate(rows), np.concatenate(scale)
    keep = {c: v[rows] for c, v in keep.items()}
    w = w[rows] * scale
    out = paths.results_dir() / "reco" / "wall_rates.npz"
    np.savez(out, a=np.stack([keep["x_30"], keep["y_30"], keep["z_30"]], 1).astype(np.float32),
             k=np.stack([keep["kx"], keep["ky"], keep["kz"]], 1).astype(np.float32),
             b=np.stack([keep["x_82"], keep["y_82"], keep["z_82"]], 1).astype(np.float32),
             w=w, type=keep["type"], total_per_spill=w.sum())
    by_type = {int(t): float(w[keep["type"] == t].sum()) for t in np.unique(keep["type"])}
    print(f"saved {out}: {w.sum():.4g} muons per spill cross a side wall ({len(w):,} rows kept); by type {by_type}; "
          f"{time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
