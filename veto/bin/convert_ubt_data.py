#!/usr/bin/env python3
"""Convert MuonsAndMatter's UBT_data_TRY6.pkl (150 M muons at the z = 30.6 m plane,
|x|, |y| < 3 m, 4.8 GB) into a memory-mappable column store, plus a random
subsample that is small enough to copy to a laptop.

Needs ~10 GB of RAM (the pickle is read in one go), so run it on a farm node:
  bin/remote.sh run bin/convert_ubt_data.py

Known limitation of this dataset: only muons crossing the 6 x 6 m plane at the
UBT are stored. Muons outside that window (e.g. ones that bounce off the cavern
walls and only reach the SBT/SST further downstream) are missing; the SST1
outputs contain part of them (category "bouncing").
"""

import _bootstrap  # noqa: F401

import argparse
import time
from pathlib import Path

import numpy as np

from veto import io, paths

COLUMNS = ("x", "y", "z", "px", "py", "pz", "pdg", "weight")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=str(paths.ubt_raw_pickle()))
    parser.add_argument("--store", default=str(paths.store_dir("ubt_TRY6")))
    parser.add_argument("--sample", type=int, default=2_000_000, help="rows in the random subsample")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    start = time.time()
    raw = io.load_pickle(args.input)
    print(f"Loaded {args.input} in {time.time() - start:.0f} s")
    columns = {key: np.asarray(raw[key], dtype=np.float32) for key in COLUMNS}
    del raw

    n = len(columns["x"])
    w = columns["weight"]
    print(f"{n:,} muons, weight sum {w.astype(np.float64).sum():.6e}, "
          f"z {columns['z'].min():.3f} - {columns['z'].max():.3f}")

    store = Path(args.store)
    io.write_store(store, columns, meta={"source": args.input, "plane_z": 30.6,
                                         "note": "only muons crossing |x|,|y| < 3 m at z = 30.6 m"})

    # subsample: every muon kept with equal probability, weights scaled so totals are preserved
    rng = np.random.default_rng(args.seed)
    index = np.sort(rng.choice(n, size=min(args.sample, n), replace=False))
    sample = {key: value[index] for key, value in columns.items()}
    sample["weight"] = sample["weight"] * (n / len(index))
    sample["index"] = index
    sample_path = store.parent / "ubt_TRY6_sample.npz"
    np.savez_compressed(sample_path, **sample)
    print(f"Wrote subsample of {len(index):,} rows to {sample_path} (weights scaled by {n / len(index):.2f})")


if __name__ == "__main__":
    main()
