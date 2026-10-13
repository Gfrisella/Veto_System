#!/usr/bin/env python3
"""Convert all TRY6_UBT_T1_seed_*.pkl output files (20 001 files, ~26 M muons) into one
memory-mappable column store, so later passes do not re-read 20 k pickles.

Columns: x0..pz2 (start, UBT and SST1 planes), weight, evt_n, pdg, file_index (seed order).
Reading is parallel over files.

  bin/remote.sh run bin/convert_sst1_outputs.py --workers 32
"""

import _bootstrap  # noqa: F401

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from veto import io, paths


def load(filename):
    data = io.load_output_file_all_planes(filename)
    return {key: (value.astype(np.float32) if value.dtype.kind == "f" else value) for key, value in data.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pattern", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--store", default=str(paths.store_dir("sst1_TRY6")))
    args = parser.parse_args()

    files = io.list_output_files(args.pattern, args.max_files)
    start = time.time()
    with Pool(args.workers) as pool:
        chunks = pool.map(load, files, chunksize=50)
    print(f"Read {len(files)} files in {time.time() - start:.1f} s")
    for i, chunk in enumerate(chunks):
        chunk["file_index"] = np.full(len(chunk["weight"]), i, dtype=np.int32)
    data = io.concatenate(chunks)

    _, counts = np.unique(data["evt_n"], return_counts=True)
    print(f"{len(data['weight']):,} muons, {len(counts):,} unique evt_n "
          f"(each appears on average {counts.mean():.1f} times, max {counts.max()})")
    io.write_store(Path(args.store), data, meta={"n_files": len(files),
                                                 "files": [Path(f).name for f in files]})
    print(f"Total {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()
