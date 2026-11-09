#!/usr/bin/env python3
"""Recompute the muon types of a column store with the current geometry and field (FairShip layout, RK4 timing
detector): new columns type_fs, walls_fs, off_fs (the original type / walls / kx.. are kept), meta.json updated.

    python3 bin/retype_store.py <store dir> [--chunk 2000000] [--threads 32]
"""

import _bootstrap  # noqa: F401

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from veto import io
from veto import muon_types_torch as mtt

COLS = [f"{c}_{s}" for s in (30, 82) for c in ("x", "y", "z", "px", "py", "pz")] + ["pdg"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store")
    ap.add_argument("--chunk", type=int, default=2_000_000)
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    d = Path(a.store)
    st, meta = io.open_store(d)
    n = meta["n"]
    out = {k: np.lib.format.open_memmap(d / f"{k}.npy", mode="w+", dtype=dt, shape=(n,))
           for k, dt in (("type_fs", np.int8), ("walls_fs", np.uint8), ("off_fs", np.float32))}
    t0 = time.time()
    for a0 in range(0, n, a.chunk):
        a1 = min(n, a0 + a.chunk)
        c = mtt.classify({k: np.asarray(st[k][a0:a1]) for k in COLS}, device="cpu", chunk=a.chunk)
        out["type_fs"][a0:a1], out["walls_fs"][a0:a1], out["off_fs"][a0:a1] = c["type"], c["walls"], c["off"]
        print(f"  {a1:,} / {n:,}  {time.time() - t0:.0f} s", flush=True)
    for v in out.values():
        v.flush()
    meta["columns"] = list(dict.fromkeys(meta["columns"] + list(out)))
    meta["type_fs"] = "muon_types with the FairShip layout + spectrometer field (retype_store.py, 2026-11-06)"
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    old, new = np.asarray(st["type"]), np.asarray(out["type_fs"])
    w = np.asarray(st["weight"], np.float64)
    print("weighted fraction old type -> new type (rows: old 0..7, cols: new 0..7):")
    M = np.zeros((8, 8)); np.add.at(M, (old, new), w)
    np.set_printoptions(precision=4, suppress=True, linewidth=160)
    print(M / w.sum())
    print(f"done, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
