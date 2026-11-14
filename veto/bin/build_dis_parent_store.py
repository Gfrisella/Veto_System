#!/usr/bin/env python3
"""Muon store for the vessel muon DIS: the muons that reach the vessel region, from ONE design store (11 Nov 2026:
the Nov-4 production, paths.MUON_DATASET_FULL, 3220 spills, weights per spill).

Kept: the SST1 types (0 sst1_bounced, 1 sst1_selected, 2 sst1_other) and the veto-only muons (3: cross a tagger plane
or the window, miss SST1); --types to change. The other types do not reach the vessel region or stop before it. The
columns are copied one by one into memory-mapped files (the full store is 84 GB).

Before 11 Nov 2026 this combined two productions (gamma: SST1 types of the 20 001-run sample; zeta: veto-only muons of
6 000 seeds, evt_n offset by 1e12); with one complete production that stitching is not needed (git history for it).

    python3 bin/build_dis_parent_store.py [--store <name>] [--types 0 1 2 3]
    -> <VETO_STORE>/dis-parents-<store>-types0123
"""
import _bootstrap  # noqa: F401

import argparse
import json
import time

import numpy as np

from veto import io, paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=paths.MUON_DATASET_FULL)
    ap.add_argument("--types", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    t0 = time.time()
    src = paths.store_dir(a.store)
    meta = json.load(open(src / "meta.json"))
    typ = np.load(src / "type.npy", mmap_mode="r")
    keep = np.nonzero(np.isin(np.asarray(typ), a.types))[0]
    name = a.name or f"dis-parents-{a.store}-types{''.join(map(str, a.types))}"
    out = paths.store_dir(name); out.mkdir(parents=True, exist_ok=True)
    for c in meta["columns"]:
        x = np.load(src / f"{c}.npy", mmap_mode="r")
        o = np.lib.format.open_memmap(out / f"{c}.npy", mode="w+", dtype=x.dtype, shape=(len(keep),))
        for s0 in range(0, len(keep), 50_000_000):
            o[s0:s0 + 50_000_000] = x[keep[s0:s0 + 50_000_000]]
        o.flush()
        print(f"  {c}: {time.time() - t0:.0f} s", flush=True)
    m = {k: v for k, v in meta.items() if k not in ("n", "columns")}
    m.update(source_store=a.store, types_kept=a.types, built=time.strftime("%Y-%m-%d"),
             note="vessel muon DIS parents: types " + ",".join(map(str, a.types)) + " of the source store, weights unchanged")
    json.dump({"n": int(len(keep)), "columns": meta["columns"], **m}, open(out / "meta.json", "w"), indent=1)
    w = np.load(out / "weight.npy", mmap_mode="r")
    print(f"{len(keep):,} rows ({np.asarray(w, np.float64).sum():.4g} muons / spill) -> {out}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
