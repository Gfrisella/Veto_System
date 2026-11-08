#!/usr/bin/env python3
"""Edges of the input pool (bin/build_pool.py): flux per component in fine momentum bins, pool against the enriched
file, around the places where the source changes (charm 20 GeV, min. bias 30 GeV, everything 150 GeV).

    python3 bin/pool_edges.py [--pool data/source/pool_v1]
"""

import _bootstrap  # noqa: F401

import argparse
import glob
import json

import h5py
import numpy as np

from veto import paths

import build_pool as bp

EDGES = np.concatenate([np.arange(10, 60, 2.5), np.arange(60, 130, 10), np.arange(130, 180, 5), np.arange(180, 420, 40)])
NAMES = ("tail", "min. bias", "dimuon", "charm", "other")


def flux(px, py, pz, w, comp):
    p = np.sqrt(px.astype(np.float64) ** 2 + py ** 2 + pz ** 2)
    i = np.digitize(p, EDGES) - 1
    ok = (i >= 0) & (i < len(EDGES) - 1)
    return np.bincount(comp[ok].astype(np.int64) * (len(EDGES) - 1) + i[ok], weights=w[ok].astype(np.float64),
                       minlength=5 * (len(EDGES) - 1)).reshape(5, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="/disk/lhcb_data/sqasim/pool_v1")
    a = ap.parse_args()
    with h5py.File(paths.enriched_input_h5(), "r") as f:
        comp = bp.components(f["weight"][:], f["source"][:])
        F = flux(f["px"][:], f["py"][:], f["pz"][:], f["weight"][:], comp)
    files = sorted(glob.glob(f"{a.pool}/train_*.h5"))
    P = 0
    for fn in files:                       # each training chunk is one spill: average them
        with h5py.File(fn, "r") as g:
            P = P + flux(g["px"][:], g["py"][:], g["pz"][:], g["weight"][:], g["component"][:])
    P = P / len(files)
    # the file's tail rows (component 0) are its min. bias above 150 GeV: compare min. bias + tail together
    F2, P2 = F.copy(), P.copy()
    F2[1] += F2[0]; P2[1] += P2[0]
    out = {}
    lo = EDGES[:-1]
    for c, name in ((1, "min. bias + tail"), (2, "dimuon"), (3, "charm")):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = P2[c] / F2[c]
        out[name] = dict(p_lo=lo.tolist(), ratio=np.nan_to_num(r).tolist())
        print(f"\n{name}: pool / file per momentum bin")
        print("  " + "  ".join(f"{l:.0f}:{x:.2f}" for l, x in zip(lo, r) if np.isfinite(x)))
    out["spectra"] = dict(edges=EDGES.tolist(), names=NAMES, file=F.tolist(), pool=P.tolist(),
                          parts=json.load(open(f"{a.pool}/pool.json"))["parts"])
    json.dump(out, open(f"{a.pool}/edges.json", "w"), indent=1)


if __name__ == "__main__":
    main()
