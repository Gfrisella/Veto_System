#!/usr/bin/env python3
"""Flux and its statistical precision in (p, pT) at production, for input samples: Guglielmo's 506 M file, the 601 M
enriched file (datasets a, b) and an input pool (dataset c; its training chunks, each one spill, averaged).

Per bin: sum w (muons per spill) and sum w^2 (relative error sqrt(sum w^2) / sum w; n_eff = (sum w)^2 / sum w^2).
Also per source component (tail, min. bias, dimuon, charm, other).

    python3 bin/source_hist2d.py [--pool /disk/lhcb_data/sqasim/pool_v2]      # writes <pool>/hist2d.npz
"""

import _bootstrap  # noqa: F401

import argparse
import glob

import h5py
import numpy as np

from veto import paths

import build_pool as bp

P_EDGES = np.geomspace(10, 400, 41)
PT_EDGES = np.linspace(0, 4, 41)
CHUNK = 50_000_000


def hist(f, comp_of=None, scale=1.0, comp_col=None):
    """sum w, sum w^2 per (component, p, pT) for one h5 file, read in chunks."""
    s1 = np.zeros((5, len(P_EDGES) - 1, len(PT_EDGES) - 1))
    s2 = np.zeros_like(s1)
    with h5py.File(f, "r") as g:
        n = len(g["px"])
        for a in range(0, n, CHUNK):
            b = min(a + CHUNK, n)
            px, py, pz = (g[k][a:b].astype(np.float64) for k in ("px", "py", "pz"))
            w = g["weight"][a:b].astype(np.float64) * scale
            if comp_col:
                c = g[comp_col][a:b].astype(np.int64)
            else:
                src = g["source"][a:b] if "source" in g else np.zeros(b - a, np.int8)
                c = bp.components(g["weight"][a:b], src).astype(np.int64)
            pt = np.hypot(px, py)
            p = np.sqrt(pt ** 2 + pz ** 2)
            i = np.digitize(p, P_EDGES) - 1
            j = np.digitize(pt, PT_EDGES) - 1
            ok = (i >= 0) & (i < len(P_EDGES) - 1) & (j >= 0) & (j < len(PT_EDGES) - 1)
            k = (c[ok] * (len(P_EDGES) - 1) + i[ok]) * (len(PT_EDGES) - 1) + j[ok]
            s1 += np.bincount(k, weights=w[ok], minlength=s1.size).reshape(s1.shape)
            s2 += np.bincount(k, weights=w[ok] ** 2, minlength=s2.size).reshape(s2.shape)
            print(f"  {f}: {b:,} / {n:,}", flush=True)
    return s1, s2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="/disk/lhcb_data/sqasim/pool_v2")
    a = ap.parse_args()
    out = dict(p_edges=P_EDGES, pt_edges=PT_EDGES)
    out["old506_s1"], out["old506_s2"] = hist(str(paths.MUONS_AND_MATTER / "data/muons/full_sample_after_target.h5"))
    out["enr601_s1"], out["enr601_s2"] = hist(str(paths.enriched_input_h5()))
    files = sorted(glob.glob(f"{a.pool}/train_*.h5"))
    s1 = s2 = 0
    for f in files:                       # each chunk is one spill: the union is one spill with weights / n
        x, y = hist(f, scale=1.0 / len(files), comp_col="component")
        s1, s2 = s1 + x, s2 + y
    out["pool_s1"], out["pool_s2"] = s1, s2
    np.savez(f"{a.pool}/hist2d.npz", **out)
    print("wrote", f"{a.pool}/hist2d.npz")


if __name__ == "__main__":
    main()
