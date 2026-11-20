#!/usr/bin/env python3
"""Fine (2 x 2 cm^2) rate maps near the muon flames, for the PEAK rate of a plane (ISSUES A3).

The design-explorer fin maps have 20 x 20 cm^2 bins and the UBT map is binned in y only (summed over
|x| < 0.5 m): both average the peak down for cells much smaller than a bin (2 / 4 cm tiles). This builds,
from every muon of the full dataset (Full_dataset_z_3060_8200.pkl, one run, weights = muons per spill):

  ubt    x-y map at z = 30.6 m, |x| < UBT_HALF_X, -4 < y < 4 m           (muons entering the UBT window)
  fin    per side (0: +x left, 1: -x right) and z slice 30.6 .. 40.6 m (every 1 m, = the first 11
         design-explorer slices), u = |x| - X(z) in 0 .. 2 m, -4 < y < 4 m (outside the vessel)

each as sum w and sum w^2 per bin (so reco.plane_rates can ignore bins with too few muons: the maximum over
~10^5 sparsely populated bins is biased up). Muons of type 7 (never reach z = 30.6 m) are left out, as in
the coarse maps. Paths A -> K -> B from muon_types.classify / path_at_z.

  python3 bin/make_peak_maps.py [--workers 32] [--stop N]     # -> results/rate_maps/peak_maps.npz

Since 11 Nov 2026 the default source is a design store (--store, default paths.MUON_DATASET_FULL: the Nov-4 production,
3220 spills; weights per spill, the type and the bounce point already stored), and the UBT map is at the straw plane
of the default layout (--ubt-z, geometry.Z_UBT_STRAW = 30.73 m). --full <pickle> keeps the old one-run source.
"""

import _bootstrap  # noqa: F401

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import make_design_dataset as mdd
from veto import geometry as geo
from veto import io, paths
from veto import muon_types as mt

BIN = 0.02
FIN_Z = np.arange(geo.Z_UBT, geo.Z_UBT + 10.01, 1.0)          # 31.13 .. 41.13 m (30.6 .. 40.6 before 2026-11-05)
FIN_U_BINS = np.arange(0, 2.0 + 1e-9, BIN)
Y_BINS = np.arange(-4.0, 4.0 + 1e-9, BIN)
UBT_X_BINS = np.arange(-geo.UBT_HALF_X, geo.UBT_HALF_X + 1e-9, BIN)

_COLUMNS = None
UBT_Z = geo.Z_UBT_STRAW


def _open_store(name):
    global _COLUMNS
    if _COLUMNS is None:
        _COLUMNS, _ = io.open_store(paths.store_dir(name))
    return _COLUMNS


def _open(path):
    global _COLUMNS
    if _COLUMNS is None:
        _COLUMNS, _ = io.open_pickle_columns(path)
    return _COLUMNS


def _hist2(a, b, ea, eb, w):
    """sum w and sum w^2 in the (ea, eb) grid (values outside are dropped)."""
    ia = np.floor((a - ea[0]) / BIN).astype(np.int64)
    ib = np.floor((b - eb[0]) / BIN).astype(np.int64)
    na, nb = len(ea) - 1, len(eb) - 1
    ok = (ia >= 0) & (ia < na) & (ib >= 0) & (ib < nb)
    k = ia[ok] * nb + ib[ok]
    return (np.bincount(k, weights=w[ok], minlength=na * nb).reshape(na, nb),
            np.bincount(k, weights=w[ok] ** 2, minlength=na * nb).reshape(na, nb))


def chunk(job):
    src, a, b, ubt_z = job
    if src.startswith("store:"):                               # design store: type and bounce point already stored
        cols = _open_store(src[6:])
        d = {k: np.asarray(cols[k][a:b]) for k in ("weight", "type", "kx", "ky", "kz", "pz_30",
                                                    "x_30", "y_30", "z_30", "x_82", "y_82", "z_82")}
        c = {k: d[k] for k in ("type", "kx", "ky", "kz")}
    else:
        d = mdd.new_rows(_open(src), slice(a, b))
        c = mt.classify(d)
    reach = c["type"] != 7
    w = d["weight"][reach].astype(np.float64)
    pa = np.stack([d[f"{k}_30"][reach] for k in "xyz"], 1).astype(np.float64)
    pb = np.stack([d[f"{k}_82"][reach] for k in "xyz"], 1).astype(np.float64)
    pk = np.stack([c[k][reach] for k in ("kx", "ky", "kz")], 1).astype(np.float64)
    pz = d["pz_30"][reach]

    xu, yu, at_u = mt.path_at_z(pa, pk, pb, ubt_z)            # straw UBT (30.73 m default), on the path from A (30.6 m)
    on = at_u & (np.abs(xu) < geo.UBT_HALF_X) & (pz > 0) & (pa[:, 2] > geo.Z_STATE_A - 0.05)
    ubt = _hist2(xu[on], yu[on], UBT_X_BINS, Y_BINS, w[on])

    nu, ny = len(FIN_U_BINS) - 1, len(Y_BINS) - 1
    fin_w, fin_w2 = np.zeros((2, len(FIN_Z), nu, ny)), np.zeros((2, len(FIN_Z), nu, ny))
    for iz, z in enumerate(FIN_Z):
        x, y, ok = mt.path_at_z(pa, pk, pb, z)
        u = np.abs(x) - mt.side_plane_x(z)
        for sd in (0, 1):
            m = ok & (u >= 0) & ((x < 0) == bool(sd))
            hw, hw2 = _hist2(u[m], y[m], FIN_U_BINS, Y_BINS, w[m])
            fin_w[sd, iz] += hw
            fin_w2[sd, iz] += hw2
    return ubt[0], ubt[1], fin_w, fin_w2, int(reach.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", default=None, help="old source: the one-run pickle (paths.full_dataset_pickle())")
    ap.add_argument("--store", default=paths.MUON_DATASET_FULL, help="design store (default: Nov-4, 3220 spills)")
    ap.add_argument("--ubt-z", type=float, default=UBT_Z, help="z of the straw UBT map [m]")
    ap.add_argument("--chunk", type=int, default=4_000_000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--stop", type=int, default=None, help="only the first N rows (testing)")
    ap.add_argument("--out", default=str(paths.results_dir() / "rate_maps" / "peak_maps.npz"))
    args = ap.parse_args()
    start = time.time()
    if args.full:
        src = args.full
        _, n = io.open_pickle_columns(args.full)
    else:
        src = "store:" + args.store
        _, smeta = io.open_store(paths.store_dir(args.store))
        n = int(smeta["n"])
    n = min(n, args.stop or n)
    jobs = [(src, a, min(a + args.chunk, n), args.ubt_z) for a in range(0, n, args.chunk)]
    acc, n_reach = None, 0
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(chunk, jobs)):
            acc = list(r[:4]) if acc is None else [s + v for s, v in zip(acc, r[:4])]
            n_reach += r[4]
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                print(f"  {jobs[i][2]:,} / {n:,} rows, {time.time() - start:.0f} s", flush=True)
    ubt_w, ubt_w2, fin_w, fin_w2 = acc
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(source=str(src), rows=n, rows_reaching_30m=n_reach, bin_m=BIN, ubt_z=args.ubt_z, built=time.strftime("%Y-%m-%d"))
    np.savez(out, ubt_w=ubt_w, ubt_w2=ubt_w2, ubt_x_bins=UBT_X_BINS, fin_w=fin_w, fin_w2=fin_w2, fin_z=FIN_Z,
             fin_u_bins=FIN_U_BINS, y_bins=Y_BINS, meta=json.dumps(meta))
    print(f"saved {out}: UBT {ubt_w.sum():.4g}, fins (all slices) {fin_w.sum():.4g} muons/spill, "
          f"{time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
