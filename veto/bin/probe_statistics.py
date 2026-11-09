#!/usr/bin/env python3
"""How well does a muon dataset sample every place a veto could be? A dense probe of planes, cell by cell.

Not a veto design (the reconstruction does not know floor planes): a geometry to measure the dataset. Planes:
  UBT         at z = 30.6, |x| < 0.5 (the fixed 1 m), y = -3.2 .. 3.2 (the full height a long UBT could have)
  side fins   every --dz m from z = 30.6 to 80.6 on both sides, touching the vessel wall; right ones reach the
              cavern wall, left ones --left-reach m; y = -3.2 .. 3.2
  floor       planes under the vessel at --floor-z, tilted by --floor-tilt degrees towards the spectrometer
              (rotated about x): from the floor-line y = -3.2 up to the vessel bottom, |x| < X(z) + 1 m
Every muon path of the dataset (A -> bounce K -> B) is intersected with every plane; per --cell x --cell cell:
  rate     muons per spill (= Hz)
  rows     simulated rows crossing it
  neff     effective number of rows, (sum w)^2 / sum w^2
  neff_in  effective number of independent INPUT muons: weights first summed over the rows of one input muon
           (evt_n; the seeds re-simulate the same input sample, so rows of one evt_n are not independent)
Writes results/probe/probe_<dataset>.npz and an interactive page results/probe/probe.html (3D + maps per plane).

  GPU_HOST=farm-gpu5 bin/remote.sh run bin/probe_statistics.py          # ~10 min, CPU
"""

import _bootstrap  # noqa: F401

import argparse
import json
import time

import numpy as np

from veto import io, paths, probe

CHUNK = 8_000_000


def planes(args):
    """The probe planes of veto.probe for these command-line settings."""
    return probe.planes(args.dz, args.left_reach, args.floor_z, args.floor_tilt, args.ubt, args.only_ubt)


crossing, local = probe.crossing, probe.local


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="design-optimization-dataset-gamma-2026-10-17")
    ap.add_argument("--dz", type=float, default=2.5)
    ap.add_argument("--left-reach", type=float, default=5.0)
    ap.add_argument("--floor-z", type=float, nargs="*", default=[32.0, 40.0, 50.0, 60.0, 70.0])
    ap.add_argument("--floor-tilt", type=float, default=30.0, help="degrees, towards the spectrometer")
    ap.add_argument("--cell", type=float, default=0.25)
    ap.add_argument("--ubt", action=argparse.BooleanOptionalAction, default=True, help="UBT column |x| < 0.5, y = -3.2 .. 3.2")
    ap.add_argument("--only-ubt", action="store_true", help="only the UBT plane (writes probe_<dataset>_ubt.npz)")
    args = ap.parse_args()
    t0 = time.time()
    store, meta = io.open_store(paths.store_dir(args.dataset))
    n = meta["n"]
    pl = planes(args)
    grids = []
    for p in pl:
        nu = int(np.ceil((p["u"][1] - p["u"][0]) / args.cell))
        nv = int(np.ceil((p["y"][1] - p["y"][0]) / args.cell))
        grids.append(dict(nu=nu, nv=nv, rate=np.zeros(nu * nv), rows=np.zeros(nu * nv, np.int64),
                          w2=np.zeros(nu * nv), keys=[], kw=[]))
    for c0 in range(0, n, CHUNK):
        c1 = min(c0 + CHUNK, n)
        typ = np.asarray(store["type"][c0:c1])
        keep = np.nonzero(typ <= 6)[0]                       # type 7 stops in the shield: crosses nothing here
        get = lambda col: np.asarray(store[col][c0:c1])[keep].astype(np.float64)
        a = np.stack([get(f"{c}_30") for c in "xyz"], 1)
        b = np.stack([get(f"{c}_82") for c in "xyz"], 1)
        k = np.stack([get(f"k{c}") for c in "xyz"], 1)
        w = get("weight")
        ev = np.asarray(store["evt_n"][c0:c1])[keep]
        for p, g in zip(pl, grids):
            x, y, z, ok = crossing(a, k, b, p)
            u, v = local(p, x, y, z)
            ok &= (u >= p["u"][0]) & (u < p["u"][1]) & (v >= p["y"][0]) & (v < p["y"][1])
            iu = ((u[ok] - p["u"][0]) / args.cell).astype(np.int64).clip(0, g["nu"] - 1)
            iv = ((v[ok] - p["y"][0]) / args.cell).astype(np.int64).clip(0, g["nv"] - 1)
            cell = iu * g["nv"] + iv
            np.add.at(g["rate"], cell, w[ok]); np.add.at(g["rows"], cell, 1); np.add.at(g["w2"], cell, w[ok] ** 2)
            g["keys"].append(cell.astype(np.int64) * (1 << 31) + ev[ok].astype(np.int64)); g["kw"].append(w[ok])
        print(f"rows {c1:,} / {n:,}, {time.time() - t0:.0f} s", flush=True)
    out = {}
    summary = []
    for i, (p, g) in enumerate(zip(pl, grids)):
        keys, kw = np.concatenate(g["keys"]), np.concatenate(g["kw"])
        uk, inv = np.unique(keys, return_inverse=True)
        wg = np.bincount(inv, weights=kw)                    # weight per (cell, input muon)
        cell_of = (uk >> 31).astype(np.int64)
        s1, s2 = np.bincount(cell_of, weights=wg, minlength=len(g["rate"])), np.bincount(cell_of, weights=wg ** 2, minlength=len(g["rate"]))
        with np.errstate(invalid="ignore", divide="ignore"):
            neff = np.where(g["w2"] > 0, g["rate"] ** 2 / g["w2"], 0)
            neff_in = np.where(s2 > 0, s1 ** 2 / s2, 0)
        shape = (g["nu"], g["nv"])
        out.update({f"p{i}_rate": g["rate"].reshape(shape), f"p{i}_rows": g["rows"].reshape(shape),
                    f"p{i}_neff": neff.reshape(shape), f"p{i}_neff_in": neff_in.reshape(shape)})
        busy = g["rate"] > 0
        summary.append(dict(name=p["name"], rate_hz=float(g["rate"].sum()), rows=int(g["rows"].sum()),
                            cells=int(busy.sum()), median_neff_in=float(np.median(neff_in[busy])) if busy.any() else 0.0,
                            cells_neff_in_below_10=int((busy & (neff_in < 10)).sum())))
    out_dir = paths.results_dir() / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"probe_{args.dataset}{'_ubt' if args.only_ubt else ''}.npz", **out, planes=json.dumps(pl), cell=args.cell, summary=json.dumps(summary))
    print(f"\n{'plane':<20}{'rate [Hz]':>12}{'rows':>12}{'cells':>7}{'median n_eff(input)':>21}{'cells < 10':>12}")
    for s in summary:
        print(f"{s['name']:<20}{s['rate_hz']:>12.3g}{s['rows']:>12,}{s['cells']:>7}{s['median_neff_in']:>21.3g}{s['cells_neff_in_below_10']:>12}")
    print(f"wrote {out_dir}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
