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

from veto import geometry as geo
from veto import io, paths
from veto import muon_types as mt

CHUNK = 8_000_000


def planes(args):
    """List of planes: dict(name, kind, z0, tilt, u-range / x-range, y-range, cell grid)."""
    out = []
    if args.ubt:                                          # UBT column in front of the vessel, as tall as the fins
        out.append(dict(name="UBT z=30.6", kind="ubt", z0=float(geo.Z_UBT), tilt=0.0,
                        u=(-float(geo.UBT_HALF_X), float(geo.UBT_HALF_X)), y=(mt.FIN_Y_MIN, 3.2)))
    if args.only_ubt:
        return out
    for z in np.arange(geo.Z_VESSEL_MIN, geo.Z_VESSEL_MAX + 1e-6, args.dz):
        X = mt.side_plane_x(z)
        for side, sign, reach in (("right", -1, -mt.CAVERN_X_WALL - X), ("left", 1, args.left_reach)):
            out.append(dict(name=f"{side} fin z={z:.1f}", kind="fin", side=side, sign=sign, z0=float(z), tilt=0.0,
                            u=(0.0, float(reach)), y=(mt.FIN_Y_MIN, 3.2)))
    for z in args.floor_z:
        X = mt.side_plane_x(z) + 1.0
        _, hy = geo.vessel_half_widths(np.array([z]))
        out.append(dict(name=f"floor z={z:.1f}", kind="floor", z0=float(z), tilt=float(args.floor_tilt),
                        u=(-float(X), float(X)), y=(mt.FIN_Y_MIN, -float(hy[0]))))
    return out


def crossing(a, k, b, plane):
    """Where the paths cross the plane: global (x, y, z) and a mask. Fins: z = z0. Floor: the plane through
    the line (z0, y = -3.2) rotated by tilt about x: f = (z - z0) - (y + 3.2) tan(tilt) = 0."""
    if plane["kind"] in ("fin", "ubt"):
        x, y, ok = mt.path_at_z(a, k, b, plane["z0"])
        return x, y, np.full(len(x), plane["z0"]), ok
    t = np.tan(np.radians(plane["tilt"]))
    f = lambda p: (p[:, 2] - plane["z0"]) - (p[:, 1] - mt.FIN_Y_MIN) * t
    bounced = np.isfinite(k[:, 2])
    hit = np.zeros(len(a), bool)
    pt = np.full((len(a), 3), np.nan)
    for p0, p1, use in ((a, np.where(bounced[:, None], k, b), np.ones(len(a), bool)), (k, b, bounced)):
        f0, f1 = f(p0), f(p1)
        with np.errstate(invalid="ignore", divide="ignore"):
            s = f0 / (f0 - f1)
        m = use & ~hit & np.isfinite(s) & (s >= 0) & (s <= 1)
        pt[m] = p0[m] + s[m, None] * (p1[m] - p0[m])
        hit |= m
    return pt[:, 0], pt[:, 1], pt[:, 2], hit


def local(plane, x, y, z):
    """Local (u, v) on the plane: fins u = distance from the vessel wall, v = y; floor u = x, v = y."""
    if plane["kind"] == "fin":
        return plane["sign"] * x - mt.side_plane_x(plane["z0"]), y
    return x, y


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
