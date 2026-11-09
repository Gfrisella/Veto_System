#!/usr/bin/env python3
"""Which stored muons touch the cavern rock (wall x = CAVERN_X_WALL, floor y = CAVERN_Y_FLOOR) between z = 30.6 and
82 m, and with what momentum and grazing angle: the input of a forced-DIS study in the cavern walls (ISSUES D7,
docs/full_sim_inventory.md). From a --storage shell transport (Horvitz-Thompson weights, one seed = one spill).

A muon "goes in" when its line from A (z = 30.6 m, direction p_30) meets the wall or the floor before z = 82 m.
  came back   it reached z = 82 m inside the cavern (types 0-5) with a credible bounce point K within NEAR of that
              surface (muon_types.bounce_points); a stopped muon's B is its stop point, so its K means nothing
  stayed      otherwise (stopped in the rock, or went on through it)
Grazing angle in = angle of p_30 to the surface; out (came back) = angle of p_82 to it. Path in the rock of a muon
that came back: L ~ (|p_30| - |p_82|) / DEDX (the transport has no energy loss in the cavern air).

    python3 bin/rock_hits.py <seed dir> [--max-seeds 300] --out rock_hits.npz
"""
import _bootstrap  # noqa: F401
import argparse
import glob

import numpy as np

from veto import geometry as geo
from veto import muon_types as mt

NEAR = 0.3                                     # m: bounce point this close to (or behind) the surface = a rock bounce
DEDX = 0.5                                     # GeV/m, concrete-like rock, muons of a few GeV (approximate)
PB = np.geomspace(0.1, 400, 41)                # GeV
AB = np.geomspace(1e-3, 1.0, 31)               # rad, grazing angle
LB = np.geomspace(0.01, 100, 31)               # m, path in the rock


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir"); ap.add_argument("--max-seeds", type=int, default=300); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(f"{a.dir}/seed_*.npz"))[:a.max_seeds]
    XW, YF = geo.CAVERN_X_WALL, geo.CAVERN_Y_FLOOR
    H = {}

    def add(key, val, w, bins):
        h, h2 = np.histogram(val, bins, weights=w)[0], np.histogram(val, bins, weights=w ** 2)[0]
        if key not in H:
            H[key] = np.zeros((2,) + h.shape)
        H[key][0] += h; H[key][1] += h2

    tot = {}
    for n, f in enumerate(files):
        d = np.load(f)
        r = np.nonzero((d["type"] <= 6) & (d["z_30"] > mt.REACH_30))[0]
        A = np.stack([d[f"{c}_30"][r] for c in "xyz"], 1).astype(np.float64)
        PA = np.stack([d[f"p{c}_30"][r] for c in "xyz"], 1).astype(np.float64)
        PBv = np.stack([d[f"p{c}_82"][r] for c in "xyz"], 1).astype(np.float64)
        K = np.stack([d["kx"][r], d["ky"][r], d["kz"][r]], 1).astype(np.float64)
        w, t = d["weight"][r].astype(np.float64), d["type"][r]
        pa, pb = np.linalg.norm(PA, axis=1), np.linalg.norm(PBv, axis=1)
        ua, ub = PA / pa[:, None], PBv / np.maximum(pb, 1e-12)[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            zw = np.where(ua[:, 0] < 0, A[:, 2] + (XW - A[:, 0]) / ua[:, 0] * ua[:, 2], np.inf)   # z where the A line meets
            zf = np.where(ua[:, 1] < 0, A[:, 2] + (YF - A[:, 1]) / ua[:, 1] * ua[:, 2], np.inf)   # the wall / the floor
        wall_first = zw <= zf
        zhit = np.minimum(zw, zf)
        goes_in = (zhit > A[:, 2]) & (zhit < 82.0)
        bounced = np.isfinite(K[:, 2])
        B = np.stack([d[f"{c}_82"][r] for c in "xyz"], 1).astype(np.float64)
        in_cavern = (t <= 5) & (B[:, 0] > XW) & (B[:, 1] > YF)     # reached z = 82 m in the cavern (a stopped muon's B is
        back = goes_in & bounced & in_cavern & np.where(wall_first, K[:, 0] < XW + NEAR, K[:, 1] < YF + NEAR)   # its stop point)
        a_in = np.where(wall_first, np.arcsin(np.clip(-ua[:, 0], 0, 1)), np.arcsin(np.clip(-ua[:, 1], 0, 1)))
        a_out = np.where(wall_first, np.arcsin(np.clip(ub[:, 0], 0, 1)), np.arcsin(np.clip(ub[:, 1], 0, 1)))
        L = np.clip(pa - pb, 0, None) / DEDX
        for surf, sm in (("wall", wall_first), ("floor", ~wall_first)):
            for fate, fm in (("back", back), ("stay", goes_in & ~back)):
                m = sm & fm
                k = f"{surf}_{fate}"
                tot[k] = tot.get(k, np.zeros(3)) + [w[m].sum(), (w[m] ** 2).sum(), m.sum()]
                add(f"{k}_p", pa[m], w[m], PB); add(f"{k}_ain", a_in[m], w[m], AB)
                for tt in range(7):
                    kk = f"{k}_type{tt}"
                    tot[kk] = tot.get(kk, np.zeros(3)) + [w[m & (t == tt)].sum(), (w[m & (t == tt)] ** 2).sum(), (m & (t == tt)).sum()]
                if fate == "back":
                    add(f"{k}_aout", a_out[m], w[m], AB); add(f"{k}_L", L[m], w[m], LB); add(f"{k}_pout", pb[m], w[m], PB)
        if n % 50 == 0:
            print(f"{n + 1}/{len(files)}", flush=True)
    S = len(files)
    print(f"\n{S} seeds (spills). Muons per spill that meet the rock between z = 30.6 and 82 m:")
    print(f"{'':12s} {'per spill':>11s} {'n_eff':>9s} {'rows':>9s}")
    for k in ("wall_back", "wall_stay", "floor_back", "floor_stay"):
        s, s2, nr = tot[k]
        print(f"{k:12s} {s / S:11.3g} {s * s / max(s2, 1e-300):9.0f} {int(nr):9d}")
    for surf in ("wall", "floor"):
        b, st = tot[f"{surf}_back"][0], tot[f"{surf}_stay"][0]
        print(f"{surf}: {100 * b / (b + st):.1f} % come back;  of those, by type: " +
              ", ".join(f"{mt.TYPES[tt]} {100 * tot[f'{surf}_back_type{tt}'][0] / b:.2g} %" for tt in range(7) if tot[f'{surf}_back_type{tt}'][0] > 0)
              + ";  stayed, by type: " + ", ".join(f"{mt.TYPES[tt]} {100 * tot[f'{surf}_stay_type{tt}'][0] / st:.2g} %" for tt in range(7) if tot[f'{surf}_stay_type{tt}'][0] > 0))
    for k in ("wall_back", "floor_back"):
        h = H[f"{k}_ain"][0]; c = np.cumsum(h) / h.sum()
        hl = H[f"{k}_L"][0]; cl = np.cumsum(hl) / hl.sum()
        hp = H[f"{k}_p"][0]; cp = np.cumsum(hp) / hp.sum()
        med = lambda cc, bins: np.sqrt(bins[np.searchsorted(cc, 0.5)] * bins[np.searchsorted(cc, 0.5) + 1])
        print(f"{k}: median grazing angle in {1e3 * med(c, AB):.0f} mrad, median p in {med(cp, PB):.2g} GeV, median path in rock ~{med(cl, LB):.2g} m")
    np.savez(a.out, seeds=S, PB=PB, AB=AB, LB=LB, NEAR=NEAR, DEDX=DEDX, **{f"h_{k}": v / S for k, v in H.items()},
             **{f"tot_{k}": v for k, v in tot.items()})
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
