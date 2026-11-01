#!/usr/bin/env python3
"""Smoke test of the shield-iron target (veto.shield_iron, cpp/g4shield): field lookup, and single particles from
the start of the last 50 cm of iron along +z, what reaches the plane behind the shield.

    bin/g4.sh python3 bin/shield_dis_smoke.py [--n 2000] [--no-field]
"""

import _bootstrap  # noqa: F401
import argparse
import time

import numpy as np

from veto import shield_iron as si


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--no-field", action="store_true")
    a = ap.parse_args()
    fm = si.load_field()
    pts = [(0.0, 0.0), (0.3, 0.2), (-0.3, 0.2), (-0.3, -0.2), (1.2, 0.3), (0.3, 1.0), (2.0, 0.3)]
    for z in (si.Z_IRON0, si.Z_SHIELD_END - 0.05, si.Z_SHIELD_END + 0.3):
        b = si.field_at(fm, *np.array([(x, y, z) for x, y in pts]).T)
        print(f"z = {z:.2f}: " + "  ".join(f"({x:+.1f},{y:+.1f}) {bb[0]:+.2f} {bb[1]:+.2f} {bb[2]:+.2f}" for (x, y), bb in zip(pts, b)))
    g4 = si.init_g4shield(seed=1, field=not a.no_field)
    print(g4.layout())
    for pdg, p in ((13, 50.0), (-13, 10.0), (211, 50.0), (130, 30.0), (2112, 30.0), (310, 30.0)):
        n = a.n
        off = np.arange(n + 1, dtype=np.int64)
        t0 = time.time()
        r = g4.transport(off, np.full(n, pdg, np.int32), np.zeros(n), np.zeros(n), np.full(n, p),
                         np.full(n, 0.3), np.full(n, 0.2), np.full(n, si.Z_IRON0 + 1e-4))
        dt = (time.time() - t0) / n * 1e3
        r = {k: np.asarray(v) for k, v in r.items()}
        prim = r["primary"] == 1
        P = np.sqrt(r["px"] ** 2 + r["py"] ** 2 + r["pz"] ** 2)
        types, cnt = np.unique(r["pdg"][~prim], return_counts=True)
        top = ", ".join(f"{t}:{c / n:.2f}" for t, c in sorted(zip(types, cnt), key=lambda x: -x[1])[:6])
        print(f"{pdg:6d} {p:5.0f} GeV: primary out {prim.sum() / n:.3f} (mean p {P[prim].mean() if prim.any() else 0:.1f}, "
              f"mean x {r['x'][prim].mean() if prim.any() else 0:+.3f} m), secondaries/event {(~prim).sum() / n:.2f} "
              f"[{top}], {dt:.2f} ms/event")


if __name__ == "__main__":
    main()
