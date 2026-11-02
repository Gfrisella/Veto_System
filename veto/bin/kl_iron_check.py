#!/usr/bin/env python3
"""K_L (and n, pi+) through the MS7 iron in cpp/g4shield (same set-up as bin/shield_dis.py): fraction that reaches the
plane as the unchanged primary vs exp(-d / lambda_I).

    bin/g4.sh python3 bin/kl_iron_check.py [--n 5000]
"""
import _bootstrap  # noqa: F401
import argparse
import numpy as np
from veto import shield_iron as si

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=5000); a = ap.parse_args()
g4 = si.init_g4shield(seed=7, field=True)
n = a.n
off = np.arange(n + 1, dtype=np.int64)
for d in (0.50, 0.25, 0.10):
    for pdg, p in ((130, 5.0), (130, 30.0), (130, 100.0), (2112, 30.0), (211, 30.0)):
        r = g4.transport(off, np.full(n, pdg, np.int32), np.zeros(n), np.zeros(n), np.full(n, p),
                         np.full(n, 0.3), np.full(n, 0.2), np.full(n, si.Z_SHIELD_END - d))
        r = {k: np.asarray(v) for k, v in r.items()}
        prim = (r["primary"] == 1) & (r["pdg"] == pdg)
        P = np.sqrt(r["px"] ** 2 + r["py"] ** 2 + r["pz"] ** 2)
        anyk = (r["pdg"] == pdg) & (P > 1)
        f = prim.sum() / n
        lam = -d / np.log(f) if 0 < f < 1 else float("nan")
        print(f"d={d:.2f} m  pdg {pdg:5d} {p:5.0f} GeV: primary survives {f:.3f} -> lambda_eff {100*lam:5.1f} cm; "
              f"same species p>1 GeV at plane (incl. secondaries) {anyk.sum()/n:.3f}", flush=True)
