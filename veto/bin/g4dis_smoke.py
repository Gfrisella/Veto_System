"""Smoke test of the g4dis module: speed, cross section vs fastveto.kokoulin, nu distribution, multiplicities."""

import _bootstrap  # noqa: F401
import sys
import time

import numpy as np

from veto import paths

sys.path.insert(0, str(paths.cpp_build()))
import g4dis  # noqa: E402
from fastveto import kokoulin  # noqa: E402

mat = sys.argv[1] if len(sys.argv) > 1 else "G4_He"
g4dis.initialize(seed=5, material=mat)
for E in (3.0, 10.0, 100.0):
    t0 = time.time()
    o = g4dis.generate(13, E, 3000)
    dt = time.time() - t0
    ev, pr = o["events"], o["products"]
    nu = np.asarray(ev["nu"])
    pdg = np.asarray(pr["pdg"])
    p = np.sqrt(np.asarray(pr["px"]) ** 2 + np.asarray(pr["py"]) ** 2 + np.asarray(pr["pz"]) ** 2)
    ch = np.isin(np.abs(pdg), [211, 321, 2212, 11, 13])
    pev = np.asarray(pr["ev"])
    nch1 = np.bincount(pev[ch & (p > 1)], minlength=len(nu))
    W2 = 0.9383 ** 2 + 2 * 0.9383 * nu        # Q2 ~ 0 neglected here
    print(f"{mat} E={E:5.1f}: {dt / 3000 * 1e3:.2f} ms/event, done {np.mean(ev['done']):.3f}, xs {g4dis.cross_section(13, E):.4g}/m; "
          f"nu q10/50/90 {np.round(np.nanquantile(nu, [.1, .5, .9]), 2)}, frac W<2 (Q2=0) {np.mean(W2 < 4):.3f}, "
          f"frac nu>1 {np.mean(nu > 1):.3f}; >=1 charged hadron p>1 GeV: {np.mean(nch1 >= 1):.3f}; top pdgs "
          f"{dict(zip(*np.unique(pdg, return_counts=True)))}" if E == 3.0 else
          f"{mat} E={E:5.1f}: {dt / 3000 * 1e3:.2f} ms/event, done {np.mean(ev['done']):.3f}, xs {g4dis.cross_section(13, E):.4g}/m; "
          f"nu q10/50/90 {np.round(np.nanquantile(nu, [.1, .5, .9]), 2)}, frac W<2 (Q2=0) {np.mean(W2 < 4):.3f}, frac nu>1 {np.mean(nu > 1):.3f}; "
          f">=1 charged p>1 GeV: {np.mean(nch1 >= 1):.3f}")
