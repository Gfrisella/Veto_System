#!/usr/bin/env python3
"""Post-processing straw reconstruction (FairShip-like): straw hits in SST1-4 and the GenFit DAF fit, giving per track
n_hits, chi2 / ndf, fitted momentum and charge (lib/veto/straws.py, bin/genfit_straws.py).

    python3 bin/straw_reco.py --toy 2000                       # toy muons 1-40 GeV from z = 81.58 m: validation
    bin/g4ship.sh python3 bin/straw_reco.py --toy 2000 --g4    # the same with Geant4 trajectories (scattering)
    python3 bin/straw_reco.py --tracks tracks.npz --out fit.npz   # tracks.npz: pos (n,3) [m], mom (n,3) [GeV], q (n,)

Tracks are propagated with RK4 in the field map (no scattering); the fit runs in the SHiP CVMFS ROOT / GenFit
(bin/shipenv.sh), --workers processes.
"""

import _bootstrap  # noqa: F401

import argparse
import time
from pathlib import Path

import numpy as np

from veto import geometry as geo
from veto import paths
from veto import straws


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--toy", type=int, default=0)
    ap.add_argument("--tracks", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--g4", action="store_true", help="trajectories from Geant4 (run under bin/g4ship.sh): scattering, decays")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    if a.toy:
        n = a.toy
        p = rng.uniform(1, 40, n)
        tx, ty = rng.normal(0, 0.03, n), rng.normal(0, 0.03, n)
        mom = np.stack([tx, ty, np.ones(n)], 1) / np.sqrt(1 + tx ** 2 + ty ** 2)[:, None] * p[:, None]
        pos = np.stack([rng.uniform(-1.6, 1.6, n), rng.uniform(-2.5, 2.5, n), np.full(n, geo.Z_SST1 - 0.5)], 1)
        q = rng.choice([-1.0, 1.0], n)
    else:
        f = np.load(a.tracks)
        pos, mom, q = f["pos"], f["mom"], f["q"]
    t0 = time.time()
    work = paths.results_dir() / "straws" / ("toy" if a.toy else Path(a.tracks).stem)
    states = lost = None
    if a.g4:
        states, _, lost = straws.layer_states_g4(pos, mom, (-13 * q).astype(int), seed=a.seed)
        work = work.with_name(work.name + "_g4")
    r = straws.reconstruct(pos, mom, q, rng, work, workers=a.workers, states=states, lost=lost)
    out = Path(a.out) if a.out else work / "fit.npz"
    np.savez(out, **r)
    ok = r["fitted"] == 1
    g = r["good"]
    print(f"{len(q)} tracks in {time.time() - t0:.0f} s: fitted {ok.mean():.3f}, converged {np.nanmean(r['converged'][ok]):.3f}, "
          f"good (chi2/ndf < 50) {g.mean():.3f}")
    print(f"  hits: median {np.median(r['n_hits']):.0f}; chi2/ndf median {np.nanmedian(r['chi2ndf'][g]):.2f}; "
          f"ndf median {np.nanmedian(r['ndf'][g]):.1f}")
    dp = (r["p"] - r["p_true"]) / r["p_true"]
    for lo, hi in ((1, 5), (5, 15), (15, 40)):
        s = g & (r["p_true"] >= lo) & (r["p_true"] < hi)
        if s.any():
            print(f"  p {lo}-{hi} GeV: dp/p median {np.median(dp[s]):+.4f}, rms(core) {np.std(dp[s][np.abs(dp[s]) < 0.2]):.4f}, "
                  f"charge right {np.mean(np.sign(r['charge'][s]) == r['q_true'][s]):.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
