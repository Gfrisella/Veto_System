#!/usr/bin/env python3
"""Rate sample for a veto panel under the decay vessel (design "bottom" plane).

Every muon of the design-optimisation dataset whose path A -> (K) -> B passes below the vessel
(between the floor y = -3.36 m and the vessel bottom -Y(z), |x| < 4.56 m) somewhere between
z = 30.6 and 82 m. Any panel geometry is then intersected with these paths exactly (reco.py), so
its rate map needs no binning in z or y. Rows above --cap are subsampled with their weights scaled up.

  srun -p standard -c 4 --mem 32G bin/make_bottom_rates.py      # -> results/reco/bottom_rates.npz
"""

import _bootstrap  # noqa: F401

import argparse
import time

import numpy as np

from veto import geometry as geo
from veto import io, paths
from veto import muon_types as mt

DATASET = "design-optimization-dataset-alpha-14-oct-2026"
Z_TEST = np.arange(geo.Z_UBT, geo.Z_SST1 + 0.01, 0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunk", type=int, default=4_000_000)
    parser.add_argument("--cap", type=int, default=4_000_000)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()
    start = time.time()
    store, meta = io.open_store(paths.store_dir(DATASET))
    n = meta["n"]
    cols = ["x_30", "y_30", "z_30", "x_82", "y_82", "z_82", "kx", "ky", "kz", "weight", "type"]
    keep = {c: [] for c in cols}
    for lo in range(0, n, args.chunk):
        d = {c: np.asarray(store[c][lo:lo + args.chunk]) for c in cols}
        a = np.stack([d["x_30"], d["y_30"], d["z_30"]], 1).astype(np.float64)
        k = np.stack([d["kx"], d["ky"], d["kz"]], 1).astype(np.float64)
        b = np.stack([d["x_82"], d["y_82"], d["z_82"]], 1).astype(np.float64)
        under = np.zeros(len(a), bool)
        for z in Z_TEST:
            x, y, ok = mt.path_at_z(a, k, b, z)
            hy = geo.vessel_half_widths(min(z, geo.Z_VESSEL_MAX))[1]
            under |= ok & (y >= geo.CAVERN_Y_FLOOR) & (y < -hy) & (np.abs(x) < -geo.CAVERN_X_WALL)
        for c in cols:
            keep[c].append(d[c][under])
        print(f"  {min(lo + args.chunk, n):,} / {n:,}: {sum(len(v) for v in keep['weight']):,} kept, "
              f"{time.time() - start:.0f} s", flush=True)
    keep = {c: np.concatenate(v) for c, v in keep.items()}
    w = keep["weight"].astype(np.float64)
    total = w.sum()
    if len(w) > args.cap:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(len(w), args.cap, replace=False))
        keep = {c: v[idx] for c, v in keep.items()}
        w = keep["weight"].astype(np.float64) * total / keep["weight"].astype(np.float64).sum()
    out = paths.results_dir() / "reco" / "bottom_rates.npz"
    np.savez(out, a=np.stack([keep["x_30"], keep["y_30"], keep["z_30"]], 1).astype(np.float32),
             k=np.stack([keep["kx"], keep["ky"], keep["kz"]], 1).astype(np.float32),
             b=np.stack([keep["x_82"], keep["y_82"], keep["z_82"]], 1).astype(np.float32),
             w=w, type=keep["type"], total_per_spill=total)
    print(f"saved {out}: {len(w):,} muons, {total:.4g} muons per spill pass under the vessel, "
          f"{time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
