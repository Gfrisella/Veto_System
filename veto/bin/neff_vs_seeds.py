#!/usr/bin/env python3
"""How the independent input muons (n_eff, weights summed per evt_n) grow with the number of seeds of a production
dataset: every seed propagates the SAME input muons again, so n_eff saturates once each input muon has been seen.
Per type, n_eff and the distinct input muons using the first k seeds (k = all / 8, / 4, / 2, all); weights are
rescaled to per spill (x all / k), which does not change n_eff.

  bin/remote.sh run bin/neff_vs_seeds.py <dataset>
"""

import _bootstrap  # noqa: F401

import sys

import numpy as np

from veto import io, paths
from veto import muon_types as mt


def main():
    store, meta = io.open_store(paths.store_dir(sys.argv[1]))
    col = next(c for c in ("seed", "run", "run_id", "file_id") if c in meta["columns"])
    typ, w, ev, sd = (np.asarray(store[c]) for c in ("type", "weight", "evt_n", col))
    seeds = np.unique(sd)
    ks = [len(seeds) // 8, len(seeds) // 4, len(seeds) // 2, len(seeds)]
    print(f"{sys.argv[1]}: {len(seeds)} seeds (column {col}); n_eff / distinct input muons with the first k seeds")
    print(f"{'type':<15}" + "".join(f"{'k=' + str(k):>22}" for k in ks))
    for t, label in mt.TYPES.items():
        out = []
        for k in ks:
            m = (typ == t) & np.isin(sd, seeds[:k])
            if not m.any():
                out.append("-"); continue
            _, inv = np.unique(ev[m], return_inverse=True)
            wi = np.bincount(inv, weights=w[m].astype(np.float64))
            out.append(f"{wi.sum() ** 2 / np.sum(wi ** 2):.4g} / {len(wi):,}")
        print(f"{label:<15}" + "".join(f"{o:>22}" for o in out))


if __name__ == "__main__":
    main()
