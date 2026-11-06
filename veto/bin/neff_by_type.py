#!/usr/bin/env python3
"""Per muon type: rows, rate and the effective number of independent INPUT muons (weights summed per evt_n first,
then (sum w)^2 / sum w^2), for one or more design datasets. Also the share of the rate carried by the top 10 inputs.

  bin/remote.sh run bin/neff_by_type.py <dataset> [<dataset> ...]
"""

import _bootstrap  # noqa: F401

import sys

import numpy as np

from veto import io, paths
from veto import muon_types as mt


NEFF_TARGET = 1000      # C5 (2026-11-05): effective input muons wanted per class


def main():
    for name in sys.argv[1:]:
        store, meta = io.open_store(paths.store_dir(name))
        typ, w, ev = (np.asarray(store[c]) for c in ("type", "weight", "evt_n"))
        print(f"\n{name}\n{'type':<15}{'rows':>13}{'muons/spill':>13}{'n_eff(input)':>14}{'top-10 share':>14}  (C5 target n_eff > {NEFF_TARGET})")
        for t, label in mt.TYPES.items():
            m = typ == t
            if not m.any():
                continue
            _, inv = np.unique(ev[m], return_inverse=True)
            wi = np.bincount(inv, weights=w[m].astype(np.float64))
            top = np.sort(wi)[-10:].sum() / wi.sum()
            neff = wi.sum() ** 2 / np.sum(wi ** 2)
            flag = "" if neff > NEFF_TARGET else f"   <- below target: ~x{NEFF_TARGET / max(neff, 1):.0f} more seeds"
            print(f"{label:<15}{m.sum():>13,}{wi.sum():>13.4g}{neff:>14.4g}{top:>14.1%}{flag}")


if __name__ == "__main__":
    main()
