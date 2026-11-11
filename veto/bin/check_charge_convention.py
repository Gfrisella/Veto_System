#!/usr/bin/env python3
"""Regression check of the muon charge convention (docs/STATUS.md 2026-11-11, charge-sign fix): the stored datasets' "pdg"
column is the charge sign (+1 = mu+), the source pools hold PDG ids (13 = mu-). Matches the rows of a transported seed file
to its source pool chunk by pool_index and asserts that muon_types.charge_from_pdg_column gives the same charge for both;
optionally checks dataset stores by the side of the shield each charge leaves on (mu- at +x at z = 30.6 m for the
current shield polarity). Exit code 1 on any mismatch.

    python3 bin/check_charge_convention.py --seed /disk/lhcb_data/sqasim/nov4_pool/shell/train/seed_0.npz \
        --pool /disk/lhcb_data/sqasim/nov4_pool/train_000.h5 [--store <dataset store dir> ...]
"""
import _bootstrap  # noqa: F401
import argparse
import sys
from pathlib import Path

import numpy as np

from veto import muon_types as mt


def check_helper():
    got = mt.charge_from_pdg_column(np.array([13, -13, 1, -1, 13.0, -13.0, 1.0, -1.0]))
    want = np.array([-1, 1, 1, -1, -1, 1, 1, -1], float)
    assert np.array_equal(got, want), f"charge_from_pdg_column: {got} != {want}"
    print("helper: 13 -> mu-, -13 -> mu+, +1 -> mu+, -1 -> mu-  OK")


def check_seed(seed, pool):
    import h5py
    z = np.load(seed)
    stored, ev = np.asarray(z["pdg"]), np.asarray(z["evt_n"])
    with h5py.File(pool, "r") as h:
        pi, src = h["pool_index"][:], h["pdg"][:]
    o = np.argsort(pi)
    pos = np.clip(np.searchsorted(pi[o], ev), 0, len(pi) - 1)
    ok = pi[o][pos] == ev
    if ok.mean() < 0.99:
        raise SystemExit(f"only {ok.mean():.3f} of the seed rows found in {pool} by pool_index: wrong chunk?")
    q_st = mt.charge_from_pdg_column(stored[ok])
    q_src = mt.charge_from_pdg_column(src[o][pos][ok])
    bad = int((q_st != q_src).sum())
    print(f"{Path(seed).name} vs {Path(pool).name}: {ok.sum():,} rows matched, stored values {np.unique(stored)}, "
          f"source {np.unique(src[o][pos][ok])}; charge mismatches: {bad}")
    return bad == 0


def check_store(d):
    d = Path(d)
    q = mt.charge_from_pdg_column(np.load(d / "pdg.npy", mmap_mode="r")[:3_000_000])
    x = np.asarray(np.load(d / "x_30.npy", mmap_mode="r")[:3_000_000])
    xm, xp = x[q < 0].mean(), x[q > 0].mean()
    good = xm > 0 > xp
    print(f"{d.name}: mean x at 30.6 m, mu- {xm:+.2f} m, mu+ {xp:+.2f} m -> {'OK' if good else 'FLIPPED?'}")
    return good


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed"); ap.add_argument("--pool"); ap.add_argument("--store", nargs="*", default=[])
    a = ap.parse_args()
    check_helper()
    ok = True
    if a.seed and a.pool:
        ok &= check_seed(a.seed, a.pool)
    for s in a.store:
        ok &= check_store(s)
    print("ALL OK" if ok else "CHARGE CONVENTION MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
