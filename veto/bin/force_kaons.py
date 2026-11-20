#!/usr/bin/env python3
"""Forced K+- decay copies for an existing MS7 production file (bin/shield_dis.py output; 16 Nov 2026, Guglielmo).

A charged kaon from the MS7 interaction that decays in flight in the vessel can fake a vertex (one 6.3 GeV K- carried 55 %
of the MS7 total of the V0-statistics production: one MC event). Like the V0 copies: for every K+- at the exit plane of an
ORIGINAL event (code 3), `copies` copies (code 4, v0 = +-321) in which it decays in the vessel at an independent point
(truncated exponential, c tau 3.712 m) and channel (veto.evstore table), weight x P(decay in the vessel) / copies; with
--acc-cut only copies with >= 2 charged products of p > 1 GeV into the SST1 aperture are kept. The store builder
(bin/evstore_build.py) then keeps every forced species stable except each copy's own parent (no double counting); the
parent K+- is drawn from the exit plane to its decay point (a charged track: it leaves its own tagger hits).
No Geant4 re-run: works on the stored exit-plane particles.

    python3 bin/force_kaons.py <in.npz> <out.npz> --copies 20 --acc-cut --seed 1
"""
import _bootstrap  # noqa: F401
import argparse
import time

import numpy as np

from run_interactions import load_npz
from shield_dis import KPM, force_v0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--copies", type=int, default=20)
    ap.add_argument("--acc-cut", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    t0 = time.time()
    inter, part = load_npz(a.inp)
    n0 = len(inter["weight"])
    inter2, part2 = force_v0(inter, part, np.random.default_rng(a.seed), a.copies, acc_cut=a.acc_cut, species=KPM,
                             only_codes=(3,))
    k = len(inter2["weight"]) - n0
    w15 = inter2["weight"][n0:].sum() * 1.5e7
    np.savez(a.out, **{f"int_{x}": v for x, v in inter2.items()}, **{f"part_{x}": v for x, v in part2.items()})
    print(f"{a.inp}: {n0:,} events -> +{k:,} K+- copies (weight {w15:.3g} / 15 y of kept copies), {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
