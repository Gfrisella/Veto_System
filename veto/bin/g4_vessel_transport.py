#!/usr/bin/env python3
"""Geant4 transport (cpp/g4vessel) of the final states of the interactions in the helium (ISSUES A6).

Reads an interaction npz of bin/run_interactions.py (results/interactions/<design>_muon_interactions<tag>.npz, or a
neutrino apply file), keeps the interactions with target code 0 (helium; --codes for others), pre-selects those that
can give a vertex candidate (at least 2 charged particles with p > --p-min that point, as straight lines, into an
SST1 window enlarged by --margin), and carries every final-state particle with Geant4 through the helium, the Mylar
walls, the air and the spectrometer field to SST1 (z = 82.08 m) and the timing detector (94.01 m). Writes
<input>_g4.npz with the plane crossings (inter, row, plane, pdg, track, x, y, px, py, pz, t) and "transported" (the
interaction ids that were sent; the others keep their straight-line tracks). Use it with
    bin/run_interactions.py --design D6_straw --interactions <input>.npz --g4 <input>_g4.npz

Runs in the Geant4 container (cpp/g4vessel must be built first):
    bin/g4.sh bash cpp/build.sh
    bin/g4.sh python3 bin/g4_vessel_transport.py --interactions results/interactions/D6_straw_muon_interactions.npz --workers 32
"""

import _bootstrap  # noqa: F401

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from veto import geometry as geo
from veto import g4vessel as G
from veto.interactions import charge

_G4 = None
_PART = None


def _init(seed, part):
    global _G4, _PART
    _G4 = G.init(seed * 1000003 + os.getpid())
    _PART = part


def _run(inters):
    return G.transport_parts(_G4, _PART, inters)


def preselect(inter, part, codes, p_min, margin):
    """Interactions of the given target codes with >= 2 charged particles of p > p_min pointing (straight) into the
    SST1 window grown by margin."""
    want = np.nonzero(np.isin(inter["code"], codes))[0]
    q = charge(part["pdg"])
    p, v = part["p"][:, :3], part["v"]
    with np.errstate(divide="ignore", invalid="ignore"):
        xb = v[:, 0] + p[:, 0] / p[:, 2] * (geo.Z_SST1 - v[:, 2])
        yb = v[:, 1] + p[:, 1] / p[:, 2] * (geo.Z_SST1 - v[:, 2])
    ok = ((q != 0) & (p[:, 2] > 0) & (np.linalg.norm(p, axis=1) > p_min) & (v[:, 2] < geo.Z_SST1)
          & (np.abs(xb) < geo.SST1_HALF_X + margin) & (np.abs(yb) < geo.SST1_HALF_Y + margin))
    n_ok = np.bincount(part["int"][ok], minlength=len(inter["code"]))
    return want[n_ok[want] >= 2], len(want)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interactions", required=True, help="interaction npz (int_* / part_* arrays)")
    ap.add_argument("--codes", type=int, nargs="+", default=[0], help="target codes to transport (0 = helium)")
    ap.add_argument("--drop-v0-codes", type=int, nargs="*", default=[],
                    help="in interactions of these codes K_L, K_S, Lambda, Xi0 are not transported (their decays are the "
                         "forced copies of bin/shield_dis.py, code 4): use 3 for the shield output")
    ap.add_argument("--p-min", type=float, default=0.8, help="pre-selection: charged p [GeV] (selection: 1 GeV)")
    ap.add_argument("--margin", type=float, default=1.0, help="pre-selection: SST1 window grown by this [m]")
    ap.add_argument("--chunk", type=int, default=500, help="interactions per Geant4 call")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    f = np.load(args.interactions)
    inter = {k[4:]: f[k] for k in f.files if k.startswith("int_")}
    part = {k[5:]: f[k] for k in f.files if k.startswith("part_")}
    if np.any(np.diff(part["int"]) < 0):
        raise ValueError("part table must be sorted by interaction (fastveto.dis.run output is)")
    if args.drop_v0_codes:
        v0 = np.isin(np.abs(part["pdg"]), [130, 310, 3122, 3322]) & np.isin(inter["code"][part["int"]], args.drop_v0_codes)
        part = dict(part, pdg=np.where(v0, 0, part["pdg"]))        # pdg 0: skipped by g4vessel (unknown particle)
        print(f"{int(v0.sum()):,} V0s not transported (forced-decay copies cover them)")
    outside = (part["v"][:, 2] < G.Z_WORLD[0]) | (part["v"][:, 2] > G.PLANES[-1])   # g4vessel refuses them: skip (pdg 0)
    if outside.any():                                   # removed: g4vessel checks every production point
        part = {k: v[~outside] for k, v in part.items()}
        print(f"{int(outside.sum()):,} particles produced outside z {G.Z_WORLD[0]:.2f} .. {G.PLANES[-1]:.2f} m not transported")
    sel, n_code = preselect(inter, part, args.codes, args.p_min, args.margin)
    print(f"{n_code:,} interactions with codes {args.codes}, {len(sel):,} pre-selected for Geant4", flush=True)
    jobs = [sel[i:i + args.chunk] for i in range(0, len(sel), args.chunk)]
    res = []
    with mp.get_context("fork").Pool(args.workers, initializer=_init, initargs=(args.seed, part)) as pool:
        for i, r in enumerate(pool.imap_unordered(_run, jobs)):
            res.append(r)
            if (i + 1) % max(1, len(jobs) // 20) == 0:
                print(f"  {i + 1}/{len(jobs)} chunks, {time.time() - t0:.0f} s", flush=True)
    keys = ("inter", "row", "plane", "pdg", "track", "x", "y", "px", "py", "pz", "t")
    out = {k: np.concatenate([r[k] for r in res]) if res else np.zeros(0) for k in keys}
    order = np.lexsort((out["track"], out["inter"])) if len(out["inter"]) else np.zeros(0, int)
    out = {k: v[order] for k, v in out.items()}
    path = Path(args.out or str(Path(args.interactions).with_suffix("")) + "_g4.npz")
    np.savez_compressed(path, transported=sel, drop_v0_codes=np.array(args.drop_v0_codes, int), planes=np.array(G.PLANES), codes=np.array(args.codes),
                        p_min=args.p_min, margin=args.margin, **out)
    n_s1 = int(((out["plane"] == 0) & (charge(out["pdg"]) != 0)).sum()) if len(out["plane"]) else 0
    print(f"wrote {path}: {len(out['plane']):,} plane crossings ({n_s1:,} charged at SST1), {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
