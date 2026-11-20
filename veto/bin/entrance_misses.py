#!/usr/bin/env python3
"""ISSUES B1 follow-up (16 Nov 2026, Guglielmo: keep r_ms = 6 mrad GeV): muons that cross the decay-vessel ENTRANCE
window (|x| < VESSEL_X_IN, |y| < VESSEL_Y_IN at z = 31.13 m, where the UBT is) but cannot be tagged by the UBT because
their scattered hit falls outside the search window of their measured track. Regions outside the entrance are left to
the optimiser (fins, redundancy).

Per muon (same chain as reco.run_events with multiple scattering: kicks in the real materials with tails, measured
scattered state at SST1, own hit = straight crossing + scattering offset): "entrance" = its scattered crossing of the
tile plane at 31.13 m lies inside the entrance window; "lost" = no UBT plane (straws, tiles) has its crossing inside the
window r = r0 + (r_slope + r_ms / p) L (+ half cell). Averaged over --repeats scattering draws.

Reported: single SST1 muons (non-bounced) and the legs of the candidate pairs (IP < 10 cm and IP < 2.5 m), weighted.

    bin/entrance_misses.py [--design designs/R2_sbt_ubt.json] [--repeats 5] --out results_rerun/reco/entrance_misses.json
"""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

import numpy as np

from run_reco import load_inputs, take
from veto import design as dsg
from veto import geometry as geo
from veto import muon_types as mt
from veto import reco, scattering
from veto import vertex_selection as vsel
from veto.config import SPILLS_15Y

P_BINS = (0.0, 1.0, 2.0, 5.0, 10.0, 1e9)


def lost_at_entrance(sub, design, cfg, rng):
    """Per muon: (crosses the entrance window, lost = no UBT plane has the hit in the window), p measured."""
    kicks = scattering.kicks(sub, design, rng, media=cfg.msc_media, tails=cfg.msc_tails)
    meas = reco.measure(scattering.scattered_tracks(sub, kicks), cfg, rng)
    n = len(meas["p"])
    entrance = np.zeros(n, bool); crossed_any = np.zeros(n, bool); found = np.zeros(n, bool)
    for plane in design.planes:
        if plane.kind not in ("ubt", "tiles"):
            continue
        u, y, _, _ = reco.true_crossing(sub, plane)
        dx, dy = scattering.offset_at(kicks, plane.z)
        ok = np.isfinite(u) & np.isfinite(y) & (sub["apz"] > 0) & (sub["az"] < plane.z + mt.A_TOLERANCE)
        u, y = u + dx, y + dy
        ok &= plane.inside(u, y)
        x_glob = plane.to_global(u)
        if abs(plane.z - geo.Z_UBT_TILES) < 1e-6:
            entrance |= ok & (np.abs(x_glob) < geo.VESSEL_X_IN) & (np.abs(y) < geo.VESSEL_Y_IN)
        dz = meas["z"] - plane.z
        pu = plane.to_local(meas["x"] - meas["tx"] * dz)
        py = meas["y"] - meas["ty"] * dz
        dist = dz * np.sqrt(1 + meas["tx"] ** 2 + meas["ty"] ** 2)
        r = cfg.r0 + (cfg.r_slope + cfg.r_ms / np.maximum(meas["p"], 0.1)) * dist
        uc, yc = plane.cell_centre(u, y)
        inside = (np.abs(uc - pu) <= r + plane.cell_u / 2) & (np.abs(yc - py) <= r + plane.cell_y / 2)
        crossed_any |= ok
        found |= ok & inside
    return entrance, entrance & ~found, meas["p"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", default="designs/R2_sbt_ubt.json"); ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=21); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    design = dsg.load(Path(a.design))
    tr, inp = load_inputs()
    cfg = reco.RecoConfig(seed=a.seed, multiple_scattering=True)
    res = dict(design=design.name, r_ms=cfg.r_ms, repeats=a.repeats)

    # muons involved: non-bounced singles + every pair leg
    single = np.asarray(inp["single"]); sw = np.asarray(inp["single_w"], float)
    nb = tr["type"][single].astype(int) != 0
    cand, cand_w = inp["cand"], np.asarray(inp["cand_w"], float)
    rows = np.unique(np.concatenate([single[nb], cand.ravel()]))
    sub = take(tr, rows)
    ent = np.zeros(len(rows)); lost = np.zeros(len(rows)); p = None
    for k in range(a.repeats):
        e, l, p_k = lost_at_entrance(sub, design, cfg, np.random.default_rng(a.seed + k))
        ent += e; lost += l; p = p_k if p is None else p
    ent /= a.repeats; lost /= a.repeats                         # probabilities per muon
    idx = {int(r): i for i, r in enumerate(rows)}
    pos = lambda rr: np.array([idx[int(r)] for r in rr])

    # singles
    i_s = pos(single[nb]); w = sw[nb]
    s = dict(rate_per_spill=float(w.sum()), entrance_per_spill=float((w * ent[i_s]).sum()),
             lost_per_spill=float((w * lost[i_s]).sum()))
    s["lost_frac_of_entrance"] = s["lost_per_spill"] / max(s["entrance_per_spill"], 1e-30)
    s["by_p"] = {}
    for lo, hi in zip(P_BINS[:-1], P_BINS[1:]):
        m = (p[i_s] >= lo) & (p[i_s] < hi)
        s["by_p"][f"{lo:g}-{hi:g} GeV"] = dict(entrance_per_spill=float((w * ent[i_s])[m].sum()),
                                              lost_per_spill=float((w * lost[i_s])[m].sum()))
    res["single_non_bounced"] = s

    # pairs, both pointing cuts (measured IP as run_reco, same rng seed 777)
    st = lambda i: (np.stack([tr["bx"][i], tr["by"][i], tr["bz"][i]], 1), np.stack([tr["bpx"][i], tr["bpy"][i], tr["bpz"][i]], 1))
    ip = vsel.measured_ip(*st(cand[:, 0]), *st(cand[:, 1]), cfg, np.random.default_rng(777))
    l0, l1 = lost[pos(cand[:, 0])], lost[pos(cand[:, 1])]
    e0, e1 = ent[pos(cand[:, 0])], ent[pos(cand[:, 1])]
    res["pairs"] = {}
    for name, cut in vsel.IP_CUTS.items():
        m = ip < cut
        wp = cand_w[m]
        res["pairs"][name] = dict(
            n_pairs=int(m.sum()),
            frac_one_leg_through_entrance=float((wp * (1 - (1 - e0[m]) * (1 - e1[m]))).sum() / wp.sum()),
            frac_one_leg_lost=float((wp * (1 - (1 - l0[m]) * (1 - l1[m]))).sum() / wp.sum()),
            frac_both_legs_lost=float((wp * l0[m] * l1[m]).sum() / wp.sum()),
            n_pairs_with_a_lost_leg=int(((l0[m] > 0) | (l1[m] > 0)).sum()))
    print(json.dumps(res, indent=1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
