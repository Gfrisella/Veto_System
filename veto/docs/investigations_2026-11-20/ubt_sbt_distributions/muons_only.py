#!/usr/bin/env python3
"""Class "mu only": the Nov-4 muon store (paths.MUON_DATASET, 3220 spills, weights = muons per spill of 4e13 p.o.t.).
Every muon of types 0-6 (all that reach z = 30.6 m; type 7 stops before) as its stored path A (30.6) -> K (bounce) -> B
(82 m; for type 6 'stops_between' the stored end point), crossed with the UBT planes and the SBT shell. Weighted
histograms are filled on the fly (the store has 770 M rows; types 5 and 6, 328 M rows,
are subsampled to --frac56 = 20 % with weights x 5; --seeds N uses the first N of the 3220 spills, weights x 3220 / N).

Groups: all (types 0-6), sst1 (0-2: in the SST1 acceptance), sel (1: p > 1 GeV and timing = the muons entering the
combinatorial background selection), vetoonly (3), outside (4-6: miss SST1 and do not cross a vessel wall / the window).

    python3 muons_only.py [--max-rows N]   -> hist_mu_only.npz
"""
import argparse
import time

import numpy as np

import common as C
from veto import io, paths

COLS = ("x_30", "y_30", "z_30", "px_30", "py_30", "pz_30", "kx", "ky", "kz", "x_82", "y_82", "z_82",
        "px_82", "py_82", "pz_82", "weight")
GROUPS = {"all": (0, 1, 2, 3, 4, 5, 6), "sst1": (0, 1, 2), "sel": (1,), "vetoonly": (3,), "outside": (4, 5, 6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None, help="use only the first N spills (rows are ordered by run); weights x 3220 / N")
    ap.add_argument("--chunk", type=int, default=20_000_000)
    ap.add_argument("--frac56", type=float, default=0.2, help="random fraction of types 5 / 6 kept (weights / frac)")
    a = ap.parse_args()
    s, meta = io.open_store(paths.store_dir(paths.MUON_DATASET))
    n = len(s["type"]) if a.max_rows is None else a.max_rows
    n_seeds = meta["seeds"]
    scale = 1.0
    if a.seeds:
        n = int(np.searchsorted(s["run"], a.seeds))          # first row of run a.seeds
        scale = n_seeds / a.seeds
    rng = np.random.default_rng(11)
    H = C.Hists()
    tot_w = np.zeros(8)
    t0 = time.time()
    for c0 in range(0, n, a.chunk):
        c1 = min(n, c0 + a.chunk)
        typ = np.asarray(s["type"][c0:c1])
        tot_w += np.bincount(typ, weights=np.asarray(s["weight"][c0:c1]) * scale, minlength=8)[:8]
        keep = np.nonzero((typ <= 4) | (((typ == 5) | (typ == 6)) & (rng.random(len(typ)) < a.frac56)))[0]
        d = {k: np.asarray(s[k][c0:c1])[keep].astype(np.float64) for k in COLS}
        d["weight"] = np.where(typ[keep] >= 5, d["weight"] / a.frac56, d["weight"]) * scale
        typ = typ[keep]
        m = len(keep)
        A = np.stack([d["x_30"], d["y_30"], d["z_30"]], 1)
        K = np.stack([d["kx"], d["ky"], d["kz"]], 1)
        B = np.stack([d["x_82"], d["y_82"], d["z_82"]], 1)
        pa = np.sqrt(d["px_30"] ** 2 + d["py_30"] ** 2 + d["pz_30"] ** 2)
        pb = np.sqrt(d["px_82"] ** 2 + d["py_82"] ** 2 + d["pz_82"] ** 2)
        bnc = np.isfinite(K[:, 2])
        P = np.concatenate([A, K[bnc]]); Q = np.concatenate([np.where(bnc[:, None], K, B), B[bnc]])
        row = np.concatenate([np.arange(m), np.nonzero(bnc)[0]])
        pseg = np.concatenate([pa, pb[bnc]])
        good = np.all(np.isfinite(P), 1) & np.all(np.isfinite(Q), 1)
        P, Q, row, pseg = P[good], Q[good], row[good], pseg[good]
        w = d["weight"]
        n_pl = {}
        ub = C.ubt(P, Q)
        for name, (i, x, y, inside) in ub.items():
            r = row[i]
            for g, ts in GROUPS.items():
                mg = np.isin(typ[r], ts)
                C.fill_ubt(H, g, name, x[mg], y[mg], inside[mg], pseg[i][mg], w[r][mg])
            n_pl[name] = np.bincount(r[inside], minlength=m)
        # SBT shell pieces; prefilter: the segment's bounding box reaches the vessel region grown by 0.3 m
        hx1, hy1 = C.geo.vessel_half_widths(np.minimum(np.maximum(P[:, 2], Q[:, 2]), C.geo.Z_VESSEL_MAX))
        cand = ((np.maximum(P[:, 2], Q[:, 2]) > C.geo.Z_VESSEL_MIN) & (np.minimum(P[:, 2], Q[:, 2]) < C.geo.Z_VESSEL_MAX)
                & (np.minimum(P[:, 0], Q[:, 0]) < hx1 + 0.3) & (np.maximum(P[:, 0], Q[:, 0]) > -hx1 - 0.3)
                & (np.minimum(P[:, 1], Q[:, 1]) < hy1 + 0.3) & (np.maximum(P[:, 1], Q[:, 1]) > -hy1 - 0.3))
        ci = np.nonzero(cand)[0]
        pc = C.sbt_pieces(P[ci], Q[ci], rng)
        sg = ci[pc["seg"].astype(int)]
        r = row[sg]
        fired = pc["fired"].astype(bool)
        for g, ts in GROUPS.items():
            mg = np.isin(typ[r], ts)
            C.fill_sbt(H, g, pc["face"][mg], pc["ring"][mg], pc["z"][mg], pc["across"][mg], fired[mg], pseg[sg][mg],
                       w[r][mg], edep=pc["edep"][mg])
        if fired.any():
            fc = np.unique(np.stack([r[fired], pc["cell"][fired].astype(np.int64)], 1), axis=0)
            nf = np.bincount(fc[:, 0], minlength=m)
        else:
            nf = np.zeros(m, int)
        for g, ts in GROUPS.items():
            mg = np.isin(typ, ts)
            C.fill_events(H, g, n_pl["straw"][mg], n_pl["tile"][mg], nf[mg], w[mg])
        print(f"rows {c1:,}/{n:,}: kept {m:,}, {time.time() - t0:.0f} s", flush=True)
    H["type_weight_all"] = tot_w
    H["rows_used"] = np.array(n)
    H["rows_total"] = np.array(len(s["type"]))
    H["frac56"] = np.array(a.frac56)
    H["seeds_used"] = np.array(a.seeds or n_seeds)
    H.save(C.OUT + "/hist_mu_only.npz")
    print("wrote hist_mu_only.npz")


if __name__ == "__main__":
    main()
