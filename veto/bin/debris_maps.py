#!/usr/bin/env python3
"""Rate maps of the EM debris of the muon shield (fastveto.debris), in the binning of bin/make_design_explorer.py,
added to a dataset's muon rate maps: the reco inputs then see muons + debris (bin/replace_rate_maps.py).

Every muon of the dataset that reached z = 30.6 m (type != 7) and left the shield through its iron gets one library
event; its e+- (kinetic > 1 MeV) and photons (> 0.1 MeV) go straight from the exit face to the UBT (z = 30.6, |x| < 0.5)
and the fin slices (outward of the vessel wall). Written as separate em_* maps (em_ubt_e / _g, em_fin_e / _g, the
format of bin/em_debris.py, ISSUES B2 / D12): the muon maps ubt_hist / fin_maps are left unchanged, and
reco.plane_rate_map adds P_e e + P_gamma gamma per plane technology (em_debris.detection_probability). Debris is then
counted once; run_reco --em-debris replaces these em_* maps with the g4shield -> g4vessel ones (the main-chain source).
(Before 2026-11-09 this script added e + 0.05 x gamma into ubt_hist / fin_maps.)

    python3 bin/debris_maps.py --dataset <name> --maps sample_<x>.npz --out sample_<x>_em.npz
"""

import _bootstrap  # noqa: F401

import argparse
import time

import numpy as np

from fastveto import debris
from make_design_explorer import U_BINS, UBT_Y_BINS, Y_BINS, Z_SLICES
from veto import geometry as geo
from veto import io, paths
from veto import muon_types as mt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--maps", required=True, help="the dataset's muon rate maps: results/design_explorer/<this>")
    ap.add_argument("--out", required=True, help="results/design_explorer/<this>: maps with the debris added")
    ap.add_argument("--lib", default=str(paths.results_dir() / "debris_library"))
    ap.add_argument("--chunk", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--every", type=int, default=1, help="use every k-th chunk (seed blocks), weights x k: faster")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    lib = debris.Library(a.lib)
    store, meta = io.open_store(paths.store_dir(a.dataset))
    n = meta["n"]
    nu, ny = len(U_BINS) - 1, len(Y_BINS) - 1
    maps = {k: dict(ubt=np.zeros(len(UBT_Y_BINS) - 1), fin=np.zeros((2, len(Z_SLICES), nu * ny))) for k in ("e", "g")}
    w_mu = w_iron = 0.0
    t0 = time.time()
    for ic, c0 in enumerate(range(0, n, a.chunk)):
        if ic % a.every:
            continue
        c1 = min(c0 + a.chunk, n)
        keep = np.asarray(store["type"][c0:c1]) != 7
        g = lambda col: np.asarray(store[col][c0:c1])[keep].astype(np.float64)
        A = np.stack([g("x_30"), g("y_30"), g("z_30")], 1)
        P = np.stack([g("px_30"), g("py_30"), g("pz_30")], 1)
        w = g("weight") * a.every
        ok = P[:, 2] > 0
        A, P, w = A[ok], P[:, :][ok], w[ok]
        x_exit, u, inside = debris.shield_exit(A, P)
        w_mu += w.sum(); w_iron += w[inside].sum()
        idx = np.nonzero(inside)[0]
        D = debris.draw(lib, np.linalg.norm(P[idx], axis=1), u[idx], x_exit[idx], rng)
        wd = w[idx][D["mu"]]
        ap_ = np.abs(D["pdg"])
        for kind, m in (("e", (ap_ == 11) & (D["ekin"] > 1e-3)), ("g", (D["pdg"] == 22) & (D["ekin"] > 1e-4))):
            x0, d, ww = D["x"][m], D["d"][m], wd[m]
            fwd = d[:, 2] > 0
            x0, d, ww = x0[fwd], d[fwd], ww[fwd]
            # UBT
            t = (geo.Z_UBT - x0[:, 2]) / d[:, 2]
            xu, yu = x0[:, 0] + t * d[:, 0], x0[:, 1] + t * d[:, 1]
            on = (t >= 0) & (np.abs(xu) < geo.UBT_HALF_X)
            maps[kind]["ubt"] += np.histogram(yu[on], UBT_Y_BINS, weights=ww[on])[0]
            # fin slices
            for iz, z in enumerate(Z_SLICES):
                t = (z - x0[:, 2]) / d[:, 2]
                x, y = x0[:, 0] + t * d[:, 0], x0[:, 1] + t * d[:, 1]
                uu = np.abs(x) - mt.side_plane_x(z)
                s = (t >= 0) & (uu >= 0) & (uu < U_BINS[-1]) & (y >= Y_BINS[0]) & (y < Y_BINS[-1])
                iu = (uu[s] / (U_BINS[1] - U_BINS[0])).astype(int)
                iy = ((y[s] - Y_BINS[0]) / (Y_BINS[1] - Y_BINS[0])).astype(int)
                side = (x[s] < 0).astype(int)
                for sd in (0, 1):
                    mm = side == sd
                    maps[kind]["fin"][sd, iz] += np.bincount(iu[mm] * ny + iy[mm], weights=ww[s][mm], minlength=nu * ny)
        print(f"  {c1:,} / {n:,} rows, {len(D['mu']):,} debris particles, {time.time() - t0:.0f} s", flush=True)
    base = dict(np.load(paths.results_dir() / "design_explorer" / a.maps))
    out = dict(base)
    shape = base["fin_maps"].shape
    out["em_ubt_e"], out["em_fin_e"] = maps["e"]["ubt"], maps["e"]["fin"].reshape(shape)
    out["em_ubt_g"], out["em_fin_g"] = maps["g"]["ubt"], maps["g"]["fin"].reshape(shape)
    out["em_source"] = np.array("debris_maps (g4slab library, straight lines)")
    np.savez(paths.results_dir() / "design_explorer" / a.out, **out)
    print(f"muons reaching 30.6 m: {w_mu:.4g} /spill, of which left through the shield iron: {w_iron / w_mu:.3f}")
    for k, lab in (("ubt_hist", "UBT"), ("fin_maps", "fins (all slices)")):
        mu, e, gm = base[k].sum(), out[k + "_e"].sum(), out[k + "_g"].sum()
        print(f"{lab:18s}: muons {mu:.4g} /spill, debris e+- {e:.4g} ({e / mu:.3f} x muons), photons {gm:.4g} "
              f"({gm / mu:.3f} x; x {a.p_gamma} detected = {a.p_gamma * gm / mu:.3f})")
    print(f"wrote results/design_explorer/{a.out}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
