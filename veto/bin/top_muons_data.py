#!/usr/bin/env python3
"""Compact data of the "top muons" study (fast-veto-sim scripts/plot_top_muons.py) for the live target talk: which input
muons make the muon-nuclear interaction rate in the decay vessel, per design-optimisation dataset.

Contribution of a row = weight x (muon-nuclear interaction probability in the vessel's helium + PVC), the path from z = 30.6 m
to z = 82 m (or to the bounce point and on), as plot_top_muons.py; summed over the rows of the same input muon (evt_n).
SST1 types (type <= 3) only.

    python3 bin/top_muons_data.py --fvs /disk/users/sqasim/workspace/uzh/fast-veto-sim --store <store dir> \
        --datasets gamma=design-optimization-dataset-gamma-2026-10-17 c=... --out results/target_talk/top_muons.npz
Writes per dataset <tag>_*: cumulative share vs rank, (p, theta) source map, x-y map at 30.6 m, share per source momentum
and per type, and the top TOP input muons with a subsample of their copies (x, y, p at 30.6 m).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TOP, KEEP = 20, 3000
PB, TB, XB = np.linspace(0, 400, 81), np.linspace(0, 40, 81), np.linspace(-3, 3, 121)
RANKS = np.unique(np.round(np.geomspace(1, 1e8, 241)).astype(np.int64))


def one(store, V, rates, rng):
    E = np.array(rates["E_grid"])
    xs = {k: np.array(v) for k, v in rates["xs_per_m"].items()}
    col = lambda k: np.load(store / f"{k}.npy", mmap_mode="r")
    typ_all = np.asarray(col("type"))
    idx = np.nonzero(typ_all <= 3)[0]
    g = lambda k: np.asarray(col(k)[idx], np.float64)
    D = {k: g(k) for k in ("x_30", "y_30", "z_30", "px_30", "py_30", "pz_30", "x_82", "y_82", "z_82", "kx", "ky", "kz",
                           "px_82", "py_82", "pz_82", "px_start", "py_start", "pz_start", "weight", "scale")}
    q = np.where(np.asarray(col("pdg")[idx]) > 0, -13, 13)
    evt = np.asarray(col("evt_n")[idx])
    typ = typ_all[idx]
    src = np.asarray(col("source")[idx]) if (store / "source.npy").exists() else np.zeros(len(idx), np.int8)
    A = np.stack([D["x_30"], D["y_30"], D["z_30"]], 1)
    Bp = np.stack([D["x_82"], D["y_82"], D["z_82"]], 1)
    K = np.stack([D["kx"], D["ky"], D["kz"]], 1)
    pA = np.linalg.norm(np.stack([D["px_30"], D["py_30"], D["pz_30"]], 1), axis=1)
    pB = np.linalg.norm(np.stack([D["px_82"], D["py_82"], D["pz_82"]], 1), axis=1)
    p0 = np.linalg.norm(np.stack([D["px_start"], D["py_start"], D["pz_start"]], 1), axis=1)
    th0 = 1e3 * np.hypot(D["px_start"], D["py_start"]) / D["pz_start"]

    def sig(vol, E_):
        a = np.interp(np.log(np.maximum(E_, 0.25)), np.log(E), xs[f"{vol}_13"])
        b = np.interp(np.log(np.maximum(E_, 0.25)), np.log(E), xs[f"{vol}_-13"])
        return np.where(q == 13, a, b)

    bounced = np.isfinite(K[:, 2])
    Q1 = np.where(bounced[:, None], K, Bp)
    d1 = Q1 - A
    ok = d1[:, 2] > 1e-6
    P1 = A - d1 * (0.05 / np.where(ok, d1[:, 2], 1))[:, None]
    xint = np.zeros(len(idx))
    for P, Q, Em, use in ((P1, Q1, pA, ok), (K, Bp, pB, bounced & (Bp[:, 2] > K[:, 2]))):
        if use.any():
            iv = V.piece_intervals(P[use], Q[use])
            xint[use] += sig("helium", Em)[use] * (iv["helium"][1] - iv["helium"][0])
            xint[use] += sig("pvc", Em)[use] * ((iv["pvc_in"][1] - iv["pvc_in"][0]) + (iv["pvc_out"][1] - iv["pvc_out"][0]))
    v = D["weight"] * xint
    R = v.sum()
    uniq, inv = np.unique(evt, return_inverse=True)
    contrib = np.bincount(inv, weights=v)
    order = np.argsort(-contrib)
    cum = np.cumsum(contrib[order]) / R
    rk = RANKS[RANKS <= len(cum)]
    first = np.unique(inv, return_index=True)[1]
    out = dict(rate=R, n_input=len(uniq), n_rows=len(idx), rank=rk, cum=cum[rk - 1],
               src_map=np.histogram2d(p0[first], th0[first], [PB, TB], weights=contrib / R)[0],
               xy_map=np.histogram2d(A[:, 0], A[:, 1], [XB, XB], weights=v / R)[0],
               p_src=np.stack([np.histogram(p0, PB, weights=v * (src == s) / R)[0] for s in range(int(src.max()) + 1)]),
               type_share=np.array([v[typ == t].sum() / R for t in range(4)]))
    top = uniq[order[:TOP]]
    info, cx, cy, cp, ci = [], [], [], [], []
    for r, e in enumerate(top):
        m = np.nonzero(evt == e)[0]
        info.append(dict(rank=r + 1, evt_n=int(e), share=float(contrib[order[r]] / R), rows=int(len(m)),
                         types={int(t): int(np.sum(typ[m] == t)) for t in np.unique(typ[m])}, charge=int(-np.sign(q[m][0])),
                         p_start=float(p0[m][0]), theta_mrad=float(th0[m][0]), source=int(src[m][0]),
                         p30_mean=float(pA[m].mean()), p30_rms=float(pA[m].std())))
        s = rng.choice(m, min(KEEP, len(m)), replace=False)
        cx.append(A[s, 0]); cy.append(A[s, 1]); cp.append(pA[s]); ci.append(np.full(len(s), r, np.int16))
    out.update(top=json.dumps(info), top_x=np.concatenate(cx).astype(np.float32), top_y=np.concatenate(cy).astype(np.float32),
               top_p=np.concatenate(cp).astype(np.float32), top_i=np.concatenate(ci))
    print(f"{store.name}: rate {R:.4g} from {len(uniq):,} input muons ({len(idx):,} rows); top 10 {cum[9]:.3f}, "
          f"top 100 {cum[min(99, len(cum) - 1)]:.3f}, top 1000 {cum[min(999, len(cum) - 1)]:.3f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fvs", required=True, help="fast-veto-sim checkout (fastveto.vessel, data/muon_dis/rates.json)")
    ap.add_argument("--store", required=True)
    ap.add_argument("--datasets", nargs="+", required=True, help="tag=dataset name")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.fvs)
    from fastveto import vessel as V
    rates = json.load(open(Path(a.fvs) / "data/muon_dis/rates.json"))
    rng = np.random.default_rng(1)
    res = dict(PB=PB, TB=TB, XB=XB, tags=json.dumps([d.split("=")[0] for d in a.datasets]),
               names=json.dumps([d.split("=")[1] for d in a.datasets]),
               vessel_hw=np.array(V.half_widths(V.Z_VESSEL_MIN)))
    for d in a.datasets:
        tag, name = d.split("=")
        for k, x in one(Path(a.store) / name, V, rates, rng).items():
            res[f"{tag}_{k}"] = x
        np.savez_compressed(a.out, **res)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
