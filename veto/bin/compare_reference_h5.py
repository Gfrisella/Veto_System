#!/usr/bin/env python3
"""The 2018 reference (bin/build_reference_2018.py) against MuonsAndMatter's full_sample_after_target.h5:
  1. totals and flux per weight class (768.75 min. bias, 7.6875 dimuon / photon, 326.16 charm, others)
  2. per class, the flux ratio reference / h5 per production-momentum bin
  3. muon by muon: rows matched on the exact production momentum (float32 px, py, pz): matched fraction, the h5 weight
     of matched rows against FairShip's weight, and what the unmatched muons are (class, mother, process, p, z)

    python3 bin/compare_reference_h5.py [--ref /disk/lhcb_data/sqasim/reference-2018-fairship-506M/reference_2018.h5]
"""

import argparse

import h5py
import numpy as np
import os
SHARED = os.environ.get("VETO_SHARED", "/disk/lhcb_data/sqasim")   # Shah Rukh's shared data (read only)

H5 = "/disk/users/gfrise/MuonsAndMatter/data/muons/full_sample_after_target.h5"
CLASSES = {"min. bias 768.75": 768.75, "dimuon/photon 7.6875": 7.6875, "charm 326.16": 326.16}
PE = np.array([0, 10, 20, 30, 40, 60, 100, 150, 200, 300, 1000])


def cls_of(w):
    c = np.full(len(w), -1, np.int8)
    for i, v in enumerate(CLASSES.values()):
        c[np.abs(w - v) < 0.01 * v] = i
    return c


def keys(px, py, pz):
    return np.stack([px.astype(np.float32).view(np.uint32), py.astype(np.float32).view(np.uint32),
                     pz.astype(np.float32).view(np.uint32)], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=f"{SHARED}/reference-2018-fairship-506M/reference_2018.h5")
    a = ap.parse_args()
    with h5py.File(a.ref, "r") as f:
        R = {k: f[k][:] for k in ("px", "py", "pz", "weight", "w_fairship", "mother", "proc", "z_start", "pdg")}
    with h5py.File(H5, "r") as f:
        H = {k: f[k][:] for k in ("px", "py", "pz", "weight", "pdg")}
    pR = np.sqrt(R["px"].astype(np.float64) ** 2 + R["py"] ** 2 + R["pz"] ** 2)
    pH = np.sqrt(H["px"].astype(np.float64) ** 2 + H["py"] ** 2 + H["pz"] ** 2)
    cR, cH = cls_of(R["w_fairship"]), cls_of(H["weight"])
    print(f"rows: reference {len(pR):,}, h5 {len(pH):,}")
    print(f"total flux per spill: reference {R['weight'].astype(np.float64).sum():.4g}, h5 {H['weight'].astype(np.float64).sum():.4g}")
    names = list(CLASSES) + ["other"]
    for i, name in enumerate(names):
        ir, ih = (cR == i) if i < 3 else (cR < 0), (cH == i) if i < 3 else (cH < 0)
        fr, fh = R["weight"][ir].astype(np.float64).sum(), H["weight"][ih].astype(np.float64).sum()
        print(f"\n{name}: rows reference {ir.sum():,}, h5 {ih.sum():,}; flux reference {fr:.4g}, h5 {fh:.4g}, ratio {fr / max(fh, 1e-300):.3f}")
        hr = np.histogram(pR[ir], PE, weights=R["weight"][ir].astype(np.float64))[0]
        hh = np.histogram(pH[ih], PE, weights=H["weight"][ih].astype(np.float64))[0]
        print("   ratio ref/h5 per p bin: " + "  ".join(f"{x:.0f}-{y:.0f}:{u / v:.2f}" for x, y, u, v in zip(PE[:-1], PE[1:], hr, hh) if v > 0))
    # muon by muon
    kR, kH = keys(R["px"], R["py"], R["pz"]), keys(H["px"], H["py"], H["pz"])
    allk = np.concatenate([kR, kH])
    src = np.concatenate([np.zeros(len(kR), np.int8), np.ones(len(kH), np.int8)])
    idx = np.concatenate([np.arange(len(kR)), np.arange(len(kH))])
    o = np.lexsort((src, allk[:, 2], allk[:, 1], allk[:, 0]))
    ks, ss, ii = allk[o], src[o], idx[o]
    same = np.all(ks[1:] == ks[:-1], axis=1) & (ss[:-1] == 0) & (ss[1:] == 1)
    mr, mh = ii[:-1][same], ii[1:][same]
    print(f"\nmatched on exact production momentum: {len(mr):,} muons = {len(mr) / len(kR):.3f} of the reference, {len(mh) / len(kH):.3f} of the h5")
    wr, wh = R["w_fairship"][mr].astype(np.float64), H["weight"][mh].astype(np.float64)
    ratio = wh / wr
    vals, cnt = np.unique(np.round(ratio, 3), return_counts=True)
    top = np.argsort(cnt)[::-1][:8]
    print("   h5 weight / FairShip weight of matched muons (top values):", list(zip(vals[top].tolist(), cnt[top].tolist())))
    for i, name in enumerate(list(CLASSES)):
        m = cR[mr] == i
        if m.any():
            v, c = np.unique(np.round(wh[m], 4), return_counts=True)
            t = np.argsort(c)[::-1][:4]
            print(f"   {name}: h5 weights of its matched muons {list(zip(v[t].tolist(), c[t].tolist()))}")
    un = np.ones(len(kR), bool)
    un[mr] = False
    print(f"\nunmatched reference muons: {un.sum():,} ({un.mean():.3f}), flux {R['weight'][un].astype(np.float64).sum():.4g} per spill "
          f"({R['weight'][un].astype(np.float64).sum() / R['weight'].astype(np.float64).sum():.3f} of the reference)")
    for i, name in enumerate(names):
        m = un & ((cR == i) if i < 3 else (cR < 0))
        tot = (cR == i).sum() if i < 3 else (cR < 0).sum()
        print(f"   {name}: {m.sum():,} unmatched of {tot:,} ({m.sum() / max(tot, 1):.3f})")
    for lab, arr, edges in (("p [GeV]", pR, PE), ("z_start [m]", R["z_start"].astype(np.float64), np.array([-80, -70.85, -70.7, -70.5, -70.0, -69.5, -68, -60]))):
        a_, _ = np.histogram(arr[un], edges)
        b_, _ = np.histogram(arr, edges)
        print(f"   unmatched fraction vs {lab}: " + "  ".join(f"{x:g}..{y:g}:{u / max(v, 1):.3f}" for x, y, u, v in zip(edges[:-1], edges[1:], a_, b_) if v > 0))
    for lab, arr in (("mother", R["mother"]), ("process", R["proc"])):
        v, c = np.unique(arr[un], return_counts=True)
        vt, ct = np.unique(arr, return_counts=True)
        tot = dict(zip(vt.tolist(), ct.tolist()))
        t = np.argsort(c)[::-1][:10]
        print(f"   unmatched by {lab}: " + "  ".join(f"{v[j]}:{c[j]:,}/{tot[v[j]]:,}" for j in t))


if __name__ == "__main__":
    main()
