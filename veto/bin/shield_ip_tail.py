#!/usr/bin/env python3
"""MS7 V0 background below the IP cut by extrapolation: the forced-decay copies (code 4) of shield_dis outputs through
every selection step except the measured-IP cut (vertex pair: DOCA, fiducial, true IP < 3 m; <= 2 tracks in SST1), then
the rate below IP = 0.1 / 0.25 / 0.5 / 1 / 2 m, averaged over the files (one per seed). Near IP = 0 the rate grows ~ IP^2.

    python3 bin/shield_ip_tail.py "results/shield_dis/shield_muon_g4v112_s1*.npz"
"""
import sys, glob
import _bootstrap  # noqa
import numpy as np
from run_interactions import load_npz
from veto import interactions as ix, reco, vertex_selection as vsel
cfg = reco.RecoConfig(seed=1, multiple_scattering=True, ip_max=vsel.IP_CUTS["full"])
CUTS = [0.1, 0.25, 0.5, 1.0, 2.0]
files = sorted(glob.glob(sys.argv[1]))
acc = {c: 0.0 for c in CUTS}; acc2 = {c: 0.0 for c in CUTS}; npass = {c: 0 for c in CUTS}; nparents = {c: set() for c in CUTS}
tot_pair = 0.0; tot_v0 = 0.0
by_v0 = {}                                    # D16: per decayed V0 species, all-but-IP and IP < 0.5 / 1 m
for f in files:
    inter, part = load_npz(f)
    nf = len(files)
    m = inter["code"] == 4
    idx = np.nonzero(m)[0]; remap = np.full(len(m), -1); remap[idx] = np.arange(len(idx))
    keep = m[part["int"]]
    I = {k: v[m] for k, v in inter.items()}; Pp = {k: v[keep] for k, v in part.items()}; Pp["int"] = remap[Pp["int"]]
    w15 = I["weight"] * ix.SPILLS_15Y / nf          # average over the files
    tot_v0 += w15.sum()
    tr = ix.charged_tracks(Pp)
    sel, pair, vtx = ix.select(tr, len(w15), cfg=None)          # DOCA, fiducial, true IP < 3 m; no measured-IP cut
    if not len(sel): continue
    n_acc = np.bincount(tr["inter"][tr["acc"]], minlength=len(w15))[sel]
    ok = n_acc <= cfg.max_tracks
    sel, pair = sel[ok], pair[ok]
    g0, g1 = pair[:, 0], pair[:, 1]
    ip = vsel.measured_ip(tr["b"][g0], tr["p"][g0], tr["b"][g1], tr["p"][g1], cfg, np.random.default_rng(779))
    tot_pair += w15[sel].sum()
    for code in np.unique(I["v0"][sel]):
        mv = I["v0"][sel] == code
        b = by_v0.setdefault(int(code), [0.0, 0.0, 0.0, 0])
        b[0] += w15[sel][mv].sum(); b[1] += w15[sel][mv & (ip < 0.5)].sum(); b[2] += w15[sel][mv & (ip < 1.0)].sum(); b[3] += int(mv.sum())
    for c in CUTS:
        q = ip < c
        acc[c] += w15[sel][q].sum(); acc2[c] += (w15[sel][q] ** 2).sum(); npass[c] += int(q.sum())
        nparents[c].update((f + ":" + str(p)) for p in I["parent"][sel][q])
    print(f"{f.split('/')[-1]}: {len(sel)} pairs passing everything but IP", flush=True)
print(f"\nV0 decays in vessel {tot_v0:.3g} / 15 y; vertex pair + <= 2 tracks (no IP cut) {tot_pair:.3g} / 15 y")
for c in CUTS:
    print(f"  measured IP < {c:4.2f} m: {acc[c]:10.3g} +- {np.sqrt(acc2[c]):8.2g} / 15 y  ({npass[c]} copies, {len(nparents[c])} parent interactions)")
for code, (b0, b5, b10, n) in sorted(by_v0.items()):
    print(f"  v0 {code:6d}: all-but-IP {b0:9.3g}, IP < 0.5 m {b5:9.3g}, IP < 1 m {b10:9.3g} / 15 y  ({n} copies)")
