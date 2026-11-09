#!/usr/bin/env python3
"""MS7 K_L / K_S / Lambda events for the event page: from shield_dis outputs (one per seed), the forced-decay copies
(code 4) with the parent muon, the interaction vertex in the iron, every particle at the shield exit plane, the V0 line
from the plane to its decay point and the decay products. Two samples: random decays (with >= 2 charged daughters) and
every copy that passes the selection except the IP cut (vertex pair: DOCA, fiducial, true IP < 3 m; <= 2 tracks), with
its measured IP. Plus weighted histograms of where the V0 decay (z, x-y) and of the IP.

    python3 bin/ms7_events_viz.py "results/shield_dis/shield_muon_g4v112_s1*.npz" --out <dir>/ms7_events.json
"""
import _bootstrap  # noqa: F401
import argparse
import glob
import json

import numpy as np

from run_interactions import load_npz
from veto import geometry as G
from veto import interactions as ix
from veto import reco
from veto import shield_iron as si
from veto import vertex_selection as vsel

ZE = np.linspace(29.9, 81.13, 52)


def event(inter, part, by_int, k, ip=None):
    """One forced-decay copy k: parent muon, vertex, plane particles, V0 line, daughters."""
    par = int(inter["parent"][k])
    rows = by_int.get(k, np.zeros(0, int))
    prow = by_int.get(par, np.zeros(0, int))
    v0 = int(inter["v0"][k])
    zd = float(inter["z_decay"][k])
    dau = rows[np.abs(part["v"][rows, 2] - si.Z_PLANE) > 0.05]
    plane = rows[np.abs(part["v"][rows, 2] - si.Z_PLANE) <= 0.05]
    dv = part["v"][dau[0]] if len(dau) else np.array([np.nan, np.nan, zd])
    # the V0 at the plane: the parent's particle of that pdg whose line passes closest to the decay point
    cand = prow[part["pdg"][prow] == v0]
    best, vline = None, None
    for c in cand:
        p, v = part["p"][c, :3], part["v"][c]
        t = (dv[2] - v[2]) / p[2]
        d = np.hypot(v[0] + p[0] * t - dv[0], v[1] + p[1] * t - dv[1])
        if best is None or d < best:
            best, vline = d, (v, p)
    P = lambda r, tag: [int(part["pdg"][r]), *np.round(part["v"][r], 3).tolist(), *np.round(part["p"][r, :3], 4).tolist(), tag]
    return dict(w=float(inter["weight"][k] * ix.SPILLS_15Y), v0=v0, ip=None if ip is None else round(float(ip), 4),
                vtx=np.round([inter["x"][k], inter["y"][k], inter["z"][k]], 3).tolist(), E_mu=round(float(inter["E"][k]), 2),
                u_mu=np.round([inter["ux"][k], inter["uy"][k], inter["uz"][k]], 5).tolist(), q_mu=int(-np.sign(inter["pdg_parent"][k])),
                dec=np.round(dv, 3).tolist(),
                v0line=None if vline is None else [*np.round(vline[0], 3).tolist(), *np.round(vline[1], 4).tolist()],
                parts=[P(r, 0) for r in plane] + [P(r, 1) for r in dau])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files"); ap.add_argument("--out", required=True)
    ap.add_argument("--n-random", type=int, default=400); ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    files = sorted(glob.glob(a.files))
    rng = np.random.default_rng(a.seed)
    cfg = reco.RecoConfig(seed=1, multiple_scattering=True, ip_max=vsel.IP_CUTS["full"])
    H = dict(z_all=np.zeros(len(ZE) - 1), z_pair=np.zeros(len(ZE) - 1), z_invessel=np.zeros(len(ZE) - 1))
    XY = np.zeros((40, 40)); xyr = 4.0
    rand, sel = [], []
    tot = dict(decays=0.0, invessel=0.0, two_charged=0.0, pair=0.0)
    for f in files:
        inter, part = load_npz(f)
        nf = len(files)
        o = np.argsort(part["int"], kind="stable")
        ints, starts = np.unique(part["int"][o], return_index=True)
        ends = np.r_[starts[1:], len(o)]
        by_int = {int(i): o[s:e] for i, s, e in zip(ints, starts, ends)}
        c4 = np.nonzero(inter["code"] == 4)[0]
        w15 = inter["weight"] * ix.SPILLS_15Y / nf
        zd = inter["z_decay"][c4]
        H["z_all"] += np.histogram(zd, ZE, weights=w15[c4])[0]
        tot["decays"] += w15[c4].sum()
        # decay point and inside the vessel
        dxy = np.full((len(c4), 2), np.nan)
        nchg = np.zeros(len(c4), int)
        for j, k in enumerate(c4):
            r = by_int.get(int(k), np.zeros(0, int))
            d = r[np.abs(part["v"][r, 2] - si.Z_PLANE) > 0.05]
            if len(d):
                dxy[j] = part["v"][d[0], :2]
                nchg[j] = int((ix.charge(part["pdg"][d]) != 0).sum())
        hx, hy = G.vessel_half_widths(zd)
        inv = (np.abs(dxy[:, 0]) < hx) & (np.abs(dxy[:, 1]) < hy)
        H["z_invessel"] += np.histogram(zd[inv], ZE, weights=w15[c4][inv])[0]
        tot["invessel"] += w15[c4][inv].sum(); tot["two_charged"] += w15[c4][inv & (nchg >= 2)].sum()
        XY += np.histogram2d(dxy[:, 0], dxy[:, 1], (np.linspace(-xyr, xyr, 41),) * 2, weights=w15[c4])[0]
        # random decays with >= 2 charged daughters
        pool = c4[(nchg >= 2)]
        for k in rng.choice(pool, min(a.n_random // nf + 1, len(pool)), replace=False):
            rand.append(event(inter, part, by_int, int(k)))
        # selection except the IP cut
        m = inter["code"] == 4
        idx = np.nonzero(m)[0]; remap = np.full(len(m), -1); remap[idx] = np.arange(len(idx))
        keep = m[part["int"]]
        Pp = {kk: v[keep] for kk, v in part.items()}; Pp["int"] = remap[Pp["int"]]
        tr = ix.charged_tracks(Pp)
        s_, pair, _ = ix.select(tr, len(idx), cfg=None)
        if len(s_):
            n_acc = np.bincount(tr["inter"][tr["acc"]], minlength=len(idx))[s_]
            okk = n_acc <= cfg.max_tracks
            s_, pair = s_[okk], pair[okk]
            g0, g1 = pair[:, 0], pair[:, 1]
            ip = vsel.measured_ip(tr["b"][g0], tr["p"][g0], tr["b"][g1], tr["p"][g1], cfg, np.random.default_rng(779))
            kk = idx[s_]
            H["z_pair"] += np.histogram(inter["z_decay"][kk], ZE, weights=w15[kk])[0]
            tot["pair"] += w15[kk].sum()
            for k_, ip_ in zip(kk, ip):
                sel.append(event(inter, part, by_int, int(k_), ip_))
        print(f"{f.split('/')[-1]}: {len(c4):,} decay copies, pairs {len(s_)}", flush=True)
    for e in rand + sel:
        e["w"] /= len(files)
    out = dict(files=files, n_files=len(files), z_edges=ZE.tolist(), hist={k: v.tolist() for k, v in H.items()},
               xy=dict(range=xyr, n=40, h=XY.tolist()), totals=tot, random=rand, selected=sel,
               geometry=dict(shield=dict(z0=si.Z_IRON0, z1=si.Z_SHIELD_END, half_x=si.HALF_X, half_y=si.HALF_Y),
                             z_plane=si.Z_PLANE, z_parent=29.20, vessel=dict(z0=G.Z_VESSEL_MIN, z1=G.Z_VESSEL_MAX,
                             hx0=G.VESSEL_X_IN, hx1=G.VESSEL_X_OUT, hy0=G.VESSEL_Y_IN, hy1=G.VESSEL_Y_OUT),
                             sst1=dict(z=G.Z_SST1, half_x=G.SST1_HALF_X, half_y=G.SST1_HALF_Y), z_target=G.Z_TARGET),
               ip_cut=float(vsel.IP_CUTS["full"]))
    json.dump(out, open(a.out, "w"))
    print(f"wrote {a.out}: {len(rand)} random, {len(sel)} selected-but-IP events; totals per 15 y {tot}")


if __name__ == "__main__":
    main()
