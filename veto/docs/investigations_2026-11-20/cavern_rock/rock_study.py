#!/usr/bin/env python3
"""Cavern-rock combinatorial investigation (20 Nov 2026): what the rock tracks are, their momentum / direction / origin at
SST1, and the step-by-step cut flow of mu x rock next to mu x mu, with the same pair cuts as bin/rock_combinatorial.py
(benchmark_designs._pair_cut: DOCA < 2 cm, vertex in the vessel frustum, true IP < 3 m; measured IP < 2.5 m / 10 cm).

Read-only use of the veto library (worktree nov4-source). Output: data.npz + cutflow.json in this folder; plots by plots.py.

    cd <checkout>; export VETO_THREADS=8; source bin/batch/env.sh
    python3 <this folder>/rock_study.py --mu-sample 400000 --mumu-sample 80000
"""
import argparse
import json
import os
import sys
import time

W = "/disk/gfs_lhcb/gfrise/Study_on_Veto_System/Optimization_Veto_System/.claude/worktrees/nov4-source/veto"
sys.path[:0] = [W + "/bin", W + "/lib"]
import _bootstrap  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rock_combinatorial import muon_tracks  # noqa: E402
from rock_select import build  # noqa: E402
from run_reco import SPILLS_15Y, load_inputs  # noqa: E402
from veto import geometry as G  # noqa: E402
from veto import interactions as ix  # noqa: E402
from veto import paths, reco  # noqa: E402
from veto import muon_types as mt  # noqa: E402
from veto import vertex_selection as vsel  # noqa: E402
from veto.config import DEFAULT_CONFIG  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(int(os.environ.get("VETO_THREADS", 8)))


def rock_all(path, seeds):
    """Every charged rock particle moving downstream (charged_tracks), with its draw info."""
    z = np.load(path)
    inter, part = build(z, "muon", seeds)
    tr = ix.charged_tracks(part)
    pdg = part["pdg"][tr["row"]]
    w = inter["weight"][tr["inter"]]
    return dict(pdg=pdg, q=tr["q"].astype(float), v=tr["v"], p=tr["p"], b=tr["b"], acc=tr["acc"], good=tr["good"], w=w,
                draw=tr["inter"], surf=inter["code"][tr["inter"]] - 5, ivz=inter["z"][tr["inter"]],
                ivx=inter["x"][tr["inter"]], ivy=inter["y"][tr["inter"]]), inter, part


def vessel_entry(v, p, dz=0.05):
    """First point of the straight line (from the rock exit v, direction p) inside the vessel frustum: z, the face it
    crosses (0 side +-x, 1 top/bottom +-y, 2 entrance window), angle of the line to that face (rad)."""
    u = p / np.linalg.norm(p, axis=1, keepdims=True)
    n = len(v)
    zin = np.full(n, np.nan); face = np.full(n, -1); ang = np.full(n, np.nan)
    todo = np.arange(n)
    s = np.zeros(n)
    prev_in = np.zeros(n, bool)
    for k in range(int(80 / dz)):
        if not len(todo):
            break
        s_ = (k + 1) * dz
        x = v[todo] + s_ * u[todo]
        ins = G.inside_vessel(x[:, 0], x[:, 1], x[:, 2])
        hit = todo[ins]
        if len(hit):
            xh = x[ins]
            zin[hit] = xh[:, 2]
            hx, hy = G.vessel_half_widths(xh[:, 2])
            dxw, dyw, dzw = hx - np.abs(xh[:, 0]), hy - np.abs(xh[:, 1]), xh[:, 2] - G.Z_VESSEL_MIN
            f = np.argmin(np.stack([dxw, dyw, dzw], 1), 1)
            face[hit] = f
            uh = u[hit]
            ang[hit] = np.where(f == 0, np.arcsin(np.abs(uh[:, 0])), np.where(f == 1, np.arcsin(np.abs(uh[:, 1])),
                                                                                 np.arcsin(np.abs(uh[:, 2]))))
        todo = todo[~ins & (x[:, 2] < G.Z_VESSEL_MAX)]
    return zin, face, ang


def pair_flow(A, B, same, cfg, block=4096, keep_cand=True):
    """Step-by-step weighted cut flow (sum wA wB, per spill^2 -> x fac = per 15 y) for pairs A x B (same: i < j)."""
    def T(d):
        return (torch.as_tensor(np.stack([d["bx"], d["by"], d["bz"]], 1)), torch.as_tensor(np.stack([d["bpx"], d["bpy"], d["bpz"]], 1)),
                torch.as_tensor(d["q"]), torch.as_tensor(d["weight"]))
    rA, pA, qA, wA = T(A)
    rB, pB, qB, wB = T(B)
    nA, nB = len(rA), len(rB)
    steps = ["all pairs", "opposite charge", "DOCA < 2 cm", "vertex z in vessel", "vertex inside frustum", "true IP < 3 m"]
    S = {k: 0.0 for k in steps}; N = {k: 0 for k in steps}
    fid5 = 0.0
    cand = []
    for i0 in range(0, nA, block):
        for j0 in range(i0 if same else 0, nB, block):
            ni, nj = min(block, nA - i0), min(block, nB - j0)
            ia = torch.arange(i0, i0 + ni)[:, None].expand(ni, nj).reshape(-1)
            jb = torch.arange(j0, j0 + nj)[None, :].expand(ni, nj).reshape(-1)
            if same:
                m = ia < jb
                ia, jb = ia[m], jb[m]
            w = wA[ia] * wB[jb]
            S["all pairs"] += float(w.sum()); N["all pairs"] += len(ia)
            m = qA[ia] * qB[jb] < 0
            ia, jb, w = ia[m], jb[m], w[m]
            S["opposite charge"] += float(w.sum()); N["opposite charge"] += len(ia)
            r1, r2, p1, p2 = rA[ia], rB[jb], pA[ia], pB[jb]
            dr = r1 - r2
            cr = torch.linalg.cross(p1, p2)
            doca = (dr * cr).sum(1).abs() / cr.norm(dim=1)
            m = doca < 0.02
            ia, jb, w, r1, r2, p1, p2, dr, doca = ia[m], jb[m], w[m], r1[m], r2[m], p1[m], p2[m], dr[m], doca[m]
            S["DOCA < 2 cm"] += float(w.sum()); N["DOCA < 2 cm"] += len(ia)
            aa, bb, cc = (p1 * p1).sum(1), (p1 * p2).sum(1), (p2 * p2).sum(1)
            dd, ee = (p1 * dr).sum(1), (p2 * dr).sum(1)
            den = aa * cc - bb * bb
            v = 0.5 * (r1 + ((bb * ee - cc * dd) / den)[:, None] * p1 + r2 + ((aa * ee - bb * dd) / den)[:, None] * p2)
            m = (v[:, 2] > G.Z_VESSEL_MIN) & (v[:, 2] < G.Z_VESSEL_MAX)
            S["vertex z in vessel"] += float(w[m].sum()); N["vertex z in vessel"] += int(m.sum())
            hx = G.VESSEL_X_IN + (G.VESSEL_X_OUT - G.VESSEL_X_IN) * (v[:, 2] - G.Z_VESSEL_MIN) / 50
            hy = G.VESSEL_Y_IN + (G.VESSEL_Y_OUT - G.VESSEL_Y_IN) * (v[:, 2] - G.Z_VESSEL_MIN) / 50
            m &= (v[:, 0].abs() < hx) & (v[:, 1].abs() < hy)
            S["vertex inside frustum"] += float(w[m].sum()); N["vertex inside frustum"] += int(m.sum())
            mom = p1 + p2
            dzt = G.Z_TARGET - v[:, 2]
            ip = torch.hypot(v[:, 0] + mom[:, 0] / mom[:, 2] * dzt, v[:, 1] + mom[:, 1] / mom[:, 2] * dzt)
            m &= ip < 3.0
            S["true IP < 3 m"] += float(w[m].sum()); N["true IP < 3 m"] += int(m.sum())
            f5 = m & ((v[:, 0].abs() < hx - 0.05) & (v[:, 1].abs() < hy - 0.05) & (v[:, 2] > G.Z_VESSEL_MIN + 0.05)
                      & (v[:, 2] < G.Z_VESSEL_MAX - 0.05))
            fid5 += float(w[f5].sum())
            if keep_cand and m.any():
                cand.append(np.stack([ia[m].numpy(), jb[m].numpy()], 1));
                cand.append(None)  # placeholder to keep vertex / ip next to it
                cand[-1] = (v[m].numpy(), ip[m].numpy(), doca[m].numpy(), w[m].numpy(), f5[m].numpy())
    pairs = np.concatenate([c for c in cand[0::2]]) if cand else np.zeros((0, 2), int)
    extra = [np.concatenate([c[k] for c in cand[1::2]]) for k in range(5)] if cand else [np.zeros((0, 3))] + [np.zeros(0)] * 4
    vtx, tip, doca, w, f5 = extra
    # measured IP (same smearing as the chain)
    st = lambda d, i: (np.stack([d["bx"][i], d["by"][i], d["bz"][i]], 1), np.stack([d["bpx"][i], d["bpy"][i], d["bpz"][i]], 1))
    mip = vsel.measured_ip(*st(A, pairs[:, 0]), *st(B, pairs[:, 1]), cfg, np.random.default_rng(777)) if len(pairs) else np.zeros(0)
    for name, cut in (("measured IP < 2.5 m", 2.5), ("measured IP < 10 cm", 0.10)):
        S[name] = float(w[mip < cut].sum()); N[name] = int((mip < cut).sum())
    S["(fiducial 5 cm, true IP < 3 m)"] = fid5
    return dict(S=S, N=N), dict(pairs=pairs, vtx=vtx, tip=tip, mip=mip, doca=doca, w=w, fid5=f5.astype(bool))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu-sample", type=int, default=400_000)
    ap.add_argument("--mumu-sample", type=int, default=80_000)
    ap.add_argument("--seeds", type=int, default=300)
    a = ap.parse_args()
    t0 = time.time()
    cfg = reco.RecoConfig(seed=1, multiple_scattering=True, cone=0.0, ip_max=0.10)
    fac = 2 * DEFAULT_CONFIG["time_window"] / cfg.spill_s * SPILLS_15Y
    rock_path = os.environ.get("ROCK_MU", "/disk/lhcb_data/sqasim/rock_dis/apply_D6_straw_mod_draws.npz")
    R, inter, part = rock_all(rock_path, a.seeds)
    print(f"rock: {len(inter['weight']):,} draws ({inter['weight'].sum():.4g} interactions / spill in the file), "
          f"{len(R['pdg']):,} charged downstream, acc {R['w'][R['acc']].sum():.4g}/spill, good {R['w'][R['good']].sum():.4g}/spill "
          f"({R['good'].sum()} rows); {time.time() - t0:.0f} s", flush=True)
    zin, face, ang = vessel_entry(R["v"][R["good"]], R["p"][R["good"]])
    g = R["good"]
    rk = dict(bx=R["b"][g, 0], by=R["b"][g, 1], bz=R["b"][g, 2], bpx=R["p"][g, 0], bpy=R["p"][g, 1], bpz=R["p"][g, 2],
              q=R["q"][g], weight=R["w"][g])
    mu = muon_tracks(paths.MUON_DATASET, a.mu_sample, seed=5)
    mu["q"] = mt.charge_from_pdg_column(mu["pdg"])
    print(f"muons: {mu['weight'].sum():.4g} good / spill ({len(mu['ax']):,} rows); {time.time() - t0:.0f} s", flush=True)
    mu2 = muon_tracks(paths.MUON_DATASET, a.mumu_sample, seed=11)
    mu2["q"] = mt.charge_from_pdg_column(mu2["pdg"])

    flow_mr, cand_mr = pair_flow(mu, rk, False, cfg)
    print(f"mu x rock flow done {time.time() - t0:.0f} s", flush=True)
    flow_mm, cand_mm = pair_flow(mu2, mu2, True, cfg)
    print(f"mu x mu flow done {time.time() - t0:.0f} s", flush=True)

    # official mu x mu candidates (reco inputs) for the IP-level distributions
    tr_in, inp = load_inputs(paths.RECO_INPUTS)
    c = inp["cand"]; cw = inp["cand_w"].astype(float)
    st = lambda i: (np.stack([tr_in["bx"][i], tr_in["by"][i], tr_in["bz"][i]], 1), np.stack([tr_in["bpx"][i], tr_in["bpy"][i], tr_in["bpz"][i]], 1))
    mip_off = vsel.measured_ip(*st(c[:, 0]), *st(c[:, 1]), cfg, np.random.default_rng(777))
    off = dict(cand_15y=float(cw.sum() * fac), sel_15y=float(cw[mip_off < 0.1].sum() * fac), n=len(c))
    print(f"official mu x mu: {off}", flush=True)

    res = dict(fac_15y=fac, rock_file=rock_path, seeds=a.seeds, n_draws_file=int(len(inter["weight"])),
               interactions_per_spill_in_file=float(inter["weight"].sum()),
               rates_per_spill=dict(muons_good=float(mu["weight"].sum()), rock_charged_downstream=float(R["w"].sum()),
                                    rock_sst1_acc=float(R["w"][R["acc"]].sum()), rock_good=float(R["w"][R["good"]].sum()),
                                    rock_good_rows=int(g.sum())),
               mu_x_rock={k: dict(per_spill2=v, per_15y=v * fac, n=flow_mr["N"].get(k)) for k, v in flow_mr["S"].items()},
               mu_x_mu={k: dict(per_spill2=v, per_15y=v * fac, n=flow_mm["N"].get(k)) for k, v in flow_mm["S"].items()},
               mu_x_mu_official=off, mu_sample=a.mu_sample, mumu_sample=a.mumu_sample)
    json.dump(res, open(f"{OUT}/cutflow.json", "w"), indent=1, default=float)
    sv = dict(
        r_pdg=R["pdg"], r_q=R["q"], r_v=R["v"], r_p=R["p"], r_b=R["b"], r_acc=R["acc"], r_good=R["good"], r_w=R["w"],
        r_surf=R["surf"], r_ivz=R["ivz"], r_ivx=R["ivx"], r_ivy=R["ivy"], r_draw=R["draw"],
        g_zin=zin, g_face=face, g_ang=ang,
        mu_b=np.stack([mu["bx"], mu["by"], mu["bz"]], 1), mu_p=np.stack([mu["bpx"], mu["bpy"], mu["bpz"]], 1), mu_q=mu["q"],
        mu_w=mu["weight"], mu_type=mu["type"],
        mr_pairs=cand_mr["pairs"], mr_vtx=cand_mr["vtx"], mr_tip=cand_mr["tip"], mr_mip=cand_mr["mip"], mr_w=cand_mr["w"], mr_fid5=cand_mr["fid5"],
        mm_pairs=cand_mm["pairs"], mm_vtx=cand_mm["vtx"], mm_tip=cand_mm["tip"], mm_mip=cand_mm["mip"], mm_w=cand_mm["w"], mm_fid5=cand_mm["fid5"],
        mu2_b=np.stack([mu2["bx"], mu2["by"], mu2["bz"]], 1), mu2_p=np.stack([mu2["bpx"], mu2["bpy"], mu2["bpz"]], 1), mu2_q=mu2["q"],
        mu2_w=mu2["weight"],
        off_b=np.stack([tr_in["bx"], tr_in["by"], tr_in["bz"]], 1), off_p=np.stack([tr_in["bpx"], tr_in["bpy"], tr_in["bpz"]], 1),
        off_cand=c, off_w=cw, off_mip=mip_off)
    np.savez_compressed(f"{OUT}/data.npz", **sv)
    print(json.dumps(res, indent=1, default=float))
    print(f"done {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
