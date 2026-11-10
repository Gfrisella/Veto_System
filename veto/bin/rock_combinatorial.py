#!/usr/bin/env python3
"""Combinatorial background with the cavern-rock products: two unrelated tracks in time, where one or both come out of
the rock (bin/rock_apply.py muon draws, bin/nu_rock.py neutrino draws) instead of the shield. Same chain as the muon-muon
term of bin/run_reco.py evaluate: candidate cuts (DOCA < 2 cm, vertex in the vessel, true IP < 3 m), measured IP < cfg.ip_max,
veto (reco.run_events, own hits + accidental hits), rate = sum of pair weights x 2 dt / spill.

Populations (good tracks: p > 1 GeV, SST1 and timing acceptance): the SST1 muons of the muon dataset (weights per spill) and
the rock tracks (one row per particle, weight per spill of 4e13 of its draw). Pairs: muon x rock (every pair: independent
sources) and rock x rock from different draws; the pairs within one draw are the rock vertex term (bin/rock_select.py).
The muon x muon term is recomputed from the reco inputs with the same code for the ratio.

    bin/remote.sh run bin/rock_combinatorial.py --design designs/modules/D6_straw_mod.json \
        --rock-mu /disk/lhcb_data/sqasim/rock_dis/apply_D6_straw_mod_draws.npz --rock-nu .../nu_rock_D6_straw_mod_draws.npz
"""
import _bootstrap  # noqa: F401

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from benchmark_designs import _pair_cut
from rock_select import build
from run_reco import SPILLS_15Y, load_inputs, take
from veto import design as dsg
from veto import interactions as ix
from veto import io, paths, reco
from veto import muon_types as mt
from veto import vertex_selection as vsel
from veto.config import DEFAULT_CONFIG

COLS = {"ax": "x_30", "ay": "y_30", "az": "z_30", "apz": "pz_30", "kx": "kx", "ky": "ky", "kz": "kz",
        "bx": "x_82", "by": "y_82", "bz": "z_82", "bpx": "px_82", "bpy": "py_82", "bpz": "pz_82",
        "weight": "weight", "type": "type", "pdg": "pdg"}
KEYS = ("ax", "ay", "az", "apz", "kx", "ky", "kz", "bx", "by", "bz", "bpx", "bpy", "bpz", "weight", "type", "pdg")


def muon_tracks(dataset, n_sample, seed=5):
    """Good SST1 muons: a random subsample of n_sample SST1 rows, weights scaled up to keep the total."""
    store, _ = io.open_store(paths.store_dir(dataset))
    sst1 = np.where(np.asarray(store["type"]) <= 2)[0]
    scale = 1.0
    if n_sample and n_sample < len(sst1):
        scale = len(sst1) / n_sample
        sst1 = np.sort(np.random.default_rng(seed).choice(sst1, n_sample, replace=False))
    d = {k: np.asarray(store[c][sst1]).astype(np.float64) for k, c in COLS.items()}
    d["weight"] *= scale
    p = np.sqrt(d["bpx"] ** 2 + d["bpy"] ** 2 + d["bpz"] ** 2)
    good = (p > 1) & (d["bpz"] > 0)
    g = np.nonzero(good)[0]
    good[g] = mt.timing_ok(np.stack([d["bx"][g], d["by"][g], d["bz"][g]], 1),
                           np.stack([d["bpx"][g], d["bpy"][g], d["bpz"][g]], 1), -np.sign(d["pdg"][g]))
    d = {k: v[good] for k, v in d.items()}
    d["draw"] = np.full(len(d["ax"]), -1)
    return d


def rock_tracks(path, kind, seeds, draw_offset):
    inter, part = build(np.load(path), kind, seeds)
    tr = ix.charged_tracks(part)
    g = tr["good"]
    v, b, p = tr["v"][g], tr["b"][g], tr["p"][g]
    nan = np.full(len(v), np.nan)
    return dict(ax=v[:, 0], ay=v[:, 1], az=v[:, 2], apz=p[:, 2], kx=nan, ky=nan, kz=nan.copy(),
                bx=b[:, 0], by=b[:, 1], bz=b[:, 2], bpx=p[:, 0], bpy=p[:, 1], bpz=p[:, 2],
                weight=inter["weight"][tr["inter"][g]],                       # per spill of 4e13
                type=np.full(len(v), 3.0), pdg=-13.0 * tr["q"][g],             # muon convention: pdg 13 has charge -1
                draw=tr["inter"][g] + draw_offset)


def cross_pairs(A, B, dev, block, same=False):
    """Candidate pairs (i in A, j in B) passing the pair cuts; same=True: A is B, i < j, different draws."""
    def tens(d):
        return (torch.as_tensor(np.stack([d["bx"], d["by"], d["bz"]], 1), device=dev),
                torch.as_tensor(np.stack([d["bpx"], d["bpy"], d["bpz"]], 1), device=dev),
                torch.as_tensor(d["pdg"], device=dev), torch.as_tensor(d["draw"], device=dev))
    rA, pA, qA, dA = tens(A)
    rB, pB, qB, dB = tens(B)
    nA, nB = len(rA), len(rB)
    r, p = torch.cat([rA, rB]), torch.cat([pA, pB])
    out_a, out_b, out_v, n_pairs = [], [], [], 0
    for i0 in range(0, nA, block):
        for j0 in range(i0 if same else 0, nB, block):
            ni, nj = min(block, nA - i0), min(block, nB - j0)
            ia = torch.arange(i0, i0 + ni, device=dev)[:, None].expand(ni, nj).reshape(-1)
            jb = torch.arange(j0, j0 + nj, device=dev)[None, :].expand(ni, nj).reshape(-1)
            m = qA[ia] * qB[jb] < 0
            if same:
                m &= (ia < jb) & (dA[ia] != dB[jb])
            ia, jb = ia[m], jb[m]
            n_pairs += len(ia)
            a, bb, v = _pair_cut(ia, jb + nA, r, p)
            out_a.append(a.cpu().numpy()); out_b.append((bb - nA).cpu().numpy()); out_v.append(v.cpu().numpy())
    return np.concatenate(out_a), np.concatenate(out_b), np.concatenate(out_v), n_pairs


def veto_term(tr, cand, cand_w, cand_v, design, inp, cfg, rng):
    """Measured IP cut and veto for candidate pairs (rows of tr); returns per-15-year before / after IP / after veto."""
    dt = DEFAULT_CONFIG["time_window"]
    w = cand_w * 2 * dt / cfg.spill_s * SPILLS_15Y
    out = dict(n_candidates=int(len(cand)), candidates_15y=float(w.sum()))
    if not len(cand):
        return dict(out, selected_15y=0.0, n_selected=0, after_15y=0.0, n_after=0)
    st = lambda i: (np.stack([tr["bx"][i], tr["by"][i], tr["bz"][i]], 1), np.stack([tr["bpx"][i], tr["bpy"][i], tr["bpz"][i]], 1))
    ip = vsel.measured_ip(*st(cand[:, 0]), *st(cand[:, 1]), cfg, np.random.default_rng(777))
    keep = np.nonzero(ip < cfg.ip_max)[0]
    out.update(selected_15y=float(w[keep].sum()), n_selected=int(len(keep)))
    if not len(keep):
        return dict(out, after_15y=0.0, n_after=0, after_upper_15y=float(2.3 * w.mean() * len(keep) / max(len(cand), 1)))
    rows = cand[keep].ravel()
    event = np.repeat(np.arange(len(keep)), 2)
    z_lim = np.repeat(cand_v[keep, 2], 2)
    b = reco.run_events(take(tr, rows), event, z_lim, design, inp, cfg, rng)
    surv = ~b["vetoed"]
    wk = w[keep]
    expl = b["explained"].reshape(-1, 2)
    out.update(after_15y=float(wk[surv].sum()), n_after=int(surv.sum()),
               after_err_15y=float(np.sqrt((wk[surv] ** 2).sum())),
               after_upper_15y=float(2.3 * wk.max()) if not surv.any() else None,
               leg_explained=[float((wk * expl[:, 0]).sum() / wk.sum()), float((wk * expl[:, 1]).sum() / wk.sum())])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", required=True)
    ap.add_argument("--dataset", default="design-optimization-dataset-c-pool-v3-6000seeds-2026-10-31")
    ap.add_argument("--inputs", default="inputs_c_s.npz")
    ap.add_argument("--rock-mu", required=True); ap.add_argument("--seeds", type=int, default=300)
    ap.add_argument("--rock-nu", default=None)
    ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--mu-sample", type=int, default=2_000_000, help="random SST1 muon rows (weights scaled); 0 = all")
    ap.add_argument("--cone", type=float, default=0.0, help="mrad")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = Path(a.design)
    design = dsg.load(p if p.exists() else a.design)
    cfg = reco.RecoConfig(seed=1, multiple_scattering=True, cone=a.cone * 1e-3)
    tr_in, inp = load_inputs(a.inputs)
    mu = muon_tracks(a.dataset, a.mu_sample)
    rk = rock_tracks(a.rock_mu, "muon", a.seeds, 0)
    if a.rock_nu:
        rn = rock_tracks(a.rock_nu, "nu", 1, 10_000_000)
        rk = {k: np.concatenate([rk[k], rn[k]]) for k in rk}
    print(f"good tracks per spill: muons {mu['weight'].sum():.4g} ({len(mu['ax']):,} rows), rock {rk['weight'].sum():.4g} "
          f"({len(rk['ax']):,} rows); {time.time() - t0:.0f} s", flush=True)

    res = {}
    # muon x muon from the reco inputs (the existing combinatorial term), same code
    res["mu x mu"] = veto_term(tr_in, inp["cand"], inp["cand_w"].astype(float), inp["cand_vertex"], design, inp, cfg,
                               np.random.default_rng(1))
    print(f"mu x mu done, {time.time() - t0:.0f} s", flush=True)
    for name, A, B, same in (("mu x rock", mu, rk, False), ("rock x rock", rk, rk, True)):
        ia, ib, v, n_pairs = cross_pairs(A, B, dev, a.block, same)
        tr = {k: np.concatenate([np.asarray(A[k], float), np.asarray(B[k], float)]) for k in KEYS}
        cand = np.stack([ia, ib + len(A["ax"])], 1)
        w = A["weight"][ia] * B["weight"][ib]
        res[name] = dict(veto_term(tr, cand, w, v, design, inp, cfg, np.random.default_rng(2)), opposite_pairs=n_pairs)
        print(f"{name}: {n_pairs:.3g} opposite-charge pairs, {len(ia):,} candidates, {time.time() - t0:.0f} s", flush=True)
    print(f"\ndesign {design.name}, cone {a.cone:g} mrad; per 15 years (time window +-{DEFAULT_CONFIG['time_window'] * 1e12:.0f} ps)")
    print(f"  {'term':14s}{'candidates':>14s}{'IP < cut':>12s}{'n':>7s}{'after veto':>13s}{'n':>6s}   legs explained")
    for k, r in res.items():
        aft = f"{r['after_15y']:.3g}" if r.get("n_after") else f"< {r.get('after_upper_15y') or 0:.2g}"
        le = " / ".join(f"{x:.4f}" for x in r.get("leg_explained", []))
        print(f"  {k:14s}{r['candidates_15y']:>14.4g}{r['selected_15y']:>12.4g}{r['n_selected']:>7d}{aft:>13s}{r.get('n_after', 0):>6d}   {le}")
    json.dump(dict(design=design.name, dataset=a.dataset, inputs=a.inputs, rock_mu=a.rock_mu, rock_nu=a.rock_nu, cone_mrad=a.cone,
                   rates_per_spill=dict(muons=float(mu["weight"].sum()), rock=float(rk["weight"].sum())), terms=res),
              open(a.out, "w"), indent=1, default=float)
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
