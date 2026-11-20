#!/usr/bin/env python3
"""Hit positions behind bin/sbt_firing.py (Guglielmo, 20 Nov 2026), R2, own hits only:

  mumu   the muons of the mu x mu candidate pairs (pair cuts, IP < 2.5 m and < 10 cm): the point where each leg first enters
         the SBT shell (outer surface of the 21.6 cm cell shell), its face, whether that cell fires (> 30 MeV, 99.3 %).
  dis    the selected events of an event store (survivors of the selection, before tagging; IP < 2.5 m and < 10 cm): the
         SBT cells their own particles fire (sbt_firing.event_fired: products, scattered muon, parent muon) and the UBT
         crossings: every charged track of the event tree and the parent muon (A or bounce point K -> vertex) crossing the
         z of a UBT plane of R2 inside its acceptance (Plane.inside: straws at 30.73 m, |y| < 1.35 m; tiles at 31.13 m).
         No efficiency on the UBT (geometric crossing).

Output: <out>/hits_<label>.npz (+ .json with the SBT / UBT / both / none fractions per selection).

    python3 bin/sbt_firing_hits.py mumu --out $VETO_RESULTS/sbt_firing
    python3 bin/sbt_firing_hits.py dis --label mu_vessel --store <store dir> --muons --out $VETO_RESULTS/sbt_firing
"""
import _bootstrap  # noqa: F401

import argparse
import json
import time
from pathlib import Path

import numpy as np

import sbt_firing as sf
from veto import design as dsg
from veto import evstore as es
from veto import interactions as ix
from veto import io, paths, reco
from veto import sbt
from veto import sbt_material as sm
from veto.config import DEFAULT_CONFIG, SPILLS_15Y


def entry_points(mu, cfg, rng):
    """First entry of each muon path into the SBT shell: row, face, x, y, z, fired (bool)."""
    tr = {k: mu[k] for k in ("ax", "ay", "az", "bx", "by", "bz", "kx", "ky", "kz")}
    p, q, row, _ = sbt.path_segments(tr)
    n = len(mu["ax"])
    c = sbt.shell_pieces(p, q)
    ok = c["ring"] >= 0
    s = c["seg"].astype(int)[ok]
    ta = c["ta"][ok]
    order_key = (s >= n).astype(float) + ta                          # first segment (A -> K / B) before the second (K -> B)
    r = row[s]
    o = np.lexsort((order_key, r))
    first = o[np.r_[True, r[o][1:] != r[o][:-1]]]
    pt = p[s[first]] + ta[first, None] * (q[s[first]] - p[s[first]])
    dep = sbt.DEDX_LS * c["length"][ok][first]
    fired = (dep > cfg["threshold_mev"]) & (rng.random(len(first)) < cfg["efficiency"])
    return dict(row=r[first], face=c["face"][ok][first].astype(int), x=pt[:, 0], y=pt[:, 1], z=pt[:, 2], fired=fired)


def run_mumu(a, cfg):
    import torch
    from rock_combinatorial import muon_tracks
    from veto import muon_types as mt
    torch.set_num_threads(a.threads)
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    mu = muon_tracks(paths.MUON_DATASET, a.mu_sample)
    mu["q"] = mt.charge_from_pdg_column(mu["pdg"]).astype(float)
    ent = entry_points(mu, cfg, rng)
    n = len(mu["ax"])
    pos = np.full(n, -1); pos[ent["row"]] = np.arange(len(ent["row"]))
    rcfg = reco.RecoConfig(seed=1, multiple_scattering=True)
    ia, ib, vtx, ip = sf.pair_candidates(mu, rcfg, a.block)
    fac = 2 * DEFAULT_CONFIG["time_window"] / rcfg.spill_s * SPILLS_15Y
    w = mu["weight"][ia] * mu["weight"][ib] * fac
    keep = ip < 2.5
    ia, ib, vtx, ip, w = ia[keep], ib[keep], vtx[keep], ip[keep], w[keep]
    legs = {}
    for leg, rows in (("a", ia), ("b", ib)):
        e = pos[rows]
        has = e >= 0
        g = lambda k, fill: np.where(has, ent[k][np.maximum(e, 0)], fill)
        legs.update({f"{leg}_row": rows, f"{leg}_q": mu["q"][rows], f"{leg}_has": has, f"{leg}_face": g("face", -1),
                     f"{leg}_x": g("x", np.nan), f"{leg}_y": g("y", np.nan), f"{leg}_z": g("z", np.nan),
                     f"{leg}_fired": g("fired", False)})
    # a subsample of muon paths for the top view (A -> K -> B)
    sub = np.unique(np.concatenate([ia, ib]))
    out = dict(w15=w, ip=ip, vtx=vtx, path_rows=sub, **legs,
               **{f"path_{k}": mu[k][sub] for k in ("ax", "ay", "az", "kx", "ky", "kz", "bx", "by", "bz", "q")})
    res = {}
    for lev, cut in (("ip25", 2.5), ("ip01", 0.10)):
        m = ip < cut
        W = w[m].sum()
        f = {}
        for leg in ("a", "b"):
            for face, name in enumerate(sbt.FACES):
                f[f"{leg}_{name}"] = float(w[m & (legs[f"{leg}_face"] == face)].sum() / W)
            f[f"{leg}_none"] = float(w[m & ~legs[f"{leg}_has"]].sum() / W)
            f[f"{leg}_fired"] = float(w[m & legs[f"{leg}_fired"]].sum() / W)
        res[lev] = dict(per_15y=float(W), n_mc=int(m.sum()), entry_face_fractions=f,
                        note="leg a = mu+ (positive charge), leg b = mu-")
        print(f"  {lev}: {W:.4g} / 15 y ({m.sum():,} MC); entry faces {json.dumps({k: round(v, 3) for k, v in f.items()})}", flush=True)
    print(f"mumu hits done ({time.time() - t0:.0f} s)")
    return res, out


def ubt_crossings(design, v0, v1):
    """Crossings of segments v0 -> v1 (n, 3) with the UBT planes of the design: (segment, plane index, x, y)."""
    out = ([], [], [], [])
    zs = sorted({round(p.z, 4) for p in design.planes if p.kind in ("ubt", "tiles")})
    for zp in zs:
        lo, hi = np.minimum(v0[:, 2], v1[:, 2]), np.maximum(v0[:, 2], v1[:, 2])
        i = np.nonzero((lo < zp) & (hi > zp))[0]
        if not len(i):
            continue
        t = (zp - v0[i, 2]) / (v1[i, 2] - v0[i, 2])
        x = v0[i, 0] + t * (v1[i, 0] - v0[i, 0]); y = v0[i, 1] + t * (v1[i, 1] - v0[i, 1])
        hit = np.zeros(len(i), bool); pl = np.full(len(i), -1)
        for p in design.planes:
            if p.kind in ("ubt", "tiles") and abs(p.z - zp) < 1e-3:
                m = p.inside(p.to_local(x), y) & ~hit
                hit |= m; pl[m] = p.index
        out[0].append(i[hit]); out[1].append(pl[hit]); out[2].append(x[hit]); out[3].append(y[hit])
    return tuple(np.concatenate(o) if o else np.zeros(0) for o in out)


def run_dis(a, cfg, design):
    from dis_combinatorial import drop_v0_products_of_originals
    t0 = time.time()
    sm.register()
    meta, ev, tr = es.read(paths.evstore(a.store))
    ms7 = "ms7" in Path(a.store).name
    if ms7 and "muon" in Path(a.store).name:
        tr["beam"] = sf.ms7_beam_flag(ev, tr)
    if ms7:
        tr = drop_v0_products_of_originals(ev, tr)
    muons = io.open_store(paths.store_dir(paths.MUON_DATASET))[0] if a.muons else None
    eo = np.argsort(ev["event"], kind="stable")
    ev_index = eo[np.searchsorted(np.asarray(ev["event"])[eo], tr["event"])]
    w = np.asarray(ev["weight"], float); code = np.asarray(ev["code"])
    vtx_all = np.stack([ev["x"], ev["y"], ev["z"]], 1).astype(float)
    rng = np.random.default_rng(a.seed)
    out, res = {}, {}
    for ip_cut in ("partial", "full"):
        sel = sf.selected_events(ev, tr, ip_cut, a.seed)
        pre = f"{ip_cut}_"
        out[pre + "w15"] = w[sel] * SPILLS_15Y; out[pre + "code"] = code[sel]; out[pre + "vtx"] = vtx_all[sel]
        if not len(sel):
            res[ip_cut] = dict(per_15y=0.0, n_mc=0); continue
        e, cell, edep, comp, dz, dt = sf.event_fired(sel, ev, tr, ev_index, muons, cfg, rng)
        out.update({pre + "sbt_ev": e, pre + "sbt_cell": cell, pre + "sbt_edep": edep, pre + "sbt_comp": comp})
        # UBT: charged tracks of the selected events + the parent muon
        pos = np.full(len(ev["event"]), -1); pos[sel] = np.arange(len(sel))
        rows = np.nonzero(pos[ev_index] >= 0)[0]
        ch = ix.charge(np.asarray(tr["pdg"])[rows]) != 0
        rows = rows[ch]
        v0 = np.stack([tr["x0"][rows], tr["y0"][rows], tr["z0"][rows]], 1).astype(float)
        v1 = np.stack([tr["x1"][rows], tr["y1"][rows], tr["z1"][rows]], 1).astype(float)
        u_ev = pos[ev_index[rows]]
        u_comp = np.where(np.asarray(tr["beam"], bool)[rows], 1, 0)
        if muons is not None:
            mrow = np.asarray(ev["parent_row"], np.int64)[sel]
            A = np.stack([np.asarray(muons[f"{c}_30"][mrow], float) for c in "xyz"], 1)
            K = np.stack([np.asarray(muons[f"k{c}"][mrow], float) for c in "xyz"], 1)
            V = vtx_all[sel]
            bounced = np.isfinite(K[:, 2]) & (K[:, 2] < V[:, 2])
            # A -> K and K -> V for bounced parents, A -> V otherwise
            v0 = np.concatenate([v0, A, K[bounced]]); v1 = np.concatenate([v1, np.where(bounced[:, None], K, V), V[bounced]])
            u_ev = np.concatenate([u_ev, np.arange(len(sel)), np.nonzero(bounced)[0]])
            u_comp = np.concatenate([u_comp, np.full(len(sel) + bounced.sum(), 2)])
        si, pl, x, y = ubt_crossings(design, v0, v1)
        si = si.astype(int)
        out.update({pre + "ubt_ev": u_ev[si], pre + "ubt_plane": pl, pre + "ubt_x": x, pre + "ubt_y": y, pre + "ubt_comp": u_comp[si]})
        sbt_any = np.zeros(len(sel), bool); sbt_any[e] = True
        ubt_any = np.zeros(len(sel), bool); ubt_any[u_ev[si]] = True
        W = out[pre + "w15"]; Wt = W.sum()
        frac = lambda m: float(W[m].sum() / Wt)
        r = dict(per_15y=float(Wt), n_mc=int(len(sel)), sbt_and_ubt=frac(sbt_any & ubt_any), sbt_only=frac(sbt_any & ~ubt_any),
                 ubt_only=frac(~sbt_any & ubt_any), none=frac(~sbt_any & ~ubt_any), n_mc_none=int((~sbt_any & ~ubt_any).sum()))
        for ci, cn in enumerate(sf.COMP):
            sm_ = np.zeros(len(sel), bool); sm_[e[comp == ci]] = True
            um_ = np.zeros(len(sel), bool); um_[u_ev[si][u_comp[si] == ci]] = True
            r[f"sbt_by_{cn}"] = frac(sm_); r[f"ubt_by_{cn}"] = frac(um_)
        r["by_code"] = {str(int(c)): dict(per_15y=float(W[out[pre + "code"] == c].sum()),
                                          none=float(W[(out[pre + "code"] == c) & ~sbt_any & ~ubt_any].sum()))
                        for c in np.unique(out[pre + "code"])}
        res[ip_cut] = r
        print(f"  {a.label} {ip_cut}: {Wt:.4g} / 15 y ({len(sel)} MC): SBT+UBT {r['sbt_and_ubt']:.3f}, SBT only {r['sbt_only']:.3f}, "
              f"UBT only {r['ubt_only']:.3f}, none {r['none']:.3f} ({r['n_mc_none']} MC) ({time.time() - t0:.0f} s)", flush=True)
    return res, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("dis", "mumu"))
    ap.add_argument("--label"); ap.add_argument("--store"); ap.add_argument("--muons", action="store_true")
    ap.add_argument("--design", default="designs/R2_sbt_ubt.json")
    ap.add_argument("--mu-sample", type=int, default=300_000); ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    design = dsg.load(Path(a.design))
    cfg = design.sbt
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.what == "mumu":
        res, arr = run_mumu(a, cfg); label = "mumu"
    else:
        res, arr = run_dis(a, cfg, design); label = a.label
    np.savez_compressed(out / f"hits_{label}.npz", ring_edges=sbt.RING_EDGES, **arr)
    json.dump(dict(label=label, store=a.store, design=a.design, results=res), open(out / f"hits_{label}.json", "w"), indent=1, default=float)
    print(f"wrote {out}/hits_{label}.json / .npz")


if __name__ == "__main__":
    main()
