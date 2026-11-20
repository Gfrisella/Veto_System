#!/usr/bin/env python3
"""DIS classes from the event stores (lib/veto/evstore.py): every charged track segment of the stored trees (straight
lines, decays in flight) crossed with the UBT planes and the SBT shell; for the vessel muon DIS also the parent muon's
path from z = 30.6 m up to the vertex (DIS parent store, rows = parent_row).

Event subsets: all, vtx (two good opposite-charge tracks of the event with DOCA < 2 cm, vertex in the vessel, IP < 2.5 m:
interactions.select without the measured-IP cut) and sel (the full background selection: + measured IP < 0.1 m and at
most 2 tracks in the SST1 acceptance), plus groups per target (helium / wall / air / UBT material) for the vessel stores.
MS7 stores: code 3 = the interaction with its V0 undecayed; code 4 = copies with one V0 forced to decay (weight x P).
Track-level histograms of "all" = code-3 tracks + the forced-decay products (and their descendants) of the copies; event-level
"all" = code-3 events (V0 undecayed); "v0dec" = the forced-decay part of the copies alone; vtx / sel use every event
(code 3 and 4, full content), as bin/evstore_eval.py does.

    python3 dis_classes.py --store vessel_mu_R2_sbt_ubt [--frac 1.0]   -> hist_<store>.npz, events_<store>.npz
"""
import argparse
import os
import time

import numpy as np

import common as C
from veto import evstore as es
from veto import interactions as ix
from veto import io, paths, reco
from veto import vertex_selection as vsel

TCOLS = ("event", "track", "parent", "pdg", "x0", "y0", "z0", "x1", "y1", "z1", "px", "py", "pz", "end", "proc", "beam")
TARGETS = {"helium": (0,), "wall": (2,), "air": (1, 8), "ubtmat": tuple(range(10, 18))}


def descendants_of_forced(tr):
    """Mask of tracks that are forced-decay products or descend from one (per-event track ids -> global rows)."""
    ev = tr["event"]
    first = np.r_[0, np.nonzero(np.diff(ev))[0] + 1]
    start = np.repeat(first, np.diff(np.r_[first, len(ev)]))
    par = np.where(tr["parent"] >= 0, start + tr["parent"], -1)
    flag = tr["proc"] == es.PROC_FORCED
    for _ in range(8):
        new = flag | np.where(par >= 0, flag[np.maximum(par, 0)], False)
        if (new == flag).all():
            break
        flag = new
    return flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--frac", type=float, default=1.0, help="random fraction of events (weights / frac)")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    t0 = time.time()
    import torch
    torch.set_num_threads(6)
    d = C.EVSTORE + "/" + a.store
    rng = np.random.default_rng(a.seed)
    evf = np.load(d + "/events.npz")
    ev = {k: evf[k] for k in evf.files}
    keep_ev = rng.random(len(ev["event"])) < a.frac if a.frac < 1 else np.ones(len(ev["event"]), bool)
    ev = {k: v[keep_ev] for k, v in ev.items()}
    ev["weight"] = ev["weight"] / a.frac
    trf = np.load(d + "/tracks.npz")
    tev = trf["event"]
    tmask = np.isin(tev, ev["event"]) if a.frac < 1 else np.ones(len(tev), bool)
    tr = {"event": tev[tmask]}
    del tev
    for k in TCOLS[1:]:
        tr[k] = trf[k][tmask]
    print(f"{a.store}: {len(ev['event']):,} events, {len(tr['event']):,} tracks loaded, {time.time() - t0:.0f} s", flush=True)
    ms7 = a.store.startswith("ms7")
    n = len(ev["event"])
    row_of = np.full(int(ev["event"].max()) + 1, -1); row_of[ev["event"]] = np.arange(n)
    trow = row_of[tr["event"]]
    w_ev, code = ev["weight"], ev["code"]

    # ---- selection (as bin/evstore_eval.py: straight lines from the production point to SST1)
    inter, part = es.to_fastveto(ev, tr)
    ct = ix.charged_tracks(part)
    sel_vtx = ix.select(ct, n, cfg=None)[0]
    cfg = reco.RecoConfig(ip_max=vsel.IP_CUTS["full"])
    sel, pair, vertex = ix.select(ct, n, cfg=cfg)
    n_acc = np.bincount(ct["inter"][ct["acc"]], minlength=n)
    sel = sel[n_acc[sel] <= cfg.max_tracks]
    n_good = np.bincount(ct["inter"][ct["good"]], minlength=n)
    is_vtx = np.zeros(n, bool); is_vtx[sel_vtx] = True
    is_sel = np.zeros(n, bool); is_sel[sel] = True
    print(f"  selection: vtx {is_vtx.sum():,} ({w_ev[is_vtx].sum():.3g}/spill), sel {is_sel.sum():,} "
          f"({w_ev[is_sel].sum():.3g}/spill), {time.time() - t0:.0f} s", flush=True)
    del inter, part, ct

    # ---- charged segments
    q = ix.charge(tr["pdg"])
    ch = np.nonzero(q != 0)[0]
    P = np.stack([tr["x0"][ch], tr["y0"][ch], tr["z0"][ch]], 1)
    Q = np.stack([tr["x1"][ch], tr["y1"][ch], tr["z1"][ch]], 1)
    pm = np.sqrt(tr["px"][ch] ** 2 + tr["py"][ch] ** 2 + tr["pz"][ch] ** 2)
    srow, spdg = trow[ch], tr["pdg"][ch].astype(np.int64)
    is_par = np.zeros(len(ch), bool)
    forced_desc = descendants_of_forced(tr)[ch] if ms7 else np.zeros(len(ch), bool)
    if a.store.startswith("vessel_mu"):            # the parent muon from z = 30.6 m to the vertex
        s, _ = io.open_store(paths.store_dir(os.environ["DIS_PARENTS"]))
        pr = ev["parent_row"]
        o = np.argsort(pr)
        g = {k: np.empty(n) for k in ("x_30", "y_30", "z_30", "px_30", "py_30", "pz_30", "kx", "ky", "kz", "pdg")}
        for k in g:
            g[k][o] = np.asarray(s[k][pr[o]], np.float64)
        A = np.stack([g["x_30"], g["y_30"], g["z_30"]], 1); K = np.stack([g["kx"], g["ky"], g["kz"]], 1)
        V = np.stack([ev["x"], ev["y"], ev["z"]], 1)
        ok = V[:, 2] > A[:, 2]
        kb = ok & np.isfinite(K[:, 2]) & (K[:, 2] < V[:, 2]) & (K[:, 2] > A[:, 2])
        # check: the vertex lies on the parent's line from A (or K)
        st = np.where(kb[:, None], K, A)
        dz = V[:, 2] - st[:, 2]
        pdir = np.stack([g["px_30"], g["py_30"], g["pz_30"]], 1)
        nob = ok & ~kb & (dz > 1)
        pred = A[nob] + pdir[nob] / pdir[nob, 2:3] * dz[nob, None]
        dev = np.linalg.norm(pred[:, :2] - V[nob, :2], axis=1)
        print(f"  parent check: |vertex - parent line| median {np.median(dev) * 100:.2f} cm, 90% {np.quantile(dev, .9) * 100:.2f} cm "
              f"(non-bounced, {nob.sum():,})", flush=True)
        i1 = np.nonzero(ok)[0]; i2 = np.nonzero(kb)[0]
        Pp = np.concatenate([A[i1], K[i2]])
        Qp = np.concatenate([np.where(kb[i1, None], K[i1], V[i1]), V[i2]])
        rp = np.concatenate([i1, i2])
        pp = np.linalg.norm(pdir, axis=1)[rp]
        P = np.concatenate([P, Pp]); Q = np.concatenate([Q, Qp]); pm = np.concatenate([pm, pp])
        srow = np.concatenate([srow, rp]); spdg = np.concatenate([spdg, (-13 * np.sign(g["pdg"][rp])).astype(np.int64)])
        is_par = np.concatenate([is_par, np.ones(len(rp), bool)]); forced_desc = np.concatenate([forced_desc, np.zeros(len(rp), bool)])
    scode = code[srow]

    # ---- crossings
    ub = C.ubt(P, Q)
    pc = C.sbt_pieces(P, Q, rng)
    sg = pc["seg"].astype(int)
    fired = pc["fired"].astype(bool)
    print(f"  {len(P):,} charged segments, {len(sg):,} SBT pieces, {time.time() - t0:.0f} s", flush=True)

    def counts(seg_mask):
        """per event: straw / tile tracks in aperture, fired cells, using segments in seg_mask."""
        out = {}
        for name, (i, x, y, inside) in ub.items():
            k = inside & seg_mask[i]
            out[name] = np.bincount(srow[i[k]], minlength=n)
        k = fired & seg_mask[sg]
        if k.any():
            fc = np.unique(np.stack([srow[sg[k]], pc["cell"][k].astype(np.int64)], 1), axis=0)
            out["fired"] = np.bincount(fc[:, 0], minlength=n)
        else:
            out["fired"] = np.zeros(n, int)
        return out

    H = C.Hists()

    def fill(grp, seg_mask, ev_mask):
        for name, (i, x, y, inside) in ub.items():
            k = seg_mask[i]
            C.fill_ubt(H, grp, name, x[k], y[k], inside[k], pm[i][k], w_ev[srow[i][k]], spdg[i][k])
        k = seg_mask[sg]
        C.fill_sbt(H, grp, pc["face"][k], pc["ring"][k], pc["z"][k], pc["across"][k], fired[k], pm[sg][k],
                   w_ev[srow[sg][k]], edep=pc["edep"][k], pdg=spdg[sg][k])
        if ev_mask is not None:
            c = counts(seg_mask)
            C.fill_events(H, grp, c["straw"][ev_mask], c["tile"][ev_mask], c["fired"][ev_mask], w_ev[ev_mask])
            H.h1(f"{grp}|ev_cat_n", (((c["straw"] > 0) | (c["tile"] > 0)).astype(int) + 2 * (c["fired"] > 0))[ev_mask],
                 np.ones(ev_mask.sum()), "cat")
            H[f"{grp}|n_events"] = np.array([ev_mask.sum(), w_ev[ev_mask].sum()])
        return

    allseg = np.ones(len(P), bool)
    if ms7:
        c3 = code == 3
        fill("all", (scode == 3) | forced_desc, None)
        # event level of "all": code-3 events
        c = counts(scode == 3)
        C.fill_events(H, "all", c["straw"][c3], c["tile"][c3], c["fired"][c3], w_ev[c3])
        H.h1("all|ev_cat_n", (((c["straw"] > 0) | (c["tile"] > 0)).astype(int) + 2 * (c["fired"] > 0))[c3], np.ones(c3.sum()), "cat")
        H["all|n_events"] = np.array([c3.sum(), w_ev[c3].sum()])
        c4 = code == 4
        fill("v0dec", forced_desc, c4)
        fill("vtx", is_vtx[srow], is_vtx)
        fill("sel", is_sel[srow], is_sel)
    else:
        fill("all", allseg, np.ones(n, bool))
        fill("vtx", is_vtx[srow], is_vtx)
        fill("sel", is_sel[srow], is_sel)
        for name, codes in TARGETS.items():
            m = np.isin(code, codes)
            fill(name, np.isin(scode, codes), m)
        if is_par.any():                         # the products alone (no parent muon)
            fill("products", ~is_par, np.ones(n, bool))
            fill("sel_products", ~is_par & is_sel[srow], is_sel)
    H["frac"] = np.array(a.frac)
    H.save(f"{C.OUT}/hist_{a.store}.npz")
    # compact per-event table for the record
    cnt = counts(allseg)
    np.savez(f"{C.OUT}/events_{a.store}.npz", weight=w_ev, code=code, x=ev["x"], y=ev["y"], z=ev["z"], E=ev["E"],
             n_straw=cnt["straw"], n_tile=cnt["tile"], n_fired=cnt["fired"], n_good=n_good, vtx=is_vtx, sel=is_sel, frac=a.frac)
    print(f"  wrote hist_{a.store}.npz, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
