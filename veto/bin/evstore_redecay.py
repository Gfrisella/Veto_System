#!/usr/bin/env python3
"""More forced decays of the stored V0 (MS7 store of bin/evstore_build.py shield): every distinct V0 of the forced-decay
copies (code 4) is decayed again --n times in the vessel z range (truncated exponential along its line, as shield_dis
force_v0), with the rest of its event, weight = (weight of its original copies) / n. More decay kinematics per K_L: the
pair's pointing (IP) depends on the decay (the neutrino of K_L -> pi l nu), so the IP tail fills n / copies times faster
with no new Geant4. Pre-selection: V0 with p > --p-min whose line at SST1 is within the enlarged SST1 window.

    python3 bin/evstore_redecay.py <ms7 store> --n 100 --out <store>_redecay100
"""
import _bootstrap  # noqa: F401
import argparse
import time

import numpy as np

from fastveto import decay as dc
from veto import evstore as es
from veto import geometry as G

CTAU = {130: 15.34, 310: 0.026844, 3122: 0.0789, -3122: 0.0789, 3322: 0.0871, -3322: 0.0871}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store"); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--p-min", type=float, default=1.5); ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=20000); ap.add_argument("--seed", type=int, default=31)
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    meta, ev, tr = es.read(a.store)
    c4 = ev["code"] == 4
    evrow = {int(e): i for i, e in enumerate(ev["event"])}
    vroot = np.nonzero((tr["parent"] < 0) & (tr["end"] == es.END_FORCED))[0]
    vroot = vroot[c4[[evrow[int(e)] for e in tr["event"][vroot]]]]
    copy_of = ev["copy_of"][[evrow[int(e)] for e in tr["event"][vroot]]]
    key = np.stack([copy_of, np.round(tr["px"][vroot], 4) * 1e4, np.round(tr["py"][vroot], 4) * 1e4, np.round(tr["pz"][vroot], 4) * 1e4], 1)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    w_copy = ev["weight"][[evrow[int(e)] for e in tr["event"][vroot]]]
    w_v0 = np.bincount(inv.ravel(), w_copy)                               # summed weight of the copies of each V0
    v = vroot[first]
    p = np.stack([tr["px"][v], tr["py"][v], tr["pz"][v]], 1).astype(float)
    x0 = np.stack([tr["x0"][v], tr["y0"][v], tr["z0"][v]], 1).astype(float)
    pdg = tr["pdg"][v].astype(np.int64)
    pm = np.linalg.norm(p, axis=1); u = p / pm[:, None]
    zs = G.Z_SST1
    xs, ys = x0[:, 0] + u[:, 0] * (zs - x0[:, 2]) / u[:, 2], x0[:, 1] + u[:, 1] * (zs - x0[:, 2]) / u[:, 2]
    pre = (pm > a.p_min) & (u[:, 2] > 0) & (np.abs(xs) < G.SST1_HALF_X + a.margin) & (np.abs(ys) < G.SST1_HALF_Y + a.margin)
    print(f"{len(v):,} distinct V0 in {c4.sum():,} copies ({w_v0.sum():.4g} decays / spill); {pre.sum():,} pre-selected "
          f"({w_v0[pre].sum():.4g} / spill), {time.time() - t0:.0f} s", flush=True)
    sel = np.nonzero(pre)[0]
    # rest of each event: the roots of the first copy other than the V0 and the forced products
    copy_ev = tr["event"][v]
    o = np.argsort(tr["event"], kind="stable")
    tev = tr["event"][o]
    lo, hi = np.searchsorted(tev, copy_ev, "left"), np.searchsorted(tev, copy_ev, "right")
    EV = {k: [] for k in es.EVENT_COLS}; TR = {k: [] for k in es.TRACK_COLS}
    next_id = 0
    for b0 in range(0, len(sel), a.batch):
        b = sel[b0:b0 + a.batch]
        n = a.n
        ids = next_id + np.arange(len(b) * n); next_id += len(ids)
        rep = np.repeat(b, n)
        # decay points: truncated exponential along the line between the vessel ends
        lam = pm[rep] / np.array([dc.MASS[abs(int(q))] for q in pdg[rep]]) * np.array([CTAU[int(q)] for q in pdg[rep]])
        d1 = (G.Z_VESSEL_MIN - x0[rep, 2]) / u[rep, 2]; d2 = (G.Z_VESSEL_MAX - x0[rep, 2]) / u[rep, 2]
        e1, e2 = np.exp(-d1 / lam), np.exp(-d2 / lam)
        d = -lam * np.log(e1 - rng.random(len(rep)) * (e1 - e2))
        vdec = x0[rep] + u[rep] * d[:, None]
        p4 = np.concatenate([p[rep], np.sqrt((p[rep] ** 2).sum(1) + np.array([dc.MASS[abs(int(q))] for q in pdg[rep]]) ** 2)[:, None]], 1)
        ev_d, pdg_d, p_d, v_d = dc.force_decay(np.arange(len(rep)), pdg[rep], p4, vdec, rng)
        # generation-0 lists: rest of the event (per re-decay), then the V0 roots (forced: given products)
        r_ev, r_pdg, r_x, r_p, r_t, r_b = [], [], [], [], [], []
        for j, q in enumerate(b):
            rows = o[lo[q]:hi[q]]
            rows = rows[(tr["parent"][rows] < 0) & (tr["end"][rows] != es.END_FORCED)]
            if len(rows):
                k = len(rows)
                r_ev.append(np.repeat(np.arange(j * n, (j + 1) * n), k)); r_pdg.append(np.tile(tr["pdg"][rows], n))
                r_x.append(np.tile(np.stack([tr["x0"][rows], tr["y0"][rows], tr["z0"][rows]], 1), (n, 1)))
                r_p.append(np.tile(np.stack([tr["px"][rows], tr["py"][rows], tr["pz"][rows]], 1), (n, 1)))
                r_t.append(np.tile(tr["t0"][rows], n)); r_b.append(np.tile(tr["beam"][rows], n))
        cat = lambda L, shape: np.concatenate(L) if L else np.zeros(shape)
        rest_ev, rest_pdg = cat(r_ev, (0,)).astype(np.int64), cat(r_pdg, (0,)).astype(np.int64)
        rest_x, rest_p, rest_t, rest_b = cat(r_x, (0, 3)), cat(r_p, (0, 3)), cat(r_t, (0,)), cat(r_b, (0,)).astype(bool)
        nr = len(rest_ev)
        g_ev = np.concatenate([rest_ev, np.arange(len(rep))]); g_pdg = np.concatenate([rest_pdg, pdg[rep]])
        g_x = np.concatenate([rest_x, x0[rep]]); g_p = np.concatenate([rest_p, p[rep]])
        g_t = np.concatenate([rest_t, tr["t0"][v][rep]]); g_b = np.concatenate([rest_b, np.zeros(len(rep), bool)])
        forced = dict(rows=nr + np.arange(len(rep)), dec=vdec, prod_owner=ev_d, prod_pdg=pdg_d, prod_p=p_d[:, :3])
        t = es.propagate(g_ev + ids[0], g_pdg, g_x, g_p, g_t, rng, forced=forced, beam=g_b)
        src_ev = np.array([evrow[int(e)] for e in copy_ev[rep]])
        evb = dict(event=ids, weight=w_v0[rep] / n, code=np.full(len(rep), 4), parent_row=ev["parent_row"][src_ev],
                   x=ev["x"][src_ev], y=ev["y"][src_ev], z=ev["z"][src_ev], t=ev["t"][src_ev], E=ev["E"][src_ev],
                   copy_of=ev["copy_of"][src_ev], v0=pdg[rep])
        for k in es.EVENT_COLS:
            EV[k].append(np.asarray(evb[k]))
        for k in es.TRACK_COLS:
            TR[k].append(t[k])
        print(f"  {b0 + len(b):,} / {len(sel):,} V0, {len(t['event']):,} tracks, {time.time() - t0:.0f} s", flush=True)
    evo = {k: np.concatenate(x) for k, x in EV.items()}; tro = {k: np.concatenate(x) for k, x in TR.items()}
    m = es.write(a.out, dict(meta, source=meta["source"] + f" + {a.n} re-decays per pre-selected V0 (bin/evstore_redecay.py)",
                             parent_store=a.store, redecays=a.n, p_min=a.p_min, margin=a.margin), evo, tro)
    print(f"-> {a.out}: {m['n_events']:,} events, {m['n_tracks']:,} tracks ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
