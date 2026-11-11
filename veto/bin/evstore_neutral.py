#!/usr/bin/env python3
"""Neutral hadrons interacting in the vessel helium, forced (docs/STATUS.md 2026-11-10, "rock neutrons"): the store's
propagation moves n, anti-n and K_L straight (K_L: decays only), so their inelastic interactions in the helium are in no
chain unless Geant4 downstream was run. For every such root (parent -1) with p > --p-min moving downstream whose line
crosses the vessel: --n copies, each with an interaction at a point uniform along its path inside the vessel, weight =
w_event x sigma(E) L x S(l) / n (sigma: unbiased inelastic cross section per metre of helium from the library, L: path
inside, S: K_L survival to that point; sigma L ~ 1 %, so first order is exact enough). The final state is a library event
(bin/neutral_library.py) of the nearest energy, momenta scaled to the root's momentum, rotated onto its direction with a
random azimuth. The rest of the event (its other roots) is kept, so the veto still sees anything charged. Event code 7
("neutral_he"); copy_of = the source event, v0 = the neutral's pdg.

    python3 bin/evstore_neutral.py <store> --library neutral_he.npz --n 50 --out <store>_nhe
"""
import _bootstrap  # noqa: F401
import argparse
import json
import time

import numpy as np

from fastveto.dis import rotate_to
from veto import evstore as es
from veto import geometry as G

CODE = 7
CTAU_KL = 15.34
MASS = {2112: 0.939565, -2112: 0.939565, 130: 0.497611}


class Library:
    def __init__(self, path):
        z = np.load(path)
        self.E, self.species, self.xs = z["E"], [int(s) for s in z["species"]], z["xs"]
        self.ev = {k[3:]: z[k] for k in z.files if k.startswith("ev_")}
        self.x = {k[2:]: z[k] for k in z.files if k.startswith("x_")}
        self.cells = {}
        for js, q in enumerate(self.species):
            for ie in range(len(self.E)):
                self.cells[(q, ie)] = np.nonzero((self.ev["species"] == q) & (self.ev["ie"] == ie))[0]

    def sigma(self, q, E):
        js = self.species.index(int(q))
        return np.interp(np.log(E), np.log(self.E), self.xs[js])

    def ie(self, E):
        return np.abs(np.log(E)[:, None] - np.log(self.E)[None]).argmin(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store"); ap.add_argument("--library", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50); ap.add_argument("--p-min", type=float, default=2.0)
    ap.add_argument("--max-roots", type=int, default=0, help="use only the first N roots (tests)")
    ap.add_argument("--batch", type=int, default=5000); ap.add_argument("--seed", type=int, default=41)
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    lib = Library(a.library)
    meta, ev, tr = es.read(a.store)
    evrow = {int(e): i for i, e in enumerate(ev["event"])}
    pdg = np.asarray(tr["pdg"]).astype(np.int64)
    p = np.stack([tr["px"], tr["py"], tr["pz"]], 1).astype(float)
    pm = np.linalg.norm(p, axis=1)
    c = np.nonzero((tr["parent"] < 0) & np.isin(pdg, lib.species) & (pm > a.p_min) & (p[:, 2] > 0))[0]
    u = p[c] / pm[c, None]
    x0 = np.stack([tr["x0"][c], tr["y0"][c], tr["z0"][c]], 1).astype(float)
    # path inside the vessel (sampled every 10 cm in z)
    l_in = np.zeros(len(c)); l_out = np.zeros(len(c)); has = np.zeros(len(c), bool)
    zs = np.arange(G.Z_VESSEL_MIN + 0.05, G.Z_VESSEL_MAX, 0.1)
    for b0 in range(0, len(c), 200000):
        s = slice(b0, b0 + 200000)
        ell = (zs[None] - x0[s, 2:3]) / u[s, 2:3]
        X = x0[s, 0:1] + u[s, 0:1] * ell; Y = x0[s, 1:2] + u[s, 1:2] * ell
        ins = (ell > 0) & G.inside_vessel(X, Y, np.broadcast_to(zs[None], X.shape))
        has[s] = ins.any(1)
        l_in[s] = np.where(ins, ell, np.inf).min(1); l_out[s] = np.where(ins, ell, -np.inf).max(1)
    c, u, x0, l_in, l_out = c[has], u[has], x0[has], l_in[has], l_out[has]
    if a.max_roots:
        c, u, x0, l_in, l_out = c[:a.max_roots], u[:a.max_roots], x0[:a.max_roots], l_in[:a.max_roots], l_out[:a.max_roots]
    q = pdg[c]
    E = np.sqrt(pm[c] ** 2 + np.array([MASS[int(k)] for k in q]) ** 2)
    sig = np.array([lib.sigma(k, e) for k, e in zip(q, E)])
    src = np.array([evrow[int(e)] for e in tr["event"][c]])
    w_src = np.asarray(ev["weight"], float)[src]
    print(f"{len(c):,} neutral roots (p > {a.p_min} GeV) crossing the vessel: " +
          ", ".join(f"{k}: {(q == k).sum():,}" for k in lib.species) +
          f"; interactions in the helium {(w_src * sig * (l_out - l_in)).sum():.3g} / spill ({time.time() - t0:.0f} s)", flush=True)
    o = np.argsort(tr["event"], kind="stable")
    tev = tr["event"][o]
    src_ev = tr["event"][c]
    lo, hi = np.searchsorted(tev, src_ev, "left"), np.searchsorted(tev, src_ev, "right")
    EV = {k: [] for k in es.EVENT_COLS}; TR = {k: [] for k in es.TRACK_COLS}
    next_id = 0
    for b0 in range(0, len(c), a.batch):
        b = np.arange(b0, min(b0 + a.batch, len(c)))
        n = a.n
        rep = np.repeat(b, n)
        ids = next_id + np.arange(len(rep)); next_id += len(rep)
        ell = l_in[rep] + rng.random(len(rep)) * (l_out[rep] - l_in[rep])
        surv = np.where(q[rep] == 130, np.exp(-ell / (pm[c][rep] / MASS[130] * CTAU_KL)), 1.0)
        w = w_src[rep] * sig[rep] * (l_out[rep] - l_in[rep]) * surv / n
        vtx = x0[rep] + u[rep] * ell[:, None]
        # library events: nearest energy bin, random event of that (species, bin)
        ie = lib.ie(E[rep])
        pick = np.empty(len(rep), np.int64)
        for key in set(zip(q[rep].tolist(), ie.tolist())):
            m = (q[rep] == key[0]) & (ie == key[1])
            cell = lib.cells[key]
            pick[m] = cell[rng.integers(0, len(cell), m.sum())]
        cnt = lib.ev["n"][pick]; first = lib.ev["first"][pick]
        owner = np.repeat(np.arange(len(rep)), cnt)
        rows = np.repeat(first, cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        scale = (pm[c][rep] / np.sqrt(lib.E[ie] ** 2 - np.array([MASS[int(k)] for k in q[rep]]) ** 2))[owner]
        pl = np.stack([lib.x["px"][rows], lib.x["py"][rows], lib.x["pz"][rows]], 1) * scale[:, None]
        prod_p = rotate_to(pl, u[rep][owner], rng.uniform(0, 2 * np.pi, len(rep))[owner])
        # generation 0: the rest of each source event, then the neutral (forced: the library products)
        r_ev, r_pdg, r_x, r_p, r_t, r_b = [], [], [], [], [], []
        for j, k in enumerate(rep):
            rr = o[lo[k]:hi[k]]
            rr = rr[(tr["parent"][rr] < 0) & (rr != c[k])]
            if len(rr):
                r_ev.append(np.full(len(rr), j)); r_pdg.append(tr["pdg"][rr])
                r_x.append(np.stack([tr["x0"][rr], tr["y0"][rr], tr["z0"][rr]], 1)); r_p.append(np.stack([tr["px"][rr], tr["py"][rr], tr["pz"][rr]], 1))
                r_t.append(tr["t0"][rr]); r_b.append(tr["beam"][rr])
        cat = lambda L, shape: np.concatenate(L) if L else np.zeros(shape)
        rest_ev = cat(r_ev, (0,)).astype(np.int64); nr = len(rest_ev)
        g_ev = np.concatenate([rest_ev, np.arange(len(rep))]) + ids[0]
        g_pdg = np.concatenate([cat(r_pdg, (0,)).astype(np.int64), q[rep]])
        g_x = np.concatenate([cat(r_x, (0, 3)), x0[rep]]); g_p = np.concatenate([cat(r_p, (0, 3)), p[c][rep]])
        g_t = np.concatenate([cat(r_t, (0,)), tr["t0"][c][rep]]); g_b = np.concatenate([cat(r_b, (0,)).astype(bool), np.zeros(len(rep), bool)])
        forced = dict(rows=nr + np.arange(len(rep)), dec=vtx, prod_owner=owner, prod_pdg=lib.x["pdg"][rows].astype(np.int64), prod_p=prod_p)
        t = es.propagate(g_ev, g_pdg, g_x, g_p, g_t, rng, forced=forced, beam=g_b)
        evb = dict(event=ids, weight=w, code=np.full(len(rep), CODE), parent_row=np.asarray(ev["parent_row"])[src[rep]],
                   x=vtx[:, 0], y=vtx[:, 1], z=vtx[:, 2], t=np.asarray(ev["t"])[src[rep]], E=E[rep],
                   copy_of=np.asarray(ev["event"])[src[rep]], v0=q[rep])
        for k in es.EVENT_COLS:
            EV[k].append(np.asarray(evb[k]))
        for k in es.TRACK_COLS:
            TR[k].append(t[k])
        print(f"  {b[-1] + 1:,} / {len(c):,} roots, {len(t['event']):,} tracks, {time.time() - t0:.0f} s", flush=True)
    evo = {k: np.concatenate(x) for k, x in EV.items()}; tro = {k: np.concatenate(x) for k, x in TR.items()}
    m = es.write(a.out, dict(meta, source=meta["source"] + f" + forced n / nbar / K_L interactions in the helium ({a.n} per root, "
                             "bin/evstore_neutral.py)", parent_store=a.store, library=a.library, copies=a.n, p_min=a.p_min,
                             codes=dict(meta.get("codes", {}), **{str(CODE): "neutral_he"})), evo, tro)
    print(f"-> {a.out}: {m['n_events']:,} events, {m['n_tracks']:,} tracks ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
