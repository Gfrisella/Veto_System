#!/usr/bin/env python3
"""Build event stores (lib/veto/evstore.py) from the existing productions: events + track trees down to SST1.

    python3 bin/evstore_build.py shield "<shield_dis npz glob>" --out <dir>        # MS7 muon or neutrino DIS (shield_dis)
    python3 bin/evstore_build.py apply  <apply npz> --out <dir> [--pot-spill 4e13]  # DIS in helium / air / walls / planes
    python3 bin/evstore_build.py rock   <rock_apply --store-out npz> --out <dir>    # muon DIS in the cavern rock

shield: code 3 (the interaction; its V0 stay undecayed: their decays are the copies) and code 4 (forced-decay copies:
the V0 from the exit plane to its simulated decay point, its products, the rest of the event), files averaged.
apply: every particle from the vertex (K_L etc. now decay in flight). rock: every particle from its exit point on the
rock surface, code 5 (-x wall) / 6 (floor). Times from the target (beta = 1 for the parent).
"""
import _bootstrap  # noqa: F401
import argparse
import glob
import json
import time

import numpy as np

from run_interactions import load_npz
from veto import evstore as es
from veto import geometry as G
from veto import shield_iron as si
from veto.config import POT_PER_SPILL

V0 = (130, 310, 3122, -3122, 3322, -3322)


def shield(files, out, seed):
    rng = np.random.default_rng(seed)
    EV = {k: [] for k in es.EVENT_COLS}; TR = {k: [] for k in es.TRACK_COLS}
    off = 0
    nf = len(files)
    for f in files:
        t0 = time.time()
        inter, part = load_npz(f)
        n = len(inter["weight"])
        code = inter["code"].astype(np.int64)
        copy_of = np.where(code == 4, inter["parent"] + off, -1) if "parent" in inter else np.full(n, -1)
        v0 = inter["v0"] if "v0" in inter else np.zeros(n, np.int64)
        tv = (inter["z"] - G.Z_TARGET) / es.C_LIGHT
        ev = dict(event=np.arange(n) + off, weight=inter["weight"] / nf, code=code, parent_row=inter["parent_row"],
                  x=inter["x"], y=inter["y"], z=inter["z"], t=tv, E=inter["E"], copy_of=copy_of, v0=v0)
        pi = part["int"]
        at_plane = np.abs(part["v"][:, 2] - si.Z_PLANE) < 0.05
        dau = ~at_plane                                                     # forced-decay products (code 4 only)
        # generation 0: the particles at the exit plane, plus for each copy its V0 from the plane to the decay point
        g_ev, g_pdg, g_x, g_p, g_t, g_beam = [], [], [], [], [], []
        pl = np.nonzero(at_plane)[0]
        g_ev.append(pi[pl] + off); g_pdg.append(part["pdg"][pl]); g_x.append(part["v"][pl]); g_p.append(part["p"][pl, :3])
        g_t.append(tv[pi[pl]] + (si.Z_PLANE - inter["z"][pi[pl]]) / es.C_LIGHT); g_beam.append(part["beam"][pl])
        di = np.nonzero(dau)[0]
        cps, first = np.unique(pi[di], return_index=True)                  # copies with products
        dec = part["v"][di[first]]
        psum = np.zeros((len(cps), 3)); own = np.searchsorted(cps, pi[di]); np.add.at(psum, own, part["p"][di, :3])
        u = psum / np.linalg.norm(psum, axis=1, keepdims=True)
        x_pl = dec - u * ((dec[:, 2] - si.Z_PLANE) / u[:, 2])[:, None]
        n_pl = len(pl)
        g_ev.append(cps + off); g_pdg.append(v0[cps]); g_x.append(x_pl); g_p.append(psum)
        g_t.append(tv[cps] + (si.Z_PLANE - inter["z"][cps]) / es.C_LIGHT); g_beam.append(np.zeros(len(cps), bool))
        gev, gpdg, gx, gp, gt, gb = (np.concatenate(v) for v in (g_ev, g_pdg, g_x, g_p, g_t, g_beam))
        forced = dict(rows=n_pl + np.arange(len(cps)), dec=dec, prod_owner=own, prod_pdg=part["pdg"][di], prod_p=part["p"][di, :3])
        # code 3 events: their V0 do not decay here (the copies carry them): give them an infinite lifetime by marking
        stable = np.isin(gpdg, V0) & (code[gev - off] == 3)
        tr = propagate_with_stable(gev, gpdg, gx, gp, gt, rng, forced, gb, stable)
        for k in es.EVENT_COLS:
            EV[k].append(np.asarray(ev[k]))
        for k in es.TRACK_COLS:
            TR[k].append(tr[k])
        print(f"{f.split('/')[-1]}: {n:,} events, {len(tr['event']):,} tracks, {time.time() - t0:.0f} s", flush=True)
        off += n
    ev = {k: np.concatenate(v) for k, v in EV.items()}; tr = {k: np.concatenate(v) for k, v in TR.items()}
    meta = dict(source="MS7 last 50 cm (bin/shield_dis.py)", files=files, reference="shield exit plane z = %.2f m" % si.Z_PLANE,
                pot_per_spill=POT_PER_SPILL, codes={"3": "interaction (V0 undecayed)", "4": "forced V0 decay copy"},
                downstream="evstore.propagate: straight lines, decays in flight", weights="per spill, averaged over the files")
    return es.write(out, meta, ev, tr)


def propagate_with_stable(ev, pdg, x, p, t, rng, forced, beam, stable):
    """propagate() with some generation-0 particles kept undecayed (their decays are elsewhere)."""
    saved = dict(es.CTAU)
    # mark by a pdg offset: the stable ones get a pseudo-code with no channel, restored afterwards
    code = pdg.copy(); code[stable] = code[stable] + np.sign(code[stable]) * 10_000_000
    try:
        tr = es.propagate(ev, code, x, p, t, rng, forced=forced, beam=beam)
    finally:
        es.CTAU.clear(); es.CTAU.update(saved)
    big = np.abs(tr["pdg"]) >= 10_000_000
    tr["pdg"][big] = tr["pdg"][big] - np.sign(tr["pdg"][big]) * 10_000_000
    return tr


def apply(path, out, seed, pot):
    rng = np.random.default_rng(seed)
    inter, part = load_npz(path)
    n = len(inter["weight"])
    w = inter["weight"] * (POT_PER_SPILL / pot)
    tv = (inter["z"] - G.Z_TARGET) / es.C_LIGHT
    ev = dict(event=np.arange(n), weight=w, code=inter["code"].astype(np.int64),
              parent_row=inter["muon"] if "muon" in inter else np.full(n, -1), x=inter["x"], y=inter["y"], z=inter["z"],
              t=tv, E=inter["E"], copy_of=np.full(n, -1), v0=np.zeros(n, np.int64))
    tr = es.propagate(part["int"], part["pdg"], part["v"], part["p"][:, :3], tv[part["int"]], rng, beam=part["beam"])
    meta = dict(source=f"vessel interactions ({path})", reference="interaction vertex", pot_per_spill=POT_PER_SPILL,
                downstream="evstore.propagate: straight lines, decays in flight (K_L etc. decay naturally)",
                extra={k: inter[k][:0].dtype.str for k in ("flavour", "component") if k in inter})
    if "flavour" in inter:
        np.save(str(out) + "_flavour.npy", inter["flavour"])
    return es.write(out, meta, ev, tr)


def rock(path, out, seed):
    rng = np.random.default_rng(seed)
    z = np.load(path)
    ns = int(z["n_seeds"])
    draw = z["draw"]
    ids, first, inv = np.unique(draw, return_index=True, return_inverse=True)
    n = len(ids)
    w = z["w"][first].astype(float) / ns
    tv = z["t_mu"][first].astype(float) if "t_mu" in z.files else z["t"][first].astype(float)
    ev = dict(event=np.arange(n), weight=w, code=5 + z["surf"][first].astype(np.int64),
              parent_row=z["mu_row"][first] if "mu_row" in z.files else np.full(n, -1),
              x=z["vx"][first], y=z["vy"][first], z=z["vz"][first], t=tv, E=z["E_mu"][first] if "E_mu" in z.files else np.zeros(n),
              copy_of=np.full(n, -1), v0=np.zeros(n, np.int64))
    pos = np.stack([z["x"], z["y"], z["z"]], 1).astype(float); mom = np.stack([z["px"], z["py"], z["pz"]], 1).astype(float)
    beam = np.abs(z["pdg"]) == 13 if "beam" not in z.files else z["beam"]
    tr = es.propagate(inv, z["pdg"].astype(np.int64), pos, mom, z["t"].astype(float), rng, beam=beam)
    meta = dict(source=f"muon DIS in the cavern rock ({path})", reference="rock surface", pot_per_spill=POT_PER_SPILL,
                codes={"5": "-x wall", "6": "floor"}, n_seeds=ns, parent_table="seed files (seed, mu_row)",
                downstream="evstore.propagate: straight lines, decays in flight")
    if "seed" in z.files:
        np.save(str(out) + "_seed.npy", z["seed"][first])
    return es.write(out, meta, ev, tr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=("shield", "apply", "rock")); ap.add_argument("inp"); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=21); ap.add_argument("--pot-spill", type=float, default=POT_PER_SPILL)
    a = ap.parse_args()
    t0 = time.time()
    if a.kind == "shield":
        meta = shield(sorted(glob.glob(a.inp)), a.out, a.seed)
    elif a.kind == "apply":
        meta = apply(a.inp, a.out, a.seed, a.pot_spill)
    else:
        meta = rock(a.inp, a.out, a.seed)
    print(f"-> {a.out}: {meta['n_events']:,} events, {meta['n_tracks']:,} tracks ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
