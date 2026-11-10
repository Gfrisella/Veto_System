#!/usr/bin/env python3
"""EM debris (e+-, gamma) from the end of the muon shield on the UBT and fin planes (ISSUES B2), with the host Geant4.

    bin/g4ship.sh python3 bin/em_debris.py --n 20000 --workers 32        # stages 1 + 2, then the maps

Stage 1: g4shield (MS7 last 50 cm, field) on --n muons of data/shield_dis/muons_iron.npz (drawn with replacement in
proportion to their weight: each carries W / n per spill); e / gamma above 1 MeV at z = 29.90 m. Stage 2: g4vessel
(air, window, helium) to the planes em_debris.Z_PLANES. Maps in the binning of results/reco/<--inputs>:
results/em_debris/em_maps.npz (load with run_reco.py --em-debris). Each stage runs in its own worker processes.
"""

import _bootstrap  # noqa: F401

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from veto import em_debris as em
from veto import paths, weights
from veto.config import POT_PER_SPILL
from veto import shield_iron as si

OUT = paths.results_dir() / "em_debris"
_G = None


def _init_shield(seed, z_iron0):
    global _G
    _G = si.init_g4shield(seed=seed * 1000003 + os.getpid(), e_cut=em.E_CUT, z_iron0=z_iron0)


def _init_vessel(seed, _unused=None):
    global _G
    from veto import g4vessel
    _G = g4vessel.init(seed * 1000003 + os.getpid(), e_cut=em.E_CUT, planes=em.Z_PLANES)


def _run(job):
    ev0, off, pdg, p, v = job
    r = {k: np.asarray(x) for k, x in _G.transport(off, pdg, p[:, 0], p[:, 1], p[:, 2], v[:, 0], v[:, 1], v[:, 2]).items()}
    r["ev"] = r["ev"] + ev0
    return r


def pool_run(init, seed, jobs, workers, extra=None):
    res = []
    with mp.get_context("spawn").Pool(workers, initializer=init, initargs=(seed, extra)) as pool:
        for r in pool.imap_unordered(_run, jobs):
            res.append(r)
    return {k: np.concatenate([r[k] for r in res]) for k in res[0]} if res else {}


def chunks(ev, pdg, p, v, n_ev, size):
    """Jobs of `size` events each from flat particle arrays sorted by event."""
    starts = np.searchsorted(ev, np.arange(n_ev + 1))
    for e0 in range(0, n_ev, size):
        e1 = min(n_ev, e0 + size)
        a, b = starts[e0], starts[e1]
        yield (e0, (starts[e0:e1 + 1] - a).astype(np.int64), pdg[a:b].astype(np.int32), p[a:b].astype(np.float64),
               v[a:b].astype(np.float64))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--depth", type=float, default=em.DEPTH_X0, help="iron simulated in Geant4, radiation lengths")
    ap.add_argument("--inputs", default="inputs.npz", help="results/reco file whose map binning is used")
    ap.add_argument("--parents", default=None, help="MS7 parent muons (shield_dis.py prepare output; default data/shield_dis/muons_iron.npz)")
    ap.add_argument("--tag", default="", help="suffix of the output files")
    a = ap.parse_args()
    t0 = time.time()
    parents = a.parents or (paths.PROJECT_DIR / "data" / "shield_dis" / "muons_iron.npz")     # shield_dis.py prepare
    f = np.load(parents)
    w = f["weight"].astype(np.float64) * (POT_PER_SPILL / weights.source_pot(json.loads(str(f["meta"])), "muons_iron"))  # D13
    rng = np.random.default_rng(a.seed)
    idx = rng.choice(len(w), a.n, p=w / w.sum())
    w_mu = np.full(a.n, w.sum() / a.n)                               # per spill, per drawn muon
    pos = np.stack([f["x"][idx], f["y"][idx], f["z"][idx]], 1)
    mom = np.stack([f["px"][idx], f["py"][idx], f["pz"][idx]], 1)
    pdg = (-13 * f["q"][idx].astype(int))
    # only the last --depth X0 of iron in Geant4: carry the muons there first (field, mean dE/dx)
    z_iron0 = si.Z_SHIELD_END - a.depth * em.X0_FE
    p_mag = np.linalg.norm(mom, axis=1)
    u = mom / p_mag[:, None]
    z_start = z_iron0 - 1e-3
    pos, u, p_new = si.curved_points(pos, u, p_mag, -np.sign(pdg).astype(float), (z_start - pos[:, 2]) / u[:, 2])
    alive = p_new > 0.2
    mom = u * np.maximum(p_new, 0.0)[:, None]
    pos[:, 2] = z_start
    print(f"  Geant4 slab {z_iron0:.4f} - {si.Z_SHIELD_END} m ({a.depth:g} X0 = {a.depth * em.X0_FE * 100:.1f} cm); "
          f"{(~alive).sum()} muons below 0.2 GeV before it dropped", flush=True)
    pdg = np.where(alive, pdg, 0)                                      # pdg 0: not shot (g4shield skips it)
    print(f"{a.n:,} muons drawn of {len(w):,} ({w.sum():.4g} per spill cross the iron), {time.time() - t0:.0f} s", flush=True)
    s1 = pool_run(_init_shield, a.seed, list(chunks(np.arange(a.n), pdg, mom, pos, a.n, a.chunk)), a.workers, z_iron0)
    em_ = np.isin(s1["pdg"], [11, -11, 22])
    print(f"stage 1: {len(s1['pdg']):,} particles at z = {si.Z_PLANE} m, {em_.sum():,} e / gamma "
          f"({(w_mu[s1['ev'][em_]]).sum():.4g} per spill), {time.time() - t0:.0f} s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)                      # stage 1 e / gamma at the exit plane, linked to the parent rows
    np.savez(OUT / f"em_exit_n{a.n}_d{a.depth:g}{a.tag}.npz", w_mu=w_mu, parent_row=idx, parents=str(parents),
             **{k: np.asarray(v)[em_] for k, v in s1.items()})
    ev = s1["ev"][em_]
    o = np.argsort(ev, kind="stable")
    p1 = np.stack([s1["px"], s1["py"], s1["pz"]], 1)[em_][o]
    v1 = np.stack([s1["x"], s1["y"], np.full(len(s1["x"]), si.Z_PLANE)], 1)[em_][o]
    s2 = pool_run(_init_vessel, a.seed + 1, list(chunks(ev[o], s1["pdg"][em_][o], p1, v1, a.n, a.chunk)), a.workers)
    print(f"stage 2: {len(s2.get('pdg', [])):,} plane crossings, {time.time() - t0:.0f} s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)                      # raw crossings: maps can be rebinned for new inputs
    np.savez(OUT / f"em_raw_n{a.n}_d{a.depth:g}{a.tag}.npz", w_mu=w_mu, **{k: v for k, v in s2.items()})
    inp = np.load(paths.results_dir() / "reco" / a.inputs)
    maps = em.make_maps(s2, w_mu, {k: inp[k] for k in ("u_bins", "y_bins", "z_slices", "ubt_y_bins")})
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / em.EM_FILE.replace(".npz", f"{a.tag}.npz"), **maps, n_muons=a.n, muons_per_spill=float(w.sum()), e_cut=em.E_CUT, depth_x0=a.depth)
    for k in ("e", "g"):
        print(f"  {'electrons' if k == 'e' else 'photons':9s}: UBT {maps[f'em_ubt_{k}'].sum():.4g} / spill; fins at z = "
              + ", ".join(f"{z:.2f}: {maps[f'em_fin_{k}'][:, i].sum():.3g}" for i, z in enumerate(inp['z_slices'][:4])))
    print(f"wrote {OUT / em.EM_FILE.replace('.npz', a.tag + '.npz')}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
