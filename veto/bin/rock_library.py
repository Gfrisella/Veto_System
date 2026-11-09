#!/usr/bin/env python3
"""Cavern-rock interaction library (docs/full_sim_inventory.md, ISSUES D7): what comes out of the concrete surface after a
muon-nuclear interaction under it, for muon energy x grazing angle, the vertex depth drawn per event. Stored: only the
particles crossing the surface into the air (kinetic >= EXIT_MIN), in the frame of cpp/g4rock (surface y = 0, rock y < 0,
vertex at (0, -depth, 0), muon direction (0, -sin a, cos a)); every generated event is kept in the event table, so empty
ones count.

Two parts, split in energy transfer nu at the crossover nu_x(E) of the vessel library (fastveto.dis "crossover"):
  comp 0  Geant4 muonNuclear (Kokoulin), forced in g4rock, events with nu < nu_x(E) kept
  comp 1  Pythia 6 (gamma / mu- on p or n, Z/A of concrete), events with nu >= nu_x(E) from data/interactions/dis/mu_library.npz,
          rotated onto the muon direction (random azimuth), injected at the vertex and transported by g4rock (Geant4 FTFP_BERT)
Cross section of each part per metre of rock (sig_g4, sig_p6 [1/m]) at the library energies: the application picks the part
with probability proportional to them and an event of that part in its (energy, angle, depth) cell.

    python3 bin/rock_library.py prepare                    # Pythia subset at the library energies -> <out>/p6_subset.npz
    bin/g4.sh python3 bin/rock_library.py task --index K   # one (energy, angle, part) -> <out>/parts/part_K.npz
    python3 bin/rock_library.py combine                    # -> <out>/rock_library.npz
    python3 bin/rock_library.py list                       # the task list
"""
import _bootstrap  # noqa: F401
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from veto import paths
from fastveto.dis import rotate_to   # the same frame rotation as the muon-DIS library
SHARED = os.environ.get("VETO_SHARED", "/disk/lhcb_data/sqasim")   # Shah Rukh's shared data (read only)

OUT = Path(os.environ.get("ROCK_LIB", f"{SHARED}/rock_dis/library"))    # /disk/lhcb_data = /disk/gfs_lhcb (the
MU_LIB = Path(os.environ.get("ROCK_MU_LIB", paths.mu_library()))   # only its Pythia 6 part is used (same in every copy)
#                                                            Geant4 container mounts only the first name)
IE_LIB = list(range(10, 50, 3))                    # mu_library grid indices: 2.05, 3.13, ... 327.6 GeV (x 1.52)
ALPHA = np.geomspace(0.003, 0.6, 8)                # rad
N_EVENTS = {0: 20000, 1: 5000}                     # per (energy, angle) and part
EXIT_MIN = 1e-3                                    # GeV kinetic: charged exits stored above this
EXIT_MIN_NEUTRAL = 3e-3                            # GeV kinetic: photons and neutrons (most of the volume, rarely a hit)
Z_OVER_A = 0.50274                                 # G4_CONCRETE
RHO = 2.3e3                                        # kg / m^3
N_NUCLEON = RHO * 1e3 * 6.02214076e23              # nucleons per m^3 (mass in g / 1 g per mol of nucleons)
UB = 1e-34                                         # m^2 per microbarn


MASS = {11: 0.000511, 13: 0.10566, 211: 0.13957, 321: 0.49368, 2212: 0.93827, 2112: 0.93957, 22: 0.0, 130: 0.49761, 310: 0.49761,
        3122: 1.11568, 111: 0.13498, 221: 0.54786}


def mass_of(pdg):
    a = np.abs(pdg)
    return np.array([MASS.get(int(q), 0.5) for q in a]) if len(a) else np.zeros(0)


def depths(n, rng):
    """Vertex depths: half 0-0.3 m, 35 % 0.3-1.5 m, 15 % 1.5-3 m (neutrons still leak from there)."""
    u = rng.random(n)
    return np.where(u < 0.5, rng.uniform(1e-3, 0.3, n), np.where(u < 0.85, rng.uniform(0.3, 1.5, n), rng.uniform(1.5, 3.0, n)))


def tasks():
    return [(ie, ia, comp) for ie in IE_LIB for ia in range(len(ALPHA)) for comp in (0, 1)]


def prepare(a):
    z = np.load(MU_LIB)
    E, nu_x = z["E_grid"], z["nu_x"]
    ie, nu = z["p6_ev_ie"].astype(int), z["p6_ev_nu"]
    keep = np.isin(ie, IE_LIB) & (nu >= nu_x[ie])
    ev = np.nonzero(keep)[0]
    start, n = z["p6_ev_start"][ev], z["p6_ev_n"][ev]
    rows = np.concatenate([np.arange(s, s + k) for s, k in zip(start, n)])
    # fraction of the Pythia events above nu_x, per energy and target: the Pythia part's cross section
    tgt = z["p6_ev_tgt"].astype(int)
    frac = np.zeros((2, len(E)))
    for t in (0, 1):
        for i in IE_LIB:
            m = (ie == i) & (tgt == t)
            frac[t, i] = (m & (nu >= nu_x[ie])).sum() / max(m.sum(), 1)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "p6_subset.npz", E=E, nu_x=nu_x, p6_sigma=z["p6_sigma"], frac=frac, ev_ie=ie[ev].astype(np.int16),
             ev_tgt=tgt[ev].astype(np.int8), ev_nu=nu[ev], ev_n=n.astype(np.int32), ev_start=(np.cumsum(n) - n).astype(np.int64),
             pdg=z["p6_pdg"][rows], p=z["p6_p"][rows], v=z["p6_v"][rows], beam=z["p6_beam"][rows])
    print(f"Pythia subset: {len(ev):,} events above nu_x at {len(IE_LIB)} energies, {len(rows):,} particles -> {OUT / 'p6_subset.npz'}")


def task(a):
    ie, ia, comp = tasks()[a.index]
    out = Path(a.out) if a.out else OUT / "parts" / f"part_{a.index:04d}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not a.force:
        print(f"{out} exists"); return
    sys.path.insert(0, str(paths.cpp_build()))
    import g4rock
    seed = 1_000_003 * (a.index + 1)
    rng = np.random.default_rng(seed)
    g4rock.initialize(seed=seed, material="G4_CONCRETE", e_cut=1e-3)
    sub = np.load(OUT / "p6_subset.npz")
    E, nu_x = float(sub["E"][ie]), float(sub["nu_x"][ie])
    al = float(ALPHA[ia]); u_mu = np.array([0.0, -np.sin(al), np.cos(al)])
    n = a.n or N_EVENTS[comp]
    d = depths(n, rng)
    t0 = time.time()
    extra = None                                   # comp 1: products that start in the air (displaced decays)
    if comp == 0:
        res = []
        for c0 in range(0, n, 2000):
            res.append(g4rock.shoot_depths(pdg=13, E=E, alpha=al, depths=d[c0:c0 + 2000]))
        done = np.concatenate([np.asarray(r["events"]["done"]) for r in res]).astype(bool)
        nu = np.concatenate([np.asarray(r["events"]["nu"]) for r in res])
        ex = {k: np.concatenate([np.asarray(r["exits"][k]) + (c0 if k == "ev" else 0) for c0, r in zip(range(0, n, 2000), res)])
              for k in res[0]["exits"]}
        keep = done & (nu < nu_x)
    else:
        sel = np.nonzero(sub["ev_ie"] == ie)[0]
        if not len(sel) or not all(len(sel[sub["ev_tgt"][sel] == t]) for t in (0, 1)):     # no Pythia events above nu_x here
            np.savez(out, ie=ie, ia=ia, comp=comp, E=E, alpha=al, nu_x=nu_x, n_gen=n, depth=d.astype(np.float32), nu=np.zeros(n, np.float32),
                     keep=np.zeros(n, bool), g4_xs=g4rock.cross_section(13, E), sec=0.0,
                     **{f"x_{k}": np.zeros(0, t) for k, t in (("ev", np.int32), ("pdg", np.int32), ("track", np.int32), ("parent", np.int32),
                        ("ekin", float), ("x", float), ("z", float), ("px", float), ("py", float), ("pz", float), ("t", float))})
            print(f"task {a.index}: E {E:.3g} GeV: no Pythia events above nu_x = {nu_x:.3g} GeV (Pythia part 0) -> {out}")
            return
        # target nucleon by Z / A and the Pythia cross sections per target
        sp, sn = Z_OVER_A * sub["p6_sigma"][0, ie] * sub["frac"][0, ie], (1 - Z_OVER_A) * sub["p6_sigma"][1, ie] * sub["frac"][1, ie]
        tgt = (rng.random(n) * (sp + sn) >= sp).astype(int)
        pick = np.empty(n, int)
        for t in (0, 1):
            pool = sel[sub["ev_tgt"][sel] == t]
            m = tgt == t
            pick[m] = pool[rng.integers(0, len(pool), m.sum())]
        nu = sub["ev_nu"][pick].astype(float)
        starts, cnt = sub["ev_start"][pick], sub["ev_n"][pick]
        rows = np.concatenate([np.arange(s, s + k) for s, k in zip(starts, cnt)])
        owner = np.repeat(np.arange(n), cnt)
        phi = rng.uniform(0, 2 * np.pi, n)[owner]
        U = np.repeat(u_mu[None], len(rows), 0)
        p3 = rotate_to(sub["p"][rows, :3].astype(float), U, phi)
        v3 = rotate_to(sub["v"][rows].astype(float), U, phi) + np.stack([np.zeros(n), -d, np.zeros(n)], 1)[owner]
        pdg = sub["pdg"][rows].astype(int)
        beam = sub["beam"][rows]
        # beam muon first in each event (track 1); products made above the surface go straight to the exits
        order = np.lexsort((~beam, owner))
        p3, v3, pdg, beam, owner = p3[order], v3[order], pdg[order], beam[order], owner[order]
        air = v3[:, 1] > 0
        mass = mass_of(pdg)
        ek = np.sqrt((p3 ** 2).sum(1) + mass ** 2) - mass
        extra = dict(ev=owner[air], pdg=pdg[air], track=np.where(beam[air], 1, 0), parent=np.zeros(air.sum(), int), ekin=ek[air],
                     x=v3[air, 0], z=v3[air, 2], px=p3[air, 0], py=p3[air, 1], pz=p3[air, 2], t=np.zeros(air.sum()))
        # an event whose products all start in the air still needs a primary: a 1 keV photon at the vertex
        empty = np.setdiff1d(np.arange(n), owner[~air])
        if len(empty):
            p3 = np.concatenate([p3, np.tile([0.0, 0.0, 1e-6], (len(empty), 1))]); v3 = np.concatenate([v3, np.stack([np.zeros(len(empty)), -d[empty], np.zeros(len(empty))], 1)])
            pdg = np.concatenate([pdg, np.full(len(empty), 22)]); beam = np.concatenate([beam, np.zeros(len(empty), bool)])
            owner = np.concatenate([owner, empty]); air = np.concatenate([air, np.zeros(len(empty), bool)])
            o2 = np.lexsort((~beam, owner)); p3, v3, pdg, beam, owner, air = p3[o2], v3[o2], pdg[o2], beam[o2], owner[o2], air[o2]
        inj = ~air
        cnt_in = np.bincount(owner[inj], minlength=n)
        off = np.r_[0, np.cumsum(cnt_in)].astype(np.int64)
        res = []
        for c0 in range(0, n, 1000):
            c1 = min(n, c0 + 1000)
            sl = slice(off[c0], off[c1])
            r = g4rock.inject(offsets=off[c0:c1 + 1] - off[c0], pdg=pdg[inj][sl], px=p3[inj][sl, 0], py=p3[inj][sl, 1], pz=p3[inj][sl, 2],
                              x=v3[inj][sl, 0], y=v3[inj][sl, 1], z=v3[inj][sl, 2])
            res.append((c0, r))
        ex = {k: np.concatenate([np.asarray(r["exits"][k]) + (c0 if k == "ev" else 0) for c0, r in res]) for k in res[0][1]["exits"]}
        ex = {k: np.concatenate([ex[k], np.asarray(extra[k]).astype(ex[k].dtype)]) for k in ex}
        keep = np.ones(n, bool)
    neutral = np.isin(np.abs(ex["pdg"]), (22, 2112))
    m = keep[ex["ev"]] & (ex["ekin"] >= np.where(neutral, EXIT_MIN_NEUTRAL, EXIT_MIN))
    ex = {k: v[m] for k, v in ex.items()}
    o = np.argsort(ex["ev"], kind="stable"); ex = {k: v[o] for k, v in ex.items()}
    xs = g4rock.cross_section(13, E)                                 # 1/m, Geant4 muonNuclear in concrete (all nu)
    np.savez(out, ie=ie, ia=ia, comp=comp, E=E, alpha=al, nu_x=nu_x, n_gen=n, depth=d.astype(np.float32), nu=nu.astype(np.float32),
             keep=keep, g4_xs=xs, sec=time.time() - t0,
             **{f"x_{k}": v for k, v in ex.items()})
    print(f"task {a.index}: E {E:.3g} GeV, alpha {1e3 * al:.1f} mrad, part {comp}: {keep.sum()} of {n} kept, {m.sum()} exits, "
          f"{time.time() - t0:.0f} s -> {out}")


def combine(a):
    sub = np.load(OUT / "p6_subset.npz")
    files = sorted((OUT / "parts").glob("part_*.npz"))
    if len(files) != len(tasks()):
        print(f"warning: {len(files)} of {len(tasks())} parts")
    ev = {k: [] for k in ("ie", "ia", "comp", "depth", "nu", "first", "n")}
    xs = {k: [] for k in ("pdg", "beam", "ekin", "x", "z", "px", "py", "pz", "t")}
    keep_frac, g4_xs, sec = {}, {}, 0.0
    nx = 0
    for f in files:
        z = np.load(f)
        ie, ia, comp, keep = int(z["ie"]), int(z["ia"]), int(z["comp"]), z["keep"]
        if comp == 0:
            keep_frac.setdefault(ie, []).append(keep.mean()); g4_xs[ie] = float(z["g4_xs"])
        sec += float(z["sec"])
        idx = np.nonzero(keep)[0]
        remap = -np.ones(len(keep), int); remap[idx] = np.arange(len(idx))
        xev = remap[z["x_ev"]]
        cnt = np.bincount(xev, minlength=len(idx))
        ev["ie"].append(np.full(len(idx), ie, np.int8)); ev["ia"].append(np.full(len(idx), ia, np.int8))
        ev["comp"].append(np.full(len(idx), comp, np.int8)); ev["depth"].append(z["depth"][idx]); ev["nu"].append(z["nu"][idx])
        ev["first"].append((nx + np.cumsum(cnt) - cnt).astype(np.int64)); ev["n"].append(cnt.astype(np.int32))
        nx += len(xev)
        xs["pdg"].append(z["x_pdg"].astype(np.int32)); xs["beam"].append(z["x_track"] == 1)
        for k in ("ekin", "x", "z"):
            xs[k].append(z[f"x_{k}"].astype(np.float16))
        for k in ("px", "py", "pz", "t"):
            xs[k].append(z[f"x_{k}"].astype(np.float32))
    ev = {k: np.concatenate(v) for k, v in ev.items()}; xs = {k: np.concatenate(v) for k, v in xs.items()}
    have = [i for i in IE_LIB if i in g4_xs]                     # all of them once every part is in
    pos = {i: k for k, i in enumerate(have)}
    ev["ie"] = np.array([pos.get(int(i), -1) for i in ev["ie"]], np.int8)      # index into E below
    keep_ev = ev["ie"] >= 0
    if not keep_ev.all():                                         # partial: drop events of energies without a Geant4 part
        rows_keep = np.repeat(keep_ev, ev["n"])
        xs = {k: v[rows_keep] for k, v in xs.items()}
        ev = {k: v[keep_ev] for k, v in ev.items()}
        ev["first"] = (np.cumsum(ev["n"]) - ev["n"]).astype(np.int64)
    E = np.array([float(sub["E"][i]) for i in have])
    sig_g4 = np.array([g4_xs[i] * np.mean(keep_frac[i]) for i in have])                         # 1/m, nu < nu_x
    sig_p6 = np.array([(Z_OVER_A * sub["p6_sigma"][0, i] * sub["frac"][0, i] + (1 - Z_OVER_A) * sub["p6_sigma"][1, i] * sub["frac"][1, i])
                       * UB * N_NUCLEON for i in have])                                           # 1/m, nu >= nu_x
    meta = dict(material="G4_CONCRETE", rho=RHO, z_over_a=Z_OVER_A, physics="Geant4 11.2.2 FTFP_BERT; Pythia 6.4.28 above nu_x (fastveto crossover)",
                frame="surface y = 0, rock y < 0, vertex (0, -depth, 0), muon (0, -sin a, cos a); exits: x, z on the surface (m), p (GeV), t (ns)",
                exit_min=EXIT_MIN, exit_min_photon_neutron=EXIT_MIN_NEUTRAL, n_events=N_EVENTS, cpu_hours=sec / 3600, created=time.strftime("%Y-%m-%d %H:%M"))
    np.savez_compressed(OUT / "rock_library.npz", E=E, ie_lib=np.array(have), alpha=ALPHA, nu_x=np.array([float(sub["nu_x"][i]) for i in have]),
                        sig_g4=sig_g4, sig_p6=sig_p6, meta=json.dumps(meta), **{f"ev_{k}": v for k, v in ev.items()}, **{f"x_{k}": v for k, v in xs.items()})
    print(f"{len(ev['ie']):,} events, {len(xs['pdg']):,} exits, {sec / 3600:.1f} CPU h -> {OUT / 'rock_library.npz'} "
          f"({(OUT / 'rock_library.npz').stat().st_size / 1e6:.0f} MB)")
    for i, e in enumerate(E):
        print(f"  E {e:7.2f} GeV: sigma Geant4 part {sig_g4[i]:.3g} / m, Pythia part {sig_p6[i]:.3g} / m ({100 * sig_p6[i] / (sig_g4[i] + sig_p6[i]):.1f} %)")
    ch = np.isin(np.abs(xs["pdg"]), (211, 321, 2212)) & (xs["ekin"].astype(float) > 1.0)
    for c in (0, 1):
        evc = ev["comp"] == c
        print(f"  part {c}: {evc.sum():,} events, charged hadrons > 1 GeV out per event {np.bincount(np.repeat(np.arange(len(ev['n'])), ev['n'])[ch], minlength=len(ev['n']))[evc].mean():.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare"); sub.add_parser("combine"); sub.add_parser("list")
    t = sub.add_parser("task"); t.add_argument("--index", type=int, required=True); t.add_argument("--force", action="store_true")
    t.add_argument("--out", default=None, help="output file (default <OUT>/parts/part_K.npz)")
    t.add_argument("--n", type=int, default=0, help="events (default N_EVENTS of the part; for tests)")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a)
    elif a.cmd == "task":
        task(a)
    elif a.cmd == "combine":
        combine(a)
    else:
        for k, (ie, ia, comp) in enumerate(tasks()):
            print(k, ie, f"{1e3 * ALPHA[ia]:.1f} mrad", comp)
        print(len(tasks()), "tasks")


if __name__ == "__main__":
    main()
