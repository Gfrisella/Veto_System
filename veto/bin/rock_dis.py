#!/usr/bin/env python3
"""Cavern-wall DIS (docs/full_sim_inventory.md): forced muon-nuclear interactions in the rock (cpp/g4rock, Geant4
FTFP_BERT, G4_CONCRETE as the transport's cavern), and what comes back out of the surface into the cavern.

    bin/g4.sh python3 bin/rock_dis.py scan --workers 12 --n 1000 --out results/rock_dis/scan.npz
        grid of muon energy x grazing angle x depth of the vertex below the surface: per point the chance that
        anything comes out (by kind and energy), and the muon-nuclear cross section per metre of rock. Decides how deep
        the library has to go and how fine its grid must be.
    bin/g4.sh python3 bin/rock_dis.py viz --workers 16 --out results/rock_dis/viz.json
        for the interactive page: per grid point (E x angle x depth) the escape statistics by particle group (chance,
        multiplicity, energy spectrum, angle above the surface, where on the surface), and example events with their
        trajectories in the rock (set_record) at a few depths; the cross section on a fine energy grid.
"""
import _bootstrap  # noqa: F401
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from veto import paths

E_GRID = (3.0, 10.0, 30.0, 100.0)                 # GeV, muon total energy at the vertex
A_GRID = (0.03, 0.1, 0.3)                         # rad, grazing angle (moving away from the surface)
D_GRID = (0.02, 0.1, 0.3, 1.0, 3.0)               # m, depth of the vertex
# what comes out: (name, selection on the exit rows)
KINDS = {
    "any (not the muon)":       lambda x: x["track"] != 1,
    "charged hadron p > 1 GeV": lambda x: np.isin(np.abs(x["pdg"]), (211, 321, 2212)) & (_p(x) > 1.0),
    "charged hadron p > 0.1":   lambda x: np.isin(np.abs(x["pdg"]), (211, 321, 2212)) & (_p(x) > 0.1),
    "neutral hadron > 0.1 GeV": lambda x: np.isin(np.abs(x["pdg"]), (130, 310, 2112, 3122)) & (x["ekin"] > 0.1),
    "e+- > 1 MeV":              lambda x: (np.abs(x["pdg"]) == 11) & (x["ekin"] > 1e-3),
    "gamma > 1 MeV":            lambda x: (x["pdg"] == 22) & (x["ekin"] > 1e-3),
    "muon (not the parent)":    lambda x: (np.abs(x["pdg"]) == 13) & (x["track"] != 1),
}


def _p(x):
    return np.sqrt(x["px"] ** 2 + x["py"] ** 2 + x["pz"] ** 2)


_G = None


def _init(seed0):
    global _G
    sys.path.insert(0, str(paths.cpp_build()))
    import g4rock
    import os
    g4rock.initialize(seed=seed0 + os.getpid(), material="G4_CONCRETE", e_cut=1e-3)
    _G = g4rock


def _point(args):
    E, a, d, n, pdg = args
    t0 = time.time()
    out = _G.shoot(pdg=pdg, E=E, n=n, depth=d, alpha=a)
    ev, x = out["events"], {k: np.asarray(v) for k, v in out["exits"].items()}
    done = np.asarray(ev["done"]).astype(bool)
    frac, mult = {}, {}
    for k, sel in KINDS.items():
        m = sel(x) if len(x["ev"]) else np.zeros(0, bool)
        evs = np.unique(x["ev"][m]) if m.any() else np.zeros(0, int)
        frac[k] = float(np.isin(np.nonzero(done)[0], evs).mean()) if done.any() else 0.0
        mult[k] = float(m.sum() / max(done.sum(), 1))
    mu_out = float(np.isin(np.nonzero(done)[0], x["ev"][x["track"] == 1]).mean()) if done.any() else 0.0
    return dict(E=E, alpha=a, depth=d, n=int(done.sum()), frac=frac, mult=mult, mu_out=mu_out,
                nu_mean=float(np.nanmean(np.asarray(ev["nu"])[done])) if done.any() else 0.0, sec=time.time() - t0)


GROUPS = ("muon", "pion / kaon", "proton / ion", "neutron", "other neutral hadron", "electron / positron", "photon")
VZ_D = (0.02, 0.1, 0.3, 0.6, 1.0, 1.5, 3.0)           # depths with statistics
VZ_SHOW = (0.02, 0.1, 0.3, 1.0)                      # depths with example events
EKB = np.geomspace(1e-3, 100, 26)                    # GeV kinetic
ELB = np.linspace(0, 90, 19)                         # degrees above the surface
MX, MZ = np.linspace(-3, 3, 25), np.linspace(-2, 6, 33)   # m, exit point on the surface relative to the vertex


def group_of(pdg):
    a = np.abs(pdg)
    g = np.full(len(a), 4)
    g[a == 13] = 0; g[(a == 211) | (a == 321)] = 1; g[(a == 2212) | (a > 1_000_000_000)] = 2; g[a == 2112] = 3
    g[a == 11] = 5; g[a == 22] = 6
    other_charged = np.isin(a, (3222, 3112, 3312, 3334, 411, 421, 431, 4122, 15))
    g[other_charged] = 1
    return g


def _viz_point(args):
    E, al, d, n_stat, n_show = args
    t0 = time.time()
    _G.set_record(False)
    o = _G.shoot(pdg=13, E=E, n=n_stat, depth=d, alpha=al)
    done = np.asarray(o["events"]["done"]).astype(bool); nd = max(int(done.sum()), 1)
    x = {k: np.asarray(v) for k, v in o["exits"].items()}
    x = {k: v[done[x["ev"]]] for k, v in x.items()} if len(x["ev"]) else x
    g = group_of(x["pdg"]) if len(x["ev"]) else np.zeros(0, int)
    parent_mu = (x["track"] == 1) if len(x["ev"]) else np.zeros(0, bool)
    p = np.sqrt(x["px"] ** 2 + x["py"] ** 2 + x["pz"] ** 2) if len(x["ev"]) else np.zeros(0)
    elev = np.degrees(np.arcsin(np.clip(x["py"] / np.maximum(p, 1e-12), 0, 1))) if len(p) else p
    st = dict(E=E, alpha=al, depth=d, n=nd, nu=float(np.nanmean(np.asarray(o["events"]["nu"])[done])),
              parent_out=float(len(np.unique(x["ev"][parent_mu])) / nd) if len(p) else 0.0, groups=[])
    for gi in range(len(GROUPS)):
        m = (g == gi) & ~parent_mu
        st["groups"].append(dict(frac=round(float(len(np.unique(x["ev"][m])) / nd), 5), mult=round(float(m.sum() / nd), 5),
                                 spec=np.round(np.histogram(x["ekin"][m], EKB)[0] / nd, 6).tolist(),
                                 elev=np.round(np.histogram(elev[m], ELB)[0] / nd, 6).tolist()))
    had = np.isin(g, (1, 2, 3, 4)) & ~parent_mu; em = np.isin(g, (5, 6))
    st["map_had"] = np.round(np.histogram2d(x["x"][had], x["z"][had], [MX, MZ])[0] / nd, 6).tolist() if len(p) else None
    st["map_em"] = np.round(np.histogram2d(x["x"][em], x["z"][em], [MX, MZ])[0] / nd, 6).tolist() if len(p) else None
    st["any"] = round(float(len(np.unique(x["ev"][~parent_mu])) / nd), 5) if len(p) else 0.0
    events = []
    if d in VZ_SHOW and n_show:
        _G.set_record(True, 0.03, 0.04, 250)
        o = _G.shoot(pdg=13, E=E, n=n_show, depth=d, alpha=al)
        _G.set_record(False)
        tk = {k: np.asarray(v) for k, v in o["tracks"].items()}; pt = {k: np.asarray(v) for k, v in o["points"].items()}
        ex = {k: np.asarray(v) for k, v in o["exits"].items()}
        nu = np.asarray(o["events"]["nu"])
        for e in range(n_show):
            if not o["events"]["done"][e]:
                continue
            ee = np.nonzero(ex["ev"] == e)[0]
            exits = {int(ex["track"][k]): [round(float(ex["px"][k]), 4), round(float(ex["py"][k]), 4), round(float(ex["pz"][k]), 4),
                                           round(float(ex["ekin"][k]), 5)] for k in ee}
            trs = []
            for k in np.nonzero(tk["ev"] == e)[0]:
                f0, nn = int(tk["first"][k]), int(tk["n"][k])
                if nn < 2:
                    continue
                q = np.stack([pt["x"][f0:f0 + nn], pt["y"][f0:f0 + nn], pt["z"][f0:f0 + nn], pt["t"][f0:f0 + nn]], 1)
                trs.append(dict(id=int(tk["track"][k]), par=int(tk["parent"][k]), pdg=int(tk["pdg"][k]), g=int(group_of(tk["pdg"][k:k + 1])[0]),
                                ek=round(float(tk["ekin0"][k]), 4), pts=np.round(q, 3).ravel().tolist(), out=exits.get(int(tk["track"][k]))))
            events.append(dict(nu=round(float(nu[e]), 3), tracks=trs, n_out=len([t for t in exits if t != 1])))
    return dict(stat=st, events=events, sec=time.time() - t0)


def viz(a):
    tasks = [(E, al, d, a.n, a.show) for E in E_GRID for al in A_GRID for d in VZ_D]
    tasks.sort(key=lambda t: -t[0])
    t0 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(a.seed,)) as pool:
        res = pool.map(_viz_point, tasks, chunksize=1)
    sys.path.insert(0, str(paths.cpp_build()))
    import g4rock
    g4rock.initialize(seed=a.seed, material="G4_CONCRETE")
    efine = np.geomspace(1.0, 400.0, 40)
    xs = [g4rock.cross_section(13, float(E)) for E in efine]
    res.sort(key=lambda r: (r["stat"]["E"], r["stat"]["alpha"], r["stat"]["depth"]))
    out = dict(E=E_GRID, A=A_GRID, D=VZ_D, D_SHOW=VZ_SHOW, groups=GROUPS, EKB=EKB.tolist(), ELB=ELB.tolist(), MX=MX.tolist(), MZ=MZ.tolist(),
               xs_E=efine.tolist(), xs=xs, material="G4_CONCRETE", physics="FTFP_BERT (Geant4 11.2.2), muonNuclear forced",
               e_cut=1e-3, em_traj_min=0.03, stats=[r["stat"] for r in res],
               events={f"{r['stat']['E']:g}|{r['stat']['alpha']:g}|{r['stat']['depth']:g}": r["events"] for r in res if r["events"]})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), separators=(",", ":"))
    print(f"{len(tasks)} points, {time.time() - t0:.0f} s wall, cpu {sum(r['sec'] for r in res):.0f} s -> {a.out} "
          f"({Path(a.out).stat().st_size / 1e6:.1f} MB)")


def scan(a):
    tasks = [(E, al, d, a.n, 13) for E in E_GRID for al in A_GRID for d in D_GRID]
    tasks.sort(key=lambda t: -t[0])                    # the expensive (high energy) points first
    t0 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(a.seed,)) as pool:
        res = pool.map(_point, tasks, chunksize=1)
    sys.path.insert(0, str(paths.cpp_build()))
    import g4rock
    g4rock.initialize(seed=a.seed, material="G4_CONCRETE")
    xs = {E: g4rock.cross_section(13, E) for E in E_GRID}            # 1/m
    res.sort(key=lambda r: (r["E"], r["alpha"], r["depth"]))
    print(f"{len(tasks)} points x {a.n} forced interactions, {time.time() - t0:.0f} s wall, {a.workers} workers\n")
    for E in E_GRID:
        print(f"E = {E:g} GeV: muon-nuclear interaction length in concrete {1 / xs[E]:.3g} m (sigma n = {xs[E]:.3g} / m)")
    keys = list(KINDS)
    for k in keys[:5]:
        print(f"\nP(at least one '{k}' comes out) per forced interaction; rows E [GeV] / alpha [mrad], columns depth [m]")
        print(f"{'':16s}" + "".join(f"{d:>9g}" for d in D_GRID))
        for E in E_GRID:
            for al in A_GRID:
                row = [r for r in res if r["E"] == E and r["alpha"] == al]
                print(f"{E:6g} / {1e3 * al:5.0f}   " + "".join(f"{r['frac'][k]:9.3f}" for r in row))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, results=json.dumps(res), xs=json.dumps({str(k): v for k, v in xs.items()}), E=E_GRID, A=A_GRID, D=D_GRID)
    print(f"\n-> {a.out}  (cpu {sum(r['sec'] for r in res):.0f} s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--workers", type=int, default=12); s.add_argument("--n", type=int, default=1000)
    s.add_argument("--seed", type=int, default=1); s.add_argument("--out", default="results/rock_dis/scan.npz")
    v = sub.add_parser("viz")
    v.add_argument("--workers", type=int, default=16); v.add_argument("--n", type=int, default=2000)
    v.add_argument("--show", type=int, default=12); v.add_argument("--seed", type=int, default=7)
    v.add_argument("--out", default="results/rock_dis/viz.json")
    a = ap.parse_args()
    if a.cmd == "scan":
        scan(a)
    elif a.cmd == "viz":
        viz(a)


if __name__ == "__main__":
    main()
