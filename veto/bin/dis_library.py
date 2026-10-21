"""Muon-nucleon interaction library for the fast veto simulation (fastveto.dis): Geant4 + Pythia 6, matched in nu.

    bin/g4.sh python3 bin/dis_library.py [--jobs 300] [--n-p6 20000] [--n-g4 20000] [--out data/dis/mu_library.npz]

Two generators, all events stored with their energy transfer nu and hadronic mass W (nucleon at rest):
  * Pythia 6.4.28 (cpp/tailgen/pythia6_dis, beam gamma/mu-: real + virtual photons, VMD / direct / anomalous / DIS)
    on p+ and n0, W > 2 GeV (CKIN(77) default), grid energies from 2.2 GeV.
  * Geant4 11.2.2 muonNuclear (cpp/g4dis: G4KokoulinMuonNuclearXS + G4MuonVDNuclearModel, FTFP_BERT), one forced
    interaction per event, per material (nuclear break-up included, ions not kept): helium, PVC, Mylar,
    polystyrene, silicon, PVT, air.
Which generator covers which part is chosen when the library is loaded (fastveto.dis.MuonLibrary, split=...):
  "crossover" (default): Geant4 for nu < nu_x(E), Pythia above. nu_x(E) = where Pythia's dsigma/dnu (p + n mean)
      first reaches Geant4's (Kokoulin, helium per nucleon) above nu = 2 GeV (Pythia ramps up from W = 2 GeV and is
      ~10 x below Geant4 at nu ~ 2-3 GeV; at large nu it exceeds Geant4, which has no DIS growth with Q^2).
      No crossing (low E): Geant4 only.
  "w2": Geant4 for W < 2 GeV, Pythia for W > 2 GeV.
Cross section per nucleon of a part = generator's cross section x fraction of its events in the part.
Only mu- is generated: one-photon exchange gives the same hadronic final state for mu+, the beam muon's sign is flipped.
Particles: final state in the frame of the incoming muon (along +z), from the interaction point; K_S, Lambda, Sigma,
Xi, Omega, pi0, eta decayed (Pythia's own decays; Geant4 products decayed by fastveto.decay), products start at the
decay point.
Pythia sometimes loops forever in PYEVNT (11 chunks of 5000 in 1.6 M events): every chunk runs with a time limit;
a killed chunk keeps its complete events and the rest is generated again with another seed.
"""

import _bootstrap  # noqa: F401
import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from veto import paths

from fastveto import decay, kokoulin

P6 = str(paths.PROJECT_DIR / "cpp/tailgen/build/pythia6_dis")
N_A = 6.02214076e23
M_N, M_MU = 0.9389, 0.10566
E_GRID = np.round(np.geomspace(0.5, 500.0, 50), 4)        # Geant4 at all; Pythia from E_P6_MIN
E_P6_MIN = 2.0
NU_X_MIN = 2.0
# Geant4 materials of fastveto.materials (name -> NIST name, density g/cm^3)
G4_MATERIALS = {"helium": ("G4_He", 0.000166322), "pvc": ("G4_POLYVINYL_CHLORIDE", 1.30),
                "mylar": ("G4_MYLAR", 1.40), "polystyrene": ("G4_POLYSTYRENE", 1.06), "silicon": ("G4_Si", 2.33),
                "pvt": ("G4_PLASTIC_SC_VINYLTOLUENE", 1.032), "air": ("G4_AIR", 0.00120479)}
CHUNK = 2500
KEEP_NEUTRAL_EMIN = 0.01          # GeV kinetic, neutral particles below are dropped (charged all kept)
NEUTRAL = [22, 111, 130, 310, 2112, 3122]
MASS = {22: 0.0, 111: 0.13498, 130: 0.49761, 310: 0.49761, 2112: 0.93957, 3122: 1.11568}


def _mass(pdg):
    out = np.zeros(len(pdg))
    for k, m in MASS.items():
        out[np.abs(pdg) == k] = m
    return out


def kinematics(E, mu):
    """nu, Q^2, W of muon-nucleon events (scattered muon momenta mu (n,3), beam mu- of total energy E along +z)."""
    e_out = np.sqrt((mu ** 2).sum(1) + M_MU ** 2)
    p_in = np.sqrt(E ** 2 - M_MU ** 2)
    nu = E - e_out
    q2 = 2 * (E * e_out - p_in * mu[:, 2]) - 2 * M_MU ** 2
    return nu, q2, np.sqrt(np.maximum(M_N ** 2 + 2 * M_N * nu - q2, 0))


# ------------------------------------------------------------------ Pythia 6
def p6_chunk(job):
    """One chunk of n events at (E, target); retried with new seeds until n complete events (hangs killed)."""
    ie, E, tgt, n, seed, tmp = job
    recs, done, sigs, kills = [], 0, [], 0
    while done < n:
        k = n - done
        out = Path(tmp) / f"p6_{tgt}_{ie}_{seed}.bin"
        try:
            subprocess.run([P6, "gamma/mu-", tgt, str(E), str(k), str(seed), str(out)], timeout=20 + 0.01 * k,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            killed = False
        except subprocess.TimeoutExpired:
            killed = True
        seed += 7919
        r = np.fromfile(out, dtype=np.float64) if out.exists() else np.zeros(0)
        r = r[: len(r) // 9 * 9].reshape(-1, 9)
        txt = Path(str(out) + ".txt")
        meta = dict(l.split()[:2] for l in open(txt)) if txt.exists() else {}
        for f in (out, txt):
            f.unlink(missing_ok=True)
        if killed:
            kills += 1
            if len(r):
                r = r[r[:, 0] < r[:, 0].max()]            # the event being written when it hung
        if kills > 30:
            raise RuntimeError(f"Pythia keeps hanging at E={E} {tgt}")
        d = int(r[:, 0].max()) + 1 if len(r) else 0
        if d == 0:
            continue
        r[:, 0] += done
        recs.append(r)
        if "sigma_mb" in meta and int(meta.get("events", 0)) > 0:
            sigs.append((float(meta["sigma_mb"]) * 1e3, int(meta["events"])))
        done += d
    return ie, tgt, np.concatenate(recs), done, sigs, kills


def p6_pack(r, E):
    """Records of one point (event ids 0..n-1) -> per-event arrays (nu, W) and kept particles (sorted by event)."""
    ev = r[:, 0].astype(np.int64)
    order = np.lexsort((-r[:, 5], ev))
    r, ev = r[order], ev[order]
    pdg = r[:, 1].astype(np.int32)
    n_ev = int(ev.max()) + 1
    # scattered muon: the most energetic mu- of the event (first mu- in each event after the sort)
    is_mu = pdg == 13
    first_mu = np.full(n_ev, -1)
    idx = np.nonzero(is_mu)[0]
    _, f = np.unique(ev[idx], return_index=True)
    first_mu[ev[idx[f]]] = idx[f]
    beam = np.zeros(len(r), bool)
    beam[first_mu[first_mu >= 0]] = True
    soft = np.isin(np.abs(pdg), NEUTRAL) & (r[:, 5] - _mass(pdg) < KEEP_NEUTRAL_EMIN)
    keep = beam | (~soft & ~np.isin(np.abs(pdg), [12, 14, 16]))
    mu = np.where((first_mu >= 0)[:, None], r[np.maximum(first_mu, 0), 2:5], np.nan)
    nu, q2, w = kinematics(E, mu)
    return dict(nu=nu, w=w, ev=ev[keep], pdg=pdg[keep], p=r[keep, 2:6], v=r[keep, 6:9], beam=beam[keep])


# ------------------------------------------------------------------ Geant4 (one process per material)
def g4_worker(material, n, out):
    sys.path.insert(0, str(paths.cpp_build()))
    import g4dis
    nist, rho = G4_MATERIALS[material]
    g4dis.initialize(seed=4242 + list(G4_MATERIALS).index(material), material=nist)
    rng = np.random.default_rng(99 + list(G4_MATERIALS).index(material))
    res = {}
    for ie, E in enumerate(E_GRID):
        o = g4dis.generate(13, float(E), n)
        ev, pr = {k: np.asarray(v) for k, v in o["events"].items()}, {k: np.asarray(v) for k, v in o["products"].items()}
        mu = np.stack([ev["mu_px"], ev["mu_py"], ev["mu_pz"]], 1)
        ok = (ev["done"] > 0) & np.isfinite(mu).all(1)
        nu, q2, w = kinematics(float(E), np.where(ok[:, None], mu, 0.0))
        xs = g4dis.cross_section(13, float(E)) / (rho * N_A * 1e6) * 1e34     # 1/m -> microbarn per nucleon
        soft = np.isin(np.abs(pr["pdg"]), NEUTRAL) & (pr["E"] - _mass(pr["pdg"]) < KEEP_NEUTRAL_EMIN)
        keep = ok[pr["ev"]] & (np.abs(pr["pdg"]) < 1000000000) & ~soft
        # renumber the good events 0..k-1; short-lived products decayed (fastveto.decay); the scattered muon first
        new = np.cumsum(ok) - 1
        k = int(ok.sum())
        pe, pq, pp, pv = decay.decay_all(new[pr["ev"][keep]], pr["pdg"][keep].astype(np.int32),
                                         np.stack([pr["px"][keep], pr["py"][keep], pr["pz"][keep], pr["E"][keep]], 1),
                                         np.zeros((int(keep.sum()), 3)), rng)
        soft2 = np.isin(np.abs(pq), NEUTRAL) & (pp[:, 3] - _mass(pq) < KEEP_NEUTRAL_EMIN)
        pe, pq, pp, pv = pe[~soft2], pq[~soft2], pp[~soft2], pv[~soft2]
        pm = mu[ok]
        pdg = np.concatenate([np.full(k, 13, np.int32), pq.astype(np.int32)])
        evs = np.concatenate([np.arange(k), pe])
        p = np.concatenate([np.concatenate([pm, np.sqrt((pm ** 2).sum(1) + M_MU ** 2)[:, None]], 1), pp])
        v = np.concatenate([np.zeros((k, 3)), pv])
        beam = np.concatenate([np.ones(k, bool), np.zeros(len(pe), bool)])
        order = np.argsort(evs, kind="stable")
        res[ie] = dict(sigma=xs, nu=nu[ok], w=w[ok], ev=evs[order], pdg=pdg[order], p=p[order], v=v[order], beam=beam[order])
        print(f"  g4 {material} E={E:7.2f}: sigma {xs:.4g} ub/nucleon, {k} events, W<2 fraction {np.mean(w[ok] < 2):.3f}", flush=True)
    np.save(out, res, allow_pickle=True)


def pack(points):
    """[(ie, tgt, point dict)] -> flat event and particle arrays of a table."""
    cols = {k: [] for k in ("ev_ie", "ev_tgt", "ev_nu", "ev_w", "ev_start", "ev_n", "pdg", "p", "v", "beam")}
    n_part = 0
    for ie, tgt, d in points:
        n_ev = len(d["nu"])
        cnt = np.bincount(d["ev"], minlength=n_ev)
        cols["ev_ie"].append(np.full(n_ev, ie, np.int16)); cols["ev_tgt"].append(np.full(n_ev, tgt, np.int8))
        cols["ev_nu"].append(d["nu"].astype(np.float32)); cols["ev_w"].append(d["w"].astype(np.float32))
        cols["ev_start"].append(n_part + np.cumsum(cnt) - cnt); cols["ev_n"].append(cnt.astype(np.int32))
        cols["pdg"].append(d["pdg"].astype(np.int32)); cols["p"].append(d["p"].astype(np.float32))
        cols["v"].append(d["v"].astype(np.float32)); cols["beam"].append(d["beam"].astype(bool))
        n_part += len(d["pdg"])
    return {k: np.concatenate(v) for k, v in cols.items()}


def crossover(E, nu, sig, n_gen):
    """nu_x for one energy: first nu > NU_X_MIN where Pythia's dsigma/dnu (events nu, cross section sig over n_gen
    events) reaches Kokoulin's (helium per nucleon); E (no Pythia) if it never does."""
    if sig <= 0 or not len(nu):
        return float(E)
    edges = np.geomspace(NU_X_MIN, E, 25)
    h, _ = np.histogram(nu, edges)
    p6 = h / n_gen * sig / np.diff(edges)
    kk = np.array([kokoulin.sigma(E, nu_min=x, A=4)[0] / 4 for x in edges])
    g4 = -np.diff(kk) / np.diff(edges)
    above = np.nonzero((p6 >= g4) & (h >= 20))[0]
    return float(edges[above[0]]) if len(above) else float(E)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=min(300, os.cpu_count()))
    ap.add_argument("--n-p6", type=int, default=20000, help="Pythia events per energy and target nucleon")
    ap.add_argument("--n-g4", type=int, default=20000, help="Geant4 events per energy and material")
    ap.add_argument("--out", default=str(paths.interactions_data() / "dis/mu_library.npz"))
    ap.add_argument("--g4-worker", nargs=2, metavar=("MATERIAL", "OUT"), help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.g4_worker:
        g4_worker(a.g4_worker[0], a.n_g4, a.g4_worker[1])
        return
    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="dislib_", dir=str(Path(a.out).parent))
    g4p = {m: subprocess.Popen([sys.executable, __file__, "--n-g4", str(a.n_g4), "--g4-worker", m, f"{tmp}/g4_{m}.npy"],
                               stdout=open(f"{tmp}/g4_{m}.log", "w"), stderr=subprocess.STDOUT) for m in G4_MATERIALS}
    jobs = []
    for ie, E in enumerate(E_GRID):
        if E < E_P6_MIN:
            continue
        for ti, tgt in enumerate(("p+", "n0")):
            for c in range(0, a.n_p6, CHUNK):
                jobs.append((ie, float(E), tgt, min(CHUNK, a.n_p6 - c), 100000 * ie + 50000 * ti + c, tmp))
    # processes, not threads: many threads spawning subprocesses segfault the container's python
    with ProcessPoolExecutor(max(1, a.jobs - len(G4_MATERIALS)), mp_context=mp.get_context("fork")) as ex:
        res = list(ex.map(p6_chunk, jobs))
    sig_p6 = np.zeros((2, len(E_GRID)))
    nu_x = E_GRID.astype(float).copy()
    points, kills = [], 0
    for ie in range(len(E_GRID)):
        nus, sigs_e = [], []
        for ti, tgt in enumerate(("p+", "n0")):
            parts = [x for x in res if x[0] == ie and x[1] == tgt]
            if not parts:
                continue
            recs, off = [], 0
            for _, _, r, done, sigs, k in parts:
                r = r.copy(); r[:, 0] += off; off += done
                recs.append(r); kills += k
            allsig = [s for x in parts for s in x[4]]
            sig_p6[ti, ie] = sum(s * n for s, n in allsig) / sum(n for s, n in allsig)
            d = p6_pack(np.concatenate(recs), float(E_GRID[ie]))
            points.append((ie, ti, d))
            nus.append(d["nu"]); sigs_e.append(sig_p6[ti, ie])
        if nus:
            nu_all = np.concatenate(nus)
            nu_x[ie] = crossover(float(E_GRID[ie]), nu_all, float(np.mean(sigs_e)), len(nu_all))
    print(f"Pythia done in {time.time() - t0:.0f} s, {sum(len(p[2]['nu']) for p in points):,} events, {kills} hung runs "
          f"re-done; nu_x: " + ", ".join(f"{E:.3g}:{x:.3g}" for E, x in zip(E_GRID, nu_x) if E >= E_P6_MIN), flush=True)
    lib = {f"p6_{k}": v for k, v in pack(points).items()}
    lib["p6_sigma"] = sig_p6
    lib["nu_x"] = nu_x
    for m, p in g4p.items():
        if p.wait() != 0:
            raise RuntimeError(f"Geant4 worker {m} failed, see {tmp}/g4_{m}.log")
        r = np.load(f"{tmp}/g4_{m}.npy", allow_pickle=True).item()
        for k, v in pack([(ie, 0, r[ie]) for ie in range(len(E_GRID))]).items():
            lib[f"g4_{m}_{k}"] = v
        lib[f"g4_{m}_sigma"] = np.array([r[i]["sigma"] for i in range(len(E_GRID))])
        print(f"  {m}: {len(lib[f'g4_{m}_ev_nu']):,} events", flush=True)
    lib["E_grid"] = E_GRID
    lib["meta"] = np.array(json.dumps({
        "n_p6": a.n_p6, "n_g4": a.n_g4, "materials": list(G4_MATERIALS), "nu_x_min": NU_X_MIN,
        "units": "sigma microbarn per nucleon; p GeV (px, py, pz, E), v m; incoming mu- along +z",
        "pythia": "6.4.28 gamma/mu- FIXT, PARP(2)=2, defaults otherwise", "geant4": "11.2.2 FTFP_BERT muonNuclear"}))
    np.savez(a.out, **lib)
    for f in Path(tmp).iterdir():
        f.unlink()
    Path(tmp).rmdir()
    print(f"library {a.out}: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
