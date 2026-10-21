"""Charm from the whole hadronic cascade, as FairShip's macro/makeCascade.py (one slice), + the chi table.

    bin/g4.sh python3 bin/charm_cascade.py --table                       # once: chi table (Pythia 6)
    bin/g4.sh python3 bin/charm_cascade.py --slice K --n-slices N [--events 100000]

1. Pythia 6 minimum-bias events (first collision; target p / n with makeCascade.py's fracp = 0.43), vertex
   by optical depth in the Mo+W target, then the Geant4 cascade of every hadron above 34 GeV (makeCascade.py's
   pbeaml for charm) with every inelastic interaction of p, n, pi, K, K_L, K_S above 34 GeV RECORDED. The
   first collision is the first interaction (400 GeV proton at the vertex).
2. For each interaction ONE forced c cbar event (Pythia 6, MSEL=4, PoorE791 tune: cpp/tailgen/pythia6_charm),
   generated in bins of projectile and momentum and rotated to the projectile's direction at the interaction
   point; weight = chi(projectile, p) / chi(p, 400 GeV) x chicc (1.7e-3, makeCascade/FixedTargetGenerator) per
   proton on target, with chi = sigma(c cbar, MSEL 4) / sigma(tot, MSEL 2) on p / n (fracp 0.43), as the
   makeCascade.py tables. Muons above 2 GeV are carried to z = -0.0045 m by Geant4.
Output data/tail/charmc_<K>.npz (tail_produce.py layout: w_decay = chicc x chi ratio, events = protons).
"""

import _bootstrap  # noqa: F401
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from veto import paths

sys.path.insert(0, str(paths.PROJECT_DIR / "cpp/tailgen/build"))
from tail_produce import Layout, sample_along, OUT  # noqa: E402

P6C = str(paths.PROJECT_DIR / "cpp/tailgen/build/pythia6_charm")
P6M = str(paths.PROJECT_DIR / "cpp/tailgen/build/pythia6_mesons")
TABLE = OUT / "charm_chi_table.json"
NAMES = {2212: "p+", 2112: "n0", 211: "pi+", -211: "pi-", 321: "K+", -321: "K-", 130: "KL0", 310: "KS0",
         -2212: "pbar-", -2112: "nbar0"}
GRID = [34, 45, 60, 80, 100, 125, 150, 200, 250, 300, 350, 400]
PBINS = np.array([34, 40, 50, 60, 70, 80, 100, 120, 150, 180, 220, 260, 300, 350, 401])
FRACP, CHICC, EMIN_CASC, EMIN_MU = 0.43, 1.7e-3, 34.0, 2.0


def make_table():
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(b, p) for b in sorted(set(NAMES.values())) for p in GRID]

    def run(job):
        b, p = job
        out = subprocess.run([P6C, "xsec", b, str(p), "20000", "7"], capture_output=True, text=True).stdout
        vals = {}
        for line in out.splitlines():
            f = line.split()
            if len(f) == 5:
                vals[f[2]] = (float(f[3]), float(f[4]))
        cc = FRACP * vals["p+"][0] + (1 - FRACP) * vals["n0"][0]
        tot = FRACP * vals["p+"][1] + (1 - FRACP) * vals["n0"][1]
        return b, p, cc / tot
    with ThreadPoolExecutor(32) as ex:
        res = list(ex.map(run, jobs))
    chi = {}
    for b, p, c in res:
        chi.setdefault(b, {})[p] = c
    ref = chi["p+"][400]
    table = {b: {str(p): v / ref for p, v in sorted(d.items())} for b, d in chi.items()}
    json.dump({"grid": GRID, "ratio_to_p400": table, "chi_p400": ref, "fracp": FRACP}, open(TABLE, "w"), indent=1)
    print(json.dumps(table, indent=1))


def chi_ratio(table, name, p):
    t = table["ratio_to_p400"][name]
    g = np.array(table["grid"], float)
    v = np.array([t[str(x)] for x in table["grid"]])
    return np.where(p < g[0], 0.0, np.interp(p, g, v))


def rotate_to(pz_frame, u):
    """Rotate vectors (n,3) given in a frame with the projectile along +z to the direction u (n,3)."""
    a = np.where(np.abs(u[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
    e1 = np.cross(u, a); e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(u, e1)
    return pz_frame[:, 0:1] * e1 + pz_frame[:, 1:2] * e2 + pz_frame[:, 2:3] * u


def run_slice(k, events):
    import tailtransport as T
    t0 = time.time()
    table = json.load(open(TABLE))
    binf = OUT / f"charmc_mb_{k}.bin"
    subprocess.run([P6M, str(events), str(4000 + k), str(EMIN_CASC), str(binf), str(FRACP)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    meta = dict(l.split()[:2] for l in open(str(binf) + ".txt") if len(l.split()) >= 2)
    n_ev = int(meta["events_generated"])
    rec = np.fromfile(binf, dtype=np.float64).reshape(-1, 6)
    binf.unlink()
    T.initialize(seed=6000 + k, target="MoW")
    T.set_pik(False)
    T.set_log_interactions(True, EMIN_CASC)
    lay = Layout(T, T.target_layout())
    rng = np.random.default_rng(6500 + k)
    spot = np.load(OUT / "beamspot.npy")
    # vertex of every proton (also those whose event has no hadron above threshold: their first collision counts)
    zs = np.full(n_ev, lay.z0[0])
    d_p, _ = sample_along(lay.path_pieces(zs, np.ones(n_ev)), lay.lam(2212, np.full(n_ev, 400.0)), rng)
    xy = spot[rng.integers(0, len(spot), n_ev)].astype(np.float64)
    vtx = np.stack([xy[:, 0], xy[:, 1], zs + d_p], 1)
    ev_ids, first = np.unique(rec[:, 0].astype(np.int64), return_index=True)
    offsets = np.append(first, len(rec))
    o = T.cascade(offsets.tolist(), rec[:, 1].astype(int).tolist(), rec[:, 2].tolist(), rec[:, 3].tolist(),
                  rec[:, 4].tolist(), vtx[ev_ids, 0].tolist(), vtx[ev_ids, 1].tolist(), vtx[ev_ids, 2].tolist(),
                  EMIN_CASC, 1e9)
    it = {kk: np.asarray(v) for kk, v in o["interactions"].items()}
    t_casc = time.time() - t0
    # interaction list: first collisions (all protons) + cascade interactions
    names = np.concatenate([np.full(n_ev, "p+"), np.array([NAMES.get(int(q), "p+") for q in it["pdg"]])])
    pvec = np.concatenate([np.tile([0.0, 0.0, 400.0], (n_ev, 1)), np.stack([it["px"], it["py"], it["pz"]], 1)])
    pos = np.concatenate([vtx, np.stack([it["x"], it["y"], it["z"]], 1)])
    wint = np.concatenate([np.ones(n_ev), it["w"]])
    pmag = np.linalg.norm(pvec, axis=1)
    chi = np.array([chi_ratio(table, nm, pp) for nm, pp in zip(names, pmag)]) if len(names) < 0 else None
    chi = np.zeros(len(names))
    for nm in np.unique(names):
        m = names == nm
        chi[m] = chi_ratio(table, nm, pmag[m])
    mu = {kk: [] for kk in ("pdg", "px", "py", "pz", "x", "y", "z", "w", "src")}
    pb = np.clip(np.digitize(pmag, PBINS) - 1, 0, len(PBINS) - 2)
    gid = 0
    for nm in np.unique(names):
        for b in np.unique(pb[names == nm]):
            sel = np.nonzero((names == nm) & (pb == b))[0]
            pc = 0.5 * (PBINS[b] + PBINS[b + 1]) if b < len(PBINS) - 2 else 400.0
            cf = OUT / f"charmc_{k}_{gid}.bin"
            gid += 1
            subprocess.run([P6C, "gen", nm, f"{pc:.1f}", str(len(sel)), str(100000 * k + gid), str(EMIN_MU), str(cf), str(FRACP)],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            r = np.fromfile(cf, dtype=np.float64).reshape(-1, 6)
            cf.unlink(); Path(str(cf) + ".txt").unlink()
            if not len(r):
                continue
            idx = sel[r[:, 0].astype(int)]                       # charm event i belongs to interaction sel[i]
            u = pvec[idx] / pmag[idx, None]
            p_lab = rotate_to(r[:, 2:5], u)
            for kk, v in (("pdg", r[:, 1].astype(int)), ("px", p_lab[:, 0]), ("py", p_lab[:, 1]), ("pz", p_lab[:, 2]),
                          ("x", pos[idx, 0]), ("y", pos[idx, 1]), ("z", pos[idx, 2]), ("w", wint[idx] * chi[idx]),
                          ("src", (idx >= n_ev).astype(int))):
                mu[kk].append(v)
    mu = {kk: np.concatenate(v) for kk, v in mu.items()}
    fwd = mu["pz"] > 0
    mu = {kk: v[fwd] for kk, v in mu.items()}
    tr = T.transport(mu["pdg"].tolist(), mu["x"].tolist(), mu["y"].tolist(), mu["z"].tolist(),
                     mu["px"].tolist(), mu["py"].tolist(), mu["pz"].tolist())
    ok = tr["ok"] > 0
    n = int(ok.sum())
    np.savez(OUT / f"charmc_{k}.npz", events=n_ev, rep=1, emin=EMIN_MU, pmin=EMIN_MU,
             pdg=mu["pdg"][ok], x=tr["x"][ok], y=tr["y"][ok], z=tr["z"][ok], px=tr["px"][ok], py=tr["py"][ok], pz=tr["pz"][ok],
             w_decay=CHICC * mu["w"][ok], parent=np.where(mu["src"][ok] > 0, 41, 40), z_decay=mu["z"][ok],
             gen=mu["src"][ok], event=np.zeros(n, int), z_int=mu["z"][ok], p_prod=np.zeros(n))
    w_first = (chi[:n_ev] * wint[:n_ev]).sum()
    w_casc = (chi[n_ev:] * wint[n_ev:]).sum()
    print(f"slice {k}: {n_ev} protons, {len(it['pdg'])} cascade interactions > {EMIN_CASC:.0f} GeV; charm weight first "
          f"{w_first:.4g}, cascade {w_casc:.4g} (cascade / first {w_casc / w_first:.3f}); {n} muons at the plane; "
          f"cascade {t_casc:.0f} s, total {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--slice", type=int, default=0)
    ap.add_argument("--n-slices", type=int, default=1)
    ap.add_argument("--events", type=int, default=100000)
    a = ap.parse_args()
    if a.table:
        make_table()
    else:
        run_slice(a.slice, a.events)


if __name__ == "__main__":
    main()
