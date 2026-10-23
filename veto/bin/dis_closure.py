"""Closure of the fast muon-interaction step against the full Geant4 simulation of the same interactions.

    bin/lcg.sh python3 bin/dis_closure.py [--dir data/muon_dis] [--lib data/dis/mu_library.npz] [--out results/dis_closure]

Full Geant4 (muon_dis_forced.py): muons drawn by muon_dis_rates.py (vertex in the helium / PVC, momentum), one forced
muon-nuclear interaction each, everything transported (FTFP_BERT) to z = 82 m. Fast: the same draws, final state from
the library (fastveto.dis.MuonLibrary), straight lines to z = 82 m. Split "geant4" tests the fast chain itself (same
physics, no transport), "crossover" / "w2" show the effect of Pythia 6. Compared per material: charged tracks
other than the primary muon with p > 1 GeV in the SST1 acceptance (|x| < 2, |y| < 3 m at z = 82 m), and the
same with the primary muon in SST1.
"""

import _bootstrap  # noqa: F401
import argparse
import glob
import json
from pathlib import Path

import numpy as np

from veto import paths

from fastveto import dis

CHARGED = (11, 13, 211, 321, 2212, 3112, 3222, 3312, 3334)
SPILLS_15Y = 6e20 / 5e13
M_MU = 0.10566


def full_geant4(d):
    """Per draw (event index): number of SST1 tracks besides the muon, muon in SST1; and the events simulated."""
    ev_all, k_all, mu_all = [], [], []
    for f in sorted(glob.glob(f"{d}/forced_*.npz")):
        z = np.load(f)
        s, e = int(z["start"]), int(z["stop"])
        ev = z["cross_ev"]
        pc = np.sqrt(z["cross_px"] ** 2 + z["cross_py"] ** 2 + z["cross_pz"] ** 2)
        acc = (z["cross_plane"] == 1) & (np.abs(z["cross_x"]) < 2) & (np.abs(z["cross_y"]) < 3)
        trk = acc & (z["cross_track"] != 1) & (pc > 1)
        has_int = np.zeros(e - s, bool)
        has_int[np.unique(z["int_ev"])] = True
        k = np.bincount(ev[trk], minlength=e - s)
        mu = np.zeros(e - s, bool)
        mu[np.unique(ev[acc & (z["cross_track"] == 1)])] = True
        idx = np.nonzero(has_int)[0]
        ev_all.append(idx + s); k_all.append(k[idx]); mu_all.append(mu[idx])
    return np.concatenate(ev_all), np.concatenate(k_all), np.concatenate(mu_all)


def fast(lib, dr, ev, rng):
    v, p = dr["vertex"][ev], dr["momentum"][ev]
    pm = np.linalg.norm(p, axis=1)
    E = np.sqrt(pm ** 2 + M_MU ** 2)
    u = p / pm[:, None]
    mat = np.where(dr["material"][ev] == "helium", "helium", "pvc")
    k = np.zeros(len(ev), int)
    mu = np.zeros(len(ev), bool)
    for m in ("helium", "pvc"):
        sel = np.nonzero(mat == m)[0]
        o = lib.sample(m, E[sel], u[sel], v[sel], dr["pdg"][ev][sel], rng)
        pp, x0 = o["p"][:, :3], o["v"]
        with np.errstate(invalid="ignore", divide="ignore"):
            t = (82.0 - x0[:, 2]) / pp[:, 2]
        x, y = x0[:, 0] + t * pp[:, 0], x0[:, 1] + t * pp[:, 1]
        acc = (pp[:, 2] > 0) & (x0[:, 2] < 82) & (np.abs(x) < 2) & (np.abs(y) < 3)
        ch = np.isin(np.abs(o["pdg"]), CHARGED) & (np.linalg.norm(pp, axis=1) > 1)
        k[sel] = np.bincount(o["int"][acc & ch & ~o["beam"]], minlength=len(sel))
        mu[sel[np.unique(o["int"][acc & o["beam"]])]] = True
    return k, mu


def summary(k, mu, mat, w):
    out = {}
    for m in ("helium", "pvc"):
        s = mat == m
        out[m] = {"n": int(s.sum()), "frac_k": [float(np.mean(k[s] == j)) for j in range(4)] + [float(np.mean(k[s] >= 4))],
                  ">=1_15y": float(np.sum(k[s] >= 1) * w * SPILLS_15Y), ">=2_15y": float(np.sum(k[s] >= 2) * w * SPILLS_15Y),
                  "mu+>=1_15y": float(np.sum(mu[s] & (k[s] >= 1)) * w * SPILLS_15Y)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(paths.interactions_data() / "muon_dis"))
    ap.add_argument("--lib", default=str(paths.interactions_data() / "dis/mu_library.npz"))
    ap.add_argument("--out", default=str(paths.results_dir() / "dis_closure"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dr = np.load(f"{a.dir}/draws.npz")
    w = float(dr["w_per_interaction"])
    ev, k_g4, mu_g4 = full_geant4(a.dir)
    mat = np.where(dr["material"][ev] == "helium", "helium", "pvc")
    res = {"full_geant4": summary(k_g4, mu_g4, mat, w)}
    g4lib = dis.MuonLibrary(a.lib, split="geant4")
    for split in ("geant4", "crossover", "w2"):
        lib = g4lib if split == "geant4" else dis.MuonLibrary(a.lib, split=split)
        k, mu = fast(lib, dr, ev, np.random.default_rng(7))
        # the draws were made with Geant4's total cross section: other splits change the rate by sigma_split / sigma_G4
        E = np.sqrt(np.linalg.norm(dr["momentum"][ev], axis=1) ** 2 + M_MU ** 2)
        ratio = np.ones(len(ev))
        if split != "geant4":
            for m in ("helium", "pvc"):
                s = mat == m
                num, den = lib.sigma(m, E[s]), g4lib.sigma(m, E[s])
                ratio[s] = np.divide(num, den, out=np.ones(len(num)), where=den > 0)
        r = {}
        for m in ("helium", "pvc"):
            s = mat == m
            r[m] = {"n": int(s.sum()), "frac_k": [float(np.mean(k[s] == j)) for j in range(4)] + [float(np.mean(k[s] >= 4))],
                    ">=1_15y": float(np.sum(ratio[s] * (k[s] >= 1)) * w * SPILLS_15Y),
                    ">=2_15y": float(np.sum(ratio[s] * (k[s] >= 2)) * w * SPILLS_15Y),
                    "mu+>=1_15y": float(np.sum(ratio[s] * (mu[s] & (k[s] >= 1))) * w * SPILLS_15Y),
                    "rate_ratio_to_geant4": float(np.mean(ratio[s]))}
        res[f"fast_{split}"] = r
    json.dump(res, open(out / "closure.json", "w"), indent=1)
    print(f"{len(ev):,} interactions (helium {np.sum(mat == 'helium'):,}, PVC {np.sum(mat == 'pvc'):,}); per 15 years:")
    for m in ("helium", "pvc"):
        print(f"  {m}:")
        for name, r in res.items():
            x = r[m]
            print(f"    {name:16s} tracks besides mu in SST1 (p>1): P(0..3,4+) {np.round(x['frac_k'], 4)}  "
                  f">=1 {x['>=1_15y']:.3g}  >=2 {x['>=2_15y']:.3g}  mu+>=1 {x['mu+>=1_15y']:.3g}")


if __name__ == "__main__":
    main()
