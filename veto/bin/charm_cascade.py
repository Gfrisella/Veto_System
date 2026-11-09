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

--hadrons: the weakly decaying charm hadrons are written instead (pythia6_charm had: undecayed, as FairShip's charm input
file) to data/tail/<tag>_had_<K>.npz (pdg, momentum, production point, weight per proton, events); bin/hf_decay.py decays
and transports them with tailtransport's complete mode (Pythia 8 decays, every muon source, as FairShip's charm production).
--flavour beauty: b bbar (MSEL 5), chibb = 1.6e-7 (FairShip run_fixedTarget.py), its own chi table, tag beautyc.
"""

import _bootstrap  # noqa: F401
import argparse
import json
import os
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
BATCH = 100                                         # interactions per Pythia 6 run


def p6_seed(k, gid):
    """Pythia 6 seed of run gid of slice k: unique for k < 50000, gid < 4000, and MRPY(1) = 19780503 + 2 seed + it stays
    below Pythia 6's limit of 9e8 (the earlier 1e5 k + gid overflowed for k > 10737)."""
    return 4000 * k + gid


FLAVOUR = {"charm": dict(msel=4, chi=1.7e-3, table=OUT / "charm_chi_table.json", tag="charmc"),
           "beauty": dict(msel=5, chi=1.6e-7, table=OUT / "beauty_chi_table.json", tag="beautyc")}
NAMES = {2212: "p+", 2112: "n0", 211: "pi+", -211: "pi-", 321: "K+", -321: "K-", 130: "KL0", 310: "KS0",
         -2212: "pbar-", -2112: "nbar0"}
GRID = [34, 45, 60, 80, 100, 125, 150, 200, 250, 300, 350, 400]
PBINS = np.array([34, 40, 50, 60, 70, 80, 100, 120, 150, 180, 220, 260, 300, 350, 401])
FRACP, CHICC, EMIN_CASC, EMIN_MU = 0.43, 1.7e-3, 34.0, 2.0
FRACP_TARGET = {"FairShip": 0.40, "W": 0.40, "W+Fe": 0.40}          # makeCascade.py: tungsten 74 / 184, else (Mo) 0.43
# FairShip python/heavyFlavourScaling.py: chicc / chibb are given for Mo (A = 98) and scale with the target as
# (A / 98)^(1 - 0.71); tungsten (A = 184): x 1.200
A_TARGET = {"FairShip": 184.0, "W": 184.0, "W+Fe": 184.0}


def hf_scale(target):
    return (A_TARGET.get(target, 98.0) / 98.0) ** (1 - 0.71)
EMIN_HAD = 20.0                    # --hadrons: hadrons above this energy (muons above 20 GeV at the plane)


def make_table(flavour="charm"):
    from concurrent.futures import ThreadPoolExecutor
    fl = FLAVOUR[flavour]
    jobs = [(b, p) for b in sorted(set(NAMES.values())) for p in GRID]

    def run(job):
        b, p = job
        out = subprocess.run([P6C, "xsec", b, str(p), "20000", "7", str(fl["msel"])], capture_output=True, text=True).stdout
        vals = {}
        for line in out.splitlines():
            f = line.split()
            if len(f) == 5:
                vals[f[2]] = (float(f[3]), float(f[4]))
        if "p+" not in vals or "n0" not in vals:      # below threshold (b bbar at low momentum): no production
            return b, p, 0.0
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
    json.dump({"grid": GRID, "ratio_to_p400": table, "chi_p400": ref, "fracp": FRACP}, open(fl["table"], "w"), indent=1)
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


def run_slice(k, events, flavour="charm", hadrons=False, target="MoW"):
    global EMIN_HAD, FRACP
    FRACP = FRACP_TARGET.get(target, 0.43)
    if os.environ.get("HF_EMIN"):                     # hadron threshold of --hadrons (default 20 GeV)
        EMIN_HAD = float(os.environ["HF_EMIN"])
    import tailtransport as T
    t0 = time.time()
    fl = FLAVOUR[flavour]
    table = json.load(open(fl["table"]))
    binf = OUT / f"charmc_mb_{k}.bin"
    subprocess.run([P6M, str(events), str(4000 + k), str(EMIN_CASC), str(binf), str(FRACP)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    meta = dict(l.split()[:2] for l in open(str(binf) + ".txt") if len(l.split()) >= 2)
    n_ev = int(meta["events_generated"])
    rec = np.fromfile(binf, dtype=np.float64).reshape(-1, 6)
    binf.unlink()
    T.initialize(seed=6000 + k, target=target)
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
    lost_w = []
    for nm in np.unique(names):
        for b in np.unique(pb[names == nm]):
            sel = np.nonzero((names == nm) & (pb == b))[0]
            if chi[sel].sum() <= 0:                       # below threshold (b bbar at low momentum): nothing produced
                continue
            pc = 0.5 * (PBINS[b] + PBINS[b + 1]) if b < len(PBINS) - 2 else 400.0
            parts = []
            # batches of <= BATCH interactions, each with its own seed and a short time limit; a batch that fails is retried
            # with new seeds, then split in halves down to single interactions (Pythia 6 can loop near threshold)
            todo = [sel[i:i + BATCH] for i in range(0, len(sel), BATCH)]
            while todo:
                bsel = todo.pop()
                ok = False
                for attempt in range(3):
                    gid += 1
                    if gid >= 4000:
                        raise RuntimeError("more than 4000 Pythia 6 runs in a slice: seed range exceeded")
                    cf = OUT / f"{fl['tag']}_{k}_{gid}.bin"
                    try:
                        subprocess.run([P6C, "had" if hadrons else "gen", nm, f"{pc:.1f}", str(len(bsel)), str(p6_seed(k, gid)),
                                        str(EMIN_HAD if hadrons else EMIN_MU), str(cf), str(FRACP), str(fl["msel"])], check=True,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30 + 0.2 * len(bsel))
                        ok = Path(str(cf) + ".txt").exists()
                    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                        ok = False
                    if ok:
                        break
                    for f_ in (cf, Path(str(cf) + ".txt")):
                        if f_.exists():
                            f_.unlink()
                if not ok:
                    if len(bsel) > 1:                        # split and try the halves
                        todo += [bsel[:len(bsel) // 2], bsel[len(bsel) // 2:]]
                        continue
                    lost_w.append(float((wint[bsel] * chi[bsel]).sum()))
                    print(f"   no output for {nm} at {pc:.1f} GeV (1 interaction, chi weight {chi[bsel].sum():.3g})", flush=True)
                    continue
                r = np.fromfile(cf, dtype=np.float64).reshape(-1, 6)
                cf.unlink(); Path(str(cf) + ".txt").unlink()
                if len(r):
                    r = r.copy(); r[:, 0] = bsel[r[:, 0].astype(int)]   # batch index -> interaction index
                    parts.append(r)
            if not parts:
                continue
            r = np.concatenate(parts)
            idx = r[:, 0].astype(int)                            # charm event i belongs to interaction idx
            u = pvec[idx] / pmag[idx, None]
            p_lab = rotate_to(r[:, 2:5], u)
            for kk, v in (("pdg", r[:, 1].astype(int)), ("px", p_lab[:, 0]), ("py", p_lab[:, 1]), ("pz", p_lab[:, 2]),
                          ("x", pos[idx, 0]), ("y", pos[idx, 1]), ("z", pos[idx, 2]), ("w", wint[idx] * chi[idx]),
                          ("src", (idx >= n_ev).astype(int))):
                mu[kk].append(v)
    mu = {kk: np.concatenate(v) for kk, v in mu.items()}
    fwd = mu["pz"] > 0
    mu = {kk: v[fwd] for kk, v in mu.items()}
    if hadrons:                                       # charm / beauty hadrons for bin/hf_decay.py
        np.savez(OUT / f"{fl['tag']}_had_{k}.npz", events=n_ev, pdg=mu["pdg"], px=mu["px"], py=mu["py"], pz=mu["pz"],
                 x=mu["x"], y=mu["y"], z=mu["z"], w=fl["chi"] * hf_scale(target) * mu["w"], src=mu["src"])
        print(f"slice {k}: {n_ev} protons, {len(it['pdg'])} cascade interactions > {EMIN_CASC:.0f} GeV; {len(mu['pdg'])} "
              f"{flavour} hadrons > {EMIN_HAD:.0f} GeV, weight per proton {fl['chi'] * hf_scale(target) * mu['w'].sum() / n_ev:.4g} "
              f"(chi x {hf_scale(target):.3f} for the target, fracp {FRACP}); production "
              f"weight lost in failed bins {sum(lost_w) / max((wint * chi).sum(), 1e-300):.2%}; "
              f"total {time.time() - t0:.0f} s", flush=True)
        return
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
    ap.add_argument("--flavour", choices=["charm", "beauty"], default="charm")
    ap.add_argument("--hadrons", action="store_true", help="write the charm / beauty hadrons (bin/hf_decay.py) instead of muons")
    ap.add_argument("--target", default="MoW", help="target of the cascade (tailtransport targets)")
    a = ap.parse_args()
    if a.table:
        global FRACP
        FRACP = FRACP_TARGET.get(a.target, 0.43)
        make_table(a.flavour)
    else:
        run_slice(a.slice, a.events, a.flavour, a.hadrons, a.target)


if __name__ == "__main__":
    main()
