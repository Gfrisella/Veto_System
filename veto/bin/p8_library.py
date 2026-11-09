#!/usr/bin/env python3
"""Pythia 8 vs Pythia 6 on the same phase space: hard DIS, Q^2 > 1 GeV^2 and W > 2 GeV (option 1, 2026-11-06).

    python3 bin/p8_library.py [--n 20000] [--jobs 48]

Writes two copies of data/interactions/dis/mu_library_iron.npz whose "p6" part (used with split "pythia") is replaced:
  mu_library_iron_p8dis.npz : Pythia 8.230 (CVMFS), mu- on p / n at rest, WeakBosonExchange:ff2ff(t:gmZ), Q^2 > 1,
                              events with W < 2 removed, sigma = sigmaGen x kept fraction (cpp/pythia8dis/pythia8_dis)
  mu_library_iron_p6q2.npz  : the existing Pythia 6 events with Q^2 > 1 (Q^2 = M^2 + 2 M nu - W^2), sigma x fraction
Same decays in both (pi+-, K+-, K_L, n, mu stable; K_S, Lambda, pi0 decayed). Geant4 parts untouched (not used).
Pythia 8.230 vertices are unusable (hadrons at ~1e6 m): written as 0, weak decays redone by fastveto.decay.

    python3 bin/p8_library.py --variant full [--n 20000]      (2026-11-09, SHiP Annual Report 2026 Table 7)
  mu_library_iron_p8full.npz : Pythia 8 photoproduction (PDF:lepton2gamma, Photon:Q2max = 1.5, Photon:Wmin = 5, the
                              lowest value 8.230 allows; only where W_max > 5, E_mu >~ 13 GeV) + DIS (Q^2 > 1.5, W > 2),
                              mixed per (energy, target) in proportion to their cross sections. Not covered (as in the
                              report): Q^2 < 1.5 with W < 5 GeV, and photoproduction below ~13 GeV.
"""
import _bootstrap  # noqa: F401
import argparse, json, subprocess, tempfile, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from veto import paths
import importlib.util
spec = importlib.util.spec_from_file_location("dis_library", str(Path(__file__).with_name("dis_library.py")))
dl = importlib.util.module_from_spec(spec); spec.loader.exec_module(dl)

P8 = str(paths.PROJECT_DIR / "cpp/pythia8dis/pythia8_dis")
DIS = paths.PROJECT_DIR / "data/interactions/dis"
CHUNK = 2500


PHOTO_SETTINGS = ("Photon:Wmin = 5.", "Photon:Q2max = 1.5")     # report Table 7 (W_min default 10 -> 5)
DIS_SETTINGS_FULL = ("PhaseSpace:Q2Min = 1.5",)                   # DIS above the photoproduction boundary
W_MIN_PHOTO = 5.0


def w_max(E):
    """Largest hadronic mass of a mu- of energy E on a nucleon at rest (Q^2 -> 0, nu -> E - m_mu)."""
    return np.sqrt(dl.M_N ** 2 + 2 * dl.M_N * (E - 0.10566))


def p8_chunk(job):
    ie, E, tgt, n, seed, tmp = job[:6]
    mode, settings = (job[6], job[7]) if len(job) > 6 else ("dis", ())
    out = Path(tmp) / f"p8_{mode}_{tgt}_{ie}_{seed}.bin"
    try:
        subprocess.run([P8, mode, tgt, str(E), str(n), str(seed), str(out), *settings], timeout=60 + 0.05 * n,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        pass
    r = np.fromfile(out, dtype=np.float64) if out.exists() else np.zeros(0)
    r = r[: len(r) // 9 * 9].reshape(-1, 9)
    txt = Path(str(out) + ".txt")
    meta = dict(l.split()[:2] for l in open(txt)) if txt.exists() else {}
    for f in (out, txt):
        f.unlink(missing_ok=True)
    if "events" not in meta and len(r):                 # killed: drop the event being written
        r = r[r[:, 0] < r[:, 0].max()]
    return ie, tgt, r, float(meta.get("sigma_mb", 0)) * 1e3, int(meta.get("events", 0)), mode


def decay_products(d, rng):
    """Weak decays (K_S, Lambda, Sigma, Xi, Omega; kept stable in Pythia 8) with decay lengths, by fastveto.decay,
    as for the Geant4 part (dis_library.g4_worker). Beam muon untouched; events re-sorted."""
    from fastveto import decay
    b = d["beam"]
    pe, pq, pp, pv = decay.decay_all(d["ev"][~b], d["pdg"][~b].astype(np.int32), d["p"][~b].astype(np.float64),
                                     d["v"][~b].astype(np.float64), rng, stable=dl.G4_STABLE)
    soft = np.isin(np.abs(pq), dl.NEUTRAL) & (pp[:, 3] - dl._mass(pq) < dl.KEEP_NEUTRAL_EMIN)
    pe, pq, pp, pv = pe[~soft], pq[~soft], pp[~soft], pv[~soft]
    ev = np.concatenate([d["ev"][b], pe]); o = np.argsort(ev, kind="stable")
    cat = lambda x, y: np.concatenate([x, y])[o]
    return dict(nu=d["nu"], w=d["w"], ev=ev[o], pdg=cat(d["pdg"][b], pq), p=cat(d["p"][b], pp), v=cat(d["v"][b], pv),
                beam=cat(np.ones(int(b.sum()), bool), np.zeros(len(pe), bool)))


def subset(lib, prefix, keep_ev):
    """Table `prefix` restricted to events keep_ev (bool per event), re-packed."""
    st, nn = lib[f"{prefix}ev_start"], lib[f"{prefix}ev_n"]
    s, k = st[keep_ev].astype(np.int64), nn[keep_ev].astype(np.int64)
    rows = np.repeat(s - (np.cumsum(k) - k), k) + np.arange(k.sum())
    out = {f"{prefix}{k}": lib[f"{prefix}{k}"][keep_ev] for k in ("ev_ie", "ev_tgt", "ev_nu", "ev_w", "ev_n")}
    cnt = out[f"{prefix}ev_n"].astype(np.int64)
    out[f"{prefix}ev_start"] = np.cumsum(cnt) - cnt
    for k in ("pdg", "p", "v", "beam"):
        out[f"{prefix}{k}"] = lib[f"{prefix}{k}"][rows]
    return out


def fraction(lib, keep, sigma):
    """sigma (2, nE) x kept fraction per (target, energy)."""
    key = lib["p6_ev_tgt"].astype(np.int64) * 1000 + lib["p6_ev_ie"]
    n_all = np.bincount(key, minlength=2000); n_k = np.bincount(key[keep], minlength=2000)
    f = np.divide(n_k, n_all, out=np.zeros(2000), where=n_all > 0)
    nE = sigma.shape[1]
    return sigma * np.stack([f[:nE], f[1000:1000 + nE]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20000, help="Pythia 8 events per energy and target")
    ap.add_argument("--jobs", type=int, default=48)
    ap.add_argument("--base", default=str(DIS / "mu_library_iron.npz"))
    ap.add_argument("--variant", choices=["dis", "full"], default="dis",
                    help="dis: Pythia 8 DIS Q2 > 1 + the P6 Q2 > 1 copy (2026-11-06); full: photoproduction + DIS")
    a = ap.parse_args()
    if a.variant == "full":
        return main_full(a)
    t0 = time.time()
    base = np.load(a.base); lib = {k: base[k] for k in base.files}
    meta = json.loads(str(lib["meta"]))
    E = lib["E_grid"]

    # ---- Pythia 6, Q^2 > 1
    nu, w = lib["p6_ev_nu"].astype(np.float64), lib["p6_ev_w"].astype(np.float64)
    q2 = dl.M_N ** 2 + 2 * dl.M_N * nu - w ** 2
    keep = q2 > 1.0
    p6 = dict(lib); p6.update(subset(lib, "p6_", keep)); p6["p6_sigma"] = fraction(lib, keep, lib["p6_sigma"])
    p6["meta"] = np.array(json.dumps({**meta, "p6_cut": "Q2 > 1 GeV^2 (bin/p8_library.py)"}))
    np.savez(DIS / "mu_library_iron_p6q2.npz", **p6); del p6
    print(f"P6 Q2>1: {keep.mean():.3f} of the events, {time.time() - t0:.0f} s", flush=True)

    # ---- Pythia 8
    tmp = tempfile.mkdtemp(prefix="p8lib_", dir=str(DIS))
    jobs = [(ie, float(E[ie]), tgt, min(CHUNK, a.n - c), 100000 * ie + 50000 * ti + c + 1, tmp)
            for ie in range(len(E)) if E[ie] >= dl.E_P6_MIN for ti, tgt in enumerate(("p+", "n0"))
            for c in range(0, a.n, CHUNK)]
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(p8_chunk, jobs))
    points, sig = [], np.zeros((2, len(E)))
    rng = np.random.default_rng(8230)
    for ie in range(len(E)):
        for ti, tgt in enumerate(("p+", "n0")):
            parts = [x for x in res if x[0] == ie and x[1] == tgt and len(x[2])]
            if not parts:
                continue
            recs, off = [], 0
            for _, _, r, s, k, _m in parts:
                r = r.copy(); d = int(r[:, 0].max()) + 1; r[:, 0] += off; off += d; recs.append(r)
            sg = [(s, k) for _, _, _, s, k, _m in parts if k > 0 and s > 0]
            d = dl.p6_pack(np.concatenate(recs), float(E[ie]))
            d = decay_products(d, rng)
            ok = d["w"] > 2.0
            frac = ok.mean() if len(ok) else 0.0
            sig[ti, ie] = (sum(s * k for s, k in sg) / sum(k for s, k in sg) if sg else 0.0) * frac
            # keep W > 2 events, renumber
            new = np.cumsum(ok) - 1; m = ok[d["ev"]]
            points.append((ie, ti, dict(nu=d["nu"][ok], w=d["w"][ok], ev=new[d["ev"][m]], pdg=d["pdg"][m],
                                        p=d["p"][m], v=d["v"][m], beam=d["beam"][m])))
            print(f"  P8 {tgt} E={E[ie]:7.2f}: {int(ok.sum())} events W>2 (of {len(ok)}), sigma {sig[ti, ie]:.4g} ub", flush=True)
    for k, v in dl.pack(points).items():
        lib[f"p6_{k}"] = v
    lib["p6_sigma"] = sig
    lib["meta"] = np.array(json.dumps({**meta, "pythia": "8.230 (CVMFS) mu- DIS WeakBosonExchange:ff2ff(t:gmZ), Q2 > 1, "
                                       "W > 2 (bin/p8_library.py), stored in the p6_ tables"}))
    np.savez(DIS / "mu_library_iron_p8dis.npz", **lib)
    Path(tmp).rmdir() if not any(Path(tmp).iterdir()) else None
    print(f"done {time.time() - t0:.0f} s")


def collect(res, ie, tgt, mode, E, rng):
    """Events of one (energy, target, mode): packed, weak decays done; (dict, sigma_gen ub) or (None, 0)."""
    parts = [x for x in res if x[0] == ie and x[1] == tgt and x[5] == mode and len(x[2])]
    if not parts:
        return None, 0.0
    recs, off = [], 0
    for _, _, r, s_, k, _m in parts:
        r = r.copy(); d = int(r[:, 0].max()) + 1; r[:, 0] += off; off += d; recs.append(r)
    sg = [(s_, k) for _, _, _, s_, k, _m in parts if k > 0 and s_ > 0]
    sigma = sum(s_ * k for s_, k in sg) / sum(k for s_, k in sg) if sg else 0.0
    return decay_products(dl.p6_pack(np.concatenate(recs), E), rng), sigma


def take(d, ok):
    """Events of d with ok (bool per event), renumbered."""
    new = np.cumsum(ok) - 1; m = ok[d["ev"]]
    return dict(nu=d["nu"][ok], w=d["w"][ok], ev=new[d["ev"][m]], pdg=d["pdg"][m], p=d["p"][m], v=d["v"][m],
                beam=d["beam"][m])


def main_full(a):
    """Photoproduction (Q2 < 1.5, W > 5) + DIS (Q2 > 1.5, W > 2), mixed by cross section per (energy, target)."""
    t0 = time.time()
    base = np.load(a.base); lib = {k: base[k] for k in base.files}
    meta = json.loads(str(lib["meta"]))
    E = lib["E_grid"]
    tmp = tempfile.mkdtemp(prefix="p8full_", dir=str(DIS))
    jobs = []
    for ie in range(len(E)):
        if E[ie] < dl.E_P6_MIN:
            continue
        for ti, tgt in enumerate(("p+", "n0")):
            for c in range(0, a.n, CHUNK):
                n = min(CHUNK, a.n - c); seed = 100000 * ie + 50000 * ti + c + 1
                jobs.append((ie, float(E[ie]), tgt, n, seed, tmp, "dis", DIS_SETTINGS_FULL))
                if w_max(E[ie]) > W_MIN_PHOTO + 0.05:            # photoproduction only where W > 5 is reachable
                    jobs.append((ie, float(E[ie]), tgt, n, seed + 7, tmp, "photo", PHOTO_SETTINGS))
    print(f"{len(jobs)} Pythia 8 jobs ({sum(j[6] == 'photo' for j in jobs)} photoproduction)", flush=True)
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(p8_chunk, jobs))
    points, sig = [], np.zeros((2, len(E)))
    sig_parts = {"photo": np.zeros((2, len(E))), "dis": np.zeros((2, len(E)))}
    rng = np.random.default_rng(8231)
    for ie in range(len(E)):
        for ti, tgt in enumerate(("p+", "n0")):
            got = {}
            for mode in ("photo", "dis"):
                d, sg = collect(res, ie, tgt, mode, float(E[ie]), rng)
                if d is None or sg <= 0:
                    continue
                if mode == "dis":                               # W > 2 as Pythia 6 (and Q2 > 1.5 from Pythia)
                    ok = d["w"] > 2.0
                else:                                           # generated with Q2 < 1.5, W > 5 by Pythia
                    ok = np.isfinite(d["w"])
                frac = ok.mean() if len(ok) else 0.0
                if ok.sum() == 0 or sg * frac <= 0:             # nothing left after the cut at this energy
                    continue
                got[mode] = (take(d, ok), sg * frac)
                sig_parts[mode][ti, ie] = sg * frac
            if not got:
                continue
            # mix: event counts proportional to the cross sections (subsample the over-represented part)
            scale = min(len(g[0]["nu"]) / g[1] for g in got.values())
            mixed = []
            for mode, (d, sg) in got.items():
                k = int(round(scale * sg)); n_ev = len(d["nu"])
                keep = np.zeros(n_ev, bool); keep[rng.choice(n_ev, min(k, n_ev), replace=False)] = True
                mixed.append(take(d, keep))
            off, cat = 0, {k: [] for k in ("nu", "w", "ev", "pdg", "p", "v", "beam")}
            for d in mixed:
                for k in cat:
                    cat[k].append(d[k] + off if k == "ev" else d[k])
                off += len(d["nu"])
            sig[ti, ie] = sum(g[1] for g in got.values())
            points.append((ie, ti, {k: np.concatenate(v) for k, v in cat.items()}))
            print(f"  P8 {tgt} E={E[ie]:7.2f}: sigma photo {sig_parts['photo'][ti, ie]:.4g} + DIS "
                  f"{sig_parts['dis'][ti, ie]:.4g} ub, {off} events", flush=True)
    for k, v in dl.pack(points).items():
        lib[f"p6_{k}"] = v
    lib["p6_sigma"] = sig
    lib["p8_sigma_photo"], lib["p8_sigma_dis"] = sig_parts["photo"], sig_parts["dis"]
    lib["meta"] = np.array(json.dumps({**meta, "pythia": "8.230 (CVMFS): photoproduction PDF:lepton2gamma Q2 < 1.5, "
                                       "W > 5 (Photon:Wmin) + DIS WeakBosonExchange Q2 > 1.5, W > 2, mixed by sigma "
                                       "(bin/p8_library.py --variant full, Annual Report 2026 Table 7), in the p6_ tables"}))
    np.savez(DIS / "mu_library_iron_p8full.npz", **lib)
    Path(tmp).rmdir() if not any(Path(tmp).iterdir()) else None
    print(f"done {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
