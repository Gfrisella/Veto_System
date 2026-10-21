"""Neutrino-nucleus interaction library with our own GENIE 3.06.02 (tune G18_02a_00_000) for fastveto.nudis.

    bin/lcg.sh python3 bin/nu_library.py [--jobs 300] [--n 2000] [--out data/genie/nu_library.npz]

GENIE (bin/genie.sh: cpp/genie/build_genie.sh, Pythia 6, default configuration + charm and tau decays) at fixed
energies (grid 1 .. 500 GeV) along +z for nu_mu, anti-nu_mu, nu_e, anti-nu_e on the nuclei of our materials (H1, He4,
C12, N14, O16, Si28, Cl35, Cl37, Ar40): gevgen -> gntpc (gst). All processes (QE, MEC, RES, DIS, COH, CC + NC),
nuclear model and final-state interactions as in the tune. Per event the final state after FSI (gst "f" particles)
plus the primary lepton (flag `beam`); K0 / anti-K0 -> K_S or K_L (50 / 50); K_S, Lambda, Sigma, Xi, Omega, pi0,
eta decayed by fastveto.decay (displaced decay points); nuclear remnants / ions dropped; neutral particles below
10 MeV kinetic dropped. Cross sections per nucleus [cm^2]: the sum of the spline set's processes (official GENIE
splines, bin/genie_splines.py -> data/genie/xsec_tot.npz), interpolated to the grid.
"""

import _bootstrap  # noqa: F401
import argparse
import json
import multiprocessing as mp
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from veto import paths

from fastveto import decay

GENIE = str(paths.PROJECT_DIR / "bin/genie.sh")
SPL = paths.interactions_data() / "genie/gxspl-fvs.xml"
TUNE = "G18_02a_00_000"
E_GRID = np.round(np.geomspace(1.0, 500.0, 36), 4)
NUS = [14, -14, 12, -12]
NUCLEI = [1000010010, 1000020040, 1000060120, 1000070140, 1000080160, 1000140280, 1000170350, 1000170370, 1000180400]
KEEP_NEUTRAL_EMIN = 0.01
NEUTRAL = [22, 111, 130, 310, 2112, 3122, 12, 14, 16]


def run_point(job):
    """gevgen + gntpc for one (energy, neutrino, nucleus); returns the gst path (or None)."""
    ie, E, nu, tgt, n, seed, outdir = job
    stem = Path(outdir) / f"nu{nu}_t{tgt}_e{ie}"
    gst = Path(f"{stem}.gst.root")
    if gst.exists() and gst.stat().st_size > 1000:
        return job, str(gst)
    for attempt in range(3):
        r = subprocess.run([GENIE, "gevgen", "-n", str(n), "-p", str(nu), "-t", str(tgt), "-e", str(E), "-r", str(ie),
                            "--seed", str(seed + 7919 * attempt), "--tune", TUNE, "--cross-sections", str(SPL),
                            "--message-thresholds", "Messenger_laconic.xml", "-o", f"{stem}.ghep.root"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3600)
        if Path(f"{stem}.ghep.root").exists():
            subprocess.run([GENIE, "gntpc", "-i", f"{stem}.ghep.root", "-f", "gst", "-o", str(gst),
                            "--message-thresholds", "Messenger_laconic.xml"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3600)
            Path(f"{stem}.ghep.root").unlink(missing_ok=True)
            if gst.exists():
                return job, str(gst)
    return job, None


def lepton_pdg(nu, cc):
    return np.where(cc > 0, nu - np.sign(nu), nu)


def read_point(path, nu, rng):
    """gst -> per-event arrays and decayed particles (sorted by event)."""
    import awkward as ak
    import uproot
    t = uproot.open(path)["gst"]
    a = t.arrays(["cc", "qel", "res", "dis", "coh", "mec", "charm", "W", "Q2", "y", "El", "pxl", "pyl", "pzl",
                  "pdgf", "Ef", "pxf", "pyf", "pzf"], library="ak")
    n = len(a)
    cc = ak.to_numpy(a.cc).astype(np.int8)
    proc = (ak.to_numpy(a.qel) * 1 + ak.to_numpy(a.res) * 2 + ak.to_numpy(a.dis) * 4 + ak.to_numpy(a.coh) * 8
            + ak.to_numpy(a.mec) * 16).astype(np.int8)
    nf = ak.to_numpy(ak.num(a.pdgf))
    ev_f = np.repeat(np.arange(n), nf)
    pdg = ak.to_numpy(ak.flatten(a.pdgf)).astype(np.int64)
    p = np.stack([ak.to_numpy(ak.flatten(a[k])) for k in ("pxf", "pyf", "pzf", "Ef")], 1)
    keep = np.abs(pdg) < 1000000000
    ev_f, pdg, p = ev_f[keep], pdg[keep], p[keep]
    k0 = np.isin(pdg, [311, -311])
    pdg[k0] = np.where(rng.random(k0.sum()) < 0.5, 310, 130)
    ev_f, pdg, p, v = decay.decay_all(ev_f, pdg, p, np.zeros((len(pdg), 3)), rng)
    soft = np.isin(np.abs(pdg), NEUTRAL) & (p[:, 3] - decay.mass(pdg) < KEEP_NEUTRAL_EMIN) & ~np.isin(np.abs(pdg), [12, 14, 16])
    ev_f, pdg, p, v = ev_f[~soft], pdg[~soft], p[~soft], v[~soft]
    lep = np.stack([ak.to_numpy(a[k]) for k in ("pxl", "pyl", "pzl", "El")], 1)
    evs = np.concatenate([np.arange(n), ev_f])
    allp = np.concatenate([lep, p])
    allpdg = np.concatenate([lepton_pdg(nu, cc), pdg])
    allv = np.concatenate([np.zeros((n, 3)), v])
    beam = np.concatenate([np.ones(n, bool), np.zeros(len(pdg), bool)])
    order = np.argsort(evs, kind="stable")
    return dict(cc=cc, proc=proc, charm=ak.to_numpy(a.charm).astype(np.int8), W=ak.to_numpy(a.W), Q2=ak.to_numpy(a.Q2),
                y=ak.to_numpy(a.y), ev=evs[order], pdg=allpdg[order], p=allp[order], v=allv[order], beam=beam[order])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=300)
    ap.add_argument("--n", type=int, default=2000, help="events per energy, neutrino and nucleus")
    ap.add_argument("--out", default=str(paths.interactions_data() / "genie/nu_library.npz"))
    a = ap.parse_args()
    t0 = time.time()
    gdir = paths.interactions_data() / "genie/lib_gst"
    gdir.mkdir(parents=True, exist_ok=True)
    jobs = [(ie, float(E), nu, tgt, a.n, 100000 * ie + 1000 * NUS.index(nu) + NUCLEI.index(tgt), str(gdir))
            for ie, E in enumerate(E_GRID) for nu in NUS for tgt in NUCLEI]
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("fork")) as ex:
        res = list(ex.map(run_point, jobs))
    failed = [j[:4] for j, g in res if g is None]
    print(f"GENIE: {len(res) - len(failed)} of {len(res)} points in {time.time() - t0:.0f} s; failed: {failed[:10]}", flush=True)
    rng = np.random.default_rng(17)
    cols = {k: [] for k in ("ev_ie", "ev_nu", "ev_tgt", "ev_cc", "ev_proc", "ev_charm", "ev_W", "ev_Q2", "ev_y",
                            "ev_start", "ev_n", "pdg", "p", "v", "beam")}
    n_part = 0
    for (ie, E, nu, tgt, *_), g in res:
        if g is None:
            continue
        d = read_point(g, nu, rng)
        n_ev = len(d["cc"])
        cnt = np.bincount(d["ev"], minlength=n_ev)
        for k, v in (("ev_ie", np.full(n_ev, ie, np.int16)), ("ev_nu", np.full(n_ev, NUS.index(nu), np.int8)),
                     ("ev_tgt", np.full(n_ev, NUCLEI.index(tgt), np.int8)), ("ev_cc", d["cc"]), ("ev_proc", d["proc"]),
                     ("ev_charm", d["charm"]), ("ev_W", d["W"].astype(np.float32)), ("ev_Q2", d["Q2"].astype(np.float32)),
                     ("ev_y", d["y"].astype(np.float32)), ("ev_start", n_part + np.cumsum(cnt) - cnt), ("ev_n", cnt.astype(np.int32)),
                     ("pdg", d["pdg"].astype(np.int32)), ("p", d["p"].astype(np.float32)), ("v", d["v"].astype(np.float32)),
                     ("beam", d["beam"])):
            cols[k].append(v)
        n_part += len(d["pdg"])
    lib = {k: np.concatenate(v) for k, v in cols.items()}
    xs = np.load(paths.interactions_data() / "genie/xsec_tot.npz")
    sigma = np.zeros((len(NUS), len(NUCLEI), len(E_GRID)))
    for i, nu in enumerate(NUS):
        for j, tgt in enumerate(NUCLEI):
            m = np.nonzero((xs["nu"] == nu) & (xs["tgt"] == tgt))[0]
            if len(m):
                sigma[i, j] = np.interp(E_GRID, xs["E"], xs["sigma"][m[0]])
    lib.update(E_grid=E_GRID, nus=np.array(NUS), nuclei=np.array(NUCLEI), sigma=sigma,
               meta=np.array(json.dumps({"genie": "3.06.02 (own build, Pythia 6, charm + tau decays)", "tune": TUNE,
                                         "splines": "genie_xsec v3_06_00 G1802a00000-k250-e1000 (NUsmall)", "n": a.n,
                                         "units": "sigma cm^2 per nucleus; p GeV (px, py, pz, E), v m; neutrino along +z",
                                         "proc_bits": {"qel": 1, "res": 2, "dis": 4, "coh": 8, "mec": 16}})))
    np.savez(a.out, **lib)
    print(f"library {a.out}: {len(lib['ev_ie']):,} events, {len(lib['pdg']):,} particles, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
