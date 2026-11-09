#!/usr/bin/env python3
"""The full 2018-equivalent production (exact 2018 software, eiger: p18_mb, p18_onia, charmc_mu, beautyc_mu) against the
2018 FairShip files, class by class, process by process and mother by mother (the inventory of
fairship_match/inventory2018.py), with two weights per muon:
  physical  our weights (every source at its physical rate)
  2018      FairShip's 2018 weighting emulated per muon (muonShieldOptimization/extractMuonsAndUpdateWeight.py + the
            diMu100 boost of FixedTargetGenerator + AddDiMuonDecayChannelsToG4): eta / omega Dalitz muons dropped
            (not boosted, weight / 100), eta / eta' decays in the cascade dropped (no muon channels in 2018's Geant4),
            resonances decayed inside the hadronic models: only omega / phi / rho -> mu mu kept, muons whose direct
            mother is an open-charm hadron dropped from min. bias (2018: "noCharm", taken from the charm production)
Components and 2018 classes:
  min. bias (p18_mb)   768.75: long-lived decays, Pythia muons, other, prompt decays of non-vector-meson species
                       7.6875: gamma -> mu mu, e+ e- -> mu mu, resonances in hadronic models, vector-meson decays
  onia (p18_onia)      768.75 (quarkonium muons; the min. bias ones are dropped, no double counting)
  charm (charmc_mu)    326.16 (3.2616: its vector-meson muons)      beauty (beautyc_mu) 9.37 (0.0937)
Selection on both sides: production momentum >= P_MIN (ours: 2018 selection E > 10 GeV at the 2018 plane emulated).

    python3 bin/compare_inventory_2018.py compute --src $S/cv/tail --out $S/cv/inventory_ours.npz   (eiger, many cores)
    python3 bin/compare_inventory_2018.py refs --out /disk/lhcb_data/sqasim/reference-2018-fairship-506M/inventory_ref.npz  (farm)
    python3 bin/compare_inventory_2018.py report inventory_ours.npz inventory_ref.npz                 (local)
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import os
SHARED = os.environ.get("VETO_SHARED", "/disk/lhcb_data/sqasim")   # Shah Rukh's shared data (read only)

P_MIN = 25.0
PE = np.geomspace(10, 400, 33)
TE = np.linspace(0, 4, 21)
VM = [221, 223, 113, 333, 331]                    # 2018's muSources
CHARM_EXTERN = [4332, 4232, 4132, 4122, 431, 411, 421]
QQ = [443, 100443, 553, 100553, 200553]
CLASSES = {"768.75": 768.75, "7.6875": 7.6875, "326.16": 326.16, "3.2616": 3.2616, "9.37": 9.37, "0.0937": 0.0937}
# our muon source (gen) -> the 2018 creation process (TMC id in the files)
PROC_2018 = {1: 4, 2: 0, 7: 4, 3: 5, 4: 10, 5: 0, 6: 23, 8: 23}


def classify(comp, gen, par, ch, hf):
    """2018 class of each muon and its 2018 weight factor (1 or 0)."""
    apar = np.abs(par)
    vm = np.isin(apar, VM) & np.isin(gen, [2, 7])
    keep = np.ones(len(gen))
    keep[vm & np.isin(apar, [221, 223]) & (ch > 0)] = 0                   # Dalitz-type: not boosted, / 100
    keep[(gen == 7) & np.isin(apar, [221, 331])] = 0                      # cascade eta / eta': no channels in 2018
    keep[(gen == 8) & ~(np.isin(apar, [223, 333, 113]) & (ch == 0))] = 0  # in-model: only omega / phi / rho -> mu mu
    if comp == "mb":
        keep[np.isin(apar, CHARM_EXTERN)] = 0                              # noCharm
        dimu = vm | np.isin(gen, [3, 4, 8])
        cls = np.where(dimu, "7.6875", "768.75")
    elif comp == "onia":
        cls = np.full(len(gen), "768.75")
    elif comp == "charm":
        cls = np.where(vm | np.isin(gen, [3, 4, 8]), "3.2616", "326.16")
    else:
        cls = np.where(vm | np.isin(gen, [3, 4, 8]), "0.0937", "9.37")
    return cls, keep


def one_file(args):
    fn, comp = args
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _bootstrap  # noqa: F401
    import old_convention as oc
    table = oc.loss_map()
    d = np.load(fn)
    pp = d["p_prod"].astype(np.float64)
    pt = np.hypot(d["pcx"], d["pcy"]).astype(np.float64)
    gen, par, ch, hf = d["gen"], d["parent"], d["ch"], d["hf"]
    w = d["w_decay"].astype(np.float64) / float(d["rep"])
    sel = (pp >= P_MIN) & (pp - oc.draw_loss(table, pp) > 10.0)
    if comp == "mb":
        sel &= ~np.isin(np.abs(par), QQ)                                   # quarkonium muons from the onia run
        phys = sel & (hf == 0)                                             # charm / beauty from their own runs
    else:
        phys = sel.copy()
    cls, keep = classify(comp, gen, par, ch, hf)
    out = {}
    for lab, m, ww in (("phys", phys, w), ("2018", sel, w * keep)):
        for c in CLASSES:
            mc = m & (cls == c)
            if not mc.any():
                continue
            out[(lab, c, "flux")] = ww[mc].sum()
            out[(lab, c, "w2")] = (ww[mc] ** 2).sum()
            for g in np.unique(gen[mc]):
                out[(lab, c, "proc", int(PROC_2018.get(int(g), -1)))] = out.get((lab, c, "proc", int(PROC_2018.get(int(g), -1))), 0) + ww[mc & (gen == g)].sum()
            for p in np.unique(par[mc]):
                out[(lab, c, "mother", int(p))] = ww[mc & (par == p)].sum()
            h = np.histogram2d(pp[mc], pt[mc], [PE, TE], weights=ww[mc])[0]
            h2 = np.histogram2d(pp[mc], pt[mc], [PE, TE], weights=ww[mc] ** 2)[0]
            out[(lab, c, "h")], out[(lab, c, "h2")] = h, h2
    return comp, float(d["events"]), out


def compute(a):
    from multiprocessing import Pool
    jobs = []
    for comp, pat in (("mb", "p18_mb_[0-9]*.npz"), ("onia", "p18_onia_[0-9]*.npz"), ("charm", "charmc_mu_[0-9]*.npz"),
                      ("beauty", "beautyc_mu_[0-9]*.npz")):
        jobs += [(f, comp) for f in sorted(glob.glob(f"{a.src}/{pat}"))]
    ev, acc = {}, {}
    with Pool(a.workers) as pool:
        for comp, e, out in pool.imap_unordered(one_file, jobs, chunksize=4):
            ev[comp] = ev.get(comp, 0) + e
            for k, v in out.items():
                acc[(comp,) + k] = acc.get((comp,) + k, 0) + v
    res = {"events": json.dumps(ev), "keys": json.dumps([list(k) for k in acc])}
    for i, (k, v) in enumerate(acc.items()):
        res[f"v{i}"] = np.asarray(v) * (5e13 / ev[k[0]]) ** (2 if k[-1] in ("w2", "h2") else 1)
    np.savez(a.out, PE=PE, TE=TE, **res)
    print({c: len([j for j in jobs if j[1] == c]) for c in ("mb", "onia", "charm", "beauty")}, "files ->", a.out)


def refs(a):
    import h5py
    with h5py.File(f"{SHARED}/reference-2018-fairship-506M/reference_2018.h5", "r") as f:
        wf = f["w_fairship"][:].astype(np.float64)
        w = f["weight"][:].astype(np.float64)
        px, py, pz = (f[k][:].astype(np.float64) for k in ("px", "py", "pz"))
        mo, pr = f["mother"][:], f["proc"][:]
    pt = np.hypot(px, py); p = np.sqrt(pt ** 2 + pz ** 2)
    sel = p >= P_MIN
    res = {}
    for c, v in CLASSES.items():
        m = sel & (np.abs(wf - v) < 0.01 * v)
        res[f"{c}|flux"] = w[m].sum(); res[f"{c}|w2"] = (w[m] ** 2).sum()
        res[f"{c}|h"] = np.histogram2d(p[m], pt[m], [PE, TE], weights=w[m])[0]
        res[f"{c}|h2"] = np.histogram2d(p[m], pt[m], [PE, TE], weights=w[m] ** 2)[0]
        for q in np.unique(pr[m]):
            res[f"{c}|proc|{int(q)}"] = w[m & (pr == q)].sum()
        for q in np.unique(mo[m]):
            res[f"{c}|mother|{int(q)}"] = w[m & (mo == q)].sum()
    np.savez(a.out, **res)
    print("wrote", a.out)


def report(a):
    o, r = np.load(a.ours), np.load(a.ref)
    keys = [tuple(k) for k in json.loads(str(o["keys"]))]
    val = {k: o[f"v{i}"] for i, k in enumerate(keys)}
    print(f"Muons per spill with production momentum >= {P_MIN:.0f} GeV (2018 selection). events: {o['events']}\n")
    for lab in ("2018", "phys"):
        print(f"=== ours with {'FairShip 2018 weighting' if lab == '2018' else 'physical weights'} vs the 2018 files ===")
        tot_o = tot_r = 0.0
        for c in CLASSES:
            fo = sum(float(v) for k, v in val.items() if k[1] == lab and k[2] == c and k[3] == "flux")
            eo = np.sqrt(sum(float(v) for k, v in val.items() if k[1] == lab and k[2] == c and k[3] == "w2"))
            fr, er = float(r.get(f"{c}|flux", 0)), float(np.sqrt(r.get(f"{c}|w2", 0)))
            tot_o += fo; tot_r += fr
            print(f"  class {c:>7}: ours {fo:10.4g} +- {eo / max(fo, 1e-300):6.2%}   2018 {fr:10.4g} +- {er / max(fr, 1e-300):6.2%}   "
                  f"ratio {fo / fr if fr else float('nan'):6.3f}")
            procs = sorted({int(k[4]) for k in val if k[1] == lab and k[2] == c and k[3] == "proc"} |
                           {int(x.split('|')[2]) for x in r.files if x.startswith(f"{c}|proc|")})
            for q in procs:
                po = sum(float(v) for k, v in val.items() if k[1] == lab and k[2] == c and k[3] == "proc" and k[4] == q)
                pr_ = float(r[f"{c}|proc|{q}"]) if f"{c}|proc|{q}" in r.files else 0.0
                if max(po, pr_) > 1e-3 * max(fo, fr, 1e-300):
                    print(f"      process {q:3d}: ours {po:10.4g}   2018 {pr_:10.4g}   ratio {po / pr_ if pr_ else float('inf'):6.3f}")
            moth = {}
            for k, v in val.items():
                if k[1] == lab and k[2] == c and k[3] == "mother":
                    moth[k[4]] = moth.get(k[4], [0.0, 0.0]); moth[k[4]][0] += float(v)
            for x in r.files:
                if x.startswith(f"{c}|mother|"):
                    q = int(x.split("|")[2]); moth[q] = moth.get(q, [0.0, 0.0]); moth[q][1] += float(r[x])
            top = sorted(moth.items(), key=lambda kv: -max(kv[1]))[:10]
            print("      mothers (ours / 2018): " + ", ".join(f"{q}: {u:.3g}/{v_:.3g}" for q, (u, v_) in top))
        print(f"  TOTAL: ours {tot_o:.4g}, 2018 {tot_r:.4g}, ratio {tot_o / tot_r:.3f}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute"); c.add_argument("--src", required=True); c.add_argument("--out", required=True)
    c.add_argument("--workers", type=int, default=120)
    f = sub.add_parser("refs"); f.add_argument("--out", required=True)
    r = sub.add_parser("report"); r.add_argument("ours"); r.add_argument("ref")
    a = ap.parse_args()
    {"compute": compute, "refs": refs, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
