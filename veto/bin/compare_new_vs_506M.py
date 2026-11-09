#!/usr/bin/env python3
"""The same histograms for the complete target production (target-production-complete-november-2-2026) and the 506 M
2018 FairShip sample (reference-2018-fairship-506M), for the target talk and its comparison plots.

Both in muons per spill (--pot, default 4e13 p.o.t.; the 506 M file's weights, per 5e13, are rescaled), production momentum (the 506 M sample stores it; ours keeps it per muon
as pcx, pcy, pcz), with the 2018 selection on both sides: p >= P_MIN and, for ours, the 2018 plane emulated (E > 10 GeV
after the 2018 losses, bin/old_convention.py). Per component (min. bias incl. dimuon / photon sources, onia, charm,
beauty), per charge, per parent; sums of w, w^2 and rows per bin (statistics). Ours also with FairShip's 2018 weighting
emulated per muon (compare_inventory_2018.classify), to separate physics from that.

    python3 bin/compare_new_vs_506M.py new --src <dataset>/production --out new_hists.npz --workers 120   (eiger)
    python3 bin/compare_new_vs_506M.py ref --h5 <506M dir>/reference_2018.h5 --out ref_hists.npz          (farm)
    python3 bin/compare_new_vs_506M.py all --src <dataset>/production --out new_all.npz                   (eiger: no
            selection, after-target and production momentum, rows and n_eff: the sample itself)
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

P_MIN = 25.0
PE = np.geomspace(10, 400, 33)
TE = np.linspace(0, 4, 21)
COMPS = ("minbias", "onia", "charm", "beauty")
QQ = [443, 100443, 553, 100553, 200553]
CHARM = [411, 421, 431, 4122, 4132, 4232, 4332]
GROUPS = {"pi": [211], "K": [321, 130, 310], "eta/omega/rho/phi/eta'": [221, 223, 113, 333, 331], "gamma": [22],
          "quarkonia": QQ, "charm": CHARM + [15], "beauty": [511, 521, 531, 5122], "other": []}


def group_of(par):
    a = np.abs(par)
    g = np.full(len(a), len(GROUPS) - 1)
    for i, (k, v) in enumerate(GROUPS.items()):
        if v:
            g[np.isin(a, v)] = i
    return g


def hists(p, pt, w, q, grp):
    """1D p, 1D pT (p >= P_MIN), 2D, per charge; per parent group: flux and w2."""
    out = {}
    for ch, m0 in (("mu-", q > 0), ("mu+", q < 0)):          # pdg 13 = mu-
        for k, wk in (("s1", w), ("s2", w * w), ("n", np.ones_like(w))):
            out[f"{ch}_2d_{k}"] = np.histogram2d(p[m0], pt[m0], [PE, TE], weights=wk[m0])[0]
    sel = p >= P_MIN
    for i, k in enumerate(GROUPS):
        m = sel & (grp == i)
        out[f"grp_{i}_s1"] = np.array([w[m].sum()])
        out[f"grp_{i}_s2"] = np.array([(w[m] ** 2).sum()])
        out[f"grp_{i}_n"] = np.array([m.sum()])
    return out


def one_file(args):
    fn, comp, mode = args
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _bootstrap  # noqa: F401
    d = np.load(fn)
    gen, par, hf = d["gen"], d["parent"], d["hf"]
    w = d["w_decay"].astype(np.float64) / float(d["rep"])
    keep = np.ones(len(w), bool)
    if comp == "minbias":
        keep = (hf == 0) & ~(np.isin(gen, [2, 7]) & np.isin(np.abs(par), QQ))
    q = d["pdg"]
    out = {}
    if mode == "all":                                       # the sample itself, no selection
        pa = np.sqrt(d["px"] ** 2 + d["py"] ** 2 + d["pz"] ** 2).astype(np.float64)
        pta = np.hypot(d["px"], d["py"]).astype(np.float64)
        for k, wk in (("s1", w), ("s2", w * w), ("n", np.ones_like(w))):
            out[f"exit_{k}"] = np.histogram2d(pa[keep], pta[keep], [PE, TE], weights=wk[keep])[0]
        return comp, float(d["events"]), out
    import old_convention as oc
    from compare_inventory_2018 import classify
    pp = d["p_prod"].astype(np.float64)
    pt = np.hypot(d["pcx"], d["pcy"]).astype(np.float64)
    sel = keep & (pp - oc.draw_loss(oc.loss_map(), pp) > 10.0)
    _, k18 = classify({"minbias": "mb"}.get(comp, comp), gen, par, d["ch"], hf)
    grp = group_of(par)
    for lab, ww in (("phys", w), ("w2018", w * k18)):
        for k, v in hists(pp[sel], pt[sel], ww[sel], q[sel], grp[sel]).items():
            out[f"{lab}_{k}"] = v
    return comp, float(d["events"]), out


def new(a, mode):
    from multiprocessing import Pool
    tags = dict(zip(COMPS, a.tags.split(",")))
    jobs = [(f, c, mode) for c in COMPS for f in sorted(glob.glob(f"{a.src}/{tags[c]}_[0-9]*.npz"))]
    missing = [tags[c] for c in COMPS if not any(j[1] == c for j in jobs)]
    if missing:                                       # a wrong tag would silently drop a whole component
        raise SystemExit(f"no files for {missing} in {a.src}")
    ev, acc = {c: 0.0 for c in COMPS}, {}
    with Pool(a.workers) as pool:
        for i, (comp, e, out) in enumerate(pool.imap_unordered(one_file, jobs, chunksize=2)):
            ev[comp] += e
            for k, v in out.items():
                acc[(comp, k)] = acc.get((comp, k), 0) + v
            if i % 500 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
    res = {"events": json.dumps(ev), "groups": json.dumps(list(GROUPS)), "PE": PE, "TE": TE, "p_min": P_MIN, "pot": a.pot}
    for (comp, k), v in acc.items():                         # per proton -> per spill
        f = a.pot / ev[comp]
        res[f"{comp}|{k}"] = np.asarray(v) * (f * f if k.endswith("_s2") else (1.0 if k.endswith("_n") else f))
    np.savez_compressed(a.out, **res)
    print({c: len([j for j in jobs if j[1] == c]) for c in COMPS}, "files ->", a.out)


def ref(a):
    import h5py
    CLS = {"minbias": (768.75, 7.6875), "charm": (326.16, 3.2616), "beauty": (9.37, 0.0937)}
    acc = {}
    with h5py.File(a.h5, "r") as f:
        n = len(f["weight"])
        for i0 in range(0, n, 20_000_000):
            sl = slice(i0, min(n, i0 + 20_000_000))
            px, py, pz = (f[k][sl].astype(np.float64) for k in ("px", "py", "pz"))
            w, wf = f["weight"][sl].astype(np.float64) * (a.pot / 5e13), f["w_fairship"][sl]   # file: per 5e13 p.o.t.
            mo, q = f["mother"][sl], f["pdg"][sl]
            pt = np.hypot(px, py); p = np.sqrt(pt ** 2 + pz ** 2)
            grp = group_of(mo)
            for comp, vals in CLS.items():
                m = np.zeros(len(w), bool)
                for v in vals:
                    m |= np.abs(wf - v) < 0.01 * v
                if comp == "minbias":                         # quarkonia: the onia component, as ours
                    for c2, mm in (("minbias", m & ~np.isin(np.abs(mo), QQ)), ("onia", m & np.isin(np.abs(mo), QQ))):
                        for k, v in hists(p[mm], pt[mm], w[mm], q[mm], grp[mm]).items():
                            acc[f"{c2}|ref_{k}"] = acc.get(f"{c2}|ref_{k}", 0) + v
                else:
                    for k, v in hists(p[m], pt[m], w[m], q[m], grp[m]).items():
                        acc[f"{comp}|ref_{k}"] = acc.get(f"{comp}|ref_{k}", 0) + v
            print(f"{sl.stop:,}/{n:,}", flush=True)
        attrs = dict(f.attrs)
    np.savez_compressed(a.out, PE=PE, TE=TE, p_min=P_MIN, groups=json.dumps(list(GROUPS)), pot=a.pot, pot_file=float(attrs.get("pot", 0)),
                        **{k: np.asarray(v) for k, v in acc.items()})
    print("wrote", a.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("new", "all"):
        s = sub.add_parser(name); s.add_argument("--src", required=True); s.add_argument("--out", required=True)
        s.add_argument("--workers", type=int, default=120)
        s.add_argument("--tags", default="minbias,onia,charmc_mu,beautyc_mu", help="file tags of min. bias, onia, charm, beauty")
        s.add_argument("--pot", type=float, default=4e13, help="protons on target per spill")
    r = sub.add_parser("ref"); r.add_argument("--h5", required=True); r.add_argument("--out", required=True)
    r.add_argument("--pot", type=float, default=4e13, help="protons on target per spill")
    a = ap.parse_args()
    if a.cmd == "ref":
        ref(a)
    else:
        new(a, a.cmd)


if __name__ == "__main__":
    main()
