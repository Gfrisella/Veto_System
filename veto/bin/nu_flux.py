#!/usr/bin/env python3
"""Neutrino flux from the target + shield production (bin/nu_prod.sbatch): per flavour, energy and component, per proton on
target and per spill (4e13), with errors from the spread between the files; and the cross-check of the soft charm hadron
set (HF_EMIN = 1) against the Nov-4 set (> 10 GeV) above 10 GeV.

Components: min bias (tag nuMB, rows with hf > 0 dropped: charm / beauty decays inside the min-bias cascade are counted by
their own components), charm (charmc), beauty (beautyc). Writes <dir>/nu_flux_summary.npz with (E, theta) histograms per
component and flavour for the next steps (neutrino interactions in the detector).

    python3 bin/nu_flux.py /disk/lhcb_data/sqasim/nu_prod/tail [--had-old /disk/lhcb_data/sqasim/nov4_raw/tail]
"""
import argparse
import glob
import os

import numpy as np

SPILL = 4e13
EB = np.array([1, 2, 5, 10, 20, 50, 100, 400.0])
EH = np.geomspace(1, 400, 41)                       # stored histograms
TH = np.linspace(0, 0.3, 31)
FLAV = {14: "nu_mu", -14: "anti nu_mu", 12: "nu_e", -12: "anti nu_e", 16: "nu_tau", -16: "anti nu_tau"}
COMP = {"min bias": ("nuMB", True), "charm": ("charmc", False), "beauty": ("beautyc", False)}
PARENTS = {"pi": (211,), "K+-": (321,), "K_L": (130,), "K_S": (310,), "hyperons": (3122, 3222, 3112, 3312, 3322, 3334),
           "mu": (13,)}
HAD = {"D0": 421, "D+": 411, "Ds": 431, "Lambda_c": 4122}


def accumulate(d, tag, drop_hf):
    """per file: protons, per (flavour, coarse E bin) sums, per (flavour, parent) sums, fine (E, theta) histograms"""
    files = sorted(glob.glob(f"{d}/{tag}_nu_*.npz"))
    nf, nb = len(FLAV), len(EB) - 1
    prot, coarse, par = [], [], []
    fine = np.zeros((nf, len(EH) - 1, len(TH) - 1))
    zq = []
    for f in files:
        z = np.load(f)
        prot.append(float(z["events"]) * float(z["rep"]))
        m = (z["hf"] == 0) if drop_hf else np.ones(len(z["pdg"]), bool)
        pdg, w = z["pdg"][m], z["w_decay"][m]
        px, py, pz = z["px"][m], z["py"][m], z["pz"][m]
        E = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
        th = np.arctan2(np.hypot(px, py), pz)
        ap = np.abs(z["parent"][m])
        c = np.zeros((nf, nb)); p = np.zeros((nf, len(PARENTS) + 1))
        for i, fl in enumerate(FLAV):
            s = pdg == fl
            c[i] = np.histogram(E[s], EB, weights=w[s])[0]
            fine[i] += np.histogram2d(E[s], th[s], (EH, TH), weights=w[s])[0]
            known = np.zeros(s.sum(), bool)
            for j, ids in enumerate(PARENTS.values()):
                q = np.isin(ap[s], ids); known |= q
                p[i, j] = w[s][q].sum()
            p[i, -1] = w[s][~known].sum()
        coarse.append(c); par.append(p)
        if len(zq) < 50 and len(w):
            zq.append(np.stack([z["z"][m], w], 1))
    return dict(files=len(files), protons=np.array(prot), coarse=np.array(coarse), parents=np.array(par), fine=fine,
                z=np.concatenate(zq) if zq else np.zeros((0, 2)))


def per_proton(a, key):
    """mean per proton over the files (each file normalised by its protons) and its error"""
    x = a[key] / a["protons"][:, None, None]
    n = len(x)
    return x.mean(0), (x.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(x.shape[1:]))


def wquant(z, w, q):
    o = np.argsort(z); c = np.cumsum(w[o]); c /= c[-1]
    return np.interp(q, c, z[o])


def had_check(new_dir, old_dir):
    print("\ncharm hadron cross-check, hadrons with E > 10 GeV, weight per proton (new soft set vs Nov-4 > 10 GeV set)")
    res = {}
    for name, d in (("new", new_dir), ("old", old_dir)):
        files = sorted(glob.glob(f"{d}/charmc_had_*.npz"))
        tot, by, below = [], {k: [] for k in HAD}, []
        for f in files:
            z = np.load(f)
            n = float(z["events"])
            E = np.sqrt(z["px"] ** 2 + z["py"] ** 2 + z["pz"] ** 2)       # |p|; mass 2 GeV is small next to 10 GeV
            hi = E > 10
            tot.append(z["w"][hi].sum() / n)
            below.append(z["w"][~hi].sum() / n)
            for k, v in HAD.items():
                by[k].append(z["w"][hi & (np.abs(z["pdg"]) == v)].sum() / n)
        f = lambda x: (np.mean(x), np.std(x, ddof=1) / np.sqrt(len(x)))
        res[name] = dict(n=len(files), tot=f(tot), below=f(below), by={k: f(v) for k, v in by.items()})
    for name in ("new", "old"):
        r = res[name]
        print(f"  {name}: {r['n']} files, > 10 GeV {r['tot'][0]:.4g} +- {r['tot'][1]:.2g}, <= 10 GeV {r['below'][0]:.4g}")
    (a, ea), (b, eb) = res["new"]["tot"], res["old"]["tot"]
    print(f"  ratio new / old above 10 GeV: {a / b:.4f}, pull {(a - b) / np.hypot(ea, eb):+.2f}")
    for k in HAD:
        (a, ea), (b, eb) = res["new"]["by"][k], res["old"]["by"][k]
        print(f"    {k:9s} new {a:.4g}  old {b:.4g}  ratio {a / max(b, 1e-300):.3f}  pull {(a - b) / max(np.hypot(ea, eb), 1e-300):+.2f}")
    r = res["new"]
    print(f"  soft part (<= 10 GeV) adds x{(r['tot'][0] + r['below'][0]) / r['tot'][0]:.3f} to the charm hadron weight")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--had-old", default=None, help="directory of the Nov-4 charmc_had files (> 10 GeV) for the cross-check")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    acc = {}
    for name, (tag, drop) in COMP.items():
        acc[name] = accumulate(a.dir, tag, drop)
        print(f"{name:9s}: {acc[name]['files']} files, {acc[name]['protons'].sum():.4g} protons")
        if not acc[name]["files"]:
            del acc[name]
    comps = list(acc)
    hdr = "".join(f"{f'{EB[k]:g}-{EB[k + 1]:g}':>11s}" for k in range(len(EB) - 1))
    print(f"\nneutrinos per proton on target (E >= 1 GeV, angle <= 0.3 rad, made upstream of the shield end) by energy [GeV]")
    tot = {}
    for i, (fl, fname) in enumerate(FLAV.items()):
        print(f"\n{fname}")
        print(f"  {'':10s}{hdr}{'total':>12s}{'+-':>9s}{'per spill':>12s}")
        s_all = 0
        for name in comps:
            v, e = per_proton(acc[name], "coarse")
            t = acc[name]["coarse"][:, i].sum(1) / acc[name]["protons"]
            tt, te = t.mean(), (t.std(ddof=1) / np.sqrt(len(t)) if len(t) > 1 else 0)
            tot[(fl, name)] = (tt, te)
            s_all = s_all + v[i]
            print(f"  {name:10s}" + "".join(f"{x:11.3e}" for x in v[i]) + f"{tt:12.4g}{te:9.2g}{tt * SPILL:12.4g}")
        T = sum(tot[(fl, n)][0] for n in comps)
        print(f"  {'sum':10s}" + "".join(f"{x:11.3e}" for x in s_all) + f"{T:12.4g}{'':9s}{T * SPILL:12.4g}")
    print("\nshares of each flavour (per proton): " + ", ".join(
        f"{FLAV[fl]} " + "/".join(f"{tot[(fl, n)][0] / max(sum(tot[(fl, m)][0] for m in comps), 1e-300):.3f}" for n in comps)
        for fl in FLAV) + "  (min bias / charm / beauty)")
    print("\nmin bias by parent, per proton:")
    v, _ = per_proton(acc["min bias"], "parents")
    names = list(PARENTS) + ["other"]
    print(f"  {'':12s}" + "".join(f"{n:>11s}" for n in names))
    for i, fl in enumerate(FLAV):
        print(f"  {FLAV[fl]:12s}" + "".join(f"{x:11.3e}" for x in v[i]))
    print("\nwhere they are made (weighted z quantiles 5 / 50 / 95 %, m, from the first 50 files):")
    for name in comps:
        z = acc[name]["z"]
        if len(z):
            print(f"  {name:9s} " + " / ".join(f"{q:.2f}" for q in wquant(z[:, 0], z[:, 1], [0.05, 0.5, 0.95])))
    out = a.out or os.path.join(a.dir, "nu_flux_summary.npz")
    np.savez(out, E_edges=EH, theta_edges=TH, flavours=np.array(list(FLAV)), components=np.array(comps),
             **{f"hist_{n.replace(' ', '_')}": acc[n]["fine"] / acc[n]["protons"].sum() for n in comps},
             **{f"protons_{n.replace(' ', '_')}": acc[n]["protons"].sum() for n in comps})
    print(f"\n(E, theta) histograms per proton written to {out}")
    if a.had_old:
        had_check(a.dir, a.had_old)


if __name__ == "__main__":
    main()
