#!/usr/bin/env python3
"""Validation of tailtransport's neutrino mode: the forced (complete) sample against the unforced (analog) one, per
flavour, energy and parent, per proton on target; and a loose comparison with the 2018 FairShip flux (fastveto.neutrinos:
only neutrinos with a stored sister muon, p > 10 GeV, no nu_e: a lower bound for nu_mu above 10 GeV).

    python3 bin/nu_validate.py <tail dir> [--analog nuA --forced nuF] [--fairship2018]
"""
import _bootstrap  # noqa: F401
import argparse
import glob

import numpy as np

EB = np.array([1, 2, 5, 10, 20, 50, 100, 400.0])
FLAV = {14: "nu_mu", -14: "anti nu_mu", 12: "nu_e", -12: "anti nu_e", 16: "nu_tau", -16: "anti nu_tau"}
PARENTS = {"pi": (211,), "K+-": (321,), "K_L": (130,), "K_S": (310,), "hyperons": (3122, 3222, 3112, 3312, 3322, 3334)}


def load(d, tag):
    files = sorted(glob.glob(f"{d}/{tag}_nu_*.npz"))
    cols = {k: [] for k in ("pdg", "px", "py", "pz", "w_decay", "parent", "proc", "file")}
    protons = 0.0
    for j, f in enumerate(files):
        z = np.load(f)
        protons += float(z["events"]) * float(z["rep"])
        for k in [k for k in cols if k != "file"]:
            cols[k].append(z[k])
        cols["file"].append(np.full(len(z["w_decay"]), j))
    c = {k: np.concatenate(v) for k, v in cols.items()}
    c["nfiles"] = len(files)
    c["E"] = np.sqrt(c["px"] ** 2 + c["py"] ** 2 + c["pz"] ** 2)
    c["apar"] = np.abs(c["parent"])
    return c, protons, len(files)


def table(c, n, sel):
    """per proton (value, error) in the energy bins for rows sel; error from the spread between the files (independent
    slices of equal size: rows of one event are correlated)"""
    nb, nf = len(EB) - 1, c["nfiles"]
    i = np.digitize(c["E"][sel], EB) - 1
    w, f = c["w_decay"][sel], c["file"][sel]
    ok = (i >= 0) & (i < nb)
    per = np.bincount(f[ok] * nb + i[ok], w[ok], nf * nb).reshape(nf, nb) / (n / nf)
    return per.mean(0), per.std(0, ddof=1) / np.sqrt(nf)


def total(c, n, sel):
    nf = c["nfiles"]
    per = np.bincount(c["file"][sel], c["w_decay"][sel], nf) / (n / nf)
    return per.mean(), per.std(ddof=1) / np.sqrt(nf)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir"); ap.add_argument("--analog", default="nuA"); ap.add_argument("--forced", default="nuF")
    ap.add_argument("--fairship2018", action="store_true")
    a = ap.parse_args()
    A, nA, fA = load(a.dir, a.analog)
    F, nF, fF = load(a.dir, a.forced)
    print(f"analog: {fA} files, {nA:.4g} protons, {len(A['E'])} rows; forced: {fF} files, {nF:.4g} protons, {len(F['E'])} rows")
    print(f"neutrinos per proton (E >= 1 GeV, angle <= 0.3 rad) by energy bin [GeV]; forced / analog, pull = (F - A) / sigma")
    hdr = "".join(f"{f'{EB[k]:g}-{EB[k + 1]:g}':>17s}" for k in range(len(EB) - 1))
    tot_pulls = []
    for f, name in FLAV.items():
        sa, ea = table(A, nA, A["pdg"] == f)
        sf, ef = table(F, nF, F["pdg"] == f)
        if sa.sum() == 0 and sf.sum() == 0:
            continue
        (ta, ta_e), (tf, tf_e) = total(A, nA, A["pdg"] == f), total(F, nF, F["pdg"] == f)
        print(f"\n{name:12s} total: analog {ta:.4g} +- {ta_e:.2g}, forced {tf:.4g} +- {tf_e:.2g}  (ratio {tf / max(ta, 1e-300):.3f}, "
              f"pull {(tf - ta) / max(np.hypot(ta_e, tf_e), 1e-300):+.2f})")
        print(f"{'':12s}" + hdr)
        print(f"{'  analog':12s}" + "".join(f"{v:17.3e}" for v in sa))
        print(f"{'  forced':12s}" + "".join(f"{v:17.3e}" for v in sf))
        pull = np.where(np.hypot(ea, ef) > 0, (sf - sa) / np.where(np.hypot(ea, ef) > 0, np.hypot(ea, ef), 1), 0)
        print(f"{'  pull':12s}" + "".join(f"{v:17.2f}" for v in pull))
        tot_pulls += [p for p, x, y in zip(pull, sa, sf) if x > 0 and y > 0]
    print("\nby parent (all flavours, E >= 1 GeV): per proton analog / forced")
    for name, ids in PARENTS.items():
        (va, ea), (vf, ef) = total(A, nA, np.isin(A["apar"], ids)), total(F, nF, np.isin(F["apar"], ids))
        print(f"  {name:9s} {va:.4g} +- {ea:.2g}   {vf:.4g} +- {ef:.2g}   ratio {vf / max(va, 1e-300):.3f}  pull {(vf - va) / max(np.hypot(ea, ef), 1e-300):+.2f}")
    other_a = ~np.isin(A["apar"], sum(PARENTS.values(), ())); other_f = ~np.isin(F["apar"], sum(PARENTS.values(), ()))
    print(f"  {'other':9s} {A['w_decay'][other_a].sum() / nA:.4g}   {F['w_decay'][other_f].sum() / nF:.4g}  (parents: analog "
          f"{dict(zip(*np.unique(A['parent'][other_a], return_counts=True)))})")
    if tot_pulls:
        p = np.array(tot_pulls)
        print(f"\npulls of the {len(p)} filled (flavour, energy) cells: mean {p.mean():+.2f}, rms {p.std():.2f}, max |pull| {np.abs(p).max():.2f}")
    if a.fairship2018:
        from fastveto import neutrinos
        d = neutrinos.load()
        th = np.arctan2(np.hypot(d["px"], d["py"]), d["pz"])
        for f in (14, -14):
            m = (d["pdg"] == f) & (th <= 0.3) & (d["p"] >= 10)
            v18 = d["w"][m].sum() / neutrinos.POT_PER_SPILL if "w" in d else d["mu_w"][m].sum() / neutrinos.POT_PER_SPILL
            mf = (F["pdg"] == f) & (F["E"] >= 10)
            print(f"2018 FairShip {FLAV[f]:11s} E >= 10 GeV, angle <= 0.3: {v18:.4g} per proton (only with a stored sister muon); "
                  f"ours (forced) {F['w_decay'][mf].sum() / nF:.4g}")


if __name__ == "__main__":
    main()
