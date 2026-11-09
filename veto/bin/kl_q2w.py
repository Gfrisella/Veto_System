#!/usr/bin/env python3
"""Q^2 and W of the muon-nucleon interactions that make K_L in the MS7 shield chain, per DIS generator.

    python3 bin/kl_q2w.py label=chain.npz:library.npz:split [...] [--out results/shield_dis/kl_q2w]

Generator level (K_L in the library final state, before the iron transport). Each library event of part c
(0 = Pythia "p6" table, 1 = Geant4 iron), grid energy ie, target t gets weight
    R(c, ie) * f_t(ie) / N(c, ie, t)
R = interactions per spill of the chain that used that part and grid energy (int_component, int_E_lib), f_t = the
target's share of the part's cross section in iron (Z/A sigma_p vs (1 - Z/A) sigma_n), N = library events.
Q^2 = M^2 + 2 M nu - W^2 (nucleon at rest). Writes <out>.json, <out>.md and kl_q2w_<label>.png.
"""
import _bootstrap  # noqa: F401
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from fastveto import dis
from fastveto import materials as mat
sys.path.insert(0, "/disk/users/gfrise/.claude-config/skills/cern-ship-slides/theme")
from plotstyle import CMAP, save

M_N = 0.9389
Q2_BINS = [0, 0.1, 1, 10, 1e9]
W_BINS = [0, 2, 3, 5, 10, 1e9]


def events(chain, libpath, split):
    z = np.load(chain); lib = dis.MuonLibrary(libpath, split=split); raw = np.load(libpath)
    c3 = z["int_code"] == 3
    comp, Elib, w = z["int_component"][c3], z["int_E_lib"][c3], z["int_weight"][c3]
    ie_int = np.argmin(np.abs(np.log(Elib)[:, None] - lib.logE[None, :]), axis=1)
    za = mat.MATERIALS["iron"].z_over_a
    out = []
    for c, name in ((0, "p6"), (1, "iron")):
        order, groups = lib.groups[name]
        pre = "p6_" if name == "p6" else "g4_iron_"
        nu_all, w_all = raw[pre + "ev_nu"].astype(float), raw[pre + "ev_w"].astype(float)
        tgt_all, st, nn = raw[pre + "ev_tgt"], raw[pre + "ev_start"], raw[pre + "ev_n"]
        pdg = raw[pre + "pdg"]
        R = np.bincount(ie_int[comp == c], weights=w[comp == c], minlength=len(lib.E))
        for key, (f, cnt) in groups.items():
            t, ie = key // 1000, key % 1000
            if R[ie] <= 0:
                continue
            if name == "p6":
                sp, sn = za * lib.p6_sigma[0, ie], (1 - za) * lib.p6_sigma[1, ie]
                ft = (sp if t == 0 else sn) / (sp + sn) if sp + sn > 0 else 0
            else:
                ft = 1.0
            ev = order[f:f + cnt]
            nkl = np.array([(pdg[st[e]:st[e] + nn[e]] == 130).sum() for e in ev])
            nu, W = nu_all[ev], w_all[ev]
            q2 = M_N ** 2 + 2 * M_N * nu - W ** 2
            out.append(np.stack([np.full(cnt, c), q2, W, np.full(cnt, R[ie] * ft / cnt), nkl], 1))
    return np.concatenate(out)


def table(a):
    comp, q2, W, wt, nkl = a.T
    res = {"interactions_per_spill": float(wt.sum()), "K_L_per_spill_generator": float((wt * nkl).sum())}
    for nm, bins, x in (("Q2", Q2_BINS, q2), ("W", W_BINS, W)):
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (x >= lo) & (x < hi)
            lab = f"{nm} {lo:g}-{hi:g}" if hi < 1e8 else f"{nm} > {lo:g}"
            res[lab] = dict(interactions=float(wt[m].sum() / wt.sum()), K_L=float((wt * nkl)[m].sum() / (wt * nkl).sum()))
    for c, nm in ((0, "Pythia part"), (1, "Geant4 part")):
        m = comp == c
        res[nm] = dict(interactions=float(wt[m].sum() / wt.sum()), K_L=float((wt * nkl)[m].sum() / max((wt * nkl).sum(), 1e-30)))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("items", nargs="+")
    ap.add_argument("--out", default="results/shield_dis/kl_q2w")
    a = ap.parse_args()
    res = {}
    for it in a.items:
        lab, spec = it.split("=", 1); chain, libp, split = spec.split(":")
        arr = events(chain, libp, split)
        res[lab] = table(arr)
        comp, q2, W, wt, nkl = arr.T
        fig, ax = plt.subplots(figsize=(7, 5.5))
        m = nkl > 0
        h = ax.hist2d(np.log10(np.maximum(q2[m], 1e-4)), W[m], bins=[np.linspace(-4, 2.5, 60), np.linspace(0, 20, 60)],
                      weights=(wt * nkl)[m], cmap=CMAP, norm=LogNorm())
        ax.axvline(0, color="k", ls="--", lw=1); ax.axhline(2, color="k", ls=":", lw=1)
        ax.set_xlabel(r"log$_{10}$ $Q^2$ [GeV$^2$]"); ax.set_ylabel("W [GeV]")
        fig.colorbar(h[3], ax=ax, label="K$_L$ per spill per bin (generator level)")
        fig.tight_layout(); save(fig, str(Path(a.out).parent / f"kl_q2w_{lab}.png"))
    json.dump(res, open(a.out + ".json", "w"), indent=1)
    labs = list(res)
    rows = [k for k in res[labs[0]] if isinstance(res[labs[0]][k], dict)]
    L = ["| | " + " | ".join(f"{l}: interactions / K_L" for l in labs) + " |", "|---" * (len(labs) + 1) + "|"]
    L.append("| per spill | " + " | ".join(f"{res[l]['interactions_per_spill']:.3g} / {res[l]['K_L_per_spill_generator']:.3g}" for l in labs) + " |")
    for r in rows:
        L.append(f"| {r} | " + " | ".join(f"{100*res[l][r]['interactions']:.1f} % / {100*res[l][r]['K_L']:.1f} %" for l in labs) + " |")
    open(a.out + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
