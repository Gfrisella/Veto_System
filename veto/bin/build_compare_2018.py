#!/usr/bin/env python3
"""Comparison file closest to the 2018 FairShip production, made in our own chain: Pythia 8 + Geant4 10.00.p04
QGSP_BERT (2018: QGSP_BERT_HP_PEN; HP / PEN only change low-energy neutrons), the 2018 target (--target SHiP2018),
min. bias above 20 GeV (tag fsm_p8_qgsp_t18_g410, run_fsm3.sh / run_fsm5.sh in /disk/lhcb_data/sqasim/fairship_match).
Not in it: dimuon / photon and charm classes (the Geant4 10 build has no dimuon options), K_mu3 / K_L decays.

Rows in the h5's convention: px, py, pz, x, y where the muon leaves the target (z = -0.0045 m), pdg, weight (muons per
spill, 5e13 p.o.t.), kept if they reach the 2018 plane with E > 10 GeV (loss drawn from the 2018 map,
bin/old_convention.py); extra p_prod (momentum at production).

  --compute (farm): <out>/compare_2018_minbias.h5 and <out>/compare_2018.npz, histograms of it, of the h5's and of the
                    2018 reference's min. bias rows (h5 and ours in the momentum after the target, reference and ours
                    in the momentum at production)
  (local)          : python3 bin/build_compare_2018.py results/ref2018/compare_2018.npz -> compare_2018.png
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import os
SHARED = os.environ.get("VETO_SHARED", "/disk/lhcb_data/sqasim")   # Shah Rukh's shared data (read only)

SRC = f"{SHARED}/fairship_match/tail/fsm_p8_qgsp_t18_g410_[0-9]*.npz"
H5 = "/disk/users/gfrise/MuonsAndMatter/data/muons/full_sample_after_target.h5"
REF = f"{SHARED}/reference-2018-fairship-506M/reference_2018.h5"
PE = np.geomspace(10, 400, 33)
TE = np.linspace(0, 4, 21)
MB = 768.75


def hist(p, pt, w):
    i, j = np.digitize(p, PE) - 1, np.digitize(pt, TE) - 1
    ok = (i >= 0) & (i < len(PE) - 1) & (j >= 0) & (j < len(TE) - 1)
    k = i[ok] * (len(TE) - 1) + j[ok]
    n, shape = (len(PE) - 1) * (len(TE) - 1), (len(PE) - 1, len(TE) - 1)
    return (np.bincount(k, weights=w[ok], minlength=n).reshape(shape),
            np.bincount(k, weights=w[ok] ** 2, minlength=n).reshape(shape))


def compute(out):
    import h5py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _bootstrap  # noqa: F401
    import old_convention as oc
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    table = oc.loss_map()
    cols = {k: [] for k in ("px", "py", "pz", "x", "y", "pdg", "weight", "p_prod")}
    events = 0.0
    files = sorted(glob.glob(SRC))
    for fn in files:
        d = np.load(fn)
        events += float(d["events"])
        pp = d["p_prod"].astype(np.float64)
        m = pp - oc.draw_loss(table, pp) > 10.0                      # the 2018 selection
        for k in ("px", "py", "pz", "x", "y", "pdg"):
            cols[k].append(d[k][m].astype(np.float32))
        cols["weight"].append(d["w_decay"][m].astype(np.float64) / float(d["rep"]))
        cols["p_prod"].append(pp[m].astype(np.float32))
    c = {k: np.concatenate(v) for k, v in cols.items()}
    c["weight"] = (c["weight"] * 5e13 / events).astype(np.float32)
    with h5py.File(out / "compare_2018_minbias.h5.tmp", "w") as g:
        for k, v in c.items():
            g.create_dataset(k, data=v, chunks=True)
        g.create_dataset("z", data=np.full(len(c["px"]), -0.0045, np.float32), chunks=True)
        g.attrs["description"] = ("2018-like min. bias (Pythia 8 + Geant4 10.00.p04 QGSP_BERT, 2018 target, >= 20 GeV), "
                                  "momentum after the target, E > 10 GeV at the 2018 plane; bin/build_compare_2018.py")
        g.attrs["pot"] = events
    (out / "compare_2018_minbias.h5.tmp").rename(out / "compare_2018_minbias.h5")
    print(f"{len(files)} slices, {events:.4g} p.o.t., {len(c['px']):,} muons -> {out / 'compare_2018_minbias.h5'}", flush=True)
    res = dict(PE=PE, TE=TE, pot=events)
    pt = np.hypot(c["px"].astype(np.float64), c["py"])
    p = np.sqrt(pt ** 2 + c["pz"].astype(np.float64) ** 2)
    w = c["weight"].astype(np.float64)
    res["ours_exit_s1"], res["ours_exit_s2"] = hist(p, pt, w)
    pp = c["p_prod"].astype(np.float64)
    res["ours_prod_s1"], res["ours_prod_s2"] = hist(pp, pt * pp / p, w)
    del c, pt, p, pp, w
    for tag, path, wcol in (("h5", H5, "weight"), ("ref", REF, "w_fairship")):
        with h5py.File(path, "r") as f:
            m = np.abs(f[wcol][:] - MB) < 1
            px, py, pz = (f[k][:][m].astype(np.float64) for k in ("px", "py", "pz"))
            w = f["weight"][:][m].astype(np.float64)
        pt = np.hypot(px, py)
        res[f"{tag}_s1"], res[f"{tag}_s2"] = hist(np.sqrt(pt ** 2 + pz ** 2), pt, w)
        del px, py, pz, w, pt, m
    np.savez(out / "compare_2018.npz", **res)
    print("wrote", out / "compare_2018.npz")


def plot(npz):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = np.load(npz)
    pc, tc = np.sqrt(PE[1:] * PE[:-1]), 0.5 * (TE[1:] + TE[:-1])
    fig, axs = plt.subplots(2, 3, figsize=(20, 10), gridspec_kw=dict(height_ratios=[3, 1.6]))
    panels = ((0, "momentum AFTER the target: ours vs the 601 M h5", "h5", "ours_exit", "601 M h5", "#eb6834"),
              (1, "momentum at PRODUCTION: ours vs the 2018 FairShip files", "ref", "ours_prod", "2018 FairShip (67 files)", "#222222"))
    for j, title, base, ours, lab, col in panels:
        ax, bx = axs[0, j], axs[1, j]
        b1, b2 = h[f"{base}_s1"].sum(1), h[f"{base}_s2"].sum(1)
        o1, o2 = h[f"{ours}_s1"].sum(1), h[f"{ours}_s2"].sum(1)
        for s1, s2, c_, l_ in ((b1, b2, col, lab), (o1, o2, "#2a78d6", "ours: Pythia 8 + Geant4 10.0 QGSP_BERT, 2018 target")):
            ax.errorbar(pc, np.where(s1 > 0, s1 / np.diff(PE), np.nan), yerr=np.sqrt(s2) / np.diff(PE), color=c_, lw=2,
                        ds="steps-mid", label=l_)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where((b1 > 0) & (o1 > 0), o1 / b1, np.nan)
            e = r * np.sqrt(o2 / o1 ** 2 + b2 / b1 ** 2)
        bx.errorbar(pc, r, yerr=e, color="#2a78d6", marker="o", ms=4, lw=2)
        bx.axhline(1, color="k", lw=1); bx.set_ylim(0.5, 1.5); bx.set_xscale("log"); bx.set_ylabel(f"ours / {base}")
        bx.axvspan(10, 30, color="0.9", zorder=0); bx.text(11, 1.4, "ours starts at 20 GeV\n(production)", fontsize=8)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(title); ax.set_ylabel("min. bias muons per spill per GeV")
        ax.legend(fontsize=9); bx.set_xlabel("p [GeV]")
    ax = axs[0, 2]
    m = (pc >= 30) & (pc < 150)
    for k, ls, l_ in (("h5", "--", "601 M h5 (after target)"), ("ours_exit", "-", "ours (after target)"),
                      ("ref", ":", "2018 FairShip (production)"), ("ours_prod", "-.", "ours (production)")):
        s = h[f"{k}_s1"][m].sum(0)
        ax.step(tc, s / s.sum(), where="mid", ls=ls, lw=2, label=l_)
    ax.set_yscale("log"); ax.set_xlabel("pT [GeV]"); ax.set_ylabel("fraction"); ax.set_title("pT shape, p = 30-150 GeV")
    ax.legend(fontsize=9)
    axs[1, 2].axis("off")
    for a in axs.flat:
        a.grid(alpha=0.3)
    fig.suptitle(f"2018-like min. bias in our chain ({float(h['pot']):.3g} p.o.t.) against the h5 and the 2018 FairShip files",
                 fontsize=13)
    fig.tight_layout()
    o = Path(npz).with_name("compare_2018.png")
    fig.savefig(o, dpi=95)
    print(o)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", nargs="?")
    ap.add_argument("--compute", action="store_true")
    ap.add_argument("--out", default=f"{SHARED}/compare2018")
    a = ap.parse_args()
    compute(a.out) if a.compute else plot(a.npz)


if __name__ == "__main__":
    main()
