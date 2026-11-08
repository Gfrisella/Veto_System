#!/usr/bin/env python3
"""The 2018 FairShip reference (bin/build_reference_2018.py) against the h5 (MuonsAndMatter's full_sample_after_target)
and our own productions, per muon class, in the h5's convention (momentum at production, reached the 2018 plane with
E > 10 GeV; ours: the selection emulated with the 2018 loss map, bin/old_convention.py), muons per spill.

  --compute (farm): histograms -> <out>/reference_comparison.npz
  (local)          : python3 bin/plot_reference_comparison.py results/ref2018/reference_comparison.npz

Ours: min. bias = pool_mb20 (30 - 150 GeV, Pythia 6 + Geant4 11, --emin 20) + pool_tail (>= 150 GeV); dimuon =
pool_dmhigh (>= 150 GeV); charm = charmc (>= 20 GeV) — the fresh productions of input pool v1-v3, raw (no reweighting).
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

PE = np.geomspace(10, 400, 33)
TE = np.linspace(0, 4, 21)
CLASSES = {"minbias": 768.75, "dimuon": 7.6875, "charm": 326.16}
OURS = {"minbias": [("pool_mb20", (30, 150)), ("pool_tail", (150, 1e9))], "dimuon": [("pool_dmhigh", (150, 1e9))],
        "charm": [("charmc", (20, 1e9))]}
SRC = "/disk/lhcb_data/sqasim/veto_source_pool1/tail"


def hist(p, pt, q, w):
    i, j = np.digitize(p, PE) - 1, np.digitize(pt, TE) - 1
    ok = (i >= 0) & (i < len(PE) - 1) & (j >= 0) & (j < len(TE) - 1)
    k = (q[ok] * (len(PE) - 1) + i[ok]) * (len(TE) - 1) + j[ok]
    n = 2 * (len(PE) - 1) * (len(TE) - 1)
    return (np.bincount(k, weights=w[ok], minlength=n).reshape(2, len(PE) - 1, len(TE) - 1),
            np.bincount(k, weights=w[ok] ** 2, minlength=n).reshape(2, len(PE) - 1, len(TE) - 1))


def compute(out):
    import h5py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _bootstrap  # noqa: F401
    import old_convention as oc
    res = {}
    for tag, path, wcol in (("ref", "/disk/lhcb_data/sqasim/reference-2018-fairship-506M/reference_2018.h5", "w_fairship"),
                            ("h5", "/disk/users/gfrise/MuonsAndMatter/data/muons/full_sample_after_target.h5", "weight")):
        with h5py.File(path, "r") as f:
            wcls = f[wcol][:]
            w = f["weight"][:].astype(np.float64)
            px, py, pz = (f[k][:].astype(np.float64) for k in ("px", "py", "pz"))
            q = (f["pdg"][:] > 0).astype(np.int64)
        pt = np.hypot(px, py); p = np.sqrt(pt ** 2 + pz ** 2)
        for c, v in CLASSES.items():
            m = np.abs(wcls - v) < 0.01 * v
            res[f"{tag}_{c}_s1"], res[f"{tag}_{c}_s2"] = hist(p[m], pt[m], q[m], w[m])
        del w, px, py, pz, q, pt, p, wcls
    table = oc.loss_map()
    mean_loss = None
    for c, parts in OURS.items():
        s1 = s2 = 0
        for tag, (lo, hi) in parts:
            files = sorted(glob.glob(f"{SRC}/{tag}_[0-9]*.npz"))[:60]
            ev = 0.0
            a1 = a2 = 0
            for fn in files:
                d = np.load(fn)
                ev += float(d["events"])
                pl = np.sqrt(d["px"].astype(np.float64) ** 2 + d["py"] ** 2 + d["pz"] ** 2)
                if tag == "charmc":
                    if mean_loss is None:
                        mb = [np.load(x) for x in sorted(glob.glob(f"{SRC}/pool_mb20_[0-9]*.npz"))[:5]]
                        pm = np.concatenate([np.sqrt(x["px"].astype(np.float64) ** 2 + x["py"] ** 2 + x["pz"] ** 2) for x in mb])
                        pp_ = np.concatenate([x["p_prod"] for x in mb])
                        bins = np.geomspace(20, 400, 25); kk = np.digitize(pm, bins)
                        ml = np.array([np.median((pp_ - pm)[kk == i]) if np.sum(kk == i) > 50 else np.nan for i in range(1, len(bins))])
                        g = np.isfinite(ml)
                        mean_loss = lambda x, b=bins, ml=ml, g=g: np.interp(x, np.sqrt(b[1:] * b[:-1])[g], ml[g])
                    pp = pl + mean_loss(pl)
                else:
                    pp = d["p_prod"].astype(np.float64)
                m = (pp >= lo) & (pp < hi)
                m[m] &= pp[m] - oc.draw_loss(table, pp[m]) > 10.0
                s = pp / pl
                pt = np.hypot(d["px"], d["py"]) * s
                w = d["w_decay"].astype(np.float64) / float(d["rep"])
                b1, b2 = hist(pp[m], pt[m], (d["pdg"][m] < 0).astype(np.int64) if False else (d["pdg"][m] > 0).astype(np.int64), w[m])
                a1, a2 = a1 + b1, a2 + b2
            norm = 5e13 / ev
            s1, s2 = s1 + a1 * norm, s2 + a2 * norm ** 2
        res[f"ours_{c}_s1"], res[f"ours_{c}_s2"] = s1, s2
    np.savez(out, PE=PE, TE=TE, **res)
    print("wrote", out)


def plot(npz):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = np.load(npz)
    pc, tc = np.sqrt(PE[1:] * PE[:-1]), 0.5 * (TE[1:] + TE[:-1])
    COL = {"ref": "#222222", "h5": "#eb6834", "ours": "#2a78d6"}
    LAB = {"ref": "2018 FairShip, all 67 files (reference)", "h5": "601 M h5 (what our datasets used)", "ours": "our production (Pythia + Geant4)"}
    TIT = {"minbias": "minimum bias (weight 768.75)", "dimuon": "dimuon / photon class (weight 7.69)", "charm": "charm (weight 326.16)"}
    fig, axs = plt.subplots(3, 4, figsize=(24, 15), gridspec_kw=dict(height_ratios=[3, 1.6, 3]))
    for j, c in enumerate(CLASSES):
        ax, bx, cx = axs[0, j], axs[1, j], axs[2, j]
        ref = h[f"ref_{c}_s1"].sum((0, 2))
        for k in ("ref", "h5", "ours"):
            s1, s2 = h[f"{k}_{c}_s1"].sum((0, 2)), h[f"{k}_{c}_s2"].sum((0, 2))
            y = np.where(s1 > 0, s1 / np.diff(PE), np.nan)
            ax.errorbar(pc, y, yerr=np.sqrt(s2) / np.diff(PE), color=COL[k], lw=2, ds="steps-mid", label=LAB[k])
            if k != "ref":
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = np.where((ref > 0) & (s1 > 0), s1 / ref, np.nan)
                    e = r * np.sqrt(s2 / s1 ** 2 + h[f"ref_{c}_s2"].sum((0, 2)) / ref ** 2)
                bx.errorbar(pc, r, yerr=e, color=COL[k], lw=2, marker="o", ms=4, label=f"{k} / reference")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(TIT[c]); ax.set_ylabel("muons per spill per GeV"); ax.legend(fontsize=8)
        bx.set_xscale("log"); bx.axhline(1, color="k", lw=1); bx.set_ylim(0.4, 1.8); bx.set_ylabel("ratio to the reference"); bx.legend(fontsize=8)
        bx.set_xlabel("momentum at production [GeV]")
        # pT shape for 30 - 150 GeV, by charge
        m = (pc >= 30) & (pc < 150)
        for k, ls in (("ref", "-"), ("h5", "--"), ("ours", ":")):
            for qi, qn, col in ((0, "mu-", "#2a78d6"), (1, "mu+", "#eb6834")):
                s1 = h[f"{k}_{c}_s1"][qi][m].sum(0)
                if s1.sum() > 0:
                    cx.step(tc, s1, where="mid", color=col, ls=ls, lw=2, label=f"{qn}, {k}")
        cx.set_yscale("log"); cx.set_xlabel("pT [GeV]"); cx.set_ylabel("muons per spill (30-150 GeV)")
        cx.set_title("pT and charge, p = 30-150 GeV (solid: reference, dashed: h5, dotted: ours)"); cx.legend(fontsize=7, ncol=3)
    # summary panel: ratios of totals per class and charge
    ax = axs[0, 3]
    rows = []
    for c in CLASSES:
        for qi, qn in ((0, "mu-"), (1, "mu+")):
            ref = h[f"ref_{c}_s1"][qi].sum()
            rows.append((f"{c} {qn}", h[f"h5_{c}_s1"][qi].sum() / ref, h[f"ours_{c}_s1"][qi].sum() / ref if h[f"ours_{c}_s1"][qi].sum() > 0 else np.nan))
    yy = np.arange(len(rows))
    ax.barh(yy - 0.2, [r[1] for r in rows], 0.4, color=COL["h5"], label="h5 / reference")
    ax.barh(yy + 0.2, [r[2] for r in rows], 0.4, color=COL["ours"], label="ours / reference")
    ax.set_yticks(yy, [r[0] for r in rows]); ax.axvline(1, color="k"); ax.set_title("total flux per class and charge / reference\n(ours: its momentum range only)")
    ax.legend(fontsize=8)
    axs[1, 3].axis("off"); axs[2, 3].axis("off")
    for a in axs.flat:
        a.grid(alpha=0.3)
    fig.suptitle("Muons after the 2018 target + absorber (E > 10 GeV at the 2018 plane): FairShip 2018 reference vs the 601 M h5 vs our production",
                 fontsize=13)
    fig.tight_layout()
    out = Path(npz).with_name("reference_comparison.png")
    fig.savefig(out, dpi=95)
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="?")
    ap.add_argument("--compute", action="store_true")
    ap.add_argument("--out", default="/disk/lhcb_data/sqasim/reference-2018-fairship-506M/reference_comparison.npz")
    a = ap.parse_args()
    compute(a.out) if a.compute else plot(a.npz)


if __name__ == "__main__":
    main()
