"""Old input muon file vs our FairShip-matched regeneration: histograms per class and in total.

    python3 bin/plot_reproduction.py [--out results/reproduction_v2] [--tags mb2,tail2,dmlow2,dmhigh2,charmc]

Classes of the old file by weight: minimum bias 768.75, dimuon 7.6875, charm 326.16. Ours (weights in muons per
spill at the same plane, z = -0.0045 m): minimum bias = MB below P_SPLIT + TAIL above; dimuon = DMLOW below +
DMHIGH above; charm = CHARM (first collision + cascade). Plots (one PNG per class + total):
  row 1: momentum spectrum (log-log) and ratio ours / old (with errors); mu+ fraction vs momentum
  row 2: theta_x, theta_y, x, y for p > 20 GeV          row 3: the same for p > 150 GeV
Also summary.json with the per-bin ratios and integrated fluxes above 20 / 150 GeV.
"""

import _bootstrap  # noqa: F401
import argparse
import glob
import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from veto import paths

INPUT = str(paths.MUONS_AND_MATTER / "data/muons/full_sample_after_target.h5")
P_SPLIT = 150.0
PB = np.geomspace(10, 400, 33)


def load(tag):
    files = sorted(glob.glob(str(paths.source_data() / "tail" / f"{tag}_[0-9]*.npz")))
    if not files:
        return None
    parts = [dict(np.load(f)) for f in files]
    ev = sum(int(p["events"]) for p in parts)
    d = {k: np.concatenate([p[k] for p in parts]) for k in ("px", "py", "pz", "x", "y", "pdg")}
    d["w"] = np.concatenate([p["w_decay"] / p["rep"] for p in parts]) * 5e13 / ev
    d["p"] = np.sqrt(d["px"] ** 2 + d["py"] ** 2 + d["pz"] ** 2)
    print(f"  {tag}: {len(files)} slices, {ev:.3g} protons, {len(d['w']):,} muons")
    return d


def merge(lo, hi):
    if hi is None:
        return lo
    a, b = lo["p"] < P_SPLIT, hi["p"] >= P_SPLIT
    return {k: np.concatenate([lo[k][a], hi[k][b]]) for k in lo}


def derived(d):
    d["tx"] = 1e3 * d["px"] / d["pz"]
    d["ty"] = 1e3 * d["py"] / d["pz"]
    d["xc"] = 100 * d["x"]
    d["yc"] = 100 * d["y"]
    return d


def panel_hist(ax, old, new, key, bins, sel_o, sel_n, xlabel, logy=False):
    ho, _ = np.histogram(old[key][sel_o], bins, weights=old["w"][sel_o])
    hn, _ = np.histogram(new[key][sel_n], bins, weights=new["w"][sel_n])
    ax.stairs(ho, bins, color="k", lw=1.8, label=f"old {ho.sum():.3g}/spill")
    ax.stairs(hn, bins, color="crimson", lw=1.8, label=f"ours {hn.sum():.3g}/spill")
    ax.set_xlabel(xlabel)
    if logy:
        ax.set_yscale("log")
    ax.legend(fontsize=7)


def plot_class(name, old, new, out):
    fig, axs = plt.subplots(3, 4, figsize=(24, 15))
    ho, _ = np.histogram(old["p"], PB, weights=old["w"])
    no, _ = np.histogram(old["p"], PB)
    hn, _ = np.histogram(new["p"], PB, weights=new["w"])
    hn2, _ = np.histogram(new["p"], PB, weights=new["w"] ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = hn / ho
        e = r * np.hypot(1 / np.sqrt(np.maximum(no, 1)), np.sqrt(hn2) / np.maximum(hn, 1e-30))
    cen = np.sqrt(PB[1:] * PB[:-1])
    ax = axs[0, 0]
    ax.stairs(ho, PB, color="k", lw=2, label="old file")
    ax.stairs(hn, PB, color="crimson", lw=2, label="ours")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("p [GeV]"); ax.set_ylabel("muons / spill / bin")
    ax.set_title(name); ax.legend()
    ax = axs[0, 1]
    ax.errorbar(cen, r, yerr=e, fmt="o", color="crimson", ms=4)
    ax.axhline(1, color="k", lw=0.8); ax.axhspan(0.8, 1.2, color="grey", alpha=0.15)
    ax.set_xscale("log"); ax.set_ylim(0, 2); ax.set_xlabel("p [GeV]"); ax.set_ylabel("ours / old")
    ax.set_title("ratio (band: +-20 %)")
    ax = axs[0, 2]
    for d, c, lab in ((old, "k", "old"), (new, "crimson", "ours")):
        hp, _ = np.histogram(d["p"], PB, weights=d["w"] * (d["pdg"] == -13))
        ht, _ = np.histogram(d["p"], PB, weights=d["w"])
        with np.errstate(invalid="ignore", divide="ignore"):
            ax.plot(cen, hp / ht, "o-", color=c, ms=3, label=lab)
    ax.set_xscale("log"); ax.set_ylim(0.3, 1.0); ax.set_xlabel("p [GeV]"); ax.set_ylabel("mu+ fraction"); ax.legend()
    ax = axs[0, 3]
    ax.axis("off")
    lines = [f"{name}", ""]
    for c in (20, 50, 150, 250):
        so, sn = old["w"][old["p"] > c].sum(), new["w"][new["p"] > c].sum()
        lines.append(f"p > {c:3d} GeV: old {so:.3g}, ours {sn:.3g}/spill  ratio {sn / so:.3f}")
    ax.text(0, 0.9, "\n".join(lines), va="top", family="monospace", fontsize=11)
    for row, pcut in ((1, 20), (2, 150)):
        so, sn = old["p"] > pcut, new["p"] > pcut
        lim = 30 if pcut == 20 else 8
        panel_hist(axs[row, 0], old, new, "tx", np.linspace(-lim, lim, 61), so, sn, f"theta_x [mrad], p > {pcut} GeV")
        panel_hist(axs[row, 1], old, new, "ty", np.linspace(-lim, lim, 61), so, sn, f"theta_y [mrad], p > {pcut} GeV")
        panel_hist(axs[row, 2], old, new, "xc", np.linspace(-20, 20, 61), so, sn, f"x [cm], p > {pcut} GeV")
        panel_hist(axs[row, 3], old, new, "yc", np.linspace(-20, 20, 61), so, sn, f"y [cm], p > {pcut} GeV")
    fig.tight_layout(); fig.savefig(out / f"{name.replace(' ', '_')}.png", dpi=90); plt.close(fig)
    return {"bins": PB.tolist(), "ratio": [float(x) for x in r], "err": [float(x) for x in e],
            "flux_above": {str(c): [float(old["w"][old["p"] > c].sum()), float(new["w"][new["p"] > c].sum())] for c in (20, 50, 150, 250)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.results_dir() / "reproduction_v2"))
    ap.add_argument("--tags", default="mb2,tail2,dmlow2,dmhigh2,charmc")
    a = ap.parse_args()
    mb, tail, dml, dmh, ch = a.tags.split(",")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(INPUT, "r") as f:
        w = f["weight"][:].astype(np.float64)
        cols = {k: f[k][:] for k in ("px", "py", "pz", "x", "y", "pdg")}
    p = np.sqrt(cols["px"].astype(np.float64) ** 2 + cols["py"] ** 2 + cols["pz"] ** 2)
    classes = {"minimum bias": (768.75, merge(load(mb), load(tail))),
               "dimuon": (7.6875, merge(load(dml), load(dmh))),
               "charm": (326.16, load(ch))}
    summary, olds, news = {}, [], []
    for name, (cw, new) in classes.items():
        if new is None:
            print(f"  {name}: no production yet, skipped")
            continue
        m = (np.abs(w - cw) < 0.01 * cw) & (p > 10)
        old = derived({**{k: v[m] for k, v in cols.items()}, "p": p[m], "w": w[m]})
        new = derived({k: v[new["p"] > 10] for k, v in new.items()})
        summary[name] = plot_class(name, old, new, out)
        olds.append(old); news.append(new)
        print(f"{name}: " + ", ".join(f"p>{c}: {v[1] / v[0]:.3f}" for c, v in summary[name]["flux_above"].items()))
    if len(olds) > 1:
        cat = lambda L: {k: np.concatenate([d[k] for d in L]) for k in L[0]}
        summary["all classes"] = plot_class("all classes", cat(olds), cat(news), out)
        print("all classes: " + ", ".join(f"p>{c}: {v[1] / v[0]:.3f}" for c, v in summary["all classes"]["flux_above"].items()))
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print("plots in", out)


if __name__ == "__main__":
    main()
