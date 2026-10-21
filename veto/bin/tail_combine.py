"""Combine the tail production slices and validate them against the input sample's min.-bias tail.

    python3 bin/tail_combine.py [--pattern 'tail_*.npz'] [--out results/tail]

Weights: muons per spill = w_decay / rep * 5e13 / (events of all combined slices). Compared with
MuonsAndMatter's full_sample_after_target.h5, class w = 768.75 (minimum bias), at the same plane: flux above
thresholds, momentum spectrum, angles, charge ratio, position; ratio ours / input per momentum bin with the
input's Poisson error. Writes <out>/validation.png, <out>/validation.json and data/tail/tail_combined.npz.
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


def kish(w):
    return float(w.sum() ** 2 / np.sum(w ** 2)) if len(w) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="tail_[0-9]*.npz")
    ap.add_argument("--out", default=str(paths.results_dir() / "tail"))
    ap.add_argument("--outfile", default="tail_combined.npz", help="combined file name in data/tail")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(paths.source_data() / "tail" / a.pattern)))
    parts = [dict(np.load(f)) for f in files]
    events = sum(int(p["events"]) for p in parts)
    cols = ("pdg", "x", "y", "z", "px", "py", "pz", "parent", "z_decay", "z_int", "p_prod", "event")
    d = {c: np.concatenate([p[c] for p in parts]) for c in cols}
    # cluster id of the proton collision (slice, event): muons of one collision are correlated
    d["cluster"] = np.concatenate([np.full(len(p["w_decay"]), k, np.int64) * 10_000_000 + p["event"]
                                   for k, p in enumerate(parts)])
    d["w"] = np.concatenate([p["w_decay"] / p["rep"] for p in parts]) * 5e13 / events
    d["p"] = np.sqrt(d["px"] ** 2 + d["py"] ** 2 + d["pz"] ** 2)
    np.savez(paths.source_data() / "tail" / a.outfile, events=events, **d)
    print(f"{len(files)} slices, {events:.4g} events (protons on target), {len(d['w']):,} muons at the plane")

    with h5py.File(INPUT, "r") as f:
        w = f["weight"][:]
        m = np.abs(w - 768.75) < 1
        r = {k: f[k][:][m].astype(np.float64) for k in ("px", "py", "pz", "x", "y", "pdg")}
    r["p"] = np.sqrt(r["px"] ** 2 + r["py"] ** 2 + r["pz"] ** 2)
    r["w"] = np.full(len(r["p"]), 768.75)

    summary = {"events": events, "muons": int(len(d["w"])), "thresholds": {}}
    for c in (150, 200, 250, 300, 350):
        s, t = d["p"] > c, r["p"] > c
        summary["thresholds"][c] = {
            "ours_per_spill": float(d["w"][s].sum()), "ours_rows": int(s.sum()), "ours_kish": kish(d["w"][s]),
            "input_per_spill": float(r["w"][t].sum()), "input_rows": int(t.sum()),
            "ratio": float(d["w"][s].sum() / max(r["w"][t].sum(), 1e-30)),
            "ours_mu_plus": float(np.average(d["pdg"][s] == -13, weights=d["w"][s])) if s.any() else None,
            "input_mu_plus": float(np.mean(r["pdg"][t] == -13)) if t.any() else None,
            "ours_K_share": float(np.average(np.abs(d["parent"][s]) == 321, weights=d["w"][s])) if s.any() else None}
        x = summary["thresholds"][c]
        print(f"p > {c}: ours {x['ours_per_spill']:.3g}/spill ({x['ours_rows']} rows, Kish {x['ours_kish']:.0f}) vs input "
              f"{x['input_per_spill']:.3g} ({x['input_rows']} muons): ratio {x['ratio']:.2f}; mu+ {x['ours_mu_plus']} vs {x['input_mu_plus']}; "
              f"K share {x['ours_K_share']}")
    json.dump(summary, open(out / "validation.json", "w"), indent=1)

    fig, axs = plt.subplots(2, 4, figsize=(24, 9), gridspec_kw={"height_ratios": [2, 1]})
    sel_d, sel_r = d["p"] > 150, r["p"] > 150
    specs = [("p at z = -0.0045 m [GeV]", "p", np.linspace(150, 400, 51)),
             (r"$\theta_x$ [mrad] (p > 150 GeV)", "tx", np.linspace(-10, 10, 41)),
             ("x [cm] (p > 150 GeV)", "x", np.linspace(-15, 15, 31)),
             (r"$\theta$ [mrad] (p > 150 GeV)", "th", np.linspace(0, 15, 31))]
    for D in (d, r):
        D["tx"] = 1e3 * D["px"] / D["pz"]
        D["th"] = 1e3 * np.hypot(D["px"], D["py"]) / D["pz"]
        D["x_cm"] = 100 * D["x"]
    for c, (lab, key, b) in enumerate(specs):
        k = "x_cm" if key == "x" else key
        hd, _ = np.histogram(d[k][sel_d], b, weights=d["w"][sel_d])
        hd2, _ = np.histogram(d[k][sel_d], b, weights=d["w"][sel_d] ** 2)
        hr, _ = np.histogram(r[k][sel_r], b, weights=r["w"][sel_r])
        nr, _ = np.histogram(r[k][sel_r], b)
        axs[0, c].stairs(hr, b, color="k", lw=2, label=f"input, min. bias ({sel_r.sum():,} muons)")
        axs[0, c].stairs(hd, b, color="crimson", lw=2, label=f"forced decays ({sel_d.sum():,} muons)")
        if key == "p":
            for q, ls in ((-13, ":"), (13, "--")):
                mq, mr = sel_d & (d["pdg"] == q), sel_r & (r["pdg"] == q)
                axs[0, c].stairs(np.histogram(d[k][mq], b, weights=d["w"][mq])[0], b, color="crimson", lw=1, ls=ls)
                axs[0, c].stairs(np.histogram(r[k][mr], b, weights=r["w"][mr])[0], b, color="k", lw=1, ls=ls)
            axs[0, c].set_yscale("log")
        axs[0, c].set_ylabel("muons / spill / bin"); axs[0, c].legend(fontsize=8)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = hd / hr
            err_r = ratio / np.sqrt(np.maximum(nr, 1))
            err_d = np.sqrt(hd2) / np.maximum(hr, 1e-30)
        cen = 0.5 * (b[1:] + b[:-1])
        axs[1, c].errorbar(cen, ratio, yerr=np.hypot(err_r, err_d), fmt="o", color="crimson", ms=3)
        axs[1, c].axhline(1, color="k", lw=0.8); axs[1, c].set_ylim(0, 2); axs[1, c].set_xlabel(lab)
        axs[1, c].set_ylabel("forced / input")
    fig.suptitle(f"high-energy muon tail: forced decays in flight ({events:.3g} protons on target) vs the input sample's "
                 f"min.-bias class; dotted mu+, dashed mu-")
    fig.tight_layout(); fig.savefig(out / "validation.png", dpi=100)
    print("plots in", out)


if __name__ == "__main__":
    main()
