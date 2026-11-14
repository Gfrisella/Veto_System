#!/usr/bin/env python3
"""Every V0 created in the last 50 cm of MS7 and what happens to it (ISSUES D16): reads the V0 tally of
shield_dis.py generate --v0-tally (cpp/g4shield v0_tally) and the chain summaries, per spill at 4e13 p.o.t.

    python3 bin/v0_fate.py --runs default=results/shield_dis/shield_muon_g4v114_default_s401 \\
                           --runs v0stable=results/shield_dis/shield_muon_g4v114_v0stable_s402 \\
                           [--pr '/disk/lhcb_data/.../shield_muon_g4v112_s1??.json'] [--out results/v0_production]

Origin: "vertex" = injected by the muon interaction (library final state), "shower" = made by a secondary in the
iron (creator process). Fate: below_cut (T < 0.5 GeV: never tracked), decayed, absorbed (hadronic interaction),
exit (z = 29.90 m), other. Writes v0_fate.json, v0_fate_*.png/pdf.
"""

import _bootstrap  # noqa: F401
import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from veto import paths
from veto.config import SPILLS_15Y

PR = "/disk/lhcb_data/sqasim/workspace/uzh/Veto_System/veto/results/shield_dis/shield_muon_g4v112_s1??.json"
NAMES = {130: "K_L", 310: "K_S", 3122: "Lambda", -3122: "anti-Lambda", 3322: "Xi0", -3322: "anti-Xi0"}
FATES = ["below_cut", "decayed", "absorbed", "exit", "other"]
MASS = {130: 0.497611, 310: 0.497611, 3122: 1.115683, 3322: 1.31486}


def load(prefix):
    t = dict(np.load(prefix + "_v0tally.npz"))
    summ = json.load(open(prefix + ".json"))
    w = t["int_weight"][t["ev"]]
    origin = np.where(t["parent"] == 0, "vertex", "shower")
    return t, summ, w, origin


def table(t, w, origin):
    out = {}
    for code, name in NAMES.items():
        m = t["pdg"] == code
        if not m.any():
            continue
        row = {}
        for o in ("vertex", "shower", "all"):
            mo = m & ((origin == o) if o != "all" else True)
            row[o] = {"made": float(w[mo].sum()), "n_mc": int(mo.sum())}
            for i, f in enumerate(FATES):
                row[o][f] = float(w[mo & (t["fate"] == i)].sum())
        out[name] = row
    return out


def creators(t, w, code, top=8):
    m = (t["pdg"] == code) & (t["parent"] != 0)
    c = t["creator"][m]
    names, inv = np.unique(c, return_inverse=True)
    s = np.bincount(inv, w[m])
    o = np.argsort(-s)[:top]
    tot = w[m].sum()
    return {str(names[i]): float(s[i] / tot) for i in o}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", action="append", required=True, help="label=prefix (without .npz)")
    ap.add_argument("--pr", default=PR)
    ap.add_argument("--out", default=str(paths.results_dir() / "v0_production"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R = {"per": "spill (4e13 p.o.t.)", "spills_15y": SPILLS_15Y, "runs": {}}
    runs = {}
    for spec in a.runs:
        lab, pre = spec.split("=", 1)
        t, summ, w, origin = load(pre)
        runs[lab] = (t, w, origin)
        R["runs"][lab] = {
            "prefix": pre, "mu_library": summ.get("mu_library"), "interactions_per_spill": summ["interactions_per_spill"],
            "draws": summ["draws"], "table": table(t, w, origin),
            "shower_creators": {NAMES[c]: creators(t, w, c) for c in (130, 310, 3122)},
            "exit_p_gt_1GeV": {k: v["per_spill"] for k, v in summ["exit"].items()},
            "v0_decays_in_vessel_per_spill": summ["v0_decays_in_vessel_per_spill"],
            "v0_decays_in_vessel_15y_total": SPILLS_15Y * sum(summ["v0_decays_in_vessel_per_spill"].values()),
        }
    pr = sorted(glob.glob(a.pr))
    if pr:
        js = [json.load(open(f)) for f in pr]
        R["pr15_g4v112"] = {
            "files": len(js),
            "exit_p_gt_1GeV": {k: float(np.mean([j["exit"][k]["per_spill"] for j in js])) for k in js[0]["exit"]},
            "v0_decays_in_vessel_per_spill": {k: float(np.mean([j["v0_decays_in_vessel_per_spill"].get(k, 0) for j in js]))
                                              for k in js[0]["v0_decays_in_vessel_per_spill"]}}
    json.dump(R, open(out / "v0_fate.json", "w"), indent=1)

    # ---- printout
    for lab, r in R["runs"].items():
        print(f"\n== {lab}: {r['mu_library']}  ({r['interactions_per_spill']:.3g} interactions / spill)")
        print(f"{'':12s} {'origin':7s} {'made':>9s} " + " ".join(f"{f:>10s}" for f in FATES) + "   [per spill]")
        for name, row in r["table"].items():
            for o in ("vertex", "shower", "all"):
                x = row[o]
                print(f"{name:12s} {o:7s} {x['made']:9.4g} " + " ".join(f"{x[f]:10.4g}" for f in FATES))
        print("shower creators:", json.dumps(r["shower_creators"]))
        print("vessel decays / spill:", {k: round(v, 3) for k, v in r["v0_decays_in_vessel_per_spill"].items()},
              f"-> {r['v0_decays_in_vessel_15y_total']:.3g} / 15 y")
    if pr:
        print("\nPR #15 (Geant4 11.2):", R["pr15_g4v112"])

    # ---- figures: fate bars and creation z / momentum per origin
    labs = list(runs)
    fig, axs = plt.subplots(1, len(labs), figsize=(5.5 * len(labs), 4.2), squeeze=False)
    cols = {"below_cut": "0.7", "decayed": "C2", "absorbed": "C3", "exit": "C0", "other": "C7"}
    for ax, lab in zip(axs[0], labs):
        tb = R["runs"][lab]["table"]
        names = [n for n in ("K_L", "K_S", "Lambda", "anti-Lambda") if n in tb]
        xs, ticks = [], []
        for i, n in enumerate(names):
            for j, o in enumerate(("vertex", "shower")):
                x = 3 * i + j
                bottom = 0
                for f in FATES:
                    v = tb[n][o][f]
                    ax.bar(x, v, bottom=bottom, color=cols[f], label=f if (i == 0 and j == 0) else None)
                    bottom += v
                xs.append(x)
                ticks.append(f"{n}\n{o}")
        ax.set_xticks(xs, ticks, fontsize=7)
        ax.set_yscale("log")
        ax.set_ylim(10, None)
        ax.set_ylabel("V0 made per spill (stacked by fate)")
        ax.set_title(f"{lab}: every V0 made in the last 50 cm of MS7", fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"v0_fate_bars.{ext}", dpi=140)

    lab = labs[-1]
    t, w, origin = runs[lab]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    pm = np.sqrt(t["px"] ** 2 + t["py"] ** 2 + t["pz"] ** 2)
    for code, col in ((130, "C0"), (310, "C1"), (3122, "C2")):
        for o, ls in (("vertex", "-"), ("shower", "--")):
            m = (t["pdg"] == code) & (origin == o)
            ax[0].hist(t["z"][m], np.linspace(29.39, 29.89, 26), weights=w[m], histtype="step", color=col, ls=ls,
                       label=f"{NAMES[code]} {o}")
            ax[1].hist(pm[m], np.logspace(-1, 2, 40), weights=w[m], histtype="step", color=col, ls=ls,
                       label=f"{NAMES[code]} {o}")
    ax[0].set_xlabel("z of creation [m] (shield end 29.89)")
    ax[0].set_ylabel("V0 made / spill / bin")
    ax[0].set_yscale("log")
    ax[0].legend(fontsize=7)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("p at creation [GeV]")
    ax[1].legend(fontsize=7)
    ax[0].set_title(f"{lab}: where V0 are made", fontsize=9)
    ax[1].set_title(f"{lab}: momentum at creation", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"v0_fate_origin.{ext}", dpi=140)
    print(f"\nwrote {out / 'v0_fate.json'}")


if __name__ == "__main__":
    main()
