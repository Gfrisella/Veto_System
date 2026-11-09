#!/usr/bin/env python3
"""Validation of tailtransport's complete mode (every muon source, forced and weighted) against its analog mode (the
same set-up, nothing forced: the unbiased truth).

  gun      single particles of one species, analog or complete: muons (p > PMIN) at the plane per gun particle
           target "Air": decays in flight over the 2.14 m to the plane (every long-lived and prompt species with a
           muonic channel); target "MoW": photons and positrons in the ECN3 target (gamma -> mu mu, e+ e- -> mu mu)
             bin/g4.sh python3 bin/complete_validation.py gun --mode analog --target Air --pdg 130 --n 2000000 --seed 1
  report   the gun runs and the full-event slices (tail_produce.py --mode complete / analog, tags cv_complete and
           cv_analog): yields and spectra, complete / analog, per species / per muon source; figures and a json summary
             python3 bin/complete_validation.py report

Outputs in $VETO_SOURCE/complete_validation (guns: gun_<mode>_<target>_<pdg>_<seed>.npz) and results/complete_validation.
"""

import _bootstrap  # noqa: F401

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from veto import paths

GUN_P = 50.0                 # GeV/c
PMIN = 5.0                   # GeV: muons kept above, particles followed above this total energy
NAMES = {211: "pi+", -211: "pi-", 321: "K+", -321: "K-", 130: "K_L", 310: "K_S", 3122: "Lambda", 3112: "Sigma-",
         3312: "Xi-", 3322: "Xi0", 221: "eta", 223: "omega", 113: "rho0", 333: "phi", 331: "eta'", 443: "J/psi",
         100443: "psi(2S)", 553: "Upsilon", 411: "D+", 421: "D0", 431: "D_s+", 4122: "Lambda_c+", 521: "B+", 511: "B0",
         15: "tau-", 22: "gamma", -11: "e+"}
PROC = {1: "long-lived decay", 2: "prompt decay", 7: "prompt decay (cascade)", 8: "resonance in a hadronic model", 3: "gamma -> mu mu", 4: "e+ e- -> mu mu",
        5: "Pythia muon", 6: "other"}
P_EDGES = np.geomspace(5, 60, 13)


def out_dir():
    return Path(os.environ.get("VETO_SOURCE", str(paths.source_data()))) / "complete_validation"


def gun(a):
    sys.path.insert(0, os.environ.get("TAILTRANSPORT_BUILD", str(paths.PROJECT_DIR / "cpp/tailgen/build")))
    import tailtransport as T
    T.initialize(seed=9000 + a.seed, target=a.target, physics=a.physics, mode=a.mode, ep_factor=a.ep_factor,
                 gamma_factor=a.gamma_factor)
    t0 = time.time()
    z0 = T.Z_TARGET + 0.001
    cols = {k: [] for k in ("w", "px", "py", "pz", "gen", "parent", "q", "hf")}
    done = 0
    while done < a.n:
        n = min(a.chunk, a.n - done)
        o = T.cascade(list(range(n + 1)), [a.pdg] * n, [0.0] * n, [0.0] * n, [a.p] * n, [0.0] * n, [0.0] * n, [z0] * n,
                      PMIN, PMIN, 0.0)
        for k in cols:
            cols[k].append(np.asarray(o[k]))
        done += n
    c = {k: np.concatenate(v) for k, v in cols.items()}
    out = out_dir()
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"gun_{a.mode}_{a.target}_{a.pdg}_{a.seed}.npz"
    np.savez(f, n=a.n, pdg=a.pdg, p_gun=a.p, mode=a.mode, target=a.target, **c)
    print(f"{f.name}: {a.n} guns, {len(c['w'])} muons, sum w {c['w'].sum():.4g}, {time.time() - t0:.0f} s")


def load_guns():
    res = {}
    for f in glob.glob(str(out_dir() / "gun_*.npz")):
        d = np.load(f)
        key = (str(d["mode"]), str(d["target"]), int(d["pdg"]))
        r = res.setdefault(key, dict(n=0, w=[], p=[], pt=[]))
        r["n"] += int(d["n"])
        p = np.sqrt(d["px"] ** 2 + d["py"] ** 2 + d["pz"] ** 2)
        r["w"].append(d["w"]); r["p"].append(p); r["pt"].append(np.hypot(d["px"], d["py"]))
    for r in res.values():
        for k in ("w", "p", "pt"):
            r[k] = np.concatenate(r[k]) if r[k] else np.zeros(0)
    return res


def yield_err(r, sel=None):
    w = r["w"] if sel is None else r["w"][sel]
    return w.sum() / r["n"], np.sqrt((w ** 2).sum()) / r["n"]


def shape_chi2(ra, rc, edges):
    """chi2 / ndf of the two spectra (per gun), errors from sum w^2."""
    ha = np.histogram(ra["p"], edges, weights=ra["w"])[0] / ra["n"]
    ea = np.sqrt(np.histogram(ra["p"], edges, weights=ra["w"] ** 2)[0]) / ra["n"]
    hc = np.histogram(rc["p"], edges, weights=rc["w"])[0] / rc["n"]
    ec = np.sqrt(np.histogram(rc["p"], edges, weights=rc["w"] ** 2)[0]) / rc["n"]
    ok = (ea > 0) & (ec > 0)
    return float(np.sum((ha[ok] - hc[ok]) ** 2 / (ea[ok] ** 2 + ec[ok] ** 2))), int(ok.sum()), (ha, ea, hc, ec)


def load_slices(tag):
    files = sorted(glob.glob(str(Path(os.environ.get("VETO_SOURCE", "")) / "tail" / f"{tag}_[0-9]*.npz")))
    ev, cols = 0.0, {k: [] for k in ("w", "p", "pt", "gen", "parent", "hf")}
    for f in files:
        d = np.load(f)
        ev += float(d["events"]) * float(d["rep"])
        cols["w"].append(d["w_decay"]); cols["p"].append(np.sqrt(d["px"] ** 2 + d["py"] ** 2 + d["pz"] ** 2))
        cols["pt"].append(np.hypot(d["px"], d["py"])); cols["gen"].append(d["gen"]); cols["parent"].append(d["parent"])
        cols["hf"].append(d["hf"])
    c = {k: np.concatenate(v) if v else np.zeros(0) for k, v in cols.items()}
    c["events"] = ev
    c["slices"] = len(files)
    return c


def report(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outd = paths.results_dir() / "complete_validation"
    outd.mkdir(parents=True, exist_ok=True)
    guns = load_guns()
    rows = []
    for (mode, target, pdg), ra in sorted(guns.items()):
        if mode != "analog" or (("complete", target, pdg) not in guns):
            continue
        rc = guns[("complete", target, pdg)]
        ya, ea = yield_err(ra)
        yc, ec = yield_err(rc)
        chi2, ndf, _ = shape_chi2(ra, rc, P_EDGES if target == "Air" else np.geomspace(5, 120, 13))
        r = yc / ya if ya > 0 else np.nan
        er = r * np.hypot(ea / ya, ec / yc) if ya > 0 and yc > 0 else np.nan
        rows.append(dict(pdg=pdg, name=NAMES.get(pdg, str(pdg)), target=target, n_analog=ra["n"], n_complete=rc["n"],
                         muons_analog=int(len(ra["w"])), yield_analog=ya, err_analog=ea, yield_complete=yc, err_complete=ec,
                         ratio=r, err_ratio=er, pull=(yc - ya) / np.hypot(ea, ec) if ya > 0 else np.nan,
                         chi2=chi2, ndf=ndf))
    print(f"{'species':>10} {'target':>6} {'analog muons':>12} {'yield analog':>14} {'yield complete':>15} {'ratio':>14} {'pull':>6} {'shape chi2/ndf':>15}")
    for r in rows:
        print(f"{r['name']:>10} {r['target']:>6} {r['muons_analog']:>12} {r['yield_analog']:>9.3e}±{r['err_analog'] / max(r['yield_analog'], 1e-300) * 100:3.0f}% "
              f"{r['yield_complete']:>10.3e}±{r['err_complete'] / max(r['yield_complete'], 1e-300) * 100:3.1f}% "
              f"{r['ratio']:>7.3f}±{r['err_ratio']:.3f} {r['pull']:>6.2f} {r['chi2']:>9.1f}/{r['ndf']}")
    # full events
    full = {}
    sa, sc = load_slices(a.analog_tag), load_slices(a.complete_tag)
    if sa["slices"] and sc["slices"]:
        for lab, sel_a, sel_c in [("all", None, None)] + [(PROC[k], sa["gen"] == k, sc["gen"] == k) for k in PROC]:
            wa = sa["w"] if sel_a is None else sa["w"][sel_a]
            wc = sc["w"] if sel_c is None else sc["w"][sel_c]
            ya, ea = wa.sum() / sa["events"], np.sqrt((wa ** 2).sum()) / sa["events"]
            yc, ec = wc.sum() / sc["events"], np.sqrt((wc ** 2).sum()) / sc["events"]
            full[lab] = dict(analog=ya, err_analog=ea, complete=yc, err_complete=ec, muons_analog=int(len(wa)))
        print(f"\nfull events (400 GeV p on the target, muons > {a.emin:.0f} GeV at the plane, per p.o.t.): analog "
              f"{sa['events']:.3g} p.o.t. ({sa['slices']} slices), complete {sc['events']:.3g} p.o.t. ({sc['slices']} slices)")
        for lab, f in full.items():
            r = f["complete"] / f["analog"] if f["analog"] > 0 else np.nan
            print(f"  {lab:>18}: analog {f['analog']:.4e} ± {f['err_analog'] / max(f['analog'], 1e-300) * 100:.1f}% ({f['muons_analog']} muons), "
                  f"complete {f['complete']:.4e} ± {f['err_complete'] / max(f['complete'], 1e-300) * 100:.2f}%, ratio {r:.3f}")
    json.dump(dict(guns=rows, full=full), open(outd / "summary.json", "w"), indent=1, default=float)
    # figure: per species ratio, full-event spectra
    fig, axs = plt.subplots(1, 3, figsize=(22, 7), gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
    ax = axs[0]
    yy = np.arange(len(rows))
    ax.errorbar([r["ratio"] for r in rows], yy, xerr=[r["err_ratio"] for r in rows], fmt="o", color="#2a78d6")
    ax.set_yticks(yy, [f"{r['name']} ({r['target']}, {r['muons_analog']} analog muons)" for r in rows], fontsize=8)
    ax.axvline(1, color="k", lw=1); ax.axvspan(0.95, 1.05, color="0.92", zorder=0)
    ax.set_xlabel("muon yield per particle: complete (forced) / analog"); ax.set_title(f"single particles, p = {GUN_P:.0f} GeV, muons > {PMIN:.0f} GeV")
    ax.set_xlim(0.6, 1.4); ax.invert_yaxis()
    if sa["slices"] and sc["slices"]:
        edges = np.geomspace(a.emin, 300, 21)
        pc = np.sqrt(edges[1:] * edges[:-1])
        for j, (var, lab) in enumerate((("p", "muon momentum at the plane [GeV]"), ("pt", "muon pT [GeV]"))):
            ax = axs[1 + j]
            e = edges if var == "p" else np.linspace(0, 4, 21)
            x = pc if var == "p" else 0.5 * (e[1:] + e[:-1])
            for s, col, l in ((sa, "#222222", "analog"), (sc, "#2a78d6", "complete (forced)")):
                h = np.histogram(s[var], e, weights=s["w"])[0] / s["events"]
                eh = np.sqrt(np.histogram(s[var], e, weights=s["w"] ** 2)[0]) / s["events"]
                ax.errorbar(x, h / np.diff(e), yerr=eh / np.diff(e), ds="steps-mid", color=col, lw=2, label=f"{l}, all sources")
            for k, col in ((1, "#eb6834"), (2, "#1baa7a"), (3, "#a35ec4"), (4, "#d4a017"), (5, "#777777"), (6, "#bb3333"), (7, "#0f7c5c"), (8, "#7a5c0f")):
                m = sc["gen"] == k
                if m.any():
                    h = np.histogram(sc[var][m], e, weights=sc["w"][m])[0] / sc["events"]
                    ax.step(x, h / np.diff(e), where="mid", color=col, lw=1.2, ls="--", label=f"complete: {PROC[k]}")
            ax.set_yscale("log"); ax.set_xlabel(lab); ax.set_ylabel("muons per p.o.t. per GeV")
            if var == "p":
                ax.set_xscale("log")
            ax.legend(fontsize=7); ax.set_title("full 400 GeV proton events on the ECN3 target")
    for x in axs:
        x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outd / "complete_validation.png", dpi=95)
    print(outd / "complete_validation.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gun")
    g.add_argument("--mode", choices=["complete", "analog"], required=True)
    g.add_argument("--target", default="Air")
    g.add_argument("--physics", default="FTFP_BERT")
    g.add_argument("--pdg", type=int, required=True)
    g.add_argument("--p", type=float, default=GUN_P)
    g.add_argument("--n", type=int, default=100000)
    g.add_argument("--chunk", type=int, default=500000)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--gamma-factor", type=float, default=500.0)
    g.add_argument("--ep-factor", type=float, default=100.0)
    r = sub.add_parser("report")
    r.add_argument("--analog-tag", default="cv_analog")
    r.add_argument("--complete-tag", default="cv_complete")
    r.add_argument("--emin", type=float, default=20.0)
    a = ap.parse_args()
    gun(a) if a.cmd == "gun" else report(a)


if __name__ == "__main__":
    main()
