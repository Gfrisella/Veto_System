#!/usr/bin/env python3
"""Run digitization + veto reconstruction (algorithm A) on designs and report what comes out.

For each design (designs/*.json):
  background  every muon-pair candidate (results/reco/inputs.npz) is one event: its two muons'
              own hits + accidental hits; vetoed if one of its tracks is explained.
              TP = vetoed, FN = survives. Surviving rate = sum of pair weights * 2 dt / spill.
  signal      hypothetical long-lived neutrals from the target (nearly along the beam) decaying in the
              vessel into two tracks, with the background pairs' selection (--signal old: tracks
              aimed at random points of SST1). No own hits: only accidental hits can veto them.
              FP = vetoed (signal lost), TN = kept.
  single      single SST1 muons: veto efficiency per muon type.
Also: detector rates per plane and cell, how well the search window contains the true
crossings, number of hit comparisons (the combinatorics) and runtime per event.

  bin/run_reco.py                          # all designs, local (needs results/reco/inputs.npz)
  bin/run_reco.py D2_default D4_dense --spill 4.8
"""

import _bootstrap  # noqa: F401

import argparse
import json
import time

import numpy as np

from veto import design as dsg
from veto import paths, reco
from veto.config import DEFAULT_CONFIG

TYPE_NAMES = {0: "SST1 bounced", 1: "SST1 p>1 GeV & timing", 2: "SST1 other"}
SPILLS_15Y = DEFAULT_CONFIG["pot_15_years"] / DEFAULT_CONFIG["pot_spill"]


def load_inputs():
    f = np.load(paths.results_dir() / "reco" / "inputs.npz")
    tr = {k[3:]: f[k].astype(np.float64) for k in f.files if k.startswith("tr_")}
    return tr, {k: f[k] for k in f.files if not k.startswith("tr_")}


def take(tr, rows):
    return {k: v[rows] for k, v in tr.items()}


def containment(tr, rows, design, cfg, rng):
    """Per muon type: fraction of true crossings that fall inside the algorithm's spatial window.
    Bounced muons are only predictable after their bounce (the spectrometer sees that part)."""
    sub = take(tr, rows)
    meas = reco.measure(sub, cfg, rng)
    ty = sub["type"].astype(int)
    inside, total = np.zeros(3), np.zeros(3)
    for plane in design.planes:
        u, y, _, ok = reco.true_crossing(sub, plane)
        dz = meas["z"] - plane.z
        pu = plane.to_local(meas["x"] - meas["tx"] * dz)
        py = meas["y"] - meas["ty"] * dz
        dist = dz * np.sqrt(1 + meas["tx"] ** 2 + meas["ty"] ** 2)
        r = cfg.r0 + (cfg.r_slope + cfg.r_ms / np.maximum(meas["p"], 0.1)) * dist
        uc, yc = plane.cell_centre(u, y)
        good = (np.abs(uc - pu) <= r + plane.cell_u / 2) & (np.abs(yc - py) <= r + plane.cell_y / 2)
        inside += np.bincount(ty[good & ok], minlength=3); total += np.bincount(ty[ok], minlength=3)
    return {TYPE_NAMES[k]: float(inside[k] / max(total[k], 1)) for k in range(3)}


def evaluate(design, tr, inp, cfg, n_signal, dt_pair, signal_model="target"):
    rng = np.random.default_rng(cfg.seed)
    res = {"design": design.name, "description": design.raw.get("description", ""), "notes": design.notes,
           "area_m2": design.area, "n_cells": design.n_cells, "n_planes": len(design.planes)}

    # background candidates
    cand = inp["cand"]
    rows = cand.ravel()
    event = np.repeat(np.arange(len(cand)), 2)
    z_lim = np.repeat(inp["cand_vertex"][:, 2], 2)
    t0 = time.perf_counter()
    b = reco.run_events(take(tr, rows), event, z_lim, design, inp, cfg, rng)
    t_bkg = time.perf_counter() - t0
    w = inp["cand_w"] * 2 * dt_pair / cfg.spill_s           # random-coincidence candidates per spill
    surv = ~b["vetoed"]
    res["background"] = dict(n=len(cand), n_vetoed=int(b["vetoed"].sum()), n_survive=int(surv.sum()),
                             TP_frac=float(1 - w[surv].sum() / w.sum()), FN_frac=float(w[surv].sum() / w.sum()),
                             per_spill_before=float(w.sum()), per_spill_after=float(w[surv].sum()),
                             per_15y_before=float(w.sum() * SPILLS_15Y), per_15y_after=float(w[surv].sum() * SPILLS_15Y),
                             hits_per_event=b["n_hits"] / len(cand), comparisons=b["n_compared"],
                             seconds=t_bkg, us_per_event=1e6 * t_bkg / len(cand), timing=b["timing"])

    # hypothetical signal
    if signal_model == "target":
        sig, z_v, ev, acceptance = reco.signal_from_target(n_signal, rng)
    else:
        (sig, z_v, ev), acceptance = reco.signal_tracks(n_signal, rng), {}
    t0 = time.perf_counter()
    s = reco.run_events(sig, ev, z_v, design, inp, cfg, rng, with_own_hits=False)
    t_sig = time.perf_counter() - t0
    n_ev = len(s["vetoed"])
    res["signal"] = dict(n=n_ev, FP_frac=float(s["vetoed"].mean()), TN_frac=float(1 - s["vetoed"].mean()),
                         FP_by_plane={design.planes[i].name: int(c) for i, c in enumerate(s["by_plane"]) if c},
                         hits_per_event=s["n_hits"] / n_ev, us_per_event=1e6 * t_sig / n_ev,
                         model=signal_model, acceptance=acceptance)

    # figure of merit. Positive = vetoed. Background rates are weighted (expected events), signal
    # rates are fractions of selected signal decays.
    w15 = w * SPILLS_15Y
    b_after = float(w15[surv].sum())
    b_err = float(np.sqrt((w15[surv] ** 2).sum())) if surv.any() else float(2.3 * w15.mean())
    eff = 1 - res["signal"]["FP_frac"]
    res["fom"] = dict(TPR=res["background"]["TP_frac"], FNR=res["background"]["FN_frac"],
                      FPR=res["signal"]["FP_frac"], TNR=eff, B_15y=b_after, B_15y_err=b_err,
                      B_15y_is_upper_limit=bool(not surv.any()), signal_efficiency=eff,
                      punzi=float(eff / (1.5 + np.sqrt(b_after))))

    # single muons
    single = inp["single"]
    sw = inp["single_w"]
    t0 = time.perf_counter()
    m = reco.run_events(take(tr, single), np.arange(len(single)), np.full(len(single), 1e9), design, inp, cfg, rng)
    ty = tr["type"][single].astype(int)
    res["single"] = {TYPE_NAMES[k]: float((sw * m["vetoed"])[ty == k].sum() / sw[ty == k].sum()) for k in range(3)}
    res["single"]["all SST1"] = float((sw * m["vetoed"]).sum() / sw.sum())
    res["single"]["us_per_event"] = 1e6 * (time.perf_counter() - t0) / len(single)
    res["window_containment"] = containment(tr, rng.choice(single, 50_000, replace=False), design, cfg, rng)

    rates = reco.plane_rates(design, inp)
    res["planes"] = [dict(r, peak_cell_hz=r["peak_cell"] / cfg.spill_s, mean_cell_hz=r["mean_cell"] / cfg.spill_s)
                     for r in rates]
    res["total_rate_per_spill"] = float(sum(r["rate"] for r in rates))
    return res


def report(r, cfg):
    b, s, m = r["background"], r["signal"], r["single"]
    print(f"\n=== {r['design']}: {r['description']}")
    for n in r["notes"]:
        print(f"    note: {n}")
    print(f"    {r['n_planes']} planes, {r['area_m2']:.1f} m2, {r['n_cells']:,} cells, "
          f"{r['total_rate_per_spill']:.3g} muon hits / spill ({r['total_rate_per_spill'] / cfg.spill_s:.3g} Hz)")
    print(f"    background: {b['n']:,} candidates, TP {100 * b['TP_frac']:.3f} %  FN {100 * b['FN_frac']:.3f} % "
          f"({b['n_survive']:,} survive) -> {b['per_15y_before']:.3g} -> {b['per_15y_after']:.3g} events in 15 years")
    print(f"    signal:     FP (lost to accidental hits) {100 * s['FP_frac']:.3f} %, TN {100 * s['TN_frac']:.3f} %"
          + (f"  [{', '.join(f'{k}: {v}' for k, v in sorted(s['FP_by_plane'].items(), key=lambda t: -t[1])[:3])}]"
             if s["FP_by_plane"] else ""))
    f = r["fom"]
    print(f"    FoM: TPR {100 * f['TPR']:.4f} %  FNR {100 * f['FNR']:.4f} %  FPR {100 * f['FPR']:.3f} %  "
          f"TNR {100 * f['TNR']:.3f} %;  B(15 y) {f['B_15y']:.2g} ± {f['B_15y_err']:.1g};  Punzi {f['punzi']:.3f}")
    print(f"    single SST1 muons vetoed: " + ", ".join(f"{k} {100 * v:.2f} %" for k, v in m.items() if k != "us_per_event"))
    print("    search window contains the true crossings of: "
          + ", ".join(f"{k} {100 * v:.1f} %" for k, v in r["window_containment"].items()))
    print(f"    hits / event: bkg {b['hits_per_event']:.1f}, signal {s['hits_per_event']:.1f};  "
          f"{b['comparisons']:,} hit comparisons for the background")
    print(f"    runtime: {b['us_per_event']:.1f} us / background event, {s['us_per_event']:.1f} us / signal event "
          f"(numpy, one core)")
    worst = max(r["planes"], key=lambda p: p["peak_cell_hz"])
    print(f"    busiest cell: {worst['peak_cell_hz']:.3g} Hz on {worst['plane']} "
          f"(mean over that plane {worst['mean_cell_hz']:.3g} Hz)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("designs", nargs="*", help="design names or json files (default: all in designs/)")
    parser.add_argument("--spill", type=float, default=1.0, help="spill length [s]")
    parser.add_argument("--n-signal", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--signal", choices=("target", "old"), default="target",
                        help="target: neutral from the target, decays in the vessel (default); old: tracks aimed at random")
    parser.add_argument("--out", default="results.json")
    args = parser.parse_args()
    names = args.designs or sorted(p.stem for p in dsg.DESIGN_DIR.glob("*.json"))
    cfg = reco.RecoConfig(spill_s=args.spill, seed=args.seed)
    tr, inp = load_inputs()
    dt_pair = DEFAULT_CONFIG["time_window"] * 1e9 * 1e-9    # two-track coincidence window [s]
    results = []
    for name in names:
        r = evaluate(dsg.load(name), tr, inp, cfg, args.n_signal, dt_pair, args.signal)
        report(r, cfg)
        results.append(r)
    print(f"\nFIGURE OF MERIT (signal model: {args.signal}; positive = vetoed; spill {args.spill:g} s)")
    print(f"{'design':<15}{'area m2':>8}{'TPR %':>10}{'FNR %':>9}{'FPR %':>8}{'TNR %':>8}{'pairs left':>11}"
          f"{'B / 15 y':>18}{'Punzi':>7}")
    for r in results:
        f, b = r["fom"], r["background"]
        lim = "<" if f["B_15y_is_upper_limit"] else ""
        print(f"{r['design']:<15}{r['area_m2']:>8.0f}{100 * f['TPR']:>10.4f}{100 * f['FNR']:>9.4f}{100 * f['FPR']:>8.3f}"
              f"{100 * f['TNR']:>8.3f}{b['n_survive']:>11,}{lim + format(f['B_15y'], '.2g') + ' ± ' + format(f['B_15y_err'], '.1g'):>18}"
              f"{f['punzi']:>7.3f}")
    out = paths.results_dir() / "reco" / args.out
    out.write_text(json.dumps(dict(config=vars(cfg), spill_s=args.spill, signal=args.signal, results=results),
                              indent=1, default=float))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
