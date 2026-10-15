#!/usr/bin/env python3
"""Run digitization + veto reconstruction (algorithm A) on designs and report what comes out.

For each design (designs/*.json):
  background  every muon-pair candidate (results/reco/inputs.npz) is one event: its two muons'
              own hits + accidental hits; vetoed if one of its tracks is explained.
              TP = vetoed, FN = survives. Surviving rate = sum of pair weights * 2 dt / spill.
  signal      hypothetical neutral decays in the vessel (two tracks, no own hits): only
              accidental hits can veto them. FP = vetoed (signal lost), TN = kept.
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


def evaluate(design, tr, inp, cfg, n_signal, dt_pair):
    rng = np.random.default_rng(cfg.seed)
    res = {"design": design.name, "description": design.raw.get("description", ""), "notes": design.notes,
           "area_m2": design.area, "n_cells": design.n_cells, "n_planes": len(design.planes),
           "technologies": sorted({p.technology or "generic" for p in design.planes}),
           "n_channels": design.n_channels, "cost_kchf": design.cost, "rel_cost": design.rel_cost,
           "ubt_x0": design.ubt_x0}

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
    sig, z_v, ev = reco.signal_tracks(n_signal, rng)
    t0 = time.perf_counter()
    s = reco.run_events(sig, ev, z_v, design, inp, cfg, rng, with_own_hits=False)
    t_sig = time.perf_counter() - t0
    n_ev = len(s["vetoed"])
    res["signal"] = dict(n=n_ev, FP_frac=float(s["vetoed"].mean()), TN_frac=float(1 - s["vetoed"].mean()),
                         FP_by_plane={design.planes[i].name: int(c) for i, c in enumerate(s["by_plane"]) if c},
                         hits_per_event=s["n_hits"] / n_ev, us_per_event=1e6 * t_sig / n_ev)

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

    rates = reco.plane_rates(design, inp, cfg.spill_s)
    res["planes"] = [dict(r, peak_cell_hz=r["peak_cell"] / cfg.spill_s, mean_cell_hz=r["mean_cell"] / cfg.spill_s,
                          peak_channel_hz=r["peak_channel"] / cfg.spill_s, mean_channel_hz=r["mean_channel"] / cfg.spill_s)
                     for r in rates]
    res["total_rate_per_spill"] = float(sum(r["rate"] for r in rates))
    res["over_rate_limit"] = [dict(plane=r["plane"], technology=r["technology"], peak_hz_cm2=r["peak_hz_cm2"],
                                   area_m2=r["area_over_limit"], peak_channel_hz=r["peak_channel"] / cfg.spill_s)
                              for r in rates if r["area_over_limit"] > 0 or r["channel_over_limit"]]
    return res


def tag(cfg):
    """Output-name suffix: '' for the default window, '_tw100' for a fixed 100 ns one."""
    return f"_tw{cfg.time_window_ns:g}" if cfg.time_window_ns else ""


def report(r, cfg):
    b, s, m = r["background"], r["signal"], r["single"]
    print(f"\n=== {r['design']}: {r['description']}")
    for n in r["notes"]:
        print(f"    note: {n}")
    print(f"    {r['n_planes']} planes, {r['area_m2']:.1f} m2, {r['n_cells']:,} cells, "
          f"{r['total_rate_per_spill']:.3g} muon hits / spill ({r['total_rate_per_spill'] / cfg.spill_s:.3g} Hz)")
    print(f"    technology: {', '.join(r['technologies'])}; {r['n_channels']:,} readout channels, "
          f"cost {r['cost_kchf']:.4g} kCHF ({r['rel_cost']:.4g} straw-channel units), "
          f"UBT material {100 * r['ubt_x0']:.2f} % X0")
    print(f"    background: {b['n']:,} candidates, TP {100 * b['TP_frac']:.3f} %  FN {100 * b['FN_frac']:.3f} % "
          f"({b['n_survive']:,} survive) -> {b['per_15y_before']:.3g} -> {b['per_15y_after']:.3g} events in 15 years")
    print(f"    signal:     FP (lost to accidental hits) {100 * s['FP_frac']:.3f} %, TN {100 * s['TN_frac']:.3f} %"
          + (f"  [{', '.join(f'{k}: {v}' for k, v in sorted(s['FP_by_plane'].items(), key=lambda t: -t[1])[:3])}]"
             if s["FP_by_plane"] else ""))
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
    for o in r["over_rate_limit"]:
        print(f"    OVER RATE LIMIT: {o['plane']} ({o['technology']}) peaks at {o['peak_hz_cm2'] / 1e6:.2f} MHz/cm2 "
              f"({o['area_m2']:.2f} m2 above the density limit), {o['peak_channel_hz'] / 1e6:.2f} MHz per channel")
    worst = max(r["planes"], key=lambda p: p["peak_channel_hz"])
    print(f"    busiest readout channel: {worst['peak_channel_hz']:.3g} Hz on {worst['plane']} "
          f"({worst['technology'] or 'generic'}, mean over that plane {worst['mean_channel_hz']:.3g} Hz)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("designs", nargs="*", help="design names or json files (default: all in designs/)")
    parser.add_argument("--spill", type=float, default=1.0, help="spill length [s]")
    parser.add_argument("--n-signal", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--time-window", type=float, default=None,
                        help="fixed full width of the veto time window [ns] (default: +-3 sigma per plane)")
    args = parser.parse_args()
    names = args.designs or sorted(p.stem for p in dsg.DESIGN_DIR.glob("*.json"))
    cfg = reco.RecoConfig(spill_s=args.spill, seed=args.seed, time_window_ns=args.time_window)
    tr, inp = load_inputs()
    dt_pair = DEFAULT_CONFIG["time_window"] * 1e9 * 1e-9    # two-track coincidence window [s]
    results = []
    for name in names:
        r = evaluate(dsg.load(name), tr, inp, cfg, args.n_signal, dt_pair)
        report(r, cfg)
        results.append(r)
    out = paths.results_dir() / "reco" / f"results{tag(cfg)}.json"
    out.write_text(json.dumps(dict(config=vars(cfg), spill_s=args.spill, results=results), indent=1, default=float))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
