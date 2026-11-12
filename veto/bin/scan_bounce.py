#!/usr/bin/env python3
"""ISSUES A2 (11 Nov 2026): scan the bounce search (RecoConfig.bounce_r_slope, bounce_r0) per design / technology and
write the start value and allowed range of the optimiser knobs (results/reco/bounce_scan.json, read by
evolve/evaluate_designs.py bounce_knobs).

Per (design, slope, r0): run_reco.evaluate on the Nov-4 pair inputs (paths.RECO_INPUTS), common random numbers:
bounced SST1 muons vetoed (single muons, type sst1_bounced), all SST1 muons vetoed, signal lost, combinatorial after the
veto and the objective reward (veto.objective.score). Start = the best reward per design (ties: the smaller window);
range = [start / 2, 2 x start] for the slope, [r0 / 2, 2 x r0] (or 0.02 - 0.5 m when the best is the track's own radius).

    python3 bin/scan_bounce.py --designs D6_straw D6_fiber D6_tile2 --slopes 0 0.1 0.15 0.21 0.3 0.45 --r0 none 0.1 0.25
"""
import _bootstrap  # noqa: F401

import argparse
import json
from multiprocessing import Pool

from run_reco import evaluate, load_inputs
from veto import design as dsg
from veto import objective, paths, reco
from veto.config import DEFAULT_CONFIG

_S = {}


def _init(signal, n_signal):
    _S["tr"], _S["inp"] = load_inputs()
    _S["stats"] = objective.sample_stats()          # reads results/reco/<RECO_INPUTS> (the raw arrays)
    _S["signal"], _S["n_signal"] = signal, n_signal


def job(arg):
    name, slope, r0 = arg
    d = dsg.load(name)
    cfg = reco.RecoConfig(seed=1, multiple_scattering=True, bounce_search=slope > 0, bounce_r_slope=slope, bounce_r0=r0)
    r = evaluate(d, _S["tr"], _S["inp"], cfg, _S["n_signal"], DEFAULT_CONFIG["time_window"], _S["signal"])
    s = objective.score(r, _S["stats"])
    return dict(design=name, technologies=r.get("technologies"), bounce_r_slope=slope, bounce_r0=r0,
                bounced_vetoed=r["single"].get("SST1 bounced"), sst1_vetoed=r["single"].get("all SST1"),
                signal_lost=s["confusion"]["signal"]["FP"], signal_lost_err=s["confusion"]["signal"]["FP_err"],
                combinatorial_15y=s["bkg_events_15y"], reward=s["reward"], reward_err=s.get("reward_err"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--designs", nargs="+", default=["D6_straw", "D6_fiber", "D6_tile2"])
    ap.add_argument("--slopes", type=float, nargs="+", default=[0.0, 0.1, 0.15, 0.21, 0.3, 0.45])
    ap.add_argument("--r0", nargs="+", default=["none", "0.1", "0.25"], help="bounce base radius [m]; none = the track's own")
    ap.add_argument("--signal", choices=("eventcalc", "target"), default="eventcalc")
    ap.add_argument("--n-signal", type=int, default=40_000)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    r0s = [None if x.lower() == "none" else float(x) for x in a.r0]
    jobs = [(d, s, r) for d in a.designs for s in a.slopes for r in (r0s if s > 0 else [None])]
    with Pool(min(a.workers, len(jobs)), initializer=_init, initargs=(a.signal, a.n_signal)) as pool:
        rows = pool.map(job, jobs)
    print(f"{'design':<12}{'slope':>6}{'r0':>6}{'bounced vetoed':>16}{'SST1 vetoed':>13}{'signal lost':>13}{'comb./15y':>11}{'reward':>9}")
    for x in rows:
        print(f"{x['design']:<12}{x['bounce_r_slope']:>6g}{str(x['bounce_r0']):>6}{100 * x['bounced_vetoed']:>15.2f}%"
              f"{100 * x['sst1_vetoed']:>12.3f}%{100 * x['signal_lost']:>12.2f}%{x['combinatorial_15y']:>11.2g}{x['reward']:>9.3f}")
    best = {}
    for d in a.designs:
        rr = sorted((x for x in rows if x["design"] == d), key=lambda x: (-round(x["reward"], 4), x["bounce_r_slope"]))
        best[d] = rr[0]
    slopes = [b["bounce_r_slope"] for b in best.values() if b["bounce_r_slope"] > 0] or [0.21]
    s0 = sorted(slopes)[len(slopes) // 2]
    r0v = [b["bounce_r0"] for b in best.values() if b["bounce_r0"] is not None]
    r00 = sorted(r0v)[len(r0v) // 2] if r0v else None
    ranges = dict(bounce_r_slope=dict(start=s0, lo=s0 / 2, hi=2 * s0),
                  bounce_r0=dict(start=r00, lo=(r00 / 2 if r00 else 0.02), hi=(2 * r00 if r00 else 0.5)))
    out = paths.results_dir() / "reco" / "bounce_scan.json"
    json.dump(dict(rows=rows, best=best, ranges=ranges, inputs=paths.RECO_INPUTS, signal=a.signal), open(out, "w"), indent=1)
    print(f"best per design: { {d: (b['bounce_r_slope'], b['bounce_r0']) for d, b in best.items()} } -> start {ranges}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
