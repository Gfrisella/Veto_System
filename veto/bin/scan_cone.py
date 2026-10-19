#!/usr/bin/env python3
"""Scan the vertex-cone width (RecoConfig.cone): signal lost, combinatorial and muon-interaction backgrounds.

For each design and cone: run_reco.evaluate (40 k signal decays, the 196 736 pairs, scattering with tails) and
the muon interactions saved by bin/run_interactions.py (results/interactions/<design>_muon_interactions.npz)
through interactions.evaluate (veto levels a and b). Writes results/interactions/cone_scan.json.

  GPU_HOST=farm-gpu5 bin/remote.sh run bin/scan_cone.py --designs D6_tile2 E3_robust --cones 0 5 10 20 30 50
"""

import _bootstrap  # noqa: F401

import argparse
import json
from multiprocessing import Pool

import numpy as np

from run_interactions import load_npz
from run_reco import evaluate, load_inputs
from veto import design as dsg
from veto import interactions as ix
from veto import io, objective, paths, reco
from veto.config import DEFAULT_CONFIG

_S = {}


def _init(max_tracks=None):
    _S["max_tracks"] = max_tracks
    _S["tr"], _S["inp"] = load_inputs()
    _S["muons"] = io.open_store(paths.store_dir("design-optimization-dataset-gamma-2026-10-17"))[0]
    _S["stats"] = objective.sample_stats()


def job(arg):
    name, cone = arg
    d = dsg.load(name)
    cfg = reco.RecoConfig(seed=1, multiple_scattering=True, cone=cone * 1e-3)
    r = evaluate(d, _S["tr"], _S["inp"], cfg, 40_000, DEFAULT_CONFIG["time_window"], "target")
    s = objective.score(r, _S["stats"], objective.ObjectiveConfig(bkg_estimate="max"))
    inter, part = load_npz(paths.results_dir() / "interactions" / f"{name}_muon_interactions.npz")
    res = ix.evaluate(inter, part, d, _S["inp"], cfg, np.random.default_rng(1), muons=_S["muons"], max_tracks=_S["max_tracks"] if _S["max_tracks"] is not None else "cfg")
    tot = lambda k: sum(v.get(k, 0.0) for v in res.values())
    return dict(design=name, cone_mrad=cone, signal_lost=s["confusion"]["signal"]["FP"],
                signal_lost_err=s["confusion"]["signal"]["FP_err"], combinatorial_15y=s["bkg_events_15y"],
                mu_selected=tot("selected"), mu_after_a=tot("survive_a"), mu_after_b=tot("survive_b"),
                mu_after_b_by_target={k: v.get("survive_b", 0.0) for k, v in res.items() if v["selected"] > 0})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--designs", nargs="+", default=["D6_tile2", "E3_robust"])
    ap.add_argument("--cones", type=float, nargs="+", default=[0, 5, 10, 20, 30, 50], help="mrad")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-tracks", type=int, default=None, help="selection: at most this many tracks in SST1")
    args = ap.parse_args()
    jobs = [(d, c) for d in args.designs for c in args.cones]
    with Pool(min(args.workers, len(jobs)), initializer=_init, initargs=(args.max_tracks,)) as pool:
        rows = pool.map(job, jobs)
    print(f"{'design':<12}{'cone':>6}{'signal lost':>14}{'combinatorial':>15}{'mu selected':>13}{'after a':>10}{'after b':>10}   (per 15 years)")
    for r in rows:
        print(f"{r['design']:<12}{r['cone_mrad']:>6g}{100 * r['signal_lost']:>9.3f} ± {100 * r['signal_lost_err']:.2f}"
              f"{r['combinatorial_15y']:>13.2g}{r['mu_selected']:>13.3g}{r['mu_after_a']:>10.3g}{r['mu_after_b']:>10.3g}")
    out = paths.results_dir() / "interactions" / f"cone_scan{'' if args.max_tracks is None else f'_max{args.max_tracks}'}.json"
    json.dump(rows, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
