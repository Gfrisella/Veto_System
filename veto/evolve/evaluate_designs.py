#!/usr/bin/env python3
"""Evaluate a batch of veto designs in parallel (run on the cluster) and print one report per design.

Input: a JSON list of designs (the designs/*.json format) on stdin or --input. Output: the last line
of stdout is a JSON list of reports, one per design, in the same order:

  name, error (or None), notes (what design.validate changed: cut at the cavern wall, ...)
  result       the run_reco.py evaluate() dict (what veto.objective scores): background, signal,
               fom, single, pair_muons, planes (rates and limits), cost_kchf, n_channels, ...
  planes       per plane, geometry for the LLM: name, kind, side, z, reach, y range, technology
  seconds
The full chain of run_reco.py (196 736 background pairs, --n-signal signal decays, 200 k single
muons) with multiple scattering in the PVC walls, the UBT and the fins.

  ssh ... 'python3 evolve/evaluate_designs.py --workers 16' < designs.json
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "lib"), str(HERE.parent / "bin")]

from run_reco import evaluate, load_inputs  # noqa: E402
from veto import design as dsg  # noqa: E402
from veto import reco  # noqa: E402
from veto.config import DEFAULT_CONFIG  # noqa: E402

_STATE = {}


def _init(n_signal, seed):
    _STATE["tr"], _STATE["inp"] = load_inputs()
    _STATE["n_signal"], _STATE["seed"] = n_signal, seed


def report(raw):
    t0 = time.time()
    name = raw.get("name", "unnamed")
    try:
        design = dsg.from_dict(raw)
        cfg = reco.RecoConfig(seed=_STATE["seed"], multiple_scattering=True)
        r = evaluate(design, _STATE["tr"], _STATE["inp"], cfg, _STATE["n_signal"], DEFAULT_CONFIG["time_window"], "target")
    except Exception as e:  # noqa: BLE001  (an invalid design is a normal outcome here)
        return dict(name=name, error=f"{type(e).__name__}: {e}")
    planes = [dict(name=p.name, kind=p.kind, side={-1: "right", 1: "left", 0: "-"}[p.sign], z=p.z,
                   reach_m=round(p.u_max, 3), y_lo=p.y_lo, y_hi=p.y_hi, technology=p.technology,
                   area_m2=round(p.area, 2)) for p in design.planes]
    r.pop("window_containment", None)
    r["background"].pop("timing", None)
    return dict(name=name, error=None, notes=design.notes, result=r, planes=planes, seconds=round(time.time() - t0, 1))

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="-")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n-signal", type=int, default=40_000)
    ap.add_argument("--seed", type=int, default=1, help="same seed for every design: common random numbers")
    args = ap.parse_args()
    designs = json.load(sys.stdin if args.input == "-" else open(args.input))
    with Pool(min(args.workers, len(designs)), initializer=_init, initargs=(args.n_signal, args.seed)) as pool:
        reports = pool.map(report, designs)
    print(json.dumps(reports, default=lambda o: None if o != o else float(o)))


if __name__ == "__main__":
    main()
