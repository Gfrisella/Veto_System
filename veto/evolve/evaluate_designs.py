#!/usr/bin/env python3
"""Evaluate a batch of veto designs in parallel (run on the cluster) and print one report per design.

Input: a JSON list of designs (the designs/*.json format) on stdin or --input. Output: the last line
of stdout is a JSON list of reports, one per design, in the same order:

  name, error (or None), notes (what design.validate changed: cut at the cavern wall, ...)
  result       the run_reco.py evaluate() dict (what veto.objective scores): background, signal,
               fom, single, pair_muons, planes (rates and limits), cost_kchf, n_channels, ...
  planes       per plane, geometry for the LLM: name, kind, side, z, reach, y range, technology
  seconds
The full chain of run_reco.py (the background pairs, --n-signal signal decays, 200 k single muons) with multiple
scattering (tails, helium and air, exact PVC chords), the selection of RecoConfig (measured IP < 0.1 m, <= 2
tracks), the vertex cone of the design ("detector": {"cone_mrad": ...}, 0 - 50) and, in result["interactions"],
the muon- and neutrino-interaction backgrounds after the veto (bin/prepare_interactions.py caches).

  ssh ... 'python3 evolve/evaluate_designs.py --workers 16' < designs.json
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "lib"), str(HERE.parent / "bin")]

from run_reco import evaluate, load_inputs  # noqa: E402
from veto import design as dsg  # noqa: E402
from veto import interactions as ix  # noqa: E402
from veto import paths  # noqa: E402
from veto import reco  # noqa: E402
from veto.config import DEFAULT_CONFIG  # noqa: E402

_STATE = {}


CONE_MAX_MRAD = 50.0
# ISSUES A2 (11 Nov 2026): the bounce search is a design knob ("detector": {"bounce_r_slope", "bounce_r0"}), kept inside
# a range around the start value of bin/scan_bounce.py (results/reco/bounce_scan.json); defaults if no scan yet.
BOUNCE_DEFAULT = dict(bounce_r_slope=dict(start=0.21, lo=0.05, hi=0.5), bounce_r0=dict(start=None, lo=0.02, hi=0.5))


def bounce_ranges():
    f = paths.results_dir() / "reco" / "bounce_scan.json"
    try:
        return json.load(open(f))["ranges"]
    except (OSError, KeyError, ValueError):
        return BOUNCE_DEFAULT


def bounce_knobs(raw):
    """(bounce_r_slope, bounce_r0) of a design: its "detector" values clamped to the scan range, else the scan start."""
    det, rg = raw.get("detector") or {}, bounce_ranges()
    out = []
    for k in ("bounce_r_slope", "bounce_r0"):
        r = rg.get(k, BOUNCE_DEFAULT[k])
        v = det.get(k, r["start"])
        out.append(None if v is None else float(min(max(float(v), r["lo"]), r["hi"])))
    return out


# Interaction caches per kind (17 Nov 2026): vessel muon / neutrino interactions of the Nov-4 sample, the V0-enriched MS7
# samples (_v0statk) and the SBT-material term with the shell energy loss (D19, _eloss). --prep-tag "<default>[,kind=tag...]".
PREP_TAGS = "_nov4full_d19,ms7=_v0statk,sbt=_nov4full_eloss"

# Physics model of the SBT (D19, Guglielmo 17 Nov 2026): fixed for every design, as in R2 / R3; a design chooses only the
# layout ("ring_set" / "rings" / "faces"), "threshold_mev" and "mode". Without this a design could switch the model off.
SBT_LAYOUT_KEYS = ("ring_set", "rings", "faces", "threshold_mev", "mode")
SBT_MODEL = {k: v for k, v in json.load(open(Path(__file__).resolve().parents[1] / "designs" / "R2_sbt_ubt.json"))["sbt"].items()
             if k not in SBT_LAYOUT_KEYS}


def prep_tags(spec):
    """'<default>[,kind=tag,...]' -> {kind: tag} for muon, nu, ms7, sbt."""
    parts = [x for x in (spec or "").split(",")]
    default = parts[0] if parts and "=" not in parts[0] else ""
    out = {k: default for k in ("muon", "nu", "ms7", "sbt")}
    out.update(dict(x.split("=", 1) for x in parts if "=" in x))
    return out


def fix_model(raw):
    """The design as evaluated: its SBT with the fixed physics model (SBT_MODEL)."""
    if raw.get("sbt"):
        raw = dict(raw, sbt={**{k: v for k, v in raw["sbt"].items() if k in SBT_LAYOUT_KEYS}, **SBT_MODEL})
    return raw


def _init(n_signal, seed, inputs=paths.RECO_INPUTS, prep_tag=PREP_TAGS, signal="eventcalc"):
    _STATE["tr"], _STATE["inp"] = load_inputs(inputs)
    _STATE["n_signal"], _STATE["seed"], _STATE["signal"] = n_signal, seed, signal
    _STATE["prep"] = {}
    for kind, tag in prep_tags(prep_tag).items():    # bin/prepare_interactions.py (sbt: bin/sbt_prepare.py, D19): selected events
        f = paths.results_dir() / "interactions" / f"prepared_{kind}{tag}.npz"
        if f.exists():
            _STATE["prep"][kind] = dict(np.load(f))


def report(raw):
    t0 = time.time()
    name = raw.get("name", "unnamed")
    try:
        design = dsg.from_dict(fix_model(raw))
        cone = float(min(max((raw.get("detector") or {}).get("cone_mrad", 0.0), 0.0), CONE_MAX_MRAD))
        slope, r0 = bounce_knobs(raw)
        cfg = reco.RecoConfig(seed=_STATE["seed"], multiple_scattering=True, cone=cone * 1e-3, bounce_r_slope=slope, bounce_r0=r0)
        r = evaluate(design, _STATE["tr"], _STATE["inp"], cfg, _STATE["n_signal"], DEFAULT_CONFIG["time_window"], _STATE["signal"])
        r["cone_mrad"] = cone
        r["bounce_r_slope"], r["bounce_r0"] = slope, r0
        r["interactions"] = {kind: ix.veto(prep, design, _STATE["inp"], cfg, np.random.default_rng(_STATE["seed"] + 7))
                             for kind, prep in _STATE["prep"].items()}
    except Exception as e:  # noqa: BLE001  (an invalid design is a normal outcome here)
        return dict(name=name, error=f"{type(e).__name__}: {e}")
    planes = [dict(name=p.name, kind=p.kind, side={-1: "right", 1: "left", 0: "-"}[p.sign], z=p.z,
                   reach_m=round(p.u_off + p.u_max, 3), u_from=round(p.u_off, 3), y_lo=p.y_lo, y_hi=p.y_hi,
                   technology=p.technology, n_modules=p.n_modules, active=round(p.active_fraction, 3),
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
    ap.add_argument("--inputs", default=paths.RECO_INPUTS, help="results/reco file (Nov-4, charge-fixed; *_xrun for cross-run pairs)")
    ap.add_argument("--signal", choices=("eventcalc", "target"), default="eventcalc",
                    help="ISSUES B4 (11 Nov 2026): the equal-weight EventCalc benchmark set (default) or the old toy (target)")
    ap.add_argument("--prep-tag", default=PREP_TAGS, help="interaction caches results/interactions/prepared_<kind><tag>.npz: "
                    "'<default>[,kind=tag,...]'")
    args = ap.parse_args()
    designs = json.load(sys.stdin if args.input == "-" else open(args.input))
    with Pool(min(args.workers, len(designs)), initializer=_init, initargs=(args.n_signal, args.seed, args.inputs, args.prep_tag, args.signal)) as pool:
        reports = pool.map(report, designs)
    print(json.dumps(reports, default=lambda o: None if o != o else float(o)))


if __name__ == "__main__":
    main()
