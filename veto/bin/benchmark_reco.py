#!/usr/bin/env python3
"""Time every stage of the veto simulation + reconstruction, per event, for a few designs.

Stages (algorithm A, numpy, one core):
  scatter      multiple-scattering kicks of the events' muons (only with scattering on)
  own hits     the muons' own UBT / fin hits: crossings, efficiency, cells, time
  random hits  accidental hits from the rate maps (Poisson per plane, digitized)
  measure      smeared spectrometer tracks
  predict      backward predictions on every plane upstream of the vertex
  match        sort hits, binary search per (event, plane), space + time window
Background events are muon pairs (2 tracks), as in run_reco.py.

  bin/remote.sh run bin/benchmark_reco.py                       # on the cluster
  bin/benchmark_reco.py --designs D2_default --n-events 20000
"""

import _bootstrap  # noqa: F401

import argparse
import time

import numpy as np

from run_reco import load_inputs, take
from veto import design as dsg
from veto import reco, scattering


def bench(design, tr, inp, n_events, ms, seed=1):
    cfg = reco.RecoConfig(multiple_scattering=ms)
    rng = np.random.default_rng(seed)
    cand = inp["cand"][:n_events]
    sub = take(tr, cand.ravel())
    event = np.repeat(np.arange(len(cand)), 2)
    vz = np.repeat(inp["cand_vertex"][:len(cand), 2], 2)
    t = {}
    t0 = time.perf_counter()
    if ms:
        k = scattering.kicks(sub, design, rng)
        sub_true, sub = dict(sub, _kicks=k), scattering.scattered_tracks(sub, k)
    else:
        sub_true = sub
    t["scatter"] = time.perf_counter() - t0
    t0 = time.perf_counter(); own = reco.own_hits(sub_true, event, design, rng); t["own hits"] = time.perf_counter() - t0
    window = reco.time_window(design, cfg) + reco.ACC_MARGIN_NS
    t0 = time.perf_counter(); acc = reco.accidental_hits(design, inp, len(cand), window, cfg, rng)
    t["random hits"] = time.perf_counter() - t0
    acc["track"] = np.full(len(acc["event"]), -1)
    hits = {k: np.concatenate([own[k], acc[k]]) for k in own}
    for k in ("event", "plane", "track"):
        hits[k] = hits[k].astype(np.int64)
    t0 = time.perf_counter(); meas = reco.measure(sub, cfg, rng); t["measure"] = time.perf_counter() - t0
    t0 = time.perf_counter(); q = reco.predictions(meas, event, vz, design, cfg); t["predict"] = time.perf_counter() - t0
    t0 = time.perf_counter(); found, n_cmp, _ = reco.match(q, hits, design, cfg); t["match"] = time.perf_counter() - t0
    explained = np.bincount(q["track"][found], minlength=len(event)) >= 1
    vetoed = np.zeros(len(cand), bool); vetoed[event[explained]] = True
    return t, len(hits["event"]) / len(cand), vetoed.mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--designs", nargs="*", default=["D0_minimal", "D1_wide_ubt", "D2_default", "D4_dense", "D6_tile2"])
    parser.add_argument("--n-events", type=int, default=50_000)
    args = parser.parse_args()
    tr, inp = load_inputs()
    stages = ["scatter", "own hits", "random hits", "measure", "predict", "match"]
    print(f"{args.n_events:,} background events (muon pairs) per row; microseconds per event, one core\n")
    print(f"{'design':<13}{'MS':>4}{'hits/ev':>9}" + "".join(f"{s:>12}" for s in stages) + f"{'total':>10}{'sim':>8}{'reco':>8}{'vetoed':>9}")
    for name in args.designs:
        d = dsg.load(name)
        for ms in (False, True):
            t, hpe, frac = bench(d, tr, inp, args.n_events, ms)
            us = {k: 1e6 * v / args.n_events for k, v in t.items()}
            sim = us["scatter"] + us["own hits"] + us["random hits"]
            rec = us["measure"] + us["predict"] + us["match"]
            print(f"{name:<13}{'on' if ms else 'off':>4}{hpe:>9.0f}" + "".join(f"{us[s]:>12.2f}" for s in stages)
                  + f"{sim + rec:>10.1f}{sim:>8.1f}{rec:>8.1f}{100 * frac:>8.3f}%", flush=True)


if __name__ == "__main__":
    main()
