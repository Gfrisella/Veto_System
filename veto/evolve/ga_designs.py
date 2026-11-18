#!/usr/bin/env python3
"""Genetic-algorithm search over veto designs: no LLM, no tokens (Guglielmo 18 Nov 2026).

Same evaluation, objective and bookkeeping as evolve_designs.py (designs scored on the farm with evaluate_designs.py,
reward of objective.py, results/evolve/ga_<date>/history.jsonl + best.json + confirm.json), but the children are made
by mutation and crossover of a fixed list of named genes instead of being proposed by a language model:

  SBT        on / off, which of the ten rings (10 bits), threshold_mev 10 .. 60
  UBT        straw y_lo -2.0 .. -1.35, y_hi 1.35 .. 2.0, straw_t0_sigma_ns (3 ns) on / off, segment_cut on / off
  UBT tiles  plane half_x 1.0 .. 3.0, y_lo -3.0 .. -1.35, y_hi 1.35 .. 3.0; flame blocks from |x| = xin 0.5 .. 1.5 to half_x,
             |y| < fy 0.2 .. 1.2; tile type of the flame blocks (tile2 2 x 2 cm / tile 4 x 4 cm) and of the rest (18 Nov 2026,
             Guglielmo: "it can change also the UBT tiles size"); layout as design.default_tile_regions
  V0 rule    on / off, theta 5 .. 50 mrad, r0 2 .. 15 cm, early_ns 1 .. 5, min_hits 1 / 2, p_max on / off and 5 .. 60 GeV,
             theta_p on / off and 0.05 .. 0.6, early_tof on / off
  strips     on / off, which of the three R3 layers (3 bits) and the eight R3 stations (8 bits), theta 3 .. 30 mrad,
             rule point / vertex, time_window_ns 0.5 .. 5
  detector   bounce_r_slope 0.105 .. 0.42 (ISSUES A2 range)
The SBT physics model stays as in R1 / R2 / R3 (evaluate_designs.fix_model); no fins.

Each generation: tournament selection (size 3) on the population, uniform crossover (prob --crossover), then mutation
(every gene with prob --mutation; reals take a Gaussian step of --sigma x their range, at least one gene changes).
Children above --max-cost (default: the budget, 333.5 drachmas) or rejected by the design loader are dropped here, the
rest go to ONE Slurm job per generation (the same evaluate_slurm as evolve_designs). (mu + lambda) truncation on the
reward. All generations use the same random seed (common random numbers: differences between designs are not blurred by
seed noise); the best --confirm-top are re-scored with --confirm-seeds at the end.

  evolve/ga_designs.py --generations 8 --children 12            # seeds R1, R2, R3 (R3 is over budget: kept as a parent
                                                                 # for its genes, its children must fit the budget)
  evolve/ga_designs.py --resume results_rerun/evolve/ga_<run>    # continue a run from its history.jsonl
"""

import argparse
import copy
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "lib")]

import evolve_designs as ev  # noqa: E402  (evaluation on Slurm, de-duplication, objective configuration)
import objective  # noqa: E402
from evaluate_designs import fix_model  # noqa: E402
from veto import budget  # noqa: E402
from veto import design as dsg  # noqa: E402
from veto import geometry as geo  # noqa: E402
from veto import paths  # noqa: E402

print = ev.print

# R3's strip layout: the candidate positions of the strip genes
STRIP_LAYERS = [29.39, 29.59, 29.79]
STRIP_STATIONS = [[-0.9, -0.15], [-0.9, 0.15], [-1.2, -0.15], [-1.2, 0.15],
                  [0.9, -0.15], [0.9, 0.15], [1.2, -0.15], [1.2, 0.15]]

# gene: (kind, lo, hi) for reals, ("bool",) for switches, ("choice", [..]) for categories
GENES = {
    "sbt_on": ("bool",), **{f"ring{k}": ("bool",) for k in range(10)}, "sbt_thr": ("real", 10.0, 60.0),
    "ubt_ylo": ("real", -2.0, -1.35), "ubt_yhi": ("real", 1.35, 2.0), "straw_t0": ("bool",), "segment_cut": ("bool",),
    "tiles_hx": ("real", 1.0, 3.0), "tiles_ylo": ("real", -3.0, -1.35), "tiles_yhi": ("real", 1.35, 3.0),
    "tiles_xin": ("real", 0.5, 1.5), "tiles_fy": ("real", 0.2, 1.2),
    "tiles_fine": ("choice", ["tile2", "tile"]), "tiles_coarse": ("choice", ["tile", "tile2"]),
    "v0_on": ("bool",), "v0_theta": ("real", 0.005, 0.05), "v0_r0": ("real", 0.02, 0.15), "v0_early_ns": ("real", 1.0, 5.0),
    "v0_min_hits": ("choice", [1, 2]), "v0_pmax_on": ("bool",), "v0_pmax": ("real", 5.0, 60.0),
    "v0_thetap_on": ("bool",), "v0_thetap": ("real", 0.05, 0.6), "v0_early_tof": ("bool",),
    "strips_on": ("bool",), **{f"layer{k}": ("bool",) for k in range(3)}, **{f"station{k}": ("bool",) for k in range(8)},
    "strip_theta": ("real", 0.003, 0.03), "strip_rule": ("choice", ["point", "vertex"]), "strip_tw": ("real", 0.5, 5.0),
    "bounce_slope": ("real", 0.105, 0.42),
}
DEFAULTS = dict(sbt_on=False, **{f"ring{k}": True for k in range(10)}, sbt_thr=30.0, ubt_ylo=-1.35, ubt_yhi=1.35,
                straw_t0=False, segment_cut=False,
                tiles_hx=geo.UBT_TILES_HALF_X, tiles_ylo=-geo.UBT_TILES_HALF_Y, tiles_yhi=geo.UBT_TILES_HALF_Y,
                tiles_xin=geo.UBT_TILES_FLAME_X_IN, tiles_fy=geo.UBT_TILES_FLAME_Y, tiles_fine="tile2", tiles_coarse="tile",
                v0_on=False, v0_theta=0.025, v0_r0=0.05, v0_early_ns=3.0,
                v0_min_hits=1, v0_pmax_on=False, v0_pmax=20.0, v0_thetap_on=False, v0_thetap=0.25, v0_early_tof=False,
                strips_on=False, **{f"layer{k}": True for k in range(3)}, **{f"station{k}": True for k in range(8)},
                strip_theta=0.01, strip_rule="point", strip_tw=1.5, bounce_slope=0.21)


def encode(d):
    """Design JSON (R1 / R2 / R3 style) -> genes."""
    g = dict(DEFAULTS)
    ubt = d.get("ubt") or {}
    g.update(ubt_ylo=ubt.get("y_lo", -1.35), ubt_yhi=ubt.get("y_hi", 1.35),
             straw_t0=bool(ubt.get("straw_t0_sigma_ns")), segment_cut=bool(ubt.get("segment_cut")))
    t = d.get("ubt_tiles") or {}
    for gk, tk in (("tiles_hx", "half_x"), ("tiles_ylo", "y_lo"), ("tiles_yhi", "y_hi"), ("tiles_xin", "flame_x_in"),
                   ("tiles_fy", "flame_y"), ("tiles_fine", "flame_technology"), ("tiles_coarse", "technology")):
        if tk in t:
            g[gk] = t[tk]
    s = d.get("sbt")
    if s:
        rings = s.get("ring_set", list(range(10)))
        g.update(sbt_on=True, sbt_thr=s.get("threshold_mev", 30.0), **{f"ring{k}": k in rings for k in range(10)})
    det = d.get("detector") or {}
    v = det.get("v0_rule")
    if v:
        g.update(v0_on=True, v0_theta=v.get("theta", 0.025), v0_r0=v.get("r0", 0.05), v0_early_ns=v.get("early_ns", 3.0),
                 v0_min_hits=v.get("min_hits", 1), v0_pmax_on=v.get("p_max") is not None, v0_pmax=v.get("p_max") or 20.0,
                 v0_thetap_on=v.get("theta_p") is not None, v0_thetap=v.get("theta_p") or 0.25,
                 v0_early_tof=bool(v.get("early_tof")))
    if "bounce_r_slope" in det:
        g["bounce_slope"] = det["bounce_r_slope"]
    ss = d.get("shield_strips")
    if ss:
        g.update(strips_on=True, strip_theta=ss.get("theta", 0.01), strip_rule=ss.get("rule", "point"),
                 strip_tw=ss.get("time_window_ns", 1.5),
                 **{f"layer{k}": z in ss.get("layers", []) for k, z in enumerate(STRIP_LAYERS)},
                 **{f"station{k}": list(c) in [list(x) for x in ss.get("stations", [])] for k, c in enumerate(STRIP_STATIONS)})
    return g


def decode(g, name):
    """Genes -> design JSON for evaluate_designs.py (rounded, so that de-duplication sees near-identical designs)."""
    r = lambda x, n=3: round(float(x), n)  # noqa: E731
    d = {"name": name, "description": "", "frame": "fairship-2026", "ubt_layout": "2026-11-11",
         "ubt": {"y_lo": r(g["ubt_ylo"], 2), "y_hi": r(g["ubt_yhi"], 2)}, "fins": [], "detector": {}}
    hx, ylo, yhi = r(g["tiles_hx"], 2), r(g["tiles_ylo"], 2), r(g["tiles_yhi"], 2)
    xin = r(min(g["tiles_xin"], hx - 0.1), 2)                 # flame blocks inside the plane
    fy = r(min(g["tiles_fy"], -ylo - 0.1, yhi - 0.1), 2)
    d["ubt_tiles"] = {"half_x": hx, "y_lo": ylo, "y_hi": yhi, "technology": g["tiles_coarse"],
                      "regions": dsg.default_tile_regions(hx, ylo, yhi, xin, fy, g["tiles_fine"], g["tiles_coarse"])}
    if g["straw_t0"]:
        d["ubt"]["straw_t0_sigma_ns"] = 3.0
    if g["segment_cut"]:
        d["ubt"]["segment_cut"] = True
    rings = [k for k in range(10) if g[f"ring{k}"]]
    if g["sbt_on"] and rings:
        d["sbt"] = {"ring_set": rings, "threshold_mev": r(g["sbt_thr"], 1), "mode": "pointing"}
    if g["v0_on"]:
        v = {"theta": r(g["v0_theta"], 4), "r0": r(g["v0_r0"]), "early_ns": r(g["v0_early_ns"], 2),
             "min_hits": g["v0_min_hits"], "early_tof": bool(g["v0_early_tof"])}
        if g["v0_pmax_on"]:
            v["p_max"] = r(g["v0_pmax"], 1)
        if g["v0_thetap_on"]:
            v["theta_p"] = r(g["v0_thetap"])
        d["detector"]["v0_rule"] = v
    d["detector"]["bounce_r_slope"] = r(g["bounce_slope"])
    layers = [z for k, z in enumerate(STRIP_LAYERS) if g[f"layer{k}"]]
    stations = [c for k, c in enumerate(STRIP_STATIONS) if g[f"station{k}"]]
    if g["strips_on"] and layers and stations:
        d["shield_strips"] = {"layers": layers, "stations": stations, "size": 0.3, "rule": g["strip_rule"],
                              "theta": r(g["strip_theta"], 4), "time_window_ns": r(g["strip_tw"], 2)}
    d["description"] = describe(d)
    return d


def describe(d):
    s = d.get("sbt")
    v = (d.get("detector") or {}).get("v0_rule")
    ss = d.get("shield_strips")
    return ("GA: " + (f"SBT rings {s['ring_set']} thr {s['threshold_mev']} MeV" if s else "no SBT")
            + f", UBT straws {d['ubt']['y_lo']}..{d['ubt']['y_hi']}"
            + (", t0" if d["ubt"].get("straw_t0_sigma_ns") else "")
            + f", tiles |x|<{d['ubt_tiles']['half_x']} y {d['ubt_tiles']['y_lo']}..{d['ubt_tiles']['y_hi']} flames "
              f"{d['ubt_tiles']['regions'][2]['x']}x{d['ubt_tiles']['regions'][2]['y']} {d['ubt_tiles']['regions'][2]['technology']}"
              f" rest {d['ubt_tiles']['technology']}" + (", seg" if d["ubt"].get("segment_cut") else "")
            + (f", V0 rule {v}" if v else ", no V0 rule")
            + (f", strips {len(ss['layers'])} x {len(ss['stations'])} {ss['rule']}" if ss else ", no strips")
            + f", bounce {d['detector'].get('bounce_r_slope')}")


def mutate(g, rng, p_mut, sigma):
    g = dict(g)
    changed = False
    while not changed:
        for k, spec in GENES.items():
            if rng.random() >= p_mut:
                continue
            if spec[0] == "bool":
                g[k] = not g[k]
            elif spec[0] == "choice":
                g[k] = rng.choice([c for c in spec[1] if c != g[k]])
            else:
                lo, hi = spec[1], spec[2]
                g[k] = min(max(g[k] + rng.gauss(0.0, sigma * (hi - lo)), lo), hi)
            changed = True
    return g


def crossover(a, b, rng):
    """Uniform crossover; the ring / layer / station bits of a block travel together half of the time."""
    blocks = {"tiles": [k for k in GENES if k.startswith("tiles_")], "ring": [f"ring{k}" for k in range(10)], "layer": [f"layer{k}" for k in range(3)],
              "station": [f"station{k}" for k in range(8)]}
    child = {}
    for k in GENES:
        child[k] = a[k] if rng.random() < 0.5 else b[k]
    for keys in blocks.values():
        if rng.random() < 0.5:
            src = a if rng.random() < 0.5 else b
            child.update({k: src[k] for k in keys})
    return child


def cost_drachma(d):
    return budget.drachma(dsg.from_dict(fix_model(d)).cost)


def tournament(pop, rng, k=3):
    return max(rng.sample(pop, min(k, len(pop))), key=lambda c: c["score"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--children", type=int, default=12, help="children per generation (= workers of the Slurm job)")
    ap.add_argument("--population", type=int, default=12)
    ap.add_argument("--seeds", nargs="*", default=["R1_ubt_only", "R2_sbt_ubt", "R3_sbt_ubt_strips"])
    ap.add_argument("--crossover", type=float, default=0.5)
    ap.add_argument("--mutation", type=float, default=0.08, help="probability per gene")
    ap.add_argument("--sigma", type=float, default=0.15, help="Gaussian step of a real gene, as a fraction of its range")
    ap.add_argument("--max-cost", type=float, default=None, help="drachmas (default: the budget)")
    ap.add_argument("--n-signal", type=int, default=100_000)
    ap.add_argument("--confirm-top", type=int, default=5)
    ap.add_argument("--confirm-seeds", type=int, nargs="*", default=[2, 3])
    ap.add_argument("--bkg-estimate", choices=["max", "factorised", "direct"], default="direct")
    ap.add_argument("--host", default="slurm")
    ap.add_argument("--proxy-port", default="2226")
    ap.add_argument("--inputs", default=paths.RECO_INPUTS)
    ap.add_argument("--prep-tag", default=ev.EVAL_DATA["prep_tag"])
    ap.add_argument("--resume", default=None, help="an earlier ga_ run directory: continue from its history")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="make and check generation-1 children, evaluate nothing")
    args = ap.parse_args()

    ev.EVAL_DATA.update(inputs=args.inputs, prep_tag=args.prep_tag)
    objective.INPUTS = args.inputs
    objective.configure(bkg_estimate=args.bkg_estimate)
    rng = random.Random(args.seed)
    max_cost = args.max_cost if args.max_cost is not None else budget.budget_drachma()

    run = Path(args.resume) if args.resume else paths.results_dir() / "evolve" / datetime.now().strftime("ga_%Y%m%d_%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(vars(args), indent=2))
    history = run / "history.jsonl"

    def record(c):
        with history.open("a") as fh:
            fh.write(json.dumps(c, default=float) + "\n")

    def evaluate(cands, seed=1):
        reps = ev.evaluate_remote([c["design"] for c in cands], args.proxy_port, args.host, len(cands), args.n_signal, seed)
        for c, r in zip(cands, reps):
            c["report"] = r
            c["score"], _ = objective.score(r)
            c["score_err"] = ev.reward_err(r)
            c["cost_drachma"] = c.get("cost_drachma")

    if args.resume and history.exists():
        hist = [json.loads(line) for line in history.open()]
        hist = [c for c in hist if c.get("score") is not None]
        start = max(c["gen"] for c in hist) + 1
        population = sorted((c for c in hist if math.isfinite(c["score"])), key=lambda c: -c["score"])[:args.population]
        seen = {ev.design_key(c["design"]) for c in hist}
        print(f"resume {run}: {len(hist)} candidates, next generation {start}")
    else:
        seeds = []
        for n in args.seeds:
            raw = json.loads((dsg.DESIGN_DIR / f"{n}.json").read_text())
            seeds.append(dict(id=n, gen=0, design=raw, genes=encode(raw), cost_drachma=cost_drachma(raw), parent=None))
        if args.dry_run:
            population, start = [dict(c, score=0.0) for c in seeds], 1
        else:
            t0 = time.time()
            evaluate(seeds)
            for c in seeds:
                record(c)
            population = sorted(seeds, key=lambda c: -c["score"])
            print(f"gen 0  {len(seeds)} seeds in {time.time() - t0:.0f} s:  "
                  + "  ".join(f"{c['id']}={c['score']:.2f} ({c['cost_drachma']:.0f} dr)" for c in population))
            start = 1
        seen = {ev.design_key(decode(c["genes"], "")) for c in seeds}   # a child equal to a seed is a duplicate
    best = population[0]
    (run / "best.json").write_text(json.dumps(best["design"], indent=1))

    for gen in range(start, start + args.generations):
        t0 = time.time()
        children, tries = [], 0
        while len(children) < args.children and tries < 200 * args.children:
            tries += 1
            a = tournament(population, rng)
            g = crossover(a["genes"], tournament(population, rng)["genes"], rng) if rng.random() < args.crossover else a["genes"]
            g = mutate(g, rng, args.mutation, args.sigma)
            d = decode(g, f"ga{gen}c{len(children)}")
            key = ev.design_key(dict(d, name=""))
            if key in seen or ev.valid_locally(d):
                continue
            cost = cost_drachma(d)
            if cost > max_cost * 1.0001:
                continue
            seen.add(key)
            children.append(dict(id=d["name"], gen=gen, design=d, genes=g, cost_drachma=cost, parent=a["id"]))
        if args.dry_run:
            for c in children:
                print(f"  {c['id']}  {c['cost_drachma']:6.1f} dr  <- {c['parent']}: {c['design']['description']}")
            print(f"dry run: {len(children)} children in {tries} tries (max cost {max_cost:.1f} drachmas)")
            return
        evaluate(children)
        for c in children:
            record(c)
        population = sorted(population + children, key=lambda c: -c["score"])[:args.population]
        if population[0]["score"] > best["score"]:
            best = population[0]
            (run / "best.json").write_text(json.dumps(best["design"], indent=1))
        ok = [c for c in children if math.isfinite(c["score"])]
        print(f"gen {gen}  {len(ok)}/{len(children)} scored  best {best['id']}={best['score']:.3f}  "
              f"pop mean {np.mean([c['score'] for c in population if math.isfinite(c['score'])]):.3f}  {time.time() - t0:.0f} s")
        for c in sorted(ok, key=lambda c: -c["score"])[:3]:
            print(f"    {c['score']:9.3f} +- {c['score_err']:.3f}  {c['cost_drachma']:6.1f} dr  {c['id']}: {c['design']['description']}")

    hist = [json.loads(line) for line in history.open()]
    top = sorted((c for c in hist if c.get("score") is not None and math.isfinite(c["score"])), key=lambda c: -c["score"])[:args.confirm_top]
    if top and args.confirm_seeds:
        scores = {c["id"]: [c["score"]] for c in top}
        for s in args.confirm_seeds:
            cands = [dict(id=c["id"], design=c["design"]) for c in top]
            evaluate(cands, s)
            for c in cands:
                scores[c["id"]].append(c["score"])
        rows = sorted(({"id": c["id"], "scores": scores[c["id"]], "mean": float(np.mean(scores[c["id"]])),
                        "spread": float(np.std(scores[c["id"]])), "cost_drachma": c["cost_drachma"], "design": c["design"]}
                       for c in top), key=lambda x: -x["mean"])
        (run / "confirm.json").write_text(json.dumps(rows, indent=1, default=float))
        print(f"\nre-evaluated with seeds {args.confirm_seeds}:")
        for x in rows:
            print(f"  {x['mean']:9.3f} (spread {x['spread']:.3f})  {x['cost_drachma']:6.1f} dr  {x['id']}  {np.round(x['scores'], 3).tolist()}")
        best = next(c for c in top if c["id"] == rows[0]["id"])
        (run / "best.json").write_text(json.dumps(best["design"], indent=1))
    print(f"\nbest: {best['id']} score {best['score']:.3f} -> {run / 'best.json'}")
    print(json.dumps(ev.compact(best["report"]), indent=1)[:3000])


if __name__ == "__main__":
    main()
