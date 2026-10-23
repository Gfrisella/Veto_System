#!/usr/bin/env python3
"""Evolutionary search of the fin layout: any number of fins, on either side, tilted, any technology.

Each design: the straw UBT plus a list of fins (side, z, length, y_lo, y_hi, tilt [deg], technology).
The UBT is straws over the vessel entrance width (|x| <= 0.5 m) up to y = 1.35 m; its bottom is searched
between the floor (y = -3.36 m) and the vessel entrance (y = -1.35 m). Beside it (|x| > 0.5 m) only fins.
Optionally ONE small panel under the vessel (--bottom, design.py "bottom": z, length, x range, height,
pitch, technology; at most design.BOTTOM_MAX_AREA).
Score: veto.objective, with the rate limits and the 8 MCHF budget as walls. --goal sst1_muons:
SST1-acceptance muons not vetoed per spill, then signal loss, then cost. --goal pairs: false vertices
in 15 years first (no time coincidence unless --coincidence-ps), then every SST1 muon missed
(--w-mu), then signal loss, then cost. --sigma-t-track is the track time the veto is matched to:
0.1 ns with the timing detector, a few ns from the straw tracker alone. Every design is evaluated
on the same events (fixed seed): the 200 k single SST1 muons of results/reco/inputs.npz (own hits +
accidental hits, algorithm A, as run_reco.py) and --n-signal hypothetical decays. No pairs.

(mu + lambda) evolution: the --mu best designs so far are the parents; each generation makes
--lam children by mutating a parent (move / resize / tilt a fin, add or remove a fin, change its
technology or side) or by joining the right fins of one parent with the left fins of another.
New fins go to the right (-x, the side of the cavern wall the muons bounce off) 70 % of the time.

  bin/optimize_fins.py --generations 25 --workers 12        # results/optimize_fins/
  bin/optimize_fins.py --resume                             # continue from archive.jsonl
  bin/optimize_fins.py ... --fixed-tilt 0                   # every fin perpendicular to the beam
  bin/optimize_fins.py --bottom --goal pairs --sigma-t-track 5 --out optimize_bottom \
      --seed-from results/optimize_fins/archive.jsonl --seed-design results/optimize_fins/OPT1_tile2_flames.json
"""

import _bootstrap  # noqa: F401

import argparse
import copy
import json
import time
from multiprocessing import Pool

import numpy as np

from run_reco import TYPE_NAMES, load_inputs, take
from veto import design as dsg
from veto import geometry as geo
from veto import muon_types as mt
from veto import objective as obj
from veto import paths, reco

TECHS = ["straw", "fiber", "tile", "tile2", "pixel"]
BOUNDS = {"z": (geo.Z_UBT, geo.Z_VESSEL_MAX),     # fins attached to the vessel structure
          "length": (0.3, 4.5), "y_lo": (mt.FIN_Y_MIN, 0.0), "y_hi": (0.0, 4.0),
          "tilt": (-60.0, 60.0)}
UBT_Y_LO = (mt.FIN_Y_MIN, -geo.UBT_HALF_Y)     # UBT bottom: floor .. vessel entrance; top fixed at +1.35 m
BOTTOM_BOUNDS = {"z": (30.6, 80.0), "length": (0.5, 10.0), "x_c": (-3.5, 3.5), "width": (0.5, 4.0),
                 "y": (mt.FIN_Y_MIN, -1.4), "pitch": (-15.0, 15.0)}
SEEDS = ["D2_default", "D4_dense", "D6_straw", "D6_tile", "D6_tile2"]
_S = {}
OBJ = obj.ObjectiveConfig()
FIXED_TILT = None       # --fixed-tilt: every fin gets this tilt [deg], never mutated


# ---------------------------------------------------------------- evaluation (worker processes)

def _init(n_signal, seed, chunk, sigma_t_track=0.1, n_sigma_t=3.0):
    _S["tr"], _S["inp"] = load_inputs()
    _S["cfg"] = reco.RecoConfig(seed=seed, sigma_t_track=sigma_t_track, n_sigma_t=n_sigma_t)
    _S["n_signal"], _S["chunk"] = n_signal, chunk


def evaluate(raw):
    """run_reco-like result dict (single muons, signal, rates, cost) of one design, or {"error": ...}."""
    try:
        return _evaluate(raw)
    except Exception as e:      # a broken design must not stop the search
        return {"design": raw.get("name"), "error": f"{type(e).__name__}: {e}", "raw": raw}


def _evaluate(raw):
    t0 = time.time()
    try:
        d = dsg.from_dict(raw)
    except (ValueError, KeyError) as e:
        return {"design": raw.get("name"), "error": str(e), "raw": raw}
    tr, inp, cfg = _S["tr"], _S["inp"], _S["cfg"]
    rng = np.random.default_rng(cfg.seed)
    single, sw = inp["single"], inp["single_w"]
    vetoed = np.zeros(len(single), bool)
    for lo in range(0, len(single), _S["chunk"]):
        rows = single[lo:lo + _S["chunk"]]
        m = reco.run_events(take(tr, rows), np.arange(len(rows)), np.full(len(rows), 1e9), d, inp, cfg, rng)
        vetoed[lo:lo + len(rows)] = m["vetoed"]
    ty = tr["type"][single].astype(int)
    res = {"design": d.name, "raw": d.raw, "notes": d.notes, "n_fins": len(d.planes) - 1,
           "cost_kchf": d.cost, "rel_cost": d.rel_cost, "area_m2": d.area,
           "single": {TYPE_NAMES[k]: float((sw * vetoed)[ty == k].sum() / sw[ty == k].sum()) for k in range(3)},
           "single_unweighted": {TYPE_NAMES[k]: float(vetoed[ty == k].mean()) for k in range(3)}}
    res.update(obj.pairable_entries(vetoed, ty, sw, obj.pairable(take(tr, single))))
    sig, z_v, ev = reco.signal_tracks(_S["n_signal"], np.random.default_rng(cfg.seed + 12345))   # same decays for every design
    n_ev = int(ev.max()) + 1
    lost = np.zeros(n_ev, bool)
    for lo in range(0, n_ev, _S["chunk"]):
        m = (ev >= lo) & (ev < lo + _S["chunk"])
        s = reco.run_events(take(sig, m), ev[m] - lo, z_v[m], d, inp, cfg, rng, with_own_hits=False)
        lost[lo:lo + len(s["vetoed"])] = s["vetoed"]
    res["signal"] = {"n": int(len(lost)), "FP_frac": float(lost.mean()), "TN_frac": float(1 - lost.mean())}
    res["planes"] = [dict(r, peak_channel_hz=r["peak_channel"] / cfg.spill_s) for r in reco.plane_rates(d, inp, cfg.spill_s)]
    res["seconds"] = time.time() - t0
    return res


# ---------------------------------------------------------------- variation

def random_fin(rng):
    tilt = float(rng.uniform(-45, 45)) if rng.random() < 0.6 else 0.0
    tilt = tilt if FIXED_TILT is None else float(FIXED_TILT)
    return {"side": "right" if rng.random() < 0.7 else "left", "z": float(rng.uniform(40, geo.Z_VESSEL_MAX)),
            "length": float(rng.uniform(1, 4)), "y_lo": float(rng.choice([mt.FIN_Y_MIN, rng.uniform(-3.36, -1)])),
            "y_hi": float(rng.uniform(1, 4)), "tilt": round(tilt, 1), "technology": str(rng.choice(TECHS))}


def clean_fin(f, default_tech):
    out = {"side": f["side"], "technology": f.get("technology") or default_tech}
    for k, (lo, hi) in BOUNDS.items():
        out[k] = float(np.clip(f.get(k, 0.0), lo, hi))
    out["y_hi"] = max(out["y_hi"], out["y_lo"] + 0.3)
    out["tilt"] = round(out["tilt"], 1) if FIXED_TILT is None else float(FIXED_TILT)
    return out


def heuristic_seeds():
    """Buildable starting points: straw fins away from the flames at z = 30.6 m, denser on the right
    (-x, the bouncing side), down to the floor, some tilted."""
    def fin(side, z, length=4.5, tilt=0.0, tech="straw", y_lo=mt.FIN_Y_MIN, y_hi=3.5):
        return clean_fin({"side": side, "z": z, "length": length, "tilt": tilt, "y_lo": y_lo, "y_hi": y_hi,
                          "technology": tech}, tech)
    right = [45.6, 55.0, 62.0, 68.0, 74.0, 79.5]
    return [
        as_raw([fin("right", z) for z in right] + [fin("left", z, 3) for z in (45.6, 75.6)], "h_right6"),
        as_raw([fin("right", z) for z in right[1:]] + [fin("left", 70, 3)], "h_right5"),
        as_raw([fin("right", z, tilt=30) for z in right] + [fin("left", z, 3) for z in (45.6, 75.6)], "h_right6_t30"),
        as_raw([fin("right", z, tilt=-30) for z in right] + [fin("left", 60, 3)], "h_right6_tm30"),
        as_raw([fin("right", z) for z in (50, 58, 64, 70, 75, 80)] + [fin("left", z, 3) for z in (50, 65, 80)],
               "h_right6_left3"),
        as_raw([fin("right", z, tech="tile2") for z in (62, 68, 74, 79.5)] + [fin("right", z) for z in (45.6, 55)]
               + [fin("left", 75.6, 3, tech="tile2")], "h_mixed"),
        as_raw([fin("right", z) for z in (40, 50, 58, 64, 70, 75, 80)], "h_right7_only"),
        as_raw([fin(sd, z, 3, y_lo=-3.0, y_hi=3.0) for z in (45.6, 60.6, 75.6) for sd in ("left", "right")],
               "D6_straw_no30"),
    ] + [
        # the muons beside the UBT (the flames) need a plane near z = 30.6 m: straws a little downstream,
        # where the flames are less dense, or a narrow fibre fin (no rate limit, 340 kCHF / m^2)
        as_raw([fin(sd, z0, 3, y_lo=-3.0, y_hi=3.0) for sd in ("left", "right")]
               + [fin("right", z) for z in right] + [fin("left", 75.6, 3)], f"h_flames_straw{z0:.0f}")
        for z0 in (31.6, 33.0, 35.0)
    ] + [
        as_raw([fin(sd, 30.6, 2.0, tech="fiber", y_lo=-2.0, y_hi=2.0) for sd in ("left", "right")]
               + [fin("right", z) for z in right[1:]], "h_flames_fiber"),
    ]


def as_raw(fins, name, ubt_y_lo=mt.FIN_Y_MIN, bottom=None):
    raw = {"name": name, "ubt": {"y_lo": float(np.clip(ubt_y_lo, *UBT_Y_LO)), "y_hi": geo.UBT_HALF_Y},
           "fins": fins, "detector": {}}
    if bottom:
        raw["bottom"] = bottom
    return raw


def clean_bottom(b):
    """Clip the panel to the bounds and fit it between the floor and the vessel bottom."""
    g = {"z": b["z"], "length": b["length"], "x_c": 0.5 * (b["x_lo"] + b["x_hi"]),
         "width": b["x_hi"] - b["x_lo"], "y": b["y"], "pitch": b.get("pitch", 0.0)}
    g = {k: float(np.clip(g[k], lo, hi)) for k, (lo, hi) in BOTTOM_BOUNDS.items()}
    g["length"] = min(g["length"], dsg.BOTTOM_MAX_AREA / g["width"],
                      (dsg.BOTTOM_Z_MAX - g["z"]) / np.cos(np.radians(g["pitch"])))
    for pitch in (round(g["pitch"], 1), 0.0):
        sn = g["length"] * np.sin(np.radians(pitch))
        z1 = g["z"] + g["length"] * np.cos(np.radians(pitch))
        lo = max(mt.FIN_Y_MIN, mt.FIN_Y_MIN - sn)
        hi = min(dsg.vessel_bottom(g["z"]), dsg.vessel_bottom(z1) - sn) - dsg.BOTTOM_GAP
        if lo <= hi:
            g["pitch"], g["y"] = pitch, float(np.clip(g["y"], lo + 1e-6, hi - 1e-6))
            break
    return {"z": g["z"], "length": g["length"], "x_lo": g["x_c"] - g["width"] / 2, "x_hi": g["x_c"] + g["width"] / 2,
            "y": g["y"], "pitch": g["pitch"], "technology": b.get("technology", "straw")}


def random_bottom(rng):
    w = float(rng.uniform(1, 4))
    x_c = float(rng.uniform(-2.5, 2.5))
    return clean_bottom({"z": float(rng.uniform(38, 78)), "length": float(rng.uniform(1, 10 / w)),
                         "x_lo": x_c - w / 2, "x_hi": x_c + w / 2, "y": float(rng.uniform(-3.36, -2.8)),
                         "pitch": float(rng.uniform(-10, 10)) if rng.random() < 0.4 else 0.0,
                         "technology": str(rng.choice(TECHS))})


def mutate_bottom(b, rng):
    if b is None or rng.random() < 0.15:
        return random_bottom(rng)
    if rng.random() < 0.2:
        return dict(b, technology=str(rng.choice(TECHS)))
    g = {"z": b["z"], "length": b["length"], "x_c": 0.5 * (b["x_lo"] + b["x_hi"]),
         "width": b["x_hi"] - b["x_lo"], "y": b["y"], "pitch": b.get("pitch", 0.0)}
    for k, (lo, hi) in BOTTOM_BOUNDS.items():
        if rng.random() < 0.5:
            g[k] += rng.normal(0, 0.1 * (hi - lo))
    return clean_bottom({"z": g["z"], "length": g["length"], "x_lo": g["x_c"] - g["width"] / 2,
                         "x_hi": g["x_c"] + g["width"] / 2, "y": g["y"], "pitch": g["pitch"],
                         "technology": b.get("technology", "straw")})


def ubt_y_lo(raw):
    return raw.get("ubt", {}).get("y_lo", mt.FIN_Y_MIN)


def mutate(raw, rng, bottom=False):
    """Returns (fins, UBT bottom, bottom panel, op)."""
    fins = copy.deepcopy(raw["fins"])
    y_lo = ubt_y_lo(raw)
    panel = copy.deepcopy(raw.get("bottom"))
    ops = ["tweak", "tweak", "tweak", "add", "remove", "technology", "side", "ubt"]
    op = rng.choice(ops + ["bottom", "bottom"] if bottom else ops)
    if op == "bottom":
        panel = mutate_bottom(panel, rng)
    elif op == "ubt":
        y_lo = mt.FIN_Y_MIN if rng.random() < 0.3 else y_lo + rng.normal(0, 0.3)
    elif op == "add" or not fins:
        fins.append(random_fin(rng))
    elif op == "remove" and len(fins) > 1:
        fins.pop(int(rng.integers(len(fins))))
    elif op == "technology":
        fins[int(rng.integers(len(fins)))]["technology"] = str(rng.choice(TECHS))
    elif op == "side":
        f = fins[int(rng.integers(len(fins)))]
        f["side"] = "left" if f["side"] == "right" else "right"
    else:
        for f in fins if rng.random() < 0.2 else [fins[int(rng.integers(len(fins)))]]:
            for k, (lo, hi) in BOUNDS.items():
                if rng.random() < 0.5:
                    f[k] = f[k] + rng.normal(0, 0.1 * (hi - lo))
    if bottom and panel is None:
        panel = random_bottom(rng)
    return [clean_fin(f, "straw") for f in fins], y_lo, panel, str(op)


def crossover(a, b, rng=None):
    fins = [f for f in a["fins"] if f["side"] == "right"] + [f for f in b["fins"] if f["side"] == "left"]
    panel = (b if rng is not None and rng.random() < 0.5 and b.get("bottom") else a).get("bottom")
    return copy.deepcopy(fins) or copy.deepcopy(a["fins"]), ubt_y_lo(a), copy.deepcopy(panel), "crossover"


# ---------------------------------------------------------------- bookkeeping

def summary(res, sc):
    cm = sc["confusion"]["sst1_muons"]
    bg = sc["confusion"]["background"]
    return {"name": res["design"], "reward": sc["reward"], "reward_err": sc["reward_err"],
            "fake_15y": bg["events_15y_after"], "fake_15y_err": bg["events_15y_after_err"],
            "fake_per_spill": bg["per_spill_after"], "fake_15y_before": bg["events_15y_before"],
            "single_ineff": {k: v[0] for k, v in sc["confusion"]["single_inefficiency"].items()},
            "terms": {k: v for k, (v, _) in sc["terms"].items()}, "feasible": sc["feasible"],
            "missed_per_spill": cm["per_spill_after"], "missed_err": cm["per_spill_after_err"],
            "missed_by_type": {k: v["after"] for k, v in cm["by_type"].items()},
            "signal_loss": res["signal"]["FP_frac"], "cost_kchf": sc["cost_kchf"], "n_fins": res["n_fins"],
            "over_rate_limit": sc["over_rate_limit"], "notes": res["notes"], "seconds": res["seconds"],
            "raw": res["raw"]}


def line(s):
    return (f"{s['name']:<16} R {s['reward']:9.2f}  fake {s['fake_per_spill']:9.3g}/spill  "
            f"missed {s['missed_per_spill']:7.0f}/spill  "
            f"signal loss {100 * s['signal_loss']:5.2f} %  cost {s['cost_kchf']:6.0f} kCHF  "
            f"fins {s['n_fins']:2d}  {'ok' if s['feasible'] else 'INFEASIBLE'}")


def bottom_seeds():
    """Panels where the floor bounces are (z = 40 .. 75 m, under the vessel), on the floor, 10 m^2."""
    out = [clean_bottom({"z": z, "length": 3.3, "x_lo": -1.5, "x_hi": 1.5, "y": mt.FIN_Y_MIN, "pitch": 0.0,
                         "technology": "straw"}) for z in (42, 50, 58, 66, 74)]
    out.append(clean_bottom({"z": 60, "length": 2.5, "x_lo": -2, "x_hi": 2, "y": -3.2, "pitch": 0.0, "technology": "tile2"}))
    out.append(clean_bottom({"z": 70, "length": 5, "x_lo": -1, "x_hi": 1, "y": -3.1, "pitch": 3.0, "technology": "straw"}))
    return out


def initial_designs(args, rng):
    layouts = []
    for n in SEEDS:
        d = dsg.load(n)
        tech = d.raw.get("detector", {}).get("technology", "straw")
        layouts.append(as_raw([clean_fin(f, tech) for f in d.raw["fins"]], n))
    layouts += heuristic_seeds()
    best = []
    for path in args.seed_design:                      # e.g. OPT1: kept as it is, UBT included
        raw = json.loads(open(path).read())
        best.append(as_raw([clean_fin(f, "straw") for f in raw["fins"]], raw["name"], ubt_y_lo(raw)))
    for path in args.seed_from:                        # the best designs of an earlier search
        old = [json.loads(s) for s in open(path).read().splitlines()]
        for s in sorted(old, key=lambda s: -s["reward"])[:args.seed_top]:
            best.append(as_raw([clean_fin(f, "straw") for f in s["raw"]["fins"]], "s_" + s["name"], ubt_y_lo(s["raw"])))
    seeds = []
    if args.bottom:
        panels = bottom_seeds()
        for raw in best:                               # every good layout with every heuristic panel
            for j, b in enumerate(panels):
                seeds.append(dict(copy.deepcopy(raw), name=f"{raw['name']}_b{j}", bottom=b))
        for raw in layouts:
            seeds.append(dict(raw, bottom=random_bottom(rng)))
    else:
        seeds = best + layouts
    # asymmetric random starts: mostly on the bounce (right) side, reaching the floor, some tilted
    for i in range(max(args.lam - len(seeds), 4)):
        fins = [random_fin(rng) for _ in range(int(rng.integers(3, 9)))]
        seeds.append(as_raw(fins, f"g0_{i}", bottom=random_bottom(rng) if args.bottom else None))
    return seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--mu", type=int, default=8)
    parser.add_argument("--lam", type=int, default=12)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--n-signal", type=int, default=30_000)
    parser.add_argument("--chunk", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1, help="event seed, the same for every design")
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", default="optimize_fins", help="sub-folder of results/")
    parser.add_argument("--bottom", action="store_true", help="one small panel under the vessel")
    parser.add_argument("--goal", default="sst1_muons", choices=["sst1_muons", "pairs"])
    parser.add_argument("--coincidence-ps", type=float, default=None,
                        help="two-track time coincidence, half window [ps]; default: none (same spill)")
    parser.add_argument("--w-mu", type=float, default=30.0, help="goal pairs: weight of the SST1-muon tier")
    parser.add_argument("--sigma-t-track", type=float, default=0.1, help="track time resolution [ns]")
    parser.add_argument("--n-sigma-t", type=float, default=3.0, help="veto time window: +- n sigma")
    parser.add_argument("--eps", default="weighted", choices=["weighted", "unweighted"],
                        help="single-muon inefficiency used for the ranking")
    parser.add_argument("--seed-from", nargs="*", default=[], help="archive.jsonl of an earlier search")
    parser.add_argument("--seed-top", type=int, default=8)
    parser.add_argument("--seed-design", nargs="*", default=[], help="design JSONs to start from")
    parser.add_argument("--fixed-tilt", type=float, default=None,
                        help="every fin at this tilt [deg] (seeds included), tilt not searched")
    args = parser.parse_args()
    global FIXED_TILT
    FIXED_TILT = args.fixed_tilt
    OBJ.goal = args.goal
    OBJ.coincidence_s = None if args.coincidence_ps is None else args.coincidence_ps * 1e-12
    if args.goal == "pairs":
        OBJ.bkg_form, OBJ.bkg_target, OBJ.w_mu = "log", 1.0, args.w_mu
    OBJ.eps_weighting = args.eps

    rng = np.random.default_rng(args.search_seed)
    outdir = paths.results_dir() / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    archive_path = outdir / "archive.jsonl"
    (outdir / "settings.json").write_text(json.dumps({"args": vars(args), "objective": obj.asdict(OBJ)}, indent=1))
    stats = obj.sample_stats()
    archive = [json.loads(s) for s in archive_path.read_text().splitlines()] if args.resume and archive_path.exists() else []
    if not args.resume:
        archive_path.write_text("")

    def record(results, gen, ops):
        out = []
        with archive_path.open("a") as fh:
            for res, op in zip(results, ops):
                if res.get("error"):
                    print(f"  {res['design']}: {res['error']}")
                    continue
                s = summary(res, obj.score(res, stats, OBJ))
                s.update(generation=gen, op=op)
                fh.write(json.dumps(s, default=float) + "\n")
                out.append(s)
        return out

    with Pool(args.workers, _init, (args.n_signal, args.seed, args.chunk, args.sigma_t_track, args.n_sigma_t)) as pool:
        start = time.time()
        if not archive:
            seeds = initial_designs(args, rng)
            archive += record(pool.map(evaluate, seeds), 0, ["seed"] * len(seeds))
            print(f"generation 0: {len(archive)} designs, {time.time() - start:.0f} s", flush=True)
        gen0 = max(s["generation"] for s in archive) + 1
        for gen in range(gen0, gen0 + args.generations):
            parents = sorted(archive, key=lambda s: -s["reward"])[:args.mu]
            children, ops = [], []
            for i in range(args.lam):
                if rng.random() < 0.15 and len(parents) > 1:
                    a, b = rng.choice(len(parents), 2, replace=False)
                    fins, y_lo, panel, op = crossover(parents[a]["raw"], parents[b]["raw"], rng)
                else:
                    fins, y_lo, panel, op = mutate(parents[int(rng.integers(len(parents)))]["raw"], rng, args.bottom)
                children.append(as_raw(fins, f"g{gen}_{i}", y_lo, panel))
                ops.append(op)
            archive += record(pool.map(evaluate, children), gen, ops)
            best = sorted(archive, key=lambda s: -s["reward"])
            print(f"generation {gen}: {len(archive)} designs, {time.time() - start:.0f} s", flush=True)
            for s in best[:3]:
                print("  " + line(s), flush=True)
            (outdir / "best.json").write_text(json.dumps(best[0]["raw"], indent=1))
            (outdir / "top10.json").write_text(json.dumps(best[:10], indent=1, default=float))


if __name__ == "__main__":
    main()
