#!/usr/bin/env python3
"""Full-statistics validation of veto designs, with no timing-detector assumption by default.

For each design:
  single muons   EVERY SST1-acceptance muon of the design-optimisation dataset (26 M, their weights),
                 split into wall bounces, floor bounces, selected (p > 1 GeV, timing acceptance) and
                 other: per spill before / after the veto (= Hz at 5e13 PoT/s), weighted and unweighted
  fake signal    two-muon false vertices: N = pairs per spill * sum_kl f_kl eps_k eps_l (objective.py),
                 with NO time coincidence (same 1 s spill) unless --coincidence-ps; per spill and in 15 y.
                 Also the direct count: the 196 736 candidate pairs of results/reco/inputs.npz run
                 through the veto (planes upstream of the vertex)
  signal         hypothetical decays from the target (run_reco "target" model) and uniform in the vessel:
                 fraction lost to accidental vetoes
  rates, cost    per plane, from reco.plane_rates
Results: results/validate/<tag>.json (+ a muon-path sample per category for the HTML viewer).

  sbatch -c 64 --mem 200G --wrap "bin/validate_design.py OPT1 OPT2 --sigma-t-track 5 --n-sigma-t 5"
"""

import _bootstrap  # noqa: F401

import argparse
import copy
import dataclasses
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from run_reco import load_inputs, take
from veto import design as dsg
from veto import geometry as geo
from veto import io, paths, reco
from veto import objective as obj

DATASET = "design-optimization-dataset-alpha-14-oct-2026"
COLS = {"ax": "x_30", "ay": "y_30", "az": "z_30", "apz": "pz_30", "kx": "kx", "ky": "ky", "kz": "kz",
        "bx": "x_82", "by": "y_82", "bz": "z_82", "bpx": "px_82", "bpy": "py_82", "bpz": "pz_82",
        "weight": "weight", "type": "type"}
CATS = ("wall bounce", "floor bounce", "selected", "other")
_S = {}


def categories(tr):
    """0 wall bounce, 1 floor bounce, 2 selected (p > 1 GeV & timing acc.), 3 other."""
    ty = tr["type"].astype(int)
    bounced = (ty == 0) & np.isfinite(tr["kz"])
    wall = bounced & ((tr["kx"] - geo.CAVERN_X_WALL) < (tr["ky"] - geo.CAVERN_Y_FLOOR))
    cat = np.full(len(ty), 3)
    cat[ty == 1] = 2
    cat[ty == 0] = 1            # bounced: floor unless the wall is nearer (as plot_survivors.py)
    cat[wall] = 0
    return cat


def _init(store_dir, cfg):
    _S["store"], _ = io.open_store(Path(store_dir))
    _S["cfg"] = cfg
    _S["inp"] = load_inputs()[1]


def _chunk(job):
    raw, rows, seed = job
    tr = {k: np.asarray(_S["store"][c][rows]).astype(np.float64) for k, c in COLS.items()}
    d = dsg.from_dict(copy.deepcopy(raw))
    rng = np.random.default_rng(seed)
    m = reco.run_events(tr, np.arange(len(rows)), np.full(len(rows), 1e9), d, _S["inp"], _S["cfg"], rng)
    return m["vetoed"]


def evaluate(raw, pool, sst1, chunk, seed, cat, w, stats, cfg, inp_tr, inp, args, pair_ok):
    t0 = time.time()
    d = dsg.from_dict(copy.deepcopy(raw))
    jobs = [(raw, sst1[lo:lo + chunk], seed * 100_003 + i) for i, lo in enumerate(range(0, len(sst1), chunk))]
    vetoed = np.concatenate(pool.map(_chunk, jobs, chunksize=1))
    out = {"design": raw["name"], "raw": d.raw, "notes": d.notes, "cost_kchf": d.cost, "area_m2": d.area,
           "n_channels": d.n_channels, "seconds_single": time.time() - t0}
    cats = {}
    for k, name in enumerate(CATS):
        m = cat == k
        miss = m & ~vetoed
        cats[name] = {"rows": int(m.sum()), "before": float(w[m].sum()), "after": float(w[miss].sum()),
                      "after_err": float(np.sqrt((w[miss] ** 2).sum())), "rows_missed": int(miss.sum()),
                      "unweighted_ineff": float(miss.sum() / max(m.sum(), 1))}
    out["single"] = cats
    # objective types: 0 = bounced (wall + floor), 1 = selected, 2 = other
    ty = np.where(cat <= 1, 0, cat - 1)
    res = {"design": raw["name"], "cost_kchf": d.cost, "rel_cost": d.rel_cost,
           "single": {obj.TYPES[k]: float((w * vetoed)[ty == k].sum() / w[ty == k].sum()) for k in range(3)},
           "single_unweighted": {obj.TYPES[k]: float(vetoed[ty == k].mean()) for k in range(3)}}
    res.update(obj.pairable_entries(vetoed, ty, w, pair_ok))
    n_eff = np.array([w[ty == k].sum() ** 2 / (w[ty == k] ** 2).sum() for k in range(3)])
    st = dict(stats, n_eff_single=n_eff, n_single=np.array([float((ty == k).sum()) for k in range(3)]),
              rate_single=np.array([w[ty == k].sum() for k in range(3)]))

    # direct pair count (planes upstream of the vertex)
    rng = np.random.default_rng(seed + 7)
    cand = inp["cand"]
    rows = cand.ravel()
    ev = np.repeat(np.arange(len(cand)), 2)
    b = reco.run_events(take(inp_tr, rows), ev, np.repeat(inp["cand_vertex"][:, 2], 2), d, inp, cfg, rng)
    surv = ~b["vetoed"]
    cw = inp["cand_w"]
    res["background"] = dict(n=len(cand), n_survive=int(surv.sum()), FN_frac=float(cw[surv].sum() / cw.sum()))
    # signal: from the target (physics-like) and uniform in the vessel
    sig, z_v, evs, acc = reco.signal_from_target(args.n_signal, np.random.default_rng(seed + 12345))
    s = reco.run_events(sig, evs, z_v, d, inp, cfg, rng, with_own_hits=False)
    sig_u, z_u, ev_u = reco.signal_tracks(args.n_signal, np.random.default_rng(seed + 54321))
    su = reco.run_events(sig_u, ev_u, z_u, d, inp, cfg, rng, with_own_hits=False)
    res["signal"] = {"n": int(len(s["vetoed"])), "FP_frac": float(s["vetoed"].mean()), "TN_frac": float(1 - s["vetoed"].mean())}
    out["signal"] = {"target": {"n": int(len(s["vetoed"])), "lost": float(s["vetoed"].mean())},
                     "uniform": {"n": int(len(su["vetoed"])), "lost": float(su["vetoed"].mean())},
                     "lost_by_plane": {d.planes[i].name: int(c) for i, c in enumerate(s["by_plane"]) if c}}
    res["planes"] = [dict(r, peak_channel_hz=r["peak_channel"] / cfg.spill_s) for r in reco.plane_rates(d, inp, cfg.spill_s)]
    out["planes"] = [{k: r[k] for k in ("plane", "technology", "area", "rate", "peak_hz_cm2", "max_rate_hz_cm2",
                                         "peak_channel_hz", "cost", "n_channels")} for r in res["planes"]]
    for weighting in ("weighted", "unweighted"):
        oc = obj.ObjectiveConfig(goal="pairs", coincidence_s=args.coincidence_s, eps_weighting=weighting,
                                 bkg_form="log", bkg_target=1.0, w_mu=30.0)
        sc = obj.score(res, st, oc)
        bg = sc["confusion"]["background"]
        out[f"fake_{weighting}"] = {"per_spill": bg["per_spill_after"], "per_spill_err": bg["per_spill_after_err"],
                                    "per_15y": bg["events_15y_after"], "per_15y_err": bg["events_15y_after_err"],
                                    "before_per_spill": bg["events_15y_before"] / obj.SPILLS_15Y,
                                    "FN": bg["FN"], "FN_err": bg["FN_err"],
                                    "eps": sc["confusion"]["single_inefficiency"], "reward": sc["reward"],
                                    "feasible": sc["feasible"], "over_rate_limit": sc["over_rate_limit"]}
    out["pair_muon_ineff"] = {k: {"weighted": 1 - res["pairable"][k], "unweighted": 1 - res["pairable_unweighted"][k],
                                  "n": n, "n_eff": ne}
                              for k, n, ne in zip(obj.TYPES, res["pairable_n"], res["pairable_n_eff"])}
    out["fake_direct"] = {"pairs": len(cand), "survive": int(surv.sum()), "FN": res["background"]["FN_frac"],
                          "per_spill": float(cw[surv].sum()) * (1 if args.coincidence_s is None else 2 * args.coincidence_s)}
    out["seconds"] = time.time() - t0
    return out, vetoed


def path_sample(cat, vetoed, w, store, sst1, n_per, rng):
    """A few hundred muons per category for the viewer: half missed (if any), half vetoed."""
    rows = []
    for k in range(len(CATS)):
        for flag in (False, True):
            idx = np.where((cat == k) & (vetoed == flag))[0]
            take_n = min(n_per // 2, len(idx))
            if take_n:
                p = w[idx] / w[idx].sum()
                rows.append(rng.choice(idx, take_n, replace=False, p=p))
    rows = np.sort(np.concatenate(rows))
    g = {k: np.asarray(store[c][sst1[rows]]).astype(np.float64) for k, c in COLS.items()}
    p = np.sqrt(g["bpx"] ** 2 + g["bpy"] ** 2 + g["bpz"] ** 2)
    r = lambda a, n=3: np.round(np.nan_to_num(a, nan=-999), n).tolist()
    return {"cat": cat[rows].tolist(), "vetoed": vetoed[rows].astype(int).tolist(), "p": r(p, 2),
            "a": [r(g["ax"]), r(g["ay"]), r(g["az"])], "k": [r(g["kx"]), r(g["ky"]), r(g["kz"])],
            "b": [r(g["bx"]), r(g["by"]), r(g["bz"])]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("designs", nargs="+", help="design names / JSON files")
    parser.add_argument("--tag", default="validation")
    parser.add_argument("--sigma-t-track", type=float, default=5.0, help="track time resolution [ns] (no timing detector)")
    parser.add_argument("--n-sigma-t", type=float, default=5.0)
    parser.add_argument("--coincidence-ps", type=float, default=None, help="default: no time coincidence")
    parser.add_argument("--ubt-efficiency", type=float, default=None, help="what-if: UBT plane efficiency")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--n-signal", type=int, default=100_000)
    parser.add_argument("--chunk", type=int, default=20_000)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-per-type", type=int, default=3_000_000,
                        help="random rows kept per muon type (weights scaled up); 0 = all")
    parser.add_argument("--paths", type=int, default=600, help="muon paths per category for the viewer")
    args = parser.parse_args()
    args.coincidence_s = None if args.coincidence_ps is None else args.coincidence_ps * 1e-12
    start = time.time()
    cfg = reco.RecoConfig(seed=args.seed, sigma_t_track=args.sigma_t_track, n_sigma_t=args.n_sigma_t)
    store_dir = paths.store_dir(DATASET)
    store, meta = io.open_store(store_dir)
    types = np.asarray(store["type"])
    sel_rng = np.random.default_rng(args.seed + 1)
    sst1, scale = [], []
    for k in range(3):
        idx = np.where(types == k)[0]
        keep = np.sort(sel_rng.choice(idx, args.max_per_type, replace=False)) if 0 < args.max_per_type < len(idx) else idx
        sst1.append(keep)
        scale.append(np.full(len(keep), len(idx) / len(keep)))
    order = np.argsort(np.concatenate(sst1))
    sst1, scale = np.concatenate(sst1)[order], np.concatenate(scale)[order]
    t = {k: np.asarray(store[c][sst1]).astype(np.float64) for k, c in COLS.items() if k in ("type", "kx", "ky", "kz", "weight", "bx", "by", "bz", "bpx", "bpy", "bpz")}
    cat, pair_ok = categories(t), obj.pairable(t)
    w = t["weight"] * scale        # exact for the totals only on average: the cap keeps a random subset
    print(f"{len(sst1):,} SST1 muons, {w.sum():.0f} per spill; " + ", ".join(
        f"{c} {w[cat == k].sum():.0f}" for k, c in enumerate(CATS)), flush=True)
    inp_tr, inp = load_inputs()
    stats = obj.sample_stats()
    outdir = paths.results_dir() / "validate"
    outdir.mkdir(parents=True, exist_ok=True)
    results, samples = [], {}
    rng = np.random.default_rng(11)
    with Pool(args.workers, _init, (str(store_dir), cfg)) as pool:
        for name in args.designs:
            raw = json.loads(Path(name).read_text()) if Path(name).exists() else dsg.load(name).raw
            raw = copy.deepcopy(raw)
            if args.ubt_efficiency:
                raw["ubt"]["efficiency"] = args.ubt_efficiency
                raw["name"] = raw["name"] + f"_ubt{args.ubt_efficiency:g}"
            out, vetoed = evaluate(raw, pool, sst1, args.chunk, args.seed, cat, w, stats, cfg, inp_tr, inp, args, pair_ok)
            out["settings"] = {"sigma_t_track_ns": args.sigma_t_track, "n_sigma_t": args.n_sigma_t,
                               "coincidence_ps": args.coincidence_ps, "seed": args.seed}
            results.append(out)
            samples[raw["name"]] = path_sample(cat, vetoed, w, store, sst1, args.paths, rng)
            fw, fu = out["fake_weighted"], out["fake_unweighted"]
            print(f"{raw['name']}: missed " + ", ".join(f"{c} {v['after']:.1f}±{v['after_err']:.1f}"
                                                         for c, v in out["single"].items())
                  + f" per spill | fake {fw['per_spill']:.3g}±{fw['per_spill_err']:.2g} /spill weighted, "
                  f"{fu['per_spill']:.3g}±{fu['per_spill_err']:.2g} unweighted; direct {out['fake_direct']['survive']} pairs | "
                  f"signal lost {100 * out['signal']['target']['lost']:.2f} % (target), "
                  f"{100 * out['signal']['uniform']['lost']:.2f} % (uniform) | cost {out['cost_kchf']:.0f} kCHF | "
                  f"{out['seconds']:.0f} s", flush=True)
            (outdir / f"{args.tag}.json").write_text(json.dumps({"results": results}, indent=1, default=float))
            (outdir / f"{args.tag}_paths.json").write_text(json.dumps(samples))
    print(f"saved {outdir / (args.tag + '.json')}, {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
