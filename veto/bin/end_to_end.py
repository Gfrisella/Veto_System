#!/usr/bin/env python3
"""End-to-end figure of merit of designs on one muon dataset: every background source, before and after the veto.

Sources (per 15 years, after the selection of RecoConfig: measured IP < 0.1 m, at most 2 tracks in SST1):
  combinatorial        two unrelated SST1 muons in time (run_reco.evaluate + veto.objective, bkg_estimate "max")
  muon interactions    fastveto.dis.run on the muon dataset (library split crossover), per target: helium, air,
                       PVC walls and every plane of the design; veto level b (interactions.evaluate)
  neutrino interactions  fast-veto-sim scripts/nu_dis_apply.py for the design (GENIE), same targets and veto
Plus the signal lost to random hits, the cost and the objective (veto.objective with all sources).
Stage 1 (--stage generate) draws the muon interactions of each design (results/end_to_end/<design>_mu.npz);
stage 2 (--stage evaluate) runs every (design, cone) and writes results/end_to_end/<tag>.json.

  GPU_HOST=farm-gpu5 bin/remote.sh run bin/end_to_end.py --stage generate --designs E3_robust D6_tile2
  GPU_HOST=farm-gpu5 bin/remote.sh run bin/end_to_end.py --stage evaluate --designs E3_robust D6_tile2 --cones 0 10 20
"""

import _bootstrap  # noqa: F401

import argparse
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from run_interactions import FVS, load_npz
from run_reco import evaluate, load_inputs
from veto import design as dsg
from veto import interactions as ix
from veto import io, objective, paths, reco
from veto.config import DEFAULT_CONFIG

DATASET = "design-optimization-dataset-delta-enriched-2026-10-19"
OUT = paths.results_dir() / "end_to_end"
_S = {}


def load_design(name):
    p = Path(name)
    return dsg.load(p if p.exists() else name)


def generate(args):
    sys.path.insert(0, str(FVS))
    from fastveto import dis
    lib = dis.MuonLibrary(str(FVS / "data/dis/mu_library.npz"), split="crossover")
    muons, _ = io.open_store(paths.store_dir(args.dataset))
    draws = dict(zip(("helium", "pvc", "air", "planes"), args.draws))
    for name in args.designs:
        t0 = time.time()
        design = load_design(name)
        inter, part, _ = dis.run(muons, design, lib, draws, np.random.default_rng(args.seed), workers=args.workers)
        np.savez(OUT / f"{design.name}_mu.npz", **{f"int_{k}": v for k, v in inter.items()},
                 **{f"part_{k}": v for k, v in part.items()})
        print(f"{design.name}: {len(inter['weight']):,} muon interactions, {time.time() - t0:.0f} s", flush=True)


def _init(args):
    _S["args"] = args
    _S["tr"], _S["inp"] = load_inputs(args.inputs)
    _S["muons"] = io.open_store(paths.store_dir(args.dataset))[0]
    _S["stats"] = objective.sample_stats(np.load(paths.results_dir() / "reco" / args.inputs))


def group(res, design):
    """Per-target rows of interactions.evaluate -> helium / air / pvc / planes (all planes of the design together)."""
    out = {}
    for name, r in res.items():
        key = name if name in ("helium", "air", "pvc") else "planes"
        g = out.setdefault(key, dict(interactions=0.0, selected=0.0, after=0.0, after_err2=0.0, truth=0.0,
                                     n_selected=0, n_after=0))
        g["interactions"] += r["interactions"]; g["selected"] += r["selected"]; g["n_selected"] += r["n_selected"]
        g["after"] += r.get("survive_b", 0.0); g["after_err2"] += r.get("survive_b_err", 0.0) ** 2
        g["n_after"] += r.get("n_survive_b", 0); g["truth"] += r.get("survive_c", 0.0)
    for g in out.values():
        g["after_err"] = math.sqrt(g.pop("after_err2"))
    return out


def job(arg):
    name, cone = arg
    args, t0 = _S["args"], time.time()
    d = load_design(name)
    cfg = reco.RecoConfig(seed=1, multiple_scattering=True, cone=cone * 1e-3)
    r = evaluate(d, _S["tr"], _S["inp"], cfg, args.n_signal, DEFAULT_CONFIG["time_window"], "target")
    r["cost_kchf"] += float(d.raw.get("extra_cost_kchf", 0.0))     # e.g. second stations (efficiency overrides)
    comb = objective.score(r, _S["stats"], objective.ObjectiveConfig(bkg_estimate="max"))
    mu = ix.evaluate(*load_npz(OUT / f"{d.name}_mu.npz"), d, _S["inp"], cfg, np.random.default_rng(1), muons=_S["muons"])
    nu_file = OUT / f"{d.name}_nu.npz"
    nu = ix.evaluate(*load_npz(nu_file), d, _S["inp"], cfg, np.random.default_rng(2)) if nu_file.exists() else {}
    mu_g, nu_g = group(mu, d), group(nu, d)
    tot = lambda g: (sum(v["after"] for v in g.values()), math.sqrt(sum(v["after_err"] ** 2 for v in g.values())))
    r["interactions"] = {"muon": dict(zip(("total", "total_err"), tot(mu_g))), "nu": dict(zip(("total", "total_err"), tot(nu_g)))}
    s = objective.score(r, _S["stats"], objective.ObjectiveConfig(bkg_estimate="max"))
    cb = comb["confusion"]["background"]
    return dict(design=d.name, cone_mrad=cone, cost_kchf=s["cost_kchf"], n_channels=r["n_channels"],
                feasible=s["feasible"], over_rate_limit=s["over_rate_limit"],
                signal_lost=s["confusion"]["signal"]["FP"], signal_lost_err=s["confusion"]["signal"]["FP_err"],
                combinatorial=dict(before=cb["events_15y_before"], after=comb["bkg_events_15y"], after_err=comb["bkg_events_15y_err"],
                                   single_inefficiency=comb["confusion"]["single_inefficiency"]),
                muon=mu_g, nu=nu_g, total=s["bkg_events_15y"], total_err=s["bkg_events_15y_err"],
                reward=s["reward"], reward_err=s["reward_err"], terms={k: v[0] for k, v in s["terms"].items()},
                per_plane_mu={k: dict(interactions=v["interactions"], selected=v["selected"], after=v.get("survive_b", 0.0))
                              for k, v in mu.items() if k not in ("helium", "air", "pvc")},
                planes=[dict(name=p["plane"], technology=p["technology"], peak_hz_cm2=p["peak_hz_cm2"],
                             peak_channel_hz=p["peak_channel_hz"]) for p in r["planes"]],
                seconds=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("generate", "evaluate"), required=True)
    ap.add_argument("--designs", nargs="+", required=True)
    ap.add_argument("--cones", type=float, nargs="+", default=[0, 10, 20], help="mrad")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--inputs", default="inputs_delta_s.npz")
    ap.add_argument("--draws", type=float, nargs=4, default=[5e6, 5e6, 1e6, 1e6], help="helium pvc air planes (each)")
    ap.add_argument("--n-signal", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--tag", default="e2e")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "generate":
        return generate(args)
    jobs = [(d, c) for d in args.designs for c in args.cones]
    with Pool(min(args.workers, len(jobs)), initializer=_init, initargs=(args,)) as pool:
        rows = pool.map(job, jobs)
    f = lambda v, e=None: f"{v:.3g}" + (f" ± {e:.2g}" if e is not None else "")
    print(f"\n{'design':<12}{'cone':>5}{'signal lost':>13}{'comb.':>10}{'μ He':>9}{'μ air':>9}{'μ PVC':>9}{'μ planes':>9}"
          f"{'ν He':>8}{'ν air':>8}{'ν PVC':>8}{'ν planes':>9}{'total':>17}{'reward':>10}   (per 15 years, after the veto)")
    for r in rows:
        g = lambda kind, t: r[kind].get(t, {}).get("after", 0.0)
        print(f"{r['design']:<12}{r['cone_mrad']:>5g}{100 * r['signal_lost']:>11.2f} %{r['combinatorial']['after']:>10.2g}"
              + "".join(f"{g('muon', t):>9.3g}" for t in ("helium", "air", "pvc", "planes"))
              + "".join(f"{g('nu', t):>8.3g}" for t in ("helium", "air", "pvc")) + f"{g('nu', 'planes'):>9.3g}"
              + f"{f(r['total'], r['total_err']):>17}{r['reward']:>10.2f}")
    print(f"\nconfusion matrix per source, events per 15 years (TP vetoed, FN survive; signal: FP lost, TN kept)")
    for r in rows:
        print(f"\n{r['design']}, cone {r['cone_mrad']:g} mrad: signal FP {100 * r['signal_lost']:.2f} %  TN {100 * (1 - r['signal_lost']):.2f} %")
        srcs = [("combinatorial", r["combinatorial"]["before"], r["combinatorial"]["after"])]
        srcs += [(f"{'mu' if k == 'muon' else 'nu'} {t}", v["selected"], v["after"]) for k in ("muon", "nu") for t, v in r[k].items()]
        for name, before, after in srcs:
            tpr = 1 - after / before if before > 0 else float("nan")
            print(f"  {name:<16} before {before:>10.4g}   TP {before - after:>10.4g}   FN {after:>9.3g}   TPR {100 * tpr:8.3f} %")
    out = OUT / f"{args.tag}.json"
    json.dump(dict(dataset=args.dataset, inputs=args.inputs, draws=args.draws, n_signal=args.n_signal, rows=rows),
              open(out, "w"), indent=1, default=float)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
