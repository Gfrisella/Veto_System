#!/usr/bin/env python3
"""Seeds of the veto muon production: Guglielmo's simulation, our categories, every category kept.

The physics is MuonsAndMatter's (imported read-only from --mm-dir, unchanged): the detector, field and
material tables of cuda_muons_ship.run_from_params with the settings of his shah_study runs (TRY6, diluted
iron, NI from B, cavern, simulated field, 5000 steps), and its propagate_muons_with_cuda, plane by plane
(z = 30.6, then 82 m; seed + i for plane i). As run_from_params with return_all, EVERY input muon goes on
to the next plane, so each has its start state, its state at z = 30.6 m (or where it stopped, in the
shield) and at z = 82 m (or where it stopped). One process sets up the geometry and field and loads the
~506 M input muons ONCE, then runs its seeds one after the other. The selection is ours
(lib/veto/muon_types.classify, exclusive types, from the real path A -> (bounce K) -> B):

  0 sst1_bounced  1 sst1_selected  2 sst1_other   in SST1, any momentum
  3 veto_only     crosses a vessel side wall or the 1 x 2.7 m entrance window, misses SST1
  4 rock_bounced  5 straight_past  6 stops_between  7 stops_before (dropped in the shield)

Storage: every type fills a budget of --budget rows (15 M) as fast as possible, not an equal share per
seed. From the muons a seed has of each type (RATES, from the alpha runs), type t is kept by the first
n_t = max(ceil(budget / rate_t), --min-seeds) seeds of the production (rank = seed - --first-seed), each
keeping a random fraction f_t = min(1, budget / (n_t * rate_t)) re-weighted to the seed's total of the type;
later seeds keep none of it. So rare types (SST1, veto-only: ~300-1000 per seed) are kept COMPLETELY in every
seed, rock-bounced is complete after 27 seeds, and straight-past, stops-between and the shield stoppers
are spread thinly over the first 20 seeds. One seed = one spill (all ~506 M input muons): a dataset of the
completed contributing seeds of a type divides that type's weights by their number (kept_types in the file).

Output (atomic: written to a temporary name, then renamed): <out-dir>/seed_<N>.npz with
  x/y/z/px/py/pz _start, _30, _82 (float32), weight (picked, re-weighted), weight_raw, type, evt_n, pdg
  n_by_type, w_by_type, w2_by_type (ALL muons of the seed, per type)
  kept_types (bool per type: this seed contributes), kept_fraction, rank, seed, meta (json)
Seeds with an output are skipped.

  inside the MuonsAndMatter container, on one GPU (bin/run_muon_study.sh runs one process per GPU):
  python3 bin/generate_muon_study.py --seeds 0 --gpu 0                        # one seed
  python3 bin/generate_muon_study.py --seeds 0-19999 --step 4 --offset 1      # seeds 1, 5, 9, ...
  python3 bin/generate_muon_study.py --seeds 0 --n-muons 2000000 --out-dir /tmp/test   # quick test
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))

from veto import muon_types as mt  # noqa: E402

MM_DIR = Path("/disk/users/gfrise/MuonsAndMatter")       # = /home/hep/gfrise/work_dir/MuonsAndMatter
INPUT = "data/muons/full_sample_after_target.h5"
PLANES = (30.6, 82.0)
PLANE_DZ = 0.0001
N_STEPS = 5000
STATE = ("x", "y", "z", "px", "py", "pz")
N_TYPES = 8
CHUNK = 20_000_000
# muons per seed of each type (alpha: 20 001 old runs for the SST1 types, one full run for the rest)
RATES = np.array([299.3, 300.7, 723.0, 975.0, 575_321.0, 124_942_004.0, 25_833_925.0, 355_020_487.0])


def storage_plan(rank, budget, min_seeds):
    """(kept, fraction) per type for the seed of this rank in the production."""
    n_seeds = np.maximum(np.ceil(budget / RATES), min_seeds)
    fraction = np.minimum(1.0, budget / (n_seeds * RATES))
    kept = rank < n_seeds
    return kept, np.where(kept, fraction, 0.0)


class Simulation:
    """His geometry, field, materials and input muons, set up once; run(seed) propagates every muon."""

    def __init__(self, mm_dir, params_name, device, n_muons=0):
        mm = Path(mm_dir)
        os.environ.setdefault("PROJECTS_DIR", str(mm.parent))        # as his shell_container.sh: materials
        os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        sys.dont_write_bytecode = True
        # as his set_env.sh, plus the compiled CUDA extension first (the source folder has the same name)
        for sub in ("cuda_muons", "cpp/build", "muons_and_matter",
                    "cuda_muons/faster_muons_torch/build/lib.linux-x86_64-3.10"):
            sys.path.insert(0, str(mm / sub))
        import h5py
        import torch
        import lib.reference_designs.params as params_lib
        from cuda_muons import propagate_muons_with_cuda, set_environment
        from cuda_muons_ship import (get_cavern_from_detector, get_corners_from_detector,
                                     get_magnetic_field_from_detector, load_histogram)
        from lib.ship_muon_shield import get_design_from_params
        self.torch, self.propagate, self.device = torch, propagate_muons_with_cuda, device
        torch.cuda.set_device(device)

        t0 = time.time()
        with h5py.File(mm / INPUT, "r") as f:
            n = n_muons or len(f["px"])
            muons = np.empty((n, 8), np.float32)
            for j, k in enumerate(("px", "py", "pz", "x", "y", "z", "pdg", "weight")):
                muons[:, j] = f[k][:n]
        self.muons = muons
        print(f"loaded {n:,} muons, {time.time() - t0:.0f} s", flush=True)

        # as run_from_params: detector, field, materials, environment (all independent of the seed)
        params = np.asarray(params_lib.params[params_name], dtype=float).reshape(-1, 15)
        params[-1][12] = 0.0                                          # TriangleCut, as in his runs
        detector = get_design_from_params(params=params, fSC_mag=False, field_mode="simulate", sensitive_film_params=None,
                                          field_map_file=None, add_cavern=True, add_target=False,
                                          sensitive_decay_vessel=False, NI_from_B=True, use_diluted=True, SND=False,
                                          cores_field=8)
        corners = get_corners_from_detector(detector, use_symmetry=True)
        cavern = get_cavern_from_detector(detector)
        field = get_magnetic_field_from_detector(detector, use_symmetry=True)
        hist_dir = mm / "cuda_muons" / "data"
        iron, self.step_length = load_histogram(str(hist_dir / "alias_histograms_G4_Fe.pkl"))
        concrete, step2 = load_histogram(str(hist_dir / "alias_histograms_G4_CONCRETE.pkl"))
        assert step2 == self.step_length
        self.environment = set_environment(corners, cavern, {"iron": iron, "concrete": concrete}, field, device=device)
        self.charge = torch.from_numpy(muons[:, 6].copy())
        if self.charge.abs().eq(13).all():                            # as run_from_params: PDG id -> charge
            self.charge = self.charge.div(-13)
        assert self.charge.abs().eq(1).all()
        print(f"geometry, field and materials ready, {time.time() - t0:.0f} s", flush=True)

    def run(self, seed):
        """{f'{k}_{tag}'}: every muon at the start, at z = 30.6 and at z = 82 m (or where it stopped)."""
        torch = self.torch
        pos = torch.from_numpy(self.muons[:, 3:6])
        mom = torch.from_numpy(self.muons[:, 0:3])
        cols = {}
        for k, j in zip(STATE, (3, 4, 5, 0, 1, 2)):
            cols[f"{k}_start"] = self.muons[:, j]
        for i, (tag, z) in enumerate(zip(("30", "82"), PLANES)):
            pos, mom = self.propagate(pos, mom, self.charge, self.environment, z - PLANE_DZ / 2, N_STEPS,
                                      self.step_length, True, seed + i, self.device)
            p, m = pos.numpy(), mom.numpy()
            for k, a, c in (("x", p, 0), ("y", p, 1), ("z", p, 2), ("px", m, 0), ("py", m, 1), ("pz", m, 2)):
                cols[f"{k}_{tag}"] = a[:, c]
        return cols


def classify(cols, how="gpu", device=0, check=False):
    """Our type of every muon; those that never reach z = 30.6 m are type 7 without the full classification.
    how: "gpu" (muon_types_torch, float64) or "cpu" (muon_types). check: also the other one, compared muon by
    muon on every field; returns (type, report)."""
    typ = np.full(len(cols["z_30"]), 7, np.int8)
    reach = np.nonzero(cols["z_30"] > mt.REACH_30)[0]
    keys = [k for k in cols if k.endswith(("_30", "_82"))]
    sub = {k: cols[k][reach] for k in keys}
    report = {}
    t0 = time.time()
    if how == "gpu":
        from veto import muon_types_torch as mtt
        out = mtt.classify(sub, device=f"cuda:{device}")
    else:
        parts = [mt.classify({k: v[a:a + CHUNK] for k, v in sub.items()}) for a in range(0, len(reach), CHUNK)]
        out = {f: np.concatenate([q[f] for q in parts]) for f in ("type", "walls", "kx", "ky", "kz", "off")}
    report[f"{how}_s"] = time.time() - t0
    if check:
        t0 = time.time()
        other = classify(cols, "cpu" if how == "gpu" else "gpu", device)[1]["out"]
        report["other_s"] = time.time() - t0
        report["n"] = len(reach)
        for f in ("type", "walls"):
            report[f"{f}_differ"] = int((out[f] != other[f]).sum())
        for f in ("kx", "ky", "kz", "off"):
            a, b = out[f], other[f]
            same = (a == b) | (np.isnan(a) & np.isnan(b))
            report[f"{f}_not_bitwise"] = int((~same).sum())
            fin = np.isfinite(a) & np.isfinite(b)
            report[f"{f}_max_abs_diff"] = float(np.abs(a[fin] - b[fin]).max()) if fin.any() else 0.0
        if report["type_differ"]:
            bad = np.nonzero(out["type"] != other["type"])[0][:5]
            report["examples"] = [(int(reach[i]), int(out["type"][i]), int(other["type"][i]), float(out["off"][i])) for i in bad]
    typ[reach] = out["type"]
    report["out"] = out
    return typ, report


def select(cols, w, typ, kept, fraction, rng):
    """A random `fraction` of the muons of every kept type, re-weighted to the type's total weight in the seed;
    per-type totals of all muons (kept or not)."""
    n_by, w_by, w2_by = (np.zeros(N_TYPES, t) for t in (np.int64, np.float64, np.float64))
    picks, weights = [], []
    for t in range(N_TYPES):
        idx = np.nonzero(typ == t)[0]
        n_by[t], w_by[t], w2_by[t] = len(idx), w[idx].sum(), (w[idx] ** 2).sum()
        if not len(idx) or not kept[t]:
            continue
        n_take = len(idx) if fraction[t] >= 1 else max(1, int(round(fraction[t] * len(idx))))
        take = idx if n_take == len(idx) else np.sort(rng.choice(idx, n_take, replace=False))
        picks.append(take)
        weights.append(w[take] * (w_by[t] / w[take].sum()))
    picks = np.concatenate(picks)
    order = np.argsort(picks)
    picks = picks[order]
    rows = {k: np.ascontiguousarray(v[picks], np.float32) for k, v in cols.items()}
    rows["weight"] = np.concatenate(weights)[order]
    rows["weight_raw"] = w[picks].astype(np.float64)
    rows["type"] = typ[picks]
    rows["evt_n"] = picks.astype(np.int64)
    return rows, n_by, w_by, w2_by


def parse_seeds(text):
    seeds = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        seeds += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return seeds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", required=True, help="e.g. 0, 0-19999, 0-9,20-29")
    ap.add_argument("--step", type=int, default=1, help="take every step-th seed of the list ...")
    ap.add_argument("--offset", type=int, default=0, help="... starting at this one (one process per GPU)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--budget", type=float, default=15e6, help="rows per type over the production")
    ap.add_argument("--min-seeds", type=int, default=20, help="spread even the most common types over this many seeds")
    ap.add_argument("--first-seed", type=int, default=0, help="first seed of the production (rank = seed - first seed)")
    ap.add_argument("--n-muons", type=int, default=0, help="only the first N input muons (testing); 0 = all")
    ap.add_argument("--classify", choices=["gpu", "cpu"], default="gpu", help="muon_types_torch (GPU) or muon_types (CPU)")
    ap.add_argument("--check-classify", action="store_true", help="also classify the other way and compare muon by muon")
    ap.add_argument("--params", default="TRY6")
    ap.add_argument("--mm-dir", default=str(MM_DIR))
    ap.add_argument("--out-dir", default=str(HERE.parent / "data" / "sim" / "veto_muons_v1"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)[args.offset::args.step]
    todo = [s for s in seeds if not (out_dir / f"seed_{s}.npz").exists()]
    print(f"{len(todo)} of {len(seeds)} seeds to run on GPU {args.gpu}", flush=True)
    if not todo:
        return
    sim = Simulation(args.mm_dir, args.params, args.gpu, args.n_muons)
    w = sim.muons[:, 7].astype(np.float64)
    for seed in todo:
        if (out_dir / f"seed_{seed}.npz").exists():                  # another process got there first
            continue
        t0 = time.time()
        cols = sim.run(seed)
        t_sim = time.time() - t0
        typ, report = classify(cols, args.classify, args.gpu, args.check_classify)
        t_cls = time.time() - t0 - t_sim
        if args.check_classify:
            print("classification check: " + json.dumps({k: v for k, v in report.items() if k != "out"}), flush=True)
        rank = seed - args.first_seed
        kept, fraction = storage_plan(rank, args.budget, args.min_seeds)
        rows, n_by, w_by, w2_by = select(cols, w, typ, kept, fraction, np.random.default_rng([seed, 20261016]))
        rows["pdg"] = np.sign(sim.charge.numpy()[rows["evt_n"]]).astype(np.int8)   # charge sign, as run_from_params' "pdg"
        del cols
        meta = dict(seed=seed, rank=rank, budget=args.budget, min_seeds=args.min_seeds, rates=RATES.tolist(),
                    n_input=int(len(typ)), params=args.params, mm_dir=args.mm_dir, input=INPUT, planes=PLANES,
                    types=mt.TYPES, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                    settings="TRY6, diluted iron, NI from B, cavern, field simulated, n_steps 5000, every muon to every plane")
        tmp = out_dir / f".seed_{seed}.tmp.npz"
        np.savez(tmp, **rows, n_by_type=n_by, w_by_type=w_by, w2_by_type=w2_by, kept_types=kept,
                 kept_fraction=fraction, rank=rank, seed=seed, meta=json.dumps(meta))
        os.replace(tmp, out_dir / f"seed_{seed}.npz")                 # readers never see a partial file
        print(f"seed {seed}: {len(rows['type']):,} rows, simulation {t_sim:.0f} s, classification {t_cls:.0f} s, "
              f"total {time.time() - t0:.0f} s | " + " ".join(f"{t}:{n_by[t]}" for t in range(N_TYPES)), flush=True)


if __name__ == "__main__":
    main()
