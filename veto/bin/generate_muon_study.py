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

Storage "alpha" (--storage alpha; experimental, docs/storage_policy.md): every seed contributes every type, the
storage goes where the veto needs statistics, and kept rows carry Horvitz-Thompson weights w / p:
  types 0-3   kept completely, as above
  types 4-7   each muon kept with p = max(p_floor, p_cell): a uniform floor of --alpha-floor rows per type per seed
              (a complete, unbiased picture of every type, the shield stoppers included), and, for muons crossing the
              probe planes (veto.probe: fins every 2.5 m on both sides, floor planes, the full-height UBT column, 25 cm
              cells), p_cell = min(1, lam w max_c 1 / V_c), V_c = muons per spill of types 4-6 crossing cell c in this
              seed, lam such that the seed keeps --alpha-rows-per-seed of them: every cell gets about the same number
              of equally weighted rows, the quiet ones near the walls and on the floor complete
  hist        rate maps of the probe cells from EVERY muon of types 0-6 of the seed (counts, sum w, sum w^2): the
              random-hit maps at the statistics of all muons and seeds, without storing them
In the seed file: kept_types all True, p_keep per row, hist (3 x cells), probe settings in meta.

Storage "shell" (--storage shell; 2026-11-07, lib/veto/shell.py): statistics everywhere on a closed shell around the vessel,
rows with Horvitz-Thompson weights w / p:
  types 0-3, last-metre stops   kept completely, up to --keep-cap rows per class per seed (beyond: uniform, flagged in meta)
  types 4-6 crossing the shell  p_cell = min(1, lam w max_c 1 / V_c) over the shell cells c the muon crosses (as alpha, with
                                the shell's cells), lam: --shell-rows-per-seed rows
  types 4-7                     a floor of --shell-floor rows per type per seed
  weights                       plain w / p, not rescaled per type (cell estimates unbiased; type totals in w_by_type)
  shell_hist                    counts, sum w, sum w^2 per shell cell from ALL muons of types 0-6 (shell_cell_names: labels)
In the seed file also kx/ky/kz (bounce point), walls and last_metre per row, the settings in meta["shell"].

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
from veto import paths, weights  # noqa: E402
from veto.config import POT_PER_SPILL  # noqa: E402

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


def ext_build(mm):
    """The compiled faster_muons_torch for this machine and Python: build/lib.linux-x86_64-3.10 on the farm (his
    container), lib.linux-aarch64-cpython-314 on the GH200 nodes (cudabear_gh200)."""
    import platform
    v = sys.version_info
    for d in sorted((mm / "cuda_muons/faster_muons_torch/build").glob("lib.*")):
        if platform.machine() in d.name and (f"{v.major}.{v.minor}" in d.name or f"{v.major}{v.minor}" in d.name):
            return str(d.relative_to(mm))
    return "cuda_muons/faster_muons_torch/build/lib.linux-x86_64-3.10"


class Simulation:
    """His geometry, field, materials and input muons, set up once; run(seed) propagates every muon."""

    def __init__(self, mm_dir, params_name, device, n_muons=0, input_file=None):
        mm = Path(mm_dir)
        os.environ.setdefault("PROJECTS_DIR", str(mm.parent))        # as his shell_container.sh: materials
        os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        sys.dont_write_bytecode = True
        # as his set_env.sh, plus the compiled CUDA extension first (the source folder has the same name)
        for sub in ("cuda_muons", "cpp/build", "muons_and_matter", ext_build(mm)):
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

        self.n_muons = n_muons
        self.load(input_file or (mm / INPUT))
        t0 = time.time()

        # as run_from_params: detector, field, materials, environment (all independent of the seed). With
        # VETO_GEOMETRY_CACHE=<file>: corners, cavern and field from that file if it exists (no snoopy needed: the
        # GH200 nodes), else computed and saved there (the field simulation is deterministic, so the same physics)
        cache = os.environ.get("VETO_GEOMETRY_CACHE")
        if cache and Path(cache).exists():
            corners, cavern, field = torch.load(cache, weights_only=False)
            print(f"geometry and field from {cache}", flush=True)
        else:
            params = np.asarray(params_lib.params[params_name], dtype=float).reshape(-1, 15)
            params[-1][12] = 0.0                                      # TriangleCut, as in his runs
            detector = get_design_from_params(params=params, fSC_mag=False, field_mode="simulate", sensitive_film_params=None,
                                              field_map_file=None, add_cavern=True, add_target=False,
                                              sensitive_decay_vessel=False, NI_from_B=True, use_diluted=True, SND=False,
                                              cores_field=8)
            corners = get_corners_from_detector(detector, use_symmetry=True)
            cavern = get_cavern_from_detector(detector)
            field = get_magnetic_field_from_detector(detector, use_symmetry=True)
            if cache:
                torch.save((corners, cavern, field), cache)
                print(f"geometry and field saved to {cache}", flush=True)
        hist_dir = mm / "cuda_muons" / "data"
        iron, self.step_length = load_histogram(str(hist_dir / "alias_histograms_G4_Fe.pkl"))
        concrete, step2 = load_histogram(str(hist_dir / "alias_histograms_G4_CONCRETE.pkl"))
        assert step2 == self.step_length
        self.environment = set_environment(corners, cavern, {"iron": iron, "concrete": concrete}, field, device=device)
        print(f"geometry, field and materials ready, {time.time() - t0:.0f} s", flush=True)

    def load(self, input_file):
        """The input muons (all, or the first n_muons). A pool chunk (bin/build_pool.py) carries pool_index, the muon's
        id in the whole pool: stored rows then get that as evt_n, so muons of different chunks never share an id."""
        import h5py
        torch = self.torch
        t0 = time.time()
        with h5py.File(input_file, "r") as f:
            n = self.n_muons or len(f["px"])
            muons = np.empty((n, 8), np.float32)
            for j, k in enumerate(("px", "py", "pz", "x", "y", "z", "pdg", "weight")):
                muons[:, j] = f[k][:n]
            self.pool_index = f["pool_index"][:n] if "pool_index" in f else None
        pool_json = Path(input_file).parent / "pool.json"           # D13: the pool's p.o.t. per spill (v5 records it)
        pot = weights.source_pot(json.load(open(pool_json)) if pool_json.exists() else None, input_file)
        muons[:, 7] = weights.to_spill(muons[:, 7], pot)              # weights per spill of config.POT_PER_SPILL
        self.muons, self.input_file = muons, str(input_file)
        self.charge = torch.from_numpy(muons[:, 6].copy())
        if self.charge.abs().eq(13).all():                            # as run_from_params: PDG id -> charge
            self.charge = self.charge.div(-13)
        assert self.charge.abs().eq(1).all()
        print(f"loaded {n:,} muons from {input_file}, {time.time() - t0:.0f} s", flush=True)

    def run(self, seed):
        """{f'{k}_{tag}'}: every muon at the start, at z = 30.6 and at z = 82 m (or where it stopped)."""
        torch = self.torch
        pos = torch.from_numpy(self.muons[:, 3:6])
        mom = torch.from_numpy(self.muons[:, 0:3])
        cols = {}
        for k, j in zip(STATE, (3, 4, 5, 0, 1, 2)):
            cols[f"{k}_start"] = self.muons[:, j]
        cols["pdg"] = (-13 * self.charge.numpy()).astype(np.float32)     # the classification needs the charge (field map)
        for i, (tag, z) in enumerate(zip(("30", "82"), PLANES)):
            if KEEP_GPU:                                              # leave his kernels PROPAGATION_GB, hold the rest
                torch.cuda.empty_cache()
                ballast = reserve_gpu(self.device, PROPAGATION_GB)
            pos, mom = self.propagate(pos, mom, self.charge, self.environment, z - PLANE_DZ / 2, N_STEPS,
                                      self.step_length, True, seed + i, self.device)
            if KEEP_GPU:
                del ballast
                torch.cuda.empty_cache()
            p, m = pos.numpy(), mom.numpy()
            for k, a, c in (("x", p, 0), ("y", p, 1), ("z", p, 2), ("px", m, 0), ("py", m, 1), ("pz", m, 2)):
                cols[f"{k}_{tag}"] = a[:, c]
        return cols


KEEP_GPU = bool(os.environ.get("VETO_KEEP_GPU_CACHE"))           # hold the card while we run (bin/hold_gpu.py)
PROPAGATION_GB = float(os.environ.get("VETO_PROPAGATION_GB", 20))   # his kernels allocate outside torch, ~16 GB


def reserve_gpu(device, leave_gb=1.5):
    """Take all free memory but leave_gb, so nobody else gets it. Returns the block: kept, it holds the memory
    during the propagation (which allocates outside torch); dropped, it stays in torch's cache for our classification."""
    import torch
    free, _ = torch.cuda.mem_get_info(device)
    n = int(free - leave_gb * 2**30)
    return torch.empty(n, dtype=torch.uint8, device=f"cuda:{device}") if n > 0 else None


def classify(cols, how="gpu", device=0, check=False):
    """Our type of every muon; those that never reach z = 30.6 m are type 7 without the full classification.
    how: "gpu" (muon_types_torch, float64) or "cpu" (muon_types). check: also the other one, compared muon by
    muon on every field; returns (type, report)."""
    typ = np.full(len(cols["z_30"]), 7, np.int8)
    reach = np.nonzero(cols["z_30"] > mt.REACH_30)[0]
    keys = [k for k in cols if k.endswith(("_30", "_82"))]
    sub = {k: cols[k][reach] for k in keys}
    if "pdg" in cols:
        sub["pdg"] = cols["pdg"][reach]
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


ALPHA_COMMON = (4, 5, 6, 7)


def select_alpha(cols, w, typ, out, rng, rows_per_seed, floor_rows, device, chunk=4_000_000):
    """The alpha storage (module doc): rows with Horvitz-Thompson weights w / p, totals per type, the probe histogram."""
    import torch
    from veto import probe
    dev = torch.device(f"cuda:{device}")
    if not KEEP_GPU:
        torch.cuda.empty_cache()                                    # the propagation's cached blocks
    n = len(typ)
    p = np.zeros(n)
    p[typ <= 3] = 1.0
    for t in ALPHA_COMMON:
        m = typ == t
        if m.any():
            p[m] = min(1.0, floor_rows / m.sum())
    pl = probe.planes()
    _, off = probe.grid(pl)
    NC = int(off[-1])
    reach = np.nonzero(cols["z_30"] > mt.REACH_30)[0]          # the classified muons (classify): A at z = 30.6
    k_all = np.stack([out["kx"], out["ky"], out["kz"]], 1)
    hist = torch.zeros((3, NC), dtype=torch.float64, device=dev)
    v_common = torch.zeros(NC, dtype=torch.float64, device=dev)
    q = torch.zeros(len(reach), dtype=torch.float64, device=dev)
    w_t = torch.from_numpy(w[reach]).to(dev)
    common = torch.from_numpy(np.isin(typ[reach], (4, 5, 6))).to(dev)
    seen = torch.from_numpy(typ[reach] <= 6).to(dev)

    def chunks():
        for c0 in range(0, len(reach), chunk):
            r = reach[c0:c0 + chunk]
            g = lambda tag: torch.from_numpy(np.stack([cols[f"{c}_{tag}"][r] for c in "xyz"], 1).astype(np.float64)).to(dev)
            k = torch.from_numpy(k_all[c0:c0 + chunk].astype(np.float64)).to(dev)
            yield c0, probe.cells_torch(g("30"), k, g("82"), pl)

    for c0, (i, c) in chunks():                                 # pass 1: rate maps and the cell rates of types 4-6
        j = i + c0
        use = seen[j]
        hist[0].index_add_(0, c[use], torch.ones(int(use.sum()), dtype=torch.float64, device=dev))
        hist[1].index_add_(0, c[use], w_t[j[use]])
        hist[2].index_add_(0, c[use], w_t[j[use]] ** 2)
        cm = common[j]
        v_common.index_add_(0, c[cm], w_t[j[cm]])
    inv_v = torch.where(v_common > 0, 1.0 / v_common, torch.zeros_like(v_common))
    for c0, (i, c) in chunks():                                 # pass 2: per muon the largest need of its cells
        j = i + c0
        cm = common[j]
        q.scatter_reduce_(0, j[cm], inv_v[c[cm]], reduce="amax")
    f = w_t * q
    lo, hi = 1e-30, 1e30
    for _ in range(200):                                        # lam: rows_per_seed rows of types 4-6 in the cells
        lam = (lo * hi) ** 0.5
        lo, hi = (lam, hi) if float(torch.clamp(lam * f, max=1.0).sum()) < rows_per_seed else (lo, lam)
    p_cell = torch.clamp(lam * f, max=1.0).cpu().numpy()
    p[reach] = np.maximum(p[reach], p_cell)
    keep = np.nonzero(rng.random(n) < p)[0]
    n_by, w_by, w2_by = (np.zeros(N_TYPES, x) for x in (np.int64, np.float64, np.float64))
    for t in range(N_TYPES):
        m = typ == t
        n_by[t], w_by[t], w2_by[t] = m.sum(), w[m].sum(), (w[m] ** 2).sum()
    rows = {k: np.ascontiguousarray(v[keep], np.float32) for k, v in cols.items()}
    rows["weight"] = w[keep] / p[keep]                          # Horvitz-Thompson, then per type to the seed's
    for t in range(N_TYPES):                                    # exact total (as the "type" storage; Hajek)
        m = typ[keep] == t
        if m.any() and rows["weight"][m].sum() > 0:
            rows["weight"][m] *= w_by[t] / rows["weight"][m].sum()
    rows["weight_raw"] = w[keep].astype(np.float64)
    rows["p_keep"] = p[keep].astype(np.float32)
    rows["type"] = typ[keep]
    rows["evt_n"] = keep.astype(np.int64)
    info = dict(lam=lam, rows_cells=int((p_cell > 0).sum()), probe=probe.DEFAULTS)
    return rows, n_by, w_by, w2_by, hist.cpu().numpy(), info


def select_shell(cols, w, typ, out, rng, rows_per_seed, floor_rows, keep_cap, device, chunk=4_000_000):
    """The shell storage (module doc, lib/veto/shell.py), each row kept with probability p and weight w / p:
      types 0-3 (SST1, veto only)        p = 1, up to keep_cap rows per type per seed, beyond that uniform (p = cap / n)
      stopped in the last metre (type 7) p = 1, up to keep_cap rows per seed, likewise
      crossing the shell (types 4-6)     p = min(1, lam w max_c 1 / V_c), V_c = muons per spill of types 4-6 crossing shell cell
                                         c in this seed, lam such that ~rows_per_seed rows are kept: every cell gets about the
                                         same number of equally weighted rows
      every row of types 4-7             at least p = floor_rows / n_type: an unbiased picture of each type
    Also the shell rate map of ALL muons of types 0-6 (counts, sum w, sum w^2 per cell, before any thinning), to check the
    stored rows against. Weights are plain w / p (no per-type rescaling: unbiased per cell)."""
    import torch
    from veto import shell as sh
    dev = torch.device(f"cuda:{device}")
    if not KEEP_GPU:
        torch.cuda.empty_cache()
    n = len(typ)
    p = np.zeros(n)
    capped = {}
    for t in (0, 1, 2, 3):                                      # keep all, with a ceiling
        m = typ == t
        if m.any():
            p[m] = min(1.0, keep_cap / m.sum())
            capped[mt.TYPES[t]] = bool(m.sum() > keep_cap)
    last = (typ == 7) & (cols["z_30"] >= sh.LAST_METRE[0]) & (cols["z_30"] < sh.LAST_METRE[1])
    if last.any():
        p[last] = min(1.0, keep_cap / last.sum())
        capped["stopped in the last metre"] = bool(last.sum() > keep_cap)
    for t in (4, 5, 6, 7):                                      # the floor
        m = typ == t
        if m.any():
            p[m] = np.maximum(p[m], min(1.0, floor_rows / m.sum()))
    reach = np.nonzero(cols["z_30"] > mt.REACH_30)[0]          # the classified muons: A at z = 30.6
    k_all = np.stack([out["kx"], out["ky"], out["kz"]], 1)
    hist = torch.zeros((3, sh.N_CELLS), dtype=torch.float64, device=dev)
    v_common = torch.zeros(sh.N_CELLS, dtype=torch.float64, device=dev)
    q = torch.zeros(len(reach), dtype=torch.float64, device=dev)
    w_t = torch.from_numpy(w[reach]).to(dev)
    common = torch.from_numpy(np.isin(typ[reach], (4, 5, 6))).to(dev)
    seen = torch.from_numpy(typ[reach] <= 6).to(dev)

    def chunks():
        for c0 in range(0, len(reach), chunk):
            r = reach[c0:c0 + chunk]
            g = lambda tag: torch.from_numpy(np.stack([cols[f"{c}_{tag}"][r] for c in "xyz"], 1).astype(np.float64)).to(dev)
            k = torch.from_numpy(k_all[c0:c0 + chunk].astype(np.float64)).to(dev)
            yield c0, sh.crossings_torch(g("30"), k, g("82"))

    for c0, (i, c) in chunks():                                 # pass 1: rate map (all types 0-6), cell rates of types 4-6
        j = i + c0
        use = seen[j]
        hist[0].index_add_(0, c[use], torch.ones(int(use.sum()), dtype=torch.float64, device=dev))
        hist[1].index_add_(0, c[use], w_t[j[use]])
        hist[2].index_add_(0, c[use], w_t[j[use]] ** 2)
        cm = common[j]
        v_common.index_add_(0, c[cm], w_t[j[cm]])
    inv_v = torch.where(v_common > 0, 1.0 / v_common, torch.zeros_like(v_common))
    for c0, (i, c) in chunks():                                 # pass 2: per muon the largest need of the cells it crosses
        j = i + c0
        cm = common[j]
        q.scatter_reduce_(0, j[cm], inv_v[c[cm]], reduce="amax")
    f = w_t * q
    lo, hi = 1e-30, 1e30
    for _ in range(200):
        lam = (lo * hi) ** 0.5
        lo, hi = (lam, hi) if float(torch.clamp(lam * f, max=1.0).sum()) < rows_per_seed else (lo, lam)
    p_cell = torch.clamp(lam * f, max=1.0).cpu().numpy()
    p[reach] = np.maximum(p[reach], p_cell)
    keep = np.nonzero(rng.random(n) < p)[0]
    n_by, w_by, w2_by = (np.zeros(N_TYPES, x) for x in (np.int64, np.float64, np.float64))
    for t in range(N_TYPES):
        m = typ == t
        n_by[t], w_by[t], w2_by[t] = m.sum(), w[m].sum(), (w[m] ** 2).sum()
    rows = {k: np.ascontiguousarray(v[keep], np.float32) for k, v in cols.items()}
    # plain Horvitz-Thompson, NOT rescaled per type to the seed's totals: that rescaling (the other storages) moves each
    # thinned type's total noise coherently into every cell it crosses (seen: +4 % on the shell, 2026-11-07 test)
    rows["weight"] = w[keep] / p[keep]
    rows["weight_raw"] = w[keep].astype(np.float64)
    rows["p_keep"] = p[keep].astype(np.float32)
    rows["type"] = typ[keep]
    rows["evt_n"] = keep.astype(np.int64)
    for f_ in ("kx", "ky", "kz"):                               # the bounce point (shell path), NaN without one
        full = np.full(n, np.nan, np.float32)
        full[reach] = out[f_]
        rows[f_] = full[keep]
    full = np.zeros(n, np.int16)
    full[reach] = out["walls"]
    rows["walls"] = full[keep]
    rows["last_metre"] = last[keep]
    info = dict(lam=lam, rows_shell=int((p_cell > 0).sum()), capped=capped, keep_cap=keep_cap, rows_per_seed=rows_per_seed,
                floor_rows=floor_rows, margin=sh.MARGIN, z_range=[sh.Z0, sh.Z1], cells=[sh.NZ, sh.NA, sh.NC],
                last_metre=list(sh.LAST_METRE))
    return rows, n_by, w_by, w2_by, hist.cpu().numpy(), info


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
    ap.add_argument("--claim", action="store_true", help="processes share one seed list: each claims a seed (.claim_<N>) "
                    "before running it, so faster GPUs take more")
    ap.add_argument("--storage", choices=["type", "alpha", "shell"], default="type", help="type: the budget per type (default); "
                    "alpha: experimental, per probe cell and a floor for every type (docs/storage_policy.md); "
                    "shell: per cell of the veto shell (lib/veto/shell.py), SST1 / veto-only / last-metre stops kept (capped)")
    ap.add_argument("--shell-rows-per-seed", type=float, default=100_000, help="shell: rows of types 4-6 per seed steered by the cells")
    ap.add_argument("--shell-floor", type=float, default=2000, help="shell: uniform rows per seed of each of types 4-7")
    ap.add_argument("--keep-cap", type=float, default=200_000, help="shell: ceiling per seed on the 'keep all' classes "
                    "(each of types 0-3, the last-metre stops); above it they are thinned uniformly, unbiased")
    ap.add_argument("--alpha-rows-per-seed", type=float, default=10_000, help="alpha: rows of types 4-6 per seed steered by the cells")
    ap.add_argument("--alpha-floor", type=float, default=500, help="alpha: uniform rows per seed of each of types 4-7")
    ap.add_argument("--chunk-start", type=int, default=0, help="with --n-chunks: work through the chunks from this one on")
    ap.add_argument("--n-chunks", type=int, default=paths.NOV4_POOL_TRAIN_CHUNKS, help="with --input '.../chunk_{chunk:03d}.h5': seed s uses chunk s %% n")
    ap.add_argument("--input", default=paths.nov4_pool_pattern(), help="input muon h5 (default: the Nov-4 pool chunks, 2026-11-11; "
                    "legacy: MuonsAndMatter's full_sample_after_target.h5 or paths.enriched_input_h5(), with --n-chunks 1)")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)[args.offset::args.step]
    todo = [s for s in seeds if not (out_dir / f"seed_{s}.npz").exists()]
    if args.n_chunks > 1:                     # one chunk at a time (a chunk is ~1e8 muons to load), starting at our own
        todo.sort(key=lambda s: ((s % args.n_chunks - args.chunk_start) % args.n_chunks, s))
    print(f"{len(todo)} of {len(seeds)} seeds to run on GPU {args.gpu}", flush=True)
    if not todo:
        return
    chunk_of = lambda s: args.input.format(chunk=s % args.n_chunks) if "{chunk" in (args.input or "") else args.input
    sim = None
    for seed in todo:
        if (out_dir / f"seed_{seed}.npz").exists():                  # another process got there first
            continue
        if args.claim:
            try:
                os.close(os.open(out_dir / f".claim_{seed}", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                continue
        if sim is None:
            sim = Simulation(args.mm_dir, args.params, args.gpu, args.n_muons, chunk_of(seed))
        elif sim.input_file != str(chunk_of(seed)):
            sim.load(chunk_of(seed))
        w = sim.muons[:, 7].astype(np.float64)
        t0 = time.time()
        cols = sim.run(seed)
        if KEEP_GPU:
            reserve_gpu(args.gpu)
        t_sim = time.time() - t0
        typ, report = classify(cols, args.classify, args.gpu, args.check_classify)
        t_cls = time.time() - t0 - t_sim
        if args.check_classify:
            print("classification check: " + json.dumps({k: v for k, v in report.items() if k != "out"}), flush=True)
        rank = seed - args.first_seed
        extra = {}
        if args.storage == "alpha":
            kept, fraction = np.ones(N_TYPES, bool), np.full(N_TYPES, np.nan)
            rows, n_by, w_by, w2_by, hist, info = select_alpha(cols, w, typ, report["out"], np.random.default_rng([seed, 20261021]),
                                                               args.alpha_rows_per_seed, args.alpha_floor, args.gpu)
            extra = dict(hist=hist)
        elif args.storage == "shell":
            from veto import shell as sh
            kept, fraction = np.ones(N_TYPES, bool), np.full(N_TYPES, np.nan)
            rows, n_by, w_by, w2_by, hist, info = select_shell(cols, w, typ, report["out"], np.random.default_rng([seed, 20261107]),
                                                               args.shell_rows_per_seed, args.shell_floor, args.keep_cap, args.gpu)
            extra = dict(shell_hist=hist, shell_cell_names=np.array(sh.cell_names()))
        else:
            kept, fraction = storage_plan(rank, args.budget, args.min_seeds)
            rows, n_by, w_by, w2_by = select(cols, w, typ, kept, fraction, np.random.default_rng([seed, 20261016]))
        rows["pdg"] = np.sign(sim.charge.numpy()[rows["evt_n"]]).astype(np.int8)   # charge sign, as run_from_params' "pdg"
        if sim.pool_index is not None:
            rows["evt_n"] = sim.pool_index[rows["evt_n"]].astype(np.int64)
        del cols
        meta = dict(seed=seed, rank=rank, budget=args.budget, min_seeds=args.min_seeds, rates=RATES.tolist(),
                    n_input=int(len(typ)), params=args.params, mm_dir=args.mm_dir, input=sim.input_file, planes=PLANES,
                    types=mt.TYPES, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                    settings="TRY6, diluted iron, NI from B, cavern, field simulated, n_steps 5000, every muon to every plane",
                    storage=args.storage, pot_per_spill=POT_PER_SPILL)
        if args.storage == "alpha":
            meta.update(alpha_rows_per_seed=args.alpha_rows_per_seed, alpha_floor=args.alpha_floor, alpha=info)
        elif args.storage == "shell":
            meta.update(shell=info)
            if any(info["capped"].values()):
                print(f"seed {seed}: keep-all ceiling reached for {[k for k, v in info['capped'].items() if v]} "
                      f"(thinned uniformly, weights raised)", flush=True)
        tmp = out_dir / f".seed_{seed}.tmp.npz"
        np.savez(tmp, **rows, n_by_type=n_by, w_by_type=w_by, w2_by_type=w2_by, kept_types=kept,
                 kept_fraction=fraction, rank=rank, seed=seed, meta=json.dumps(meta), **extra)
        os.replace(tmp, out_dir / f"seed_{seed}.npz")                 # readers never see a partial file
        print(f"seed {seed}: {len(rows['type']):,} rows, simulation {t_sim:.0f} s, classification {t_cls:.0f} s, "
              f"total {time.time() - t0:.0f} s | " + " ".join(f"{t}:{n_by[t]}" for t in range(N_TYPES)), flush=True)


if __name__ == "__main__":
    main()
