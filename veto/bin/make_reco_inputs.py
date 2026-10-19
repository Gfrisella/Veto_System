#!/usr/bin/env python3
"""Inputs of the reconstruction study, built once from the design-optimisation dataset.

  background candidates  pairs of SST1 muons of the same old run (p > 1 GeV, timing acceptance,
                         opposite charge) with DOCA < 2 cm, vertex in the vessel, IP < 2.5 m.
                         weight = expected such pairs per spill BEFORE the time coincidence
                         (w_a * w_b * n_runs; multiply by 2 dt / spill for random coincidences)
  single tracks          random sample of SST1-acceptance muons (all three types), weights
                         rescaled so every type keeps its total: single-muon veto efficiency
  rate maps              UBT and fin rate maps of every muon (from results/design_explorer)

Tracks keep their true path (A at 30.6, bounce point K, B at 82) for the hits and their state at
B for the spectrometer. Writes results/reco/inputs.npz.

  bin/remote.sh run bin/make_reco_inputs.py            # GPU, ~1 min
"""

import _bootstrap  # noqa: F401

import argparse
import time

import numpy as np
import torch

from benchmark_designs import candidates, candidates_blocked
from make_design_explorer import U_BINS, UBT_Y_BINS, Y_BINS, Z_SLICES
from veto import geometry as geo
from veto import io, paths
from veto import muon_types as mt

DATASET = "design-optimization-dataset-alpha-14-oct-2026"
COLS = {"ax": "x_30", "ay": "y_30", "az": "z_30", "apz": "pz_30", "kx": "kx", "ky": "ky", "kz": "kz",
        "bx": "x_82", "by": "y_82", "bz": "z_82", "bpx": "px_82", "bpy": "py_82", "bpz": "pz_82",
        "weight": "weight", "type": "type", "run": "run", "pdg": "pdg"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-single", type=int, default=200_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--maps", default="sample.npz", help="rate maps: results/design_explorer/<this>")
    parser.add_argument("--out", default="inputs.npz", help="file name in results/reco")
    parser.add_argument("--pair-block", type=int, default=0, help="> 0: pairs in blocks of this many muons per spill (big spills)")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    start = time.time()

    store, meta = io.open_store(paths.store_dir(args.dataset))
    types = np.asarray(store["type"])
    sst1 = np.where(types <= 2)[0]                     # the SST1 types: every run / seed keeps all of them
    if "sources" in meta:                              # alpha: the old runs
        n_runs = meta["sources"]["old"]["n_files"]
    else:                                              # our production (gamma builder): seeds that kept SST1
        n_runs = {meta["per_type"][mt.TYPES[t]]["seeds"] for t in range(3)}
        assert len(n_runs) == 1, f"SST1 types from different numbers of seeds: {n_runs}"
        n_runs = n_runs.pop()
    d = {k: np.asarray(store[c][sst1]) for k, c in COLS.items()}
    print(f"{len(sst1):,} SST1-acceptance muons loaded, {time.time() - start:.0f} s")

    p = np.sqrt(d["bpx"] ** 2 + d["bpy"] ** 2 + d["bpz"] ** 2)
    dz = geo.Z_TIMING - d["bz"]
    good = ((p > 1) & (d["bpz"] > 0) & (np.abs(d["bx"] + d["bpx"] / d["bpz"] * dz) < geo.TIMING_HALF_X)
            & (np.abs(d["by"] + d["bpy"] / d["bpz"] * dz) < geo.TIMING_HALF_Y))
    g = np.where(good)[0]
    dev = args.device if torch.cuda.is_available() else "cpu"
    t = {k: torch.as_tensor(d[c][g].astype(np.float64), device=dev)
         for k, c in (("x", "bx"), ("y", "by"), ("z", "bz"), ("px", "bpx"), ("py", "bpy"), ("pz", "bpz"))}
    t["pdg"] = torch.as_tensor(d["pdg"][g].astype(np.float64), device=dev)
    t["file_index"] = torch.as_tensor(d["run"][g].astype(np.int64), device=dev)
    pairs, vertex, n_pairs = (candidates_blocked(t, dev, args.pair_block) if args.pair_block else candidates(t, dev))
    pairs, vertex = g[pairs.cpu().numpy()], vertex.cpu().numpy()
    w_pair = d["weight"][pairs[:, 0]].astype(np.float64) * d["weight"][pairs[:, 1]] * n_runs
    print(f"{good.sum():,} muons with p > 1 GeV in the timing acceptance, {n_pairs:,} opposite-charge pairs "
          f"in the same run -> {len(pairs):,} candidates ({w_pair.sum():.4g} pairs / spill before the time "
          f"coincidence), {time.time() - start:.0f} s")

    single = np.sort(rng.choice(len(sst1), min(args.n_single, len(sst1)), replace=False))
    w_single = d["weight"][single].astype(np.float64)
    for ty in range(3):
        m = d["type"][single] == ty
        w_single[m] *= d["weight"][d["type"] == ty].astype(np.float64).sum() / w_single[m].sum()

    rows = np.unique(np.concatenate([pairs.ravel(), single]))
    index = {k: np.searchsorted(rows, v) for k, v in (("pairs", pairs), ("single", single))}
    maps = np.load(paths.results_dir() / "design_explorer" / args.maps)
    outdir = paths.results_dir() / "reco"
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(outdir / args.out,
             **{f"tr_{k}": v[rows] for k, v in d.items()},
             cand=index["pairs"], cand_vertex=vertex, cand_w=w_pair,
             single=index["single"], single_w=w_single,
             fin_maps=maps["fin_maps"], z_slices=Z_SLICES, u_bins=U_BINS, y_bins=Y_BINS,
             ubt_hist=maps["ubt_hist"], ubt_y_bins=UBT_Y_BINS, n_runs=n_runs)
    print(f"saved {outdir / args.out}: {len(rows):,} tracks, {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
