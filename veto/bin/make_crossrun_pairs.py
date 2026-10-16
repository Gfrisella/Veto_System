#!/usr/bin/env python3
"""An independent sample of background pair candidates: muons of DIFFERENT old runs paired together.

inputs.npz pairs the muons of the same run (every within-run pair passing the cuts: 196 736). Muons of
different runs are just as independent, so pairing run r with run r + s (for each --shift s) gives new
pairs from the same muons, with the same cuts (opposite charge, DOCA < 2 cm, vertex in the vessel,
IP < 2.5 m). It tests everything that depends on which muons are paired (the surviving-pair count, the
factorised estimate), not the muons themselves: a design tuned to these muons is not re-tested.

Weights: a run pair (r, r + s) has about twice as many muon pairs as one run has within itself
(n^2 against n^2 / 2), so w = w_a * w_b * n_runs / (2 * number of shifts): the total matches inputs.npz.
Writes results/reco/inputs_xrun.npz, same layout as inputs.npz (same single muons and rate maps).

  bin/remote.sh run bin/make_crossrun_pairs.py            # GPU, ~1 min
"""

import _bootstrap  # noqa: F401

import argparse
import time

import numpy as np
import torch

from benchmark_designs import candidates
from make_reco_inputs import COLS, DATASET
from veto import geometry as geo
from veto import io, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shift", type=int, nargs="+", default=[1, 2], help="pair run r with run r + s")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    start = time.time()

    store, meta = io.open_store(paths.store_dir(DATASET))
    n_runs = meta["sources"]["old"]["n_files"]
    types = np.asarray(store["type"])
    sst1 = np.where(types <= 2)[0]
    d = {k: np.asarray(store[c][sst1]) for k, c in COLS.items()}
    p = np.sqrt(d["bpx"] ** 2 + d["bpy"] ** 2 + d["bpz"] ** 2)
    dz = geo.Z_TIMING - d["bz"]
    good = ((p > 1) & (d["bpz"] > 0) & (np.abs(d["bx"] + d["bpx"] / d["bpz"] * dz) < geo.TIMING_HALF_X)
            & (np.abs(d["by"] + d["bpy"] / d["bpz"] * dz) < geo.TIMING_HALF_Y))
    g = np.where(good)[0]
    run_ids, run = np.unique(d["run"][g], return_inverse=True)       # 0 .. N-1
    n = len(run_ids)
    dev = args.device if torch.cuda.is_available() else "cpu"

    all_pairs, all_vertex = [], []
    for s in args.shift:
        # each muon twice: as the first member of group run, and as the second member of group run - s
        idx = np.concatenate([g, g])
        role = np.concatenate([np.zeros(len(g), int), np.ones(len(g), int)])
        group = np.concatenate([run, (run - s) % n])
        t = {k: torch.as_tensor(d[c][idx].astype(np.float64), device=dev)
             for k, c in (("x", "bx"), ("y", "by"), ("z", "bz"), ("px", "bpx"), ("py", "bpy"), ("pz", "bpz"))}
        t["pdg"] = torch.as_tensor(d["pdg"][idx].astype(np.float64), device=dev)
        t["file_index"] = torch.as_tensor(group, device=dev)
        pairs, vertex, _ = candidates(t, dev, spill_chunk=100)
        pairs, vertex = pairs.cpu().numpy(), vertex.cpu().numpy()
        cross = role[pairs[:, 0]] != role[pairs[:, 1]]               # one muon of run r, one of run r + s
        all_pairs.append(idx[pairs[cross]])
        all_vertex.append(vertex[cross])
        print(f"shift {s}: {cross.sum():,} cross-run candidates, {time.time() - start:.0f} s")
    pairs, vertex = np.concatenate(all_pairs), np.concatenate(all_vertex)
    w_pair = d["weight"][pairs[:, 0]].astype(np.float64) * d["weight"][pairs[:, 1]] * n_runs / (2 * len(args.shift))

    ref = np.load(paths.results_dir() / "reco" / "inputs.npz")
    print(f"{len(pairs):,} candidates, {w_pair.sum():.4g} pairs / spill before the time coincidence "
          f"(within-run sample: {len(ref['cand']):,}, {ref['cand_w'].sum():.4g})")
    # the same single muons as inputs.npz: make_reco_inputs.py's draw (seed 3), same SST1 index space
    single = np.sort(np.random.default_rng(3).choice(len(sst1), min(200_000, len(sst1)), replace=False))
    rows = np.unique(np.concatenate([pairs.ravel(), single]))
    np.savez(paths.results_dir() / "reco" / "inputs_xrun.npz",
             **{f"tr_{k}": v[rows] for k, v in d.items()},
             cand=np.searchsorted(rows, pairs), cand_vertex=vertex, cand_w=w_pair,
             single=np.searchsorted(rows, single), single_w=ref["single_w"],
             **{k: ref[k] for k in ("fin_maps", "z_slices", "u_bins", "y_bins", "ubt_hist", "ubt_y_bins", "n_runs")})
    print(f"saved inputs_xrun.npz: {len(rows):,} tracks, {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
