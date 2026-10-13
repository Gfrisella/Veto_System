#!/usr/bin/env python3
"""Timing + feasibility of digitization and hit-to-track matching per veto design.
Prototype for measurement only, not the veto algorithm.

Per design:
  1. digitize: every muon crossing a live UBT/SBT cell makes a hit (cell, time) with
     efficiency eps and time resolution sigma_t; muons get uniform times in a 1 s spill
  2. hit table: hits sorted by (spill, cell)
  3. matching: each candidate track (from its vertex upstream) predicts the cell it would
     cross; look up hits in that cell and its 8 neighbours within +-dt -> veto
  4. same for virtual signal vertices -> accidental veto probability (signal loss)

No combinatorics: matching is a sorted-table lookup per (track, surface, neighbour).

  bin/remote.sh run bin/benchmark_digitization.py
"""

import _bootstrap  # noqa: F401

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from benchmark import virtual_signal
from benchmark_designs import G, candidates, random_design, sync, track_hits
from veto import io, paths

KEY = 1 << 18        # cells per spill in the hit key (SBT ids < 2^17, UBT ids offset by 2^17)
UBT_OFFSET = 1 << 17
TIME_BITS = 30       # time in ns inside a 1 s spill: key = (spill * KEY + cell) << 30 | t_ns


def cell_neighbours(cells, n_a, ubt):
    """3x3 neighbourhood (SBT: z x along; UBT: x x y with 1000 stride)."""
    stride = 1000 if ubt else n_a
    offs = torch.tensor([dz * stride + da for dz in (-1, 0, 1) for da in (-1, 0, 1)], device=cells.device)
    return cells[:, None] + offs[None, :]


def to_ns(t):
    return torch.clamp(torch.round(t * 1e9), 0, (1 << TIME_BITS) - 1).long()


def lookup(sorted_codes, q_key, q_time, window):
    """Any hit in cell q_key with |t - q_time| < window? Exact, two binary searches per query."""
    base = q_key << TIME_BITS
    lo = torch.searchsorted(sorted_codes, base + to_ns(q_time - window))
    hi = torch.searchsorted(sorted_codes, base + to_ns(q_time + window), right=True)
    return hi > lo


def predicted_keys(tr, z_from, spill, d, dead, dev):
    _, _, ubt, n_a = track_hits(tr, z_from, d, dead, dev)
    walls = track_hits.per_wall
    out = []
    for cells, is_ubt, offset in [(walls[:, w], False, 0) for w in range(4)] + [(ubt, True, UBT_OFFSET)]:
        nb = cell_neighbours(cells, n_a, is_ubt) + offset
        valid = (cells >= 0)[:, None].expand_as(nb)
        out.append(torch.where(valid, spill[:, None] * KEY + nb, -1))
    return torch.cat(out, 1)          # (n, 45)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default=str(paths.store_dir("sst1_TRY6")))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--designs", type=int, default=20)
    parser.add_argument("--efficiency", type=float, default=0.99)
    parser.add_argument("--sigma-t", type=float, default=0.5e-9)
    parser.add_argument("--window", type=float, default=1.5e-9)
    args = parser.parse_args()
    dev = args.device
    rng = np.random.default_rng(0)
    gen = torch.Generator(device=dev).manual_seed(0)

    store, meta = io.open_store(Path(args.store))
    keys = {"x": "x2", "y": "y2", "z": "z2", "px": "px2", "py": "py2", "pz": "pz2", "weight": "weight",
            "pdg": "pdg", "file_index": "file_index"}
    host = {k: np.asarray(store[v]) for k, v in keys.items()}

    T = {}

    def tick(name, start):
        sync(dev)
        T[name] = time.perf_counter() - start
        return time.perf_counter()

    t0 = time.perf_counter()
    t = {k: torch.as_tensor(np.array(v), device=dev) for k, v in host.items()}
    t["file_index"] = t["file_index"].long()
    t0 = tick("copy to device", t0)
    p = torch.sqrt(t["px"] ** 2 + t["py"] ** 2 + t["pz"] ** 2)
    dz = G.Z_TIMING - t["z"]
    keep = (p > 1) & ((t["x"] + t["px"] / t["pz"] * dz).abs() < 2) & ((t["y"] + t["py"] / t["pz"] * dz).abs() < 3)
    s = {k: v[keep] for k, v in t.items()}
    s["time"] = torch.rand(len(s["x"]), device=dev, generator=gen)          # 1 s spill
    t0 = tick("cuts", t0)
    pairs, vertex, _ = candidates(s, dev, dtype=torch.float32)
    t0 = tick("candidates (pairs, DOCA, vessel, IP)", t0)

    a, b = pairs[:, 0], pairs[:, 1]
    n_c = len(a)
    cand = {k: torch.cat([s[k][a], s[k][b]]) for k in ("x", "y", "z", "px", "py", "pz", "time", "file_index")}
    z_from = torch.cat([vertex[:, 2], vertex[:, 2]])
    pair_w = s["weight"][a].double() * s["weight"][b].double()

    # virtual signal: vertices in the vessel, random spill and time
    (xv, yv, zv), sig = virtual_signal(200_000, rng)
    n_s = len(zv)
    sig_t = {k: torch.as_tensor(np.concatenate([sig[0][k], sig[1][k]]), dtype=torch.float32, device=dev)
             for k in ("x", "y", "z", "px", "py", "pz")}
    sig_spill = torch.as_tensor(rng.integers(0, int(s["file_index"].max()) + 1, n_s), device=dev).repeat(2)
    sig_time = torch.rand(n_s, device=dev, generator=gen).repeat(2)
    sig_z = torch.as_tensor(np.concatenate([zv, zv]), dtype=torch.float32, device=dev)
    z_all = torch.full_like(s["x"], G.Z_VESSEL_MAX + 1)

    print(f"{len(s['x']):,} muons, {n_c:,} background candidates, {n_s:,} virtual signal vertices\n")
    per_design = []
    for i in range(args.designs):
        d = random_design(rng)
        dead = torch.as_tensor(rng.uniform(size=200_000) < d["dead_fraction"], device=dev)
        sync(dev)
        start = time.perf_counter()
        # 1. digitize all muons (every SBT wall crossing + UBT crossing)
        _, _, ubt, n_a = track_hits(s, z_all, d, dead, dev)
        walls = track_hits.per_wall
        hit_cells = torch.cat([walls.T.reshape(-1), torch.where(ubt >= 0, ubt + (UBT_OFFSET), -1)])
        hit_spill = s["file_index"].repeat(5)
        hit_time = s["time"].repeat(5) + args.sigma_t * torch.randn(5 * len(ubt), device=dev, generator=gen)
        live = (hit_cells >= 0) & (torch.rand(len(hit_cells), device=dev, generator=gen) < args.efficiency)
        live &= ~dead[torch.clamp(hit_cells, 0, len(dead) - 1)] | (hit_cells >= (UBT_OFFSET))
        # 2. table
        codes = torch.sort(((hit_spill[live] * KEY + hit_cells[live]) << TIME_BITS) | to_ns(hit_time[live])).values
        # 3. background candidates
        q = predicted_keys(cand, z_from, cand["file_index"], d, dead, dev)
        found = lookup(codes, q, cand["time"][:, None].expand_as(q), args.window) & (q >= 0)
        track_veto = found.any(1)
        event_veto = track_veto[:n_c] | track_veto[n_c:]
        surviving = pair_w[~event_veto].sum()
        # 4. signal
        qs = predicted_keys(sig_t, sig_z, sig_spill, d, dead, dev)
        fs = (lookup(codes, qs, sig_time[:, None].expand_as(qs), args.window) & (qs >= 0)).any(1)
        sig_loss = (fs[:n_s] | fs[n_s:]).float().mean()
        sync(dev)
        per_design.append(time.perf_counter() - start)
        if i < 3:
            print(f"  design {i}: {int(live.sum()):,} hits, background candidates vetoed "
                  f"{100 * event_veto.float().mean():.2f} %, surviving weight {float(surviving):.3e}, "
                  f"signal accidentally vetoed {100 * float(sig_loss):.4f} %")

    pd = np.array(per_design[1:])
    print("\nOnce per session (data in memory):")
    for name, dt in T.items():
        print(f"  {name:<40}{dt:8.2f} s")
    print(f"Per design (digitize + table + match background + match signal): "
          f"median {1e3 * np.median(pd):.0f} ms, max {1e3 * pd.max():.0f} ms on {dev}")
    print(f"End to end for ONE design from data in memory: {sum(T.values()) + np.median(pd):.2f} s")


if __name__ == "__main__":
    main()
