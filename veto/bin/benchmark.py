#!/usr/bin/env python3
"""Timing and feasibility checks for a design-independent veto evaluation.

Not the veto algorithm itself: this times the building blocks such an algorithm
would need, on the full SST1 store (26 M muons at z = 82 m), on CPU (numpy) and GPU
(torch), and checks a few physics questions with virtual signal vertices.

  bin/remote.sh run bin/benchmark.py
"""

import _bootstrap  # noqa: F401

import argparse
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from veto import geometry as geo
from veto import io, paths
from veto.config import DEFAULT_CONFIG
from veto.pairs import doca_and_vertex, time_window_pairs
from veto.tracks import extrapolate_to_z, vessel_crossings

TIMINGS = []


@contextmanager
def timer(name, n=None, sync=None):
    if sync:
        sync()
    start = time.perf_counter()
    yield
    if sync:
        sync()
    dt = time.perf_counter() - start
    TIMINGS.append((name, n, dt))
    rate = f"   {n / dt / 1e6:8.1f} M/s" if n else ""
    print(f"  {name:<58}{dt * 1e3:10.1f} ms{rate}", flush=True)


# ============================================================
# SBT segmentation used for the per-design evaluation
# ============================================================

N_Z_CELLS, N_ALONG_CELLS = 50, 12      # per wall -> 4 * 600 = 2 400 cells (~1 m x 0.5 m)


def cell_index(face, z, x, y):
    """Cell id on the SBT walls (0 .. 4*N_Z*N_ALONG-1), -1 if not on a wall."""
    wall = face.astype(np.int64) - 2
    half_x, half_y = geo.vessel_half_widths(np.nan_to_num(z, nan=geo.Z_VESSEL_MIN))
    along = np.where(wall < 2, y / half_y, x / half_x)          # -1 .. 1 across the wall
    iz = np.clip(((z - geo.Z_VESSEL_MIN) / (geo.Z_VESSEL_MAX - geo.Z_VESSEL_MIN) * N_Z_CELLS), 0, N_Z_CELLS - 1)
    ia = np.clip((along + 1) / 2 * N_ALONG_CELLS, 0, N_ALONG_CELLS - 1)
    cell = (wall * N_Z_CELLS + np.nan_to_num(iz).astype(np.int64)) * N_ALONG_CELLS + np.nan_to_num(ia).astype(np.int64)
    return np.where((face >= 2) & (face <= 5), cell, -1)


# ============================================================
# torch versions (GPU)
# ============================================================

def torch_crossings(t, x, y, z, px, py, pz):
    vx, vy = px / pz, py / pz
    ax = (geo.VESSEL_X_OUT - geo.VESSEL_X_IN) / (geo.Z_VESSEL_MAX - geo.Z_VESSEL_MIN)
    ay = (geo.VESSEL_Y_OUT - geo.VESSEL_Y_IN) / (geo.Z_VESSEL_MAX - geo.Z_VESSEL_MIN)
    x0, y0 = x - vx * z, y - vy * z
    xw0 = geo.VESSEL_X_IN - ax * geo.Z_VESSEL_MIN
    yw0 = geo.VESSEL_Y_IN - ay * geo.Z_VESSEL_MIN
    c = t.stack([vx - ax, -vx - ax, vy - ay, -vy - ay])
    d = t.stack([xw0 - x0, xw0 + x0, yw0 - y0, yw0 + y0])
    bound = d / c
    inf = t.full_like(bound, float("inf"))
    upper = t.where(c > 0, bound, inf)
    lower = t.where(c < 0, bound, -inf)
    zmin = t.full_like(x, geo.Z_VESSEL_MIN)
    zmax = t.full_like(x, geo.Z_VESSEL_MAX)
    lo, face_in = t.max(t.cat([zmin[None], lower]), dim=0)
    hi, face_out = t.min(t.cat([zmax[None], upper]), dim=0)
    inside = (lo <= hi) & ~((c == 0) & (d < 0)).any(0)
    face_out = t.where(face_out == 0, 6, face_out + 1)
    return lo, hi, t.where(inside, face_in + 1, 0), t.where(inside, face_out, 0)


def torch_doca(t, r, p, pairs):
    r1, r2, p1, p2 = r[pairs[:, 0]], r[pairs[:, 1]], p[pairs[:, 0]], p[pairs[:, 1]]
    dr = r1 - r2
    cross = t.linalg.cross(p1, p2)
    doca = (dr * cross).sum(1).abs() / cross.norm(dim=1)
    a, b, c = (p1 * p1).sum(1), (p1 * p2).sum(1), (p2 * p2).sum(1)
    d, e = (p1 * dr).sum(1), (p2 * dr).sum(1)
    den = a * c - b * b
    s, u = (b * e - c * d) / den, (a * e - b * d) / den
    return doca, 0.5 * (r1 + s[:, None] * p1 + r2 + u[:, None] * p2)


# ============================================================
# Virtual signal: two-track vertices inside the vessel
# ============================================================

def virtual_signal(n, rng):
    """Vertices uniform in the vessel volume, two muon-like tracks pointing into SST1."""
    z_v = rng.uniform(geo.Z_VESSEL_MIN, geo.Z_VESSEL_MAX, 4 * n)
    hx, hy = geo.vessel_half_widths(z_v)
    x_v, y_v = rng.uniform(-1, 1, 4 * n) * hx, rng.uniform(-1, 1, 4 * n) * hy
    # accept uniformly in volume (cross-section grows with z)
    keep = rng.uniform(0, 1, 4 * n) < (hx * hy) / (geo.VESSEL_X_OUT * geo.VESSEL_Y_OUT)
    z_v, x_v, y_v = z_v[keep][:n], x_v[keep][:n], y_v[keep][:n]
    tracks = []
    for _ in range(2):
        # aim at a random point of SST1
        xt = rng.uniform(-geo.SST1_HALF_X, geo.SST1_HALF_X, n)
        yt = rng.uniform(-geo.SST1_HALF_Y, geo.SST1_HALF_Y, n)
        p = rng.uniform(1, 50, n)
        d = np.stack([xt - x_v, yt - y_v, geo.Z_SST1 - z_v])
        d = d / np.linalg.norm(d, axis=0) * p
        tracks.append({"x": xt, "y": yt, "z": np.full(n, geo.Z_SST1), "px": d[0], "py": d[1], "pz": d[2]})
    return (x_v, y_v, z_v), tracks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default=str(paths.store_dir("sst1_TRY6")))
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    print("\n1. LOADING")
    store, meta = io.open_store(Path(args.store))
    n_all = min(meta["n"], args.max_rows or meta["n"])
    keys = ("x2", "y2", "z2", "px2", "py2", "pz2", "weight", "pdg", "file_index")
    with timer("read 9 columns from the memory-mapped store", n_all):
        d = {k.rstrip("2"): np.asarray(store[k][:n_all]) for k in keys}

    print("\n2. SINGLE MUONS, CPU (numpy, one core)")
    with timer("momentum + timing-plane extrapolation + cuts", n_all):
        p = np.sqrt(d["px"] ** 2 + d["py"] ** 2 + d["pz"] ** 2)
        xt, yt = extrapolate_to_z(d["x"], d["y"], d["z"], d["px"], d["py"], d["pz"], DEFAULT_CONFIG["z_timing"])
        keep = (p > 1) & (np.abs(xt) < geo.TIMING_HALF_X) & (np.abs(yt) < geo.TIMING_HALF_Y)
    s = {k: v[keep] for k, v in d.items()}
    n = len(s["x"])
    print(f"     -> {n:,} muons after cuts")
    with timer("vessel clipping (entry + exit face and point), all muons", n_all):
        vessel_crossings(d["x"], d["y"], d["z"], d["px"], d["py"], d["pz"])
    with timer("vessel clipping, muons after cuts", n):
        c = vessel_crossings(s["x"], s["y"], s["z"], s["px"], s["py"], s["pz"])
    with timer("SBT cell ids for entry and exit crossings", n):
        cell_in = cell_index(c["face_in"], c["z_in"], c["x_in"], c["y_in"])
        cell_out = cell_index(c["face_out"], c["z_out"], c["x_out"], c["y_out"])
    ubt_hit = c["face_in"] == 1

    print("\n3. PER-DESIGN EVALUATION (precomputed crossings, design = which cells are live)")
    n_cells = 4 * N_Z_CELLS * N_ALONG_CELLS
    w = s["weight"].astype(np.float64)
    with timer("1 design: veto mask + surviving weight", n):
        active = rng.uniform(size=n_cells) < 0.8
        vetoed = (cell_in >= 0) & active[np.maximum(cell_in, 0)] | ubt_hit
        survive = w[~vetoed].sum()
    designs = rng.uniform(size=(256, n_cells)) < 0.8
    with timer("256 designs at once (CPU)", n * 256):
        hit = (cell_in >= 0)[None, :] & designs[:, np.maximum(cell_in, 0)]
        survive_all = (w[None, :] * ~(hit | ubt_hit[None, :])).sum(1)
    with timer("per-cell weighted hit rates (bincount)", n):
        per_cell = np.bincount(cell_in[cell_in >= 0], weights=w[cell_in >= 0], minlength=n_cells)

    print("\n4. PAIRS, CPU")
    files = s["file_index"]
    order = np.argsort(files, kind="stable")
    counts = np.bincount(files)
    print(f"     muons per spill after cuts: mean {counts.mean():.0f}, max {counts.max()}")
    n_spills = min(2000, len(counts))
    print(f"     pair steps use the first {n_spills} spills; scale by {len(counts) / n_spills:.0f} for all")
    with timer(f"all pairs within each spill (index building, {n_spills} spills)", n):
        starts = np.cumsum(counts) - counts
        pair_list = [order[starts[f]:starts[f] + k] for f, k in enumerate(counts[:n_spills]) if k > 1]
        pairs = np.concatenate([np.column_stack([idx[i], idx[j]]) for idx in pair_list
                                for i, j in [np.triu_indices(len(idx), 1)]])
    print(f"     -> {len(pairs):,} pairs ({len(pairs) / n_spills:,.0f} per spill)")
    sub = pairs[:20_000_000]
    with timer("DOCA + vertex, first 20 M pairs (float64)", len(sub)):
        doca, vertex, _ = doca_and_vertex(s, sub)
    t_rand = rng.uniform(0, 1, n)
    with timer("random times + +-200 ps time-window pairing, all spills merged", n):
        tw_pairs, _ = time_window_pairs(t_rand, 200e-12)
    print(f"     -> {len(tw_pairs):,} time-coincident pairs")

    if not args.no_gpu:
        import torch as t
        dev = "cuda"
        sync = t.cuda.synchronize
        print(f"\n5. GPU ({t.cuda.get_device_name(0)}, shared with other jobs)")
        with timer("host -> GPU copy, muons after cuts", n, sync):
            g = {k: t.as_tensor(s[k], dtype=t.float32, device=dev) for k in ("x", "y", "z", "px", "py", "pz")}
        for _ in range(2):  # second round is warm
            with timer("vessel clipping", n, sync):
                torch_crossings(t, g["x"], g["y"], g["z"], g["px"], g["py"], g["pz"])
        gall = {k: t.as_tensor(d[k], dtype=t.float32, device=dev) for k in ("x", "y", "z", "px", "py", "pz")}
        with timer("vessel clipping, all 26 M muons", n_all, sync):
            torch_crossings(t, gall["x"], gall["y"], gall["z"], gall["px"], gall["py"], gall["pz"])
        del gall
        r = t.stack([g["x"], g["y"], g["z"]], 1)
        pp = t.stack([g["px"], g["py"], g["pz"]], 1)
        gp = t.as_tensor(pairs, device=dev)
        for _ in range(2):
            with timer(f"DOCA + vertex, {len(pairs) / 1e6:.0f} M within-spill pairs (float32, chunks)",
                       len(pairs), sync):
                n_doca = 0
                for k in range(0, len(gp), 50_000_000):
                    dd, _ = torch_doca(t, r, pp, gp[k:k + 50_000_000])
                    n_doca += int((dd < 0.02).sum())
        print(f"     -> {n_doca:,} pairs with DOCA < 2 cm")
        gw = t.as_tensor(w, device=dev)
        gcell = t.as_tensor(np.maximum(cell_in, 0), device=dev)
        gon = t.as_tensor(cell_in >= 0, device=dev)
        gubt = t.as_tensor(ubt_hit, device=dev)
        gdes = t.as_tensor(rng.uniform(size=(1024, n_cells)) < 0.8, device=dev)
        for _ in range(2):
            with timer("1024 designs at once", n * 1024, sync):
                for k in range(0, 1024, 64):
                    hit = gon[None] & gdes[k:k + 64][:, gcell]
                    (gw[None] * ~(hit | gubt[None])).sum(1)

    print("\n6. VIRTUAL SIGNAL (two-track vertices uniform in the vessel, tracks into SST1)")
    (x_v, y_v, z_v), sig = virtual_signal(1_000_000, rng)
    with timer("generate + clip 2 x 1 M signal tracks", 2_000_000):
        sc = [vessel_crossings(tr["x"], tr["y"], tr["z"], tr["px"], tr["py"], tr["pz"]) for tr in sig]
    naive = np.zeros(len(z_v), dtype=bool)
    for cc in sc:
        naive |= cc["face_in"] >= 1         # back-extrapolation reaches the UBT window or an SBT wall
    print(f"     naive back-extrapolation to the vessel boundary would 'hit' a veto for "
          f"{100 * naive.mean():.1f} % of signal events")
    print("     (the veto must only look upstream of the reconstructed vertex; with the true vertex "
          "the signal tracks never touch the SBT before it)")
    sig_r = {k: np.concatenate([sig[0][k], sig[1][k]]) for k in ("x", "y", "z", "px", "py", "pz")}
    nn = len(z_v)
    with timer("DOCA + vertex for the 1 M signal pairs", nn):
        ds, vs, _ = doca_and_vertex(sig_r, np.column_stack([np.arange(nn), np.arange(nn) + nn]))
    print(f"     vertex reconstructed to < 1 mm: {100 * np.mean(np.linalg.norm(vs - np.c_[x_v, y_v, z_v], axis=1) < 1e-3):.1f} %")

    total = sum(dt for _, _, dt in TIMINGS)
    print(f"\nTotal wall time of all timed steps: {total:.1f} s")


if __name__ == "__main__":
    main()
