#!/usr/bin/env python3
"""Timing check for evaluating arbitrary veto geometries (not the veto algorithm).

Design-independent part (once per run, data already in memory):
  cuts -> all within-spill pairs -> DOCA < 2 cm -> vertex in vessel -> IP  => candidate vertices

Per design (random, to measure cost):
  UBT  : rectangle at z_ubt with arbitrary half sizes and cell size
  SBT  : the 4 walls pushed out by an offset, each covering only [z_a, z_b], arbitrary
         cell sizes, a random fraction of dead cells
  every candidate track is followed from its vertex back to the UBT plane; it is vetoed
  if it crosses a live cell. Also: per-cell occupancy from all single muons.

  bin/remote.sh run bin/benchmark_designs.py [--device cpu]
"""

import _bootstrap  # noqa: F401

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from veto import geometry as geo
from veto import io, paths

G = geo


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def candidates(t, dev, spill_chunk=400, dtype=torch.float64, opposite_charge=True):
    """Within-spill pairs passing DOCA < 2 cm, vertex in vessel, IP < 2.5 m (torch, chunked)."""
    file_index = t["file_index"]
    order = torch.argsort(file_index, stable=True)
    counts = torch.bincount(file_index)
    starts = torch.cumsum(counts, 0) - counts
    m = int(counts.max())
    n_spill = len(counts)
    # padded (spill, slot) -> muon index
    slot = torch.arange(m, device=dev)
    pad = starts[:, None] + slot[None, :]
    valid = slot[None, :] < counts[:, None]
    pad = torch.where(valid, order[torch.clamp(pad, max=len(order) - 1)], -1)
    iu, ju = torch.triu_indices(m, m, 1, device=dev)
    r = torch.stack([t["x"], t["y"], t["z"]], 1).to(dtype)
    p = torch.stack([t["px"], t["py"], t["pz"]], 1).to(dtype)
    out, n_pairs = [], 0
    for s in range(0, n_spill, spill_chunk):
        a, b = pad[s:s + spill_chunk][:, iu].reshape(-1), pad[s:s + spill_chunk][:, ju].reshape(-1)
        ok = (a >= 0) & (b >= 0)
        a, b = a[ok], b[ok]
        if opposite_charge:
            q = t["pdg"][a] * t["pdg"][b] < 0
            a, b = a[q], b[q]
        n_pairs += len(a)
        a, b, v = _pair_cut(a, b, r, p)
        out.append((torch.stack([a, b], 1), v))
    pairs = torch.cat([o[0] for o in out])
    vertex = torch.cat([o[1] for o in out])
    return pairs, vertex, n_pairs


def _pair_cut(a, b, r, p):
    """The candidate cuts on muon pairs (a, b): DOCA < 2 cm, vertex in the vessel, IP < 2.5 m."""
    r1, r2, p1, p2 = r[a], r[b], p[a], p[b]
    dr = r1 - r2
    cross = torch.linalg.cross(p1, p2)
    doca = (dr * cross).sum(1).abs() / cross.norm(dim=1)
    aa, bb, cc = (p1 * p1).sum(1), (p1 * p2).sum(1), (p2 * p2).sum(1)
    dd, ee = (p1 * dr).sum(1), (p2 * dr).sum(1)
    den = aa * cc - bb * bb
    v = 0.5 * (r1 + ((bb * ee - cc * dd) / den)[:, None] * p1 + r2 + ((aa * ee - bb * dd) / den)[:, None] * p2)
    hx = G.VESSEL_X_IN + (G.VESSEL_X_OUT - G.VESSEL_X_IN) * (v[:, 2] - G.Z_VESSEL_MIN) / 50
    hy = G.VESSEL_Y_IN + (G.VESSEL_Y_OUT - G.VESSEL_Y_IN) * (v[:, 2] - G.Z_VESSEL_MIN) / 50
    mom = p1 + p2
    dz = G.Z_TARGET - v[:, 2]
    ip = torch.hypot(v[:, 0] + mom[:, 0] / mom[:, 2] * dz, v[:, 1] + mom[:, 1] / mom[:, 2] * dz)
    keep = ((doca < 0.02) & (v[:, 2] > G.Z_VESSEL_MIN) & (v[:, 2] < G.Z_VESSEL_MAX)
            & (v[:, 0].abs() < hx) & (v[:, 1].abs() < hy) & (ip < 2.5))
    return a[keep], b[keep], v[keep]


def candidates_blocked(t, dev, block=4096, dtype=torch.float64, opposite_charge=True):
    """candidates() for big spills (tens of thousands of muons each): every spill's pairs in blocks of
    block x block muons, so memory stays bounded. Same pairs, same cuts."""
    fi = t["file_index"].cpu().numpy()
    order = np.argsort(fi, kind="stable")
    counts = np.bincount(fi)
    starts = np.cumsum(counts) - counts
    r = torch.stack([t["x"], t["y"], t["z"]], 1).to(dtype)
    p = torch.stack([t["px"], t["py"], t["pz"]], 1).to(dtype)
    out, n_pairs = [], 0
    for s0, c in zip(starts, counts):
        if c < 2:
            continue
        idx = torch.as_tensor(order[s0:s0 + c], device=dev)
        for i0 in range(0, c, block):
            for j0 in range(i0, c, block):
                ni, nj = min(block, c - i0), min(block, c - j0)
                ia = torch.arange(ni, device=dev)[:, None].expand(ni, nj)
                jb = torch.arange(nj, device=dev)[None, :].expand(ni, nj)
                m = (i0 + ia) < (j0 + jb)
                a, b = idx[i0 + ia[m]], idx[j0 + jb[m]]
                if opposite_charge:
                    q = t["pdg"][a] * t["pdg"][b] < 0
                    a, b = a[q], b[q]
                n_pairs += len(a)
                a, b, v = _pair_cut(a, b, r, p)
                out.append((torch.stack([a, b], 1), v))
    return torch.cat([o[0] for o in out]), torch.cat([o[1] for o in out]), n_pairs


def random_design(rng):
    d = {
        "z_ubt": rng.uniform(29.0, 30.6),
        "ubt_hx": rng.uniform(0.5, 3.0), "ubt_hy": rng.uniform(1.35, 3.0),
        "ubt_cell": rng.uniform(0.05, 0.5),
        "sbt_offset": rng.uniform(0.0, 0.3),
        "sbt_cell_z": rng.uniform(0.2, 2.0), "sbt_cell_a": rng.uniform(0.1, 1.0),
        "dead_fraction": rng.uniform(0.0, 0.2),
    }
    d["sbt_z"] = np.sort(rng.uniform(G.Z_VESSEL_MIN, G.Z_VESSEL_MAX, (4, 2)), axis=1)   # coverage per wall
    d["sbt_z"][:, 0] = np.minimum(d["sbt_z"][:, 0], G.Z_VESSEL_MIN + 10)
    return d


def track_hits(tr, z_from, d, dead_cells, dev):
    """Does each track cross a live UBT or SBT cell between z_from (its vertex) and the UBT?

    tr: dict of tensors x, y, z, px, py, pz (state at SST1). Returns (vetoed, cell_ids of SBT hits)."""
    vx, vy = tr["px"] / tr["pz"], tr["py"] / tr["pz"]
    x0, y0 = tr["x"] - vx * tr["z"], tr["y"] - vy * tr["z"]
    vetoed = torch.zeros_like(vx, dtype=torch.bool)

    # UBT: plane at z_ubt, rectangle, cells
    xu, yu = x0 + vx * d["z_ubt"], y0 + vy * d["z_ubt"]
    ubt_on = (xu.abs() < d["ubt_hx"]) & (yu.abs() < d["ubt_hy"]) & (z_from > d["z_ubt"])
    vetoed |= ubt_on
    ubt_cell = (torch.floor((xu + d["ubt_hx"]) / d["ubt_cell"]) * 1000
                + torch.floor((yu + d["ubt_hy"]) / d["ubt_cell"])).long()

    # SBT walls: x = +-(X(z) + offset), y = +-(Y(z) + offset)
    ax = (G.VESSEL_X_OUT - G.VESSEL_X_IN) / 50
    ay = (G.VESSEL_Y_OUT - G.VESSEL_Y_IN) / 50
    xw0 = G.VESSEL_X_IN - ax * G.Z_VESSEL_MIN + d["sbt_offset"]
    yw0 = G.VESSEL_Y_IN - ay * G.Z_VESSEL_MIN + d["sbt_offset"]
    n_a = int(np.ceil(2 * (G.VESSEL_Y_OUT + 0.3) / d["sbt_cell_a"]))
    n_z = int(np.ceil(50 / d["sbt_cell_z"]))
    cells = torch.full_like(vx, -1, dtype=torch.long)
    per_wall = []
    for w, (sign, along_x) in enumerate(((1, False), (-1, False), (1, True), (-1, True))):
        slope, w0, coord0, v = (ay, yw0, y0, vy) if along_x else (ax, xw0, x0, vx)
        # coord0 + v z = sign * (w0 + slope z)
        z_hit = (sign * w0 - coord0) / (v - sign * slope)
        x_hit, y_hit = x0 + vx * z_hit, y0 + vy * z_hit
        other = x_hit if along_x else y_hit
        half_other = (xw0 + ax * z_hit) if along_x else (yw0 + ay * z_hit)
        z_a, z_b = d["sbt_z"][w]
        on = (z_hit >= z_a) & (z_hit <= z_b) & (z_hit < z_from) & (other.abs() <= half_other)
        cid = (w * n_z + torch.floor((z_hit - G.Z_VESSEL_MIN) / d["sbt_cell_z"]).long()) * n_a \
            + torch.floor((other + half_other) / d["sbt_cell_a"]).long()
        cid = torch.clamp(cid, 0, len(dead_cells) - 1)
        live = on & ~dead_cells[cid]
        vetoed |= live
        cells = torch.where(on, cid, cells)
        per_wall.append(torch.where(on, cid, -1))
    track_hits.per_wall = torch.stack(per_wall, 1)      # (n, 4): every wall crossing
    return vetoed, cells, torch.where(ubt_on, ubt_cell, -1), n_a


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default=str(paths.store_dir("sst1_TRY6")))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--designs", type=int, default=50)
    args = parser.parse_args()
    dev = args.device
    rng = np.random.default_rng(0)

    store, meta = io.open_store(Path(args.store))
    keys = {"x": "x2", "y": "y2", "z": "z2", "px": "px2", "py": "py2", "pz": "pz2", "weight": "weight",
            "pdg": "pdg", "file_index": "file_index"}
    host = {k: np.asarray(store[v]) for k, v in keys.items()}
    print(f"loaded {meta['n']:,} muons (not timed)\n")

    t0 = time.perf_counter()
    t = {k: torch.as_tensor(np.array(v), device=dev) for k, v in host.items()}
    t["file_index"] = t["file_index"].long()
    sync(dev)
    t_copy = time.perf_counter() - t0

    # ---- design-independent ----
    t0 = time.perf_counter()
    p = torch.sqrt(t["px"] ** 2 + t["py"] ** 2 + t["pz"] ** 2)
    dz = G.Z_TIMING - t["z"]
    keep = (p > 1) & ((t["x"] + t["px"] / t["pz"] * dz).abs() < 2) & ((t["y"] + t["py"] / t["pz"] * dz).abs() < 3)
    s = {k: v[keep] for k, v in t.items()}
    sync(dev)
    t_cuts = time.perf_counter() - t0

    for dtype in (torch.float32, torch.float64):
        t0 = time.perf_counter()
        pairs, vertex, n_pairs = candidates(s, dev, dtype=dtype)
        sync(dev)
        t_cand = time.perf_counter() - t0
        print(f"  {str(dtype):<14} opposite-charge pairs+DOCA+vessel+IP {t_cand:.2f} s -> {len(pairs):,} candidates")
    print(f"design-independent:  host->{dev} copy {t_copy:.2f} s | cuts {t_cuts:.3f} s ({len(s['x']):,} muons) | "
          f"pairs+DOCA+vessel+IP {t_cand:.2f} s ({n_pairs:,} pairs -> {len(pairs):,} candidates)")

    # candidate tracks, followed from their vertex upstream
    a, b = pairs[:, 0], pairs[:, 1]
    tracks = {k: torch.cat([s[k][a], s[k][b]]) for k in ("x", "y", "z", "px", "py", "pz")}
    z_from = torch.cat([vertex[:, 2], vertex[:, 2]])
    pair_w = s["weight"][a].double() * s["weight"][b].double()
    single = {k: s[k] for k in ("x", "y", "z", "px", "py", "pz")}
    z_single = torch.full_like(s["x"], G.Z_VESSEL_MAX)

    times = []
    for i in range(args.designs):
        d = random_design(rng)
        dead = torch.as_tensor(rng.uniform(size=200_000) < d["dead_fraction"], device=dev)
        sync(dev)
        t0 = time.perf_counter()
        vetoed, *_ = track_hits(tracks, z_from, d, dead, dev)
        n = len(pair_w)
        event_veto = vetoed[:n] | vetoed[n:]
        background = pair_w[~event_veto].sum()
        # occupancy of every SBT cell from all single muons (accidental-veto / rate input)
        _, cells, *_ = track_hits(single, z_single, d, dead, dev)
        occupancy = torch.bincount(cells[cells >= 0], weights=s["weight"][cells >= 0].double(), minlength=1)
        sync(dev)
        times.append(time.perf_counter() - t0)
        if i < 3:
            print(f"  design {i}: UBT {2 * d['ubt_hx']:.1f}x{2 * d['ubt_hy']:.1f} m at z={d['z_ubt']:.1f}, "
                  f"SBT offset {d['sbt_offset']:.2f} m, cells {d['sbt_cell_z']:.2f}x{d['sbt_cell_a']:.2f} m, "
                  f"dead {100 * d['dead_fraction']:.0f} % -> vetoed {100 * event_veto.double().mean():.1f} % "
                  f"of candidates, surviving weight {float(background):.3e}, {int((occupancy > 0).sum())} hit cells")
    times = np.array(times[1:])
    print(f"\nper design ({dev}): median {1e3 * np.median(times):.1f} ms, max {1e3 * times.max():.1f} ms "
          f"(candidates: {len(pair_w):,} vertices, occupancy: {len(single['x']):,} muons)")


if __name__ == "__main__":
    main()
