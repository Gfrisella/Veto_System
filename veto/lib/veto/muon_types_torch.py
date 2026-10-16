"""muon_types.classify on the GPU (torch, float64): the same formulas, line by line, for the production.

classify(d, device) takes the same dict of numpy arrays (x_30 .. pz_30, x_82 .. pz_82) and returns the
same dict (type, walls, kx, ky, kz, off) as numpy arrays. Every step mirrors muon_types.py / tracks.py
in float64 and in the same order, so the result matches the CPU version (checked muon by muon in
bin/generate_muon_study.py --check-classify); hypot and the square roots are the only library calls
that may differ in the last bit between CPU and GPU.
"""

import math

import numpy as np
import torch

from . import geometry as geo
from . import muon_types as mt

_FACE_BIT = {2: 1, 3: 2, 4: 4, 5: 8}
_AX = (geo.VESSEL_X_OUT - geo.VESSEL_X_IN) / (geo.Z_VESSEL_MAX - geo.Z_VESSEL_MIN)
_AY = (geo.VESSEL_Y_OUT - geo.VESSEL_Y_IN) / (geo.Z_VESSEL_MAX - geo.Z_VESSEL_MIN)
_X_W0 = geo.VESSEL_X_IN - _AX * geo.Z_VESSEL_MIN
_Y_W0 = geo.VESSEL_Y_IN - _AY * geo.Z_VESSEL_MIN


def _norm(p):
    return torch.sqrt((p * p).sum(1))          # np.linalg.norm(p, axis=1)


def vessel_crossings(x, y, z, px, py, pz):
    """tracks.vessel_crossings: z_in, z_out, face_in, face_out of the line inside the vessel."""
    vx, vy = px / pz, py / pz
    x_at0 = x - vx * z
    y_at0 = y - vy * z
    lower = torch.full_like(x, geo.Z_VESSEL_MIN)
    upper = torch.full_like(x, geo.Z_VESSEL_MAX)
    face_in = torch.ones_like(x, dtype=torch.int8)
    face_out = torch.full_like(x, 6, dtype=torch.int8)
    feasible = torch.isfinite(vx) & torch.isfinite(vy)
    for face, c, d in ((2, vx - _AX, _X_W0 - x_at0), (3, -vx - _AX, _X_W0 + x_at0),
                       (4, vy - _AY, _Y_W0 - y_at0), (5, -vy - _AY, _Y_W0 + y_at0)):
        bound = d / c
        feasible &= ~((c == 0) & (d < 0))
        tighter = (c > 0) & (bound < upper)
        upper = torch.where(tighter, bound, upper)
        face_out = torch.where(tighter, torch.tensor(face, dtype=torch.int8, device=x.device), face_out)
        tighter = (c < 0) & (bound > lower)
        lower = torch.where(tighter, bound, lower)
        face_in = torch.where(tighter, torch.tensor(face, dtype=torch.int8, device=x.device), face_in)
    inside = feasible & (lower <= upper)
    nan = torch.tensor(float("nan"), dtype=x.dtype, device=x.device)
    zero = torch.tensor(0, dtype=torch.int8, device=x.device)
    return dict(z_in=torch.where(inside, lower, nan), z_out=torch.where(inside, upper, nan),
                face_in=torch.where(inside, face_in, zero), face_out=torch.where(inside, face_out, zero))


def segment_walls(p, q):
    """muon_types.segment_walls: bitmask of the lateral vessel walls crossed by the segments p -> q."""
    d = q - p
    c = vessel_crossings(p[:, 0], p[:, 1], p[:, 2], d[:, 0], d[:, 1], d[:, 2])
    lo, hi = torch.minimum(p[:, 2], q[:, 2]), torch.maximum(p[:, 2], q[:, 2])
    bits = torch.zeros(len(p), dtype=torch.uint8, device=p.device)
    for face, z in ((c["face_in"], c["z_in"]), (c["face_out"], c["z_out"])):
        on_segment = (z >= lo) & (z <= hi)
        for f, bit in _FACE_BIT.items():
            bits |= ((face == f) & on_segment).to(torch.uint8) * bit
    return bits


def bounce_points(a, pa, b, pb, max_gap=0.5, min_angle=0.02):
    """muon_types.bounce_points: closest approach of the line forward from A and backward from B."""
    u, v = pa / _norm(pa)[:, None], pb / _norm(pb)[:, None]
    w = a - b
    uv, uw, vw = (u * v).sum(1), (u * w).sum(1), (v * w).sum(1)
    sin2 = 1 - uv ** 2
    parallel = sin2 < math.sin(min_angle) ** 2
    den = torch.where(parallel, torch.ones_like(sin2), sin2)
    s, t = (uv * vw - uw) / den, (vw - uv * uw) / den
    p, q = a + s[:, None] * u, b + t[:, None] * v
    k = (p + q) / 2
    ok = (~parallel & (s > 0) & (t < 0) & (_norm(p - q) < max_gap) & (k[:, 2] > a[:, 2]) & (k[:, 2] < b[:, 2]))
    return k, ok


def _classify_chunk(d, device):
    f = lambda k: torch.as_tensor(np.ascontiguousarray(d[k]), device=device).to(torch.float64)
    a = torch.stack([f(f"{k}_30") for k in "xyz"], 1)
    pa = torch.stack([f(f"p{k}_30") for k in "xyz"], 1)
    b = torch.stack([f(f"{k}_82") for k in "xyz"], 1)
    pb = torch.stack([f(f"p{k}_82") for k in "xyz"], 1)
    n = len(a)
    reach_a, reach_b = a[:, 2] > mt.REACH_30, b[:, 2] > mt.REACH_82
    p_b = _norm(pb)
    dz = geo.Z_TIMING - b[:, 2]
    x_t = b[:, 0] + pb[:, 0] / pb[:, 2] * dz
    y_t = b[:, 1] + pb[:, 1] / pb[:, 2] * dz
    t = (b[:, 2] - a[:, 2]) / pa[:, 2]
    off = torch.hypot(a[:, 0] + pa[:, 0] * t - b[:, 0], a[:, 1] + pa[:, 1] * t - b[:, 1])
    off = torch.where(reach_b & (pa[:, 2] > 0) & torch.isfinite(off), off, torch.full_like(off, float("inf")))
    acc = reach_b & (b[:, 0].abs() < geo.SST1_HALF_X) & (b[:, 1].abs() < geo.SST1_HALF_Y)
    good = (acc & (p_b > 1) & (pb[:, 2] > 0) & (x_t.abs() < geo.TIMING_HALF_X) & (y_t.abs() < geo.TIMING_HALF_Y))

    k = torch.full((n, 3), float("nan"), dtype=torch.float64, device=device)
    walls = torch.zeros(n, dtype=torch.uint8, device=device)
    m = torch.nonzero(reach_a & (b[:, 2] > a[:, 2])).squeeze(1)
    if len(m):
        km, ok = bounce_points(a[m], pa[m], b[m], pb[m])
        ok &= off[m] > mt.BOUNCE_CUT
        k[m[ok]] = km[ok]
        mid = torch.where(ok[:, None], km, b[m])
        walls[m] = segment_walls(a[m], mid)
        if ok.any():
            walls[m[ok]] |= segment_walls(km[ok], b[m[ok]])
    window = reach_a & (pa[:, 2] > 0) & (a[:, 0].abs() < geo.UBT_HALF_X) & (a[:, 1].abs() < geo.UBT_HALF_Y)
    walls |= window.to(torch.uint8) * mt.WALL_BITS["ubt_window"]

    kind = torch.full((n,), 7, dtype=torch.int8, device=device)
    kind[reach_a & ~reach_b] = 6
    kind[reach_b] = 5
    kind[reach_b & (off > mt.BOUNCE_CUT)] = 4
    kind[reach_a & (walls != 0)] = 3
    kind[acc] = 2
    kind[good] = 1
    kind[acc & (off > mt.BOUNCE_CUT)] = 0
    cpu = lambda x: x.cpu().numpy()
    return dict(type=cpu(kind), walls=cpu(walls), kx=cpu(k[:, 0]), ky=cpu(k[:, 1]), kz=cpu(k[:, 2]), off=cpu(off))


def classify(d, device="cuda", chunk=20_000_000):
    """muon_types.classify on the GPU, in chunks of `chunk` muons."""
    n = len(next(iter(d.values())))
    parts = [_classify_chunk({key: v[i:i + chunk] for key, v in d.items()}, device) for i in range(0, n, chunk)]
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
