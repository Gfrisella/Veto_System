"""Muon types from two stored states: A at z = 30.6 m and B at z = 82 m (or where the muon stopped).

Used by the design-optimisation dataset and the full-dataset visualisation, so both agree.

A muon's path is the chord A -> B, or A -> K -> B when it bounced (K = estimated bounce
point, see bounce_points). Wall crossings of that path are what a full SBT would see.
"""

import numpy as np

from . import geometry as geo
from .tracks import vessel_crossings

REACH_30 = geo.Z_UBT - 0.01     # A really is at z = 30.6 (stored z is 30.600 .. 30.620)
REACH_82 = geo.Z_SST1 - 0.01    # B really is at z = 82
BOUNCE_CUT = 0.5                # B more than this [m] off the straight line from A = bounced / scattered

TYPES = {
    0: "sst1_bounced",       # in SST1 acceptance, > 50 cm off its straight line (bounced off the cavern)
    1: "sst1_selected",      # in SST1 acceptance, p > 1 GeV, inside the timing-detector acceptance
    2: "sst1_other",         # in SST1 acceptance, rest
    3: "veto_only",          # crosses an SBT wall or the UBT window, misses the SST1 acceptance
    4: "rock_bounced",       # reaches z = 82 outside the acceptance, > 50 cm off its straight line
    5: "straight_past",      # reaches z = 82 outside the acceptance, close to its straight line
    6: "stops_between",      # crosses z = 30.6, stops before z = 82
    7: "stops_before",       # stops before z = 30.6
}

# bits of the `walls` column
WALL_BITS = {"+x": 1, "-x": 2, "+y": 4, "-y": 8, "ubt_window": 16}
_FACE_BIT = {2: 1, 3: 2, 4: 4, 5: 8}


def _unit(p):
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def bounce_points(a, pa, b, pb, max_gap=0.5, min_angle=0.02):
    """Single-bounce estimate: closest approach of the line forward from A and the line backward from B.

    a, b: positions (n, 3); pa, pb: momenta (n, 3). Returns the midpoint K (n, 3) and a mask of
    credible bounces: K after A along A's direction, before B along B's direction, z between the
    two, lines within max_gap metres and at least min_angle rad apart (nearly parallel lines have
    no defined meeting point: those muons only scattered a little).
    """
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    u, v = _unit(np.asarray(pa, np.float64)), _unit(np.asarray(pb, np.float64))
    w = a - b
    uv, uw, vw = (u * v).sum(1), (u * w).sum(1), (v * w).sum(1)
    sin2 = 1 - uv ** 2
    parallel = sin2 < np.sin(min_angle) ** 2
    den = np.where(parallel, 1, sin2)
    s, t = (uv * vw - uw) / den, (vw - uv * uw) / den
    p, q = a + s[:, None] * u, b + t[:, None] * v
    k = (p + q) / 2
    ok = (~parallel & (s > 0) & (t < 0) & (np.linalg.norm(p - q, axis=1) < max_gap)
          & (k[:, 2] > a[:, 2]) & (k[:, 2] < b[:, 2]))
    return k, ok


def segment_walls(p, q):
    """Bitmask of the lateral vessel walls (SBT) crossed by the segments p -> q (arrays (n, 3))."""
    d = q - p
    c = vessel_crossings(p[:, 0], p[:, 1], p[:, 2], d[:, 0], d[:, 1], d[:, 2])
    lo, hi = np.minimum(p[:, 2], q[:, 2]), np.maximum(p[:, 2], q[:, 2])
    bits = np.zeros(len(p), dtype=np.uint8)
    for face, z in ((c["face_in"], c["z_in"]), (c["face_out"], c["z_out"])):
        on_segment = (z >= lo) & (z <= hi)
        for f, bit in _FACE_BIT.items():
            bits |= np.where((face == f) & on_segment, bit, 0).astype(np.uint8)
    return bits


def classify(d):
    """Type, wall bitmask, bounce point and straight-line miss distance per muon.

    d: arrays x_30, y_30, z_30, px_30, ... (state A) and x_82, ... (state B).
    Returns dict(type, walls, kx, ky, kz (NaN without a credible bounce), off).
    """
    def state(s):
        return (np.stack([d[f"{k}_{s}"] for k in "xyz"], 1).astype(np.float64),
                np.stack([d[f"p{k}_{s}"] for k in "xyz"], 1).astype(np.float64))

    a, pa = state(30)
    b, pb = state(82)
    n = len(a)
    reach_a, reach_b = a[:, 2] > REACH_30, b[:, 2] > REACH_82

    p_b = np.linalg.norm(pb, axis=1)
    dz = geo.Z_TIMING - b[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x_t = b[:, 0] + pb[:, 0] / pb[:, 2] * dz
        y_t = b[:, 1] + pb[:, 1] / pb[:, 2] * dz
        t = (b[:, 2] - a[:, 2]) / pa[:, 2]
        off = np.hypot(a[:, 0] + pa[:, 0] * t - b[:, 0], a[:, 1] + pa[:, 1] * t - b[:, 1])
    off = np.where(reach_b & (pa[:, 2] > 0) & np.isfinite(off), off, np.inf)
    acc = reach_b & (np.abs(b[:, 0]) < geo.SST1_HALF_X) & (np.abs(b[:, 1]) < geo.SST1_HALF_Y)
    good = (acc & (p_b > 1) & (pb[:, 2] > 0)
            & (np.abs(x_t) < geo.TIMING_HALF_X) & (np.abs(y_t) < geo.TIMING_HALF_Y))

    # path A -> B or A -> K -> B, only for muons that got to z = 30.6 and moved on
    k = np.full((n, 3), np.nan)
    walls = np.zeros(n, dtype=np.uint8)
    m = np.where(reach_a & (b[:, 2] > a[:, 2]))[0]
    if len(m):
        km, ok = bounce_points(a[m], pa[m], b[m], pb[m])
        ok &= off[m] > BOUNCE_CUT
        k[m[ok]] = km[ok]
        mid = np.where(ok[:, None], km, b[m])
        walls[m] = segment_walls(a[m], mid)
        if ok.any():
            walls[m[ok]] |= segment_walls(km[ok], b[m[ok]])
    window = reach_a & (pa[:, 2] > 0) & (np.abs(a[:, 0]) < geo.UBT_HALF_X) & (np.abs(a[:, 1]) < geo.UBT_HALF_Y)
    walls |= np.where(window, WALL_BITS["ubt_window"], 0).astype(np.uint8)

    kind = np.full(n, 7, dtype=np.int8)
    kind[reach_a & ~reach_b] = 6
    kind[reach_b] = 5
    kind[reach_b & (off > BOUNCE_CUT)] = 4
    kind[reach_a & (walls != 0)] = 3
    kind[acc] = 2
    kind[good] = 1
    kind[acc & (off > BOUNCE_CUT)] = 0
    return dict(type=kind, walls=walls, kx=k[:, 0], ky=k[:, 1], kz=k[:, 2], off=off)
