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
A_TOLERANCE = 0.05            # [m] the z = 30.6 state is stored up to 2 cm after the plane
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


def side_plane_x(z):
    """Half width of the side-wall planes: the vessel's lateral walls, continued past z = 80.6 m."""
    slope = (geo.VESSEL_X_OUT - geo.VESSEL_X_IN) / (geo.Z_VESSEL_MAX - geo.Z_VESSEL_MIN)
    return geo.VESSEL_X_IN + slope * (z - geo.Z_VESSEL_MIN)


def path_at_z(a, k, b, z):
    """Position of the path a -> (k) -> b at plane z (scalar or per-muon array).

    a, k, b: (n, 3), k NaN without a bounce; the path is monotonic in z (k lies between a and b).
    Returns x, y and a mask of muons whose path reaches z.
    """
    a, k, b = (np.asarray(v, np.float64) for v in (a, k, b))
    z = np.broadcast_to(np.asarray(z, np.float64), (len(a),))
    # A is stored a few mm after z = 30.6 (30.600 .. 30.620): a plane just before A uses A itself
    z = np.where((z < a[:, 2]) & (a[:, 2] - z < A_TOLERANCE), a[:, 2], z)
    bounced = np.isfinite(k[:, 2])
    end = np.where(bounced[:, None], k, b)
    x, y = np.full(len(a), np.nan), np.full(len(a), np.nan)
    for p, q, use in ((a, end, np.ones(len(a), bool)), (k, b, bounced)):
        dz = q[:, 2] - p[:, 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            t = (z - p[:, 2]) / dz
        m = use & (dz > 0) & (t >= 0) & (t <= 1) & np.isnan(x)
        x[m] = p[m, 0] + t[m] * (q[m, 0] - p[m, 0])
        y[m] = p[m, 1] + t[m] * (q[m, 1] - p[m, 1])
    return x, y, np.isfinite(x)


CAVERN_X_WALL = geo.CAVERN_X_WALL   # fins on the -x side (right, looking from the target) stop here
FIN_Y_MIN = geo.CAVERN_Y_FLOOR       # every detector (UBT, fins) may reach down to the cavern floor, y = -3.36 m


def fin_geometry(z0, sign, theta, length):
    """Inner and outer edge of a fin: it pivots about its inner edge on the side wall at (sign * X(z0), z0)
    and leans downstream by theta (0 = perpendicular to the beam). Returns (x_in, z_in, x_out, z_out)."""
    x_in = sign * side_plane_x(z0)
    return x_in, z0, x_in + sign * length * np.cos(theta), z0 + length * np.sin(theta)


def fin_hits(a, k, b, z0, sign, theta, length, y_lo, y_hi):
    """Which paths a -> (k) -> b cross a fin (see fin_geometry), and where.

    The fin plane: (z - z0) cos(theta) = (sign x - X(z0)) sin(theta). A muon hits it if its path crosses
    the plane at distance 0 .. length from the inner edge and bottom .. top in y. The first segment is
    extended A_TOLERANCE back because A is stored a few mm after z = 30.6.
    Returns hit mask and crossing points (n, 3), NaN where no hit.
    """
    a, k, b = (np.asarray(v, np.float64) for v in (a, k, b))
    c, s = np.cos(theta), np.sin(theta)
    x0 = side_plane_x(z0)
    bounced = np.isfinite(k[:, 2])
    end = np.where(bounced[:, None], k, b)
    g = lambda p: (p[:, 2] - z0) * c - (sign * p[:, 0] - x0) * s
    point = np.full((len(a), 3), np.nan)
    for p, q, use, first in ((a, end, np.ones(len(a), bool), True), (k, b, bounced, False)):
        gp, gq = g(p), g(q)
        dz = q[:, 2] - p[:, 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            t = gp / (gp - gq)
            t_min = -A_TOLERANCE / dz if first else 0.0
            cross = p + t[:, None] * (q - p)
        along = (sign * cross[:, 0] - x0) * c + (cross[:, 2] - z0) * s
        hit = (use & (dz > 0) & (t >= t_min) & (t <= 1) & (along >= 0) & (along <= length)
               & (cross[:, 1] >= y_lo) & (cross[:, 1] <= y_hi) & np.isnan(point[:, 0]))
        point[hit] = cross[hit]
    return np.isfinite(point[:, 0]), point


def bottom_crossings(a, k, b, z0, y0, pitch, x_lo, x_hi, length):
    """Where paths a -> (k) -> b cross a flat panel under the vessel: upstream edge at (z0, y0), rising
    downstream by pitch, across x_lo .. x_hi, `length` along the panel.

    The panel plane: (y - y0) cos(pitch) = (z - z0) sin(pitch). A path can cross it twice (down, and
    up again after a floor bounce). Returns two (hit mask, points (n, 3), distance v along the panel):
    the first segment (A -> K, or A -> B without a bounce) and the second (K -> B).
    """
    a, k, b = (np.asarray(v, np.float64) for v in (a, k, b))
    c, s = np.cos(pitch), np.sin(pitch)
    bounced = np.isfinite(k[:, 2])
    end = np.where(bounced[:, None], k, b)
    g = lambda p: (p[:, 1] - y0) * c - (p[:, 2] - z0) * s
    out = []
    for p, q, use in ((a, end, np.ones(len(a), bool)), (k, b, bounced)):
        gp, gq = g(p), g(q)
        with np.errstate(invalid="ignore", divide="ignore"):
            t = gp / (gp - gq)
            cross = p + t[:, None] * (q - p)
        v = (cross[:, 2] - z0) * c + (cross[:, 1] - y0) * s
        hit = (use & (q[:, 2] > p[:, 2]) & (t >= 0) & (t <= 1) & (v >= 0) & (v <= length)
               & (cross[:, 0] >= x_lo) & (cross[:, 0] <= x_hi))
        out.append((hit, np.where(hit[:, None], cross, np.nan), np.where(hit, v, np.nan)))
    return out
