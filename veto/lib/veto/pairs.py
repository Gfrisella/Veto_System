"""Building muon pairs and their geometry (DOCA, vertex)."""

import numpy as np

from . import geometry as geo
from .tracks import extrapolate_to_z


def all_pairs(n):
    """Every unique (i, j) with i < j."""
    if n < 2:
        return np.empty((0, 2), dtype=np.int64)
    i, j = np.triu_indices(n, k=1)
    return np.column_stack([i, j])


def time_window_pairs(time, half_window):
    """Pairs whose times differ by at most half_window (vectorised, O(N log N + pairs))."""
    order = np.argsort(time, kind="stable")
    t = time[order]
    right = np.searchsorted(t, t + half_window, side="right")
    n_after = right - np.arange(len(t)) - 1
    n_in_window = n_after + (np.arange(len(t)) - np.searchsorted(t, t - half_window, side="left"))

    first = np.repeat(np.arange(len(t)), n_after)
    offsets = np.arange(n_after.sum()) - np.repeat(np.cumsum(n_after) - n_after, n_after)
    second = first + 1 + offsets
    pairs = np.column_stack([order[first], order[second]])

    in_window = np.empty_like(n_in_window)
    in_window[order] = n_in_window
    return pairs, in_window


def kdtree_candidate_pairs(data, z_min=geo.Z_VESSEL_MIN, z_max=geo.Z_VESSEL_MAX,
                           z_step=0.5, radius=0.10, x_max=2.1, y_max=3.1):
    """Opposite-charge pairs that come within `radius` of each other inside the vessel.

    Tracks are sampled every z_step; at each sample positive and negative tracks
    are matched with a 2D KD-tree. Same candidates as temp1.py (whose 3D tree only
    ever matched equal-z samples because radius < z_step), without the Python loop.
    """
    from scipy.spatial import cKDTree

    positive = np.where(data["pdg"] > 0)[0]
    negative = np.where(data["pdg"] < 0)[0]
    z_samples = np.arange(z_min, z_max + z_step, z_step)
    print(f"Positive tracks: {len(positive):,}  Negative tracks: {len(negative):,}  z samples: {len(z_samples)}")

    def sample(indices, z):
        x, y = extrapolate_to_z(*(data[k][indices].astype(np.float64) for k in ("x", "y", "z", "px", "py", "pz")), z)
        ok = np.isfinite(x) & np.isfinite(y) & (np.abs(x) <= x_max) & (np.abs(y) <= y_max)
        return indices[ok], np.column_stack([x[ok], y[ok]])

    found = []
    for z in z_samples:
        pos_idx, pos_xy = sample(positive, z)
        neg_idx, neg_xy = sample(negative, z)
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue
        matches = cKDTree(pos_xy).sparse_distance_matrix(cKDTree(neg_xy), radius, output_type="ndarray")
        if len(matches):
            found.append(np.column_stack([pos_idx[matches["i"]], neg_idx[matches["j"]]]))

    if not found:
        return np.empty((0, 2), dtype=np.int64)
    pairs = np.sort(np.vstack(found), axis=1)
    pairs = np.unique(pairs, axis=0)
    print(f"Unique candidate pairs: {len(pairs):,}")
    return pairs


def doca_and_vertex(data, pairs):
    """Distance of closest approach between the two straight tracks of each pair.

    Returns doca [m], vertex (midpoint of closest points) and summed momentum.
    Parallel tracks get doca = inf, vertex = momentum = NaN.
    """
    if len(pairs) == 0:
        return np.empty(0), np.empty((0, 3)), np.empty((0, 3))

    r = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)
    p = np.column_stack([data["px"], data["py"], data["pz"]]).astype(np.float64)
    r1, r2 = r[pairs[:, 0]], r[pairs[:, 1]]
    p1, p2 = p[pairs[:, 0]], p[pairs[:, 1]]
    dr = r1 - r2

    cross = np.cross(p1, p2)
    cross_norm = np.linalg.norm(cross, axis=1)
    valid = cross_norm > 1e-12

    doca = np.full(len(pairs), np.inf)
    doca[valid] = np.abs(np.sum(dr[valid] * cross[valid], axis=1)) / cross_norm[valid]

    a = np.sum(p1 * p1, axis=1)
    b = np.sum(p1 * p2, axis=1)
    c = np.sum(p2 * p2, axis=1)
    d = np.sum(p1 * dr, axis=1)
    e = np.sum(p2 * dr, axis=1)
    den = a * c - b * b
    valid &= np.abs(den) > 1e-12

    s = np.zeros(len(pairs))
    t = np.zeros(len(pairs))
    s[valid] = (b[valid] * e[valid] - c[valid] * d[valid]) / den[valid]
    t[valid] = (a[valid] * e[valid] - b[valid] * d[valid]) / den[valid]

    vertex = 0.5 * ((r1 + s[:, None] * p1) + (r2 + t[:, None] * p2))
    momentum = p1 + p2

    doca[~valid] = np.inf
    vertex[~valid] = np.nan
    momentum[~valid] = np.nan
    return doca, vertex, momentum
