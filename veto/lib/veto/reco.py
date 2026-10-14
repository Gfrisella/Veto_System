"""Digitization, spectrometer abstraction and veto reconstruction (algorithm A) for a design.

Events are short time windows around a spectrometer candidate, not whole spills:
  - own hits: the candidate's muons cross UBT / fins along their true path (A -> K -> B),
    each crossing is detected with the design efficiency, digitized to a cell and smeared in time;
  - accidental hits: every other muon of the spill, drawn per plane from the full-statistics rate
    maps with Poisson mean rate * window / spill length.

Algorithm A (track driven): every spectrometer track is extrapolated backward in a straight line
(no field in the vessel) to each detector plane upstream of its vertex. A hit within the spatial
window (grows with distance and 1/p for multiple scattering) and the time window (time of flight
corrected) explains the track; a candidate with an explained track is vetoed. No combinatorics:
one binary search per (event, plane) and a short scan of that plane's hits.

Times are relative to the candidate's arrival at z = 82 m; muons move at c.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from . import geometry as geo
from . import muon_types as mt

C = 0.299792458   # m / ns


@dataclass
class RecoConfig:
    spill_s: float = 1.0               # spill length [s]: occupancy scales with 1 / spill_s
    sigma_t_track: float = 0.1         # spectrometer / timing detector time resolution [ns]
    sigma_pos: float = 1e-3            # spectrometer position resolution at z = 82 m [m]
    sigma_slope: float = 1e-4          # spectrometer slope resolution, high momentum [rad]
    ms_slope: float = 2e-3             # extra slope smearing * 1 GeV / p (multiple scattering) [rad]
    n_sigma_t: float = 3.0             # time window: +- n_sigma_t * combined resolution
    r0: float = 0.05                   # spatial window: r0 + (r_slope + r_ms / p) * distance [m]
    r_slope: float = 5e-4
    r_ms: float = 6e-3
    seed: int = 1
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- truth -> hits

def true_crossing(tr, plane):
    """Where each track's true path crosses a plane: local (u, y), path length to B [m], mask."""
    a = np.stack([tr["ax"], tr["ay"], tr["az"]], 1)
    k = np.stack([tr["kx"], tr["ky"], tr["kz"]], 1)
    b = np.stack([tr["bx"], tr["by"], tr["bz"]], 1)
    if plane.kind == "ubt":
        x, y = a[:, 0], a[:, 1]
        ok = tr["apz"] > 0
        p = a
    else:
        x, y, ok = mt.path_at_z(a, k, b, plane.z)
        p = np.stack([x, y, np.full(len(x), plane.z)], 1)
    u = plane.to_local(x)
    ok = ok & plane.inside(u, y)
    bounced = np.isfinite(k[:, 2])
    before_k = bounced & (p[:, 2] < k[:, 2])
    length = np.where(before_k, np.linalg.norm(k - p, axis=1) + np.linalg.norm(b - k, axis=1),
                      np.linalg.norm(b - p, axis=1))
    return u, y, length, ok


def own_hits(tr, event, design, rng, tracks_mask=None):
    """Hits of the events' own muons. Returns dict of arrays event, plane, u, y, t."""
    out = {k: [] for k in ("event", "plane", "u", "y", "t")}
    sel = np.ones(len(event), bool) if tracks_mask is None else tracks_mask
    for plane in design.planes:
        u, y, length, ok = true_crossing(tr, plane)
        ok &= sel & (rng.random(len(ok)) < design.efficiency)
        uc, yc = plane.cell_centre(u[ok], y[ok])
        t = -length[ok] / C + rng.normal(0, design.sigma_t * 1e9, ok.sum())
        out["event"].append(event[ok]); out["plane"].append(np.full(ok.sum(), plane.index))
        out["u"].append(uc); out["y"].append(yc); out["t"].append(t)
    return {k: np.concatenate(v) for k, v in out.items()}


# ---------------------------------------------------------------- accidental hits from the rate maps

def plane_rate_map(plane, maps):
    """Rate map of a plane: bin edges in u, y and muons per spill per bin, clipped to the plane."""
    if plane.kind == "ubt":
        yb = maps["ubt_y_bins"]
        ub = np.array([0.0, plane.u_max])
        h = maps["ubt_hist"][None, :]
    else:
        side = 0 if plane.sign > 0 else 1
        iz = int(np.clip(np.round(plane.z - maps["z_slices"][0]), 0, len(maps["z_slices"]) - 1))
        h = maps["fin_maps"][side, iz]
        ub, yb = maps["u_bins"], maps["y_bins"]
    uc, yc = 0.5 * (ub[1:] + ub[:-1]), 0.5 * (yb[1:] + yb[:-1])
    keep = (uc <= plane.u_max)[:, None] & ((yc >= plane.y_lo) & (yc <= plane.y_hi))[None, :]
    return ub, yb, np.where(keep, h, 0.0)


def plane_rates(design, maps):
    """Per plane: total muons per spill, mean and peak muons per spill per cell."""
    out = []
    for plane in design.planes:
        ub, yb, h = plane_rate_map(plane, maps)
        density = h / np.outer(np.diff(ub), np.diff(yb))
        cell = plane.cell_u * plane.cell_y
        out.append(dict(plane=plane.name, area=plane.area, n_cells=plane.n_cells, rate=h.sum(),
                        mean_cell=h.sum() / max(plane.n_cells, 1), peak_cell=density.max() * cell))
    return out


def accidental_hits(design, maps, n_events, window_ns, cfg, rng):
    """Random hits of all other muons in +- window_ns around each plane's nominal time."""
    out = {k: [] for k in ("event", "plane", "u", "y", "t")}
    for plane in design.planes:
        ub, yb, h = plane_rate_map(plane, maps)
        rate = h.sum()
        if rate <= 0:
            continue
        mean = rate * (2 * window_ns * 1e-9) / cfg.spill_s
        n = rng.poisson(mean, n_events)
        total = int(n.sum())
        prob = (h / rate).ravel()
        b = rng.choice(len(prob), total, p=prob)
        iu, iy = np.unravel_index(b, h.shape)
        u = np.clip(ub[iu] + rng.random(total) * np.diff(ub)[iu], 0, plane.u_max)
        y = np.clip(yb[iy] + rng.random(total) * np.diff(yb)[iy], plane.y_lo, plane.y_hi)
        u, y = plane.cell_centre(u, y)
        nominal = -(geo.Z_SST1 - plane.z) / C
        out["event"].append(np.repeat(np.arange(n_events), n)); out["plane"].append(np.full(total, plane.index))
        out["u"].append(u); out["y"].append(y)
        out["t"].append(nominal + rng.uniform(-window_ns, window_ns, total))
    return {k: np.concatenate(v) if v else np.empty(0) for k, v in out.items()}


# ---------------------------------------------------------------- spectrometer

def measure(tr, cfg, rng):
    """Spectrometer tracks at z = 82 m: smeared position, slopes and time (true time 0)."""
    n = len(tr["bx"])
    p = np.sqrt(tr["bpx"] ** 2 + tr["bpy"] ** 2 + tr["bpz"] ** 2)
    sig_slope = np.hypot(cfg.sigma_slope, cfg.ms_slope / np.maximum(p, 0.1))
    return dict(x=tr["bx"] + rng.normal(0, cfg.sigma_pos, n), y=tr["by"] + rng.normal(0, cfg.sigma_pos, n),
                z=tr["bz"], tx=tr["bpx"] / tr["bpz"] + rng.normal(0, 1, n) * sig_slope,
                ty=tr["bpy"] / tr["bpz"] + rng.normal(0, 1, n) * sig_slope, p=p,
                t=rng.normal(0, cfg.sigma_t_track, n))


# ---------------------------------------------------------------- algorithm A

def predictions(meas, event, z_limit, design, cfg):
    """Backward straight-line prediction on every plane upstream of z_limit: one query per
    (track, plane) that lands within reach of the plane. Returns dict of query arrays."""
    q = {k: [] for k in ("track", "event", "plane", "u", "y", "t", "r")}
    track = np.arange(len(event))
    for plane in design.planes:
        dz = meas["z"] - plane.z
        x = meas["x"] - meas["tx"] * dz
        y = meas["y"] - meas["ty"] * dz
        dist = dz * np.sqrt(1 + meas["tx"] ** 2 + meas["ty"] ** 2)
        r = cfg.r0 + (cfg.r_slope + cfg.r_ms / np.maximum(meas["p"], 0.1)) * dist
        u = plane.to_local(x)
        reach = ((plane.z < z_limit) & (u >= -r - plane.cell_u) & (u <= plane.u_max + r + plane.cell_u)
                 & (y >= plane.y_lo - r - plane.cell_y) & (y <= plane.y_hi + r + plane.cell_y))
        for k, v in (("track", track), ("event", event), ("u", u), ("y", y), ("t", meas["t"] - dist / C),
                     ("r", r)):
            q[k].append(v[reach])
        q["plane"].append(np.full(reach.sum(), plane.index))
    return {k: np.concatenate(v) for k, v in q.items()}


def match(queries, hits, design, cfg):
    """For every query: is there a hit on its plane within the space and time window?"""
    n_planes = len(design.planes)
    cell_u = np.array([p.cell_u for p in design.planes])
    cell_y = np.array([p.cell_y for p in design.planes])
    window_t = cfg.n_sigma_t * np.hypot(design.sigma_t * 1e9, cfg.sigma_t_track)
    key_h = hits["event"].astype(np.int64) * n_planes + hits["plane"]
    order = np.argsort(key_h, kind="stable")
    key_h = key_h[order]
    hu, hy, ht, hp = hits["u"][order], hits["y"][order], hits["t"][order], hits["plane"][order]
    key_q = queries["event"].astype(np.int64) * n_planes + queries["plane"]
    lo = np.searchsorted(key_h, key_q, "left")
    hi = np.searchsorted(key_h, key_q, "right")
    counts = hi - lo
    qi = np.repeat(np.arange(len(key_q)), counts)
    hi_idx = np.repeat(lo - np.cumsum(counts) + counts, counts) + np.arange(counts.sum())
    du = np.abs(hu[hi_idx] - queries["u"][qi]) <= queries["r"][qi] + cell_u[hp[hi_idx]] / 2
    dy = np.abs(hy[hi_idx] - queries["y"][qi]) <= queries["r"][qi] + cell_y[hp[hi_idx]] / 2
    dt = np.abs(ht[hi_idx] - queries["t"][qi]) <= window_t
    found = np.zeros(len(key_q), bool)
    found[qi[du & dy & dt]] = True
    return found, int(counts.sum())


def run_events(tr, event, z_limit, design, maps, cfg, rng, with_own_hits=True):
    """Full chain for a set of events. tr: true tracks (one row per track), event: event id per
    track, z_limit: search only planes upstream of this z (the vertex; 82 m for single tracks).
    Returns per-event veto flag, per-track explained flag, per-plane veto counts and timings."""
    timing = {}
    n_events = int(event.max()) + 1 if len(event) else 0
    t0 = time.perf_counter()
    own = own_hits(tr, event, design, rng) if with_own_hits else {k: np.empty(0) for k in
                                                                   ("event", "plane", "u", "y", "t")}
    window = cfg.n_sigma_t * np.hypot(design.sigma_t * 1e9, cfg.sigma_t_track) + 0.5
    acc = accidental_hits(design, maps, n_events, window, cfg, rng)
    hits = {k: np.concatenate([own[k], acc[k]]) for k in own}
    hits["event"] = hits["event"].astype(np.int64); hits["plane"] = hits["plane"].astype(np.int64)
    timing["digitize"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    meas = measure(tr, cfg, rng)
    q = predictions(meas, event, z_limit, design, cfg)
    found, n_compared = match(q, hits, design, cfg)
    explained = np.zeros(len(event), bool)
    explained[q["track"][found]] = True
    vetoed = np.zeros(n_events, bool)
    vetoed[event[explained]] = True
    timing["reconstruct"] = time.perf_counter() - t0
    by_plane = np.bincount(q["plane"][found], minlength=len(design.planes))
    return dict(vetoed=vetoed, explained=explained, by_plane=by_plane, timing=timing,
                n_hits=len(hits["event"]), n_own=len(own["event"]), n_queries=len(q["event"]),
                n_compared=n_compared)


# ---------------------------------------------------------------- hypothetical signal

def signal_tracks(n, rng, p_range=(1.0, 50.0)):
    """Hypothetical neutral particles decaying uniformly in the vessel volume into two charged
    tracks that reach the SST1 acceptance (momentum uniform in p_range). No veto hits of their own.
    Returns (track dict with the B state at z = 82, vertex z per track, event id per track)."""
    z_v = rng.uniform(geo.Z_VESSEL_MIN, geo.Z_VESSEL_MAX, 4 * n)
    hx, hy = geo.vessel_half_widths(z_v)
    x_v, y_v = rng.uniform(-1, 1, 4 * n) * hx, rng.uniform(-1, 1, 4 * n) * hy
    keep = rng.uniform(0, 1, 4 * n) < (hx * hy) / (geo.VESSEL_X_OUT * geo.VESSEL_Y_OUT)
    z_v, x_v, y_v = z_v[keep][:n], x_v[keep][:n], y_v[keep][:n]
    n = len(z_v)
    xt = rng.uniform(-geo.SST1_HALF_X, geo.SST1_HALF_X, 2 * n)
    yt = rng.uniform(-geo.SST1_HALF_Y, geo.SST1_HALF_Y, 2 * n)
    xv, yv, zv = np.tile(x_v, 2), np.tile(y_v, 2), np.tile(z_v, 2)
    d = np.stack([xt - xv, yt - yv, geo.Z_SST1 - zv], 1)
    p = rng.uniform(*p_range, 2 * n)
    mom = d / np.linalg.norm(d, axis=1, keepdims=True) * p[:, None]
    tr = dict(bx=xt, by=yt, bz=np.full(2 * n, geo.Z_SST1), bpx=mom[:, 0], bpy=mom[:, 1], bpz=mom[:, 2])
    return tr, zv, np.tile(np.arange(n), 2)
