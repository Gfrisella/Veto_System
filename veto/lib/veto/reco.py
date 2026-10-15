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
from . import scattering
from . import technology as tech

C = 0.299792458   # m / ns


@dataclass
class RecoConfig:
    spill_s: float = 1.0               # spill length [s]: occupancy scales with 1 / spill_s
    sigma_t_track: float = 0.1         # spectrometer / timing detector time resolution [ns]
    sigma_pos: float = 1e-3            # spectrometer position resolution at z = 82 m [m]
    sigma_slope: float = 1e-4          # spectrometer slope resolution, high momentum [rad]
    ms_slope: float = 2e-3             # extra slope smearing * 1 GeV / p (multiple scattering) [rad]
    n_sigma_t: float = 3.0             # time window: +- n_sigma_t * combined resolution
    time_window_ns: float = None       # if set: fixed full width of the time window on every plane [ns]
    r0: float = 0.05                   # spatial window: r0 + (r_slope + r_ms / p) * distance [m]
    r_slope: float = 5e-4
    r_ms: float = 6e-3
    min_planes: int = 1                # a track is explained by hits on at least this many planes
    multiple_scattering: bool = False  # scatter the muons after z = 30.6 m (scattering.py)
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
    """Hits of the events' own muons. Returns dict of arrays event, track, plane, u, y, t."""
    out = {k: [] for k in ("event", "track", "plane", "u", "y", "t")}
    sel = np.ones(len(event), bool) if tracks_mask is None else tracks_mask
    kicks = tr.get("_kicks")
    for plane in design.planes:
        u, y, length, ok = true_crossing(tr, plane)
        if kicks is not None:                   # multiple scattering upstream of this plane
            dx, dy = scattering.offset_at(kicks, plane.z)
            crossed = np.isfinite(u) & np.isfinite(y)
            if plane.kind == "ubt":
                crossed &= tr["apz"] > 0
            u, y = u + (dx if plane.kind == "ubt" else plane.sign * dx), y + dy
            ok = crossed & plane.inside(u, y)
        ok &= sel & (rng.random(len(ok)) < plane.efficiency)
        uc, yc = plane.cell_centre(u[ok], y[ok])
        t = -length[ok] / C + rng.normal(0, plane.sigma_t * 1e9, ok.sum())
        if plane.time_window:                   # drift-limited (straws): anywhere in the window
            t += rng.uniform(-0.5, 0.5, ok.sum()) * plane.time_window * 1e9
        out["event"].append(event[ok]); out["track"].append(np.where(ok)[0])
        out["plane"].append(np.full(ok.sum(), plane.index))
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


def plane_rates(design, maps, spill_s=1.0):
    """Per plane: total muons per spill, mean and peak muons per spill per cell and per readout channel,
    peak rate density [Hz / cm^2] and the area above the technology's rate limit [m^2]."""
    out = []
    for plane in design.planes:
        ub, yb, h = plane_rate_map(plane, maps)
        bin_area = np.outer(np.diff(ub), np.diff(yb))
        density = h / bin_area
        cell = plane.cell_u * plane.cell_y
        hz_cm2 = density / spill_s / 1e4
        t = tech.TECHNOLOGIES[plane.technology] if plane.technology else None
        limit = t["max_rate"] if t else None
        r = dict(plane=plane.name, area=plane.area, n_cells=plane.n_cells, rate=h.sum(),
                 mean_cell=h.sum() / max(plane.n_cells, 1), peak_cell=density.max() * cell,
                 technology=plane.technology, n_channels=plane.n_channels, cost=plane.cost,
                 rel_cost=plane.rel_cost, x0=plane.x0,
                 peak_hz_cm2=float(hz_cm2.max()), max_rate_hz_cm2=limit,
                 area_over_limit=float(bin_area[hz_cm2 > limit].sum()) if limit else 0.0)
        if t is None:
            r["peak_channel"], r["mean_channel"] = r["peak_cell"], r["mean_cell"]
        elif t["readout"] == "pad":
            r["peak_channel"] = density.max() * t["pitch"] ** 2
            r["mean_channel"] = tech.n_layers(t) * h.sum() / max(plane.n_channels, 1)
        else:
            # one strip: the map summed over the segment it covers, times the pitch across it
            uc, yc = 0.5 * (ub[1:] + ub[:-1]), 0.5 * (yb[1:] + yb[:-1])
            height = plane.y_hi - plane.y_lo
            ny, nu = tech.n_segments(t, height), tech.n_segments(t, plane.u_max)
            seg_y = np.minimum((np.clip(yc - plane.y_lo, 0, None) / (height / ny)).astype(int), ny - 1)
            seg_u = np.minimum((np.clip(uc, 0, None) / (plane.u_max / nu)).astype(int), nu - 1)
            per_u = np.stack([h[:, seg_y == k].sum(1) for k in np.unique(seg_y)]) / np.diff(ub)
            per_y = np.stack([h[seg_u == k, :].sum(0) for k in np.unique(seg_u)]) / np.diff(yb)
            r["peak_channel"] = t["pitch"] * max((per_u if tech.vertical(a) else per_y).max() for a in t["views"])
            r["mean_channel"] = tech.n_layers(t) * h.sum() / max(plane.n_channels, 1)   # a muon fires every layer
        ch_limit = t["max_channel_rate"] if t else None
        r["channel_over_limit"] = bool(ch_limit and r["peak_channel"] / spill_s > ch_limit)
        out.append(r)
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


def time_window(design, cfg):
    """Half width of the time window of each plane [ns]."""
    if cfg.time_window_ns:
        return np.full(len(design.planes), cfg.time_window_ns / 2)
    sig = cfg.n_sigma_t * np.hypot(np.array([p.sigma_t for p in design.planes]) * 1e9, cfg.sigma_t_track)
    drift = np.array([(p.time_window or 0.0) * 1e9 / 2 for p in design.planes])
    # technologies with a fixed veto window (tiles: +- 1.5 ns) use it instead of +- n_sigma_t
    fixed = np.array([(tech.TECHNOLOGIES[p.technology].get("veto_window") or np.nan) * 1e9 if p.technology
                      else np.nan for p in design.planes])
    return np.where(np.isnan(fixed), drift + sig, fixed)


def match(queries, hits, design, cfg):
    """For every query: is there a hit on its plane within the space and time window?"""
    n_planes = len(design.planes)
    cell_u = np.array([p.cell_u for p in design.planes])
    cell_y = np.array([p.cell_y for p in design.planes])
    window_t = time_window(design, cfg)                       # per plane [ns]
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
    dt = np.abs(ht[hi_idx] - queries["t"][qi]) <= window_t[hp[hi_idx]]
    ok = du & dy & dt
    found = np.zeros(len(key_q), bool)
    found[qi[ok]] = True
    hit_of_query = np.full(len(key_q), -1)
    hit_of_query[qi[ok][::-1]] = order[hi_idx[ok][::-1]]     # first matching hit (index into `hits`)
    return found, int(counts.sum()), hit_of_query


def run_events(tr, event, z_limit, design, maps, cfg, rng, with_own_hits=True, details=False):
    """Full chain for a set of events. tr: true tracks (one row per track), event: event id per
    track, z_limit: search only planes upstream of this z (the vertex; 82 m for single tracks).
    Returns per-event veto flag, per-track explained flag, per-plane veto counts and timings."""
    timing = {}
    n_events = int(event.max()) + 1 if len(event) else 0
    t0 = time.perf_counter()
    if cfg.multiple_scattering:
        kicks = scattering.kicks(tr, design, rng)
        tr_true, tr = dict(tr, _kicks=kicks), scattering.scattered_tracks(tr, kicks)   # tr: as measured
        timing["scatter"] = time.perf_counter() - t0
        t0 = time.perf_counter()
    else:
        tr_true = tr
    own = own_hits(tr_true, event, design, rng) if with_own_hits else {k: np.empty(0) for k in
                                                                   ("event", "track", "plane", "u", "y", "t")}
    window = time_window(design, cfg).max() + 0.5
    acc = accidental_hits(design, maps, n_events, window, cfg, rng)
    acc["track"] = np.full(len(acc["event"]), -1)            # accidental: no track of this event
    hits = {k: np.concatenate([own[k], acc[k]]) for k in own}
    hits["track"] = hits["track"].astype(np.int64)
    hits["event"] = hits["event"].astype(np.int64); hits["plane"] = hits["plane"].astype(np.int64)
    timing["digitize"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    meas = measure(tr, cfg, rng)
    q = predictions(meas, event, z_limit, design, cfg)
    found, n_compared, hit_of_query = match(q, hits, design, cfg)
    explained = np.bincount(q["track"][found], minlength=len(event)) >= cfg.min_planes
    vetoed = np.zeros(n_events, bool)
    vetoed[event[explained]] = True
    timing["reconstruct"] = time.perf_counter() - t0
    by_plane = np.bincount(q["plane"][found], minlength=len(design.planes))
    out = dict(vetoed=vetoed, explained=explained, by_plane=by_plane, timing=timing,
               n_hits=len(hits["event"]), n_own=len(own["event"]), n_queries=len(q["event"]),
               n_compared=n_compared)
    if details:   # for visualisation: every hit (own ones first), query, match and measured track
        out.update(hits=hits, queries=q, found=found, hit_of_query=hit_of_query, meas=meas)
    return out


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
    tr = dict(bx=xt, by=yt, bz=np.full(2 * n, geo.Z_SST1), bpx=mom[:, 0], bpy=mom[:, 1], bpz=mom[:, 2],
              vx=xv, vy=yv, vz=zv)
    return tr, zv, np.tile(np.arange(n), 2)


def _boost(p_rest, e_rest, beta):
    """Lorentz boost of momenta p_rest (n, 3) with energies e_rest (n,) by velocity beta (n, 3)."""
    b2 = (beta ** 2).sum(1)
    gamma = 1 / np.sqrt(1 - b2)
    bp = (beta * p_rest).sum(1)
    factor = np.where(b2 > 0, (gamma - 1) * bp / np.where(b2 > 0, b2, 1), 0)
    return p_rest + (factor + gamma * e_rest)[:, None] * beta


def signal_from_target(n, rng, mass=1.0, m_daughter=0.106, p_range=(10.0, 100.0), pt_mean=1.0, max_rounds=50):
    """Hypothetical long-lived neutral particles from the target, decaying in the vessel into two
    charged tracks, with the same selection as the background pairs.

    The parent starts at the target (x = y = z = 0) with log-uniform momentum in p_range and a
    transverse momentum drawn from an exponential of mean pt_mean, so it flies nearly along the
    beam (angle ~ pt / p). It decays uniformly in z inside the vessel, isotropically in its rest
    frame into two particles of mass m_daughter. Both must reach the SST1 acceptance with
    p > 1 GeV, point into the timing detector, and the pair must point back to the target
    (IP < 2.5 m at z = -2.14 m); the vertex is inside the vessel by construction.
    Returns (tracks with the B state at z = 82 and the vertex, vertex z per track, event id per
    track, acceptance counts). Tracks are ordered all first daughters, then all second daughters.
    """
    keep_rows, stats = [], dict(generated=0, vertex_in_vessel=0, in_sst1=0, p_and_timing=0, ip=0)
    have = 0
    for _ in range(max_rounds):
        m = max(4 * (n - have), 10_000)
        p = 10 ** rng.uniform(np.log10(p_range[0]), np.log10(p_range[1]), m)
        pt = rng.exponential(pt_mean, m)
        phi = rng.uniform(0, 2 * np.pi, m)
        tx, ty = pt / p * np.cos(phi), pt / p * np.sin(phi)
        z_v = rng.uniform(geo.Z_VESSEL_MIN, geo.Z_VESSEL_MAX, m)
        x_v, y_v = tx * z_v, ty * z_v
        hx, hy = geo.vessel_half_widths(z_v)
        stats["generated"] += m
        ok = (np.abs(x_v) < hx) & (np.abs(y_v) < hy)
        stats["vertex_in_vessel"] += int(ok.sum())
        direction = np.stack([tx, ty, np.ones(m)], 1)
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        e_parent = np.sqrt(p ** 2 + mass ** 2)
        beta = direction * (p / e_parent)[:, None]
        e_star = mass / 2
        p_star = np.sqrt(max(e_star ** 2 - m_daughter ** 2, 0))
        u = rng.normal(size=(m, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        d1 = _boost(u * p_star, np.full(m, e_star), beta)
        d2 = _boost(-u * p_star, np.full(m, e_star), beta)
        states = []
        for d in (d1, d2):
            with np.errstate(divide="ignore", invalid="ignore"):
                x82 = x_v + d[:, 0] / d[:, 2] * (geo.Z_SST1 - z_v)
                y82 = y_v + d[:, 1] / d[:, 2] * (geo.Z_SST1 - z_v)
                xt = x_v + d[:, 0] / d[:, 2] * (geo.Z_TIMING - z_v)
                yt = y_v + d[:, 1] / d[:, 2] * (geo.Z_TIMING - z_v)
            acc = (d[:, 2] > 0) & (np.abs(x82) < geo.SST1_HALF_X) & (np.abs(y82) < geo.SST1_HALF_Y)
            sel = acc & (np.linalg.norm(d, axis=1) > 1) & (np.abs(xt) < geo.TIMING_HALF_X) & (np.abs(yt) < geo.TIMING_HALF_Y)
            states.append((x82, y82, d, acc, sel))
        acc2 = ok & states[0][3] & states[1][3]
        sel2 = acc2 & states[0][4] & states[1][4]
        tot = d1 + d2
        dz = geo.Z_TARGET - z_v
        ip = np.hypot(x_v + tot[:, 0] / tot[:, 2] * dz, y_v + tot[:, 1] / tot[:, 2] * dz)
        good = sel2 & (ip < 2.5)
        stats["in_sst1"] += int(acc2.sum()); stats["p_and_timing"] += int(sel2.sum()); stats["ip"] += int(good.sum())
        idx = np.where(good)[0][:n - have]
        keep_rows.append((idx, x_v, y_v, z_v, states))
        have += len(idx)
        if have >= n:
            break
    cols = {k: [] for k in ("bx", "by", "bz", "bpx", "bpy", "bpz", "vx", "vy", "vz")}
    for daughter in (0, 1):
        for idx, x_v, y_v, z_v, states in keep_rows:
            x82, y82, d, _, _ = states[daughter]
            cols["bx"].append(x82[idx]); cols["by"].append(y82[idx]); cols["bz"].append(np.full(len(idx), geo.Z_SST1))
            cols["bpx"].append(d[idx, 0]); cols["bpy"].append(d[idx, 1]); cols["bpz"].append(d[idx, 2])
            cols["vx"].append(x_v[idx]); cols["vy"].append(y_v[idx]); cols["vz"].append(z_v[idx])
    tr = {k: np.concatenate(v) for k, v in cols.items()}
    n_ev = len(tr["bx"]) // 2
    return tr, tr["vz"], np.tile(np.arange(n_ev), 2), stats
