"""Surrounding Background Tagger (SBT) as a detector (12 Nov 2026, Guglielmo: reference design "SBT + UBT").

Geometry and response from SPSC-SR-370 (Annual Report 2025, 3.3.3), SPSC-SR-389 (2026, 4.2.3, 5) and FairShip 26.10
(geometry/veto_config_helium.yaml, veto/veto.cxx):
  cells of AlMg4.5 filled with 20 cm of liquid scintillator (LAB + PPO) on the four lateral faces of the decay vessel
  (frustum 31.13 - 81.13 m, half widths geometry.vessel_half_widths); rings 82 cm long in z (80 cm the first);
  each side wall (x = +-X(z)) split into 4 cells in y (3 longitudinal ribs), top / bottom (y = +-Y(z)) into 3 cells in x;
  2 WOMs per cell (40 SiPMs in 8 groups), time resolution ~1 ns; threshold 30 - 90 MeV by region (FairShip reco 45 MeV),
  efficiency > 99.3 % above it. The entrance window is not covered (the UBT is there).
Model:
  deposit = DEDX_LS x the path of the track inside the cell shell (20 cm scintillator outside the balloon), split per
  ring (cell): grazing muons running along the vessel deposit tens of MeV per cell; a hit when above the threshold, with
  the efficiency.
  own hits: crossings of the event's muon paths (A -> K -> B, or A -> B) with the SBT surface.
  random hits: per-cell rates from every muon of the design store crossing the surface (cell_rates, cached per store and
  threshold), Poisson per event in the time window.
  tagging, mode "pointing" (default): each measured track is extrapolated back from SST1 to where it leaves the vessel
  upstream of its vertex (z_limit); a hit in a cell within the search radius r (reco's r0 + (r_slope + r_ms / p) L) and in
  time (+- n_sigma x 1 ns) tags it. Mode "any": any hit in time anywhere in the SBT tags the event (as SR-389 sec. 5).
Not modelled yet: neutral particles (photons, neutrons) and EM debris in the cells, the dead space of ribs / walls,
the Al / LS as interaction targets (interactions in the SBT material).
"""

import numpy as np

from . import geometry as geo
from . import paths

T_LS = 0.20                  # m of liquid scintillator
DEDX_LS = 150.0              # MeV / m (~1.75 MeV cm^2/g x 0.86 g/cm^3): 30 MeV for a perpendicular MIP (SR-389)
RING0, RING = 0.80, 0.82     # m, first ring, other rings (FairShip veto.cxx)
N_SEG_X_FACE, N_SEG_Y_FACE = 4, 3   # cells across a side wall (3 ribs) and across top / bottom (2 ribs)
FACES = ("+x", "-x", "+y", "-y")
THRESHOLDS = (30.0, 45.0, 60.0, 90.0)   # MeV, the cached rate maps
C = 0.299792458              # m / ns
DEFAULT = dict(threshold_mev=30.0, efficiency=0.993, sigma_t=1.0e-9, n_sigma=3.0, mode="pointing",
               cost_fraction_full=0.59, budget_kchf=8000.0,   # full SBT ~59 % of the reference budget (placeholder)
               rate_scale=1.0)   # x random-hit rate: our cells see muons only (Nov-4: 2.07 MHz > 30 MeV); SR-389 quotes
                                 # 27 MHz > 30 MeV with the particles scattered from the cavern walls and e / gamma -> 13
CHANNELS_PER_CELL = 16       # 2 WOMs x 8 SiPM groups (SR-370)

_Z0, _Z1 = geo.Z_VESSEL_MIN, geo.Z_VESSEL_MAX
_SX = (geo.VESSEL_X_OUT - geo.VESSEL_X_IN) / (_Z1 - _Z0)
_SY = (geo.VESSEL_Y_OUT - geo.VESSEL_Y_IN) / (_Z1 - _Z0)
RING_EDGES = np.concatenate([[_Z0, _Z0 + RING0], np.arange(_Z0 + RING0 + RING, _Z1 + 1e-9, RING)])
if RING_EDGES[-1] < _Z1 - 1e-6:
    RING_EDGES = np.append(RING_EDGES, _Z1)
N_RING = len(RING_EDGES) - 1


def config(raw):
    """The SBT settings of a design ("sbt" in the JSON; None / missing = no SBT)."""
    s = raw.get("sbt")
    if not s:
        return None
    out = dict(DEFAULT)
    out.update({k: v for k, v in s.items() if k in DEFAULT or k in ("rings", "faces")})
    out.setdefault("rings", [0, N_RING])            # instrumented rings [first, last) (Run-4 partial scenarios)
    out.setdefault("faces", list(FACES))
    return out


def n_cells(cfg):
    r0, r1 = cfg["rings"]
    per_ring = sum(N_SEG_X_FACE if f in ("+x", "-x") else N_SEG_Y_FACE for f in cfg["faces"])
    return (r1 - r0) * per_ring


def area(cfg):
    """m^2 of instrumented surface."""
    r0, r1 = cfg["rings"]
    z = np.linspace(RING_EDGES[r0], RING_EDGES[r1], 200)
    hx, hy = geo.vessel_half_widths(z)
    a = 0.0
    for f in cfg["faces"]:
        a += np.trapezoid(2 * (hy if f in ("+x", "-x") else hx), z)
    return float(a)


def full_area():
    return area(dict(rings=[0, N_RING], faces=list(FACES)))


def cost_kchf(cfg, budget_kchf):
    """Placeholder: the full SBT is cost_fraction_full of the reference budget, scaled by the instrumented area."""
    return cfg["cost_fraction_full"] * budget_kchf * area(cfg) / full_area()


# ---------------------------------------------------------------- geometry of crossings

def crossings(p, q):
    """Crossings of the segments p -> q (n, 3) with the four lateral faces of the vessel frustum (z0 < z < z1).
    Returns dict: seg (index), face (0..3), t (fraction along the segment), x, y, z, cos (|cos| to the wall normal)."""
    d = q - p
    out = {k: [] for k in ("seg", "face", "t", "x", "y", "z", "cos")}
    for fi, (axis, sign) in enumerate(((0, 1), (0, -1), (1, 1), (1, -1))):
        h0, s = (geo.VESSEL_X_IN, _SX) if axis == 0 else (geo.VESSEL_Y_IN, _SY)
        # sign * (p_a + t d_a) = h0 + s (p_z + t d_z - z0)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (h0 + s * (p[:, 2] - _Z0) - sign * p[:, axis]) / (sign * d[:, axis] - s * d[:, 2])
        ok = np.isfinite(t) & (t > 0) & (t <= 1)
        i = np.nonzero(ok)[0]
        if not len(i):
            continue
        pt = p[i] + t[i, None] * d[i]
        hx, hy = geo.vessel_half_widths(pt[:, 2])
        other = np.abs(pt[:, 1 - axis]) <= (hy if axis == 0 else hx)
        inz = (pt[:, 2] > _Z0) & (pt[:, 2] < _Z1)
        k = other & inz
        i, pt = i[k], pt[k]
        nrm = np.zeros(3); nrm[axis] = sign; nrm[2] = -s
        nrm /= np.linalg.norm(nrm)
        dd = d[i] / np.linalg.norm(d[i], axis=1, keepdims=True)
        out["seg"].append(i); out["face"].append(np.full(len(i), fi)); out["t"].append(t[i])
        out["x"].append(pt[:, 0]); out["y"].append(pt[:, 1]); out["z"].append(pt[:, 2])
        out["cos"].append(np.abs(dd @ nrm))
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in out.items()}


def cell_of(face, x, y, z):
    """(ring, segment) of points on the surface; ring = -1 outside the rings."""
    ring = np.clip(np.searchsorted(RING_EDGES, z, side="right") - 1, -1, N_RING - 1)
    hx, hy = geo.vessel_half_widths(z)
    side = np.isin(face, (0, 1))
    across = np.where(side, (y + hy) / (2 * hy), (x + hx) / (2 * hx))
    nseg = np.where(side, N_SEG_X_FACE, N_SEG_Y_FACE)
    seg = np.clip((across * nseg).astype(int), 0, nseg - 1)
    return ring, seg


def cell_id(face, ring, seg):
    return (face * 1000 + ring) * 10 + seg


def instrumented(cfg, face, ring):
    fi = [FACES.index(f) for f in cfg["faces"]]
    return np.isin(face, fi) & (ring >= cfg["rings"][0]) & (ring < cfg["rings"][1])


def deposit_mev(cos):
    """Energy in the scintillator of a crossing charged track: path T_LS / cos, capped at one ring length."""
    path = np.minimum(T_LS / np.maximum(cos, 1e-6), RING)
    return DEDX_LS * path


# ---------------------------------------------------------------- hits

def path_segments(tr):
    """The stored muon paths as straight segments: A -> K -> B (bounced) or A -> B. Returns (p, q, row, len_after) with
    len_after the path length from q to B (0 for the last segment)."""
    a = np.stack([tr["ax"], tr["ay"], tr["az"]], 1).astype(np.float64)
    b = np.stack([tr["bx"], tr["by"], tr["bz"]], 1).astype(np.float64)
    k = np.stack([tr["kx"], tr["ky"], tr["kz"]], 1).astype(np.float64) if "kx" in tr else np.full_like(a, np.nan)
    bounced = np.isfinite(k[:, 2])
    n = len(a)
    idx = np.arange(n)
    pk = np.where(bounced[:, None], k, b)
    p = np.concatenate([a, pk[bounced]]); q = np.concatenate([pk, b[bounced]])
    row = np.concatenate([idx, idx[bounced]])
    after = np.concatenate([np.where(bounced, np.linalg.norm(b - k, axis=1), 0.0), np.zeros(bounced.sum())])
    return p, q, row, after


T_SHELL = T_LS + 0.016       # m, scintillator + the two 0.8 cm AlMg walls: the cell shell outside the balloon


def _frustum_interval(p, d, grow):
    """[t_in, t_out] of the lines p + t d (t in [0, 1]) inside the vessel frustum grown by `grow` in x and y
    (Cyrus-Beck clipping on the six planes; empty: t_in >= t_out)."""
    n = len(p)
    lo, hi = np.zeros(n), np.ones(n)
    planes = [((0, 0, -1), -_Z0), ((0, 0, 1), _Z1)]                 # z >= z0, z <= z1
    for axis, h0, sl in ((0, geo.VESSEL_X_IN, _SX), (1, geo.VESSEL_Y_IN, _SY)):
        for sign in (1, -1):
            nv = [0.0, 0.0, -sl]; nv[axis] = sign
            planes.append((tuple(nv), h0 + grow - sl * _Z0))        # sign * x_a - sl z <= h0 + grow - sl z0
    for nv, c in planes:
        nv = np.array(nv)
        num = c - p @ nv
        den = d @ nv
        with np.errstate(divide="ignore", invalid="ignore"):
            t = num / den
        enter = den < 0; leave = den > 0; par = den == 0
        lo = np.where(enter, np.maximum(lo, t), lo)
        hi = np.where(leave, np.minimum(hi, t), hi)
        bad = par & (num < 0)
        lo = np.where(bad, 1.0, lo); hi = np.where(bad, 0.0, hi)
    return lo, hi


def shell_pieces(p, q):
    """Pieces of the segments p -> q inside the SBT shell (between the balloon and the balloon grown by T_SHELL), split at
    the ring edges. Returns dict seg, face (0..3), ring, cellseg, length [m] in the scintillator, z (mid), t_mid."""
    d = q - p
    L = np.linalg.norm(d, axis=1)
    a0, b0 = _frustum_interval(p, d, T_SHELL)
    a1, b1 = _frustum_interval(p, d, 0.0)
    has_in = b1 > a1
    pieces = []                                                     # (seg, ta, tb)
    for ta, tb in ((a0, np.where(has_in, np.minimum(b0, a1), b0)), (np.where(has_in, np.maximum(a0, b1), 1.0), b0)):
        ok = (tb > ta) & (b0 > a0)
        if not has_in.all():
            ok &= has_in | (ta == a0)                               # no inner chord: one piece only
        i = np.nonzero(ok)[0]
        pieces.append((i, ta[i], tb[i]))
    seg = np.concatenate([x[0] for x in pieces]); ta = np.concatenate([x[1] for x in pieces]); tb = np.concatenate([x[2] for x in pieces])
    if not len(seg):
        return {k: np.zeros(0) for k in ("seg", "face", "ring", "cellseg", "length", "z", "t_mid")}
    # split at the ring edges (z linear in t)
    za, zb = p[seg, 2] + ta * d[seg, 2], p[seg, 2] + tb * d[seg, 2]
    zlo, zhi = np.minimum(za, zb), np.maximum(za, zb)
    out = {k: [] for k in ("seg", "ta", "tb")}
    for r in range(N_RING):
        e0, e1 = RING_EDGES[r], RING_EDGES[r + 1]
        m = (zhi > e0) & (zlo < e1)
        if not m.any():
            continue
        dz = d[seg[m], 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t0 = np.where(dz != 0, (e0 - p[seg[m], 2]) / dz, ta[m]); t1 = np.where(dz != 0, (e1 - p[seg[m], 2]) / dz, tb[m])
        lo_, hi_ = np.minimum(t0, t1), np.maximum(t0, t1)
        u0, u1 = np.maximum(ta[m], lo_), np.minimum(tb[m], hi_)
        k = u1 > u0
        out["seg"].append(seg[m][k]); out["ta"].append(u0[k]); out["tb"].append(u1[k])
    sg = np.concatenate(out["seg"]); ta = np.concatenate(out["ta"]); tb = np.concatenate(out["tb"])
    tm = 0.5 * (ta + tb)
    mid = p[sg] + tm[:, None] * d[sg]
    hx, hy = geo.vessel_half_widths(np.clip(mid[:, 2], _Z0, _Z1))
    xface = (np.abs(mid[:, 0]) - hx) >= (np.abs(mid[:, 1]) - hy)
    face = np.where(xface, np.where(mid[:, 0] > 0, 0, 1), np.where(mid[:, 1] > 0, 2, 3))
    ring, cseg = cell_of(face, np.clip(mid[:, 0], -hx, hx), np.clip(mid[:, 1], -hy, hy), mid[:, 2])
    length = (tb - ta) * L[sg] * T_LS / T_SHELL                      # the scintillator part of the shell path
    return dict(seg=sg, face=face, ring=ring, cellseg=cseg, length=length, z=mid[:, 2], t_mid=tm)


def own_hits(tr, event, cfg, rng):
    """SBT hits of the events' own tracks (path inside the shell, per cell): event, track, cell, t [ns] (t = 0 at B)."""
    p, q, row, after = path_segments(tr)
    c = shell_pieces(p, q)
    if not len(c["seg"]):
        return dict(event=np.zeros(0, int), track=np.zeros(0, int), cell=np.zeros(0, int), t=np.zeros(0))
    s = c["seg"].astype(int)
    ok = (c["ring"] >= 0) & instrumented(cfg, c["face"], c["ring"]) & (DEDX_LS * c["length"] > cfg["threshold_mev"])
    ok &= rng.random(len(ok)) < cfg["efficiency"]
    s = s[ok]
    length = np.linalg.norm(q[s] - p[s], axis=1) * (1 - c["t_mid"][ok]) + after[s]
    t = -length / C + rng.normal(0, cfg["sigma_t"] * 1e9, ok.sum())
    tr_row = row[s]
    return dict(event=event[tr_row], track=tr_row, cell=cell_id(c["face"][ok], c["ring"][ok], c["cellseg"][ok]), t=t)


def cell_rates(threshold_mev, dataset=None):
    """Muons per spill per SBT cell above the threshold, from every muon path of the design store (path inside the shell
    per cell; cached). Returns (cell ids, rate per spill)."""
    from . import io
    dataset = dataset or paths.MUON_DATASET
    thr = min(THRESHOLDS, key=lambda x: abs(x - threshold_mev))
    f = paths.results_dir() / "rate_maps" / f"sbt_cells_shell_{dataset}.npz"
    if not f.exists():
        c, _ = io.open_store(paths.store_dir(dataset))
        n = len(c["weight"])
        acc = {t_: {} for t_ in THRESHOLDS}
        for s0 in range(0, n, 10_000_000):
            sl = slice(s0, min(n, s0 + 10_000_000))
            d = {k: np.asarray(c[k][sl]) for k in ("x_30", "y_30", "z_30", "x_82", "y_82", "z_82", "kx", "ky", "kz",
                                                     "weight", "type")}
            m = (d["type"] <= 6) & (d["z_30"] > 30.5)
            tr = dict(ax=d["x_30"][m], ay=d["y_30"][m], az=d["z_30"][m], bx=d["x_82"][m], by=d["y_82"][m],
                      bz=d["z_82"][m], kx=d["kx"][m], ky=d["ky"][m], kz=d["kz"][m])
            w = d["weight"][m].astype(np.float64)
            p, q, row, _ = path_segments(tr)
            cr = shell_pieces(p, q)
            ok = cr["ring"] >= 0
            dep = DEDX_LS * cr["length"]
            ids = cell_id(cr["face"], cr["ring"], cr["cellseg"])
            for t_ in THRESHOLDS:
                k = ok & (dep > t_)
                u, inv = np.unique(ids[k], return_inverse=True)
                ssum = np.bincount(inv, w[row[cr["seg"][k].astype(int)]])
                for cid, val in zip(u, ssum):
                    acc[t_][int(cid)] = acc[t_].get(int(cid), 0.0) + val
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez(f, **{f"ids_{int(t_)}": np.array(sorted(acc[t_]), np.int64) for t_ in THRESHOLDS},
                 **{f"rate_{int(t_)}": np.array([acc[t_][k] for k in sorted(acc[t_])]) for t_ in THRESHOLDS},
                 dataset=dataset)
    d = np.load(f)
    return d[f"ids_{int(thr)}"], d[f"rate_{int(thr)}"]


def random_hits(n_events, cfg, spill_s, rng, window_ns):
    """Random SBT hits per event within +- window_ns of each cell's nominal time (as reco.accidental_hits)."""
    ids, rate = cell_rates(cfg["threshold_mev"])
    face, ring = ids // 10 // 1000, ids // 10 % 1000
    keep = instrumented(cfg, face, ring)
    ids, rate = ids[keep], rate[keep] * cfg.get("rate_scale", 1.0)
    if not len(ids) or rate.sum() <= 0:
        return dict(event=np.zeros(0, int), cell=np.zeros(0, int), t=np.zeros(0))
    mean = rate.sum() * 2 * window_ns * 1e-9 / spill_s
    n = rng.poisson(mean, n_events)
    k = rng.choice(len(ids), int(n.sum()), p=rate / rate.sum())
    zc = 0.5 * (RING_EDGES[ids[k] // 10 % 1000] + RING_EDGES[ids[k] // 10 % 1000 + 1])
    nominal = -(geo.Z_SST1 - zc) / C
    return dict(event=np.repeat(np.arange(n_events), n), cell=ids[k], t=nominal + rng.uniform(-window_ns, window_ns, len(k)))


# ---------------------------------------------------------------- tagging

def tag(tr_true, meas, event, z_limit, cfg, reco_cfg, rng, n_events, with_own_hits=True):
    """SBT decision per event (and per track for "pointing"). Returns dict vetoed (n_events), explained (n_tracks)."""
    window = cfg["n_sigma"] * np.hypot(cfg["sigma_t"] * 1e9, reco_cfg.sigma_t_track)      # ns
    own = own_hits(tr_true, event, cfg, rng) if with_own_hits else dict(event=np.zeros(0, int), cell=np.zeros(0, int),
                                                                         t=np.zeros(0), track=np.zeros(0, int))
    acc = random_hits(n_events, cfg, reco_cfg.spill_s, rng, window + 2.0)
    h_ev = np.concatenate([own["event"], acc["event"]]).astype(np.int64)
    h_cell = np.concatenate([own["cell"], acc["cell"]]).astype(np.int64)
    h_t = np.concatenate([own["t"], acc["t"]])
    vetoed = np.zeros(n_events, bool)
    explained = np.zeros(len(event), bool)
    if cfg["mode"] == "any":                       # any hit in time (relative to the cell's nominal time)
        zc = 0.5 * (RING_EDGES[h_cell // 10 % 1000] + RING_EDGES[h_cell // 10 % 1000 + 1])
        intime = np.abs(h_t + (geo.Z_SST1 - zc) / C) < window
        vetoed[np.unique(h_ev[intime])] = True
        return dict(vetoed=vetoed, explained=explained, n_hits=len(h_ev))
    # pointing: the backward line of each measured track, from its vertex (z_limit) upstream
    zv = np.minimum(z_limit, geo.Z_SST1)
    dzv = meas["z"] - zv
    start = np.stack([meas["x"] - meas["tx"] * dzv, meas["y"] - meas["ty"] * dzv, zv], 1)
    dz_end = meas["z"] - (_Z0 - 0.01)
    end = np.stack([meas["x"] - meas["tx"] * dz_end, meas["y"] - meas["ty"] * dz_end, np.full(len(zv), _Z0 - 0.01)], 1)
    c = crossings(start, end)
    if not len(c["seg"]):
        return dict(vetoed=vetoed, explained=explained, n_hits=len(h_ev))
    s = c["seg"]
    L = meas["z"][s] - c["z"]                                           # ~ path length along z from SST1
    r = reco_cfg.r0 + (reco_cfg.r_slope + reco_cfg.r_ms / np.maximum(meas["p"][s], 0.1)) * L
    t_pred = meas["t"][s] - L / C
    # candidate cells: the ring range and segment range within r of the crossing point
    hits_by_ev = {}
    order = np.argsort(h_ev)
    he, hc, ht = h_ev[order], h_cell[order], h_t[order]
    bounds = np.searchsorted(he, np.arange(n_events + 1))
    for j in range(len(s)):
        e = int(event[s[j]])
        lo, hi = bounds[e], bounds[e + 1]
        if lo == hi:
            continue
        face = int(c["face"][j])
        zlo, zhi = c["z"][j] - r[j], c["z"][j] + r[j]
        r_lo = max(int(np.searchsorted(RING_EDGES, zlo, side="right") - 1), 0)
        r_hi = min(int(np.searchsorted(RING_EDGES, zhi, side="right") - 1), N_RING - 1)
        hx, hy = geo.vessel_half_widths(np.array([c["z"][j]]))
        if face in (0, 1):
            nseg, coord, half = N_SEG_X_FACE, c["y"][j], hy[0]
        else:
            nseg, coord, half = N_SEG_Y_FACE, c["x"][j], hx[0]
        s_lo = int(np.clip((coord - r[j] + half) / (2 * half) * nseg, 0, nseg - 1))
        s_hi = int(np.clip((coord + r[j] + half) / (2 * half) * nseg, 0, nseg - 1))
        cand = {cell_id(face, rr, ss) for rr in range(r_lo, r_hi + 1) for ss in range(s_lo, s_hi + 1)}
        m = np.isin(hc[lo:hi], list(cand)) & (np.abs(ht[lo:hi] - t_pred[j]) < window)
        if m.any():
            explained[s[j]] = True
            vetoed[e] = True
    return dict(vetoed=vetoed, explained=explained, n_hits=len(h_ev))
