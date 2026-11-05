"""Straw tube stations (SST1-4) as in FairShip, and their digitisation (post-processing realism, ISSUES A6 follow-up).

Geometry: FairShip strawtubes.cxx + geometry/strawtubes_config.yaml (SHiP CVMFS 25.09), stations at geometry.Z_SST:
    4 views per station: y1 (0), u (+4.57 deg), v (-4.57 deg), y2 (0); 2 layers per view -> 32 layers.
    layer z = z_station + (view - 1.5) * 12 cm + (layer - 0.5) * 1.732 cm
    straws: outer diameter 2 cm, wall 36 um (gas radius 0.9964 cm), half length = aperture half width 200 cm,
    along (cos a, sin a, 0) through (0, y_s), y_s = H + g - (s - 1/2) p_s + l o_s (s = 1 .. n), H = 300 cm,
    g = tan|a| 200 cm (stereo growth), p_s = 2 cm / cos a, o_s = 1 cm / cos a, n = ceil(2 (H + g) / p_s); layer 1 has
    one extra straw at the bottom.
Digitisation as FairShip (strawtubesHit + shipDigiReco): a straw is hit if the track crosses its gas (DOCA to the wire
< 0.9964 cm); measured drift distance |Gauss(DOCA, 120 um)|; only the earliest hit per straw (one track: one per
straw). No inefficiency, no noise hits, no t0 smearing (FairShip default withT0 = False).
Track fit: bin/genfit_straws.py (GenFit DAF, as shipDigiReco.findTracks), driven by bin/straw_reco.py.

    LAYERS                          table of the 32 layers (station, view, layer, z [m], angle [rad])
    layer_states(pos, mom, q)       (x, y, tx, ty) of tracks at every layer z (RK4 in the field map), lost mask
    digitize(states, rng)           hit table: track, detID (FairShip encoding), wire end points [cm], drift [cm]
"""

import math

import numpy as np

from . import geometry as geo

VIEW_ANGLE_DEG = 4.57
STRAW_DIAMETER = 0.02          # m
WALL = 36e-6                   # m
R_GAS = STRAW_DIAMETER / 2 - WALL
PITCH = 0.02                   # m, in y
LAYER_OFFSET = 0.01            # m
DZ_LAYER = 0.01732             # m
DZ_VIEW = 0.12                 # m
HALF_WIDTH = 2.00              # m, aperture (x) = straw half length
HALF_HEIGHT = 3.00             # m, aperture (y)
SIGMA_SPATIAL = 120e-6         # m
VIEW_ANGLES = (0.0, VIEW_ANGLE_DEG, -VIEW_ANGLE_DEG, 0.0)


def _layers():
    rows = []
    for st, zst in enumerate(geo.Z_SST, start=1):
        for v, adeg in enumerate(VIEW_ANGLES):
            for l in (0, 1):
                a = math.radians(adeg)
                growth = math.tan(abs(a)) * HALF_WIDTH
                pitch = PITCH / math.cos(a)
                offset = LAYER_OFFSET / math.cos(a)
                n = math.ceil(2 * (HALF_HEIGHT + growth) / pitch)
                ys = HALF_HEIGHT + growth - (np.arange(1, n + 1) - 0.5) * pitch + l * offset
                if l == 1:
                    ys = np.append(ys, HALF_HEIGHT + growth - (n - 0.5) * pitch - l * offset)
                rows.append(dict(station=st, view=v, layer=l, z=zst + (v - 1.5) * DZ_VIEW + (l - 0.5) * DZ_LAYER,
                                 angle=a, wire_y=ys))
    return rows


LAYERS = _layers()
LAYER_Z = np.array([r["z"] for r in LAYERS])


def detector_id(station, view, layer, straw):
    """FairShip strawtubes detector ID of the gas volume (statnb 1e6 + vnb 1e5 + lnb 1e4 + 2e3 + snb)."""
    return station * 1_000_000 + view * 100_000 + layer * 10_000 + 2_000 + straw


def layer_states(pos, mom, q, **kw):
    """States (n, 32, 4) = x, y, tx, ty [m] of the tracks at every layer z, propagated plane by plane with RK4 in the
    field map (spectrometer.to_z, no scattering), from pos / mom (n, 3) at a common plane upstream of SST1.
    Returns (states, p (n,), lost (n, 32): lost in the yoke before that layer)."""
    from . import spectrometer
    pos, mom = np.asarray(pos, np.float64), np.asarray(mom, np.float64)
    n = len(pos)
    states = np.full((n, len(LAYER_Z), 4), np.nan)
    lost = np.zeros((n, len(LAYER_Z)), bool)
    cur_p, cur_m, cur_lost = pos, mom, np.zeros(n, bool)
    for k, z in enumerate(LAYER_Z):
        cur_p, cur_m, l = spectrometer.to_z(cur_p, cur_m, q, z, **kw)
        cur_lost |= l
        states[:, k] = np.stack([cur_p[:, 0], cur_p[:, 1], cur_m[:, 0] / cur_m[:, 2], cur_m[:, 1] / cur_m[:, 2]], 1)
        lost[:, k] = cur_lost
    return states, np.linalg.norm(mom, axis=1), lost


def layer_states_g4(pos, mom, pdg, seed=1, g4=None):
    """As layer_states, but every track carried by Geant4 (bin/g4ship.sh: g4vessel with the 32 layer planes + the
    TD): multiple scattering in air, decays in flight (pi -> mu keeps the track), interactions, field, yoke losses.
    pos (n, 3) must be inside the Geant4 world (z > 29 m) and upstream of the first layer. Returns (states (n, 32, 4),
    p (n,), lost (n, 32): the track (or its decay product) did not reach that layer)."""
    from . import g4vessel as G
    g4 = g4 or G.init(seed, planes=tuple(LAYER_Z) + (geo.Z_TIMING,))
    pos, mom = np.asarray(pos, np.float64), np.asarray(mom, np.float64)
    n = len(pos)
    r = g4.transport(np.arange(n + 1), np.asarray(pdg, np.int32), mom[:, 0], mom[:, 1], mom[:, 2],
                     pos[:, 0], pos[:, 1], pos[:, 2])
    r = {k: np.asarray(v) for k, v in r.items()}
    states = np.full((n, len(LAYER_Z), 4), np.nan)
    m = (r["origin"] == 0) & (r["plane"] < len(LAYER_Z)) & (r["pz"] > 0)
    ev, pl = r["ev"][m], r["plane"][m]
    states[ev, pl] = np.stack([r["x"][m], r["y"][m], r["px"][m] / r["pz"][m], r["py"][m] / r["pz"][m]], 1)
    return states, np.linalg.norm(mom, axis=1), ~np.isfinite(states[:, :, 0])


def digitize(states, rng, lost=None, smear=True):
    """Hits of tracks with states (n, 32, 4) at the layers. Returns a dict of arrays (one row per hit):
    track, layer (0..31), station, detID, x1 y1 x2 y2 z (wire end points, cm, FairShip order bottom / top),
    dist (measured drift distance, cm), doca (true, cm)."""
    n = len(states)
    out = {k: [] for k in ("track", "layer", "station", "detID", "x1", "y1", "x2", "y2", "z", "dist", "doca")}
    for k, L in enumerate(LAYERS):
        x, y, tx, ty = (states[:, k, i] for i in range(4))
        ok = np.isfinite(x) & (np.abs(x) < HALF_WIDTH)
        if lost is not None:
            ok &= ~lost[:, k]
        a = L["angle"]
        ca, sa = math.cos(a), math.sin(a)
        w = y * ca - x * sa                                   # coordinate across the wires
        wires = L["wire_y"] * ca                              # ... of every wire (they pass through (0, y_s))
        order = np.argsort(wires)
        ws = wires[order]
        j = np.clip(np.searchsorted(ws, np.where(ok, w, 0.0)), 1, len(ws) - 1)
        norm = np.sqrt(1 + (tx * sa - ty * ca) ** 2)          # 3D distance factor (track inclination)
        for cand in (j - 2, j - 1, j, j + 1):                 # every straw whose gas the track can cross
            c = np.clip(cand, 0, len(ws) - 1)
            doca = np.abs(w - ws[c]) / norm
            t_along = x * ca + y * sa                         # along the wire: inside the straw length
            hit = ok & (cand >= 0) & (cand < len(ws)) & (doca < R_GAS) & (np.abs(t_along) < HALF_WIDTH)
            idx = np.nonzero(hit)[0]
            if not len(idx):
                continue
            s_idx = order[c[idx]]                             # straw index 0.. -> snb = s_idx + 1
            ysw = L["wire_y"][s_idx]
            d = doca[idx]
            dist = np.abs(rng.normal(d, SIGMA_SPATIAL)) if smear else d
            out["track"].append(idx); out["layer"].append(np.full(len(idx), k))
            out["station"].append(np.full(len(idx), L["station"]))
            out["detID"].append(detector_id(L["station"], L["view"], L["layer"], s_idx + 1))
            out["x1"].append(np.full(len(idx), -HALF_WIDTH * ca * 100)); out["y1"].append((ysw - HALF_WIDTH * sa) * 100)
            out["x2"].append(np.full(len(idx), HALF_WIDTH * ca * 100)); out["y2"].append((ysw + HALF_WIDTH * sa) * 100)
            out["z"].append(np.full(len(idx), L["z"] * 100)); out["dist"].append(dist * 100); out["doca"].append(d * 100)
    res = {k: np.concatenate(v) if v else np.zeros(0) for k, v in out.items()}
    o = np.lexsort((res["layer"], res["track"])) if len(res["track"]) else np.zeros(0, int)
    res = {k: v[o] for k, v in res.items()}
    # one hit per straw and track (earliest = the only one here); duplicates from the candidate loop removed
    if len(res["track"]):
        key = res["track"].astype(np.int64) * 10_000_000 + res["detID"]
        _, first = np.unique(key, return_index=True)
        first.sort()
        res = {k: v[first] for k, v in res.items()}
    return res


def hit_summary(h, n_tracks):
    """Per track: number of hits and of stations crossed (FairShip fit requirements: >= 25 hits, >= 3 stations)."""
    nh = np.bincount(h["track"], minlength=n_tracks)
    st = np.zeros((n_tracks, 5), bool)
    st[h["track"], h["station"]] = True
    return nh, st[:, 1:].sum(1)


def reconstruct(pos, mom, q, rng, workdir, workers=8, states=None, lost=None, min_hits=25, min_stations=3):
    """Post-processing reconstruction of tracks with the straws: layer states (RK4, or given: e.g. Geant4 with
    scattering), digitisation, GenFit DAF fit (bin/genfit_straws.py in the SHiP CVMFS ROOT, `workers` processes).
    pos / mom (n, 3), q (n,): states at a common plane before SST1. Returns dict of per-track arrays: n_hits,
    n_stations, fitted, converged, chi2, ndf, chi2ndf, p (fit), charge (fit), state at SST1 (x_ref, ..), good
    (FairShip: converged and 0 <= chi2/ndf < 50), plus p_true."""
    import subprocess
    from pathlib import Path
    from . import paths, spectrometer
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    n = len(pos)
    if states is None:
        states, _, lost = layer_states(pos, mom, q)
    h = digitize(states, rng, lost)
    f = spectrometer.field_map()
    grid = f.grid.reshape(*f.n, 3)
    chunks = np.array_split(np.arange(n), max(1, min(workers, n)))
    procs = []
    for i, c in enumerate(chunks):
        if not len(c):
            continue
        m = (h["track"] >= c[0]) & (h["track"] <= c[-1])
        sub = {k: v[m] for k, v in h.items()}
        sub["track"] = sub["track"] - c[0]
        hp, op = workdir / f"hits_{i}.npz", workdir / f"fit_{i}.npz"
        np.savez(hp, **sub, n_tracks=len(c), field=grid, x0=f.x0[0], y0=f.x0[1], z0=f.x0[2], dx=f.d[0], dy=f.d[1],
                 dz=f.d[2], z_seed_cm=100 * 0.5 * (geo.Z_VESSEL_MIN + geo.Z_VESSEL_MAX), z_ref_cm=100 * geo.Z_SST1,
                 min_hits=min_hits, min_stations=min_stations)
        log = open(workdir / f"fit_{i}.log", "w")
        procs.append((c, op, subprocess.Popen([str(paths.PROJECT_DIR / "bin/shipenv.sh"), "python3",
                                               str(paths.PROJECT_DIR / "bin/genfit_straws.py"), str(hp), str(op)],
                                              stdout=log, stderr=subprocess.STDOUT)))
    out = None
    for c, op, pr in procs:
        if pr.wait() != 0:
            raise RuntimeError(f"genfit_straws.py failed, see {op.with_suffix('.log')}")
        r = np.load(op)
        if out is None:
            out = {k: np.full(n, np.nan) for k in r.files}
        for k in r.files:
            out[k][c] = r[k]
    out["chi2ndf"] = out["chi2"] / out["ndf"]
    out["good"] = (out["converged"] == 1) & (out["chi2ndf"] >= 0) & (out["chi2ndf"] < 50)
    out["p_true"] = np.linalg.norm(mom, axis=1)
    out["q_true"] = np.asarray(q, float)
    return out
