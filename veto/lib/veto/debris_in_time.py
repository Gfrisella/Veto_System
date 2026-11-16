"""EM debris IN TIME with its parent muon (ISSUES D14.1, 16 Nov 2026).

Until now the shield-end e / gamma (bin/em_debris.py, ISSUES B2) entered only as random hits (rate maps, run_reco
--em-debris). Here each muon of an event also brings its OWN debris to the UBT planes, in time with it, so that it can
tag the muon (or an event) even where the muon's own hit misses the plane or the window.

There is no row link between the reco inputs (states at 30.6 / 82 m) and the MS7 parents of the debris simulation
(paths.ms7_parents, state at 29.20 m; em_exit_*: 2e6 parents drawn with replacement). So the debris is attached by
KINEMATIC MATCHING: for each muon, one of the K nearest simulated parents in (x, y at the UBT tile plane, slopes, log p)
is drawn (parents with no debris included, so the multiplicity is unbiased); its e / gamma crossings of the UBT plane
(z = Z_UBT, em_raw plane 0) are shifted by the muon - parent offset at that plane. The straw plane (30.73 m) gets the
same particles projected back along their own direction. Muons with no parent within MAX_DPOS / MAX_DLOGP get none
(muons not from the MS7 iron region).

Detection as em_debris.detection_probability: e+- with the plane efficiency, gamma only if it converts in the plane.
Time: the muon's own time at the plane + the debris delay (t - flight time of a straight light path from the shield
exit), smeared with the plane resolution (straws: anywhere in the drift window).
"""

from functools import lru_cache
from pathlib import Path

import numpy as np

from . import em_debris
from . import geometry as geo
from . import muon_types as mt
from . import paths

C = 0.299792458                       # m / ns
Z_EXIT = 29.90                        # m, stage-1 exit plane of bin/em_debris.py
K_NEIGHBOURS = 5
SCALE = np.array([0.05, 0.05, 0.005, 0.005, 0.2])   # m, m, rad, rad, ln p: matching metric
MAX_DPOS, MAX_DLOGP = 0.10, 0.3       # no debris if the nearest parent is further than this
RAW = "em_raw_n2000000_d28.4576.npz"
EXIT = "em_exit_n2000000_d28.4576.npz"


@lru_cache(maxsize=1)
def library(em_dir=None):
    """Parents (kinematics at the UBT tile plane) and their e / gamma crossings of plane 0, grouped by parent."""
    from scipy.spatial import cKDTree
    d = Path(em_dir) if em_dir else paths.results_dir() / "em_debris"
    ex = np.load(d / EXIT, allow_pickle=True)
    par = np.load(str(ex["parents"]))
    row = ex["parent_row"]
    z0 = par["z"][row]
    tx, ty = par["px"][row] / par["pz"][row], par["py"][row] / par["pz"][row]
    px = par["x"][row] + tx * (geo.Z_UBT - z0)
    py = par["y"][row] + ty * (geo.Z_UBT - z0)
    lp = np.log(np.sqrt(par["px"][row] ** 2 + par["py"][row] ** 2 + par["pz"][row] ** 2))
    raw = np.load(d / RAW)
    m = (raw["plane"] == 0) & np.isin(np.abs(raw["pdg"]), (11, 22))
    ev = raw["ev"][m]
    order = np.argsort(ev, kind="stable")
    deb = {k: raw[k][m][order] for k in ("pdg", "x", "y", "px", "py", "pz", "t")}
    ev = ev[order]
    start = np.searchsorted(ev, np.arange(len(row) + 1))
    feats = np.stack([px, py, tx, ty, lp], 1) / SCALE
    return dict(tree=cKDTree(feats), feats=feats, px=px, py=py, lp=lp, start=start, deb=deb)


def hits(tr, event, design, rng, muon_time_at=None):
    """Debris hits of each track's own parent muon on the UBT planes: dict event, track, plane, u, y, t (as own_hits).
    tr: the true (scattered) tracks of run_events; muon_time_at(plane) -> (n,) own time of each track at the plane."""
    L = library()
    a = np.stack([tr["ax"], tr["ay"], tr["az"]], 1)
    k = np.stack([tr["kx"], tr["ky"], tr["kz"]], 1)
    b = np.stack([tr["bx"], tr["by"], tr["bz"]], 1)
    xu, yu, ok = mt.path_at_z(a, k, b, geo.Z_UBT)
    ok &= (tr["apz"] > 0) & (tr["az"] < geo.Z_UBT + mt.A_TOLERANCE)
    # slopes of the first segment (A -> K for bounced tracks, else A -> B)
    nxt = np.where(np.isfinite(k[:, 2:3]) & (k[:, 2:3] > a[:, 2:3]), k, b)
    dz = nxt[:, 2] - a[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        tx, ty = (nxt[:, 0] - a[:, 0]) / dz, (nxt[:, 1] - a[:, 1]) / dz
    p = np.sqrt(tr["bpx"] ** 2 + tr["bpy"] ** 2 + tr["bpz"] ** 2)
    out = {k_: [] for k_ in ("event", "track", "plane", "u", "y", "t")}
    idx = np.nonzero(ok & np.isfinite(tx) & np.isfinite(ty) & (p > 0))[0]
    if not len(idx):
        return {k_: np.empty(0) for k_ in out}
    q = np.stack([xu[idx], yu[idx], tx[idx], ty[idx], np.log(p[idx])], 1) / SCALE
    _, nn = L["tree"].query(q, k=K_NEIGHBOURS)
    pick = nn[np.arange(len(idx)), rng.integers(0, K_NEIGHBOURS, len(idx))]
    close = (np.hypot(L["px"][pick] - xu[idx], L["py"][pick] - yu[idx]) < MAX_DPOS) & (np.abs(L["lp"][pick] - np.log(p[idx])) < MAX_DLOGP)
    idx, pick = idx[close], pick[close]
    n_deb = L["start"][pick + 1] - L["start"][pick]
    trk = np.repeat(idx, n_deb)
    rows = np.concatenate([np.arange(L["start"][j], L["start"][j + 1]) for j in pick]) if len(pick) else np.zeros(0, int)
    if not len(rows):
        return {k_: np.empty(0) for k_ in out}
    D = L["deb"]
    shift_x, shift_y = np.repeat(xu[idx] - L["px"][pick], n_deb), np.repeat(yu[idx] - L["py"][pick], n_deb)
    dx, dy = D["x"][rows] + shift_x, D["y"][rows] + shift_y
    delay = D["t"][rows] - (geo.Z_UBT - Z_EXIT) / C
    is_e = np.abs(D["pdg"][rows]) == 11
    for plane in design.planes:
        if plane.kind not in ("ubt", "tiles"):
            continue
        dzp = plane.z - geo.Z_UBT                  # straws 0.40 m upstream: project back along the particle
        with np.errstate(invalid="ignore", divide="ignore"):
            x = dx + D["px"][rows] / D["pz"][rows] * dzp
            y = dy + D["py"][rows] / D["pz"][rows] * dzp
        u = plane.to_local(x)
        inside = plane.inside(u, y) & np.isfinite(u)
        p_e, p_g = em_debris.detection_probability(plane)
        fire = inside & (rng.random(len(rows)) < np.where(is_e, p_e, p_g))
        if not fire.any():
            continue
        t_mu = muon_time_at(plane)[trk[fire]] if muon_time_at else np.zeros(fire.sum())
        t = t_mu + delay[fire] + rng.normal(0, plane.sigma_t * 1e9, fire.sum())
        if plane.time_window:
            t = t + rng.uniform(-0.5, 0.5, fire.sum()) * plane.time_window * 1e9
        uc, yc = plane.cell_centre(u[fire], y[fire])
        out["event"].append(event[trk[fire]]); out["track"].append(trk[fire])
        out["plane"].append(np.full(fire.sum(), plane.index)); out["u"].append(uc); out["y"].append(yc); out["t"].append(t)
    return {k_: (np.concatenate(v) if v else np.empty(0)) for k_, v in out.items()}
