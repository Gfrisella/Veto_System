"""The probe: a dense set of planes covering every place a veto could be, to measure (and to steer) the statistics
of a muon dataset cell by cell (bin/probe_statistics.py, bin/storage_policy_study.py, the "alpha" storage of
bin/generate_muon_study.py).

  UBT    z = 30.6, |x| < 0.5, y = -3.2 .. 3.2 (the full height a long UBT could have)
  fins   every dz from z = 30.6 to 80.6 on both sides, touching the vessel wall; right ones reach the cavern wall,
         left ones left_reach; y = -3.2 .. 3.2
  floor  planes under the vessel at floor_z, tilted by floor_tilt degrees towards the spectrometer (about x), from the
         floor line y = -3.2 up to the vessel bottom, |x| < X(z) + 1 m
Paths are A (z = 30.6) -> bounce K (NaN without) -> B (z = 82 or the stop), straight lines.
"""

import numpy as np

from . import geometry as geo
from . import muon_types as mt

DEFAULTS = dict(dz=2.5, left_reach=5.0, floor_z=(32.0, 40.0, 50.0, 60.0, 70.0), floor_tilt=30.0, cell=0.25)


def planes(dz=2.5, left_reach=5.0, floor_z=(32.0, 40.0, 50.0, 60.0, 70.0), floor_tilt=30.0, ubt=True, only_ubt=False):
    """List of planes: dict(name, kind, z0, tilt, u-range, y-range)."""
    out = []
    if ubt:
        out.append(dict(name="UBT z=30.6", kind="ubt", z0=float(geo.Z_UBT), tilt=0.0,
                        u=(-float(geo.UBT_HALF_X), float(geo.UBT_HALF_X)), y=(mt.FIN_Y_MIN, 3.2)))
    if only_ubt:
        return out
    for z in np.arange(geo.Z_VESSEL_MIN, geo.Z_VESSEL_MAX + 1e-6, dz):
        X = mt.side_plane_x(z)
        for side, sign, reach in (("right", -1, -mt.CAVERN_X_WALL - X), ("left", 1, left_reach)):
            out.append(dict(name=f"{side} fin z={z:.1f}", kind="fin", side=side, sign=sign, z0=float(z), tilt=0.0,
                            u=(0.0, float(reach)), y=(mt.FIN_Y_MIN, 3.2)))
    for z in floor_z:
        X = mt.side_plane_x(z) + 1.0
        _, hy = geo.vessel_half_widths(np.array([z]))
        out.append(dict(name=f"floor z={z:.1f}", kind="floor", z0=float(z), tilt=float(floor_tilt),
                        u=(-float(X), float(X)), y=(mt.FIN_Y_MIN, -float(hy[0]))))
    return out


def grid(pl, cell=0.25):
    """Cells per plane (nu, nv) and the offset of each plane's first cell in one flat index."""
    shape = [(int(np.ceil((p["u"][1] - p["u"][0]) / cell)), int(np.ceil((p["y"][1] - p["y"][0]) / cell))) for p in pl]
    off = np.concatenate([[0], np.cumsum([a * b for a, b in shape])]).astype(np.int64)
    return shape, off


BANDS = ("fin 0-0.5 m", "fin 0.5-1 m", "fin 1-2 m", "fin > 2 m", "floor", "ubt")


def band_labels(pl, cell=0.25):
    """Per plane, a label per cell (flat, as the probe's maps): fins by distance from the vessel wall, floor, ubt."""
    out = []
    for p in pl:
        nu = int(np.ceil((p["u"][1] - p["u"][0]) / cell))
        nv = int(np.ceil((p["y"][1] - p["y"][0]) / cell))
        u = p["u"][0] + (np.arange(nu) + 0.5) * cell
        if p["kind"] == "fin":
            lab = np.select([u < 0.5, u < 1.0, u < 2.0], list(BANDS[:3]), BANDS[3])
        else:
            lab = np.full(nu, p["kind"])
        out.append(np.repeat(lab, nv))
    return out


def crossing(a, k, b, plane):
    """Where the paths cross the plane: global (x, y, z) and a mask. Fins / UBT: z = z0. Floor: the plane through
    the line (z0, y = -3.2) rotated by tilt about x: f = (z - z0) - (y + 3.2) tan(tilt) = 0."""
    if plane["kind"] in ("fin", "ubt"):
        x, y, ok = mt.path_at_z(a, k, b, plane["z0"])
        return x, y, np.full(len(x), plane["z0"]), ok
    t = np.tan(np.radians(plane["tilt"]))
    f = lambda p: (p[:, 2] - plane["z0"]) - (p[:, 1] - mt.FIN_Y_MIN) * t
    bounced = np.isfinite(k[:, 2])
    hit = np.zeros(len(a), bool)
    pt = np.full((len(a), 3), np.nan)
    for p0, p1, use in ((a, np.where(bounced[:, None], k, b), np.ones(len(a), bool)), (k, b, bounced)):
        f0, f1 = f(p0), f(p1)
        with np.errstate(invalid="ignore", divide="ignore"):
            s = f0 / (f0 - f1)
        m = use & ~hit & np.isfinite(s) & (s >= 0) & (s <= 1)
        pt[m] = p0[m] + s[m, None] * (p1[m] - p0[m])
        hit |= m
    return pt[:, 0], pt[:, 1], pt[:, 2], hit


def local(plane, x, y, z):
    """Local (u, v) on the plane: fins u = distance from the vessel wall, v = y; UBT and floor u = x, v = y."""
    if plane["kind"] == "fin":
        return plane["sign"] * x - mt.side_plane_x(plane["z0"]), y
    return x, y


# ---------------------------------------------------------------------------------------------- torch (production)
def cells_torch(a, k, b, pl, cell=0.25):
    """Every (muon, cell) crossing on the GPU: a, k, b (n, 3) float64 torch tensors (k NaN without a bounce).
    Returns (muon index, flat cell index) as int64 tensors. Same geometry as crossing() / local() above."""
    import torch
    shape, off = grid(pl, cell)
    bounced = torch.isfinite(k[:, 2])
    end = torch.where(bounced[:, None], k, b)
    segs = ((a, end, torch.ones(len(a), dtype=torch.bool, device=a.device)), (k, b, bounced))
    idx, cel = [], []
    for i, p in enumerate(pl):
        x = torch.full((len(a),), float("nan"), dtype=a.dtype, device=a.device)
        y = torch.full_like(x, float("nan"))
        if p["kind"] in ("fin", "ubt"):
            z = torch.full_like(x, p["z0"])
            z = torch.where((z < a[:, 2]) & (a[:, 2] - z < mt.A_TOLERANCE), a[:, 2], z)
            for p0, p1, use in segs:
                dz = p1[:, 2] - p0[:, 2]
                t = (z - p0[:, 2]) / dz
                m = use & (dz > 0) & (t >= 0) & (t <= 1) & torch.isnan(x)
                x = torch.where(m, p0[:, 0] + t * (p1[:, 0] - p0[:, 0]), x)
                y = torch.where(m, p0[:, 1] + t * (p1[:, 1] - p0[:, 1]), y)
        else:
            tt = float(np.tan(np.radians(p["tilt"])))
            f = lambda q: (q[:, 2] - p["z0"]) - (q[:, 1] - mt.FIN_Y_MIN) * tt
            for p0, p1, use in segs:
                f0, f1 = f(p0), f(p1)
                s = f0 / (f0 - f1)
                m = use & torch.isnan(x) & torch.isfinite(s) & (s >= 0) & (s <= 1)
                x = torch.where(m, p0[:, 0] + s * (p1[:, 0] - p0[:, 0]), x)
                y = torch.where(m, p0[:, 1] + s * (p1[:, 1] - p0[:, 1]), y)
        u = p["sign"] * x - float(mt.side_plane_x(p["z0"])) if p["kind"] == "fin" else x
        ok = torch.isfinite(x) & (u >= p["u"][0]) & (u < p["u"][1]) & (y >= p["y"][0]) & (y < p["y"][1])
        r = torch.nonzero(ok).flatten()
        nu, nv = shape[i]
        iu = ((u[r] - p["u"][0]) / cell).long().clamp(0, nu - 1)
        iv = ((y[r] - p["y"][0]) / cell).long().clamp(0, nv - 1)
        idx.append(r)
        cel.append(int(off[i]) + iu * nv + iv)
    return torch.cat(idx), torch.cat(cel)
