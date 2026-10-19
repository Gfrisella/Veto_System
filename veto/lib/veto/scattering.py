"""Multiple Coulomb scattering of the muons after z = 30.6 m (ISSUES.md B1), small-angle and analytic.

The stored muon states have no scattering after the shield: A (z = 30.6 m) and B (z = 82 m) are
joined by straight lines (through the bounce point K for bounced muons). Here every muon collects
the scatterers along its true path, with x/X0:

    PVC walls, 0.3 mm (X0 = 19.63 cm)     vessel entrance (z = 30.6 m, inside the aperture),
                                          exit (z = 80.6 m) and the lateral walls where crossed;
                                          path in the wall = thickness / |cos(incidence)|
    UBT                                   straw plane at z = 30.6 m (technology x0) if crossed
    fins                                  technology x0 / |cos(incidence)| where crossed
    helium, X0 = 5671 m; air, 303.9 m     OFF by default (MEDIA = False): to be done once with
                                          Geant4; with MEDIA on, the path inside / outside the
                                          vessel, z = 30.6 .. 82 m, sliced in DZ_SLICE slabs

The angular spread follows Highland for the muon's TOTAL x/X0, shared between the scatterers in
proportion to their x/X0 (Gaussian kicks in x and y). Bounced muons scatter only after their bounce
point (the transport handled the rock). Signal daughters scatter from their decay vertex.

A kick theta at z_i moves the muon at any later plane z by theta * (z - z_i) and turns its slope by
theta (small angles): `offset_at(z)` and the scattered state at z = 82 m follow without re-tracking.
Planes and media are slices in z: the volumes are cut into DZ_SLICE slabs, each a thin scatterer
at its centre.
"""

import numpy as np

from . import geometry as geo
from . import muon_types as mt
from . import technology as tech
from .tracks import vessel_crossings

X0_AIR, X0_HELIUM, X0_PVC = 303.9, 5671.0, 0.1963        # m (PDG)
PVC_THICKNESS = 0.3e-3                                     # m, vessel walls
DZ_SLICE = 1.0                                             # m, slicing of air / helium
MEDIA = False                                              # air and helium: later, from Geant4
M_MU = 0.10566                                             # GeV
MIN_COS = 0.02                                             # cap grazing incidence (path <= 50 x thickness)


def highland(p, x_total):
    """theta0 [rad] for momentum p [GeV] (muon) after x_total radiation lengths."""
    beta = p / np.sqrt(p ** 2 + M_MU ** 2)
    x = np.maximum(x_total, 1e-12)
    return 0.0136 / (beta * p) * np.sqrt(x) * np.maximum(1 + 0.038 * np.log(x), 0.1)


def _segments(tr):
    """Path pieces of each track: list of (p, q, use) with p, q (n, 3) and use a mask.
    Muons: A -> B or A -> K -> K -> B (scattering only after K). Signal: vertex -> B."""
    b = np.stack([tr["bx"], tr["by"], tr["bz"]], 1).astype(float)
    if "ax" not in tr:                                       # signal: from the decay vertex
        v = np.stack([tr["vx"], tr["vy"], tr["vz"]], 1).astype(float)
        return [(v, b, np.ones(len(b), bool))], np.zeros(len(b), bool)
    a = np.stack([tr["ax"], tr["ay"], tr["az"]], 1).astype(float)
    k = np.stack([tr["kx"], tr["ky"], tr["kz"]], 1).astype(float)
    bounced = np.isfinite(k[:, 2])
    # A is stored up to 2 cm after z = 30.6: start 5 cm earlier so planes at 30.6 are crossed
    with np.errstate(invalid="ignore", divide="ignore"):
        a_ext = a - (b - a) * (mt.A_TOLERANCE / (b[:, 2] - a[:, 2]))[:, None]
    a_ext = np.where(np.isfinite(a_ext), a_ext, a)
    start = np.where(bounced[:, None], k, a_ext)            # bounced: only after the bounce
    return [(start, b, np.ones(len(b), bool))], bounced


def _at_z(p, q, z):
    """Point of segment p -> q at plane z (n,), NaN where z is outside the segment."""
    dz = q[:, 2] - p[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (z - p[:, 2]) / dz
    ok = (dz > 0) & (t >= 0) & (t <= 1)
    pt = p + t[:, None] * (q - p)
    pt[~ok] = np.nan
    return pt


def scatterers(tr, design, media=None):
    """All scatterers of every track: z (n, S), x/X0 (n, S), NaN / 0 where a scatterer is not crossed."""
    (segs, bounced) = _segments(tr)
    p, q, _ = segs[0]
    n = len(p)
    d = q - p
    with np.errstate(invalid="ignore", divide="ignore"):
        slope2 = (d[:, 0] ** 2 + d[:, 1] ** 2) / d[:, 2] ** 2
    path_per_dz = np.sqrt(1 + np.nan_to_num(slope2))           # path length per metre of z
    cos_z = 1 / path_per_dz                                    # incidence on planes of constant z
    zs, xs = [], []

    # UBT and the vessel entrance window, at z = 30.6 m (start of the unbounced muons)
    ubt = design.planes[0]
    pa = p
    near_a = np.abs(p[:, 2] - geo.Z_UBT) < 2 * mt.A_TOLERANCE
    in_ubt = near_a & (np.abs(pa[:, 0]) < geo.UBT_HALF_X) & (pa[:, 1] >= ubt.y_lo) & (pa[:, 1] <= ubt.y_hi)
    x_ubt = tech.x0(tech.TECHNOLOGIES[ubt.technology]) if ubt.technology else 0.0
    zs.append(np.where(in_ubt, geo.Z_UBT, np.nan)); xs.append(np.where(in_ubt, x_ubt / cos_z, 0.0))
    in_win = near_a & (np.abs(pa[:, 0]) < geo.VESSEL_X_IN) & (np.abs(pa[:, 1]) < geo.VESSEL_Y_IN)
    zs.append(np.where(in_win, geo.Z_UBT, np.nan)); xs.append(np.where(in_win, PVC_THICKNESS / X0_PVC / cos_z, 0.0))

    # fins: planes of constant z
    for plane in [pl for pl in design.planes[1:] if pl.kind == "fin"]:     # bottom panel: not modelled yet
        pt = _at_z(p, q, np.full(n, plane.z))
        u = plane.to_local(pt[:, 0])
        hit = np.isfinite(pt[:, 2]) & plane.inside(u, pt[:, 1])
        x_fin = tech.x0(tech.TECHNOLOGIES[plane.technology]) if plane.technology else 0.0
        zs.append(np.where(hit, plane.z, np.nan)); xs.append(np.where(hit, x_fin / cos_z, 0.0))

    # vessel lateral walls (entry / exit through the SBT walls) and the exit wall at z = 80.6 m
    c = vessel_crossings(p[:, 0], p[:, 1], p[:, 2], d[:, 0], d[:, 1], d[:, 2])
    lo, hi = np.minimum(p[:, 2], q[:, 2]), np.maximum(p[:, 2], q[:, 2])
    dn = d / np.linalg.norm(d, axis=1, keepdims=True)
    for face, zc in ((c["face_in"], c["z_in"]), (c["face_out"], c["z_out"])):
        on = (face >= 2) & (face <= 5) & (zc > lo) & (zc < hi)
        # wall normals: x walls (-+1, 0, -+slope_x), y walls likewise; |cos| with the track direction
        sx = (geo.VESSEL_X_OUT - geo.VESSEL_X_IN) / (geo.Z_VESSEL_MAX - geo.Z_VESSEL_MIN)
        sy = (geo.VESSEL_Y_OUT - geo.VESSEL_Y_IN) / (geo.Z_VESSEL_MAX - geo.Z_VESSEL_MIN)
        is_x = (face == 2) | (face == 3)
        nx = np.where(is_x, 1.0, 0.0); ny = np.where(is_x, 0.0, 1.0); nz = -np.where(is_x, sx, sy)
        norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
        cos_w = np.abs(np.abs(dn[:, 0]) * nx + np.abs(dn[:, 1]) * ny + dn[:, 2] * nz) / norm
        cos_w = np.maximum(cos_w, MIN_COS)
        zs.append(np.where(on, zc, np.nan)); xs.append(np.where(on, PVC_THICKNESS / X0_PVC / cos_w, 0.0))
    pe = _at_z(p, q, np.full(n, geo.Z_VESSEL_MAX))
    hx, hy = geo.vessel_half_widths(np.full(n, geo.Z_VESSEL_MAX))
    in_exit = np.isfinite(pe[:, 2]) & (np.abs(pe[:, 0]) < hx) & (np.abs(pe[:, 1]) < hy)
    zs.append(np.where(in_exit, geo.Z_VESSEL_MAX, np.nan)); xs.append(np.where(in_exit, PVC_THICKNESS / X0_PVC / cos_z, 0.0))

    # air / helium, sliced in z (off by default)
    slices = np.arange(geo.Z_UBT + DZ_SLICE / 2, geo.Z_SST1, DZ_SLICE) if (MEDIA if media is None else media) else []
    for zc in slices:
        pt = _at_z(p, q, np.full(n, zc))
        ok = np.isfinite(pt[:, 2])
        hx, hy = geo.vessel_half_widths(np.full(n, zc))
        inside = ok & (zc < geo.Z_VESSEL_MAX) & (np.abs(pt[:, 0]) < hx) & (np.abs(pt[:, 1]) < hy)
        x0 = np.where(inside, X0_HELIUM, X0_AIR)
        zs.append(np.where(ok, zc, np.nan)); xs.append(np.where(ok, DZ_SLICE * path_per_dz / x0, 0.0))
    return np.stack(zs, 1), np.stack(xs, 1)


def kicks(tr, design, rng, media=None):
    """Scattering of every track: dict with z, theta_x, theta_y (n, S) and the total x/X0 (n,)."""
    z, x = scatterers(tr, design, media)
    p = np.maximum(np.sqrt(tr["bpx"] ** 2 + tr["bpy"] ** 2 + tr["bpz"] ** 2), 0.05)   # at z = 82 m
    x_tot = x.sum(1)
    theta = highland(p, x_tot)
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma = theta[:, None] * np.sqrt(np.where(x_tot[:, None] > 0, x / x_tot[:, None], 0))
    tx = rng.normal(size=x.shape) * sigma
    ty = rng.normal(size=x.shape) * sigma
    return dict(z=z, tx=np.where(np.isfinite(z), tx, 0.0), ty=np.where(np.isfinite(z), ty, 0.0), x_total=x_tot)


def offset_at(k, z):
    """Displacement (dx, dy) of each track at plane z (scalar or (n,)) from the kicks upstream of it."""
    z = np.broadcast_to(np.asarray(z, float), (k["z"].shape[0],))[:, None]
    lever = np.where(np.isfinite(k["z"]) & (k["z"] < z), z - k["z"], 0.0)
    return (k["tx"] * lever).sum(1), (k["ty"] * lever).sum(1)


def scattered_tracks(tr, k):
    """Copy of the tracks with the state at z = 82 m scattered: position shifted, slopes turned."""
    out = dict(tr)
    dx, dy = offset_at(k, tr["bz"])
    up = np.isfinite(k["z"]) & (k["z"] < tr["bz"][:, None])
    dtx, dty = (k["tx"] * up).sum(1), (k["ty"] * up).sum(1)
    out["bx"], out["by"] = tr["bx"] + dx, tr["by"] + dy
    tx, ty = tr["bpx"] / tr["bpz"] + dtx, tr["bpy"] / tr["bpz"] + dty
    p = np.sqrt(tr["bpx"] ** 2 + tr["bpy"] ** 2 + tr["bpz"] ** 2)
    pz = p / np.sqrt(1 + tx ** 2 + ty ** 2)
    out["bpx"], out["bpy"], out["bpz"] = tx * pz, ty * pz, pz
    return out
