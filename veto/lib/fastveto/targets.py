"""What every muon path crosses between z = 30.6 m and z = 82 m: the interaction targets of one veto design.

A muon of the design-optimisation dataset (cuda-muons output, Veto_System column store) is the path A -> B, or
A -> K -> B when it bounced (veto.muon_types: A at z = 30.6, B at z = 82 m or where it stopped). Along it:

    helium          inside the vessel (Liang-Barsky chord, veto.tracks.vessel_crossings)
    air             the rest of the path (cavern air, 30.6 .. 82 m); since 2026-11-05 (ISSUES C7) only the air within
                    AIR_MARGIN = 0.4 m of the vessel's outer surface or of a fin (slab |z - z_fin| < AIR_MARGIN around
                    the fin, its extent grown by AIR_MARGIN): the air far from the vessel cannot give a vertex inside
                    it, and the draws now go where it can. AIR_MARGIN = None: all the air, as before.
    pvc             vessel wall (historical key): entrance window, lateral walls, exit wall, 350 um Mylar
                    (fastveto.vessel.WALL_MATERIAL / WALL_THICKNESS): exact chords (fastveto.vessel.piece_intervals)
    plane i         every plane of the design (UBT and fins) where crossed, material of its technology
                    (fastveto.materials.tech_slab; thickness / |cos incidence| for planes of constant z)

One entry per crossed target: muon index, target code (TARGET_HELIUM, TARGET_AIR, TARGET_PVC, TARGET_PLANE0 + i),
material name, the piece of path (points P0 -> P1; P0 = P1 for thin targets), nucleons per m^2 along the path and
the muon energy there (|p| interpolated in z between A and B).
"""
import numpy as np

from veto import geometry as geo
from veto import muon_types as mt
from veto import technology as tech  # noqa: F401
from veto.tracks import vessel_crossings

from . import materials as mat
from . import vessel as V

TARGET_HELIUM, TARGET_AIR, TARGET_PVC, TARGET_PLANE0 = 0, 1, 2, 10
TARGET_WALL = TARGET_PVC     # ISSUES D8 (11 Nov 2026): the vessel-wall target group is "wall" (Mylar); "pvc" accepted as an alias
WALL_ALIASES = ("wall", "pvc")
TARGET_AIR_UP = 8            # C7 / D7 (12 Nov 2026): the air between the shield end and the vessel entrance, its own group
                             # (own number of draws: it is statistically empty as part of "air"); 8: free in CODE_NAMES
AIR_MARGIN = 0.4                   # m, air kept around the vessel and the fins (None = all the air)
Z_SHIELD_END = geo.Z_SHIELD_END   # m, TRY6 shield end
UPSTREAM_AIR = True                # D7 (2026-11-05): paths start at the shield end (straight back from A, no field there),
                                   # so the 1.24 m of air before the vessel is a target (only air lies there)
MIN_COS = 0.02                     # planes of constant z: grazing incidence capped (as veto.scattering)
M_MU = 0.10566


def target_name(code, design=None):
    if code == TARGET_HELIUM:
        return "helium"
    if code == TARGET_AIR:
        return "air"
    if code == TARGET_AIR_UP:
        return "air_upstream"
    if code == TARGET_PVC:
        return "wall"
    i = code - TARGET_PLANE0
    return design.planes[i].name if design is not None else f"plane {i}"


def target_material(code):
    """fastveto.materials name of a design-free target (helium, air, vessel wall)."""
    if code == TARGET_PVC:
        return V.WALL_MATERIAL
    if code == TARGET_AIR_UP:
        return "air"
    return target_name(code)


def plane_material(plane):
    """(material name, nucleons per m^2 at normal incidence) of a design plane."""
    if plane.technology is None:
        return None, 0.0
    s = mat.tech_slab(plane.technology, plane.x0)       # x/X0 of the whole plane, all its stations
    return s.material.name, s.nucleons_per_m2


def paths(d):
    """Segments of every muon: list of (P, Q, muon index) with P, Q (k, 3) and dz > 0.
    d: columns x_30 .. pz_82, kz (NaN without bounce). Muons that never reached z = 30.6 or did not move on are left out."""
    a = np.stack([d["x_30"], d["y_30"], d["z_30"]], 1).astype(np.float64)
    b = np.stack([d["x_82"], d["y_82"], d["z_82"]], 1).astype(np.float64)
    k = np.stack([d["kx"], d["ky"], d["kz"]], 1).astype(np.float64)
    use = (a[:, 2] > mt.REACH_30) & (b[:, 2] > a[:, 2])
    # A is stored up to 2 cm after z = 30.6: start A_TOLERANCE earlier; with UPSTREAM_AIR (D7) start at the shield end
    first_end = np.where(np.isfinite(k[:, 2])[:, None], k, b)
    back = (a[:, 2] - Z_SHIELD_END) if UPSTREAM_AIR else np.full(len(a), mt.A_TOLERANCE)
    with np.errstate(invalid="ignore", divide="ignore"):
        a_ext = a - (first_end - a) * (back / (first_end[:, 2] - a[:, 2]))[:, None]
    a_ext = np.where(np.isfinite(a_ext), a_ext, a)
    idx = np.nonzero(use)[0]
    out = [(a_ext[idx], first_end[idx], idx)]
    bounced = use & np.isfinite(k[:, 2])
    j = np.nonzero(bounced)[0]
    if len(j):
        out.append((k[j], b[j], j))
    return out


def momentum_at(d, idx, z):
    """|p| at z: linear in z between A and B (energy loss in 50 m of air / helium is small)."""
    pa = np.sqrt(d["px_30"][idx] ** 2 + d["py_30"][idx] ** 2 + d["pz_30"][idx] ** 2).astype(np.float64)
    pb = np.sqrt(d["px_82"][idx] ** 2 + d["py_82"][idx] ** 2 + d["pz_82"][idx] ** 2).astype(np.float64)
    za, zb = d["z_30"][idx].astype(np.float64), d["z_82"][idx].astype(np.float64)
    f = np.clip((z - za) / np.maximum(zb - za, 1e-9), 0, 1)
    return pa + f * (pb - pa)


def _near_air_tests(design, margin):
    """Membership tests f(points (k, 3)) -> bool of the kept air region: vessel grown by margin (minus the vessel
    itself, tested separately) and a box around every fin."""
    z0, z1, hx0, hx1, hy0, hy1 = V.OUTER_BOX
    sx, sy = (hx1 - hx0) / (z1 - z0), (hy1 - hy0) / (z1 - z0)

    def near_vessel(pt):
        x, y, z = pt[:, 0], pt[:, 1], pt[:, 2]
        hx, hy = hx0 + sx * (z - z0), hy0 + sy * (z - z0)
        return (z > z0 - margin) & (z < z1 + margin) & (np.abs(x) < hx + margin) & (np.abs(y) < hy + margin)

    def upstream(pt):                      # D7: shield end -> vessel entrance, around the window grown by margin
        return ((pt[:, 2] >= Z_SHIELD_END - 1e-9) & (pt[:, 2] <= z0) & (np.abs(pt[:, 0]) < hx0 + margin)
                & (np.abs(pt[:, 1]) < hy0 + margin))

    tests = [near_vessel] + ([upstream] if UPSTREAM_AIR else [])
    for plane in (design.planes if design is not None else []):
        if plane.kind != "fin":
            continue
        def near_fin(pt, plane=plane):
            u = plane.to_local(pt[:, 0])
            return ((np.abs(pt[:, 2] - plane.z) < margin) & (u > -margin) & (u < plane.u_max + margin)
                    & (pt[:, 1] > plane.y_lo - margin) & (pt[:, 1] < plane.y_hi + margin))
        tests.append(near_fin)
    return tests


def _near_air_pieces(p, q, design, margin):
    """Air pieces of the segments p -> q inside the kept region and outside the vessel's outer surface.
    Returns (segment index, t0, t1) with t the fraction along the segment. Exact for the region's planar faces:
    every face is crossed at most once along a segment, so between consecutive face crossings membership is
    constant and tested at the midpoint."""
    z0, z1, hx0, hx1, hy0, hy1 = V.OUTER_BOX
    n = len(p)
    d = q - p
    ts = [np.zeros(n), np.ones(n)]
    with np.errstate(invalid="ignore", divide="ignore"):
        def add_z(z):
            ts.append((z - p[:, 2]) / d[:, 2])
        for zz in (z0 - margin, z1 + margin, z0, z1, Z_SHIELD_END):
            add_z(zz)
        # lateral faces of the grown and the outer frustum: x = +-(X(z) + m), y likewise, linear in t
        for m in (margin, 0.0):
            for c, h0, h1 in ((0, hx0, hx1), (1, hy0, hy1)):
                s_ = (h1 - h0) / (z1 - z0)
                for sign in (1, -1):
                    # sign * (p_c + t d_c) = h0 + m + s (p_z + t d_z - z0)
                    num = h0 + m + s_ * (p[:, 2] - z0) - sign * p[:, c]
                    den = sign * d[:, c] - s_ * d[:, 2]
                    ts.append(num / den)
        for plane in (design.planes if design is not None else []):
            if plane.kind != "fin":
                continue
            for zz in (plane.z - margin, plane.z + margin):
                add_z(zz)
            for yy in (plane.y_lo - margin, plane.y_hi + margin):
                ts.append((yy - p[:, 1]) / d[:, 1])
            # u = sign x - X(z_fin): x = sign (u + X(z_fin))
            xw = plane.to_local(np.zeros(1))[0]            # u at x = 0  ->  u = sign x + xw
            for uu in (-margin, plane.u_max + margin):
                ts.append(((uu - xw) / plane.sign - p[:, 0]) / d[:, 0])
    T = np.stack(ts, 1)
    T = np.where(np.isfinite(T), np.clip(T, 0, 1), 1.0)
    T.sort(axis=1)
    t0, t1 = T[:, :-1], T[:, 1:]
    mid = 0.5 * (t0 + t1)
    seg = np.broadcast_to(np.arange(n)[:, None], mid.shape)
    long = (t1 - t0) > 1e-9
    seg, t0, t1, mid = seg[long], t0[long], t1[long], mid[long]
    pt = p[seg] + mid[:, None] * d[seg]
    inside_vessel = V.inside(pt[:, 0], pt[:, 1], pt[:, 2]) | _in_outer(pt)
    keep = np.zeros(len(pt), bool)
    for f in _near_air_tests(design, margin):
        keep |= f(pt)
    keep &= ~inside_vessel
    return seg[keep], t0[keep], t1[keep]


def _in_outer(pt):
    z0, z1, hx0, hx1, hy0, hy1 = V.OUTER_BOX
    x, y, z = pt[:, 0], pt[:, 1], pt[:, 2]
    hx, hy = hx0 + (hx1 - hx0) / (z1 - z0) * (z - z0), hy0 + (hy1 - hy0) / (z1 - z0) * (z - z0)
    return (z >= z0) & (z <= z1) & (np.abs(x) <= hx) & (np.abs(y) <= hy)


def crossings(d, design, include=("helium", "air", "wall", "planes")):
    """All target crossings of the muons in d (dict of columns). Returns dict of flat arrays (one row per crossing):
    muon, code, material (index into MATERIAL_NAMES), p0 (k,3), p1 (k,3), u (k,3) muon direction, n_area [nucleons / m^2],
    E [GeV]."""
    rows = {k: [] for k in ("muon", "code", "material", "p0", "p1", "n_area", "z", "u")}
    cur = {}

    def add(muon, code, material, p0, p1, n_area, m):
        if not len(muon):
            return
        rows["muon"].append(muon)
        rows["code"].append(np.full(len(muon), code, np.int16))
        rows["material"].append(np.full(len(muon), MATERIAL_NAMES.index(material), np.int8))
        rows["p0"].append(p0); rows["p1"].append(p1); rows["n_area"].append(n_area)
        rows["z"].append(0.5 * (p0[:, 2] + p1[:, 2]))
        rows["u"].append(cur["dn"][m])

    he, air, wall = mat.MATERIALS["helium"], mat.MATERIALS["air"], mat.MATERIALS[V.WALL_MATERIAL]
    n_he = he.nucleons_per_m2(1.0)                      # per metre of path
    n_air = air.nucleons_per_m2(1.0)
    n_wall = wall.nucleons_per_m2(1.0)                  # per metre of path
    planes = [(i, p) + plane_material(p) for i, p in enumerate(design.planes)] if design is not None else []

    for p, q, idx in paths(d):
        good = np.isfinite(p).all(1) & np.isfinite(q).all(1) & (q[:, 2] - p[:, 2] > 1e-9)
        p, q, idx = p[good], q[good], idx[good]
        dvec = q - p
        length = np.linalg.norm(dvec, axis=1)
        dn = dvec / length[:, None]
        cur["dn"] = dn
        cos_z = np.maximum(dn[:, 2], MIN_COS)
        c = vessel_crossings(p[:, 0], p[:, 1], p[:, 2], dvec[:, 0], dvec[:, 1], dvec[:, 2])
        z_lo = np.maximum(c["z_in"], p[:, 2])
        z_hi = np.minimum(c["z_out"], q[:, 2])
        inside = np.isfinite(z_lo) & (z_hi > z_lo)
        at = lambda z: p + ((z - p[:, 2]) / dvec[:, 2])[:, None] * dvec
        if "helium" in include:
            m = inside
            add(idx[m], TARGET_HELIUM, "helium", at(z_lo)[m], at(z_hi)[m], (n_he * (z_hi - z_lo) / cos_z)[m], m)
        if "air" in include and AIR_MARGIN is None:
            # outside the vessel: [p, z_lo] and [z_hi, q] (whole segment without a vessel chord)
            m1 = ~inside
            add(idx[m1], TARGET_AIR, "air", p[m1], q[m1], (n_air * length)[m1], m1)
            m2 = inside & (z_lo > p[:, 2] + 1e-6)
            add(idx[m2], TARGET_AIR, "air", p[m2], at(z_lo)[m2], (n_air * (z_lo - p[:, 2]) / cos_z)[m2], m2)
            m3 = inside & (z_hi < q[:, 2] - 1e-6)
            add(idx[m3], TARGET_AIR, "air", at(z_hi)[m3], q[m3], (n_air * (q[:, 2] - z_hi) / cos_z)[m3], m3)
        elif "air" in include:
            # only the air near the vessel and the fins (C7); one row per piece
            seg, t0, t1 = _near_air_pieces(p, q, design, AIR_MARGIN)
            zmid = p[seg, 2] + 0.5 * (t0 + t1) * dvec[seg, 2]
            up = zmid < V.OUTER_BOX[0]                     # upstream of the vessel entrance: its own group (C7 / D7)
            cur_dn = cur["dn"]
            for code, m in ((TARGET_AIR, ~up), (TARGET_AIR_UP, up)):
                s_ = seg[m]
                cur["dn"] = cur_dn[s_]
                add(idx[s_], code, "air", p[s_] + t0[m][:, None] * dvec[s_], p[s_] + t1[m][:, None] * dvec[s_],
                    n_air * (t1[m] - t0[m]) * length[s_], slice(None))
            cur["dn"] = cur_dn
        if any(k in include for k in WALL_ALIASES):
            # exact chords through the wall shell (outer chord minus helium chord, fastveto.vessel): entrance window,
            # lateral walls (also grazing: a line nearly parallel to an inclined wall can stay in it for metres) and exit
            iv = V.piece_intervals(p, q)
            for part in ("pvc_in", "pvc_out"):
                a0, b0 = iv[part]
                L = b0 - a0
                m = L > 0
                add(idx[m], TARGET_PVC, V.WALL_MATERIAL, (p + dn * a0[:, None])[m], (p + dn * b0[:, None])[m], (n_wall * L)[m], m)
        if "planes" in include:
            for i, plane, mname, n_area in planes:
                if mname is None:
                    continue
                with np.errstate(invalid="ignore", divide="ignore"):
                    t = (plane.z - p[:, 2]) / dvec[:, 2]
                pt = p + t[:, None] * dvec
                hit = (t >= 0) & (t <= 1) & plane.inside(plane.to_local(pt[:, 0]), pt[:, 1])
                add(idx[hit], TARGET_PLANE0 + i, mname, pt[hit], pt[hit], (n_area / cos_z)[hit], hit)
    if not rows["muon"]:
        return {k: np.zeros(0) for k in rows} | {"E": np.zeros(0)}
    out = {k: np.concatenate(v) for k, v in rows.items()}
    out["E"] = np.sqrt(momentum_at(d, out["muon"], out["z"]) ** 2 + M_MU ** 2)
    return out


MATERIAL_NAMES = ["helium", "air", "pvc", "mylar", "polystyrene", "silicon", "pvt"]


def sample_vertex(cr, rows, rng):
    """Interaction point for crossing rows: uniform along P0 -> P1 (P(int) << 1 everywhere)."""
    u = rng.random(len(rows))[:, None]
    return cr["p0"][rows] + u * (cr["p1"][rows] - cr["p0"][rows])
