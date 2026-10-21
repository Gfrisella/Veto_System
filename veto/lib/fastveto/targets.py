"""What every muon path crosses between z = 30.6 m and z = 82 m: the interaction targets of one veto design.

A muon of the design-optimisation dataset (cuda-muons output, Veto_System column store) is the path A -> B, or
A -> K -> B when it bounced (veto.muon_types: A at z = 30.6, B at z = 82 m or where it stopped). Along it:

    helium          inside the vessel (Liang-Barsky chord, veto.tracks.vessel_crossings)
    air             the rest of the path (cavern air, 30.6 .. 82 m)
    pvc             entrance window, lateral walls, exit wall (0.3 mm): exact chords (fastveto.vessel.piece_intervals)
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
MIN_COS = 0.02                     # planes of constant z: grazing incidence capped (as veto.scattering)
M_MU = 0.10566


def target_name(code, design=None):
    if code == TARGET_HELIUM:
        return "helium"
    if code == TARGET_AIR:
        return "air"
    if code == TARGET_PVC:
        return "pvc"
    i = code - TARGET_PLANE0
    return design.planes[i].name if design is not None else f"plane {i}"


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
    # A is stored up to 2 cm after z = 30.6: start A_TOLERANCE earlier so the window / UBT at 30.6 are crossed
    first_end = np.where(np.isfinite(k[:, 2])[:, None], k, b)
    with np.errstate(invalid="ignore", divide="ignore"):
        a_ext = a - (first_end - a) * (mt.A_TOLERANCE / (first_end[:, 2] - a[:, 2]))[:, None]
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


def crossings(d, design, include=("helium", "air", "pvc", "planes")):
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

    he, air, pvc = mat.MATERIALS["helium"], mat.MATERIALS["air"], mat.MATERIALS["pvc"]
    n_he = he.nucleons_per_m2(1.0)                      # per metre of path
    n_air = air.nucleons_per_m2(1.0)
    n_pvc = pvc.nucleons_per_m2(1.0)                    # per metre of path
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
        if "air" in include:
            # outside the vessel: [p, z_lo] and [z_hi, q] (whole segment without a vessel chord)
            m1 = ~inside
            add(idx[m1], TARGET_AIR, "air", p[m1], q[m1], (n_air * length)[m1], m1)
            m2 = inside & (z_lo > p[:, 2] + 1e-6)
            add(idx[m2], TARGET_AIR, "air", p[m2], at(z_lo)[m2], (n_air * (z_lo - p[:, 2]) / cos_z)[m2], m2)
            m3 = inside & (z_hi < q[:, 2] - 1e-6)
            add(idx[m3], TARGET_AIR, "air", at(z_hi)[m3], q[m3], (n_air * (q[:, 2] - z_hi) / cos_z)[m3], m3)
        if "pvc" in include:
            # exact chords through the 0.3 mm shell (outer chord minus helium chord, fastveto.vessel): entrance window,
            # lateral walls (also grazing: a line nearly parallel to an inclined wall can stay in it for metres) and exit
            iv = V.piece_intervals(p, q)
            for part in ("pvc_in", "pvc_out"):
                a0, b0 = iv[part]
                L = b0 - a0
                m = L > 0
                add(idx[m], TARGET_PVC, "pvc", (p + dn * a0[:, None])[m], (p + dn * b0[:, None])[m], (n_pvc * L)[m], m)
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
