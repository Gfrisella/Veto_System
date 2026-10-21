"""SHiP decay vessel (helium balloon), the volume used for neutrino DIS.

Copied from Veto_System, branch veto-muon-production, commit a8dcc3c:
    veto/lib/veto/geometry.py    Z_VESSEL_*, VESSEL_*_IN/OUT, vessel_half_widths, inside_vessel
    veto/lib/veto/tracks.py      vessel_crossings (Liang-Barsky clipping of a line)
    veto/lib/veto/scattering.py  PVC wall thickness, X0 of helium / air / PVC
    veto/ISSUES.txt B1           material budget (SR-370 balloon)

All lengths in metres, z along the beam (target at z ~ -2.14 m, see Veto_System geometry.py).

The vessel is a rectangular frustum: half widths grow linearly in z from (0.5, 1.35) m at the
entrance (z = 30.6 m, = UBT aperture) to (2.0, 3.0) m at the exit (z = 80.6 m). It is filled with
helium; the entrance and exit windows and the lateral walls are 0.3 mm PVC. Outside it (and
between the exit and SST1 at z = 82 m) is air.
"""

import numpy as np

Z_VESSEL_MIN = 30.6
Z_VESSEL_MAX = 80.6
VESSEL_X_IN, VESSEL_X_OUT = 0.5, 2.0     # half widths at entrance / exit
VESSEL_Y_IN, VESSEL_Y_OUT = 1.35, 3.0
LENGTH = Z_VESSEL_MAX - Z_VESSEL_MIN

# slopes of the walls: X(z) = X_IN + SLOPE_X (z - Z_VESSEL_MIN)
SLOPE_X = (VESSEL_X_OUT - VESSEL_X_IN) / LENGTH
SLOPE_Y = (VESSEL_Y_OUT - VESSEL_Y_IN) / LENGTH

FACE_NAMES = {0: "never inside", 1: "entrance", 2: "+x wall", 3: "-x wall",
              4: "+y wall", 5: "-y wall", 6: "exit"}

# ------------------------------------------------------------
# Media and walls. X0 from Veto_System (PDG). Densities are NOT in Veto_System:
# PDG values; helium / air ASSUMED at 1 atm, 20 C (balloon pressure / temperature to be confirmed).
# ------------------------------------------------------------

MEDIA = {
    #           X0 [m]   density [kg/m^3]   Z, A (mean, for nucleon counting)
    "helium": {"X0": 5671.0, "density": 0.1664, "Z": 2.0, "A": 4.0026},
    "air":    {"X0": 303.9,  "density": 1.205,  "Z": 7.31, "A": 14.61},   # ~ N2/O2/Ar mix
    "pvc":    {"X0": 0.1963, "density": 1380.0, "Z": None, "A": None},    # C2H3Cl, see pvc_composition
}
PVC_THICKNESS = 0.3e-3     # m, entrance / exit windows and lateral walls
# PVC (C2H3Cl)_n by mass fraction, for target-nucleus composition
PVC_COMPOSITION = {"C": 0.3844, "H": 0.0484, "Cl": 0.5672}


def half_widths(z):
    """Half widths (x, y) of the vessel at z (linear in z; extrapolated outside the z range)."""
    dz = np.asarray(z) - Z_VESSEL_MIN
    return VESSEL_X_IN + SLOPE_X * dz, VESSEL_Y_IN + SLOPE_Y * dz


def inside(x, y, z):
    """True where (x, y, z) is inside the helium volume."""
    hx, hy = half_widths(z)
    return (z >= Z_VESSEL_MIN) & (z <= Z_VESSEL_MAX) & (np.abs(x) <= hx) & (np.abs(y) <= hy)


def volume():
    """Helium volume [m^3]: integral of 4 hx(z) hy(z) dz, exact for linear walls (585 m^3)."""
    dx, dy = VESSEL_X_OUT - VESSEL_X_IN, VESSEL_Y_OUT - VESSEL_Y_IN
    return 4 * LENGTH * (VESSEL_X_IN * VESSEL_Y_IN + (VESSEL_X_IN * dy + VESSEL_Y_IN * dx) / 2
                         + dx * dy / 3)


def cross_section_area(z):
    hx, hy = half_widths(z)
    return 4 * hx * hy


def wall_area():
    """PVC surface [m^2]: entrance + exit windows + the four (trapezoidal, inclined) lateral walls."""
    windows = 4 * (VESSEL_X_IN * VESSEL_Y_IN + VESSEL_X_OUT * VESSEL_Y_OUT)
    # +-x walls: trapezoid of heights 2*Y_IN .. 2*Y_OUT, length along the slope
    lx = LENGTH * np.hypot(1.0, SLOPE_X)
    ly = LENGTH * np.hypot(1.0, SLOPE_Y)
    x_walls = 2 * lx * (VESSEL_Y_IN + VESSEL_Y_OUT)
    y_walls = 2 * ly * (VESSEL_X_IN + VESSEL_X_OUT)
    return windows + x_walls + y_walls


HELIUM_BOX = (Z_VESSEL_MIN, Z_VESSEL_MAX, VESSEL_X_IN, VESSEL_X_OUT, VESSEL_Y_IN, VESSEL_Y_OUT)
# outer surface of the PVC shell, as in cpp/nuvessel.cc (G4Trd with every half size + thickness)
OUTER_BOX = (Z_VESSEL_MIN - PVC_THICKNESS, Z_VESSEL_MAX + PVC_THICKNESS, VESSEL_X_IN + PVC_THICKNESS,
             VESSEL_X_OUT + PVC_THICKNESS, VESSEL_Y_IN + PVC_THICKNESS, VESSEL_Y_OUT + PVC_THICKNESS)


def crossings(x, y, z, px, py, pz, box=HELIUM_BOX):
    """Where the straight line through (x, y, z) with direction (px, py, pz) is inside the vessel
    (box = (z0, z1, hx0, hx1, hy0, hy1), half widths linear in z between z0 and z1; default the helium).

    Same as Veto_System tracks.vessel_crossings: every boundary is a half-space linear in z along
    the line, so the inside part is one interval [z_in, z_out] (Liang-Barsky). Needs pz != 0.

    Returns dict with z_in, z_out, x_in, y_in, x_out, y_out (nan if never inside) and
    face_in, face_out (FACE_NAMES codes, 0 = never inside).
    """
    x, y, z, px, py, pz = (np.asarray(a, dtype=np.float64) for a in (x, y, z, px, py, pz))
    n = len(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        vx, vy = px / pz, py / pz
    z0, z1, hx0, hx1, hy0, hy1 = box
    sx, sy = (hx1 - hx0) / (z1 - z0), (hy1 - hy0) / (z1 - z0)
    x_at0 = x - vx * z          # line: x(z) = x_at0 + vx z
    y_at0 = y - vy * z
    x_w0 = hx0 - sx * z0        # wall: X(z) = x_w0 + sx z
    y_w0 = hy0 - sy * z0

    lower = np.full(n, float(z0))
    upper = np.full(n, float(z1))
    face_in = np.full(n, 1, dtype=np.int8)
    face_out = np.full(n, 6, dtype=np.int8)
    feasible = np.isfinite(vx) & np.isfinite(vy)

    # each wall: c z <= d
    walls = (
        (2, vx - sx, x_w0 - x_at0),      # x <= +X(z)
        (3, -vx - sx, x_w0 + x_at0),     # x >= -X(z)
        (4, vy - sy, y_w0 - y_at0),      # y <= +Y(z)
        (5, -vy - sy, y_w0 + y_at0),     # y >= -Y(z)
    )
    for face, c, d in walls:
        with np.errstate(divide="ignore", invalid="ignore"):
            bound = d / c
        is_upper = c > 0
        is_lower = c < 0
        feasible &= ~((c == 0) & (d < 0))
        tighter = is_upper & (bound < upper)
        upper = np.where(tighter, bound, upper)
        face_out = np.where(tighter, face, face_out)
        tighter = is_lower & (bound > lower)
        lower = np.where(tighter, bound, lower)
        face_in = np.where(tighter, face, face_in)

    ok = feasible & (lower <= upper)
    z_in = np.where(ok, lower, np.nan)
    z_out = np.where(ok, upper, np.nan)
    return {
        "z_in": z_in, "x_in": x_at0 + vx * z_in, "y_in": y_at0 + vy * z_in,
        "face_in": np.where(ok, face_in, 0).astype(np.int8),
        "z_out": z_out, "x_out": x_at0 + vx * z_out, "y_out": y_at0 + vy * z_out,
        "face_out": np.where(ok, face_out, 0).astype(np.int8),
    }


def segments(x, y, z, px, py, pz):
    """Exact helium and PVC path pieces of each straight line (convex helium inside the convex outer
    shell surface, so along a line: PVC = outer chord minus helium chord).

    Returns dict with unit direction u and, as points (n, 3) along the line:
      o_in, o_out   outer-surface entry / exit (nan if the line misses the vessel)
      h_in, h_out   helium entry / exit (nan if the line misses the helium)
    and lengths L_he, L_pvc_in (o_in -> h_in), L_pvc_out (h_out -> o_out). A line that clips only the
    shell has L_he = 0 and its whole outer chord in L_pvc_in. Lines start at their own (x, y, z); a
    start inside the vessel is not handled (the neutrinos start at the target).
    """
    p = np.stack([np.asarray(a, np.float64) for a in (px, py, pz)], 1)
    u = p / np.linalg.norm(p, axis=1, keepdims=True)
    start = np.stack([np.asarray(a, np.float64) for a in (x, y, z)], 1)
    co = crossings(x, y, z, px, py, pz, OUTER_BOX)
    ch = crossings(x, y, z, px, py, pz, HELIUM_BOX)

    def point(zz):
        return start + u * ((zz - start[:, 2]) / u[:, 2])[:, None]

    o_in, o_out, h_in, h_out = point(co["z_in"]), point(co["z_out"]), point(ch["z_in"]), point(ch["z_out"])
    dist = lambda a, b: np.nan_to_num(np.linalg.norm(b - a, axis=1), nan=0.0)
    hit_he = ch["face_in"] > 0
    L_outer = dist(o_in, o_out)
    L_he = np.where(hit_he, dist(h_in, h_out), 0.0)
    L_in = np.where(hit_he, dist(o_in, h_in), L_outer)
    L_out = np.where(hit_he, dist(h_out, o_out), 0.0)
    return {"u": u, "o_in": o_in, "o_out": o_out, "h_in": h_in, "h_out": h_out,
            "L_he": L_he, "L_pvc_in": L_in, "L_pvc_out": L_out,
            "face_in": ch["face_in"], "face_out": ch["face_out"], "hits_vessel": co["face_in"] > 0}


def piece_intervals(P, Q):
    """Material intervals of the straight pieces P -> Q (arrays (n, 3), metres), clipped to the piece.

    Returns u (unit direction), L (piece length) and, for "pvc_in", "helium", "pvc_out", the parameter
    interval [a, b] along u from P (b - a = path length in that material, 0 if none). pvc_in is the shell
    from the outer surface to the helium (or the whole shell chord if the line misses the helium), pvc_out
    from the helium exit to the outer surface. A point inside: P + u * (a + r (b - a)), r uniform.
    """
    P, Q = np.asarray(P, np.float64), np.asarray(Q, np.float64)
    d = Q - P
    L = np.linalg.norm(d, axis=1)
    u = d / np.where(L > 0, L, 1)[:, None]
    co = crossings(P[:, 0], P[:, 1], P[:, 2], u[:, 0], u[:, 1], u[:, 2], OUTER_BOX)
    ch = crossings(P[:, 0], P[:, 1], P[:, 2], u[:, 0], u[:, 1], u[:, 2], HELIUM_BOX)
    with np.errstate(invalid="ignore", divide="ignore"):
        to = lambda zz: (zz - P[:, 2]) / u[:, 2]          # line parameter of a z on the line
        oi, oo, hi, ho = to(co["z_in"]), to(co["z_out"]), to(ch["z_in"]), to(ch["z_out"])
    hit_o = (co["face_in"] > 0) & (L > 0)
    hit_h = (ch["face_in"] > 0) & (L > 0)

    def clip(a, b, ok):
        a2 = np.clip(np.nan_to_num(a), 0, L)
        b2 = np.clip(np.nan_to_num(b), 0, L)
        b2 = np.where(ok, np.maximum(b2, a2), a2)
        return np.where(ok, a2, 0.0), np.where(ok, b2, 0.0)

    out = {"u": u, "L": L}
    out["helium"] = clip(hi, ho, hit_h)
    out["pvc_in"] = clip(oi, np.where(hit_h, hi, oo), hit_o)
    out["pvc_out"] = clip(ho, oo, hit_h)
    return out


def path_length(x, y, z, px, py, pz, box=HELIUM_BOX):
    """3D path length [m] of the line inside the helium, or another box (0 if it misses it)."""
    c = crossings(x, y, z, px, py, pz, box)
    px, py, pz = (np.asarray(a, dtype=np.float64) for a in (px, py, pz))
    stretch = np.sqrt(px ** 2 + py ** 2 + pz ** 2) / np.abs(pz)
    return np.nan_to_num((c["z_out"] - c["z_in"]) * stretch, nan=0.0)
