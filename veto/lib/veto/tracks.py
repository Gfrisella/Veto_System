"""Straight-line track extrapolation and muon categories."""

import numpy as np

from . import geometry as geo


def momentum(data):
    return np.sqrt(data["px"] ** 2 + data["py"] ** 2 + data["pz"] ** 2)


def extrapolate_to_z(x, y, z, px, py, pz, z_target):
    """(x, y) at z_target assuming straight-line motion."""
    dz = z_target - z
    with np.errstate(divide="ignore", invalid="ignore"):
        return x + px / pz * dz, y + py / pz * dz


def extrapolate(data, z_target):
    return extrapolate_to_z(data["x"], data["y"], data["z"],
                            data["px"], data["py"], data["pz"], z_target)


def backtrack_to_lateral(x0, y0, z0, px, py, pz):
    """Walk the track back from (x0, y0, z0) and find where it entered the vessel.

    Returns (z, x, y) of the entry point (UBT window or an SBT wall); NaN where the
    line never passes through the vessel.

    The original version in the notebooks intersected the infinite planes of the walls
    without checking that the point lies on the wall itself. For ~3 % of the SST1 tracks
    it returned points off the vessel surface. This one uses vessel_crossings().
    """
    c = vessel_crossings(x0, y0, z0, px, py, pz)
    before = c["z_in"] < np.asarray(z0)
    return (np.where(before, c["z_in"], np.nan), np.where(before, c["x_in"], np.nan),
            np.where(before, c["y_in"], np.nan))


FACE_NAMES = {0: "none", 1: "UBT (entrance)", 2: "SBT +x", 3: "SBT -x", 4: "SBT +y", 5: "SBT -y",
              6: "vessel end"}


def vessel_crossings(x, y, z, px, py, pz):
    """Where the straight line through (x, y, z) with direction p is inside the decay vessel.

    Works for any reference z (forward from the UBT plane or backward from SST1).
    Every vessel boundary is a half-space that is linear in z along the line, so the
    inside part is an interval [z_in, z_out] (Liang-Barsky clipping).

    Returns dict with z_in, z_out, face_in, face_out (FACE_NAMES codes, 0 = never inside)
    and x_in, y_in, x_out, y_out.
    """
    x, y, z, px, py, pz = (np.asarray(a, dtype=np.float64) for a in (x, y, z, px, py, pz))
    n = len(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        vx, vy = px / pz, py / pz
    ax = (geo.VESSEL_X_OUT - geo.VESSEL_X_IN) / (geo.Z_VESSEL_MAX - geo.Z_VESSEL_MIN)
    ay = (geo.VESSEL_Y_OUT - geo.VESSEL_Y_IN) / (geo.Z_VESSEL_MAX - geo.Z_VESSEL_MIN)
    x_at0 = x - vx * z          # line: x(z) = x_at0 + vx z
    y_at0 = y - vy * z
    x_w0 = geo.VESSEL_X_IN - ax * geo.Z_VESSEL_MIN   # wall: X(z) = x_w0 + ax z
    y_w0 = geo.VESSEL_Y_IN - ay * geo.Z_VESSEL_MIN

    lower = np.full(n, geo.Z_VESSEL_MIN)
    upper = np.full(n, geo.Z_VESSEL_MAX)
    face_in = np.full(n, 1, dtype=np.int8)
    face_out = np.full(n, 6, dtype=np.int8)
    feasible = np.isfinite(vx) & np.isfinite(vy)

    # each wall: c z <= d
    walls = (
        (2, vx - ax, x_w0 - x_at0),      # x <= +X(z)
        (3, -vx - ax, x_w0 + x_at0),     # x >= -X(z)
        (4, vy - ay, y_w0 - y_at0),      # y <= +Y(z)
        (5, -vy - ay, y_w0 + y_at0),     # y >= -Y(z)
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

    inside = feasible & (lower <= upper)
    z_in = np.where(inside, lower, np.nan)
    z_out = np.where(inside, upper, np.nan)
    face_in = np.where(inside, face_in, 0).astype(np.int8)
    face_out = np.where(inside, face_out, 0).astype(np.int8)
    return {
        "z_in": z_in, "x_in": x_at0 + vx * z_in, "y_in": y_at0 + vy * z_in, "face_in": face_in,
        "z_out": z_out, "x_out": x_at0 + vx * z_out, "y_out": y_at0 + vy * z_out, "face_out": face_out,
    }


def entry_wall(x_vessel, y_vessel, z_vessel, tolerance=1e-6):
    """Label the vessel face a track entered through.

    0 = none, 1 = UBT (entrance window), 2 = +x, 3 = -x, 4 = +y, 5 = -y (SBT walls).
    """
    label = np.zeros(len(z_vessel), dtype=np.int8)
    ok = np.isfinite(z_vessel)
    half_x, half_y = geo.vessel_half_widths(z_vessel)
    label[ok & (np.abs(z_vessel - geo.Z_VESSEL_MIN) < tolerance)] = 1
    lateral = ok & (label == 0)
    on_x = lateral & (np.abs(np.abs(x_vessel) - half_x) < 1e-4)
    on_y = lateral & ~on_x
    label[on_x & (x_vessel > 0)] = 2
    label[on_x & (x_vessel < 0)] = 3
    label[on_y & (y_vessel > 0)] = 4
    label[on_y & (y_vessel < 0)] = 5
    return label


WALL_NAMES = {0: "no vessel crossing", 1: "UBT (entrance)", 2: "SBT +x", 3: "SBT -x", 4: "SBT +y", 5: "SBT -y"}


def categories(x_ubt, y_ubt, pdg):
    """Muon categories defined in analysis.ipynb / make_vessel_dataset.

    bouncing   : back-projection at the UBT plane lands outside the cavern (|x| > 3 or y < -3.36)
    front      : back-projection passes through the UBT aperture
    cross_over : x at the UBT plane has the sign of the pdg code (crossed the beam axis in the shield)
    """
    return {
        "bouncing": (np.abs(x_ubt) > 3) | (y_ubt < geo.CAVERN_Y_FLOOR),
        "front": (np.abs(x_ubt) < geo.UBT_HALF_X) & (np.abs(y_ubt) < geo.UBT_HALF_Y),
        "cross_over": (x_ubt * pdg) > 0,
    }
