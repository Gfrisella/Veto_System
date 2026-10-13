"""SHiP detector geometry used by the veto analysis.

All lengths are in metres, z along the beam. Values come from the original
analysis scripts and from MuonsAndMatter
(muons_and_matter/lib/ship_muon_shield.py, reference_designs/params.py).
"""

import numpy as np

# ------------------------------------------------------------
# Reference planes
# ------------------------------------------------------------

Z_TARGET = -2.14            # pointing (IP) plane used for the vertex IP cut
Z_START = -0.0045           # muons start here (full_sample_after_target.h5); output planes[0] has no z
Z_UBT = 30.6                # Upstream Background Tagger = output planes[1]
Z_SST1 = 82.0               # first straw station = output planes[2]
Z_TIMING = 94.850 - 2.14    # timing detector

SST1_HALF_X, SST1_HALF_Y = 2.0, 3.0
TIMING_HALF_X, TIMING_HALF_Y = 2.0, 3.0

# ------------------------------------------------------------
# Decay vessel: rectangular frustum whose lateral walls carry the SBT
# ------------------------------------------------------------

Z_VESSEL_MIN = 30.6
Z_VESSEL_MAX = 80.6
VESSEL_X_IN, VESSEL_X_OUT = 0.5, 2.0     # half widths at entrance / exit
VESSEL_Y_IN, VESSEL_Y_OUT = 1.35, 3.0

# UBT aperture = vessel entrance window
UBT_HALF_X, UBT_HALF_Y = VESSEL_X_IN, VESSEL_Y_IN

# ------------------------------------------------------------
# Cavern walls seen by "bouncing" muons (from Muon_Flames_rate.py / vessel dataset)
# ------------------------------------------------------------

CAVERN_X_WALL = -4.56
CAVERN_Y_FLOOR = -3.36

# ------------------------------------------------------------
# Spectrometer magnet: region where |B| > 10 % of max in MainSpectrometerField.h5
# ------------------------------------------------------------

SPECTROMETER_Z = (84.36, 89.56)
SPECTROMETER_HALF_X = 2.7
SPECTROMETER_HALF_Y = 3.8
FIELD_MAP_Z = (77.06, 96.86)

# ------------------------------------------------------------
# Muon shield TRY6 (cm, MuonsAndMatter column order):
# zgap, dZ, dXIn, dXOut, dYIn, dYOut, gapIn, gapOut,
# x_yokeIn, x_yokeOut, dY_yokeIn, dY_yokeOut, midGapIn, midGapOut, NI
# ------------------------------------------------------------

SHIELD_TRY6 = np.array([
    [0.0, 115.0, 40.0, 40.0, 119.0, 119.0, 61.5, 61.5, 61.5, 61.5, 50.0, 50.0, 0.0, 0.0, 1.9875],
    [15, 150, 62, 62, 11, 11, 12, 12, 62, 62.0, 69, 69, 0, 0, 1.875],
    [15, 225, 70, 70, 12, 12, 9, 9, 70, 70.0, 77, 77, 0, 0, 1.87],
    [16, 225, 64, 64, 14, 14, 8, 8, 83, 83, 71, 71, 0, 0, 1.9],
    [15, 225, 42, 42, 13, 13, 8, 8, 109, 109, 47, 47, 0, 0, 1.89],
    [16, 131, 20, 20, 30, 30, 0, 0, 110.0, 110.0, 22, 22, 0, 0, 0],
    [15, 168, 20, 30, 30, 30, 22, 12, 108.0, 108.0, 33, 33, 0, 0, -1.32],
    [19, 200, 38, 69, 51, 51, 8, 8, 125, 94, 76, 76, 0, 0, -2.03],
])


def shield_magnets(params=SHIELD_TRY6):
    """Outer envelope of each shield magnet, mirroring design_muon_shield()."""
    magnets = []
    z = 0.0
    for row in params:
        zgap, dz, dx_in, dx_out, dy_in, dy_out, gap_in, gap_out, \
            xy_in, xy_out, dyy_in, dyy_out, mid_in, mid_out, ni = row
        if dz < 1 or dx_in < 1:
            z += dz + zgap
            continue
        z += zgap
        magnets.append({
            "z_start": z / 100,
            "z_end": (z + 2 * dz) / 100,
            "half_x_in": (dx_in + gap_in + xy_in + mid_in) / 100,
            "half_x_out": (dx_out + gap_out + xy_out + mid_out) / 100,
            "half_y_in": (dy_in + dyy_in) / 100,
            "half_y_out": (dy_out + dyy_out) / 100,
            "field": ni,
        })
        z += 2 * dz
    return magnets


def vessel_half_widths(z):
    """Half widths (x, y) of the decay vessel at z (linear in z)."""
    f = (np.asarray(z) - Z_VESSEL_MIN) / (Z_VESSEL_MAX - Z_VESSEL_MIN)
    return (
        VESSEL_X_IN + (VESSEL_X_OUT - VESSEL_X_IN) * f,
        VESSEL_Y_IN + (VESSEL_Y_OUT - VESSEL_Y_IN) * f,
    )


def inside_vessel(x, y, z):
    half_x, half_y = vessel_half_widths(z)
    return (
        (z >= Z_VESSEL_MIN) & (z <= Z_VESSEL_MAX)
        & (np.abs(x) <= half_x) & (np.abs(y) <= half_y)
    )
