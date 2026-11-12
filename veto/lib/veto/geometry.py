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

# FairShip frame (geometry_config.py, layout 2026-0.1) -> this frame: z = z_FS - FS_OFFSET (targets aligned).
# Since 2026-11-05 the vessel, UBT, straw stations and timing detector sit at their FairShip positions.
# Before: vessel / UBT 30.6 - 80.6, SST1 82.0, timing 92.71 (4 x 6 m), i.e. the "deck" layout (OLD_* below).
FS_OFFSET = 2.14

# Planes where the simulated muons were STORED (MuonsAndMatter output planes[1], planes[2]). They are not
# detectors: every detector crossing is computed from these states (straight lines in air, field map after SST1).
Z_STATE_A = 30.6            # state A (stored z 30.600 .. 30.620)
Z_STATE_B = 82.0            # state B

Z_VESSEL_MIN = 33.27 - FS_OFFSET     # 31.13, decay-vessel entrance window
Z_VESSEL_MAX = 83.27 - FS_OFFSET     # 81.13, exit window
Z_UBT = Z_VESSEL_MIN                 # Upstream Background Tagger, on the entrance window
Z_SST = tuple(z - FS_OFFSET for z in (84.22, 86.22, 93.22, 95.22))   # straw stations 1-4: 82.08 84.08 91.08 93.08
Z_SST1 = Z_SST[0]                    # first straw station: acceptance, measured track state, veto reference
Z_TIMING = 96.15 - FS_OFFSET         # 94.01, timing detector (TimeDetPoint), behind the spectrometer magnet

SST1_HALF_X, SST1_HALF_Y = 2.0, 3.0          # straw station aperture (FairShip SSTScoringPoint plane 4 x 6 m)
TIMING_HALF_X, TIMING_HALF_Y = 2.75, 3.25    # FairShip timing detector 5.5 x 6.5 m

OLD_Z_VESSEL = (30.6, 80.6)          # deck layout, kept for designs written in that frame (design.py)
OLD_Z_SST1, OLD_Z_TIMING = 82.0, 94.850 - 2.14
OLD_TIMING_HALF = (2.0, 3.0)
VESSEL_SHIFT = Z_VESSEL_MIN - OLD_Z_VESSEL[0]   # +0.53 m: fins of old-frame designs move with the vessel

# ------------------------------------------------------------
# Decay vessel: rectangular frustum whose lateral walls carry the SBT (same size in FairShip)
# ------------------------------------------------------------

VESSEL_X_IN, VESSEL_X_OUT = 0.5, 2.0     # half widths at entrance / exit
VESSEL_Y_IN, VESSEL_Y_OUT = 1.35, 3.0

# End of the muon shield (TRY6, MuonsAndMatter frame): exit face of the last magnet (MS7) and its outer iron envelope,
# constant along z (dX + gap + x_yoke = 69 + 8 + 94 cm, dY + dY_yoke = 51 + 76 cm). Used by shield_iron, fastveto
# (targets, debris) and the shell storage.
Z_SHIELD_END = 29.89
SHIELD_END_HALF_X, SHIELD_END_HALF_Y = 1.71, 1.27

# UBT aperture = vessel entrance window
UBT_HALF_X, UBT_HALF_Y = VESSEL_X_IN, VESSEL_Y_IN
# Default UBT layout since 11 Nov 2026 (Guglielmo): a straw plane the size of the entrance window 40 cm upstream of it
# (the FairShip UpstreamTagger position) and a 4 x 4 m^2 plane of 4 x 4 cm^2 scintillating tiles on the window, covering
# the window and the region around it; no fins at the entrance z. Z_UBT stays the window (muon classification).
Z_UBT_STRAW = Z_VESSEL_MIN - 0.40     # 30.73
Z_UBT_TILES = Z_VESSEL_MIN            # 31.13
UBT_TILES_HALF_X = UBT_TILES_HALF_Y = 2.0
# The muon flames on the tile plane (Nov-4: a band |y| < ~0.5 m at |x| > ~1.2 m, peak 0.69 MHz/cm^2): 2 x 2 cm^2 tiles
# there (Guglielmo, 11 Nov 2026), 4 x 4 cm^2 elsewhere. Flame blocks: FLAME_X_IN <= |x| <= 2 m, |y| <= FLAME_Y.
UBT_TILES_FLAME_X_IN, UBT_TILES_FLAME_Y = 1.0, 0.6

# ------------------------------------------------------------
# Cavern walls seen by "bouncing" muons (from Muon_Flames_rate.py / vessel dataset)
# ------------------------------------------------------------

CAVERN_X_WALL = -4.56
CAVERN_Y_FLOOR = -3.36

# ------------------------------------------------------------
# Spectrometer magnet
# ------------------------------------------------------------

# FairShip places the map (local z -9.9 .. +9.9 m, data/MainSpectrometerField.root) at the magnet centre Bfield.z = c.z,
# 89.72 m in the TRY2026 layout (stations c.z -5.5 / -3.5 / +3.5 / +5.5 m = 84.22 .. 95.22). The .h5 copy has it centred
# at 86.96 (an older c.z = 89.10): spectrometer.field_map() re-centres it at Z_MAGNET (+0.62 m, fixed 2026-11-05).
Z_MAGNET = 89.72 - FS_OFFSET          # 87.58
SPECTROMETER_Z = (84.98, 90.18)       # |B| > 10 % of max, after re-centring (was 84.36 - 89.56 in the .h5 frame)
SPECTROMETER_HALF_X = 2.7
SPECTROMETER_HALF_Y = 3.8
FIELD_MAP_Z = (Z_MAGNET - 9.90, Z_MAGNET + 9.90)

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


def vessel_half_widths(z, clip=False):
    """Half widths (x, y) of the decay vessel at z (linear in z). z: number, numpy array or torch tensor (same type
    back). clip=False: the walls continued past both ends (fins / side planes); clip=True: entrance widths upstream of
    the vessel, exit widths downstream."""
    if not hasattr(z, "clip"):
        z = np.asarray(z, dtype=float)
    f = (z - Z_VESSEL_MIN) / (Z_VESSEL_MAX - Z_VESSEL_MIN)
    if clip:
        f = f.clip(0, 1)
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
