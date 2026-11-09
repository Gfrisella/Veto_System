"""EM debris from the end of the muon shield (ISSUES B2, decided 2026-11-05: our own Geant4, host build).

Two Geant4 stages (separate processes: one G4RunManager per process), driven by bin/em_debris.py:
  1. shield   the muons crossing the last 50 cm of MS7 iron (data/shield_dis/muons_iron.npz, state at z = 29.20 m),
              carried to the last DEPTH_X0 radiation lengths (shield_iron.curved_points: field, mean dE/dx) and through
              that slab with Geant4 (cpp/g4shield, FTFP_BERT incl. EM, field); every e+-, gamma above E_CUT at the plane
              z = 29.90 m is kept.
  2. vessel   those e / gamma through air, the Mylar window and the helium (cpp/g4vessel, no field there) to the UBT
              plane and the fin planes z = Z_UBT .. Z_UBT + 10 m (every metre); every e+- / gamma crossing is recorded.
Maps (make_maps) in the binning of the muon rate maps of results/reco/<inputs>.npz: UBT (|x| < 0.5, y bins) and fins
(u = |x| - X(z), y; per side and z slice), e and gamma separately, particles per spill. Beyond z = Z_UBT + 10 m: 0.

Detection (detection_probability): an electron fires a plane with its efficiency; a photon only if it converts in the
plane, P = 1 - exp(-7/9 x/X0) with the plane's material (technology.x0). reco.plane_rate_map adds
P_e * e-map + P_gamma * gamma-map to the muon map when the maps carry "em_*" (run_reco.py --em-debris).
"""

import numpy as np

from . import geometry as geo
from . import technology as tech

E_CUT = 0.001                    # GeV kinetic: e / gamma followed and recorded above 1 MeV
X0_FE = 0.01757                  # m, radiation length of iron
DEPTH_X0 = 0.5 / X0_FE           # the whole last 50 cm of MS7 (28.5 X0, 29.39 - 29.89 m). Tested 2026-11-06 (2000 muons):
                                 # 3 / 6 / 10 / 20 X0 give 17 / 30 / 72 / 72 % of the e / gamma of the full depth (deep
                                 # showers, MeV photons ~3.4 X0), same run time -> Guglielmo: full 50 cm. --depth to vary.
Z_PLANES = tuple(geo.Z_UBT + np.arange(0, 10.001, 1.0))
EM_FILE = "em_maps.npz"          # results/em_debris/


def detection_probability(plane):
    """(P_electron, P_gamma) of a plane: its efficiency, and conversion in its material for photons."""
    if plane.technology is None:
        return plane.efficiency, 0.0
    x0 = tech.x0(tech.TECHNOLOGIES[plane.technology]) * plane.stations
    return plane.efficiency, plane.efficiency * (1 - np.exp(-7.0 / 9.0 * x0))


def make_maps(rec, w_per_spill, maps):
    """rec: plane crossings of stage 2 (plane, pdg, x, y; ev -> muon index), w_per_spill: weight per muon index.
    maps: the muon maps (u_bins, y_bins, z_slices, ubt_y_bins). Returns dict of em_* arrays."""
    z_planes = np.array(Z_PLANES)
    out = {"em_z": z_planes}
    w = w_per_spill[rec["ev"]]
    kinds = {"e": np.abs(rec["pdg"]) == 11, "g": rec["pdg"] == 22}
    ub, yb, zs, uyb = maps["u_bins"], maps["y_bins"], maps["z_slices"], maps["ubt_y_bins"]
    for k, m in kinds.items():
        at_ubt = m & (rec["plane"] == 0) & (np.abs(rec["x"]) < geo.UBT_HALF_X)
        out[f"em_ubt_{k}"] = np.histogram(rec["y"][at_ubt], uyb, weights=w[at_ubt])[0]
        fin = np.zeros((2, len(zs), len(ub) - 1, len(yb) - 1))
        for iz, z in enumerate(zs):
            ip = np.nonzero(np.abs(z_planes - z) <= 0.51)[0]   # nearest EM plane (old inputs: slices from 30.6 m)
            if not len(ip):
                continue                                 # no EM plane within 0.5 m (beyond Z_UBT + 10 m): 0
            sel = m & (rec["plane"] == ip[0])
            xw = geo.vessel_half_widths(z)[0]
            u = np.abs(rec["x"][sel]) - xw
            side = (rec["x"][sel] < 0).astype(int)       # 0 = left (+x), 1 = right (-x): as fin_maps
            for sd in (0, 1):
                s = side == sd
                fin[sd, iz] = np.histogram2d(u[s], rec["y"][sel][s], [ub, yb], weights=w[sel][s])[0]
        out[f"em_fin_{k}"] = fin
    return out
