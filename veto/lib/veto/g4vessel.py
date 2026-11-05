"""Geant4 transport of interaction final states through the vessel (helium, Mylar walls, air) and the spectrometer
field to the straw stations and the timing detector (cpp/g4vessel.cc; ISSUES A6, D4).

Used for the products of the interactions in the helium (bin/g4_vessel_transport.py) instead of straight lines +
RK4: they decay in flight (pi, K -> mu), K_L / neutrons interact in the helium / air, everything bends in the
spectrometer field and is lost in the magnet iron (|B| > spectrometer.YOKE_B).

    init(seed)                          the module, initialised with the veto geometry (geometry.py) and the field map
    transport_parts(g4, part, inters)   run interactions `inters` of a fastveto part table
    tracks_from_g4(res, part, n_inter)  charged tracks at SST1 in interactions.charged_tracks' format
"""

import sys

import numpy as np

from . import geometry as geo
from . import paths
from . import spectrometer

PLANES = (geo.Z_SST1, geo.Z_TIMING)   # recorded planes: 0 = SST1 (track state), 1 = timing detector
Z_WORLD = (geo.Z_STATE_A - 1.6, geo.Z_TIMING + 1.0)   # 29.0 .. 95.0 m: from behind the shield to past the TD
E_CUT = 0.1                           # GeV kinetic: below it a particle is not followed


def init(seed, e_cut=E_CUT, field=True, planes=PLANES):
    """Import and initialise cpp/build/g4vessel with the veto geometry."""
    # cpp/build: the container build (bin/g4.sh, Geant4 11.2.2); cpp/build_ship: host python + SHiP CVMFS Geant4 10.7.3
    # (bin/g4ship.sh). The Python-version suffix of the .so picks the right one.
    for d in (paths.cpp_build(), paths.PROJECT_DIR / "cpp" / "build_ship"):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    import g4vessel
    from fastveto import vessel as V
    if field:
        f = spectrometer.field_map()
        grid = f.grid.reshape(*f.n, 3).astype(np.float32)
        x0, y0, z0 = f.x0
        dx, dy, dz = f.d
    else:
        grid, x0, y0, z0, dx, dy, dz = np.zeros((0, 0, 0, 3), np.float32), 0, 0, 0, 1, 1, 1
    wall = {"mylar": "G4_MYLAR", "pvc": "G4_POLYVINYL_CHLORIDE"}[V.WALL_MATERIAL]
    g4vessel.initialize(seed=seed, z_world0=Z_WORLD[0], z_world1=Z_WORLD[1], z_vessel0=geo.Z_VESSEL_MIN,
                        z_vessel1=geo.Z_VESSEL_MAX, hx0=geo.VESSEL_X_IN, hx1=geo.VESSEL_X_OUT, hy0=geo.VESSEL_Y_IN,
                        hy1=geo.VESSEL_Y_OUT, wall_thickness=V.WALL_THICKNESS, wall_material=wall,
                        z_planes=np.array(planes), e_cut=e_cut, yoke_b=spectrometer.YOKE_B, field=grid,
                        x0=float(x0), dx=float(dx), y0=float(y0), dy=float(dy), z0=float(z0), dz=float(dz))
    return g4vessel


def chunk_inputs(part, inters):
    """Inputs of g4vessel.transport for the interactions `inters` (sorted ids) of a part table sorted by "int".
    Returns (offsets, rows) with rows the part rows sent, in order."""
    starts = np.searchsorted(part["int"], inters)
    ends = np.searchsorted(part["int"], inters, side="right")
    rows = np.concatenate([np.arange(a, b) for a, b in zip(starts, ends)]) if len(inters) else np.zeros(0, int)
    offsets = np.concatenate([[0], np.cumsum(ends - starts)]).astype(np.int64)
    return offsets, rows


def transport_parts(g4, part, inters):
    """Transport the particles of interactions `inters`. Returns the crossings with "inter" (interaction id) and
    "row" (part row of the origin particle, -1 for secondaries) instead of the chunk-local ev / origin."""
    offsets, rows = chunk_inputs(part, inters)
    p, v = part["p"][rows, :3].astype(np.float64), part["v"][rows].astype(np.float64)
    r = g4.transport(offsets, part["pdg"][rows].astype(np.int32), p[:, 0], p[:, 1], p[:, 2], v[:, 0], v[:, 1], v[:, 2])
    r = {k: np.asarray(x) for k, x in r.items()}
    ev, origin = r.pop("ev"), r.pop("origin")
    r["inter"] = np.asarray(inters)[ev]
    r["row"] = np.where(origin >= 0, rows[np.minimum(offsets[ev] + np.maximum(origin, 0), len(rows) - 1)], -1)
    return r


def tracks_from_g4(g, part, charge):
    """Charged tracks of the Geant4-transported interactions in interactions.charged_tracks' format.
    g: the saved crossings (transport_parts, concatenated); part: the fastveto part table; charge: pdg -> charge.
    A track = a charged particle crossing SST1 (plane 0); good = in the SST1 acceptance, p > 1 GeV and the same
    Geant4 track crosses the timing detector (plane 1) inside it. v = the production point of its origin particle
    (a decay product keeps its ancestor's: the kink is not modelled in the reco); secondaries (row -1) start at
    their SST1 point."""
    q_all = charge(g["pdg"])
    s1 = np.nonzero((g["plane"] == 0) & (q_all != 0) & (g["pz"] > 0))[0]
    td = g["plane"] == 1
    key = lambda i: g["inter"][i].astype(np.int64) * (1 << 24) + g["track"][i]
    td_keys = key(np.nonzero(td)[0])
    td_in = (np.abs(g["x"][td]) < geo.TIMING_HALF_X) & (np.abs(g["y"][td]) < geo.TIMING_HALF_Y)
    reach_td = np.isin(key(s1), td_keys[td_in])
    b = np.stack([g["x"][s1], g["y"][s1], np.full(len(s1), geo.Z_SST1)], 1)
    p = np.stack([g["px"][s1], g["py"][s1], g["pz"][s1]], 1)
    row = g["row"][s1]
    v = np.where((row >= 0)[:, None], part["v"][np.maximum(row, 0)], b)
    beam = np.where(row >= 0, part["beam"][np.maximum(row, 0)], False)
    acc = (np.abs(b[:, 0]) < geo.SST1_HALF_X) & (np.abs(b[:, 1]) < geo.SST1_HALF_Y)
    good = acc & (np.linalg.norm(p, axis=1) > 1) & reach_td
    return dict(row=row, inter=g["inter"][s1].astype(np.int64), q=q_all[s1], beam=beam.astype(bool), v=v, p=p, b=b,
                acc=acc, good=good)
