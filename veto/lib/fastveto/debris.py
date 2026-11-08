"""EM debris of muons leaving the muon shield: whole Geant4 events of the debris library (bin/debris_library.py,
results/debris_library/iron_B0_a<angle>.npz: 0.5 m iron, the field changes it by < 5 %) drawn for each muon and
rotated onto it.

A muon of a dataset is at A (z = 30.6 m) with momentum p_30; straight back to the shield's exit face (Z_EXIT = 29.89 m,
end of the last magnet) it left the iron if that point is inside the magnet's outer face (|x| < EXIT_HALF_X,
|y| < EXIT_HALF_Y; the geometry cache has only the outer box, not the air gaps inside: an upper bound). For such a
muon one library event of the nearest (energy, angle) bin is drawn (also when it has no debris) and turned into the
lab: the library muon frame has the muon along +z and the slab normal at (-sin a, 0, cos a); the lab frame keeps the
muon direction and the plane (muon, z axis), so the debris keeps its orientation relative to the exit face.
"""

from pathlib import Path

import numpy as np

Z_EXIT = 29.89                                   # m, end of the last magnet's iron (geometry cache of TRY6)
EXIT_HALF_X, EXIT_HALF_Y = 1.71, 1.27            # m, its outer face at z = Z_EXIT
ANGLES = (0.0, 0.02, 0.05, 0.1, 0.2)


class Library:
    """Debris events per (angle, energy): CSR arrays of pdg, kinetic energy [GeV], offset [m] and momentum [GeV] in
    the muon frame, and the number of muons simulated."""

    def __init__(self, directory, kind="iron_B0"):
        self.angles = np.array(ANGLES)
        self.bins = {}
        for ia, a in enumerate(ANGLES):
            d = np.load(Path(directory) / f"{kind}_a{a:g}.npz")
            self.energies = np.array(d["energies"], dtype=float)
            n = int(d["n"])
            for ie, E in enumerate(self.energies):
                k = f"E{E:g}_"
                ev = d[k + "ev"].astype(np.int64)
                o = np.argsort(ev, kind="stable")
                start = np.searchsorted(ev[o], np.arange(n + 1))
                self.bins[(ia, ie)] = dict(n=n, start=start, pdg=d[k + "pdg"][o], ekin=d[k + "ekin"][o],
                                           off=d[k + "off"][o].astype(np.float64), p=d[k + "p"][o].astype(np.float64))

    def bin_of(self, E, angle):
        ie = np.abs(np.log(np.maximum(E, 1e-3))[:, None] - np.log(self.energies)[None, :]).argmin(1)
        ia = np.abs(angle[:, None] - self.angles[None, :]).argmin(1)
        return ia, ie


def frames(u, rng):
    """Per muon (n, 3) unit directions: orthonormal e1, e2, e3 = u with e1 = (cos t u - z) / sin t (the library's x)."""
    z = np.array([0.0, 0.0, 1.0])
    c = np.clip(u[:, 2], -1, 1)
    s = np.sqrt(np.maximum(0, 1 - c * c))
    e1 = (c[:, None] * u - z[None, :]) / np.where(s > 1e-9, s, 1)[:, None]
    flat = s <= 1e-9                                  # along z: any perpendicular, random azimuth
    if flat.any():
        phi = rng.uniform(0, 2 * np.pi, flat.sum())
        e1[flat] = np.stack([np.cos(phi), np.sin(phi), np.zeros(flat.sum())], 1)
    e2 = np.cross(u, e1)
    return e1, e2, u


def draw(lib, E, u, x_exit, rng):
    """Debris of muons (energy E [GeV], direction u, exit point x_exit [m], all (n,)/(n, 3)) that left the iron.
    Returns per particle: muon index, pdg, kinetic energy [GeV], start point [m] and unit direction (lab)."""
    angle = np.arccos(np.clip(u[:, 2], -1, 1))
    ia, ie = lib.bin_of(E, angle)
    e1, e2, e3 = frames(u, rng)
    out = {k: [] for k in ("mu", "pdg", "ekin", "x", "d")}
    for key in set(zip(ia.tolist(), ie.tolist())):
        b = lib.bins[key]
        m = np.nonzero((ia == key[0]) & (ie == key[1]))[0]
        ev = rng.integers(0, b["n"], len(m))
        cnt = b["start"][ev + 1] - b["start"][ev]
        if cnt.sum() == 0:
            continue
        mu = np.repeat(m, cnt)
        rec = np.concatenate([np.arange(b["start"][e], b["start"][e + 1]) for e in ev[cnt > 0]])
        off, p = b["off"][rec], b["p"][rec]
        lab = lambda v: v[:, :1] * e1[mu] + v[:, 1:2] * e2[mu] + v[:, 2:3] * e3[mu]
        dirn = lab(p)
        dirn /= np.maximum(np.linalg.norm(dirn, axis=1, keepdims=True), 1e-12)
        out["mu"].append(mu); out["pdg"].append(b["pdg"][rec]); out["ekin"].append(b["ekin"][rec])
        out["x"].append(x_exit[mu] + lab(off)); out["d"].append(dirn)
    if not out["mu"]:
        return dict(mu=np.zeros(0, int), pdg=np.zeros(0, int), ekin=np.zeros(0), x=np.zeros((0, 3)), d=np.zeros((0, 3)))
    return {k: np.concatenate(v) for k, v in out.items()}


def shield_exit(a, p):
    """Back-projection of muons at A (positions a, momenta p, (n, 3)) to the exit face; mask of those in its iron."""
    u = p / np.linalg.norm(p, axis=1, keepdims=True)
    t = (Z_EXIT - a[:, 2]) / u[:, 2]
    x = a + t[:, None] * u
    inside = (u[:, 2] > 0) & (np.abs(x[:, 0]) < EXIT_HALF_X) & (np.abs(x[:, 1]) < EXIT_HALF_Y)
    return x, u, inside
