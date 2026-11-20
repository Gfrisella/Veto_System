"""Shared geometry helpers for the UBT / SBT distribution study (20 Nov 2026).

Everything is truth level: straight charged-track segments p -> q (no field before the spectrometer, no material), crossed
with the UBT straw plane (z = 30.73 m, |x| < 0.5, |y| < 1.35), the UBT tile plane (z = 31.13 m, |x|, |y| < 2) and the SBT
cell shell (lib/veto/sbt.py shell_pieces: 20 cm LS + 2 x 0.8 cm Al outside the vessel frustum, 61 cell rings x 14 cells).
A cell "fires" when 150 MeV/m x path in the LS > 30 MeV (R2 threshold), with efficiency 0.993.
"""
import os
import sys

W = "/disk/gfs_lhcb/gfrise/Study_on_Veto_System/Optimization_Veto_System/.claude/worktrees/nov4-source/veto"
sys.path[:0] = [W + "/bin", W + "/lib"]
os.chdir(W)
import _bootstrap  # noqa: E402,F401
import numpy as np  # noqa: E402

from veto import geometry as geo  # noqa: E402
from veto import sbt  # noqa: E402

OUT = "/disk/gfs_lhcb/gfrise/Study_on_Veto_System/Optimization_Veto_System/veto/results_rerun/investigations/ubt_sbt_distributions"
EVSTORE = "/disk/gfs_lhcb/gfrise/Study_on_Veto_System/Optimization_Veto_System/veto/results_rerun/evstore"
Z_STRAW, Z_TILE = geo.Z_UBT_STRAW, geo.Z_UBT_TILES
STRAW_HALF = (geo.UBT_HALF_X, geo.UBT_HALF_Y)          # 0.5, 1.35 (R2: straws over the entrance window)
TILE_HALF = (geo.UBT_TILES_HALF_X, geo.UBT_TILES_HALF_Y)  # 2, 2
THRESHOLD_MEV, EFFICIENCY = 30.0, 0.993
SPILLS_15Y = 1.5e7

# cell ring (0..60) -> SR-389 ring (0..9)
SR_OF_CELLRING = np.full(sbt.N_RING, -1)
for _k in range(sbt.N_SR_RINGS):
    SR_OF_CELLRING[sbt.sr_ring_cells(_k)] = _k


def plane_crossing(p, q, z):
    """Crossing of the segments p -> q with the plane z (either direction). Returns idx, x, y."""
    lo, hi = np.minimum(p[:, 2], q[:, 2]), np.maximum(p[:, 2], q[:, 2])
    m = (lo < z) & (hi >= z) & (hi > lo)
    i = np.nonzero(m)[0]
    t = (z - p[i, 2]) / (q[i, 2] - p[i, 2])
    return i, p[i, 0] + t * (q[i, 0] - p[i, 0]), p[i, 1] + t * (q[i, 1] - p[i, 1])


def ubt(p, q):
    """dict straw / tile: (idx, x, y, inside-aperture mask)."""
    out = {}
    for name, z, (hx, hy) in (("straw", Z_STRAW, STRAW_HALF), ("tile", Z_TILE, TILE_HALF)):
        i, x, y = plane_crossing(p, q, z)
        out[name] = (i, x, y, (np.abs(x) < hx) & (np.abs(y) < hy))
    return out


def sbt_pieces(p, q, rng, chunk=2_000_000):
    """SBT shell pieces of the segments: seg, face (0 +x, 1 -x, 2 +y, 3 -y), ring (cell ring), cellseg, z, across
    (y on the side walls, x on top / bottom, at the piece middle), edep [MeV], fired (bool), cell id."""
    cols = {k: [] for k in ("seg", "face", "ring", "cellseg", "z", "across", "edep", "fired", "cell")}
    for s0 in range(0, len(p), chunk):
        pp, qq = p[s0:s0 + chunk], q[s0:s0 + chunk]
        c = sbt.shell_pieces(pp, qq)
        if not len(c["seg"]):
            continue
        sg = c["seg"].astype(int)
        mid = pp[sg] + c["t_mid"][:, None] * (qq[sg] - pp[sg])
        face = c["face"].astype(int)
        across = np.where(face < 2, mid[:, 1], mid[:, 0])
        edep = sbt.DEDX_LS * c["length"]
        ok = c["ring"] >= 0
        fired = ok & (edep > THRESHOLD_MEV) & (rng.random(len(edep)) < EFFICIENCY)
        cols["seg"].append(sg[ok] + s0); cols["face"].append(face[ok]); cols["ring"].append(c["ring"][ok].astype(int))
        cols["cellseg"].append(c["cellseg"][ok].astype(int)); cols["z"].append(c["z"][ok]); cols["across"].append(across[ok])
        cols["edep"].append(edep[ok]); cols["fired"].append(fired[ok])
        cols["cell"].append(sbt.cell_id(face[ok], c["ring"][ok].astype(int), c["cellseg"][ok].astype(int)))
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}


def species(pdg):
    """Coarse particle class of PDG codes: mu, pi, K, p, e, other."""
    a = np.abs(np.asarray(pdg))
    out = np.full(len(a), "other", dtype=object)
    for code, name in ((13, "mu"), (211, "pi"), (321, "K"), (2212, "p"), (11, "e")):
        out[a == code] = name
    return out


# ---------------------------------------------------------------- weighted histograms (all per spill)
SPECIES = ("mu", "pi", "K", "p", "e", "other")
BINS = dict(xy=np.linspace(-4, 4, 161), p=np.logspace(-1, 3, 61), ring=sbt.RING_EDGES, sr=np.arange(11) - 0.5,
            face=np.arange(5) - 0.5, phi=np.linspace(-180, 180, 73), side=np.linspace(-3.2, 3.2, 65),
            tb=np.linspace(-2.2, 2.2, 45), mult=np.arange(0, 61) - 0.5, edep=np.logspace(-1, 3.5, 46),
            species=np.arange(len(SPECIES) + 1) - 0.5, cat=np.arange(5) - 0.5)
CATS = ("neither", "UBT only", "SBT only", "UBT and SBT")


def species_idx(pdg):
    a = np.abs(np.asarray(pdg))
    out = np.full(len(a), 5)
    for k, code in enumerate((13, 211, 321, 2212, 11)):
        out[a == code] = k
    return out


class Hists(dict):
    def h1(self, key, x, w, b):
        h, _ = np.histogram(x, bins=BINS[b], weights=w)
        self[key] = self.get(key, 0) + h
        self.setdefault("_w2|" + key, 0)
        self["_w2|" + key] = self["_w2|" + key] + np.histogram(x, bins=BINS[b], weights=np.asarray(w) ** 2)[0]

    def h2(self, key, x, y, w, bx, by):
        h, _, _ = np.histogram2d(x, y, bins=(BINS[bx], BINS[by]), weights=w)
        self[key] = self.get(key, 0) + h

    def save(self, path):
        np.savez(path, **{k: np.asarray(v) for k, v in self.items()}, **{"_bins_" + k: v for k, v in BINS.items()})


def phi_norm(face, z, across):
    """Azimuth [deg] on the frustum in coordinates normalised to the local half widths (corners at +-45, +-135 deg)."""
    hx, hy = geo.vessel_half_widths(np.clip(z, geo.Z_VESSEL_MIN, geo.Z_VESSEL_MAX))
    u = np.where(face == 0, 1.0, np.where(face == 1, -1.0, across / hx))
    v = np.where(face == 2, 1.0, np.where(face == 3, -1.0, across / hy))
    return np.degrees(np.arctan2(v, u))


def fill_ubt(H, grp, plane, x, y, inside, p, w, pdg=None):
    H.h2(f"{grp}|{plane}_xy", x, y, w, "xy", "xy")
    H.h1(f"{grp}|{plane}_p", p[inside], w[inside], "p")
    if pdg is not None:
        H.h1(f"{grp}|{plane}_species", species_idx(pdg[inside]), w[inside], "species")


def fill_sbt(H, grp, face, ring, z, across, fired, p, w, edep=None, pdg=None):
    H.h1(f"{grp}|sbt_cross_ring", z, w, "ring")
    H.h1(f"{grp}|sbt_cross_face", face, w, "face")
    if edep is not None:
        H.h1(f"{grp}|sbt_edep", edep, w, "edep")
    f = fired.astype(bool)
    face, ring, z, across, p, w = face[f], ring[f], z[f], across[f], p[f], w[f]
    H.h1(f"{grp}|sbt_fired_ring", z, w, "ring")
    H.h1(f"{grp}|sbt_fired_sr", SR_OF_CELLRING[ring.astype(int)], w, "sr")
    H.h1(f"{grp}|sbt_fired_face", face, w, "face")
    H.h1(f"{grp}|sbt_fired_p", p, w, "p")
    H.h2(f"{grp}|sbt_fired_zphi", z, phi_norm(face, z, across), w, "ring", "phi")
    for fi in range(4):
        m = face == fi
        H.h2(f"{grp}|sbt_fired_face{fi}", z[m], across[m], w[m], "ring", "side" if fi < 2 else "tb")
    if pdg is not None:
        H.h1(f"{grp}|sbt_fired_species", species_idx(np.asarray(pdg)[f]), w, "species")


def fill_events(H, grp, n_straw, n_tile, n_fired, w):
    H.h1(f"{grp}|ev_n_straw", np.minimum(n_straw, 59), w, "mult")
    H.h1(f"{grp}|ev_n_tile", np.minimum(n_tile, 59), w, "mult")
    H.h1(f"{grp}|ev_n_fired", np.minimum(n_fired, 59), w, "mult")
    u, s = (n_straw > 0) | (n_tile > 0), n_fired > 0
    H.h1(f"{grp}|ev_cat", u.astype(int) + 2 * s.astype(int), w, "cat")
    H.h1(f"{grp}|ev_cat_tile", (n_tile > 0).astype(int) + 2 * s.astype(int), w, "cat")
