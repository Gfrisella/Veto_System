"""The last iron of the muon shield (TRY6 MS7) as an interaction target: geometry, field map, input muons.

Frame: MuonsAndMatter's shield frame, as geometry.py (shield end Z_SHIELD_END = 29.89 m, decay vessel from 30.6 m).

  iron       MS7 is ONE solid G4_Fe Arb8 (MuonsAndMatter create_magnet: the coil gap is filled with iron) whose
             outer envelope is constant along z: |x| < 1.71 m (dX + gap + x_yoke = 38 + 8 + 125 = 69 + 8 + 94 cm),
             |y| < 1.27 m (dY + dY_yoke = 51 + 76 cm). The target is its last DZ = 0.5 m: Z_IRON0 .. Z_SHIELD_END.
  field      MuonsAndMatter/data/outputs/fields_mm.h5 (FEM, diluted iron, first quadrant, float16 points): grid
             x 0 - 2.2 m, y 0 - 2.18 m (2 cm), z -0.5 - 32 m (5 cm). In the map MS7 ends at 30.04 m
             (magnet_params.csv: Z_pos 26.04 + 4 m), 0.15 m after the iron: FIELD_Z_SHIFT aligns the two ends.
  muons      Muons_2950_50cm_before_end_MS.pkl: every muon of one run (old 506 M input) at a plane stored at
             z = 29.50 m, which is really at z = 29.20 m (translated by 30 cm, Guglielmo 2026-10-30): MUON_Z_SHIFT.
             They are carried in a straight line from there (no field, no energy loss before the vertex except a
             mean dE/dx) through the target.
"""

from pathlib import Path

import numpy as np

from . import geometry as geo

Z_SHIELD_END = geo.Z_SHIELD_END
DZ = 0.5
Z_IRON0 = Z_SHIELD_END - DZ
HALF_X, HALF_Y = geo.SHIELD_END_HALF_X, geo.SHIELD_END_HALF_Y
RHO_FE = 7.874                      # g/cm^3 (G4_Fe)
A_FE, Z_FE = 55.845, 26.0
DEDX_MU = 1.6                       # GeV/m, mean muon energy loss in iron (10 - 300 GeV, ionisation + radiative)
Z_PLANE = Z_SHIELD_END + 0.01       # where the transported final states are recorded

MM_OUTPUTS = Path("/disk/users/gfrise/MuonsAndMatter/data/outputs")
MUON_FILE = MM_OUTPUTS / "Muons_2950_50cm_before_end_MS.pkl"
MUON_Z_SHIFT = -0.30                # true z = stored z - 0.30 m
FIELD_FILE = MM_OUTPUTS / "fields_mm.h5"
FIELD_GRID = ((0.0, 0.02, 111), (0.0, 0.02, 110), (-0.5, 0.05, 651))   # (start, step, n) in m
FIELD_Z_SHIFT = 30.04 - Z_SHIELD_END


def iron_chord(x, y, z, ux, uy, uz, z0=Z_IRON0, z1=Z_SHIELD_END):
    """Straight lines from (x, y, z) along u through the target box: path length inside [m] and the entry / exit
    parameters s_in, s_out along u (s = 0 at the start point). Lines that miss it get length 0."""
    x, y, z, ux, uy, uz = (np.asarray(a, np.float64) for a in (x, y, z, ux, uy, uz))
    lo = np.full(x.shape, -np.inf)
    hi = np.full(x.shape, np.inf)
    for p, u, a, b in ((x, ux, -HALF_X, HALF_X), (y, uy, -HALF_Y, HALF_Y), (z, uz, z0, z1)):
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (a - p) / u, (b - p) / u
        par = u == 0
        inside = (p >= a) & (p <= b)
        lo = np.maximum(lo, np.where(par, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2)))
        hi = np.minimum(hi, np.where(par, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2)))
    ok = hi > lo
    return np.where(ok, hi - lo, 0.0), np.where(ok, lo, np.nan), np.where(ok, hi, np.nan)


def nuclei_per_m3():
    return RHO_FE * 1e6 / A_FE * 6.02214076e23


def nucleons_per_m3():
    return RHO_FE * 1e6 * 6.02214076e23


def load_field(path=FIELD_FILE, z_range=(Z_IRON0 - 1.0, Z_PLANE + 1.0)):
    """The field map near the target as a regular grid: dict(field (nx, ny, nz, 3) T, x0, dx, y0, dy, z0, dz, z_shift)
    in the map's own z frame (look up at z + z_shift). Points are placed by their rounded grid indices."""
    import h5py
    (x0, dx, nx), (y0, dy, ny), (zg0, dz, nz) = FIELD_GRID
    with h5py.File(path) as h:
        P = h["points"][:].astype(np.float64)
        B = h["B"][:].astype(np.float64)
    zlo, zhi = z_range[0] + FIELD_Z_SHIFT, z_range[1] + FIELD_Z_SHIFT
    k0 = max(int(np.floor((zlo - zg0) / dz)) - 1, 0)
    k1 = min(int(np.ceil((zhi - zg0) / dz)) + 1, nz - 1)
    ix = np.rint((P[:, 0] - x0) / dx).astype(np.int64)
    iy = np.rint((P[:, 1] - y0) / dy).astype(np.int64)
    iz = np.rint((P[:, 2] - zg0) / dz).astype(np.int64)
    m = (iz >= k0) & (iz <= k1)
    grid = np.full((nx, ny, k1 - k0 + 1, 3), np.nan)
    grid[ix[m], iy[m], iz[m] - k0] = B[m]
    if not np.isfinite(grid).all():
        raise ValueError(f"field map {path}: {np.isnan(grid[..., 0]).sum()} grid points missing")
    return dict(field=grid, x0=x0, dx=dx, y0=y0, dy=dy, z0=zg0 + k0 * dz, dz=dz, z_shift=FIELD_Z_SHIFT)


def field_at(fm, x, y, z):
    """Field [T] at shield-frame points, as cpp/g4shield.cc does (mirrored quadrant, trilinear)."""
    x, y, z = (np.asarray(a, np.float64) for a in (x, y, z))
    g = fm["field"]
    f = [(np.abs(x) - fm["x0"]) / fm["dx"], (np.abs(y) - fm["y0"]) / fm["dy"], (z + fm["z_shift"] - fm["z0"]) / fm["dz"]]
    ok = np.ones(x.shape, bool)
    i = []
    for a, n in zip(f, g.shape[:3]):
        ok &= (a >= 0) & (a < n - 1)
        i.append(np.clip(a.astype(np.int64), 0, n - 2))
    out = np.zeros(x.shape + (3,))
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                c = ((f[0] - i[0]) if di else (1 - (f[0] - i[0]))) * ((f[1] - i[1]) if dj else (1 - (f[1] - i[1]))) \
                    * ((f[2] - i[2]) if dk else (1 - (f[2] - i[2])))
                out += c[..., None] * g[i[0] + di, i[1] + dj, i[2] + dk]
    sx, sy = np.where(x < 0, -1.0, 1.0), np.where(y < 0, -1.0, 1.0)
    out[..., 0] *= sx * sy
    out[..., 2] *= sy
    return np.where(ok[..., None], out, 0.0)


def g4shield_module(natural=None, retries=6):
    """Import the Geant4 module (g4shield, or g4shieldnat when natural): loads the Geant4 libraries from CVMFS.
    ISSUES D15: call it in the PARENT before the worker pool forks, so the workers inherit the loaded libraries and never
    load them from CVMFS themselves (64 workers loading at once made CVMFS fail: ImportError libG4analysis.so). If an
    import still fails it is retried with backoff (fallback) and reported once."""
    import importlib
    import os
    import sys
    import time
    from . import paths
    for d in (str(paths.cpp_build()), str(paths.PROJECT_DIR / "cpp" / "build_ship"), os.environ.get("G4SHIELD_BUILD")):
        if d and d not in sys.path:                                       # host build (bin/g4ship.sh); G4SHIELD_BUILD
            sys.path.insert(0, d)                                         # e.g. bin/g4pixi.sh (Geant4 11.4.2, CVMFS)
    name = "g4shieldnat" if natural else "g4shield"
    for k in range(retries + 1):
        try:
            return importlib.import_module(name)
        except ImportError as e:
            if k == retries or "No module named" in str(e):
                raise
            print(f"[pid {os.getpid()}] import {name} failed ({e}); retry {k + 1}/{retries}", file=sys.stderr, flush=True)
            time.sleep(min(30.0, 0.5 * 2 ** k))


def init_g4shield(seed, e_cut=0.5, field=True, z_iron0=None, natural=None):
    """Import and initialise the Geant4 module (cpp/build/g4shield) for this target. natural = dict(physics_list,
    mu_bias): the g4shieldnat module instead (natural, biased muonNuclear; bin/shield_natural_check.py, D3)."""
    g4shield = g4shield_module(natural)
    extra = dict(physics_list=natural["physics_list"], mu_bias=float(natural["mu_bias"])) if natural else {}
    fm = load_field() if field else dict(field=np.zeros((0, 0, 0, 3)), x0=0, dx=1, y0=0, dy=1, z0=0, dz=1, z_shift=0)
    g4shield.initialize(seed=seed, z_iron0=Z_IRON0 - 0.2 if z_iron0 is None else z_iron0, z_iron1=Z_SHIELD_END, half_x=HALF_X, half_y=HALF_Y,
                        z_plane=Z_PLANE, e_cut=e_cut, field=fm["field"], x0=fm["x0"], dx=fm["dx"], y0=fm["y0"],
                        dy=fm["dy"], z0=fm["z0"], dz=fm["dz"], z_shift=fm["z_shift"], **extra)
    return g4shield


def curved_points(start, u, p, q, s, fm=None, step=0.01):
    """Points and directions of muons bent by the MS7 field (ISSUES D4: was a straight line), at path lengths s.

    start (n, 3) [m], u (n, 3) unit directions, p (n,) [GeV] at start, q (n,) charge, s (n,) path length from start
    to the wanted point [m]. RK4 in path length, du/ds = (q k / p) u x B (k = 0.2998 GeV / (T m)), p(s) = p - DEDX_MU s
    (mean loss; the muon is assumed to get there). Field: field_at (mirrored quadrant map). Returns (x (n, 3),
    u (n, 3), p (n,)) at s."""
    fm = fm or load_field()
    start, u = np.asarray(start, np.float64), np.asarray(u, np.float64)
    p, q, s = np.asarray(p, np.float64), np.asarray(q, np.float64), np.asarray(s, np.float64)
    k = 0.299792458
    x, d = start.copy(), u.copy()
    out_x, out_u = start.copy(), u.copy()
    done = s <= 0
    s_now = 0.0
    s_max = float(s.max()) if len(s) else 0.0

    def du(xx, dd, pp):
        B = field_at(fm, xx[:, 0], xx[:, 1], xx[:, 2])
        return (q * k / np.maximum(pp, 0.05))[:, None] * np.cross(dd, B)

    while s_now < s_max - 1e-12:
        h = min(step, s_max - s_now)
        pp = p - DEDX_MU * s_now
        k1u = du(x, d, pp); k1x = d
        k2u = du(x + h / 2 * k1x, d + h / 2 * k1u, pp - DEDX_MU * h / 2); k2x = d + h / 2 * k1u
        k3u = du(x + h / 2 * k2x, d + h / 2 * k2u, pp - DEDX_MU * h / 2); k3x = d + h / 2 * k2u
        k4u = du(x + h * k3x, d + h * k3u, pp - DEDX_MU * h); k4x = d + h * k3u
        x_new = x + h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        d_new = d + h / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
        d_new /= np.linalg.norm(d_new, axis=1, keepdims=True)
        hit = ~done & (s <= s_now + h)
        if hit.any():                                  # linear between the two steps
            f = ((s[hit] - s_now) / h)[:, None]
            out_x[hit] = x[hit] + f * (x_new[hit] - x[hit])
            v = d[hit] + f * (d_new[hit] - d[hit])
            out_u[hit] = v / np.linalg.norm(v, axis=1, keepdims=True)
            done |= hit
        x, d = x_new, d_new
        s_now += h
    return out_x, out_u, p - DEDX_MU * s
