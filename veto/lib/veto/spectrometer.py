"""Charged tracks through the spectrometer field (ISSUES A6): RK4 in the field map, straight lines elsewhere.

Field: MainSpectrometerField.h5 (io.load_field_map), a regular 10 cm grid over |x| < 4.7, |y| < 4.05, 19.8 m long,
re-centred at geometry.Z_MAGNET = 87.58 m (FairShip c.z = 89.72; the .h5 z values are centred at 86.96, i.e. 0.62 m
upstream): yoke 84.98 - 90.18 m, between straw stations 2 (84.08) and 3 (91.08). Trilinear interpolation, zero outside the map. In the aperture the field is
mostly Bx (~ -0.15 T on axis, integral ~ 0.75 T m), so tracks bend in y; the yoke (|y| > ~3.3 m) carries
~1.5 T.

Track model: state (x, y, tx = px/pz, ty = py/pz, q/p) with z as the free parameter,
    dtx/dz = k q/p N (tx ty Bx - (1 + tx^2) By + ty Bz)
    dty/dz = k q/p N ((1 + ty^2) Bx - tx ty By - tx Bz),   N = sqrt(1 + tx^2 + ty^2), k = 0.299792458 GeV / (T m),
classic RK4 with a fixed step in z. No energy loss, no scattering (air). Neutral tracks (q = 0) stay straight.
A track is LOST when it meets |B| > YOKE_B (it is in the magnet iron) or turns back (pz <= 0).

    to_z(pos, mom, q, z)                  states at plane z (arrays (n, 3), q (n,))
    timing_acceptance(pos, mom, q)        mask: reaches the timing detector inside its aperture, not lost
"""

from functools import lru_cache

import numpy as np

from . import geometry as geo

K_GEV = 0.299792458        # GeV / (T m) per unit charge
YOKE_B = 0.5               # T: |B| above this = inside the magnet iron (aperture field <= 0.2 T)
STEP = 0.10                # m, RK4 step in z inside the map
CHUNK = 2_000_000


class FieldMap:
    """Regular-grid field map, trilinear, zero outside."""

    def __init__(self, points, B):
        axes = [np.unique(np.round(points[:, i], 4)) for i in range(3)]
        self.x0 = np.array([a[0] for a in axes])
        self.d = np.array([(a[-1] - a[0]) / (len(a) - 1) for a in axes])
        self.n = np.array([len(a) for a in axes])
        idx = np.rint((points - self.x0) / self.d).astype(np.int64)
        grid = np.zeros((*self.n, 3), np.float32)
        grid[idx[:, 0], idx[:, 1], idx[:, 2]] = B
        self.grid = grid.reshape(-1, 3)
        self.lo = self.x0
        self.hi = self.x0 + self.d * (self.n - 1)
        self._tgrid = {}

    def __call__(self, x, y, z):
        """B (n, 3) [T] at the points (x, y, z) (arrays (n,))."""
        f = (np.stack([x, y, z], 1) - self.x0) / self.d
        inside = np.all((f >= 0) & (f < self.n - 1), axis=1)
        f = np.where(inside[:, None], f, 0.0)
        i = np.minimum(f.astype(np.int64), self.n - 2)
        u = f - i
        sy, sz = self.n[2], 1
        sx = self.n[1] * self.n[2]
        base = i[:, 0] * sx + i[:, 1] * sy + i[:, 2] * sz
        out = np.zeros((len(x), 3))
        for dx in (0, 1):
            wx = u[:, 0] if dx else 1 - u[:, 0]
            for dy in (0, 1):
                wy = wx * (u[:, 1] if dy else 1 - u[:, 1])
                for dz in (0, 1):
                    w = wy * (u[:, 2] if dz else 1 - u[:, 2])
                    out += w[:, None] * self.grid[base + dx * sx + dy * sy + dz * sz]
        out[~inside] = 0.0
        return out


@lru_cache(maxsize=1)
def field_map(filename=None):
    """The map re-centred at geometry.Z_MAGNET (FairShip places it at Bfield.z; the .h5 copy is centred 0.62 m
    upstream of the TRY2026 magnet centre)."""
    from . import io
    pts, B = io.load_field_map(filename)
    pts = np.array(pts, np.float64)
    pts[:, 2] += geo.Z_MAGNET - 0.5 * (pts[:, 2].min() + pts[:, 2].max())
    return FieldMap(pts, B)


def _torch_grid(field, device):
    import torch
    key = str(device)
    if key not in field._tgrid:
        field._tgrid[key] = (torch.as_tensor(field.grid, dtype=torch.float32, device=device),
                             torch.as_tensor(field.x0, dtype=torch.float64, device=device),
                             torch.as_tensor(field.d, dtype=torch.float64, device=device),
                             torch.as_tensor(field.n, dtype=torch.int64, device=device))
    return field._tgrid[key]


def _field_t(tg, x, y, z):
    """Trilinear field at torch points (n,), torch version of FieldMap.__call__."""
    import torch
    grid, x0, d, n = tg
    f = (torch.stack([x, y, z], 1) - x0) / d
    inside = ((f >= 0) & (f < (n - 1))).all(1)
    f = torch.where(inside[:, None], f, torch.zeros_like(f))
    i = torch.minimum(f.long(), n - 2)
    u = (f - i).to(torch.float32)
    sx, sy = int(n[1] * n[2]), int(n[2])
    base = i[:, 0] * sx + i[:, 1] * sy + i[:, 2]
    out = torch.zeros((len(x), 3), dtype=torch.float32, device=x.device)
    for dx in (0, 1):
        wx = u[:, 0] if dx else 1 - u[:, 0]
        for dy in (0, 1):
            wy = wx * (u[:, 1] if dy else 1 - u[:, 1])
            for dz in (0, 1):
                w = wy * (u[:, 2] if dz else 1 - u[:, 2])
                out += w[:, None] * grid[base + dx * sx + dy * sy + dz]
    out[~inside] = 0
    return out.to(torch.float64)


def _deriv(tg, z, s, qk):
    import torch
    x, y, tx, ty = s
    B = _field_t(tg, x, y, torch.full_like(x, z))
    n = torch.sqrt(1 + tx * tx + ty * ty)
    a = qk * n
    dtx = a * (tx * ty * B[:, 0] - (1 + tx * tx) * B[:, 1] + ty * B[:, 2])
    dty = a * ((1 + ty * ty) * B[:, 0] - tx * ty * B[:, 1] - tx * B[:, 2])
    return torch.stack([tx, ty, dtx, dty]), torch.linalg.norm(B, dim=1)


def _rk4(field, x, y, tx, ty, qk, z0, z1, step, device=None):
    """RK4 (torch, CPU threads or GPU) from the common plane z0 to z1 (> z0). Returns numpy x, y, tx, ty and the
    max |B| met."""
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tg = _torch_grid(field, device)
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64, device=device)
    s = torch.stack([T(x), T(y), T(tx), T(ty)])
    qk = T(qk)
    bmax = torch.zeros(len(x), dtype=torch.float64, device=device)
    z = z0
    with torch.no_grad():
        while z < z1 - 1e-9:
            h = min(step, z1 - z)
            k1, b1 = _deriv(tg, z, s, qk)
            k2, b2 = _deriv(tg, z + h / 2, s + h / 2 * k1, qk)
            k3, b3 = _deriv(tg, z + h / 2, s + h / 2 * k2, qk)
            k4, b4 = _deriv(tg, z + h, s + h * k3, qk)
            s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            bmax = torch.maximum(bmax, torch.maximum(torch.maximum(b1, b2), torch.maximum(b3, b4)))
            z += h
    s = s.cpu().numpy()
    return s[0], s[1], s[2], s[3], bmax.cpu().numpy()


def to_z(pos, mom, q, z, step=STEP, field=None, device=None):
    """Propagate states pos (n, 3) [m], mom (n, 3) [GeV], charge q (n,) to the plane z (scalar, downstream of
    every pos). Straight outside the map's z range, RK4 inside it. Returns (pos (n, 3), mom (n, 3), lost (n,))."""
    pos, mom = np.asarray(pos, np.float64), np.asarray(mom, np.float64)
    q = np.asarray(q, np.float64)
    n = len(pos)
    out_pos, out_mom = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
    lost = np.ones(n, bool)
    ok = mom[:, 2] > 0
    out_pos[:, 2] = z
    if not ok.any():
        return out_pos, out_mom, lost
    field = field or field_map()
    zf0, zf1 = float(field.lo[2]), float(field.hi[2])
    for start in range(0, n, CHUNK):
        sl = np.arange(start, min(start + CHUNK, n))
        sl = sl[ok[sl]]
        if not len(sl):
            continue
        p, m, c = pos[sl], mom[sl], q[sl]
        pm = np.linalg.norm(m, axis=1)
        tx, ty = m[:, 0] / m[:, 2], m[:, 1] / m[:, 2]
        # straight to the common start plane z_s = max(z0, start of the map), then RK4 (charged) / straight (neutral)
        z_s = max(float(p[:, 2].max()), zf0)
        x, y = p[:, 0] + tx * (z_s - p[:, 2]), p[:, 1] + ty * (z_s - p[:, 2])
        bmax = np.zeros(len(sl))
        z_e = min(z, zf1)
        ch = c != 0
        if z_e > z_s and ch.any():
            qk = K_GEV * c[ch] / pm[ch]
            x[ch], y[ch], tx[ch], ty[ch], bmax[ch] = _rk4(field, x[ch], y[ch], tx[ch], ty[ch], qk, z_s, z_e, step, device)
        z_now = max(z_e, z_s)
        x, y = x + tx * (z - z_now), y + ty * (z - z_now)
        pz = pm / np.sqrt(1 + tx * tx + ty * ty)
        out_pos[sl, 0], out_pos[sl, 1] = x, y
        out_mom[sl] = np.stack([tx * pz, ty * pz, pz], 1)
        lost[sl] = (bmax > YOKE_B) | ~np.isfinite(x)
    return out_pos, out_mom, lost


def timing_acceptance(pos, mom, q, **kw):
    """Tracks (state downstream of the vessel, at or before SST1) that reach the timing detector inside its
    aperture without entering the magnet iron. Returns (mask, x_t, y_t)."""
    p_t, _, lost = to_z(pos, mom, q, geo.Z_TIMING, **kw)
    acc = ~lost & (np.abs(p_t[:, 0]) < geo.TIMING_HALF_X) & (np.abs(p_t[:, 1]) < geo.TIMING_HALF_Y)
    return acc, p_t[:, 0], p_t[:, 1]


def charge_of_pdg(pdg):
    """Charge of PDG codes (interactions.charge: 0 for neutral / unknown)."""
    from .interactions import charge
    return charge(np.asarray(pdg)).astype(np.float64)
