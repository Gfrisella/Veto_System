"""The veto shell: a closed surface around the decay vessel on which the final muon production should have statistics
everywhere (docs/STATUS.md, 2026-11-06 plan).

Shape: an elliptic arch around the vessel at MARGIN (half widths of the vessel frustum + MARGIN, widening behind the vessel to
clear the spectrometer), cut by the cavern floor and wall, from the shield end (Z0) on. Cells:
  side       NZ (along z) x NA (around the ellipse)
  up cap     NC x NC cells of the normalised ellipse at z = Z_STATE_A (30.6 m, where the stored state A is)
  down cap   NC x NC cells at z = Z_STATE_B (82 m, state B: the SST1 region)
The transport stores states at 30.6 and 82 m only, so the side cells used are those up to 82 m; the shell's z range runs on to
Z1 for a later third plane. A muon's path is A -> (bounce point K) -> B, straight pieces (as the fin hits of muon_types),
sampled every DZ; each inside / outside change is one crossing of a side cell.
Stopped muons (type 7: A is where they stopped) in the last metre of the shield are a separate class (LAST_METRE).
"""
import numpy as np

from . import geometry as geo

MARGIN = 1.5                         # m around the vessel / spectrometer
Z0, Z1 = geo.Z_SHIELD_END, 94.5     # shield end .. just past the timing detector
ZA, ZB = geo.Z_STATE_A, geo.Z_STATE_B
XS, YS, Z_SPEC = 2.7, 3.8, 84.98     # spectrometer half widths, its upstream end
NZ, NA, NC = 32, 16, 8
DZ = 0.5
LAST_METRE = (Z0 - 1.0, Z0)          # stopping points kept completely
N_SIDE, N_CAP = NZ * NA, NC * NC
OFF_UP, OFF_DOWN = N_SIDE, N_SIDE + N_CAP
N_CELLS = N_SIDE + 2 * N_CAP


def half(z, margin=MARGIN, xp=np):
    """half widths (x, y) of the shell at z"""
    hx, hy = geo.vessel_half_widths(z, clip=True)
    s = xp.clip((z - geo.Z_VESSEL_MAX) / (Z_SPEC - geo.Z_VESSEL_MAX), 0, 1)
    return hx + s * (XS - hx) + margin, hy + s * (YS - hy) + margin


def cell_names():
    """readable name of every cell (for the output files)"""
    zc = Z0 + (np.arange(NZ) + 0.5) * (Z1 - Z0) / NZ
    side = [f"side z={zc[i]:.1f} m, sector {j} ({(j + 0.5) * 360 / NA - 180:.0f} deg)" for i in range(NZ) for j in range(NA)]
    cap = lambda tag: [f"{tag} u={i} v={j}" for i in range(NC) for j in range(NC)]
    return side + cap(f"up cap z={ZA}") + cap(f"down cap z={ZB}")


def crossings_torch(a, k, b, margin=MARGIN):
    """Shell cells crossed by muons with states a (z = 30.6), b (z = 82 or where they stopped) and bounce point k (NaN = none),
    torch (n, 3) float64 tensors on any device. Returns (row, cell) index tensors: one entry per crossing / cap cell."""
    import torch
    dev = a.device
    kk = torch.where(torch.isfinite(k).all(1, keepdim=True), k, 0.5 * (a + b))
    rows, cells = [], []

    def cap(x, y, z, off):
        hx, hy = half(z, margin, torch)
        u = torch.clamp((x / hx + 1) / 2, 0, 0.999); v = torch.clamp((y / hy + 1) / 2, 0, 0.999)
        return off + (u * NC).long() * NC + (v * NC).long()

    za = torch.full_like(a[:, 0], ZA)
    hxa, hya = half(za, margin, torch)
    ins_a = (a[:, 0] / hxa) ** 2 + (a[:, 1] / hya) ** 2 < 1
    ia = torch.nonzero(ins_a).squeeze(1)
    rows.append(ia); cells.append(cap(a[ia, 0], a[ia, 1], za[ia], OFF_UP))
    zb = torch.full_like(b[:, 0], ZB)
    hxb, hyb = half(zb, margin, torch)
    ins_b = (b[:, 2] > ZB - 0.05) & ((b[:, 0] / hxb) ** 2 + (b[:, 1] / hyb) ** 2 < 1)
    ib = torch.nonzero(ins_b).squeeze(1)
    rows.append(ib); cells.append(cap(b[ib, 0], b[ib, 1], zb[ib], OFF_DOWN))
    prev = ins_a.clone()
    for z in np.arange(ZA, ZB + 1e-9, DZ)[1:]:
        f1 = torch.clamp((z - a[:, 2]) / torch.clamp(kk[:, 2] - a[:, 2], min=1e-9), 0, 1)
        f2 = torch.clamp((z - kk[:, 2]) / torch.clamp(b[:, 2] - kk[:, 2], min=1e-9), 0, 1)
        first = z <= kk[:, 2]
        x = torch.where(first, a[:, 0] + f1 * (kk[:, 0] - a[:, 0]), kk[:, 0] + f2 * (b[:, 0] - kk[:, 0]))
        y = torch.where(first, a[:, 1] + f1 * (kk[:, 1] - a[:, 1]), kk[:, 1] + f2 * (b[:, 1] - kk[:, 1]))
        alive = b[:, 2] >= z - 1e-6                    # a muon that stopped before z has left the path
        hx, hy = half(torch.tensor(z, dtype=torch.float64, device=dev), margin, torch)
        ins = (x / hx) ** 2 + (y / hy) ** 2 < 1
        cr = torch.nonzero(alive & (ins != prev)).squeeze(1)
        if len(cr):
            phi = torch.atan2(y[cr] / hy, x[cr] / hx)
            sec = ((phi + np.pi) / (2 * np.pi) * NA).long() % NA
            iz = min(NZ - 1, int((z - Z0) / (Z1 - Z0) * NZ))
            rows.append(cr); cells.append(iz * NA + sec)
        prev = torch.where(alive, ins, prev)
    return torch.cat(rows), torch.cat(cells)
