#!/usr/bin/env python3
"""Compact sample of the shield-end EM debris (bin/em_debris.py raw crossings: e+- / gamma > 1 MeV on the planes
z = Z_UBT + 0 .. 10 m) for the interactive debris page. Per debris track: its point on the shield end face (the first
crossing projected straight back to z = shield_iron.Z_PLANE; tracks made downstream, e.g. conversion electrons, take the
point of their origin particle), species, energy band (|p| at the first crossing) and its crossing on every plane.
Also the full-statistics face map (debris crossing the first plane, by face cell) and the geometry of a design.

    bin/lcg.sh python3 bin/em_viz_data.py <em_raw.npz> --design designs/modules/D6_straw_mod.json --out <dir>/em_viz
Writes <out>.bin.gz (little-endian int16 / uint8 blocks, layout in the json) and <out>.json.
"""
import _bootstrap  # noqa: F401
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from veto import design as dsg
from veto import em_debris as em
from veto import geometry as G
from veto import shield_iron as si
from veto import weights
from veto.config import DEFAULT_CONFIG, POT_PER_SPILL

BANDS = [1, 3, 10, 30, 100, 1e6]          # MeV, |p| at the first crossing
SPECIES = {22: 0, 11: 1, -11: 2}           # gamma, e-, e+
FACE_X, FACE_Y, CELL = 2.0, 1.5, 0.25      # face map |x| < FACE_X, |y| < FACE_Y, cell size [m]
MISSING = -32768


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw"); ap.add_argument("--design", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-tracks", type=int, default=150_000, help="tracks in the sample sent to the page")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    z = np.load(a.raw)
    fs = POT_PER_SPILL / weights.source_pot(path=a.raw)                 # D13: legacy raw files are per 5e13
    zpl = np.asarray(em.Z_PLANES, float)
    ev, tr, org, pl, pdg = (z[k].astype(np.int64) for k in ("ev", "track", "origin", "plane", "pdg"))
    x, y, px, py, pz = (z[k].astype(np.float64) for k in ("x", "y", "px", "py", "pz"))
    w_mu = z["w_mu"].astype(np.float64) * fs
    # tracks: (ev, track); first crossing of each
    key = ev * 1_000_000 + tr
    o = np.lexsort((pl, key))
    key, ev, tr, org, pl, pdg, x, y, px, py, pz = (v[o] for v in (key, ev, tr, org, pl, pdg, x, y, px, py, pz))
    first = np.r_[True, key[1:] != key[:-1]]
    tid = np.cumsum(first) - 1
    nt = int(tid[-1]) + 1
    f = np.nonzero(first)[0]
    p_first = np.sqrt(px[f] ** 2 + py[f] ** 2 + pz[f] ** 2)
    zf = zpl[pl[f]]
    fx = x[f] - px[f] / pz[f] * (zf - si.Z_PLANE)
    fy = y[f] - py[f] / pz[f] * (zf - si.Z_PLANE)
    t_ev, t_org, t_pl0, t_pdg = ev[f], org[f], pl[f], pdg[f]
    # face point of a group (ev, origin): its most energetic track crossing the first plane
    gkey = t_ev * 100_000 + np.where(t_org >= 0, t_org, 99_999)
    cand = np.nonzero((t_pl0 == 0) & (t_org >= 0) & (pz[f] > 0))[0]
    oc = cand[np.lexsort((-p_first[cand], gkey[cand]))]
    gfirst = np.r_[True, gkey[oc][1:] != gkey[oc][:-1]]
    gk, gi = gkey[oc][gfirst], oc[gfirst]
    pos = np.searchsorted(gk, gkey)
    has = (pos < len(gk)) & (gk[np.minimum(pos, len(gk) - 1)] == gkey)
    src = np.where(has, gi[np.minimum(pos, len(gk) - 1)], -1)
    ffx = np.where(src >= 0, fx[np.maximum(src, 0)], np.nan)
    ffy = np.where(src >= 0, fy[np.maximum(src, 0)], np.nan)
    sp = np.full(nt, 255, np.uint8)
    for code, s in SPECIES.items():
        sp[t_pdg == code] = s
    band = np.clip(np.digitize(p_first * 1e3, BANDS) - 1, 0, len(BANDS) - 2).astype(np.uint8)
    ok = (src >= 0) & (sp != 255) & np.isfinite(ffx)
    w_tr = w_mu[t_ev]
    print(f"{nt:,} tracks, {ok.sum():,} with a face point and e / gamma ({w_tr[ok].sum():.4g} per spill)")
    # full-statistics face map: tracks crossing the first plane, per face cell, species, band
    nx, ny = int(round(2 * FACE_X / CELL)), int(round(2 * FACE_Y / CELL))
    ix = np.floor((ffx + FACE_X) / CELL).astype(np.int64)
    iy = np.floor((ffy + FACE_Y) / CELL).astype(np.int64)
    inface = ok & (t_pl0 == 0) & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    nb = len(BANDS) - 1
    k = ((sp[inface].astype(np.int64) * nb + band[inface]) * ny + iy[inface]) * nx + ix[inface]
    face = np.bincount(k, w_tr[inface], 3 * nb * ny * nx).reshape(3, nb, ny, nx)
    print(f"face map: {face.sum():.4g} per spill crossing z = {zpl[0]:.2f} m from |x| < {FACE_X}, |y| < {FACE_Y}")
    # sample of tracks for rays and plane maps
    rng = np.random.default_rng(a.seed)
    good = np.nonzero(ok)[0]
    take = np.sort(rng.choice(good, min(a.n_tracks, len(good)), replace=False))
    w_sample = float(w_tr[good].sum() / len(take))                       # per spill per sampled track
    np_ = len(zpl)
    P = np.full((len(take), np_, 2), MISSING, np.int16)
    row = np.full(nt, -1); row[take] = np.arange(len(take))
    m = row[tid] >= 0
    r, q = row[tid][m], pl[m]
    P[r, q, 0] = np.clip(np.round(x[m] * 100), -32767, 32767)
    P[r, q, 1] = np.clip(np.round(y[m] * 100), -32767, 32767)
    face_xy = np.stack([np.round(ffx[take] * 1000), np.round(ffy[take] * 1000)], 1).astype(np.int16)   # mm
    logE = np.clip(np.round((np.log10(p_first[take] * 1e3)) * 40), 0, 255).astype(np.uint8)           # 0.025 dex
    blocks = [("face_xy", face_xy), ("planes_xy", P), ("species", sp[take]), ("band", band[take]), ("logE", logE)]
    out = Path(a.out)
    layout, off = {}, 0
    with gzip.open(str(out) + ".bin.gz", "wb", compresslevel=9) as g:
        for name, arr in blocks:
            b = np.ascontiguousarray(arr).astype(arr.dtype.newbyteorder("<")).tobytes()
            layout[name] = dict(offset=off, dtype=str(arr.dtype), shape=list(arr.shape)); off += len(b)
            pad = (-len(b)) % 4
            g.write(b + b"\0" * pad); off += pad
    # geometry
    d = dsg.load(a.design)
    planes = []
    for p in d.planes:
        x0, x1 = sorted([float(p.to_global(0.0)), float(p.to_global(p.u_max))])
        planes.append(dict(name=p.name, kind=p.kind, z=float(p.z), x0=x0, x1=x1, y0=float(p.y_lo), y1=float(p.y_hi)))
    zs = np.linspace(G.Z_VESSEL_MIN, G.Z_VESSEL_MAX, 11)
    hx, hy = G.vessel_half_widths(zs)
    meta = dict(source=str(a.raw), design=d.name, pot_per_spill=POT_PER_SPILL, spill_s=DEFAULT_CONFIG.get("spill_s"),
                n_muons=int(len(w_mu)), muons_per_spill=float(w_mu.sum()), e_cut_mev=em.E_CUT * 1e3,
                n_tracks_total=int(ok.sum()), per_spill_total=float(w_tr[ok].sum()), n_sample=int(len(take)),
                w_sample=w_sample, bands_mev=BANDS, species=["gamma", "e-", "e+"], z_planes=zpl.tolist(),
                z_face=si.Z_PLANE, face=dict(half_x=FACE_X, half_y=FACE_Y, cell=CELL, nx=nx, ny=ny, map=face.tolist()),
                shield=dict(z0=si.Z_IRON0, z1=si.Z_SHIELD_END, half_x=si.HALF_X, half_y=si.HALF_Y),
                vessel=dict(z=zs.tolist(), hx=np.asarray(hx).tolist(), hy=np.asarray(hy).tolist()),
                ubt=dict(z=G.Z_UBT, half_x=G.UBT_HALF_X, half_y=G.UBT_HALF_Y), design_planes=planes,
                layout=layout, missing=MISSING, units=dict(face_xy="mm", planes_xy="cm", logE="40 * log10(MeV)"))
    json.dump(meta, open(str(out) + ".json", "w"))
    print(f"wrote {out}.bin.gz ({off / 1e6:.1f} MB raw), {out}.json; {len(take):,} tracks, {w_sample:.4g} per spill each")


if __name__ == "__main__":
    main()
