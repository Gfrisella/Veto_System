#!/usr/bin/env python3
"""Compact sample of the EM debris out of the cavern rock (bin/rock_apply.py --em-out: e+- > 1 MeV, gamma > 3 MeV leaving
the -x wall and the floor, from the muons running into the rock) for the interactive debris page. Per particle: where it
leaves the rock (wall: z, y; floor: z, x), species, energy band, its straight line to where it leaves the cavern (ECN3 box)
or, for e+-, the end of its range in air, and its crossings of the design's plane z values and SST1 (both directions).
Also the full-statistics source map and the geometry.

    bin/lcg.sh python3 bin/em_viz_rock_data.py <rock_em.npz> --design designs/modules/D6_straw_mod.json --out <dir>/em_rock
"""
import _bootstrap  # noqa: F401
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from veto import design as dsg
from veto import geometry as G

BANDS = [1, 3, 10, 30, 100, 1e6]               # MeV kinetic
SPECIES = {22: 0, 11: 1, -11: 2}
CAVERN = dict(x0=G.CAVERN_X_WALL, x1=11.43, y0=G.CAVERN_Y_FLOOR, y1=10.64, z0=18.378, z1=100.0)   # ECN3 (MuonsAndMatter)
DEDX_AIR = 0.22e-3                               # GeV / m, e+- in air (minimum ionising, 1.8 MeV cm2/g x 1.2e-3 g/cm3)
SRC = dict(z0=29.0, z1=83.0, dz=1.0, wall_y1=4.0, floor_x1=6.0, dv=0.5)    # source map: z cells x (wall y | floor x) cells
MISSING = -32768


def ray_end(pos, d):
    """Path length to where the straight line leaves the cavern box."""
    s = np.full(len(pos), np.inf)
    for k, (lo, hi) in enumerate(((CAVERN["x0"], CAVERN["x1"]), (CAVERN["y0"], CAVERN["y1"]), (CAVERN["z0"], CAVERN["z1"]))):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(d[:, k] > 0, (hi - pos[:, k]) / d[:, k], np.where(d[:, k] < 0, (lo - pos[:, k]) / d[:, k], np.inf))
        s = np.minimum(s, np.where(t > 0, t, np.inf))
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("em"); ap.add_argument("--design", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-tracks", type=int, default=150_000); ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    z = np.load(a.em)
    ns = int(z["n_seeds"])
    w = z["w"].astype(np.float64) / ns                                    # per spill (4e13)
    pdg = z["pdg"].astype(np.int64)
    pos = np.stack([z["x"], z["y"], z["z"]], 1).astype(np.float64)
    mom = np.stack([z["px"], z["py"], z["pz"]], 1).astype(np.float64)
    ek = z["ekin"].astype(np.float64)
    surf = z["surf"].astype(np.int64)                                     # 0 wall (x = CAVERN_X_WALL), 1 floor
    pm = np.linalg.norm(mom, axis=1)
    d = mom / np.maximum(pm, 1e-12)[:, None]
    into = np.where(surf == 0, d[:, 0] > 0, d[:, 1] > 0)                 # leaving the rock into the cavern
    sp = np.full(len(pdg), 255, np.uint8)
    for c, s in SPECIES.items():
        sp[pdg == c] = s
    ok = into & (sp != 255) & (pos[:, 2] > SRC["z0"]) & (pos[:, 2] < SRC["z1"])
    band = np.clip(np.digitize(ek * 1e3, BANDS) - 1, 0, len(BANDS) - 2).astype(np.uint8)
    s_end = ray_end(pos, d)
    rng_e = np.where(sp > 0, ek / DEDX_AIR, np.inf)                       # electrons stop in the air
    s_end = np.minimum(s_end, rng_e)
    print(f"{len(pdg):,} e / gamma rows, {ok.sum():,} into the cavern ({w[ok].sum():.4g} per spill, {ns} seeds); "
          f"gamma {w[ok & (sp == 0)].sum():.3g}, e- {w[ok & (sp == 1)].sum():.3g}, e+ {w[ok & (sp == 2)].sum():.3g}")
    # full-statistics source map: surface (2) x species (3) x band (5) x v cells x z cells
    nz = int(round((SRC["z1"] - SRC["z0"]) / SRC["dz"]))
    nvw = int(round((SRC["wall_y1"] - G.CAVERN_Y_FLOOR) / SRC["dv"]))
    nvf = int(round((SRC["floor_x1"] - G.CAVERN_X_WALL) / SRC["dv"]))
    nv = max(nvw, nvf)
    iz = np.floor((pos[:, 2] - SRC["z0"]) / SRC["dz"]).astype(np.int64)
    v = np.where(surf == 0, pos[:, 1] - G.CAVERN_Y_FLOOR, pos[:, 0] - G.CAVERN_X_WALL)
    iv = np.floor(v / SRC["dv"]).astype(np.int64)
    inmap = ok & (iz >= 0) & (iz < nz) & (iv >= 0) & (iv < np.where(surf == 0, nvw, nvf))
    nb = len(BANDS) - 1
    key = (((surf[inmap] * 3 + sp[inmap]) * nb + band[inmap]) * nv + iv[inmap]) * nz + iz[inmap]
    smap = np.bincount(key, w[inmap], 2 * 3 * nb * nv * nz).reshape(2, 3, nb, nv, nz)
    # sample (uniform in weight: draws within a seed share one weight, so sample proportional to w)
    rr = np.random.default_rng(a.seed)
    good = np.nonzero(ok)[0]
    pw = w[good] / w[good].sum()
    take = np.sort(rr.choice(good, min(a.n_tracks, len(good)), replace=False, p=pw))
    # importance correction: sampled ~ w without replacement ~ uniform weight per sampled particle
    w_sample = float(w[good].sum() / len(take))
    d_ = d[take]; p_ = pos[take]; se = s_end[take]
    zs = sorted({round(float(p.z), 2) for p in dsg.load(a.design).planes}) + [round(G.Z_SST1, 2)]
    C = np.full((len(take), len(zs), 2), MISSING, np.int16)
    for k, zz in enumerate(zs):
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (zz - p_[:, 2]) / d_[:, 2]
        hit = (s > 0) & (s < se) & np.isfinite(s)
        C[hit, k, 0] = np.clip(np.round((p_[hit, 0] + s[hit] * d_[hit, 0]) * 100), -32767, 32767)
        C[hit, k, 1] = np.clip(np.round((p_[hit, 1] + s[hit] * d_[hit, 1]) * 100), -32767, 32767)
    end = p_ + d_ * np.minimum(se, 200.0)[:, None]
    src = np.stack([np.round(p_[:, 0] * 100), np.round(p_[:, 1] * 100), np.round(p_[:, 2] * 100)], 1).astype(np.int16)
    endc = np.clip(np.round(end * 100), -32767, 32767).astype(np.int16)
    logE = np.clip(np.round(np.log10(ek[take] * 1e3) * 40), 0, 255).astype(np.uint8)
    blocks = [("src_xyz", src), ("end_xyz", endc), ("cross_xy", C), ("surf", surf[take].astype(np.uint8)),
              ("species", sp[take]), ("band", band[take]), ("logE", logE)]
    out = Path(a.out)
    layout, off = {}, 0
    with gzip.open(str(out) + ".bin.gz", "wb", compresslevel=9) as g:
        for name, arr in blocks:
            b = np.ascontiguousarray(arr).astype(arr.dtype.newbyteorder("<")).tobytes()
            layout[name] = dict(offset=off, dtype=str(arr.dtype), shape=list(arr.shape)); off += len(b)
            pad = (-len(b)) % 4
            g.write(b + b"\0" * pad); off += pad
    des = dsg.load(a.design)
    planes = []
    for p in des.planes:
        x0, x1 = sorted([float(p.to_global(0.0)), float(p.to_global(p.u_max))])
        planes.append(dict(name=p.name, kind=p.kind, z=float(p.z), x0=x0, x1=x1, y0=float(p.y_lo), y1=float(p.y_hi)))
    zv = np.linspace(G.Z_VESSEL_MIN, G.Z_VESSEL_MAX, 11)
    hx, hy = G.vessel_half_widths(zv)
    meta = dict(source=str(a.em), design=des.name, n_seeds=ns, pot_per_spill=4e13, spill_s=1.0,
                n_total=int(ok.sum()), per_spill_total=float(w[ok].sum()), n_sample=int(len(take)), w_sample=w_sample,
                thresholds_mev=dict(e=1.0, gamma=3.0), bands_mev=BANDS, species=["gamma", "e-", "e+"], cross_z=zs,
                src=dict(SRC, nz=nz, nv=nv, nv_wall=nvw, nv_floor=nvf, map=smap.tolist()), cavern=CAVERN,
                vessel=dict(z=zv.tolist(), hx=np.asarray(hx).tolist(), hy=np.asarray(hy).tolist()), design_planes=planes,
                sst1=dict(z=G.Z_SST1, half_x=G.SST1_HALF_X, half_y=G.SST1_HALF_Y), dedx_air=DEDX_AIR,
                layout=layout, missing=MISSING, units=dict(src_xyz="cm", end_xyz="cm", cross_xy="cm", logE="40 * log10(MeV)"))
    json.dump(meta, open(str(out) + ".json", "w"))
    print(f"wrote {out}.bin.gz ({off / 1e6:.1f} MB raw), {out}.json; {len(take):,} particles, {w_sample:.4g} per spill each; "
          f"crossing planes z = {zs}")


if __name__ == "__main__":
    main()
