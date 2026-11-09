#!/usr/bin/env python3
"""2D maps of the target / shield neutrino production (bin/nu_prod.sbatch) for the interactive flux page: per component
(min bias, charm, beauty), flavour and energy bin, sum of weights and sum of squared weights per cell (per proton on
target), in four views: production point (x vs z), and x vs y on straight lines at z = 0 (after the target), the vessel
entrance and SST1. Writes <out>.bin (float32, gzip) + <out>.json (layout).

    python3 bin/nu_maps.py /disk/lhcb_data/sqasim/nu_prod/tail --out /disk/lhcb_data/sqasim/nu_prod/nu_maps
"""
import _bootstrap  # noqa: F401
import argparse
import glob
import gzip
import json

import numpy as np
from veto import geometry as G

EB = [1, 2, 5, 10, 20, 50, 100, 400.0]
FLAV = [14, -14, 12, -12, 16, -16]
COMP = [("min bias", "nuMB", True), ("charm", "charmc", False), ("beauty", "beautyc", False)]
NB = 60
VIEWS = [  # name, kind, z of the plane, (u range), (v range): u horizontal, v vertical
    dict(name="production point", kind="xz", z=None, u=(-2.4, 0.0), v=(-0.3, 0.3), ul="z [m]", vl="x [m]"),
    dict(name="after the target, z = 0 m", kind="xy", z=0.0, u=(-0.6, 0.6), v=(-0.6, 0.6), ul="x [m]", vl="y [m]"),
    dict(name=f"vessel entrance, z = {G.Z_VESSEL_MIN:.2f} m", kind="xy", z=G.Z_VESSEL_MIN, u=(-6, 6), v=(-6, 6),
         ul="x [m]", vl="y [m]", box=(G.VESSEL_X_IN, G.VESSEL_Y_IN)),
    dict(name=f"SST1, z = {G.Z_SST1:.2f} m", kind="xy", z=G.Z_SST1, u=(-12, 12), v=(-12, 12), ul="x [m]", vl="y [m]",
         box=(G.SST1_HALF_X, G.SST1_HALF_Y)),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    nv, nc, nf, ne = len(VIEWS), len(COMP), len(FLAV), len(EB) - 1
    W = np.zeros((nv, nc, nf, ne, 2, NB * NB))               # [.., 0] sum w, [.., 1] sum w^2
    tot = np.zeros((nc, nf, ne, 3))                          # sum w, sum w^2, rows (all, inside or not)
    prot = np.zeros(nc)
    for ic, (cname, tag, drop) in enumerate(COMP):
        files = sorted(glob.glob(f"{a.dir}/{tag}_nu_*.npz"))
        for f in files:
            z = np.load(f)
            prot[ic] += float(z["events"]) * float(z["rep"])
            m = (z["hf"] == 0) if drop else np.ones(len(z["pdg"]), bool)
            pdg, w = z["pdg"][m], z["w_decay"][m]
            x0, y0, z0 = z["x"][m], z["y"][m], z["z"][m]
            px, py, pz = z["px"][m], z["py"][m], z["pz"][m]
            E = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
            ie = np.digitize(E, EB) - 1
            fi = np.full(len(pdg), -1)
            for k, fl in enumerate(FLAV):
                fi[pdg == fl] = k
            ok = (fi >= 0) & (ie >= 0) & (ie < ne) & (pz > 0)
            key = fi * ne + ie
            for q, ww in enumerate((w, w * w, np.ones_like(w))):
                tot[ic, :, :, q] += np.bincount(key[ok], ww[ok], nf * ne).reshape(nf, ne)
            for iv, v in enumerate(VIEWS):
                if v["kind"] == "xz":
                    u, vv = z0, x0
                else:
                    t = (v["z"] - z0) / pz
                    u, vv = x0 + px * t, y0 + py * t
                iu = np.floor((u - v["u"][0]) / (v["u"][1] - v["u"][0]) * NB).astype(int)
                jv = np.floor((vv - v["v"][0]) / (v["v"][1] - v["v"][0]) * NB).astype(int)
                s = ok & (iu >= 0) & (iu < NB) & (jv >= 0) & (jv < NB)
                cell = jv * NB + iu
                kk = key[s] * NB * NB + cell[s]
                for q, ww in enumerate((w, w * w)):
                    W[iv, ic, :, :, q] += np.bincount(kk, ww[s], nf * ne * NB * NB).reshape(nf, ne, NB * NB)
        print(f"{cname}: {len(files)} files, {prot[ic]:.4g} protons", flush=True)
    W /= np.maximum(prot, 1)[None, :, None, None, None, None] ** np.array([1, 2])[None, None, None, None, :, None]
    tot[..., 0] /= prot[:, None, None]
    tot[..., 1] /= prot[:, None, None] ** 2
    with gzip.open(a.out + ".bin.gz", "wb", compresslevel=9) as g:
        g.write(W.astype(np.float32).tobytes())
    meta = dict(nb=NB, eb=EB, flav=FLAV, comp=[c[0] for c in COMP], protons=prot.tolist(),
                views=[{k: v for k, v in vw.items()} for vw in VIEWS], shape=list(W.shape), spill=4e13,
                totals=tot.tolist())
    json.dump(meta, open(a.out + ".json", "w"))
    print("written", a.out + ".bin.gz", a.out + ".json")


if __name__ == "__main__":
    main()
