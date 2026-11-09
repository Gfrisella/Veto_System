#!/usr/bin/env python3
"""Muons through the decay-vessel surface (and the shell, for comparison) from a --storage shell transport, per cell, for all
muons and for p > 1 GeV at the crossing (stored rows, Horvitz-Thompson weights; each seed = one spill).

Vessel: the frustum 31.13 - 81.13 m (half widths 0.5 -> 2.0 m in x, 1.35 -> 3.0 m in y), faces +x, -x, +y, -y in NZV x NU cells
(z x across the face), the entrance window (z = 31.13, NE x NE cells, where the UBT is) and the exit window (z = 81.13).
Path of a muon: 30.6 m -> (bounce point) -> 82 m (or where it stopped), straight pieces sampled every DZ; every change
inside / outside the frustum between the windows is a side-face crossing (in or out); a path inside at a window is a window
crossing. Momentum at a crossing: the lower of the two stored momenta (no energy loss in the cavern air in this transport).

    python3 bin/vessel_surface_map.py <seed dir> [<seed dir> ...] --out vessel.json [--max-seeds 600 --gpu 0]
"""
import _bootstrap  # noqa: F401
import argparse
import glob
import json

import numpy as np
import torch

from veto import geometry as geo
from veto import muon_types as mt
from veto import shell as sh

V0, V1 = geo.Z_VESSEL_MIN, geo.Z_VESSEL_MAX
NZV, NU, NE = 25, 8, 8
DZ = 0.25
FACES = ("+x", "-x", "+y", "-y")
N_SIDE = 4 * NZV * NU
OFF_IN, OFF_OUT = N_SIDE, N_SIDE + NE * NE
N_CELLS = N_SIDE + 2 * NE * NE


def vhalf(z, xp=np):
    return geo.vessel_half_widths(z, clip=True)


def vessel_crossings(a, k, b):
    """(row, cell) of vessel-surface crossings for paths a -> k -> b (torch (n, 3) float64)"""
    kk = torch.where(torch.isfinite(k).all(1, keepdim=True), k, 0.5 * (a + b))
    rows, cells = [], []

    def at(z):
        f1 = torch.clamp((z - a[:, 2]) / torch.clamp(kk[:, 2] - a[:, 2], min=1e-9), 0, 1)
        f2 = torch.clamp((z - kk[:, 2]) / torch.clamp(b[:, 2] - kk[:, 2], min=1e-9), 0, 1)
        first = z <= kk[:, 2]
        x = torch.where(first, a[:, 0] + f1 * (kk[:, 0] - a[:, 0]), kk[:, 0] + f2 * (b[:, 0] - kk[:, 0]))
        y = torch.where(first, a[:, 1] + f1 * (kk[:, 1] - a[:, 1]), kk[:, 1] + f2 * (b[:, 1] - kk[:, 1]))
        return x, y

    def window(z, off):
        x, y = at(z)
        hx, hy = vhalf(torch.tensor(z, dtype=torch.float64, device=a.device), torch)
        ins = (b[:, 2] >= z - 1e-6) & (x.abs() < hx) & (y.abs() < hy)
        r = torch.nonzero(ins).squeeze(1)
        u = torch.clamp(((x[r] / hx + 1) / 2 * NE).long(), 0, NE - 1); v = torch.clamp(((y[r] / hy + 1) / 2 * NE).long(), 0, NE - 1)
        rows.append(r); cells.append(off + u * NE + v)

    window(V0, OFF_IN)
    window(V1, OFF_OUT)
    zs = np.arange(V0, V1 + 1e-9, DZ)
    x, y = at(float(zs[0]))
    hx, hy = vhalf(torch.tensor(float(zs[0]), dtype=torch.float64, device=a.device), torch)
    prev = (x.abs() < hx) & (y.abs() < hy)
    for z in zs[1:]:
        x, y = at(float(z))
        hx, hy = vhalf(torch.tensor(float(z), dtype=torch.float64, device=a.device), torch)
        alive = b[:, 2] >= z - 1e-6
        ins = (x.abs() < hx) & (y.abs() < hy)
        cr = torch.nonzero(alive & (ins != prev)).squeeze(1)
        if len(cr):
            rx, ry = x[cr] / hx, y[cr] / hy                       # which face: the larger normalised coordinate
            xface = rx.abs() >= ry.abs()
            face = torch.where(xface, torch.where(rx > 0, 0, 1), torch.where(ry > 0, 2, 3))
            u = torch.where(xface, ry, rx)                        # position across the face, -1 .. 1
            iu = torch.clamp(((u + 1) / 2 * NU).long(), 0, NU - 1)
            iz = min(NZV - 1, int((z - V0) / (V1 - V0) * NZV))
            rows.append(cr); cells.append((face * NZV + iz) * NU + iu)
        prev = torch.where(alive, ins, prev)
    return torch.cat(rows), torch.cat(cells)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--max-seeds", type=int, default=600); ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    files = sorted(sum((glob.glob(f"{d}/seed_*.npz") for d in a.dirs), []))[:a.max_seeds]
    FILTERS = ("all", "sst1", "sst1_bounced", "veto_only", "bounced")   # by the muon's fate (muon_types) / a bounce point
    KEYS = [f"{f}|{c}" for f in FILTERS for c in ("all", "p1")]
    V = {c: np.zeros((3, N_CELLS)) for c in KEYS}          # sum w, sum w^2, rows
    SH = {c: np.zeros((3, sh.N_CELLS)) for c in KEYS}
    for n, f in enumerate(files):
        d = dict(np.load(f))
        r = np.nonzero((d["type"] <= 6) & (d["z_30"] > mt.REACH_30))[0]
        g = lambda t: torch.from_numpy(np.stack([d[f"{c}_{t}"][r] for c in "xyz"], 1).astype(np.float64)).to(dev)
        k = torch.from_numpy(np.stack([d["kx"][r], d["ky"][r], d["kz"][r]], 1).astype(np.float64)).to(dev)
        A, B = g("30"), g("82")
        pa = np.sqrt(d["px_30"].astype(float) ** 2 + d["py_30"] ** 2 + d["pz_30"] ** 2)
        pb = np.sqrt(d["px_82"].astype(float) ** 2 + d["py_82"] ** 2 + d["pz_82"] ** 2)
        pc = np.minimum(pa, pb)
        w = d["weight"].astype(np.float64)
        t = d["type"]
        fmask = {"all": np.ones(len(t), bool), "sst1": t <= 2, "sst1_bounced": t == 0, "veto_only": t == 3,
                 "bounced": np.isfinite(d["kz"])}
        for fn, store, nc in ((vessel_crossings, V, N_CELLS), (sh.crossings_torch, SH, sh.N_CELLS)):
            i, c = fn(A, k, B)
            i, c = r[i.cpu().numpy()], c.cpu().numpy()
            for fl in FILTERS:
                for cut, m in (("all", fmask[fl][i]), ("p1", fmask[fl][i] & (pc[i] > 1.0))):
                    key = f"{fl}|{cut}"
                    store[key][0] += np.bincount(c[m], w[i][m], nc)
                    store[key][1] += np.bincount(c[m], w[i][m] ** 2, nc)
                    store[key][2] += np.bincount(c[m], minlength=nc)
        if n % 50 == 0:
            print(f"{n + 1}/{len(files)} seeds", flush=True)
    S = len(files)
    out = dict(seeds=S, NZV=NZV, NU=NU, NE=NE, V0=V0, V1=V1, faces=FACES,
               vessel={c: dict(rate=(V[c][0] / S).tolist(), neff=np.where(V[c][1] > 0, V[c][0] ** 2 / np.where(V[c][1] > 0, V[c][1], 1), 0).tolist(),
                               n=V[c][2].astype(int).tolist()) for c in V},
               shell={c: dict(rate=(SH[c][0] / S).tolist(), neff=np.where(SH[c][1] > 0, SH[c][0] ** 2 / np.where(SH[c][1] > 0, SH[c][1], 1), 0).tolist(),
                              n=SH[c][2].astype(int).tolist()) for c in SH})
    for c in KEYS:
        rr = np.array(out["vessel"][c]["rate"])[:N_SIDE].reshape(4, NZV, NU).sum((1, 2))
        print(f"vessel faces [{c:20s}] " + ", ".join(f"{FACES[j]} {rr[j]:9.3g}" for j in range(4)) +
              f"; -x / +x = {rr[1] / max(rr[0], 1e-300):6.2f}; entrance {sum(out['vessel'][c]['rate'][OFF_IN:OFF_OUT]):9.3g}")
    json.dump(out, open(a.out, "w"))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
