#!/usr/bin/env python3
"""Controlled restarts (sandbox): sensitivity of SST1 arrival to the direction at a checkpoint. NOT probability estimates of the
beam: each state is restarted M times with its own direction and with injected angle offsets, physics unchanged.

    python3 restart_test.py --tracks tracks.npz --classes diag_classes.npz --ref ref.npz --input chunk.h5 --out restart [--m 500]

States: trajectories that reached isolated / low-occupancy / common SST1 regions (diag_angles.py classes), taken at the
checkpoints 19.41 (mid-shield), 25.70 and 29.89 m (shield exit). Offsets: 0, +-1, +-5, +-10, +-20 mrad on the x slope and,
separately, the y slope; position and |p| fixed. Output per (state, offset): survival to SST1, arrival x, y, p, SST1 class.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import splitting as SP                                                    # noqa: E402

OFFSETS = [0, 1, -1, 5, -5, 10, -10, 20, -20]          # mrad
CK = {"19.41": 4, "25.70": 6, "29.89": 7}               # checkpoint index in the tracks array


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("tracks", "classes", "ref", "input", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--m", type=int, default=500)
    ap.add_argument("--per-class", type=int, default=2, help="trajectories per SST1 class")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(9)
    T, C = np.load(a.tracks), np.load(a.classes)
    tr, q, cls = T["tracks"], T["q"].astype(np.int8), C["cls"]
    R = np.load(a.ref)
    XY = R["XY"]; r = int(round(1.0 / (XY[1] - XY[0]))); e = XY[::r]
    occ = R["xy_n"][-1].sum(0).reshape(len(e) - 1, r, len(e) - 1, r).sum((1, 3))
    picks = []
    for c in (2, 1, 0):                                  # isolated, low, common
        idx = np.nonzero(cls == c)[0]
        for i in rng.choice(idx, min(a.per_class, len(idx)), replace=False):
            picks.append((int(i), int(c)))
    sim = SP.G.Simulation(SP.G.MM_DIR, "TRY6", a.gpu, 1, a.input)
    rows, meta = [], []
    for i, c in picks:
        for zname, k in CK.items():
            st = tr[k, i]
            if not np.isfinite(st[0]):
                continue
            x, y, z, px, py, pz = (float(v) for v in st)
            p = np.sqrt(px * px + py * py + pz * pz)
            for axis in ("x", "y"):
                for d in OFFSETS:
                    if d == 0 and axis == "y":
                        continue
                    tx, ty = px / pz + (1e-3 * d if axis == "x" else 0), py / pz + (1e-3 * d if axis == "y" else 0)
                    pz2 = p / np.sqrt(1 + tx * tx + ty * ty)
                    m = a.m
                    P = dict(x=np.full(m, x, np.float32), y=np.full(m, y, np.float32), z=np.full(m, z, np.float32),
                             px=np.full(m, tx * pz2, np.float32), py=np.full(m, ty * pz2, np.float32), pz=np.full(m, pz2, np.float32),
                             q=np.full(m, q[i], np.int8), w=np.ones(m), root=np.arange(m), id=np.arange(m), parent=np.full(m, -1),
                             depth=np.zeros(m, np.int16), batch=np.zeros(m, np.int16))
                    out, _, _, _ = SP.advance(sim, P, SP.Z_SST, int(rng.integers(1 << 30)))
                    ok = np.zeros(m, bool); ok[out["root"]] = True
                    ax_ = np.full(m, np.nan); ay_ = np.full(m, np.nan); ap_ = np.full(m, np.nan)
                    ax_[out["root"]], ay_[out["root"]] = out["x"], out["y"]
                    ap_[out["root"]] = np.sqrt(out["px"].astype(float) ** 2 + out["py"] ** 2 + out["pz"] ** 2)
                    jx, jy = np.digitize(ax_, e) - 1, np.digitize(ay_, e) - 1
                    ins = ok & (jx >= 0) & (jx < len(e) - 1) & (jy >= 0) & (jy < len(e) - 1)
                    o = np.where(ins, occ[np.clip(jx, 0, len(e) - 2), np.clip(jy, 0, len(e) - 2)], 0)
                    acl = np.where(~ok, -1, np.where(o >= 1000, 0, np.where(o >= 10, 1, 2)))
                    rows.append(np.stack([ax_, ay_, ap_, acl.astype(float)], 1))
                    meta.append(dict(traj=i, cls=c, z=zname, p=p, q=int(q[i]), axis=axis, offset=d))
            print(f"trajectory {i} (class {c}) at {zname} m: p {p:.1f} GeV done", flush=True)
    np.savez(a.out + ".npz", arrivals=np.stack(rows), meta=json.dumps(meta), e=e)
    print(f"-> {a.out}.npz: {len(meta)} configurations x {a.m}")


if __name__ == "__main__":
    main()
