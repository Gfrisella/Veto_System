#!/usr/bin/env python3
"""Thought experiment: the fate of a few chosen muons, each transported many times with different seeds (the production's
transport: generate_muon_study.Simulation, MuonsAndMatter TRY6, field, cavern), as histograms only (no rows kept):
  type fractions (muon_types: 0-7), where they stop (z), x-y at z = 30.6 m and at z = 82 m, momentum at 82 m, shell cells.

    python3 bin/muon_fate.py --muons five_muons.npz --copies 20000000 --passes 5 --gpu 0 --out fate.npz
five_muons.npz: px_start py_start pz_start x_start y_start z_start pdg (+-13) names.  Each pass transports every muon
`copies` times (one GPU batch of n_muons x copies), with seed = pass index.
"""
import _bootstrap  # noqa: F401
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import generate_muon_study as G                                    # noqa: E402
from veto import muon_types as mt                                  # noqa: E402
from veto import shell as sh                                       # noqa: E402

XY30 = np.linspace(-6, 6, 241)            # m, 5 cm
XY82 = np.linspace(-12, 12, 241)          # m, 10 cm
ZSTOP = np.arange(0, 82.5, 0.5)           # m
PB = np.geomspace(0.1, 400, 81)           # GeV
KZ, KX = np.linspace(28, 84, 113), np.linspace(-8, 11, 77)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--muons", required=True)
    ap.add_argument("--copies", type=int, default=20_000_000)
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mm-dir", default=str(G.MM_DIR))
    a = ap.parse_args()
    import torch
    m = np.load(a.muons)
    names = [str(x) for x in m["names"]]
    k = len(names)
    tmp = Path(tempfile.mkdtemp()) / "muons.h5"
    with h5py.File(tmp, "w") as f:                                  # the Simulation loads a file: the k muons, weight 1
        for c, s in (("px", "px_start"), ("py", "py_start"), ("pz", "pz_start"), ("x", "x_start"), ("y", "y_start"), ("z", "z_start")):
            f.create_dataset(c, data=m[s].astype(np.float32))
        f.create_dataset("pdg", data=m["pdg"].astype(np.float32)); f.create_dataset("weight", data=np.ones(k))
    sim = G.Simulation(a.mm_dir, "TRY6", a.gpu, 0, tmp)
    base, charge = sim.muons.copy(), sim.charge.clone()
    sim.muons = np.repeat(base, a.copies, axis=0)                  # muon i occupies rows [i * copies, (i + 1) * copies)
    sim.charge = charge.repeat_interleave(a.copies)
    who = np.repeat(np.arange(k), a.copies)
    H = dict(types=np.zeros((k, 8)), stop_z=np.zeros((k, len(ZSTOP) - 1)), xy30=np.zeros((k, len(XY30) - 1, len(XY30) - 1)),
             xy82=np.zeros((k, len(XY82) - 1, len(XY82) - 1)), p82=np.zeros((k, len(PB) - 1)), shell=np.zeros((k, sh.N_CELLS)),
             bounce=np.zeros((k, 4)),          # bounce point: none, wall (x < -4.3 m), floor (y < -3.1 m), elsewhere
             sst_side=np.zeros((k, 3, 2)),     # SST1 types 0-2: arrivals at x < 0, x > 0
             kzx=np.zeros((k, len(KZ) - 1, len(KX) - 1)))
    dev = torch.device(f"cuda:{a.gpu}")
    t0 = time.time()
    for s in range(a.passes):
        cols = sim.run(1000 + s)
        typ, rep = G.classify(cols, "gpu", a.gpu)
        out = rep["out"]
        for i in range(k):
            sl = slice(i * a.copies, (i + 1) * a.copies)
            t = typ[sl]
            H["types"][i] += np.bincount(t, minlength=8)
            st = t == 7
            H["stop_z"][i] += np.histogram(cols["z_30"][sl][st], ZSTOP)[0]
            s6 = t == 6
            H["stop_z"][i] += np.histogram(cols["z_82"][sl][s6], ZSTOP)[0]
            r30 = cols["z_30"][sl] > mt.REACH_30
            H["xy30"][i] += np.histogram2d(cols["x_30"][sl][r30], cols["y_30"][sl][r30], [XY30, XY30])[0]
            r82 = t <= 5
            H["xy82"][i] += np.histogram2d(cols["x_82"][sl][r82], cols["y_82"][sl][r82], [XY82, XY82])[0]
            p82 = np.sqrt(cols["px_82"][sl][r82].astype(np.float64) ** 2 + cols["py_82"][sl][r82] ** 2 + cols["pz_82"][sl][r82] ** 2)
            H["p82"][i] += np.histogram(p82, PB)[0]
        # bounce points (out[...] covers the rows that reach z = 30.6, in order)
        reach = np.nonzero(cols["z_30"] > mt.REACH_30)[0]
        kxa, kya, kza = np.full(len(who), np.nan), np.full(len(who), np.nan), np.full(len(who), np.nan)
        kxa[reach], kya[reach], kza[reach] = out["kx"], out["ky"], out["kz"]
        for i in range(k):
            sl = slice(i * a.copies, (i + 1) * a.copies)
            kx_, ky_, kz_ = kxa[sl], kya[sl], kza[sl]
            fin = np.isfinite(kz_)
            wall = fin & (kx_ < -4.3); floor = fin & ~wall & (ky_ < -3.1)
            H["bounce"][i] += [(~fin).sum(), wall.sum(), floor.sum(), (fin & ~wall & ~floor).sum()]
            H["kzx"][i] += np.histogram2d(kz_[fin], kx_[fin], [KZ, KX])[0]
            t = typ[sl]; x82 = cols["x_82"][sl]
            for tt in (0, 1, 2):
                H["sst_side"][i, tt] += [((t == tt) & (x82 < 0)).sum(), ((t == tt) & (x82 > 0)).sum()]
        kk = np.stack([out["kx"], out["ky"], out["kz"]], 1)
        for c0 in range(0, len(reach), 4_000_000):
            r = reach[c0:c0 + 4_000_000]
            g = lambda tag: torch.from_numpy(np.stack([cols[f"{c}_{tag}"][r] for c in "xyz"], 1).astype(np.float64)).to(dev)
            ii, cc = sh.crossings_torch(g("30"), torch.from_numpy(kk[c0:c0 + 4_000_000].astype(np.float64)).to(dev), g("82"))
            ii, cc = ii.cpu().numpy(), cc.cpu().numpy()
            np.add.at(H["shell"], (who[r[ii]], cc), 1)
        print(f"pass {s + 1}/{a.passes}: {len(who):,} transports, {time.time() - t0:.0f} s", flush=True)
    np.savez(a.out, names=np.array(names), copies=a.copies, passes=a.passes, XY30=XY30, XY82=XY82, ZSTOP=ZSTOP, PB=PB, KZ=KZ, KX=KX,
             shell_cells=np.array(sh.cell_names()), type_names=np.array([mt.TYPES[t] for t in range(8)]),
             **{c: m[c] for c in m.files if c != "names"}, **H, meta=json.dumps(dict(mm_dir=a.mm_dir, seeds=[1000 + s for s in range(a.passes)])))
    print(f"-> {a.out}")
    for i, n in enumerate(names):
        bb = H["bounce"][i] / H["bounce"][i].sum()
        print(f"{n:30s} bounce: wall {100 * bb[1]:.3g}%  floor {100 * bb[2]:.3g}%  elsewhere {100 * bb[3]:.3g}%  none {100 * bb[0]:.3g}%;"
              f"  SST1 bounced arrivals x<0 / x>0: {H['sst_side'][i, 0, 0]:.0f} / {H['sst_side'][i, 0, 1]:.0f}")
    for i, n in enumerate(names):
        f = H["types"][i] / H["types"][i].sum()
        print(f"{n:22s} " + "  ".join(f"{mt.TYPES[t]} {100 * f[t]:.3g}%" for t in range(8) if f[t] > 0))


if __name__ == "__main__":
    main()
