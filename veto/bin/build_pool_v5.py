#!/usr/bin/env python3
"""Input pool v5: built only from the complete target production (tailtransport complete mode, every muon source at its
physical rate; no rows of the 601 M h5, no reweighting to it). Muons where they leave the target (z = -0.0045 m).

  component 0  min. bias + dimuon / photon (tail_produce.py --mode complete): every source except muons with an open
               charm / beauty ancestor (from components 2, 3) and decays of the Pythia event's quarkonia (from component 1)
  component 1  quarkonia (tail_produce.py --mode complete --onia, tag v5_onia)
  component 2  charm (charm_cascade.py --hadrons + hf_decay.py, tag charmc_mu)
  component 3  beauty (tag beautyc_mu)
Weights: muons per spill (--pot, default 4e13 p.o.t.: FairShip run_fixedTarget.py) per component from its own production
(w_decay / rep x pot / protons).
Hidden set: HIDDEN of the source events (hash of component, slice, event), never used for training. The rest goes in
random chunks of ~CHUNK rows; each chunk's weights are then scaled per component to the whole pool's flux, so every
chunk is a full spill on its own (as pool v4).

    python3 bin/build_pool_v5.py --src <production> --out <pool> [--tags minbias,onia,charmc_mu,beautyc_mu]
Writes <out>/train_NNN.h5, <out>/hidden_NNN.h5 (px py pz x y z pdg weight pool_index component gen parent p_prod ch)
and <out>/pool.json.
"""

import argparse
import glob
import json
import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

QQ = [443, 100443, 553, 100553, 200553]
FLUSH = 4_000_000                                     # rows buffered per chunk before a write
P_MAX = 400.0                                         # GeV: the beam energy
FLOATS = ("px", "py", "pz", "x", "y", "z", "pdg", "weight", "p_prod")
INTS = {"pool_index": np.int64, "component": np.int8, "gen": np.int8, "parent": np.int32, "ch": np.int8}


def rows_of(args):
    """Selected rows of one production file, per-proton weights, and the hidden / train split."""
    fn, comp, hidden = args
    d = np.load(fn)
    sl = int(Path(fn).stem.rsplit("_", 1)[1])
    gen, par, hf = d["gen"], np.abs(d["parent"]), d["hf"]
    keep = np.ones(len(gen), bool)
    if comp == 0:
        keep = (hf == 0) & ~((gen == 2) & np.isin(par, QQ))   # quarkonia of the Pythia event: from the onia run (those
                                                              # made in the cascade, gen 7, are not in it: kept here)
    w = d["w_decay"].astype(np.float64) / float(d["rep"])
    # muons above the beam energy are unphysical (seen: 2 beauty events with pions of 500-870 GeV, a rare Geant4 glitch);
    # the transport's tables also end at 400 GeV: dropped and counted
    p = np.sqrt(d["px"].astype(np.float64) ** 2 + d["py"] ** 2 + d["pz"] ** 2)
    unphys = keep & (p >= P_MAX)
    dropped = (int(unphys.sum()), float(w[unphys].sum()))
    keep = keep & ~unphys
    ev = d["event"].astype(np.int64)
    r = {"px": d["px"][keep], "py": d["py"][keep], "pz": d["pz"][keep], "x": d["x"][keep], "y": d["y"][keep],
         "z": d["z"][keep], "pdg": d["pdg"][keep].astype(np.float32), "weight": w[keep], "p_prod": d["p_prod"][keep],
         "component": np.full(keep.sum(), comp, np.int8), "gen": gen[keep].astype(np.int8),
         "parent": d["parent"][keep].astype(np.int32), "ch": d["ch"][keep].astype(np.int8) if "ch" in d.files else np.full(keep.sum(), -1, np.int8)}
    key = (np.uint64(comp + 1) << np.uint64(56)) + (np.uint64(sl) << np.uint64(32)) + ev[keep].astype(np.uint64)
    u = ((key * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(40)).astype(np.float64) / float(1 << 24)
    r["hidden"] = u < hidden
    return comp, float(d["events"]), r, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tags", default="minbias,onia,charmc_mu,beautyc_mu", help="file tags of the 4 components (min. bias, "
                    "onia, charm, beauty): <tag>_<slice>.npz")
    ap.add_argument("--chunk", type=float, default=1e8)
    ap.add_argument("--hidden", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--pot", type=float, default=4e13, help="protons on target per spill (weights = muons per spill)")
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(20261102)
    tags = a.tags.split(",")
    comps = {0: f"{tags[0]}_[0-9]*.npz", 1: f"{tags[1]}_[0-9]*.npz", 2: f"{tags[2]}_[0-9]*.npz", 3: f"{tags[3]}_[0-9]*.npz"}
    files = {c: sorted(glob.glob(f"{a.src}/{p}")) for c, p in comps.items()}
    empty = [tags[c] for c in files if not files[c]]
    if empty:                                         # a misspelt tag would silently drop a whole component
        raise SystemExit(f"no files for component(s) {empty} in {a.src}")
    n_rows = sum(len(np.load(f)["w_decay"]) for c in files for f in files[c])
    n_train = max(1, int(round(n_rows * (1 - a.hidden) / a.chunk)))
    n_hidden = max(1, int(round(n_rows * a.hidden / a.chunk)))
    print(f"files {[len(v) for v in files.values()]}, ~{n_rows:,} rows -> {n_train} train + {n_hidden} hidden chunks", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    h5 = {}
    for name, n in (("train", n_train), ("hidden", n_hidden)):
        for j in range(n):
            g = h5py.File(out / f"{name}_{j:03d}.h5.tmp", "w")
            for k in FLOATS:
                g.create_dataset(k, (0,), np.float32 if k != "weight" else np.float64, maxshape=(None,), chunks=(1 << 20,))
            for k, t in INTS.items():
                g.create_dataset(k, (0,), t, maxshape=(None,), chunks=(1 << 20,))
            h5[(name, j)] = g
    events = {c: 0.0 for c in comps}
    flux = np.zeros(4)                                        # per-proton weight sums (before / events)
    nxt = 0
    buf, nbuf = {}, {}
    dropped = {c: [0, 0.0] for c in comps}             # rows and per-proton weight above P_MAX

    def flush(name, j):
        parts = buf.pop((name, j), [])
        nbuf[(name, j)] = 0
        if not parts:
            return
        g = h5[(name, j)]
        n0 = g["px"].shape[0]
        add = sum(len(q["px"]) for q in parts)
        for k in list(FLOATS) + list(INTS):
            g[k].resize((n0 + add,))
            g[k][n0:] = np.concatenate([q[k] for q in parts])

    jobs = [(f, c, a.hidden) for c in files for f in files[c]]
    with Pool(a.workers) as pool:
        for i, (comp, ev, r, dr) in enumerate(pool.imap_unordered(rows_of, jobs, chunksize=1)):
            events[comp] += ev
            dropped[comp][0] += dr[0]; dropped[comp][1] += dr[1]
            hid = r.pop("hidden")
            n = len(hid)
            r["pool_index"] = np.arange(nxt, nxt + n, dtype=np.int64)
            nxt += n
            flux[comp] += r["weight"].sum()
            dest = np.where(hid, rng.integers(0, n_hidden, n), rng.integers(0, n_train, n))
            for name, nch, m0 in (("train", n_train, ~hid), ("hidden", n_hidden, hid)):
                for j in range(nch):
                    m = m0 & (dest == j)
                    if m.any():                       # buffered: written in blocks of >= FLUSH rows (many small files)
                        buf.setdefault((name, j), []).append({k: r[k][m] for k in list(FLOATS) + list(INTS)})
                        nbuf[(name, j)] = nbuf.get((name, j), 0) + int(m.sum())
                        if nbuf[(name, j)] >= FLUSH:
                            flush(name, j)
            if i % 50 == 0:
                print(f"  {i + 1}/{len(jobs)} files, {nxt:,} rows, {time.time() - t0:.0f} s", flush=True)
    for key in list(buf):
        flush(*key)
    # per-proton -> per spill, and every chunk scaled per component to the whole pool's flux
    total = np.array([flux[c] * a.pot / events[c] if events[c] else 0.0 for c in range(4)])
    info = dict(pot_per_spill=a.pot, dropped_above_beam_energy={c: dict(rows=dropped[c][0], flux_per_spill=dropped[c][1] * a.pot /
                                                                         events[c] if events[c] else 0.0) for c in comps},
                events=events, flux_per_spill=total.tolist(), rows=int(nxt), train_chunks=n_train, hidden_chunks=n_hidden,
                hidden_fraction=a.hidden, sources={c: len(files[c]) for c in files}, tags=a.tags, chunks={})
    for (name, j), g in h5.items():
        w = g["weight"][:]
        comp = g["component"][:]
        sw = np.bincount(comp, weights=w, minlength=4)
        scale = np.where(sw > 0, total / np.maximum(sw, 1e-300), 0.0)
        g["weight"][:] = w * scale[comp]
        info["chunks"][f"{name}_{j:03d}"] = dict(rows=int(len(w)), scale=scale.tolist())
        g.attrs["description"] = f"input pool v5 ({name} {j}): complete target production, bin/build_pool_v5.py"
        g.close()
        (out / f"{name}_{j:03d}.h5.tmp").rename(out / f"{name}_{j:03d}.h5")
    json.dump(info, open(out / "pool.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "chunks"}), f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
