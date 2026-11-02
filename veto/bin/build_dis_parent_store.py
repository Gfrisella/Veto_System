#!/usr/bin/env python3
"""Muon store for the vessel muon DIS (user, 2026-11-02): the SST1 types (0 sst1_bounced, 1 sst1_selected, 2 sst1_other)
from the 20 001-run sample (gamma store = alpha, weight = per spill / 20 001), and the veto-only muons (type 3: cross a
veto plane, miss SST1) from the 6 000-seed zeta production instead of gamma's 1 715 seeds. --zeta-seeds N keeps the
seeds (run) < N with weight x 6000 / N. Other types are not needed for the vessel DIS (they do not reach the vessel
region or stop before it) and are left out. evt_n of the zeta rows is offset by 1e12 (different input file:
full_sample_after_target_enriched_p6.h5) so the per-input-muon errors do not merge the two samples.

    python3 bin/build_dis_parent_store.py [--zeta-seeds 2000]
    -> data/store/dis-parents-sst1-20k-vetoonly-zeta<N>-2026-11-02 (+ symlinks to the shared stores in data/store)
"""
import argparse, json, os
from pathlib import Path
import numpy as np

SHARED = Path("/disk/users/sqasim/workspace/uzh/Veto_System/veto/data/store")
GAMMA, ZETA = "design-optimization-dataset-gamma-2026-10-17", "design-optimization-dataset-zeta-new-storage-6000seeds-2026-11-01"
OUT_ROOT = Path(__file__).resolve().parents[1] / "data" / "store"

ap = argparse.ArgumentParser(); ap.add_argument("--zeta-seeds", type=int, default=2000); a = ap.parse_args()
OUT_ROOT.mkdir(parents=True, exist_ok=True)
for d in SHARED.iterdir():                          # the shared stores stay usable with VETO_STORE=data/store
    if d.is_dir() and not (OUT_ROOT / d.name).exists():
        os.symlink(d, OUT_ROOT / d.name)
gm = json.load(open(SHARED / GAMMA / "meta.json")); cols = gm["columns"]
tg = np.load(SHARED / GAMMA / "type.npy", mmap_mode="r"); ig = np.nonzero(np.asarray(tg) <= 2)[0]
tz = np.load(SHARED / ZETA / "type.npy", mmap_mode="r"); rz = np.load(SHARED / ZETA / "run.npy", mmap_mode="r")
iz = np.nonzero((np.asarray(tz) == 3) & (np.asarray(rz) < a.zeta_seeds))[0]
scale = 6000 / a.zeta_seeds
name = f"dis-parents-sst1-20k-vetoonly-zeta{a.zeta_seeds}-2026-11-02"
out = OUT_ROOT / name; out.mkdir(exist_ok=True)
n = len(ig) + len(iz)
for c in cols:
    g = np.load(SHARED / GAMMA / f"{c}.npy", mmap_mode="r"); z = np.load(SHARED / ZETA / f"{c}.npy", mmap_mode="r")
    o = np.lib.format.open_memmap(out / f"{c}.npy", mode="w+", dtype=g.dtype, shape=(n,))
    o[:len(ig)] = g[ig]
    zz = np.asarray(z[iz])
    if c == "weight": zz = (zz.astype(np.float64) * scale).astype(g.dtype)
    if c == "evt_n": zz = zz + 10 ** 12
    o[len(ig):] = zz
    o.flush(); del o
    print(c, flush=True)
w = np.load(out / "weight.npy", mmap_mode="r")
meta = dict(n=n, name=name, created="2026-11-02", columns=cols,
            description="vessel muon DIS parents: SST1 types 0-2 from the 20 001-run sample (gamma/alpha), type 3 veto_only "
                        f"from zeta seeds < {a.zeta_seeds} (weight x {scale:g}, evt_n + 1e12)",
            rows={"sst1_20k": int(len(ig)), "veto_only_zeta": int(len(iz))},
            muons_per_spill={"sst1_20k": float(np.asarray(w[:len(ig)], np.float64).sum()), "veto_only_zeta": float(np.asarray(w[len(ig):], np.float64).sum())},
            sources={"sst1": GAMMA, "veto_only": ZETA}, types=gm["types"], wall_bits=gm["wall_bits"])
json.dump(meta, open(out / "meta.json", "w"), indent=1)
print(json.dumps({k: meta[k] for k in ("n", "rows", "muons_per_spill")}))
