#!/usr/bin/env python3
"""Checks of transport seeds written with --storage shell (bin/generate_muon_study.py, lib/veto/shell.py):
  1. rows per class, keep-all ceilings
  2. no bias: per shell cell, the stored rows' weight crossing it against the cell's rate from ALL muons (shell_hist)
  3. coverage: stored rows and n_eff per cell
  4. stopped muons in the last metre of the shield
    python3 bin/shell_check.py <seed dir> [--json out.json]
Run where torch is available (the transport container); the crossings are recomputed with veto.shell on the CPU.
"""
import _bootstrap  # noqa: F401
import argparse
import glob
import json

import numpy as np
import torch

from veto import muon_types as mt
from veto import shell as sh


def cells_of(d):
    """(row, cell) of the stored rows' shell crossings, as the storage computed them"""
    r = np.nonzero((d["type"] <= 6) & (d["z_30"] > mt.REACH_30))[0]
    g = lambda tag: torch.from_numpy(np.stack([d[f"{c}_{tag}"][r] for c in "xyz"], 1).astype(np.float64))
    k = torch.from_numpy(np.stack([d["kx"][r], d["ky"][r], d["kz"][r]], 1).astype(np.float64))
    i, c = sh.crossings_torch(g("30"), k, g("82"))
    return r[i.numpy()], c.numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--json", default=None)
    ap.add_argument("--ht", action="store_true", help="plain Horvitz-Thompson weights (weight_raw / p_keep), without the per-type "
                    "rescaling to the seed's exact totals")
    a = ap.parse_args()
    files = sorted(glob.glob(f"{a.dir}/seed_*.npz"))
    names = {**{t: n for t, n in mt.TYPES.items()}}
    S = len(files)
    s1 = np.zeros(sh.N_CELLS); s2 = np.zeros(sh.N_CELLS); nrow = np.zeros(sh.N_CELLS)   # stored rows
    h1 = np.zeros(sh.N_CELLS); h2 = np.zeros(sh.N_CELLS)                                 # all muons (shell_hist)
    rows_by, last_rows, last_rate, capped = [], [], [], []
    print(f"{S} seeds in {a.dir}\n")
    for f in files:
        d = dict(np.load(f, allow_pickle=False))
        meta = json.loads(str(d["meta"]))
        info = meta["shell"]
        rows_by.append([int((d["type"] == t).sum()) for t in range(8)])
        last_rows.append(int(d["last_metre"].sum())); last_rate.append(float(d["weight"][d["last_metre"]].sum()))
        capped.append([k for k, v in info["capped"].items() if v])
        i, c = cells_of(d)
        wt = (d["weight_raw"].astype(np.float64) / d["p_keep"]) if a.ht else d["weight"].astype(np.float64)
        w = wt[i]
        s1 += np.bincount(c, w, sh.N_CELLS); s2 += np.bincount(c, w * w, sh.N_CELLS); nrow += np.bincount(c, minlength=sh.N_CELLS)
        h1 += d["shell_hist"][1]; h2 += d["shell_hist"][2]
    rows_by = np.array(rows_by)
    print("1. rows per seed (mean):")
    for t in range(8):
        print(f"   {names[t]:16s} {rows_by[:, t].mean():10.0f}")
    print(f"   {'last-metre stops':16s} {np.mean(last_rows):10.0f}   ({np.mean(last_rate):.4g} per spill)")
    print(f"   total {rows_by.sum(1).mean():.0f} rows per seed; ceilings hit: {sum(map(len, capped))} times {sorted(set(sum(capped, [])))}")
    # 2. bias: the stored estimate of each cell's rate against the truth from all muons (per seed averages)
    est, tru = s1 / S, h1 / S
    err = np.sqrt(s2) / S
    used = np.arange(sh.N_SIDE)[(np.arange(sh.N_SIDE) // sh.NA) * (sh.Z1 - sh.Z0) / sh.NZ + sh.Z0 < sh.ZB]
    cells = np.r_[used, np.arange(sh.OFF_UP, sh.N_CELLS)]
    ok = cells[(tru[cells] > 0) & (nrow[cells] >= 10)]
    pull = (est[ok] - tru[ok]) / err[ok]
    print(f"\n2. no bias, {len(ok)} cells with >= 10 stored crossings: total stored / true {est[ok].sum() / tru[ok].sum():.4f}; "
          f"pull mean {pull.mean():.2f}, rms {pull.std():.2f} (expected 0, 1); |pull| > 3: {(np.abs(pull) > 3).sum()}")
    # 3. coverage
    ne = np.where(s2 > 0, s1 ** 2 / np.where(s2 > 0, s2, 1), 0)[cells]
    ne_all = np.where(h2 > 0, h1 ** 2 / np.where(h2 > 0, h2, 1), 0)[cells]
    seen = tru[cells] > 0
    q = lambda x: np.percentile(x, [0, 10, 50]) if len(x) else [0, 0, 0]
    print(f"\n3. coverage of the {seen.sum()} cells the muons reach (of {len(cells)} up to z = {sh.ZB} m), over all {S} seeds:")
    print(f"   stored rows per cell: min {q(nrow[cells][seen])[0]:.0f}, 10 % {q(nrow[cells][seen])[1]:.0f}, median {q(nrow[cells][seen])[2]:.0f}")
    print(f"   n_eff per cell, stored rows: min {q(ne[seen])[0]:.1f}, 10 % {q(ne[seen])[1]:.1f}, median {q(ne[seen])[2]:.1f}; "
          f"all muons (no thinning): median {np.median(ne_all[seen]):.1f}")
    print(f"   cells with n_eff < 25: stored {(ne[seen] < 25).sum()}, all muons {(ne_all[seen] < 25).sum()}")
    print(f"\n4. last metre of the shield: {np.mean(last_rows):.0f} rows per seed, {np.mean(last_rate):.4g} per spill")
    if a.json:
        json.dump(dict(seeds=S, rows_per_seed=rows_by.mean(0).tolist(), last_metre_rows=float(np.mean(last_rows)),
                       last_metre_rate=float(np.mean(last_rate)), ceilings_hit=sum(map(len, capped)),
                       bias_ratio=float(est[ok].sum() / tru[ok].sum()), pull_mean=float(pull.mean()), pull_rms=float(pull.std()),
                       cell_names=[sh.cell_names()[c] for c in cells], cell_rate=tru[cells].tolist(), cell_rows=nrow[cells].tolist(),
                       cell_neff=ne.tolist(), cell_neff_all=ne_all.tolist()), open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
