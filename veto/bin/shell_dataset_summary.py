#!/usr/bin/env python3
"""Numbers of a --storage shell transport (bin/generate_muon_study.py): per type rates per spill (mean over seeds, error from
the seed spread), n_eff per seed and over DISTINCT production muons (weights summed per pool_index over all seeds), train vs
hidden; shell cells from the full-muon rate maps (shell_hist): rate, n_eff, measured / negligible / holes, lifetime crossings.

    python3 bin/shell_dataset_summary.py <shell dir with train/ hidden/> [--spills 1.5e7]
"""
import _bootstrap  # noqa: F401
import argparse
import glob
import json

import numpy as np

from veto import muon_types as mt
from veto import shell as sh

KEEP = (0, 1, 2, 3)                       # kept completely in every seed


def scan(d):
    files = sorted(glob.glob(f"{d}/seed_*.npz"))
    S = len(files)
    rate = np.zeros((S, 8)); n_rows = np.zeros(8); neff_seed = np.zeros((S, 4))
    last = np.zeros(S); last_rows = 0
    h = np.zeros((3, sh.N_CELLS))
    keys, ws = {t: [] for t in KEEP}, {t: [] for t in KEEP}
    for i, f in enumerate(files):
        z = np.load(f)
        w, t = z["weight"].astype(np.float64), z["type"]
        rate[i] = z["w_by_type"]                                   # exact per-type totals of the seed (all muons)
        for k in KEEP:
            m = t == k
            wk = w[m]
            neff_seed[i, k] = wk.sum() ** 2 / (wk ** 2).sum() if m.any() else 0
            keys[k].append(z["evt_n"][m]); ws[k].append(wk)
        n_rows += np.bincount(t, minlength=8)
        lm = z["last_metre"]
        last[i] = w[lm].sum(); last_rows += int(lm.sum())
        h += z["shell_hist"]
    dist = {}
    for k in KEEP:
        ev, wk = np.concatenate(keys[k]), np.concatenate(ws[k]) / S
        u, inv = np.unique(ev, return_inverse=True)
        W = np.bincount(inv, wk)
        dist[k] = (len(u), W.sum() ** 2 / (W ** 2).sum(), W.max() / W.sum())
    return dict(S=S, rate=rate, n_rows=n_rows, neff_seed=neff_seed, dist=dist, last=last, last_rows=last_rows, h=h)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir"); ap.add_argument("--spills", type=float, default=1.5e7)
    a = ap.parse_args()
    R = {s: scan(f"{a.dir}/{s}") for s in ("train", "hidden")}
    out = {}
    print(f"seeds: train {R['train']['S']}, hidden {R['hidden']['S']} (each seed = one spill of 4e13 p.o.t.)\n")
    print(f"{'type':16s} {'rate / spill (train)':>24s} {'hidden':>24s} {'pull':>6s} {'n_eff/seed':>11s} {'distinct muons':>15s} "
          f"{'distinct n_eff':>15s} {'top muon':>9s}")
    for t in range(8):
        r = {s: (R[s]["rate"][:, t].mean(), R[s]["rate"][:, t].std(ddof=1) / np.sqrt(R[s]["S"])) for s in R}
        pull = (r["train"][0] - r["hidden"][0]) / np.hypot(r["train"][1], r["hidden"][1])
        extra = ""
        if t in KEEP:
            u, ne, top = R["train"]["dist"][t]
            extra = f" {R['train']['neff_seed'][:, t].mean():11.1f} {u:15,d} {ne:15,.0f} {100 * top:8.2f}%"
        print(f"{mt.TYPES[t]:16s} {r['train'][0]:12.4g} +- {r['train'][1]:8.2g} {r['hidden'][0]:12.4g} +- {r['hidden'][1]:8.2g} "
              f"{pull:6.2f}" + extra)
        out[mt.TYPES[t]] = dict(train=r["train"], hidden=r["hidden"], pull=pull)
    T = R["train"]
    lr = T["last"].mean(); le = T["last"].std(ddof=1) / np.sqrt(T["S"])
    print(f"\nstopped in the last metre of the shield: {lr:.4g} +- {le:.2g} per spill ({T['last_rows']:,} stored rows, "
          f"{lr * a.spills:.3g} over {a.spills:g} spills)")
    # shell cells: all muons of types 0-6, both sets together
    h = R["train"]["h"] + R["hidden"]["h"]
    S = R["train"]["S"] + R["hidden"]["S"]
    rate, n_all = h[1] / S, h[0]
    neff = np.where(h[2] > 0, h[1] ** 2 / np.where(h[2] > 0, h[2], 1), 0)
    zc = sh.Z0 + (np.arange(sh.N_SIDE) // sh.NA + 0.5) * (sh.Z1 - sh.Z0) / sh.NZ
    cells = np.r_[np.arange(sh.N_SIDE)[zc < sh.ZB], np.arange(sh.OFF_UP, sh.N_CELLS)]
    reached = cells[n_all[cells] > 0]
    w_hit = np.median(h[1][reached] / n_all[reached]) / S             # what one muon contributes per spill
    ul = np.where(n_all[cells] > 0, rate[cells] + 1.645 * np.sqrt(h[2][cells]) / S, 3.0 * w_hit)
    meas = neff[cells] >= 25
    print(f"\nshell (arch, margin {sh.MARGIN} m; {len(cells)} cells up to z = {sh.ZB} m incl. both caps), all muons of all "
          f"{S} seeds:")
    print(f"   cells reached {len(reached)}; measured (n_eff >= 25) {meas.sum()}; reached but n_eff < 25 "
          f"{((n_all[cells] > 0) & ~meas).sum()}; never reached {(n_all[cells] == 0).sum()}")
    q = np.percentile(neff[reached], [0, 10, 50])
    print(f"   n_eff per reached cell: min {q[0]:.1f}, 10 % {q[1]:.0f}, median {q[2]:.0f}")
    life = rate[reached] * a.spills
    print(f"   lifetime crossings per reached cell ({a.spills:g} spills): min {life.min():.3g}, 10 % {np.percentile(life, 10):.3g}, "
          f"median {np.median(life):.3g}, max {life.max():.3g}")
    print(f"   one simulated muon per cell = {w_hit:.3g} per spill; an empty cell's 95 % limit = {3 * w_hit * a.spills:.3g} "
          f"crossings over the lifetime")
    for name, idx in (("side 29.9-46 m", cells[(cells < sh.N_SIDE) & (zc[np.minimum(cells, sh.N_SIDE - 1)] < 46)]),
                      ("side 46-62 m", cells[(cells < sh.N_SIDE) & (zc[np.minimum(cells, sh.N_SIDE - 1)] >= 46) & (zc[np.minimum(cells, sh.N_SIDE - 1)] < 62)]),
                      ("side 62-82 m", cells[(cells < sh.N_SIDE) & (zc[np.minimum(cells, sh.N_SIDE - 1)] >= 62)]),
                      ("upstream cap 30.6 m", np.arange(sh.OFF_UP, sh.OFF_DOWN)), ("downstream cap 82 m", np.arange(sh.OFF_DOWN, sh.N_CELLS))):
        r_ = idx[n_all[idx] > 0]
        print(f"   {name:22s} cells {len(idx):4d}, reached {len(r_):4d}, measured {(neff[idx] >= 25).sum():4d}; "
              f"rate {rate[idx].sum():10.4g} / spill; median n_eff {np.median(neff[r_]) if len(r_) else 0:9.0f}")
    json.dump(dict(types=out, last_metre=[lr, le], cells=len(cells), reached=int(len(reached)), measured=int(meas.sum()),
                   w_hit=w_hit), open(f"{a.dir}/summary.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
