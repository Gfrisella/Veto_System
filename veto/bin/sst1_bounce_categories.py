#!/usr/bin/env python3
"""Old TRY6 run (data/store/sst1_TRY6, 20 001 files): SST1 muons split into straight / bounced off the cavern wall /
bounced off the floor, with the FairShip layout and the spectrometer field map, and the fraction of each with
>= 3 of 4 straw stations hit, then >= 3 stations and p > 1 GeV.

- muon_types_torch.classify (FairShip layout): SST1 acceptance at 82.08 m, bounce = > 0.5 m off the straight line
  from the 30.6 m state, bounce point K (closest approach of the two lines).
- bounce surface: nearest of the cavern wall (x = -4.56 m) and floor (y = -3.36 m) to K, within --surface-tol m;
  otherwise "bounced, other" (no credible K, or K far from both, e.g. +x side / structures).
- straws: 82 m state -> 32 straw layers by RK4 in MainSpectrometerField.h5 (straws.layer_states), yoke losses,
  digitize, stations with >= 1 hit (straws.hit_summary).
- weights: per spill of 4e13 p.o.t.; repeated evt_n handled as C5 (copies kept, w / k, total kept).

    python3 bin/sst1_bounce_categories.py [--max-rows N] [--threads 48] [--surface-tol 1.0] [--device cuda]
"""

import _bootstrap  # noqa: F401

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from veto import geometry as geo
from veto import io, weights
from veto import muon_types_torch as mtt

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data/store/sst1_TRY6"
RENAME = {f"{c}1": f"{c}_30" for c in ("x", "y", "z", "px", "py", "pz")}
RENAME.update({f"{c}2": f"{c}_82" for c in ("x", "y", "z", "px", "py", "pz")})


def straw_stations(args):
    pos, mom, q, seed, device = args
    from veto import straws
    states, _, lost = straws.layer_states(pos, mom, q, device=device)
    h = straws.digitize(states, np.random.default_rng(seed), lost)
    return straws.hit_summary(h, len(pos))[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--chunk", type=int, default=2_000_000)
    ap.add_argument("--surface-tol", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(ROOT / "results/sst1_bounce_categories"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    st, meta = io.open_store(STORE)
    n = int(meta["n"]) if not a.max_rows else min(a.max_rows, int(meta["n"]))
    n_files = int(meta["n_files"])
    w_raw = np.asarray(st["weight"][:n], np.float64)
    evt = np.asarray(st["evt_n"][:n])
    # C5: repetitions of an evt_n counted within its file (evt_n is per file)
    key = np.asarray(st["file_index"][:n]).astype(np.int64) * (1 << 32) + evt.astype(np.int64)
    k = weights.repetition_counts(key)
    w = w_raw / k
    w *= w_raw.sum() / w.sum()
    print(f"{n:,} rows, {n_files} files, total {w.sum() / n_files:.1f} muons / spill (4e13); "
          f"repeated rows {np.mean(k > 1):.3%}", flush=True)

    kind = np.empty(n, np.int8); kx = np.empty(n); ky = np.empty(n); pb = np.empty(n)
    for a0 in range(0, n, a.chunk):
        a1 = min(n, a0 + a.chunk)
        d = {new: np.asarray(st[old][a0:a1]) for old, new in RENAME.items()}
        d["pdg"] = np.asarray(st["pdg"][a0:a1])
        c = mtt.classify(d, device=a.device, chunk=a.chunk)
        kind[a0:a1], kx[a0:a1], ky[a0:a1] = c["type"], c["kx"], c["ky"]
        pb[a0:a1] = np.sqrt(d["px_82"].astype(float) ** 2 + d["py_82"] ** 2 + d["pz_82"] ** 2)
        print(f"  classified {a1:,} / {n:,}  {time.time() - t0:.0f} s", flush=True)

    sst1 = np.isin(kind, (0, 1, 2))
    bounced = kind == 0
    d_floor = np.abs(ky - geo.CAVERN_Y_FLOOR)
    d_wall = np.abs(kx - geo.CAVERN_X_WALL)
    credible = np.isfinite(kx)
    floor = bounced & credible & (d_floor <= d_wall) & (d_floor < a.surface_tol)
    wall = bounced & credible & (d_wall < d_floor) & (d_wall < a.surface_tol)
    other = bounced & ~floor & ~wall
    cats = {"straight": sst1 & ~bounced, "bounced_wall": wall, "bounced_floor": floor, "bounced_other": other}

    # straw stations for every SST1 muon going forward at 82 m (no momentum cut: the p > 1 GeV cut comes after)
    pz82 = np.asarray(st["pz2"][:n])
    cand = np.flatnonzero(sst1 & (pz82 > 0))
    n_st = np.zeros(n, np.int8)
    pos = np.stack([np.asarray(st[c][:n])[cand] for c in ("x2", "y2", "z2")], 1).astype(float)
    mom = np.stack([np.asarray(st[c][:n])[cand] for c in ("px2", "py2", "pz2")], 1).astype(float)
    q = -np.sign(np.asarray(st["pdg"][:n])[cand]).astype(float)
    step = 200_000 if a.device == "cpu" else 2_000_000
    jobs = [(pos[i:i + step], mom[i:i + step], q[i:i + step], i, a.device) for i in range(0, len(cand), step)]
    if a.device == "cpu":
        with ProcessPoolExecutor(a.threads) as ex:
            res = list(ex.map(straw_stations, jobs))
    else:
        res = []
        for j in jobs:
            res.append(straw_stations(j))
            print(f"  straws {j[3] + len(j[0]):,} / {len(cand):,}  {time.time() - t0:.0f} s", flush=True)
    n_st[cand] = np.concatenate(res) if res else []
    print(f"  straws for {len(cand):,} tracks  {time.time() - t0:.0f} s", flush=True)

    tot = w[sst1].sum()
    table = {}
    for name, m in cats.items():
        wm = w[m].sum()
        table[name] = {
            "rows": int(m.sum()),
            "rate_kHz": wm / n_files * 1e-3,
            "frac_of_sst1_pct": 100 * wm / tot,
            "st3_pct_of_cat": 100 * w[m & (n_st == 3)].sum() / wm if wm else float("nan"),
            "st4_pct_of_cat": 100 * w[m & (n_st == 4)].sum() / wm if wm else float("nan"),
            "st34_pct_of_cat": 100 * w[m & (n_st >= 3)].sum() / wm if wm else float("nan"),
            "st34_p_gt1_pct_of_cat": 100 * w[m & (n_st >= 3) & (pb > 1)].sum() / wm if wm else float("nan"),
            "st34_p_gt1_pct_of_sst1": 100 * w[m & (n_st >= 3) & (pb > 1)].sum() / tot,
            "p_gt1_pct_of_cat": 100 * w[m & (pb > 1)].sum() / wm if wm else float("nan"),
        }
    summary = {"rows_used": n, "files": n_files, "sst1_rate_kHz": tot / n_files * 1e-3,
               "surface_tol_m": a.surface_tol, "categories": table, "runtime_s": time.time() - t0}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    np.savez(out / "per_row.npz", kind=kind, kx=kx.astype(np.float32), ky=ky.astype(np.float32),
             p82=pb.astype(np.float32), n_stations=n_st, weight=w.astype(np.float32))
    print(f"\nSST1 rate {tot / n_files * 1e-3:.2f} kHz (4e13 p.o.t. spill)")
    print(f"{'category':15s} {'rate kHz':>9s} {'% of SST1':>10s} {'% 3st':>7s} {'% 4st':>7s} {'% >=3st':>8s} "
          f"{'% >=3st & p>1':>14s} {'(% of SST1)':>12s}")
    for name, r in table.items():
        print(f"{name:15s} {r['rate_kHz']:9.3f} {r['frac_of_sst1_pct']:10.2f} {r['st3_pct_of_cat']:7.2f} "
              f"{r['st4_pct_of_cat']:7.2f} {r['st34_pct_of_cat']:8.2f} {r['st34_p_gt1_pct_of_cat']:14.2f} "
              f"{r['st34_p_gt1_pct_of_sst1']:12.2f}")


if __name__ == "__main__":
    main()
