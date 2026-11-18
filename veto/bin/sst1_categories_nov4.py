#!/usr/bin/env python3
"""SST1 muons of a shell-storage dataset (default: the Nov-4 production, 3 220 seeds) split into straight / bounced off
the cavern wall / floor / other, with the fraction that hits >= 3 of the 4 straw stations and then has p > 1 GeV
(report Section 2, tab:fw:bounce; the same as bin/sst1_bounce_categories.py, which reads the old 2018-source run).
Types and bounce points (kx, ky) come from the store; weights are already per spill of 4e13 p.o.t.

    python3 bin/sst1_categories_nov4.py [--dataset NAME] [--threads 48] [--out DIR]
"""
import _bootstrap  # noqa: F401
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from veto import geometry as geo
from veto import io, paths
from veto import muon_types as mt


def straw_stations(args):
    pos, mom, q, seed = args
    from veto import straws
    states, _, lost = straws.layer_states(pos, mom, q, device="cpu")
    h = straws.digitize(states, np.random.default_rng(seed), lost)
    return straws.hit_summary(h, len(pos))[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=paths.MUON_DATASET)
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--surface-tol", type=float, default=1.0)
    ap.add_argument("--out", default=str(paths.results_dir() / "nov4_flow"))
    ap.add_argument("--sample", type=int, default=0, help="random subsample of the SST1 rows (weights scaled up)")
    a = ap.parse_args()
    t0 = time.time()
    st, meta = io.open_store(paths.store_dir(a.dataset))
    typ = np.asarray(st["type"])
    rows = np.flatnonzero(np.isin(typ, (0, 1, 2)))
    scale = 1.0
    if a.sample and a.sample < len(rows):
        scale = len(rows) / a.sample
        rows = np.sort(np.random.default_rng(3).choice(rows, a.sample, replace=False))
    g = lambda c: np.asarray(st[c])[rows]
    w, kind, kx, ky = g("weight").astype(float) * scale, typ[rows], g("kx").astype(float), g("ky").astype(float)
    pos = np.stack([g(c) for c in ("x_82", "y_82", "z_82")], 1).astype(float)
    mom = np.stack([g(c) for c in ("px_82", "py_82", "pz_82")], 1).astype(float)
    q = mt.charge_from_pdg_column(g("pdg"))
    pb = np.linalg.norm(mom, axis=1)
    print(f"{len(rows):,} SST1 rows, {w.sum():.1f} muons / spill  {time.time() - t0:.0f} s", flush=True)
    bounced = kind == 0
    d_floor, d_wall = np.abs(ky - geo.CAVERN_Y_FLOOR), np.abs(kx - geo.CAVERN_X_WALL)
    cred = np.isfinite(kx) & np.isfinite(ky)
    floor = bounced & cred & (d_floor <= d_wall) & (d_floor < a.surface_tol)
    wall = bounced & cred & (d_wall < d_floor) & (d_wall < a.surface_tol)
    cats = {"straight": ~bounced, "bounced_wall": wall, "bounced_floor": floor, "bounced_other": bounced & ~floor & ~wall}
    fwd = np.flatnonzero(mom[:, 2] > 0)
    step = 200_000
    jobs = [(pos[fwd[i:i + step]], mom[fwd[i:i + step]], q[fwd[i:i + step]], i) for i in range(0, len(fwd), step)]
    n_st = np.zeros(len(rows), np.int8)
    with ProcessPoolExecutor(a.threads) as ex:
        n_st[fwd] = np.concatenate(list(ex.map(straw_stations, jobs)))
    print(f"straws done  {time.time() - t0:.0f} s", flush=True)
    tot = w.sum(); table = {}
    for name, m in cats.items():
        wm = w[m].sum()
        table[name] = dict(rows=int(m.sum()), rate_kHz=wm * 1e-3, st34_kHz=w[m & (n_st >= 3)].sum() * 1e-3,
                           st34_p1_kHz=w[m & (n_st >= 3) & (pb > 1)].sum() * 1e-3,
                           n_eff_st34_p1=float(w[m & (n_st >= 3) & (pb > 1)].sum() ** 2 / max((w[m & (n_st >= 3) & (pb > 1)] ** 2).sum(), 1e-300)))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    s = dict(dataset=a.dataset, sst1_kHz=tot * 1e-3, categories=table, by_type_kHz={int(t): float(w[kind == t].sum() * 1e-3) for t in (0, 1, 2)})
    (out / f"sst1_categories_nov4{'_s' + str(a.sample) if a.sample else ''}.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
