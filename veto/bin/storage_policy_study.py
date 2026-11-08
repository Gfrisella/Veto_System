#!/usr/bin/env python3
"""Which storage policy of the muon production keeps good statistics everywhere a veto could be?

The production keeps, per fate type, a uniform random fraction of the muons of the first n_t seeds (15 M rows per
type, generate_muon_study.storage_plan). The rare types (SST1, veto only) are kept completely in every seed; the
common ones (rock bounced, straight past, stops) only by the first 20 - 27 seeds. The probe (probe_statistics.py)
found ~10 independent input muons per 25 cm cell in the first metre beside the vessel wall and on the floor.

Here the stored rows of a dataset stand for the whole population: row i has the input weight v_i (weight / scale,
muons per spill) and stands for scale_i muons per seed (it was kept with probability 1 / (scale_i n_seeds_t)).
So any policy "keep a muon of a seed with probability p_i" over S seeds can be emulated exactly in expectation:
  rows kept           S sum_i scale_i p_i
  estimate per cell   sum over kept rows of v_i / (S p_i)         (Horvitz-Thompson: unbiased for any p)
  n_eff per cell      S (sum_i scale_i v_i)^2 / sum_i scale_i v_i^2 / p_i
(the limit from finitely many input muons is checked separately: n_eff_in of the stored rows, by evt_n).
Policies for the common types 4 - 6 (the rare types 0 - 3 stay complete, type 7 crosses nothing here), all at the
same number of stored rows B and S seeds:
  uniform     p = const per type (the current rule, spread over S seeds instead of the first n_t)
  weight      p proportional to v (Poisson sampling by weight: equal final weights)
  cell        every probe cell a muon crosses asks for T rows: p_i = min(1, T max_c 1 / (S P_c)),
              P_c = muons per seed crossing cell c
  cell+weight as cell, proportional to v within a cell: p_i = min(1, T v_i max_c 1 / (S V_c)), V_c = rate of c
Muons that cross no probe cell get p = base (enough for their rates).

  GPU_HOST=farm-gpu4 bin/remote.sh run bin/storage_policy_study.py --dataset <name> --budget 6e7 --seeds 6000
"""

import _bootstrap  # noqa: F401

import argparse
import json
import time
from types import SimpleNamespace

import numpy as np

from probe_statistics import crossing, local, planes
from veto import io, paths, probe

CHUNK = 8_000_000
COMMON = (4, 5, 6)


bands = probe.band_labels


def cells_of_chunk(store, c0, c1, pl, cell):
    """Rows of the common types in [c0, c1): their index, v, scale, evt_n and per plane (row, cell) pairs."""
    typ = np.asarray(store["type"][c0:c1])
    keep = np.nonzero(np.isin(typ, COMMON))[0]
    get = lambda col: np.asarray(store[col][c0:c1])[keep].astype(np.float64)
    a = np.stack([get(f"{c}_30") for c in "xyz"], 1)
    b = np.stack([get(f"{c}_82") for c in "xyz"], 1)
    k = np.stack([get(f"k{c}") for c in "xyz"], 1)
    w, s = get("weight"), get("scale")
    v = np.where(s > 0, w / np.where(s > 0, s, 1), 0.0)              # input weight, muons per spill
    mult = s                                                         # muons per seed this row stands for
    ev = np.asarray(store["evt_n"][c0:c1])[keep]
    hits = []
    for p in pl:
        nu = int(np.ceil((p["u"][1] - p["u"][0]) / cell)); nv = int(np.ceil((p["y"][1] - p["y"][0]) / cell))
        x, y, z, ok = crossing(a, k, b, p)
        uu, vv = local(p, x, y, z)
        ok &= (uu >= p["u"][0]) & (uu < p["u"][1]) & (vv >= p["y"][0]) & (vv < p["y"][1])
        r = np.nonzero(ok)[0]
        iu = ((uu[r] - p["u"][0]) / cell).astype(np.int64).clip(0, nu - 1)
        iv = ((vv[r] - p["y"][0]) / cell).astype(np.int64).clip(0, nv - 1)
        hits.append((r, iu * nv + iv))
    return typ[keep], v, mult, w, ev, hits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="design-optimization-dataset-delta-enriched-2026-10-19")
    ap.add_argument("--budget", type=float, default=None, help="rows of the common types to store (default: as now)")
    ap.add_argument("--seeds", type=int, default=6000, help="seeds of the production the policy is spread over")
    ap.add_argument("--base", type=float, default=0.02, help="share of the budget for muons crossing no probe cell")
    ap.add_argument("--cell", type=float, default=0.25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t0 = time.time()
    store, meta = io.open_store(paths.store_dir(args.dataset))
    n = meta["n"]
    pargs = SimpleNamespace(dz=2.5, left_reach=5.0, floor_z=[32.0, 40.0, 50.0, 60.0, 70.0], floor_tilt=30.0,
                            cell=args.cell, ubt=True, only_ubt=False)
    pl = planes(pargs)
    ncell = [int(np.ceil((p["u"][1] - p["u"][0]) / args.cell)) * int(np.ceil((p["y"][1] - p["y"][0]) / args.cell)) for p in pl]
    off = np.concatenate([[0], np.cumsum(ncell)])
    NC = int(off[-1])
    label = np.concatenate(bands(pl, args.cell))
    S = args.seeds

    # pass 1: per cell, muons per seed P_c, rate V_c, current rows and n_eff of input muons; per row its data
    P, V, rows_now, w2_now = (np.zeros(NC) for _ in range(4))
    keys, kws = [], []
    T_ = {"v": [], "mult": [], "typ": [], "cross": []}
    for c0 in range(0, n, CHUNK):
        typ, v, mult, w, ev, hits = cells_of_chunk(store, c0, min(c0 + CHUNK, n), pl, args.cell)
        crossed = np.zeros(len(v), bool)
        for i, (r, c) in enumerate(hits):
            g = off[i] + c
            np.add.at(P, g, mult[r]); np.add.at(V, g, mult[r] * v[r])
            np.add.at(rows_now, g, 1); np.add.at(w2_now, g, w[r] ** 2)
            keys.append(g.astype(np.int64) * (1 << 31) + ev[r].astype(np.int64)); kws.append(w[r])
            crossed[r] = True
        for k_, x in (("v", v), ("mult", mult), ("typ", typ), ("cross", crossed)):
            T_[k_].append(x)
        print(f"pass 1: rows {min(c0 + CHUNK, n):,} / {n:,}, {time.time() - t0:.0f} s", flush=True)
    R = {k: np.concatenate(x) for k, x in T_.items()}
    uk, inv = np.unique(np.concatenate(keys), return_inverse=True)
    wg = np.bincount(inv, weights=np.concatenate(kws))
    co = (uk >> 31).astype(np.int64)
    s1, s2 = np.bincount(co, weights=wg, minlength=NC), np.bincount(co, weights=wg ** 2, minlength=NC)
    with np.errstate(invalid="ignore", divide="ignore"):
        neff_in_now = np.where(s2 > 0, s1 ** 2 / s2, 0)
    del keys, kws, uk, inv, wg, co
    busy = P > 0

    # pass 2: per row, the largest need over the cells it crosses (cell, cell+weight)
    q_cell, q_cw = np.zeros(len(R["v"])), np.zeros(len(R["v"]))
    j0 = 0
    for c0 in range(0, n, CHUNK):
        typ, v, mult, w, ev, hits = cells_of_chunk(store, c0, min(c0 + CHUNK, n), pl, args.cell)
        for i, (r, c) in enumerate(hits):
            g = off[i] + c
            np.maximum.at(q_cell, j0 + r, 1.0 / P[g]); np.maximum.at(q_cw, j0 + r, 1.0 / V[g])
        j0 += len(v)
        print(f"pass 2: rows {min(c0 + CHUNK, n):,} / {n:,}, {time.time() - t0:.0f} s", flush=True)

    v, mult, typ, cross = R["v"], R["mult"], R["typ"], R["cross"]
    B = args.budget or float(len(v))                     # default: as many rows of the common types as now
    B_base, B_main = args.base * B, (1 - args.base) * B

    def solve(f, budget):
        """lam with S sum mult min(1, lam f) = budget (bisection in log lam)."""
        lo, hi = 1e-30, 1e30
        for _ in range(200):
            lam = np.sqrt(lo * hi)
            lo, hi = (lam, hi) if S * np.sum(mult * np.minimum(1.0, lam * f)) < budget else (lo, lam)
        return np.sqrt(lo * hi)

    pol = {}
    # uniform: per type, rows in proportion to the type's current share
    p = np.zeros(len(v))
    for t in COMMON:
        m = typ == t
        if m.any():
            share = B * m.sum() / len(v)
            p[m] = min(1.0, share / (S * mult[m].sum()))
    pol["uniform"] = p
    pol["weight"] = np.minimum(1.0, solve(v, B) * v)
    base = np.minimum(1.0, B_base / (S * mult[~cross].sum())) if (~cross).any() else 0.0
    for name, q in (("cell", q_cell), ("cell+weight", v * q_cw)):
        pc = np.zeros(len(v))
        lam = solve(np.where(cross, q, 0.0), B_main - 0.0)
        pc[cross] = np.minimum(1.0, lam * q[cross])
        pc[~cross] = base
        pol[name] = pc

    # pass 3: per cell n_eff of every policy, S (sum mult v)^2 / sum mult v^2 / p
    num = V ** 2 * S
    den = {k: np.zeros(NC) for k in pol}
    j0 = 0
    for c0 in range(0, n, CHUNK):
        typ_, v_, mult_, w_, ev_, hits = cells_of_chunk(store, c0, min(c0 + CHUNK, n), pl, args.cell)
        for i, (r, c) in enumerate(hits):
            g = off[i] + c
            for k, pk in pol.items():
                pr = pk[j0 + r]
                np.add.at(den[k], g, np.where(pr > 0, mult_[r] * v_[r] ** 2 / np.where(pr > 0, pr, 1), np.inf))
        j0 += len(v_)
        print(f"pass 3: rows {min(c0 + CHUNK, n):,} / {n:,}, {time.time() - t0:.0f} s", flush=True)

    res = {"dataset": args.dataset, "seeds": S, "budget_rows": B, "cell_m": args.cell, "policies": {}}
    groups = ["fin 0-0.5 m", "fin 0.5-1 m", "fin 1-2 m", "fin > 2 m", "floor", "ubt"]
    print(f"\ncommon types (4-6), {len(v):,} stored rows now; policies at {B:.3g} rows over {S} seeds; "
          f"cells with muons: {busy.sum():,}\nmedian / 10th percentile of n_eff per {args.cell * 100:.0f} cm cell, "
          f"and the share of cells below 30")
    print(f"{'':<14}" + "".join(f"{g:>22}" for g in groups) + f"{'rows':>12}")

    def line(name, ne, rows):
        cells = {}
        txt = f"{name:<14}"
        for gname in groups:
            m = busy & (label == gname)
            x = ne[m]
            cells[gname] = dict(median=float(np.median(x)), p10=float(np.percentile(x, 10)), below30=float(np.mean(x < 30)))
            txt += f"{cells[gname]['median']:>9.3g} /{cells[gname]['p10']:>6.3g} {100 * cells[gname]['below30']:>4.0f}%"
        print(txt + f"{rows:>12.3g}")
        return dict(groups=cells, rows=rows)

    res["policies"]["now (measured, n_eff of input muons)"] = line("now", neff_in_now, float(len(v)))
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in pol:
            ne = np.where(den[k] > 0, num / den[k], 0)
            res["policies"][k] = line(k, ne, float(S * np.sum(mult * pol[k])))
    print("\nshare of the stored rows that cross at least one probe cell, now:", f"{cross.mean():.1%}")
    out = paths.results_dir() / "storage_policy" / (args.out or f"study_{args.dataset}_S{S}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(f"wrote {out}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
