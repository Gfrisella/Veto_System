#!/usr/bin/env python3
"""How many of the 601 M input muons do we need? Thinning the input where it matters less.

Every seed re-propagates the whole enriched input file. Keeping input muon i with probability q_i (weight 1 / q_i,
fixed once, so unbiased for every later seed) cuts the GPU time per seed to ~ sum q / N. The price: a quantity t
(a probe cell's rate, a type's rate) built from W_it (muon i's weight in t, summed over the seeds) has
    n_eff_t = (sum_i W_it)^2 / sum_i W_it^2          ->   (sum_i W_it)^2 / sum_i W_it^2 / q_i
(independent INPUT muons, as bin/probe_statistics.py). Two rules, with a floor q0 everywhere (statistics everywhere):
  muon  q_i = clip(lam max_t W_it / sqrt(S2_t), q0, 1)                       (Neyman per muon, worst target)
  bin   q_b = clip(lam max_t sqrt(A_bt / (n_b S2_t)), q0, 1), A_bt = sum_{i in b} W_it^2
        bins of the source: component (tail, min. bias, dimuon, charm, other) x log p x theta x charge,
        so the bin rule also works for a fresh source production (not tied to these input muons)
Targets: every cell of the probe (veto.probe: fins, floor, UBT) and the rate of each type (0 - 7). q is set on the
even seeds and evaluated on the odd ones (muons never seen in the even seeds get q0: the honest penalty).

    bin/remote.sh run bin/input_thinning_study.py --dataset <name>          # CPU, ~30 min, ~100 GB RAM
Writes results/thinning/thinning_<dataset>.json and plots.
"""

import _bootstrap  # noqa: F401

import argparse
import json
import time

import numpy as np

from veto import io, paths, probe

CHUNK = 4_000_000
COMPONENTS = ("tail", "min. bias", "dimuon", "charm", "other")
LOGP = np.arange(0.0, 2.81, 0.1)                                          # log10 p [GeV] at the source
THETA = np.array([0, 2, 4, 7, 10, 15, 20, 30, 50, 100, 1e9]) * 1e-3      # rad
FRACTIONS = (0.05, 0.1, 0.2, 0.3, 0.5)


def source_bins(h5):
    """Per input muon: its source bin, its component, n per bin."""
    import h5py
    with h5py.File(h5, "r") as f:
        w = f["weight"][:]
        src = f["source"][:]
        comp = np.full(len(w), 4, np.int8)
        comp[np.abs(w - 768.75) < 0.5] = 1
        comp[np.abs(w - 7.6875) < 0.01] = 2
        comp[np.abs(w - 326.16) < 0.5] = 3
        comp[src == 1] = 0
        del w, src
        px, py, pz = f["px"][:], f["py"][:], f["pz"][:]
        q = (f["pdg"][:] > 0).astype(np.int8)
    pt = np.hypot(px, py)
    p = np.sqrt(pt.astype(np.float64) ** 2 + pz.astype(np.float64) ** 2)
    ip = np.clip(np.digitize(np.log10(p), LOGP) - 1, 0, len(LOGP) - 1)
    it = np.clip(np.digitize(np.arctan2(pt, pz), THETA) - 1, 0, len(THETA) - 2)
    b = ((comp.astype(np.int32) * len(LOGP) + ip) * (len(THETA) - 1) + it) * 2 + q
    nb = (len(COMPONENTS) * len(LOGP) * (len(THETA) - 1) * 2)
    return b.astype(np.int32), comp, np.bincount(b, minlength=nb), nb


def contributions(store, n, pl, ncell):
    """Unique (target, half, input muon) and the summed weight W. Targets: cells, then ncell + type."""
    import torch
    keys, ws = [], []
    t0 = time.time()
    for c0 in range(0, n, CHUNK):
        c1 = min(c0 + CHUNK, n)
        typ = np.asarray(store["type"][c0:c1]).astype(np.int64)
        ev = np.asarray(store["evt_n"][c0:c1]).astype(np.int64)
        half = np.asarray(store["run"][c0:c1]).astype(np.int64) & 1
        w = np.asarray(store["weight"][c0:c1]).astype(np.float64)
        tk = [ncell + typ]
        tw = [w]
        tr = [np.arange(len(typ))]
        m = np.nonzero(typ <= 6)[0]
        if len(m):
            get = lambda cols: torch.from_numpy(np.stack([np.asarray(store[c][c0:c1])[m] for c in cols], 1).astype(np.float64))
            idx, cel = probe.cells_torch(get(("x_30", "y_30", "z_30")), get(("kx", "ky", "kz")), get(("x_82", "y_82", "z_82")), pl)
            r = m[idx.numpy()]
            tk.append(cel.numpy())
            tw.append(w[r])
            tr.append(r)
        r = np.concatenate(tr)
        k = ((np.concatenate(tk) * 2 + half[r]) << 30) | ev[r]
        uk, inv = np.unique(k, return_inverse=True)
        keys.append(uk)
        ws.append(np.bincount(inv, weights=np.concatenate(tw)))
        print(f"rows {c1:,} / {n:,}, {time.time() - t0:.0f} s", flush=True)
    uk, inv = np.unique(np.concatenate(keys), return_inverse=True)
    W = np.bincount(inv, weights=np.concatenate(ws))
    th = uk >> 30
    return (th >> 1).astype(np.int32), (th & 1).astype(np.int8), (uk & ((1 << 30) - 1)).astype(np.int64), W


def max_by(index, values, size):
    """Per index, the largest value (0 where none); sort based (np.maximum.at is slow for 1e8 entries)."""
    out = np.zeros(size, values.dtype)
    o = np.argsort(index, kind="stable")
    idx, val = index[o], values[o]
    start = np.concatenate([[0], np.nonzero(np.diff(idx))[0] + 1])
    out[idx[start]] = np.maximum.reduceat(val, start)
    return out


def fit_lambda(score, count, q0, frac, total):
    """lam such that sum count clip(lam score, q0, 1) = frac total (bisection in log lam)."""
    lo, hi = -40.0, 40.0
    for _ in range(100):
        mid = (lo + hi) / 2
        kept = np.sum(count * np.clip(np.exp(mid) * score, q0, 1.0))
        lo, hi = (mid, hi) if kept < frac * total else (lo, mid)
    return np.exp((lo + hi) / 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="design-optimization-dataset-alpha-storage-v0-800seeds-2026-10-22")
    ap.add_argument("--q0", type=float, nargs="*", default=[0.01, 0.03])
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    t0 = time.time()
    store, meta = io.open_store(paths.store_dir(args.dataset))
    pl = probe.planes()
    shape, off = probe.grid(pl)
    ncell = int(off[-1])
    ntgt = ncell + 8
    bands = np.concatenate(probe.band_labels(pl))

    binof, comp, nb, nbins = source_bins(paths.enriched_input_h5())
    N = len(binof)
    print(f"{N:,} input muons, {int((nb > 0).sum())} source bins, {time.time() - t0:.0f} s", flush=True)
    cache = paths.results_dir() / "thinning" / f"pairs_{args.dataset}.npz"
    if cache.exists():
        z = np.load(cache)
        tgt, half, ev, W = z["tgt"], z["half"], z["ev"], z["W"]
    else:
        tgt, half, ev, W = contributions(store, meta["n"], pl, ncell)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, tgt=tgt, half=half, ev=ev, W=W)
    bad = ~np.isfinite(W) | (W <= 0)
    print(f"{int(bad.sum())} pairs with a non-finite or non-positive weight dropped", flush=True)
    tgt, half, ev, W = tgt[~bad], half[~bad], ev[~bad], W[~bad]
    print(f"{len(W):,} (target, half, input muon) pairs, {time.time() - t0:.0f} s", flush=True)

    A, B = half == 0, half == 1
    S1 = {h: np.bincount(tgt[m], weights=W[m], minlength=ntgt) for h, m in ((0, A), (1, B))}
    S2 = {h: np.bincount(tgt[m], weights=W[m] ** 2, minlength=ntgt) for h, m in ((0, A), (1, B))}
    with np.errstate(divide="ignore", invalid="ignore"):
        neff_before = np.where(S2[1] > 0, S1[1] ** 2 / S2[1], 0.0)
    # per-muon score (even seeds)
    sa = max_by(ev[A], (W[A] / np.sqrt(S2[0][tgt[A]])).astype(np.float32), N)
    print(f"muon scores: {int((~np.isfinite(sa)).sum())} non-finite", flush=True)
    sa = np.nan_to_num(sa, nan=0.0, posinf=1.0)
    # per-bin score (even seeds)
    kb = binof[ev[A]].astype(np.int64) * ntgt + tgt[A]
    uk, inv = np.unique(kb, return_inverse=True)
    Abt = np.bincount(inv, weights=W[A] ** 2)
    bb, tt = uk // ntgt, uk % ntgt
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.sqrt(Abt / (nb[bb] * S2[0][tt]))
    sb = max_by(bb, np.nan_to_num(s, nan=0.0, posinf=1.0), nbins)
    print(f"scores ready, {time.time() - t0:.0f} s", flush=True)

    out_dir = paths.results_dir() / "thinning"
    out_dir.mkdir(parents=True, exist_ok=True)
    busy = neff_before[:ncell] >= 1
    band_names = list(probe.BANDS)
    out = dict(dataset=args.dataset, n_input=N, pairs=int(len(W)), components=COMPONENTS,
               input_by_component=np.bincount(comp, minlength=5).tolist(), runs=[])
    evb, tb, Wb = ev[B], tgt[B], W[B]
    for rule in ("muon", "bin"):
        for q0 in args.q0:
            for frac in FRACTIONS:
                if rule == "muon":
                    lam = fit_lambda(sa, 1.0, q0, frac, N)
                    q = np.clip(lam * sa, q0, 1.0)
                else:
                    lam = fit_lambda(sb, nb, q0, frac, N)
                    q = np.clip(lam * sb, q0, 1.0)[binof]
                after_s2 = np.bincount(tb, weights=Wb ** 2 / q[evb], minlength=ntgt)
                with np.errstate(divide="ignore", invalid="ignore"):
                    neff_after = np.where(after_s2 > 0, S1[1] ** 2 / after_s2, 0.0)
                    ratio = np.where(neff_before > 0, neff_after / neff_before, np.nan)
                run = dict(rule=rule, q0=q0, frac=frac, kept=float(q.sum() / N),
                           kept_by_component=np.bincount(comp, weights=q, minlength=5).tolist(), bands={}, types={})
                for bname in band_names:
                    m = busy & (bands == bname)
                    run["bands"][bname] = dict(cells=int(m.sum()),
                                               ratio_median=float(np.nanmedian(ratio[:ncell][m])),
                                               ratio_p10=float(np.nanpercentile(ratio[:ncell][m], 10)),
                                               neff_p10_before=float(np.percentile(neff_before[:ncell][m], 10)),
                                               neff_p10_after=float(np.percentile(neff_after[:ncell][m], 10)),
                                               neff_median_before=float(np.median(neff_before[:ncell][m])),
                                               neff_median_after=float(np.median(neff_after[:ncell][m])))
                for t in range(8):
                    run["types"][t] = dict(neff_before=float(neff_before[ncell + t]), neff_after=float(neff_after[ncell + t]))
                out["runs"].append(run)
                worst = min(v["ratio_p10"] for v in run["bands"].values())
                print(f"{rule:5s} q0 {q0:.3f} target {frac:.2f} kept {run['kept']:.3f}: worst band p10 ratio {worst:.2f}, "
                      f"types " + " ".join(f"{run['types'][t]['neff_after'] / max(run['types'][t]['neff_before'], 1e-9):.2f}" for t in range(8)), flush=True)
                if rule == "bin" and q0 == args.q0[0] and frac == 0.2:
                    np.save(out_dir / "q_bin.npy", np.clip(lam * sb, q0, 1.0))
    out["bins"] = dict(logp=LOGP.tolist(), theta=THETA.tolist(), n=nb.tolist())
    json.dump(out, open(out_dir / f"thinning_{args.dataset}.json", "w"), indent=1)
    print(f"wrote {out_dir}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
