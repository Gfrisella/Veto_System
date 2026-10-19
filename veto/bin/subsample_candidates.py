#!/usr/bin/env python3
"""Weighted subsample of the pair candidates of a reco-inputs file, for fast design evaluation.

The enriched dataset gives ~25 M candidates (its high-energy tail is many low-weight muons), 100x alpha's: too many
to evaluate a design in seconds. Poisson sampling proportional to weight: candidate i is kept with probability
p_i = min(1, c w_i), c set so the expected number kept is --n, and gets weight w_i / p_i. Unbiased for every
weighted sum (the background before and after the veto), heavy candidates are all kept, and the light ones, which
barely move any sum, are thinned. The single-muon sample and the rate maps are copied; tracks no longer used are dropped.

  bin/remote.sh run bin/subsample_candidates.py --inputs inputs_delta.npz --out inputs_delta_s.npz --n 400000
"""

import _bootstrap  # noqa: F401

import argparse

import numpy as np

from veto import paths


def poisson_scale(w, n):
    """c with sum(min(1, c w)) = n (bisection in log c)."""
    lo, hi = 1e-30, 1e30
    for _ in range(200):
        c = np.sqrt(lo * hi)
        lo, hi = (c, hi) if np.minimum(1.0, c * w).sum() < n else (lo, c)
    return np.sqrt(lo * hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", default="inputs_delta.npz")
    ap.add_argument("--out", default="inputs_delta_s.npz")
    ap.add_argument("--n", type=float, default=4e5, help="expected number of candidates kept")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    d = dict(np.load(paths.results_dir() / "reco" / args.inputs))
    w = d["cand_w"].astype(np.float64)
    p = np.minimum(1.0, poisson_scale(w, args.n) * w)
    keep = np.nonzero(np.random.default_rng(args.seed).random(len(w)) < p)[0]
    w_new = w[keep] / p[keep]
    rows = np.unique(np.concatenate([d["cand"][keep].ravel(), d["single"]]))
    remap = np.full(len(d["tr_type"]), -1, np.int64)
    remap[rows] = np.arange(len(rows))
    out = {k: v[rows] for k, v in d.items() if k.startswith("tr_")}
    out.update({k: v for k, v in d.items() if not k.startswith("tr_") and not k.startswith("cand") and k not in ("single",)})
    out.update(cand=remap[d["cand"][keep]], cand_vertex=d["cand_vertex"][keep], cand_w=w_new, single=remap[d["single"]],
               subsample=np.array([len(w), len(keep), args.seed]))
    np.savez(paths.results_dir() / "reco" / args.out, **out)
    neff = lambda x: x.sum() ** 2 / (x ** 2).sum()
    print(f"{len(w):,} candidates -> {len(keep):,} ({(p >= 1).sum():,} kept with certainty); weight {w.sum():.5g} -> "
          f"{w_new.sum():.5g} (sampling error {np.sqrt(((1 - p) / p * w ** 2).sum()) / w.sum():.2%}); "
          f"n_eff {neff(w):.4g} -> {neff(w_new):.4g}; {len(rows):,} tracks -> {args.out}")


if __name__ == "__main__":
    main()
