#!/usr/bin/env python3
"""A1 closure check: the factorised background estimate against the direct count of surviving pairs.

For each design and each candidate file (results/reco/inputs.npz: within-run pairs; inputs_xrun.npz: the independent
cross-run pairs of bin/make_crossrun_pairs.py), run the veto on the candidate pairs (as run_reco.evaluate) and compare

    observed   n_obs  = pairs with neither muon explained (the direct count, unweighted)
    predicted  n_pred = sum over pairs of eps_k1 eps_k2, eps_k = unexplained fraction of the pair muons of type k
                        (measured on these same pairs, as objective.confusion_matrix with eps_source "pairs")

and the weighted fractions FN_direct vs FN_factorised. Poisson p-values of n_obs given n_pred (two-sided); when
n_obs = 0 the direct estimate is a 90 % CL upper limit (2.3 / n_pairs, objective). The factorisation holds if the
two muons of a pair are vetoed independently: a large excess of n_obs means correlated escapes (e.g. both muons
bounced off the same wall region).

    python3 bin/closure_check.py D6_straw E3_robust [--inputs inputs.npz inputs_xrun.npz] [--out closure.json]
"""

import _bootstrap  # noqa: F401

import argparse
import json
import math
from pathlib import Path

import numpy as np

from run_reco import TYPE_NAMES, load_inputs, take
from veto import design as dsg
from veto import paths, reco
from veto import vertex_selection as vsel


def poisson_p_two_sided(n, mu):
    """Two-sided Poisson p-value of observing n given mean mu (sum of the probabilities not larger than P(n))."""
    if mu <= 0:
        return 1.0 if n == 0 else 0.0
    logp = lambda k: k * math.log(mu) - mu - math.lgamma(k + 1)
    p_n = logp(n)
    kmax = int(mu + 20 * math.sqrt(mu) + 50 + n)
    return float(min(1.0, sum(math.exp(logp(k)) for k in range(kmax) if logp(k) <= p_n + 1e-12)))


def check(design, tr, inp, cfg, rng):
    keep = np.arange(len(inp["cand"]))
    if cfg.ip_max is not None:
        c = inp["cand"]
        st = lambda i: (np.stack([tr["bx"][i], tr["by"][i], tr["bz"][i]], 1), np.stack([tr["bpx"][i], tr["bpy"][i], tr["bpz"][i]], 1))
        ip = vsel.measured_ip(*st(c[:, 0]), *st(c[:, 1]), cfg, np.random.default_rng(777))
        keep = np.nonzero(ip < cfg.ip_max)[0]
    cand, w = inp["cand"][keep], inp["cand_w"][keep].astype(np.float64)
    rows = cand.ravel()
    event = np.repeat(np.arange(len(cand)), 2)
    b = reco.run_events(take(tr, rows), event, np.repeat(inp["cand_vertex"][keep, 2], 2), design, inp, cfg, rng)
    une = ~b["explained"].reshape(-1, 2)                       # per pair muon: not explained
    ty = tr["type"][rows].astype(int).reshape(-1, 2)
    wt = np.repeat(w, 2).reshape(-1, 2)
    eps = {k: float((wt * une)[ty == k].sum() / wt[ty == k].sum()) for k in range(3) if (ty == k).any()}   # weighted
    eps_n = {k: float(une[ty == k].mean()) for k in range(3) if (ty == k).any()}                           # unweighted
    e = np.vectorize(lambda k: eps.get(int(k), 0.0))(ty)
    en = np.vectorize(lambda k: eps_n.get(int(k), 0.0))(ty)
    surv = ~b["vetoed"]
    n_obs, n_pred = int(surv.sum()), float((en[:, 0] * en[:, 1]).sum())
    fn_dir, fn_fact = float(w[surv].sum() / w.sum()), float((w * e[:, 0] * e[:, 1]).sum() / w.sum())
    return dict(n_pairs=int(len(cand)), n_obs=n_obs, n_pred=n_pred, p_value=poisson_p_two_sided(n_obs, n_pred),
                FN_direct=fn_dir, FN_factorised=fn_fact,
                FN_direct_ul90=(2.3 / len(cand)) if n_obs == 0 else None,
                eps={TYPE_NAMES[k]: v for k, v in eps.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("designs", nargs="+")
    ap.add_argument("--inputs", nargs="+", default=[paths.RECO_INPUTS, paths.RECO_INPUTS.replace(".npz", "_xrun.npz")])
    ap.add_argument("--out", default="closure.json")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    cfg = reco.RecoConfig(seed=a.seed, multiple_scattering=True)
    out = {}
    for f in a.inputs:
        if not (paths.results_dir() / "reco" / f).exists():
            print(f"{f}: missing (cross-run pairs: bin/make_crossrun_pairs.py), skipped")
            continue
        tr, inp = load_inputs(f)
        for name in a.designs:
            r = check(dsg.load(name), tr, inp, cfg, np.random.default_rng(a.seed))
            out[f"{name}|{f}"] = r
            ul = f", direct < {r['FN_direct_ul90']:.2g} (90 % CL)" if r["FN_direct_ul90"] else ""
            print(f"{name:<14}{f:<18} pairs {r['n_pairs']:>8,}  surviving: observed {r['n_obs']:>4}, factorised "
                  f"{r['n_pred']:>7.2f}  (p = {r['p_value']:.2f})  FN direct {r['FN_direct']:.2e} vs factorised "
                  f"{r['FN_factorised']:.2e}{ul}")
    path = paths.results_dir() / "reco" / a.out
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
