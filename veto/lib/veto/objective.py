"""Objective (reward) for the veto design optimisation, e.g. as the reward of an RL agent.

Input: the result dict of one design as produced by bin/run_reco.py `evaluate()`
(the entries of results/reco/results.json). Nothing is re-simulated here.

Priorities, in order (lexicographic, implemented as tiers with weights 1000 / 100 / 10 / 1):
  0. constraints      rate limits of every plane and the 8 MCHF budget: each violation costs
                      at least 1000, more than all the tiers below together (max 111)
  1. background       combinatorial pairs that survive the veto = false vertices accepted as
                      true (FN of the background confusion matrix), N events in 15 years
  2. signal           signal lost to accidental vetoes (FP of the signal confusion matrix)
  3. cost             kCHF

Every tier term is normalised to [0, 1] (the rate penalty is open-ended), so one tier only
competes with the next one when it changes by less than ~1/10 of its range.

Background term (default bkg_form = "exp"):  T_bkg = 1 - exp(-N / bkg_scale).
With bkg_scale = 1 event it is the Poisson probability of at least one false vertex in 15 years.
It saturates for N >> bkg_scale (no gradient there); bkg_form = "log" keeps one.

Background estimate (ISSUES A1): the direct count of surviving pairs is 0-3 out of 196 736, i.e.
Poisson noise. The default estimate factorises the single-muon inefficiencies per muon type,
    N = N_before * sum_kl f_kl eps_k eps_l ,
with f_kl the weighted pair composition of the candidates and eps_k = 1 - veto efficiency of the
muons of type k. By default (eps_source = "pairs") eps_k is measured on the muons of the pairs
themselves (run_reco "pair_muons"), else on the single-muon sample: the two agree for selected muons,
but bounced muons of pairs are about twice as often unvetoed (0.47 vs 0.25), and bounced x selected
pairs dominate N. On D6_tile / D6_straw the single-muon version predicts 1.4 / 1.2 surviving pairs vs 3 / 2 observed.

Errors (1 sigma, statistical only): eps_k from the weighted muon sample (Wilson interval with the
effective number of muons (sum w)^2 / sum w^2, a muon's weight summed over the pairs it is in, so
eps = 0 still gets an error), N_before
from the candidate weights, signal FP binomial. They are propagated linearly to N, to every term
and to the reward. Cost and rate limits have no statistical error (cost is an order-of-magnitude
model). Every design is run on the same events (fixed seed), so errors are largely correlated
between designs: differences between designs are better known than the absolute numbers.

    python -m veto.objective                      # score results/reco/results.json
"""

import json
import math
from dataclasses import asdict, dataclass

import numpy as np

from veto import paths
from veto.technology import TECHNOLOGIES

TYPES = ("SST1 bounced", "SST1 p>1 GeV & timing", "SST1 other")   # tr_type 0, 1, 2 (run_reco.TYPE_NAMES)


@dataclass
class ObjectiveConfig:
    bkg_form: str = "exp"             # "exp": 1 - exp(-N / bkg_scale) | "log": log10(1 + N / bkg_target) / ...
    bkg_scale: float = 1.0            # exp: false vertices in 15 years at which the term reaches 1 - 1/e
    bkg_target: float = 0.01          # log: events in 15 years regarded as background free
    bkg_estimate: str = "factorised"  # "factorised" | "direct" | "max" (conservative)
    eps_source: str = "pairs"         # eps_k from the muons of the pairs ("pairs") or the single-muon sample ("single")
    sig_loss_max: float = 0.10        # signal loss mapped to the top of the signal tier
    cost_min_kchf: float = 97.2       # straw UBT alone (D0)
    cost_max_kchf: float = 8000.0     # top of the cost tier = the budget
    cost_budget_kchf: float = 8000.0  # above it: penalty w_budget * (1 + (cost - budget) / budget)
    straw_channel_kchf: float = 97.2 / 3600   # converts rel_cost when the area cost is nan (pixels)
    rate_margin: float = 1.0          # penalise above rate_margin * limit (e.g. 0.5 for a factor 2 safety margin)
    w_rate: float = 1000.0            # per limit exceeded: w_rate * (1 + log10(peak / (margin * limit)))
    w_budget: float = 1000.0
    w_bkg: float = 100.0
    w_sig: float = 10.0
    w_cost: float = 1.0


def wilson_sigma(p, n):
    """1-sigma half-width of the Wilson score interval (finite for p = 0 or 1)."""
    return math.sqrt(p * (1 - p) / n + 1 / (4 * n * n)) / (1 + 1 / n)


def sample_stats(inp=None):
    """Design-independent statistics of the reco inputs: pair composition f_kl, effective sample sizes."""
    if inp is None:
        inp = np.load(paths.results_dir() / "reco" / "inputs.npz")
    t = inp["tr_type"][inp["cand"]].astype(int)
    w = inp["cand_w"]
    f = np.zeros((3, 3))
    np.add.at(f, (t[:, 0], t[:, 1]), w)
    f = (f + f.T) / 2
    ts, ws = inp["tr_type"][inp["single"]].astype(int), inp["single_w"]
    n_eff = [ws[ts == k].sum() ** 2 / (ws[ts == k] ** 2).sum() for k in range(3)]
    # muons of the pairs: a muon is in several pairs, so its weight is summed over them first
    rows, inv = np.unique(inp["cand"].ravel(), return_inverse=True)
    wm = np.bincount(inv.ravel(), weights=np.repeat(w, 2), minlength=len(rows))
    tm = inp["tr_type"][rows].astype(int)
    n_eff_pair = [wm[tm == k].sum() ** 2 / (wm[tm == k] ** 2).sum() if (tm == k).any() else 0.0 for k in range(3)]
    return {"f_pair": f / f.sum(), "n_eff_single": np.array(n_eff), "n_eff_pair": np.array(n_eff_pair),
            "n_before_rel_err": float(np.sqrt((w ** 2).sum()) / w.sum())}


def confusion_matrix(res, stats, cfg=ObjectiveConfig()):
    """Background (pairs) and signal confusion matrices with 1-sigma errors; background in events / 15 y."""
    b, s = res["background"], res["signal"]
    f = stats["f_pair"]
    # eps_k from the muons of the pairs when the result has them (run_reco "pair_muons"): the single-muon
    # sample under-represents how badly bounced muons of pairs are vetoed (~0.47 vs ~0.25 unvetoed)
    pairs = res.get("pair_muons") if cfg.eps_source == "pairs" else None
    use_pair = [bool(pairs) and k in pairs and stats["n_eff_pair"][i] > 0 for i, k in enumerate(TYPES)]
    eps = np.array([1 - (pairs[k] if up else res["single"][k]) for k, up in zip(TYPES, use_pair)])
    n_eff = [stats["n_eff_pair"][i] if up else stats["n_eff_single"][i] for i, up in enumerate(use_pair)]
    eps_err = np.array([wilson_sigma(e, n) for e, n in zip(eps, n_eff)])

    fn_fact = float(eps @ f @ eps)
    fn_fact_err = float(np.sqrt(((2 * f @ eps) * eps_err) @ ((2 * f @ eps) * eps_err)))
    n_surv = b["n_survive"]
    fn_dir = b["FN_frac"]
    fn_dir_err = fn_dir / math.sqrt(n_surv) if n_surv else 2.3 / b["n"]     # 0 survivors: 90 % CL limit
    fn, fn_err = {"factorised": (fn_fact, fn_fact_err), "direct": (fn_dir, fn_dir_err),
                  "max": max((fn_fact, fn_fact_err), (fn_dir, fn_dir_err))}[cfg.bkg_estimate]

    n0 = b["per_15y_before"]
    n0_err = n0 * stats["n_before_rel_err"]
    n = fn * n0
    n_err = n * math.hypot(fn_err / fn, n0_err / n0) if fn > 0 else fn_err * n0
    fp, n_sig = s["FP_frac"], s["n"]
    fp_err = wilson_sigma(fp, n_sig)
    return {
        "background": {"TP": 1 - fn, "FN": fn, "FN_err": fn_err,
                       "FN_factorised": fn_fact, "FN_factorised_err": fn_fact_err,
                       "FN_direct": fn_dir, "FN_direct_err": fn_dir_err, "n_survive_direct": n_surv,
                       "n_candidates": b["n"], "events_15y_before": n0, "events_15y_before_err": n0_err,
                       "events_15y_after": n, "events_15y_after_err": n_err},
        "signal": {"TN": 1 - fp, "FP": fp, "FP_err": fp_err, "efficiency": 1 - fp, "efficiency_err": fp_err, "n": n_sig},
        "single_inefficiency": {k: (float(e), float(de)) for k, e, de in zip(TYPES, eps, eps_err)},
    }


def rate_penalty(res, cfg=ObjectiveConfig()):
    """Sum over planes and limits exceeded of 1 + log10(peak / (margin * limit)); 0 if all are respected.

    The 1 makes any violation cost at least w_rate, more than all the other tiers together."""
    p, over = 0.0, []
    for pl in res["planes"]:
        tech = TECHNOLOGIES.get(pl["technology"]) if pl["technology"] else None
        checks = [(pl["peak_hz_cm2"], pl["max_rate_hz_cm2"], "Hz/cm2"),
                  (pl["peak_channel_hz"], tech and tech.get("max_channel_rate"), "Hz/channel")]
        for peak, limit, unit in checks:
            if limit and peak > cfg.rate_margin * limit:
                p += 1 + math.log10(peak / (cfg.rate_margin * limit))
                over.append(f"{pl['plane']}: {peak:.3g} > {cfg.rate_margin:g} x {limit:.3g} {unit}")
    return p, over


def budget_penalty(cost, cfg=ObjectiveConfig()):
    """0 within budget, else 1 + relative overshoot (so >= 1, i.e. >= w_budget in the reward)."""
    return 0.0 if cost <= cfg.cost_budget_kchf else 1 + (cost - cfg.cost_budget_kchf) / cfg.cost_budget_kchf


def cost_kchf(res, cfg=ObjectiveConfig()):
    c = res["cost_kchf"]
    if c is None or not math.isfinite(c):
        c = res["rel_cost"] * cfg.straw_channel_kchf
    return c


def background_term(n, n_max, cfg=ObjectiveConfig()):
    """T_bkg in [0, 1] and dT/dN."""
    if cfg.bkg_form == "exp":
        e = math.exp(-n / cfg.bkg_scale)
        return 1 - e, e / cfg.bkg_scale
    norm = math.log10(1 + n_max / cfg.bkg_target)
    return (math.log10(1 + n / cfg.bkg_target) / norm,
            1 / ((cfg.bkg_target + n) * math.log(10) * norm))


def score(res, stats, cfg=ObjectiveConfig()):
    """Reward (higher is better, <= 0) with its 1-sigma statistical error, and a breakdown for logging."""
    cm = confusion_matrix(res, stats, cfg)
    bg, sg = cm["background"], cm["signal"]
    t_bkg, dt_dn = background_term(bg["events_15y_after"], bg["events_15y_before"], cfg)
    t_bkg_err = dt_dn * bg["events_15y_after_err"]
    clipped = sg["FP"] > cfg.sig_loss_max
    t_sig = min(sg["FP"] / cfg.sig_loss_max, 1.0)
    t_sig_err = 0.0 if clipped else sg["FP_err"] / cfg.sig_loss_max
    cost = cost_kchf(res, cfg)
    t_cost = min(max(math.log(cost / cfg.cost_min_kchf) / math.log(cfg.cost_max_kchf / cfg.cost_min_kchf), 0.0), 1.0)
    p_rate, over = rate_penalty(res, cfg)
    p_budget = budget_penalty(cost, cfg)
    terms = {"rate": (cfg.w_rate * p_rate, 0.0), "budget": (cfg.w_budget * p_budget, 0.0), "background": (cfg.w_bkg * t_bkg, cfg.w_bkg * t_bkg_err),
             "signal": (cfg.w_sig * t_sig, cfg.w_sig * t_sig_err), "cost": (cfg.w_cost * t_cost, 0.0)}
    return {"design": res["design"],
            "reward": -sum(v for v, _ in terms.values()),
            "reward_err": math.sqrt(sum(e * e for _, e in terms.values())),
            "terms": terms, "feasible": not over and not p_budget, "over_rate_limit": over,
            "over_budget": bool(p_budget), "cost_kchf": cost,
            "bkg_events_15y": bg["events_15y_after"], "bkg_events_15y_err": bg["events_15y_after_err"],
            "p_at_least_one_false_vertex": 1 - math.exp(-bg["events_15y_after"]),
            "confusion": cm, "config": asdict(cfg)}


def main():
    stats = sample_stats()
    results = json.loads((paths.results_dir() / "reco" / "results.json").read_text())["results"]
    print("pair composition f_kl (bounced, selected, other):\n", np.round(stats["f_pair"], 5))
    print("effective single muons per type:", dict(zip(TYPES, np.round(stats["n_eff_single"]).astype(int).tolist())))
    for r in sorted((score(r, stats) for r in results), key=lambda s: -s["reward"]):
        cm = r["confusion"]
        bg, sg = cm["background"], cm["signal"]
        print(f"\n{r['design']:<14} reward {r['reward']:9.3f} +- {r['reward_err']:.3f}   "
              + "  ".join(f"{k} {v:.4g}+-{e:.2g}" for k, (v, e) in r["terms"].items()))
        print(f"  false vertices: {r['bkg_events_15y']:.3g} +- {r['bkg_events_15y_err']:.2g} / 15 y "
              f"(P >= 1: {r['p_at_least_one_false_vertex']:.2g}; FN {bg['FN']:.3g} +- {bg['FN_err']:.2g}, "
              f"direct {bg['n_survive_direct']} pairs, FN {bg['FN_direct']:.3g} +- {bg['FN_direct_err']:.2g})")
        print("  single inefficiency: " + ", ".join(f"{k} {e:.3g} +- {de:.2g}" for k, (e, de) in cm["single_inefficiency"].items()))
        print(f"  signal efficiency {100 * sg['efficiency']:.2f} +- {100 * sg['efficiency_err']:.2f} %   "
              f"cost {r['cost_kchf']:.0f} kCHF")
        for o in r["over_rate_limit"]:
            print(f"  OVER RATE LIMIT  {o}")
        if r["over_budget"]:
            print(f"  OVER BUDGET  {r['cost_kchf']:.0f} > {r['config']['cost_budget_kchf']:.0f} kCHF")


if __name__ == "__main__":
    main()
