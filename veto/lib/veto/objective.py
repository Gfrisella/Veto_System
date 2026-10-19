"""Objective (reward) for the veto design optimisation, e.g. as the reward of an RL agent.

Input: the result dict of one design as produced by bin/run_reco.py `evaluate()`
(the entries of results/reco/results.json). Nothing is re-simulated here.

Priorities, in order (lexicographic, implemented as tiers with weights 1000 / 100 / 10 / 1):
  0. constraints      rate limits of every plane and the 8 MCHF budget: each violation costs
                      at least 1000, more than all the tiers below together (max 111)
  1. background       goal = "sst1_muons" (default): SST1-acceptance muons the veto misses, per
                      spill, N = sum_k R_k eps_k (R_k muons per spill of type k), term
                      log10(1 + N / sst1_target) / log10(1 + N_before / sst1_target);
                      goal = "pairs": combinatorial pairs that survive the veto = false vertices
                      accepted as true (FN of the background confusion matrix), N events in 15 years
  1b. SST1 muons      (w_mu > 0, goal = "pairs"): every SST1-acceptance muon the veto misses, per spill,
                      log10(1 + N / sst1_target) / log10(1 + N_before / sst1_target)
  2. signal           signal lost to accidental vetoes (FP of the signal confusion matrix)
  3. cost             kCHF

Every tier term is normalised to [0, 1] (the rate penalty is open-ended), so one tier only
competes with the next one when it changes by less than ~1/10 of its range.

Background term (default bkg_form = "exp"):  T_bkg = 1 - exp(-N / bkg_scale).
With bkg_scale = 1 event it is the Poisson probability of at least one false vertex in 15 years.
It saturates for N >> bkg_scale (no gradient there); bkg_form = "log" keeps one.

Time coincidence of the two tracks (coincidence_s): None = NO time coincidence (two muons of the same
1 s spill, no timing detector assumed): N_before = candidate pairs per spill * spills in 15 years.
A half window dt (e.g. 200e-12 with the timing detector) multiplies it by 2 dt / spill.

Background estimate (ISSUES A1): the direct count of surviving pairs is 0-3 out of 196 736, i.e.
Poisson noise. The default estimate factorises the single-muon inefficiencies per muon type,
    N = N_before * sum_kl f_kl eps_k eps_l ,
with f_kl the weighted pair composition of the candidates and eps_k = 1 - single veto efficiency
of type k. On D6_tile / D6_straw it predicts 1.4 / 1.2 surviving pairs vs 3 / 2 observed.

Errors (1 sigma, statistical only): eps_k from the weighted single-muon sample (Wilson interval
with the effective number of muons (sum w)^2 / sum w^2, so eps = 0 still gets an error), N_before
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

from veto import geometry as geo
from veto import paths
from veto.config import DEFAULT_CONFIG
from veto.technology import TECHNOLOGIES

SPILLS_15Y = DEFAULT_CONFIG["pot_15_years"] / DEFAULT_CONFIG["pot_spill"]
N_PAIRS_15Y_BEFORE = 36.32   # combinatorial candidates in 15 years before the veto with +-200 ps (run_reco.py)
TYPES = ("SST1 bounced", "SST1 p>1 GeV & timing", "SST1 other")   # tr_type 0, 1, 2 (run_reco.TYPE_NAMES)


@dataclass
class ObjectiveConfig:
    goal: str = "sst1_muons"          # tier 1: "sst1_muons" (every SST1 muon) | "pairs" (false vertices)
    coincidence_s: float = None       # two-track time coincidence, half window [s]; None = none (same spill, no timing detector)
    spill_s: float = 1.0
    sst1_target: float = 1.0          # sst1_muons: unvetoed SST1 muons per spill regarded as "all vetoed"
    bkg_form: str = "exp"            # "exp": 1 - exp(-N / bkg_scale) | "log": log10(1 + N / bkg_target) / ...
    bkg_scale: float = 1.0            # exp: false vertices in 15 years at which the term reaches 1 - 1/e
    bkg_target: float = 0.01          # log: events in 15 years regarded as background free
    bkg_estimate: str = "factorised"  # "factorised" | "direct" | "max" (conservative)
    eps_weighting: str = "weighted"   # single-muon inefficiency per type: "weighted" (muon weights) |
                                      # "unweighted" (every sampled muon counts once: far less noisy, for ranking)
    sig_loss_max: float = 0.10        # signal loss mapped to the top of the signal tier
    cost_min_kchf: float = 97.2       # straw UBT alone (D0)
    cost_max_kchf: float = 8000.0     # top of the cost tier = the budget
    cost_budget_kchf: float = 8000.0  # above it: penalty w_budget * (1 + (cost - budget) / budget)
    straw_channel_kchf: float = 97.2 / 3600   # converts rel_cost when the area cost is nan (pixels)
    rate_margin: float = 1.0          # penalise above rate_margin * limit (e.g. 0.5 for a factor 2 safety margin)
    w_rate: float = 1000.0            # per limit exceeded: w_rate * (1 + log10(peak / (margin * limit)))
    w_budget: float = 1000.0
    w_bkg: float = 100.0
    w_mu: float = 0.0                 # goal = "pairs": weight of the SST1-muon tier (0 = off)
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
    rate = [ws[ts == k].sum() for k in range(3)]           # SST1 muons per spill of each type
    return {"f_pair": f / f.sum(), "n_eff_single": np.array(n_eff), "rate_single": np.array(rate),
            "pairs_per_spill": float(w.sum()),     # candidates per spill before any time coincidence
            "n_single": np.array([float((ts == k).sum()) for k in range(3)]),
            "n_before_rel_err": float(np.sqrt((w ** 2).sum()) / w.sum())}


def confusion_matrix(res, stats, cfg=ObjectiveConfig()):
    """Background (pairs) and signal confusion matrices with 1-sigma errors; background in events / 15 y."""
    s = res["signal"]
    b = res.get("background")        # None: single muons only (goal = "sst1_muons", fast evaluation)
    f = stats["f_pair"]
    if cfg.eps_weighting == "unweighted" and "single_unweighted" in res:
        eps = np.array([1 - res["single_unweighted"][k] for k in TYPES])
        eps_err = np.array([wilson_sigma(e, n) for e, n in zip(eps, stats["n_single"])])
    else:
        eps = np.array([1 - res["single"][k] for k in TYPES])
        eps_err = np.array([wilson_sigma(e, n) for e, n in zip(eps, stats["n_eff_single"])])
    # pairs are made of muons with p > 1 GeV in the timing acceptance only: their own inefficiency per type
    # (a bounced muon in a pair is mostly a wall bounce; all bounced muons are mostly soft floor bounces)
    pk = "pairable_unweighted" if cfg.eps_weighting == "unweighted" else "pairable"
    if pk in res:
        n_pk = res["pairable_n"] if cfg.eps_weighting == "unweighted" else res["pairable_n_eff"]
        eps_p = np.array([1 - res[pk][k] if n_pk[i] else 0.0 for i, k in enumerate(TYPES)])
        eps_p_err = np.array([wilson_sigma(e, n) if n else 0.0 for e, n in zip(eps_p, n_pk)])
    else:
        eps_p, eps_p_err = eps, eps_err

    fn_fact = float(eps_p @ f @ eps_p)
    fn_fact_err = float(np.sqrt(((2 * f @ eps_p) * eps_p_err) @ ((2 * f @ eps_p) * eps_p_err)))
    if b is None:
        b = dict(n=0, n_survive=-1, FN_frac=math.nan)
        fn_dir = fn_dir_err = math.nan
        n_surv = -1
        fn, fn_err = fn_fact, fn_fact_err
    else:
        n_surv = b["n_survive"]
        fn_dir = b["FN_frac"]
        fn_dir_err = fn_dir / math.sqrt(n_surv) if n_surv else 2.3 / b["n"]     # 0 survivors: 90 % CL limit
        fn, fn_err = {"factorised": (fn_fact, fn_fact_err), "direct": (fn_dir, fn_dir_err),
                      "max": max((fn_fact, fn_fact_err), (fn_dir, fn_dir_err))}[cfg.bkg_estimate]

    coinc = 1.0 if cfg.coincidence_s is None else 2 * cfg.coincidence_s / cfg.spill_s
    n0 = stats["pairs_per_spill"] * coinc * SPILLS_15Y
    n0_err = n0 * stats["n_before_rel_err"]
    n = fn * n0
    n_err = n * math.hypot(fn_err / fn, n0_err / n0) if fn > 0 else fn_err * n0
    fp, n_sig = s["FP_frac"], s["n"]
    fp_err = wilson_sigma(fp, n_sig)
    rate = stats["rate_single"]
    mu_surv, mu_surv_err = rate * eps, rate * eps_err      # unvetoed SST1 muons per spill, per type
    return {
        "sst1_muons": {"per_spill_before": float(rate.sum()),
                       "per_spill_after": float(mu_surv.sum()),
                       "per_spill_after_err": float(np.sqrt((mu_surv_err ** 2).sum())),
                       "TP": float(1 - mu_surv.sum() / rate.sum()), "FN": float(mu_surv.sum() / rate.sum()),
                       "by_type": {k: {"before": float(r), "after": float(a), "after_err": float(e)}
                                   for k, r, a, e in zip(TYPES, rate, mu_surv, mu_surv_err)}},
        "background": {"TP": 1 - fn, "FN": fn, "FN_err": fn_err,
                       "FN_factorised": fn_fact, "FN_factorised_err": fn_fact_err,
                       "FN_direct": fn_dir, "FN_direct_err": fn_dir_err, "n_survive_direct": n_surv,
                       "n_candidates": b["n"], "events_15y_before": n0, "events_15y_before_err": n0_err,
                       "events_15y_after": n, "events_15y_after_err": n_err,
                       "per_spill_after": n / SPILLS_15Y, "per_spill_after_err": n_err / SPILLS_15Y,
                       "coincidence_s": cfg.coincidence_s},
        "signal": {"TN": 1 - fp, "FP": fp, "FP_err": fp_err, "efficiency": 1 - fp, "efficiency_err": fp_err, "n": n_sig},
        "single_inefficiency": {k: (float(e), float(de)) for k, e, de in zip(TYPES, eps, eps_err)},
        "pair_muon_inefficiency": {k: (float(e), float(de)) for k, e, de in zip(TYPES, eps_p, eps_p_err)},
    }



def pairable(tr):
    """Muons that can be in a background pair: p > 1 GeV at SST1 and pointing into the timing detector
    (the candidate pre-selection of make_reco_inputs.py)."""
    p = np.sqrt(tr["bpx"] ** 2 + tr["bpy"] ** 2 + tr["bpz"] ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        dz = geo.Z_TIMING - tr["bz"]
        xt, yt = tr["bx"] + tr["bpx"] / tr["bpz"] * dz, tr["by"] + tr["bpy"] / tr["bpz"] * dz
    return (p > 1) & (tr["bpz"] > 0) & (np.abs(xt) < geo.TIMING_HALF_X) & (np.abs(yt) < geo.TIMING_HALF_Y)


def pairable_entries(vetoed, ty, w, ok):
    """Result entries for objective.confusion_matrix: veto efficiency of the pairable muons per type."""
    out = {"pairable": {}, "pairable_unweighted": {}, "pairable_n": [], "pairable_n_eff": []}
    for k, name in enumerate(TYPES):
        m = ok & (ty == k)
        ww = w[m]
        out["pairable"][name] = float((ww * vetoed[m]).sum() / ww.sum()) if m.any() else 1.0
        out["pairable_unweighted"][name] = float(vetoed[m].mean()) if m.any() else 1.0
        out["pairable_n"].append(float(m.sum()))
        out["pairable_n_eff"].append(float(ww.sum() ** 2 / (ww ** 2).sum()) if m.any() else 0.0)
    return out

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
    if cfg.goal == "sst1_muons":
        mu = cm["sst1_muons"]
        n, n_max = mu["per_spill_after"], mu["per_spill_before"]
        norm = math.log10(1 + n_max / cfg.sst1_target)
        t_bkg = math.log10(1 + n / cfg.sst1_target) / norm
        t_bkg_err = mu["per_spill_after_err"] / ((cfg.sst1_target + n) * math.log(10) * norm)
    else:
        t_bkg, dt_dn = background_term(bg["events_15y_after"], bg["events_15y_before"], cfg)
        t_bkg_err = dt_dn * bg["events_15y_after_err"]
    t_mu = t_mu_err = 0.0
    if cfg.goal != "sst1_muons" and cfg.w_mu:
        mu = cm["sst1_muons"]
        norm = math.log10(1 + mu["per_spill_before"] / cfg.sst1_target)
        t_mu = math.log10(1 + mu["per_spill_after"] / cfg.sst1_target) / norm
        t_mu_err = mu["per_spill_after_err"] / ((cfg.sst1_target + mu["per_spill_after"]) * math.log(10) * norm)
    clipped = sg["FP"] > cfg.sig_loss_max
    t_sig = min(sg["FP"] / cfg.sig_loss_max, 1.0)
    t_sig_err = 0.0 if clipped else sg["FP_err"] / cfg.sig_loss_max
    cost = cost_kchf(res, cfg)
    t_cost = min(max(math.log(cost / cfg.cost_min_kchf) / math.log(cfg.cost_max_kchf / cfg.cost_min_kchf), 0.0), 1.0)
    p_rate, over = rate_penalty(res, cfg)
    p_budget = budget_penalty(cost, cfg)
    terms = {"rate": (cfg.w_rate * p_rate, 0.0), "budget": (cfg.w_budget * p_budget, 0.0), "background": (cfg.w_bkg * t_bkg, cfg.w_bkg * t_bkg_err),
             "sst1_muons": (cfg.w_mu * t_mu, cfg.w_mu * t_mu_err),
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
        mu = cm["sst1_muons"]
        print(f"  unvetoed SST1 muons: {mu['per_spill_after']:.4g} +- {mu['per_spill_after_err']:.2g} per spill "
              f"of {mu['per_spill_before']:.4g} ({100 * mu['FN']:.2f} %): "
              + ", ".join(f"{k} {v['after']:.3g}" for k, v in mu["by_type"].items()))
        print("  single inefficiency: " + ", ".join(f"{k} {e:.3g} +- {de:.2g}" for k, (e, de) in cm["single_inefficiency"].items()))
        print(f"  signal efficiency {100 * sg['efficiency']:.2f} +- {100 * sg['efficiency_err']:.2f} %   "
              f"cost {r['cost_kchf']:.0f} kCHF")
        for o in r["over_rate_limit"]:
            print(f"  OVER RATE LIMIT  {o}")
        if r["over_budget"]:
            print(f"  OVER BUDGET  {r['cost_kchf']:.0f} > {r['config']['cost_budget_kchf']:.0f} kCHF")


if __name__ == "__main__":
    main()
