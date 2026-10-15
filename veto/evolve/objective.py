"""Score of a veto design from its evaluation report (HIGHER is better). PLACEHOLDER objective.

Replace `score` with the real objective when it is ready; evolve_designs.py only calls score(report).
The report is what evaluate_designs.py returns for one design (see its docstring).

Placeholder, in "signal-efficiency percent" units:
    score = 100 * signal_kept                                   (TNR: signal not lost to random hits)
            - COST_PER_MCHF * cost [MCHF]                        (one cost model: kCHF from technology.py)
            - RATE_PENALTY * planes over their rate limit
            - BKG_PENALTY * max(0, log10(B_15y_upper / B_TARGET))
with B_15y_upper = surviving background in 15 years, or its 90 % upper limit when nothing survives.
So: keep the background below B_TARGET, then trade signal efficiency against cost; a plane over its
rate limit costs as much as RATE_PENALTY points of signal efficiency.
"""

import math

B_TARGET = 0.1          # background events in 15 years
COST_PER_MCHF = 0.1     # 1 MCHF costs 0.1 % of signal efficiency
RATE_PENALTY = 1.0      # per plane over its technology's rate limit
BKG_PENALTY = 20.0      # per decade of background above B_TARGET


def score(report):
    """(score, breakdown dict). Invalid / failed designs get -inf."""
    if report.get("error"):
        return -math.inf, {"error": report["error"]}
    f = report["fom"]
    b_upper = f["B_15y"] + 1.64 * f["B_15y_err"] if f["B_15y"] > 0 else f["B_15y_err"]
    terms = {
        "signal_kept": 100 * f["TNR"],
        "cost": -COST_PER_MCHF * report["cost_kchf"] / 1000,
        "rate_limits": -RATE_PENALTY * report["n_planes_over_limit"],
        "background": -BKG_PENALTY * max(0.0, math.log10(max(b_upper, 1e-12) / B_TARGET)),
    }
    return sum(terms.values()), terms
