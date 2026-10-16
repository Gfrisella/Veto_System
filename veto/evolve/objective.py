"""Reward of a design: veto.objective (Guglielmo's) on the run_reco result. HIGHER is better, at most 0.

The reward is minus the sum of four tiers, in order of priority:
  1. constraints  each plane over its technology's rate limit (Hz per channel or Hz/cm^2) costs
                  1000 * (1 + log10(peak / limit)); a design over the 8 MCHF budget costs
                  1000 * (1 + overshoot / 8 MCHF). A design with any of these is INFEASIBLE: it loses
                  to every feasible design. Getting feasible comes first.
  2. background   100 * (1 - exp(-N)), N = false vertices (background pairs surviving the veto) in
                  15 years: about 100 * N. 0.01 events cost 1 point. N is the larger of two estimates:
                  the surviving pairs counted directly, and a factorised one (product of how often
                  each muon type escapes the veto) that still works when no pair survives. The
                  factorised one assumes the two muons of a pair escape independently: wrong when
                  both enter through the same gap in the coverage, then the direct count is larger.
  3. signal       10 * (fraction of signal lost to random hits) / 0.10: 1 point per 1 % of signal lost.
  4. cost         1 * log(cost / 97 kCHF) / log(8000 / 97): from the UBT alone (0) to the budget (1);
                  doubling the cost costs 0.16 points.
So, once feasible and with N well below 0.01: 1 % of signal is worth a factor 82 in cost.
Every reward has a 1-sigma statistical error (reward_err, mostly from the signal loss and N):
differences smaller than about 2 x reward_err are noise.
"""

import functools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from veto import objective as vo  # noqa: E402


CONFIG = vo.ObjectiveConfig(bkg_estimate="max")


def configure(**kw):
    """Change the objective settings (veto.objective.ObjectiveConfig fields), e.g. bkg_estimate."""
    global CONFIG
    CONFIG = vo.ObjectiveConfig(**{**vo.asdict(CONFIG), **kw})


@functools.lru_cache(maxsize=1)
def stats():
    return vo.sample_stats()


def evaluate(report):
    """veto.objective.score of a report, or None for a failed design."""
    if report.get("error"):
        return None
    return vo.score(report["result"], stats(), CONFIG)


def score(report):
    """(reward, breakdown dict). Invalid / failed designs get -inf."""
    s = evaluate(report)
    if s is None:
        return -math.inf, {"error": report["error"]}
    return s["reward"], {k: -v for k, (v, _) in s["terms"].items()}
