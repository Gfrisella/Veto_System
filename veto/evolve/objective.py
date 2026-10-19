"""Reward of a design: veto.objective (Guglielmo's) on the run_reco result. HIGHER is better, at most 0.

The reward is minus the sum of four tiers, in order of priority:
  1. constraints  each plane over its technology's rate limit (Hz per channel or Hz/cm^2) costs
                  1000 * (1 + log10(peak / limit)); a design over the 8 MCHF budget costs
                  1000 * (1 + overshoot / 8 MCHF). A design with any of these is INFEASIBLE: it loses
                  to every feasible design. Getting feasible comes first.
  2. background   100 * (1 - exp(-N)), N = false vertices surviving the selection and the veto in 15 years:
                  about 100 * N for small N, saturating at 100. 0.01 events cost 1 point. N has three sources:
                  - combinatorial: two unrelated muons coincident in time (now tiny, ~1e-3);
                  - muon interactions: a muon interacting in the helium or the PVC walls makes a real vertex;
                    its parent crossed the UBT or a fin, but the interaction bends it by a few mrad, so the
                    straight backward search misses the parent's hit: the vertex cone ("cone_mrad") recovers it;
                  - neutrino interactions: no incoming charged particle, nothing to veto: a floor of ~0.5.
                  Selection for every sample: measured IP < 0.1 m at the target, at most 2 tracks in SST1.
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
INPUTS = "inputs.npz"          # results/reco file the designs are evaluated on (evolve_designs --inputs)


def configure(**kw):
    """Change the objective settings (veto.objective.ObjectiveConfig fields), e.g. bkg_estimate."""
    global CONFIG
    CONFIG = vo.ObjectiveConfig(**{**vo.asdict(CONFIG), **kw})


@functools.lru_cache(maxsize=1)
def stats():
    import numpy as np
    from veto import paths
    return vo.sample_stats(np.load(paths.results_dir() / "reco" / INPUTS))


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
