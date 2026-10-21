"""Reward of a design: veto.objective (form "sensitivity") on the run_reco result. HIGHER is better, at most 0.

  reward = - 100 * (1 - sensitivity) - cost points - constraint penalties

1. constraints  each plane over its technology's rate limit (Hz per channel or Hz/cm^2) costs
                1000 * (1 + log10(peak / limit)); a design over the 10 MCHF budget costs
                1000 * (1 + overshoot / 10 MCHF). A design with any of these is INFEASIBLE: it loses
                to every feasible design. Getting feasible comes first.
2. sensitivity  = (signal efficiency) x 2.30 / S90(N), in (0, 1]: how small a signal the experiment can
                exclude relative to a perfect detector (no signal lost, no background).
                - signal efficiency = 1 - fraction of signal decays vetoed by random hits (FP);
                - N = background events surviving the selection and the veto in 15 years (FN), all sources:
                  combinatorial (two unrelated muons in time, now tiny); muon interactions in the helium / PVC /
                  planes (their parent crossed the UBT or a fin, but the interaction bends it by a few mrad, so
                  the straight backward search misses the parent's hit: the vertex cone "cone_mrad" recovers it;
                  a parent crossing a single plane leaves no hit with probability 1 - efficiency = 1e-3:
                  "stations": 2 on that plane cures it); neutrino interactions in the helium (no incoming
                  charged particle: only vetoed by random hits, exactly like the signal, a floor of ~0.6);
                - S90(N) = expected 90 % CL upper limit on the signal with N expected background events:
                  2.30 at N = 0, 2.39 at 0.06, 2.71 at 0.3, 3.02 at 0.6, 3.36 at 1, 4.55 at 3, 7.0 at 10.
                1 point = 1 % of sensitivity = about 1 % of signal lost, or about 0.03 background events in
                15 years near N = 0.6 (0.015 near N = 0). Vetoing more of everything to remove neutrinos does
                not pay: each 1 % of signal given up buys back only ~0.07 % of sensitivity.
3. cost         2 points per MCHF up to 5 MCHF, 4 points per MCHF above (10 MCHF = 30 points). Under 1 MCHF is
                very good, ~5 MCHF acceptable if the sensitivity it buys is worth it, 10 MCHF is the hard limit.
Every reward has a 1-sigma statistical error (reward_err, from N and the signal loss):
differences smaller than about 2 x reward_err are noise.
"""

import functools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from veto import objective as vo  # noqa: E402


CONFIG = vo.ObjectiveConfig(bkg_estimate="max", form="sensitivity", cost_budget_kchf=10000.0)
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
