#!/usr/bin/env python3
"""Our rock 'good' rate with the timing-detector aperture of the FairShip noSBT_full production (4 x 6 m, the hits end at
|x| 2.0, |y| 3.0) instead of the library's 5.5 x 6.5 m. Read-only (rock_study.rock_all + spectrometer.timing_acceptance)."""
import sys
import numpy as np
C = "/disk/gfs_lhcb/gfrise/Study_on_Veto_System/Optimization_Veto_System/.claude/worktrees/nov4-source/veto/docs/investigations_2026-11-20/cavern_rock"
sys.path.insert(0, C)
from rock_study import rock_all  # noqa
from veto import spectrometer  # noqa
R, _, _ = rock_all("/disk/lhcb_data/sqasim/rock_dis/apply_D6_straw_mod_draws.npz", 300)
g = np.nonzero(R['good'])[0]
m, xt, yt = spectrometer.timing_acceptance(R['b'][g], R['p'][g], R['q'][g])
w = R['w'][g]; pdg = np.abs(R['pdg'][g])
print(f'ours good (5.5x6.5 TD): {w[m].sum():.4g}/spill (4e13), {m.sum()} rows')
for xe, ye in ((2.2, 3.0), (2.0, 3.0)):
    k = m & (np.abs(xt) < xe) & (np.abs(yt) < ye)
    print(f'ours good with TD |x|<{xe}, |y|<{ye}: {w[k].sum():.4g}/spill ({k.sum()} rows) = {w[k].sum() / w[m].sum():.3f} of good; '
          + ', '.join(f'{n} {w[k & np.isin(pdg, i)].sum():.3g}' for n, i in (('pi', [211]), ('p', [2212]), ('e', [11]), ('K', [321]))))
