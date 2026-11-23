#!/usr/bin/env python3
"""Back-up slide plot (deck bt_beamer): momentum at SST1 of the charged tracks made in the cavern concrete,
FairShip TRY_2026_EM_noSBT_full vs our rock library, all (in the SST1 aperture) and good tracks (p > 1 GeV, all 4 straw
stations + timing detector, our field map on both). Weighted, per spill of 4e13 PoT.
Out: plots/fs_vs_ours_cavern_p.{png,pdf}; numbers in results/slide_numbers.json."""
import os, sys, pickle, json
import numpy as np
W = "/disk/gfs_lhcb/gfrise/Study_on_Veto_System/Optimization_Veto_System/.claude/worktrees/nov4-source/veto"
sys.path[:0] = [W + "/bin", W + "/lib", W + "/docs/investigations_2026-11-20/cavern_rock",
                "/disk/data1/hep/gfrise/.claude-config/skills/cern-ship-slides/theme"]
import _bootstrap  # noqa
from veto import geometry as geo, spectrometer as sp  # noqa
from veto.interactions import charge  # noqa
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt  # noqa
from plotstyle import C1, C2, save  # noqa
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
F4 = 0.8 * 411947589 / (3086 * 1e5)
O = pickle.load(open(f'{H}/data/origin_sst_noSBT_merged.pkl', 'rb'))
pdg = O['pdg'].astype(int); q = charge(pdg).astype(float)
P = np.stack([O['px'], O['py'], O['pz']], 1).astype(float); p = np.linalg.norm(P, axis=1)
sel = (O['mat0'] == 'Concrete') & (q != 0) & (P[:, 2] > 0)
pos = np.stack([O['x'][sel], O['y'][sel], O['z'][sel] - 2.14], 1).astype(float)
mom, qq, ww, ps = P[sel], q[sel], O['w'][sel].astype(float) * F4, p[sel]
b = pos + mom / mom[:, 2:3] * (geo.Z_SST1 - pos[:, 2:3])
acc = (np.abs(b[:, 0]) < geo.SST1_HALF_X) & (np.abs(b[:, 1]) < geo.SST1_HALF_Y)
fg = np.zeros(len(ps), bool); h1 = ps > 1
fg[h1] = sp.good_track_acceptance(b[h1], mom[h1], qq[h1])[0]
from rock_study import rock_all  # noqa
R = rock_all("/disk/lhcb_data/sqasim/rock_dis/apply_D6_straw_mod_draws.npz", 300)[0]
op, ow, oacc, og = np.linalg.norm(R['p'], axis=1), R['w'], R['acc'], R['good']
lb = np.logspace(-2, 2, 33)
fig, ax = plt.subplots(figsize=(8, 5))
def st(v, w, **kw):
    h, _ = np.histogram(np.clip(v, 1.0001e-2, 99.9), lb, weights=w); ax.stairs(h, lb, lw=2, **kw)
st(ps[acc], ww[acc], color=C1, label=f'FairShip, all: {ww[acc].sum():.2g}')
st(op[oacc], ow[oacc], color=C2, label=f'ours, all: {ow[oacc].sum():.2g}')
st(ps[fg], ww[fg], color=C1, ls='--', label=f'FairShip, good: {ww[fg].sum():.2g}')
st(op[og], ow[og], color=C2, ls='--', label=f'ours, good: {ow[og].sum():.2g}')
ax.axvline(1, color='0.5', lw=1, ls=':')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(10, None)
ax.set_xlabel('momentum at SST1 [GeV]'); ax.set_ylabel('charged tracks / bin / spill')
ax.set_title('Charged tracks from the cavern concrete at SST1 (4e13 PoT)')
ax.grid(alpha=0.3); ax.legend(frameon=True, loc='upper right')
fig.tight_layout()
save(fig, f'{H}/plots/fs_vs_ours_cavern_p.png')
num = dict(fs_all=ww[acc].sum(), ours_all=float(ow[oacc].sum()), fs_good=ww[fg].sum(), fs_good_err=float(np.sqrt((ww[fg] ** 2).sum())),
           fs_good_mc=int(fg.sum()), ours_good=float(ow[og].sum()))
json.dump(num, open(f'{H}/results/slide_numbers.json', 'w'), indent=1, default=float); print(num)
