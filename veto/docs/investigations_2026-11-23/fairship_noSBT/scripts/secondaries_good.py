#!/usr/bin/env python3
"""Secondaries at DSST1 (not input muons, charged, pz>0) with the note's track requirement:
p > 1 GeV/c + real FairShip TD hit + >= 3 of the 4 straw stations inside |x|<2, |y|<3 (RK4 in our field map from the
state at the SST scoring plane, veto lib good_track_acceptance with n_min=3). Also all 4 stations for reference.
Writes results/secondaries_good.npz (row index, n_stations) and results/secondaries_good.txt (table by origin/species)."""
import os, sys, pickle
import numpy as np
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
W = "/disk/gfs_lhcb/gfrise/Study_on_Veto_System/Optimization_Veto_System/.claude/worktrees/nov4-source/veto"
sys.path[:0] = [W + "/bin", W + "/lib"]
import _bootstrap  # noqa
from veto import spectrometer as sp  # noqa
F = 411947589 / (3086 * 1e5)
O = {k: np.asarray(v) for k, v in pickle.load(open(f'{H}/data/origin_sst_noSBT_merged.pkl', 'rb')).items()
     if k not in ('n_files', 'n_files_from_restarts', 'cluster')}
w = O['w'].astype(float) * F; pdg = O['pdg'].astype(int); a = np.abs(pdg)
P = np.stack([O['px'], O['py'], O['pz']], 1).astype(float); p = np.linalg.norm(P, axis=1)
sec = (P[:, 2] > 0) & (O['proc'] != 'Transportation') & np.isin(a, [11, 13, 211, 2212, 321])
i = np.nonzero(sec & (p > 1))[0]
pos = np.stack([O['x'][i], O['y'][i], O['z'][i] - 2.14], 1).astype(float)
q = np.where(np.isin(pdg[i], [11, 13]), -np.sign(pdg[i]), np.sign(pdg[i])).astype(float)
_, nst, _, _ = sp.good_track_acceptance(pos, P[i], q, n_min=3)
td = O['td'][i].astype(bool)
np.savez(f'{H}/results/secondaries_good.npz', idx=i, nst=nst, td=td)
mat, vol = O['mat0'][i], O['vol0'][i]
G = {'MS7': (mat == 'iron') & (vol == 'Magn7'), 'MS4-MS6': (mat == 'iron') & (vol != 'Magn7'),
     'Cavern walls': mat == 'Concrete', 'Cavern air': mat == 'air',
     'Decay vessel': np.isin(mat, ['helium', 'mylar_linerHe'])}
G['other'] = ~np.any(list(G.values()), axis=0)
S = {'mu (13)': a[i] == 13, 'e (11)': a[i] == 11, 'pi (211)': a[i] == 211, 'p (2212)': a[i] == 2212, 'K (321)': a[i] == 321}
ww = w[i]
out = []
for lab, cut in [('p>1', np.ones(len(i), bool)), ('p>1 + TD', td), ('p>1 + TD + >=3 st', td & (nst >= 3)),
                 ('p>1 + TD + 4 st', td & (nst == 4))]:
    out.append(f'== {lab} [kHz]   ' + '  '.join(f'{k:>9s}' for k in S) + '      total     MC')
    for g, gm in G.items():
        out.append(f'  {g:14s} ' + '  '.join(f'{ww[gm & sm & cut].sum() / 1e3:9.3f}' for sm in S.values()) +
                   f'  {ww[gm & cut].sum() / 1e3:9.2f} {(gm & cut).sum():6d}')
    t = cut
    out.append(f'  {"total":14s} {ww[t].sum() / 1e3:.2f} ± {np.sqrt((ww[t] ** 2).sum()) / 1e3:.2f} kHz ({t.sum()} MC)')
open(f'{H}/results/secondaries_good.txt', 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))
