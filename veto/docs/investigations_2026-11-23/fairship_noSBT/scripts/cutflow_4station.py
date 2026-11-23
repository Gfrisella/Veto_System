#!/usr/bin/env python3
"""Cut flow p > 1 GeV -> all 4 straw stations -> + timing detector (= good track, veto definition since 2026-11-23),
FairShip concrete-made charged tracks (state at the SST plane, our field map) vs our rock library. Per 4e13 spill.
FairShip also with its real TD hit. Read-only use of the veto lib."""
import os, sys, pickle, json
import numpy as np
W = "/disk/gfs_lhcb/gfrise/Study_on_Veto_System/Optimization_Veto_System/.claude/worktrees/nov4-source/veto"
sys.path[:0] = [W + "/bin", W + "/lib", W + "/docs/investigations_2026-11-20/cavern_rock"]
import _bootstrap  # noqa
from veto import geometry as geo, spectrometer as sp  # noqa
from veto.interactions import charge  # noqa
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
F4 = 0.8 * 411947589 / (3086 * 1e5)
O = pickle.load(open(f'{H}/data/origin_sst_noSBT_merged.pkl', 'rb'))
pdg = O['pdg'].astype(int); q = charge(pdg).astype(float)
P = np.stack([O['px'], O['py'], O['pz']], 1).astype(float); p = np.linalg.norm(P, axis=1)
sel = (O['mat0'] == 'Concrete') & (q != 0) & (p > 1) & (P[:, 2] > 0)
pos = np.stack([O['x'][sel], O['y'][sel], O['z'][sel] - 2.14], 1).astype(float)
mom, qq, ww, td = P[sel], q[sel], O['w'][sel].astype(float) * F4, O['td'][sel].astype(bool)
ap = np.abs(pdg[sel])
b = pos + mom / mom[:, 2:3] * (geo.Z_SST1 - pos[:, 2:3])
good, nst, xt, yt = sp.good_track_acceptance(b, mom, qq)
st4 = nst == 4
R = __import__('rock_study').rock_all("/disk/lhcb_data/sqasim/rock_dis/apply_D6_straw_mod_draws.npz", 300)[0]
c = np.linalg.norm(R['p'], axis=1) > 1
og, onst, _, _ = sp.good_track_acceptance(R['b'][c], R['p'][c], R['q'][c])
ow, opdg = R['w'][c], np.abs(R['pdg'][c])
oacc = R['acc'][c]
rows = []
def line(lab, fs, fs_mc, ours):
    rows.append((lab, fs, fs_mc, ours)); print(f'{lab:42s} FS {fs:9.3g} ({fs_mc:5d} MC)  ours {ours:9.3g}  ours/FS {ours / fs:5.2f}')
print('per 4e13 spill, charged tracks made in concrete / rock')
line('p > 1 GeV, at the SST1 plane aperture', ww.sum(), len(ww), ow[oacc].sum())
line('p > 1 + all 4 straw stations', ww[st4].sum(), int(st4.sum()), ow[onst == 4].sum())
line('p > 1 + 4 stations + TD (our geometry)', ww[good].sum(), int(good.sum()), ow[og].sum())
line('FS: p > 1 + 4 stations + real TD hit', ww[good & td].sum(), int((good & td).sum()), ow[og].sum())
line('FS: p > 1 + real TD hit (any stations)', ww[td].sum(), int(td.sum()), ow[og].sum())
for nm, ids in (('pi', [211]), ('p', [2212]), ('e', [11]), ('K', [321])):
    k, ko = np.isin(ap, ids), np.isin(opdg, ids)
    print(f'   {nm:3s}: p>1 FS {ww[k].sum():8.3g} ours {ow[oacc & ko].sum():8.3g} | 4 st FS {ww[k & st4].sum():8.3g} ours {ow[ko & (onst == 4)].sum():8.3g} | '
          f'good(our geo) FS {ww[k & good].sum():8.3g} ours {ow[ko & og].sum():8.3g} | FS real TD {ww[k & good & td].sum():8.3g}')
print(f'fraction kept by 4 stations: FS {ww[st4].sum() / ww.sum():.3f}, ours {ow[onst == 4].sum() / ow[oacc].sum():.3f}; '
      f'by TD after 4 st: FS {ww[good].sum() / ww[st4].sum():.3f}, ours {ow[og].sum() / ow[onst == 4].sum():.3f}; '
      f'FS real hit | our good {ww[good & td].sum() / ww[good].sum():.3f} (unweighted {(good & td).sum() / good.sum():.3f})')
json.dump([dict(step=a, fs=b_, fs_mc=c_, ours=d) for a, b_, c_, d in rows], open(f'{H}/results/cutflow_4station.json', 'w'), indent=1, default=float)
