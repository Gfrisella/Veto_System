#!/usr/bin/env python3
"""TD test (2026-11-23): apply OUR timing acceptance (veto lib: straight line to SST1 aperture, then
spectrometer.timing_acceptance through the field map to Z_TIMING 94.01 ours) to the FairShip concrete-made charged tracks
at the SST scoring plane (z_FS 83.71 = 81.57 ours), and compare with FairShip's real TimeDetPoint hit on the same tracks.
Read-only use of the veto library (worktree nov4-source). Rates per 4e13 spill.
    cd <veto checkout>; export VETO_THREADS=8; source bin/batch/env.sh; python3 td_acceptance_test.py
"""
import os, sys, pickle, json
import numpy as np
W = "/disk/gfs_lhcb/gfrise/Study_on_Veto_System/Optimization_Veto_System/.claude/worktrees/nov4-source/veto"
sys.path[:0] = [W + "/bin", W + "/lib"]
import _bootstrap  # noqa
from veto import geometry as geo, spectrometer  # noqa
from veto.interactions import charge  # noqa
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
F4 = 0.8 * 411947589 / (3086 * 1e5)
O = pickle.load(open(f'{H}/data/origin_sst_noSBT_merged.pkl', 'rb'))
pdg = O['pdg'].astype(int); w = O['w'].astype(float) * F4
q = charge(pdg).astype(float)
P = np.stack([O['px'], O['py'], O['pz']], 1).astype(float); p = np.linalg.norm(P, axis=1)
sel = (O['mat0'] == 'Concrete') & (q != 0) & (p > 1) & (P[:, 2] > 0)
lines = []
def say(t):
    print(t); lines.append(t)
def run(selm, label):
    pos = np.stack([O['x'][selm].astype(float), O['y'][selm].astype(float), O['z'][selm].astype(float) - 2.14], 1)
    mom = P[selm]; qq = q[selm]; ww = w[selm]; td = O['td'][selm].astype(bool); ap = np.abs(pdg[selm])
    b = pos + mom / mom[:, 2:3] * (geo.Z_SST1 - pos[:, 2:3])
    acc = (np.abs(b[:, 0]) < geo.SST1_HALF_X) & (np.abs(b[:, 1]) < geo.SST1_HALF_Y)
    ours = np.zeros(len(b), bool)
    g = np.nonzero(acc)[0]
    ours[g] = spectrometer.timing_acceptance(b[g], mom[g], qq[g])[0]
    T = ww.sum()
    say(f'== {label}: {T:.3g}/spill ({len(ww)} MC)')
    say(f'   SST1 aperture (straight to z {geo.Z_SST1:.2f}): {ww[acc].sum() / T:.3f}')
    say(f'   OUR TD acceptance: {ww[ours].sum() / T:.3f}  ({ww[ours].sum():.3g}/spill, {ours.sum()} MC)')
    say(f'   FairShip real TD hit: {ww[td].sum() / T:.3f}  ({ww[td].sum():.3g}/spill, {td.sum()} MC)')
    say(f'   both {ww[ours & td].sum() / T:.3f}; ours-only {ww[ours & ~td].sum() / T:.3f}; FS-only {ww[~ours & td].sum() / T:.3f}')
    say(f'   FS TD hit among our-accepted: {ww[ours & td].sum() / max(ww[ours].sum(), 1e-30):.3f}')
    for nm, ids in (('pi', [211]), ('p', [2212]), ('e', [11])):
        k = np.isin(ap, ids)
        if k.sum():
            say(f'   {nm:3s}: ours {ww[k & ours].sum() / ww[k].sum():.3f}  FS {ww[k & td].sum() / ww[k].sum():.3f}  '
                f'FS-hit|ours-acc {ww[k & ours & td].sum() / max(ww[k & ours].sum(), 1e-30):.3f}  ({k.sum()} MC)')
    for lo, hi in ((1, 1.5), (1.5, 2), (2, 3), (3, 100)):
        k = (np.linalg.norm(mom, axis=1) > lo) & (np.linalg.norm(mom, axis=1) <= hi)
        if k.sum():
            say(f'   p {lo}-{hi} GeV: ours {ww[k & ours].sum() / ww[k].sum():.3f}  FS {ww[k & td].sum() / ww[k].sum():.3f}  '
                f'FS-hit|ours-acc {ww[k & ours & td].sum() / max(ww[k & ours].sum(), 1e-30):.3f}  ({k.sum()} MC)')
    for s_, nm in ((qq > 0, 'positive'), (qq < 0, 'negative')):
        say(f'   {nm}: ours {ww[s_ & ours].sum() / ww[s_].sum():.3f}  FS {ww[s_ & td].sum() / ww[s_].sum():.3f}  ({s_.sum()} MC)')
    return dict(n=int(len(ww)), rate=T, ours=ww[ours].sum() / T, fs=ww[td].sum() / T)
res = {'concrete p>1': run(sel, 'FairShip concrete-made charged, p > 1 GeV'),
       'all p>1 (any material)': run((q != 0) & (p > 1) & (P[:, 2] > 0), 'FairShip ALL charged at the SST plane, p > 1 GeV (any origin)')}
Z = np.load(f'{H}/data/ours_rock_tracks.npz')
a = Z['acc'].astype(bool) & (Z['p'] > 1)
say(f'== ours (rock library, straight lines): good / (aperture, p>1) = {Z["w"][Z["good"].astype(bool)].sum() / Z["w"][a].sum():.3f}')
json.dump(res, open(f'{H}/results/td_acceptance_test.json', 'w'), indent=1, default=float)
open(f'{H}/results/td_acceptance_test.txt', 'w').write('\n'.join(lines) + '\n')
