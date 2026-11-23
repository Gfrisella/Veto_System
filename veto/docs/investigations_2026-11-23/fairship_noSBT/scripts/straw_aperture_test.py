#!/usr/bin/env python3
"""Exact straw-aperture test (2026-11-23). Acceptance variants on the same tracks:
  A  current lib : SST1 aperture (straight to 82.08) + field map to TD 94.01, TD 5.5 x 6.5, not lost in the yoke
  B  approx      : A + |x|<2, |y|<3 at the TD plane (the earlier 4 x 6 proxy)
  C  exact       : A + |x|<2, |y|<3 at SST2 84.08, SST3 91.08, SST4 93.08 (field-map transport to each station; FairShip's
                   solid 1 m steel frames absorb anything outside the straw aperture)
Applied to the FairShip concrete-made / all charged tracks (p>1, state at the SST plane 81.57 ours) with their real TD hit,
and to our rock library tracks (rock_study.rock_all). Read-only use of the veto lib. Rates per 4e13 spill."""
import os, sys, pickle, json
import numpy as np
W = "/disk/gfs_lhcb/gfrise/Study_on_Veto_System/Optimization_Veto_System/.claude/worktrees/nov4-source/veto"
C = W + "/docs/investigations_2026-11-20/cavern_rock"
sys.path[:0] = [W + "/bin", W + "/lib", C]
import _bootstrap  # noqa
from veto import geometry as geo, spectrometer as sp  # noqa
from veto.interactions import charge  # noqa
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
HX, HY = geo.SST1_HALF_X, geo.SST1_HALF_Y

def variants(b, mom, q):
    """b: state at SST1 (z = Z_SST1) inside the SST1 aperture; returns masks A, B, C and per-station failure masks."""
    pt, _, lost_t = sp.to_z(b, mom, q, geo.Z_TIMING)
    A = ~lost_t & (np.abs(pt[:, 0]) < geo.TIMING_HALF_X) & (np.abs(pt[:, 1]) < geo.TIMING_HALF_Y)
    B = A & (np.abs(pt[:, 0]) < HX) & (np.abs(pt[:, 1]) < HY)
    inside = np.ones(len(b), bool); fail = {}
    for name, z in zip(('SST2', 'SST3', 'SST4'), geo.Z_SST[1:]):
        ps, _, lost = sp.to_z(b, mom, q, z)
        ok = ~lost & (np.abs(ps[:, 0]) < HX) & (np.abs(ps[:, 1]) < HY)
        fail[name] = inside & ~ok          # first station where the track leaves the aperture
        inside &= ok
    Cm = A & inside
    insideD = inside.copy()                # D: also at the front / back faces of each 1 m deep station (z +- 0.5 m)
    for z in geo.Z_SST[1:]:
        for dz in (-0.5, 0.5):
            ps, _, lost = sp.to_z(b, mom, q, z + dz)
            insideD &= ~lost & (np.abs(ps[:, 0]) < HX) & (np.abs(ps[:, 1]) < HY)
    fail['faces (D only)'] = inside & ~insideD
    return A, B, Cm, A & insideD, fail

lines, res = [], {}
def say(t):
    print(t); lines.append(t)
say(f'stations z (ours): SST {[round(z, 2) for z in geo.Z_SST]}, TD {geo.Z_TIMING:.2f}; straw aperture |x|<{HX}, |y|<{HY}; TD {geo.TIMING_HALF_X} x {geo.TIMING_HALF_Y}')

# ---- FairShip tracks
F4 = 0.8 * 411947589 / (3086 * 1e5)
O = pickle.load(open(f'{H}/data/origin_sst_noSBT_merged.pkl', 'rb'))
pdg = O['pdg'].astype(int); q = charge(pdg).astype(float)
P = np.stack([O['px'], O['py'], O['pz']], 1).astype(float); p = np.linalg.norm(P, axis=1)
for lab, sel in (('FairShip concrete p>1', (O['mat0'] == 'Concrete') & (q != 0) & (p > 1) & (P[:, 2] > 0)),
                 ('FairShip all origins p>1', (q != 0) & (p > 1) & (P[:, 2] > 0))):
    pos = np.stack([O['x'][sel], O['y'][sel], O['z'][sel] - 2.14], 1).astype(float)
    mom, qq, ww, td = P[sel], q[sel], O['w'][sel].astype(float) * F4, O['td'][sel].astype(bool)
    b = pos + mom / mom[:, 2:3] * (geo.Z_SST1 - pos[:, 2:3])
    acc = (np.abs(b[:, 0]) < HX) & (np.abs(b[:, 1]) < HY)
    A, B, Cm, D, fail = variants(b, mom, qq)
    A &= acc; B &= acc; Cm &= acc; D &= acc
    T = ww.sum()
    say(f'== {lab}: {T:.3g}/spill ({len(ww)} MC); FairShip real TD hit {ww[td].sum() / T:.3f} ({td.sum()} MC)')
    for nm, m in (('A current lib', A), ('B approx 4x6 at TD', B), ('C straw SST2-4 centre', Cm), ('D straw SST2-4 faces', D)):
        say(f'   {nm:22s}: acc {ww[m].sum() / T:.3f} ({m.sum()} MC); FS hit | acc  w {ww[m & td].sum() / max(ww[m].sum(), 1e-30):.3f} '
            f'unw {(m & td).sum() / max(m.sum(), 1):.3f}; FS hits outside acc {ww[~m & td].sum() / max(ww[td].sum(), 1e-30):.3f}')
    say('   tracks in A removed by C, first station left: ' + ', '.join(f'{k} {ww[A & v].sum() / T:.3f}' for k, v in fail.items()))
    say(f'   B vs C on the same tracks: in B not C {ww[B & ~Cm].sum() / T:.3f}, in C not B {ww[Cm & ~B].sum() / T:.3f}')
    res[lab] = dict(fs=ww[td].sum() / T, A=ww[A].sum() / T, B=ww[B].sum() / T, C=ww[Cm].sum() / T, D=ww[D].sum() / T)

# ---- our rock tracks
from rock_study import rock_all  # noqa
R, _, _ = rock_all("/disk/lhcb_data/sqasim/rock_dis/apply_D6_straw_mod_draws.npz", 300)
g = np.nonzero(R['acc'] & (np.linalg.norm(R['p'], axis=1) > 1))[0]
b, mom, qq, ww, pdg_r = R['b'][g], R['p'][g], R['q'][g].astype(float), R['w'][g], np.abs(R['pdg'][g])
A, B, Cm, D, fail = variants(b, mom, qq)
say(f'== ours rock, SST1 aperture p>1: {ww.sum():.4g}/spill ({len(ww)} rows); lib good (check) {R["w"][R["good"]].sum():.4g}')
for nm, m in (('A current lib', A), ('B approx 4x6 at TD', B), ('C straw SST2-4 centre', Cm), ('D straw SST2-4 faces', D)):
    say(f'   {nm:22s}: good {ww[m].sum():.4g}/spill ({m.sum()} rows), frac {ww[m].sum() / ww.sum():.3f}; '
        + ', '.join(f'{n} {ww[m & np.isin(pdg_r, i)].sum():.3g}' for n, i in (('pi', [211]), ('p', [2212]), ('e', [11]), ('K', [321]))))
say('   removed by C, first station left: ' + ', '.join(f'{k} {ww[A & v].sum() / ww.sum():.3f}' for k, v in fail.items()))
FS_GOOD, FS_ERR = 5.48e3, 1.4e3
for nm, m in (('A', A), ('B', B), ('C', Cm), ('D', D)):
    r = ww[m].sum() / FS_GOOD
    say(f'   ours/FairShip good ({nm}): {r:.2f} ± {r * FS_ERR / FS_GOOD:.2f}')
    res[f'ours_{nm}'] = float(ww[m].sum())
json.dump(res, open(f'{H}/results/straw_aperture_test.json', 'w'), indent=1, default=float)
open(f'{H}/results/straw_aperture_test.txt', 'w').write('\n'.join(lines) + '\n')
