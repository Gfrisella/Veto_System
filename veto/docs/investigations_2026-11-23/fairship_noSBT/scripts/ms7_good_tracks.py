#!/usr/bin/env python3
"""Charged tracks at SST1 made in MS7, all vs GOOD (p > 1 GeV, straw aperture at all 4 stations + timing detector,
spectrometer.good_track_acceptance, field map), FairShip TRY_2026_EM_noSBT_full vs our G4 10.7 MS7 chain.

FairShip: origin run (3086 files), tracks with vol0 == Magn7 at the SST scoring plane (z_FS 83.71, |x|<2, |y|<3),
  straight to our SST1 (82.08 ours), then the good-track filter. K_L-decay tracks: mother K_L made at z_FS 28.0-32.18.
Ours: results/shield_dis/shield_muon_fs2026_geant4v1073.npz (G4 10.7 chain of generators_natural.md, per 5e13 -> x0.8),
  interaction products (code 3) and forced K_L-decay daughters (code 4), straight line to SST1, same filter.
Rates per spill of 4e13 PoT (= Hz). Out: results/ms7_good_tracks.{json,txt}
"""
import os, sys, pickle, json
import numpy as np
W = "/disk/gfs_lhcb/gfrise/Study_on_Veto_System/Optimization_Veto_System/.claude/worktrees/nov4-source/veto"
sys.path[:0] = [W + "/lib"]
from veto import geometry as geo, spectrometer as sp  # noqa
from veto.interactions import charge  # noqa
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
F4 = 0.8 * 411947589 / (3086 * 1e5)
CH = {'π±': [211], 'p': [2212], 'K±': [321], 'e±': [11], 'μ±': [13]}
out, lines = {}, []


def say(t):
    print(t); lines.append(t)


def good_of(b, mom, q):
    g = np.zeros(len(b), bool); h = np.linalg.norm(mom, axis=1) > 1
    if h.any():
        g[h] = sp.good_track_acceptance(b[h], mom[h], q[h])[0]
    return g


def at_sst1(pos, mom):
    b = pos + mom / mom[:, 2:3] * (geo.Z_SST1 - pos[:, 2:3])
    return b, (np.abs(b[:, 0]) < geo.SST1_HALF_X) & (np.abs(b[:, 1]) < geo.SST1_HALF_Y)


# ---------------- FairShip
O = pickle.load(open(f'{H}/data/origin_sst_noSBT_merged.pkl', 'rb'))
pdg = O['pdg'].astype(int); apdg = np.abs(pdg)
sel = (O['vol0'] == 'Magn7') & np.isin(apdg, sum(CH.values(), [])) & (O['pz'] > 0)
pos = np.stack([O['x'][sel], O['y'][sel], O['z'][sel] - 2.14], 1).astype(float)
mom = np.stack([O['px'][sel], O['py'][sel], O['pz'][sel]], 1).astype(float)
w = O['w'][sel].astype(float) * F4; a = apdg[sel]
kl = (np.abs(O['mpdg'][sel]) == 130) & (O['mz0'][sel] > 28.0) & (O['mz0'][sel] <= 32.18)
b, acc = at_sst1(pos, mom)
g = good_of(b, mom, charge(pdg[sel]).astype(float)) & acc
# K_L-decay tracks: any volume (decay in air / vessel), mother made in MS7
selk = np.isin(apdg, sum(CH.values(), [])) & (O['pz'] > 0) & (np.abs(O['mpdg']) == 130) & (O['mz0'] > 28.0) & (O['mz0'] <= 32.18)
posk = np.stack([O['x'][selk], O['y'][selk], O['z'][selk] - 2.14], 1).astype(float)
momk = np.stack([O['px'][selk], O['py'][selk], O['pz'][selk]], 1).astype(float)
wk = O['w'][selk].astype(float) * F4
bk, acck = at_sst1(posk, momk)
gk = good_of(bk, momk, charge(pdg[selk]).astype(float)) & acck


def rr(ww, m):
    return dict(rate=float(ww[m].sum()), err=float(np.sqrt((ww[m] ** 2).sum())), mc=int(m.sum()))


fs = {}
for n, ids in CH.items():
    m = np.isin(a, ids)
    fs[n] = dict(all=rr(w, m & acc), good=rr(w, m & g))
fs['all charged'] = dict(all=rr(w, acc), good=rr(w, g))
fs['K_L-decay tracks'] = dict(all=rr(wk, acck), good=rr(wk, gk))

# ---------------- ours (G4 10.7 MS7 chain)
Z = np.load('/disk/gfs_lhcb/gfrise/Study_on_Veto_System/Optimization_Veto_System/veto/results/shield_dis/shield_muon_fs2026_geant4v1073.npz')
wi = Z['int_weight'] * 0.8; code = Z['int_code']; v0 = Z['int_v0']
pi, ppdg, P, V = Z['part_int'], Z['part_pdg'], Z['part_p'][:, :3].astype(float), Z['part_v'].astype(float)
ww = wi[pi]; ap = np.abs(ppdg)
ch = np.isin(ap, sum(CH.values(), [])) & (P[:, 2] > 0)
c3 = ch & (code[pi] == 3)
dk = ch & (code[pi] == 4) & (V[:, 2] > 29.91) & (v0[pi] == 130)
ours = {}
for lab, base in (('prod', c3), ('kl', dk)):
    idx = np.nonzero(base)[0]
    bb, ac = at_sst1(V[idx], P[idx])
    gg = good_of(bb, P[idx], charge(ppdg[idx]).astype(float)) & ac
    A = np.zeros(len(ap), bool); G = np.zeros(len(ap), bool); A[idx] = ac; G[idx] = gg
    if lab == 'prod':
        for n, ids in CH.items():
            m = np.isin(ap, ids)
            ours[n] = dict(all=rr(ww, m & A), good=rr(ww, m & G))
        ours['all charged'] = dict(all=rr(ww, A), good=rr(ww, G))
    else:
        ours['K_L-decay tracks'] = dict(all=rr(ww, A), good=rr(ww, G))

say('MS7-made charged tracks at SST1, per spill 4e13 (Hz): all (aperture 4x6) | good (p>1, 4 straw stations + TD)')
say(f'{"":18s} {"FS all":>10s} {"ours all":>10s} {"FS good":>16s} {"ours good":>16s}  ours/FS good')
for n in list(CH) + ['all charged', 'K_L-decay tracks']:
    f, o = fs[n], ours[n]
    rt = o['good']['rate'] / f['good']['rate'] if f['good']['rate'] > 0 else float('nan')
    say(f'{n:18s} {f["all"]["rate"]:10.3g} {o["all"]["rate"]:10.3g} {f["good"]["rate"]:8.3g}±{f["good"]["err"]:<6.2g}'
        f'({f["good"]["mc"]:3d}) {o["good"]["rate"]:8.3g} ({o["good"]["mc"]:5d})  {rt:.2f}')
json.dump(dict(fairship=fs, ours=ours), open(f'{H}/results/ms7_good_tracks.json', 'w'), indent=1)
open(f'{H}/results/ms7_good_tracks.txt', 'w').write('\n'.join(lines) + '\n')
