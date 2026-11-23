#!/usr/bin/env python3
"""Numbers of TRY_2026_EM_noSBT_full (FairShip full physics, TRY_2026 = TRY6 + 0.3 m M1-M2 gap) for the MS note
(Optimized Muon Shield with Tuned Return Yoke), in the note's units (sample weights x 411947589 / n_input ~ per 5e13).
 1. UBT plane 2 (z_FS 32.28) muons vs the note's Geant4 FastMuon TRY6 UBT map (Fast_readout.pkl, z_FS 32.6), in the
    UBT active area and on the MS7 face (outside the UBT the sample is incomplete: only events with a real hit saved).
 2. DSST1 scoring plane (z_FS 83.71, |x|<2, |y|<3): rates per species, all / p>1 GeV / p>1 + real TD hit.
 3. Muons from the target at DSST1: plane, >=3 straw stations (our field map, veto lib), per P0 range of the note.
Output: results/note_numbers.txt (+ .json). Read-only use of the merged pickles and of the veto lib."""
import os, sys, json, pickle
import numpy as np
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
W = "/disk/gfs_lhcb/gfrise/Study_on_Veto_System/Optimization_Veto_System/.claude/worktrees/nov4-source/veto"
sys.path[:0] = [W + "/bin", W + "/lib"]
import _bootstrap  # noqa
from veto import spectrometer as sp  # noqa
G4 = "/disk/gfs_lhcb/gfrise/Study_on_Magnets/CUDA_validation_TRY6_planes/g4_reference"
NIN = 411947589
out, js = [], {}
def pr(s=''):
    print(s); out.append(s)

# ---- 1. UBT
D = pickle.load(open(f'{H}/data/flux_ms7_merged.pkl', 'rb'))['mu']
F1 = NIN / (2671 * 1e5)
ux, uy, uw = (np.asarray(D[k], float) for k in ('x', 'y', 'w')); uw = uw * F1
g = np.load(f'{G4}/unbiased_TRY6_mu_hists.npz')
Hg = g['ubt_hist_xy'] * NIN / g['total_entries']; c = 0.5 * (g['ubt_binxy'][1:] + g['ubt_binxy'][:-1])
X, Y = np.meshgrid(c, c, indexing='ij')
pr('1. Muons at the UBT plane [Hz]            FairShip full (z 32.28)   note G4 FastMuon (z 32.6)   ratio')
for lab, f in [('UBT active |x|<1.7, -3.34<y<1.7', lambda x, y: (np.abs(x) < 1.7) & (y > -3.34) & (y < 1.7)),
               ('MS7 face |x|<1.71, |y|<1.27', lambda x, y: (np.abs(x) < 1.71) & (np.abs(y) < 1.27))]:
    a, b = uw[f(ux, uy)].sum(), Hg[f(X, Y)].sum()
    pr(f'   {lab:34s} {a:10.3e}               {b:10.3e}             {a / b:.2f}')
    js['ubt_' + lab.split()[0]] = [a, b]
pr(f'   (full 8x8 m: note {Hg.sum():.3e}; FairShip sample incomplete outside the UBT, not comparable)')

# ---- 2. DSST1 composition
O = {k: np.asarray(v) for k, v in pickle.load(open(f'{H}/data/origin_sst_noSBT_merged.pkl', 'rb')).items()
     if k not in ('n_files', 'n_files_from_restarts', 'cluster')}
F2 = NIN / (3086 * 1e5)
w = O['w'].astype(float) * F2; pdg = O['pdg'].astype(int); a = np.abs(pdg)
P = np.stack([O['px'], O['py'], O['pz']], 1).astype(float); p = np.linalg.norm(P, axis=1)
fwd = P[:, 2] > 0; td = O['td'].astype(bool); prim = O['proc'] == 'Transportation'
S = {'mu from target': (a == 13) & prim, 'mu decay in flight': (a == 13) & ~prim, 'e+-': a == 11, 'pi+-': a == 211,
     'p, pbar': a == 2212, 'K+-': a == 321, 'gamma': a == 22, 'n': a == 2112}
S['other'] = ~np.isin(a, [11, 13, 211, 2212, 321, 22, 2112])
def r(m): return w[m].sum() / 1e3, np.sqrt((w[m] ** 2).sum()) / 1e3, int(m.sum())
def f(t): return f'{t[0]:9.3g} ± {t[1]:<7.2g} ({t[2]:>7d})'
pr(); pr('2. DSST1 plane z_FS 83.71, |x|<2, |y|<3 [kHz] (MC)')
pr(f'   {"":20s} {"all":>28s} {"p > 1 GeV":>28s} {"p > 1 GeV + TD hit":>28s}')
for k, m in S.items():
    m = m & fwd; row = [r(m), r(m & (p > 1)), r(m & (p > 1) & td)]; js['sst_' + k] = row
    pr(f'   {k:20s} ' + ' '.join(f(t) for t in row))
ch = np.isin(a, [11, 13, 211, 2212, 321]) & fwd
row = [r(ch), r(ch & (p > 1)), r(ch & (p > 1) & td)]; js['sst_charged'] = row
pr(f'   {"all charged":20s} ' + ' '.join(f(t) for t in row))
nm = ch & (a != 13) & (p > 1) & td
pr('   non-muon charged, p>1 + TD, by creation material [kHz]: ' +
   ', '.join(f'{k} {w[nm & (O["mat0"] == k)].sum() / 1e3:.2f}' for k in np.unique(O['mat0'][nm])))

# ---- 3. muons from the target, station count with our field map
mu = (a == 13) & prim & fwd; i = np.nonzero(mu)[0]
pos = np.stack([O['x'][i], O['y'][i], O['z'][i] - 2.14], 1).astype(float)
_, nst, _, _ = sp.good_track_acceptance(pos, P[i], -np.sign(pdg[i]).astype(float), n_min=3)
st3 = np.zeros(len(a), bool); st3[i] = nst >= 3
p0 = O['p0'].astype(float)
note = {(0, 40): '1 ± 1', (40, 200): '10 ± 2', (200, 400): '5 ± 2', (0, 400): '16 ± 3'}
pr(); pr('3. Muons from the target at DSST1 [kHz]    plane                  >=3 stations (field map)   note (G4 FastMuon, >=3 st. hits)')
for lo, hi in [(0, 40), (40, 200), (200, 400), (0, 400)]:
    m = mu & (p0 >= lo) & (p0 < hi); js[f'mu_P0_{lo}_{hi}'] = [r(m), r(m & st3)]
    pr(f'   P0 in [{lo},{hi}]{"":12s} {f(r(m))}  {f(r(m & st3))}    {note[(lo, hi)]}')
pr('   P0 = momentum of the input muon (FairShip start z 6.25 m); weights 768.75/326.16 dominate the errors.')
open(f'{H}/results/note_numbers.txt', 'w').write('\n'.join(out) + '\n')
json.dump(js, open(f'{H}/results/note_numbers.json', 'w'), indent=1)
