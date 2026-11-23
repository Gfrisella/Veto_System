#!/usr/bin/env python3
"""Run 2: tracks made in the cavern concrete reaching the SST scoring plane, FairShip TRY_2026_EM_noSBT_full (balloon-only
vessel, no SBT) vs our cavern-rock library (apply_D6_straw_mod_draws, straight lines from the rock surface, no vessel wall).

FairShip: data/origin_sst_noSBT_merged.pkl (Origin_SST_noSBT.py, 3086 files; first point per track on the SST scoring
plane z_FS 83.71 = 81.57 ours, |x| < 2, |y| < 3 m). Rate per 5e13 spill = sum w x 411947589 / (3086 x 1e5); per 4e13 x 0.8.
'Good' FairShip = charged, p > 1 GeV at the plane, real TimeDetPoint hit. Ours: data/ours_rock_tracks.npz
(scripts/ours_rock_spectrum.py; weights per 4e13 spill; acc = SST1 aperture, good = p > 1 GeV + SST1 + timing-detector
acceptance through the field map).
Outputs: results/cavern_summary.json, results/cavern_tables.txt, plots/cavern_*.png
"""
import json, os, pickle, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
FAC5 = 411947589 / (3086 * 1e5)
FAC4 = 0.8 * FAC5
CHARGED = [11, 13, 211, 321, 2212, 3112, 3222, 3312, 3334, 1000010020, 1000020040]
SPEC = [('π±', [211]), ('p/p̄', [2212]), ('e±', [11]), ('K±', [321]), ('μ±', [13]), ('other', [3112, 3222, 3312, 3334, 1000010020, 1000020040])]
OURS_GOOD = 1.84e4

O = pickle.load(open(f'{H}/data/origin_sst_noSBT_merged.pkl', 'rb'))
pdg = np.abs(O['pdg'].astype(int)); w = O['w'].astype(float)
p = np.sqrt(O['px'].astype(float) ** 2 + O['py'].astype(float) ** 2 + O['pz'].astype(float) ** 2)
x0, y0, z0 = (O[k].astype(float) for k in ('x0', 'y0', 'z0'))
td = O['td'].astype(bool); mat = O['mat0']; vol = O['vol0']
q = np.isin(pdg, CHARGED)
conc = mat == 'Concrete'
lines, out = [], {}
def say(t=''):
    print(t); lines.append(t)
def r4(k):
    return w[k].sum() * FAC4, np.sqrt((w[k] ** 2).sum()) * FAC4

say('rates per spill of 4e13 PoT (x1.25 for 5e13); [MC]; err = sqrt(sum w^2)')
# ---- levels by origin material
groups = [('Concrete (cavern)', conc), ('iron (shield / magnet yokes)', mat == 'iron'), ('air (cave)', mat == 'air'),
          ('vessel balloon (mylar_linerHe)', mat == 'mylar_linerHe'), ('helium (decay volume)', mat == 'helium'),
          ('steel', mat == 'steel'), ('Scintillator (UBT etc.)', mat == 'Scintillator'),
          ('other', ~np.isin(mat, ['Concrete', 'iron', 'air', 'mylar_linerHe', 'helium', 'steel', 'Scintillator']))]
say(f'{"made in":32s} {"charged all":>20s} {"p>1":>20s} {"p>1 + TD":>20s}  median p (all charged)')
out['levels'] = {}
for lab, g in groups:
    row = []
    for sel in (q & g, q & g & (p > 1), q & g & (p > 1) & td):
        r = r4(sel); row.append((r[0], r[1], int(sel.sum())))
    out['levels'][lab] = row
    say(f'{lab:32s} ' + ' '.join(f'{a:9.3g}±{b:7.2g} [{n:6d}]'.replace('[', '[')[:20].rjust(20) if False else f'{a:9.3g} ± {b:6.2g} ({n})'.rjust(24) for a, b, n in row)
        + f'   {np.median(p[q & g]) if (q & g).any() else 0:.3f} GeV')
# non-muon concrete
for lab, sel in (('concrete non-μ charged, all', q & conc & (pdg != 13)), ('concrete non-μ, p>1', q & conc & (pdg != 13) & (p > 1)),
                 ('concrete non-μ, p>1 + TD', q & conc & (pdg != 13) & (p > 1) & td)):
    r = r4(sel); say(f'{lab:32s} {r[0]:9.3g} ± {r[1]:6.2g} ({sel.sum()})')
    out[lab] = dict(rate=r[0], err=r[1], mc=int(sel.sum()))

# ---- ours
Z = np.load(f'{H}/data/ours_rock_tracks.npz')
op, ow, oacc, ogood, opdg = Z['p'], Z['w'], Z['acc'].astype(bool), Z['good'].astype(bool), np.abs(Z['pdg'].astype(int))
say(f'ours: charged downstream {ow.sum():.3g}; SST1 aperture (any p) {ow[oacc].sum():.3g} ({oacc.sum()} rows), median p '
    f'{np.median(op[oacc]):.3f} GeV; p>1 in aperture {ow[oacc & (op > 1)].sum():.3g}; good {ow[ogood].sum():.3g} ({ogood.sum()} rows; official 1.84e4)')
out['ours'] = dict(acc=float(ow[oacc].sum()), acc_p1=float(ow[oacc & (op > 1)].sum()), good=float(ow[ogood].sum()), good_official=OURS_GOOD)

# ---- ratios
fs_all = r4(q & conc); fs_p1 = r4(q & conc & (p > 1)); fs_good = r4(q & conc & (p > 1) & td)
say(f'ratios ours/FairShip (concrete, charged): aperture any p {ow[oacc].sum() / fs_all[0]:.2f}; p>1 {ow[oacc & (op > 1)].sum() / fs_p1[0]:.2f}; '
    f'good (official 1.84e4 vs FS p>1+TD) {OURS_GOOD / fs_good[0]:.1f}  [FS good {fs_good[0]:.3g} ± {fs_good[1]:.2g}, '
    f'{(q & conc & (p > 1) & td).sum()} MC, rel. err {fs_good[1] / fs_good[0]:.0%}]')
out['ratios'] = dict(aperture=ow[oacc].sum() / fs_all[0], p1=ow[oacc & (op > 1)].sum() / fs_p1[0], good=OURS_GOOD / fs_good[0],
                     fs_good=fs_good, fs_p1=fs_p1, fs_all=fs_all)

# ---- species
say('species (concrete-made):  FairShip all | FairShip p>1 | FairShip p>1+TD | ours good fraction')
for name, ids in SPEC:
    vals = [r4(q & conc & np.isin(pdg, ids) & c) for c in (p >= 0, p > 1, (p > 1) & td)]
    ns = [(q & conc & np.isin(pdg, ids) & c).sum() for c in (p >= 0, p > 1, (p > 1) & td)]
    fo = ow[ogood & np.isin(opdg, ids)].sum() / ow[ogood].sum()
    say(f'   {name:6s} ' + ' | '.join(f'{v[0]:9.3g} ({n})' for v, n in zip(vals, ns)) + f' | {fo:.2f}')
    out.setdefault('species', {})[name] = dict(fs=[v[0] for v in vals], mc=[int(n) for n in ns], ours_good_frac=fo)

# ---- where: wall / depth
k = q & conc
say(f'concrete-made charged at the plane: x0 pct 1/50/99 {np.percentile(x0[k], [1, 50, 99]).round(2)}, '
    f'y0 {np.percentile(y0[k], [1, 50, 99]).round(2)}, z0 (FS) {np.percentile(z0[k], [1, 50, 99]).round(1)}')
kk = q & conc & (p > 1)
if kk.any():
    say(f'   p>1: x0 {np.percentile(x0[kk], [1, 50, 99]).round(2)}, y0 {np.percentile(y0[kk], [1, 50, 99]).round(2)}, z0 {np.percentile(z0[kk], [1, 50, 99]).round(1)}')
# side walls: the inner faces are at |x| = 4.56 m (FairShip cavern, the same as our -x wall); floor y = -3.36 (ours)
xw_neg, xw_pos = -4.56, 4.56
say(f'   inner wall faces: -x at x = {xw_neg}, +x at x = {xw_pos} (concrete points with x0 > -4.56 lie in the floor/other structures)')
def where(xx, yy):
    lab = np.full(len(xx), 'other', dtype=object)
    lab[xx <= xw_neg + 0.005] = '-x wall'
    lab[xx >= xw_pos - 0.005] = '+x wall'
    lab[(lab == 'other') & (yy < -3.3)] = 'floor'
    lab[(lab == 'other') & (yy > 2.5)] = 'ceiling'
    return lab
wl = where(x0, y0)
for cl, c in (('all', p >= 0), ('p>1', p > 1), ('p>1+TD', (p > 1) & td)):
    s = q & conc & c
    tot = w[s].sum()
    say(f'   share by surface ({cl}): ' + ', '.join(f'{lab} {w[s & (wl == lab)].sum() / tot:.2f}' for lab in ('-x wall', '+x wall', 'floor', 'ceiling', 'other')))
depth = np.where(wl == '-x wall', xw_neg - x0, np.where(wl == '+x wall', x0 - xw_pos, np.nan))
for cl, c in (('all', p >= 0), ('p>1', p > 1)):
    s = q & conc & np.isfinite(depth) & c
    say(f'   depth into the side wall ({cl}): median {np.median(depth[s]) * 100:.1f} cm, 90 % {np.percentile(depth[s], 90) * 100:.1f} cm, '
        f'99 % {np.percentile(depth[s], 99) * 100:.1f} cm (ours good: median 5.3 cm, 90 % 19 cm)')
s = q & conc & (p > 1)
say(f'   creation z (ours frame) of p>1 tracks: 10/50/90 % {np.percentile(z0[s] - 2.14, [10, 50, 90]).round(1)} m (ours: rock exit 55-75 m)')
say(f'   charge of good tracks: positive {w[s & td & (O["pdg"].astype(int) > 0)].sum() / w[s & td].sum():.2f} (ours 0.72)')
say(f'   timing-detector step: FairShip TD-hit fraction of concrete p>1 tracks {w[s & td].sum() / w[s].sum():.3f}; ours good / (aperture, p>1) '
    f'{ow[ogood].sum() / ow[oacc & (op > 1)].sum():.3f}')
say('species FairShip vs ours (per 4e13): aperture any p | p>1 | good ; ratio ours/FS ; median p FS/ours')
for name, ids in SPEC[:5]:
    cells = []
    for fsel, osel in ((p >= 0, oacc), (p > 1, oacc & (op > 1)), ((p > 1) & td, ogood)):
        a = q & conc & np.isin(pdg, ids) & fsel; b = osel & np.isin(opdg, ids)
        fv, ov = w[a].sum() * FAC4, ow[b].sum()
        cells.append(f'FS {fv:8.3g} ({a.sum()}) ours {ov:8.3g} x{ov / fv if fv > 0 else float("inf"):.2g} p~{np.median(p[a]) if a.any() else 0:.2f}/{np.median(op[b]) if b.any() else 0:.2f}')
    say(f'   {name:5s} ' + ' | '.join(cells))
out['where'] = dict(xw_neg=xw_neg, xw_pos=xw_pos)
json.dump(out, open(f'{H}/results/cavern_summary.json', 'w'), indent=1, default=float)
open(f'{H}/results/cavern_tables.txt', 'w').write('\n'.join(lines) + '\n')

# ---- plots
lb = np.logspace(-2, 2, 41)
fig, ax = plt.subplots(figsize=(7.5, 5))
def st(vals, wts, **kw):
    h, _ = np.histogram(np.clip(vals, 1.0001e-2, 99.99), lb, weights=wts); ax.step(lb[:-1], h, where='post', **kw); return h
st(op[oacc], ow[oacc], color='#d62728', lw=1.6, label=f'ours: rock library, SST1 aperture, any p ({ow[oacc].sum():.2g}/spill)')
st(op[ogood], ow[ogood], color='#d62728', ls='--', lw=1.6, label=f'ours: good (p>1 + SST1 + TD acc.) ({ow[ogood].sum():.2g})')
s = q & conc
st(p[s], w[s] * FAC4, color='#1f77b4', lw=1.6, label=f'FairShip: made in concrete, at SST plane ({fs_all[0]:.2g}, {s.sum()} MC)')
s = q & conc & td
st(p[s], w[s] * FAC4, color='#1f77b4', ls='--', lw=1.6, label=f'FairShip: + TD hit ({w[s].sum() * FAC4:.2g}; p>1: {fs_good[0]:.2g}, {(s & (p > 1)).sum()} MC)')
s = q & conc & (pdg != 11)
st(p[s], w[s] * FAC4, color='#2ca02c', lw=1.2, label=f'FairShip: concrete, hadrons + μ only ({w[s].sum() * FAC4:.2g})')
st(op[oacc & (opdg != 11)], ow[oacc & (opdg != 11)], color='#ff7f0e', lw=1.2, label=f'ours: hadrons + μ only, aperture ({ow[oacc & (opdg != 11)].sum():.2g})')
ax.axvline(1, color='grey', lw=0.8, ls=':')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('momentum at SST1 / SST plane [GeV]'); ax.set_ylabel('charged tracks / bin / spill (4e13 PoT)')
ax.set_title('Cavern (concrete) products reaching SST1: FairShip full G4 vs our rock library', fontsize=10)
ax.legend(fontsize=7.5, loc='lower left'); fig.tight_layout(); fig.savefig(f'{H}/plots/cavern_p_spectrum_fs_vs_ours.png', dpi=150); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
for c, lab, ls in ((q & conc, 'all charged', '-'), (q & conc & (p > 1), 'p > 1 GeV', '--')):
    h, e = np.histogram(z0[c] - 2.14, np.arange(25, 90, 1), weights=w[c] * FAC4); axs[0].step(e[:-1], h, where='post', ls=ls, label=f'FairShip {lab}')
    h, e = np.histogram(x0[c], np.arange(-12, 12.01, 0.25), weights=w[c] * FAC4); axs[1].step(e[:-1], h, where='post', ls=ls, label=f'FairShip {lab}')
axs[0].set_xlabel('creation z (ours = z_FS − 2.14) [m]'); axs[1].set_xlabel('creation x [m]')
for a in axs:
    a.set_yscale('log'); a.set_ylabel('tracks / bin / spill (4e13)'); a.legend(fontsize=8)
axs[0].set_title('Where concrete products reaching the SST plane are made'); axs[1].set_title('creation x (side walls)')
fig.tight_layout(); fig.savefig(f'{H}/plots/cavern_creation_z_x.png', dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 5))
for (lab, g), c in zip(groups, plt.cm.tab10.colors):
    s = q & g
    if s.sum() < 5: continue
    h, _ = np.histogram(np.clip(p[s], 1.0001e-2, 99.99), lb, weights=w[s] * FAC4)
    ax.step(lb[:-1], h, where='post', color=c, label=f'{lab} ({s.sum()} MC)')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('momentum at SST plane [GeV]'); ax.set_ylabel('charged tracks / bin / spill (4e13 PoT)')
ax.set_title('FairShip: charged tracks at the SST plane by creation material'); ax.legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(f'{H}/plots/sst_p_by_material.png', dpi=150); plt.close(fig)
print('done')
