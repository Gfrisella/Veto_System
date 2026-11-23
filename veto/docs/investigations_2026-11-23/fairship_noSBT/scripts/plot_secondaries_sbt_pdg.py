#!/usr/bin/env python3
"""Secondaries at DSST1 (charged, not input muons, pz > 0), p > 1 GeV/c and good tracks (p > 1 + real TD hit + >= 3 of
4 straw stations, results/secondaries_good.npz from secondaries_good.py). Rates per spill. z in the FairShip frame.
 1. secondaries_sbt_walls.png : where they enter the decay vessel through the -x and +x lateral walls (= SBT position),
    rate map in (z, y) on each wall. Trajectory = straight line from the production point to the DSST1 crossing (the
    region upstream of DSST1 is field free; multiple scattering ignored). Vessel = He balloon frustum, z 32.85-82.85,
    half-widths x 0.5 -> 2.0, y 1.35 -> 3.0 m. Entry = last outside -> inside crossing of the frustum before DSST1.
 2. secondaries_pdg.png : PDG code of the particle and of its mother, p > 1 vs good track.
Output also results/secondaries_sbt_walls.txt (entry face fractions)."""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT = f'{H}/plots/note'
F = 411947589 / (3086 * 1e5)
O = {k: np.asarray(v) for k, v in pickle.load(open(f'{H}/data/origin_sst_noSBT_merged.pkl', 'rb')).items()
     if k not in ('n_files', 'n_files_from_restarts', 'cluster')}
G = np.load(f'{H}/results/secondaries_good.npz')
i = G['idx']; good = G['td'] & (G['nst'] >= 3)              # rows: secondaries with p > 1
w = O['w'][i].astype(float) * F; pdg = O['pdg'][i].astype(int); mpdg = O['mpdg'][i].astype(int)
x0, y0, z0 = (O[k][i].astype(float) for k in ('x0', 'y0', 'z0'))
x1, y1, z1 = (O[k][i].astype(float) for k in ('x', 'y', 'z'))

# ------------------------------------------------------------------ vessel entry along the straight line
ZA, ZB = 32.85, 82.85
def half(z):
    f = np.clip((z - ZA) / (ZB - ZA), 0, 1)
    return 0.5 + 1.5 * f, 1.35 + 1.65 * f
t = np.linspace(0, 1, 4001)[None, :]
X = x0[:, None] + (x1 - x0)[:, None] * t; Y = y0[:, None] + (y1 - y0)[:, None] * t; Z = z0[:, None] + (z1 - z0)[:, None] * t
hx, hy = half(Z)
inside = (Z >= ZA) & (Z <= ZB) & (np.abs(X) < hx) & (np.abs(Y) < hy)
enter = inside[:, 1:] & ~inside[:, :-1]
has = enter.any(1)
k = np.where(has, enter.shape[1] - 1 - np.argmax(enter[:, ::-1], 1), 0) + 1      # last entry step
r = np.arange(len(i))
ex, ey, ez = X[r, k], Y[r, k], Z[r, k]
fx, fy = half(ez)
face = np.full(len(i), 'none', object)
face[inside[:, 0]] = 'inside'                                                     # produced in the vessel
dx, dy = np.abs(ex) / fx, np.abs(ey) / fy
m = has & ~inside[:, 0]
face[m & (ez < ZA + 0.05)] = 'front'
lat = m & (ez >= ZA + 0.05)
face[lat & (dx >= dy) & (ex < 0)] = '-x wall'; face[lat & (dx >= dy) & (ex > 0)] = '+x wall'
face[lat & (dx < dy) & (ey > 0)] = 'top'; face[lat & (dx < dy) & (ey < 0)] = 'bottom'
txt = []
for lab, sel in [('p > 1', np.ones(len(i), bool)), ('good track', good)]:
    tot = w[sel].sum()
    txt.append(f'{lab}: total {tot / 1e3:.2f} kHz')
    for f_ in ['-x wall', '+x wall', 'top', 'bottom', 'front', 'inside', 'none']:
        s = sel & (face == f_)
        txt.append(f'   {f_:8s} {w[s].sum() / 1e3:7.2f} kHz ({100 * w[s].sum() / tot:5.1f} %, {s.sum()} MC)')
open(f'{H}/results/secondaries_sbt_walls.txt', 'w').write('\n'.join(txt) + '\n'); print('\n'.join(txt))

plt.rcParams.update({'font.size': 14})
fig, axs = plt.subplots(2, 2, figsize=(16, 9), sharex=True, sharey=True, constrained_layout=True)
bz = np.arange(32, 84, 1.0); by = np.arange(-3.2, 3.21, 0.2)
hs = []
for c, (lab, sel) in enumerate([('$p>1$ GeV/c', np.ones(len(i), bool)), ('$p>1$ GeV/c + TD + $\\geq$3 of 4 stations', good)]):
    for rr, wall in enumerate(['-x wall', '+x wall']):
        s = sel & (face == wall)
        h, _, _ = np.histogram2d(ez[s], ey[s], [bz, by], weights=w[s])
        hs.append((axs[rr, c], h, f'{"−" if wall[0] == "-" else "+"}x wall, {lab}: {w[s].sum() / 1e3:.1f} kHz'))
vmax = max(h.max() for _, h, _ in hs); vmin = min(h[h > 0].min() for _, h, _ in hs if (h > 0).any())
for ax, h, title in hs:
    pc = ax.pcolormesh(bz, by, np.where(h > 0, h, np.nan).T, cmap='viridis', norm=LogNorm(vmin, vmax))
    zz = np.linspace(ZA, ZB, 2); ax.plot(zz, half(zz)[1], color='#888', lw=1.2); ax.plot(zz, -half(zz)[1], color='#888', lw=1.2)
    ax.set_title(title, fontsize=14, loc='left'); ax.grid(color='#e6e6e6', lw=0.6)
for ax in axs[1]: ax.set_xlabel('z [m]')
for ax in axs[:, 0]: ax.set_ylabel('y [m]')
cb = fig.colorbar(pc, ax=axs, shrink=0.9); cb.set_label('Rate [Hz] per 1 m × 0.2 m bin')
fig.savefig(f'{OUT}/secondaries_sbt_walls.png', dpi=100); plt.close(fig)

# ------------------------------------------------------------------ PDG histograms
NAME = {11: 'e$^-$', -11: 'e$^+$', 13: '$\\mu^-$', -13: '$\\mu^+$', 211: '$\\pi^+$', -211: '$\\pi^-$', 2212: 'p',
        -2212: '$\\bar{p}$', 321: 'K$^+$', -321: 'K$^-$', 22: '$\\gamma$', 2112: 'n', -2112: '$\\bar{n}$', 130: 'K$^0_L$',
        310: 'K$^0_S$', 113: '$\rho^0$', 111: '$\\pi^0$', 221: '$\\eta$', 3122: '$\\Lambda$', -3122: '$\\bar\\Lambda$', 0: 'none'}
def bars(ax, codes, title, nmax=14):
    allw = {}
    for v in np.unique(codes):
        allw[v] = w[codes == v].sum()
    order = sorted(allw, key=lambda v: -allw[v])[:nmax]
    xs = np.arange(len(order)); bw = 0.4
    for j, (lab, sel, col) in enumerate([('$p>1$ GeV/c', np.ones(len(i), bool), '#3a7fd9'),
                                         ('$p>1$ GeV/c + TD + $\\geq$3 of 4 stations', good, '#e8643a')]):
        vals = np.array([w[sel & (codes == v)].sum() / 1e3 for v in order])
        err = np.array([np.sqrt((w[sel & (codes == v)] ** 2).sum()) / 1e3 for v in order])
        ax.bar(xs + (j - 0.5) * bw, vals, bw, yerr=err, color=col, label=f'{lab}: {w[sel].sum() / 1e3:.1f} kHz',
               error_kw=dict(lw=0.8, capsize=2, ecolor='#555'))
    ax.set_xticks(xs); ax.set_xticklabels([f'{NAME.get(v, str(v))}\n{v}' for v in order], fontsize=12)
    ax.set_yscale('log'); ax.set_ylabel('Rate at DSST1 [kHz]'); ax.set_title(title, loc='left')
    ax.grid(axis='y', color='#e6e6e6'); ax.legend(frameon=False, fontsize=12)
fig, axs = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
bars(axs[0], pdg, '(a) Particle at DSST1 (PDG code)')
bars(axs[1], mpdg, '(b) Mother particle (PDG code)')
fig.savefig(f'{OUT}/secondaries_pdg.png', dpi=100); plt.close(fig)
