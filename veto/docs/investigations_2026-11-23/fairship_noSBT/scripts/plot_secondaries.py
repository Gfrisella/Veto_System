#!/usr/bin/env python3
"""Plots of the secondaries at DSST1 for the MS note (Subsection sub:secondaries), FairShip TRY_2026_EM_noSBT_full.
Secondary = charged track (e, mu, pi, p, K) crossing the DSST1 scoring plane (|x|<2, |y|<3, z_FS 83.71), pz > 0, not
the muon of the input sample. Rates per spill (x 411947589 / (3086e5)). z in the FairShip frame.
 1. secondaries_origin_map.png : production point, top (x-z) and side (y-z) view, by origin (as hit_categories), p>1;
    secondaries_origin_z.png   : rate at DSST1 vs production z, by origin, p>1.
 2. secondaries_xy_DSST1.png    : non-muon secondaries at DSST1, p>1, colour = momentum (as Scatter_Momentum_Map_T1).
 3. secondaries_p_pt_DSST1.png  : non-muon secondaries at DSST1, p-pT rate (as P_vs_Pt), all momenta."""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT = f'{H}/plots/note'; os.makedirs(OUT, exist_ok=True)
F = 411947589 / (3086 * 1e5)
O = {k: np.asarray(v) for k, v in pickle.load(open(f'{H}/data/origin_sst_noSBT_merged.pkl', 'rb')).items()
     if k not in ('n_files', 'n_files_from_restarts', 'cluster')}
w = O['w'].astype(float) * F; pdg = O['pdg'].astype(int); a = np.abs(pdg)
px, py, pz = (O[k].astype(float) for k in ('px', 'py', 'pz'))
p = np.sqrt(px ** 2 + py ** 2 + pz ** 2); pt = np.hypot(px, py)
sec = (pz > 0) & (O['proc'] != 'Transportation') & np.isin(a, [11, 13, 211, 2212, 321])
mat, vol = O['mat0'], O['vol0']
x0, y0, z0 = (O[k].astype(float) for k in ('x0', 'y0', 'z0'))
x, y = O['x'].astype(float), O['y'].astype(float)

# ------------------------------------------------------------------ 1. origin map (hit_categories style)
plt.rcParams.update({'font.size': 13, 'axes.edgecolor': '#b0b0b0', 'axes.labelcolor': '#333333',
                     'xtick.color': '#555555', 'ytick.color': '#555555', 'axes.titlesize': 17,
                     'axes.titlecolor': '#333333', 'axes.titlelocation': 'left'})
CAT = [('MS7', (mat == 'iron') & (vol == 'Magn7'), '#3a7fd9', 'o'),
       ('MS4–MS6', (mat == 'iron') & (vol != 'Magn7'), '#7b5cc4', 'v'),
       ('Cavern walls', mat == 'Concrete', '#e8643a', 's'),
       ('Cavern air', mat == 'air', '#1aaa7a', '^'),
       ('Decay vessel', np.isin(mat, ['helium', 'mylar_linerHe']), '#f0a417', 'D')]
sel = sec & (p > 1)
# magnets: FairShip TRY_2026 rows [Zgap, dZ] (cm), MS7 back face at z_FS 32.18
rows = [(0, 115), (15, 150), (30, 225), (16, 225), (15, 225), (16, 131), (15, 168), (19, 200)]
names = ['MHS', 'MS1', 'MS2', 'MS3', 'MS4', 'MS5', 'MS6', 'MS7']
zs = 32.18 - sum(g + 2 * d for g, d in rows) / 100; mags = []
for (g, d), n in zip(rows, names):
    zs += g / 100; mags.append((n, zs, zs + 2 * d / 100)); zs += 2 * d / 100
fig = plt.figure(figsize=(13, 11.5))
gs = fig.add_gridspec(2, 1, hspace=0.28)
axt, axs = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
def geometry(ax, view):
    for n, za, zb in mags:
        ax.axvspan(za, zb, color='#e9e6df' if n != 'MS5' else '#f3f1ec', zorder=0)
    zv = np.array([32.85, 82.85]); hv = np.array([0.5, 2.0]) if view == 'x' else np.array([1.35, 3.0])
    ax.plot(zv, hv, color='#888', lw=1.2); ax.plot(zv, -hv, color='#888', lw=1.2)
    ax.plot([zv[0]] * 2, [-hv[0], hv[0]], color='#888', lw=1.2); ax.plot([zv[1]] * 2, [-hv[1], hv[1]], color='#888', lw=1.2)
    ha = 2.0 if view == 'x' else 3.0
    ax.plot([83.71, 83.71], [-ha, ha], color='#333', lw=2.5)
    if view == 'x':
        ax.fill_between([-1, 20.53], -3.57, -6, color='#d9d4c7', zorder=0); ax.fill_between([20.53, 90], -4.566, -6, color='#d9d4c7', zorder=0)
        ax.fill_between([-1, 20.53], 6.43, 12.5, color='#d9d4c7', zorder=0); ax.fill_between([20.53, 90], 11.434, 12.5, color='#d9d4c7', zorder=0)
        ax.text(60, -5.5, 'Salève-side wall', color='#666', fontsize=11, ha='center')
    else:
        ax.fill_between([-1, 20.53], -1.70, -4.5, color='#d9d4c7', zorder=0); ax.fill_between([20.53, 90], -3.364, -4.5, color='#d9d4c7', zorder=0)
        ax.fill_between([-1, 20.53], 5.80, 11, color='#d9d4c7', zorder=0); ax.fill_between([20.53, 90], 10.64, 11, color='#d9d4c7', zorder=0)
        ax.text(60, -4.1, 'floor', color='#666', fontsize=11, ha='center')
    ax.text(83.0, ha + 0.3, 'DSST1', ha='right', fontsize=10, color='#333')
    ax.text(57.85, (2.6 if view == 'x' else 3.6), 'decay vessel', ha='center', fontsize=10, color='#666')
    ax.grid(color='#e6e6e6', lw=0.8); ax.set_xlim(0, 86); ax.set_xlabel('z [m]')
for n, za, zb in mags:
    axt.text(0.5 * (za + zb), 7.3, n, rotation=90, fontsize=9, color='#666', ha='center', va='bottom')
geometry(axt, 'x'); geometry(axs, 'y')
axt.set_ylim(-6, 9.5); axt.set_ylabel('x [m]'); axs.set_ylim(-4.5, 6); axs.set_ylabel('y [m]')
axt.set_title('(a) Production point of the secondaries at DSST1 ($p>1$ GeV/c), top view')
axs.set_title('(b) Side view')
rng = np.random.default_rng(1); handles = []
for lab, m, col, mk in CAT:
    i = np.nonzero(sel & m)[0]
    tot, err = w[i].sum() / 1e3, np.sqrt((w[i] ** 2).sum()) / 1e3
    s = 6 + 18 * np.log10(p[i])
    for ax, c in [(axt, x0), (axs, y0)]:
        ax.scatter(z0[i], c[i], s=s, c=col, marker=mk, alpha=0.75, edgecolors='white', linewidths=0.3, zorder=3)
    handles.append(Line2D([], [], ls='', marker=mk, color=col, ms=9, label=f'{lab}: {len(i)} tracks, {tot:.1f} ± {err:.1f} kHz'))
axt.legend(handles=handles, loc='upper right', frameon=True, framealpha=0.9, edgecolor='none', fontsize=11.5)
szh = [Line2D([], [], ls='', marker='o', color='#b0ada5', ms=np.sqrt(6 + 18 * np.log10(v)), label=f'{v:g} GeV/c') for v in (1.5, 10, 100)]
axs.legend(handles=szh, title='P at DSST1', loc='upper left', frameon=False, fontsize=10.5, title_fontsize=10.5)
fig.savefig(f'{OUT}/secondaries_origin_map.png', dpi=110, bbox_inches='tight'); plt.close(fig)
# rate vs production z: separate figure
fig, axz = plt.subplots(figsize=(13, 5.5))
bins = np.arange(0, 86, 1.0)
G = np.load(f'{H}/results/secondaries_good.npz')   # from scripts/secondaries_good.py
good = np.zeros(len(w), bool); good[G['idx']] = G['td'] & (G['nst'] >= 3)
for lab, m, col in [('$p>1$ GeV/c', sel, '#3a7fd9'), ('$p>1$ GeV/c + TD + $\\geq$3 of 4 stations', sel & good, '#e8643a')]:
    h, _ = np.histogram(z0[m], bins, weights=w[m] / 1e3)
    axz.stairs(h, bins, color=col, lw=2,
               label=f'{lab}: {w[m].sum() / 1e3:.1f} ± {np.sqrt((w[m] ** 2).sum()) / 1e3:.1f} kHz')
axz.set_yscale('log'); axz.set_ylim(1e-3, 50); axz.set_xlim(0, 86)
for n, za, zb in mags:
    axz.axvspan(za, zb, color='#e9e6df' if n != 'MS5' else '#f3f1ec', zorder=0)
    axz.text(0.5 * (za + zb), 20, n, rotation=90, fontsize=9, color='#666', ha='center', va='center')
axz.axvspan(32.85, 82.85, color='#f4f4f4', zorder=0); axz.text(57.85, 2e-3, 'decay vessel', ha='center', fontsize=10, color='#666')
axz.axvline(83.71, color='#333', lw=2)
axz.set_xlabel('production z [m]'); axz.set_ylabel('rate at DSST1 [kHz / m]')
axz.set_title('Rate of the secondaries at DSST1 vs production z')
axz.grid(color='#e6e6e6', lw=0.8); axz.legend(frameon=True, framealpha=0.9, edgecolor='none', fontsize=11, loc='upper left', bbox_to_anchor=(0.43, 1.0))
fig.savefig(f'{OUT}/secondaries_origin_z.png', dpi=110, bbox_inches='tight'); plt.close(fig)

# ------------------------------------------------------------------ 2. x-y at DSST1 (Scatter_Momentum_Map style)
plt.rcdefaults(); plt.rcParams.update({'font.size': 20})
nm = sec & (a != 13)
i = np.nonzero(nm & (p > 1))[0]; i = i[np.argsort(p[i])]
fig, ax = plt.subplots(figsize=(8.5, 8.5))
sc = ax.scatter(x[i], y[i], c=p[i], cmap='viridis', norm=LogNorm(1, max(p[i].max(), 10)), s=12)
cb = fig.colorbar(sc, ax=ax); cb.set_label('Particle Momentum [GeV/c]')
ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_title('Secondary particles transverse distribution')
ax.set_xlim(-2.2, 2.2); ax.set_ylim(-3.25, 3.25); ax.grid(True)
fig.savefig(f'{OUT}/secondaries_xy_DSST1.png', dpi=100, bbox_inches='tight'); plt.close(fig)

# ------------------------------------------------------------------ 3. p-pT at DSST1 (P_vs_Pt style)
i = nm & (pt > 0)
bp = np.logspace(-3, np.log10(50), 61); bt = np.logspace(-4, np.log10(10), 61)
h, _, _ = np.histogram2d(p[i], pt[i], [bp, bt], weights=w[i])
fig, ax = plt.subplots(figsize=(8.5, 8.5))
pc = ax.pcolormesh(bp, bt, np.where(h > 0, h, np.nan).T, cmap='viridis', norm=LogNorm())
cb = fig.colorbar(pc, ax=ax); cb.set_label('Rate [Hz]')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Momentum [GeV/c]'); ax.set_ylabel('Transverse Momentum [GeV/c]')
fig.savefig(f'{OUT}/secondaries_p_pt_DSST1.png', dpi=100, bbox_inches='tight'); plt.close(fig)
print('bin widths: 60 log bins per axis, p 1e-3..50 GeV/c, pT 1e-4..10 GeV/c')
print('non-mu p>1 tracks in xy plot:', (nm & (p > 1)).sum(), ' rate', w[nm & (p > 1)].sum() / 1e3, 'kHz')
