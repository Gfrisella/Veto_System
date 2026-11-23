#!/usr/bin/env python3
"""Where do FairShip concrete tracks (p>1) that pass OUR TD acceptance but have no FairShip TD hit land (our field-map
propagation to Z_TIMING)? Edge-concentrated -> geometry/field mismatch; uniform -> material / decay. Also weight
concentration. Writes plots/td_miss_xy.png and appends to results/td_acceptance_test.txt."""
import os, sys, pickle
import numpy as np
W = "/disk/gfs_lhcb/gfrise/Study_on_Veto_System/Optimization_Veto_System/.claude/worktrees/nov4-source/veto"
sys.path[:0] = [W + "/bin", W + "/lib"]
import _bootstrap  # noqa
from veto import geometry as geo, spectrometer  # noqa
from veto.interactions import charge  # noqa
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
F4 = 0.8 * 411947589 / (3086 * 1e5)
O = pickle.load(open(f'{H}/data/origin_sst_noSBT_merged.pkl', 'rb'))
pdg = O['pdg'].astype(int); q = charge(pdg).astype(float)
P = np.stack([O['px'], O['py'], O['pz']], 1).astype(float); p = np.linalg.norm(P, axis=1)
out = []
for lab, sel in (('concrete', (O['mat0'] == 'Concrete') & (q != 0) & (p > 1) & (P[:, 2] > 0)),
                 ('all origins', (q != 0) & (p > 1) & (P[:, 2] > 0))):
    pos = np.stack([O['x'][sel], O['y'][sel], O['z'][sel] - 2.14], 1).astype(float)
    mom, qq, ww, td = P[sel], q[sel], O['w'][sel].astype(float) * F4, O['td'][sel].astype(bool)
    b = pos + mom / mom[:, 2:3] * (geo.Z_SST1 - pos[:, 2:3])
    acc = (np.abs(b[:, 0]) < geo.SST1_HALF_X) & (np.abs(b[:, 1]) < geo.SST1_HALF_Y)
    m, xt, yt = spectrometer.timing_acceptance(b, mom, qq)
    m &= acc
    hit, miss = m & td, m & ~td
    for nm, k in (('hit', hit), ('miss', miss)):
        out.append(f'{lab} {nm}: {k.sum()} MC, {ww[k].sum():.3g}/spill; x_t pct 5/50/95 {np.percentile(xt[k], [5, 50, 95]).round(2)}, '
                   f'|y_t| 50/95 {np.percentile(np.abs(yt[k]), [50, 95]).round(2)}, |x_t|>2.2: {ww[k & (np.abs(xt) > 2.2)].sum() / ww[k].sum():.2f}; '
                   f'x at SST1 pct 5/50/95 {np.percentile(b[k, 0], [5, 50, 95]).round(2)}; '
                   f'top-5 weights share {np.sort(ww[k])[-5:].sum() / ww[k].sum():.2f}')
    # unweighted MC fraction (weight-concentration check)
    out.append(f'{lab}: unweighted FS-hit fraction among our-accepted {hit.sum() / m.sum():.3f} (weighted {ww[hit].sum() / ww[m].sum():.3f})')
    if lab == 'concrete':
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
        for k, c, nm in ((hit, '#1f77b4', 'FS TD hit'), (miss, '#d62728', 'no FS TD hit')):
            axs[0].scatter(xt[k], yt[k], s=6, c=c, alpha=0.6, label=f'{nm} ({k.sum()} MC)')
            axs[1].hist(xt[k], np.linspace(-3, 3, 31), weights=ww[k], histtype='step', color=c, label=nm)
        axs[0].add_patch(plt.Rectangle((-geo.TIMING_HALF_X, -geo.TIMING_HALF_Y), 2 * geo.TIMING_HALF_X, 2 * geo.TIMING_HALF_Y, fill=False, ls='--'))
        axs[0].set_xlabel('x at timing detector, our field-map propagation [m]'); axs[0].set_ylabel('y [m]'); axs[0].legend(fontsize=8)
        axs[1].set_xlabel('x at timing detector [m]'); axs[1].set_ylabel('tracks / bin / spill (4e13)'); axs[1].legend(fontsize=8)
        fig.suptitle('FairShip concrete tracks (p>1) inside OUR TD acceptance: with / without a real FairShip TD hit', fontsize=10)
        fig.tight_layout(); fig.savefig(f'{H}/plots/td_miss_xy.png', dpi=150); plt.close(fig)
print('\n'.join(out))
open(f'{H}/results/td_acceptance_test.txt', 'a').write('\n-- miss diagnostic --\n' + '\n'.join(out) + '\n')
