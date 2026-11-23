#!/usr/bin/env python3
"""Test A: production in the last iron of the muon shield (MS7) in FairShip TRY_2026_EM_noSBT_full vs our veto chain.

Inputs (merged by merge_parts.py): data/flux_ms7_merged.pkl (Flux_MS7.py, 2671 files), data/origin_sst_noSBT_merged.pkl
(Origin_SST_noSBT.py, 3086 files). Rates per 1 s spill of 5e13 PoT: sum(MCTrack weight) x 411947589 / (n_files x 1e5);
MC entries scaled to the full input (411.9 M muons) with x 4119.5 / n_files. Stat. error = sqrt(sum w^2) x factor.
FairShip z throughout (z_ours = z_FS - 2.14 m). The MS7 back face in this geometry is at z_FS = 32.18 m (end of the
interaction vertices; UBT scoring plane 2 at 32.28 m), i.e. 30.04 m ours (our Geant4 MS7 block ends at 29.89 m ours).
Outputs: results/ms7_summary.json, results/ms7_tables.txt, plots/ms7_*.png
"""
import json, os, pickle, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
N_TOT, N_JOB = 411947589, 100000
NF_FLUX, NF_ORIG = 2671, 3086
FAC = N_TOT / (NF_FLUX * N_JOB)          # per 5e13 spill
FAC_O = N_TOT / (NF_ORIG * N_JOB)
FULL = N_TOT / N_JOB                       # 4119.5 files in the full input
Z_END = 32.18
LAST = (Z_END - 0.5, Z_END)
SPECIES = [('K_L', [130]), ('K_S', [310]), ('K±', [321]), ('π±', [211]), ('p/p̄', [2212]), ('n', [2112])]
# ours (veto chain, Geant4 10.7 MS7 transport), per 5e13 spill; and expected FairShip MC entries for the full input
OURS = dict(mu_foot=2.5e8, KL_exit=3.4e3, pi_sst1=1.2e5, p_sst1=7.4e4)
OURS_MC_FULL = dict(interactions=5.3e4, KL_exit=300, pi_sst1=480)

os.makedirs(f'{H}/results', exist_ok=True); os.makedirs(f'{H}/plots', exist_ok=True)
F = pickle.load(open(f'{H}/data/flux_ms7_merged.pkl', 'rb'))
mu, s = F['mu'], F['sec']
out, lines = {}, []
def say(t=''):
    print(t); lines.append(t)

def rate(w, fac=FAC):
    w = np.asarray(w, float)
    return w.sum() * fac, np.sqrt((w ** 2).sum()) * fac

# ---------------- 1. muon flux out of MS7 (UBT scoring plane 2, z 32.28)
mx, my, mw = mu['x'].astype(float), mu['y'].astype(float), mu['w'].astype(float)
mp = np.sqrt(mu['px'].astype(float) ** 2 + mu['py'].astype(float) ** 2 + mu['pz'].astype(float) ** 2)
foot = (np.abs(mx) < 1.71) & (np.abs(my) < 1.27)
R_all, R_foot = rate(mw), rate(mw[foot])
say(f'normalisation: flux run {NF_FLUX} files -> x{FAC:.4f}; origin run {NF_ORIG} files -> x{FAC_O:.4f} (per 5e13 spill)')
say(f'1. muons on UBT plane 2 (z_FS 32.28): all {R_all[0]:.4g}/spill ({len(mw)} MC); MS7 footprint |x|<1.71,|y|<1.27: '
    f'{R_foot[0]:.4g} ± {R_foot[1]:.2g}/spill ({foot.sum()} MC), UBT hit {mu["ubt"][foot].astype(bool).mean():.3f}; '
    f'ours {OURS["mu_foot"]:.2g} -> FS/ours {R_foot[0] / OURS["mu_foot"]:.2f}')
out['mu_flux'] = dict(all=R_all[0], foot=R_foot[0], foot_err=R_foot[1], foot_mc=int(foot.sum()), ours=OURS['mu_foot'],
                      median_p_foot=float(np.median(mp[foot])))
MU6 = R_foot[0] / 1e6          # per 1e6 muons through MS7

# ---------------- 2. muon-nuclear interactions (vertices = distinct start points of 'Muon nuclear interaction' tracks)
z0 = s['z0'].astype(float); x0 = s['x0'].astype(float); y0 = s['y0'].astype(float)
pdg = np.abs(s['pdg'].astype(int)); w = s['w'].astype(float); proc = s['proc']
p0 = np.sqrt(s['px0'].astype(float) ** 2 + s['py0'].astype(float) ** 2 + s['pz0'].astype(float) ** 2)
ubt, sbt, sst = (s[k].astype(bool) for k in ('ubt', 'sbt', 'sst'))
evk = s['part'].astype(np.int64) * 10 ** 7 + s['fseg'].astype(np.int64) * 10 ** 5 + s['event'].astype(np.int64)
mn = proc == 'Muon nuclear interaction'
vkey = np.stack([evk[mn], np.round(x0[mn] * 1e5), np.round(y0[mn] * 1e5), np.round(z0[mn] * 1e5)], 1)
_, first = np.unique(vkey, axis=0, return_index=True)
vz, vw = z0[mn][first], w[mn][first]
out['interactions'] = {}
for lab, lo, hi in (('last 50 cm', *LAST), ('28.0 m - back face (4.2 m)', 28.0, Z_END)):
    k = (vz > lo) & (vz <= hi)
    r = rate(vw[k])
    say(f'2. muon-nuclear vertices {lab}: {k.sum()} MC (full input {k.sum() * FULL / NF_FLUX:.3g} MC; ours expects '
        f'{OURS_MC_FULL["interactions"]:.2g} for MS7), {r[0]:.4g} ± {r[1]:.2g}/spill, {r[0] / MU6:.4g} per 1e6 mu')
    out['interactions'][lab] = dict(mc=int(k.sum()), mc_full=k.sum() * FULL / NF_FLUX, rate=r[0], err=r[1], per1e6mu=r[0] / MU6)
say(f'   products per vertex (stored, last 50 cm): {(mn & (z0 > LAST[0])).sum() / max(1, ((vz > LAST[0])).sum()):.2f}')

# ---------------- 3. hadron yields made in the last 50 cm (any process, and direct muon-nuclear products)
last = (z0 > LAST[0]) & (z0 <= Z_END)
whole = (z0 > 28.0) & (z0 <= Z_END)
say('3. hadrons made in MS7 (FairShip z window), rate per 5e13 spill [MC], per 1e6 mu through MS7, median p, '
    'fractions with UBT / SBT / SST(scoring plane) point')
say(f'   {"species":6s} {"window":10s} {"origin":8s} {"MC":>6s} {"MC full":>8s} {"rate/spill":>10s} {"±":>8s} {"/1e6 mu":>8s} '
    f'{"p med":>6s} {"UBT":>5s} {"SBT":>5s} {"SST":>5s} {"SST rate":>9s}')
out['yields'] = {}
for name, ids in SPECIES:
    for wl, win in (('last50', last), ('28-end', whole)):
        for ol, org in (('all', np.ones_like(mn)), ('muN', mn)):
            k = win & org & np.isin(pdg, ids)
            r = rate(w[k]); rs = rate(w[k & sst])
            d = dict(mc=int(k.sum()), mc_full=k.sum() * FULL / NF_FLUX, rate=r[0], err=r[1], per1e6mu=r[0] / MU6,
                     p_med=float(np.median(p0[k])) if k.any() else 0., f_ubt=float(ubt[k].mean()) if k.any() else 0.,
                     f_sbt=float(sbt[k].mean()) if k.any() else 0., f_sst=float(sst[k].mean()) if k.any() else 0.,
                     sst_rate=rs[0], sst_err=rs[1], sst_mc=int((k & sst).sum()))
            out['yields'][f'{name}|{wl}|{ol}'] = d
            say(f'   {name:6s} {wl:10s} {ol:8s} {d["mc"]:6d} {d["mc_full"]:8.0f} {d["rate"]:10.4g} {d["err"]:8.2g} '
                f'{d["per1e6mu"]:8.3g} {d["p_med"]:6.2f} {d["f_ubt"]:5.2f} {d["f_sbt"]:5.2f} {d["f_sst"]:5.2f} {d["sst_rate"]:9.3g}')
say(f'   processes making K_L in the last 50 cm: {dict(collections.Counter(proc[last & (pdg == 130)].tolist()))}')
say(f'   mothers of K_L in the last 50 cm: {dict(collections.Counter(np.abs(s["mpdg"][last & (pdg == 130)].astype(int)).tolist()).most_common(6))}')

# ---------------- 4. cross-check at the SST scoring plane (z_FS 83.71) from the origin run: tracks made in Magn7
O = pickle.load(open(f'{H}/data/origin_sst_noSBT_merged.pkl', 'rb'))
opdg = np.abs(O['pdg'].astype(int)); ow = O['w'].astype(float)
op = np.sqrt(O['px'].astype(float) ** 2 + O['py'].astype(float) ** 2 + O['pz'].astype(float) ** 2)
m7 = O['vol0'] == 'Magn7'
oz0 = O['z0'].astype(float)
say(f'4. SST scoring plane (z_FS 83.71, |x|<2, |y|<3), tracks made in volume Magn7 (z0 {np.percentile(oz0[m7], 0.1):.2f}-'
    f'{np.percentile(oz0[m7], 99.9):.2f}), origin run: rate per 5e13 spill [MC, MC full input]; ours (straight lines, all p)')
out['sst_from_magn7'] = {}
for name, ids in SPECIES:
    for pl, pc in (('all p', op >= 0), ('p>1', op > 1)):
        k = m7 & np.isin(opdg, ids) & pc
        r = rate(ow[k], FAC_O)
        ref = {'π±': OURS['pi_sst1'], 'p/p̄': OURS['p_sst1']}.get(name) if pl == 'all p' else None
        out['sst_from_magn7'][f'{name}|{pl}'] = dict(mc=int(k.sum()), mc_full=k.sum() * FULL / NF_ORIG, rate=r[0], err=r[1],
                                                     ours=ref, p_med=float(np.median(op[k])) if k.any() else 0.)
        say(f'   {name:6s} {pl:6s} {r[0]:10.4g} ± {r[1]:8.2g} [{k.sum()} MC, {k.sum() * FULL / NF_ORIG:.0f} full]'
            + (f'   ours {ref:.2g} -> ours/FS {ref / r[0]:.2f}' if ref and r[0] > 0 else ''))

json.dump(out, open(f'{H}/results/ms7_summary.json', 'w'), indent=1, default=float)
open(f'{H}/results/ms7_tables.txt', 'w').write('\n'.join(lines) + '\n')

# ---------------- plots
C = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
fig, ax = plt.subplots(figsize=(7, 4.5))
e = np.arange(28.0, 32.31, 0.02)
h, _ = np.histogram(vz, e, weights=vw * FAC)
ax.step(e[:-1], h / 0.02, where='post', color=C[0], label='μ-nuclear vertices (FairShip)')
ax.axvline(Z_END, color='k', ls='--', lw=1); ax.text(Z_END - 0.02, ax.get_ylim()[1] * 0.5, 'MS7 back face 32.18', rotation=90, ha='right', fontsize=8)
ax.axvspan(*LAST, color='grey', alpha=0.15, label='last 50 cm')
ax.set_yscale('log'); ax.set_xlabel('vertex z (FairShip) [m]   (ours = −2.14 m)'); ax.set_ylabel('vertices / m / spill (5e13 PoT)')
ax.set_title('Muon-nuclear interactions in the end of the muon shield'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{H}/plots/ms7_munucl_vertex_z.png', dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
lb = np.logspace(-2, 2, 41)
for c, (name, ids) in zip(C, SPECIES):
    k = last & np.isin(pdg, ids)
    hh, _ = np.histogram(p0[k], lb, weights=w[k] * FAC)
    ax.step(lb[:-1], hh, where='post', color=c, label=f'{name} ({k.sum()} MC)')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('momentum at production [GeV]'); ax.set_ylabel('tracks / bin / spill (5e13 PoT)')
ax.set_title('Hadrons made in the last 50 cm of MS7 (FairShip, any process)'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{H}/plots/ms7_hadron_p_spectra.png', dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
for c, (name, ids) in zip(C, SPECIES):
    k = m7 & np.isin(opdg, ids)
    if k.sum() == 0: continue
    hh, _ = np.histogram(op[k], lb, weights=ow[k] * FAC_O)
    ax.step(lb[:-1], hh, where='post', color=c, label=f'{name} ({k.sum()} MC)')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('momentum at the SST plane [GeV]'); ax.set_ylabel('tracks / bin / spill (5e13 PoT)')
ax.set_title('Tracks made in Magn7 reaching the SST scoring plane (FairShip)'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{H}/plots/ms7_to_sst_p_spectra.png', dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
lb2 = np.logspace(-1, 3, 41)
hh, _ = np.histogram(mp[foot], lb2, weights=mw[foot] * FAC)
ax.step(lb2[:-1], hh, where='post', color=C[0], label=f'FairShip, MS7 footprint ({foot.sum()} MC)')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('muon momentum [GeV]'); ax.set_ylabel('muons / bin / spill (5e13 PoT)')
ax.set_title(f'Muons leaving MS7 (UBT plane, z_FS 32.28): {R_foot[0]:.3g}/spill (ours 2.5e8)'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{H}/plots/ms7_muon_flux_p.png', dpi=150); plt.close(fig)
print('done')
