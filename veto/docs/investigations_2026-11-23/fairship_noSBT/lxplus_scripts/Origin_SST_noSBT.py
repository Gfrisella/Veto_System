# %%%
# Cavern-rock check (2026-11-20, Guglielmo: "in FairShip they are way less, they have almost no momentum left"):
# every track crossing the SST scoring plane in TRY_2026_EM_noSBT_full (full physics, balloon-only vessel, no SBT, no
# --MuonBackSSTOnly), first point per track: species, momentum at the plane, whether it hits the timing detector, and
# where it was made: start point, Geant4 volume and material there (geo file of the job), mother species / process.
# Rates: W = MCTrack weight x n_tot / processed muons (per 1 s spill of 5e13 PoT), as Combinatorial_SST.py.
# Modes (run from Muon_Shield_Analyzer, pixi env /afs/cern.ch/work/g/gfrisell/FairShip):
#   python Origin_SST_noSBT.py test <root file>   one file -> printout
#   python Origin_SST_noSBT.py chunk i n          every n-th file of $FILE_LIST from i -> part pickle in $PART_DIR
#   python Origin_SST_noSBT.py merge              parts -> $PART_DIR/../origin_sst_noSBT.pkl + summary printout
import os
import sys
import glob
import pickle
from time import time
import numpy as np
import ROOT

ROOT.gErrorIgnoreLevel = ROOT.kError
N_TOT = 411947589          # muons in the input file (pythia8_Geant4_10.0_withCharmandBeauty0_mu.root), as Combinatorial_SST.py
N_PER_JOB = 100000
FIELDS = ['job', 'event', 'trid', 'pdg', 'w', 'x', 'y', 'z', 'px', 'py', 'pz', 'x0', 'y0', 'z0', 'p0', 'proc',
          'mpdg', 'mproc', 'mz0', 'td', 'vol0', 'mat0']
_geo = {}


def material_at(geo_file, x, y, z):
    if geo_file not in _geo:
        _geo.clear()
        f = ROOT.TFile.Open(geo_file)
        _geo[geo_file] = (f, f.Get('FAIRGeom'))
    gm = _geo[geo_file][1]
    n = gm.FindNode(float(x), float(y), float(z))
    if not n:
        return 'outside', '?'
    v = n.GetVolume()
    return v.GetName(), (v.GetMedium().GetName() if v.GetMedium() else '?')


def process_root_file(root_file):
    run_id = root_file.split('/')[-2]
    geo = sorted(glob.glob(os.path.join(os.path.dirname(root_file), 'geo_*.root')))
    out = {k: [] for k in FIELDS}
    f = ROOT.TFile.Open(root_file)
    tree = f.Get('cbmsim') if f and not f.IsZombie() else None
    if not tree:
        return None
    tree.SetBranchStatus('*', 0)
    for b in ['MCTrack*', 'SSTScoringPoint*', 'TimeDetPoint*']:
        tree.SetBranchStatus(b, 1)
    n_ev = 0
    for ev, event in enumerate(tree):
        n_ev += 1
        sst = event.SSTScoringPoint
        if sst.size() == 0:
            continue
        mc = event.MCTrack
        n_mc = mc.size()
        td = {p.GetTrackID() for p in event.TimeDetPoint}
        seen = set()
        for v in sst:
            trid = v.GetTrackID()
            if trid < 0 or trid >= n_mc or trid in seen:
                continue
            seen.add(trid)
            trk = mc[trid]
            m = trk.GetMotherId()
            mo = mc[m] if 0 <= m < n_mc else None
            x0, y0, z0 = trk.GetStartX(), trk.GetStartY(), trk.GetStartZ()
            vol, mat = material_at(geo[0], x0, y0, z0) if geo else ('?', '?')
            out['job'].append(run_id); out['event'].append(ev); out['trid'].append(trid)
            out['pdg'].append(v.PdgCode()); out['w'].append(trk.GetWeight())
            out['x'].append(v.GetX() / 100); out['y'].append(v.GetY() / 100); out['z'].append(v.GetZ() / 100)
            out['px'].append(v.GetPx()); out['py'].append(v.GetPy()); out['pz'].append(v.GetPz())
            out['x0'].append(x0 / 100); out['y0'].append(y0 / 100); out['z0'].append(z0 / 100); out['p0'].append(trk.GetP())
            out['proc'].append(str(trk.GetProcName()))
            out['mpdg'].append(mo.GetPdgCode() if mo is not None else 0)
            out['mproc'].append(str(mo.GetProcName()) if mo is not None else '')
            out['mz0'].append(mo.GetStartZ() / 100 if mo is not None else np.nan)
            out['td'].append(trid in td); out['vol0'].append(vol); out['mat0'].append(mat)
    f.Close()
    return out, run_id, n_ev


def read(files, out_pickle):
    t0 = time()
    data = {k: [] for k in FIELDS}
    jobs, bad = {}, []
    for i, fn in enumerate(files):
        try:
            r = process_root_file(fn)
        except Exception as e:
            print(f'Error {fn}: {e}', flush=True)
            r = None
        if r is None:
            bad.append(fn)
            continue
        o, run_id, n = r
        jobs[run_id] = n
        for k in FIELDS:
            data[k].extend(o[k])
        if i % 5 == 0:
            print(f'  {i}/{len(files)} files, {len(data["pdg"])} tracks, {time() - t0:.0f} s', flush=True)
    data = {k: np.array(v, dtype=object if k in ('job', 'proc', 'mproc', 'vol0', 'mat0') else None) for k, v in data.items()}
    data['jobs'] = jobs; data['bad'] = bad
    with open(out_pickle, 'wb') as f:
        pickle.dump(data, f)
    print(f'saved {out_pickle}: {len(data["pdg"])} tracks, {len(jobs)} files, {time() - t0:.0f} s', flush=True)
    return data


def summary(d):
    n_jobs = len(d['jobs'])
    factor = N_TOT / (n_jobs * N_PER_JOB)
    W = d['w'].astype(float) * factor
    p = np.sqrt(d['px'].astype(float) ** 2 + d['py'].astype(float) ** 2 + d['pz'].astype(float) ** 2)
    pdg = np.abs(d['pdg'].astype(int))
    q = np.isin(pdg, [11, 13, 211, 321, 2212, 3112, 3222, 3312, 3334])
    mat = d['mat0']
    good = q & (p > 1) & d['td'].astype(bool)
    print(f'{n_jobs} files, factor {factor:.4g}; rates in kHz per 5e13 spill; good = charged, p > 1 GeV, timing-detector hit')
    for lab, sel in (('all charged', q), ('good', good)):
        print(f'== {lab}: {W[sel].sum() / 1e3:.3f} kHz ({sel.sum()} MC)')
        for m_ in sorted(set(mat[sel]), key=lambda m: -W[sel & (mat == m)].sum())[:8]:
            s = sel & (mat == m_)
            ss = s & (pdg != 13)
            print(f'   made in {m_:22s} {W[s].sum() / 1e3:9.3f} kHz ({s.sum():6d} MC); non-mu {W[ss].sum() / 1e3:8.3f} kHz '
                  f'({ss.sum()} MC), median p non-mu {np.median(p[ss]) if ss.any() else 0:.2f} GeV')
    conc = good & np.array(['oncrete' in str(m) or 'rock' in str(m).lower() for m in mat]) & (pdg != 13)
    print(f'good non-muon tracks made in concrete/rock: {W[conc].sum() / 1e3:.3f} kHz ({conc.sum()} MC)')
    for name, ids in (('pi', [211]), ('p', [2212]), ('K', [321]), ('e', [11])):
        s = conc & np.isin(pdg, ids)
        print(f'   {name:3s} {W[s].sum() / 1e3:.3f} kHz ({s.sum()} MC) p median {np.median(p[s]) if s.any() else 0:.2f} GeV')
    edges = [0, 0.1, 0.3, 1, 2, 3, 5, 10, 30, 100, 1000]
    allc = q & np.array(['oncrete' in str(m) or 'rock' in str(m).lower() for m in mat]) & (pdg != 13)
    h, _ = np.histogram(p[allc], edges, weights=W[allc])
    print('momentum of charged non-mu made in concrete (all, no cut), kHz per bin:', dict(zip([f'{a}-{b}' for a, b in zip(edges[:-1], edges[1:])], np.round(h / 1e3, 3))))


if __name__ == '__main__':
    opt = sys.argv[1]
    if opt == 'test':
        summary(read(sys.argv[2:], 'origin_sst_noSBT_test.pkl'))
    elif opt == 'chunk':
        i, n = int(sys.argv[2]), int(sys.argv[3])
        with open(os.environ['FILE_LIST']) as f:
            files = [l.strip() for l in f if l.strip()][i::n]
        read(files, os.path.join(os.environ['PART_DIR'], f'origin_sst_noSBT.part{i:03d}'))
    elif opt == 'merge':
        P = os.environ['PART_DIR']
        parts = sorted(glob.glob(os.path.join(P, 'origin_sst_noSBT.part*')))
        data = {k: [] for k in FIELDS}; jobs, bad = {}, []
        for pp in parts:
            with open(pp, 'rb') as f:
                d = pickle.load(f)
            for k in FIELDS:
                data[k].append(d[k])
            jobs.update(d['jobs']); bad += d['bad']
        data = {k: np.concatenate(v) for k, v in data.items()}
        data['jobs'] = jobs; data['bad'] = bad
        with open(os.path.join(os.path.dirname(P.rstrip('/')), 'origin_sst_noSBT.pkl'), 'wb') as f:
            pickle.dump(data, f)
        print(f'merged {len(parts)} parts')
        summary(data)
