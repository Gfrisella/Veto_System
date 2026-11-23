# %%%
# Test A of the MS7 K_L gap (2026-11-20): how many muons leave the MS7 iron in FairShip TRY_2026, and how many
# MS7-made secondaries are kept.
#   1. muons on UBT scoring plane 2 (detID 2, muons only, 10 cm behind the muon shield, KeepTracks): first point per
#      track: x, y, z, px, py, pz, pdg, MCTrack weight, start z, whether the track has a UBT (UpstreamTaggerPoint) hit.
#      The UBT covers |x| < 1.7 m, y -3.34..1.7 m, i.e. the whole MS7 back face (|x| < 1.71, |y| < 1.27): every muon
#      leaving that face makes a real hit, so its event is saved (--MuonBackSSTOnly keeps events with any real hit).
#   2. every MCTrack made in the last 4.5 m of the shield (28.0 < start z < 32.5 m): pdg, start x, y, z, p, process,
#      mother pdg, weight, whether it has a UBT / SBT / SST point.
# Modes (run from Muon_Shield_Analyzer, pixi env FairShip_sst1):
#   python Flux_MS7.py test <root file>     one file -> flux_ms7_test.pkl, printout
#   python Flux_MS7.py chunk i n            every n-th file of $FILE_LIST from i -> part pickle in $PART_DIR
#   python Flux_MS7.py merge                parts -> Output_data_condor/Analysis_<folder>/Data_<folder>_flux_ms7.pkl
import os
import sys
import glob
import pickle
from time import time
import numpy as np
import ROOT

ROOT.gErrorIgnoreLevel = ROOT.kError
base_dir = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.environ.get('FOLDER_DATA', 'TRY_2026_EM_SST1_origin')
PICKLE = os.path.join(base_dir, 'Output_data_condor', f'Analysis_{FOLDER}', f'Data_{FOLDER}_flux_ms7.pkl')
ZMIN, ZMAX = 28.0, 32.5  # m, start z window for shield secondaries

MU = ['job', 'event', 'trid', 'pdg', 'w', 'x', 'y', 'z', 'px', 'py', 'pz', 'z0', 'ubt']
SEC = ['job', 'event', 'trid', 'pdg', 'w', 'x0', 'y0', 'z0', 'px0', 'py0', 'pz0', 'proc', 'mpdg', 'ubt', 'sbt', 'sst']


def tracks_with(coll):
    return {v.GetTrackID() for v in coll}


def process_root_file(root_file):
    run_id = root_file.split('/')[-2]
    mu = {k: [] for k in MU}
    sec = {k: [] for k in SEC}
    f = ROOT.TFile.Open(root_file)
    tree = f.Get("cbmsim") if f and not f.IsZombie() else None
    if not tree:
        return None
    tree.SetBranchStatus("*", 0)
    for b in ["MCTrack*", "SSTScoringPoint*", "UpstreamTaggerScoringPoint*", "UpstreamTaggerPoint*", "vetoPoint*"]:
        tree.SetBranchStatus(b, 1)
    for ev, event in enumerate(tree):
        mc = event.MCTrack
        n_mc = mc.size()
        ubt = tracks_with(event.UpstreamTaggerPoint)
        sbt = tracks_with(event.vetoPoint)
        sst = tracks_with(event.SSTScoringPoint)
        seen = set()
        for v in event.UpstreamTaggerScoringPoint:
            if v.GetDetectorID() != 2:
                continue
            t = v.GetTrackID()
            if t in seen or not (0 <= t < n_mc):
                continue
            seen.add(t)
            trk = mc[t]
            for k, val in zip(MU, [run_id, ev, t, v.PdgCode(), trk.GetWeight(), v.GetX() / 100, v.GetY() / 100,
                                   v.GetZ() / 100, v.GetPx(), v.GetPy(), v.GetPz(), trk.GetStartZ() / 100, t in ubt]):
                mu[k].append(val)
        for t in range(n_mc):
            trk = mc[t]
            z0 = trk.GetStartZ() / 100
            if not (ZMIN < z0 < ZMAX):
                continue
            m = trk.GetMotherId()
            mpdg = mc[m].GetPdgCode() if 0 <= m < n_mc else 0
            for k, val in zip(SEC, [run_id, ev, t, trk.GetPdgCode(), trk.GetWeight(), trk.GetStartX() / 100,
                                    trk.GetStartY() / 100, z0, trk.GetPx(), trk.GetPy(), trk.GetPz(),
                                    trk.GetProcName().Data(), mpdg, t in ubt, t in sbt, t in sst]):
                sec[k].append(val)
    n = tree.GetEntries()
    f.Close()
    return mu, sec, run_id, n


def read(files, out_pickle):
    t0 = time()
    mu = {k: [] for k in MU}
    sec = {k: [] for k in SEC}
    entries, bad = {}, []
    for i, fn in enumerate(files):
        try:
            r = process_root_file(fn)
        except Exception as e:
            print(f'Error {fn}: {e}', flush=True)
            r = None
        if r is None:
            bad.append(fn)
            continue
        m, s, run_id, n = r
        entries[run_id] = n
        for k in MU:
            mu[k].extend(m[k])
        for k in SEC:
            sec[k].extend(s[k])
        if i % 10 == 0:
            print(f'  {i}/{len(files)} files, {time() - t0:.0f} s', flush=True)
    data = {'mu': {k: np.array(v, dtype=object if k in ('job', 'proc') else None) for k, v in mu.items()},
            'sec': {k: np.array(v, dtype=object if k in ('job', 'proc') else None) for k, v in sec.items()},
            'Entries per files': entries, 'deprecated_files': bad}
    with open(out_pickle, 'wb') as f:
        pickle.dump(data, f)
    print(f'saved {out_pickle}: {len(data["mu"]["pdg"])} plane-2 muons, {len(data["sec"]["pdg"])} shield tracks, '
          f'{len(entries)} files, {time() - t0:.0f} s', flush=True)
    return data


def summary(d):
    mu, sec = d['mu'], d['sec']
    n_sim = sum(d['Entries per files'].values())
    print(f'events stored in the files: {n_sim}')
    if len(mu['pdg']):
        foot = (np.abs(mu['x']) < 1.71) & (np.abs(mu['y']) < 1.27)
        print(f'plane-2 muons: {len(mu["pdg"])} MC, sum w {mu["w"].sum():.4g}; MS7 back face: {foot.sum()} MC, '
              f'sum w {mu["w"][foot].sum():.4g}; with UBT hit {np.mean(mu["ubt"][foot]):.3f}; z {np.unique(np.round(mu["z"], 3))[:5]}')
        p = np.sqrt(mu['px']**2 + mu['py']**2 + mu['pz']**2)
        for lo, hi in ((0, 5), (5, 20), (20, 100), (100, 1e9)):
            k = foot & (p >= lo) & (p < hi)
            print(f'   p {lo}-{hi}: {k.sum()} MC, sum w {mu["w"][k].sum():.4g}')
    if len(sec['pdg']):
        last = sec['z0'] > 31.78
        print(f'shield tracks 28-32.5 m: {len(sec["pdg"])} MC; last 50 cm: {last.sum()}')
        for pid in (13, 211, 2212, 2112, 130, 310, 321, 11, 22):
            k = last & (np.abs(sec['pdg']) == pid)
            print(f'   |pdg| {pid}: {k.sum()} MC, sum w {sec["w"][k].sum():.4g}, UBT {sec["ubt"][k].sum()}')
        procs, cnt = np.unique(sec['proc'][last], return_counts=True)
        print('   processes (last 50 cm):', dict(zip(procs, cnt)))


if __name__ == '__main__':
    opt = sys.argv[1]
    if opt == 'test':
        summary(read(sys.argv[2:], 'flux_ms7_test.pkl'))
    elif opt == 'chunk':
        i, n = int(sys.argv[2]), int(sys.argv[3])
        with open(os.environ['FILE_LIST']) as f:
            files = [l.strip() for l in f if l.strip()][i::n]
        read(files, os.path.join(os.environ.get('PART_DIR', '.'), os.path.basename(PICKLE) + f'.part{i:03d}'))
    elif opt == 'merge':
        parts = sorted(glob.glob(os.path.join(os.environ.get('PART_DIR', os.path.dirname(PICKLE)),
                                              os.path.basename(PICKLE) + '.part*')))
        data = {'mu': {k: [] for k in MU}, 'sec': {k: [] for k in SEC}, 'Entries per files': {}, 'deprecated_files': []}
        for p in parts:
            with open(p, 'rb') as f:
                d = pickle.load(f)
            for g in ('mu', 'sec'):
                for k in d[g]:
                    data[g][k].append(d[g][k])
            data['Entries per files'].update(d['Entries per files'])
            data['deprecated_files'] += d['deprecated_files']
        for g in ('mu', 'sec'):
            data[g] = {k: np.concatenate(v) for k, v in data[g].items()}
        with open(PICKLE, 'wb') as f:
            pickle.dump(data, f)
        print(f'merged {len(parts)} parts -> {PICKLE}')
        summary(data)
