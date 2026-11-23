#!/usr/bin/env python3
"""Merge the condor part pickles of Flux_MS7.py (cluster 21080019) and Origin_SST_noSBT.py (cluster 21080523).

Bug found in the condor runs: run_id = path.split('/')[-2] was '' for every file, so 'Entries per files' / 'jobs' hold one
key per part and the 'job' column is empty -> the scripts' own n_jobs (and their summary normalisation) is wrong.
Fix here: each row gets its part index, and a file index inside the part (the event index restarts at every new file;
rows are written file by file). The number of files read is counted from those restarts and checked against the
condor split (FILE_LIST[i::100]: 2671 files -> 27 x71 + 26 x29; 3086 files -> 31 x86 + 30 x14).
Event key for grouping = (part, file-in-part, event). Empty parts are cast to the dtype of the non-empty ones.
Output: data/flux_ms7_merged.pkl, data/origin_sst_noSBT_merged.pkl"""
import glob, os, pickle, collections
import numpy as np
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

def cat(arrs):
    ne = [a for a in arrs if len(a)]
    dt = ne[0].dtype
    return np.concatenate([a.astype(dt) if len(a) else np.array([], dtype=dt) for a in arrs])

def seg(ev):
    ev = np.asarray(ev, int)
    return np.concatenate([[0], np.cumsum(np.diff(ev) < 0)]) if len(ev) else np.zeros(0, int)

def expected(n_files, n=100):
    return np.array([len(range(i, n_files, n)) for i in range(n)])

# ---- flux
parts = sorted(glob.glob(f'{D}/flux_parts/*.part*'))
mu, sec, nf_mu, nf_sec = collections.defaultdict(list), collections.defaultdict(list), [], []
for ip, p in enumerate(parts):
    d = pickle.load(open(p, 'rb'))
    assert not d['deprecated_files'], p
    for g, store, nf in (('mu', mu, nf_mu), ('sec', sec, nf_sec)):
        for k, v in d[g].items(): store[k].append(np.asarray(v))
        s = seg(d[g]['event'])
        store['part'].append(np.full(len(s), ip)); store['fseg'].append(s); nf.append(s.max() + 1 if len(s) else 0)
nf_mu, nf_sec = np.array(nf_mu), np.array(nf_sec)
exp = expected(2671)
print(f'flux: {len(parts)} parts; files from mu-row restarts {nf_mu.sum()}, from sec-row restarts {nf_sec.sum()}, '
      f'expected 2671; parts matching split: mu {np.sum(nf_mu == exp)}, sec {np.sum(nf_sec == exp)}')
F = {'mu': {k: cat(v) for k, v in mu.items()}, 'sec': {k: cat(v) for k, v in sec.items()},
     'n_files': 2671, 'n_files_from_restarts': int(nf_mu.sum()), 'cluster': 21080019}
print(f'  mu rows {len(F["mu"]["pdg"])}, sec rows {len(F["sec"]["pdg"])}')
pickle.dump(F, open(f'{D}/flux_ms7_merged.pkl', 'wb'))

# ---- origin
parts = sorted(glob.glob(f'{D}/origin_parts/origin_sst_noSBT.part*'))
o, nf = collections.defaultdict(list), []
for ip, p in enumerate(parts):
    d = pickle.load(open(p, 'rb'))
    assert not d['bad'], p
    for k, v in d.items():
        if k not in ('jobs', 'bad'): o[k].append(np.asarray(v))
    s = seg(d['event']); o['part'].append(np.full(len(s), ip)); o['fseg'].append(s); nf.append(s.max() + 1 if len(s) else 0)
nf = np.array(nf); exp = expected(3086)
print(f'origin: {len(parts)} parts; files from restarts {nf.sum()}, expected 3086; parts matching split {np.sum(nf == exp)}; '
      f'diff hist {collections.Counter((nf - exp).tolist())}')
O = {k: cat(v) for k, v in o.items()}
O['n_files'] = 3086; O['n_files_from_restarts'] = int(nf.sum()); O['cluster'] = 21080523
print(f'  tracks {len(O["pdg"])}')
pickle.dump(O, open(f'{D}/origin_sst_noSBT_merged.pkl', 'wb'))
