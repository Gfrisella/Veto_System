#!/usr/bin/env python3
"""Reference input from the 67 original 2018 FairShip muon files (pythia8_Geant4_10.0_withCharmandBeauty*_mu.root,
6.60e10 p.o.t., "ecut=10.0 diMu100.0 X100.0"): every muon that reached the vetoPoint plane behind the absorber with
E > 10 GeV, once per muon — the definition of MuonsAndMatter's full_sample_after_target.h5, taken straight from the
files (no filter, no emulated selection).

Columns as the h5: px, py, pz (GeV, momentum at production: MCTrack), x, y (m, production point), z (= -0.0045 m), pdg
(+-13), weight (muons per spill: FairShip's MCTrack weight, 768.75 / 7.6875 / 326.16 / ..., x 6.504e10 / total p.o.t.,
the h5's normalisation is 5e13 / 768.75 = 6.504e10 p.o.t.). Extra: w_fairship (the MCTrack weight), mother (pdg of the
mother track, 0 if none), proc (Geant4 / TMC process of the muon's creation), z_start (m, production z in FairShip),
p_plane (GeV, momentum at the vetoPoint), x_plane, y_plane (m), file (run number of the file), event (entry in it).

    python3 bin/build_reference_2018.py --src /disk/lhcb_data/sqasim/fairship2018 --out /disk/lhcb_data/sqasim/reference-2018-fairship-506M
Writes <out>/parts/<run>.npz and <out>/reference_2018.h5.
"""

import argparse
import glob
import re
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

H5_POT = 5e13 / 768.75                         # p.o.t. the h5's weights assume
BR = ["MCTrack/MCTrack.fPdgCode", "MCTrack/MCTrack.fMotherId", "MCTrack/MCTrack.fPx", "MCTrack/MCTrack.fPy",
      "MCTrack/MCTrack.fPz", "MCTrack/MCTrack.fStartX", "MCTrack/MCTrack.fStartY", "MCTrack/MCTrack.fStartZ",
      "MCTrack/MCTrack.fW", "MCTrack/MCTrack.fProcID", "vetoPoint/vetoPoint.fTrackID", "vetoPoint/vetoPoint.fPdgCode",
      "vetoPoint/vetoPoint.fPx", "vetoPoint/vetoPoint.fPy", "vetoPoint/vetoPoint.fPz", "vetoPoint/vetoPoint.fX",
      "vetoPoint/vetoPoint.fY"]


def one(args):
    f, out = args
    import awkward as ak
    import uproot
    run = int(re.search(r"Beauty(\d+)_mu", f).group(1))
    part = Path(out) / "parts" / f"{run}.npz"
    u = uproot.open(f)
    pot = float(re.search(r"POT = (\d+)", u["FileHeader"].all_members["fTitle"]).group(1))
    if part.exists():
        return run, pot, int(np.load(part)["n"])
    cols = {k: [] for k in ("px", "py", "pz", "x", "y", "pdg", "w_fairship", "mother", "proc", "z_start", "p_plane",
                            "x_plane", "y_plane", "event")}
    e0 = 0
    for a in u["cbmsim"].iterate(BR, step_size=400000, library="ak"):
        g = lambda k: a[k]
        pdg, mom = g("MCTrack/MCTrack.fPdgCode"), g("MCTrack/MCTrack.fMotherId")
        tid = g("vetoPoint/vetoPoint.fTrackID")
        vpx, vpy, vpz = g("vetoPoint/vetoPoint.fPx"), g("vetoPoint/vetoPoint.fPy"), g("vetoPoint/vetoPoint.fPz")
        pv = np.sqrt(vpx ** 2 + vpy ** 2 + vpz ** 2)
        ok = (abs(g("vetoPoint/vetoPoint.fPdgCode")) == 13) & (tid >= 0) & (tid < ak.num(pdg)) & (pv > 10)
        ev_local = ak.local_index(tid, axis=0)
        ev = ak.to_numpy(ak.flatten(ak.broadcast_arrays(ev_local, tid)[0][ok])).astype(np.int64)
        tr = ak.to_numpy(ak.flatten(tid[ok])).astype(np.int64)
        pvf = ak.to_numpy(ak.flatten(pv[ok]))
        vx, vy = ak.to_numpy(ak.flatten(g("vetoPoint/vetoPoint.fX")[ok])), ak.to_numpy(ak.flatten(g("vetoPoint/vetoPoint.fY")[ok]))
        key = ev * 1000000 + tr
        _, first = np.unique(key, return_index=True)            # once per muon (its first vetoPoint)
        ev, tr, pvf, vx, vy = ev[first], tr[first], pvf[first], vx[first], vy[first]
        off = np.concatenate([[0], np.cumsum(ak.to_numpy(ak.num(pdg)))])
        gi = off[ev] + tr
        flat = lambda k: ak.to_numpy(ak.flatten(g(k)))
        mpdg_all = flat("MCTrack/MCTrack.fPdgCode")
        mo = flat("MCTrack/MCTrack.fMotherId")[gi]
        mother = np.where(mo >= 0, mpdg_all[off[ev] + np.maximum(mo, 0)], 0)
        cols["px"].append(flat("MCTrack/MCTrack.fPx")[gi]); cols["py"].append(flat("MCTrack/MCTrack.fPy")[gi])
        cols["pz"].append(flat("MCTrack/MCTrack.fPz")[gi])
        cols["x"].append(flat("MCTrack/MCTrack.fStartX")[gi] / 100.0); cols["y"].append(flat("MCTrack/MCTrack.fStartY")[gi] / 100.0)
        cols["z_start"].append(flat("MCTrack/MCTrack.fStartZ")[gi] / 100.0)
        cols["pdg"].append(mpdg_all[gi]); cols["w_fairship"].append(flat("MCTrack/MCTrack.fW")[gi])
        cols["mother"].append(mother); cols["proc"].append(flat("MCTrack/MCTrack.fProcID")[gi])
        cols["p_plane"].append(pvf); cols["x_plane"].append(vx / 100.0); cols["y_plane"].append(vy / 100.0)
        cols["event"].append(ev + e0)
        e0 += len(pdg)
    d = {k: np.concatenate(v) for k, v in cols.items()}
    tmp = part.with_suffix(".tmp.npz")
    np.savez(tmp, n=len(d["px"]), pot=pot, run=run, **d)
    tmp.rename(part)
    return run, pot, len(d["px"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="/disk/lhcb_data/sqasim/fairship2018")
    ap.add_argument("--out", default="/disk/lhcb_data/sqasim/reference-2018-fairship-506M")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    t0 = time.time()
    (Path(a.out) / "parts").mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(f"{a.src}/*_mu.root"))
    with Pool(a.workers) as pool:
        res = []
        for r in pool.imap_unordered(one, [(f, a.out) for f in files]):
            res.append(r)
            print(f"  run {r[0]:>6}: {r[2]:,} muons, {len(res)}/{len(files)} files, {time.time() - t0:.0f} s", flush=True)
    pot = sum(r[1] for r in res)
    scale = H5_POT / pot
    import h5py
    runs = sorted(r[0] for r in res)
    n = sum(r[2] for r in res)
    out = Path(a.out) / "reference_2018.h5"
    tmp = out.with_suffix(".tmp.h5")
    spec = {"px": "f4", "py": "f4", "pz": "f4", "x": "f4", "y": "f4", "z": "f4", "pdg": "f4", "weight": "f4",
            "w_fairship": "f4", "mother": "i4", "proc": "i2", "z_start": "f4", "p_plane": "f4", "x_plane": "f4",
            "y_plane": "f4", "file": "i4", "event": "i4"}
    with h5py.File(tmp, "w") as h:
        ds = {k: h.create_dataset(k, (n,), dtype=t, chunks=(1 << 20,)) for k, t in spec.items()}
        i = 0
        for run in runs:
            d = np.load(Path(a.out) / "parts" / f"{run}.npz")
            m = int(d["n"])
            for k in spec:
                if k == "z":
                    v = np.full(m, -0.0045)
                elif k == "weight":
                    v = d["w_fairship"] * scale
                elif k == "file":
                    v = np.full(m, run)
                else:
                    v = d[k]
                ds[k][i:i + m] = v
            i += m
        h.attrs["description"] = ("2018 FairShip muon background (67 files, pythia8_Geant4_10.0_withCharmandBeauty*_mu.root, "
                                  "ecut=10 diMu100 X100): muons at the vetoPoint plane with E > 10 GeV, momentum at production; "
                                  "weight per spill = FairShip weight x 6.504e10 / total p.o.t. (bin/build_reference_2018.py)")
        h.attrs["pot"] = pot
        h.attrs["weight_scale"] = scale
    tmp.rename(out)
    print(f"{len(files)} files, {pot:.4g} p.o.t. (weight scale {scale:.4f}), {n:,} muons -> {out}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
