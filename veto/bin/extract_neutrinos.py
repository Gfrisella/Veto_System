"""Extract every neutrino from the MCTrack branch of the FairShip background files.

    python3 bin/extract_neutrinos.py [--in-dir data/flux] [--out-dir data/neutrinos] [--workers 24]

One output per input file, <out-dir>/<stem>.npz (written atomically; files already done are
skipped, so it can be rerun after an interruption), plus <out-dir>/meta.json with the number of
events and neutrinos per file. Every row is one MCTrack entry with |pdg| in (12, 14, 16):

    event       entry number in cbmsim (event index in the file)
    track       index of the neutrino in that event's MCTrack list
    pdg         neutrino pdg code
    px, py, pz  momentum [GeV]
    x, y, z     start point = production vertex [cm, FairShip frame]
    t           start time [ns]
    w           MCTrack fW
    proc        Geant4 process id of the production (fProcID)
    mother      index of the mother track in the event (-1 = primary)
    mother_pdg  pdg of the mother (0 if none)
    mu_sibling  1 if the mother also has a muon among its daughters (e.g. pi/K -> mu nu)
    mu_w, mu_p  fW and momentum [GeV] of that sister muon (nan if none)
    w_event     fW of track 0 of the event (the event weight in the muon background production)

Reads only the MCTrack leaf branches (the MCTrack object branch does not deserialize in uproot).
Needs uproot + awkward: on the cluster use the LCG view (see bin/run_on_gpu.sh).
"""

import _bootstrap  # noqa: F401
import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

LEAVES = ["fPdgCode", "fMotherId", "fPx", "fPy", "fPz", "fStartX", "fStartY", "fStartZ", "fStartT",
          "fW", "fProcID"]
BRANCHES = [f"MCTrack/MCTrack.{l}" for l in LEAVES]
STEP = "200 MB"


def extract(path, out_dir, entry_stop=None):
    path = Path(path)
    out = Path(out_dir) / (path.stem + ".npz")
    if out.exists():
        with np.load(out) as d:
            return path.name, int(d["n_events"]), len(d["pdg"]), 0.0, "skipped"
    t0 = time.time()
    cols = {k: [] for k in ("event", "track", "pdg", "px", "py", "pz", "x", "y", "z", "t", "w", "proc",
                            "mother", "mother_pdg", "mu_sibling", "mu_w", "mu_p", "w_event")}
    tree = uproot.open(path)["cbmsim"]
    n_events = tree.num_entries if entry_stop is None else min(entry_stop, tree.num_entries)
    for a, report in tree.iterate(BRANCHES, step_size=STEP, entry_stop=n_events, library="ak", report=True):
        a = {l: a[b] for l, b in zip(LEAVES, BRANCHES)}
        pdg = a["fPdgCode"]
        mid = ak.values_astype(a["fMotherId"], np.int64)
        local = ak.local_index(pdg)
        safe = ak.where(mid < 0, 0, mid)
        mother_pdg = ak.where(mid < 0, 0, pdg[safe])
        # muon sibling: some muon in the event has the same mother as this track
        is_mu = abs(pdg) == 13
        mu_p = np.sqrt(a["fPx"] ** 2 + a["fPy"] ** 2 + a["fPz"] ** 2)
        mus = ak.zip({"mid": mid[is_mu], "w": a["fW"][is_mu], "p": mu_p[is_mu]})
        pairs = ak.cartesian([mid, mus], nested=True)
        match = (pairs["0"] == pairs["1"].mid) & (mid >= 0)
        mu_sib = ak.any(match, axis=-1)
        mu_w = ak.fill_none(ak.max(ak.where(match, pairs["1"].w, -np.inf), axis=-1), np.nan)
        mu_pp = ak.fill_none(ak.max(ak.where(match, pairs["1"].p, -np.inf), axis=-1), np.nan)
        evt = ak.broadcast_arrays(ak.local_index(pdg, axis=0) + report.tree_entry_start, pdg)[0]
        w_event = ak.broadcast_arrays(ak.firsts(a["fW"]), pdg)[0]
        nu = (abs(pdg) == 12) | (abs(pdg) == 14) | (abs(pdg) == 16)

        def take(v, dtype):
            return ak.to_numpy(ak.flatten(v[nu])).astype(dtype)

        cols["event"].append(take(evt, np.int64))
        cols["track"].append(take(local, np.int32))
        cols["pdg"].append(take(pdg, np.int32))
        for k, l in (("px", "fPx"), ("py", "fPy"), ("pz", "fPz"), ("x", "fStartX"), ("y", "fStartY"),
                     ("z", "fStartZ"), ("t", "fStartT"), ("w", "fW")):
            cols[k].append(take(a[l], np.float32))
        cols["proc"].append(take(a["fProcID"], np.int32))
        cols["mother"].append(take(mid, np.int32))
        cols["mother_pdg"].append(take(mother_pdg, np.int32))
        cols["mu_sibling"].append(take(mu_sib, np.int8))
        cols["mu_w"].append(np.where(cols["mu_sibling"][-1] > 0, take(mu_w, np.float32), np.nan))
        cols["mu_p"].append(np.where(cols["mu_sibling"][-1] > 0, take(mu_pp, np.float32), np.nan))
        cols["w_event"].append(take(w_event, np.float32))
    cols = {k: np.concatenate(v) for k, v in cols.items()}
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, n_events=n_events, source=str(path), **cols)
    os.replace(tmp, out)
    return path.name, n_events, len(cols["pdg"]), time.time() - t0, "done"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default="data/flux")
    ap.add_argument("--pattern", default="*_mu.root")
    ap.add_argument("--out-dir", default="data/neutrinos")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--entry-stop", type=int, default=None, help="only the first N events per file (tests)")
    args = ap.parse_args()

    files = sorted(Path(args.in_dir).glob(args.pattern))
    if args.max_files:
        files = files[:args.max_files]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} files -> {args.out_dir}, {args.workers} workers", flush=True)
    t0 = time.time()
    meta = {}
    with Pool(args.workers) as pool:
        jobs = [pool.apply_async(extract, (f, args.out_dir, args.entry_stop)) for f in files]
        for i, j in enumerate(jobs):
            name, n_ev, n_nu, dt, status = j.get()
            meta[name] = {"n_events": n_ev, "n_neutrinos": n_nu}
            print(f"[{i + 1:3d}/{len(files)}] {name}: {n_ev} events, {n_nu} neutrinos, {dt:.0f} s ({status})"
                  f"  elapsed {time.time() - t0:.0f} s", flush=True)
    tot_ev = sum(m["n_events"] for m in meta.values())
    tot_nu = sum(m["n_neutrinos"] for m in meta.values())
    with open(Path(args.out_dir) / "meta.json", "w") as f:
        json.dump({"files": meta, "n_events": tot_ev, "n_neutrinos": tot_nu,
                   "columns": __doc__.split("\n\n")[3]}, f, indent=1)
    print(f"total: {tot_ev} events, {tot_nu} neutrinos, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
