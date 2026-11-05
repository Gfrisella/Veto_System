"""EventCalc LLP signal samples (ISSUES B4), converted by bin/import_eventcalc.py.

Store: veto/data/signal/eventcalc/ (see the README.md there)
    events/     one row per LLP decay (43 benchmarks concatenated, column `bench`)
    particles/  one row per final-state particle, events/first and events/n_part index it
    benchmarks.json, meta.json

Frame: veto frame (z_veto = z_FairShip - 2.14 = z_EventCalc - 0.87 for these samples), metres, GeV.
Weight: expected visible decays in EventCalc's decay volume for 6e20 PoT at the benchmark coupling;
sum(weight) per benchmark = EventCalc's N_ev_tot. For efficiencies use weight (or p_decay) ratios.

    from veto import signal_eventcalc as sig
    sig.benchmarks()                                  # list of dicts (name, model, mass, ctau, n_events, ...)
    ev, pa = sig.load("HNL-mu_m1_ct100")              # dicts of arrays (one benchmark, particle `event` re-indexed)
    vtx, mom, q, ev_idx = sig.charged_tracks(ev, pa)  # visible charged tracks ready for propagation to SST1
"""

import json
from pathlib import Path

import numpy as np

from . import paths
from .io import open_store

STORE = paths.PROJECT_DIR / "data" / "signal" / "eventcalc"


def store_dir() -> Path:
    return STORE


def meta():
    with open(STORE / "meta.json") as f:
        return json.load(f)


def benchmarks():
    """Per-benchmark records (id, name, model, mass, ctau, coupling_squared, n_events, event_range, n_ev_tot, ...)."""
    with open(STORE / "benchmarks.json") as f:
        return json.load(f)


def _bench_ids(benchmark):
    if benchmark is None:
        return None
    sel = benchmark if isinstance(benchmark, (list, tuple, set)) else [benchmark]
    out = []
    by_name = {b["name"]: b["id"] for b in benchmarks()}
    for s in sel:
        if isinstance(s, (int, np.integer)):
            out.append(int(s))
        elif s in by_name:
            out.append(by_name[s])
        else:
            raise KeyError(f"unknown benchmark {s!r}; known: {sorted(by_name)}")
    return sorted(out)


def load(benchmark=None, columns=None, mmap=False):
    """(events, particles) as dicts of numpy arrays.

    benchmark: None (all), a name ("HNL-mu_m1_ct100"), an id, or a list of them.
    columns:   event columns to return (default all); particle columns are always all.
    mmap:      with benchmark=None, return the memory-mapped arrays without copying.
    Selected subsets are contiguous per benchmark; particles' `event` and events' `first` are re-indexed to
    the returned arrays. Extra event keys: `channel_name` table via meta()["channels"], `model_name` likewise.
    """
    ev_store, ev_meta = open_store(STORE / "events")
    pa_store, _ = open_store(STORE / "particles")
    keys = columns or list(ev_store)
    ids = _bench_ids(benchmark)
    if ids is None:
        ev = {k: (ev_store[k] if mmap else np.asarray(ev_store[k])) for k in keys}
        pa = {k: (v if mmap else np.asarray(v)) for k, v in pa_store.items()}
        return ev, pa
    bl = benchmarks()
    ev_parts, pa_parts, n_ev, n_pa = {k: [] for k in set(keys) | {"first", "n_part"}}, {k: [] for k in pa_store}, 0, 0
    for i in ids:
        a, b = bl[i]["event_range"]
        first = np.asarray(ev_store["first"][a:b])
        npart = np.asarray(ev_store["n_part"][a:b])
        pa0, pa1 = int(first[0]), int(first[-1] + npart[-1])
        for k in ev_parts:
            v = np.asarray(ev_store[k][a:b])
            if k == "first":
                v = v - pa0 + n_pa
            ev_parts[k].append(v)
        for k in pa_parts:
            v = np.asarray(pa_store[k][pa0:pa1])
            if k == "event":
                v = v - a + n_ev
            pa_parts[k].append(v)
        n_ev += b - a
        n_pa += pa1 - pa0
    ev = {k: np.concatenate(v) for k, v in ev_parts.items() if k in keys or columns is None}
    pa = {k: np.concatenate(v) for k, v in pa_parts.items()}
    return ev, pa


def channel_names(ev):
    """Decay-channel string per event."""
    return np.asarray(meta()["channels"], dtype=object)[ev["channel"]]


def charged_tracks(ev, pa, tracks_only=True, forward_only=True):
    """Charged final-state particles as arrays for propagation to SST1 (veto frame).

    Returns vertex (n, 3) [m], momentum (n, 3) [GeV], charge (n,) [e], event index (n,) into `ev`.
    tracks_only: keep only long-lived charged species (e, mu, pi, K, p: particles/is_track); else any charge != 0.
    forward_only: drop pz <= 0.
    """
    sel = pa["is_track"].astype(bool) if tracks_only else (pa["charge"] != 0)
    if forward_only:
        sel &= pa["pz"] > 0
    idx = np.asarray(pa["event"])[sel]
    vtx = np.stack([np.asarray(ev["x"])[idx], np.asarray(ev["y"])[idx], np.asarray(ev["z"])[idx]], -1)
    mom = np.stack([pa["px"][sel], pa["py"][sel], pa["pz"][sel]], -1).astype(np.float64)
    return vtx, mom, np.asarray(pa["charge"])[sel].astype(np.int8), idx


def production_point():
    """LLP production point (EventCalc origin) in the veto frame. NOTE: -0.87 m, not geometry.Z_TARGET (-2.14)."""
    return np.array(meta()["frame"]["production_point_veto"])
