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
import os
from pathlib import Path

import numpy as np

from . import paths
from .io import open_store

_SIG = paths.PROJECT_DIR / "data" / "signal"
STORE = Path(os.environ.get("VETO_SIGNAL_STORE", _SIG / "eventcalc_dcdcfc1" if (_SIG / "eventcalc_dcdcfc1").exists()
                            else _SIG / "eventcalc"))   # B4 done 12 Nov 2026: the regenerated set is the default
# ISSUES B4 (11 Nov 2026): the regenerated set (EventCalc dcdcfc1, decay volume = FairShip vessel, LLP from the target)
# is imported to data/signal/eventcalc_dcdcfc1 by bin/eventcalc_b4.sh; set VETO_SIGNAL_STORE there (or move it here).


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


# ---------------------------------------------------------------- B4: the signal model of the optimisation (2026-11-05)
BENCHMARK_SET = {"HNL-mu_m1_ct100": 1.0, "Dark-photons_m0.9_ct100": 1.0, "Scalar-mixing_m2_ct100": 1.0,
                 "ALP-fermion_m1_ct100": 1.0}   # name: weight of its signal loss in the average; EQUAL weights (Guglielmo, ISSUES B4, 11 Nov 2026)


def selected_signal(name, cfg, rng, n_max=None):
    """EventCalc decays of one benchmark through the same selection as every sample: charged tracks from the decay
    vertex (straight to SST1, timing detector through the field), pairs of good tracks (opposite charge, DOCA < 2 cm,
    vertex in the vessel, true IP < IP_PRECUT), measured IP < cfg.ip_max (10 cm full / 2.5 m partial) and at most
    cfg.max_tracks tracks in SST1. Returns (tracks in reco's signal format: bx.. bpz, vx vy vz; z_vertex per row; event
    per row; event weight per selected event; acceptance dict). n_max: at most this many decays (weighted draw)."""
    from . import geometry as geo
    from . import interactions as ix
    from . import spectrometer
    ev, pa = load(name)
    n_ev = len(ev["weight"])
    w_all = np.asarray(ev["weight"], np.float64)
    if n_max is not None and n_max < n_ev:
        pick = np.sort(rng.choice(n_ev, n_max, replace=False, p=w_all / w_all.sum()))
        w_ev = np.full(n_max, w_all.sum() / n_max)            # drawn proportional to w: equal weights
    else:
        pick, w_ev = np.arange(n_ev), w_all
    remap = np.full(n_ev, -1); remap[pick] = np.arange(len(pick))
    vtx, mom, q, idx = charged_tracks(ev, pa)
    keep = remap[idx] >= 0
    vtx, mom, q, inter = vtx[keep], mom[keep], q[keep].astype(float), remap[idx[keep]]
    in_vessel = np.asarray(ev["in_vessel"]).astype(bool)[pick]
    b = vtx + mom / mom[:, 2:3] * (geo.Z_SST1 - vtx[:, 2:3])
    acc = (np.abs(b[:, 0]) < geo.SST1_HALF_X) & (np.abs(b[:, 1]) < geo.SST1_HALF_Y)
    good = acc & (np.linalg.norm(mom, axis=1) > 1)
    g = np.nonzero(good)[0]
    if len(g):
        good[g] = spectrometer.timing_acceptance(b[g], mom[g], q[g])[0]
    good &= in_vessel[inter]
    tr = dict(row=np.arange(len(q)), inter=inter, q=q, beam=np.zeros(len(q), bool), v=vtx, p=mom, b=b, acc=acc, good=good)
    sel, pair, vertex = ix.select(tr, len(pick), cfg=cfg, rng=rng)
    if cfg.max_tracks is not None and len(sel):
        n_acc = np.bincount(inter[acc], minlength=len(pick))[sel]
        m = n_acc <= cfg.max_tracks
        sel, pair, vertex = sel[m], pair[m], vertex[m]
    rows = pair.T.ravel()                                       # all first tracks, then all second (reco's layout)
    n_sel = len(sel)
    sig = dict(bx=b[rows, 0], by=b[rows, 1], bz=np.full(len(rows), geo.Z_SST1), bpx=mom[rows, 0], bpy=mom[rows, 1],
               bpz=mom[rows, 2], vx=vtx[rows, 0], vy=vtx[rows, 1], vz=vtx[rows, 2])
    evt = np.tile(np.arange(n_sel), 2)
    acceptance = dict(benchmark=name, decays=int(len(pick)), weight_total=float(w_ev.sum()),
                      selected=int(n_sel), selected_weight_frac=float(w_ev[sel].sum() / w_ev.sum()) if len(pick) else 0.0)
    return sig, sig["vz"].copy(), evt, w_ev[sel], acceptance
