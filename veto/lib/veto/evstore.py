"""Design-independent event store: every background source as events + track trees from its reference surface down to
SST1, so that a design is evaluated by intersecting the stored tracks with its planes (no new physics per design).

Layout (one directory per source and version):
  meta.json      source, version, pot_per_spill, reference surface, generator, downstream model, parent table
  events.npz     one row per interaction (or forced-decay copy):
                 event, weight (per spill of POT_PER_SPILL), code (interactions.CODE_NAMES / 5+ rock regions),
                 parent_row (row in the source's parent table, -1 if none), x, y, z (vertex), t (ns from the target),
                 E (parent energy), copy_of (-1 or the event this one copies), v0 (pdg of the forced V0 or 0)
  tracks.npz     one row per particle, the tree: event, track (within the event), parent (track id, -1 = leaves the
                 reference surface), pdg, x0, y0, z0, t0 (start), px, py, pz (momentum at the start), x1, y1, z1 (end),
                 end (END_*), proc (creator: PROC_*), beam (the parent muon after the interaction)

Downstream model of propagate(): straight lines (no field before the spectrometer, no material), decays in flight with
the lifetimes and main channels (fastveto.decay plus pi+-, K+-), until the track leaves the cavern box, reaches
SST1 (z = geometry.Z_SST1) or passes Z_END. Not modelled (yet): conversions and hadronic interactions in air / helium /
vessel walls / planes, the spectrometer field (the reconstruction uses the state at SST1).
"""
import json
from pathlib import Path

import numpy as np

from fastveto import decay as dc

from . import geometry as G

VERSION = "evstore-1"
END_LEFT, END_DECAY, END_FORCED, END_SST1, END_STOP = 0, 1, 2, 3, 4
PROC_PRIMARY, PROC_DECAY, PROC_FORCED = 0, 1, 2
CAVERN = dict(x0=G.CAVERN_X_WALL, x1=11.43, y0=G.CAVERN_Y_FLOOR, y1=10.64, z_min=-5.0)
C_LIGHT = 0.299792458                                   # m / ns
CTAU = {**dc.CTAU, 211: 7.8045, 321: 3.712}
CHANNELS = {**dc.CHANNELS, **dc.KL_CHANNELS,
            211: [(1.0, (-13, 14))],
            321: [(0.6356, (-13, 14)), (0.2067, (211, 111)), (0.0559, (211, 211, -211)), (0.0507, (111, -11, 12)),
                  (0.0335, (111, -13, 14)), (0.0176, (211, 111, 111))]}
MASS = {**dc.MASS, 16: 0.0, 321: dc.MASS[321]}
EVENT_COLS = ("event", "weight", "code", "parent_row", "x", "y", "z", "t", "E", "copy_of", "v0")
TRACK_COLS = ("event", "track", "parent", "pdg", "x0", "y0", "z0", "t0", "px", "py", "pz", "x1", "y1", "z1", "end",
              "proc", "beam")


def mass(pdg):
    return np.array([MASS.get(abs(int(q)), 0.0) for q in np.asarray(pdg)])


def _exit_length(x, u, z_end):
    """Path length along u to leave the cavern box (x, y), to reach z_end (forward) or z_min (backward)."""
    s = np.full(len(x), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for k, (lo, hi) in enumerate(((CAVERN["x0"], CAVERN["x1"]), (CAVERN["y0"], CAVERN["y1"]))):
            t = np.where(u[:, k] > 0, (hi - x[:, k]) / u[:, k], np.where(u[:, k] < 0, (lo - x[:, k]) / u[:, k], np.inf))
            s = np.minimum(s, t)
        tz = np.where(u[:, 2] > 0, (z_end - x[:, 2]) / u[:, 2], np.where(u[:, 2] < 0, (CAVERN["z_min"] - x[:, 2]) / u[:, 2], np.inf))
    return np.maximum(np.minimum(s, tz), 0.0)


def _decay_products(pdg, p, rng):
    """Vectorised decays of particles (pdg (n,), p (n,3)) at rest-frame channels: returns (owner, pdg, p) of the products."""
    own, out_pdg, out_p = [], [], []
    for a in np.unique(np.abs(pdg)):
        sel = np.nonzero(np.abs(pdg) == a)[0]
        M = MASS[a]
        P = p[sel]
        br = np.array([c[0] for c in CHANNELS[a]])
        ch = np.searchsorted(np.cumsum(br / br.sum()), rng.random(len(sel)), side="right")
        for ci, (_, prods) in enumerate(CHANNELS[a]):
            m = ch == ci
            if not m.any():
                continue
            s = sel[m]
            k = len(s)
            ms = [MASS[abs(q)] for q in prods]
            if len(prods) == 2:
                q = dc._two_body(M, ms[0], ms[1], k, rng)
                rest = [q, -q]
            else:
                r3 = dc._three_body(M, ms, k, rng)
                rest = [r3[:, 0], r3[:, 1], r3[:, 2]]
            sign = np.sign(pdg[s])
            for q_rest, code, mi in zip(rest, prods, ms):
                e_rest = np.sqrt((q_rest ** 2).sum(1) + mi ** 2)
                plab, _ = dc._boost(q_rest, e_rest, P[m], M)
                own.append(s); out_pdg.append(np.where(sign > 0, code, dc._conj(code)).astype(np.int64)); out_p.append(plab)
    if not own:
        return np.zeros(0, int), np.zeros(0, np.int64), np.zeros((0, 3))
    return np.concatenate(own), np.concatenate(out_pdg), np.concatenate(out_p)


def propagate(event, pdg, x, p, t, rng, forced=None, beam=None, z_end=95.0, max_gen=6):
    """Grow the trees from the particles at the reference surface.

    event, pdg (n,), x (n,3) start, p (n,3), t (n,) [ns]; beam (n,) bool. forced: optional dict for particles whose
    decay is already simulated (forced V0 decays): rows (m,) indices into the input, dec (m,3) decay points, and the
    products prod_owner (k,) (index into rows), prod_pdg (k,), prod_p (k,3).
    Returns the track columns (dict of arrays), with track ids per event and parent links."""
    n0 = len(pdg)
    ev = [np.asarray(event, np.int64)]; pg = [np.asarray(pdg, np.int64)]; X0 = [np.asarray(x, float)]
    P0 = [np.asarray(p, float)]; T0 = [np.asarray(t, float)]; par = [np.full(n0, -1, np.int64)]
    proc = [np.zeros(n0, np.int8)]; bm = [np.zeros(n0, bool) if beam is None else np.asarray(beam, bool)]
    X1, END = [], []
    gen_ev, gen_pdg, gen_x, gen_p, gen_t = ev[0], pg[0], X0[0], P0[0], T0[0]
    gen_idx = np.arange(n0)                                # global row of each particle of this generation
    total = n0
    f_rows = np.asarray(forced["rows"], int) if forced else np.zeros(0, int)
    is_f0 = np.zeros(n0, bool); is_f0[f_rows] = True
    f_dec = np.zeros((n0, 3)); f_dec[f_rows] = np.asarray(forced["dec"], float).reshape(-1, 3) if forced else 0
    for _ in range(max_gen):
        k = len(gen_pdg)
        if k == 0:
            break
        pm = np.linalg.norm(gen_p, axis=1)
        u = gen_p / np.maximum(pm, 1e-12)[:, None]
        a = np.abs(gen_pdg)
        s_exit = _exit_length(gen_x, u, z_end)
        unstable = np.isin(a, list(CHANNELS)) & np.isin(a, list(CTAU)) & ~((gen_pdg < 0) & np.isin(-gen_pdg, [111, 221, 310, 130]))
        m_ = mass(gen_pdg)
        L = np.full(k, np.inf)
        if unstable.any():
            ctau = np.array([CTAU.get(int(q), np.inf) for q in a[unstable]])
            L[unstable] = rng.exponential(1.0, unstable.sum()) * ctau * pm[unstable] / np.maximum(m_[unstable], 1e-12)
        is_forced = is_f0 if _ == 0 else np.zeros(k, bool)
        dec = (L < s_exit) & ~is_forced
        end = np.where(dec, END_DECAY, END_LEFT).astype(np.int8)
        s_end = np.where(dec, L, s_exit)
        x1 = gen_x + u * np.where(np.isfinite(s_end), s_end, 0.0)[:, None]
        # forced decays: end at their simulated decay point
        fi = np.nonzero(is_forced)[0]
        x1[fi] = f_dec[gen_idx[fi]]; end[fi] = END_FORCED
        reach = (end == END_LEFT) & (x1[:, 2] >= G.Z_SST1 - 1e-6) & (gen_x[:, 2] < G.Z_SST1)
        end[reach] = END_SST1
        X1.append(x1); END.append(end)
        # next generation: products of the decays (random) and of the forced decays (given)
        di = np.nonzero(dec)[0]
        own, npdg, npp = _decay_products(gen_pdg[di], gen_p[di], rng)
        own = di[own] if len(own) else own
        if len(fi):                                        # generation 0 only: rows of the input = indices
            f_own = f_rows[np.asarray(forced["prod_owner"], int)]
            f_pdg = np.asarray(forced["prod_pdg"], np.int64); f_p = np.asarray(forced["prod_p"], float).reshape(-1, 3)
        else:
            f_own, f_pdg, f_p = np.zeros(0, int), np.zeros(0, np.int64), np.zeros((0, 3))
        nf = len(f_own)
        own = np.concatenate([own, f_own]).astype(int)
        npdg = np.concatenate([npdg, f_pdg])
        npp = np.concatenate([npp.reshape(-1, 3), f_p])
        nproc = np.concatenate([np.full(len(own) - nf, PROC_DECAY), np.full(nf, PROC_FORCED)]).astype(np.int8)
        if not len(own):
            break
        nx = x1[own]
        flight = np.linalg.norm(x1[own] - gen_x[own], axis=1)
        beta = pm[own] / np.sqrt(pm[own] ** 2 + m_[own] ** 2)
        nt = gen_t[own] + flight / (np.maximum(beta, 1e-6) * C_LIGHT)
        nidx = np.arange(total, total + len(own)); total += len(own)
        ev.append(gen_ev[own]); pg.append(npdg); X0.append(nx); P0.append(npp); T0.append(nt)
        par.append(gen_idx[own]); proc.append(nproc); bm.append(np.zeros(len(own), bool))
        gen_ev, gen_pdg, gen_x, gen_p, gen_t, gen_idx = gen_ev[own], npdg, nx, npp, nt, nidx
    if len(X1) < len(X0):                                   # products of the last allowed generation: not propagated
        k = len(X0[-1]); X1.append(X0[-1].copy()); END.append(np.full(k, END_STOP, np.int8))
    cols = dict(event=np.concatenate(ev), pdg=np.concatenate(pg), parent_global=np.concatenate(par),
                proc=np.concatenate(proc), beam=np.concatenate(bm), t0=np.concatenate(T0), end=np.concatenate(END))
    x0, p0, x1 = np.concatenate(X0), np.concatenate(P0), np.concatenate(X1)
    for i, c in enumerate("xyz"):
        cols[f"{c}0"] = x0[:, i]; cols[f"p{c}"] = p0[:, i]; cols[f"{c}1"] = x1[:, i]
    # per-event track ids: order by event (stable), track = position within the event, parent -> that numbering
    order = np.argsort(cols["event"], kind="stable")
    inv = np.empty_like(order); inv[order] = np.arange(len(order))
    out = {k: v[order] for k, v in cols.items()}
    first = np.r_[0, np.nonzero(np.diff(out["event"]))[0] + 1]
    start = np.repeat(first, np.diff(np.r_[first, len(order)]))
    out["track"] = (np.arange(len(order)) - start).astype(np.int32)
    pg_ = out.pop("parent_global")
    out["parent"] = np.where(pg_ >= 0, inv[np.maximum(pg_, 0)] - start, -1).astype(np.int32)
    return out


def write(directory, meta, events, tracks):
    d = Path(directory); d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "events.npz", **{k: np.asarray(events[k]) for k in EVENT_COLS})
    np.savez(d / "tracks.npz", **{k: np.asarray(tracks[k]) for k in TRACK_COLS})
    meta = dict(meta, version=VERSION, n_events=int(len(events["event"])), n_tracks=int(len(tracks["event"])))
    json.dump(meta, open(d / "meta.json", "w"), indent=1)
    return meta


def read(directory):
    d = Path(directory)
    meta = json.load(open(d / "meta.json"))
    ev = dict(np.load(d / "events.npz")); tr = dict(np.load(d / "tracks.npz"))
    return meta, ev, tr


def to_fastveto(ev, tr):
    """inter / part dicts for interactions.evaluate: one interaction per stored event; its particles are the charged
    tracks that reach SST1 (straight from their start point, as the store propagates them) plus everything else at its
    start (neutral or not reaching SST1, ignored by charged_tracks unless it heads downstream)."""
    idx = {int(e): i for i, e in enumerate(ev["event"])}
    row = np.array([idx[int(e)] for e in tr["event"]]) if len(tr["event"]) else np.zeros(0, int)
    inter = dict(weight=np.asarray(ev["weight"], float), code=np.asarray(ev["code"]), evt_n=np.asarray(ev["parent_row"]),
                 x=np.asarray(ev["x"], float), y=np.asarray(ev["y"], float), z=np.asarray(ev["z"], float))
    p = np.stack([tr["px"], tr["py"], tr["pz"]], 1).astype(float)
    E = np.sqrt((p ** 2).sum(1) + mass(tr["pdg"]) ** 2)
    keep = (tr["end"] == END_SST1) | ((tr["end"] != END_DECAY) & (tr["end"] != END_FORCED))
    part = dict(int=row[keep], pdg=np.asarray(tr["pdg"])[keep], p=np.concatenate([p, E[:, None]], 1)[keep],
                v=np.stack([tr["x0"], tr["y0"], tr["z0"]], 1).astype(float)[keep], beam=np.asarray(tr["beam"], bool)[keep])
    return inter, part
