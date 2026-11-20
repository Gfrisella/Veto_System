#!/usr/bin/env python3
"""How the SBT walls fire for three background classes (Guglielmo, 20 Nov 2026): (1) neutrino DIS, (2) muon DIS,
(3) independent muon combinatorial mu x mu. Design R2 (full SBT: 854 cells, threshold / efficiency of designs/R2_sbt_ubt.json).
Own deposits only: no random hits (background is tagged by its own hits, 20 Nov 2026).

Two questions per class:
  response  for every event (or mu x mu pair) of the class: which cells fire, how many, how much energy, where in z with
            respect to the vertex, when with respect to light from the vertex; for ALL events (random subsample, weights
            scaled; per spill) and for the SELECTED ones (selection before tagging: <= 2 tracks in SST1, pair cuts,
            measured IP < 2.5 m "partial" and < 10 cm "full"; per 15 years).
  rate      fired cells per spill per cell (spill 1 s: Hz) from the products of the class, next to the muon-crossing map
            that sets the SBT random rate now (sbt.cell_rates, 2.07 MHz > 30 MeV, x rate_scale 4 in R2).

Sources (existing stores, no new production):
  nu DIS   evstore/vessel_nu2026_R2_sbt_ubt (helium, vessel walls, air, planes), evstore/ms7_nu2026 (last 50 cm of MS7),
           sbt_d19/store_nu_noenrich (inside the SBT material: Al / LS / ribs)
  mu DIS   evstore/vessel_mu_R2_sbt_ubt (parent muon from its z = 30.6 m state to the vertex added), evstore/ms7_muon_v0s,
           sbt_d19/store_mu_{alin,ls,rib,alout}_noenrich (parent muon added)
  mu x mu  good SST1 muons of the Nov-4 dataset (rock_combinatorial.muon_tracks, 300 k rows, weights scaled) and their
           opposite-charge pairs through the combi_single pair cuts and measured IP (factor 2 dt / T_spill x spills / 15 y).
Deposits: sbt_material.deposits (charged dE/dx / beta^2 capped by the kinetic energy, photon conversions, neutral-hadron
interactions in the LS) summed per event and cell, then the threshold and efficiency (sbt_material.fired). Each fired
cell is attributed to the component that deposited most in it: products of the interaction, the scattered muon (beam
flag), the parent muon upstream of the vertex. MS7 stores: "all" and rate = code-3 events with their natural V0 decays;
"selected" = code 3 with V0 products dropped + the code-4 forced-decay copies (as dis_combinatorial). Muon paths of
mu x mu: stored A -> (K) -> B lines through the shell (sbt.shell_pieces), as sbt.own_hits.

    source bin/batch/env.sh
    python3 bin/sbt_firing.py dis --label nu_vessel --store vessel_nu2026_R2_sbt_ubt --out $VETO_RESULTS/sbt_firing
    python3 bin/sbt_firing.py dis --label mu_vessel --store vessel_mu_R2_sbt_ubt --muons --out ...
    python3 bin/sbt_firing.py mumu --out $VETO_RESULTS/sbt_firing
"""
import _bootstrap  # noqa: F401

import argparse
import json
import time
from pathlib import Path

import numpy as np

from veto import design as dsg
from veto import evstore as es
from veto import interactions as ix
from veto import io, paths, reco
from veto import sbt
from veto import sbt_material as sm
from veto import vertex_selection as vsel
from veto.config import DEFAULT_CONFIG, SPILLS_15Y

COMP = ("products", "scattered_mu", "parent_mu")
NC_BINS = np.arange(0, 42)                       # fired cells per event (last bin = overflow)
E_BINS = np.logspace(1, 4, 31)                   # MeV per fired cell
DZ_BINS = np.arange(-52, 53, 1.0)                # m, z(cell) - z(vertex)
DT_BINS = np.arange(-20, 20.5, 0.5)              # ns, t - t(vertex) - |dz| / c
Z_BINS = sbt.RING_EDGES
SHAPE = (4, sbt.N_RING, 4)                       # face, ring, segment
WINDOW_NS = 3.0 * np.hypot(1.0, 0.0) + 1.0       # zone-rule window (+-3 sigma of 1 ns + 1 ns), used for "in time" counts


def design_cfg(path):
    d = dsg.load(Path(path))
    return d, d.sbt


def zc_of(cell):
    ring = np.clip(cell // 10 % 1000, 0, sbt.N_RING - 1)
    return 0.5 * (sbt.RING_EDGES[ring] + sbt.RING_EDGES[ring + 1])


class Stats:
    """Weighted (and raw) accumulators for one event set."""
    def __init__(self, unit):
        self.unit = unit
        self.w_events = 0.0; self.n_events = 0
        self.w_fired = np.zeros(len(COMP) + 1); self.n_fired = np.zeros(len(COMP) + 1, np.int64)   # >= 1 cell (any = last)
        self.w_intime = 0.0; self.w_up = 0.0; self.w_down = 0.0
        self.ncell = np.zeros(len(NC_BINS)); self.ncell_raw = np.zeros(len(NC_BINS), np.int64)
        self.cellmap = np.zeros(SHAPE); self.cellmap_comp = np.zeros((len(COMP),) + SHAPE); self.cellmap_raw = np.zeros(SHAPE, np.int64)
        self.edep = np.zeros(len(E_BINS) - 1); self.dz = np.zeros(len(DZ_BINS) - 1); self.dt = np.zeros(len(DT_BINS) - 1)
        self.by_code = {}

    def add(self, w_ev, code, fired_ev, fired_cell, fired_edep, fired_comp, fired_dz, fired_dt):
        """w_ev, code (n,), fired_* per fired (event, cell) with fired_ev = index into w_ev."""
        n = len(w_ev)
        self.w_events += float(w_ev.sum()); self.n_events += n
        nc = np.bincount(fired_ev, minlength=n)
        k = np.minimum(nc, NC_BINS[-1])
        self.ncell += np.bincount(k, w_ev, len(NC_BINS)); self.ncell_raw += np.bincount(k, minlength=len(NC_BINS))
        anyf = nc > 0
        self.w_fired[-1] += float(w_ev[anyf].sum()); self.n_fired[-1] += int(anyf.sum())
        for ci in range(len(COMP)):
            m = np.zeros(n, bool); m[fired_ev[fired_comp == ci]] = True
            self.w_fired[ci] += float(w_ev[m].sum()); self.n_fired[ci] += int(m.sum())
        it = np.zeros(n, bool); it[fired_ev[(np.abs(fired_dt) < WINDOW_NS) & (fired_dz > -1.0)]] = True
        self.w_intime += float(w_ev[it].sum())
        up = np.zeros(n, bool); up[fired_ev[fired_dz < 0]] = True
        dn = np.zeros(n, bool); dn[fired_ev[fired_dz >= 0]] = True
        self.w_up += float(w_ev[up].sum()); self.w_down += float(w_ev[dn].sum())
        w = w_ev[fired_ev]
        face, ring, seg = fired_cell // 10 // 1000, fired_cell // 10 % 1000, fired_cell % 10
        flat = np.ravel_multi_index((face, ring, seg), SHAPE)
        size = int(np.prod(SHAPE))
        self.cellmap += np.bincount(flat, w, size).reshape(SHAPE)
        self.cellmap_raw += np.bincount(flat, minlength=size).reshape(SHAPE)
        for ci in range(len(COMP)):
            m = fired_comp == ci
            self.cellmap_comp[ci] += np.bincount(flat[m], w[m], size).reshape(SHAPE)
        self.edep += np.histogram(np.clip(fired_edep, E_BINS[0], E_BINS[-1] * 0.999), E_BINS, weights=w)[0]
        self.dz += np.histogram(np.clip(fired_dz, DZ_BINS[0], DZ_BINS[-1] - 1e-6), DZ_BINS, weights=w)[0]
        self.dt += np.histogram(np.clip(fired_dt, DT_BINS[0], DT_BINS[-1] - 1e-6), DT_BINS, weights=w)[0]
        for c in np.unique(code):
            m = code == c
            b = self.by_code.setdefault(int(c), [0.0, 0.0, 0, 0])
            b[0] += float(w_ev[m].sum()); b[1] += float(w_ev[m & anyf].sum()); b[2] += int(m.sum()); b[3] += int((m & anyf).sum())

    def summary(self):
        W = max(self.w_events, 1e-300)
        mean_nc = float((self.ncell * np.minimum(NC_BINS, NC_BINS[-1])).sum() / W)
        return dict(unit=self.unit, events=self.w_events, n_events_mc=self.n_events,
                    frac_fired_any=self.w_fired[-1] / W, n_fired_any_mc=int(self.n_fired[-1]),
                    frac_fired_by={c: self.w_fired[i] / W for i, c in enumerate(COMP)},
                    frac_fired_upstream_of_vertex=self.w_up / W, frac_fired_downstream_of_vertex=self.w_down / W,
                    frac_fired_intime_zone=self.w_intime / W, mean_fired_cells=mean_nc,
                    fired_cells_total=float(self.cellmap.sum()),
                    fired_cells_by_face={f: float(self.cellmap[i].sum()) for i, f in enumerate(sbt.FACES)},
                    by_code={str(c): dict(events=v[0], frac_fired=v[1] / max(v[0], 1e-300), n_mc=v[2], n_fired_mc=v[3])
                             for c, v in sorted(self.by_code.items())})

    def arrays(self, prefix):
        return {f"{prefix}_{k}": v for k, v in dict(ncell=self.ncell, ncell_raw=self.ncell_raw, cellmap=self.cellmap,
                                                    cellmap_comp=self.cellmap_comp, cellmap_raw=self.cellmap_raw,
                                                    edep=self.edep, dz=self.dz, dt=self.dt).items()}


# ------------------------------------------------------------------------------------------------ DIS stores

def event_fired(evsel, ev, tr, ev_index, muons, cfg, rng, chunk_tracks=4_000_000):
    """Fired cells of the events evsel (indices into ev): returns (event position in evsel, cell, edep, comp, dz, dt)."""
    n_ev = len(ev["event"])
    pos = np.full(n_ev, -1, np.int64); pos[evsel] = np.arange(len(evsel))
    rows = np.nonzero(pos[ev_index] >= 0)[0]                       # track rows of the chosen events (sorted by event)
    vtx = np.stack([ev["x"], ev["y"], ev["z"]], 1).astype(float)
    t_v = np.asarray(ev["t"], float)
    out = []
    # chunk boundaries on event changes, so that no (event, cell) sum is split
    starts = [0]
    while starts[-1] < len(rows):
        j = min(starts[-1] + chunk_tracks, len(rows))
        if j < len(rows):
            e_j = ev_index[rows[j]]
            while j < len(rows) and ev_index[rows[j]] == e_j:
                j += 1
        starts.append(j)
    parent = None
    if muons is not None:                                          # parent muon A (or bounce K) -> vertex
        mrow = np.asarray(ev["parent_row"], np.int64)[evsel]
        a = np.stack([np.asarray(muons[f"{c}_30"][mrow], float) for c in "xyz"], 1)
        k = np.stack([np.asarray(muons[f"k{c}"][mrow], float) for c in "xyz"], 1)
        vv = vtx[evsel]
        bounced = np.isfinite(k[:, 2]) & (k[:, 2] < vv[:, 2])
        a = np.where(bounced[:, None], k, a)
        dv = vv - a
        L = np.linalg.norm(dv, axis=1)
        pp = dv / np.maximum(L, 1e-9)[:, None] * np.asarray(ev["E"], float)[evsel][:, None]
        parent = dict(v=a, p=pp, L=L, t0=t_v[evsel] - L / sbt.C)
    for i0, i1 in zip(starts[:-1], starts[1:]):
        r = rows[i0:i1]
        if not len(r):
            continue
        epos = pos[ev_index[r]]
        comp = np.where(np.asarray(tr["beam"], bool)[r], 1, 0)
        v = np.stack([tr["x0"][r], tr["y0"][r], tr["z0"][r]], 1).astype(float)
        x1 = np.stack([tr["x1"][r], tr["y1"][r], tr["z1"][r]], 1).astype(float)
        p = np.stack([tr["px"][r], tr["py"][r], tr["pz"][r]], 1).astype(float)
        L = np.linalg.norm(x1 - v, axis=1)
        pdg = np.asarray(tr["pdg"])[r]; t0 = np.asarray(tr["t0"], float)[r]
        if parent is not None:
            e_lo, e_hi = epos.min(), epos.max() + 1
            m = np.arange(e_lo, e_hi)
            epos = np.concatenate([epos, m]); comp = np.concatenate([comp, np.full(len(m), 2)])
            v = np.concatenate([v, parent["v"][m]]); p = np.concatenate([p, parent["p"][m]])
            L = np.concatenate([L, parent["L"][m]]); pdg = np.concatenate([pdg, np.full(len(m), 13)])
            t0 = np.concatenate([t0, parent["t0"][m]])
        d = sm.deposits(epos * 4 + comp, pdg, v, p, t0, rng, length=np.minimum(L, 200.0))
        if not len(d["pid"]):
            continue
        e = d["pid"] // 4; c = d["pid"] % 4
        key = e * 10_000_000 + d["cell"]                           # sum the components per (event, cell)
        u, inv = np.unique(key, return_inverse=True)
        esum = np.bincount(inv, d["edep"])
        tsum = np.bincount(inv, d["edep"] * d["t"]) / np.maximum(esum, 1e-12)
        best = np.zeros(len(u), np.int64); bestE = np.zeros(len(u))
        for ci in range(3):
            m = c == ci
            ec = np.bincount(inv[m], d["edep"][m], len(u))
            better = ec > bestE
            best[better] = ci; bestE[better] = ec[better]
        f = sm.fired(dict(pid=u // 10_000_000, cell=u % 10_000_000, edep=esum, t=tsum, comp=best), cfg, rng)
        ge = evsel[f["pid"]]
        zc = zc_of(f["cell"])
        dz = zc - vtx[ge, 2]
        dt = f["t"] - t_v[ge] - np.abs(dz) / sbt.C
        out.append((f["pid"], f["cell"], f["edep"], f["comp"], dz, dt))
    if not out:
        z = np.zeros(0)
        return np.zeros(0, np.int64), np.zeros(0, np.int64), z, np.zeros(0, np.int64), z, z
    return tuple(np.concatenate([o[i] for o in out]) for i in range(6))


def ms7_beam_flag(ev, tr):
    """MS7 muon stores carry no beam flag: the scattered muon = a mu+- leaving the MS7 exit plane (tree parent -1) with the
    charge of the parent muon (combi_single rule; copies without a parent row take their original's)."""
    pr = np.asarray(ev["parent_row"]).astype(np.int64)
    if (pr < 0).any():
        idx = {int(x): i for i, x in enumerate(ev["event"])}
        cp = np.asarray(ev["copy_of"])
        pr = np.array([pr[i] if pr[i] >= 0 or cp[i] < 0 else pr[idx[int(cp[i])]] for i in range(len(pr))])
    eo = np.argsort(ev["event"], kind="stable")
    e = eo[np.searchsorted(np.asarray(ev["event"])[eo], tr["event"])]
    qpar = np.asarray(np.load(paths.ms7_parents(), mmap_mode="r")["q"])
    pdg = np.asarray(tr["pdg"])
    return (np.abs(pdg) == 13) & (np.asarray(tr["parent"]) < 0) & (pr[e] >= 0) & (np.sign(-pdg) == qpar[np.maximum(pr[e], 0)])


def selected_events(ev, tr, ip_cut, seed):
    cfg = reco.RecoConfig(seed=seed, multiple_scattering=True, ip_max=vsel.IP_CUTS[ip_cut])
    inter, part = es.to_fastveto(ev, tr)
    n = len(inter["weight"])
    tc = ix.charged_tracks(part)
    sel, pair, vertex = ix.select(tc, n, cfg=cfg)
    if len(sel):
        n_acc = np.bincount(tc["inter"][tc["acc"]], minlength=n)[sel]
        sel = sel[n_acc <= cfg.max_tracks]
    return np.sort(sel)


def run_dis(a, cfg):
    from dis_combinatorial import drop_v0_products_of_originals
    t0 = time.time()
    sm.register()
    meta, ev, tr = es.read(paths.evstore(a.store))
    n = len(ev["event"])
    ms7 = "ms7" in Path(a.store).name
    if ms7 and "muon" in Path(a.store).name:
        tr["beam"] = ms7_beam_flag(ev, tr)
        print(f"  MS7 muon store: scattered muon flagged on {int(tr['beam'].sum()):,} tracks", flush=True)
    print(f"{a.label}: {n:,} events, {len(tr['event']):,} tracks ({time.time() - t0:.0f} s)", flush=True)
    muons = io.open_store(paths.store_dir(paths.MUON_DATASET))[0] if a.muons else None
    eo = np.argsort(ev["event"], kind="stable")
    ev_index = eo[np.searchsorted(np.asarray(ev["event"])[eo], tr["event"])]
    w = np.asarray(ev["weight"], float); code = np.asarray(ev["code"])
    rng = np.random.default_rng(a.seed)
    res = dict(label=a.label, store=str(paths.evstore(a.store)), design=a.design, sbt_cfg=cfg, n_events=n, sets={})
    arrays = {}
    # ALL events: random subsample, per spill
    base = np.nonzero(code == 3)[0] if ms7 else np.arange(n)
    idx, scale = base, 1.0
    if a.all_max and len(base) > a.all_max:
        idx = np.sort(rng.choice(base, a.all_max, replace=False)); scale = len(base) / a.all_max
    st = Stats("per spill (4e13 PoT), spill 1 s: = Hz")
    f = event_fired(idx, ev, tr, ev_index, muons, cfg, rng)
    st.add(w[idx] * scale, code[idx], *f)
    res["sets"]["all"] = dict(st.summary(), subsample=len(idx), scale=scale, interactions_per_spill=float(w[base].sum()))
    arrays.update(st.arrays("all"))
    print(f"  all: {len(idx):,} events (x{scale:.3g}), fired >= 1 cell {res['sets']['all']['frac_fired_any']:.3f}, "
          f"fired cells {st.cellmap.sum():.4g} / spill ({time.time() - t0:.0f} s)", flush=True)
    # SELECTED events (before tagging), per 15 y
    tr_sel, ev_index_sel = tr, ev_index
    if ms7:
        tr_sel = drop_v0_products_of_originals(ev, tr)
        ev_index_sel = eo[np.searchsorted(np.asarray(ev["event"])[eo], tr_sel["event"])]
    for ip_cut in ("partial", "full"):
        sel = selected_events(ev, tr_sel, ip_cut, a.seed)
        st = Stats("per 15 years")
        if len(sel):
            f = event_fired(sel, ev, tr_sel, ev_index_sel, muons, cfg, rng)
            st.add(w[sel] * SPILLS_15Y, code[sel], *f)
        res["sets"][f"sel_{ip_cut}"] = st.summary()
        arrays.update(st.arrays(f"sel_{ip_cut}"))
        s = res["sets"][f"sel_{ip_cut}"]
        print(f"  selected {ip_cut}: {s['events']:.4g} / 15 y ({s['n_events_mc']} MC), fired >= 1 cell "
              f"{s['frac_fired_any']:.3f}, in-time zone {s['frac_fired_intime_zone']:.3f} ({time.time() - t0:.0f} s)", flush=True)
    return res, arrays


# ------------------------------------------------------------------------------------------------ mu x mu

def muon_fired(mu, cfg, rng):
    """Fired cells of each muon path (the sbt.own_hits model with the deposit kept): (row, cell, edep, z, t)."""
    tr = {k: mu[k] for k in ("ax", "ay", "az", "bx", "by", "bz", "kx", "ky", "kz")}
    p, q, row, after = sbt.path_segments(tr)
    c = sbt.shell_pieces(p, q)
    s = c["seg"].astype(int)
    dep = sbt.DEDX_LS * c["length"]
    ok = (c["ring"] >= 0) & (dep > cfg["threshold_mev"]) & (rng.random(len(dep)) < cfg["efficiency"])
    s = s[ok]
    length = np.linalg.norm(q[s] - p[s], axis=1) * (1 - c["t_mid"][ok]) + after[s]
    cell = sbt.cell_id(c["face"][ok], c["ring"][ok], c["cellseg"][ok])
    r = row[s]
    # one entry per (muon, cell): a bounced path may cross a cell twice
    key = r.astype(np.int64) * 10_000_000 + cell
    u, first, inv = np.unique(key, return_index=True, return_inverse=True)
    e = np.bincount(inv, dep[ok])
    return u // 10_000_000, u % 10_000_000, e, (-length / sbt.C)[first]


def pair_candidates(mu, cfg, block):
    """All opposite-charge muon pairs through the combi_single pair cuts, with the measured IP (rows a < b)."""
    import torch
    from benchmark_designs import _pair_cut
    r = torch.as_tensor(np.stack([mu["bx"], mu["by"], mu["bz"]], 1)); p = torch.as_tensor(np.stack([mu["bpx"], mu["bpy"], mu["bpz"]], 1))
    q = np.sign(mu["q"])
    B = np.stack([mu["bx"], mu["by"], mu["bz"]], 1); P = np.stack([mu["bpx"], mu["bpy"], mu["bpz"]], 1)
    pos, neg = np.nonzero(q > 0)[0], np.nonzero(q < 0)[0]
    out, rng = [], np.random.default_rng(777)
    t0 = time.time()
    for i0 in range(0, len(pos), block):
        ii = torch.as_tensor(pos[i0:i0 + block])
        for j0 in range(0, len(neg), block):
            jj = torch.as_tensor(neg[j0:j0 + block])
            ia = ii[:, None].expand(len(ii), len(jj)).reshape(-1); jb = jj[None, :].expand(len(ii), len(jj)).reshape(-1)
            x, y, v = _pair_cut(ia, jb, r, p)
            if not len(x):
                continue
            x, y, v = x.numpy(), y.numpy(), v.numpy()
            ip = vsel.measured_ip(B[x], P[x], B[y], P[y], cfg, rng)
            out.append((x, y, v, ip))
        if (i0 // block) % 8 == 0:
            print(f"    pairs: {i0 + block:,} / {len(pos):,} mu+ rows, {sum(len(o[0]) for o in out):,} candidates, "
                  f"{time.time() - t0:.0f} s", flush=True)
    return tuple(np.concatenate([o[i] for o in out]) for i in range(4))


def run_mumu(a, cfg):
    import torch
    from rock_combinatorial import muon_tracks
    from veto import muon_types as mt
    torch.set_num_threads(a.threads)
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    mu = muon_tracks(paths.MUON_DATASET, a.mu_sample)
    mu["q"] = mt.charge_from_pdg_column(mu["pdg"]).astype(float)
    n = len(mu["ax"])
    print(f"mu x mu: {n:,} good SST1 muons, {mu['weight'].sum():.4g} / spill ({time.time() - t0:.0f} s)", flush=True)
    frow, fcell, fe, ft = muon_fired(mu, cfg, rng)
    res = dict(label="mumu", dataset=paths.MUON_DATASET, design=a.design, sbt_cfg=cfg, mu_rows=n, sets={})
    arrays = {}
    # single muons ("all"): per spill
    st = Stats("per spill (4e13 PoT), spill 1 s: = Hz")
    zc = zc_of(fcell)
    st.add(mu["weight"], np.zeros(n, int), frow, fcell, fe, np.full(len(frow), 1), zc - 31.13, np.zeros(len(frow)))
    res["sets"]["muons"] = st.summary(); arrays.update(st.arrays("muons"))
    res["sets"]["muons"]["note"] = "dz = z(cell) - 31.13 m (no vertex), dt = 0 (own time), component = 'scattered_mu' slot = the muon"
    print(f"  single SST1 muons: fired >= 1 cell {res['sets']['muons']['frac_fired_any']:.3f} ({time.time() - t0:.0f} s)", flush=True)
    # pairs
    rcfg = reco.RecoConfig(seed=1, multiple_scattering=True)
    ia, ib, vtx, ip = pair_candidates(mu, rcfg, a.block)
    fac = 2 * DEFAULT_CONFIG["time_window"] / rcfg.spill_s * SPILLS_15Y
    wpair = mu["weight"][ia] * mu["weight"][ib] * fac
    order = np.argsort(frow, kind="stable")
    frow_s = frow[order]
    lo = np.searchsorted(frow_s, np.arange(n)); hi = np.searchsorted(frow_s, np.arange(n), side="right")
    nf = hi - lo
    print(f"  {len(ia):,} candidate pairs, {wpair.sum():.4g} / 15 y ({time.time() - t0:.0f} s)", flush=True)
    for lev, cut in (("cand", None), ("ip25", 2.5), ("ip01", 0.10)):
        k = np.ones(len(ia), bool) if cut is None else ip < cut
        pa, pb, pv, pw = ia[k], ib[k], vtx[k], wpair[k]
        # hits of the pair = hits of leg a (comp 0 slot -> "leg a") and leg b (slot 1); pid = pair index
        pid, cell, e, comp, t = [], [], [], [], []
        for leg, rows in ((0, pa), (1, pb)):
            cnt = nf[rows]
            rep = np.repeat(np.arange(len(rows)), cnt)
            src = order[np.concatenate([np.arange(lo[r], hi[r]) for r in rows[cnt > 0]])] if cnt.sum() else np.zeros(0, int)
            pid.append(rep); cell.append(fcell[src]); e.append(fe[src]); comp.append(np.full(len(src), leg)); t.append(ft[src])
        pid, cell, e, comp = (np.concatenate(x) for x in (pid, cell, e, comp))
        # a cell crossed by both legs: one fired cell (the energies add)
        key = pid * 10_000_000 + cell
        u, inv = np.unique(key, return_inverse=True)
        esum = np.bincount(inv, e)
        both = np.bincount(inv, minlength=len(u)) > 1
        cu = u % 10_000_000; pu = u // 10_000_000
        cmp_ = np.where(both, 2, np.bincount(inv, comp, len(u)).astype(int))
        dz = zc_of(cu) - pv[pu, 2]
        st = Stats("per 15 years (pair weight x 2 dt / T_spill x spills)")
        st.add(pw, np.zeros(len(pw), int), pu, cu, esum, cmp_, dz, np.zeros(len(pu)))
        s = st.summary()
        legs = np.zeros((len(pw), 2), bool)
        for leg in (0, 1):
            legs[np.unique(pid[comp == leg]), leg] = True
        W = max(pw.sum(), 1e-300)
        s.update(frac_leg_a_fired=float(pw[legs[:, 0]].sum() / W), frac_leg_b_fired=float(pw[legs[:, 1]].sum() / W),
                 frac_both_legs_fired=float(pw[legs.all(1)].sum() / W), frac_no_leg_fired=float(pw[~legs.any(1)].sum() / W),
                 frac_fired_upstream_of_vertex=s["frac_fired_upstream_of_vertex"],
                 note="components: products = leg a (mu+), scattered_mu = leg b (mu-), parent_mu = cell crossed by both")
        res["sets"][f"pairs_{lev}"] = s
        arrays.update(st.arrays(f"pairs_{lev}"))
        print(f"  pairs {lev}: {s['events']:.4g} / 15 y ({s['n_events_mc']:,} MC), >= 1 leg fired {s['frac_fired_any']:.3f}, "
              f"both legs {s['frac_both_legs_fired']:.3f}, upstream of vertex {s['frac_fired_upstream_of_vertex']:.3f}", flush=True)
    return res, arrays


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("dis", "mumu", "muonmap"))
    ap.add_argument("--label"); ap.add_argument("--store"); ap.add_argument("--muons", action="store_true")
    ap.add_argument("--design", default="designs/R2_sbt_ubt.json")
    ap.add_argument("--all-max", type=int, default=2_000_000, help="events of the 'all' set (random subsample, weights scaled)")
    ap.add_argument("--mu-sample", type=int, default=300_000); ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=8); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    _, cfg = design_cfg(a.design)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.what == "dis":
        res, arrays = run_dis(a, cfg)
        label = a.label
    elif a.what == "mumu":
        res, arrays = run_mumu(a, cfg)
        label = "mumu"
    else:                                   # the muon-crossing map that sets the random rate now (cached file)
        res, arrays, label = dict(label="muonmap", sets={}), {}, "muonmap"
        for thr in sbt.THRESHOLDS:
            ids, rate = sbt.cell_rates(thr)
            m = np.zeros(SHAPE)
            np.add.at(m, (ids // 10 // 1000, ids // 10 % 1000, ids % 10), rate)
            arrays[f"thr{int(thr)}_cellmap"] = m
            res["sets"][f"thr{int(thr)}"] = dict(total_hz=float(rate.sum()), max_cell_hz=float(rate.max()),
                                                  by_face={f: float(m[i].sum()) for i, f in enumerate(sbt.FACES)})
    res["seconds"] = round(time.time() - t0, 1)
    np.savez_compressed(out / f"{label}.npz", ring_edges=sbt.RING_EDGES, nc_bins=NC_BINS, e_bins=E_BINS, dz_bins=DZ_BINS,
                        dt_bins=DT_BINS, **arrays)
    json.dump(res, open(out / f"{label}.json", "w"), indent=1, default=float)
    print(f"wrote {out / label}.json / .npz ({res['seconds']} s)")


if __name__ == "__main__":
    main()
