"""Backgrounds from particle interactions after the muon shield (lib/fastveto) through our selection and veto.

fastveto (fastveto.dis for muons, fastveto.nudis for neutrinos) forces interactions in the vessel helium, the
air (z = 30.6 - 82 m), the PVC walls and every plane of a design, with a weight (interactions per spill) and the
final-state particles (pdg, momentum, production point, beam = the scattered muon / outgoing lepton). Here:

  tracks     charged particles are straight lines from their production point (no field before the
             spectrometer); for muon interactions the scattered muon's path starts at the PARENT muon's
             z = 30.6 m state (its row of the muon dataset), kinks at the interaction vertex and goes on to
             z = 82 m, so the parent's real hits upstream of the vertex are digitised
  selection  as the combinatorial pairs (bin/benchmark_designs.candidates): two opposite-charge tracks of the
             same interaction with p > 1 GeV in the SST1 and timing acceptance, DOCA < 2 cm, vertex in the
             vessel, IP < 2.5 m. No time-coincidence factor: the particles of one interaction are in time.
             Then, as for every sample (RecoConfig): measured IP < ip_max (0.1 m) and at most max_tracks (2)
             tracks in the SST1 acceptance in the event (a two-body decay has two; interactions usually more).
  veto       reco.run_events (multiple scattering, own and random hits, algorithm A) at three levels:
               a  algorithm A on the two vertex tracks (planes upstream of the vertex)
               b  algorithm A on every reconstructed track of the event (SST1 acceptance): the scattered
                  muon, if it is not a vertex track, is followed back past the vertex to its parent's hits
               c  truth bound: the parent muon crossed a plane upstream of the interaction (no efficiency,
                  no reconstruction): the best any algorithm can do. Neutrinos have no parent.
Results per target (helium, air, PVC, each plane): interactions, selected, surviving a / b / c, per 15 years,
with errors from a bootstrap over input muons (evt_n: rows of one input muon are not independent).
"""

import numpy as np

from . import geometry as geo
from . import muon_types as mt
from . import reco

SPILLS_15Y = 6e20 / 5e13
CHARGE = {11: -1, 13: -1, 15: -1, 211: 1, 321: 1, 2212: 1, 3222: 1, 3112: -1, 3312: -1, 3334: -1, 411: 1, 431: 1,
          4122: 1, 521: 1}
CODE_NAMES = {0: "helium", 1: "air", 2: "pvc"}


def charge(pdg):
    """Electric charge of PDG codes (0 for neutral or unknown)."""
    a = np.abs(pdg)
    q = np.zeros(len(pdg), np.int8)
    for code, c in CHARGE.items():
        q[a == code] = c
    return np.where(pdg < 0, -q, q)


def target_name(code, design):
    code = int(code)
    return CODE_NAMES[code] if code < 10 else design.planes[code - 10].name


def charged_tracks(part):
    """Charged final-state particles moving downstream, as straight lines to z = 82 m (state B), with
    the SST1 / timing acceptance of the vertex tracks. Returns a dict of arrays (one row per particle)."""
    q = charge(part["pdg"])
    p, v = part["p"][:, :3], part["v"]
    keep = np.nonzero((q != 0) & (p[:, 2] > 0) & (v[:, 2] < geo.Z_SST1))[0]
    p, v = p[keep], v[keep]
    b = v + p / p[:, 2:3] * (geo.Z_SST1 - v[:, 2:3])
    pt = np.linalg.norm(p, axis=1)
    dz = geo.Z_TIMING - b[:, 2]
    xt, yt = b[:, 0] + p[:, 0] / p[:, 2] * dz, b[:, 1] + p[:, 1] / p[:, 2] * dz
    acc = (np.abs(b[:, 0]) < geo.SST1_HALF_X) & (np.abs(b[:, 1]) < geo.SST1_HALF_Y)
    good = acc & (pt > 1) & (np.abs(xt) < geo.TIMING_HALF_X) & (np.abs(yt) < geo.TIMING_HALF_Y)
    return dict(row=keep, inter=part["int"][keep], q=q[keep], beam=part["beam"][keep], v=v, p=p, b=b, acc=acc, good=good)


def select(tr, n_inter, device="cpu", cfg=None, rng=None):
    """Vertex candidates among the good tracks of each interaction (benchmark_designs.candidates), with the
    pointing cut on measured tracks (cfg.ip_max, vertex_selection) if cfg is given. Returns (selected
    interactions, the first passing pair per selected interaction as track indices, its vertex)."""
    import torch
    from benchmark_designs import candidates
    good = np.nonzero(tr["good"])[0]
    n_good = np.bincount(tr["inter"][good], minlength=n_inter)
    good = good[n_good[tr["inter"][good]] >= 2]
    if not len(good):
        return np.zeros(0, int), np.zeros((0, 2), int), np.zeros((0, 3))
    ids, group = np.unique(tr["inter"][good], return_inverse=True)
    t = {k: torch.as_tensor(a, dtype=torch.float64, device=device) for k, a in
         (("x", tr["b"][good, 0]), ("y", tr["b"][good, 1]), ("z", tr["b"][good, 2]),
          ("px", tr["p"][good, 0]), ("py", tr["p"][good, 1]), ("pz", tr["p"][good, 2]),
          ("pdg", -tr["q"][good].astype(float)))}         # candidates() wants opposite pdg signs
    t["file_index"] = torch.as_tensor(group, device=device)
    pairs, vertex, _ = candidates(t, device, spill_chunk=20000)
    pairs, vertex = pairs.cpu().numpy(), vertex.cpu().numpy()
    if cfg is not None and cfg.ip_max is not None and len(pairs):
        from . import vertex_selection as vsel
        g0, g1 = good[pairs[:, 0]], good[pairs[:, 1]]
        ip = vsel.measured_ip(tr["b"][g0], tr["p"][g0], tr["b"][g1], tr["p"][g1], cfg, rng or np.random.default_rng(779))
        pairs, vertex = pairs[ip < cfg.ip_max], vertex[ip < cfg.ip_max]
    if not len(pairs):
        return np.zeros(0, int), np.zeros((0, 2), int), np.zeros((0, 3))
    inter = ids[group[pairs[:, 0]]]
    first = np.unique(inter, return_index=True)[1]
    return inter[first], good[pairs[first]], vertex[first]


def event_tracks(inter, tr, sel, pair, vertex, muons=None):
    """Tracks of the selected events in reco's muon-track format, their event ids and z limits for the
    veto levels a (vertex tracks only) and b (every reconstructed track)."""
    rows = np.nonzero(np.isin(tr["inter"], sel))[0]
    ev_of = {int(i): k for k, i in enumerate(sel)}
    ev = np.array([ev_of[int(i)] for i in tr["inter"][rows]], dtype=np.int64)
    n = len(rows)
    a, k, b, p = tr["v"][rows].copy(), tr["v"][rows].copy(), tr["b"][rows], tr["p"][rows]
    ap = p.copy()
    if muons is not None:                       # the scattered muon starts as its parent at z = 30.6 m
        beam = np.nonzero(tr["beam"][rows])[0]
        mrow = inter["muon"][tr["inter"][rows[beam]]]
        a_par = np.stack([np.asarray(muons[f"{c}_30"][mrow], float) for c in "xyz"], 1)
        k_par = np.stack([np.asarray(muons[f"k{c}"][mrow], float) for c in "xyz"], 1)
        bounced = np.isfinite(k_par[:, 2]) & (k_par[:, 2] < k[beam, 2])
        a[beam] = np.where(bounced[:, None], k_par, a_par)
        ap[beam] = np.stack([np.asarray(muons[f"p{c}_30"][mrow], float) for c in "xyz"], 1)
        # k stays at the interaction vertex: the path is A -> vertex -> B (its own scattering after the vertex)
        k[beam] = np.stack([inter["x"], inter["y"], inter["z"]], 1)[tr["inter"][rows[beam]]]
    trk = dict(ax=a[:, 0], ay=a[:, 1], az=a[:, 2], apx=ap[:, 0], apy=ap[:, 1], apz=ap[:, 2],
               kx=k[:, 0], ky=k[:, 1], kz=k[:, 2], bx=b[:, 0], by=b[:, 1], bz=b[:, 2],
               bpx=p[:, 0], bpy=p[:, 1], bpz=p[:, 2])
    zv = vertex[ev, 2]
    in_pair = np.isin(rows, pair.ravel())
    z_a = np.where(in_pair, zv, -np.inf)                  # level a: only the two vertex tracks search
    z_b = np.where(tr["acc"][rows], zv, -np.inf)          # level b: every track in the SST1 acceptance
    return trk, ev, z_a, z_b, rows


def parent_crossed(inter, sel, muons, design):
    """Truth (level c): the parent muon of each selected interaction crossed a plane upstream of the vertex."""
    mrow = inter["muon"][sel]
    tr = {f"{s}{c}": np.asarray(muons[f"{c}_30" if s == "a" else f"{c}_82"][mrow], float) for s in "ab" for c in "xyz"}
    tr.update({f"k{c}": np.asarray(muons[f"k{c}"][mrow], float) for c in "xyz"})
    tr["apz"] = np.asarray(muons["pz_30"][mrow], float)
    zint = inter["z"][sel]
    hit = np.zeros(len(sel), bool)
    for plane in design.planes:
        _, _, _, ok = reco.true_crossing(tr, plane)
        hit |= ok & (plane.z < zint + 1e-3)      # a plane at the vertex counts (interactions in the plane)
    return hit


def bootstrap(w, groups, n_boot=200, rng=None):
    """1-sigma error of sum(w), resampling the groups (input muons) with replacement; at least sum / sqrt(number of
    groups), so that a sum resting on one input muon gets 100 % (the bootstrap cannot vary a single group)."""
    if not len(w):
        return 0.0
    rng = rng or np.random.default_rng(0)
    g, inv = np.unique(groups, return_inverse=True)
    wg = np.bincount(inv, weights=w)
    sums = [wg[rng.integers(0, len(wg), len(wg))].sum() for _ in range(n_boot)]
    return float(max(np.std(sums), w.sum() / np.sqrt(len(wg))))


def evaluate(inter, part, design, maps, cfg, rng, muons=None, levels=("a", "b"), max_tracks="cfg"):
    """Selection and veto of one design's interactions. inter / part: the int_* / part_* arrays of fastveto
    (muons: the muon dataset the interactions were drawn from, for the parents; None for neutrinos).
    Returns per target: interactions, selected and surviving (a, b, c) per 15 years, errors, counts."""
    n_inter = len(inter["weight"])
    tr = charged_tracks(part)
    sel, pair, vertex = select(tr, n_inter, cfg=cfg)
    max_tracks = cfg.max_tracks if max_tracks == "cfg" else max_tracks
    if max_tracks is not None and len(sel):        # selection: at most max_tracks reconstructed tracks in the event
        n_acc = np.bincount(tr["inter"][tr["acc"]], minlength=n_inter)[sel]
        keep = n_acc <= max_tracks
        sel, pair, vertex = sel[keep], pair[keep], vertex[keep]
    w15 = inter["weight"] * SPILLS_15Y
    survive = {}
    if len(sel):
        trk, ev, z_a, z_b, _ = event_tracks(inter, tr, sel, pair, vertex, muons)
        for lev, z in (("a", z_a), ("b", z_b)):
            if lev in levels:
                r = reco.run_events(trk, ev, z, design, maps, cfg, rng)
                survive[lev] = ~r["vetoed"][:len(sel)]
        survive["c"] = ~parent_crossed(inter, sel, muons, design) if muons is not None else np.ones(len(sel), bool)
    groups = inter["evt_n"] if "evt_n" in inter else np.arange(n_inter)
    diag = {}
    if len(sel) and muons is not None:           # why level b misses: per selected event
        beam = np.nonzero(tr["beam"])[0]
        b_of = np.full(n_inter, -1); b_of[tr["inter"][beam]] = beam
        bi = b_of[sel]
        has = bi >= 0
        mrow = inter["muon"][sel]
        pin = np.stack([np.asarray(muons[f"p{c}_30"][mrow], float) for c in "xyz"], 1)
        pout = np.where(has[:, None], tr["p"][np.maximum(bi, 0)], np.nan)
        cosang = (pin * pout).sum(1) / np.linalg.norm(pin, axis=1) / np.linalg.norm(pout, axis=1)
        diag = dict(beam_reconstructed=has & tr["acc"][np.maximum(bi, 0)], beam_in_pair=has & np.isin(bi, pair.ravel()),
                    parent_tagged=~survive["c"], angle_mrad=1e3 * np.arccos(np.clip(cosang, -1, 1)),
                    parent_p=np.linalg.norm(pin, axis=1))
    out = {}
    for code in np.unique(inter["code"]):
        m_all = inter["code"] == code
        name = target_name(code, design)
        m_sel = inter["code"][sel] == code
        row = dict(interactions=float(w15[m_all].sum()), n_interactions=int(m_all.sum()),
                   selected=float(w15[sel][m_sel].sum()), n_selected=int(m_sel.sum()),
                   selected_err=bootstrap(w15[sel][m_sel], groups[sel][m_sel], rng=rng))
        for lev, s in survive.items():
            mm = m_sel & s
            row[f"survive_{lev}"] = float(w15[sel][mm].sum())
            row[f"survive_{lev}_err"] = bootstrap(w15[sel][mm], groups[sel][mm], rng=rng)
            row[f"n_survive_{lev}"] = int(mm.sum())
        if diag:
            ws = w15[sel][m_sel]
            frac = lambda k: float((ws * diag[k][m_sel]).sum() / max(ws.sum(), 1e-300))
            row["diag"] = dict(beam_reconstructed=frac("beam_reconstructed"), beam_in_pair=frac("beam_in_pair"),
                               parent_tagged=frac("parent_tagged"),
                               angle_mrad_median=float(np.nanmedian(diag["angle_mrad"][m_sel])) if m_sel.any() else None,
                               parent_p_median=float(np.median(diag["parent_p"][m_sel])) if m_sel.any() else None)
            if "b" in survive and m_sel.any():
                miss = m_sel & survive["b"] & diag["parent_tagged"]
                row["diag"]["b_missed_but_taggable_15y"] = float(w15[sel][miss].sum())
                row["diag"]["missed_angle_mrad_median"] = float(np.nanmedian(diag["angle_mrad"][miss])) if miss.any() else None
                row["diag"]["missed_beam_reconstructed"] = float((w15[sel] * diag["beam_reconstructed"])[miss].sum()
                                                                 / max(w15[sel][miss].sum(), 1e-300))
        out[name] = row
    return out


# ---------------------------------------------------------------- cached, for the objective
DESIGN_FREE_CODES = (0, 1, 2)   # helium, air, PVC: drawn from the muon dataset alone, the same for every design


def prepare(inter, part, cfg, muons=None, codes=DESIGN_FREE_CODES):
    """Everything before the veto, for the design-independent targets: the selected events (cfg.ip_max,
    cfg.max_tracks) with their tracks in reco's format. The planes' own interactions are left out: in every
    design studied they are either never selected (fins, outside the vessel) or always vetoed by the parent's
    hit in that very plane (UBT). Returns a flat dict of arrays (np.savez-able)."""
    keep = np.isin(inter["code"], codes)
    idx = np.nonzero(keep)[0]
    inter = {k: v[idx] for k, v in inter.items()}
    pmask = np.isin(part["int"], idx)
    remap = np.full(len(keep), -1); remap[idx] = np.arange(len(idx))
    part = {k: v[pmask] for k, v in part.items()}
    part["int"] = remap[part["int"]]
    n_inter = len(idx)
    tr = charged_tracks(part)
    sel, pair, vertex = select(tr, n_inter, cfg=cfg)
    if cfg.max_tracks is not None and len(sel):
        n_acc = np.bincount(tr["inter"][tr["acc"]], minlength=n_inter)[sel]
        sel, pair, vertex = sel[n_acc <= cfg.max_tracks], pair[n_acc <= cfg.max_tracks], vertex[n_acc <= cfg.max_tracks]
    out = dict(event_weight15=inter["weight"][sel] * SPILLS_15Y, event_code=inter["code"][sel],
               event_group=(inter["evt_n"] if "evt_n" in inter else np.arange(n_inter))[sel],
               event_zint=inter["z"][sel], event_muon=(inter["muon"][sel] if "muon" in inter else np.full(len(sel), -1)),
               interactions15={int(c): float(inter["weight"][inter["code"] == c].sum() * SPILLS_15Y) for c in codes})
    out["interactions15"] = np.array([out["interactions15"].get(c, 0.0) for c in codes])
    out["codes"] = np.array(codes)
    if len(sel):
        trk, ev, z_a, z_b, _ = event_tracks(inter, tr, sel, pair, vertex, muons)
        out.update({f"trk_{k}": v for k, v in trk.items()}, track_event=ev, track_z_a=z_a, track_z_b=z_b)
    return out


def veto(prep, design, maps, cfg, rng, level="b", muons=None):
    """Veto of the prepared events for one design: surviving events per 15 years per target, bootstrap errors
    over input muons, counts; and the truth bound (parent crossed a plane upstream) if muons are given."""
    w, code, grp = prep["event_weight15"], prep["event_code"], prep["event_group"]
    n = len(w)
    surv = np.zeros(0, bool)
    if n:
        trk = {k[4:]: prep[k] for k in prep if k.startswith("trk_")}
        z = prep["track_z_b" if level == "b" else "track_z_a"]
        surv = ~reco.run_events(trk, prep["track_event"], z, design, maps, cfg, rng)["vetoed"][:n]
    out = {}
    for i, c in enumerate(prep["codes"]):
        m = code == c
        out[CODE_NAMES[int(c)]] = dict(interactions=float(prep["interactions15"][i]), selected=float(w[m].sum()),
                                       n_selected=int(m.sum()), survive=float(w[m & surv].sum()),
                                       survive_err=bootstrap(w[m & surv], grp[m & surv], rng=rng),
                                       n_survive=int((m & surv).sum()))
    total = sum(r["survive"] for r in out.values())
    err = bootstrap(w[surv], grp[surv], rng=rng) if n else 0.0
    return dict(per_target=out, total=float(total), total_err=float(err), n_events=int(n), n_survive=int(surv.sum()))
