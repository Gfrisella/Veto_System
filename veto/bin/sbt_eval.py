#!/usr/bin/env python3
"""ISSUES D19: selection and SBT tagging of event stores with the SBT as a material (lib/veto/sbt_material.py).

    python3 bin/sbt_eval.py --design designs/R2_sbt_ubt.json --store <dir> [<dir> ...] --ip-cut full|partial
        [--muons] [--signal 40000] [--codes 9101 9102 9103 9104] --out <json>

Per store (event stores of bin/evstore_build.py: the SBT-material interactions of bin/sbt_interactions.py, or any other,
e.g. vessel_nu2026_* for the neutral products of helium interactions, item 2) and target code, per 15 y:
  selected     the selection of interactions.evaluate (measured IP < 10 cm / 2.5 m, <= 2 tracks in SST1)
  std          after the design as it is (level b: every reconstructed track; planes, V0 rule, SBT pointing, strips)
  v0line       + the SBT V0 line (sbt cfg v0_line: the candidate traced back as a neutral, a fired cell where it leaves)
  v0line_self  + the event's own LS deposits (sbt_material.deposits: products, parent muon) as hits (selftag_hits)
  own_any      bound: some cell of the event fired by its own deposits (no association, no timing)
  zone         item 2 rule: an own fired cell downstream of z_vertex - ZONE_UP in time with the vertex (accidental
               cost = P(random hit in those cells within +-window), see signal)
with bootstrap errors over parent muons (or events). --muons: the interactions' parent muons (int muon = store parent_row,
rows of --dataset) start their paths at z = 30.6 m (their own hits upstream of the vertex: parent crossing the LS).
In a store without forced copies (no "_kl" in its name) the pairs from natural K_L decays are dropped: the forced-decay
store (bin/evstore_redecay.py --natural) carries them (evstore_redecay's double-counting note).
--signal N: signal loss (B4 EventCalc set, N per benchmark) of std / v0line, and the accidental cost of zone.
"""
import _bootstrap  # noqa: F401
import argparse
import json
import time
from pathlib import Path

import numpy as np

from run_reco import load_inputs
from veto import design as dsg
from veto import evstore as es
from veto import interactions as ix
from veto import io, paths, reco
from veto import sbt
from veto import sbt_material as sm
from veto import vertex_selection as vsel
from veto.config import SPILLS_15Y

ZONE_UP = 1.0          # m: zone rule cells from z_vertex - ZONE_UP downstream


def variant(design, **flags):
    raw = json.loads(json.dumps(design.raw))
    raw["sbt"] = dict(raw.get("sbt") or {}, **flags)
    return dsg.from_dict(raw)


def store_inputs(ev, tr):
    """inter / part as es.to_fastveto plus, per part row, the start time t0, the end point and the parent's pdg."""
    inter, part = es.to_fastveto(ev, tr)
    keep = (tr["end"] == es.END_SST1) | ((tr["end"] != es.END_DECAY) & (tr["end"] != es.END_FORCED))
    ev_ids = np.asarray(tr["event"])
    first = np.r_[0, np.nonzero(np.diff(ev_ids))[0] + 1]
    start = np.repeat(first, np.diff(np.r_[first, len(ev_ids)]))
    par = np.asarray(tr["parent"])
    ppdg = np.where(par >= 0, np.asarray(tr["pdg"])[np.maximum(start + par, 0)], 0)
    part["t0"] = np.asarray(tr["t0"], float)[keep]
    part["parent_pdg"] = ppdg[keep]
    inter["t"] = np.asarray(ev["t"], float)
    return inter, part


def event_deposits(sel, inter, tr_store, ev_index, vertex_t_arr, muons, rng):
    """Own LS deposits of the selected events: every particle of their store trees (lines up to their end point) and the
    parent muon from its z = 30.6 m state (or bounce point) to the vertex. Times relative to the pair's arrival at SST1."""
    rows = np.nonzero(np.isin(ev_index, sel))[0]
    pos = np.full(len(inter["weight"]), -1); pos[sel] = np.arange(len(sel))
    pid = pos[ev_index[rows]]
    v = np.stack([tr_store["x0"], tr_store["y0"], tr_store["z0"]], 1)[rows].astype(float)
    x1 = np.stack([tr_store["x1"], tr_store["y1"], tr_store["z1"]], 1)[rows].astype(float)
    p = np.stack([tr_store["px"], tr_store["py"], tr_store["pz"]], 1)[rows].astype(float)
    length = np.linalg.norm(x1 - v, axis=1)
    beam = np.asarray(tr_store["beam"], bool)[rows]
    pdg = np.asarray(tr_store["pdg"])[rows]
    t0 = np.asarray(tr_store["t0"], float)[rows]
    keep = ~beam                                          # the scattered muon: its own hits come from the reco tracks
    P = dict(pid=[pid[keep]], pdg=[pdg[keep]], v=[v[keep]], p=[p[keep]], t0=[t0[keep]], L=[length[keep]])
    if muons is not None and "muon" in inter:              # parent muon: A (or K) -> vertex
        mrow = inter["muon"][sel]
        a = np.stack([np.asarray(muons[f"{c}_30"][mrow], float) for c in "xyz"], 1)
        k = np.stack([np.asarray(muons[f"k{c}"][mrow], float) for c in "xyz"], 1)
        vtx = np.stack([inter["x"][sel], inter["y"][sel], inter["z"][sel]], 1)
        bounced = np.isfinite(k[:, 2]) & (k[:, 2] < vtx[:, 2])
        a = np.where(bounced[:, None], k, a)
        dv = vtx - a
        L = np.linalg.norm(dv, axis=1)
        pp = dv / np.maximum(L, 1e-9)[:, None] * inter["E"][sel][:, None]
        P["pid"].append(np.arange(len(sel))); P["pdg"].append(np.full(len(sel), 13)); P["v"].append(a)
        P["p"].append(pp); P["t0"].append(inter["t"][sel] - L / sbt.C); P["L"].append(L)
    P = {k_: np.concatenate(x) for k_, x in P.items()}
    d = sm.deposits(P["pid"], P["pdg"], P["v"], P["p"], P["t0"], rng, length=np.minimum(P["L"], 200.0))
    d["t"] = d["t"] - vertex_t_arr[d["pid"]]
    return d


def zone_rate(cfg, z_vertex, window_ns, spill_s=1.0):
    """Random-hit probability in the zone cells (ring centre > z_vertex - ZONE_UP) within +- window_ns."""
    ids, rate = sbt.cell_rates(cfg["threshold_mev"])
    face, ring = ids // 10 // 1000, ids // 10 % 1000
    rate = rate[sbt.instrumented(cfg, face, ring)] * cfg.get("rate_scale", 1.0)
    zc = 0.5 * (sbt.RING_EDGES[ring] + sbt.RING_EDGES[ring + 1])[sbt.instrumented(cfg, face, ring)]
    order = np.argsort(zc)
    cum = np.cumsum(rate[order][::-1])[::-1]               # rate of cells with zc >= zc_i
    j = np.searchsorted(zc[order], np.asarray(z_vertex) - ZONE_UP)
    r = np.where(j < len(cum), cum[np.minimum(j, len(cum) - 1)], 0.0)
    return 1 - np.exp(-r * 2 * window_ns * 1e-9 / spill_s)


def evaluate_store(path, design, maps, cfg, muons, rng, codes):
    meta, ev, tr = es.read(path)
    inter, part = store_inputs(ev, tr)
    if muons is not None:
        inter["muon"] = np.asarray(ev["parent_row"])
    inter["E"] = np.asarray(ev["E"], float)
    n = len(inter["weight"])
    tc = ix.charged_tracks(part)
    sel, pair, vertex = ix.select(tc, n, cfg=cfg)
    if len(sel):
        n_acc = np.bincount(tc["inter"][tc["acc"]], minlength=n)[sel]
        k = n_acc <= cfg.max_tracks
        sel, pair, vertex = sel[k], pair[k], vertex[k]
    kl_dropped = 0.0
    if len(sel) and "_kl" not in Path(path).name:          # natural K_L decay pairs: carried by the forced-decay store
        from_kl = (part["parent_pdg"][tc["row"][pair[:, 0]]] == 130) | (part["parent_pdg"][tc["row"][pair[:, 1]]] == 130)
        kl_dropped = float(inter["weight"][sel][from_kl].sum() * SPILLS_15Y)
        sel, pair, vertex = sel[~from_kl], pair[~from_kl], vertex[~from_kl]
    out = dict(meta=dict(source=meta.get("source"), n_events=meta.get("n_events")), kl_natural_dropped_15y=kl_dropped)
    # enrichment check: selected events whose final state has no strange hadron and no displaced root (should be none)
    if len(sel):
        root = np.asarray(tr["parent"]) < 0
        eo_ = np.argsort(ev["event"], kind="stable")
        evi = eo_[np.searchsorted(np.asarray(ev["event"])[eo_], tr["event"])]
        vx = np.stack([inter["x"], inter["y"], inter["z"]], 1)[evi]
        disp = np.linalg.norm(np.stack([tr["x0"], tr["y0"], tr["z0"]], 1) - vx, axis=1) > sm.DISPLACED_MIN
        fl = root & (np.isin(np.abs(tr["pdg"]), sm.STRANGE) | disp)
        flagged = np.bincount(evi[fl], minlength=n) > 0
        out["selected_unflagged_15y"] = float(inter["weight"][sel][~flagged[sel]].sum() * SPILLS_15Y)
        out["n_selected_unflagged"] = int((~flagged[sel]).sum())
    w15 = inter["weight"] * SPILLS_15Y
    groups = inter["evt_n"]
    res = {}
    surv = {}
    if len(sel):
        trk, evx, z_a, z_b, rows = ix.event_tracks(inter, tc, sel, pair, vertex, muons)
        pv = tc["p"][pair[:, 0]] + tc["p"][pair[:, 1]]
        # pair arrival time at SST1 (first vertex track): its start time + flight (beta = 1)
        r0 = tc["row"][pair[:, 0]]
        t_arr = part["t0"][r0] + np.linalg.norm(tc["b"][pair[:, 0]] - tc["v"][pair[:, 0]], axis=1) / sbt.C
        eo = np.argsort(ev["event"], kind="stable")
        ev_index = eo[np.searchsorted(np.asarray(ev["event"])[eo], tr["event"])]
        dep = event_deposits(sel, inter, tr, ev_index, t_arr, muons, np.random.default_rng(cfg.seed + 19))
        scfg = design.sbt or dict(sbt.DEFAULT)
        hits = sm.fired(dep, scfg, np.random.default_rng(cfg.seed + 20))
        variants = dict(std=design, v0line=variant(design, v0_line=True),
                        v0line_self=variant(design, v0_line=True, selftag_hits=True))
        v0 = dict(vtx=vertex, pvis=pv, sbt_hits=dict(event=hits["pid"], cell=hits["cell"], t=hits["t"]))
        for name, dv in variants.items():
            if dv.sbt is None:
                continue
            r = reco.run_events(trk, evx, z_b, dv, maps, cfg, np.random.default_rng(cfg.seed), v0=v0)
            surv[name] = ~r["vetoed"][:len(sel)]
        own_any = np.zeros(len(sel), bool); own_any[hits["pid"]] = True
        surv["own_any"] = surv.get("std", np.ones(len(sel), bool)) & ~own_any
        window = scfg["n_sigma"] * np.hypot(scfg["sigma_t"] * 1e9, cfg.sigma_t_track)
        # zone rule: own fired cell with ring centre > z_vertex - ZONE_UP, in time with the vertex (light from the vertex)
        ring = hits["cell"] // 10 % 1000
        zc = 0.5 * (sbt.RING_EDGES[ring] + sbt.RING_EDGES[ring + 1])
        t_vtx = inter["t"][sel] - t_arr                     # vertex time relative to the pair arrival
        dist = np.abs(zc - vertex[hits["pid"], 2])
        dt = hits["t"] - t_vtx[hits["pid"]] - dist / sbt.C
        z_ok = zc > vertex[hits["pid"], 2] - ZONE_UP
        zone = np.zeros(len(sel), bool); zone[hits["pid"][z_ok & (np.abs(dt) < window + 1.0)]] = True
        surv["zone"] = surv.get("std", np.ones(len(sel), bool)) & ~zone
        out["deposit_stats"] = dict(n_cells_fired=int(len(hits["pid"])), events_with_own_fired=int(own_any.sum()))
    for code in codes if codes else np.unique(inter["code"]):
        m_all = inter["code"] == code
        if not m_all.any():
            continue
        m = inter["code"][sel] == code
        ws, gs = w15[sel][m], groups[sel][m]
        row = dict(interactions=float(w15[m_all].sum()), n_interactions=int(m_all.sum()), selected=float(ws.sum()),
                   selected_err=ix.bootstrap(ws, gs, rng=rng), n_selected=int(m.sum()))
        for name, s_ in surv.items():
            mm = s_[m]
            row[name] = float(ws[mm].sum()); row[f"{name}_err"] = ix.bootstrap(ws[mm], gs[mm], rng=rng); row[f"n_{name}"] = int(mm.sum())
        res[ix.CODE_NAMES.get(int(code), str(code))] = row
    out["per_target"] = res
    return out


def signal_loss(design, maps, cfg, n_signal):
    """B4 signal loss of std / v0line (same random stream: the difference is the V0 line) and the zone rule's accidental cost."""
    from veto import shield_strips
    from veto import signal_eventcalc as se
    out = {}
    scfg = design.sbt or dict(sbt.DEFAULT)
    window = scfg["n_sigma"] * np.hypot(scfg["sigma_t"] * 1e9, cfg.sigma_t_track)
    for vname, dv in (("std", design), ("v0line", variant(design, v0_line=True))):
        fp_sum, w_sum, per = 0.0, 0.0, {}
        for name, wb in se.BENCHMARK_SET.items():
            sig, z_v, evs, w_ev, _ = se.selected_signal(name, cfg, np.random.default_rng(779), n_max=n_signal)
            if not len(w_ev):
                continue
            vtx, pv = shield_strips.candidates_from_tracks(sig, evs, z_v, len(w_ev))
            s = reco.run_events(sig, evs, z_v, dv, maps, cfg, np.random.default_rng(cfg.seed), with_own_hits=False,
                                v0=dict(vtx=vtx, pvis=pv))
            fp = float((w_ev * s["vetoed"]).sum() / w_ev.sum())
            per[name] = fp
            if vname == "std":
                pz = zone_rate(scfg, vtx[:, 2], window + 1.0)
                per[name + "_zone_accidental"] = float((w_ev * pz).sum() / w_ev.sum())
                pz4 = zone_rate(dict(scfg, rate_scale=4.0 * scfg.get("rate_scale", 1.0)), vtx[:, 2], window + 1.0)
                per[name + "_zone_accidental_x4"] = float((w_ev * pz4).sum() / w_ev.sum())
                # the V0 line alone on random hits (no other tagging): its own signal cost, free of random-stream noise
                lcfg = dict(scfg, v0_line=True)
                early = lcfg.get("v0_line_early_ns", 3.0)
                for sc, lab in ((1.0, ""), (4.0, "_x4")):
                    c_ = dict(lcfg, rate_scale=sc * scfg.get("rate_scale", 1.0))
                    acc = sbt.random_hits(len(w_ev), c_, cfg.spill_s, np.random.default_rng(cfg.seed + 7), window + 2.0 + early)
                    lt = sbt.v0_line_tag(acc["event"].astype(np.int64), acc["cell"].astype(np.int64), acc["t"],
                                         dict(vtx=vtx, pvis=pv), c_, len(w_ev), window, early)
                    per[name + "_v0line_alone" + lab] = float((w_ev * lt).sum() / w_ev.sum())
            fp_sum, w_sum = fp_sum + wb * fp, w_sum + wb
        out[vname] = dict(FP_frac=fp_sum / max(w_sum, 1e-300), per_benchmark=per)
    zs = [v for k, v in out["std"]["per_benchmark"].items() if k.endswith("_zone_accidental")]  # equal-weight mean
    out["zone_accidental_mean"] = float(np.mean(zs)) if zs else None
    for key in ("_zone_accidental_x4", "_v0line_alone", "_v0line_alone_x4"):
        v = [x for k, x in out["std"]["per_benchmark"].items() if k.endswith(key)]
        out[key.strip("_") + "_mean"] = float(np.mean(v)) if v else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", required=True); ap.add_argument("--store", nargs="*", default=[])
    ap.add_argument("--ip-cut", choices=("full", "partial"), default="full")
    ap.add_argument("--muons", action="store_true", help="the stores' parent_row are rows of --dataset (muon parents)")
    ap.add_argument("--dataset", default=paths.MUON_DATASET)
    ap.add_argument("--codes", type=int, nargs="*", default=None)
    ap.add_argument("--signal", type=int, default=0, help="signal decays per benchmark for the signal loss (0: none)")
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sm.register()
    t0 = time.time()
    p = Path(a.design)
    design = dsg.load(p if p.exists() else a.design)
    _, maps = load_inputs()
    cfg = reco.RecoConfig(seed=a.seed, multiple_scattering=True, ip_max=vsel.IP_CUTS[a.ip_cut])
    muons = io.open_store(paths.store_dir(a.dataset))[0] if a.muons else None
    res = dict(design=design.name, ip_cut=a.ip_cut, zone_up_m=ZONE_UP, params=dict(
        neutral_lambda=sm.NEUTRAL_LAMBDA, neutral_vis=sm.NEUTRAL_VIS, dedx_ls=sm.DEDX_LS), stores={})
    rng = np.random.default_rng(a.seed)
    for s in a.store:
        t1 = time.time()
        r = evaluate_store(str(paths.evstore(s)), design, maps, cfg, muons, rng, a.codes)
        res["stores"][s] = r
        print(f"\n{s} ({time.time() - t1:.0f} s), K_L natural pairs dropped {r['kl_natural_dropped_15y']:.3g} / 15 y")
        for name, v in r["per_target"].items():
            print(f"  {name:<14} int {v['interactions']:.3g}  sel {v['selected']:.3g} +- {v['selected_err']:.2g} ({v['n_selected']})  "
                  + "  ".join(f"{k} {v[k]:.3g}" for k in ("std", "v0line", "v0line_self", "own_any", "zone") if k in v), flush=True)
    if a.signal:
        res["signal"] = signal_loss(design, maps, cfg, a.signal)
        print("signal loss:", json.dumps(res["signal"], indent=None, default=float))
    res["seconds"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
