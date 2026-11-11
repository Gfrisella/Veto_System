#!/usr/bin/env python3
"""ISSUES D9 (decided 11 Nov 2026): combinatorial pairs of ONE track of a muon / neutrino interaction with an unrelated
muon of the same spill ("DIS x mu"), with the veto. Replaces the out-of-repo dis_comb_pairs*.py numbers (void: they
paired same-sign tracks, ISSUES D17).

DIS tracks: the good charged tracks at SST1 (interactions.charged_tracks: SST1 + timing acceptance, p > 1 GeV) of every
event of the given event stores (default: the MS7 muon chain and the vessel muon DIS on Nov-4), weight = event weight
per spill. Multi-vertex rejection (9 Nov): only events with exactly ONE good track are used; an event with >= 2 good
tracks forms its own vertex and, with the partner muon, has >= 3 tracks in the window (rejected). MS7 stores: the decay
products of V0 in the original interactions (code 3) are dropped, their forced copies (code 4) carry those decays.
Muons: the good SST1 muons of the design store (rock_combinatorial.muon_tracks, charge_from_pdg_column). Pairs:
opposite charge (charges on both sides), the pair cuts of make_reco_inputs (DOCA, vertex in the vessel, true IP
pre-cut), then rock_combinatorial.veto_term: time coincidence, measured IP < cut, veto of both legs.
Bound (option c, printed first): candidates x (1 - veto efficiency of the SST1 muons) = what survives if the DIS track
is never vetoed and the muon leg is vetoed like an average SST1 muon.

    python3 bin/dis_combinatorial.py --design D6_straw --out results/reco/dis_comb_D6_straw.json
"""
import _bootstrap  # noqa: F401

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from rock_combinatorial import KEYS, cross_pairs, muon_tracks, veto_term
from run_reco import load_inputs
from veto import design as dsg
from veto import evstore as es
from veto import interactions as ix
from veto import paths, reco
from veto.config import DEFAULT_CONFIG

V0 = (130, 310, 3122, 3322)
DEFAULT_STORES = ("ms7_muon_g4v112_nov4", "vessel_mu_nov4q_D6_straw_mod")


def drop_v0_products_of_originals(ev, tr):
    """MS7 stores: tracks descending from a V0 in code-3 (original) events are removed (the code-4 copies decay them)."""
    code = dict(zip(ev["event"].tolist(), ev["code"].tolist()))
    is3 = np.array([code.get(int(e), -1) == 3 for e in tr["event"]]) if len(tr["event"]) else np.zeros(0, bool)
    if not is3.any():
        return tr
    key = tr["event"].astype(np.int64) * 1_000_000 + tr["track"].astype(np.int64)
    pkey = tr["event"].astype(np.int64) * 1_000_000 + tr["parent"].astype(np.int64)
    o = np.argsort(key)
    pos = np.clip(np.searchsorted(key[o], pkey), 0, len(key) - 1)
    has_parent = (tr["parent"] >= 0) & (key[o][pos] == pkey)
    ppdg = np.where(has_parent, np.abs(tr["pdg"][o][pos]), 0)
    bad = np.zeros(len(key), bool)
    cur = is3 & np.isin(ppdg, V0)                      # direct V0 daughters, then their descendants
    while cur.any():
        bad |= cur
        bad_keys = np.sort(key[cur])
        i = np.clip(np.searchsorted(bad_keys, pkey), 0, len(bad_keys) - 1)
        cur = is3 & ~bad & has_parent & (bad_keys[i] == pkey)
    return {k: v[~bad] for k, v in tr.items()}


def dis_tracks(store):
    meta, ev, tr = es.read(paths.evstore(store))
    n0 = len(tr["event"])
    tr = drop_v0_products_of_originals(ev, tr)
    inter, part = es.to_fastveto(ev, tr)
    t = ix.charged_tracks(part)
    g = np.nonzero(t["good"])[0]
    n_good = np.bincount(t["inter"][g], minlength=len(inter["weight"]))
    one = g[n_good[t["inter"][g]] == 1]                 # D9: events with exactly one good track
    nan = np.full(len(one), np.nan)
    v, b, p = t["v"][one], t["b"][one], t["p"][one]
    out = dict(ax=v[:, 0], ay=v[:, 1], az=v[:, 2], apz=p[:, 2], kx=nan, ky=nan, kz=nan.copy(),
               bx=b[:, 0], by=b[:, 1], bz=b[:, 2], bpx=p[:, 0], bpy=p[:, 1], bpz=p[:, 2],
               weight=inter["weight"][t["inter"][one]], type=np.full(len(one), 3.0),
               pdg=-13.0 * t["q"][one], draw=t["inter"][one].astype(np.int64))   # -13 q: a PDG-like id, read as charge
    info = dict(store=str(paths.evstore(store)), tracks=n0, tracks_after_v0_drop=len(tr["event"]), good=len(g),
                events_with_good=int((n_good > 0).sum()), single_track_events=int((n_good == 1).sum()),
                multi_track_events=int((n_good >= 2).sum()), single_rate_per_spill=float(out["weight"].sum()))
    return out, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", required=True)
    ap.add_argument("--stores", nargs="+", default=list(DEFAULT_STORES), help="event stores (names under paths.EVSTORE_DIR or paths)")
    ap.add_argument("--dataset", default=paths.MUON_DATASET)
    ap.add_argument("--inputs", default=paths.RECO_INPUTS)
    ap.add_argument("--mu-sample", type=int, default=2_000_000, help="random SST1 muon rows (weights scaled); 0 = all")
    ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = Path(a.design)
    design = dsg.load(p if p.exists() else a.design)
    cfg = reco.RecoConfig(seed=1, multiple_scattering=True)
    tr_in, inp = load_inputs(a.inputs)
    mu = muon_tracks(a.dataset, a.mu_sample)
    print(f"good SST1 muons: {mu['weight'].sum():.4g} / spill ({len(mu['ax']):,} rows), {time.time() - t0:.0f} s", flush=True)
    # veto efficiency of an average SST1 muon (for the bound): the single-muon sample of the reco inputs
    single = inp["single"]; sw = inp["single_w"]
    m = reco.run_events({k: tr_in[k][single] for k in tr_in}, np.arange(len(single)), np.full(len(single), 1e9), design, inp, cfg,
                        np.random.default_rng(3))
    mu_ineff = float((sw * ~m["vetoed"]).sum() / sw.sum())
    res, infos = {}, {}
    for st in a.stores:
        B, info = dis_tracks(st)
        infos[st] = info
        print(f"{st}: {info['single_track_events']:,} single-track events ({info['single_rate_per_spill']:.4g} tracks / spill), "
              f"{info['multi_track_events']:,} multi-track events rejected (D9), {time.time() - t0:.0f} s", flush=True)
        ia, ib, v, n_pairs = cross_pairs(mu, B, dev, a.block)
        tr = {k: np.concatenate([np.asarray(mu[k], float), np.asarray(B[k], float)]) for k in KEYS}
        cand = np.stack([ia, ib + len(mu["ax"])], 1)
        w = mu["weight"][ia] * B["weight"][ib]
        r = veto_term(tr, cand, w, v, design, inp, cfg, np.random.default_rng(2))
        r.update(opposite_pairs=n_pairs, bound_15y=r["selected_15y"] * mu_ineff)
        res[st] = r
        print(f"  {n_pairs:.3g} opposite-charge pairs, {len(ia):,} candidates; selected {r['selected_15y']:.3g} / 15 y "
              f"(bound x mu inefficiency {mu_ineff:.2e}: {r['bound_15y']:.3g}), after veto {r.get('after_15y', 0):.3g} "
              f"({r.get('n_after', 0)} MC), {time.time() - t0:.0f} s", flush=True)
    print(f"\ndesign {design.name}: DIS track x muon, per 15 years (time window +-{DEFAULT_CONFIG['time_window'] * 1e12:.0f} ps)")
    print(f"  {'store':34s}{'candidates':>13s}{'IP < cut':>11s}{'n':>7s}{'bound':>10s}{'after veto':>12s}{'n':>6s}")
    for k, r in res.items():
        aft = f"{r['after_15y']:.3g}" if r.get("n_after") else f"< {r.get('after_upper_15y') or 0:.2g}"
        print(f"  {k:34s}{r['candidates_15y']:>13.4g}{r['selected_15y']:>11.4g}{r['n_selected']:>7d}{r['bound_15y']:>10.3g}"
              f"{aft:>12s}{r.get('n_after', 0):>6d}")
    json.dump(dict(design=design.name, dataset=a.dataset, inputs=a.inputs, mu_inefficiency=mu_ineff, stores=infos, terms=res),
              open(a.out, "w"), indent=1, default=float)
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
