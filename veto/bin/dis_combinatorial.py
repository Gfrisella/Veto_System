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

    python3 bin/dis_combinatorial.py --design designs/R2_sbt_ubt.json --out results/reco/dis_comb_R2_sbt_ubt.json
"""
import _bootstrap  # noqa: F401

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from rock_combinatorial import KEYS, cross_pairs, muon_tracks, veto_term
from run_reco import load_inputs, shell_loss_populations
from veto import design as dsg
from veto import evstore as es
from veto import interactions as ix
from veto import paths, reco
from veto import geometry as geo
from veto.config import DEFAULT_CONFIG

V0 = (130, 310, 3122, 3322)
DEFAULT_STORES = ("ms7_muon_v0s", "vessel_mu_R2_sbt_ubt")
Z_PRETAG = geo.Z_VESSEL_MIN + 0.05   # m: vertex selection, at least 5 cm inside the vessel
COMP_PMIN = 0.1   # GeV: other products of the interaction kept as companions (hits in coincidence)


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


CHARGED = (11, 13, 211, 321, 2212, 3112, 3222, 3312, 3334)


def n_charged_sst1(ev, tr):
    """Charged tracks reaching SST1 per event (any momentum, any acceptance; the scattered muon included)."""
    eo = np.argsort(ev["event"], kind="stable")
    evi = eo[np.searchsorted(np.asarray(ev["event"])[eo], tr["event"])]
    at = np.isin(np.abs(np.asarray(tr["pdg"])), CHARGED) & (np.asarray(tr["end"]) == es.END_SST1)
    return np.bincount(evi[at], minlength=len(ev["event"]))


def dis_tracks(store, sst1_one=False):
    meta, ev, tr = es.read(paths.evstore(store))
    n0 = len(tr["event"])
    tr = drop_v0_products_of_originals(ev, tr)
    inter, part = es.to_fastveto(ev, tr)
    t = ix.charged_tracks(part)
    g = np.nonzero(t["good"])[0]
    n_good = np.bincount(t["inter"][g], minlength=len(inter["weight"]))
    one = g[n_good[t["inter"][g]] == 1]                 # D9: events with exactly one good track
    if sst1_one:                                        # 23 Nov: and no other charged track at SST1 (any p / acceptance)
        one = one[n_charged_sst1(ev, tr)[t["inter"][one]] == 1]
    nan = np.full(len(one), np.nan)
    v, b, p = t["v"][one], t["b"][one], t["p"][one]
    out = dict(ax=v[:, 0], ay=v[:, 1], az=v[:, 2], apz=p[:, 2], kx=nan, ky=nan, kz=nan.copy(),
               bx=b[:, 0], by=b[:, 1], bz=b[:, 2], bpx=p[:, 0], bpy=p[:, 1], bpz=p[:, 2],
               weight=inter["weight"][t["inter"][one]], type=np.full(len(one), 3.0),
               pdg=-13.0 * t["q"][one], draw=t["inter"][one].astype(np.int64))   # -13 q: a PDG-like id, read as charge
    # 20 Nov 2026: the rest of each single-track interaction (charged, moving downstream, p > COMP_PMIN; straight lines like the
    # track itself): its hits are in coincidence with the candidate (reco.run_events companions), sorted by interaction (draw)
    single_int = np.zeros(len(inter["weight"]), bool); single_int[t["inter"][one]] = True
    is_one = np.zeros(len(t["inter"]), bool); is_one[one] = True
    c = np.nonzero(single_int[t["inter"]] & ~is_one & (np.linalg.norm(t["p"], axis=1) > COMP_PMIN))[0]
    c = c[np.argsort(t["inter"][c], kind="stable")]
    cn = np.full(len(c), np.nan)
    cv, cb, cp = t["v"][c], t["b"][c], t["p"][c]
    comp = dict(draw=t["inter"][c].astype(np.int64),
                tr=dict(ax=cv[:, 0], ay=cv[:, 1], az=cv[:, 2], apz=cp[:, 2], kx=cn, ky=cn, kz=cn.copy(),
                        bx=cb[:, 0], by=cb[:, 1], bz=cb[:, 2], bpx=cp[:, 0], bpy=cp[:, 1], bpz=cp[:, 2],
                        weight=inter["weight"][t["inter"][c]], type=np.full(len(c), 3.0), pdg=-13.0 * t["q"][c]))
    info = dict(store=str(paths.evstore(store)), tracks=n0, companions=int(len(c)), tracks_after_v0_drop=len(tr["event"]), good=len(g),
                events_with_good=int((n_good > 0).sum()), single_track_events=int(len(np.unique(t["inter"][one]))),
                single_good_events_any_sst1=int((n_good == 1).sum()), sst1_one=bool(sst1_one),
                multi_track_events=int((n_good >= 2).sum()), single_rate_per_spill=float(out["weight"].sum()))
    return out, info, comp


def companions_of(comp, draws, event):
    """Companion tracks of the interactions `draws` (one per track row; -1 = none) as reco.run_events companions."""
    if comp is None or not len(comp["draw"]):
        return None
    cd = comp["draw"]
    dr = np.asarray(draws, np.int64)
    lo, hi = np.searchsorted(cd, dr, "left"), np.searchsorted(cd, dr, "right")
    n_c = np.where(dr >= 0, hi - lo, 0)
    if not n_c.sum():
        return None
    ci = np.repeat(lo - np.cumsum(n_c) + n_c, n_c) + np.arange(n_c.sum())
    return dict(tr={k: v[ci] for k, v in comp["tr"].items()}, event=np.repeat(np.asarray(event), n_c))


def pretag(trk, design, inp, cfg, rng, comp=None, chunk=200_000):
    """20 Nov 2026 (Guglielmo: tag each leg once, pair only the ones not tagged): every track traced back from SST1 alone,
    searching the planes / SBT upstream of Z_PRETAG (every selected vertex is downstream of it): a leg tagged here is
    tagged in every pair it could make. comp: the rest of its interaction (companions). Returns the tagged mask."""
    n = len(trk["ax"])
    tagged = np.zeros(n, bool)
    for c0 in range(0, n, chunk):
        rows = np.arange(c0, min(c0 + chunk, n))
        ev = np.arange(len(rows))
        sub_ = {k: np.asarray(v)[rows] for k, v in trk.items()}
        cmp_ = companions_of(comp, sub_["draw"], ev) if comp is not None else None
        m = reco.run_events(sub_, ev, np.full(len(rows), Z_PRETAG), design, inp, cfg, rng, companions=cmp_)
        tagged[rows] = m["vetoed"]
    return tagged


def merge_terms(parts):
    """Sum the veto_term results of blocks of DIS tracks (pair weights w_mu x w_DIS: blocks add up)."""
    if len(parts) == 1:
        return parts[0]
    out = {k: sum(p_.get(k, 0) for p_ in parts) for k in ("n_candidates", "candidates_15y", "selected_15y", "n_selected",
                                                          "after_15y", "n_after", "opposite_pairs")}
    out["after_err_15y"] = float(np.sqrt(sum(p_.get("after_err_15y", 0.0) ** 2 for p_ in parts)))
    ups = [p_.get("after_upper_15y") for p_ in parts if p_.get("after_upper_15y") is not None]
    out["after_upper_15y"] = (max(ups) if ups else None) if out["n_after"] == 0 else None
    sel = [p_ for p_ in parts if p_.get("leg_explained") and p_["selected_15y"] > 0]
    if sel:
        tot = sum(p_["selected_15y"] for p_ in sel)
        out["leg_explained"] = [sum(p_["selected_15y"] * p_["leg_explained"][i] for p_ in sel) / tot for i in (0, 1)]
    out["blocks"] = len(parts)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", required=True)
    ap.add_argument("--stores", nargs="+", default=list(DEFAULT_STORES), help="event stores (names under paths.EVSTORE_DIR or paths)")
    ap.add_argument("--dataset", default=paths.MUON_DATASET)
    ap.add_argument("--inputs", default=paths.RECO_INPUTS)
    ap.add_argument("--mu-sample", type=int, default=2_000_000, help="random SST1 muon rows (weights scaled); 0 = all")
    ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--dis-block", type=int, default=0,
                    help="process the DIS tracks in blocks of this many rows (memory: the vessel store needed > 128 GB in one go, S7 16 Nov 2026); 0 = all at once")
    ap.add_argument("--out", required=True)
    ap.add_argument("--random-hits", action="store_true", help="also tag with random hits (default off: background tagged only by its own hits, Guglielmo 20 Nov 2026; random hits count only for the signal loss)")
    ap.add_argument("--no-companions", action="store_true", help="do not use the other products of the interaction (pre-20 Nov behaviour)")
    ap.add_argument("--pretag", action="store_true", help="tag each leg alone first and pair only the untagged ones (fast, same expectation, but the detector inefficiency is drawn once per simulated leg: large variance, 20 Nov 2026 test 6 vs 154 MC pairs)")
    ap.add_argument("--sst1-one", action="store_true", help="single-track events: the good track must be the only charged track at SST1 (23 Nov 2026)")
    ap.add_argument("--no-v0", action="store_true", help="do not pass the pair vertex to the V0 rule (pre-20 Nov behaviour)")
    a = ap.parse_args()
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = Path(a.design)
    design = dsg.load(p if p.exists() else a.design)
    cfg = reco.RecoConfig(seed=1, multiple_scattering=True, random_hits=a.random_hits)
    tr_in, inp = load_inputs(a.inputs)
    mu = muon_tracks(a.dataset, a.mu_sample)
    scfg = getattr(design, "sbt", None) or {}
    n_stopped = 0
    if scfg.get("shell_energy_loss"):   # 20 Nov 2026: muons that stop in the SBT shell never reach SST1 (as run_reco)
        from veto import sbt_material
        st = sbt_material.shell_energy_loss(mu, scfg.get("shell_eloss_beta", True))["stopped"]
        n_stopped = float(mu["weight"][st].sum())
        mu = {k: v[~st] for k, v in mu.items()}
        tr_in, inp, _ = shell_loss_populations(tr_in, inp, scfg)
        print(f"SBT shell: {n_stopped:.4g} muons / spill stop in the shell, removed from the pairs and the inefficiency", flush=True)
    pre_info = {}
    if a.pretag:                 # tag each muon once; only the untagged ones can make an untagged pair
        tg = pretag({k: mu[k] for k in list(KEYS) + ["draw"]}, design, inp, cfg, np.random.default_rng(11))
        pre_info["mu_tagged_w_frac"] = float(mu["weight"][tg].sum() / mu["weight"].sum())
        pre_info["mu_rows"], pre_info["mu_rows_left"] = int(len(tg)), int((~tg).sum())
        mu = {k: v[~tg] for k, v in mu.items()}
        print(f"muons tagged alone (upstream of {Z_PRETAG:.2f} m): {pre_info['mu_tagged_w_frac']:.4f} of the weight; "
              f"{pre_info['mu_rows_left']:,} of {pre_info['mu_rows']:,} rows left to pair", flush=True)
    print(f"good SST1 muons: {mu['weight'].sum():.4g} / spill ({len(mu['ax']):,} rows), {time.time() - t0:.0f} s", flush=True)
    # veto efficiency of an average SST1 muon (for the bound): the single-muon sample of the reco inputs
    single = inp["single"]; sw = inp["single_w"]
    m = reco.run_events({k: tr_in[k][single] for k in tr_in}, np.arange(len(single)), np.full(len(single), 1e9), design, inp, cfg,
                        np.random.default_rng(3))
    mu_ineff = float((sw * ~m["vetoed"]).sum() / sw.sum())
    res, infos = {}, {}
    for st in a.stores:
        B, info, comp = dis_tracks(st, sst1_one=a.sst1_one)
        if a.no_companions:
            comp = None
        if a.pretag:             # tag each single-track interaction once (its track + the rest of the interaction)
            tg = pretag({k: B[k] for k in list(KEYS) + ["draw"]}, design, inp, cfg, np.random.default_rng(13), comp=comp)
            info["pretag_tagged_w_frac"] = float(B["weight"][tg].sum() / max(B["weight"].sum(), 1e-300))
            info["pretag_rows_left"] = int((~tg).sum())
            B = {k: v[~tg] for k, v in B.items()}
            print(f"  DIS tracks tagged alone: {info['pretag_tagged_w_frac']:.4f} of the weight; {info['pretag_rows_left']:,} left to pair",
                  flush=True)
        infos[st] = info
        print(f"{st}: {info['single_track_events']:,} single-track events ({info['single_rate_per_spill']:.4g} tracks / spill), "
              f"{info['multi_track_events']:,} multi-track events rejected (D9), {time.time() - t0:.0f} s", flush=True)
        nb = len(B["ax"])
        step = a.dis_block if a.dis_block > 0 else nb
        parts = []
        for k0 in range(0, nb, step):                   # blocks of DIS tracks: pairs, IP cut and veto, then free the memory
            Bk = {k: v_[k0:k0 + step] for k, v_ in B.items()}
            ia, ib, v, n_pairs, pre = cross_pairs(mu, Bk, dev, a.block, sel=cfg)      # streaming IP cut (memory)
            tr = {k: np.concatenate([np.asarray(mu[k], float), np.asarray(Bk[k], float)]) for k in KEYS}
            tr["draw"] = np.concatenate([np.full(len(mu["ax"]), -1, np.int64), np.asarray(Bk["draw"], np.int64)])
            cand = np.stack([ia, ib + len(mu["ax"])], 1)
            w = mu["weight"][ia] * Bk["weight"][ib]
            rk = veto_term(tr, cand, w, v, design, inp, cfg, np.random.default_rng(2 + k0 // step), pre=pre,
                           companions=comp, with_v0=not a.no_v0)
            rk.pop("_surv", None)
            rk["opposite_pairs"] = n_pairs
            parts.append(rk)
            del tr, cand, w, ia, ib, v, pre, Bk
            if step < nb:
                print(f"    DIS rows {k0:,}-{min(k0 + step, nb):,}: selected {rk['selected_15y']:.3g}, after veto "
                      f"{rk.get('after_15y', 0):.3g} ({rk.get('n_after', 0)} MC), {time.time() - t0:.0f} s", flush=True)
        r = merge_terms(parts)
        n_pairs = r["opposite_pairs"]
        r.update(bound_15y=r["selected_15y"] * mu_ineff)
        res[st] = r
        print(f"  {n_pairs:.3g} opposite-charge pairs, {r['n_candidates']:,} candidates; selected {r['selected_15y']:.3g} / 15 y "
              f"(bound x mu inefficiency {mu_ineff:.2e}: {r['bound_15y']:.3g}), after veto {r.get('after_15y', 0):.3g} "
              f"({r.get('n_after', 0)} MC), {time.time() - t0:.0f} s", flush=True)
    print(f"\ndesign {design.name}: DIS track x muon, per 15 years (time window +-{DEFAULT_CONFIG['time_window'] * 1e12:.0f} ps)")
    print(f"  {'store':34s}{'candidates':>13s}{'IP < cut':>11s}{'n':>7s}{'bound':>10s}{'after veto':>12s}{'n':>6s}")
    for k, r in res.items():
        aft = f"{r['after_15y']:.3g}" if r.get("n_after") else f"< {r.get('after_upper_15y') or 0:.2g}"
        print(f"  {k:34s}{r['candidates_15y']:>13.4g}{r['selected_15y']:>11.4g}{r['n_selected']:>7d}{r['bound_15y']:>10.3g}"
              f"{aft:>12s}{r.get('n_after', 0):>6d}")
    json.dump(dict(design=design.name, dataset=a.dataset, inputs=a.inputs, mu_inefficiency=mu_ineff, stores=infos, terms=res,
                   options=dict(random_hits=a.random_hits, companions=not a.no_companions, v0=not a.no_v0,
                                companion_pmin=COMP_PMIN, mu_stopped_per_spill=n_stopped, pretag=a.pretag,
                                z_pretag=Z_PRETAG, **pre_info)),
              open(a.out, "w"), indent=1, default=float)
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
