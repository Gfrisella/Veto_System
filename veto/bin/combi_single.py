#!/usr/bin/env python3
"""Combinatorial background "single-track DIS event x independent muon" (and DIS x DIS) before tagging, per 15 years,
for the three selection levels of the mu x mu combinatorial: pair cuts (opposite charge, DOCA < 2 cm, vertex 5 cm
inside the vessel... as benchmark_designs._pair_cut, with the true-IP pre-cut < 3 m), measured IP < 2.5 m, measured
IP < 10 cm. Time coincidence: rate = sum over pairs of w_a w_b x 2 dt / T_spill x SPILLS_15Y (dt = +-200 ps,
DEFAULT_CONFIG time_window), the factor of rock_combinatorial.veto_term (mu x mu 35.8 / 34.4 / 2.36).

DIS side (dis_combinatorial.dis_tracks recipe, ISSUES D9): events of an event store with EXACTLY ONE good track
(SST1 + timing acceptance, p > 1 GeV); events with >= 2 good tracks are their own vertex and are not used. MS7 stores:
V0 products of the code-3 originals dropped, the code-4 forced-decay copies carry them. Each track is flagged
"scattered muon" when it is the interacting muon itself (vessel stores: the beam flag; MS7 stores, beam flag not
stored: a mu+- leaving the MS7 exit plane (tree parent -1) with the charge of the parent muon), so that the numbers are
given with and without these events (the scattered muon is already a muon of the flux, counted in mu x mu).

Muon side: rock_combinatorial.muon_tracks (good SST1 muons of the Nov-4 dataset, random subsample, weights scaled).
Statistical errors: the pair sum is a two-sample sum, so the variance is estimated as the sum in quadrature of the
DIS-event-clustered term sum_e (sum of pair weights of event e)^2 and the muon-clustered term sum_m (...)^2.

    python3 bin/combi_single.py --stores vessel_mu=$R/evstore/vessel_mu_R2_sbt_ubt:mu ms7_mu=$R/evstore/ms7_muon_v0s:mu \\
        --out $R/combi_single/raw_vessel_mu.json [--self]  [--closure]  [--veto-design designs/R2_sbt_ubt.json]
"""
import _bootstrap  # noqa: F401

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from benchmark_designs import _pair_cut
from dis_combinatorial import drop_v0_products_of_originals, n_charged_sst1
from rock_combinatorial import KEYS, muon_tracks, veto_term
from veto import evstore as es
from veto import interactions as ix
from veto import paths, reco
from veto import vertex_selection as vsel
from veto.config import DEFAULT_CONFIG, SPILLS_15Y

FAC = 2 * DEFAULT_CONFIG["time_window"] / 1.0 * SPILLS_15Y      # 2 dt / T_spill (1 s, RecoConfig.spill_s) x spills
LEVELS = ("cand", "ip25", "ip01")


def single_tracks(store, kind, sst1_one=False):
    """Good tracks of the single-good-track events of a store, with flags (scattered muon, code, draw id).
    sst1_one: also require that the good track is the ONLY charged track reaching SST1 in the event (23 Nov definition:
    events with >= 2 tracks at the SSTs belong to the vertex category, whatever their momentum / acceptance)."""
    meta, ev, tr = es.read(paths.evstore(store))
    tr = drop_v0_products_of_originals(ev, tr)
    keep = (tr["end"] == es.END_SST1) | ((tr["end"] != es.END_DECAY) & (tr["end"] != es.END_FORCED))
    tr_rows = np.nonzero(keep)[0]                                  # part row -> tracks row (as es.to_fastveto)
    inter, part = es.to_fastveto(ev, tr)
    t = ix.charged_tracks(part)
    g = np.nonzero(t["good"])[0]
    n_good = np.bincount(t["inter"][g], minlength=len(inter["weight"]))
    one = g[n_good[t["inter"][g]] == 1]
    n_sst1 = None
    if sst1_one:
        n_sst1 = n_charged_sst1(ev, tr)
        one = one[n_sst1[t["inter"][one]] == 1]
    trow = tr_rows[t["row"][one]]
    e = t["inter"][one]
    pdg = np.asarray(tr["pdg"])[trow]
    code = np.asarray(ev["code"])[e]
    copy_of = np.asarray(ev["copy_of"])[e]
    draw = np.where(copy_of >= 0, copy_of, np.asarray(ev["event"])[e]).astype(np.int64)   # copies = their original
    scat = np.zeros(len(one), bool)
    rule = "none (neutrino store)"
    if kind == "mu":
        if np.asarray(tr["beam"]).any():
            scat = np.asarray(tr["beam"], bool)[trow]
            rule = "beam flag"
        else:                                                      # MS7 stores: parent muon charge from the parent table
            pr = np.asarray(ev["parent_row"]).astype(np.int64)
            if (pr[e] < 0).any():                                  # copies without a parent row: take the original's
                idx = {int(x): i for i, x in enumerate(ev["event"])}
                po = np.array([pr[idx[int(d)]] for d in draw])
                pr_e = np.where(pr[e] >= 0, pr[e], po)
            else:
                pr_e = pr[e]
            qpar = np.load(paths.ms7_parents(), mmap_mode="r")["q"]
            qp = np.asarray(qpar)[pr_e].astype(int)
            scat = (np.abs(pdg) == 13) & (np.asarray(tr["parent"])[trow] < 0) & (np.sign(-pdg) == qp)
            rule = "mu+- from the MS7 exit plane (parent -1) with the parent-muon charge"
    v, b, p = t["v"][one], t["b"][one], t["p"][one]
    nan = np.full(len(one), np.nan)
    out = dict(ax=v[:, 0], ay=v[:, 1], az=v[:, 2], apz=p[:, 2], kx=nan, ky=nan, kz=nan.copy(),
               bx=b[:, 0], by=b[:, 1], bz=b[:, 2], bpx=p[:, 0], bpy=p[:, 1], bpz=p[:, 2],
               weight=inter["weight"][e], type=np.full(len(one), 3.0), pdg=-13.0 * t["q"][one],
               draw=draw, scat=scat, code=code, ev=e.astype(np.int64), q=t["q"][one].astype(float), true_pdg=pdg)
    info = dict(store=str(paths.evstore(store)), events=len(ev["event"]), events_with_good=int((n_good > 0).sum()),
                single_track_events=int(len(np.unique(e))), multi_track_events=int((n_good >= 2).sum()),
                single_good_events_any_sst1=int((n_good == 1).sum()), sst1_one=bool(sst1_one),
                single_rate_per_spill=float(out["weight"].sum()),
                single_rate_per_spill_noscat=float(out["weight"][~scat].sum()),
                single_events_noscat=int((~scat).sum()), scattered_muon_rule=rule,
                single_rate_by_code={str(int(c)): float(out["weight"][code == c].sum()) for c in np.unique(code)},
                single_rate_by_code_noscat={str(int(c)): float(out["weight"][(code == c) & ~scat].sum()) for c in np.unique(code)})
    return out, info


class Acc:
    """Per-row sums of pair weights (x FAC) for every level: rows of side A and of side B (clustered errors)."""
    def __init__(self, nA, nB):
        self.A = {k: np.zeros(nA) for k in LEVELS}
        self.B = {k: np.zeros(nB) for k in LEVELS}
        self.nA = {k: np.zeros(nA, np.int64) for k in LEVELS}
        self.nB = {k: np.zeros(nB, np.int64) for k in LEVELS}

    def add(self, lev, a, b, w):
        self.A[lev] += np.bincount(a, w, len(self.A[lev])); self.B[lev] += np.bincount(b, w, len(self.B[lev]))
        self.nA[lev] += np.bincount(a, minlength=len(self.A[lev])); self.nB[lev] += np.bincount(b, minlength=len(self.B[lev]))


def pair_loop(A, B, cfg, block, same=False, keep01=False, log=""):
    """All opposite-charge pairs (a in A, b in B; same=True: A is B, a < b, different draws) through the pair cuts and
    the measured IP. Returns Acc, number of opposite-charge pairs, and the IP < 10 cm candidates if keep01."""
    tens = lambda d: (torch.as_tensor(np.stack([d["bx"], d["by"], d["bz"]], 1)),
                      torch.as_tensor(np.stack([d["bpx"], d["bpy"], d["bpz"]], 1)))
    rA, pA = tens(A)
    nA, nB = len(rA), len(B["bx"])
    if same:
        r, p = rA, pA
        off = 0
    else:
        rB, pB = tens(B)
        r, p = torch.cat([rA, rB]), torch.cat([pA, pB])
        off = nA
    qA, qB = np.sign(np.asarray(A["q"])), np.sign(np.asarray(B["q"]))
    dA, dB = torch.as_tensor(np.asarray(A["draw"])), torch.as_tensor(np.asarray(B["draw"]))
    wA, wB = np.asarray(A["weight"], float), np.asarray(B["weight"], float)
    bA = np.stack([A["bx"], A["by"], A["bz"]], 1); pAn = np.stack([A["bpx"], A["bpy"], A["bpz"]], 1)
    bB = np.stack([B["bx"], B["by"], B["bz"]], 1); pBn = np.stack([B["bpx"], B["bpy"], B["bpz"]], 1)
    acc = Acc(nA, nB)
    rng = np.random.default_rng(777)
    keep = []
    n_pairs = 0
    t0 = time.time()
    combos = ((np.nonzero(qA > 0)[0], np.nonzero(qB < 0)[0]), (np.nonzero(qA < 0)[0], np.nonzero(qB > 0)[0]))
    tot = sum(len(x) * len(y) for x, y in combos)
    for ia_all, jb_all in combos:
        ia_all, jb_all = torch.as_tensor(ia_all), torch.as_tensor(jb_all)
        for i0 in range(0, len(ia_all), block):
            ii = ia_all[i0:i0 + block]
            for j0 in range(0, len(jb_all), block):
                jj = jb_all[j0:j0 + block]
                ia = ii[:, None].expand(len(ii), len(jj)).reshape(-1)
                jb = jj[None, :].expand(len(ii), len(jj)).reshape(-1)
                if same:                                           # each unordered pair once, different events
                    m = (dA[ia] != dB[jb])
                    ia, jb = ia[m], jb[m]
                n_pairs += len(ia)
                a, bb, v = _pair_cut(ia, jb + off, r, p)
                if not len(a):
                    continue
                a, bb, v = a.numpy(), (bb - off).numpy(), v.numpy()
                w = wA[a] * wB[bb] * FAC
                acc.add("cand", a, bb, w)
                ip = vsel.measured_ip(bA[a], pAn[a], bB[bb], pBn[bb], cfg, rng)
                for lev, cut in (("ip25", 2.5), ("ip01", 0.1)):
                    k = ip < cut
                    if k.any():
                        acc.add(lev, a[k], bb[k], w[k])
                if keep01:
                    k = ip < 0.1
                    keep.append((a[k], bb[k], v[k], w[k]))
            if log and (i0 // block) % 8 == 0:
                print(f"    {log}: {n_pairs:.3g} pairs ({n_pairs / tot:.1%}), cand {acc.A['cand'].sum():.4g}, "
                      f"ip01 {acc.A['ip01'].sum():.4g} / 15 y, {time.time() - t0:.0f} s", flush=True)
    if same:                                                       # unordered pairs counted twice (a > b and a < b)
        for d in (acc.A, acc.B):
            for k in d:
                d[k] *= 0.5
        n_pairs //= 2
    cand01 = None
    if keep01:
        cand01 = tuple(np.concatenate([x[i] for x in keep]) if keep else np.zeros(0) for i in range(4))
    return acc, n_pairs, cand01


def summarise(acc, maskB, maskA=None, same=False):
    if same:                                                       # A is B: a row's pairs are split over A and B
        out = {}
        for lev in LEVELS:
            s = acc.A[lev] + acc.B[lev]
            out[lev] = dict(per_15y=float(acc.B[lev].sum()), err=float(np.sqrt((s ** 2).sum())),
                            n_pairs_mc=int(acc.nB[lev].sum()), n_dis_events_mc=int(((acc.nA[lev] + acc.nB[lev]) > 0).sum()))
        return out
    """Totals, raw counts and clustered errors of the pair sums, restricted to side-B rows maskB (and A rows maskA).
    The muon-clustered term uses all A rows (the per-row sums over the restricted B set are not kept), so for a
    subset of B it is an upper bound."""
    out = {}
    for lev in LEVELS:
        sB = acc.B[lev][maskB]
        tot = float(sB.sum())
        errB = float(np.sqrt((sB ** 2).sum()))
        sA = acc.A[lev] if maskA is None else acc.A[lev][maskA]
        fA = tot / acc.A[lev].sum() if acc.A[lev].sum() > 0 else 0.0     # scale the A-cluster term to the subset
        errA = float(np.sqrt((sA ** 2).sum())) * fA
        out[lev] = dict(per_15y=tot, err=float(np.hypot(errA, errB)), err_dis_cluster=errB, err_mu_cluster=errA,
                        n_pairs_mc=int(acc.nB[lev][maskB].sum()), n_dis_events_mc=int((acc.nB[lev][maskB] > 0).sum()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stores", nargs="*", default=[], help="label=store:kind (kind mu | nu)")
    ap.add_argument("--dataset", default=paths.MUON_DATASET)
    ap.add_argument("--mu-sample", type=int, default=300_000, help="random SST1 muon rows (weights scaled), as D9")
    ap.add_argument("--dis-sample", type=int, default=0, help="random single-track rows per store (weights scaled); 0 = all")
    ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--sst1-one", action="store_true", help="the good track must be the only charged track at SST1 (23 Nov)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--self", nargs="*", default=[], help="labels of stores for which DIS x DIS (different events) is also computed")
    ap.add_argument("--closure", action="store_true", help="mu x mu of the muon sample with the same code")
    ap.add_argument("--veto-design", default=None, help="after tagging (IP < 10 cm), on a random subsample of candidates")
    ap.add_argument("--veto-max", type=int, default=200_000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    cfg = reco.RecoConfig(seed=1, multiple_scattering=True)
    res = dict(method=__doc__.split("\n\n")[0], sst1_one=a.sst1_one, factor_15y_per_pair_weight=FAC, dataset=a.dataset, mu_sample=a.mu_sample,
               dis_sample=a.dis_sample, time_window_s=DEFAULT_CONFIG["time_window"], spills_15y=SPILLS_15Y, terms={}, stores={})
    mu = muon_tracks(a.dataset, a.mu_sample)
    from veto import muon_types as mt
    mu["q"] = mt.charge_from_pdg_column(mu["pdg"]).astype(float)
    mu["draw"] = np.arange(len(mu["ax"])) + 10**12
    res["muons"] = dict(rows=len(mu["ax"]), per_spill=float(mu["weight"].sum()))
    print(f"good SST1 muons {mu['weight'].sum():.4g} / spill ({len(mu['ax']):,} rows), {time.time() - t0:.0f} s", flush=True)
    if a.closure:
        acc, n, _ = pair_loop(mu, mu, cfg, a.block, same=True, log="mu x mu")
        res["terms"]["mu x mu (closure)"] = dict(summarise(acc, None, same=True), opposite_pairs=n)
        print("mu x mu closure", json.dumps(res["terms"]["mu x mu (closure)"]), flush=True)
    design = None
    if a.veto_design:
        from run_reco import load_inputs
        from veto import design as dsg
        design = dsg.load(Path(a.veto_design))
        _, inp = load_inputs(paths.RECO_INPUTS)
    for spec in a.stores:
        label, rest = spec.split("=", 1)
        store, kind = rest.rsplit(":", 1)
        B, info = single_tracks(store, kind, sst1_one=a.sst1_one)
        if a.dis_sample and a.dis_sample < len(B["bx"]):
            sel = np.sort(np.random.default_rng(11).choice(len(B["bx"]), a.dis_sample, replace=False))
            sc = len(B["bx"]) / a.dis_sample
            B = {k: v[sel] for k, v in B.items()}
            B["weight"] = B["weight"] * sc
            info["dis_subsample"] = dict(rows=a.dis_sample, scale=sc)
        res["stores"][label] = info
        print(f"{label}: {info['single_track_events']:,} single-track events, {info['single_rate_per_spill']:.4g} / spill "
              f"(no scattered mu: {info['single_rate_per_spill_noscat']:.4g}), {time.time() - t0:.0f} s", flush=True)
        acc, n, c01 = pair_loop(mu, B, cfg, a.block, keep01=design is not None, log=label)
        allB = np.ones(len(B["bx"]), bool)
        term = dict(opposite_pairs=n, single_rows=len(B["bx"]), all=summarise(acc, allB), noscat=summarise(acc, ~B["scat"]),
                    by_code={str(int(c)): summarise(acc, B["code"] == c) for c in np.unique(B["code"])},
                    by_code_noscat={str(int(c)): summarise(acc, (B["code"] == c) & ~B["scat"]) for c in np.unique(B["code"])})
        if design is not None and c01 is not None and len(c01[0]):
          try:
            ia, ib, v, w = c01
            for tag, m in (("all", np.ones(len(ia), bool)), ("noscat", ~B["scat"][ib])):
                k = np.nonzero(m)[0]
                if not len(k):
                    term[f"after_veto_{tag}"] = dict(per_15y=0.0, n=0); continue
                sc = 1.0
                if len(k) > a.veto_max:
                    k = np.sort(np.random.default_rng(5).choice(k, a.veto_max, replace=False)); sc = m.sum() / a.veto_max
                tr = {kk: np.concatenate([np.asarray(mu[kk], float), np.asarray(B[kk], float)]) for kk in KEYS}
                cand = np.stack([ia[k], ib[k] + len(mu["ax"])], 1)
                pre = dict(n=len(k), w=float(w[k].sum() / FAC * sc), ip=np.zeros(len(k)))   # all already IP < 10 cm
                r = veto_term(tr, cand, w[k] / FAC * sc, v[k], design, inp, cfg, np.random.default_rng(2), pre=pre)
                r.pop("_surv", None)
                term[f"after_veto_{tag}"] = dict(design=design.name, per_15y=r.get("after_15y", 0.0), err=r.get("after_err_15y", 0.0),
                                                 n_mc=r.get("n_after", 0), upper90=r.get("after_upper_15y"), sampled=len(k), scale=sc,
                                                 before_check=r["selected_15y"])
                print(f"  after veto ({tag}, {design.name}): {term[f'after_veto_{tag}']}", flush=True)
          except Exception as exc:                                  # the before-tagging numbers must survive
            term["after_veto_error"] = repr(exc); print("  veto failed:", repr(exc), flush=True)
        res["terms"][f"{label} x mu"] = term
        print(f"  {label} x mu: {n:.3g} opposite pairs; all {json.dumps({k: (round(v['per_15y'], 6), v['n_pairs_mc']) for k, v in term['all'].items()})}; "
              f"noscat {json.dumps({k: (round(v['per_15y'], 6), v['n_pairs_mc']) for k, v in term['noscat'].items()})}, {time.time() - t0:.0f} s", flush=True)
        if label in a.self:                                                 # DIS x DIS (different events) of this store
            dd = {}
            for tag, m in (("noscat", ~B["scat"]), ("all", allB)):
                Bs = {k: v[m] for k, v in B.items()}
                acc2, n2, _ = pair_loop(Bs, Bs, cfg, a.block, same=True, log=f"{label} x {label} {tag}")
                dd[tag] = dict(summarise(acc2, None, same=True), opposite_pairs=n2)
                print(f"  {label} x {label} ({tag}): {json.dumps({k: (v['per_15y'], v['n_pairs_mc']) for k, v in dd[tag].items() if k in LEVELS})}", flush=True)
                if not B["scat"].any():
                    dd["all"] = dd[tag]; break
            res["terms"][f"{label} x {label}"] = dd
        json.dump(res, open(a.out, "w"), indent=1, default=float)
        del B, acc
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
