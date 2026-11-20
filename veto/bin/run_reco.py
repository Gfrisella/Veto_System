#!/usr/bin/env python3
"""Run digitization + veto reconstruction (algorithm A) on designs and report what comes out.

For each design (designs/*.json):
  background  every muon-pair candidate (results/reco/inputs.npz) is one event: its two muons'
              own hits + accidental hits; vetoed if one of its tracks is explained.
              TP = vetoed, FN = survives. Surviving rate = sum of pair weights * 2 dt / spill.
  signal      hypothetical long-lived neutrals from the target (nearly along the beam) decaying in the
              vessel into two tracks, with the background pairs' selection (--signal old: tracks
              aimed at random points of SST1). No own hits: only accidental hits can veto them.
              FP = vetoed (signal lost), TN = kept.
  single      single SST1 muons: veto efficiency per muon type.
Also: detector rates per plane and cell, how well the search window contains the true
crossings, number of hit comparisons (the combinatorics) and runtime per event.

  bin/run_reco.py                          # all designs, local (needs results/reco/inputs.npz)
  bin/run_reco.py R1_ubt_only R2_sbt_ubt --spill 4.8
"""

import _bootstrap  # noqa: F401

import argparse
import json
import time

import numpy as np

from veto import vertex_selection as vsel  # noqa: E402
from veto import design as dsg
from veto import budget, shield_strips
from veto import paths, reco
from veto.config import DEFAULT_CONFIG

TYPE_NAMES = {0: "SST1 bounced", 1: "SST1 p>1 GeV & timing", 2: "SST1 other"}
from veto.config import SPILLS_15Y  # noqa: E402


def load_inputs(name=paths.RECO_INPUTS):
    """Tracks and candidates of results/reco/<name> (inputs.npz; inputs_xrun.npz: cross-run pairs)."""
    f = np.load(paths.results_dir() / "reco" / name)
    tr = {k[3:]: f[k].astype(np.float64) for k in f.files if k.startswith("tr_")}
    return tr, {k: f[k] for k in f.files if not k.startswith("tr_")}


def take(tr, rows):
    return {k: v[rows] for k, v in tr.items()}


def containment(tr, rows, design, cfg, rng):
    """Per muon type: fraction of true crossings that fall inside the algorithm's spatial window.
    Bounced muons are only predictable after their bounce (the spectrometer sees that part)."""
    sub = take(tr, rows)
    meas = reco.measure(sub, cfg, rng)
    ty = sub["type"].astype(int)
    inside, total = np.zeros(3), np.zeros(3)
    for plane in design.planes:
        u, y, _, ok = reco.true_crossing(sub, plane)
        dz = meas["z"] - plane.z
        pu = plane.to_local(meas["x"] - meas["tx"] * dz)
        py = meas["y"] - meas["ty"] * dz
        dist = dz * np.sqrt(1 + meas["tx"] ** 2 + meas["ty"] ** 2)
        r = cfg.r0 + (cfg.r_slope + cfg.r_ms / np.maximum(meas["p"], 0.1)) * dist
        uc, yc = plane.cell_centre(u, y)
        good = (np.abs(uc - pu) <= r + plane.cell_u / 2) & (np.abs(yc - py) <= r + plane.cell_y / 2)
        inside += np.bincount(ty[good & ok], minlength=3); total += np.bincount(ty[ok], minlength=3)
    return {TYPE_NAMES[k]: float(inside[k] / max(total[k], 1)) for k in range(3)}


def shell_loss_populations(tr, inp, scfg):
    """D19 (17 Nov 2026, sbt cfg shell_energy_loss): with the mean energy loss in the SBT shell along the stored paths, a muon
    that stops in the shell never reaches SST1: it leaves the single SST1 population and the pair candidates. The others reach
    SST1 with less momentum: a pair muon that now fails p > 1 GeV or the timing-detector acceptance (through the field) drops
    its pair; a single of type "p > 1 GeV & timing" that fails them becomes "SST1 other". Returns tr, inp (filtered) and info."""
    from veto import muon_types as mt
    from veto import sbt_material as sm
    el = sm.shell_energy_loss(tr, scfg.get("shell_eloss_beta", True))
    st = el["stopped"]
    p_old = np.sqrt(tr["bpx"] ** 2 + tr["bpy"] ** 2 + tr["bpz"] ** 2)
    crossed = el["de_mip"] > 0
    p_new = np.where(st, 0.0, np.where(crossed, np.sqrt(np.maximum(el["E_B"] ** 2 - sm.M_MU ** 2, 0.0)), p_old))
    with np.errstate(invalid="ignore", divide="ignore"):
        sc = np.where(p_old > 0, p_new / p_old, 1.0)
    acc = ~st & (p_new > 1) & (tr["bpz"] > 0)
    ty = tr["type"].astype(int)
    used = np.zeros(len(st), bool); used[inp["cand"].ravel()] = True; used[inp["single"]] |= ty[inp["single"]] == 1
    chk = np.nonzero(acc & used & crossed)[0]                   # only muons that lost energy can change
    if len(chk):
        b = np.stack([tr["bx"][chk], tr["by"][chk], tr["bz"][chk]], 1)
        pb = np.stack([tr["bpx"][chk], tr["bpy"][chk], tr["bpz"][chk]], 1) * sc[chk, None]
        acc[chk] = mt.timing_ok(b, pb, mt.charge_from_pdg_column(tr["pdg"][chk]))
    acc |= ~crossed & used                                     # untouched muons keep their selection
    cand, cw = inp["cand"], inp["cand_w"]
    keep_c = acc[cand[:, 0]] & acc[cand[:, 1]]
    single, sw = np.asarray(inp["single"]), np.asarray(inp["single_w"], float)
    keep_s = ~st[single]
    new_ty = ty.copy()
    retype = (ty == 1) & ~acc & ~st
    new_ty[retype] = 2
    s_ty = ty[single]
    T_old = np.sqrt(p_old ** 2 + sm.M_MU ** 2) - sm.M_MU
    info = dict(beta=bool(scfg.get("shell_eloss_beta", True)),
                single_stopped_w_frac=float(sw[~keep_s].sum() / sw.sum()),
                single_stopped_w_frac_by_type={TYPE_NAMES[k]: float(sw[~keep_s & (s_ty == k)].sum() / max(sw[s_ty == k].sum(), 1e-300))
                                               for k in range(3)},
                single_n_stopped=int((~keep_s).sum()),
                single_retyped_w_frac=float(sw[keep_s & retype[single]].sum() / sw.sum()),
                single_old_rangeout_w_frac=float(sw[el["de_mip"][single] >= T_old[single]].sum() / sw.sum()),
                pairs_removed_n=int((~keep_c).sum()), pairs_n=int(len(cand)),
                pairs_removed_w_frac=float(cw[~keep_c].sum() / max(cw.sum(), 1e-300)),
                pair_muons_stopped=int(st[np.unique(cand.ravel())].sum()))
    info["_old_rangeout"] = el["de_mip"] >= T_old                # per tr row, for the split of the single result
    inp = dict(inp, cand=cand[keep_c], cand_w=cw[keep_c], cand_vertex=inp["cand_vertex"][keep_c],
               single=single[keep_s], single_w=sw[keep_s])
    return dict(tr, type=new_ty.astype(tr["type"].dtype)), inp, info


def evaluate(design, tr, inp, cfg, n_signal, dt_pair, signal_model="target"):
    rng = np.random.default_rng(cfg.seed)
    eloss = None
    if design.sbt and design.sbt.get("shell_energy_loss"):     # D19: stopped muons are not SST1 muons
        tr, inp, eloss = shell_loss_populations(tr, inp, design.sbt)
        print(f"    shell energy loss: single SST1 weight stopped {100 * eloss['single_stopped_w_frac']:.1f} %, "
              f"pairs removed {eloss['pairs_removed_n']:,} of {eloss['pairs_n']:,} ({100 * eloss['pairs_removed_w_frac']:.2f} % "
              f"of the weight)", flush=True)
    res = {"design": design.name, "description": design.raw.get("description", ""), "notes": design.notes,
           "area_m2": design.area, "n_cells": design.n_cells, "n_planes": len(design.planes),
           "technologies": sorted({p.technology or "generic" for p in design.planes}),
           "n_channels": design.n_channels, "cost_kchf": design.cost, "rel_cost": design.rel_cost,
           "ubt_x0": design.ubt_x0}

    # background candidates, with the pointing cut on measured tracks (fixed seed: the same pairs for every design)
    keep = np.arange(len(inp["cand"]))
    if cfg.ip_max is not None:
        c = inp["cand"]
        st = lambda i: (np.stack([tr["bx"][i], tr["by"][i], tr["bz"][i]], 1), np.stack([tr["bpx"][i], tr["bpy"][i], tr["bpz"][i]], 1))
        ip = vsel.measured_ip(*st(c[:, 0]), *st(c[:, 1]), cfg, np.random.default_rng(777))
        keep = np.nonzero(ip < cfg.ip_max)[0]
    cand, cand_w = inp["cand"][keep], inp["cand_w"][keep]
    res["candidates_kept"] = int(len(keep))
    rows = cand.ravel()
    event = np.repeat(np.arange(len(cand)), 2)
    z_lim = np.repeat(inp["cand_vertex"][keep, 2], 2)
    t0 = time.perf_counter()
    pv = np.stack([tr["bpx"][cand[:, 0]] + tr["bpx"][cand[:, 1]], tr["bpy"][cand[:, 0]] + tr["bpy"][cand[:, 1]],
                   tr["bpz"][cand[:, 0]] + tr["bpz"][cand[:, 1]]], 1)
    b = reco.run_events(take(tr, rows), event, z_lim, design, inp, cfg, rng,
                        v0=dict(vtx=inp["cand_vertex"][keep], pvis=pv))       # V0 rule (if the design has it)
    t_bkg = time.perf_counter() - t0
    w = cand_w * 2 * dt_pair / cfg.spill_s                  # random-coincidence candidates per spill
    surv = ~b["vetoed"]
    ws = w[surv]                                            # A1 (16 Nov 2026): direct count is the estimate;
    top = np.argsort(-w * surv)[:min(20, int(surv.sum()))]  # its error is weighted (a few heavy pairs dominate)
    res["background"] = dict(n=len(cand), n_vetoed=int(b["vetoed"].sum()), n_survive=int(surv.sum()),
                             TP_frac=float(1 - w[surv].sum() / w.sum()), FN_frac=float(w[surv].sum() / w.sum()),
                             FN_frac_err=float(np.sqrt((ws ** 2).sum()) / w.sum()),
                             n_eff_survive=float(ws.sum() ** 2 / (ws ** 2).sum()) if len(ws) else 0.0,
                             survivors=[dict(pair=int(i), muons=[int(x) for x in cand[i]], w_15y=float(w[i] * SPILLS_15Y),
                                             share=float(w[i] / ws.sum())) for i in top],
                             per_spill_before=float(w.sum()), per_spill_after=float(w[surv].sum()),
                             per_15y_before=float(w.sum() * SPILLS_15Y), per_15y_after=float(w[surv].sum() * SPILLS_15Y),
                             hits_per_event=b["n_hits"] / len(cand), comparisons=b["n_compared"],
                             veto_by_plane={design.planes[i].name: int(c) for i, c in enumerate(b["by_plane"]) if c},
                             seconds=t_bkg, us_per_event=1e6 * t_bkg / len(cand), timing=b["timing"])
    # veto efficiency of the muons in the pairs, per type (for the factorised background, objective.py)
    ty, wt = tr["type"][rows].astype(int), np.repeat(cand_w, 2)
    res["pair_muons"] = {TYPE_NAMES[k]: float((wt * b["explained"])[ty == k].sum() / wt[ty == k].sum())
                         for k in range(3) if (ty == k).any()}

    # hypothetical signal
    if signal_model == "eventcalc":                          # B4: EventCalc benchmark set, full selection per benchmark
        from veto import signal_eventcalc as se
        per, fp_sum, w_sum, n_tot, n_hits = {}, 0.0, 0.0, 0, 0
        t0 = time.perf_counter()
        by_plane = np.zeros(len(design.planes), np.int64)
        for name, wb in se.BENCHMARK_SET.items():
            sig, z_v, ev, w_ev, acc_b = se.selected_signal(name, cfg, np.random.default_rng(779), n_max=n_signal)
            if not len(w_ev):
                per[name] = dict(acc_b, FP_frac=None)
                continue
            vtx, pv = shield_strips.candidates_from_tracks(sig, ev, z_v, len(w_ev))
            s = reco.run_events(sig, ev, z_v, design, inp, cfg, rng, with_own_hits=False, v0=dict(vtx=vtx, pvis=pv))
            st = shield_strips.signal_tag(design, sig, ev, z_v, len(s["vetoed"]), rng)
            if st is not None:                                     # silicon strip stations at the MS7 exit
                s["vetoed"] = s["vetoed"] | st
            fp = float((w_ev * s["vetoed"]).sum() / w_ev.sum())
            per[name] = dict(acc_b, FP_frac=fp, n=len(w_ev))
            fp_sum, w_sum, n_tot = fp_sum + wb * fp, w_sum + wb, n_tot + len(w_ev)
            by_plane += s["by_plane"]
            n_hits += s["n_hits"]
        fp = fp_sum / w_sum if w_sum else 0.0
        t_sig = time.perf_counter() - t0
        res["signal"] = dict(n=n_tot, FP_frac=fp, TN_frac=1 - fp,
                             FP_by_plane={design.planes[i].name: int(c) for i, c in enumerate(by_plane) if c},
                             hits_per_event=n_hits / max(n_tot, 1),
                             us_per_event=1e6 * t_sig / max(n_tot, 1), model=signal_model, per_benchmark=per,
                             acceptance={k: v["selected_weight_frac"] for k, v in per.items()})
    elif signal_model == "target":
        sig, z_v, ev, acceptance = reco.signal_from_target(n_signal, rng)
    else:
        (sig, z_v, ev), acceptance = reco.signal_tracks(n_signal, rng), {}
    if signal_model != "eventcalc" and cfg.ip_max is not None:   # the same pointing cut (rows i and i + n pair up)
        n2 = len(ev) // 2
        st = lambda i: (np.stack([sig["bx"][i], sig["by"][i], sig["bz"][i]], 1), np.stack([sig["bpx"][i], sig["bpy"][i], sig["bpz"][i]], 1))
        ok = vsel.measured_ip(*st(np.arange(n2)), *st(np.arange(n2, 2 * n2)), cfg, np.random.default_rng(778)) < cfg.ip_max
        acceptance = dict(acceptance, ip_cut_efficiency=float(ok.mean()))
        new_id = np.cumsum(ok) - 1
        rows_ok = ok[ev]
        sig = {k: v[rows_ok] for k, v in sig.items()}
        z_v, ev = z_v[rows_ok], new_id[ev[rows_ok]]
    if signal_model != "eventcalc":
        t0 = time.perf_counter()
        n_ev0 = int(ev.max()) + 1 if len(ev) else 0
        vtx, pv = shield_strips.candidates_from_tracks(sig, ev, z_v, n_ev0)
        s = reco.run_events(sig, ev, z_v, design, inp, cfg, rng, with_own_hits=False, v0=dict(vtx=vtx, pvis=pv))
        st = shield_strips.signal_tag(design, sig, ev, z_v, len(s["vetoed"]), rng)
        if st is not None:
            s["vetoed"] = s["vetoed"] | st
        t_sig = time.perf_counter() - t0
        n_ev = len(s["vetoed"])
        res["signal"] = dict(n=n_ev, FP_frac=float(s["vetoed"].mean()), TN_frac=float(1 - s["vetoed"].mean()),
                             FP_by_plane={design.planes[i].name: int(c) for i, c in enumerate(s["by_plane"]) if c},
                             hits_per_event=s["n_hits"] / n_ev, us_per_event=1e6 * t_sig / n_ev,
                             model=signal_model, acceptance=acceptance)

    # figure of merit. Positive = vetoed. Background rates are weighted (expected events), signal
    # rates are fractions of selected signal decays.
    w15 = w * SPILLS_15Y
    b_after = float(w15[surv].sum())
    from veto.interactions import ul90
    b_err = float(np.sqrt((w15[surv] ** 2).sum())) if surv.any() else ul90(w15)   # none left: 90 % CL limit
    eff = 1 - res["signal"]["FP_frac"]
    res["fom"] = dict(TPR=res["background"]["TP_frac"], FNR=res["background"]["FN_frac"],
                      FPR=res["signal"]["FP_frac"], TNR=eff, B_15y=b_after, B_15y_err=b_err,
                      B_15y_is_upper_limit=bool(not surv.any()), signal_efficiency=eff,
                      punzi=float(eff / (1.5 + np.sqrt(b_after))))

    # single muons
    single = inp["single"]
    sw = inp["single_w"]
    t0 = time.perf_counter()
    m = reco.run_events(take(tr, single), np.arange(len(single)), np.full(len(single), 1e9), design, inp, cfg, rng)
    ty = tr["type"][single].astype(int)
    res["single"] = {TYPE_NAMES[k]: float((sw * m["vetoed"])[ty == k].sum() / sw[ty == k].sum()) for k in range(3)}
    res["single"]["all SST1"] = float((sw * m["vetoed"]).sum() / sw.sum())
    if eloss is not None:                # split as 17 Nov (flags17/phys_single.py): MIP shell loss < / >= T at SST1 (stored)
        ro = eloss.pop("_old_rangeout")[single]
        f = lambda k: float((sw * m["vetoed"])[k].sum() / max(sw[k].sum(), 1e-300))
        eloss.update(single_tagged_physical=f(~ro), single_tagged_old_rangeout_kept=f(ro),
                     single_w_frac_old_rangeout_kept=float(sw[ro].sum() / sw.sum()))
        res["shell_energy_loss"] = eloss
    res["single"]["us_per_event"] = 1e6 * (time.perf_counter() - t0) / len(single)
    res["window_containment"] = containment(tr, rng.choice(single, 50_000, replace=False), design, cfg, rng)

    rates = reco.plane_rates(design, inp, cfg.spill_s)
    res["planes"] = [dict(r, peak_cell_hz=r["peak_cell"] / cfg.spill_s, mean_cell_hz=r["mean_cell"] / cfg.spill_s,
                          peak_channel_hz=r["peak_channel"] / cfg.spill_s, mean_channel_hz=r["mean_channel"] / cfg.spill_s)
                     for r in rates]
    res["total_rate_per_spill"] = float(sum(r["rate"] for r in rates))
    res["over_rate_limit"] = [dict(plane=r["plane"], technology=r["technology"], peak_hz_cm2=r["peak_hz_cm2"],
                                   area_m2=r["area_over_limit"], peak_channel_hz=r["peak_channel"] / cfg.spill_s)
                              for r in rates if r["area_over_limit"] > 0 or r["channel_over_limit"] or r.get("density_over_limit")]
    return res


def tag(cfg):
    """Output-name suffix: '' for the default window, '_tw100' for a fixed 100 ns one."""
    t = f"_tw{cfg.time_window_ns:g}" if cfg.time_window_ns else ""
    return t + ("_partialreco" if cfg.ip_max is not None and cfg.ip_max > 0.5 else "")


def report(r, cfg):
    b, s, m = r["background"], r["signal"], r["single"]
    print(f"\n=== {r['design']}: {r['description']}")
    for n in r["notes"]:
        print(f"    note: {n}")
    print(f"    {r['n_planes']} planes, {r['area_m2']:.1f} m2, {r['n_cells']:,} cells, "
          f"{r['total_rate_per_spill']:.3g} muon hits / spill ({r['total_rate_per_spill'] / cfg.spill_s:.3g} Hz)")
    print(f"    technology: {', '.join(r['technologies'])}; {r['n_channels']:,} readout channels, "
          f"cost {budget.drachma(r['cost_kchf']):.4g} drachmas ({budget.pct(r['cost_kchf']):.0f} % of the budget), "
          f"UBT material {100 * r['ubt_x0']:.2f} % X0")
    print(f"    background: {b['n']:,} candidates, TP {100 * b['TP_frac']:.3f} %  FN {100 * b['FN_frac']:.3f} % "
          f"({b['n_survive']:,} survive) -> {b['per_15y_before']:.3g} -> {b['per_15y_after']:.3g} events in 15 years")
    print(f"    signal:     FP (lost to accidental hits) {100 * s['FP_frac']:.3f} %, TN {100 * s['TN_frac']:.3f} %"
          + (f"  [{', '.join(f'{k}: {v}' for k, v in sorted(s['FP_by_plane'].items(), key=lambda t: -t[1])[:3])}]"
             if s["FP_by_plane"] else ""))
    f = r["fom"]
    print(f"    FoM: TPR {100 * f['TPR']:.4f} %  FNR {100 * f['FNR']:.4f} %  FPR {100 * f['FPR']:.3f} %  "
          f"TNR {100 * f['TNR']:.3f} %;  B(15 y) {f['B_15y']:.2g} ± {f['B_15y_err']:.1g};  Punzi {f['punzi']:.3f}")
    print(f"    single SST1 muons vetoed: " + ", ".join(f"{k} {100 * v:.2f} %" for k, v in m.items() if k != "us_per_event"))
    print("    search window contains the true crossings of: "
          + ", ".join(f"{k} {100 * v:.1f} %" for k, v in r["window_containment"].items()))
    print(f"    hits / event: bkg {b['hits_per_event']:.1f}, signal {s['hits_per_event']:.1f};  "
          f"{b['comparisons']:,} hit comparisons for the background")
    print(f"    runtime: {b['us_per_event']:.1f} us / background event, {s['us_per_event']:.1f} us / signal event "
          f"(numpy, one core)")
    worst = max(r["planes"], key=lambda p: p["peak_cell_hz"])
    print(f"    busiest cell: {worst['peak_cell_hz']:.3g} Hz on {worst['plane']} "
          f"(mean over that plane {worst['mean_cell_hz']:.3g} Hz)")
    for o in r["over_rate_limit"]:
        print(f"    OVER RATE LIMIT: {o['plane']} ({o['technology']}) peaks at {o['peak_hz_cm2'] / 1e6:.2f} MHz/cm2 "
              f"({o['area_m2']:.2f} m2 above the density limit), {o['peak_channel_hz'] / 1e6:.2f} MHz per channel")
    worst = max(r["planes"], key=lambda p: p["peak_channel_hz"])
    print(f"    busiest readout channel: {worst['peak_channel_hz']:.3g} Hz on {worst['plane']} "
          f"({worst['technology'] or 'generic'}, mean over that plane {worst['mean_channel_hz']:.3g} Hz)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("designs", nargs="*", help="design names or json files (default: all in designs/)")
    parser.add_argument("--spill", type=float, default=1.0, help="spill length [s]")
    parser.add_argument("--plane-rate-scale", type=float, default=1.0, help="x random-hit rate on the UBT planes (systematic)")
    parser.add_argument("--n-signal", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--signal", choices=("target", "old", "eventcalc"), default="target",
                        help="target: neutral from the target, decays in the vessel (default); old: tracks aimed at random; "
                             "eventcalc: the EventCalc benchmark set (signal_eventcalc.BENCHMARK_SET, B4), --n-signal per benchmark")
    parser.add_argument("--time-window", type=float, default=None,
                        help="fixed full width of the veto time window [ns] (default: +-3 sigma per plane)")
    parser.add_argument("--multiple-scattering", action="store_true",
                        help="scatter the muons after z = 30.6 m: walls, helium, air, UBT and fins (scattering.py)")
    parser.add_argument("--ip-cut", choices=("full", "partial"), default="full",
                        help="measured-IP selection: full (fully reconstructed, IP < 10 cm) | partial (IP < 2.5 m) (ISSUES C8)")
    parser.add_argument("--out", default=None, help="output file name in results/reco (default: results[_tw..][_oldsignal].json)")
    parser.add_argument("--inputs", default=paths.RECO_INPUTS, help="results/reco file from make_reco_inputs.py")
    parser.add_argument("--debris-in-time", action="store_true",
                        help="D14.1: each muon's own shield-end e / gamma on the UBT planes, in time with it (lib/veto/debris_in_time.py)")
    parser.add_argument("--acts-resolution", action="store_true",
                        help="D14.2: spectrometer smearing from the ACTS fit (lib/veto/acts_resolution.py) instead of the RecoConfig constants")
    parser.add_argument("--em-debris", action="store_true",
                        help="B2: add the EM debris maps (results/em_debris/em_maps.npz, bin/em_debris.py) to the accidentals")
    parser.add_argument("--no-ghosts", action="store_true", help="C1: no analytic ghost hits in strip planes (as before 2026-11-11)")
    args = parser.parse_args()
    names = args.designs or sorted(p.stem for p in dsg.DESIGN_DIR.glob("*.json"))
    cfg = reco.RecoConfig(spill_s=args.spill, plane_rate_scale=args.plane_rate_scale, seed=args.seed, time_window_ns=args.time_window,
                          multiple_scattering=args.multiple_scattering, ip_max=vsel.IP_CUTS[args.ip_cut],
                          ghost_hits=not args.no_ghosts, debris_in_time=args.debris_in_time,
                          acts_resolution=args.acts_resolution)
    tr, inp = load_inputs(args.inputs)
    if args.em_debris:                                   # B2: e / gamma from the shield end in the rate maps
        from veto import em_debris
        e = np.load(paths.results_dir() / "em_debris" / em_debris.EM_FILE)
        inp = dict(inp, **{k: e[k] for k in e.files if k.startswith("em_")})
    dt_pair = DEFAULT_CONFIG["time_window"] * 1e9 * 1e-9    # two-track coincidence window [s]
    results = []
    for name in names:
        r = evaluate(dsg.load(name), tr, inp, cfg, args.n_signal, dt_pair, args.signal)
        report(r, cfg)
        results.append(r)
    print(f"\nFIGURE OF MERIT (signal model: {args.signal}; positive = vetoed; spill {args.spill:g} s)")
    print(f"{'design':<15}{'area m2':>8}{'TPR %':>10}{'FNR %':>9}{'FPR %':>8}{'TNR %':>8}{'pairs left':>11}"
          f"{'B / 15 y':>18}{'Punzi':>7}{'drachmas':>11}{'channels':>14}")
    for r in results:
        f, b = r["fom"], r["background"]
        lim = "<" if f["B_15y_is_upper_limit"] else ""
        print(f"{r['design']:<15}{r['area_m2']:>8.0f}{100 * f['TPR']:>10.4f}{100 * f['FNR']:>9.4f}{100 * f['FPR']:>8.3f}"
              f"{100 * f['TNR']:>8.3f}{b['n_survive']:>11,}{(('< ' + format(f['B_15y_err'], '.2g') + ' 90%CL') if f['B_15y_is_upper_limit'] else (format(f['B_15y'], '.2g') + ' ± ' + format(f['B_15y_err'], '.1g'))):>18}"
              f"{f['punzi']:>7.3f}{budget.drachma(r['cost_kchf']):>11.1f}{r['n_channels']:>14,}")
    name = args.out or f"results{tag(cfg)}{'_oldsignal' if args.signal == 'old' else ''}.json"
    out = paths.results_dir() / "reco" / name
    out.write_text(json.dumps(dict(config=vars(cfg), spill_s=args.spill, inputs=args.inputs, signal=args.signal, results=results),
                              indent=1, default=float))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
