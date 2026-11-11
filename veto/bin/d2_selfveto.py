#!/usr/bin/env python3
"""ISSUES D2 (decided 11 Nov 2026): can a neutrino interaction in the helium veto itself through its OWN lateral products
on the veto planes, and what does the same rule cost in signal?

Rule ("self-veto"): an event is rejected when any of its charged particles (p > --p-min) crosses a veto plane of the
design inside its active area (plane.inside), the plane firing with its efficiency. Straight lines from the start point
to the end point (event stores: (x0, y0, z0) -> (x1, y1, z1), the propagation of evstore; signal: from the decay vertex
to SST1). Photon conversions are not counted (the neutrino self-veto is a lower bound).

  neutrinos: the 2026 vessel stores (default evstore vessel_nu2026_<design>_nu<pdg>, 6 flavours): selection + veto b
             (interactions.select / event_tracks / reco.run_events, as interactions.evaluate), then the self-veto on the
             survivors -> per 15 y: after veto b, after veto b + self-veto, fraction removed.
  signal:    the B4 benchmark set (equal weights): EventCalc decays through the selection (as signal_eventcalc.
             selected_signal), the self-veto on ALL charged daughters of the selected decays -> signal lost per benchmark.

    python3 bin/d2_selfveto.py --design D6_straw_mod --out results/interactions/d2_selfveto_D6_straw_mod.json
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
from veto import geometry as geo
from veto import interactions as ix
from veto import paths, reco, spectrometer
from veto import signal_eventcalc as sg
from veto.config import SPILLS_15Y

NU = (14, -14, 12, -12, 16, -16)


def plane_hits(start, end, design, rng):
    """Mask (n,): the straight segment start -> end crosses an active veto plane (with the plane's efficiency)."""
    hit = np.zeros(len(start), bool)
    d = end - start
    for pl in design.planes:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (pl.z - start[:, 2]) / d[:, 2]
        cross = (t > 0) & (t <= 1) & np.isfinite(t)
        if not cross.any():
            continue
        c = np.nonzero(cross)[0]
        x = start[c, 0] + t[c] * d[c, 0]; y = start[c, 1] + t[c] * d[c, 1]
        ok = pl.inside(pl.to_local(x), y)
        eff = getattr(pl, "efficiency", None) or 0.999
        ok &= rng.random(len(c)) < eff
        hit[c[ok]] = True
    return hit


def nu_store(store, design, maps, cfg, rng, p_min):
    meta, ev, tr = es.read(paths.evstore(store))
    inter, part = es.to_fastveto(ev, tr)
    t = ix.charged_tracks(part)
    sel, pair, vertex = ix.select(t, len(inter["weight"]), cfg=cfg)
    if cfg.max_tracks is not None and len(sel):
        n_acc = np.bincount(t["inter"][t["acc"]], minlength=len(inter["weight"]))[sel]
        k = n_acc <= cfg.max_tracks
        sel, pair, vertex = sel[k], pair[k], vertex[k]
    w15 = inter["weight"] * SPILLS_15Y
    he = inter["code"][sel] == 0                                   # helium only (D2)
    sel, pair, vertex = sel[he], pair[he], vertex[he]
    if not len(sel):
        return dict(selected=0.0, after_b=0.0, after_b_self=0.0, n_selected=0, n_after_b=0, n_after_b_self=0)
    trk, evi, z_a, z_b, _ = ix.event_tracks(inter, t, sel, pair, vertex, None)
    surv = ~reco.run_events(trk, evi, z_b, design, maps, cfg, rng)["vetoed"][:len(sel)]
    # self-veto: every charged track of the surviving events, from the store (start -> end)
    ev_ids = ev["event"][sel]
    q = ix.charge(tr["pdg"])
    pm = np.sqrt(tr["px"] ** 2 + tr["py"] ** 2 + tr["pz"] ** 2)
    m = np.isin(tr["event"], ev_ids) & (q != 0) & (pm > p_min)
    st = np.stack([tr["x0"][m], tr["y0"][m], tr["z0"][m]], 1).astype(float)
    en = np.stack([tr["x1"][m], tr["y1"][m], tr["z1"][m]], 1).astype(float)
    h = plane_hits(st, en, design, rng)
    selfv = np.isin(ev_ids, np.unique(tr["event"][m][h]))
    ws = w15[sel]
    return dict(selected=float(ws.sum()), after_b=float(ws[surv].sum()), after_b_self=float(ws[surv & ~selfv].sum()),
                n_selected=int(len(sel)), n_after_b=int(surv.sum()), n_after_b_self=int((surv & ~selfv).sum()))


def signal(name, design, cfg, rng, p_min):
    ev, pa = sg.load(name)
    n = len(ev["weight"]); w = np.asarray(ev["weight"], float)
    vtx, mom, q, idx = sg.charged_tracks(ev, pa)
    q = q.astype(float)
    b = vtx + mom / mom[:, 2:3] * (geo.Z_SST1 - vtx[:, 2:3])
    acc = (np.abs(b[:, 0]) < geo.SST1_HALF_X) & (np.abs(b[:, 1]) < geo.SST1_HALF_Y)
    good = acc & (np.linalg.norm(mom, axis=1) > 1)
    g = np.nonzero(good)[0]
    if len(g):
        good[g] = spectrometer.timing_acceptance(b[g], mom[g], q[g])[0]
    good &= np.asarray(ev["in_vessel"]).astype(bool)[idx]
    t = dict(row=np.arange(len(q)), inter=idx, q=q, beam=np.zeros(len(q), bool), v=vtx, p=mom, b=b, acc=acc, good=good)
    sel, pair, vertex = ix.select(t, n, cfg=cfg, rng=rng)
    if cfg.max_tracks is not None and len(sel):
        k = np.bincount(idx[acc], minlength=n)[sel] <= cfg.max_tracks
        sel = sel[k]
    m = np.isin(idx, sel) & (np.linalg.norm(mom, axis=1) > p_min) & (mom[:, 2] > 0)
    end = np.where(mom[m, 2:3] > 0, b[m], vtx[m])
    h = plane_hits(vtx[m], end, design, rng)
    lost = np.isin(sel, np.unique(idx[m][h]))
    return dict(selected_weight=float(w[sel].sum()), self_vetoed_frac=float(w[sel][lost].sum() / max(w[sel].sum(), 1e-300)),
                n_selected=int(len(sel)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", required=True)
    ap.add_argument("--stores", nargs="*", default=None, help="neutrino event stores (default vessel_nu2026_<design>_nu<pdg>)")
    ap.add_argument("--p-min", type=float, default=0.1, help="GeV, charged particles that can fire a plane")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    p = Path(a.design)
    design = dsg.load(p if p.exists() else a.design)
    cfg = reco.RecoConfig(seed=1, multiple_scattering=True)
    _, maps = load_inputs()
    rng = np.random.default_rng(1)
    stores = a.stores or [f"vessel_nu2026_{design.name}_nu{f}" for f in NU]
    nu = {}
    for st in stores:
        if not paths.evstore(st).exists():
            print(f"  {st}: missing, skipped"); continue
        nu[st] = nu_store(st, design, maps, cfg, rng, a.p_min)
        r = nu[st]
        print(f"  {st}: helium selected {r['selected']:.3g}, after veto b {r['after_b']:.3g} ({r['n_after_b']} MC), "
              f"+ self-veto {r['after_b_self']:.3g} ({r['n_after_b_self']} MC), {time.time() - t0:.0f} s", flush=True)
    tot = {k: sum(r[k] for r in nu.values()) for k in ("selected", "after_b", "after_b_self")}
    sig = {}
    for name, wgt in sg.BENCHMARK_SET.items():
        sig[name] = signal(name, design, cfg, rng, a.p_min)
        print(f"  signal {name}: self-vetoed {100 * sig[name]['self_vetoed_frac']:.2f} % of {sig[name]['n_selected']} selected", flush=True)
    wsum = sum(sg.BENCHMARK_SET.values())
    sig_loss = sum(sg.BENCHMARK_SET[k] * v["self_vetoed_frac"] for k, v in sig.items()) / wsum
    removed = 1 - tot["after_b_self"] / tot["after_b"] if tot["after_b"] > 0 else float("nan")
    print(f"\n{design.name}: neutrino DIS in the helium per 15 y: after veto b {tot['after_b']:.3g} -> with the self-veto "
          f"{tot['after_b_self']:.3g} ({100 * removed:.0f} % removed); signal lost to the self-veto (equal-weight set) "
          f"{100 * sig_loss:.2f} %   ({time.time() - t0:.0f} s)")
    if a.out:
        json.dump(dict(design=design.name, p_min=a.p_min, neutrinos=nu, total=tot, removed_frac=removed, signal=sig,
                       signal_lost=sig_loss), open(a.out, "w"), indent=1, default=float)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
