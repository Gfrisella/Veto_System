#!/usr/bin/env python3
"""Example MS7 V0 events for an event display (Guglielmo, 16 Nov 2026: "evaluate the V0 trajectories in the experiment").

From the MS7 event stores (muon V0 chain, MS7 neutrinos): the events selected with the partial (IP < 2.5 m) or full
(IP < 0.1 m) selection, all of them for the muon chain and a weight-sampled subset for the neutrinos, with
  - every particle as its straight segment (start, end, momentum, species, parent, how it ended),
  - the parent muon at the MS7 parents plane (z = 29.20 m) when known,
  - the vertex pair, the visible momentum, the back-traced point at the MS7 exit (z = 29.90 m),
  - R2's decision (planes + SBT, level b, as interactions.evaluate) and the strip-station decision (layout, rule, theta),
  - the strip stations of the layout.
Writes a JSON for the page (results_rerun/reco/v0_examples.json).

    python3 bin/v0_examples.py --out results_rerun/reco/v0_examples.json
"""
import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np

from run_reco import load_inputs
from veto import design as dsg
from veto import evstore as es
from veto import interactions as ix
from veto import paths, reco
from veto import shield_strips as ss
from veto import vertex_selection as vsel


def hottest(origins_x, origins_y, w, n, h=0.20):
    H, xe, ye = np.histogram2d(origins_x, origins_y, [np.arange(-2.2, 2.2 + 1e-9, h), np.arange(-1.6, 1.6 + 1e-9, h)], weights=w)
    order = np.argsort(H.ravel())[::-1][:n]
    return [[float(0.5 * (xe[i] + xe[i + 1])), float(0.5 * (ye[j] + ye[j + 1]))] for i, j in zip(*np.unravel_index(order, H.shape))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    R = paths.results_dir()
    ap.add_argument("--stores", nargs="+", default=[str(R / "evstore" / "ms7_muon_v0s"), str(R / "evstore" / "ms7_nu2026")])
    ap.add_argument("--n-nu", type=int, default=150, help="neutrino events kept (weight-sampled)")
    ap.add_argument("--theta", type=float, default=0.01); ap.add_argument("--rule", default="point")
    ap.add_argument("--n-stations", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(5)
    design = dsg.load("R2_sbt_ubt")
    _, maps = load_inputs()
    par = np.load(paths.ms7_parents(), mmap_mode="r")
    events, origins = [], []
    for store in a.stores:
        meta, ev, trs = es.read(store)
        inter, part = es.to_fastveto(ev, trs)
        n = len(inter["weight"])
        tr = ix.charged_tracks(part)
        chosen = {}
        for ip in ("partial", "full"):
            cfg = reco.RecoConfig(seed=1, multiple_scattering=True, ip_max=vsel.IP_CUTS[ip])
            sel, pair, vertex = ix.select(tr, n, cfg=cfg)
            if len(sel) and cfg.max_tracks is not None:
                n_acc = np.bincount(tr["inter"][tr["acc"]], minlength=n)[sel]
                k = n_acc <= cfg.max_tracks
                sel, pair, vertex = sel[k], pair[k], vertex[k]
            if not len(sel):
                continue
            trk, evx, z_a, z_b, _ = ix.event_tracks(inter, tr, sel, pair, vertex, None)
            vetoed = reco.run_events(trk, evx, z_b, design, maps, cfg, np.random.default_rng(1))["vetoed"][:len(sel)]
            for i, s in enumerate(sel):
                c = chosen.setdefault(int(s), dict(sel=[], pair=pair[i], vertex=vertex[i], r2={}))
                c["sel"].append(ip); c["r2"][ip] = bool(vetoed[i])
        idx = np.array(sorted(chosen))
        if "nu" in Path(store).name and len(idx) > a.n_nu:
            w = inter["weight"][idx]; idx = np.sort(rng.choice(idx, a.n_nu, replace=False, p=w / w.sum()))
        print(f"{Path(store).name}: {len(chosen)} selected events, {len(idx)} kept", flush=True)
        ev_id = np.asarray(ev["event"])
        tev = np.asarray(trs["event"])
        order = np.argsort(tev, kind="stable"); tsorted = tev[order]
        for s in idx:
            c = chosen[int(s)]
            eid = int(ev_id[s])
            lo, hi = np.searchsorted(tsorted, eid), np.searchsorted(tsorted, eid, side="right")
            rows = order[lo:hi]
            segs = [dict(pdg=int(trs["pdg"][r]), parent=int(trs["parent"][r]), track=int(trs["track"][r]),
                         beam=bool(trs["beam"][r]), end=int(trs["end"][r]),
                         a=[round(float(trs[k][r]), 4) for k in ("x0", "y0", "z0")],
                         b=[round(float(trs[k][r]), 4) for k in ("x1", "y1", "z1")],
                         p=[round(float(trs[k][r]), 4) for k in ("px", "py", "pz")]) for r in rows]
            pv = tr["p"][c["pair"][0]] + tr["p"][c["pair"][1]]
            pr = int(ev["parent_row"][s])
            parent = None
            if "muon" in Path(store).name and 0 <= pr < len(par["x"]):
                parent = dict(x=[float(par[k][pr]) for k in ("x", "y", "z")], p=[float(par[k][pr]) for k in ("px", "py", "pz")])
            v = c["vertex"]
            dz = v[2] - ss.Z_EXIT
            bt = [float(v[0] - pv[0] / pv[2] * dz), float(v[1] - pv[1] / pv[2] * dz)]
            events.append(dict(store=Path(store).name, event=eid, w15=float(inter["weight"][s] * ix.SPILLS_15Y),
                               code=int(inter["code"][s]), v0=int(ev["v0"][s]),
                               iron=[float(inter[k][s]) for k in ("x", "y", "z")], vertex=[float(x) for x in v],
                               pvis=[float(x) for x in pv], backtrace=bt, dz=float(dz), sel=c["sel"], r2=c["r2"],
                               pair=[int(tr["row"][c["pair"][0]]), int(tr["row"][c["pair"][1]])], parent=parent, segs=segs,
                               _hits=ss.exit_hits(part["int"][part["int"] == s], part["pdg"][part["int"] == s],
                                                  part["p"][part["int"] == s], part["v"][part["int"] == s])))
            origins.append((float(inter["x"][s]), float(inter["y"][s]), float(inter["weight"][s])))
        del ev, trs, inter, part, tr
    # strip stations: the hottest cells of the selected parents (as bin/strip_study.py), and the decision per event
    ox, oy, ow = (np.array(c) for c in zip(*origins))
    st = hottest(ox, oy, ow, a.n_stations)
    cfg = ss.config({"shield_strips": {"stations": st, "rule": a.rule, "theta": a.theta}})
    for e in events:
        he, hx, hy = e.pop("_hits")
        hits = ss.on_stations(cfg, np.zeros(len(hx), int), hx, hy, rng)
        t = ss.tag(cfg, np.array([e["vertex"]]), np.array([e["pvis"]]), rng, hits=hits)
        e["strips"] = dict(tagged=bool(t["tagged"][0]), n_own=int(t["n_own"][0]), mu_random=float(t["mu_random"][0]), r=float(t["r"][0]))
        e["exit_hits"] = [[round(float(x), 4), round(float(y), 4)] for x, y in zip(hx, hy)]
    events.sort(key=lambda e: -e["w15"])
    out = dict(events=events, stations=st, station_size=cfg["size"], theta=a.theta, rule=a.rule, design="R2_sbt_ubt")
    json.dump(out, open(a.out, "w"), default=float)
    print(f"wrote {a.out}: {len(events)} events")


if __name__ == "__main__":
    main()
