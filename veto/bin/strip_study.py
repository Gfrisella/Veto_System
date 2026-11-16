#!/usr/bin/env python3
"""Silicon strip stations at the MS7 exit (lib/veto/shield_strips.py): what they tag of the MS7 V0 background and what
they cost in signal, per layout, association rule and back-trace window (Guglielmo, 16 Nov 2026).

On the MS7 event stores (muon V0 chain, MS7 neutrinos), for the two IP selections:
  1. back-trace error: the selected candidate's vertex traced back along the visible momentum (pair sum) to z = 29.90 m,
     against the true interaction point in the iron -> the angular window theta (weighted quantiles);
  2. layouts: (a) the whole cavern-side yoke face, (b) both yokes, (c) the n hottest 20 x 20 cm cells of the selected
     background parents;
  3. per layout x rule (point / vertex) x theta: tagged fraction and residual per 15 years of the selected background,
     and the signal lost to random strip hits (EventCalc B4 benchmarks, equal weights).

    python3 bin/strip_study.py --stores ms7_muon_v0s ms7_nu2026 --out results_rerun/reco/strip_study.json
"""
import _bootstrap  # noqa: F401

import argparse
import json
import time

import numpy as np

from veto import evstore as es
from veto import interactions as ix
from veto import paths, reco
from veto import shield_strips as ss
from veto import signal_eventcalc as se
from veto import vertex_selection as vsel


def wq(x, w, q):
    o = np.argsort(x); cw = np.cumsum(w[o]); return np.interp(q, cw / cw[-1], x[o])


def background(store, ip_name, rng):
    from pathlib import Path
    meta, ev, trs = es.read(store if Path(store).exists() else str(paths.evstore(store)))
    store = Path(store).name
    inter, part = es.to_fastveto(ev, trs)
    del ev, trs
    cfg = reco.RecoConfig(seed=1, multiple_scattering=True, ip_max=vsel.IP_CUTS[ip_name])
    tr = ix.charged_tracks(part)
    n = len(inter["weight"])
    sel, pair, vertex = ix.select(tr, n, cfg=cfg)
    if len(sel) and cfg.max_tracks is not None:
        n_acc = np.bincount(tr["inter"][tr["acc"]], minlength=n)[sel]
        k = n_acc <= cfg.max_tracks
        sel, pair, vertex = sel[k], pair[k], vertex[k]
    pvis = tr["p"][pair[:, 0]] + tr["p"][pair[:, 1]]
    pos = np.full(n, -1); pos[sel] = np.arange(len(sel))
    return dict(store=store, ip=ip_name, w15=inter["weight"][sel] * ix.SPILLS_15Y, vtx=vertex, pvis=pvis,
                origin=np.stack([inter["x"][sel], inter["y"][sel], inter["z"][sel]], 1), part=part, pos=pos)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stores", nargs="+", default=["ms7_muon_v0s", "ms7_nu2026"])
    ap.add_argument("--ips", nargs="+", default=["partial", "full"])
    ap.add_argument("--thetas", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.04])
    ap.add_argument("--n-signal", type=int, default=40000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(1)
    t0 = time.time()
    res = dict(backtrace={}, layouts={}, results=[])
    bkg = []
    for s in a.stores:
        for ip in a.ips:
            b = background(s, ip, rng)
            w = b["w15"]
            if not len(w):
                print(f"{s} {ip}: nothing selected"); continue
            dz = b["vtx"][:, 2] - ss.Z_EXIT
            bx = b["vtx"][:, 0] - b["pvis"][:, 0] / b["pvis"][:, 2] * dz
            by = b["vtx"][:, 1] - b["pvis"][:, 1] / b["pvis"][:, 2] * dz
            d = np.hypot(bx - b["origin"][:, 0], by - b["origin"][:, 1])
            th = d / np.maximum(dz, 0.1)
            q = [0.5, 0.68, 0.9, 0.95]
            res["backtrace"][f"{s}/{ip}"] = dict(n=int(len(w)), selected_15y=float(w.sum()),
                                                 miss_m=dict(zip(map(str, q), wq(d, w, q).tolist())),
                                                 theta=dict(zip(map(str, q), wq(th, w, q).tolist())),
                                                 dz_median=float(wq(dz, w, [0.5])[0]),
                                                 origin_absx=dict(zip(map(str, q), wq(np.abs(b["origin"][:, 0]), w, q).tolist())))
            print(f"{s} {ip}: {len(w)} selected, {w.sum():.3g} / 15 y; back-trace miss median {wq(d, w, [.5])[0]:.3f} m, "
                  f"theta 68/90 % {wq(th, w, [.68])[0] * 1e3:.1f} / {wq(th, w, [.9])[0] * 1e3:.1f} mrad, {time.time() - t0:.0f} s")
            bkg.append(b)
    # layouts
    allx = np.concatenate([b["origin"][:, 0] for b in bkg]); ally = np.concatenate([b["origin"][:, 1] for b in bkg])
    allw = np.concatenate([b["w15"] / b["w15"].sum() for b in bkg])        # each store / selection counts equally
    h = 0.20
    H, xe, ye = np.histogram2d(allx, ally, [np.arange(-2.2, 2.2 + 1e-9, h), np.arange(-1.6, 1.6 + 1e-9, h)], weights=allw)
    order = np.argsort(H.ravel())[::-1]
    cells = [(0.5 * (xe[i] + xe[i + 1]), 0.5 * (ye[j] + ye[j + 1])) for i, j in zip(*np.unravel_index(order, H.shape))]
    layouts = {"cavern-side yoke": {"grid": [-1.8, -1.0, -1.2, 1.2]},
               "both yokes": [[x, y] for x in np.r_[np.arange(-1.7, -0.95, h), np.arange(1.1, 1.85, h)]
                              for y in np.arange(-1.1, 1.15, h)],
               "hottest 10": [list(c) for c in cells[:10]], "hottest 25": [list(c) for c in cells[:25]]}
    for name, st in layouts.items():
        c0 = ss.config({"shield_strips": {"stations": st}})
        res["layouts"][name] = {k: v for k, v in ss.summary(c0).items() if k != "over_strip_limit"} | dict(cost_kchf=ss.cost_kchf(c0))
    # signal candidates per IP selection (EventCalc benchmarks, equal weights)
    sig = {}
    for ip in a.ips:
        cfg = reco.RecoConfig(seed=1, multiple_scattering=True, ip_max=vsel.IP_CUTS[ip])
        sig[ip] = []
        for name in se.BENCHMARK_SET:
            t, z_v, ev, w_ev, _ = se.selected_signal(name, cfg, np.random.default_rng(779), n_max=a.n_signal)
            if len(w_ev):
                vtx, p = ss.candidates_from_tracks(t, ev, z_v, len(w_ev))
                sig[ip].append((vtx, p, w_ev / w_ev.sum()))
    for name, st in layouts.items():
        for rule in ("point", "vertex"):
            for th in a.thetas:
                cfg = ss.config({"shield_strips": {"stations": st, "rule": rule, "theta": th}})
                row = dict(layout=name, rule=rule, theta=th)
                for b in bkg:
                    part = b["part"]
                    he, hx, hy = ss.own_hits(cfg, part["int"], part["pdg"], part["p"], part["v"], rng)
                    k = b["pos"][he] >= 0
                    t = ss.tag(cfg, b["vtx"], b["pvis"], rng, hits=(b["pos"][he[k]], hx[k], hy[k]))
                    w = b["w15"]
                    row[f"{b['store']}/{b['ip']}"] = dict(tagged=float((w * t["tagged"]).sum() / w.sum()),
                                                          left_15y=float((w * ~t["tagged"]).sum()),
                                                          on_station=float((w * t["on_station"]).sum() / w.sum()))
                for ip in a.ips:
                    loss = [float((ww * ss.tag(cfg, v, p, rng)["tagged"]).sum()) for v, p, ww in sig[ip]]
                    row[f"signal_loss/{ip}"] = float(np.mean(loss)) if loss else None
                res["results"].append(row)
                print(name, rule, th, {k: (round(v["tagged"], 3) if isinstance(v, dict) else v) for k, v in row.items()
                                       if k not in ("layout", "rule", "theta")})
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
