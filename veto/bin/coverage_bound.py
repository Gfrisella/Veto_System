#!/usr/bin/env python3
"""Worst-case bounds for the places the Monte Carlo barely visits (docs/STATUS.md 2026-11-10, "rare corners").

Each source is cut into regions of its interaction point; per region: MC events, interactions per 15 y, selected
(pre-veto: vertex + IP cut + max tracks, the same selection as interactions.evaluate) and 'loose' (all but the IP cut).
A region with no selected MC event is not a zero: it can hide up to 2.3 x (its mean MC weight) selected events (90 % CL).
Bounds printed per source:
  observed        selected per 15 y (pre-veto, so >= the veto-level number)
  sum of limits   sum over the regions with none selected of 2.3 W_r / n_r (every such region at its limit: very pessimistic)
  joint limit     2.3 x the largest mean weight of those regions (all of them together hold < 2.3 selected MC events)
  IP-extrapolated the loose events of those regions x (selected / loose of the whole source)
rock adds a physics proxy, since nothing there is selected: per interaction, the probability that a long-lived neutral
(K_L, K_S, Lambda, Xi0: decay; n, nbar: interaction in the helium) with p > 1 GeV makes a vertex inside the vessel while
its direction is within theta_max = k / p of the line to the target (k = 0.25 GeV for the decays, a neutrino's kick;
0.6 GeV for the neutron interactions), i.e. a vertex that could still point back. Charged products are left out: they
cross the SBT on the way in (veto inefficiency, a separate number).
rockflux: the same rock patches from the muons (no draw sampling): interactions per patch from every stored muon, to see
whether a patch without draws is empty or under-drawn. shell: the analog muon counts per veto-shell cell (shell_hist).

    python3 bin/coverage_bound.py store <store dir> --kind vessel|ms7|rock [--design ...] [--out f.json]
    python3 bin/coverage_bound.py rockflux <seed dir> --library rock_library.npz --seeds 300
    python3 bin/coverage_bound.py shell <seed dir> [--seeds N]
"""
import _bootstrap  # noqa: F401
import argparse
import glob
import json
import time

import numpy as np

from veto import evstore as es
from veto import geometry as G
from veto import interactions as ix
from veto.config import SPILLS_15Y

NEUTRAL_DECAY = {130: 15.34, 310: 0.026844, 3122: 0.0789, -3122: 0.0789, 3322: 0.0871, -3322: 0.0871}
NEUTRAL_INT = (2112, -2112)
MASS = {130: 0.497611, 310: 0.497611, 3122: 1.115683, -3122: 1.115683, 3322: 1.31486, -3322: 1.31486}
HE_PER_M = 1.66e-4 * 100 / 65.2          # helium at 1 atm: inelastic interactions per metre (lambda_I = 65.2 g / cm^2)
K_DECAY, K_INT, TH0 = 0.25, 0.6, 0.005


def regions(kind, ev):
    """region index and label of every event, by its interaction point (and the muon energy for MS7)"""
    x, y, z = (np.asarray(ev[k], float) for k in "xyz")
    code = np.asarray(ev["code"])
    if kind == "vessel":
        hx, hy = G.vessel_half_widths(np.clip(z, G.Z_VESSEL_MIN, G.Z_VESSEL_MAX), clip=True)
        iz = np.clip(((z - G.Z_VESSEL_MIN) / 2.5).astype(int), -1, 20) + 1           # 22 bins: before, 20 in, after
        sec = ((np.arctan2(y / hy, x / hx) + np.pi) / (2 * np.pi) * 8).astype(int) % 8
        r = np.hypot(x / hx, y / hy)
        ir = np.digitize(r, [0.5, 0.8, 0.95, 1.05])
        idx = ((code * 22 + iz) * 8 + sec) * 5 + ir
        lab = lambda i: (f"{ix.CODE_NAMES.get(i // 880, i // 880)} z{G.Z_VESSEL_MIN + 2.5 * ((i // 40) % 22 - 1):.1f}"
                         f" sec{(i // 5) % 8} r{['<.5', '.5-.8', '.8-.95', '.95-1.05', '>1.05'][i % 5]}")
    elif kind == "ms7":
        E = np.asarray(ev["E"], float)
        ixx = np.clip(((x + 1.75) / 0.25).astype(int), 0, 13); iyy = np.clip(((y + 1.25) / 0.25).astype(int), 0, 9)
        ie = np.digitize(E, [10, 30, 100, 300])
        idx = (ixx * 10 + iyy) * 5 + ie
        lab = lambda i: (f"x{-1.75 + 0.25 * (i // 50):+.2f} y{-1.25 + 0.25 * ((i // 5) % 10):+.2f} "
                         f"E{['<10', '10-30', '30-100', '100-300', '>300'][i % 5]}")
    else:                                                                              # rock: -x wall (5), floor (6)
        iz = np.clip(((z - 28.0) / 2.0).astype(int), 0, 29)
        it = np.where(code == 5, np.clip(((y - G.CAVERN_Y_FLOOR) / 1.0).astype(int), 0, 11),
                      np.clip(((x - G.CAVERN_X_WALL) / 1.0).astype(int), 0, 11))
        idx = ((code - 5) * 30 + iz) * 12 + it
        lab = lambda i: (f"{'wall' if i // 360 == 0 else 'floor'} z{28 + 2 * ((i // 12) % 30)}-{30 + 2 * ((i // 12) % 30)}"
                         f" {'y' if i // 360 == 0 else 'x'}{(G.CAVERN_Y_FLOOR if i // 360 == 0 else G.CAVERN_X_WALL) + i % 12:+.1f}")
    return idx, lab


def neutral_danger(ev, tr, chunk=200_000):
    """per event: expected pointing vertices in the vessel from its long-lived neutral roots (module doc)"""
    evrow = {int(e): i for i, e in enumerate(ev["event"])}
    pdg = np.asarray(tr["pdg"]).astype(np.int64)
    p = np.stack([tr["px"], tr["py"], tr["pz"]], 1).astype(float)
    pm = np.linalg.norm(p, axis=1)
    cand = np.nonzero((tr["parent"] < 0) & np.isin(pdg, list(NEUTRAL_DECAY) + list(NEUTRAL_INT)) & (pm > 1) & (p[:, 2] > 0))[0]
    out = np.zeros(len(ev["event"]))
    zs = np.arange(G.Z_VESSEL_MIN + 0.125, G.Z_VESSEL_MAX, 0.25)
    T = np.array([0.0, 0.0, G.Z_TARGET])
    for c0 in range(0, len(cand), chunk):
        c = cand[c0:c0 + chunk]
        u = p[c] / pm[c, None]
        x0 = np.stack([tr["x0"][c], tr["y0"][c], tr["z0"][c]], 1).astype(float)
        ell = (zs[None] - x0[:, 2:3]) / u[:, 2:3]                                   # path to each z step
        X = x0[:, 0:1] + u[:, 0:1] * ell; Y = x0[:, 1:2] + u[:, 1:2] * ell
        Z = np.broadcast_to(zs[None], X.shape)
        inside = (ell > 0) & G.inside_vessel(X, Y, Z)
        dl = 0.25 / u[:, 2:3]
        isdec = np.isin(pdg[c], list(NEUTRAL_DECAY))
        lam = np.where(isdec, pm[c] / np.array([MASS.get(int(q), 1.0) for q in pdg[c]]) *
                       np.array([NEUTRAL_DECAY.get(int(q), 1.0) for q in pdg[c]]), np.inf)[:, None]
        prob = np.where(isdec[:, None], np.exp(-np.maximum(ell, 0) / lam) * dl / lam, HE_PER_M * dl)
        d = np.stack([X - T[0], Y - T[1], Z - T[2]], -1)
        cosang = (d * u[:, None, :]).sum(-1) / np.linalg.norm(d, axis=-1)
        thmax = np.where(isdec, K_DECAY, K_INT) / pm[c] + TH0
        point = np.arccos(np.clip(cosang, -1, 1)) < thmax[:, None]
        r = (prob * (inside & point)).sum(1)
        np.add.at(out, [evrow[int(e)] for e in tr["event"][c]], r)
    return out


def store(a):
    from veto import reco
    from veto import vertex_selection as vsel
    t0 = time.time()
    meta, ev, tr = es.read(a.dir)
    inter, part = es.to_fastveto(ev, tr)
    n = len(inter["weight"])
    w15 = inter["weight"] * SPILLS_15Y
    cfg = reco.RecoConfig(seed=1, multiple_scattering=True, ip_max=vsel.IP_CUTS["full"])
    trk = ix.charged_tracks(part)
    flags = {}
    for name, c in (("sel", cfg), ("loose", None)):
        s, _, _ = ix.select(trk, n, cfg=c)
        if len(s):
            n_acc = np.bincount(trk["inter"][trk["acc"]], minlength=n)[s]
            s = s[n_acc <= cfg.max_tracks]
        f = np.zeros(n, bool); f[s] = True
        flags[name] = f
    print(f"{a.dir}: {n:,} events, selected {flags['sel'].sum():,}, loose {flags['loose'].sum():,} ({time.time() - t0:.0f} s)",
          flush=True)
    reg, lab = regions(a.kind, ev)
    danger = neutral_danger(ev, tr) if a.kind == "rock" else None
    res = {}
    for code in np.unique(inter["code"]):
        m = inter["code"] == code
        u, inv = np.unique(reg[m], return_inverse=True)
        nr = np.bincount(inv); W = np.bincount(inv, w15[m])
        S = np.bincount(inv, w15[m] * flags["sel"][m]); nS = np.bincount(inv, flags["sel"][m])
        L = np.bincount(inv, w15[m] * flags["loose"][m]); nL = np.bincount(inv, flags["loose"][m])
        zero = nS == 0
        mw = W / nr
        f_ip = S.sum() / L.sum() if L.sum() > 0 else 0.0
        row = dict(regions=len(u), regions_with_selected=int((~zero).sum()), mc_events=int(m.sum()), interactions15=float(W.sum()),
                   observed=float(S.sum()), n_observed=int(nS.sum()), loose=float(L.sum()), n_loose=int(nL.sum()),
                   sum_of_limits=float((2.3 * mw[zero]).sum()), joint_limit=float(2.3 * mw[zero].max()) if zero.any() else 0.0,
                   ip_extrapolated=float(L[zero].sum() * f_ip), mc_per_region_median=float(np.median(nr)),
                   regions_lt10_mc=int((nr < 10).sum()))
        # where the limit is biggest: the regions to look at
        top = np.argsort(-(2.3 * mw * zero))[:8]
        row["worst_regions"] = [dict(region=lab(int(u[k])), mc=int(nr[k]), interactions15=float(W[k]), limit=float(2.3 * mw[k]),
                                     loose=float(L[k])) for k in top if zero[k]]
        top_obs = np.argsort(-S)[:5]
        row["top_observed"] = [dict(region=lab(int(u[k])), mc=int(nr[k]), selected=float(S[k]), n=int(nS[k])) for k in top_obs if nS[k]]
        if danger is not None:
            D = np.bincount(inv, w15[m] * danger[m]); D2 = np.bincount(inv, (w15[m] * danger[m]) ** 2)
            row["neutral_pointing"] = float(D.sum()); row["neutral_pointing_err"] = float(np.sqrt(D2.sum()))
            row["neutral_pointing_events"] = int((danger[m] > 0).sum())
            topd = np.argsort(-D)[:5]
            row["top_neutral_pointing"] = [dict(region=lab(int(u[k])), mc=int(nr[k]), value=float(D[k])) for k in topd if D[k] > 0]
            # an undrawn region with the face's worst per-interaction danger
            Rmax = float(np.max(np.where(nr >= 100, D / np.maximum(W, 1e-300), 0)))
            row["neutral_pointing_rmax"] = Rmax
            row["neutral_pointing_joint_limit"] = float(2.3 * mw.max() * Rmax)
        name = ix.CODE_NAMES.get(int(code), str(code))
        res[name] = row
        print(f"\n  {name}: {row['regions']} regions ({row['regions_with_selected']} with a selected event), {row['mc_events']:,} MC "
              f"events (median {row['mc_per_region_median']:.0f} per region, {row['regions_lt10_mc']} with < 10), "
              f"{row['interactions15']:.3g} interactions / 15 y")
        print(f"    observed {row['observed']:.3g} ({row['n_observed']}), loose {row['loose']:.3g} ({row['n_loose']}); empty regions: "
              f"joint limit {row['joint_limit']:.3g}, sum of limits {row['sum_of_limits']:.3g}, IP-extrapolated {row['ip_extrapolated']:.3g}")
        for r in row["worst_regions"][:5]:
            print(f"      limit {r['limit']:9.3g}  {r['region']:<40} mc {r['mc']:>8}  interactions {r['interactions15']:.3g}  loose {r['loose']:.3g}")
        if danger is not None:
            print(f"    neutral pointing proxy: {row['neutral_pointing']:.3g} +- {row['neutral_pointing_err']:.2g} / 15 y "
                  f"({row['neutral_pointing_events']} MC events), worst patch per interaction {row['neutral_pointing_rmax']:.3g}, "
                  f"undrawn-patch limit {row['neutral_pointing_joint_limit']:.3g}")
            for r in row["top_neutral_pointing"]:
                print(f"      {r['value']:9.3g}  {r['region']:<40} mc {r['mc']}")
    print(f"({time.time() - t0:.0f} s)")
    return dict(store=a.dir, kind=a.kind, per_code=res)


def rockflux(a):
    import rock_apply as ra
    lib = ra.Library(a.library)
    files = sorted(glob.glob(f"{a.dir}/seed_*.npz"))[:a.seeds]
    S = len(files)
    acc = {}
    wmin = {0: np.inf, 1: np.inf}
    lam_max = {0: 0.0, 1: 0.0}
    for k, f in enumerate(files):
        d = np.load(f)
        r, P_e, u, wall, alpha, E0 = ra.rock_entries(d)
        if not len(r):
            continue
        lam, _, _ = ra.expected(P_e, u, alpha, E0, lib, ra.DEDX)
        L = lam.sum(1)
        w = d["weight"][r].astype(float)
        ev = dict(x=P_e[:, 0], y=P_e[:, 1], z=P_e[:, 2], code=np.where(wall, 5, 6))
        reg, lab = regions("rock", ev)
        for code, cw in ((5, True), (6, False)):
            m = wall == cw
            if m.any():
                wmin[code - 5] = min(wmin[code - 5], float(w[m].min())); lam_max[code - 5] = max(lam_max[code - 5], float(L[m].max()))
        u_, inv = np.unique(reg, return_inverse=True)
        for key, n_, s_, sw in zip(u_, np.bincount(inv), np.bincount(inv, w * L), np.bincount(inv, w)):
            a_ = acc.setdefault(int(key), [0, 0.0, 0.0])
            a_[0] += int(n_); a_[1] += s_; a_[2] += sw
        if k % 50 == 0:
            print(f"{k + 1}/{S} seeds", flush=True)
    keys = sorted(acc)
    _, lab = regions("rock", dict(x=np.zeros(1), y=np.zeros(1), z=np.zeros(1), code=np.array([5])))
    tot = {5: [0, 0.0], 6: [0, 0.0]}
    rows = []
    for key in keys:
        n_, s_, sw = acc[key]
        code = 5 if key < 360 else 6
        tot[code][0] += n_; tot[code][1] += s_
        rows.append(dict(region=lab(key), key=key, muon_rows=n_, interactions15=s_ / S * SPILLS_15Y, muons15=sw / S * SPILLS_15Y))
    print(f"\n{S} seeds; patches with muons: wall {sum(r['key'] < 360 for r in rows)} / 360, floor {sum(r['key'] >= 360 for r in rows)} / 360")
    for code in (5, 6):
        hole = 2.3 * wmin[code - 5] / S * SPILLS_15Y * lam_max[code - 5]
        print(f"  {'wall' if code == 5 else 'floor'}: {tot[code][0]:,} muon rows, {tot[code][1] / S * SPILLS_15Y:.3g} interactions / 15 y; "
              f"a patch with no muon row: < 2.3 x w_min x lambda_max = {hole:.3g} interactions / 15 y "
              f"(w_min {wmin[code - 5]:.3g} / spill, lambda_max {lam_max[code - 5]:.3g})")
    return dict(seeds=S, patches=rows, w_min=wmin, lambda_max=lam_max)


def shell(a):
    files = sorted(glob.glob(f"{a.dir}/seed_*.npz"))[:a.seeds]
    h = None
    for k, f in enumerate(files):
        z = np.load(f)
        x = np.asarray(z["shell_hist"], float)
        x = x if x.shape[0] == 3 else x.T
        h = x if h is None else h + x
        names = z["shell_cell_names"] if k == 0 else names
        if k % 500 == 0:
            print(f"{k + 1}/{len(files)}", flush=True)
    S = len(files)
    cnt, sw = h[0], h[1]
    print(f"{S} seeds: {len(cnt)} shell cells, {int((cnt == 0).sum())} with no muon at all, {int((cnt < 10).sum())} with < 10")
    w_an = np.median(np.where(cnt > 0, sw / np.maximum(cnt, 1), np.nan)[cnt > 0])
    print(f"  a cell with no muon: < 2.3 / {S} spills = {2.3 / S * SPILLS_15Y:.3g} muons / 15 y (unit-weight muons; median weight "
          f"per muon {w_an:.3g})")
    for i in np.argsort(cnt)[:12]:
        print(f"    {str(names[i]):<40} {int(cnt[i]):>8} muons, {sw[i] / S * SPILLS_15Y:.3g} / 15 y")
    return dict(seeds=S, counts=cnt.tolist(), sumw=sw.tolist(), names=[str(n) for n in names])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("store", "rockflux", "shell")); ap.add_argument("dir")
    ap.add_argument("--kind", choices=("vessel", "ms7", "rock"), default="vessel")
    ap.add_argument("--library", default=None); ap.add_argument("--seeds", type=int, default=300)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = dict(store=store, rockflux=rockflux, shell=shell)[a.mode](a)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=float)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
