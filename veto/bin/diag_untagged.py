#!/usr/bin/env python3
"""Why are single SST1 muons not tagged? (Guglielmo, 14 Nov 2026: "30 % of muons are not tagged" in R2)

Runs the single-muon part of run_reco.evaluate for one design (same seed / config as run_reco: RecoConfig(seed=1),
multiple scattering), with details, and records per muon the UBT decision (planes) and the SBT decision separately,
the true geometry (A, K, B, p, type), whether the true path crosses the UBT planes and the SBT liquid (max deposit per
cell), and a failure category for the untagged ones. Writes a JSON (weighted summaries + a point sample) for the
diagnostic page.

    python3 bin/diag_untagged.py R2_sbt_ubt --out results_test/reco/diag_untagged_R2.json
"""
import _bootstrap  # noqa: F401

import argparse
import json

import numpy as np

from run_reco import TYPE_NAMES, load_inputs, take
from veto import design as dsg
from veto import geometry as geo
from veto import reco, sbt

CATS = ["no detector on the path", "SBT deposit below threshold", "SBT hit, pointing missed",
        "UBT crossed, no own hit", "UBT hit, window missed", "other"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("design")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-sample", type=int, default=12000, help="points per class (untagged / tagged) in the JSON")
    a = ap.parse_args()

    d = dsg.load(a.design)
    tr_all, inp = load_inputs()
    single, w = inp["single"], inp["single_w"].astype(np.float64)
    tr = take(tr_all, single)
    n = len(single)
    cfg = reco.RecoConfig(seed=1, multiple_scattering=True)
    rng = np.random.default_rng(cfg.seed)
    m = reco.run_events(tr, np.arange(n), np.full(n, 1e9), d, inp, cfg, rng, details=True)
    tagged = m["vetoed"]
    q, found = m["queries"], m["found"]
    ubt_expl = np.bincount(q["track"][found], minlength=n) >= cfg.min_planes
    sbt_only = tagged & ~ubt_expl
    h = m["hits"]
    own = h["track"] >= 0
    ubt_own = np.bincount(h["track"][own], minlength=n) > 0

    # true UBT crossing (any UBT plane)
    crosses = np.zeros(n, bool)
    for pl in d.planes:
        if pl.kind in ("ubt", "tiles"):
            crosses |= reco.true_crossing(tr, pl)[3]
    # SBT liquid on the true path: max deposit per cell, first z in the shell
    p, qq, row, _ = sbt.path_segments(tr)
    c = sbt.shell_pieces(p, qq)
    dep_max = np.zeros(n); z_in = np.full(n, np.nan)
    if len(c["seg"]):
        r = row[c["seg"].astype(int)]
        ok = c["ring"] >= 0
        np.maximum.at(dep_max, r[ok], sbt.DEDX_LS * c["length"][ok])
        zz = np.full(n, np.inf); np.minimum.at(zz, r[ok], c["z"][ok]); z_in = np.where(np.isfinite(zz), zz, np.nan)
    scfg = d.sbt or {}
    thr = scfg.get("threshold_mev", 30.0)
    # does the measured backward line (from SST1) cross the SBT at all
    meas = m["meas"]
    dz_end = meas["z"] - (geo.Z_VESSEL_MIN - 0.01)
    start = np.stack([meas["x"], meas["y"], meas["z"]], 1)
    end = np.stack([meas["x"] - meas["tx"] * dz_end, meas["y"] - meas["ty"] * dz_end, np.full(n, geo.Z_VESSEL_MIN - 0.01)], 1)
    cr = sbt.crossings(start, end)
    line_sbt = np.zeros(n, bool); line_sbt[cr["seg"]] = True
    # first (most downstream) crossing of the backward line, and the cell with the largest true deposit
    z_cross = np.full(n, np.nan); f_cross = np.full(n, -1)
    if len(cr["seg"]):
        o = np.argsort(-cr["z"])
        sg, first = np.unique(cr["seg"][o], return_index=True)
        z_cross[sg] = cr["z"][o][first]; f_cross[sg] = cr["face"][o][first]
    z_dep = np.full(n, np.nan); f_dep = np.full(n, -1)
    if len(c["seg"]):
        dd = np.where(c["ring"] >= 0, sbt.DEDX_LS * c["length"], -1.0)
        o = np.lexsort((-dd, r))
        rr, first = np.unique(r[o], return_index=True)
        k = o[first]; good = dd[k] > 0
        z_dep[rr[good]] = c["z"][k][good]; f_dep[rr[good]] = c["face"][k][good]

    cat = np.full(n, -1)
    un = ~tagged
    has_sbt = dep_max > thr
    cat[un & crosses & ubt_own] = 4
    cat[un & crosses & ~ubt_own & (cat < 0)] = 3
    cat[un & has_sbt & (cat < 0)] = 2
    cat[un & (dep_max > 0) & ~has_sbt & (cat < 0)] = 1
    cat[un & (dep_max == 0) & ~crosses & (cat < 0)] = 0
    cat[un & (cat < 0)] = 5

    ty = tr["type"].astype(int)
    pmag = np.sqrt(tr["bpx"] ** 2 + tr["bpy"] ** 2 + tr["bpz"] ** 2)
    W = w.sum()
    summary = {"design": d.name, "n": int(n), "tagged": float(w[tagged].sum() / W),
               "ubt": float(w[ubt_expl].sum() / W), "sbt_only": float(w[sbt_only].sum() / W),
               "threshold_mev": thr, "categories": CATS, "types": [TYPE_NAMES[k] for k in range(3)],
               "by_type": {}, "cat_of_all": [float(w[cat == k].sum() / W) for k in range(len(CATS))]}
    for k in range(3):
        mk = ty == k
        wk = w[mk].sum()
        summary["by_type"][TYPE_NAMES[k]] = dict(
            share=float(wk / W), tagged=float(w[mk & tagged].sum() / wk), ubt=float(w[mk & ubt_expl].sum() / wk),
            sbt_only=float(w[mk & sbt_only].sum() / wk),
            cats=[float(w[mk & (cat == c_)].sum() / wk) for c_ in range(len(CATS))])
    # extra diagnostics on the untagged
    summary["untagged_line_crosses_sbt"] = float(w[un & line_sbt].sum() / max(w[un].sum(), 1e-30))
    summary["untagged_true_in_sbt"] = float(w[un & (dep_max > 0)].sum() / max(w[un].sum(), 1e-30))

    summary["raw"] = {CATS[k]: dict(n=int((cat == k).sum()), w=float(w[cat == k].sum() / W), n_eff=float(w[cat == k].sum() ** 2 / max((w[cat == k] ** 2).sum(), 1e-300))) for k in range(len(CATS))}
    summary["raw"]["tagged"] = dict(n=int(tagged.sum()), w=float(w[tagged].sum() / W), n_eff=float(w[tagged].sum() ** 2 / (w[tagged] ** 2).sum()))
    # sample per class: the heaviest a.n_sample/4 muons with their own weight, plus a uniform sample of the rest with the
    # weight scaled by (rest / sampled), so weighted sums over the points reproduce the full set (n_eff is only ~500:
    # a few hundred muons carry most of the weight)
    pick, wp = [], []
    for cls in (un, tagged):
        i = np.nonzero(cls)[0]
        o = i[np.argsort(-w[i])]
        top, rest = o[: a.n_sample // 4], o[a.n_sample // 4:]
        k = min(a.n_sample - len(top), len(rest))
        rs = rng.choice(rest, k, replace=False) if k else rest[:0]
        pick += [top, rs]
        wp += [w[top], w[rs] * (len(rest) / max(k, 1))]
    i = np.concatenate(pick)
    wpt = np.concatenate(wp) / W
    f = lambda x, nd=3: [None if not np.isfinite(v) else round(float(v), nd) for v in x]
    pts = dict(w=[float(f"{v:.4g}") for v in wpt], tagged=tagged[i].astype(int).tolist(), cat=cat[i].tolist(), type=ty[i].tolist(),
               ubt=ubt_expl[i].astype(int).tolist(), sbt=sbt_only[i].astype(int).tolist(),
               p=f(pmag[i], 2), ax=f(tr["ax"][i]), ay=f(tr["ay"][i]), az=f(tr["az"][i], 2),
               kx=f(tr["kx"][i]), ky=f(tr["ky"][i]), kz=f(tr["kz"][i], 2), bx=f(tr["bx"][i]), by=f(tr["by"][i]),
               dep=f(dep_max[i], 1), zin=f(z_in[i], 2), cross=crosses[i].astype(int).tolist(),
               own=ubt_own[i].astype(int).tolist(), line=line_sbt[i].astype(int).tolist(), q=np.sign(tr["pdg"][i]).astype(int).tolist(),
               zc=f(z_cross[i], 2), fc=f_cross[i].tolist(), zd=f(z_dep[i], 2), fd=f_dep[i].tolist())
    pm = un & has_sbt & ~crosses
    summary["pointing_missed"] = dict(
        line_misses_sbt=float(w[pm & ~line_sbt].sum() / max(w[pm].sum(), 1e-30)),
        face_differs=float(w[pm & line_sbt & (f_cross != f_dep)].sum() / max(w[pm].sum(), 1e-30)),
        dz_median=float(np.nanmedian(np.abs(z_cross - z_dep)[pm & line_sbt & (f_cross == f_dep)])) if (pm & line_sbt & (f_cross == f_dep)).any() else None)
    summary["geometry"] = dict(z_vessel=[geo.Z_VESSEL_MIN, geo.Z_VESSEL_MAX], z_sst1=geo.Z_SST1,
                               x_in=geo.VESSEL_X_IN, y_in=geo.VESSEL_Y_IN, x_out=geo.VESSEL_X_OUT, y_out=geo.VESSEL_Y_OUT)
    json.dump(dict(summary=summary, points=pts), open(a.out, "w"))
    print(json.dumps({k: v for k, v in summary.items() if k != "geometry"}, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
