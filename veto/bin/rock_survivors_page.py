#!/usr/bin/env python3
"""Page data for the muon x rock pairs surviving the veto (bin/rock_combinatorial.py --survivors), per design.

    python3 bin/rock_survivors_page.py --designs R1_ubt_only R2_sbt_ubt --template <t.html> --out <page.html>

Histograms are weighted over ALL survivors; the event display gets a weight-proportional sample (or all if few).
"""
import argparse
import json
from pathlib import Path

import numpy as np

T = "muxrock__"


def hist(x, w, edges):
    h, _ = np.histogram(np.clip(x, edges[0], edges[-1] - 1e-9), edges, weights=w)
    return dict(edges=[round(float(e), 3) for e in edges], h=[float(v) for v in h])


def origin(ax, ay):
    """Where the rock track starts: cavern wall (-x), floor, other side (+x), other."""
    return np.where(ay < -3.0, 1, np.where(ax < -3.5, 0, np.where(ax > 3.5, 2, 3)))


def design_data(f, js, n_show, rng):
    w = f[T + "w15"]
    n = len(w)
    r = lambda k, leg: f[T + k][:, leg].astype(float)
    p_rock = np.sqrt(r("bpx", 1) ** 2 + r("bpy", 1) ** 2 + r("bpz", 1) ** 2)
    p_mu = np.sqrt(r("bpx", 0) ** 2 + r("bpy", 0) ** 2 + r("bpz", 0) ** 2)
    vtx = f[T + "vtx"]
    org = origin(r("ax", 1), r("ay", 1))
    mu_type = r("type", 0)
    out = dict(n=int(n), total=float(w.sum()), err=float(np.sqrt((w ** 2).sum())),
               n_eff=float(w.sum() ** 2 / max((w ** 2).sum(), 1e-300)),
               terms={k: {x: v for x, v in t.items() if x != "leg_explained"} | {"leg_explained": t.get("leg_explained")}
                      for k, t in js["terms"].items()},
               h=dict(vz=hist(vtx[:, 2], w, np.arange(31, 82.5, 1.25)),
                      rz=hist(r("az", 1), w, np.arange(20, 96, 2.5)),
                      prock=hist(np.log10(p_rock), w, np.arange(0, 2.61, 0.1)),
                      pmu=hist(np.log10(p_mu), w, np.arange(0, 2.61, 0.1))),
               origin=[float(w[org == k].sum()) for k in range(4)],
               mu_type=[float(w[mu_type == k].sum()) for k in (0, 1, 2)])
    if n <= n_show:
        idx = np.argsort(-w)
    else:
        idx = rng.choice(n, n_show, replace=False, p=w / w.sum())
        idx = idx[np.argsort(-w[idx])]
    rd = lambda a, d=3: [None if not np.isfinite(v) else round(float(v), d) for v in a]
    ev = dict(w=[float(v) for v in w[idx]], vx=rd(vtx[idx, 0]), vy=rd(vtx[idx, 1]), vz=rd(vtx[idx, 2]), ip=rd(f[T + "ip"][idx]),
              prock=rd(p_rock[idx], 2), pmu=rd(p_mu[idx], 2), org=[int(v) for v in org[idx]], mtype=[int(v) for v in mu_type[idx]],
              rpdg=[int(v) for v in r("pdg", 1)[idx]])
    for leg, tag in ((0, "m"), (1, "r")):
        for k in ("ax", "ay", "az", "kx", "ky", "kz", "bx", "by", "bz"):
            ev[tag + k] = rd(r(k, leg)[idx])
    out["ev"] = ev
    # the heaviest survivors (table)
    top = np.argsort(-w)[:12]
    out["top"] = [dict(w=float(w[i]), vz=round(float(vtx[i, 2]), 2), vx=round(float(vtx[i, 0]), 2), vy=round(float(vtx[i, 1]), 2),
                       prock=round(float(p_rock[i]), 2), pmu=round(float(p_mu[i]), 2), org=int(org[i]),
                       rz=round(float(r("az", 1)[i]), 2), mtype=int(mu_type[i]), ip=round(float(f[T + "ip"][i]), 3)) for i in top]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", nargs="+", default=["R1_ubt_only", "R2_sbt_ubt"])
    ap.add_argument("--dir", default=None, help="results_rerun/reco")
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-show", type=int, default=500)
    a = ap.parse_args()
    d = Path(a.dir) if a.dir else Path(__file__).resolve().parents[1] / "results_rerun" / "reco"
    data = {}
    for name in a.designs:
        f = np.load(d / f"rock_comb_surv_{name}.npz")
        js = json.load(open(d / f"rock_comb_{name}.json"))
        data[name] = design_data(f, js, a.n_show, np.random.default_rng(3))
        data[name]["rates"] = js["rates_per_spill"]
        print(name, data[name]["n"], f"{data[name]['total']:.4g} / 15 y", "origin", [round(x, 4) for x in data[name]["origin"]])
    Path(a.out).write_text(Path(a.template).read_text().replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
