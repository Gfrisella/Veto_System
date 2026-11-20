#!/usr/bin/env python3
"""Step-by-step cut flow of the combinatorial pair terms (mu x mu, single-track DIS x mu), per spill before the time
factor (sum of w_a w_b over pairs) and per 15 y after it (x 2 dt / T_spill x SPILLS_15Y, dt = 200 ps).

Steps: all pairs of good tracks, opposite charge (both exact, from the weight sums), then on random opposite-charge
pairs: DOCA < 2 cm, vertex in the vessel (no margin = benchmark_designs._pair_cut; and 5 cm inside), true IP < 3 m
(the candidate pre-cut), measured IP < 2.5 m, measured IP < 10 cm. The sampled pairs are drawn uniformly in (a, b)
within each charge combination and scaled by n_a n_b / K, so the late steps carry a sampling error; the exact numbers of
the late steps (all pairs) are those of bin/combi_single.py. Diagnostics per population: momentum and angle at SST1.

    python3 bin/combi_cutflow.py --stores ms7_mu=$R/evstore/ms7_muon_v0s:mu ... --out $R/combi_single/cutflow.json
"""
import _bootstrap  # noqa: F401

import argparse
import json
import time

import numpy as np

from combi_single import FAC, single_tracks
from rock_combinatorial import muon_tracks
from veto import geometry as G
from veto import muon_types as mt
from veto import paths, reco
from veto import vertex_selection as vsel

STEPS = ("all", "opposite", "doca", "vertex", "vertex_5cm", "true_ip3", "ip25", "ip01")


def wq(x, w, q):
    o = np.argsort(x); c = np.cumsum(w[o]); return float(x[o][np.searchsorted(c, q * c[-1])])


def population_info(d):
    p = np.sqrt(d["bpx"] ** 2 + d["bpy"] ** 2 + d["bpz"] ** 2)
    th = np.hypot(d["bpx"], d["bpy"]) / d["bpz"]
    w = np.asarray(d["weight"], float)
    return dict(per_spill=float(w.sum()), rows=len(w), p_median=wq(p, w, 0.5), p_q10=wq(p, w, 0.1), p_q90=wq(p, w, 0.9),
                theta_median_mrad=1e3 * wq(th, w, 0.5), theta_q90_mrad=1e3 * wq(th, w, 0.9),
                frac_p_below_5GeV=float(w[p < 5].sum() / w.sum()))


def cutflow(A, B, K, cfg, same, rng, chunk=5_000_000):
    wA, wB = np.asarray(A["weight"], float), np.asarray(B["weight"], float)
    qA, qB = np.sign(A["q"]), np.sign(B["q"])
    res = {k: 0.0 for k in STEPS}; cnt = {k: 0 for k in STEPS}
    if same:
        W = wA.sum(); res["all"] = (W * W - (wA ** 2).sum()) / 2
        res["opposite"] = wA[qA > 0].sum() * wA[qA < 0].sum()
    else:
        res["all"] = wA.sum() * wB.sum()
        res["opposite"] = wA[qA > 0].sum() * wB[qB < 0].sum() + wA[qA < 0].sum() * wB[qB > 0].sum()
    combos = [(np.nonzero(qA > 0)[0], np.nonzero(qB < 0)[0])]
    if not same:
        combos.append((np.nonzero(qA < 0)[0], np.nonzero(qB > 0)[0]))
    dz = []
    for ia_all, jb_all in combos:
        n_pairs = len(ia_all) * len(jb_all)
        k_c = K // len(combos)
        scale = n_pairs / k_c
        for c0 in range(0, k_c, chunk):
            k = min(chunk, k_c - c0)
            a = ia_all[rng.integers(0, len(ia_all), k)]; b = jb_all[rng.integers(0, len(jb_all), k)]
            if same:
                m = A["draw"][a] != B["draw"][b]; a, b = a[m], b[m]
            w = wA[a] * wB[b] * scale
            cnt["opposite"] += len(a)
            b1 = np.stack([A["bx"][a], A["by"][a], A["bz"][a]], 1); p1 = np.stack([A["bpx"][a], A["bpy"][a], A["bpz"][a]], 1)
            b2 = np.stack([B["bx"][b], B["by"][b], B["bz"][b]], 1); p2 = np.stack([B["bpx"][b], B["bpy"][b], B["bpz"][b]], 1)
            with np.errstate(all="ignore"):
                v, doca, ip = vsel.vertex_ip(b1, p1, b2, p2)
            m = doca < 0.02
            res["doca"] += w[m].sum(); cnt["doca"] += int(m.sum())
            hx = G.VESSEL_X_IN + (G.VESSEL_X_OUT - G.VESSEL_X_IN) * (v[:, 2] - G.Z_VESSEL_MIN) / 50
            hy = G.VESSEL_Y_IN + (G.VESSEL_Y_OUT - G.VESSEL_Y_IN) * (v[:, 2] - G.Z_VESSEL_MIN) / 50
            inv = lambda e: ((v[:, 2] > G.Z_VESSEL_MIN + e) & (v[:, 2] < G.Z_VESSEL_MAX - e)
                             & (np.abs(v[:, 0]) < hx - e) & (np.abs(v[:, 1]) < hy - e))
            m1 = m & inv(0.0); m5 = m & inv(0.05)
            res["vertex"] += w[m1].sum(); cnt["vertex"] += int(m1.sum())
            res["vertex_5cm"] += w[m5].sum(); cnt["vertex_5cm"] += int(m5.sum())
            dz.append(np.stack([v[m, 2], w[m]], 1))
            m3 = m5 & (ip < vsel.IP_PRECUT)
            res["true_ip3"] += w[m3].sum(); cnt["true_ip3"] += int(m3.sum())
            if m3.any():
                ipm = vsel.measured_ip(b1[m3], p1[m3], b2[m3], p2[m3], cfg, rng)
                for lev, cut in (("ip25", 2.5), ("ip01", 0.1)):
                    k2 = ipm < cut
                    res[lev] += w[m3][k2].sum(); cnt[lev] += int(k2.sum())
    if same:                                                       # one combination = all unordered opposite pairs
        pass
    dz = np.concatenate(dz) if dz else np.zeros((0, 2))
    zinfo = {}
    if len(dz):
        z, w = dz[:, 0], dz[:, 1]
        zinfo = dict(frac_doca_vertex_upstream_of_vessel=float(w[z < G.Z_VESSEL_MIN].sum() / w.sum()),
                     frac_doca_vertex_in_vessel_z=float(w[(z > G.Z_VESSEL_MIN) & (z < G.Z_VESSEL_MAX)].sum() / w.sum()),
                     frac_doca_vertex_downstream=float(w[z > G.Z_VESSEL_MAX].sum() / w.sum()))
    out = {s: dict(per_spill=float(res[s]), per_15y=float(res[s] * FAC), n_sampled=int(cnt[s])) for s in STEPS}
    out["sampled_opposite_pairs"] = int(cnt["opposite"])
    out["doca_vertex_z"] = zinfo
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stores", nargs="*", default=[], help="label=store:kind")
    ap.add_argument("--dataset", default=paths.MUON_DATASET)
    ap.add_argument("--mu-sample", type=int, default=300_000)
    ap.add_argument("--pairs", type=int, default=200_000_000, help="random opposite-charge pairs per term")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    cfg = reco.RecoConfig(seed=1, multiple_scattering=True)
    rng = np.random.default_rng(31)
    mu = muon_tracks(a.dataset, a.mu_sample)
    mu["q"] = mt.charge_from_pdg_column(mu["pdg"]).astype(float)
    mu["draw"] = np.arange(len(mu["ax"]))
    res = dict(method=__doc__.split("\n\n")[0], factor_15y=FAC, populations=dict(mu=population_info(mu)), terms={})
    res["terms"]["mu x mu"] = cutflow(mu, mu, a.pairs, cfg, True, rng)
    print("mu x mu", json.dumps({s: res["terms"]["mu x mu"][s] for s in STEPS}), f"{time.time() - t0:.0f} s", flush=True)
    for spec in a.stores:
        label, rest = spec.split("=", 1)
        store, kind = rest.rsplit(":", 1)
        B, info = single_tracks(store, kind)
        for tag, m in (("all", np.ones(len(B["bx"]), bool)), ("noscat", ~B["scat"])):
            if tag == "noscat" and not B["scat"].any():
                continue
            Bs = {k: v[m] for k, v in B.items()}
            res["populations"][f"{label}_{tag}"] = population_info(Bs)
            res["terms"][f"{label} x mu ({tag})"] = cutflow(mu, Bs, a.pairs, cfg, False, rng)
            print(label, tag, json.dumps({s: res["terms"][f"{label} x mu ({tag})"][s] for s in STEPS}), f"{time.time() - t0:.0f} s", flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
        del B
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
