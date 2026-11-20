# Diagnostic (20 Nov 2026): single-track MS7 interactions (dis_combinatorial's D9 events) - how often do the OTHER
# products of the same interaction cross the UBT (straight lines, charged, p > p_min), i.e. the interaction is "seen"
# in time with the track that reaches SST1. Also the stopped-in-shell muon spectra for the page.
import sys, json
W = "/disk/gfs_lhcb/gfrise/Study_on_Veto_System/Optimization_Veto_System/.claude/worktrees/nov4-source/veto"
sys.path[:0] = [W + "/bin", W + "/lib"]
import numpy as np
import _bootstrap  # noqa: F401
from veto import evstore as es, interactions as ix, paths
from dis_combinatorial import drop_v0_products_of_originals

Z_STRAW, Z_TILE = 30.73, 31.13
PLANES = {"R1/R2 tiles 4x4 m": (-2.0, 2.0, -2.0, 2.0), "Z2/Z3 tiles |x|<1.7": (-1.7, 1.7, -2.0, 1.7)}
out = {}
for store in ("/disk/gfs_lhcb/gfrise/Study_on_Veto_System/Optimization_Veto_System/veto/results_rerun/evstore/ms7_muon_v0s",):
    meta, ev, tr = es.read(paths.evstore(store))
    tr = drop_v0_products_of_originals(ev, tr)
    inter, part = es.to_fastveto(ev, tr)
    t = ix.charged_tracks(part)
    g = np.nonzero(t["good"])[0]
    n_good = np.bincount(t["inter"][g], minlength=len(inter["weight"]))
    single_int = np.nonzero(n_good == 1)[0]
    is_single = np.zeros(len(inter["weight"]), bool); is_single[single_int] = True
    good_row = np.zeros(len(part["int"]), bool); good_row[t["row"][g]] = True
    w_int = inter["weight"]
    # every other product of a single-track interaction
    q = ix.charge(part["pdg"])
    pp, v = part["p"][:, :3], part["v"]
    pabs = np.linalg.norm(pp, axis=1)
    other = is_single[part["int"]] & ~good_row & (pp[:, 2] > 0) & (v[:, 2] < Z_TILE)
    res = {"n_single_events": int(len(single_int)), "single_rate_per_spill": float(w_int[single_int].sum())}
    for pmin in (0.01, 0.1, 1.0):
        for name, (x0, x1, y0, y1) in PLANES.items():
            for kind, sel in (("charged", q != 0), ("charged+neutral", np.ones(len(q), bool))):
                m = other & sel & (pabs > pmin)
                idx = np.nonzero(m)[0]
                xt = v[idx, 0] + pp[idx, 0] / pp[idx, 2] * (Z_TILE - v[idx, 2])
                yt = v[idx, 1] + pp[idx, 1] / pp[idx, 2] * (Z_TILE - v[idx, 2])
                hit = (xt > x0) & (xt < x1) & (yt > y0) & (yt < y1)
                seen = np.zeros(len(w_int), bool); seen[part["int"][idx[hit]]] = True
                res[f"{kind} p>{pmin:g} | {name}"] = float(w_int[single_int][seen[single_int]].sum() / w_int[single_int].sum())
    # multiplicity of other products and who they are (weighted, single-track events)
    oth = other & (pabs > 0.1)
    n_oth = np.bincount(part["int"][oth], minlength=len(w_int))[single_int]
    res["other_products_hist"] = {str(k): float(w_int[single_int][np.minimum(n_oth, 10) == k].sum() / w_int[single_int].sum())
                                  for k in range(11)}
    pdg = np.abs(part["pdg"][oth])
    cats = {"γ": pdg == 22, "n": pdg == 2112, "p": pdg == 2212, "π±": pdg == 211, "μ": pdg == 13, "e": pdg == 11,
            "K": np.isin(pdg, [130, 310, 321]), "other": ~np.isin(pdg, [22, 2112, 2212, 211, 13, 11, 130, 310, 321])}
    res["other_products_pdg_w"] = {k: float(w_int[part["int"][oth][c]].sum()) for k, c in cats.items()}
    out[store] = res
    print(store, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if not isinstance(v, dict)}), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
