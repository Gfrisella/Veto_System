#!/usr/bin/env python3
"""Follow-up (20 Nov 2026): depth of the library interaction behind each good rock track at SST1.

The draws file does not store the depth, so each good rock track is matched to its library exit through (pdg, |p|).
A rotation keeps |p|, so the match is exact to float precision. From the matched library event: vertex depth, muon
energy, alpha and the exit momentum. Also the library-level view: charged exits with p > 1 GeV per depth bin, and the
fraction that a naive "no transport" picture (exp(-path / lambda_I), path = depth / sin(exit angle to the surface))
would keep, as a check of what Geant4 transport already does.
Output: depth_match.json, depth.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
L = np.load("/disk/lhcb_data/sqasim/rock_dis/library/rock_library.npz")
z = np.load(f"{D}/data.npz")
good = z["r_good"]
gp = np.linalg.norm(z["r_p"][good], axis=1); gpdg = z["r_pdg"][good]; gw = z["r_w"][good]

xp = np.sqrt(L["x_px"].astype(np.float64) ** 2 + L["x_py"].astype(np.float64) ** 2 + L["x_pz"].astype(np.float64) ** 2)
sel = (~L["x_beam"]) & (xp > 0.9) & np.isin(np.abs(L["x_pdg"]), (11, 13, 211, 321, 2212))
xi = np.nonzero(sel)[0]
# owning event of each exit
evn = L["ev_n"]; first = L["ev_first"]
owner = np.repeat(np.arange(len(evn)), evn)
xo = owner[xi]
key_l = xp[xi]
o = np.argsort(key_l); key_s = key_l[o]
pos = np.clip(np.searchsorted(key_s, gp), 1, len(key_s) - 1)
cand = np.stack([pos - 1, pos], 1)
best = cand[np.arange(len(gp)), np.argmin(np.abs(key_s[cand] - gp[:, None]), 1)]
rel = np.abs(key_s[best] - gp) / gp
li = xi[o[best]]
ok = (rel < 1e-5) & (L["x_pdg"][li] == gpdg)
ev = owner[li]
dep = L["ev_depth"][ev].astype(float); E = L["E"][L["ev_ia"][ev] * 0 + L["ev_ie"][ev]]; al = L["alpha"][L["ev_ia"][ev]]
comp = L["ev_comp"][ev]
# angle of the exit to the surface (library frame: surface normal = +y)
sa_exit = L["x_py"][li] / xp[li]
res = dict(n_good=int(len(gp)), matched_frac=float(ok.mean()), matched_weight_frac=float(gw[ok].sum() / gw.sum()))
w = gw[ok]; d = dep[ok]; p = gp[ok]
edges = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
h = np.histogram(d, edges, weights=w)[0]
res["depth_bins_m"] = edges.tolist()
res["good_tracks_per_spill_by_depth"] = h.tolist()
res["frac_by_depth"] = (h / h.sum()).tolist()
def wq(x, ww, q):
    oo = np.argsort(x); c = np.cumsum(ww[oo]); c /= c[-1]
    return [float(x[oo][np.searchsorted(c, qq)]) for qq in q]
res["depth_quantiles_10_50_90_m"] = wq(d, w, [.1, .5, .9])
res["frac_depth_lt_10cm"] = float(w[d < 0.1].sum() / w.sum())
res["frac_depth_lt_30cm"] = float(w[d < 0.3].sum() / w.sum())
res["median_p_by_depth"] = [wq(p[(d >= a) & (d < b)], w[(d >= a) & (d < b)], [.5])[0] if ((d >= a) & (d < b)).sum() > 3 else None
                            for a, b in zip(edges[:-1], edges[1:])]
res["muon_energy_quantiles_10_50_90"] = wq(E[ok], w, [.1, .5, .9])
res["alpha_quantiles_10_50_90"] = wq(al[ok], w, [.1, .5, .9])
res["pythia_part_frac"] = float(w[comp[ok] == 1].sum() / w.sum())
# naive no-transport survival for the same tracks: path in rock = depth / sin(exit angle), lambda_I = 0.40 m
path = d / np.clip(sa_exit[ok], 1e-3, 1)
res["naive_path_median_m"] = wq(path, w, [.5])[0]
res["naive_survival_if_untransported"] = float((np.exp(-path / 0.40) * w).sum() / w.sum())
# parent of the exit: the exit itself is a Geant4-transported particle (exits are recorded at the surface crossing)
json.dump(res, open(f"{D}/depth_match.json", "w"), indent=1)
print(json.dumps(res, indent=1))

fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
axs[0].hist(d, edges, weights=w, color="#d62728", alpha=.7)
axs[0].set_xlabel("depth of the muon interaction below the rock surface [m]"); axs[0].set_ylabel("good rock tracks / spill / bin")
axs[0].set_title("depth behind the good rock tracks at SST1")
axs[1].scatter(d, p, s=4 + 0 * w, c="#d62728", alpha=.4)
axs[1].set_xscale("log"); axs[1].set_yscale("log"); axs[1].set_xlabel("depth [m]"); axs[1].set_ylabel("|p| at the rock surface = at SST1 [GeV/c]")
axs[1].set_title("momentum vs depth (Geant4-transported exits)")
fig.tight_layout(); fig.savefig(f"{D}/depth.png")
