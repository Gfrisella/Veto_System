"""Compare our muons with the h5 input in the OLD file's convention: momentum at production, selected by reaching the
2018 production's sensitive plane (behind the magnetised hadron absorber) with E > 10 GeV.

    python3 bin/old_convention.py [--out results/old_convention]

1. Energy loss production -> 2018 plane, from the 2018 file itself (data/flux/...22000_mu.root): every muon's
   MCTrack production momentum and its vetoPoint momentum (matched by track id), binned in production momentum.
2. Ours: momentum at creation (p_prod, stored by the forced decays; charm: plane momentum + our own mean loss),
   an energy loss drawn from the 2018 map for the same production momentum, kept if p_prod - loss > 10 GeV.
3. Production-momentum spectra vs the h5 classes (minimum bias 768.75, dimuon 7.6875, charm 326.16) + ratios.
"""

import _bootstrap  # noqa: F401
import argparse
import glob
from pathlib import Path

import awkward as ak
import h5py
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from veto import paths

INPUT = str(paths.MUONS_AND_MATTER / "data/muons/full_sample_after_target.h5")
LB = np.geomspace(5, 400, 41)                 # production-momentum bins of the loss map
PB = np.array([10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 130, 160, 200, 250, 300, 350, 400])
rng = np.random.default_rng(11)


def loss_map():
    t = uproot.open(str(paths.source_data() / "flux/pythia8_Geant4_10.0_withCharmandBeauty22000_mu.root"))["cbmsim"]
    mc = t.arrays(["MCTrack/MCTrack.fPdgCode", "MCTrack/MCTrack.fPx", "MCTrack/MCTrack.fPy", "MCTrack/MCTrack.fPz"], library="ak")
    vp = t.arrays(["vetoPoint/vetoPoint.fTrackID", "vetoPoint/vetoPoint.fPdgCode", "vetoPoint/vetoPoint.fPx",
                   "vetoPoint/vetoPoint.fPy", "vetoPoint/vetoPoint.fPz"], library="ak")
    tid = vp["vetoPoint/vetoPoint.fTrackID"]
    ismu = abs(vp["vetoPoint/vetoPoint.fPdgCode"]) == 13
    pveto = np.sqrt(vp["vetoPoint/vetoPoint.fPx"] ** 2 + vp["vetoPoint/vetoPoint.fPy"] ** 2 + vp["vetoPoint/vetoPoint.fPz"] ** 2)
    pmc = np.sqrt(mc["MCTrack/MCTrack.fPx"] ** 2 + mc["MCTrack/MCTrack.fPy"] ** 2 + mc["MCTrack/MCTrack.fPz"] ** 2)
    ok = ismu & (tid >= 0) & (tid < ak.num(pmc))
    tid_c = ak.where(ok, tid, 0)
    pprod = pmc[tid_c]
    pp = ak.to_numpy(ak.flatten(pprod[ok]))
    pv = ak.to_numpy(ak.flatten(pveto[ok]))
    loss = pp - pv
    idx = np.digitize(pp, LB) - 1
    table = [loss[idx == i] for i in range(len(LB) - 1)]
    print(f"loss map from {len(pp):,} muons (production -> 2018 plane):")
    for i in range(0, len(LB) - 1, 5):
        if len(table[i]):
            print(f"   p_prod {LB[i]:6.1f}-{LB[i + 1]:6.1f} GeV: {len(table[i]):7d} muons, loss median {np.median(table[i]):6.2f}, "
                  f"q10/q90 {np.quantile(table[i], 0.1):6.2f}/{np.quantile(table[i], 0.9):6.2f} GeV")
    return table


def draw_loss(table, p):
    """Energy loss for production momenta p: a random entry of the 2018 map in the same bin (nearest filled bin)."""
    idx = np.clip(np.digitize(p, LB) - 1, 0, len(LB) - 2)
    out = np.zeros(len(p))
    filled = np.array([len(t) > 20 for t in table])
    for i in np.unique(idx):
        j = i if filled[i] else np.nonzero(filled)[0][np.argmin(np.abs(np.nonzero(filled)[0] - i))]
        m = idx == i
        out[m] = table[j][rng.integers(0, len(table[j]), m.sum())]
    return out


def load(tag, use_plane_plus_loss=False, mean_loss=None):
    ps = [dict(np.load(f)) for f in sorted(glob.glob(str(paths.source_data() / "tail" / f"{tag}_[0-9]*.npz")))]
    ev = sum(int(p["events"]) for p in ps)
    d = {k: np.concatenate([p[k] for p in ps]) for k in ("px", "py", "pz", "p_prod")}
    d["w"] = np.concatenate([p["w_decay"] / p["rep"] for p in ps]) * 5e13 / ev
    d["p"] = np.sqrt(d["px"] ** 2 + d["py"] ** 2 + d["pz"] ** 2)
    if use_plane_plus_loss:
        d["p_prod"] = d["p"] + mean_loss(d["p"])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.results_dir() / "old_convention"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table = loss_map()
    mb_lo, mb_hi = load("mb2"), load("tail2")
    # our own mean loss creation -> our plane (from the min. bias run), for charm (no creation momentum stored)
    pl = np.geomspace(5, 400, 30)
    k = np.digitize(mb_lo["p"], pl)
    ml = np.array([np.median((mb_lo["p_prod"] - mb_lo["p"])[k == i]) if np.sum(k == i) > 50 else np.nan for i in range(1, len(pl))])
    good = np.isfinite(ml)
    mean_loss = lambda p: np.interp(p, np.sqrt(pl[1:] * pl[:-1])[good], ml[good])
    classes = {
        "minimum bias": (768.75, [(mb_lo, lambda d: d["p_prod"] < 150), (mb_hi, lambda d: d["p_prod"] >= 150)]),
        "dimuon": (7.6875, [(load("dmlow2"), lambda d: d["p_prod"] < 150), (load("dmhigh2"), lambda d: d["p_prod"] >= 150)]),
        "charm": (326.16, [(load("charmc", True, mean_loss), lambda d: d["p_prod"] > 0)]),
    }
    with h5py.File(INPUT, "r") as f:
        w_in = f["weight"][:]
        p_in = np.sqrt(sum(f[c][:].astype(np.float64) ** 2 for c in ("px", "py", "pz")))
    fig, axs = plt.subplots(2, 3, figsize=(21, 9), gridspec_kw={"height_ratios": [2, 1]})
    for j, (name, (cw, parts)) in enumerate(classes.items()):
        pp, ww = [], []
        for d, sel in parts:
            s = sel(d)
            keep = s & ((d["p_prod"] - draw_loss(table, d["p_prod"])) > 10.0)
            pp.append(d["p_prod"][keep]); ww.append(d["w"][keep])
        pp, ww = np.concatenate(pp), np.concatenate(ww)
        m = np.abs(w_in - cw) < 0.01 * cw
        ho, _ = np.histogram(p_in[m], PB, weights=w_in[m].astype(np.float64))
        no, _ = np.histogram(p_in[m], PB)
        hn, _ = np.histogram(pp, PB, weights=ww)
        hn2, _ = np.histogram(pp, PB, weights=ww ** 2)
        r = hn / ho
        e = r * np.hypot(1 / np.sqrt(np.maximum(no, 1)), np.sqrt(hn2) / np.maximum(hn, 1e-30))
        print(f"\n=== {name} (old convention: production momentum, reaches the 2018 plane with E > 10 GeV)")
        for i in range(len(PB) - 1):
            print(f"  {PB[i]:4.0f}-{PB[i + 1]:4.0f} GeV: old {ho[i]:9.3g}  ours {hn[i]:9.3g}  ratio {r[i]:6.3f} +- {e[i]:.3f}")
        print(f"  total above 10 GeV: old {ho.sum():.4g}, ours {hn.sum():.4g}, ratio {hn.sum() / ho.sum():.3f}")
        cen = np.sqrt(PB[1:] * PB[:-1])
        axs[0, j].stairs(ho, PB, color="k", lw=2, label="old h5")
        axs[0, j].stairs(hn, PB, color="crimson", lw=2, label="ours, old convention")
        axs[0, j].set_xscale("log"); axs[0, j].set_yscale("log"); axs[0, j].set_title(name); axs[0, j].legend()
        axs[1, j].errorbar(cen, r, yerr=e, fmt="o", color="crimson"); axs[1, j].axhline(1, color="k")
        axs[1, j].axhspan(0.9, 1.1, color="grey", alpha=0.15); axs[1, j].set_xscale("log"); axs[1, j].set_ylim(0, 2)
        axs[1, j].set_xlabel("muon momentum at production [GeV]"); axs[1, j].set_ylabel("ours / old")
    fig.suptitle("old convention: production momentum, kept if the muon reaches the 2018 plane with E > 10 GeV (loss map from the 2018 file)")
    fig.tight_layout(); fig.savefig(out / "old_convention.png", dpi=100)
    print("plot in", out)


if __name__ == "__main__":
    main()
