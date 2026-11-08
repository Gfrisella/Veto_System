#!/usr/bin/env python3
"""Diagnostic: are rare SST1 arrivals driven by unusual angular deflections in the shield? (sandbox, analog tracks)

    python3 diag_angles.py --tracks tracks.npz --ref ref.npz --out diag

tracks.npz (splitting.py --analog --record-tracks): the state of every source muon at the 14 checkpoints.
Per segment (source -> 2.30, 2.30 -> 5.45, ... 70 -> 82 m), for muons alive at both ends:
  dtheta      change of the slopes (px/pz, py/pz) [mrad]; split into
    field     the conditional mean given (segment, charge, log p, x, y at the segment start): the deterministic bending
              (+ mean energy-loss effects) of the transport, estimated from the population itself
    stoch     dtheta - field: the stochastic part (multiple scattering, energy-loss fluctuations)
  dp          change of momentum [GeV]
  dperp       transverse displacement from the straight line of the segment start [m]
Per trajectory: the largest |stoch| and |dtheta| and the segment where they occur.
SST1 classes from the reference occupancy (ref.npz, 97 M analog sources, 1 m regions): common (>= 1000 rows in the region),
low (10-999), isolated (< 10 rows, or outside +-12 m). Predictive power of each variable for rare (low + isolated) vs common:
ROC AUC (0.5 = none) with a bootstrap 90 % interval. Counts are reported: small rare samples are not significant.
"""
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PB = np.geomspace(1, 400, 13)


def auc(pos, neg, rng, n_boot=200, cap=200_000):
    """ROC AUC of a score separating pos from neg (Mann-Whitney), with a bootstrap 90 % interval"""
    def one(a, b):
        x = np.concatenate([a, b]); r = np.argsort(np.argsort(x)) + 1
        return (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
    if len(neg) > cap:
        neg = rng.choice(neg, cap, replace=False)
    v = one(pos, neg)
    bs = [one(rng.choice(pos, len(pos)), rng.choice(neg, min(len(neg), 50_000))) for _ in range(n_boot)]
    return v, np.percentile(bs, 5), np.percentile(bs, 95)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", required=True); ap.add_argument("--ref", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(5)
    T = np.load(a.tracks)
    tr, src, q, Z = T["tracks"], T["source"], T["q"].astype(int), T["checkpoints"]
    K, n = tr.shape[0], tr.shape[1]
    S = np.concatenate([src[None], tr], 0)                     # state index 0 = source, 1..K = checkpoints
    zs = np.concatenate([[float(np.median(src[:, 2]))], Z])
    alive = np.isfinite(S[..., 0])
    p = np.sqrt(S[..., 3].astype(np.float64) ** 2 + S[..., 4] ** 2 + S[..., 5] ** 2)
    tx, ty = S[..., 3] / S[..., 5], S[..., 4] / S[..., 5]
    seg_names = [f"{zs[j]:.1f}-{zs[j + 1]:.1f}" for j in range(K)]
    D = {k: np.full((K, n), np.nan) for k in ("dtx", "dty", "ftx", "fty", "dp", "dperp")}
    for j in range(K):
        m = alive[j] & alive[j + 1]
        D["dtx"][j, m] = 1e3 * (tx[j + 1, m] - tx[j, m]); D["dty"][j, m] = 1e3 * (ty[j + 1, m] - ty[j, m])
        D["dp"][j, m] = p[j + 1, m] - p[j, m]
        dz = S[j + 1, m, 2] - S[j, m, 2]
        D["dperp"][j, m] = np.hypot(S[j + 1, m, 0] - (S[j, m, 0] + tx[j, m] * dz), S[j + 1, m, 1] - (S[j, m, 1] + ty[j, m] * dz))
        # deterministic part: mean dtheta given (charge, log p, x, y) at the segment start; coarser (charge, log p) if thin
        ip = np.clip(np.digitize(p[j, m], PB) - 1, 0, len(PB) - 2)
        ix = np.clip(np.digitize(S[j, m, 0], np.linspace(-6, 6, 49)), 0, 49)
        iy = np.clip(np.digitize(S[j, m, 1], np.linspace(-6, 6, 49)), 0, 49)
        fine = (((q[m] > 0) * len(PB) + ip) * 50 + ix) * 50 + iy
        coarse = (q[m] > 0) * len(PB) + ip
        for c in ("tx", "ty"):
            d = D[f"d{c}"][j, m]
            uf, invf, cf = np.unique(fine, return_inverse=True, return_counts=True)
            mf = np.bincount(invf, d) / cf
            uc, invc, cc = np.unique(coarse, return_inverse=True, return_counts=True)
            mc = np.bincount(invc, d) / cc
            D[f"f{c}"][j, m] = np.where(cf[invf] >= 20, mf[invf], mc[invc])
    D["dth"] = np.hypot(D["dtx"], D["dty"])
    D["stoch"] = np.hypot(D["dtx"] - D["ftx"], D["dty"] - D["fty"])
    D["field"] = np.hypot(D["ftx"], D["fty"])
    # per trajectory: largest deviations and where
    def argmax_nan(x):
        y = np.where(np.isfinite(x), x, -1.0)
        return y.max(0), y.argmax(0)
    max_st, where_st = argmax_nan(D["stoch"])
    max_th, where_th = argmax_nan(D["dth"])
    # SST1 classes
    R = np.load(a.ref)
    XY = R["XY"]; r = int(round(1.0 / (XY[1] - XY[0]))); e = XY[::r]
    occ = R["xy_n"][-1].sum(0).reshape(len(e) - 1, r, len(e) - 1, r).sum((1, 3))
    arr = alive[K]
    x82, y82 = S[K, :, 0], S[K, :, 1]
    ix = np.digitize(x82, e) - 1; iy = np.digitize(y82, e) - 1
    inside = arr & (ix >= 0) & (ix < len(e) - 1) & (iy >= 0) & (iy < len(e) - 1)
    o = np.where(inside, occ[np.clip(ix, 0, len(e) - 2), np.clip(iy, 0, len(e) - 2)], 0)
    cls = np.full(n, -1)                                         # -1: did not reach SST1
    cls[arr] = 2                                                  # isolated (incl. outside the grid)
    cls[arr & (o >= 10)] = 1
    cls[arr & (o >= 1000)] = 0
    names = {0: "common", 1: "low occupancy", 2: "isolated"}
    cnt = {names[c]: int((cls == c).sum()) for c in names}
    lines = [f"{n:,} source muons; reached SST1 {arr.sum():,}: " + ", ".join(f"{k} {v:,}" for k, v in cnt.items())]
    print(lines[0])
    rare = (cls == 1) | (cls == 2)
    com = cls == 0
    # features
    p0 = p[0]; th0 = 1e3 * np.hypot(tx[0], ty[0]); bend0 = 1e3 * tx[0] * q              # source direction (charge-signed x)
    shield = slice(0, 8)                                                                   # segments up to 29.89 m
    feats = {"source p": p0, "source angle": th0, "source x-slope x charge": bend0,
             "momentum lost in the shield": -np.nansum(D["dp"][shield], 0),
             "largest stochastic kick": max_st, "largest total deflection": max_th,
             "sum of stochastic kicks (shield)": np.nansum(D["stoch"][shield], 0),
             "displacement at shield end (|x|, |y|)": np.hypot(S[8, :, 0], S[8, :, 1])}
    for j in range(K):
        feats[f"stochastic kick {seg_names[j]} m"] = D["stoch"][j]
        feats[f"field bend {seg_names[j]} m"] = D["field"][j]
    lines.append(f"\nPredictive power for RARE (low + isolated, n = {rare.sum()}) vs COMMON (n = {com.sum():,}) arrivals at SST1: "
                 "ROC AUC (0.5 = none, 1 / 0 = perfect), bootstrap 90 %")
    res = {}
    for k, v in feats.items():
        ok = np.isfinite(v)
        if (rare & ok).sum() < 10:
            continue
        A, lo, hi = auc(v[rare & ok], v[com & ok], rng)
        res[k] = (A, lo, hi)
    for k, (A, lo, hi) in sorted(res.items(), key=lambda t: -abs(t[1][0] - 0.5)):
        lines.append(f"   {k:42s} AUC {A:.3f}  [{lo:.3f}, {hi:.3f}]")
    lines.append("\nWhere the largest stochastic kick happens (fraction of trajectories per segment): common / low / isolated")
    for j in range(K):
        fr = [np.mean(where_st[cls == c] == j) if (cls == c).any() else np.nan for c in (0, 1, 2)]
        lines.append(f"   {seg_names[j]:>12s} m   " + "  ".join(f"{100 * f:5.1f} %" for f in fr))
    lines.append("\nMedian per segment (all muons alive across it): |dtheta| total, field part, stochastic part [mrad]; dp [GeV]")
    for j in range(K):
        ok = np.isfinite(D["dth"][j])
        lines.append(f"   {seg_names[j]:>12s} m  n {ok.sum():9,}  total {np.median(D['dth'][j][ok]):7.2f}  field "
                     f"{np.median(D['field'][j][ok]):7.2f}  stoch {np.median(D['stoch'][j][ok]):7.2f}  dp {np.median(D['dp'][j][ok]):7.2f}")
    open(a.out + ".txt", "w").write("\n".join(lines) + "\n")
    json.dump({k: list(v) for k, v in res.items()}, open(a.out + "_auc.json", "w"), indent=1)
    print("\n".join(lines[1:]))
    # ---- figures
    zc = 0.5 * (zs[1:] + zs[:-1])
    col = {0: "#2a78d6", 1: "#c77700", 2: "#c0392b"}
    fig, axs = plt.subplots(2, 3, figsize=(27, 14), constrained_layout=True)
    ax = axs[0, 0]                                    # 1. deflection vs z per class
    for c in (0, 1, 2):
        m = cls == c
        if m.sum() < 5:
            continue
        for key, ls in (("stoch", "-"), ("field", "--")):
            med = [np.nanmedian(D[key][j][m]) if np.isfinite(D[key][j][m]).any() else np.nan for j in range(K)]
            p90 = [np.nanpercentile(D[key][j][m], 90) if np.isfinite(D[key][j][m]).any() else np.nan for j in range(K)]
            ax.plot(zc, med, ls, color=col[c], marker="o", ms=3, label=f"{names[c]} ({m.sum()}): {key} median")
            if key == "stoch":
                ax.plot(zc, p90, ":", color=col[c], lw=1, label=f"{names[c]}: stoch 90 %")
    ax.axvspan(0, 29.89, color="0.9", zorder=0)
    ax.set_yscale("log"); ax.set_xlabel("z (segment centre) [m]"); ax.set_ylabel("|delta theta| per segment [mrad]")
    ax.set_title("angular change per segment: field (dashed) and stochastic (solid), SST1 arrivals by class"); ax.legend(fontsize=7)
    ax = axs[0, 1]                                    # 2. deflection vs momentum
    pc = np.sqrt(PB[1:] * PB[:-1])
    for j, cc in zip((1, 3, 5, 7, 9, 12), plt.cm.viridis(np.linspace(0, 1, 6))):
        ok = np.isfinite(D["stoch"][j])
        ip = np.digitize(p[j][ok], PB) - 1
        med = [np.median(D["stoch"][j][ok][ip == i]) if (ip == i).sum() > 20 else np.nan for i in range(len(pc))]
        fmed = [np.median(D["field"][j][ok][ip == i]) if (ip == i).sum() > 20 else np.nan for i in range(len(pc))]
        ax.plot(pc, med, "-", color=cc, marker="o", ms=3, label=f"{seg_names[j]} m stoch")
        ax.plot(pc, fmed, "--", color=cc, lw=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("p at the segment start [GeV]"); ax.set_ylabel("median |delta theta| [mrad]")
    ax.set_title("deflection vs momentum (solid stochastic, dashed field), selected segments"); ax.legend(fontsize=8)
    ax = axs[0, 2]                                    # 3. AUC bars
    top = sorted(res.items(), key=lambda t: -abs(t[1][0] - 0.5))[:18][::-1]
    ax.barh(range(len(top)), [v[0] - 0.5 for _, v in top], left=0.5, color="#2a78d6",
            xerr=np.array([[v[0] - v[1] for _, v in top], [v[2] - v[0] for _, v in top]]), capsize=2)
    ax.set_yticks(range(len(top))); ax.set_yticklabels([k for k, _ in top], fontsize=8); ax.axvline(0.5, color="k", lw=0.8)
    ax.set_xlim(0, 1); ax.set_xlabel("ROC AUC, rare vs common SST1 arrivals (0.5 = no information)")
    ax.set_title(f"predictive power (rare n = {rare.sum()})")
    for ax, (key, lab, bins, logx) in zip((axs[1, 0], axs[1, 1], axs[1, 2]),
                                          ((p0, "source momentum [GeV]", np.geomspace(5, 400, 40), True),
                                           (max_st, "largest stochastic kick along the track [mrad]", np.geomspace(0.01, 1e3, 50), True),
                                           (where_st, "segment of the largest stochastic kick", np.arange(K + 1) - 0.5, False))):
        for c in (0, 1, 2):
            m = (cls == c) & np.isfinite(key)
            if m.sum() >= 5:
                ax.hist(key[m], bins, density=True, histtype="step", lw=2, color=col[c], label=f"{names[c]} ({m.sum()})")
        if logx:
            ax.set_xscale("log")
        else:
            ax.set_xticks(range(K)); ax.set_xticklabels(seg_names, rotation=60, fontsize=7)
        ax.set_xlabel(lab); ax.set_ylabel("density (per trajectory)"); ax.legend(fontsize=9)
    fig.suptitle("Are rare SST1 arrivals driven by unusual angular deflections in the shield? (analog tracks, Nov-4 pool)", fontsize=15)
    fig.savefig(a.out + ".png", dpi=65)
    np.savez(a.out + "_classes.npz", cls=cls, max_st=max_st, where_st=where_st, x82=x82, y82=y82)


if __name__ == "__main__":
    main()
