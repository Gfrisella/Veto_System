#!/usr/bin/env python3
"""Validation and rare-region coverage of splitting.py runs (sandbox).

    python3 analyse.py --ref analog_big.npz --split split.npz --analog analog_eq.npz --out report [--spills 1e7]

  1. unbiased?    live weight per checkpoint, split and equal-cost analog against the big analog reference (batch errors)
  2. regions      the SST plane (z = 82 m) in REGION x REGION m cells: rate, batch error, rows, distinct source muons
  3. coverage     regions with entries, relative errors, figure of merit 1 / (rel. error^2 x cost) split / analog
  4. agreement    pulls split vs reference in regions both see
  5. limits       expected crossings in `spills` (default 1e7 = 10 years of 4e13 p.o.t. spills) per region: the estimate and
                  a 95 % upper limit; for EMPTY regions 3.0 x (weight a hit there would carry, from the neighbouring
                  regions) / batches; regions fed by < 5 distinct source muons flagged (rare transitions may be missed)
Writes <out>.txt, <out>.json, <out>.png.
"""
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

REGION = 1.0                                   # m


def load(f):
    d = np.load(f, allow_pickle=False)
    return {k: d[k] for k in d.files}


def regions(d, k):
    """per region (B, nx, ny): rate estimates of each batch at checkpoint k; rows"""
    r = int(round(REGION / (d["XY"][1] - d["XY"][0])))
    s1, n = d["xy_s1"][k], d["xy_n"][k]
    B, nx, ny = s1.shape
    s1 = s1.reshape(B, nx // r, r, ny // r, r).sum((2, 4))
    n = n.reshape(B, nx // r, r, ny // r, r).sum((2, 4))
    return s1, n


def est(s1):
    """batches -> (mean, error of the mean)"""
    B = s1.shape[0]
    return s1.mean(0), s1.std(0, ddof=1) / np.sqrt(B)


def roots_per_region(d):
    x, y, root = d["final_x"], d["final_y"], d["final_root"]
    e = d["XY"][::int(round(REGION / (d["XY"][1] - d["XY"][0])))]
    ix, iy = np.digitize(x, e) - 1, np.digitize(y, e) - 1
    ok = (ix >= 0) & (ix < len(e) - 1) & (iy >= 0) & (iy < len(e) - 1)
    key = (ix[ok] * (len(e) - 1) + iy[ok]) * (int(root.max()) + 1) + root[ok]
    cell = np.unique(key) // (int(root.max()) + 1)
    return np.bincount(cell, minlength=(len(e) - 1) ** 2).reshape(len(e) - 1, len(e) - 1), e


def hit_weight(d):
    """per region: median weight of the final rows there (what one hit contributes), filled for empty regions from the
    3 x 3 neighbourhood, else the global maximum (conservative)"""
    x, y, w = d["final_x"], d["final_y"], d["final_w"]
    e = d["XY"][::int(round(REGION / (d["XY"][1] - d["XY"][0])))]
    n = len(e) - 1
    ix, iy = np.digitize(x, e) - 1, np.digitize(y, e) - 1
    ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    med = np.full((n, n), np.nan)
    for i, j in set(zip(ix[ok].tolist(), iy[ok].tolist())):
        med[i, j] = np.median(w[ok][(ix[ok] == i) & (iy[ok] == j)])
    out = med.copy()
    for i in range(n):
        for j in range(n):
            if np.isnan(out[i, j]):
                nb = med[max(0, i - 1):i + 2, max(0, j - 1):j + 2]
                out[i, j] = np.nanmax(nb) if np.isfinite(nb).any() else np.nanmax(w)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True); ap.add_argument("--split", required=True); ap.add_argument("--analog", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--spills", type=float, default=1e7)
    a = ap.parse_args()
    R, S, A = load(a.ref), load(a.split), load(a.analog)
    lines, js = [], {}
    pr = lambda s="": (lines.append(s), print(s))
    cost = {n: float(d["cost"].sum()) for n, d in (("ref", R), ("split", S), ("analog", A))}
    pr(f"budget (transported row-metres): reference {cost['ref']:.3g}, split {cost['split']:.3g}, analog {cost['analog']:.3g}")
    # 1. unbiasedness of the live weight per checkpoint
    pr("\n1. live muons per spill at each checkpoint (mean of batches +- error); pulls against the reference")
    pr(f"{'z [m]':>7s} {'reference':>22s} {'split':>22s} {'pull':>6s} {'analog (equal cost)':>22s} {'pull':>6s}")
    js["alive"] = []
    for k, z in enumerate(R["checkpoints"]):
        m = {n: est(d["alive_w"][k][:, None])[0][0] for n, d in (("r", R), ("s", S), ("a", A))}
        e = {n: est(d["alive_w"][k][:, None])[1][0] for n, d in (("r", R), ("s", S), ("a", A))}
        ps, pa = (m["s"] - m["r"]) / np.hypot(e["s"], e["r"]), (m["a"] - m["r"]) / np.hypot(e["a"], e["r"])
        pr(f"{z:7.2f} {m['r']:12.4g} +- {e['r']:7.2g} {m['s']:12.4g} +- {e['s']:7.2g} {ps:6.2f} {m['a']:12.4g} +- {e['a']:7.2g} {pa:6.2f}")
        js["alive"].append(dict(z=float(z), ref=m["r"], split=m["s"], analog=m["a"], pull_split=ps, pull_analog=pa))
    # 2-4. regions at the SST plane
    K = len(R["checkpoints"]) - 1
    (rs, rn), (ss, sn), (as_, an) = regions(R, K), regions(S, K), regions(A, K)
    rm, re_ = est(rs); sm, se = est(ss); am, ae = est(as_)
    roots_s, edges = roots_per_region(S)
    roots_a, _ = roots_per_region(A)
    seen = lambda m, n: (m > 0) & (n.sum(0) > 0)
    pr(f"\n2-3. SST plane, {REGION:g} m regions over +-{edges[-1]:g} m ({rm.size} regions)")
    for name, m, e, n, ro, c in (("reference", rm, re_, rn, None, cost["ref"]), ("split", sm, se, sn, roots_s, cost["split"]),
                                 ("analog (equal cost)", am, ae, an, roots_a, cost["analog"])):
        v = seen(m, n)
        rel = e[v] / m[v]
        pr(f"   {name:20s} regions with entries {v.sum():4d}; rel. error median {np.median(rel):.3f}, worst 10 % "
           f"{np.percentile(rel, 90):.3f}" + (f"; regions fed by < 5 source muons {int(((ro < 5) & v).sum())}" if ro is not None else ""))
    both = seen(sm, sn) & seen(am, an)
    # figure of merit 1 / (relative variance x cost): split over analog
    fom = ((ae[both] / am[both]) ** 2 * cost["analog"]) / ((se[both] / sm[both]) ** 2 * cost["split"])
    pr(f"   figure of merit split / analog (1 / (rel. error^2 x cost)) in the {both.sum()} regions both see: median "
       f"{np.median(fom):.2f}, 10 % {np.percentile(fom, 10):.2f}, 90 % {np.percentile(fom, 90):.2f}")
    only_s = seen(sm, sn) & ~seen(am, an)
    pr(f"   regions seen by split only: {only_s.sum()} (rate there {sm[only_s].sum():.3g} per spill); by analog only: "
       f"{(seen(am, an) & ~seen(sm, sn)).sum()}")
    ok = seen(sm, sn) & seen(rm, rn) & (re_ > 0) & (se > 0)
    pull = (sm[ok] - rm[ok]) / np.hypot(se[ok], re_[ok])
    pr(f"\n4. split vs reference in {ok.sum()} regions: total ratio {sm[ok].sum() / rm[ok].sum():.4f}; pull mean {pull.mean():.2f}, "
       f"rms {pull.std():.2f}; |pull| > 3: {(np.abs(pull) > 3).sum()}")
    # 5. limits on lifetime crossings
    B = S["xy_s1"].shape[1]
    w_hit = hit_weight(S) / B
    ul = np.where(seen(sm, sn), sm + 1.645 * se, 3.0 * w_hit) * a.spills
    pr(f"\n5. crossings in {a.spills:g} spills per {REGION:g} m region at the SST plane (split run):")
    empty = ~seen(sm, sn)
    pr(f"   {empty.sum()} empty regions: 95 % upper limits median {np.median(ul[empty]):.3g}, max {ul[empty].max():.3g}, "
       f"min {ul[empty].min():.3g}")
    pr(f"   regions with an upper limit below 1 crossing in {a.spills:g} spills: {(ul < 1).sum()} of {ul.size}")
    js.update(cost=cost, fom_median=float(np.median(fom)), pull_mean=float(pull.mean()), pull_rms=float(pull.std()),
              empty=int(empty.sum()), seen_split=int(seen(sm, sn).sum()), seen_analog=int(seen(am, an).sum()))
    open(a.out + ".txt", "w").write("\n".join(lines) + "\n")
    json.dump(js, open(a.out + ".json", "w"), indent=1, default=float)
    # figure
    fig, axs = plt.subplots(2, 4, figsize=(26, 12), constrained_layout=True)
    ext = [edges[0], edges[-1], edges[0], edges[-1]]
    norm = LogNorm(max(min(rm[rm > 0].min(), sm[sm > 0].min()), 1e-6), max(rm.max(), sm.max()))
    for ax, m, t in ((axs[0, 0], rm, "reference analog: muons / spill"), (axs[0, 1], sm, "split: muons / spill"),
                     (axs[0, 2], am, "analog, equal cost: muons / spill")):
        im = ax.imshow(np.where(m > 0, m, np.nan).T, origin="lower", extent=ext, norm=norm, cmap="viridis")
        ax.set_title(t); fig.colorbar(im, ax=ax)
    for ax, m, e, t in ((axs[1, 0], am, ae, "analog equal cost: relative error"), (axs[1, 1], sm, se, "split: relative error")):
        im = ax.imshow(np.where(m > 0, e / np.where(m > 0, m, 1), np.nan).T, origin="lower", extent=ext, norm=LogNorm(0.01, 1),
                       cmap="magma_r")
        ax.set_title(t); fig.colorbar(im, ax=ax)
    im = axs[1, 2].imshow(np.where(roots_s > 0, roots_s, np.nan).T, origin="lower", extent=ext, norm=LogNorm(1, None), cmap="cividis")
    axs[1, 2].set_title("split: distinct source muons per region"); fig.colorbar(im, ax=axs[1, 2])
    im = axs[0, 3].imshow(ul.T, origin="lower", extent=ext, norm=LogNorm(), cmap="inferno")
    axs[0, 3].set_title(f"95 % upper limit, crossings in {a.spills:g} spills (empty: from hit weight)"); fig.colorbar(im, ax=axs[0, 3])
    z = R["checkpoints"]
    for d, lab in ((R, "reference"), (S, "split"), (A, "analog equal cost")):
        axs[1, 3].errorbar(z, d["alive_w"].mean(1), yerr=d["alive_w"].std(1, ddof=1) / np.sqrt(d["alive_w"].shape[1]), label=lab,
                           marker="o", ms=3, capsize=2)
    axs[1, 3].set_yscale("log"); axs[1, 3].set_xlabel("z [m]"); axs[1, 3].set_ylabel("live muons / spill"); axs[1, 3].legend()
    axs[1, 3].set_title("live weight per checkpoint")
    for ax in axs.ravel()[:7]:
        if ax is not axs[1, 3]:
            ax.add_patch(plt.Rectangle((-2, -3), 4, 6, fill=False, ec="w", lw=1)); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.suptitle("Splitting prototype: SST plane (z = 82 m), 1 m regions", fontsize=15)
    fig.savefig(a.out + ".png", dpi=70)


if __name__ == "__main__":
    main()
