#!/usr/bin/env python3
"""Figures: target-production-complete-november-4-2026 against the 506 M 2018 FairShip sample, from the histograms of
bin/compare_new_vs_506M.py (production momentum, the 2018 selection: p > 25 GeV and the 2018 veto plane emulated; both in
muons per spill of 4e13 p.o.t.).

    python3 bin/plot_new_vs_506M.py results/target_talk/new_selected.npz results/target_talk/histograms_506M_4e13.npz OUTDIR
Writes OUTDIR/{spectra,maps,statistics,parents}.png and OUTDIR/summary.txt.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

COMPS = {"all": ("minbias", "onia", "charm", "beauty"), "min. bias": ("minbias",), "quarkonia": ("onia",),
         "charm": ("charm",), "beauty": ("beauty",)}
NEW_C, REF_C = "#2a78d6", "#5b6475"


def h(src, comps, lab, k, charge=("mu-", "mu+")):
    out = None
    for c in comps:
        for ch in charge:
            key = f"{c}|{lab}_{ch}_2d_{k}"
            if key in src.files:
                out = src[key] if out is None else out + src[key]
    return out


def main():
    new, ref, out = np.load(sys.argv[1]), np.load(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    PE, TE, pmin = new["PE"], new["TE"], float(new["p_min"])
    pc = np.sqrt(PE[1:] * PE[:-1]); tc = 0.5 * (TE[1:] + TE[:-1]); sel = pc >= pmin
    lines = [f"new: {Path(sys.argv[1]).name} (pot per spill {float(new['pot']):.3g}), 506 M: {Path(sys.argv[2]).name} "
             f"(pot per spill {float(ref['pot']):.3g}); production p > {pmin:.0f} GeV, 2018 selection", ""]
    lines.append(f"{'component':10s} {'new [/spill]':>13s} {'w. 2018 weighting':>18s} {'506 M':>11s} {'new/506M':>9s} "
                 f"{'(2018 wt)':>9s}   {'rows new':>10s} {'rows 506M':>10s} {'n_eff new':>10s} {'n_eff 506M':>10s}")
    for name, cs in COMPS.items():
        a, a2, an = (h(new, cs, "phys", k)[sel].sum() for k in ("s1", "s2", "n"))
        b, b2, bn = (h(ref, cs, "ref", k)[sel].sum() for k in ("s1", "s2", "n"))
        a18 = h(new, cs, "w2018", "s1")[sel].sum()
        lines.append(f"{name:10s} {a:13.4g} {a18:18.4g} {b:11.4g} {a / b:9.3f} {a18 / b:9.3f}   {an:10.3g} {bn:10.3g} "
                     f"{a * a / a2:10.3g} {b * b / b2:10.3g}")

    # 1. spectra in p and pT per component, with the ratio
    fig, axs = plt.subplots(4, 5, figsize=(26, 15), gridspec_kw=dict(height_ratios=[3, 1.3, 3, 1.3]), constrained_layout=True)
    for j, (name, cs) in enumerate(COMPS.items()):
        for row, var in ((0, "p"), (2, "pT")):
            a1, a2 = h(new, cs, "phys", "s1"), h(new, cs, "phys", "s2")
            q1 = h(new, cs, "w2018", "s1")
            b1, b2 = h(ref, cs, "ref", "s1"), h(ref, cs, "ref", "s2")
            if var == "p":
                x, w_ = pc[sel], np.diff(PE)[sel]
                A, A2, Q, B, B2 = (v.sum(1)[sel] for v in (a1, a2, q1, b1, b2))
            else:
                x, w_ = tc, np.diff(TE)
                A, A2, Q, B, B2 = (v[sel].sum(0) for v in (a1, a2, q1, b1, b2))
            ax = axs[row, j]
            ax.step(x, B / w_, where="mid", color=REF_C, lw=2, label="506 M sample (2018)")
            ax.step(x, A / w_, where="mid", color=NEW_C, lw=2, label="new sample")
            ax.step(x, Q / w_, where="mid", color=NEW_C, lw=1, ls="--", label="new, 2018 weighting")
            ax.set_yscale("log"); ax.set_title(f"{name}: {var}", fontsize=12)
            ax.set_ylabel("muons / spill / GeV"); ax.grid(alpha=0.3)
            if j == 0 and row == 0:
                ax.legend(fontsize=9)
            r = axs[row + 1, j]
            with np.errstate(divide="ignore", invalid="ignore"):
                rat = A / B; err = rat * np.sqrt(A2 / A ** 2 + B2 / B ** 2); rq = Q / B
            ok = (A > 0) & (B > 0) & (err < 0.3 * rat)
            r.errorbar(x[ok], rat[ok], yerr=err[ok], fmt="o", ms=3, color=NEW_C)
            r.plot(x[ok], rq[ok], "--", color=NEW_C, lw=1)
            r.axhline(1, color="k", lw=0.8); r.set_ylim(0.3, 1.9); r.grid(alpha=0.3)
            r.set_ylabel("new / 506 M"); r.set_xlabel("production p [GeV]" if var == "p" else "production pT [GeV] (p > 25 GeV)")
    fig.suptitle("Muons after the target: target-production-complete-november-4-2026 vs the 506 M 2018 sample "
                 "(per spill of 4e13 p.o.t., 2018 selection)", fontsize=15)
    fig.savefig(out / "spectra.png", dpi=80); plt.close(fig)

    # 2. 2D rate ratio maps: physical and with the 2018 weighting, all components and min. bias
    fig, axs = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    for i, (lab, title) in enumerate((("phys", "physical weights"), ("w2018", "FairShip 2018 weighting emulated"))):
        for j, name in enumerate(("all", "min. bias")):
            cs = COMPS[name]
            o, o2 = h(new, cs, lab, "s1"), h(new, cs, lab, "s2")
            r1, r2 = h(ref, cs, "ref", "s1"), h(ref, cs, "ref", "s2")
            with np.errstate(divide="ignore", invalid="ignore"):
                z = o / r1
                bad = ~((o > 0) & (r1 > 0) & (np.sqrt(o2) / o < 0.3) & (np.sqrt(r2) / r1 < 0.3))
            z[bad] = np.nan; z[~sel] = np.nan
            ax = axs[i, j]
            m = ax.pcolormesh(PE, TE, z.T, cmap="RdBu_r", norm=TwoSlopeNorm(1.0, 0.4, 1.6), shading="flat")
            ax.set_facecolor("0.85"); ax.set_xlim(pmin, 400)
            ax.set_xlabel("production p [GeV]"); ax.set_ylabel("production pT [GeV]")
            ax.set_title(f"{name}, {title}: new / 506 M"); fig.colorbar(m, ax=ax)
    fig.suptitle("Rate ratio per cell (grey: either side above 30 % error)", fontsize=14)
    fig.savefig(out / "maps.png", dpi=80); plt.close(fig)

    # 3. statistics: n_eff per cell, new and 506 M, and their ratio, all components and quarkonia + charm
    fig, axs = plt.subplots(2, 3, figsize=(24, 12), constrained_layout=True)
    for i, (name, cs) in enumerate((("all", COMPS["all"]), ("quarkonia + charm", ("onia", "charm")))):
        o1, o2 = h(new, cs, "phys", "s1"), h(new, cs, "phys", "s2")
        r1, r2 = h(ref, cs, "ref", "s1"), h(ref, cs, "ref", "s2")
        with np.errstate(divide="ignore", invalid="ignore"):
            no = np.where(o2 > 0, o1 ** 2 / o2, np.nan); nr = np.where(r2 > 0, r1 ** 2 / r2, np.nan)
        no[~sel] = np.nan; nr[~sel] = np.nan
        for j, (z, title, norm, cmap) in enumerate(((no, "n_eff, new", LogNorm(1, 1e8), "viridis"),
                                                   (nr, "n_eff, 506 M", LogNorm(1, 1e8), "viridis"),
                                                   (no / nr, "n_eff new / 506 M", LogNorm(0.01, 100), "PiYG"))):
            ax = axs[i, j]
            m = ax.pcolormesh(PE, TE, z.T, norm=norm, cmap=cmap, shading="flat")
            ax.set_facecolor("0.85"); ax.set_xlim(pmin, 400)
            ax.set_xlabel("production p [GeV]"); ax.set_ylabel("production pT [GeV]"); ax.set_title(f"{name}: {title}")
            fig.colorbar(m, ax=ax)
    fig.suptitle("Effective number of muons per cell, (sum w)^2 / sum w^2 (2018 selection)", fontsize=14)
    fig.savefig(out / "statistics.png", dpi=80); plt.close(fig)

    # 4. parents: muons per spill by parent group, and mu+ / mu-
    groups = json.loads(str(new["groups"]))
    fig, axs = plt.subplots(1, 2, figsize=(20, 7), constrained_layout=True)
    a = [sum(float(new[f"{c}|phys_grp_{i}_s1"][0]) for c in COMPS["all"] if f"{c}|phys_grp_{i}_s1" in new.files) for i in range(len(groups))]
    b = [sum(float(ref[f"{c}|ref_grp_{i}_s1"][0]) for c in COMPS["all"] if f"{c}|ref_grp_{i}_s1" in ref.files) for i in range(len(groups))]
    xx = np.arange(len(groups))
    axs[0].bar(xx - 0.2, b, 0.4, color=REF_C, label="506 M"); axs[0].bar(xx + 0.2, a, 0.4, color=NEW_C, label="new")
    for k in range(len(groups)):
        if b[k] > 0:
            axs[0].text(xx[k], max(a[k], b[k]) * 1.3, f"{a[k] / b[k]:.2f}", ha="center", fontsize=10)
    axs[0].set_xticks(xx); axs[0].set_xticklabels(groups, rotation=20); axs[0].set_yscale("log")
    axs[0].set_ylabel("muons / spill (p > 25 GeV)"); axs[0].legend(); axs[0].set_title("by parent (numbers: new / 506 M)")
    for src, lab, col, nm in ((ref, "ref", REF_C, "506 M"), (new, "phys", NEW_C, "new")):
        mp = h(src, COMPS["all"], lab, "s1", ("mu+",)).sum(1); mm = h(src, COMPS["all"], lab, "s1", ("mu-",)).sum(1)
        ok = sel & (mp > 0) & (mm > 0)
        axs[1].plot(pc[ok], mp[ok] / mm[ok], "o-", color=col, label=nm)
    axs[1].axhline(1, color="k", lw=0.8, ls=":"); axs[1].set_xlabel("production p [GeV]"); axs[1].set_ylabel("mu+ / mu-")
    axs[1].legend(); axs[1].grid(alpha=0.3); axs[1].set_title("charge ratio")
    fig.savefig(out / "parents.png", dpi=80); plt.close(fig)
    for i, g in enumerate(groups):
        lines.append(f"parent {g:24s}: new {a[i]:11.4g}  506 M {b[i]:11.4g}  ratio {a[i] / b[i] if b[i] else float('nan'):.3f}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
