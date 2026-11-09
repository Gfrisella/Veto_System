#!/usr/bin/env python3
"""The input pool's spectra against the enriched file, per component: before (our generators, raw) and after the
normalisation (the pool), with the source boundaries. From <pool>/edges.json (bin/pool_edges.py).

    python3 bin/plot_pool_spectra.py results/pool_v1/edges.json [results/pool_v1/source_timing.json]
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#555555"


def main():
    d = json.load(open(sys.argv[1]))["spectra"]
    timing = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None
    e = np.array(d["edges"])
    c, w = np.sqrt(e[1:] * e[:-1]), np.diff(e)
    F, P = np.array(d["file"]), np.array(d["pool"])
    raw = np.zeros_like(P)                                   # our generators before the flux factor (fresh ranges)
    fresh_ranges = {}
    for tag, p in d["parts"].items():
        lo, hi = p["p_range"]
        m = (c >= lo) & (c < hi)
        if "bin_factors" in p:                               # pool v2: a factor per 4 % bin (bin/build_pool.py SHAPE_BINS)
            fb = np.array(p["bin_factors"])
            fac = fb[np.digitize(c, np.geomspace(10.0, 1000.0, 121))]
        else:
            fac = p["flux_factor"]
        raw[p["component"]][m] = (P[p["component"]] / fac)[m]
        fresh_ranges.setdefault(p["component"], []).append((lo, hi, tag, p["flux_factor"]))
    panels = (("min. bias (+ tail above 150 GeV)", (1, 0)), ("dimuon class (weight 7.69)", (2,)), ("charm", (3,)))
    fig, axs = plt.subplots(2, 3, figsize=(18, 9), gridspec_kw=dict(height_ratios=[3, 2]), sharex=True)
    for j, (title, comps) in enumerate(panels):
        f, p, r = sum(F[k] for k in comps), sum(P[k] for k in comps), sum(raw[k] for k in comps)
        old = r == 0
        ax, bx = axs[0, j], axs[1, j]
        ax.step(c, f / w, where="mid", color=GREY, lw=2.5, label="the 601 M file (what a and b use)")
        ax.step(c, np.where(old, np.nan, r / w), where="mid", color=ORANGE, lw=2, ls="--", label="our fresh muons, raw")
        ax.step(c, p / w, where="mid", color=BLUE, lw=2, label="pool (fresh x flux factor, old rows below)")
        with np.errstate(divide="ignore", invalid="ignore"):
            bx.step(c, np.where(old, np.nan, r / f), where="mid", color=ORANGE, lw=2, ls="--", label="raw / file")
            bx.step(c, p / f, where="mid", color=BLUE, lw=2, label="pool / file")
        if j == 0 and timing is not None:                   # the earlier, validated min. bias run (mb2), coarse bins
            t = timing["components"]["min. bias"]
            pc = 10 ** (np.array(timing["logp"]) + 0.05)
            o, fl = np.array(t["ours_flux"]), np.array(t["file_flux"])
            m = (fl > 0) & (pc >= 15) & (pc < 150)
            bx.plot(pc[m], o[m] / fl[m], "o", color=GREEN, ms=7, label="earlier run mb2 (raw / file)")
        for comp in comps:
            for lo, hi, tag, fac in fresh_ranges.get(comp, []):
                for x in (lo, hi):
                    if 10 < x < 400:
                        for a in (ax, bx):
                            a.axvline(x, color="k", ls=":", lw=1)
                bx.text(np.sqrt(max(lo, 10) * min(hi, 400)), 0.62, f"fresh {tag}\nx {fac:.2f}", ha="center", fontsize=8)
        ax.set_yscale("log"); ax.set_xscale("log"); ax.set_title(title); ax.grid(alpha=0.3)
        ax.set_ylabel("muons per spill per GeV")
        bx.axhline(1, color="k", lw=1); bx.set_ylim(0.55, 1.9); bx.grid(alpha=0.3)
        bx.set_xlabel("momentum at production [GeV]"); bx.set_ylabel("ratio to the file")
        ax.legend(fontsize=8); bx.legend(fontsize=8, loc="upper right")
    fig.suptitle(sys.argv[3] if len(sys.argv) > 3 else "Input pool vs the 601 M input file (dotted: where the source changes)",
                 fontsize=11)
    fig.tight_layout()
    out = Path(sys.argv[1]).with_name("pool_spectra.png")
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()
