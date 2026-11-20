#!/usr/bin/env python3
"""Figures and tables of bin/sbt_firing.py (SBT walls fired by nu DIS, mu DIS, mu x mu; R2).

    python3 bin/sbt_firing_plots.py --dir $VETO_RESULTS/sbt_firing

Writes <dir>/figures/*.png and <dir>/summary.{md,json}. Sub-sources of one class are summed (weights are additive).
"""
import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from veto import sbt

CLASSES = {   # label -> sub-sources (files), fixed order and colour (categorical slots 1-3; muon flux in neutral ink)
    "nu DIS": (["nu_vessel", "nu_ms7", "nu_sbt"], "#2a78d6"),
    "mu DIS": (["mu_vessel", "mu_ms7", "mu_sbt_alin", "mu_sbt_ls", "mu_sbt_rib", "mu_sbt_alout"], "#eb6834"),
    "mu x mu": (["mumu"], "#1baf7a"),
}
SUB_NAMES = {"nu_vessel": "vessel (He, walls, air, planes)", "nu_ms7": "MS7 last 50 cm", "nu_sbt": "SBT material",
             "mu_vessel": "vessel (He, walls, air, planes)", "mu_ms7": "MS7 last 50 cm", "mu_sbt_alin": "SBT inner Al",
             "mu_sbt_ls": "SBT LS", "mu_sbt_rib": "SBT ribs", "mu_sbt_alout": "SBT outer Al", "mumu": "SST1 muon pairs"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ROWS = [(0, s) for s in range(4)] + [(1, s) for s in range(4)] + [(2, s) for s in range(3)] + [(3, s) for s in range(3)]


def style(ax):
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2)


def load(d, name):
    j, n = d / f"{name}.json", d / f"{name}.npz"
    if not j.exists():
        return None, None
    return json.load(open(j)), dict(np.load(n))


def class_sum(d, files, prefix):
    tot, parts = None, {}
    for f in files:
        j, a = load(d, f)
        if j is None or f"{prefix}_cellmap" not in a:
            continue
        parts[f] = (j, a)
        tot = {k: v.copy() for k, v in a.items() if k.startswith(prefix + "_")} if tot is None else \
            {k: tot[k] + a[k] for k in tot}
    return tot, parts


def combine_summary(parts, setname):
    keys = ("frac_fired_any", "frac_fired_upstream_of_vertex", "frac_fired_downstream_of_vertex", "frac_fired_intime_zone",
            "mean_fired_cells")
    W, acc, n_mc, by = 0.0, {k: 0.0 for k in keys}, 0, {}
    for f, (j, _) in parts.items():
        s = j["sets"].get(setname)
        if not s:
            continue
        W += s["events"]; n_mc += s["n_events_mc"]
        for k in keys:
            acc[k] += s[k] * s["events"]
        for c, v in s.get("frac_fired_by", {}).items():
            by[c] = by.get(c, 0.0) + v * s["events"]
    if W <= 0:
        return dict(events=0.0, n_events_mc=n_mc)
    return dict(events=W, n_events_mc=n_mc, **{k: v / W for k, v in acc.items()}, frac_fired_by={c: v / W for c, v in by.items()})


def unrolled(m):
    """(4, rings, 4) -> (14 rows: +x 4, -x 4, +y 3, -y 3; rings)."""
    return np.array([m[f, :, s] for f, s in ROWS])


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    d = Path(a.dir); fig_dir = d / "figures"; fig_dir.mkdir(exist_ok=True)
    edges = sbt.RING_EDGES; zc = 0.5 * (edges[1:] + edges[:-1])
    mj, mm = load(d, "muonmap")
    summary = dict(muon_flux=mj["sets"] if mj else None, classes={})

    # ---------------- F1: fired-cell rate per ring (Hz, spill 1 s), all faces, vs the muon-crossing map
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if mm is not None:
        r = mm["thr30_cellmap"].sum(axis=(0, 2))
        ax.step(edges, np.r_[r, r[-1]], where="post", color=INK2, lw=2, label="muons crossing the shell (rate now, x1)")
        ax.step(edges, 4 * np.r_[r, r[-1]], where="post", color=INK2, lw=1, ls=":", label="x4 (rate_scale of R2)")
    for name, (files, col) in CLASSES.items():
        pref = "muons" if name == "mu x mu" else "all"
        if name == "mu x mu":
            continue                                        # the SST1 muons are part of the muon-crossing map
        tot, _ = class_sum(d, files, pref)
        if tot is None:
            continue
        r = tot[f"{pref}_cellmap"].sum(axis=(0, 2))
        ax.step(edges, np.r_[r, r[-1]], where="post", color=col, lw=2, label=f"{name} products (all sources)")
    ax.set_yscale("log"); ax.set_xlabel("z [m] (cell ring)"); ax.set_ylabel("fired cells per ring [Hz]  (4e13 PoT / 1 s spill)")
    ax.set_title("SBT fired-cell rate per ring, R2, > 30 MeV", loc="left", color=INK)
    style(ax); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "rate_per_ring.png", dpi=150); plt.close(fig)

    # ---------------- F2: per-source rate per ring (small multiples per class)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, name in zip(axs, ("nu DIS", "mu DIS")):
        files, _ = CLASSES[name]
        _, parts = class_sum(d, files, "all")
        shades = ["#0d366b", "#2a78d6", "#86b6ef"] if name == "nu DIS" else ["#7a2d0c", "#eb6834", "#f2a07e", "#c98500", "#eda100", "#f6cf74"]
        for (f, (j, arr)), c in zip(parts.items(), shades):
            r = arr["all_cellmap"].sum(axis=(0, 2))
            ax.step(edges, np.r_[r, r[-1]], where="post", color=c, lw=1.6, label=f"{SUB_NAMES[f]}  ({r.sum():.3g} Hz)")
        ax.set_yscale("log"); ax.set_xlabel("z [m]"); ax.set_title(name, loc="left", color=INK); style(ax)
        ax.legend(frameon=False, fontsize=7.5)
    axs[0].set_ylabel("fired cells per ring [Hz]")
    fig.tight_layout(); fig.savefig(fig_dir / "rate_per_ring_sources.png", dpi=150); plt.close(fig)

    # ---------------- F3: unrolled wall maps (all events / muon map), and selected (partial)
    for setname, title in (("all", "all events, per spill"), ("sel_partial", "selected (IP < 2.5 m), per 15 y"),
                           ("sel_full", "selected (IP < 10 cm), per 15 y")):
        panels = []
        if setname == "all" and mm is not None:
            panels.append(("muons crossing the shell [Hz]", mm["thr30_cellmap"]))
        for name, (files, _) in CLASSES.items():
            pref = {"all": "muons", "sel_partial": "pairs_ip25", "sel_full": "pairs_ip01"}[setname] if name == "mu x mu" else setname
            tot, _ = class_sum(d, files, pref)
            if tot is not None:
                panels.append((f"{name}", tot[f"{pref}_cellmap"]))
        fig, axs = plt.subplots(len(panels), 1, figsize=(11, 2.0 * len(panels) + 0.6), squeeze=False)
        for ax, (lab, m) in zip(axs[:, 0], panels):
            u = unrolled(m)
            pos = u[u > 0]
            if not len(pos):
                ax.set_title(f"{lab}: no fired cell", loc="left", fontsize=9); ax.axis("off"); continue
            im = ax.pcolormesh(edges, np.arange(15), u, cmap="Blues", norm=LogNorm(max(pos.min(), pos.max() * 1e-5), pos.max()),
                               shading="flat")
            ax.set_yticks([2, 6, 9.5, 12.5]); ax.set_yticklabels(["+x", "-x", "+y", "-y"])
            for yl in (4, 8, 11):
                ax.axhline(yl, color=INK2, lw=0.6)
            ax.set_title(f"{lab}: {u.sum():.3g} fired cells", loc="left", fontsize=9, color=INK)
            fig.colorbar(im, ax=ax, pad=0.01)
        axs[-1, 0].set_xlabel("z [m]")
        fig.suptitle(f"SBT cells fired (R2), {title}", x=0.01, ha="left", color=INK)
        fig.tight_layout(); fig.savefig(fig_dir / f"wall_map_{setname}.png", dpi=150); plt.close(fig)

    # ---------------- F4: multiplicity, dz, edep (normalised shapes), all vs selected
    nc_bins = np.arange(0, 42); dz_bins = np.arange(-52, 53, 1.0); e_bins = np.logspace(1, 4, 31)
    fig, axs = plt.subplots(2, 3, figsize=(14, 7.5))
    for row, setname in enumerate(("all", "sel_partial")):
        for name, (files, col) in CLASSES.items():
            pref = ({"all": "muons", "sel_partial": "pairs_ip25"}[setname]) if name == "mu x mu" else setname
            tot, _ = class_sum(d, files, pref)
            if tot is None:
                continue
            nc = tot[f"{pref}_ncell"]
            if nc.sum() > 0:
                axs[row, 0].step(nc_bins, nc / nc.sum(), where="mid", color=col, lw=1.6, label=name)
            dz = tot[f"{pref}_dz"]
            if dz.sum() > 0 and not (name == "mu x mu" and setname == "all"):
                axs[row, 1].step(dz_bins[:-1], dz / dz.sum(), where="post", color=col, lw=1.6, label=name)
            e = tot[f"{pref}_edep"]
            if e.sum() > 0:
                axs[row, 2].step(e_bins[:-1], e / e.sum(), where="post", color=col, lw=1.6, label=name)
        lab = "all events" if setname == "all" else "selected, IP < 2.5 m"
        axs[row, 0].set_yscale("log"); axs[row, 0].set_xlabel("fired cells per event (40 = overflow)"); axs[row, 0].set_title(f"multiplicity, {lab}", loc="left", fontsize=10)
        axs[row, 1].set_yscale("log"); axs[row, 1].set_xlabel("z(cell) - z(vertex) [m]"); axs[row, 1].set_title(f"where (vs vertex), {lab}", loc="left", fontsize=10)
        axs[row, 2].set_xscale("log"); axs[row, 2].set_yscale("log"); axs[row, 2].set_xlabel("E deposited per fired cell [MeV]"); axs[row, 2].set_title(f"energy per fired cell, {lab}", loc="left", fontsize=10)
        for ax in axs[row]:
            style(ax); ax.set_ylabel("fraction (weighted)"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "shapes.png", dpi=150); plt.close(fig)

    # ---------------- tables
    lines = ["# SBT walls fired by nu DIS, mu DIS, mu x mu (R2, own deposits only)", ""]
    if mj:
        lines += [f"Muons crossing the shell (random-hit source now): {mj['sets']['thr30']['total_hz'] / 1e6:.3g} MHz > 30 MeV, "
                  f"hottest cell {mj['sets']['thr30']['max_cell_hz'] / 1e3:.3g} kHz (x4 in R2).", ""]
    lines += ["## Rate (all events, products of the class, per spill = Hz)", "",
              "| class | source | interactions / spill | fired cells / spill [kHz] | hottest cell [kHz] | events with >= 1 fired cell |",
              "|---|---|---|---|---|---|"]
    for name, (files, _) in CLASSES.items():
        if name == "mu x mu":
            continue
        for f in files:
            j, arr = load(d, f)
            if j is None:
                continue
            s = j["sets"]["all"]
            prod = arr["all_cellmap_comp"][0]
            summary["classes"].setdefault(name, {})[f] = dict(all=s, products_hz=float(prod.sum()),
                                                            sel_partial=j["sets"].get("sel_partial"), sel_full=j["sets"].get("sel_full"))
            lines.append(f"| {name} | {SUB_NAMES[f]} | {s['interactions_per_spill']:.3g} | {s['fired_cells_total'] / 1e3:.3g} "
                         f"(products {prod.sum() / 1e3:.3g}) | {arr['all_cellmap'].max() / 1e3:.3g} | {s['frac_fired_any']:.3f} |")
    lines += ["", "## Response of the selected events (before tagging, per 15 y)", "",
              "| class | selection | events / 15 y (MC) | >= 1 fired cell | by products / scattered mu / parent mu | upstream of vertex | in time, zone | mean fired cells |",
              "|---|---|---|---|---|---|---|---|"]
    for name, (files, _) in CLASSES.items():
        for setname, lab in (("sel_partial", "IP < 2.5 m"), ("sel_full", "IP < 10 cm")):
            if name == "mu x mu":
                j, _ = load(d, "mumu")
                if j is None:
                    continue
                s = j["sets"]["pairs_ip25" if setname == "sel_partial" else "pairs_ip01"]
                byc = f"leg a {s['frac_leg_a_fired']:.3f} / leg b {s['frac_leg_b_fired']:.3f} / both {s['frac_both_legs_fired']:.3f}"
                zone = "-"
            else:
                _, parts = class_sum(d, files, setname)
                s = combine_summary(parts, setname)
                if not s.get("events"):
                    lines.append(f"| {name} | {lab} | 0 ({s['n_events_mc']}) | | | | | |"); continue
                byc = " / ".join(f"{s['frac_fired_by'].get(c, 0):.3f}" for c in ("products", "scattered_mu", "parent_mu"))
                zone = f"{s['frac_fired_intime_zone']:.3f}"
            summary["classes"].setdefault(name, {})[f"combined_{setname}"] = s
            lines.append(f"| {name} | {lab} | {s['events']:.3g} ({s['n_events_mc']}) | {s['frac_fired_any']:.3f} | {byc} | "
                         f"{s['frac_fired_upstream_of_vertex']:.3f} | {zone} | {s['mean_fired_cells']:.2g} |")
    (d / "summary.md").write_text("\n".join(lines) + "\n")
    json.dump(summary, open(d / "summary.json", "w"), indent=1, default=float)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
