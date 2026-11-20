#!/usr/bin/env python3
"""Figures of bin/sbt_firing_hits.py (R2, own hits only):

  hits_mumu_topview.png   top view x-z of the +x / -x walls: muon paths of the IP < 10 cm mu x mu pairs (subsample) and
                          their first entry point into the SBT shell (filled: that cell fires)
  hits_mumu_walls.png     the two side walls unfolded (z along the beam, y up the wall): weighted density of the entry
                          points of the pair muons, IP < 2.5 m and < 10 cm, mu+ and mu- legs
  hits_<cls>_walls.png    nu / mu DIS selected events (IP < 2.5 m): fired SBT cells on the +x / -x walls (per 15 y)
  hits_<cls>_ubt.png      the same events: UBT crossings in x-y (straws 30.73 m, tiles 31.13 m)
  hits_seen_by.png        fraction of the selected events seen by SBT and UBT, SBT only, UBT only, none, per source

    python3 bin/sbt_firing_hits_plots.py --dir $VETO_RESULTS/sbt_firing
"""
import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import numpy as np

from veto import design as dsg
from veto import geometry as geo
from veto import sbt

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_POS, C_NEG = "#2a78d6", "#eb6834"              # mu+ / mu- (categorical slots 1, 2)
SEEN = (("sbt_and_ubt", "SBT and UBT", "#2a78d6"), ("sbt_only", "SBT only", "#1baf7a"),
        ("ubt_only", "UBT only", "#eda100"), ("none", "neither", "#e34948"))
SOURCES = {"nu DIS": ["nu_vessel", "nu_ms7", "nu_sbt"],
           "mu DIS": ["mu_vessel", "mu_ms7", "mu_sbt_alin", "mu_sbt_ls", "mu_sbt_rib", "mu_sbt_alout"]}
NAMES = {"nu_vessel": "ν vessel", "nu_ms7": "ν MS7", "nu_sbt": "ν SBT material", "mu_vessel": "μ vessel",
         "mu_ms7": "μ MS7", "mu_sbt_alin": "μ SBT inner Al", "mu_sbt_ls": "μ SBT LS", "mu_sbt_rib": "μ SBT ribs",
         "mu_sbt_alout": "μ SBT outer Al"}
Z = np.linspace(geo.Z_VESSEL_MIN, geo.Z_VESSEL_MAX, 200)


def style(ax):
    ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2)


def wall_outline(ax):
    hx, hy = geo.vessel_half_widths(Z)
    ax.plot(Z, hy, color=INK2, lw=0.8); ax.plot(Z, -hy, color=INK2, lw=0.8)
    for e in sbt.RING_EDGES[::6]:
        h = geo.vessel_half_widths(np.array([e]))[1][0]
        ax.plot([e, e], [-h, h], color=GRID, lw=0.5)


def topview(d, fig_dir):
    h = dict(np.load(d / "hits_mumu.npz"))
    m = np.nonzero(h["ip"] < 0.10)[0]
    rng = np.random.default_rng(3)
    show = rng.choice(m, min(60, len(m)), replace=False, p=h["w15"][m] / h["w15"][m].sum())
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    hx, _ = geo.vessel_half_widths(Z)
    for sgn in (1, -1):
        ax.fill_between(Z, sgn * hx, sgn * (hx + sbt.T_SHELL), color="#cde2fb", lw=0)
        ax.plot(Z, sgn * hx, color=INK2, lw=0.8)
    pr = h["path_rows"]
    for i in show:
        for leg, col in (("a", C_POS), ("b", C_NEG)):
            j = np.searchsorted(pr, h[f"{leg}_row"][i])
            ax_, az = h["path_ax"][j], h["path_az"][j]
            kx, kz = h["path_kx"][j], h["path_kz"][j]
            bx, bz = h["path_bx"][j], h["path_bz"][j]
            xs, zs = ([ax_, kx, bx], [az, kz, bz]) if np.isfinite(kz) else ([ax_, bx], [az, bz])
            ax.plot(zs, xs, color=col, lw=0.5, alpha=0.5)
            if h[f"{leg}_has"][i]:
                f = h[f"{leg}_fired"][i]
                ax.plot(h[f"{leg}_z"][i], h[f"{leg}_x"][i], "o", ms=4.5, mec=col, mfc=col if f else "white", mew=1.0)
        ax.plot(h["vtx"][i, 2], h["vtx"][i, 0], "x", color=INK, ms=4, mew=0.8)
    ax.set_xlim(29, 84); ax.set_ylim(-4.5, 4.5)
    ax.set_xlabel("z [m]"); ax.set_ylabel("x [m]")
    ax.set_title(f"μ × μ pairs, IP < 10 cm ({len(show)} of {len(m):,} MC pairs, drawn by weight): paths and SBT entry points",
                 loc="left", color=INK, fontsize=10)
    ax.plot([], [], color=C_POS, label="μ+ leg"); ax.plot([], [], color=C_NEG, label="μ− leg")
    ax.plot([], [], "o", color=INK2, mfc=INK2, label="entry, cell fires"); ax.plot([], [], "o", color=INK2, mfc="white", label="entry, below threshold")
    ax.plot([], [], "x", color=INK, label="pair vertex")
    ax.text(31.5, 4.0, "+x wall (SBT shell shaded)", color=INK2, fontsize=8); ax.text(31.5, -4.2, "−x wall", color=INK2, fontsize=8)
    style(ax); ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout(); fig.savefig(fig_dir / "hits_mumu_topview.png", dpi=150); plt.close(fig)


def walls_mumu(d, fig_dir):
    h = dict(np.load(d / "hits_mumu.npz"))
    zb = np.linspace(31.0, 81.5, 102); yb = np.linspace(-3.4, 3.4, 69)
    fig, axs = plt.subplots(2, 2, figsize=(13, 6.4), sharex=True, sharey=True)
    for row, (cut, lab) in enumerate(((2.5, "IP < 2.5 m"), (0.10, "IP < 10 cm"))):
        for col, (face, wname) in enumerate(((0, "+x wall"), (1, "−x wall"))):
            ax = axs[row, col]
            Hs = []
            for leg in ("a", "b"):
                m = (h["ip"] < cut) & (h[f"{leg}_face"] == face)
                Hs.append(np.histogram2d(h[f"{leg}_z"][m], h[f"{leg}_y"][m], (zb, yb), weights=h["w15"][m])[0])
            H = Hs[0] + Hs[1]
            pos = H[H > 0]
            if len(pos):
                im = ax.pcolormesh(zb, yb, H.T, cmap="Blues", norm=LogNorm(max(pos.min(), pos.max() * 1e-4), pos.max()))
                fig.colorbar(im, ax=ax, pad=0.01, label="pairs / 15 y per bin")
            wall_outline(ax)
            fa = Hs[0].sum() / max(H.sum(), 1e-300)
            ax.set_title(f"{wname}, {lab}: {H.sum():.3g} pair legs / 15 y (μ+ {fa:.0%}, μ− {1 - fa:.0%})", loc="left", fontsize=9, color=INK)
            style(ax)
    for ax in axs[1]:
        ax.set_xlabel("z [m]")
    for ax in axs[:, 0]:
        ax.set_ylabel("y [m] (up the wall)")
    fig.suptitle("μ × μ pairs: where each muon first enters the SBT side walls (R2)", x=0.01, ha="left", color=INK)
    fig.tight_layout(); fig.savefig(fig_dir / "hits_mumu_walls.png", dpi=150); plt.close(fig)


def cell_rects(ax, face, w_cell, norm, cmap):
    """Draw the cells of a side wall (4 across y) with weights w_cell[(ring, seg)]."""
    for (ring, seg), w in w_cell.items():
        z0, z1 = sbt.RING_EDGES[ring], sbt.RING_EDGES[ring + 1]
        hy = geo.vessel_half_widths(np.array([0.5 * (z0 + z1)]))[1][0]
        y0 = -hy + seg * 2 * hy / sbt.N_SEG_X_FACE
        ax.add_patch(Rectangle((z0, y0), z1 - z0, 2 * hy / sbt.N_SEG_X_FACE, color=cmap(norm(w)), lw=0))


def dis_plots(d, fig_dir, design):
    cmap = plt.get_cmap("Blues")
    for cls, files in SOURCES.items():
        tag = "nu" if cls.startswith("nu") else "mu"
        acc = {0: {}, 1: {}}
        ubt = []
        for f in files:
            p = d / f"hits_{f}.npz"
            if not p.exists():
                continue
            h = dict(np.load(p))
            if "partial_sbt_ev" not in h:
                continue
            w = h["partial_w15"]
            cell = h["partial_sbt_cell"]; e = h["partial_sbt_ev"]
            face, ring, seg = cell // 10 // 1000, cell // 10 % 1000, cell % 10
            for fc in (0, 1):
                m = face == fc
                for r, s, ww in zip(ring[m], seg[m], w[e[m]]):
                    acc[fc][(r, s)] = acc[fc].get((r, s), 0.0) + ww
            if "partial_ubt_ev" in h and len(h["partial_ubt_ev"]):
                ubt.append((h["partial_ubt_x"], h["partial_ubt_y"], w[h["partial_ubt_ev"]], h["partial_ubt_plane"], h["partial_ubt_comp"]))
        allw = [v for fc in acc for v in acc[fc].values()]
        if allw:
            norm = LogNorm(max(min(allw), max(allw) * 1e-5), max(allw))
            fig, axs = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)
            for ax, (fc, wname) in zip(axs, ((0, "+x wall"), (1, "−x wall"))):
                cell_rects(ax, fc, acc[fc], norm, cmap)
                wall_outline(ax); ax.set_xlim(31, 81.5); ax.set_ylim(-3.4, 3.4)
                ax.set_title(f"{wname}: fired cells of the selected {cls} events, {sum(acc[fc].values()):.3g} cells / 15 y",
                             loc="left", fontsize=9, color=INK)
                ax.set_ylabel("y [m]"); style(ax)
            sm_ = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm_.set_array([])
            fig.colorbar(sm_, ax=axs, pad=0.01, label="fired cells / 15 y")
            axs[-1].set_xlabel("z [m]")
            fig.savefig(fig_dir / f"hits_{tag}_walls.png", dpi=150, bbox_inches="tight"); plt.close(fig)
        if ubt:
            x = np.concatenate([u[0] for u in ubt]); y = np.concatenate([u[1] for u in ubt]); w = np.concatenate([u[2] for u in ubt])
            pl = np.concatenate([u[3] for u in ubt]); comp = np.concatenate([u[4] for u in ubt])
            fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
            for ax, (kind, lab) in zip(axs, (("ubt", "straws, z = 30.73 m"), ("tiles", "tiles, z = 31.13 m"))):
                idx = [p.index for p in design.planes if p.kind == kind]
                m = np.isin(pl, idx)
                H, xe, ye = np.histogram2d(x[m], y[m], (np.linspace(-2.6, 2.6, 53), np.linspace(-2.6, 2.6, 53)), weights=w[m])
                pos = H[H > 0]
                if len(pos):
                    im = ax.pcolormesh(xe, ye, H.T, cmap="Blues", norm=LogNorm(max(pos.min(), pos.max() * 1e-4), pos.max()))
                    fig.colorbar(im, ax=ax, pad=0.01, label="crossings / 15 y")
                for p in design.planes:
                    if p.kind == kind:
                        x0 = p.to_global(0.0)
                        ax.add_patch(Rectangle((x0, p.y_lo), p.u_max, p.y_hi - p.y_lo, fill=False, ec=INK2, lw=0.6))
                fr = {c: w[m & (comp == i)].sum() for i, c in enumerate(("products", "scattered μ", "parent μ"))}
                ax.set_title(f"UBT {lab}: {w[m].sum():.3g} / 15 y\n" + ", ".join(f"{k} {v:.2g}" for k, v in fr.items() if v > 0),
                             loc="left", fontsize=8.5, color=INK)
                ax.set_xlabel("x [m]"); style(ax); ax.set_aspect("equal")
            axs[0].set_ylabel("y [m]")
            fig.suptitle(f"Selected {cls} events (IP < 2.5 m): UBT crossings of their charged particles", x=0.01, ha="left", color=INK)
            fig.tight_layout(); fig.savefig(fig_dir / f"hits_{tag}_ubt.png", dpi=150); plt.close(fig)


def seen_by(d, fig_dir):
    rows = []
    for cls, files in SOURCES.items():
        for f in files:
            p = d / f"hits_{f}.json"
            if p.exists():
                r = json.load(open(p))["results"]
                for cut in ("partial", "full"):
                    if r.get(cut, {}).get("n_mc"):
                        rows.append((f"{NAMES[f]}, IP {'< 2.5 m' if cut == 'partial' else '< 10 cm'} ({r[cut]['n_mc']} MC)", r[cut]))
    if not rows:
        return []
    fig, ax = plt.subplots(figsize=(10, 0.36 * len(rows) + 1.3))
    for i, (lab, r) in enumerate(rows):
        left = 0.0
        for k, name, col in SEEN:
            v = r.get(k, 0.0)
            ax.barh(i, v, left=left, color=col, height=0.7, edgecolor="white", linewidth=1.5, label=name if i == 0 else None)
            left += v
        ax.text(1.01, i, f"neither {r['none']:.1%}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=8); ax.invert_yaxis()
    ax.set_xlim(0, 1.15); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xlabel("fraction of selected events (weighted)")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.45, 1.0))
    style(ax); ax.grid(False, axis="y")
    fig.tight_layout(); fig.savefig(fig_dir / "hits_seen_by.png", dpi=150); plt.close(fig)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--dir", required=True)
    ap.add_argument("--design", default="designs/R2_sbt_ubt.json")
    a = ap.parse_args()
    d = Path(a.dir); fig_dir = d / "figures"; fig_dir.mkdir(exist_ok=True)
    design = dsg.load(Path(a.design))
    if (d / "hits_mumu.npz").exists():
        topview(d, fig_dir); walls_mumu(d, fig_dir)
    dis_plots(d, fig_dir, design)
    rows = seen_by(d, fig_dir)
    lines = ["| source, selection | SBT and UBT | SBT only | UBT only | neither | SBT by products / scat. μ / parent μ | UBT by products / scat. μ / parent μ |",
             "|---|---|---|---|---|---|---|"]
    for lab, r in rows:
        lines.append(f"| {lab} | {r['sbt_and_ubt']:.3f} | {r['sbt_only']:.3f} | {r['ubt_only']:.3f} | {r['none']:.3f} | "
                     + " / ".join(f"{r.get('sbt_by_' + c, 0):.3f}" for c in ("products", "scattered_mu", "parent_mu")) + " | "
                     + " / ".join(f"{r.get('ubt_by_' + c, 0):.3f}" for c in ("products", "scattered_mu", "parent_mu")) + " |")
    (d / "hits_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
