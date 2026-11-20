#!/usr/bin/env python3
"""Plots and summary numbers from the hist_*.npz files (muons_only.py, dis_classes.py).  python3 plots.py"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
SPILLS_15Y = 1.5e7
CLASSES = {   # key: (file, label, colour, groups)
    "mu": ("hist_mu_only.npz", r"$\mu$ only (no interaction)", "#2a78d6"),
    "dmv": ("hist_vessel_mu_R2_sbt_ubt.npz", r"DIS $\mu$, vessel region", "#eb6834"),
    "dm7": ("hist_ms7_muon_v0s.npz", r"DIS $\mu$, MS7 last 50 cm", "#1baf7a"),
    "dnv": ("hist_vessel_nu2026_R2_sbt_ubt.npz", r"DIS $\nu$, vessel region", "#eda100"),
    "dn7": ("hist_ms7_nu2026.npz", r"DIS $\nu$, MS7 last 50 cm", "#e87ba4"),
}
SEL = {"mu": "sel"}            # the "selected" subset of each class (mu only: SST1 muons with p > 1 GeV and timing)
CATS = ("neither", "UBT only", "SBT only", "UBT and SBT")
SPECIES = ("mu", "pi", "K", "p", "e", "other")
FACES = ("+x", "-x", "+y (top)", "-y (bottom)")
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 110, "savefig.bbox": "tight"})

H = {k: dict(np.load(os.path.join(OUT, v[0]))) for k, v in CLASSES.items() if os.path.exists(os.path.join(OUT, v[0]))}


def get(k, grp, key):
    return H[k].get(f"{grp}|{key}")


def B(k, name):
    return H[k]["_bins_" + name]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name)); plt.close(fig); print("  ", name)


def fmt(x):
    return f"{x:.3g}"


# ------------------------------------------------------------ per class: UBT x-y maps
def ubt_maps(k, grp, tag):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    xb = B(k, "xy")
    area = (xb[1] - xb[0]) ** 2 * 1e4                            # cm^2
    for ax, plane, z, (hx, hy) in ((axes[0], "straw", 30.73, (0.5, 1.35)), (axes[1], "tile", 31.13, (2.0, 2.0))):
        h = get(k, grp, f"{plane}_xy")
        if h is None or h.sum() <= 0:
            ax.set_axis_off(); continue
        d = h.T / area
        pos = d[d > 0]
        im = ax.pcolormesh(xb, xb, np.ma.masked_less_equal(d, 0), norm=LogNorm(vmin=max(pos.min(), pos.max() * 1e-6), vmax=pos.max()),
                           cmap="magma_r", rasterized=True)
        ax.add_patch(Rectangle((-hx, -hy), 2 * hx, 2 * hy, fill=False, lw=1.5, ec="#2a78d6", ls="-", label="plane aperture"))
        ax.add_patch(Rectangle((-0.5, -1.35), 1.0, 2.7, fill=False, lw=1.0, ec="k", ls="--", label="vessel entrance window"))
        ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        inside = get(k, grp, f"{plane}_p").sum()
        ax.set_title(f"UBT {plane} plane z = {z} m\nin aperture: {fmt(inside)} tracks / spill")
        fig.colorbar(im, ax=ax, label="crossing tracks / cm$^2$ / spill")
        ax.legend(loc="lower left", fontsize=8, framealpha=0.8)
    fig.suptitle(f"{CLASSES[k][1]} - {tag}: charged-track crossings of the UBT planes (all crossings, inside and outside the aperture)")
    save(fig, f"{k}_{grp}_ubt_xy.png")


# ------------------------------------------------------------ per class: SBT maps
def sbt_maps(k, grp, tag):
    zb, pb = B(k, "ring"), B(k, "phi")
    h = get(k, grp, "sbt_fired_zphi")
    if h is None or h.sum() <= 0:
        return
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1])
    ax = fig.add_subplot(gs[0, :])
    pos = h[h > 0]
    im = ax.pcolormesh(zb, pb, np.ma.masked_less_equal(h.T, 0), norm=LogNorm(vmin=max(pos.min(), pos.max() * 1e-5), vmax=pos.max()),
                       cmap="magma_r", rasterized=True)
    for c, lab in ((45, ""), (135, ""), (-45, ""), (-135, "")):
        ax.axhline(c, color="#52514e", lw=0.6, ls=":")
    for y, lab in ((0, "+x wall"), (90, "top (+y)"), (180, "-x wall"), (-90, "bottom (-y)")):
        ax.text(31.6, y if abs(y) < 180 else 165, lab, fontsize=9, va="center", bbox=dict(fc="white", ec="none", alpha=0.7))
    ax.set_xlabel("z [m] (cell rings)"); ax.set_ylabel("azimuth on the vessel [deg]\n(normalised to the local half widths)")
    ax.set_yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    fig.colorbar(im, ax=ax, label="fired cells / spill per bin (cell ring x 5 deg)")
    ax.set_title(f"SBT cells fired (> 30 MeV) by the event's own tracks: {fmt(h.sum())} / spill")
    ax2 = fig.add_subplot(gs[1, 0])
    for key, lab, ls in (("sbt_cross_ring", "track pieces in the cell shell (any deposit)", "--"), ("sbt_fired_ring", "fired cells (> 30 MeV)", "-")):
        y = get(k, grp, key)
        ax2.stairs(y, zb, label=lab, lw=2, ls=ls, color=CLASSES[k][2])
    sr = get(k, grp, "sbt_fired_sr")
    ax2.set_yscale("log"); ax2.set_xlabel("z [m]"); ax2.set_ylabel("per cell ring / spill"); ax2.legend(fontsize=8)
    ax2.set_title("along z (61 cell rings, 0.8 m)")
    ax3 = fig.add_subplot(gs[1, 1])
    f = get(k, grp, "sbt_fired_face")
    ax3.bar(range(4), f / max(f.sum(), 1e-300), color=CLASSES[k][2], width=0.6)
    ax3.set_xticks(range(4), FACES); ax3.set_ylabel("fraction of fired cells")
    ax3.set_title("per face; SR-389 rings (fraction): " + " ".join(f"{v / max(sr.sum(), 1e-300):.2f}" for v in sr), fontsize=9)
    fig.suptitle(f"{CLASSES[k][1]} - {tag}")
    fig.tight_layout()
    save(fig, f"{k}_{grp}_sbt_map.png")
    # z vs across, per face
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True)
    for fi, ax in enumerate(axes.ravel()):
        h = get(k, grp, f"sbt_fired_face{fi}")
        yb = B(k, "side" if fi < 2 else "tb")
        if h.sum() > 0:
            pos = h[h > 0]
            im = ax.pcolormesh(zb, yb, np.ma.masked_less_equal(h.T, 0), norm=LogNorm(vmin=max(pos.min(), pos.max() * 1e-5), vmax=pos.max()),
                               cmap="magma_r", rasterized=True)
            fig.colorbar(im, ax=ax, label="fired cells / spill / bin")
        z = np.linspace(31.13, 81.13, 50)
        half = 1.35 + 1.65 * (z - 31.13) / 50 if fi < 2 else 0.5 + 1.5 * (z - 31.13) / 50
        ax.plot(z, half, "k:", lw=0.8); ax.plot(z, -half, "k:", lw=0.8)
        ax.set_title(f"face {FACES[fi]}: {fmt(h.sum())} / spill"); ax.set_ylabel("y [m]" if fi < 2 else "x [m]")
        if fi >= 2:
            ax.set_xlabel("z [m]")
    fig.suptitle(f"{CLASSES[k][1]} - {tag}: fired SBT cells on each face (dotted: face edges)")
    fig.tight_layout()
    save(fig, f"{k}_{grp}_sbt_faces.png")


# ------------------------------------------------------------ comparisons
def sel_grp(k):
    return SEL.get(k, "sel")


def cmp_p():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for row, grpf in enumerate((lambda k: "all", sel_grp)):
        for ax, key, title in zip(axes[row], ("straw_p", "tile_p", "sbt_fired_p"),
                                  ("UBT straws (in aperture)", "UBT tiles (in aperture)", "SBT: tracks firing a cell (per fired cell)")):
            for k in H:
                y = get(k, grpf(k), key)
                if y is None or y.sum() <= 0:
                    continue
                ax.stairs(y, B(k, "p"), color=CLASSES[k][2], lw=2, label=f"{CLASSES[k][1]} ({fmt(y.sum())})")
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("|p| [GeV/c]"); ax.set_ylabel("tracks / spill / bin")
            ax.set_title(("all events: " if row == 0 else "selected: ") + title, fontsize=10)
            ax.legend(fontsize=7)
    fig.suptitle("Momentum of the charged tracks crossing the UBT / firing the SBT (per spill; 15 bins per decade)")
    fig.tight_layout(); save(fig, "cmp_momentum.png")


def cmp_ring():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, grpf, t in ((axes[0], lambda k: "all", "all events"), (axes[1], sel_grp, "selected events")):
        for k in H:
            y = get(k, grpf(k), "sbt_fired_ring")
            if y is None or y.sum() <= 0:
                continue
            ax.stairs(y, B(k, "ring"), color=CLASSES[k][2], lw=2, label=CLASSES[k][1])
        ax.set_yscale("log"); ax.set_xlabel("z [m]"); ax.set_ylabel("fired cells / spill per cell ring")
        ax.set_title(f"SBT fired cells along z, {t}"); ax.legend(fontsize=8)
    fig.tight_layout(); save(fig, "cmp_sbt_z.png")


def cmp_mult():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for row, grpf in enumerate((lambda k: "all", sel_grp)):
        for ax, key, title in zip(axes[row], ("ev_n_straw", "ev_n_tile", "ev_n_fired"),
                                  ("charged tracks in the UBT straw aperture", "charged tracks in the UBT tile aperture",
                                   "SBT cells fired")):
            for k in H:
                y = get(k, grpf(k), key)
                if y is None or y.sum() <= 0:
                    continue
                ax.stairs(y / y.sum(), B(k, "mult"), color=CLASSES[k][2], lw=2,
                          label=f"{CLASSES[k][1]}: mean {np.dot(np.arange(len(y)), y) / y.sum():.2f}")
            ax.set_yscale("log"); ax.set_xlim(-0.5, 40); ax.set_xlabel("per event (59 = overflow)"); ax.set_ylabel("fraction of events (weighted)")
            ax.set_title(("all events: " if row == 0 else "selected: ") + title, fontsize=10); ax.legend(fontsize=7)
    fig.suptitle("Multiplicity per event (mu only: per muon). Vessel DIS mu: the parent muon is included.")
    fig.tight_layout(); save(fig, "cmp_multiplicity.png")


def cmp_species():
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5))
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
    for row, grpf in enumerate((lambda k: "all", sel_grp)):
        for ax, key, title in zip(axes[row], ("tile_species", "sbt_fired_species"), ("UBT tile plane (in aperture)", "SBT fired cells")):
            names = []
            for i, k in enumerate([k for k in H if k != "mu"]):
                y = get(k, grpf(k), key)
                names.append(CLASSES[k][1])
                if y is None or y.sum() <= 0:
                    continue
                f = y / y.sum(); left = 0
                for j, s in enumerate(SPECIES):
                    ax.barh(i, f[j], left=left, color=cols[j], edgecolor="white", lw=1.5, label=s if i == 0 else None)
                    if f[j] > 0.06:
                        ax.text(left + f[j] / 2, i, f"{s} {f[j]:.0%}", ha="center", va="center", fontsize=8, color="white")
                    left += f[j]
                ax.text(1.01, i, f"{fmt(y.sum())}/spill", va="center", fontsize=8)
            ax.set_yticks(range(len(names)), names); ax.set_xlim(0, 1.15); ax.set_xlabel("fraction of crossing tracks")
            ax.set_title(("all events: " if row == 0 else "selected: ") + title, fontsize=10)
            ax.grid(False)
        axes[row, 1].legend(fontsize=8, loc="lower right", ncol=6)
    fig.suptitle("Particle composition of the charged tracks crossing (mu-only class: 100 % muons)")
    fig.tight_layout(); save(fig, "cmp_species.png")


def cmp_categories(summary):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    cols = ["#c3c2b7", "#2a78d6", "#eb6834", "#6250d6"]
    for ax, grpf, t in ((axes[0], lambda k: "all", "all events"), (axes[1], sel_grp, "selected events")):
        names = []
        for i, k in enumerate(H):
            y = get(k, grpf(k), "ev_cat")
            names.append(CLASSES[k][1])
            if y is None or y.sum() <= 0:
                continue
            f = y / y.sum(); left = 0
            for j, c in enumerate(CATS):
                ax.barh(i, f[j], left=left, color=cols[j], edgecolor="white", lw=1.5, label=c if i == 0 else None)
                if f[j] > 0.07:
                    ax.text(left + f[j] / 2, i, f"{f[j]:.0%}", ha="center", va="center", fontsize=8, color="white" if j else "black")
                left += f[j]
        ax.set_yticks(range(len(names)), names); ax.set_xlim(0, 1); ax.set_xlabel("fraction of events (weighted)")
        ax.set_title(f"Which tagger does the event reach? {t}\n(UBT = a track in the straw or tile aperture; SBT = >= 1 fired cell)", fontsize=10)
        ax.grid(False); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); save(fig, "cmp_categories.png")


def summary_table():
    out = {}
    for k in H:
        for grp in sorted({g.split("|")[0] for g in H[k] if "|" in g and not g.startswith("_")}):
            cat = get(k, grp, "ev_cat")
            if cat is None:
                continue
            tot = cat.sum()
            ne = get(k, grp, "n_events")
            r = dict(events_per_spill=tot, events_15y=tot * SPILLS_15Y, n_mc=int(ne[0]) if ne is not None else None,
                     **{f"frac_{c.replace(' ', '_')}": (cat[i] / tot if tot else 0) for i, c in enumerate(CATS)},
                     frac_tile_any=(get(k, grp, "ev_cat_tile")[[1, 3]].sum() / tot if tot else 0),
                     straw_tracks_per_spill=get(k, grp, "straw_p").sum(), tile_tracks_per_spill=get(k, grp, "tile_p").sum(),
                     sbt_fired_cells_per_spill=get(k, grp, "sbt_fired_ring").sum(),
                     sbt_fired_per_face=list(get(k, grp, "sbt_fired_face")), sbt_fired_per_sr_ring=list(get(k, grp, "sbt_fired_sr")))
            n_cat = get(k, grp, "ev_cat_n")
            if n_cat is not None:
                r["n_mc_per_cat"] = [int(x) for x in n_cat]
            out[f"{k}|{grp}"] = r
    json.dump(out, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
    lines = ["| class | subset | events / spill | events / 15 y | MC events | neither | UBT only | SBT only | both | tile tracks / spill | SBT fired cells / spill |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, r in out.items():
        k, grp = key.split("|")
        lines.append(f"| {CLASSES[k][1]} | {grp} | {fmt(r['events_per_spill'])} | {fmt(r['events_15y'])} | {r['n_mc'] or ''} | "
                     f"{r['frac_neither']:.3f} | {r['frac_UBT_only']:.3f} | {r['frac_SBT_only']:.3f} | {r['frac_UBT_and_SBT']:.3f} | "
                     f"{fmt(r['tile_tracks_per_spill'])} | {fmt(r['sbt_fired_cells_per_spill'])} |")
    open(os.path.join(OUT, "summary_table.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return out


if __name__ == "__main__":
    s = summary_table()
    for k in H:
        groups = ["all", sel_grp(k)]
        if k == "dmv":
            groups += ["products"]
        if k in ("dm7", "dn7"):
            groups += ["v0dec"]
        for grp in groups:
            if get(k, grp, "tile_xy") is None:
                continue
            tag = {"all": "all events", "sel": "selected events", "products": "products only (no parent muon)",
                   "v0dec": "forced V0 decays only"}.get(grp, grp)
            if k == "mu" and grp == "sel":
                tag = "SST1 muons with p > 1 GeV and timing (selection input)"
            ubt_maps(k, grp, tag)
            sbt_maps(k, grp, tag)
    cmp_p(); cmp_ring(); cmp_mult(); cmp_species(); cmp_categories(s)
