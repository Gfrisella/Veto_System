"""Matplotlib helpers: style, saving, geometry overlays and the diagnostic plots
from analysis.ipynb."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Rectangle

from . import geometry as geo


def set_style():
    plt.rcParams.update({
        "figure.figsize": (9, 9),
        "figure.dpi": 100,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "font.size": 18,
    })


def save_fig(fig, name, outdir, dpi=200, show=False):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    print(f"  saved {outdir / name}.{{pdf,png}}")
    if show:
        plt.show()
    plt.close(fig)


# ============================================================
# Geometry overlays
# ============================================================

def draw_vessel_side(ax, axis="x", label="Decay vessel (SBT walls)"):
    """Decay-vessel outline on a z-vs-x (axis='x') or z-vs-y (axis='y') plot."""
    z = np.linspace(geo.Z_VESSEL_MIN, geo.Z_VESSEL_MAX, 2)
    half = geo.vessel_half_widths(z)[0 if axis == "x" else 1]
    ax.plot(z, half, "b--", lw=2, label=label)
    ax.plot(z, -half, "b--", lw=2)
    ax.fill_between(z, -half, half, alpha=0.12, color="b")


def draw_plane(ax, z, half, label, color="r"):
    ax.vlines(z, -half, half, color=color, linestyle="--", lw=1.5, label=label)


def draw_detector_side(ax, axis="x", shield=True, spectrometer=True):
    """Shield magnets, vessel, UBT, SST1, spectrometer magnet and timing plane in a side view."""
    i = 0 if axis == "x" else 1
    if shield:
        for k, m in enumerate(geo.shield_magnets()):
            half_in = m["half_x_in"] if axis == "x" else m["half_y_in"]
            half_out = m["half_x_out"] if axis == "x" else m["half_y_out"]
            ax.fill([m["z_start"], m["z_end"], m["z_end"], m["z_start"]],
                    [-half_in, -half_out, half_out, half_in],
                    color="0.5", alpha=0.35, lw=0, label="Muon shield (TRY6)" if k == 0 else None)
    draw_vessel_side(ax, axis)
    draw_plane(ax, geo.Z_UBT, (geo.UBT_HALF_X, geo.UBT_HALF_Y)[i], "UBT", color="tab:orange")
    draw_plane(ax, geo.Z_SST1, (geo.SST1_HALF_X, geo.SST1_HALF_Y)[i], "SST1", color="r")
    draw_plane(ax, geo.Z_TIMING, (geo.TIMING_HALF_X, geo.TIMING_HALF_Y)[i], "Timing", color="tab:purple")
    if spectrometer:
        half = (geo.SPECTROMETER_HALF_X, geo.SPECTROMETER_HALF_Y)[i]
        z0, z1 = geo.SPECTROMETER_Z
        ax.add_patch(Rectangle((z0, -half), z1 - z0, 2 * half, color="tab:green", alpha=0.2,
                               label="Spectrometer magnet"))


def draw_ubt_aperture(ax, label="UBT"):
    ax.add_patch(Rectangle((-geo.UBT_HALF_X, -geo.UBT_HALF_Y), 2 * geo.UBT_HALF_X, 2 * geo.UBT_HALF_Y,
                           fill=False, ec="red", ls="--", lw=2, label=label))


# ============================================================
# Diagnostic plots (analysis.ipynb "check" cells)
# ============================================================

def cumulative_cut_plot(values, cut, bins, xlabel, title):
    accepted = 100 * np.mean(values > cut)
    fig, ax = plt.subplots()
    ax.hist(values[values > 0], bins=bins, cumulative=True, density=True, histtype="step", lw=2, label="All muons")
    ax.set_xscale("log")
    ax.axvline(cut, color="red", ls="--", lw=2, label=f"Cut at {cut:g}")
    ax.axhline(1 - accepted / 100, color="orange", ls=":", lw=2, label=f"Below cut: {100 - accepted:.2f}%")
    ax.yaxis.set_major_formatter(lambda v, pos: f"{100 * v:.0f}%")
    ax.set(xlabel=xlabel, ylabel="Cumulative fraction", title=title)
    ax.grid(alpha=0.2)
    ax.legend()
    return fig


def xy_map(x, y, weights, title, bins=100, extent=None, label="Rate [Hz]", ubt=False):
    fig, ax = plt.subplots(figsize=(9, 8))
    h = ax.hist2d(x, y, bins=bins, range=extent, weights=weights, norm=LogNorm(), cmap="viridis")
    fig.colorbar(h[3], ax=ax, label=label)
    if ubt:
        draw_ubt_aperture(ax)
        ax.legend(loc="upper right")
    ax.set(xlabel="x [m]", ylabel="y [m]", title=title)
    return fig


def weight_distribution(weights):
    positive = weights[weights > 0]
    bins = np.logspace(np.log10(positive.min()), np.log10(positive.max()), 50)
    fig, ax = plt.subplots()
    ax.hist(weights, bins=bins)
    ax.set(xscale="log", yscale="log", xlabel="Weight", ylabel="Entries")
    ax.grid(alpha=0.2)
    return fig


def backtracking_validation(data, z_vessel, x_vessel, y_vessel, amount=100):
    """Side views of a few tracks back-projected from SST1 to their vessel entry point."""
    valid = np.where(np.isfinite(z_vessel))[0][:amount]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, axis, coord, entry in ((axes[0], "x", data["x"], x_vessel), (axes[1], "y", data["y"], y_vessel)):
        draw_vessel_side(ax, axis)
        draw_plane(ax, geo.Z_SST1, geo.SST1_HALF_X if axis == "x" else geo.SST1_HALF_Y, "SST1")
        for k in valid:
            ax.plot([data["z"][k], z_vessel[k]], [coord[k], entry[k]], "-", alpha=0.5)
            ax.plot(z_vessel[k], entry[k], "x", ms=7, mew=2)
        ax.set(xlabel="z [m]", ylabel=f"{axis} [m]", title=f"Backtracking validation {axis.upper()}-Z",
               xlim=(25, 85), ylim=(-4, 4))
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    return fig


def doca_plots(doca):
    doca_cm = 100 * doca[np.isfinite(doca)]
    figs = {}
    fig, ax = plt.subplots(figsize=(8, 5))
    ordered = np.sort(doca_cm)
    ax.plot(ordered, np.arange(1, len(ordered) + 1) / max(len(ordered), 1), lw=2)
    fraction = np.mean(doca_cm <= 2.0) if len(doca_cm) else 0.0
    ax.axvline(2.0, color="r", ls="--", label="2 cm")
    ax.axhline(fraction, color="r", ls=":", alpha=0.7)
    ax.set(xlabel="DOCA [cm]", ylabel="Cumulative fraction", title="Cumulative DOCA distribution", xlim=(0, 20))
    ax.grid(alpha=0.2)
    ax.legend()
    figs["doca_cumulative"] = fig

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(doca_cm, bins=np.linspace(0, 5, 101), histtype="step", lw=2)
    ax.axvline(2.0, color="r", ls="--", lw=2, label="2 cm")
    ax.set(xlabel="DOCA [cm]", ylabel="Entries", title="DOCA distribution", yscale="log", xlim=(0, 5))
    ax.grid(alpha=0.2)
    ax.legend()
    figs["doca_histogram"] = fig
    return figs


def vertices_and_tracks(data, pairs, vertex, amount=100):
    """Selected pairs: both tracks drawn back to z=30 plus their vertex."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, axis in zip(axes, "xy"):
        draw_vessel_side(ax, axis)
        draw_plane(ax, geo.Z_SST1, geo.SST1_HALF_X if axis == "x" else geo.SST1_HALF_Y, "SST1")
    for k in range(min(amount, len(pairs))):
        color = f"C{k % 10}"
        for i in pairs[k]:
            z_line = np.linspace(data["z"][i], 30, 2)
            x_line = data["x"][i] + (z_line - data["z"][i]) * data["px"][i] / data["pz"][i]
            y_line = data["y"][i] + (z_line - data["z"][i]) * data["py"][i] / data["pz"][i]
            axes[0].plot(z_line, x_line, color=color, alpha=0.7)
            axes[1].plot(z_line, y_line, color=color, alpha=0.7)
        axes[0].scatter(vertex[k, 2], vertex[k, 0], color=color, s=40, zorder=5)
        axes[1].scatter(vertex[k, 2], vertex[k, 1], color=color, s=40, zorder=5)
    for ax, axis in zip(axes, "xy"):
        ax.set(xlabel="z [m]", ylabel=f"{axis} [m]", title=f"{axis.upper()}-Z: selected tracks and vertices")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    return fig


def ip_plot(x_ip, y_ip, ip_max):
    fig, ax = plt.subplots()
    ax.plot(x_ip, y_ip, "x", label=r"@$z_{target}$")
    ax.add_patch(Circle((0, 0), ip_max, fill=False, ls="--", lw=2, label="IP cut"))
    ax.set_aspect("equal", adjustable="box")
    ax.set(xlabel="x [m]", ylabel="y [m]")
    ax.grid(alpha=0.2)
    ax.legend()
    return fig


def final_vertex_plots(vertex, doca, ip, x_ip, y_ip, ip_max):
    """Plots of make_plots() in Analysis_singlefiles.py."""
    figs = {}
    fig, ax = plt.subplots()
    ax.hist(doca * 100, bins=100, histtype="step", lw=2)
    ax.set(xlabel="DOCA [cm]", ylabel="Pairs")
    ax.grid(alpha=0.2)
    figs["final_doca"] = fig

    fig, ax = plt.subplots()
    h = ax.hist2d(vertex[:, 0], vertex[:, 1], bins=100, norm=LogNorm())
    fig.colorbar(h[3], ax=ax)
    ax.set(xlabel="Vertex x [m]", ylabel="Vertex y [m]")
    figs["final_vertex_xy"] = fig

    fig, ax = plt.subplots()
    ax.hist(vertex[:, 2], bins=100, histtype="step", lw=2)
    ax.set(xlabel="Vertex z [m]", ylabel="Pairs")
    figs["final_vertex_z"] = fig

    fig, ax = plt.subplots()
    ax.hist(ip, bins=100, histtype="step", lw=2)
    ax.axvline(ip_max, ls="--", label="IP cut")
    ax.set(xlabel="IP [m]", ylabel="Pairs")
    ax.legend()
    figs["final_ip"] = fig

    fig = ip_plot(x_ip, y_ip, ip_max)
    figs["final_ip_xy"] = fig
    return figs


# ============================================================
# 3D muon-density histogram (muon_distribution_3d.pkl)
# ============================================================

def _mesh(ax, xe, ye, values, log):
    if log:
        positive = values[values > 0]
        norm = LogNorm(vmin=positive.min(), vmax=positive.max()) if len(positive) else None
        return ax.pcolormesh(xe, ye, values, shading="auto", norm=norm)
    return ax.pcolormesh(xe, ye, values, shading="auto")


def distribution_xy(distribution, iz=0, log=True):
    H, xe, ye, ze = (distribution[k] for k in ("H", "x_edges", "y_edges", "z_edges"))
    fig, ax = plt.subplots(figsize=(9, 8))
    mesh = _mesh(ax, xe, ye, H[:, :, iz].T, log)
    ax.set(xlabel="x [m]", ylabel="y [m]", title=f"Muon distribution at z = {ze[iz]:.3f} m")
    ax.set_aspect("equal")
    fig.colorbar(mesh, ax=ax, label="Weighted counts")
    return fig


def distribution_side(distribution, axis="x", log=True):
    """z-x (integrated over y) or z-y (integrated over x) projection."""
    H, ze = distribution["H"], distribution["z_edges"]
    edges = distribution["x_edges" if axis == "x" else "y_edges"]
    values = H.sum(axis=1 if axis == "x" else 0)
    fig, ax = plt.subplots(figsize=(12, 7))
    mesh = _mesh(ax, ze, edges, values, log)
    draw_vessel_side(ax, axis)
    ax.set(xlabel="z [m]", ylabel=f"{axis} [m]",
           title=f"Muon distribution integrated over {'y' if axis == 'x' else 'x'}")
    fig.colorbar(mesh, ax=ax, label="Weighted counts")
    return fig
