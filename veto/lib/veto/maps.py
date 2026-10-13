"""Histograms that can be filled chunk by chunk (for the 150 M-muon UBT dataset)
and plotted afterwards: SBT wall maps, side-view track density, x-y maps."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from . import geometry as geo
from . import plotting
from .tracks import FACE_NAMES, extrapolate_to_z

WALLS = (2, 3, 4, 5)
WALL_Z_EDGES = np.linspace(geo.Z_VESSEL_MIN, geo.Z_VESSEL_MAX, 101)
WALL_ALONG_EDGES = np.linspace(-3.2, 3.2, 65)

SIDE_Z_EDGES = np.arange(geo.Z_VESSEL_MIN, geo.Z_SST1 + 0.25, 0.25)
SIDE_T_EDGES = np.linspace(-6, 6, 145)


def wall_histograms(face, z, x, y, w):
    """(4, nz, n_along) weighted hits on the SBT walls +x, -x, +y, -y.

    Coordinate along the wall is y for the x walls and x for the y walls."""
    out = np.zeros((4, len(WALL_Z_EDGES) - 1, len(WALL_ALONG_EDGES) - 1))
    for i, wall in enumerate(WALLS):
        m = face == wall
        along = y[m] if wall in (2, 3) else x[m]
        out[i], _, _ = np.histogram2d(z[m], along, bins=[WALL_Z_EDGES, WALL_ALONG_EDGES], weights=w[m])
    return out


def plot_wall_maps(H, title, per_area=True):
    """Unfolded SBT walls. per_area: divide by the bin area on the wall (Hz / m^2)."""
    dz = np.diff(WALL_Z_EDGES)[:, None]
    da = np.diff(WALL_ALONG_EDGES)[None, :]
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
    z = WALL_Z_EDGES[[0, -1]]
    for ax, i in zip(axes.ravel(), range(4)):
        wall = WALLS[i]
        # wall surface is tilted: slant length per unit z
        slope = (geo.VESSEL_X_OUT - geo.VESSEL_X_IN) if wall in (2, 3) else (geo.VESSEL_Y_OUT - geo.VESSEL_Y_IN)
        slant = np.sqrt(1 + (slope / (geo.Z_VESSEL_MAX - geo.Z_VESSEL_MIN)) ** 2)
        values = H[i] / (dz * da * slant) if per_area else H[i]
        if (values > 0).any():
            mesh = ax.pcolormesh(WALL_Z_EDGES, WALL_ALONG_EDGES, values.T, norm=LogNorm(), cmap="magma")
            fig.colorbar(mesh, ax=ax, label="Hz / m$^2$" if per_area else "Hz / bin")
        half = geo.vessel_half_widths(z)[1 if wall in (2, 3) else 0]
        ax.plot(z, half, "w--", z, -half, "w--", lw=1)
        ax.set_facecolor("black")
        ax.set(title=f"{FACE_NAMES[wall]}: {H[i].sum():.3e} Hz",
               ylabel=("y" if wall in (2, 3) else "x") + " [m]")
    for ax in axes[1]:
        ax.set_xlabel("z [m]")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def side_histograms(data, z_start, z_end, w, chunk=100_000):
    """(2, nz, nt) weighted path density of straight segments [z_start, z_end] in z-x and z-y."""
    zc = 0.5 * (SIDE_Z_EDGES[:-1] + SIDE_Z_EDGES[1:])
    out = np.zeros((2, len(zc), len(SIDE_T_EDGES) - 1))
    for s in range(0, len(w), chunk):
        sl = slice(s, s + chunk)
        zz = np.broadcast_to(zc, (len(w[sl]), len(zc)))
        on = (zz >= z_start[sl, None]) & (zz <= z_end[sl, None])
        x, y = extrapolate_to_z(*(np.asarray(data[k][sl], dtype=np.float64)[:, None]
                                  for k in ("x", "y", "z", "px", "py", "pz")), zz)
        ww = np.broadcast_to(w[sl, None], zz.shape)
        for i, t in enumerate((x, y)):
            h, _, _ = np.histogram2d(zz[on], t[on], bins=[SIDE_Z_EDGES, SIDE_T_EDGES], weights=ww[on])
            out[i] += h
    return out


def plot_side(H, title):
    fig, axes = plt.subplots(2, 1, figsize=(18, 12), sharex=True)
    for ax, i, axis in ((axes[0], 0, "x"), (axes[1], 1, "y")):
        if (H[i] > 0).any():
            mesh = ax.pcolormesh(SIDE_Z_EDGES, SIDE_T_EDGES, H[i].T, norm=LogNorm(), cmap="viridis")
            fig.colorbar(mesh, ax=ax, label="rate x path [Hz / bin]")
        plotting.draw_detector_side(ax, axis)
        ax.set(ylabel=f"{axis} [m]", xlim=(-3, 98), ylim=(-6, 6))
        ax.legend(loc="upper left", fontsize=11, ncol=3)
    axes[1].set_xlabel("z [m]")
    axes[0].set_title(title)
    fig.tight_layout()
    return fig


def plot_xy(H, edges, title, ubt=True, label="Hz / bin"):
    fig, ax = plt.subplots(figsize=(9, 8))
    if (H > 0).any():
        mesh = ax.pcolormesh(edges[0], edges[1], H.T, norm=LogNorm(), cmap="viridis")
        fig.colorbar(mesh, ax=ax, label=label)
    if ubt:
        plotting.draw_ubt_aperture(ax)
        ax.legend(loc="upper right")
    ax.set(xlabel="x [m]", ylabel="y [m]", title=title)
    ax.set_aspect("equal")
    return fig
