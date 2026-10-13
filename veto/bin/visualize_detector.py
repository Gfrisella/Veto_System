#!/usr/bin/env python3
"""Detector + muon visualisation, for either muon dataset.

  --source sst1  muons at SST1 (z = 82 m) from the per-seed output files, drawn back to
                 where they entered the decay vessel (what the pair analysis uses)
  --source ubt   muons at the UBT plane (z = 30.6 m) from UBT_data_TRY6, drawn forward to
                 z = 82 m. Uses results/ubt/entering_vessel.npz (every muon that enters the
                 vessel, from bin/analyse_ubt_plane.py) plus the random subsample for the rest.
                 Muons that never cross the 6 x 6 m UBT plane are not in this dataset.

Produces in results/visualization/<source>/:
  detector_3d.html  interactive 3D view: TRY6 shield, target, decay vessel (SBT walls), UBT,
                    SST1, spectrometer magnet, timing plane and muons coloured by vessel entry
  side_views.png    x-z and y-z density of the drawn track segments
  ubt_plane.png     muons at (sst1: back-projected to) the UBT plane
  sbt_walls.png     unfolded SBT walls: where muons enter the vessel
  field_map.png     |B| of the spectrometer field map at y=0 and x=0
"""

import _bootstrap  # noqa: F401

import argparse

import matplotlib.pyplot as plt
import numpy as np

from veto import geometry as geo
from veto import io, maps, paths, plotting, selection, viz3d
from veto import weights as wt
from veto.config import DEFAULT_CONFIG
from veto.tracks import FACE_NAMES, extrapolate_to_z, vessel_crossings

GROUP_COLORS = {
    "enters through UBT": "#e8820c",
    "enters through SBT wall": "#1f77b4",
    "misses vessel": "#7f7f7f",
}


def add_crossings(data):
    c = vessel_crossings(data["x"], data["y"], data["z"], data["px"], data["py"], data["pz"])
    data.update(face_in=c["face_in"], z_vessel=c["z_in"], x_vessel=c["x_in"], y_vessel=c["y_in"])
    return data


def load_sst1(max_files, apply_cuts):
    files = io.list_output_files(max_files=max_files)
    data = io.load_outputs(files, all_planes=True, progress_every=0)
    data["weight"] = wt.corrected_weights(data["weight"], data["evt_n"], len(files))
    for key in io.TRACK_KEYS:          # reference state = SST1
        data[key] = data[f"{key}2"]
    if apply_cuts:
        mask, flow = selection.apply_single_muon_cuts(data, DEFAULT_CONFIG)
        selection.print_flow("SINGLE-MUON SELECTION", flow)
        data = io.select(data, mask)
    add_crossings(data)
    data["z_start"] = np.where(np.isfinite(data["z_vessel"]), data["z_vessel"], geo.Z_UBT)
    data["z_end"] = data["z"].astype(np.float64)
    return data


def load_ubt(n_misses, rng):
    """All vessel-entering muons + a random set of the others (weights rescaled to stay correct)."""
    entering = dict(np.load(paths.results_dir() / "ubt" / "entering_vessel.npz"))
    sample = dict(np.load(paths.store_dir("ubt_TRY6_sample.npz")))
    keys = ("x", "y", "z", "px", "py", "pz", "pdg", "weight")
    c = vessel_crossings(*(sample[k] for k in ("x", "y", "z", "px", "py", "pz")))
    misses = np.where(c["face_in"] == 0)[0]
    pick = rng.choice(misses, size=min(n_misses, len(misses)), replace=False)
    other = {k: sample[k][pick] for k in keys}
    other["weight"] = other["weight"] * len(misses) / len(pick)
    data = io.concatenate([{k: entering[k].astype(np.float32) for k in keys}, other])
    add_crossings(data)
    data["z_start"] = data["z"].astype(np.float64)
    data["z_end"] = np.full(len(data["x"]), geo.Z_SST1)
    return data


def groups_of(data):
    face = data["face_in"]
    return {
        "enters through UBT": face == 1,
        "enters through SBT wall": face >= 2,
        "misses vessel": face == 0,
    }


def subsample(groups, n_per_group, rng):
    out = {}
    for name, mask in groups.items():
        idx = np.where(mask)[0]
        keep = np.zeros_like(mask)
        keep[rng.choice(idx, size=min(n_per_group, len(idx)), replace=False)] = True
        out[name] = keep
    return out


def print_summary(data, source):
    w = data["weight"].astype(np.float64)
    print()
    print(f"WHERE MUONS ENTER THE DECAY VESSEL ({source})")
    print("-" * 70)
    for face in range(6):
        m = data["face_in"] == face
        print(f"{FACE_NAMES[face] if face else 'never inside':<22}{m.sum():>12,}   weight {w[m].sum():.4e}"
              f"   ({100 * w[m].sum() / w.sum():8.4f} %)")


def ubt_plane(data, outdir, source):
    x, y = extrapolate_to_z(data["x"], data["y"], data["z"], data["px"], data["py"], data["pz"], geo.Z_UBT)
    title = ("Back-projection to the UBT plane" if source == "sst1" else "Muons at the UBT plane") \
        + f" (z = {geo.Z_UBT} m)"
    fig = plotting.xy_map(x, y, data["weight"], title, bins=[160, 160], extent=[[-6, 6], [-6, 6]], ubt=True)
    ax = fig.axes[0]
    ax.axvline(geo.CAVERN_X_WALL, color="k", ls=":", lw=1.5)
    ax.axhline(geo.CAVERN_Y_FLOOR, color="k", ls=":", lw=1.5)
    plotting.save_fig(fig, "ubt_plane", outdir)


def field_map(outdir):
    try:
        points, B = io.load_field_map()
    except (OSError, KeyError, ImportError) as error:
        print(f"Skipping field map: {error}")
        return
    # scatter the points into a regular grid by their coordinates
    axes_values = [np.unique(np.round(points[:, i], 3)) for i in range(3)]
    index = tuple(np.searchsorted(v, np.round(points[:, i], 3)) for i, v in enumerate(axes_values))
    magnitude = np.full([len(v) for v in axes_values], np.nan)
    magnitude[index] = np.linalg.norm(B, axis=1)
    xs, ys, zs = axes_values
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, axis in zip(axes, ("x", "y")):
        if axis == "x":
            values, coord, other = magnitude[:, np.argmin(np.abs(ys)), :], xs, f"y = {ys[np.argmin(np.abs(ys))]:.2f}"
        else:
            values, coord, other = magnitude[np.argmin(np.abs(xs)), :, :], ys, f"x = {xs[np.argmin(np.abs(xs))]:.2f}"
        h = ax.pcolormesh(zs, coord, values, shading="nearest", cmap="inferno")
        fig.colorbar(h, ax=ax, label="|B| [T]")
        ax.set(xlabel="z [m]", ylabel=f"{axis} [m]", title=f"|B| at {other} m")
    fig.tight_layout()
    plotting.save_fig(fig, "field_map", outdir)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=("sst1", "ubt"), default="sst1")
    parser.add_argument("--max-files", type=int, default=100, help="sst1: number of output files")
    parser.add_argument("--n-tracks", type=int, default=400, help="tracks per category in the 3D view")
    parser.add_argument("--no-cuts", action="store_true", help="sst1: skip the momentum / timing cuts")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    outdir = paths.results_dir() / "visualization" / args.source
    outdir.mkdir(parents=True, exist_ok=True)
    plotting.set_style()

    if args.source == "sst1":
        data = load_sst1(args.max_files, apply_cuts=not args.no_cuts)
        label = "muons at SST1, drawn back to their vessel entry"
    else:
        data = load_ubt(n_misses=200_000, rng=rng)
        label = "muons at the UBT plane, drawn forward to z = 82 m"
    print_summary(data, args.source)
    w = data["weight"].astype(np.float64)

    groups = subsample(groups_of(data), args.n_tracks, rng)
    fig = viz3d.detector_figure(data, groups, GROUP_COLORS, title=f"SHiP muon background: {label}")
    html = outdir / "detector_3d.html"
    fig.write_html(html, include_plotlyjs="cdn")
    print(f"  saved {html}")

    side = maps.side_histograms(data, data["z_start"], data["z_end"], w)
    plotting.save_fig(maps.plot_side(side, label), "side_views", outdir)
    ubt_plane(data, outdir, args.source)
    walls = maps.wall_histograms(data["face_in"], data["z_vessel"], data["x_vessel"], data["y_vessel"], w)
    plotting.save_fig(maps.plot_wall_maps(walls, f"SBT: where muons enter the decay vessel ({args.source} data)"),
                      "sbt_walls", outdir)
    field_map(outdir)


if __name__ == "__main__":
    main()
