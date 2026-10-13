#!/usr/bin/env python3
"""3D weighted muon-density histogram inside the decay vessel (Muon_Flames_rate.py).

Reads the flat MuonsAndMatter output (x, y, z, px, py, pz, weight at one plane),
keeps y <= 0, reports (but, like the original, does not remove) muons that reach
the cavern walls before the vessel end, straight-line propagates every muon to each z slice and mirrors y<0 onto y>0.
Writes muon_distribution_3d.pkl with H[x, y, z] and the bin edges.
"""

import _bootstrap  # noqa: F401

import argparse
import pickle

import numpy as np

from veto import geometry as geo
from veto import io, paths, plotting
from veto.tracks import extrapolate_to_z


def load_flat(filename):
    raw = io.load_pickle(filename)
    data = {key: np.asarray(raw[key], dtype=float) for key in ("x", "y", "px", "py", "pz", "weight")}
    data["z"] = np.asarray(raw["z"], dtype=float) if "z" in raw else np.zeros_like(data["x"])
    lengths = {key: len(v) for key, v in data.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Inconsistent array lengths in {filename}: {lengths}")
    return data


def symmetry_report(x, y, w, nbins=200):
    quadrants = [w[(x >= 0) & (y >= 0)].sum(), w[(x < 0) & (y >= 0)].sum(),
                 w[(x < 0) & (y < 0)].sum(), w[(x >= 0) & (y < 0)].sum()]
    print("Weighted quadrant rates:", ", ".join(f"Q{i + 1}={q:.6g}" for i, q in enumerate(quadrants)))
    H, _, _ = np.histogram2d(x, y, bins=nbins, weights=w)
    for name, mirrored in (("x", H[::-1, :]), ("y", H[:, ::-1])):
        total = H + mirrored
        rel = np.divide(np.abs(H - mirrored), total, out=np.zeros_like(H), where=total > 0)
        print(f"{name} symmetry relative difference: mean {rel[total > 0].mean():.4f}, "
              f"median {np.median(rel[total > 0]):.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(paths.MUONS_AND_MATTER / "data/outputs/outputs_cuda.pkl"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--dx", type=float, default=0.1)
    parser.add_argument("--dy", type=float, default=0.1)
    parser.add_argument("--dz", type=float, default=0.5)
    parser.add_argument("--symmetry-check", action="store_true")
    args = parser.parse_args()

    x_min, x_max = -4.5, 4.5
    y_min, y_max = -3.3, 3.3
    z_min, z_max = geo.Z_VESSEL_MIN, geo.Z_VESSEL_MAX + 0.5

    data = load_flat(args.input)
    if args.symmetry_check:
        symmetry_report(data["x"], data["y"], data["weight"])

    keep = data["y"] <= 0
    print(f"Total particles: {len(keep):,}   with y <= 0: {keep.sum():,}")
    d = {key: value[keep] for key, value in data.items()}

    x_end, y_end = extrapolate_to_z(d["x"], d["y"], d["z"], d["px"], d["py"], d["pz"], z_max)
    discarded = (x_end <= geo.CAVERN_X_WALL) | (y_end <= geo.CAVERN_Y_FLOOR)
    print(f"Reaching the cavern walls: {discarded.sum():,} "
          f"(weight {d['weight'][discarded].sum():.6g} of {d['weight'].sum():.6g})")

    x_edges = np.arange(x_min, x_max + args.dx, args.dx)
    y_edges = np.arange(y_min, y_max + args.dy, args.dy)
    z_edges = np.arange(z_min, z_max, args.dz)
    if z_edges[-1] < z_max:
        z_edges = np.append(z_edges, z_max)

    H = np.zeros((len(x_edges) - 1, len(y_edges) - 1, len(z_edges) - 1))
    for iz in range(len(z_edges) - 1):
        # the original uses the lower edge of the z bin
        xp, yp = extrapolate_to_z(d["x"], d["y"], d["z"], d["px"], d["py"], d["pz"], z_edges[iz])
        m = (xp >= x_min) & (xp < x_max) & (yp >= y_min) & (yp < y_max)
        H[:, :, iz], _, _ = np.histogram2d(xp[m], yp[m], bins=[x_edges, y_edges], weights=d["weight"][m])

    # mirror the y < 0 half onto y > 0
    mid = H.shape[1] // 2
    H_full = np.zeros_like(H)
    H_full[:, :mid, :] = H[:, :mid, :]
    H_full[:, mid:, :] = H[:, :mid, :][:, ::-1, :]
    print(f"Full distribution weight: {H_full.sum():.6g} (expected {2 * H[:, :mid, :].sum():.6g})")

    distribution = {"H": H_full, "x_edges": x_edges, "y_edges": y_edges, "z_edges": z_edges,
                    "dx": args.dx, "dy": args.dy, "dz": args.dz,
                    "y_symmetry": True, "y_range_stored": (y_min, y_max)}
    output = args.output or paths.results_dir() / "muon_distribution_3d.pkl"
    with open(output, "wb") as f:
        pickle.dump(distribution, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {output}, shape {H_full.shape}")

    plotting.set_style()
    plotting.save_fig(plotting.distribution_xy(distribution, iz=0), "distribution_first_plane",
                      paths.results_dir() / "figures")


if __name__ == "__main__":
    main()
