#!/usr/bin/env python3
"""Plots of muon_distribution_3d.pkl (cell 32 of analysis.ipynb): x-y slices and side projections."""

import _bootstrap  # noqa: F401

import argparse

import numpy as np

from veto import io, paths, plotting


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None, help="default: VETO_DATA_ROOT/data/muon_distribution_3d.pkl")
    parser.add_argument("--z", type=float, nargs="*", default=[30.6, 55.6, 80.6], help="z of the x-y slices")
    parser.add_argument("--all-z", action="store_true", help="one x-y plot per z bin")
    parser.add_argument("--linear", action="store_true")
    args = parser.parse_args()

    distribution = io.load_distribution(args.input)
    H, ze = distribution["H"], distribution["z_edges"]
    print(f"Histogram shape {H.shape}, z {ze[0]} -> {ze[-1]}, total weight {H.sum():.6g}")

    outdir = paths.results_dir() / "figures" / "muon_distribution"
    plotting.set_style()
    log = not args.linear
    centers = 0.5 * (ze[:-1] + ze[1:])
    slices = range(H.shape[2]) if args.all_z else [int(np.argmin(np.abs(centers - z))) for z in args.z]
    for iz in slices:
        plotting.save_fig(plotting.distribution_xy(distribution, iz, log), f"xy_z{ze[iz]:.1f}", outdir)
    for axis in "xy":
        plotting.save_fig(plotting.distribution_side(distribution, axis, log), f"z{axis}_projection", outdir)


if __name__ == "__main__":
    main()
