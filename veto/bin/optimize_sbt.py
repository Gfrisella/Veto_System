#!/usr/bin/env python3
"""Starting point for the SBT/UBT optimisation (Optimization_SBT.ipynb).

Loads muon_vessel_dataset.npz and summarises how much muon rate enters the decay
vessel through each face and per muon category. The optimisation itself is still
to be written.
"""

import _bootstrap  # noqa: F401

import argparse

import numpy as np

from veto import io
from veto.tracks import WALL_NAMES, entry_wall


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, help="default: VETO_DATA_ROOT/data/muon_vessel_dataset.npz")
    args = parser.parse_args()

    data = io.load_vessel_dataset(args.dataset)
    w = data["weight"].astype(float)
    print(f"Muons: {len(w):,}   total weight: {w.sum():.6e}")

    # older datasets do not store the entry face
    wall = data["entry_wall"] if "entry_wall" in data else entry_wall(data["x_vessel"], data["y_vessel"],
                                                                     data["z_vessel"])
    print()
    print(f"{'entry face':<22}{'muons':>12}{'weight fraction':>18}")
    for label, name in WALL_NAMES.items():
        m = wall == label
        print(f"{name:<22}{m.sum():>12,}{100 * w[m].sum() / w.sum():>17.3f}%")

    print()
    print(f"{'category':<22}{'muons':>12}{'weight fraction':>18}")
    for name in ("front", "bouncing", "cross_over"):
        m = data[name].astype(bool)
        print(f"{name:<22}{m.sum():>12,}{100 * w[m].sum() / w.sum():>17.3f}%")

    finite = np.isfinite(data["z_vessel"])
    print()
    print(f"Vessel entry z: median {np.median(data['z_vessel'][finite]):.2f} m, "
          f"5-95% {np.percentile(data['z_vessel'][finite], 5):.2f} - "
          f"{np.percentile(data['z_vessel'][finite], 95):.2f} m")


if __name__ == "__main__":
    main()
