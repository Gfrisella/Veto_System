#!/usr/bin/env python3
"""Build muon_vessel_dataset.npz (cell 21 of analysis.ipynb): muons passing the momentum
and timing-detector cuts, with their back-projection to the UBT plane, the point where
they entered the decay vessel (UBT window or SBT wall) and their category.

Input for the SBT/UBT optimisation (bin/optimize_sbt.py, bin/visualize_detector.py).
"""

import _bootstrap  # noqa: F401

import argparse

import numpy as np

from veto import geometry as geo
from veto import io, paths, selection
from veto import weights as wt
from veto.config import add_config_args, config_from_args
from veto.tracks import backtrack_to_lateral, categories, entry_wall, extrapolate


def make_vessel_dataset(data):
    x_ubt, y_ubt = extrapolate(data, geo.Z_UBT)
    z_vessel, x_vessel, y_vessel = backtrack_to_lateral(
        data["x"], data["y"], data["z"], data["px"], data["py"], data["pz"])
    dataset = {key: data[key] for key in ("x", "y", "z", "px", "py", "pz", "weight", "pdg", "evt_n")}
    dataset.update(x_ubt=x_ubt, y_ubt=y_ubt, x_vessel=x_vessel, y_vessel=y_vessel, z_vessel=z_vessel,
                   entry_wall=entry_wall(x_vessel, y_vessel, z_vessel))
    if "weight_corrected" in data:
        dataset["weight_corrected"] = data["weight_corrected"]
    dataset.update(categories(x_ubt, y_ubt, data["pdg"]))
    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--output", default=None)
    add_config_args(parser)
    args = parser.parse_args()
    config = config_from_args(args)

    files = io.list_output_files(args.pattern, args.max_files)
    data = io.load_outputs(files)
    data["weight_corrected"] = wt.corrected_weights(data["weight"], data["evt_n"], len(files))
    mask, flow = selection.apply_single_muon_cuts(data, config)
    selection.print_flow("SINGLE-MUON SELECTION", flow)

    dataset = make_vessel_dataset(io.select(data, mask))
    output = args.output or paths.results_dir() / "muon_vessel_dataset.npz"
    np.savez_compressed(output, **dataset)
    print(f"Saved {len(dataset['x']):,} muons to {output}")


if __name__ == "__main__":
    main()
