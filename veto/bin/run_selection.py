#!/usr/bin/env python3
"""Muon-pair background selection: single-muon cuts -> pairing -> opposite charge
-> DOCA -> vertex in decay vessel -> IP, then scaled to 15 years of POT.

Pairing modes (each reproduces one of the original analyses):

  per-file     every pair inside one output file, cut flows averaged over files
               (Analysis_singlefiles.py, analysis_multifiles.ipynb)
  time-window  all files combined, muons get a random time in [0, 1) s, pairs within
               +-time_window; weights corrected for repeated evt_n
               (analysis.ipynb, Analysis_multifiles.py)
  kdtree       all files combined, one entry per unique evt_n, pairs of opposite-charge
               tracks passing within candidate_radius inside the vessel (temp1.py)

Example:
  bin/run_selection.py --pairing per-file --max-files 200 --plots
"""

import _bootstrap  # noqa: F401

import argparse
import pickle

import numpy as np

from veto import io, paths, plotting, selection
from veto import pairs as pairing
from veto import weights as wt
from veto.config import add_config_args, config_from_args
from veto.tracks import backtrack_to_lateral, extrapolate, momentum


def run_per_file(files, config):
    single_flows, pair_flows, finals = [], [], []
    for i, filename in enumerate(files):
        if i % 500 == 0:
            print(f"  [{i + 1}/{len(files)}] {filename}")
        data = io.load_output_file(filename)
        mask, single_flow = selection.apply_single_muon_cuts(data, config)
        selected = io.select(data, mask)
        result = selection.analyse_pairs(selected, pairing.all_pairs(len(selected["x"])), config)
        single_flows.append(single_flow)
        pair_flows.append(result["pair_flow"])
        finals.append(result)

    selection.print_average_flow("SINGLE-MUON SELECTION (average over files)", selection.average_flows(single_flows))
    selection.print_average_flow("PAIR SELECTION (average over files)", selection.average_flows(pair_flows))

    final_weight = np.array([r["weights"].sum() for r in finals])
    final_pairs = np.array([len(r["pairs"]) for r in finals])
    print()
    print(f"Final pairs per file      : {final_pairs.mean():.4f} +/- {final_pairs.std(ddof=1):.4f}")
    print(f"Final weighted rate/file  : {final_weight.mean():.6e} +/- {final_weight.std(ddof=1):.6e}")

    merged = {key: np.concatenate([r[key] for r in finals])
              for key in ("doca", "vertex", "momentum", "ip", "x_ip", "y_ip", "weights")}
    merged["final_weight"] = final_weight.mean()
    merged["single_flows"], merged["pair_flows"] = single_flows, pair_flows
    return merged


def run_combined(files, config, mode, plots_dir):
    data = io.load_outputs(files)
    n_files = len(files)

    if mode == "kdtree":
        n_before = len(data["weight"])
        data = wt.keep_unique_events(data, n_files)
        print(f"Unique events: {len(data['weight']):,} of {n_before:,} "
              f"(weight sum {data['weight'].sum():.6e})")
    else:
        data["weight"] = wt.corrected_weights(data["weight"], data["evt_n"], n_files)
        data["time"] = np.random.default_rng(config.get("seed")).uniform(0, 1, len(data["weight"]))
        print(f"Corrected weight sum per file: {data['weight'].sum():.3e} Hz")

    if plots_dir:
        plotting.save_fig(plotting.weight_distribution(data["weight"]), "weight_distribution", plots_dir)
        plotting.save_fig(plotting.cumulative_cut_plot(
            momentum(data), config["p_min"], np.logspace(-1, np.log10(400), 100),
            r"$p$ [GeV/$c$]", "Momentum selection"), "momentum_selection", plots_dir)
        plotting.save_fig(plotting.xy_map(data["x"], data["y"], data["weight"],
                                          r"Muon distribution at SST1 ($z=82$ m)"), "muon_xy_sst1", plots_dir)

    mask, single_flow = selection.apply_single_muon_cuts(data, config)
    selection.print_flow("SINGLE-MUON SELECTION", single_flow)
    data = io.select(data, mask)

    if plots_dir:
        x_t, y_t = extrapolate(data, config["z_timing"])
        plotting.save_fig(plotting.xy_map(x_t, y_t, data["weight"],
                                          f"Muon distribution at z = {config['z_timing']:.2f} m"),
                          "muon_xy_timing", plots_dir)
        z_v, x_v, y_v = backtrack_to_lateral(data["x"], data["y"], data["z"],
                                             data["px"], data["py"], data["pz"])
        plotting.save_fig(plotting.backtracking_validation(data, z_v, x_v, y_v), "decay_vessel_backtracking",
                          plots_dir)

    if mode == "kdtree":
        candidate_pairs = pairing.kdtree_candidate_pairs(
            data, z_step=config["z_sample_step"], radius=config["candidate_radius"])
    else:
        candidate_pairs, n_in_window = pairing.time_window_pairs(data["time"], config["time_window"])
        print(f"Time-compatible pairs (+-{config['time_window'] * 1e12:.0f} ps): {len(candidate_pairs):,}")

    if plots_dir and len(candidate_pairs):
        doca, _, _ = pairing.doca_and_vertex(data, candidate_pairs)
        for name, fig in plotting.doca_plots(doca).items():
            plotting.save_fig(fig, name, plots_dir)

    result = selection.analyse_pairs(data, candidate_pairs, config)
    selection.print_flow("PAIR SELECTION", result["pair_flow"])
    result["final_weight"] = result["weights"].sum()
    result["single_flow"] = single_flow

    if plots_dir and len(result["pairs"]):
        plotting.save_fig(plotting.vertices_and_tracks(data, result["pairs"], result["vertex"]),
                          "selected_vertices", plots_dir)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairing", choices=("per-file", "time-window", "kdtree"), default="per-file")
    parser.add_argument("--pattern", default=None, help="glob of output files (default: VETO_DATA_ROOT/outputs)")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for the random muon times")
    parser.add_argument("--plots", action="store_true", help="save diagnostic plots")
    parser.add_argument("--tag", default=None, help="name of the results sub-folder")
    add_config_args(parser)
    args = parser.parse_args()

    config = config_from_args(args)
    config["seed"] = args.seed
    outdir = paths.results_dir() / "selection" / (args.tag or args.pairing)
    plots_dir = outdir / "figures" if args.plots else None
    plotting.set_style()

    files = io.list_output_files(args.pattern, args.max_files)
    if args.pairing == "per-file":
        result = run_per_file(files, config)
    else:
        result = run_combined(files, config, args.pairing, plots_dir)

    normalization, expected = selection.expected_background(result["final_weight"], config)
    print()
    print("FINAL")
    print("-" * 100)
    print(f"Final pairs                 : {len(result['vertex']):,}")
    print(f"Final weighted rate         : {result['final_weight']:.6e}")
    print(f"15-year normalization       : {normalization:.6e}")
    print(f"Expected 15-year background : {expected:.6e}")

    if plots_dir and len(result["vertex"]):
        for name, fig in plotting.final_vertex_plots(result["vertex"], result["doca"], result["ip"],
                                                     result["x_ip"], result["y_ip"], config["ip_max"]).items():
            plotting.save_fig(fig, name, plots_dir)

    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "result.pkl", "wb") as f:
        pickle.dump({"config": config, "files": files, "result": result,
                     "expected_15_years": expected}, f)
    print(f"Saved {outdir / 'result.pkl'}")


if __name__ == "__main__":
    main()
