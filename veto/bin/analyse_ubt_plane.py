#!/usr/bin/env python3
"""Muons at the UBT plane (UBT_data_TRY6, 150 M muons at z = 30.6 m, |x|, |y| < 3 m).

Every muon is propagated forward in a straight line (no field after the shield) and
clipped against the decay vessel, giving
  entry face : UBT window, one of the four SBT walls, or never inside the vessel
  exit face  : one of the four SBT walls, or the vessel end (z = 80.6 m)
  SST1       : whether the straight line lands inside the SST1 acceptance at z = 82 m

The full store is processed in chunks and only histograms are kept (results/ubt/hists.npz),
so plots can be redone locally with --replot without touching the 4.5 GB store.
Every muon that enters the vessel is saved with its crossings to results/ubt/entering_vessel.npz.

Missing data: muons that do not cross the 6 x 6 m plane at z = 30.6 m are not in this
dataset (e.g. those bouncing off the cavern walls and reaching the SBT/SST further
downstream). Straight lines also ignore scattering in the vessel walls.

  bin/remote.sh run bin/analyse_ubt_plane.py          # full pass on the cluster (~minutes)
  bin/remote.sh fetch-results ubt
  bin/analyse_ubt_plane.py --replot                    # plots from hists.npz
"""

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from veto import geometry as geo
from veto import io, maps, paths, plotting
from veto.tracks import FACE_NAMES, extrapolate_to_z, vessel_crossings

XY_EDGES = (np.linspace(-3, 3, 121), np.linspace(-3, 3, 121))
SST1_EDGES = (np.linspace(-6, 6, 121), np.linspace(-6, 6, 121))
P_EDGES = np.logspace(-1.5, np.log10(400), 81)
FATES = ("UBT window", "SBT wall (downstream)", "never in vessel")


def fate_of(face_in):
    return np.select([face_in == 1, face_in >= 2], [0, 1], default=2)


def accumulate(chunk, hists, entering):
    w = chunk["weight"].astype(np.float64)
    c = vessel_crossings(chunk["x"], chunk["y"], chunk["z"], chunk["px"], chunk["py"], chunk["pz"])
    fate = fate_of(c["face_in"])
    p = np.sqrt(chunk["px"].astype(np.float64) ** 2 + chunk["py"] ** 2 + chunk["pz"] ** 2)
    x82, y82 = extrapolate_to_z(*(chunk[k].astype(np.float64) for k in ("x", "y", "z", "px", "py", "pz")),
                                geo.Z_SST1)
    in_sst1 = (np.abs(x82) < geo.SST1_HALF_X) & (np.abs(y82) < geo.SST1_HALF_Y)

    np.add.at(hists["faces_w"], (c["face_in"], c["face_out"]), w)
    np.add.at(hists["faces_n"], (c["face_in"], c["face_out"]), 1)
    np.add.at(hists["fate_sst1_w"], (fate, in_sst1.astype(int)), w)
    for f in range(3):
        m = fate == f
        hists["xy"][f] += np.histogram2d(chunk["x"][m], chunk["y"][m], bins=XY_EDGES, weights=w[m])[0]
        hists["p"][f] += np.histogram(p[m], bins=P_EDGES, weights=w[m])[0]
    for q, sign in enumerate((1, -1)):
        m = chunk["pdg"] * sign > 0
        hists["xy_charge"][q] += np.histogram2d(chunk["x"][m], chunk["y"][m], bins=XY_EDGES, weights=w[m])[0]
    hists["sbt_in"] += maps.wall_histograms(c["face_in"], c["z_in"], c["x_in"], c["y_in"], w)
    hists["sbt_out"] += maps.wall_histograms(c["face_out"], c["z_out"], c["x_out"], c["y_out"], w)
    hists["sst1_xy"] += np.histogram2d(x82, y82, bins=SST1_EDGES, weights=w)[0]

    # keep every muon that enters the vessel (a tiny fraction) with its crossings
    m = c["face_in"] > 0
    entering.append({**{k: v[m] for k, v in chunk.items()}, **{k: v[m] for k, v in c.items()},
                     "in_sst1": in_sst1[m]})


def empty_hists():
    nz, na = len(maps.WALL_Z_EDGES) - 1, len(maps.WALL_ALONG_EDGES) - 1
    return {
        "faces_w": np.zeros((7, 7)), "faces_n": np.zeros((7, 7), dtype=np.int64),
        "fate_sst1_w": np.zeros((3, 2)),
        "xy": np.zeros((3, 120, 120)), "xy_charge": np.zeros((2, 120, 120)),
        "p": np.zeros((3, len(P_EDGES) - 1)),
        "sbt_in": np.zeros((4, nz, na)), "sbt_out": np.zeros((4, nz, na)),
        "sst1_xy": np.zeros((120, 120)),
    }


def summary(h):
    lines = []
    total = h["faces_w"].sum()
    lines.append(f"Total muons {h['faces_n'].sum():,}, total weight {total:.6e}")
    lines.append("")
    lines.append(f"{'Fate (straight line from the UBT plane)':<36}{'muons':>12}{'weight':>13}{'fraction':>12}"
                 f"{'-> SST1 acc.':>14}")
    for f, name in enumerate(FATES):
        wf = h["fate_sst1_w"][f].sum()
        n = h["faces_n"][1].sum() if f == 0 else h["faces_n"][2:6].sum() if f == 1 else h["faces_n"][0].sum()
        lines.append(f"  {name:<34}{n:>12,}{wf:>13.4e}{wf / total:>12.2e}{h['fate_sst1_w'][f, 1]:>14.4e}")
    lines.append("")
    lines.append("Entry face -> exit face (weight)")
    faces = [1, 2, 3, 4, 5]
    lines.append(" " * 18 + "".join(f"{FACE_NAMES[o]:>14}" for o in (2, 3, 4, 5, 6)))
    for i in faces:
        lines.append(f"  {FACE_NAMES[i]:<16}" + "".join(f"{h['faces_w'][i, o]:>14.4e}" for o in (2, 3, 4, 5, 6)))
    lines.append(f"  {'never inside':<16}{h['faces_w'][0, 0]:>14.4e}")
    lines.append("")
    hits = h["sbt_in"].sum() + h["sbt_out"].sum()
    lines.append(f"Muons touching any SBT wall (entering or leaving): weight "
                 f"{h['faces_w'][2:6, :].sum() + h['faces_w'][1, 2:6].sum():.4e}")
    lines.append(f"SBT wall crossings (entries + exits) weight: {hits:.6e}")
    return "\n".join(lines)


def make_plots(h, outdir):
    total = h["faces_w"].sum()
    for f, name in enumerate(FATES):
        fig = maps.plot_xy(h["xy"][f], XY_EDGES, f"UBT plane: {name} ({100 * h['xy'][f].sum() / total:.1f} %)")
        plotting.save_fig(fig, f"ubt_xy_{name.split()[0].lower()}_{f}", outdir)
    fig = maps.plot_xy(h["xy"].sum(0), XY_EDGES, "All muons at the UBT plane (z = 30.6 m)")
    plotting.save_fig(fig, "ubt_xy_all", outdir)
    for q, name in enumerate(("mu+", "mu-")):
        plotting.save_fig(maps.plot_xy(h["xy_charge"][q], XY_EDGES, f"{name} at the UBT plane"),
                          f"ubt_xy_{'plus' if q == 0 else 'minus'}", outdir)

    fig, ax = plt.subplots(figsize=(10, 7))
    centers = np.sqrt(P_EDGES[:-1] * P_EDGES[1:])
    for f, name in enumerate(FATES):
        ax.step(centers, h["p"][f], where="mid", lw=2, label=name)
    ax.set(xscale="log", yscale="log", xlabel="p [GeV/c]", ylabel="weighted muons / bin",
           title="Momentum at the UBT plane")
    ax.grid(alpha=0.2)
    ax.legend()
    plotting.save_fig(fig, "ubt_momentum", outdir)

    plotting.save_fig(maps.plot_wall_maps(h["sbt_in"], "SBT: muons from the UBT plane entering the vessel"),
                      "sbt_entering", outdir)
    plotting.save_fig(maps.plot_wall_maps(h["sbt_out"], "SBT: muons leaving the vessel through the walls"),
                      "sbt_leaving", outdir)
    plotting.save_fig(maps.plot_wall_maps(h["sbt_in"] + h["sbt_out"], "SBT: all wall crossings"),
                      "sbt_all_crossings", outdir)

    fig = maps.plot_xy(h["sst1_xy"], SST1_EDGES, "Straight-line extrapolation to SST1 (z = 82 m)", ubt=False)
    ax = fig.axes[0]
    ax.add_patch(plt.Rectangle((-geo.SST1_HALF_X, -geo.SST1_HALF_Y), 2 * geo.SST1_HALF_X, 2 * geo.SST1_HALF_Y,
                               fill=False, ec="red", ls="--", lw=2, label="SST1"))
    ax.legend(loc="upper right")
    plotting.save_fig(fig, "sst1_xy_from_ubt", outdir)

    if "side" in h:
        plotting.save_fig(maps.plot_side(h["side"], "UBT-plane muons propagated forward (subsample)"),
                          "side_views", outdir)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default=str(paths.store_dir("ubt_TRY6")))
    parser.add_argument("--sample", default=str(paths.store_dir("ubt_TRY6_sample.npz")))
    parser.add_argument("--chunk-size", type=int, default=10_000_000)
    parser.add_argument("--max-rows", type=int, default=None, help="process only the first N rows")
    parser.add_argument("--replot", action="store_true", help="only redo plots from results/ubt/hists.npz")
    args = parser.parse_args()

    outdir = paths.results_dir() / "ubt"
    outdir.mkdir(parents=True, exist_ok=True)
    plotting.set_style()

    if args.replot:
        with np.load(outdir / "hists.npz") as f:
            h = {k: f[k] for k in f.files}
    else:
        h = empty_hists()
        entering = []
        for start, chunk in io.iterate_store(Path(args.store), args.chunk_size, stop=args.max_rows):
            print(f"  rows {start:,} ...", flush=True)
            accumulate(chunk, h, entering)
        entering = io.concatenate(entering)
        np.savez_compressed(outdir / "entering_vessel.npz", **entering)
        print(f"Saved {len(entering['x']):,} vessel-entering muons to {outdir / 'entering_vessel.npz'}")

        # side views from the random subsample (the full set would need ~10^10 samples)
        sample = dict(np.load(args.sample))
        h["side"] = maps.side_histograms(sample, sample["z"].astype(np.float64),
                                         np.full(len(sample["x"]), geo.Z_SST1), sample["weight"].astype(np.float64))
        np.savez_compressed(outdir / "hists.npz", **h)

    # path inside the vessel: every vessel-entering muon (the subsample has only ~15 of them)
    entering = dict(np.load(outdir / "entering_vessel.npz"))
    h["side_in_vessel"] = maps.side_histograms(entering, entering["z_in"], entering["z_out"],
                                               entering["weight"].astype(np.float64))

    text = summary(h)
    print(text)
    (outdir / "summary.txt").write_text(text + "\n")
    make_plots(h, outdir / "figures")
    if "side_in_vessel" in h:
        plotting.save_fig(maps.plot_side(h["side_in_vessel"], "UBT-plane muons: path inside the decay vessel"),
                          "side_views_in_vessel", outdir / "figures")


if __name__ == "__main__":
    main()
