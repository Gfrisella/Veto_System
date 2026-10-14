#!/usr/bin/env python3
"""3D + side views of Guglielmo's full dataset (Full_dataset_z_3060_8200.pkl), one sample per muon type.

Every simulated muon (506 M) has two states: A at z = 30.6 m and B at z = 82 m, or the point
where it stopped (left the world or ranged out) if it never got there. The pickle is read with
io.open_pickle_columns: nothing is unpickled, columns are memory-mapped and read in chunks.
There is no start point and no trajectory in between, so tracks are drawn as the chord A -> B.
The straight line from A's momentum (dashed) shows how far the muon scattered or bounced.
Bounced muons are drawn A -> K -> B instead, K being the estimated bounce point: the closest
approach of the line forward from A and the line backward from B. For muons bounced into the
SST1 acceptance the two lines meet within 4 cm (median), at x ~ -4.8 m: the -x cavern wall
(a few at y ~ -3.4 m: the floor). Bounced muons without a credible single bounce (see
bounce_points) are not drawn as a chord; the 3D view lists them as a hidden dotted entry.
Dashed straight lines from A are hidden by default and stop at the |x|, |y| = 10 m box.

Types (exclusive, first match wins):
  bounced back into SST1 acceptance                in acceptance at z = 82 but > 50 cm off the straight
                                                   line from A: scattered off the cavern wall (-x side)
  SST1 acceptance, p > 1 GeV & timing acceptance   the inputs of the pair background
  SST1 acceptance, other                           |x| < 2, |y| < 3 at z = 82
  enters vessel through SBT wall                   straight line from A crosses a lateral wall
  enters vessel through UBT window                 straight line from A crosses the entrance window
  bounced / scattered                              reaches z = 82 > 50 cm away from its straight line
  straight past the vessel                         reaches z = 82 within 50 cm of its straight line
  stops between z = 30.6 and 82                    B is where it stopped (mostly rock at x ~ -5 m)
  stops before z = 30.6                            only A, the stop point (shield, rock)

The pass keeps all muons of the rare types and a random sample of the others (--n-sample);
--n-draw of each type are drawn.

Produces in results/visualization/full/:
  samples.npz        the drawn muons and per-type totals (redraw with --replot)
  summary.txt        counts and weight (muons per spill) of every type
  detector_3d.html   interactive 3D view; click legend entries to toggle types (needs plotly)
  side_views.png     the same tracks in x-z and y-z

  bin/remote.sh run bin/visualize_full_dataset.py      # pass over the 32 GB file, ~1 min
  bin/remote.sh fetch-results visualization/full
  bin/visualize_full_dataset.py --replot               # locally, with plotly
"""

import _bootstrap  # noqa: F401

import argparse
import time

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from veto import geometry as geo
from veto import io, paths, plotting
from veto import muon_types as mt
from veto.tracks import vessel_crossings

try:                      # not in the cluster container: draw with --replot where plotly is installed
    import plotly.graph_objects as go
    from veto import viz3d
except ImportError:
    go = None

STATE = ("x", "y", "z", "px", "py", "pz")
REACH_A, REACH_B = geo.Z_UBT - 0.01, geo.Z_SST1 - 0.01
BOUNCE_CUT = 0.5

# name, colour, draw all (None) or a sample
TYPES = [
    ("bounced back into SST1 acceptance (> 50 cm off its straight line)", "#9400d3", None),
    ("SST1 acceptance, p > 1 GeV & timing acceptance", "#d62728", None),
    ("SST1 acceptance, other", "#ff9896", None),
    ("enters vessel through SBT wall", "#1f77b4", None),
    ("enters vessel through UBT window", "#ff7f0e", None),
    (f"bounced / scattered in the rock (> {100 * BOUNCE_CUT:.0f} cm off its straight line at z = 82)", "#e377c2",
     "sample"),
    ("straight past the vessel to z = 82", "#7f7f7f", "sample"),
    ("stops between z = 30.6 and 82 m", "#8c564b", "sample"),
    ("stops before z = 30.6 m", "#2b2b2b", "sample"),
]


def classify(d):
    """Type index (0..8) per muon, and the straight-line prediction of A at B's z."""
    n = len(d["x0"])
    reach_a, reach_b = d["z0"] > REACH_A, d["z1"] > REACH_B
    p1 = np.sqrt(d["px1"] ** 2 + d["py1"] ** 2 + d["pz1"] ** 2)
    dz = geo.Z_TIMING - d["z1"]
    with np.errstate(divide="ignore", invalid="ignore"):
        x_t = d["x1"] + d["px1"] / d["pz1"] * dz
        y_t = d["y1"] + d["py1"] / d["pz1"] * dz
        t = (d["z1"] - d["z0"]) / d["pz0"]
    acc = reach_b & (np.abs(d["x1"]) < geo.SST1_HALF_X) & (np.abs(d["y1"]) < geo.SST1_HALF_Y)
    good = acc & (p1 > 1) & (d["pz1"] > 0) & (np.abs(x_t) < geo.TIMING_HALF_X) & (np.abs(y_t) < geo.TIMING_HALF_Y)
    off = np.hypot(d["x0"] + d["px0"] * t - d["x1"], d["y0"] + d["py0"] * t - d["y1"])
    off = np.where(reach_b & (d["pz0"] > 0), off, np.inf)

    face = np.zeros(n, dtype=np.int8)
    m = reach_a & (d["pz0"] > 0)
    face[m] = vessel_crossings(*(d[f"{k}0"][m].astype(np.float64) for k in STATE))["face_in"]

    kind = np.full(n, 8, dtype=np.int8)
    kind[reach_a & ~reach_b] = 7
    kind[reach_b] = 6
    kind[reach_b & (off > BOUNCE_CUT)] = 5
    kind[face == 1] = 4
    kind[face >= 2] = 3
    kind[acc] = 2
    kind[good] = 1
    kind[acc & (off > BOUNCE_CUT)] = 0
    return kind


def collect(columns, n, n_sample, chunk, rng):
    """One pass: totals per type, all rows of the rare types and a uniform random sample of the rest."""
    k = len(TYPES)
    count, weight = np.zeros(k, dtype=np.int64), np.zeros(k)
    kept = [dict(key=np.empty(0), rows=[]) for _ in range(k)]
    start = time.time()
    for a in range(0, n, chunk):
        d = {c: np.asarray(columns[c][a:a + chunk]) for c in columns if c != "evt_n"}
        kind = classify(d)
        w = d["weight"].astype(np.float64)
        count += np.bincount(kind, minlength=k)
        weight += np.bincount(kind, weights=w, minlength=k)
        for i, (_, _, mode) in enumerate(TYPES):
            idx = np.where(kind == i)[0]
            keys = rng.random(len(idx))           # keep the n_sample smallest keys overall -> uniform
            if mode == "sample" and len(idx) > n_sample:
                top = np.argpartition(keys, n_sample)[:n_sample]
                idx, keys = idx[top], keys[top]
            rows = {c: v[idx] for c, v in d.items()}
            kept[i]["rows"].append(rows)
            kept[i]["key"] = np.concatenate([kept[i]["key"], keys])
            if mode == "sample" and len(kept[i]["key"]) > n_sample:
                merged = io.concatenate(kept[i]["rows"])
                top = np.argpartition(kept[i]["key"], n_sample)[:n_sample]
                kept[i] = dict(key=kept[i]["key"][top], rows=[{c: v[top] for c, v in merged.items()}])
        print(f"  {min(a + chunk, n):,} / {n:,} muons, {time.time() - start:.0f} s", flush=True)
    samples = [io.concatenate(s["rows"]) for s in kept]
    return count, weight, samples


def bounce_points(rows):
    """Estimated bounce point K and credibility mask (see veto.muon_types.bounce_points)."""
    def stack(fmt):
        return np.stack([rows[fmt.format(k)] for k in "xyz"], 1)
    return mt.bounce_points(stack("{}0"), stack("p{}0"), stack("{}1"), stack("p{}1"))


def paths_of(i, rows):
    """Points (x, y, z) per muon: A -> B, or A -> K -> B for credible bounces of the bounced types.

    Returns (paths, kinks, unresolved): for the bounced types, muons without a credible single
    bounce are not drawn as a chord (the muon never went that way) but returned separately."""
    a = np.stack([rows["x0"], rows["y0"], rows["z0"]], 1)
    b = np.stack([rows["x1"], rows["y1"], rows["z1"]], 1)
    if "bounced" not in TYPES[i][0]:
        return [[pa, pb] for pa, pb in zip(a, b)], None, []
    k, ok = bounce_points(rows)
    paths = [[pa, pk, pb] for pa, pk, pb in zip(a[ok], k[ok], b[ok])]
    unresolved = [[pa, pb] for pa, pb in zip(a[~ok], b[~ok])]
    return paths, k[ok], unresolved


def surface_of(k):
    """Which cavern surface a bounce point is nearest to."""
    return np.where(k[:, 0] - geo.CAVERN_X_WALL < k[:, 1] - geo.CAVERN_Y_FLOOR, "wall x", "floor y")


def polylines(paths):
    """Paths in plotly (z, x, y) coordinates, separated by None."""
    zs, xs, ys = [], [], []
    for path in paths:
        for x, y, z in path:
            zs.append(z); xs.append(x); ys.append(y)
        zs.append(None); xs.append(None); ys.append(None)
    return zs, xs, ys


def straight(rows, z_end, half=10.0):
    """Straight line from A towards z_end, stopped where it leaves the |x|, |y| < half box.

    Returns the end point (x, y, z)."""
    tx, ty = rows["px0"] / rows["pz0"], rows["py0"] / rows["pz0"]
    with np.errstate(divide="ignore"):
        dz_x = np.where(tx != 0, (np.sign(tx) * half - rows["x0"]) / tx, np.inf)
        dz_y = np.where(ty != 0, (np.sign(ty) * half - rows["y0"]) / ty, np.inf)
    z = np.minimum(z_end, rows["z0"] + np.minimum(dz_x, dz_y))
    return rows["x0"] + tx * (z - rows["z0"]), rows["y0"] + ty * (z - rows["z0"]), z


def lines(x0, y0, z0, x1, y1, z1):
    """Segments in plotly (z, x, y) coordinates, separated by None."""
    n = len(x0)
    out = []
    for a, b in ((z0, z1), (x0, x1), (y0, y1)):
        v = np.empty(3 * n, dtype=object)
        v[0::3], v[1::3], v[2::3] = a, b, None
        out.append(v)
    return out


def hover(rows):
    p0 = np.sqrt(rows["px0"] ** 2 + rows["py0"] ** 2 + rows["pz0"] ** 2)
    p1 = np.sqrt(rows["px1"] ** 2 + rows["py1"] ** 2 + rows["pz1"] ** 2)
    return [f"A: z={za:.2f} x={xa:.2f} y={ya:.2f} p={pa:.2f} GeV<br>B: z={zb:.2f} x={xb:.2f} y={yb:.2f} "
            f"p={pb:.2f} GeV<br>weight {w:.3g}, pdg {q:+.0f}"
            for za, xa, ya, pa, zb, xb, yb, pb, w, q in
            zip(rows["z0"], rows["x0"], rows["y0"], p0, rows["z1"], rows["x1"], rows["y1"], p1,
                rows["weight"], rows["pdg"])]


def cavern_traces():
    """The two cavern surfaces the analysis uses for 'bouncing' (approximate, drawn z = 30.6..82)."""
    z = [geo.Z_UBT, geo.Z_SST1]
    wall = go.Surface(x=[z, z], y=[[geo.CAVERN_X_WALL] * 2] * 2, z=[[geo.CAVERN_Y_FLOOR] * 2, [6, 6]],
                      colorscale=[[0, "sienna"], [1, "sienna"]], showscale=False, opacity=0.12,
                      name="cavern wall x = -4.56 m / floor y = -3.36 m (approx.)", showlegend=True,
                      legendgroup="cavern", hoverinfo="name")
    floor = go.Surface(x=[z, z], y=[[geo.CAVERN_X_WALL] * 2, [6, 6]], z=[[geo.CAVERN_Y_FLOOR] * 2] * 2,
                       colorscale=[[0, "sienna"], [1, "sienna"]], showscale=False, opacity=0.12,
                       showlegend=False, legendgroup="cavern", hoverinfo="name",
                       name="cavern floor y = -3.36 m (approx.)")
    planes = [viz3d.frustum_edges(z_p, z_p, 10, 10, 10, 10, f"scoring plane z = {z_p} m (±10 m)", "black",
                                  legendgroup="scoring", dash="dot") for z_p in (geo.Z_UBT, geo.Z_SST1)]
    planes[0].update(showlegend=True, name="scoring planes z = 30.6 / 82 m (±10 m)")
    return [wall, floor] + planes


def figure_3d(count, weight, samples):
    fig = viz3d.detector_figure(title="SHiP full muon dataset (TRY6): one sample per muon type. "
                                      "Solid: chord between the two stored states. Dashed: straight line "
                                      "from the z = 30.6 m state", x_range=(-10, 10), y_range=(-10, 10))
    fig.add_traces(cavern_traces())
    for i, (name, color, mode) in enumerate(TYPES):
        rows = samples[i]
        if len(rows["x0"]) == 0:
            continue
        label = f"{name}: {len(rows['x0'])} of {count[i]:,} ({weight[i]:.3g}/spill)"
        if i == len(TYPES) - 1:   # only the stop point exists
            fig.add_trace(go.Scatter3d(x=rows["z0"], y=rows["x0"], z=rows["y0"], mode="markers",
                                       marker=dict(size=2, color=color), name=label, legendgroup=str(i),
                                       hovertext=hover(rows), hoverinfo="text"))
            continue
        paths, kinks, unresolved = paths_of(i, rows)
        z, x, y = polylines(paths)
        fig.add_trace(go.Scatter3d(x=z, y=x, z=y, mode="lines", line=dict(color=color, width=2), opacity=0.7,
                                   name=label, legendgroup=str(i), hoverinfo="skip"))
        if kinks is not None and len(kinks):
            fig.add_trace(go.Scatter3d(x=kinks[:, 2], y=kinks[:, 0], z=kinks[:, 1], mode="markers",
                                       marker=dict(size=3, color=color, symbol="x"), legendgroup=str(i),
                                       name=f"  estimated bounce points ({len(kinks)})", text=surface_of(kinks),
                                       hovertemplate="bounce (%{text}) z=%{x:.1f} x=%{y:.2f} y=%{z:.2f}"
                                                     "<extra></extra>"))
        if unresolved:
            z, x, y = polylines(unresolved)
            fig.add_trace(go.Scatter3d(x=z, y=x, z=y, mode="lines", line=dict(color=color, width=1, dash="dot"),
                                       opacity=0.5, legendgroup=f"{i}u", visible="legendonly",
                                       name=f"  no single-bounce solution, chord only ({len(unresolved)})",
                                       hoverinfo="skip"))
        fig.add_trace(go.Scatter3d(x=rows["z1"], y=rows["x1"], z=rows["y1"], mode="markers",
                                   marker=dict(size=2, color=color), legendgroup=str(i), showlegend=False,
                                   hovertext=hover(rows), hoverinfo="text"))
        fwd = rows["pz0"] > 0
        xs, ys, zs = straight({k: v[fwd] for k, v in rows.items()}, rows["z1"][fwd])
        z, x, y = lines(rows["x0"][fwd], rows["y0"][fwd], rows["z0"][fwd], xs, ys, zs)
        fig.add_trace(go.Scatter3d(x=z, y=x, z=y, mode="lines", line=dict(color=color, width=1.5, dash="dash"),
                                   opacity=0.5, name=f"  straight line from z = 30.6 ({name.split(' (')[0]})",
                                   legendgroup=f"{i}s", visible="legendonly", hoverinfo="skip"))
    fig.update_layout(legend=dict(itemsizing="constant", font=dict(size=11)), title=dict(font=dict(size=13)))
    return fig


def side_views(count, samples):
    fig, axes = plt.subplots(2, 1, figsize=(20, 14), sharex=True)
    for ax, axis in zip(axes, ("x", "y")):
        plotting.draw_detector_side(ax, axis)
        for i, (name, color, mode) in enumerate(TYPES):
            rows = samples[i]
            if len(rows["x0"]) == 0:
                continue
            a0, a1 = rows[f"{axis}0"], rows[f"{axis}1"]
            label = f"{name} ({len(a0):,} of {count[i]:,})"
            if "bounced" in name:
                label = label[:-1] + "; without a single-bounce solution not drawn)"
            if i == len(TYPES) - 1:
                ax.scatter(rows["z0"], a0, s=2, color=color, label=label)
                continue
            j = "xyz".index(axis)
            paths, kinks, _ = paths_of(i, rows)
            segs = [[(pt[2], pt[j]) for pt in path] for path in paths]
            bounced = kinks is not None
            ax.add_collection(LineCollection(segs, colors=color, lw=1.0 if bounced else 0.6,
                                             alpha=0.9 if bounced else 0.6, label=label,
                                             zorder=3 if bounced else 2))
            if bounced and len(kinks):
                floor = surface_of(kinks) == "floor y"
                ax.scatter(kinks[~floor, 2], kinks[~floor, j], s=14, marker="x", color=color, zorder=4)
                ax.scatter(kinks[floor, 2], kinks[floor, j], s=14, marker="^", color=color, zorder=4)
        wall = geo.CAVERN_X_WALL if axis == "x" else geo.CAVERN_Y_FLOOR
        ax.axhline(wall, color="sienna", ls=":", lw=1.5, label="cavern wall / floor (approx.)")
        ax.set(xlim=(-3, 98), ylim=(-10.5, 10.5), ylabel=f"{axis} [m]")
    axes[0].legend(loc="upper left", fontsize=8, ncol=2)
    axes[1].set_xlabel("z [m]")
    axes[0].set_title("Full dataset: chord between the stored states (z = 30.6 m and z = 82 m, or the stop point); "
                      "bounced muons drawn through their estimated bounce point (x: off the -x wall, ^: off the floor)")
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=str(paths.full_dataset_pickle()))
    parser.add_argument("--n-sample", type=int, default=300, help="muons drawn per common type")
    parser.add_argument("--chunk", type=int, default=25_000_000)
    parser.add_argument("--stop", type=int, default=None, help="only the first N muons (testing)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-draw", type=int, default=15, help="muons drawn per type (from the saved sample)")
    parser.add_argument("--replot", action="store_true", help="redraw from results/.../samples.npz")
    args = parser.parse_args()

    outdir = paths.results_dir() / "visualization" / "full"
    outdir.mkdir(parents=True, exist_ok=True)
    if args.replot:
        saved = np.load(outdir / "samples.npz")
        count, weight, n = saved["count"], saved["weight"], int(saved["n"])
        samples = [{k[len(f"{i}_"):]: saved[k] for k in saved.files if k.startswith(f"{i}_")}
                   for i in range(len(TYPES))]
    else:
        columns, n = io.open_pickle_columns(args.input)
        n = min(n, args.stop or n)
        count, weight, samples = collect(columns, n, args.n_sample, args.chunk, np.random.default_rng(args.seed))
        np.savez(outdir / "samples.npz", count=count, weight=weight, n=n,
                 **{f"{i}_{k}": v for i, rows in enumerate(samples) for k, v in rows.items()})
    rng = np.random.default_rng(args.seed)
    drawn = []
    for rows in samples:
        n_rows = len(rows["x0"])
        pick = np.sort(rng.choice(n_rows, min(args.n_draw, n_rows), replace=False))
        drawn.append({k: v[pick] for k, v in rows.items()})

    lines_out = [f"{paths.full_dataset_pickle().name}: {n:,} muons, total weight {weight.sum():.4e}", "",
                 f"{'type':<75}{'muons':>14}{'weight':>12}{'drawn':>8}"]
    for i, (name, _, _) in enumerate(TYPES):
        lines_out.append(f"{name:<75}{count[i]:>14,}{weight[i]:>12.4e}{len(samples[i]['x0']):>8}")
    text = "\n".join(lines_out)
    print(text)
    (outdir / "summary.txt").write_text(text + "\n")

    plotting.set_style()
    plotting.save_fig(side_views(count, drawn), "side_views", outdir)
    if go is None:
        print("plotly not installed: fetch the results and run with --replot for the 3D view")
        return
    html = outdir / "detector_3d.html"
    figure_3d(count, weight, drawn).write_html(html, include_plotlyjs="cdn")
    print(f"  saved {html}")


if __name__ == "__main__":
    main()
