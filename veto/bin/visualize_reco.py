#!/usr/bin/env python3
"""What the veto reconstruction does on a few designs: interactive 3D events + summary figures.

Per design (results/reco/visualization/<design>.html), real events from algorithm A:
  vetoed background   muon pairs (true path A -> K -> B, vertex), the hit that explained each muon
                      and the backward prediction that found it
  missed background   every surviving pair, with the planes its muons crossed but were not seen on
  signal lost (FP)    hypothetical decays killed by an accidental hit: tracks, prediction, the hit
  missed bounces      bounced SST1 muons not vetoed: true path vs. the backward prediction
  accidental hits     all random hits of one event window
  detectors           coloured by muons per m^2 per spill
Plus results/reco/visualization/<design>_maps.png (rate maps, which planes veto / cause FP) and
comparison.png over the designs.

  bin/visualize_reco.py                              # D0, D1, D2, D4
  bin/visualize_reco.py D3_heavy_right D5_fine
"""

import _bootstrap  # noqa: F401

import argparse

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from run_reco import TYPE_NAMES, load_inputs, take
from veto import design as dsg
from veto import paths, plotting, reco, viz3d

DEFAULT = ["D0_minimal", "D1_wide_ubt", "D2_default", "D4_dense"]
N_SHOW = 8


def path_points(tr, i):
    pts = [(tr["ax"][i], tr["ay"][i], tr["az"][i])]
    if np.isfinite(tr["kz"][i]):
        pts.append((tr["kx"][i], tr["ky"][i], tr["kz"][i]))
    pts.append((tr["bx"][i], tr["by"][i], tr["bz"][i]))
    return pts


def lines(paths_, **kw):
    zs, xs, ys = [], [], []
    for pts in paths_:
        for x, y, z in pts:
            zs.append(float(z)); xs.append(float(x)); ys.append(float(y))
        zs.append(None); xs.append(None); ys.append(None)
    return go.Scatter3d(x=zs, y=xs, z=ys, mode="lines", hoverinfo="skip", **kw)


def markers(pts, text=None, **kw):
    pts = np.asarray(pts, float).reshape(-1, 3)
    return go.Scatter3d(x=pts[:, 2], y=pts[:, 0], z=pts[:, 1], mode="markers", text=text,
                        hoverinfo="text" if text is not None else "skip", **kw)


def hit_xyz(design, hits, idx):
    """Global (x, y, z) of hits idx (cell centres)."""
    out = []
    for h in np.atleast_1d(idx):
        plane = design.planes[int(hits["plane"][h])]
        out.append((plane.to_global(hits["u"][h]), hits["y"][h], plane.z))
    return out


def prediction_line(meas, i, z_to):
    """Backward straight line of measured track i from z = 82 to z_to."""
    dz = meas["z"][i] - z_to
    return [(meas["x"][i], meas["y"][i], meas["z"][i]),
            (meas["x"][i] - meas["tx"][i] * dz, meas["y"][i] - meas["ty"][i] * dz, z_to)]


def detector_traces(design, inp, spill):
    """UBT and fins as rectangles coloured by muons / m^2 / spill (log10)."""
    rates = reco.plane_rates(design, inp)
    out = []
    for k, (plane, r) in enumerate(zip(design.planes, rates)):
        density = r["rate"] / max(plane.area, 1e-9)
        u = [0, plane.u_max, plane.u_max, 0]
        x = [plane.to_global(v) for v in u]
        y = [plane.y_lo, plane.y_lo, plane.y_hi, plane.y_hi]
        out.append(go.Mesh3d(x=[plane.z] * 4, y=x, z=y, i=[0, 0], j=[1, 2], k=[2, 3],
                             intensity=[np.log10(max(density, 1))] * 4, cmin=2, cmax=9, colorscale="Viridis",
                             showscale=k == 0, opacity=0.55, flatshading=True,
                             colorbar=dict(title="log10 muons<br>/ m² / spill", x=1.0, len=0.5),
                             name=f"{plane.name}: {r['rate']:.3g} muons/spill, peak cell {r['peak_cell'] / spill:.3g} Hz",
                             hoverinfo="name", showlegend=False))
    return out


def figure(design, tr, inp, cfg, n_signal, rng_seed):
    rng = np.random.default_rng(rng_seed)
    traces = [t for t in viz3d.geometry_traces() if t.name != "UBT"] + detector_traces(design, inp, cfg.spill_s)

    # ---- background candidates
    cand = inp["cand"]
    rows = cand.ravel()
    sub = take(tr, rows)
    event = np.repeat(np.arange(len(cand)), 2)
    vz = np.repeat(inp["cand_vertex"][:, 2], 2)
    b = reco.run_events(sub, event, vz, design, inp, cfg, rng, details=True)
    q, hits, hoq = b["queries"], b["hits"], b["hit_of_query"]
    pick = np.random.default_rng(5)
    vetoed_ev = pick.choice(np.where(b["vetoed"])[0], min(N_SHOW, b["vetoed"].sum()), replace=False)
    missed_ev = np.where(~b["vetoed"])[0][:40]

    paths_v, hit_pts, pred_v, vtx_v = [], [], [], []
    for e in vetoed_ev:
        vtx_v.append(inp["cand_vertex"][e])
        for t in (2 * e, 2 * e + 1):
            paths_v.append(path_points(sub, t))
            found = np.where((q["track"] == t) & b["found"])[0]
            if len(found):
                h = hoq[found[0]]
                hit_pts += hit_xyz(design, hits, h)
                pred_v.append(prediction_line(b["meas"], t, design.planes[int(hits["plane"][h])].z))
    traces += [lines(paths_v, line=dict(color="#2ca02c", width=3), name=f"background pair, VETOED ({len(vetoed_ev)} shown)",
                     legendgroup="v"),
               lines(pred_v, line=dict(color="#2ca02c", width=2, dash="dot"), name="  backward prediction that found the hit",
                     legendgroup="v"),
               markers(hit_pts, marker=dict(size=6, color="#2ca02c", symbol="diamond", line=dict(color="black", width=1)),
                       name="  hit that explained the muon", legendgroup="v"),
               markers(vtx_v, marker=dict(size=4, color="#2ca02c", symbol="cross"), name="  fake vertex", legendgroup="v")]

    paths_m, crossed, vtx_m, text_m = [], [], [], []
    for e in missed_ev:
        vtx_m.append(inp["cand_vertex"][e])
        for t in (2 * e, 2 * e + 1):
            paths_m.append(path_points(sub, t))
            one = {k: v[t:t + 1] for k, v in sub.items()}
            for plane in design.planes:
                u, y, _, ok = reco.true_crossing(one, plane)
                if ok[0] and plane.z < vz[t]:
                    crossed.append((plane.to_global(u[0]), y[0], plane.z))
                    text_m.append(f"crossed {plane.name}, not detected or not predicted<br>"
                                  f"muon type: {TYPE_NAMES[int(sub['type'][t])]}")
    traces += [lines(paths_m, line=dict(color="#d62728", width=4), name=f"background pair, MISSED (all {len(missed_ev)})",
                     legendgroup="m"),
               markers(crossed, text=text_m, marker=dict(size=7, color="white", line=dict(color="#d62728", width=3)),
                       name="  plane crossed but no veto (inefficiency / bounce)", legendgroup="m"),
               markers(vtx_m, marker=dict(size=4, color="#d62728", symbol="cross"), name="  fake vertex", legendgroup="m")]

    # ---- signal lost to accidental hits
    sig, z_v, ev = reco.signal_tracks(n_signal, rng)
    s = reco.run_events(sig, ev, z_v, design, inp, cfg, rng, with_own_hits=False, details=True)
    fp_ev = np.where(s["vetoed"])[0][:N_SHOW]
    paths_s, pred_s, acc_s, vtx_s = [], [], [], []
    for e in fp_ev:
        for t in np.where(ev == e)[0]:
            vtx = (sig["vx"][t], sig["vy"][t], sig["vz"][t])
            paths_s.append([vtx, (sig["bx"][t], sig["by"][t], sig["bz"][t])])
            found = np.where((s["queries"]["track"] == t) & s["found"])[0]
            if len(found):
                h = s["hit_of_query"][found[0]]
                acc_s += hit_xyz(design, s["hits"], h)
                pred_s.append(prediction_line(s["meas"], t, design.planes[int(s["hits"]["plane"][h])].z))
        vtx_s.append(vtx)
    if len(fp_ev):
        traces += [lines(paths_s, line=dict(color="#ff7f0e", width=4),
                         name=f"signal LOST to an accidental hit ({len(fp_ev)} shown, {100 * s['vetoed'].mean():.2f} % of signal)",
                         legendgroup="s"),
                   lines(pred_s, line=dict(color="#ff7f0e", width=2, dash="dot"), name="  backward prediction",
                         legendgroup="s"),
                   markers(acc_s, marker=dict(size=7, color="black", symbol="x"), name="  accidental hit that vetoed it",
                           legendgroup="s"),
                   markers(vtx_s, marker=dict(size=5, color="#ff7f0e"), name="  decay vertex", legendgroup="s")]

    # ---- bounced single muons that are not vetoed
    single = inp["single"]
    st = take(tr, single)
    m = reco.run_events(st, np.arange(len(single)), np.full(len(single), 1e9), design, inp, cfg, rng, details=True)
    miss_b = np.where(~m["vetoed"] & (st["type"] == 0))[0][:N_SHOW]
    if len(miss_b):
        traces += [lines([path_points(st, i) for i in miss_b], line=dict(color="#9400d3", width=3),
                         name=f"bounced SST1 muon, NOT vetoed ({len(miss_b)} shown)", legendgroup="b"),
                   lines([prediction_line(m["meas"], i, 30.6) for i in miss_b],
                         line=dict(color="#9400d3", width=2, dash="dot"),
                         name="  what the spectrometer extrapolates back", legendgroup="b")]

    # ---- accidental hits of one event window
    e0 = int(vetoed_ev[0]) if len(vetoed_ev) else 0
    acc_idx = np.where((hits["event"] == e0) & (np.arange(len(hits["event"])) >= b["n_own"]))[0]
    if len(acc_idx):
        traces.append(markers(hit_xyz(design, hits, acc_idx), marker=dict(size=2.5, color="#555"),
                              name=f"accidental hits in one ±{cfg.n_sigma_t * np.hypot(design.sigma_t * 1e9, cfg.sigma_t_track) + 0.5:.1f} ns "
                                   f"window ({len(acc_idx)})", visible="legendonly"))

    fig = go.Figure(traces)
    fig.update_layout(
        title=dict(text=f"{design.name}: {design.raw.get('description', '')}<br><sup>background vetoed "
                        f"{100 * b['vetoed'].mean():.3f} % of pairs, {(~b['vetoed']).sum()} missed; signal lost "
                        f"{100 * s['vetoed'].mean():.2f} %; spill {cfg.spill_s:g} s, ε = {design.efficiency}</sup>",
                   font=dict(size=14)),
        scene=dict(xaxis=dict(title="z [m]", range=[-3, 98]), yaxis=dict(title="x [m]", range=[-8, 8]),
                   zaxis=dict(title="y [m]", range=[-6, 8]), aspectmode="manual",
                   aspectratio=dict(x=4, y=0.65, z=0.6), camera=dict(eye=dict(x=-0.5, y=-2.6, z=1.3))),
        legend=dict(itemsizing="constant", font=dict(size=11)), margin=dict(l=0, r=0, t=60, b=0))
    summary = dict(by_plane_veto=b["by_plane"], by_plane_fp=s["by_plane"], single=m, single_types=st["type"],
                   single_w=inp["single_w"], fp=s["vetoed"].mean(), bkg_missed=(~b["vetoed"]).sum(),
                   bkg_w_missed=inp["cand_w"][~b["vetoed"]].sum() / inp["cand_w"].sum())
    return fig, summary


def maps_figure(design, inp, summary, spill):
    planes = design.planes
    fins = [p for p in planes if p.kind == "fin"]
    ncol = min(max(len(fins), 1), 8)
    nrow = int(np.ceil(len(fins) / ncol)) if fins else 0
    fig = plt.figure(figsize=(22, 3.3 * nrow + 5.5), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[3.3 * max(nrow, 1), 5.5]) if fins else (None, fig)
    if fins:
        axes = np.atleast_1d(top.subplots(nrow, ncol, sharey=True)).ravel()
        for ax, plane in zip(axes, fins):
            ub, yb, h = reco.plane_rate_map(plane, inp)
            density = h / np.outer(np.diff(ub), np.diff(yb)) / spill
            im = ax.pcolormesh(ub, yb, np.log10(np.maximum(density.T, 1)), vmin=2, vmax=9, cmap="viridis")
            ax.set_xlim(0, plane.u_max); ax.set_ylim(plane.y_lo, plane.y_hi)
            ax.set_title(plane.name, fontsize=11)
            ax.tick_params(labelsize=9)
        for ax in axes[len(fins):]:
            ax.axis("off")
        top.supxlabel("u: distance from the vessel wall [m]", fontsize=11)
        top.supylabel("y [m]", fontsize=11)
        top.colorbar(im, ax=axes, label="log10 muons / m² / s", shrink=0.8)
    ax1, ax2 = bottom.subplots(1, 2, width_ratios=[1.6, 1])
    names = [p.name for p in planes]
    xpos = np.arange(len(planes))
    ax1.bar(xpos - 0.2, summary["by_plane_veto"], 0.4, label="background muons explained", color="#2ca02c")
    ax1.bar(xpos + 0.2, summary["by_plane_fp"], 0.4, label="accidental matches on signal", color="#ff7f0e")
    ax1.set_xticks(xpos, names, rotation=60, ha="right", fontsize=9)
    ax1.set_yscale("symlog")
    ax1.set(ylabel="count", title="which plane does the work")
    ax1.legend(fontsize=9)
    ty, w, v = summary["single_types"].astype(int), summary["single_w"], summary["single"]["vetoed"]
    eff = [100 * (w * v)[ty == k].sum() / w[ty == k].sum() for k in range(3)]
    ax2.barh([TYPE_NAMES[k] for k in range(3)], eff, color=["#9400d3", "#d62728", "#ff9896"])
    for k, e in enumerate(eff):
        ax2.text(2, k, f"{e:.2f} %", va="center", fontsize=11, color="white", fontweight="bold")
    ax2.set(xlim=(0, 100), xlabel="single muons vetoed [%]", title="single SST1 muons")
    ax2.tick_params(labelsize=10)
    fig.suptitle(f"{design.name}: rates on the fins, which plane vetoes, single-muon efficiency")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("designs", nargs="*", default=DEFAULT)
    parser.add_argument("--spill", type=float, default=1.0)
    parser.add_argument("--n-signal", type=int, default=100_000)
    args = parser.parse_args()
    cfg = reco.RecoConfig(spill_s=args.spill)
    tr, inp = load_inputs()
    outdir = paths.results_dir() / "reco" / "visualization"
    outdir.mkdir(parents=True, exist_ok=True)
    plotting.set_style()
    rows = []
    for name in args.designs:
        design = dsg.load(name)
        fig, summary = figure(design, tr, inp, cfg, args.n_signal, cfg.seed)
        html = outdir / f"{name}.html"
        fig.write_html(html, include_plotlyjs="cdn")
        plotting.save_fig(maps_figure(design, inp, summary, args.spill), f"{name}_maps", outdir)
        w, v, ty = summary["single_w"], summary["single"]["vetoed"], summary["single_types"]
        rows.append((name, design.area, summary["fp"], summary["bkg_w_missed"], summary["bkg_missed"],
                     (w * v)[ty == 0].sum() / w[ty == 0].sum()))
        print(f"  saved {html}")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for name, area, fp, missed, n_missed, bounced in rows:
        axes[0].scatter(area, 100 * fp, s=60); axes[0].annotate(name, (area, 100 * fp), fontsize=8)
        axes[1].scatter(100 * fp, max(missed, 1e-7), s=60); axes[1].annotate(f"{name} ({n_missed})", (100 * fp, max(missed, 1e-7)), fontsize=8)
        axes[2].scatter(area, 100 * bounced, s=60); axes[2].annotate(name, (area, 100 * bounced), fontsize=8)
    axes[0].set(xlabel="detector area [m²]", ylabel="signal lost [%]", title="signal lost to accidental hits")
    axes[1].set(xlabel="signal lost [%]", ylabel="background surviving (weighted)", yscale="log",
                title="background vs signal (survivors in brackets)")
    axes[2].set(xlabel="detector area [m²]", ylabel="bounced muons vetoed [%]", title="bounced SST1 muons")
    fig.tight_layout()
    plotting.save_fig(fig, "comparison", outdir)


if __name__ == "__main__":
    main()
