#!/usr/bin/env python3
"""Interactive page of the statistics probe (bin/probe_statistics.py): every plane in 3D, coloured cell by cell.

Colour (buttons): independent input muons per cell (n_eff of input muons, the statistics that count), rate density
(Hz / cm^2), simulated rows. Below: per plane, the median n_eff and the cells with fewer than 10.

  python3 bin/make_probe_html.py [results/probe/probe_<dataset>.npz]      # -> results/probe/probe_<dataset>.html
"""

import _bootstrap  # noqa: F401

import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from veto import muon_types as mt
from veto import paths, viz3d


def main():
    f = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted((paths.results_dir() / "probe").glob("probe_*.npz"))[-1]
    d = np.load(f)
    pl, cell, summary = json.loads(str(d["planes"])), float(d["cell"]), json.loads(str(d["summary"]))
    quantities = {"independent input muons per cell (log₁₀ n_eff)": ("neff_in", lambda v, i: np.log10(np.where(v > 0, v, np.nan)), (0, 4)),
                  "rate density (log₁₀ Hz/cm²)": ("rate", lambda v, i: np.log10(np.where(v > 0, v / (cell * 100) ** 2, np.nan)), (-2, 6)),
                  "simulated rows per cell (log₁₀)": ("rows", lambda v, i: np.log10(np.where(v > 0, v, np.nan)), (0, 5))}
    fig = go.Figure([t for t in viz3d.geometry_traces() if t.name != "UBT"])
    n_geo = len(fig.data)
    colors = {q: [] for q in quantities}
    for i, p in enumerate(pl):
        nu, nv = d[f"p{i}_rate"].shape
        u = p["u"][0] + (np.arange(nu + 1)) * cell
        v = p["y"][0] + (np.arange(nv + 1)) * cell
        U, V = np.meshgrid(np.minimum(u, p["u"][1]), np.minimum(v, p["y"][1]), indexing="ij")
        if p["kind"] == "fin":
            X = p["sign"] * (U + mt.side_plane_x(p["z0"]))
            Z = np.full_like(U, p["z0"])
        elif p["kind"] == "ubt":
            X, Z = U, np.full_like(U, p["z0"])
        else:
            X = U
            Z = p["z0"] + (V - mt.FIN_Y_MIN) * np.tan(np.radians(p["tilt"]))
        for q, (key, fn, rng) in quantities.items():
            c = fn(d[f"p{i}_{key}"], i)
            colors[q].append(np.pad(c, ((0, 1), (0, 1)), mode="edge"))     # surface colours at the grid nodes
        q0 = next(iter(quantities))
        rng = quantities[q0][2]
        fig.add_trace(go.Surface(x=Z, y=X, z=V, surfacecolor=colors[q0][-1], cmin=rng[0], cmax=rng[1], colorscale="Viridis",
                                 showscale=i == 0, opacity=0.92, name=p["name"], hovertemplate=p["name"] + "<extra></extra>",
                                 colorbar=dict(title=dict(text=q0, side="right"), len=0.7)))
    buttons = []
    for q, (key, fn, rng) in quantities.items():
        buttons.append(dict(label=q.split(" (")[0], method="restyle",
                            args=[{"surfacecolor": colors[q], "cmin": [rng[0]] * len(pl), "cmax": [rng[1]] * len(pl),
                                   "colorbar.title.text": [q] * len(pl)}, list(range(n_geo, n_geo + len(pl)))]))
    fig.update_layout(title=f"Statistics probe: {len(pl)} planes, {cell * 100:.0f} cm cells ({f.stem.replace('probe_', '')})",
                      updatemenus=[dict(buttons=buttons, direction="down", x=0.01, y=0.99, xanchor="left")],
                      scene=dict(xaxis=dict(title="z [m]", range=[28, 84]), yaxis=dict(title="x [m]", range=[-6, 6]),
                                 zaxis=dict(title="y [m]", range=[-3.5, 3.5]), aspectmode="manual",
                                 aspectratio=dict(x=2.6, y=1.0, z=0.7), camera=dict(eye=dict(x=-0.9, y=-1.3, z=0.6))),
                      height=760, margin=dict(l=0, r=0, t=50, b=0))
    # summary per plane
    names = [s["name"] for s in summary]
    bars = make_subplots(rows=1, cols=2, subplot_titles=("median independent input muons per cell (log)",
                                                         "cells with fewer than 10 independent input muons"))
    bars.add_trace(go.Bar(x=names, y=[s["median_neff_in"] for s in summary], marker_color="#2459A8"), 1, 1)
    bars.add_trace(go.Bar(x=names, y=[s["cells_neff_in_below_10"] for s in summary], marker_color="#B0352B"), 1, 2)
    bars.update_yaxes(type="log", row=1, col=1)
    bars.update_layout(height=420, showlegend=False, margin=dict(l=40, r=10, t=40, b=120))
    out = f.with_suffix(".html")
    html = ("<html><head><meta charset='utf-8'><title>Statistics probe</title></head><body style='font-family:sans-serif'>"
            + fig.to_html(full_html=False, include_plotlyjs="cdn") + bars.to_html(full_html=False, include_plotlyjs=False)
            + "<p style='font-size:13px;color:#555'>n_eff of input muons = (Σw)² / Σw² with the weights first summed over the rows of "
              "one input muon (evt_n): the seeds re-simulate the same input sample, so rows of one input muon are not independent. "
              "Fins every 2.5 m on both sides (right to the cavern wall), floor planes tilted 30° towards the spectrometer.</p></body></html>")
    out.write_text(html)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
