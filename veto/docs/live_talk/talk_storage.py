"""Live-talk widgets for the statistics of the muon dataset and the storage policy (registered by veto_talk.py).

  st_seeds   independent input muons per type against the number of seeds (delta's first k seeds, epsilon's 6000)
  st_repeat  do more seeds bring new input muons into a cell? distinct (cell, input muon) pairs per band
  st_plane   one probe plane: muon density, and independent input muons per cell with epsilon and with the alpha storage
  st_policy  the storage policies emulated on delta's rows, at the same storage over 6000 seeds
  st_bands   delta, epsilon and the alpha-storage test production, per band of probe cells
Data: results/storage_policy (study json, seeds_scaling.json), results/probe/probe_<dataset>.npz. docs/storage_policy.md.
"""
import json

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from veto import muon_types as mt
from veto import paths, probe

INK, MUT, GRID = "#1C2230", "#5B6475", "#E6E8EC"
FONT = dict(family="Fira Sans, Helvetica, sans-serif", size=7, color=INK)
# categorical, fixed order (validated: scripts/validate_palette.js, light surface); low-contrast slots carry value labels
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
RES = paths.results_dir()
DATASETS = {"delta (484 seeds, 111 M rows)": "design-optimization-dataset-delta-enriched-2026-10-19",
            "epsilon (6000 seeds, 552 M rows)": "design-optimization-dataset-epsilon-enriched-6k-2026-10-21",
            "new storage (800 seeds, 73 M rows)": "design-optimization-dataset-alpha-storage-v0-800seeds-2026-10-22"}
BAND_NAMES = {"fin 0-0.5 m": "fin 0–½ m", "fin 0.5-1 m": "fin ½–1 m", "fin 1-2 m": "fin 1–2 m",
              "fin > 2 m": "fin > 2 m", "floor": "floor", "ubt": "UBT"}
METRICS = {"10th percentile": lambda x: np.percentile(x, 10), "median": np.median,
           "cells below 10 [%]": lambda x: 100 * np.mean(x < 10)}
PLANES_SHOWN = {"R fin 30.6": "right fin z=30.6", "L fin 40.6": "left fin z=40.6", "R fin 55.6": "right fin z=55.6",
                "floor 50": "floor z=50.0", "floor 60": "floor z=60.0", "UBT": "UBT z=30.6"}
PLANE_TITLES = {"right fin z=30.6": "right fin at z = 30.6 m (vessel entrance)", "left fin z=40.6": "left fin at z = 40.6 m",
                "right fin z=55.6": "right fin at z = 55.6 m", "floor z=50.0": "floor plane at z = 50 m (tilted 30°)",
                "floor z=60.0": "floor plane at z = 60 m (tilted 30°)", "UBT z=30.6": "UBT column at z = 30.6 m, full height"}


LEGEND = dict(orientation="h", x=0, y=1.0, yanchor="bottom", font=dict(size=5.5), itemwidth=30, bgcolor="rgba(0,0,0,0)",
              tracegroupgap=0)


def layout(fig, **kw):
    lay = dict(uirevision="keep", font=FONT, paper_bgcolor="white", plot_bgcolor="white",
               margin=dict(l=40, r=8, t=8, b=26), showlegend=False)
    lay.update(kw)
    fig.update_layout(**lay)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor="#C9CDD4")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="#C9CDD4")
    return fig


def note(text):
    fig = go.Figure()
    fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", text=text, showarrow=False, font=dict(size=8, color=MUT))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return layout(fig, margin=dict(l=0, r=0, t=0, b=0))


def _load_probe(ds):
    f = RES / "probe" / f"probe_{ds}.npz"
    if not f.exists():
        return None
    d = np.load(f)
    pl = json.loads(str(d["planes"]))
    lab = np.concatenate(probe.band_labels(pl, float(d["cell"])))
    ne = np.concatenate([d[f"p{i}_neff_in"].ravel() for i in range(len(pl))])
    rate = np.concatenate([d[f"p{i}_rate"].ravel() for i in range(len(pl))])
    maps = {p["name"]: (p, d[f"p{i}_rate"], d[f"p{i}_neff_in"]) for i, p in enumerate(pl)}
    return dict(lab=lab, ne=ne, busy=rate > 0, maps=maps, cell=float(d["cell"]))


def plane_corners(p):
    """The 4 corners (x, y, z) of a probe plane, in order around it."""
    (u0, u1), (y0, y1) = p["u"], p["y"]
    if p["kind"] == "fin":
        X = float(mt.side_plane_x(p["z0"]))
        return [(p["sign"] * (u + X), y, p["z0"]) for u, y in ((u0, y0), (u1, y0), (u1, y1), (u0, y1))]
    if p["kind"] == "ubt":
        return [(u, y, p["z0"]) for u, y in ((u0, y0), (u1, y0), (u1, y1), (u0, y1))]
    t = np.tan(np.radians(p["tilt"]))
    return [(u, y, p["z0"] + (y - mt.FIN_Y_MIN) * t) for u, y in ((u0, y0), (u1, y0), (u1, y1), (u0, y1))]


def plane_mesh(p, color, opacity):
    c = plane_corners(p)
    return go.Mesh3d(x=[q[2] for q in c], y=[q[0] for q in c], z=[q[1] for q in c], i=[0, 0], j=[1, 2], k=[2, 3],
                     color=color, opacity=opacity, flatshading=True, hoverinfo="skip", showscale=False)


def register(pres, geometry=None, scene=None):
    SC = json.loads((RES / "storage_policy" / "seeds_scaling.json").read_text()) if (RES / "storage_policy" / "seeds_scaling.json").exists() else None
    study_f = RES / "storage_policy" / "study_design-optimization-dataset-delta-enriched-2026-10-19_S6000.json"
    STUDY = json.loads(study_f.read_text()) if study_f.exists() else None
    PROBES = {label: _load_probe(ds) for label, ds in DATASETS.items()}

    # ------------------------------------------------------------------ more seeds: what they buy
    @pres.output("st_seeds")
    def st_seeds():
        if not SC:
            return note("results/storage_policy/seeds_scaling.json missing")
        s = np.array(SC["seeds"], float)
        fig = go.Figure()
        for (name, y), col in zip(SC["neff"].items(), CAT):
            slope = np.polyfit(np.log(s[:4]), np.log(np.array(y[:4], float)), 1)[0]
            label = f"{name} (first seeds only)" if name == "straight past" else f"{name}: ∝ seeds<sup>{slope:.1f}</sup>"
            fig.add_trace(go.Scatter(x=s, y=y, mode="lines+markers", line=dict(color=col, width=2), name=label,
                                     marker=dict(size=6, color=col, line=dict(width=1.2, color="white")),
                                     hovertemplate=f"{name}<br>%{{x:.0f}} seeds: %{{y:,.0f}} independent input muons<extra></extra>"))
            shift = {"straight past": 7, "SST1 bounced": 8, "SST1 other": 9, "SST1 selected": 0, "veto only": -9}[name]
            fig.add_annotation(x=np.log10(s[-1]), y=np.log10(y[-1]), text=label, showarrow=False, xanchor="right", yanchor="middle",
                               xshift=-7, yshift=shift, font=dict(size=5.5, color=col), bgcolor="rgba(255,255,255,0.75)")
        fig.update_xaxes(type="log", title_text="seeds", tickvals=[60, 480, 6000], ticktext=["60", "480", "6000"], title_standoff=2)
        fig.update_yaxes(type="log", title_text="independent input muons", dtick=1, title_standoff=2)
        return layout(fig)

    @pres.output("st_repeat")
    def st_repeat():
        if not SC:
            return note("no data")
        R = SC["repetition"]
        s = np.array(R["seeds"], float)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s, y=s / s[0], mode="lines", line=dict(color=MUT, width=1, dash="dot"), name="no repetition (∝ seeds)",
                                 hoverinfo="skip"))
        for (band, label), col in zip((("fin 0.5-1 m", "fins 0.5–1 m"), ("floor", "floor"), ("fin 0-0.5 m", "fins 0–0.5 m")), CAT):
            d = np.array(R[band]["distinct"], float)
            slope = np.polyfit(np.log(s), np.log(d), 1)[0]
            fig.add_trace(go.Scatter(x=s, y=d / d[0], mode="lines+markers", line=dict(color=col, width=2), name=f"{label}: ∝ seeds<sup>{slope:.2f}</sup>",
                                     marker=dict(size=6, color=col, line=dict(width=1.2, color="white")),
                                     hovertemplate=f"{label}<br>%{{x:.0f}} seeds: ×%{{y:.2f}} distinct input muons<extra></extra>"))
            fig.add_annotation(x=np.log10(s[-1]), y=np.log10(d[-1] / d[0]), text=f"{label}: ∝ seeds<sup>{slope:.2f}</sup>", showarrow=False,
                               xanchor="right", yanchor="middle", xshift=-7, font=dict(size=5.5, color=col), bgcolor="rgba(255,255,255,0.8)")
        fig.update_xaxes(type="log", title_text="seeds", tickvals=[60, 120, 240, 480], ticktext=["60", "120", "240", "480"], title_standoff=2)
        fig.update_yaxes(type="log", title_text="new input muons in a cell, ×", tickvals=[1, 2, 4, 8], ticktext=["×1", "×2", "×4", "×8"],
                         title_standoff=2)
        return layout(fig)

    # ------------------------------------------------------------------ where the storage goes: one plane
    pres.radio("st_plane", list(PLANES_SHOWN), value="R fin 30.6", label="plane")
    pres.radio("st_view", ["maps", "where (3D)"], value="maps", label="view", style="segmented")

    def where_3d(name):
        """The detector with every probe plane faint and the chosen one in red."""
        P = PROBES["epsilon (6000 seeds, 552 M rows)"]
        if not P or name not in P["maps"] or geometry is None:
            return note("no geometry")
        traces = list(geometry())
        for other, (p, _, _) in P["maps"].items():
            if other != name:
                traces.append(plane_mesh(p, "#9aa3b2", 0.10))
        p = P["maps"][name][0]
        traces.append(plane_mesh(p, "#e8000b", 0.95))
        c = plane_corners(p) + [plane_corners(p)[0]]
        traces.append(go.Scatter3d(x=[q[2] for q in c], y=[q[0] for q in c], z=[q[1] for q in c], mode="lines",
                                   line=dict(color="#a01010", width=5), hoverinfo="skip", showlegend=False))
        top = max(plane_corners(p), key=lambda q: (q[1], abs(q[0])))
        traces.append(go.Scatter3d(x=[top[2]], y=[top[0]], z=[top[1] + 0.4], mode="text", text=[PLANE_TITLES.get(name, name)],
                                   textfont=dict(size=8, color="#a01010"), hoverinfo="skip", showlegend=False))
        fig = go.Figure(traces)
        fig.add_annotation(xref="paper", yref="paper", x=0.01, y=0.99, xanchor="left", yanchor="top", showarrow=False,
                           text=f"<b>{PLANE_TITLES.get(name, name)}</b> (red); grey: all 48 probe planes", font=dict(size=7, color=INK))
        return layout(fig, scene=scene(z_range=(26, 86), x_range=(-6, 6), y_range=(-3.6, 3.6)), margin=dict(l=0, r=0, t=0, b=0))

    @pres.output("st_plane_maps", inputs=["st_plane", "st_view"])
    def st_plane_maps(choice, view="maps"):
        name = PLANES_SHOWN.get(choice, choice)
        if view != "maps":
            return where_3d(name)
        eps, alp = PROBES["epsilon (6000 seeds, 552 M rows)"], PROBES["new storage (800 seeds, 73 M rows)"]
        if not eps or not alp or name not in eps["maps"]:
            return note("probe files missing")
        p, rate, ne_e = eps["maps"][name]
        _, _, ne_a = alp["maps"][name]
        cell = eps["cell"]
        u = p["u"][0] + (np.arange(rate.shape[0]) + 0.5) * cell
        v = p["y"][0] + (np.arange(rate.shape[1]) + 0.5) * cell
        fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.03,
                            subplot_titles=("muon density", "epsilon: 6000 seeds", "new storage: 800 seeds"))
        dens = rate / cell ** 2
        fig.add_trace(go.Heatmap(x=u, y=v, z=np.log10(np.where(dens > 0, dens, np.nan)).T, colorscale="Greys", zmin=3, zmax=9,
                                 showscale=False,
                                 hovertemplate="u %{x:.2f} m, y %{y:.2f} m<br>10^%{z:.1f} muons/m²/spill<extra></extra>"), 1, 1)
        blues = [[0, "#eef4fc"], [0.33, "#9cc3ee"], [0.66, "#2a78d6"], [1, "#0f3a73"]]
        for j, ne in ((2, ne_e), (3, ne_a)):
            fig.add_trace(go.Heatmap(x=u, y=v, z=np.log10(np.where(rate > 0, np.maximum(ne, 1), np.nan)).T, colorscale=blues, zmin=0, zmax=3,
                                     showscale=j == 3, colorbar=dict(title=dict(text="per cell", side="top", font=dict(size=5.5)),
                                                                     x=1.01, len=1.0, thickness=6, tickfont=dict(size=5.5),
                                                                     tickvals=[0, 1, 2, 3], ticktext=["1", "10", "100", "1k"]),
                                     hovertemplate="u %{x:.2f} m, y %{y:.2f} m<br>10^%{z:.2f} independent muons<extra></extra>"), 1, j)
        xt = "from the wall [m]" if p["kind"] == "fin" else "x [m]"
        for j in (1, 2, 3):
            fig.update_xaxes(title_text=xt, row=1, col=j, title_font=dict(size=6), title_standoff=1, dtick=1)
        fig.update_yaxes(title_text="y [m]", row=1, col=1, title_standoff=1)
        fig.update_annotations(font=dict(size=6.5))
        fig.add_annotation(xref="paper", yref="paper", x=0.0, y=1.0, yshift=14, xanchor="left", yanchor="bottom", showarrow=False,
                           text=f"<b>{PLANE_TITLES.get(name, name)}</b>", font=dict(size=7, color=INK))
        return layout(fig, margin=dict(l=30, r=38, t=26, b=24))

    # ------------------------------------------------------------------ the policies, emulated
    pres.radio("st_metric", list(METRICS)[:2], value="10th percentile", label="cells", style="segmented")

    @pres.output("st_policy", inputs=["st_metric"])
    def st_policy(metric):
        if not STUDY:
            return note("storage study missing")
        key = {"10th percentile": "p10", "median": "median"}[metric]
        names = {"now (measured, n_eff of input muons)": "now", "uniform": "today's rule, 6000 seeds",
                 "weight": "∝ weight", "cell": "by cell", "cell+weight": "by cell, ∝ weight"}
        bands = list(BAND_NAMES)
        fig = go.Figure()
        for (k, label), col in zip(names.items(), CAT):
            g = STUDY["policies"][k]["groups"]
            y = [max(g[b][key], 0.05) for b in bands]
            show = k in ("now (measured, n_eff of input muons)", "cell+weight")
            fig.add_trace(go.Bar(x=[BAND_NAMES[b] for b in bands], y=y, name=label, marker=dict(color=col, line=dict(width=1, color="white")),
                                 texttemplate="%{y:.2s}" if show else "", textposition="outside", textfont=dict(size=5, color=INK), cliponaxis=False,
                                 hovertemplate=f"{label}<br>%{{x}}: %{{y:.3g}}<extra></extra>"))
        fig.update_yaxes(type="log", title_text=f"muons per cell, {metric}", range=[-1.3, 4.6], dtick=1, title_standoff=2)
        fig.update_xaxes(tickangle=0, tickfont=dict(size=5.8))
        return layout(fig, barmode="group", bargap=0.16, bargroupgap=0.04, margin=dict(l=36, r=6, t=8, b=18))

    # ------------------------------------------------------------------ the test production
    pres.radio("st_bands_metric", list(METRICS), value="10th percentile", label="cells", style="segmented")

    @pres.output("st_bands", inputs=["st_bands_metric"])
    def st_bands(metric):
        f = METRICS[metric]
        bands = list(BAND_NAMES)
        fig = go.Figure()
        for (label, P), col in zip(PROBES.items(), CAT):
            if P is None:
                continue
            y = [f(P["ne"][(P["lab"] == b) & P["busy"]]) for b in bands]
            fig.add_trace(go.Bar(x=[BAND_NAMES[b] for b in bands], y=y, name=label, marker=dict(color=col, line=dict(width=1, color="white")),
                                 text=[f"{v:.2g}" if metric != "cells below 10 [%]" else f"{v:.0f}%" for v in y], textposition="outside",
                                 textfont=dict(size=5.2, color=INK), cliponaxis=False, hovertemplate=f"{label}<br>%{{x}}: %{{y:.3g}}<extra></extra>"))
        if metric == "cells below 10 [%]":
            fig.update_yaxes(title_text="cells below 10 [%]", range=[0, 52], title_standoff=2)
        else:
            fig.update_yaxes(type="log", title_text=f"muons per cell, {metric}", range=[0, 2.8], dtick=1, title_standoff=2)
        fig.update_xaxes(tickangle=0, tickfont=dict(size=5.8))
        return layout(fig, barmode="group", bargap=0.2, bargroupgap=0.05, margin=dict(l=36, r=6, t=8, b=18))
