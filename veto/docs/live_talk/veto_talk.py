"""Backend of the live veto talk (PictureThis): every widget on the slides is computed here, with lib/veto.

  data          the full run (types, bounces) and the design-optimisation dataset
  design space  UBT + fins built from sliders, geometric coverage, rate maps
  reconstruction  search windows, timing, accidental hits, algorithm A on real events, diagnosis
  results       knobs (efficiency, time window, time resolution), design comparison, granularity

The spill is 1 s (SPILL_S): rates in Hz equal muons per spill.

  ./run.sh        (build the PDF and present; the app starts this script)
Needs veto/results/{reco/inputs.npz, reco/results.json, design_explorer/sample.npz,
visualization/full/samples.npz} (bin/make_reco_inputs.py, bin/run_reco.py, ...).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import math
import sys
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
VETO = HERE.parents[1]
sys.path[:0] = [str(VETO / "lib"), str(VETO / "bin")]
os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore")

from picture_this import LinePlot, Presentation, check_stale  # noqa: E402

from run_reco import TYPE_NAMES, load_inputs, take  # noqa: E402
from veto import design as dsg  # noqa: E402
from veto import geometry as geo  # noqa: E402
from veto import muon_types as mt  # noqa: E402
from veto import paths, reco, viz3d  # noqa: E402

pres = Presentation("veto_talk", workers=3, scan_budget=90, scan_point_limit=15)
go.Figure(go.Scatter(x=[0], y=[0])).to_json()          # warm up plotly's JSON engine before the workers race

INK, ACC, ORA, GRN, RED, MUT, LINE = "#1C2230", "#2459A8", "#D07A2D", "#2A7F3F", "#B0352B", "#5B6475", "#DDE1E8"
PUR, PINK, GREY = "#9400d3", "#e377c2", "#7f7f7f"
FONT = dict(family="Fira Sans, Helvetica, sans-serif", size=7, color=INK)
SST1_COL = {0: PUR, 1: RED, 2: "#ff9896", 3: "#1f77b4"}

# ---------------------------------------------------------------- data, loaded once
TR, INP = load_inputs()
EXPL = dict(np.load(paths.results_dir() / "design_explorer" / "sample.npz"))
FULL = dict(np.load(paths.results_dir() / "visualization" / "full" / "samples.npz"))
RESULTS = json.loads((paths.results_dir() / "reco" / "results.json").read_text())
DESIGNS = sorted(p.stem for p in dsg.DESIGN_DIR.glob("*.json"))
SPILL_S = 1.0                                                    # spill length [s]
CFG = reco.RecoConfig(spill_s=SPILL_S)
VIZ_TYPES = [("bounced back into SST1", PUR), ("SST1, p > 1 GeV & timing", RED), ("SST1, other", "#ff9896"),
             ("enters through SBT wall", "#1f77b4"), ("enters through UBT window", "#ff7f0e"),
             ("bounced / scattered in rock", PINK), ("straight past the vessel", GREY),
             ("stops in rock, 30.6–82 m", "#8c564b"), ("stops before 30.6 m", "#2b2b2b")]


def compact(n):
    n = float(n)
    return f"{n / 1e6:.0f} M" if n >= 1e6 else f"{n / 1e3:.0f} k" if n >= 1e4 else f"{n:.0f}"


def text_block(fig, lines, x0, y0=0.97, dy=0.085, size=6.5, bold_first=True):
    """Lines of text as one annotation each (wrapped lines would overlap in the slide renderer)."""
    for k, line in enumerate(lines):
        fig.add_annotation(text=f"<b>{line}</b>" if bold_first and k == 0 else line, x=x0, y=y0 - k * dy,
                           xref="paper", yref="paper", xanchor="left", yanchor="top", showarrow=False,
                           font=dict(size=size, color=INK), align="left")
    return fig


def layout(fig, **kw):
    lay = dict(font=FONT, paper_bgcolor="white", plot_bgcolor="white", margin=dict(l=40, r=10, t=22, b=28),
               showlegend=False)
    lay.update(kw)
    fig.update_layout(**lay)
    return fig


def scene(x_range=(-8, 8), y_range=(-6, 8), eye=(-0.75, -1.05, 0.5), z_range=(-3, 98)):
    """3D scene in the example talk's style: white background, pale grid, compact aspect ratio
    (a stretched box draws everything small, and the grey template background washes it out)."""
    def axis(title, rng):
        return dict(title=title, range=list(rng), backgroundcolor="white", gridcolor="#e6e8ee", zeroline=False,
                    showspikes=False, tickfont=dict(size=6))
    return dict(xaxis=axis("z [m]", z_range), yaxis=axis("x [m]", x_range), zaxis=axis("y [m]", y_range),
                aspectmode="manual", aspectratio=dict(x=2.6, y=1.0, z=0.8),
                camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2])))


def geometry(ubt=False):
    return [t for t in viz3d.geometry_traces() if ubt or t.name != "UBT"]


def lines3d(paths_, color, width=2, dash=None, name="", opacity=0.85):
    zs, xs, ys = [], [], []
    for pts in paths_:
        for x, y, z in pts:
            zs.append(float(z)); xs.append(float(x)); ys.append(float(y))
        zs.append(None); xs.append(None); ys.append(None)
    return go.Scatter3d(x=zs, y=xs, z=ys, mode="lines", line=dict(color=color, width=width, dash=dash),
                        opacity=opacity, name=name, hoverinfo="skip")


def pts3d(pts, color, size=4, symbol="circle", name="", line=None):
    p = np.asarray(pts, float).reshape(-1, 3)
    return go.Scatter3d(x=p[:, 2], y=p[:, 0], z=p[:, 1], mode="markers", name=name, hoverinfo="skip",
                        marker=dict(size=size, color=color, symbol=symbol, line=line or dict(width=0)))


def path_of(d, i, fmt):
    """Points A (-> K) -> B of row i; fmt is '{}0'/'{}1' (full-run samples) or 'x_30' style (reco tracks)."""
    a = [d[fmt[0].format(k)][i] for k in "xyz"]
    b = [d[fmt[1].format(k)][i] for k in "xyz"]
    k = [d[fmt[2].format(k)][i] for k in "xyz"] if fmt[2] else [np.nan] * 3
    return [a, k, b] if np.isfinite(k[2]) else [a, b]


# ====================================================================== PART 1: the detector and its muons
pres.slider("n_show", label="muons", min=1, max=25, value=6, step=1, decimals=0)


@pres.output("det3d", inputs=["n_show"])
def det3d(n_show):
    n = int(n_show)
    traces = geometry(ubt=True)
    for t, (name, col) in enumerate(VIZ_TYPES):
        rows = {k[len(f"{t}_"):]: FULL[k] for k in FULL if k.startswith(f"{t}_")}
        if not len(rows.get("x0", [])):
            continue
        m = min(n, len(rows["x0"]))
        if t == 8:
            traces.append(pts3d([(rows["x0"][i], rows["y0"][i], rows["z0"][i]) for i in range(m)], col, 2, name=name))
            continue
        a = np.stack([rows[f"{k}0"][:m] for k in "xyz"], 1); b = np.stack([rows[f"{k}1"][:m] for k in "xyz"], 1)
        pa = np.stack([rows[f"p{k}0"][:m] for k in "xyz"], 1); pb = np.stack([rows[f"p{k}1"][:m] for k in "xyz"], 1)
        kk, ok = mt.bounce_points(a, pa, b, pb)
        bounced = "bounced" in name
        traces.append(lines3d([[a[i], kk[i], b[i]] if bounced and ok[i] else [a[i], b[i]] for i in range(m)
                               if not bounced or ok[i]], col, 2.5 if t <= 4 else 1.5, name=name))
    return layout(go.Figure(traces), scene=scene(), margin=dict(l=0, r=0, t=0, b=0))


# ====================================================================== PART 2: the data
@pres.output("where_bar")
def where_bar():
    count, weight = FULL["count"], FULL["weight"]
    order = list(range(len(VIZ_TYPES)))[::-1]
    names = [VIZ_TYPES[i][0] for i in order]
    fig = go.Figure(go.Bar(y=names, x=[count[i] for i in order], orientation="h", marker_color=[VIZ_TYPES[i][1] for i in order],
                           text=[compact(count[i]) for i in order], textposition="outside",
                           textfont=dict(size=6.5), cliponaxis=False))
    fig.update_xaxes(type="log", title="muons in the full run (506 M simulated)", range=[2, 9.4],
                     tickvals=[1e2, 1e4, 1e6, 1e8], ticktext=["10²", "10⁴", "10⁶", "10⁸"])
    return layout(fig, margin=dict(l=112, r=6, t=8, b=30))


def si_hz(v):
    for unit, f in (("GHz", 1e9), ("MHz", 1e6), ("kHz", 1e3)):
        if v >= f:
            return f"{v / f:.3g} {unit}"
    return f"{v:.3g} Hz"


@pres.output("where_hz")
def where_hz():
    """The same categories as rates: muons per spill / spill length."""
    rate = FULL["weight"] / SPILL_S
    order = list(range(len(VIZ_TYPES)))[::-1]
    fig = go.Figure(go.Bar(y=[VIZ_TYPES[i][0] for i in order], x=[rate[i] for i in order], orientation="h",
                           marker_color=[VIZ_TYPES[i][1] for i in order], text=[si_hz(rate[i]) for i in order],
                           textposition="outside", textfont=dict(size=6.5), cliponaxis=False))
    fig.update_xaxes(type="log", title=f"rate for a {SPILL_S:g} s spill [Hz]", range=[2.7, 12.3],
                     tickvals=[1e3, 1e6, 1e9], ticktext=["1 kHz", "1 MHz", "1 GHz"])
    return layout(fig, title=dict(text=f"total {si_hz(rate.sum())} · {si_hz(rate[:8].sum())} past z = 30.6 m",
                                  font=dict(size=8), x=0.02), margin=dict(l=112, r=6, t=18, b=30))


BOUNCED = np.where((EXPL["s_type"] == 0) & np.isfinite(EXPL["s_kz"]))[0]
pres.slider("bounce_ev", label="muon", min=0, max=39, value=3, step=1, decimals=0)


@pres.output("bounce_view", inputs=["bounce_ev"])
def bounce_view(k):
    i = BOUNCED[int(k) % len(BOUNCED)]
    s = EXPL
    a = np.array([s["s_x_30"][i], s["s_y_30"][i], s["s_z_30"][i]]); b = np.array([s["s_x_82"][i], s["s_y_82"][i], s["s_z_82"][i]])
    kk = np.array([s["s_kx"][i], s["s_ky"][i], s["s_kz"][i]])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.62, 0.38])
    zz = np.linspace(geo.Z_VESSEL_MIN, geo.Z_VESSEL_MAX, 2)
    for r, j, half, wall in ((1, 0, [geo.VESSEL_X_IN, geo.VESSEL_X_OUT], geo.CAVERN_X_WALL),
                             (2, 1, [geo.VESSEL_Y_IN, geo.VESSEL_Y_OUT], geo.CAVERN_Y_FLOOR)):
        for sgn in (1, -1):
            fig.add_trace(go.Scatter(x=zz, y=[sgn * h for h in half], mode="lines", line=dict(color=ACC, width=1.5)), r, 1)
        fig.add_trace(go.Scatter(x=[28, 84], y=[wall, wall], mode="lines", line=dict(color="#8c564b", width=1, dash="dot")), r, 1)
        # forward from A (with B's z range), backward from B: built from the two points and K
        fwd = a + (kk - a) * 1.6
        bwd = b + (kk - b) * 1.6
        fig.add_trace(go.Scatter(x=[a[2], fwd[2]], y=[a[j], fwd[j]], mode="lines", line=dict(color=GREY, width=1, dash="dash")), r, 1)
        fig.add_trace(go.Scatter(x=[b[2], bwd[2]], y=[b[j], bwd[j]], mode="lines", line=dict(color=ORA, width=1, dash="dot")), r, 1)
        fig.add_trace(go.Scatter(x=[a[2], kk[2], b[2]], y=[a[j], kk[j], b[j]], mode="lines+markers",
                                 line=dict(color=PUR, width=2.5), marker=dict(size=[6, 9, 6], color=[INK, PUR, INK],
                                 symbol=["circle", "x", "circle"])), r, 1)
    fig.update_yaxes(title_text="x [m]", row=1, col=1, range=[-7, 3.5])
    fig.update_yaxes(title_text="y [m]", row=2, col=1, range=[-4.5, 4.5])
    fig.update_xaxes(title_text="z [m]", row=2, col=1, range=[28, 84])
    title = f"bounce at z = {kk[2]:.1f} m, x = {kk[0]:.2f} m, y = {kk[1]:.2f} m"
    return layout(fig, title=dict(text=title, font=dict(size=8), x=0.02), margin=dict(l=36, r=8, t=18, b=26))


@pres.output("bounce_map")
def bounce_map():
    k = BOUNCED
    fig = go.Figure(go.Scatter(x=EXPL["s_kx"][k], y=EXPL["s_ky"][k], mode="markers",
                               marker=dict(size=2.5, color=PUR, opacity=0.35)))
    fig.add_shape(type="line", x0=geo.CAVERN_X_WALL, x1=geo.CAVERN_X_WALL, y0=-5, y1=5, line=dict(color="#8c564b", dash="dot"))
    fig.add_shape(type="line", x0=-8, x1=6, y0=geo.CAVERN_Y_FLOOR, y1=geo.CAVERN_Y_FLOOR, line=dict(color="#8c564b", dash="dot"))
    fig.add_shape(type="rect", x0=-geo.VESSEL_X_OUT, x1=geo.VESSEL_X_OUT, y0=-geo.VESSEL_Y_OUT, y1=geo.VESSEL_Y_OUT,
                  line=dict(color=ACC, width=1))
    fig.update_xaxes(title="x of the bounce point [m]", range=[-8, 6])
    fig.update_yaxes(title="y [m]", range=[-5, 5], scaleanchor="x")
    return layout(fig, title=dict(text=f"{len(k):,} reconstructed bounce points", font=dict(size=8), x=0.02),
                  margin=dict(l=34, r=6, t=18, b=28))


# ====================================================================== PART 3: the design space
pres.slider("d_ubt", label="UBT", min=1.35, max=6, value=1.35, decimals=2)
pres.slider("d_nfin", label="fins", min=0, max=8, value=4, step=1, decimals=0)
pres.slider("d_gap", label="gap", min=2, max=17, value=15, decimals=1)
pres.slider("d_len", label="len", min=0.5, max=7.5, value=3, decimals=2)
pres.slider("d_top", label="top", min=0.5, max=8, value=3, decimals=2)

S_A = np.stack([EXPL["s_x_30"], EXPL["s_y_30"], EXPL["s_z_30"]], 1)
S_K = np.stack([EXPL["s_kx"], EXPL["s_ky"], EXPL["s_kz"]], 1)
S_B = np.stack([EXPL["s_x_82"], EXPL["s_y_82"], EXPL["s_z_82"]], 1)


def slider_design(ubt_top, nfin, gap, length, top, cell=0.1):
    fins = []
    for k in range(int(nfin)):
        z = geo.Z_UBT + k * gap
        if z > geo.Z_SST1:
            break
        for side in ("right", "left"):
            fins.append(dict(side=side, z=z, length=length, y_lo=-3.2, y_hi=top, tilt=0, cell_u=cell, cell_y=cell))
    raw = dict(name="slider design", ubt=dict(y_lo=-1.35, y_hi=ubt_top, cell_x=cell, cell_y=cell), fins=fins,
               detector=dict(efficiency=0.99, sigma_t=0.5e-9))
    return dsg.from_dict(raw)


def geometric_hits(design):
    """Which sampled muons cross any detector of the design (no efficiency, no reconstruction)."""
    hit = np.zeros(len(S_A), bool)
    for plane in design.planes:
        if plane.kind == "ubt":
            on = (np.abs(S_A[:, 0]) < geo.UBT_HALF_X) & (EXPL["s_pz_30"] > 0) & (S_A[:, 1] >= plane.y_lo) & (S_A[:, 1] <= plane.y_hi)
        else:
            x, y, ok = mt.path_at_z(S_A, S_K, S_B, plane.z)
            u = plane.to_local(x)
            on = ok & plane.inside(u, y)
        hit |= on
    return hit


def plane_meshes(design, scale=True):
    rates = reco.plane_rates(design, INP)
    out = []
    for k, (plane, r) in enumerate(zip(design.planes, rates)):
        x = [plane.to_global(v) for v in (0, plane.u_max, plane.u_max, 0)]
        y = [plane.y_lo, plane.y_lo, plane.y_hi, plane.y_hi]
        dens = np.log10(max(r["rate"] / max(plane.area, 1e-9), 1))
        out.append(go.Mesh3d(x=[plane.z] * 4, y=x, z=y, i=[0, 0], j=[1, 2], k=[2, 3], intensity=[dens] * 4, cmin=2, cmax=9,
                             colorscale="Viridis", showscale=scale and k == 0, opacity=0.6, flatshading=True, hoverinfo="skip",
                             colorbar=dict(title=dict(text="log₁₀ muons/m²/spill", side="top", font=dict(size=6)),
                                           orientation="h", x=0.5, y=0.0, len=0.5, thickness=7, tickfont=dict(size=6))))
    return out


@pres.output("design3d", inputs=["d_ubt", "d_nfin", "d_gap", "d_len", "d_top"])
def design3d(ubt_top, nfin, gap, length, top):
    d = slider_design(ubt_top, nfin, gap, length, top)
    hit = geometric_hits(d)
    traces = geometry() + plane_meshes(d, scale=False)      # a colour bar makes plotly redraw the WebGL scene
    rng = np.random.default_rng(4)
    for t in (1, 2, 0, 3):
        idx = np.where(EXPL["s_type"] == t)[0]
        pick = rng.choice(idx, 6, replace=False)
        paths_ = [[S_A[i], S_K[i], S_B[i]] if np.isfinite(S_K[i, 2]) else [S_A[i], S_B[i]] for i in pick]
        traces.append(lines3d([p for p, i in zip(paths_, pick) if hit[i]], SST1_COL[t], 2))
        traces.append(lines3d([p for p, i in zip(paths_, pick) if not hit[i]], "black", 5))
    return layout(go.Figure(traces), scene=scene(z_range=(20, 90)), margin=dict(l=0, r=0, t=0, b=0))


@pres.output("design_eff", inputs=["d_ubt", "d_nfin", "d_gap", "d_len", "d_top"])
def design_eff(ubt_top, nfin, gap, length, top):
    d = slider_design(ubt_top, nfin, gap, length, top)
    hit = geometric_hits(d)
    w, ty = EXPL["s_w"], EXPL["s_type"]
    labels = {1: "SST1 p>1, timing", 2: "SST1 other", 0: "SST1 bounced", 3: "veto only"}
    vals = [100 * w[(ty == t) & hit].sum() / w[ty == t].sum() for t in (3, 0, 2, 1)]
    fig = go.Figure(go.Bar(y=[f"{labels[t]}  {v:.1f} %" for t, v in zip((3, 0, 2, 1), vals)], x=vals, orientation="h",
                           marker_color=[SST1_COL[t] for t in (3, 0, 2, 1)]))
    rates = reco.plane_rates(d, INP)
    fig.update_xaxes(range=[0, 100], tickvals=[0, 50, 100], title="crossing a detector [%]")
    title = f"{d.area:.0f} m² · {len(d.planes) - 1} fins · {sum(r['rate'] for r in rates):.2g} hits/spill"
    return layout(fig, title=dict(text=title, font=dict(size=8), x=0.02), margin=dict(l=100, r=8, t=18, b=28))


pres.slider("map_z", label="z [m]", min=30.6, max=82, value=30.6, decimals=1)
pres.slider("map_side", label="side", min=0, max=1, value=0, step=1, decimals=0)


@pres.output("rate_map", inputs=["map_z", "map_side"])
def rate_map(z, side):
    iz = int(np.clip(round(z - INP["z_slices"][0]), 0, len(INP["z_slices"]) - 1))
    s = 1 if int(side) == 0 else 0                       # maps: 0 = +x (left), 1 = -x (right)
    ub, yb = INP["u_bins"], INP["y_bins"]
    dens = INP["fin_maps"][s, iz] / np.outer(np.diff(ub), np.diff(yb))
    fig = go.Figure(go.Heatmap(x=0.5 * (ub[1:] + ub[:-1]), y=0.5 * (yb[1:] + yb[:-1]), z=np.log10(np.maximum(dens.T, 1)),
                               zmin=2, zmax=9, colorscale="Viridis",
                               colorbar=dict(title=dict(text="log₁₀ muons/m²/spill", side="right", font=dict(size=6)),
                                             thickness=7, tickfont=dict(size=6))))
    xw = mt.side_plane_x(INP["z_slices"][iz])
    if int(side) == 0:
        fig.add_shape(type="line", x0=-mt.CAVERN_X_WALL - xw, x1=-mt.CAVERN_X_WALL - xw, y0=-8, y1=8,
                      line=dict(color="white", dash="dot"))
    fig.add_shape(type="line", x0=0, x1=8, y0=mt.FIN_Y_MIN, y1=mt.FIN_Y_MIN, line=dict(color="white", dash="dot"))
    fig.update_xaxes(title="distance from the vessel wall u [m]", range=[0, 8])
    fig.update_yaxes(title="y [m]", range=[-8, 8])
    name = "right (−x, cavern wall dotted)" if int(side) == 0 else "left (+x)"
    return layout(fig, title=dict(text=f"{name}, z = {INP['z_slices'][iz]:.1f} m, wall at |x| = {xw:.2f} m",
                                  font=dict(size=8), x=0.02), margin=dict(l=34, r=6, t=18, b=28))


# ====================================================================== PART 4: reconstruction
# --- spatial window: residuals of the true crossings w.r.t. the backward prediction, computed once (D2)
def _residuals():
    d = dsg.load("D2_default")
    rows = np.random.default_rng(0).choice(INP["single"], 60_000, replace=False)
    sub = take(TR, rows)
    meas = reco.measure(sub, CFG, np.random.default_rng(1))
    out = []
    for plane in d.planes:
        u, y, _, ok = reco.true_crossing(sub, plane)
        dz = meas["z"] - plane.z
        pu = plane.to_local(meas["x"] - meas["tx"] * dz); py = meas["y"] - meas["ty"] * dz
        dist = dz * np.sqrt(1 + meas["tx"] ** 2 + meas["ty"] ** 2)
        uc, yc = plane.cell_centre(u, y)
        m = ok & (sub["type"] != 0)
        out.append(np.stack([np.abs(uc - pu)[m], np.abs(yc - py)[m], dist[m], meas["p"][m]], 1))
    return np.concatenate(out)


RES = _residuals()
pres.slider("r_ms", label="r_ms", min=0, max=20, value=6, decimals=1)


@pres.output("window_plot", inputs=["r_ms"])
def window_plot(r_ms):
    d = np.linspace(0, 55, 60)
    r = CFG.r0 + (CFG.r_slope + 1e-3 * r_ms / RES[:, 3]) * RES[:, 2]
    inside = np.mean((RES[:, 0] <= r + 0.05) & (RES[:, 1] <= r + 0.05))
    plot = LinePlot(xlabel="extrapolation distance [m]", ylabel="search radius r [cm]", yrange=(0, 120),
                    title=f"true crossings of straight SST1 muons inside the window: {100 * inside:.2f} %")
    for p in (1, 3, 10, 50):
        plot.add(d, 100 * (CFG.r0 + (CFG.r_slope + 1e-3 * r_ms / p) * d), label=f"p = {p} GeV")
    return plot


pres.slider("sig_t", label="σt veto", min=0.1, max=2.0, value=0.5, decimals=2)
pres.slider("n_sig", label="window", min=1, max=5, value=3, decimals=1)


def tail(n):
    return math.erfc(n / math.sqrt(2))


@pres.output("time_plot", inputs=["sig_t", "n_sig"])
def time_plot(sig_t, n_sig):
    sc = math.hypot(sig_t, CFG.sigma_t_track)
    x = np.linspace(-6, 6, 241)
    g = np.exp(-0.5 * (x / sc) ** 2) / (sc * math.sqrt(2 * math.pi))
    w = n_sig * sc
    top = 1.1 * g.max()
    plot = LinePlot(xlabel="hit time − predicted time [ns]", ylabel="density", xrange=(-6, 6), yrange=(0, top),
                    title=f"σ = {sc:.2f} ns · window ±{w:.2f} ns · own hits outside: {100 * tail(n_sig):.2f} %")
    plot.add(x, g, label="own hit")
    plot.add([-w, -w], [0, top], label="−window")
    plot.add([w, w], [0, top], label="+window")
    return plot


@pres.output("time_trade", inputs=["sig_t"])
def time_trade(sig_t):
    sc = math.hypot(sig_t, CFG.sigma_t_track)
    n = np.linspace(1.5, 5, 71)
    fp_d2 = next(r["signal"]["FP_frac"] for r in RESULTS["results"] if r["design"] == "D2_default")
    ref = 3 * math.hypot(0.5, CFG.sigma_t_track)
    plot = LinePlot(xlabel="window [σ]", ylabel="%", xrange=(1.5, 5), yrange=(0, 15), title="misses against accidental vetoes")
    plot.add(n, [100 * tail(v) for v in n], label="own hits outside")
    plot.add(n, 100 * fp_d2 * (n * sc) / ref, label="signal lost, D2 (≈)")
    return plot


# --- accidental hits on one fin
pres.slider("cell_cm", label="cell", min=2, max=50, value=10, decimals=0)


@pres.output("acc_cloud", inputs=["cell_cm"])
def acc_cloud(cell_cm):
    spill = SPILL_S
    d = dsg.load("D2_default")
    plane = next(p for p in d.planes if p.kind == "fin" and p.sign < 0)          # right fin at z = 30.6
    cell = cell_cm / 100
    ub, yb, h = reco.plane_rate_map(plane, INP)
    window = 2 * CFG.n_sigma_t * math.hypot(0.5, CFG.sigma_t_track) * 1e-9
    lam = h.sum() * window / spill
    rng = np.random.default_rng(int(cell_cm * 7))
    n = rng.poisson(lam)
    b = rng.choice(h.size, n, p=(h / h.sum()).ravel())
    iu, iy = np.unravel_index(b, h.shape)
    u = ub[iu] + rng.random(n) * np.diff(ub)[iu]; y = yb[iy] + rng.random(n) * np.diff(yb)[iy]
    uc = (np.floor(u / cell) + 0.5) * cell; yc = plane.y_lo + (np.floor((y - plane.y_lo) / cell) + 0.5) * cell
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=0.5 * (ub[1:] + ub[:-1]), y=0.5 * (yb[1:] + yb[:-1]),
                             z=np.log10(np.maximum((h / np.outer(np.diff(ub), np.diff(yb))).T, 1)), zmin=2, zmax=9,
                             colorscale="Greys", showscale=False, opacity=0.35))
    fig.add_trace(go.Scatter(x=uc, y=yc, mode="markers", marker=dict(size=max(2, min(12, cell_cm * 0.5)), symbol="square",
                                                                      color=ORA, line=dict(color=INK, width=0.5))))
    # a search box for a track predicted near the wall
    r = CFG.r0 + (CFG.r_slope + CFG.r_ms / 5) * 45 + cell / 2
    pu, py = 0.4, 0.0
    fig.add_shape(type="rect", x0=pu - r, x1=pu + r, y0=py - r, y1=py + r, line=dict(color=ACC, width=1.5))
    ib = np.clip(np.searchsorted(ub, pu) - 1, 0, len(ub) - 2); jb = np.clip(np.searchsorted(yb, py) - 1, 0, len(yb) - 2)
    rho = h[ib, jb] / (np.diff(ub)[ib] * np.diff(yb)[jb])
    p_acc = 1 - math.exp(-rho * (2 * r) ** 2 * window / spill)
    fig.update_xaxes(title="u [m]", range=[0, plane.u_max]); fig.update_yaxes(title="y [m]", range=[plane.y_lo, plane.y_hi])
    title = f"{lam:.1f} random hits per ±1.5 ns window · P(one in the blue box) = {100 * p_acc:.2f} %"
    return layout(fig, title=dict(text=title, font=dict(size=7.5), x=0.02), margin=dict(l=32, r=6, t=18, b=28))


# --- algorithm A on real events: one cached run per design
_CACHE, _LOCK = {}, threading.Lock()


def run_design(name, quick=False):
    """Background, signal and single-muon events of a design with full details (cached)."""
    key = (name, quick)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
    d = dsg.load(name)
    rng = np.random.default_rng(11)
    cand = INP["cand"] if not quick else INP["cand"][:30_000]
    rows = cand.ravel()
    sub = take(TR, rows)
    vz = np.repeat(INP["cand_vertex"][:len(cand), 2], 2)
    check_stale()
    b = reco.run_events(sub, np.repeat(np.arange(len(cand)), 2), vz, d, INP, CFG, rng, details=True)
    check_stale()
    sig, zv, ev, _ = reco.signal_from_target(30_000 if quick else 60_000, rng)
    s = reco.run_events(sig, ev, zv, d, INP, CFG, rng, with_own_hits=False, details=True)
    check_stale()
    single = INP["single"] if not quick else INP["single"][::5]
    st = take(TR, single)
    m = reco.run_events(st, np.arange(len(single)), np.full(len(single), 1e9), d, INP, CFG, rng, details=True)
    out = dict(design=d, sub=sub, vz=vz, b=b, sig=sig, s=s, st=st, m=m, single_w=INP["single_w"][:len(single)] if not quick
               else INP["single_w"][::5], cand_w=INP["cand_w"][:len(cand)])
    with _LOCK:
        _CACHE[key] = out
    return out


pres.slider("ev_design", label="design", min=0, max=len(DESIGNS) - 1, value=DESIGNS.index("D2_default"), step=1, decimals=0)
pres.slider("ev_kind", label="kind", min=0, max=3, value=0, step=1, decimals=0)
pres.slider("ev_index", label="event", min=0, max=9, value=0, step=1, decimals=0)
KINDS = ["background pair, vetoed", "background pair, missed", "signal lost to a random hit", "bounced muon, missed"]


def pick_event(r, kind, k):
    """Return (event tracks, tracks dict, run result, vertex z per track) for the k-th event of a kind."""
    if kind in (0, 1):
        ev = np.where(r["b"]["vetoed"] if kind == 0 else ~r["b"]["vetoed"])[0]
        if not len(ev):
            return None
        e = ev[(k * 7919) % len(ev)] if kind == 0 else ev[k % len(ev)]
        return [2 * e, 2 * e + 1], r["sub"], r["b"], r["vz"], e
    if kind == 2:
        ev = np.where(r["s"]["vetoed"])[0]
        if not len(ev):
            return None
        e = ev[k % len(ev)]
        n = len(r["s"]["vetoed"])
        return [e, e + n], r["sig"], r["s"], r["sig"]["vz"], e
    ev = np.where(~r["m"]["vetoed"] & (r["st"]["type"] == 0))[0]
    if not len(ev):
        return None
    e = ev[k % len(ev)]
    return [e], r["st"], r["m"], np.full(len(r["st"]["bx"]), 1e9), e


def event_parts(r, kind, k):
    got = pick_event(r, kind, int(k))
    if got is None:
        return None
    tracks, tr, res, vz, e = got
    d = r["design"]
    hits, q, meas = res["hits"], res["queries"], res["meas"]
    in_ev = hits["event"] == e
    parts = dict(tracks=tracks, e=e, design=d, paths=[], preds=[], windows=[], matched=[], own=[], acc=[], vtx=None, info=[])
    for t in tracks:
        if "ax" in tr:
            parts["paths"].append(path_of(tr, t, ("a{}", "b{}", "k{}")))
        else:
            parts["paths"].append([[tr["vx"][t], tr["vy"][t], tr["vz"][t]], [tr["bx"][t], tr["by"][t], tr["bz"][t]]])
            parts["vtx"] = [tr["vx"][t], tr["vy"][t], tr["vz"][t]]
        for j in np.where(q["track"] == t)[0]:
            plane = d.planes[int(q["plane"][j])]
            dz = meas["z"][t] - plane.z
            x0, y0 = meas["x"][t], meas["y"][t]
            xp, yp = plane.to_global(q["u"][j]), q["y"][j]
            parts["preds"].append([[x0, y0, meas["z"][t]], [xp, yp, plane.z]])
            parts["windows"].append((plane, xp, yp, q["r"][j]))
            if res["found"][j]:
                h = res["hit_of_query"][j]
                hp = d.planes[int(hits["plane"][h])]
                parts["matched"].append([hp.to_global(hits["u"][h]), hits["y"][h], hp.z])
                parts["info"].append(f"track {tracks.index(t) + 1} hit: {hp.name.replace(' fin z=', ' ')} m")
                parts["info"].append(f"   Δt = {hits['t'][h] - q['t'][j]:+.2f} ns")
        p = math.sqrt(tr["bpx"][t] ** 2 + tr["bpy"][t] ** 2 + tr["bpz"][t] ** 2)
        typ = TYPE_NAMES.get(int(tr["type"][t]), "") if "type" in tr else "signal"
        short = {"SST1 bounced": "bounced", "SST1 p>1 GeV & timing": "selected", "SST1 other": "other"}.get(typ, typ)
        parts["info"].insert(0, f"track {tracks.index(t) + 1}: {short} muon, {p:.1f} GeV")
    for h in np.where(in_ev)[0]:
        hp = d.planes[int(hits["plane"][h])]
        pt = [hp.to_global(hits["u"][h]), hits["y"][h], hp.z]
        (parts["own"] if hits["track"][h] in tracks else parts["acc"]).append(pt)
    if kind in (0, 1):
        parts["vtx"] = list(INP["cand_vertex"][e])
    return parts


def design_planes_3d(d):
    return plane_meshes(d, scale=False)


@pres.output("ev3d", inputs=["ev_design", "ev_kind", "ev_index"], background=True)
def ev3d(di, kind, k):
    r = run_design(DESIGNS[int(di)])
    parts = event_parts(r, int(kind), k)
    traces = geometry() + design_planes_3d(r["design"])
    sc = scene(z_range=(20, 90))
    if parts is None:
        return layout(go.Figure(traces), scene=sc, title=dict(text="no such event in this design", font=dict(size=8)))
    col = {0: GRN, 1: RED, 2: ORA, 3: PUR}[int(kind)]
    traces += [lines3d(parts["paths"], col, 4), lines3d(parts["preds"], col, 2, dash="dot")]
    if parts["acc"]:
        traces.append(pts3d(parts["acc"], "#666", 2.5))
    if parts["own"]:
        traces.append(pts3d(parts["own"], "white", 5, line=dict(color=col, width=2)))
    if parts["matched"]:
        traces.append(pts3d(parts["matched"], "black" if int(kind) == 2 else col, 7, "diamond", line=dict(color="black", width=1)))
    if parts["vtx"] is not None:
        traces.append(pts3d([parts["vtx"]], col, 4, "cross"))
    return layout(go.Figure(traces), scene=sc, margin=dict(l=0, r=0, t=0, b=0))


@pres.output("ev_side", inputs=["ev_design", "ev_kind", "ev_index"], background=True)
def ev_side(di, kind, k):
    r = run_design(DESIGNS[int(di)])
    parts = event_parts(r, int(kind), k)
    d = r["design"]
    fig = make_subplots(rows=1, cols=1)
    fig.update_xaxes(domain=[0, 0.7])
    zz = np.array([geo.Z_VESSEL_MIN, geo.Z_VESSEL_MAX])
    for sgn in (1, -1):
        fig.add_trace(go.Scatter(x=zz, y=sgn * np.array([geo.VESSEL_X_IN, geo.VESSEL_X_OUT]), mode="lines",
                                 line=dict(color=ACC, width=1.2)), 1, 1)
    fig.add_trace(go.Scatter(x=[28, 84], y=[geo.CAVERN_X_WALL] * 2, mode="lines", line=dict(color="#8c564b", dash="dot", width=1)), 1, 1)
    for plane in d.planes:
        xs = [plane.to_global(0), plane.to_global(plane.u_max)]
        fig.add_trace(go.Scatter(x=[plane.z, plane.z], y=xs, mode="lines", line=dict(color=ORA if plane.kind == "ubt" else "#2ca02c",
                                                                                   width=4), opacity=0.6), 1, 1)
    info = ["no such event in this design"]
    if parts is not None:
        col = {0: GRN, 1: RED, 2: ORA, 3: PUR}[int(kind)]
        for pth in parts["paths"]:
            fig.add_trace(go.Scatter(x=[p[2] for p in pth], y=[p[0] for p in pth], mode="lines", line=dict(color=col, width=2.5)), 1, 1)
        for pr in parts["preds"]:
            fig.add_trace(go.Scatter(x=[pr[0][2], pr[1][2]], y=[pr[0][0], pr[1][0]], mode="lines",
                                     line=dict(color=col, width=1, dash="dot")), 1, 1)
        for plane, xp, yp, rr in parts["windows"]:
            half = rr + plane.cell_u / 2
            fig.add_trace(go.Scatter(x=[plane.z, plane.z], y=[xp - half, xp + half], mode="lines",
                                     line=dict(color=ACC, width=7), opacity=0.35), 1, 1)
        for name, pts, mk in (("acc", parts["acc"], dict(size=4, color="#777")),
                              ("own", parts["own"], dict(size=7, color="white", line=dict(color=col, width=2))),
                              ("matched", parts["matched"], dict(size=9, color="black" if int(kind) == 2 else col, symbol="diamond",
                                                                 line=dict(color="black", width=1)))):
            if pts:
                fig.add_trace(go.Scatter(x=[p[2] for p in pts], y=[p[0] for p in pts], mode="markers", marker=mk), 1, 1)
        if parts["vtx"] is not None:
            fig.add_trace(go.Scatter(x=[parts["vtx"][2]], y=[parts["vtx"][0]], mode="markers",
                                     marker=dict(size=9, color=col, symbol="x")), 1, 1)
        info = [KINDS[int(kind)], f"design {d.name}"] + parts["info"] + \
               [f"{len(parts['own'])} own hits, {len(parts['acc'])} random hits"]
    text_block(fig, info, 0.72, y0=0.98, dy=0.09, size=6)
    fig.update_xaxes(title="z [m]", range=[28, 84], row=1, col=1)
    fig.update_yaxes(title="x [m]", range=[-7.5, 7.5], row=1, col=1)
    return layout(fig, margin=dict(l=32, r=4, t=6, b=26))


# ====================================================================== PART 5: results
pres.slider("k_design", label="design", min=0, max=len(DESIGNS) - 1, value=DESIGNS.index("D2_default"), step=1, decimals=0)
pres.slider("k_eps", label="ε", min=0.9, max=1.0, value=0.99, decimals=3)
pres.slider("k_nsig", label="nσ", min=1.5, max=5, value=3, decimals=1)
pres.slider("k_sigt", label="σt", min=0.1, max=2, value=0.5, decimals=2)


def evaluate(name, eps, spill, nsig, sigt, n_cand, n_sig, n_single):
    raw = dsg.load(name).raw
    raw = json.loads(json.dumps(raw))
    raw["detector"] = dict(efficiency=eps, sigma_t=sigt * 1e-9)
    d = dsg.from_dict(raw)
    cfg = reco.RecoConfig(spill_s=spill, n_sigma_t=nsig)
    rng = np.random.default_rng(21)
    cand = INP["cand"][:n_cand]
    t0 = time.perf_counter()
    b = reco.run_events(take(TR, cand.ravel()), np.repeat(np.arange(len(cand)), 2),
                        np.repeat(INP["cand_vertex"][:len(cand), 2], 2), d, INP, cfg, rng)
    t_b = (time.perf_counter() - t0) / len(cand)
    check_stale()
    sig, zv, ev, _ = reco.signal_from_target(n_sig, rng)
    s = reco.run_events(sig, ev, zv, d, INP, cfg, rng, with_own_hits=False)
    check_stale()
    step = max(len(INP["single"]) // n_single, 1)                      # rows are sorted by type: stride
    single, sw = INP["single"][::step], INP["single_w"][::step]
    m = reco.run_events(take(TR, single), np.arange(len(single)), np.full(len(single), 1e9), d, INP, cfg, rng)
    ty = TR["type"][single].astype(int)
    eff = {k: float((sw * m["vetoed"])[ty == k].sum() / sw[ty == k].sum()) for k in range(3)}
    w = INP["cand_w"][:len(cand)]
    surv = ~b["vetoed"]
    return dict(d=d, n=len(cand), surv=int(surv.sum()), fn=float(w[surv].sum() / w.sum()), fp=float(s["vetoed"].mean()),
                eff=eff, us=1e6 * t_b, hits=b["n_hits"] / len(cand), spill=spill)


def knobs_figure(res, final):
    fig = make_subplots(rows=1, cols=1)
    fig.update_xaxes(domain=[0, 0.6])
    names = ["background pairs vetoed", "signal kept", "single SST1, p>1 & timing", "single SST1, other", "single SST1, bounced"]
    vals = [100 * (1 - res["fn"]), 100 * (1 - res["fp"]), 100 * res["eff"][1], 100 * res["eff"][2], 100 * res["eff"][0]]
    cols = [GRN, ORA, RED, "#ff9896", PUR]
    fig.add_trace(go.Bar(y=names[::-1], x=vals[::-1], orientation="h", marker_color=cols[::-1],
                         text=[f"{v:.3f} %" if i >= 3 else f"{v:.2f} %" for i, v in enumerate(vals[::-1])],
                         textposition="inside", insidetextanchor="start", textfont=dict(color="white", size=7)), 1, 1)
    fig.update_xaxes(range=[0, 100], row=1, col=1, title="%")
    spills = 6e20 / 5e13
    before = INP["cand_w"].sum() * 2 * 200e-12 / res["spill"] * spills
    rows = [["design", f"{res['d'].name}, {res['d'].area:.0f} m²"],
            ["pairs surviving", f"{res['surv']:,} of {res['n']:,}"],
            ["events / 15 y", f"{before:.3g} → {before * res['fn']:.2g}"],
            ["signal lost", f"{100 * res['fp']:.2f} %"], ["hits / event", f"{res['hits']:.1f}"],
            ["time / event", f"{res['us']:.1f} µs"], ["sample", "all events" if final else "quick, 15 %"]]
    text_block(fig, [f"{a}: {b}" for a, b in rows], 0.64, y0=0.97, dy=0.12, size=7, bold_first=False)
    return layout(fig, margin=dict(l=100, r=4, t=4, b=26))


@pres.output("knobs", inputs=["k_design", "k_eps", "k_nsig", "k_sigt"], scan_result="first")
def knobs(di, eps, nsig, sigt):
    spill = SPILL_S
    name = DESIGNS[int(di)]
    yield knobs_figure(evaluate(name, eps, spill, nsig, sigt, 30_000, 15_000, 30_000), False)
    yield knobs_figure(evaluate(name, eps, spill, nsig, sigt, len(INP["cand"]), 60_000, len(INP["single"])), True)


@pres.output("compare")
def compare():
    res = RESULTS["results"]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12)
    for r in res:
        b, s = r["background"], r["signal"]
        y = max(b["FN_frac"], 3e-7)
        label = r["design"].replace("_", " ") if r["design"].startswith("D5") else r["design"].split("_")[0]
        pos = {"D1": "top center", "D2": "bottom center", "D3": "top right", "D5 fine": "bottom left",
               "D5 coarse": "middle right", "D0": "middle right", "D4": "middle right"}.get(label, "middle right")
        fig.add_trace(go.Scatter(x=[100 * s["FP_frac"]], y=[y], mode="markers+text", text=[label],
                                 textposition=pos, textfont=dict(size=6.5),
                                 marker=dict(size=7, color=ACC, opacity=0.8)), 1, 1)
        worst = max(p["peak_cell_hz"] for p in r["planes"])
        pos2 = {"D1": "bottom center", "D2": "top center", "D3": "bottom right", "D5 fine": "bottom center",
                "D5 coarse": "top center"}.get(label, "middle right")
        fig.add_trace(go.Scatter(x=[r["area_m2"]], y=[worst], mode="markers+text", text=[label],
                                 textposition=pos2, textfont=dict(size=6.5), marker=dict(size=7, color=ORA)), 1, 2)
    fig.update_xaxes(title="signal lost to random hits [%]", range=[-0.3, 7.5], row=1, col=1)
    fig.update_yaxes(title="bkg surviving", type="log", range=[-6.8, 0.4], row=1, col=1,
                     tickvals=[1e-6, 1e-4, 1e-2, 1], ticktext=["10⁻⁶", "10⁻⁴", "10⁻²", "1"])
    fig.update_xaxes(title="detector area [m²]", range=[-20, 720], row=1, col=2)
    fig.update_yaxes(title="busiest cell [Hz]", type="log", range=[1.5, 9.8], row=1, col=2, tickvals=[1e2, 1e4, 1e6, 1e8],
                     ticktext=["10²", "10⁴", "10⁶", "10⁸"])
    return layout(fig, margin=dict(l=40, r=8, t=8, b=30))


pres.slider("mw_design", label="design", min=0, max=len(DESIGNS) - 1, value=DESIGNS.index("D1_wide_ubt"), step=1, decimals=0)
REASONS = ["crosses no detector upstream", "hit lost (inefficiency)", "bounced: prediction off by metres",
           "outside the space window", "outside the time window"]


@pres.output("missed_why", inputs=["mw_design"], background=True)
def missed_why(di):
    r = run_design(DESIGNS[int(di)])
    d, sub, b, vz = r["design"], r["sub"], r["b"], r["vz"]
    hits, q, meas = b["hits"], b["queries"], b["meas"]
    window_t = CFG.n_sigma_t * math.hypot(d.sigma_t * 1e9, CFG.sigma_t_track)
    counts = np.zeros(len(REASONS), int)
    surv = np.where(~b["vetoed"])[0]
    for e in surv:
        check_stale()
        for t in (2 * e, 2 * e + 1):
            one = {k: v[t:t + 1] for k, v in sub.items()}
            crossed = [p for p in d.planes if p.z < vz[t] and reco.true_crossing(one, p)[3][0]]
            mine = np.where(hits["track"] == t)[0]
            if not crossed:
                counts[0] += 1
                continue
            on = [h for h in mine if d.planes[int(hits["plane"][h])] in crossed]
            if not on:
                counts[1] += 1
                continue
            h = on[0]
            plane = d.planes[int(hits["plane"][h])]
            dz = meas["z"][t] - plane.z
            pu = plane.to_local(meas["x"][t] - meas["tx"][t] * dz); py = meas["y"][t] - meas["ty"][t] * dz
            dist = dz * math.sqrt(1 + meas["tx"][t] ** 2 + meas["ty"][t] ** 2)
            rr = CFG.r0 + (CFG.r_slope + CFG.r_ms / max(meas["p"][t], 0.1)) * dist
            far = max(abs(hits["u"][h] - pu), abs(hits["y"][h] - py))
            if far > rr + plane.cell_u / 2:
                counts[2 if (np.isfinite(sub["kz"][t]) and far > 1) else 3] += 1
            else:
                counts[4] += 1
    fig = go.Figure(go.Bar(y=REASONS[::-1], x=counts[::-1], orientation="h", marker_color=[ORA, RED, PUR, ACC, MUT],
                           text=[str(c) for c in counts[::-1]], textposition="outside", cliponaxis=False))
    fig.update_xaxes(title="muons of the surviving pairs")
    title = f"{DESIGNS[int(di)]}: {len(surv)} of {len(b['vetoed']):,} pairs survive"
    return layout(fig, title=dict(text=title, font=dict(size=8), x=0.02), margin=dict(l=150, r=20, t=18, b=28))


pres.slider("cs_design", label="design", min=0, max=len(DESIGNS) - 1, value=DESIGNS.index("D2_default"), step=1, decimals=0)
CELLS = [0.03, 0.05, 0.1, 0.2, 0.35, 0.5]


@pres.output("cell_scan", inputs=["cs_design"], background=True)
def cell_scan(di):
    raw0 = dsg.load(DESIGNS[int(di)]).raw
    fp, peak = [], []
    for c in CELLS:
        check_stale()
        raw = json.loads(json.dumps(raw0))
        raw["ubt"].update(cell_x=c, cell_y=c)
        for f in raw["fins"]:
            f.update(cell_u=c, cell_y=c)
        d = dsg.from_dict(raw)
        rng = np.random.default_rng(5)
        sig, zv, ev, _ = reco.signal_from_target(30_000, rng)
        s = reco.run_events(sig, ev, zv, d, INP, CFG, rng, with_own_hits=False)
        fp.append(100 * s["vetoed"].mean())
        peak.append(max(p["peak_cell"] for p in reco.plane_rates(d, INP)))
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12)
    fig.add_trace(go.Scatter(x=[100 * c for c in CELLS], y=fp, mode="lines+markers", line=dict(color=ORA)), 1, 1)
    fig.add_trace(go.Scatter(x=[100 * c for c in CELLS], y=peak, mode="lines+markers", line=dict(color=RED)), 1, 2)
    fig.update_xaxes(title="cell size [cm]", row=1, col=1); fig.update_xaxes(title="cell size [cm]", row=1, col=2)
    fig.update_yaxes(title="signal lost [%]", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title="busiest cell [Hz]", type="log", row=1, col=2, tickvals=[1e5, 1e6, 1e7, 1e8, 1e9],
                     ticktext=["10⁵", "10⁶", "10⁷", "10⁸", "10⁹"])
    return layout(fig, title=dict(text=DESIGNS[int(di)], font=dict(size=8), x=0.02), margin=dict(l=40, r=8, t=18, b=30))


# ====================================================================== the signal, and the figure of merit
@pres.output("sig_cross")
def sig_cross():
    """Where selected signal tracks, followed back from the spectrometer, cross z = 30.6 m, over the muon density."""
    rng = np.random.default_rng(3)
    sig, zv, ev, acc = reco.signal_from_target(4000, rng)
    meas = reco.measure(sig, CFG, rng)
    dz = meas["z"] - geo.Z_UBT
    xs, ys = meas["x"] - meas["tx"] * dz, meas["y"] - meas["ty"] * dz
    # muon density at z = 30.6 m in x-y: fins' maps outside the walls, the UBT's profile inside
    ub, yb = INP["u_bins"], INP["y_bins"]
    xw = mt.side_plane_x(geo.Z_UBT)
    dens = lambda side: INP["fin_maps"][side, 0] / np.outer(np.diff(ub), np.diff(yb))
    yc = 0.5 * (yb[1:] + yb[:-1])
    x_left = xw + 0.5 * (ub[1:] + ub[:-1])
    fig = go.Figure()
    for xx, dd in ((x_left, dens(0)), (-x_left[::-1], dens(1)[::-1])):
        fig.add_trace(go.Heatmap(x=xx, y=yc, z=np.log10(np.maximum(dd.T, 1)), zmin=2, zmax=9, colorscale="Greys",
                                 showscale=False, opacity=0.55))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(size=2.5, color=ORA, opacity=0.6)))
    fig.add_shape(type="rect", x0=-geo.UBT_HALF_X, x1=geo.UBT_HALF_X, y0=-geo.UBT_HALF_Y, y1=geo.UBT_HALF_Y,
                  line=dict(color=ACC, width=1.5))
    inside = np.mean((np.abs(xs) < geo.UBT_HALF_X) & (np.abs(ys) < geo.UBT_HALF_Y))
    near = np.mean((np.abs(xs) >= geo.UBT_HALF_X) & (np.abs(xs) < xw + 1.0))
    fig.update_xaxes(title="x at z = 30.6 m [m]", range=[-4, 4])
    fig.update_yaxes(title="y [m]", range=[-3.5, 3.5], scaleanchor="x")
    title = f"{100 * inside:.0f} % through the UBT · {100 * near:.0f} % within 1 m outside it"
    return layout(fig, title=dict(text=title, font=dict(size=7.5), x=0.02), margin=dict(l=32, r=6, t=18, b=28))


pres.slider("fom_design", label="design", min=0, max=len(DESIGNS) - 1, value=DESIGNS.index("D2_default"), step=1, decimals=0)


@pres.output("fom", inputs=["fom_design"], background=True)
def fom(di):
    """Confusion matrix of one design: background pairs (expected events in 15 years) and signal (fractions)."""
    r = run_design(DESIGNS[int(di)])
    w15 = r["cand_w"] * 2 * 200e-12 / SPILL_S * (6e20 / 5e13)
    veto_b, veto_s = r["b"]["vetoed"], r["s"]["vetoed"]
    tp, fn = w15[veto_b].sum(), w15[~veto_b].sum()
    fn_err = math.sqrt((w15[~veto_b] ** 2).sum()) if (~veto_b).any() else 2.3 * w15.mean()
    fp, tn = veto_s.mean(), 1 - veto_s.mean()
    fig = go.Figure()
    cells = [(0, 1, "TP", f"{tp:.3g} events", f"{100 * tp / (tp + fn):.4f} % of bkg", GRN),
             (1, 1, "FN", f"{fn:.1e} ± {fn_err:.0e}", f"events · {(~veto_b).sum()} of {len(veto_b):,} pairs", RED),
             (0, 0, "FP", f"{100 * fp:.2f} %", "of the signal", ORA),
             (1, 0, "TN", f"{100 * tn:.2f} %", "of the signal", ACC)]
    for cx, cy, name, big, small, col in cells:
        fig.add_shape(type="rect", x0=cx, x1=cx + 1, y0=cy, y1=cy + 1, line=dict(color="white", width=3),
                      fillcolor=col, opacity=0.18)
        for dy, text, size in ((0.72, f"<b>{name}</b>", 9), (0.47, big, 8.5), (0.24, small, 6.5)):
            fig.add_annotation(x=cx + 0.5, y=cy + dy, text=text, showarrow=False, font=dict(size=size, color=INK))
    for x, text in ((0.5, "vetoed"), (1.5, "not vetoed")):
        fig.add_annotation(x=x, y=2.08, text=f"<b>{text}</b>", showarrow=False, font=dict(size=7, color=MUT))
    for y, text in ((1.5, "muon pairs"), (0.5, "signal")):
        fig.add_annotation(x=-0.06, y=y, text=f"<b>{text}</b>", showarrow=False, textangle=-90, font=dict(size=7, color=MUT))
    fig.update_xaxes(visible=False, range=[-0.15, 2.02]); fig.update_yaxes(visible=False, range=[-0.02, 2.18])
    punzi = tn / (1.5 + math.sqrt(fn))
    return layout(fig, title=dict(text=f"{DESIGNS[int(di)]} · Punzi {punzi:.3f}",
                                  font=dict(size=7.5), x=0.02), margin=dict(l=4, r=4, t=18, b=4))


# ---------------------------------------------------------------- warm the default design while the talk starts
threading.Thread(target=lambda: run_design("D2_default"), daemon=True).start()

if __name__ == "__main__":
    pres.run()
