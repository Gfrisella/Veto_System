"""Backend of the live talk on the muons from the target (PictureThis): every widget is computed here from the study's
result files. Independent of the veto talk (docs/live_talk), with the same look; the storage-policy widgets are
docs/live_talk/talk_storage.py.

  top muons     which input muons make the muon-interaction rate in the vessel (results/target_talk/top_muons.npz,
                bin/top_muons_data.py on the farm)
  2018 match    the ladder from "40 % too many" to 1.01 (results/complete_validation: compare_*.npz, rule2018_*.npz,
                inventory_*.npz)
  generator     complete-mode validation (summary.json), Pythia 8.223 vs 8.316 hadrons (results/pythia_versions)
  statistics    the importance-sampling survival probability (sliders)

  ./run.sh        (build the PDF and present; the app starts this script)
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
VETO = HERE.parents[1]
sys.path[:0] = [str(VETO / "lib"), str(VETO / "bin"), str(HERE.parent / "live_talk")]
warnings.filterwarnings("ignore")

from picture_this import Presentation  # noqa: E402

pres = Presentation("target_talk", workers=2)
go.Figure(go.Scatter(x=[0], y=[0])).to_json()          # warm up plotly's JSON engine

INK, MUT, GRID, LINE = "#1C2230", "#5B6475", "#E6E8EC", "#C9CDD4"
FONT = dict(family="Fira Sans, Helvetica, sans-serif", size=7, color=INK)
# categorical, fixed order (as talk_storage.py, validated on the light surface)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
REF = "#1C2230"
RES = VETO / "results"
CV = RES / "complete_validation"


def layout(fig, **kw):
    lay = dict(uirevision="keep", font=FONT, paper_bgcolor="white", plot_bgcolor="white",
               margin=dict(l=40, r=8, t=8, b=26), showlegend=False)
    lay.update(kw)
    fig.update_layout(**lay)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=LINE)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=LINE)
    return fig


def note(text):
    fig = go.Figure()
    fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", text=text, showarrow=False, font=dict(size=8, color=MUT))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return layout(fig, margin=dict(l=0, r=0, t=0, b=0))


def legend(**kw):
    d = dict(orientation="h", x=0, y=1.0, yanchor="bottom", font=dict(size=6), bgcolor="rgba(0,0,0,0)")
    d.update(kw)
    return d


def grey_scale():
    return [[0, "#ffffff"], [0.15, "#e9ebef"], [0.5, "#9aa1ad"], [1, "#1c2230"]]


# ====================================================================== top muons
TM = dict(np.load(RES / "target_talk" / "top_muons.npz")) if (RES / "target_talk" / "top_muons.npz").exists() else None
TM_LABELS = {"gamma": "veto design dataset on the 2018 sample"}
TM_TAGS = ["gamma"] if TM is not None and "gamma_rate" in TM else []
TOPC = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7b5cd6", "#8c564b", "#17becf", "#b0352b", "#5b6475"]


pres.slider("tm_n", label="top", min=1, max=10, value=5, step=1, decimals=0)


@pres.output("tm_cum")
def tm_cum():
    if TM is None:
        return note("results/target_talk/top_muons.npz missing (bin/top_muons_data.py)")
    fig = go.Figure()
    for i, t in enumerate(TM_TAGS):
        rk, cum = TM[f"{t}_rank"], TM[f"{t}_cum"]
        fig.add_trace(go.Scatter(x=rk, y=cum, mode="lines", line=dict(color=CAT[i], width=2), name=TM_LABELS[t],
                                 hovertemplate="%{x:,} input muons: %{y:.2f}<extra></extra>"))
    for k in (10, 100, 1000, 10000):
        fig.add_vline(x=k, line=dict(color=LINE, width=1, dash="dot"))
    fig.update_xaxes(type="log", title="input muons, sorted by their share", exponentformat="power")
    fig.update_yaxes(title="cumulative share of the vessel rate", range=[0, 1.02])
    return layout(fig)


def _top(t):
    return json.loads(str(TM[f"{t}_top"]))


@pres.output("tm_src", inputs=["tm_n"])
def tm_src(n):
    if TM is None:
        return note("no data")
    t, n = "gamma", int(n)
    h = TM[f"{t}_src_map"]
    pb, tb = TM["PB"], TM["TB"]
    z = np.where(h > 0, np.log10(np.maximum(h, 1e-12)), np.nan)
    fig = go.Figure(go.Heatmap(x=0.5 * (pb[1:] + pb[:-1]), y=0.5 * (tb[1:] + tb[:-1]), z=z.T, zmin=-6, zmax=-0.5,
                               colorscale=grey_scale(), colorbar=dict(title=dict(text="share", side="right"), thickness=8,
                                                                      tickvals=[-6, -4, -2, -1], ticktext=["10⁻⁶", "10⁻⁴", "0.01", "0.1"]),
                               hovertemplate="p %{x:.0f} GeV, θ %{y:.1f} mrad<extra></extra>"))
    for r in _top(t)[:n]:
        fig.add_trace(go.Scatter(x=[r["p_start"]], y=[r["theta_mrad"]], mode="markers",
                                 marker=dict(size=6 + 120 * r["share"], color=TOPC[r["rank"] - 1], line=dict(color=INK, width=1)),
                                 hovertemplate=f"#{r['rank']}: {100 * r['share']:.1f} % of the rate, p {r['p_start']:.0f} GeV, "
                                               f"θ {r['theta_mrad']:.1f} mrad, {r['rows']:,} copies<extra></extra>"))
    fig.update_xaxes(title="p at the source [GeV]", range=[0, 400])
    fig.update_yaxes(title="θ at the source [mrad]", range=[0, 40])
    return layout(fig, margin=dict(l=36, r=4, t=6, b=26))


@pres.output("tm_xy", inputs=["tm_n"])
def tm_xy(n):
    if TM is None:
        return note("no data")
    t, n = "gamma", int(n)
    xb = TM["XB"]
    h = TM[f"{t}_xy_map"]
    z = np.where(h > 0, np.log10(np.maximum(h, 1e-12)), np.nan)
    xc = 0.5 * (xb[1:] + xb[:-1])
    fig = go.Figure(go.Heatmap(x=xc, y=xc, z=z.T, zmin=-7, zmax=-1, colorscale=grey_scale(), showscale=False,
                               hovertemplate="x %{x:.2f}, y %{y:.2f} m<extra></extra>"))
    hx, hy = TM["vessel_hw"]
    fig.add_trace(go.Scatter(x=[-hx, hx, hx, -hx, -hx], y=[-hy, -hy, hy, hy, -hy], mode="lines",
                             line=dict(color="#17becf", width=1.5), hoverinfo="skip"))
    ti, tx, ty = TM[f"{t}_top_i"], TM[f"{t}_top_x"], TM[f"{t}_top_y"]
    for r in range(n):
        m = ti == r
        fig.add_trace(go.Scattergl(x=tx[m][:1500], y=ty[m][:1500], mode="markers", marker=dict(size=2, color=TOPC[r], opacity=0.5),
                                   hovertemplate=f"#{r + 1}<extra></extra>"))
    fig.update_xaxes(title="x at z = 30.6 m [m]", range=[-2.2, 2.2])
    fig.update_yaxes(title="y [m]", range=[-2.0, 2.0], scaleanchor="x")
    return layout(fig, margin=dict(l=34, r=4, t=6, b=26))


@pres.output("tm_p30", inputs=["tm_n"])
def tm_p30(n):
    if TM is None:
        return note("no data")
    t, n = "gamma", int(n)
    ti, tp = TM[f"{t}_top_i"], TM[f"{t}_top_p"]
    top = _top(t)
    fig = go.Figure()
    b = np.linspace(0, 350, 71)
    for r in range(n):
        h = np.histogram(tp[ti == r], b)[0]
        h = h / max(h.sum(), 1)
        fig.add_trace(go.Scatter(x=0.5 * (b[1:] + b[:-1]), y=h, mode="lines", line=dict(color=TOPC[r], width=1.5, shape="hvh"),
                                 name=f"#{r + 1}: {top[r]['p_start']:.0f} GeV at the source",
                                 hovertemplate=f"#{r + 1}: %{{x:.0f}} GeV<extra></extra>"))
    fig.update_xaxes(title="p at z = 30.6 m over the seeds [GeV]", range=[0, 350])
    fig.update_yaxes(title="fraction of copies / 5 GeV", rangemode="tozero")
    return layout(fig, showlegend=True, legend=legend(x=0.0, y=1.0, yanchor="top", orientation="v", font=dict(size=5.5)))


@pres.output("tm_psrc")
def tm_psrc():
    if TM is None:
        return note("no data")
    pb = TM["PB"]
    fig = go.Figure()
    for i, t in enumerate(TM_TAGS):
        s = TM[f"{t}_p_src"].sum(0)
        fig.add_trace(go.Scatter(x=0.5 * (pb[1:] + pb[:-1]), y=s, mode="lines", line=dict(color=CAT[i], width=1.6, shape="hvh"),
                                 name=TM_LABELS[t], hovertemplate="%{x:.0f} GeV: %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title="p at the source [GeV]", range=[0, 400])
    fig.update_yaxes(title="share of the vessel rate / 5 GeV", rangemode="tozero")
    return layout(fig, showlegend=True, legend=legend(x=0.0, y=1.0, yanchor="top", orientation="v",
                                                      bgcolor="rgba(255,255,255,0.8)"))


# ====================================================================== the ladder to 2018
H18 = np.load(CV / "compare_ours_cv_2018exact.npz")
R18 = np.load(CV / "compare_refs.npz")
Q18 = np.load(CV / "rule2018_cv_2018exact_c.npz")
PE, TE = R18["PE"], R18["TE"]
PC = np.sqrt(PE[1:] * PE[:-1])
P_MIN = 25.0
REF2 = {k: R18[f"ref_minbias_{k}"] + R18[f"ref_dimuon_{k}"] for k in ("s1", "s2")}
H52 = {k: R18[f"h5_minbias_{k}"] + R18[f"h5_dimuon_{k}"] for k in ("s1", "s2")}
LADDER = {
    "ours, physical weights": ((Q18["all"], Q18["all_w2"]), REF2, "every muon at its physical rate"),
    "ours, FairShip's 2018 weighting": ((Q18["rule2018"], Q18["rule2018_w2"]), REF2,
                                        "η / ω Dalitz muons weighted as in 2018"),
}
pres.radio("lad", list(LADDER), value=list(LADDER)[0], label="step", style="list")
pres.radio("lad_var", ["p", "pT"], value="p", label="in", style="segmented")


@pres.output("lad_plot", inputs=["lad", "lad_var"])
def lad_plot(step, var):
    fig = go.Figure()
    for i, (k, (o, r, _)) in enumerate(LADDER.items()):
        if var == "p":
            o1, o2 = o[0].sum(1), o[1].sum(1)
            r1, r2 = r["s1"].sum(1), r["s2"].sum(1)
            x, m = PC, PC >= P_MIN
        else:
            sel = PC >= P_MIN
            o1, o2 = o[0][sel].sum(0), o[1][sel].sum(0)
            r1, r2 = r["s1"][sel].sum(0), r["s2"][sel].sum(0)
            x = 0.5 * (TE[1:] + TE[:-1]); m = (x < 3.0) & (r1 > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            rat = o1 / r1
            err = rat * np.sqrt(o2 / o1 ** 2 + r2 / r1 ** 2)
            m = m & (o1 > 0) & (r1 > 0) & (err < 0.3 * rat)
        on = k == step
        fig.add_trace(go.Scatter(x=x[m], y=rat[m], mode="lines+markers" if on else "lines",
                                 line=dict(color=CAT[i], width=2.4 if on else 1.0, dash=None if on else "dot"),
                                 marker=dict(size=4), opacity=1 if on else 0.45,
                                 error_y=dict(type="data", array=err[m], visible=on, thickness=1, width=2),
                                 name=k, hovertemplate="%{x:.1f}: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=1, line=dict(color=INK, width=1))
    tot = {}
    for k, (o, r, _) in LADDER.items():
        sel = PC >= P_MIN
        tot[k] = o[0][sel].sum() / r["s1"][sel].sum()
    fig.add_annotation(x=0.99, y=0.97, xref="paper", yref="paper", xanchor="right", yanchor="top", showarrow=False, align="right",
                       text=f"<b>{tot[step]:.3f}</b> in total (p > 25 GeV)<br>{LADDER[step][2]}", font=dict(size=7, color=INK),
                       bgcolor="rgba(255,255,255,0.85)")
    if var == "p":
        fig.update_xaxes(title="muon momentum [GeV]", range=[20, 350])
    else:
        fig.update_xaxes(title="muon pT [GeV] (p > 25 GeV)", range=[0, 3])
    fig.update_yaxes(title="ours / 506 M sample", range=[0.6, 1.6])
    return layout(fig)


# 2D maps (as bin/plot_match_2018_2d.py)
pres.radio("m2_w", ["FairShip 2018 weighting", "physical weights"], value="FairShip 2018 weighting", label="ours with",
           style="segmented")
pres.radio("m2_show", ["ratio", "pull"], value="ratio", label="show", style="segmented")
pres.radio("m2_rebin", ["fine", "2 × 2 merged"], value="fine", label="cells", style="segmented")


@pres.output("m2_map", inputs=["m2_w", "m2_show", "m2_rebin"])
def m2_map(w, show, rebin):
    k = "rule2018" if w.startswith("FairShip") else "all"
    o, o2, r, r2, pe, te = Q18[k], Q18[f"{k}_w2"], REF2["s1"], REF2["s2"], PE, TE
    if rebin != "fine":
        n = 2
        cut = lambda a: a[: a.shape[0] // n * n, : a.shape[1] // n * n]
        merge = lambda a: cut(a).reshape(a.shape[0] // n, n, a.shape[1] // n, n).sum((1, 3))
        o, o2, r, r2 = merge(o), merge(o2), merge(r), merge(r2)
        pe, te = pe[: (len(pe) - 1) // n * n + 1: n], te[: (len(te) - 1) // n * n + 1: n]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = o / r
        pull = (o - r) / np.sqrt(o2 + r2)
        bad = ~((o > 0) & (r > 0) & (np.sqrt(o2) / o < 0.3) & (np.sqrt(r2) / r < 0.3))
    ratio[bad] = np.nan; pull[bad] = np.nan
    pc = np.sqrt(pe[1:] * pe[:-1])
    ratio[pc < P_MIN] = np.nan; pull[pc < P_MIN] = np.nan
    if show == "ratio":
        z, zmin, zmax, cs, ct = ratio, 0.6, 1.4, "RdBu_r", "ours / 2018"
    else:
        z, zmin, zmax, cs, ct = pull, -5, 5, "PuOr_r", "pull"
    fig = go.Figure(go.Heatmap(x=pe, y=te, z=z.T, zmin=zmin, zmax=zmax, zmid=1 if show == "ratio" else 0, colorscale=cs,
                               colorbar=dict(title=dict(text=ct, side="right"), thickness=8),
                               hovertemplate="p %{x:.0f} GeV, pT %{y:.1f} GeV: %{z:.2f}<extra></extra>"))
    good = ~np.isnan(pull)
    fig.add_annotation(x=0.99, y=0.97, xref="paper", yref="paper", xanchor="right", yanchor="top", showarrow=False, align="right",
                       text=f"pulls: mean {np.nanmean(pull):.2f}, rms {np.nanstd(pull):.2f} ({good.sum()} cells)",
                       font=dict(size=7), bgcolor="rgba(255,255,255,0.85)")
    fig.update_xaxes(title="p at production [GeV]", range=[P_MIN, 400])
    fig.update_yaxes(title="pT at production [GeV]", range=[0, 3])
    return layout(fig, plot_bgcolor="#d9dce1", margin=dict(l=36, r=4, t=6, b=26))


# inventory: class by class, mother by mother (bin/compare_inventory_2018.py report)
_IO = np.load(CV / "inventory_ours.npz")
_IR = np.load(CV / "inventory_ref.npz")
INV = {tuple(k): float(np.sum(_IO[f"v{i}"])) for i, k in enumerate(json.loads(str(_IO["keys"]))) if k[3] in ("flux", "w2", "mother")}
CLASS_LAB = {"768.75": "min. bias (768.75)", "7.6875": "dimuon class (7.69)", "326.16": "charm (326.16)",
             "9.37": "beauty (9.37)"}
PDG = {211: "π⁺", -211: "π⁻", 321: "K⁺", -321: "K⁻", 130: "K_L", 310: "K_S", 443: "J/ψ", 22: "γ", 221: "η", 223: "ω",
       113: "ρ⁰", 333: "φ", 331: "η′", 2212: "p", 2112: "n", 421: "D⁰", -421: "D̄⁰", 411: "D⁺", -411: "D⁻", 431: "D_s⁺",
       -431: "D_s⁻", 4122: "Λ_c", -4122: "Λ̄_c", 15: "τ⁻", -15: "τ⁺", 521: "B⁺", -521: "B⁻", 511: "B⁰", -511: "B̄⁰",
       531: "B_s", -531: "B̄_s", 0: "(none)", -13: "μ⁺", 13: "μ⁻"}
pres.radio("inv_w", ["FairShip 2018 weighting", "physical weights"], value="FairShip 2018 weighting", label="ours with",
           style="segmented")
pres.radio("inv_c", list(CLASS_LAB.values()), value=CLASS_LAB["768.75"], label="2018 class", style="list")


@pres.output("inv_plot", inputs=["inv_w", "inv_c"])
def inv_plot(w, c_lab):
    lab = "2018" if w.startswith("FairShip") else "phys"
    c = next(k for k, v in CLASS_LAB.items() if v == c_lab)
    moth = {}
    for k, v in INV.items():
        if k[1] == lab and k[2] == c and k[3] == "mother":
            moth.setdefault(k[4], [0.0, 0.0])[0] += v
    for x in _IR.files:
        if x.startswith(f"{c}|mother|"):
            moth.setdefault(int(x.split("|")[2]), [0.0, 0.0])[1] += float(_IR[x])
    top = sorted(moth.items(), key=lambda kv: -max(kv[1]))[:10]
    names = [PDG.get(q, str(q)) for q, _ in top]
    fo = sum(v for k, v in INV.items() if k[1] == lab and k[2] == c and k[3] == "flux")
    fr = float(_IR[f"{c}|flux"])
    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38], horizontal_spacing=0.08)
    fig.add_trace(go.Bar(x=names, y=[u for _, (u, _) in top], name="ours", marker_color=CAT[0],
                         hovertemplate="%{x}: %{y:.3g} / spill<extra>ours</extra>"), 1, 1)
    fig.add_trace(go.Bar(x=names, y=[v for _, (_, v) in top], name="2018 files", marker_color="#9aa1ad",
                         hovertemplate="%{x}: %{y:.3g} / spill<extra>2018</extra>"), 1, 1)
    rat = [u / v if v > 0 else np.nan for _, (u, v) in top]
    fig.add_trace(go.Bar(y=names, x=rat, orientation="h", marker_color=[CAT[1] if abs(r - 1) > 0.1 else CAT[2] for r in rat],
                         text=[f"{r:.2f}" if np.isfinite(r) else "only ours" for r in rat], textposition="outside",
                         textfont=dict(size=6.5), hovertemplate="%{y}: %{x:.3f}<extra></extra>", showlegend=False), 1, 2)
    fig.add_vline(x=1, line=dict(color=INK, width=1), row=1, col=2)
    fig.update_yaxes(title="muons / spill (p > 25 GeV)", row=1, col=1)
    fig.update_xaxes(title="parent of the muon", row=1, col=1)
    fig.update_xaxes(title="ours / 2018", range=[0, max(2.0, np.nanmax(rat) * 1.15 if np.isfinite(np.nanmax(rat)) else 2)],
                     row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.add_annotation(x=0.0, y=1.0, xref="paper", yref="paper", xanchor="left", yanchor="bottom", showarrow=False,
                       text=f"class total: ours {fo:.3g}, 2018 {fr:.3g} per spill, <b>ratio {fo / fr:.3f}</b>", font=dict(size=7))
    return layout(fig, barmode="group", bargap=0.25, showlegend=True, legend=legend(x=0.45, y=1.0, yanchor="bottom"),
                  margin=dict(l=40, r=8, t=16, b=26))


# ====================================================================== the complete generator
SUMMARY = json.loads((CV / "summary.json").read_text())


@pres.output("val_guns")
def val_guns():
    g = sorted(SUMMARY["guns"], key=lambda r: -r["yield_analog"])
    names = [r["name"] for r in g]
    fig = go.Figure(go.Scatter(x=names, y=[r["ratio"] for r in g], mode="markers",
                               error_y=dict(type="data", array=[r["err_ratio"] for r in g], thickness=1, width=2),
                               marker=dict(size=6, color=CAT[0]),
                               customdata=[[r["chi2"], r["ndf"], r["yield_analog"], r["target"]] for r in g],
                               hovertemplate="%{x} (%{customdata[3]}): %{y:.3f}, shape χ²/ndf %{customdata[0]:.1f}/%{customdata[1]}, "
                                             "%{customdata[2]:.2g} muons per particle<extra></extra>"))
    fig.add_hrect(y0=0.98, y1=1.02, fillcolor="#e9f3ec", line_width=0, layer="below")
    fig.add_hline(y=1, line=dict(color=INK, width=1))
    fig.update_yaxes(title="forced / unforced muon yield", range=[0.85, 1.15])
    fig.update_xaxes(title="particle gun (sorted by muon yield)", tickangle=-45, tickfont=dict(size=6))
    return layout(fig, margin=dict(l=40, r=8, t=6, b=44))


PV = np.load(RES / "pythia_versions" / "pythia_versions.npz")
pres.radio("pv_sp", ["π⁺", "π⁻", "K⁺", "K⁻", "K_L", "p", "γ"], value="π⁺", label="hadron", style="segmented")
_PV_IDX = {"π⁺": "pi+", "π⁻": "pi-", "K⁺": "K+", "K⁻": "K-", "K_L": "K_L", "p": "p", "γ": "gamma"}


@pres.output("pv_plot", inputs=["pv_sp"])
def pv_plot(sp):
    i = list(PV["names"]).index(_PV_IDX[sp])
    pe = PV["pe"]
    pc = np.sqrt(pe[1:] * pe[:-1])
    a, b = PV["A"][i].sum(1), PV["B"][i].sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (b / float(PV["nb"])) / (a / float(PV["na"]))
        e = r * np.sqrt(1 / np.maximum(a, 1) + 1 / np.maximum(b, 1))
    fig = go.Figure(go.Scatter(x=pc, y=r, mode="lines+markers", line=dict(color=CAT[1], width=2), marker=dict(size=4),
                               error_y=dict(type="data", array=e, thickness=1, width=2),
                               hovertemplate="%{x:.0f} GeV: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=1, line=dict(color=INK, width=1))
    fig.add_vrect(x0=200, x1=400, fillcolor="#fdf0e6", line_width=0, layer="below")
    fig.add_annotation(x=300, y=1.25, text="where the vessel rate comes from", showarrow=False, font=dict(size=6.5, color=MUT))
    fig.update_xaxes(title=f"{sp} momentum [GeV] (2 M p + Mo events each)", range=[20, 400])
    fig.update_yaxes(title="Pythia 8.316 / 8.223", range=[0.2, 1.3])
    return layout(fig)


# ====================================================================== statistics: importance sampling
pres.slider("rr_qmin", label="q_min", min=0.001, max=0.05, value=0.007, decimals=3)
pres.slider("rr_e1", label="E₁ [GeV]", min=50, max=300, value=150, step=10, decimals=0)
pres.slider("rr_n", label="N", min=1, max=4, value=3, step=0.5, decimals=1)
pres.slider("rr_pt0", label="pT₀ [GeV]", min=0.2, max=2.0, value=0.5, decimals=1)


@pres.output("rr_map", inputs=["rr_qmin", "rr_e1", "rr_n", "rr_pt0"])
def rr_map(qmin, e1, n, pt0):
    E = np.linspace(1, 400, 200)
    pt = np.linspace(0, 3, 121)
    q = np.minimum(1, np.maximum(qmin, (E[None, :] / e1) ** n) * np.maximum(1, (pt[:, None] / pt0) ** 2))
    fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5], horizontal_spacing=0.12)
    fig.add_trace(go.Heatmap(x=E, y=pt, z=np.log10(q), zmin=-3, zmax=0, colorscale="Blues", showscale=False,
                             hovertemplate="E %{x:.0f} GeV, pT %{y:.2f} GeV: q = %{customdata:.3f}<extra></extra>",
                             customdata=q), 1, 1)
    for j, (p, c) in enumerate(((0.0, CAT[0]), (1.0, CAT[1]), (2.0, CAT[2]))):
        qq = np.minimum(1, np.maximum(qmin, (E / e1) ** n) * max(1, (p / pt0) ** 2))
        fig.add_trace(go.Scatter(x=E, y=qq, mode="lines", line=dict(color=c, width=2), name=f"pT = {p:.0f} GeV"), 1, 2)
    eavg = np.trapz(np.minimum(1, np.maximum(qmin, (E / e1) ** n)) * E ** -2.5, E) / np.trapz(E ** -2.5, E)
    fig.update_xaxes(title="particle energy [GeV] (dark: always followed)", row=1, col=1); fig.update_yaxes(title="pT [GeV]", row=1, col=1)
    fig.update_xaxes(title="particle energy [GeV]", row=1, col=2)
    fig.update_yaxes(title="survival probability q (weight 1/q)", type="log", range=[-3.2, 0.05], dtick=1, row=1, col=2)
    fig.add_annotation(x=1.0, y=0.02, xref="paper", yref="paper", xanchor="right", yanchor="bottom", showarrow=False,
                       text=f"a falling E⁻²·⁵ spectrum keeps {eavg:.3f} of its particles (pT = 0)", font=dict(size=6.5, color=MUT))
    return layout(fig, showlegend=True, legend=legend(x=0.66, y=1.0, yanchor="bottom"), margin=dict(l=36, r=8, t=16, b=26))


# ====================================================================== the new sample vs the 506 M sample
NEW = np.load(RES / "target_talk" / "new_selected.npz") if (RES / "target_talk" / "new_selected.npz").exists() else None
REF5 = np.load(RES / "target_talk" / "histograms_506M.npz") if (RES / "target_talk" / "histograms_506M.npz").exists() else None
CPE, CTE = (NEW["PE"], NEW["TE"]) if NEW is not None else (PE, TE)
CPC = np.sqrt(CPE[1:] * CPE[:-1])
CMP_COMPS = {"all": ("minbias", "onia", "charm", "beauty"), "min. bias": ("minbias",), "quarkonia": ("onia",),
             "charm": ("charm",), "beauty": ("beauty",)}
GROUP_LAB = {"pi": "π", "K": "K", "eta/omega/rho/phi/eta'": "η ω ρ φ η′", "gamma": "γ → μμ", "quarkonia": "J/ψ, ψ′, Υ",
             "charm": "charm, τ", "beauty": "beauty", "other": "other"}
pres.radio("cmp_c", list(CMP_COMPS), value="all", label="muons from", style="segmented")
pres.radio("cmp_w", ["physical", "2018 weighting"], value="physical", label="new sample", style="segmented")
pres.radio("cmp_v", ["p", "pT"], value="p", label="in", style="segmented")
pres.radio("cmp_y", ["linear", "log"], value="log", label="y axis", style="segmented")


def _cmp(src, comps, lab, k, charge=("mu-", "mu+")):
    """Sum of the (p, pT) histograms of the components and charges (zeros where a component is missing)."""
    out = np.zeros((len(CPE) - 1, len(CTE) - 1))
    for c in comps:
        for ch in charge:
            key = f"{c}|{lab}_{ch}_2d_{k}"
            if key in src.files:
                out = out + src[key]
    return out


def _new_lab(w):
    return "phys" if w == "physical" else "w2018"


@pres.output("cmp_1d", inputs=["cmp_c", "cmp_w", "cmp_v", "cmp_y"])
def cmp_1d(comp, w, var, yscale):
    if NEW is None or REF5 is None:
        return note("histograms missing (bin/compare_new_vs_506M.py)")
    cs, lab = CMP_COMPS[comp], _new_lab(w)
    o1, o2 = _cmp(NEW, cs, lab, "s1"), _cmp(NEW, cs, lab, "s2")
    r1, r2 = _cmp(REF5, cs, "ref", "s1"), _cmp(REF5, cs, "ref", "s2")
    sel = CPC >= float(NEW["p_min"])
    if var == "p":
        x, edges = CPC, CPE
        a1, a2, b1, b2 = o1.sum(1), o2.sum(1), r1.sum(1), r2.sum(1)
        m = sel
    else:
        x, edges = 0.5 * (CTE[1:] + CTE[:-1]), CTE
        a1, a2, b1, b2 = o1[sel].sum(0), o2[sel].sum(0), r1[sel].sum(0), r2[sel].sum(0)
        m = np.ones(len(x), bool)
    dw = np.diff(edges)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.64, 0.36], vertical_spacing=0.04)
    for y, y2, nm, col in ((b1, b2, "506 M sample (2018)", "#9aa1ad"), (a1, a2, "new sample", CAT[0])):
        fig.add_trace(go.Scatter(x=x[m], y=(y / dw)[m], mode="lines", line=dict(color=col, width=2, shape="hvh"), name=nm,
                                 error_y=dict(type="data", array=(np.sqrt(y2) / dw)[m], thickness=1, width=0),
                                 hovertemplate="%{x:.2f}: %{y:.3g} Hz / GeV<extra>" + nm + "</extra>"), 1, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rat = a1 / b1
        err = rat * np.sqrt(a2 / a1 ** 2 + b2 / b1 ** 2)
    ok = m & (a1 > 0) & (b1 > 0) & (err < 0.5 * rat)
    fig.add_trace(go.Scatter(x=x[ok], y=rat[ok], mode="markers", marker=dict(size=4, color=CAT[1]),
                             error_y=dict(type="data", array=err[ok], thickness=1, width=2), showlegend=False,
                             hovertemplate="%{x:.2f}: %{y:.3f}<extra></extra>"), 2, 1)
    fig.add_hline(y=1, line=dict(color=INK, width=1), row=2, col=1)
    tot = a1[m].sum() / b1[m].sum()
    fig.add_annotation(x=0.99, y=0.98, xref="paper", yref="paper", xanchor="right", yanchor="top", showarrow=False, align="right",
                       text=f"new {a1[m].sum():.3g} Hz, 506 M {b1[m].sum():.3g} Hz: <b>{tot:.3f}</b>", font=dict(size=7),
                       bgcolor="rgba(255,255,255,0.85)")
    fig.update_yaxes(title="muons / spill / GeV (= Hz / GeV)", type="log" if yscale == "log" else "linear", row=1, col=1)
    fig.update_yaxes(title="new / 506 M", range=[0.4, 1.8], row=2, col=1)
    fig.update_xaxes(title="momentum at production [GeV]" if var == "p" else "pT at production [GeV] (p > 25 GeV)",
                     range=[20, 400] if var == "p" else [0, 4], row=2, col=1)
    return layout(fig, showlegend=True, legend=legend(x=0.0, y=1.0, yanchor="bottom"), margin=dict(l=44, r=8, t=14, b=26))


pres.radio("cmp_map", ["rate ratio new / 506 M", "statistics: n_eff new / 506 M"], value="rate ratio new / 506 M",
           label="show", style="list")


@pres.output("cmp_2d", inputs=["cmp_c", "cmp_w", "cmp_map"])
def cmp_2d(comp, w, show):
    if NEW is None or REF5 is None:
        return note("histograms missing")
    cs, lab = CMP_COMPS[comp], _new_lab(w)
    o1, o2 = _cmp(NEW, cs, lab, "s1"), _cmp(NEW, cs, lab, "s2")
    r1, r2 = _cmp(REF5, cs, "ref", "s1"), _cmp(REF5, cs, "ref", "s2")
    with np.errstate(divide="ignore", invalid="ignore"):
        if show.startswith("rate"):
            z = o1 / r1
            bad = ~((o1 > 0) & (r1 > 0) & (np.sqrt(o2) / o1 < 0.3) & (np.sqrt(r2) / r1 < 0.3))
            z[bad] = np.nan
            zz, zmin, zmax, cs_, ct, mid = z, 0.5, 1.5, "RdBu_r", "new / 506 M", 1.0
            cd, ht = z, "%{customdata:.2f}"
        else:
            ne_o = np.where(o2 > 0, o1 ** 2 / np.where(o2 > 0, o2, 1), 0)
            ne_r = np.where(r2 > 0, r1 ** 2 / np.where(r2 > 0, r2, 1), 0)
            z = ne_o / ne_r
            z[(ne_o == 0) & (ne_r == 0)] = np.nan
            zz = np.log10(np.clip(z, 1e-3, 1e3))
            zz[np.isnan(z)] = np.nan
            zmin, zmax, cs_, ct, mid = -2, 2, "PiYG", "n_eff ratio", 0.0
            cd, ht = np.stack([ne_o, ne_r], -1), "n_eff new %{customdata[0]:.3g}, 506 M %{customdata[1]:.3g}"
    zz[CPC < float(NEW["p_min"])] = np.nan
    fig = go.Figure(go.Heatmap(x=CPE, y=CTE, z=zz.T, zmin=zmin, zmax=zmax, zmid=mid, colorscale=cs_, customdata=np.swapaxes(cd, 0, 1),
                               colorbar=dict(title=dict(text=ct, side="right"), thickness=8,
                                             **({} if show.startswith("rate") else
                                                dict(tickvals=[-2, -1, 0, 1, 2], ticktext=["0.01", "0.1", "1", "10", "100"]))),
                               hovertemplate="p %{x:.0f} GeV, pT %{y:.1f} GeV: " + ht + "<extra></extra>"))
    fig.update_xaxes(title="p at production [GeV]", range=[25, 400])
    fig.update_yaxes(title="pT at production [GeV]", range=[0, 4])
    return layout(fig, plot_bgcolor="#d9dce1", margin=dict(l=36, r=4, t=6, b=26))


@pres.output("cmp_grp", inputs=["cmp_c", "cmp_w"])
def cmp_grp(comp, w):
    if NEW is None or REF5 is None:
        return note("histograms missing")
    groups = json.loads(str(NEW["groups"]))
    cs, lab = CMP_COMPS[comp], _new_lab(w)
    rows = []
    for i, g in enumerate(groups):
        a = sum(float(NEW[f"{c}|{lab}_grp_{i}_s1"][0]) for c in cs if f"{c}|{lab}_grp_{i}_s1" in NEW.files)
        b = sum(float(REF5[f"{c}|ref_grp_{i}_s1"][0]) for c in cs if f"{c}|ref_grp_{i}_s1" in REF5.files)
        na = sum(float(NEW[f"{c}|{lab}_grp_{i}_n"][0]) for c in cs if f"{c}|{lab}_grp_{i}_n" in NEW.files)
        nb = sum(float(REF5[f"{c}|ref_grp_{i}_n"][0]) for c in cs if f"{c}|ref_grp_{i}_n" in REF5.files)
        if a > 0 or b > 0:
            rows.append((GROUP_LAB.get(g, g), a, b, na, nb))
    rows.sort(key=lambda r: -max(r[1], r[2]))
    names = [r[0] for r in rows]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38], horizontal_spacing=0.1)
    fig.add_trace(go.Bar(x=names, y=[r[2] for r in rows], name="506 M sample", marker_color="#9aa1ad",
                         customdata=[r[4] for r in rows], hovertemplate="%{x}: %{y:.3g} Hz, %{customdata:.3g} rows<extra>506 M</extra>"), 1, 1)
    fig.add_trace(go.Bar(x=names, y=[r[1] for r in rows], name="new sample", marker_color=CAT[0],
                         customdata=[r[3] for r in rows], hovertemplate="%{x}: %{y:.3g} Hz, %{customdata:.3g} rows<extra>new</extra>"), 1, 1)
    rat = [r[1] / r[2] if r[2] > 0 else np.nan for r in rows]
    fig.add_trace(go.Bar(y=names, x=rat, orientation="h", marker_color=[CAT[1] if abs(x - 1) > 0.1 else CAT[2] for x in rat],
                         text=[f"{x:.2f}" if np.isfinite(x) else "only new" for x in rat], textposition="outside",
                         textfont=dict(size=6.5), showlegend=False, hovertemplate="%{y}: %{x:.3f}<extra></extra>"), 1, 2)
    fig.add_vline(x=1, line=dict(color=INK, width=1), row=1, col=2)
    fig.update_yaxes(title="muons / spill = Hz (p > 25 GeV)", type="log", row=1, col=1)
    fig.update_xaxes(title="parent of the muon", row=1, col=1)
    fin = [x for x in rat if np.isfinite(x)]
    fig.update_xaxes(title="new / 506 M", range=[0, max(2.0, 1.2 * max(fin)) if fin else 2], row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    return layout(fig, barmode="group", bargap=0.25, showlegend=True, legend=legend(x=0.3, y=1.0, yanchor="bottom"),
                  margin=dict(l=44, r=8, t=16, b=26))


@pres.output("cmp_charge", inputs=["cmp_c", "cmp_w"])
def cmp_charge(comp, w):
    if NEW is None or REF5 is None:
        return note("histograms missing")
    cs, lab = CMP_COMPS[comp], _new_lab(w)
    sel = CPC >= float(NEW["p_min"])
    fig = go.Figure()
    for src, lb, nm, col in ((REF5, "ref", "506 M sample", "#9aa1ad"), (NEW, lab, "new sample", CAT[0])):
        mp = _cmp(src, cs, lb, "s1", ("mu+",)).sum(1)
        mm = _cmp(src, cs, lb, "s1", ("mu-",)).sum(1)
        ep = _cmp(src, cs, lb, "s2", ("mu+",)).sum(1)
        em = _cmp(src, cs, lb, "s2", ("mu-",)).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = mp / mm
            e = r * np.sqrt(ep / mp ** 2 + em / mm ** 2)
        ok = sel & (mp > 0) & (mm > 0) & (e < 0.3 * r)
        fig.add_trace(go.Scatter(x=CPC[ok], y=r[ok], mode="lines+markers", line=dict(color=col, width=1.6), marker=dict(size=3),
                                 error_y=dict(type="data", array=e[ok], thickness=1, width=0), name=nm,
                                 hovertemplate="%{x:.0f} GeV: %{y:.3f}<extra>" + nm + "</extra>"))
    fig.add_hline(y=1, line=dict(color=INK, width=1, dash="dot"))
    fig.update_xaxes(title="momentum at production [GeV]", range=[20, 400])
    fig.update_yaxes(title="μ⁺ / μ⁻", rangemode="tozero")
    return layout(fig, showlegend=True, legend=legend(x=0.0, y=1.0, yanchor="bottom"))


# ====================================================================== storage policy (the veto talk's widgets)
import talk_storage  # noqa: E402

talk_storage.register(pres)

if __name__ == "__main__":
    pres.run()
