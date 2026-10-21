"""Live-talk widgets for the datasets and the end-to-end results (registered by veto_talk.py).

  ds_*     the muon datasets: independent input muons per type (gamma vs the enriched delta), the statistics probe
  e2e_*    end to end on the enriched dataset (bin/end_to_end.py): every background source before and after the
           veto, per design and cone; the signal-background trade-off; the design in 3D
Data: results/datasets/neff_by_type.json, results/probe/probe_<dataset>.npz, results/end_to_end/e2e.json.
"""
import json

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from veto import design as dsg
from veto import muon_types as mt
from veto import paths, viz3d

INK, ACC, ORA, GRN, RED, MUT = "#1C2230", "#2459A8", "#D07A2D", "#2A7F3F", "#B0352B", "#5B6475"
FONT = dict(family="Fira Sans, Helvetica, sans-serif", size=7, color=INK)
RES = paths.results_dir()
TYPE_LABELS = ["SST1 bounced", "SST1 selected", "SST1 other", "veto only", "rock bounced", "straight past",
               "stops 30.6–82 m", "stops in shield"]
PROBES = {"gamma (old input)": "design-optimization-dataset-gamma-2026-10-17_all",
          "delta (enriched input)": "design-optimization-dataset-delta-enriched-2026-10-19"}
SOURCES = [("combinatorial", "combinatorial", None), ("μ helium", "muon", "helium"), ("μ air", "muon", "air"),
           ("μ PVC walls", "muon", "pvc"), ("μ veto planes", "muon", "planes"), ("ν helium", "nu", "helium"),
           ("ν air", "nu", "air"), ("ν PVC walls", "nu", "pvc"), ("ν veto planes", "nu", "planes")]
SRC_COL = {"combinatorial": MUT, "muon": RED, "nu": ACC}


def layout(fig, **kw):
    # uirevision: a redraw (slider moved) keeps what the presenter did, e.g. a rotated 3D camera
    lay = dict(uirevision="keep", font=FONT, paper_bgcolor="white", plot_bgcolor="white", margin=dict(l=40, r=10, t=22, b=28), showlegend=False)
    lay.update(kw)
    fig.update_layout(**lay)
    return fig


def note(text):
    fig = go.Figure()
    fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", text=text, showarrow=False, font=dict(size=8, color=MUT))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return layout(fig, margin=dict(l=0, r=0, t=0, b=0))


def register(pres, plane_meshes, geometry, scene):
    # ---------------------------------------------------------------- datasets
    NEFF = json.loads((RES / "datasets" / "neff_by_type.json").read_text())

    @pres.output("ds_neff")
    def ds_neff():
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                            subplot_titles=("independent input muons per type", "muons per spill"))
        for name, col in (("gamma", MUT), ("delta", RED)):
            d = NEFF[name]
            fig.add_trace(go.Bar(x=TYPE_LABELS, y=d["neff"], name=f"{name}", marker_color=col,
                                 text=[f"{v:.2g}" for v in d["neff"]], textposition="outside", textfont=dict(size=5.5)), 1, 1)
            fig.add_trace(go.Bar(x=TYPE_LABELS, y=d["rate"], name=f"{name}", marker_color=col, showlegend=False), 1, 2)
        fig.update_yaxes(type="log", row=1, col=1, range=[1, 6.9])
        fig.update_yaxes(type="log", row=1, col=2)
        fig.update_xaxes(tickangle=-40, tickfont=dict(size=6))
        fig.update_annotations(font=dict(size=7))
        return layout(fig, showlegend=True, barmode="group", legend=dict(x=0.3, y=1.0, orientation="h", font=dict(size=6.5)),
                      margin=dict(l=36, r=6, t=26, b=56))

    probe = {}
    for label, stem in PROBES.items():
        f = RES / "probe" / f"probe_{stem}.npz"
        if f.exists():
            probe[label] = dict(np.load(f))
    pres.radio("probe_ds", list(PROBES), value="delta (enriched input)", label="dataset")

    @pres.output("probe3d", inputs=["probe_ds"])
    def probe3d(label):
        if label not in probe:
            return note(f"no probe file for {label}")
        d = probe[label]
        pl, cell = json.loads(str(d["planes"])), float(d["cell"])
        traces = geometry()
        for i, p in enumerate(pl):
            nu, nv = d[f"p{i}_rate"].shape
            u = np.minimum(p["u"][0] + np.arange(nu + 1) * cell, p["u"][1])
            v = np.minimum(p["y"][0] + np.arange(nv + 1) * cell, p["y"][1])
            U, V = np.meshgrid(u, v, indexing="ij")
            if p["kind"] == "fin":
                X, Z = p["sign"] * (U + mt.side_plane_x(p["z0"])), np.full_like(U, p["z0"])
            elif p["kind"] == "ubt":
                X, Z = U, np.full_like(U, p["z0"])
            else:
                X, Z = U, p["z0"] + (V - mt.FIN_Y_MIN) * np.tan(np.radians(p["tilt"]))
            n = d[f"p{i}_neff_in"]
            c = np.log10(np.where(n > 0, n, np.nan))
            traces.append(go.Surface(x=Z, y=X, z=V, surfacecolor=np.pad(c, ((0, 1), (0, 1)), mode="edge"), cmin=0, cmax=3,
                                     colorscale="Viridis", showscale=i == 0, opacity=0.95, hoverinfo="skip",
                                     colorbar=dict(title=dict(text="log₁₀ independent input muons / 25 cm cell", side="top",
                                                              font=dict(size=6)), orientation="h", x=0.5, y=0.02, len=0.55,
                                                   thickness=7, tickfont=dict(size=6))))
        return layout(go.Figure(traces), scene=scene(z_range=(26, 86), x_range=(-6, 6), y_range=(-3.6, 3.6)),
                      margin=dict(l=0, r=0, t=0, b=0))

    @pres.output("probe_bands")
    def probe_bands():
        """Median independent input muons per cell in bands of distance from the vessel wall (fins), and per floor plane."""
        fig = go.Figure()
        bands = ((0, 0.5), (0.5, 1), (1, 2), (2, 5))
        for (label, d), col in zip(probe.items(), (MUT, RED)):
            pl = json.loads(str(d["planes"]))
            xs, ys = [], []
            for lo, hi in bands:
                ns = []
                for i, p in enumerate(pl):
                    if p["kind"] == "fin":
                        r, n = d[f"p{i}_rate"], d[f"p{i}_neff_in"]
                        sl = slice(int(lo / 0.25), int(hi / 0.25))
                        ns += list(n[sl][r[sl] > 0])
                xs.append(f"fins {lo:g}–{hi:g} m"); ys.append(float(np.median(ns)))
            for i, p in enumerate(pl):
                if p["kind"] in ("floor", "ubt"):
                    r, n = d[f"p{i}_rate"], d[f"p{i}_neff_in"]
                    xs.append(p["name"].replace(" z=", " ")); ys.append(float(np.median(n[r > 0])))
            fig.add_trace(go.Bar(x=xs, y=ys, name=label.split(" ")[0], marker_color=col, text=[f"{v:.0f}" for v in ys],
                                 textposition="outside", textfont=dict(size=5.5)))
        fig.add_hline(y=10, line=dict(color=RED, dash="dot"))
        fig.update_yaxes(type="log", title_text="median per 25 cm cell", range=[0.5, 3.1])
        fig.update_xaxes(tickangle=-35, tickfont=dict(size=6))
        return layout(fig, showlegend=True, barmode="group", legend=dict(x=0.7, y=1.0, font=dict(size=6.5)),
                      margin=dict(l=40, r=6, t=10, b=50))

    # ---------------------------------------------------------------- end to end
    f = RES / "end_to_end" / "e2e.json"
    E2E = json.loads(f.read_text()) if f.exists() else {"rows": []}
    rows = E2E["rows"]
    designs = list(dict.fromkeys(r["design"] for r in rows)) or ["E3_robust"]
    cones = sorted({r["cone_mrad"] for r in rows}) or [0.0]
    pres.radio("e2e_design", designs, value=designs[0], label="design")
    pres.radio("e2e_cone", [f"{c:g} mrad" for c in cones], value=f"{cones[min(1, len(cones) - 1)]:g} mrad", label="cone")

    def row(design, cone_label):
        cone = float(cone_label.split()[0])
        return next((r for r in rows if r["design"] == design and r["cone_mrad"] == cone), None)

    def value(r, kind, tgt, key):
        if kind == "combinatorial":
            return r["combinatorial"]["before" if key == "selected" else "after"]
        return (r[kind].get(tgt) or {}).get(key, 0.0)

    @pres.output("e2e_sources", inputs=["e2e_design", "e2e_cone"])
    def e2e_sources(design, cone_label):
        r = row(design, cone_label)
        if r is None:
            return note("no end-to-end result yet (bin/end_to_end.py)")
        names = [s[0] for s in SOURCES][::-1]
        fig = go.Figure()
        for key, label, op in (("selected", "selected, before the veto", 0.35), ("after", "after the veto", 1.0)):
            vals = [max(value(r, k, t, key), 1e-4) for _, k, t in SOURCES][::-1]
            fig.add_trace(go.Bar(y=names, x=vals, orientation="h", name=label, opacity=op,
                                 marker_color=[SRC_COL[k] for _, k, _ in SOURCES][::-1],
                                 text=[f"{v:.2g}" if v > 1e-4 else "0" for v in vals] if key == "after" else None,
                                 textposition="outside", textfont=dict(size=6)))
        fig.update_xaxes(type="log", title_text="events per 15 years", range=[-4.2, 5.6])
        fig.add_annotation(x=0.99, y=0.02, xref="paper", yref="paper", xanchor="right", showarrow=False, align="right",
                           text=(f"<b>FN {r['total']:.2g} ± {r['total_err']:.1g} / 15 y</b><br>FP {100 * r['signal_lost']:.2f} "
                                 f"of 100 signal  ·  {r['cost_kchf']:.0f} kCHF  ·  reward {r['reward']:.1f}"),
                           font=dict(size=6.5))
        return layout(fig, showlegend=True, barmode="overlay", legend=dict(x=0.55, y=1.06, orientation="h", font=dict(size=6)),
                      margin=dict(l=70, r=10, t=14, b=30))

    @pres.output("e2e_trade")
    def e2e_trade():
        if not rows:
            return note("no end-to-end result yet")
        fig = go.Figure()
        cols = [ACC, ORA, GRN, RED, "#7b3294", MUT]
        for name, col in zip(designs, cols):
            rr = sorted((r for r in rows if r["design"] == name), key=lambda r: r["cone_mrad"])
            fig.add_trace(go.Scatter(x=[max(100 * r["signal_lost"], 1e-2) for r in rr], y=[r["total"] for r in rr],
                                     error_y=dict(array=[r["total_err"] for r in rr], thickness=0.8, width=2),
                                     mode="lines+markers+text", text=[f"{r['cone_mrad']:g}" for r in rr], textposition="top right",
                                     textfont=dict(size=6, color=col), line=dict(color=col, width=2), marker=dict(size=6), name=name))
        fig.update_xaxes(type="log", title_text="FP: signal events vetoed, of 100")
        fig.update_yaxes(type="log", title_text="FN: background events / 15 y")
        return layout(fig, showlegend=True, legend=dict(x=0.02, y=0.04, font=dict(size=6.5)), margin=dict(l=46, r=10, t=10, b=30))

    @pres.output("e2e_design3d", inputs=["e2e_design"])
    def e2e_design3d(design):
        p = RES / "end_to_end" / "designs" / f"{design}.json"
        try:
            d = dsg.load(p if p.exists() else design)
        except Exception as e:  # noqa: BLE001
            return note(f"{design}: {e}")
        return layout(go.Figure(geometry(ubt=True) + plane_meshes(d, scale=False)),
                      scene=scene(z_range=(26, 86), x_range=(-6, 6), y_range=(-3.6, 3.6)), margin=dict(l=0, r=0, t=0, b=0))
