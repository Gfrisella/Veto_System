"""Live-talk widgets for the simulation after the shield and the interaction backgrounds (registered by veto_talk.py).

  sim_*  ported from fast-veto-sim's talk (docs/live_talk/fvs_talk.py there): what a muon crosses, cross sections,
         one library interaction at a time, grazing walls, forced interactions in a design, decays, multiple
         scattering with tails, closure against Geant4. Data: fast-veto-sim's docs/live_talk/data (talk_data.npz,
         geant4_samples.npz) and results; code: fastveto, loaded after our veto package (so it runs on ours).
  iv_*   our selection and veto on the interaction backgrounds: one selected muon-interaction event with the
         backward search and the vertex cone, the cone trade-off, the background by source.
"""
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from veto import design as dsg
from veto import geometry as geo
from veto import muon_types as mt
from veto import paths, reco, scattering, viz3d

FVS = Path.home() / "workspace/uzh/fast-veto-sim"
sys.path.insert(0, str(FVS))
from fastveto import decay as fdecay  # noqa: E402
from fastveto import kokoulin  # noqa: E402
from fastveto import targets as tg  # noqa: E402

INK, ACC, ORA, GRN, RED, MUT = "#1C2230", "#2459A8", "#D07A2D", "#2A7F3F", "#B0352B", "#5B6475"
FONT = dict(family="Fira Sans, Helvetica, sans-serif", size=7, color=INK)
SPILLS_15Y = 6e20 / 5e13
D = dict(np.load(FVS / "docs/live_talk/data/talk_data.npz"))
G4MSC = dict(np.load(FVS / "docs/live_talk/data/geant4_samples.npz"))
CLOSURE = json.loads((FVS / "results/dis_closure/closure.json").read_text())
IX = paths.results_dir() / "interactions"
TECH_COL = {"straw": "#8c6bb1", "fiber": "#41ab5d", "pixel": "#2171b5", "tile": ORA, "tile2": "#e6550d"}
PART = {13: ("μ⁻", RED, None), -13: ("μ⁺", RED, None), 11: ("e⁻", ORA, None), -11: ("e⁺", ORA, None),
        211: ("π⁺", ACC, None), -211: ("π⁻", ACC, None), 321: ("K⁺", "#7b3294", None), -321: ("K⁻", "#7b3294", None),
        2212: ("p", GRN, None), -2212: ("p̄", GRN, None), 2112: ("n", "#999999", "dash"), -2112: ("n̄", "#999999", "dash"),
        22: ("γ", "#d9b300", "dot"), 130: ("K⁰L", "#8c510a", "dash"), 310: ("K⁰S", "#8c510a", "dash"),
        3122: ("Λ", "#01665e", "dash"), 14: ("ν_μ", "#bbbbbb", "dot"), -14: ("ν̄_μ", "#bbbbbb", "dot"),
        12: ("ν_e", "#bbbbbb", "dot"), -12: ("ν̄_e", "#bbbbbb", "dot"), 111: ("π⁰", "#d9b300", "dot")}
CHARGED = {11, 13, 211, 321, 2212, 3112, 3222, 3312, 3334, 15, 411, 431}
MATCOL = {"helium": "#6baed6", "air": "#bdbdbd", "pvc": RED, "mylar": "#8c6bb1", "polystyrene": "#41ab5d",
          "silicon": "#2171b5", "pvt": ORA}
TYPE_NAMES = {0: "bounced into SST1", 1: "SST1, selected", 2: "SST1, other", 3: "through a wall / the window",
              4: "bounced in the rock", 5: "straight past", 6: "stops at 30.6–82 m"}


def pname(q):
    return PART.get(int(q), (f"{int(q)}", MUT, "dot"))


def layout(fig, **kw):
    lay = dict(font=FONT, paper_bgcolor="white", plot_bgcolor="white", margin=dict(l=40, r=10, t=22, b=28), showlegend=False)
    lay.update(kw)
    fig.update_layout(**lay)
    return fig


def scene(x_range=(-8, 8), y_range=(-6, 8), z_range=(-3, 98), eye=(-0.75, -1.05, 0.5), aspect=(2.6, 1.0, 0.8),
          titles=("z [m]", "x [m]", "y [m]")):
    def axis(title, rng):
        return dict(title=title, range=list(rng), backgroundcolor="white", gridcolor="#e6e8ee", zeroline=False,
                    showspikes=False, tickfont=dict(size=6))
    return dict(xaxis=axis(titles[0], z_range), yaxis=axis(titles[1], x_range), zaxis=axis(titles[2], y_range),
                aspectmode="manual", aspectratio=dict(x=aspect[0], y=aspect[1], z=aspect[2]),
                camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2])))


def lines3d(paths_, color, width=2, dash=None, name="", opacity=0.9):
    zs, xs, ys = [], [], []
    for pts in paths_:
        for x, y, z in pts:
            zs.append(float(z)); xs.append(float(x)); ys.append(float(y))
        zs.append(None); xs.append(None); ys.append(None)
    return go.Scatter3d(x=zs, y=xs, z=ys, mode="lines", line=dict(color=color, width=width, dash=dash),
                        opacity=opacity, name=name, hoverinfo="skip")


def pts3d(pts, color, size=4, symbol="circle", name="", opacity=1.0):
    p = np.asarray(pts, float).reshape(-1, 3)
    return go.Scatter3d(x=p[:, 2], y=p[:, 0], z=p[:, 1], mode="markers", name=name, hoverinfo="skip", opacity=opacity,
                        marker=dict(size=size, color=color, symbol=symbol, line=dict(width=0)))


def text_block(fig, lines, x0, y0=0.97, dy=0.07, size=6.5, bold_first=True):
    for k, line in enumerate(lines):
        fig.add_annotation(text=f"<b>{line}</b>" if bold_first and k == 0 else line, x=x0, y=y0 - k * dy, xref="paper",
                           yref="paper", xanchor="left", yanchor="top", showarrow=False, font=dict(size=size, color=INK),
                           align="left")
    return fig


def design_traces(d, vessel=True, opacity=0.35):
    tr = [t for t in viz3d.geometry_traces() if t.name not in ("UBT",)] if vessel else []
    for pl in d.planes:
        col = TECH_COL.get(pl.technology, MUT)
        if pl.kind == "ubt":
            x0, x1 = -geo.UBT_HALF_X, geo.UBT_HALF_X
        else:
            xin = pl.sign * mt.side_plane_x(pl.z)
            x0, x1 = sorted((xin, xin + pl.sign * pl.u_max))
        tr.append(go.Mesh3d(x=[pl.z] * 4, y=[x0, x1, x1, x0], z=[pl.y_lo, pl.y_lo, pl.y_hi, pl.y_hi], i=[0, 0], j=[1, 2],
                            k=[2, 3], color=col, opacity=opacity, hoverinfo="skip", flatshading=True))
    return tr


class SigmaTable:
    """sigma(material, E) per nucleon [ub] from the extracted library tables (no 2 GB library on the laptop)."""

    def sigma(self, material, E, split="crossover"):
        x = np.log(np.clip(E, D["E_fine"][0], D["E_fine"][-1]))
        tot = D[f"sig_mu_{split}_{material}_p6"] + D[f"sig_mu_{split}_{material}_g4"]
        return np.where(E < D["E_fine"][0], 0.0, np.interp(x, np.log(D["E_fine"]), tot))


def mu_path(r, i):
    a = [r["x_30"][i], r["y_30"][i], r["z_30"][i]]
    b = [r["x_82"][i], r["y_82"][i], r["z_82"][i]]
    k = [r["kx"][i], r["ky"][i], r["kz"][i]]
    return [a, k, b] if np.isfinite(k[2]) else [a, b]


def register(pres, maps):
    """Declare the sliders, radios and outputs on `pres`; maps: the reco inputs (rate maps) of the talk."""
    DES = {n: dsg.load(n) for n in ("E3_robust", "D6_straw", "D6_tile2")}

    # ---------------------------------------------------------------- what a muon crosses
    keep = np.nonzero(np.isin(D["ai_code"], [0, 2, 10]))[0]
    mu_int = {k[len("ai_par_"):]: D[k][keep].astype(np.float64) for k in D if k.startswith("ai_par_")}
    pres.slider("sim_mu_pick", label="muon", min=0, max=299, value=7, step=1, decimals=0)
    pres.radio("sim_mu_design", list(DES), value="E3_robust", label="design", style="segmented")

    @pres.output("sim_mu_one3d", inputs=["sim_mu_pick", "sim_mu_design"])
    def sim_mu_one3d(pick, dname):
        i = int(pick) % len(keep)
        r = {k: v[i:i + 1] for k, v in mu_int.items()}
        d = DES[dname]
        cr = tg.crossings(r, d)
        lib = SigmaTable()
        P = np.zeros(len(cr["E"]))
        for mi, mname in enumerate(tg.MATERIAL_NAMES):
            m = cr["material"] == mi
            P[m] = cr["n_area"][m] * lib.sigma(mname, cr["E"][m]) * 1e-34
        fig = make_subplots(rows=1, cols=2, column_widths=[0.66, 0.34], specs=[[{"type": "scene"}, {"type": "xy"}]])
        for t in design_traces(d, opacity=0.25):
            fig.add_trace(t, 1, 1)
        fig.add_trace(lines3d([mu_path(r, 0)], INK, 2.5), 1, 1)
        lines = [f"muon {int(pick)}: |p| {np.linalg.norm([r['px_30'][0], r['py_30'][0], r['pz_30'][0]]):.1f} GeV at 30.6 m",
                 f"type: {TYPE_NAMES.get(int(r['type'][0]), '?')}", ""]
        for j in np.argsort(cr["z"]):
            mname = tg.MATERIAL_NAMES[cr["material"][j]]
            p0, p1 = cr["p0"][j], cr["p1"][j]
            if np.allclose(p0, p1):
                fig.add_trace(pts3d([p0], MATCOL[mname], 5), 1, 1)
            else:
                fig.add_trace(lines3d([[p0, p1]], MATCOL[mname], 7 if mname == "helium" else 4, opacity=0.6), 1, 1)
            L = np.linalg.norm(p1 - p0)
            lines.append(f"{tg.target_name(int(cr['code'][j]), d)[:16]:16s} z {cr['z'][j]:5.1f} m  "
                         f"{'L ' + format(L, '.2f') + ' m' if L > 1e-3 else 'thin'}  P = {P[j]:.1e}")
        lines += ["", f"sum P(interaction) = {P.sum():.2e}"]
        fig.update_layout(scene=scene(z_range=(28, 84), x_range=(-5, 5), y_range=(-4, 4), aspect=(2.2, 1.0, 0.8)))
        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        text_block(fig, lines, 0.67, 0.98, 0.052, size=5.8)
        return layout(fig, margin=dict(l=0, r=0, t=0, b=0))

    # ---------------------------------------------------------------- cross sections
    pres.radio("sim_xs_split", ["crossover", "w2", "geant4"], value="crossover", label="split", style="segmented")
    pres.radio("sim_xs_mat", ["helium", "pvc", "mylar", "silicon", "pvt"], value="helium", label="material", style="segmented")

    @pres.output("sim_xs_mu", inputs=["sim_xs_split", "sim_xs_mat"])
    def sim_xs_mu(split, mat):
        E = D["E_fine"]
        A = {"helium": 4.0, "pvc": 20.8, "mylar": 12.7, "silicon": 28.1, "pvt": 11.7}[mat]
        k = kokoulin.sigma(E, A=A) / A
        p6, g4 = D[f"sig_mu_{split}_{mat}_p6"], D[f"sig_mu_{split}_{mat}_g4"]
        fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4], horizontal_spacing=0.12)
        fig.add_trace(go.Scatter(x=E, y=k, line=dict(color=INK, width=2), name="Geant4 total (Kokoulin)"), 1, 1)
        fig.add_trace(go.Scatter(x=E, y=g4, line=dict(color=GRN, width=1.5), name="Geant4 part"), 1, 1)
        fig.add_trace(go.Scatter(x=E, y=p6, line=dict(color=RED, width=1.5), name="Pythia 6 part"), 1, 1)
        fig.add_trace(go.Scatter(x=E, y=p6 + g4, line=dict(color=ACC, width=3), opacity=0.6, name="library total"), 1, 1)
        fig.add_trace(go.Scatter(x=E, y=(p6 + g4) / np.where(k > 0, k, np.nan), line=dict(color=ACC, width=2), showlegend=False), 1, 2)
        fig.add_trace(go.Scatter(x=E, y=np.ones_like(E), line=dict(color=MUT, width=0.8, dash="dot"), showlegend=False), 1, 2)
        fig.update_xaxes(type="log", title_text="muon energy [GeV]")
        fig.update_yaxes(type="log", range=[-3, 1.6], title_text="σ per nucleon [μb]", row=1, col=1)
        fig.update_yaxes(range=[0, 1.5], title_text="library / Geant4", row=1, col=2)
        return layout(fig, showlegend=True, legend=dict(x=0.02, y=0.02, font=dict(size=6)))

    # ---------------------------------------------------------------- one library interaction at a time
    pres.slider("sim_i_E", label="E", min=0, max=4, value=3, step=1, decimals=0)
    pres.slider("sim_i_ev", label="event", min=0, max=299, value=11, step=1, decimals=0)
    pres.slider("sim_i_zoom", label="angle ×", min=0, max=3, value=1, step=1, decimals=0)
    pres.radio("sim_i_gen", ["any", "Pythia 6", "Geant4"], value="Pythia 6", label="generator", style="segmented")

    def event_3d(pdg, p, v, beam, zoom):
        traces, rows = [lines3d([[[0, 0, -1.2], [0, 0, 0]]], RED, 4)], []
        pm = np.linalg.norm(p[:, :3], axis=1)
        for j in np.argsort(-pm):
            name, col, dash = pname(pdg[j])
            u = p[j, :3] / max(pm[j], 1e-9)
            L = 0.35 + 0.45 * np.log10(1 + pm[j] / 0.1)
            start = v[j] * zoom * np.array([1, 1, 1 / zoom]) if np.any(v[j]) else np.zeros(3)
            end = start + L * np.array([u[0] * zoom, u[1] * zoom, u[2]])
            traces.append(lines3d([[start, end]], col, 5 if beam[j] else 3 if int(abs(pdg[j])) in CHARGED else 2,
                                  dash=None if beam[j] else dash))
            if np.any(v[j]):
                traces.append(pts3d([start], col, 3, symbol="diamond"))
            rows.append((name + (" (beam)" if beam[j] else ""), pm[j], np.arccos(np.clip(u[2], -1, 1)) * 1e3,
                         np.linalg.norm(v[j]) * 100))
        traces.append(pts3d([[0, 0, 0]], INK, 5))
        return traces, rows

    @pres.output("sim_int3d", inputs=["sim_i_E", "sim_i_ev", "sim_i_zoom", "sim_i_gen"])
    def sim_int3d(ei, ev, zoom_i, gen):
        E = float(D["mulib_Egrid"][int(ei)])
        ints = np.nonzero(D["mulib_E"] == E)[0]
        if gen != "any":
            c = 0 if gen == "Pythia 6" else 1
            ints = ints[D["mulib_comp"][ints] == c] if np.any(D["mulib_comp"][ints] == c) else ints
        k = int(ints[int(ev) % len(ints)])
        m = D["mulib_int"] == k
        pdg, p, v, beam = D["mulib_pdg"][m], D["mulib_p"][m].astype(float), D["mulib_v"][m].astype(float), D["mulib_beam"][m]
        zoom = [1, 10, 100, 1000][int(zoom_i)]
        traces, rows = event_3d(pdg, p, v, beam, zoom)
        fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38], specs=[[{"type": "scene"}, {"type": "xy"}]])
        for t in traces:
            fig.add_trace(t, 1, 1)
        nu = D["mulib_Elib"][k] - (p[beam, 3].sum() if beam.any() else 0)
        lines = [f"μ⁻ of {E:g} GeV on helium: {'Pythia 6 (γ*N, W > 2)' if D['mulib_comp'][k] == 0 else 'Geant4 muonNuclear'}",
                 f"energy transfer ν = {nu:.2f} GeV, {len(pdg)} particles", "particle      p [GeV]   θ [mrad]   from [cm]"]
        for name, pmag, th, dv in rows[:16]:
            lines.append(f"{name:12s} {pmag:8.2f} {th:9.1f} {dv:9.1f}" if dv > 0.01 else f"{name:12s} {pmag:8.2f} {th:9.1f}")
        if len(rows) > 16:
            lines.append(f"... {len(rows) - 16} more (soft)")
        fig.update_layout(scene=scene(x_range=(-1.5, 1.5), y_range=(-1.5, 1.5), z_range=(-1.3, 2.2), aspect=(1.6, 1, 1),
                                      eye=(-1.2, -1.4, 0.6), titles=("z (beam) [m]", f"x × {zoom}", f"y × {zoom}")))
        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        text_block(fig, lines, 0.63, 0.98, 0.048, size=5.8)
        return layout(fig, margin=dict(l=0, r=0, t=0, b=0))

    # ---------------------------------------------------------------- grazing walls
    pres.slider("sim_w_ang", label="incidence", min=0.3, max=90, value=2, decimals=1)
    lat_angles = _wall_angles()

    @pres.output("sim_wall", inputs=["sim_w_ang"])
    def sim_wall(ang):
        a = np.geomspace(0.05, 90, 300)
        t = 0.3
        L = t / np.sin(np.radians(a))
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.14)
        fig.add_trace(go.Scatter(x=a, y=L, line=dict(color=INK, width=2), name="exact chord"), 1, 1)
        fig.add_trace(go.Scatter(x=a, y=np.minimum(L, 50 * t), line=dict(color=RED, width=2, dash="dash"), name="old: capped at 50 t"), 1, 1)
        la = t / np.sin(np.radians(ang))
        fig.add_trace(go.Scatter(x=[ang], y=[la], mode="markers", marker=dict(size=7, color=ACC), showlegend=False), 1, 1)
        fig.add_annotation(x=np.log10(ang), y=np.log10(la), text=f"{la:.1f} mm", showarrow=True, ax=30, ay=-20, font=dict(size=7))
        h, e = lat_angles
        fig.add_trace(go.Scatter(x=np.sqrt(e[1:] * e[:-1]), y=h / h.sum(), line=dict(color=ACC, shape="hvh", width=2),
                                 showlegend=False), 1, 2)
        fig.add_vline(x=np.degrees(np.arcsin(1 / 50)), line=dict(color=RED, dash="dot"), row=1, col=2)
        fig.update_xaxes(type="log", title_text="angle to the wall [deg]")
        fig.update_yaxes(type="log", title_text="path in 0.3 mm PVC [mm]", row=1, col=1)
        fig.update_yaxes(title_text="muons through a side wall (weighted)", row=1, col=2)
        return layout(fig, showlegend=True, legend=dict(x=0.25, y=0.98, font=dict(size=6)))

    # ---------------------------------------------------------------- forced interactions in a design (D6_straw)
    targets = {"helium": [0], "PVC walls": [2], "air": [1], "UBT": [10], "fins": list(range(11, 19))}
    pres.radio("sim_v_tgt", list(targets), value="helium", label="target", style="segmented")
    pres.radio("sim_v_kind", ["muons", "neutrinos"], value="muons", label="from", style="segmented")
    pres.slider("sim_v_ev", label="interaction", min=0, max=59, value=3, step=1, decimals=0)

    def applied(kind):
        t = "a" if kind == "muons" else "n"
        return t, {k[len(t) + 2:]: D[k] for k in D if k.startswith(f"{t}i_")}, {k[len(t) + 2:]: D[k] for k in D if k.startswith(f"{t}p_")}

    @pres.output("sim_vtx3d", inputs=["sim_v_tgt", "sim_v_kind"])
    def sim_vtx3d(tgt, kind):
        t, I, _ = applied(kind)
        traces = design_traces(DES["D6_straw"], opacity=0.25)
        for c, col in ((0, "#6baed6"), (2, RED)):
            traces.append(pts3d(D[f"{t}cloud{c}"][:2500], col, 1.4, opacity=0.5))
        m = np.isin(I["code"], targets[tgt])
        traces.append(pts3d(np.stack([I["x"][m], I["y"][m], I["z"][m]], 1), INK, 3))
        return layout(go.Figure(traces), scene=scene(z_range=(25, 86), x_range=(-6, 6), y_range=(-4, 5), aspect=(2.4, 1, 0.8)),
                      margin=dict(l=0, r=0, t=0, b=0))

    @pres.output("sim_ai3d", inputs=["sim_v_tgt", "sim_v_kind", "sim_v_ev"])
    def sim_ai3d(tgt, kind, ev):
        t, I, P = applied(kind)
        cand = np.nonzero(np.isin(I["code"], targets[tgt]))[0]
        if not len(cand):
            return layout(go.Figure())
        k = int(cand[int(ev) % len(cand)])
        m = P["int"] == k
        vtx = np.array([I["x"][k], I["y"][k], I["z"][k]])
        fig = make_subplots(rows=1, cols=2, column_widths=[0.64, 0.36], specs=[[{"type": "scene"}, {"type": "xy"}]])
        for tr in design_traces(DES["D6_straw"], opacity=0.2):
            fig.add_trace(tr, 1, 1)
        if kind == "muons":
            a = [I["par_x_30"][k], I["par_y_30"][k], I["par_z_30"][k]]
            b = [I["par_x_82"][k], I["par_y_82"][k], I["par_z_82"][k]]
            kk = [I["par_kx"][k], I["par_ky"][k], I["par_kz"][k]]
            fig.add_trace(lines3d([[a, kk, b] if np.isfinite(kk[2]) else [a, b]], INK, 2.5, opacity=0.7), 1, 1)
            parent = f"{'μ⁺' if I['pdg_mu'][k] < 0 else 'μ⁻'} of {I['E'][k]:.1f} GeV"
            extra = "Pythia 6" if I["component"][k] == 0 else "Geant4 muonNuclear"
        else:
            u = np.array([I["ux"][k], I["uy"][k], I["uz"][k]])
            fig.add_trace(lines3d([[vtx - u * (vtx[2] - 25) / u[2], vtx]], "#999999", 2.5, dash="dot"), 1, 1)
            parent = f"{pname(I['flavour'][k])[0]} of {I['E'][k]:.1f} GeV"
            bits = int(I["proc"][k])
            procs = [n for b_, n in ((1, "QE"), (2, "RES"), (4, "DIS"), (8, "COH"), (16, "MEC")) if bits & b_]
            extra = f"{'CC' if I['component'][k] == 0 else 'NC'} {'/'.join(procs)}" + (" + charm" if I["charm"][k] else "")
        rows, n_sst1 = [], 0
        pdg, p, v, beam = P["pdg"][m], P["p"][m].astype(float), P["v"][m].astype(float), P["beam"][m]
        for j in np.argsort(-np.linalg.norm(p[:, :3], axis=1)):
            name, col, dash = pname(pdg[j])
            pm = np.linalg.norm(p[j, :3])
            if pm < 0.2 or abs(int(pdg[j])) in (12, 14, 16):
                continue
            u = p[j, :3] / pm
            zend = 82.0 if u[2] > 0 else max(v[j, 2] - 5, 25)
            s = min(max((zend - v[j, 2]) / u[2] if abs(u[2]) > 1e-6 else 5, 0.5), 60)
            end = v[j] + u * s
            charged = abs(int(pdg[j])) in CHARGED
            fig.add_trace(lines3d([[v[j], end]], col, 4 if beam[j] else 2.5 if charged else 1.5, dash=None if charged else dash), 1, 1)
            hit = charged and pm > 1 and u[2] > 0 and abs(end[0]) < 2 and abs(end[1]) < 3 and abs(end[2] - 82) < 1e-6
            n_sst1 += hit
            rows.append(f"{name + (' (μ)' if beam[j] else ''):10s} {pm:7.2f} GeV{'  → SST1' if hit else ''}")
        fig.add_trace(pts3d([vtx], INK, 5), 1, 1)
        lines = [f"{parent} in the {tg.target_name(int(I['code'][k]), DES['D6_straw'])}", extra,
                 f"vertex z = {vtx[2]:.2f} m;  weight {I['weight'][k] * SPILLS_15Y:.3g} per 15 years",
                 f"charged p > 1 GeV into SST1: {n_sst1}", ""] + rows[:15]
        fig.update_layout(scene=scene(z_range=(25, 86), x_range=(-6, 6), y_range=(-4, 5), aspect=(2.4, 1, 0.8)))
        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        text_block(fig, lines, 0.65, 0.98, 0.052, size=5.8)
        return layout(fig, margin=dict(l=0, r=0, t=0, b=0))

    # ---------------------------------------------------------------- decays
    dk = {"K⁰S": 310, "Λ": 3122, "Σ⁺": 3222, "Ω⁻": 3334, "η": 221, "π⁰": 111}
    pres.radio("sim_dk_part", list(dk), value="K⁰S", label="particle", style="segmented")
    pres.slider("sim_dk_p", label="p", min=0.5, max=100, value=10, decimals=1)

    @pres.output("sim_decay3d", inputs=["sim_dk_part", "sim_dk_p"])
    def sim_decay3d(name, pmom):
        code = dk[name]
        M = fdecay.MASS[abs(code)]
        n, rng = 25, np.random.default_rng(5)
        p = np.tile([0, 0, pmom, np.sqrt(pmom ** 2 + M ** 2)], (n, 1))
        ev, pdg, pp, vv = fdecay.decay_all(np.arange(n), np.full(n, code), p, np.zeros((n, 3)), rng)
        big = fdecay.decay_all(np.arange(4000), np.full(4000, code), np.tile(p[0], (4000, 1)), np.zeros((4000, 3)), rng)
        fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38], specs=[[{"type": "scene"}, {"type": "xy"}]])
        L = 0.3 + pmom / M * fdecay.CTAU.get(abs(code), 1e-9) * 2
        fig.add_trace(lines3d([[[0, 0, -0.1 * L], [0, 0, 0]]], INK, 3), 1, 1)
        for i in range(n):
            for j in np.nonzero(ev == i)[0]:
                nm, col, dash = pname(pdg[j])
                u = pp[j, :3] / max(np.linalg.norm(pp[j, :3]), 1e-9)
                ch = abs(int(pdg[j])) in CHARGED
                fig.add_trace(lines3d([[[0, 0, 0], vv[j]], [vv[j], vv[j] + u * L * 0.6]], col, 2.5 if ch else 1.2,
                                      dash=None if ch else dash, opacity=0.7), 1, 1)
        u_, c_ = np.unique(big[1], return_counts=True)
        first_vtx = np.array([np.linalg.norm(big[3][big[0] == i][0]) for i in range(400)])
        lines = [f"{name}, p = {pmom:.1f} GeV:  βγcτ = {pmom / M * fdecay.CTAU[abs(code)]:.3g} m",
                 f"median decay distance {np.median(first_vtx):.3g} m", "", "final products per decay:"]
        lines += [f"  {pname(q)[0]:5s} {c / 4000:.2f}" for q, c in sorted(zip(u_, c_), key=lambda x: -x[1])]
        fig.update_layout(scene=scene(x_range=(-L / 2, L / 2), y_range=(-L / 2, L / 2), z_range=(-0.1 * L, 1.4 * L),
                                      aspect=(1.8, 1, 1), eye=(-1.2, -1.5, 0.6)))
        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        text_block(fig, lines, 0.64, 0.95, 0.06, size=6.2)
        return layout(fig, margin=dict(l=0, r=0, t=0, b=0))

    # ---------------------------------------------------------------- multiple scattering with tails (our scattering.py)
    p_list = (1.5, 3.0, 10.0, 30.0, 100.0)

    class Bare:
        planes = [dsg.Plane(0, "UBT", "ubt", 30.6, 0, 1.0, -1.35, 1.35, 1.0, 1.0, technology=None)]

    def msc_model(p_b, n, tails, seed=3):
        tr = {"ax": np.zeros(n), "ay": np.zeros(n), "az": np.full(n, 30.61), "bx": np.zeros(n), "by": np.zeros(n),
              "bz": np.full(n, 82.0), "kx": np.full(n, np.nan), "ky": np.full(n, np.nan), "kz": np.full(n, np.nan),
              "bpx": np.zeros(n), "bpy": np.zeros(n), "bpz": np.full(n, p_b)}
        k = scattering.kicks(tr, Bare, np.random.default_rng(seed), media=True, tails=tails)
        return scattering.scattered_tracks(tr, k), k

    pres.slider("sim_ms_p", label="p", min=0, max=4, value=1, step=1, decimals=0)
    pres.radio("sim_ms_q", ["position", "angle"], value="position", label="at z = 82 m", style="segmented")

    @pres.output("sim_msc_hist", inputs=["sim_ms_p", "sim_ms_q"])
    def sim_msc_hist(pi, q):
        p = p_list[int(pi)]
        gx, gt = G4MSC[f"x_{p}"], G4MSC[f"tx_{p}"]
        n = len(gx)
        out = {}
        for tails in (False, True):
            st, _ = msc_model(float(np.mean(G4MSC[f"p_{p}"])), n, tails)
            out[tails] = (st["bx"], st["bpx"] / st["bpz"])
        j, unit = (0, "|x| [mm]") if q == "position" else (1, "|θ<sub>x</sub>| [mrad]")
        ref = np.median(np.abs((gx, gt)[j])) * 1e3
        bins = np.linspace(0, 14 * ref, 70)
        fig = go.Figure()
        for v, col, lab in (((gx, gt)[j], INK, "Geant4"), (out[False][j], ACC, "Gaussian core"), (out[True][j], RED, "+ single-scattering tails")):
            h, _ = np.histogram(np.abs(v) * 1e3, bins)
            fig.add_trace(go.Scatter(x=0.5 * (bins[1:] + bins[:-1]), y=np.maximum(h, 0.5) / n, mode="lines",
                                     line=dict(color=col, shape="hvh", width=2), name=lab))
        fig.update_yaxes(type="log", title_text="fraction / bin", range=[np.log10(0.5 / n), 0])
        fig.update_xaxes(title_text=unit)
        fig.update_layout(title=dict(text=f"μ⁻ of {p:g} GeV: window, 50 m helium, exit wall, air to 82 m", font=dict(size=8), x=0.5))
        return layout(fig, showlegend=True, legend=dict(x=0.55, y=0.97, font=dict(size=6)))

    pres.slider("sim_ms_traj_p", label="p", min=0, max=4, value=0, step=1, decimals=0)

    @pres.output("sim_msc_traj", inputs=["sim_ms_traj_p"])
    def sim_msc_traj(pi):
        p = p_list[int(pi)]
        n = 400
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                            subplot_titles=("Gaussian kicks", "with single-scattering tails"))
        for col, tails in ((1, False), (2, True)):
            _, k = msc_model(p, n, tails, seed=11)
            z = np.linspace(30.6, 82, 60)
            xs = np.stack([scattering.offset_at(k, zz)[0] for zz in z], 1) * 1e3
            hard = k["n_hard"] > 0 if tails else np.zeros(n, bool)
            for i in range(60):
                fig.add_trace(go.Scatter(x=z, y=xs[i], mode="lines", line=dict(color=RED if hard[i] else ACC,
                                         width=1.6 if hard[i] else 0.7), opacity=0.9 if hard[i] else 0.45), 1, col)
        fig.update_xaxes(title_text="z [m]")
        fig.update_yaxes(title_text=f"x offset [mm], p = {p:g} GeV", row=1, col=1)
        fig.update_annotations(font_size=8)
        return layout(fig)

    # ---------------------------------------------------------------- closure against full Geant4
    @pres.output("sim_closure")
    def sim_closure():
        fig = make_subplots(rows=1, cols=2, subplot_titles=("helium", "PVC walls"), horizontal_spacing=0.12)
        for col, m in ((1, "helium"), (2, "pvc")):
            for key, lab, c in (("full_geant4", "full Geant4", INK), ("fast_geant4", "fast, Geant4 physics", ACC),
                                ("fast_crossover", "fast, crossover (ours)", RED), ("fast_w2", "fast, W = 2 split", ORA)):
                r = CLOSURE[key][m]
                fig.add_trace(go.Bar(x=["≥ 1 track", "≥ 2 tracks", "μ + ≥ 1"], y=[r[">=1_15y"], r[">=2_15y"], r["mu+>=1_15y"]],
                                     marker_color=c, name=lab, showlegend=col == 1), 1, col)
        fig.update_yaxes(title_text="interactions / 15 years", row=1, col=1)
        fig.update_layout(barmode="group")
        return layout(fig, showlegend=True, legend=dict(orientation="h", x=0, y=-0.18, font=dict(size=6)))

    _register_veto(pres, maps)


def _wall_angles():
    """Angle to the lateral wall of the sample muons that cross one (weighted histogram)."""
    r = {k[3:]: D[k].astype(np.float64) for k in D if k.startswith("mu_") and k != "mu_total_w"}
    p, q, idx = list(tg.paths(r))[0]
    dn = (q - p) / np.linalg.norm(q - p, axis=1, keepdims=True)
    sx = (geo.VESSEL_X_OUT - geo.VESSEL_X_IN) / (geo.Z_VESSEL_MAX - geo.Z_VESSEL_MIN)
    ang_x = np.degrees(np.abs(np.arcsin((np.abs(dn[:, 0]) - sx * dn[:, 2]) / np.sqrt(1 + sx ** 2))))
    from veto.tracks import vessel_crossings
    c = vessel_crossings(p[:, 0], p[:, 1], p[:, 2], (q - p)[:, 0], (q - p)[:, 1], (q - p)[:, 2])
    lat = np.isin(c["face_in"], [2, 3]) | np.isin(c["face_out"], [2, 3])
    return np.histogram(ang_x[lat], np.geomspace(0.05, 90, 40), weights=r["weight"][idx][lat])


# ====================================================================== our selection and veto on interactions
def _register_veto(pres, maps):
    prep = dict(np.load(IX / "prepared_muon.npz"))
    trk = {k[4:]: prep[k] for k in prep if k.startswith("trk_")}
    ev_of = prep["track_event"]
    is_beam = np.abs(trk["az"] - trk["kz"]) > 1e-6
    code_names = {0: "helium", 1: "air", 2: "PVC wall"}
    d = dsg.load("E3_robust")
    # show events where the scattered muon is reconstructed (its back-extrapolation is the interesting part)
    events = np.unique(ev_of[is_beam & (np.abs(trk["bx"]) < 2) & (np.abs(trk["by"]) < 3)])
    events = events[np.argsort(-prep["event_weight15"][events])][:200]
    pres.slider("iv_ev", label="event", min=0, max=len(events) - 1, value=0, step=1, decimals=0)
    pres.radio("iv_cone", ["0 (algorithm A)", "10 mrad", "20 mrad"], value="0 (algorithm A)", label="vertex cone", style="segmented")

    @pres.output("iv_event3d", inputs=["iv_ev", "iv_cone"])
    def iv_event3d(k, cone_label):
        e = int(events[int(k) % len(events)])
        rows = np.nonzero(ev_of == e)[0]
        sub = {kk: v[rows] for kk, v in trk.items()}
        cone = {"0 (algorithm A)": 0.0, "10 mrad": 0.01, "20 mrad": 0.02}[cone_label]
        cfg = reco.RecoConfig(seed=1, multiple_scattering=True, cone=cone)
        r = reco.run_events(sub, np.zeros(len(rows), np.int64), prep["track_z_b"][rows], d, maps, cfg,
                            np.random.default_rng(int(k)), details=True)
        hits, q, meas = r["hits"], r["queries"], r["meas"]
        fig = make_subplots(rows=1, cols=2, column_widths=[0.64, 0.36], specs=[[{"type": "scene"}, {"type": "xy"}]])
        for t in design_traces(d, opacity=0.25):
            fig.add_trace(t, 1, 1)
        vtx_z = prep["track_z_b"][rows][np.isfinite(prep["track_z_b"][rows])].max()
        lines = []
        for i in range(len(rows)):
            a, kk, b = ([sub[f"{c}x"][i], sub[f"{c}y"][i], sub[f"{c}z"][i]] for c in "akb")
            if is_beam[rows[i]]:
                fig.add_trace(lines3d([[a, kk]], INK, 3, opacity=0.8), 1, 1)          # the parent, up to the vertex
                fig.add_trace(lines3d([[kk, b]], RED, 4), 1, 1)                      # the scattered muon
                pin = np.array([sub["apx"][i], sub["apy"][i], sub["apz"][i]])
                pout = np.array([sub["bpx"][i], sub["bpy"][i], sub["bpz"][i]])
                kink = 1e3 * np.arccos(np.clip(pin @ pout / np.linalg.norm(pin) / np.linalg.norm(pout), -1, 1))
                lines.append(f"parent μ {np.linalg.norm(pin):.0f} GeV, bent by {kink:.1f} mrad at the vertex")
            else:
                fig.add_trace(lines3d([[kk, b]], ACC, 2.5), 1, 1)
            if np.isfinite(prep["track_z_b"][rows[i]]):                             # searched: backward line
                z0 = 30.0
                bb = np.array([meas["x"][i], meas["y"][i], meas["z"][i]])
                start = bb + np.array([meas["tx"][i], meas["ty"][i], 1.0]) * (z0 - bb[2])
                fig.add_trace(lines3d([[bb, start]], MUT, 1.2, dash="dot"), 1, 1)
        # search windows (circles) on the planes, and the hits
        th = np.linspace(0, 2 * np.pi, 40)
        for j in range(len(q["track"])):
            pl = d.planes[int(q["plane"][j])]
            x0, y0 = pl.to_global(q["u"][j]), q["y"][j]
            col = GRN if r["found"][j] else MUT
            fig.add_trace(lines3d([[[x0 + q["r"][j] * np.cos(t), y0 + q["r"][j] * np.sin(t), pl.z] for t in th]], col, 2), 1, 1)
        own = hits["track"] >= 0
        if own.any():
            fig.add_trace(pts3d([[d.planes[int(hits["plane"][h])].to_global(hits["u"][h]), hits["y"][h], d.planes[int(hits["plane"][h])].z]
                                 for h in np.nonzero(own)[0]], INK, 4, "diamond"), 1, 1)
        if (~own).any():
            fig.add_trace(pts3d([[d.planes[int(hits["plane"][h])].to_global(hits["u"][h]), hits["y"][h], d.planes[int(hits["plane"][h])].z]
                                 for h in np.nonzero(~own)[0]], "#999999", 2), 1, 1)
        fig.add_trace(pts3d([[sub["kx"][is_beam[rows]][0], sub["ky"][is_beam[rows]][0], sub["kz"][is_beam[rows]][0]]] if is_beam[rows].any()
                            else [[0, 0, vtx_z]], INK, 5, "x"), 1, 1)
        vetoed = bool(r["vetoed"][0])
        head = [f"muon interaction in the {code_names[int(prep['event_code'][e])]}, z = {prep['event_zint'][e]:.1f} m",
                f"weight {prep['event_weight15'][e]:.3g} per 15 years, {len(rows)} tracks",
                f"cone {cone_label}: {'VETOED' if vetoed else 'survives'}", ""]
        foot = ["", "dotted: backward lines from SST1", "circles: search windows (green: a hit inside)",
                "◆ the parent's own hits, · random hits"]
        fig.update_layout(scene=scene(z_range=(25, 86), x_range=(-6, 6), y_range=(-4, 5), aspect=(2.4, 1, 0.8)))
        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        text_block(fig, head + lines + foot, 0.65, 0.98, 0.055, size=6.0)
        return layout(fig, margin=dict(l=0, r=0, t=0, b=0))

    scan = json.loads((IX / "cone_scan_max2.json").read_text())

    @pres.output("iv_cone_scan")
    def iv_cone_scan():
        fig = go.Figure()
        for name, col in (("E3_robust", ACC), ("D6_tile2", ORA)):
            rows = [r for r in scan if r["design"] == name]
            fig.add_trace(go.Scatter(x=[100 * r["signal_lost"] for r in rows], y=[max(r["mu_after_b"], 1e-2) for r in rows],
                                     mode="lines+markers+text", text=[f"{r['cone_mrad']:g}" for r in rows], textposition="top right",
                                     textfont=dict(size=6, color=col), line=dict(color=col, width=2), marker=dict(size=6), name=name))
        fig.add_hline(y=0.55, line=dict(color=RED, dash="dash"))
        fig.add_annotation(x=0.98, y=np.log10(0.55), xref="paper", text="neutrino floor ≈ 0.5", showarrow=False, yshift=7,
                           xanchor="right", font=dict(size=6.5, color=RED))
        fig.update_xaxes(type="log", title_text="signal lost to random hits [%]", range=[-1.5, 1.75])
        fig.update_yaxes(type="log", title_text="μ interactions after veto / 15 y", range=[-0.7, 3.9])
        return layout(fig, showlegend=True, legend=dict(x=0.02, y=0.06, font=dict(size=6.5)), margin=dict(l=46, r=10, t=26, b=30))
