"""Interactive 3D detector + muon view (plotly).

Plot coordinates are (z, x, y) so the beam runs left to right; the scene is
stretched along z (the detector is ~100 m long and ~10 m wide).
"""

import numpy as np
import plotly.graph_objects as go

from . import geometry as geo
from .tracks import extrapolate_to_z

# 12 triangles of a hexahedron given its 8 corners (4 upstream, then 4 downstream, same order)
_BOX_I = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
_BOX_J = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4]
_BOX_K = [2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7]


def frustum(z0, z1, hx0, hy0, hx1, hy1, name, color, opacity=0.15, showlegend=True, legendgroup=None):
    xs = [-hx0, hx0, hx0, -hx0, -hx1, hx1, hx1, -hx1]
    ys = [-hy0, -hy0, hy0, hy0, -hy1, -hy1, hy1, hy1]
    zs = [z0] * 4 + [z1] * 4
    return go.Mesh3d(x=zs, y=xs, z=ys, i=_BOX_I, j=_BOX_J, k=_BOX_K, color=color, opacity=opacity,
                     name=name, showlegend=showlegend, legendgroup=legendgroup or name,
                     hoverinfo="name", flatshading=True)


def frustum_edges(z0, z1, hx0, hy0, hx1, hy1, name, color, legendgroup=None, dash=None):
    """Wire-frame of a frustum (12 edges) as one line trace."""
    c0 = [(-hx0, -hy0), (hx0, -hy0), (hx0, hy0), (-hx0, hy0)]
    c1 = [(-hx1, -hy1), (hx1, -hy1), (hx1, hy1), (-hx1, hy1)]
    xs, ys, zs = [], [], []
    for corners, z in ((c0, z0), (c1, z1)):
        for x, y in corners + corners[:1]:
            xs.append(x); ys.append(y); zs.append(z)
        xs.append(None); ys.append(None); zs.append(None)
    for (x0, y0), (x1, y1) in zip(c0, c1):
        xs += [x0, x1, None]; ys += [y0, y1, None]; zs += [z0, z1, None]
    return go.Scatter3d(x=zs, y=xs, z=ys, mode="lines", line=dict(color=color, width=3, dash=dash),
                        name=name, showlegend=False, legendgroup=legendgroup or name, hoverinfo="skip")


def plane(z, hx, hy, name, color, opacity=0.25):
    return frustum(z - 0.05, z + 0.05, hx, hy, hx, hy, name, color, opacity)


def geometry_traces():
    traces = []
    for k, m in enumerate(geo.shield_magnets()):
        traces.append(frustum(m["z_start"], m["z_end"], m["half_x_in"], m["half_y_in"],
                              m["half_x_out"], m["half_y_out"], "Muon shield (TRY6)", "gray",
                              opacity=0.18, showlegend=k == 0, legendgroup="shield"))
    traces.append(frustum(geo.Z_TARGET, geo.Z_START, 0.15, 0.15, 0.15, 0.15, "Target", "black", 0.6))

    vessel = (geo.Z_VESSEL_MIN, geo.Z_VESSEL_MAX, geo.VESSEL_X_IN, geo.VESSEL_Y_IN,
              geo.VESSEL_X_OUT, geo.VESSEL_Y_OUT)
    traces.append(frustum(*vessel, "Decay vessel / SBT walls", "royalblue", 0.10, legendgroup="vessel"))
    traces.append(frustum_edges(*vessel, "Decay vessel / SBT walls", "royalblue", legendgroup="vessel"))

    traces.append(plane(geo.Z_UBT, geo.UBT_HALF_X, geo.UBT_HALF_Y, "UBT", "orange", 0.5))
    traces.append(plane(geo.Z_SST1, geo.SST1_HALF_X, geo.SST1_HALF_Y, "SST1 (z=82 m)", "red", 0.25))
    traces.append(plane(geo.Z_TIMING, geo.TIMING_HALF_X, geo.TIMING_HALF_Y, "Timing detector", "purple", 0.25))
    z0, z1 = geo.SPECTROMETER_Z
    traces.append(frustum(z0, z1, geo.SPECTROMETER_HALF_X, geo.SPECTROMETER_HALF_Y,
                          geo.SPECTROMETER_HALF_X, geo.SPECTROMETER_HALF_Y,
                          "Spectrometer magnet (|B|>10%)", "green", 0.12))
    return traces


def _polylines(points_per_track):
    """List of (n_points, 3) arrays in (x, y, z) -> concatenated plotly coordinates with None breaks."""
    xs, ys, zs = [], [], []
    for pts in points_per_track:
        for x, y, z in pts:
            xs.append(x); ys.append(y); zs.append(z)
        xs.append(None); ys.append(None); zs.append(None)
    return zs, xs, ys


def track_traces(tracks, groups, colors):
    """Straight muon tracks between z_start and z_end, coloured by group.

    tracks: x, y, z, px, py, pz (reference state), z_start, z_end, z_vessel/x_vessel/y_vessel
    (vessel entry, NaN if none). If x0..z2 (all output planes) are present, the start -> UBT
    -> SST1 true crossing points are added as a hidden dotted trace: these are chords, the
    real path bends in the shield field and may scatter on the walls.
    """
    traces = []
    has_planes = "x0" in tracks
    for name, mask in groups.items():
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        color = colors[name]

        segments = []
        for k in idx:
            ends = []
            for z_end in (tracks["z_start"][k], tracks["z_end"][k]):
                x_e, y_e = extrapolate_to_z(tracks["x"][k], tracks["y"][k], tracks["z"][k],
                                            tracks["px"][k], tracks["py"][k], tracks["pz"][k], z_end)
                ends.append((x_e, y_e, z_end))
            segments.append(ends)
        z, x, y = _polylines(segments)
        traces.append(go.Scatter3d(x=z, y=x, z=y, mode="lines", line=dict(color=color, width=2),
                                   opacity=0.7, name=f"{name} ({len(idx)})", legendgroup=name,
                                   hoverinfo="skip"))

        if has_planes:
            true_path = [[(tracks[f"x{p}"][k], tracks[f"y{p}"][k], tracks[f"z{p}"][k]) for p in (0, 1, 2)]
                         for k in idx]
            z, x, y = _polylines(true_path)
            traces.append(go.Scatter3d(x=z, y=x, z=y, mode="lines", line=dict(color=color, width=1, dash="dot"),
                                       opacity=0.35, name=f"{name}: true plane crossings", legendgroup=name,
                                       showlegend=False, hoverinfo="skip", visible="legendonly"))

        entries = np.isfinite(tracks["z_vessel"][idx])
        traces.append(go.Scatter3d(
            x=tracks["z_vessel"][idx][entries], y=tracks["x_vessel"][idx][entries],
            z=tracks["y_vessel"][idx][entries], mode="markers",
            marker=dict(size=2.5, color=color), name=f"{name}: vessel entry", legendgroup=name,
            showlegend=False,
            hovertemplate="vessel entry<br>z=%{x:.2f} m<br>x=%{y:.2f} m<br>y=%{z:.2f} m<extra></extra>",
        ))
    return traces


def detector_figure(tracks=None, groups=None, colors=None, title="SHiP muon background: shield, decay vessel and veto"):
    fig = go.Figure(geometry_traces())
    if tracks is not None:
        fig.add_traces(track_traces(tracks, groups, colors))
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="z [m]", range=[-3, 98]),
            yaxis=dict(title="x [m]", range=[-6, 6]),
            zaxis=dict(title="y [m]", range=[-5, 5]),
            aspectmode="manual", aspectratio=dict(x=4, y=0.6, z=0.6),
            camera=dict(eye=dict(x=-0.4, y=-3.4, z=1.6)),
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig
