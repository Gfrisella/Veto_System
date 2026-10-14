#!/usr/bin/env python3
"""Interactive design explorer: UBT + side fins ("wings") with sliders, veto numbers recomputed live.

Design (what the RL environment will choose):
  UBT    plane at z = 30.6 m, |x| < 0.5 m (the vessel aperture), free bottom / top edge in y
  wings  thin vertical fins perpendicular to the beam (x-y plane at a fixed z) on the left and right
         of the vessel. A fin touches the vessel only with its inner edge, on the side wall at
         |x| = X(z), and reaches out in x by `reach`; it covers y from `bottom` to `top`. A fin at
         z = 30.6 m extends the UBT sideways. Left and right are independent and every fin has its own
         z (the RL environment will place them one after another with a minimum gap instead).
         Nothing on the top or bottom of the vessel.

A muon is vetoed with probability 1 - (1 - eps)^n_hits. Its position at any z comes from its path
A (z = 30.6) -> K (bounce point) -> B (z = 82 or stop point), see lib/veto/muon_types.py.
Detector rates come from x-y maps of every muon of the dataset outside the vessel, every 1 m in z.

  bin/remote.sh run bin/make_design_explorer.py       # on the cluster: rate maps + muon sample
  bin/remote.sh fetch-results design_explorer
  bin/make_design_explorer.py --html                  # locally (needs plotly): design_explorer.html
"""

import _bootstrap  # noqa: F401

import argparse
import json
import time

import numpy as np

from veto import geometry as geo
from veto import io, paths
from veto import muon_types as mt

DATASET = "design-optimization-dataset-alpha-14-oct-2026"
SAMPLE = {0: 5000, 1: 5000, 2: 5000, 3: 5000, 4: 3000, 5: 3000, 6: 3000}
N_DRAW = 8
Z_SLICES = np.arange(geo.Z_UBT, geo.Z_SST1 + 0.01, 1.0)     # fin rate maps, every 1 m
U_BINS = np.arange(0, 8.01, 0.2)                              # outward distance from the vessel wall
Y_BINS = np.arange(-8, 8.01, 0.2)
UBT_Y_BINS = np.arange(-8, 8.025, 0.05)
COLORS = {0: "#9400d3", 1: "#d62728", 2: "#ff9896", 3: "#1f77b4", 4: "#e377c2", 5: "#7f7f7f", 6: "#8c564b"}
LABELS = {0: "SST1, bounced off the cavern", 1: "SST1, p > 1 GeV & timing acc.", 2: "SST1, other",
          3: "veto only (misses SST1)", 4: "bounced / scattered in the rock", 5: "straight past the vessel",
          6: "stops between 30.6 and 82 m"}

STATE = ("x", "y", "z", "px", "py", "pz")


def stack(d, fmt):
    return np.stack([d[fmt.format(k)] for k in "xyz"], 1).astype(np.float64)


def collect(args):
    store, meta = io.open_store(paths.store_dir(DATASET))
    n = meta["n"]
    rng = np.random.default_rng(args.seed)
    types = np.asarray(store["type"])
    weight_all = np.asarray(store["weight"]).astype(np.float64)
    picks = {}
    for t, k in SAMPLE.items():
        idx = np.where(types == t)[0]
        picks[t] = np.sort(rng.choice(idx, min(k, len(idx)), replace=False))

    # rate maps from every muon: UBT (y) and, per side and z slice, (outward distance, y) outside the vessel
    nu, ny = len(U_BINS) - 1, len(Y_BINS) - 1
    fin_maps = np.zeros((2, len(Z_SLICES), nu * ny))
    ubt_hist = np.zeros(len(UBT_Y_BINS) - 1)
    cols = ["x_30", "y_30", "z_30", "pz_30", "x_82", "y_82", "z_82", "kx", "ky", "kz", "weight"]
    start = time.time()
    for a in range(0, n, args.chunk):
        d = {c: np.asarray(store[c][a:a + args.chunk]) for c in cols}
        w = d["weight"].astype(np.float64)
        on_ubt = (np.abs(d["x_30"]) < geo.UBT_HALF_X) & (d["pz_30"] > 0)
        ubt_hist += np.histogram(d["y_30"][on_ubt], UBT_Y_BINS, weights=w[on_ubt])[0]
        pa, pk, pb = stack(d, "{}_30"), np.stack([d["kx"], d["ky"], d["kz"]], 1), stack(d, "{}_82")
        for iz, z in enumerate(Z_SLICES):
            x, y, ok = mt.path_at_z(pa, pk, pb, z)
            u = np.abs(x) - mt.side_plane_x(z)
            ok &= (u >= 0) & (u < U_BINS[-1]) & (y >= Y_BINS[0]) & (y < Y_BINS[-1])
            iu = (u[ok] / (U_BINS[1] - U_BINS[0])).astype(int)
            iy = ((y[ok] - Y_BINS[0]) / (Y_BINS[1] - Y_BINS[0])).astype(int)
            side = (x[ok] < 0).astype(int)
            for sd in (0, 1):
                m = side == sd
                fin_maps[sd, iz] += np.bincount(iu[m] * ny + iy[m], weights=w[ok][m], minlength=nu * ny)
        print(f"  {min(a + args.chunk, n):,} / {n:,} muons, {time.time() - start:.0f} s", flush=True)

    # sample: per-type weights rescaled so each type keeps its total
    out = {"fin_maps": fin_maps.reshape(2, len(Z_SLICES), nu, ny), "ubt_hist": ubt_hist}
    rows = np.concatenate(list(picks.values()))
    d = {c: np.asarray(store[c][rows]) for c in cols + ["type", "px_30", "py_30", "px_82", "py_82", "pz_82"]}
    w = d["weight"].astype(np.float64)
    for t, idx in picks.items():
        m = d["type"] == t
        w[m] *= weight_all[types == t].sum() / w[m].sum()
    out.update({f"s_{c}": d[c] for c in cols + ["type"]})
    out.update(s_w=w)
    out["type_totals"] = np.array([weight_all[types == t].sum() for t in range(7)])
    outdir = paths.results_dir() / "design_explorer"
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(outdir / "sample.npz", **out)
    print(f"saved {outdir / 'sample.npz'}: {len(w):,} sampled muons")


def r(a, digits=3):
    return np.round(np.asarray(a, np.float64), digits).tolist()


def build_html(args):
    import plotly.graph_objects as go
    from veto import viz3d

    outdir = paths.results_dir() / "design_explorer"
    s = dict(np.load(outdir / "sample.npz"))
    n = len(s["s_w"])
    fin_maps = s["fin_maps"]                       # (side, z slice, u, y)
    data = {
        "type": s["s_type"].astype(int).tolist(),
        "w": r(s["s_w"], 6),
        # path points per muon: A, K (NaN if none), B
        "ax": r(s["s_x_30"]), "ay": r(s["s_y_30"]), "az": r(s["s_z_30"]),
        "kx": [None if not np.isfinite(v) else v for v in r(s["s_kx"])],
        "ky": [None if not np.isfinite(v) else v for v in r(s["s_ky"])],
        "kz": [None if not np.isfinite(v) else v for v in r(s["s_kz"])],
        "bx": r(s["s_x_82"]), "by": r(s["s_y_82"]), "bz": r(s["s_z_82"]),
        "ubt_on": ((np.abs(s["s_x_30"]) < geo.UBT_HALF_X) & (s["s_pz_30"] > 0)).astype(int).tolist(),
        "type_totals": r(s["type_totals"], 2),
        # fin rate maps: cumulative over u (outward distance) so a fin of reach L is one lookup per y bin
        "fin_cum": np.round(np.cumsum(fin_maps, axis=2), 1).tolist(),
        "z_slices": r(Z_SLICES), "u_bins": r(U_BINS), "y_bins": r(Y_BINS),
        "ubt_hist": r(s["ubt_hist"], 2), "ubt_y_bins": r(UBT_Y_BINS),
        "labels": LABELS, "colors": COLORS,
        "geo": {"z_ubt": geo.Z_UBT, "z_max": geo.Z_SST1, "ubt_half_x": geo.UBT_HALF_X,
                "x_in": geo.VESSEL_X_IN, "x_out": geo.VESSEL_X_OUT,
                "z_vmin": geo.Z_VESSEL_MIN, "z_vmax": geo.Z_VESSEL_MAX},
    }

    rng = np.random.default_rng(args.seed)
    drawn = []
    for t in range(7):
        idx = np.where(s["s_type"] == t)[0]
        drawn += sorted(rng.choice(idx, min(N_DRAW, len(idx)), replace=False).tolist())
    data["drawn"] = drawn

    fig = go.Figure([tr for tr in viz3d.geometry_traces() if tr.name != "UBT"])
    for t in range(7):
        idx = [i for i in drawn if s["s_type"][i] == t]
        zs, xs, ys = [], [], []
        for i in idx:
            pts = [(s["s_x_30"][i], s["s_y_30"][i], s["s_z_30"][i])]
            if np.isfinite(s["s_kz"][i]):
                pts.append((s["s_kx"][i], s["s_ky"][i], s["s_kz"][i]))
            pts.append((s["s_x_82"][i], s["s_y_82"][i], s["s_z_82"][i]))
            for x, y, z in pts:
                zs.append(float(z)); xs.append(float(x)); ys.append(float(y))
            zs.append(None); xs.append(None); ys.append(None)
        fig.add_trace(go.Scatter3d(x=zs, y=xs, z=ys, mode="lines", line=dict(color=COLORS[t], width=2.5),
                                   opacity=0.8, name=f"{LABELS[t]} ({len(idx)} drawn)", hoverinfo="skip"))
    n_static = len(fig.data)
    fig.update_layout(
        scene=dict(xaxis=dict(title="z [m]", range=[-3, 98]), yaxis=dict(title="x [m]", range=[-9, 9]),
                   zaxis=dict(title="y [m]", range=[-8, 8]), aspectmode="manual",
                   aspectratio=dict(x=4, y=0.72, z=0.65), camera=dict(eye=dict(x=-0.5, y=-2.6, z=1.3))),
        legend=dict(itemsizing="constant", font=dict(size=10), x=0, y=1), margin=dict(l=0, r=0, t=0, b=0),
        uirevision="keep")
    html = TEMPLATE.replace("__FIG__", fig.to_json()).replace("__DATA__", json.dumps(data)) \
                   .replace("__NSTATIC__", str(n_static))
    path = outdir / "design_explorer.html"
    path.write_text(html)
    print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB, {n:,} sampled muons)")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Veto Design Explorer</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
  :root { --bg:#fafafa; --panel:#fff; --fg:#222; --muted:#666; --line:#ddd; }
  body { margin:0; font:13px/1.35 system-ui, sans-serif; background:var(--bg); color:var(--fg); }
  #wrap { display:flex; height:100vh; }
  #side { width:410px; min-width:410px; overflow-y:auto; padding:10px 12px; background:var(--panel);
          border-right:1px solid var(--line); box-sizing:border-box; }
  #plot { flex:1; min-width:0; }
  h1 { font-size:16px; margin:2px 0 6px; }
  h2 { font-size:13px; margin:12px 0 4px; }
  details { border:1px solid var(--line); border-radius:6px; padding:4px 8px; margin:6px 0; }
  summary { cursor:pointer; font-weight:600; }
  .row { display:grid; grid-template-columns:86px 1fr 58px; gap:6px; align-items:center; margin:2px 0; }
  .row input { width:100%; }
  .val { text-align:right; font-variant-numeric:tabular-nums; }
  .zinfo { color:var(--muted); font-size:12px; margin:2px 0 6px; }
  table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
  td, th { padding:2px 4px; border-bottom:1px solid #eee; text-align:right; }
  td:first-child, th:first-child { text-align:left; }
  .sw { display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:4px; }
  .note { color:var(--muted); font-size:11.5px; }
  .big { font-size:15px; font-weight:600; }
  @media (max-width: 800px) { #wrap { flex-direction:column; height:auto; }
    #side { width:100%; min-width:0; } #plot { height:70vh; } }
</style></head>
<body><div id="wrap">
<div id="side">
  <h1>Veto design explorer</h1>
  <div class="note">UBT at z = 30.6 m (|x| &lt; 0.5 m) and fins ("wings") on the left and right: thin x–y
  planes that touch the side wall with their inner edge and reach out in x, each at its own z.
  Muons of <i>design-optimization-dataset-alpha-14-oct-2026</i>; rates in muons per spill.</div>

  <h2>Result</h2>
  <div id="summary" class="big"></div>
  <table id="vetotab"></table>
  <table id="dettab" style="margin-top:6px"></table>

  <h2>Global</h2>
  <div id="global"></div>
  <details open><summary>UBT</summary><div id="ubt"></div></details>
  <details open><summary>Right side (+x)</summary><div id="right"></div></details>
  <details open><summary>Left side (−x, the bounce side)</summary><div id="left"></div></details>
  <p class="note">Drawn: a few muons per type; big markers are their hits on the UBT / fins; thick black are
  drawn SST1 / veto-only muons with no hit. Veto numbers use a sample of ~29 k muons; detector rates use
  every muon of the dataset (maps every 1 m in z). Overlapping detectors count as separate layers.</p>
</div>
<div id="plot"></div>
</div>
<script>
const FIG = __FIG__;
const D = __DATA__;
const NSTATIC = __NSTATIC__;
const G = D.geo;
const X = z => G.x_in + (G.x_out - G.x_in) / (G.z_vmax - G.z_vmin) * (z - G.z_vmin);
const N_FINS = 4;
const S = {};

function slider(parent, key, label, min, max, step, value, unit) {
  const row = document.createElement('div'); row.className = 'row';
  row.innerHTML = `<label for="${key}">${label}</label><input type="range" id="${key}" min="${min}" max="${max}"
    step="${step}" value="${value}"><span class="val" id="${key}_v"></span>`;
  parent.appendChild(row);
  const inp = row.querySelector('input'), out = row.querySelector('.val');
  const set = () => { S[key] = parseFloat(inp.value); out.textContent = S[key].toFixed(step < 0.01 ? 3 : 2) + (unit || ''); };
  set();
  inp.addEventListener('input', () => { set(); schedule(); });
}

slider(document.getElementById('global'), 'eps', 'efficiency', 0.9, 1, 0.001, 0.99, '');
const ubtBox = document.getElementById('ubt');
slider(ubtBox, 'ubt_lo', 'bottom y', -8, 0, 0.05, -1.35, ' m');
slider(ubtBox, 'ubt_hi', 'top y', 0, 8, 0.05, 1.35, ' m');
const defaults = [[30.6, 3, -3, 3], [45.6, 3, -3, 3], [60.6, 3, -3, 3], [75.6, 3, -3, 3]];
for (const [side, el] of [['R', 'right'], ['L', 'left']]) {
  const box = document.getElementById(el);
  for (let i = 0; i < N_FINS; i++) {
    const h = document.createElement('div'); h.innerHTML = `<b>fin ${i + 1}</b>`; box.appendChild(h);
    const [z, reach, lo, hi] = defaults[i];
    slider(box, `${side}${i}_zpos`, 'z position', 30.6, 82, 0.1, z, ' m');
    slider(box, `${side}${i}_reach`, 'reach out (x)', 0, 7.5, 0.05, reach, ' m');
    slider(box, `${side}${i}_lo`, 'bottom y', -8, 0, 0.05, lo, ' m');
    slider(box, `${side}${i}_hi`, 'top y', 0, 8, 0.05, hi, ' m');
    const info = document.createElement('div'); info.className = 'zinfo'; info.id = `${side}${i}_z`; box.appendChild(info);
  }
}

function design() {
  const fins = {R: [], L: []};
  for (const side of ['R', 'L']) {
    for (let i = 0; i < N_FINS; i++) {
      const z = S[`${side}${i}_zpos`];
      const reach = S[`${side}${i}_reach`], lo = S[`${side}${i}_lo`], hi = S[`${side}${i}_hi`];
      const active = z <= G.z_max && reach > 0 && hi > lo;
      fins[side].push({z, reach, lo, hi, active});
      document.getElementById(`${side}${i}_z`).textContent = active ? `z = ${z.toFixed(1)} m, |x| = ${X(z).toFixed(2)} … ${(X(z) + reach).toFixed(2)} m, `
                   + `area ${(reach * (hi - lo)).toFixed(1)} m²` : 'empty';
    }
  }
  return {ubt: {lo: S.ubt_lo, hi: S.ubt_hi}, fins, eps: S.eps};
}

// position of muon i's path at plane z (A -> K -> B), null if the path does not reach z
function at(i, z) {
  if (z < D.az[i] && D.az[i] - z < 0.05) z = D.az[i];   // A is stored a few mm after z = 30.6
  const segs = D.kz[i] === null ? [[D.ax[i], D.ay[i], D.az[i], D.bx[i], D.by[i], D.bz[i]]]
    : [[D.ax[i], D.ay[i], D.az[i], D.kx[i], D.ky[i], D.kz[i]], [D.kx[i], D.ky[i], D.kz[i], D.bx[i], D.by[i], D.bz[i]]];
  for (const [x0, y0, z0, x1, y1, z1] of segs) {
    if (z1 > z0 && z >= z0 && z <= z1) { const t = (z - z0) / (z1 - z0); return [x0 + t * (x1 - x0), y0 + t * (y1 - y0)]; }
  }
  return null;
}

function finHit(f, sign, p, z) {
  if (!p) return false;
  const u = sign * p[0] - X(z);
  return u >= 0 && u <= f.reach && p[1] >= f.lo && p[1] <= f.hi;
}
const ubtHit = (des, i) => D.ubt_on[i] && D.ay[i] >= des.ubt.lo && D.ay[i] <= des.ubt.hi;

// hit points of muon i: list of [x, y, z]
function hitPoints(des, i) {
  const pts = [];
  if (ubtHit(des, i)) pts.push([D.ax[i], D.ay[i], G.z_ubt]);
  for (const [side, sign] of [['R', 1], ['L', -1]]) for (const f of des.fins[side]) {
    if (!f.active) continue;
    const p = at(i, f.z);
    if (finHit(f, sign, p, f.z)) pts.push([p[0], p[1], f.z]);
  }
  return pts;
}

const fmt = v => v === 0 ? '0' : (Math.abs(v) >= 1e4 || Math.abs(v) < 0.01 ? v.toExponential(2) : v.toPrecision(3));

function finRate(f, sideIdx) {
  // nearest 1 m slice; cumulative over u, sum over y bins inside [lo, hi]
  const iz = Math.max(0, Math.min(D.z_slices.length - 1, Math.round(f.z - D.z_slices[0])));
  const du = D.u_bins[1] - D.u_bins[0];
  const iu = Math.min(D.u_bins.length - 2, Math.floor(f.reach / du + 1e-9) - 1);
  if (iu < 0) return 0;
  const m = D.fin_cum[sideIdx][iz][iu];
  let s = 0;
  for (let b = 0; b < D.y_bins.length - 1; b++) {
    const yc = (D.y_bins[b] + D.y_bins[b + 1]) / 2; if (yc >= f.lo && yc <= f.hi) s += m[b];
  }
  return s;
}

function tables(des, hits) {
  const W = new Array(7).fill(0), V = new Array(7).fill(0);
  for (let i = 0; i < hits.length; i++) {
    const t = D.type[i], p = 1 - Math.pow(1 - des.eps, hits[i]);
    W[t] += D.w[i]; V[t] += D.w[i] * p;
  }
  let html = '<tr><th>type</th><th>muons/spill</th><th>vetoed</th><th>missed /spill</th></tr>';
  for (const t of [1, 2, 0, 3]) {
    html += `<tr><td><span class="sw" style="background:${D.colors[t]}"></span>${D.labels[t]}</td>
      <td>${fmt(W[t])}</td><td>${(100 * V[t] / W[t]).toFixed(2)}%</td><td>${fmt(W[t] - V[t])}</td></tr>`;
  }
  document.getElementById('vetotab').innerHTML = html;
  const miss = [0, 1, 2].reduce((a, t) => a + W[t] - V[t], 0), all = [0, 1, 2].reduce((a, t) => a + W[t], 0);
  document.getElementById('summary').innerHTML =
    `SST1 muons missed: ${fmt(miss)} / spill (${(100 * miss / all).toFixed(2)}%)` +
    `<br><span class="note">selected (p &gt; 1 GeV, timing): ${fmt(W[1] - V[1])} / spill missed</span>`;

  let ubtRate = 0;
  for (let b = 0; b < D.ubt_y_bins.length - 1; b++) {
    const yc = (D.ubt_y_bins[b] + D.ubt_y_bins[b + 1]) / 2;
    if (yc >= des.ubt.lo && yc <= des.ubt.hi) ubtRate += D.ubt_hist[b];
  }
  const ubtArea = 2 * G.ubt_half_x * (des.ubt.hi - des.ubt.lo);
  let rows = '<tr><th>detector</th><th>area m²</th><th>muons/spill</th><th>per m²</th></tr>';
  rows += `<tr><td>UBT</td><td>${ubtArea.toFixed(1)}</td><td>${fmt(ubtRate)}</td><td>${fmt(ubtRate / Math.max(ubtArea, 1e-9))}</td></tr>`;
  let totArea = ubtArea;
  for (const [side, k, name] of [['R', 0, 'right fins'], ['L', 1, 'left fins']]) {
    let area = 0, rate = 0;
    for (const f of des.fins[side]) if (f.active) { area += f.reach * (f.hi - f.lo); rate += finRate(f, k); }
    totArea += area;
    rows += `<tr><td>${name}</td><td>${area.toFixed(1)}</td><td>${fmt(rate)}</td><td>${fmt(rate / Math.max(area, 1e-9))}</td></tr>`;
  }
  rows += `<tr><td><b>total</b></td><td><b>${totArea.toFixed(1)}</b></td><td></td><td></td></tr>`;
  document.getElementById('dettab').innerHTML = rows;
}

function hitCounts(des) {
  const n = D.w.length, hits = new Int32Array(n);
  const fins = [];
  for (const [side, sign] of [['R', 1], ['L', -1]]) for (const f of des.fins[side]) if (f.active) fins.push([f, sign]);
  for (let i = 0; i < n; i++) {
    let h = ubtHit(des, i) ? 1 : 0;
    for (const [f, sign] of fins) if (finHit(f, sign, at(i, f.z), f.z)) h++;
    hits[i] = h;
  }
  return hits;
}

function rect(zs, xs, ys, color, name, show) {
  return {type: 'mesh3d', x: zs, y: xs, z: ys, i: [0, 0], j: [1, 2], k: [2, 3], color, opacity: 0.5,
          name, showlegend: show, legendgroup: name, hoverinfo: 'name', flatshading: true};
}

function geometryTraces(des) {
  const t = [rect([G.z_ubt, G.z_ubt, G.z_ubt, G.z_ubt], [-G.ubt_half_x, G.ubt_half_x, G.ubt_half_x, -G.ubt_half_x],
                  [des.ubt.lo, des.ubt.lo, des.ubt.hi, des.ubt.hi], '#ff7f0e', 'UBT', true)];
  for (const [side, sign] of [['R', 1], ['L', -1]]) des.fins[side].forEach((f, i) => {
    const z = f.active ? f.z : G.z_ubt, x0 = sign * X(z), x1 = sign * (X(z) + (f.active ? f.reach : 0));
    const lo = f.active ? f.lo : 0, hi = f.active ? f.hi : 0;
    t.push(rect([z, z, z, z], [x0, x1, x1, x0], [lo, lo, hi, hi], sign > 0 ? '#2ca02c' : '#17becf',
                sign > 0 ? 'fins, right (+x)' : 'fins, left (−x)', i === 0));
  });
  return t;
}

function muonTraces(des, hits) {
  const hz = [], hx = [], hy = [], hc = [], mz = [], mx = [], my = [];
  for (const i of D.drawn) {
    for (const [x, y, z] of hitPoints(des, i)) { hz.push(z); hx.push(x); hy.push(y); hc.push(D.colors[D.type[i]]); }
    if (D.type[i] <= 3 && hits[i] === 0) {
      const pts = [[D.ax[i], D.ay[i], D.az[i]]];
      if (D.kz[i] !== null) pts.push([D.kx[i], D.ky[i], D.kz[i]]);
      pts.push([D.bx[i], D.by[i], D.bz[i]]);
      for (const [x, y, z] of pts) { mz.push(z); mx.push(x); my.push(y); }
      mz.push(null); mx.push(null); my.push(null);
    }
  }
  return [
    {type: 'scatter3d', mode: 'markers', x: hz, y: hx, z: hy, marker: {size: 4.5, color: hc, line: {color: '#000', width: 1}},
     name: 'hits of drawn muons', hoverinfo: 'skip'},
    {type: 'scatter3d', mode: 'lines', x: mz, y: mx, z: my, line: {color: '#000', width: 6},
     name: 'drawn SST1 / veto-only muons with no hit', hoverinfo: 'skip'},
  ];
}

let pending = false;
function schedule() { if (!pending) { pending = true; requestAnimationFrame(update); } }
let first = true;
function update() {
  pending = false;
  const des = design(), hits = hitCounts(des);
  tables(des, hits);
  const dyn = geometryTraces(des).concat(muonTraces(des, hits));
  const div = document.getElementById('plot');
  if (first) { Plotly.newPlot(div, FIG.data.concat(dyn), FIG.layout, {responsive: true, displaylogo: false}); first = false; }
  else Plotly.react(div, FIG.data.concat(dyn), div.layout);
}
update();
</script></body></html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--html", action="store_true", help="build the HTML from results/design_explorer/sample.npz")
    parser.add_argument("--chunk", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.html:
        build_html(args)
    else:
        collect(args)


if __name__ == "__main__":
    main()
