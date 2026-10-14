#!/usr/bin/env python3
"""Interactive design explorer: UBT + side wings with sliders, veto numbers recomputed live.

Design (what the RL environment will choose):
  UBT    plane at z = 30.6 m, |x| < 0.5 m (the vessel aperture), free bottom / top edge in y
  wings  rectangles on the two side-wall planes x = +-X(z) (the vessel's lateral walls, continued
         above / below the vessel and up to z = 82 m), left and right independent. On each side
         they are placed one after another along z: the first starts `gap` after z = 30.6, every
         next one at least `min gap` after the previous end. Each has a length and a y range.
         Nothing on the top or bottom of the vessel. Overlapping detectors count as two layers.

A muon is vetoed with probability 1 - (1 - eps)^n_hits. Hits come from the muon's precomputed
crossings with the UBT plane and the side planes (path A -> K -> B, see lib/veto/muon_types.py),
so the page needs no physics, only rectangle tests.

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
Z_BINS = np.arange(geo.Z_UBT, geo.Z_SST1 + 0.25, 0.5)
Y_BINS = np.arange(-8, 8.05, 0.1)
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

    # rate maps from every muon
    side_hist = np.zeros((2, len(Z_BINS) - 1, len(Y_BINS) - 1))
    ubt_hist = np.zeros(len(UBT_Y_BINS) - 1)
    cols = ["x_30", "y_30", "z_30", "pz_30", "x_82", "y_82", "z_82", "kx", "ky", "kz", "weight"]
    start = time.time()
    for a in range(0, n, args.chunk):
        d = {c: np.asarray(store[c][a:a + args.chunk]) for c in cols}
        w = d["weight"].astype(np.float64)
        on_ubt = (np.abs(d["x_30"]) < geo.UBT_HALF_X) & (d["pz_30"] > 0)
        ubt_hist += np.histogram(d["y_30"][on_ubt], UBT_Y_BINS, weights=w[on_ubt])[0]
        cz, cy, cs = mt.side_plane_crossings(stack(d, "{}_30"), np.stack([d["kx"], d["ky"], d["kz"]], 1),
                                             stack(d, "{}_82"))
        ww = np.repeat(w[:, None], 4, 1)
        for i, side in enumerate((1, -1)):
            m = cs == side
            side_hist[i] += np.histogram2d(cz[m], cy[m], [Z_BINS, Y_BINS], weights=ww[m])[0]
        print(f"  {min(a + args.chunk, n):,} / {n:,} muons, {time.time() - start:.0f} s", flush=True)

    # sample: per-type weights rescaled so each type keeps its total
    out = {"side_hist": side_hist, "ubt_hist": ubt_hist}
    rows = np.concatenate(list(picks.values()))
    d = {c: np.asarray(store[c][rows]) for c in cols + ["type", "px_30", "py_30", "px_82", "py_82", "pz_82"]}
    w = d["weight"].astype(np.float64)
    for t, idx in picks.items():
        m = d["type"] == t
        w[m] *= weight_all[types == t].sum() / w[m].sum()
    cz, cy, cs = mt.side_plane_crossings(stack(d, "{}_30"), np.stack([d["kx"], d["ky"], d["kz"]], 1),
                                         stack(d, "{}_82"))
    out.update({f"s_{c}": d[c] for c in cols + ["type"]})
    out.update(s_w=w, s_cz=cz, s_cy=cy, s_cs=cs)
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
    has = np.isfinite(s["s_cz"])
    mu, col = np.nonzero(has)
    data = {
        "type": s["s_type"].astype(int).tolist(),
        "w": r(s["s_w"], 6),
        "ubt_on": ((np.abs(s["s_x_30"]) < geo.UBT_HALF_X) & (s["s_pz_30"] > 0)).astype(int).tolist(),
        "ubt_y": r(s["s_y_30"]),
        "ubt_x": r(s["s_x_30"]),
        "c_mu": mu.tolist(), "c_z": r(s["s_cz"][mu, col]), "c_y": r(s["s_cy"][mu, col]),
        "c_side": s["s_cs"][mu, col].astype(int).tolist(),
        "type_totals": r(s["type_totals"], 2),
        "side_hist": [np.round(h, 2).tolist() for h in s["side_hist"]],
        "ubt_hist": r(s["ubt_hist"], 2),
        "z_bins": r(Z_BINS), "y_bins": r(Y_BINS), "ubt_y_bins": r(UBT_Y_BINS),
        "labels": LABELS, "colors": COLORS,
        "geo": {"z_ubt": geo.Z_UBT, "z_max": geo.Z_SST1, "ubt_half_x": geo.UBT_HALF_X,
                "x_in": geo.VESSEL_X_IN, "x_out": geo.VESSEL_X_OUT,
                "z_vmin": geo.Z_VESSEL_MIN, "z_vmax": geo.Z_VESSEL_MAX},
    }

    # drawn muons: a few per type, path A -> K -> B
    rng = np.random.default_rng(args.seed)
    drawn = []
    for t in range(7):
        idx = np.where(s["s_type"] == t)[0]
        drawn += sorted(rng.choice(idx, min(N_DRAW, len(idx)), replace=False).tolist())
    data["drawn"] = drawn
    paths_ = {}
    for i in drawn:
        pts = [(s["s_x_30"][i], s["s_y_30"][i], s["s_z_30"][i])]
        if np.isfinite(s["s_kz"][i]):
            pts.append((s["s_kx"][i], s["s_ky"][i], s["s_kz"][i]))
        pts.append((s["s_x_82"][i], s["s_y_82"][i], s["s_z_82"][i]))
        paths_[int(i)] = [r(pt) for pt in pts]
    data["paths"] = paths_

    fig = go.Figure([tr for tr in viz3d.geometry_traces() if tr.name != "UBT"])
    # the two side planes, where wings may go
    for side in (1, -1):
        z = [geo.Z_UBT, geo.Z_SST1, geo.Z_SST1, geo.Z_UBT, geo.Z_UBT]
        x = [side * mt.side_plane_x(v) for v in z]
        for y0, y1 in ((-8, 8),):
            fig.add_trace(go.Scatter3d(x=z, y=x, z=[y0, y0, y1, y1, y0], mode="lines",
                                       line=dict(color="#2ca02c", width=2, dash="dot"),
                                       name="side-wall planes (where wings go)", legendgroup="planes",
                                       showlegend=side == 1, hoverinfo="skip"))
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
        scene=dict(xaxis=dict(title="z [m]", range=[-3, 98]), yaxis=dict(title="x [m]", range=[-8, 8]),
                   zaxis=dict(title="y [m]", range=[-8, 8]), aspectmode="manual",
                   aspectratio=dict(x=4, y=0.65, z=0.65), camera=dict(eye=dict(x=-0.5, y=-2.6, z=1.3))),
        legend=dict(itemsizing="constant", font=dict(size=10), x=0, y=1), margin=dict(l=0, r=0, t=0, b=0),
        uirevision="keep")
    fig_json = fig.to_json()

    html = TEMPLATE.replace("__FIG__", fig_json).replace("__DATA__", json.dumps(data)) \
                   .replace("__NSTATIC__", str(n_static))
    path = outdir / "design_explorer.html"
    path.write_text(html)
    print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB, {n:,} sampled muons, {len(mu):,} plane crossings)")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Veto Design Explorer</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
  :root { --bg:#fafafa; --panel:#fff; --fg:#222; --muted:#666; --line:#ddd; --accent:#1f77b4; }
  body { margin:0; font:13px/1.35 system-ui, sans-serif; background:var(--bg); color:var(--fg); }
  #wrap { display:flex; height:100vh; }
  #side { width:400px; min-width:400px; overflow-y:auto; padding:10px 12px; background:var(--panel);
          border-right:1px solid var(--line); box-sizing:border-box; }
  #plot { flex:1; min-width:0; }
  h1 { font-size:16px; margin:2px 0 6px; }
  h2 { font-size:13px; margin:12px 0 4px; }
  details { border:1px solid var(--line); border-radius:6px; padding:4px 8px; margin:6px 0; }
  summary { cursor:pointer; font-weight:600; }
  .row { display:grid; grid-template-columns:78px 1fr 58px; gap:6px; align-items:center; margin:2px 0; }
  .row input { width:100%; }
  .val { text-align:right; font-variant-numeric:tabular-nums; }
  .zinfo { color:var(--muted); font-size:12px; margin:2px 0 4px; }
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
  <div class="note">UBT at z = 30.6 m (|x| &lt; 0.5 m) and wings on the side-wall planes, left and right.
  Wings are placed one after another along z. Hits from precomputed crossings of the
  <i>design-optimization-dataset-alpha-14-oct-2026</i> muons; rates in muons per spill.</div>

  <h2>Result</h2>
  <div id="summary" class="big"></div>
  <table id="vetotab"></table>
  <table id="dettab" style="margin-top:6px"></table>

  <h2>Global</h2>
  <div id="global"></div>
  <details open><summary>UBT</summary><div id="ubt"></div></details>
  <details open><summary>Right side (+x)</summary><div id="right"></div></details>
  <details><summary>Left side (−x, the bounce side)</summary><div id="left"></div></details>
  <p class="note">Muons drawn: a few per type. Big markers: their hits on the UBT / wings. Thick black:
  drawn SST1 / veto-only muons with no hit. Veto numbers use the full sample (~29 k muons);
  detector rates use every muon of the dataset. Overlapping detectors count as two layers.</p>
</div>
<div id="plot"></div>
</div>
<script>
const FIG = __FIG__;
const D = __DATA__;
const NSTATIC = __NSTATIC__;
const G = D.geo;
const X = z => G.x_in + (G.x_out - G.x_in) / (G.z_vmax - G.z_vmin) * (z - G.z_vmin);
const N_WINGS = 3;
const S = {};   // slider values

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

const glob = document.getElementById('global');
slider(glob, 'mingap', 'min gap', 0, 5, 0.1, 0.5, ' m');
slider(glob, 'eps', 'efficiency', 0.9, 1, 0.001, 0.99, '');
const ubt = document.getElementById('ubt');
slider(ubt, 'ubt_ylo', 'bottom y', -8, 0, 0.05, -1.35, ' m');
slider(ubt, 'ubt_yhi', 'top y', 0, 8, 0.05, 1.35, ' m');
const defaults = [[0, 17, -3, 3], [0.5, 17, -3, 3], [0.5, 17, -3, 3]];
for (const [side, el] of [['R', 'right'], ['L', 'left']]) {
  const box = document.getElementById(el);
  for (let i = 0; i < N_WINGS; i++) {
    const h = document.createElement('div'); h.innerHTML = `<b>wing ${i + 1}</b>`; box.appendChild(h);
    const [g, l, lo, hi] = defaults[i];
    slider(box, `${side}${i}_gap`, i ? 'gap after prev' : 'start offset', 0, 20, 0.1, g, ' m');
    slider(box, `${side}${i}_len`, 'length', 0, 51.4, 0.1, l, ' m');
    slider(box, `${side}${i}_lo`, 'bottom y', -8, 0, 0.05, lo, ' m');
    slider(box, `${side}${i}_hi`, 'top y', 0, 8, 0.05, hi, ' m');
    const z = document.createElement('div'); z.className = 'zinfo'; z.id = `${side}${i}_z`; box.appendChild(z);
  }
}

function design() {
  const wings = {R: [], L: []};
  for (const side of ['R', 'L']) {
    let end = G.z_ubt;
    for (let i = 0; i < N_WINGS; i++) {
      const gap = i ? Math.max(S[`${side}${i}_gap`], S.mingap) : S[`${side}${i}_gap`];
      const z0 = end + gap, z1 = Math.min(z0 + S[`${side}${i}_len`], G.z_max);
      const active = z1 > z0 && S[`${side}${i}_hi`] > S[`${side}${i}_lo`];
      wings[side].push({z0, z1, lo: S[`${side}${i}_lo`], hi: S[`${side}${i}_hi`], active});
      if (z1 > z0) end = z1; else end = Math.max(end, Math.min(z0, G.z_max));
      document.getElementById(`${side}${i}_z`).textContent = active
        ? `z = ${z0.toFixed(1)} … ${z1.toFixed(1)} m, area ${((z1 - z0) * (S[`${side}${i}_hi`] - S[`${side}${i}_lo`])).toFixed(1)} m²`
        : (z0 >= G.z_max ? 'past z = 82 m: not placed' : 'empty');
    }
  }
  return {ubt: {lo: S.ubt_ylo, hi: S.ubt_yhi}, wings, eps: S.eps};
}

// per-muon number of detector layers hit
function hitCounts(des) {
  const n = D.w.length, hits = new Int32Array(n);
  for (let i = 0; i < n; i++)
    if (D.ubt_on[i] && D.ubt_y[i] >= des.ubt.lo && D.ubt_y[i] <= des.ubt.hi) hits[i]++;
  for (let j = 0; j < D.c_mu.length; j++) {
    const ws = des.wings[D.c_side[j] > 0 ? 'R' : 'L'], z = D.c_z[j], y = D.c_y[j];
    for (const w of ws) if (w.active && z >= w.z0 && z <= w.z1 && y >= w.lo && y <= w.hi) hits[D.c_mu[j]]++;
  }
  return hits;
}

function sumHist2(h, z0, z1, lo, hi) {
  let s = 0;
  for (let a = 0; a < D.z_bins.length - 1; a++) {
    const zc = (D.z_bins[a] + D.z_bins[a + 1]) / 2; if (zc < z0 || zc > z1) continue;
    for (let b = 0; b < D.y_bins.length - 1; b++) {
      const yc = (D.y_bins[b] + D.y_bins[b + 1]) / 2; if (yc >= lo && yc <= hi) s += h[a][b];
    }
  }
  return s;
}

const fmt = v => v === 0 ? '0' : (Math.abs(v) >= 1e4 || Math.abs(v) < 0.01 ? v.toExponential(2) : v.toPrecision(3));

function tables(des, hits) {
  const W = new Array(7).fill(0), V = new Array(7).fill(0), N = new Array(7).fill(0);
  for (let i = 0; i < hits.length; i++) {
    const t = D.type[i], p = 1 - Math.pow(1 - des.eps, hits[i]);
    W[t] += D.w[i]; V[t] += D.w[i] * p; N[t]++;
  }
  let html = '<tr><th>type</th><th>muons/spill</th><th>vetoed</th><th>missed /spill</th></tr>';
  for (const t of [1, 2, 0, 3]) {
    const miss = W[t] - V[t];
    html += `<tr><td><span class="sw" style="background:${D.colors[t]}"></span>${D.labels[t]}</td>
      <td>${fmt(W[t])}</td><td>${(100 * V[t] / W[t]).toFixed(2)}%</td><td>${fmt(miss)}</td></tr>`;
  }
  document.getElementById('vetotab').innerHTML = html;
  const sstMiss = [0, 1, 2].reduce((a, t) => a + W[t] - V[t], 0);
  const sstAll = [0, 1, 2].reduce((a, t) => a + W[t], 0);
  document.getElementById('summary').innerHTML =
    `SST1 muons missed: ${fmt(sstMiss)} / spill (${(100 * sstMiss / sstAll).toFixed(2)}%)` +
    `<br><span class="note">selected (p &gt; 1 GeV, timing): ${fmt(W[1] - V[1])} / spill missed</span>`;

  // detector rates from the full-statistics maps
  let ubtRate = 0;
  for (let b = 0; b < D.ubt_y_bins.length - 1; b++) {
    const yc = (D.ubt_y_bins[b] + D.ubt_y_bins[b + 1]) / 2;
    if (yc >= des.ubt.lo && yc <= des.ubt.hi) ubtRate += D.ubt_hist[b];
  }
  const ubtArea = 2 * G.ubt_half_x * (des.ubt.hi - des.ubt.lo);
  let rows = '<tr><th>detector</th><th>area m²</th><th>muons/spill</th><th>per m²</th></tr>';
  rows += `<tr><td>UBT</td><td>${ubtArea.toFixed(1)}</td><td>${fmt(ubtRate)}</td><td>${fmt(ubtRate / Math.max(ubtArea, 1e-9))}</td></tr>`;
  let totArea = ubtArea;
  for (const [side, k, name] of [['R', 0, 'right wings'], ['L', 1, 'left wings']]) {
    let area = 0, rate = 0;
    for (const w of des.wings[side]) if (w.active) {
      area += (w.z1 - w.z0) * (w.hi - w.lo); rate += sumHist2(D.side_hist[k], w.z0, w.z1, w.lo, w.hi);
    }
    totArea += area;
    rows += `<tr><td>${name}</td><td>${area.toFixed(1)}</td><td>${fmt(rate)}</td><td>${fmt(rate / Math.max(area, 1e-9))}</td></tr>`;
  }
  rows += `<tr><td><b>total</b></td><td><b>${totArea.toFixed(1)}</b></td><td></td><td></td></tr>`;
  document.getElementById('dettab').innerHTML = rows;
}

function rect(zs, xs, ys, color, name, show) {
  return {type: 'mesh3d', x: zs, y: xs, z: ys, i: [0, 0], j: [1, 2], k: [2, 3], color, opacity: 0.45,
          name, showlegend: show, legendgroup: name, hoverinfo: 'name', flatshading: true};
}

function geometryTraces(des) {
  const t = [];
  const u = des.ubt;
  t.push(rect([G.z_ubt, G.z_ubt, G.z_ubt, G.z_ubt], [-G.ubt_half_x, G.ubt_half_x, G.ubt_half_x, -G.ubt_half_x],
              [u.lo, u.lo, u.hi, u.hi], '#ff7f0e', 'UBT', true));
  for (const [side, sign] of [['R', 1], ['L', -1]]) des.wings[side].forEach((w, i) => {
    const z0 = w.active ? w.z0 : G.z_ubt, z1 = w.active ? w.z1 : G.z_ubt;
    const lo = w.active ? w.lo : 0, hi = w.active ? w.hi : 0;
    t.push(rect([z0, z1, z1, z0], [sign * X(z0), sign * X(z1), sign * X(z1), sign * X(z0)], [lo, lo, hi, hi],
                sign > 0 ? '#2ca02c' : '#17becf', sign > 0 ? 'wings, right (+x)' : 'wings, left (−x)', i === 0));
  });
  return t;
}

function muonTraces(des, hits) {
  const hz = [], hx = [], hy = [], hc = [];
  const mz = [], mx = [], my = [];
  for (const i of D.drawn) {
    if (D.ubt_on[i] && D.ubt_y[i] >= des.ubt.lo && D.ubt_y[i] <= des.ubt.hi) {
      hz.push(G.z_ubt); hx.push(D.ubt_x[i]); hy.push(D.ubt_y[i]); hc.push(D.colors[D.type[i]]);
    }
  }
  const drawnSet = new Set(D.drawn);
  for (let j = 0; j < D.c_mu.length; j++) {
    const i = D.c_mu[j]; if (!drawnSet.has(i)) continue;
    const sign = D.c_side[j], ws = des.wings[sign > 0 ? 'R' : 'L'], z = D.c_z[j], y = D.c_y[j];
    if (ws.some(w => w.active && z >= w.z0 && z <= w.z1 && y >= w.lo && y <= w.hi)) {
      hz.push(z); hx.push(sign * X(z)); hy.push(y); hc.push(D.colors[D.type[i]]);
    }
  }
  for (const i of D.drawn) if (D.type[i] <= 3 && hits[i] === 0) {
    const p = MUON_PATHS[i];
    for (const [x, y, z] of p) { mz.push(z); mx.push(x); my.push(y); }
    mz.push(null); mx.push(null); my.push(null);
  }
  return [
    {type: 'scatter3d', mode: 'markers', x: hz, y: hx, z: hy, marker: {size: 4.5, color: hc, line: {color: '#000', width: 1}},
     name: 'hits of drawn muons', hoverinfo: 'skip'},
    {type: 'scatter3d', mode: 'lines', x: mz, y: mx, z: my, line: {color: '#000', width: 6},
     name: 'drawn SST1 / veto-only muons with no hit', hoverinfo: 'skip'},
  ];
}

const MUON_PATHS = D.paths;

let pending = false;
function schedule() { if (!pending) { pending = true; requestAnimationFrame(update); } }

let first = true;
function update() {
  pending = false;
  const des = design(), hits = hitCounts(des);
  tables(des, hits);
  const dyn = geometryTraces(des).concat(muonTraces(des, hits));
  const div = document.getElementById('plot');
  if (first) {
    Plotly.newPlot(div, FIG.data.concat(dyn), FIG.layout, {responsive: true, displaylogo: false});
    first = false;
  } else {
    Plotly.react(div, FIG.data.concat(dyn), div.layout);
  }
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
