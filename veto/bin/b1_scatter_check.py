#!/usr/bin/env python3
"""ISSUES B1 check (16 Nov 2026): does the search window r = r0 + (r_slope + r_ms / p) * L cover the TRUE multiple
scattering of the muons?

Same path as reco.run_events with multiple_scattering: kicks in the real materials (scattering.kicks, media + tails as
RecoConfig), the spectrometer measures the scattered track at SST1 (scattering.scattered_tracks -> reco.measure), the own
hit is the straight crossing plus the scattering offset at the plane (as reco.own_hits). The window centre and radius are
reco.predictions' (direct line; bounced muons excluded: they are predicted along the mirrored line, A2).

Per plane kind and momentum bin: fraction of crossings inside the window (with and without the half-cell margin that
the matching uses for cell-segmented planes), and the 95 / 99 % quantiles of distance / r. Control: the same without
scattering (window driven by r0 and the measurement only). Windows compared: the current r_ms / p, r_ms = 9 mrad GeV,
and the material-aware window r0 + r_slope L + k sigma_d (D18; sigma_d from the scatterers between plane and SST1).

    bin/b1_scatter_check.py --design designs/R2_sbt_ubt.json [--n 200000] --out results_rerun/reco/b1_scatter_R2.json
"""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

import numpy as np

from run_reco import TYPE_NAMES, load_inputs, take
from veto import design as dsg
from veto import muon_types as mt
from veto import reco, scattering

P_BINS = (0.0, 1.0, 2.0, 5.0, 10.0, 30.0, 1e9)


def sigma_material(sub, design, cfg, plane_z):
    """Material-aware window (ISSUES D18): expected miss of the back-extrapolated line at the plane from the scatterers
    BETWEEN the plane and SST1 (kicks upstream of the plane move hit and prediction alike), Highland on the total x/X0
    shared by x_i / x_tot as scattering.kicks. One projection; the 2D radius is k x this."""
    z, x = scattering.scatterers(sub, design, cfg.msc_media)
    p = np.maximum(np.sqrt(sub["bpx"] ** 2 + sub["bpy"] ** 2 + sub["bpz"] ** 2), 0.05)
    x_tot = x.sum(1)
    theta = scattering.highland(p, x_tot)
    with np.errstate(invalid="ignore", divide="ignore"):
        s_i = theta[:, None] * np.sqrt(np.where(x_tot[:, None] > 0, x / x_tot[:, None], 0))
    lever = np.where(np.isfinite(z) & (z > plane_z), z - plane_z, 0.0)
    return np.sqrt(((s_i * lever) ** 2).sum(1))


K_MAT = (2.5, 3.0, 3.5)


def check(sub, design, cfg, rng, scatter):
    if scatter:
        kicks = scattering.kicks(sub, design, rng, media=cfg.msc_media, tails=cfg.msc_tails)
        measured = scattering.scattered_tracks(sub, kicks)
    else:
        kicks, measured = None, sub
    meas = reco.measure(measured, cfg, rng)
    p = meas["p"]
    rows = []
    for plane in design.planes:
        u, y, _, ok = reco.true_crossing(sub, plane)
        if kicks is not None:                                  # as reco.own_hits
            dx, dy = scattering.offset_at(kicks, plane.z)
            crossed = np.isfinite(u) & np.isfinite(y)
            if plane.kind in ("ubt", "tiles"):
                crossed &= (sub["apz"] > 0) & (sub["az"] < plane.z + mt.A_TOLERANCE)
            u, y = u + (dx if plane.kind in ("ubt", "tiles") else plane.sign * dx), y + dy
            ok = crossed & plane.inside(u, y)
        dz = meas["z"] - plane.z
        pu = plane.to_local(meas["x"] - meas["tx"] * dz)
        py = meas["y"] - meas["ty"] * dz
        dist = dz * np.sqrt(1 + meas["tx"] ** 2 + meas["ty"] ** 2)
        pp = np.maximum(p, 0.1)
        sig = sigma_material(sub, design, cfg, plane.z)
        radii = {"current r_ms 6": cfg.r0 + (cfg.r_slope + cfg.r_ms / pp) * dist,
                 "r_ms 9": cfg.r0 + (cfg.r_slope + 9e-3 / pp) * dist}
        for k in K_MAT:
            radii[f"material k={k:g}"] = cfg.r0 + cfg.r_slope * dist + k * sig
        d = np.hypot(u - pu, y - py)
        uc, yc = plane.cell_centre(u, y)
        m = ok & np.isfinite(d)
        row = dict(kind=plane.kind, name=plane.name, p=p[m], d=d[m], r={}, in_r={}, in_cell={})
        for name, r in radii.items():
            row["r"][name] = r[m]; row["in_r"][name] = (d <= r)[m]
            row["in_cell"][name] = ((np.abs(uc - pu) <= r + plane.cell_u / 2) & (np.abs(yc - py) <= r + plane.cell_y / 2))[m]
        rows.append(row)
    return rows


def summarise(rows, w_of):
    out = {}
    names = list(rows[0]["r"])
    for kind in sorted({r["kind"] for r in rows}):
        rr = [r for r in rows if r["kind"] == kind]
        p = np.concatenate([r["p"] for r in rr]); d = np.concatenate([r["d"] for r in rr])
        out[kind] = {"n": int(len(p))}
        ref_area = None
        for name in names:
            rad = np.concatenate([r["r"][name] for r in rr])
            in_r = np.concatenate([r["in_r"][name] for r in rr]); in_c = np.concatenate([r["in_cell"][name] for r in rr])
            area = float((rad ** 2).mean())
            ref_area = ref_area or area
            bins = {}
            for lo, hi in zip(P_BINS[:-1], P_BINS[1:]):
                m = (p >= lo) & (p < hi)
                if m.sum() < 20:
                    continue
                q = d[m] / rad[m]
                bins[f"{lo:g}-{hi:g} GeV"] = dict(n=int(m.sum()), inside_r=float(in_r[m].mean()), inside_with_cell=float(in_c[m].mean()),
                                                q99_d_over_r=float(np.quantile(q, 0.99)), median_r_cm=float(100 * np.median(rad[m])),
                                                area_rel=float((rad[m] ** 2).mean()) / max(float((np.concatenate([r["r"][names[0]] for r in rr])[m] ** 2).mean()), 1e-30))
            out[kind][name] = dict(inside_r=float(in_r.mean()), inside_with_cell=float(in_c.mean()), area_rel=area / ref_area, by_p=bins)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", required=True); ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=11); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    design = dsg.load(Path(a.design))
    tr, inp = load_inputs()
    cfg = reco.RecoConfig(seed=a.seed, multiple_scattering=True)
    rng = np.random.default_rng(a.seed)
    single = np.asarray(inp["single"])
    single = single[tr["type"][single].astype(int) != 0]              # bounced: mirrored prediction (A2), not here
    rows_idx = rng.choice(single, min(a.n, len(single)), replace=False)
    sub = take(tr, rows_idx)
    res = dict(design=design.name, n_muons=len(rows_idx), config=dict(r0=cfg.r0, r_slope=cfg.r_slope, r_ms=cfg.r_ms,
               ms_slope=cfg.ms_slope, msc_media=cfg.msc_media, msc_tails=cfg.msc_tails), types=[TYPE_NAMES[1], TYPE_NAMES[2]])
    for label, sc in (("with_scattering", True), ("no_scattering", False)):
        res[label] = summarise(check(sub, design, cfg, np.random.default_rng(a.seed), sc), None)
        print(f"\n{design.name} {label}")
        for kind, v in res[label].items():
            print(f"  {kind} (n = {v['n']})")
            for name, x in v.items():
                if name == "n":
                    continue
                print(f"    {name:<16} inside {100 * x['inside_with_cell']:7.3f} %   mean window area x{x['area_rel']:.2f}")
                for b, y in x["by_p"].items():
                    print(f"        {b:<12} n={y['n']:>7}  inside {100 * y['inside_with_cell']:7.3f} %  q99 d/r {y['q99_d_over_r']:.2f}"
                          f"  median r {y['median_r_cm']:6.1f} cm  area x{y['area_rel']:.2f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
