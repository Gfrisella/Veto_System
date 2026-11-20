#!/usr/bin/env python3
"""ISSUES D19 (d): coverage of the SBT pointing window when the shell scatters the muons (sbt material_scattering), as
bin/b1_scatter_check.py does for the planes. Chooses k of the shell term (sbt cfg shell_ms_k).

Single SST1 muons of the reco inputs (all types), same chain as reco.run_events with multiple scattering: kicks in the real
materials + the SBT shell, the measured (scattered) track at SST1, own SBT hits from the true path. Random SBT hits OFF
(rate_scale 0) and efficiency 1, so the fraction tagged among the muons with an own fired cell is the window's coverage.
Per k: coverage (all / by type / by p), and for the misses the reason: the measured backward line meets no side wall
(no window can recover them), meets another face, or meets the right face outside the window.
"physical": the muon's energy loss in the shell along its stored path (Al 0.436, LS 0.150 GeV/m) is below its kinetic energy
at SST1; the others could not have crossed that much shell (the muon transport has no SBT material, D19 caveat 1): their
scattering is not meaningful (Highland at theta0 > 0.1 rad) and k is chosen on the physical muons.

    bin/sbt_shell_ms_check.py --design designs/R2_sbt_ubt.json --n 200000 --out results_rerun/sbt_d19/shell_ms_check_R2.json
"""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

import numpy as np

from run_reco import TYPE_NAMES, load_inputs, take
from veto import design as dsg
from veto import geometry as geo
from veto import reco, sbt, sbt_material as sm, scattering


def variant(path, **flags):
    raw = json.loads(Path(path).read_text())
    raw["sbt"] = dict(raw.get("sbt") or {}, **flags)
    return dsg.from_dict(raw)

P_BINS = (0.0, 1.0, 2.0, 5.0, 10.0, 30.0, 1e9)
KS = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", required=True); ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=11); ap.add_argument("--out", required=True)
    ap.add_argument("--ks", type=float, nargs="*", default=KS)
    a = ap.parse_args()
    design = dsg.load(Path(a.design))
    tr, inp = load_inputs()
    cfg = reco.RecoConfig(seed=a.seed, multiple_scattering=True)
    rng = np.random.default_rng(a.seed)
    single, sw_all = np.asarray(inp["single"]), np.asarray(inp["single_w"], float)
    pick = np.sort(rng.choice(len(single), min(a.n, len(single)), replace=False))
    rows, sw = single[pick], sw_all[pick]                 # weights as run_reco's single-muon efficiency
    sub = take(tr, rows)
    ty = tr["type"][rows].astype(int)
    n = len(rows)
    event = np.arange(n)
    zlim = np.full(n, 1e9)
    base = dict(rate_scale=0.0, efficiency=1.0, v0_line=False, selftag_hits=False, zone_rule=False)
    res = dict(design=design.name, n_muons=n, ks=list(a.ks))
    pp, qq, prow, _ = sbt.path_segments(sub)
    de = np.zeros(n)
    for g0, g1, dedx in ((0.0, sm.G_LS0, 0.436), (sm.G_LS0, sm.G_LS1, 0.150), (sm.G_LS1, sm.G_OUT, 0.436)):   # GeV / m
        seg, ta, tb = sm.layer_pieces(pp, qq, g0, g1)
        np.add.at(de, prow[seg], (tb - ta) * np.linalg.norm(qq[seg] - pp[seg], axis=1) * dedx)
    p_true = np.sqrt(sub["bpx"] ** 2 + sub["bpy"] ** 2 + sub["bpz"] ** 2)
    physical = de < np.sqrt(p_true ** 2 + 0.10566 ** 2) - 0.10566
    for label, msc in (("no_shell_scattering", False), ("shell_scattering", True)):
        dsc = variant(a.design, **base, material_scattering=msc)
        kicks = scattering.kicks(sub, dsc, np.random.default_rng(a.seed), media=cfg.msc_media, tails=cfg.msc_tails)
        tr_true, meas_tr = dict(sub, _kicks=kicks), scattering.scattered_tracks(sub, kicks)
        meas = reco.measure(meas_tr, cfg, np.random.default_rng(a.seed + 1))
        own = sbt.own_hits(tr_true, event, dsc.sbt, np.random.default_rng(a.seed + 2))
        has_own = np.bincount(own["event"], minlength=n) > 0
        # where does the measured backward line meet the side walls? (as sbt.tag)
        dzv = meas["z"] - np.minimum(zlim, geo.Z_SST1)
        start = np.stack([meas["x"] - meas["tx"] * dzv, meas["y"] - meas["ty"] * dzv, np.minimum(zlim, geo.Z_SST1)], 1)
        dz_end = meas["z"] - (geo.Z_VESSEL_MIN - 0.01)
        end = np.stack([meas["x"] - meas["tx"] * dz_end, meas["y"] - meas["ty"] * dz_end, np.full(n, geo.Z_VESSEL_MIN - 0.01)], 1)
        c = sbt.crossings(start, end)
        crosses = np.zeros(n, bool); crosses[c["seg"].astype(int)] = True
        own_faces = {}
        for e, cid in zip(own["event"], own["cell"]):
            own_faces.setdefault(int(e), set()).add(int(cid) // 10 // 1000)
        same_face = np.zeros(n, bool)
        for s_, f_ in zip(c["seg"].astype(int), c["face"].astype(int)):
            if f_ in own_faces.get(s_, ()):
                same_face[s_] = True
        p = meas["p"]
        out = {}
        for k in (a.ks if msc else (0.0,)):
            dk = variant(a.design, **base, material_scattering=msc, shell_ms_k=k)
            t = sbt.tag(tr_true, meas, event, zlim, dk.sbt, cfg, np.random.default_rng(a.seed + 3), n, v0=None)["vetoed"]
            m = has_own
            wm = lambda mm: float((sw * t)[mm].sum() / max(sw[mm].sum(), 1e-300))
            miss = m & ~t
            row = dict(n_with_own=int(m.sum()), coverage=float(t[m].mean()), coverage_w=wm(m),
                       by_type_w={TYPE_NAMES[j]: wm(m & (ty == j)) for j in range(3) if (m & (ty == j)).sum() > 20},
                       miss_no_wall_w=float(sw[miss & ~crosses].sum() / max(sw[miss].sum(), 1e-300)),
                       tagged_all_singles_w=float((sw * t).sum() / sw.sum()),
                       coverage_w_physical=wm(m & physical), coverage_w_unphysical=wm(m & ~physical),
                       weight_share_unphysical=float(sw[m & ~physical].sum() / sw[m].sum()),
                       by_type={TYPE_NAMES[j]: float(t[m & (ty == j)].mean()) for j in range(3) if (m & (ty == j)).sum() > 20},
                       by_p={f"{lo:g}-{hi:g}": dict(n=int((m & (p >= lo) & (p < hi)).sum()), coverage=float(t[m & (p >= lo) & (p < hi)].mean()))
                             for lo, hi in zip(P_BINS[:-1], P_BINS[1:]) if (m & (p >= lo) & (p < hi)).sum() > 20},
                       misses=dict(n=int(miss.sum()), line_meets_no_wall=float((miss & ~crosses).sum() / max(miss.sum(), 1)),
                                   other_face_only=float((miss & crosses & ~same_face).sum() / max(miss.sum(), 1)),
                                   same_face_outside_window=float((miss & same_face).sum() / max(miss.sum(), 1))),
                       tagged_all_singles=float(t.mean()))
            out[f"k={k:g}"] = row
            print(f"{label:<22} k={k:<4g} with own fired cell {m.sum():>7}: coverage {100 * row['coverage']:6.2f} % (physical w {100 * row['coverage_w_physical']:6.2f} %, unphysical w {100 * row['coverage_w_unphysical']:6.2f} % [{100 * row['weight_share_unphysical']:.0f} % of w]; weighted {100 * row['coverage_w']:6.2f} %, by type {', '.join(f'{k_[5:9]} {100 * v_:.1f}' for k_, v_ in row['by_type_w'].items())}; all singles {100 * row['tagged_all_singles_w']:.2f} %)  "
                  f"misses {miss.sum():>6}: no wall {100 * row['misses']['line_meets_no_wall']:5.1f} %, other face "
                  f"{100 * row['misses']['other_face_only']:5.1f} %, same face outside {100 * row['misses']['same_face_outside_window']:5.1f} %"
                  f"  | by p " + " ".join(f"{b}:{100 * v['coverage']:.1f}" for b, v in row["by_p"].items()), flush=True)
        res[label] = out
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
