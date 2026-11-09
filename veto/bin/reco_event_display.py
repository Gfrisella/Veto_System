#!/usr/bin/env python3
"""Event display of the veto reconstruction (lib/veto/reco.py, algorithm A with the A2 bounce search) on real transported
muons: each sampled SST1 muon is one event; what it leaves in the veto planes, what the backward search sees, and why it is
vetoed or missed. Accidental hits are OFF (own hits only), so the display shows the algorithm's logic on the muon itself.

    python3 bin/reco_event_display.py <seed dir> --design D6_straw_mod --out reco_display [--per-type 3000 --seeds 200]

Outcome per muon (single-track events, search on every plane upstream of SST1):
  vetoed (straight)   a hit matches the straight backward extrapolation of the measured SST1 track
  vetoed (bounce)     no straight match, but a hit matches the mirrored line from the wall / floor point (A2)
  missed, hit off     the muon left hits in veto planes, none inside a search window
  missed, no hit      the muon crossed no veto plane (or the planes did not register it)
"""
import _bootstrap  # noqa: F401
import argparse
import glob
import json
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from veto import design as dmod
from veto import geometry as geo
from veto import muon_types as mt
from veto import reco

TYPES = {0: "SST1 bounced", 1: "SST1 selected", 2: "SST1 other"}
OUT = ("vetoed (straight)", "vetoed (bounce search)", "missed, hit outside the windows", "missed, no hit")


def no_accidentals(design, maps, n_events, window_ns, cfg, rng):
    return {k: np.empty(0) for k in ("event", "plane", "u", "y", "t")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir"); ap.add_argument("--design", default="D6_straw_mod"); ap.add_argument("--out", required=True)
    ap.add_argument("--per-type", type=int, default=3000); ap.add_argument("--seeds", type=int, default=200)
    a = ap.parse_args()
    reco.accidental_hits = no_accidentals
    rng = np.random.default_rng(4)
    design = dmod.load(f"designs/modules/{a.design}.json" if not a.design.endswith(".json") else a.design)
    cols = {}
    for f in sorted(glob.glob(f"{a.dir}/seed_*.npz"))[:a.seeds]:
        d = np.load(f)
        m = d["type"] <= 2
        for k in ("x_30", "y_30", "z_30", "px_30", "py_30", "pz_30", "x_82", "y_82", "z_82", "px_82", "py_82", "pz_82",
                  "kx", "ky", "kz", "weight", "type", "pdg"):
            cols.setdefault(k, []).append(d[k][m])
    c = {k: np.concatenate(v) for k, v in cols.items()}
    picks = []
    for t in TYPES:                                            # sample by weight: each pick stands for an equal share
        idx = np.nonzero(c["type"] == t)[0]
        picks.append(rng.choice(idx, min(a.per_type, len(idx)), replace=True, p=c["weight"][idx] / c["weight"][idx].sum()))
    pick = np.concatenate(picks)
    tr = {"ax": c["x_30"][pick], "ay": c["y_30"][pick], "az": c["z_30"][pick], "apx": c["px_30"][pick], "apy": c["py_30"][pick],
          "apz": c["pz_30"][pick], "kx": c["kx"][pick], "ky": c["ky"][pick], "kz": c["kz"][pick],
          "bx": c["x_82"][pick], "by": c["y_82"][pick], "bz": c["z_82"][pick], "bpx": c["px_82"][pick], "bpy": c["py_82"][pick],
          "bpz": c["pz_82"][pick], "pdg": -13.0 * c["pdg"][pick]}
    tr = {k: v.astype(np.float64) for k, v in tr.items()}
    typ = c["type"][pick]
    n = len(pick)
    ev, zl = np.arange(n), np.full(n, 1e9)
    cfg_b = reco.RecoConfig(seed=1)
    cfg_s = replace(cfg_b, bounce_search=False)
    rb = reco.run_events(tr, ev, zl, design, None, cfg_b, np.random.default_rng(1), details=True)
    rs = reco.run_events(tr, ev, zl, design, None, cfg_s, np.random.default_rng(1))
    own_any = np.bincount(rb["hits"]["track"][rb["hits"]["track"] >= 0], minlength=n) > 0
    outcome = np.where(rs["explained"], 0, np.where(rb["explained"], 1, np.where(own_any, 2, 3)))
    # which plane vetoed (first matching query)
    q, found = rb["queries"], rb["found"]
    plane_of = np.full(n, -1)
    plane_of[q["track"][found][::-1]] = q["plane"][found][::-1]
    print(f"design {design.name}: {len(design.planes)} planes; {n} muons ({a.per_type} per type, sampled by weight); "
          f"accidental hits OFF")
    lines, res = [], {}
    for t, name in TYPES.items():
        m = typ == t
        fr = [np.mean(outcome[m] == o) for o in range(4)]
        res[name] = fr
        lines.append(f"{name:14s} " + "  ".join(f"{OUT[o]} {100 * fr[o]:5.1f}%" for o in range(4)))
        pl = np.bincount(plane_of[m & (outcome <= 1)], minlength=len(design.planes)) if (m & (outcome <= 1)).any() else []
        if len(pl):
            top = np.argsort(pl)[::-1][:4]
            lines.append(f"{'':14s}   vetoed by: " + ", ".join(f"{design.planes[j].name} {100 * pl[j] / max(pl.sum(), 1):.0f}%" for j in top if pl[j]))
    print("\n".join(lines))
    json.dump(dict(design=design.name, per_type=a.per_type, outcome=res), open(a.out + ".json", "w"), indent=1)
    # ---- export for the interactive display: planes, and up to N_EXPORT muons per (type, outcome) with everything drawn
    meas_, hits_, q_, found_ = rb["meas"], rb["hits"], rb["queries"], rb["found"]
    zW_all, wall_all = reco.bounce_points(meas_)
    planes = [dict(name=pl.name, kind=pl.kind, z=float(pl.z), x0=float(pl.to_global(0.0)), x1=float(pl.to_global(pl.u_max)),
                   y0=float(pl.y_lo), y1=float(pl.y_hi)) for pl in design.planes]
    r3 = lambda v: round(float(v), 3)
    muons = []
    for t in TYPES:
        for o in range(4):
            for i in np.nonzero((typ == t) & (outcome == o))[0][:30]:
                path = [[r3(tr["ax"][i]), r3(tr["ay"][i]), r3(tr["az"][i])]]
                if np.isfinite(tr["kz"][i]):
                    path.append([r3(tr["kx"][i]), r3(tr["ky"][i]), r3(tr["kz"][i])])
                path.append([r3(tr["bx"][i]), r3(tr["by"][i]), r3(tr["bz"][i])])
                z0 = 28.0
                line = [[r3(meas_["x"][i] - meas_["tx"][i] * (meas_["z"][i] - z0)), r3(meas_["y"][i] - meas_["ty"][i] * (meas_["z"][i] - z0)), z0],
                        [r3(meas_["x"][i]), r3(meas_["y"][i]), r3(meas_["z"][i])]]
                mirror = None
                if np.isfinite(zW_all[i]):
                    zw = zW_all[i]
                    xw, yw = meas_["x"][i] - meas_["tx"][i] * (meas_["z"][i] - zw), meas_["y"][i] - meas_["ty"][i] * (meas_["z"][i] - zw)
                    txm = -meas_["tx"][i] if wall_all[i] else meas_["tx"][i]; tym = meas_["ty"][i] if wall_all[i] else -meas_["ty"][i]
                    mirror = [[r3(xw - txm * (zw - z0)), r3(yw - tym * (zw - z0)), z0], [r3(xw), r3(yw), r3(zw)]]
                qq = np.nonzero(q_["track"] == i)[0]
                wins = [[int(q_["plane"][k]), r3(design.planes[q_["plane"][k]].to_global(q_["u"][k])), r3(q_["y"][k]), r3(q_["r"][k]),
                         int(found_[k])] for k in qq]
                hh = np.nonzero(hits_["track"] == i)[0]
                hits = [[int(hits_["plane"][k]), r3(design.planes[hits_["plane"][k]].to_global(hits_["u"][k])), r3(hits_["y"][k])] for k in hh]
                muons.append(dict(type=int(t), outcome=int(o), q=1 if tr["pdg"][i] < 0 else -1,
                                  p=round(float(np.sqrt(tr["bpx"][i] ** 2 + tr["bpy"][i] ** 2 + tr["bpz"][i] ** 2)), 2),
                                  plane=int(plane_of[i]), path=path, line=line, mirror=mirror, wins=wins, hits=hits))
    json.dump(dict(design=design.name, planes=planes, types=TYPES, outcomes=OUT, fractions=res, per_type=a.per_type,
                   muons=muons, wall=geo.CAVERN_X_WALL, floor=geo.CAVERN_Y_FLOOR), open(a.out + "_events.json", "w"))
    print(f"exported {len(muons)} muons -> {a.out}_events.json")
    # ---- event display: examples per (type, outcome)
    ex = []
    for t, o in ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 2), (1, 3), (2, 0)):
        idx = np.nonzero((typ == t) & (outcome == o))[0]
        if len(idx):
            ex.append((int(idx[0]), t, o))
    fig, axs = plt.subplots(len(ex), 2, figsize=(26, 4.3 * len(ex)), constrained_layout=True)
    axs = np.atleast_2d(axs)
    meas, hits = rb["meas"], rb["hits"]
    zz = np.linspace(28, 84, 400)
    for row, (i, t, o) in enumerate(ex):
        A = np.array([tr["ax"][i], tr["ay"][i], tr["az"][i]]); B = np.array([tr["bx"][i], tr["by"][i], tr["bz"][i]])
        K = np.array([tr["kx"][i], tr["ky"][i], tr["kz"][i]])
        path = [A] + ([K] if np.isfinite(K[2]) else []) + [B]
        P = np.array(path)
        zW, wall = reco.bounce_points({k: v[i:i + 1] for k, v in meas.items()})
        for col, (v, j, lim) in enumerate((("x", 0, (-8, 11)), ("y", 1, (-5, 8)))):
            ax = axs[row, col]
            ax.axhline(geo.CAVERN_X_WALL if v == "x" else geo.CAVERN_Y_FLOOR, color="#8a7b63", lw=2)
            for pl in design.planes:                           # the veto planes
                if v == "x":
                    xs = [pl.to_global(0.0), pl.to_global(pl.u_max)]
                else:
                    xs = [pl.y_lo, pl.y_hi]
                ax.plot([pl.z, pl.z], xs, color="#2b8a3e" if pl.kind == "ubt" else "0.55", lw=3 if pl.kind == "ubt" else 2, alpha=0.7)
            ax.plot(P[:, 2], P[:, j], "k-", lw=2, label="true path (30.6 m -> bounce -> 82 m)")
            # straight backward extrapolation of the measured track
            mx = meas["x"][i] - meas["tx"][i] * (meas["z"][i] - zz) if v == "x" else meas["y"][i] - meas["ty"][i] * (meas["z"][i] - zz)
            ax.plot(zz, mx, "--", color="#2a78d6", lw=1.5, label="measured track, straight back")
            if np.isfinite(zW[0]):                              # A2: mirrored line from the wall / floor point W
                xW = meas["x"][i] - meas["tx"][i] * (meas["z"][i] - zW[0]); yW = meas["y"][i] - meas["ty"][i] * (meas["z"][i] - zW[0])
                txm = -meas["tx"][i] if wall[0] else meas["tx"][i]; tym = meas["ty"][i] if wall[0] else -meas["ty"][i]
                z2 = zz[zz < zW[0]]
                mb = (xW - txm * (zW[0] - z2)) if v == "x" else (yW - tym * (zW[0] - z2))
                ax.plot(z2, mb, "-.", color="#c77700", lw=1.5, label="bounce search: mirrored from the wall / floor")
            qm = q["track"] == i                                # search windows on the planes
            for k_ in np.nonzero(qm)[0]:
                pl = design.planes[q["plane"][k_]]
                cen = pl.to_global(q["u"][k_]) if v == "x" else q["y"][k_]
                ax.plot([pl.z, pl.z], [cen - q["r"][k_], cen + q["r"][k_]], color="#c0392b" if found[k_] else "#e8a3a3", lw=6, alpha=0.5)
            hm = hits["track"] == i                             # own hits
            for k_ in np.nonzero(hm)[0]:
                pl = design.planes[hits["plane"][k_]]
                hv = pl.to_global(hits["u"][k_]) if v == "x" else hits["y"][k_]
                ax.plot(pl.z, hv, "o", color="k", ms=7, mfc="#ffd400")
            ax.set_xlim(28, 84); ax.set_ylim(*lim); ax.set_xlabel("z [m]"); ax.set_ylabel(f"{v} [m]")
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="upper left")
        p = float(np.sqrt(tr["bpx"][i] ** 2 + tr["bpy"][i] ** 2 + tr["bpz"][i] ** 2))
        axs[row, 0].set_title(f"{TYPES[t]}, mu{'+' if tr['pdg'][i] < 0 else '-'}, p at SST1 {p:.1f} GeV: {OUT[o]}"
                              + (f" by {design.planes[plane_of[i]].name}" if o <= 1 and plane_of[i] >= 0 else ""), fontsize=11)
    fig.suptitle(f"Veto reconstruction on transported muons, design {design.name}: grey = fins, green = UBT, yellow dots = the "
                 f"muon's own hits, red bars = search windows (dark: matched), accidental hits off", fontsize=13)
    fig.savefig(a.out + ".png", dpi=55)
    print(a.out + ".png")


if __name__ == "__main__":
    main()
