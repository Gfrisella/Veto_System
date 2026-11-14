#!/usr/bin/env python3
"""Muons in the cavern concrete (report appendix, 11 Nov 2026): how deep they go into the -x wall and the floor, for the
muons that come back into the cavern and for those that do not, and the z and angle where they enter and leave.

From --storage shell seeds of the transport (Nov-4 production, TRY6, cavern rock simulated by the transport; one seed =
one spill, Horvitz-Thompson weights). Only the states at z = 30.6 m (A) and z = 82 m (B, or the stop point) and the
bounce point K are stored, so the depth is reconstructed per fate:
  came back  K (where the incoming and outgoing lines meet, muon_types.bounce_points) within NEAR of the surface or
             behind it, B in the cavern: depth = distance of K behind the surface (0 if in front); exit z / angle from
             the B line traced back to the surface; path in the concrete ~ (|p_A| - |p_B|) / DEDX (approximate)
  stopped    B (the stop point) behind the surface: depth = distance of B behind it (exact, the muon ranged out there)
  through    B at z = 82 m still behind the surface: depth at z = 82 m (a lower bound of the maximum depth)
Entry: z where the A line meets the surface (the first of wall / floor), grazing angle of p_A to it.

    python3 bin/rock_depth_study.py /disk/lhcb_data/sqasim/nov4_pool/shell/train --max-seeds 300 --out results/rock_depth
"""
import _bootstrap  # noqa: F401
import argparse
import glob
import json
from pathlib import Path

import numpy as np

from veto import geometry as geo
from veto import muon_types as mt

NEAR = 0.3       # m, as bin/rock_hits.py
DEDX = 0.5       # GeV/m, concrete, muons of a few GeV (approximate; only for the path-length estimate)
FATES = ("back", "stopped", "through")


def wq(x, w, qs=(0.1, 0.5, 0.9)):
    if len(x) == 0:
        return [float("nan")] * len(qs)
    o = np.argsort(x); c = np.cumsum(w[o]) / w.sum()
    return [float(x[o][min(np.searchsorted(c, q), len(x) - 1)]) for q in qs]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir"); ap.add_argument("--max-seeds", type=int, default=300); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(f"{a.dir}/seed_*.npz"), key=lambda f: int(f.split("seed_")[-1].split(".")[0]))[:a.max_seeds]
    XW, YF = geo.CAVERN_X_WALL, geo.CAVERN_Y_FLOOR
    cols = {k: [] for k in ("surf", "fate", "w", "z_in", "a_in", "depth", "z_out", "a_out", "L", "p_in", "p_out")}
    for n, f in enumerate(files):
        d = np.load(f)
        r = np.nonzero((d["type"] <= 6) & (d["z_30"] > mt.REACH_30))[0]
        A = np.stack([d[f"{c}_30"][r] for c in "xyz"], 1).astype(np.float64)
        PA = np.stack([d[f"p{c}_30"][r] for c in "xyz"], 1).astype(np.float64)
        B = np.stack([d[f"{c}_82"][r] for c in "xyz"], 1).astype(np.float64)
        PBv = np.stack([d[f"p{c}_82"][r] for c in "xyz"], 1).astype(np.float64)
        K = np.stack([d["kx"][r], d["ky"][r], d["kz"][r]], 1).astype(np.float64)
        w, t = d["weight"][r].astype(np.float64), d["type"][r]
        pa, pb = np.linalg.norm(PA, axis=1), np.linalg.norm(PBv, axis=1)
        ua, ub = PA / pa[:, None], PBv / np.maximum(pb, 1e-12)[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            zw = np.where(ua[:, 0] < 0, A[:, 2] + (XW - A[:, 0]) / ua[:, 0] * ua[:, 2], np.inf)
            zf = np.where(ua[:, 1] < 0, A[:, 2] + (YF - A[:, 1]) / ua[:, 1] * ua[:, 2], np.inf)
        wall = zw <= zf
        z_in = np.minimum(zw, zf)
        goes = (z_in > A[:, 2]) & (z_in < 82.0)
        behind_B = np.where(wall, XW - B[:, 0], YF - B[:, 1])           # > 0: B is inside the concrete
        behind_K = np.where(wall, XW - K[:, 0], YF - K[:, 1])
        in_cavern = (t <= 5) & (behind_B < 0)
        back = goes & np.isfinite(K[:, 2]) & in_cavern & (behind_K > -NEAR)
        reached = B[:, 2] > mt.REACH_82
        stopped = goes & ~back & ~reached & (behind_B > 0)
        through = goes & ~back & reached & (behind_B > 0)
        fate = np.full(len(r), -1); fate[back] = 0; fate[stopped] = 1; fate[through] = 2
        a_in = np.where(wall, np.arcsin(np.clip(-ua[:, 0], 0, 1)), np.arcsin(np.clip(-ua[:, 1], 0, 1)))
        a_out = np.where(wall, np.arcsin(np.clip(ub[:, 0], 0, 1)), np.arcsin(np.clip(ub[:, 1], 0, 1)))
        with np.errstate(divide="ignore", invalid="ignore"):
            z_out = np.where(wall, B[:, 2] - (B[:, 0] - XW) / ub[:, 0] * ub[:, 2], B[:, 2] - (B[:, 1] - YF) / ub[:, 1] * ub[:, 2])
        depth = np.where(back, np.maximum(behind_K, 0), np.where(fate >= 1, behind_B, np.nan))
        m = fate >= 0
        for k, v in (("surf", np.where(wall, 0, 1)), ("fate", fate), ("w", w), ("z_in", z_in), ("a_in", a_in), ("depth", depth),
                     ("z_out", np.where(back, z_out, np.nan)), ("a_out", np.where(back, a_out, np.nan)),
                     ("L", np.where(back, np.clip(pa - pb, 0, None) / DEDX, np.nan)), ("p_in", pa), ("p_out", np.where(back, pb, np.nan))):
            cols[k].append(v[m])
        if n % 50 == 0:
            print(f"{n + 1}/{len(files)} seeds", flush=True)
    c = {k: np.concatenate(v) for k, v in cols.items()}
    S = len(files)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = dict(seeds=S, near=NEAR, dedx=DEDX, source=a.dir, rows={})
    print(f"\n{S} spills. Muons meeting the cavern concrete between z = 30.6 and 82 m (per spill; weighted mean / median "
          f"[10 %, 90 %]):")
    for si, surf in enumerate(("wall", "floor")):
        tot = c["w"][c["surf"] == si].sum() / S
        for fi, fn in enumerate(FATES):
            m = (c["surf"] == si) & (c["fate"] == fi)
            if not m.any():
                continue
            w = c["w"][m]
            row = dict(per_spill=float(w.sum() / S), share=float(w.sum() / S / tot), n=int(m.sum()),
                       n_eff=float(w.sum() ** 2 / (w ** 2).sum()))
            for q in ("z_in", "a_in", "depth", "p_in", "z_out", "a_out", "L", "p_out"):
                x = c[q][m]; ok = np.isfinite(x)
                if ok.any():
                    row[q] = dict(mean=float(np.average(x[ok], weights=w[ok])), q=wq(x[ok], w[ok]))
            res["rows"][f"{surf}_{fn}"] = row
            fmt = lambda q, s=1.0, u="": (f"{row[q]['mean'] * s:.3g} / {row[q]['q'][1] * s:.3g} [{row[q]['q'][0] * s:.3g}, "
                                         f"{row[q]['q'][2] * s:.3g}]{u}") if q in row else "-"
            print(f"  {surf:5s} {fn:8s} {row['per_spill']:.3g} / spill ({100 * row['share']:.1f} %, n_eff {row['n_eff']:.0f})")
            print(f"      depth {fmt('depth', 1, ' m')}; in: z {fmt('z_in', 1, ' m')}, angle {fmt('a_in', 1e3, ' mrad')}, "
                  f"p {fmt('p_in', 1, ' GeV')}")
            if fn == "back":
                print(f"      out: z {fmt('z_out', 1, ' m')}, angle {fmt('a_out', 1e3, ' mrad')}, p {fmt('p_out', 1, ' GeV')}; "
                      f"path in concrete ~ {fmt('L', 1, ' m')}")
    json.dump(res, open(out / "rock_depth.json", "w"), indent=1)
    np.savez_compressed(out / "rock_depth_rows.npz", **c, seeds=S)
    plot(c, S, out)
    print(f"-> {out}")


def plot(c, S, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {0: "#0053a1", 1: "#d55e00", 2: "#1b7f3b"}
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.6))
    for si, surf in enumerate(("cavern wall ($x=-4.56$ m)", "floor ($y=-3.36$ m)")):
        ax_d, ax_z, ax_a = axs[si]
        for fi, fn in enumerate(("came back", "stopped in it", "through, at $z$=82 m")):
            m = (c["surf"] == si) & (c["fate"] == fi)
            if not m.any():
                continue
            w = c["w"][m] / S
            ax_d.hist(np.clip(c["depth"][m], 1e-3, None), np.geomspace(1e-3, 30, 50), weights=w, histtype="step", lw=1.5,
                      color=col[fi], label=fn)
            ax_z.hist(c["z_in"][m], np.linspace(30, 82, 53), weights=w, histtype="step", lw=1.5, color=col[fi],
                      label=f"in, {fn}")
            ax_a.hist(np.clip(c["a_in"][m] * 1e3, 1e-1, None), np.geomspace(0.1, 1000, 50), weights=w, histtype="step",
                      lw=1.5, color=col[fi], label=f"in, {fn}")
            if fi == 0:
                ax_z.hist(c["z_out"][m], np.linspace(30, 82, 53), weights=w, histtype="step", lw=1.5, ls="--", color=col[0],
                          label="out, came back")
                ax_a.hist(np.clip(c["a_out"][m] * 1e3, 1e-1, None), np.geomspace(0.1, 1000, 50), weights=w,
                          histtype="step", lw=1.5, ls="--", color=col[0], label="out, came back")
        ax_d.set_xscale("log"); ax_d.set_yscale("log"); ax_d.set_xlabel("depth behind the surface [m]")
        ax_d.set_ylabel("muons per spill per bin"); ax_d.set_title(surf, fontsize=10)
        ax_z.set_yscale("log"); ax_z.set_xlabel("$z$ at the surface [m]"); ax_z.set_title(surf, fontsize=10)
        ax_a.set_xscale("log"); ax_a.set_yscale("log"); ax_a.set_xlabel("grazing angle to the surface [mrad]")
        ax_a.set_title(surf, fontsize=10)
        for ax in (ax_d, ax_z, ax_a):
            ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=8); axs[0, 1].legend(fontsize=7); axs[0, 2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "rock_depth.pdf"); fig.savefig(out / "rock_depth.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
