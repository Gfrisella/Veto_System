#!/usr/bin/env python3
"""Peak rates of every design in designs/: coarse maps (results/reco/inputs.npz) vs the fine 2 x 2 cm^2 peak maps
(results/rate_maps/peak_maps.npz), and the rate penalty of the objective, old (log10) vs new (linear, rate_slope).

  python3 bin/check_peak_rates.py [--inputs inputs.npz]   # -> results/rate_maps/check_peak_rates.json
"""

import _bootstrap  # noqa: F401

import argparse
import json
import math

import numpy as np

from veto import design as vd
from veto import objective as vo
from veto import paths, reco
from veto.technology import TECHNOLOGIES


def old_penalty(res, cfg):
    """rate_penalty before 2026-11-04: 1 + log10(peak / limit) per limit exceeded, coarse peaks."""
    p = 0.0
    for pl in res["planes"]:
        tech = TECHNOLOGIES.get(pl["technology"]) if pl["technology"] else None
        for peak, limit in ((pl["peak_hz_cm2_coarse"], pl["max_rate_hz_cm2"]),
                            (pl["peak_channel_hz_coarse"], tech and tech.get("max_channel_rate"))):
            if limit and peak > limit:
                p += 1 + math.log10(peak / limit)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", default="inputs.npz")
    ap.add_argument("--spill", type=float, default=1.0)
    args = ap.parse_args()
    inp = np.load(paths.results_dir() / "reco" / args.inputs)
    coarse_only = {k: inp[k] for k in ("fin_maps", "z_slices", "u_bins", "y_bins", "ubt_hist", "ubt_y_bins")}
    coarse_only["peak_maps"] = None
    cfg = vo.ObjectiveConfig()
    out = {}
    for f in sorted((paths.PROJECT_DIR / "designs").glob("*.json")):
        d = vd.load(f)
        fine = reco.plane_rates(d, inp, args.spill)
        coarse = reco.plane_rates(d, coarse_only, args.spill)
        planes = []
        for a, c in zip(fine, coarse):
            planes.append(dict(plane=a["plane"], technology=a["technology"], peak_source=a["peak_source"],
                               peak_hz_cm2=a["peak_hz_cm2"], peak_hz_cm2_err=a["peak_hz_cm2_err"],
                               peak_hz_cm2_coarse=c["peak_hz_cm2"], max_rate_hz_cm2=a["max_rate_hz_cm2"],
                               peak_channel_hz=a["peak_channel"] / args.spill,
                               peak_channel_hz_coarse=c["peak_channel"] / args.spill))
        res = {"planes": planes}
        p_new, over = vo.rate_penalty(res, cfg)
        p_old = old_penalty(res, cfg)
        out[d.name] = dict(planes=planes, penalty_old=p_old * cfg.w_rate, penalty_new=p_new * cfg.w_rate, over=over)
        print(f"\n{d.name:<14} rate penalty old {p_old * cfg.w_rate:9.0f}   new {p_new * cfg.w_rate:9.0f}")
        for p in planes:
            if not p["technology"]:
                continue
            lim = TECHNOLOGIES[p["technology"]].get("max_channel_rate")
            print(f"  {p['plane']:<16} {p['technology']:<6} density {p['peak_hz_cm2_coarse'] / 1e3:8.1f} -> "
                  f"{p['peak_hz_cm2'] / 1e3:8.1f} +- {p['peak_hz_cm2_err'] / 1e3:5.1f} kHz/cm2 ({p['peak_source']}); "
                  f"channel {p['peak_channel_hz_coarse'] / 1e6:7.3f} -> {p['peak_channel_hz'] / 1e6:7.3f} MHz"
                  + (f" (limit {lim / 1e6:g})" if lim else "")
                  + (f" (density limit {p['max_rate_hz_cm2'] / 1e3:g} kHz/cm2)" if p["max_rate_hz_cm2"] else ""))
    dst = paths.results_dir() / "rate_maps" / "check_peak_rates.json"
    dst.write_text(json.dumps(out, indent=1))
    print(f"\nsaved {dst}")


if __name__ == "__main__":
    main()
