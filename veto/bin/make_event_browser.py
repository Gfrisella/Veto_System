#!/usr/bin/env python3
"""Data for the interactive veto event browser (one muon at a time, by category, full geometry).

Reads the validation of bin/validate_design.py (results/validate/<tag>.json and <tag>_paths.json:
summary numbers and a sample of muon paths per category, vetoed or missed) and adds, for every
sampled muon and design, where its true path crosses each veto plane (reco.true_crossing).
Writes results/validate/<tag>_browser.json and results/validate/event_browser_<tag>.html
(bin/event_browser_template.html with the data inlined). Wrapper: bin/event_browser.sh.

  bin/make_event_browser.py --tag bottom_main --designs D6_straw OPT1_tile2_flames OPT2_rank1
"""

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np

from veto import design as dsg
from veto import geometry as geo
from veto import muon_types as mt
from veto import paths, reco

CATS = ("wall bounce", "floor bounce", "selected", "other")


def r(a, n=3):
    return np.round(np.asarray(a, np.float64), n).tolist()


def plane_shape(p):
    """Corners (x, y, z) of a design plane, in order around the quad."""
    if p.kind == "ubt":
        return [[-geo.UBT_HALF_X, p.y_lo, p.z], [geo.UBT_HALF_X, p.y_lo, p.z],
                [geo.UBT_HALF_X, p.y_hi, p.z], [-geo.UBT_HALF_X, p.y_hi, p.z]]
    if p.kind == "bottom":
        x0, x1 = p.x_lo, p.to_global(p.u_max)
        return [[x0, p.y0, p.z], [x1, p.y0, p.z], [x1, p.y_end, p.z_end], [x0, p.y_end, p.z_end]]
    xi, zi = p.to_global(0.0), p.z_at(0.0)
    xo, zo = p.to_global(p.u_max), p.z_at(p.u_max)
    return [[xi, p.y_lo, zi], [xo, p.y_lo, zo], [xo, p.y_hi, zo], [xi, p.y_hi, zi]]


def crossings(design, tr):
    """Per muon: list of [plane index, x, y, z] where the true path crosses a plane."""
    out = [[] for _ in range(len(tr["ax"]))]
    for p in design.planes:
        u, v, _, ok = reco.true_crossing(tr, p)
        if p.kind == "ubt":
            x, y, z = tr["ax"], tr["ay"], np.full(len(u), p.z)
        elif p.kind == "bottom":
            x, z, y = p.to_global(u), p.z + v * np.cos(p.tilt), p.y0 + v * np.sin(p.tilt)
        else:
            x, y, z = p.to_global(u), v, p.z_at(u)
        for i in np.where(ok)[0]:
            out[i].append([p.index, round(float(x[i]), 3), round(float(y[i]), 3), round(float(z[i]), 3)])
    return out


def static_geometry():
    boxes = [dict(name="Muon shield (TRY6)", group="shield", z0=m["z_start"], z1=m["z_end"],
                  hx0=m["half_x_in"], hy0=m["half_y_in"], hx1=m["half_x_out"], hy1=m["half_y_out"])
             for m in geo.shield_magnets()]
    boxes.append(dict(name="Target", group="target", z0=geo.Z_TARGET, z1=geo.Z_START, hx0=0.15, hy0=0.15, hx1=0.15, hy1=0.15))
    boxes.append(dict(name="Decay vessel", group="vessel", z0=geo.Z_VESSEL_MIN, z1=geo.Z_VESSEL_MAX,
                      hx0=geo.VESSEL_X_IN, hy0=geo.VESSEL_Y_IN, hx1=geo.VESSEL_X_OUT, hy1=geo.VESSEL_Y_OUT))
    z0, z1 = geo.SPECTROMETER_Z
    boxes.append(dict(name="Spectrometer magnet", group="magnet", z0=z0, z1=z1, hx0=geo.SPECTROMETER_HALF_X,
                      hy0=geo.SPECTROMETER_HALF_Y, hx1=geo.SPECTROMETER_HALF_X, hy1=geo.SPECTROMETER_HALF_Y))
    planes = [dict(name="SST1 (z = 82 m)", group="sst1", z=geo.Z_SST1, hx=geo.SST1_HALF_X, hy=geo.SST1_HALF_Y),
              dict(name="Timing detector", group="timing", z=geo.Z_TIMING, hx=geo.TIMING_HALF_X, hy=geo.TIMING_HALF_Y)]
    cavern = dict(x_wall=geo.CAVERN_X_WALL, y_floor=geo.CAVERN_Y_FLOOR, z0=geo.Z_UBT - 2, z1=geo.Z_TIMING + 2)
    return dict(boxes=boxes, planes=planes, cavern=cavern)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="bottom_main")
    parser.add_argument("--designs", nargs="+", default=["D6_straw", "OPT1_tile2_flames", "OPT2_rank1"],
                        help="design names in the validation (prefix match)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="names on the page, one per design (default: the design names; for the default designs\n"
                             "'D6_straw (reference)', 'OPT1 (fins)', 'OPT2 (fins + bottom panel)')")
    parser.add_argument("--out", default=None, help="HTML file (default results/validate/event_browser_<tag>.html)")
    args = parser.parse_args()
    vdir = paths.results_dir() / "validate"
    results = json.loads((vdir / f"{args.tag}.json").read_text())["results"]
    samples = json.loads((vdir / f"{args.tag}_paths.json").read_text())
    default_labels = {"D6_straw": "D6_straw (reference)", "OPT1_tile2_flames": "OPT1 (fins)", "OPT2_rank1": "OPT2 (fins + bottom panel)"}
    labels = args.labels or [default_labels.get(k, k) for k in args.designs]
    if len(labels) != len(args.designs):
        parser.error("--labels needs one name per design")
    out = {"geometry": static_geometry(), "cats": CATS, "designs": []}
    for key, label in zip(args.designs, labels):
        found = [x for x in results if x["design"].startswith(key)]
        if not found:
            parser.error(f"no design starting with {key!r} in {args.tag}.json: {[x['design'] for x in results]}")
        res = found[0]
        s = samples[res["design"]]
        d = dsg.from_dict(res["raw"])
        a, k, b = (np.array(s[c]).T for c in ("a", "k", "b"))
        k = np.where(k == -999, np.nan, k)
        tr = {"ax": a[:, 0], "ay": a[:, 1], "az": a[:, 2], "apz": np.ones(len(a)),
              "kx": k[:, 0], "ky": k[:, 1], "kz": k[:, 2], "bx": b[:, 0], "by": b[:, 1], "bz": b[:, 2]}
        planes = [dict(name=p.name, kind=p.kind, technology=p.technology, corners=r(plane_shape(p)))
                  for p in d.planes]
        fw = res["fake_weighted"]
        out["designs"].append({
            "key": res["design"], "label": label, "cost_kchf": res["cost_kchf"], "planes": planes,
            "single": res["single"], "signal": res["signal"], "fake": fw,
            "fake_unweighted": res["fake_unweighted"], "fake_direct": res["fake_direct"],
            "settings": res["settings"],
            "muons": {"cat": s["cat"], "vetoed": s["vetoed"], "p": s["p"],
                      "a": r(a), "k": [[None if not np.isfinite(v) else round(float(v), 3) for v in row] for row in k],
                      "b": r(b), "cross": crossings(d, tr)},
        })
        print(f"{label}: {len(s['cat'])} muons, {len(planes)} planes")
    path = vdir / f"{args.tag}_browser.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"saved {path} ({path.stat().st_size / 1e6:.2f} MB)")
    template = (Path(__file__).parent / "event_browser_template.html").read_text()
    html = Path(args.out) if args.out else vdir / f"event_browser_{args.tag}.html"
    html.write_text(template.replace("__DATA__", json.dumps(out, separators=(",", ":"))))
    print(f"saved {html} ({html.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
