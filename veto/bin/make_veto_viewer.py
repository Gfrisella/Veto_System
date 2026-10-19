#!/usr/bin/env python3
"""Interactive viewer of the SST1 muons and the veto designs, by muon category (one self-contained HTML).

Reads results/validate/<tag>.json and <tag>_paths.json (bin/validate_design.py): per design the
per-category rates before / after the veto, fake-signal rate, signal loss, plane rates, and a sample
of muon paths (A at 30.6 m -> bounce point K -> B at 82 m) with their veto flag.

  bin/make_veto_viewer.py --tag validation      # -> results/validate/veto_viewer.html
"""

import _bootstrap  # noqa: F401

import argparse
import json

import numpy as np

from veto import design as dsg
from veto import geometry as geo
from veto import muon_types as mt
from veto import paths

HERE = paths.results_dir() / "validate"


def geometry(raw):
    """Drawing primitives of a design: side view (z, y) and top view (z, x)."""
    d = dsg.from_dict(raw)
    side, top = [], []
    for p in d.planes:
        tech = p.technology or "generic"
        if p.kind == "ubt":
            side.append({"kind": "ubt", "z": [p.z, p.z], "y": [p.y_lo, p.y_hi], "tech": tech, "name": p.name})
            top.append({"kind": "ubt", "z": [p.z, p.z], "x": [-geo.UBT_HALF_X, geo.UBT_HALF_X], "tech": tech, "name": p.name})
        elif p.kind == "fin":
            z0, z1 = float(p.z_at(0.0)), float(p.z_at(p.u_max))
            side.append({"kind": "fin", "z": sorted([z0, z1]), "y": [p.y_lo, p.y_hi], "tech": tech, "name": p.name})
            top.append({"kind": "fin", "z": [z0, z1], "x": [float(p.to_global(0.0)), float(p.to_global(p.u_max))],
                        "tech": tech, "name": p.name})
        else:
            side.append({"kind": "bottom", "z": [p.z, float(p.z_end)], "y": [p.y0, float(p.y_end)], "tech": tech, "name": p.name})
            top.append({"kind": "bottom", "z": [p.z, float(p.z_end)], "x": [p.x_lo, float(p.to_global(p.u_max))],
                        "tech": tech, "name": p.name})
    return {"side": side, "top": top}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="validation")
    parser.add_argument("--labels", nargs="*", default=[], help="display names, in the order of the results")
    parser.add_argument("--out", default=str(HERE / "veto_viewer.html"))
    args = parser.parse_args()
    res = json.loads((HERE / f"{args.tag}.json").read_text())["results"]
    samples = json.loads((HERE / f"{args.tag}_paths.json").read_text())
    designs = []
    for i, r in enumerate(res):
        designs.append({
            "name": r["design"], "label": args.labels[i] if i < len(args.labels) else r["design"],
            "single": r["single"], "fake": r["fake_unweighted"], "fake_w": r["fake_weighted"],
            "fake_direct": r["fake_direct"], "signal": r["signal"], "cost": r["cost_kchf"],
            "planes": r["planes"], "pair_ineff": r["pair_muon_ineff"], "settings": r["settings"],
            "geo": geometry(r["raw"]), "paths": samples[r["design"]],
        })
    zv = np.linspace(geo.Z_VESSEL_MIN, geo.Z_VESSEL_MAX, 2)
    hx, hy = geo.vessel_half_widths(zv)
    data = {"designs": designs, "vessel": {"z": zv.tolist(), "hx": np.asarray(hx).tolist(), "hy": np.asarray(hy).tolist()},
            "floor": geo.CAVERN_Y_FLOOR, "wall": geo.CAVERN_X_WALL, "sst1": [geo.Z_SST1, geo.SST1_HALF_X, geo.SST1_HALF_Y]}
    html = (HERE.parent.parent / "bin" / "veto_viewer_template.html").read_text().replace("__DATA__", json.dumps(data))
    open(args.out, "w").write(html)
    print(f"saved {args.out} ({len(html) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
