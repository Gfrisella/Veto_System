#!/usr/bin/env python3
"""Put new rate maps (results/design_explorer/<maps>) into existing reco-inputs files, without rebuilding the pairs.

  bin/remote.sh run bin/replace_rate_maps.py --maps sample_delta.npz inputs_delta.npz inputs_delta_s.npz
"""

import _bootstrap  # noqa: F401

import argparse

import numpy as np

from veto import paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", required=True)
    ap.add_argument("inputs", nargs="+", help="file names in results/reco")
    args = ap.parse_args()
    maps = np.load(paths.results_dir() / "design_explorer" / args.maps)
    for name in args.inputs:
        f = paths.results_dir() / "reco" / name
        d = dict(np.load(f))
        print(f"{name}: UBT map {d['ubt_hist'].sum():.4g} -> {maps['ubt_hist'].sum():.4g} muons/spill, "
              f"fin maps {d['fin_maps'].sum():.4g} -> {maps['fin_maps'].sum():.4g}")
        d.update(fin_maps=maps["fin_maps"], ubt_hist=maps["ubt_hist"])
        tmp = f.with_name(f.stem + ".tmp.npz")
        np.savez(tmp, **d)
        tmp.replace(f)


if __name__ == "__main__":
    main()
