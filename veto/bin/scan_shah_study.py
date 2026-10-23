#!/usr/bin/env python3
"""Look at Guglielmo's shah_study seeds (read only): completed files, normalisation, our types vs his categories.

Each seed keeps up to 1500 muons per category (several categories per muon possible) with
weight_by_category[cat] scaled so that the picked muons of a category sum to its total weight. A file is
used only when its log says the run completed and it is older than --min-age minutes (still-running
seeds are skipped). Prints, for the seeds used:
  - per category: summed weight per seed (mean and spread): one seed = one spill if these match the
    old runs' per-spill totals
  - per our type (lib/veto/muon_types.py, classified from the z = 30.6 and last states): which of his
    categories its muons are in, weighted

  bin/remote.sh run bin/scan_shah_study.py --max-seeds 300
"""

import _bootstrap  # noqa: F401

import argparse
import pickle
import re
import time
from pathlib import Path

import numpy as np

from veto import muon_types as mt

SHAH = Path("/home/hep/gfrise/work_dir/MuonsAndMatter/data/outputs/shah_study")
STATE = ("x", "y", "z", "px", "py", "pz")


def completed_seeds(min_age_min=10.0):
    """Seeds whose log ends with 'Run completed' and whose file was last written > min_age minutes ago."""
    now = time.time()
    out = []
    for f in SHAH.glob("output_seed*.pkl"):
        seed = int(re.search(r"seed(\d+)", f.name).group(1))
        log = SHAH / "logs" / f"seed{seed}.log"
        if now - f.stat().st_mtime < 60 * min_age_min or not log.exists():
            continue
        tail = log.read_bytes()[-400:].decode(errors="ignore")
        if "Run completed" in tail and "Data saved" in tail:
            out.append(seed)
    return sorted(out)


def load(seed):
    with open(SHAH / f"output_seed{seed}.pkl", "rb") as fh:
        d = pickle.load(fh)
    tonp = lambda v: v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
    p0, p1 = d["planes"][0], d["planes"][1]
    out = {f"{k}_30": tonp(p0[k]).astype(np.float64) for k in STATE}
    out.update({f"{k}_82": tonp(p1[k]).astype(np.float64) for k in STATE})
    out["pdg"] = tonp(d["pdg"])
    out["evt_n"] = tonp(d["evt_n"])
    out["cat"] = {k: tonp(v).astype(bool) for k, v in d["categories"].items()}
    out["wcat"] = {k: tonp(v).astype(np.float64) for k, v in d["weight_by_category"].items()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-seeds", type=int, default=300)
    ap.add_argument("--min-age", type=float, default=10.0, help="minutes since the file was last written")
    args = ap.parse_args()
    seeds = completed_seeds(args.min_age)
    print(f"{len(seeds)} completed seeds (of {len(list(SHAH.glob('output_seed*.pkl')))} files); using {min(len(seeds), args.max_seeds)}")
    seeds = seeds[:args.max_seeds]
    cats = None
    per_seed = []
    tw = {}        # (our type, category) -> summed weight
    tn = {}
    for s in seeds:
        d = load(s)
        cats = cats or list(d["cat"])
        per_seed.append([d["wcat"][c].sum() for c in cats])
        typ = mt.classify(d)["type"]
        for t in np.unique(typ):
            m = typ == t
            for c in cats:
                tw[t, c] = tw.get((t, c), 0.0) + d["wcat"][c][m].sum()
                tn[t, c] = tn.get((t, c), 0) + int((m & (d["wcat"][c] > 0)).sum())
    per_seed = np.array(per_seed)
    print("\nper category: summed weight per seed (mean, rel. spread), muons picked")
    for j, c in enumerate(cats):
        print(f"  {c:<22} {per_seed[:, j].mean():12.4g}  {per_seed[:, j].std() / max(per_seed[:, j].mean(), 1e-30):6.3f}")
    print("\nour type x category: weight per seed (muons picked in total)")
    print("  " + " " * 15 + "".join(f"{c[:16]:>18}" for c in cats))
    for t in sorted({t for t, _ in tw}):
        print(f"  {t} {mt.TYPES[t]:<13}" + "".join(f"{tw[t, c] / len(seeds):>11.3g} ({tn[t, c]:>5})" for c in cats))


if __name__ == "__main__":
    main()
