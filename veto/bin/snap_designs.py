#!/usr/bin/env python3
"""Put the existing (continuous) designs on the 50 cm module grid (ISSUES A7): designs/<name>.json ->
designs/modules/<name>_mod.json (design.snap_to_modules; originals untouched), and print area, modules, active
fraction and cost (% of the 8 MCHF budget, A4).

    python3 bin/snap_designs.py [names ...]
"""

import _bootstrap  # noqa: F401

import argparse
import json

from veto import design as dsg
from veto import technology as tech


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*")
    a = ap.parse_args()
    out_dir = dsg.DESIGN_DIR / "modules"
    out_dir.mkdir(exist_ok=True)
    names = a.names or sorted(p.stem for p in dsg.DESIGN_DIR.glob("*.json"))
    print(f"{'design':<22}{'area m2':>9}{'modules':>9}{'active':>8}{'cost % budget':>15}{'(before)':>10}")
    for n in names:
        raw = json.loads((dsg.DESIGN_DIR / f"{n}.json").read_text())
        try:
            m = dsg.snap_to_modules(raw)
            d0, d1 = dsg.from_dict(raw), dsg.from_dict(m)
        except (ValueError, NotImplementedError) as e:
            print(f"{n:<22} skipped: {e}")
            continue
        (out_dir / f"{m['name']}.json").write_text(json.dumps(m, indent=1))
        act = sum(p.area * p.active_fraction for p in d1.planes) / d1.area
        print(f"{n:<22}{d1.area:>9.1f}{sum(p.n_modules for p in d1.planes):>9}{act:>8.3f}"
              f"{100 * d1.cost / tech.BUDGET_KCHF:>15.1f}{100 * d0.cost / tech.BUDGET_KCHF:>10.1f}")


if __name__ == "__main__":
    main()
