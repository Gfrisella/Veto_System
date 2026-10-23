"""The neutrino flux sample in the layout of the design-optimisation muon dataset, one column store per flavour.

    python3 bin/nu_prepare.py [--max-xy 10] [--out data/neutrinos/store]

fastveto.neutrinos.load (2018 production, p > 10 GeV, weight per spill, ECN3 frame); straight lines from the
production point: state A at z = 30.6 m and B at z = 82 m (x, y, z, px, py, pz), no bounce (kx, ky, kz = NaN),
type 0, evt_n = row index. Kept: lines within |x|, |y| < MAX_XY at both planes (the cavern). Output
<out>_nu<pdg>/ with one .npy per column and meta.json (veto.io.write_store layout).
"""

import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

import numpy as np

from veto import paths

from fastveto import neutrinos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-xy", type=float, default=10.0)
    ap.add_argument("--out", default=str(paths.interactions_data() / "neutrinos/store"))
    a = ap.parse_args()
    d = neutrinos.load()
    for pdg in np.unique(d["pdg"]):
        m = d["pdg"] == pdg
        cols = {}
        for tag, zp in (("30", 30.6), ("82", 82.0)):
            dz = zp - d["z"][m]
            cols[f"x_{tag}"] = d["x"][m] + d["px"][m] / d["pz"][m] * dz
            cols[f"y_{tag}"] = d["y"][m] + d["py"][m] / d["pz"][m] * dz
            cols[f"z_{tag}"] = np.full(m.sum(), zp)
            for k in ("px", "py", "pz"):
                cols[f"{k}_{tag}"] = d[k][m]
        keep = ((np.abs(cols["x_30"]) < a.max_xy) & (np.abs(cols["y_30"]) < a.max_xy) & (np.abs(cols["x_82"]) < a.max_xy)
                & (np.abs(cols["y_82"]) < a.max_xy) & (d["pz"][m] > 0))
        cols = {k: v[keep] for k, v in cols.items()}
        n = int(keep.sum())
        cols.update(kx=np.full(n, np.nan), ky=np.full(n, np.nan), kz=np.full(n, np.nan), weight=d["w"][m][keep],
                    pdg=np.full(n, int(pdg), np.int32), evt_n=np.nonzero(m)[0][keep], type=np.zeros(n, np.int8),
                    mother_pdg=d["mother_pdg"][m][keep])
        out = Path(f"{a.out}_nu{int(pdg)}")
        out.mkdir(parents=True, exist_ok=True)
        for k, v in cols.items():
            np.save(out / f"{k}.npy", np.ascontiguousarray(v))
        json.dump({"n": n, "columns": list(cols), "flavour": int(pdg), "max_xy": a.max_xy,
                   "weight": "neutrinos per spill (fastveto.neutrinos)", "source": "fastveto.neutrinos.load"},
                  open(out / "meta.json", "w"), indent=1)
        print(f"nu {int(pdg)}: {n:,} of {int(m.sum()):,} rows, {cols['weight'].sum():.4g} per spill -> {out}")


if __name__ == "__main__":
    main()
