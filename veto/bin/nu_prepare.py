"""The neutrino flux sample in the layout of the design-optimisation muon dataset, one column store per flavour.

    python3 bin/nu_prepare.py [--max-xy 10] [--out data/neutrinos/store]
    python3 bin/nu_prepare.py --production /disk/lhcb_data/sqasim/nu_prod/tail --out data/interactions/neutrinos/store2026

--production: the 2026-11 target / shield production (bin/nu_prod.sbatch; min bias nuMB without its hf > 0 rows, charm
charmc, beauty beautyc; E >= 1 GeV, theta <= 0.3 rad), weight per spill = w_decay x 4e13 / protons of its component;
extra columns comp (0 min bias, 1 charm, 2 beauty) and E. Otherwise the 2018 FairShip flux:

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

SPILL = 4e13
PROD = (("nuMB", True), ("charmc", False), ("beautyc", False))


def load_production(d):
    import glob
    out = {k: [] for k in ("pdg", "x", "y", "z", "px", "py", "pz", "w", "mother_pdg", "comp")}
    for ic, (tag, drop) in enumerate(PROD):
        files = sorted(glob.glob(f"{d}/{tag}_nu_*.npz"))
        prot = sum(float(np.load(f)["events"]) * float(np.load(f)["rep"]) for f in files)
        for f in files:
            z = np.load(f)
            m = (z["hf"] == 0) if drop else np.ones(len(z["pdg"]), bool)
            for k in ("pdg", "x", "y", "z", "px", "py", "pz"):
                out[k].append(z[k][m])
            out["w"].append(z["w_decay"][m] * SPILL / prot)
            out["mother_pdg"].append(z["parent"][m].astype(np.int32))
            out["comp"].append(np.full(int(m.sum()), ic, np.int8))
        print(f"{tag}: {len(files)} files, {prot:.4g} protons", flush=True)
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-xy", type=float, default=10.0)
    ap.add_argument("--out", default=str(paths.interactions_data() / "neutrinos/store"))
    ap.add_argument("--production", default=None, help="directory of the 2026-11 production (bin/nu_prod.sbatch)")
    a = ap.parse_args()
    d = load_production(a.production) if a.production else neutrinos.load()
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
        if a.production:
            cols.update(comp=d["comp"][m][keep], E=np.sqrt(d["px"][m] ** 2 + d["py"][m] ** 2 + d["pz"][m] ** 2)[keep])
        out = Path(f"{a.out}_nu{int(pdg)}")
        out.mkdir(parents=True, exist_ok=True)
        for k, v in cols.items():
            np.save(out / f"{k}.npy", np.ascontiguousarray(v))
        json.dump({"n": n, "columns": list(cols), "flavour": int(pdg), "max_xy": a.max_xy,
                   "weight": "neutrinos per spill" + (" (4e13 p.o.t.)" if a.production else " (fastveto.neutrinos)"),
                   "source": a.production or "fastveto.neutrinos.load"},
                  open(out / "meta.json", "w"), indent=1)
        print(f"nu {int(pdg)}: {n:,} of {int(m.sum()):,} rows, {cols['weight'].sum():.4g} per spill -> {out}")


if __name__ == "__main__":
    main()
