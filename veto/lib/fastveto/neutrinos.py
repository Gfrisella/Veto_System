"""The neutrino flux sample in the veto frame: Shah Rukh's 2026-11-09 production (the only neutrino flux used).

Source: /disk/gfs_lhcb/sqasim/nu_prod/tail (override with $VETO_NU_DIR), nu_prod.sbatch there: tailtransport neutrino
mode, FairShip target, FTFP_BERT, hadrons followed > 1.1 GeV, neutrinos > 1 GeV within 0.3 rad, decays upstream of
z = -0.0045 m. Parts: min bias (nuMB_nu_*.npz, rows with hf == 0 only, 500 x 2e4 p), charm (charmc_nu_*, charm_cascade
+ hf_decay, 1000 x 1e4 p), beauty (beautyc_nu_*, negligible). Rows: decay point x, y, z [m, veto frame], p [GeV],
w_decay (neutrinos per proton of that part). Weight per spill = w_decay x POT_PER_SPILL / protons of the part
(protons = sum events x rep), as Study_on_Veto_System/Neutrino_flux_gsimple2026/compare_flux.py (checked there against
the GSimple 2026 final-target flux: nu_mu within 1-4 %).

The 2018 FairShip flux (p > 10 GeV, no nu_e, x 2 - 2.7 low) was discarded on 2026-11-09 (Guglielmo).
"""
import glob
import os
from pathlib import Path

import numpy as np

from . import vessel as V

NU_DIR = Path(os.environ.get("VETO_NU_DIR", "/disk/gfs_lhcb/sqasim/nu_prod/tail"))
PARTS = (("nuMB", True), ("charmc", False), ("beautyc", False))     # (tag, keep only hf == 0)
SOURCE = "sqasim nu_prod 2026-11-09"
from veto.config import POT_PER_SPILL, SPILLS_15Y as SPILLS_15_YEARS  # noqa: E402

FLUX_POT = POT_PER_SPILL          # weights are made per POT_PER_SPILL here: nothing to rescale when read


def load(in_dir=NU_DIR, max_rows=None, vessel_only=False, pdgs=None, seed=0, parts=PARTS):
    """All weighted neutrinos as a dict of arrays (m, GeV, veto frame): pdg, x, y, z, px, py, pz, p, w (per spill),
    mother_pdg (decaying parent), part (0 min bias, 1 charm, 2 beauty).

    max_rows: random subsample (weights scaled up to keep the total). vessel_only: keep those whose straight line
    enters the helium. pdgs: keep only these pdg codes.
    """
    out = {k: [] for k in ("pdg", "x", "y", "z", "px", "py", "pz", "w", "mother_pdg", "part")}
    for ip, (tag, mb_only) in enumerate(parts):
        files = sorted(glob.glob(f"{in_dir}/{tag}_nu_*.npz"))
        if not files:
            raise FileNotFoundError(f"no {tag}_nu_*.npz in {in_dir}")
        protons, chunks = 0.0, []
        for f in files:
            with np.load(f) as z:
                protons += float(z["events"]) * float(z["rep"])
                m = z["pz"] > 0
                if mb_only:
                    m &= z["hf"] == 0
                if pdgs is not None:
                    m &= np.isin(z["pdg"], pdgs)
                chunks.append({k: z[k][m] for k in ("pdg", "x", "y", "z", "px", "py", "pz", "w_decay", "parent")})
        scale = POT_PER_SPILL / protons
        for c in chunks:
            for k in ("pdg", "x", "y", "z", "px", "py", "pz"):
                out[k].append(c[k])
            out["w"].append(c["w_decay"] * scale)
            out["mother_pdg"].append(c["parent"])
            out["part"].append(np.full(len(c["pdg"]), ip, np.int8))
    d = {k: np.concatenate(v) for k, v in out.items()}
    for k in ("x", "y", "z", "px", "py", "pz", "w"):
        d[k] = d[k].astype(np.float64)
    d["pdg"] = d["pdg"].astype(np.int64)
    d["p"] = np.sqrt(d["px"] ** 2 + d["py"] ** 2 + d["pz"] ** 2)
    if vessel_only:
        c = V.crossings(d["x"], d["y"], d["z"], d["px"], d["py"], d["pz"])
        m = c["face_in"] > 0
        d = {k: v[m] for k, v in d.items()}
    if max_rows is not None and max_rows < len(d["pdg"]):
        total = d["w"].sum()
        idx = np.random.default_rng(seed).choice(len(d["pdg"]), max_rows, replace=False)
        d = {k: v[idx] for k, v in d.items()}
        d["w"] *= total / d["w"].sum()
    return d
