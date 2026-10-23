"""The neutrino flux sample (scripts/extract_neutrinos.py output) in the ECN3 frame.

Source: Guglielmo's 2018 FairShip production (pythia8_Geant4_10.0_withCharmandBeauty*_mu.root, 67 files,
411.9 M events). Only neutrinos with p > 10 GeV that came with a stored muon; no nu_e.

Frame: positions are converted cm -> m and shifted so that the first production vertex of the sample
(fStartZ = -7104.5 cm) sits at the front of the ECN3 target, z = -2.14 m (Veto_System Z_TARGET,
MuonsAndMatter CreateTarget(z_start=-214 cm)).

Weight: the sister muon's fW. Summed over all 67 files it is per spill (5e13 PoT). Neutrinos without a
sister muon (~2 %) have no weight and are dropped.
"""
import glob
from pathlib import Path

import numpy as np

from veto import paths

from . import vessel as V

DEFAULT_DIR = paths.interactions_data() / "neutrinos"
Z_FIRST_VERTEX_OLD = -71.045      # m, fStartZ min of the sample
Z_TARGET_ECN3 = -2.14             # m
SHIFT = Z_TARGET_ECN3 - Z_FIRST_VERTEX_OLD
TOTAL_EVENTS = 411_947_589
POT_PER_SPILL = 5e13
SPILLS_15_YEARS = 6e20 / POT_PER_SPILL

COLUMNS = ("pdg", "px", "py", "pz", "x", "y", "z", "mother_pdg", "mu_sibling", "mu_w")


def load(in_dir=DEFAULT_DIR, max_rows=None, vessel_only=False, pdgs=None, seed=0):
    """All weighted neutrinos as a dict of arrays (m, GeV, ECN3 frame).

    w is per spill, scaled by TOTAL_EVENTS / events read when only some files are present.
    max_rows: random subsample (weights scaled up to keep the total). vessel_only: keep those whose
    straight line enters the helium. pdgs: keep only these pdg codes.
    """
    files = sorted(glob.glob(f"{in_dir}/*_mu.npz"))
    if not files:
        raise FileNotFoundError(f"no neutrino files in {in_dir}")
    cols = {k: [] for k in COLUMNS}
    n_events = 0
    for f in files:
        with np.load(f) as d:
            n_events += int(d["n_events"])
            for k in COLUMNS:
                cols[k].append(d[k])
    d = {k: np.concatenate(v) for k, v in cols.items()}
    keep = d["mu_sibling"] > 0
    if pdgs is not None:
        keep &= np.isin(d["pdg"], pdgs)
    d = {k: v[keep] for k, v in d.items()}
    d["x"] = d["x"].astype(np.float64) / 100
    d["y"] = d["y"].astype(np.float64) / 100
    d["z"] = d["z"].astype(np.float64) / 100 + SHIFT
    for k in ("px", "py", "pz"):
        d[k] = d[k].astype(np.float64)
    d["p"] = np.sqrt(d["px"] ** 2 + d["py"] ** 2 + d["pz"] ** 2)
    d["w"] = d.pop("mu_w").astype(np.float64) * TOTAL_EVENTS / n_events
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
