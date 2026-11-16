"""Straw-spectrometer resolution from the FairShip ACTS fit (ISSUES D14 (2), bin/acts_resolution_scan.py,
docs/acts_resolution.md).

NOT wired into reco.measure: reco.RecoConfig keeps its analytic constants until the switch is decided. Parametrisation in
data/acts/resolution_param.json (particle-gun mu+/- scan 1-400 GeV, FairShip 26.10 geometry and field V21, state at the
first measurement, our z = 81.89 m; core sigma = half the central 68 % interval):

    sigma(p) / p      = sqrt(a^2 + (b p)^2)                 p in GeV
    sigma(tx), (ty)   = sqrt(c^2 + (d / p)^2)   [rad]
    sigma(x), (y)     = sqrt(c^2 + (d / p)^2)   [m]

x is the non-bending, poorly measured projection (straws along x, x from the stereo views only), y the bending one.

field_scale multiplies sigma(p)/p only: it scales as 1 / int B dl. FairShip V21 has 0.84 T m on axis, our
MainSpectrometerField.h5 1.07 T m, so the default 0.84 / 1.07 = 0.78 gives the value in our field. Pass 1.0 for the raw
FairShip numbers.
"""
import json
from pathlib import Path

import numpy as np

PARAM_FILE = Path(__file__).resolve().parents[2] / "data" / "acts" / "resolution_param.json"
FIELD_SCALE_OURS = 0.84 / 1.07


def load(path=PARAM_FILE):
    return json.load(open(path))["fits"]


def sigmas(p, field_scale=FIELD_SCALE_OURS, fits=None):
    """Core sigmas for momenta p [GeV] (array): dict dp_over_p, tx_rad, ty_rad, x_m, y_m."""
    f = fits or load()
    p = np.maximum(np.asarray(p, float), 0.5)
    out = dict(dp_over_p=np.hypot(f["dp_over_p"]["a"], f["dp_over_p"]["b"] * p) * field_scale)
    for k in ("tx_rad", "ty_rad", "x_m", "y_m"):
        out[k] = np.hypot(f[k]["c"], f[k]["d"] / p)
    return out
