"""Detector technologies for the veto planes (UBT and fins).

Each entry describes one complete plane (all its layers). Sources: SPSC-P-369 (2023 proposal) and
SPSC-SR-370 (2025 annual report, sec. 3.3.2 UBT), as summarised in conversation.txt. "assumed"
marks numbers that are not in the documents.

    straw   Liverpool "UBT0" straws (SR-370), but VERTICAL: 5 mm straws at 5 mm pitch per layer
            (colleague's numbers, Oct 2026; SR-370 has 1 mm dead space, 6 mm pitch), two staggered
            layers per view, three views at 0 and +-15 deg from vertical = 6 layers. Straws at most
            1 m long, so a taller plane has overlapping rows (overlap ignored). sigma_x 110 +- 20 um
            (at least 100 um; 40-60 um achievable in principle). Efficiency > 97 % per straw and
            > 99.9 % per track (the plane number used here). Rate limit 0.5 MHz/cm^2 (ATLAS TRT
            reference). Time: drift limited, ~50 ns maximum drift, used as a 50 ns window (no straw
            time resolution yet). ~0.02 % X0 per layer (the report's 1.2 % for six layers is a
            factor-10 slip). Cost ~6 kCHF per layer-m^2 (P-369 table 11, SST).
    fiber   SciFi mats of Kuraray SCSF-78MJ, 250 um fibres, 6 staggered fibre layers (1.35 mm),
            Hamamatsu S13552 SiPM arrays at one end (250 um channels). Two crossed mats (x, y).
            ~100 um. sigma_t 0.5 ns is the user's number (decay time 2.8 ns); it can go down to
            ~100 ps, to be confirmed with Ettore. Efficiency basically 100 % for the detector (99.9 %
            used), a bit lower per fibre / SiPM channel (not modelled). Mats at most 3 m long
            (assumed: "a few metres", light attenuation). ~0.3 % X0 per mat (estimate). Cost
            ~170 kCHF per mat-m^2 (P-369, SND SciFi).
    pixel   Timepix-class hybrid pixels, 55 um. ~200 ps (Timepix4 bins). Pitch is variable and the
            timing depends on the chosen technology: much better with LGADs. 99.9 % (the user's
            number). Several tenths of a % X0 (0.5 % assumed). The SHiP documents have no pixel
            numbers: no rate limit. Cost ESTIMATE ~3 800 kCHF/m^2 from the ATLAS ITk pixel detector
            (hybrid, bump-bonded, same class): the UK share of its CORE cost (parts only) is 10.4 %
            = 5.1 MCHF (UK Phase-II project report, Dec 2024, indico.cern.ch/event/1487535), so
            ~49 MCHF in total, for ~13 m^2 of active area (ITk pixel TDR, CERN-LHCC-2017-021).
            Monolithic sensors (MAPS) would be cheaper per m^2; no number found.
    tile    UBT timing layer: scintillating tiles, one SiPM each (colleague's numbers, Oct 2026):
            4 x 4 cm, ~120 ps (`tile`) or 2 x 2 cm, ~80 ps (`tile2`); sigma_x ~ the pitch; ~99.9 %
            depending on the configuration; veto window +-1.5 ns. SR-370: up to 10 MHz per 5 x 5 cm^2
            tile in the muon flames, < 1 MHz per channel wanted (max_channel_rate). ~2.4 % X0 (1 cm
            plastic). Cost ~30 kCHF/m^2 (P-369, timing detector).

Relative cost per channel (cost_channel, straw = 1): fibre 4 and pixel 20 (the user's table), tile 5
(colleague). veto_window: fixed half width of the veto time window [s], instead of +- 3 sigma.

Readout: strip technologies have `views` (strip angles from vertical, deg) with `layers_per_view`
layers each; pads (pixel, tile) measure both coordinates. A hit sits at the centre of a cell (u, y):
u = the pitch if a near-vertical view exists; y = the pitch with a horizontal view, pitch / tan(stereo)
with a stereo view, else one strip segment. Ghost combinations of several hits in one time window
are ignored.
"""

import numpy as np

TECHNOLOGIES = {
    "straw": dict(name="Straw (Liverpool UBT0, vertical)", readout="strip", views=(0.0, 15.0, -15.0),
                  layers_per_view=2, pitch=5e-3, diameter=5e-3, sigma_x=110e-6, sigma_t=None,
                  time_window=50e-9, veto_window=None, efficiency=0.999, straw_efficiency=0.97,
                  max_length=1.0, max_rate=0.5e6, max_channel_rate=None, x0_per_layer=0.0002,
                  cost_m2_per_layer=6.0, cost_channel=1.0),
    "fiber": dict(name="Scintillating fibre (SCSF-78MJ mats)", readout="strip", views=(0.0, 90.0),
                  layers_per_view=1, pitch=250e-6, sigma_x=100e-6, sigma_t=0.5e-9, time_window=None,
                  veto_window=None, efficiency=0.999, max_length=3.0, max_rate=None, max_channel_rate=None,
                  x0_per_layer=0.003, cost_m2_per_layer=170.0, cost_channel=4.0),
    "pixel": dict(name="Silicon pixel (Timepix-class)", readout="pad", views=None, layers_per_view=1,
                  pitch=55e-6, sigma_x=16e-6, sigma_t=0.2e-9, time_window=None, veto_window=None,
                  efficiency=0.999, max_length=None, max_rate=None, max_channel_rate=None,
                  x0_per_layer=0.005, cost_m2_per_layer=3800.0, cost_channel=20.0),   # cost: estimate, see above
    "tile": dict(name="Scintillating tile (4 x 4 cm, SiPM)", readout="pad", views=None, layers_per_view=1,
                 pitch=0.04, sigma_x=0.04, sigma_t=0.12e-9, time_window=None, veto_window=1.5e-9,
                 efficiency=0.999, max_length=None, max_rate=None, max_channel_rate=1e6,
                 x0_per_layer=0.024, cost_m2_per_layer=30.0, cost_channel=5.0),
    "tile2": dict(name="Scintillating tile (2 x 2 cm, SiPM)", readout="pad", views=None, layers_per_view=1,
                  pitch=0.02, sigma_x=0.02, sigma_t=0.08e-9, time_window=None, veto_window=1.5e-9,
                  efficiency=0.999, max_length=None, max_rate=None, max_channel_rate=1e6,
                  x0_per_layer=0.024, cost_m2_per_layer=30.0, cost_channel=5.0),
}
ALIASES = {"A": "straw", "B": "fiber", "C": "pixel", "fibre": "fiber", "tiles": "tile", "tile4": "tile"}


def get(name):
    key = ALIASES.get(name, name)
    if key not in TECHNOLOGIES:
        raise ValueError(f"unknown technology {name!r}, choose from {sorted(TECHNOLOGIES)}")
    return key, TECHNOLOGIES[key]


def n_layers(tech):
    return len(tech["views"]) * tech["layers_per_view"] if tech["views"] else tech["layers_per_view"]


def n_segments(tech, length):
    """Strips needed end to end to cover `length` [m]."""
    if not tech.get("max_length"):
        return 1
    return max(int(np.ceil(length / tech["max_length"] - 1e-9)), 1)


def vertical(angle):
    return abs(angle) < 45


def n_channels(tech, width, height):
    """Readout channels of a width x height [m] plane (all layers)."""
    p = tech["pitch"]
    if tech["readout"] == "pad":
        return int(np.ceil(width / p) * np.ceil(height / p)) * tech["layers_per_view"]
    n = 0
    for a in tech["views"]:
        across, along = (width, height) if vertical(a) else (height, width)
        tilt = np.radians(a if vertical(a) else 90 - abs(a))
        n += np.ceil(across / p) * n_segments(tech, along / np.cos(tilt))
    return int(n * tech["layers_per_view"])


def cells(tech, width, height):
    """Cell size (u, y) [m] a hit is digitized to."""
    p = tech["pitch"]
    if tech["readout"] == "pad":
        return p, p
    views = tech["views"]
    cu = p if any(vertical(a) for a in views) else width / n_segments(tech, width)
    stereo = [abs(a) for a in views if 0 < abs(a) < 45]
    if any(not vertical(a) for a in views):
        cy = p
    elif stereo:
        cy = p / np.tan(np.radians(max(stereo)))
    else:
        cy = height / n_segments(tech, height)
    return cu, cy


def x0(tech):
    """Material of one plane, fraction of a radiation length."""
    return n_layers(tech) * tech["x0_per_layer"]


def cost(tech, area):
    """kCHF for a plane of `area` [m^2]; nan when the documents give no number."""
    if tech["cost_m2_per_layer"] is None:
        return float("nan")
    return area * n_layers(tech) * tech["cost_m2_per_layer"]


def relative_cost(tech, width, height):
    """Cost of a width x height [m] plane in units of one straw channel."""
    return n_channels(tech, width, height) * tech["cost_channel"]
