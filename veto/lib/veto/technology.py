"""Detector technologies for the veto planes (UBT and fins).

Each entry describes one complete plane (all its layers). Sources: SPSC-P-369 (2023 proposal) and
SPSC-SR-370 (2025 annual report, sec. 3.3.2 UBT), as summarised in conversation.txt. "assumed"
marks numbers that are not in the documents.

    straw   Liverpool "UBT0" straws (SR-370), but VERTICAL: 5 mm straws, 1 mm dead space, two
            staggered layers per view (6 mm pitch per layer, 3 mm effective), three views at
            0 and +-15 deg from vertical = 6 layers. Straws at most 1 m long, so a taller plane has
            overlapping rows (overlap ignored). ~100 um; 0.5 % single-straw inefficiency gives
            > 99.99 % with 6 staggered layers. Rate limit 0.5 MHz/cm^2 (ATLAS TRT reference).
            Time: drift limited, a fixed 100 ns window (the user's assumption; the documents give
            no straw time resolution). ~0.02 % X0 per layer (the report's 1.2 % for six layers is
            a factor-10 slip). Cost ~6 kCHF per layer-m^2 (P-369 table 11, SST).
    fiber   SciFi mats of Kuraray SCSF-78MJ, 250 um fibres, 6 staggered fibre layers (1.35 mm),
            Hamamatsu S13552 SiPM arrays at one end (250 um channels). Two crossed mats (x, y).
            ~100 um. sigma_t 0.5 ns and 99.5 % are the user's numbers (decay time 2.8 ns; no SciFi
            timing in the documents). Mats at most 3 m long (assumed: "a few metres", light
            attenuation). ~0.3 % X0 per mat (estimate). Cost ~170 kCHF per mat-m^2 (P-369, SND SciFi).
    pixel   Timepix-class hybrid pixels, 55 um. ~200 ps (Timepix4 bins; 50 ps would need LGADs).
            99.9 % (the user's number). Several tenths of a % X0 (0.5 % assumed). The documents have
            no pixel numbers: no rate limit, no cost (cost = nan).
    tile    UBT timing layer (SR-370): 5 x 5 x 1 cm^3 scintillating tiles, one SiPM each,
            ~200 ps. Up to 10 MHz per tile in the muon flames; tiles are made smaller to keep
            < 1 MHz per channel (max_channel_rate). ~2.4 % X0 (1 cm plastic). 99.5 % assumed.
            Cost ~30 kCHF/m^2 (P-369, timing detector).

Readout: strip technologies have `views` (strip angles from vertical, deg) with `layers_per_view`
layers each; pads (pixel, tile) measure both coordinates. A hit sits at the centre of a cell (u, y):
u = the pitch if a near-vertical view exists; y = the pitch with a horizontal view, pitch / tan(stereo)
with a stereo view, else one strip segment. Ghost combinations of several hits in one time window
are ignored.
"""

import numpy as np

TECHNOLOGIES = {
    "straw": dict(name="Straw (Liverpool UBT0, vertical)", readout="strip", views=(0.0, 15.0, -15.0),
                  layers_per_view=2, pitch=6e-3, diameter=5e-3, sigma_x=100e-6, sigma_t=None,
                  time_window=100e-9, efficiency=0.9999, max_length=1.0, max_rate=0.5e6,
                  max_channel_rate=None, x0_per_layer=0.0002, cost_m2_per_layer=6.0),
    "fiber": dict(name="Scintillating fibre (SCSF-78MJ mats)", readout="strip", views=(0.0, 90.0),
                  layers_per_view=1, pitch=250e-6, sigma_x=100e-6, sigma_t=0.5e-9, time_window=None,
                  efficiency=0.995, max_length=3.0, max_rate=None, max_channel_rate=None,
                  x0_per_layer=0.003, cost_m2_per_layer=170.0),
    "pixel": dict(name="Silicon pixel (Timepix-class)", readout="pad", views=None, layers_per_view=1,
                  pitch=55e-6, sigma_x=16e-6, sigma_t=0.2e-9, time_window=None, efficiency=0.999,
                  max_length=None, max_rate=None, max_channel_rate=None, x0_per_layer=0.005,
                  cost_m2_per_layer=None),
    "tile": dict(name="Scintillating tile (5 x 5 cm, SiPM)", readout="pad", views=None, layers_per_view=1,
                 pitch=0.05, sigma_x=0.05 / np.sqrt(12), sigma_t=0.2e-9, time_window=None, efficiency=0.995,
                 max_length=None, max_rate=None, max_channel_rate=1e6, x0_per_layer=0.024,
                 cost_m2_per_layer=30.0),
}
ALIASES = {"A": "straw", "B": "fiber", "C": "pixel", "fibre": "fiber", "tiles": "tile"}


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
