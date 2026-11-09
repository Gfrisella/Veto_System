"""Default selection cuts (same values as get_config() in the original scripts)."""

import argparse

from . import geometry as geo

# Normalisation (ISSUES D13, decided 2026-11-09): every weight is muons (neutrinos, ...) per spill of 4e13 p.o.t.
# (FairShip run_fixedTarget.py), a 1 s spill, so rates per spill = Hz. 6e20 p.o.t. in 15 years -> 1.5e7 spills.
# Inputs normalised to another p.o.t. are rescaled when they are read (weights.to_spill, weights.source_pot).
POT_PER_SPILL = 4e13
POT_15_YEARS = 6e20
SPILLS_15Y = POT_15_YEARS / POT_PER_SPILL

DEFAULT_CONFIG = {
    # single-muon cuts
    "p_min": 1.0,                     # GeV/c
    "z_timing": geo.Z_TIMING,
    "timing_x_max": geo.TIMING_HALF_X,
    "timing_y_max": geo.TIMING_HALF_Y,

    # pair cuts
    "time_window": 200e-12,           # s, half window for time-window pairing
    "opposite_charge": True,
    "doca_max": 0.02,                 # m

    # IP
    "z_ip": geo.Z_TARGET,
    "ip_max": 2.5,                    # m

    # KD-tree pairing
    "z_sample_step": 0.5,             # m
    "candidate_radius": 0.10,         # m, deliberately larger than doca_max

    # normalization
    "pot_spill": POT_PER_SPILL,
    "pot_15_years": POT_15_YEARS,
}


def add_config_args(parser: argparse.ArgumentParser):
    """Expose every config value as --key on the command line."""
    group = parser.add_argument_group("selection config")
    for key, value in DEFAULT_CONFIG.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            group.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"),
                               default=value, metavar="BOOL")
        else:
            group.add_argument(flag, type=type(value), default=value)


def config_from_args(args):
    return {key: getattr(args, key) for key in DEFAULT_CONFIG}
