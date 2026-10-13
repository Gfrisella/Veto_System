"""Default selection cuts (same values as get_config() in the original scripts)."""

import argparse

from . import geometry as geo

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
    "pot_spill": 5e13,
    "pot_15_years": 6e20,
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
