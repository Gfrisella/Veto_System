"""Weight treatments for muons that appear in several files (same evt_n, different seeds)."""

import numpy as np

from .config import POT_PER_SPILL

LEGACY_POT = 5e13          # p.o.t. per spill of every input made before 2026-11-09 (except the Nov-4 pool v5: 4e13)
POOL_4E13_TAGS = ("nov4_pool", "pool_v5")   # inputs whose weights are already per 4e13 (bin/build_pool_v5.py --pot 4e13)


def source_pot(meta=None, path=None):
    """p.o.t. per spill that a file's weights assume: meta["pot_per_spill"] when recorded; else 4e13 for inputs built
    from the Nov-4 pool v5 (meta "input" / "input_muons" or the path naming it), else the legacy 5e13."""
    meta = meta or {}
    if meta.get("pot_per_spill"):
        return float(meta["pot_per_spill"])
    text = " ".join(str(v) for v in (meta.get("input"), meta.get("input_muons"), path) if v)
    return 4e13 if any(t in text for t in POOL_4E13_TAGS) else LEGACY_POT


def to_spill(w, pot):
    """Weights normalised per spill of `pot` p.o.t. -> per spill of config.POT_PER_SPILL."""
    return w if pot == POT_PER_SPILL else w * (POT_PER_SPILL / pot)


def repetition_counts(evt_n):
    _, inverse, counts = np.unique(evt_n, return_inverse=True, return_counts=True)
    return counts[inverse]


def corrected_weights(weight, evt_n, n_files, counts=None):
    """Divide by the number of repetitions, keep the total, express it per file (per spill).

    counts: repetitions of each row's evt_n counted on the full sample (default: counted on `evt_n` itself).
    Pass it when `weight` / `evt_n` are a random subsample, otherwise repetitions are under-counted.
    Used by analysis.ipynb, Analysis_multifiles.py, make_design_dataset.py, run_selection.py.
    """
    corrected = weight / (repetition_counts(evt_n) if counts is None else counts)
    return corrected * (weight.sum() / n_files) / corrected.sum()


def keep_unique_events(data, n_files):
    """Keep only the first occurrence of each evt_n, weight *= repetitions / n_files.

    Used by temp1.py (KD-tree pairing).
    """
    _, first_idx, counts = np.unique(data["evt_n"], return_index=True, return_counts=True)
    order = np.argsort(first_idx)
    first_idx, counts = first_idx[order], counts[order]
    unique = {key: value[first_idx] for key, value in data.items()}
    unique["weight"] = unique["weight"] * counts / n_files
    return unique
