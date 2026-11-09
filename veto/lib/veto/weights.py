"""Weight treatments for muons that appear in several files (same evt_n, different seeds)."""

import numpy as np


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
