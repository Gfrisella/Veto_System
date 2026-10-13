"""Weight treatments for muons that appear in several files (same evt_n, different seeds)."""

import numpy as np


def repetition_counts(evt_n):
    _, inverse, counts = np.unique(evt_n, return_inverse=True, return_counts=True)
    return counts[inverse]


def corrected_weights(weight, evt_n, n_files):
    """Divide by the number of repetitions, keep the total, express it per file (per spill).

    Used by analysis.ipynb and Analysis_multifiles.py.
    """
    corrected = weight / repetition_counts(evt_n)
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
