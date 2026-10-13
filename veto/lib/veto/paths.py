"""Where data lives.

Inputs are read from VETO_DATA_ROOT, which mirrors Guglielmo's project:
    <root>/outputs/TRY6_UBT_T1_seed_*.pkl
    <root>/data/{muon_vessel_dataset.npz, muon_distribution_3d.pkl, MainSpectrometerField.root}
    <root>/MainSpectrometerField.h5

On the cluster the default is his directory (read only, never write there).
Locally the default is veto/data, filled by bin/fetch_data.sh.
Everything we produce goes to VETO_RESULTS (default veto/results).

The UBT-plane dataset (MuonsAndMatter/data/outputs/UBT_data_TRY6.pkl, 150 M muons,
4.8 GB) is converted once by bin/convert_ubt_data.py into a column store
(one .npy per column, memory-mappable) in VETO_STORE (default veto/data/store).
"""

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
CLUSTER_DATA_ROOT = Path("/disk/users/gfrise/Optimization_Veto_System")
MUONS_AND_MATTER = Path("/disk/users/gfrise/MuonsAndMatter")


def data_root() -> Path:
    if "VETO_DATA_ROOT" in os.environ:
        return Path(os.environ["VETO_DATA_ROOT"])
    if CLUSTER_DATA_ROOT.exists():
        return CLUSTER_DATA_ROOT
    return PROJECT_DIR / "data"


def outputs_dir() -> Path:
    return data_root() / "outputs"


def output_pattern() -> str:
    return str(outputs_dir() / "TRY6_UBT_T1_seed_*.pkl")


def derived(name: str) -> Path:
    return data_root() / "data" / name


def ubt_raw_pickle() -> Path:
    return MUONS_AND_MATTER / "data/outputs/UBT_data_TRY6.pkl"


def store_dir(name: str) -> Path:
    """Our own converted datasets (never inside Guglielmo's directories)."""
    return Path(os.environ.get("VETO_STORE", PROJECT_DIR / "data" / "store")) / name


def field_map_h5() -> Path:
    for candidate in (data_root() / "MainSpectrometerField.h5",
                      PROJECT_DIR.parent / "MainSpectrometerField.h5"):
        if candidate.exists():
            return candidate
    return data_root() / "MainSpectrometerField.h5"


def results_dir() -> Path:
    path = Path(os.environ.get("VETO_RESULTS", PROJECT_DIR / "results"))
    path.mkdir(parents=True, exist_ok=True)
    return path
