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


def full_dataset_pickle() -> Path:
    """Every simulated muon at z = 30.6 m and z = 82 m (or where it stopped first)."""
    return data_root() / "data" / "Full_dataset_z_3060_8200.pkl"


def ubt_raw_pickle() -> Path:
    return MUONS_AND_MATTER / "data/outputs/UBT_data_TRY6.pkl"


def store_dir(name: str) -> Path:
    """Our own converted datasets (never inside Guglielmo's directories); the Nov-4 store is read where it lives."""
    if name == MUON_DATASET and "VETO_STORE" not in os.environ:
        return MUON_DATASET_DIR
    return Path(os.environ.get("VETO_STORE", PROJECT_DIR / "data" / "store")) / name


# Muon source for every study (Guglielmo, 2026-11-11: "use now the Nov-4 production for all", incl. D5).
# Nov-4 = complete target production (FairShip target, Pythia 8.223, Geant4 11.2 FTFP_BERT, 10 GeV, 4e13 p.o.t./spill,
# no stitched enriched tail), transported with --storage shell, 300 seeds. Older sources (alpha / gamma / delta-enriched /
# c-pool-v3, 2018 one-run files) only on explicit request; their results are kept until the Nov-4 ones are checked.
# Locations as given by Guglielmo (2026-11-11); Shah Rukh's production on /disk/lhcb_data/sqasim.
SQ = Path(os.environ.get("VETO_SQ_DATA", "/disk/lhcb_data/sqasim"))
SQ_VETO = SQ / "workspace" / "uzh" / "Veto_System" / "veto"
MUON_DATASET = "design-optimization-dataset-nov4-shell-300seeds-2026-11-10"
MUON_DATASET_DIR = SQ_VETO / "data" / "store" / MUON_DATASET   # design-optimisation store (300 shell seeds)
NOV4_POOL_DIR = Path(os.environ.get("VETO_NOV4_POOL", SQ / "nov4_pool"))   # source muons (PDG +-13) at the target
NOV4_POOL_TRAIN_CHUNKS = 23                       # train_000 - train_022
NOV4_POOL_HIDDEN_CHUNKS = 4                       # hidden_000 - hidden_003 (kept aside)
NOV4_SHELL_DIR = NOV4_POOL_DIR / "shell"          # after the shield transport, one file per spill: train/ hidden/ summary.txt
NU_FLUX_PREFIX = SQ_VETO / "data" / "interactions" / "neutrinos" / "store2026"   # flux stores <prefix>_nu<pdg>, 6 flavours
NU_FLAVOURS = (12, -12, 14, -14, 16, -16)
NU_PROD_RAW = SQ / "nu_prod" / "tail"             # raw neutrino production
GENIE_APPLY_DIR = SQ_VETO / "data" / "interactions" / "genie"   # GENIE interactions in the vessel, apply2026_<design>_nu<pdg>.npz
EVSTORE_DIR = Path(os.environ.get("VETO_EVSTORE", SQ / "evstore"))   # downstream event stores, one subdirectory per process
RECO_INPUTS = "inputs_nov4.npz"                   # results/reco pair inputs built from MUON_DATASET
RATE_MAPS = "sample_nov4.npz"                     # results/design_explorer rate maps built from MUON_DATASET


def nov4_pool_pattern(part: str = "train") -> str:
    """Transport input (bin/generate_muon_study.py --input, with --n-chunks NOV4_POOL_TRAIN_CHUNKS / _HIDDEN_CHUNKS)."""
    return str(NOV4_POOL_DIR / (part + "_{chunk:03d}.h5"))


def nu_apply_2026(design: str, pdg: int) -> Path:
    """GENIE interactions of one flavour in a design's vessel (Shah Rukh's 2026 flux, bin/nu_dis_apply.py)."""
    return GENIE_APPLY_DIR / f"apply2026_{design}_nu{pdg}.npz"


def evstore(name: str) -> Path:
    """An event store: a path, or a process name under EVSTORE_DIR (e.g. vessel_mu_nov4_D6_straw_mod)."""
    p = Path(name)
    return p if p.exists() or p.is_absolute() else EVSTORE_DIR / name


def ms7_parents() -> Path:
    """Muons crossing the last 50 cm of MS7 iron, from our transport of the Nov-4 pool (shield_dis prepare
    --from-transport, 3.95e7 rows); replaces the 2018 one-run data/shield_dis/muons_iron.npz."""
    return Path(os.environ.get("VETO_MS7_PARENTS", "/disk/lhcb_data/sqasim/ms7_parents/muons_iron_nov4.npz"))


def field_map_h5() -> Path:
    for candidate in (data_root() / "MainSpectrometerField.h5",
                      PROJECT_DIR.parent / "MainSpectrometerField.h5"):
        if candidate.exists():
            return candidate
    return data_root() / "MainSpectrometerField.h5"


def interactions_data() -> Path:
    """Interaction libraries (muon DIS, GENIE), GENIE splines, neutrino flux stores, cached apply files."""
    return Path(os.environ.get("VETO_INTERACTIONS", PROJECT_DIR / "data" / "interactions"))


def source_data() -> Path:
    """Source muons at the target: the enriched input file and the tail production it is built from."""
    return Path(os.environ.get("VETO_SOURCE", PROJECT_DIR / "data" / "source"))


def mu_library() -> Path:
    return interactions_data() / "dis" / "mu_library.npz"


def nu_library() -> Path:
    return interactions_data() / "genie" / "nu_library.npz"


def apply_file(kind: str, design: str) -> Path:
    """Cached per-design interaction file: kind "mu" (bin/muon_dis_apply.py) or "nu" (bin/nu_dis_apply.py)."""
    return interactions_data() / {"mu": "dis", "nu": "genie"}[kind] / f"apply_{design}.npz"


def enriched_input_h5() -> Path:
    return source_data() / "full_sample_after_target_enriched_p6.h5"


def cpp_build() -> Path:
    """Compiled modules and executables (outside lib/ and bin/, which bin/remote.sh sync replaces)."""
    return PROJECT_DIR / "cpp" / "build"


def results_dir() -> Path:
    path = Path(os.environ.get("VETO_RESULTS", PROJECT_DIR / "results"))
    path.mkdir(parents=True, exist_ok=True)
    return path
