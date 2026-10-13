"""Reading simulation outputs and derived datasets."""

import glob
import json
import pickle
import re

import numpy as np

from . import geometry as geo
from . import paths

TRACK_KEYS = ("x", "y", "z", "px", "py", "pz")


class _Unpickler(pickle.Unpickler):
    """Read pickles written with numpy 2 when running numpy 1.x."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            if not module.startswith("numpy._core"):
                raise
            return super().find_class("numpy.core" + module[len("numpy._core"):], name)


def load_pickle(filename):
    with open(filename, "rb") as f:
        return _Unpickler(f).load()


def seed_of(filename):
    match = re.search(r"seed_(\d+)", str(filename))
    return int(match.group(1)) if match else None


def list_output_files(pattern=None, max_files=None, verbose=True):
    """Simulation output files sorted by seed."""
    pattern = pattern or paths.output_pattern()
    files = sorted(glob.glob(pattern), key=seed_of)
    if not files:
        raise RuntimeError(f"No files found with pattern:\n{pattern}")

    if verbose:
        seeds = [seed_of(f) for f in files]
        missing = sorted(set(range(min(seeds), max(seeds) + 1)) - set(seeds))
        print(f"Found {len(files)} files, seeds {min(seeds)} -> {max(seeds)}")
        if missing:
            print(f"Missing seeds ({len(missing)}): {missing[:20]}{' ...' if len(missing) > 20 else ''}")

    if max_files is not None:
        files = files[:max_files]
    return files


def load_output_file(filename, plane=2):
    """One simulation output: track state at `plane` plus per-muon info.

    planes[0] = start (no z stored), planes[1] = UBT (z=30.6), planes[2] = SST1 (z=82).
    """
    raw = load_pickle(filename)

    state = raw["planes"][plane]
    data = {key: np.asarray(state[key]) for key in ("x", "y", "px", "py", "pz")}
    data["z"] = np.asarray(state["z"]) if "z" in state else np.full_like(data["x"], geo.Z_START)
    data["weight"] = np.asarray(raw["weight"])
    data["evt_n"] = np.asarray(raw["evt_n"])
    data["pdg"] = np.asarray(raw["pdg"])
    return data


def load_output_file_all_planes(filename):
    """All three planes, keys suffixed with the plane index (x0, x1, x2, ...)."""
    raw = load_pickle(filename)

    data = {
        "weight": np.asarray(raw["weight"]),
        "evt_n": np.asarray(raw["evt_n"]),
        "pdg": np.asarray(raw["pdg"]),
    }
    for index, state in raw["planes"].items():
        for key in TRACK_KEYS:
            if key in state:
                data[f"{key}{index}"] = np.asarray(state[key])
        if "z" not in state:
            data[f"z{index}"] = np.full_like(data[f"x{index}"], geo.Z_START)
    return data


def concatenate(chunks):
    return {key: np.concatenate([c[key] for c in chunks]) for key in chunks[0]}


def load_outputs(files, plane=2, all_planes=False, progress_every=1000):
    """Concatenate many output files; adds 'file_index'."""
    chunks = []
    for i, filename in enumerate(files):
        if progress_every and i % progress_every == 0:
            print(f"  loading [{i + 1}/{len(files)}] {filename}")
        data = load_output_file_all_planes(filename) if all_planes else load_output_file(filename, plane)
        data["file_index"] = np.full(len(data["weight"]), i, dtype=np.int32)
        chunks.append(data)
    data = concatenate(chunks)
    print(f"Loaded {len(data['weight']):,} muons from {len(files)} files")
    return data


def select(data, mask):
    return {key: value[mask] for key, value in data.items()}


def load_vessel_dataset(filename=None):
    filename = filename or paths.derived("muon_vessel_dataset.npz")
    with np.load(filename) as npz:
        return {key: npz[key] for key in npz.files}


def load_distribution(filename=None):
    filename = filename or paths.derived("muon_distribution_3d.pkl")
    distribution = load_pickle(filename)
    missing = [k for k in ("H", "x_edges", "y_edges", "z_edges") if k not in distribution]
    if missing:
        raise ValueError(f"Missing keys in {filename}: {missing}")
    return distribution


def load_field_map(filename=None):
    """Spectrometer field map: points [m] (N, 3) and B [T] (N, 3)."""
    import h5py
    filename = filename or paths.field_map_h5()
    with h5py.File(filename, "r") as f:
        return np.asarray(f["points"]), np.asarray(f["B"])


# ============================================================
# Column store: one .npy per column, read with memory mapping
# ============================================================

def write_store(directory, columns, meta=None):
    directory.mkdir(parents=True, exist_ok=True)
    for key, value in columns.items():
        np.save(directory / f"{key}.npy", np.ascontiguousarray(value))
    n = len(next(iter(columns.values())))
    with open(directory / "meta.json", "w") as f:
        json.dump({"n": int(n), "columns": list(columns), **(meta or {})}, f, indent=2)
    print(f"Wrote {n:,} rows, columns {list(columns)} to {directory}")


def open_store(directory):
    """Memory-mapped columns of a store (nothing is read until sliced)."""
    with open(directory / "meta.json") as f:
        meta = json.load(f)
    columns = {key: np.load(directory / f"{key}.npy", mmap_mode="r") for key in meta["columns"]}
    return columns, meta


def iterate_store(directory, chunk_size=10_000_000, columns=None, stop=None):
    """Yield (start, dict of in-memory chunks) over a store."""
    store, meta = open_store(directory)
    keys = columns or list(store)
    n = min(meta["n"], stop) if stop else meta["n"]
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        yield start, {key: np.asarray(store[key][start:end]) for key in keys}
