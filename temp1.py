# ============================================================
# IMPORTS
# ============================================================

import glob
import pickle
import re

import numpy as np
from scipy.spatial import cKDTree


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATTERN = "outputs/TRY6_UBT_T1_seed_*.pkl"

POT_TARGET = 5e13
POT_MC = 5e13

NORMALIZATION = POT_TARGET / POT_MC

N_FILES = None

# Single-muon cuts
P_MIN = 1.0                       # GeV/c

Z_TIMING = 94.850 - 2.14          # m
TIMING_X_MAX = 2.0                # m
TIMING_Y_MAX = 3.0                # m

# Decay vessel
Z_VESSEL_MIN = 30.6
Z_VESSEL_MAX = 80.6

X_VESSEL_MIN = 0.5
X_VESSEL_MAX = 2.0

Y_VESSEL_MIN = 1.35
Y_VESSEL_MAX = 3.0

# Pair selection
DOCA_MAX = 0.02                   # m = 2 cm

# IP
Z_IP = -2.14                      # m
IP_MAX = 2.5                      # m

# ------------------------------------------------------------
# Pair candidate search
#
# We do NOT construct all positive x negative combinations.
# Instead, tracks are sampled along z inside the decay vessel
# and a KDTree is used to find geometrically nearby opposite
# charge tracks.
# ------------------------------------------------------------

Z_SAMPLE_STEP = 0.5               # m

# This is deliberately larger than DOCA_MAX to avoid rejecting
# candidates during the preselection.
CANDIDATE_RADIUS = 0.10           # m

# Number of tracks inserted in KDTree at once
TREE_CHUNK_SIZE = 100000


# ============================================================
# FILE HANDLING
# ============================================================

def extrapolate_to_z(x, y, z, px, py, pz, z_target):
    """
    Straight-line extrapolation to z_target.
    """

    dz = z_target - z

    with np.errstate(divide="ignore", invalid="ignore"):
        x_target = x + (px / pz) * dz
        y_target = y + (py / pz) * dz

    return x_target, y_target


def load_simulation_file(filename, normalization=1.0):

    print(f"Loading: {filename}")

    with open(filename, "rb") as f:
        data = pickle.load(f)

    data_plane = data["planes"][2]

    x = np.asarray(data_plane["x"])
    y = np.asarray(data_plane["y"])

    if "z" in data_plane:
        z = np.asarray(data_plane["z"])
    else:
        z = np.zeros_like(x)

    px = np.asarray(data_plane["px"])
    py = np.asarray(data_plane["py"])
    pz = np.asarray(data_plane["pz"])

    weight = np.asarray(data["weight"])
    evt_n = np.asarray(data["evt_n"])
    pdg = np.asarray(data["pdg"])

    # --------------------------------------------------------
    # Backtrack to vessel entrance
    # --------------------------------------------------------

    x_at_30, y_at_30 = extrapolate_to_z(
        x,
        y,
        z,
        px,
        py,
        pz,
        Z_VESSEL_MIN
    )

    bouncing = ~(
        (np.abs(x_at_30) < 3)
        & (np.abs(y_at_30) < 3)
    )

    front = (
        (np.abs(x_at_30) < 0.5)
        & (np.abs(y_at_30) < 1.35)
    )

    weight = weight * normalization

    arrays = {
        "x": x,
        "y": y,
        "z": z,
        "px": px,
        "py": py,
        "pz": pz,
        "weight": weight,
        "evt_n": evt_n,
        "bouncing": bouncing,
        "front": front,
        "pdg": pdg,
    }

    lengths = {
        key: len(value)
        for key, value in arrays.items()
    }

    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"Inconsistent array lengths in {filename}: {lengths}"
        )

    return arrays


# ============================================================
# FIND FILES
# ============================================================

files = glob.glob(INPUT_PATTERN)

files = sorted(
    files,
    key=lambda f: int(
        re.search(r"seed_(\d+)\.pkl$", f).group(1)
    )
)

if len(files) == 0:
    raise RuntimeError(
        f"No files found with pattern: {INPUT_PATTERN}"
    )

print()
print("============================================================")
print("FILES")
print("============================================================")
print(f"Number of files: {len(files)}")

seed_numbers = [
    int(
        re.search(r"seed_(\d+)\.pkl$", f).group(1)
    )
    for f in files
]

print(
    f"Seed range: {min(seed_numbers)} -> {max(seed_numbers)}"
)

missing = sorted(
    set(range(min(seed_numbers), max(seed_numbers) + 1))
    - set(seed_numbers)
)

if missing:
    print(f"Missing seeds: {missing}")
else:
    print("No missing seeds.")

N_FILES = len(files)


# ============================================================
# LOAD ALL FILES
# ============================================================

data_planes = {
    "x": [],
    "y": [],
    "z": [],
    "px": [],
    "py": [],
    "pz": [],
    "weight": [],
    "evt_n": [],
    "bouncing": [],
    "front": [],
    "pdg": [],
}

for filename in files:

    data = load_simulation_file(
        filename,
        normalization=NORMALIZATION
    )

    for key in data:
        data_planes[key].append(data[key])


# ============================================================
# CONCATENATE
# ============================================================

for key in data_planes:

    data_planes[key] = np.concatenate(
        data_planes[key]
    )


print()
print("============================================================")
print("LOADED SAMPLE")
print("============================================================")
print(
    f"Total tracks before unique-event selection: "
    f"{len(data_planes['x']):,}"
)


# ============================================================
# UNIQUE EVENTS
#
# Keep ONLY the first occurrence of each evt_n.
#
# The weight of that first occurrence is then multiplied by
# the number of repetitions of that event and divided by
# the number of files.
# ============================================================

unique_evt, first_idx, counts = np.unique(
    data_planes["evt_n"],
    return_index=True,
    return_counts=True
)

# Sort according to the original sample order
order = np.argsort(first_idx)

first_idx = first_idx[order]
counts = counts[order]
unique_evt = unique_evt[order]

n_total_before_unique = len(data_planes["evt_n"])
n_unique = len(first_idx)

# ------------------------------------------------------------
# Save repetition count BEFORE reducing arrays
# ------------------------------------------------------------

n_repetitions = counts.copy()

# ------------------------------------------------------------
# Keep first occurrence only
# ------------------------------------------------------------

for key in data_planes:

    data_planes[key] = data_planes[key][first_idx]


# ------------------------------------------------------------
# Correct the weight
#
# w = w_first * repetitions / N_files
# ------------------------------------------------------------

data_planes["weight"] = (
    data_planes["weight"]
    * n_repetitions
    / N_FILES
)


print()
print("============================================================")
print("UNIQUE EVENT SELECTION")
print("============================================================")
print(
    f"Tracks before unique selection : "
    f"{n_total_before_unique:,}"
)
print(
    f"Unique events                  : "
    f"{n_unique:,}"
)
print(
    f"Reduction                      : "
    f"{100 * (1 - n_unique / n_total_before_unique):.2f}%"
)
print(
    f"Weight sum after correction    : "
    f"{data_planes['weight'].sum():.6e}"
)


# ============================================================
# MOMENTUM CUT
# ============================================================

p = np.sqrt(
    data_planes["px"] ** 2
    + data_planes["py"] ** 2
    + data_planes["pz"] ** 2
)

mask_p = p > P_MIN

n_before = len(data_planes["x"])
n_after = np.sum(mask_p)

print()
print("============================================================")
print("MOMENTUM CUT")
print("============================================================")
print(f"Before: {n_before:,}")
print(f"After : {n_after:,}")
print(
    f"Efficiency: "
    f"{100 * n_after / n_before:.3f}%"
)

for key in data_planes:
    data_planes[key] = data_planes[key][mask_p]


# ============================================================
# TIMING DETECTOR POSITION CUT
# ============================================================

x_timing, y_timing = extrapolate_to_z(
    data_planes["x"],
    data_planes["y"],
    data_planes["z"],
    data_planes["px"],
    data_planes["py"],
    data_planes["pz"],
    Z_TIMING
)

mask_timing = (
    (np.abs(x_timing) < TIMING_X_MAX)
    & (np.abs(y_timing) < TIMING_Y_MAX)
)

n_before = len(data_planes["x"])
n_after = np.sum(mask_timing)

print()
print("============================================================")
print("TIMING DETECTOR CUT")
print("============================================================")
print(f"Before: {n_before:,}")
print(f"After : {n_after:,}")
print(
    f"Efficiency: "
    f"{100 * n_after / n_before:.3f}%"
)

for key in data_planes:
    data_planes[key] = data_planes[key][mask_timing]


# ============================================================
# CROSS OVER
# ============================================================

x_at_30, y_at_30 = extrapolate_to_z(
    data_planes["x"],
    data_planes["y"],
    data_planes["z"],
    data_planes["px"],
    data_planes["py"],
    data_planes["pz"],
    Z_VESSEL_MIN
)

data_planes["cross_over"] = (
    x_at_30 * data_planes["pdg"] > 0
)

print()
print(
    f"Cross-over fraction: "
    f"{100 * data_planes['cross_over'].mean():.3f}%"
)


# ============================================================
# BACKTRACK TO DECAY VESSEL LATERAL WALL
# ============================================================

def backtrack_to_lateral(
    x0, y0, z0,
    px, py, pz,
    z_min=30.6,
    z_max=80.6,
    x_min=0.5,
    x_max=2.0,
    y_min=1.35,
    y_max=3.0,
):

    ax = (x_max - x_min) / (z_max - z_min)
    ay = (y_max - y_min) / (z_max - z_min)

    with np.errstate(
        divide="ignore",
        invalid="ignore"
    ):

        vx = px / pz
        vy = py / pz

        z_xplus = (
            x_min - x0
            + vx * z0
            - ax * z_min
        ) / (vx - ax)

        z_xminus = (
            -x_min - x0
            + vx * z0
            + ax * z_min
        ) / (vx + ax)

        z_yplus = (
            y_min - y0
            + vy * z0
            - ay * z_min
        ) / (vy - ay)

        z_yminus = (
            -y_min - y0
            + vy * z0
            + ay * z_min
        ) / (vy + ay)

        x_front, y_front = extrapolate_to_z(
            x0,
            y0,
            z0,
            px,
            py,
            pz,
            z_min
        )

    front_valid = (
        (np.abs(x_front) <= x_min)
        & (np.abs(y_front) <= y_min)
        & (z_min < z0)
    )

    z_candidates = np.column_stack([
        z_xplus,
        z_xminus,
        z_yplus,
        z_yminus,
        np.where(
            front_valid,
            z_min,
            np.nan
        )
    ])

    valid = (
        np.isfinite(z_candidates)
        & (z_candidates >= z_min)
        & (z_candidates < z_max)
        & (z_candidates < z0[:, None])
    )

    z_int = np.max(
        np.where(
            valid,
            z_candidates,
            -np.inf
        ),
        axis=1
    )

    x_int, y_int = extrapolate_to_z(
        x0,
        y0,
        z0,
        px,
        py,
        pz,
        z_int
    )

    no_intersection = ~np.isfinite(z_int)

    x_int[no_intersection] = np.nan
    y_int[no_intersection] = np.nan

    return z_int, x_int, y_int


z_decay_vessel, x_decay_vessel, y_decay_vessel = \
    backtrack_to_lateral(
        data_planes["x"],
        data_planes["y"],
        data_planes["z"],
        data_planes["px"],
        data_planes["py"],
        data_planes["pz"]
    )

print()
print("============================================================")
print("DECAY VESSEL BACKTRACKING")
print("============================================================")

valid_vessel_intersection = np.isfinite(
    z_decay_vessel
)

print(
    f"Tracks with vessel intersection: "
    f"{valid_vessel_intersection.sum():,}"
)


# ============================================================
# PAIR CANDIDATE SEARCH
#
# IMPORTANT:
#
# Do NOT do:
#
#     np.repeat(pos, len(neg))
#     np.tile(neg, len(pos))
#
# because this creates billions of pairs.
#
# Instead:
#
# 1. Sample every track inside the decay vessel.
# 2. Put negative-charge samples into a KDTree.
# 3. Search for nearby positive-charge samples.
# 4. Build only those candidate track pairs.
#
# The exact DOCA is calculated afterwards.
# ============================================================

def build_track_samples(
    data_planes,
    indices,
    z_samples
):

    sample_positions = []
    sample_tracks = []

    x = data_planes["x"]
    y = data_planes["y"]
    z = data_planes["z"]
    px = data_planes["px"]
    py = data_planes["py"]
    pz = data_planes["pz"]

    for idx in indices:

        if abs(pz[idx]) < 1e-12:
            continue

        xs, ys = extrapolate_to_z(
            x[idx],
            y[idx],
            z[idx],
            px[idx],
            py[idx],
            pz[idx],
            z_samples
        )

        valid = (
            np.isfinite(xs)
            & np.isfinite(ys)
            & (np.abs(xs) <= 2.1)
            & (np.abs(ys) <= 3.1)
        )

        if not np.any(valid):
            continue

        positions = np.column_stack([
            xs[valid],
            ys[valid],
            np.full(
                np.sum(valid),
                0.0
            )
        ])

        # We use a transformed coordinate system:
        #
        # [x, y, z_scaled]
        #
        # z is inserted below by replacing the third
        # coordinate.

        valid_z = z_samples[valid]

        positions[:, 2] = valid_z

        sample_positions.append(positions)

        sample_tracks.append(
            np.full(
                len(positions),
                idx,
                dtype=np.int64
            )
        )

    if not sample_positions:

        return (
            np.empty((0, 3)),
            np.empty(0, dtype=np.int64)
        )

    return (
        np.vstack(sample_positions),
        np.concatenate(sample_tracks)
    )


def find_candidate_pairs(
    data_planes,
    z_min=30.6,
    z_max=80.6,
    z_step=0.5,
    candidate_radius=0.10
):

    pdg = data_planes["pdg"]

    positive = np.where(pdg > 0)[0]
    negative = np.where(pdg < 0)[0]

    print()
    print("============================================================")
    print("OPPOSITE-CHARGE PAIR SEARCH")
    print("============================================================")
    print(f"Positive tracks: {len(positive):,}")
    print(f"Negative tracks: {len(negative):,}")

    z_samples = np.arange(
        z_min,
        z_max + z_step,
        z_step
    )

    print(
        f"Number of z sampling points: "
        f"{len(z_samples)}"
    )

    # --------------------------------------------------------
    # Build negative-track spatial samples
    # --------------------------------------------------------

    neg_positions, neg_tracks = build_track_samples(
        data_planes,
        negative,
        z_samples
    )

    print(
        f"Negative spatial samples: "
        f"{len(neg_positions):,}"
    )

    if len(neg_positions) == 0:

        return np.empty(
            (0, 2),
            dtype=np.int64
        )

    tree = cKDTree(neg_positions)

    candidate_set = set()

    # --------------------------------------------------------
    # Positive tracks
    # --------------------------------------------------------

    for counter, idx in enumerate(positive):

        if counter % 10000 == 0:
            print(
                f"Processed positive tracks: "
                f"{counter:,}/{len(positive):,}"
            )

        if abs(data_planes["pz"][idx]) < 1e-12:
            continue

        xs, ys = extrapolate_to_z(
            data_planes["x"][idx],
            data_planes["y"][idx],
            data_planes["z"][idx],
            data_planes["px"][idx],
            data_planes["py"][idx],
            data_planes["pz"][idx],
            z_samples
        )

        valid = (
            np.isfinite(xs)
            & np.isfinite(ys)
            & (np.abs(xs) <= 2.1)
            & (np.abs(ys) <= 3.1)
        )

        if not np.any(valid):
            continue

        query_points = np.column_stack([
            xs[valid],
            ys[valid],
            z_samples[valid]
        ])

        nearby = tree.query_ball_point(
            query_points,
            r=candidate_radius
        )

        for neighbours in nearby:

            for neg_sample_idx in neighbours:

                j = neg_tracks[neg_sample_idx]

                if idx == j:
                    continue

                if idx < j:
                    pair = (idx, j)
                else:
                    pair = (j, idx)

                candidate_set.add(pair)

    print(
        f"Unique candidate pairs: "
        f"{len(candidate_set):,}"
    )

    if not candidate_set:

        return np.empty(
            (0, 2),
            dtype=np.int64
        )

    return np.asarray(
        list(candidate_set),
        dtype=np.int64
    )


# ============================================================
# EXACT DOCA CALCULATION
# ============================================================

def calculate_doca(
    data_planes,
    pairs
):

    if len(pairs) == 0:

        return (
            np.empty(0),
            np.empty((0, 3)),
            np.empty((0, 3))
        )

    r = np.column_stack([
        data_planes["x"],
        data_planes["y"],
        data_planes["z"]
    ])

    p = np.column_stack([
        data_planes["px"],
        data_planes["py"],
        data_planes["pz"]
    ])

    i = pairs[:, 0]
    j = pairs[:, 1]

    r1 = r[i]
    r2 = r[j]

    p1 = p[i]
    p2 = p[j]

    dr = r1 - r2

    cross = np.cross(
        p1,
        p2
    )

    cross_norm = np.linalg.norm(
        cross,
        axis=1
    )

    valid = cross_norm > 1e-12

    doca = np.full(
        len(pairs),
        np.inf
    )

    doca[valid] = (
        np.abs(
            np.sum(
                dr[valid] * cross[valid],
                axis=1
            )
        )
        / cross_norm[valid]
    )

    # --------------------------------------------------------
    # Closest points
    # --------------------------------------------------------

    w = r1 - r2

    a = np.sum(
        p1 * p1,
        axis=1
    )

    b = np.sum(
        p1 * p2,
        axis=1
    )

    c = np.sum(
        p2 * p2,
        axis=1
    )

    d = np.sum(
        p1 * w,
        axis=1
    )

    e = np.sum(
        p2 * w,
        axis=1
    )

    den = a * c - b * b

    valid &= (
        np.abs(den) > 1e-12
    )

    s = np.zeros(len(pairs))
    u = np.zeros(len(pairs))

    s[valid] = (
        b[valid] * e[valid]
        - c[valid] * d[valid]
    ) / den[valid]

    u[valid] = (
        a[valid] * e[valid]
        - b[valid] * d[valid]
    ) / den[valid]

    point1 = (
        r1
        + s[:, None] * p1
    )

    point2 = (
        r2
        + u[:, None] * p2
    )

    vertex = 0.5 * (
        point1 + point2
    )

    momentum = p1 + p2

    # Invalid geometry
    doca[~valid] = np.inf
    vertex[~valid] = np.nan
    momentum[~valid] = np.nan

    return doca, vertex, momentum


# ============================================================
# FIND CANDIDATE OPPOSITE-CHARGE PAIRS
# ============================================================

candidate_pairs = find_candidate_pairs(
    data_planes,
    z_min=Z_VESSEL_MIN,
    z_max=Z_VESSEL_MAX,
    z_step=Z_SAMPLE_STEP,
    candidate_radius=CANDIDATE_RADIUS
)


# ============================================================
# EXACT DOCA
# ============================================================

print()
print("============================================================")
print("DOCA")
print("============================================================")

doca, vertex, momentum = calculate_doca(
    data_planes,
    candidate_pairs
)

mask_doca = (
    np.isfinite(doca)
    & (doca < DOCA_MAX)
)

selected_pairs = candidate_pairs[mask_doca]
selected_doca = doca[mask_doca]
selected_vertex = vertex[mask_doca]
selected_momentum = momentum[mask_doca]

print(
    f"Candidate pairs : "
    f"{len(candidate_pairs):,}"
)

print(
    f"DOCA < {DOCA_MAX * 100:.1f} cm : "
    f"{len(selected_pairs):,}"
)

if len(selected_doca) > 0:

    print(
        f"Minimum DOCA : "
        f"{selected_doca.min() * 100:.4f} cm"
    )

    print(
        f"Maximum selected DOCA : "
        f"{selected_doca.max() * 100:.4f} cm"
    )


# ============================================================
# DECAY VESSEL SELECTION
# ============================================================

def select_vertices_in_cone(vertex):

    if len(vertex) == 0:
        return np.zeros(0, dtype=bool)

    z = vertex[:, 2]

    x_limit = (
        X_VESSEL_MIN
        + (X_VESSEL_MAX - X_VESSEL_MIN)
        * (z - Z_VESSEL_MIN)
        / (Z_VESSEL_MAX - Z_VESSEL_MIN)
    )

    y_limit = (
        Y_VESSEL_MIN
        + (Y_VESSEL_MAX - Y_VESSEL_MIN)
        * (z - Z_VESSEL_MIN)
        / (Z_VESSEL_MAX - Z_VESSEL_MIN)
    )

    keep = (
        (z >= Z_VESSEL_MIN)
        & (z <= Z_VESSEL_MAX)
        & (np.abs(vertex[:, 0]) <= x_limit)
        & (np.abs(vertex[:, 1]) <= y_limit)
    )

    return keep


mask_vessel = select_vertices_in_cone(
    selected_vertex
)

selected_pairs = selected_pairs[mask_vessel]
selected_doca = selected_doca[mask_vessel]
selected_vertex = selected_vertex[mask_vessel]
selected_momentum = selected_momentum[mask_vessel]

print()
print("============================================================")
print("DECAY VESSEL")
print("============================================================")
print(
    f"Pairs after DOCA      : "
    f"{np.sum(mask_doca):,}"
)
print(
    f"Vertices in vessel    : "
    f"{len(selected_pairs):,}"
)


# ============================================================
# IP EXTRAPOLATION
# ============================================================

if len(selected_vertex) > 0:

    x_ip, y_ip = extrapolate_to_z(
        selected_vertex[:, 0],
        selected_vertex[:, 1],
        selected_vertex[:, 2],
        selected_momentum[:, 0],
        selected_momentum[:, 1],
        selected_momentum[:, 2],
        Z_IP
    )

    ip = np.sqrt(
        x_ip ** 2
        + y_ip ** 2
    )

    mask_ip = ip < IP_MAX

    selected_pairs = selected_pairs[mask_ip]
    selected_doca = selected_doca[mask_ip]
    selected_vertex = selected_vertex[mask_ip]
    selected_momentum = selected_momentum[mask_ip]

    x_ip = x_ip[mask_ip]
    y_ip = y_ip[mask_ip]
    ip = ip[mask_ip]

else:

    x_ip = np.empty(0)
    y_ip = np.empty(0)
    ip = np.empty(0)


print()
print("============================================================")
print("IP SELECTION")
print("============================================================")
print(
    f"Vertices before IP cut : "
    f"{np.sum(mask_vessel):,}"
)
print(
    f"IP < {IP_MAX:.2f} m       : "
    f"{len(selected_pairs):,}"
)


# ============================================================
# FINAL PAIR WEIGHTS
# ============================================================

if len(selected_pairs) > 0:

    selected_pair_weight = np.prod(
        data_planes["weight"][selected_pairs],
        axis=1
    )

else:

    selected_pair_weight = np.empty(0)


# ============================================================
# FINAL NORMALIZATION
# ============================================================

NORMALIZATION_15 = (
    6e20 / 5e13
)


total_pair_rate = (
    selected_pair_weight.sum()
)


expected_15_years = (
    NORMALIZATION_15
    * total_pair_rate
)


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("============================================================")
print("FINAL RESULT")
print("============================================================")

print(
    f"Unique events                  : "
    f"{n_unique:,}"
)

print(
    f"Surviving single muons         : "
    f"{len(data_planes['x']):,}"
)

print(
    f"Final opposite-charge pairs    : "
    f"{len(selected_pairs):,}"
)

print(
    f"Sum of pair weights            : "
    f"{total_pair_rate:.6e}"
)

print(
    f"15-year normalization          : "
    f"{NORMALIZATION_15:.6e}"
)

print(
    f"Expected 15-year background    : "
    f"{expected_15_years:.6e}"
)

print("============================================================")


# ============================================================
# OPTIONAL: STORE FINAL INFORMATION IN MEMORY
#
# Nothing is saved to disk.
# ============================================================

final_results = {
    "pairs": selected_pairs,
    "doca": selected_doca,
    "vertex": selected_vertex,
    "momentum": selected_momentum,
    "x_ip": x_ip,
    "y_ip": y_ip,
    "ip": ip,
    "pair_weight": selected_pair_weight,
    "unique_events": n_unique,
    "normalization_15_years": NORMALIZATION_15,
    "total_pair_rate": total_pair_rate,
    "expected_15_years": expected_15_years,
}