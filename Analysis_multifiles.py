import glob
import pickle
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle


# ============================================================
# 1. CONFIGURATION
# ============================================================

def get_config():

    config = {

        # ----------------------------------------------------
        # Input
        # ----------------------------------------------------
        "input_pattern": "outputs/TRY6_UBT_T1_seed_*.pkl",

        # ----------------------------------------------------
        # General
        # ----------------------------------------------------
        "check": True,

        # ----------------------------------------------------
        # Single-muon cuts
        # ----------------------------------------------------
        "p_min": 1.0,                 # GeV/c

        "z_timing": 94.850 - 2.14,    # m

        "timing_x_max": 2.0,          # m
        "timing_y_max": 3.0,          # m

        # ----------------------------------------------------
        # Pair cuts
        # ----------------------------------------------------
        "time_window": 200e-12,        # s
        "use_time_cut": True,
        "opposite_charge": True,

        "doca_max": 0.02,             # m

        # ----------------------------------------------------
        # Decay vessel
        # ----------------------------------------------------
        "z_decay_min": 30.6,
        "z_decay_max": 80.6,

        "x_decay_min": 0.5,
        "x_decay_max": 2.0,

        "y_decay_min": 1.35,
        "y_decay_max": 3.0,

        # ----------------------------------------------------
        # IP
        # ----------------------------------------------------
        "z_ip": -2.14,
        "ip_max": 2.5,

        # ----------------------------------------------------
        # Normalization
        # ----------------------------------------------------
        "pot_spill": 5e13,
        "pot_15_years": 6e20,

        
    }

    return config


# ============================================================
# 2. LOAD ONE FILE
# ============================================================

def load_file(filename):

    print(f"Loading {filename}")

    with open(filename, "rb") as f:
        data = pickle.load(f)

    plane = data["planes"][2]

    result = {

        "x": np.asarray(plane["x"]),
        "y": np.asarray(plane["y"]),
        "z": np.asarray(plane["z"]),

        "px": np.asarray(plane["px"]),
        "py": np.asarray(plane["py"]),
        "pz": np.asarray(plane["pz"]),

        "weight": np.asarray(data["weight"]),
        "evt_n": np.asarray(data["evt_n"]),
        "pdg": np.asarray(data["pdg"]),
    }

    return result


# ============================================================
# 3. LOAD ALL FILES
# ============================================================

def load_all_files(pattern):

    files = glob.glob(pattern)

    if not files:
        raise RuntimeError(
            f"No files found with pattern:\n{pattern}"
        )

    files = sorted(
        files,
        key=lambda f: int(
            re.search(r"seed_(\d+)", f).group(1)
        )
    )

    print()
    print("=" * 70)
    print("INPUT")
    print("=" * 70)

    print(f"Found {len(files)} files")

    arrays = {
        "x": [],
        "y": [],
        "z": [],
        "px": [],
        "py": [],
        "pz": [],
        "weight": [],
        "evt_n": [],
        "pdg": [],
    }

    for filename in files:

        data = load_file(filename)

        for key in arrays:
            arrays[key].append(data[key])

    for key in arrays:
        arrays[key] = np.concatenate(arrays[key])

    print(
        f"Total muons: {len(arrays['x']):,}"
    )

    return arrays, files


# ============================================================
# 4. EXTRAPOLATION
# ============================================================

def extrapolate_to_z(
    x,
    y,
    z,
    px,
    py,
    pz,
    z_target
):

    dz = z_target - z

    with np.errstate(
        divide="ignore",
        invalid="ignore"
    ):

        x_target = (
            x + px / pz * dz
        )

        y_target = (
            y + py / pz * dz
        )

    return x_target, y_target


# ============================================================
# 5. ADD DERIVED QUANTITIES
# ============================================================

def prepare_muons(data, config, n_files):

    data = {
        key: np.asarray(value).copy()
        for key, value in data.items()
    }

    # Momentum
    data["p"] = np.sqrt(
        data["px"]**2
        + data["py"]**2
        + data["pz"]**2
    )

    # --------------------------------------------------------
    # Timing detector
    # --------------------------------------------------------

    (
        data["x_timing"],
        data["y_timing"]
    ) = extrapolate_to_z(

        data["x"],
        data["y"],
        data["z"],

        data["px"],
        data["py"],
        data["pz"],

        config["z_timing"]
    )

    # --------------------------------------------------------
    # Decay-vessel entrance
    # --------------------------------------------------------

    (
        data["x_at_30"],
        data["y_at_30"]
    ) = extrapolate_to_z(

        data["x"],
        data["y"],
        data["z"],

        data["px"],
        data["py"],
        data["pz"],

        config["z_decay_min"]
    )

    # --------------------------------------------------------
    # Random time
    # --------------------------------------------------------

    rng = np.random.default_rng()

    data["time"] = rng.uniform(
        0,
        1,
        len(data["x"])
    )

    # --------------------------------------------------------
    # Corrected weight
    # --------------------------------------------------------

    unique_evt, counts = np.unique(
        data["evt_n"],
        return_counts=True
    )

    count_dict = dict(
        zip(unique_evt, counts)
    )

    n_repeated = np.array([
        count_dict[event]
        for event in data["evt_n"]
    ])

    total_rate = data["weight"].sum()

    corrected = (
        data["weight"]
        / n_repeated
    )
    corrected = corrected * (total_rate/n_files) /corrected.sum()

    data["weight"] = corrected

    return data


# ============================================================
# 6. APPLY SINGLE-MUON CUTS
# ============================================================

def apply_single_muon_cuts(data, config):

    n = len(data["x"])

    alive = np.ones(
        n,
        dtype=bool
    )

    cut_masks = {}

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    mask = (
        data["p"]
        > config["p_min"]
    )

    cut_masks["Momentum"] = mask

    alive &= mask

    # --------------------------------------------------------
    # Timing detector
    # --------------------------------------------------------

    mask = (

        np.abs(data["x_timing"])
        < config["timing_x_max"]

    ) & (

        np.abs(data["y_timing"])
        < config["timing_y_max"]

    )

    cut_masks["Timing detector"] = mask

    alive &= mask

    return alive, cut_masks


# ============================================================
# 7. PRINT SINGLE-MUON CUT FLOW
# ============================================================
def print_single_cutflow(
    cut_masks,
    n_total,
    weights
):

    print()
    print("=" * 80)
    print("SINGLE-MUON CUT FLOW")
    print("=" * 80)

    previous = np.ones(
        n_total,
        dtype=bool
    )

    for name, mask in cut_masks.items():

        before = previous.sum()

        after = (
            previous & mask
        ).sum()

        rejected = before - after

        efficiency = (
            100 * after / before
            if before > 0
            else 0
        )

        weighted_before = weights[previous].sum()

        weighted_after = (
            weights[previous & mask].sum()
        )

        weighted_rejected = (
            weighted_before - weighted_after
        )

        weighted_efficiency = (
            100
            * weighted_after
            / weighted_before
            if weighted_before > 0
            else 0
        )

        print()
        print(name)

        print(
            f"  before              : {before:,}"
        )

        print(
            f"  after               : {after:,}"
        )

        print(
            f"  rejected            : {rejected:,}"
        )

        print(
            f"  efficiency          : "
            f"{efficiency:.4f} %"
        )

        print(
            f"  weight sum before   : "
            f"{weighted_before:.6f}"
        )

        print(
            f"  weight sum after    : "
            f"{weighted_after:.6f}"
        )

        print(
            f"  weight sum rejected : "
            f"{weighted_rejected:.6f}"
        )

        print(
            f"  weighted efficiency : "
            f"{weighted_efficiency:.4f} %"
        )

        previous &= mask

# ============================================================
# 8. APPLY SINGLE-MUON MASK
# ============================================================

def select_muons(data, mask):

    return {
        key: value[mask]
        for key, value in data.items()
    }


# ============================================================
# 9. FIND TIME PAIRS
# ============================================================
def find_time_pairs(data, half_window=None):

    n = len(data["x"])

    # --------------------------------------------------------
    # No time cut: return every unique pair
    # --------------------------------------------------------

    if half_window is None:

        i, j = np.triu_indices(
            n,
            k=1
        )

        return np.column_stack([
            i,
            j
        ])

    # --------------------------------------------------------
    # Apply time-window cut
    # --------------------------------------------------------

    time = data["time"]

    order = np.argsort(time)

    time_sorted = time[order]

    pairs = []

    for i in range(len(time_sorted)):

        left = np.searchsorted(
            time_sorted,
            time_sorted[i] - half_window,
            side="left"
        )

        right = np.searchsorted(
            time_sorted,
            time_sorted[i] + half_window,
            side="right"
        )

        for j in range(
            max(i + 1, left),
            right
        ):

            pairs.append([
                order[i],
                order[j]
            ])

    if not pairs:

        return np.empty(
            (0, 2),
            dtype=int
        )

    return np.asarray(
        pairs,
        dtype=int
    )


# ============================================================
# 10. PAIR QUANTITIES
# ============================================================

def calculate_doca_and_vertex(
    data,
    pairs
):

    if len(pairs) == 0:

        return (
            np.array([]),
            np.empty((0, 3)),
            np.empty((0, 3))
        )

    r = np.column_stack([
        data["x"],
        data["y"],
        data["z"]
    ])

    p = np.column_stack([
        data["px"],
        data["py"],
        data["pz"]
    ])

    r1 = r[pairs[:, 0]]
    r2 = r[pairs[:, 1]]

    p1 = p[pairs[:, 0]]
    p2 = p[pairs[:, 1]]

    dr = r1 - r2

    cross = np.cross(
        p1,
        p2
    )

    cross_norm = np.linalg.norm(
        cross,
        axis=1
    )

    doca = np.full(
        len(pairs),
        np.inf
    )

    valid = cross_norm > 1e-12

    doca[valid] = (
        np.abs(
            np.sum(
                dr[valid] * cross[valid],
                axis=1
            )
        )
        / cross_norm[valid]
    )

    # Closest points
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
        p1 * dr,
        axis=1
    )

    e = np.sum(
        p2 * dr,
        axis=1
    )

    denominator = (
        a * c - b * b
    )

    valid &= (
        np.abs(denominator)
        > 1e-12
    )

    s = np.zeros(len(pairs))
    t = np.zeros(len(pairs))

    s[valid] = (
        b[valid] * e[valid]
        - c[valid] * d[valid]
    ) / denominator[valid]

    t[valid] = (
        a[valid] * e[valid]
        - b[valid] * d[valid]
    ) / denominator[valid]

    point1 = (
        r1
        + s[:, None] * p1
    )

    point2 = (
        r2
        + t[:, None] * p2
    )

    vertex = (
        point1 + point2
    ) / 2

    momentum = p1 + p2

    return (
        doca,
        vertex,
        momentum
    )


# ============================================================
# 11. PAIR CUT FLOW
# ============================================================

def analyse_pairs(
    data,
    config
):

    if config["use_time_cut"]:

        pairs = find_time_pairs(
            data,
            config["time_window"]
        )

    else:

        pairs = find_time_pairs(
            data,
            None
        )

    print()
    print("=" * 80)
    print("PAIR SELECTION")
    print("=" * 80)

    print(
        f"Time-compatible pairs: "
        f"{len(pairs):,}"
    )

    if len(pairs) == 0:
        return None

    pair_weights = np.prod(
        data["weight"][pairs],
        axis=1
    )

    # --------------------------------------------------------
    # Pair cut 1: charge
    # --------------------------------------------------------

    if config["opposite_charge"]:

        charge_mask = (
            data["pdg"][pairs[:, 0]]
            *
            data["pdg"][pairs[:, 1]]
            < 0
        )

    else:

        charge_mask = np.ones(
            len(pairs),
            dtype=bool
        )

    pairs = pairs[charge_mask]
    pair_weights = pair_weights[charge_mask]

    print()
    print("Opposite charge")
    print(
        f"  pairs: {len(pairs):,}"
    )

    # --------------------------------------------------------
    # DOCA
    # --------------------------------------------------------

    doca, vertex, momentum = (
        calculate_doca_and_vertex(
            data,
            pairs
        )
    )

    doca_mask = (
        doca
        < config["doca_max"]
    )

    pairs = pairs[doca_mask]
    pair_weights = pair_weights[doca_mask]
    vertex = vertex[doca_mask]
    momentum = momentum[doca_mask]
    doca = doca[doca_mask]

    print()
    print(
        f"DOCA < "
        f"{config['doca_max'] * 100:.2f} cm"
    )

    print(
        f"  pairs: {len(pairs):,}"
    )

    # --------------------------------------------------------
    # Decay vessel
    # --------------------------------------------------------

    z = vertex[:, 2]

    x_limit = (
        config["x_decay_min"]
        +
        (
            config["x_decay_max"]
            - config["x_decay_min"]
        )
        *
        (
            z
            - config["z_decay_min"]
        )
        /
        (
            config["z_decay_max"]
            - config["z_decay_min"]
        )
    )

    y_limit = (
        config["y_decay_min"]
        +
        (
            config["y_decay_max"]
            - config["y_decay_min"]
        )
        *
        (
            z
            - config["z_decay_min"]
        )
        /
        (
            config["z_decay_max"]
            - config["z_decay_min"]
        )
    )

    decay_mask = (

        (z >= config["z_decay_min"])
        &
        (z <= config["z_decay_max"])

        &
        (
            np.abs(vertex[:, 0])
            <= x_limit
        )

        &
        (
            np.abs(vertex[:, 1])
            <= y_limit
        )
    )

    pairs = pairs[decay_mask]
    pair_weights = pair_weights[decay_mask]
    vertex = vertex[decay_mask]
    momentum = momentum[decay_mask]
    doca = doca[decay_mask]

    print()
    print("Decay vessel")

    print(
        f"  pairs: {len(pairs):,}"
    )

    # --------------------------------------------------------
    # IP
    # --------------------------------------------------------

    x_ip, y_ip = extrapolate_to_z(

        vertex[:, 0],
        vertex[:, 1],
        vertex[:, 2],

        momentum[:, 0],
        momentum[:, 1],
        momentum[:, 2],

        config["z_ip"]
    )

    ip = np.sqrt(
        x_ip**2
        + y_ip**2
    )

    ip_mask = (
        ip < config["ip_max"]
    )

    pairs = pairs[ip_mask]
    pair_weights = pair_weights[ip_mask]
    vertex = vertex[ip_mask]
    momentum = momentum[ip_mask]
    doca = doca[ip_mask]
    ip = ip[ip_mask]

    print()
    print(
        f"IP < {config['ip_max']:.2f} m"
    )

    print(
        f"  pairs: {len(pairs):,}"
    )

    return {

        "pairs": pairs,

        "weights": pair_weights,

        "doca": doca,

        "vertex": vertex,

        "momentum": momentum,

        "ip": ip,

        "x_ip": x_ip[ip_mask],

        "y_ip": y_ip[ip_mask],
    }


# ============================================================
# 12. PLOT FINAL SAMPLE
# ============================================================

def make_plots(
    data,
    result,
    config
):

    if result is None:
        return

    # --------------------------------------------------------
    # DOCA
    # --------------------------------------------------------

    plt.figure()

    plt.hist(
        result["doca"] * 100,
        bins=100,
        histtype="step",
        linewidth=2
    )

    plt.xlabel("DOCA [cm]")
    plt.ylabel("Pairs")

    plt.grid(alpha=0.2)

    plt.show()

    # --------------------------------------------------------
    # Vertex
    # --------------------------------------------------------

    if len(result["vertex"]) > 0:

        plt.figure()

        plt.hist2d(
            result["vertex"][:, 0],
            result["vertex"][:, 1],
            bins=100,
            norm=LogNorm()
        )

        plt.xlabel("Vertex x [m]")
        plt.ylabel("Vertex y [m]")

        plt.colorbar()

        plt.show()

        # ----------------------------------------------------
        # Vertex z
        # ----------------------------------------------------

        plt.figure()

        plt.hist(
            result["vertex"][:, 2],
            bins=100,
            histtype="step",
            linewidth=2
        )

        plt.xlabel("Vertex z [m]")
        plt.ylabel("Pairs")

        plt.show()

    # --------------------------------------------------------
    # IP
    # --------------------------------------------------------

    plt.figure()

    plt.hist(
        result["ip"],
        bins=100,
        histtype="step",
        linewidth=2
    )

    plt.axvline(
        config["ip_max"],
        linestyle="--",
        label="IP cut"
    )

    plt.xlabel("IP [m]")
    plt.ylabel("Pairs")

    plt.legend()

    plt.show()

    # --------------------------------------------------------
    # IP x-y
    # --------------------------------------------------------

    if len(result["ip"]) > 0:

        plt.figure()

        plt.hist2d(
            result["x_ip"],
            result["y_ip"],
            bins=100,
            norm=LogNorm()
        )

        circle = Circle(
            (0, 0),
            config["ip_max"],
            fill=False,
            linestyle="--",
            linewidth=2
        )

        plt.gca().add_patch(circle)

        plt.xlabel("x at IP plane [m]")
        plt.ylabel("y at IP plane [m]")

        plt.axis("equal")

        plt.colorbar()

        plt.show()


# ============================================================
# 13. FINAL RESULT
# ============================================================

def print_final_result(
    result,
    config
):

    print()
    print("=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    if result is None:

        print("No pairs survived.")

        return

    weights = result["weights"]

    rate = weights.sum()

    normalization_15 = (
        config["pot_15_years"]
        / config["pot_spill"]
    )

    expected = (
        rate
        * normalization_15
    )

    print(
        f"Final pairs                 : "
        f"{len(result['pairs']):,}"
    )

    print(
        f"Final weighted rate         : "
        f"{rate:.6e}"
    )

    print(
        f"15-year normalization       : "
        f"{normalization_15:.6e}"
    )

    print(
        f"Expected 15-year background: "
        f"{expected:.6e}"
    )


# ============================================================
# 14. MAIN
# ============================================================

def main():

    # ========================================================
    # CHANGE YOUR CUTS HERE
    # ========================================================

    config = get_config()

    # Examples:
    #
    # config["p_min"] = 2.0
    #
    # config["timing_x_max"] = 1.5
    #
    # config["timing_y_max"] = 1.5
    #
    # config["time_window"] = 50e-9
    #
    # config["doca_max"] = 0.01
    #
    # config["ip_max"] = 1.0
    #
    # config["opposite_charge"] = False

    # config["use_time_cut"] = False
    # config["time_window"] = 200e-9

    # ========================================================
    # LOAD
    # ========================================================

    raw_data, files = load_all_files(
        config["input_pattern"]
    )

    # ========================================================
    # PREPARE
    # ========================================================

    data = prepare_muons(
        raw_data,
        config,
        len(files)
    )

    # ========================================================
    # SINGLE MUON CUTS
    # ========================================================

    muon_mask, cut_masks = (
        apply_single_muon_cuts(
            data,
            config
        )
    )

    print_single_cutflow(
        cut_masks,
        len(data["x"]),
        data["weight"]
    )

    # ========================================================
    # APPLY SINGLE-MUON SELECTION
    # ========================================================

    data_selected = select_muons(
        data,
        muon_mask
    )

    print()
    print(
        f"Muons after single-muon cuts: "
        f"{len(data_selected['x']):,}"
    )

    # ========================================================
    # PAIR ANALYSIS
    # ========================================================

    result = analyse_pairs(
        data_selected,
        config
    )

    # ========================================================
    # PLOTS
    # ========================================================

    if config["check"]:

        make_plots(
            data_selected,
            result,
            config
        )

    # ========================================================
    # FINAL
    # ========================================================

    print_final_result(
        result,
        config
    )

    return data_selected, result, config


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    data, result, config = main()