"""Single-muon and pair selections with cut-flow bookkeeping."""

import numpy as np

from . import geometry as geo
from .pairs import doca_and_vertex
from .tracks import extrapolate, extrapolate_to_z, momentum


# ============================================================
# Cut flow
# ============================================================

def flow_row(name, before, after, weighted_before, weighted_after):
    return {
        "cut": name,
        "before": int(before),
        "after": int(after),
        "weighted_before": float(weighted_before),
        "weighted_after": float(weighted_after),
        "efficiency": after / before if before > 0 else 0.0,
        "weighted_efficiency": weighted_after / weighted_before if weighted_before > 0 else 0.0,
    }


def add_cumulative(flow):
    n0, w0 = flow[0]["before"], flow[0]["weighted_before"]
    for row in flow:
        row["cumulative_efficiency"] = row["after"] / n0 if n0 > 0 else 0.0
        row["cumulative_weighted_efficiency"] = row["weighted_after"] / w0 if w0 > 0 else 0.0
    return flow


def print_flow(title, flow):
    print()
    print(title)
    print("-" * 100)
    print(f"{'cut':<20}{'before':>14}{'after':>14}{'eff':>11}{'w. eff':>11}{'cum. w. eff':>14}{'w. after':>15}")
    for row in flow:
        print(
            f"{row['cut']:<20}{row['before']:>14,}{row['after']:>14,}"
            f"{100 * row['efficiency']:>10.4f}%{100 * row['weighted_efficiency']:>10.4f}%"
            f"{100 * row.get('cumulative_weighted_efficiency', np.nan):>13.4f}%"
            f"{row['weighted_after']:>15.6e}"
        )


# ============================================================
# Single muons
# ============================================================

def single_muon_masks(data, config):
    """Ordered dict of cut name -> mask (momentum, timing-detector acceptance)."""
    x_timing, y_timing = extrapolate(data, config["z_timing"])
    return {
        "Momentum": momentum(data) > config["p_min"],
        "Timing detector": (np.abs(x_timing) < config["timing_x_max"])
                           & (np.abs(y_timing) < config["timing_y_max"]),
    }


def apply_single_muon_cuts(data, config):
    """Returns the combined mask and the sequential cut flow."""
    weights = data["weight"]
    alive = np.ones(len(weights), dtype=bool)
    flow = []
    for name, mask in single_muon_masks(data, config).items():
        after = alive & mask
        flow.append(flow_row(name, alive.sum(), after.sum(), weights[alive].sum(), weights[after].sum()))
        alive = after
    return alive, add_cumulative(flow)


# ============================================================
# Pairs
# ============================================================

def analyse_pairs(data, pairs, config):
    """Opposite charge -> DOCA -> vertex in decay vessel -> IP, starting from candidate pairs."""
    pair_weight = (np.prod(data["weight"][pairs], axis=1) if len(pairs)
                   else np.empty(0))
    result = {"pairs": pairs, "weights": pair_weight}
    flow = [flow_row("All pairs", len(pairs), len(pairs), pair_weight.sum(), pair_weight.sum())]

    def keep(name, mask):
        before_n, before_w = len(result["pairs"]), result["weights"].sum()
        for key, value in result.items():
            result[key] = value[mask]
        flow.append(flow_row(name, before_n, len(result["pairs"]), before_w, result["weights"].sum()))

    if config["opposite_charge"]:
        keep("Opposite charge", data["pdg"][pairs[:, 0]] * data["pdg"][pairs[:, 1]] < 0)
    else:
        keep("Opposite charge", np.ones(len(pairs), dtype=bool))

    doca, vertex, mom = doca_and_vertex(data, result["pairs"])
    result.update(doca=doca, vertex=vertex, momentum=mom)
    keep("DOCA", np.isfinite(doca) & (doca < config["doca_max"]))

    v = result["vertex"]
    keep("Decay vessel", geo.inside_vessel(v[:, 0], v[:, 1], v[:, 2]))

    v, m = result["vertex"], result["momentum"]
    x_ip, y_ip = extrapolate_to_z(v[:, 0], v[:, 1], v[:, 2], m[:, 0], m[:, 1], m[:, 2], config["z_ip"])
    result.update(x_ip=x_ip, y_ip=y_ip, ip=np.sqrt(x_ip ** 2 + y_ip ** 2))
    keep("IP", result["ip"] < config["ip_max"])

    result["pair_flow"] = add_cumulative(flow)
    return result


def expected_background(final_weight, config):
    """Final pair rate per spill scaled to the 15-year POT."""
    normalization = config["pot_15_years"] / config["pot_spill"]
    return normalization, final_weight * normalization


# ============================================================
# Averaging over files (per-file mode)
# ============================================================

def average_flows(flows):
    """Mean and std (ddof=1) of every numeric column across files."""
    averaged = []
    for rows in zip(*flows):
        entry = {"cut": rows[0]["cut"]}
        for key, value in rows[0].items():
            if key == "cut":
                continue
            values = np.array([row[key] for row in rows], dtype=float)
            entry[f"mean_{key}"] = values.mean()
            entry[f"std_{key}"] = values.std(ddof=1) if len(values) > 1 else 0.0
        averaged.append(entry)
    return averaged


def print_average_flow(title, averaged):
    print()
    print(title)
    print("-" * 100)
    for row in averaged:
        print(f"{row['cut']:<22} eff = {100 * row['mean_efficiency']:.4f} +/- {100 * row['std_efficiency']:.4f} %"
              f"   weighted eff = {100 * row['mean_weighted_efficiency']:.4f} +/- "
              f"{100 * row['std_weighted_efficiency']:.4f} %")
        if "mean_cumulative_weighted_efficiency" in row:
            print(f"{'':22} cumulative weighted eff = {100 * row['mean_cumulative_weighted_efficiency']:.4f}"
                  f" +/- {100 * row['std_cumulative_weighted_efficiency']:.4f} %")
