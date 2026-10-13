import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

plt.rcParams['figure.figsize'] = (9, 9)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['font.size'] = 18

# ============================================================
# test symmetry
# ============================================================

def test_symmetry(data):

    x = np.asarray(data["x"])
    y = np.asarray(data["y"])
    w = np.asarray(data["weight"])

    # ------------------------------------------------------------
    # Weighted quadrant populations
    # ------------------------------------------------------------

    Q1 = np.sum(w[(x >= 0) & (y >= 0)])
    Q2 = np.sum(w[(x <  0) & (y >= 0)])
    Q3 = np.sum(w[(x <  0) & (y <  0)])
    Q4 = np.sum(w[(x >= 0) & (y <  0)])

    quadrants = np.array([Q1, Q2, Q3, Q4])

    print("Weighted quadrant rates:")
    print(f"Q1 (+x,+y): {Q1:.8g}")
    print(f"Q2 (-x,+y): {Q2:.8g}")
    print(f"Q3 (-x,-y): {Q3:.8g}")
    print(f"Q4 (+x,-y): {Q4:.8g}")

    print("\nRelative to Q1:")
    print(f"Q2/Q1 = {Q2/Q1:.6f}")
    print(f"Q3/Q1 = {Q3/Q1:.6f}")
    print(f"Q4/Q1 = {Q4/Q1:.6f}")



    # ------------------------------------------------------------
    # 2D weighted distribution
    # ------------------------------------------------------------

    nbins = 200

    x_edges = np.linspace(np.min(x), np.max(x), nbins + 1)
    y_edges = np.linspace(np.min(y), np.max(y), nbins + 1)

    H, _, _ = np.histogram2d(
        x,
        y,
        bins=[x_edges, y_edges],
        weights=w
    )

    # ------------------------------------------------------------
    # X symmetry
    # ------------------------------------------------------------

    # Mirror H in x
    H_xmirror = H[::-1, :]

    x_difference = np.abs(H - H_xmirror)

    x_denominator = H + H_xmirror

    x_relative_difference = np.divide(
        x_difference,
        x_denominator,
        out=np.zeros_like(x_difference),
        where=x_denominator > 0
    )

    # ------------------------------------------------------------
    # Y symmetry
    # ------------------------------------------------------------

    H_ymirror = H[:, ::-1]

    y_difference = np.abs(H - H_ymirror)

    y_denominator = H + H_ymirror

    y_relative_difference = np.divide(
        y_difference,
        y_denominator,
        out=np.zeros_like(y_difference),
        where=y_denominator > 0
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    occupied_x = x_denominator > 0
    occupied_y = y_denominator > 0

    print("\nSymmetry diagnostics")

    print(
        "X symmetry mean relative difference:",
        np.mean(x_relative_difference[occupied_x])
    )

    print(
        "X symmetry median relative difference:",
        np.median(x_relative_difference[occupied_x])
    )

    print(
        "Y symmetry mean relative difference:",
        np.mean(y_relative_difference[occupied_y])
    )

    print(
        "Y symmetry median relative difference:",
        np.median(y_relative_difference[occupied_y])
    )



# ============================================================
# extrapolate_to_z
# ============================================================

def extrapolate_to_z(x, y, z, px, py, pz, z_target):
    """
    Extrapolate particle positions to z_target assuming straight-line motion.

    Returns
    -------
    x_target, y_target : np.ndarray
        x and y coordinates at z = z_target.
    """
    dz = z_target - z

    x_target = x + (px / pz) * dz
    y_target = y + (py / pz) * dz

    return x_target, y_target



# ============================================================
# Function to load one simulation file
# ============================================================

def load_simulation_file(filename):

    print(f"Loading: {filename}")

    with open(filename, "rb") as f:
        data = pickle.load(f)

    weight = np.asarray(data["weight"])

    x = np.asarray(data["x"])
    y = np.asarray(data["y"])
    try:
        z = np.asarray(data["z"])
    except:
        z = np.zeros_like(x)

    px = np.asarray(data["px"])
    py = np.asarray(data["py"])
    pz = np.asarray(data["pz"])

    # ----------------------------------------------------
    # Check lengths
    # ----------------------------------------------------

    arrays = {
        "x": x,
        "y": y,
        "z": z,
        "px": px,
        "py": py,
        "pz": pz,
        "weight": weight,
    }

    lengths = {key: len(value) for key, value in arrays.items()}

    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"Inconsistent array lengths in {filename}: {lengths}"
        )

    return arrays


# ============================================================
# Load all simulation outputs
# ============================================================

filename = "/home/hep/gfrise/work_dir/MuonsAndMatter/data/outputs/outputs_cuda.pkl"

data = load_simulation_file(filename)

# ============================================================
# Configuration
# ============================================================

x_boundary = -4.56
y_boundary = -3.36

dx = 0.1
dy = 0.1
dz = 0.5

x_min, x_max = -4.5, 4.5
y_min, y_max = -3.3, 3.3
z_min, z_max = 30.6, 80.6+0.5

# ============================================================
# Extract arrays and keep only y < 0
# ============================================================

y_all = np.asarray(data["y"], dtype=float)

mask_y = y_all <= 0


print(f"Total particles:       {len(y_all):,}")
print(f"Particles with y < 0:  {mask_y.sum():,}")

x = np.asarray(data["x"][mask_y], dtype=float)
y = y_all[mask_y]

z = np.asarray(data["z"][mask_y], dtype=float)

px = np.asarray(data["px"][mask_y], dtype=float)
py = np.asarray(data["py"][mask_y], dtype=float)
pz = np.asarray(data["pz"][mask_y], dtype=float)

weight = np.asarray(data["weight"][mask_y], dtype=float)

# ============================================================
# 4. Extrapolate to z_end
# ============================================================

x_end, y_end = extrapolate_to_z(
    x, y, z, px, py, pz, z_max
)

crosses_x = x_end <= x_boundary
crosses_y = y_end <= y_boundary

discarded = crosses_x | crosses_y

N_discarded = np.count_nonzero(discarded)
N_surviving = np.count_nonzero(~discarded)

W_discarded = np.sum(weight[discarded])
W_surviving = np.sum(weight[~discarded])
W_initial = np.sum(weight)

print("\n========== ACCEPTANCE ==========")
print(f"Initially accepted:    {len(x):,}")
print(f"Discarded at boundary: {N_discarded:,}")
print(f"Surviving to z_end:    {N_surviving:,}")

print("\n========== WEIGHTED ==========")
print(f"Initial weight:        {W_initial:.8g}")
print(f"Discarded weight:      {W_discarded:.8g}")
print(f"Surviving weight:      {W_surviving:.8g}")

if W_initial != 0:
    print(f"Weighted survival:     {W_surviving / W_initial:.6%}")

# ============================================================
# Bin edges
# ============================================================


# Bin edges
x_edges = np.arange(x_min, x_max + dx, dx)
y_edges = np.arange(y_min, y_max + dy, dy)

z_edges = np.arange(z_min, z_max, dz)
if z_edges[-1] < z_max:
    z_edges = np.append(z_edges, z_max)

print("\n========== BINNING ==========")
print("Nx =", len(x_edges) - 1)
print("Ny =", len(y_edges) - 1)
print("Nz =", len(z_edges) - 1)

print("x range:", x_edges[0], "->", x_edges[-1])
print("y range:", y_edges[0], "->", y_edges[-1])
print("z range:", z_edges[0], "->", z_edges[-1])

# ============================================================
# Build 3D histogram
# ============================================================

H = np.zeros(
    (
        len(x_edges) - 1,
        len(y_edges) - 1,
        len(z_edges) - 1
    ),
    dtype=np.float64
)
print(z_edges)

# ============================================================
# Loop over z bins
# ============================================================

for iz in range(len(z_edges) - 1):

    # Use the centre of the z bin
    z_plane = z_edges[iz]

    print(
        f"z plane: {z_plane}"
        )

    # Propagate particles to this z
    x_plane, y_plane = extrapolate_to_z(
        x, y, z, px, py, pz, z_plane
    )

    # Only y < 0 representation
    mask = (
        (x_plane >= x_min) &
        (x_plane < x_max) &
        (y_plane >= y_min) &
        (y_plane < y_max)
    )

    H[:, :, iz], _, _ = np.histogram2d(
        x_plane[mask],
        y_plane[mask],
        bins=[x_edges, y_edges],
        weights=weight[mask]
    )

# ============================================================
# Symmetry
# ============================================================

H_full = H.copy()

ny = H.shape[1]

# Negative-y bins are the first half.
# Mirror them into the positive-y half.
H_full[:, :, :] = 0.0

mid = ny // 2

H_full[:, :mid, :] = H[:, :mid, :]
H_full[:, mid:, :] = H[:, :mid, :][:, ::-1, :]

# ============================================================
# Verify symmetry
# ============================================================

negative_part = H_full[:, :mid, :]
positive_part = H_full[:, mid:, :]

symmetry_difference = np.max(
    np.abs(
        negative_part -
        positive_part[:, ::-1, :]
    )
)

print("\n========== Y SYMMETRY CHECK ==========")
print("Maximum absolute bin difference:",
      symmetry_difference)

print("Negative-y weight:",
      negative_part.sum())

print("Positive-y weight:",
      positive_part.sum())

print("Full distribution weight:",
      H_full.sum())

print("Expected full weight:",
      2.0 * H[:, :mid, :].sum())

# ============================================================
# Plot before and after mirroring
# ============================================================
fig, axes = plt.subplots(
    1, 2,
    figsize=(18, 10),
    constrained_layout=True
)

# ============================================================
# BEFORE
# ============================================================

ax = axes[0]

mesh = ax.pcolormesh(
    x_edges,
    y_edges,
    H[:, :, 0].T,
    shading="auto",
    norm = LogNorm()
)

# ax.axhline(0, color="black", linewidth=1)

ax.set_title(
    f"Before y → -y copy, z = {z_edges[0]:.2f}"
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_aspect("equal")


# ============================================================
# AFTER
# ============================================================

ax = axes[1]

mesh = ax.pcolormesh(
    x_edges,
    y_edges,
    H_full[:, :, 0].T,
    shading="auto",
    norm = LogNorm()
)

# ax.axhline(0, color="black", linewidth=1)

ax.set_title(
    f"After y → -y copy, z = {z_edges[0]:.2f}"
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_aspect("equal")

fig.colorbar(
    mesh,
    ax=axes,
    label="Weighted counts"
)

plt.savefig("temp2.png")
plt.show()

# ============================================================
# Save full distribution
# ============================================================

distribution = {
    "H": H_full,
    "x_edges": x_edges,
    "y_edges": y_edges,
    "z_edges": z_edges,
    "dx": dx,
    "dy": dy,
    "dz": dz,
    "y_symmetry": True,
    "y_range_stored": (y_min, y_max),
}

output_file = "muon_distribution_3d.pkl"

with open(output_file, "wb") as f:
    pickle.dump(
        distribution,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )

print("\n========== SAVED ==========")
print(f"File: {output_file}")
print("Histogram shape:", H_full.shape)