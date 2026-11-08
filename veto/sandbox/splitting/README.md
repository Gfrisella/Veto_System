# Splitting prototype (sandbox: safe to delete)

Adaptive, weighted particle splitting for the SHiP muon shield with the existing fast GPU transport (MuonsAndMatter
`propagate_muons_with_cuda`, through `bin/generate_muon_study.Simulation`): no generative models.

Everything here is a prototype. Delete this folder and `/disk/gfs_lhcb/sqasim/splitting_proto/` (farm outputs) at any time;
nothing else depends on them.

## Method (`splitting.py`)

- **Checkpoints** (z, m): the downstream end of every shield piece (TRY6: 2.30, 5.45, 10.10, 14.76, 19.41, 22.19, 25.70,
  29.89), the stored state plane 30.6, then 40, 50, 60, 70 and the SST plane 82. Every surviving muon is paused there with
  its full state (x, y, z, px, py, pz, charge), weight, and genealogy (root = source muon, parent, own id, split depth,
  batch). Continuations restart from that exact state; the physics between checkpoints is the existing transport, untouched.
- **Population control at each checkpoint** (unbiased for any choice of cells): muons are binned by charge, momentum (log
  bins) and the straight-line predicted (x, y) at z = 82 m. With n_c muons in cell c and a target T per cell:
  - n_c < T: each muon becomes s copies, E[s] = T / n_c (at most S_MAX), weight / s each: independent continuations (copies
    sit at different indices of the GPU batch, so they get independent random streams);
  - n_c > T: Russian roulette, kept with p = max(K_MIN, T / n_c), weight / p (K_MIN keeps common trajectories sampled);
  - a global cap N_MAX per batch (uniform roulette) keeps the budget fixed.
- **Scoring** at every checkpoint (after transport, before control): weighted x-y, momentum and angle histograms (sum w,
  sum w^2, rows) per batch; stopping points; at the SST plane the full particle list (state, weight, root, depth, batch).
- **Uncertainties**: sources are split into independent batches (different seeds and source muons): batch means give
  errors that include the correlation between branches of one source muon; also grouped by root.
- **Analog mode** (`--analog`): the same code without splitting or roulette = conventional Monte Carlo, for validation.

## Files

- `splitting.py`: the method and the driver (`python3 splitting.py --help`)
- `analyse.py`: comparison with analog runs, per-region rates, pulls, distinct-root counts, upper limits on lifetime crossings
- `run_farm.sh`: the farm commands
