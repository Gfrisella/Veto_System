# Changelog

Changes to the veto optimisation code, newest first. Each entry gives the branch, the base commit and
what changed. Open problems are in `ISSUES.md`; running jobs and results are in `STATUS.md`.

## 2026-10-19 — branch `veto-all-sst1` (base `main` @ `dd7c5f6`, not committed yet)

Goal: a veto for every SST1-acceptance muon (not only the muons that make background pairs), with no
timing detector assumed. New pieces: tilted fins and a panel under the vessel.

### Geometry and designs (`lib/veto/design.py`, `lib/veto/muon_types.py`)
- **Tilted fins.** A fin leans downstream (`tilt` > 0) or upstream (< 0) about its inner edge, up to ±75°.
  The point at distance u along the fin is at x = sign·(X(z) + u cos tilt), z + u sin tilt.
  `validate()` cuts the fin so its outer edge stays between the UBT and SST1 (and at the cavern wall).
  Earlier versions raised `NotImplementedError` for a tilted fin.
- **Bottom panel** (optional, one per design): a flat panel under the decay vessel.
  ```
  "bottom": {"z": 55, "length": 3, "x_lo": -1.5, "x_hi": 1.5, "y": -3.3, "pitch": 0, "technology": "straw"}
  ```
  - Limits in `validate_bottom()`:
    - |pitch| ≤ 30°;
    - area ≤ 10 m²;
    - between the floor and 5 cm below the vessel bottom;
    - downstream end at z ≤ 81.5 m;
    - width ≥ 0.2 m.
  - `Plane` gains `kind = "bottom"`, `tilt` (the fin tilt or the panel pitch), `x_lo`, `y0`, and the
    methods `z_at`, `z_local`, `z_end`, `y_end`.
  - `muon_types.bottom_crossings()` intersects the paths A → (K) → B with the panel. A path can cross
    it twice: on the way down and again after a floor bounce.
- **Floor limit:** the UBT and fins may now reach the cavern floor, `FIN_Y_MIN` = −3.36 m (was −3.2 m).
  This is also changed in `bin/make_design_explorer.py` and in the prompt of `evolve/evolve_designs.py`.
- **Fins are attached to the vessel structure** (the side wall and its support frame down to the floor):
  - the inner edge must be at 30.6 ≤ z ≤ 80.6 m, on the vessel (before: up to SST1 at 82 m);
  - `validate()` rejects a fin outside this range;
  - the same range is now used by the `optimize_fins.py` bounds and random fins, the design-explorer
    slider and the evolve prompt;
  - a tilted fin's outer edge may still reach downstream to SST1;
  - in x a fin starts at the vessel wall |x| = X(z) and extends outward only, so no fin is inside the
    decay vessel; `validate()` checks every point of a tilted fin against the vessel wall;
  - the fins of the existing OPT1/OPT2 designs are all on the vessel, so their results stand.
  - The bottom panel stays free-standing, with its own supports.

### Reconstruction (`lib/veto/reco.py`, `bin/run_reco.py`)
- **True crossings** (`true_crossing`) for tilted fins and the bottom panel.
- **Track queries** (`measure`):
  - each query is at the z where the measured track meets the tilted or pitched plane;
  - the search radius along the plane is stretched: r / cos(tilt) on a fin, r / sin(track–panel angle)
    on the bottom panel;
  - new query field `ru`, the radius along u, used in `match`.
- **Rate maps:**
  - tilted fins read the fin maps at the z of each u bin;
  - the bottom panel uses `bottom_rate_map()`, a histogram of the crossings of the muons passing under
    the vessel (`results/reco/bottom_rates.npz`, made by `bin/make_bottom_rates.py`, loaded by
    `run_reco.py` when present), cached in memory.
- **Accidental hits:** their nominal time uses the local z of the hit (`z_local`).
- **Time window:** a fixed window is never smaller than drift + track-time sigma. A smaller one would
  lose the muon's own hits when there is no precise track time.
- **Common random numbers** (`RecoConfig.common_random = True`, the default):
  - per-track efficiency rolls, hit times and track smearing come from a seed built from the tracks,
    `cfg.seed` and the plane geometry (`crn_seed`, `plane_key`);
  - two designs see the same random numbers on the same planes, so their difference is not noise.
  - With `False`, the old shared `rng` is used.

### Scattering (`lib/veto/scattering.py`)
- Multiple scattering is applied at the fins only. It is not modelled for the bottom panel yet.

### Objective (`lib/veto/objective.py`)
- **New default tier-1 goal, `goal = "sst1_muons"`:**
  - N = Σ_k R_k ε_k, the SST1 muons per spill that the veto misses (R_k muons per spill of type k);
  - term log10(1 + N / `sst1_target`) / log10(1 + N_before / `sst1_target`), with `sst1_target` = 1 per spill;
  - the old false-vertex goal is still available as `goal = "pairs"`;
  - with `goal = "pairs"`, the SST1-muon term can be added with weight `w_mu`.
- **Time coincidence** (`coincidence_s`):
  - `None` (the default): no time coincidence, both muons in the same 1 s spill;
  - N_before = candidate pairs per spill × spills in 15 years;
  - a half window dt multiplies this by 2 dt / spill (e.g. 200 ps with the timing detector).
- **Pairable muons:** `pairable()` (p > 1 GeV at SST1, pointing into the timing detector) and
  `pairable_entries()` give per-type efficiencies. The factorised pair background now uses these
  efficiencies when they are present.
- **Weighting:** `eps_weighting = "unweighted"` uses unweighted inefficiencies.
- **New output:**
  - `confusion_matrix` gains an `sst1_muons` block (before/after per spill, by type) and
    `pair_muon_inefficiency`;
  - the background block also gives per-spill numbers;
  - `background = None` is allowed, for fast evaluations with single muons only.
- `sample_stats` also returns `rate_single`, `pairs_per_spill` and `n_single`.

### New scripts (`bin/`)
| script | what it does |
|---|---|
| `validate_design.py` | full-statistics validation: every SST1 muon (3 M per type, weighted), no timing detector by default → `results/validate/<tag>.json` |
| `optimize_fins.py` | evolutionary search of the layout: any number of fins on either side, tilted, any technology, UBT bottom, bottom panel |
| `make_bottom_rates.py` | rate sample of the muons passing under the vessel (z = 30.6–82 m) for the bottom panel |
| `make_event_browser.py`, `event_browser.sh`, `event_browser_template.html` | interactive event browser, one muon at a time by category (rebuild / validate / serve / list) |
| `make_veto_viewer.py` | single-HTML viewer of SST1 muons and designs by category |
| `plot_survivors.py` | where the missed SST1 muons go: side and top views per design |

### Docs and other files
- `ISSUES.txt` is now `ISSUES.md` and has been expanded.
- New `STATUS.md`: handoff between sessions (running jobs, results, pending decisions).
- `README.md` lists the new scripts.
- `Optimization_SBT.ipynb`: dataset path moved to `data/muon_vessel_dataset.npz`.

### Results obtained with this code (details in `STATUS.md`)
- **Search:** `results/optimize_bottom/`, 30 generations, 1715 designs.
- **Best design:** OPT2 (`OPT2_rank1_g29_10`).
- **OPT2 numbers:**
  - missed SST1 muons: 0.74 kHz (OPT1: 1.13 kHz, D6: 2.41 kHz);
  - fake vertices: 2.2e-3 per spill, 2.7e4 in 15 years;
  - signal lost: 4.9 %;
  - cost: 7.6 MCHF.

## 2026-10-15 and earlier — `main` up to `dd7c5f6`
See `git log`. Summary:
- PR #4 `veto-signal-fom`: figure of merit as counts over the 15-year run.
- PR #5 `veto-objective`: the reward for the design optimisation.
- LLM evolution of designs and per-plane random-hit windows.
- Multiple scattering after z = 30.6 m.
- Technologies and costs in every design.
