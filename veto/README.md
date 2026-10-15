# veto — muon background for the SHiP veto system

A cleaned-up version of the analysis in the repository root (`analysis.ipynb`,
`analysis_multifiles.ipynb`, `Optimization_SBT.ipynb`, `Analysis_*.py`,
`Muon_Flames_rate.py`, `temp1.py`). Shared code is in `lib/veto`; every notebook
or script became a command-line program in `bin/`. The original files in the
root are left as they were.

```
veto/
  lib/veto/
    geometry.py    z planes, decay vessel, UBT/SBT, SST1, timing plane, spectrometer, TRY6 shield
    config.py      default cuts (p > 1 GeV, timing acceptance, DOCA < 2 cm, IP < 2.5 m, ...)
    paths.py       where inputs are read from / results written to
    io.py          output pickles, vessel dataset, 3D distribution, field map, column stores,
                   memory-mapped columns of large pickles (open_pickle_columns)
    muon_types.py  muon types from the z = 30.6 / 82 m states, bounce points, SBT wall crossings,
                   path position at any z, fin hits
    design.py      veto designs (UBT + fins) from designs/*.json, physical limits
    technology.py  detector technologies (straw, fibre, pixel, tile, tile2): layers, timing, rates, material, cost
    reco.py        digitization, accidental hits from rate maps, spectrometer, veto algorithm A
    tracks.py      straight-line extrapolation, back-tracking to the vessel walls, categories
    weights.py     repeated-evt_n weight corrections
    pairs.py       all pairs / time-window pairs / KD-tree candidates, DOCA + vertex
    selection.py   single-muon and pair cut flows
    plotting.py    matplotlib style, geometry overlays, diagnostic plots
    maps.py        chunk-fillable histograms: SBT wall maps, side views, x-y maps
    viz3d.py       interactive plotly 3D view
  bin/
    run_selection.py          pair background: --pairing per-file | time-window | kdtree
    make_vessel_dataset.py    -> muon_vessel_dataset.npz (input to the SBT optimisation)
    make_muon_distribution.py -> muon_distribution_3d.pkl (was Muon_Flames_rate.py)
    plot_muon_distribution.py plots of the 3D distribution
    optimize_sbt.py           SBT optimisation entry point (summary only so far)
    visualize_detector.py     3D html + side views + UBT plane + unfolded SBT walls + field map
                              (--source sst1 | ubt)
    convert_ubt_data.py       UBT_data_TRY6.pkl -> memory-mapped column store + 2 M subsample
    analyse_ubt_plane.py      chunked pass over the 150 M UBT-plane muons
    convert_sst1_outputs.py   all 20 001 output pickles -> one column store (78 s on 32 cores)
    visualize_full_dataset.py 3D + side views of the full dataset, a few muons of each type
    make_design_dataset.py    -> design-optimization-dataset-alpha-14-oct-2026 (see below)
    make_design_explorer.py   interactive HTML: UBT + fins with sliders, live veto numbers
    make_reco_inputs.py       muon-pair candidates, single-track sample, rate maps -> results/reco
    run_reco.py               designs -> digitization -> reconstruction -> TP / FN / FP / TN, rates
    benchmark.py              timing of the building blocks, CPU vs GPU
    benchmark_designs.py      per-design cost for arbitrary UBT/SBT geometries (truth crossings)
    benchmark_digitization.py per-design cost with digitization + hit-to-track matching
    fetch_data.sh             copy N output files (+ derived datasets) from the cluster
    remote.sh                 sync code / run / launch tmux jobs on the cluster
  designs/   veto designs (JSON): D0 minimal ... D4 dense, D5 cell-size variants,
             D6 technology variants (straw, fiber, pixel, tile, tile2 fins)
  ISSUES.txt open problems before optimising, with status (keep it updated)
  data/      local copy of the inputs (git-ignored)
  results/   everything we produce (git-ignored)
```

## Where the originals went

| original | now |
|---|---|
| `Analysis_singlefiles.py`, `analysis_multifiles.ipynb` | `run_selection.py --pairing per-file` |
| `analysis.ipynb` (selection cells), `Analysis_multifiles.py` | `run_selection.py --pairing time-window` (`--opposite-charge false` reproduces the notebook's DOCA cell) |
| `temp1.py` | `run_selection.py --pairing kdtree` |
| `analysis.ipynb` cell 21 | `make_vessel_dataset.py` |
| `analysis.ipynb` cell 32 | `plot_muon_distribution.py` |
| `Muon_Flames_rate.py` | `make_muon_distribution.py` |
| `Optimization_SBT.ipynb` | `optimize_sbt.py` |
| `analysis.ipynb` "check" plots | `--plots` on `run_selection.py` |

## Data

Inputs are Guglielmo's, on the physik cluster, in
`/disk/users/gfrise/Optimization_Veto_System` (same directory as
`/home/hep/gfrise/work_dir/Optimization_Veto_System`):

- `outputs/TRY6_UBT_T1_seed_*.pkl`: 20 001 files, one per seed, about 1 300 muons each.
  `planes[0]` is the muon start (z ≈ 0, no z stored), `planes[1]` is at the UBT (z = 30.6 m)
  and `planes[2]` is at SST1 (z = 82 m). Each file also has `weight`, `evt_n` and `pdg`.
- `data/muon_vessel_dataset.npz`, `data/muon_distribution_3d.pkl`, `data/MainSpectrometerField.root`.
- `MainSpectrometerField.h5`: field map of the spectrometer, z = 77–97 m.

On the cluster the code reads that directory directly; locally it reads
`veto/data` (`bin/fetch_data.sh 200 --derived`). Set `VETO_DATA_ROOT` to point
somewhere else and `VETO_RESULTS` to change the output directory. Nothing is
ever written into Guglielmo's directory.

### The UBT-plane dataset

`/disk/users/gfrise/MuonsAndMatter/data/outputs/UBT_data_TRY6.pkl` holds 150 M muons
(4.8 GB) crossing a 6 x 6 m plane at z = 30.6 m: x, y, z, px, py, pz, pdg and weight.
There is no evt_n, so the repeated-event weight correction cannot be applied.
`convert_ubt_data.py` turns it into a column store (`data/store/ubt_TRY6/*.npy`, one
memory-mappable file per column) and a 2 M random subsample (`data/store/ubt_TRY6_sample.npz`,
weights scaled up so totals are kept). On the cluster these live in
`/disk/users/sqasim/workspace/uzh/Veto_System/veto/data/store`.

**Missing data:** muons that never cross that 6 x 6 m plane are not stored, for example
those that bounce off the cavern walls and only reach the SBT or SST further downstream.
The SST1 outputs contain some of them (category `bouncing`).

`analyse_ubt_plane.py` propagates every muon forward in a straight line and clips it
against the vessel. That gives the entry face (UBT window, an SBT wall, or never inside),
the exit face, and whether the line reaches the SST1 acceptance. It keeps only histograms
(`results/ubt/hists.npz`, replot with `--replot`) and every vessel-entering muon
(`results/ubt/entering_vessel.npz`).

### The full dataset

`data/Full_dataset_z_3060_8200.pkl` (32 GB) has every simulated muon of one run, 506 374 000
rows: the state at z = 30.6 m and at z = 82 m, or where the muon stopped first, plus pdg,
weight and evt_n (= row index = input muon index, the same index as evt_n in the old files).
It is a rerun of seed 42: muons that reach the SST1 acceptance match old file 42, except the
ones that bounce off the cavern (that part differs from run to run). `io.open_pickle_columns`
memory-maps its columns in place (no unpickling, no torch); a full pass takes about a minute.

| type (`lib/veto/muon_types.py`) | muons per run | per spill | in the old 20 001 files |
|---|---|---|---|
| stops before z = 30.6 m | 355 M | 1.45e10 | no |
| stops between z = 30.6 and 82 m (rock at x ~ -5 m) | 25.8 M | 7.7e8 | no |
| reaches z = 82 outside SST1, straight | 125 M | 2.7e9 | no |
| reaches z = 82 outside SST1, bounced / scattered in the rock | 575 k | 1.8e7 | no |
| crosses an SBT wall or the UBT window, misses SST1 (veto only) | 975 | 3.3e4 | no |
| SST1 acceptance, p > 1 GeV and timing acceptance | 313 | 1.1e4 | yes, ~301 per file |
| SST1 acceptance, other | 688 | 9.6e3 | yes, ~723 per file |
| SST1 acceptance, bounced off the cavern wall / floor | 287 | 8.4e3 | yes, ~299 per file |

Bounced muons: the line forward from the z = 30.6 m state and the line backward from the
z = 82 m state meet (median 4 cm apart) on the -x cavern wall (x ~ -4.6 .. -5 m) or the
floor (y ~ -3.4 m); `muon_types.bounce_points` gives that point. For muons that stopped,
the backward line uses the momentum at the stop point.

### The design-optimisation dataset

`bin/make_design_dataset.py` combines both sources into
`data/store/design-optimization-dataset-alpha-14-oct-2026` (57 M rows, 5.8 GB, column store,
documented in its `meta.json`):

| type | source | rows kept | muons per spill |
|---|---|---|---|
| sst1_bounced | old, 20 001 runs | 5 986 931 | 1.0e4 |
| sst1_selected | old | 6 013 719 | 8.2e3 |
| sst1_other | old | 14 459 811 | 1.2e4 |
| veto_only | new, 1 run | 975 | 3.3e4 |
| rock_bounced | new | 575 321 | 1.8e7 |
| straight_past | new | 15 M of 125 M | 2.7e9 |
| stops_between | new | 15 M of 25.8 M | 7.7e8 |

`weight` is muons per spill: old rows divided by the number of files, capped types scaled by
rows in source / rows kept (column `scale`). The repeated-evt_n correction of the original
analysis is not applied. Columns: start state (old rows only), states at 30.6 and 82 m,
bounce point `kx, ky, kz`, `walls` (bitmask of SBT walls +x, -x, +y, -y and the UBT window
crossed by the path), `type`, `run`, `source`, `evt_n`, `pdg`. The veto_only and
rock_bounced types come from a single run: limited statistics, easy to overfit.

### Designs and reconstruction

A design (`designs/*.json`) is a UBT at z = 30.6 m (|x| < 0.5 m, at least the vessel entrance)
plus fins on the right (-x, looking from the target, stops at the cavern wall x = -4.56 m) and
left (+x): thin x-y planes touching the side wall with their inner edge and reaching out in x.
Nothing below y = -3.2 m. `bin/run_reco.py` builds events as short time windows around each
spectrometer candidate: the candidate's own hits (true path, efficiency, cells, time smearing)
plus accidental hits drawn from the full-statistics rate maps. Algorithm A extrapolates every
spectrometer track backward in a straight line to the planes upstream of its vertex and looks
for a hit in a space / time window; no combinatorics. Background = muon pairs of the old runs
(TP vetoed, FN survive), signal = hypothetical neutral decays in the vessel (FP = lost to
accidental hits). Bounced muons can only be predicted after their bounce.

### Detector technology

The goal is to veto the combinatorial background: pairs of muons from the muon shield that
enter the decay vessel region and reach SST1. Every veto plane is one of five technologies
(`lib/veto/technology.py`). Numbers are from SPSC-P-369 (2023 proposal) and SPSC-SR-370 (2025
annual report, UBT sec. 3.3.2), summarised in `conversation.txt`, updated with a colleague's
comments (Oct 2026: straw pitch, resolution, drift time and efficiency; fibre and pixel timing;
tiles). *Assumed* marks what nobody has given yet. Each entry describes one complete plane, all
layers included.

| | `straw` (UBT, fins) | `fiber` (fins) | `pixel` (fins) | `tile` / `tile2` (fins) |
|---|---|---|---|---|
| design | Liverpool UBT0 straws, **vertical** | Kuraray SCSF-78MJ mats, S13552 SiPM | Timepix-class hybrid | scintillator tiles, SiPM each |
| layers | 6: double staggered layers at 0°, +15°, −15° from vertical | 2 crossed mats (x, y) of 6 fibre layers | 1 | 1 |
| pitch | 5 mm per layer (5 mm straws) | 250 µm | 55 µm (variable, depends on the technology) | 4 cm / 2 cm |
| cell a hit gives (u × y) | 5 mm × 19 mm (stereo) | 250 × 250 µm | 55 × 55 µm | 4 × 4 cm / 2 × 2 cm |
| σ_x | 110 ± 20 µm (≥ 100 µm; 40–60 µm possible) | ~100 µm | ~16 µm (pitch/√12) | ~4 cm / ~2 cm |
| timing | drift limited: ~50 ns max drift, **50 ns window** | 0.5 ns (user's; can reach ~100 ps, check with Ettore) | ~200 ps (Timepix4; much better with LGADs) | ~120 ps / ~80 ps |
| veto window | ±25 ns (drift) | ±3σ ≈ ±1.5 ns | ±3σ ≈ ±0.7 ns | **±1.5 ns** fixed |
| plane efficiency | 99.9 % per track (> 97 % per straw) | 99.9 % (~100 % per detector, lower per fibre / SiPM) | 99.9 % (user's) | ~99.9 % (depends on the configuration) |
| max length | 1 m (overlapping rows) | 3 m *assumed* ("a few metres") | – | – |
| rate limit | 0.5 MHz/cm² (ATLAS TRT reference) | none given | none given | < 1 MHz per channel (up to 10 MHz per 5 × 5 cm tile in the flames) |
| material | 0.12 % X0 (0.02 % per layer) | 0.6 % X0 (estimate) | 0.5 % X0 *assumed* | 2.4 % X0 |
| cost | 6 kCHF per layer-m² (SST) | 170 kCHF per mat-m² (SND SciFi) | none given (nan) | 30 kCHF/m² (timing detector) |
| cost per channel (straw = 1) | 1 | 4 (user's) | 20 (user's) | 5 |
| 1 × 2.7 m plane | 3 600 ch., 97 kCHF, 3.6 k units | 14 800 ch., 920 kCHF, 59 k units | 8.9 × 10⁸ ch., 1.8 × 10¹⁰ units | 1 700 / 6 750 ch., 81 kCHF, 8.5 k / 34 k units |
| 3 × 6 m fin | 24 000 ch., 650 kCHF, 24 k units | 48 000 ch., 6.1 MCHF, 190 k units | 6.0 × 10⁹ ch., 1.2 × 10¹¹ units | 11 250 / 45 000 ch., 540 kCHF, 56 k / 225 k units |

**The UBT is always Liverpool straws, vertical:** `design.validate` sets it in every design,
D0–D5 included, and notes it if the JSON asked for another technology. The report's own design
is horizontal (two horizontal + two ±15° views) with 1 × 1 m modules on a 3 × 1 m detector; here it is
turned vertical with the same 1 m maximum straw length. A taller plane has overlapping rows of
straws (the overlap is ignored). The report's active area is 1.3 × 2.7–3.0 m; the UBT here stays
1 m wide (the vessel entrance). Fins can be any technology: set `"technology"` in the `detector`
block for every fin, or in a fin for that fin only (`designs/D6_*.json`). Explicit `efficiency`,
`sigma_t` or cell sizes next to it still win, except on the UBT. Without a technology a fin is
the generic detector of D0–D5 (no cost, no material).

- **Reconstruction:** each plane uses its own efficiency and time window. Drift-limited planes
  (straws) put the hit time anywhere in their 50 ns window and match within ±25 ns; tiles use
  their fixed ±1.5 ns veto window; the others use ±3 √(σ_t² + σ_t,spectrometer²), ±1.5 ns for
  fibres and ±0.7 ns for pixels. The accidental-hit window follows the slowest plane, so with a
  straw UBT every event is ±25 ns.
  `--time-window 100` on `run_reco.py` / `visualize_reco.py` forces one window on every plane
  and writes `results_tw100.json` / `visualization_tw100/`.
- **Channels:** strips across × segments along, summed over views and layers; pads area / pitch².
  Cost is area-driven (kCHF per layer-m², P-369 table 11 anchors, order of magnitude); a second,
  channel-driven figure is channels × relative cost per channel (straw channel = 1).
- **Rates:** a strip's rate is the rate map summed over its own segment; a pad's is the density
  × its area. `run_reco.py` prints `OVER RATE LIMIT` for a plane above its density limit
  (area in m²) or its per-channel limit. Rates assume a 1 s spill (`--spill`); the fin maps have
  20 × 20 cm bins, so a narrower peak is averaged down.
- **Not modelled yet:** σ_x (the 5 cm search window is far larger); ghost hits from combining
  views when several muons share a time window; the material of the planes (it is reported,
  not simulated: no scattering, conversions or neutrino interactions).

For reference, the OPT_BT layout had three straw entrance planes at z = 30.7, 30.8, 30.9 m
(±1.8 × ±1.35 m) and lateral rings around the tapered vessel at z = 40–78 m, with its own
simplified vessel geometry.

### Open problems before optimising

`ISSUES.txt` lists what is still missing, with a status (OPEN / IN PROGRESS / DONE / DEFERRED / OK)
and a priority (A: before optimising, B: changes the results, C: second order). In short:

- **A, blockers:** the pair background is saturated (2-3 survivors, use single-muon inefficiency
  and factorisation); bounced muons are only 74-77 % vetoed; rate limits are not enforced; no
  objective and two cost models; spill / PoT normalisation and spill structure.
- **B, physics:** multiple scattering (air 29.89-30.6 m and 80.6-82 m, 0.3 mm PVC walls at the
  vessel entrance and exit, helium in the vessel, the veto planes; for bounced muons only after
  the bounce); EM debris from the last 20 cm of shield iron, estimated from the full dataset;
  a literature-based HNL / dark-photon signal model. Neutrino and muon DIS are deferred.
- **C, detector:** response simplifications, dead material, validation against FairShip,
  weights and statistics of the single-run classes.

### Design optimisation: LLM evolution (AlphaEvolve-style)

`evolve/evolve_designs.py` runs on the laptop: an LLM (OpenRouter, key from `OPENROUTER_API_KEY` or the
supply2 `.env`) proposes complete design JSONs from a rank-weighted parent, the top designs as
inspiration and a directive (refine / explore / combine), with the problem context: geometry and limits,
the JSON schema, technologies, the muon density per side / z / distance from the wall, the objective.
Each generation's designs are evaluated in parallel on farm-gpu3 (`evolve/evaluate_designs.py`, sent over
the SSH proxy, so the key stays local): the full run_reco chain with multiple scattering, reported back
per plane (rates, limits, cost, which planes veto and which lose signal). `evolve/objective.py` is a
PLACEHOLDER score (signal kept - cost - rate-limit and background penalties). Everything goes to
`results/evolve/<run>/` (`history.jsonl`, `best.json`, the system prompt).

```bash
bin/remote.sh sync                                   # the cluster needs the current code
evolve/evolve_designs.py --generations 20 --children 8 [--model anthropic/claude-sonnet-4.5]
```

## Running

```bash
bin/fetch_data.sh 200 --derived                     # local subset
bin/visualize_detector.py --max-files 100           # results/visualization/
bin/run_selection.py --pairing per-file --max-files 200 --plots
bin/optimize_sbt.py

# full statistics on the cluster (bin/remote.sh run/launch use farm-gpu3; farm-ui is too small)
bin/remote.sh sync
bin/remote.sh run bin/convert_ubt_data.py           # once
bin/remote.sh launch ubt_full bin/analyse_ubt_plane.py
bin/remote.sh fetch-results ubt
bin/visualize_detector.py --source ubt
bin/remote.sh run bin/optimize_sbt.py
bin/remote.sh launch sel_tw bin/run_selection.py --pairing time-window --plots
bin/remote.sh peek sel_tw
bin/remote.sh fetch-results
```

Needs numpy, scipy, matplotlib, h5py and plotly. The cluster container
(`exec_cudabear_november_2025.sh`) has these.

## Things to know

- All tracks are straight lines built from the SST1 state. Inside the vessel there is no
  field, so back-projecting to the vessel walls is fine. The start → UBT → SST1 segments
  drawn in the 3D view only join the true crossing points. The real path bends in the
  shield field and can scatter off the walls.
- `bouncing` uses `|x| > 3 or y < -3.36` in the vessel dataset but `|x| > 3 or |y| > 3`
  in `load_simulation_file` of `analysis.ipynb`. `tracks.categories` follows the dataset.
- The original `backtrack_to_lateral` intersected the *infinite* planes of the walls without
  checking that the point is on the wall. For about 3 % of the SST1 tracks it returned entry
  points off the vessel surface, and most of the "SBT ±y" entries came from this. It is
  replaced by `tracks.vessel_crossings` (exact clipping of the line against the frustum).
  The existing `muon_vessel_dataset.npz` was made with the old version.
- Straight lines ignore multiple scattering in the vessel walls and in air.
- `analysis_multifiles.ipynb` unpacked `backtrack_to_lateral` as `(x, y, z)` although it
  returns `(z, x, y)`, so its `x_decay_vessel`/`z_decay_vessel` columns were swapped.
  Those columns were never used downstream, and the vessel dataset is not affected.
- The time-window pairing uses random times uniform in [0, 1) s, as in the original.
