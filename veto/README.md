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
    muon_types_torch.py  the same classification on the GPU (float64, for the muon production)
    design.py      veto designs (UBT + fins) from designs/*.json, physical limits, stations per plane
    technology.py  detector technologies (straw, fibre, pixel, tile, tile2): layers, timing, rates, material, cost
    reco.py        digitization, accidental hits from rate maps, spectrometer, veto algorithm A, vertex cone
    scattering.py  multiple scattering: planes, exact PVC chords, helium / air slices, single-scattering tails
    vertex_selection.py  the signal selection shared by every sample (measured IP < 0.1 m, <= 2 SST1 tracks)
    interactions.py  forced muon / neutrino interactions through the same selection and veto
    objective.py   reward of a design: "tiers" or "sensitivity" (expected 90 % CL limit), all background sources
    tracks.py      straight-line extrapolation, back-tracking to the vessel walls, categories
    weights.py     repeated-evt_n weight corrections
    pairs.py       all pairs / time-window pairs / KD-tree candidates, DOCA + vertex
    selection.py   single-muon and pair cut flows
    plotting.py    matplotlib style, geometry overlays, diagnostic plots
    maps.py        chunk-fillable histograms: SBT wall maps, side views, x-y maps
    viz3d.py       interactive plotly 3D view
  lib/fastveto/    ported from fast-veto-sim (2026-10-21): muon DIS (dis), neutrino DIS (nudis, neutrinos),
                   targets and materials, vessel geometry, Kokoulin cross sections, decays
                   (docs/fast_veto_sim.md, docs/fastveto_physics.md)
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
    -- muon production and datasets (see "Muon production" below)
    generate_muon_study.py    seeds of the new production, every muon type kept (run_muon_study.sh,
                              status_muon_study.sh)
    make_design_dataset_beta.py / _gamma.py   beta and gamma design datasets
    make_crossrun_pairs.py    independent pair sample (muons of different runs)
    subsample_candidates.py   Poisson subsample of the pair candidates proportional to weight
    replace_rate_maps.py      patch corrected rate maps into a reco-inputs file
    neff_by_type.py, neff_vs_seeds.py, probe_statistics.py, make_probe_html.py, scan_shah_study.py
                              statistics checks: independent input muons per type, per-cell probe
    -- interactions and end to end (see "Interactions" below)
    prepare_interactions.py   cache the selected muon / neutrino interactions (design independent)
    run_interactions.py       one design: interactions through selection and veto
    scan_cone.py              scan of the vertex cone
    end_to_end.py             every background source before / after the veto, per design and cone
    muon_dis_apply.py, nu_dis_apply.py   per-design forced muon DIS / GENIE neutrino interactions
    dis_library.py, genie_splines.py, nu_library.py, extract_neutrinos.py, nu_prepare.py
                              interaction libraries and neutrino flux (checks: dis_closure.py, dis_check.py,
                              g4dis_smoke.py, plot_dis_library.py)
    tail_produce.py, build_enriched.py, run_slices_any.sh, charm_*.py, tail_*.py, ...
                              Pythia 6 tail production and the enriched input file
    g4.sh, lcg.sh, genie.sh   containers (cudabear + Geant4, LCG_106, GENIE)
  cpp/       g4dis.cc (Geant4 DIS), tailgen/ (Pythia 6 / 8 + Geant4 transport), genie/ (GENIE build);
             builds in cpp/build, cpp/tailgen/build, external/ (git-ignored, not touched by remote.sh sync)
  evolve/    LLM evolution of designs (see below)
  docs/      SUMMARY_2026-10-20.md (state of the study), fast_veto_sim.md, fastveto_physics.md,
             geant4_neutrino_notes.md, live_talk/ (the talk: veto_talk.tex, run.sh)
  designs/   veto designs (JSON): D0 minimal ... D4 dense, D5 cell-size variants,
             D6 technology variants (straw, fiber, pixel, tile, tile2 fins),
             E1–E5 evolved designs (E3_robust, E4_ubt2 / E4_entry2 with second stations, E5_evolved)
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
somewhere else and `VETO_RESULTS` to change the output directory. The interaction
libraries are in `data/interactions` (`VETO_INTERACTIONS`) and the source muons at the
target in `data/source` (`VETO_SOURCE`). Nothing is ever written into Guglielmo's directory.

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

### Muon production and the later datasets

`bin/generate_muon_study.py` reruns Guglielmo's MuonsAndMatter simulation (imported read only,
TRY6 settings of his shah_study runs) and keeps **every** muon type, including type 7
`stops_before` (stopped in the shield; its "30.6 m state" is the stop point). One seed = one
spill; each type fills a fixed row budget, so the rare types (SST1, veto-only) are kept complete
in every seed. The classification runs on the GPU (`lib/veto/muon_types_torch.py`, checked muon
by muon against `muon_types.py`). Run with `bin/run_muon_study.sh`, monitor with
`bin/status_muon_study.sh`.

| dataset | rows | made from | `$R/data/store/` |
|---|---|---|---|
| alpha | 57 M | old SST1 runs + one full run | `design-optimization-dataset-alpha-14-oct-2026` |
| beta | alpha + ~240 per seed | alpha + veto-only muons of the shah_study seeds (central entrance) | (`bin/make_design_dataset_beta.py`) |
| gamma | 88 M | old SST1 runs + the new production (every type) | `design-optimization-dataset-gamma-2026-10-17` |
| **delta** | 111.4 M | 484 seeds of the enriched production | `design-optimization-dataset-delta-enriched-2026-10-19` |

`$R` = `/disk/users/sqasim/workspace/uzh/Veto_System/veto`. **Enriched input**: in
`full_sample_after_target.h5` the muons above ~150 GeV rest on ~9 k minimum-bias rows, so the
SST1-selected rate had a 13.5 % error that more seeds cannot reduce. `full_sample_after_target_enriched_p6.h5`
replaces them by a Pythia 6.4.28 + Geant4 tail production (1.2e8 PoT, `bin/tail_produce.py`,
`bin/build_enriched.py`) with the same flux normalisation: n_eff of SST1-selected goes from 55 to
18 170 (`bin/neff_by_type.py`). Details in `docs/fast_veto_sim.md`. Weights are muons per spill;
15 years = 1.2e7 spills. Near the walls (first metre) statistics stay thin
(`bin/probe_statistics.py`, `results/probe/`).

The 25 M delta pair candidates are Poisson-subsampled to 400 k (`bin/subsample_candidates.py`,
unbiased for every weighted sum): `results/reco/inputs_delta_s.npz`.

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

Every sample (signal, pairs, interactions) goes through the same selection
(`lib/veto/vertex_selection.py`): measured IP < 0.1 m at the target and at most 2 tracks in the
SST1 acceptance. Two design options:

- `"stations": 2` (or 3) on the `ubt` or a fin: identical layers behind each other read out as an
  OR, efficiency 1 − (1 − e)ⁿ, n × channels, cost and material; random hits and per-channel rates
  are those of one station.
- `"detector": {"cone_mrad": ...}` (vertex cone, `RecoConfig.cone`): upstream of the vertex the search
  radius is max(r_A, r0 + cone · (z_vertex − z_plane)). It recovers the parent of a muon interaction
  that kinked by a few mrad at the vertex, at the price of more random hits (signal loss).

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
- **Material:** multiple scattering in the planes, the PVC walls (exact chords) and helium / air
  (`lib/veto/scattering.py`, with single-scattering tails tuned to Geant4); muon and neutrino
  interactions in the planes, walls, air and helium (see "Interactions").
- **Not modelled yet:** σ_x (the 5 cm search window is far larger); ghost hits from combining
  views when several muons share a time window; photon conversions in the planes.

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
  a literature-based HNL / dark-photon signal model. Neutrino and muon DIS are now modelled
  (see "Interactions"); `ISSUES.txt` may still list them as deferred.
- **C, detector:** response simplifications, dead material, validation against FairShip,
  weights and statistics of the single-run classes.

### Interactions and end to end

Ported from fast-veto-sim into `lib/fastveto` (nothing reads the old repository any more): forced
muon DIS (Geant4 library, `cpp/g4dis.cc`) and neutrino interactions (GENIE 3.06.02) in every target:
helium, air, PVC walls and every plane of the design. `lib/veto/interactions.py` puts them through
the same selection and veto; the scattered muon's path starts from its parent's 30.6 m state and
kinks at the vertex, so the parent's real upstream hits are digitised. Veto levels: (a) the two
vertex tracks, (b) every reconstructed track, (c) a truth bound.

`bin/end_to_end.py` gives every background source per 15 years, before and after the veto, per
design and cone (`results/end_to_end/e2e.json`); neutrinos come from `bin/nu_dis_apply.py`. On
delta (`docs/SUMMARY_2026-10-20.md`): combinatorial falls from 16.8 to ~1e-5, muon interactions in
PVC and planes are all vetoed; E4_entry2 (second stations at the entrance, cone 17.5 mrad, 0.61 MCHF)
leaves 0.62 ± 0.06 events, all neutrino DIS in the helium, with 2.85 % of the signal lost. That
neutrino floor (no incoming charged particle) can only be vetoed by random hits, like the signal.
Builds and containers: see `docs/fast_veto_sim.md`.

```bash
GPU_HOST=farm-gpu5 bin/remote.sh run bin/end_to_end.py --stage generate --designs E3_robust D6_tile2
GPU_HOST=farm-gpu5 bin/remote.sh run bin/end_to_end.py --stage evaluate --designs E3_robust D6_tile2 --cones 0 10 20
```

### Design optimisation: LLM evolution (AlphaEvolve-style)

`evolve/evolve_designs.py`: an LLM proposes complete design JSONs from a rank-weighted parent, the top
designs as inspiration and a directive (refine / explore / combine), with the problem context: geometry and
limits, the JSON schema, technologies, the muon density per side / z / distance from the wall, the objective.
Backends: `--backend claude` (Claude Code CLI, default), `codex` (Codex CLI) or `openrouter` (key from
`OPENROUTER_API_KEY` or the supply2 `.env`). Each generation's designs are evaluated in parallel
(`evolve/evaluate_designs.py`, on farm-gpu3 over the SSH proxy, or `--host local`): the full run_reco chain
with multiple scattering and the interactions, reported back per plane (rates, limits, cost, which planes
veto and which lose signal). `--inputs` / `--prep-tag` pick the dataset (`inputs_delta_s.npz`, `_delta`).
Everything goes to `results/evolve/<run>/` (`config.json`, `history.jsonl`, `best.json`, the system prompt).

**Objective** (`evolve/objective.py`, `lib/veto/objective.py` form `"sensitivity"`, since 2026-10-20):
```
reward = - 100 (1 - sensitivity) - cost points - constraint penalties
sensitivity = (1 - signal lost) * S90(0) / S90(N)        S90 = expected 90 % CL upper limit
```
N sums every source (combinatorial + muon + neutrino interactions); cost is 2 points per MCHF up to 5 MCHF,
4 above; a plane over its rate limit or a design over 10 MCHF is infeasible. The older "tiers" form capped the
signal term at 10 % loss, and Codex run 3 exploited it (74 % of the signal given up to cut neutrinos 0.6 → 0.06).

```bash
bin/remote.sh sync                                   # the cluster needs the current code
evolve/evolve_designs.py --generations 20 --children 8 [--backend codex] [--inputs inputs_delta_s.npz --prep-tag _delta]
# or entirely on a cluster node, in tmux (log in to claude / codex there once):
evolve/run_on_cluster.sh --generations 25 --children 8
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

Needs numpy, scipy, matplotlib, h5py and plotly (torch for the muon production and the GPU paths).
The cluster container (`exec_cudabear_november_2025.sh`) has these; Geant4, LCG and GENIE jobs run
through `bin/g4.sh`, `bin/lcg.sh`, `bin/genie.sh`.

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
- The geometry-only scripts (`tracks.py`, `analyse_ubt_plane.py`, the selection) use straight lines
  without scattering; `run_reco.py` and the evolution add it through `lib/veto/scattering.py`.
- `analysis_multifiles.ipynb` unpacked `backtrack_to_lateral` as `(x, y, z)` although it
  returns `(z, x, y)`, so its `x_decay_vessel`/`z_decay_vessel` columns were swapped.
  Those columns were never used downstream, and the vessel dataset is not affected.
- The time-window pairing uses random times uniform in [0, 1) s, as in the original.
