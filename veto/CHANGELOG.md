# Changelog

Changes to the veto optimisation code, newest first. Each entry gives the branch, the base commit and
what changed. Open problems are in `ISSUES.txt`. Earlier entries (2026-10-19/20) are in this file on branch
`veto-all-sst1`.

## 2026-11-06 — A2 bounce-aware search, A1 closure check, C5 weights (branch `veto-shield-dis`)

- `reco`: `bounce_points`, second (mirrored) search window from the cavern wall / floor point for planes upstream of
  it; `RecoConfig.bounce_search = True`, `bounce_r_slope = 0.21`. D6_straw bounced vetoed 0.760 -> 0.975 (signal lost
  5.15 -> 6.38 %, toy signal).
- New `bin/closure_check.py` (A1): direct vs factorised surviving pairs per design / candidate file, Poisson p, 90 % CL
  limits. inputs.npz D0_minimal 1694 observed vs 1702 predicted.
- `make_design_dataset.py`: repeated-evt_n correction of the old rows by default (`--no-evt-n-correction`);
  `neff_by_type.py`: n_eff target 1000 with the seed factor needed (C5).

## 2026-11-05 (night) — A7 modules, C2 dead frames, A4 one cost model (branch `veto-shield-dis`)

- `technology`: `SLOT` = 0.5 m, `MODULES` (straw 1 x 1 m, tile2 / tile 0.5 slot, fibre 0.5 x 2.5 m, pixel quads in a
  slot; dead frames 3 / 1 / 1 / 2 (readout end) / 0.1 cm), `module_cost` (electronics per channel + 10 % P-369 area
  mechanics), `cost()` now per m^2 of module footprint (old P-369-only cost replaced), `BUDGET_KCHF`.
- `design`: Plane gains u_off, mod_u / mod_y, dead_u / dead_y, sub / sub_dead, n_modules, `active_fraction`;
  `inside()` rejects dead borders; "modules" design format, `module_blocks`, `validate_modules`, `snap_to_modules`,
  `GRID_Y0`. Continuous designs still load (old format).
- `reco`: rate maps / fine peaks shifted by the block's u_off. `objective`: `cost_pct_budget`.
- New `bin/snap_designs.py` -> `designs/modules/*_mod.json` (all 18 designs).
- `evolve/evolve_designs.py --grid modules` (default): module format in the prompt (module sizes, frames, cost per
  module in % of budget, FairShip geometry numbers), continuous proposals rejected, seeds from designs/modules;
  `evaluate_designs.py` reports u_from / n_modules / active per plane block.

## 2026-11-05 (evening) — issue review: A3 limits, D3 Pythia-only, D7 upstream air, A6b straw cuts, nu-He weight (branch `veto-shield-dis`)

- Decisions of the review recorded in ISSUES.txt (A1, A2, A4, A5, A7, B1, B2, B3, B4, C2, C5, D3, D6, D7, A6b; C1, C3,
  C9, D5, D9 left open).
- `objective`: only neutrino DIS in the helium is down-weighted, `w_src_nu_helium = 0.01` (split from per_target);
  everything else weight 1. `end_to_end.py` passes the neutrino per-target rows.
- `technology`: fibre max_channel_rate 1 MHz, pixel max_rate 3.5 MHz/mm^2 (placeholders, A3).
- `fastveto.dis.MuonLibrary(split="pythia")`; `shield_dis.MU_SPLIT = "pythia"` (D3; DIS part only, sigma x 1/5 - 1/8).
- `fastveto.targets.UPSTREAM_AIR`: muon paths start at the shield end (29.89 m) for the air target (D7).
- `RecoConfig.straw_cuts` / `run_interactions.py --straw-cuts fairship`: GenFit quality cuts on the vertex tracks (A6b);
  `straws.track_quality`; `bin/shipenv.sh` race-safe for parallel workers.

## 2026-11-05 — FairShip geometry, spectrometer field (A6), Geant4 vessel transport, C7, C8, D1/D2, D4, B4 import (branch `veto-shield-dis`, commit ffd5570)

### Geometry: FairShip layout, target-aligned (z_veto = z_FS - 2.14)
- `lib/veto/geometry.py`: vessel 31.13 - 81.13 m (was 30.6 - 80.6), `Z_UBT` = entrance window 31.13, `Z_SST` = 82.08 /
  84.08 / 91.08 / 93.08 (`Z_SST1` was 82.0), `Z_TIMING` 94.01 m, `TIMING_HALF` 2.75 x 3.25 (was 92.71, 2 x 3). New
  `Z_STATE_A / Z_STATE_B` = 30.6 / 82.0: where the muons are stored (not detectors); `OLD_*`, `VESSEL_SHIFT` (+0.53 m).
  `fastveto.vessel` follows.
- `muon_types.classify` (+ torch, identical on 2e5 muons): REACH from the state planes, SST1 acceptance after moving B
  straight to 82.08, UBT window on the path at 31.13. `scattering`: UBT / window at 31.13 on the path, air slices from
  30.6. `reco.true_crossing`: UBT hit on the path at its z. UBT maps (`make_design_explorer`, `make_peak_maps`) at 31.13.
- `design.py`: designs without `"frame": "fairship-2026"` get their fins moved +0.53 m with the vessel (note printed).

### A6: spectrometer field
- Field map re-centred at the FairShip magnet centre (`geometry.Z_MAGNET` = 89.72 - 2.14 = 87.58 m): the .h5 copy was
  0.62 m upstream (centred at 86.96, an older c.z); found from data/MainSpectrometerField.root (local z +-9.9 m, placed at
  Bfield.z by geomGeant4.py) with the SHiP CVMFS stack.
- New `lib/veto/spectrometer.py`: field map grid (trilinear), RK4 in z (torch, CPU / GPU), yoke loss |B| > 0.5 T;
  `to_z`, `timing_acceptance`. Integral SST1 -> TD on axis -0.97 T m (Bx): 1 GeV -> 2.04 m at the TD.
- Used by `muon_types(.torch).classify`, `interactions.charged_tracks`, `reco.signal_from_target`,
  `selection.single_muon_masks`, `bin/make_reco_inputs.py`, `bin/make_crossrun_pairs.py`, `bin/visualize_full_dataset.py`.
  Needs the `pdg` column for the charge (error if missing).

### Geant4 transport of the helium products (A6, D4)
- New `cpp/g4vessel.cc` (+ CMakeLists, build.sh), `lib/veto/g4vessel.py`, `bin/g4_vessel_transport.py`: helium / Mylar
  vessel, air, the field map, FTFP_BERT with decays; crossings at SST1 and the TD, decay products keep their ancestor's
  origin. `interactions.charged_tracks(part, g4)`, `evaluate(..., g4=)`, `prepare(..., g4=)`; `run_interactions.py --g4`.
  Built on the host with `bin/g4ship.sh build` (SHiP CVMFS Geant4 10.7.3, see below); validated against RK4.

### Geant4 on the host + straws + GenFit (post-processing)
- SHiP CVMFS (25.09, slc9 build) runs on this host: `bin/shipenv.sh` (FairShip / ROOT 6.30 / GenFit; libcrypt.so.2 ->
  libxcrypt .so.1 link), `bin/g4ship.sh` (host g++ / python3 + CVMFS Geant4 10.7.3; `bin/g4ship.sh build` ->
  `cpp/build_ship/g4vessel*.so`, no container). g4vessel validated: vs RK4 from SST1 to the TD median <= 1 mm, spread =
  air scattering (7 / 3 / 1 mm at 2 / 5 / 20 GeV). `g4vessel.init(planes=)`.
- New `lib/veto/straws.py`: FairShip strawtubes geometry (4 stations x y1/u/v/y2 x 2 layers, 2 cm straws, +-4.57 deg,
  strawtubes_config.yaml), digitisation as strawtubesHit (|Gauss(DOCA, 120 um)|, gas radius 0.9964 cm), FairShip
  detector IDs; `layer_states` (RK4) / `layer_states_g4` (Geant4); `reconstruct()`.
- New `bin/genfit_straws.py` (SHiP python + GenFit, as shipDigiReco.findTracks: pdg 13, >= 25 hits in >= 3 stations,
  RKTrackRep, FairShip seed, WireMeasurement, DAF 50 iterations, air TGeo world, our field map in kGauss) and
  `bin/straw_reco.py` (driver, --toy / --tracks, --g4). Toy muons 1-40 GeV: 98.5 % fitted, all converged, chi2/ndf
  median 0.96 (Geant4 trajectories), sigma_p/p 1.6 / 0.9 / 1.3 % at 1-5 / 5-15 / 15-40 GeV, charge always right.

### C7: air near the vessel and the fins
- `fastveto.targets.AIR_MARGIN = 0.4`: air pieces only within 0.4 m of the vessel's outer surface or a fin
  (`_near_air_pieces`, exact, checked to 3 mm); `run_interactions.py --air-margin` (< 0 = all the air).

### C8: two selections on the measured IP
- `vertex_selection.IP_CUTS = {full: 0.10, partial: 2.5}`, `IP_PRECUT = 3.0` (true IP of the candidates, was 2.5);
  `run_reco.py` / `run_interactions.py --ip-cut full|partial` (`*_partialreco` outputs).

### D1 / D2: reward weights
- `objective.ObjectiveConfig`: `split_sensitivity` (default True): `w_bkg_tag` (1 - S90(0)/S90(N_w)) and `w_sig_veto`
  (signal lost, uncapped) as separate terms; `w_src_nu_helium = 0.01` for neutrino DIS in the helium only (split from
  `per_target`), every other source `w_src_muon = 1` (was w_src_nu = 0.1 on all neutrino DIS, same day).

### D4: shield DIS chain
- `shield_iron.curved_points`: parent muon bent by the MS7 field to the vertex (`shield_dis.BEND_PARENT`).
- `shield_dis.NU_FLAVOURS` (14, -14, 12, -12): flavours with a store and an iron-library entry (nu_e needs the library
  rebuilt). `g4_vessel_transport.py --codes 3 4 --drop-v0-codes 3` for decays in flight / K_L, n interactions.

### B4: EventCalc signal
- New `bin/import_eventcalc.py`, `lib/veto/signal_eventcalc.py`, `data/signal/eventcalc/` (43 benchmarks, z_EC - 0.87).

### To re-run (old layout everywhere; waiting for approval)
1. `bin/g4.sh bash cpp/build.sh` (g4vessel).  2. stores: recompute `type` / `walls` with the new classify.
3. `make_design_explorer` maps + `make_reco_inputs.py` (inputs.npz, pre-cut 3 m) + `make_peak_maps.py`.
4. `run_reco.py` for `--ip-cut full` and `partial`.  5. muon interactions (`--air-margin 0.4`) + `g4_vessel_transport.py`
   + `run_interactions.py --g4`, both IP cuts.  6. shield chain (`shield_dis.py generate`, Geant4 after the plane).
7. iron GENIE library with nu_e.

## 2026-11-04 — rate limits: steep penalty + fine peak maps (ISSUES A3, branch `veto-shield-dis`)

- `lib/veto/objective.py` `rate_penalty`: per limit exceeded `1 + rate_slope (peak / (margin limit) - 1)` (new
  `ObjectiveConfig.rate_slope = 10`), was `1 + log10(peak / (margin limit))`; times `w_rate = 1000` as before.
- `bin/make_peak_maps.py` (new): 2 x 2 cm^2 rate maps (sum w, sum w^2) from all 506 M muons of the full dataset, UBT x-y
  at 30.6 m and fins at z = 30.6 - 40.6 m, u < 2 m -> `results/rate_maps/peak_maps.npz` (5 min, 32 processes).
- `lib/veto/reco.py`: `peak_maps`, `fine_peak`; `plane_rates` uses the peak over a sliding window of the channel size
  (pads) or 4 cm (strips) with >= 10 effective muons, max with the coarse peak; new keys `peak_hz_cm2_coarse`,
  `peak_hz_cm2_fine`, `peak_hz_cm2_err`, `peak_source`, `density_over_limit`. Without the file: coarse only, unchanged.
- `bin/run_reco.py`: `over_rate_limit` also lists density violations seen only in the fine maps.
- `bin/check_peak_rates.py` (new): coarse vs fine peaks and old vs new penalty for designs/*.json. At z = 30.6 m the
  peak is x 1.39 the coarse one (right fin 880 +- 5 vs 634 kHz/cm^2; 4 cm tile 14.1 MHz, 2 cm tile 3.6 MHz per channel).

## 2026-11-04 — vessel wall: 0.3 mm PVC -> 350 um Mylar (branch `veto-shield-dis`, commit a1ffeed)

- One switch: `fastveto.vessel.WALL_MATERIAL = "mylar"`, `WALL_THICKNESS = 350e-6` (+ `WALL_COMPOSITION` C 0.625, H 0.042,
  O 0.333); `veto.scattering.X0_WALL = 0.2854 m`, `WALL_THICKNESS = 350e-6`. Shell `OUTER_BOX` (exact chords) follows.
- `fastveto.targets`: wall crossings use the Mylar nucleon density and material; new `target_material(code)`, used by
  `fastveto.dis.run`, so wall DIS final states / cross sections come from the Mylar entries of the muon (Geant4 + Pythia 6)
  and GENIE libraries (already built for the straws). `materials.vessel_wall()` (alias `pvc_wall`).
- Kept for compatibility: target-group key `"pvc"` (code 2, `--pvc` / `draws["pvc"]`, `CODE_NAMES`, saved npz), and the
  old names `PVC_THICKNESS`, `X0_PVC` (now = wall values). `end_to_end.py` prints "wall".
- Wall per normal crossing: x/X0 1.53e-3 -> 1.23e-3, 0.0414 -> 0.0490 g/cm^2 (+18 % nucleons, no Cl). Results computed
  before this (results/interactions, apply / reco-input files with "pvc" rows, scattering-dependent reco) are PVC-based.

## 2026-11-04 — UBT-plane muon rate maps (results only, no library change)

- `results/ubt_xy_rate/` (copy of the plots in `work_dir/UBT_muon_rate_maps/`): x-y rate maps at z = 30.6 m, |x|,|y| < 3 m,
  2x2 cm^2 bins, kHz per bin. All muons (UBT_data_TRY6.pkl) 3.48 GHz; shah_study `straight_past_vessel` 2.74 GHz; straight AND
  single-track cuts before the coincidence (SST1 acc., p > 1 GeV, timing-detector acc.) 6.24 kHz of 8.50 kHz selected.
  Plus the two straight maps with full-sample iso-rate contours (`make_ubt_xy_contours.py`).
- Correction: "straight" = reaches z = 82 within 0.5 m of the line from the 30.6 m state (BOUNCE_CUT), not the
  simulation category `straight_past_vessel` (excluded straight muons through the vessel, kept some bounced). Straight
  2.73 GHz; straight & selected 8.13 kHz (was 6.24) of 8.50 kHz. shah_study category `bounced_back` holds straight muons
  (bounce flag of cuda_muons_ship.categorize_muons not reliable).

## 2026-11-02 (afternoon) — branch `veto-shield-dis` (uncommitted)

### Vessel muon DIS parents: 20 001-run SST1 sample + zeta veto-only
- `bin/build_dis_parent_store.py`: store `data/store/dis-parents-sst1-20k-vetoonly-zeta2000-2026-11-02` = types 0-2 from
  gamma (= alpha, 20 001 runs, w / 20 001) + type 3 veto_only from zeta seeds < 2000 (w x 3, evt_n + 1e12); symlinks to the
  shared stores in data/store. Totals 30 598 + 32 321 muons / spill (zeta all seeds: 32 415).
- `bin/run_interactions.py --tag`: suffix of the saved interaction npz (no overwrite of the default file).
- D6_straw, one interaction per crossing: 1.35e8 interactions; pie selection (DOCA < 1 cm, fiducial, IP < 2.5 m):
  helium 5.91e4 ± 1.38e4 (was 5.90e4), PVC 7 676 ± 941 (was 7 680 ± 1 342).

## 2026-11-02 — branch `veto-shield-dis` (commit 822e1fe)

### K_S and Lambda tracked in the MS7 iron (`--stable-v0`)
- `lib/fastveto/decay.py` `decay_all(..., stable=())`: codes left undecayed (default unchanged).
- `bin/dis_library.py --stable-v0 [--materials iron]`: Pythia `MDCY(C310,1)=0`, `MDCY(C3122,1)=0` (PYGIVE) and the Geant4
  part with `stable=(310, 3122)`; uses sqasim's pythia6_dis binary when this checkout has none. Built
  `data/interactions/dis/mu_library_iron_v0stable.npz` (2 min; K_S 172 097 vs K_L 171 380 in the Pythia part).
- `bin/shield_dis.py generate --mu-lib`; `bin/shield_chain_probs.py --mu-lib --tag`.
- MS7 μ per track, v0 copies 10 → `results/shield_dis/shield_muon_pertrack_v0stable.npz`: decays in the vessel per spill
  K_L 4 743 → 5 273 (K_S interactions in the iron make K_L), K_S 94 → 717, Λ + Λ̄ 41 → 390. After DOCA < 1 cm, fiducial,
  IP < 2.5 m: 5.87e7 → (6.73 ± 0.59)e7 / 15 y; K_S, Λ contribute 3e4 each (they decay near the narrow entrance).

### New: `bin/shield_natural_check.py` (unforced Geant4 through the last 50 cm vs the forced chain)
- K_L natural / forced = 0.52 ± 0.15; excess from the Pythia 6 component of mu_library_iron (90 % of the K_L). Details in docs/shield_iron_dis.md.

### New: `bin/shield_chain_probs.py` (per-species chain probabilities, results/shield_dis/chain_probs.json)
- P_int, multiplicity, P_out (primary / all, Geant4), P_dec in the vessel z range, per species; table in docs/shield_iron_dis.md.

### Check: K_L absorption in the MS7 iron (`bin/shield_kl_check.py`, new; no change to the chain)
- Geant4 (`cpp/g4shield`) already transports every DIS final state, K_L included, through the remaining iron:
  single K_L λ_eff = 16 - 18 cm; μ DIS K_L made 2.07e4 / spill → 1.04e4 / spill at the plane (6.4e3 from primary
  survival with λ = 17 cm, the rest made in the cascade). No extra survival weight added (would double count).
  Details in `docs/shield_iron_dis.md`.

## 2026-11-01 — branch `veto-shield-dis` (commit 700eb10, with the 2026-10-30 entries)

### Bug fix: own UBT hits of tracks born downstream (`lib/veto/reco.py` `own_hits`)
- With multiple scattering on, `own_hits` recomputed `ok = crossed & inside` and dropped `true_crossing`'s condition
  that a track must start upstream of the UBT (`az < Z_UBT + A_TOLERANCE`). Every track then got a UBT hit at the
  (x, y) of its start, so DIS tracks born inside the vessel could veto themselves (ν DIS in helium with the minimal
  UBT: 9.5 % fake vetoes). Muons from the transport start at the UBT and are not affected; fins were not affected
  (`path_at_z` returns NaN for tracks that do not cross them).
- Fix: `crossed &= (apz > 0) & (az < Z_UBT + A_TOLERANCE)` for the UBT.
- Veto step rerun on the stored D6_straw files (`results/dis_per_track/rerun_ownhits_fix.sh`, old outputs
  `*.before_ownhits_fix.json`), per 15 y after veto b (old selection, measured IP < 10 cm, <= 2 tracks):
  ν vessel 0.70 -> 0.93 (of 1.03 selected), MS7 μ K_L 2.1e4 -> 3.4e4 (of 3.4e4: no veto), MS7 ν 0.77 -> 0.83
  (of 1.68), μ He 32 -> 37 ± 12 (of 1 913; the random stream changes with the number of hits, within errors).

## 2026-10-30 — branch `veto-shield-dis` (on top of `e35533a`, not committed yet)

### DIS sampling: one draw per crossing / per track (Guglielmo, 2026-10-30)
- `lib/fastveto/dis.py` `run`: a draw count M = −k gives k draws on EVERY crossing of the group, weight w·P / k each
  (default M > 0 unchanged: M draws picked with probability w·P / S, weight S / M). Both unbiased; per crossing
  every track contributes once instead of the heaviest tracks being redrawn many times.
- `bin/shield_dis.py generate --per-track`: one interaction per parent line, weight w·P.
- Rerun: `results/dis_per_track/run.sh` (μ vessel He / PVC / planes per crossing, air 2e5 draws as before; ν every
  target per crossing; shield μ / ν per track). The w = 1 alternative (flattened weights) was rejected: the
  w = 768.75 minimum-bias class is 40.5 % of the true shield DIS rate and would become 0.8 %.

### DIS in the last 50 cm of the muon shield (`docs/shield_iron_dis.md`)
- `lib/veto/shield_iron.py` (new): MS7 iron box, field map loader / lookup, chord, Geant4 set-up; input muons
  `Muons_2950_50cm_before_end_MS.pkl` at true z = 29.20 m (stored z − 0.30 m).
- `cpp/g4shield.cc` (new) + `cpp/CMakeLists.txt`, `cpp/build.sh`: Geant4 transport of final states through the iron
  with the field map, recorded at z = 29.90 m. `build.sh` finds pybind11 in `external/pybind11_pkg` if the container
  has none.
- `bin/shield_dis.py` (new): prepare / generate (forced interactions, transport, forced V0 decays in the vessel);
  `bin/shield_dis_smoke.py` (new) checks.
- `bin/dis_library.py`: iron (G4_Fe) material; `--only-g4 MATERIAL --base LIB` adds Geant4 tables to an existing
  library. `lib/fastveto/materials.py`: iron.
- `bin/genie_splines.py`, `bin/nu_library.py`: Fe56; `--nuclei`, `--nus` options. `lib/fastveto/nudis.py`: iron.
- `lib/fastveto/decay.py`: one decay round factored out (`_decay_round`, `decay_all` unchanged), `force_decay`
  (decay exactly at the given point) with K_L channels.
- `lib/veto/interactions.py`: target codes 3 `shield_iron`, 4 `shield_iron_v0`.
- `bin/g4.sh`: puts `external/pylib` (numpy, h5py, scipy for the container's python) on PYTHONPATH if present.
