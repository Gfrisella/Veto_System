# Changelog

Changes to the veto optimisation code, newest first. Each entry gives the branch, the base commit and
what changed. Open problems are in `ISSUES.txt`. Earlier entries (2026-10-19/20) are in this file on branch
`veto-all-sst1`.

## 2026-11-10 — V0 tally in g4shield, V0-stable MS7 chain, low-W data, K_L -> pi+ pi- (branch `veto-v0-production`)

- `cpp/g4shield.cc`: `v0_tally()` (every K_L / K_S / Lambda / Xi0 created: creator process, vertex, p, fate below cut /
  decayed / absorbed / exit); additive, the transport output is unchanged. `bin/shield_dis.py generate --v0-tally`
  (-> shield_<kind><tag>_v0tally.npz). `lib/veto/shield_iron.py`: `G4SHIELD_BUILD` overrides the module path.
- `bin/g4pixi.sh`: Geant4 11.4.2 from the FairShip pixi env 26.10 (CVMFS), build + run without the container.
- `bin/v0_fate.py` (tally analysis), `bin/v0_lowW.py` + `data/reference/lowW_strangeness_photoproduction.json`
  (SAPHIR / CLAS / ABBHHM / SLAC, sigma_tot), `bin/shield_ip_tail.py` split by V0 species.
- `lib/fastveto/decay.py`: K_L -> pi+ pi- (1.967e-3) in KL_CHANNELS.
- Runs: results/shield_dis/shield_muon_g4v114_{default_s401,v0stable_s402}(.npz, _v0tally.npz). Results:
  docs/v0_production.md slides 4-6, ISSUES D16.

## 2026-11-10 — status-quo report (branch `veto-v0-production`)

- New `docs/status_report_2026-11-10/`: Overleaf-ready LaTeX article (main.tex, sections/, refs.bib, figures/, logos/,
  README.md), 9 pages: background budget for D6_straw_mod per source, setup, combinatorial, muon DIS, MS7 K_L incl. the
  V0-vs-data check, neutrinos, rock / EM debris, statistics, open issues. Numbers from docs only, nothing re-run.

## 2026-11-10 — V0 production in the MS7 muon DIS chain vs data (branch `veto-v0-production`, base 4904c4f)

- New `bin/v0_production.py` (host python, existing files only): every V0 made at the MS7 vertex (V0-stable library
  folded with the PR #15 vertex-energy spectrum), V0 vs W against gamma p 20 GeV (SLAC SHF) and NOMAD yields, V0 at the
  exit plane vs vertex depth. Results in `results/v0_production/README.md`: Geant4 K0_S / Lambda at W ~ 6 GeV within
  ~6 % of the gamma p data, Pythia 6 ~x1.9 high; per spill 7.9e3 K_L, 7.8e3 K_S, 5.4e3 Lambda made, 2.9e3 K_L at the
  exit (21-37 % of them made by the shower in the iron); write-up docs/v0_production.md.

## 2026-11-10 — event store, MS7 on the Nov-4 muons with Geant4, debris and event viewers (branch `veto-neutrino-2026`)

- `lib/veto/evstore.py`: design-independent event stores, events + track trees (parent links, end reasons, times) from each
  source's reference surface to SST1; propagator with decays in flight (pi+-, K+-, K_L, K_S, Lambda, ...); `to_fastveto` /
  `roots_to_fastveto` feed `interactions.evaluate` (and `g4_vessel_transport.py`). `bin/evstore_build.py` (shield / apply /
  rock), `bin/evstore_eval.py` (any design, `--ip-tail`, `--export-roots`, `--g4`), `bin/evstore_redecay.py` (more forced
  decays per stored V0: the IP tail without new Geant4), `bin/shell_to_store.py` (shell seeds -> dataset store).
- MS7 parents from our own transport, as the 2018 one-run file was made: `bin/cpu_transport.py run --planes 29.2 30.6
  --keep-box 2 1.6` (Nov-4 pool, 8 s per chunk on one GPU) -> `shield_dis.py prepare --from-transport`. 4.62e8 / spill
  through the iron (x1.19 the 2018 file, same shape). `shield_dis.py`: `--prepared`, `--nu-lib`, `--nu-stores`, neutrino
  defaults GENIE v2 + 6 flavours, **`MU_SPLIT` default "geant4"** (the 9 Nov D3 decision; was still "pythia" in the code).
- MS7 K_L with Geant4 11.2 (10 x 2e6, 5 copies): ~1.5e4 per 15 y after selection (IP-tail extrapolation,
  `bin/shield_ip_tail.py`), on both the 2018 and the Nov-4 parents.
- `em_debris.py --parents --tag` and stage-1 exit states saved; `rock_apply.py --em-out / --store-out`; `nu_rock.py combine
  --store-out`; `interactions.CODE_NAMES` 5 / 6 = rock wall / floor.
- Viewers' data: `bin/em_viz_data.py`, `bin/em_viz_rock_data.py` (EM debris page), `bin/ms7_events_viz.py`,
  `bin/ms7_event_chains.py` (MS7 event-by-event page).

## 2026-11-10 — neutrinos with the 2026 target: production, GENIE v2, vessel / rock backgrounds (branch `veto-neutrino-2026`)

- Production: `bin/nu_prod.sbatch` (min bias, charm, beauty), `tailtransport.cc` / `tail_produce.py` neutrino mode,
  `charm_cascade.py` scratch on node-local /tmp, `hf_decay.py --neutrino`; summaries `bin/nu_flux.py`, `bin/nu_maps.py`,
  `bin/nu_validate.py`. Paths through `VETO_SHARED` / `VETO_DIR` / `APPTAINER_ENV` / `CUDABEAR_SIF` (also `bin/genie.sh`).
- GENIE library v2: 6 flavours x 15 nuclei incl. Fe56 and the concrete nuclei (`genie_splines.py`, `nu_library.py`; tau
  decayed by GENIE); `fastveto.nudis.COMPOSITION["concrete"]`.
- `nu_dis_apply.py --chunk` (memory); `run_interactions.py --pot-spill` for weights not at POT_PER_SPILL.
- New: `bin/nu_rock.py` (nu DIS in the cavern concrete through g4rock), `bin/rock_select.py` (rock draws through
  interactions.evaluate, cut flow), `bin/rock_combinatorial.py` (rock tracks in the combinatorial term),
  `bin/nu_activity.py` (lateral activity of selected nu-He events). Results: `docs/neutrinos_2026.md`, ISSUES D2 / D4 /
  D7 / D14.

## 2026-11-10 — ACTS straw tracking: FairShip pixi environment + PR #1545 (branch `veto-shield-dis`)

- New `bin/fairship_env.sh`: runs a command in the FairShip pixi environment 26.10 on CVMFS (ROOT 6.40, Geant4 11.4,
  ACTS 48.2; works on this host without workarounds) with our `libShipActsTracking` first.
- New `external/fairship_acts/` (force-added, the rest of `external/` stays ignored): FairShip `ActsTracking/` from PR
  #1545 head 786b361 (straw material from the TGeo straw volumes), `build.sh` rebuilds the library in that
  environment, README with provenance and the validation (499 / 500 muons of 10 GeV fitted, p median 9.992 GeV, core
  sigma(p)/p 0.94 %, chi2/ndf 0.92).
- `bin/shipenv.sh`: comment points to `bin/fairship_env.sh` for ACTS. The attempt on the slc9 release 26.10 (patched
  ACTS-fork library, own ACTS-examples chain, EL9 compatibility libraries) was dropped; findings in ISSUES D14.

## 2026-11-09 (night) — D5 momentum-convention check (results only, no code change)

- `results/d5_convention/` (scripts, json, plots, README): the tail rows of the enriched input carry the production momentum
  (target-exit momentum in the old file's convention would lower SST1-selected by ~10 %). The 1.55x excess comes mostly from
  the tail's wider input angles (x 0.56 with the old angular shape). ISSUES D5.

## 2026-11-09 (night) — new neutrino flux, 2018 flux discarded; GenFit removed (branch `veto-shield-dis`)

- `fastveto.neutrinos.load`: Shah Rukh's 2026-11-09 production (`/disk/gfs_lhcb/sqasim/nu_prod/tail`, `$VETO_NU_DIR`;
  parts nuMB hf == 0, charmc, beautyc; weight w_decay x POT_PER_SPILL / protons per part; keys + `part`). The 2018
  loader and `bin/extract_neutrinos.py` removed. `bin/nu_prepare.py` -> `data/interactions/neutrinos2026/store_nu*`
  (6 flavours, 1.4 min); `shield_dis.NU_STORES`, `nu_dis_apply.py --store` default to it.
- Discarded (Guglielmo): every file made with the 2018 flux (`logs/nu2026/discarded_2018_nu.txt`, ~42 GB) and the
  symlink `data/interactions/neutrinos` (target in Shah Rukh's area untouched).
- GenFit removed in favour of ACTS: `bin/genfit_straws.py`, `bin/straw_reco.py`, `straws.reconstruct` deleted;
  `straws.track_quality` and `run_interactions.py --straw-cuts acts` raise NotImplementedError until the ACTS
  parametrisation exists (ISSUES D14).
- `bin/shipenv.sh`: `SHIP_RELEASE` (default 25.09); for 26.x the host FreeType is preloaded so ACTS imports
  (host harfbuzz vs CVMFS FreeType 2.10.1). `results/straws` (GenFit fits, 429 MB) deleted.

## 2026-11-09 (night) — issue review: decisions for every open item (branch `veto-shield-dis`, docs only)

- ISSUES.txt: dated "9 Nov 2026" decision under every open item, action list at the top, new D15 (CVMFS library loading
  under load). D3 default -> Geant4 muonNuclear; D4 range cut; C7 forced K_L decays in fastveto; ACTS (CVMFS 26.10)
  for the track-fit parametrisation; D14 debris in time; B4 benchmark set into the objective.
- New docs/detector_questions.md (C2 + C6). docs/STATUS.md: section for the veto side (+ the GSimple 2026 neutrino-flux
  note from another session, committed with it).

## 2026-11-09 (evening) — natural Geant4 validation of the MS7 chain (branch `veto-shield-dis`)

- New `cpp/g4shieldnat.cc` (g4shield + physics list by name via G4PhysListFactory, muonNuclear x bias with weights,
  origin per particle), built by `bin/g4ship.sh build` (`G4SHIP_MODULES` selects modules).
- `shield_iron.init_g4shield(natural=...)`, `shield_dis.transport(natural=...)`; new `bin/shield_natural.py`.
- Results: results/shield_dis/natural_{QGSP_BERT_HP_PEN,FTFP_BERT}_b10.json (summary in docs/shield_iron_dis.md).

## 2026-11-09 (evening) — D3 natural-Geant4 validation of the MS7 chain; issue decisions (branch `veto-shield-dis`)

- New `cpp/g4shieldnat.cc` (g4shield geometry + field, natural muonNuclear biased by `--mu-bias`, products weighted
  1/bias, origin 1 = muonNuclear / 2 = other secondaries) and `bin/shield_natural.py` (draws the chain's MS7 parents,
  counts what crosses z = 29.90 m per species and origin, compares with the chain runs). `--save` (+ `--v0-copies`, 50)
  writes the muonNuclear products at the plane in the `shield_dis.py` npz format (one "interaction" per particle,
  forced V0 decays in the vessel) so `compare_shield_generators.py` reads it like a chain run.
- Runs (9.8e6 muons, bias 10, ~4-6 min each): `results/shield_dis/natural_{QGSP_BERT_HP_PEN,FTFP_BERT}_b10.json`,
  `shield_muon_natural_*_b10.{npz,json}`, table `generators_natural.md`: exit K_L 3 510 / 3 540 per spill vs chain
  G4 10.7 3 390 (Pythia 6 9 490); K0-decay tracks (>= 3/4 st, p > 1) 48.4 / 42.7 Hz vs 38.5 -> 7.9 expected FairShip
  entries, FairShip 0 (ISSUES D3).
- Also committed: `bin/sst1_bounce_categories.py` (straight / wall / floor SST1 muons), `bin/verify_zeta.py`, small
  fixes in `shield_dis.py`, `shield_iron.py` (natural module loader), `g4ship.sh`.
- ISSUES decisions (Guglielmo, 9 Nov): D12 ours main, C9 K-based split + MuonsAndMatter patch, C3 deferred, D5 test
  the momentum convention, C1 ghost hits, D9 redo with multi-vertex rejection, D2 lateral-track veto study, D6 energy
  rescaling branch, D4/D14 new neutrino sample + nu_e library.

## 2026-11-09 (afternoon) — Pythia 8 photoproduction library, comparison at 4e13 (branch `veto-shield-dis`)

- `bin/p8_library.py --variant full`: Pythia 8 photoproduction (Q² < 1.5, W > 5, Annual Report 2026 Table 7) + DIS
  (Q² > 1.5, W > 2) mixed by σ -> `mu_library_iron_p8full.npz`; empty parts skipped per energy.
- `bin/shield_dis.py` records `pot_per_spill` in its json; `bin/compare_shield_generators.py` puts every run on 4e13
  (older runs × 0.8, FairShip π 800 -> 640 Hz). New chain run `shield_muon_fs2026_p8full` (2e6 draws, all transported).
- Note: run shield_dis with `bin/g4ship.sh` (Geant4 10.7.3 env); under load many workers fail to load CVMFS libraries and
  are restarted (16 416 ImportErrors in the log, all 2e6 interactions transported).

## 2026-11-09 — 4e13 p.o.t. per spill everywhere (D13), shared geometry (branch `veto-shield-dis`)

- **Normalisation (D13, decided by Guglielmo):** `config.POT_PER_SPILL = 4e13`, `SPILLS_15Y = 6e20 / 4e13 = 1.5e7`; every
  `SPILLS_15Y` (interactions, shield_dis, dis_closure, muon / nu_dis_apply, run_reco, rock_apply, fastveto.neutrinos)
  imports it. Inputs normalised to another p.o.t. are rescaled when READ: `weights.source_pot` (meta `pot_per_spill`;
  else 4e13 for the Nov-4 pool v5 by its path; else legacy 5e13) and `weights.to_spill`, applied in io.load_output_file
  (+ all planes), io.open_pickle_columns, io.open_store (weight / weight_raw; EventCalc stores untouched),
  io.load_vessel_dataset / load_distribution, generate_muon_study (pool.json or path), make_design_dataset_gamma (seed
  meta), shield_dis prepare / parents, em_debris, convert_ubt_data, make_muon_distribution, the neutrino flux and
  nu stores. New stores / seeds / prepared files record `pot_per_spill = 4e13` (io.write_store), so nothing is scaled
  twice. Source builders (tail_combine, build_enriched, build_pool) and the 2018-file comparisons keep their 5e13
  internals (same normalisation as the 2018 file); build_pool records pot_per_spill = 5e13 in pool.json.
- Effect (checked): legacy stores read x 0.8 (test stores; shield parents 0.8000); 15-year totals of single-muon
  backgrounds unchanged, rates per spill x 0.8, combinatorial pairs x 0.64 per spill / x 0.8 per 15 years.
  Result files made before this keep their 5e13 numbers until re-run.
- **Shared geometry:** `geometry.Z_SHIELD_END`, `SHIELD_END_HALF_X / _Y` (were in shield_iron, fastveto.debris,
  fastveto.targets, shell); `geometry.vessel_half_widths(z, clip=False)` takes numpy or torch and clips on request, used
  by muon_types.side_plane_x, em_debris, shell.half, vessel_surface_map.vhalf; rock_apply.in_vessel = geometry.inside_vessel.
  Checked identical to the old copies (side_plane_x within 4e-16 m).

## 2026-11-09 — origin/main: PR #13 (Shah Rukh, `veto-end-to-end`, 53dcc57) fetched, not yet merged here

- Shell storage, source-pool / 2018-file comparisons, EM debris (g4slab, emdebris, debris_maps), CPU transport, cavern-rock
  DIS (g4rock, rock_hits / rock_library / rock_apply), splitting sandbox, docs/STATUS.md, full_sim_inventory.md.
- Trial merge into `veto-shield-dis`: clean, no file touched by both. Review notes: macOS binary cpp/emdebris/emdebris
  committed; ~20 files with sqasim / CSCS paths; two EM-debris implementations; muon flow quoted at 4e13 PoT / spill.

## 2026-11-09 — tracking cuts on the MS7 K0-decay tracks (branch `veto-shield-dis`, uncommitted)

- `bin/compare_shield_generators.py`: new rows "K0/K_L/K_S-decay trk (3/4 st, p>1)" and "K0 decays, 2 trk": |p| > 1 GeV
  and >= 3 of 4 straw stations with a hit (`straws.layer_states` + `digitize` + `hit_summary`, field map RK4).
  Re-made `results/shield_dis/generators.{md,json}` from the existing chain outputs (no regeneration); previous table
  kept as `generators_2026-11-06.*`. Summary in docs/shield_iron_dis.md.

## 2026-11-06 (night) — DIS generator comparison for the MS7 K_L (D3) (branch `veto-shield-dis`, uncommitted)

- `shield_dis.py generate --mu-split {pythia,geant4,crossover,w2}` (default still `pythia`); split stored in the json.
- `bin/g4ship.sh build` also builds `g4dis` against FairShip's Geant4 10.7.3; `dis_library.py` finds it in cpp/build_ship.
  New library `data/interactions/dis/mu_library_iron_g4v1073.npz` (iron Geant4 part with 10.7.3, 1 M events).
- New `cpp/pythia8dis/` (pythia8_dis.cc + build.sh, CVMFS Pythia 8.230: NC DIS / photoproduction modes) and
  `bin/p8_library.py` -> `mu_library_iron_p8dis.npz` (Pythia 8 DIS Q2 > 1, W > 2; vertices 0, weak decays by
  fastveto.decay because Pythia 8.230 vertices are unusable) and `mu_library_iron_p6q2.npz` (Pythia 6 with Q2 > 1).
- New `bin/compare_shield_generators.py` (chain vs FairShip table), `bin/kl_q2w.py` (Q2 / W of the K_L-making
  interactions), `bin/plot_dis_sigma_generators.py`.
- Runs: results/shield_dis/shield_muon_fs2026_{crossover,geant4only,geant4v1073,p6q2,p8dis}.{npz,json}; summary in
  docs/shield_iron_dis.md "DIS generator comparison". Geant4 only = K_L / 2.6; Geant4 10.7 = 11.2; FairShip still
  ~9-15x below; 86 % of the default K_L from Q2 < 1.
- Note: under heavy load some g4ship workers fail to load libG4physicslists.so from CVMFS; the pool restarts them and
  all chunks complete (checked: 1000/1000).

## 2026-11-06 (evening) — C5 repeated-evt_n correction made consistent (branch `veto-shield-dis`, uncommitted)

- `weights.corrected_weights(..., counts=None)`: optional repetitions counted on the full sample (for subsamples).
- `make_design_dataset.py`: the repetitions k of each evt_n are now counted on ALL old rows of its type before the
  per-type cap (before: inside the capped subsample, k under-counted); weight w*scale/k, type total kept per file.
- `run_selection.py --pairing kdtree`: all copies kept with `corrected_weights` like time-window (C5);
  `--kdtree-unique` restores the old one-entry-per-evt_n (`keep_unique_events`, temp1.py) behaviour.
- Decision (Guglielmo, 6 Nov): keep every copy, divide by the repetitions, rescale to keep the rate.
  Not re-run: datasets / results built before keep the old weights until regenerated.

## 2026-11-06 — re-runs started; fixes (branch `veto-shield-dis`)

- `fastveto.dis.MuonLibrary.sample`: with split "pythia" an interaction below the Pythia grid had zero cross section in
  both parts and picked the empty Geant4 part (StopIteration): now the kept part, nearest grid energy with events.
- `bin/em_debris.py` also saves the raw plane crossings (`em_raw_n*_d*.npz`) to rebin the maps for new inputs.
- New `bin/retype_store.py`: muon types of a store with the FairShip geometry + field into new columns `*_fs`.
- `bin/g4ship.sh`: geant4.sh may return non-zero in non-interactive shells, which killed background jobs silently
  under `set -e`: now sourced with `set +e`.
- Results: DIS-parent store re-typed (selected 12.96 -> 16.81 % of the weight, ~8.2 -> 10.6 kHz); peak maps at
  31.13 .. 41.13 m (UBT 8.2e6, first fin slice 1.37 / 1.85e9 per spill); shield chain K_L 5 604 decays / spill (D3).
- Runs (logs/rerun_2026-11-06): EM debris 2e4 and 2e5 muons (1.25e9 e/gamma per spill leave the shield; first fin
  planes 1.2e8 e + 8.5e8 gamma per spill); MS7 shield chain muons (Pythia only: 1.78e5 interactions / spill),
  peak maps at the new positions, DIS-parent store re-typed.

## 2026-11-06 — B2 EM debris with the host Geant4 (branch `veto-shield-dis`)

- New `lib/veto/em_debris.py`, `bin/em_debris.py`: g4shield (MS7 iron + field) -> e / gamma at 29.90 m -> g4vessel
  to the UBT / fin planes; maps in the muon-map binning; `reco.plane_rate_map` adds P_e e + P_gamma gamma when the
  maps carry em_*; `run_reco.py --em-debris`. `bin/g4ship.sh build` also builds g4shield; shield_iron finds it.
- `em_debris.DEPTH_X0` = whole last 50 cm (28.5 X0), `--depth` [X0]: muons carried to the slab with
  `shield_iron.curved_points`, `init_g4shield(z_iron0=)`. 3 X0 gives only 17 % of the full-depth yield.

## 2026-11-06 — B4 EventCalc signal in run_reco (branch `veto-shield-dis`)

- `signal_eventcalc.BENCHMARK_SET`, `selected_signal()`: EventCalc decays through the common selection (field, both IP
  categories, max_tracks); `run_reco.py --signal eventcalc`: weighted average signal loss + per-benchmark results.

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
