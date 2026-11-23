# FairShip TRY_2026_EM_noSBT_full vs our veto chain: MS7 production and cavern tracks at SST1 (2026-11-23)

Sample: `/eos/user/g/gfrisell/MS/TRY_2026_EM_noSBT_full` is a full-physics FairShip production, shield TRY_2026,
2018 muon input (411 947 589 μ, 1e5 per job), `--balloonOnly` (He balloon, **no SBT material**), no `--MuonBackSSTOnly`.
Two condor analysis runs on lxplus:
- **Run 1, Test A, MS7:** `Flux_MS7.py`, cluster 21080019, 100 jobs, **2671 files**.
- **Run 2, cavern at the SST:** `Origin_SST_noSBT.py`, cluster 21080523, 100 jobs, **3086 files**.

The scripts are in `../lxplus_scripts/`.

## Method and normalisation
- **Merge.** `scripts/merge_parts.py` merges the 100 + 100 part pickles, which were copied to `data/flux_parts`,
  `data/origin_parts`. The outputs are `data/flux_ms7_merged.pkl` (3.3 GB) and `data/origin_sst_noSBT_merged.pkl` (1 GB).
- **Bug found and handled at the merge.**
  - In both condor scripts `run_id = path.split('/')[-2]` was `''`. So the `job` column is empty, and
    `Entries per files` / `jobs` hold one key per part.
  - The scripts' own `merge` summary would therefore use n_jobs = 1 and get the normalisation wrong. The rows themselves
    are complete.
  - Fix: each row gets its part index and a file index. The file index is found where the event index restarts, because
    rows are written file by file.
  - The restarts give exactly 2671 and 3086 files. Each of the 100 parts matches the `FILE_LIST[i::100]` split.
  - No part is empty or unreadable, and the deprecated/bad lists are empty.
  - The 3-job condor test did not write into these folders: the parts have uniform timestamps and all 100 match the split.
  - Event key = (part, file, event).
- **Rates per 1 s spill of 5e13 PoT** = Σ MCTrack weight × 411947589 / (n_files × 1e5):
  - Run 1: × 1.5423.
  - Run 2: × 1.3349.
  - Per 4e13: × 0.8.
- **MC entries for the full input** = MC × 4119.5 / n_files.
- **Statistical error** = √Σw² × factor.
- **z convention.** All z are FairShip z unless marked; z_ours = z_FS − 2.14 m.
  - The MS7 back face in this geometry is at **z_FS 32.18** (30.04 ours), the end of the muon-nuclear vertices. Our
    Geant4 MS7 block ends at 29.89 ours.
  - UBT scoring plane 2 is at 32.28.
  - The SST scoring plane is at z_FS 83.71 (81.57 ours), covering |x| < 2, |y| < 3 m.
- **Our references** are per 5e13 spill:
  - MS7 numbers: `veto/results/shield_dis/generators_natural.md`, G4 10.7 chain column, and
    `shield_muon_fs2026_geant4v1073.json`.
  - Rock numbers: `docs/investigations_2026-11-20/cavern_rock/README.md`, per 4e13.
  - Our rock spectrum: `scripts/ours_rock_spectrum.py` → `data/ours_rock_tracks.npz`. This is a read-only load of
    `apply_D6_straw_mod_draws.npz` with `rock_all`. It gives 1.90e4 good per spill, against the official 1.84e4.

**Correction to the brief.** The "π at SST1 1.2e5, p 7.4e4 / spill" values are our MS7 *exit* rates (`…geant4v1073.json`:
π± exit 1.16e5, p 7.4e4). Our SST1 rates are π 5.85e3 and p 1.28e3 / spill (generators_natural.md). The expected
"~480 MC" refers to SST1 π.

## A. Production in MS7 (`results/ms7_tables.txt`, `ms7_summary.json`, `ms7_kl_decay_tracks.txt`)

| per 5e13 spill | FairShip noSBT_full | ours (G4 10.7 chain) | FS / ours |
|---|---|---|---|
| μ through MS7 footprint (\|x\|<1.71, \|y\|<1.27) | 2.39e8 (1.0e7 MC), 97 % UBT hit | 2.5e8 | 0.96 |
| μ-nuclear vertices, last 50 cm | 2.9e5 ± 0.1e5 (12 902 MC; 1.2e3 / 1e6 μ) | 7.9e5 | 0.36* |
| K_L made, last 50 cm (all processes / direct μN) | 7.5e3 ± 1.0e3 (467 MC) / 3.6e3 | exit K_L 3.4e3 | production ≥ exit, consistent |
| K_S, K±, made, last 50 cm | 7.2e3, 2.4e4 | exit 5.8e2, 8.4e3 | |
| π±, p, n made, last 50 cm | 6.2e5, 2.9e5, 1.2e6 | exit 0.93–1.16e5, 0.59–0.74e5, 0.69–0.87e5 | |
| **π± at SST plane, made in Magn7** | **5.6e3 ± 1.3e3** (319 MC, 426 full-input MC) | **5.85e3** (expected ~480 MC) | **0.95** |
| p at SST plane (all p / p > 1) | 2.2e3 ± 0.5e3 / 1.37e3 ± 0.12e3 (174 MC) | 1.28e3 | 1.7 / 1.07 |
| K± at SST plane | 82 ± 29 (8 MC) | 384 | 0.2 (low MC) |
| K_L-decay tracks at SST (K_L made in MS7) | 144 ± 38 (14 MC) | 97 | 1.5 ± 0.4 |
| K_S-decay tracks at SST | 82 ± 29 (8 MC) | 0.9 | open, low MC |

\*The vertex counts are not like-for-like:
- FairShip only sees vertices whose products were stored (E > cut or hits).
- In FairShip the vertex density rises about 20× in the last 50 cm (`ms7_munucl_vertex_z.png`; vertices at |x| 1.27–1.71 m,
  |y| < 0.23 m).
- Ours counts every interaction of the 4.9e8 parents entering the last 50 cm.

The yields per track that reaches the SST are the meaningful comparison.

**Conclusion A:**
- FairShip's MS7 production agrees with ours at every step that can be compared:
  - muon flux: 0.96;
  - π at SST1: 0.95;
  - p (p > 1): 1.1;
  - K_L-decay tracks: 1.5 ± 0.4;
  - K_L production in the last 50 cm (7.5e3) is consistent with our 3.4e3 *exit*.
- The ~100× deficit of the old `TRY_2026_EM_SST1_origin` sample does not appear here.
  - The old sample had 5 MS7 π MC and 640 π per spill.
  - This sample has 426 full-input MC against our expected 480.
  - So the deficit was a property of the old production (SBT material and/or event filtering), not of Geant4 or of our
    chain.

## B. Cavern (concrete) tracks at the SST plane (`results/cavern_tables.txt`, `cavern_summary.json`)

| per 4e13 spill, charged, made in concrete | FairShip noSBT_full | ours (rock library) | ours / FS |
|---|---|---|---|
| all at plane / aperture | 2.07e7 (7.9e5 MC; 99 % e±, median 9 MeV) | 6.4e5 | 0.03 |
| hadrons + μ only, any p | 1.73e5 (median 0.70 GeV) | 2.44e5 (median 0.69) | 1.4 |
| p > 1 GeV | 4.6e4 ± 0.4e4 (2021 MC) | 7.7e4 | 1.7 |
| **p > 1 GeV + TD (good)** | **5.5e3 ± 1.4e3 (262 MC)** | **1.84e4** | **3.4 ± 0.9** |
| good π± / p / e± / K± | 2.7e3 (105) / 2.7e3 (150) / 58 (7) / 0 (0) | 8.8e3 / 7.7e3 / 0.9e3 / 0.9e3 | 3.4 / 2.8 / 16 / – |

Where the tracks come from:
- 97–99 % from the −x wall, whose inner face is at x = −4.56 in FairShip as in our model.
- Creation z (ours frame) for p > 1: 60 / 69 / 76 m (10 / 50 / 90 %).
- Depth for p > 1: median 6.2 cm, 90 % 22 cm. Ours: 5.3 / 19 cm.
- Good tracks are 80 % positive. Ours: 72 %.
- The momentum spectra of π and p have the same shape. Median π: 0.57 / 1.40 / 1.69 GeV in FS against 0.57 / 1.49 /
  1.63 in ours, for any p / p > 1 / good.

**Conclusion B:**
- FairShip confirms that most cavern products reach the SST with almost no momentum: 2e7 soft e± per spill.
- It does **not** confirm a large overestimate of the hadron tail that matters.
  - At p > 1 GeV ours is only 1.7× above FairShip.
  - At the good-track level ours is 3.4× ± 0.9 above (π 3.4×, p 2.8×). The 28× (π 55×) of the old cross-check is not
    confirmed; the old sample had SBT material.
- The good-level factor breaks down as about 1.7 from production/p > 1 and about 2.1 from the timing detector:
  - FS real TD hit: 12 % of p > 1 tracks;
  - our field-map TD acceptance: 25 %.
- K± (ours about 30× higher, on 12–16 MC) and e± at p > 1 (4–16×) are the species where our library is clearly higher.
  Candidate cause: the Pythia 6 high-ν part.

## Context: tracks at the SST plane by origin material (per 4e13; all / p > 1 / p > 1 + TD)

| material | all | p > 1 | p > 1 + TD |
|---|---|---|---|
| Concrete | 2.07e7 | 4.6e4 | 5.5e3 |
| iron (shield, yokes) | 8.4e6 | 2.0e4 | 1.33e4 |
| air (cave) | 1.4e7 | 5.9e3 | 3.5e3 |
| He balloon (mylar_linerHe) | 8.2e5 | 66 | 16 |
| helium | 3.3e5 | 805 | 493 |
| steel / scintillator / other | 2.5e5 / 9e4 / 1.4e5 | 0 / 33 / 0 | 0 / 33 / 0 |

## Plots (`plots/`)
- `ms7_muon_flux_p.png`: momentum of the muons leaving MS7.
- `ms7_munucl_vertex_z.png`: μ-nuclear vertex density against z.
- `ms7_hadron_p_spectra.png`: hadrons made in the last 50 cm, momentum at production.
- `ms7_to_sst_p_spectra.png`: Magn7-made tracks at the SST plane.
- `cavern_p_spectrum_fs_vs_ours.png`: FS against ours, all, hadrons, p > 1 + TD; log bins 0.01–100 GeV.
- `cavern_creation_z_x.png`: creation z and x of the concrete tracks.
- `sst_p_by_material.png`: momentum at the SST plane by creation material.

## Caveats
- **MC statistics.** The good concrete tracks are 262 MC (±26 %); MS7 π at SST 319 MC; K_L-decay tracks 14 MC; the K_S
  and K± numbers are under 20 MC.
- **Straight lines against field.** Ours uses straight lines from the rock surface or iron exit, with a field-map TD
  acceptance. FairShip uses full transport with real hits. The TD step accounts for about 2× of the good-level factor.
- **Planes.** The FS SST scoring plane (81.57 ours, |x| < 2, |y| < 3) is 0.5 m upstream of our SST1 (82.08); the
  apertures are similar but not identical.
- **5e13 against 4e13.** MS7 tables are per 5e13 and cavern tables per 4e13; 15 y gives the same PoT.
- **Geometry.** FairShip TRY_2026 MS7 ends at 30.04 ours, against our 29.89. There is no SBT in this production, which
  is like-for-like with our rock library. A real SBT shell would remove about 60 % more (cavern_rock README, section 5).
- **Storage.** FairShip keeps MCTracks with hits or above its energy cut, so MS7 vertex counts are a lower bound.
  Hadrons reaching the SST are unaffected.
- **Different file sets.** The SST-plane analysis uses the 3086 files of run 2 and MS7 uses the 2671 of run 1; each is
  normalised on its own.
- **Disk.** `data/` holds 8 GB (parts plus merged pickles). The parts can be deleted, because the merged pickles contain
  them all.

## TD acceptance test (2026-11-23, `scripts/td_acceptance_test.py`, `td_miss_diagnostic.py`, `ours_rock_td4x6.py`; `results/td_acceptance_test.*`, `plots/td_miss_xy.png`)

Question: is the factor of about 2 at the timing-detector step our acceptance, or a difference in the track distributions?

Method: our acceptance (veto lib `interactions.charged_tracks` → `spectrometer.timing_acceptance`, field map to
Z_TIMING 94.01) is applied to FairShip's own tracks, starting from their state at the SST plane.

| FairShip charged tracks, p > 1 GeV | our TD acc. (5.5 × 6.5 m) | our TD acc. (4 × 6 m) | FairShip real TD hit |
|---|---|---|---|
| made in concrete (2018 MC) | 0.266 | 0.140 | 0.119 |
| all origins (3469 MC) | 0.453 | 0.325 | 0.314 |

- **FairShip TD hits lie entirely inside our acceptance** (FS-only 0.0 %).
- With our 5.5 × 6.5 m aperture, only 45 % (weighted) of the tracks we accept get a real hit. The missed ones land at
  |x_t| > 2.2 (39 %) or large |y_t|.
- **FairShip's hits end at |x| = 2.0 m and |y| = 3.0 m** (100th percentile 2.14 / 3.00). In this production the TD is
  effectively 4 × 6 m, the same as `geometry.OLD_TIMING_HALF`.
- With 4 × 6 m, 85–94 % of our accepted tracks get a real hit. The rest is decays and material.
- The library's `TIMING_HALF_X, Y = 2.75, 3.25` (5.5 × 6.5 m) is therefore larger than the TD in this production.
  Which is right for SHiP depends on the current FairShip TD geometry. The geo file was not inspected (lxplus socket
  closed).
- **Our rock good rate with a 4 × 6 m TD:** 1.26e4 / spill (4e13), against 1.90e4 with 5.5 × 6.5 m.
  - π 6.1e3, p 5.0e3, e 0.67e3, K 0.54e3.
  - **Ours / FairShip at the good level: 2.3 ± 0.6** (was 3.4).
  - This is 1.7 from production at p > 1 and about 1.35 at the TD step. Of that 1.35, about 1.1 is real FairShip
    decay/material loss.
- Energy loss in air (about 4 MeV over 15 m) is negligible, and the identical spectrum shapes confirm it is not missing.

### Resolution: where the 4 × 6 m edge comes from (checked on lxplus, 2026-11-23)
The production ran `run_simScript.py -n 100000 --MuonBack --shieldName TRY_2026 --balloonOnly` from
`/afs/cern.ch/work/g/gfrisell/FairShip`, commit 78cc91ea (2026-11-05). It uses the defaults, so strawDesign 10.

- **The timing detector is 5.5 × 6.5 m in FairShip too:** `python/geometry_config.py` has `c.TimeDet.DX = 275 cm` and
  `DY = 325 cm`. Our `TIMING_HALF = 2.75 × 3.25` is correct. The annual report 2026 gives no TD size; section 4.2.6 is
  R&D only. The report's only aperture is for the straw stations, "about 4 m × 6 m".
- **The edge comes from the straw-station frames.** `strawtubes.cxx` builds each station frame as a solid box with the
  aperture removed:
  - the aperture is ±200 × ±300 cm (`strawtubes_config.yaml` width/height);
  - the frame is 49 cm wide (the "HEA 500 I-beam width");
  - it is 1 m deep in z (station_length = 50 cm half-length);
  - it is made of **steel** for strawDesign 10. The data confirm this: Tr1–4_frame = steel.
- One metre of solid steel is about 6 λ_I, so every track that crosses a station outside |x| < 2, |y| < 3 m is absorbed.
  The last such station, SST4, is 1 m upstream of the TD. The real HEA 500 is an I-beam, mostly empty, so the solid
  block overstates the material.
- **The magnet does not cause the edge.** The spectrometer magnet half-aperture is 2.2 × 3.5 m (MISIS) or 2.4 m in x,
  and our field-map "lost" test (|B| > 0.5 T) only sees the yoke at |y| > about 3.3 m.

**Consequence for our framework.** Our `timing_acceptance` propagates through the field to the TD but does not require
the track to stay inside the straw aperture at SST2–4 (84.08, 91.08, 93.08 ours). FairShip's frames enforce that.
Adding that requirement (|x| < 2, |y| < 3 at each straw station) is equivalent to the 4 × 6 m test above:
- our acceptance on FairShip tracks 0.140 against FS 0.119;
- rock good 1.26e4 / spill;
- ours / FS 2.3.

A track outside the straw aperture cannot be measured by SST3/4 anyway, so the cut is physical in either case. Not
changed in the veto code (user to decide).

### Exact straw-aperture test (`scripts/straw_aperture_test.py`, `results/straw_aperture_test.*`)
Four acceptance variants were applied to the same tracks: FairShip p > 1 GeV tracks with their real TD hit, and our rock
tracks. All transports use our field map.
- **A**, current lib: SST1 aperture, then the TD at 5.5 × 6.5 m.
- **B**, 4 × 6 m at the TD plane: the earlier proxy.
- **C**: A + |x| < 2, |y| < 3 at the centre of SST2/3/4 (84.08 / 91.08 / 93.08).
- **D**: as C, but also at the front and back faces (z ± 0.5 m) of each 1 m-deep station frame. This is closest to the
  FairShip geometry.

| | A | B | C | D | FairShip real |
|---|---|---|---|---|---|
| FS concrete p>1: accepted fraction | 0.266 | 0.140 | 0.198 | **0.168** | 0.119 |
| FS concrete p>1: FS hit among accepted (weighted / unweighted) | 0.45 / 0.61 | 0.85 / 0.93 | 0.60 / 0.80 | 0.70 / **0.87** | |
| FS all origins p>1: accepted fraction | 0.453 | 0.325 | 0.370 | **0.349** | 0.314 |
| FS all origins p>1: FS hit among accepted (w / unw) | 0.68 / 0.77 | 0.94 / 0.95 | 0.83 / 0.89 | 0.88 / **0.92** | |
| ours rock good / spill (4e13) | 1.90e4 | 1.26e4 | 1.47e4 | **1.36e4** | FS 5.48e3 ± 1.4e3 |
| ours / FairShip | 3.5 ± 0.9 | 2.3 ± 0.6 | 2.7 ± 0.7 | **2.5 ± 0.6** | |

- SST2 removes almost nothing.
- SST4 removes the most (4.5 %), then SST3 (1–3 %), then the station faces (1.5–3 %).
- B was too tight: it removed tracks that are still inside SST4 but move out to |x| > 2 m before the TD.
- After D, the remaining TD-step difference on identical tracks is 0.168 against 0.119 (weighted) and 0.87 (unweighted)
  of the tracks accepted. This is consistent with π decays over the last ~12 m, material and TD efficiency, plus weight
  fluctuations: the top-5 weights carry about 50 % of the missed weight.
- **Updated cavern conclusion:** with the straw aperture enforced, ours / FairShip = **2.5 ± 0.6** at the good level.
  This splits into 1.7 from production at p > 1 and about 1.45 at the TD step. About 1.15–1.4 of that TD step is physics
  (decays/material) absent from straight lines.

### Good-track definition changed in the veto code (2026-11-23)
On Guglielmo's instruction the veto library now defines a **good track** as p > 1 GeV, inside the straw aperture
(|x| < 2, |y| < 3) at **all 4 stations**, and reaching the timing detector. This is `spectrometer.good_track_acceptance`
(N_STATIONS_GOOD = 4); see the veto CHANGELOG for 2026-11-23. SST1 + TD is NOT a good track.

- **Our rock library under the new definition:** 1.47e4 / spill (4e13), against 1.90e4 under the old SST1 + TD.
  - π 7.1e3, p 5.9e3, e 0.76e3, K 0.65e3, μ 0.28e3.
- **Against FairShip (TD hit, 5.48e3 ± 1.4e3):** ours / FS = **2.7 ± 0.7**.
  - This splits into 1.7 from production at p > 1 and about 1.6 at the TD step.
  - Of the 1.6, about 1.2–1.4 is FairShip decay/material loss plus the frame faces (variant D gives 2.5).

## MS7 good tracks at SST1 (2026-11-23, `scripts/ms7_good_tracks.py` → `results/ms7_good_tracks.{json,txt}`)
Same good-track filter on both (p > 1, straw aperture at all 4 stations + TD, field map), per 4e13 spill:

| MS7-made, at SST1 | FS all | ours all | FS good | ours good | ours / FS good |
|---|---|---|---|---|---|
| π± | 4.36e3 | 5.84e3 | 3.56e3 ± 1.0e3 (210 MC) | 3.42e3 | 0.96 ± 0.27 |
| p | 1.75e3 | 1.28e3 | 755 ± 79 (92) | 814 | 1.08 ± 0.11 |
| K± | 66 | 384 | 57 ± 22 (7) | 263 | 4.6 ± 1.8 |
| e± | 4.2e6 | 3.4e3 | 3.2e3 ± 1.0e3 (206) | 242 | 0.08 |
| K_L-decay tracks | 98 | 97 | 49 ± 20 (6) | 36 | 0.7 ± 0.3 |

- **Correction:** `generators_natural.md` is already per 4e13: `compare_shield_generators.py` rescales the pre-9-Nov files.
  This script reproduces π 5.84e3 and p 1.28e3 with ×0.8 applied to the raw npz. So the "ours" MS7 numbers above (and in
  the first MS7 table) were read as per 5e13 by mistake. At all p ours/FS for π is 1.34, not 1.05.
- At the good-track level the MS7 hadrons agree (π 0.96, p 1.08). FairShip's e± are EM (δ-rays, conversions in the iron),
  which our DIS-only chain does not have.

## Numbers for the MS note (Optimized Muon Shield with Tuned Return Yoke), 2026-11-23
`scripts/note_numbers.py` → `results/note_numbers.txt/.json`. Units as in the note (Σw × 411947589/n_input ≈ per 5e13).
- **Geometry differs from the note.** FairShip `TRY_2026` (commit 78cc91ea, `python/geometry_config.py`) = TRY6 +
  **0.3 m gap between the M1 and M2 yokes** (integration layout 2026-0.1); MS7 ends at z_FS 32.18 (note TRY6 32.04).
  Full physics (no `--FastMuon`), He balloon only, SND design 2 on. Only events with a real hit are stored.
- **UBT:** muons in the UBT active area 2.44e8 vs note G4 2.72e8 Hz (0.90); MS7 face 2.39e8 vs 2.48e8 (0.97).
  Outside the UBT the sample is incomplete (event filter), so no full 8×8 m map. No SND scoring points → no SD check.
- **DSST1 plane (|x|<2, |y|<3), all particles [kHz]:** μ from target 22.5 ± 3.5 (p>1: 12.1, p>1+TD: 7.1); μ from
  decays in flight 53.9 (p median 0.3 GeV; p>1: 4.0); e± 5.3e4 (p>1: 7.7); π± 104 (26 / 9.0); p 126 (41 / 5.7);
  γ 7.4e4; n 205. All charged: 5.35e4 / 91 / 28.6 (all / p>1 / p>1+TD). Non-μ charged p>1+TD by origin: iron 9.4,
  concrete 6.9, air 3.7.
- **μ from target vs note Table (≥3 stations):** plane 22.5 ± 3.5; ≥3 stations with our field map 10.3 ± 2.4 (note
  16 ± 3, CUDA plane 25.7 ± 3.2). Composition differs from the note even for weight-7.69 muons (P0 <40 / 40–200 /
  200–400 counts): noSBT plane 272/136/175, CUDA plane 202/334/773 (506 M input), note hit 26/184/239.
  - Core muons (P0>200) reach DSST1 at |x| median 1.88 m (note 1.69): ~20 cm further out, many leave the 2 m straw
    aperture → fewer core muons. Likely the 0.3 m gap (MS7 shifted downstream); not verified.
  - Low-P0 muons at the plane are soft (median 0.34 GeV) and mostly bounced (CUDA: 78 %).
  - Matching muons by (P0, w): only 14/474 note hit muons found; expected for the stochastic trajectories
    (Robustness section: 91 % of DSST1 events seen in 1 of 200 seeds).
- Missing for a like-for-like note number: the real straw-station count (strawtubesPoint) per muon, not extracted.
- **2026-11-23, put in the MS note:** user chose to report only the *secondaries* (not the target-muon comparison) →
  Overleaf `Section/Analysis.tex` Subsection `sub:secondaries`, Table `tab:secondaries` (charged, non-input, DSST1
  plane, per 5e13): all 5.4e4 kHz (99.5 % e±, median 11 MeV); p>1: 79 ± 6 (cavern −x wall 57.7, MS7 12.1, air 7.4,
  MS4–6 0.6, vessel 1.1); p>1+TD: 21 ± 3 (MS7 9.0, walls 6.9, air 4.4, vessel 0.6, MS4–6 0.5). MS7 secondaries made
  in the last ~15 cm (median z0 32.03), |x0| 0.9–1.7. Straw-station extraction job: not launched (not needed).
- **2026-11-23, plots for the note:** `scripts/plot_secondaries.py` → `plots/note/` (copied to Overleaf `TRY6/Secondaries/`):
  `secondaries_origin_map.png` (production point top/side view + rate vs production z, by origin, p>1; hit_categories
  style), `secondaries_xy_DSST1.png` (non-μ, p>1, colour = p; Scatter_Momentum_Map style, 2904 tracks 74.8 kHz),
  `secondaries_p_pt_DSST1.png` (non-μ, all p, 60 log bins/axis; P_vs_Pt style). Figures `fig:secondaries_origin`,
  `fig:secondaries_DSST1` in Subsection `sub:secondaries`.
