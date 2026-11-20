# UBT / SBT distributions of the background classes (20 Nov 2026)

Where and how the three background classes (plain muons, muon DIS, neutrino DIS) cross the Upstream Background Tagger (UBT)
and the Surrounding Background Tagger (SBT) of design R2 (`designs/R2_sbt_ubt.json`, FairShip layout since 5 Nov).
Everything here is **truth level and design-geometry only**: no reconstruction, no random hits, no tagging algorithm.

## Inputs (Nov-4 source, code: worktree `nov4-source`, branch veto-nov4-source, read-only)

| class | source | statistics used | weight |
|---|---|---|---|
| mu only | muon store `design-optimization-dataset-nov4-shell-3220seeds-2026-11-11`, types 0-6 (all muons reaching z = 30.6 m) | first **800 of 3220 spills** (rows are ordered by spill), weights x 3220/800; types 5/6 (straight_past, stops_between, 328 M rows) further subsampled to 20 % (weights x 5) | muons / spill |
| DIS mu, vessel region | evstore `vessel_mu_R2_sbt_ubt` (helium, air, Mylar wall, air upstream 29.9-31.13 m, UBT plane material) + the **parent muon** from z = 30.6 m to the vertex (DIS parent store, `parent_row`; check: vertex on the parent line to 0.00 cm) | all 5.8 M events | interactions / spill |
| DIS mu, MS7 last 50 cm | evstore `ms7_muon_v0s` (products at the shield exit plane z = 29.90 m) | all 20.7 M events | interactions / spill |
| DIS nu, vessel region | evstore `vessel_nu2026_R2_sbt_ubt` | random 25 % of the 13.2 M events (weights x 4) | interactions / spill |
| DIS nu, MS7 last 50 cm | evstore `ms7_nu2026` | all 2.0 M events | interactions / spill |

Per 15 years: x 1.5e7 spills (4e13 p.o.t. per spill).

### Model
* Charged tracks = straight segments of the stored trees (start -> end, decays in flight as in `evstore.propagate`); muons =
  the stored path A (30.6 m) -> K (bounce) -> B (82 m). No field before the spectrometer, no material (no conversions, no
  hadronic interactions, no delta rays/EM debris in the cells). Neutral particles are never counted.
* **UBT**: crossing of the straw plane z = 30.73 m (aperture |x| < 0.5, |y| < 1.35 m) and of the tile plane z = 31.13 m
  (|x|, |y| < 2 m; no dead borders, efficiency 1). "UBT" in the category plots = at least one track in the straw OR tile aperture.
* **SBT**: `sbt.shell_pieces` (20 cm LS + 2 x 0.8 cm Al outside the vessel frustum 31.13-81.13 m, 61 cell rings x 14 cells);
  a cell **fires** if 150 MeV/m x path in the LS > 30 MeV, efficiency 0.993 (R2 settings). "SBT" = at least one fired cell.
  "SR rings" = the 10 SR-389 rings of ~5 m (`sbt.sr_ring_cells`).
* **Subsets**: `all`; `vtx` = two good (p > 1 GeV, SST1 + timing) opposite-charge tracks with DOCA < 2 cm and vertex in the
  vessel (`interactions.select` without the measured-IP cut); `sel` = the full background selection as `bin/evstore_eval.py`
  (+ measured IP < 0.1 m, at most 2 tracks in SST1). For **mu only**, `sel` = type 1 (SST1 muons with p > 1 GeV and timing,
  the input of the combinatorial selection) - the pair selection itself is not applied.
* MS7 stores: code 3 = interaction with V0 undecayed; code 4 = copies with one V0 forced to decay (weight x P(decay)).
  Track-level plots of `all` = code-3 tracks + the forced-decay products (and descendants) of the copies; event-level
  numbers of `all` = code-3 events (V0 undecayed); `v0dec` = the forced V0 decays alone; `vtx`/`sel` use every event as evstore_eval.

## Key numbers (full table: `summary_table.md`, `summary.json`)

Fractions of events (weighted) reaching neither / UBT only / SBT only / both:

| class | subset | per spill | per 15 y | MC ev. | neither | UBT only | SBT only | both |
|---|---|---|---|---|---|---|---|---|
| mu only, SST1 p>1 GeV + timing | sel | 8.2e3 | 1.2e11 | - | 0.000 | 0.144 | 0.007 | 0.849 |
| mu only, all SST1 (types 0-2) | sst1 | 3.2e4 | 4.8e11 | - | 0.005 | 0.042 | 0.366 | 0.587 |
| mu only, veto-only (type 3) | vetoonly | 2.5e4 | 3.7e11 | - | 0.001 | 0.002 | 0.261 | 0.737 |
| DIS mu vessel (incl. parent) | all | 2.16 | 3.2e7 | 5.8 M | 0.000 | 0.008 | 0.005 | 0.987 |
| DIS mu vessel (incl. parent) | sel | 1.6e-4 | 2.4e3 | 1426 | 0.000 | 0.011 | 0.000 | 0.989 |
| DIS mu vessel, products only | sel | 1.6e-4 | 2.4e3 | 1426 | 0.025 | 0.000 | 0.947 | 0.028 |
| DIS mu MS7 | all | 9.3e5 | 1.4e13 | 20 M | 0.025 | 0.905 | 0.000 | 0.070 |
| DIS mu MS7 | vtx | 2.5 | 3.8e7 | 807 | 0.000 | 0.116 | 0.000 | 0.884 |
| DIS mu MS7 | sel | 4.1e-3 | 6.1e4 | **2** | 0 | 1.000 | 0 | 0 |
| DIS nu vessel | all | 0.0138 | 2.1e5 | 3.3 M | 0.100 | 0.017 | 0.755 | 0.128 |
| DIS nu vessel | sel | 4.0e-7 | 6.0 | 101 | **0.273** | 0.000 | 0.526 | 0.200 |
| DIS nu MS7 | all | 1.34 | 2.0e7 | 0.5 M | 0.270 | 0.182 | 0.000 | 0.548 |
| DIS nu MS7 | sel | 6.6e-7 | 9.9 | 121 | 0.052 | 0.000 | 0.011 | 0.937 |

Rates on the detectors (per spill, = Hz):
* **mu only**: tile plane (4 x 4 m) **1.28e9 muons** (the flames at |x| ~ 2 m, |y| < 0.5 m; the window region is almost empty),
  straw plane 1.2e4; SBT **2.05e6 fired cells** (matches the 2.07 MHz quoted in `lib/veto/sbt.py`), 39 % / 55 % on the +x / -x
  walls, 2 % / 4 % top / bottom, 22 % in the first SR ring; 77 % of them from muons that miss SST1 and never cross a vessel
  wall surface (types 4-6: they graze the 21.6 cm shell outside it). Each SST1-selected muon fires ~8 cells on average
  (it runs along a side wall).
* **DIS mu vessel**: tile 2.3, straw 0.14 tracks, SBT 81 fired cells per spill (products alone: 27). By target the Mylar wall
  dominates (1.47 of 2.16 interactions/spill). The UBT sees the **parent** (91-96 % of the tile crossings are muons); the
  SBT sees the products (products-only: 94 % of all events fire >= 1 cell, mean several cells). For the selected events,
  the products alone leave 2.5 % with no UBT/SBT signal - those are tagged only through the parent's UBT hit.
* **DIS mu MS7**: the scattered muon dominates (75 % of tile crossings); 91 % of events reach only the UBT (products exit the
  iron at |x| < 1.7 m and stay near the axis); forced V0 decays (`v0dec`, 2.1e3/spill) never cross the UBT (decay downstream)
  and fire the SBT in only 16 % of the cases.
* **DIS nu vessel**: products hit the SBT (76 % SBT only), little UBT (1.7 %, from air/UBT-material vertices upstream);
  10 % of all and **27 % of the selected** events (26 of 101 MC events, ~1.6 per 15 y) leave no truth-level signal in
  either tagger. Composition of SBT hits: pi 41 %, mu 25 %, p 21 %, e 10 %.
* **DIS nu MS7**: 27 % of all events reach neither (products stopped or going outside both); selected events (121 MC) are
  94 % "both". Its V0 decays: 23 % fire the SBT, none the UBT.

## Plots
Comparisons (all classes, colour = class):
* `cmp_categories.png` - fraction of events reaching neither / UBT only / SBT only / both, all and selected events.
* `cmp_momentum.png` - |p| of the charged tracks crossing the straw / tile aperture and of tracks firing SBT cells (absolute per spill).
* `cmp_multiplicity.png` - per event: tracks in the straw aperture, in the tile aperture, fired SBT cells (normalised; mean in legend).
* `cmp_sbt_z.png` - fired SBT cells per cell ring along z (absolute per spill).
* `cmp_species.png` - particle composition at the tile plane and in the fired SBT cells (DIS classes; mu only = 100 % muons).

Per class (`mu` mu only, `dmv` DIS mu vessel, `dm7` DIS mu MS7, `dnv` DIS nu vessel, `dn7` DIS nu MS7) and subset
(`all`, `sel`, `products` = vessel DIS mu without the parent, `v0dec` = MS7 forced V0 decays):
* `<cls>_<subset>_ubt_xy.png` - x-y density of the crossings of the straw (30.73 m) and tile (31.13 m) planes per cm^2 per spill,
  with the plane aperture (blue) and the entrance window (dashed); title = rate inside the aperture.
* `<cls>_<subset>_sbt_map.png` - fired cells in z vs azimuth (normalised to the local half widths: +x = 0, top = 90, -x = 180,
  bottom = -90 deg, corners +-45 / +-135 deg), the z profile (pieces in the shell vs fired cells) and the share per face and SR ring.
* `<cls>_<subset>_sbt_faces.png` - z vs transverse coordinate on each face (dotted: the face edges).
(`dm7_sel` has only 2 MC events and no SBT maps.)

## Caveats
* Straight lines, no field, no material; neutrals, conversions, EM debris and back-splash from walls/cavern are not modelled,
  so SBT occupancies are lower limits (SR-389 quotes 27 MHz > 30 MeV with the cavern-scattered particles vs our 2 MHz).
* "UBT/SBT reached" is truth: crossing/fired cell, no timing, no pointing (the real veto requires the hit to match the track),
  so these fractions are upper limits of the tagging efficiency; "neither" is a hard lower limit of what survives a perfect veto.
* Vessel DIS mu: the parent's path from 30.6 m is the stored one (interactions upstream of 30.6 m in `air_upstream` have no
  parent segment). MS7 classes have no parent (it is in the iron). DIS mu MS7 and mu only overlap physically (the MS7 parents
  are part of the muon flux); the classes are not additive.
* Small selected samples: DIS mu MS7 `sel` 2 events, DIS nu vessel `sel` 101, DIS nu MS7 `sel` 121 MC events.
* mu only uses 800/3220 spills (statistical only; spills are independent).
* Event-level "all" for the MS7 classes = code-3 events, i.e. without the V0 decays (they are in `v0dec`).

## Files / scripts
`common.py` (geometry, histograms), `muons_only.py --seeds 800` -> `hist_mu_only.npz`,
`dis_classes.py --store <name> [--frac f]` -> `hist_<store>.npz` + `events_<store>.npz` (per-event table: weight, code,
vertex, n_straw, n_tile, n_fired, vtx, sel; 2.4 GB in total, can be deleted), `plots.py` -> PNGs, `summary.json`,
`summary_table.md`. Logs: `log_mu_only.txt`, `log_dis.txt`. Run from anywhere after
`export VETO_THREADS=8; source <worktree>/bin/batch/env.sh` (the scripts chdir to the worktree; nothing in it is modified).
Runtime: mu only 15 min, DIS 40 min (vessel mu 23 min), 1 core + torch 6 threads.
