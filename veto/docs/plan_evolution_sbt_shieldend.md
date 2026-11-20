# Plan: design evolution with shield-end planes, compared with the traditional SBT

Status 2026-11-11 (Shah Rukh + Claude). What has to be done before we run (1) the design evolution (`evolve/`), (2) a
design with Guglielmo's shield-end planes and (3) the traditional SBT as a reference, all through the same chain.
Numbers are D6_straw_mod, Nov-4 muons, 15 years (1.5e7 spills x 4e13 p.o.t.), after the charge-sign fix (docs/STATUS.md).

## Where we stand

| Background (after veto, 15 y) | Events | Seen by the veto? |
|---|---|---|
| K_L / K_S / Lambda from muon DIS in the last 50 cm of MS7 | ~1e4 (13 MC selected) | no, neutral; ~30 % tagged via the scattered muon |
| Muon DIS in the helium | ~520 | yes; misses = parent crossed no plane (side entry) |
| Muon DIS in the vessel walls | ~22-44 (under-sampled) | yes |
| Neutrinos (vessel + MS7) | ~2.5 | no (floor) |
| Rock (neutrals in the helium, K_L) | < 0.1 | no |
| Combinatorial | 0.003-0.004 | yes |
| Signal lost (accidental vetoes incl. EM debris) | 7.3 % | |

What the evolution scores today (`evolve/objective.py`, `evolve/evaluate_designs.py`): combinatorial (`inputs.npz`), muon
interactions in helium / PVC / planes and neutrinos in the helium (`results/interactions/prepared_{muon,nu}<tag>.npz` from
`bin/prepare_interactions.py`), cost, rate limits. Not included: MS7 V0, MS7 neutrinos, rock, EM debris in the accidentals.
The plane model (`lib/veto/design.py`) knows two kinds: `ubt` and `fin`, both planes at fixed z.

## A. Bring the evolution onto the current physics (needed for all three runs)

| # | Task | Notes | Effort |
|---|---|---|---|
| A1 | Pair inputs from Nov-4 with the fixed charges | done: `results/reco/inputs_nov4_q.npz`; pass `--inputs inputs_nov4_q.npz` | - |
| A2 | Interaction caches on Nov-4 | rerun `bin/prepare_interactions.py` on `design-optimization-dataset-nov4-shell-300seeds-2026-11-10` (the old caches are from older datasets and the flipped charge) and on the 2026 neutrino samples (`data/interactions/genie/apply2026_*`, Geant4 downstream where it matters); tag `_nov4q` | hours |
| A3 | EM debris in the accidentals | `run_reco --em-debris` exists; `evaluate_designs.py` has to pass it (maps `results/em_debris/em_maps.npz`, Nov-4 parents) | small |
| A4 | Statistics before optimising | vessel walls: 13 MC selected, regions with none could hold ~200 -> more wall draws; helium x20 for a fine scan (reward noise ~ 2 x reward_err) | minutes of farm |
| A5 | MS7 V0 in the objective | design-independent except for the scattered-muon tag and (C) the shield-end tag. Decide with Guglielmo: count them as is, or after an analysis mass cut (visible m > m_K removes K_L semileptonic, K_S / Lambda windows) -> needs the two-track mass split of the MS7 events | small (mass split ~1 h) |
| A6 | MS7 neutrinos, rock in the objective | constants per design for now (1.0 and < 0.1) | small |
| A7 | Hold-out confirmation | best designs re-evaluated on the 560 hidden shell seeds (not only other random seeds) | small |

## B. Guglielmo's shield-end planes

| # | Task | Notes | Effort |
|---|---|---|---|
| B1 | Geometry and technology from Guglielmo | z positions in / behind the last shield section, size, technology, time resolution | input needed |
| B2 | New plane kind `shield` in `design.py` / `reco.py` | planes at fixed z like `ubt` (cells, efficiency, timing, cost), placed before the vessel | ~1 day |
| B3 | Muon states at those z | `bin/cpu_transport.py run --planes <z...>` on the Nov-4 chunks (farm GPU 0): (i) rates / occupancy (~2e9 muons per spill near z = 29 m), (ii) the MS7 parents at the planes (the MS7 store links each event to its parent row), (iii) the parents of the helium-DIS events, which the same planes may tag on their way to a side entry | ~1 GPU-hour |
| B4 | EM debris at those z | `bin/em_debris.py` stage 1 already saves the exit states; maps at the plane positions | few core-hours |
| B5 | Tagging rule | a plain "any hit in time" veto at 2e9 muons per spill kills all signal: veto only if a muon track at the plane points to the reconstructed V0 / vertex origin within (space, time) windows; define with Guglielmo, implement in `interactions` / `reco` | design + ~1 day |
| B6 | Rate limits / radiation for that technology | from B3 + B4, into the objective's constraints | small |

## C. Traditional SBT (reference design)

| # | Task | Notes | Effort |
|---|---|---|---|
| C1 | SBT specification from the collaboration | steel wall thicknesses, liquid scintillator thickness, cell size, threshold, WOM timing, efficiency | input needed |
| C2 | New plane kind `wall` | detector on the vessel surface (cells along z and around), hits from the stored muon paths (A -> bounce -> B, the `walls` bits / shell crossings) | ~1-2 days |
| C3 | Muon DIS libraries for steel and LS | `cpp/g4dis` with G4_Fe and an LS material (as helium / PVC) | hours |
| C4 | Muon DIS in the SBT walls | `fastveto.dis` / `muon_dis_apply` with the new targets; ~x100 the wall mass -> importance sampling (force neutral into the helium, vertex near the inner wall), as for the rock | ~1 day |
| C5 | Neutrino DIS in steel and LS | GENIE library v2 has Fe56; check C / H for LS; `nu_dis_apply` with the SBT targets | hours |
| C6 | Geant4 through the thick wall | `cpp/g4vessel` from one wall layer to steel / LS / steel with energy deposits per LS cell; fixed geometry, run once and store | ~1 day + tens of core-hours |
| C7 | LS response (efficiency, noise) | `cpp/g4slab` library: muons through steel / LS / steel by angle and energy -> deposit spectrum; with C1's threshold: efficiency and the delta rays / brems made in the steel. Decides the answer (0.1 % vs 1 % inefficiency = 2 vs 20 helium events) | few core-hours |
| C8 | SBT accidentals (signal loss) | muon crossings + debris on the walls (`em_debris` onto the SBT surfaces) + C7 | few core-hours |
| C9 | Technology entry `ls` in `tech.py` | segmentation, threshold, time window, cost per channel | small |

## D. The comparison

- Same Nov-4 inputs, same selection (measured IP < 0.1 m, <= 2 tracks), same seeds, hidden-seed confirmation.
- Report per design: background per source (table above), signal lost, cost, channels, peak rates vs limits.
- Runs: (1) evolution of fins + UBT (A only); (2) evolution with shield-end planes allowed (A + B); (3) full SBT + UBT
  fixed (A + C); optionally (4) fins with SBT patches where the parents enter (A + C2-C9, partial geometry).
- Expected (estimate, no simulation yet): full SBT takes the helium muon DIS from ~520 to ~2-20 (set by its
  inefficiency), with more neutrino interactions in the steel (the main risk), a higher signal loss and much more
  material and cost. The shield-end planes act on the MS7 V0 and possibly on the helium-DIS parents.

## Questions for Guglielmo

1. Shield-end planes: where (z, size), which technology, which time resolution? Also meant for shield monitoring.
2. The tagging rule for B5: what association between a shield-end hit and a vessel vertex is acceptable?
3. MS7 V0 in the objective: as is, or after a mass cut (which signal masses do we optimise for)?
4. SBT reference: which version of the collaboration's SBT (thicknesses, threshold, cell size)? Do you have the files?
5. Is the signal model of `run_reco` (target decays) the one to keep for the comparison?

## Effort and compute

Code ~1-2 weeks in total (A ~2-3 days, B ~3 days after B1, C ~1 week after C1). Compute modest: below ~100 core-hours
and ~1 GPU-hour, no new muon production (everything on the Nov-4 sample). The open physics choices (B5, C1, C7, A5)
decide the comparison more than the compute.

## Answers of Guglielmo (12 Nov 2026) and first numbers

1. **Shield-end planes: silicon strips**, also used for shield monitoring. One station 20 x 20 x 3 cm^3 or 30 x 30 x 3 cm^3;
   for now superimposed on the geometry (the stations sit in the MS7 field, not modelled yet). Rate limit 1 MHz per strip.
   Technology references: LHCb Upstream Tracker (p-in-n, 250 um; 93.5 um pitch x 5 cm strips near the beam, 187.5 um x 10 cm
   outside; SALT ASIC, 25 ns slots), ATLAS ITk strips (75.5 um pitch, 2.4 / 4.8 cm, ABCStar, 25 ns).
   Rates at the MS7 exit z = 29.90 m (Nov-4 MS7 parents carried straight from 29.20 m + the EM debris exit states, 2e5
   parents; photons x 0.25 % conversion in 300 um Si): muons 4.5e8 / spill, all in the return yokes (|x| > ~1.1 m; none on
   the axis), e+- 1.1e8, gamma 9.1e8; peak 0.37 MHz/cm^2 at x ~ -1.1 m (e+- 0.33, muons 0.03). Hottest station position
   x ~ -1.6 m, y ~ 0: 63 MHz (20 x 20) / 113 MHz (30 x 30). Per strip at the peak: UT C/D 93.5 um x 5 cm 0.02 MHz, UT A
   187.5 um x 10 cm 0.07 MHz, ITk 75.5 um x 2.4 cm 0.01 MHz: the 1 MHz limit allows strips up to 1.4-3.6 m. Rate does not
   limit the strip size; the 25 ns time slot gives ~1.6 random hits per window in a 63 MHz station -> the association has to
   be spatial. Default proposed: UT-like 93.5 um x 5 cm. Stations must sit where the parent muons are (yokes, both sides).
2. **Tagging rule**: reconstruct the K_L (V0) vertex in the vessel, trace the V0 back to the shield end, and look in the
   shield-end stations for the parent: a muon with a kink (inelastic interaction) or a vertex of several tracks, compatible in
   space (and time) with the back-traced line. (K_L -> pi l nu: the V0 direction is only partly known, missing neutrino.)
3. MS7 V0 in the objective (mass cut): explained to Guglielmo, decision pending; first measure what a visible-mass cut removes
   from the V0 background and from each benchmark.
4. **SBT reference** (SHiP Annual Report 2026 + FairShip 26.10 geometry/veto_config_helium.yaml, veto/veto.cxx): helium balloon
   350 um Mylar; liquid scintillator 20 cm between 0.8 cm aluminium inner / outer walls, 1.5 cm aluminium ribs; cells 82 cm in z
   (80 cm first ring), 2 longitudinal ribs per x side and 3 per y side; ten rings; WOM readout (FastIC+), threshold 30-90 MeV
   region-dependent (FairShip reco 45 MeV per segment), 200 ns ToT window; TRY2026 rate 966 MHz raw, 27 MHz above 30 MeV,
   5.7 % random-hit probability per window. UBT in the report = our default (4 x 4 m^2 tiles < 200 ps + six 1 x 3 m^2 straw
   layers). Also in the report: SciFi + 1 cm scintillator pads in the 3 cm gaps of MS6 (an in-shield detector concept).
5. **Signal model**: yes, the B4 benchmark set (regenerated 12 Nov).

## SBT and UBT from the official reports (12 Nov 2026)
Sources: SPSC-SR-370 (BDF/SHiP Annual Report 2025, sec. 3.3.3) and SPSC-SR-389 (Annual Report 2026, CERN-SPSC-2026-035,
sec. 4.2.2, 4.2.3, 5.4), both in Study_on_Veto_System/Optimization_Veto_System/; FairShip 26.10 geometry.
- **SBT** (SR-370): cells of AlMg4.5 (EN AW-5083) filled with liquid scintillator (LAB + 2 g/l PPO), 20 cm thick (for neutral
  products of helium interactions), mirror-polished walls; 80 cm along the beam everywhere, up to 150 cm across (test cell
  110 x 80 x 20 cm); 2 WOMs per cell, 40 SiPMs in 8 groups; time resolution ~1 ns; crossing point from the WOM light ratio,
  time difference and SiPM groups. Containers from 6 m sheets with separation walls, on I-beams (aluminium preferred, SR-389).
  SR-389: thresholds 30-90 MeV by region (FairShip reco 45 MeV per segment), FastIC+ readout, 200 ns ToT window, TRY2026 cell
  rate 966 MHz raw / 27 MHz above 30 MeV, 5.7 % random-hit probability per window; ten rings; Run-4 partial scenarios (side
  walls only, first six or last five rings). FairShip 26.10: 0.8 cm Al inner / outer walls, 1.5 cm Al ribs, cells 82 cm in z.
- **UBT** (SR-389 4.2.2): tile plane covering at least the MS7 exit face, avoiding most swept-out muons (extent being optimised;
  abstract: 4 x 4 m^2); 40 x 40 mm tiles (< 120 ps), 20 x 20 mm in the muon flames (90 ps); 99.9 %; max 2.5 MHz per 20 x 20 tile
  (TRY2026; ours on Nov-4: 2.8 MHz); tagging window +-200 ps. Straws: VERTICAL, 1 m long, 5 mm, three sections over the
  decay-volume aperture, upstream of the tile plane; 100 um, 100 kHz readout; "six 1 x 3 m^2 layers"; purpose includes
  reconstructing a vertex in the last part of the muon shield and K_S / Lambda decays before the UBT.
- **Differences to our model, to decide**: (1) tile window +-200 ps (ours: +-1.5 ns -> ~7.5x more random hits on the tile
  planes); (2) straw views: SR-389 vertical only (ours: 0 / +-15 deg stereo, colleague's October numbers).
- SR-389 5.4: combinatorial 9e10 pairs / 15 y -> 36 after the +-200 ps timing coincidence; upstream tagging must reject ~1e2
  (inefficiency ~1e-2); Table 7 = our SST1 categories; a tile plane of the MS7 exit-face size tags essentially all straight
  muons. Next in SR-389: machine learning to combine the subdetectors.
