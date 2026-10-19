# Veto simulation: open problems and status

Started 15 Oct 2026. Update the **Status** line of an item when it changes and add a dated note
under it. Keep finished items (DONE) so we know what was decided and why.

| Status | Meaning |
|---|---|
| OPEN | not started |
| IN PROGRESS | partly done, see the dated notes |
| DONE | finished |
| DEFERRED | decided to do later |
| OK | already correct, nothing to do |

**Priority:** A = must be fixed before optimising, B = changes the results, C = second order.

Numbers are from `run_reco.py` on D6_tile / D6_straw (15 Oct 2026) unless noted.
Document references: SPSC-P-369 (2023 proposal), SPSC-SR-370 (2025 annual report), in the repository root.

## Overview

| Item | Title | Status |
|---|---|---|
| A1 | Background figure of merit is saturated | IN PROGRESS |
| A2 | Bounced muons: the veto barely sees them | IN PROGRESS |
| A3 | Rate limits are reported, not enforced | IN PROGRESS |
| A4 | No objective, two inconsistent cost models | IN PROGRESS |
| A5 | Spill and PoT normalisation | OPEN |
| A6 | No timing detector: track time and pair coincidence | IN PROGRESS |
| B1 | Multiple scattering | OPEN |
| B2 | EM debris rate | OPEN |
| B3 | Neutrino and muon deep inelastic scattering | DEFERRED |
| B4 | Signal model | OPEN |
| B5 | The SBT on the vessel walls is not a detector in the simulation | OPEN |
| B6 | UBT geometry and efficiency differ from SR-370 | OPEN |
| B7 | Pre-selection differs from P-369 Table 3 | OPEN |
| C1 | Detector response simplifications | OPEN |
| C2 | Dead material and dead regions | OPEN |
| C3 | Validation against full simulation | OPEN |
| C4 | Energy loss of bounced muons in the cavern | OK |
| C5 | Weights and statistics | OPEN |
| C6 | Detector technology numbers | DONE |
| C7 | Cross-checks against the P-369 background numbers | OPEN |
| C8 | Cosmic muons | OPEN |

---

## A. Blockers: fix before optimising

### A1. Background figure of merit is saturated

**Status:** IN PROGRESS

The pair background is 2-3 surviving pairs out of 196 736 candidates (about 1e-5 events in
15 years). Every reasonable design gives about 0, so differences between designs are Poisson noise.

Plan: score designs on the single-muon veto inefficiency per muon type (large statistics), build
the pair background by factorisation, rate_pairs × eps_1 × eps_2 (check that this holds on the pairs
we do have), and quote an upper limit when there are zero survivors.

- 16 Oct 2026: factorised estimate in `objective.py` (`goal = "pairs"`). The inefficiency per type
  is now measured only on the muons that can be in a pair (p > 1 GeV, timing acceptance,
  `objective.pairable`): after the veto the pairs are dominated by bounced + selected, and the
  bounced muons in pairs are mostly wall bounces (vetoed at 99 %), not the soft floor bounces
  (51 % missed). Using all bounced muons overestimated the fake rate. Option `eps_weighting =
  "unweighted"` for ranking (see C5). Still to do: check the factorisation against the direct count.

### A2. Bounced muons: the veto barely sees them

**Status:** IN PROGRESS

Only 74-77 % of bounced SST1 muons are vetoed. The search window contains the true crossing for
only 30 % of them: a straight line extrapolated back from the spectrometer cannot follow the
bounce off the cavern wall or floor. This term will probably decide the fin layout.

Plan: study this class on its own (where it crosses the planes, before/after the bounce point),
consider planes downstream of the bounce point or a search aimed at the wall.
Statistics: rock_bounced and veto_only come from ONE simulation run (overfitting risk).
The energy loss in the cavern rock is well modelled in the stored muons (see C4).

- 16 Oct 2026: floor bounces checked on OPT1 (seed 2). A UBT down to the floor (|x| <= 0.5 m,
  y -3.36 .. 1.35 m) vetoes almost none of them (floor-bounce misses -6 %, total 1032 -> 1030 per
  spill): any plane upstream of the bounce point (bounce z 38-74 m) is searched along the straight
  back-extrapolation from SST1, which runs under the floor there, not where the muon really
  crossed. Only planes downstream of the bounce and under the vessel can catch them, or a
  reconstruction that flags tracks whose back-extrapolation leaves through the floor/wall.
  `optimize_fins.py` now searches the UBT bottom (floor .. -1.35 m; seeds start at the floor).
- 16 Oct 2026: wall bounces with OPT1: 76 / 8529 per spill missed (1 %); floor bounces:
  852 / 1687 (51 %). Of the floor bounces only 53 per spill have p > 1 GeV.
- 16 Oct 2026: one small panel under the vessel added (`design.py` "bottom", at most 10 m²,
  between the floor and the vessel bottom; rate sample `bin/make_bottom_rates.py`: 9.6e7 muons
  per spill pass under the vessel, a panel at z ~ 50-70 m sees ~1-2 k per spill, peak 1.4 Hz/cm²).
  A 10 m² panel catches ~100 of the 1687 floor bounces per spill. Optimisation running
  (`results/optimize_bottom`). See also B5: in P-369 the SBT itself covers the vessel bottom.

### A3. Rate limits are reported, not enforced

**Status:** IN PROGRESS

The 4 cm tiles on the z = 30.6 m fins reach 10 MHz per channel (limit 1 MHz; consistent with
SR-370's "up to 10 MHz in the flames", a good sanity check). The straw fin at z = 30.6 m is above
0.5 MHz/cm². A design over its limit still gets a normal score.

Plan: rate limits as hard constraints or penalties. The fin rate maps have 20 × 20 cm bins, which
averages the peak down for smaller cells: use finer maps near the flames.

- 16 Oct 2026: limits are penalties in `objective.py` (`rate_penalty`, at least 1000 per limit
  exceeded, more than all other tiers together); the optimisers use them. Finer maps near the
  flames still to do.

### A4. No objective, two inconsistent cost models

**Status:** IN PROGRESS

`optimize_sbt.py` is a stub. To decide:

- objective: e.g. minimise cost subject to background < X and signal loss < Y;
- one cost model: kCHF per layer-m² (P-369 table 11) and relative cost per channel (straw = 1)
  disagree (fibres are expensive per area, straws cheapest per channel); pixel cost is nan in the
  area model and would break an optimiser;
- free parameters and fixed limits (cavern wall x = -4.56 m, 1 m straw length, nothing below
  y = -3.2 m);
- Monte Carlo noise: fixed seed / same events for every design (common random numbers).

Notes:

- 16 Oct 2026: objective in `objective.py`, lexicographic tiers: rate limits and 8 MCHF budget
  (walls), then false vertices in 15 years, then every missed SST1 muon (`w_mu`), then signal loss,
  then cost. Floor is now y = -3.36 m.
- 16 Oct 2026: common random numbers in `reco.py` (`RecoConfig.common_random`, default on): each
  track's efficiency roll, hit-time smearing and spectrometer smearing come from a stream fixed by
  the track and the plane geometry, and the signal decays from a fixed generator. Before, adding
  one plane re-shuffled every random number and the fake rate jumped by ±25 % between designs
  differing only in a plane that no selected muon crosses: five times the real effect.

### A5. Spill and PoT normalisation

**Status:** OPEN

Accidental hits use flat times inside a 1 s spill (`--spill`). `config.py` has 5e13 PoT per spill
and 6e20 PoT in 15 years. Check that rates match the 5e13 PoT/s convention, and model the real
spill structure (50 Hz ripple, duty factor): accidentals scale with the instantaneous rate, and the
straw signal loss (5.3 % with a ±25 ns window) scales with them.

- 16 Oct 2026, from the documents: 4e13 PoT per spill (P-369, SR-370), 1.2 s flat top every 7.2 s
  (P-369: "1 s spills"); at the start of data taking the spill may be stretched to 2.4 or 4.8 s
  (SR-370). The P-369 combinatorial estimate assumes a flat time structure.

### A6. No timing detector: track time and pair coincidence

**Status:** IN PROGRESS

Decision (Guglielmo, 16 Oct 2026): never assume the 200 ps timing cut of the timing detector.

Two places used it:

- the two-track coincidence of the false-vertex estimate (`config.py` `time_window` = ±200 ps):
  `objective.ObjectiveConfig.coincidence_s` now defaults to None, meaning two muons of the same
  1 s spill: 7567 candidate pairs per spill before the veto, 9.1e10 in 15 years (36 with ±200 ps);
- the track time the veto hits are matched to (`RecoConfig.sigma_t_track` = 0.1 ns,
  "spectrometer / timing detector"). P-369 says the timing detector "also provides time
  information for the straw tracker"; no straw-tracker time resolution is given in the
  documents. Assumed 5 ns (`--sigma-t-track 5`), to be varied (1, 20 ns).

With ±3σ windows and one smeared track time per muon, the Gaussian tail fails every plane at once
(extra inefficiency ~3e-4 for selected muons): use ±5σ (`--n-sigma-t 5`) without the timing
detector. Fixed veto windows (tiles ±1.5 ns) are never narrower than the track time allows now.

## B. Physics missing: changes the results

### B1. Multiple scattering (at least this interaction must be modelled)

**Status:** OPEN (material budget defined 15 Oct 2026, ready to implement)

The stored muons have NO multiple scattering after the shield: the states at z = 30.6 m and
z = 82 m are joined by straight lines and the transport did not include the air / vessel.

Material along a muon's path (now fully defined, so MS can be computed for every muon):

| z [m] | material | thickness | x/X0 |
|---|---|---|---|
| 29.89 - 30.6 | air (shield end, TRY6, to the UBT) | 0.71 m | 0.0023 |
| 30.6 | PVC wall (decay-vessel entrance) | 0.3 mm | 0.0015 |
| 30.6 - 80.6 | helium (decay volume, SR-370 balloon) | 50 m | 0.0088 |
| 80.6 | PVC wall (decay-vessel exit) | 0.3 mm | 0.0015 |
| 80.6 - 82 | air (to SST1) | 1.4 m | 0.0046 |

Plus every veto plane crossed (`technology.x0`: straw 0.12 %, fibre 0.6 %, pixel 0.5 %, tile 2.4 %
X0 per plane). X0 used: air 303.9 m, helium 5671 m, PVC 19.63 cm (PDG).

Total for a straight muon through the vessel: x/X0 = 0.0188 without the UBT, 0.020 with a straw
UBT, 0.043 with a tile UBT. Highland, θ0 = 13.6 MeV / p × sqrt(x/X0) × (1 + 0.038 ln(x/X0)):
1.6 mrad / p[GeV] (straw UBT), 2.5 mrad / p[GeV] (tile UBT). Displacement at SST1 (z = 82 m) from
one scatterer at p = 1 GeV: PVC entrance ~2 cm, straw UBT ~2 cm, tile UBT ~9 cm, PVC exit ~0.1 cm.
Small at high momentum, relevant for the soft muons (1 - few GeV) that dominate the fin rates.

Bounced muons: apply MS only AFTER the bounce point (`muon_types.bounce_points`), i.e. from the
bounce on the cavern wall / floor to z = 82 m in air plus whatever planes / walls it crosses after
it. The path before the bounce and the energy loss in the rock are already in the transport (C4).

Plan:

- for every muon, walk its true path (A -> K -> B) through the list above, sum x/X0 per segment,
  apply Gaussian angle and position kicks per segment (variance proportional to x_i/X0, scaled so
  the total matches Highland for the full x/X0), momentum from the stored states;
- use the scattered path for the own hits on the UBT and the fins, and the scattered state at SST1
  for the spectrometer measurement;
- open: material of the vessel SIDE walls (muons entering through the SBT walls) is not defined yet;
- check that `reco.RecoConfig.ms_slope` / `r_ms` (search window growth with 1/p) still cover the
  true scatter, otherwise the window loses veto efficiency or grows the accidentals.

Notes:

- 16 Oct 2026, from the documents: the P-369 vacuum design had aluminium end-caps of
  0.18 λI / 0.8 X0 (upstream end-cap behind the UBT), far more than the 0.3 mm PVC used here. With
  the SR-370 helium balloon (atmospheric pressure + 20-50 mbar, ~685 m², ~543 m³, tracker in air
  behind a thin helium membrane) the entrance window is thinner but its thickness is not given:
  confirm before implementing. The bottom panel (A2) is not in the scattering model yet
  (`scattering.py` skips it).

### B2. EM debris rate (electrons / photons from the end of the shield)

**Status:** OPEN

Only muons are simulated: no delta rays, bremsstrahlung or pair-production debris leaking from the
shield. These add hits (occupancy, rate limits) and accidental vetoes (signal loss).

Plan (estimate from data we have):

- take the full dataset (`data/Full_dataset_z_3060_8200.pkl`, 506 M rows of ONE run, seed 42):
  every muon that reaches z = 30.6 m, with its momentum and position;
- for each, compute the probability that a muon of that energy produces an electron or photon in
  the last 20 cm of iron of the last shield magnet (z = 29.69-29.89 m) that escapes it: delta rays
  (dN/dT ~ 0.56 MeV/cm / T² in iron, for T << T_max), bremsstrahlung and direct pair production,
  with the escape length set by the electron range / photon conversion length (X0 iron = 1.76 cm,
  so 20 cm = 11.4 X0 is effectively the equilibrium layer);
- turn that into e / γ rate maps on the UBT and the fins (same binning as the muon maps), apply a
  per-technology detection probability (photons convert rarely in thin straws, more in 1 cm tiles)
  and add them to the accidental hits.

Cross-check later against a Geant4 run of a muon beam on 20 cm of iron.

- 16 Oct 2026, from SR-370 §3.3.2: the UBT material itself also produces secondaries: "a muon rate
  of 100 Hz/cm², integrated over the active area of a six-layer detector, would generate an
  additional 3 MHz of particles into the decay volume". Not modelled.

### B3. Neutrino and muon deep inelastic scattering

**Status:** DEFERRED (to be evaluated later)

Not taken into account: DIS in the veto planes (tiles 2.4 % X0 vs straws 0.12 % X0), in the vessel
walls and in the cavern. Until then the optimiser will favour thick fast planes for free; keep the
material reported (x0 per plane) so it can be added as a constraint.

- 16 Oct 2026, numbers from the documents: P-369 §4.2.1 expects ~2e10 muon DIS interactions near
  the fiducial volume in 6e20 PoT; after pre-selection ~6000 (fully) and 2.7e5 (partially
  reconstructed) background events; combined UBT + SBT veto suppression < 8e-7 (upper limit),
  giving < 5e-3 and < 0.2 events. SR-370 §3.3.2: about 100 neutrino interactions per 1e20 PoT in a
  six-layer straw UBT; SR-370 §3.3.3: neutrino interactions on the helium inside the decay volume
  are tagged by the SBT.

### B4. Signal model

**Status:** OPEN

Now: neutral particle decaying uniformly in the vessel volume, two daughters uniform over the SST1
acceptance, p uniform 1-50 GeV. Signal is lost only through accidental hits.

Plan: use an HNL (and dark photon) model from the literature:

- decay position: probability to decay in the vessel P = exp(-L1/λ) - exp(-L2/λ), λ = βγcτ; for
  the long lifetimes near the sensitivity limit (λ >> 50 m) the vertex is close to uniform in z,
  otherwise exponential;
- momentum spectrum and production from charm / beauty meson decays at 400 GeV, and decay channels
  (e.g. HNL -> μπ, eπ, μμν): take them from the FairShip HNL generator or from the references
  below, NOT invented;
- daughter opening angles from the 2-body / 3-body kinematics, so the daughters' spread in the
  SST1 acceptance is realistic.

References:

1. SHiP Collaboration, "Sensitivity of the SHiP experiment to Heavy Neutral Leptons",
   JHEP 04 (2019) 077, arXiv:1811.00930.
2. K. Bondarenko, A. Boyarsky, D. Gorbunov, O. Ruchayskiy, "Phenomenology of GeV-scale Heavy
   Neutral Leptons", JHEP 11 (2018) 032, arXiv:1805.08567 (production and decay widths, branching
   ratios).
3. SHiP Collaboration, "Sensitivity of the SHiP experiment to dark photons decaying to a pair of
   charged particles", Eur. Phys. J. C 81 (2021) 451, arXiv:2011.05115.
4. S. Alekhin et al., "A facility to Search for Hidden Particles at the CERN SPS: the SHiP physics
   case", Rep. Prog. Phys. 79 (2016) 124201, arXiv:1504.04855.
5. SPSC-P-369 (CERN-SPSC-2023-033), BDF/SHiP ECN3 proposal: signal selection, IP and veto
   requirements (summarised in conversation.txt).

(Check volume / page numbers against INSPIRE before citing in a talk.)

### B5. The SBT on the vessel walls is not a detector in the simulation

**Status:** OPEN

P-369 §3.2.3: "The baseline option to cover the top-, the bottom-, and the side-walls of the vacuum
vessel is using a state-of-the-art liquid scintillator (LS-SBT)": 20 cm of LAB + 2 g/l PPO in cells
of the wall structure, each read out by two WOMs (40 SiPMs each), ~1 ns time resolution, ~10 cm
spatial resolution, > 99.3 % efficiency for ≥ 45 MeV deposits (99.9 % expected with coated walls).
SR-370 §3.3.3 (helium balloon): AlMg4.5 cells filled with LS, 80 cm along the beam and up to 150 cm
across, in containers around the decay volume; "instrumented upstream and on the sides" (the bottom
is not named explicitly).

In our simulation the vessel walls only set a flag (`walls` bitmask, "veto_only" type); the veto
is the UBT plus external fins (and now one bottom panel). A muon entering through a wall is not
vetoed by the wall itself, and a floor-bounced muon rising into the vessel would be seen by a
bottom SBT.

Plan: decide with Guglielmo whether the fin system replaces the LS-SBT (then keep as is and say so)
or complements it (then add the wall cells as planes: efficiency, 1 ns, 10 cm cells, rate maps on
the walls from `maps.py`).

### B6. UBT geometry and efficiency differ from SR-370

**Status:** OPEN

SR-370 §3.3.2 (DUBT):

- straws cover the entrance window 1.3 × 2.7 m (the module layout covers 1.3 × 3.0 m); ours is
  1.0 m wide (|x| < 0.5 m), so muons at 0.5 < |x| < 0.65 m are covered only by the z = 30.6 m fins;
- horizontal tensioned straws in ~1 × 1 m modules: two horizontal layers and two at ±15°, each
  module two staggered layers, 5 mm straws, 1 mm spacing; ours are vertical (colleague's choice,
  `technology.py`);
- a scintillating-tile timing layer (5 × 5 × 1 cm³, ~200 ps, up to 10 MHz per tile in the flames,
  smaller tiles near the flames to stay < ~1 MHz per channel) in front of the straws; our UBT is
  straws only;
- the UBT needs lateral coverage up to ±2 m to see all background muons reaching the spectrometer
  (Fig. 20, ref. [38] there);
- tracking efficiency "will, with suitable staggering, exceed ε_track = 99.99 %" for six layers
  (target > 99.9 %); we use 99.9 % per plane.

Impact (16 Oct 2026, OPT1, 100 k selected muons, same events): UBT at 99.9 % -> 46 missed
(4.6e-4), at 99.99 % -> 35 (3.5e-4); fake rate (unweighted, no coincidence) 3.65e-3 -> 2.49e-3 per
spill. The UBT inefficiency is only ~1/4 of the selected-muon misses; the other 3.5e-4 are not
understood yet (candidates: the 0.5 < |x| < 0.65 m gap beside our 1 m UBT, the spatial window,
the time window). Still to try: a 1.3 m wide UBT (needs the UBT width as a design parameter).
- 16 Oct 2026, OPT2 with full statistics (3 M selected muons, `results/validate/bottom_ubt9999.json`):
  UBT 99.9 % -> 559 selected missed, 99.99 % -> 281 (3.2 -> 1.4 Hz weighted); fake rate 2.2e-3 ->
  6.8e-4 per spill (no coincidence). With better fins the UBT is about half of the remaining misses.

### B7. Pre-selection differs from P-369 Table 3

**Status:** OPEN

| Criterion | P-369 Table 3 | `config.py` |
|---|---|---|
| Track momentum | > 1.0 GeV/c | > 1 GeV/c |
| Track pair DOCA | < 1 cm | < 2 cm |
| Vertex position | > 5 cm from the inner wall; > 100 cm from the entrance (partially reconstructed) | inside the vessel |
| IP w.r.t. target | < 10 cm (fully), < 250 cm (partially reconstructed) | < 2.5 m |
| Track quality | ndof > 25 and χ²/ndof < 5 (§4.2.1) | none |
| Time coincidence | 340 ps window (§4.2.1) | none (A6) |

Our candidate selection is looser: conservative for the background, but the numbers are not
directly comparable to P-369 (see C7).

## C. Detector and bookkeeping: second order

### C1. Detector response simplifications

**Status:** OPEN

- sigma_x is not used (the 5 cm search window r0 is much larger);
- no ghost hits from combining views when several muons share a time window;
- straw time = uniform smearing in the 50 ns drift window; with a tile t0 the drift radius could be
  used (better time and position);
- no cluster size, no tile cross-talk, plane efficiency independent per plane.

### C2. Dead material and dead regions

**Status:** OPEN

No frames, manifolds, overlaps between 1 m straw rows, module gaps, SiPMs or cables. For the UBT
this matters most: the tensioned-straw frame sits in the acceptance (SR-370 open question; SR-370
also names the straw manifolds, and a fallback to 3 m vertical straws if the frame material is a
problem).

### C3. Validation against full simulation

**Status:** OPEN

Run one baseline design through FairShip and compare rates and veto inefficiency before trusting
an optimum.

### C4. Energy loss of bounced muons in the cavern

**Status:** OK

Well modelled in the stored muons (from the transport); nothing to do.

### C5. Weights and statistics

**Status:** OPEN

The repeated-evt_n weight correction is not applied in the design-optimisation dataset; veto_only
and rock_bounced are from one run. Decide whether to apply the correction and how many extra runs
are needed for the rare classes.

- 16 Oct 2026: within each SST1 type the top 2 % of rows (weight ~5.2, about 100 times the median)
  carry 57 % of the weight (selected muons). The 45 k selected muons of the 200 k single sample are
  only 2473 effective muons, so a weighted inefficiency of ~1e-3 rests on ~2.5 effective missed
  muons. Below the tail the miss rate does not depend on the weight, so the optimiser ranks on the
  unweighted inefficiency (`--eps unweighted`); final numbers with weights on 3 M rows per type
  (`bin/validate_design.py`). Probably the same issue as the repeated-evt_n correction: check.

### C6. Detector technology numbers

**Status:** DONE (15 Oct 2026, commit f6d13ec)

Straw / fibre / pixel / tile numbers updated with the colleague's comments (`technology.py`).
Still to confirm: fibre timing (0.5 ns kept, can reach ~100 ps: ask Ettore); fibre plane efficiency
99.9 % ("basically 100 %"), per-SiPM efficiency not modelled; tile channel count on a 1 × 2.7 m
plane depends on the tile combination (mixed sizes not modelled).

### C7. Cross-checks against the P-369 background numbers

**Status:** OPEN

P-369 §4.2.1, muon combinatorial background (6e20 PoT, 4e13 PoT per spill):

| Quantity | P-369 | ours (5e13 PoT per 1 s spill) |
|---|---|---|
| muons in the spectrometer, p > 1 GeV, track quality | ~12 kHz | ~9 kHz selected + bounced with p > 1 GeV (~7 kHz at 4e13) |
| pairs in 1 s spills | 1.08e15 in 6e20 PoT | |
| after pre-selection | × 3.6e-3, i.e. ~2.6e5 per spill | 7567 candidate pairs per spill (opposite charge) |
| time coincidence | × 3.4e-10 (340 ps) | none (A6) |
| UBT + SBT veto | × 9.8e-7 | design dependent |
| result | 1.3e-3 events | |

Our pair count before the veto is ~35 times lower (opposite-charge pairing explains part of it)
and our muon rate ~1.7 times lower. Understand both before quoting absolute fake rates.

### C8. Cosmic muons

**Status:** OPEN

P-369 §3.2: the background taggers must also detect "particles entering the volume from outside,
including cosmic particles". No cosmic sample in the simulation.
