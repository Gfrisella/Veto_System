# D19 — the SBT as a material (16 Nov 2026)

ISSUES D19 ("SBT cell model: missing ingredients"). Branch `veto-nov4-source`. Everything new is behind flags that are
OFF by default: no existing result changes.

## 1. Geometry (FairShip 26.10, `geometry/veto_config_helium.yaml`, `veto/veto.cxx`, `geometry/media.geo`)

Going out from the helium (balloon 350 um Mylar = the existing `wall` target):

| layer | thickness | material | g/cm^2 (normal) | role |
|---|---|---|---|---|
| inner support wall (`VetoInnerWall`) | 0.8 cm | Aluminium (AlMg4.5 in the reports; FairShip "Aluminium", 2.7 g/cm^3) | 2.16 | inert, faces the helium |
| liquid scintillator | 20 cm | LS: H 12.16 %, C 87.84 % by mass, 0.86 g/cm^3 | 17.2 | active |
| outer support wall (`VetoOuterWall`) | 0.8 cm | Aluminium | 2.16 | inert |
| ribs | 1.5 cm thick, 20 cm deep | Aluminium | ~0.7-2 (smeared) | inert, inside the LS layer |

Ribs: a transverse rib at every cell edge in z (82 cm cells, 80 cm the first), 3 longitudinal ribs on each +-x face
(4 cells across), 2 on each +-y face (3 cells across), 4 corner ribs (neglected). Smeared over the LS layer with the
fraction `sbt_material.rib_fraction(face, z)` = 1.5/82 + n_long x 1.5 cm / (2 h(z)) (3-5 %).
Offsets are added to the vessel half widths in x and y (FairShip wx + 2 thickness, as `sbt.T_SHELL`).
Total ~23 g/cm^2 per unit wall = ~470 x the Mylar wall (0.049 g/cm^2); x/X0 = 0.18 (Al) + 0.39 (LS) = 0.57 at normal
incidence. Mg (4.5 %) neglected; nothing else (cables, WOMs, support frame) is included.

The muon transport (MuonsAndMatter cuda, iron + concrete histograms only) has NO vessel / SBT material: the stored
paths A -> (K) -> B are straight through the shell region, so adding the SBT as a target / scatterer double counts
nothing (but the paths ignore the energy loss and scattering the real shell would cause, see section 6).

## 2. Code (all new physics default off)

| file | what |
|---|---|
| `lib/veto/sbt_material.py` (new) | layers, materials registration (`register()`: fastveto materials `aluminium`, `ls`; GENIE compositions; codes 9101-9104 in `interactions.CODE_NAMES`), `layer_pieces` (shell layer chords), `crossings`, forced-interaction sampler `run` (as `fastveto.dis.run`), final-state enrichment, parametrised LS response `deposits` / `fired` |
| `bin/sbt_interactions.py` (new) | muon (Nov-4 3220-spill store) / neutrino (2026 flux, 6 flavours) interactions in the SBT layers -> int_/part_ npz (then `evstore_build.py apply` + `evstore_redecay.py --natural --n 10`) |
| `bin/sbt_eval.py` (new) | selection + tagging variants per store / target, both IP cuts, signal loss of the new rules |
| `bin/dis_library.py` | materials `aluminium` (G4_Al) and `ls` (G4_POLYETHYLENE as LAB stand-in) in `G4_MATERIALS`; `--g4-npy-dir` (merge worker tables made in the Geant4 container, which cannot read /disk/gfs_lhcb) |
| `lib/veto/sbt.py` | DEFAULT flags `v0_line` (+ `v0_line_r0/theta/early_ns`), `selftag_hits`, `material_scattering` (all False); `v0_line_tag()`; `tag(..., v0=None)` pools `v0["sbt_hits"]` (selftag_hits) and ORs the V0-line decision. The track-pointing window block is untouched. |
| `lib/veto/reco.py` | one line: `sbt.tag(..., v0=v0)` (the vertex candidate was already passed to `run_events`) |
| `lib/veto/scattering.py` | with sbt `material_scattering`: inner Al / LS / outer Al pieces of every track as thin scatterers |

Flags live in the design's `"sbt"` block, e.g. `"sbt": {..., "v0_line": true}`. Test variants (not reference designs):
`results_rerun/sbt_d19/R2_sbt_ubt_{msc,v0line}.json`.

### Interaction libraries
- Muons: `data/interactions/dis/mu_library_sbt.npz` = `mu_library.npz` (-> `mu_library_iron.npz`: helium ... iron) +
  Geant4 11.2.2 FTFP_BERT muonNuclear tables for G4_Al and G4_POLYETHYLENE (20 000 events x 50 energies each; 1 min in
  the cudabear container). Pythia 6 part (W > 2 GeV) shared, split crossover as for every target. LS per-nucleon
  cross section and final states from CH2 (H 14.4 % by mass vs LAB 12.2 %): a stand-in, labelled.
  sigma per nucleon at 20 / 300 GeV (Geant4 part): Al 1.6e-3 / 3.5e-3 1/m-equivalent (see the library log); LS ~ CH2.
- Neutrinos: GENIE v2 library (`nu_library_v2.npz`) already has Al27, C12, H1; LS = C12 + H1 by mass fraction.

### Sampling and enrichment
Per layer, the crossing is drawn with probability w P (P = nucleons/m^2 x sigma(E)), weight S/M (fastveto.dis.run).
Only final states that can give a vertex in the helium are drawn: a strange hadron (K_L, K_S, K+-, Lambda, Sigma, Xi,
Omega) or any product starting >= 0.1 mm from the vertex (decay products of the generator-decayed K_S / Lambda / ...);
weight x their fraction at that energy. Every other final state has all its particles at the vertex in the wall,
outside the helium, and fails "vertex in the vessel" (the candidate vertex is computed from the true lines). Checked
with an unenriched sample (section 4).

### Own deposits (parametrised response; ASSUMPTIONS)
`sbt_material.deposits`: straight lines of every particle of the event tree (up to its end / decay point) and of the
parent muon (from its 30.6 m state or bounce point to the vertex):
- charged: 1.5 MeV/cm / beta^2 (cap x20) in the LS, 4.36 MeV/cm / beta^2 in Al, stops when T is spent; no Birks.
- photons: conversion length 9/7 X0 (Al 11.4 cm, LS 66.4 cm); after conversion 2 x 1.5 MeV/cm along the remaining
  LS path, at most E_gamma; Compton and showering neglected (0.4 X0 of LS).
- neutral hadrons (n, K_L, Lambda, ...): one interaction with lambda = 0.80 m in LS (`NEUTRAL_LAMBDA`), 0.397 m in Al;
  in the LS it deposits 0.3 x T (`NEUTRAL_VIS`: quenched recoil protons, break-up), in Al nothing.
- not modelled: nuclear fragments (ions not kept by the libraries), delta rays, light yield vs position (WOM ratio).
Cells fire above 30 MeV with 99.3 % (the design's SBT config).

### Tagging rules evaluated (`bin/sbt_eval.py`)
- `std`: the design as it is (level b: every reconstructed track; UBT planes, V0 rule (R3), SBT track pointing incl.
  the shell window of b87da94, strips (R3)). Own SBT hits of the event's tracks, incl. the parent muon's path.
- `v0line` (flag `v0_line`): + the vertex candidate traced back as a neutral along the pair momentum to where it leaves
  the vessel through a side wall; a hit within r = 5 cm + 25 mrad x distance (along the wall r / sin(alpha)), in time
  with a light-speed back-extrapolation (window +- 3 ns, opened 3 ns early: a V0 is slower than light). As the R3 V0
  rule of the planes, extended to the SBT cells.
- `v0line_self` (flags `v0_line` + `selftag_hits`): + the event's own LS deposits (above) as hits.
- `own_any`: bound, any own-deposit cell fired (no association, no time).
- `zone` (item 2): an own fired cell with ring centre > z_vertex - 1 m, in time with light from the vertex (+- 4 ns);
  its signal cost is the random-hit probability in those cells (analytic from the cell rates).

## 3. What ran (16 Nov 2026, farm-gpu3 locally: the farm queue was full; Slurm only for run_reco)

| step | command / output | time, memory |
|---|---|---|
| Al / LS muon tables | `--g4-worker aluminium / ls` in the cudabear container, merged with `--only-g4 aluminium ls --g4-npy-dir` -> `data/interactions/dis/mu_library_sbt.npz` | 1 min each; merge 12 s |
| muon interactions, enriched | `results_rerun/batch/sbt_d19_inter.cmd` lines 0-3: inner Al 3e6, LS 3e6, rib 2e6, outer Al 2e6 draws on the 3220-spill store (7.7e8 rows), + `evstore_build apply` + `evstore_redecay --natural --n 10` -> `results_rerun/sbt_d19/store_mu_*[_kl10]` | 12-28 min, 40-62 GB each |
| neutrino interactions, enriched | lines 5-6: 5e5 draws per flavour per layer -> `store_nu_al`, `store_nu_lsrib` (+ `_kl10`) | 12-19 min, 36-38 GB |
| unenriched checks | `sbt_d19_noenrich.cmd` + inner Al 1e6 (`store_mu_*_noenrich`, `store_nu_noenrich`, no K_L copies) | 5-7 min |
| evaluation | `bin/sbt_eval.py` per store (`results_rerun/sbt_d19/per_store/`, `eval_nu_R{2,3}_*`, `eval_*_unflagged_R2_*`); kl10 stores ~40 min / 60 GB each | |
| summary | `python3 bin/sbt_summary.py --dir results_rerun/sbt_d19 --out summary_R2.json` | |
| run_reco variants | Slurm 2091344 (6 tasks, 1-3 min each): `results_rerun/reco/d19_R2_sbt_ubt{,_msc,_v0line}_{full,partial}.json` | |

Interactions per 15 y in the SBT material (unenriched totals): muons inner Al 9.5e8, LS 2.3e10, ribs 2.0e9, outer Al 6.1e9
(Mylar wall: 2.2e7); neutrinos inner Al 2.4e6, LS 1.7e7, ribs 1.7e6, outer Al 2.1e6 (Mylar: 5.8e4). The LS dominates
because the Nov-4 muons run along the side walls: 81 % of the pair-input SST1 muons cross the shell, median 8 m of LS
path (A at |x| ~ 0.55 m at 30.6 m, i.e. in front of the shell, B at |x| ~ 1.65 m).

Enrichment check: the unenriched inner-Al sample has 45 selected events (partial IP) with no strange / displaced final
state: a prompt track plus the muon of a pi -> mu nu decay in flight (made by `evstore.propagate`, not by the
generator), nearly collinear, DOCA point just inside the helium. This category is NOT in the enriched stores; it is
added from the unenriched samples (`--only-unflagged`). With IP < 10 cm it is 1.9e3 / 15 y (2 MC events) before
tagging and 0 after R2.

## 4. Results, R2, per 15 y (`results_rerun/sbt_d19/summary_R2.json`; MC events in brackets)

Columns: selected (before tagging); std = R2 as it is (level b, shell_window on); + the V0 line (`v0_line`); + own LS
deposits as hits (`v0_line` + `selftag_hits`); bound = no own fired cell at all (any rule using the event's own
deposits); zone = item-2 rule (own fired cell downstream of z_vertex - 1 m, in time with the vertex).

IP < 10 cm (full reconstruction):

| source | layer | selected | R2 as is | + V0 line | + V0 line + own deposits | bound | zone |
|---|---|---|---|---|---|---|---|
| mu | inner Al (inert) | 6.1e3 +- 1.4e3 (488) | 32 +- 31 (4) | 0.38 +- 0.27 (2) | 0 (0) | 0 | 0.38 |
| mu | LS (active) | 4.7e3 +- 1.3e3 (382) | 468 +- 60 (60) | 251 +- 47 (29) | 217 +- 43 (25) | 0 | 468 |
| mu | ribs (inert) | 233 +- 87 (224) | 30 +- 6 (27) | 22 +- 5 (17) | 17 +- 5 (14) | 0 | 30 |
| mu | outer Al (inert) | 774 +- 290 (207) | 253 +- 27 (90) | 128 +- 21 (37) | 92 +- 18 (25) | 0 | 253 |
| mu | **sum** | **1.2e4 +- 0.2e4** | **783 +- 75** | **402 +- 52** | **326 +- 47** | 0 | 751 |
| nu | inner Al | 0.26 +- 0.10 (132) | 0.086 +- 0.016 | 0.071 | 0.048 | 0 | 0.086 |
| nu | LS | 0.47 +- 0.07 (74) | 0.28 +- 0.04 | 0.18 | 0.11 | 0 | 0.28 |
| nu | ribs | 0.058 +- 0.009 | 0.030 | 0.027 | 0.021 | 0 | 0.030 |
| nu | outer Al | 0.026 +- 0.006 | 0.015 | 0.010 | 0.006 | 0 | 0.015 |
| nu | **sum** | **0.82 +- 0.13** | **0.41 +- 0.05** | **0.29 +- 0.04** | **0.18 +- 0.03** | 0 | 0.41 |
| nu He (item 2) | helium | 3.61 +- 0.19 (376) | 3.59 +- 0.19 | 3.60 | 3.58 | 0.94 +- 0.10 | **1.02 +- 0.10** |

IP < 2.5 m (partial reconstruction):

| source | layer | selected | R2 as is | + V0 line | + V0 line + own deposits | bound | zone |
|---|---|---|---|---|---|---|---|
| mu | inner Al | 1.17e5 +- 0.07e5 | 3.2e3 +- 1.3e3 | 79 +- 15 | 45 +- 4 | 0 | 168 |
| mu | LS | 1.23e6 +- 0.07e6 | 1.53e5 +- 0.04e5 | 6.6e4 +- 0.3e4 | 5.1e4 +- 0.2e4 | 0 | 1.52e5 |
| mu | ribs | 1.06e5 +- 0.05e5 | 1.52e4 +- 0.04e4 | 6.3e3 +- 0.2e3 | 4.9e3 +- 0.2e3 | 0 | 1.48e4 |
| mu | outer Al | 3.3e5 +- 0.1e5 | 1.04e5 +- 0.02e5 | 4.7e4 +- 0.1e4 | 2.9e4 +- 0.05e4 | 59 +- 15 | 1.04e5 |
| nu | inner Al | 452 +- 18 | 156 +- 8 | 90 +- 5 | 23 +- 3 | 2.6 +- 2.4 | 47 |
| nu | LS | 704 +- 67 | 181 +- 8 | 105 +- 6 | 50 +- 4 | 0 | 158 |
| nu | ribs | 69 +- 4 | 19 +- 1 | 12 +- 1 | 5.2 +- 0.5 | 0 | 17 |
| nu | outer Al | 32 +- 2 | 17 +- 1 | 12 +- 1 | 4.7 +- 0.5 | 0.07 | 17 |
| nu He (item 2) | helium | 419 +- 2 | 413 +- 2 | 412 | 409 | 21.5 +- 0.5 | **39 +- 0.6** |

What survives: almost only K_L made in the LS / outer Al / ribs by muons running inside or just outside the shell (the
`_kl10` stores carry 100 % of the IP < 10 cm survivors). K_S / Lambda / decay-in-flight pairs (inner Al especially) are
already tagged by R2 as it is (the parent muon crosses the LS of the vertex cell, and the vertex tracks point there).
Every surviving event HAS fired cells of its own (bound = 0): the information is in the SBT, the association rules do
not use it: the V0 line misses because K_L -> pi l nu / pi pi pi0 visible momentum does not point back within
5 cm + 25 mrad x distance, and slow K_L (p < 1 GeV) arrive > 6 ns late.

R3 (neutrino stores only; the R3 muon evaluation was stopped to free the machine): IP < 10 cm SBT-material neutrinos
0.40 after R3 as is (R2 0.41), 0.17 with V0 line + own deposits; nu He 3.54 (R2 3.59). The R3 V0 rule and strips do not
act on SBT-material vertices (expected: no upstream plane on the K_L line).

### Signal cost of the new rules (B4 EventCalc set, 40 000 per benchmark, equal-weight mean; `eval_nu_R2_*` "signal")

| rule | IP < 10 cm | IP < 2.5 m |
|---|---|---|
| V0 line alone, random SBT hits (design rate) | 0.08 % | 0.11 % |
| V0 line alone, random rate x4 (Guglielmo's back-scatter factor) | 0.24 % | 0.45 % |
| zone rule (item 2), accidental | 0.58 % | 0.60 % |
| zone rule, rate x4 | 2.3 % | 2.4 % |
| run_reco R2 with EM debris: as is -> with `v0_line` | 2.58 -> 2.74 % | 2.53 -> 2.65 % |

(The run_reco difference includes random-stream noise: the V0-line window draws random hits over a wider window.)
run_reco pairs (A1 direct count) after R2: 2.5e-6 -> 7e-8 with `v0_line` (full), 1.1e-3 -> 1.4e-5 (partial): a few
heavy pairs, not significant.

### Scattering in the shell (item 3, flag `material_scattering`; run_reco R2 full, Slurm 2091344)
Single SST1 muons tagged: 90.6 % -> 84.5 % (p > 1 GeV with timing: 99.75 -> 95.5 %; bounced 87.5 -> 83.7 %); partial
91.5 -> 85.3 %. Pairs after R2: 2.5e-6 -> 1.3e-6 (full), 1.1e-3 -> 6.3e-4 (partial), within the few-pair statistics.
Signal: unchanged within noise (daughters do not cross the shell). The shell is 0.57 X0 at normal incidence but
grazing muons cross metres of it (median x/X0 27 for the pair-input muons that cross it), so the r_ms window of the
SBT pointing is too small for them. Not applied to the muon transport itself (energy loss, range-out).

## 5. Item 2: neutral products of helium interactions
nu DIS in helium (2026 flux, R2): own deposits of all products (photons from pi0, neutrons, wide-angle charged; lines
from the vertex) fire >= 1 cell in 74 % of the selected events (IP < 10 cm: 3.59 -> 0.94 bound). The zone rule (fired
cell with ring centre > z_vtx - 1 m, in time with light from the vertex, +- 4 ns) removes 72 % (3.59 -> 1.02 / 15 y)
for an accidental signal loss of 0.58 % at the design random rate (2.3 % at x4). IP < 2.5 m: 413 -> 39. Signal events
have no own SBT deposits in this model (their daughters reach SST1); EventCalc signals with extra neutrals (e.g. pi0 in
the decay) would lose more: not checked. This rests entirely on the parametrised response (section 2): NEUTRAL_VIS,
NEUTRAL_LAMBDA and the photon treatment decide it; a Geant4 check of the cell response to these products is needed
before using it.

## 6. Caveats and what is left
1. Muon paths: the stored A -> (K) -> B lines cross the shell without energy loss or scattering (the cuda transport has
   no vessel / SBT material). The 2.3e10 LS interactions come from muons travelling metres inside the 20 cm shell; the
   real ones would range out / scatter, and the shell might deflect some. A muon transport with the SBT material (or a
   Geant4 slab study, Shah Rukh's C6 / C7) is needed before trusting the absolute muon numbers.
2. LS library: CH2 (G4_POLYETHYLENE) instead of LAB for the Geant4 muonNuclear part; Mg of AlMg4.5 neglected; ribs smeared.
3. Response: parametrised (no Birks, no ions, no light collection / WOM position); thresholds 30 MeV everywhere.
4. Statistics: the non-K_L part of the LS / outer Al / rib muon samples with IP < 10 cm rests on 1-5 MC events (all
   tagged by R2); the decay-in-flight category on 2 events (full) / 45 (partial).
5. Not done: R3 muon evaluation (stopped for CPU; expected = R2), Geant4 transport of helium products with the LS
   volume (item 2 is parametrised), EM debris / back-scatter in the cells (x4 scale used for costs), WOM ratio, the
   7.6 % unassociated single SST1 muons (items 4-5, documentation only).
6. Decisions for Guglielmo: (a) whether the SBT-material muon K_L (~800 / 15 y after R2 at IP < 10 cm, 1e5 at
   IP < 2.5 m) is believed (caveat 1) and enters the background budget; (b) switch on `v0_line` (cost 0.1-0.5 %
   signal, halves that term) and / or `selftag_hits`; (c) a rule using the event's own deposits without pointing (the
   bound is 0: every surviving SBT-material event and 74 % of nu-He have own fired cells), e.g. the zone rule, at
   0.6-2.4 % signal; (d) `material_scattering` on by default (it lowers the single-muon tagging 90.6 -> 84.5 %
   unless the SBT window gets an r_ms term for the shell).

## 7. 17 Nov 2026: decisions (a)-(d) implemented, flags on in R2 / R3 (commit 19b2a0b)

Guglielmo: (a) the SBT-material term enters the background budget, (b) `v0_line` + `selftag_hits` on, (c) zone rule on,
(d) `material_scattering` on + a shell-scattering term in the SBT pointing window. Code defaults stay off; the flags are in
the `"sbt"` block of `designs/R2_sbt_ubt.json` and `designs/R3_sbt_ubt_strips.json` (`shell_ms_k: 1.0`).

- (a) `bin/sbt_prepare.py` -> `results_rerun/interactions/prepared_sbt_nov4full[_partial].npz` (kind `sbt` in
  `evolve/evaluate_designs.py`, summed by `veto.objective` with `w_src_muon` = 1; the events exist only where the design has
  SBT cells, R1 gets none). `bin/evstore_eval.py --muons` evaluates SBT-material stores and prints the class total.
- (c) zone rule in `sbt.tag` (cfg `zone_rule`, `zone_up` 1 m, `zone_extra_ns` 1 ns); the own LS deposits of an interaction
  reach it through `interactions.evaluate` / `interactions.veto` (`v0["sbt_hits"]`).
- (d) shell term: `shell_ms_k` x Highland(x/X0 of the backward line's shell path) x path / sqrt(3), in quadrature with r;
  the b87da94 window is otherwise unchanged. `scattering.py`: the shell kick now sits at the END of each shell piece along
  the motion (was its middle). The SBT own hits stay on the stored straight path, so the measured line has to meet it
  where the muon leaves the shell, as a real track (tangent at its exit) does. With the kick at the middle of a grazing
  piece (median 8 m of LS) the line missed the exit by theta0 x 4 m (3 GeV, x/X0 21: 23 mrad x 4 m = 9 cm) and in 80 % of
  the misses met no side wall at all.

### Choice of k (`bin/sbt_shell_ms_check.py`, `results_rerun/sbt_d19/shell_ms_check_R2.json`)
Single SST1 muons with an own fired cell, random hits off, efficiency 1 (window coverage), weighted as run_reco. "Physical":
the energy loss in the shell along the stored path (Al 0.436, LS 0.150 GeV/m) is below the kinetic energy at SST1.

| | no shell scattering | k 0 | k 1 | k 2-10 |
|---|---|---|---|---|
| physical muons (63 % of the weight) | 94.46 % | 93.04 % | 93.26 % | 93.26 % |
| range-out muons (37 %) | 77.0 % | 58.2 % | 58.8 % | 59.8-64.2 % |

k saturates at 1 on the physical muons -> `shell_ms_k = 1`. With the kick at the middle (first version): weighted coverage
80.5 % at k 0, 82.6 % at k 10 (`shell_ms_check_R2_kickmid.json`).

### Results, R2 (`results_rerun/sbt_d19/flags17/summary_R2.json`; per 15 y; flags off -> R2 as is)
Single SST1 muons tagged (run_reco, 19b2a0b, tile-plane em_maps): full 92.1 -> 85.1 %, partial 93.9 -> 85.4 %; physical
muons 96.5 -> 93.3 %, range-out muons 84.3 -> 66.1 % (`flags17/phys_single.json`). The range-out muons (p 0.2-0.7 GeV, shell
x/X0 5-25, mostly "SST1 other") cannot exist with the paths they are stored with (caveat 1 of section 6).

| source | layer | IP < 10 cm: off -> as is | IP < 2.5 m: off -> as is |
|---|---|---|---|
| mu | inner Al | 32 -> 0 | 3.2e3 -> 12.5 |
| mu | LS | 468 -> 43 +- 19 | 1.53e5 -> 1.84e4 |
| mu | ribs | 30 -> 4.3 +- 2.1 | 1.52e4 -> 1.84e3 |
| mu | outer Al | 253 -> 66 +- 16 | 1.04e5 -> 1.87e4 |
| mu | **sum** | **783 -> 113 +- 25** | **2.75e5 -> 3.9e4** |
| nu | inner Al / LS / ribs / outer Al | 0.074 / 0.25 / 0.034 / 0.015 -> 0.026 / 0.048 / 0.011 / 0.006 | 154 / 177 / 20 / 16 -> 9.4 / 30 / 3.4 / 3.6 |
| nu | **sum** | **0.37 -> 0.091** | **367 -> 47** |
| nu He (item 2) | helium | 3.55 -> 0.96 | 410 -> 32 |

Signal cost (B4 set, no EM debris): full 0.87 -> 1.40 %, partial 0.66 -> 1.21 % (zone 0.58 %, V0 line 0.08 % accidental).
run_reco with EM debris (19b2a0b, old em_maps): full 3.24 -> 4.08 %, partial 2.87 -> 3.52 %; signal loss numbers after the
straw-plane em_maps fix are the main session's (`reco_R*_fix1622.json`).

### Results, R3 (`flags17/summary_R3.json`)
Muon stores evaluated on 1622cf7 (R3 V0 rule p_max 20 GeV, strips +-1.5 ns), neutrino stores on 19b2a0b. Per 15 y, flags
off -> R3 as is: IP < 10 cm mu 47 -> 0 (0 MC events: the R3 V0 rule already tags most SBT-material K_L), nu 0.30 -> 0.078,
nu-He 3.49 -> 0.96; IP < 2.5 m mu 2.76e4 -> 4.3e3, nu 334 -> 38, nu-He 373 -> 24. Signal (B4, no EM debris, 19b2a0b R3):
full 5.82 -> 6.16 %, partial 7.73 -> 8.49 %.

### Objective cache
`prepared_sbt_nov4full.npz`: 1 633 selected events, 1.29e4 / 15 y before tagging (partial: 369 135 events, 1.79e6).
Closure through `interactions.veto` (the objective's path): R1 0, R2 125 +- 26 / 15 y (sbt_eval: 113 +- 25 muons + 0.09 nu).

