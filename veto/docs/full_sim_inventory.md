# Full veto simulation: what we have, what is missing (2026-11-07)

Inventory of the physics in the veto simulation (shield end -> SST1), against the user's list of sources.
Status: ✅ have, ⚠️ partial, ❌ missing, ⏸ parked (decision column = user's call, 2026-11-07).

## Source: muons

| Item | Status | What we have | What is missing | Decision |
|---|---|---|---|---|
| Transport of the target muons | ✅ | Nov-4 pool -> cuda-muons (TRY6, field, rock); shell storage, 3,780 seeds (`/disk/gfs_lhcb/sqasim/nov4_pool/shell/`) | No +x cavern wall in the model; source-model check (ISSUES D5) | +x wall: dropped for now |
| New muons made in the shield (pi/K decays in the shield, gamma -> mu mu, tridents) | ⏸ | Nothing (pool = muons at z = -0.0045 m; cuda-muons makes no secondaries) | pi/K decay before interacting ~1e-3 (lambda_int Fe 17 cm vs decay length O(100 m)); gamma -> mu mu and tridents ~(m_e/m_mu)^2 ~ 2e-5 of the e+e- analogues | Parked: rare |
| DIS in the vessel walls | ✅ | fastveto forced interactions, library Geant4 muonNuclear (low nu) + Pythia 6 (high nu); wall 350 um Mylar | Reruns with Mylar on the new pool (D8) | Mylar (PVC would have sufficed) |
| DIS in the veto planes | ✅ | Every design plane a target, technology material | Frames / dead material not targets (C2) | |
| DIS in the cavern walls | ❌ | Nothing | ISSUES D7; FairShip origin breakdown: pi / p at SST1 mostly from cavern concrete | **Do it, from the stored trajectories (plan below)** |
| DIS in helium | ✅ | Forced, products optionally through Geant4 (g4vessel) | 15 % energy grid, no nuclear effects in Pythia (D6) | |
| DIS in air | ✅ | Air within 0.4 m of the vessel / fins + shield-to-vessel gap | C7 / C8 reruns pending | |
| DIS in the last magnet -> pi, K | ⚠️ | Last 50 cm of MS7, parent bent by the field, products through Geant4 | D3 (K_L vs FairShip, major); Pythia-6-only part; parents from the old one-run pickle (the new production stores nothing inside MS7); rest of the shield | |
| EM debris, shield end | ⚠️ | Geant4 g4shield -> g4vessel (Guglielmo), g4slab library, emdebris C++ model | Production not run (300-muon smoke test); only as accidental rate maps, not as hits in time with their muon | |
| EM debris, vessel walls | ⚠️ | g4slab steel config | Mylar config, applied per muon | |
| EM debris, cavern walls (rare) | ❌ | Nothing | Debris at the bounce points | Comes with the cavern-wall study |
| EM debris / delta rays, veto planes | ⚠️ | g4slab straw / scintillator configs | Not applied to the hits | |
| Multiple scattering (planes, walls, helium, air) | ⚠️ | `lib/veto/scattering.py`: Highland per scatterer + single-scattering tails tuned to Geant4 | Off by default in reco (`--multiple-scattering`), helium / air a second switch (`MEDIA`); no frame material | ParticleShaker as a GPU backend: dropped for now |

## Source: neutrinos

| Item | Status | What we have | What is missing | Decision |
|---|---|---|---|---|
| Neutrino flux | ⚠️ | 2018 FairShip flux (`bin/extract_neutrinos.py`, `fastveto.neutrinos`), p > 10 GeV | Not from our target production (tailtransport kills neutrinos); nothing below 10 GeV; nu_tau | **New neutrino sample at the target / shield, as for the muons (plan below)** |
| DIS in vessel walls, veto planes, helium, air | ✅ | GENIE 3.06 library, forced (`bin/nu_dis_apply.py`) | Rerun on the new flux | |
| DIS in the cavern walls | ❌ | Nothing | nu in concrete / rock, incl. CC muons entering the cavern (Guglielmo's nu-induced muons) | With the new sample |
| DIS in the last magnet -> pi, K | ⚠️ | Shield chain with a GENIE iron library | nu_mu only (no nu_e library); 2018 flux, > 10 GeV; last 50 cm only | With the new sample |

## Not in the list but part of the simulation

| Item | Status |
|---|---|
| Combinatorial pairs (two unrelated muons in time) | ✅ `bin/end_to_end.py` |
| V0 made outside, decaying inside (C7) | ✅ (air / fins) |
| Signal loss to accidental hits, EventCalc signal | ✅ |
| K_L decays in the fastveto path | ❌ (stable) |
| Detector response: clusters, cross-talk, fake hits from combining views (C1) | ❌ |
| Spectrometer + straws + GenFit in the main chain | ⚠️ exists, not used by the veto |
| Cosmic muons | ❌ not looked at (likely negligible with timing; one-line estimate) |

## Plan: DIS in the cavern walls from the stored trajectories

1. **Who touches the rock** (DONE 2026-11-07, `bin/rock_hits.py`, 300 seeds,
   `/disk/gfs_lhcb/sqasim/nov4_pool/shell/rock_hits.npz`): muons whose line from z = 30.6 m meets the wall
   (x = -4.56 m) or the floor (y = -3.36 m) before z = 82 m, per spill:

   | | go in | come back into the cavern (reach 82 m in the cavern, bounce at the surface) |
   |---|---|---|
   | wall | 1.48e9 (n_eff 8.0e3) | 6.2e4 (n_eff 1.2e3), median p 3.8 GeV, grazing angle 71 mrad |
   | floor | 8.9e7 (n_eff 1.4e4) | 5.9e4 (n_eff 650), median p 2 GeV, grazing angle 71 mrad |

   Of the wall muons that stay: 71 % are still inside the rock at z = 82 m (running along the wall), 28 % stop.
   Only ~1e-4 come back. So the DIS source is the muons RUNNING INSIDE THE ROCK near its surface (1.6e9 / spill),
   not the bounced ones: products made within about an interaction length of the surface can come out. The library
   must be a half-space at grazing incidence, with the depth along the path (depth = s sin(angle): 1 m after ~14 m
   at 71 mrad).

   CORRECTION to earlier numbers (2026-11-07): "wall bounces mu+ 3.3e8 / spill" (muon_fate / plot_bounces / the
   3D page's "bounced" filter, any finite bounce point K) were 96 % muons that STOP in the rock: a stopped muon's
   82 m state is its stop point and the bounce construction then gives a fake K at the surface. Type 4
   (rock_bounced, > 50 cm off its straight line) is 99 % muons inside the rock at z = 82 m (1.5e7 / spill); only
   1e5 / spill of them are back in the cavern. Muons that really come back into the cavern: ~1.2e5 / spill.
   The SST1-bounced class (type 0) is not affected (it reaches SST1).
2. **Library** (Geant4, `g4slab` + the forcing of `g4dis`): muons on a concrete half-space at (E, grazing angle),
   muonNuclear forced uniformly along the path (as deep as products can still escape) (weight n sigma L), full transport; record
   everything that comes back out of the surface into the cavern (pdg, p, position, direction, time, in the frame of
   the entry point). The same runs without forcing give the cavern-wall EM debris.
3. **Apply** to the stored muons: one library event per rock-touching muon, rotated to its entry point; products
   straight (or through g4vessel) to the veto planes and SST1; rates of particles / hits / vertex candidates, and
   through the veto and selection.
4. **Check** against the FairShip origin breakdown (`/disk/users/gfrise/fairship_sst1_origin`).

## Plan: neutrino sample at the target and shield

Same generator and normalisation as the Nov-4 muon production (tailtransport complete mode, Pythia 8.223, Geant4 11.2,
FairShip target and beam, 4e13 p.o.t. per spill), but recording the neutrinos: every decay with a neutrino of every
species in the cascade (pi, K, K_L, K_S, hyperons, charm, beauty, tau, muon decays), no momentum cut, with their
production point. Neutrinos need no transport: straight lines to the shield iron, cavern rock, walls, helium and
planes, then GENIE forced interactions as now (libraries extended to nu_e / nu_tau and to iron / rock nuclei).
