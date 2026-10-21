<!-- From fast-veto-sim docs/downstream_sim.md (ported 2026-10-21; paths updated to Veto_System). -->

# Downstream simulation: muon interactions and multiple scattering after the shield

Input: the cuda-muons output as the design-optimisation dataset of Veto_System (column store: states A at
z = 30.6 m and B at z = 82 m or where the muon stopped, bounce point K, weight per spill, evt_n, type, walls).
Simulation code of Veto_System is copied unchanged in `vetosim/` (commit a8dcc3c); additions are in `lib/fastveto/`.

## Muon-nucleus interactions

### Library (`bin/dis_library.py` -> `data/interactions/dis/mu_library.npz`, 1.9 GB, built in 2 min on farm-gpu5)

Energy grid 0.5 .. 500 GeV (50 points, 15 % steps), all events stored with nu and W:

* **Geant4** 11.2.2 muonNuclear (G4KokoulinMuonNuclearXS + G4MuonVDNuclearModel, FTFP_BERT): one forced
  interaction per event in a block of the material (`cpp/g4dis.cc`, 0.03 ms / event), 20k events per energy
  and material: helium, air, PVC, Mylar (straw), polystyrene (fibre), silicon (pixel), PVT (tile). Nuclear
  break-up included, ions dropped. Short-lived products (K_S, Lambda, Sigma, Xi, Omega, pi0, eta) decayed by
  `lib/fastveto/decay.py` (main channels, phase space, decay point from c tau): Geant4 records them undecayed.
* **Pythia 6.4.28**, beam `gamma/mu-` on p+ and n0 (real + virtual photons: VMD, direct, anomalous, DIS),
  W > 2 GeV, PARP(2) = 2 GeV, from E = 2.2 GeV, 20k events per energy and nucleon (`cpp/tailgen/pythia6_dis.f`,
  ~1.5 ms / event). Pythia sometimes loops forever in PYEVNT (7 hangs in 1.6 M events): chunks run with a time
  limit, complete events kept, the rest regenerated with another seed.

Which generator covers which part is chosen at load time, `fastveto.dis.MuonLibrary(path, split=...)`:

* `crossover` (default): Geant4 for nu < nu_x(E), Pythia above. nu_x(E) = where Pythia's dsigma/dnu first reaches
  Geant4's above nu = 2 GeV: 7 GeV at E = 10, 11 at 30, 15 at 100, 20 at 300 GeV. dsigma/dnu is continuous.
  Reason: just above W = 2 GeV Pythia is ~10 x below Geant4 (flat ~0.1 ub/GeV where the photon flux gives ~1/nu),
  while at large nu it is 1.3 - 1.7 x above Geant4 (Kokoulin's formula has no DIS growth with Q^2).
* `w2`: Geant4 for W < 2 GeV, Pythia for W > 2 GeV (leaves the hole above W = 2: total 0.83 - 0.9 x Geant4).
* `geant4`: Geant4 only (what a full Geant4 simulation does).

Only mu- is generated: one-photon exchange gives the same hadronic final state for mu+; the scattered muon gets the
real muon's sign. Cross sections per nucleon, helium (microbarn):

| E [GeV] | Geant4 total | crossover total | of which Pythia | sigma(>= 1 charged hadron p > 1 GeV), Geant4 / crossover |
|---|---|---|---|---|
| 10 | 8.7 | 8.7 | 1 % | 1.40 / 1.42 |
| 30 | 13.3 | 13.6 | 5 % | 3.08 / 3.40 |
| 100 | 19.1 | 19.8 | 11 % | 5.47 / 6.50 |
| 300 | 24.9 | 26.2 | 15 % | 8.36 / 10.15 |

`lib/fastveto/kokoulin.py` reproduces Geant4's cross section tables to 0.1 %.

### Targets per design (`lib/fastveto/targets.py`)

Along A -> B (A -> K -> B for bounced muons): helium (vessel chord), cavern air (rest of the path), PVC (window,
lateral walls, exit wall: exact chords through the 0.3 mm shell, fastveto.vessel.piece_intervals) and every plane of the design with the material of its
technology (`lib/fastveto/materials.py`: areal density from vetosim's x/X0 of the plane and the X0 of the material).
Muon energy at the target: |p| linear in z between A and B.

| target | material | x/X0 | thickness | nucleons / cm^2 |
|---|---|---|---|---|
| straw plane (6 layers) | Mylar | 0.0012 | 0.34 mm | 2.9e22 |
| fibre plane (2 mats) | polystyrene | 0.006 | 2.5 mm | 1.6e23 |
| pixel plane | silicon | 0.005 | 0.47 mm | 6.6e22 |
| tile plane | PVT | 0.024 | 10.2 mm | 6.3e23 |
| vessel wall | PVC | 0.0015 | 0.3 mm | 2.5e22 |
| helium, 50 m | He | 0.0088 | | 5.0e23 |

### Forced interactions (`fastveto.dis.run`, driver `bin/muon_dis_apply.py --design D6_straw`)

Two passes over chunks of the dataset (forked workers, memory-mapped columns). Per target group (helium, air, PVC,
each plane) the crossings are drawn with probability w P (P = n_area sigma(E) << 1); each draw has weight
S_g / M_g (interactions per spill). Vertex uniform along the crossing. Final state from the library (neighbouring
grid energy at random, linear in log E), rotated to the muon direction with a random azimuth. Output per
interaction: muon row, evt_n, type, target, vertex, E, direction, weight, generator; per particle: pdg, momentum
(lab), production point (decay vertices), scattered-muon flag.

Full gamma dataset (88 M rows), 4 M interactions (1 M helium, 1 M PVC, 0.2 M air, 0.2 M per plane): **37 s** with
40 workers (12 s geometry, 12 s draws, 14 s final states), any technology. Interactions per 15 years (crossover):

| | helium | PVC | UBT (straw) | one fin, straw / fibre / pixel / tile |
|---|---|---|---|---|
| D6 designs | 1.50e6 | 2.16e7 | 3.2e4 | 5.1e9 / 2.9e10 / 1.1e10 / 1.2e11 (left fin z = 30.6) |

PVC: with the wall path capped at 50 x thickness for grazing incidence (as vetosim's scattering does) the rate was
4.9e6, 4.4 x too low: muons and neutrinos cross the inclined lateral walls at tens of mrad, where the true chord is
much longer. With exact chords 2.16e7, as the earlier full-Geant4 rate studies (1.8 - 1.9e7). vetosim's scattering
uses the same cap for the walls' x/X0.

The helium rate agrees with the earlier full-Geant4 rate study on the same dataset (1.46e6; +3 % from the DIS
growth in Pythia). Fins: muon flames.

### Closure against full Geant4 (`bin/dis_closure.py`)

2 M interactions (helium + PVC) simulated in full Geant4 (muon_dis_forced.py: forced interaction, everything
transported to z = 82 m) vs the same draws through the fast chain (library final state, straight lines). Charged
tracks besides the muon with p > 1 GeV in the SST1 acceptance, per 15 years:

| helium | >= 1 track | >= 2 tracks | P(>= 4) |
|---|---|---|---|
| full Geant4 | 1.95e5 | 8.6e4 | 1.71 % |
| fast, split geant4 | 1.85e5 (-5 %) | 7.8e4 (-9 %) | 1.45 % |
| fast, crossover | 2.57e5 | 1.34e5 | 2.95 % |
| fast, w2 | 2.04e5 | 1.29e5 | 3.78 % |

PVC: -3 % / -9 % for the fast chain with Geant4 physics. The remaining deficit of the fast chain is what it leaves
out (photon conversions, secondary interactions, transport). Pythia's final states give 1.4 x more events with
>= 1 and 1.7 x more with >= 2 extra tracks than Geant4's: a generator systematic for vertex backgrounds.

## Multiple scattering (fastveto.msc: vetosim's Gaussian core + single-scattering tails)

vetosim's scattering.py (Highland on the muon's total x/X0, shared as Gaussian kicks over the scatterers; helium and
air as 1 m slices with media = True) gives the core. `lib/fastveto/msc.py` adds the tails of rare large single scatters:
above theta_c = 2.0 x theta0 a Poisson number of hard scatters per scatterer, rate per radiation length from
Rutherford (nuclei + electrons) over 1/X0 = pi m_e^2 / (alpha L (p beta theta_c)^2) (L = ln(183 Z^-1/3) ~ 4.5,
nearly material independent), 1/theta^4 spectrum up to the nuclear-size cut-off. One parameter (2.0) tuned to
Geant4. `msc.kicks()` has vetosim's kicks() layout: scattering.offset_at / scattered_tracks work unchanged.

`bin/msc_validate.py`: mu- on the axis from z = 30.55 m through window, 50 m helium, exit wall, air to 82 m,
50k per momentum; Geant4 (nuvessel, FTFP_BERT) vs vetosim Gaussian vs fastveto.msc. |x| at z = 82 m:

| p [GeV] | core G4 / Gauss / tails [mm] | RMS G4 / Gauss / tails | 99 % G4 / Gauss / tails | beyond 5 cores G4 / Gauss / tails |
|---|---|---|---|---|
| 1.5 | 24.8 / 26.8 / 27.0 | 35.0 / 26.6 / 37.0 | 90 / 69 / 88 | 0.44 % / 0 / 0.52 % |
| 3 | 12.5 / 13.2 / 13.5 | 15.8 / 13.2 / 17.6 | 47 / 34 / 44 | 0.46 % / 0 / 0.51 % |
| 10 | 3.75 / 3.94 / 4.06 | 4.73 / 3.97 / 4.78 | 13.9 / 10.2 / 12.7 | 0.49 % / 0 / 0.41 % |
| 30 | 1.25 / 1.31 / 1.34 | 1.64 / 1.32 / 1.62 | 4.5 / 3.4 / 4.3 | 0.44 % / 0 / 0.45 % |
| 100 | 0.37 / 0.40 / 0.41 | 0.51 / 0.40 / 0.51 | 1.35 / 1.01 / 1.38 | 0.44 % / 0 / 0.54 % |

Angles agree as well (results/msc/msc_tails.png, msc_validation.json). The core is Highland's (2 - 8 % wide in
position, inherited). Validated for the vessel path (helium, PVC, air); dense planes (tile, 2.4 % X0) not yet
checked against Geant4 (the tail rate per X0 is nearly material independent, so expected to hold).

## Neutrinos (GENIE)

**Generator**: our own GENIE 3.06.02 (`cpp/genie/build_genie.sh`: reference configuration with Pythia 6.4.28
hadronisation and decays, GENIE's default configs; one override `cpp/genie/config/CommonDecay.xml` that decays charm
and taus, whose muons matter here). Built in the cudabear container (ROOT 6.36) with GSL, libxml2, log4cpp, LHAPDF 6,
oneTBB and the standalone EGPythia6 (TPythia6, removed from ROOT in 6.30) built from source; run via `bin/genie.sh`.
(LCG_106's gcc 13 cannot compile <mutex> on the el9 farm nodes; the FairShip pixi GENIE is a Pythia 8-only build of
SHiP's channel that needs FairShip's config and aborted in tests.) Tune **G18_02a_00_000** with the official splines
(genie_xsec v3_06_00, 250 knots to 1 TeV, NUsmall, trimmed to our nuclei by `bin/genie_splines.py`):
nu_mu (CC + NC) 0.89e-38 cm^2/GeV per nucleon at 100 GeV, 1.08 at 5 GeV (QE + RES); anti-nu_mu 0.45.

**Library** (`bin/nu_library.py` -> `data/interactions/genie/nu_library.npz`, 0.9 GB, 4.5 min on farm-gpu5): nu_mu, anti-nu_mu,
nu_e, anti-nu_e on H1, He4, C12, N14, O16, Si28, Cl35, Cl37, Ar40, 36 energies 1 .. 500 GeV, 2000 events each.
Final state after FSI + the outgoing lepton; K0 -> K_S / K_L; K_S, hyperons, pi0, eta decayed by fastveto.decay.
Process mix (nu_mu He4): 5 GeV QE 13 %, RES 28 %, DIS 56 %; 50 GeV DIS 94 %; charm 7 % at 300 GeV.

**Application** (`lib/fastveto/nudis.py` with the same `dis.run`; `bin/nu_prepare.py` puts the flux sample in the
dataset layout, one store per flavour; driver `bin/nu_dis_apply.py --design D6_straw`): materials as nuclei,
nucleus drawn by n_i sigma_i(E). 4.2 M interactions in 90 s. Per 15 years (D6_straw):

| target | nu_mu | anti-nu_mu | total | earlier estimate | >= 2 opposite-charge tracks p > 1 GeV in SST1 |
|---|---|---|---|---|---|
| helium | 4070 | 1210 | 5290 | ~5000 | 1360 |
| PVC | 2700 | 990 | 3690 | ~3400 | 374 |
| air (30.6 - 82 m) | 4050 | 1640 | 5690 | | 881 |
| UBT | 214 | 64 | 278 | | 18 |

CC fraction 0.76 (nu), 0.73 (anti-nu). The flux sample has no nu_e and ~1 % nu_tau (no tau library yet).

## Open

* Library final states are not rescaled to the exact muon energy (15 % grid).
* No nuclear effects (Fermi motion, shadowing, re-interactions) in the Pythia part.
* MSC tails: Geant4 check for a dense plane (tile).
