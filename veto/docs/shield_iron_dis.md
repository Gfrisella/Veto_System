# DIS of muons and neutrinos in the last 50 cm of the muon shield

Started 2026-10-30, branch `veto-shield-dis`. Question: interactions in the last iron of the shield (MS7) make
particles that leave the iron and can fake a vertex in the decay vessel (a V0 decaying in the vessel, or charged
products), with the scattered muon / debris possibly seen by the veto. The interaction chain of `lib/fastveto`
covers only z = 30.6 - 82 m; this adds the iron.

## Geometry and inputs (`lib/veto/shield_iron.py`)

| item | value | source |
|---|---|---|
| shield end | z = 29.89 m | MuonsAndMatter `reference_designs/params.py` TRY6 (MS7 = 25.89 - 29.89 m, NI −2.03) |
| target | G4_Fe box \|x\| < 1.71 m, \|y\| < 1.27 m, z = 29.39 - 29.89 m | MS7 is one solid iron Arb8 (`create_magnet`: coil gap filled with iron); outer envelope constant (38+8+125 = 69+8+94 cm, 51+76 cm) |
| field | `MuonsAndMatter/data/outputs/fields_mm.h5` (FEM, diluted), first quadrant mirrored (By even, Bx odd in x·y, Bz odd in y), trilinear; core −1.89 T, return yoke +1.25…1.34 T, top yoke Bx ≈ −1.1 T | the map's MS7 ends at 30.04 m (`magnet_params.csv`): looked up at z + 0.15 m so map and iron end together |
| muons | `MuonsAndMatter/data/outputs/Muons_2950_50cm_before_end_MS.pkl`: 149 M muons, 3.44e9 per spill, one run of the old 506 M input | plane stored at z = 29.50 m, **true z = 29.20 m (translated by 30 cm, Guglielmo)**; straight lines from there, mean dE/dx 1.6 GeV/m to the vertex |
| muons through the iron | 26.0 M rows, 4.87e8 per spill (the others are already outside the magnet envelope) | `bin/shield_dis.py prepare` → `data/shield_dis/muons_iron.npz` |
| neutrinos | ν_μ, ν̄_μ stores of fastveto (2018 FairShip, p > 10 GeV), straight from the target: 1.06e7 rows / 5.38e9 per spill cross the iron | `/disk/users/sqasim/.../data/interactions/neutrinos/store_nu{14,-14}` |

## Chain (`bin/shield_dis.py`)

1. **Forced interactions.** P_i = n_nucleon σ(E_i) L_i (L = chord in the iron box). S = Σ w_i P_i per spill.
   N interactions drawn ∝ w_i P_i (each carries S / N), vertex uniform on the chord.
   - muons: `data/interactions/dis/mu_library_iron.npz` = sqasim's `mu_library.npz` + a new Geant4 iron table
     (`bin/dis_library.py --only-g4 iron`, 50 energies × 20 000 events, G4_Fe muonNuclear): Geant4 below ν_x(E),
     Pythia 6 on p / n above (Z/A of iron). σ per nucleon in iron = 0.82 × helium (Kokoulin shadowing); Pythia share
     1.5 % at 10 GeV, 13 % at 100 GeV.
   - neutrinos: `data/interactions/genie/nu_library_iron.npz`, GENIE 3.06.02 G18_02a on Fe56 (`bin/nu_library.py
     --nuclei 1000260560 --nus 14 -14 --n 5000`, splines with Fe56 from `bin/genie_splines.py`). σ/E per nucleon
     0.90 (ν) / 0.44 (ν̄) × 10⁻³⁸ cm²/GeV.
2. **Geant4 transport** (`cpp/g4shield.cc`, FTFP_BERT, field map): every final-state particle through the iron to
   z = 29.90 m; kept above 0.5 GeV kinetic, neutrinos dropped. Checks (`bin/shield_dis_smoke.py`): 50 GeV π
   survive 8.9 % (e^(−0.5/0.204) = 8.6 %), 10 GeV μ⁺ bent 7 mm (0.3 B L² / 2p), 3 - 7 ms per hadron event.
3. **Forced V0 decays in the vessel**: each K_L, K_S, Λ, Ξ⁰ at the plane gets a copy of its interaction (code 4,
   `shield_iron_v0`) with the V0 decayed inside z = 30.6 - 80.6 m (truncated exponential, weight × P(decay
   there), `fastveto.decay.force_decay`, K_L channels added); the rest of the event comes along. Code 3
   (`shield_iron`) = the interaction itself with every product as a straight line from the plane.
4. **Selection and veto**: `bin/run_interactions.py --kind nu --interactions results/shield_dis/shield_<kind>.npz`
   (no parent hits: `beam` = False), cut flow `artifacts_work/scripts/shield_cutflow.py` (deck's steps).

## Limitations
- Straight lines from z = 29.20 m: no bending / scattering of the parent muon before the vertex (2 T over ≤ 0.69 m:
  ~1.4 cm at 20 GeV); muons that stop inside the slab are not removed (mean loss only).
- Library final states: K_S / Λ from the generator are decayed at their generator decay point (cm), not
  transported as K_S through the iron; Pythia part on free nucleons (no nuclear effects).
- After the plane: straight lines, no field / scattering in the 0.7 m of air; decays in flight of π / K ignored;
  neutrons and K_L interactions in the vessel / air not simulated.
- Neutrino flux p > 10 GeV only; ν_e not included.

## Files

| file | what |
|---|---|
| `lib/veto/shield_iron.py` | geometry, field map, chord, Geant4 set-up |
| `cpp/g4shield.cc` | Geant4 transport module (built by `cpp/build.sh`) |
| `bin/shield_dis.py` | prepare / generate |
| `bin/shield_dis_smoke.py` | field and transport checks |
| `results/shield_dis/` | `shield_{muon,nu}.npz/.json`, per-design veto results, logs |

## Results (2026-10-30; 2 M μ draws, 100 decays per V0; 10⁶ ν draws, 20 decays per V0)

What leaves the iron, p > 1 GeV, per spill (= Hz at 5×10¹³ PoT/s), from `results/shield_dis/shield_{muon,nu}.json`:

| | μ DIS | ν DIS |
|---|---|---|
| interactions in the 50 cm | 1.04 MHz | 0.23 Hz |
| μ / π± / p / n | 1.03 MHz / 214 kHz / 105 kHz / 120 kHz | 0.17 / 0.16 / 0.069 / 0.073 Hz |
| K_L / K_S / Λ | 10.4 kHz / 1.78 kHz / 0.85 kHz | 8.5 / 1.7 / 0.78 mHz |
| K_L decaying in the vessel | 4.89 kHz | 4.0 mHz |

Cut flow per 15 years (`artifacts_work/scripts/shield_cutflow.py`, deck steps; shield interactions + V0 copies):

| step | μ DIS | ν DIS |
|---|---|---|
| interactions | 1.25×10¹³ | 2.76×10⁶ |
| fiducial vertex | 9.8×10⁷ | 729 |
| IP < 2.5 m | (6.6 ± 0.3)×10⁷ (6 058 MC) | 702 ± 25 |
| + measured IP < 10 cm | (1.2 ± 0.4)×10⁵ (9 MC) | 7.2 ± 2.2 |
| + ≤ 2 tracks in SST1 | (5.0 ± 2.9)×10⁴ (3 MC) | 1.07 ± 0.24 (78 MC) |
| after the veto, D6_straw, level b (`run_interactions.py`) | 3.8×10⁴ of 3.8×10⁴ (2 MC) | 0.54 of 1.06 |

- No prompt product makes a vertex inside the vessel: everything that survives is a V0 decaying in the vessel,
  96 % K_L (the K_S mostly decay in the 0.7 m of air before the vessel).
- The veto (algorithm A) cannot remove the μ-induced K_L: the decay tracks have no hits upstream of their vertex
  and the scattered muon is not near their back-extrapolation. A veto on any in-time hit (e.g. the scattered
  muon in the UBT / fins) is not modelled.
- Statistics: the μ number rests on 3 MC events (±60 %, errors summed per input muon evt_n). More statistics:
  more draws, or K_L-enriched draws.
- Bug found and fixed before these numbers: muon final states were sampled twice per interaction (2 muons at the plane).

Deck: https://claude.ai/artifact/NopX9q7y7h9cMN19qYn6m8 v116, slides 25 – 28.

## One interaction per crossing / per track (2026-10-30, `results/dis_per_track/run.sh`)

Every crossing (vessel He, PVC; ν also air) or every shield track gets exactly one forced interaction with weight
w·P (air and veto planes for μ: 2e5 draws ∝ w·P, vertices outside the vessel). Per 15 years:

| | draws ∝ w·P | one per crossing / track |
|---|---|---|
| μ He, IP < 2.5 m | (6.0 ± 1.4)×10⁴ | (5.9 ± 1.4)×10⁴ (2.4e7 MC) |
| μ He, old selection | 1 800 ± 650 | 1 806 ± 681 |
| ν He, IP < 2.5 m | 467 ± 2 | 470 ± 1 |
| MS7 μ, old selection | (5.0 ± 2.9)×10⁴, 3 MC | (5.9 ± 2.3)×10⁴, 7 MC |
| MS7 ν, old selection | 1.07 ± 0.24 | 1.90 ± 0.33 |

The μ errors do not shrink: they come from the few independent input muons (errors per evt_n), not from the draws.
D6_straw veto, MS7 μ V0: 3.4e4 selected (4 MC) → 2.1e4 after level b (`results/dis_per_track/D6_straw_shield_muon.json`).
