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

## K_L attenuation in the iron: already in the chain (check 2026-11-02, `bin/shield_kl_check.py`)

Question (Guglielmo): a K_L made at the start of the 50 cm sees ~3 interaction lengths of iron (λ ≈ 17 cm); is its
survival to the shield end in the weight? Answer: yes, through step 2. Every final-state particle, K_L included, is
tracked by Geant4 (FTFP_BERT) from its vertex through the iron; a K_L that interacts is gone and only what comes out
of the cascade reaches the plane. No extra exp(−d/λ) weight is needed: adding one would count the absorption twice.

| check | result |
|---|---|
| single K_L in g4shield (2 - 100 GeV, 5 / 25 / 50 cm of iron, 4 000 each) | primary K_L out: λ_eff = 16 - 18 cm (e.g. 10 GeV, 50 cm: 6.7 %) |
| K_L made by μ DIS, p > 1 GeV (library, before transport, 3×10⁵ draws) | 2.07×10⁴ per spill, mean iron left 25.8 cm |
| same × exp(−d / 17 cm) (primaries only) | 6.4×10³ per spill |
| K_L at the plane after Geant4 (`shield_muon.json`) | 1.04×10⁴ per spill |

The difference 1.04×10⁴ − 6.4×10³ is K_L made in the hadronic cascade inside the iron (π, n, K_L interactions):
in the single-particle runs "any K_L out" exceeds "primary K_L out" (30 GeV, 25 cm: 0.39 vs 0.25 per K_L), so a
pure exponential weight would undercount. The vessel step then weights each K_L at the plane by P(decay in
30.6 - 80.6 m) = e^(−d₁/λ_dec) − e^(−d₂/λ_dec), which contains its survival (no decay) up to the vessel; the 0.7 m of
air to the vessel is not simulated (λ_I of air ≈ 7 km).

## Chain probabilities per species (2026-11-02, `bin/shield_chain_probs.py` → `results/shield_dis/chain_probs.json`)

40 000 forced μ interactions (∝ w·P) through Geant4, p > 1 GeV. P_int (rate mean, ≤ 50 cm Fe) = 2.1×10⁻³; for the full 50 cm:
1.2×10⁻³ (5 GeV), 1.7×10⁻³ (10), 2.6×10⁻³ (30), 3.9×10⁻³ (100), 5.2×10⁻³ (300 GeV). 4.87×10⁸ μ/spill → 1.04×10⁶ interactions/spill.

| species | made / interaction | P_out primary | P_out all | out / spill | P_dec in vessel z range |
|---|---|---|---|---|---|
| K_L | 0.019 | 0.31 | 0.51 | 1.03×10⁴ | 0.43 (→ 4.4×10³ decays, ×0.80 two charged) |
| K_S | 0 (decayed by the generator) | – | – | 1.56×10³ (from the shower) | 0.038 |
| Λ | 0 (idem) | – | – | 755 | 0.058 |
| π± | 0.44 | 0.29 | 0.46 | 2.08×10⁵ | 0.34 |
| K± | 0.040 | 0.35 | 0.51 | 2.16×10⁴ | 0.76 |
| p | 0.20 | 0.21 | 0.49 | 1.04×10⁵ | stable |
| n | 0.21 | 0.25 | 0.55 | 1.18×10⁵ | stable |
| μ | 0.99 | 1.00 | 1.00 | 1.03×10⁶ | 4.5×10⁻⁴ |

P_out all / made > primary: hadrons made in the shower inside the iron. P_dec = decay between z = 30.6 and 80.6 m on a straight
line (not restricted to the vessel volume), as `force_v0` does.

## Is the MS7 K_L number too big? Checks (2026-11-02)

| check | result |
|---|---|
| muon flux in the MS7 footprint at z = 30.6 m: TRY6 sample (straight) vs Cuda store | 2.57×10⁸ vs 2.43×10⁸ /spill (p > 20 GeV: 1.95 vs 2.01×10⁸): same |
| FairShip TRY_2026 Magn7 cross-section (`fairship_sst1_origin/magn7_xsection.py`) | solid iron \|x\| < 1.71, \|y\| < 1.27 m, as TRY6 MS7 |
| FairShip run options | no FastMuon; muonNuclear on (KokoulinMuonNuclearXS), FTFP_BERT_HP_EMZ |
| plain (unforced) Geant4, all 25.7 M muons through the 50 cm (`bin/shield_natural_check.py` → `results/shield_dis/natural_check.json`) | out of the iron /spill, natural ÷ forced chain: K_L 5.3×10³ ± 1.6×10³ (0.52), K_S 0.73, Λ 0.77, π± 0.67, K± 0.40, p 0.84, n 0.78 |
| K_L by library component (4×10⁵ draws) | Pythia 6 part (component 0): 9.5 % of interactions, 0.19 K_L each = 1.85×10⁴ /spill (90 % of the K_L); Geant4 part: 0.0021 K_L per interaction |

So the forced chain makes ~2× more K_L than plain Geant4 muonNuclear, all from the Pythia (high-ν DIS) component. FairShip uses
Geant4 only, so on this point it would sit with "natural". The FairShip MS7 deficit (π at SST1 ~17× below the chain) is
not explained by this (natural π = 0.67 × chain): still open.

## K_S and Lambda tracked (2026-11-02)

The default library decays K_S and Λ at the generator (fine in helium, wrong in iron: no absorption, and decays after the
shield end were never recorded by g4shield). `mu_library_iron_v0stable.npz` keeps them; Geant4 decays them in flight.
Effect on the MS7 μ chain (per 15 y after DOCA < 1 cm, fiducial, IP < 2.5 m): 5.87e7 → (6.73 ± 0.59)e7, K_S and Λ
3e4 each; measured IP < 10 cm + ≤ 2 tracks 5.9e4 (7 MC) → 3.8e4 (5 MC). Through the veto (scripts/dis_veto_ubt4x6_v0stable.py):
D0_minimal 6.73e7, U1_ubt4x6_tile 5.28e7 (level b).

## Limitations
- Straight lines from z = 29.20 m: no bending / scattering of the parent muon before the vertex (2 T over ≤ 0.69 m:
  ~1.4 cm at 20 GeV); muons that stop inside the slab are not removed (mean loss only).
- Library final states (default library): K_S / Λ decayed at their generator decay point; use mu_library_iron_v0stable.npz; Pythia part on free nucleons (no nuclear effects).
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
| `bin/shield_kl_check.py` | K_L made by μ DIS vs K_L at the plane (iron attenuation check) |
| `results/shield_dis/` | `shield_{muon,nu}.npz/.json`, per-design veto results, logs |
| `docs/dis_chain_diagrams.html` | flow-chart explainer of the vessel + shield μ DIS chains (2026-11-03), https://claude.ai/artifact/XsCHB2iJyWDsNXgsTJTcEz |
| `docs/dis_chain_pdf/` | PDF version (`dis_chain.pdf`, LaTeX + Graphviz sources), incl. the forced-V0-decay explanation (2026-11-03) |

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
