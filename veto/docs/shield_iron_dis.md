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

## DIS generator comparison (2026-11-06, D3)

Question: how much of the MS7 K_L comes from the choice of muon-nucleon generator, and does it explain FairShip?
Same chain for every column (`shield_dis.py generate --kind muon --n 2000000`, bent parents, FairShip-2026 vessel
layout, Geant4 10.7.3 transport through the iron, 1 forced decay per V0), only the library part changes
(`--mu-split`, `--mu-lib`, new 2026-11-06). Logs: `logs/generators_2026-11-06/`.

| column | generator of the muon-nucleon interaction | library |
|---|---|---|
| Pythia 6 (default since 5 Nov) | Pythia 6.4.28, γ*/μ on p, n: VMD + direct + anomalous + DIS, W > 2 GeV, all Q² | `mu_library_iron.npz`, split `pythia` |
| G4 + P6 crossover (old default) | Geant4 below ν_x(E), Pythia 6 above | same, split `crossover` |
| A: Geant4 11.2 | Geant4 11.2.2 muonNuclear only (Kokoulin σ + G4MuonVDNuclearModel, FTFP_BERT), G4_Fe | same, split `geant4` |
| C: Geant4 10.7 | the same with **FairShip's** Geant4 10.7.3 (CVMFS; `g4dis` built by `bin/g4ship.sh build`) | `mu_library_iron_g4v1073.npz`, split `geant4` |
| P6 Q² > 1 | Pythia 6 events with Q² > 1 GeV² only (σ × kept fraction) | `mu_library_iron_p6q2.npz` (`bin/p8_library.py`) |
| P8 DIS Q² > 1 | Pythia 8.230 (CVMFS) NC DIS `WeakBosonExchange:ff2ff(t:gmZ)`, Q² > 1, W > 2, mHatMin = 0 | `mu_library_iron_p8dis.npz` (`cpp/pythia8dis`, `bin/p8_library.py`) |

### Cross sections (`bin/plot_dis_sigma_generators.py` → `results/shield_dis/dis_sigma_generators.png`)

Per nucleon in iron [μb]:

| E_μ [GeV] | Geant4 (all ν) | Pythia 6 (W > 2, all Q²) | Pythia 6, Q² > 1 | Pythia 8 DIS, Q² > 1 |
|---|---|---|---|---|
| 8.4 | 6.46 | 0.42 | 0.112 | 0.008 |
| 34 | 11.3 | 1.83 | 0.386 | 0.079 |
| 141 | 16.9 | 4.08 | 0.566 | 0.190 |
| 500 | 22.5 | 6.76 | 0.695 | 0.316 |

- Geant4 10.7.3 and 11.2.2 give the same σ (both Kokoulin; 9.37 vs 9.28 μb at 20 GeV).
- In the hard region (Q² > 1) Pythia 6 is 2–3× above Pythia 8 at high E and up to ~10× at low E: Pythia 6's γ*
  model adds resolved (VMD / anomalous) components to plain DIS; Pythia 8 has only the hard process.
  A rough F2 ≈ 0.3 estimate gives ~0.3 μb at 100 GeV, in between.
- Per hard (Q² > 1) interaction the final states are close: K_L 0.084 (P6) vs 0.066 (P8), π± 2.63 vs 2.34,
  K± 0.19 vs 0.16 (`results/shield_dis/p6_vs_p8_q2gt1_sigma.txt`).
- Pythia 8.230 photoproduction (Q² < 1, `PDF:lepton2gamma`) runs at 100 GeV but hangs at 20 GeV (√s ≈ 6 GeV): the
  soft region, where most K_L come from, cannot be done with this version (needs Pythia 8.3).
- Pythia 8.230 production vertices are unusable (hadrons from strings / resonances at ~10⁶ m): vertices written as 0,
  K_S, Λ, Σ, Ξ, Ω decayed by `fastveto.decay` with their decay lengths (as the Geant4 part). Found because the first
  P8 chain had 28× fewer K_L out of the iron than its generator-level K_L implied.

### Chain results (`bin/compare_shield_generators.py` → `results/shield_dis/generators.{md,json}`)

Per spill (Hz at 5×10¹³ PoT/s); SST1 = tracks made by the chain, straight from their start to z = 82 m, |x| < 2,
|y| < 3, all p, as in the FairShip comparison; (n) = MC entries. FairShip: TRY_2026_EM_SST1_origin, tracks made in the
MS7 last 50 cm, cut 1 (`../FairShip_TRY2026_EM_SST1/fairship_sst1_origin/README.md`).

| per spill | Pythia 6 (default) | G4+P6 crossover | A: G4 11.2 | C: G4 10.7 | P6 Q2>1 | P8 DIS Q2>1 | FairShip (full physics) |
|---|---|---|---|---|---|---|---|
| interactions | 1.78e+05 | 1.04e+06 | 9.93e+05 | 9.93e+05 | 2.97e+04 | 7.96e+03 |  |
| exit K_L | 1.19e+04 | 1.11e+04 | 4.01e+03 | 4.23e+03 | 1.75e+03 | 286 |  |
| exit K_S | 2.1e+03 | 2.09e+03 | 680 | 727 | 351 | 66.5 |  |
| exit Lambda | 946 | 923 | 350 | 357 | 158 | 30.5 |  |
| exit pi+- | 2.04e+05 | 2.1e+05 | 1.18e+05 | 1.16e+05 | 2.97e+04 | 5.77e+03 |  |
| exit p | 7.06e+04 | 1.05e+05 | 7.38e+04 | 7.42e+04 | 1.16e+04 | 2.36e+03 |  |
| exit n | 7.75e+04 | 1.2e+05 | 8.65e+04 | 8.68e+04 | 1.26e+04 | 2.56e+03 |  |
| K_L decays in vessel | 5.6e+03 | 5.17e+03 | 2.14e+03 | 2.26e+03 | 827 | 139 |  |
| K_S decays in vessel | 35.6 | 35.1 | 8.54 | 11 | 7.31 | 1.37 |  |
| SST1 pi | 1.27e+04 (142522) | 1.33e+04 (25586) | 7.4e+03 (14909) | 7.31e+03 (14737) | 1.76e+03 (118432) | 352 (88553) | 800 |
| SST1 p | 1.84e+03 (20673) | 2.44e+03 (4682) | 1.58e+03 (3187) | 1.6e+03 (3223) | 274 (18419) | 65.9 (16563) | 0 |
| SST1 K | 1.46e+03 (16359) | 1.38e+03 (2653) | 451 (908) | 480 (968) | 220 (14829) | 38.2 (9604) |  |
| SST1 e | 7.15e+03 (80181) | 7.96e+03 (15285) | 4.39e+03 (8846) | 4.27e+03 (8608) | 1.11e+03 (74839) | 240 (60292) |  |
| SST1 mu | 489 (5485) | 424 (815) | 211 (425) | 195 (393) | 310 (20857) | 47 (11824) |  |
| SST1 K0-decay tracks | 340 (13060) | 307 (2048) | 122 (719) | 123 (761) | 46.8 (11182) | 7.92 (7280) | 8 |
| SST1 K_L-decay tracks | 335 (11524) | 303 (1794) | 121 (638) | 121 (672) | 45.9 (9729) | 7.73 (6189) |  |
| SST1 K_S-decay tracks | 5.26 (1536) | 4.32 (254) | 0.869 (81) | 1.13 (89) | 0.849 (1453) | 0.189 (1091) |  |

- **Pythia 6 → Geant4 only: K_L ÷ 2.6** (K_L decays in the vessel 5 604 → 2 140, K0-decay tracks at SST1 340 → 122 Hz).
- **Geant4 version does not matter**: C = A within a few % (2 260 vs 2 140 K_L decays, 123 vs 122 Hz).
- The crossover mix (old default) ≈ Pythia 6 only for the K_L: its Pythia part is 10 % of the interactions but
  85 % of the K_L.
- Even with FairShip's own physics (C) the chain is ~15× (K0-decay tracks) and ~9× (π) above FairShip: the generator
  explains a factor ~2.6, not the FairShip discrepancy. FairShip numbers are single MC entries.
- (Closed 2026-11-09, see below: not a muon-reaches-SST filter.) The FairShip sample was made with `--MuonBackSSTOnly`. If that keeps only muons / events reaching the SST, most
  MS7 interactions are missing from it by construction (the scattered muon reaches SST1 in ~2×10⁻⁴ of the chain's
  interactions), and FairShip is a lower limit. To be checked in the FairShip_sst1 branch on lxplus.

### K0-decay tracks with the tracking cuts (2026-11-09)

Same chain outputs, K0-decay tracks with |p| > 1 GeV seen in ≥ 3 of the 4 straw stations (seen = ≥ 1 straw hit;
`lib/veto/straws.py` as implemented: 32 layers, FairShip strawtubes geometry, RK4 in `MainSpectrometerField.h5` at
Z_MAGNET = 87.58 m, yoke losses; no scattering, no fit), independent of the straight-line SST1 box. Rows "trk" in
`results/shield_dis/generators.md` (old table kept as `generators_2026-11-06.md`). Hz, (n) = MC entries:

| per spill | Pythia 6 (default) | G4+P6 crossover | A: G4 11.2 | C: G4 10.7 | P6 Q2>1 | P8 DIS Q2>1 |
|---|---|---|---|---|---|---|
| K0-decay tracks at SST1 (box, all p) | 340 | 307 | 122 | 123 | 46.8 | 7.92 |
| K0-decay tracks, ≥ 3/4 stations, p > 1 | 141 (7129) | 127 (1135) | 47.7 (370) | 48.1 (392) | 19.8 (6229) | 3.47 (4225) |
| of which K_L / K_S | 137 / 4.19 | 123 / 3.62 | 47.2 / 0.52 | 47.2 / 0.94 | 19.1 / 0.69 | 3.32 / 0.16 |
| K0 decays with ≥ 2 such tracks | 14.2 (883) | 14.0 (145) | 3.90 (38) | 3.84 (38) | 1.99 (792) | 0.384 (577) |

- The cuts keep ~40 % of the K0-decay tracks (mostly the p > 1 GeV cut); ~10 % of the tracks belong to decays with
  two good tracks (14 Hz default, 3.9 Hz with FairShip's Geant4 10.7).
- FairShip's 8 Hz is not a K0-decay track. The only charged track at SST1 with a K0 parent made in the shield
  (`tracks_after_cut1_vol.pkl`, row 2295436) is a π⁻ of 0.61 GeV made by a **hadronic interaction of the K_L in the
  cavern concrete at z_FS = 78.4 m** (K_L made at z_FS = 31.82 m = MS7 last 50 cm), seen in 2 stations. It fails
  p > 1 GeV and 3/4 stations. FairShip has **0 MS7 K0-decay tracks**: < 18 Hz at 90 % CL (2.3 × 7.69 Hz per MC entry).
  Expected MC entries from our chain at FairShip's weight: box, all p 44 (Pythia 6) / 16 (G4 10.7); with the cuts
  18 / 6.3 (P(0) = 0.002 for G4 10.7); 2-track decays 1.8 / 0.5 (not constraining).

### What `--MuonBackSSTOnly` does (2026-11-09, checked on lxplus from job 0_100000_1 of TRY_2026_EM_SST1_origin)

The branch code (/afs/cern.ch/work/g/gfrisell/FairShip_sst1) could not be read (AFS work volume timed out); answered
from the job's log + ROOT file and the upstream code the branch inherits:
- Every muon is transported (100 000 / 100 000); after transport, events with no hit in any `*Point` container (Calo and
  UpstreamTagger scoring planes ignored; SSTScoringPoint counts) are removed: 822 kept, 799 with SST hits, none of the
  926 SST hits is a muon. The save condition is "any real hit", not "the muon reaches the SST": **FairShip is not a
  lower limit for that reason** (the open point above is closed). No sign of track killing during transport
  (5 315 stored shield secondaries at z 0–20 m, 10-generation chains, SST-hit tracks starting up to 127 m).
- `MCTracksWithHitsOnly` (ShipStack::SelectTracks) keeps a track only if it, or a descendant, has a hit: a K_L whose
  decay products reach SST1 is kept, so the K0-decay count above is valid.
- SSTScoringPoint also records neutrinos (414 of the 926 hits) and photons (453); the comparison numbers use charged
  tracks only (`origin_breakdown.py`), so they are not affected.
- The sample cannot give the fraction of MS7 interactions reaching SST1 (no-hit interactions are not stored): compare
  rates per particle type at SST1.

### FairShip's muon-nuclear definition (2026-11-09, checked on lxplus in /cvmfs/ship.cern.ch/26.10)

- `--MuonBack` transport (TRY_2026 sample): Geant4 muonNuclear switched on (`SetProcess("MUNU", 1)`, SetCuts.C), no
  muon-nuclear specific cut in SetCuts.C / g4config.in. G4KokoulinMuonNuclearXS + G4MuonVDNuclearModel: only cut
  ν > 0.2 GeV (CutFixed); Q² between the kinematic limits, no W cut; γ* → Bertini below 10 GeV, π⁰ in FTFP above.
- `muonDIS/makeMuonDIS.py` (Pythia 6.4.28, not used in `--MuonBack`): MSEL = 2, PARP(2) = 2, CKIN defaults → W ≥ 2 GeV.
- Ours: the Geant4 part (`cpp/g4dis.cc`, FTFP_BERT) is the same process with no extra cut; the Pythia 6 part is the
  makeMuonDIS set-up (W > 2 GeV). So **columns A / C (Geant4 only) define an interaction exactly as the FairShip
  sample** (9.9e5 / spill); the Pythia-6 default (1.8e5, W > 2) is not directly comparable by interaction count.
- Consequence for D3: the like-for-like column C still expects 16 (box) / 6.3 (≥ 3/4 stations, p > 1) MC entries
  where FairShip has 0. The generator / cut definition is ruled out as the explanation; left: the parent muon flux
  into the last 50 cm (our TRY6 one-run pickle vs FairShip TRY_2026 from the target), production cuts / track killing
  in the FairShip run, and our chain's V0 transport and forced-decay weights.

### Pythia 8 photoproduction + DIS (2026-11-09, Annual Report 2026 Table 7 settings)

`bin/p8_library.py --variant full` → `mu_library_iron_p8full.npz`: Pythia 8.230 photoproduction (`PDF:lepton2gamma`,
`Photon:Q2max = 1.5`, `Photon:Wmin = 5`, the lowest value 8.230 allows; it hung before because the default W_min = 10
GeV is above W_max at 20 GeV) + DIS (Q² > 1.5, W > 2), mixed by cross section. Photoproduction only above E_μ ≈ 14
GeV (W_max > 5). σ per nucleon (p): 0.13 / 0.51 / 1.85 / 5.0 μb photoproduction + 0.04 / 0.07 / 0.13 / 0.24 μb DIS at
19.5 / 34 / 106 / 500 GeV; total 35–75 % of Pythia 6. Not covered: Q² < 1.5 with W < 5 GeV.
Chain (`shield_muon_fs2026_p8full`, all columns now per 4e13 p.o.t.; older runs × 0.8 in `generators.md`):

| per spill (4e13) | Pythia 6 | Geant4 10.7 | P8 PP W>5 + DIS | FairShip |
|---|---|---|---|---|
| K_L decays in the vessel | 4 480 | 1 810 | 2 640 | |
| K0-decay tracks at SST1 (box, all p) | 272 Hz | 98 Hz | 164 Hz | 0 (< 14 Hz) |
| ≥ 3/4 stations, p > 1 | 113 Hz (18.4 entries) | 38.5 Hz (6.3) | 71.8 Hz (11.7, P(0) = 9e-6) | 0 |
| K0 decays with 2 such tracks | 11.4 Hz | 3.1 Hz | 6.8 Hz | |

Expected FairShip MC entries use 6.15 Hz per entry (7.69 Hz at 5e13 × 0.8). Deck: `deck_d3` (9 Nov, pending approval).

### Q² and W of the interactions that make the K_L (`bin/kl_q2w.py` → `results/shield_dis/kl_q2w*.{md,json}`, `kl_q2w_<label>.png`)

Generator level (K_L in the library final state, before the iron). Every library event weighted by how often the
chain used its grid energy and part (muon spectrum × σ), target share by Z/A σ_p vs (1 − Z/A) σ_n. Q² = M² + 2Mν − W².
Share of interactions / share of K_L:

| | Pythia 6 (default) | G4 + P6 crossover | A: Geant4 11.2 | C: Geant4 10.7 | P6 Q² > 1 | P8 DIS Q² > 1 |
|---|---|---|---|---|---|---|
| interactions / K_L made, per spill | 1.78e5 / 2.51e4 | 1.04e6 / 2.32e4 | 9.92e5 / 9.84e3 | 9.92e5 / 1.11e4 | 2.97e4 / 3.46e3 | 7.96e3 / 543 |
| K_L per interaction | 0.14 | 0.022 | 0.010 | 0.011 | 0.12 | 0.068 |
| Q² < 0.1 GeV² | 45 % / 49 % | 92 % / 56 % | 96 % / 84 % | 96 % / 84 % | – | – |
| Q² 0.1 – 1 | 38 % / 37 % | 6 % / 32 % | 4 % / 14 % | 4 % / 14 % | – | – |
| Q² 1 – 10 | 16 % / 13 % | 1.4 % / 11 % | 0.3 % / 2 % | 0.3 % / 2 % | 96 % / 94 % | 97 % / 93 % |
| Q² > 10 | 0.7 % / 0.8 % | 0.1 % / 0.8 % | 0 % / 0.1 % | 0 % / 0.2 % | 4 % / 6 % | 3 % / 7 % |
| W < 2 GeV | 0 / 0 | 68 % / 1 % | 72 % / 3 % | 72 % / 3 % | 0 / 0 | 0 / 0 |
| W 2 – 3 | 14 % / 3 % | 13 % / 3 % | 14 % / 8 % | 14 % / 7 % | 23 % / 5 % | 26 % / 8 % |
| W 3 – 5 | 29 % / 17 % | 9 % / 10 % | 9 % / 19 % | 9 % / 17 % | 32 % / 22 % | 37 % / 32 % |
| W 5 – 10 | 49 % / 63 % | 8 % / 67 % | 5 % / 55 % | 5 % / 59 % | 36 % / 53 % | 34 % / 53 % |
| W > 10 | 9 % / 18 % | 1.5 % / 20 % | 0.6 % / 15 % | 0.6 % / 14 % | 9 % / 19 % | 3 % / 8 % |

- **In the default (Pythia 6) chain 86 % of the K_L come from Q² < 1** (quasi-real photons), only 14 % from hard DIS
  (Q² > 1); 98 % for Geant4. The K_L come from the high-W interactions (W 5 – 10 GeV: 55 – 67 %); W < 2 GeV (70 % of
  the Geant4 interactions) makes almost none.
- Same Q², W region, different K_L yield: Pythia 6 makes ~14× more K_L per soft interaction than Geant4. That is the
  generator difference; no measurement here says which is right.
- The Pythia 8 test therefore checks only the hard 14 %. On that part **Pythia 8 gives ÷ 6 fewer K_L than Pythia 6**
  (K_L decays in the vessel 827 → 139 per spill, K0-decay tracks at SST1 47 → 7.9 Hz): ÷ 3.7 from the cross section,
  ÷ 1.7 from the K_L per interaction (0.12 → 0.068, Pythia 8 makes fewer W > 10 GeV events). Same Q², W shape of the K_L
  otherwise. The same fraction of the K_L made gets out of the iron (0.51 vs 0.53), so the transport is consistent.
  If the soft part behaved the same way the default would drop by a similar factor, but Pythia 8.230 cannot generate it.
- Not yet known: the Q² / W of the interactions whose K_L survive the selection (DOCA, fiducial, IP). The chain output
  does not keep the library event (ν, W); the selection keeps 3 – 7 MC events. Needs `int_nu`, `int_W` stored and
  K_L-enriched draws (proposed, not run).

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
