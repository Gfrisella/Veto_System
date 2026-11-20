# Cavern-rock combinatorial: why mu x rock is ~90x mu x mu (20 Nov 2026)

Question: per 15 y, ±200 ps, before tagging, the slides show mu x mu 35.8 (pair cuts) / 2.36 (IP < 10 cm) but
cavern rock x mu 634 / 211. Is the rock term real, overestimated or a bug?

**Short answer: there is no normalisation bug. The 211 is what the model really gives, but the model overestimates it.**
The value comes from one kinematic configuration: a hard μ+ crossed near the vessel exit by a soft negative hadron from
the −x cavern wall. The model sends that hadron in a straight line through the vessel wall with no material. About 1 λ_I
and more than 2 X0 of SBT shell along its grazing path would remove about 65 % of these tracks. The realistic value is
about 75 / 15 y before tagging (±50 %, see below). With the SBT (R2), the term is tagged at 99.99 %.

## Files
- `rock_study.py`: builds the rock tracks and the muon tracks, runs the step-by-step pair cut flow (same cuts as
  `benchmark_designs._pair_cut` / `rock_combinatorial.cross_pairs`) and the measured IP. Writes `cutflow.json` and `data.npz`.
  It ran with `--mu-sample 1000000` (452k good muon rows) and `--mumu-sample 150000`; log in `rock_study.log`.
- `plots.py`: makes the plots and `summary.json`.
- Plots: `p_sst1.png`, `species.png`, `direction.png`, `origin.png`, `cutflow.png`, `ip_mechanism.png`, `ip_kick.png`,
  `pairs_geometry.png`.

Check that the run reproduces the official numbers: mu x rock = 648 (pair cuts) / 215 (IP < 10 cm) per 15 y. The
official values are 634 / 211.

## 1. What the rock sample is
- File `apply_D6_straw_mod_draws.npz`, by Shah Rukh, from `bin/rock_apply.py` on 8 Nov 2026.
- Shield muons from the Nov-4 shell seeds: 300 seeds, one seed = one spill of 4e13 p.o.t.
- Their straight line from the 30.6 m state goes into the −x wall (x = −4.56 m) or the floor (y = −3.36 m).
- The model has no +x wall.
- Muon-nuclear interactions in the top 3 m of G4_CONCRETE come from a library: Geant4 FTFP_BERT plus Pythia 6.
- Rate: 2.87e7 interactions / spill, 99 % of them in the wall, from about 1.46e9 μ+ / spill entering the wall.
- Products leave as straight lines from the rock surface. No material and no field are applied before SST1.
- 20 000 draws per seed. Weight per draw = S/M per seed. `build()` divides by `seeds = 300`, which gives the weight per
  spill (median 4.57 / spill per good track).
- Draw ids are unique, and the file holds 300 distinct seeds. There is no double counting.

| per spill (4e13 PoT) | rate |
|---|---|
| charged rock particles, downstream, in SST1 aperture (any p) | 6.4e5 |
| **good rock tracks** (p > 1 GeV, SST1 + timing acceptance) | **1.84e4** (3 887 MC rows, 3 823 draws) |
| good shield muons (same definition) | 8.5e3 |

- Good rock tracks by species: π± 48 %, p/p̄ 42 %, K± 4 %, e± 5 %, μ 1.6 %.
- Charge: 72 % positive, 28 % negative.
- Origin: 99.8 % from the −x wall, rock exit at z ≈ 55–75 m.

## 2. Momentum and direction at SST1 (`p_sst1.png`, `direction.png`, `origin.png`)

| | p 10 % / 50 % / 90 % [GeV] |
|---|---|
| rock good tracks | 1.1 / **1.7** / 3.5 (65 % below 2 GeV, 87 % below 3 GeV) |
| rock leg of the selected pairs (IP < 10 cm) | 1.2 / **1.9** / 2.9 |
| shield muons, good | 1.6 / 81 / 253 |
| muon leg of the selected pairs | 155 / **229** / 269 |

Direction and entry point of the good rock tracks:
- Polar angle at SST1: median 0.22 rad for rock tracks, against 0.024 rad for muons.
- 99.9 % have px > 0, because they all come from the −x wall.
- 85 % enter the vessel through the −x side wall, at z = 70–81 m, at a grazing angle of 0.1–0.35 rad (median about 0.2).
- The other 15 % pass behind the vessel exit and cannot form a vertex inside the vessel.

## 3. Cut flow (`cutflow.png`, `cutflow.json`; per 15 y, ±200 ps; factor 2·200 ps/1 s·1.5e7 = 6e-3 per spill²)

| step | μ×μ (own 150k subsample) | μ×μ official | μ×rock | ratio |
|---|---|---|---|---|
| opposite-charge pairs | 1.15e5 | | 4.45e5 | 3.9 |
| DOCA < 2 cm (anywhere) | 6.1e3 | | 3.4e3 | 0.55 |
| + vertex z in the vessel | 48 | | 883 | 18 |
| + vertex inside the frustum | 48 | | 788 | 16 |
| + true IP < 3 m (= candidates) | 47 | **35.8** | **648** (634) | 14–18 |
| measured IP < 2.5 m | 45 | | 639 | 14 |
| **measured IP < 10 cm** | 0.46* | **2.36** | **215** (211) | **≈ 90** |

\*The IP < 10 cm tail of the μ×μ subsample is statistically poor. Use the official 2.36.

In the μ×μ rows, DOCA < 2 cm alone is not meaningful. Two muon lines from the target always meet upstream, near the
shield. The step that matters is DOCA < 2 cm together with the vertex inside the vessel.

The factor of about 90 breaks down as follows:
1. **About 4×, rate:** 1.84e4 rock tracks × 8.5e3 muons, against 8.5e3² / 2 unordered μ×μ pairs.
2. **About 4.5×, geometry (DOCA + vertex in the vessel):** the pass fraction per opposite pair is 2.0e-3 for rock and
   4.2e-4 for μ×μ.
   - The rock lines cross the muon lines at a large angle: median opening angle 0.24 rad, against 0.05 rad for μ×μ.
   - So the point of closest approach falls inside the vessel, at the downstream end (vertex z median 78.9 m).
   - Nearly parallel muon lines instead meet upstream, outside the vessel.
3. **About 5×, IP:** 33 % of the μ×rock candidates pass IP < 10 cm, against 6.6 % for μ×μ (`ip_mechanism.png`,
   `ip_kick.png`). This step is a kinematic coincidence, not a statistical fluke:
   - Hard μ+ (about 230 GeV) at SST1 point back to x ≈ +0.22 m at the target plane (residual shield kick). μ− point to
     x ≈ −0.22 m.
   - The rock partner must be negative: 99.9 % of the selected pairs are μ+ × (π−, K−, e−, p̄).
   - The partner carries p_x ≈ +0.45 GeV, which is 0.9 % of p_μ.
   - This tilts the summed momentum by about 2 mrad. Over the 81 m lever arm, the line moves by about −0.17 m in x, onto
     the target.
   - 99 % of the selected pairs have a muon leg with IP between 10 and 30 cm. The rock track only shifts it below 10 cm.
   - Half of all good muons have IP < 30 cm, so there are many muons available for this.

## 4. Normalisation and double counting
- Weights per spill are correct: one seed = one spill of 4e13 (shell summary); w = S/M per seed, divided by 300 seeds.
- The good-track rate (1.84e4) matches the earlier rock_select / STATUS numbers.
- No seeds or draws are counted twice.
- **No overlap with other samples:**
  - The Nov-4 store contains muons only.
  - `rock_apply` excludes the parent muon.
  - The vessel and MS7 interaction samples are interactions in the helium, the walls or the shield iron, not in the rock.
- **Plausibility of the rate:**
  - 2.9e7 interactions / spill for 1.5e9 wall muons is about 2 % per muon over up to 3 m of grazing path. This is
    reasonable for photonuclear interactions.
  - About 6e-4 of these interactions give a charged track with p > 1 GeV that reaches SST1. This is plausible for hadrons
    that escape only from the top ~30 cm of rock.
  - The rate rests on the quick library estimate. The direct forced-Geant4 cross-check planned on 8 Nov has not been
    done.
- **MC statistics:** 602 distinct rock rows enter the selected pairs (n_eff 145), so the statistical error is about ±8 %.
  The 211 is not a few large weights.

## 5. Conclusion: real, overestimated, or a bug?
- **Not a bug:** the normalisation and the pairing are right, and the number is reproduced.
- **Overestimated: the realistic value is about 75 / 15 y, roughly 3× below 211.**
  - The model takes the rock track straight from the rock to SST1 with no material.
  - In SHiP, the track must cross the vessel wall, which is also the SBT shell: 0.8 + 0.8 cm Al plus 20 cm LS plus ribs,
    about 23 g/cm² at normal incidence.
  - At the grazing angles of these tracks (about 0.2 rad), that is about 105 g/cm²: about 1 λ_I and more than 2.5 X0.
  - The weighted probability that the selected rock legs cross it without a nuclear interaction is 0.35. Electrons are
    counted as lost.
  - This gives about 75 / 15 y for the selected pairs.
  - In R1, with no SBT, the physical vessel wall still has to be crossed. How much it removes depends on how R1 models
    that wall.
- **Missing contribution (the other way):** the model has no +x wall, although about 1.4e9 μ− / spill fly toward +x.
  The mirror configuration (hard μ− with x ≈ −0.22 m, crossed by a positive rock hadron with p_x < 0) is absent. With a
  +x wall at a similar distance, the term could roughly double.
- **Net:** of order 50–150 / 15 y before tagging, against 2.4 for μ×μ.
  - The term is real physics: soft, large-angle tracks entering through the side wall in the last 10 m of the vessel.
  - These tracks cross the SBT at grazing incidence (long LS path, large deposit), so the SBT tags them: R2 leaves
    0.004 / 15 y.
  - Without side coverage (R1), this term dominates.
- **Suggested next steps:**
  1. Transport the rock products through the vessel or SBT shell, with the existing `sbt_material` / g4vessel route.
  2. Add the +x wall.
  3. Cross-check the library rate against direct Geant4 for a few seeds.

## Comparison with FairShip (follow-up, 20 Nov 2026)

Claim tested: in FairShip (full Geant4), far fewer cavern-made tracks reach SST1, because they keep almost no momentum
after the interaction in the concrete. The question is whether our library skips the concrete transport.

### (1) The library does transport the products through the concrete with Geant4
The products are transported from the vertex to the rock surface by Geant4 FTFP_BERT, with energy loss, re-interactions
and secondaries. What is stored is the particle as it crosses the surface.
- `cpp/g4rock.cc`, header:
  > "its first step (~1e-8 m) is a muon-nuclear interaction; right after it the factor is set back to 1, and the scattered
  > muon and every product are transported with full physics (FTFP_BERT; showers, re-interactions, decays). Recording:
  > every particle crossing the surface from the rock into the air is recorded and killed there"
- The stepping action records exits only at the rock → air boundary, moving into the air:
  ```
  if (step->GetPreStepPoint()->GetPhysicalVolume() == gRock && post->GetStepStatus() == fGeomBoundary &&
      post->GetPhysicalVolume() != gRock && post->GetMomentumDirection().y() > 0) { ... record x, p at the surface ...; fStopAndKill }
  ```
- The Pythia part (comp 1, ν ≥ ν_x) goes through the same chain. `bin/rock_library.py`:
  > "injected at the vertex and transported by g4rock (Geant4 FTFP_BERT)" (`g4rock.inject(...)`)

  Only products whose Pythia vertex is already in the air are moved straight to the exits.
- `bin/rock_apply.py` only rotates and places those surface exits. Then it draws straight lines from the surface into the
  cavern. Only this last step has no material: cavern air, and the vessel / SBT wall (section 5).

### (2) Depth and momentum behind the 1.84e4 good tracks / spill
I matched each good track to its library exit by pdg and |p| (`depth_match.py`, `depth_match.json`, `depth.png`). This
matches 98.5 % of the weight.

**Depth:**
- Median 5.3 cm (10 % / 90 %: 1.2 / 19 cm).
- 74 % come from less than 10 cm, 97 % from less than 30 cm, and none from deeper than 0.75 m.
- The parent muons graze the wall: α ≈ 0.06 rad, E_μ 26 / 40 / 60 GeV (10 / 50 / 90 %).
- 55 % of the good tracks come from the Pythia high-ν part.

**Momentum vs depth:** flat. The median at the surface is 1.6–1.8 GeV in every depth bin up to 30 cm.

**What a naive no-transport picture would give, for comparison:**
- Path in the rock = depth / sin(exit angle), median 0.27 m. With λ_I = 0.40 m, about 49 % would survive.
- These numbers are already the Geant4-transported exits. The absorption between the vertex and the surface is
  included, and it is small because the interactions are shallow.
- So the missing concrete transport does **not** explain a large deficit.

### (3) Numbers against the FairShip full-Geant4 cross-check
Source: `FairShip_TRY2026_EM_SST1/origin/origin_breakdown_td.txt`, after the TD hit and p > 1 GeV; 5e13 → 4e13 is ×0.8.

| good non-μ tracks at SST1 from cavern concrete | FairShip (per spill, 4e13) | ours (per spill) | ours / FairShip |
|---|---|---|---|
| π± | 0.200 kHz × 0.8 = **160** (26 MC) | 0.48 × 1.84e4 = **8.8e3** | **≈ 55** |
| p | 0.611 kHz × 0.8 = **490** (38 MC, large errors) | 0.42 × 1.84e4 = **7.7e3** | **≈ 16** |
| all (π + p; e and K are small in FairShip) | **≈ 650** | **1.84e4** | **≈ 28** |

**Explained:**
- FairShip has the SBT and vessel walls between the cavern and SST1; our straight lines do not. Section 5 gives a
  weighted survival of about 0.4 for the good rock tracks.
- This explains a factor of about 2.5. A factor of about 10 remains (about 20 for pions), and the concrete transport is
  not the reason.

**Remaining candidates, which I have not checked:**
- **(a) The muon flux into the concrete.** Ours is 1.46e9 μ+ / spill into a flat −x wall at x = −4.56 m. It is a
  straight-line extrapolation of the 30.6 m states, with no other cavern structures. FairShip's cavern geometry and wall
  distance may differ.
- **(b) The final-state model at high ν.** 55 % of our good tracks come from the Pythia 6 DIS part, which is only 6 % of
  the interactions. FairShip uses Geant4 muonNuclear throughout.
- **(c) FairShip statistics.** π has 26 MC entries and p has 38, with heavy weights.

**Suggested next check:** compare the FairShip muon flux crossing the concrete surface (z 50–80 m, by side) with our
1.46e9 / spill. Then run g4rock with Geant4-only physics (comp 0 for all ν) on the same muons.
