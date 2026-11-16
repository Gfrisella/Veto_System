# MS7 V0 statistics increase (16 Nov 2026)

Guglielmo: "go for the V0 statistics increase and document it". Goal: lower the per-event weight of the MS7 V0
background (2 events at full selection carried 6.1e4 / 15 y) by ~1e3, so that the background and the optimiser's
objective do not rest on a handful of events.

## Where the weight comes from

Per spill ~2 093 V0 (K_L / K_S / Lambda) from muon interactions in the last 50 cm of MS7 decay in the vessel (10 unbiased
seeds, seed-to-seed rms 25), i.e. 3.1e10 in 15 years. The unbiased chain (S3, `--v0-copies 5`, 10 seeds x 2e6 forced
interactions) simulates 7.3e5 forced decays: one decay = ~4e4 real ones per 15 years. Most of the CPU is wasted: only
0.13 % (3 GeV) to 2.5 % (100 GeV) of the iron interactions have a V0 in their final state (library `mu_library_iron_v0stable`,
split geant4), and most forced decays can never pass the selection.

## Three levers (multiplicative)

1. **Importance sampling of the final state** (`MuonLibrary.sample(..., v0_only=True)`, `shield_dis.py generate --v0-only`):
   per interaction (energy E from the parent), draw only library final states that contain a K_L / K_S / Lambda; the
   component (Pythia / Geant4) and the target (p / n) in proportion to sigma x (their V0 fraction); weight x
   w_factor = P(V0 final state | E). Exact for V0 made in the primary interaction. Check: at 3 / 10 / 30 / 100 GeV the mean
   w_factor equals the unbiased V0 fraction (0.0013 / 0.0040 / 0.0102 / 0.0250 vs 0.0013 / 0.0040 / 0.0101 / 0.0252).
   Gain 40-770 per draw. Not included: V0 made later by hadrons of V0-free final states during the Geant4 transport through
   the remaining iron (secondary V0); every interaction carries `lib_v0`, so the unbiased production restricted to
   `lib_v0 == False` gives that part if needed.
2. **More forced decays per V0** (`--v0-copies`, 5 -> up to 100): the decay point and channel are resampled; weight / copies.
3. **Acceptance pre-cut at generation** (`--v0-acc-cut`): keep only copies with >= 2 charged decay products of p > 1 GeV
   heading into the SST1 aperture (+10 % margin); the others never pass the selection, so dropping them changes nothing
   but the store size. (The summary's v0_decays_in_vessel_per_spill then counts only the kept copies.)
4. More seeds.

## Validation (job 2088400)

One seed, 2e6 interactions, `--v0-only --v0-copies 5`, no pre-cut: the V0 decay rate in the vessel per spill must agree
with the unbiased 2 093 +- 25; a deficit is the secondary-V0 part. Result: see below.

## Production

See STATUS.md / HANDOFF.md for the job ids and the resulting store (`results_rerun/evstore/ms7_muon_v0only`), the rebuilt
cache `prepared_ms7_nov4full` and the new full / partial selected counts and weights.
