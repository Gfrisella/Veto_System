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
with the unbiased 2 093 +- 25; a deficit is the secondary-V0 part.

**Result: 1 524 per spill (K_L 1 443, K_S 50, Lambda 31) vs 2 093: 27 % of the V0 (~570 / spill, almost all K_L) are
secondary**, made by hadrons of V0-free final states during the Geant4 transport through the remaining iron. The V0-only
seed gave 2.6e6 forced decays (x36 an unbiased seed) but peaked at 153 GB (53 min, 16 workers).

**Fix: two complementary samples** (`--final-state v0` and `--final-state nov0`): the V0 final states (weight x P(V0|E))
and the V0-free ones (weight x P(no V0|E), their V0 come from the transport). Every interaction belongs to exactly one,
so v0 + nov0 = the unbiased chain (checked: P(V0) + P(no V0) = 1.0000 at 3 and 30 GeV, no V0 in the nov0 draws).

## Production (16 Nov 2026)

- `--final-state v0`: 80 jobs x 2.5e5 interactions (seeds 8001-8080), `--v0-copies 20 --v0-acc-cut` (~x200 forced decays
  of primary V0 vs the unbiased S3 production).
- `--final-state nov0`: 80 jobs x 5e5 interactions (seeds 9001-9080), `--v0-copies 100 --v0-acc-cut` (~x40 for the secondary V0).
- Split into small jobs (64 GB) because the farm is memory-bound (big nodes: idle cores, 1-10 GB free); first submission
  (40 x 200 GB) never started. Throttle 24 at a time, 8 CPUs / 8 workers each (16 workers needed 82 GB at Geant4 start-up, ~5 GB per worker: first array OOM-killed), BLAS/OpenMP threads capped by env.sh (farm etiquette,
  Roland Bernet's mail of 12 Nov).
- Partitions standard + hep (16 Nov evening: standard full, hep idle; peak memory per task 42 GB, 12.5 min per v0 task).
- Jobs: production 2089677 (array 0-159), event stores 2089678 (`results_rerun/evstore/ms7_muon_fsv0`,
  `ms7_muon_fsnov0`, built separately: each store averages over its own seeds; downstream the two are SUMMED), MS7 cache
  2089679 (`prepared_ms7_v0stat`: fsv0 + fsnov0 + ms7_nu2026).
- Use: `bin/prepare_interactions.py --ms7 <fsv0> <fsnov0> <nu>`; evaluations over both muon stores (sum). The old store
  `ms7_muon_v0s` (unbiased, 10 seeds) stays as the cross-check.

## Results (16 Nov 2026 evening, full selection IP < 10 cm, before tagging; prepared_ms7_v0stat)

| sample | selected MC events | per 15 y |
|---|---|---|
| old unbiased chain (ms7_muon_v0s, 10 seeds) | 2 | 6.1e4 (n_eff 1.5, heaviest 4.9e4) |
| fsv0 (V0 final states, 80 seeds) | 75 | 3.1e4 |
| fsnov0 (V0-free final states, V0 from the transport, 80 seeds) | 3 | 1.9e3 |
| **new total (fsv0 + fsnov0)** | **78** | **3.3e4** |

- Forced-decay copies (code 4, incl. the 121 MS7 neutrino events): n_eff 57, heaviest 790 / 15 y (was 4.9e4): the V0
  statistics problem is solved.
- BUT one original event (code 3) carries 1.82e4 / 15 y (55 % of the total, n_eff 1): a 6.3 GeV K- from the MS7
  interaction decaying in flight at (-1.01, 1.27, 65.46) m - a CHARGED-kaon decay in the vessel, not a V0. Legitimate (code 3
  keeps every non-V0 product; v0 + nov0 is complete), but the forced decays do not enrich charged kaons: this channel has
  one MC event. Needs its own enrichment (forced K+- / pi+- decays in the vessel) before the optimiser can score it.
- R1 / R2 / R3 after tagging: pending (waits for the tile time-window decision, see STATUS).

## Forced K+- copies (16 Nov 2026, Guglielmo; bin/force_kaons.py, e825a26)

- Per spill: 9.3e5 muon interactions in the last 50 cm of MS7; 8.3e3 K+- leave MS7 (K+ 5.4e3, K- 2.8e3; 6.9e3 above
  1 GeV; 0.75 % of the interactions have one); 79 % of them decay within the vessel z range (6.5e3 / spill). 53 % of the
  K+- come from the 1.7 % of interactions with a V0 (associated strangeness).
- 100 copies per K+- of the original events: w_copy = w_orig x P(decay in the vessel) / 100 (verified to 1e-9 on 3000
  copies); the channel is SAMPLED with the branching ratios (no channel biasing: Guglielmo, the single-track K decays
  matter for the combinatorial); --acc-cut keeps copies with >= 2 charged products into SST1.
- RULE: the forced stores (ms7_muon_fsv0[k], ms7_muon_fsnov0[k]) are for the VERTEX background only. In them the originals'
  V0 and K+- never decay and single-product copies are dropped. The single-track combinatorial (D9, dis_combinatorial.py)
  keeps using the unforced store ms7_muon_v0s (natural decays, correct branching ratios).
