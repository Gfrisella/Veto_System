# R3: silicon strips in MS7 and the V0 rule (study of 16 Nov 2026)

Owner: Guglielmo (with Claude). Code on branch `veto-nov4-source`. Pages: R3 design
https://claude.ai/artifact/JCNdVa2LEaCzg4yudFRkfW, V0 event display https://claude.ai/artifact/5hYk9446RtEnLkFePxW8AY,
R2 coverage https://claude.ai/artifact/3ejSJY4v9pPTVg75rMUVQq.

## 1. The problem: MS7 V0

A muon (or neutrino) interacts in the last 50 cm of MS7 iron (z 29.39-29.89 m, return-yoke legs |x| 0.70-1.75 m); a
K_L / K_S / Lambda decays in the vessel into two tracks that pass the selection. Per spill ~2 100 V0 decay in the vessel
(3e10 per 15 y); the chain simulates 7e5 forced decays (5 copies per V0), so one simulated decay = ~4e4 real ones per 15 y.
- Selected events (MS7 muon chain): 399 with the partial selection (IP < 2.5 m), **2** with the full one (IP < 0.1 m),
  weights 4.9e4 and 1.2e4 per 15 y. Statistics, not a big probability: ~1e3 x more decays (or biased generation) needed.
- Interaction points of the selected events: |y| < 0.15 m (95 %), |x| 0.87-1.55 m, both legs ~50/50, z uniform in the iron.

## 2. Why R2 does not tag them (Guglielmo's question)

The products of the iron interaction ARE seen: in every selected event ~7 charged particles (pi, mu, p, K, e) cross the
4 x 4 m tile plane. But the association looks along each DAUGHTER's backward line, which points to the decay, not to the
production point (decay opening angle, missing neutrino of K_L -> pi l nu). On the two heavy full events: no daughter line
lands on a UBT plane / the nearest product is 0.33-0.61 m away vs a 0.155 m window; wide time window changes nothing.
In reality also time: a V0 flies slower than light, products arrive a median 0.35 ns early (6.5 ns for a 1.2 GeV K_L over
23 m) vs +-0.2 ns tiles. NOTE: the simulation sets each track's clock to 0 at SST1, so it ignores this delay.

## 3. The V0 rule (`reco.v0_tag`, design `"detector": {"v0_rule": true | {...}}`)

Candidate traced back as a neutral along the summed pair momentum to every plane upstream of the vertex; ANY hit within
r = r0 + theta dz (r0 5 cm, theta 25 mrad = 68 % of the MS7 K_L back-trace errors) counts, time window opened 3 ns early,
random hits drawn over the widened window (signal cost included). Options: min_hits (1 / 2), skip_tech (e.g. tile2).
Applied identically to pairs (run_reco), signal (run_reco), interactions (interactions.evaluate / veto).
First check (cached MS7 sample, full selection, R2): surviving events 43 -> 15; the 1.2e4 event is tagged, the 4.9e4 one
escapes (back-trace 41 mrad off: slow K_L, missing neutrino).

## 4. Silicon strips in MS7 (`lib/veto/shield_strips.py`, design `"shield_strips"`)

Stations size x size x 3 cm, two views of 187.5 um strips spanning the station (LHCb UT outer), 25 ns slots, 99 %,
1 MHz/strip limit, superimposed on the iron and field (not modelled). Layers at given z ("layers"); a layer upstream of
the interaction sees the incoming parent muon (MS7 parents, straight from 29.20 m), downstream the products (exit-face
states extrapolated back). Rule: candidate traced back (as the V0 rule), any hit within r0 + theta dz (2 cm + 10 mrad)
on a station ("point") or >= 2 ("vertex"). Random hits: rate map per layer (muons at that z + EM debris at the exit face:
the shower inside the iron is not simulated). Cost 0.4 straw-channel units / channel (assumed, C6): 20 cm station 1.26,
30 cm station 1.88 drachmas.

Exit-face study (`bin/strip_study.py`, results_rerun/reco/strip_study.json): back-trace error (68 / 90 %) MS7 mu V0 25 / 37
mrad, MS7 nu 14 / 21 mrad (partial), 2 / 10 (full). theta >= 20 mrad costs 2-40 % signal. 10 hottest 20 cm stations,
theta 10 mrad: MS7 nu tagged 74 % (full) / 36 % (partial), MS7 mu V0 23 % (partial), signal 0.2 / 0.9 %.

## 5. R3 layout (final, `designs/R3_sbt_ubt_strips.json`)

R2 + V0 rule + strips: 3 layers at 29.39 / 29.59 / 29.79 m (entrance, middle, 10 cm before the end), each 8 stations of
30 x 30 cm at x = +-0.9, +-1.2, y = +-0.15 (inner part of both legs, |x| 0.75-1.35, |y| < 0.3).
- Coverage of the selected V0 interaction points: 82 % (weighted). Station rates 4-34 MHz, strips <= 61 kHz.
- Last layer 29.79 m: 78 % of the V0 interactions are upstream of it (it sees their products).
- Cost: strips 45.2 drachmas (24 stations), R3 = 378.7 drachmas = 13.6 % over the R2 budget (333.5); 108 564 channels.
- History: 5 layers x 3 x 20 cm (18.8 dr) -> 3 layers x 12 x 30 cm (95 % coverage, 67.8 dr, outer stations 66-82 MHz) ->
  outer column removed, shifted inward (options 0.85/1.15: 79 %, 29 MHz; 0.9/1.2: 82 %, 34 MHz; 0.95/1.25: 83 %, 37 MHz;
  1.0/1.3: 85 %, 42 MHz) -> 0.9/1.2 chosen; last layer moved from 15 to 10 cm before the end.

## 6. Still to do

- Run the R1 / R2 / R3 test on the full sample (HANDOFF.md, command there).
- Kink rule (parent before / product after), the MS7 field in the stations, the EM shower inside the iron.
- Time of flight of the V0 in the simulation (per-track clocks) - affects the V0 rule in reality.
- MS7 V0 statistics (2 events at full selection) before the optimiser can score this source.
