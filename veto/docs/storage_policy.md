# Storage policy of the muon production: where the statistics go, and a better rule

Study of 2026-10-21 (`bin/storage_policy_study.py`, `results/storage_policy/`), on dataset delta (enriched input,
484 seeds). The common types are identical in epsilon (6,000 seeds), so the conclusions hold for both.

## The problem
`bin/generate_muon_study.py` (`storage_plan`, `select`) keeps per fate type a uniform random fraction of the muons
of the first n_t seeds, n_t = budget / RATES_t (15 M rows):
- the rare types (SST1 bounced / selected / other, veto only) completely, in every seed;
- the common types (rock bounced, straight past, stops between, stops in the shield) only in the first 20 - 27 seeds,
  ~0.3 % of their muons.

So the storage goes where the muons are, not where the questions are. Independent input muons per 25 cm cell of
the probe planes (`bin/probe_statistics.py`: fins every 2.5 m on both sides, floor planes, the UBT column):

| band | median | 10th percentile | cells below 30 |
|---|---|---|---|
| fin, 0 - 0.5 m from the vessel wall | 5 | 1.3 | 78 % |
| fin, 0.5 - 1 m | 9 | 1.8 | 71 % |
| fin, 1 - 2 m | 153 | 23 | 14 % |
| fin, beyond 2 m | 323 | 50 | 6 % |
| floor planes | 5 | 1.2 | 79 % |
| UBT column (full height) | 148 | 3.7 | 25 % |

More seeds of the same production do not help these cells: their muons are common types, kept by the first seeds
only (epsilon has exactly delta's rows of them).

## Policies compared (same storage: 57.7 M rows of the common types, spread over 6,000 seeds)
Every stored row stands for its population: input weight v = weight / scale, scale = muons per seed it represents.
A rule "keep a muon of a seed with probability p" is emulated exactly in expectation (Horvitz-Thompson: kept rows
get weight v / (S p), unbiased for any p > 0). The rare types stay complete in all policies.

| policy | fin 0 - 0.5 m | fin 0.5 - 1 m | fin 1 - 2 m | fin > 2 m | floor | UBT |
|---|---|---|---|---|---|---|
| now (measured) | 5 / 1.3 | 9 / 1.8 | 153 / 23 | 323 / 50 | 5 / 1.2 | 148 / 3.7 |
| uniform (today's rule, all seeds) | 5 / 1.3 | 9 / 1.8 | 153 / 23 | 323 / 50 | 5 / 1.2 | 148 / 4.0 |
| weight (p proportional to v) | 1.5 / 0.1 | 189 / 1.8 | 8.3 k / 0.9 k | 11 k / 1.0 k | 11 / 0.07 | 2.1 k / 18 |
| cell (each cell asks for T rows) | 4.4 k / 0.9 k | 1.8 k / 0.4 k | 1.4 k / 0.6 k | 1.0 k / 0.6 k | 1.6 k / 0.4 k | 1.1 k / 0.8 k |
| **cell + weight** | **15 k / 8.4 k** | **16 k / 11 k** | **21 k / 11 k** | **15 k / 11 k** | **12 k / 1.2 k** | **12 k / 11 k** |

(median / 10th percentile of the effective number of rows per cell.) "uniform" reproduces "now": the emulation is
right. Spreading the same rule over more seeds changes nothing; weighting alone starves the quiet cells; only a
rule that knows **where a muon crosses** fixes them, and weighting within a cell then equalises the final weights.

### The limit: the same input muons again
Every seed replays the same 601 M input muons, so a cell can only have as many independent muons as distinct input
muons reach it. Measured on the types kept completely (`results/storage_policy/repetition.py`, 60 -> 484 seeds):

| band | distinct (cell, input muon) pairs | growth with seeds |
|---|---|---|
| fin 0 - 0.5 m | 9.9 M -> 27.5 M | ~ seeds^0.5 |
| fin 0.5 - 1 m | 0.21 M -> 1.70 M | ~ seeds^1 (no repetition) |
| floor | 44 k -> 220 k | ~ seeds^0.8 |

So the table above is an upper bound where muons repeat (right at the wall, the floor): there the gain is smaller,
but new input muons keep arriving with more seeds, and from ~5 there is a factor 100 - 1000 of room.

## Recommendation for the next production
1. **Store by where a muon crosses, not by its fate.** At production time intersect every muon with the probe
   planes (A -> bounce -> B, straight lines, cheap on the GPU), and keep it with
   `p = min(1, T v max_c 1 / (S V_c))`, V_c = muons per spill crossing cell c (from a pilot seed, like `RATES`
   now), weight v / (S p). Muons crossing nothing (stops in the shield, far away) with a small base rate.
   The rare types stay complete.
2. **Rate maps as histograms of every muon**, accumulated on the GPU in every seed: the common types enter the
   reconstruction only through the rate maps (random hits, rate limits), so the maps get the statistics of all
   seeds x 601 M muons at ~zero storage; the stored rows are then needed only for the tracks and checks.
3. **Drop the rows of type 7** (stops in the shield, 15 M rows, cross nothing after 30.6 m): keep its counts and
   weights per seed and ~1 M rows.
4. **`RATES` from a pilot seed of the actual input**, not the old table (with the enriched input some classes
   overshoot their budget: straight past 26 M rows instead of 15 M).
5. Where the same input muons repeat (right at the wall, the floor): a targeted source production (more input muons
   in the directions that reach there) rather than more seeds.

## Test: the alpha storage (implemented 2026-10-21)
`bin/generate_muon_study.py --storage alpha` (experimental; the default is still the budget per type): rare types
complete; types 4 - 7 each muon with p = max(500 / N_type, p_cell), p_cell = min(1, lam w max_c 1 / V_c) over the probe
cells it crosses (`veto.probe`, cell rates V_c from the seed itself), lam for 10,000 such rows per seed; weights
w / p, then per type to the seed's exact total; plus the probe-cell histogram (counts, sum w, sum w^2) of every
muon of types 0 - 6. ~91 k rows and ~195 s per seed (113 s with the default storage).

48 seeds on the enriched input (`data/sim/veto_muons_alpha_v0`, dataset
`design-optimization-dataset-alpha-storage-v0-48seeds-2026-10-21`), probed like delta and epsilon (all types):

| dataset | fin 0 - 0.5 m | fin 0.5 - 1 m | fin 1 - 2 m | fin > 2 m | floor | UBT |
|---|---|---|---|---|---|---|
| delta (484 seeds, 111 M rows) | 27 / 2.4 / 32 % | 13 / 2.1 / 44 % | 153 / 23 / 3 % | 323 / 50 / 1 % | 13 / 2.4 / 40 % | 181 / 9 / 11 % |
| epsilon (6000 seeds, 552 M rows) | 42 / 2.6 / 30 % | 14 / 2.1 / 44 % | 153 / 23 / 3 % | 323 / 50 / 1 % | 16 / 2.5 / 38 % | 207 / 8.8 / 11 % |
| **alpha storage (48 seeds, 4.4 M rows)** | 32 / **10 / 10 %** | **129 / 33 / 0 %** | **182 / 98 / 0 %** | 133 / **90 / 0 %** | **50 / 7.3 / 16 %** | 92 / **16 / 0 %** |

(median / 10th percentile of independent input muons per 25 cm cell / share of cells below 10.) With 1/126 of
epsilon's rows the weakest cells are 3 - 15 times better everywhere; medians are lower only where epsilon spends
hundreds of millions of rows (hot lobe, rare types over 6000 seeds), which are fine in every dataset. From 36 to
48 seeds the 10th percentiles grew 1.22 - 1.34 x (seeds: 1.33 x): about linear so far, also at the wall and on the
floor. The rates per type agree with epsilon within ~1 %.

Next: the rate maps from the histograms against epsilon's; ~500 - 1000 alpha seeds (~45 - 90 M rows) as the next
design dataset; RATES from a pilot seed for the default storage.

## The 800-seed production (2026-10-22)
Seeds 0 - 799 with `--storage alpha` (enriched input, farm-gpu3, 14 h on 3 GPUs), dataset
`design-optimization-dataset-alpha-storage-v0-800seeds-2026-10-22`: 73,005,004 rows (delta: 111 M), rates per type as
epsilon within 0.3 %.

| dataset | fin 0 - 0.5 m | fin 0.5 - 1 m | fin 1 - 2 m | fin > 2 m | floor | UBT |
|---|---|---|---|---|---|---|
| epsilon (6000 seeds, 552 M rows) | 42 / 2.6 / 30 % | 14 / 2.1 / 44 % | 153 / 23 / 3 % | 323 / 50 / 1 % | 16 / 2.5 / 38 % | 207 / 8.8 / 11 % |
| **new storage (800 seeds, 73 M rows)** | **343 / 110 / 1 %** | **2050 / 170 / 0 %** | **2970 / 1600 / 0 %** | **2200 / 1500 / 0 %** | **419 / 69 / 0 %** | **1540 / 200 / 0 %** |

(median / 10th percentile of independent input muons per 25 cm cell / share of cells below 10.) The weakest cells
are 23 - 80 x better than epsilon's with 1/7.6 of its rows. For the studies: SST1 and veto-only types from epsilon
(6000 seeds), the common types and the rate maps (histograms) from this production.
