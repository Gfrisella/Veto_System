# Status (handoff between Claude sessions)

Read this first in a new session, then `ISSUES.md` for the open problems. Update it at the end of
every working session: the date, what is running, what finished, and what is waiting on Guglielmo.

_Last update: 2026-10-20 09:40_

## Running jobs
None. Check with `squeue -u gfrise`. The tmux session `veto` on farm-ui1 has a `jobs` window for
monitoring.

## Done (2026-10-20): pair categories (no timing, full 1 s spill)
`results/scratch/pair_categories/` (`pair_categories.pdf`, `pairflow.py`/`.json`, `surv.py` = job 1871264, `plot.py`).
- 1.8e7 opposite-charge pairs per spill (pairable muons) -> 7567 /spill pass DOCA/vessel/IP (196 736 MC pairs):
  straight+straight 98.68 %, wall bounce+straight 0.71 % (2 350), floor bounce+straight 0.61 % (513);
  no bounce+bounce pair passes (5.5e3 /spill before the cuts).
- Pairable bouncing muons (p > 1 GeV, timing acceptance): 361 Hz (wall 326, floor 36) out of 10.3 kHz bouncing.
- Survivors: OPT1/2/3 0; D6 2 (1 floor+straight); W1 35 (34 straight); U1 tile 6 (5 with a bounce), tile2 7 (6);
  D0 170 753 (1.5 % with a bounce).

## Done (2026-10-19 evening): fins at tilt 0, and side walls fully covered
Validation: 3 M muons per type, weighted, seed 2, no timing detector (5 ns, ±5σ, same spill), as `bottom_main`.

- **Tilt-0 search** `results/optimize_tilt0/` (job 1866124, 30 generations, 1827 designs; `run.sh`),
  `top_designs/rank1..5.json` (OPT3_tilt0_*), validation `results/validate/tilt0_main.json` (with OPT2 again: identical numbers).
  Ranks 3-5 are the same design.
- **Side walls** `designs/W1_side_walls_straw.json`: both lateral walls fully covered with straws, minimal UBT, no fins, no
  bottom panel; validation `results/validate/walls_full.json` (job 1866186). New `wall` plane kind, see CHANGELOG.

  | design | missed SST1 /spill (wall b., floor b.) | fake /spill | fake 15 y | signal lost | cost |
  |---|---|---|---|---|---|
  | OPT2 (tilted) | 737 (63, 657) | 2.2e-3 | 2.7e4 | 4.9 % | 7.64 MCHF |
  | OPT3 tilt0 rank1 | 747 (122, 607) | 3.0e-3 | 3.6e4 | 3.1 % | 7.53 MCHF |
  | OPT3 tilt0 rank2 | 736 (121, 595) | 3.3e-3 | 3.9e4 | 2.9 % | 6.99 MCHF |
  | W1 side walls | 2729 (1490, 1131) | 7.8e-2 | 9.3e5 | 0.01 % | 15.76 MCHF |

- Tilt 0 costs nothing in missed muons; it doubles the missed wall bounces (63 -> 121) and gains on floor bounces and
  signal loss. Fake vertices +35-50 % (within ~1.5σ). The side walls miss the floor bounces (enter through the vessel bottom)
  and the wall bounces that reach SST1 through the gap z = 80.6-82 m beside the vessel.

## Done (2026-10-16): bottom panel, no timing detector
- **Optimisation**: `results/optimize_bottom/`, 30 generations and 1715 designs; `top_designs/rank1..5.json`.
  - Best design: OPT2 = `rank1` (name `OPT2_rank1_g29_10`).
  - The search was still improving at generation 30.
- **Validation** (3 M muons per type, weighted, seed 2): `results/validate/<tag>.json`.

  | tag | designs | settings |
  |---|---|---|
  | `bottom_main` | D6_straw, OPT1, OPT2 rank1-5 | no timing detector, track time 5 ns, ±5σ, pairs = same spill |
  | `bottom_sigma1` / `bottom_sigma20` | OPT2 | track time 1 / 20 ns |
  | `bottom_ubt9999` | OPT2 | UBT 99.99 % |
  | `bottom_with_td` | OPT2, OPT1 | 0.1 ns, ±3σ, tiles ±1.5 ns, ±200 ps pairs |

- **OPT2 headline**:
  - missed SST1 muons 0.74 kHz (OPT1 1.13, D6 2.41);
  - fake vertices 2.2e-3 per spill, 2.7e4 in 15 y (with the timing detector: 2e-5);
  - signal lost 4.9 %; cost 7.6 MCHF.
- **Variations** (selected muons missed, of 3 M):

  | case | missed |
  |---|---|
  | baseline | 559 |
  | 1 ns | 582 |
  | 20 ns | 579 |
  | UBT 99.99 % | 281 |
  | timing detector | 888 (not understood: ISSUES A6/B6) |

- **Event browser**:
  - published page: https://claude.ai/artifact/6Q5YG15y5zh8GKQLfveKWV;
  - rebuild or serve it with `bin/event_browser.sh`;
  - skill: `../.claude/skills/veto-event-browser`.

## Waiting on Guglielmo
- **HTML deck "Combinatorial Muons and the Veto"** (19 Oct, first draft, 15 slides, interactive):
  - https://claude.ai/artifact/Qz2QdqLCr6TW4VW9mp4RW4
  - 20 Oct: standalone copies in `results/presentation/`: `combinatorial_muons.pdf` (9 static pages),
    `combinatorial_muons_offline.html` (interactive, Plotly inlined, works offline), `png/`, and the sources in `src/`.
  - sources in the session scratchpad `.../33d0489f-.../scratchpad/deck/` (`extract.py` -> `deck_data.json`,
    `template.html`, `build.py`); copy to `results/presentation/` once approved.
  - after review (19 Oct): 9 slides, results removed (to be shown another time); the removed result
    slides are in `results_slides_removed.html` in the same folder.
  - 20 Oct: the example layout on the design-space slide is now OPT3_tilt0_rank1 (was OPT1); its geometry was
    added to `deck_data.json` (`deck_data_before_tilt0.json` = backup).
  - 20 Oct: non-exclusive class "never enters the vessel" (path 30.6 m -> bounce -> SST1 never inside the frustum;
    implies missing the UBT = entrance window), `extract_outside.py` (exact segment-frustum test): 1.49 kHz = 4.9 %
    of SST1; wall bounce 1.39 (16 % of them), floor bounce 0.10, straight ~0. Overlay on the 3D slide, row on the
    categories slide.
  - 20 Oct: outlook slide: ISSUES tags removed, detector-model box lists the detectors used, floor-detector bullet
    added under Goals. Template backups `template_v13_backup.html` / `template_v14_backup.html`.
  - 20 Oct: new slide 6 "A huge UBT (4 x 6 m^2) is not enough" (bridge to the optimisation) from
    `results/validate/ubt4x6.json` (see CHANGELOG 2026-10-20); costs relative to 8 MCHF. Fake-vertex row also with a ±35 ns
    coincidence (`ubt4x6_t5ns.json`: 0.07 in 15 y for 4 x 6). Open: asked whether slide 3 should show the coincidence case too.
- The old `veto_v2.pdf` beamer draft is gone (its scratchpad was cleared).
- **Slide deck draft** (not yet approved; the plot-centred style of the new_2296 deck):
  - `/tmp/claude-21615/-disk-users-gfrise/e4c322df-ac14-4a48-9288-ff2163afe97c/scratchpad/slides/veto_v2.pdf`;
  - its sources are in the same folder (`veto_v2.tex`, `make_plots2.py`, `make_tables.py`);
  - the older text-heavy version, `veto_bottom.pdf`, was rejected;
  - once approved, copy the deck to `results/presentation/`;
  - the scratchpad is temporary: if it is gone, rebuild the deck with the two scripts and the
    `cern-ship-slides` skill.
- **Nothing is committed** (branch `veto-all-sst1`); commit only when asked.

## Possible next steps (only if asked)
- A longer search (more generations), seeded from `top_designs`.
- The UBT as in SR-370 (1.3 m, horizontal straws, 99.99 %) and the SBT walls in the model (ISSUES B5, B6).
- Understand why the timing-detector settings miss 60 % more selected muons.
- Cross-check against the P-369 selection (Table 3) and normalisation (ISSUES B7, C7).
