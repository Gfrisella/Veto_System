# Status (handoff between Claude sessions)

Read this first in a new session, then `ISSUES.md` for the open problems. Update it at the end of
every working session: the date, what is running, what finished, and what is waiting on Guglielmo.

_Last update: 2026-10-16 17:10_

## Running jobs
None. Check with `squeue -u gfrise`. The tmux session `veto` on farm-ui1 has a `jobs` window for
monitoring.

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
