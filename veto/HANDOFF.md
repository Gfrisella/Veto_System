# Veto optimisation — handoff (16 Nov 2026)

Read this first in a new session. Details: `docs/STATUS.md` (dated log), `ISSUES.txt` (tracker), `CHANGELOG.md`,
`docs/r3_shield_strips_v0rule.md` (latest study), `docs/plan_evolution_sbt_shieldend.md` (Shah Rukh's plan + answers).

## Where things are

- Code: branch `veto-nov4-source` (worktree `.claude/worktrees/nov4-source/veto`), pushed. PR to main: to be opened by Guglielmo.
- Results of the re-run: `<main checkout>/veto/results_rerun/` (env: `bin/batch/env.sh`, VETO_RESULTS there).
- Report: Background Tagging Report, worktree `Veto_Status_Report/.claude/worktrees/bt-report`, pushed to GitHub `main` (Overleaf).
- Slurm: `sbatch` from this machine needs the unsandboxed shell; batch scripts in `bin/batch/` (task.sbatch runs line N of a .cmd).

## Conventions decided (Guglielmo)

- Everything on the **Nov-4 production**; quote **only full-sample (3 220 spills) statistics**.
- **Only three reference designs**: R1 (`designs/R1_ubt_only.json`, UBT only), R2 (`R2_sbt_ubt.json`, full SBT + UBT,
  = the budget), R3 (`R3_sbt_ubt_strips.json`, R2 + V0 rule + strips in MS7). Old D / E designs were deleted on 16 Nov (only in git history now).
- **Costs in drachmas**: 1 drachma = one 0.5 x 0.5 m module of 4 x 4 cm tiles; budget = R2 = 333.5; R1 99.7; R3 378.7.
  Never money. (Named in honour of Archimedes / Syracusia in the report.)
- Tiles as SR-389 (2 x 2 cm: 90 ps, 2.5 MHz; window +-200 ps); UBT straws vertical; SBT pointing window r / sin(alpha).
- Explain before acting on physics / weights; ask before re-running productions.

## Results so far (full sample, `results_rerun/reco/reco_R{1,2}_*_{full,partial}.json`, S7)

| | pairs / 15 y full (2.36 before) | partial (34.4 before) | signal lost F / P | single SST1 mu tagged |
|---|---|---|---|---|
| R1 | 0.012 +- 0.006 | 0.79 +- 0.2 | 2.64 / 2.73 % | 53 % |
| R2 | 2.5e-6 | 0.0011 +- 0.001 | 2.87 / 2.62 % | 90 % |

Signal loss is dominated by the UBT straws (EM debris in the 25 ns window). R3: not run yet.
Pair numbers are DIRECT counts (A1 decided 16 Nov: direct everywhere, also in the optimiser). The R2 2.5e-6 rests on 9
pairs; the closure check (no EM debris) gets 7 pairs and ~5e-4 on the same candidates: a few heavy pairs decide the
weighted R2 value. run_reco now saves FN_frac_err (weighted), n_eff_survive and the heaviest survivors.
Other: R2 misses 2.1 % of selected muons (n_eff ~3); late-bounced muons via the vessel-end gap never form pairs (D20, C).
MS7 V0: 399 selected events (partial) but only **2** at full selection carrying 6.1e4 / 15 y (statistics, see below).

## Pages (claude.ai artifacts, private)

- R3 design, strip layers, V0 production histograms: https://claude.ai/artifact/JCNdVa2LEaCzg4yudFRkfW
- MS7 V0 event display + strip tagging: https://claude.ai/artifact/5hYk9446RtEnLkFePxW8AY
- R2 coverage, untagged muons by category (full sample): https://claude.ai/artifact/3ejSJY4v9pPTVg75rMUVQq
- (older, 300-spill) R2 untagged muons / r-sin(alpha) fix: https://claude.ai/artifact/FnWCuP3xHpYcQewWnZ17U8
- A1 explained (product rule vs direct count, weak-spot toy, closure numbers): https://claude.ai/artifact/RkWTjSXM8gHe4yWWg5Sg2P

## Jobs

- S7 (2085945): all done except 3 and 8 (DIS x mu, R1 / R2), running. Task 11 (rock x mu, D6) failed OOM and is dropped
  (D7 closed: not needed). Check: `sacct -j 2085945 -X`; logs `results_rerun/batch/logs/S7_eval_*.log`.
- S6b cut flow without tagging (2089733, `bin/cutflow_notag.py`): done, `results_rerun/reco/cutflow_notag_*.json`
  (per target, nu flavour, V0 species; interactions / >= 2 tracks / pair no IP / IP < 10 cm / IP < 2.5 m) for Section 6.
- R1 / R2 / R3 test: cancelled before running (R3 changed). Resubmit (layer maps for 29.39 / 29.59 / 29.79 are cached):
  ```
  cd <worktree>/veto && export VETO_CODE=$PWD && source bin/batch/env.sh && R=$VETO_RESULTS
  D="designs/R1_ubt_only.json designs/R2_sbt_ubt.json designs/R3_sbt_ubt_strips.json"
  { echo "python3 bin/v0rule_study.py --designs $D --out $R/reco/r123_study.json"
    for d in $D; do n=$(basename $d .json); for ip in full partial; do
      echo "python3 bin/run_reco.py $d --ip-cut $ip --em-debris --signal eventcalc --n-signal 40000 --multiple-scattering --out r123_${n}_${ip}.json"; done; done; } > $R/batch/r123.cmd
  sbatch --array=0-6 --job-name=r123 --output=$R/batch/logs/r123_%a.log --export=ALL,CMDFILE=$R/batch/r123.cmd,VETO_CODE=$VETO_CODE --mem=200G bin/batch/task.sbatch
  ```

## Next steps (in order)

1. **MS7 V0 statistics** (GO given 16 Nov; running, see docs/v0_statistics.md: jobs 2089677 production, 2089678 stores,
   2089679 cache `prepared_ms7_v0stat`; 27 % of the V0 are secondary -> two complementary samples fsv0 + fsnov0, SUM them).
   Original proposal: v0-copies 5 -> 100 (x20, nearly free); draw only
   library interactions with a V0 among the products, weight x fraction (x10-70: only 1.4 % of Geant4 and 20 % of Pythia
   iron interactions have a K_L / K_S / Lambda; needs code in `bin/shield_dis.py`); more seeds (x10). Target x1e3.
   Then rebuild `prepared_ms7_nov4full` (`bin/prepare_interactions.py --ms7 ...`, now with strip layer data).
2. Run the R1 / R2 / R3 test (above); fill report Table `tab:tag:refeval` (R3 row, MS7 V0 columns).
3. Report Section 6 (no-tagging yields): FILLED 16 Nov (report f9fe9c9, not pushed); pending in red: MS7 muon V0 row (step 1)
   and vessel-DIS x mu pairs (S7 3 / 8 OOM, D9).
   Rock pairs: one sentence (negligible, not re-evaluated).
4. Optimiser trial (`evolve/evolve_designs.py --generations 5 --children 8 --inputs inputs_nov4full.npz
   --prep-tag _nov4full`, host slurm, seeds R1 / R2; add R3): only after 1 (the MS7 V0 source would dominate on 2 events).
5. Strips: kink rule, MS7 field in the stations, EM shower inside the iron; V0 time of flight in the simulation.
6. Signal loss from the UBT straws with EM debris (~2.8 %): check the debris map and the straw window.

## Shah Rukh's requests still open (`docs/plan_evolution_sbt_shieldend.md`)

- Q3 (mass cut on the MS7 V0 in the objective): explained, decision pending; first measure what a visible-mass cut removes
  from the V0 background and from each benchmark.
- C3-C8 (traditional SBT as a material): muon DIS libraries for steel / LS, muon DIS in the SBT walls, neutrino DIS in steel /
  LS, Geant4 through the thick wall, LS response library (efficiency, delta rays), SBT accidentals with debris. Our SBT is
  a cell model without material (ISSUES D19); neutral products of helium interactions not in it (relevant for D2).
- D (the comparison): evolution runs (1) fins + UBT, (2) with shield-end planes, (3) SBT + UBT fixed; we now use R1 / R2 / R3.
- Done from his list: strips (B, as R3), tagging rule (V0 rule + strips), SBT from the reports, B4 signal set.

## Open issues (ISSUES.txt)

Closed 16 Nov: A1 (direct count), A5, C1 (no ghosts with one-view straws), C5 (3220 spills), D7 (no rock pairs), D20.
Also closed 16 Nov: A2, C7, C8, D2 (R2: nu-He 3.6 / 15 y), D3, D18. Deferred: C3, D6, D14 (debris in time, ACTS).
B1 closed (r_ms = 6 kept; nothing lost at the vessel entrance). Still open: D9 (S7 3 / 8), D16 (MS7 V0 production),
D19 (SBT model).
