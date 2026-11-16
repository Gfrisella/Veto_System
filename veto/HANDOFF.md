# Veto optimisation — handoff (16 Nov 2026, evening)

Read this first in a new session. Details: `docs/STATUS.md` (dated log), `ISSUES.txt` (tracker), `CHANGELOG.md`,
`docs/v0_statistics.md` (MS7 V0 / K+- statistics, closure), `docs/r3_shield_strips_v0rule.md` (R3 study),
`docs/plan_evolution_sbt_shieldend.md` (Shah Rukh's plan + answers).

## Where things are

- Code: branch `veto-nov4-source` (worktree `.claude/worktrees/nov4-source/veto`), pushed. PR to main: to be opened by Guglielmo.
  Another session (gfrise-f5, with background D19 / ACTS agents) works in the same worktree: commit only your own hunks
  (stage a HEAD-based copy with `git hash-object -w` + `git update-index --cacheinfo`), announce batch jobs to avoid duplicates.
- Results of the re-run: `<main checkout>/veto/results_rerun/` (env: `bin/batch/env.sh`, VETO_RESULTS there).
- Report: Background Tagging Report, worktree `Veto_Status_Report/.claude/worktrees/bt-report`, pushed to GitHub `main`
  (Overleaf), last 285ed5d.
- Slurm: `sbatch` from this machine needs the unsandboxed shell; batch scripts in `bin/batch/` (task.sbatch runs line N of a
  .cmd). Use `--partition=standard,hep` (hep is our group's, often idle); farm etiquette (Roland): jobs stay in their CPUs.

## Conventions decided (Guglielmo)

- Everything on the **Nov-4 production**; quote **only full-sample (3 220 spills) statistics**.
- **Only three reference designs**: R1 (`designs/R1_ubt_only.json`, UBT only), R2 (`R2_sbt_ubt.json`, full SBT + UBT,
  = the budget), R3 (`R3_sbt_ubt_strips.json`, R2 + V0 rule + strips in MS7). Old D / E designs deleted (git history only).
- **Costs in drachmas**: 1 drachma = one 0.5 x 0.5 m module of 4 x 4 cm tiles; budget = R2 = 333.5; R1 99.7; R3 378.7. Never money.
- Tiles as SR-389 (2 x 2 cm 90 ps, 4 x 4 cm 120 ps, 2.5 MHz) but with a **+-3 sigma time window** (+-0.469 / +-0.404 ns,
  since 5f7a770; the +-200 ps lost ~20 % of straight muons on time). UBT straws vertical.
- SBT pointing window: r / sin(alpha) along the wall AND spanning the whole 21.6 cm shell crossing (`shell_window`, b87da94).
- Rock x rock is not evaluated (Guglielmo). D14.1 debris in time stays off.
- MS7: forced stores (`ms7_muon_fsv0[k]`, `ms7_muon_fsnov0[k]`, SUM them) only for the VERTEX background; the single-track
  combinatorial (D9) keeps `ms7_muon_v0s` (natural decays). No channel biasing of the forced decays (BR sampled).
- Explain before acting on physics / weights; ask before re-running productions.

## Results of 16 Nov (details STATUS; numbers marked * are being re-run with the +-3 sigma tiles)

- Muon x rock (full selection, before tagging): candidates 634 / 15 y, IP < 10 cm **211** (mu x mu 2.36). After tagging*
  R1 35.0 (rock legs tagged 16 %: R1 has nothing on the vessel walls), R2 2.3e-6 with the shell window. Rock tracks: pi- /
  K- / pbar / e- of 1-6 GeV from the -x cavern wall at z 67-73 m, entering through the vessel side wall; partner muons hard
  (~250 GeV, near axis), which sets the pair direction. Page: https://claude.ai/artifact/CTdxuttxYrQAK6UfNbQmv1
- Tile +-3 sigma test (`*_w200` = old outputs): R1 selected muons tagged 81.7 -> 97.7 %, R1 full pairs 0.012 -> 0
  (partial 0.79 -> 3e-8), R2 pairs -> 0; signal loss +0.3-0.5 points (R1 full 2.64 -> 3.14 %, R2 full 2.88 -> 3.24 %).
- SBT shell window on R2: single SST1 tagged 90.5 -> 93.3 %, signal loss +0.02 points, pairs unchanged.
- MS7 V0 statistics (done): importance sampling closes on the unbiased chain (all exit rates within 1.6 sigma); full
  selection 78 events, 3.3e4 / 15 y before tagging (was 2 events, 6.1e4); V0 copies n_eff 57, heaviest 790. One code-3
  event (6.3 GeV K- decaying in flight at z 65.5 m) carried 1.8e4 -> forced K+- copies added (e825a26).

## Jobs running at hand-over (check `sacct -j <id> -X`; logs `results_rerun/batch/logs/`)

- **2095750** `tile3s` (17 tasks, `batch/tile3s.cmd`): R1 / R2 run_reco full + partial, evstore_eval R1 / R2, D9
  (dis_combinatorial) R1 / R2 (~4 h), d2_selfveto R1 / R2, closure, R3 run_reco full + partial, rock x mu R1 / R2 full
  (+ survivors npz) and partial. All with +-3 sigma tiles + shell window.
- **2095744** `kpm` -> **2095745** `kpmprep`: forced K+- copies (100 / K+-) on all 160 MS7 files (`*_kpm.npz`), stores
  `evstore/ms7_muon_fsv0k`, `ms7_muon_fsnov0k`, cache `interactions/prepared_ms7_v0statk.npz`.

## Next steps (in order)

1. When 2095744/5 are done: R1 / R2 / R3 on the new MS7 sample, partial and full:
   `python3 bin/v0rule_study.py --designs designs/R1_ubt_only.json designs/R2_sbt_ubt.json designs/R3_sbt_ubt_strips.json
   --stores $R/evstore/ms7_muon_fsv0k $R/evstore/ms7_muon_fsnov0k $R/evstore/ms7_nu2026 --prep-tag _nov4full --out
   $R/reco/r123_study.json` (batch, 200 GB: the stores are 20 M + 40 M events). SUM fsv0k + fsnov0k.
   End-to-end check: partial-selection rate old (`ms7_muon_v0s`, 399 events) vs new.
2. When 2095750 is done: update the report — Section 6 rock row IP < 2.5 m (`rock_comb_R*_partial.json`; before-tagging
   numbers are design independent), Table `tab:tag:refeval` (R1 / R2 / R3, both selections, rock and MS7 columns), remove
   the "being re-evaluated" note in Section 7, Appendix G results of the K+- copies; tell gfrise-f5 the Section 6
   after-tagging numbers. Rebuild the rock page with `bin/rock_survivors_page.py` (+ `--dir`) and republish
   (same file `docs/pages/rock_survivors.html`, artifact CTdxuttxYrQAK6UfNbQmv1).
3. Optimiser trial (`evolve/evolve_designs.py --generations 5 --children 8 --inputs inputs_nov4full.npz --prep-tag
   _v0statk`, host slurm, seeds R1 / R2 / R3) — the MS7 source now has statistics.
4. Check the rock library's antiproton yield (7.5 % of the R1 rock survivors are pbar of 1-5 GeV: suspicious).
5. Strips: kink rule, MS7 field in the stations, EM shower inside the iron; V0 time of flight in the simulation.
6. Signal loss: UBT straws with EM debris (~2.8 %) + the wider tile window (+0.3-0.5): check the debris map / straw window.

## Shah Rukh's requests still open (`docs/plan_evolution_sbt_shieldend.md`)

- Q3 (mass cut on the MS7 V0 in the objective): explained, decision pending.
- C3-C8 (traditional SBT as a material): D19 work in the other session (flags default off).
- D (the comparison): we use R1 / R2 / R3. Done: strips (R3), tagging rule (V0 rule + strips), SBT from the reports, B4.

## Open issues (ISSUES.txt)

D7 REOPENED 16 Nov evening (rock x mu not negligible for R1; partial selection pending). D16 (MS7 V0: statistics done,
K+- copies running), D19 (SBT model, other session). Closed 16 Nov: A1, A2, A5, B1, C1, C5, C7, C8, D2, D3, D9, D18, D20.
Deferred: C3, D6, D14 (ACTS, other session).
