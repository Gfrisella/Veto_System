# Investigations of 20 Nov 2026 (copies of the write-ups, scripts and plots; data tables stay in results_rerun)

| folder | question | main result |
|---|---|---|
| `interaction_x_mu/` | why interaction product x mu went up on 20 Nov | partner muons stopped in the SBT shell (45 % of the single-SST1 weight) were kept, and the rest of the interaction was ignored; fixed in bin/dis_combinatorial.py (own hits, companions, V0 vertex). Page: https://claude.ai/artifact/7mp1KojngTYPHmUwJTVn5U |
| `cavern_rock/` | why cavern rock x mu is 211 / 15 y, rock-track momentum at SST1 | no normalisation bug; soft (median 1.7 GeV) large-angle hadrons from the -x wall; library already transports through concrete; still ~28x above FairShip (2.5x from the missing vessel / SBT wall, ~10x open) |
| `ubt_sbt_distributions/` | where mu-only, DIS mu, DIS nu cross the UBT and SBT (R2, truth level) | 27 % of selected vessel-nu DIS reach neither tagger (~1.6 / 15 y); MS7 mu DIS 91 % UBT only |
| `fairship_lxplus/` | FairShip TRY_2026_EM_noSBT_full readers (lxplus) | Flux_MS7.py (Test A, condor 21080019), Origin_SST_noSBT.py (origin / momentum of the tracks at the SST plane) |
