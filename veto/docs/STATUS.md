# Status — where everything is (living file, updated by Claude; last update: Fri 2026-10-30 20:30 CET)

Pick-up file: what runs where, what is done, what is next, how to reach every machine. Paths: `R` =
`/disk/users/sqasim/workspace/uzh/Veto_System/veto` (farm), `S` = `/disk/lhcb_data/sqasim/veto_source_pool1` (farm).

## 1. Running right now
| what | where | how to look | expected end |
|---|---|---|---|
| **zeta**: 6000-seed new-storage production (seeds 0-5999, 4134 done at 20:23) | farm-gpu5 GPUs 1-3, tmux `veto_zeta`, script `$R/data/sim/veto_muons_alpha_v0/run_6000_gpu5.sh` | `ls $R/data/sim/veto_muons_alpha_v0 \| grep -c ^seed_`; logs `.../logs/run6000_g5_gpu{1,2,3}.log`, `.../logs/run6000.log` | Sun ~morning; then it builds `design-optimization-dataset-zeta-new-storage-<N>seeds-<date>` + probe + n_eff by itself ("ZETA ALL DONE" in run6000.log) |
| GPU memory hold for zeta | farm-gpu5, tmux `veto_zeta_hold`, `$R/bin/hold_gpu.py` (5 GB/GPU) + workers with `VETO_KEEP_GPU_CACHE=1` | `$R/data/sim/veto_muons_alpha_v0/logs/hold_gpu.log` | releases itself 5 min after the workers end |
| ~~copy of the pool to daint~~ **done 20:28**: pool v1 as built (integral normalisation) is at `/ritom/scratch/cscs/sqasim/pool_v1_incoming/pool_v1/` (45 GB) | — | — | — |

| **dataset (c) transport**: daint job **5216231** (array 0-3 = 4 nodes x 4 GH200), script `/ritom/scratch/cscs/sqasim/sim/daint_c.sbatch`, input `/ritom/scratch/cscs/sqasim/pool_v2/` | daint | `squeue -u sqasim`; seeds in `/ritom/scratch/cscs/sqasim/sim/veto_muons_pool_v2/{train,hidden}/`, logs `.../logs/gpuK.log`, slurm logs `/ritom/scratch/cscs/sqasim/sim/logs/` | ~3.5 h after start (submitted Fri 22:00; daint full, limit lowered to 5 h at 22:10 for backfill; first estimate was Sat 07:54) |

**Pool v3 BUILDING (Fri 22:45, farm-gpu4 tmux `veto_pool3`)**: fresh parts reweighted per (p, pT) cell, because the 2D
check (`bin/source_hist2d.py`, `bin/plot_source_hist2d.py`, `veto/results/pool_v2/source_hist2d_*.png`) showed v2 matches the
file in p but differs by +-20 % in pT (angle; partly our rows carry the direction at the target exit, the file the one at
production). Log `/disk/lhcb_data/sqasim/pool_v3_build.log` ("POOL3 DONE"), then `pool_v3_hist2d.log`, `pool_v3_edges.log`.
**DONE Fri 23:40**: pool v3 built and checked (2D ratio within a few % below pT 2.5 GeV; above, the file itself is too sparse,
p-only factor used), copied to daint `/ritom/scratch/cscs/sqasim/pool_v3`, and on daint `pool_v2 -> pool_v3` (symlink; old v2
moved to `pool_v2_ponly`): the pending job 5216231 will read **v3**. Plots `veto/results/pool_v3/source_hist2d_*.png`.

Pool v2 DONE (Fri 21:50): `/disk/lhcb_data/sqasim/pool_v2/` (farm) and `/ritom/scratch/cscs/sqasim/pool_v2/` (daint, sizes checked);
pool / file = 1.00 in every momentum bin for all components (`pool_v2_edges.log`). Pool v1 (flat factors, with steps) kept for
reference on both sides. Note: the farm proxy (2226) hung after the 45 GB copy (Fri ~22:00) — retry / restart it if needed.

Nothing runs on eiger or daint. Next after v2: plot (`python3 bin/plot_pool_spectra.py results/pool_v2/edges.json`, locally after
copying edges.json), copy `/disk/lhcb_data/sqasim/pool_v2` to daint `/ritom/scratch/cscs/sqasim/`, launch (c) there.

## 2. Done today (results)
- **(a)** dataset `design-optimization-dataset-zeta-snapshot-3084seeds-2026-10-30` (seeds 0-3083 of zeta, 281 M rows), probe in
  `$R/results/compare_abc/probe_a.log`, `$R/results/probe/probe_<name>.npz`. **(b)** = epsilon (6000 seeds, default storage).
- **Input thinning study** (`bin/input_thinning_study.py`, `results/thinning/`): source-bin rule keeping 20 % of the 601 M input
  leaves the weakest cells and the rare types unchanged; `q_bin.npy` = that rule (q0 = 0.01).
- **Source importance** (`bin/source_importance.py`): the weak cells / rare types are limited by **min. bias 20-150 GeV and charm
  20-50 GeV** (heavy rows), not by the dimuon class.
- **Source timing** (`bin/source_timing_study.py`, `results/timing/`): n_eff per core-hour per generator, file convention applied.
  `--emin 20` min. bias: ~950 k n_eff / core-h in 30-150 GeV (12x mb2).
- **Fresh pool production** (all done): `$S/tail/` pool_mb20 430 slices (k 10000-), pool_tail 1500 (k 20000-), pool_dmhigh 2220
  (k 30000-, **made on eiger**), charmc 2179 of 2180 (k 1000-3179; slice 2245 failed, not rerun). Logs `$S/log/`.
- **Pool v1** built: `/disk/lhcb_data/sqasim/pool_v1/` — `train_000..008.h5` (9 x 97.4 M), `hidden_000..001.h5` (2 x 77.3 M,
  15 % of source events, never for training), `pool.json` (flux factors: mb20 0.717, tail 0.953, dmhigh 0.608, charm 0.858).
  Build log `/disk/lhcb_data/sqasim/pool_v1_build.log`.
- **OPEN ISSUE — spectrum edges** (`bin/pool_edges.py` → `pool_v1/edges.json`): our min. bias generator is softer than
  Guglielmo's file (mb2 and mb20 alike: ~1.7x at 20 GeV → 1.0 at 126 GeV). With integral normalisation per part the pool has
  a +10 % step at 30 GeV and a **-25 % step in min. bias at 150 GeV**; dimuon 0.91 just above 150, 1.3-1.6 at 260-340; charm
  +2-6 % at 20-28. **Decision pending with the user**: (1) reweight fresh muons bin by bin to the file shape (recommended,
  rebuild ~40 min), (2) keep, (3) our generator's shape. Plots: `veto/results/pool_v1/` (local).
- **EM debris study** (`bin/em_debris_study.py`, `cpp/g4slab.cc`, `$R/results/em_debris/`): P(charged secondary > 1 MeV behind
  1 m iron) 11-23 % (5-300 GeV), mostly a different 25 cm cell at 1-5 m; 1 cm scintillator ~4 %; straws ~0.1 %. Concrete at
  normal incidence not representative (side walls are grazing).
- **eiger** works: Apptainer (unprivileged) in `~/apptainer`, image + code + numpy on `/capstor/scratch/cscs/sqasim/`.
  Dimuon part (2220 slices) ran there in 48 min on 18 nodes (job 8265127, ~14 node-h).
- **Farm vs eiger, same slices**: Pythia identical; Geant4 cascade diverges (Intel vs AMD floating point) → statistically
  equal, not bit-identical (54652 vs 54567 muons in slice 30000).
- **daint GH200 transport validated**: user's container `cudabear_gh200_oct_2026` (Python 3.14, torch 2.14+cu132) via the
  CSCS Container Engine; faster_muons_torch rebuilt for sm_90; geometry cached → **bit-identical** to the farm on 1 M muons
  (positions/momenta exactly equal; weights to 1e-16), transport ~1.6x faster per GPU.

**Chunked pool mode tested on daint (30 Oct ~21:00, pool v1)**: chunk load 5 s, ~30 s per seed of 97 M muons on one GH200
(transport 18 s); rare-type rows per seed as in zeta. Job script ready (laptop copy:
`~/.claude/jobs/e59a0386/tmp/daint_c.sbatch`; to be placed at `/ritom/scratch/cscs/sqasim/sim/daint_c.sbatch`): array of 4
nodes x 4 GPUs, `--chunk-start` per worker, 6000 training seeds → `/ritom/scratch/cscs/sqasim/sim/veto_muons_pool_v2/train`,
then 1000 hidden seeds → `.../hidden`. ~3 h on 4 nodes (~12 node-h). New generator options: `--n-chunks`, `--chunk-start`.

**Parked physics question (revisit later, agreed with the user Fri 30 Oct)**: our Pythia 6 + Geant4 source is softer than
Guglielmo's FairShip 2018 file — min. bias raw/file 1.5 at 30 GeV → 1.05 at 150 GeV (mb2 and mb20 alike), dimuon >= 150 GeV
1.5-1.9, charm ~1.2 at 20-30 GeV. Pool v2 hides it by reweighting to the file; which spectrum is right is open (a systematic).
Plots: `veto/results/pool_v1/pool_spectra.png` (before), `veto/results/pool_v2/pool_spectra.png` (after).

**2018-recipe test running (Fri 23:30, farm-gpu4 tmux `veto_fsm`)**: min. bias >= 20 GeV with Pythia 8 + FTFP_BERT (tag
`fsm_p8_ftfp`, slices 50000-50019) and Pythia 8 + QGSP_BERT (`fsm_p8_qgsp`, 51000-51019) into `/disk/lhcb_data/sqasim/fairship_match/`
(done marker `log/done.txt`); compare with the file and with pool_mb20 (Pythia 6 + FTFP_BERT). FairShip 2018 recipe (read from
CVMFS, `veto/results/fairship_ref/`): Pythia 8 SoftQCD:inelastic (+ gmgm2mumu, PromptPhoton, WeakBosonExchange), p/n by Z/A,
cτ > 1 mm stable (decays in Geant4), Geant4 10.x QGSP_BERT_HP_PEN, dimuon class = BR(η,ω,ρ,η',φ → μμ) x100 (weight 768.75/100),
charm/beauty from a Pythia 6 cascade file with chicc 1.7e-3, chibb 1.6e-7. User: no FairShip runs (30x slow), reproduce the
recipe in our own chain. Next idea: compare everything with the SHiP 2018 muon-flux measurement (data).

**Fossil hunt result (Sat 31 Oct 00:50)**: min. bias 20-150 GeV vs the 2018 file (file selection emulated):
A Pythia 6 + G4 11 FTFP 1.36, B Pythia 8 + FTFP 1.40, C Pythia 8 + QGSP 1.42 (flux ratio 30-150 GeV). Momentum excess
(1.7 at 20 GeV → 1.0 at 150) common to A/B/C → not Pythia, not the physics list. pT: Pythia 8 has 0.3-0.5x fewer
large-pT muons above 40 GeV; Pythia 6 closer to the file. Plot `veto/results/pool_v3/recipe_compare.png`
(`bin/recipe_compare.py`, `bin/plot_recipe_compare.py`). **Suspect**: survivor bias in our emulation of the 2018 selection
(the loss map is built only from muons that reached the 2018 plane) → test with an unbiased survival probability from the
2018 ROOT file's MCTracks (~15 min). **Then (user request)**: compile Geant4 10.0 (10.00.p04) from the official source with
our own patches (NOT from FairShip/CVMFS), build tailtransport against it, rerun C.

**EM debris library running (Sat 00:00, farm-gpu4 tmux `veto_debris`)**: `bin/debris_library.py` (21 jobs: 0.5 m iron
B = 0 / 1.7 T mu- / mu+, 5 angles; scint, straw, steel), output `$R/results/debris_library/*.npz`, logs `.../log/`, marker
`log/done.txt`. `cpp/g4slab.cc` now has an optional uniform B_y in the layers. Plan (agreed): debris generated in the veto
sim (fastveto-like), from the muon paths: back-project A to the shield exit face z = 29.89 m (iron 25.89-29.89, last
magnet) and check iron there with the geometry cache corners; stations crossed; walls at grazing incidence (bounce K).
Library + a plain C++ model validated against it. NOT inside cuda-muons.

**Plan B for the dataset question (user, not now)**: scale the fresh flux up so it is above the old sample everywhere
(differences are about +-20 % in pT, up to +50 % in p → a factor ~1.25 to be conservative).

**Sat 31 Oct 11:30 update**:
- (c) transport job 5216231: nodes 0-2 ran 02:52-07:52 (TIMEOUT at 5 h) → train 5996/6000, hidden 796/1000; node 3
  (nid006377) had no working GPU (torch: CUDA not available) → crashed at start. Stale claims cleared; top-up job
  **5219422** (1 node, 1.5 h, --exclude=nid006377), pending. Hidden = transport of the 15 % hidden source muons (validation).
- Dataset (c) build + probe + n_eff on daint: job **5219435** (`/ritom/scratch/cscs/sqasim/sim/daint_build_c.sbatch`),
  store `VETO_STORE=/ritom/scratch/cscs/sqasim/store`, probe log `/ritom/scratch/cscs/sqasim/sim/logs/probe_c.log`.
- Debris library DONE (21 files, `$R/results/debris_library/`): magnetised iron changes the debris by <= 5 % → no field
  needed; iron 7.5-23 % of muons with e+- > 1 MeV (2-300 GeV), 0.7-13 photons; scint ~4 %, straw ~0.1 %.
- Fossil hunt: the 2018 ROOT file contains only muons that reached the plane → no unbiased survival from it; but the excess
  is 1.2-1.3 at 60-100 GeV where everyone survives → not the selection emulation. New suspect: the TARGET — 2018
  (targetOpt 17): 13 Mo (58 cm) + 4 W (8,8,10,35 = 61 cm); ours ECN3 "MoW": same Mo + 5 W (5,8,10,20,35 = 78 cm).
  Added `--target SHiP2018` (cpp/tailgen/tailtransport.cc, bin/tail_produce.py; rebuilt on the farm). Running (farm-gpu4,
  tmux `veto_fsm2`): C' = Pythia 8 + QGSP + 2018 target (`fsm_p8_qgsp_t18`, 52000-), A' = Pythia 6 + FTFP + 2018 target
  (`fsm_p6_ftfp_t18`, 53000-), marker `/disk/lhcb_data/sqasim/fairship_match/log/done2.txt`. Then Geant4 10.0 (own patches).

**Sat 31 Oct 15:45 — end-to-end on (c), with and without EM debris (farm-gpu4 tmux `veto_c_e2e`,
`$R/results/end_to_end/run_c_e2e.sh`, log `$R/results/end_to_end/run_c.log`, "ALL DONE" at the end)**: training seeds copied
daint → `/disk/lhcb_data/sqasim/sim_c/train`; dataset `design-optimization-dataset-c-pool-v3-6000seeds-2026-10-31` built on the
farm (daint build job cancelled); maps `sample_c.npz`; inputs `inputs_c.npz` → `inputs_c_s.npz`; muon DIS
`results/end_to_end/<design>_c_mu.npz` (new `--mu-suffix`); evaluations tags `e2e_c`, `e2e_delta_em`, `e2e_c_em` (designs D6_tile2
E3_robust E4_entry2 E4_ubt2 E5_evolved, cones 0 10 16 17.5 20; reference `e2e.json` = delta). EM debris: `lib/fastveto/debris.py`
(library sampler), `bin/debris_maps.py` (UBT + fin maps of debris added to the muon maps; e+- full, photons x P_GAMMA = 0.05
placeholder; iron = the last magnet's outer box |x| < 1.71, |y| < 1.27 at z = 29.89 m → upper bound). C++ model
`cpp/emdebris/emdebris.cc` vs Geant4 (`bin/debris_model_compare.py`): shapes OK, P(e) -10..-37 %, ~50-100x faster → research track.

**Sat 31 Oct ~20:30**:
- Geant4 10.00.p04 BUILT from the official source (`/disk/lhcb_data/sqasim/geant4_10/{install,data}`, built on /scratch of
  farm-gpu4; our patch `patches/01_glibc_fsqrt.patch`; build script `build_g410.sh`). tailtransport against it:
  `cpp/tailgen_g410/` (C++14), run with `bin/g4_10.sh` (sets data + TAILTRANSPORT_BUILD); tailtransport.cc has version guards.
- Fossil hunt (30-150 GeV flux / 2018 file): A 1.36, B 1.40, C 1.42, A18 1.36, C18 1.42, **A10 1.45, C10 1.57 (Geant4 10
  makes it worse)**. Not Pythia, physics list, target layout, or Geant4 version. New suspect: our target stands in AIR
  (FairShip's was enclosed in iron shielding) → added `--target SHiP2018+Fe` (target inside a 2x2 m iron block); running
  (tmux `veto_fsm4`, tags fsm_p8_qgsp_t18fe 56000-, fsm_p6_ftfp_t18fe 57000-, marker log/done4.txt).
- e2e chain: make_reco_inputs ran OUT OF GPU MEMORY after 2 h 20 (6000 runs, shared GPU) → rerun with `--pair-runs 1000`
  (new option; pairs from 1000 runs, weights x 6) `--pair-block 2048` in tmux `veto_inputs_c` (log `results/reco/inputs_c.log`,
  "INPUTS DONE"); the chain's muon-interaction generation continues; its e2e_c / e2e_c_em evaluations will fail (no inputs)
  → rerun them by hand after INPUTS DONE (e2e_delta_em works).

**Plan (user, Sat 31 Oct)**: validate our muon source against DATA — SHiP Collaboration, "Measurement of the muon flux
from 400 GeV/c protons interacting in a thick molybdenum/tungsten target", Eur. Phys. J. C 80 (2020) 284,
doi:10.1140/epjc/s10052-020-7788-y, arXiv:2001.04784 (H4 beam 2018, Mo/W target 13 lambda + 2.4 m iron absorber, drift-tube
spectrometer, 3.27e11 p.o.t.). Needs their geometry as a tailtransport option and the spectra at their measurement plane.

## 3. Next steps (in order)
1. User decides the spectrum-edge handling (section 2). If "file shape": add per-bin factors to `bin/build_pool.py`, rebuild
   on farm-gpu4 (tmux, ~40 min), re-run `bin/pool_edges.py`, re-copy to daint.
2. Write the daint job: 2 nodes x 4 GH200, each GPU the seeds of one chunk (seed s uses chunk s % 9),
   `generate_muon_study.py --input '<pool>/train_{chunk:03d}.h5' --n-chunks 9 --storage alpha --claim`, 6000 seeds;
   the hidden chunks as a separate production. Env: `VETO_GEOMETRY_CACHE=/ritom/scratch/cscs/sqasim/work/geometry_TRY6.pt`,
   `PYTHONPATH=/ritom/scratch/cscs/sqasim/work/pylib314`, `MPLBACKEND=Agg`, `--mm-dir /ritom/scratch/cscs/sqasim/work/MuonsAndMatter`.
   ~4-5 h on 2 nodes.
3. Dataset (c) (`make_design_dataset_gamma.py --old-types=`), probe, compare c vs a and b: rates per type, rate maps,
   n_eff per band.
4. Sunday: zeta final dataset (automatic), its band comparison.
5. Commit (nothing of today is committed; branch `veto-end-to-end`, PR #8): see section 7.

## 4. Access
- **Farm**: our own OTP proxy on `127.0.0.1:2226` (`physik_deployment/custom_ssh/custom_ssh_2.py`; never print or sync
  `custom_authenticator.py`). Login node: `ssh -p 2226 127.0.0.1` (farm-ui, shared disks). Nodes: `ssh -J 127.0.0.1:2226 farm-gpu4`
  (gpu3/4/5). After a network change the proxy can hang once; retry.
- **CSCS** (user `sqasim`, account `uzh42`): certificate `~/.ssh/cscs-key-cert.pub` valid 1 day — renew at
  https://user-account.cscs.ch (SSH Keys → Sign Key, paste `~/.ssh/cscs-key.pub`). Jump host Ela; use
  `ProxyCommand="ssh -o IdentitiesOnly=yes -i ~/.ssh/cscs-key -W %h:%p sqasim@ela.cscs.ch"` with `-i ~/.ssh/cscs-key -o IdentitiesOnly=yes`
  to `sqasim@eiger.alps.cscs.ch` / `sqasim@daint.alps.cscs.ch` (`-J` does not pass the key). Too many failed tries → Ela blocks ~1 min.
- eiger: x86, 128 cores/node, `/capstor/scratch/cscs/sqasim`. daint: aarch64 GH200 (4/node, driver 595, CUDA ≤ 13.2), scratch
  `/ritom/scratch/cscs/sqasim`; container `srun --environment=/ritom/scratch/cscs/sqasim/cudabear_gh200/cudabear-gh200.toml`
  (workdir is reset: always `cd` first; `CUDABEAR_SILENT=1`).

## 5. Containers
- Farm transport: `/disk/users/lprate/containers/MuonsAndMatterContainer.sif` (Python 3.10, snoopy at /opt/snoopy).
- Farm Geant4 / generators: `bin/g4.sh` → `exec_cudabear_november_2025.sh` (image `/disk/lhcb_data/sqasim/images/cudabear_november_2025.sif`;
  numpy etc. come from the farm `~/.local`, not the image).
- Farm uproot/awkward work (old_convention, build_pool, pool_edges): `/disk/lhcb_data/sqasim/images/shared/srq_ml_and_gen_container_latest_stable.sif`.
- Newer, not yet used here: `/disk/users/sqasim/images/cudabear_blackwell_oct_2026_base.sif`.

## 6. Restart commands
- Pool copy to daint (if it died): from the laptop,
  `ssh -p 2226 127.0.0.1 "cd /disk/lhcb_data/sqasim && tar cf - pool_v1" | ssh <daint as above> "cd /ritom/scratch/cscs/sqasim/pool_v1_incoming && tar xf -"`
  (check sizes after; ~30 min off VPN, ~3x slower on the UZH VPN; a VPN switch kills the stream).
- zeta: if `veto_zeta` is gone, clear stale claims (`.claim_N` without `seed_N.npz`) and `tmux new -d -s veto_zeta bash run_6000_gpu5.sh`.

## 7. Uncommitted code (Veto_System/veto)
New: `bin/build_pool.py`, `bin/pool_edges.py`, `bin/source_importance.py`, `bin/source_timing_study.py`, `bin/input_thinning_study.py`,
`bin/storage_policy_study.py`, `bin/em_debris_study.py`, `bin/hold_gpu.py`, `cpp/g4slab.cc`, `lib/veto/probe.py`, `docs/storage_policy.md`,
`docs/live_talk/talk_storage.py`, this file. Modified: `bin/generate_muon_study.py` (keep-cache / reserve, pool chunks
`--input ...{chunk}` + `--n-chunks`, `pool_index` → evt_n, extension path per machine, `VETO_GEOMETRY_CACHE`), `bin/probe_statistics.py`,
`cpp/CMakeLists.txt`, `cpp/build.sh`, talk files. On the farm the production uses the older `bin/generate_muon_study.py`
(md5 72d8b521634b); the new one is deployed there as `bin/generate_muon_study_next.py` and on daint as `bin/generate_muon_study.py`.

## 8. Lessons (do not repeat)
- `pkill -f` patterns can match `xargs`' own `bash -c` wrappers → xargs stops the whole queue (lost gpu4 6 h on 30 Oct).
- `charm_cascade.py` seeds Pythia with 100000 x slice + group: slices must stay < ~9000 (Pythia 6 limit 9e8).
- 300 slices on a 384-thread farm node run ~half speed (192 physical cores).
- Laptop-relayed copies: ~8 MB/s on the VPN, ~24 MB/s off; switching networks kills them.

## Fossil hunt: solved (Sun 1 Nov, ~06:00)
- Reference built from all 67 FairShip 2018 files: /disk/lhcb_data/sqasim/ref2018/reference_2018.h5 (506,355,249 muons, 6.602e10 POT, weight scale 0.9852; bin/build_reference_2018.py). Figure: results/ref2018/reference_comparison.png (bin/plot_reference_comparison.py).
- h5 vs reference: same muons (row count), same total flux per class and charge (ratio ~1.00), but the h5 momenta are shifted down.
- The h5 momentum is the momentum AFTER THE TARGET, not at production: our min. bias, histogrammed in our after-target momentum, over the h5:
  30-40: 1.01, 40-60: 0.98, 60-100: 0.93, 100-150: 0.87 (in production momentum: 1.49 / 1.36 / 1.22 / 1.09). Script: /disk/lhcb_data/sqasim/fairship_match/afterTarget.py.
- So the "40 % more muons" was comparing our production momentum with the h5's after-target momentum. Recipe/Geant4-10 differences are secondary (~5-15 %).
- Exact-match job (bin/compare_reference_h5.py) ended without output (likely memory/timeout); not needed any more.
- Next: check the pool v3 reweighting used the after-target momentum on both sides (consistent with the h5), and the 100-150 GeV 0.87.
- CONSEQUENCE for pool v3 / dataset c: bin/build_pool.py (lines ~128-135) scales our fresh muons UP to the production momentum
  (s = p_prod / p) because we believed the h5 used it. The h5 actually stores the after-target momentum. The (p, pT) reweighting
  to the h5 then pushed the flux back into line, but per muon the momentum and angle no longer belong together. That likely
  explains the angle/pT mismatches and maybe the c vs delta SST1 excess. Fix = pool v4 with s = 1 (no scaling); the reweight
  factors should then be ~0.87-1.0. Needs a new transport of the fresh part -> awaiting the user's decision.

## Running (Sun 1 Nov, 13:50)
- **Pool v4** (farm-gpu5 tmux `veto_pool4`): bin/build_pool.py now keeps the after-target momentum (no scaling), parts selected
  in the after-target momentum, extra column p_prod (nan for old rows). Output /disk/lhcb_data/sqasim/pool_v4, log
  /disk/lhcb_data/sqasim/pool_v4_build.log ("POOL4 DONE" at the end; then hist2d + edges). Check: the per-part flux factors /
  bin_factors in pool.json should be ~0.87-1.0. Then GPU transport (farm / daint) = dataset c v4, after the user decides.
- **2018-like comparison file** (farm-gpu4 tmux `veto_fsm5`, /disk/lhcb_data/sqasim/fairship_match/run_fsm5.sh): C10 recipe
  (Pythia 8 + Geant4 10.00.p04 QGSP_BERT, 2018 target, min. bias >= 20 GeV), slices 54020-54219 (+ the 20 existing ->
  1.1e7 p.o.t.), done marker log/done5.txt. Then on the farm: `bin/build_compare_2018.py --compute` ->
  /disk/lhcb_data/sqasim/compare2018/compare_2018_minbias.h5 + compare_2018.npz; locally plot -> results/ref2018/compare_2018.png.
  Missing vs 2018: dimuon/photon and charm classes (Geant4 10 build has no dimuon options), K_mu3 / K_L.
- DONE 2018-like file: /disk/lhcb_data/sqasim/compare2018/compare_2018_minbias.h5 (1.1e7 p.o.t., 593 M muons), figure
  results/ref2018/compare_2018.png. Production momentum vs the 2018 files: ours 1.13 (30 GeV) -> 1.0 (100) -> 0.77 (200)
  -> ~0.55 (300): softer spectrum. After target vs h5: 1.2 at 30 GeV, crosses 1 at ~110. pT tail (> 1.5 GeV) narrower
  than 2018 by up to 2-3x; the h5's pT shape follows the 2018 files. Our production pool_mb20 (Pythia 6 + Geant4 11) is
  closer to the h5 (1.02-0.87) than this "2018 recipe": Geant4 10 adds ~+10 %. Suspects for the pT/high-p gap: decay channels
  we don't force (K_mu3, K_L, hyperons), Pythia 8 version/tune.
- Pool v4: pool_mb20 flux factor 1.017 (v3: ~1.36) -> convention fixed.

## Complete target generator (Sun 1 Nov, evening) — user: "no holes" in the target production
- Pool v4 DONE (/disk/lhcb_data/sqasim/pool_v4: 7 train chunks, 1 hidden; flux factors mb20 1.017, tail 1.193,
  dmhigh 0.737, charm 1.151; v3 was 0.717 / 0.953 / 0.608 / 0.858). Ready for GPU transport, awaiting the user.
- cpp/tailgen/tailtransport.cc: new initialize(mode="complete" | "analog") (needs Pythia: TT_WITH_PYTHIA, CMake finds
  it). Generic forcing from Pythia's decay tables (+ K_S -> pi mu nu added): long-lived species per step, prompt species
  (eta, omega, rho, phi, eta', J/psi, D, B, tau, ...) decayed by an embedded Pythia decayer wherever they appear,
  gamma -> mu mu and e+ e- -> mu mu boosted, the Pythia event's own muons kept (pythia_mesons "complete" option),
  other-process muons kept. Analog = same without forcing. Target "Air" for decay tests. Legacy mode unchanged.
  Build used: /disk/lhcb_data/sqasim/tt_complete_build/tailtransport*.so (TAILTRANSPORT_BUILD).
  bin/tail_produce.py --mode complete|analog; npz extra columns hf, mode, diag; gen = proc code.
- Validation (bin/complete_validation.py, outputs /disk/lhcb_data/sqasim/complete_validation, runner run_cv.sh):
  guns on farm-gpu5 tmux veto_cv_guns (done marker log/done_guns.txt), full events on farm-gpu4 tmux veto_cv_full
  (cv_analog 2000 slices = 1e8 p.o.t., cv_complete 80 slices FTFP_BERT, cv_complete_qgsp 80 slices; log/done_full.txt).
  Then: VETO_SOURCE=/disk/lhcb_data/sqasim/complete_validation python3 bin/complete_validation.py report (container).
- (Sun ~18:00) Validation round 1 (guns) found: B mesons -12.5 % (Pythia lists B/D channels per charge, onMode 2/3:
  BR_mu normalisation fixed per charge), K_L muon shape (now Geant4's own decay kinematics for forced decays where
  Geant4 has the channel), Geant4 tables lack K_S / Lambda / Sigma- / Xi semileptonic muon channels (now added to
  Geant4's tables in both modes). Everything else agreed to <= 1 %. Old outputs in complete_validation/v1; all reruns
  on the new build (tt_complete_build md5 67825c6d...).
- FairShip 2018 production decoded from git (FairShip muonShieldOptimization/run_fixedTarget.py, shipgen/
  FixedTargetGenerator.cxx, gconfig/*, shipdist): Pythia 8.223 (Monash default), Geant4 10.1.p03 (NOT 10.0: "10.0" in
  the file names is ecut), QGSP_BERT_HP_PEN, Pythia settings = ours (SoftQCD inelastic, gmgm2mumu, PromptPhoton,
  WeakBosonExchange; tau0 > 1 mm stable), all Pythia final particles incl. muons passed to Geant4, Pythia8 external
  decayer for D, Lambda_c, D_s, tau; gamma->mumu AND e+e- -> mumu x100; eta/omega/rho/eta'/phi mumu BRs x100 in Pythia;
  charm removed from min. bias at the merge (mergeMbias.py), charm class added. Beam spot x = y = 0.
- Pythia 8.223 built: /disk/lhcb_data/sqasim/pythia8223 (+ pythia_mesons_8223). Hadron spectra 8.316 / 8.223 (same
  settings): +3-5 % at 20-40 GeV, -13 % pi+ at 100-400 GeV, -11..-23 % at pT > 1.5 GeV (results/pythia_versions).
  Running: cv_complete_t18_p8223 (farm-gpu5 tmux veto_cv_p8223, 80 slices, SHiP2018 + QGSP_BERT + Pythia 8.223 events).
- (Sun ~17:25) EIGER (user: "use eiger to get faster results"): code + new builds at /capstor/scratch/cscs/sqasim/
  {veto, tt_complete_build, pythia8223}; job script veto/eiger_cv.sbatch (TAG, MODE, FIRST, LAST, PER, TARGET, PHYSICS,
  EXTRA_ENV); outputs $S/cv/tail. Jobs: 8267973 cv_analog 60600-61999 (11 nodes; the farm has EMPTY PLACEHOLDER files
  for these slices in complete_validation/tail -> replace with eiger's before the report!), 8267984 cv_complete_t18
  62280-62599, 8267985 cv_complete_t18_p8223 62380-62699 (Pythia 8.223). Farm keeps cv_analog 60000-60599, cv_complete,
  cv_complete_qgsp (gpu4), guns + t18 62200-62279 + p8223 62300-62379 (gpu5).
- Geant4 10.01.p03 build running on farm-gpu4 (tmux veto_g4101, /disk/lhcb_data/sqasim/geant4_101/build_g4101.sh);
  next: tailtransport against it + Pythia 8.223 (complete mode) = the exact 2018 combination.
- (Sun ~17:35) Geant4 10.01.p03 BUILT (/disk/lhcb_data/sqasim/geant4_101; env bin/g4_101.sh; module
  cpp/tailgen_g4101 = tailtransport + Pythia 8.223 static). Pythia 8.2 API: particle loops nextId(1) (nextId(0) returns 0
  in 8.223: the first 8.223 runs had an empty species table / undecayed list -> all rerun). Generator Pythia data:
  PYTHIA_MESONS_DATA (separate from the module's PYTHIA8DATA). The first p8223 attempts failed at start (no outputs).
- Eiger jobs now: 8267984 cv_complete_t18 (8.316 + G4 11.2), 8267996 cv_complete_t18_p8223 62300-62699 (8.223 + G4 11.2),
  8268000 cv_2018exact 62700-63099 (8.223 + G4 10.1.p03 = 2018), all SHiP2018 + QGSP_BERT, 20 M p.o.t. each;
  Eiger copies: $S/geant4_101, $S/tt_g4101_build. cv_analog 60600-61999 DONE on eiger (12 min) -> copy to the farm
  (replace placeholders) before the report.
- (Sun ~17:55) VALIDATION PASSED. Guns (new build): all 27 species complete/analog within ~2 sigma, shape chi2/ndf ~ 1.
  Full events (ECN3, FTFP_BERT, >20 GeV): analog 1e8 p.o.t. 3.930e-4/pot +-0.5 %, complete 4e6 p.o.t. 3.944e-4 +-0.2 %,
  ratio 1.003; long-lived 1.003, prompt (+ "Pythia muon" label in analog) 1.01, gamma->mumu 0.95+-0.02 (boost x500 side
  effect?), e+e- 1.05+-0.2. Figure results/complete_validation/complete_validation.png, summary.json.
  NOTE nextId(0) in pythia_mesons ALSO returns nothing with 8.316 -> runs with binaries built 17:00-17:26 (analog slices,
  eiger cv_complete_t18 job 8267984) had prompt species decayed by Pythia (labelled proc 5, unboosted, still unbiased).
  Prompt decays dominate the muon flux above ~80 GeV and the high-pT tail.
- (Sun ~18:15) 2018 COMPARISON (complete mode, SHiP2018 + QGSP_BERT, no charm, production momentum, / 2018 files):
  8.223 + G4 11.2: 1.26 flat to 100 GeV -> 1.10 (200-300) -> 0.99 (300-400); exact 2018 (8.223 + G4 10.1.p03): 1.27-1.28
  -> 1.10 -> 1.05; pT>1.5 share 0.79-0.89 of 2018's. Shape now close; ~+27 % normalisation. G4 10.1 vs 11.2 and Pythia
  8.223 vs 8.316 matter little at muon level. Figures results/complete_validation/compare_*.png (local), overview
  compare_complete_overview.png. Flux by parent (exact 2018): pi 42 %, K+- 17 %, eta 15 %, omega 14 %, gamma 5 %, rho 4 %.
  Hypothesis: 2018 lacked cascade-level eta / eta' muons and the Dalitz channels (FairShip copied only pure mumu of
  omega / phi / rho to Geant4). Test running: eiger job 8268057 cv_2018exact_b (120 slices, proc 7 = prompt decay in the
  cascade; module $S/tt_g4101_build_b).
- (Sun 18:35) Blue curve (cv_complete_t18, ours) is noisy: its pythia_mesons had the nextId bug -> first-collision prompt
  mesons decayed unforced in Pythia. Rerun with fixed binaries: eiger job 8268067 cv_complete_t18_c 63300-63699
  (module $S/tt_complete_build_b with proc 7). FairShip 2018 python/AddDiMuonDecayChannelsToG4.py: replaced Geant4's
  tables of omega, phi, rho (223, 333, 113) with Pythia's (all channels, boosted BRs) -> only cascade eta / eta' lacked
  muon channels in 2018 (the suspect for the +27 %).
- (Sun ~19:10) THE 2018 EXCESS EXPLAINED. FairShip's muonShieldOptimization/extractMuonsAndUpdateWeight.py (2018) gave
  weight 768.75/100 to EVERY muon with mother eta/omega/phi/rho0/eta' (and to 'Hadronic inelastic' muons), but the
  diMu100 boost (FixedTargetGenerator channel indices) hit only eta->mumu (ch7), omega->mumu (ch7), rho->mumu, eta'->mumugamma,
  phi->mumu in Pythia 8.223; eta->mumugamma (ch6, 3.1e-4: product order 22 13 -13 fails FairShip's check) and
  omega->pi0mumu (ch5) were NOT boosted but still /100 -> effectively dropped. Plus cascade eta/eta' had no muon channels.
  Emulating that on our exact-2018 run (cascade_test.py on eiger, cv_2018exact_b): ours/2018 total >25 GeV 1.29 -> 0.98;
  per bin 1.01 (25-40), 0.96, 0.92, 0.86 (100-150), 0.87, 0.78 (200-300), 0.75 (300-400) (approximate: uniform scaling
  of eta/omega muons). Remaining: ~15-25 % softer above 100 GeV; pT shape bump (suspect: our production pT is built from
  the exit direction -> store creation direction next). Also: the J/psi muons were wrongly cut by my hf flag (is_hf counts
  quarkonia) -> keep quarkonia in comparisons; fix is_hf for open flavour only.
  => The 2018 sample and the h5 (and so pool v1-v4) lack ~24 % of the physical muon flux (eta Dalitz, omega Dalitz).
- (Sun 20:25) 2018 MATCH DONE. Exact-2018 run with channel tags (cv_2018exact_c, eiger, 1e7 p.o.t.; module
  $S/tt_g4101_build_c: new columns ch, pcx/pcy/pcz; is_hf open flavour only) + FairShip's 2018 weighting applied per muon
  (rule2018.py: eta/omega Dalitz muons dropped, cascade eta/eta' dropped): ours/2018 total >25 GeV 1.016, every bin
  0.99-1.03 from 25 to 400 GeV, pT shape ratio 1.00 (0-2.7 GeV). Without the 2018 weighting (our physics): 1.29.
  The earlier pT bump was my exit-direction pT construction. 2018 file check (dalitz2018.py, 1 file): eta pairs 73 %
  pure mumu (natural 2 %) -> eta Dalitz under-weighted x100 in 2018; omega Dalitz ~10x suppressed.
  Figure results/complete_validation/match_2018.png. Pythia 8.316 (ours) vs 8.223: high-p muons 200-400 GeV 0.6x / 0.3x.
- (Sun ~21:00) 2D maps: results/complete_validation/match_2018_2d.png, match_2018_2d_rebin2.png (bin/plot_match_2018_2d.py
  <rule2018 npz> <compare_refs.npz> [rebin]). With the 2018 weighting: bulk pT < 1 GeV within +-3-5 %; at pT > 1-1.5 GeV
  (J/psi 50-73 % there) ours red 1.2-1.3 at 80-160 GeV and blue 0.6-0.8 at 160-320 GeV, region totals 0.97 / 1.09 with the
  same parent mix -> J/psi muon momentum shift and/or our thin J/psi statistics (few hundred J/psi per 1e7 p.o.t.).
  Without the weighting: excess concentrated at pT < 0.5 GeV (eta / omega Dalitz). Optional next: J/psi-enhanced run.

### SUMMARY OF SUN 1 NOV (for the user to digest)
1. The h5 stores the momentum AFTER the target (not production) -> pool v1-v3 scaled our fresh muons wrongly; pool v4
   (/disk/lhcb_data/sqasim/pool_v4) fixes it (flux factor 1.017). Not yet transported.
2. 2018-like comparison file (old recipe C10): /disk/lhcb_data/sqasim/compare2018/compare_2018_minbias.h5.
3. Complete target generator (tailtransport complete / analog modes): every muon source, forced + weighted; validated
   against unforced: 27 particle types within ~2 sigma, full events 1.003 +- 0.5 %.
4. FairShip 2018 decoded: Pythia 8.223, Geant4 10.1.p03 (built both), weighting script. With that software + FairShip's
   weighting we reproduce the 2018 files to 1-3 % (1D) and the 2D map in the bulk.
5. The 2018 files / h5 lack ~22 % of the physical muon flux (eta -> mu mu gamma, omega -> pi0 mu mu weighted /100
   without the x100 boost) -> all our datasets inherit it. Pythia 8.316 vs 8.223: high-p muons (200-400 GeV) 0.6x / 0.3x.
Open decisions for the user: new pool from the complete generator; shield transport + impact on rates / pair
backgrounds; Pythia version vs the muon-flux measurement; optional J/psi-enhanced run.

## Full 2018-equivalent production (Sun 1 Nov, night) — user: "exact, don't come later and say we missed this"
- 2018 inventory (fairship_match/inventory2018.py on reference_2018.h5): 7 weight classes: 768.75 min bias 59.4 %
  (decays 98 %, primary 1.6 % = J/psi...), 7.6875 dimuon 20.5 % (primary VM decays 56 %, gamma->mumu 24 %, HADRONIC
  INELASTIC 19 % = resonances decayed inside Geant4's models with FairShip's boosted tables), 326.16 charm 20.1 %,
  3.26 (charm/100) 0.01 %, 9.37 beauty 0.01 %, 1 and 0.094 negligible.
- New in tailtransport complete mode: proc 8 = muons of resonances decayed INSIDE hadronic models (Geant4 tables of
  eta, omega, rho, phi, eta' replaced by Pythia's, <= 3-body channels (4-body crash in G4 10.1 models), muons as tag
  particles, x100 boost); measured 3.9 % of our flux = 2018's 3.9 %. ch column, creation momentum (pcx..), is_hf = open
  flavour only.
- Charm / beauty: pythia6_charm "had" mode (weak charm / beauty hadrons undecayed, MSEL 4/5), charm_cascade.py
  --hadrons [--flavour beauty] -> <tag>_had_K.npz, bin/hf_decay.py -> <tag>_mu_K.npz (complete mode decays). Beauty chi
  table beauty_chi_table.json. Pythia 6 seed = 1e5 x slice: slices must be < 21474 (charm 14000-, beauty 15000-).
- Onia: pythia_mesons "onia" (only quarkonia written), tail_produce --onia.
- Eiger generic job: veto/eiger_run.sbatch (CMD with @ = slice, DONE, LOG, FIRST/LAST/PER, TT_BUILD); module
  $S/tt_g4101_build_d; outputs $S/cv/tail. Jobs: 8268226 p18_mb 70000-71999 (1e8 p.o.t.), 8268242 p18_onia 72000-73499
  (3e9 Pythia events), charm test 8268151 (14000-14003). Next: charm 14000-14299, beauty 15000-15099, then the
  class / process / mother comparison vs the inventory with physical and 2018 weights.
- (Sun 22:35) Running on eiger: p18_mb (8268226, 2000 slices, started 22:11), p18_onia DONE (1500), charm 8268280
  (14008-14299; 14000-14007 done as test: 3.1-3.3e9 charm muons / spill > 20 GeV, 0 % production weight lost),
  beauty 8268283 (15000-15099); then 8268284 (afterany) = compare_inventory_2018.py compute -> $S/cv/inventory_ours.npz.
  Reference inventory: /disk/lhcb_data/sqasim/ref2018/inventory_ref.npz (farm). Report locally:
  python3 bin/compare_inventory_2018.py report inventory_ours.npz inventory_ref.npz.
  charm_cascade.py: Pythia 6 bins get a time limit + one retry (hangs near threshold); lost weight logged.

## POOL v5 for the veto design studies (Mon 2 Nov, 00:00) — user: "let's just do it, we can always regenerate"
- Choices (agreed): complete target production, latest Geant4 11.2 (FTFP_BERT), ECN3 target (MoW), threshold 10 GeV,
  baseline generator Pythia 8.223 (conservative at > 200 GeV, reproduces 2018), variant Pythia 8.316 (source only);
  charm / beauty from the Pythia 6 cascade, decays by Pythia 8; onia filter run. No h5 rows.
- Eiger (outputs $S/v5/tail, logs $S/v5/log, job script veto/eiger_run11.sbatch, module $S/tt_g411_build_e):
  8268342 v5_mb 80000-80399 (8.223, 2e7 p.o.t.), 8268343 v5_mb316 81000-81199, 8268348 v5_onia 82000-82499 DONE,
  8268349 charm 16000-16299, 8268350 beauty 17000-17099.
- Next: sbatch veto/eiger_pool5.sbatch (bin/build_pool_v5.py -> $S/v5/pool), then daint:
  POOL=/capstor/scratch/cscs/sqasim/v5/pool OUT=/ritom/scratch/cscs/sqasim/sim/veto_muons_pool_v5 NTRAIN=.. NHIDDEN=..
  sbatch /ritom/scratch/cscs/sqasim/sim/daint_c.sbatch (daint mounts /capstor).
- Open (for the talk with Guglielmo): kaon-decay muons x1.35 vs 2018 with identical software (pions 1.00-1.03); charm /
  beauty normalisation 0.825 of 2018's; Pythia 8.223 vs 8.316 above 200 GeV.
- TODO (agreed, tomorrow): Pythia 6 hangs in charm_cascade bins (e.g. pi+ 110 GeV, p 135 GeV; slices 14081, 14294 hung
  repeatedly): (1) reproduce one, attach gdb to find the loop; (2) split bins into ~50-event batches with own seeds and
  short timeouts. Current loss: charm 0.03 %, beauty ~1 % of production weight; costs only wall time.
- (Mon 03:15) POOL v5 DONE: /capstor/scratch/cscs/sqasim/v5/pool (2.43e9 rows, 21 train + 4 hidden chunks, 125 GB;
  flux per spill: mb 6.20e10, onia 2.31e8, charm 7.56e9, beauty 2.54e6). DAINT TRANSPORT SUBMITTED: job 5233970 (array
  0-3, name pool_v5, script /ritom/scratch/cscs/sqasim/sim/daint_v5.sbatch = daint_c.sbatch with
  sim/cudabear-gh200-v5.toml (adds /capstor mount)), OUT /ritom/scratch/cscs/sqasim/sim/veto_muons_pool_v5 (train seeds
  0-5999, hidden 0-999). The user's own daint jobs (pf-design4, pf-jets4, 1 node each): DO NOT TOUCH.
- (Mon ~04:30) POOL v5 DIAGNOSIS (user: let v5 finish, diagnose, wait for instructions): v5 transport works but the
  SST1 / veto-feeding phase space (source p 200-400 GeV, pT 1.5-3 GeV; c: SST1-selected sources p 199-342, pT median 2.7)
  is statistically empty: per chunk n_eff 200-300 GeV pT 1.5-3: v3 49.5k vs v5 12; 300-400: 1061 vs 1.5; 100-200 pT
  1.5-3: 37.7k vs 177. Per seed SST1-selected n_eff 472 (c) vs 1.5 (v5). Cause: v5 min bias = 2e7 p.o.t. in full
  (10 GeV); v3's tail came from cheap high-threshold productions over billions of p.o.t. (pool_tail forced pi/K > 130 GeV,
  pool_dmhigh > 150 GeV). Fluxes where v5 has statistics are compatible. FIX (proposed, not started): complete-mode tail
  runs (threshold 130-150 GeV, ~1e10 p.o.t., ~800-1000 core-h; mid 50 GeV ~1e9 p.o.t.) stitched by after-target momentum.
  Alternative for tomorrow: transport pool v4 (h5-based, convention-fixed). Waiting for the user's decision.
  Local seeds for the diagnosis: ~/.claude/jobs/e59a0386/tmp/seedcmp; cellstat.py.
- (Mon 10:10) v5 transport FINISHED (6000 train + 1000 hidden seeds, /ritom/scratch/cscs/sqasim/sim/veto_muons_pool_v5).
- User: do everything in ONE production (no stitching), verify properly before daint. Implemented importance sampling
  (Russian roulette) in the complete mode: pythia_mesons "roulette=QMIN,E1,N,PT0" (7th column weight), tailtransport
  set_roulette (cascade secondaries), tail_produce --roulette. q = min(1, max(QMIN, (E/E1)^N) max(1, (pT/PT0)^2)).
  Tests running on farm-gpu4 (tmux veto_rr, /disk/lhcb_data/sqasim/rr_test): unbiasedness rr_none vs rr_q05 (40 slices
  x 20k events, same slices) + cost rr_prod (qmin 0.007). Check: rr_check.py. Plan: ~3e9 events, qmin ~0.007, E1 150,
  N 3, PT0 0.5; then pool v6, verification (tail n_eff vs v3, bulk vs v5, a short GPU transport test), then daint.
- (Mon 11:15) Roulette verified (rr_check: flux 0.995 +- 0.009, chi2/ndf 49/40; 8x cheaper per event, same tail stats
  per event). Weight window of recorded muons (tailtransport set_muon_window, tail_produce --muon-window, table
  /disk/lhcb_data/sqasim/rr_test/ww_table.npz = F 0.05 x sum w^2/sum w per (p, pT) cell, >= 1000 rows): C++ check
  17x fewer rows, n_eff -9 %, chi2/ndf 48.7/40. USER APPROVED 3e9 events ("go for it"; can be extended).
  EIGER v6 PRODUCTION SUBMITTED: job 8269279 (24 nodes), tag v6_mb, slices 100000-102999 x 1e6 events (Pythia 8.223,
  G4 11.2 FTFP_BERT, ECN3 MoW, 10 GeV, --roulette 0.007,150,3,0.5, --muon-window $S/v6/ww_table.npz), outputs
  $S/v6/tail (onia / charm / beauty linked from v5). Then: pool v6 (build_pool_v5.py --mb-tag v6_mb --src $S/v6/tail),
  verification (bulk vs v5, tail n_eff vs v3, GPU transport test) BEFORE daint.
- (Mon 11:25) 15 v6 slices failed (generator file not whole 7-column records: short write on capstor); tail_produce now
  regenerates up to 3x. QUEUED CHAIN on eiger (runs without us): 8269279 (main, 24 tasks) -> 8269317 (retry of missing
  slices, afterany) -> 8269318 pool6 (eiger_pool6.sbatch: build_pool_v5.py --src $S/v6/tail --out $S/v6/pool
  --mb-tag v6_mb, then cellstat on v6 & v5 train_000 -> $S/v6/log/verify_cells.txt).
  CSCS CERT EXPIRES Mon 11:50 -> user renews at user-account.cscs.ch. After that: read verify_cells.txt; compare tail
  cells with v3 (farm: /disk/lhcb_data/sqasim/pool_v3/train_000.h5 numbers in this file above); GPU transport test
  (daint debug partition, a few seeds of $S/v6/pool/train_000.h5, per-seed SST1 / veto n_eff vs c); only then
  daint_v5.sbatch with POOL=$S/v6/pool OUT=/ritom/.../sim/veto_muons_pool_v6 NTRAIN/NHIDDEN from pool.json.
- (Mon 11:57) THE 11:12 v6 RUN WAS BROKEN (my error): sbatch --export splits at commas, "--roulette 0.007,150,3,0.5"
  truncated CMD -> every task ran slice 0 without roulette (same file, 6 columns). Cancelled (8269279/8269317/8269318),
  ~570 core-h lost. FIX: eiger_run11.sbatch reads CMDFILE / EXTRAFILE ($S/v6/cmd_v6.txt, $S/v6/extra_8223.txt);
  2-slice test OK (roulette 1, window: 0.7 rows/event, 6.8 ms/event on eiger). RESUBMITTED: 8269386 (24 nodes, main),
  8269387 (retry missing, afterany), 8269388 (pool6 build + cellstat, afterany). At 11:56: 2990 distinct generator
  files, all roulette 1, 0 failures. Slices ~1.9 h -> production ~14:00, pool ~16:00. NEVER put commas in --export.

## Mon 2 Nov, afternoon
- CPU port of cuda-muons' transport: `cpp/cpu_muons/` (OpenMP, same arguments as faster_muons_torch.propagate_muons_with_alias_sampling,
  xoshiro RNG), driver `bin/cpu_transport.py` (dump / run / compare). Environment file (CPU tensors, TRY6):
  /disk/lhcb_data/sqasim/cpu_muons/env_TRY6.pt. Validated vs GPU on 1 M muons (reach fractions and spectra = GPU seed-to-seed).
  Full chunk (101 M muons, pool v4 train_000): farm GPU 15.0 s; CPU 28 threads (EPYC 9654, farm-gpu5) 1006 s
  (3 600 muons/s/thread) -> ~300 s per seed on a 128-core eiger node (estimate), ~13x one GH200. Not run on eiger (user: verify
  estimates with him first). Optimisation ideas (not done): drop acos/sin/cos in the rotation, cheaper field lookup, SIMD batches.
- Incident: killing hung local ssh clients let old queued commands finish; one resubmitted eiger cv job 8269498 (tasks skipped
  done slices in 2 s; leftover cancelled). v6 inputs checked unchanged (tail_produce.py md5, pythia binary).
- LIVE TALK on the target studies (user, for a colleague): `docs/target_talk/` (target_talk.tex, target_talk.py, run.sh; theme and
  storage widgets from ../live_talk). Data: results/complete_validation, results/pythia_versions, results/storage_policy,
  results/target_talk/top_muons.npz (bin/top_muons_data.py on the farm from the gamma / c / zeta stores; output
  /disk/lhcb_data/sqasim/target_talk/top_muons.npz).
- (Mon ~18:30) THE COMPLETE PRODUCTION IS NOW CALLED `target-production-complete-november-2-2026` (user: refer only to it and to the
  506 M 2018 reference from now on): /capstor/scratch/cscs/sqasim/target-production-complete-november-2-2026/{pool (20 train + 4
  hidden, 2.33 G rows, 121 GB), production (minbias_*, minbias_generator_*, onia_*, charm_*, beauty_*; 300 GB; real files, no
  links), config, logs, README.txt}. pool.json / h5 attrs carry the name and per-component recipe. Old folder v6/ keeps only the
  broken first attempt (log_bad), the old links and MOVED.txt.
  Cells (per ~99 M-row chunk, n_eff): min-bias-dominated cells 5-100x the previous production (300-400 GeV pT 0.75-1.5: 1 600);
  pT 1.5-3 GeV at 100-300 GeV still 12-200 (onia + charm make 62-68 % of the flux there and are not importance-sampled).
  Proposed (awaiting the user): biased onia + charm reruns, pool rebuild, recheck, then daint.
- (Mon ~20:30) USER: only target-production-complete-november-2-2026 and the 506 M 2018 sample from now on.
  * Dataset complete and self-contained: production/ now also has charm_hadrons_*, beauty_hadrons_*, onia_generator_*;
    config/ chi tables; logs per component; README.md (versions, commands, seeds, columns, limitations); code/ (snapshot of
    veto bin/lib/cpp + checksums of container and binaries); histograms/ (bin/compare_new_vs_506M.py, job 8269932).
  * 506 M sample: /disk/lhcb_data/sqasim/reference-2018-fairship-506M/ (was ref2018; README.txt; histograms_506M.npz). Scripts
    updated to the new path.
  * JUNK: eiger $S/junk/{v5,v6,pool1,test1,veto_source_pool1}; farm /disk/lhcb_data/sqasim/junk/{pool_v1..v4 + logs,
    veto_source_pool1}; daint /ritom/scratch/cscs/sqasim/junk/{pool_v1_incoming,pool_v2,pool_v2_ponly,pool_v3,veto_muons_pool_v5}.
    Design datasets (alpha..zeta, c) and their transports untouched.
  * New vs 506 M (2018 selection, production p > 25 GeV): all 1.17; min. bias 1.27 (0.975 with the 2018 weighting); onia
    1.00; charm 0.82; beauty 0.83; eta/omega/rho/phi/eta' parents x2.8; mu+/mu- 1.3 vs 1.4. n_eff: new 10-100x above 250 GeV,
    charm 7.7e5 vs 8.3e6 total.
  * Talk docs/target_talk rewritten (29 pages): no old pools / datasets, no fixes of our own code; new part "The new sample"
    (cmp_1d, cmp_2d, cmp_grp, cmp_charge widgets; data results/target_talk/{new_selected,new_all,histograms_506M}.npz).

## Tue 3 Nov — A-to-Z production being prepared (CSCS cert expired 11:18; user renews)
- Daint debug of the Nov-2 sample (job 5242774): transport fine (30 s / seed), but SST1-selected n_eff per seed 9 (one J/psi
  row 94 %). Causes: onia had one muonic decay per J/psi (few J/psi), min bias rare prompt decays (J/psi, D, rho/omega) one
  full-weight muon each. NOT roulette compounding (checked).
- Fixes (validated on eiger, tpc_fix/test): decay splitting (set_split / --split; onia split 20: 0.998 +- 0.015, n_eff x20;
  min bias roulette + split 10: 1.010 +- 0.008, high-p cells 50-100x n_eff per CPU); roulette rule now a weight window
  (no compounding); new muon window for split weights: F = 0.1 -> 0.9 rows / event (tpc_fix/weight_window_split10_F0.1.npz,
  MoW target: recompute for the FairShip target in the smoke test).
- Found and fixed: cascade quarkonia (gen 7) were dropped by build_pool_v5 (1.1 % of quarkonium muons); Pythia 6 seed
  19780503 + 2 x (1e5 slice + bin) overflowed (> 9e8 limit) for all our charm / beauty slices -> p6_seed = 4000 k + gid;
  Pythia 6 bins now in batches of 100 with retries / halving (no weight lost).
- User decisions: target = FairShip default target_config.yaml (19 W plates, Ta cladding, He gaps, Inconel vessel, steel
  enclosure, copper proximity shielding in a vacuum box; tailtransport target "FairShip"); Pythia 8.223 everywhere (module
  cpp/tailgen_g411_p8223, run with PYTHIA8DATA=$S/pythia8223/share/Pythia8/xmldoc); everything new, nothing reused.
- Also from FairShip for this target: beam = Gaussian 0.8 cm + painting radius 5 cm (tail_produce --prepare-fairship-beam);
  charm fracp 0.40 (W); chicc / chibb x (184/98)^0.29 = 1.200 (heavyFlavourScaling.py); chi tables must be REBUILT
  (charm_cascade --table --target FairShip, charm and beauty) before the production.
- Open for the user: protons per spill 4e13 (FairShip run_fixedTarget default) vs our 5e13 (normalisation only).
- Next (after cert renewal): build module, layout + overlap check, smoke test (FairShip target: min bias / onia / charm /
  beauty, window table, no lost Pythia 6 weight, timing), then the production (~80 node-h) on the user's go.
- (Tue ~15:30) VALIDATION PASSED on the farm (farm-gpu4, /disk/lhcb_data/sqasim/tpc_validation, module
  /disk/lhcb_data/sqasim/tt_g411_p8223 = G4 11.2 + Pythia 8.223 static): FairShip geometry no overlaps, on-axis layout as
  FairShip (144.3 cm W, 5.7 Ta, 25 cm Cu); FairShip / MoW flux 0.795 (pi 0.74, K 0.67, eta/omega 0.98, gamma 1.09);
  onia 4.52e-6 / proton (split 20, n_eff x20); charm x1.19, beauty x1.19 (A scaling), 0 % Pythia 6 weight lost; no errors,
  no Pythia version mismatch. hf_decay now passes the hadron weights as primary weights (full muon weights) and takes
  --muon-window; windows (F = 0.1) in tpc_validation/weight_window_{FairShip_split10,charmc_split10,beautyc_split10}_F0.1.npz:
  min bias 19.6 -> 0.58 rows / event, charm 5.3x and beauty 3.9x fewer rows, flux 1.000, n_eff 99.9 %.
  Production plan (Eiger, ~100 node-h): min bias 6000 x 5e5 (48 nodes), onia 2500 x 2e7 (20), charm 3000 x 1e4 (24),
  beauty 400 x 2.5e4 (4), then pool (--pot 4e13 --tags minbias,onia,charmc_mu,beautyc_mu). Command files:
  ~/.claude/jobs/e59a0386/tmp/tpcprod/. Waiting: CSCS cert renewal + the user's go.

## Wed 4 Nov 00:35 — A-TO-Z PRODUCTION SUBMITTED (user: "yes", ~102 node-h)
- Dataset: /capstor/scratch/cscs/sqasim/target-production-complete-november-4-2026 (tail/ outputs, log/, config/ windows,
  cmds/). FairShip target, Pythia 8.223 (generator + decays: module $S/tt_g411_p8223), Geant4 11.2 FTFP_BERT, 10 GeV,
  beam = FairShip (0.8 cm + 5 cm painting), 4e13 p.o.t. per spill. Job script veto/eiger_tpc.sbatch (VSRC required).
- Jobs: min bias 8274546 (47 tasks, slices 400000-405999 x 5e5, roulette + split 10 + window), onia 8274547 (20 tasks,
  410000-412499 x 2e7, split 20), charm 8274548 (24 tasks, 18000-20999 x 1e4, split 10 + window), beauty 8274549 (4 tasks,
  21000-21399 x 2.5e4); retries (afterany, skip done) 8274550-8274553; pool + 506M histograms 8274554 (veto/eiger_tpcpool.sbatch,
  "POOL DONE" in log/pool_*.out).
- Smoke test passed (tpc_smoke): vs farm validation min bias 0.99, onia 1.06, charm 0.997, beauty 1.001, 0 % lost weight.
- After the pool: README (versions, commands, seeds, code snapshot, checksums), cell check vs Nov-2, comparison with 506M
  (note: compare_new_vs_506M.py uses 5e13 per spill), daint debug, transport. CSCS cert valid to Wed 11:54.
- (Wed 11:30) Production COMPLETE: 6000 minbias + 2500 onia + 3000 charm + 400 beauty slices, 0 failed, 354 GB (Slurm put
  25 tasks on "user env retrieval failed requeued held" -> released by hand + auto-release watcher). Pool job 8274554 TIMED
  OUT at 6 h (4451/11900 files: the single-process writer appended small blocks); build_pool_v5.py now buffers 4 M rows per
  chunk (identical output on the farm test); resubmitted as 8275986 (12 h, started 11:05; ~2.65 G rows -> 23 train + 4
  hidden; expected ~15:00-16:00). TODO: rewrite the builder in parallel (per-worker parts + per-chunk concatenation).
- CSCS scratch is purged after ~30 days: copy everything to the farm. Copy of tail/config/cmds started 11:30 (laptop relay)
  to /disk/lhcb_data/sqasim/target-production-complete-november-4-2026; the pool to follow after it is built.
- (Wed 14:40) BUG FOUND AND FIXED: tailtransport's Pythia decayers ran with Check:event = off and py_decay did not validate
  the result; Pythia 8.223 returns broken B+ / B0 decays (~3e-7, e.g. B+ -> D*- pi+ pi+ pi0 with NaN pi0 photons and a
  3.5 TeV D*-) -> 2 beauty events with 500-870 GeV muons (crashed the daint training seeds: momentum table ends at 400 GeV).
  Standalone check (cpp/tailgen/decay_check.cc): D, eta, omega, rho, phi, eta', J/psi, Upsilon, tau, B_s clean in 9.6e6
  each. Fix: Check:event on + py_decay validates (finite, energy conserved 0.1 %, max 5 retries; diag pythia_retries) +
  T.decay_test(); module $S/tt_g411_p8223_v2 (verified: 9.6e6 B+ / B0 each, 0 bad returned). Charm + beauty DECAYS rerun from
  the existing hadron files (jobs 8276442, 8276443; old outputs in $S/junk/target-production-november-4-2026-first-hf-decays),
  pool rebuild 8276444 (also drops p >= 400 GeV, counted in pool.json), then daint debug (local chain).

## 2026-11-07 (night): final-production plan, tests on the farm
- Plan (artifact https://claude.ai/artifact/BrxHAGdA6ptFhkRnjgw3tt): statistics everywhere on a closed elliptic-arch SHELL around
  the vessel (lib/veto/shell.py: margin 1.5 m, 29.89 -> 94.5 m, 32 x 16 side cells + 8 x 8 caps at 30.6 / 82 m), all muons that
  stop in the last metre of the shield kept (with a ceiling), 1-in-N of the rest.
- Response map (zeta store, /disk/gfs_lhcb/sqasim/shell_response): shell fed by 33-89 GeV sources (78 %), nothing < 20 GeV.
- E1 test (min bias, farm Slurm `hep`, /disk/gfs_lhcb/sqasim/e1_test): A = Nov-4 settings, B = E1 25 no window, C = E1 25 +
  per-cell window (window_percell.npz). Rates unbiased (1.000). C vs A, distinct-muon n_eff: median shell cell 3.8x, last metre
  10.5x, 5.3x CPU, 3.3 vs 0.57 rows per collision.
- KEY: shell statistics are transport-limited (seed-counted / distinct n_eff ~1.1). 10 seeds on ONE Nov-4 chunk
  (/disk/gfs_lhcb/sqasim/nov4_pool/train_000.h5, copied Eiger -> farm directly, md5 OK): heavy rows are different muons every
  seed (top-10 overlap 0 %); distinct n_eff grows ~linearly (9x at 10 seeds) except SST1 selected (4.2x: saturates ~10 seeds per
  chunk). => The Nov-4 production is usable; what is needed is transport with ~10-20 seeds per chunk + shell storage. New
  production only where the full shell map shows holes (likely last metre).
- NEW transport storage `--storage shell` (bin/generate_muon_study.py): types 0-3 + last-metre stops kept (cap --keep-cap per
  seed), shell-steered rows (--shell-rows-per-seed), floor per type; PLAIN Horvitz-Thompson weights (the per-type rescaling of
  the other storages gave a coherent +4 %). Check: bin/shell_check.py <dir>. 3 seeds: bias 1.002, pull 0.00, no n_eff lost,
  ~240 k rows / 28 MB per seed.
- Farm notes: Slurm only from farm-gpu3/4/5 (not farm-ui1: no munge); batch jobs must call apptainer directly (the shared exec
  script binds /run/user/<uid>); containers need -B /disk/gfs_lhcb now that data/ dirs are symlinks there; transport runs in
  /disk/users/lprate/containers/MuonsAndMatterContainer.sif (Python 3.10); field map copied to Veto_System/MainSpectrometerField.h5.
  generate_muon_study.py now passes pdg to the classifier (Guglielmo's field-map classify needs the charge).
- Disk move done 2026-11-06: 1.5 TB from /disk/users to /disk/gfs_lhcb (symlinks), caches cleaned (56 GB).
- Merged Gfrisella/Veto_System main locally (fast-forward + our uncommitted work on top; nothing committed).
- Next: copy the full Nov-4 pool (136 GB) to the farm; transport 27 chunks x ~20 seeds with --storage shell on farm-gpu3
  (~7.5 GPU-h, ~15 GB); shell map -> decide on a targeted production.

## 2026-11-07 (evening): full-sim inventory, cavern-wall census
- `docs/full_sim_inventory.md`: what we have / miss per source (muons, neutrinos), user's decisions (+x wall dropped for
  now, new muons made in the shield parked as rare, Mylar kept, cavern-wall DIS and a new neutrino sample at the target
  to do), plans for both.
- 8-GPU shell extension DONE: 3,220 train + 560 hidden seeds (`/disk/gfs_lhcb/sqasim/nov4_pool/shell/`, 100 GB).
- `bin/rock_hits.py` (300 seeds): 1.5e9 muons / spill go into the wall, 8.9e7 into the floor; only ~1.2e5 / spill come
  back into the cavern. CORRECTION: the earlier "wall bounces 3.3e8 / spill" were muons stopping in the rock (fake bounce
  point from the stop state); type 4 rock_bounced is 99 % muons inside the rock at 82 m. Type 0 unaffected.
- Reco display tab "Veto reco" on https://claude.ai/artifact/8ttnXaQfgCcwR3CyvbvGpy (bounce-search windows ~ +-7.5 m).
- Raw Nov-4 production (354 GB) still ONLY on CSCS scratch (purge ~Dec 4): copy pending the user's go.

## 2026-11-08: cavern-wall DIS (ISSUES D7)
- Muon flow per spill (4e13 p.o.t.): 4.69e10 from the target, 4.38e10 stop in the shield, 3.11e9 reach 30.6 m; of those
  1.46e9 go into the -x wall (mu+), 9.1e7 into the floor, 1.39e9 fly past (mostly mu- to +x, where the model has no wall),
  ~3e4 reach SST1. 15 years: 2.4e16 muons into the rock.
- `cpp/g4rock.cc` (Geant4 half-space of G4_CONCRETE: forced muonNuclear, per-event depths, injection of given final states,
  trajectory recording) + `bin/rock_dis.py` (scan, viz). Page: https://claude.ai/artifact/7e4c5hkyvZy1cupHhpqNGv.
  Charged hadrons escape only from the top ~30 cm; neutrons / photons from ~1 m.
- Library `bin/rock_library.py` (+ `bin/rock_library.sbatch`, job 2021285 on hep, 224 tasks): Geant4 part (nu < nu_x) +
  Pythia 6 part (nu >= nu_x, from mu_library.npz) injected and transported; output /disk/lhcb_data/sqasim/rock_dis/library/.
  `bin/rock_apply.py` applies it to the shell seeds (planes, vessel, SST1). Partial-library test: 1.6e7 interactions / spill
  in the top 3 m, ~9e3 charged p > 1 GeV / spill at SST1 (preliminary).
- Decided next (user, 2026-11-08): direct forced Geant4 from the stored 30.6 m states in the real cavern (wall + floor,
  G4BOptrForceCollision, forced K_L / neutral decays in the vessel), storing design-independent trees to SST1, mapped onto
  designs afterwards (straight lines or ParticleShaker). The library stays as the quick estimate and cross-check.

## 2026-11-09 Neutrino production with the new target (DONE)
- Data: /disk/lhcb_data/sqasim/nu_prod/tail/ — nuMB_nu_600000-600499 (500 x 2e4 p, 3.3 GB), charmc_nu_30000-30999
  (1000 x 1e4 p, charm hadrons > 1 GeV, keep 0.5, 1.7 GB), beautyc_nu_21000-21399 (400 x 2.5e4 p from the Nov-4 beauty
  hadrons, keep 0.05, 210 MB), charmc_had_* (soft charm hadrons, 5.7 GB), nu_flux_summary.npz ((E, theta) histograms per
  proton per component and flavour). Per proton = sum w_decay / (events x rep), per component; drop hf > 0 rows in nuMB.
  Cuts: E >= 1 GeV, theta <= 0.3 rad, made upstream of z = -0.0045 m. Summary: bin/nu_flux.py <dir> [--had-old <nov4 dir>].
- Per proton (per spill 4e13): nu_mu 5.50e-3 (2.2e11; charm 4.6 %), anti nu_mu 3.94e-3 (1.6e11), nu_e 5.55e-4 (2.2e10;
  charm 46 %), anti nu_e 5.29e-4 (2.1e10; charm 52 %), nu_tau 3.90e-5 and anti nu_tau 3.95e-5 (1.6e9 each, all charm, D_s);
  beauty ~5e-8 per flavour. Min-bias nu_e: K+- 1.3e-4, K_L 9.8e-5, K_S 6.7e-5. Made at z -2.2 .. -1.4 m.
- Soft charm vs Nov-4 > 10 GeV set above 10 GeV: ratio 1.0000 (pull -0.17; D0, D+, Ds, Lambda_c all 1.000); the soft part adds
  x1.16 to the charm hadron weight.
- Fixes during the run: charm_cascade Pythia 6 scratch files now on node-local /tmp (112 slices on GFS stalled in D state);
  nu_prod.sbatch charm step skips charm_cascade if charmc_had exists.
- Next: GENIE / neutrino interactions in the detector and surroundings from nu_flux_summary.npz or the row files.

### 2026-11-09 Neutrino DIS apply on the 2026 flux + flux validation
- Apply done (library v2, one process per flavour): D6_straw_mod, 15 y, all flavours: helium 2.87e4, air 6.57e4, PVC 5.77e4,
  UBT + fins 5.8e3 (`data/interactions/genie/apply2026_D6_straw_mod_nu*.json`). D6_straw nu_mu / anti-nu_mu also done.
- vs 2018 flux: x3.7 (helium) to x31 (fins). Not a bug: the 2018 sample (neutrinos from Guglielmo's `_mu` files) only kept
  neutrinos with p > 10 GeV whose sister muon was stored, and that muon has p >= 15 GeV (5 % quantile 21 GeV). Above 10 GeV
  pion-parent nu_mu agree (new/old 0.94, 1.01 at 10-20 GeV); K+- (x6.5, x9.8 at 10-20 GeV falling to x1.1 > 100 GeV) and
  charm (x3.7-5.5) are what the muon cut removed. Plus: no E < 10 GeV (65 % of the sigma-weighted new flux), off-axis spread.
- Independent check vs SHiP 2025 (arXiv 2504.06692, Table 1, CC DIS in 3 t W, 15 y): 2026 flux folded on a 40-50 cm W block at
  z = 26-30 m gives nu_mu 3.8-5.2e6 (SHiP 5.8e6), anti-nu_mu 1.1-1.5e6 (1.3e6), nu_e 1.6-2.2e6 (2.0e6), anti-nu_e 4.3-5.7e5
  (4.0e5), nu_tau ~1e5 (5.9e4; no tau threshold suppression in the estimate). The 2018 sample gives nu_mu 2.0-2.9e6, nu_tau 4e3.
  Scripts: /disk/lhcb_data/sqasim/nu_lib/{cmp,par,snd}.py.
- Selection + veto (bin/run_interactions.py --kind nu, new --pot-spill 4e13; current RecoConfig) on the 2026 samples, per 15 y,
  after veto level b: D6_straw_mod nu_mu 1.02, anti-nu_mu 0.51, nu_e 1.37, anti-nu_e 0.39, nu_tau 0.02, anti-nu_tau 0.02 ->
  total 3.3 (selected 3.9), all but ~0.07 from helium (not vetoable: a ~ b ~ selected). Air / PVC: 0-1 selected draws each
  (upper limit ~0.1-0.2 per flavour). Like for like: the 2018 sample through today's code gives 1.03 selected / 0.89 after veto
  (the 469 in results/interactions/D6_straw_nu.json is from the pre-Oct-21 selection, stale). New D6_straw nu_mu + anti: 1.58.
  So the neutrino background is x1.8 (muon flavours) and x3.7 with nu_e, not x4-30. Outputs:
  /disk/lhcb_data/sqasim/workspace/uzh/Veto_System/veto/results/interactions/*_nu2026_nu*.json, D6_straw_nu2018_rerun.json.
  Run with EXEC="... exec_cudabear... env VETO_RESULTS=/disk/lhcb_data/.../results VETO_INTERACTIONS=..." (gfs not mounted).
- Cavern rock through the same selection (new bin/rock_select.py: rock_apply / nu_rock draws -> interactions.evaluate, cut flow):
  D6_straw_mod, 0 selected for both. Neutrino rock: 3.2e3 / 15 y with >= 2 good tracks (84 draws), best-pair measured IP
  >= 6.8 m. Muon rock: 4.6e9 / 15 y with >= 2 good tracks (64 draws, 7e7 each), IP >= 12.6 m. Vertices are killed by geometry.
  BUT single tracks: the muon rock puts 1.84e4 good tracks (p > 1 GeV, SST1 + timing acceptance) per spill into SST1, mostly
  hadrons / electrons (muons 300), vs ~3e4 direct muons -> feeds the combinatorial background (rate^2), not included yet.
  Neutrino rock: 0.024 per spill (negligible).
- Rock tracks in the combinatorial term (bin/rock_combinatorial.py, D6_straw_mod, dataset c, inputs_c_s, cone 0, 2M-muon
  subsample x scale): per 15 y, candidates / IP < 10 cm / after veto: mu x mu 852 / 405 / 51 (legs explained 0.74 / 0.64);
  mu x rock 8190 / 2927 / 0.023 (662 MC; rock leg explained 0.9999: rock tracks cross the walls / planes on the way in);
  rock x rock 5e-7 / 0 / 0. Rock raises the candidates x8 but adds ~nothing after the veto. NOTE mu x mu after veto = 51 for
  D6_straw_mod vs ~1e-4 for D6_tile2 / E3 / E4 (e2e_c): this straw design leaves 26-36 % of pair legs unexplained - check.
  Run time 48 min (pairing 1.3e6 x 6.4e3 rows on farm-gpu3). Full dataset c (262 M SST1 rows) is too big to load: subsample.

## 2026-11-09: GSimple 2026 final-target neutrino flux
- Converted A. Iuliano's GSimple flux (EOS, 16.1 M rows, 1.46e9 PoT) to h5 in the veto frame (FS z - 2.14), per-spill weights:
  `Study_on_Veto_System/Neutrino_flux_gsimple2026/` (README there). Compared with Shah Rukh's Nov-9 production and fastveto.neutrinos (2018):
  nu_mu agrees with Shah Rukh to 1-4 %, nu_e -20 %, nu_tau x0.55; the 2018 flux used by the veto is x2-2.7 low for nu_mu > 10 GeV.

## 2026-11-09 (evening): Guglielmo's veto side (merged here; open problems in ISSUES.txt, code log in CHANGELOG.md)
- Branch `veto-shield-dis`, commits d2c867b + issue-review commit. Nothing running at the end of the session.
- D3: natural Geant4 (bin/shield_natural.py, cpp/g4shieldnat) validates the Geant4 MS7 chain (K_L 3.5e3 / spill at the
  exit vs 3.4e3); Pythia 6 x 2.7 high. Still 7.9 expected FairShip entries vs 0. Decision: Geant4 default; next a small
  FairShip run with our MS7 parents. Table results/shield_dis/generators_natural.md.
- Issue review with decisions for every open item: action list at the top of ISSUES.txt. Pending Guglielmo: approval
  of the D5 convention check and of the re-run batch (after D5 / D3), which neutrino flux to adopt.
- Questions for the detector colleagues: docs/detector_questions.md (to send).
- D3 deck to update with the natural-Geant4 result.
- Neutrino DIS in MS7 (last 50 cm) with the 2026 flux + GENIE v2, all 6 flavours (bin/shield_dis.py generate --kind nu
  --n 500000 --v0-copies 100 --nu-stores .../neutrinos/store2026; new options --nu-lib / --nu-stores, defaults v2 + 6 flavours):
  1.34 interactions / spill (2.0e7 / 15 y), V0 decays in the vessel 2.1e5 / 15 y; D6_straw_mod selected 3.9 +- 0.5, after veto
  b 2.4 +- 0.4 per 15 y (108 MC copies; 1 copy: 9.2 / 5.4 from 2 MC). ~1e-4 of the muon MS7 K_L, comparable to nu in helium.
- MS7 muon DIS -> K_L with Geant4 (2026-11-10, chosen default; Pythia 6 ~x2.7 higher noted as the generator spread):
  Slurm 2035587, 10 x 2e6 interactions, Geant4 11.2.2 library (Guglielmo's mu_library_iron.npz, split geant4), his MS7
  parents, 5 copies / V0 (results/shield_dis/shield_muon_g4v112_s101-110.npz). V0 decays in the vessel 2.33e10 / 15 y
  (K_L 1540 / spill). D6_straw_mod full selection: 0 in each file. Passing everything but the IP cut 1.4e7 / 15 y; measured
  IP < 2 / 1 / 0.5 / 0.25 / 0.1 m: 8.1e6 / 2.1e6 / 3.8e5 / 7.6e4 / 1.8e4 (356 / 98 / 18 / 3 / 1 copies). ~IP^2 near 0 ->
  **~1.5e4 per 15 y (range 1-2e4)**, not vetoable (bin/shield_ip_tail.py). Geant4 10.7.3 (FairShip's) run: 0 selected,
  consistent. Selection step segfaults in the batch container (external/pylib + torch): run it with exec_cudabear.

## 2026-11-10 Event store (lib/veto/evstore.py) and MS7 on the Nov-4 muons
- Design-independent stores, one directory per process: events (weight / spill, code, parent row, vertex, time, forced-copy
  link) + tracks (tree: parent track, pdg, start, momentum, end, end reason, creator, beam). Downstream model: straight
  lines, decays in flight (pi+-, K+-, K_L, K_S, Lambda, Sigma, Xi, pi0) until the cavern wall / z = 95 m; state at SST1.
  bin/evstore_build.py (shield / apply / rock), bin/evstore_eval.py (any design, --ip-tail), rock_apply --store-out.
  Stores in /disk/lhcb_data/sqasim/evstore/: ms7_muon_g4v112 (20.5 M events), ms7_muon_g4v112_nov4, ms7_nu2026,
  vessel_nu2026_D6_straw_mod_nu{14,-14,12,-12,16,-16}, rock_muon (300 spills, every exit, 2.2 M events).
- Checks vs the old path (D6_straw_mod, per 15 y): MS7 muon tail identical within statistics; MS7 nu 2.2 (was 2.4); vessel nu
  3.0 (was 3.3: pi / K now decay before SST1); rock 0 (0).
- MS7 parents from our own transport (bin/cpu_transport.py run --planes 29.2 30.6 --keep-box 2 1.6 on GPU 0 of farm-gpu3,
  Nov-4 pool train_000-003, 8 s per chunk, MuonsAndMatter container) -> shield_dis prepare --from-transport -> 
  /disk/lhcb_data/sqasim/ms7_parents/muons_iron_nov4.npz: 3.95e7 rows, 4.62e8 / spill through the iron vs Guglielmo's 2018
  one-run file 3.89e8 (x1.19 in every p bin < 100 GeV, x0.96 > 200 GeV; same footprint and chord; D5 source difference).
- MS7 chain on them (Slurm 2036253, 10 x 2e6, Geant4 11.2): V0 decays in the vessel 2.69e10 / 15 y; passing all but IP 1.95e7;
  IP < 2 / 1 / 0.5 / 0.25 / 0.1 m: 1.1e7 / 2.2e6 / 3.7e5 / 6.1e4 / 0 -> ~1.5e4 per 15 y (1-2e4) by IP^2: the stored number.
- 2026-11-10 night, remaining compute (farm-gpu3, /disk/lhcb_data/sqasim/evstore/remaining.log, logs/):
  - Nov-4 dataset store (bin/shell_to_store.py, 300 shell seeds): design-optimization-dataset-nov4-shell-300seeds-2026-11-10,
    7.17e7 rows, 4.72e10 muons / spill (pool 4.69e10). Pair inputs from it: results/reco/inputs_nov4.npz (21 097 candidates,
    5 739 pairs / spill before the time coincidence).
  - **Correction**: muon DIS in the HELIUM is NOT small. D6_straw_mod on Nov-4 (run_interactions --kind muon): helium 1.82e3
    selected -> 472 after veto b (673 at truth level c: the parent crossed no plane), walls 176 -> 44 per 15 y; e2e_c
    (dataset c) had 210-250 (helium) for D6_tile2 / E3 / E4 / E5. Second background after MS7 K_L; side-entry coverage gap.
    Store: vessel_mu_nov4_D6_straw_mod (7.0e6 events).
  - MS7 K_L direct count (bin/evstore_redecay.py, 100 re-decays of the 14 168 pre-selected V0 = 68 % of the all-but-IP
    weight): 1.39e4 selected (13 MC), 9.7e3 after veto; all but IP: IP < 0.1 / 0.5 / 1 m 2.1e4 / 4.2e5 / 1.8e6 (17 / 328 /
    1458 MC). Confirms the IP^2 extrapolation (~1.5e4).
  - Geant4 downstream (g4vessel on the stores' roots, g4_vessel_transport + evstore_eval --g4): MS7 muon all-but-IP 1.44e7
    (straight 1.95e7), IP < 1 m 2.0e6; MS7 nu 1.0 after veto (straight 2.2); vessel nu after veto: anti-nu_mu 0.16 (0.43),
    nu_e 0.59 (1.30), anti-nu_e 0.15 (0.37) + walls 0.03-0.10; nu_mu, nu_tau, anti-nu_tau rerunning (2 particles produced
    beyond z = 94 m made g4vessel stop: now skipped, bin/g4_vessel_transport.py).
  - EM debris test on the Nov-4 parents (em_debris.py --n 2000 --tag _nov4_test): 10 s for 2000 muons (16 workers), 1.23e9
    e / gamma per spill at the exit plane -> a 2e5-muon run ~ 20 min, 2e6 ~ 3 h: a few core-hours, not tens.
  - Neutrino-rock store rock_nu2026 (412k events).
  - Geant4 downstream, all 6 vessel-nu flavours done (2-3 particles produced beyond z = 94 m are now removed before g4vessel):
    after veto b, helium / walls: nu_mu 0.42 / 0.06, anti-nu_mu 0.16 / 0.10, nu_e 0.59 / 0.02, anti-nu_e 0.15 / 0.03, nu_tau 0.005
    / 0.006, anti-nu_tau 0.001 / 0.001 -> 1.3 + 0.2 = **1.5 per 15 y** (straight lines 3.0). With MS7 nu (1.0): neutrinos ~2.5.
  - EM debris on the Nov-4 parents (2e5 muons, 30 s): 1.19e9 e / gamma per spill out of MS7, UBT 5.2e6 e + 2.5e7 gamma per
    spill; maps results/em_debris/em_maps.npz. run_reco D6_straw_mod on inputs_nov4: combinatorial 2.86 -> 0.014 (0.003 with
    debris), signal lost 6.4 % -> 7.5 % with debris. CHECK: inputs_nov4 has x140 fewer pair candidates than inputs_c_s (2.86
    vs 405 / 15 y); the "51" of rock_combinatorial on dataset c is probably an old-layout (pre-5-Nov) vs fairship-2026 design
    mismatch. Not resolved.
    RESOLVED (2026-11-10 day): not the layout. Dataset c has 3-4x more SST1 muons above 30-100 GeV near the axis (source
    pool, D5); the pair rate goes as the square, times the pointing preference, times the old 5e13 normalisation of
    inputs_c_s. Reclassifying c with the new layout raises its selected class. Which tail is right: compare the pool
    v2/v3 and Nov-4 input spectra > 100 GeV by component (open).
- Neutrino DIS in MS7 (last 50 cm) with the 2026 flux + GENIE v2, all 6 flavours (bin/shield_dis.py generate --kind nu
  --n 500000 --v0-copies 100 --nu-stores .../neutrinos/store2026; new options --nu-lib / --nu-stores, defaults v2 + 6 flavours):
  1.34 interactions / spill (2.0e7 / 15 y), V0 decays in the vessel 2.1e5 / 15 y; D6_straw_mod selected 3.9, after veto b 2.4
  per 15 y (108 MC copies; 1 copy: 9.2 / 5.4 from 2 MC). ~1e-4 of the muon MS7 K_L, but comparable to nu in helium (3.3).

## 2026-11-10 (day): statistics, storage, rare corners
- Statistics for a 15-year design number (D6_straw_mod): MS7 K_L lacking (13 MC selected, ~1e3 per MC event: K_L-enriched
  generation needed); walls lacking (22 per MC event, cheap); helium OK now (x20 later); neutrinos sufficient; combinatorial
  limited by the muon production (spills), not by transport, but enough for the budget.
- Storage: evstore 71 GB (Geant4 files 19 GB), shell seeds 106 GB, Nov-4 pool h5 145 GB, dataset c 60 GB, Nov-4 store 8.3 GB,
  rock apply 15.7 GB, MS7 parents 10.8 GB; MS7 shield_dis 5.7 GB / 10 files, debris 0.4 GB.
- Tail extension (answer, nothing run): tail_produce --mode complete with a high EMIN + an energy split in build_pool_v5,
  ~1-2 days of work + 1-5 k core-h; first decide which tail is right (above).
- Rare corners (user: rare places where more statistics blindly does not help): step 1 = worst-case bounds from the
  stores, bin/coverage_bound.py (store --kind vessel|ms7|rock: per region of the interaction point, MC count, selected /
  loose, 2.3 x mean weight limits for regions with none selected; rock: neutral-pointing proxy; rockflux: rock patches from
  the muons; shell: analog counts per shell cell). Note: the rock "0" is 2.2 M draws for 1.57e14 interactions / 15 y,
  ~7e7 per MC event: statistically only < 1.6e8; it rests on geometry (a vertex must point back within 0.1 m). Running.
  Later steps if a corner is not bounded: splitting / weight windows in the shield transport (splitting_proto exists on
  lhcb_data), forced neutral chains for rock and walls.
- Rare-corner bounds, results (D6_straw_mod, per 15 y, /disk/lhcb_data/sqasim/coverage/*.json; selection = pre-veto):
  - Muon flux around the vessel (shell_hist, 3 220 spills): every side cell < 82 m has >= 635 MC muons; the 8 empty cap
    cells are outside the ellipse. Rock patches from the muons (rockflux): every reachable patch has muons, an empty one
    < 0.02 interactions. -> the flux has no holes; the weak points are draws and responses.
  - Helium: 279 / 800 regions with a selected event, joint limit of the rest 2.9, IP-extrapolated share 125 (vs 1.78e3
    observed). OK. Vessel walls: 12 / 384 regions (13 MC), IP-extrapolated share of the empty ones ~200 vs 286 observed:
    under-sampled. MS7 K_L (redecay100): 9 / 73 regions, empty ones ~0.9e4 by IP -> MS7 1.4-2.3e4.
  - Air, plane interactions: large statistical limits (1e2-1e6) but bounded by physics (products enter through the SBT;
    plane interactions vetoed by the parent's hit).
  - Rock: the "0" is geometric. K_L the draws make (631 with p > 2 GeV, wall z 55-75 m) are >= 134 mrad off any
    target-pointing line; the visible pair of K_L -> pi l nu with both tracks > 1 GeV turns by <= ~120 mrad: closed.
    Geometry map (no data): the window is OPEN from the floor (44-90 mrad up to z ~ 70 m) and the wall upstream of
    z ~ 52 m, for K_L of 2-6 GeV, where the draws have almost nothing (floor 25.6k draws, 3 K_L).
    Synthetic K_L there (klmap --window, 2.0e6 aimed into the pointing window, real selection): IP < 1 m 17 MC,
    < 0.25 m 1, < 0.1 m 0; per K_L into the vessel P(IP < 1 m) ~ 1.6e-6 (floor z 30, x 0) falling to ~5e-8 at z 50.
    Folded (K_L p > 2 GeV yield 2.9e-4 per interaction, ~0.5 into the vessel, floor interactions z 30-60 m 6.5e11 with
    exits, spectrum factor 2, IP^2 below 1 m): floor ~0.25 per 15 y (0.1-3 with linear IP scaling), upstream wall ~0.1;
    before the veto, and the veto sees nothing for a K_L-only event. Needs a targeted run to become a number.
  - GAP: neutrons (and K_L hadronic interactions) from the rock interacting in the helium are in no chain: the rock store
    propagates neutrals straight, no Geant4 downstream for rock. (MS7 / vessel stores have it via g4_vessel_transport.)
- Hole 2, neutral hadrons interacting in the helium (new: cpp/g4hadlib, bin/neutral_library.py -> /disk/lhcb_data/sqasim/
  neutral_lib/neutral_he.npz, 3 species x 24 E x 20k, lambda_I(n-He) 3.7-4.1 km; bin/evstore_neutral.py, code 7):
  rock store 5.4e8 interactions / 15 y in the helium (5 524 n, 30 nbar, 236 K_L roots > 2 GeV), 289k copies: 0 selected,
  0 all-but-IP (soft: median 2.6 GeV; the 2.5k copies with 2 tracks in SST1 have vertices at z 77-81 m).
- Hole 1, targeted rock draws (rock_apply --region corner: floor z < 70 m, wall z < 52 m; 1.71 M draws, floor 317k = x12):
  K_L > 2 GeV 3-4e-5 per interaction (x10 below the wall-z60 assumption), none from the floor heads to the vessel; with
  klmap: floor < 0.02 / 15 y (<= 0.2 with linear IP scaling), upstream wall ~0.002. Neutrals in the helium from the corner
  9.6e6 / 15 y, 0 selected, 0 all-but-IP. CLOSED.
- Neutron map (klmap --species 2112 --window, 3.3e6 synthetic n, library interactions): 0 selected, IP < 1 m 14 MC, < 0.25 m
  1. P(IP < 1 m | interaction) 8e-6 (floor z 30) ... 5e-7 (wall z 55), 0 at z >= 60 (where most rock neutrons come from).
  Folded: 5.4e8 x <= 5e-7 x 1/100 (IP^2 to 0.1 m) x 0.02 (real spectrum: 99 % < 4.9 GeV; two tracks > 1 GeV need ~5 GeV+)
  -> rock neutrons ~0.05 per 15 y (0.01-0.5); 2.7 if the spectrum softness is ignored (upper end). No veto for these.

## 2026-11-10 (Guglielmo) V0 production in the MS7 muon chain vs data (branch veto-v0-production)
- bin/v0_production.py -> results/v0_production/README.md. Made at the vertex (Geant4 muonNuclear, V0-stable library, PR #15
  vertex spectrum): per spill K_L 7.9e3, K_S 7.8e3, Lambda 5.4e3, anti-Lambda 260 (~1 % per interaction: median W 1.3 GeV).
- Data check at W = 6.2 GeV (gamma p 20 GeV, SLAC SHF): K0_S 0.087 vs 0.082 +- 0.003, Lambda 0.051 vs 0.049 +- 0.002;
  Pythia 6 x1.9 high -> supports Geant4 as default (D3); the Pythia "x2.7" spread looks one-sided.
- Exit plane: K_L 2.9e3 / spill = 0.40 of made; primaries alone give 0.25-0.31 -> 25-35 % cascade secondaries. K_S / Lambda
  at the plane all from the cascade (vertex ones decayed, D9). Not done: full V0 tally in the iron (needs g4shield change + run).
- 10 Nov (later): V0 tally in g4shield + V0-stable chain (Geant4 11.4.2 pixi; 11.4 = 11.2 within 1-3 % for K_L): 1.5e4 K_L
  made per spill (half in the shower), exit K_L 37 % shower; K_S / Lambda kept at the vertex: vessel decays x8-30 but none
  passes the selection (< 1e5 / 15 y each before IP vs 1.7e7 K_L). Low W: Geant4 >= x2-3 low but only ~8 % of the relevant
  K_L -> +8-15 %. K_L -> pi+ pi- added. docs/v0_production.md slides 4-6, ISSUES D16.

## 2026-11-11 (Guglielmo) Nov-4 production for everything (branch veto-nov4-source)
- Decision: the Nov-4 complete target production is the muon source for every study, D5 included (ISSUES D5 note 11 Nov).
  No enriched tail -> D5's convention / angle mismatch gone; "which tail is right" (dataset c vs Nov-4) -> Nov-4.
- Code defaults switched via lib/veto/paths.py (MUON_DATASET, RECO_INPUTS, RATE_MAPS, ms7_parents, nov4_pool_pattern), see
  CHANGELOG 2026-11-11. Checked: store 4.72e10 mu / spill, 3.22e4 / spill reaching SST1 (types 0-2), MS7 parents 3.95e7 rows
  4.62e8 / spill, both recorded at 4e13 (no rescale). Nothing run.
- Branch veto-nov4-source (worktree .claude/worktrees/nov4-source): commit 48a0f36 = the V0 work staged on veto-v0-production,
  then the switch. The main checkout is unchanged except the store symlink and the inputs_nov4.npz copy.
- Data locations (Guglielmo's list, 11 Nov; all in lib/veto/paths.py, all checked to exist):
  Muons: source (PDG +-13) /disk/lhcb_data/sqasim/nov4_pool/train_000-022.h5 + hidden_000-003.h5 (nov4_pool_pattern("train" |
  "hidden")); after the shield transport, one file per spill: nov4_pool/shell/train/ (3221 files) + shell/hidden/ (561) +
  shell/summary.txt (NOV4_SHELL_DIR); design-optimisation store /disk/lhcb_data/sqasim/workspace/uzh/Veto_System/veto/data/
  store/design-optimization-dataset-nov4-shell-300seeds-2026-11-10 (MUON_DATASET_DIR, store_dir(MUON_DATASET) reads it there;
  identical to the /disk/users/sqasim copy); through the shield end /disk/lhcb_data/sqasim/ms7_parents/muons_iron_nov4.npz.
  Neutrinos: flux stores .../veto/data/interactions/neutrinos/store2026_nu{12,-12,14,-14,16,-16} (NU_FLUX_PREFIX; per spill at
  4e13: nu_mu 1.33e11, anti-nu_mu 1.06e11, nu_e 1.52e10, anti-nu_e 1.51e10, nu_tau / anti 1.28e9; same production as our
  data/interactions/neutrinos2026 copy, plus E / comp columns); raw /disk/lhcb_data/sqasim/nu_prod/tail/ (NU_PROD_RAW); GENIE in
  the vessel .../veto/data/interactions/genie/apply2026_<design>_nu<pdg>.npz (nu_apply_2026; run_interactions --kind nu --nu-pdg).
  Event stores: /disk/lhcb_data/sqasim/evstore/<process> (EVSTORE_DIR; evstore_eval --store accepts bare names).
  Defaults now: shield_dis and nu_dis_apply read the store2026 flux, nu_dis_apply runs all 6 flavours.
- Next (re-run batch, needs approval): rate maps sample_nov4 (make_design_explorer) and peak maps from the store (make_peak_maps
  to adapt), then the step-3 code items (D15, D9, C1, C7, A2, D12, B4, D14) and the batch of action list step 5.

## 2026-11-11 Muon charge-sign fix
- BUG: every stored muon dataset (alpha .. nov4, sst1/ubt_TRY6) has pdg = charge sign (+1 = mu+; generate_muon_study writes
  sign(charge); verified on Nov-4 seed_0 against the pool by pool_index: all +1 rows are -13, and in every store -1 sits at
  +x, +1 at -x). muon_types.muon_charge, muon_types_torch, selection, make_reco_inputs, make_crossrun_pairs, rock_combinatorial,
  visualize_full_dataset, sst1_bounce_categories read it as PDG (q = -sign) and fastveto.dis gave the scattered muon the
  opposite charge -> timing acceptance with the wrong bending, wrong scattered-muon charge. Fix: muon_types.charge_from_pdg_column
  (+-13 PDG, +-1 charge sign) everywhere. Not affected: rock_apply, em_debris, shield_dis / MS7 (true q), neutrinos, cuda-muons.
- Rerun on Nov-4, D6_straw_mod (evstore/fixq.sh, logs/q_*.log), old -> new:
  pair candidates 5 739 -> 6 005 / spill; combinatorial 0.014 +- 0.010 -> 0.0038 +- 0.0009 (with debris 0.0028 -> 0.0031);
  signal lost 6.37 -> 6.36 % (debris 7.46 -> 7.30 %); muon DIS helium selected 1.82e3 -> 1.73e3, after veto 472 -> 520,
  truth 673 -> 657; walls after veto 44 -> 22 (8 MC selected: statistics). New store evstore/vessel_mu_nov4q_D6_straw_mod,
  inputs results/reco/inputs_nov4_q.npz, reco_nov4q_D6_straw_mod(_em).json. Older datasets: same bug, rerun if reused.
- Regression check bin/check_charge_convention.py (helper, seed rows vs the source pool by pool_index, stores by the side
  each charge leaves on): Nov-4 seed_0 239 470 rows 0 mismatches; Nov-4, c, gamma stores OK. Convention now also in the
  metadata of new seed files and stores (pdg_convention).

## 2026-11-11 (Guglielmo) merge of PR #16 into veto-nov4-source
- origin/main (PR #16) merged; overlap with our work = the charge fix (our ISSUES D17, found in parallel): theirs kept.
  Completed on our side: rock_combinatorial "mu x rock" opposite-charge test (was same-sign), paths.RECO_INPUTS ->
  inputs_nov4_q.npz. Old D9 (dis_comb_pairs*, out of repo) numbers void (same inversion).
- Still to decide with Guglielmo: the 5 questions of docs/plan_evolution_sbt_shieldend.md; issue review continues
  (D8, D15, D9, C1 pending; then C7, A2, D12, B4, D4, C9). C5 store rebuild from 3220 spills: coordinate the new store
  name with Shah Rukh (his plan uses the 300-seed store).

