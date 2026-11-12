#!/bin/bash
# Environment of the re-run batch (12 Nov 2026). Sourced by bin/batch/task.sbatch and submit_rerun.sh.
# Code: this checkout (branch veto-nov4-source). Data and results: the main veto checkout on /disk/gfs_lhcb.
export VETO_CODE=${VETO_CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export VETO_MAIN=${VETO_MAIN:-/disk/gfs_lhcb/gfrise/Study_on_Veto_System/Optimization_Veto_System/veto}
export VETO_STORE=${VETO_STORE:-$VETO_MAIN/data/store}
export VETO_INTERACTIONS=${VETO_INTERACTIONS:-$VETO_MAIN/data/interactions}
export VETO_SIGNAL_STORE=${VETO_SIGNAL_STORE:-$VETO_MAIN/data/signal/eventcalc_dcdcfc1}      # B4: regenerated set
export VETO_RESULTS=${VETO_RESULTS:-$VETO_MAIN/results_rerun}        # NEW results root: nothing old is overwritten
# the full Nov-4 production (3220 spills, C5) and the inputs built from it in this batch
export VETO_MUON_DATASET=${VETO_MUON_DATASET:-design-optimization-dataset-nov4-shell-3220seeds-2026-11-11}
export VETO_RECO_INPUTS=${VETO_RECO_INPUTS:-inputs_nov4full.npz}
export VETO_RATE_MAPS=${VETO_RATE_MAPS:-sample_nov4full.npz}
export DIS_PARENTS=${DIS_PARENTS:-dis-parents-$VETO_MUON_DATASET-types0123}
export MU_LIB_V0STABLE=${MU_LIB_V0STABLE:-$VETO_INTERACTIONS/dis/mu_library_iron_v0stable.npz}
export ROCK_MU=${ROCK_MU:-/disk/lhcb_data/sqasim/rock_dis/apply_D6_straw_mod_draws.npz}
export DESIGNS=${DESIGNS:-"designs/R1_ubt_only.json"}   # reference designs (12 Nov 2026): R1 UBT only; R2 SBT + UBT once the SBT model exists
export G4SHIELD_BUILD=${G4SHIELD_BUILD:-$VETO_MAIN/cpp/build_ship}
# Thread limits (12 Nov 2026, farm-wn81 load 1300 on 80 cores: every numpy/OpenBLAS call opened one thread per node
# core in each of the 32 Pool workers). Slurm does not pin cores here, so the job must stay inside its allocation itself:
# a command that parallelises with processes (--workers / --jobs) gets 1 BLAS/OpenMP thread per process, any other
# command gets the CPUs of the job (outside Slurm: 8). VETO_THREADS overrides. Read from the job's own CMDFILE line,
# so requeued and pending jobs (which keep their submitted task.sbatch but source this file at start) are covered too.
_veto_line=""
[ -n "${CMDFILE:-}" ] && [ -r "$CMDFILE" ] && _veto_line=$(sed -n "$((${SLURM_ARRAY_TASK_ID:-0} + 1))p" "$CMDFILE")
case "$_veto_line" in *--workers*|*--jobs*) _veto_t=1 ;; *) _veto_t=${SLURM_CPUS_PER_TASK:-8} ;; esac
export VETO_THREADS=${VETO_THREADS:-$_veto_t}
for _v in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS NUMEXPR_MAX_THREADS \
          VECLIB_MAXIMUM_THREADS NUMBA_NUM_THREADS GOTO_NUM_THREADS; do export $_v=$VETO_THREADS; done
unset _veto_line _veto_t _v
mkdir -p "$VETO_RESULTS"/{reco,interactions,evstore,em_debris,shield_dis,rate_maps,logs}
cvmfs_warm() {   # D15: read the Geant4 libraries once (first CVMFS access on a node can fail under load)
    cat /cvmfs/ship.cern.ch/25.09/sw/slc9_x86-64/GEANT4/v10.7.3-local1/lib64/*.so > /dev/null 2>&1 || true
}
