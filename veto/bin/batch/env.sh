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
mkdir -p "$VETO_RESULTS"/{reco,interactions,evstore,em_debris,shield_dis,rate_maps,logs}
cvmfs_warm() {   # D15: read the Geant4 libraries once (first CVMFS access on a node can fail under load)
    cat /cvmfs/ship.cern.ch/25.09/sw/slc9_x86-64/GEANT4/v10.7.3-local1/lib64/*.so > /dev/null 2>&1 || true
}
