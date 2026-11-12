#!/bin/bash
#SBATCH --job-name=d6_libs
#SBATCH --partition=standard
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
# ISSUES D6 (approved 12 Nov 2026), branch veto-d6-energy-rescale: interaction libraries on the 400 GeV / 5 % grid.
#   task 0: muon DIS library, all materials (helium, pvc, mylar, air, planes)  -> dis/mu_library_5pct.npz
#   task 1: muon DIS library, iron (MS7 chain)                                  -> dis/mu_library_iron_5pct.npz
#   task 2: neutrino library (GENIE, 6 flavours, all nuclei)                    -> genie/nu_library_v2_5pct.npz
# New names next to the current libraries: nothing is replaced. Submit from the d6 worktree's veto/ directory:
#     sbatch --output=<logdir>/d6_%a.log bin/batch/d6_libraries.sbatch
# then compare (host python):  python3 bin/d6_compare_libraries.py --libs <old> <new> --material iron
R=${VETO_CODE:-$SLURM_SUBMIT_DIR}
MAIN=${VETO_MAIN:-/disk/gfs_lhcb/gfrise/Study_on_Veto_System/Optimization_Veto_System/veto}
LIB=$MAIN/data/interactions
export VETO_LIB_EMAX=400 VETO_LIB_STEP=0.05 VETO_INTERACTIONS=$LIB
cd $R
case $SLURM_ARRAY_TASK_ID in
  0|1)
    source ${APPTAINER_ENV:-/disk/users/sqasim/apptainer_shared/env.sh}
    OUT=$LIB/dis/mu_library_5pct.npz; MAT=""
    [ "$SLURM_ARRAY_TASK_ID" = 1 ] && { OUT=$LIB/dis/mu_library_iron_5pct.npz; MAT="--materials iron"; }
    apptainer exec -B /disk/users -B /disk/lhcb_data -B /disk/gfs_lhcb ${CUDABEAR_SIF:-/disk/lhcb_data/sqasim/images/cudabear_november_2025.sif} \
      env GLIBC_TUNABLES=glibc.rtld.optional_static_tls=16777216 PYTHONPATH=$R/external/pylib \
      VETO_LIB_EMAX=$VETO_LIB_EMAX VETO_LIB_STEP=$VETO_LIB_STEP VETO_INTERACTIONS=$LIB \
      python3 bin/dis_library.py --jobs $SLURM_CPUS_PER_TASK $MAT --out $OUT ;;
  2)
    bin/lcg.sh python3 bin/nu_library.py --jobs $SLURM_CPUS_PER_TASK --out $LIB/genie/nu_library_v2_5pct.npz ;;
esac
