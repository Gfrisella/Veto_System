#!/bin/bash
#SBATCH --job-name=rocklib
#SBATCH --partition=hep
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --array=0-223%128
#SBATCH --output=/disk/lhcb_data/sqasim/rock_dis/library/logs/task_%a.log
# Cavern-rock interaction library (bin/rock_library.py), one (energy, angle, part) per array task. Calls apptainer directly
# (batch jobs: no /run/user bind), with what bin/g4.sh sets. Submit from veto/ (R = the submit dir) and override the
# log path for your own output: sbatch --output=<dir>/task_%a.log bin/rock_library.sbatch (ROCK_LIB = library dir).
R=${VETO_DIR:-$SLURM_SUBMIT_DIR}
source ${APPTAINER_ENV:-/disk/users/sqasim/apptainer_shared/env.sh}
cd $R
apptainer exec -B /disk/users -B /disk/lhcb_data -B /disk/gfs_lhcb ${CUDABEAR_SIF:-/disk/lhcb_data/sqasim/images/cudabear_november_2025.sif} \
  env GLIBC_TUNABLES=glibc.rtld.optional_static_tls=16777216 PYTHONPATH=$R/external/pylib \
  python3 bin/rock_library.py task --index $SLURM_ARRAY_TASK_ID
