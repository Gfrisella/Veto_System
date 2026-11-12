#!/bin/bash
#SBATCH --partition=hep
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
# Generic task of the re-run batch: line (SLURM_ARRAY_TASK_ID + 1) of $CMDFILE is one command, run from the code
# checkout. A line starting with "g4:" runs through bin/g4ship.sh (host Python + CVMFS Geant4 10.7.3: shield_dis,
# em_debris, g4_vessel_transport); other lines run with the host python3. Commands come from a FILE, never from
# --export (Slurm splits --export at commas). Submitted by bin/batch/submit_rerun.sh.
set -o pipefail
source "${VETO_CODE:?}/bin/batch/env.sh"
cd "$VETO_CODE"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${CMDFILE:?}")
[ -n "$line" ] || { echo "no line $((SLURM_ARRAY_TASK_ID + 1)) in $CMDFILE"; exit 1; }
echo "[$(date '+%F %T')] $(hostname) task $SLURM_ARRAY_TASK_ID: $line"
if [[ "$line" == g4:* ]]; then
    cvmfs_warm
    eval "$VETO_MAIN/bin/g4ship.sh ${line#g4:}"
else
    eval "$line"
fi
rc=$?
echo "[$(date '+%F %T')] exit $rc"
exit $rc
