#!/bin/bash
#SBATCH --job-name=acts_scan
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --array=0-9%5
# ISSUES D14 (2): ACTS resolution scan, one momentum point per task (bin/acts_resolution_scan.py).
#   sbatch --output=<logdir>/acts_scan_%a.log --export=ALL,VETO_CODE=$PWD bin/batch/acts_scan.sbatch
# then:  bin/fairship_env.sh python3 bin/acts_resolution_scan.py fit $OUT/p*/point.npz --out data/acts/resolution_param.json
P_LIST=(1 2 3 5 10 20 50 100 200 400)
N=${ACTS_N:-4000}
R=${VETO_CODE:-$SLURM_SUBMIT_DIR}
OUT=${ACTS_OUT:-/disk/gfs_lhcb/gfrise/Study_on_Veto_System/Optimization_Veto_System/veto/data/acts/pg_scan_2611}
P=${P_LIST[$SLURM_ARRAY_TASK_ID]}
D=$OUT/p$P
mkdir -p $D && cd $D || exit 1
unset DISPLAY; export QT_QPA_PLATFORM=offscreen G4UI_USE_TCSH=1   # no X on the nodes: G4 would start the Qt UI and abort (job 2090827)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
FS=/cvmfs/ship.cern.ch/pixi-envs/fairship/26.10/macro
echo "[$(date '+%F %T')] $(hostname) p = $P GeV, $N muons -> $D"
rm -f sim_*.root geo_*.root params_*.root
$R/bin/fairship_env.sh python3 $FS/run_simScript.py -n $N -s $((1000 + SLURM_ARRAY_TASK_ID)) -o $D PG --pID 13 --bothCharges \
    --Estart $P --Eend $P --multiplePG --Vx 0 --Dx 300 --Vy 0 --Dy 500 --Vz 8335 --thetaMin 0 --thetaMax 1.5 > sim.log 2>&1 || { echo sim failed; tail sim.log; exit 1; }
S=$(ls sim_*.root | grep -v _rec.root); G=$(ls geo_*.root)
$R/bin/fairship_env.sh python3 $FS/ShipReco.py -f $S -g $G --trackFitter acts --patRec Truth > reco.log 2>&1 || { echo reco failed; tail reco.log; exit 1; }
$R/bin/fairship_env.sh python3 $R/bin/acts_resolution_scan.py extract $D --out $D/point.npz || exit 1
echo "[$(date '+%F %T')] done"
