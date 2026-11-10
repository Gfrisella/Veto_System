#!/bin/bash
#SBATCH --job-name=nuprod
#SBATCH --partition=hep
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --array=0-1899%256
#SBATCH --output=/disk/lhcb_data/sqasim/nu_prod/log/task_%a.log
# Neutrino production with the new target (2026-11-09): tailtransport neutrino mode, FairShip target, FTFP_BERT, hadrons
# followed > 1.1 GeV, neutrinos > 1 GeV within 0.3 rad, upstream of z = -0.0045 m.
#   tasks    0- 499  min bias: 500 x 2e4 protons, tag nuMB, slices 600000-600499
#   tasks  500-1499  charm: charm_cascade --hadrons with charm hadrons > 1 GeV (HF_EMIN=1), 1000 x 1e4 protons, slices
#                    30000-30999, then hf_decay --neutrino on 50 % of the hadrons (tag charmc)
#   tasks 1500-1899  beauty: hf_decay --neutrino on the Nov-4 beauty hadron files (> 10 GeV, negligible), 5 % of the hadrons
# Submit from veto/ (R = the submit dir); override the log path for your own output:
#   sbatch --output=<dir>/task_%a.log bin/nu_prod.sbatch   (NU_PROD = output root, default $VETO_SHARED/nu_prod)
R=${VETO_DIR:-$SLURM_SUBMIT_DIR}
SHARED=${VETO_SHARED:-/disk/lhcb_data/sqasim}
NU=${NU_PROD:-$SHARED/nu_prod}
X=${PYTHIA8_XMLDOC:-$SHARED/pythia8223/share/Pythia8/xmldoc}
source ${APPTAINER_ENV:-/disk/users/sqasim/apptainer_shared/env.sh}
export APPTAINERENV_TAILTRANSPORT_BUILD=${TAILTRANSPORT_BUILD:-$SHARED/tt_nu_build}
export APPTAINERENV_PYTHIA_MESONS_EXE=${PYTHIA_MESONS_EXE:-$SHARED/pythia8223/pythia_mesons_8223}
export APPTAINERENV_PYTHIA_MESONS_DATA=$X APPTAINERENV_PYTHIA8DATA=$X
export APPTAINERENV_VETO_SOURCE=$NU
export APPTAINERENV_GLIBC_TUNABLES=glibc.rtld.optional_static_tls=16777216 APPTAINERENV_PYTHONPATH=$R/external/pylib
export APPTAINERENV_HF_EMIN=1
K=$SLURM_ARRAY_TASK_ID
RUN="apptainer exec -B /disk/users -B /disk/lhcb_data -B /disk/gfs_lhcb ${CUDABEAR_SIF:-$SHARED/images/cudabear_november_2025.sif} python3"
COMMON="--neutrino --target FairShip --emin 1.1 --l0 0.5"
cd $R
if [ $K -lt 500 ]; then
  S=$((600000 + K))
  [ -f $NU/tail/nuMB_nu_$S.npz ] && exit 0
  $RUN bin/tail_produce.py --mode complete $COMMON --events 20000 --tag nuMB --slice $S --n-slices 1000000
elif [ $K -lt 1500 ]; then
  S=$((30000 + K - 500))
  [ -f $NU/tail/charmc_nu_$S.npz ] && exit 0
  [ -f $NU/tail/charmc_had_$S.npz ] || \
  $RUN bin/charm_cascade.py --hadrons --flavour charm --target FairShip --slice $S --events 10000 || exit 1
  $RUN bin/hf_decay.py --tag charmc --slice $S $COMMON --physics FTFP_BERT --keep 0.5
else
  S=$((21000 + K - 1500))
  [ -f $NU/tail/beautyc_nu_$S.npz ] && exit 0
  $RUN bin/hf_decay.py --tag beautyc --slice $S --had-dir ${BEAUTY_HAD_DIR:-$SHARED/nov4_raw/tail} $COMMON --physics FTFP_BERT --keep 0.05
fi
