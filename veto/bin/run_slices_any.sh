#!/bin/bash
# Slice-parallel jobs in the cudabear container (bin/g4.sh), one log per slice in results/log:
#   bin/run_slices_any.sh TAG N_SLICES N_PARALLEL bin/script.py [args]
#   bin/run_slices_any.sh tail2 150 150 bin/tail_produce.py --events 800000 --tag tail2 --primary pythia6
set -u
TAG=$1; N=$2; J=$3; S=$4; shift 4
cd "$(dirname "$0")/.."
mkdir -p results/log
seq 0 $((N - 1)) | xargs -P "$J" -I{} bash -c \
  "bin/g4.sh python3 $S --slice {} --n-slices $N $* > results/log/${TAG}_{}.log 2>&1 && grep -h 'slice {}:' results/log/${TAG}_{}.log || echo 'slice {} FAILED'"
echo "all $N slices finished"
