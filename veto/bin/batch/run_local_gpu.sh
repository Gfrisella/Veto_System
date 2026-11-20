#!/bin/bash
# Stage S4 / S4b of the re-run batch on the farm-gpu3 GPUs (no GPUs in Slurm): pair inputs and cross-run pairs from the
# full store, after S1 (rate maps). Run on farm-gpu3:  nohup bash bin/batch/run_local_gpu.sh > <log> 2>&1 &
# CUDA_VISIBLE_DEVICES picks the GPU (they are shared: check nvidia-smi for the freest one).
set -euo pipefail
export VETO_CODE=${VETO_CODE:-$(cd "$(dirname "$0")/../.." && pwd)}
source "$VETO_CODE/bin/batch/env.sh"
cd "$VETO_CODE"
echo "[$(date '+%F %T')] S4 make_reco_inputs on GPU ${CUDA_VISIBLE_DEVICES:-all}"
python3 bin/make_reco_inputs.py --dataset $VETO_MUON_DATASET --maps $VETO_RATE_MAPS --out $VETO_RECO_INPUTS --device cuda --pair-block 20000
echo "[$(date '+%F %T')] S4b make_crossrun_pairs"
python3 bin/make_crossrun_pairs.py
echo "[$(date '+%F %T')] done"
