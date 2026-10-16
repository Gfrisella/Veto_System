#!/bin/bash
# Run bin/generate_muon_study.py over many seeds: one long process per GPU (geometry, field and the ~506 M input
# muons are set up once per process), each taking every n-th seed, in the MuonsAndMatter container.
# Guglielmo's code and inputs are mounted read only; outputs go to OUT_DIR (ours).
#
#   bin/run_muon_study.sh FIRST_SEED LAST_SEED [OUT_DIR]          # on farm-gpu3 / farm-gpu5, e.g. in tmux
#   GPUS="0 1" bin/run_muon_study.sh 0 19999                        # only these GPUs
#   BUDGET=15e6 MIN_SEEDS=20 BASE_SEED=0 N_MUONS=0 ...             # passed to generate_muon_study.py
# BASE_SEED is the first seed of the whole production (it fixes which seeds keep the common types): keep it
# the same on every node and when the production is continued in pieces. Two nodes can share one OUT_DIR
# if they run disjoint seed ranges. Seeds with an output (seed_<N>.npz, written atomically) are skipped, so
# a worker that dies is restarted and goes on; logs in OUT_DIR/logs/<host>_gpu<k>.log.
set -u
FIRST=${1:?first seed}
LAST=${2:?last seed}
HERE="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR=${3:-$HERE/data/sim/veto_muons_v1}
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}
BUDGET=${BUDGET:-15e6}
MIN_SEEDS=${MIN_SEEDS:-20}
BASE_SEED=${BASE_SEED:-0}
N_MUONS=${N_MUONS:-0}
MAX_RESTARTS=${MAX_RESTARTS:-5}
SIF=/disk/users/lprate/containers/MuonsAndMatterContainer.sif
MM=/disk/users/gfrise
HOST=$(hostname -s)
mkdir -p "$OUT_DIR/logs"
read -ra GPU_LIST <<< "$GPUS"
N=${#GPU_LIST[@]}
echo "$HOST: seeds $FIRST .. $LAST on GPU(s) ${GPU_LIST[*]} -> $OUT_DIR"

worker() {
    local k=$1 gpu=$2
    local log="$OUT_DIR/logs/${HOST}_gpu${gpu}.log"
    for ((attempt = 0; attempt <= MAX_RESTARTS; attempt++)); do
        PYTHONNOUSERSITE=1 PYTHONPATH= PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="$gpu" \
            apptainer exec --nv -B /cvmfs -B "$MM:$MM:ro" -B /disk/users/"$(whoami)" "$SIF" \
            python3 "$HERE/bin/generate_muon_study.py" --seeds "$FIRST-$LAST" --step "$N" --offset "$k" --gpu 0 \
            --budget "$BUDGET" --min-seeds "$MIN_SEEDS" --first-seed "$BASE_SEED" --n-muons "$N_MUONS" \
            --out-dir "$OUT_DIR" >> "$log" 2>&1 && { echo "[$HOST GPU $gpu] done"; return 0; }
        echo "[$HOST GPU $gpu] worker stopped (attempt $attempt), restarting; see $log"
        sleep 30
    done
    echo "[$HOST GPU $gpu] gave up after $MAX_RESTARTS restarts"
}

for k in "${!GPU_LIST[@]}"; do worker "$k" "${GPU_LIST[$k]}" & done
wait
echo "finished: $(ls "$OUT_DIR"/seed_*.npz 2>/dev/null | wc -l) seed files in $OUT_DIR"
