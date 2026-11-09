"""Hold the free memory of some GPUs while a production of ours runs on them, so other jobs don't squeeze in.

    python3 bin/hold_gpu.py --gpus 1,2,3 --logs 'data/sim/.../logs/run6000_g5_gpu*.log' --hold-gb 5

The memory above our workers' peak (the propagation, which allocates outside torch): --hold-gb per GPU, at most
the free memory minus a margin. The rest of the card our workers hold themselves between propagations
(VETO_KEEP_GPU_CACHE=1, generate_muon_study.reserve_gpu). Waits until every log has a new "seed" line, frees it
once no generate_muon_study.py runs.
"""
import argparse
import glob
import subprocess
import time

import torch


def seed_lines(pattern):
    return sum(open(f).read().count("\nseed ") for f in glob.glob(pattern))


def ours_running():
    return subprocess.run(["pgrep", "-f", "generate_muon_study.py"], capture_output=True).returncode == 0


ap = argparse.ArgumentParser()
ap.add_argument("--gpus", required=True)
ap.add_argument("--logs", required=True)
ap.add_argument("--margin-gb", type=float, default=1.5)
ap.add_argument("--hold-gb", type=float, required=True)
args = ap.parse_args()
gpus = [int(g) for g in args.gpus.split(",")]

start = seed_lines(args.logs)
while seed_lines(args.logs) < start + len(gpus):
    time.sleep(20)
held = []
for g in gpus:
    free, total = torch.cuda.mem_get_info(g)
    n = int(min(args.hold_gb * 2**30, free - args.margin_gb * 2**30))
    if n > 0:
        held.append(torch.empty(n, dtype=torch.uint8, device=f"cuda:{g}"))
    print(f"{time.ctime()} gpu {g}: holding {max(n, 0) / 2**30:.1f} GB of {total / 2**30:.1f}", flush=True)
idle = 0
while idle < 5:                     # 5 minutes without a worker: done (not a restart or the gap between passes)
    idle = 0 if ours_running() else idle + 1
    time.sleep(60)
print(f"{time.ctime()} production done, released", flush=True)
