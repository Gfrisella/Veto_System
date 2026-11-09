#!/bin/bash
# Run a command in the FairShip pixi environment (CVMFS, self-contained conda build: ROOT 6.40, Geant4 11.4, ACTS 48.2)
# with our libShipActsTracking (PR #1545 straw material, external/fairship_acts) in front of the installed one.
#   bin/fairship_env.sh python3 <script>        FAIRSHIP_PIXI=<other env dir> to change the environment
ENV=${FAIRSHIP_PIXI:-/cvmfs/ship.cern.ch/pixi-envs/fairship/26.10}
source "$ENV/activate" >/dev/null 2>&1
export LD_LIBRARY_PATH="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)/external/fairship_acts/lib:$LD_LIBRARY_PATH"
exec "$@"
