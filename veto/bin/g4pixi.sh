#!/bin/bash
# Geant4 11.4.2 from the FairShip pixi environment 26.10 on CVMFS (own compiler, Python 3.13, datasets), no container:
#   bin/g4pixi.sh build                      # cpp/g4shield.cc -> cpp/build_pixi (G4PIXI_MODULES to choose)
#   bin/g4pixi.sh python3 bin/v0_tally.py ...
# Used where the cudabear container (Geant4 11.2.2) cannot start (no user namespaces, e.g. sandboxed sessions).
# Physics differs from the chain's Geant4 11.2.2: compare like with like (both libraries through the same build).
set -euo pipefail
ENV=${FAIRSHIP_PIXI:-/cvmfs/ship.cern.ch/pixi-envs/fairship/26.10}
HERE="$(cd "$(dirname "$0")/.." && pwd)"
set +eu; source "$ENV/activate" >/dev/null 2>&1; set -eu
export G4SHIELD_BUILD="$HERE/cpp/build_pixi"
if [ "${1:-}" = "build" ]; then
    EXT=$(python3 -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")
    INC=$(python3 -c "import sysconfig;print(sysconfig.get_paths()['include'])")
    mkdir -p "$G4SHIELD_BUILD"
    for mod in ${G4PIXI_MODULES:-g4shield}; do
        g++ -O2 -shared -fPIC -std=c++17 -w -I"$HERE/external/pybind11_pkg/pybind11/include" -I"$INC" \
            $(geant4-config --cflags-without-gui) "$HERE/cpp/$mod.cc" -o "$G4SHIELD_BUILD/$mod$EXT" \
            -Wl,-rpath,"$CONDA_PREFIX/lib" $(geant4-config --libs-without-gui)
    done
    ls -la "$G4SHIELD_BUILD"
    exit 0
fi
exec "$@"
