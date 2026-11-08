#!/bin/bash
# Build tailtransport against Geant4 10.00.p04 (G4_10 = its install prefix), in cpp/tailgen_g410/build.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build && cd build
G4_10=${G4_10:-/disk/lhcb_data/sqasim/geant4_10/install}
G4DIR=$(dirname "$(find "$G4_10" -name Geant4Config.cmake | head -1)")
cmake -DGeant4_DIR="$G4DIR" -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)" -DPython_EXECUTABLE="$(which python3)" .. > cmake.log
make -j"${JOBS:-16}" 2>&1 | grep -E "error|Built target" || true
ls -la tailtransport*.so
