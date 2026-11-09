#!/bin/bash
# Build tailtransport against Geant4 10.01.p03 (G4_101: install prefix) and Pythia 8.223 (PYTHIA8_223), in
# cpp/tailgen_g4101/build: bin/g4_101.sh bash cpp/tailgen_g4101/build.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build && cd build
G4_101=${G4_101:-/disk/lhcb_data/sqasim/geant4_101/install}
P8=${PYTHIA8_223:-/disk/lhcb_data/sqasim/pythia8223}
G4DIR=$(dirname "$(find "$G4_101" -name Geant4Config.cmake | head -1)")
cmake -DGeant4_DIR="$G4DIR" -DPYTHIA8_223="$P8" -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)" \
      -DPython_EXECUTABLE="$(which python3)" .. > cmake.log
make -j"${JOBS:-16}" 2>&1 | grep -E "error|Built target" || true
ls -la tailtransport*.so
