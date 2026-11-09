#!/bin/bash
# Build tailtransport against the container's Geant4 11.2 and Pythia 8.223 (PYTHIA8_223: install prefix), in
# cpp/tailgen_g411_p8223/build (inside the cudabear container). Run with PYTHIA8DATA=$PYTHIA8_223/share/Pythia8/xmldoc.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build && cd build
P8=${PYTHIA8_223:-/capstor/scratch/cscs/sqasim/pythia8223}
cmake -DPYTHIA8_223="$P8" -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)" -DPython_EXECUTABLE="$(which python3)" .. > cmake.log
make -j"${JOBS:-32}" 2>&1 | grep -E "error|Built target" || true
ls -la tailtransport*.so
