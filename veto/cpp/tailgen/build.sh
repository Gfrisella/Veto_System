#!/bin/bash
# Build the tail-production tools inside the cudabear container (Pythia 8.316, Geant4 11.2.2):
#   bin/g4.sh bash cpp/tailgen/build.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
g++ -O2 -std=c++17 pythia_mesons.cc -o build/pythia_mesons -I/usr/local/include -L/usr/local/lib -lpythia8 -ldl \
    -Wl,-rpath,/usr/local/lib
echo "built build/pythia_mesons"
cd build
cmake -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)" -DPython_EXECUTABLE="$(which python3)" .. > cmake.log
make -j16 2>&1 | grep -E "error|Built target" || true
ls tailtransport*.so
