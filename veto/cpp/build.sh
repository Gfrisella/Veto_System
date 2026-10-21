#!/bin/bash
# Build the g4dis Geant4 module (muon-nucleus thin target, bin/dis_library.py) inside the cudabear container
# (Geant4 11.2.2, pybind11, g++ 13):
#   bin/g4.sh bash cpp/build.sh
# Output: cpp/build/g4dis*.so (paths.cpp_build(); outside lib/ and bin/, which bin/remote.sh sync replaces).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build && cd build
cmake -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)" -DPython_EXECUTABLE="$(which python3)" .. > cmake.log
make -j"${JOBS:-16}" 2>&1 | grep -E "error|warning: unused|Built target" || true
ls -la g4dis*.so
