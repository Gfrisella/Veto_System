#!/bin/bash
# Run a command with the CERN LCG_106 software stack (python3 with uproot, awkward, ROOT) from cvmfs.
# The cudabear container has no uproot. Works on farm-ui and the farm-gpu nodes.
#   bin/lcg.sh python3 bin/extract_neutrinos.py
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh >/dev/null 2>&1
exec "$@"
