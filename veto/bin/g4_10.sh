#!/bin/bash
# As bin/g4.sh, with Geant4 10.00.p04 (built by us, /disk/lhcb_data/sqasim/geant4_10) instead of the container's 11.2:
# its libraries, its data sets, and our tailtransport built against it (TAILTRANSPORT_BUILD for bin/tail_produce.py).
G=${G4_10_ROOT:-/disk/lhcb_data/sqasim/geant4_10}
D=$G/data
R="$(cd "$(dirname "$0")/.." && pwd)"
exec bash /disk/lhcb_data/sqasim/images/shared/exec_cudabear_november_2025.sh \
    env GLIBC_TUNABLES=glibc.rtld.optional_static_tls=16777216 \
    LD_LIBRARY_PATH="$G/install/lib:$G/install/lib64:${LD_LIBRARY_PATH:-}" \
    G4LEDATA=$D/G4EMLOW6.35 G4LEVELGAMMADATA=$D/PhotonEvaporation3.0 G4RADIOACTIVEDATA=$D/RadioactiveDecay4.0 \
    G4NEUTRONXSDATA=$D/G4NEUTRONXS1.4 G4SAIDXSDATA=$D/G4SAIDDATA1.1 G4ENSDFSTATEDATA=$D/G4ENSDFSTATE1.0 \
    TAILTRANSPORT_BUILD="$R/cpp/tailgen_g410/build" "$@"
