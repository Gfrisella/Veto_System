#!/bin/bash
# As bin/g4.sh, with Geant4 10.01.p03 and Pythia 8.223 (the 2018 FairShip production's versions, built by us:
# /disk/lhcb_data/sqasim/geant4_101, /disk/lhcb_data/sqasim/pythia8223): their libraries and data sets, our tailtransport
# built against them (TAILTRANSPORT_BUILD), the Pythia 8.223 generator (PYTHIA_MESONS_EXE) for bin/tail_produce.py.
G=${G4_101_ROOT:-/disk/lhcb_data/sqasim/geant4_101}
P=${PYTHIA8_223:-/disk/lhcb_data/sqasim/pythia8223}
D=$G/data
R="$(cd "$(dirname "$0")/.." && pwd)"
exec bash /disk/lhcb_data/sqasim/images/shared/exec_cudabear_november_2025.sh \
    env GLIBC_TUNABLES=glibc.rtld.optional_static_tls=16777216 \
    LD_LIBRARY_PATH="$G/install/lib:$G/install/lib64:${LD_LIBRARY_PATH:-}" \
    G4LEDATA=$D/G4EMLOW6.41 G4LEVELGAMMADATA=$D/PhotonEvaporation3.1 G4RADIOACTIVEDATA=$D/RadioactiveDecay4.2 \
    G4NEUTRONXSDATA=$D/G4NEUTRONXS1.4 G4SAIDXSDATA=$D/G4SAIDDATA1.1 G4ENSDFSTATEDATA=$D/G4ENSDFSTATE1.0 \
    PYTHIA8DATA=$P/share/Pythia8/xmldoc PYTHIA_MESONS_DATA=$P/share/Pythia8/xmldoc PYTHIA_MESONS_EXE=$P/pythia_mesons_8223 \
    TAILTRANSPORT_BUILD="${TAILTRANSPORT_BUILD:-$R/cpp/tailgen_g4101/build}" "$@"
