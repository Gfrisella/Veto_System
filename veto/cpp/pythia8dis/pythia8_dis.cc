// Muon-nucleon interactions with Pythia 8 (fixed target), same output as cpp/tailgen/pythia6_dis:
//   pythia8_dis <mode> <target p+|n0> <E_mu GeV> <n> <seed> <out.bin> [settings...]
//   mode: dis   = neutral-current DIS, WeakBosonExchange:ff2ff(t:gmZ), Q^2 > Q2min (default 1 GeV^2)
//         photo = quasi-real photons (Q^2 < 1) from the muon, PDF:lepton2gamma, soft + hard QCD (resolved and direct)
// out.bin: float64 records (event, pdg, px, py, pz, E, vx, vy, vz) of final-state particles (m, GeV),
// beam along +z, nucleon at rest; out.bin.txt: "sigma_mb <x>" and "events <n>".
#include "Pythia8/Pythia.h"
#include <cstdio>
#include <fstream>
#include <string>
using namespace Pythia8;
int main(int argc, char** argv) {
  if (argc < 7) { fprintf(stderr, "usage: pythia8_dis mode target E n seed out [settings]\n"); return 1; }
  std::string mode = argv[1], tgt = argv[2]; double E = atof(argv[3]); int n = atoi(argv[4]);
  int seed = atoi(argv[5]); std::string out = argv[6];
  Pythia py("/cvmfs/ship.cern.ch/25.09/sw/slc9_x86-64/pythia/v8230-ship-local1/share/Pythia8/xmldoc", false);
  py.readString("Beams:frameType = 2");
  py.readString("Beams:idA = 13");
  py.readString(std::string("Beams:idB = ") + (tgt == "p+" ? "2212" : "2112"));
  py.readString("Beams:eA = " + std::to_string(E));
  py.readString("Beams:eB = 0.");
  py.readString("Random:setSeed = on");
  py.readString("Random:seed = " + std::to_string(seed % 900000000));
  py.readString("Next:numberCount = 0");
  py.readString("Check:event = off");
  py.readString("130:mayDecay = off");      // K_L stable (transported later)
  py.readString("13:mayDecay = off"); py.readString("211:mayDecay = off"); py.readString("321:mayDecay = off");
  py.readString("2112:mayDecay = off");    // as pythia6_dis: pi+-, K+-, K_L, n, mu stable
  // Pythia 8.230 production vertices are not usable here (hadrons at ~1e6 m): weakly decaying hadrons are kept and
  // decayed with their decay lengths by fastveto.decay (bin/p8_library.py), every vertex is written as 0.
  for (const char* id : {"310", "3122", "3112", "3222", "3312", "3322", "3334"})
    py.readString(std::string(id) + ":mayDecay = off");
  if (mode == "dis") {
    py.readString("WeakBosonExchange:ff2ff(t:gmZ) = on");
    py.readString("PhaseSpace:Q2Min = 1.");
    py.readString("PhaseSpace:mHatMin = 0.");   // default 4 GeV would cut low-energy DIS (s-hat = x s)
    py.readString("SpaceShower:dipoleRecoil = on");
    py.readString("PDF:lepton = off");
    py.readString("TimeShower:QEDshowerByL = off");
  } else if (mode == "photo") {
    py.readString("PDF:lepton2gamma = on");
    py.readString("Photon:ProcessType = 0");
    py.readString("Photon:Q2max = 1.");
    py.readString("SoftQCD:nonDiffractive = on");
    py.readString("HardQCD:all = on");
    py.readString("PhotonParton:all = on");
  } else { fprintf(stderr, "unknown mode\n"); return 1; }
  for (int i = 7; i < argc; ++i) py.readString(argv[i]);
  if (!py.init()) { fprintf(stderr, "init failed\n"); return 2; }
  FILE* f = fopen(out.c_str(), "wb");
  int done = 0, fails = 0;
  while (done < n) {
    if (!py.next()) { if (++fails > 10 * n + 100) break; continue; }
    for (int i = 0; i < py.event.size(); ++i) {
      const Particle& p = py.event[i];
      if (!p.isFinal()) continue;
      double r[9] = {double(done), double(p.id()), p.px(), p.py(), p.pz(), p.e(), 0., 0., 0.};
      fwrite(r, sizeof(double), 9, f);
    }
    ++done;
  }
  fclose(f);
  std::ofstream t(out + ".txt");
  t << "sigma_mb " << py.info.sigmaGen() << "\nevents " << done << "\nfails " << fails << "\n";
  return 0;
}
