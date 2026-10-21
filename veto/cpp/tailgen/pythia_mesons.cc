// Forward mesons of 400 GeV proton-nucleon collisions (Pythia 8), the first step of the high-energy muon
// tail production (see bin/tail_produce.py).
//
// Settings as FairShip's FixedTargetGenerator (minimum bias): beam p 400 GeV on a nucleon at rest, target
// p or n with probability Z/A (default tungsten 74/183.84), SoftQCD:inelastic + gm gm -> mu mu, prompt
// photons, weak boson exchange / single boson; long-lived particles (c tau > 1 cm) stable, as FairShip
// leaves their decays to Geant4. Writes every final-state hadron (no leptons, photons) with energy > EMIN,
// the input of the Geant4 cascade (tail_produce.py):
//
//   pythia_mesons N_EVENTS SEED EMIN OUT.bin [Z_OVER_A] [extra Pythia settings ...]
//
// OUT.bin: float64 records (event, pdg, px, py, pz, E) [GeV]; OUT.bin.txt: events generated, settings.
#include "Pythia8/Pythia.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

using namespace Pythia8;

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: %s N_EVENTS SEED EMIN OUT.bin [Z_OVER_A]\n", argv[0]);
    return 1;
  }
  const long n_events = std::atol(argv[1]);
  const int seed = std::atoi(argv[2]);
  const double emin = std::atof(argv[3]);
  const std::string out = argv[4];
  const double z_over_a = argc > 5 ? std::atof(argv[5]) : 74.0 / 183.84;

  Pythia* gen[2];
  for (int k = 0; k < 2; ++k) {
    gen[k] = new Pythia("../share/Pythia8/xmldoc", false);
    Pythia& p = *gen[k];
    p.readString("Beams:frameType = 2");
    p.readString("Beams:idA = 2212");
    p.readString(k == 0 ? "Beams:idB = 2212" : "Beams:idB = 2112");
    p.readString("Beams:eA = 400.");
    p.readString("Beams:eB = 0.");
    p.readString("SoftQCD:inelastic = on");
    p.readString("PhotonCollision:gmgm2mumu = on");
    p.readString("PromptPhoton:all = on");
    p.readString("WeakBosonExchange:all = on");
    p.readString("WeakSingleBoson:all = on");
    p.readString("ParticleDecays:limitTau0 = on");     // particles with c tau > 10 mm stay stable (Geant4 decays them)
    p.readString("ParticleDecays:tau0Max = 10.");
    p.readString("Next:numberCount = 0");
    p.readString("Random:setSeed = on");
    p.readString("Random:seed = " + std::to_string(seed * 2 + k + 1));
    for (int a = 6; a < argc; ++a) p.readString(argv[a]);
    if (!p.init()) {
      std::fprintf(stderr, "Pythia init failed\n");
      return 1;
    }
  }
  std::FILE* f = std::fopen(out.c_str(), "wb");
  long n_written = 0, n_ok = 0, n_p = 0;
  for (long iev = 0; iev < n_events; ++iev) {
    const int k = (gen[0]->rndm.flat() < z_over_a) ? 0 : 1;
    Pythia& p = *gen[k];
    if (!p.next()) continue;
    ++n_ok;
    n_p += (k == 0);
    for (int i = 0; i < p.event.size(); ++i) {
      const Particle& q = p.event[i];
      if (!q.isFinal()) continue;
      const int a = std::abs(q.id());
      if (a == 11 || a == 12 || a == 13 || a == 14 || a == 16) continue;      // photons kept (gamma -> mu mu)
      if (q.e() < emin) continue;
      const double rec[6] = {double(iev), double(q.id()), q.px(), q.py(), q.pz(), q.e()};
      std::fwrite(rec, sizeof(double), 6, f);
      ++n_written;
    }
  }
  std::fclose(f);
  std::ofstream meta(out + ".txt");
  meta << "events_requested " << n_events << "\nevents_generated " << n_ok << "\nevents_on_proton " << n_p
       << "\nmesons_written " << n_written << "\nemin " << emin << "\nz_over_a " << z_over_a << "\nseed " << seed
       << "\nsigma_inel_pp_mb " << gen[0]->info.sigmaGen() << "\nsigma_inel_pn_mb " << gen[1]->info.sigmaGen() << "\n";
  std::printf("%ld events, %ld mesons above %.0f GeV\n", n_ok, n_written, emin);
  return 0;
}
