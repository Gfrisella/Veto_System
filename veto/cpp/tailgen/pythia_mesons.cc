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
// Extra argument "complete" (tailtransport's complete / analog modes): prompt species with a muonic decay channel
// (hadrons and tau with c tau <= 10 mm and a channel with a muon: eta, omega, rho, phi, J/psi, D, B, ...) are left
// undecayed and written (tailtransport decays them, with forced muonic decays), and the event's own muons (Drell-Yan,
// gamma gamma -> mu mu) are written too.
// Extra argument "onia" (with "complete"): only the quarkonia (J/psi, psi(2S), Upsilon(1S-3S)) of each event are written
// (statistics of the rare quarkonium muons: many Pythia events, only these particles go to Geant4).
// Extra argument "roulette=QMIN,E1,N,PT0": importance sampling of the written particles (not muons): each kept with
// q = min(1, max(QMIN, (E / E1)^N) x max(1, (pT / PT0)^2)) and weight 1 / q (written as a 7th column): hard / high-pT
// particles from every event, soft ones from a fraction (tailtransport set_roulette does the same in the cascade).
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

  bool complete = false, onia = false, rr = false;
  double rq[4] = {0, 0, 0, 0};
  for (int a = 6; a < argc; ++a) complete = complete || std::string(argv[a]) == "complete";
  for (int a = 6; a < argc; ++a) onia = onia || std::string(argv[a]) == "onia";
  for (int a = 6; a < argc; ++a)
    if (std::string(argv[a]).rfind("roulette=", 0) == 0) {
      rr = std::sscanf(argv[a] + 9, "%lf,%lf,%lf,%lf", &rq[0], &rq[1], &rq[2], &rq[3]) == 4;
    }
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
    for (int a = 6; a < argc; ++a)
      if (std::string(argv[a]) != "complete" && std::string(argv[a]) != "onia" && std::string(argv[a]).rfind("roulette=", 0) != 0)
        p.readString(argv[a]);
    if (!p.init()) {
      std::fprintf(stderr, "Pythia init failed\n");
      return 1;
    }
    if (complete)                                  // (nextId loop: also builds against Pythia 8.2, e.g. 8.223 of 2018)
      for (int id = p.particleData.nextId(1); id != 0; id = p.particleData.nextId(id)) {
        auto e = p.particleData.particleDataEntryPtr(id);
        if (!(e->isHadron() || e->id() == 15) || e->tau0() > 10.0) continue;
        for (int i = 0; i < e->sizeChannels(); ++i) {
          bool mu = false;
          for (int j = 0; j < e->channel(i).multiplicity(); ++j) mu = mu || std::abs(e->channel(i).product(j)) == 13;
          if (mu) { e->setMayDecay(false); break; }
        }
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
      if (a == 11 || a == 12 || a == 14 || a == 16) continue;      // photons kept (gamma -> mu mu)
      if (a == 13 && !complete) continue;
      if (onia && a != 443 && a != 100443 && a != 553 && a != 100553 && a != 200553) continue;
      if (q.e() < emin) continue;
      double wq = 1.0;
      if (rr && a != 13) {                          // importance sampling (roulette), weight 1 / q
        const double pt = q.pT(), qq = std::min(1.0, std::max(rq[0], std::pow(q.e() / rq[1], rq[2])) *
                                                       std::max(1.0, (pt / rq[3]) * (pt / rq[3])));
        if (gen[0]->rndm.flat() > qq) continue;
        wq = 1.0 / qq;
      }
      const double rec[7] = {double(iev), double(q.id()), q.px(), q.py(), q.pz(), q.e(), wq};
      std::fwrite(rec, sizeof(double), rr ? 7 : 6, f);
      ++n_written;
    }
  }
  std::fclose(f);
  std::ofstream meta(out + ".txt");
  meta << "events_requested " << n_events << "\nevents_generated " << n_ok << "\nevents_on_proton " << n_p
       << "\nmesons_written " << n_written << "\nemin " << emin << "\nz_over_a " << z_over_a << "\nseed " << seed
       << "\ncomplete " << int(complete) << "\nonia " << int(onia) << "\nroulette " << int(rr) << "\nsigma_inel_pp_mb " << gen[0]->info.sigmaGen() << "\nsigma_inel_pn_mb " << gen[1]->info.sigmaGen() << "\n";
  std::printf("%ld events, %ld mesons above %.0f GeV\n", n_ok, n_written, emin);
  return 0;
}
