// Standalone check of tailtransport's Pythia decayer (init_pythia / py_decay): decay one particle many times with the same
// settings and count what comes back: next() failures, products with non-finite momenta, and energy not conserved.
//   decay_check ID PX PY PZ N CHECK_EVENT(0/1) SEED      (run with PYTHIA8DATA pointing to the xmldoc of the same version)
// Built against the static Pythia 8.223: g++ -O2 decay_check.cc -I$P8/include $P8/lib/libpythia8.a -ldl -o decay_check

#include "Pythia8/Pythia.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
  const int id = std::atoi(argv[1]);
  const double px = std::atof(argv[2]), py_ = std::atof(argv[3]), pz = std::atof(argv[4]);
  const long n = std::atol(argv[5]);
  const bool check = std::atoi(argv[6]) != 0;
  Pythia8::Pythia py("../share/Pythia8/xmldoc", false);
  for (auto c : {"ProcessLevel:all = off", "Next:numberCount = 0", "Print:quiet = on", "Random:setSeed = on"}) py.readString(c);
  py.readString(std::string("Check:event = ") + (check ? "on" : "off"));
  py.readString(std::string("Random:seed = ") + argv[7]);
  auto& pd = py.particleData;
  // the species tailtransport decays itself (a muonic channel; tau0 <= 10 mm prompt) and every long-lived one stay undecayed,
  // as in init_pythia: hadrons and tau with a muonic decay channel, or tau0 > 10 mm
  std::vector<int> off;
  for (int i = pd.nextId(1); i != 0; i = pd.nextId(i)) {
    auto e = pd.particleDataEntryPtr(i);
    bool mu = false;
    if (e->isHadron() || i == 15)
      for (int c = 0; c < e->sizeChannels(); ++c)
        for (int j = 0; j < e->channel(c).multiplicity(); ++j) mu = mu || std::abs(e->channel(c).product(j)) == 13;
    if (mu || e->tau0() > 10.0) off.push_back(i);
  }
  if (!py.init()) return 1;
  for (int i : off) pd.mayDecay(i, false);
  const double m = pd.m0(id), E = std::sqrt(px * px + py_ * py_ + pz * pz + m * m);
  long fail = 0, nonfinite = 0, econs = 0, above = 0;
  for (long k = 0; k < n; ++k) {
    pd.mayDecay(id, true);
    py.event.reset();
    py.event.append(id, 1, 0, 0, px, py_, pz, E, m);
    const bool ok = py.next();
    pd.mayDecay(id, false);
    if (!ok) { ++fail; continue; }
    double sum = 0, emax = 0;
    bool bad = false;
    for (int i = 0; i < py.event.size(); ++i) {
      const auto& q = py.event[i];
      if (!q.isFinal()) continue;
      if (!std::isfinite(q.e()) || !std::isfinite(q.px()) || !std::isfinite(q.py()) || !std::isfinite(q.pz())) bad = true;
      sum += q.e(); emax = std::max(emax, q.e());
    }
    if (bad) ++nonfinite;
    if (!(std::abs(sum - E) <= 1e-3 * E)) ++econs;
    if (!(emax <= E * 1.001)) ++above;
    if ((bad || !(std::abs(sum - E) <= 1e-3 * E)) && nonfinite + econs <= 3) {
      std::printf("--- decay %ld: final-state sum E %g (parent %g), max %g\n", k, sum, E, emax);
      py.event.list();
    }
  }
  std::printf("id %d E %.3f GeV, Check:event %s: %ld decays: next() failed %ld, non-finite products %ld, energy not conserved %ld, "
              "product above the parent's energy %ld\n", id, E, check ? "on" : "off", n, fail, nonfinite, econs, above);
  return 0;
}
