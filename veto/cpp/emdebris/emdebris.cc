// emdebris: a plain C++ model (no Geant4) of the EM debris leaving a thick slab behind a muon, to compare with the
// Geant4 library (bin/debris_library.py) and to be fast enough for the veto simulation.
//
// The muon crosses the last D cm of the slab (default 20 cm of iron, ~11 X0: debris escaping the exit face is made
// there), straight, at angle theta to the slab normal (about y). Along its path it makes
//   delta rays    spin-1/2 knock-on spectrum, dN/dT dx = K/2 Z/A rho / beta^2 / T^2 (1 - beta^2 T/Tmax + T^2/(2 E^2)),
//                 T > T_CUT, the knock-on angle cos = (T/p_e) (E + m_e) / p_mu
//   direct pairs  dN/d eps ~ 1/eps^2 above 2 MeV, normalised to the pair energy loss b_pair rho E; energy shared
//                 uniformly between e+ and e-, emitted along the muon
//   brems photons dN/dv = b_brems rho (4/3 - 4/3 v + v^2) / v, along the muon
// and the secondaries shower (condensed history): e+- continuous loss (stopping power table), bremsstrahlung
// (complete screening, k > K_CUT), Highland multiple scattering, positron annihilation at rest (2 x 511 keV); photons:
// total attenuation (NIST-like table), photo-effect / Compton (Klein-Nishina) / pair. Every e+- and photon crossing the
// exit face with kinetic energy > 0.1 MeV is recorded in the MUON FRAME (offset from the muon's exit point and
// momentum, rotated so the muon direction is +z), as the Geant4 library.
//
//   emdebris E_GeV theta_rad n_muons seed out.bin [depth_cm]
// out.bin: int32 n_muons, then records {int32 ev, int32 pdg, float ekin_GeV, float offx_m, float offy_m, float offz_m,
// float px_GeV, float py_GeV, float pz_GeV, float dt_ns}. Prints the time per muon to stderr.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

// iron
constexpr double Z = 26, A = 55.845, RHO = 7.874;             // g/cm^3
constexpr double X0 = 13.84 / RHO;                            // cm
constexpr double K = 0.307075;                                // MeV cm^2 / mol
constexpr double ME = 0.51099895, MMU = 105.6583755;          // MeV
constexpr double B_PAIR = 1.5e-6, B_BREMS = 1.1e-6;           // g^-1 cm^2, muon energy-loss coefficients (~100 GeV, Fe)
constexpr double T_CUT = 0.5;                                 // MeV, delta rays tracked above this
constexpr double K_CUT = 0.1;                                 // MeV, bremsstrahlung photons above this
constexpr double E_STOP = 0.2;                                // MeV, e+- stopped below this (kinetic)
constexpr double G_STOP = 0.1;                                // MeV, photons absorbed below this
constexpr double C_CM_NS = 29.9792458;                        // cm / ns
double MARGIN = 15.0;                                         // cm: an e+- farther than its range + MARGIN from the exit is dropped

std::mt19937_64 rng;
inline double U() { return std::uniform_real_distribution<double>(0.0, 1.0)(rng); }

struct Particle { int pdg; double x, y, z, ux, uy, uz, T, t; };
struct Record { int32_t ev, pdg; float ekin, ox, oy, oz, px, py, pz, dt; };

double interp_loglog(const double* x, const double* y, int n, double v) {
  if (v <= x[0]) return y[0];
  if (v >= x[n - 1]) return y[n - 1];
  int i = 0;
  while (v > x[i + 1]) ++i;
  double f = std::log(v / x[i]) / std::log(x[i + 1] / x[i]);
  return std::exp(std::log(y[i]) + f * std::log(y[i + 1] / y[i]));
}

// e+- collision stopping power in iron [MeV cm^2/g] vs kinetic energy [MeV] (ESTAR-like)
const double SP_T[] = {0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 100, 1000};
const double SP_V[] = {2.60, 1.95, 1.55, 1.47, 1.46, 1.50, 1.55, 1.62, 1.70, 1.80};
double dedx(double T) { return RHO * interp_loglog(SP_T, SP_V, 10, T); }   // MeV / cm

// photon mass attenuation in iron [cm^2/g] (XCOM-like, without coherent) and the pair / photo-effect fractions
const double MU_E[] = {0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.25, 1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100, 1000};
const double MU_V[] = {0.3717, 0.1964, 0.146, 0.1099, 0.094, 0.0841, 0.0769, 0.0669, 0.0599, 0.0534, 0.0488, 0.0426,
                       0.0362, 0.0331, 0.0315, 0.0306, 0.0299, 0.0299, 0.0308, 0.0321, 0.0343, 0.0372, 0.04, 0.042};
const double PF_E[] = {1.1, 2, 3, 5, 10, 20, 50, 100, 1000};
const double PF_V[] = {1e-4, 0.012, 0.05, 0.15, 0.37, 0.6, 0.82, 0.9, 0.98};
double pair_frac(double E) { return E < 1.1 ? 0.0 : interp_loglog(PF_E, PF_V, 9, E); }
double pe_frac(double E) { return std::min(1.0, 0.6 * std::pow(0.1 / E, 2.8)); }

void rotate(double& ux, double& uy, double& uz, double cost, double phi) {     // turn (ux,uy,uz) by polar cost, azimuth phi
  double sint = std::sqrt(std::max(0.0, 1 - cost * cost)), cp = std::cos(phi), sp = std::sin(phi);
  double nx, ny, nz;
  if (std::fabs(uz) < 0.99999) {
    double s = std::sqrt(1 - uz * uz);
    nx = ux * cost + sint * (ux * uz * cp - uy * sp) / s;
    ny = uy * cost + sint * (uy * uz * cp + ux * sp) / s;
    nz = uz * cost - sint * s * cp;
  } else {
    double sg = uz > 0 ? 1 : -1;
    nx = sint * cp; ny = sint * sp; nz = sg * cost;
  }
  double n = std::sqrt(nx * nx + ny * ny + nz * nz);
  ux = nx / n; uy = ny / n; uz = nz / n;
}

struct Model {
  double D, theta, mux_exit, t_mu_exit;
  std::vector<Record>* out;
  int ev;
  std::vector<Particle> stack;

  void record(const Particle& p, double xe, double ye, double te) {
    // muon frame: rotation about y by -theta takes the muon direction (sin, 0, cos) to +z
    double c = std::cos(theta), s = std::sin(theta);
    double dx = xe - mux_exit, dy = ye;
    double pm = p.pdg == 22 ? p.T : std::sqrt(p.T * (p.T + 2 * ME));
    double px = pm * p.ux, py = pm * p.uy, pz = pm * p.uz;
    Record r{ev, p.pdg, float(p.T * 1e-3), float(dx * c * 1e-2), float(dy * 1e-2), float(dx * s * 1e-2),
             float((px * c - pz * s) * 1e-3), float(py * 1e-3), float((px * s + pz * c) * 1e-3), float(te - t_mu_exit)};
    out->push_back(r);
  }

  // move by s; true if the exit face was crossed (then recorded) or the particle left upstream
  bool move(Particle& p, double s) {
    double z1 = p.z + s * p.uz;
    if (z1 >= D) {
      double f = (D - p.z) / p.uz;
      double beta = p.pdg == 22 ? 1.0 : std::sqrt(p.T * (p.T + 2 * ME)) / (p.T + ME);
      record(p, p.x + f * p.ux, p.y + f * p.uy, p.t + f / (beta * C_CM_NS));
      return true;
    }
    if (z1 < 0) return true;
    double beta = p.pdg == 22 ? 1.0 : std::sqrt(p.T * (p.T + 2 * ME)) / (p.T + ME);
    p.x += s * p.ux; p.y += s * p.uy; p.z = z1; p.t += s / (beta * C_CM_NS);
    return false;
  }

  void electron(Particle p) {
    while (p.T > E_STOP) {
      double el = dedx(p.T);
      // cannot reach the exit face any more (its range, plus a few cm for a photon it may still radiate)
      if (D - p.z > p.T / el + MARGIN) return;
      double s = std::min(0.1, std::max(0.005, 0.2 * p.T / el));
      // bremsstrahlung in this step
      if (p.T > K_CUT) {
        double ymin = K_CUT / p.T;
        double rate = (4.0 / 3 * std::log(1 / ymin) - 4.0 / 3 * (1 - ymin) + 0.5 * (1 - ymin * ymin)) / X0;
        int n = std::poisson_distribution<int>(rate * s)(rng);
        for (int i = 0; i < n && p.T > K_CUT; ++i) {
          double y;
          do { y = ymin * std::pow(1 / ymin, U()); } while (U() * 4.0 / 3 > 4.0 / 3 - 4.0 / 3 * y + y * y);
          double k = y * p.T;
          stack.push_back({22, p.x, p.y, p.z, p.ux, p.uy, p.uz, k, p.t});
          p.T -= k;
        }
      }
      // Moller (e-) / Bhabha (e+) knock-on electrons above T_CUT, 1/T^2 (the low-transfer limit), T_d < T/2
      if (p.T > 2 * T_CUT) {
        double rate_m = 0.5 * K * Z / A * RHO * (1 / T_CUT - 2 / p.T);
        int nm = std::poisson_distribution<int>(rate_m * s)(rng);
        for (int i = 0; i < nm && p.T > 2 * T_CUT; ++i) {
          double Td = 1 / (1 / T_CUT - U() * (1 / T_CUT - 2 / p.T));
          double cost = std::sqrt(Td * (p.T + 2 * ME) / (p.T * (Td + 2 * ME)));
          Particle d{11, p.x, p.y, p.z, p.ux, p.uy, p.uz, Td, p.t};
          rotate(d.ux, d.uy, d.uz, std::min(1.0, cost), 2 * M_PI * U());
          stack.push_back(d);
          p.T -= Td;
        }
      }
      p.T -= el * s;
      if (p.T <= E_STOP) break;
      double pm = std::sqrt(p.T * (p.T + 2 * ME)), beta = pm / (p.T + ME);
      double t0 = 13.6 / (beta * pm) * std::sqrt(s / X0) * (1 + 0.038 * std::log(s / X0));
      double th = std::sqrt(-2 * std::log(std::max(U(), 1e-300))) * t0;   // 2D Gaussian: polar angle
      rotate(p.ux, p.uy, p.uz, std::cos(th), 2 * M_PI * U());
      if (move(p, s)) return;
    }
    if (p.pdg == -11) {                                       // positron annihilates at rest: 2 x 511 keV
      double cost = 2 * U() - 1, phi = 2 * M_PI * U(), st = std::sqrt(1 - cost * cost);
      double ux = st * std::cos(phi), uy = st * std::sin(phi), uz = cost;
      stack.push_back({22, p.x, p.y, p.z, ux, uy, uz, ME, p.t});
      stack.push_back({22, p.x, p.y, p.z, -ux, -uy, -uz, ME, p.t});
    }
  }

  void photon(Particle p) {
    while (p.T > G_STOP) {
      double lam = 1.0 / (RHO * interp_loglog(MU_E, MU_V, 24, p.T));
      double s = -lam * std::log(std::max(U(), 1e-300));
      if (move(p, s)) return;
      double r = U(), fpair = pair_frac(p.T), fpe = pe_frac(p.T);
      if (r < fpair) {                                        // pair
        double f = ME / p.T + (1 - 2 * ME / p.T) * U();
        for (int q : {11, -11}) {
          double Ek = (q == 11 ? f : 1 - f) * p.T - ME;
          if (Ek <= 0) continue;
          Particle e{q, p.x, p.y, p.z, p.ux, p.uy, p.uz, Ek, p.t};
          rotate(e.ux, e.uy, e.uz, std::cos(ME / (Ek + ME)), 2 * M_PI * U());
          stack.push_back(e);
        }
        return;
      }
      if (r < fpair + fpe * (1 - fpair)) {                    // photo-effect: an electron with (almost) all the energy
        if (p.T > E_STOP) stack.push_back({11, p.x, p.y, p.z, p.ux, p.uy, p.uz, p.T - 0.007, p.t});
        return;
      }
      // Compton (Klein-Nishina, as Geant4's G4KleinNishinaCompton)
      double e0m = p.T / ME, eps0 = 1 / (1 + 2 * e0m), eps0sq = eps0 * eps0;
      double a1 = -std::log(eps0), a2 = 0.5 * (1 - eps0sq), eps, epssq, onecost, sint2, g;
      do {
        if (a1 / (a1 + a2) > U()) { eps = std::exp(-a1 * U()); epssq = eps * eps; }
        else { epssq = eps0sq + (1 - eps0sq) * U(); eps = std::sqrt(epssq); }
        onecost = (1 - eps) / (eps * e0m);
        sint2 = onecost * (2 - onecost);
        g = 1 - eps * sint2 / (1 + epssq);
      } while (g < U());
      double phi = 2 * M_PI * U();
      double E1 = eps * p.T, Te = p.T - E1;
      double gx = p.ux * p.T, gy = p.uy * p.T, gz = p.uz * p.T;   // momentum of the incoming photon
      rotate(p.ux, p.uy, p.uz, 1 - onecost, phi);
      p.T = E1;
      if (Te > E_STOP) {                                      // electron: p_e = p_gamma - p_gamma'
        double ex = gx - p.ux * E1, ey = gy - p.uy * E1, ez = gz - p.uz * E1, n = std::sqrt(ex * ex + ey * ey + ez * ez);
        stack.push_back({11, p.x, p.y, p.z, ex / n, ey / n, ez / n, Te, p.t});
      }
    }
  }

  void muon(double E_GeV) {
    double E = E_GeV * 1e3, pmu = std::sqrt(E * E - MMU * MMU), beta = pmu / E, gam = E / MMU;
    double ux = std::sin(theta), uz = std::cos(theta), L = D / uz;
    double tmax = 2 * ME * beta * beta * gam * gam / (1 + 2 * gam * ME / MMU + (ME / MMU) * (ME / MMU));
    auto emit = [&](int pdg, double s, double T, double cost) {
      Particle q{pdg, s * ux, 0, s * uz, ux, 0, uz, T, s / C_CM_NS};
      rotate(q.ux, q.uy, q.uz, cost, 2 * M_PI * U());
      stack.push_back(q);
    };
    // delta rays
    if (tmax > T_CUT) {
      double rate = 0.5 * K * Z / A * RHO / (beta * beta) * (1 / T_CUT - 1 / tmax);
      int n = std::poisson_distribution<int>(rate * L)(rng);
      for (int i = 0; i < n; ++i) {
        double T;
        do { T = 1 / (1 / T_CUT - U() * (1 / T_CUT - 1 / tmax)); } while (U() > 1 - beta * beta * T / tmax + T * T / (2 * E * E));
        double pe = std::sqrt(T * (T + 2 * ME)), cost = std::min(1.0, T / pe * (E + ME) / pmu);
        emit(11, U() * L, T, cost);
      }
    }
    // direct pairs (pair energy eps from 2 MeV to E, dN/deps ~ 1/eps^2, normalised to b_pair rho E)
    double emin = 2.0, emax = E;
    if (emax > emin) {
      double c = B_PAIR * RHO * E / std::log(emax / emin);
      int n = std::poisson_distribution<int>(c * (1 / emin - 1 / emax) * L)(rng);
      for (int i = 0; i < n; ++i) {
        double eps = 1 / (1 / emin - U() * (1 / emin - 1 / emax)), s = U() * L, f = U();
        for (int q : {11, -11}) {
          double Ek = (q == 11 ? f : 1 - f) * eps - ME;
          if (Ek > E_STOP) emit(q, s, Ek, std::cos(ME / (Ek + ME)));
        }
      }
    }
    // muon bremsstrahlung (photons above K_CUT)
    double vmin = K_CUT / E;
    double rate = B_BREMS * RHO * (4.0 / 3 * std::log(1 / vmin) - 4.0 / 3 * (1 - vmin) + 0.5 * (1 - vmin * vmin));
    int n = std::poisson_distribution<int>(rate * L)(rng);
    for (int i = 0; i < n; ++i) {
      double v;
      do { v = vmin * std::pow(1 / vmin, U()); } while (U() * 4.0 / 3 > 4.0 / 3 - 4.0 / 3 * v + v * v);
      emit(22, U() * L, v * E, 1.0);
    }
    while (!stack.empty()) {
      Particle p = stack.back();
      stack.pop_back();
      if (p.pdg == 22) photon(p); else electron(p);
    }
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 6) { std::fprintf(stderr, "usage: emdebris E_GeV theta n seed out.bin [depth_cm]\n"); return 1; }
  double E = std::atof(argv[1]), theta = std::atof(argv[2]);
  int n = std::atoi(argv[3]);
  rng.seed(std::strtoull(argv[4], nullptr, 10));
  double D = argc > 6 ? std::atof(argv[6]) : 20.0;
  std::vector<Record> out;
  Model m{D, theta, D * std::tan(theta), D / std::cos(theta) / C_CM_NS, &out, 0, {}};
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < n; ++i) { m.ev = i; m.muon(E); }
  double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / n;
  FILE* f = std::fopen(argv[5], "wb");
  int32_t nn = n;
  std::fwrite(&nn, sizeof nn, 1, f);
  std::fwrite(out.data(), sizeof(Record), out.size(), f);
  std::fclose(f);
  std::fprintf(stderr, "E %g GeV theta %g: %d muons, %zu records, %.2f us per muon\n", E, theta, n, out.size(), us);
  return 0;
}
