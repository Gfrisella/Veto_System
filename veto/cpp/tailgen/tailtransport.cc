// tailtransport: Geant4 transport of muons from their production point in the ECN3 target to the plane
// where MuonsAndMatter's input sample is stored (z = -0.0045 m), and hadronic cross sections in the target.
//
// Geometry (metres; MuonsAndMatter muons_and_matter/lib/ship_muon_shield_customfield.py CreateTarget with
// z_start = -2.14, 0.5 cm water slits between blocks, diameter 30 cm; then the "IronAfterTarget" block up to
// z = 0, here a 2 x 2 m iron slab). World: air. Two targets (initialize(target=...)):
//   "W"   MuonsAndMatter today: N = 0 molybdenum, T = 18 tungsten blocks (last 50 cm)
//   "MoW" the original ECN3 design in the same function's comments (N = 13, T = 5, last block 35 cm):
//         13 molybdenum (TZM, here G4_Mo) blocks then 5 tungsten.
//   "Air" no target and no iron (world air): decays in flight only, for tests of the forced decays.
//   "SHiP2018" the target of the 2018 FairShip muon production (geometry_config.py targetOpt = 17): the same 13
//         Mo blocks, then 4 W (8, 8, 10, 35 cm) — 17 cm less tungsten than "MoW".
// Physics: FTFP_BERT. Only the primary muon is tracked (secondaries killed); recorded where it crosses the
// plane, then stopped.
//
//   initialize(seed)
//   transport(pdg, x, y, z, px, py, pz) -> dict(ok, x, y, z, px, py, pz)  (ok = reached the plane)
//   cascade(offsets, pdg, px, py, pz, vx, vy, vz, e_cut, p_min_mu) -> dict of the muons at the plane
//       event i = particles offsets[i] .. offsets[i+1]-1, all starting at (vx[i], vy[i], vz[i]) (the Pythia
//       primary interaction). Geant4 (FTFP_BERT) cascades every hadron above e_cut [GeV kinetic] (lower ones,
//       e+-, photons, neutrinos killed). FORCED DECAYS: every step of a pi+- / K+- above e_cut emits one muon
//       of the two-body decay at a uniform point of the step with weight
//       w = (track weight) BR(mu nu) step / lambda_dec, lambda_dec = (p / m) c tau; kept if p_mu > p_min_mu.
//       Real decay muons are killed (no double counting). Muons are recorded at the plane: ev, weight, x..pz,
//       parent (pdg of the decaying meson), z_decay, generation (0: meson from the Pythia event).
//   COMPLETE MODE (initialize(mode="complete"), needs Pythia 8: TT_WITH_PYTHIA): every muon source Geant4 and Pythia
//       know, each with increased probability and the weight that undoes it:
//         long-lived species with a muonic decay channel in Pythia's tables (pi, K+-, K_L, K_S, Lambda, Sigma-, Xi):
//           every step of the track emits one muon from its muonic channels (picked by BR, kinematics by Pythia's
//           decayer) with weight (track weight) BR_mu step / lambda_dec, lambda_dec from Geant4's lifetime; muons of
//           Geant4's own decays are killed (no double counting), the other products are kept
//         prompt species (c tau <= 10 mm, muonic channel: eta, omega, rho, phi, eta', J/psi, psi', Upsilon, D, D_s,
//           B, charm / beauty baryons, tau): decayed by Pythia where they appear (Pythia event or Geant4 cascade):
//           all channels without their muons (products tracked, prompt products decayed the same way) plus one
//           muonic decay with weight BR_mu
//         gamma -> mu mu (cross section x gamma_factor) and e+ e- -> mu mu (x ep_factor, e+ tracked above 43.7 GeV),
//           weight / factor; photons tracked above 4 GeV
//         muons of the Pythia event itself (Drell-Yan, gamma gamma -> mu mu) and muons made by any other Geant4 process
//           at their natural weight
//         resonances decayed INSIDE Geant4's hadronic models (they use Geant4's decay tables): the prompt species'
//           tables are replaced by Pythia's (as FairShip's AddDiMuonDecayChannelsToG4 did for omega, phi, rho), muonic
//           channels x RES_BOOST, their muons emitted as tag particles that carry resonance and channel, turned into
//           muons (weight / RES_BOOST) at their first step
//       Muons get proc (gen): 1 decay of a long-lived hadron, 2 prompt decay of a species from the Pythia event,
//       7 prompt decay of a species made in the Geant4 cascade, 8 resonance decayed inside a hadronic model,
//       3 gamma -> mu mu, 4 e+ e- -> mu mu, 5 muon of the Pythia event, 6 other process; parent (pdg); hf (a charm / beauty hadron among the ancestors).
//   ANALOG MODE (mode="analog"): the same set-up without any forcing (natural decays, muons kept at weight 1, factors
//       1): the unbiased reference for the complete mode.
//   inelastic_xs_per_m(pdg, E, material)   material "W", "Fe", "water": inelastic cross section [1/m]
//   target_layout()                         list of (z_start, z_end, material) [m]

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FTFP_BERT.hh"
#include "QGSP_BERT.hh"
#include "G4Box.hh"
#include "G4CrossSectionDataStore.hh"
#include "G4DynamicParticle.hh"
#include "G4Event.hh"
#include "G4HadronicProcess.hh"
#include "G4LogicalVolume.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4ProcessManager.hh"
#include "G4RunManager.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4Tubs.hh"
#include "G4SubtractionSolid.hh"
#include "G4UImanager.hh"
#include "G4UserStackingAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "Randomize.hh"
#include "G4SteppingManager.hh"
#include "G4EventManager.hh"
#include "G4TrackingManager.hh"
#include "G4VUserTrackInformation.hh"
#include "G4MuonPlus.hh"
#include "G4MuonMinus.hh"
#include "G4PhaseSpaceDecayChannel.hh"
#include "G4DecayProducts.hh"
#include "G4GammaConversionToMuons.hh"
#include "G4ProcessTable.hh"
#include "G4EmExtraPhysics.hh"
#include "G4Version.hh"
#if G4VERSION_NUMBER >= 1040                  // also builds against Geant4 10.0 (cpp/tailgen_g410): no dimuon options there
#include "G4EmParameters.hh"
#define TT_HAS_DIMUON 1
#endif
#include "G4VModularPhysicsList.hh"
#include "G4VPhysicsConstructor.hh"
#include "G4AnnihiToMuPair.hh"
#include "G4Positron.hh"
#include "G4Gamma.hh"
#include "G4PrimaryParticle.hh"
#include "G4PrimaryVertex.hh"
#include "G4VUserPrimaryParticleInformation.hh"
#include "G4DecayTable.hh"
#include "G4VDecayChannel.hh"
#ifdef TT_WITH_PYTHIA
#include "Pythia8/Pythia.h"
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double Z_TARGET = -2.14, Z_PLANE = -0.0045, Z_IRON_END = 0.0;
constexpr double TARGET_RADIUS = 0.15, WATER = 0.005;
std::vector<double> BLOCKS = {8., 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 5.0, 5.0,
                              6.5, 8., 8., 5., 8., 10., 20., 50.};          // cm
int N_MO = 0;                                                               // first N_MO blocks molybdenum
bool gShielded = false;                                                     // target "<name>+Fe": iron around it
bool gAir = false;                                                          // target "Air": no material (decay tests)
bool gFairShip = false;                                                     // target "FairShip": FairShip's target station
std::vector<std::tuple<double, double, std::string>> gLayout;            // m

// FairShip's target station (geometry/target_config.yaml, the default "version 1" design, and passive/ShipTargetStation.cxx;
// FairShip commit 5987c9d, 2026-11-01), in MuonsAndMatter's frame: target front face at z = -2.14 m, hadron absorber at 0.
// 19 tungsten plates (lengths L, gaps G of pressurised He at 90 C), each clad in 1.5 mm tantalum, diameter 25 cm; inside a
// steel 316L enclosure (66.9 mm, minus a 16 x 28 cm cut-out) and an Inconel 718 vessel (8 mm, inner diameter 40 cm, 6.2 cm
// before and after the plates) filled with the He; all in a vacuum box with copper proximity shielding (1.6 x 1.126 x 3 m:
// 55 cm front wall with a 20 cm hole for the beam, 25 cm side and back walls), copper above (60 cm) and below (54.5 cm),
// an iron pedestal; behind the box air to z = 0.
const std::vector<double> FS_L = {4.5, 1.8, 1.7, 1.7, 1.8, 1.9, 2.1, 1.9, 2.1, 2.4, 2.8, 3.3, 4.1, 5.6, 8.8, 17.2,
                                  28.7, 28.8, 28.8};                                                    // cm
const std::vector<double> FS_G = {0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45,
                                  0.48, 0.5, 0.68, 0.68, 0.0};                                          // cm

// the materials of all targets, by the names of target_layout() (also used by inelastic_xs_per_m)
G4Material* tt_material(const std::string& name) {
  auto* nist = G4NistManager::Instance();
  static const std::map<std::string, std::string> nistnames = {
      {"W", "G4_W"}, {"Mo", "G4_Mo"}, {"Fe", "G4_Fe"}, {"water", "G4_WATER"}, {"Ta", "G4_Ta"}, {"Cu", "G4_Cu"},
      {"air", "G4_AIR"}, {"vacuum", "G4_Galactic"}};
  auto it = nistnames.find(name);
  if (it != nistnames.end()) return nist->FindOrBuildMaterial(it->second);
  if (auto* m0 = G4Material::GetMaterial("TT_" + name, false)) return m0;
  if (name == "He") {                       // FairShip media.geo PressurisedHe90: 2.121e-3 g/cm3
    auto* m1 = new G4Material("TT_He", 2.121e-3 * g / cm3, 1, kStateGas, 363.15 * kelvin, 16 * bar);
    m1->AddElement(nist->FindOrBuildElement("He"), 1.0);
    return m1;
  }
  if (name == "Inconel") {                  // Inconel 718 (vessel), 8.19 g/cm3, mass fractions
    auto* m1 = new G4Material("TT_Inconel", 8.19 * g / cm3, 7);
    const std::vector<std::pair<const char*, double>> f = {{"Ni", 0.525}, {"Cr", 0.19}, {"Fe", 0.185}, {"Nb", 0.051},
                                                           {"Mo", 0.03}, {"Ti", 0.009}, {"Al", 0.010}};
    for (auto& e : f) m1->AddElement(nist->FindOrBuildElement(e.first), e.second);
    return m1;
  }
  if (name == "steel") {                    // steel 316L (enclosure), FairShip media.geo: 7.99 g/cm3
    auto* m1 = new G4Material("TT_steel", 7.99 * g / cm3, 6);
    const std::vector<std::pair<const char*, double>> f = {{"Fe", 0.66}, {"Cr", 0.17}, {"Ni", 0.12}, {"Mo", 0.025},
                                                           {"Mn", 0.02}, {"Si", 0.0075}};
    double sum = 0;
    for (auto& e : f) sum += e.second;
    for (auto& e : f) m1->AddElement(nist->FindOrBuildElement(e.first), e.second / sum);
    return m1;
  }
  throw std::runtime_error("unknown material " + name);
}

void build_fairship(G4LogicalVolume* worldLV) {
  auto* vac = tt_material("vacuum");
  auto* cu = tt_material("Cu");
  auto box = [](const std::string& n, double dx, double dy, double dz) { return new G4Box(n, 0.5 * dx * m, 0.5 * dy * m, 0.5 * dz * m); };
  const bool check = std::getenv("TT_CHECK_OVERLAPS") != nullptr;                      // Geant4's overlap check (tests)
  auto place = [check](G4LogicalVolume* lv, G4LogicalVolume* mother, double x, double y, double z) {
    new G4PVPlacement(nullptr, G4ThreeVector(x * m, y * m, z * m), lv, lv->GetName(), mother, false, 0, check);
  };
  double tlen = 0;
  for (size_t i = 0; i < FS_L.size(); ++i) tlen += (FS_L[i] + FS_G[i]) / 100;          // 1.5864 m
  const double s = Z_TARGET;                                                            // start of the target
  // vacuum box with the copper shielding (FairShip: target_vacuum_box, its centre y = -14.45 cm)
  const double W = 1.6, H = 1.126, LEN = 3.0, TH = 0.25, THF = 0.55, HOLE_R = 0.1, HOLE_Y = 0.735, TOP = 0.6, BOT = 0.545;
  const double zbox = s + tlen + 0.0961 + TH - 0.5 * LEN;                               // centre of the box
  const double yprox = -(HOLE_Y - 0.5 * H);                                             // proximity shielding centre
  const double ybox_lo = yprox - 0.5 * H - BOT, ybox_hi = yprox + 0.5 * H + TOP;
  auto* vbox = new G4LogicalVolume(box("TargetVacuumBox", W, ybox_hi - ybox_lo, LEN), vac, "TargetVacuumBox");
  const double ybc = 0.5 * (ybox_lo + ybox_hi);
  place(vbox, worldLV, 0, ybc, zbox);
  // proximity shielding: envelope minus the inner cavity (open in y) minus the beam hole in the front wall
  G4VSolid* prox = new G4SubtractionSolid("ProxEnvMinusInner", box("ProxEnv", W, H, LEN), box("ProxInner", W - 2 * TH, H + 0.01, LEN - THF - TH),
                                          nullptr, G4ThreeVector(0, 0, 0.5 * (THF - TH) * m));
  prox = new G4SubtractionSolid("ProximityShielding", prox, new G4Tubs("ProxHole", 0, HOLE_R * m, 0.5 * THF * m + 1 * mm, 0, 360 * deg),
                                nullptr, G4ThreeVector(0, (HOLE_Y - 0.5 * H) * m, (-0.5 * LEN + 0.5 * THF) * m));
  place(new G4LogicalVolume(prox, cu, "ProximityShielding"), vbox, 0, yprox - ybc, 0);
  place(new G4LogicalVolume(box("TopShielding", W, TOP, LEN), cu, "TopShielding"), vbox, 0, yprox + 0.5 * H + 0.5 * TOP - ybc, 0);
  place(new G4LogicalVolume(box("BottomShielding", W, BOT, LEN), cu, "BottomShielding"), vbox, 0, yprox - 0.5 * H - 0.5 * BOT - ybc, 0);
  place(new G4LogicalVolume(box("Pedestal", 1.07, 0.15, 2.17), tt_material("Fe"), "Pedestal"), vbox, 0, -HOLE_Y + 0.075 - ybc,
        -0.5 * LEN + 0.565 + 0.5 * 2.17);
  // Inconel vessel filled with He, the steel enclosure and the clad plates inside it (all on the beam axis, y = 0)
  const double VT = 0.008, VR = 0.2, VSHIFT = 0.062, vlen = tlen + 2 * VSHIFT;
  const double zv = s - VSHIFT + 0.5 * vlen - zbox;                                     // vessel centre in the box
  auto tub = [](const std::string& n, double r0, double r1, double len) { return new G4Tubs(n, r0 * m, r1 * m, 0.5 * len * m, 0, 360 * deg); };
  auto* inc = tt_material("Inconel");
  place(new G4LogicalVolume(tub("VesselWall", VR, VR + VT, vlen), inc, "VesselWall"), vbox, 0, -ybc, zv);
  place(new G4LogicalVolume(tub("VesselFront", 0, VR + VT, VT), inc, "VesselFront"), vbox, 0, -ybc, zv - 0.5 * vlen - 0.5 * VT);
  place(new G4LogicalVolume(tub("VesselBack", 0, VR + VT, VT), inc, "VesselBack"), vbox, 0, -ybc, zv + 0.5 * vlen + 0.5 * VT);
  auto* he = new G4LogicalVolume(tub("HeVolume", 0, VR, vlen), tt_material("He"), "HeVolume");
  place(he, vbox, 0, -ybc, zv);
  const double R = 0.125, CL = 0.0015;
  G4VSolid* encl = new G4SubtractionSolid("Enclosure", tub("EnclTube", R + 0.0001, R + 0.0001 + 0.0669, tlen),
                                          box("EnclCut", 0.16, 0.28, tlen + 0.01));
  place(new G4LogicalVolume(encl, tt_material("steel"), "Enclosure"), he, 0, 0, -0.5 * vlen + VSHIFT + 0.5 * tlen);
  auto* ta = tt_material("Ta");
  auto* w = tt_material("W");
  double z = 0;                                                                         // from the target front face
  gLayout.clear();
  gLayout.emplace_back(s - VSHIFT - VT, s - VSHIFT, "Inconel");
  gLayout.emplace_back(s - VSHIFT, s, "He");
  for (size_t i = 0; i < FS_L.size(); ++i) {
    const double L = FS_L[i] / 100;
    auto* clad = new G4LogicalVolume(tub("Clad" + std::to_string(i), 0, R, L), ta, "Clad" + std::to_string(i));
    place(new G4LogicalVolume(tub("Plate" + std::to_string(i), 0, R - CL, L - 2 * CL), w, "Plate" + std::to_string(i)), clad, 0, 0, 0);
    place(clad, he, 0, 0, -0.5 * vlen + VSHIFT + z + 0.5 * L);
    gLayout.emplace_back(s + z, s + z + CL, "Ta");
    gLayout.emplace_back(s + z + CL, s + z + L - CL, "W");
    gLayout.emplace_back(s + z + L - CL, s + z + L, "Ta");
    if (FS_G[i] > 0) gLayout.emplace_back(s + z + L, s + z + L + FS_G[i] / 100, "He");
    z += L + FS_G[i] / 100;
  }
  gLayout.emplace_back(s + tlen, s + tlen + VSHIFT, "He");
  gLayout.emplace_back(s + tlen + VSHIFT, s + tlen + VSHIFT + VT, "Inconel");
  gLayout.emplace_back(s + tlen + VSHIFT + VT, zbox + 0.5 * LEN - TH, "vacuum");
  gLayout.emplace_back(zbox + 0.5 * LEN - TH, zbox + 0.5 * LEN, "Cu");
  gLayout.emplace_back(zbox + 0.5 * LEN, Z_IRON_END, "air");
}

class Detector : public G4VUserDetectorConstruction {
 public:
  G4VPhysicalVolume* Construct() override {
    auto* nist = G4NistManager::Instance();
    auto* air = nist->FindOrBuildMaterial("G4_AIR");
    auto* w = nist->FindOrBuildMaterial("G4_W");
    auto* water = nist->FindOrBuildMaterial("G4_WATER");
    auto* fe = nist->FindOrBuildMaterial("G4_Fe");
    auto* mo = nist->FindOrBuildMaterial("G4_Mo");
    auto* worldLV = new G4LogicalVolume(new G4Box("World", 5 * m, 5 * m, 5 * m), air, "World");
    auto* worldPV = new G4PVPlacement(nullptr, {}, worldLV, "World", nullptr, false, 0);
    gLayout.clear();
    if (gFairShip) {
      build_fairship(worldLV);
      return worldPV;
    }
    if (gAir) {
      gLayout.emplace_back(Z_TARGET, Z_IRON_END, "air");
      return worldPV;
    }
    double z = Z_TARGET;
    // gShielded: the target blocks sit inside an iron block (2 x 2 m, the target's length) instead of in air, as the
    // shielding around the FairShip target: hadrons leaving the target sideways are absorbed instead of decaying in air
    double tlen = 0;
    for (size_t i = 0; i < BLOCKS.size(); ++i) tlen += BLOCKS[i] / 100 + (i > 0 ? WATER : 0);
    G4LogicalVolume* mother = worldLV;
    double zc = 0;
    if (gShielded) {
      zc = Z_TARGET + 0.5 * tlen;
      mother = new G4LogicalVolume(new G4Box("TargetShield", 1 * m, 1 * m, 0.5 * tlen * m), fe, "TargetShield");
      new G4PVPlacement(nullptr, G4ThreeVector(0, 0, zc * m), mother, "TargetShield", worldLV, false, 0);
    }
    auto place_tub = [&](double z0, double len, G4Material* mat, const std::string& name) {
      auto* s = new G4Tubs(name, 0, TARGET_RADIUS * m, 0.5 * len * m, 0, 360 * deg);
      new G4PVPlacement(nullptr, G4ThreeVector(0, 0, (z0 + 0.5 * len - zc) * m), new G4LogicalVolume(s, mat, name), name,
                        mother, false, 0);
      gLayout.emplace_back(z0, z0 + len, mat == w ? "W" : (mat == mo ? "Mo" : "water"));
    };
    for (size_t i = 0; i < BLOCKS.size(); ++i) {
      if (i > 0) {
        place_tub(z, WATER, water, "Water" + std::to_string(i));
        z += WATER;
      }
      place_tub(z, BLOCKS[i] / 100, int(i) < N_MO ? mo : w, "Block" + std::to_string(i));
      z += BLOCKS[i] / 100;
    }
    const double len = Z_IRON_END - z;
    new G4PVPlacement(nullptr, G4ThreeVector(0, 0, (z + 0.5 * len) * m),
                      new G4LogicalVolume(new G4Box("Iron", 1 * m, 1 * m, 0.5 * len * m), fe, "Iron"), "Iron", worldLV,
                      false, 0);
    gLayout.emplace_back(z, Z_IRON_END, "Fe");
    return worldPV;
  }
};

struct Inputs { std::vector<int> pdg; std::vector<double> x, y, z, px, py, pz; } gIn;
std::vector<double> gE;                    // energies of the cascade inputs (optional: masses of prompt species)
std::vector<double> gW;                    // weights of the cascade inputs (optional: importance sampling in the generator)
// importance sampling of the cascade's new tracks (not muons), as a weight window: a track's target weight is 1 / q,
// q = min(1, max(QMIN, (E / E1)^N) x max(1, (pT / PT0)^2)) (as pythia_mesons "roulette="). A track lighter than its target
// survives with probability w q and gets the target weight; a track already at or above it continues unchanged (weights
// never compound down the cascade: at most 1 / QMIN times the event's).
bool gRR = false;
const bool gCheckE = std::getenv("TT_CHECK_ENERGY") != nullptr;   // diagnostics: energy above the parent's / the beam's
// decay splitting: each prompt decay makes gSplit independent muonic decays, each with weight w BR_mu / gSplit
int gSplit = 1;
// weight window of the recorded muons (rows): per (p, pT) cell at the plane, a muon with weight below w_min is kept with
// probability w / w_min and weight w_min (unbiased; removes the many tiny-weight rows of the forced decays)
bool gWW = false;
std::vector<double> gWWp, gWWt, gWWmin;
double gRq[4] = {0, 150, 3, 0.5};
double rr_q(double e_gev, double pt_gev) {
  return std::min(1.0, std::max(gRq[0], std::pow(e_gev / gRq[1], gRq[2])) * std::max(1.0, (pt_gev / gRq[3]) * (pt_gev / gRq[3])));
}
// cascade mode
bool gCascade = false;
double gECut = 130.0, gPMinMu = 130.0;     // GeV
double gL0 = 0.0;                          // m: Russian roulette length for the forced decays (0 = every step)
bool gDimuon = false;                      // forced eta/omega/rho/eta'/phi -> mu mu (X) and boosted gamma -> mu mu
double gEGamma = 4.0;                      // GeV: photons tracked above this (dimuon mode)
double gGammaFactor = 500.0;               // gamma -> mu mu cross-section factor (muon weight / factor)
bool gPiK = true;                          // forced pi / K -> mu nu
// dimuon channels (PDG BRs; 3-body by phase space, as FairShip's Geant4 tables): pdg -> (BR, daughters)
struct DiMuChannel { double br; std::vector<std::string> d; G4PhaseSpaceDecayChannel* ch = nullptr; };
std::map<int, std::vector<DiMuChannel>> gDiMu;
std::vector<long> gOff;                    // event offsets into gIn (particles); vertex per event in gVtx
std::vector<double> gVx, gVy, gVz;
struct MuOut { std::vector<int> ev, parent, gen, q, hf, ch; std::vector<double> w, x, y, z, px, py, pz, zdec, pcre, pcx, pcy, pcz; } gMu;
// interaction logging (cascade charm): inelastic interactions of p, n, pi+-, K+-, K_L, K_S above gLogEmin
bool gLogInt = false;
double gLogEmin = 34.0;   // GeV (FairShip makeCascade.py pbeaml for charm)
struct IntOut { std::vector<int> ev, pdg; std::vector<double> w, x, y, z, px, py, pz; } gInt;
bool gVmFairShip = false;
// forced K decays besides K -> mu nu (phase space): K+- -> pi0 mu nu (K_mu3), K_L -> pi+- mu-+ nu
struct KChannel { double br; std::vector<std::string> d; G4PhaseSpaceDecayChannel* ch = nullptr; };
std::map<int, std::vector<KChannel>> gKmu;     // keyed by signed pdg   // dimuon forcing as the old FairShip production: Pythia-level (primary) eta/omega/rho/eta'/phi,
                            // cascade-level only omega/phi/rho (Geant4 decay tables copied from Pythia; eta not)

class ForcedInfo : public G4VUserTrackInformation {   // attached to the forced-decay muons
 public:
  ForcedInfo(int parent, double zdec, int gen, double pc, int hf = 0, int ch = -1, G4ThreeVector pv = G4ThreeVector())
      : parent_(parent), zdec_(zdec), gen_(gen), pc_(pc), hf_(hf), ch_(ch), pv_(pv) {}
  int parent_; double zdec_; int gen_; double pc_;     // pc_: muon momentum at creation [GeV]
  int hf_;                                             // an open charm / beauty hadron among the ancestors
  int ch_;                                             // prompt decays: products besides muons and neutrinos (0: pure mu mu)
  G4ThreeVector pv_;                                   // momentum vector at creation [GeV] (0 if not recorded)
};
class TakenInfo : public G4VUserTrackInformation {};   // products of our own (Pythia) decays: kept by the stacking
class PrimInfo : public G4VUserPrimaryParticleInformation {   // primaries made by the complete / analog generator
 public:
  PrimInfo(int proc, int parent, int hf, double zdec, int ch = -1) : proc_(proc), parent_(parent), hf_(hf), zdec_(zdec), ch_(ch) {}
  void Print() const override {}
  int proc_, parent_, hf_; double zdec_; int ch_;
};

// ---------- complete / analog mode ----------
int gMode = 0;                                 // 0 legacy, 1 complete (forced, weighted), 2 analog
double gEpFactor = 100.0;                      // e+ e- -> mu mu cross-section factor
constexpr double E_POSITRON_MIN = 43.7;        // GeV: threshold of e+ e- -> mu mu on electrons at rest
struct Species { double brmu; double brmu_bar; double tau0_mm; bool prompt; };   // BR to muonic channels: particle, antiparticle
std::unordered_map<int, Species> gSpec;        // |pdg| -> muonic branching ratios, Pythia tau0, prompt or long-lived
double br_mu(const Species& sp, int pdg) { return pdg < 0 ? sp.brmu_bar : sp.brmu; }
// Geant4's own muonic decay channels of a long-lived species (used for the forced decays where Geant4 has them: its
// kinematics, e.g. K_l3 form factors; Pythia's only for channels missing in Geant4: K_S, hyperons)
struct G4Mu { double br = 0; std::vector<std::pair<double, G4VDecayChannel*>> ch; bool done = false; };
std::unordered_map<int, G4Mu> gG4Mu;           // keyed by signed pdg
struct Tag { int pdg; int hf; };
std::unordered_map<int, Tag> gTags;            // per event: track id -> pdg, heavy-flavour ancestry
std::unordered_set<int> gTaken;                // per event: tracks decayed by us (their Geant4 products are dropped)
struct Diag { double unknown_w = 0, unforced_mu_w = 0, py_fail = 0; long py_retry = 0; std::map<int, double> unforced_by_parent; long taken = 0;
              long tags = 0, tags_hadronic = 0, hadronic_mu = 0; } gDiag;
// tag particles of muons from resonance decays inside the hadronic models (see the header)
struct MuTag { int parent; int ch; };
std::unordered_map<const G4ParticleDefinition*, MuTag> gMuTag;
std::map<std::tuple<int, int, int>, G4ParticleDefinition*> gTagDef;   // (signed parent, channel index, muon pdg) -> tag
double gResBoost = 100.0;
bool is_hf(int pdg) {                          // open charm / beauty (quarkonia like J/psi, Upsilon are not)
  const int a = std::abs(pdg) % 10000;
  if (a >= 1000) return (a / 1000) % 10 >= 4 || (a / 100) % 10 >= 4 || (a / 10) % 10 >= 4;   // baryons
  const int q1 = (a / 100) % 10, q2 = (a / 10) % 10;                                           // mesons
  return std::max(q1, q2) >= 4 && q1 != q2;
}
#ifdef TT_WITH_PYTHIA
Pythia8::Pythia *gPyAll = nullptr, *gPyMu = nullptr;   // decayers: all channels / muonic channels only
#endif
struct Outputs { std::vector<int> ok; std::vector<double> x, y, z, px, py, pz; } gOut;

// A particle made by our decays: muons (muon = true: proc, parent) or products to track.
struct Item { int pdg; G4ThreeVector p; G4ThreeVector pos; double t; double w; int proc; int parent; int hf; bool muon; int ch = -1; };

#ifdef TT_WITH_PYTHIA
struct PyProd { int id; G4ThreeVector p; double m; };
// One decay of particle id (lab momentum p [GeV], mass [GeV]) by the decayer py: its final products. Every species that
// is long-lived or prompt-muonic has mayDecay off in the decayers, except the one being decayed (switched on here).
bool py_decay(Pythia8::Pythia* py, int id, const G4ThreeVector& p, double mass, std::vector<PyProd>& out) {
  // Every decay is validated: Pythia's own check (Check:event on) and ours, finite momenta and the products' energy equal to
  // the parent's (0.1 %). Pythia 8.223 occasionally returns broken multi-body B decays (~3e-7: NaN momenta, products far
  // above the parent's energy); such a decay is redone with new random numbers, at most 5 times, and counted.
  auto& pd = py->particleData;
  const double E0 = std::sqrt(p.mag2() + mass * mass);
  const size_t n0 = out.size();
  for (int attempt = 0; attempt < 5; ++attempt) {
    out.resize(n0);
    pd.mayDecay(id, true);
    py->event.reset();
    py->event.append(id, 1, 0, 0, p.x(), p.y(), p.z(), E0, mass);
    const bool ok = py->next();
    pd.mayDecay(id, false);
    if (!ok) { gDiag.py_retry += 1; continue; }
    double sum = 0, emax = 0;
    bool finite = true;
    for (int i = 0; i < py->event.size(); ++i) {
      const Pythia8::Particle& q = py->event[i];
      if (!q.isFinal()) continue;
      if (!(std::isfinite(q.px()) && std::isfinite(q.py()) && std::isfinite(q.pz()) && std::isfinite(q.e()))) finite = false;
      sum += q.e();
      emax = std::max(emax, q.e());
      out.push_back(PyProd{q.id(), G4ThreeVector(q.px(), q.py(), q.pz()), q.m()});
    }
    if (finite && std::abs(sum - E0) <= 1e-3 * E0 + 1e-6 && emax <= 1.001 * E0) return true;
    gDiag.py_retry += 1;
    if (gCheckE)
      std::fprintf(stderr, "TT_CHECK py_decay id %d E %.3f mass %.4f: products sum E %.3f, max %.3f, attempt %d rejected\n",
                   id, E0, mass, sum, emax, attempt);
  }
  out.resize(n0);
  gDiag.py_fail += 1;
  return false;
}

// Decay of a prompt species (weight w, at pos [Geant4 units] after its flight): all channels without their muons (analog:
// with them), prompt products decayed the same way, and (complete) one muonic decay with weight w BR_mu.
void prompt_decay(int id, const G4ThreeVector& p, double mass, const G4ThreeVector& pos, double t, double w, int hf,
                  std::vector<Item>& out, int depth = 0, int proc = 2) {
  const Species& sp = gSpec.at(std::abs(id));
  hf = hf || is_hf(id);
  const double P = p.mag(), E = std::sqrt(P * P + mass * mass);
  const double l = -std::log(1 - G4UniformRand()) * (P / mass) * sp.tau0_mm * mm;     // flight (material ignored)
  const G4ThreeVector x = pos + l * p.unit();
  const double tt = t + l / (c_light * P / E);
  std::vector<PyProd> prod;
  auto extra = [](const std::vector<PyProd>& v) {                  // products besides muons and neutrinos
    int n = 0;
    for (const PyProd& q : v) { const int a = std::abs(q.id); if (a != 13 && a != 12 && a != 14 && a != 16) ++n; }
    return n;
  };
  if (py_decay(gPyAll, id, p, mass, prod)) {
    const int ch = extra(prod);
    for (const PyProd& q : prod) {
      const int aq = std::abs(q.id);
      if (aq == 12 || aq == 14 || aq == 16) continue;
      if (aq == 13) {
        if (gMode == 2) out.push_back(Item{q.id, q.p, x, tt, w, proc, id, hf, true, ch});
        continue;
      }
      auto it = gSpec.find(aq);
      if (it != gSpec.end() && it->second.prompt && depth < 10) {
        prompt_decay(q.id, q.p, q.m, x, tt, w, hf, out, depth + 1, proc);
        continue;
      }
      out.push_back(Item{q.id, q.p, x, tt, w, 0, id, hf, false});
    }
  }
  const double bm = br_mu(sp, id);
  if (gMode == 1 && bm > 0) {
    for (int k = 0; k < gSplit; ++k) {                            // decay splitting: gSplit muonic decays, weight / gSplit
      prod.clear();
      if (py_decay(gPyMu, id, p, mass, prod)) {
        const int ch = extra(prod);
        for (const PyProd& q : prod)
          if (std::abs(q.id) == 13) out.push_back(Item{q.id, q.p, x, tt, w * bm / gSplit, proc, id, hf, true, ch});
      }
    }
  }
}
#endif

// Items into the Geant4 event: as primaries (vertex of the Pythia event) ...
void add_primary(G4Event* ev, const Item& it) {
  auto* def = G4ParticleTable::GetParticleTable()->FindParticle(it.pdg);
  if (def == nullptr) { gDiag.unknown_w += it.w; return; }
  auto* v = new G4PrimaryVertex(it.pos, it.t);
  auto* pp = new G4PrimaryParticle(def, it.p.x() * GeV, it.p.y() * GeV, it.p.z() * GeV);
  pp->SetWeight(it.w);
  pp->SetUserInformation(new PrimInfo(it.muon ? it.proc : 0, it.parent, it.hf, it.pos.z() / m, it.ch));
  v->SetPrimary(pp);
  ev->AddPrimaryVertex(v);
}

// ... or as secondaries of the current step (track tr)
void push_items(G4Track* tr, const std::vector<Item>& items) {
  auto* sec = G4EventManager::GetEventManager()->GetTrackingManager()->GetSteppingManager()->GetfSecondary();
  for (const Item& it : items) {
    auto* def = G4ParticleTable::GetParticleTable()->FindParticle(it.pdg);
    if (def == nullptr) { gDiag.unknown_w += it.w; continue; }
    if (it.muon && it.p.mag() < gPMinMu) continue;
    auto* nt = new G4Track(new G4DynamicParticle(def, it.p * GeV), it.t, it.pos);
    nt->SetWeight(it.w);
    nt->SetParentID(tr->GetTrackID());
    if (it.muon) nt->SetUserInformation(new ForcedInfo(it.parent, it.pos.z() / m, it.proc, it.p.mag(), it.hf, it.ch, it.p));
    else nt->SetUserInformation(new TakenInfo());
    sec->push_back(nt);
  }
}

class Generator : public G4VUserPrimaryGeneratorAction {
 public:
  Generator() : gun_(std::make_unique<G4ParticleGun>(1)) {}
  void GeneratePrimaries(G4Event* ev) override {
    if (gCascade && gMode != 0) {             // complete / analog: primaries with weights, prompt species decayed here
      gTags.clear();
      gTaken.clear();
      const size_t e = ev->GetEventID();
      const G4ThreeVector vtx = G4ThreeVector(gVx[e], gVy[e], gVz[e]) * m;
      for (long j = gOff[e]; j < gOff[e + 1]; ++j) {
        const int id = gIn.pdg[j];
        const G4ThreeVector p(gIn.px[j], gIn.py[j], gIn.pz[j]);
        auto it = gSpec.find(std::abs(id));
        if (it != gSpec.end() && it->second.prompt) {
#ifdef TT_WITH_PYTHIA
          double mass = gPyAll->particleData.m0(id);
          if (!gE.empty()) mass = std::sqrt(std::max(gE[j] * gE[j] - p.mag2(), 1e-12));
          std::vector<Item> items;
          prompt_decay(id, p, mass, vtx, 0.0, gW.empty() ? 1.0 : gW[j], 0, items);
          for (const Item& x : items) add_primary(ev, x);
#endif
          continue;
        }
        add_primary(ev, Item{id, p, vtx, 0.0, gW.empty() ? 1.0 : gW[j], std::abs(id) == 13 ? 5 : 0, 0, 0, std::abs(id) == 13});
      }
      return;
    }
    if (gCascade) {
      const size_t e = ev->GetEventID();
      for (long j = gOff[e]; j < gOff[e + 1]; ++j) {
        auto* def = G4ParticleTable::GetParticleTable()->FindParticle(gIn.pdg[j]);
        if (def == nullptr) continue;
        G4ThreeVector p(gIn.px[j], gIn.py[j], gIn.pz[j]);
        gun_->SetParticleDefinition(def);
        gun_->SetParticleMomentumDirection(p.unit());
        const double mass = def->GetPDGMass();
        gun_->SetParticleEnergy(std::sqrt(p.mag2() * GeV * GeV + mass * mass) - mass);
        gun_->SetParticlePosition(G4ThreeVector(gVx[e], gVy[e], gVz[e]) * m);
        gun_->GeneratePrimaryVertex(ev);
      }
      return;
    }
    const size_t i = ev->GetEventID();
    auto* def = G4ParticleTable::GetParticleTable()->FindParticle(gIn.pdg[i]);
    G4ThreeVector p(gIn.px[i], gIn.py[i], gIn.pz[i]);
    gun_->SetParticleDefinition(def);
    gun_->SetParticleMomentumDirection(p.unit());
    const double mass = def->GetPDGMass();
    gun_->SetParticleEnergy(std::sqrt(p.mag2() * GeV * GeV + mass * mass) - mass);
    gun_->SetParticlePosition(G4ThreeVector(gIn.x[i], gIn.y[i], gIn.z[i]) * m);
    gun_->GeneratePrimaryVertex(ev);
  }
 private:
  std::unique_ptr<G4ParticleGun> gun_;
};


void push_muon(G4Track* parent, const G4ThreeVector& p, int charge_sign, const G4ThreeVector& pos, double t, double w,
               int parent_pdg) {
  if (p.mag() < gPMinMu) return;
  G4ParticleDefinition* mu = charge_sign > 0 ? static_cast<G4ParticleDefinition*>(G4MuonPlus::Definition())
                                             : static_cast<G4ParticleDefinition*>(G4MuonMinus::Definition());
  const double mmu = 0.1056584;
  auto* dp = new G4DynamicParticle(mu, p.unit(), (std::sqrt(p.mag2() + mmu * mmu) - mmu) * GeV);
  auto* nt = new G4Track(dp, t, pos);
  nt->SetWeight(w);
  nt->SetParentID(parent->GetTrackID());
  nt->SetUserInformation(new ForcedInfo(parent_pdg, pos.z() / m, 3, p.mag()));
  G4EventManager::GetEventManager()->GetTrackingManager()->GetSteppingManager()->GetfSecondary()->push_back(nt);
}

void emit_dimuon(G4Track* tr, const G4StepPoint* pre, int a) {
  auto& chans = gDiMu[a];
  // FairShip mode: its boost only took channels whose first two products are muons (FixedTargetGenerator.cxx):
  // pure mu mu for eta / omega (no Dalitz eta -> mu mu gamma, omega -> pi0 mu mu); eta' -> mu mu gamma kept
  auto use = [&](const DiMuChannel& c) { return !gVmFairShip || a == 331 || c.d.size() == 2; };
  double btot = 0;
  for (auto& c : chans) if (use(c)) btot += c.br;
  if (btot <= 0) return;
  double r = G4UniformRand() * btot;
  DiMuChannel* pick = nullptr;
  for (auto& c : chans) { if (!use(c)) continue; pick = &c; if (r < c.br) break; r -= c.br; }
  const double mass = tr->GetDefinition()->GetPDGMass();
  G4DecayProducts* prod = pick->ch->DecayIt(mass);
  const G4ThreeVector pv = pre->GetMomentum();
  const double E = std::sqrt(pv.mag2() + mass * mass);
  prod->Boost(E, pv.unit());
  for (int i = 0; i < prod->entries(); ++i) {
    const G4DynamicParticle* d = (*prod)[i];
    const int q = d->GetDefinition()->GetPDGEncoding();
    if (std::abs(q) != 13) continue;
    push_muon(tr, d->GetMomentum() / GeV, q < 0 ? +1 : -1, pre->GetPosition(), pre->GetGlobalTime(),
              tr->GetWeight() * btot, a);
  }
  delete prod;
}

void forced_kmu3(const G4Step* step, G4Track* tr, int pdg) {
  const G4StepPoint* pre = step->GetPreStepPoint();
  const G4StepPoint* post = step->GetPostStepPoint();
  if (pre->GetKineticEnergy() < gECut * GeV || !gKmu.count(pdg)) return;
  const double mass = tr->GetDefinition()->GetPDGMass();
  const double ctau = std::abs(pdg) == 130 ? 15.34 : 3.7118;                     // m
  const G4ThreeVector pv = pre->GetMomentum();
  const double lam_dec = pv.mag() / mass * ctau;
  const double len = step->GetStepLength() / m;
  if (len <= 0) return;
  double btot = 0;
  for (auto& c : gKmu[pdg]) btot += c.br;
  double w = tr->GetWeight() * btot * len / lam_dec;
  if (gL0 > 0 && len < gL0) {
    const double q = len / gL0;
    if (G4UniformRand() > q) return;
    w /= q;
  }
  double r = G4UniformRand() * btot;
  KChannel* pick = &gKmu[pdg].back();
  for (auto& c : gKmu[pdg]) { if (r < c.br) { pick = &c; break; } r -= c.br; }
  G4DecayProducts* prod = pick->ch->DecayIt(mass);
  prod->Boost(std::sqrt(pv.mag2() + mass * mass), pv.unit());
  const double f = G4UniformRand();
  const G4ThreeVector pos = pre->GetPosition() + f * (post->GetPosition() - pre->GetPosition());
  const double t = pre->GetGlobalTime() + f * (post->GetGlobalTime() - pre->GetGlobalTime());
  for (int i = 0; i < prod->entries(); ++i) {
    const G4DynamicParticle* d = (*prod)[i];
    const int q = d->GetDefinition()->GetPDGEncoding();
    if (std::abs(q) == 13) push_muon(tr, d->GetMomentum() / GeV, q < 0 ? +1 : -1, pos, t, w, pdg);
  }
  delete prod;
}

void record_muon(const G4Step* step, G4Track* tr, int pdg) {
  const G4StepPoint* pre = step->GetPreStepPoint();
  const G4StepPoint* post = step->GetPostStepPoint();
  const double z0 = pre->GetPosition().z() / m, z1 = post->GetPosition().z() / m;
  if (!(z0 < Z_PLANE && z1 >= Z_PLANE)) return;
  const double f = (Z_PLANE - z0) / (z1 - z0);
  const G4ThreeVector x = pre->GetPosition() + f * (post->GetPosition() - pre->GetPosition());
  const G4ThreeVector p = pre->GetMomentum() + f * (post->GetMomentum() - pre->GetMomentum());
  auto* info = static_cast<ForcedInfo*>(tr->GetUserInformation());
  double wrec = tr->GetWeight();
  if (gWW) {
    const double pg = p.mag() / GeV, ptg = p.perp() / GeV;
    const int i = std::max(0, std::min(int(std::upper_bound(gWWp.begin(), gWWp.end(), pg) - gWWp.begin()) - 1, int(gWWp.size()) - 2));
    const int j = std::max(0, std::min(int(std::upper_bound(gWWt.begin(), gWWt.end(), ptg) - gWWt.begin()) - 1, int(gWWt.size()) - 2));
    const double wmin = gWWmin[i * (gWWt.size() - 1) + j];
    if (wrec < wmin) {
      if (G4UniformRand() > wrec / wmin) { tr->SetTrackStatus(fStopAndKill); return; }
      wrec = wmin;
    }
  }
  gMu.ev.push_back(G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID()); gMu.w.push_back(wrec);
  gMu.x.push_back(x.x() / m); gMu.y.push_back(x.y() / m); gMu.z.push_back(Z_PLANE);
  gMu.px.push_back(p.x() / GeV); gMu.py.push_back(p.y() / GeV); gMu.pz.push_back(p.z() / GeV);
  gMu.parent.push_back(info ? info->parent_ : 0); gMu.zdec.push_back(info ? info->zdec_ : 0);
  gMu.gen.push_back(info ? info->gen_ : -1);
  gMu.pcre.push_back(info ? info->pc_ : 0);
  gMu.hf.push_back(info ? info->hf_ : 0);
  gMu.ch.push_back(info ? info->ch_ : -1);
  gMu.pcx.push_back(info ? info->pv_.x() : 0); gMu.pcy.push_back(info ? info->pv_.y() : 0); gMu.pcz.push_back(info ? info->pv_.z() : 0);
  gMu.q.push_back(pdg);                                            // the muon's own pdg (13 mu-, -13 mu+)
  tr->SetTrackStatus(fStopAndKill);
}

#ifdef TT_WITH_PYTHIA
// complete mode: the forced muonic decay of a long-lived species in this step (weight BR_mu step / lambda_dec)
void forced_longlived(const G4Step* step, G4Track* tr, int pdg, const Species& sp, int hf) {
  const G4StepPoint* pre = step->GetPreStepPoint();
  const G4StepPoint* post = step->GetPostStepPoint();
  if (pre->GetTotalEnergy() < gECut * GeV) return;
  const double len = step->GetStepLength() / m;
  if (len <= 0) return;
  const G4ParticleDefinition* def = tr->GetDefinition();
  const double mass = def->GetPDGMass() / GeV;
  double ctau = def->GetPDGLifeTime() * c_light / m;                 // Geant4's lifetime: as its own decays
  if (!(ctau > 0)) ctau = sp.tau0_mm / 1000.0;
  const G4ThreeVector pv = pre->GetMomentum() / GeV;
  G4Mu& g = gG4Mu[pdg];
  if (!g.done) {                                                     // Geant4's muonic channels of this species
    g.done = true;
    G4DecayTable* dt = def->GetDecayTable();
    double tot = 0;
    for (int i = 0; dt && i < dt->entries(); ++i) {
      G4VDecayChannel* c = dt->GetDecayChannel(i);
      tot += c->GetBR();
      bool mu = false;
      for (int j = 0; j < c->GetNumberOfDaughters(); ++j) mu = mu || std::abs(c->GetDaughter(j)->GetPDGEncoding()) == 13;
      if (mu) { g.br += c->GetBR(); g.ch.emplace_back(c->GetBR(), c); }
    }
    if (tot > 0) g.br /= tot;
  }
  const double bm = g.br > 0 ? g.br : br_mu(sp, pdg);
  double w = tr->GetWeight() * bm * len / (pv.mag() / mass * ctau);
  if (gL0 > 0 && len < gL0) {
    const double q = len / gL0;
    if (G4UniformRand() > q) return;
    w /= q;
  }
  const double f = G4UniformRand();
  const G4ThreeVector pos = pre->GetPosition() + f * (post->GetPosition() - pre->GetPosition());
  const double t = pre->GetGlobalTime() + f * (post->GetGlobalTime() - pre->GetGlobalTime());
  std::vector<Item> items;
  if (g.br > 0) {                                                    // Geant4's channel and kinematics
    double sum = 0;
    for (auto& c : g.ch) sum += c.first;
    double r = G4UniformRand() * sum;
    G4VDecayChannel* pick = g.ch.back().second;
    for (auto& c : g.ch) { if (r < c.first) { pick = c.second; break; } r -= c.first; }
    G4DecayProducts* dp = pick->DecayIt(def->GetPDGMass());
    dp->Boost(std::sqrt(pre->GetMomentum().mag2() + def->GetPDGMass() * def->GetPDGMass()), pre->GetMomentumDirection());
    for (int i = 0; i < dp->entries(); ++i) {
      const G4DynamicParticle* d = (*dp)[i];
      const int q = d->GetDefinition()->GetPDGEncoding();
      if (std::abs(q) == 13) items.push_back(Item{q, d->GetMomentum() / GeV, pos, t, w, 1, pdg, hf, true});
    }
    delete dp;
  } else {                                                           // missing in Geant4: Pythia's
    std::vector<PyProd> prod;
    if (!py_decay(gPyMu, pdg, pv, mass, prod)) return;
    for (const PyProd& q : prod)
      if (std::abs(q.id) == 13) items.push_back(Item{q.id, q.p, pos, t, w, 1, pdg, hf, true});
  }
  push_items(tr, items);
}
#endif

// complete / analog mode stepping
void complete_step(const G4Step* step, G4Track* tr) {
  const int pdg = tr->GetDefinition()->GetPDGEncoding();
  const int a = std::abs(pdg);
  const G4StepPoint* pre = step->GetPreStepPoint();
  if (tr->GetCurrentStepNumber() == 1) {                           // ancestry tag of the track
    int hf = 0;
    if (tr->GetParentID() == 0) {
      const G4PrimaryParticle* pp = tr->GetDynamicParticle()->GetPrimaryParticle();
      auto* pi = pp ? dynamic_cast<PrimInfo*>(pp->GetUserInformation()) : nullptr;
      if (pi) hf = pi->hf_;
    } else {
      auto it = gTags.find(tr->GetParentID());
      if (it != gTags.end()) hf = it->second.hf;
    }
    gTags[tr->GetTrackID()] = Tag{pdg, hf || is_hf(pdg)};
  }
  if (a == 13) { record_muon(step, tr, pdg); return; }
  auto mt = gMuTag.find(tr->GetDefinition());
  if (mt != gMuTag.end()) {                                        // tagged muon of a resonance decayed inside a model
    const int q = tr->GetDefinition()->GetPDGCharge() < 0 ? 13 : -13;
    std::vector<Item> items{Item{q, pre->GetMomentum() / GeV, pre->GetPosition(), pre->GetGlobalTime(), tr->GetWeight() / gResBoost,
                                 8, mt->second.parent, gTags[tr->GetTrackID()].hf, true, mt->second.ch}};
    push_items(tr, items);
    tr->SetTrackStatus(fStopAndKill);
    return;
  }
  auto sp = gSpec.find(a);
#ifdef TT_WITH_PYTHIA
  if (sp != gSpec.end() && sp->second.prompt) {                    // prompt species made by Geant4: decayed by us
    if (tr->GetCurrentStepNumber() == 1) {
      std::vector<Item> items;
      prompt_decay(pdg, pre->GetMomentum() / GeV, tr->GetDynamicParticle()->GetMass() / GeV, pre->GetPosition(),
                   pre->GetGlobalTime(), tr->GetWeight(), gTags[tr->GetTrackID()].hf, items, 0, 7);
      push_items(tr, items);
      gTaken.insert(tr->GetTrackID());
      ++gDiag.taken;
    }
    tr->SetTrackStatus(fStopAndKill);
    return;
  }
#endif
  if (pre->GetTotalEnergy() < gECut * GeV && tr->GetTotalEnergy() < gECut * GeV) {   // can no longer make a muon above it
    tr->SetTrackStatus(fStopAndKill);
    return;
  }
#ifdef TT_WITH_PYTHIA
  if (gMode == 1 && sp != gSpec.end() && br_mu(sp->second, pdg) > 0) forced_longlived(step, tr, pdg, sp->second, gTags[tr->GetTrackID()].hf);
#endif
}

void cascade_step(const G4Step* step, G4Track* tr) {
  const int pdg = tr->GetDefinition()->GetPDGEncoding();
  const int a = std::abs(pdg);
  if (gLogInt && (a == 2212 || a == 2112 || a == 211 || a == 321 || a == 130 || a == 310)) {
    const G4VProcess* pr = step->GetPostStepPoint()->GetProcessDefinedStep();
    if (pr && pr->GetProcessType() == fHadronic && pr->GetProcessName().find("Inelastic") != std::string::npos &&
        step->GetPreStepPoint()->GetKineticEnergy() > gLogEmin * GeV) {
      const G4ThreeVector pos = step->GetPostStepPoint()->GetPosition(), p = step->GetPreStepPoint()->GetMomentum();
      gInt.ev.push_back(G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID());
      gInt.pdg.push_back(pdg); gInt.w.push_back(tr->GetWeight());
      gInt.x.push_back(pos.x() / m); gInt.y.push_back(pos.y() / m); gInt.z.push_back(pos.z() / m);
      gInt.px.push_back(p.x() / GeV); gInt.py.push_back(p.y() / GeV); gInt.pz.push_back(p.z() / GeV);
    }
  }
  const G4StepPoint* pre = step->GetPreStepPoint();
  const G4StepPoint* post = step->GetPostStepPoint();
  if (a == 13) {                                                   // forced muon: record at the plane
    record_muon(step, tr, pdg);
    return;
  }
  if (tr->GetKineticEnergy() < gECut * GeV && pre->GetKineticEnergy() < gECut * GeV) {
    tr->SetTrackStatus(fStopAndKill);
    return;
  }
  if (gDimuon && tr->GetCurrentStepNumber() == 1 && gDiMu.count(a) &&             // forced dimuon decay, once per meson
      (!gVmFairShip || tr->GetParentID() == 0 || a == 223 || a == 333 || a == 113)) {
    emit_dimuon(tr, pre, a);
  }
  if (!gPiK) return;
  if (a == 321 || a == 130) forced_kmu3(step, tr, pdg);
  if (a != 211 && a != 321) return;
  if (pre->GetKineticEnergy() < gECut * GeV) return;
  const double mass = tr->GetDefinition()->GetPDGMass() / GeV;
  const double ctau = (a == 211 ? 7.8045 : 3.7118), br = (a == 211 ? 0.999877 : 0.6356);
  const G4ThreeVector pv = pre->GetMomentum() / GeV;
  const double P = pv.mag();
  const double lam_dec = P / mass * ctau;
  const double len = step->GetStepLength() / m;
  if (len <= 0) return;
  double w = tr->GetWeight() * br * len / lam_dec;
  if (gL0 > 0 && len < gL0) {                       // keep this step's muon with probability len / L0, weight / that
    const double q = len / gL0;
    if (G4UniformRand() > q) return;
    w /= q;
  }
  // two-body decay, isotropic in the rest frame
  const double mmu = 0.1056584;
  const double pstar = (mass * mass - mmu * mmu) / (2 * mass), Es = std::sqrt(pstar * pstar + mmu * mmu);
  const double cth = 2 * G4UniformRand() - 1, sth = std::sqrt(1 - cth * cth), phi = 2 * M_PI * G4UniformRand();
  const double E = std::sqrt(P * P + mass * mass), g = E / mass, bg = P / mass;
  G4ThreeVector prest(pstar * sth * std::cos(phi), pstar * sth * std::sin(phi), g * pstar * cth + bg * Es);
  prest.rotateUz(pv.unit());
  if (prest.mag() < gPMinMu) return;
  const double r = G4UniformRand();
  const G4ThreeVector pos = pre->GetPosition() + r * (post->GetPosition() - pre->GetPosition());
  const double t = pre->GetGlobalTime() + r * (post->GetGlobalTime() - pre->GetGlobalTime());
  G4ParticleDefinition* mu = pdg > 0 ? static_cast<G4ParticleDefinition*>(G4MuonPlus::Definition())
                                     : static_cast<G4ParticleDefinition*>(G4MuonMinus::Definition());
  const double ke = std::sqrt(prest.mag2() + mmu * mmu) - mmu;
  auto* dp = new G4DynamicParticle(mu, prest.unit(), ke * GeV);
  auto* nt = new G4Track(dp, t, pos);
  nt->SetWeight(w);
  nt->SetParentID(tr->GetTrackID());
  nt->SetUserInformation(new ForcedInfo(pdg, pos.z() / m, tr->GetParentID() == 0 ? 0 : 1, prest.mag()));
  G4SteppingManager* sm = G4EventManager::GetEventManager()->GetTrackingManager()->GetSteppingManager();
  sm->GetfSecondary()->push_back(nt);
}

class Stepping : public G4UserSteppingAction {
 public:
  void UserSteppingAction(const G4Step* step) override {
    G4Track* tr = step->GetTrack();
    if (gCascade) { if (gMode != 0) complete_step(step, tr); else cascade_step(step, tr); return; }
    if (tr->GetTrackID() != 1) return;
    const double z0 = step->GetPreStepPoint()->GetPosition().z() / m;
    const double z1 = step->GetPostStepPoint()->GetPosition().z() / m;
    if (!(z0 < Z_PLANE && z1 >= Z_PLANE)) return;
    const double f = (Z_PLANE - z0) / (z1 - z0);
    const G4ThreeVector a = step->GetPreStepPoint()->GetPosition(), b = step->GetPostStepPoint()->GetPosition();
    const G4ThreeVector pa = step->GetPreStepPoint()->GetMomentum(), pb = step->GetPostStepPoint()->GetMomentum();
    const G4ThreeVector x = a + f * (b - a), p = pa + f * (pb - pa);
    const size_t i = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();
    gOut.ok[i] = 1;
    gOut.x[i] = x.x() / m; gOut.y[i] = x.y() / m; gOut.z[i] = Z_PLANE;
    gOut.px[i] = p.x() / GeV; gOut.py[i] = p.y() / GeV; gOut.pz[i] = p.z() / GeV;
    tr->SetTrackStatus(fStopAndKill);
  }
};

// complete / analog mode stacking: which new tracks to follow, and the bookkeeping of the muons
G4ClassificationOfNewTrack classify_complete(const G4Track* t) {
  auto* tt = const_cast<G4Track*>(t);
  const int pdg = t->GetDefinition()->GetPDGEncoding(), a = std::abs(pdg);
  if (gCheckE && t->GetTotalEnergy() > 400 * GeV) {               // above the beam energy: who made it?
    const G4Track* par = G4EventManager::GetEventManager()->GetTrackingManager()->GetTrack();
    const G4VProcess* cp = t->GetCreatorProcess();
    std::fprintf(stderr, "TT_CHECK track pdg %d E %.3f GeV event %d parentID %d creator %s info %s; current track pdg %d E %.3f GeV id %d\n",
                 pdg, t->GetTotalEnergy() / GeV, G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID(), t->GetParentID(),
                 cp ? cp->GetProcessName().c_str() : "none(ours or primary)", t->GetUserInformation() ? "yes" : "no",
                 (par && t->GetParentID() != 0) ? par->GetDefinition()->GetPDGEncoding() : 0,
                 (par && t->GetParentID() != 0) ? par->GetTotalEnergy() / GeV : 0.0, par ? par->GetTrackID() : -1);
  }
  const G4VUserTrackInformation* ui = t->GetUserInformation();
  if (t->GetParentID() == 0) {                                     // primaries: weight and muon bookkeeping
    const G4PrimaryParticle* pp = t->GetDynamicParticle()->GetPrimaryParticle();
    auto* pi = pp ? dynamic_cast<PrimInfo*>(pp->GetUserInformation()) : nullptr;
    if (pp) tt->SetWeight(pp->GetWeight());
    if (a == 13) {
      if (t->GetMomentum().mag() < gPMinMu * GeV) return fKill;
      tt->SetUserInformation(new ForcedInfo(pi ? pi->parent_ : 0, pi ? pi->zdec_ : t->GetPosition().z() / m,
                                            pi ? pi->proc_ : 5, t->GetMomentum().mag() / GeV, pi ? pi->hf_ : 0,
                                            pi ? pi->ch_ : -1, t->GetMomentum() / GeV));
      return fUrgent;
    }
  } else if (ui == nullptr && gTaken.count(t->GetParentID())) {
    return fKill;                                                  // Geant4's own products of a species we decayed
  }
  if (a == 13) {
    if (ui != nullptr) return t->GetMomentum().mag() > gPMinMu * GeV ? fUrgent : fKill;   // ours
    if (t->GetMomentum().mag() < gPMinMu * GeV) return fKill;
    const G4VProcess* cp = t->GetCreatorProcess();
    auto it = gTags.find(t->GetParentID());
    const int par = it != gTags.end() ? it->second.pdg : 0, hf = it != gTags.end() ? it->second.hf : 0;
    int proc = 6;
    if (cp && cp->GetProcessType() == fHadronic) ++gDiag.hadronic_mu;
    if (cp && cp->GetProcessName() == "GammaToMuPair") { proc = 3; if (gMode == 1) tt->SetWeight(t->GetWeight() / gGammaFactor); }
    else if (cp && cp->GetProcessName() == "AnnihiToMuPair") { proc = 4; if (gMode == 1) tt->SetWeight(t->GetWeight() / gEpFactor); }
    else if (cp && cp->GetProcessType() == fDecay) {
      const bool forced = gSpec.count(std::abs(par)) && br_mu(gSpec.at(std::abs(par)), par) > 0;
      if (gMode == 1 && forced) return fKill;                      // made by the forced decays instead
      if (gMode == 1) { gDiag.unforced_mu_w += t->GetWeight(); gDiag.unforced_by_parent[par] += t->GetWeight(); }
      proc = 1;                                                    // analog, or a channel Pythia's tables lack: as is
    }
    tt->SetUserInformation(new ForcedInfo(par, t->GetPosition().z() / m, proc, t->GetMomentum().mag() / GeV, hf, -1,
                                          t->GetMomentum() / GeV));
    return fUrgent;
  }
  if (gMuTag.count(t->GetDefinition())) {
    ++gDiag.tags;
    if (t->GetCreatorProcess() && t->GetCreatorProcess()->GetProcessType() == fHadronic) ++gDiag.tags_hadronic;
    return t->GetMomentum().mag() > gPMinMu * GeV ? fUrgent : fKill;
  }
  G4ClassificationOfNewTrack c;
  if (a == 22) c = t->GetKineticEnergy() > gEGamma * GeV ? fUrgent : fKill;
  else if (pdg == -11) c = t->GetTotalEnergy() > E_POSITRON_MIN * GeV ? fUrgent : fKill;
  else if (a == 11 || a == 12 || a == 14 || a == 16) return fKill;
  else if (pdg == 0 || t->GetDefinition()->GetParticleType() == "nucleus") return fKill;
  else if (gSpec.count(a) && gSpec.at(a).prompt) c = fUrgent;     // decayed by us at its first step
  else c = t->GetTotalEnergy() > gECut * GeV ? fUrgent : fKill;
  if (c == fUrgent && gRR && t->GetParentID() != 0) {             // importance sampling of the cascade (primaries: generator)
    const double wt = 1.0 / rr_q(t->GetTotalEnergy() / GeV, t->GetMomentum().perp() / GeV);   // target weight
    const double w = t->GetWeight();
    if (w < wt) {                                                  // roulette up to the target; heavier tracks unchanged
      if (G4UniformRand() > w / wt) return fKill;
      tt->SetWeight(wt);
    }
  }
  return c;
}

class Stacking : public G4UserStackingAction {
 public:
  G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track* t) override {
    if (!gCascade) return t->GetParentID() == 0 ? fUrgent : fKill;
    if (gMode != 0) return classify_complete(t);
    const int a = std::abs(t->GetDefinition()->GetPDGEncoding());
    if (a == 13) {
      if (t->GetUserInformation() != nullptr) return fUrgent;                     // forced-decay muons
      const G4VProcess* cp = t->GetCreatorProcess();
      if (gDimuon && cp && cp->GetProcessName() == "GammaToMuPair") {           // boosted gamma -> mu mu
        auto* tt = const_cast<G4Track*>(t);
        tt->SetWeight(t->GetWeight() / gGammaFactor);
        tt->SetUserInformation(new ForcedInfo(22, t->GetPosition().z() / m, 2, t->GetMomentum().mag() / GeV));
        return t->GetMomentum().mag() > gPMinMu * GeV ? fUrgent : fKill;
      }
      return fKill;
    }
    if (a == 13) return fKill;
    if (a == 22) return (gDimuon && t->GetKineticEnergy() > gEGamma * GeV) ? fUrgent : fKill;
    if (a == 11 || a == 12 || a == 14 || a == 16) return fKill;
    if (t->GetDefinition()->GetPDGEncoding() == 0) return fKill;               // nuclear fragments
    if (t->GetDefinition()->GetParticleType() == "nucleus") return fKill;
    return t->GetKineticEnergy() > gECut * GeV ? fUrgent : fKill;
  }
};

G4RunManager* gRun = nullptr;

// gamma -> mu mu and e+ e- -> mu mu with cross-section factors (complete: boosted, analog: 1)
class ExtraMuPairs : public G4VPhysicsConstructor {
 public:
  ExtraMuPairs(double fg, double fe) : G4VPhysicsConstructor("ExtraMuPairs"), fg_(fg), fe_(fe) {}
  void ConstructParticle() override;
  void ConstructProcess() override {
    auto* g = new G4GammaConversionToMuons();
    g->SetCrossSecFactor(fg_);
    G4Gamma::Gamma()->GetProcessManager()->AddDiscreteProcess(g);
    auto* e = new G4AnnihiToMuPair();
    e->SetCrossSecFactor(fe_);
    G4Positron::Positron()->GetProcessManager()->AddDiscreteProcess(e);
  }
 private:
  double fg_, fe_;
};

// One muon-like tag particle per (resonance, channel, muon charge) of the replaced decay tables
void ExtraMuPairs::ConstructParticle() {
#ifdef TT_WITH_PYTHIA
  int code = 980000000;
  auto& pd = gPyAll->particleData;
  for (const auto& kv : gSpec) {
    if (!kv.second.prompt) continue;
    for (int sgn : {1, -1}) {
      if (sgn < 0 && !pd.hasAnti(kv.first)) continue;
      auto e = pd.particleDataEntryPtr(kv.first);
      for (int i = 0; i < e->sizeChannels(); ++i) {
        Pythia8::DecayChannel& c = e->channel(i);
        for (int j = 0; j < c.multiplicity(); ++j) {
          const int q = c.product(j) * sgn;
          if (std::abs(q) != 13) continue;
          auto key = std::make_tuple(sgn * kv.first, i, q);
          if (gTagDef.count(key)) continue;
          const std::string name = "mutag_" + std::to_string(sgn * kv.first) + "_" + std::to_string(i) + (q > 0 ? "_m" : "_p");
          auto* d = new G4ParticleDefinition(name, 105.6583755 * MeV, 0, q > 0 ? -eplus : eplus, 1, 0, 0, 0, 0, 0, "lepton",
                                             q > 0 ? 1 : -1, 0, code++, true, -1.0, nullptr, false, "mutag");
          gTagDef[key] = d;
          int extra = 0;
          for (int k = 0; k < c.multiplicity(); ++k) {
            const int a = std::abs(c.product(k));
            if (a != 13 && a != 12 && a != 14 && a != 16) ++extra;
          }
          gMuTag[d] = MuTag{sgn * kv.first, extra};
        }
      }
    }
  }
#endif
}

#ifdef TT_WITH_PYTHIA
// Geant4's tables of the prompt species -> Pythia's (all channels, products with Geant4 names), muons -> tag particles,
// muonic channels x gResBoost. Species with channels Geant4 cannot represent (partons, > 3 products) beyond 1 % keep theirs.
void replace_resonance_tables() {
  auto* pt = G4ParticleTable::GetParticleTable();
  auto& pd = gPyAll->particleData;
  int n_rep = 0;
  for (const auto& kv : gSpec) {
    if (!kv.second.prompt) continue;
    for (int sgn : {1, -1}) {
      G4ParticleDefinition* def = pt->FindParticle(sgn * kv.first);
      if (def == nullptr || (sgn < 0 && def == pt->FindParticle(kv.first))) continue;
      auto e = pd.particleDataEntryPtr(kv.first);
      auto* dt = new G4DecayTable();
      double tot = 0, lost = 0;
      for (int i = 0; i < e->sizeChannels(); ++i) {
        Pythia8::DecayChannel& c = e->channel(i);
        if (!(c.onMode() == 1 || c.onMode() == (sgn > 0 ? 2 : 3))) continue;
        tot += c.bRatio();
        std::vector<std::string> names;
        bool mu = false;
        // at most 3 products: Geant4 10.1's hadronic models mishandle 4-body channels of resonances they decay internally
        // (an empty 4th daughter, then a crash); the dropped channels are tiny (rho -> 4 pi 1.8e-5, eta -> pi pi e e 2.7e-4)
        for (int j = 0; j < c.multiplicity() && c.multiplicity() <= 3; ++j) {
          const int q = c.product(j) * sgn;
          G4ParticleDefinition* d = nullptr;
          const int aq = std::abs(q);
          if (!(pd.isHadron(aq) || (aq >= 11 && aq <= 16) || aq == 22)) break;   // partons (heavy-flavour tables): not here
          if (aq == 13) { d = gTagDef[std::make_tuple(sgn * kv.first, i, q)]; mu = true; }
          else { d = pt->FindParticle(q); if (d == nullptr && sgn < 0) d = pt->FindParticle(-q); }
          if (d == nullptr) { names.clear(); break; }
          names.push_back(d->GetParticleName());
        }
        if (int(names.size()) != c.multiplicity()) { lost += c.bRatio(); continue; }
        const int nd = int(names.size());
        if (std::getenv("TT_DEBUG")) {
          std::printf("table %s ch %d BR %g nd %d:", def->GetParticleName().c_str(), i, c.bRatio(), nd);
          for (auto& x : names) std::printf(" [%s]", x.c_str());
          std::printf("\n");
        }
        while (names.size() < 4) names.push_back("");
        dt->Insert(new G4PhaseSpaceDecayChannel(def->GetParticleName(), c.bRatio() * (mu ? gResBoost : 1.0), nd,
                                                names[0], names[1], names[2], names[3]));
      }
      if (tot <= 0 || lost / tot > 0.01) { delete dt; continue; }
      def->SetDecayTable(dt);
      if (std::getenv("TT_DEBUG")) {
        dt->DumpInfo();
        for (int ic = 0; ic < dt->entries(); ++ic) {               // fill every channel's daughters now (errors show here)
          G4VDecayChannel* c = dt->GetDecayChannel(ic);
          std::printf("  check %s channel %d (%d daughters)\n", def->GetParticleName().c_str(), ic, c->GetNumberOfDaughters());
          for (int j = 0; j < c->GetNumberOfDaughters(); ++j) std::printf("     %d %s\n", j, c->GetDaughter(j) ? c->GetDaughter(j)->GetParticleName().c_str() : "NULL");
        }
      }
      ++n_rep;
    }
  }
  std::printf("tailtransport: %d Geant4 decay tables replaced by Pythia's (muons tagged, muonic channels x %g)\n", n_rep, gResBoost);
}

// Pythia decayers and the muonic species table (complete / analog mode)
void init_pythia(long seed) {
  for (int k = 0; k < 2; ++k) {
    auto* py = new Pythia8::Pythia("../share/Pythia8/xmldoc", false);
    for (auto c : {"ProcessLevel:all = off", "Next:numberCount = 0", "Print:quiet = on", "Check:event = on",
                   "Random:setSeed = on"})
      py->readString(c);
    py->readString("Random:seed = " + std::to_string(seed % 900000000 + 1 + k));
    // channels missing in Pythia's tables (PDG): K_S -> pi mu nu, 4.56e-4
    py->readString("310:addChannel = 1 0.000228 0 211 13 -14");
    py->readString("310:addChannel = 1 0.000228 0 -211 -13 14");
    (k == 0 ? gPyAll : gPyMu) = py;
  }
  auto& pd = gPyAll->particleData;                                   // (nextId loops: also Pythia 8.2, e.g. 8.223 of 2018)
  for (int id = pd.nextId(1); id != 0; id = pd.nextId(id)) {
    auto e = pd.particleDataEntryPtr(id);
    if (!(e->isHadron() || id == 15)) continue;
    // onMode 1: on, 2: particle only, 3: antiparticle only (Pythia lists many B / D channels twice that way)
    double bt[2] = {0, 0}, bm[2] = {0, 0};
    for (int i = 0; i < e->sizeChannels(); ++i) {
      Pythia8::DecayChannel& c = e->channel(i);
      bool mu = false;
      for (int j = 0; j < c.multiplicity(); ++j) mu = mu || std::abs(c.product(j)) == 13;
      for (int s = 0; s < 2; ++s) {
        const bool on = c.onMode() == 1 || c.onMode() == (s == 0 ? 2 : 3);
        if (!on) continue;
        bt[s] += c.bRatio();
        if (mu) bm[s] += c.bRatio();
      }
    }
    if (bm[0] <= 0 && bm[1] <= 0) continue;
    gSpec[id] = Species{bt[0] > 0 ? bm[0] / bt[0] : 0, bt[1] > 0 ? bm[1] / bt[1] : 0, e->tau0(), e->tau0() <= 10.0};
    auto em = gPyMu->particleData.particleDataEntryPtr(id);           // muonic decayer: non-muonic channels off
    for (int i = 0; i < em->sizeChannels(); ++i) {
      Pythia8::DecayChannel& c = em->channel(i);
      bool mu = false;
      for (int j = 0; j < c.multiplicity(); ++j) mu = mu || std::abs(c.product(j)) == 13;
      if (!mu) c.onMode(0);
    }
  }
  for (auto* py : {gPyAll, gPyMu}) {
    if (!py->init()) throw std::runtime_error("Pythia decayer init failed");
    for (int id = py->particleData.nextId(1); id != 0; id = py->particleData.nextId(id))   // decayed one at a time only
      if (py->particleData.tau0(id) > 10.0 || gSpec.count(id)) py->particleData.mayDecay(id, false);
  }
}
#endif

#ifdef TT_WITH_PYTHIA
// Geant4's decay tables lack some muonic channels Pythia has (K_S -> pi mu nu, Lambda / Sigma- / Xi semileptonic): added to
// Geant4's tables (phase space, the other channels scaled down), so the analog mode has them too and the forced decays use them
void supplement_g4_tables() {
  auto* pt = G4ParticleTable::GetParticleTable();
  for (const auto& kv : gSpec) {
    if (kv.second.prompt) continue;
    for (int sgn : {1, -1}) {
      G4ParticleDefinition* def = pt->FindParticle(sgn * kv.first);
      if (def == nullptr || (sgn < 0 && def == pt->FindParticle(kv.first))) continue;
      G4DecayTable* dt = def->GetDecayTable();
      if (dt == nullptr) continue;
      bool has = false;
      for (int i = 0; i < dt->entries(); ++i)
        for (int j = 0; j < dt->GetDecayChannel(i)->GetNumberOfDaughters(); ++j)
          has = has || std::abs(dt->GetDecayChannel(i)->GetDaughter(j)->GetPDGEncoding()) == 13;
      if (has) continue;
      auto e = gPyAll->particleData.particleDataEntryPtr(kv.first);
      std::vector<std::pair<double, std::vector<std::string>>> add;
      double tot = 0, bmu = 0;
      for (int i = 0; i < e->sizeChannels(); ++i) {
        Pythia8::DecayChannel& c = e->channel(i);
        const bool on = c.onMode() == 1 || c.onMode() == (sgn > 0 ? 2 : 3);
        if (!on) continue;
        tot += c.bRatio();
        bool mu = false;
        for (int j = 0; j < c.multiplicity(); ++j) mu = mu || std::abs(c.product(j)) == 13;
        if (!mu || c.multiplicity() > 4) continue;
        std::vector<std::string> names;
        for (int j = 0; j < c.multiplicity(); ++j) {
          const int q = c.product(j) * sgn;
          G4ParticleDefinition* d = pt->FindParticle(q);
          if (d == nullptr && sgn < 0) d = pt->FindParticle(-q);           // self-conjugate product
          if (d == nullptr) { names.clear(); break; }
          names.push_back(d->GetParticleName());
        }
        if (names.empty()) continue;
        bmu += c.bRatio();
        add.emplace_back(c.bRatio(), names);
      }
      if (add.empty() || tot <= 0) continue;
      for (int i = 0; i < dt->entries(); ++i) dt->GetDecayChannel(i)->SetBR(dt->GetDecayChannel(i)->GetBR() * (1 - bmu / tot));
      for (auto& a : add) {
        const int nd = int(a.second.size());
        while (a.second.size() < 4) a.second.push_back("");
        dt->Insert(new G4PhaseSpaceDecayChannel(def->GetParticleName(), a.first / tot, nd, a.second[0], a.second[1],
                                                a.second[2], a.second[3]));
      }
      std::printf("tailtransport: added %zu muonic channel(s) (BR %.3g) to Geant4's %s\n", add.size(), bmu / tot,
                  def->GetParticleName().c_str());
    }
  }
}
#endif

void initialize(long seed, const std::string& target, const std::string& physics, bool dimuon, double gamma_factor,
                const std::string& mode, double ep_factor) {
  if (gRun) throw std::runtime_error("already initialized");
  if (mode == "complete") gMode = 1;
  else if (mode == "analog") gMode = 2;
  else if (mode != "legacy") throw std::runtime_error("mode must be legacy, complete or analog");
  if (gMode != 0 && dimuon) throw std::runtime_error("dimuon is a legacy option: complete / analog include it");
#ifndef TT_WITH_PYTHIA
  if (gMode != 0) throw std::runtime_error("complete / analog mode need the Pythia build (TT_WITH_PYTHIA)");
#endif
  std::string t = target;
  gShielded = t.size() > 3 && t.compare(t.size() - 3, 3, "+Fe") == 0;
  if (gShielded) t = t.substr(0, t.size() - 3);
  if (t == "MoW") { N_MO = 13; BLOCKS.back() = 35.; }
  else if (t == "Air") gAir = true;          // no target, no iron: decays in flight only (tests of the forced decays)
  else if (t == "SHiP2018") {        // FairShip 2018 geometry_config.py, targetOpt = 17 (the 2018 muon production)
    BLOCKS = {8., 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 5.0, 5.0, 6.5, 8., 8., 8., 8., 10., 35.};
    N_MO = 13;                            // 13 Mo (58 cm) then 4 W (61 cm), 16 water slits of 0.5 cm
  }
  else if (t == "FairShip") gFairShip = true;   // FairShip's target station (default target_config.yaml)
  else if (t != "W") throw std::runtime_error("target must be W, MoW, SHiP2018 (+Fe: in an iron shield), FairShip or Air");
  CLHEP::HepRandom::setTheSeed(seed);
  gRun = new G4RunManager;
  gRun->SetUserInitialization(new Detector);
  G4VModularPhysicsList* plist = nullptr;
  if (physics == "QGSP_BERT") plist = new QGSP_BERT(0);
  else if (physics == "FTFP_BERT") plist = new FTFP_BERT(0);
  else throw std::runtime_error("physics must be FTFP_BERT or QGSP_BERT");
  if (dimuon) {                                       // gamma -> mu mu (off in the reference lists), boosted
#ifndef TT_HAS_DIMUON
    throw std::runtime_error("dimuon (gamma -> mu mu) options need Geant4 >= 10.4");
#else
    auto* extra = dynamic_cast<G4EmExtraPhysics*>(const_cast<G4VPhysicsConstructor*>(plist->GetPhysics("G4GammaLeptoNuclearPhys")));
    if (extra == nullptr) throw std::runtime_error("G4EmExtraPhysics not found in the physics list");
    extra->GammaToMuMu(true);
    extra->GammaToMuMuFactor(gamma_factor);
    // otherwise gamma -> mu mu sits inside G4GammaGeneralProcess (11.x default): not a separate, findable process
    G4EmParameters::Instance()->SetGeneralProcessActive(false);
#endif
  }
  if (gMode != 0) {
    gGammaFactor = gMode == 1 ? gamma_factor : 1.0;
    gEpFactor = gMode == 1 ? ep_factor : 1.0;
    gResBoost = gMode == 1 ? 100.0 : 1.0;
    if (std::getenv("TT_RESBOOST")) gResBoost = std::atof(std::getenv("TT_RESBOOST"));       // tests
    if (std::getenv("TT_NO_REPLACE")) gResBoost = -1;                                         // tests: keep Geant4's tables
    plist->RegisterPhysics(new ExtraMuPairs(gGammaFactor, gEpFactor));
#ifdef TT_HAS_DIMUON
    G4EmParameters::Instance()->SetGeneralProcessActive(false);   // a separate gamma -> mu mu process
#endif
#ifdef TT_WITH_PYTHIA
    init_pythia(seed);
#endif
  }
  gRun->SetUserInitialization(plist);
  gRun->SetUserAction(new Generator);
  gRun->SetUserAction(new Stepping);
  gRun->SetUserAction(new Stacking);
  for (auto c : {"/control/verbose 0", "/run/verbose 0", "/event/verbose 0", "/tracking/verbose 0"})
    G4UImanager::GetUIpointer()->ApplyCommand(c);
  gDimuon = dimuon;
  if (gMode == 0) gGammaFactor = gamma_factor;
  gRun->Initialize();
  // K_mu3 (PDG 3.353 %) and K_L semileptonic to muons (2 x 13.52 %), phase space
  gKmu[321] = {{0.03353, {"pi0", "mu+", "nu_mu"}}};
  gKmu[-321] = {{0.03353, {"pi0", "mu-", "anti_nu_mu"}}};
  gKmu[130] = {{0.1352, {"pi+", "mu-", "anti_nu_mu"}}, {0.1352, {"pi-", "mu+", "nu_mu"}}};
  for (auto& [pdg, chans] : gKmu)
    for (auto& c : chans)
      c.ch = new G4PhaseSpaceDecayChannel(G4ParticleTable::GetParticleTable()->FindParticle(pdg)->GetParticleName(),
                                          1.0, 3, c.d[0], c.d[1], c.d[2]);
  if (dimuon) {
    auto* gm = dynamic_cast<G4GammaConversionToMuons*>(G4ProcessTable::GetProcessTable()->FindProcess("GammaToMuPair", "gamma"));
    if (gm == nullptr) throw std::runtime_error("GammaToMuPair process not found");
    gm->SetCrossSecFactor(gamma_factor);              // (also set through GammaToMuMuFactor; same value)
    // PDG branching ratios of the channels FairShip boosted (FixedTargetGenerator: eta 6,7; omega 5,7; rho0 5; eta' 6; phi 9)
    gDiMu[221] = {{5.8e-6, {"mu+", "mu-"}}, {3.1e-4, {"mu+", "mu-", "gamma"}}};
    gDiMu[223] = {{7.4e-5, {"mu+", "mu-"}}, {1.34e-4, {"pi0", "mu+", "mu-"}}};
    gDiMu[113] = {{4.55e-5, {"mu+", "mu-"}}};
    gDiMu[331] = {{1.09e-4, {"mu+", "mu-", "gamma"}}};
    gDiMu[333] = {{2.86e-4, {"mu+", "mu-"}}};
    const std::map<int, std::string> names = {{221, "eta"}, {223, "omega"}, {113, "rho0"}, {331, "eta_prime"}, {333, "phi"}};
    for (auto& [pdg, chans] : gDiMu)
      for (auto& c : chans)
        c.ch = new G4PhaseSpaceDecayChannel(names.at(pdg), 1.0, int(c.d.size()), c.d[0], c.d[1], c.d.size() > 2 ? c.d[2] : "");
  }
#ifdef TT_WITH_PYTHIA
  if (gMode != 0) { supplement_g4_tables(); if (gResBoost > 0) replace_resonance_tables(); else gResBoost = 1.0; }
#endif
  gRun->BeamOn(0);
}

template <typename T>
py::array_t<T> arr(const std::vector<T>& v) { return py::array_t<T>(v.size(), v.data()); }

py::dict transport(std::vector<int> pdg, std::vector<double> x, std::vector<double> y, std::vector<double> z,
                   std::vector<double> px, std::vector<double> py, std::vector<double> pz) {
  const size_t n = pdg.size();
  gIn = Inputs{std::move(pdg), std::move(x), std::move(y), std::move(z), std::move(px), std::move(py), std::move(pz)};
  gOut = Outputs{std::vector<int>(n, 0), std::vector<double>(n, 0), std::vector<double>(n, 0), std::vector<double>(n, 0),
                 std::vector<double>(n, 0), std::vector<double>(n, 0), std::vector<double>(n, 0)};
  {
    py::gil_scoped_release r;
    gRun->BeamOn(static_cast<int>(n));
  }
  py::dict o;
  o["ok"] = arr(gOut.ok); o["x"] = arr(gOut.x); o["y"] = arr(gOut.y); o["z"] = arr(gOut.z);
  o["px"] = arr(gOut.px); o["py"] = arr(gOut.py); o["pz"] = arr(gOut.pz);
  return o;
}

py::dict cascade(std::vector<long> offsets, std::vector<int> pdg, std::vector<double> px, std::vector<double> py,
                 std::vector<double> pz, std::vector<double> vx, std::vector<double> vy, std::vector<double> vz,
                 double e_cut, double p_min_mu, double l0, std::vector<double> e, std::vector<double> w) {
  const size_t n_ev = offsets.size() - 1;
  if (vx.size() != n_ev || vy.size() != n_ev || vz.size() != n_ev) throw std::runtime_error("one vertex per event");
  gCascade = true; gECut = e_cut; gPMinMu = p_min_mu; gL0 = l0;
  gIn = Inputs{std::move(pdg), {}, {}, {}, std::move(px), std::move(py), std::move(pz)};
  gOff = std::move(offsets); gVx = std::move(vx); gVy = std::move(vy); gVz = std::move(vz);
  gE = std::move(e);
  gW = std::move(w);
  if (!gW.empty() && gW.size() != gIn.pdg.size()) throw std::runtime_error("w: one weight per particle");
  if (!gE.empty() && gE.size() != gIn.pdg.size()) throw std::runtime_error("e: one energy per particle");
  gMu = MuOut();
  gDiag = Diag();
  gInt = IntOut();
  {
    py::gil_scoped_release r;
    gRun->BeamOn(static_cast<int>(n_ev));
  }
  gCascade = false;
  py::dict o;
  o["ev"] = arr(gMu.ev); o["w"] = arr(gMu.w); o["parent"] = arr(gMu.parent); o["gen"] = arr(gMu.gen); o["q"] = arr(gMu.q);
  o["x"] = arr(gMu.x); o["y"] = arr(gMu.y); o["z"] = arr(gMu.z);
  o["px"] = arr(gMu.px); o["py"] = arr(gMu.py); o["pz"] = arr(gMu.pz); o["z_decay"] = arr(gMu.zdec); o["p_create"] = arr(gMu.pcre);
  o["hf"] = arr(gMu.hf); o["ch"] = arr(gMu.ch);
  o["pcx"] = arr(gMu.pcx); o["pcy"] = arr(gMu.pcy); o["pcz"] = arr(gMu.pcz);
  py::dict dg;
  dg["unknown_species_w"] = gDiag.unknown_w; dg["unforced_decay_mu_w"] = gDiag.unforced_mu_w;
  dg["pythia_failures"] = gDiag.py_fail; dg["pythia_retries"] = gDiag.py_retry; dg["taken_over"] = gDiag.taken;
  dg["tags"] = gDiag.tags; dg["tags_hadronic"] = gDiag.tags_hadronic; dg["hadronic_mu"] = gDiag.hadronic_mu;
  py::dict by;
  for (const auto& kv : gDiag.unforced_by_parent) by[py::int_(kv.first)] = kv.second;
  dg["unforced_by_parent"] = by;
  o["diag"] = dg;
  py::dict it;
  it["ev"] = arr(gInt.ev); it["pdg"] = arr(gInt.pdg); it["w"] = arr(gInt.w);
  it["x"] = arr(gInt.x); it["y"] = arr(gInt.y); it["z"] = arr(gInt.z);
  it["px"] = arr(gInt.px); it["py"] = arr(gInt.py); it["pz"] = arr(gInt.pz);
  o["interactions"] = it;
  return o;
}

double inelastic_xs_per_m(int pdg, double E, const std::string& material) {
  const G4Material* mat = tt_material(material);
  auto* def = G4ParticleTable::GetParticleTable()->FindParticle(pdg);
  G4DynamicParticle dp(def, G4ThreeVector(0, 0, 1), E * GeV);
  G4ProcessVector* pv = def->GetProcessManager()->GetPostStepProcessVector(typeDoIt);
  for (size_t i = 0; i < pv->size(); ++i) {
    auto* hp = dynamic_cast<G4HadronicProcess*>((*pv)[i]);
    if (hp && hp->GetProcessName().find("Inelastic") != std::string::npos)
#if G4VERSION_NUMBER >= 1100
      return hp->GetCrossSectionDataStore()->ComputeCrossSection(&dp, mat) * m;
#else
      return hp->GetCrossSectionDataStore()->GetCrossSection(&dp, mat) * m;   // Geant4 10.x name
#endif
  }
  throw std::runtime_error("no inelastic process");
}

}  // namespace

PYBIND11_MODULE(tailtransport, mod) {
  mod.def("initialize", &initialize, py::arg("seed") = 1, py::arg("target") = "W", py::arg("physics") = "FTFP_BERT",
          py::arg("dimuon") = false, py::arg("gamma_factor") = 500.0, py::arg("mode") = "legacy", py::arg("ep_factor") = 100.0);
  mod.def("set_pik", [](bool b) { gPiK = b; }, "forced pi / K -> mu nu on (default) or off");
  mod.def("set_vm_fairship", [](bool b) { gVmFairShip = b; }, "dimuon forcing restricted as the old FairShip production");
  mod.def("set_log_interactions", [](bool b, double emin) { gLogInt = b; gLogEmin = emin; },
          "record inelastic interactions of p, n, pi, K, K_L, K_S above emin [GeV] (cascade charm)");
  mod.def("transport", &transport);
  mod.def("cascade", &cascade, py::arg("offsets"), py::arg("pdg"), py::arg("px"), py::arg("py"), py::arg("pz"),
          py::arg("vx"), py::arg("vy"), py::arg("vz"), py::arg("e_cut"), py::arg("p_min_mu"), py::arg("l0") = 0.0,
          py::arg("e") = std::vector<double>(), py::arg("w") = std::vector<double>());
  mod.def("set_muon_window", [](std::vector<double> p_edges, std::vector<double> pt_edges, std::vector<double> w_min) {
    if (w_min.size() != (p_edges.size() - 1) * (pt_edges.size() - 1)) throw std::runtime_error("w_min: one per (p, pT) cell");
    gWWp = std::move(p_edges); gWWt = std::move(pt_edges); gWWmin = std::move(w_min); gWW = true;
  }, "weight window of the recorded muons: per (p, pT) cell, w < w_min kept with probability w / w_min, weight w_min");
#ifdef TT_WITH_PYTHIA
  mod.def("decay_test", [](int id, double px, double py_, double pz, long n) {
    // the decayer's py_decay n times on one particle (after initialize in complete / analog mode): returns
    // (decays, failures, retries, products not finite or with energy beyond the parent's 0.1 %)
    if (!gPyAll) throw std::runtime_error("decay_test: initialize(mode='complete') first");
    const G4ThreeVector p(px, py_, pz);
    const double m = gPyAll->particleData.m0(id), E0 = std::sqrt(p.mag2() + m * m);
    const double f0 = gDiag.py_fail; const long r0 = gDiag.py_retry;
    long bad = 0;
    for (long k = 0; k < n; ++k) {
      std::vector<PyProd> out;
      if (!py_decay(gPyAll, id, p, m, out)) continue;
      double sum = 0; bool fin = true;
      for (auto& q : out) { const double e = std::sqrt(q.p.mag2() + q.m * q.m); fin = fin && std::isfinite(e); sum += e; }
      if (!fin || std::abs(sum - E0) > 1e-3 * E0 + 1e-6) ++bad;
    }
    return py::make_tuple(n, long(gDiag.py_fail - f0), gDiag.py_retry - r0, bad);
  }, "validation of py_decay: (decays, failures, retries, bad products returned)");
#endif
  mod.def("set_split", [](int n) { gSplit = std::max(1, n); },
          "decay splitting: each prompt decay makes n muonic decays with weight w BR_mu / n (complete mode)");
  mod.def("set_roulette", [](double qmin, double e1, double n, double pt0) {
    gRR = qmin > 0; gRq[0] = qmin; gRq[1] = e1; gRq[2] = n; gRq[3] = pt0;
  }, "importance sampling of the cascade's new tracks: q = min(1, max(qmin, (E/e1)^n) max(1, (pT/pt0)^2)), weight / q; qmin <= 0: off");
  mod.def("muonic_species", []() {
    py::dict d;
    for (const auto& kv : gSpec) d[py::int_(kv.first)] = py::make_tuple(kv.second.brmu, kv.second.brmu_bar, kv.second.tau0_mm, kv.second.prompt);
    return d;
  }, "complete / analog mode: |pdg| -> (BR to muonic channels: particle, antiparticle; Pythia tau0 [mm]; prompt)");
  mod.def("inelastic_xs_per_m", &inelastic_xs_per_m);
  mod.def("target_layout", []() { return gLayout; });
  mod.attr("Z_PLANE") = Z_PLANE;
  mod.attr("Z_TARGET") = Z_TARGET;
}
