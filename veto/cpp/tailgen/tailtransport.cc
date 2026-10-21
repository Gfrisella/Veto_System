// tailtransport: Geant4 transport of muons from their production point in the ECN3 target to the plane
// where MuonsAndMatter's input sample is stored (z = -0.0045 m), and hadronic cross sections in the target.
//
// Geometry (metres; MuonsAndMatter muons_and_matter/lib/ship_muon_shield_customfield.py CreateTarget with
// z_start = -2.14, 0.5 cm water slits between blocks, diameter 30 cm; then the "IronAfterTarget" block up to
// z = 0, here a 2 x 2 m iron slab). World: air. Two targets (initialize(target=...)):
//   "W"   MuonsAndMatter today: N = 0 molybdenum, T = 18 tungsten blocks (last 50 cm)
//   "MoW" the original ECN3 design in the same function's comments (N = 13, T = 5, last block 35 cm):
//         13 molybdenum (TZM, here G4_Mo) blocks then 5 tungsten.
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
#include "G4Tubs.hh"
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
#include "G4EmParameters.hh"
#include "G4VModularPhysicsList.hh"

#include <memory>
#include <tuple>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double Z_TARGET = -2.14, Z_PLANE = -0.0045, Z_IRON_END = 0.0;
constexpr double TARGET_RADIUS = 0.15, WATER = 0.005;
std::vector<double> BLOCKS = {8., 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 5.0, 5.0,
                              6.5, 8., 8., 5., 8., 10., 20., 50.};          // cm
int N_MO = 0;                                                               // first N_MO blocks molybdenum
std::vector<std::tuple<double, double, std::string>> gLayout;            // m

class Detector : public G4VUserDetectorConstruction {
 public:
  G4VPhysicalVolume* Construct() override {
    auto* nist = G4NistManager::Instance();
    auto* air = nist->FindOrBuildMaterial("G4_AIR");
    auto* w = nist->FindOrBuildMaterial("G4_W");
    auto* water = nist->FindOrBuildMaterial("G4_WATER");
    auto* fe = nist->FindOrBuildMaterial("G4_Fe");
    auto* mo = nist->FindOrBuildMaterial("G4_Mo");
    auto* worldLV = new G4LogicalVolume(new G4Box("World", 3 * m, 3 * m, 3 * m), air, "World");
    auto* worldPV = new G4PVPlacement(nullptr, {}, worldLV, "World", nullptr, false, 0);
    gLayout.clear();
    double z = Z_TARGET;
    auto place_tub = [&](double z0, double len, G4Material* mat, const std::string& name) {
      auto* s = new G4Tubs(name, 0, TARGET_RADIUS * m, 0.5 * len * m, 0, 360 * deg);
      new G4PVPlacement(nullptr, G4ThreeVector(0, 0, (z0 + 0.5 * len) * m), new G4LogicalVolume(s, mat, name), name,
                        worldLV, false, 0);
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
struct MuOut { std::vector<int> ev, parent, gen, q; std::vector<double> w, x, y, z, px, py, pz, zdec, pcre; } gMu;
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
  ForcedInfo(int parent, double zdec, int gen, double pc) : parent_(parent), zdec_(zdec), gen_(gen), pc_(pc) {}
  int parent_; double zdec_; int gen_; double pc_;     // pc_: muon momentum at creation [GeV]
};
struct Outputs { std::vector<int> ok; std::vector<double> x, y, z, px, py, pz; } gOut;

class Generator : public G4VUserPrimaryGeneratorAction {
 public:
  Generator() : gun_(std::make_unique<G4ParticleGun>(1)) {}
  void GeneratePrimaries(G4Event* ev) override {
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
  const int ev = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();
  if (a == 13) {                                                   // forced muon: record at the plane
    const double z0 = pre->GetPosition().z() / m, z1 = post->GetPosition().z() / m;
    if (z0 < Z_PLANE && z1 >= Z_PLANE) {
      const double f = (Z_PLANE - z0) / (z1 - z0);
      const G4ThreeVector x = pre->GetPosition() + f * (post->GetPosition() - pre->GetPosition());
      const G4ThreeVector p = pre->GetMomentum() + f * (post->GetMomentum() - pre->GetMomentum());
      auto* info = static_cast<ForcedInfo*>(tr->GetUserInformation());
      gMu.ev.push_back(ev); gMu.w.push_back(tr->GetWeight());
      gMu.x.push_back(x.x() / m); gMu.y.push_back(x.y() / m); gMu.z.push_back(Z_PLANE);
      gMu.px.push_back(p.x() / GeV); gMu.py.push_back(p.y() / GeV); gMu.pz.push_back(p.z() / GeV);
      gMu.parent.push_back(info ? info->parent_ : 0); gMu.zdec.push_back(info ? info->zdec_ : 0);
      gMu.gen.push_back(info ? info->gen_ : -1);
      gMu.pcre.push_back(info ? info->pc_ : 0);
      gMu.q.push_back(pdg);                                        // the muon's own pdg (13 mu-, -13 mu+)
      tr->SetTrackStatus(fStopAndKill);
    }
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
    if (gCascade) { cascade_step(step, tr); return; }
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

class Stacking : public G4UserStackingAction {
 public:
  G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track* t) override {
    if (!gCascade) return t->GetParentID() == 0 ? fUrgent : fKill;
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

void initialize(long seed, const std::string& target, const std::string& physics, bool dimuon, double gamma_factor) {
  if (gRun) throw std::runtime_error("already initialized");
  if (target == "MoW") { N_MO = 13; BLOCKS.back() = 35.; }
  else if (target != "W") throw std::runtime_error("target must be W or MoW");
  CLHEP::HepRandom::setTheSeed(seed);
  gRun = new G4RunManager;
  gRun->SetUserInitialization(new Detector);
  G4VModularPhysicsList* plist = nullptr;
  if (physics == "QGSP_BERT") plist = new QGSP_BERT(0);
  else if (physics == "FTFP_BERT") plist = new FTFP_BERT(0);
  else throw std::runtime_error("physics must be FTFP_BERT or QGSP_BERT");
  if (dimuon) {                                       // gamma -> mu mu (off in the reference lists), boosted
    auto* extra = dynamic_cast<G4EmExtraPhysics*>(const_cast<G4VPhysicsConstructor*>(plist->GetPhysics("G4GammaLeptoNuclearPhys")));
    if (extra == nullptr) throw std::runtime_error("G4EmExtraPhysics not found in the physics list");
    extra->GammaToMuMu(true);
    extra->GammaToMuMuFactor(gamma_factor);
    // otherwise gamma -> mu mu sits inside G4GammaGeneralProcess (11.x default): not a separate, findable process
    G4EmParameters::Instance()->SetGeneralProcessActive(false);
  }
  gRun->SetUserInitialization(plist);
  gRun->SetUserAction(new Generator);
  gRun->SetUserAction(new Stepping);
  gRun->SetUserAction(new Stacking);
  for (auto c : {"/control/verbose 0", "/run/verbose 0", "/event/verbose 0", "/tracking/verbose 0"})
    G4UImanager::GetUIpointer()->ApplyCommand(c);
  gDimuon = dimuon;
  gGammaFactor = gamma_factor;
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
                 double e_cut, double p_min_mu, double l0) {
  const size_t n_ev = offsets.size() - 1;
  if (vx.size() != n_ev || vy.size() != n_ev || vz.size() != n_ev) throw std::runtime_error("one vertex per event");
  gCascade = true; gECut = e_cut; gPMinMu = p_min_mu; gL0 = l0;
  gIn = Inputs{std::move(pdg), {}, {}, {}, std::move(px), std::move(py), std::move(pz)};
  gOff = std::move(offsets); gVx = std::move(vx); gVy = std::move(vy); gVz = std::move(vz);
  gMu = MuOut();
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
  py::dict it;
  it["ev"] = arr(gInt.ev); it["pdg"] = arr(gInt.pdg); it["w"] = arr(gInt.w);
  it["x"] = arr(gInt.x); it["y"] = arr(gInt.y); it["z"] = arr(gInt.z);
  it["px"] = arr(gInt.px); it["py"] = arr(gInt.py); it["pz"] = arr(gInt.pz);
  o["interactions"] = it;
  return o;
}

double inelastic_xs_per_m(int pdg, double E, const std::string& material) {
  const std::map<std::string, std::string> names = {{"W", "G4_W"}, {"Mo", "G4_Mo"}, {"Fe", "G4_Fe"}, {"water", "G4_WATER"}};
  const G4Material* mat = G4NistManager::Instance()->FindOrBuildMaterial(names.at(material));
  auto* def = G4ParticleTable::GetParticleTable()->FindParticle(pdg);
  G4DynamicParticle dp(def, G4ThreeVector(0, 0, 1), E * GeV);
  G4ProcessVector* pv = def->GetProcessManager()->GetPostStepProcessVector(typeDoIt);
  for (size_t i = 0; i < pv->size(); ++i) {
    auto* hp = dynamic_cast<G4HadronicProcess*>((*pv)[i]);
    if (hp && hp->GetProcessName().find("Inelastic") != std::string::npos)
      return hp->GetCrossSectionDataStore()->ComputeCrossSection(&dp, mat) * m;
  }
  throw std::runtime_error("no inelastic process");
}

}  // namespace

PYBIND11_MODULE(tailtransport, mod) {
  mod.def("initialize", &initialize, py::arg("seed") = 1, py::arg("target") = "W", py::arg("physics") = "FTFP_BERT",
          py::arg("dimuon") = false, py::arg("gamma_factor") = 500.0);
  mod.def("set_pik", [](bool b) { gPiK = b; }, "forced pi / K -> mu nu on (default) or off");
  mod.def("set_vm_fairship", [](bool b) { gVmFairShip = b; }, "dimuon forcing restricted as the old FairShip production");
  mod.def("set_log_interactions", [](bool b, double emin) { gLogInt = b; gLogEmin = emin; },
          "record inelastic interactions of p, n, pi, K, K_L, K_S above emin [GeV] (cascade charm)");
  mod.def("transport", &transport);
  mod.def("cascade", &cascade, py::arg("offsets"), py::arg("pdg"), py::arg("px"), py::arg("py"), py::arg("pz"),
          py::arg("vx"), py::arg("vy"), py::arg("vz"), py::arg("e_cut"), py::arg("p_min_mu"), py::arg("l0") = 0.0);
  mod.def("inelastic_xs_per_m", &inelastic_xs_per_m);
  mod.def("target_layout", []() { return gLayout; });
  mod.attr("Z_PLANE") = Z_PLANE;
  mod.attr("Z_TARGET") = Z_TARGET;
}
