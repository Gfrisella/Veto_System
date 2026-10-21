// Thin-target interaction library with Geant4 (FTFP_BERT): one forced muon-nuclear interaction per event.
//
// A muon starts at the centre of a block of the target material, along +z; its muonNuclear cross section is
// multiplied by a huge factor, so the first step ends in a muon-nuclear interaction (~0.01 um). The scattered
// muon and every secondary of that step are recorded, then everything is killed (no transport): the library
// holds interaction final states only, rotated / placed later by fastveto.
//
// Python API (module g4dis):
//   initialize(seed, material)   material: a Geant4 NIST name (G4_He, G4_POLYVINYL_CHLORIDE, G4_MYLAR, G4_Si,
//                                G4_POLYSTYRENE, G4_PLASTIC_SC_VINYLTOLUENE, G4_AIR, ...)
//   generate(pdg, E, n)          n events of a muon (pdg +-13) of total energy E [GeV]; returns
//       events:   mu_px, mu_py, mu_pz (scattered muon, NaN if none), nu (E - E_mu'), done (1 if interacted)
//       products: ev, pdg, px, py, pz, E (GeV, total energy)
//   cross_section(pdg, E)        unbiased muonNuclear cross section per volume [1/m] in the material

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "FTFP_BERT.hh"
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
#include "G4UImanager.hh"
#include "G4UserStackingAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "Randomize.hh"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double FORCE = 1e15;      // muonNuclear cross-section factor: mean free path ~ 1e-8 m in helium
std::string gMaterial = "G4_He";
int gPdg = 13;
double gE = 10.0;

struct Outputs {
  std::vector<double> mu_px, mu_py, mu_pz, nu;
  std::vector<int> done;
  std::vector<int> p_ev, p_pdg;
  std::vector<double> p_px, p_py, p_pz, p_E;
  void clear() { *this = Outputs(); }
} gOut;

class Detector : public G4VUserDetectorConstruction {
 public:
  G4VPhysicalVolume* Construct() override {
    G4Material* mat = G4NistManager::Instance()->FindOrBuildMaterial(gMaterial);
    if (mat == nullptr) throw std::runtime_error("unknown material " + gMaterial);
    auto* s = new G4Box("Target", 10 * m, 10 * m, 10 * m);
    auto* lv = new G4LogicalVolume(s, mat, "Target");
    return new G4PVPlacement(nullptr, G4ThreeVector(), lv, "Target", nullptr, false, 0);
  }
};

class Generator : public G4VUserPrimaryGeneratorAction {
 public:
  Generator() : gun_(std::make_unique<G4ParticleGun>(1)) {}
  void GeneratePrimaries(G4Event* ev) override {
    auto* def = G4ParticleTable::GetParticleTable()->FindParticle(gPdg);
    gun_->SetParticleDefinition(def);
    gun_->SetParticleMomentumDirection(G4ThreeVector(0, 0, 1));
    gun_->SetParticleEnergy(gE * GeV - def->GetPDGMass());
    gun_->SetParticlePosition(G4ThreeVector());
    gun_->GeneratePrimaryVertex(ev);
    gOut.mu_px.push_back(NAN); gOut.mu_py.push_back(NAN); gOut.mu_pz.push_back(NAN);
    gOut.nu.push_back(NAN); gOut.done.push_back(0);
  }

 private:
  std::unique_ptr<G4ParticleGun> gun_;
};

class Stepping : public G4UserSteppingAction {
 public:
  void UserSteppingAction(const G4Step* step) override {
    G4Track* tr = step->GetTrack();
    if (tr->GetTrackID() != 1) { tr->SetTrackStatus(fStopAndKill); return; }
    const G4VProcess* pr = step->GetPostStepPoint()->GetProcessDefinedStep();
    if (pr == nullptr || pr->GetProcessName() != "muonNuclear") return;
    const size_t ev = gOut.done.size() - 1;
    gOut.done[ev] = 1;
    double e_out = 0;
    if (tr->GetTrackStatus() == fAlive) {
      const G4ThreeVector p = step->GetPostStepPoint()->GetMomentum();
      gOut.mu_px[ev] = p.x() / GeV; gOut.mu_py[ev] = p.y() / GeV; gOut.mu_pz[ev] = p.z() / GeV;
      e_out = step->GetPostStepPoint()->GetTotalEnergy() / GeV;
    }
    gOut.nu[ev] = step->GetPreStepPoint()->GetTotalEnergy() / GeV - e_out;
    for (const G4Track* s : *step->GetSecondaryInCurrentStep()) {
      const G4ThreeVector p = s->GetMomentum();
      gOut.p_ev.push_back(static_cast<int>(ev));
      gOut.p_pdg.push_back(s->GetDefinition()->GetPDGEncoding());
      gOut.p_px.push_back(p.x() / GeV); gOut.p_py.push_back(p.y() / GeV); gOut.p_pz.push_back(p.z() / GeV);
      gOut.p_E.push_back(s->GetTotalEnergy() / GeV);
    }
    tr->SetTrackStatus(fStopAndKill);
  }
};

class Stacking : public G4UserStackingAction {
 public:
  G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track* t) override {
    return t->GetParentID() > 0 ? fKill : fUrgent;
  }
};

G4RunManager* gRun = nullptr;
std::vector<G4HadronicProcess*> gMuNuclear;

void initialize(long seed, const std::string& material) {
  if (gRun != nullptr) throw std::runtime_error("g4dis already initialized (one per process)");
  gMaterial = material;
  CLHEP::HepRandom::setTheSeed(seed);
  gRun = new G4RunManager;
  gRun->SetUserInitialization(new Detector);
  gRun->SetUserInitialization(new FTFP_BERT(0));
  gRun->SetUserAction(new Generator);
  gRun->SetUserAction(new Stepping);
  gRun->SetUserAction(new Stacking);
  auto* ui = G4UImanager::GetUIpointer();
  for (const char* c : {"/control/verbose 0", "/run/verbose 0", "/event/verbose 0", "/tracking/verbose 0"})
    ui->ApplyCommand(c);
  gRun->Initialize();
  for (int q : {13, -13}) {
    G4ProcessVector* pv = G4ParticleTable::GetParticleTable()->FindParticle(q)->GetProcessManager()->GetProcessList();
    for (size_t i = 0; i < pv->size(); ++i)
      if ((*pv)[i]->GetProcessName() == "muonNuclear") gMuNuclear.push_back(dynamic_cast<G4HadronicProcess*>((*pv)[i]));
  }
  if (gMuNuclear.size() != 2) throw std::runtime_error("muonNuclear not found for mu+ and mu-");
  gRun->BeamOn(0);
}

template <typename T>
py::array_t<T> arr(const std::vector<T>& v) { return py::array_t<T>(v.size(), v.data()); }

py::dict generate(int pdg, double E, int n) {
  if (gRun == nullptr) throw std::runtime_error("call initialize() first");
  gPdg = pdg; gE = E;
  gOut.clear();
  for (auto* p : gMuNuclear) p->MultiplyCrossSectionBy(FORCE);
  {
    py::gil_scoped_release release;
    gRun->BeamOn(n);
  }
  for (auto* p : gMuNuclear) p->MultiplyCrossSectionBy(1.0);
  py::dict ev, prod, out;
  ev["mu_px"] = arr(gOut.mu_px); ev["mu_py"] = arr(gOut.mu_py); ev["mu_pz"] = arr(gOut.mu_pz);
  ev["nu"] = arr(gOut.nu); ev["done"] = arr(gOut.done);
  prod["ev"] = arr(gOut.p_ev); prod["pdg"] = arr(gOut.p_pdg);
  prod["px"] = arr(gOut.p_px); prod["py"] = arr(gOut.p_py); prod["pz"] = arr(gOut.p_pz); prod["E"] = arr(gOut.p_E);
  out["events"] = ev; out["products"] = prod;
  return out;
}

double cross_section(int pdg, double E) {
  if (gRun == nullptr) throw std::runtime_error("call initialize() first");
  const G4Material* mat = G4NistManager::Instance()->FindOrBuildMaterial(gMaterial);
  auto* def = G4ParticleTable::GetParticleTable()->FindParticle(pdg);
  G4DynamicParticle dp(def, G4ThreeVector(0, 0, 1), E * GeV - def->GetPDGMass());
  for (auto* p : gMuNuclear)
    if (p->GetProcessManager()->GetParticleType() == def) {
      p->MultiplyCrossSectionBy(1.0);
      return p->GetCrossSectionDataStore()->ComputeCrossSection(&dp, mat) * m;
    }
  return 0.0;
}

}  // namespace

PYBIND11_MODULE(g4dis, mod) {
  mod.doc() = "Geant4 thin-target muon-nuclear final states (forced, no transport)";
  mod.def("initialize", &initialize, py::arg("seed") = 1, py::arg("material") = "G4_He");
  mod.def("generate", &generate, py::arg("pdg"), py::arg("E"), py::arg("n"));
  mod.def("cross_section", &cross_section, py::arg("pdg"), py::arg("E"));
}
