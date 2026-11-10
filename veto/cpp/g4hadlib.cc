// Thin-target hadron-nucleus library with Geant4 (FTFP_BERT): one forced inelastic interaction per event
// (bin/neutral_library.py: neutrons, anti-neutrons and K_L interacting in the helium of the decay vessel, ISSUES D4).
//
// The hadron starts at the centre of a block of the target material, along +z; the cross section of its inelastic process
// (the one whose name ends in "Inelastic") is multiplied by a huge factor, so the first step ends in that interaction.
// Every secondary of that step is recorded (and the primary if it survives), then everything is killed (no transport).
//
// Python API (module g4hadlib):
//   initialize(seed, material)   material: a Geant4 NIST name (G4_He, ...)
//   generate(pdg, E, n)          n events of a hadron of total energy E [GeV]; returns
//       events:   done (1 if interacted)
//       products: ev, pdg, px, py, pz, E (GeV, total energy)
//   cross_section(pdg, E)        unbiased inelastic cross section per volume [1/m] in the material

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

constexpr double FORCE = 1e12;      // inelastic cross-section factor: mean free path ~ 1e-9 m in helium
std::string gMaterial = "G4_He";
int gPdg = 13;
double gE = 10.0;

struct Outputs {
  std::vector<int> done;
  std::vector<int> p_ev, p_pdg;
  std::vector<double> p_px, p_py, p_pz, p_E;
  void clear() { *this = Outputs(); }
} gOut;
G4HadronicProcess* gForced = nullptr;   // the inelastic process being forced

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
    gOut.done.push_back(0);
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
    if (pr == nullptr || pr != gForced) { tr->SetTrackStatus(fStopAndKill); return; }
    const size_t ev = gOut.done.size() - 1;
    gOut.done[ev] = 1;
    if (tr->GetTrackStatus() == fAlive) {             // the primary survives (rare for inelastic): record it as a product
      const G4ThreeVector p = step->GetPostStepPoint()->GetMomentum();
      gOut.p_ev.push_back(static_cast<int>(ev)); gOut.p_pdg.push_back(tr->GetDefinition()->GetPDGEncoding());
      gOut.p_px.push_back(p.x() / GeV); gOut.p_py.push_back(p.y() / GeV); gOut.p_pz.push_back(p.z() / GeV);
      gOut.p_E.push_back(step->GetPostStepPoint()->GetTotalEnergy() / GeV);
    }
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

G4HadronicProcess* inelastic(int pdg) {
  auto* def = G4ParticleTable::GetParticleTable()->FindParticle(pdg);
  if (def == nullptr) throw std::runtime_error("unknown pdg " + std::to_string(pdg));
  G4ProcessVector* pv = def->GetProcessManager()->GetProcessList();
  for (size_t i = 0; i < pv->size(); ++i) {
    const G4String& nm = (*pv)[i]->GetProcessName();
    if (nm.size() > 9 && nm.substr(nm.size() - 9) == "Inelastic") return dynamic_cast<G4HadronicProcess*>((*pv)[i]);
  }
  throw std::runtime_error("no inelastic process for pdg " + std::to_string(pdg));
}

void initialize(long seed, const std::string& material) {
  if (gRun != nullptr) throw std::runtime_error("g4hadlib already initialized (one per process)");
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
  gRun->BeamOn(0);
}

template <typename T>
py::array_t<T> arr(const std::vector<T>& v) { return py::array_t<T>(v.size(), v.data()); }

py::dict generate(int pdg, double E, int n) {
  if (gRun == nullptr) throw std::runtime_error("call initialize() first");
  gPdg = pdg; gE = E;
  gOut.clear();
  gForced = inelastic(pdg);
  gForced->MultiplyCrossSectionBy(FORCE);
  {
    py::gil_scoped_release release;
    gRun->BeamOn(n);
  }
  gForced->MultiplyCrossSectionBy(1.0);
  py::dict ev, prod, out;
  ev["done"] = arr(gOut.done);
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
  G4HadronicProcess* p = inelastic(pdg);
  p->MultiplyCrossSectionBy(1.0);
  return p->GetCrossSectionDataStore()->ComputeCrossSection(&dp, mat) * m;
}

}  // namespace

PYBIND11_MODULE(g4hadlib, mod) {
  mod.doc() = "Geant4 thin-target hadron inelastic final states (forced, no transport)";
  mod.def("initialize", &initialize, py::arg("seed") = 1, py::arg("material") = "G4_He");
  mod.def("generate", &generate, py::arg("pdg"), py::arg("E"), py::arg("n"));
  mod.def("cross_section", &cross_section, py::arg("pdg"), py::arg("E"));
}
