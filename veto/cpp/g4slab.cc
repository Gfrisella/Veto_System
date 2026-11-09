// What comes out behind a slab traversed by a muon: EM debris (delta rays, bremsstrahlung, pairs) and the rest,
// full Geant4 (FTFP_BERT_EMZ: EM option 4), standalone (bin/em_debris_study.py).
//
// Geometry: air world; layers stacked along +z from z = 0 (each a Geant4 NIST material and a thickness); muons
// start 1 cm before z = 0 along +z (optionally tilted by theta about y). Scoring planes at z_end + d for each d
// of `planes`: every crossing (in +z) of any particle is recorded, the muon included (track 1). Tracks are killed
// beyond the last plane.
//
// Python API (module g4slab):
//   initialize(seed, layers, planes, cut_mm, field_tesla)   layers: [(material, thickness_m)], planes: [d_m], cut: range
//                                              cut, field: uniform B_y in the layers (a magnetised shield, 0 = none)
//   shoot(pdg, E, n, theta)                    n muons of total energy E [GeV]; returns a dict of arrays:
//       ev, plane, pdg, track (1 = the muon), parent, ekin (GeV), x, y (m, at the plane), px, py, pz (GeV), t (ns)

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "G4Box.hh"
#include "G4Event.hh"
#include "G4LogicalVolume.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4PhysListFactory.hh"
#include "G4RunManager.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4UImanager.hh"
#include "G4ChordFinder.hh"
#include "G4FieldManager.hh"
#include "G4UniformMagField.hh"
#include "G4UserEventAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "Randomize.hh"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

std::vector<std::pair<std::string, double>> gLayers;   // material, thickness [m]
std::vector<double> gPlanes;                           // distances behind the last layer [m]
double gFieldT = 0.0;                                  // uniform B_y [T] inside the layers (a magnetised shield), 0 = none
double gZend = 0.0;                                    // m
int gEvent = 0, gPdg = 13;
double gE = 10.0, gTheta = 0.0;

struct Out {
  std::vector<int> ev, plane, pdg, track, parent;
  std::vector<double> ekin, x, y, px, py, pz, t;
} gOut;

class Detector : public G4VUserDetectorConstruction {
 public:
  G4VPhysicalVolume* Construct() override {
    auto* nist = G4NistManager::Instance();
    double total = 0;
    for (auto& l : gLayers) total += l.second;
    gZend = total;
    double zmax = total + (gPlanes.empty() ? 0.0 : gPlanes.back()) + 1.0;
    double half = std::max(zmax, 1.0) + 1.0;
    auto* world = new G4LogicalVolume(new G4Box("world", 20 * m, 20 * m, half * m), nist->FindOrBuildMaterial("G4_AIR"), "world");
    auto* pv = new G4PVPlacement(nullptr, {}, world, "world", nullptr, false, 0);
    double z = 0;
    int i = 0;
    for (auto& l : gLayers) {
      auto* lv = new G4LogicalVolume(new G4Box("l", 19 * m, 19 * m, l.second / 2 * m),
                                     nist->FindOrBuildMaterial(l.first), "layer" + std::to_string(i));
      new G4PVPlacement(nullptr, G4ThreeVector(0, 0, (z + l.second / 2) * m), lv, "layer" + std::to_string(i), world, false, i);
      if (gFieldT != 0.0) {
        auto* field = new G4UniformMagField(G4ThreeVector(0., gFieldT * tesla, 0.));
        auto* fm = new G4FieldManager(field);
        fm->CreateChordFinder(field);
        lv->SetFieldManager(fm, true);
      }
      z += l.second;
      ++i;
    }
    return pv;
  }
};

class Gun : public G4VUserPrimaryGeneratorAction {
 public:
  Gun() : gun_(1) {}
  void GeneratePrimaries(G4Event* e) override {
    gun_.SetParticleDefinition(G4ParticleTable::GetParticleTable()->FindParticle(gPdg));
    double m_mu = gun_.GetParticleDefinition()->GetPDGMass();
    gun_.SetParticleEnergy(gE * GeV - m_mu);
    gun_.SetParticlePosition(G4ThreeVector(0, 0, -1 * cm));
    gun_.SetParticleMomentumDirection(G4ThreeVector(std::sin(gTheta), 0, std::cos(gTheta)));
    gun_.GeneratePrimaryVertex(e);
  }
 private:
  G4ParticleGun gun_;
};

class Stepping : public G4UserSteppingAction {
 public:
  void UserSteppingAction(const G4Step* s) override {
    double z0 = s->GetPreStepPoint()->GetPosition().z() / m, z1 = s->GetPostStepPoint()->GetPosition().z() / m;
    auto* tr = s->GetTrack();
    for (size_t k = 0; k < gPlanes.size(); ++k) {
      double zp = gZend + gPlanes[k] + 1e-9;
      if (z0 < zp && z1 >= zp) {
        auto p0 = s->GetPreStepPoint()->GetPosition(), p1 = s->GetPostStepPoint()->GetPosition();
        double f = (zp - z0) / (z1 - z0);
        auto mom = s->GetPostStepPoint()->GetMomentum();
        gOut.ev.push_back(gEvent);
        gOut.plane.push_back((int)k);
        gOut.pdg.push_back(tr->GetDefinition()->GetPDGEncoding());
        gOut.track.push_back(tr->GetTrackID());
        gOut.parent.push_back(tr->GetParentID());
        gOut.ekin.push_back(s->GetPostStepPoint()->GetKineticEnergy() / GeV);
        gOut.x.push_back((p0.x() + f * (p1.x() - p0.x())) / m);
        gOut.y.push_back((p0.y() + f * (p1.y() - p0.y())) / m);
        gOut.px.push_back(mom.x() / GeV);
        gOut.py.push_back(mom.y() / GeV);
        gOut.pz.push_back(mom.z() / GeV);
        gOut.t.push_back(s->GetPostStepPoint()->GetGlobalTime() / ns);
      }
    }
    if (!gPlanes.empty() && z1 > gZend + gPlanes.back() + 0.01) tr->SetTrackStatus(fStopAndKill);
  }
};

class EventAction : public G4UserEventAction {
 public:
  void BeginOfEventAction(const G4Event* e) override { gEvent = e->GetEventID(); }
};

G4RunManager* gRun = nullptr;

void initialize(long seed, std::vector<std::pair<std::string, double>> layers, std::vector<double> planes, double cut_mm,
                double field_tesla) {
  gFieldT = field_tesla;
  if (gRun) throw std::runtime_error("g4slab: initialize once per process");
  gLayers = std::move(layers);
  gPlanes = std::move(planes);
  CLHEP::HepRandom::setTheSeed(seed);
  gRun = new G4RunManager;
  gRun->SetUserInitialization(new Detector);
  G4PhysListFactory factory;
  auto* phys = factory.GetReferencePhysList("FTFP_BERT_EMZ");
  phys->SetDefaultCutValue(cut_mm * mm);
  gRun->SetUserInitialization(phys);
  gRun->SetUserAction(new Gun);
  gRun->SetUserAction(new Stepping);
  gRun->SetUserAction(new EventAction);
  G4UImanager::GetUIpointer()->ApplyCommand("/control/verbose 0");
  G4UImanager::GetUIpointer()->ApplyCommand("/run/verbose 0");
  G4UImanager::GetUIpointer()->ApplyCommand("/event/verbose 0");
  G4UImanager::GetUIpointer()->ApplyCommand("/tracking/verbose 0");
  gRun->Initialize();
}

template <class T>
py::array_t<T> arr(const std::vector<T>& v) { return py::array_t<T>(v.size(), v.data()); }

py::dict shoot(int pdg, double E, int n, double theta) {
  gPdg = pdg;
  gE = E;
  gTheta = theta;
  gOut = Out();
  gRun->BeamOn(n);
  py::dict d;
  d["ev"] = arr(gOut.ev); d["plane"] = arr(gOut.plane); d["pdg"] = arr(gOut.pdg); d["track"] = arr(gOut.track);
  d["parent"] = arr(gOut.parent); d["ekin"] = arr(gOut.ekin); d["x"] = arr(gOut.x); d["y"] = arr(gOut.y);
  d["px"] = arr(gOut.px); d["py"] = arr(gOut.py); d["pz"] = arr(gOut.pz); d["t"] = arr(gOut.t);
  return d;
}

}  // namespace

PYBIND11_MODULE(g4slab, m) {
  m.doc() = "Muon through slabs: everything crossing planes behind them (full Geant4, FTFP_BERT_EMZ)";
  m.def("initialize", &initialize, py::arg("seed"), py::arg("layers"), py::arg("planes"), py::arg("cut_mm") = 0.7,
        py::arg("field_tesla") = 0.0);
  m.def("shoot", &shoot, py::arg("pdg"), py::arg("E"), py::arg("n"), py::arg("theta") = 0.0);
}
