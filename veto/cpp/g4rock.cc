// Muon-nuclear interactions in the cavern rock and what comes back out of its surface (cavern-wall DIS,
// docs/full_sim_inventory.md, ISSUES D7).
//
// Geometry (metres): a half-space of rock (Geant4 NIST material, default G4_CONCRETE as the transport's cavern),
// surface = the plane y = 0, rock y < 0 (|x|, |z| < half_xz, depth to rock_depth), cavern air y > 0 (to air_height).
// Event: a muon starts at the vertex (0, -depth, 0) moving away from the surface at grazing angle alpha:
// direction (0, -sin alpha, cos alpha). Its muonNuclear cross section is multiplied by FORCE at the start of the event,
// so its first step (~1e-8 m) is a muon-nuclear interaction; right after it the factor is set back to 1, and the
// scattered muon and every product are transported with full physics (FTFP_BERT; showers, re-interactions, decays).
// Recording: every particle crossing the surface from the rock into the air is recorded and killed there (track 1 =
// the scattered muon). Killed when created: neutrinos, nuclei (Z > 2), anything below e_cut [GeV kinetic].
// Weight (the caller's): one event = one interaction at that depth; rate = sum over muon path pieces of n sigma ds.
//
// Python API (module g4rock):
//   initialize(seed, material, e_cut, half_xz, rock_depth, air_height)
//   shoot(pdg, E, n, depth, alpha) -> dict(events=..., exits=...)
//       events: done (1 = the forced interaction happened), nu (E - E_mu' in GeV)
//       exits:  ev, pdg, track, parent, ekin (GeV), x, z (m, on the surface), px, py, pz (GeV), t (ns)
//   shoot_depths(pdg, E, alpha, depths)  as shoot, one event per entry of depths (each its own vertex depth)
//   inject(offsets, pdg, px, py, pz, x, y, z) -> dict(events=..., exits=...)
//       no forcing: event i = the given particles offsets[i] .. offsets[i+1]-1 (GeV, metres, rock frame), transported with
//       full physics (e.g. Pythia final states placed at a vertex in the rock). Put the scattered muon first: it is track 1.
//   cross_section(pdg, E)  unbiased muonNuclear cross section per volume of the rock [1/m]
//   set_record(on, em_min, thin, max_points)  record trajectories in later shoot() calls (for event displays): every
//       track (hadrons, muons: above e_cut; e+- / gamma: above em_min GeV kinetic at creation), a point whenever it has
//       moved `thin` metres since the last stored point, at boundaries and at its end; at most max_points per track.
//       shoot() then also returns tracks: ev, track, parent, pdg, ekin0, first, n and points: x, y, z, t

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
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
#include "G4PrimaryParticle.hh"
#include "G4PrimaryVertex.hh"
#include "G4ParticleTable.hh"
#include "G4ProcessManager.hh"
#include "G4RunManager.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4UImanager.hh"
#include "G4UserEventAction.hh"
#include "G4UserStackingAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "Randomize.hh"

#include <cmath>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double FORCE = 1e15;
std::string gMaterial = "G4_CONCRETE";
double gEcut = 1e-3, gHalfXZ = 200.0, gRockDepth = 30.0, gAirHeight = 30.0;
int gPdg = 13;
double gE = 10.0, gDepth = 0.5, gAlpha = 0.1;
G4VPhysicalVolume* gRock = nullptr;
bool gInject = false;                              // inject mode: given primaries, no forcing
std::vector<double> gDepths;                       // forced mode: per-event depth (empty: gDepth for all)
std::vector<long> gOff;
std::vector<int> gIpdg;
std::vector<double> gIpx, gIpy, gIpz, gIx, gIy, gIz;
bool gRecord = false;
double gEmMin = 0.05, gThin = 0.05;
int gMaxPts = 300;
std::vector<G4HadronicProcess*> gMuNuclear;

struct Outputs {
  std::vector<int> done;
  std::vector<double> nu;
  std::vector<int> x_ev, x_pdg, x_track, x_parent;
  std::vector<double> x_ekin, x_x, x_z, x_px, x_py, x_pz, x_t;
  std::vector<int> t_ev, t_track, t_parent, t_pdg, t_first, t_n;
  std::vector<double> t_ekin0;
  std::vector<float> q_x, q_y, q_z, q_t;
  void clear() { *this = Outputs(); }
} gOut;

void force(double f) { for (auto* p : gMuNuclear) p->MultiplyCrossSectionBy(f); }

std::unordered_map<int, int> gTrk;                 // track ID -> index into t_* of this event (-1: not recorded)

void add_point(int k, const G4ThreeVector& x, double t) {
  gOut.q_x.push_back(static_cast<float>(x.x() / m)); gOut.q_y.push_back(static_cast<float>(x.y() / m));
  gOut.q_z.push_back(static_cast<float>(x.z() / m)); gOut.q_t.push_back(static_cast<float>(t / ns));
  gOut.t_n[k] += 1;
}

void record_step(const G4Step* step, size_t ev) {
  const G4Track* tr = step->GetTrack();
  auto it = gTrk.find(tr->GetTrackID());
  int k;
  if (it == gTrk.end()) {
    const int a = std::abs(tr->GetDefinition()->GetPDGEncoding());
    const double ek = step->GetPreStepPoint()->GetKineticEnergy() / GeV;
    if ((a == 11 || a == 22) && ek < gEmMin) { gTrk[tr->GetTrackID()] = -1; return; }
    k = static_cast<int>(gOut.t_ev.size());
    gTrk[tr->GetTrackID()] = k;
    gOut.t_ev.push_back(static_cast<int>(ev)); gOut.t_track.push_back(tr->GetTrackID()); gOut.t_parent.push_back(tr->GetParentID());
    gOut.t_pdg.push_back(tr->GetDefinition()->GetPDGEncoding()); gOut.t_ekin0.push_back(ek);
    gOut.t_first.push_back(static_cast<int>(gOut.q_x.size())); gOut.t_n.push_back(0);
    add_point(k, step->GetPreStepPoint()->GetPosition(), step->GetPreStepPoint()->GetGlobalTime());
  } else {
    k = it->second;
    if (k < 0) return;
  }
  if (gOut.t_n[k] >= gMaxPts) return;
  const G4StepPoint* post = step->GetPostStepPoint();
  const size_t last = gOut.q_x.size() - 1;
  const G4ThreeVector x = post->GetPosition();
  const double d = std::hypot(std::hypot(x.x() / m - gOut.q_x[last], x.y() / m - gOut.q_y[last]), x.z() / m - gOut.q_z[last]);
  const bool end = tr->GetTrackStatus() != fAlive || post->GetStepStatus() == fGeomBoundary;
  if (d >= gThin || end) add_point(k, x, post->GetGlobalTime());
}

class Detector : public G4VUserDetectorConstruction {
 public:
  G4VPhysicalVolume* Construct() override {
    auto* nist = G4NistManager::Instance();
    G4Material* rock = nist->FindOrBuildMaterial(gMaterial);
    if (rock == nullptr) throw std::runtime_error("unknown material " + gMaterial);
    G4Material* air = nist->FindOrBuildMaterial("G4_AIR");
    const double hy = 0.5 * (gRockDepth + gAirHeight);
    auto* lw = new G4LogicalVolume(new G4Box("World", gHalfXZ * m, hy * m, gHalfXZ * m), air, "World");
    auto* pw = new G4PVPlacement(nullptr, G4ThreeVector(0, (0.5 * (gAirHeight - gRockDepth)) * m, 0), lw, "World", nullptr, false, 0);
    auto* lr = new G4LogicalVolume(new G4Box("Rock", gHalfXZ * m, 0.5 * gRockDepth * m, gHalfXZ * m), rock, "Rock");
    // the rock's centre in the world's frame: world centre at y_w = (air - rock) / 2, rock centre at -rock / 2
    gRock = new G4PVPlacement(nullptr, G4ThreeVector(0, (-0.5 * gRockDepth - 0.5 * (gAirHeight - gRockDepth)) * m, 0), lr, "Rock", lw, false, 0);
    return pw;
  }
};

class Generator : public G4VUserPrimaryGeneratorAction {
 public:
  Generator() : gun_(std::make_unique<G4ParticleGun>(1)) {}
  void GeneratePrimaries(G4Event* ev) override {
    if (gInject) {
      const int id = ev->GetEventID();
      for (long k = gOff[id]; k < gOff[id + 1]; ++k) {
        auto* d = G4ParticleTable::GetParticleTable()->FindParticle(gIpdg[k]);
        if (d == nullptr) continue;
        auto* v = new G4PrimaryVertex(G4ThreeVector(gIx[k] * m, gIy[k] * m, gIz[k] * m), 0.0);
        v->SetPrimary(new G4PrimaryParticle(d, gIpx[k] * GeV, gIpy[k] * GeV, gIpz[k] * GeV));
        ev->AddPrimaryVertex(v);
      }
      gOut.done.push_back(1); gOut.nu.push_back(NAN);
      return;
    }
    const double depth = gDepths.empty() ? gDepth : gDepths[ev->GetEventID()];
    auto* def = G4ParticleTable::GetParticleTable()->FindParticle(gPdg);
    gun_->SetParticleDefinition(def);
    gun_->SetParticleMomentumDirection(G4ThreeVector(0, -std::sin(gAlpha), std::cos(gAlpha)));
    gun_->SetParticleEnergy(gE * GeV - def->GetPDGMass());
    gun_->SetParticlePosition(G4ThreeVector(0, -depth * m, 0));
    gun_->GeneratePrimaryVertex(ev);
    gOut.done.push_back(0); gOut.nu.push_back(NAN);
  }

 private:
  std::unique_ptr<G4ParticleGun> gun_;
};

class Events : public G4UserEventAction {
 public:
  void BeginOfEventAction(const G4Event*) override { if (!gInject) force(FORCE); gTrk.clear(); }
  void EndOfEventAction(const G4Event*) override { force(1.0); }
};

class Stepping : public G4UserSteppingAction {
 public:
  void UserSteppingAction(const G4Step* step) override {
    G4Track* tr = step->GetTrack();
    const size_t ev = gOut.done.size() - 1;
    if (!gInject && tr->GetTrackID() == 1 && gOut.done[ev] == 0) {
      const G4VProcess* pr = step->GetPostStepPoint()->GetProcessDefinedStep();
      if (pr != nullptr && pr->GetProcessName() == "muonNuclear") {
        gOut.done[ev] = 1;
        const double e_out = tr->GetTrackStatus() == fAlive ? step->GetPostStepPoint()->GetTotalEnergy() / GeV : 0.0;
        gOut.nu[ev] = step->GetPreStepPoint()->GetTotalEnergy() / GeV - e_out;
      } else {
        tr->SetTrackStatus(fStopAndKill);          // the first step was not the forced interaction: drop the event
        return;
      }
      force(1.0);                                  // everything after it with the real cross sections
    }
    if (gRecord) record_step(step, ev);
    const G4StepPoint* post = step->GetPostStepPoint();
    if (step->GetPreStepPoint()->GetPhysicalVolume() == gRock && post->GetStepStatus() == fGeomBoundary &&
        post->GetPhysicalVolume() != gRock && post->GetMomentumDirection().y() > 0) {
      const G4ThreeVector x = post->GetPosition(), p = post->GetMomentum();
      gOut.x_ev.push_back(static_cast<int>(ev)); gOut.x_pdg.push_back(tr->GetDefinition()->GetPDGEncoding());
      gOut.x_track.push_back(tr->GetTrackID()); gOut.x_parent.push_back(tr->GetParentID());
      gOut.x_ekin.push_back(post->GetKineticEnergy() / GeV);
      gOut.x_x.push_back(x.x() / m); gOut.x_z.push_back(x.z() / m);
      gOut.x_px.push_back(p.x() / GeV); gOut.x_py.push_back(p.y() / GeV); gOut.x_pz.push_back(p.z() / GeV);
      gOut.x_t.push_back(post->GetGlobalTime() / ns);
      tr->SetTrackStatus(fStopAndKill);
    }
  }
};

class Stacking : public G4UserStackingAction {
 public:
  G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track* t) override {
    if (t->GetParentID() == 0) return fUrgent;
    const int a = std::abs(t->GetDefinition()->GetPDGEncoding());
    if (a == 12 || a == 14 || a == 16) return fKill;
    if (t->GetDefinition()->GetParticleType() == "nucleus" && t->GetDefinition()->GetAtomicNumber() > 2) return fKill;
    if (t->GetKineticEnergy() < gEcut * GeV) return fKill;
    return fUrgent;
  }
};

G4RunManager* gRun = nullptr;

void initialize(long seed, const std::string& material, double e_cut, double half_xz, double rock_depth, double air_height) {
  if (gRun != nullptr) throw std::runtime_error("g4rock already initialized (one per process)");
  gMaterial = material; gEcut = e_cut; gHalfXZ = half_xz; gRockDepth = rock_depth; gAirHeight = air_height;
  CLHEP::HepRandom::setTheSeed(seed);
  gRun = new G4RunManager;
  gRun->SetUserInitialization(new Detector);
  gRun->SetUserInitialization(new FTFP_BERT(0));
  gRun->SetUserAction(new Generator);
  gRun->SetUserAction(new Events);
  gRun->SetUserAction(new Stepping);
  gRun->SetUserAction(new Stacking);
  auto* ui = G4UImanager::GetUIpointer();
  for (const char* c : {"/control/verbose 0", "/run/verbose 0", "/event/verbose 0", "/tracking/verbose 0", "/process/had/verbose 0",
                        "/process/em/verbose 0"})
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

py::dict pack();

py::dict run(int n) {
  gOut.clear();
  {
    py::gil_scoped_release release;
    gRun->BeamOn(n);
  }
  force(1.0);
  return pack();
}

py::dict shoot(int pdg, double E, int n, double depth, double alpha) {
  if (gRun == nullptr) throw std::runtime_error("call initialize() first");
  if (depth <= 0 || depth >= gRockDepth) throw std::runtime_error("depth must be inside the rock");
  gInject = false; gDepths.clear();
  gPdg = pdg; gE = E; gDepth = depth; gAlpha = alpha;
  return run(n);
}

py::dict shoot_depths(int pdg, double E, double alpha, py::array_t<double, py::array::c_style | py::array::forcecast> depths) {
  if (gRun == nullptr) throw std::runtime_error("call initialize() first");
  auto d = depths.unchecked<1>();
  gDepths.assign(d.data(0), d.data(0) + d.shape(0));
  for (double x : gDepths) if (x <= 0 || x >= gRockDepth) throw std::runtime_error("depth must be inside the rock");
  gInject = false; gPdg = pdg; gE = E; gAlpha = alpha;
  py::dict out = run(static_cast<int>(gDepths.size()));
  gDepths.clear();
  return out;
}

template <typename T>
std::vector<T> vec(py::array_t<T, py::array::c_style | py::array::forcecast> a) { auto u = a.template unchecked<1>(); return std::vector<T>(u.data(0), u.data(0) + u.shape(0)); }

py::dict inject(py::array_t<long, py::array::c_style | py::array::forcecast> offsets, py::array_t<int, py::array::c_style | py::array::forcecast> pdg,
                py::array_t<double, py::array::c_style | py::array::forcecast> px, py::array_t<double, py::array::c_style | py::array::forcecast> py_,
                py::array_t<double, py::array::c_style | py::array::forcecast> pz, py::array_t<double, py::array::c_style | py::array::forcecast> x,
                py::array_t<double, py::array::c_style | py::array::forcecast> y, py::array_t<double, py::array::c_style | py::array::forcecast> z) {
  if (gRun == nullptr) throw std::runtime_error("call initialize() first");
  gOff = vec<long>(offsets); gIpdg = vec<int>(pdg);
  gIpx = vec<double>(px); gIpy = vec<double>(py_); gIpz = vec<double>(pz); gIx = vec<double>(x); gIy = vec<double>(y); gIz = vec<double>(z);
  if (gOff.size() < 1 || static_cast<size_t>(gOff.back()) != gIpdg.size()) throw std::runtime_error("offsets must end at the number of particles");
  gInject = true;
  py::dict out = run(static_cast<int>(gOff.size()) - 1);
  gInject = false;
  return out;
}

py::dict pack() {
  py::dict ev, ex, out;
  ev["done"] = arr(gOut.done); ev["nu"] = arr(gOut.nu);
  ex["ev"] = arr(gOut.x_ev); ex["pdg"] = arr(gOut.x_pdg); ex["track"] = arr(gOut.x_track); ex["parent"] = arr(gOut.x_parent);
  ex["ekin"] = arr(gOut.x_ekin); ex["x"] = arr(gOut.x_x); ex["z"] = arr(gOut.x_z);
  ex["px"] = arr(gOut.x_px); ex["py"] = arr(gOut.x_py); ex["pz"] = arr(gOut.x_pz); ex["t"] = arr(gOut.x_t);
  out["events"] = ev; out["exits"] = ex;
  if (gRecord) {
    py::dict tk, pt;
    tk["ev"] = arr(gOut.t_ev); tk["track"] = arr(gOut.t_track); tk["parent"] = arr(gOut.t_parent); tk["pdg"] = arr(gOut.t_pdg);
    tk["ekin0"] = arr(gOut.t_ekin0); tk["first"] = arr(gOut.t_first); tk["n"] = arr(gOut.t_n);
    pt["x"] = arr(gOut.q_x); pt["y"] = arr(gOut.q_y); pt["z"] = arr(gOut.q_z); pt["t"] = arr(gOut.q_t);
    out["tracks"] = tk; out["points"] = pt;
  }
  return out;
}

double cross_section(int pdg, double E) {
  if (gRun == nullptr) throw std::runtime_error("call initialize() first");
  const G4Material* mat = G4NistManager::Instance()->FindOrBuildMaterial(gMaterial);
  auto* def = G4ParticleTable::GetParticleTable()->FindParticle(pdg);
  G4DynamicParticle dp(def, G4ThreeVector(0, 0, 1), E * GeV - def->GetPDGMass());
  force(1.0);
  for (auto* p : gMuNuclear)
    if (p->GetProcessManager()->GetParticleType() == def)
      return p->GetCrossSectionDataStore()->ComputeCrossSection(&dp, mat) * m;
  return 0.0;
}

}  // namespace

PYBIND11_MODULE(g4rock, mod) {
  mod.doc() = "Forced muon-nuclear interactions in a rock half-space; what comes out of its surface";
  mod.def("initialize", &initialize, py::arg("seed") = 1, py::arg("material") = "G4_CONCRETE", py::arg("e_cut") = 1e-3,
          py::arg("half_xz") = 200.0, py::arg("rock_depth") = 30.0, py::arg("air_height") = 30.0);
  mod.def("shoot", &shoot, py::arg("pdg"), py::arg("E"), py::arg("n"), py::arg("depth"), py::arg("alpha"));
  mod.def("shoot_depths", &shoot_depths, py::arg("pdg"), py::arg("E"), py::arg("alpha"), py::arg("depths"));
  mod.def("inject", &inject, py::arg("offsets"), py::arg("pdg"), py::arg("px"), py::arg("py"), py::arg("pz"), py::arg("x"), py::arg("y"), py::arg("z"));
  mod.def("cross_section", &cross_section, py::arg("pdg"), py::arg("E"));
  mod.def("set_record", [](bool on, double em_min, double thin, int max_points) { gRecord = on; gEmMin = em_min; gThin = thin; gMaxPts = max_points; },
          py::arg("on"), py::arg("em_min") = 0.05, py::arg("thin") = 0.05, py::arg("max_points") = 300);
}
