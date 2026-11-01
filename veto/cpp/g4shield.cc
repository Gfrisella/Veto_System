// Transport of interaction final states made in the last iron of the muon shield (TRY6 MS7) to a plane behind it.
//
// Geometry (metres, shield frame of MuonsAndMatter: shield end z = 29.89 m): air world; one G4_Fe box
// |x| < half_x, |y| < half_y, z_iron0 < z < z_iron1 (MS7 is one solid iron Arb8 whose outer envelope is constant,
// 1.71 x 1.27 m, MuonsAndMatter create_magnet). Magnetic field everywhere: the MuonsAndMatter field map (first
// quadrant, mirrored as magnet_simulations.get_symmetry: By even, Bx odd in x and y, Bz odd in y), trilinear,
// looked up at z + z_shift (the map's own z frame), zero outside the map.
// Physics: FTFP_BERT, decays on. Every particle crossing z_plane in +z is recorded and killed. Killed when created:
// neutrinos, nuclei, and anything below e_cut [GeV kinetic] (e+-, photons included).
//
// Python API (module g4shield):
//   initialize(seed, z_iron0, z_iron1, half_x, half_y, z_plane, e_cut, field, x0, dx, y0, dy, z0, dz, z_shift)
//       field: (nx, ny, nz, 3) float64 array [T] on the grid x0 + i dx, ... (m); empty array for no field
//   transport(offsets, pdg, px, py, pz, vx, vy, vz) -> dict of the particles at the plane
//       event i = particles offsets[i] .. offsets[i+1]-1, each from its own production point v (m), p in GeV.
//       Returns ev, pdg, track (track ID; primaries keep their input order + 1), primary (1 = an injected
//       particle, unchanged identity), x, y (m, at the plane), px, py, pz (GeV), ekin (GeV), t (ns)
//   layout() -> dict(z_iron0, z_iron1, half_x, half_y, z_plane)

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FTFP_BERT.hh"
#include "G4Box.hh"
#include "G4ChordFinder.hh"
#include "G4Event.hh"
#include "G4FieldManager.hh"
#include "G4LogicalVolume.hh"
#include "G4MagneticField.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4RunManager.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4TransportationManager.hh"
#include "G4Track.hh"
#include "G4UImanager.hh"
#include "G4UserStackingAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "Randomize.hh"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

double gZ0 = 29.39, gZ1 = 29.89, gHX = 1.71, gHY = 1.27, gZPlane = 29.90, gECut = 0.5;   // m, GeV

struct FieldGrid {
  std::vector<double> b;                 // nx * ny * nz * 3, C order
  int nx = 0, ny = 0, nz = 0;
  double x0 = 0, dx = 1, y0 = 0, dy = 1, z0 = 0, dz = 1, zshift = 0;
} gField;

class MapField : public G4MagneticField {
 public:
  void GetFieldValue(const G4double pt[4], G4double* B) const override {
    B[0] = B[1] = B[2] = 0;
    const auto& f = gField;
    if (f.nx < 2) return;
    const double x = pt[0] / m, y = pt[1] / m, z = pt[2] / m + f.zshift;
    const double ax = std::abs(x), ay = std::abs(y);
    const double fx = (ax - f.x0) / f.dx, fy = (ay - f.y0) / f.dy, fz = (z - f.z0) / f.dz;
    if (fx < 0 || fy < 0 || fz < 0 || fx >= f.nx - 1 || fy >= f.ny - 1 || fz >= f.nz - 1) return;
    const int i = int(fx), j = int(fy), k = int(fz);
    const double u = fx - i, v = fy - j, w = fz - k;
    double out[3] = {0, 0, 0};
    for (int di = 0; di < 2; ++di)
      for (int dj = 0; dj < 2; ++dj)
        for (int dk = 0; dk < 2; ++dk) {
          const double c = (di ? u : 1 - u) * (dj ? v : 1 - v) * (dk ? w : 1 - w);
          const size_t idx = ((size_t(i + di) * f.ny + (j + dj)) * f.nz + (k + dk)) * 3;
          for (int a = 0; a < 3; ++a) out[a] += c * f.b[idx + a];
        }
    const double sx = x < 0 ? -1 : 1, sy = y < 0 ? -1 : 1;
    B[0] = out[0] * sx * sy * tesla;
    B[1] = out[1] * tesla;
    B[2] = out[2] * sy * tesla;
  }
};

class Detector : public G4VUserDetectorConstruction {
 public:
  G4VPhysicalVolume* Construct() override {
    auto* nist = G4NistManager::Instance();
    const double zlo = gZ0 - 1.0, zhi = gZPlane + 1.0, zc = 0.5 * (zlo + zhi);
    auto* worldLV = new G4LogicalVolume(new G4Box("World", 8 * m, 8 * m, 0.5 * (zhi - zlo) * m),
                                        nist->FindOrBuildMaterial("G4_AIR"), "World");
    // world centred at zc: every placement is relative to it, positions handed in / out are in the shield frame
    auto* worldPV = new G4PVPlacement(nullptr, {}, worldLV, "World", nullptr, false, 0);
    gOffset = zc;
    new G4PVPlacement(nullptr, G4ThreeVector(0, 0, (0.5 * (gZ0 + gZ1) - zc) * m),
                      new G4LogicalVolume(new G4Box("Iron", gHX * m, gHY * m, 0.5 * (gZ1 - gZ0) * m),
                                          nist->FindOrBuildMaterial("G4_Fe"), "Iron"),
                      "Iron", worldLV, false, 0);
    if (gField.nx >= 2) {
      field_ = std::make_unique<ShiftedField>();
      auto* fm = G4TransportationManager::GetTransportationManager()->GetFieldManager();
      fm->SetDetectorField(field_.get());
      fm->CreateChordFinder(field_.get());
    }
    return worldPV;
  }
  static double gOffset;                          // shield-frame z of the world centre [m]

 private:
  struct ShiftedField : public MapField {         // world-frame point -> shield frame
    void GetFieldValue(const G4double pt[4], G4double* B) const override {
      const G4double q[4] = {pt[0], pt[1], pt[2] + gOffset * m, pt[3]};
      MapField::GetFieldValue(q, B);
    }
  };
  std::unique_ptr<ShiftedField> field_;
};
double Detector::gOffset = 0.0;

struct Inputs { std::vector<int> pdg; std::vector<double> px, py, pz, vx, vy, vz; std::vector<long> off; } gIn;
struct Out {
  std::vector<int> ev, pdg, track, primary;
  std::vector<double> x, y, px, py, pz, ekin, t;
} gOut;

class Generator : public G4VUserPrimaryGeneratorAction {
 public:
  Generator() : gun_(std::make_unique<G4ParticleGun>(1)) {}
  void GeneratePrimaries(G4Event* ev) override {
    const size_t e = ev->GetEventID();
    for (long j = gIn.off[e]; j < gIn.off[e + 1]; ++j) {
      auto* def = G4ParticleTable::GetParticleTable()->FindParticle(gIn.pdg[j]);
      if (def == nullptr) continue;                 // unknown code: the slot keeps its track ID, nothing is shot
      G4ThreeVector p(gIn.px[j], gIn.py[j], gIn.pz[j]);
      if (p.mag2() <= 0) continue;
      const double mass = def->GetPDGMass();
      gun_->SetParticleDefinition(def);
      gun_->SetParticleMomentumDirection(p.unit());
      gun_->SetParticleEnergy(std::sqrt(p.mag2() * GeV * GeV + mass * mass) - mass);
      gun_->SetParticlePosition(G4ThreeVector(gIn.vx[j], gIn.vy[j], gIn.vz[j] - Detector::gOffset) * m);
      gun_->GeneratePrimaryVertex(ev);
    }
  }
 private:
  std::unique_ptr<G4ParticleGun> gun_;
};

class Stepping : public G4UserSteppingAction {
 public:
  void UserSteppingAction(const G4Step* step) override {
    const G4StepPoint* pre = step->GetPreStepPoint();
    const G4StepPoint* post = step->GetPostStepPoint();
    const double zp = gZPlane - Detector::gOffset;
    const double z0 = pre->GetPosition().z() / m, z1 = post->GetPosition().z() / m;
    if (!(z0 < zp && z1 >= zp)) return;
    G4Track* tr = step->GetTrack();
    const double f = (zp - z0) / (z1 - z0);
    const G4ThreeVector x = pre->GetPosition() + f * (post->GetPosition() - pre->GetPosition());
    const G4ThreeVector p = pre->GetMomentum() + f * (post->GetMomentum() - pre->GetMomentum());
    gOut.ev.push_back(G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID());
    gOut.pdg.push_back(tr->GetDefinition()->GetPDGEncoding());
    gOut.track.push_back(tr->GetTrackID());
    gOut.primary.push_back(tr->GetParentID() == 0 ? 1 : 0);
    gOut.x.push_back(x.x() / m); gOut.y.push_back(x.y() / m);
    gOut.px.push_back(p.x() / GeV); gOut.py.push_back(p.y() / GeV); gOut.pz.push_back(p.z() / GeV);
    gOut.ekin.push_back((pre->GetKineticEnergy() + f * (post->GetKineticEnergy() - pre->GetKineticEnergy())) / GeV);
    gOut.t.push_back((pre->GetGlobalTime() + f * (post->GetGlobalTime() - pre->GetGlobalTime())) / ns);
    tr->SetTrackStatus(fStopAndKill);
  }
};

class Stacking : public G4UserStackingAction {
 public:
  G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track* t) override {
    const int a = std::abs(t->GetDefinition()->GetPDGEncoding());
    if (a == 12 || a == 14 || a == 16 || a == 0) return fKill;
    if (t->GetDefinition()->GetParticleType() == "nucleus") return fKill;
    return t->GetKineticEnergy() > gECut * GeV ? fUrgent : fKill;
  }
};

G4RunManager* gRun = nullptr;

template <typename T>
std::vector<T> vec(const py::array_t<T, py::array::c_style | py::array::forcecast>& a) {
  return std::vector<T>(a.data(), a.data() + a.size());
}

void initialize(long seed, double z_iron0, double z_iron1, double half_x, double half_y, double z_plane, double e_cut,
                py::array_t<double, py::array::c_style | py::array::forcecast> field, double x0, double dx, double y0,
                double dy, double z0, double dz, double z_shift) {
  if (gRun) throw std::runtime_error("already initialized");
  gZ0 = z_iron0; gZ1 = z_iron1; gHX = half_x; gHY = half_y; gZPlane = z_plane; gECut = e_cut;
  if (field.size() > 0) {
    if (field.ndim() != 4 || field.shape(3) != 3) throw std::runtime_error("field must be (nx, ny, nz, 3)");
    gField.b = vec<double>(field);
    gField.nx = field.shape(0); gField.ny = field.shape(1); gField.nz = field.shape(2);
    gField.x0 = x0; gField.dx = dx; gField.y0 = y0; gField.dy = dy; gField.z0 = z0; gField.dz = dz;
    gField.zshift = z_shift;
  }
  CLHEP::HepRandom::setTheSeed(seed);
  gRun = new G4RunManager;
  gRun->SetUserInitialization(new Detector);
  gRun->SetUserInitialization(new FTFP_BERT(0));
  gRun->SetUserAction(new Generator);
  gRun->SetUserAction(new Stepping);
  gRun->SetUserAction(new Stacking);
  for (auto c : {"/control/verbose 0", "/run/verbose 0", "/event/verbose 0", "/tracking/verbose 0"})
    G4UImanager::GetUIpointer()->ApplyCommand(c);
  gRun->Initialize();
}

py::dict transport(py::array_t<long, py::array::c_style | py::array::forcecast> offsets,
                   py::array_t<int, py::array::c_style | py::array::forcecast> pdg,
                   py::array_t<double, py::array::c_style | py::array::forcecast> px,
                   py::array_t<double, py::array::c_style | py::array::forcecast> py_,
                   py::array_t<double, py::array::c_style | py::array::forcecast> pz,
                   py::array_t<double, py::array::c_style | py::array::forcecast> vx,
                   py::array_t<double, py::array::c_style | py::array::forcecast> vy,
                   py::array_t<double, py::array::c_style | py::array::forcecast> vz) {
  if (!gRun) throw std::runtime_error("call initialize first");
  gIn.off = vec<long>(offsets); gIn.pdg = vec<int>(pdg);
  gIn.px = vec<double>(px); gIn.py = vec<double>(py_); gIn.pz = vec<double>(pz);
  gIn.vx = vec<double>(vx); gIn.vy = vec<double>(vy); gIn.vz = vec<double>(vz);
  const long n_ev = long(gIn.off.size()) - 1;
  if (n_ev < 0 || gIn.off.back() != long(gIn.pdg.size())) throw std::runtime_error("offsets do not match particles");
  gOut = Out();
  if (n_ev > 0) gRun->BeamOn(int(n_ev));
  py::dict d;
  d["ev"] = py::array(py::cast(gOut.ev)); d["pdg"] = py::array(py::cast(gOut.pdg));
  d["track"] = py::array(py::cast(gOut.track)); d["primary"] = py::array(py::cast(gOut.primary));
  d["x"] = py::array(py::cast(gOut.x)); d["y"] = py::array(py::cast(gOut.y));
  d["px"] = py::array(py::cast(gOut.px)); d["py"] = py::array(py::cast(gOut.py)); d["pz"] = py::array(py::cast(gOut.pz));
  d["ekin"] = py::array(py::cast(gOut.ekin)); d["t"] = py::array(py::cast(gOut.t));
  return d;
}

py::dict layout() {
  py::dict d;
  d["z_iron0"] = gZ0; d["z_iron1"] = gZ1; d["half_x"] = gHX; d["half_y"] = gHY; d["z_plane"] = gZPlane;
  return d;
}

}  // namespace

PYBIND11_MODULE(g4shield, mod) {
  mod.def("initialize", &initialize, py::arg("seed"), py::arg("z_iron0"), py::arg("z_iron1"), py::arg("half_x"),
          py::arg("half_y"), py::arg("z_plane"), py::arg("e_cut"), py::arg("field"), py::arg("x0") = 0.0,
          py::arg("dx") = 1.0, py::arg("y0") = 0.0, py::arg("dy") = 1.0, py::arg("z0") = 0.0, py::arg("dz") = 1.0,
          py::arg("z_shift") = 0.0);
  mod.def("transport", &transport, py::arg("offsets"), py::arg("pdg"), py::arg("px"), py::arg("py"), py::arg("pz"),
          py::arg("vx"), py::arg("vy"), py::arg("vz"));
  mod.def("layout", &layout);
}
