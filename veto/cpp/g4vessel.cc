// Geant4 transport of interaction final states from their production point in / around the decay vessel to the
// straw stations and the timing detector (ISSUES A6: helium products in Geant4; D4: decays in flight, K_L / neutron
// interactions in air and helium after the shield).
//
// Geometry (metres, veto frame = FairShip z - 2.14, lib/veto/geometry.py): air world |x|, |y| < 10 m,
// z_world0 < z < z_world1; the decay vessel = a G4Trd of Mylar (wall_thickness) filled with a G4Trd of helium
// (1 atm, 20 C: 0.1664 kg/m^3), z_vessel0 .. z_vessel1, half widths hx0 x hy0 -> hx1 x hy1 (inner). Nothing else:
// straw stations, magnet iron and SBT are not built. The magnet iron is handled as in lib/veto/spectrometer.py:
// a track that meets |B| > yoke_b is killed (lost in the yoke).
// Field: the spectrometer map (full grid, no mirroring, already in the veto frame), trilinear, zero outside.
// Physics: FTFP_BERT, decays on (pi / K / K_S / Lambda / mu decay in flight).
//
// Recording: every charged or neutral particle crossing one of the planes z_planes[k] in +z is recorded; it goes
// on after every plane but the last, where it is killed. Killed when created: neutrinos, nuclei, and anything
// below e_cut [GeV kinetic].
// Origin: a primary carries its input index (within its event); a particle made by the DECAY of a particle with an
// origin inherits that origin (pi -> mu nu keeps the track's identity); anything else (hadronic secondaries, delta
// rays, conversions) has origin -1.
//
// Python API (module g4vessel):
//   initialize(seed, z_world0, z_world1, z_vessel0, z_vessel1, hx0, hx1, hy0, hy1, wall_thickness, wall_material,
//              z_planes, e_cut, yoke_b, field, x0, dx, y0, dy, z0, dz)
//       wall_material: Geant4 / NIST name ("G4_MYLAR"); field (nx, ny, nz, 3) [T] on x0 + i dx ... (m), or empty
//   transport(offsets, pdg, px, py, pz, vx, vy, vz) -> dict of the plane crossings:
//       ev, plane (index into z_planes), pdg, track, origin, x, y (m), px, py, pz (GeV), t (ns)
//   layout() -> dict

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FTFP_BERT.hh"
#include "G4Box.hh"
#include "G4Event.hh"
#include "G4FieldManager.hh"
#include "G4LogicalVolume.hh"
#include "G4MagneticField.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4RunManager.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4TransportationManager.hh"
#include "G4Trd.hh"
#include "G4UImanager.hh"
#include "G4UserStackingAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4UserTrackingAction.hh"
#include "G4VProcess.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "Randomize.hh"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace {

struct Layout {
  double zw0 = 29.0, zw1 = 95.0, zv0 = 31.13, zv1 = 81.13, hx0 = 0.5, hx1 = 2.0, hy0 = 1.35, hy1 = 3.0;
  double wall = 350e-6, ecut = 0.1, yoke_b = 0.5;
  std::string wall_material = "G4_MYLAR";
  std::vector<double> planes{82.08, 94.01};
} gL;

struct FieldGrid {
  std::vector<float> b;                  // nx * ny * nz * 3, C order
  int nx = 0, ny = 0, nz = 0;
  double x0 = 0, dx = 1, y0 = 0, dy = 1, z0 = 0, dz = 1;
} gField;

double gOffset = 0.0;                    // veto-frame z of the world centre [m]

// trilinear field [T] at a veto-frame point (m); zero outside the map
void field_tesla(double x, double y, double z, double out[3]) {
  out[0] = out[1] = out[2] = 0;
  const auto& f = gField;
  if (f.nx < 2) return;
  const double fx = (x - f.x0) / f.dx, fy = (y - f.y0) / f.dy, fz = (z - f.z0) / f.dz;
  if (fx < 0 || fy < 0 || fz < 0 || fx >= f.nx - 1 || fy >= f.ny - 1 || fz >= f.nz - 1) return;
  const int i = int(fx), j = int(fy), k = int(fz);
  const double u = fx - i, v = fy - j, w = fz - k;
  for (int di = 0; di < 2; ++di)
    for (int dj = 0; dj < 2; ++dj)
      for (int dk = 0; dk < 2; ++dk) {
        const double c = (di ? u : 1 - u) * (dj ? v : 1 - v) * (dk ? w : 1 - w);
        const size_t idx = ((size_t(i + di) * f.ny + (j + dj)) * f.nz + (k + dk)) * 3;
        for (int a = 0; a < 3; ++a) out[a] += c * f.b[idx + a];
      }
}

class MapField : public G4MagneticField {
 public:
  void GetFieldValue(const G4double pt[4], G4double* B) const override {
    double t[3];
    field_tesla(pt[0] / m, pt[1] / m, pt[2] / m + gOffset, t);
    for (int a = 0; a < 3; ++a) B[a] = t[a] * tesla;
  }
};

class Detector : public G4VUserDetectorConstruction {
 public:
  G4VPhysicalVolume* Construct() override {
    auto* nist = G4NistManager::Instance();
    gOffset = 0.5 * (gL.zw0 + gL.zw1);
    auto* air = nist->FindOrBuildMaterial("G4_AIR");
    auto* worldLV = new G4LogicalVolume(new G4Box("World", 10 * m, 10 * m, 0.5 * (gL.zw1 - gL.zw0) * m), air, "World");
    auto* worldPV = new G4PVPlacement(nullptr, {}, worldLV, "World", nullptr, false, 0);
    // helium at 1 atm, 20 C (fastveto.vessel.MEDIA)
    auto* he = new G4Material("HeliumVessel", 2., 4.0026 * g / mole, 0.1664 * kg / m3, kStateGas, 293.15 * kelvin,
                              1 * atmosphere);
    auto* wall = nist->FindOrBuildMaterial(gL.wall_material);
    if (wall == nullptr) throw std::runtime_error("unknown wall material " + gL.wall_material);
    const double hz = 0.5 * (gL.zv1 - gL.zv0), zc = 0.5 * (gL.zv0 + gL.zv1) - gOffset, t = gL.wall;
    // outer Trd: every half size + the wall thickness (as fastveto.vessel.OUTER_BOX)
    auto* outerLV = new G4LogicalVolume(
        new G4Trd("VesselWall", (gL.hx0 + t) * m, (gL.hx1 + t) * m, (gL.hy0 + t) * m, (gL.hy1 + t) * m, (hz + t) * m),
        wall, "VesselWall");
    new G4PVPlacement(nullptr, G4ThreeVector(0, 0, zc * m), outerLV, "VesselWall", worldLV, false, 0);
    auto* heLV = new G4LogicalVolume(new G4Trd("Helium", gL.hx0 * m, gL.hx1 * m, gL.hy0 * m, gL.hy1 * m, hz * m), he,
                                     "Helium");
    new G4PVPlacement(nullptr, {}, heLV, "Helium", outerLV, false, 0);
    if (gField.nx >= 2) {
      field_ = std::make_unique<MapField>();
      auto* fm = G4TransportationManager::GetTransportationManager()->GetFieldManager();
      fm->SetDetectorField(field_.get());
      fm->CreateChordFinder(field_.get());
    }
    return worldPV;
  }

 private:
  std::unique_ptr<MapField> field_;
};

struct Inputs { std::vector<int> pdg; std::vector<double> px, py, pz, vx, vy, vz; std::vector<long> off; } gIn;
struct Out {
  std::vector<int> ev, plane, pdg, track, origin;
  std::vector<double> x, y, px, py, pz, t;
} gOut;

std::vector<int> gPrimaryOrigin;                 // generation order of the primaries -> input index in the event
std::unordered_map<int, int> gOrigin;            // track ID -> origin

class Generator : public G4VUserPrimaryGeneratorAction {
 public:
  Generator() : gun_(std::make_unique<G4ParticleGun>(1)) {}
  void GeneratePrimaries(G4Event* ev) override {
    const size_t e = ev->GetEventID();
    gPrimaryOrigin.clear();
    gOrigin.clear();
    for (long j = gIn.off[e]; j < gIn.off[e + 1]; ++j) {
      auto* def = G4ParticleTable::GetParticleTable()->FindParticle(gIn.pdg[j]);
      if (def == nullptr) continue;
      G4ThreeVector p(gIn.px[j], gIn.py[j], gIn.pz[j]);
      if (p.mag2() <= 0) continue;
      const double mass = def->GetPDGMass();
      gun_->SetParticleDefinition(def);
      gun_->SetParticleMomentumDirection(p.unit());
      gun_->SetParticleEnergy(std::sqrt(p.mag2() * GeV * GeV + mass * mass) - mass);
      gun_->SetParticlePosition(G4ThreeVector(gIn.vx[j], gIn.vy[j], gIn.vz[j] - gOffset) * m);
      gun_->GeneratePrimaryVertex(ev);
      gPrimaryOrigin.push_back(int(j - gIn.off[e]));
    }
  }
 private:
  std::unique_ptr<G4ParticleGun> gun_;
};

class Tracking : public G4UserTrackingAction {
 public:
  void PreUserTrackingAction(const G4Track* tr) override {
    int origin = -1;
    if (tr->GetParentID() == 0) {
      const int k = tr->GetTrackID() - 1;          // primaries get IDs 1 .. n in generation order
      if (k >= 0 && k < int(gPrimaryOrigin.size())) origin = gPrimaryOrigin[k];
    } else {
      const G4VProcess* proc = tr->GetCreatorProcess();
      if (proc != nullptr && proc->GetProcessName() == "Decay") {
        auto it = gOrigin.find(tr->GetParentID());
        if (it != gOrigin.end()) origin = it->second;
      }
    }
    gOrigin[tr->GetTrackID()] = origin;
  }
};

class Stepping : public G4UserSteppingAction {
 public:
  void UserSteppingAction(const G4Step* step) override {
    const G4StepPoint* pre = step->GetPreStepPoint();
    const G4StepPoint* post = step->GetPostStepPoint();
    G4Track* tr = step->GetTrack();
    const double z0 = pre->GetPosition().z() / m + gOffset, z1 = post->GetPosition().z() / m + gOffset;
    for (size_t k = 0; k < gL.planes.size(); ++k) {
      const double zp = gL.planes[k];
      if (!(z0 < zp && z1 >= zp)) continue;
      const double f = (zp - z0) / (z1 - z0);
      const G4ThreeVector x = pre->GetPosition() + f * (post->GetPosition() - pre->GetPosition());
      const G4ThreeVector p = pre->GetMomentum() + f * (post->GetMomentum() - pre->GetMomentum());
      gOut.ev.push_back(G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID());
      gOut.plane.push_back(int(k));
      gOut.pdg.push_back(tr->GetDefinition()->GetPDGEncoding());
      gOut.track.push_back(tr->GetTrackID());
      auto it = gOrigin.find(tr->GetTrackID());
      gOut.origin.push_back(it == gOrigin.end() ? -1 : it->second);
      gOut.x.push_back(x.x() / m); gOut.y.push_back(x.y() / m);
      gOut.px.push_back(p.x() / GeV); gOut.py.push_back(p.y() / GeV); gOut.pz.push_back(p.z() / GeV);
      gOut.t.push_back((pre->GetGlobalTime() + f * (post->GetGlobalTime() - pre->GetGlobalTime())) / ns);
      if (k + 1 == gL.planes.size()) { tr->SetTrackStatus(fStopAndKill); return; }
    }
    if (z1 >= gL.zw1 - 1e-3) { tr->SetTrackStatus(fStopAndKill); return; }
    // magnet iron (not built): killed where the field says yoke (lib/veto/spectrometer.YOKE_B)
    if (tr->GetDefinition()->GetPDGCharge() != 0) {
      double b[3];
      const G4ThreeVector q = post->GetPosition();
      field_tesla(q.x() / m, q.y() / m, z1, b);
      if (std::sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]) > gL.yoke_b) tr->SetTrackStatus(fStopAndKill);
    }
  }
};

class Stacking : public G4UserStackingAction {
 public:
  G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track* t) override {
    const int a = std::abs(t->GetDefinition()->GetPDGEncoding());
    if (a == 12 || a == 14 || a == 16 || a == 0) return fKill;
    if (t->GetDefinition()->GetParticleType() == "nucleus") return fKill;
    return t->GetKineticEnergy() > gL.ecut * GeV ? fUrgent : fKill;
  }
};

G4RunManager* gRun = nullptr;

template <typename T>
std::vector<T> vec(const py::array_t<T, py::array::c_style | py::array::forcecast>& a) {
  return std::vector<T>(a.data(), a.data() + a.size());
}

void initialize(long seed, double z_world0, double z_world1, double z_vessel0, double z_vessel1, double hx0,
                double hx1, double hy0, double hy1, double wall_thickness, std::string wall_material,
                py::array_t<double, py::array::c_style | py::array::forcecast> z_planes, double e_cut, double yoke_b,
                py::array_t<float, py::array::c_style | py::array::forcecast> field, double x0, double dx, double y0,
                double dy, double z0, double dz) {
  if (gRun) throw std::runtime_error("already initialized");
  gL.zw0 = z_world0; gL.zw1 = z_world1; gL.zv0 = z_vessel0; gL.zv1 = z_vessel1;
  gL.hx0 = hx0; gL.hx1 = hx1; gL.hy0 = hy0; gL.hy1 = hy1; gL.wall = wall_thickness; gL.wall_material = wall_material;
  gL.planes = vec<double>(z_planes); gL.ecut = e_cut; gL.yoke_b = yoke_b;
  if (gL.planes.empty() || !std::is_sorted(gL.planes.begin(), gL.planes.end()))
    throw std::runtime_error("z_planes must be non-empty and increasing");
  if (gL.planes.back() >= gL.zw1) throw std::runtime_error("last plane must be inside the world");
  if (field.size() > 0) {
    if (field.ndim() != 4 || field.shape(3) != 3) throw std::runtime_error("field must be (nx, ny, nz, 3)");
    gField.b = vec<float>(field);
    gField.nx = field.shape(0); gField.ny = field.shape(1); gField.nz = field.shape(2);
    gField.x0 = x0; gField.dx = dx; gField.y0 = y0; gField.dy = dy; gField.z0 = z0; gField.dz = dz;
  }
  CLHEP::HepRandom::setTheSeed(seed);
  gRun = new G4RunManager;
  gRun->SetUserInitialization(new Detector);
  gRun->SetUserInitialization(new FTFP_BERT(0));
  gRun->SetUserAction(new Generator);
  gRun->SetUserAction(new Tracking);
  gRun->SetUserAction(new Stepping);
  gRun->SetUserAction(new Stacking);
  for (auto c : {"/control/verbose 0", "/run/verbose 0", "/event/verbose 0", "/tracking/verbose 0",
                 "/process/had/verbose 0", "/process/em/verbose 0"})
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
  for (size_t j = 0; j < gIn.vz.size(); ++j)
    if (gIn.vz[j] <= gL.zw0 || gIn.vz[j] >= gL.planes.back())
      throw std::runtime_error("a production point is outside z_world0 .. last plane");
  gOut = Out();
  if (n_ev > 0) gRun->BeamOn(int(n_ev));
  py::dict d;
  d["ev"] = py::array(py::cast(gOut.ev)); d["plane"] = py::array(py::cast(gOut.plane));
  d["pdg"] = py::array(py::cast(gOut.pdg)); d["track"] = py::array(py::cast(gOut.track));
  d["origin"] = py::array(py::cast(gOut.origin));
  d["x"] = py::array(py::cast(gOut.x)); d["y"] = py::array(py::cast(gOut.y));
  d["px"] = py::array(py::cast(gOut.px)); d["py"] = py::array(py::cast(gOut.py)); d["pz"] = py::array(py::cast(gOut.pz));
  d["t"] = py::array(py::cast(gOut.t));
  return d;
}

py::dict layout() {
  py::dict d;
  d["z_world"] = py::make_tuple(gL.zw0, gL.zw1); d["z_vessel"] = py::make_tuple(gL.zv0, gL.zv1);
  d["half_in"] = py::make_tuple(gL.hx0, gL.hy0); d["half_out"] = py::make_tuple(gL.hx1, gL.hy1);
  d["wall"] = gL.wall; d["wall_material"] = gL.wall_material; d["planes"] = gL.planes; d["e_cut"] = gL.ecut;
  d["yoke_b"] = gL.yoke_b; d["field"] = gField.nx >= 2;
  return d;
}

}  // namespace

PYBIND11_MODULE(g4vessel, mod) {
  mod.def("initialize", &initialize, py::arg("seed"), py::arg("z_world0"), py::arg("z_world1"), py::arg("z_vessel0"),
          py::arg("z_vessel1"), py::arg("hx0"), py::arg("hx1"), py::arg("hy0"), py::arg("hy1"),
          py::arg("wall_thickness"), py::arg("wall_material"), py::arg("z_planes"), py::arg("e_cut"),
          py::arg("yoke_b"), py::arg("field"), py::arg("x0") = 0.0, py::arg("dx") = 1.0, py::arg("y0") = 0.0,
          py::arg("dy") = 1.0, py::arg("z0") = 0.0, py::arg("dz") = 1.0);
  mod.def("transport", &transport, py::arg("offsets"), py::arg("pdg"), py::arg("px"), py::arg("py"), py::arg("pz"),
          py::arg("vx"), py::arg("vy"), py::arg("vz"));
  mod.def("layout", &layout);
}
