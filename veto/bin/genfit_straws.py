#!/usr/bin/env python3
"""GenFit track fit of straw hits, as FairShip shipDigiReco.findTracks (run with the SHiP CVMFS ROOT / GenFit).

    bin/shipenv.sh python3 bin/genfit_straws.py <hits.npz> <out.npz>

Input (bin/straw_reco.py): hits of lib/veto/straws.digitize (track, station, detID, x1 y1 x2 y2 z, dist; cm) and the
spectrometer field map (field [T] (nx, ny, nz, 3), x0 dx y0 dy z0 dz [m], already placed in the veto frame).
As FairShip: pdg 13 hypothesis, tracks with >= 25 hits in >= 3 stations, genfit::RKTrackRep, seed (0, 0, vessel
centre) with p = (0, 0, 3 GeV) and the FairShip seed covariance, genfit::WireMeasurement (7 coordinates: wire end
points + drift distance; max distance = gas radius) per hit, genfit::DAF with 50 iterations. Material: air world
(genfit::TGeoMaterialInterface); field: the map, trilinear (kGauss for GenFit). Output per track: fitted (passed the
hit requirements), converged, chi2, ndf, n_hits, n_stations, p, charge, state at the first straw layer and at z_ref
(x, y, z [cm], px, py, pz [GeV]). FairShip "good track": converged and 0 <= chi2 / ndf < 50.
Only numpy + ROOT here (the SHiP python 3.9 has no torch): nothing from lib/veto is imported.
"""

import sys

import numpy as np
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gSystem.Load("libgenfit2")
ROOT.gInterpreter.Declare(r"""
#include <vector>
#include "AbsBField.h"
#include "TVector3.h"
class VetoMapField;
void veto_fill_field(VetoMapField& f, const float* p, size_t n);
class VetoMapField : public genfit::AbsBField {
 public:
  std::vector<float> b; int nx = 0, ny = 0, nz = 0;
  double x0 = 0, y0 = 0, z0 = 0, dx = 1, dy = 1, dz = 1;   // cm
  TVector3 get(const TVector3& p) const override { double bx, by, bz; get(p.X(), p.Y(), p.Z(), bx, by, bz); return TVector3(bx, by, bz); }
  void get(const double& x, const double& y, const double& z, double& Bx, double& By, double& Bz) const override {
    Bx = By = Bz = 0;
    const double fx = (x - x0) / dx, fy = (y - y0) / dy, fz = (z - z0) / dz;
    if (fx < 0 || fy < 0 || fz < 0 || fx >= nx - 1 || fy >= ny - 1 || fz >= nz - 1) return;
    const int i = int(fx), j = int(fy), k = int(fz);
    const double u = fx - i, v = fy - j, w = fz - k;
    double o[3] = {0, 0, 0};
    for (int di = 0; di < 2; ++di) for (int dj = 0; dj < 2; ++dj) for (int dk = 0; dk < 2; ++dk) {
      const double c = (di ? u : 1 - u) * (dj ? v : 1 - v) * (dk ? w : 1 - w);
      const size_t idx = ((size_t(i + di) * ny + (j + dj)) * nz + (k + dk)) * 3;
      for (int a = 0; a < 3; ++a) o[a] += c * b[idx + a];
    }
    Bx = 10 * o[0]; By = 10 * o[1]; Bz = 10 * o[2];      // T -> kGauss
  }
};
void veto_fill_field(VetoMapField& f, const float* p, size_t n) { f.b.assign(p, p + n); }
""")

R_GAS_CM = 1.0 - 0.0036
SIGMA_CM = 0.012


def build_world():
    geom = ROOT.TGeoManager("veto", "air world for GenFit")
    air = ROOT.TGeoMaterial("air", 14.61, 7.3, 0.001205)
    med = ROOT.TGeoMedium("air", 1, air)
    for o in (geom, air, med):                              # owned by the TGeoManager (Python must not delete them)
        ROOT.SetOwnership(o, False)
    top = geom.MakeBox("World", med, 1000.0, 1000.0, 10000.0)
    geom.SetTopVolume(top)
    geom.CloseGeometry()
    return geom


def main(hits_path, out_path):
    f = np.load(hits_path)
    h = {k: f[k] for k in ("track", "station", "detID", "x1", "y1", "x2", "y2", "z", "dist")}
    n_tracks = int(f["n_tracks"])
    z_seed, z_ref = float(f["z_seed_cm"]), float(f["z_ref_cm"])
    min_hits, min_stations = int(f["min_hits"]), int(f["min_stations"])

    field = ROOT.VetoMapField()
    grid = np.ascontiguousarray(f["field"], dtype=np.float32)
    field.nx, field.ny, field.nz = grid.shape[:3]
    for name in ("x0", "y0", "z0", "dx", "dy", "dz"):
        setattr(field, name, float(f[name]) * 100.0)       # m -> cm
    flat = grid.ravel()
    ROOT.veto_fill_field(field, flat, flat.size)
    ROOT.SetOwnership(field, False)                        # GenFit keeps raw pointers: C++ owns these
    ROOT.genfit.FieldManager.getInstance().init(field)
    geom = build_world()  # noqa: F841 (kept alive)
    mat_if = ROOT.genfit.TGeoMaterialInterface()
    ROOT.SetOwnership(mat_if, False)
    ROOT.genfit.MaterialEffects.getInstance().init(mat_if)
    fitter = ROOT.genfit.DAF()
    fitter.setMaxIterations(50)

    cols = ("fitted", "converged", "chi2", "ndf", "n_hits", "n_stations", "p", "charge",
            "x0", "y0", "z0", "px0", "py0", "pz0", "x_ref", "y_ref", "px_ref", "py_ref", "pz_ref")
    out = {c: np.full(n_tracks, np.nan) for c in cols}
    starts = np.searchsorted(h["track"], np.arange(n_tracks + 1))
    ref_plane = ROOT.genfit.SharedPlanePtr(ROOT.genfit.DetPlane(ROOT.TVector3(0, 0, z_ref), ROOT.TVector3(0, 0, 1)))
    for t in range(n_tracks):
        a, b = int(starts[t]), int(starts[t + 1])
        nM = b - a
        nst = len(np.unique(h["station"][a:b]))
        out["n_hits"][t], out["n_stations"][t], out["fitted"][t] = nM, nst, 0
        if nM < min_hits or nst < min_stations:
            continue
        res = SIGMA_CM
        covM = ROOT.TMatrixDSym(6)
        for i in range(3):
            covM[i][i] = res * res
        covM[0][0] = res * res * 100.0
        for i in range(3, 6):
            covM[i][i] = (res / nM / np.sqrt(3)) ** 2
        rep = ROOT.genfit.RKTrackRep(13)
        ROOT.SetOwnership(rep, False)                       # owned by the Track
        st = ROOT.genfit.MeasuredStateOnPlane(rep)
        rep.setPosMomCov(st, ROOT.TVector3(0, 0, z_seed), ROOT.TVector3(0, 0, 3.0), covM)
        seed, seedCov = ROOT.TVectorD(6), ROOT.TMatrixDSym(6)
        rep.get6DStateCov(st, seed, seedCov)
        track = ROOT.genfit.Track(rep, seed, seedCov)
        hitCov = ROOT.TMatrixDSym(7)
        hitCov[6][6] = res * res
        for k in range(a, b):
            m = ROOT.TVectorD(7)
            for i, v in enumerate((h["x1"][k], h["y1"][k], h["z"][k], h["x2"][k], h["y2"][k], h["z"][k], h["dist"][k])):
                m[i] = float(v)
            tp = ROOT.genfit.TrackPoint(track)
            ROOT.SetOwnership(tp, False)                    # owned by the Track
            meas = ROOT.genfit.WireMeasurement(m, hitCov, int(h["detID"][k]), int(k - a), tp)
            ROOT.SetOwnership(meas, False)                  # owned by the TrackPoint
            meas.setMaxDistance(R_GAS_CM)
            tp.addRawMeasurement(meas)
            track.insertPoint(tp)
        out["fitted"][t] = 1
        try:
            track.checkConsistency()
            fitter.processTrack(track)
            fs = track.getFitStatus()
            out["converged"][t] = float(fs.isFitConverged())
            out["chi2"][t], out["ndf"][t] = fs.getChi2(), fs.getNdf()
            fst = track.getFittedState()
            pos, mom = fst.getPos(), fst.getMom()
            out["x0"][t], out["y0"][t], out["z0"][t] = pos.X(), pos.Y(), pos.Z()
            out["px0"][t], out["py0"][t], out["pz0"][t] = mom.X(), mom.Y(), mom.Z()
            out["p"][t], out["charge"][t] = mom.Mag(), fst.getCharge()
            try:
                rst = ROOT.genfit.MeasuredStateOnPlane(fst)
                rst.extrapolateToPlane(ref_plane)
                rp, rm = rst.getPos(), rst.getMom()
                out["x_ref"][t], out["y_ref"][t] = rp.X(), rp.Y()
                out["px_ref"][t], out["py_ref"][t], out["pz_ref"][t] = rm.X(), rm.Y(), rm.Z()
            except Exception:
                pass
        except Exception as e:                               # genfit::Exception: counted as not converged
            out["converged"][t] = 0
        if (t + 1) % 500 == 0:
            print(f"  {t + 1}/{n_tracks} tracks", flush=True)
    np.savez(out_path, **out)
    ok = out["fitted"] == 1
    print(f"{n_tracks} tracks, {int(ok.sum())} fitted (>= {min_hits} hits, >= {min_stations} stations), "
          f"{int(np.nansum(out['converged']))} converged")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
