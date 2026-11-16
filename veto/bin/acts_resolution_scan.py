#!/usr/bin/env python3
"""ISSUES D14 (2): resolution / efficiency of the straw spectrometer with the FairShip ACTS fit (PR #1545 build,
external/fairship_acts), from a particle-gun scan, as input for reco.measure (not wired in).

Chain per momentum point (bin/batch/acts_scan.sbatch, FairShip pixi 26.10 via bin/fairship_env.sh):
  run_simScript.py PG: mu+ / mu- (--bothCharges) at fixed p, vertex uniform in |x| < 1.5 m, |y| < 2.5 m at FairShip
  z = 83.35 m (our 81.21 m: just behind the vessel exit window, so the window and the vessel are NOT in the sample; our
  scattering.py handles the material upstream of SST1), polar angle 0 - 1.5 deg, full FairShip 26.10 geometry and field
  (Geant4 11.4 transport, MainSpectrometerField V21);
  ShipReco.py --trackFitter acts --patRec Truth (Kalman fit, state at the first measurement, ~FS 84.03 m = our 81.89 m).

  extract   (pixi python)  sim + rec files of one point -> npz: per fitted track the residuals at the fit z (truth from
                           the matched MC track's straw point nearest to it, carried straight to the fit z), the fit
                           errors and chi2; per generated muon the stations hit and whether it was fitted.
  fit       (any python)   all points -> resolution table (core sigma = half the central 68.3 % interval, rms, tail
                           fraction), fit efficiency, fitted forms, data/acts/resolution_param.json + printed table.

Frames: FairShip z = our z + 2.14 m; x, y identical. Units in the outputs: m, rad, GeV.
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

FS_OFFSET = 2.14


def extract(simdir, out):
    import ROOT
    import uproot
    ROOT.gSystem.Load("libShipData")
    sim = [f for f in glob.glob(f"{simdir}/sim_*.root") if not f.endswith("_rec.root")][0]
    rec = sim.replace(".root", "_rec.root")
    s = uproot.open(sim)["cbmsim"]
    P = "strawtubesPoint/strawtubesPoint."
    pts = {k: s[P + k].array(library="np") for k in ("fTrackID", "fDetectorID", "fX", "fY", "fZ", "fPx", "fPy", "fPz")}
    mom = s["MCTrack/MCTrack.fMotherId"].array(library="np")
    mpx, mpy, mpz = (s[f"MCTrack/MCTrack.f{c}"].array(library="np") for c in ("Px", "Py", "Pz"))
    pdg = s["MCTrack/MCTrack.fPdgCode"].array(library="np")
    f = ROOT.TFile.Open(rec)
    t = f.Get("ship_reco_sim")
    rows, gen = [], []
    for ev in range(t.GetEntries()):
        t.GetEntry(ev)
        fitted_mc = {}
        for i, tr in enumerate(t.RecoTracks):
            mc = int(t.fitTrack2MC[i])
            fitted_mc[mc] = tr
        tid, did, z = pts["fTrackID"][ev], pts["fDetectorID"][ev], pts["fZ"][ev]
        for j in np.nonzero(mom[ev] == -1)[0]:                      # primary muon(s)
            on = tid == j
            st = np.unique(did[on] // 1000000)
            p_true = float(np.sqrt(mpx[ev][j] ** 2 + mpy[ev][j] ** 2 + mpz[ev][j] ** 2))
            tr = fitted_mc.get(int(j))
            gen.append((p_true, int(pdg[ev][j]), int(1 in st), int(len(set(st) & {1, 2, 3, 4}) == 4), int(tr is not None),
                        float(np.hypot(mpx[ev][j], mpy[ev][j]) / mpz[ev][j])))
            if tr is None or not on.any():
                continue
            zf = tr.z()
            k = np.nonzero(on)[0][np.argmin(np.abs(z[on] - zf))]
            tx_t = pts["fPx"][ev][k] / pts["fPz"][ev][k]; ty_t = pts["fPy"][ev][k] / pts["fPz"][ev][k]
            x_t = pts["fX"][ev][k] + tx_t * (zf - z[k]); y_t = pts["fY"][ev][k] + ty_t * (zf - z[k])
            p_pt = float(np.sqrt(pts["fPx"][ev][k] ** 2 + pts["fPy"][ev][k] ** 2 + pts["fPz"][ev][k] ** 2))
            pf = np.sqrt(tr.px() ** 2 + tr.py() ** 2 + tr.pz() ** 2)
            cov = list(tr.GetCovarianceElements())
            rows.append((p_true, p_pt, pf, tr.charge(), int(np.sign(pdg[ev][j])),
                         (tr.x() - x_t) / 100, (tr.y() - y_t) / 100,
                         tr.px() / tr.pz() - tx_t, tr.py() / tr.pz() - ty_t,
                         tr.err_x() / 100, tr.err_y() / 100, tr.chi2(), tr.nDoF(), tr.nMeasurements(), zf / 100 - FS_OFFSET,
                         float(np.hypot(tx_t, ty_t))))
    names = ("p_true", "p_at_point", "p_fit", "q_fit", "pdg_sign", "dx", "dy", "dtx", "dty", "err_x", "err_y", "chi2", "ndf",
             "n_meas", "z_fit_ours", "slope")
    r = np.array(rows, float).reshape(-1, len(names))
    g = np.array(gen, float).reshape(-1, 6)
    np.savez(out, **{n: r[:, i] for i, n in enumerate(names)},
             gen_p=g[:, 0], gen_pdg=g[:, 1], gen_sst1=g[:, 2], gen_all4=g[:, 3], gen_fitted=g[:, 4], gen_slope=g[:, 5])
    print(f"{out}: {len(r)} fitted tracks, {len(g)} generated muons")


def core(v):
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return np.nan, np.nan, np.nan
    lo, hi = np.quantile(v, [0.1587, 0.8413])
    c = 0.5 * (hi - lo)
    return float(c), float(np.std(v)), float(np.mean(np.abs(v - np.median(v)) > 5 * c))


def quad_fit(p, s, forms):
    """Least squares in s^2 for s^2 = sum_i a_i^2 f_i(p)^2 (each term non-negative); returns the a_i."""
    from scipy.optimize import nnls
    A = np.stack([fn(p) ** 2 for fn in forms], 1) / (s[:, None] ** 2)
    b = np.ones(len(p))
    a2, _ = nnls(A, b)
    return np.sqrt(a2)


def fit(files, out_json):
    pts = []
    for fn in sorted(files):
        d = dict(np.load(fn))
        if not len(d["gen_p"]):
            continue
        p0 = float(np.median(d["gen_p"]))
        res = dict(p=p0, n_gen=int(len(d["gen_p"])), n_fit=int(len(d["p_true"])))
        sst1, all4, fitted = d["gen_sst1"] > 0, d["gen_all4"] > 0, d["gen_fitted"] > 0
        res["eff_fit_given_all4"] = float((fitted & all4).sum() / max(all4.sum(), 1))
        res["eff_fit_given_sst1"] = float((fitted & sst1).sum() / max(sst1.sum(), 1))
        res["frac_all4_given_sst1"] = float((all4 & sst1).sum() / max(sst1.sum(), 1))
        dp = (d["p_fit"] - d["p_at_point"]) / d["p_at_point"]
        for name, v in (("dp_over_p", dp), ("x_m", d["dx"]), ("y_m", d["dy"]), ("tx_rad", d["dtx"]), ("ty_rad", d["dty"])):
            c, rms, tail = core(v)
            res[name] = dict(core=c, rms=rms, tail_5core=tail)
        res["charge_misid"] = float(np.mean(d["q_fit"] != d["pdg_sign"] * -1)) if len(d["q_fit"]) else np.nan  # mu- pdg 13
        res["chi2_ndf_median"] = float(np.median(d["chi2"] / np.maximum(d["ndf"], 1))) if len(d["chi2"]) else np.nan
        res["n_meas_median"] = float(np.median(d["n_meas"])) if len(d["n_meas"]) else np.nan
        res["z_fit_ours_median"] = float(np.median(d["z_fit_ours"])) if len(d["z_fit_ours"]) else np.nan
        sl = d["slope"]
        if len(sl) > 40:                                         # angle dependence: two halves in slope
            m = sl < np.median(sl)
            res["dp_over_p_core_low_slope"] = core(dp[m])[0]; res["dp_over_p_core_high_slope"] = core(dp[~m])[0]
            res["tx_core_low_slope"] = core(d["dtx"][m])[0]; res["tx_core_high_slope"] = core(d["dtx"][~m])[0]
        pts.append(res)
    pts.sort(key=lambda r: r["p"])
    p = np.array([r["p"] for r in pts])
    ok = lambda k: np.array([r[k]["core"] for r in pts])
    fits = {}
    sp = ok("dp_over_p"); m = np.isfinite(sp)
    a, b = quad_fit(p[m], sp[m], [lambda q: np.ones_like(q), lambda q: q])
    fits["dp_over_p"] = dict(form="sqrt(a^2 + (b p)^2), p in GeV", a=float(a), b=float(b))
    for k in ("tx_rad", "ty_rad", "x_m", "y_m"):              # x: non-bending, y: bending (our field is Bx)
        s = ok(k); m = np.isfinite(s)
        c, d_ = quad_fit(p[m], s[m], [lambda q: np.ones_like(q), lambda q: 1 / q])
        unit = "rad" if k.startswith("t") else "m"
        fits[k] = dict(form=f"sqrt(c^2 + (d / p)^2) [{unit}], p in GeV", c=float(c), d=float(d_),
                       max_rel_dev=float(np.max(np.abs(np.hypot(c, d_ / p[m]) / s[m] - 1))))
    fits["dp_over_p"]["max_rel_dev"] = float(np.max(np.abs(np.hypot(a, b * p) / sp - 1)[np.isfinite(sp)]))
    out = dict(description="ISSUES D14 (2): FairShip 26.10 ACTS Kalman fit (PR #1545 straw material), particle gun mu+/- from "
               "FS z = 83.35 m (our 81.21), |x| < 1.5 m, |y| < 2.5 m, theta 0-1.5 deg; state at the first measurement "
               "(our z ~ 81.89 m). Core = half the central 68.3 % interval. Truth = matched MC straw point carried straight "
               "to the fit z. Time: not provided by the fit (RecoTrack has no time).",
               units=dict(p="GeV", x="m", slope="rad"), points=pts, fits=fits,
               current_reco=dict(sigma_pos_m=1e-3, sigma_slope_rad=1e-4, ms_slope_rad_GeV=2e-3, sigma_t_ns=0.1))
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_json, "w"), indent=1)
    print(f"{'p':>6} {'n_gen':>6} {'eff|4st':>8} {'eff|SST1':>8} {'dp/p core':>10} {'rms':>8} {'tail':>6} {'x core um':>10} "
          f"{'y core um':>10} {'tx core urad':>12} {'ty core urad':>12} {'q misid':>8} {'chi2/ndf':>8}")
    for r in pts:
        print(f"{r['p']:>6.0f} {r['n_gen']:>6} {r['eff_fit_given_all4']:>8.4f} {r['eff_fit_given_sst1']:>8.4f} "
              f"{100 * r['dp_over_p']['core']:>9.3f}% {100 * r['dp_over_p']['rms']:>7.2f}% {100 * r['dp_over_p']['tail_5core']:>5.1f}% "
              f"{1e6 * r['x_m']['core']:>10.0f} {1e6 * r['y_m']['core']:>10.0f} {1e6 * r['tx_rad']['core']:>12.1f} {1e6 * r['ty_rad']['core']:>12.1f} {r['charge_misid']:>8.4f} {r['chi2_ndf_median']:>8.2f}")
    print(json.dumps(fits, indent=1))
    print(f"wrote {out_json}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("simdir"); e.add_argument("--out", required=True)
    f = sub.add_parser("fit"); f.add_argument("files", nargs="+"); f.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.simdir, a.out)
    else:
        fit(a.files, a.out)


if __name__ == "__main__":
    main()
