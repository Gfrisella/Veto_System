#!/usr/bin/env python3
"""MS7 last-50-cm muon-interaction chain (bin/shield_dis.py outputs) for several DIS generators, side by side with the
FairShip TRY_2026_EM_SST1_origin full-physics numbers (fairship_sst1_origin/README.md, tracks made in the MS7 last 50 cm).

    python3 bin/compare_shield_generators.py label=results/shield_dis/shield_muon_X.npz [...] [--out results/shield_dis/generators]

Per file (rates per spill = Hz at 5e13 PoT/s):
  interactions, what leaves the iron (p > 1 GeV, from the .json), V0 decays in the vessel (forced copies, .json);
  charged tracks at SST1 as in the FairShip comparison: straight line from their start point to z = 82 m,
  abs x < 2, abs y < 3, all p: pi, p, K, e, mu (code 3 = the interactions), "K0-decay tracks" = charged daughters of
  the forced vessel decays (code 4 rows with start z > plane + 1 cm), K_L / K_S separately.
  "trk" rows (added 2026-11-09): the K0-decay tracks with |p| > 1 GeV seen in >= 3 of the 4 straw stations
  (station seen = >= 1 straw hit; lib/veto/straws.py layer_states + digitize + hit_summary: FairShip strawtubes
  geometry, RK4 in the spectrometer field map, yoke losses; no scattering / fit),
  independent of the straight-line SST1 cut above; "K0 decays, 2 trk" = forced decays with >= 2 such tracks.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from veto import geometry as geo, spectrometer, straws

P_MIN, N_STATIONS_MIN = 1.0, 3


def stations_seen(v, p, q, seed=1):
    """Stations (of 4) with >= 1 straw hit: lib/veto/straws.py as implemented (32 layers, FairShip strawtubes geometry,
    RK4 in the spectrometer field map MainSpectrometerField.h5 at geometry.Z_MAGNET, yoke losses, digitize)."""
    states, _, lost = straws.layer_states(v, p, q)
    h = straws.digitize(states, np.random.default_rng(seed), lost)
    return straws.hit_summary(h, len(v))[1]

Z_SST1, AX, AY, Z_PLANE = 82.0, 2.0, 3.0, 29.90
FAIRSHIP = {"pi": 800.0, "p": 0.0, "K0-decay tracks": 0.0}      # Hz, cut 1 (pi: 1 MC, w = 769; K0: 0 decay tracks,
# < 18 Hz 90 % CL; the 8 Hz entry is a pi from a K_L hadronic interaction in the cavern, 2026-11-09)
CHARGED = {"pi": [211], "p": [2212], "K": [321], "e": [11], "mu": [13]}


def analyse(path):
    z = np.load(path); js = json.load(open(Path(path).with_suffix(".json")))
    w_int = z["int_weight"]; code = z["int_code"]; v0 = z["int_v0"]
    pi, pdg, p, v = z["part_int"], z["part_pdg"], z["part_p"], z["part_v"]
    w = w_int[pi]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (Z_SST1 - v[:, 2]) / p[:, 2]
        x, y = v[:, 0] + p[:, 0] * t, v[:, 1] + p[:, 1] * t
    acc = (p[:, 2] > 0) & (np.abs(x) < AX) & (np.abs(y) < AY)
    c3 = code[pi] == 3
    dec = (code[pi] == 4) & (v[:, 2] > Z_PLANE + 0.01)
    r = {"split": js.get("mu_split", "not recorded"), "interactions": js["interactions_per_spill"],
         "draws": js["draws"]}
    for k in ("K_L", "K_S", "Lambda", "pi+-", "p", "n"):
        r[f"exit {k}"] = js["exit"][k]["per_spill"]
    d = js["v0_decays_in_vessel_per_spill"]
    r["K_L decays in vessel"] = d.get("130", 0.0); r["K_S decays in vessel"] = d.get("310", 0.0)
    for name, codes in CHARGED.items():
        m = c3 & acc & np.isin(np.abs(pdg), codes)
        r[f"SST1 {name}"] = float(w[m].sum()); r[f"SST1 {name} MC"] = int(m.sum())
    ch = np.isin(np.abs(pdg), sum(CHARGED.values(), []))
    for lab, sel in (("K0-decay tracks", np.isin(v0[pi], [130, 310])), ("K_L-decay tracks", v0[pi] == 130),
                     ("K_S-decay tracks", v0[pi] == 310)):
        m = dec & acc & ch & sel
        r[f"SST1 {lab}"] = float(w[m].sum()); r[f"SST1 {lab} MC"] = int(m.sum())
    # 3 of 4 straw stations + p > 1 GeV
    pabs = np.linalg.norm(p[:, :3], axis=1)
    cand = np.nonzero(dec & ch & np.isin(v0[pi], [130, 310]) & (pabs > P_MIN) & (p[:, 2] > 0)
                      & (v[:, 2] < straws.LAYER_Z[0]))[0]
    good = np.zeros(len(pdg), bool)
    if len(cand):
        q = spectrometer.charge_of_pdg(pdg[cand])
        good[cand] = stations_seen(v[cand].astype(float), p[cand, :3].astype(float), q) >= N_STATIONS_MIN
    for lab, sel in (("K0-decay trk", np.isin(v0[pi], [130, 310])), ("K_L-decay trk", v0[pi] == 130),
                     ("K_S-decay trk", v0[pi] == 310)):
        m = good & sel
        r[f"{lab} (3/4 st, p>1)"] = float(w[m].sum()); r[f"{lab} (3/4 st, p>1) MC"] = int(m.sum())
    ng = np.bincount(pi[good], minlength=len(code))
    two = ng >= 2
    r["K0 decays, 2 trk (3/4 st, p>1)"] = float(w_int[two].sum()); r["K0 decays, 2 trk (3/4 st, p>1) MC"] = int(two.sum())
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="label=path.npz")
    ap.add_argument("--out", default="results/shield_dis/generators")
    a = ap.parse_args()
    res = {lab: analyse(p) for lab, p in (f.split("=", 1) for f in a.files)}
    res["FairShip (full physics)"] = {f"SST1 {k}": v for k, v in FAIRSHIP.items()}
    json.dump(res, open(a.out + ".json", "w"), indent=1)
    keys = [k for k in next(iter(res.values())) if not k.endswith(" MC")]
    labs = list(res)
    lines = ["| per spill | " + " | ".join(labs) + " |", "|---" * (len(labs) + 1) + "|"]
    for k in keys:
        cells = []
        for l in labs:
            v = res[l].get(k)
            mc = res[l].get(k + " MC")
            cells.append("" if v is None else (v if isinstance(v, str) else f"{v:.3g}" + (f" ({mc})" if mc is not None else "")))
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    txt = "\n".join(lines)
    open(a.out + ".md", "w").write(txt + "\n\n(n) = MC entries. FairShip: TRY_2026_EM_SST1_origin, tracks made in MS7 last 50 cm, cut 1.\n")
    print(txt)


if __name__ == "__main__":
    main()
