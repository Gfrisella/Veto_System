#!/usr/bin/env python3
"""V0 production in the MS7 muon DIS chain: every V0 made at the interaction (also the ones that never reach the
vessel), compared with measured strange-particle yields and with what Geant4 brings to the shield exit plane.

Reads existing files only (no generation, no Geant4):
  - data/interactions/dis/mu_library_iron_v0stable.npz: the same Geant4 11.2.2 FTFP_BERT muonNuclear (and Pythia 6)
    final states as mu_library_iron.npz, but K_S / Lambda not decayed -> what the vertex makes. The default library
    (used by the PR #15 chain) decays K_S / Lambda at the vertex (ISSUES D9); its K_L is used as a cross-check.
  - the PR #15 MS7 runs shield_muon_g4v112_s1??.npz (+ .json): energy spectrum of the interactions at the vertex,
    V0 at the exit plane z = 29.90 m vs the vertex depth.

    python3 bin/v0_production.py [--runs 'glob'] [--out results/v0_production]

Outputs: v0_production.json (all numbers), v0_*.png / .pdf, README in the output directory (written by hand).
"""

import _bootstrap  # noqa: F401
import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from veto import paths
from veto.config import SPILLS_15Y

LIB = paths.PROJECT_DIR / "data/interactions/dis/mu_library_iron_v0stable.npz"
LIB_DEF = paths.PROJECT_DIR / "data/interactions/dis/mu_library_iron.npz"
RUNS = "/disk/lhcb_data/sqasim/workspace/uzh/Veto_System/veto/results/shield_dis/shield_muon_g4v112_s1??.npz"
Z_END = 29.89
F_T = 0.82          # fraction of the vertex K_L with T > 0.5 GeV (the exit-plane cut) at the MS7 energies
M_N, M_MU = 0.938918, 0.105658
SPECIES = {"K_L": [130], "K_S": [310], "Lambda": [3122], "anti-Lambda": [-3122], "K+": [321], "K-": [-321],
           "pi+-": [211, -211], "p": [2212]}

# Measured yields per hadronic event (primary + feed-down from Sigma0 / K* etc., as published)
DATA = [
    dict(name="SLAC SHF gamma p, E_gamma = 20 GeV", W=6.17, K0S=9.447 / 115.0, K0S_err=0.318 / 115.0,
         Lam=5.600 / 115.0, Lam_err=0.244 / 115.0, aLam=0.329 / 115.0, aLam_err=0.040 / 115.0,
         ref="Abe et al., PRD 30 (1984) 1, SLAC-PUB-3162, Table II (nb) / sigma_tot(gamma p) = 115 ub"),
    dict(name="NOMAD nu_mu CC (C), <E_nu> = 45 GeV", W=None, K0S=0.0676, K0S_err=0.0006, Lam=0.0504, Lam_err=0.0006,
         aLam=0.0037, aLam_err=0.0002, ref="Astier et al., NPB 621 (2002) 3, hep-ex/0111057, Table 5 (primary V0)"),
]


def event_index(t):
    return np.repeat(np.arange(len(t["ev_n"])), t["ev_n"])


def load_part(z, pre):
    t = {k: z[f"{pre}_{k}"] for k in ("ev_ie", "ev_tgt", "ev_w", "ev_n", "pdg", "p")}
    return t


def kinematics(t, E_grid):
    """nu, Q2, W per event from the scattered muon (incoming mu- along +z)."""
    ev = event_index(t)
    E = E_grid[t["ev_ie"].astype(int)]
    mu = np.nonzero(np.abs(t["pdg"]) == 13)[0]
    first = np.full(len(E), -1)
    first[ev[mu][::-1]] = mu[::-1]                      # first muon of each event
    ok = first >= 0
    p = t["p"][first[ok]].astype(np.float64)
    Ep = p[:, 3]
    pz, pp = p[:, 2], np.linalg.norm(p[:, :3], axis=1)
    E0 = E[ok]
    P0 = np.sqrt(np.maximum(E0 ** 2 - M_MU ** 2, 0))
    nu = np.full(len(E), np.nan)
    Q2 = np.full(len(E), np.nan)
    nu[ok] = E0 - Ep
    Q2[ok] = 2 * (E0 * Ep - P0 * pz) - 2 * M_MU ** 2
    W2 = M_N ** 2 + 2 * M_N * nu - Q2
    return nu, Q2, np.sqrt(np.maximum(W2, 0)), E


def counts(t):
    ev = event_index(t)
    n = len(t["ev_n"])
    return {s: np.bincount(ev[np.isin(t["pdg"], c)], minlength=n).astype(np.float64) for s, c in SPECIES.items()}


def per_energy(cnt, ie, nE):
    out = {}
    nev = np.bincount(ie, minlength=nE).astype(float)
    for s, c in cnt.items():
        out[s] = np.divide(np.bincount(ie, c, minlength=nE), nev, out=np.zeros(nE), where=nev > 0)
    return out


def w_bins(W, cnt, edges):
    i = np.digitize(W, edges) - 1
    nb = len(edges) - 1
    ok = (i >= 0) & (i < nb)
    nev = np.bincount(i[ok], minlength=nb).astype(float)
    res = {"n_events": nev}
    for s, c in cnt.items():
        m = np.bincount(i[ok], c[ok], minlength=nb)
        res[s] = np.divide(m, nev, out=np.full(nb, np.nan), where=nev > 0)
        res[s + "_err"] = np.divide(np.sqrt(np.bincount(i[ok], c[ok] ** 2, minlength=nb)), nev, out=np.full(nb, np.nan),
                                    where=nev > 0)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--out", default=str(paths.results_dir() / "v0_production"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R = {}

    # ---------------------------------------------------------------- 1. generator level (library)
    z = np.load(LIB)
    E_grid = z["E_grid"]
    nE = len(E_grid)
    parts = {"geant4": load_part(z, "g4_iron"), "pythia6": load_part(z, "p6")}
    zd = np.load(LIB_DEF)
    kl_def = (np.bincount(event_index(load_part(zd, "g4_iron")), (zd["g4_iron_pdg"] == 130).astype(float),
                          minlength=len(zd["g4_iron_ev_n"]))).mean()
    lib = {}
    edges = np.array([0.9, 1.1, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0])
    for name, t in parts.items():
        nu, Q2, W, E = kinematics(t, E_grid)
        cnt = counts(t)
        ie = t["ev_ie"].astype(int)
        lib[name] = dict(t=t, W=W, Q2=Q2, nu=nu, E=E, cnt=cnt, perE=per_energy(cnt, ie, nE), Wb=w_bins(W, cnt, edges))
        R[f"library_{name}"] = {
            "events": int(len(t["ev_n"])),
            "mean_per_event_flat_in_E": {s: float(c.mean()) for s, c in cnt.items()},
            "W_quantiles_5_50_95": [float(x) for x in np.nanpercentile(W, [5, 50, 95])],
            "W_check_vs_stored_ev_w_median_absdiff": float(np.nanmedian(np.abs(W - t["ev_w"]))),
            "W_bins": {"edges": edges.tolist(), **{k: np.asarray(v).tolist() for k, v in lib[name]["Wb"].items()}},
        }
    R["library_geant4"]["K_L_default_library"] = float(kl_def)

    # ---------------------------------------------------------------- 2. MS7 interactions (PR #15 runs)
    files = sorted(glob.glob(a.runs))
    if not files:
        raise SystemExit(f"no runs match {a.runs}")
    Eint, Wint, zint = [], [], []
    plane = {s: [] for s in ("K_L", "K_S", "Lambda", "anti-Lambda", "K+-")}
    exit_json = []
    for f in files:
        r = np.load(f)
        code = r["int_code"]
        m3 = np.nonzero(code == 3)[0]
        Eint.append(r["int_E"][m3])
        Wint.append(r["int_weight"][m3] / len(files))
        zint.append(r["int_z"][m3])
        pint, ppdg = r["part_int"], r["part_pdg"]
        keep = pint < len(m3)          # code-3 interactions come first (copies appended by force_v0)
        assert np.all(code[:len(m3)] == 3)
        for s, c in (("K_L", [130]), ("K_S", [310]), ("Lambda", [3122]), ("anti-Lambda", [-3122]), ("K+-", [321, -321])):
            sel = keep & np.isin(ppdg, c)
            plane[s].append(np.bincount(pint[sel], minlength=len(m3)).astype(float))
        exit_json.append(json.load(open(f.replace(".npz", ".json")))["exit"])
        print(f"read {f}", flush=True)
    off = np.cumsum([0] + [len(x) for x in Eint])
    Eint, Wint, zint = (np.concatenate(x) for x in (Eint, Wint, zint))
    plane = {s: np.concatenate(v) for s, v in plane.items()}
    S = Wint.sum()                                           # interactions per spill
    depth = Z_END - zint
    R["ms7"] = {"files": files, "interactions_per_spill": float(S), "interactions_15y": float(S * SPILLS_15Y),
                "E_vertex_quantiles_5_50_95": [float(x) for x in np.percentile(Eint, [5, 50, 95])]}

    # V0 made per spill: library <n>(E) folded with the vertex-energy spectrum (linear interpolation in log E)
    def fold(perE):
        v = np.interp(np.log(np.clip(Eint, E_grid[0], E_grid[-1])), np.log(E_grid), perE)
        return float((Wint * v).sum())

    made = {}
    for name in ("geant4", "pythia6"):
        made[name] = {s: fold(lib[name]["perE"][s]) for s in SPECIES}
    R["ms7"]["made_per_spill"] = made
    R["ms7"]["made_per_interaction"] = {n: {s: v / S for s, v in d.items()} for n, d in made.items()}
    R["ms7"]["made_15y"] = {n: {s: v * SPILLS_15Y for s, v in d.items()} for n, d in made.items()}
    at_plane = {s: float((Wint * plane[s]).sum()) for s in plane}
    R["ms7"]["at_plane_per_spill_code3"] = at_plane
    R["ms7"]["exit_json_mean_per_spill"] = {k: float(np.mean([e[k]["per_spill"] for e in exit_json]))
                                            for k in exit_json[0]}

    # K_L at the plane vs vertex depth -> effective attenuation length, intercept = made at the vertex
    dz = np.linspace(0, 0.5, 11)
    i = np.clip(np.digitize(depth, dz) - 1, 0, len(dz) - 2)
    prof = {}
    for s in ("K_L", "K_S", "Lambda", "K+-"):
        num = np.bincount(i, Wint * plane[s], minlength=len(dz) - 1)
        num2 = np.bincount(i, Wint ** 2 * plane[s] ** 2, minlength=len(dz) - 1)
        den = np.bincount(i, Wint, minlength=len(dz) - 1)
        prof[s] = (num / den, np.sqrt(num2) / den)
    dc = 0.5 * (dz[1:] + dz[:-1])
    y, e = prof["K_L"]
    good = y > 0
    cf = np.polyfit(dc[good], np.log(y[good]), 1, w=y[good] / e[good])
    lam_eff = -1 / cf[0]
    R["ms7"]["K_L_vs_depth"] = {"depth_centres_m": dc.tolist(), **{s: v[0].tolist() for s, v in prof.items()},
                                "K_L_err": e.tolist(), "fit_lambda_eff_m": float(lam_eff),
                                "fit_intercept_per_interaction": float(np.exp(cf[1]))}
    # expectation for primary K_L only: exp(-d / lambda_I), lambda(K Fe) ~ 0.19 - 0.22 m (sigma_inel(K Fe) ~ 0.6 b)
    kl_made = made["geant4"]["K_L"] / S
    surv = {f"{lam:.2f}": float((Wint * np.exp(-depth / lam)).sum() / S) for lam in (0.168, 0.19, 0.22)}
    R["ms7"]["K_L_primary_survival_expected"] = surv
    R["ms7"]["K_L_plane_over_made"] = at_plane["K_L"] / made["geant4"]["K_L"]
    prim = {k: F_T * made["geant4"]["K_L"] * v for k, v in surv.items()}
    R["ms7"]["K_L_primary_at_plane_expected_per_spill"] = prim
    R["ms7"]["K_L_secondary_fraction_at_plane"] = {k: 1 - v / at_plane["K_L"] for k, v in prim.items()}
    R["ms7"]["K_L_vessel_decay_probability"] = 1540.2 / at_plane["K_L"]   # PR #15 s101 json v0_decays_in_vessel

    # ---------------------------------------------------------------- 3. comparison with data
    g, p6 = lib["geant4"], lib["pythia6"]
    cmp_rows = []
    for d in DATA:
        row = dict(d)
        for name, L in (("geant4", g), ("pythia6", p6)):
            if d["W"] is None:
                continue
            m = np.abs(L["W"] - d["W"]) < 0.5
            row[f"{name}_n_events_W+-0.5"] = int(m.sum())
            for s, k in (("K_S", "K0S"), ("Lambda", "Lam"), ("anti-Lambda", "aLam")):
                row[f"{name}_{k}"] = float(L["cnt"][s][m].mean()) if m.sum() else None
        cmp_rows.append(row)
    R["data_comparison"] = cmp_rows

    # ---------------------------------------------------------------- plots
    wc = 0.5 * (edges[1:] + edges[:-1])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, L, ls in (("Geant4 11.2 muonNuclear (Fe)", g, "-"), ("Pythia 6 (p, n)", p6, "--")):
        for s, col in (("K_S", "C0"), ("Lambda", "C1"), ("anti-Lambda", "C2")):
            yv = L["Wb"][s]
            ax[0].errorbar(wc, yv, L["Wb"][s + "_err"], ls=ls, color=col, marker="o" if ls == "-" else "s", ms=3,
                           label=f"{s}, {name}")
    for d, mk in zip(DATA, ("*", "D")):
        if d["W"] is None:
            continue
        for k, col in (("K0S", "C0"), ("Lam", "C1"), ("aLam", "C2")):
            ax[0].errorbar([d["W"]], [d[k]], [d[k + "_err"]], color=col, marker=mk, ms=12, mec="k", ls="none")
    ax[0].plot([], [], "k*", ms=10, label="data: gamma p 20 GeV (SLAC)")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("W [GeV]")
    ax[0].set_ylabel("mean multiplicity per interaction")
    ax[0].set_ylim(1e-4, 1)
    ax[0].legend(fontsize=6.5)
    ax[0].set_title("V0 at the vertex vs W")
    hW = np.logspace(np.log10(0.9), np.log10(30), 60)
    ax[1].hist(g["W"], hW, density=True, histtype="step", label="Geant4 muonNuclear (library, flat in log E)")
    ax[1].hist(p6["W"], hW, density=True, histtype="step", label="Pythia 6 (library)")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("W [GeV]")
    ax[1].set_ylabel("events (norm.)")
    ax[1].legend(fontsize=7)
    ax[1].set_title("W of the library events")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"v0_vs_W.{ext}", dpi=140)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for s, col in (("K_L", "C0"), ("K_S", "C1"), ("Lambda", "C2")):
        ax[0].plot(E_grid, g["perE"][s], color=col, label=f"{s} Geant4")
        ax[0].plot(E_grid, p6["perE"][s], color=col, ls="--", label=f"{s} Pythia 6")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("muon energy at the vertex [GeV]")
    ax[0].set_ylabel("V0 per interaction")
    ax2 = ax[0].twinx()
    ax2.hist(Eint, np.logspace(np.log10(0.5), np.log10(500), 40), weights=Wint, histtype="step", color="grey")
    ax2.set_ylabel("MS7 interactions / spill / bin", color="grey")
    ax[0].legend(fontsize=7, loc="upper left")
    ax[0].set_title("V0 per interaction vs E (grey: MS7 spectrum)")
    y, e = prof["K_L"]
    ax[1].errorbar(dc * 100, y, e, marker="o", label="K_L at the exit plane (Geant4 transport)")
    ax[1].plot(dc * 100, np.exp(np.polyval(cf, dc)), "C0--", label=f"fit: lambda_eff = {lam_eff * 100:.0f} cm")
    for lam, ls in ((0.19, ":"), (0.22, "-.")):
        ax[1].plot(dc * 100, F_T * kl_made * np.exp(-dc / lam), "k", ls=ls,
                   label=f"made x {F_T} (T > 0.5 GeV) x exp(-d/{lam * 100:.0f} cm): primaries only")
    for s, col in (("K_S", "C1"), ("Lambda", "C2")):
        ax[1].errorbar(dc * 100, prof[s][0], prof[s][1], marker="s", color=col, ms=3, label=f"{s} at the exit plane")
    ax[1].axhline(kl_made, color="C0", lw=0.8, alpha=0.5)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("vertex depth before the shield end [cm]")
    ax[1].set_ylabel("per interaction")
    ax[1].legend(fontsize=6.5)
    ax[1].set_title("what reaches z = 29.90 m")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"v0_ms7_fate.{ext}", dpi=140)

    json.dump(R, open(out / "v0_production.json", "w"), indent=1, default=float)
    print(json.dumps({k: R[k] for k in ("ms7", "data_comparison")}, indent=1, default=float)[:6000])
    for name in ("library_geant4", "library_pythia6"):
        print(name, R[name]["mean_per_event_flat_in_E"], R[name]["W_quantiles_5_50_95"],
              "W check", R[name]["W_check_vs_stored_ev_w_median_absdiff"])


if __name__ == "__main__":
    main()
