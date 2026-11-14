#!/usr/bin/env python3
"""Low-W check (W = 1.6 - 3.5 GeV, where most Geant4 muonNuclear events are) of strangeness production in the MS7
muon library against real-photon data on the proton (ISSUES D16 c).

Reference data: data/reference/lowW_strangeness_photoproduction.json (SAPHIR / CLAS exclusive K+Lambda, K+Sigma0,
K0Sigma+; ABBHHM lower limit on all strange channels; SLAC events with a visible strange decay; sigma_tot(gamma p)).
Simulation: data/interactions/dis/mu_library_iron_v0stable.npz (Geant4 11.2.2 muonNuclear on Fe, Pythia 6 on p / n).

Compared per event (per interaction):
  - strange fraction: events with >= 1 kaon (K+-, K_L, K_S) or anti-Lambda (every strange event has one, strangeness
    conservation) vs sigma(strange) / sigma_tot: data are LOWER LIMITS (exclusive sum, ABBHHM identified channels,
    SLAC visible decays);
  - Lambda (incl. Sigma0 -> Lambda gamma) per event vs (sigma(K+Lambda) + sigma(K+Sigma0)) / sigma_tot: the whole
    Lambda yield near threshold, a lower limit above W ~ 2.1 GeV (K Sigma* / multi-pion channels open).
Caveats: Fe (Fermi motion, nuclear rescattering) vs p; virtual (Q2 > 0) vs real photon; W from the scattered muon
with a nucleon at rest.

    python3 bin/v0_lowW.py [--out results/v0_production]
"""

import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from veto import paths

REF = paths.PROJECT_DIR / "data/reference/lowW_strangeness_photoproduction.json"
LIB = paths.PROJECT_DIR / "data/interactions/dis/mu_library_iron_v0stable.npz"
M_N, M_MU = 0.938918, 0.105658


def lib_events(z, pre):
    n = z[f"{pre}_ev_n"]
    ev = np.repeat(np.arange(len(n)), n)
    pdg = z[f"{pre}_pdg"]
    E = z["E_grid"][z[f"{pre}_ev_ie"].astype(int)]
    mu = np.nonzero(np.abs(pdg) == 13)[0]
    first = np.full(len(n), -1)
    first[ev[mu][::-1]] = mu[::-1]
    p = z[f"{pre}_p"][first].astype(float)
    P0 = np.sqrt(E ** 2 - M_MU ** 2)
    nu = E - p[:, 3]
    Q2 = 2 * (E * p[:, 3] - P0 * p[:, 2]) - 2 * M_MU ** 2
    W = np.sqrt(np.maximum(M_N ** 2 + 2 * M_N * nu - Q2, 0))
    W[first < 0] = np.nan
    cnt = lambda codes: np.bincount(ev[np.isin(pdg, codes)], minlength=len(n))   # noqa: E731
    strange = (cnt([321, -321, 130, 310, -3122]) > 0).astype(float)
    lam = cnt([3122]).astype(float)
    return W, Q2, strange, lam


def binned(W, y, edges):
    i = np.digitize(W, edges) - 1
    ok = (i >= 0) & (i < len(edges) - 1)
    n = np.bincount(i[ok], minlength=len(edges) - 1).astype(float)
    s = np.bincount(i[ok], y[ok], minlength=len(edges) - 1)
    m = np.divide(s, n, out=np.full(len(n), np.nan), where=n > 0)
    e = np.divide(np.sqrt(np.bincount(i[ok], y[ok] ** 2, minlength=len(edges) - 1) - s ** 2 / np.maximum(n, 1)),
                  n, out=np.full(len(n), np.nan), where=n > 1)
    return m, e, n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(paths.results_dir() / "v0_production"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = json.load(open(REF))["records"]

    def ch(name, src=None, kind=None):
        r = [x for x in rec if x["channel"] == name and (src is None or src in x["source"])
             and (kind is None or kind in x["kind"])]
        r.sort(key=lambda x: x["W_GeV"])
        return (np.array([x["W_GeV"] for x in r]), np.array([x["sigma_ub"] for x in r]),
                np.array([x["err_ub"] or 0 for x in r]))

    tot = [x for x in rec if x["channel"] == "gamma p -> X (total)"]
    tot.sort(key=lambda x: x["W_GeV"])
    Wt, st = np.array([x["W_GeV"] for x in tot]), np.array([x["sigma_ub"] for x in tot])
    # smooth sigma_tot: running median over 7 points (resonance region kept)
    st_s = np.array([np.median(st[max(0, i - 3):i + 4]) for i in range(len(st))])
    stot = lambda W: np.interp(W, Wt, st_s)                                       # noqa: E731

    z = np.load(LIB)
    edges = np.arange(1.5, 3.61, 0.1)
    wc = 0.5 * (edges[1:] + edges[:-1])
    sim = {}
    for lab, pre in (("Geant4 11.2 muonNuclear (Fe)", "g4_iron"), ("Pythia 6 (p, n)", "p6")):
        W, Q2, strange, lam = lib_events(z, pre)
        sim[lab] = dict(strange=binned(W, strange, edges), lam=binned(W, lam, edges),
                        Q2_median=binned(W, Q2, edges)[0])

    # data on the W grid of SAPHIR K+ Lambda
    WL, sL, eL = ch("gamma p -> K+ Lambda", "SAPHIR")
    WS, sS, eS = ch("gamma p -> K+ Sigma0", "SAPHIR")
    W0, s0, e0 = ch("gamma p -> K0 Sigma+", "SAPHIR")
    Wg = WL[(WL >= WS.min()) & (WL <= min(WS.max(), WL.max()))]
    sKL, sKS = np.interp(Wg, WL, sL), np.interp(Wg, WS, sS)
    sK0 = np.interp(Wg, W0, s0, left=0, right=0)
    eKY = np.hypot(np.interp(Wg, WL, eL), np.interp(Wg, WS, eS))
    lam_data = (sKL + sKS) / stot(Wg)
    ky_data = (sKL + sKS + sK0) / stot(Wg)
    WcL, scL, ecL = ch("gamma p -> K+ Lambda", "CLAS")
    WcS, scS, ecS = ch("gamma p -> K+ Sigma0", "CLAS")
    Wc = WcL[(WcL >= WcS.min()) & (WcL <= WcS.max())]
    lam_clas = (np.interp(Wc, WcL, scL) + np.interp(Wc, WcS, scS)) / stot(Wc)
    Wa, sa, ea = ch("gamma p -> strange X (LOWER LIMIT)")
    Wv, sv, ev = ch("gamma p -> events with VISIBLE strange-particle decay (topological)")

    R = {"W_edges": edges.tolist(), "sim": {}, "data": {
        "W_saphir": Wg.tolist(), "Lambda_per_event_KLambda_KSigma0": lam_data.tolist(),
        "strange_fraction_exclusive_KY": ky_data.tolist(),
        "W_clas": Wc.tolist(), "Lambda_per_event_clas_lower": lam_clas.tolist(),
        "W_abbhhm": Wa.tolist(), "strange_fraction_abbhhm_lower": (sa / stot(Wa)).tolist(),
        "W_slac_visible": Wv.tolist(), "strange_fraction_slac_visible_lower": (sv / stot(Wv)).tolist()}}
    for lab, s in sim.items():
        R["sim"][lab] = {"strange_fraction": s["strange"][0].tolist(), "strange_err": s["strange"][1].tolist(),
                         "Lambda_per_event": s["lam"][0].tolist(), "n_events": s["strange"][2].tolist(),
                         "Q2_median": s["Q2_median"].tolist()}
    json.dump(R, open(out / "v0_lowW.json", "w"), indent=1)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for (lab, s), mk in zip(sim.items(), ("o", "s")):
        ax[0].errorbar(wc, s["strange"][0], s["strange"][1], marker=mk, ms=3, label=lab)
        ax[1].errorbar(wc, s["lam"][0], s["lam"][1], marker=mk, ms=3, label=lab)
    ax[0].plot(Wg, ky_data, "k-", label="data: K+L + K+S0 + K0S+ (SAPHIR) / sigma_tot")
    ax[0].errorbar(Wa, sa / stot(Wa), ea / stot(Wa), fmt="kv", label="data: ABBHHM all identified (lower limit)")
    ax[0].errorbar(Wv, sv / stot(Wv), ev / stot(Wv), fmt="k^", mfc="w", label="data: SLAC visible strange decay (lower limit)")
    ax[1].errorbar(Wg, lam_data, eKY / stot(Wg), fmt="k.-", label="data: (K+L + K+S0) / sigma_tot, SAPHIR")
    ax[1].plot(Wc, lam_clas, "kD", mfc="w", ms=4, label="data: same, CLAS (measured angles only)")
    for x in ax:
        x.set_xlabel("W [GeV]")
        x.set_yscale("log")
        x.set_xlim(1.5, 3.6)
        x.legend(fontsize=6.5)
    ax[0].set_ylabel("fraction of events with strangeness")
    ax[0].set_ylim(1e-4, 0.3)
    ax[1].set_ylabel("Lambda (+Sigma0) per event")
    ax[1].set_ylim(1e-4, 0.1)
    ax[0].set_title("strangeness at low W: simulation vs gamma p data", fontsize=9)
    ax[1].set_title("Lambda per event; data = KY channels (all near threshold)", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"v0_lowW.{ext}", dpi=140)

    g = sim["Geant4 11.2 muonNuclear (Fe)"]
    print(" W    G4 strange  data KY   | G4 Lambda  data Lambda(KY)  | G4 n_ev  median Q2")
    for i, w in enumerate(wc):
        j = np.argmin(np.abs(Wg - w)) if len(Wg) else None
        dk = ky_data[j] if j is not None and abs(Wg[j] - w) < 0.06 else np.nan
        dl = lam_data[j] if j is not None and abs(Wg[j] - w) < 0.06 else np.nan
        print(f"{w:.2f}  {g['strange'][0][i]:.4f}    {dk:.4f}   | {g['lam'][0][i]:.4f}     {dl:.4f}         | "
              f"{int(g['strange'][2][i]):7d}  {g['Q2_median'][i]:.3f}")
    print("ABBHHM lower limit / sigma_tot:", np.round(sa / stot(Wa), 4).tolist(), "at W", np.round(Wa, 2).tolist())
    print("SLAC visible / sigma_tot:", np.round(sv / stot(Wv), 4).tolist(), "at W", np.round(Wv, 2).tolist())
    print(f"wrote {out / 'v0_lowW.json'}")


if __name__ == "__main__":
    main()
