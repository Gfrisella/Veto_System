#!/usr/bin/env python3
"""What comes out behind a slab when a muon goes through it? Standalone full Geant4 (cpp/g4slab.cc, FTFP_BERT_EMZ).

Muons of fixed energy at normal incidence through: 1 m of iron (the end of the shield), 50 cm of concrete (a cavern
wall), 3 cm of steel (a vessel-type wall), 1 cm of plastic scintillator (a tile / fibre station) and one straw layer
(2 x 36 um Mylar + 1 cm argon). Every particle crossing planes 0, 1 and 5 m behind the slab is recorded; per
configuration, energy and plane: the probability that anything besides the muon arrives (charged / e+- / gamma above
thresholds), multiplicities, distance from the muon and time delay.

    bin/g4.sh python3 bin/em_debris_study.py --config iron_1m          # one configuration (one Geant4 per process)
    bin/g4.sh python3 bin/em_debris_study.py --summarize               # table + plots from all configurations
Writes results/em_debris/<config>.npz, summary.json, em_debris.png.
"""

import _bootstrap  # noqa: F401

import argparse
import json
import sys
import time

import numpy as np

from veto import paths

CONFIGS = {
    "iron_1m": ("iron 1 m (shield end)", [("G4_Fe", 1.0)], 20000),
    "concrete_50cm": ("concrete 50 cm (cavern wall)", [("G4_CONCRETE", 0.5)], 20000),
    "steel_3cm": ("steel 3 cm (vessel-type wall)", [("G4_STAINLESS-STEEL", 0.03)], 50000),
    "scint_1cm": ("plastic scintillator 1 cm (station)", [("G4_PLASTIC_SC_VINYLTOLUENE", 0.01)], 100000),
    "straw": ("straw layer (Mylar + 1 cm Ar)", [("G4_MYLAR", 36e-6), ("G4_Ar", 0.0099), ("G4_MYLAR", 36e-6)], 100000),
}
ENERGIES = (5.0, 10.0, 20.0, 50.0, 100.0, 300.0)                    # GeV, total energy at the slab
PLANES = (0.0, 1.0, 5.0)                                            # m behind the slab
KEEP = 1e-4                                                         # GeV: kinetic energy kept in the files
E_CH, E_G = 1e-3, 1e-4                                              # GeV: thresholds for charged / photons in the summary
OUT = paths.results_dir() / "em_debris"


def run(name, seed):
    sys.path.insert(0, str(paths.cpp_build()))
    import g4slab
    label, layers, n = CONFIGS[name]
    g4slab.initialize(seed, layers, list(PLANES), 0.7)
    out = {}
    for E in ENERGIES:
        t0 = time.time()
        d = g4slab.shoot(13, E, n, 0.0)
        keep = (d["track"] == 1) | (d["ekin"] > KEEP)
        for k, v in d.items():
            out.setdefault(k, []).append(np.asarray(v)[keep])
        out.setdefault("E", []).append(np.full(int(keep.sum()), E))
        print(f"{name} E = {E:5.0f} GeV: {n} muons, {int(keep.sum()):,} crossings kept, {time.time() - t0:.0f} s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{name}.npz", **{k: np.concatenate(v) for k, v in out.items()}, n=n, label=label)


def summarize():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = {}
    for name, (label, _, _) in CONFIGS.items():
        f = OUT / f"{name}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        n = int(d["n"])
        res[name] = {"label": label, "n": n, "points": []}
        for E in ENERGIES:
            for ip, dz in enumerate(PLANES):
                m = (d["E"] == E) & (d["plane"] == ip)
                mu = m & (d["track"] == 1)
                sec = m & (d["track"] != 1)
                ap = np.abs(d["pdg"])
                ch = sec & ((ap == 11) | (ap == 13) | (ap == 211) | (ap == 321) | (ap == 2212)) & (d["ekin"] > E_CH)
                ee = sec & (ap == 11) & (d["ekin"] > E_CH)
                gg = sec & (d["pdg"] == 22) & (d["ekin"] > E_G)
                nn = sec & (d["pdg"] == 2112)
                # distance from the muon in the same event and plane
                mux = dict(zip(d["ev"][mu], d["x"][mu]))
                muy = dict(zip(d["ev"][mu], d["y"][mu]))
                mut = dict(zip(d["ev"][mu], d["t"][mu]))
                evc = d["ev"][ch]
                has = np.array([e in mux for e in evc], bool)
                r = np.hypot(d["x"][ch][has] - np.array([mux[e] for e in evc[has]]), d["y"][ch][has] - np.array([muy[e] for e in evc[has]]))
                dt = d["t"][ch][has] - np.array([mut[e] for e in evc[has]])
                ev_ch = np.unique(evc)
                ev_far = np.unique(evc[has][r > 0.25]) if has.any() else np.array([])
                res[name]["points"].append(dict(
                    E=E, plane_m=dz, muons_arriving=int(mu.sum()),
                    p_any_charged=len(ev_ch) / n, p_charged_25cm=len(ev_far) / n,
                    n_charged=int(ch.sum()) / n, n_electrons=int(ee.sum()) / n, n_gamma=int(gg.sum()) / n, n_neutron=int(nn.sum()) / n,
                    r_median_m=float(np.median(r)) if len(r) else None, r_p90_m=float(np.percentile(r, 90)) if len(r) else None,
                    dt_p90_ns=float(np.percentile(dt, 90)) if len(dt) else None,
                    e_median_gev=float(np.median(d["ekin"][ee])) if ee.any() else None))
    json.dump(res, open(OUT / "summary.json", "w"), indent=1)
    print(f"{'configuration':38s}{'E':>6}{'plane':>7}{'P(charged)':>12}{'P(ch >25cm)':>13}{'e+-/mu':>9}{'gamma/mu':>10}{'n/mu':>8}{'r50 [cm]':>10}{'r90 [cm]':>10}")
    for name, v in res.items():
        for p in v["points"]:
            r50 = f"{100 * p['r_median_m']:.1f}" if p["r_median_m"] is not None else "-"
            r90 = f"{100 * p['r_p90_m']:.1f}" if p["r_p90_m"] is not None else "-"
            print(f"{v['label']:38s}{p['E']:>6.0f}{p['plane_m']:>7.0f}{p['p_any_charged']:>12.4f}{p['p_charged_25cm']:>13.4f}"
                  f"{p['n_electrons']:>9.3f}{p['n_gamma']:>10.3f}{p['n_neutron']:>8.3f}{r50:>10}{r90:>10}")
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ip, (ax, dz) in enumerate(zip(axs, PLANES)):
        for c, (name, v) in zip(colors, res.items()):
            pts = [p for p in v["points"] if p["plane_m"] == dz]
            ax.plot([p["E"] for p in pts], [max(p["p_any_charged"], 1e-6) for p in pts], "o-", color=c, lw=2, label=v["label"])
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("muon energy [GeV]")
        ax.set_title(f"{dz:.0f} m behind the slab: P(any charged secondary > 1 MeV)")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("probability per muon"); axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "em_debris.png", dpi=110)
    histograms(colors)
    print(f"wrote {OUT}")


def histograms(colors, E=50.0):
    """Shapes at one muon energy, 1 m and 5 m behind each slab: distance from the muon, multiplicity, spectra,
    angle to the muon, time delay (what a parametrisation for the fast simulation needs)."""
    import matplotlib.pyplot as plt
    for ip, dz in ((1, 1.0), (2, 5.0)):
        fig, axs = plt.subplots(2, 3, figsize=(17, 9))
        for c, name in zip(colors, CONFIGS):
            f = OUT / f"{name}.npz"
            if not f.exists():
                continue
            d = np.load(f)
            n = int(d["n"])
            m = (d["E"] == E) & (d["plane"] == ip)
            mu = m & (d["track"] == 1)
            ap = np.abs(d["pdg"])
            sec = m & (d["track"] != 1)
            ch = sec & ((ap == 11) | (ap == 13) | (ap == 211) | (ap == 321) | (ap == 2212)) & (d["ekin"] > E_CH)
            ee = sec & (ap == 11) & (d["ekin"] > E_CH)
            gg = sec & (d["pdg"] == 22) & (d["ekin"] > E_G)
            ref = {k: dict(zip(d["ev"][mu], d[k][mu])) for k in ("x", "y", "t", "px", "py", "pz")}
            ok = np.array([e in ref["x"] for e in d["ev"]], bool)
            g = lambda k, sel: np.array([ref[k][e] for e in d["ev"][sel & ok]])
            sel = ch & ok
            r = np.hypot(d["x"][sel] - g("x", ch), d["y"][sel] - g("y", ch))
            dt = d["t"][sel] - g("t", ch)
            a = np.stack([d["px"][sel], d["py"][sel], d["pz"][sel]], 1)
            b = np.stack([g("px", ch), g("py", ch), g("pz", ch)], 1)
            cosang = np.sum(a * b, 1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
            ang = np.arccos(np.clip(cosang, -1, 1))
            lab = CONFIGS[name][0]
            mult = np.bincount(np.bincount(d["ev"][ch], minlength=n), minlength=8)[:8] / n
            axs[0, 0].hist(np.log10(np.maximum(r, 1e-5)), bins=60, range=(-5, 1.5), weights=np.full(len(r), 1 / n), histtype="step", lw=2, color=c, label=lab)
            axs[0, 1].step(np.arange(8), np.maximum(mult, 1e-7), where="mid", lw=2, color=c)
            axs[0, 2].hist(np.log10(d["ekin"][ee]), bins=60, range=(-3, 2.5), weights=np.full(int(ee.sum()), 1 / n), histtype="step", lw=2, color=c)
            axs[1, 0].hist(np.log10(d["ekin"][gg]), bins=60, range=(-4, 2.5), weights=np.full(int(gg.sum()), 1 / n), histtype="step", lw=2, color=c)
            axs[1, 1].hist(np.log10(np.maximum(ang, 1e-6)), bins=60, range=(-5, 0.5), weights=np.full(len(ang), 1 / n), histtype="step", lw=2, color=c)
            axs[1, 2].hist(np.clip(dt, -0.5, 10), bins=60, range=(-0.5, 10), weights=np.full(len(dt), 1 / n), histtype="step", lw=2, color=c)
        for ax, (title, xl) in zip(axs.flat, (("charged secondaries: distance from the muon", "log10 r [m] (25 cm cell = -0.6)"),
                                             ("charged secondaries per muon", "number (> 1 MeV)"),
                                             ("e+- energy", "log10 E_kin [GeV]"), ("photon energy", "log10 E_kin [GeV]"),
                                             ("charged secondaries: angle to the muon", "log10 angle [rad]"),
                                             ("charged secondaries: delay after the muon", "dt [ns]"))):
            ax.set_title(title); ax.set_xlabel(xl); ax.set_yscale("log"); ax.grid(alpha=0.3)
            ax.set_ylabel("per muon")
        axs[0, 0].axvline(np.log10(0.25), color="k", ls="--", lw=1)
        axs[0, 0].legend(fontsize=8)
        fig.suptitle(f"{E:.0f} GeV muons, {dz:.0f} m behind the slab (full Geant4, FTFP_BERT_EMZ)")
        fig.tight_layout(); fig.savefig(OUT / f"em_debris_shapes_{dz:.0f}m.png", dpi=100)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", choices=list(CONFIGS))
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        summarize()
    else:
        run(a.config, a.seed + list(CONFIGS).index(a.config))


if __name__ == "__main__":
    main()
