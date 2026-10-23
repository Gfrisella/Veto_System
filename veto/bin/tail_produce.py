"""High-energy muon tail of the SHiP source by forced decays in flight (one slice of the production).

    bin/g4.sh python3 bin/tail_produce.py --slice K --n-slices N [--events 2000000] [--rep 4]
    bin/g4.sh python3 bin/tail_produce.py --prepare          # once: beam spot from the input sample

Per slice:
  1. Pythia 8 (cpp/tailgen/build/pythia_mesons): EVENTS proton-nucleon collisions at 400 GeV (FairShip
     minimum-bias settings), final-state pi+-, K+- above EMIN. One event = one proton on target: every
     proton interacts in the 1.5 m tungsten target (survival ~ e^-17).
  2. Per event: the proton's interaction point (optical depth through the ECN3 target layout with Geant4's
     proton inelastic length; beam spot (x, y) drawn from the input sample's high-momentum muons).
  3. Geant4 (cpp/tailgen tailtransport.cascade, FTFP_BERT) cascades every hadron of the event above E_CUT
     through the target (W blocks, water slits, iron), like FairShip does after Pythia's first interaction,
     so re-interactions (leading-particle regeneration) are included. FORCED DECAYS: every step of a pi+- /
     K+- above E_CUT emits one muon of its two-body decay with weight BR step / lambda_dec (lambda_dec =
     (p / m) c tau); real decay muons are dropped. Muons above PMIN are carried to z = -0.0045 m, the plane
     of MuonsAndMatter's input sample.
Weight per muon: muons per spill = w / REP * 5e13 / (events of the whole production) (tail_combine.py).
Not included: K_mu3 and K_L decays, charm / beauty (kept from the input sample).
"""

import _bootstrap  # noqa: F401
import argparse
import subprocess
import sys
import time

import numpy as np

from veto import paths

sys.path.insert(0, str(paths.PROJECT_DIR / "cpp/tailgen/build"))      # tailtransport module
OUT = paths.source_data() / "tail"
INPUT = str(paths.MUONS_AND_MATTER / "data/muons/full_sample_after_target.h5")
EMIN, PMIN = 130.0, 130.0                    # GeV: hadrons followed / muons kept at the decay
M = {211: 0.13957039, 321: 0.493677}
CTAU = {211: 7.8045, 321: 3.7118}            # m
BR_MUNU = {211: 0.999877, 321: 0.6356}
M_MU = 0.1056584
Z_OVER_A = {"Mo": 42 / 95.95, "W": 74 / 183.84, "water": 10 / 18.015, "Fe": 26 / 55.845}


def prepare():
    import h5py
    with h5py.File(INPUT, "r") as f:
        w = f["weight"][:]
        px, py, pz = f["px"][:], f["py"][:], f["pz"][:]
        p = np.sqrt(px.astype(np.float64) ** 2 + py ** 2 + pz ** 2)
        m = (np.abs(w - 768.75) < 1) & (p > 100)
        x, y = f["x"][:][m], f["y"][:][m]
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "beamspot.npy", np.stack([x, y], 1).astype(np.float32))
    print(f"beam spot from {m.sum()} input muons (min. bias, p > 100 GeV): x rms {x.std() * 100:.2f} cm, y rms {y.std() * 100:.2f} cm")


class Layout:
    """Piecewise materials along z (ECN3 target) and optical depth along a straight line."""

    def __init__(self, T, layout):
        self.z0 = np.array([a for a, b, m in layout])
        self.z1 = np.array([b for a, b, m in layout])
        self.mat = [m for a, b, m in layout]
        self.T = T
        self._cache = {}

    def lam(self, pdg, E):
        """Inelastic length [m] per segment for particle pdg at energies E (n,) -> (n, S), on a log-E grid."""
        key = pdg
        if key not in self._cache:
            grid = np.geomspace(20, 450, 24)
            self._cache[key] = (grid, {m: np.array([1.0 / self.T.inelastic_xs_per_m(int(pdg), float(e), m) for e in grid])
                                       for m in set(self.mat)})
        grid, tab = self._cache[key]
        lE = np.log(np.clip(E, grid[0], grid[-1]))
        return np.stack([np.interp(lE, np.log(grid), tab[m]) for m in self.mat], 1)

    def path_pieces(self, zs, uz):
        """Length along the direction (uz > 0) in every segment from z = zs on: (n, S)."""
        a = np.maximum(self.z0[None, :], zs[:, None])
        b = self.z1[None, :]
        return np.clip(b - a, 0, None) / uz[:, None]


def sample_along(lengths, lam, rng):
    """Sample a distance with density exp(-tau(l)) over piecewise segments (lengths, lam: (n, S)).

    Returns (distance, integral I = int exp(-tau) dl over the whole path) per row."""
    tau_seg = lengths / lam
    tau_before = np.cumsum(tau_seg, 1) - tau_seg
    I_seg = lam * np.exp(-tau_before) * (1 - np.exp(-tau_seg))
    I = I_seg.sum(1)
    cum = np.cumsum(I_seg, 1)
    r = rng.random(len(I)) * I
    k = np.minimum((r[:, None] > cum).sum(1), lengths.shape[1] - 1)
    rows = np.arange(len(I))
    r_in = (r - (cum[rows, k] - I_seg[rows, k])) / np.maximum(I_seg[rows, k], 1e-300)   # fraction of the segment's integral
    lk, tk = lam[rows, k], tau_seg[rows, k]
    d_in = -lk * np.log1p(-r_in * (1 - np.exp(-tk)))                                     # inverse CDF of exp(-l/lam)
    before = np.cumsum(lengths, 1) - lengths
    return before[rows, k] + d_in, I


def boost_decay(pdg, p3, rng):
    """Two-body decay meson -> mu nu, isotropic in the rest frame: muon lab momentum (n, 3)."""
    m = np.where(np.abs(pdg) == 211, M[211], M[321])
    pstar = (m ** 2 - M_MU ** 2) / (2 * m)
    cth = rng.uniform(-1, 1, len(m)); phi = rng.uniform(0, 2 * np.pi, len(m))
    sth = np.sqrt(1 - cth ** 2)
    ps = pstar[:, None] * np.stack([sth * np.cos(phi), sth * np.sin(phi), cth], 1)     # rest frame, z = meson direction
    Es = np.sqrt(pstar ** 2 + M_MU ** 2)
    P = np.linalg.norm(p3, axis=1)
    E = np.sqrt(P ** 2 + m ** 2)
    g, bg = E / m, P / m
    pz_lab = g * ps[:, 2] + bg * Es
    u = p3 / P[:, None]
    a = np.where(np.abs(u[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
    e1 = np.cross(u, a); e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(u, e1)
    return ps[:, 0:1] * e1 + ps[:, 1:2] * e2 + pz_lab[:, None] * u


def run_slice(k, n_slices, events, rep, emin=None, tag="tail", target="MoW", primary="pythia", physics="FTFP_BERT",
              l0=0.0, pmin_mu=None, dimuon=False, gamma_factor=500.0, pik=True, vm_fairship=False):
    global EMIN, PMIN
    if emin is not None:
        EMIN = PMIN = emin
    if pmin_mu is not None:
        PMIN = pmin_mu
    import tailtransport as T
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    binf = OUT / f"{tag}_mesons_{k}.bin"          # unique per run: parallel runs must not share it
    if primary in ("pythia", "pythia6"):
        exe = "pythia_mesons" if primary == "pythia" else "pythia6_mesons"
        extra = ["0.5"] if primary == "pythia6" else []   # equal p / n pools; reweighted to the vertex material below
        if dimuon:                                  # vector mesons undecayed in the generator: forced mu mu in Geant4
            extra = ["0.5", "vm"] if primary == "pythia6" else \
                ["0.4025"] + [f"{v}:mayDecay = off" for v in (221, 223, 113, 331, 333)]
        subprocess.run([str(paths.PROJECT_DIR / "cpp/tailgen/build" / exe), str(events), str(1000 + k), str(EMIN), str(binf)] + extra,
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        meta = dict(l.split()[:2] for l in open(str(binf) + ".txt") if len(l.split()) >= 2)
        n_ev = int(meta["events_generated"])
        rec = np.fromfile(binf, dtype=np.float64).reshape(-1, 6)
    else:                                   # Geant4 does the first interaction too (FairShip's -G mode): beam protons
        n_ev = events
        rec = np.zeros((events, 6))
        rec[:, 0] = np.arange(events)
        rec[:, 1] = 2212
        rec[:, 4] = 400.0
        rec[:, 5] = 400.0
    t_py = time.time() - t0
    T.initialize(seed=5000 + k, target=target, physics=physics, dimuon=dimuon, gamma_factor=gamma_factor)
    T.set_pik(pik)
    T.set_vm_fairship(vm_fairship)
    lay = Layout(T, T.target_layout())
    rng = np.random.default_rng(7000 + k)
    spot = np.load(OUT / "beamspot.npy")
    ev_ids, first = np.unique(rec[:, 0].astype(np.int64), return_index=True)
    offsets = np.append(first, len(rec))
    m = len(ev_ids)
    muons = []
    for r_ in range(rep):                               # same Pythia events, new vertices / cascades
        zs = np.full(m, lay.z0[0])
        d_p, _ = sample_along(lay.path_pieces(zs, np.ones(m)), lay.lam(2212, np.full(m, 400.0)), rng)
        if primary == "geant4":
            d_p = np.full(m, -0.01)         # start 1 cm before the target; Geant4 picks the interaction point
        xy = spot[rng.integers(0, len(spot), m)].astype(np.float64)
        t1 = time.time()
        o = T.cascade(offsets.tolist(), rec[:, 1].astype(int).tolist(), rec[:, 2].tolist(), rec[:, 3].tolist(),
                      rec[:, 4].tolist(), xy[:, 0].tolist(), xy[:, 1].tolist(), (zs + d_p).tolist(), EMIN, PMIN, l0)
        o = {kk: np.asarray(v) for kk, v in o.items()}
        o["event"] = ev_ids[o["ev"]]
        if primary == "pythia6":            # FairShip: target nucleon p / n with the Z/A of the material at the vertex
            n_p = int(meta["events_on_proton"])
            seg = np.clip(np.searchsorted(lay.z1, zs + d_p), 0, len(lay.mat) - 1)
            za = np.array([Z_OVER_A[lay.mat[s_]] for s_ in seg])
            on_p = ev_ids < n_p
            f_ev = np.where(on_p, za / (n_p / n_ev), (1 - za) / (1 - n_p / n_ev))
            o["w"] = o["w"] * f_ev[o["ev"]]
        muons.append(o)
    t_g4 = time.time() - t1
    # per-muon arrays only: the module also returns the scalar "interactions" (charm logging, added to tailtransport
    # after the tail2 production), which the fast-veto-sim script tried to concatenate as well
    o = {kk: np.concatenate([mu[kk] for mu in muons]) for kk in muons[0] if np.ndim(muons[0][kk]) > 0}
    np.savez(OUT / f"{tag}_{k}.npz", events=n_ev, rep=rep, emin=EMIN, pmin=PMIN,
             pdg=o["q"].astype(int), x=o["x"], y=o["y"], z=o["z"], px=o["px"], py=o["py"], pz=o["pz"],
             w_decay=o["w"], parent=o["parent"], z_decay=o["z_decay"], gen=o["gen"], event=o["event"],
             z_int=np.zeros(len(o["w"])), p_prod=o["p_create"])
    if binf.exists():
        binf.unlink()
    print(f"slice {k}: {n_ev} events, {m} with hadrons > {EMIN:.0f} GeV ({len(rec)} hadrons), x{rep}: "
          f"{len(o['w'])} forced-decay muons at the plane; pythia {t_py:.0f} s, geant4 (last rep) {t_g4:.0f} s, "
          f"total {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--slice", type=int, default=0)
    ap.add_argument("--n-slices", type=int, default=1)
    ap.add_argument("--events", type=int, default=2_000_000)
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--emin", type=float, default=None, help="energy threshold for hadrons and muons (tests)")
    ap.add_argument("--tag", default="tail")
    ap.add_argument("--target", default="MoW", choices=["MoW", "W"], help="ECN3 target: Mo front + W (default) or all W")
    ap.add_argument("--primary", default="pythia", choices=["pythia", "pythia6", "geant4"],
                    help="first interaction by Pythia 8.316, Pythia 6.4.28 (external/pythia6) or Geant4 FTF (FairShip -G)")
    ap.add_argument("--physics", default="FTFP_BERT", choices=["FTFP_BERT", "QGSP_BERT"], help="Geant4 list for the cascade")
    ap.add_argument("--l0", type=float, default=0.0, help="Russian roulette length [m] for forced decays (0: every step)")
    ap.add_argument("--pmin-mu", type=float, default=None, help="minimum muon momentum at the decay [GeV] (default = emin)")
    ap.add_argument("--dimuon", action="store_true", help="forced eta/omega/rho/eta'/phi -> mu mu (X) and boosted gamma -> mu mu")
    ap.add_argument("--gamma-factor", type=float, default=500.0)
    ap.add_argument("--no-pik", action="store_true", help="switch off the forced pi / K -> mu nu decays")
    ap.add_argument("--vm-fairship", action="store_true",
                    help="dimuons as the old production: all 5 mesons from the first collision, only omega/phi/rho from the cascade")
    a = ap.parse_args()
    if a.prepare:
        prepare()
    else:
        run_slice(a.slice, a.n_slices, a.events, a.rep, a.emin, a.tag, a.target, a.primary, a.physics, a.l0, a.pmin_mu,
                  a.dimuon, a.gamma_factor, not a.no_pik, a.vm_fairship)


if __name__ == "__main__":
    main()
