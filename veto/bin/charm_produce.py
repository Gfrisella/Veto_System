"""Charm muons at the plane after the target (one slice): Pythia 6 forced c cbar + Geant4 muon transport.

    bin/g4.sh python3 bin/charm_produce.py --slice K --n-slices N [--events 200000] [--chi 1.7e-3]

As FairShip's makeCascade.py (charm part of the old production): PYTHIA 6.4.28, MSEL=4 (forced c cbar), PoorE791
tune, 400 GeV p on p / n (Z/A of tungsten); charm hadrons and taus decayed by Pythia. Only the FIRST collision
(FairShip also adds charm from secondary hadrons of the cascade: compare with the old file to see that part).
Each muon starts at the event's proton interaction point (prompt decays) and is carried to z = -0.0045 m by
Geant4 (ECN3 Mo+W target, iron). Weight: muons per spill = chi (c cbar per p.o.t.) x 5e13 / events.
Output data/tail/charm_<K>.npz in the tail_produce.py layout (w_decay = chi, rep = 1).
"""

import _bootstrap  # noqa: F401
import argparse
import subprocess
import sys
import time

import numpy as np

from veto import paths

sys.path.insert(0, str(paths.PROJECT_DIR / "cpp/tailgen/build"))
from tail_produce import Layout, sample_along, OUT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--slice", type=int, default=0)
ap.add_argument("--n-slices", type=int, default=1)
ap.add_argument("--events", type=int, default=200000)
ap.add_argument("--chi", type=float, default=1.7e-3, help="c cbar pairs per proton on target (FairShip chicc)")
ap.add_argument("--emin", type=float, default=2.0, help="minimum muon energy at production [GeV]")
ap.add_argument("--tag", default="charm")
a = ap.parse_args()
import tailtransport as T  # noqa: E402

t0 = time.time()
binf = OUT / f"{a.tag}_{a.slice}.bin"
subprocess.run([str(paths.PROJECT_DIR / "cpp/tailgen/build/pythia6_mesons"), str(a.events), str(3000 + a.slice), str(a.emin), str(binf),
                "0.4025", "charm"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
meta = dict(l.split()[:2] for l in open(str(binf) + ".txt") if len(l.split()) >= 2)
n_ev = int(meta["events_generated"])
rec = np.fromfile(binf, dtype=np.float64).reshape(-1, 6)
T.initialize(seed=8000 + a.slice, target="MoW")
lay = Layout(T, T.target_layout())
rng = np.random.default_rng(9000 + a.slice)
spot = np.load(OUT / "beamspot.npy")
ev_ids, inv = np.unique(rec[:, 0].astype(np.int64), return_inverse=True)
m = len(ev_ids)
zs = np.full(m, lay.z0[0])
d_p, _ = sample_along(lay.path_pieces(zs, np.ones(m)), lay.lam(2212, np.full(m, 400.0)), rng)
xy = spot[rng.integers(0, len(spot), m)].astype(np.float64)
vx, vy, vz = xy[inv, 0], xy[inv, 1], (zs + d_p)[inv]
o = T.transport(rec[:, 1].astype(int).tolist(), vx.tolist(), vy.tolist(), vz.tolist(),
                rec[:, 2].tolist(), rec[:, 3].tolist(), rec[:, 4].tolist())
ok = o["ok"] > 0
n = int(ok.sum())
np.savez(OUT / f"{a.tag}_{a.slice}.npz", events=n_ev, rep=1, emin=a.emin, pmin=a.emin,
         pdg=rec[ok, 1].astype(int), x=o["x"][ok], y=o["y"][ok], z=o["z"][ok], px=o["px"][ok], py=o["py"][ok], pz=o["pz"][ok],
         w_decay=np.full(n, a.chi), parent=np.full(n, 4), z_decay=vz[ok], gen=np.zeros(n, int), event=ev_ids[inv][ok],
         z_int=vz[ok], p_prod=np.sqrt((rec[ok, 2:5] ** 2).sum(1)))
binf.unlink()
print(f"slice {a.slice}: {n_ev} c cbar events, {len(rec)} muons above {a.emin} GeV, {n} at the plane, {time.time() - t0:.0f} s", flush=True)
