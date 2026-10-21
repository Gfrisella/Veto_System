"""Neutrino interactions from the GENIE library (scripts/nu_library.py), with the interface of fastveto.dis.MuonLibrary
so that fastveto.dis.run does the per-design forced interactions for neutrinos too (one flavour per library object).

    lib = NuLibrary("data/genie/nu_library.npz", flavour=14)
    lib.sigma("helium", E)        # microbarn per nucleon (composition of the material, GENIE splines)
    inter, part, totals = dis.run(neutrino_columns, design, lib, draws, rng)

Materials as nuclei (fastveto.materials names): helium He4; air N 75.53 %, O 23.18 %, Ar 1.28 % by mass; PVC C2H3Cl;
Mylar C10H8O4; polystyrene C8H8; PVT C9H10; silicon Si28; Cl 75.8 % Cl35, 24.2 % Cl37. The interacting nucleus is
drawn with probability n_i sigma_i(E); the event from the neighbouring grid energies (random, linear in log E),
rotated to the neutrino direction with a random azimuth.
"""
import json

import numpy as np

from .dis import _segments, rotate_to

UB_CM2 = 1e-30
# nuclei per molecule / per unit (code: count); air by mass fraction / A
COMPOSITION = {
    "helium": {1000020040: 1.0},
    "air": {1000070140: 0.7553 / 14.007, 1000080160: 0.2318 / 15.999, 1000180400: 0.0128 / 39.948},
    "pvc": {1000060120: 2, 1000010010: 3, 1000170350: 0.758, 1000170370: 0.242},
    "mylar": {1000060120: 10, 1000010010: 8, 1000080160: 4},
    "polystyrene": {1000060120: 8, 1000010010: 8},
    "pvt": {1000060120: 9, 1000010010: 10},
    "silicon": {1000140280: 1.0},
}


def mass_number(code):
    return (code % 10000) // 10


class NuLibrary:
    def __init__(self, path, flavour=14):
        z = np.load(path)
        self.meta = json.loads(str(z["meta"]))
        self.E = z["E_grid"].astype(np.float64)
        self.logE = np.log(self.E)
        self.nus = list(z["nus"])
        self.nuclei = list(z["nuclei"])
        self.flavour = flavour
        self.fi = self.nus.index(flavour)
        self.sig = z["sigma"][self.fi]                        # (n_nuclei, nE) cm^2 per nucleus
        self.t = {k: z[k] for k in ("ev_ie", "ev_nu", "ev_tgt", "ev_cc", "ev_proc", "ev_charm", "ev_start", "ev_n",
                                    "pdg", "p", "v", "beam")}
        sel = np.nonzero(self.t["ev_nu"] == self.fi)[0]
        key = self.t["ev_tgt"][sel].astype(np.int64) * 1000 + self.t["ev_ie"][sel]
        order = sel[np.argsort(key, kind="stable")]
        ks, first, cnt = np.unique(np.sort(key), return_index=True, return_counts=True)
        self.order, self.groups = order, {int(k): (f, c) for k, f, c in zip(ks, first, cnt)}

    def _sig_nuclei(self, E):
        x = np.log(np.clip(E, self.E[0], self.E[-1]))
        s = np.stack([np.interp(x, self.logE, self.sig[j]) for j in range(len(self.nuclei))], 1)
        return np.where((E < self.E[0])[:, None], 0.0, s)        # (n, n_nuclei) cm^2

    def _weights(self, material, E):
        comp = COMPOSITION[material]
        n_i = np.array([comp.get(c, 0.0) for c in self.nuclei])
        return n_i, self._sig_nuclei(np.atleast_1d(np.asarray(E, float))) * n_i[None, :]

    def sigma(self, material, E):
        """Cross section per nucleon [microbarn] of this flavour in a material at energy E."""
        n_i, w = self._weights(material, E)
        nucleons = np.sum(n_i * np.array([mass_number(c) for c in self.nuclei]))
        return w.sum(1) / nucleons / UB_CM2

    def sample(self, material, E, u, vertex, pdg_nu, rng):
        n = len(E)
        _, w = self._weights(material, E)
        cdf = np.cumsum(w, 1)
        tgt = (rng.random(n)[:, None] * cdf[:, -1:] > cdf).sum(1)
        x = np.log(np.clip(E, self.E[0], self.E[-1]))
        j = np.clip(np.searchsorted(self.logE, x) - 1, 0, len(self.E) - 2)
        ie = j + (rng.random(n) < (x - self.logE[j]) / (self.logE[j + 1] - self.logE[j]))
        keys = tgt.astype(np.int64) * 1000 + ie
        ev = np.empty(n, np.int64)
        for k in np.unique(keys):
            m = keys == k
            kk = next(x for x in (int(k), int(k) + 1, int(k) - 1, int(k) + 2, int(k) - 2) if x in self.groups)
            f, c = self.groups[kk]
            ev[m] = self.order[f + rng.integers(0, c, m.sum())]
        t = self.t
        rows, owner = _segments(t["ev_start"][ev], t["ev_n"][ev])
        phi = rng.uniform(0, 2 * np.pi, n)[owner]
        p3 = rotate_to(t["p"][rows, :3].astype(np.float64), u[owner], phi)
        v3 = rotate_to(t["v"][rows].astype(np.float64), u[owner], phi) + vertex[owner]
        return {"int": owner, "pdg": t["pdg"][rows].copy(), "beam": t["beam"][rows],
                "p": np.concatenate([p3, t["p"][rows, 3:4].astype(np.float64)], 1), "v": v3,
                "component": (1 - t["ev_cc"][ev]).astype(np.int8),          # 0 CC, 1 NC
                "E_lib": self.E[ie], "nucleus": np.array(self.nuclei)[tgt], "proc": t["ev_proc"][ev],
                "charm": t["ev_charm"][ev]}
