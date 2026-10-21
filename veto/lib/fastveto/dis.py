"""Forced muon-nucleus interactions along the cuda-muons paths, final states from the library (scripts/dis_library.py).

    lib = MuonLibrary("data/dis/mu_library.npz")     # split="crossover": Geant4 nu < nu_x(E), Pythia 6 above
    lib.sigma(material, E)                        # microbarn per nucleon
    ev = lib.sample(material, E, u, vertex, pdg_mu, rng)   # final states in the lab: particles of every interaction

    out = run(columns, design, lib, draws={"helium": 1e6, "pvc": 1e6, "air": 2e5, "planes": 2e5}, rng=rng)

Interactions are forced and importance sampled per target group (helium, air, pvc, each design plane): within a
group the muon crossing is drawn with probability w P, P = n_area sigma(E) the interaction probability (<< 1),
w the muon weight (per spill); every draw carries weight S_g / M_g, S_g = sum of w P over the group (interactions
per spill), M_g the number of draws. The vertex is uniform along the crossing. The library event comes from the
neighbouring grid energies (chosen at random, linear in log E), rotated to the muon direction with a random
azimuth; its beam muon gets the sign of the real muon. No rescaling to the exact energy (grid steps of 15 %).
"""
import json

import numpy as np

from . import materials as mat
from . import targets as tg

UB = 1e-34                      # microbarn in m^2


def _segments(start, n):
    """Indices of the concatenated slices [start_i, start_i + n_i)."""
    n = np.asarray(n, np.int64)
    tot = int(n.sum())
    if tot == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    owner = np.repeat(np.arange(len(n)), n)
    off = np.arange(tot) - np.repeat(np.cumsum(n) - n, n)
    return np.repeat(np.asarray(start, np.int64), n) + off, owner


def rotate_to(v, u, phi):
    """Vectors v (n,3) of a frame with the beam along +z: azimuth phi about z, then +z -> u (n,3)."""
    c, s = np.cos(phi), np.sin(phi)
    x, y = c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]
    a = np.where(np.abs(u[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
    e1 = np.cross(u, a)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(u, e1)
    return x[:, None] * e1 + y[:, None] * e2 + v[:, 2:3] * u


class MuonLibrary:
    """split: "crossover" (Geant4 for nu < nu_x(E), Pythia above; default), "w2" (Geant4 W < 2 GeV, Pythia W > 2)
    or "geant4" (Geant4 only: what a full Geant4 simulation does)."""

    def __init__(self, path, split="crossover"):
        z = np.load(path)
        self.meta = json.loads(str(z["meta"]))
        self.split = split
        self.E = z["E_grid"].astype(np.float64)
        self.logE = np.log(self.E)
        self.nu_x = z["nu_x"]
        self.tables = {"p6": {k[3:]: z[k] for k in z.files if k.startswith("p6_") and k != "p6_sigma"}}
        for m in self.meta["materials"]:
            self.tables[m] = {k[len(m) + 4:]: z[k] for k in z.files if k.startswith(f"g4_{m}_") and k != f"g4_{m}_sigma"}
        # which events each generator contributes, and the cross section of that part (per target / material, energy)
        nE = len(self.E)
        self.p6_sigma = np.zeros((2, nE))
        self.g4 = {}
        self.groups = {}
        for name, t in self.tables.items():
            ie = t["ev_ie"].astype(np.int64)
            if split == "crossover":
                keep = t["ev_nu"] >= self.nu_x[ie] if name == "p6" else t["ev_nu"] < self.nu_x[ie]
            elif split == "w2":
                keep = t["ev_w"] >= 2.0 if name == "p6" else t["ev_w"] < 2.0
            elif split == "geant4":
                keep = np.full(len(ie), name != "p6")
            else:
                raise ValueError(f"unknown split {split!r}")
            key = t["ev_tgt"].astype(np.int64) * 1000 + ie
            n_all = np.bincount(key, minlength=2000)
            n_keep = np.bincount(key[keep], minlength=2000)
            frac = np.divide(n_keep, n_all, out=np.zeros(2000), where=n_all > 0)
            if name == "p6":
                self.p6_sigma = z["p6_sigma"] * np.stack([frac[:nE], frac[1000:1000 + nE]])
            else:
                self.g4[name] = z[f"g4_{name}_sigma"] * frac[:nE]
            kept = np.nonzero(keep)[0]
            order = kept[np.argsort(key[kept], kind="stable")]
            ks, first, cnt = np.unique(key[order], return_index=True, return_counts=True)
            self.groups[name] = (order, {int(k): (f, c) for k, f, c in zip(ks, first, cnt)})

    def _interp(self, table, E):
        """Linear in log E on the grid; 0 below the grid, the last value above it."""
        x = np.log(np.clip(E, self.E[0], self.E[-1]))
        return np.where(E < self.E[0], 0.0, np.interp(x, self.logE, table))

    def sigma_parts(self, material, E):
        za = mat.MATERIALS[material].z_over_a
        p6 = za * self._interp(self.p6_sigma[0], E) + (1 - za) * self._interp(self.p6_sigma[1], E)
        return p6, self._interp(self.g4[material], E)

    def sigma(self, material, E):
        """Cross section per nucleon [microbarn] of muons of total energy E in a material."""
        a, b = self.sigma_parts(material, E)
        return a + b

    def _pick(self, name, tgt, ie, rng):
        order, g = self.groups[name]
        out = np.empty(len(ie), np.int64)
        keys = tgt.astype(np.int64) * 1000 + ie
        for k in np.unique(keys):
            m = keys == k
            # a part can be empty at one grid energy and not at its neighbour (interpolated cross section > 0)
            kk = next(x for x in (int(k), int(k) + 1, int(k) - 1, int(k) + 2, int(k) - 2) if x in g)
            f, c = g[kk]
            out[m] = order[f + rng.integers(0, c, m.sum())]
        return out

    def sample(self, material, E, u, vertex, pdg_mu, rng):
        """One final state per interaction (arrays of length n). Returns particles: interaction index, pdg,
        p (k,4) and production point (k,3) in the lab, beam flag (the scattered muon), and per interaction the
        component (0 Pythia, 1 Geant4) and the library energy."""
        n = len(E)
        p6, g4 = self.sigma_parts(material, E)
        comp = (rng.random(n) * (p6 + g4) >= p6).astype(np.int8)
        x = np.log(np.clip(E, self.E[0], self.E[-1]))
        j = np.clip(np.searchsorted(self.logE, x) - 1, 0, len(self.E) - 2)
        f = (x - self.logE[j]) / (self.logE[j + 1] - self.logE[j])
        ie = j + (rng.random(n) < f)
        za = mat.MATERIALS[material].z_over_a
        sp, sn = self.p6_sigma[0, ie] * za, self.p6_sigma[1, ie] * (1 - za)
        tgt = np.where(comp == 0, (rng.random(n) * (sp + sn) >= sp).astype(np.int64), 0)
        parts = {k: [] for k in ("int", "pdg", "p", "v", "beam")}
        for c, name in ((0, "p6"), (1, material)):
            m = np.nonzero(comp == c)[0]
            if not len(m):
                continue
            t = self.tables[name]
            ev = self._pick(name, tgt[m], ie[m], rng)
            rows, owner = _segments(t["ev_start"][ev], t["ev_n"][ev])
            k = m[owner]
            phi = rng.uniform(0, 2 * np.pi, n)[k]
            p3 = rotate_to(t["p"][rows, :3].astype(np.float64), u[k], phi)
            v3 = rotate_to(t["v"][rows].astype(np.float64), u[k], phi) + vertex[k]
            pdg = t["pdg"][rows].copy()
            beam = t["beam"][rows]
            pdg[beam] = np.where(pdg_mu[k][beam] < 0, -13, 13)
            parts["int"].append(k); parts["pdg"].append(pdg); parts["beam"].append(beam)
            parts["p"].append(np.concatenate([p3, t["p"][rows, 3:4].astype(np.float64)], 1)); parts["v"].append(v3)
        out = {kk: np.concatenate(v) for kk, v in parts.items()}
        order = np.argsort(out["int"], kind="stable")
        out = {kk: v[order] for kk, v in out.items()}
        out["component"] = comp
        out["E_lib"] = self.E[ie]
        return out


def group_of(code):
    return np.where(code >= tg.TARGET_PLANE0, code, np.minimum(code, tg.TARGET_PVC))


def group_label(g, design=None):
    return tg.target_name(int(g), design)


def interaction_probability(cr, lib):
    """P = n_area sigma(E) for every crossing row."""
    P = np.zeros(len(cr["E"]))
    for mi, mname in enumerate(tg.MATERIAL_NAMES):
        m = cr["material"] == mi
        if m.any():
            P[m] = cr["n_area"][m] * lib.sigma(mname, cr["E"][m]) * UB
    return P


KEYS = ("x_30", "y_30", "z_30", "px_30", "py_30", "pz_30", "x_82", "y_82", "z_82", "px_82", "py_82", "pz_82",
        "kx", "ky", "kz", "weight", "pdg", "evt_n", "type")
INTER_KEYS = ("muon", "code", "x", "y", "z", "E", "weight", "pdg_mu", "ux", "uy", "uz", "p_int", "evt_n", "type")
PER_INTERACTION_EXTRAS = ("nucleus", "proc", "charm")
_CTX = {}                     # shared with forked workers: columns, design, library, chunks, include


def _crossings(ci):
    s, e = _CTX["chunks"][ci]
    d = {k: np.asarray(_CTX["columns"][k][s:e]) for k in KEYS}
    cr = tg.crossings(d, _CTX["design"], _CTX["include"])
    P = interaction_probability(cr, _CTX["lib"])
    return d, cr, P, d["weight"][cr["muon"]].astype(np.float64) * P, group_of(cr["code"])


def _pass1(ci):
    _, cr, _, wP, g = _crossings(ci)
    return {int(gg): float(wP[g == gg].sum()) for gg in np.unique(g)}, len(cr["E"])


def _pass2(job):
    ci, need, w_draw, seed = job
    if not need:
        return None
    rng = np.random.default_rng(seed)
    s = _CTX["chunks"][ci][0]
    d, cr, P, wP, g = _crossings(ci)
    out = {k: [] for k in INTER_KEYS}
    for gg, k in need.items():
        rows = np.nonzero(g == gg)[0]
        cdf = np.cumsum(wP[rows])
        pick = rows[np.searchsorted(cdf, rng.random(k) * cdf[-1], side="right").clip(0, len(rows) - 1)]
        vtx = tg.sample_vertex(cr, pick, rng)
        mu = cr["muon"][pick]
        for kk, v in (("muon", mu + s), ("code", cr["code"][pick]), ("x", vtx[:, 0]), ("y", vtx[:, 1]), ("z", vtx[:, 2]),
                      ("E", cr["E"][pick]), ("weight", np.full(k, w_draw[gg])), ("pdg_mu", d["pdg"][mu]),
                      ("ux", cr["u"][pick, 0]), ("uy", cr["u"][pick, 1]), ("uz", cr["u"][pick, 2]), ("p_int", P[pick]),
                      ("evt_n", d["evt_n"][mu]), ("type", d["type"][mu])):
            out[kk].append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


def run(columns, design, lib, draws, rng, chunk=2_000_000, include=("helium", "air", "pvc", "planes"), workers=1,
        verbose=True):
    """Forced interactions for a dataset (dict of column arrays, may be memory-mapped). draws: per group
    {"helium": M, "air": M, "pvc": M, "planes": M (each plane)}. Two passes over chunks of rows (workers: forked
    processes). Returns (interactions, particles, totals = interactions per spill per group)."""
    import multiprocessing as mp
    import time
    t0 = time.time()
    n = len(columns["weight"])
    _CTX.update(columns=columns, design=design, lib=lib, include=include,
                chunks=[(s, min(s + chunk, n)) for s in range(0, n, chunk)])
    nch = len(_CTX["chunks"])
    pool = mp.get_context("fork").Pool(workers) if workers > 1 else None
    mapper = pool.map if pool else lambda f, it: list(map(f, it))
    try:
        r1 = mapper(_pass1, range(nch))
        sums = {}
        for ci, (sm, _) in enumerate(r1):
            for g, v in sm.items():
                sums.setdefault(g, np.zeros(nch))[ci] = v
        if verbose:
            print(f"  pass 1: {nch} chunks, {sum(x[1] for x in r1):,} crossings, {time.time() - t0:.0f} s", flush=True)
        m_g = {g: int(draws.get("planes" if g >= tg.TARGET_PLANE0 else tg.target_name(g), 0)) for g in sums}
        per_chunk = {g: rng.multinomial(m_g[g], sums[g] / sums[g].sum()) if sums[g].sum() > 0 and m_g[g] > 0
                     else np.zeros(nch, int) for g in sums}
        w_draw = {g: sums[g].sum() / max(m_g[g], 1) for g in sums}
        seeds = rng.integers(0, 2 ** 62, nch)
        jobs = [(ci, {g: int(per_chunk[g][ci]) for g in sums if per_chunk[g][ci] > 0}, w_draw, int(seeds[ci])) for ci in range(nch)]
        r2 = [x for x in mapper(_pass2, jobs) if x is not None]
        if verbose:
            print(f"  pass 2: {sum(len(x['E']) for x in r2):,} interactions, {time.time() - t0:.0f} s", flush=True)
    finally:
        if pool:
            pool.close()
    inter = {k: [x[k] for x in r2] for k in INTER_KEYS}
    inter = {k: np.concatenate(v) for k, v in inter.items()}
    # final states, per material
    material = np.zeros(len(inter["E"]), np.int8)
    for code in np.unique(inter["code"]):
        if code >= tg.TARGET_PLANE0:
            mname, _ = tg.plane_material(design.planes[code - tg.TARGET_PLANE0])
        else:
            mname = tg.target_name(code)
        material[inter["code"] == code] = tg.MATERIAL_NAMES.index(mname)
    inter["material"] = material
    parts, comp, elib, extra = [], np.zeros(len(material), np.int8), np.zeros(len(material)), {}
    for mi in np.unique(material):
        m = np.nonzero(material == mi)[0]
        u = np.stack([inter["ux"][m], inter["uy"][m], inter["uz"][m]], 1)
        vtx = np.stack([inter["x"][m], inter["y"][m], inter["z"][m]], 1)
        o = lib.sample(tg.MATERIAL_NAMES[mi], inter["E"][m], u, vtx, inter["pdg_mu"][m], rng)
        o["int"] = m[o["int"]]
        comp[m], elib[m] = o.pop("component"), o.pop("E_lib")
        for k in PER_INTERACTION_EXTRAS:                  # e.g. the neutrino library's nucleus / process / charm
            if k in o:
                v = o.pop(k)
                extra.setdefault(k, np.zeros(len(material), v.dtype))[m] = v
        parts.append(o)
    inter["component"], inter["E_lib"] = comp, elib
    inter.update(extra)
    part = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.argsort(part["int"], kind="stable")
    part = {k: v[order] for k, v in part.items()}
    totals = {group_label(g, design): float(sums[g].sum()) for g in sums}
    return inter, part, totals
