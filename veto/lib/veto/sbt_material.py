"""The SBT as a material (ISSUES D19, 16 Nov 2026): interaction targets, own energy deposits, scattering material.

lib/veto/sbt.py models the SBT cells as a detector without mass. The real cells (FairShip 26.10
geometry/veto_config_helium.yaml, veto/veto.cxx; SPSC-SR-370 3.3.3) on the four lateral faces of the frustum vessel are,
going out from the helium balloon (350 um Mylar, fastveto.vessel, the "wall" target):

    inner support wall   0.8 cm Aluminium   (FairShip innerSupport, "VetoInnerWall": the solid vessel holding the balloon)
    liquid scintillator  20  cm LAB + PPO   (sensitiveThickness; media.geo: H 12.16 %, C 87.84 % by mass, 0.86 g/cm^3)
    outer support wall   0.8 cm Aluminium   (outerSupport, "VetoOuterWall")
    ribs                 1.5 cm Aluminium, 20 cm deep, inside the LS layer: a transverse ("vertical") rib at every cell
                         edge in z (cells 82 cm, 80 cm the first), 3 longitudinal ribs on each +-x face (4 cells across),
                         2 on each +-y face (3 cells across), 4 corner ribs (neglected)
(FairShip uses "Aluminium" for AlMg4.5; Mg 4.5 % neglected.) Offsets are added to the half widths in x and y, as FairShip
(wx + 2 thickness) and sbt.T_SHELL. Per unit wall area at normal incidence: inner Al 2.16 g/cm^2, LS 17.2, outer Al
2.16, ribs (smeared) ~1-2: ~470 x the 350 um Mylar (0.049 g/cm^2).

Targets (interaction codes, interactions.CODE_NAMES; 9101-9104 cannot clash with plane codes 10 + i):
    9101 sbt_al_inner   inert   faces the helium: its products enter the vessel without crossing the LS
    9102 sbt_ls         active  (the LS, minus the rib fraction)
    9103 sbt_rib        inert   Al ribs smeared over the LS layer (fraction rib_fraction(face, z))
    9104 sbt_al_outer   inert   behind the LS
Materials: "aluminium" (G4_Al library table, bin/dis_library.py) and "ls" (LS density / Z/A; the muon library table is
G4_POLYETHYLENE as a stand-in for LAB's C / H nuclei; GENIE: C12 + H1 by the LS mass fractions).

Muon and neutrino interactions are sampled along the stored straight paths (the Nov-4 muon store, the 2026 neutrino
flux stores) as fastveto.dis.run does for the vessel walls (two passes, crossing drawn with probability w P), with an
optional enrichment: only final states that can give a vertex inside the helium are drawn (a strange hadron, or any
product starting >= 0.1 mm from the vertex, i.e. the decay products of K_S / Lambda / Sigma / Xi / Omega made by the
generator), weight x their fraction. The complement has every particle at the vertex in the wall, outside the
helium, so its pairs fail "vertex in the vessel" by construction (checked with an unenriched sample, docs/d19_sbt.md).

deposits(): parametrised response of the LS cells to every particle of an event (straight lines, no field):
    charged      dE/dx = 1.5 MeV/cm x 1/beta^2 (capped at x20) in the LS, 4.36 MeV/cm x 1/beta^2 in the Al layers,
                 stopping when its kinetic energy is spent (Birks quenching neglected)
    photons      conversion with lambda = 9/7 X0 (Al 8.90 cm, LS 51.7 cm); after it, 2 x 1.5 MeV/cm along the remaining
                 LS path, at most E_gamma (no shower development: 0.4 X0 of LS)
    neutral had. (n, K_L, anti-n, Lambda if undecayed) one interaction with lambda = NEUTRAL_LAMBDA (LS) / 0.397 m (Al);
                 in the LS it deposits NEUTRAL_VIS x T_kin (quenched recoil protons, nuclear break-up), in Al nothing
    not modelled: the nuclear fragments (ions are not kept in the libraries), Compton / delta rays, the WOM light yield
                 vs position. All parameters below are labelled assumptions, to be replaced by a Geant4 response.
"""
import multiprocessing as mp
import time

import numpy as np

from fastveto import materials as fmat
from fastveto import nudis
from fastveto import targets as tg

from . import geometry as geo
from . import sbt

# ---------------------------------------------------------------- geometry (FairShip 26.10)
T_AL_INNER = 0.008          # m  innerSupport 0.8 cm Aluminium
T_LS = sbt.T_LS             # m  sensitiveThickness 20 cm LiquidScintillator
T_AL_OUTER = 0.008          # m  outerSupport 0.8 cm Aluminium
T_RIB = 0.015               # m  rib 1.5 cm Aluminium
N_LONG_RIBS = {0: 3, 1: 3, 2: 2, 3: 2}     # longitudinal ribs on the faces +x, -x (4 cells across), +y, -y (3 cells)
G_LS0, G_LS1 = T_AL_INNER, T_AL_INNER + T_LS
G_OUT = G_LS1 + T_AL_OUTER

LS_DENSITY = 0.86                          # g/cm^3, FairShip media.geo LiquidScintillator
LS_W_H, LS_W_C = 0.1216, 0.8784            # mass fractions
LS_X0 = 1 / (LS_W_C / 42.70 + LS_W_H / 63.04)   # g/cm^2 (PDG X0 of C and H): 44.4
AL_X0 = 24.01                              # g/cm^2
CODES = {9101: "sbt_al_inner", 9102: "sbt_ls", 9103: "sbt_rib", 9104: "sbt_al_outer"}
ACTIVE = (9102,)
# (code, material, inner offset, outer offset, kind) ; kind: "full", "ls" (x (1 - rib fraction)), "rib" (x rib fraction)
LAYERS = ((9101, "aluminium", 0.0, G_LS0, "full"), (9102, "ls", G_LS0, G_LS1, "ls"), (9103, "aluminium", G_LS0, G_LS1, "rib"),
          (9104, "aluminium", G_LS1, G_OUT, "full"))
MATERIAL_NAMES = ["aluminium", "ls"]

# ---------------------------------------------------------------- response parameters (ASSUMPTIONS, see the docstring)
DEDX_LS = sbt.DEDX_LS       # MeV / m for a MIP (1.5 MeV/cm, as the cell model)
DEDX_AL = 436.0             # MeV / m, MIP in Al (1.615 MeV cm^2/g x 2.699)
BETA2_CAP = 20.0            # 1 / beta^2 capped
LAMBDA_CONV_AL = 9 / 7 * AL_X0 / 2.699 / 100     # m, photon conversion length in Al (0.114)
LAMBDA_CONV_LS = 9 / 7 * LS_X0 / LS_DENSITY / 100   # m (0.664)
NEUTRAL_LAMBDA = 0.80       # m, interaction length of neutral hadrons in LS (C inelastic + n-p elastic, ~0.1-1 GeV)
NEUTRAL_LAMBDA_AL = 0.397   # m, Al nuclear interaction length (PDG)
NEUTRAL_VIS = 0.3           # visible fraction of the kinetic energy of an interacting neutral hadron in the LS
NEUTRAL_HADRONS = (2112, 130, 3122, 3212, 3322)
MASS = {11: 0.000511, 13: 0.10566, 211: 0.13957, 321: 0.49368, 2212: 0.93827, 2112: 0.93957, 130: 0.49761, 310: 0.49761,
        3122: 1.11568, 3222: 1.18937, 3112: 1.19745, 3312: 1.32171, 3334: 1.67245, 3212: 1.19264, 3322: 1.31486, 22: 0.0,
        111: 0.13498, 221: 0.54786, 15: 1.77686, 411: 1.86966, 421: 1.86484, 431: 1.96835, 4122: 2.28646}
CHARGED = (11, 13, 15, 211, 321, 2212, 3222, 3112, 3312, 3334, 411, 431, 4122)
STRANGE = (130, 310, 3122, 321, 3222, 3112, 3212, 3312, 3322, 3334)
DISPLACED_MIN = 1e-4        # m: a product starting this far from the vertex comes from a decay in flight


def register():
    """Add the SBT materials to fastveto (materials, GENIE compositions, interaction codes); idempotent."""
    fmat.MATERIALS.setdefault("aluminium", fmat.Material("aluminium", AL_X0, 2.699, 13 / 26.9815, 27.0))
    fmat.MATERIALS.setdefault("ls", fmat.Material("ls", LS_X0, LS_DENSITY, LS_W_C * 6 / 12.011 + LS_W_H / 1.008, 12.0))
    nudis.COMPOSITION.setdefault("aluminium", {1000130270: 1.0})
    nudis.COMPOSITION.setdefault("ls", {1000060120: LS_W_C / 12.011, 1000010010: LS_W_H / 1.008})
    from . import interactions as ix
    ix.CODE_NAMES.update(CODES)


def rib_fraction(face, z):
    """Fraction of the LS layer taken by Al ribs (transverse every cell edge + longitudinal across the face)."""
    hx, hy = geo.vessel_half_widths(np.asarray(z, float))
    face = np.asarray(face)
    ring = np.where(np.asarray(z) < geo.Z_VESSEL_MIN + sbt.RING0, sbt.RING0, sbt.RING)
    n_long = np.where(np.isin(face, (0, 1)), N_LONG_RIBS[0], N_LONG_RIBS[2])
    across = np.where(np.isin(face, (0, 1)), 2 * hy, 2 * hx)
    return np.clip(T_RIB / ring + n_long * T_RIB / across, 0, 1)


# ---------------------------------------------------------------- pieces of straight segments in a layer

def layer_pieces(p, q, g0, g1):
    """Pieces of the segments p -> q (n, 3) between the frustum grown by g0 and the one grown by g1 (lateral shell layer,
    z0 < z < z1). Returns (segment index, ta, tb) with t the fraction along the segment."""
    d = q - p
    a1, b1 = sbt._frustum_interval(p, d, g1)
    a0, b0 = sbt._frustum_interval(p, d, g0)
    has_out = b1 > a1
    has_in = b0 > a0
    out = []
    # first piece: [a1, a0] (or the whole outer chord without an inner one), second: [b0, b1]
    tb1 = np.where(has_in, np.minimum(b1, a0), b1)
    ok = has_out & (tb1 > a1)
    out.append((np.nonzero(ok)[0], a1[ok], tb1[ok]))
    ta2 = np.maximum(a1, b0)
    ok = has_out & has_in & (b1 > ta2)
    out.append((np.nonzero(ok)[0], ta2[ok], b1[ok]))
    return tuple(np.concatenate([x[k] for x in out]) for k in range(3))


def face_of(mid):
    hx, hy = geo.vessel_half_widths(np.clip(mid[:, 2], geo.Z_VESSEL_MIN, geo.Z_VESSEL_MAX))
    xface = (np.abs(mid[:, 0]) - hx) >= (np.abs(mid[:, 1]) - hy)
    return np.where(xface, np.where(mid[:, 0] > 0, 0, 1), np.where(mid[:, 1] > 0, 2, 3))


def split_rings(p, d, seg, ta, tb):
    """Split pieces at the SBT ring edges (z linear in t). Returns seg, ta, tb, face, ring, cellseg."""
    za, zb = p[seg, 2] + ta * d[seg, 2], p[seg, 2] + tb * d[seg, 2]
    zlo, zhi = np.minimum(za, zb), np.maximum(za, zb)
    out = {k: [] for k in ("seg", "ta", "tb")}
    for r in range(sbt.N_RING):
        e0, e1 = sbt.RING_EDGES[r], sbt.RING_EDGES[r + 1]
        m = (zhi > e0) & (zlo < e1)
        if not m.any():
            continue
        dz = d[seg[m], 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t0 = np.where(dz != 0, (e0 - p[seg[m], 2]) / dz, ta[m]); t1 = np.where(dz != 0, (e1 - p[seg[m], 2]) / dz, tb[m])
        u0, u1 = np.maximum(ta[m], np.minimum(t0, t1)), np.minimum(tb[m], np.maximum(t0, t1))
        k = u1 > u0
        out["seg"].append(seg[m][k]); out["ta"].append(u0[k]); out["tb"].append(u1[k])
    if not out["seg"]:
        z = np.zeros(0, int)
        return z, np.zeros(0), np.zeros(0), z, z, z
    sg, a, b = (np.concatenate(out[k]) for k in ("seg", "ta", "tb"))
    mid = p[sg] + (0.5 * (a + b))[:, None] * d[sg]
    face = face_of(mid)
    hx, hy = geo.vessel_half_widths(np.clip(mid[:, 2], geo.Z_VESSEL_MIN, geo.Z_VESSEL_MAX))
    ring, cseg = sbt.cell_of(face, np.clip(mid[:, 0], -hx, hx), np.clip(mid[:, 1], -hy, hy), mid[:, 2])
    return sg, a, b, face, ring, cseg


# ---------------------------------------------------------------- interaction targets

def crossings(d):
    """SBT-layer crossings of the paths in d (columns as fastveto.targets): one row per piece; muon, code, material (index
    into MATERIAL_NAMES), p0, p1, n_area [nucleons / m^2], z, u, E, active."""
    rows = {k: [] for k in ("muon", "code", "material", "p0", "p1", "n_area", "z", "u")}
    for p, q, idx in tg.paths(d):
        good = np.isfinite(p).all(1) & np.isfinite(q).all(1) & (q[:, 2] - p[:, 2] > 1e-9)
        p, q, idx = p[good], q[good], idx[good]
        if not len(p):
            continue
        dv = q - p
        L = np.linalg.norm(dv, axis=1)
        dn = dv / L[:, None]
        cache = {}
        for code, mname, g0, g1, kind in LAYERS:
            if (g0, g1) not in cache:
                cache[(g0, g1)] = layer_pieces(p, q, g0, g1)
            seg, ta, tb = cache[(g0, g1)]
            if not len(seg):
                continue
            a, b = p[seg] + ta[:, None] * dv[seg], p[seg] + tb[:, None] * dv[seg]
            length = (tb - ta) * L[seg]
            mid = 0.5 * (a + b)
            f = 1.0
            if kind != "full":
                fr = rib_fraction(face_of(mid), mid[:, 2])
                f = fr if kind == "rib" else 1 - fr
            n_area = fmat.MATERIALS[mname].nucleons_per_m2(1.0) * length * f
            rows["muon"].append(idx[seg]); rows["code"].append(np.full(len(seg), code, np.int16))
            rows["material"].append(np.full(len(seg), MATERIAL_NAMES.index(mname), np.int8))
            rows["p0"].append(a); rows["p1"].append(b); rows["n_area"].append(n_area); rows["z"].append(mid[:, 2])
            rows["u"].append(dn[seg])
    if not rows["muon"]:
        return {k: np.zeros(0) for k in rows} | {"E": np.zeros(0)}
    out = {k: np.concatenate(v) for k, v in rows.items()}
    out["E"] = np.sqrt(tg.momentum_at(d, out["muon"], out["z"]) ** 2 + tg.M_MU ** 2)
    return out


def _flag_final_states(t, n_ev):
    """Per library event: a strange hadron among the products or a displaced product (can give a vertex in the helium)."""
    ev_of_row = np.repeat(np.arange(n_ev), np.asarray(t["ev_n"], np.int64))
    a = np.abs(np.asarray(t["pdg"][:len(ev_of_row)]))
    disp = np.linalg.norm(np.asarray(t["v"][:len(ev_of_row)], float), axis=1) > DISPLACED_MIN
    flag = np.isin(a, STRANGE) | disp
    return np.bincount(ev_of_row, weights=flag, minlength=n_ev) > 0


def enrich_muon_library(lib):
    """Redefine fastveto.dis.MuonLibrary's "v0" final-state groups as the final states that can give a vertex in the
    helium (_flag_final_states); lib.sample(..., v0_only=True) then draws only those, with w_factor = their fraction."""
    lib.f_v0 = {}
    for name, t in lib.tables.items():
        order, _ = lib.groups[name]                       # the kept events of this table (split), sorted by key
        ie = t["ev_ie"].astype(np.int64)
        key = t["ev_tgt"].astype(np.int64) * 1000 + ie
        has = _flag_final_states(t, len(ie))
        t["_has_v0"] = has
        kv = order[has[order]]
        order_v = kv[np.argsort(key[kv], kind="stable")]
        ks, first, cnt = np.unique(key[order_v], return_index=True, return_counts=True)
        lib.groups_v0[name] = (order_v, {int(k): (f, c) for k, f, c in zip(ks, first, cnt)})
        n_keep = np.bincount(key[order], minlength=2000)
        lib.f_v0[name] = np.divide(np.bincount(key[kv], minlength=2000), n_keep, out=np.zeros(2000), where=n_keep > 0)
    return lib


class EnrichedNuLibrary(nudis.NuLibrary):
    """fastveto.nudis.NuLibrary drawing only the final states of _flag_final_states (enrich=True), w_factor = their
    fraction in the (nucleus, energy) group."""

    def __init__(self, path, flavour=14, enrich=True):
        super().__init__(path, flavour)
        self.enrich = enrich
        if not enrich:
            return
        t = self.t
        has = _flag_final_states(t, len(t["ev_ie"]))
        sel = np.nonzero(t["ev_nu"] == self.fi)[0]
        key = t["ev_tgt"][sel].astype(np.int64) * 1000 + t["ev_ie"][sel]
        n_all = np.bincount(key, minlength=100000)
        n_has = np.bincount(key[has[sel]], minlength=100000)
        self.frac = np.divide(n_has, n_all, out=np.zeros(len(n_all)), where=n_all > 0)
        s2 = sel[has[sel]]
        k2 = t["ev_tgt"][s2].astype(np.int64) * 1000 + t["ev_ie"][s2]
        o = np.argsort(k2, kind="stable")
        self.order = s2[o]
        ks, first, cnt = np.unique(k2[o], return_index=True, return_counts=True)
        self.groups = {int(k): (f, c) for k, f, c in zip(ks, first, cnt)}

    def sample(self, material, E, u, vertex, pdg_nu, rng, v0_only=False):
        out = super().sample(material, E, u, vertex, pdg_nu, rng)
        if self.enrich:
            tgt = np.searchsorted(np.array(self.nuclei), out["nucleus"]) if np.all(np.diff(self.nuclei) > 0) else \
                np.array([self.nuclei.index(c) for c in out["nucleus"]])
            ie = np.searchsorted(self.E, out["E_lib"])
            out["w_factor"] = self.frac[tgt.astype(np.int64) * 1000 + ie]
        else:
            out["w_factor"] = np.ones(len(E))
        return out


# ---------------------------------------------------------------- forced interactions (as fastveto.dis.run)
INTER_KEYS = ("muon", "code", "x", "y", "z", "E", "weight", "pdg_mu", "ux", "uy", "uz", "p_int", "evt_n", "type")
KEYS = tg.__dict__.get("KEYS") or ("x_30", "y_30", "z_30", "px_30", "py_30", "pz_30", "x_82", "y_82", "z_82", "px_82",
                                   "py_82", "pz_82", "kx", "ky", "kz", "weight", "pdg", "evt_n", "type")
_CTX = {}


def _load(ci):
    s, e = _CTX["chunks"][ci]
    d = {k: np.asarray(_CTX["columns"][k][s:e]) for k in KEYS if k in _CTX["columns"]}
    if "kx" not in d:
        for c in "xyz":
            d[f"k{c}"] = np.full(e - s, np.nan, np.float32)
    if "type" not in d:
        d["type"] = np.zeros(e - s, np.int8)
    if "evt_n" not in d:
        d["evt_n"] = np.arange(s, e)
    if _CTX["types"] is not None:                         # only some muon types (zero weight for the others)
        d["weight"] = np.where(np.isin(d["type"], _CTX["types"]), d["weight"], 0)
    cr = crossings(d)
    lib = _CTX["lib"]
    P = np.zeros(len(cr["E"]))
    for mi, mname in enumerate(MATERIAL_NAMES):
        m = cr["material"] == mi
        if m.any():
            P[m] = cr["n_area"][m] * lib.sigma(mname, cr["E"][m]) * 1e-34
    return d, cr, P, np.asarray(d["weight"], np.float64)[cr["muon"]] * P


def _pass1(ci):
    _, cr, _, wP = _load(ci)
    return {int(c): float(wP[cr["code"] == c].sum()) for c in CODES}, len(cr["E"])


def _pass2(job):
    ci, need, w_draw, seed = job
    if not need:
        return None
    rng = np.random.default_rng(seed)
    s = _CTX["chunks"][ci][0]
    d, cr, P, wP = _load(ci)
    out = {k: [] for k in INTER_KEYS}
    for code, k in need.items():
        rows = np.nonzero(cr["code"] == code)[0]
        cdf = np.cumsum(wP[rows])
        pick = rows[np.searchsorted(cdf, rng.random(k) * cdf[-1], side="right").clip(0, len(rows) - 1)]
        u = rng.random(len(pick))[:, None]
        vtx = cr["p0"][pick] + u * (cr["p1"][pick] - cr["p0"][pick])
        mu = cr["muon"][pick]
        for kk, v in (("muon", mu + s), ("code", cr["code"][pick]), ("x", vtx[:, 0]), ("y", vtx[:, 1]), ("z", vtx[:, 2]),
                      ("E", cr["E"][pick]), ("weight", np.full(k, w_draw[code])), ("pdg_mu", d["pdg"][mu]),
                      ("ux", cr["u"][pick, 0]), ("uy", cr["u"][pick, 1]), ("uz", cr["u"][pick, 2]), ("p_int", P[pick]),
                      ("evt_n", d["evt_n"][mu]), ("type", d["type"][mu])):
            out[kk].append(np.asarray(v))
    return {k: np.concatenate(v) for k, v in out.items()}


def run(columns, lib, draws, rng, chunk=2_000_000, workers=1, enrich=True, types=None, verbose=True):
    """Forced interactions in the SBT layers for a dataset (dict of columns, memory-mapped). draws: {code: M}.
    Returns (interactions, particles, totals = interactions per spill per code, unenriched)."""
    register()
    t0 = time.time()
    n = len(columns["weight"])
    _CTX.update(columns=columns, lib=lib, types=types, chunks=[(s, min(s + chunk, n)) for s in range(0, n, chunk)])
    nch = len(_CTX["chunks"])
    pool = mp.get_context("fork").Pool(workers) if workers > 1 else None
    mapper = pool.map if pool else (lambda f, it: list(map(f, it)))
    try:
        r1 = mapper(_pass1, range(nch))
        sums = {c: np.array([x[0][c] for x in r1]) for c in CODES}
        if verbose:
            print(f"  pass 1: {nch} chunks, {sum(x[1] for x in r1):,} crossings, {time.time() - t0:.0f} s; per spill: "
                  + ", ".join(f"{CODES[c]} {sums[c].sum():.4g}" for c in CODES), flush=True)
        per_chunk = {c: rng.multinomial(int(draws.get(c, 0)), sums[c] / sums[c].sum()) if sums[c].sum() > 0 and draws.get(c, 0) > 0
                     else np.zeros(nch, int) for c in CODES}
        w_draw = {c: sums[c].sum() / max(int(draws.get(c, 0)), 1) for c in CODES}
        seeds = rng.integers(0, 2 ** 62, nch)
        jobs = [(ci, {c: int(per_chunk[c][ci]) for c in CODES if per_chunk[c][ci]}, w_draw, int(seeds[ci])) for ci in range(nch)]
        r2 = [x for x in mapper(_pass2, jobs) if x is not None]
        if verbose:
            print(f"  pass 2: {sum(len(x['E']) for x in r2):,} interactions, {time.time() - t0:.0f} s", flush=True)
    finally:
        if pool:
            pool.close()
    inter = {k: np.concatenate([x[k] for x in r2]) for k in INTER_KEYS}
    mat_of = {c: m for c, m, *_ in LAYERS}
    parts = []
    inter["w_factor"] = np.ones(len(inter["E"]))
    inter["component"] = np.zeros(len(inter["E"]), np.int8)
    inter["E_lib"] = np.zeros(len(inter["E"]))
    extra = {}
    for mname in MATERIAL_NAMES:
        m = np.nonzero(np.isin(inter["code"], [c for c, mm in mat_of.items() if mm == mname]))[0]
        if not len(m):
            continue
        u = np.stack([inter["ux"][m], inter["uy"][m], inter["uz"][m]], 1)
        vtx = np.stack([inter["x"][m], inter["y"][m], inter["z"][m]], 1)
        o = lib.sample(mname, inter["E"][m], u, vtx, inter["pdg_mu"][m], rng, v0_only=enrich)
        o["int"] = m[o["int"]]
        wf = o.pop("w_factor")                            # per interaction (1 without enrichment)
        inter["w_factor"][m] = wf if enrich else 1.0
        inter["component"][m], inter["E_lib"][m] = o.pop("component"), o.pop("E_lib")
        o.pop("lib_v0", None)
        for k in ("nucleus", "proc", "charm"):
            if k in o:
                v = o.pop(k)
                extra.setdefault(k, np.zeros(len(inter["E"]), v.dtype))[m] = v
        parts.append(o)
    inter.update(extra)
    inter["weight"] = inter["weight"] * inter["w_factor"]
    inter["active"] = np.isin(inter["code"], ACTIVE)
    part = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.argsort(part["int"], kind="stable")
    part = {k: v[order] for k, v in part.items()}
    totals = {CODES[c]: float(sums[c].sum()) for c in CODES}
    return inter, part, totals


# ---------------------------------------------------------------- own deposits in the LS cells (self-tagging)

def _mass(pdg):
    a = np.abs(np.asarray(pdg))
    return np.array([MASS.get(int(x), 0.0) for x in a]) if len(a) < 1000 else \
        np.select([a == k for k in MASS], list(MASS.values()), 0.0)


def deposits(pid, pdg, v, p, t0, rng, length=120.0, e_min=0.0):
    """Energy deposited in the LS cells by particles (straight lines from v along p over `length` m, scalar or (n,), e.g.
    up to a decay point; pid groups them, e.g. the event). pdg (n,), v (n, 3) start [m], p (n, 3) momentum [GeV], t0 (n,)
    start time [ns]. Returns dict pid, cell, edep [MeV], t."""
    pdg = np.asarray(pdg); v = np.asarray(v, float); p = np.asarray(p, float); t0 = np.asarray(t0, float)
    a = np.abs(pdg)
    pm = np.linalg.norm(p, axis=1)
    keep = pm > 1e-6
    is_ch, is_g, is_n = np.isin(a, CHARGED), a == 22, np.isin(a, NEUTRAL_HADRONS)
    keep &= is_ch | is_g | is_n
    idx = np.nonzero(keep)[0]
    empty = dict(pid=np.zeros(0, np.int64), cell=np.zeros(0, np.int64), edep=np.zeros(0), t=np.zeros(0))
    if not len(idx):
        return empty
    length = np.broadcast_to(np.asarray(length, float), (len(pdg),))[idx]
    length = np.maximum(length, 1e-6)
    P0 = v[idx]; U = p[idx] / pm[idx, None]; Q = P0 + U * length[:, None]
    m = _mass(pdg[idx]); e_tot = np.sqrt(pm[idx] ** 2 + m ** 2); T = e_tot - m
    beta2 = np.clip(pm[idx] ** 2 / e_tot ** 2, 1 / BETA2_CAP, 1.0)
    D = Q - P0
    # pieces of the three layers, ordered along each line
    segs, tas, tbs, lay = [], [], [], []
    for li, (g0, g1) in enumerate(((0.0, G_LS0), (G_LS0, G_LS1), (G_LS1, G_OUT))):
        s_, a_, b_ = layer_pieces(P0, Q, g0, g1)
        if li == 1:                                       # LS: per cell
            s_, a_, b_, face, ring, cseg = split_rings(P0, D, s_, a_, b_)
            cell = sbt.cell_id(face, ring, cseg)
        else:
            cell = np.full(len(s_), -1)
        segs.append(s_); tas.append(a_); tbs.append(b_); lay.append(np.column_stack([np.full(len(s_), li), cell]))
    s = np.concatenate(segs).astype(np.int64)
    if not len(s):
        return empty
    ta, tb = np.concatenate(tas), np.concatenate(tbs)
    li, cell = np.concatenate(lay)[:, 0], np.concatenate(lay)[:, 1]
    o = np.lexsort((ta, s))
    s, ta, tb, li, cell = s[o], ta[o], tb[o], li[o], cell[o]
    seglen = (tb - ta) * length[s]
    is_ls = li == 1
    first = np.r_[True, s[1:] != s[:-1]]
    grp_start = np.maximum.accumulate(np.where(first, np.arange(len(s)), 0))

    def cumsum_before(x):                                  # per line: sum of x over the earlier pieces
        c = np.cumsum(x)
        return c - x - (c - x)[grp_start] if len(x) else x
    edep = np.zeros(len(s))
    # charged: continuous loss, LS visible, capped by the kinetic energy
    ch = is_ch[idx][s]
    if ch.any():
        loss = np.where(is_ls, DEDX_LS, DEDX_AL) * seglen / beta2[s]
        loss = np.where(ch, loss, 0.0)
        before = cumsum_before(loss)
        avail = np.clip(T[s] * 1e3 - before, 0, None)
        edep = np.where(ch & is_ls, np.minimum(loss, avail), edep)
    # photons / neutral hadrons: one interaction point along the ordered pieces
    for kind in ("g", "n"):
        sel = (is_g if kind == "g" else is_n)[idx][s]
        if not sel.any():
            continue
        lam = np.where(is_ls, LAMBDA_CONV_LS if kind == "g" else NEUTRAL_LAMBDA, LAMBDA_CONV_AL if kind == "g" else NEUTRAL_LAMBDA_AL)
        tau = np.where(sel, seglen / lam, 0.0)
        tb_cum = cumsum_before(tau) + tau
        tau_star = rng.exponential(1.0, len(idx))[s]
        before = tb_cum - tau
        conv_here = sel & (before < tau_star) & (tb_cum >= tau_star)
        after = sel & (before >= tau_star) & np.isfinite(before)
        conv_line = np.zeros(len(idx), bool); conv_line[s[conv_here]] = True
        frac = np.where(conv_here, (tau_star - before) / np.maximum(tau, 1e-12), 0.0)
        if kind == "g":                                   # e+ e- pair: 2 MIP along the rest of the LS path, <= E_gamma
            path = np.where(conv_here, (1 - frac) * seglen, np.where(after & conv_line[s], seglen, 0.0))
            dep = np.where(is_ls & sel, 2 * DEDX_LS * path, 0.0)
            cap = np.clip(e_tot[s] * 1e3 - cumsum_before(dep), 0, None)
            edep = np.where(is_ls & sel, np.minimum(dep, cap), edep)
        else:                                             # neutral hadron: NEUTRAL_VIS x T at its interaction (LS only)
            edep = np.where(conv_here & is_ls, NEUTRAL_VIS * T[s] * 1e3, edep)
    ok = is_ls & (edep > e_min) & (cell >= 0)
    if not ok.any():
        return empty
    s, ta, tb, cell, edep = s[ok], ta[ok], tb[ok], cell[ok], edep[ok]
    t = t0[idx][s] + 0.5 * (ta + tb) * length[s] / sbt.C
    pid_ = np.asarray(pid)[idx][s].astype(np.int64)
    # sum per (pid, cell); time = energy-weighted
    key = pid_ * 10_000_000 + cell
    u, inv = np.unique(key, return_inverse=True)
    e_sum = np.bincount(inv, edep)
    t_w = np.bincount(inv, edep * t) / np.maximum(e_sum, 1e-12)
    return dict(pid=u // 10_000_000, cell=u % 10_000_000, edep=e_sum, t=t_w)


def fired(dep, cfg, rng):
    """Cells above the threshold, with the efficiency (sbt config)."""
    ok = (dep["edep"] > cfg["threshold_mev"]) & (rng.random(len(dep["edep"])) < cfg["efficiency"])
    return {k: v[ok] for k, v in dep.items()}


def own_deposits_of_events(sel, inter, part, t_arr, rng, muons=None):
    """Raw LS deposits (pid = index into sel, cell, edep [MeV], t) of the selected interactions sel: every particle of part
    (fastveto format; straight lines from its start, 120 m) except the scattered beam particle (its hits come from the reco
    tracks), plus the parent muon (A or K -> vertex) when muons are given. Times relative to t_arr (n_sel,), the arrival
    of the vertex pair at SST1, on the clock of part["t0"] if present (e.g. evstore, ns from the target), else of the
    interaction (t = 0, products start at |v - x_int| / c)."""
    sel = np.asarray(sel)
    n_int = len(inter["weight"])
    pos = np.full(n_int, -1); pos[sel] = np.arange(len(sel))
    rows = np.nonzero(pos[part["int"]] >= 0)[0]
    x_int = np.stack([inter["x"], inter["y"], inter["z"]], 1)
    v_all = np.asarray(part["v"], float)
    t_ref = np.asarray(inter["t"], float) if ("t0" in part and "t" in inter) else np.zeros(n_int)
    keep = rows[~np.asarray(part["beam"], bool)[rows]]
    P = dict(pid=[pos[part["int"][keep]]], pdg=[np.asarray(part["pdg"])[keep]], v=[v_all[keep]],
             p=[np.asarray(part["p"], float)[keep, :3]], t0=[part_t0(part, inter, keep)], L=[np.full(len(keep), 120.0)])
    if muons is not None and "muon" in inter:
        mrow = np.asarray(inter["muon"])[sel]
        a = np.stack([np.asarray(muons[f"{c}_30"][mrow], float) for c in "xyz"], 1)
        k = np.stack([np.asarray(muons[f"k{c}"][mrow], float) for c in "xyz"], 1)
        vtx = x_int[sel]
        a = np.where((np.isfinite(k[:, 2]) & (k[:, 2] < vtx[:, 2]))[:, None], k, a)
        dv = vtx - a
        L = np.linalg.norm(dv, axis=1)
        E = np.asarray(inter["E"], float)[sel] if "E" in inter else np.full(len(sel), 10.0)
        P["pid"].append(np.arange(len(sel))); P["pdg"].append(np.full(len(sel), 13)); P["v"].append(a)
        P["p"].append(dv / np.maximum(L, 1e-9)[:, None] * E[:, None]); P["t0"].append(t_ref[sel] - L / sbt.C)
        P["L"].append(np.minimum(L, 200.0))
    P = {k_: np.concatenate(x) for k_, x in P.items()}
    d = deposits(P["pid"], P["pdg"], P["v"], P["p"], P["t0"], rng, length=P["L"])
    d["t"] = d["t"] - np.asarray(t_arr, float)[d["pid"]]
    return d


def part_t0(part, inter, rows):
    """Start times of part rows: part["t0"] if present, else light speed from the interaction point (t = 0 there)."""
    if "t0" in part:
        return np.asarray(part["t0"], float)[rows]
    x_int = np.stack([inter["x"], inter["y"], inter["z"]], 1)
    return np.linalg.norm(np.asarray(part["v"], float)[rows] - x_int[np.asarray(part["int"])[rows]], axis=1) / sbt.C


def own_hits_from_part(sel, inter, part, tc, pair, cfg, rng, muons=None):
    """Fired LS cells of the selected interactions (D19 selftag_hits / zone_rule in interactions.evaluate): own_deposits_of_
    events with the arrival time of the first vertex track, then the threshold / efficiency of the sbt cfg."""
    r0 = tc["row"][pair[:, 0]]
    t_arr = part_t0(part, inter, r0) + np.linalg.norm(tc["b"][pair[:, 0]] - tc["v"][pair[:, 0]], axis=1) / sbt.C
    d = fired(own_deposits_of_events(sel, inter, part, t_arr, rng, muons), cfg, rng)
    return dict(event=d["pid"].astype(np.int64), cell=d["cell"].astype(np.int64), t=d["t"])
