#!/usr/bin/env python3
"""Neutrino interactions in the cavern rock: what the target / shield neutrinos (bin/nu_prod.sbatch) knock out of the rock
into the ECN3 cavern, and where it goes (veto planes of a design, the decay vessel, SST1).

Cavern (the transport's geometry, MuonsAndMatter CreateCavern, G4_CONCRETE outside): TCC8 for z < ZT = 18.378 m
(x -3.57 .. 6.43, y -1.70 .. 5.80), ECN3 after it (x -4.57 .. 11.43, y -3.36 .. 10.64). The neutrinos go straight from
where they were made; all lines diverge from the target, so in ECN3 they run INTO the side walls (only backward / wide-angle
products come out), while lines that are in the rock upstream of the TCC8 -> ECN3 step come out of its face at z = ZT
with their forward products pointing at the vessel. Regions (rock, near a cavern surface):
  0 -x wall, 1 +x wall, 2 floor, 3 ceiling: the top D_HAD = 3 m of rock along ECN3, z = ZT .. Z_SST1
  4 the upstream face: rock at z = ZT - D_HAD .. ZT inside the ECN3 cross-section and outside the TCC8 opening
  5 deep face: the same at z = ZT - D_MU .. ZT - D_HAD; only the CC muon is kept (straight, dE/dx = a + b E), the
    "neutrino-induced muons" from the rock upstream
Regions 0-4: the full GENIE final state (library bin/nu_library.py, concrete; K_S / hyperons / pi0 already decayed by
fastveto.decay at their displaced vertices) is placed at the vertex in the rock frame of the nearest surface (g4rock:
rock y < 0, surface y = 0) and transported with Geant4 (FTFP_BERT) out of the surface; products made above the surface
go straight. Exits through the part of the face inside the TCC8 opening (into the tunnel / shield) are dropped.
Draws per flavour in proportion to w x lambda (lambda = n sigma L in the region; sigma: GENIE splines, xsec_tot.npz),
each with weight S_f / M_f per spill.

    python3 bin/nu_rock.py prepare --prod /disk/lhcb_data/sqasim/nu_prod/tail --draws 200000 --out DIR
    python3 bin/nu_rock.py task --index K --ntasks N --lib nu_library_v2.npz --out DIR       (cudabear container: g4rock)
    python3 bin/nu_rock.py combine --design designs/modules/D6_straw_mod.json --out DIR
"""
import _bootstrap  # noqa: F401
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

from veto import geometry as geo
from veto import paths

ZT = 18.378
TCC8 = (-3.57, 6.43, -1.70, 5.80)
ECN3 = (-4.57, 11.43, -3.36, 10.64)
Z_END = geo.Z_SST1
D_HAD, D_MU = 3.0, 60.0
RHO = 2.3                                                     # g / cm3 (G4_CONCRETE)
NA = 6.02214e23
CONCRETE = {1000010010: (0.010, 1.008), 1000060120: (0.001, 12.011), 1000080160: (0.529107, 15.999),
            1000110230: (0.016, 22.990), 1000120240: (0.002, 24.305), 1000130270: (0.033872, 26.982),
            1000140280: (0.337021, 28.086), 1000190390: (0.013, 39.098), 1000200400: (0.044, 40.078),
            1000260560: (0.014, 55.845)}
MU_A, MU_B = 0.46, 9.2e-4                                     # muon dE/ds = a + b E in concrete [GeV/m, 1/m]
FLAV = [14, -14, 12, -12, 16, -16]
NAMES = ["-x wall", "+x wall", "floor", "ceiling", "face (top 3 m)", "face deep (CC muons)"]
NORMALS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, 1]], float)
C = 0.299792458
CHARGE = {11: -1, 13: -1, 211: 1, 321: 1, 2212: 1, 3222: 1, 3112: -1, 3312: -1, 3334: -1, 15: -1}


def charge_of(pdg):
    a = np.abs(pdg)
    q = np.array([CHARGE.get(int(x), 0) for x in a]) * np.sign(pdg)
    q[a > 1_000_000_000] = 1
    return q


def in_vessel(x, y, z):
    t = np.clip((z - geo.Z_VESSEL_MIN) / (geo.Z_VESSEL_MAX - geo.Z_VESSEL_MIN), 0, 1)
    hx = geo.VESSEL_X_IN + t * (geo.VESSEL_X_OUT - geo.VESSEL_X_IN)
    hy = geo.VESSEL_Y_IN + t * (geo.VESSEL_Y_OUT - geo.VESSEL_Y_IN)
    return (z > geo.Z_VESSEL_MIN) & (z < geo.Z_VESSEL_MAX) & (np.abs(x) < hx) & (np.abs(y) < hy)


def lambda_per_m(E, nu, xs):
    """interactions per metre of concrete for neutrinos of flavour nu at energies E (GENIE splines, all processes)"""
    lam = np.zeros(len(E))
    for tgt, (f, A) in CONCRETE.items():
        m = np.nonzero((xs["nu"] == nu) & (xs["tgt"] == tgt))[0]
        if not len(m):
            raise RuntimeError(f"no spline for nu {nu} on {tgt}: rerun bin/genie_splines.py")
        n_per_m3 = RHO * f / A * NA * 1e6
        lam += n_per_m3 * np.interp(np.log(E), np.log(xs["E"]), xs["sigma"][m[0]]) * 1e-4      # cm2 -> m2
    return lam


def clip_box(P, u, lo, hi):
    """parameter interval [s0, s1] (s >= 0) of the lines P + s u inside the axis-aligned box lo .. hi (s1 < s0: none)"""
    s0, s1 = np.zeros(len(P)), np.full(len(P), np.inf)
    for k in range(3):
        with np.errstate(divide="ignore", invalid="ignore"):
            a, b = (lo[k] - P[:, k]) / u[:, k], (hi[k] - P[:, k]) / u[:, k]
        inside = (P[:, k] >= lo[k]) & (P[:, k] <= hi[k])
        par = u[:, k] == 0
        t0, t1 = np.where(par, np.where(inside, -np.inf, np.inf), np.minimum(a, b)), np.where(par, np.where(inside, np.inf, -np.inf), np.maximum(a, b))
        s0, s1 = np.maximum(s0, t0), np.minimum(s1, t1)
    return s0, s1


def segments(P, u):
    """per region: list of (s0, s1) intervals (arrays, s1 <= s0 where empty) of each line inside the region"""
    X0, X1, Y0, Y1 = ECN3
    big = 1e3
    boxes = [((X0 - D_HAD, Y0, ZT), (X0, Y1, Z_END)), ((X1, Y0, ZT), (X1 + D_HAD, Y1, Z_END)),
             ((X0, Y0 - D_HAD, ZT), (X1, Y0, Z_END)), ((X0, Y1, ZT), (X1, Y1 + D_HAD, Z_END))]
    out = [[clip_box(P, u, np.array(lo), np.array(hi))] for lo, hi in boxes]
    # the face: ECN3 cross-section minus the TCC8 opening, for two depth ranges
    for z0, z1 in ((ZT - D_HAD, ZT), (ZT - D_MU, ZT - D_HAD)):
        a0, a1 = clip_box(P, u, np.array([X0, Y0, z0]), np.array([X1, Y1, z1]))
        h0, h1 = clip_box(P, u, np.array([TCC8[0], TCC8[2], -big]), np.array([TCC8[1], TCC8[3], big]))
        # [a0, a1] minus [h0, h1]: the part before the hole and the part after it
        b1 = np.where(h1 > h0, np.minimum(a1, h0), a1)
        c0 = np.where(h1 > h0, np.maximum(a0, h1), a1)
        out.append([(a0, b1), (c0, np.where(h1 > h0, a1, a1 * 0 - 1))])
    return out


def load_rows(d):
    sys.path.insert(0, str(Path(__file__).parent))
    from nu_prepare import load_production
    return load_production(d)


def prepare(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    xs = np.load(paths.interactions_data() / "genie/xsec_tot.npz")
    d = load_rows(a.prod)
    rng = np.random.default_rng(a.seed)
    E_all = np.sqrt(d["px"] ** 2 + d["py"] ** 2 + d["pz"] ** 2)
    summary = {"draws_per_flavour": a.draws, "regions": NAMES, "flavours": {}}
    D = {k: [] for k in ("flav", "E", "ux", "uy", "uz", "vx", "vy", "vz", "region", "w", "comp", "row")}
    for fl in FLAV:
        m = np.nonzero((d["pdg"] == fl) & (d["pz"] > 0))[0]
        P = np.stack([d["x"][m], d["y"][m], d["z"][m]], 1)
        E = E_all[m]
        u = np.stack([d["px"][m], d["py"][m], d["pz"][m]], 1) / E[:, None]
        w = d["w"][m]
        lam_m = lambda_per_m(E, fl, xs)
        segs = segments(P, u)
        pieces = []                                        # (region, s0, s1, expected interactions w lambda)
        for r, lst in enumerate(segs):
            for s0, s1 in lst:
                L = np.clip(s1 - s0, 0, None)
                L[~np.isfinite(L)] = 0
                pieces.append((r, s0, s1, w * lam_m * L))
        R = np.array([p[3] for p in pieces])               # (n_pieces, n_lines)
        S = R.sum()
        per_region = {NAMES[r]: 0.0 for r in range(len(NAMES))}
        for (r, *_), v in zip(pieces, R):
            per_region[NAMES[r]] += float(v.sum())
        summary["flavours"][str(fl)] = {"interactions_per_spill": float(S), "per_region": per_region, "lines": int(len(m))}
        print(f"nu {fl}: {len(m):,} lines, rock interactions per spill {S:.4g}: " +
              ", ".join(f"{k} {v:.3g}" for k, v in per_region.items()), flush=True)
        M = a.draws
        flat = rng.choice(R.size, size=M, p=(R / S).ravel())
        ip, il = np.unravel_index(flat, R.shape)
        s0, s1 = np.empty(M), np.empty(M)
        for k in range(len(pieces)):
            mk = ip == k
            s0[mk], s1[mk] = pieces[k][1][il[mk]], pieces[k][2][il[mk]]
        s = s0 + rng.random(M) * (s1 - s0)
        V = P[il] + s[:, None] * u[il]
        for k, v in (("flav", np.full(M, fl, np.int32)), ("E", E[il]), ("ux", u[il, 0]), ("uy", u[il, 1]), ("uz", u[il, 2]),
                     ("vx", V[:, 0]), ("vy", V[:, 1]), ("vz", V[:, 2]),
                     ("region", np.array([pieces[k][0] for k in range(len(pieces))], np.int8)[ip]),
                     ("w", np.full(M, S / M)), ("comp", d["comp"][m][il]), ("row", m[il])):
            D[k].append(v)
    D = {k: np.concatenate(v) for k, v in D.items()}
    perm = rng.permutation(len(D["E"]))                    # tasks get a mix of flavours and regions
    D = {k: v[perm] for k, v in D.items()}
    np.savez(out / "draws.npz", **D)
    json.dump(summary, open(out / "prepare.json", "w"), indent=1)
    print(f"{len(D['E']):,} draws -> {out / 'draws.npz'}")


def frame(region):
    """local rock frame of each region's surface: rows ex, ey (= normal into the cavern), ez with ex x ey = ez"""
    out = []
    for n in NORMALS[region]:
        ex = np.array([0.0, 0, 1]) if abs(n[2]) < 0.5 else np.array([1.0, 0, 0])
        ez = np.cross(ex, n)
        out.append(np.stack([ex, n, ez]))
    return np.array(out)


def depth_of(V, region):
    X0, X1, Y0, Y1 = ECN3
    return np.select([region == 0, region == 1, region == 2, region == 3],
                     [X0 - V[:, 0], V[:, 0] - X1, Y0 - V[:, 1], V[:, 1] - Y1], ZT - V[:, 2])


def mu_exit(E0, depth, cos_n):
    """energy of a muon after its path to the surface (straight, dE/ds = a + b E); <= 0: stopped"""
    s = depth / np.maximum(cos_n, 1e-6)
    return (E0 + MU_A / MU_B) * np.exp(-MU_B * s) - MU_A / MU_B, s


def task(a):
    out = Path(a.out)
    D = dict(np.load(out / "draws.npz"))
    n = len(D["E"])
    sl = np.array_split(np.arange(n), a.ntasks)[a.index]
    D = {k: v[sl] for k, v in D.items()}
    sys.path.insert(0, str(paths.cpp_build()))
    import g4rock
    from fastveto import nudis
    g4rock.initialize(seed=1000 + a.index, material="G4_CONCRETE", e_cut=1e-3)
    t0 = time.time()
    rng = np.random.default_rng(77 + a.index)
    res = {k: [] for k in ("draw", "pdg", "x", "y", "z", "px", "py", "pz", "t", "beam")}
    info = {k: np.zeros(len(sl)) for k in ("cc", "nucleus", "proc", "charm")}
    for fl in FLAV:
        idx = np.nonzero(D["flav"] == fl)[0]
        if not len(idx):
            continue
        lib = nudis.NuLibrary(a.lib, flavour=fl)
        V = np.stack([D["vx"][idx], D["vy"][idx], D["vz"][idx]], 1)
        U = np.stack([D["ux"][idx], D["uy"][idx], D["uz"][idx]], 1)
        smp = lib.sample("concrete", D["E"][idx], U, V, fl, rng)
        info["cc"][idx] = 1 - smp["component"]; info["nucleus"][idx] = smp["nucleus"]
        info["proc"][idx] = smp["proc"]; info["charm"][idx] = smp["charm"]
        nus = np.isin(np.abs(smp["pdg"]), (12, 14, 16))         # outgoing neutrinos (NC): nothing to transport
        smp = {k: (v[~nus] if k in ("int", "pdg", "beam", "p", "v") else v) for k, v in smp.items()}
        owner = idx[smp["int"]]
        pdg, beam, p4, v3 = smp["pdg"], smp["beam"], smp["p"], smp["v"]
        reg = D["region"][owner]
        tnu = D["vz"][owner] / C
        # deep face: only the CC muon, straight to the face with its energy loss
        deep = reg == 5
        mu = deep & beam & (np.abs(pdg) == 13)
        if mu.any():
            pm = np.linalg.norm(p4[mu, :3], axis=1)
            cz = p4[mu, 2] / pm
            dep = depth_of(v3[mu], np.full(mu.sum(), 5))
            Ex, sx = mu_exit(pm, dep, cz)
            okm = (Ex > 0.2) & (cz > 0)
            k = np.nonzero(mu)[0][okm]
            dirm = p4[k, :3] / pm[okm][:, None]
            pos = v3[k] + sx[okm][:, None] * dirm
            inhole = ((pos[:, 0] > TCC8[0]) & (pos[:, 0] < TCC8[1]) & (pos[:, 1] > TCC8[2]) & (pos[:, 1] < TCC8[3])) | \
                     (pos[:, 0] < ECN3[0]) | (pos[:, 0] > ECN3[1]) | (pos[:, 1] < ECN3[2]) | (pos[:, 1] > ECN3[3])
            k, pos, dirm, Ex = k[~inhole], pos[~inhole], dirm[~inhole], Ex[okm][~inhole]
            for kk, vv in (("draw", sl[owner[k]]), ("pdg", pdg[k]), ("x", pos[:, 0]), ("y", pos[:, 1]), ("z", pos[:, 2]),
                           ("px", dirm[:, 0] * Ex), ("py", dirm[:, 1] * Ex), ("pz", dirm[:, 2] * Ex),
                           ("t", tnu[k] + sx[okm][~inhole] / C), ("beam", np.ones(len(k), bool))):
                res[kk].append(vv)
        # regions 0-4: Geant4 from the vertex, in the frame of the region's surface
        full = np.nonzero(~deep)[0]
        if len(full):
            fr = frame(reg[full])                                   # (n, 3, 3) rows ex, ey, ez
            dep_v = depth_of(V[smp["int"][full]], reg[full])        # depth of the interaction vertex
            foot = V[smp["int"][full]] + dep_v[:, None] * NORMALS[reg[full]]      # the surface point above it
            rel = v3[full] - foot
            loc_v = np.einsum("nij,nj->ni", fr, rel)                # local coordinates (y = height above the surface)
            loc_p = np.einsum("nij,nj->ni", fr, p4[full, :3])
            own = owner[full]
            order = np.lexsort((~beam[full], own))
            full, loc_v, loc_p, own, fr_o, foot_o = full[order], loc_v[order], loc_p[order], own[order], fr[order], foot[order]
            air = loc_v[:, 1] > 0
            # products already above the surface (displaced decays): straight
            ka = full[air]
            for kk, vv in (("draw", sl[own[air]]), ("pdg", pdg[ka]), ("x", v3[ka, 0]), ("y", v3[ka, 1]), ("z", v3[ka, 2]),
                           ("px", p4[ka, 0]), ("py", p4[ka, 1]), ("pz", p4[ka, 2]), ("t", tnu[ka]), ("beam", beam[ka])):
                res[kk].append(vv)
            inj = ~air
            uo, first = np.unique(own[inj], return_index=True)
            ev_frame, ev_foot = fr_o[inj][first], foot_o[inj][first]
            cnt = np.bincount(np.searchsorted(uo, own[inj]), minlength=len(uo))
            off = np.r_[0, np.cumsum(cnt)].astype(np.int64)
            Pv, Pp, Pg, Pb = loc_v[inj], loc_p[inj], pdg[full][inj], beam[full][inj]
            for c0 in range(0, len(uo), 500):
                c1 = min(len(uo), c0 + 500)
                s_ = slice(off[c0], off[c1])
                r = g4rock.inject(offsets=off[c0:c1 + 1] - off[c0], pdg=Pg[s_].astype(np.int32), px=Pp[s_, 0], py=Pp[s_, 1], pz=Pp[s_, 2],
                                  x=Pv[s_, 0], y=Pv[s_, 1], z=Pv[s_, 2])
                ex = {k: np.asarray(v) for k, v in r["exits"].items()}
                if not len(ex["ev"]):
                    continue
                e = ex["ev"] + c0
                Fm, Ft = ev_frame[e], ev_foot[e]
                lp = np.stack([ex["x"], np.zeros(len(e)), ex["z"]], 1)
                gpos = Ft + np.einsum("nji,nj->ni", Fm, lp)
                gmom = np.einsum("nji,nj->ni", Fm, np.stack([ex["px"], ex["py"], ex["pz"]], 1))
                dr = sl[uo[e]]
                for kk, vv in (("draw", dr), ("pdg", ex["pdg"]), ("x", gpos[:, 0]), ("y", gpos[:, 1]), ("z", gpos[:, 2]),
                               ("px", gmom[:, 0]), ("py", gmom[:, 1]), ("pz", gmom[:, 2]), ("t", D["vz"][uo[e]] / C + ex["t"]),
                               ("beam", ex["track"] == 1)):
                    res[kk].append(vv)
        del lib
        print(f"task {a.index}: nu {fl}: {len(idx)} draws, {time.time() - t0:.0f} s", flush=True)
    R = {k: np.concatenate(v) if v else np.zeros(0) for k, v in res.items()}
    # exits through the TCC8 opening of the face go into the tunnel / shield, outside the ECN3 cross-section into the
    # side rock; exits of the side walls upstream of the step are in TCC8: none of them reach ECN3
    onface = np.abs(R["z"] - ZT) < 1e-3
    hole = onface & (((R["x"] > TCC8[0]) & (R["x"] < TCC8[1]) & (R["y"] > TCC8[2]) & (R["y"] < TCC8[3]))
                     | (R["x"] < ECN3[0] - 1e-3) | (R["x"] > ECN3[1] + 1e-3) | (R["y"] < ECN3[2] - 1e-3) | (R["y"] > ECN3[3] + 1e-3))
    hole |= ~onface & (R["z"] < ZT)
    R = {k: v[~hole] for k, v in R.items()}
    (out / "parts").mkdir(exist_ok=True)
    np.savez(out / f"parts/part_{a.index:04d}.npz", draws=sl, **{f"x_{k}": v for k, v in R.items()},
             **{f"i_{k}": v for k, v in info.items()}, sec=time.time() - t0)
    print(f"task {a.index}: {len(sl)} draws, {len(R['pdg'])} particles out, {time.time() - t0:.0f} s")


def combine(a):
    from veto import design as dmod
    out = Path(a.out)
    D = dict(np.load(out / "draws.npz"))
    files = sorted(glob.glob(str(out / "parts/part_*.npz")))
    info = {k: np.zeros(len(D["E"])) for k in ("cc", "nucleus", "proc", "charm")}
    done = np.zeros(len(D["E"]), bool)
    X = {k: [] for k in ("draw", "pdg", "x", "y", "z", "px", "py", "pz", "t", "beam")}
    for f in files:
        z = np.load(f)
        done[z["draws"]] = True
        for k in info:
            info[k][z["draws"]] = z[f"i_{k}"]
        for k in X:
            X[k].append(z[f"x_{k}"])
    X = {k: np.concatenate(v) for k, v in X.items()}
    if not done.all():
        print(f"warning: {done.sum()} of {len(done)} draws processed; totals scaled per flavour")
    des = dmod.load(a.design)
    planes = [(pl.name, float(pl.z), min(float(pl.to_global(0.0)), float(pl.to_global(pl.u_max))),
               max(float(pl.to_global(0.0)), float(pl.to_global(pl.u_max))), float(pl.y_lo), float(pl.y_hi)) for pl in des.planes]
    # weight per draw (rescaled per flavour if some tasks are missing)
    wd = D["w"].copy()
    for fl in FLAV:
        m = D["flav"] == fl
        if m.any() and done[m].any():
            wd[m] *= m.sum() / done[m].sum()
    wd[~done] = 0
    ow = X["draw"]
    if getattr(a, "store_out", None):              # every particle out of the rock, for bin/evstore_build.py rock
        reg = D["region"][ow].astype(np.int64)
        np.savez(a.store_out, n_seeds=1, pot_per_spill=4e13, design=des.name, draw=ow, w=wd[ow], surf=(reg == 2).astype(np.int8),
                 region=reg, vx=D["vx"][ow], vy=D["vy"][ow], vz=D["vz"][ow], mu_row=np.full(len(ow), -1),
                 t_mu=(D["vz"][ow] - geo.Z_TARGET) / 0.299792458, E_mu=D["E"][ow], flav=D["flav"][ow],
                 **{k: X[k] for k in ("pdg", "x", "y", "z", "px", "py", "pz", "t", "beam")})
        print(f"-> {a.store_out}: {len(ow):,} particles out of the rock")
    pdg = X["pdg"].astype(np.int64)
    q = charge_of(pdg)
    pos = np.stack([X["x"], X["y"], X["z"]], 1)
    mom = np.stack([X["px"], X["py"], X["pz"]], 1)
    pm = np.linalg.norm(mom, axis=1)
    dirv = mom / np.maximum(pm, 1e-12)[:, None]
    charged = q != 0
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (geo.Z_SST1 - pos[:, 2]) / dirv[:, 2]
    xs_, ys_ = pos[:, 0] + s * dirv[:, 0], pos[:, 1] + s * dirv[:, 1]
    sst = charged & (s > 0) & (np.abs(xs_) < geo.SST1_HALF_X) & (np.abs(ys_) < geo.SST1_HALF_Y) & (pm > 1.0)
    entv = np.zeros(len(pos), bool)
    cand = np.nonzero(charged & (dirv[:, 2] > 0))[0]
    for st in np.arange(0.05, 70, 0.1):
        if not len(cand):
            break
        p_ = pos[cand] + st * dirv[cand]
        inside = in_vessel(p_[:, 0], p_[:, 1], p_[:, 2])
        entv[cand[inside]] = True
        cand = cand[(p_[:, 2] < geo.Z_VESSEL_MAX + 1) & ~inside]
    nd = len(D["E"])
    nsst = np.bincount(ow[sst], minlength=nd)
    npl = np.bincount(ow[sst & (q > 0)], minlength=nd); nmi = np.bincount(ow[sst & (q < 0)], minlength=nd)
    rows = []
    groups = [("all", np.ones(nd, bool))] + [(f"nu {fl}", D["flav"] == fl) for fl in FLAV] + \
             [(NAMES[r], D["region"] == r) for r in range(len(NAMES))] + [("CC", info["cc"] == 1), ("NC", info["cc"] == 0)]
    res = {}
    for name, g in groups:
        gw = wd * g
        ww = gw[ow]
        r_ = {"interactions": float(gw.sum()), "out": float(ww.sum()), "out_charged": float((ww * charged).sum()),
              "vessel_charged": float((ww * entv).sum()), "sst1_charged_p1": float((ww * sst).sum()),
              "sst1_muons_p1": float((ww * (sst & (np.abs(pdg) == 13))).sum()),
              "sst1_2tracks": float((gw * (nsst >= 2)).sum()), "sst1_2opp": float((gw * ((npl >= 1) & (nmi >= 1))).sum()),
              "sst1_2tracks_err": float(np.sqrt((gw ** 2 * (nsst >= 2)).sum())),
              "sst1_charged_p1_err": float(np.sqrt(np.sum(np.bincount(ow, ww * sst, nd) ** 2)))}
        pl = []
        for (_, zp, x0, x1, y0, y1) in planes:
            with np.errstate(divide="ignore", invalid="ignore"):
                sp = (zp - pos[:, 2]) / dirv[:, 2]
            xx, yy = pos[:, 0] + sp * dirv[:, 0], pos[:, 1] + sp * dirv[:, 1]
            hit = (sp > 0) & np.isfinite(sp) & (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
            pl.append([float((ww * (hit & charged)).sum()), float((ww * (hit & (pdg == 22))).sum()), float((ww * (hit & (np.abs(pdg) == 2112))).sum())])
        r_["planes"] = pl
        res[name] = r_
        rows.append((name, r_))
    S15 = 6e20 / 4e13
    print(f"design {des.name}; per spill (x {S15:.2g} spills = 15 years)")
    print(f"  {'':22s}{'interactions':>13s}{'out':>11s}{'charged':>11s}{'-> vessel':>11s}{'SST1 ch>1GeV':>14s}{'(muons)':>10s}{'>=2 SST1':>11s}{'opp.':>10s}")
    for name, r_ in rows:
        print(f"  {name:22s}{r_['interactions']:13.3g}{r_['out']:11.3g}{r_['out_charged']:11.3g}{r_['vessel_charged']:11.3g}"
              f"{r_['sst1_charged_p1']:14.3g}{r_['sst1_muons_p1']:10.3g}{r_['sst1_2tracks']:11.3g}{r_['sst1_2opp']:10.3g}")
    print("\n  veto plane (all)                     charged / spill   photons / spill   neutrons / spill")
    for (name, *_), v in zip(planes, res["all"]["planes"]):
        print(f"  {name:36s} {v[0]:12.3g} {v[1]:17.3g} {v[2]:17.3g}")
    json.dump({"design": des.name, "spills_15y": S15, "groups": res, "planes": [p[0] for p in planes],
               "draws_done": int(done.sum()), "draws": int(nd)}, open(out / f"nu_rock_{des.name}.json", "w"), indent=1)
    keep = np.zeros(nd, bool); keep[ow[entv | sst]] = True
    km = keep[ow]
    np.savez(out / f"nu_rock_{des.name}_draws.npz", w=wd[ow[km]], draw=ow[km], pdg=pdg[km], x=pos[km, 0], y=pos[km, 1], z=pos[km, 2],
             px=mom[km, 0], py=mom[km, 1], pz=mom[km, 2], t=X["t"][km], beam=X["beam"][km], flav=D["flav"][ow[km]],
             region=D["region"][ow[km]], cc=info["cc"][ow[km]], vx=D["vx"][ow[km]], vy=D["vy"][ow[km]], vz=D["vz"][ow[km]])
    print(f"-> {out}/nu_rock_{des.name}.json, _draws.npz")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--prod", required=True); p.add_argument("--draws", type=int, default=200000)
    p.add_argument("--seed", type=int, default=3); p.add_argument("--out", required=True)
    t = sub.add_parser("task"); t.add_argument("--index", type=int, required=True); t.add_argument("--ntasks", type=int, required=True)
    t.add_argument("--lib", required=True); t.add_argument("--out", required=True)
    c = sub.add_parser("combine"); c.add_argument("--design", required=True); c.add_argument("--out", required=True); c.add_argument("--store-out", default=None)
    a = ap.parse_args()
    {"prepare": prepare, "task": task, "combine": combine}[a.cmd](a)


if __name__ == "__main__":
    main()
