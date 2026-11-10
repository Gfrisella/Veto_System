#!/usr/bin/env python3
"""Apply the cavern-rock interaction library (bin/rock_library.py) to the transported muons: what the muons running in the
wall and the floor send back into the cavern, and where it goes (veto planes of a design, the decay vessel, SST1).

Per stored muon whose straight line from its 30.6 m state meets the wall (x = CAVERN_X_WALL) or the floor (y = CAVERN_Y_FLOOR)
before z = 82 m: entry point, angle to the surface, direction along it, energy |p_30|. Along its path in the rock (straight,
constant energy loss DEDX, until E < 1 GeV, depth 3 m or z = 82 m) the expected number of interactions in each depth bin,
lambda_j = integral of sigma(E) ds (sigma = Geant4 + Pythia parts of the library). A fixed number of draws per seed is spread
over (muon, depth bin) in proportion to w lambda_j (w = the muon's weight per spill); each draw takes one library event of the
nearest energy and angle bin, of its part (by the parts' cross sections) and with its depth in that bin, rotated onto the real
surface (library e_y -> surface normal, e_z -> the muon's direction along the surface) and placed at the muon's vertex.
The products go on as straight lines (no field before the spectrometer).

Counted per spill (each draw carries S / M, S = sum w lambda over the seed, M = its draws):
  interactions; particles out of the rock; per veto plane: crossings by charged particles (> 1 MeV) and by photons;
  charged particles entering the vessel; charged with p > 1 GeV reaching SST1 (z = 82.08 m, |x| < 2, |y| < 3);
  interactions with >= 2 charged at SST1 (p > 1 GeV); the same with opposite charges.
Draws that send a charged particle into the vessel or to SST1 are written out (with all their exits) for the reconstruction.

    python3 bin/rock_apply.py <seed dir> --library rock_library.npz --design D6_straw_mod [--seeds 300 --draws 20000] --out rock_apply
"""
import _bootstrap  # noqa: F401
import argparse
import glob
import json
import time

import numpy as np

from veto import design as dmod
from veto import geometry as geo
from veto import muon_types as mt

XW, YF = geo.CAVERN_X_WALL, geo.CAVERN_Y_FLOOR
DEDX = 0.5                                                     # GeV / m in the rock
D_EDGES = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
from veto.config import POT_PER_SPILL, SPILLS_15Y as SPILLS  # noqa: E402  (1.5e7)
C = 0.299792458
CHARGED = (11, 13, 211, 321, 2212, 3222, 3112, 3312, 3334)
CHARGE = {11: -1, 13: -1, 211: 1, 321: 1, 2212: 1, 3222: 1, 3112: -1, 3312: -1, 3334: -1}


def charge_of(pdg):
    a = np.abs(pdg)
    q = np.array([CHARGE.get(int(x), 0) for x in a]) * np.sign(pdg)
    q[a > 1_000_000_000] = 1                                   # light ions: positive
    return q


in_vessel = geo.inside_vessel


class Library:
    def __init__(self, path):
        z = np.load(path)
        self.E, self.A = z["E"], z["alpha"]
        self.sig_g4, self.sig_p6 = z["sig_g4"], z["sig_p6"]
        self.ev = {k[3:]: z[k] for k in z.files if k.startswith("ev_")}
        self.x = {k[2:]: z[k] for k in z.files if k.startswith("x_")}
        self.x["ekin"] = self.x["ekin"].astype(np.float64)
        for k in ("x", "z", "t"):
            self.x[k] = self.x[k].astype(np.float64)
        self.meta = json.loads(str(z["meta"]))
        db = np.clip(np.searchsorted(D_EDGES, self.ev["depth"]) - 1, 0, len(D_EDGES) - 2)
        key = ((self.ev["ie"].astype(np.int64) * 16 + self.ev["ia"]) * 2 + self.ev["comp"]) * 16 + db
        self.order = np.argsort(key, kind="stable")
        ks, first, cnt = np.unique(key[self.order], return_index=True, return_counts=True)
        self.cells = {int(k): (int(f), int(c)) for k, f, c in zip(ks, first, cnt)}

    def sigma(self, E):
        """sigma per metre of rock of each part (Geant4, Pythia), log-linear in E, clamped to the library range."""
        x = np.log(np.clip(E, self.E[0], self.E[-1]))
        lg = np.log(self.E)
        return np.interp(x, lg, self.sig_g4), np.interp(x, lg, self.sig_p6)

    def pick(self, ie, ia, comp, db, rng):
        out = np.full(len(ie), -1, np.int64)
        key = ((ie.astype(np.int64) * 16 + ia) * 2 + comp) * 16 + db
        for k in np.unique(key):
            m = np.nonzero(key == k)[0]
            cell = self.cells.get(int(k))
            if cell is None:                                     # empty cell: the next shallower / deeper depth bin
                for dk in (-1, 1, -2, 2, -3, 3):
                    cell = self.cells.get(int(k) + dk)
                    if cell is not None:
                        break
            if cell is None:
                continue
            f, c = cell
            out[m] = self.order[f + rng.integers(0, c, len(m))]
        return out


def rock_entries(d):
    """Muons of one seed file whose line from A meets the wall or the floor before z = 82 m: row, entry point, direction,
    surface normal (0 wall, 1 floor), grazing angle, energy."""
    r = np.nonzero((d["type"] <= 6) & (d["z_30"] > mt.REACH_30))[0]
    A = np.stack([d[f"{c}_30"][r] for c in "xyz"], 1).astype(np.float64)
    P = np.stack([d[f"p{c}_30"][r] for c in "xyz"], 1).astype(np.float64)
    E = np.linalg.norm(P, axis=1)
    u = P / np.maximum(E, 1e-12)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        sw = np.where(u[:, 0] < 0, (XW - A[:, 0]) / u[:, 0], np.inf)
        sf = np.where(u[:, 1] < 0, (YF - A[:, 1]) / u[:, 1], np.inf)
    wall = sw <= sf
    s = np.minimum(sw, sf)
    ok = (s > 0) & np.isfinite(s) & (A[:, 2] + s * u[:, 2] < geo.Z_STATE_B) & (E > 1.0)
    r, A, u, E, wall, s = r[ok], A[ok], u[ok], E[ok], wall[ok], s[ok]
    P_e = A + s[:, None] * u
    alpha = np.where(wall, np.arcsin(np.clip(-u[:, 0], 1e-6, 1)), np.arcsin(np.clip(-u[:, 1], 1e-6, 1)))
    return r, P_e, u, wall, alpha, E


def expected(P_e, u, alpha, E0, lib, dedx):
    """lambda (n, n_depth_bins): expected interactions per muon in each depth bin, and the energy at each bin's middle."""
    n, nb = len(E0), len(D_EDGES) - 1
    lam, Emid, smid = np.zeros((n, nb)), np.zeros((n, nb)), np.zeros((n, nb))
    sa = np.sin(alpha)
    s_stop = (E0 - 1.0) / dedx                                   # path until E = 1 GeV
    s_z = np.where(u[:, 2] > 0, (geo.Z_STATE_B - P_e[:, 2]) / np.maximum(u[:, 2], 1e-9), np.inf)    # path until z = 82 m
    s_max = np.minimum(s_stop, s_z)
    for j in range(nb):
        s0, s1 = D_EDGES[j] / sa, np.minimum(D_EDGES[j + 1] / sa, s_max)
        L = np.clip(s1 - s0, 0, None)
        sm = s0 + L / 2
        Em = np.maximum(E0 - dedx * sm, 1.0)
        g4, p6 = lib.sigma(Em)
        lam[:, j], Emid[:, j], smid[:, j] = (g4 + p6) * L, Em, sm
    return lam, Emid, smid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir"); ap.add_argument("--library", required=True); ap.add_argument("--design", default="D6_straw_mod")
    ap.add_argument("--seeds", type=int, default=300); ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--dedx", type=float, default=DEDX); ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--em-out", default=None, help="also write every e+- / gamma leaving the rock (first --em-seeds seeds) here")
    ap.add_argument("--em-seeds", type=int, default=30)
    ap.add_argument("--store-out", default=None, help="every particle out of the rock for every draw (bin/evstore_build.py rock)")
    a = ap.parse_args()
    em_rows = {k: [] for k in ("seed", "w", "surf", "vx", "vy", "vz", "pdg", "x", "y", "z", "px", "py", "pz", "ekin", "t")}
    n_em = 0
    st_rows = {k: [] for k in ("seed", "draw", "w", "surf", "vx", "vy", "vz", "mu_row", "t_mu", "E_mu", "pdg", "x", "y", "z",
                               "px", "py", "pz", "t", "beam")}
    rng = np.random.default_rng(a.seed)
    lib = Library(a.library)
    des = dmod.load(f"designs/modules/{a.design}.json")
    planes = [(pl.name, float(pl.z), min(float(pl.to_global(0.0)), float(pl.to_global(pl.u_max))),
               max(float(pl.to_global(0.0)), float(pl.to_global(pl.u_max))), float(pl.y_lo), float(pl.y_hi)) for pl in des.planes]
    files = sorted(glob.glob(f"{a.dir}/seed_*.npz"))[:a.seeds]
    T = dict(interactions=0.0, out=0.0, out_charged=0.0, vessel_charged=0.0, sst1_charged_p1=0.0, sst1_2tracks=0.0, sst1_2opp=0.0,
             plane_charged=np.zeros(len(planes)), plane_photon=np.zeros(len(planes)), plane_neutron=np.zeros(len(planes)),
             interactions_wall=0.0, interactions_floor=0.0, interactions_p6=0.0)
    T2 = dict(sst1_charged_p1=0.0, sst1_2tracks=0.0, vessel_charged=0.0)          # sum of squared weights per seed (errors)
    keep_rows = {k: [] for k in ("seed", "w", "surf", "vx", "vy", "vz", "pdg", "x", "y", "z", "px", "py", "pz", "t", "draw")}
    t0 = time.time()
    n_draw_tot = 0
    for n, f in enumerate(files):
        d = np.load(f)
        r, P_e, u, wall, alpha, E0 = rock_entries(d)
        if not len(r):
            continue
        w = d["weight"][r].astype(np.float64)
        lam, Emid, smid = expected(P_e, u, alpha, E0, lib, a.dedx)
        wl = w[:, None] * lam
        S = wl.sum()
        T["interactions"] += S
        T["interactions_wall"] += wl[wall].sum(); T["interactions_floor"] += wl[~wall].sum()
        M = a.draws
        flat = rng.choice(wl.size, size=M, p=(wl / S).ravel())
        i, j = np.unravel_index(flat, wl.shape)
        wd = S / M
        Ed = Emid[i, j]
        g4, p6 = lib.sigma(Ed)
        comp = (rng.random(M) * (g4 + p6) < p6).astype(np.int64)
        T["interactions_p6"] += wd * comp.sum()
        ie = np.abs(np.log(Ed)[:, None] - np.log(lib.E)[None]).argmin(1)
        ia = np.abs(np.log(alpha[i])[:, None] - np.log(lib.A)[None]).argmin(1)
        ev = lib.pick(ie, ia, comp, j, rng)
        good = ev >= 0
        # frame of each draw: n = surface normal into the cavern, t = the muon's direction along the surface, l = n x t
        nrm = np.where(wall[i][:, None], np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
        uu = u[i]
        tdir = uu - (uu * nrm).sum(1, keepdims=True) * nrm
        tdir /= np.linalg.norm(tdir, axis=1, keepdims=True)
        lat = np.cross(nrm, tdir)
        dep = lib.ev["depth"][np.maximum(ev, 0)].astype(np.float64)
        surf0 = P_e[i] + tdir * (dep / np.tan(alpha[i]))[:, None]          # the surface point above the vertex
        t_mu = P_e[i][:, 2] / C                                               # muon arrival (ns, from the target, beta = 1)
        # exits of every drawn event
        first, cnt = lib.ev["first"][np.maximum(ev, 0)], np.where(good, lib.ev["n"][np.maximum(ev, 0)], 0)
        tot = int(cnt.sum())
        if tot == 0:
            continue
        owner = np.repeat(np.arange(M), cnt)
        rows = np.repeat(first, cnt) + (np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        X = lib.x
        pos = surf0[owner] + X["x"][rows][:, None] * lat[owner] + X["z"][rows][:, None] * tdir[owner]
        mom = (X["px"][rows][:, None] * lat[owner] + X["py"][rows][:, None] * nrm[owner] + X["pz"][rows][:, None] * tdir[owner])
        pdg = X["pdg"][rows].copy()
        beam = X["beam"][rows]
        pdg[beam] = np.where(d["pdg"][r][i][owner][beam] > 0, -13, 13)            # the seed's pdg column is the charge sign
        tt = t_mu[owner] + X["t"][rows]
        ek = X["ekin"][rows]
        q = charge_of(pdg)
        charged = (q != 0) & ~beam
        notbeam = ~beam
        T["out"] += wd * notbeam.sum(); T["out_charged"] += wd * charged.sum()
        if a.store_out:                                             # the design-independent store: every exit
            for k_, v_ in (("seed", np.full(tot, n)), ("draw", owner + n_draw_tot), ("w", np.full(tot, wd)),
                           ("surf", (~wall[i])[owner].astype(np.int8)), ("vx", surf0[owner, 0]), ("vy", surf0[owner, 1]),
                           ("vz", surf0[owner, 2]), ("mu_row", r[i][owner]), ("t_mu", t_mu[owner]), ("E_mu", Ed[owner]),
                           ("pdg", pdg), ("x", pos[:, 0]), ("y", pos[:, 1]), ("z", pos[:, 2]), ("px", mom[:, 0]),
                           ("py", mom[:, 1]), ("pz", mom[:, 2]), ("t", tt), ("beam", beam)):
                st_rows[k_].append(v_)
        if a.em_out and n_em < a.em_seeds:                         # every e+- / gamma out of the rock, for the debris page
            km = notbeam & np.isin(pdg, (22, 11, -11))
            for k_, v_ in (("seed", np.full(km.sum(), n)), ("w", np.full(km.sum(), wd)), ("surf", (~wall[i])[owner][km].astype(np.int8)),
                           ("vx", surf0[owner][km, 0]), ("vy", surf0[owner][km, 1]), ("vz", surf0[owner][km, 2]), ("pdg", pdg[km]),
                           ("x", pos[km, 0]), ("y", pos[km, 1]), ("z", pos[km, 2]), ("px", mom[km, 0]), ("py", mom[km, 1]),
                           ("pz", mom[km, 2]), ("ekin", ek[km]), ("t", tt[km])):
                em_rows[k_].append(v_)
            n_em += 1
        pm = np.linalg.norm(mom, axis=1)
        dirv = mom / np.maximum(pm, 1e-12)[:, None]
        # veto planes (constant z): crossing point of the straight line, inside the plane's rectangle
        for k, (_, zp, x0, x1, y0, y1) in enumerate(planes):
            with np.errstate(divide="ignore", invalid="ignore"):
                s = (zp - pos[:, 2]) / dirv[:, 2]
            ok = (s > 0) & np.isfinite(s)
            xx, yy = pos[:, 0] + s * dirv[:, 0], pos[:, 1] + s * dirv[:, 1]
            hit = ok & (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1) & (yy > YF) & (xx > XW) & notbeam
            T["plane_charged"][k] += wd * (hit & charged).sum()
            T["plane_photon"][k] += wd * (hit & (pdg == 22)).sum()
            T["plane_neutron"][k] += wd * (hit & (np.abs(pdg) == 2112)).sum()
        # the vessel: does the line enter it (sampled every 10 cm for 60 m)
        entv = np.zeros(len(pos), bool)
        cand = np.nonzero(charged)[0]
        for st in np.arange(0.05, 60, 0.1):
            if not len(cand):
                break
            p_ = pos[cand] + st * dirv[cand]
            inside = in_vessel(p_[:, 0], p_[:, 1], p_[:, 2])
            entv[cand[inside]] = True
            alive = (p_[:, 2] < geo.Z_VESSEL_MAX + 1) & (p_[:, 0] > XW - 0.01) & (p_[:, 1] > YF - 0.01) & ~inside
            cand = cand[alive]
        T["vessel_charged"] += wd * entv.sum()
        # SST1
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (geo.Z_SST1 - pos[:, 2]) / dirv[:, 2]
        xs, ys = pos[:, 0] + s * dirv[:, 0], pos[:, 1] + s * dirv[:, 1]
        sst = charged & (s > 0) & (np.abs(xs) < geo.SST1_HALF_X) & (np.abs(ys) < geo.SST1_HALF_Y) & (pm > 1.0)
        T["sst1_charged_p1"] += wd * sst.sum()
        nsst = np.bincount(owner[sst], minlength=M)
        nplus = np.bincount(owner[sst & (q > 0)], minlength=M); nminus = np.bincount(owner[sst & (q < 0)], minlength=M)
        T["sst1_2tracks"] += wd * (nsst >= 2).sum(); T["sst1_2opp"] += wd * ((nplus >= 1) & (nminus >= 1)).sum()
        T2["sst1_charged_p1"] += (wd * sst.sum()) ** 2; T2["sst1_2tracks"] += (wd * (nsst >= 2).sum()) ** 2
        T2["vessel_charged"] += (wd * entv.sum()) ** 2
        # keep the draws that matter for the reconstruction
        keepd = np.zeros(M, bool); keepd[owner[entv | sst]] = True
        km = keepd[owner]
        for k_, v_ in (("seed", np.full(km.sum(), n)), ("w", np.full(km.sum(), wd)), ("surf", (~wall[i])[owner][km].astype(np.int8)),
                       ("vx", surf0[owner][km, 0]), ("vy", surf0[owner][km, 1]), ("vz", surf0[owner][km, 2]), ("pdg", pdg[km]),
                       ("x", pos[km, 0]), ("y", pos[km, 1]), ("z", pos[km, 2]), ("px", mom[km, 0]), ("py", mom[km, 1]), ("pz", mom[km, 2]),
                       ("t", tt[km]), ("draw", owner[km] + n_draw_tot)):
            keep_rows[k_].append(v_)
        n_draw_tot += M
        if n % 25 == 0:
            print(f"{n + 1}/{len(files)} seeds, {time.time() - t0:.0f} s", flush=True)
    S = len(files)
    per = lambda v: v / S
    print(f"\n{S} seeds (spills), {a.draws} draws each, design {a.design}, dE/dx {a.dedx} GeV/m. Per spill (x {SPILLS:.1e} = 15 years):")
    rows = [("muon-nuclear interactions in the rock (top 3 m)", T["interactions"]), ("   in the wall", T["interactions_wall"]),
            ("   in the floor", T["interactions_floor"]), ("   Pythia part (nu > nu_x)", T["interactions_p6"]),
            ("particles out of the rock (above the library thresholds)", T["out"]), ("   charged", T["out_charged"]),
            ("charged entering the vessel", T["vessel_charged"]), ("charged with p > 1 GeV reaching SST1", T["sst1_charged_p1"]),
            ("interactions with >= 2 charged (p > 1 GeV) at SST1", T["sst1_2tracks"]), ("   with opposite charges", T["sst1_2opp"])]
    for name, v in rows:
        err = ""
        for k in T2:
            if v is T[k]:
                err = f" +- {np.sqrt(T2[k]) / S:.2g}"
        print(f"  {name:58s} {per(v):10.3g}{err:12s} {per(v) * SPILLS:10.3g}")
    print("\n  veto plane                          charged / spill   photons / spill   neutrons / spill")
    for k, (name, zp, *_r) in enumerate(planes):
        print(f"  {name:36s} {per(T['plane_charged'][k]):12.3g} {per(T['plane_photon'][k]):17.3g} {per(T['plane_neutron'][k]):17.3g}")
    out = {k: (per(v).tolist() if isinstance(v, np.ndarray) else per(v)) for k, v in T.items()}
    out.update(seeds=S, draws=a.draws, design=a.design, dedx=a.dedx, planes=[p[0] for p in planes], plane_z=[p[1] for p in planes],
               library=a.library, library_meta=lib.meta)
    json.dump(out, open(a.out + ".json", "w"), indent=1)
    np.savez(a.out + "_draws.npz", **{k: np.concatenate(v) if v else np.zeros(0) for k, v in keep_rows.items()})
    print(f"-> {a.out}.json, {a.out}_draws.npz ({time.time() - t0:.0f} s)")
    if a.store_out:
        np.savez(a.store_out, n_seeds=S, pot_per_spill=POT_PER_SPILL, design=a.design,
                 **{k: np.concatenate(v) if v else np.zeros(0) for k, v in st_rows.items()})
        print(f"-> {a.store_out}: {sum(len(v) for v in st_rows['pdg']):,} particles out of the rock")
    if a.em_out:                                                    # weight per spill = w / n_seeds
        np.savez(a.em_out, n_seeds=n_em, pot_per_spill=POT_PER_SPILL, design=a.design,
                 **{k: np.concatenate(v) if v else np.zeros(0) for k, v in em_rows.items()})
        print(f"-> {a.em_out}: {sum(len(v) for v in em_rows['pdg']):,} e / gamma from {n_em} seeds")


if __name__ == "__main__":
    main()
