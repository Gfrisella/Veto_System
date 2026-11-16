"""Silicon strip stations at the end of the muon shield (Guglielmo, 12 / 16 Nov 2026; plan_evolution_sbt_shieldend.md B).

Purpose: tag the MS7 V0 background (a muon interacts in the last iron of MS7, a K_L / K_S / Lambda decays in the vessel)
and monitor the shield. Rule (Guglielmo, 12 Nov): reconstruct the V0 vertex in the vessel, trace the V0 back to the shield
end along its visible momentum, and look for its parent there: hits of the outgoing muon / the other products of the
interaction ("a muon with a kink or a vertex of several tracks") compatible in space and time with the back-traced line.

Model (first version, 16 Nov 2026):
  - stations: squares of side `size` (0.20 or 0.30 m, 3 cm deep), centred at (x, y), at the MS7 exit face z = 29.90 m,
    superimposed on the geometry (no field, no material for the transport, as agreed);
  - technology: LHCb-UT-like p-in-n strips, 187.5 um pitch, strips spanning the station, two views (x, y), 25 ns time
    slots (window +-12.5 ns),
    99 % per station, rate limit 1 MHz per strip (SR-389 / LHCb UT numbers, see docs/plan_evolution_sbt_shieldend.md);
  - own hits: charged products of the interaction that start at the exit plane (the event stores' reference surface)
    and cross a station;
  - random hits: rate-density map at the station z from the MS7 parent muons (paths.ms7_parents, straight from
    z = 29.20 m) and the EM debris leaving the shield (results/em_debris em_exit_*: e+- counted, photons x conversion
    probability 7/9 X/X0 of the station);
  - association: back-traced point P = vertex - p_vis / p_z,vis * (z_vtx - z_st); radius r = r0 + theta * (z_vtx - z_st)
    (theta: the direction error of the visible V0 momentum, mostly the missing neutrino of K_L -> pi l nu);
    rule "point": >= 1 hit within r of P on a station; rule "vertex": >= 2 hits (outgoing muon + another product).
  - not yet: the kink (needs a station upstream of the iron for the incoming muon: the parents at z = 29.20 m exist),
    the MS7 field in the stations, delta rays from the station itself.
"""
import json
import math
from functools import lru_cache

import numpy as np

from . import paths

C = 0.299792458          # m / ns
Z_EXIT = 29.90           # MS7 exit face (the evstore reference surface)
Z_PARENTS = 29.20        # MS7 parents plane
X0_SI = 0.0937           # m
# pitch / strip length: LHCb UT outer sensors (187.5 um) with strips spanning the whole station (strip_length None = size):
# the rate allows strips of 1.4-3.6 m (12 Nov), so long strips cut the channels by 8 vs 93.5 um x 5 cm at ~72 kHz / strip
DEFAULT = dict(z=Z_EXIT, size=0.20, stations=[], pitch=187.5e-6, strip_length=None, n_views=2, thickness_si=300e-6,
               efficiency=0.99, time_window_ns=12.5, rule="point", r0=0.02, theta=0.01, max_strip_rate=1e6,
               cost_channel=0.4)   # relative cost per channel (straw channel = 1): ~11 CHF, LHCb-UT ballpark, assumed (C6)
MAP_BIN = 0.02
MAP_X, MAP_Y = 3.0, 2.5   # half extents of the rate map [m]


def config(raw):
    """The design's "shield_strips" block (None: no stations). stations: list of [x, y] centres, or a dict
    {"grid": [x_lo, x_hi, y_lo, y_hi]} filled with stations of `size`."""
    s = raw.get("shield_strips")
    if not s:
        return None
    out = dict(DEFAULT)
    out.update({k: v for k, v in s.items() if k in DEFAULT})
    st = s.get("stations", [])
    if isinstance(st, dict) and "grid" in st:
        x0, x1, y0, y1 = st["grid"]
        h = out["size"]
        xs = np.arange(x0 + h / 2, x1 - h / 2 + 1e-9, h)
        ys = np.arange(y0 + h / 2, y1 - h / 2 + 1e-9, h)
        st = [[float(x), float(y)] for x in xs for y in ys]
    out["stations"] = np.asarray(st, float).reshape(-1, 2)
    if out["strip_length"] is None:
        out["strip_length"] = out["size"]
    return out


def n_channels(cfg):
    per = cfg["n_views"] * math.ceil(cfg["size"] / cfg["pitch"]) * math.ceil(cfg["size"] / cfg["strip_length"])
    return int(per * len(cfg["stations"]))


def cost_kchf(cfg):
    """Placeholder cost model of veto.technology: channels x relative cost x straw-channel kCHF, + 10 % mechanics."""
    from .technology import STRAW_CHANNEL_KCHF
    return 1.1 * n_channels(cfg) * cfg["cost_channel"] * STRAW_CHANNEL_KCHF


def area(cfg):
    return float(len(cfg["stations"]) * cfg["size"] ** 2)


def station_of(cfg, x, y):
    """Index of the station containing (x, y), -1 if none (stations do not overlap)."""
    st = cfg["stations"]
    out = np.full(len(x), -1)
    if not len(st):
        return out
    h = cfg["size"] / 2
    for i, (cx, cy) in enumerate(st):
        m = (np.abs(x - cx) <= h) & (np.abs(y - cy) <= h) & (out < 0)
        out[m] = i
    return out


def conversion_prob(cfg):
    return 7.0 / 9.0 * cfg["n_views"] * cfg["thickness_si"] / X0_SI


@lru_cache(maxsize=4)
def _rate_map_cached(z, conv):
    f = paths.results_dir() / "rate_maps" / f"shield_strips_z{z:.2f}.npz"
    if f.exists():
        d = np.load(f)
        if abs(float(d["conv"]) - conv) < 1e-12:
            return d["mu"], d["e"], d["g"] * conv, d["x_edges"], d["y_edges"]
    xe = np.arange(-MAP_X, MAP_X + 1e-9, MAP_BIN); ye = np.arange(-MAP_Y, MAP_Y + 1e-9, MAP_BIN)
    mu = np.zeros((len(xe) - 1, len(ye) - 1))
    par = np.load(paths.ms7_parents(), mmap_mode="r")
    n = len(par["weight"])
    for s0 in range(0, n, 5_000_000):
        sl = slice(s0, min(n, s0 + 5_000_000))
        x, y, zz = (np.asarray(par[k][sl], float) for k in ("x", "y", "z"))
        px, py, pz = (np.asarray(par[k][sl], float) for k in ("px", "py", "pz"))
        w = np.asarray(par["weight"][sl], float)
        ok = pz > 0
        dz = z - zz
        mu += np.histogram2d(x[ok] + px[ok] / pz[ok] * dz[ok], y[ok] + py[ok] / pz[ok] * dz[ok], [xe, ye], weights=w[ok])[0]
    e = np.zeros_like(mu); g = np.zeros_like(mu)
    ef = sorted((paths.results_dir() / "em_debris").glob("em_exit_n*.npz")) or \
        sorted((paths.PROJECT_DIR / "results_rerun" / "em_debris").glob("em_exit_n*.npz"))
    if ef:
        d = np.load(ef[-1], allow_pickle=True)
        wev = np.asarray(d["w_mu"], float)[np.asarray(d["ev"])]       # weight per spill of the sampled parent muon
        pdg = np.asarray(d["pdg"]); x, y = np.asarray(d["x"], float), np.asarray(d["y"], float)
        ch = np.abs(pdg) == 11
        e = np.histogram2d(x[ch], y[ch], [xe, ye], weights=wev[ch])[0]
        gm = pdg == 22
        g = np.histogram2d(x[gm], y[gm], [xe, ye], weights=wev[gm])[0]
    a = MAP_BIN * MAP_BIN * 1e4                                          # cm^2
    mu, e, g = mu / a, e / a, g / a                                      # Hz / cm^2 (per spill of 1 s)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, mu=mu, e=e, g=g, conv=conv, x_edges=xe, y_edges=ye, z=z, em_file=str(ef[-1]) if ef else "")
    return mu, e, g * conv, xe, ye


def rate_map(cfg):
    """Hit-rate density at the station z [Hz / cm^2]: (muons, e+-, converting photons, x edges, y edges)."""
    return _rate_map_cached(round(float(cfg["z"]), 2), conversion_prob(cfg))


def density(cfg, x, y):
    mu, e, g, xe, ye = rate_map(cfg)
    tot = mu + e + g
    i = np.clip(np.searchsorted(xe, x) - 1, 0, tot.shape[0] - 1)
    j = np.clip(np.searchsorted(ye, y) - 1, 0, tot.shape[1] - 1)
    inside = (x > xe[0]) & (x < xe[-1]) & (y > ye[0]) & (y < ye[-1])
    return np.where(inside, tot[i, j], 0.0)


def station_rates(cfg):
    """Per station: total hit rate [Hz] and the peak rate per strip [Hz] (density x strip area, at the busiest bin)."""
    mu, e, g, xe, ye = rate_map(cfg)
    tot = mu + e + g
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    h = cfg["size"] / 2
    strip_cm2 = cfg["pitch"] * cfg["strip_length"] * 1e4
    out = []
    for cx, cy in cfg["stations"]:
        mx = np.abs(xc - cx) <= h; my = np.abs(yc - cy) <= h
        sub = tot[np.ix_(mx, my)]
        out.append(dict(x=float(cx), y=float(cy), rate_hz=float(sub.sum() * MAP_BIN * MAP_BIN * 1e4),
                        peak_hz_cm2=float(sub.max()) if sub.size else 0.0,
                        peak_strip_hz=float(sub.max() * strip_cm2) if sub.size else 0.0))
    return out


def exit_hits(part_int, part_pdg, part_p, part_v, z=Z_EXIT):
    """Charged particles that start at the MS7 exit plane (the store's reference surface, or in the last 60 cm of iron),
    at the plane z: (event index, x, y). Design independent (cached by interactions.prepare)."""
    from .interactions import charge
    q = charge(part_pdg)
    p, v = part_p[:, :3], part_v
    m = (q != 0) & (p[:, 2] > 0) & (v[:, 2] <= z + 1e-3) & (v[:, 2] > z - 0.6)
    x = v[m, 0] + p[m, 0] / p[m, 2] * (z - v[m, 2]); y = v[m, 1] + p[m, 1] / p[m, 2] * (z - v[m, 2])
    return part_int[m], x, y


def on_stations(cfg, he, hx, hy, rng):
    """The exit hits that fall on a station, after the station efficiency."""
    ok = (station_of(cfg, hx, hy) >= 0) & (rng.random(len(hx)) < cfg["efficiency"])
    return he[ok], hx[ok], hy[ok]


def own_hits(cfg, part_int, part_pdg, part_p, part_v, rng):
    """Hits of the events' own charged particles that start at the exit plane and cross a station: (event, x, y)."""
    return on_stations(cfg, *exit_hits(part_int, part_pdg, part_p, part_v, cfg["z"]), rng)


def tag(cfg, vtx, p_vis, rng, hits=None, n_disc=16):
    """Strip-station tag per candidate. vtx (n, 3) reconstructed vertex, p_vis (n, 3) visible momentum (sum of the pair),
    hits: (event, x, y) of the candidates' own hits (None: random hits only, e.g. signal).
    Returns dict(tagged, on_station, n_own, mu_random) per candidate."""
    n = len(vtx)
    z = cfg["z"]
    dz = vtx[:, 2] - z
    with np.errstate(divide="ignore", invalid="ignore"):
        px = vtx[:, 0] - p_vis[:, 0] / p_vis[:, 2] * dz
        py = vtx[:, 1] - p_vis[:, 1] / p_vis[:, 2] * dz
    r = cfg["r0"] + cfg["theta"] * np.maximum(dz, 0)
    # random hits: mean density over the part of the disc that lies on a station, times the disc area and the window
    ang = rng.uniform(0, 2 * np.pi, (n, n_disc)); rad = r[:, None] * np.sqrt(rng.uniform(0, 1, (n, n_disc)))
    sx, sy = px[:, None] + rad * np.cos(ang), py[:, None] + rad * np.sin(ang)
    on = station_of(cfg, sx.ravel(), sy.ravel()).reshape(n, n_disc) >= 0
    dens = density(cfg, sx.ravel(), sy.ravel()).reshape(n, n_disc) * on
    area_cm2 = np.pi * (r * 100) ** 2
    mu_rand = dens.mean(1) * area_cm2 * 2 * cfg["time_window_ns"] * 1e-9
    n_rand = rng.poisson(np.nan_to_num(mu_rand))
    n_own = np.zeros(n, int)
    if hits is not None and len(hits[0]):
        he, hx, hy = hits
        d2 = (hx - px[he]) ** 2 + (hy - py[he]) ** 2
        n_own = np.bincount(he[d2 <= r[he] ** 2], minlength=n)
    need = 2 if cfg["rule"] == "vertex" else 1
    tagged = (n_own + n_rand) >= need
    return dict(tagged=tagged, on_station=on.any(1), n_own=n_own, mu_random=mu_rand, x=px, y=py, r=r)


def candidates_from_tracks(tr, event, z_vtx, n_events):
    """Vertex (mean of the tracks extrapolated to the vertex z) and visible momentum (sum) per event, from reco-style
    track dicts (states B: bx, by, bz, bpx, bpy, bpz) and the vertex z per track."""
    sx = tr["bpx"] / tr["bpz"]; sy = tr["bpy"] / tr["bpz"]
    x = tr["bx"] - sx * (tr["bz"] - z_vtx); y = tr["by"] - sy * (tr["bz"] - z_vtx)
    cnt = np.maximum(np.bincount(event, minlength=n_events), 1)
    vtx = np.stack([np.bincount(event, x, n_events) / cnt, np.bincount(event, y, n_events) / cnt,
                    np.bincount(event, z_vtx, n_events) / cnt], 1)
    p = np.stack([np.bincount(event, tr[k], n_events) for k in ("bpx", "bpy", "bpz")], 1)
    return vtx, p


def signal_tag(design, tr, event, z_vtx, n_events, rng):
    """Signal candidates vetoed by random strip hits (None if the design has no stations)."""
    cfg = getattr(design, "shield_strips", None)
    if cfg is None or not len(cfg["stations"]):
        return None
    vtx, p = candidates_from_tracks(tr, event, z_vtx, n_events)
    return tag(cfg, vtx, p, rng)["tagged"]


def summary(cfg):
    rs = station_rates(cfg)
    return dict(n_stations=len(cfg["stations"]), area_m2=area(cfg), channels=n_channels(cfg), rule=cfg["rule"],
                theta=cfg["theta"], r0=cfg["r0"], z=cfg["z"],
                max_station_rate_hz=max((x["rate_hz"] for x in rs), default=0.0),
                max_strip_rate_hz=max((x["peak_strip_hz"] for x in rs), default=0.0),
                over_strip_limit=[x for x in rs if x["peak_strip_hz"] > cfg["max_strip_rate"]])


if __name__ == "__main__":
    c = config({"shield_strips": {"stations": {"grid": [-1.7, -1.1, -0.6, 0.6]}}})
    print(json.dumps(summary(c), indent=1, default=float))
