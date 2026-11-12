"""Veto designs: a UBT at the vessel entrance plus fins on the left / right side, and detector settings.

A design is a JSON file (veto/designs/*.json):

    {"name": "D2_default",
     "ubt": {"y_lo": -1.35, "y_hi": 1.35, "cell_x": 0.1, "cell_y": 0.1},
     "fins": [{"side": "right", "z": 30.6, "length": 3, "y_lo": -3, "y_hi": 3, "tilt": 0,
               "cell_u": 0.1, "cell_y": 0.1}, ...],
     "detector": {"efficiency": 0.99, "sigma_t": 0.5e-9}}

"stations": 2 (or 3) in the "ubt" or a fin entry puts identical layers behind each other, read out as an OR:
efficiency 1 - (1 - e)^n, n times the channels, cost and material. A muon leaves its hit in every station at the
same place, so random hits (signal loss) are those of one station; rates per channel are unchanged.

Detector technology (technology.py: straw, fiber, pixel, tile): "technology" in "detector" applies
to every fin, "technology" in a fin overrides it for that fin. A technology sets the plane's
efficiency, timing, cell size, channels, material and cost (kCHF); explicit "efficiency" /
"sigma_t" / cell sizes next to it win. The UBT is fixed: Liverpool straws, vertical (technology.py),
its cell sizes in the JSON are ignored. Without a technology a fin is the generic detector of
before (efficiency and sigma_t of "detector", cells as given, no cost).

Sides are seen from the target looking downstream: right = -x (cavern wall at x = -4.56 m),
left = +x. A fin touches the side wall at |x| = X(z) and reaches out by `length`; its local
coordinate u is the distance from that inner edge. Only untilted fins (tilt = 0) for now.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import geometry as geo
from . import muon_types as mt
from . import technology as tech

DESIGN_DIR = Path(__file__).resolve().parents[2] / "designs"
SIDES = {"right": -1, "left": 1}


@dataclass
class Plane:
    """One detector plane at fixed z. Local coordinates (u, y), cells of size (cell_u, cell_y).

    UBT: u = x + 0.5. Fin: u = sign * x - X(z).
    """
    index: int
    name: str
    kind: str            # "ubt" (straw plane on / before the entrance window), "tiles" (UBT tile plane) or "fin"
    z: float
    sign: int            # fin side (+1 left, -1 right); 0 for the UBT
    u_max: float
    y_lo: float
    y_hi: float
    cell_u: float
    cell_y: float
    efficiency: float = 0.99
    sigma_t: float = 0.5e-9     # [s], Gaussian time smearing
    technology: str = None      # key of tech.TECHNOLOGIES, None = generic detector
    time_window: float = None   # [s] drift-limited technologies: hit time uniform in this full width
    stations: int = 1           # identical layers behind each other (OR): efficiency 1 - (1 - e)^n, n x channels and cost
    # module block (ISSUES A7 / C2, 2026-11-05): a rectangle of identical modules (n_mod_u x n_mod_y) of mod_u x mod_y
    # starting at u_off on the plane's grid; dead borders per module (dead_u / dead_y: (low, high) [m]) and, for pixels,
    # quads of sub x sub with sub_dead borders. None / 0: the continuous plane of before.
    u_off: float = 0.0
    mod_u: float = None
    mod_y: float = None
    dead_u: tuple = (0.0, 0.0)
    dead_y: tuple = (0.0, 0.0)
    sub: float = None
    sub_dead: float = 0.0
    n_modules: int = 0
    x_half: float = None        # "ubt" / "tiles": u = x + x_half - u_off (None: the window half width UBT_HALF_X)

    @property
    def area(self):
        return self.u_max * (self.y_hi - self.y_lo)

    @property
    def n_channels(self):
        """Readout channels: from the technology, or one per cell for the generic detector."""
        if self.technology is None:
            return self.n_cells
        return self.stations * tech.n_channels(tech.TECHNOLOGIES[self.technology], self.u_max, self.y_hi - self.y_lo)

    @property
    def cost(self):
        """kCHF (one model, A4: module electronics + mechanics); nan for the generic detector."""
        if self.technology is None:
            return float("nan")
        if self.n_modules:
            return self.stations * self.n_modules * tech.module_cost(self.technology)
        return self.stations * tech.cost(tech.TECHNOLOGIES[self.technology], self.area)

    @property
    def rel_cost(self):
        """Channels x relative cost per channel (straw channel = 1); nan for the generic detector."""
        if self.technology is None:
            return float("nan")
        return self.n_channels * tech.TECHNOLOGIES[self.technology]["cost_channel"]

    @property
    def x0(self):
        """Material in radiation lengths; nan for the generic detector."""
        if self.technology is None:
            return float("nan")
        return self.stations * tech.x0(tech.TECHNOLOGIES[self.technology])

    @property
    def n_cells(self):
        return int(np.ceil(self.u_max / self.cell_u - 1e-9)) * int(np.ceil((self.y_hi - self.y_lo) / self.cell_y - 1e-9))

    @property
    def half_x(self):
        return geo.UBT_HALF_X if self.x_half is None else self.x_half

    def to_local(self, x):
        if self.kind in ("ubt", "tiles"):
            return x + self.half_x - self.u_off
        return self.sign * x - mt.side_plane_x(self.z) - self.u_off

    def to_global(self, u):
        if self.kind in ("ubt", "tiles"):
            return u + self.u_off - self.half_x
        return self.sign * (u + self.u_off + mt.side_plane_x(self.z))

    def inside(self, u, y):
        """In the plane and not in a dead module border (C2) or pixel-quad border."""
        ok = (u >= 0) & (u <= self.u_max) & (y >= self.y_lo) & (y <= self.y_hi)
        if self.mod_u:
            fu = np.mod(u, self.mod_u)
            fy = np.mod(y - self.y_lo, self.mod_y)
            ok &= (fu >= self.dead_u[0]) & (fu <= self.mod_u - self.dead_u[1])
            ok &= (fy >= self.dead_y[0]) & (fy <= self.mod_y - self.dead_y[1])
            if self.sub:
                su, sy = np.mod(fu - self.dead_u[0], self.sub), np.mod(fy - self.dead_y[0], self.sub)
                ok &= (su >= self.sub_dead) & (su <= self.sub - self.sub_dead)
                ok &= (sy >= self.sub_dead) & (sy <= self.sub - self.sub_dead)
        return ok

    @property
    def active_fraction(self):
        """Sensitive fraction of the plane area (1 for a continuous plane)."""
        if not self.mod_u:
            return 1.0
        fu = (self.mod_u - sum(self.dead_u)) / self.mod_u
        fy = (self.mod_y - sum(self.dead_y)) / self.mod_y
        if self.sub:
            fu *= (self.sub - 2 * self.sub_dead) / self.sub
            fy *= (self.sub - 2 * self.sub_dead) / self.sub
        return fu * fy

    def cell_centre(self, u, y):
        """Digitize: the centre of the cell (u, y) falls in."""
        iu = np.clip(np.floor(u / self.cell_u), 0, None)
        iy = np.clip(np.floor((y - self.y_lo) / self.cell_y), 0, None)
        return (iu + 0.5) * self.cell_u, self.y_lo + (iy + 0.5) * self.cell_y


@dataclass
class Design:
    name: str
    planes: list
    efficiency: float
    sigma_t: float
    raw: dict

    @property
    def area(self):
        return sum(p.area for p in self.planes)

    @property
    def n_cells(self):
        return sum(p.n_cells for p in self.planes)

    @property
    def n_channels(self):
        return sum(p.n_channels for p in self.planes)

    @property
    def cost(self):
        return sum(p.cost for p in self.planes)

    @property
    def rel_cost(self):
        return sum(p.rel_cost for p in self.planes)

    @property
    def ubt_x0(self):
        return self.planes[0].x0


FRAME = "fairship-2026"   # designs with "frame": FRAME give fin z in the FairShip layout (geometry.py); others are old


def validate_modules(raw):
    """Limits for module designs: straw UBT covering the window, fins inside the cavern (right: up to the wall) and
    above FIN_Y_MIN. Raises on violations (a module design is explicit: nothing is cut silently)."""
    u = raw["ubt"]
    if any(tech.get(m["tech"])[0] != "straw" for m in u["modules"]):
        raise ValueError("UBT modules must be straw")
    for f in raw["fins"]:
        if f.get("tilt", 0) != 0:
            raise NotImplementedError("tilted fins are not in the reconstruction yet")
        if not geo.Z_UBT - 1e-6 <= f["z"] <= geo.Z_SST1 + 1e-6:
            raise ValueError(f"fin z = {f['z']} outside {geo.Z_UBT:.2f} .. {geo.Z_SST1:.2f} m")
        y0 = f.get("y0", GRID_Y0["fin"])
        room = -mt.CAVERN_X_WALL - mt.side_plane_x(f["z"])
        for b in module_blocks(f["modules"]):
            mu, my = tech.MODULES[b["tech"]]["size"]
            if y0 + b["iy"] * tech.SLOT < mt.FIN_Y_MIN - 1e-9:
                raise ValueError(f"fin z={f['z']}: module below y = {mt.FIN_Y_MIN} m")
            if f["side"] == "right" and b["iu"] * tech.SLOT + mu > room + 1e-9:
                raise ValueError(f"right fin z={f['z']}: module beyond the cavern wall (room {room:.2f} m)")
    return []


UBT_LAYOUT_DEFAULT = "2026-11-11"


def default_tile_regions(hx, y_lo, y_hi, xin=None, fy=None, fine="tile2", coarse="tile"):
    """Rectangles of the UBT tile plane (11 Nov 2026): 2 x 2 cm^2 tiles in the two flame blocks (xin <= |x| <= hx,
    |y| <= fy), 4 x 4 cm^2 elsewhere. List of dicts x = [x0, x1], y = [y0, y1], technology."""
    xin = geo.UBT_TILES_FLAME_X_IN if xin is None else xin
    fy = geo.UBT_TILES_FLAME_Y if fy is None else fy
    out = [dict(x=[-xin, xin], y=[y_lo, y_hi], technology=coarse)]
    for x0, x1 in ((-hx, -xin), (xin, hx)):
        out += [dict(x=[x0, x1], y=[y_lo, -fy], technology=coarse), dict(x=[x0, x1], y=[-fy, fy], technology=fine),
                dict(x=[x0, x1], y=[fy, y_hi], technology=coarse)]
    return out


def apply_ubt_layout(raw):
    """The UBT layout of a design ("ubt_layout"; default UBT_LAYOUT_DEFAULT, Guglielmo 11 Nov 2026): straw plane the size of
    the window at Z_UBT_STRAW (30.73 m), a 4 x 4 m^2 plane of 4 x 4 cm^2 tiles ("ubt_tiles", technology tile) at Z_UBT_TILES
    (31.13 m), and no fins at the entrance z (they are dropped, with a note). "legacy": the straw UBT on the window
    (31.13 m), no tile plane, fins as given. "ubt_tiles": null in the design = no tile plane. Returns notes."""
    notes = []
    layout = raw.setdefault("ubt_layout", UBT_LAYOUT_DEFAULT)
    if layout == "legacy":
        raw["ubt"].setdefault("z", geo.Z_UBT)
        raw["ubt_tiles"] = None
        return notes
    if layout != UBT_LAYOUT_DEFAULT:
        raise ValueError(f"unknown ubt_layout {layout!r} (use {UBT_LAYOUT_DEFAULT!r} or 'legacy')")
    raw["ubt"].setdefault("z", geo.Z_UBT_STRAW)
    if "ubt_tiles" not in raw:
        raw["ubt_tiles"] = {}
    if raw["ubt_tiles"] is not None:
        t = raw["ubt_tiles"]
        t.setdefault("technology", "tile"); t.setdefault("z", geo.Z_UBT_TILES)
        t.setdefault("half_x", geo.UBT_TILES_HALF_X)
        t.setdefault("y_lo", -geo.UBT_TILES_HALF_Y); t.setdefault("y_hi", geo.UBT_TILES_HALF_Y)
        t.setdefault("regions", default_tile_regions(t["half_x"], t["y_lo"], t["y_hi"]))
    entrance = [f for f in raw["fins"] if abs(f["z"] - geo.Z_VESSEL_MIN) < 0.05]
    if entrance:
        raw["fins"] = [f for f in raw["fins"] if abs(f["z"] - geo.Z_VESSEL_MIN) >= 0.05]
        notes.append(f"UBT layout {layout}: {len(entrance)} fin(s) at the entrance z = {geo.Z_VESSEL_MIN:.2f} m dropped "
                     f"(the 4 x 4 m tile plane covers that region)")
    return notes


def validate(raw):
    """Apply the physical limits: UBT of straw tubes covering at least the vessel entrance, nothing below y = -3.2 m,
    right fins stop at the cavern wall. Returns a list of what was changed."""
    notes = []
    if raw.get("frame") != FRAME:
        # written in the deck layout (vessel 30.6 - 80.6 m): fins sit on the vessel, so they move with it (+0.53 m)
        for f in raw["fins"]:
            f["z"] = round(f["z"] + geo.VESSEL_SHIFT, 6)
        notes.append(f"old-frame design: fins moved by {geo.VESSEL_SHIFT:+.2f} m with the vessel (FairShip layout)")
        raw["frame"] = FRAME
    notes += apply_ubt_layout(raw)
    u = raw["ubt"]
    if "modules" in u:                                   # module design (A7): straw UBT, checked in validate_modules
        return notes + validate_modules(raw)
    if u["y_lo"] > -geo.UBT_HALF_Y or u["y_hi"] < geo.UBT_HALF_Y:
        notes.append("UBT extended to cover the vessel entrance")
        u["y_lo"], u["y_hi"] = min(u["y_lo"], -geo.UBT_HALF_Y), max(u["y_hi"], geo.UBT_HALF_Y)
    if tech.ALIASES.get(u.get("technology"), u.get("technology")) not in (None, "straw"):
        notes.append(f"UBT is straw tubes only ({u['technology']} -> straw)")
    u["technology"] = "straw"                             # Liverpool straws, vertical
    for k in ("cell_x", "cell_y"):
        u.pop(k, None)
    if u["y_lo"] < mt.FIN_Y_MIN:
        notes.append("UBT bottom raised to y = -3.2 m")
        u["y_lo"] = mt.FIN_Y_MIN
    for i, f in enumerate(raw["fins"]):
        if f.get("tilt", 0) != 0:
            raise NotImplementedError("tilted fins are not in the reconstruction yet")
        if f["y_lo"] < mt.FIN_Y_MIN:
            notes.append(f"fin {i}: bottom raised to y = -3.2 m")
            f["y_lo"] = mt.FIN_Y_MIN
        if not geo.Z_UBT - 1e-6 <= f["z"] <= geo.Z_SST1 + 1e-6:
            raise ValueError(f"fin {i}: z = {f['z']} outside {geo.Z_UBT:.2f} .. {geo.Z_SST1:.2f} m")
        if f["side"] == "right":
            room = -mt.CAVERN_X_WALL - mt.side_plane_x(f["z"])
            if f["length"] > room:
                notes.append(f"fin {i}: cut at the cavern wall ({f['length']:.2f} -> {room:.2f} m)")
                f["length"] = room
    return notes


GRID_Y0 = {"ubt": -1.5, "fin": -3.0}   # y of slot row 0 (A7: 50 cm grid; UBT 3 x 1 m straw modules cover +-1.35)


def module_blocks(modules):
    """Modules {"tech", "u", "y" (slot indices), "stations"} -> blocks of identical modules stacked in y (same
    technology, stations, u slot, contiguous): dict(tech, stations, iu, iy, n). Overlapping slots are an error."""
    used, mods = {}, []
    for m in modules:
        key, _ = tech.get(m["tech"])
        nu, ny = (int(round(v / tech.SLOT)) for v in tech.MODULES[key]["size"])
        for a in range(nu):
            for b in range(ny):
                cell = (int(m["u"]) + a, int(m["y"]) + b)
                if cell in used:
                    raise ValueError(f"module {m} overlaps slot {cell} of {used[cell]}")
                used[cell] = m
        mods.append((key, int(m.get("stations", 1)), int(m["u"]), int(m["y"])))
    blocks = []
    for key, st, iu, iy in sorted(mods):
        ny = int(round(tech.MODULES[key]["size"][1] / tech.SLOT))
        b = blocks[-1] if blocks else None
        if b and (b["tech"], b["stations"], b["iu"]) == (key, st, iu) and b["iy"] + b["n"] * ny == iy:
            b["n"] += 1
        else:
            blocks.append(dict(tech=key, stations=st, iu=iu, iy=iy, n=1))
    return blocks


def snap_to_modules(raw, fin_tech=None):
    """A continuous design -> the same design on the 50 cm module grid (A7): each fin / the UBT covered by modules of
    its technology (fin_tech or the design's), its u / y extent rounded OUT to whole modules (right fins: IN at the
    cavern wall), y on the grid from GRID_Y0. Returns a new raw dict ("frame" FRAME, name + "_mod")."""
    raw = json.loads(json.dumps(raw))
    validate(raw)                                         # frame shift, cavern-wall cut, y limits first
    det = raw.get("detector", {}).get("technology")
    out = dict(name=raw["name"] + "_mod", description=raw.get("description", "") + " (on the 50 cm module grid)",
               frame=FRAME, detector=raw.get("detector", {}), fins=[])

    def cover(u_len, y_lo, y_hi, key, y0, room=None):
        mu, my = tech.MODULES[key]["size"]
        nu = max(int(np.ceil(u_len / mu - 1e-6)), 1)
        if room is not None and nu * mu > room + 1e-9:
            nu = max(int(np.floor(room / mu + 1e-9)), 1)
        iy0 = max(int(np.floor((y_lo - y0) / tech.SLOT + 1e-9)), 0)
        ny = max(int(np.ceil((y_hi - (y0 + iy0 * tech.SLOT)) / my - 1e-6)), 1)
        step_u, step_y = int(round(mu / tech.SLOT)), int(round(my / tech.SLOT))
        return [dict(tech=key, u=a * step_u, y=iy0 + b * step_y) for a in range(nu) for b in range(ny)]

    u = raw["ubt"]
    out["ubt"] = dict(y0=GRID_Y0["ubt"], modules=cover(2 * geo.UBT_HALF_X, u["y_lo"], u["y_hi"], "straw", GRID_Y0["ubt"]))
    for f in raw["fins"]:
        key = tech.get(fin_tech or f.get("technology") or det or "straw")[0]
        room = -mt.CAVERN_X_WALL - mt.side_plane_x(f["z"]) if f["side"] == "right" else None
        mods = cover(f["length"], max(f["y_lo"], GRID_Y0["fin"]), f["y_hi"], key, GRID_Y0["fin"], room)
        st = int(f.get("stations", 1))
        for m in mods:
            m["stations"] = st
        out["fins"].append(dict(side=f["side"], z=f["z"], tilt=f.get("tilt", 0), y0=GRID_Y0["fin"], modules=mods))
    return out


def detector_settings(spec, default):
    """Technology, efficiency, sigma_t, cell sizes of one plane (spec: the "ubt" / fin entry,
    or the "detector" block). Explicit numbers win over the technology, which wins over default."""
    out = dict(default)
    if spec.get("technology"):
        key, t = tech.get(spec["technology"])
        out.update(technology=key, efficiency=t["efficiency"], sigma_t=t["sigma_t"] or 0.0,
                   time_window=t["time_window"])
    out.update({k: spec[k] for k in ("efficiency", "sigma_t") if k in spec})
    return out


def load(path_or_name):
    path = Path(path_or_name)
    if not path.exists():
        path = DESIGN_DIR / f"{path_or_name}.json"
    return from_dict(json.loads(path.read_text()))


def from_dict(raw):
    """A design from its JSON content (the dict is copied, then the physical limits applied)."""
    raw = json.loads(json.dumps(raw))
    notes = validate(raw)
    det = raw.get("detector", {})
    base = detector_settings(det, dict(technology=None, efficiency=0.99, sigma_t=0.5e-9, time_window=None))

    def plane(index, name, kind, z, sign, u_max, spec, cu_key, x_half=None):
        s = detector_settings(spec, base)
        cu = cy = None
        if s["technology"]:
            cu, cy = tech.cells(tech.TECHNOLOGIES[s["technology"]], u_max, spec["y_hi"] - spec["y_lo"])
        cu, cy = spec.get(cu_key, cu), spec.get("cell_y", cy)
        if cu is None or cy is None:
            raise ValueError(f"{name}: give {cu_key} / cell_y or a technology")
        n = int(spec.get("stations", 1))
        if not 1 <= n <= 3:
            raise ValueError(f"{name}: stations must be 1, 2 or 3")
        return Plane(index, name, kind, z, sign, u_max, spec["y_lo"], spec["y_hi"], cu, cy,
                     1 - (1 - s["efficiency"]) ** n, s["sigma_t"], s["technology"], s["time_window"], n, x_half=x_half)

    def module_planes(name, kind, z, sign, spec):
        out = []
        y0 = spec.get("y0", GRID_Y0[kind])
        for b in module_blocks(spec["modules"]):
            t, M = tech.TECHNOLOGIES[b["tech"]], tech.MODULES[b["tech"]]
            (mu, my), (au, ay), fr = M["size"], M["active"], M["frame"]
            gu, gy = (mu - au) / 2, (my - ay) / 2
            dead_u = (gu + fr, gu + fr)
            dead_y = (gy + fr, gy + fr) if M["frame_edges"] == "all" else (gy, gy + fr)
            s_ = detector_settings({"technology": b["tech"]}, base)
            cu, cy = tech.cells(t, au, ay)
            y_lo = y0 + b["iy"] * tech.SLOT
            out.append(Plane(len(planes) + len(out), f"{name} [{b['tech']} u{b['iu']} y{b['iy']}x{b['n']}]", kind, z,
                             sign, mu, y_lo, y_lo + b["n"] * my, cu, cy, 1 - (1 - s_["efficiency"]) ** b["stations"],
                             s_["sigma_t"], b["tech"], s_["time_window"], b["stations"], u_off=b["iu"] * tech.SLOT,
                             mod_u=mu, mod_y=my, dead_u=dead_u, dead_y=dead_y, sub=M.get("sub"),
                             sub_dead=M.get("sub_frame", 0.0), n_modules=b["n"]))
        return out

    planes = []
    z_ubt = raw["ubt"].get("z", geo.Z_UBT)
    if "modules" in raw["ubt"]:
        planes += module_planes("UBT", "ubt", z_ubt, 0, raw["ubt"])
    else:
        planes.append(plane(0, "UBT", "ubt", z_ubt, 0, 2 * geo.UBT_HALF_X, raw["ubt"], "cell_x"))
    if raw.get("ubt_tiles"):                           # default layout (11 Nov 2026): 4 x 4 m^2 tile plane on the window
        t = raw["ubt_tiles"]
        hx = t["half_x"]
        for rg in t.get("regions") or [dict(x=[-hx, hx], y=[t["y_lo"], t["y_hi"]], technology=t["technology"])]:
            (x0, x1), (y0, y1) = rg["x"], rg["y"]
            spec = dict(t, technology=rg["technology"], y_lo=y0, y_hi=y1)
            spec.pop("regions", None)
            tn = tech.get(rg["technology"])[0]
            p = plane(len(planes), f"UBT tiles {tn} x{x0:+.1f}..{x1:+.1f} y{y0:+.1f}..{y1:+.1f}", "tiles", t["z"], 0, x1 - x0,
                      spec, "cell_x", x_half=hx)
            p.u_off = x0 + hx                          # u = x + hx - u_off: the rectangle starts at x0
            planes.append(p)
    for f in sorted(raw["fins"], key=lambda f: (f["z"], f["side"])):
        name = f"{f['side']} fin z={f['z']:.1f}"
        if "modules" in f:
            planes += module_planes(name, "fin", f["z"], SIDES[f["side"]], f)
        else:
            planes.append(plane(len(planes), name, "fin", f["z"], SIDES[f["side"]], f["length"], f, "cell_u"))
    d = Design(raw["name"], planes, base["efficiency"], base["sigma_t"], raw)
    d.notes = notes
    return d
