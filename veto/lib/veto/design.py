"""Veto designs: a UBT at the vessel entrance plus fins on the left / right side, and detector settings.

A design is a JSON file (veto/designs/*.json):

    {"name": "D2_default",
     "ubt": {"y_lo": -1.35, "y_hi": 1.35, "cell_x": 0.1, "cell_y": 0.1},
     "fins": [{"side": "right", "z": 30.6, "length": 3, "y_lo": -3, "y_hi": 3, "tilt": 0,
               "cell_u": 0.1, "cell_y": 0.1}, ...],
     "detector": {"efficiency": 0.99, "sigma_t": 0.5e-9}}

Detector technology (technology.py: straw, fiber, pixel, tile): "technology" in "detector" applies
to every fin, "technology" in a fin overrides it for that fin. A technology sets the plane's
efficiency, timing, cell size, channels, material and cost (kCHF); explicit "efficiency" /
"sigma_t" / cell sizes next to it win. The UBT is fixed: Liverpool straws, vertical (technology.py),
its cell sizes in the JSON are ignored. Without a technology a fin is the generic detector of
before (efficiency and sigma_t of "detector", cells as given, no cost).

Sides are seen from the target looking downstream: right = -x (cavern wall at x = -4.56 m),
left = +x. A fin is attached to the vessel structure: its inner edge is on the side wall at
|x| = X(z), 30.6 <= z <= 80.6 m (the support frame carries it down to the floor), and it reaches
OUTWARD (away from the beam) by `length`: no fin is inside the decay vessel. Its local
coordinate u is the distance from that inner edge, along the fin. `tilt` [degrees] leans the fin
downstream (> 0) or upstream (< 0) about its inner edge: the point at u is at
x = sign * (X(z) + u cos(tilt)), z + u sin(tilt). Any number of fins, on either side, independently.

Optionally ONE panel under the decay vessel ("bottom", independent of the vessel structure):
    "bottom": {"z": 55, "length": 3, "x_lo": -1.5, "x_hi": 1.5, "y": -3.3, "pitch": 0, "technology": "straw"}
a flat panel from z to z + length cos(pitch), across x_lo .. x_hi, at height y at its upstream edge
rising downstream by pitch [deg]. It stays between the floor and the vessel bottom (BOTTOM_GAP below
it) and is at most BOTTOM_MAX_AREA. Local coordinates: u = x - x_lo, v = distance along the panel
(stored as y_lo = 0 .. y_hi = length).

Optionally detectors ON the lateral vessel walls (the SBT side walls, "walls", at most one per side):
    "walls": [{"side": "left", "z_lo": 30.6, "z_hi": 80.6, "technology": "straw"}, ...]
covering the wall x = sign * X(z) from z_lo to z_hi over its full height |y| <= Y(z) (a trapezoid).
Local coordinates: u = z - z_lo (along the beam), y. The plane's y_lo / y_hi are the bounding box
(-Y(z_hi) .. Y(z_hi)); `inside` also checks |y| <= Y(z); area, cost and channels are the trapezoid's.
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
MAX_TILT_DEG = 75.0
MAX_PITCH_DEG = 30.0
BOTTOM_MAX_AREA = 10.0      # m^2: "one small panel"
BOTTOM_GAP = 0.05           # m between the panel and the vessel bottom
BOTTOM_Z_MAX = 81.5         # downstream end, before SST1


@dataclass
class Plane:
    """One detector plane at fixed z. Local coordinates (u, y), cells of size (cell_u, cell_y).

    UBT: u = x + 0.5. Fin: u = sign * x - X(z). Bottom panel: u = x - x_lo, "y" = distance along the
    panel from its upstream edge (z, y0), which rises by tilt (the pitch).
    """
    index: int
    name: str
    kind: str            # "ubt", "fin", "bottom" or "wall"
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
    tilt: float = 0.0           # [rad] fins: leaning downstream about the inner edge (0 = x-y plane); bottom: pitch
    x_lo: float = 0.0           # bottom panel: x of its edge at u = 0
    y0: float = 0.0             # bottom panel: y of its upstream edge

    @property
    def area(self):
        if self.kind == "wall":     # trapezoid: full wall height 2 Y(z), linear in z
            z1 = self.z + self.u_max
            return float(self.u_max * (wall_half_height(self.z) + wall_half_height(z1)))
        return self.u_max * (self.y_hi - self.y_lo)

    @property
    def n_channels(self):
        """Readout channels: from the technology, or one per cell for the generic detector."""
        if self.technology is None:
            return self.n_cells
        n = tech.n_channels(tech.TECHNOLOGIES[self.technology], self.u_max, self.y_hi - self.y_lo)
        if self.kind == "wall":     # bounding box -> trapezoid
            n = int(np.ceil(n * self.area / (self.u_max * (self.y_hi - self.y_lo))))
        return n

    @property
    def cost(self):
        """kCHF; nan for the generic detector or a technology without a cost."""
        if self.technology is None:
            return float("nan")
        return tech.cost(tech.TECHNOLOGIES[self.technology], self.area)

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
        return tech.x0(tech.TECHNOLOGIES[self.technology])

    @property
    def n_cells(self):
        return int(np.ceil(self.u_max / self.cell_u - 1e-9)) * int(np.ceil((self.y_hi - self.y_lo) / self.cell_y - 1e-9))

    def to_local(self, x):
        """u of a point of the plane from its x (the point must be on the plane)."""
        if self.kind == "ubt":
            return x + geo.UBT_HALF_X
        if self.kind == "bottom":
            return x - self.x_lo
        if self.kind == "wall":
            raise ValueError("wall plane: u = z - z_lo, not a function of x")
        return (self.sign * x - mt.side_plane_x(self.z)) / np.cos(self.tilt)

    def to_global(self, u):
        if self.kind == "ubt":
            return u - geo.UBT_HALF_X
        if self.kind == "bottom":
            return u + self.x_lo
        if self.kind == "wall":
            return self.sign * mt.side_plane_x(self.z + u)
        return self.sign * (mt.side_plane_x(self.z) + u * np.cos(self.tilt))

    def z_at(self, u):
        """z of the point at u (fins lean by tilt; the UBT is at one z; bottom: see z_local)."""
        if self.kind == "bottom":
            return np.full(np.shape(u), self.z) if np.ndim(u) else self.z
        if self.kind == "wall":
            return self.z + u
        return self.z + u * np.sin(self.tilt)

    def z_local(self, u, v):
        """z of the local point (u, v) of any plane."""
        if self.kind == "bottom":
            return self.z + v * np.cos(self.tilt)
        return self.z_at(u)

    @property
    def z_end(self):
        """Downstream end of a bottom panel."""
        return self.z + self.y_hi * np.cos(self.tilt)

    @property
    def y_end(self):
        return self.y0 + self.y_hi * np.sin(self.tilt)

    def inside(self, u, y):
        ok = (u >= 0) & (u <= self.u_max) & (y >= self.y_lo) & (y <= self.y_hi)
        if self.kind == "wall":
            ok = ok & (np.abs(y) <= wall_half_height(self.z + np.clip(u, 0, self.u_max)))
        return ok

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


def validate(raw):
    """Apply the physical limits: UBT of straw tubes covering at least the vessel entrance, nothing below the floor (y = -3.36 m),
    fins attached to the vessel structure (inner edge at 30.6 <= z <= 80.6 m),
    right fins stop at the cavern wall. Returns a list of what was changed."""
    notes = []
    u = raw["ubt"]
    if u["y_lo"] > -geo.UBT_HALF_Y or u["y_hi"] < geo.UBT_HALF_Y:
        notes.append("UBT extended to cover the vessel entrance")
        u["y_lo"], u["y_hi"] = min(u["y_lo"], -geo.UBT_HALF_Y), max(u["y_hi"], geo.UBT_HALF_Y)
    if tech.ALIASES.get(u.get("technology"), u.get("technology")) not in (None, "straw"):
        notes.append(f"UBT is straw tubes only ({u['technology']} -> straw)")
    u["technology"] = "straw"                             # Liverpool straws, vertical
    for k in ("cell_x", "cell_y"):
        u.pop(k, None)
    if u["y_lo"] < mt.FIN_Y_MIN:
        notes.append(f"UBT bottom raised to the floor, y = {mt.FIN_Y_MIN} m")
        u["y_lo"] = mt.FIN_Y_MIN
    for i, f in enumerate(raw["fins"]):
        tilt = f.get("tilt", 0)
        if abs(tilt) > MAX_TILT_DEG:
            raise ValueError(f"fin {i}: |tilt| = {abs(tilt)} > {MAX_TILT_DEG} degrees")
        if f["y_lo"] < mt.FIN_Y_MIN:
            notes.append(f"fin {i}: bottom raised to the floor, y = {mt.FIN_Y_MIN} m")
            f["y_lo"] = mt.FIN_Y_MIN
        if not geo.Z_UBT <= f["z"] <= geo.Z_VESSEL_MAX:     # attached to the vessel structure
            raise ValueError(f"fin {i}: z = {f['z']} outside the vessel, {geo.Z_UBT} .. {geo.Z_VESSEL_MAX} m")
        c, s = np.cos(np.radians(tilt)), np.sin(np.radians(tilt))
        if f["side"] == "right":
            room = (-mt.CAVERN_X_WALL - mt.side_plane_x(f["z"])) / c
            if f["length"] > room:
                notes.append(f"fin {i}: cut at the cavern wall ({f['length']:.2f} -> {room:.2f} m)")
                f["length"] = room
        if s != 0:      # the outer edge stays between the UBT and SST1
            room = ((geo.Z_SST1 if s > 0 else geo.Z_UBT) - f["z"]) / s
            if f["length"] > room:
                notes.append(f"fin {i}: cut at z = {geo.Z_SST1 if s > 0 else geo.Z_UBT} m "
                             f"({f['length']:.2f} -> {room:.2f} m)")
                f["length"] = room
        if f["length"] < 0.1:
            raise ValueError(f"fin {i}: no room left (length {f['length']:.2f} m after the cuts)")
        # from the vessel wall outward only: every point of the fin is outside the vessel
        u = np.linspace(0, f["length"], 21)
        z_u = f["z"] + u * s
        out = mt.side_plane_x(f["z"]) + u * c - mt.side_plane_x(np.minimum(z_u, geo.Z_VESSEL_MAX))
        if (out < -1e-9).any():
            raise ValueError(f"fin {i}: enters the decay vessel (tilt {tilt} degrees)")
    if raw.get("bottom"):
        notes += validate_bottom(raw["bottom"])
    sides = [w["side"] for w in raw.get("walls", [])]
    if len(set(sides)) < len(sides):
        raise ValueError("walls: at most one per side")
    for w in raw.get("walls", []):
        w["z_lo"], w["z_hi"] = max(w.get("z_lo", geo.Z_VESSEL_MIN), geo.Z_VESSEL_MIN), \
            min(w.get("z_hi", geo.Z_VESSEL_MAX), geo.Z_VESSEL_MAX)
        if w["z_hi"] - w["z_lo"] < 0.1:
            raise ValueError(f"wall {w['side']}: z {w['z_lo']} .. {w['z_hi']} m is not on the vessel")
    return notes


def wall_half_height(z):
    """Half height Y(z) of the lateral vessel walls."""
    return geo.vessel_half_widths(np.clip(z, geo.Z_VESSEL_MIN, geo.Z_VESSEL_MAX))[1]


def vessel_bottom(z):
    """y of the vessel bottom at z (the vessel half height, constant past its end)."""
    return -geo.vessel_half_widths(np.minimum(z, geo.Z_VESSEL_MAX))[1]


def validate_bottom(b):
    """Limits of the panel under the vessel; modifies b, returns notes, ValueError if it cannot fit."""
    notes = []
    p = b.setdefault("pitch", 0.0)
    if abs(p) > MAX_PITCH_DEG:
        raise ValueError(f"bottom: |pitch| = {abs(p)} > {MAX_PITCH_DEG} degrees")
    c, s = np.cos(np.radians(p)), np.sin(np.radians(p))
    if b["z"] < geo.Z_UBT:
        raise ValueError(f"bottom: z = {b['z']} upstream of the vessel")
    b["x_lo"], b["x_hi"] = max(b["x_lo"], geo.CAVERN_X_WALL), min(b["x_hi"], -geo.CAVERN_X_WALL)
    width = b["x_hi"] - b["x_lo"]
    if width < 0.2:
        raise ValueError(f"bottom: width {width:.2f} m < 0.2 m")
    room = (BOTTOM_Z_MAX - b["z"]) / c
    if b["length"] > room:
        notes.append(f"bottom: cut at z = {BOTTOM_Z_MAX} m ({b['length']:.2f} -> {room:.2f} m)")
        b["length"] = room
    if b["length"] * width > BOTTOM_MAX_AREA:
        new = BOTTOM_MAX_AREA / width
        notes.append(f"bottom: length cut to the {BOTTOM_MAX_AREA:g} m2 limit ({b['length']:.2f} -> {new:.2f} m)")
        b["length"] = new
    if b["length"] < 0.2:
        raise ValueError(f"bottom: no room left (length {b['length']:.2f} m)")
    if b["y"] < mt.FIN_Y_MIN:
        notes.append(f"bottom: raised to the floor, y = {mt.FIN_Y_MIN} m")
        b["y"] = mt.FIN_Y_MIN
    z1, y1 = b["z"] + b["length"] * c, b["y"] + b["length"] * s
    for z, y in ((b["z"], b["y"]), (z1, y1)):
        if y < mt.FIN_Y_MIN - 1e-9 or y > vessel_bottom(z) - BOTTOM_GAP + 1e-9:
            raise ValueError(f"bottom: y = {y:.2f} m at z = {z:.1f} m is not between the floor and the vessel")
    return notes


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

    def plane(index, name, kind, z, sign, u_max, spec, cu_key):
        s = detector_settings(spec, base)
        cu = cy = None
        if s["technology"]:
            cu, cy = tech.cells(tech.TECHNOLOGIES[s["technology"]], u_max, spec["y_hi"] - spec["y_lo"])
        cu, cy = spec.get(cu_key, cu), spec.get("cell_y", cy)
        if cu is None or cy is None:
            raise ValueError(f"{name}: give {cu_key} / cell_y or a technology")
        tilt = {"fin": spec.get("tilt", 0.0), "bottom": spec.get("pitch", 0.0)}.get(kind, 0.0)
        return Plane(index, name, kind, z, sign, u_max, spec["y_lo"], spec["y_hi"], cu, cy,
                     s["efficiency"], s["sigma_t"], s["technology"], s["time_window"], np.radians(tilt),
                     spec.get("x_lo", 0.0), spec.get("y", 0.0))

    planes = [plane(0, "UBT", "ubt", geo.Z_UBT, 0, 2 * geo.UBT_HALF_X, raw["ubt"], "cell_x")]
    for f in sorted(raw["fins"], key=lambda f: (f["z"], f["side"])):
        name = f"{f['side']} fin z={f['z']:.1f}" + (f" tilt={f['tilt']:+.0f}" if f.get("tilt") else "")
        planes.append(plane(len(planes), name, "fin", f["z"], SIDES[f["side"]],
                            f["length"], f, "cell_u"))
    if raw.get("bottom"):
        b = raw["bottom"]
        name = f"bottom z={b['z']:.1f}-{b['z'] + b['length'] * np.cos(np.radians(b['pitch'])):.1f}"
        planes.append(plane(len(planes), name, "bottom", b["z"], 0, b["x_hi"] - b["x_lo"],
                            dict(b, y_lo=0.0, y_hi=b["length"]), "cell_u"))
    for w in sorted(raw.get("walls", []), key=lambda w: w["side"]):
        h = float(wall_half_height(w["z_hi"]))
        planes.append(plane(len(planes), f"{w['side']} wall z={w['z_lo']:.1f}-{w['z_hi']:.1f}", "wall", w["z_lo"],
                            SIDES[w["side"]], w["z_hi"] - w["z_lo"], dict(w, y_lo=-h, y_hi=h), "cell_u"))
    d = Design(raw["name"], planes, base["efficiency"], base["sigma_t"], raw)
    d.notes = notes
    return d
