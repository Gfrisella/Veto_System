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
    kind: str            # "ubt" or "fin"
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

    @property
    def area(self):
        return self.u_max * (self.y_hi - self.y_lo)

    @property
    def n_channels(self):
        """Readout channels: from the technology, or one per cell for the generic detector."""
        if self.technology is None:
            return self.n_cells
        return tech.n_channels(tech.TECHNOLOGIES[self.technology], self.u_max, self.y_hi - self.y_lo)

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
        if self.kind == "ubt":
            return x + geo.UBT_HALF_X
        return self.sign * x - mt.side_plane_x(self.z)

    def to_global(self, u):
        if self.kind == "ubt":
            return u - geo.UBT_HALF_X
        return self.sign * (u + mt.side_plane_x(self.z))

    def inside(self, u, y):
        return (u >= 0) & (u <= self.u_max) & (y >= self.y_lo) & (y <= self.y_hi)

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
    """Apply the physical limits: UBT of straw tubes covering at least the vessel entrance, nothing below y = -3.2 m,
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
        notes.append("UBT bottom raised to y = -3.2 m")
        u["y_lo"] = mt.FIN_Y_MIN
    for i, f in enumerate(raw["fins"]):
        if f.get("tilt", 0) != 0:
            raise NotImplementedError("tilted fins are not in the reconstruction yet")
        if f["y_lo"] < mt.FIN_Y_MIN:
            notes.append(f"fin {i}: bottom raised to y = -3.2 m")
            f["y_lo"] = mt.FIN_Y_MIN
        if not geo.Z_UBT <= f["z"] <= geo.Z_SST1:
            raise ValueError(f"fin {i}: z = {f['z']} outside 30.6 .. 82 m")
        if f["side"] == "right":
            room = -mt.CAVERN_X_WALL - mt.side_plane_x(f["z"])
            if f["length"] > room:
                notes.append(f"fin {i}: cut at the cavern wall ({f['length']:.2f} -> {room:.2f} m)")
                f["length"] = room
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
    raw = json.loads(path.read_text())
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
        return Plane(index, name, kind, z, sign, u_max, spec["y_lo"], spec["y_hi"], cu, cy,
                     s["efficiency"], s["sigma_t"], s["technology"], s["time_window"])

    planes = [plane(0, "UBT", "ubt", geo.Z_UBT, 0, 2 * geo.UBT_HALF_X, raw["ubt"], "cell_x")]
    for f in sorted(raw["fins"], key=lambda f: (f["z"], f["side"])):
        planes.append(plane(len(planes), f"{f['side']} fin z={f['z']:.1f}", "fin", f["z"], SIDES[f["side"]],
                            f["length"], f, "cell_u"))
    d = Design(raw["name"], planes, base["efficiency"], base["sigma_t"], raw)
    d.notes = notes
    return d
