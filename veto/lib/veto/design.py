"""Veto designs: a UBT at the vessel entrance plus fins on the left / right side, and detector settings.

A design is a JSON file (veto/designs/*.json):

    {"name": "D2_default",
     "ubt": {"y_lo": -1.35, "y_hi": 1.35, "cell_x": 0.1, "cell_y": 0.1},
     "fins": [{"side": "right", "z": 30.6, "length": 3, "y_lo": -3, "y_hi": 3, "tilt": 0,
               "cell_u": 0.1, "cell_y": 0.1}, ...],
     "detector": {"efficiency": 0.99, "sigma_t": 0.5e-9}}

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

    @property
    def area(self):
        return self.u_max * (self.y_hi - self.y_lo)

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


def validate(raw):
    """Apply the physical limits: UBT at least the vessel entrance, nothing below y = -3.2 m,
    right fins stop at the cavern wall. Returns a list of what was changed."""
    notes = []
    u = raw["ubt"]
    if u["y_lo"] > -geo.UBT_HALF_Y or u["y_hi"] < geo.UBT_HALF_Y:
        notes.append("UBT extended to cover the vessel entrance")
        u["y_lo"], u["y_hi"] = min(u["y_lo"], -geo.UBT_HALF_Y), max(u["y_hi"], geo.UBT_HALF_Y)
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


def load(path_or_name):
    path = Path(path_or_name)
    if not path.exists():
        path = DESIGN_DIR / f"{path_or_name}.json"
    raw = json.loads(path.read_text())
    notes = validate(raw)
    u = raw["ubt"]
    planes = [Plane(0, "UBT", "ubt", geo.Z_UBT, 0, 2 * geo.UBT_HALF_X, u["y_lo"], u["y_hi"],
                    u["cell_x"], u["cell_y"])]
    for f in sorted(raw["fins"], key=lambda f: (f["z"], f["side"])):
        planes.append(Plane(len(planes), f"{f['side']} fin z={f['z']:.1f}", "fin", f["z"], SIDES[f["side"]],
                            f["length"], f["y_lo"], f["y_hi"], f["cell_u"], f["cell_y"]))
    det = raw.get("detector", {})
    d = Design(raw["name"], planes, det.get("efficiency", 0.99), det.get("sigma_t", 0.5e-9), raw)
    d.notes = notes
    return d
