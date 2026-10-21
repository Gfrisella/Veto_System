"""Interaction targets: the material of every veto technology, the vessel walls and the helium.

What DIS and multiple scattering need from a material: radiation length, areal density (nucleons per area for the
interaction probability), Z/A (proton / neutron share of the nucleons) and a representative nucleus. Each veto
technology is one plane (all layers); its areal density follows from veto.technology.x0 (x/X0 of the whole
plane) and the X0 of its dominant material:

    straw   Mylar walls (C10H8O4; Liverpool UBT0 laminate is mostly Mylar, Al and Au films neglected), gas neglected
    fiber   polystyrene (SCSF-78MJ fibre core, C8H8)
    pixel   silicon
    tile    PVT plastic scintillator (C9H10)
    PVC     vessel walls, C2H3Cl, 0.3 mm (fastveto.vessel)
    helium  vessel volume (G4_He density, as the Geant4 runs)

X0 [g/cm^2] and densities: PDG atomic and nuclear properties tables. Nucleons per gram ~ N_A (A / molar mass ~ 1).
"""
from dataclasses import dataclass

import numpy as np

N_A = 6.02214076e23


@dataclass(frozen=True)
class Material:
    name: str
    x0_g_cm2: float          # radiation length [g/cm^2]
    density: float           # g/cm^3
    z_over_a: float          # <Z/A>: proton share of the nucleons
    a_eff: float             # representative mass number (nuclear effects, not used by the nucleon-level DIS)

    @property
    def x0_m(self):
        return self.x0_g_cm2 / self.density / 100.0

    def nucleons_per_m2(self, thickness_m):
        """Nucleons per m^2 for a slab of this thickness (normal incidence)."""
        return self.density * thickness_m * 100.0 * N_A * 1e4


MATERIALS = {
    "mylar": Material("mylar", 39.95, 1.40, 0.5206, 12.0),
    "polystyrene": Material("polystyrene", 43.79, 1.06, 0.5377, 12.0),
    "silicon": Material("silicon", 21.82, 2.329, 0.4985, 28.0),
    "pvt": Material("pvt", 43.83, 1.032, 0.5413, 12.0),
    "pvc": Material("pvc", 19.63 * 1.38, 1.38, 0.5120, 20.0),      # X0 19.63 cm as fastveto.vessel
    "helium": Material("helium", 94.32, 0.000166322, 0.4997, 4.0),
    "air": Material("air", 36.62, 0.001205, 0.4992, 14.6),
}

TECH_MATERIAL = {"straw": "mylar", "fiber": "polystyrene", "pixel": "silicon", "tile": "pvt", "tile2": "pvt"}


@dataclass(frozen=True)
class Slab:
    """A thin plane of one material: what an interaction / scattering step needs."""
    material: Material
    thickness_m: float           # equivalent thickness at normal incidence

    @property
    def x_over_x0(self):
        return self.thickness_m / self.material.x0_m

    @property
    def nucleons_per_m2(self):
        return self.material.nucleons_per_m2(self.thickness_m)


def tech_slab(tech_name, x0_plane):
    """The slab of one veto plane of a technology with total x/X0 = x0_plane (veto.technology.x0)."""
    m = MATERIALS[TECH_MATERIAL[tech_name]]
    return Slab(m, x0_plane * m.x0_m)


def pvc_wall(thickness_m=0.3e-3):
    return Slab(MATERIALS["pvc"], thickness_m)


def summary():
    """Table of the technologies: x/X0, areal density, nucleons per cm^2."""
    from veto import technology as tech
    rows = []
    for name in ("straw", "fiber", "pixel", "tile", "tile2"):
        s = tech_slab(name, tech.x0(tech.TECHNOLOGIES[name]))
        rows.append((name, s.material.name, s.x_over_x0, s.thickness_m * 1e3, s.material.density * s.thickness_m * 100,
                     s.nucleons_per_m2 * 1e-4))
    w = pvc_wall()
    rows.append(("vessel wall", "pvc", w.x_over_x0, w.thickness_m * 1e3, w.material.density * w.thickness_m * 100,
                 w.nucleons_per_m2 * 1e-4))
    he = Slab(MATERIALS["helium"], 50.0)
    rows.append(("helium 50 m", "helium", he.x_over_x0, he.thickness_m * 1e3, he.material.density * 5000, he.nucleons_per_m2 * 1e-4))
    return rows


if __name__ == "__main__":
    print(f"{'target':12s} {'material':12s} {'x/X0':>8s} {'t [mm]':>10s} {'g/cm^2':>8s} {'nucleons/cm^2':>14s}")
    for r in summary():
        print(f"{r[0]:12s} {r[1]:12s} {r[2]:8.4f} {r[3]:10.3f} {r[4]:8.4f} {r[5]:14.4g}")
