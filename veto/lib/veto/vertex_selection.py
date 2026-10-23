"""The signal selection shared by every sample (signal decays, combinatorial pairs, interaction products).

Candidates (bin/benchmark_designs.candidates, on true states): two opposite-charge tracks with p > 1 GeV in the
SST1 and timing acceptance, DOCA < 2 cm, vertex in the vessel, IP < 2.5 m. On top (Selection):
  ip_max      pointing: the IP of the two MEASURED tracks (reco.measure: position, slope and the
              momentum-dependent scattering smearing at z = 82 m) at the target plane z = -2.14 m. A decay
              of a particle from the target points back within the resolution; muons swept by the shield
              that interact, and neutrinos with invisible partners, mostly do not.
  max_tracks  at most this many tracks in the SST1 acceptance in the event: a two-body decay has two,
              interactions usually more (for combinatorial pairs: only the pair, other in-time muons
              are negligible).
"""

from dataclasses import dataclass

import numpy as np

from . import geometry as geo
from . import reco


@dataclass
class Selection:
    ip_max: float = 0.10          # m, measured IP at the target plane: keeps 99.98 % of the two-body signal
    max_tracks: int = 2


def vertex_ip(b1, p1, b2, p2):
    """Closest-approach vertex of two lines (points b, directions p; arrays (n, 3)), their DOCA and the IP of
    the summed momentum at the target plane. Same formulas as benchmark_designs.candidates."""
    dr = b1 - b2
    cross = np.cross(p1, p2)
    doca = np.abs((dr * cross).sum(1)) / np.linalg.norm(cross, axis=1)
    aa, bb, cc = (p1 * p1).sum(1), (p1 * p2).sum(1), (p2 * p2).sum(1)
    dd, ee = (p1 * dr).sum(1), (p2 * dr).sum(1)
    den = aa * cc - bb * bb
    v = 0.5 * (b1 + ((bb * ee - cc * dd) / den)[:, None] * p1 + b2 + ((aa * ee - bb * dd) / den)[:, None] * p2)
    mom = p1 + p2
    dz = geo.Z_TARGET - v[:, 2]
    ip = np.hypot(v[:, 0] + mom[:, 0] / mom[:, 2] * dz, v[:, 1] + mom[:, 1] / mom[:, 2] * dz)
    return v, doca, ip


def measured(b, p, cfg, rng):
    """Measured states (reco.measure) of tracks given by their state b and momentum p at z = 82 m."""
    tr = dict(bx=b[:, 0], by=b[:, 1], bz=b[:, 2], bpx=p[:, 0], bpy=p[:, 1], bpz=p[:, 2])
    m = reco.measure(tr, cfg, rng)
    pm = np.linalg.norm(p, axis=1)
    pz = pm / np.sqrt(1 + m["tx"] ** 2 + m["ty"] ** 2)
    return np.stack([m["x"], m["y"], np.asarray(m["z"], float)], 1), np.stack([m["tx"] * pz, m["ty"] * pz, pz], 1)


def measured_ip(b1, p1, b2, p2, cfg, rng):
    """IP of pairs from their measured tracks."""
    mb1, mp1 = measured(b1, p1, cfg, rng)
    mb2, mp2 = measured(b2, p2, cfg, rng)
    return vertex_ip(mb1, mp1, mb2, mp2)[2]
