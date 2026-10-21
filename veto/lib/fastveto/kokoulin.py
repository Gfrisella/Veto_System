"""Muon photonuclear cross section of Geant4 (G4KokoulinMuonNuclearXS, Borog & Petrukhin / Kokoulin formula).

dsigma/dnu per nucleus for energy transfer nu, with the A_eff shadowing of Geant4 and its cut nu > 0.2 GeV.
Used to compare and normalise the Pythia 6 library (which starts at W > 2 GeV) with what Geant4 does.
"""
import numpy as np

ALPHA = 1 / 137.035999
M_MU, M_P = 0.1056584, 0.9382721
ALAM2, ALAM = 0.400, 0.632456
_trapz = getattr(np, "trapezoid", None) or np.trapz
NU_CUT = 0.2                                   # GeV, Geant4 CutFixed


def a_eff(A):
    return 0.22 * A + 0.78 * A ** 0.89


def dsigma_dnu(E, nu, A=1.0):
    """dsigma/dnu [microbarn / GeV] per nucleus (A_eff) for muon total energy E, energy transfer nu (GeV)."""
    E, nu = np.broadcast_arrays(np.asarray(E, float), np.asarray(nu, float))
    ok = (nu > NU_CUT) & (nu < E - 0.5 * M_P)
    ep = np.where(ok, nu, 1.0)
    sigph = 49.2 + 11.1 * np.log(ep) + 151.8 / np.sqrt(ep)                # microbarn, real-photon cross section
    v = ep / E
    v1, v2, m2 = 1 - v, v * v, M_MU ** 2
    up = E * E * v1 / m2 * (1 + m2 * v2 / (ALAM2 * v1))
    down = 1 + ep / ALAM * (1 + ALAM / 2 / M_P + ep / ALAM)
    d = ALPHA / np.pi * a_eff(A) * sigph / ep * (-v1 + (v1 + 0.5 * v2 * (1 + 2 * m2 / ALAM2)) * np.log(up / down))
    return np.where(ok & (d > 0), d, 0.0)


def sigma(E, nu_min=NU_CUT, A=1.0, n=400):
    """Integrated cross section [microbarn] per nucleus above nu_min (log-spaced Simpson in nu)."""
    E = np.atleast_1d(np.asarray(E, float))
    out = np.zeros(len(E))
    for i, e in enumerate(E):
        hi = e - 0.5 * M_P
        if hi <= nu_min:
            continue
        t = np.linspace(np.log(nu_min), np.log(hi), n)
        nu = np.exp(t)
        out[i] = _trapz(dsigma_dnu(e, nu, A) * nu, t)
    return out
