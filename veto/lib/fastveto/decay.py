"""Decays of the short-lived particles of the Geant4 library final states (Pythia decays its own).

Geant4's muon-nuclear products are recorded at the interaction (cpp/g4dis kills them there), so K_S, Lambda,
Sigma, Xi, Omega, pi0 and eta are still undecayed. Here they decay as Pythia would by default: main channels
(PDG branching ratios, >= 99 % of each width), isotropic in the rest frame, flat phase space for three bodies,
decay point from c tau, products decayed again (Lambda from Xi, pi0 from K_S, ...). Charged pions and kaons,
K_L, neutrons, muons stay stable (as Pythia's defaults).
"""
import numpy as np

MASS = {111: 0.134977, 211: 0.139570, 221: 0.547862, 22: 0.0, 2212: 0.938272, 2112: 0.939565, 310: 0.497611,
        321: 0.493677, 3122: 1.115683, 3222: 1.18937, 3212: 1.192642, 3112: 1.197449, 3322: 1.31486, 3312: 1.32171,
        3334: 1.67245, 11: 0.000511, 13: 0.105658, 12: 0.0, 14: 0.0}
CTAU = {111: 2.55e-8, 221: 1.5e-13, 310: 0.026844, 3122: 0.0789, 3222: 0.02404, 3212: 2.22e-11, 3112: 0.0443,
        3322: 0.0871, 3312: 0.0491, 3334: 0.0246}                                       # metres
# channels (BR, products) for the particle; antiparticles: conjugated products
CHANNELS = {
    111: [(0.988, (22, 22)), (0.012, (22, 11, -11))],
    221: [(0.394, (22, 22)), (0.327, (111, 111, 111)), (0.229, (211, -211, 111)), (0.050, (22, 211, -211))],
    310: [(0.692, (211, -211)), (0.308, (111, 111))],
    3122: [(0.639, (2212, -211)), (0.361, (2112, 111))],
    3222: [(0.516, (2212, 111)), (0.484, (2112, 211))],
    3212: [(1.0, (3122, 22))],
    3112: [(1.0, (2112, -211))],
    3322: [(1.0, (3122, 111))],
    3312: [(1.0, (3122, -211))],
    3334: [(0.678, (3122, -321)), (0.236, (3322, -211)), (0.086, (3312, 111))],
}
def _conj(p):
    return p if p in (22, 111, 221, 310, 130) else -p


def mass(pdg):
    return np.array([MASS.get(abs(int(q)), 0.0) for q in pdg])


def _two_body(M, m1, m2, n, rng):
    p = np.sqrt(np.maximum((M ** 2 - (m1 + m2) ** 2) * (M ** 2 - (m1 - m2) ** 2), 0)) / (2 * M)
    c = rng.uniform(-1, 1, n)
    s = np.sqrt(1 - c ** 2)
    phi = rng.uniform(0, 2 * np.pi, n)
    d = np.stack([s * np.cos(phi), s * np.sin(phi), c], 1)
    return p[:, None] * d if np.ndim(p) else p * d


def _three_body(M, ms, n, rng):
    """Flat phase space for M -> m1 m2 m3 at rest: momenta (n, 3, 3). Uniform in (m12^2, m23^2) inside the Dalitz
    region (accept-reject), p1 along z, p3 in the xz plane, then a random rotation."""
    m1, m2, m3 = ms
    s12, s23 = np.empty(n), np.empty(n)
    todo = np.arange(n)
    while len(todo):
        k = len(todo)
        a = rng.uniform((m1 + m2) ** 2, (M - m3) ** 2, k)
        b = rng.uniform((m2 + m3) ** 2, (M - m1) ** 2, k)
        e2 = (a - m1 ** 2 + m2 ** 2) / (2 * np.sqrt(a))
        e3 = (M ** 2 - a - m3 ** 2) / (2 * np.sqrt(a))
        p2, p3 = np.sqrt(np.maximum(e2 ** 2 - m2 ** 2, 0)), np.sqrt(np.maximum(e3 ** 2 - m3 ** 2, 0))
        ok = (b >= (e2 + e3) ** 2 - (p2 + p3) ** 2) & (b <= (e2 + e3) ** 2 - (p2 - p3) ** 2)
        s12[todo[ok]], s23[todo[ok]] = a[ok], b[ok]
        todo = todo[~ok]
    E1 = (M ** 2 + m1 ** 2 - s23) / (2 * M)
    E3 = (M ** 2 + m3 ** 2 - s12) / (2 * M)
    q1 = np.sqrt(np.maximum(E1 ** 2 - m1 ** 2, 0))
    q3 = np.sqrt(np.maximum(E3 ** 2 - m3 ** 2, 0))
    s13 = M ** 2 + m1 ** 2 + m2 ** 2 + m3 ** 2 - s12 - s23
    c13 = np.clip((2 * E1 * E3 + m1 ** 2 + m3 ** 2 - s13) / np.maximum(2 * q1 * q3, 1e-12), -1, 1)
    v1 = np.stack([np.zeros(n), np.zeros(n), q1], 1)
    v3 = np.stack([q3 * np.sqrt(1 - c13 ** 2), np.zeros(n), q3 * c13], 1)
    v2 = -v1 - v3
    R = _random_rotation(n, rng)
    return np.stack([np.einsum("nij,nj->ni", R, v) for v in (v1, v2, v3)], 1)


def _random_rotation(n, rng):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    a, b, c, d = q.T
    return np.stack([np.stack([a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)], 1),
                     np.stack([2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)], 1),
                     np.stack([2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d], 1)], 1)


def _boost(p_rest, E_rest, P, M):
    """Boost rest-frame momenta (n,3), energies (n,) by parent momentum P (n,3), mass M."""
    E = np.sqrt((P ** 2).sum(1) + M ** 2)
    b = P / E[:, None]
    b2 = (b ** 2).sum(1)
    g = E / M
    bp = (b * p_rest).sum(1)
    g2 = np.where(b2 > 0, (g - 1) / np.maximum(b2, 1e-30), 0.0)
    p = p_rest + (g2 * bp + g * E_rest)[:, None] * b
    return p, g * (E_rest + bp)


def decay_all(ev, pdg, p, v, rng, max_rounds=4):
    """Decay the unstable particles of flat particle lists (event id, pdg, p (n,4), production point v (n,3)).
    Returns new (ev, pdg, p, v, mother flag) with the decayed particles replaced by their products."""
    ev, pdg, p, v = ev.copy(), pdg.copy(), p.copy(), v.copy()
    for _ in range(max_rounds):
        unstable = np.isin(np.abs(pdg), list(CHANNELS)) & ~((pdg < 0) & np.isin(-pdg, [111, 221, 310]))
        if not unstable.any():
            break
        keep_ev, keep_pdg, keep_p, keep_v = [ev[~unstable]], [pdg[~unstable]], [p[~unstable]], [v[~unstable]]
        idx = np.nonzero(unstable)[0]
        for a in np.unique(np.abs(pdg[idx])):
            sel = idx[np.abs(pdg[idx]) == a]
            M = MASS[a]
            P = p[sel, :3]
            pm = np.linalg.norm(P, axis=1)
            L = rng.exponential(CTAU[a], len(sel)) * pm / M
            dvtx = v[sel] + (P / np.maximum(pm, 1e-12)[:, None]) * L[:, None]
            br = np.array([c[0] for c in CHANNELS[a]])
            ch = np.searchsorted(np.cumsum(br / br.sum()), rng.random(len(sel)), side="right")
            for ci, (_, prods) in enumerate(CHANNELS[a]):
                m = ch == ci
                if not m.any():
                    continue
                s = sel[m]
                sign = np.sign(pdg[s])
                ms = [MASS[abs(q)] for q in prods]
                k = len(s)
                if len(prods) == 2:
                    q = _two_body(M, ms[0], ms[1], k, rng)
                    rest = [q, -q]
                else:
                    r3 = _three_body(M, ms, k, rng)
                    rest = [r3[:, 0], r3[:, 1], r3[:, 2]]
                for q_rest, code, mi in zip(rest, prods, ms):
                    e_rest = np.sqrt((q_rest ** 2).sum(1) + mi ** 2)
                    plab, elab = _boost(q_rest, e_rest, P[m], M)
                    codes = np.where(sign > 0, code, _conj(code))
                    keep_ev.append(ev[s]); keep_pdg.append(codes.astype(pdg.dtype))
                    keep_p.append(np.concatenate([plab, elab[:, None]], 1)); keep_v.append(dvtx[m])
        ev, pdg, p, v = (np.concatenate(x) for x in (keep_ev, keep_pdg, keep_p, keep_v))
        order = np.argsort(ev, kind="stable")
        ev, pdg, p, v = ev[order], pdg[order], p[order], v[order]
    return ev, pdg, p, v
