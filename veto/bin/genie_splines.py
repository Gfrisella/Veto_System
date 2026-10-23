"""GENIE cross-section splines for the neutrino library: the official G18_02a_00_000 set (GENIE 3.06.00, 250 knots up to
1 TeV, gxspl-NUsmall.xml) trimmed to our nuclei (faster gevgen start-up), and a table of total cross sections.

    bin/lcg.sh python3 bin/genie_splines.py

Nuclei: H1, He4, C12, N14, O16, Si28, Cl35, Cl37, Ar40 (helium, PVC, Mylar, polystyrene, PVT, silicon, air).
Output: data/genie/gxspl-fvs.xml and data/genie/xsec_tot.npz: E grid, sigma_tot [cm^2 per nucleus] per
(neutrino, nucleus) = sum of every process spline (CC, NC, QE, RES, DIS, COH, MEC, nu-e) of that pair.
"""

import _bootstrap  # noqa: F401
import gzip
import re

import numpy as np

from veto import paths

SRC = paths.interactions_data() / "genie/genie_xsec/v3_06_00/NULL/G1802a00000-k250-e1000/data/gxspl-NUsmall.xml"
NUCLEI = [1000010010, 1000020040, 1000060120, 1000070140, 1000080160, 1000140280, 1000170350, 1000170370, 1000180400]
NUS = [14, -14, 12, -12, 16, -16]


def main():
    text = SRC.read_text() if SRC.suffix == ".xml" else gzip.open(SRC, "rt").read()
    head = text[: text.index("<spline")]
    tail = text[text.rindex("</spline>") + len("</spline>"):]
    blocks = re.findall(r"<spline .*?</spline>", text, flags=re.S)
    keep, tot = [], {}
    E_ref = None
    for b in blocks:
        name = re.search(r'name="([^"]*)"', b).group(1)
        tgt = int(re.search(r"tgt:(\d+)", name).group(1))
        nu = int(re.search(r"nu:(-?\d+)", name).group(1))
        if tgt not in NUCLEI or nu not in NUS:
            continue
        keep.append(b)
        k = np.array(re.findall(r"<E>\s*([-0-9.eE+]+)\s*</E>\s*<xsec>\s*([-0-9.eE+]+)\s*</xsec>", b), float)
        if E_ref is None:
            E_ref = k[:, 0]
        x = np.interp(E_ref, k[:, 0], k[:, 1])
        tot[(nu, tgt)] = tot.get((nu, tgt), 0) + x
    out = paths.interactions_data() / "genie/gxspl-fvs.xml"
    out.write_text(head + "\n".join(keep) + tail)
    # GENIE splines are in natural units (GeV^-2): 1 GeV^-2 = 0.3893794e-27 cm^2
    conv = 0.3893794e-27
    keys = sorted(tot)
    np.savez(paths.interactions_data() / "genie/xsec_tot.npz", E=E_ref, nu=np.array([k[0] for k in keys]), tgt=np.array([k[1] for k in keys]),
             sigma=np.array([tot[k] for k in keys]) * conv)
    print(f"{len(keep)} of {len(blocks)} splines kept -> {out} ({out.stat().st_size / 1e6:.0f} MB)")
    for nu in (14, -14):
        for tgt in (1000020040, 1000060120):
            s = tot[(nu, tgt)] * conv / (tgt % 1000 // 10)
            print(f"  nu {nu:3d} tgt {tgt}: sigma/A/E [1e-38 cm^2/GeV] at 5, 20, 100, 300 GeV:",
                  [round(float(np.interp(e, E_ref, s)) / e / 1e-38, 3) for e in (5, 20, 100, 300)])


if __name__ == "__main__":
    main()
