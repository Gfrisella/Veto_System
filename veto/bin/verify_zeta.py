#!/usr/bin/env python3
"""Verify the zeta design dataset before using it for the re-runs (reads only, regenerates nothing).

1. Initial condition: weighted 2D p vs pT of the start state (z = -0.0045 m) of the zeta store vs the source file it
   was transported from (full_sample_after_target_enriched_p6.h5); 1D p / pT projections and their ratio.
2. SST1 rate: muons per spill (4e13 p.o.t., D13) by type, zeta vs alpha-14-oct, plus the zeta start-state 2D map of
   the SST1 types.

    python3 bin/verify_zeta.py [--workers 48] [--out results/verify_zeta]
"""

import _bootstrap  # noqa: F401

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from veto import io, weights
from veto.config import POT_PER_SPILL

sys.path.insert(0, "/disk/data1/hep/gfrise/.claude-config/skills/cern-ship-slides/theme")
from plotstyle import C1, C2, C3, plot2d, save  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data/store"
ZETA = STORE / "design-optimization-dataset-zeta-new-storage-6000seeds-2026-11-01"
ALPHA = STORE / "design-optimization-dataset-alpha-14-oct-2026"
SOURCE = Path("/disk/users/sqasim/workspace/uzh/Veto_System/veto/data/source/full_sample_after_target_enriched_p6.h5")
PE = np.linspace(0, 350, 176)
TE = np.linspace(0, 7, 71)
SST1_TYPES = (0, 1, 2)
CHUNK = 10_000_000


def hist(px, py, pz, w):
    pt = np.hypot(px, py)
    p = np.sqrt(pt * pt + pz * pz)
    return np.histogram2d(p, pt, bins=(PE, TE), weights=w)[0]


def source_chunk(args):
    a, b = args
    with h5py.File(SOURCE, "r") as f:
        d = {k: f[k][a:b] for k in ("px", "py", "pz", "weight")}
    pdg = None
    return hist(d["px"], d["py"], d["pz"], d["weight"].astype(np.float64)), float(d["weight"].sum(dtype=np.float64))


def zeta_chunk(args):
    a, b = args
    cols, _ = io.open_store(ZETA)
    w = np.asarray(cols["weight"][a:b], dtype=np.float64)
    t = np.asarray(cols["type"][a:b])
    px, py, pz = (np.asarray(cols[k][a:b]) for k in ("px_start", "py_start", "pz_start"))
    sst = np.isin(t, SST1_TYPES)
    per_type = np.bincount(t.astype(np.int64), weights=w, minlength=8)
    nan = int(np.isnan(pz).sum())
    return hist(px, py, pz, w), hist(px[sst], py[sst], pz[sst], w[sst]), per_type, nan


def ranges(n):
    return [(a, min(a + CHUNK, n)) for a in range(0, n, CHUNK)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", default=str(ROOT / "results/verify_zeta"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    with h5py.File(SOURCE, "r") as f:
        n_src = f["weight"].shape[0]
    src_pot = weights.LEGACY_POT          # 2018-file normalisation (build_enriched keeps 5e13)
    f_src = POT_PER_SPILL / src_pot
    zcols, zmeta = io.open_store(ZETA)
    n_z = len(zcols["weight"])
    z_pot = weights.source_pot(zmeta, ZETA)
    print(f"source {n_src:,} rows (x{f_src:g} to 4e13); zeta {n_z:,} rows (stored per {z_pot:.0e} p.o.t.)", flush=True)

    with ProcessPoolExecutor(args.workers) as ex:
        H_src, W_src = np.zeros((len(PE) - 1, len(TE) - 1)), 0.0
        for h, w in ex.map(source_chunk, ranges(n_src)):
            H_src += h; W_src += w
        H_src *= f_src; W_src *= f_src
        print(f"source done {time.time() - t0:.0f} s", flush=True)
        H_z = np.zeros_like(H_src); H_zs = np.zeros_like(H_src); per_type = np.zeros(8); nan = 0
        for h, hs, pt, nn in ex.map(zeta_chunk, ranges(n_z)):
            H_z += h; H_zs += hs; per_type += pt; nan += nn
        print(f"zeta done {time.time() - t0:.0f} s", flush=True)

    acols, ameta = io.open_store(ALPHA)
    a_type = np.asarray(acols["type"]).astype(np.int64)
    a_per_type = np.bincount(a_type, weights=np.asarray(acols["weight"], dtype=np.float64), minlength=8)
    types = {int(k): v for k, v in zmeta["types"].items()}

    np.savez(out / "verify_zeta.npz", pe=PE, te=TE, H_src=H_src, H_zeta=H_z, H_zeta_sst1=H_zs,
             per_type_zeta=per_type, per_type_alpha=a_per_type)

    # --- plots: rates in kHz per 4e13 p.o.t. spill (1 s)
    k = 1e-3
    lab = dict(xlabel=r"$p$ [GeV/c]", ylabel=r"$p_T$ [GeV/c]", zlabel="Rate [kHz]")
    plot2d(H_src * k, PE, TE, title="Source muons after target (enriched p6)", fname=out / "p_pt_source", **lab)
    plot2d(H_z * k, PE, TE, title="Zeta store, start state", fname=out / "p_pt_zeta_start", **lab)
    plot2d(H_zs * k, PE, TE, title="Zeta store, start state of SST1 muons", fname=out / "p_pt_zeta_start_sst1", **lab)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(H_src > 0, H_z / H_src, np.nan)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    pcm = ax.pcolormesh(PE, TE, np.ma.masked_invalid(R).T, cmap="RdBu_r", vmin=0.8, vmax=1.2)
    ax.set_xlabel(lab["xlabel"]); ax.set_ylabel(lab["ylabel"]); ax.set_title("Zeta start / source")
    fig.colorbar(pcm, ax=ax, label="ratio"); fig.tight_layout(); save(fig, out / "p_pt_ratio")

    for axis, edges, name, xl in ((1, PE, "p", lab["xlabel"]), (0, TE, "pt", lab["ylabel"])):
        s, z = H_src.sum(axis=axis) * k, H_z.sum(axis=axis) * k
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        a1.stairs(s, edges, color=C1, label="source"); a1.stairs(z, edges, color=C2, label="zeta start")
        a1.set_yscale("log"); a1.set_ylabel("Rate [kHz / bin]"); a1.legend(); a1.set_title(f"Start state, {name}")
        with np.errstate(divide="ignore", invalid="ignore"):
            a2.stairs(np.where(s > 0, z / s, np.nan), edges, color=C3)
        a2.axhline(1, color="0.6", lw=0.8); a2.set_ylim(0.8, 1.2); a2.set_ylabel("zeta / src"); a2.set_xlabel(xl)
        fig.tight_layout(); save(fig, out / f"{name}_1d")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(8)
    ax.bar(x - 0.2, a_per_type * k, 0.4, color=C1, label="alpha-14-oct")
    ax.bar(x + 0.2, per_type * k, 0.4, color=C2, label="zeta-6000")
    ax.set_xticks(x, [types[i] for i in x], rotation=35, ha="right"); ax.set_yscale("log")
    ax.set_ylabel("Rate [kHz]"); ax.set_title("Muons per spill by type"); ax.legend()
    fig.tight_layout(); save(fig, out / "rates_by_type")

    summary = {
        "pot_per_spill": POT_PER_SPILL, "source_total_kHz": W_src * k, "zeta_start_total_kHz": H_z.sum() * k,
        "zeta_total_kHz": per_type.sum() * k, "zeta_nan_start_rows": nan,
        "sst1_kHz": {"zeta": per_type[list(SST1_TYPES)].sum() * k, "alpha": a_per_type[list(SST1_TYPES)].sum() * k},
        "per_type_kHz": {types[i]: {"zeta": per_type[i] * k, "alpha": a_per_type[i] * k} for i in range(8)},
        "outside_hist_range_kHz": (per_type.sum() - H_z.sum()) * k,
        "runtime_s": time.time() - t0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
