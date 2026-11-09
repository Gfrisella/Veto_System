#!/usr/bin/env python3
"""Which source muons carry the statistics of each region? Share of the variance sum_i W_it^2 (input muon i's weight
in target t, summed over the seeds; the n_eff of t is (sum W)^2 / sum W^2) by source component and momentum.

Targets grouped as: per probe band its weak cells (n_eff below the band's 25th percentile) and all its cells; the
rare types 0 - 3. Uses the pairs cached by bin/input_thinning_study.py (results/thinning/pairs_<dataset>.npz).

    bin/remote.sh run bin/source_importance.py [--dataset <name>]
Writes results/thinning/source_importance_<dataset>.json.
"""

import _bootstrap  # noqa: F401

import argparse
import json

import numpy as np

from veto import paths, probe

import input_thinning_study as its

PB = np.array([0, 20, 50, 100, 150, 200, 250, 300, 1e9])                 # GeV, momentum at the source


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="design-optimization-dataset-alpha-storage-v0-800seeds-2026-10-22")
    args = ap.parse_args()
    z = np.load(paths.results_dir() / "thinning" / f"pairs_{args.dataset}.npz")
    tgt, ev, W = z["tgt"], z["ev"], z["W"]
    pl = probe.planes()
    _, off = probe.grid(pl)
    ncell = int(off[-1])
    bands = np.concatenate(probe.band_labels(pl))
    import h5py
    with h5py.File(paths.enriched_input_h5(), "r") as f:
        w = f["weight"][:]
        src = f["source"][:]
        comp = np.full(len(w), 4, np.int8)
        comp[np.abs(w - 768.75) < 0.5] = 1
        comp[np.abs(w - 7.6875) < 0.01] = 2
        comp[np.abs(w - 326.16) < 0.5] = 3
        comp[src == 1] = 0
        del w, src
        p = np.sqrt(f["px"][:].astype(np.float64) ** 2 + f["py"][:] ** 2 + f["pz"][:] ** 2)
    ip = np.clip(np.digitize(p, PB) - 1, 0, len(PB) - 2)
    cls = (comp.astype(np.int16) * (len(PB) - 1) + ip)[ev]                 # (component, p bin) of each pair
    ncls = 5 * (len(PB) - 1)
    S1 = np.bincount(tgt, weights=W, minlength=ncell + 8)
    S2 = np.bincount(tgt, weights=W ** 2, minlength=ncell + 8)
    with np.errstate(divide="ignore", invalid="ignore"):
        neff = np.where(S2 > 0, S1 ** 2 / S2, 0)
    groups = {}
    for b in probe.BANDS:
        cells = np.nonzero((bands == b) & (neff[:ncell] >= 1))[0]
        q25 = np.percentile(neff[cells], 25)
        groups[f"{b}: weak cells"] = cells[neff[cells] < q25]
        groups[f"{b}: all cells"] = cells
    for t, name in enumerate(("sst1_bounced", "sst1_selected", "sst1_other", "veto_only")):
        groups[f"type {t} {name}"] = np.array([ncell + t])
    out = dict(dataset=args.dataset, components=its.COMPONENTS, p_bins=PB.tolist(), groups={})
    lab = [f"{c} {PB[i]:.0f}-{PB[i + 1]:.0f}" for c in its.COMPONENTS for i in range(len(PB) - 1)]
    for g, cells in groups.items():
        m = np.isin(tgt, cells)
        # share of the variance per class, averaged over the group's targets (each target counts once)
        v = np.bincount(tgt[m] * ncls + cls[m], weights=W[m] ** 2 / S2[tgt[m]], minlength=(ncell + 8) * ncls)
        share = v.reshape(-1, ncls)[cells].mean(0)
        out["groups"][g] = dict(targets=int(len(cells)), share=share.tolist())
        top = np.argsort(share)[::-1][:5]
        print(f"{g:28s} " + ", ".join(f"{lab[k]} {share[k]:.0%}" for k in top if share[k] > 0.01), flush=True)
    json.dump(out, open(paths.results_dir() / "thinning" / f"source_importance_{args.dataset}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
