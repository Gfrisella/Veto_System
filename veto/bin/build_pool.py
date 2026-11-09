#!/usr/bin/env python3
"""The input pool v4: fresh source muons where the statistics are limited, Guglielmo's rows elsewhere, thinned by
the source-bin rule, split into a training part and a hidden part, in chunks of ~100 M muons (one chunk per seed).

Why (bin/source_importance.py): the weak probe cells and the rare types are limited by the minimum-bias muons at
20 - 150 GeV and charm at 20 - 50 GeV (heavy rows, weights 768.75 / 326.16); the dimuon class (weight 7.69, 480 M
rows) matters little below 150 GeV and is very slow to regenerate.

  component   momentum after the target   from
  min. bias   < MB_LO                  the enriched file (old rows)
              MB_LO - 150 GeV          fresh: tail_produce --emin 20 (tag pool_mb20)
              >= 150 GeV               fresh: tail_produce, forced pi / K tail (tag pool_tail)
  dimuon      < 150 GeV                the enriched file
              >= 150 GeV               fresh: tail_produce --dimuon --no-pik --vm-fairship (tag pool_dmhigh)
  charm       < 20 GeV                 the enriched file
              >= 20 GeV                fresh: charm_cascade (tag charmc, slices from 40000)
  other       all                      the enriched file
Fresh rows in the file's convention: momentum, x and y where the muon leaves the target (z = -0.0045 m; the file stores
the momentum AFTER the target, not at production: results/ref2018, docs/STATUS.md; pool v1-v3 scaled our rows up to
the production momentum, wrongly), kept if they reach the 2018 plane with E > 10 GeV (loss drawn from the 2018 map,
from the production momentum, extra column p_prod; nan for the file's rows); weights w_decay / rep x 5e13 / events, then a factor per (p, pT) cell (4 % in p, 0.2 GeV in pT; the p-only factor where the file has < 50 effective muons
in the cell) so each part has the file's flux of that component in every cell (pool v3; v2 matched p only and left
+-20 % differences in pT, partly because our rows carry the direction at the target exit, the file the one at production): the fresh muons add statistics, the file keeps the spectrum
(our Pythia 6 + Geant4 min. bias is softer than the 2018 FairShip production: up to 1.5x at 30 GeV, 1.05x at 150 GeV;
one flat factor left -25 % / +10 % steps at the source boundaries, bin/pool_edges.py). Every row is then kept with the source-bin probability q
(results/thinning/q_bin.npy, bin/input_thinning_study.py: 20 % rule, floor 0.01), weight / q.
Hidden set: HIDDEN of the source events (fresh: per proton event, so its muons stay together; old rows: per row),
never used for training. Each chunk's weights are scaled per component to the whole pool's flux: every chunk is a
full spill on its own.

    bin/remote.sh run bin/build_pool.py --src data/source_pool1 --out /disk/lhcb_data/sqasim/pool_v4
Writes <out>/train_NNN.h5, <out>/hidden_NNN.h5 (columns as the input file + pool_index, component, fresh, q) and
<out>/pool.json.
"""

import _bootstrap  # noqa: F401

import argparse
import glob
import json
import time
from pathlib import Path

import h5py
import numpy as np

from veto import paths

import input_thinning_study as its
import old_convention as oc
import os
FVS_ROOT = os.environ.get("VETO_FVS", "/disk/users/sqasim/workspace/uzh/fast-veto-sim")   # read only

FVS_MB2 = f"{FVS_ROOT}/data/tail/mb2_[0-9]*.npz"     # read only: mean loss for charm
COLS = ("px", "py", "pz", "x", "y", "z", "pdg", "weight")
MB_LO = 30.0                     # fresh min. bias from here: --emin 20 keeps muons above 20 GeV at the plane
PARTS = {                        # tag: (component, p range at production)
    "pool_mb20": (1, (MB_LO, 150.0)),                        # momentum after the target (pool v1-v3: at production)
    "pool_tail": (0, (150.0, np.inf)),
    "pool_dmhigh": (2, (150.0, np.inf)),
    "charmc": (3, (20.0, np.inf)),
}
SHAPE_BINS = np.geomspace(10.0, 1000.0, 121)  # 4 % wide momentum bins: the fresh parts take the file's flux in each
PT_BINS = np.arange(0.2, 4.01, 0.2)            # and 0.2 GeV pT bins within them (the angle at production)
MIN_NEFF_2D = 50.0                            # a (p, pT) cell needs this many effective file muons, else the p factor


def components(w, src):
    comp = np.full(len(w), 4, np.int8)
    comp[np.abs(w - 768.75) < 0.5] = 1
    comp[np.abs(w - 7.6875) < 0.01] = 2
    comp[np.abs(w - 326.16) < 0.5] = 3
    comp[src == 1] = 0
    return comp


def bin_index(comp, px, py, pz, pdg):
    pt = np.hypot(px, py).astype(np.float64)
    p = np.sqrt(pt ** 2 + pz.astype(np.float64) ** 2)
    ip = np.clip(np.digitize(np.log10(p), its.LOGP) - 1, 0, len(its.LOGP) - 1)
    it = np.clip(np.digitize(np.arctan2(pt, pz), its.THETA) - 1, 0, len(its.THETA) - 2)
    return ((comp.astype(np.int64) * len(its.LOGP) + ip) * (len(its.THETA) - 1) + it) * 2 + (pdg > 0)


def thin(rows, q_bin, rng):
    """Keep each row with its bin's q, weight / q."""
    q = q_bin[bin_index(rows["component"], rows["px"], rows["py"], rows["pz"], rows["pdg"])]
    keep = rng.random(len(q)) < q
    out = {k: v[keep] for k, v in rows.items()}
    out["q"] = q[keep].astype(np.float32)
    out["weight"] = (out["weight"] / out["q"]).astype(np.float64)
    return out


def mean_loss_fn():
    """Creation -> plane mean loss vs plane momentum, from FVS's min. bias run (as bin/old_convention.py)."""
    ps = [np.load(f) for f in sorted(glob.glob(FVS_MB2))[:20]]
    p = np.concatenate([np.sqrt(x["px"].astype(np.float64) ** 2 + x["py"] ** 2 + x["pz"] ** 2) for x in ps])
    pp = np.concatenate([x["p_prod"] for x in ps])
    pl = np.geomspace(5, 400, 30)
    k = np.digitize(p, pl)
    ml = np.array([np.median((pp - p)[k == i]) if np.sum(k == i) > 50 else np.nan for i in range(1, len(pl))])
    good = np.isfinite(ml)
    return lambda x: np.interp(x, np.sqrt(pl[1:] * pl[:-1])[good], ml[good])


def shape_bin(p):
    """Index in SHAPE_BINS (0 below 10 GeV, len(SHAPE_BINS) above 1 TeV)."""
    return np.digitize(p, SHAPE_BINS)


def shape_bin2(p, pt):
    """Index of the (p, pT) cell: SHAPE_BINS x PT_BINS (last pT bin: >= 4 GeV)."""
    return shape_bin(p) * (len(PT_BINS) + 1) + np.digitize(pt, PT_BINS)


N2 = (len(SHAPE_BINS) + 1) * (len(PT_BINS) + 1)


def fresh_part(src, tag, comp, prange, table, mean_loss, q_bin, rng):
    """One fresh part, selected and thinned slice by slice (the raw tail is > 1e9 muons). Weights per spill before
    the flux factor; also the raw sums for that factor and the Kish n_eff before thinning."""
    files = sorted(glob.glob(str(Path(src) / "tail" / f"{tag}_[0-9]*.npz")))
    events, s1, s2, nsel = 0.0, 0.0, 0.0, 0
    hb = np.zeros(len(SHAPE_BINS) + 1)                                  # raw flux per momentum bin (before q)
    hb2 = np.zeros(N2)                                                  # and per (p, pT) cell
    parts = []
    for f in files:
        d = np.load(f)
        events += float(d["events"])
        p = np.sqrt(d["px"].astype(np.float64) ** 2 + d["py"] ** 2 + d["pz"] ** 2)
        pp = p + mean_loss(p) if tag == "charmc" else d["p_prod"].astype(np.float64)
        m = (p >= prange[0]) & (p < prange[1])
        m[m] &= pp[m] - oc.draw_loss(table, pp[m]) > 10.0               # the file's selection
        k = int(Path(f).stem.rsplit("_", 1)[1])
        ev = d["event"][m].astype(np.int64) if "event" in d.files else np.arange(m.sum())
        rows = dict(px=d["px"][m].astype(np.float64), py=d["py"][m].astype(np.float64), pz=d["pz"][m].astype(np.float64),
                    x=d["x"][m].astype(np.float64), p_prod=pp[m],
                    y=d["y"][m].astype(np.float64), z=np.full(m.sum(), -0.0045), pdg=d["pdg"][m].astype(np.float64),
                    weight=d["w_decay"][m].astype(np.float64) / float(d["rep"]),
                    component=np.full(m.sum(), comp, np.int8), fresh=np.ones(m.sum(), np.int8),
                    group=((comp + 1) << 60) + (k << 32) + ev)
        s1 += rows["weight"].sum()
        s2 += np.sum(rows["weight"] ** 2)
        hb += np.bincount(shape_bin(p[m]), weights=rows["weight"], minlength=len(SHAPE_BINS) + 1)
        hb2 += np.bincount(shape_bin2(p[m], np.hypot(rows["px"], rows["py"])), weights=rows["weight"], minlength=N2)
        nsel += int(m.sum())
        parts.append(thin(rows, q_bin, rng))
    rows = {c: np.concatenate([x[c] for x in parts]) for c in parts[0]}
    norm = 5e13 / events
    rows["weight"] *= norm
    print(f"  {tag}: {len(files)} slices, {events:.3g} p.o.t., {nsel:,} muons selected, {len(rows['px']):,} after q", flush=True)
    return rows, dict(slices=len(files), pot=events, selected=nsel, kept=int(len(rows["px"])), flux_raw=s1 * norm,
                      flux_raw_bins=(hb * norm).tolist(), flux_raw_bins2=hb2 * norm,
                      kish_before_thinning=s1 ** 2 / s2 if s2 > 0 else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(paths.PROJECT_DIR / "data" / "source_pool1"))
    ap.add_argument("--out", default=str(paths.source_data() / "pool_v4"))
    ap.add_argument("--chunk", type=float, default=1e8)
    ap.add_argument("--hidden", type=float, default=0.15)
    ap.add_argument("--q", default=str(paths.results_dir() / "thinning" / "q_bin.npy"))
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(20261030)
    q_bin = np.load(args.q)
    table = oc.loss_map()
    mean_loss = mean_loss_fn()

    with h5py.File(paths.enriched_input_h5(), "r") as f:
        old = {c: f[c][:] for c in COLS}
        comp = components(old["weight"], f["source"][:])
    p = np.sqrt(old["px"].astype(np.float64) ** 2 + old["py"] ** 2 + old["pz"] ** 2)
    keep = ((comp == 1) & (p < MB_LO)) | ((comp == 2) & (p < 150)) | ((comp == 3) & (p < 20)) | (comp == 4)
    # flux of the file per replaced part (the fresh parts are normalised to it)
    flux = {tag: float(old["weight"][(comp == c) & (p >= lo) & (p < hi)].astype(np.float64).sum()) for tag, (c, (lo, hi)) in PARTS.items()}
    flux_bins = {tag: np.bincount(shape_bin(p[(comp == c) & (p >= lo) & (p < hi)]),
                                  weights=old["weight"][(comp == c) & (p >= lo) & (p < hi)].astype(np.float64),
                                  minlength=len(SHAPE_BINS) + 1) for tag, (c, (lo, hi)) in PARTS.items()}
    pt = np.hypot(old["px"].astype(np.float64), old["py"])
    flux_bins2 = {}                                                       # (sum w, sum w^2) per (p, pT) cell
    for tag, (c, (lo, hi)) in PARTS.items():
        m = (comp == c) & (p >= lo) & (p < hi)
        k, w = shape_bin2(p[m], pt[m]), old["weight"][m].astype(np.float64)
        flux_bins2[tag] = (np.bincount(k, weights=w, minlength=N2), np.bincount(k, weights=w * w, minlength=N2))
    del pt
    idx = np.nonzero(keep)[0]
    rows = {c: old[c][idx] for c in COLS}
    rows["weight"] = rows["weight"].astype(np.float64)
    rows["component"] = comp[idx]
    rows["fresh"] = np.zeros(len(idx), np.int8)
    rows["group"] = idx.astype(np.int64)                                  # old rows: one group each (< 2^30)
    rows["p_prod"] = np.full(len(idx), np.nan)                            # not in the file
    del old, comp, p
    print(f"old rows kept: {len(idx):,} of 601 M, {time.time() - t0:.0f} s", flush=True)
    pieces = [thin(rows, q_bin, rng)]
    info = dict(old_rows=int(len(idx)), parts={})
    for tag, (c, prange) in PARTS.items():
        th, pinfo = fresh_part(args.src, tag, c, prange, table, mean_loss, q_bin, rng)
        factor = flux[tag] / pinfo["flux_raw"]                            # the file's flux of this part
        raw_b = np.array(pinfo["flux_raw_bins"])
        with np.errstate(divide="ignore", invalid="ignore"):              # the file's spectrum: factor per bin
            fb = np.where((raw_b > 0) & (flux_bins[tag] > 0), flux_bins[tag] / raw_b, factor)
        f1, f2 = flux_bins2[tag]                                          # the file's (p, pT) distribution: factor per cell
        raw2 = pinfo.pop("flux_raw_bins2")
        with np.errstate(divide="ignore", invalid="ignore"):
            good = (raw2 > 0) & (f2 > 0) & (f1 ** 2 / np.where(f2 > 0, f2, 1) >= MIN_NEFF_2D)
            fb2 = np.where(good, f1 / raw2, np.repeat(fb, len(PT_BINS) + 1))
        pr = np.sqrt(th["px"] ** 2 + th["py"] ** 2 + th["pz"] ** 2)        # momentum after the target, as the file
        th["weight"] *= fb2[shape_bin2(pr, np.hypot(th["px"], th["py"]))]
        info["parts"][tag] = dict(component=c, p_range=[prange[0], min(prange[1], 1e9)], flux_factor=factor, file_flux=flux[tag],
                                  bin_factors=fb.tolist(), cells_2d=int(good.sum()), cells_fallback=int(((raw2 > 0) & ~good).sum()),
                                  **pinfo)
        print(f"  {tag}: flux factor {factor:.3f}, n_eff before thinning {pinfo['kish_before_thinning']:.4g}, {time.time() - t0:.0f} s", flush=True)
        pieces.append(th)
    pool = {k: np.concatenate([x[k] for x in pieces]) for k in pieces[0]}
    del pieces
    n = len(pool["px"])
    pool["pool_index"] = np.arange(n, dtype=np.int64)
    total = np.bincount(pool["component"], weights=pool["weight"], minlength=5)
    # hidden set by source event (hash of the group id), the rest in random chunks
    h = pool["group"].astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)       # Fibonacci hash: same event, same u
    u = (h >> np.uint64(40)).astype(np.float64) / float(1 << 24)
    del h
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, sel in (("train", u >= args.hidden), ("hidden", u < args.hidden)):
        ids = np.nonzero(sel)[0]
        rng.shuffle(ids)
        nch = max(1, int(round(len(ids) / args.chunk)))
        for j, part in enumerate(np.array_split(ids, nch)):
            part = np.sort(part)
            w = pool["weight"][part].copy()
            cw = np.bincount(pool["component"][part], weights=w, minlength=5)
            w *= np.where(cw > 0, total / np.maximum(cw, 1e-300), 0)[pool["component"][part]]   # each chunk: one spill
            with h5py.File(out / f"{name}_{j:03d}.h5.tmp", "w") as g:
                for c in COLS:
                    g.create_dataset(c, data=(w if c == "weight" else pool[c][part]).astype(np.float32), chunks=True)
                for c, dt in (("pool_index", np.int64), ("component", np.int8), ("fresh", np.int8), ("q", np.float32),
                              ("p_prod", np.float32)):
                    g.create_dataset(c, data=pool[c][part].astype(dt), chunks=True)
                g.attrs["description"] = f"input pool v4 ({name} {j} of {nch}), bin/build_pool.py"
            (out / f"{name}_{j:03d}.h5.tmp").rename(out / f"{name}_{j:03d}.h5")
            print(f"  {name}_{j:03d}: {len(part):,} muons, {time.time() - t0:.0f} s", flush=True)
        info[name] = dict(muons=int(len(ids)), chunks=nch)
    info.update(pot_per_spill=5e13,                     # D13: weights per 5e13 (as the 2018 file); readers -> 4e13
                pool_muons=int(n), flux_by_component=total.tolist(), hidden_fraction=args.hidden, chunk=args.chunk,
                kept_by_component=np.bincount(pool["component"], minlength=5).tolist(),
                fresh_by_component=np.bincount(pool["component"], weights=pool["fresh"], minlength=5).tolist())
    json.dump(info, open(out / "pool.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "parts"}), f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
