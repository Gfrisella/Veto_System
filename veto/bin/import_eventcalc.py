#!/usr/bin/env python3
"""Convert the EventCalc LLP signal samples (*_data.dat) into a memory-mappable column store (ISSUES B4).

Only READS the existing EventCalc outputs; nothing is generated or selected. Source (default):
    ~/work_dir/EventCalc_signal_rejection/EventCalc_run/outputs/<model>/eventData/*_data.dat
(43 benchmarks made on 2026-11-04 by generate.sh: Pythia 8 LCG_106, no exHad, 50k sampled points,
EventCalc 7c9be2a with the OLD decay volume z 32-82 m from the target centre, full height 6.2 m at the exit.)

Output (default veto/data/signal/eventcalc/):
    events/      column store (veto.io.write_store), one row per LLP decay, all benchmarks concatenated
    particles/   column store, one row per final-state particle; events/first + events/n_part index it
    benchmarks.json   per-benchmark header numbers, file names, event ranges
    meta.json         string tables (models, channels), frame and normalisation constants, provenance

Frame (veto frame, z_veto = z_FairShip - 2.14), following translate_for_fairship.py (2026-11-05 convention:
EventCalc decay-volume entrance on the FairShip vessel entrance 33.27 m):
    z_FairShip = z_EventCalc + (VESSEL_Z0_FAIRSHIP - VESSEL_Z0_EVENTCALC)  = z_EC + 1.27   (z_min = 32 samples)
    z_veto     = z_FairShip - FS_OFFSET                                    = z_EC - 0.87
x, y are unchanged. The LLP production point (EventCalc origin) lands at z_veto = -0.87 (NOT Z_TARGET = -2.14).

Weight: w = N_ev_tot * P_decay / sum_file(P_decay) = expected visible LLP decays inside EventCalc's decay volume
for N_pot = 6e20 (EventCalc DEFAULT_N_POT) at the benchmark coupling. sum(w) per benchmark = header N_ev_tot.

    bin/import_eventcalc.py [--src DIR] [--out DIR] [--vessel-z0-eventcalc 32.0]
"""

import _bootstrap  # noqa: F401

import argparse
import datetime
import glob
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np

from veto import geometry as geo
from veto import paths
from veto.io import write_store

SRC_DEFAULT = Path("/home/hep/gfrise/work_dir/EventCalc_signal_rejection/EventCalc_run/outputs")
EVENTCALC_REPO = Path("/home/hep/gfrise/work_dir/EventCalc")

VESSEL_Z0_FAIRSHIP = 33.27      # translate_for_fairship.py (FairShip 5987c9db decayVolume.z0)
VESSEL_Z0_EVENTCALC = 32.0      # z_min of the samples (old EventCalc geometry); 32.4768 after dcdcfc1
N_POT = 6.0e20                  # EventCalc funcs/simulation_config.py DEFAULT_N_POT (= funcs/LLP_selection.py N_pot)

HEADER = {
    "n_sampled": r"Sampled\s+([0-9.]+[eE][+-]?[0-9]+)\s+events",
    "coupling_squared": r"Squared coupling:\s+([0-9.]+[eE][+-]?[0-9]+)",
    "n_llp_tot": r"Total number of produced LLPs:\s+([0-9.]+[eE][+-]?[0-9]+)",
    "eps_polar": r"Polar acceptance:\s+([0-9.]+[eE][+-]?[0-9]+)",
    "eps_azimuthal": r"Azimuthal acceptance:\s+([0-9.]+[eE][+-]?[0-9]+)",
    "p_decay_avg": r"Averaged decay probability:\s+([0-9.]+[eE][+-]?[0-9]+)",
    "br_visible": r"Visible Br Ratio:\s+([0-9.]+[eE][+-]?[0-9]+)",
    "n_ev_tot": r"Total number of events:\s+([0-9.]+[eE][+-]?[0-9]+)",
}
BLOCK = re.compile(r"^#<process=([^;]*);\s*sample_points=(\d+)>")
NAME = re.compile(r"^(?P<llp>[^_]+)_(?P<m>[^_]+)_(?P<ct>[^_]+)_(?P<rest>.*)_data\.dat$")

# Charge (in units of e) of the particle with positive PDG code; the antiparticle has the opposite sign.
CHARGE = {11: -1, 13: -1, 15: -1, 211: 1, 321: 1, 2212: 1, 411: 1, 431: 1, 521: 1, 213: 1, 323: 1,
          3112: -1, 3222: 1, 3312: -1, 3334: -1, 24: 1, 2224: 2, 1114: -1, 4122: 1}
NEUTRAL = {12, 14, 16, 22, 111, 113, 130, 310, 2112, 3122, 3212, 3322, 221, 223, 333, 331, 421, 311, 313,
           421, 511, 531, 23, 25, 2114, 3214, 443}
# charged particles that live long enough to leave a track through the vessel and straw stations
TRACK_PDG = {11, 13, 211, 321, 2212}


def charge_of(pdg):
    a = np.abs(pdg)
    q = np.zeros(pdg.shape, np.int8)
    for code, c in CHARGE.items():
        q[a == code] = c
    unknown = set(np.unique(a[(q == 0)]).tolist()) - NEUTRAL
    return np.where(pdg < 0, -q, q).astype(np.int8), unknown


def describe(path):
    m = NAME.match(os.path.basename(path))
    llp, mass, ctau, rest = m["llp"], float(m["m"]), float(m["ct"]), m["rest"]
    model, mixing = llp, None
    if llp == "HNL":
        mixing = [float(x) for x in rest.split("_")[:3]]
        model = "HNL-" + ("e", "mu", "tau")[int(np.argmax(mixing))]
    return dict(model=model, mass=mass, ctau=ctau, mixing_pattern=mixing,
                name=f"{model}_m{mass:g}_ct{ctau:g}")


def read_file(path):
    """Header numbers and the list of (channel, declared, rows[n, 10 + 6k]) blocks; 1-row blocks are fine."""
    with open(path) as f:
        head = f.readline()
        hdr = {k: float(re.search(p, head).group(1)) for k, p in HEADER.items()}
        blocks, cur = [], None
        for line in f:
            s = line.strip()
            if not s:
                continue
            m = BLOCK.match(s)
            if m:
                cur = [m.group(1), int(m.group(2)), []]
                blocks.append(cur)
            else:
                cur[2].append(s)
    out = []
    for ch, declared, lines in blocks:
        if not lines:
            out.append((ch, declared, np.empty((0, 10))))
            continue
        ncol = len(lines[0].split())
        flat = np.fromstring(" ".join(lines), sep=" ", dtype=np.float64)
        if (ncol - 10) % 6 or flat.size != ncol * len(lines):
            raise ValueError(f"{path}: block {ch}: ragged rows or bad width {ncol}")
        out.append((ch, declared, flat.reshape(len(lines), ncol)))
    return head.strip(), hdr, out


def git_info(repo):
    try:
        return subprocess.check_output(["git", "-C", str(repo), "log", "-1", "--format=%H %ci %D"],
                                       text=True).strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({e})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=SRC_DEFAULT)
    ap.add_argument("--out", type=Path, default=paths.PROJECT_DIR / "data" / "signal" / "eventcalc")
    ap.add_argument("--vessel-z0-eventcalc", type=float, default=VESSEL_Z0_EVENTCALC)
    ap.add_argument("--vessel-z0-fairship", type=float, default=VESSEL_Z0_FAIRSHIP)
    a = ap.parse_args()

    dz_fs = a.vessel_z0_fairship - a.vessel_z0_eventcalc          # z_FairShip - z_EventCalc
    dz = dz_fs - geo.FS_OFFSET                                    # z_veto - z_EventCalc
    print(f"z_veto = z_EventCalc {dz:+.4f} m  (z_FS = z_EC {dz_fs:+.4f}, z_veto = z_FS - {geo.FS_OFFSET})")

    files = sorted(glob.glob(str(a.src / "*" / "eventData" / "*_data.dat")))
    if not files:
        raise SystemExit(f"no *_data.dat under {a.src}")

    models, channels = [], []
    ev = {k: [] for k in ("bench", "model", "channel", "mass", "ctau", "coupling_squared", "llp_pdg",
                          "x", "y", "z", "z_eventcalc", "px", "py", "pz", "e", "p_decay", "weight",
                          "in_vessel", "n_part", "n_charged", "n_track")}
    pt = {k: [] for k in ("event", "pdg", "px", "py", "pz", "e", "m", "charge", "is_track")}
    benches, unknown_all, n_ev_global = [], set(), 0

    for b, path in enumerate(files):
        info = describe(path)
        head, hdr, blocks = read_file(path)
        if info["model"] not in models:
            models.append(info["model"])
        rows = [r for _, _, r in blocks if len(r)]
        n = sum(len(r) for r in rows)
        sum_p = float(sum(r[:, 6].sum() for r in rows))
        bad_blocks = [(ch, d, len(r)) for ch, d, r in blocks if d != len(r)]
        one_row = [ch for ch, _, r in blocks if len(r) == 1]
        ev_first = n_ev_global
        zmin_ec, zmax_ec = np.inf, -np.inf
        for ch, _, r in blocks:
            if not len(r):
                continue
            if ch not in channels:
                channels.append(ch)
            k = len(r)
            nprod = (r.shape[1] - 10) // 6
            prod = r[:, 10:].reshape(k, nprod, 6)
            real = prod[:, :, 5] != -999
            pdg = prod[:, :, 5].astype(np.int64)
            q, unk = charge_of(np.where(real, pdg, 22))
            unknown_all |= unk
            trk = real & (q != 0) & np.isin(np.abs(pdg), list(TRACK_PDG))
            zec = r[:, 9]
            zmin_ec, zmax_ec = min(zmin_ec, zec.min()), max(zmax_ec, zec.max())
            zv = zec + dz
            ev["bench"].append(np.full(k, b, np.int16))
            ev["model"].append(np.full(k, models.index(info["model"]), np.int8))
            ev["channel"].append(np.full(k, channels.index(ch), np.int16))
            ev["mass"].append(np.full(k, info["mass"], np.float32))
            ev["ctau"].append(np.full(k, info["ctau"], np.float32))
            ev["coupling_squared"].append(np.full(k, hdr["coupling_squared"]))
            ev["llp_pdg"].append(r[:, 5].astype(np.int64))
            ev["x"].append(r[:, 7]); ev["y"].append(r[:, 8]); ev["z"].append(zv); ev["z_eventcalc"].append(zec)
            ev["px"].append(r[:, 0]); ev["py"].append(r[:, 1]); ev["pz"].append(r[:, 2]); ev["e"].append(r[:, 3])
            ev["p_decay"].append(r[:, 6])
            ev["weight"].append(hdr["n_ev_tot"] * r[:, 6] / sum_p)
            ev["in_vessel"].append(geo.inside_vessel(r[:, 7], r[:, 8], zv))
            ev["n_part"].append(real.sum(1).astype(np.int16))
            ev["n_charged"].append((real & (q != 0)).sum(1).astype(np.int16))
            ev["n_track"].append(trk.sum(1).astype(np.int16))
            idx = np.nonzero(real)                                 # row-major: particles stay grouped per event
            pt["event"].append((n_ev_global + idx[0]).astype(np.int64))
            pt["pdg"].append(pdg[idx].astype(np.int32))
            for j, key in enumerate(("px", "py", "pz", "e", "m")):
                pt[key].append(prod[:, :, j][idx].astype(np.float32))
            pt["charge"].append(q[idx])
            pt["is_track"].append(trk[idx])
            n_ev_global += k
        benches.append(dict(
            id=b, name=info["name"], model=info["model"], mass=info["mass"], ctau=info["ctau"],
            mixing_pattern=info["mixing_pattern"], file=str(path), header=head, **hdr,
            n_events=n, event_range=[ev_first, n_ev_global], sum_p_decay=sum_p,
            mean_p_decay=sum_p / max(n, 1), z_eventcalc_range=[float(zmin_ec), float(zmax_ec)],
            channels=[[ch, len(r)] for ch, _, r in blocks], one_row_blocks=one_row,
            mismatched_blocks=bad_blocks))
        print(f"[{b:2d}] {info['name']:<26} {n:6d} ev  {len(blocks):2d} blocks  N_ev_tot {hdr['n_ev_tot']:.4e}  "
              f"<P>={sum_p / n:.6e} (hdr {hdr['p_decay_avg']:.6e})  z_EC {zmin_ec:.3f}-{zmax_ec:.3f}"
              + (f"  1-row blocks: {one_row}" if one_row else "")
              + (f"  MISMATCH {bad_blocks}" if bad_blocks else ""))

    cat = {k: np.concatenate(v) for k, v in ev.items()}
    pcat = {k: np.concatenate(v) for k, v in pt.items()}
    counts = cat["n_part"].astype(np.int64)
    cat["first"] = np.r_[0, np.cumsum(counts)[:-1]]
    assert len(pcat["event"]) == counts.sum()
    assert np.array_equal(pcat["event"], np.repeat(np.arange(len(counts)), counts))
    if unknown_all:
        print("WARNING: PDG codes treated as neutral but not in the known-neutral list:", sorted(unknown_all))

    a.out.mkdir(parents=True, exist_ok=True)
    frame = dict(
        description="veto frame: z_veto = z_FairShip - 2.14; z_veto = z_EventCalc + dz_veto (x, y unchanged)",
        dz_veto=dz, dz_fairship=dz_fs, fs_offset=geo.FS_OFFSET,
        vessel_z0_fairship=a.vessel_z0_fairship, vessel_z0_eventcalc=a.vessel_z0_eventcalc,
        production_point_veto=[0.0, 0.0, dz], z_target_veto=geo.Z_TARGET,
        eventcalc_volume=dict(z=[a.vessel_z0_eventcalc, a.vessel_z0_eventcalc + 50.0], full_x=[1.0, 4.0],
                              full_y=[2.7, 6.2], origin="target centre"),
        in_vessel=dict(function="veto.geometry.inside_vessel", z=[geo.Z_VESSEL_MIN, geo.Z_VESSEL_MAX],
                       half_x=[geo.VESSEL_X_IN, geo.VESSEL_X_OUT], half_y=[geo.VESSEL_Y_IN, geo.VESSEL_Y_OUT]))
    meta_common = dict(models=models, channels=channels, frame=frame, n_pot=N_POT,
                       created=datetime.datetime.now().isoformat(timespec="seconds"),
                       source=str(a.src), eventcalc_git=git_info(EVENTCALC_REPO),
                       eventcalc_run_note="EventCalc_run = copy of EventCalc 7c9be2a (not a git repo); samples made "
                                          "2026-11-04 with z 32-82 m / 6.2 m geometry, Pythia 8 (LCG_106), no exHad")
    write_store(a.out / "events", cat, dict(kind="eventcalc_events", **meta_common))
    write_store(a.out / "particles", pcat, dict(kind="eventcalc_particles"))
    with open(a.out / "benchmarks.json", "w") as f:
        json.dump(benches, f, indent=1)
    with open(a.out / "meta.json", "w") as f:
        json.dump(dict(meta_common, n_events=int(len(counts)), n_particles=int(counts.sum()),
                       n_benchmarks=len(benches), track_pdg=sorted(TRACK_PDG)), f, indent=1)
    print(f"{len(benches)} benchmarks, {len(counts):,} events, {int(counts.sum()):,} particles -> {a.out}")


if __name__ == "__main__":
    main()
