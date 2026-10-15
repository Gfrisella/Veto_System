#!/usr/bin/env python3
"""AlphaEvolve-style search over veto designs: an LLM proposes design JSONs, the cluster scores them.

Same recipe as supply2/srq_alphaevolve/evolve_heuristic.py, with designs instead of code:
  1. a population of designs (seeded with designs/*.json) and their evaluation reports;
  2. each generation, children are proposed by the LLM from a rank-weighted parent, the top of the
     population as inspiration and a directive (refine / explore / combine), with the full problem
     context: geometry and limits, the JSON schema, technologies, where the muons are, the objective;
  3. all children of a generation are evaluated in parallel on the cluster (evaluate_designs.py,
     sent over our SSH proxy: the LLM key never leaves this machine);
  4. truncation selection on objective.score (placeholder until the real objective is in).
Every candidate, its report and the LLM's rationale go to results/evolve/<run>/history.jsonl; the best
design so far is written to results/evolve/<run>/best.json (a normal designs/*.json file).

  bin/remote.sh sync                                  # once, so the cluster has the current code
  evolve/evolve_designs.py --generations 10 --children 8
  evolve/evolve_designs.py --model anthropic/claude-sonnet-4.5 --generations 20

The OpenRouter key is read from OPENROUTER_API_KEY, else from the .env file at --env-file
(default: the supply2 project's .env, like its openrouter_client.py).
"""

import argparse
import functools
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VETO = HERE.parent
sys.path[:0] = [str(HERE), str(VETO / "lib")]

import objective  # noqa: E402
from veto import design as dsg  # noqa: E402
from veto import geometry as geo  # noqa: E402
from veto import muon_types as mt  # noqa: E402
from veto import paths  # noqa: E402
from veto import technology as tech  # noqa: E402

print = functools.partial(print, flush=True)

REMOTE_VETO = "/disk/users/sqasim/workspace/uzh/Veto_System/veto"
EXEC = "bash /disk/lhcb_data/sqasim/images/shared/exec_cudabear_november_2025.sh python3"
SEEDS = ["D1_wide_ubt", "D2_default", "D3_heavy_right", "D6_tile2", "D6_fiber", "D6_straw"]
DIRECTIVES = {
    "refine": "Refine the parent: keep its structure, fix its clearest weakness from the report "
              "(a plane over its rate limit, a plane costing signal for little veto, a gap that lets muons through).",
    "explore": "Try a substantially different layout from the parent (other fin positions, reaches, heights, "
               "technologies, or fewer / more fins), even at a risk.",
    "combine": "Combine the strongest ideas of the parent and the inspirations into one design.",
}


# ------------------------------------------------------------------ problem context for the LLM
def flux_summary():
    """Muons per spill crossing fin planes, per side, z and band of distance from the vessel wall."""
    f = np.load(paths.results_dir() / "reco" / "inputs.npz")
    maps, zs, ub, yb = f["fin_maps"], f["z_slices"], f["u_bins"], f["y_bins"]
    uc, yc = 0.5 * (ub[1:] + ub[:-1]), 0.5 * (yb[1:] + yb[:-1])
    bands = [(0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 5)]
    lines = ["Muons per spill (1 s spill, i.e. Hz) crossing an x-y plane at z, outside the vessel, per m^2,",
             "averaged in bands of distance u from the vessel wall, for |y| < 1 m and 1 < |y| < 3 m:"]
    for s_idx, side in ((1, "right (-x)"), (0, "left (+x)")):
        for z in (30.6, 40.6, 50.6, 60.6, 70.6, 80.6):
            iz = int(np.argmin(np.abs(zs - z)))
            cells = []
            for ylo, yhi in ((0, 1), (1, 3)):
                ym = (np.abs(yc) >= ylo) & (np.abs(yc) < yhi)
                vals = []
                for a, b in bands:
                    um = (uc >= a) & (uc < b)
                    area = (b - a) * (np.diff(yb)[ym].sum())
                    vals.append(f"{maps[s_idx, iz][np.ix_(um, ym)].sum() / area:.1e}")
                cells.append(f"|y|{'<1' if ylo == 0 else ' 1-3'}: " + " ".join(vals))
            lines.append(f"  {side:<10} z={z:5.1f}  u bands {[f'{a}-{b}' for a, b in bands]}  " + "  ".join(cells))
    f.close()
    return "\n".join(lines)


def technology_table():
    rows = []
    for key, t in tech.TECHNOLOGIES.items():
        timing = f"{t['sigma_t'] * 1e9:.2f} ns" if t["sigma_t"] else f"drift window {t['time_window'] * 1e9:.0f} ns"
        rows.append(f"  {key:<6} {t['name']}: pitch {t['pitch'] * 1e3:g} mm, timing {timing}, "
                    f"veto window {'+-%.1f ns' % (t['veto_window'] * 1e9) if t['veto_window'] else '+-3 sigma'}, "
                    f"efficiency {t['efficiency']}, x0 {tech.x0(t) * 100:.2f} % per plane, "
                    f"cost {t['cost_m2_per_layer']} kCHF per layer-m^2 x {tech.n_layers(t)} layers, "
                    f"rate limit {t['max_rate'] or '-'} Hz/cm^2, {t['max_channel_rate'] or '-'} Hz per channel")
    return "\n".join(rows)


def system_prompt():
    return f"""You design the veto (UBT + side fins) of the SHiP decay vessel. You propose COMPLETE design JSON files;
each is simulated and reconstructed (196 736 muon-pair background candidates, signal decays, 200 k single muons)
and scored. Your goal: the highest score.

GEOMETRY (metres; z along the beam; looking from the target, right = -x, left = +x):
- Decay vessel: frustum z = 30.6 .. 80.6, half widths x 0.5 -> 2.0, y 1.35 -> 3.0. SST1 tracker at z = 82
  (|x| < 2, |y| < 3), the spectrometer that measures the tracks. Cavern wall at x = -4.56, floor y = -3.36.
- UBT: a straw plane at z = 30.6 on the vessel entrance, |x| < 0.5 (fixed), y from y_lo to y_hi; it must cover
  at least y = -1.35 .. 1.35 and stays above y = -3.2. Always straws (technology fixed).
- Fins: thin vertical x-y planes at a fixed z (30.6 .. 82) on the right or left side. A fin touches the vessel
  side wall with its inner edge (|x| = X(z) = 0.5 + 0.03 (z - 30.6)) and reaches OUT in x by `length`; it covers
  y from y_lo (>= -3.2) to y_hi. Right fins are cut at the cavern wall x = -4.56 (the loader does it and notes it).
  Each fin has a technology (straw, fiber, pixel, tile, tile2); tilt must be 0.
- No detector on top of or below the vessel.

HOW THE VETO WORKS: a background candidate is a pair of muons forming a fake vertex in the vessel. Each
spectrometer track is extrapolated backwards in a straight line to every plane upstream of its vertex; a hit
in a space window (cm, grows with distance and 1/p) and time window explains the track; a pair is vetoed if
either track is explained. Signal (a neutral from the target decaying in the vessel) has no hits of its own and
is lost only when a RANDOM hit of the other 10^9 - 10^10 muons/s falls in its search window: signal loss grows
with the plane's muon density x cell size x time window, so fast, fine-grained planes away from the hot
regions lose little signal. Muons cross the planes mostly in a hot lobe near y = 0, 1.5 - 3 m from the wall.
Fins at z = 30.6 veto most of the background; later fins catch muons entering the vessel further down.
Muons that bounce off the cavern wall (right side) can only be followed back after their bounce.

TECHNOLOGIES:
{technology_table()}

WHERE THE MUONS ARE:
{flux_summary()}

OBJECTIVE (higher is better):
{objective.__doc__}

DESIGN JSON (all lengths in metres):
{{"name": "<short unique name>", "description": "<one line>",
  "ubt": {{"y_lo": -1.35, "y_hi": 1.35}},
  "fins": [{{"side": "right"|"left", "z": 30.6, "length": 3.0, "y_lo": -3.0, "y_hi": 3.0, "tilt": 0,
            "technology": "tile"}}, ...],
  "detector": {{}}}}
A fin may also give "cell_u"/"cell_y" (readout cell size, m) to read a technology out more coarsely.

Return JSON only: {{"design": <the complete design JSON>, "rationale": "<two sentences: what you changed and why>"}}"""


def compact(report):
    """The parts of an evaluation report the LLM needs, rounded."""
    if report.get("error"):
        return {"error": report["error"]}
    sc, terms = objective.score(report)
    f = report["fom"]
    return {
        "score": round(sc, 3), "score_terms": {k: round(v, 3) for k, v in terms.items()},
        "signal_lost_pct": round(100 * f["FPR"], 3), "background_vetoed_pct": round(100 * f["TPR"], 5),
        "surviving_pairs": f"{f['n_survive']} of {f['n_candidates']}",
        "background_15y": f"{'<' if f['B_15y_is_upper_limit'] else ''}{f['B_15y'] if f['B_15y'] > 0 else f['B_15y_err']:.2g}",
        "cost_MCHF": round(report["cost_kchf"] / 1000, 2), "channels": report["n_channels"],
        "planes_over_rate_limit": report["n_planes_over_limit"],
        "single_muons_vetoed_pct": {k: round(100 * v, 2) for k, v in report["single"].items() if k != "us_per_event"},
        "signal_lost_by_plane": report["signal_lost_by_plane"], "bkg_vetoed_by_plane": report["bkg_vetoed_by_plane"],
        "planes": [{"name": p["name"], "tech": p["technology"], "reach": p["reach_m"], "y": [p["y_lo"], p["y_hi"]],
                    "muons_per_s": f"{p['muons_per_spill']:.2g}", "peak_channel_MHz": round(p["peak_channel_hz"] / 1e6, 2),
                    "over_limit": bool(p["channel_over_limit"] or p["area_over_density_limit_m2"] > 0),
                    "cost_kCHF": round(p["cost_kchf"]) if p["cost_kchf"] == p["cost_kchf"] else None}
                   for p in report["planes"]],
        "loader_notes": report.get("notes", []),
    }


# ------------------------------------------------------------------ LLM (OpenRouter)
class LLM:
    def __init__(self, model, env_file, temperature, max_tokens):
        if not os.environ.get("OPENROUTER_API_KEY") and Path(env_file).exists():
            for line in Path(env_file).read_text().splitlines():
                if line.strip().startswith("OPENROUTER_API_KEY"):
                    os.environ["OPENROUTER_API_KEY"] = line.split("=", 1)[1].strip().strip("'\"")
        self.key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self.key:
            raise SystemExit(f"no OPENROUTER_API_KEY (environment or {env_file})")
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.cost, self.calls = 0.0, 0
        self.lock = threading.Lock()

    def ask(self, system, user, retries=3):
        import requests
        body = {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_object"}, "usage": {"include": True}}
        for attempt in range(retries):
            try:
                r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=body, timeout=180,
                                  headers={"Authorization": f"Bearer {self.key}"})
                r.raise_for_status()
                js = r.json()
                with self.lock:
                    self.cost += float((js.get("usage") or {}).get("cost") or 0)
                    self.calls += 1
                text = js["choices"][0]["message"]["content"].strip()
                if text.startswith("```"):
                    text = text.strip("`").split("\n", 1)[1]
                return json.loads(text)
            except Exception as e:  # noqa: BLE001
                print(f"  [llm] attempt {attempt + 1} failed: {str(e)[:200]}")
        return None


# ------------------------------------------------------------------ evaluation on the cluster
def evaluate_remote(designs, proxy_port, host, workers, n_signal):
    cmd = ["ssh", "-J", f"127.0.0.1:{proxy_port}", host,
           f"cd {REMOTE_VETO} && OMP_NUM_THREADS=1 {EXEC} evolve/evaluate_designs.py --workers {workers} --n-signal {n_signal}"]
    proc = subprocess.run(cmd, input=json.dumps(designs), capture_output=True, text=True, timeout=3600)
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        err = (proc.stderr or "")[-800:]
        return [dict(name=d.get("name"), error=f"cluster evaluation failed: {err}") for d in designs]


def valid_locally(design):
    try:
        dsg.from_dict(design)
        return None
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


# ------------------------------------------------------------------ the loop
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="openai/gpt-4.1-mini")
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--children", type=int, default=8, help="LLM proposals per generation")
    ap.add_argument("--population", type=int, default=10)
    ap.add_argument("--seeds", nargs="*", default=SEEDS, help="designs/*.json to start from")
    ap.add_argument("--inspirations", type=int, default=3)
    ap.add_argument("--n-signal", type=int, default=40_000)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--env-file", default=str(Path.home() / "workspace/uzh/supply2/.env"))
    ap.add_argument("--proxy-port", default=os.environ.get("PROXY_PORT", "2226"))
    ap.add_argument("--host", default="farm-gpu3")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    llm = LLM(args.model, args.env_file, args.temperature, args.max_tokens)
    run = paths.results_dir() / "evolve" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(vars(args), indent=2))
    system = system_prompt()
    (run / "system_prompt.txt").write_text(system)
    history = run / "history.jsonl"
    print(f"run {run}\nmodel {args.model}, {args.generations} generations x {args.children} children\n")

    def record(cand):
        with history.open("a") as fh:
            fh.write(json.dumps(cand, default=float) + "\n")

    # generation 0: the seed designs
    seeds = [json.loads((dsg.DESIGN_DIR / f"{n}.json").read_text()) for n in args.seeds]
    t0 = time.time()
    reports = evaluate_remote(seeds, args.proxy_port, args.host, len(seeds), args.n_signal)
    population = []
    for d, r in zip(seeds, reports):
        sc, _ = objective.score(r)
        cand = dict(id=d["name"], gen=0, design=d, report=r, score=sc, rationale="seed design")
        record(cand)
        population.append(cand)
    population.sort(key=lambda c: -c["score"])
    print(f"gen 0  {len(seeds)} seeds in {time.time() - t0:.0f} s:  " +
          "  ".join(f"{c['id']}={c['score']:.2f}" for c in population))
    best = population[0]
    (run / "best.json").write_text(json.dumps(best["design"], indent=1))

    seen = {json.dumps(c["design"]["fins"], sort_keys=True) + json.dumps(c["design"]["ubt"], sort_keys=True)
            for c in population}
    for gen in range(1, args.generations + 1):
        t0 = time.time()
        weights = [1.0 / (1 + i) for i in range(len(population))]
        def propose(k):
            parent = random.Random(args.seed * 1000 + gen * 100 + k).choices(population, weights=weights, k=1)[0]
            others = [c for c in population if c is not parent][:args.inspirations]
            directive = list(DIRECTIVES)[k % len(DIRECTIVES)]
            user = json.dumps({
                "directive": DIRECTIVES[directive],
                "parent": {"design": parent["design"], "evaluation": compact(parent["report"])},
                "inspirations": [{"design": c["design"], "evaluation": compact(c["report"])} for c in others],
                "population_scores": [{"name": c["id"], "score": round(c["score"], 3)} for c in population],
                "request": f"Propose one improved complete design. Name it g{gen}c{k}_<short words>.",
            })
            return k, parent, directive, llm.ask(system, user)

        with ThreadPoolExecutor(max_workers=args.children) as pool:      # the LLM calls in parallel
            proposals = list(pool.map(propose, range(args.children)))
        children = []
        for k, parent, directive, out in proposals:
            if not out or "design" not in out or not isinstance(out["design"], dict):
                continue
            d = out["design"]
            d["name"] = str(d.get("name") or f"g{gen}c{k}")[:60]
            d.setdefault("detector", {})
            d.setdefault("description", out.get("rationale", "")[:200])
            key = json.dumps(d.get("fins"), sort_keys=True) + json.dumps(d.get("ubt"), sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            children.append(dict(id=d["name"], gen=gen, design=d, parent=parent["id"], directive=directive,
                                 rationale=out.get("rationale", "")))
        # designs the loader rejects are scored without going to the cluster
        to_run = []
        for c in children:
            err = valid_locally(c["design"])
            if err:
                c["report"], c["score"] = dict(name=c["id"], error=err), -math.inf
            else:
                to_run.append(c)
        if to_run:
            for c, r in zip(to_run, evaluate_remote([c["design"] for c in to_run], args.proxy_port, args.host,
                                                    len(to_run), args.n_signal)):
                c["report"] = r
                c["score"], _ = objective.score(r)
        for c in children:
            record(c)
        population = sorted(population + children, key=lambda c: -c["score"])[:args.population]
        if population[0]["score"] > best["score"]:
            best = population[0]
            (run / "best.json").write_text(json.dumps(best["design"], indent=1))
        ok = sum(1 for c in children if math.isfinite(c["score"]))
        print(f"gen {gen}  children {ok}/{len(children)} valid  best {best['id']}={best['score']:.3f}  "
              f"pop mean {np.mean([c['score'] for c in population if math.isfinite(c['score'])]):.3f}  "
              f"{time.time() - t0:.0f} s  llm ${llm.cost:.3f} ({llm.calls} calls)")
        for c in sorted(children, key=lambda c: -c["score"])[:3]:
            print(f"    {c['score']:8.3f}  {c['id']}: {c['rationale'][:140]}")

    print(f"\nbest: {best['id']} score {best['score']:.3f} -> {run / 'best.json'}")
    print(json.dumps(compact(best["report"]), indent=1)[:2000])


if __name__ == "__main__":
    main()
