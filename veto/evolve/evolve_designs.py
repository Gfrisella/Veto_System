#!/usr/bin/env python3
"""AlphaEvolve-style search over veto designs: an LLM proposes design JSONs, the cluster scores them.

Same recipe as supply2/srq_alphaevolve/evolve_heuristic.py, with designs instead of code:
  1. a population of designs (seeded with designs/*.json) and their evaluation reports;
  2. each generation, children are proposed by the LLM from a rank-weighted parent, the top of the
     population as inspiration and a directive (refine / explore / combine), with the full problem
     context: geometry and limits, the JSON schema, technologies, where the muons are, the objective;
  3. all children of a generation are evaluated in parallel on the cluster (evaluate_designs.py,
     sent over our SSH proxy: the LLM key never leaves this machine);
  4. truncation selection on the reward of objective.py (veto.objective on the run_reco result);
  5. at the end the best designs are re-evaluated with other random seeds (--confirm-top, --confirm-seeds):
     rewards differ by about their statistical error from seed to seed, best.json is the best mean.
Every candidate, its report and the LLM's rationale go to results/evolve/<run>/history.jsonl; the best
design so far is written to results/evolve/<run>/best.json (a normal designs/*.json file).

  bin/remote.sh sync                                  # once, so the cluster has the current code
  evolve/evolve_designs.py --generations 10 --children 8                 # Claude Code CLI, sonnet
  evolve/evolve_designs.py --model opus --generations 20
  evolve/evolve_designs.py --backend codex --effort high                  # OpenAI Codex CLI (ChatGPT login)
  evolve/evolve_designs.py --backend openrouter --model anthropic/claude-sonnet-4.5
  evolve/run_on_cluster.sh                            # everything on farm-gpu3 (LLM there too), in tmux

LLM backends: "claude" runs the Claude Code CLI headless (`claude -p`, the user's own login, no tools,
no session saved, run from an empty directory); "codex" runs `codex exec` the same way (ephemeral, user
config ignored, read-only sandbox, empty directory) on the user's ChatGPT login; "openrouter" reads OPENROUTER_API_KEY from the
environment, else from the .env file at --env-file (default: the supply2 project's .env).
"""

import argparse
import functools
import json
import math
import os
import random
import subprocess
import sys
import tempfile
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
SEEDS = ["D0_minimal", "D1_wide_ubt", "D3_heavy_right", "D6_tile2", "D6_fiber", "D6_straw", "E1_llm_run1", "E2_llm_run2", "E3_robust",
         "E4_ubt2", "E4_entry2"]
DIRECTIVES = {
    "refine": "Refine the parent: keep its structure, fix its clearest weakness from the report "
              "(a violation first: a plane over its rate limit or the budget; then a plane costing signal "
              "for little veto, a gap that lets muons through, or cost that buys nothing).",
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
each is simulated and reconstructed (muon-pair background candidates, muon and neutrino interactions, signal decays, single muons)
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

RATE LIMITS: a plane's peak rate per channel (or per cm^2) is set by its technology's pitch and the muon
density where it sits (table above); readout cell sizes do not change it. The report of every design lists
each plane's peak rate next to its limit.

OBJECTIVE (the reward; higher is better):
{objective.__doc__}

DESIGN JSON (all lengths in metres):
{{"name": "<short unique name>", "description": "<one line>",
  "ubt": {{"y_lo": -1.35, "y_hi": 1.35}},
  "fins": [{{"side": "right"|"left", "z": 30.6, "length": 3.0, "y_lo": -3.0, "y_hi": 3.0, "tilt": 0,
            "technology": "tile"}}, ...],
  "detector": {{}}}}
"detector": {{"cone_mrad": 0 .. 50}} sets the vertex cone of the veto (default 0): the tracks of a vertex also search
r0 + cone x (distance from the vertex to the plane) on every plane upstream, which catches the parents of muon
interactions (bent by a few mrad at the vertex) but picks up random hits, so it costs signal on busy planes:
on the small-fin designs ~0.9 % signal at 10 mrad, ~3.7 % at 20 mrad, much more with large tile fins.
"stations": 2 (or 3) in the "ubt" or a fin entry: identical layers behind each other read out as an OR, plane
inefficiency 1e-3 -> 1e-6, n x channels, cost and material; rates and random hits as one station.
A fin may also give "cell_u"/"cell_y" (readout cell size, m) to read a technology out more coarsely: this only
loses signal (bigger search windows), it changes neither rates, channels nor cost.
Each evaluation reports: reward +- error, feasibility and violations, the reward terms and the sensitivity, false vertices in 15 years,
signal efficiency, cost, how often the muons of the background pairs escape the veto per type, and per plane:
peak rates vs limits, cost, signal decays it loses (random hits) and background pairs it vetoes (a pair is
often vetoed by several planes, so these overlap).

Return JSON only: {{"design": <the complete design JSON>, "rationale": "<two sentences: what you changed and why>"}}"""


def compact(report):
    """The parts of an evaluation report the LLM needs, rounded."""
    if report.get("error"):
        return {"error": report["error"]}
    sc, r = objective.evaluate(report), report["result"]
    bg, sg = sc["confusion"]["background"], sc["confusion"]["signal"]
    rates = {p["plane"]: p for p in r["planes"]}
    lost, vetoed = r["signal"].get("FP_by_plane", {}), r["background"].get("veto_by_plane", {})
    planes = []
    for g in report["planes"]:
        p = rates[g["name"]]
        t = tech.TECHNOLOGIES.get(p["technology"]) if p["technology"] else None
        ch_limit = t.get("max_channel_rate") if t else None
        planes.append({
            "name": g["name"], "tech": g["technology"], "reach": g["reach_m"], "y": [g["y_lo"], g["y_hi"]],
            "muons_per_s": f"{p['rate']:.2g}",
            "peak_MHz_per_channel": round(p["peak_channel_hz"] / 1e6, 3),
            "limit_MHz_per_channel": ch_limit / 1e6 if ch_limit else None,
            "peak_Hz_per_cm2": f"{p['peak_hz_cm2']:.2g}", "limit_Hz_per_cm2": p["max_rate_hz_cm2"],
            "cost_kCHF": round(p["cost"]) if p["cost"] == p["cost"] else None,
            "signal_lost": lost.get(g["name"], 0), "pairs_vetoed": vetoed.get(g["name"], 0)})
    return {
        "reward": round(sc["reward"], 3), "reward_err": round(sc["reward_err"], 3), "feasible": sc["feasible"],
        "terms": {k: round(-v, 3) for k, (v, _) in sc["terms"].items()},
        "sensitivity": round(sc["sensitivity"], 4),
        "violations": sc["over_rate_limit"] + ([f"over the {objective.CONFIG.cost_budget_kchf / 1000:g} MCHF budget"] if sc["over_budget"] else []),
        "false_vertices_15y": f"{sc['bkg_events_15y']:.2g} +- {sc['bkg_events_15y_err']:.1g}",
        "false_vertices_15y_by_source": {k: f"{v:.2g} +- {e:.1g}" for k, (v, e) in sc["bkg_by_source"].items()},
        "combinatorial_factorised_vs_direct": f"{bg['FN_factorised'] * bg['events_15y_before']:.2g} / {bg['FN_direct'] * bg['events_15y_before']:.2g}",
        "cone_mrad": r.get("cone_mrad", 0.0),
        "surviving_pairs_counted": f"{bg['n_survive_direct']} of {bg['n_candidates']}",
        "signal_efficiency_pct": f"{100 * sg['efficiency']:.2f} +- {100 * sg['efficiency_err']:.2f}",
        "signal_decays_simulated": sg["n"],
        "cost_MCHF": round(sc["cost_kchf"] / 1000, 3), "channels": r["n_channels"],
        "pair_muons_unvetoed": {k: f"{e:.2g} +- {de:.1g}" for k, (e, de) in sc["confusion"]["single_inefficiency"].items()},
        "planes": planes,
        "loader_notes": report.get("notes", []),
    }


def parse_json(text):
    """The JSON object in an LLM answer (code fences and text around it ignored)."""
    text = text.strip()
    return json.loads(text[text.index("{"):text.rindex("}") + 1])


# ------------------------------------------------------------------ LLM backends
_LIMIT_LOCK = threading.Lock()


def wait_for_limit(msg):
    """Sleep until the usage limit resets ("... resets 7:40pm (Europe/Zurich)"), else 15 min; one thread
    reports it, the others wait on the same lock."""
    import re
    with _LIMIT_LOCK:
        wait = 900
        m = re.search(r"(?:resets|try again at)\s+(\d{1,2})(?::(\d\d))?\s*([ap]m)", msg, re.I)
        if m:
            h = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "pm" else 0)
            now = datetime.now()
            reset = now.replace(hour=h, minute=int(m.group(2) or 0), second=0, microsecond=0)
            ahead = (reset - now).total_seconds()
            wait = ahead + 120 if 0 <= ahead < 6 * 3600 else (120 if -3600 < ahead < 0 else 900)
        print(f"  [llm] usage limit ({msg.strip()[:80]}): waiting {wait / 60:.0f} min")
        time.sleep(wait)


class ClaudeCLI:
    """Claude Code headless (`claude -p`) on the user's own login: no tools, no settings, no session saved,
    run from an empty directory so no project context is picked up."""

    def __init__(self, model, effort=None):
        self.model, self.effort = model, effort
        self.cost, self.calls = 0.0, 0
        self.lock = threading.Lock()
        self.cwd = tempfile.mkdtemp(prefix="veto_evolve_llm_")

    def ask(self, system, user, retries=3):
        cmd = ["claude", "-p", "--model", self.model, "--system-prompt", system, "--tools", "",
               "--no-session-persistence", "--strict-mcp-config", "--setting-sources", "", "--output-format", "json"]
        if self.effort:
            cmd += ["--effort", self.effort]
        attempt = 0
        while attempt < retries:
            try:
                proc = subprocess.run(cmd, input=user, capture_output=True, text=True, timeout=900, cwd=self.cwd)
                js = json.loads(proc.stdout)
                with self.lock:
                    self.cost += float(js.get("total_cost_usd") or 0)
                    self.calls += 1
                if js.get("is_error"):
                    msg = str(js.get("result"))
                    if "limit" in msg.lower():           # usage limit of the subscription: wait, not an attempt
                        wait_for_limit(msg)
                        continue
                    raise RuntimeError(msg[:300])
                return parse_json(js["result"])
            except Exception as e:  # noqa: BLE001
                attempt += 1
                print(f"  [llm] attempt {attempt} failed: {str(e)[:200]} {proc.stderr[-200:] if 'proc' in locals() else ''}")
        return None


class CodexCLI:
    """OpenAI Codex CLI headless (`codex exec`) on the user's own ChatGPT login: ephemeral (no session saved),
    the user's config.toml ignored (no MCP servers, plugins or rules), read-only sandbox in an empty directory.
    Codex has no system-prompt flag, so the system prompt leads the prompt; the reply is its last message."""

    def __init__(self, model=None, effort=None):
        if model is None:                              # the model of the user's Codex config (which is otherwise ignored)
            cfg = Path.home() / ".codex" / "config.toml"
            line = next((l for l in cfg.read_text().splitlines() if l.strip().startswith("model =")), None) if cfg.exists() else None
            model = line.split("=", 1)[1].strip().strip('"') if line else None
        self.model, self.effort = model, effort
        self.cost, self.calls = 0.0, 0                 # subscription: no per-call cost reported
        self.lock = threading.Lock()
        self.cwd = tempfile.mkdtemp(prefix="veto_evolve_codex_")

    def ask(self, system, user, retries=3):
        attempt = 0
        while attempt < retries:
            out = Path(tempfile.mktemp(dir=self.cwd, suffix=".txt"))
            cmd = ["codex", "exec", "--ephemeral", "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check",
                   "-s", "read-only", "-C", self.cwd, "--color", "never", "-o", str(out)]
            if self.model:
                cmd += ["-m", self.model]
            if self.effort:
                cmd += ["-c", f'model_reasoning_effort="{self.effort}"']
            prompt = (f"{system}\n\n---\nAnswer directly in this message: do not run commands or read files, "
                      f"nothing in this directory is relevant.\n---\n\n{user}")
            try:
                proc = subprocess.run(cmd + ["-"], input=prompt, capture_output=True, text=True, timeout=1200)
                text = out.read_text() if out.exists() else ""
                out.unlink(missing_ok=True)
                if proc.returncode != 0 or not text.strip():
                    msg = (proc.stderr or proc.stdout)[-600:]
                    if "usage limit" in msg.lower() or "rate limit" in msg.lower():
                        wait_for_limit(msg)
                        continue
                    raise RuntimeError(f"exit {proc.returncode}: {msg[-300:]}")
                with self.lock:
                    self.calls += 1
                return parse_json(text)
            except Exception as e:  # noqa: BLE001
                attempt += 1
                print(f"  [llm] attempt {attempt} failed: {str(e)[:300]}")
        return None


class OpenRouter:
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
                return parse_json(js["choices"][0]["message"]["content"])
            except Exception as e:  # noqa: BLE001
                print(f"  [llm] attempt {attempt + 1} failed: {str(e)[:200]}")
        return None


# ------------------------------------------------------------------ evaluation on the cluster
EVAL_DATA = dict(inputs="inputs.npz", prep_tag="")   # which reco inputs / interaction caches (--inputs, --prep-tag)


def evaluate_remote(designs, proxy_port, host, workers, n_signal, seed=1):
    """Evaluate on `host` over our SSH proxy, or here (host "local": when this runs on the cluster)."""
    run = (f"OMP_NUM_THREADS=1 {EXEC} evolve/evaluate_designs.py --workers {workers} "
           f"--n-signal {n_signal} --seed {seed} --inputs {EVAL_DATA['inputs']} --prep-tag '{EVAL_DATA['prep_tag']}'")
    if host == "local":
        cmd = ["bash", "-c", f"cd {VETO} && {run}"]
    else:
        cmd = ["ssh", "-J", f"127.0.0.1:{proxy_port}", host, f"cd {REMOTE_VETO} && {run}"]
    proc = subprocess.run(cmd, input=json.dumps(designs), capture_output=True, text=True, timeout=3600)
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        err = (proc.stderr or "")[-800:]
        return [dict(name=d.get("name"), error=f"cluster evaluation failed: {err}") for d in designs]


def reward_err(report):
    s = objective.evaluate(report)
    return s["reward_err"] if s else 0.0


def valid_locally(design):
    try:
        dsg.from_dict(design)
        return None
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


# ------------------------------------------------------------------ the loop
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["claude", "codex", "openrouter"], default="claude")
    ap.add_argument("--model", default=None, help="claude: sonnet (default) / opus / a model id; codex: an OpenAI model "
                                                 "(default: Codex's own); openrouter: its model id")
    ap.add_argument("--effort", default=None, help="claude / codex backends: reasoning effort (claude low .. max; codex low / medium / high)")
    ap.add_argument("--confirm-top", type=int, default=5, help="re-evaluate this many best designs at the end")
    ap.add_argument("--confirm-seeds", type=int, nargs="*", default=[2, 3])
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--children", type=int, default=8, help="LLM proposals per generation")
    ap.add_argument("--population", type=int, default=10)
    ap.add_argument("--seeds", nargs="*", default=SEEDS, help="designs/*.json to start from")
    ap.add_argument("--seed-runs", nargs="*", default=[], help="earlier runs (results/evolve/<run>) whose best designs join gen 0")
    ap.add_argument("--seed-top", type=int, default=6, help="designs taken from each of --seed-runs")
    ap.add_argument("--bkg-estimate", choices=["max", "factorised", "direct"], default="max",
                    help="false vertices: max of the direct count and the factorised estimate (default), or one of them")
    ap.add_argument("--inspirations", type=int, default=3)
    ap.add_argument("--n-signal", type=int, default=40_000)
    ap.add_argument("--temperature", type=float, default=0.9, help="openrouter only")
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--env-file", default=str(Path.home() / "workspace/uzh/supply2/.env"))
    ap.add_argument("--proxy-port", default=os.environ.get("PROXY_PORT", "2226"))
    ap.add_argument("--host", default="farm-gpu3", help='cluster node, or "local" to evaluate on this machine')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inputs", default="inputs.npz", help="results/reco file (inputs_delta.npz: the enriched dataset); "
                                                         "needed here too, for the pair statistics of the objective")
    ap.add_argument("--prep-tag", default="", help="interaction caches prepared_<kind><tag>.npz (_delta: enriched)")
    args = ap.parse_args()
    EVAL_DATA.update(inputs=args.inputs, prep_tag=args.prep_tag)
    objective.INPUTS = args.inputs
    random.seed(args.seed)
    objective.configure(bkg_estimate=args.bkg_estimate)

    if args.backend == "claude":
        llm = ClaudeCLI(args.model or "sonnet", args.effort)
    elif args.backend == "codex":
        llm = CodexCLI(args.model, args.effort)
    else:
        llm = OpenRouter(args.model, args.env_file, args.temperature, args.max_tokens)
    run = paths.results_dir() / "evolve" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(vars(args), indent=2))
    system = system_prompt()
    (run / "system_prompt.txt").write_text(system)
    history = run / "history.jsonl"
    print(f"run {run}\n{args.backend} {args.model}, {args.generations} generations x {args.children} children\n")

    def record(cand):
        with history.open("a") as fh:
            fh.write(json.dumps(cand, default=float) + "\n")

    # generation 0: the seed designs
    seeds = [json.loads((dsg.DESIGN_DIR / f"{n}.json").read_text()) for n in args.seeds]
    for prev in args.seed_runs:              # best designs of earlier runs, re-scored with today's objective
        prev = Path(prev) if Path(prev).is_absolute() else paths.results_dir() / "evolve" / prev
        hist = [json.loads(line) for line in (prev / "history.jsonl").open()]
        ok = sorted((c for c in hist if c["score"] is not None and math.isfinite(c["score"])), key=lambda c: -c["score"])
        seeds += [c["design"] for c in ok if c["design"]["name"] not in {d["name"] for d in seeds}][:args.seed_top]
    t0 = time.time()
    reports = evaluate_remote(seeds, args.proxy_port, args.host, len(seeds), args.n_signal)
    population = []
    for d, r in zip(seeds, reports):
        sc, _ = objective.score(r)
        cand = dict(id=d["name"], gen=0, design=d, report=r, score=sc, score_err=reward_err(r), rationale="seed design")
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
                "population_rewards": [{"name": c["id"], "reward": round(c["score"], 3), "err": round(c.get("score_err", 0), 3)} for c in population],
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
                c["score_err"] = reward_err(r)
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
            print(f"    {c['score']:9.3f} +- {c.get('score_err', 0):.3f}  {c['id']}: {c['rationale'][:140]}")

    # the best designs again with other seeds: the ranking at the top is within the statistical errors
    hist = [json.loads(line) for line in history.open()]
    top = sorted((c for c in hist if math.isfinite(c["score"])), key=lambda c: -c["score"])[:args.confirm_top]
    if top and args.confirm_seeds:
        scores = {c["id"]: [c["score"]] for c in top}
        for seed in args.confirm_seeds:
            for c, r in zip(top, evaluate_remote([c["design"] for c in top], args.proxy_port, args.host,
                                                 len(top), args.n_signal, seed)):
                scores[c["id"]].append(objective.score(r)[0])
        rows = sorted(({"id": c["id"], "scores": scores[c["id"]], "mean": float(np.mean(scores[c["id"]])),
                        "spread": float(np.std(scores[c["id"]])), "design": c["design"]} for c in top),
                      key=lambda x: -x["mean"])
        (run / "confirm.json").write_text(json.dumps(rows, indent=1, default=float))
        print(f"\nre-evaluated with seeds {args.confirm_seeds}:")
        for x in rows:
            print(f"  {x['mean']:9.3f} (spread {x['spread']:.3f})  {x['id']}   {np.round(x['scores'], 3).tolist()}")
        best = next(c for c in top if c["id"] == rows[0]["id"])
        (run / "best.json").write_text(json.dumps(best["design"], indent=1))

    print(f"\nbest: {best['id']} score {best['score']:.3f} -> {run / 'best.json'}   llm ${llm.cost:.2f} ({llm.calls} calls)")
    print(json.dumps(compact(best["report"]), indent=1)[:3000])


if __name__ == "__main__":
    main()
