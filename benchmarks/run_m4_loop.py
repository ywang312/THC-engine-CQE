"""The M4 step rules (U5) and the closed-loop finite-shot fault-tolerant CQE (U6, U7).

U5 (exact emulation, ``ft.steprules.run_step_rule_system``): from HF with the classical first step (F16), the rules
line_search (the exact line search along the measured direction), candidates (M3's grid of 11 steps), unit (R1,
with the F29 caps) and hybrid (R3, gated on planned costs at each r, F31) on the spin-orbital and spin-free
metrics, 40 steps (30 at N = 8); k* (1.6 mHa), the first k with p >= 0.9 (k_p90) and k_ref = k*, or k_p90 where
k* is not reached; F30 costs at r = 0.3, 1 and 3 for k <= k* + 2 (k <= 20 where k* is not reached, k <= 12 at
N = 8), with C_loop(k_ref); the energy precisions and the energy-to-residual cost ratios at k = 1 and k_ref - 1 (F27,
F28). Systems: h4_r{1.8, 2.5, 3.0, 3.5, 4.0}, h6_r{1.8, 2.5, 3.0, 3.5, 4.0}, h8_r{1.8, 3.0} and N2 CAS(6,6) at
1.098-2.8 A. One calculation (``run_step_rule_system`` call) per system, metric and rule, R3 once per planning r,
so that none exceeds a few minutes at N = 8; each run records its calculation's seconds ("runtime_s").

U6/U7 (``ft.closedloop.run_closed_loop_system``; R1 spin-orbital, route direct_mg, exact propagation unless
noted; 5 iterations; replicate seeds of F38):

    block         systems                               configurations                                    replicates
    headline      n2_r2.0, n2_r2.4, n2_r2.8, h4_r4.0     oracle r = 1, 2, 3; lagged r = 0.5, 1, 2           100
    controls      h6_r3.0, h4_r1.8, n2_r1.098            oracle r = 1, 3; lagged r = 1                      20
    routes        n2_r2.4, h4_r1.8                       ae_x, fixed and random offsets, oracle r = 1, 3    20
    step_rules    n2_r2.0, n2_r2.4                       line_search, candidates, hybrid; oracle r = 1, 3   20
    compiled      n2_r2.0, n2_r2.4, n2_r2.8, h4_r4.0     compiled (F25 slices): oracle r = 1, 3; lagged 1   20
    metric        n2_r2.0                                spin-free metric, oracle r = 1, 3                  20
    n8_controls   n2v_r2.4, n2v_r2.8, h8_r3.0            lagged r = 1; oracle r = 3                         10

The lagged r = 1 headline cell decides whether a measured iteration pays (F38: the 95% bootstrap interval of the
mean paired difference T(k) - T(k + 1) above 0); if the emulation-only angle cap (F29) binds in any of its
replicates, the cell is rerun with the cap off (flag "angle_cap_rerun") and the rerun decides. The oracle r-scan
is the optimistic bound and the lagged r = 0.5 and 2 cells are sensitivity; neither decides. Each control system
(the negative and the N = 8 controls) has its own deciding cell, the lagged r = 1 cell of its block (R1
spin-orbital, direct_mg, exact propagation), with the same rerun rule: it gives the control's verdict.
``--quick``: U5 on h4_r1.8, h4_r4.0, n2_r2.0, n2_r2.4 and the headline cells on n2_r2.4 and h4_r4.0 at 20
replicates.

Outputs (``--out``, default results/m4; the input cache under <out>/cache, shared with
benchmarks/run_m4_scaling.py): loop_<system>.json (U5 block), closedloop_<system>.json ("system" {c_qpe, e0, e1,
e1_any, lam, n_iter, p_hf, ...} and "configs": per configuration "config" (the LoopSpec fields, rule, metric),
"block", "label", "n", "deciding", "replicates" (seed, k, p, e_err, c_prep, c_iter, c_loop, r_realized, flags,
compiled, contaminant_below_theta (null: the hand-off layer fills it), ...) and "summary" (F38 statistics under
repetition)) and loop_summary.md (U5, U6, U7). Each loop_<system>.json is written as soon as its U5 block is done
and each closedloop_<system>.json as soon as its closed-loop block is done, so a later failure keeps the finished
systems; loop_summary.md is written at the end. Strict JSON: nan as null, +-inf as "inf"/"-inf". Every
loop_<system>.json and closedloop_<system>.json carries "provenance": the digest of the package source
(``ft.scaling.source_digest``) as "package", the SHA-256 of this script's text as "script" and, for the closed loop,
the "settings" that fix its blocks ({"quick", "replicates"}); the downstream benchmarks ignore it.

``--resume`` (for long runs that may stop part-way): a system whose loop_<system>.json (U5) or
closedloop_<system>.json (closed loop) already exists in --out with the provenance of this run is loaded instead of
recomputed; a file with another provenance (or none) is recomputed and overwritten, with the reason printed; the
other systems are computed as without it. loop_summary.md is written from all selected systems; a loaded system's
wall-time row carries its recorded seconds and is marked "(loaded)".

Threads: OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and MKL_NUM_THREADS are set to 1 before numpy is imported.

Usage:
    python benchmarks/run_m4_loop.py --quick
    python benchmarks/run_m4_loop.py --systems n2_r2.4 --replicates 20
    python benchmarks/run_m4_loop.py --resume
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import pathlib  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

SCRIPT_DIGEST = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()   # "script" of the provenance

U5_SYSTEMS = tuple(f"h4_r{r}" for r in ("1.8", "2.5", "3.0", "3.5", "4.0")) + \
    tuple(f"h6_r{r}" for r in ("1.8", "2.5", "3.0", "3.5", "4.0")) + ("h8_r1.8", "h8_r3.0") + \
    tuple(f"n2_r{r}" for r in ("1.098", "1.6", "2.0", "2.4", "2.8"))
U5_QUICK = ("h4_r1.8", "h4_r4.0", "n2_r2.0", "n2_r2.4")
HEADLINE = ("n2_r2.0", "n2_r2.4", "n2_r2.8", "h4_r4.0")
HEADLINE_QUICK = ("n2_r2.4", "h4_r4.0")
DECIDING = ("lagged", 1.0)                  # the pre-declared deciding cell (controller, r)
DECIDING_BLOCKS = ("headline", "controls", "n8_controls")   # blocks with a deciding cell (the latter two: controls)
MAX_ITER_LOOP = 5
K_TABLE = 4                                 # hand-off iterates shown in U6 and U7
R_VALUES = (0.3, 1.0, 3.0)
# Reference numbers: [design] R1 spin-orbital k* (1.6 mHa, F29 caps); [design] the first closed-loop step (100
# replicates, independent implementation): mean T_E(2) +- standard error and the median p_2, oracle r = 1 and 3.
REF_K_STAR_R1_SO = {"h4_r1.8": 3, "h4_r3.5": 7, "h6_r3.0": 11, "n2_r1.098": 3}
REF_T2 = {("n2_r2.4", 1.0): (3.958, 0.030, 0.284), ("n2_r2.4", 3.0): (3.643, 0.047, 0.279),
          ("n2_r2.0", 1.0): (2.638, 0.010, 0.471), ("n2_r2.0", 3.0): (2.259, 0.018, 0.453),
          ("n2_r2.8", 1.0): (4.691, 0.046, 0.235), ("n2_r2.8", 3.0): (4.327, 0.068, 0.235)}
# [design] intervals of the headline regression at 100 replicates: (system, controller, r) -> (lo, hi) of mean T_E(2).
REF_T2_INTERVALS = {("n2_r2.4", "oracle", 1.0): (3.85, 4.06), ("n2_r2.4", "oracle", 3.0): (3.50, 3.78),
                    ("n2_r2.4", "lagged", 1.0): (3.45, 3.80), ("n2_r2.0", "oracle", 3.0): (2.20, 2.32)}


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def blocks(quick: bool) -> dict:
    """The closed-loop blocks: name -> (systems, [(LoopSpec, rule, metric)], replicates)."""
    from thc_engine_cqe.ft.closedloop import LoopSpec

    def unit(spec, metric="spin_orbital"):
        return (spec, "unit", metric)

    headline = [unit(LoopSpec(controller="oracle", r=r)) for r in (1.0, 2.0, 3.0)] + \
        [unit(LoopSpec(controller="lagged", r=r)) for r in (0.5, 1.0, 2.0)]
    if quick:
        return {"headline": (HEADLINE_QUICK, headline, 20)}
    return {
        "headline": (HEADLINE, headline, 100),
        "controls": (("h6_r3.0", "h4_r1.8", "n2_r1.098"),
                     [unit(LoopSpec(controller="oracle", r=r)) for r in (1.0, 3.0)]
                     + [unit(LoopSpec(controller="lagged", r=1.0))], 20),
        "routes": (("n2_r2.4", "h4_r1.8"),
                   [unit(LoopSpec(route="ae_x", ae_offsets=o, controller="oracle", r=r))
                    for o in ("fixed", "random") for r in (1.0, 3.0)], 20),
        "step_rules": (("n2_r2.0", "n2_r2.4"),
                       [(LoopSpec(controller="oracle", r=r), rule, "spin_orbital")
                        for rule in ("line_search", "candidates", "hybrid") for r in (1.0, 3.0)], 20),
        "compiled": (HEADLINE, [unit(LoopSpec(propagation="compiled", controller="oracle", r=r)) for r in (1.0, 3.0)]
                     + [unit(LoopSpec(propagation="compiled", controller="lagged", r=1.0))], 20),
        "metric": (("n2_r2.0",), [unit(LoopSpec(controller="oracle", r=r), "spin_free") for r in (1.0, 3.0)], 20),
        "n8_controls": (("n2v_r2.4", "n2v_r2.8", "h8_r3.0"),
                        [unit(LoopSpec(controller="lagged", r=1.0)), unit(LoopSpec(controller="oracle", r=3.0))], 10),
    }


def is_deciding(block: str, spec, rule, metric: str) -> bool:
    """The pre-declared deciding cell of a block (module docstring): lagged r = 1, R1 (unit), spin-orbital, direct_mg,
    exact propagation, in the headline block or a control block."""
    name = rule if isinstance(rule, str) else rule.name
    return (block in DECIDING_BLOCKS and (spec.controller, spec.r) == DECIDING and name == "unit"
            and metric == "spin_orbital" and spec.route == "direct_mg" and spec.propagation == "exact")


def label(config: dict) -> str:
    """A short label of a configuration: [route offsets] [propagation] [rule] [metric] controller r."""
    parts = []
    if config["route"] != "direct_mg":
        parts.append(config["route"] + (f" {config['ae_offsets']}" if config["route"] == "ae_x" else ""))
    if config["propagation"] != "exact":
        parts.append(config["propagation"])
    if config["rule"] != "unit":
        parts.append(config["rule"])
    if config["metric"] != "spin_orbital":
        parts.append(config["metric"])
    if config.get("step_rule") and config["step_rule"].get("angle_cap") is None and config["rule"] == "unit":
        parts.append("angle cap off")
    parts.append(f"{config['controller']} r = {config['r']:g}")
    return " ".join(parts)


def _rule_label(run: dict) -> str:
    """The rule of a U5 run, with R3's planning r."""
    return run["rule"]["name"] + ("" if run["r_plan"] is None else f" (r = {run['r_plan']:g})")


def provenance(kind: str, *, quick: bool = False, replicates: int | None = None) -> dict:
    """The provenance of a loop_<s>.json (``kind`` "u5") or closedloop_<s>.json ("closed_loop"): the package source
    digest ("package"), this script's digest ("script") and, for the closed loop, the settings that fix a system's
    blocks and replicates ("settings": {"quick", "replicates"}, the --replicates override or None)."""
    from thc_engine_cqe.ft import scaling

    if kind not in ("u5", "closed_loop"):
        raise ValueError(f"unknown kind {kind!r}")
    out = {"package": scaling.source_digest(), "script": SCRIPT_DIGEST}
    if kind == "closed_loop":
        out["settings"] = {"quick": bool(quick), "replicates": None if replicates is None else int(replicates)}
    return out


def load_resumable(out_dir, filename: str, want: dict) -> dict | None:
    """The JSON block of <out_dir>/<filename> if the file exists and its "provenance" equals ``want`` (--resume);
    None otherwise, printing why an existing file is recomputed (unreadable, no provenance, or the parts that
    differ)."""
    if out_dir is None:
        return None
    path = os.path.join(out_dir, filename)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as fh:
            block = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"[resume] {filename}: unreadable ({exc}); recomputing", flush=True)
        return None
    have = block.get("provenance") if isinstance(block, dict) else None
    if have == want:
        print(f"[resume] {filename}: same provenance; loaded (recorded {_g(block.get('runtime_s'), '.1f')} s)",
              flush=True)
        return block
    if not isinstance(have, dict):
        why = "no provenance"
    else:
        why = "different " + ", ".join(k for k in sorted(set(want) | set(have)) if have.get(k) != want.get(k))
    print(f"[resume] {filename}: {why}; recomputing", flush=True)
    return None


def _write_json(out_dir, filename: str, block: dict) -> None:
    """Write one strict-JSON block (already ``scaling.to_json``) to <out_dir>/<filename>; no-op without out_dir."""
    if out_dir is None:
        return
    with open(os.path.join(out_dir, filename), "w") as fh:
        json.dump(block, fh, allow_nan=False)


def run_u5(names, cache, params, out_dir=None, resume: bool = False, loaded: set | None = None) -> dict:
    """U5 per system: the inputs once ("inputs_runtime_s"), then one calculation (``run_step_rule_system`` call) per
    metric and rule, R3 once per r, each reading the inputs from the cache; every run records its calculation's
    seconds ("runtime_s"); loop_<s>.json, with its provenance, is written as soon as the system is done. With
    ``resume`` a loop_<s>.json of the same provenance in ``out_dir`` is loaded instead (its name added to
    ``loaded``)."""
    from thc_engine_cqe.ft import scaling
    from thc_engine_cqe.ft.closedloop import system_inputs
    from thc_engine_cqe.ft.steprules import run_step_rule_system
    from thc_engine_cqe.ft.symmetry import benchmark_system
    from thc_engine_cqe.steprules import StepRule

    rules = [StepRule("line_search"), StepRule("candidates"), StepRule("unit"), StepRule("hybrid")]
    want = provenance("u5")
    out = {}
    for name in names:
        if resume:
            block = load_resumable(out_dir, f"loop_{name}.json", want)
            if block is not None:
                out[name] = block
                if loaded is not None:
                    loaded.add(name)
                continue
        t = time.perf_counter()
        max_iter = 30 if benchmark_system(name).norb >= 8 else 40
        reader = cache
        if cache is not None:
            system_inputs(name, cache=cache, params=params)           # computed (refreshed with --no-cache) once
            if cache.refresh:                                         # the calculations read the fresh entries
                reader = scaling.Cache(cache.root)
        inputs_s = time.perf_counter() - t
        block = None
        for metric in ("spin_orbital", "spin_free"):
            for rule in rules:
                for r_plan in (R_VALUES if rule.name == "hybrid" else (None,)):
                    t_calc = time.perf_counter()
                    part = run_step_rule_system(name, [rule], metrics=(metric,),
                                                r_values=R_VALUES if r_plan is None else (r_plan,),
                                                max_iter=max_iter, cache=reader, params=params)
                    seconds = time.perf_counter() - t_calc
                    (run,) = part["runs"]
                    run["runtime_s"] = seconds
                    if block is None:
                        block = part
                    else:
                        block["runs"].append(run)
                    print(f"[U5 {name}] {metric} {_rule_label(run)}: {seconds:.1f} s, k* {run['k_star']}; "
                          f"peak RSS {peak_rss_mb():.0f} MB", flush=True)
        if reader is not cache:
            cache.hits, cache.misses = cache.hits + reader.hits, cache.misses + reader.misses
        block["inputs_runtime_s"] = inputs_s
        block["runtime_s"] = time.perf_counter() - t
        block["provenance"] = want
        out[name] = scaling.to_json(block)
        _write_json(out_dir, f"loop_{name}.json", out[name])
        longest = max(block["runs"], key=lambda r: r["runtime_s"])
        print(f"[U5 {name}] {block['runtime_s']:.1f} s (inputs {inputs_s:.1f} s); longest calculation "
              f"{longest['metric']} {_rule_label(longest)} {longest['runtime_s']:.1f} s; "
              f"peak RSS {peak_rss_mb():.0f} MB", flush=True)
    return out


def run_closed_loop(selected, quick: bool, n_override, cache, params, out_dir=None, resume: bool = False,
                    loaded: set | None = None) -> dict:
    """The closed-loop blocks per system (U6, U7); closedloop_<s>.json, with its provenance, is written as soon as the
    system is done. With ``resume`` a closedloop_<s>.json of the same provenance (package, script, quick and
    replicates) in ``out_dir`` is loaded instead (its name added to ``loaded``)."""
    from thc_engine_cqe.ft import scaling
    from thc_engine_cqe.ft.closedloop import run_closed_loop_system
    from thc_engine_cqe.steprules import StepRule

    spec_blocks = blocks(quick)
    systems = []
    for block, (names, _, _) in spec_blocks.items():
        systems += [s for s in names if s not in systems and (selected is None or s in selected)]
    want = provenance("closed_loop", quick=quick, replicates=n_override)
    out = {}
    for name in systems:
        if resume:
            done = load_resumable(out_dir, f"closedloop_{name}.json", want)
            if done is not None:
                out[name] = done
                if loaded is not None:
                    loaded.add(name)
                continue
        t = time.perf_counter()
        merged = None
        for block, (names, configs, n) in spec_blocks.items():
            if name not in names:
                continue
            n_rep = n if n_override is None else n_override
            res = run_closed_loop_system(name, configs, n=n_rep, max_iter=MAX_ITER_LOOP, cache=cache, params=params)
            if merged is None:
                merged = {"system": res["system"], "configs": []}
            for cfg, (spec, rule, metric) in zip(res["configs"], configs):
                deciding = is_deciding(block, spec, rule, metric)
                cfg.update(block=block, n=n_rep, deciding=deciding, label=label(cfg["config"]))
                merged["configs"].append(cfg)
                if deciding and cfg["summary"]["flag_counts"].get("angle_cap", 0) > 0:
                    rerun = run_closed_loop_system(name, [(spec, StepRule("unit", angle_cap=None), metric)], n=n_rep,
                                                   max_iter=MAX_ITER_LOOP, cache=cache, params=params)["configs"][0]
                    rerun.update(block=block, n=n_rep, deciding=True, label=label(rerun["config"]),
                                 flags=["angle_cap_rerun"])
                    cfg["deciding"] = False
                    cfg["superseded_by_angle_cap_rerun"] = True
                    merged["configs"].append(rerun)
        merged["runtime_s"] = time.perf_counter() - t
        merged["provenance"] = want
        out[name] = scaling.to_json(merged)
        _write_json(out_dir, f"closedloop_{name}.json", out[name])
        n_cfg = len(merged["configs"])
        print(f"[closed loop {name}] {n_cfg} configurations, {merged['runtime_s']:.1f} s; "
              f"peak RSS {peak_rss_mb():.0f} MB", flush=True)
    return out


def _g(x, spec: str = ".3g") -> str:
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    x = float(x)
    if math.isnan(x):
        return "-"
    if math.isinf(x):
        return "inf"
    return format(x, spec)


def _ci(v, spec: str = ".3g") -> str:
    return "-" if v is None or v[0] is None else f"[{_g(v[0], spec)}, {_g(v[1], spec)}]"


def _num(x) -> float:
    """A strict-JSON number as a float: None -> nan, "inf" / "-inf" -> +-inf."""
    if x is None:
        return math.nan
    if isinstance(x, str):
        return {"inf": math.inf, "-inf": -math.inf}[x]
    return float(x)


def _precision(costs: dict, r: str, k: int | None) -> dict | None:
    """The F27/F28 entry of iterate k in the costs at r (None if absent)."""
    if k is None or r not in costs:
        return None
    return next((p for p in costs[r]["precision"] if p["k"] == k), None)


def _precision_cells(costs: dict, rs: list[str], k: int | None) -> list[str]:
    """sigma_A, sigma_P3 (mHa) at k, and rho_E of the Armijo pair and of the 3-point fit at k for every r of rs."""
    entries = [_precision(costs, r, k) for r in rs]
    first = next((p for p in entries if p is not None), None)
    if first is None:
        return ["-", "-", "-"]
    sig = f"{_g(1e3 * first['sigma_armijo'])}, {_g(1e3 * first['sigma_parab3'])}"
    rho = [" / ".join("-" if p is None else _g(p.get(f"rho_{key}")) for p in entries) for key in ("armijo", "parab3")]
    return [sig] + rho


def u5_table(u5: dict) -> list[str]:
    head = ["system", "metric", "rule", "k* (k_p90)", "extra", "C_iter(1..3), r = 0.3", "r = 1", "r = 3",
            "C_loop(k_ref), r = 0.3 / 1 / 3", "sigma_A, sigma_P3 (k = 1)", "rho_E Armijo (k = 1)",
            "rho_E parab3 (k = 1)", "k_ref - 1", "sigma_A, sigma_P3 (k_ref - 1)", "rho_E Armijo (k_ref - 1)",
            "rho_E parab3 (k_ref - 1)"]
    lines = ["## U5. Step rules on exact trajectories (classical first step; C_QPE units)", "",
             "k*: first iterate within 1.6 mHa (k_p90, the first k with p >= 0.9, in brackets); k_ref = k*, or k_p90 "
             "where k* is not reached (cells marked (k_p90)); extra: measurements per iteration beyond the residual at "
             "Psi_k (mean over the costed iterations); C_iter(1..3) and C_loop(k_ref) at r = 0.3 / 1 / 3 (R3 rows: at "
             "their planning r); sigma_A and sigma_P3 (mHa) and rho_E = T_E/T_res,direct of the Armijo pair and of the "
             "3-point fit at k = 1 and at k = k_ref - 1, rho_E at r = 0.3 / 1 / 3 (R3 rows: at their planning r).", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, block in u5.items():
        for run in block["runs"]:
            costs = run["costs"]
            rs = ["0.3", "1", "3"] if run["r_plan"] is None else [f"{run['r_plan']:g}"]
            k_ref = run.get("k_ref", run["k_star"])
            p90 = run.get("k_ref_kind") == "p90"
            extra = run["extra_measurements"][1:run["cost_horizon"] + 1]
            cells = []
            for r in ("0.3", "1", "3"):
                c = costs.get(r)
                cells.append("-" if c is None else " / ".join(_g(x) for x in c["c_iter_over_qpe"][1:4]))
            loop = " / ".join(_g(costs[r].get("c_loop_k_ref_over_qpe", costs[r]["c_loop_k_star_over_qpe"]))
                              if r in costs else "-" for r in ("0.3", "1", "3")) + (" (k_p90)" if p90 else "")
            k_last = k_ref - 1 if k_ref is not None and k_ref >= 2 else None
            at_last = _precision_cells(costs, rs, k_last)
            row = [name, run["metric"], _rule_label(run), f"{_g(run['k_star'])} ({_g(run['k_p90'])})",
                   _g(sum(extra) / len(extra) if extra else 0.0)] + cells + [loop] + _precision_cells(costs, rs, 1) + \
                ["-" if k_last is None else f"{k_last}{' (k_p90)' if p90 else ''}"] + at_last
            lines.append("| " + " | ".join(row) + " |")
    return lines


def _cell_label(cfg: dict) -> str:
    """The configuration's label, marked "(decides)" for the deciding cell and "(superseded)" for a deciding cell
    that the angle-cap rerun replaced."""
    mark = " (decides)" if cfg.get("deciding") else " (superseded)" if cfg.get("superseded_by_angle_cap_rerun") else ""
    return cfg["label"] + mark


def _cell_row(name: str, cfg: dict, extra: bool) -> str:
    s = cfg["summary"]
    per = s["per_k"][:K_TABLE + 1]
    t = "; ".join(f"{_g(p['T_mean'], '.4g')} {_ci(p['T_ci'], '.4g')}" for p in per)
    k1 = per[1] if len(per) > 1 else None
    k2 = per[2] if len(per) > 2 else None
    diff = "-" if k1 is None or "diff_mean" not in k1 else \
        f"{_g(k1['diff_mean'], '.3g')} {_ci(k1['diff_ci'], '.3g')}{' pays' if k1['pays'] else ''}"
    beats = "-" if k2 is None else f"{_g(k2.get('beats_psi1'), '.2f')}"
    p2 = "-" if k2 is None else f"{_g(k2['p_median'], '.3f')} {_ci(k2['p_band'], '.3f')}"
    r12 = " / ".join("-" if p is None or "r_median" not in p else _g(p["r_median"], ".2f") for p in (k1, k2))
    st = s.get("stopping", {})
    regret = " / ".join(_g(st[key]["regret"], ".3f") if key in st else "-"
                        for key in ("fixed_2", "cost_gate_0.3", "cost_gate_1", "cost_gate_3"))
    row = [name, _cell_label(cfg), t, diff, beats, p2, r12, _g(s["catastrophic_share"], ".2f"), regret, str(s["n"])]
    if extra:
        c_iter = " / ".join(_g(p.get("c_iter_mean"), ".3g") for p in per[1:3])
        errs = [max(rep["compile_errors"]) for rep in cfg["replicates"] if rep["compile_errors"]]
        flags = [", ".join(f"{k} {v}" for k, v in sorted(s["flag_counts"].items()))]
        if s.get("non_realizable"):
            flags.append("non-realizable: " + ", ".join(s["non_realizable"]))
        row += [c_iter, _g(max(errs), ".3g") if errs else "-", "; ".join(f for f in flags if f) or "-"]
    return "| " + " | ".join(row) + " |"


def closed_loop_tables(cl: dict) -> list[str]:
    head = ["system", "configuration", "mean T_rep(k) [95% CI], k = 0..4", "T(1) - T(2) [CI]", "P(T(2) < T(1))",
            "median p~_2 [16-84%]", "r_1 / r_2 (median)", "catastrophic", "regret K = 2 / gates 0.3, 1, 3", "n"]
    lines = ["## U6. Closed loop, headline cells (C_QPE units; repetition, M3's rule E)", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, block in cl.items():
        lines += [_cell_row(name, cfg, False) for cfg in block["configs"] if cfg["block"] == "headline"]
    rows = [(name, cfg) for name, block in cl.items() for cfg in block["configs"] if cfg["block"] != "headline"]
    if rows:
        head2 = head + ["mean C_iter(1) / C_iter(2)", "max compile error", "flags (replicates); non-realizable parts"]
        lines += ["", "## U7. Closed-loop ablations", "", "| " + " | ".join(["block"] + head2) + " |",
                  "|" + "---|" * (len(head2) + 1)]
        lines += ["| " + cfg["block"] + " " + _cell_row(name, cfg, True) for name, cfg in rows]
    return lines


def reference_checks(u5: dict, cl: dict) -> list[str]:
    """The reference numbers of the systems that were run, as markdown rows."""
    rows = []
    for name, k_ref in REF_K_STAR_R1_SO.items():
        run = next((r for r in u5.get(name, {}).get("runs", []) if r["metric"] == "spin_orbital"
                    and r["rule"]["name"] == "unit"), None)
        if run is not None:
            rows.append(f"| {name} R1 spin-orbital k* [design] | {k_ref} | {_g(run['k_star'])} | "
                        f"{'yes' if run['k_star'] == k_ref else 'no'} |")
    for name, block in cl.items():
        for cfg in block["configs"]:
            c, per = cfg["config"], cfg["summary"]["per_k"]
            if cfg["block"] != "headline" or len(per) < 3 or c["rule"] != "unit":
                continue
            k2, n = per[2], cfg["summary"]["n"]
            t_mean = _num(k2["T_mean"])
            ref = REF_T2.get((name, c["r"])) if c["controller"] == "oracle" else None
            if ref is not None:
                t_ref, se, p_ref = ref
                lo, hi = (_num(x) for x in k2["T_ci"])           # nan without an interval (one replicate)
                band = [_num(x) for x in k2["p_band"]]
                rows.append(f"| {name} {_cell_label(cfg)}: mean T_E(2) [design, 100 replicates] | {t_ref} +- {se} | "
                            f"{_g(k2['T_mean'], '.4g')} [{_g(lo, '.4g')}, {_g(hi, '.4g')}] ({n} replicates) | "
                            f"{'yes' if lo - 2 * se <= t_ref <= hi + 2 * se else 'no'} (reference inside the CI) |")
                rows.append(f"| {name} {_cell_label(cfg)}: median p_2 [design] | {p_ref} | "
                            f"{_g(k2['p_median'], '.3f')} [{_g(band[0], '.3f')}, {_g(band[1], '.3f')}] | "
                            f"{'yes' if band[0] <= p_ref <= band[1] else 'no'} (inside the 16-84% band) |")
            box = REF_T2_INTERVALS.get((name, c["controller"], c["r"]))
            if box is not None:
                inside = box[0] <= t_mean <= box[1]
                extra = ""
                if c["controller"] == "lagged":
                    extra = f"; P(T(2) < T(1)) = {_g(k2.get('beats_psi1'), '.2f')} (reference >= 0.9)"
                    inside = inside and (k2.get("beats_psi1") or 0.0) >= 0.9
                rows.append(f"| {name} {_cell_label(cfg)}: mean T_E(2) in the 100-replicate interval [design] | "
                            f"[{box[0]}, {box[1]}] | {_g(k2['T_mean'], '.4g')} ({n} replicates){extra} | "
                            f"{'yes' if inside else 'no'} |")
    if not rows:
        return []
    return ["## Reference numbers", "",
            "Closed-loop references are 100-replicate means; with fewer replicates (--quick: 20) the comparison "
            "is against the run's own 95% bootstrap interval, widened by two reference standard errors.", "",
            "| quantity | reference | reproduced | match |", "|---|---|---|---|"] + rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="U5 on four systems, headline cells on two at 20 replicates")
    ap.add_argument("--systems", default=None, help="restrict both parts to these comma-separated systems")
    ap.add_argument("--out", default="results/m4", help="output directory (the cache goes to <out>/cache)")
    ap.add_argument("--no-cache", action="store_true", help="recompute and overwrite every cache entry")
    ap.add_argument("--replicates", type=int, default=None, help="replicates of every closed-loop configuration "
                                                                 "(default: per block)")
    ap.add_argument("--skip-u5", action="store_true", help="closed loop only")
    ap.add_argument("--skip-closed-loop", action="store_true", help="U5 only")
    ap.add_argument("--resume", action="store_true", help="load the systems whose loop_<s>.json or "
                                                          "closedloop_<s>.json in --out has this run's provenance")
    args = ap.parse_args(argv)
    t_script = time.perf_counter()
    from thc_engine_cqe.ft import scaling
    from thc_engine_cqe.ft.primitives import FTParameters

    selected = None if not args.systems else [s.strip() for s in args.systems.split(",") if s.strip()]
    os.makedirs(args.out, exist_ok=True)
    cache = scaling.Cache(os.path.join(args.out, "cache"), refresh=args.no_cache)
    params = FTParameters()
    u5, cl = {}, {}
    loaded_u5, loaded_cl = set(), set()
    if not args.skip_u5:
        names = [n for n in (U5_QUICK if args.quick else U5_SYSTEMS) if selected is None or n in selected]
        u5 = run_u5(names, cache, params, out_dir=args.out, resume=args.resume,
                    loaded=loaded_u5)                                 # loop_<s>.json as each system finishes
    if not args.skip_closed_loop:
        cl = run_closed_loop(selected, args.quick, args.replicates, cache, params, out_dir=args.out,
                             resume=args.resume, loaded=loaded_cl)
    command = "benchmarks/run_m4_loop.py" + (" --quick" if args.quick else "") + \
        (f" --systems {args.systems}" if args.systems else "") + \
        ("" if args.replicates is None else f" --replicates {args.replicates}") + \
        f" --out {os.path.relpath(args.out)}" + (" --resume" if args.resume else "")
    lines = ["# M4 step rules and the closed-loop finite-shot CQE: U5, U6, U7", "",
             f"Generated by `{command}` (one thread).", ""]
    if u5:
        lines += u5_table(u5) + [""]
    if cl:
        lines += closed_loop_tables(cl) + [""]
    checks = reference_checks(u5, cl)
    if checks:
        lines += checks + [""]
    runtime = []
    for n, b in u5.items():
        longest = max(b["runs"], key=lambda r: r.get("runtime_s") or 0.0)
        runtime.append((f"U5 {n}" + (" (loaded)" if n in loaded_u5 else ""), b["runtime_s"],
                        f"inputs {_g(b.get('inputs_runtime_s'), '.1f')} s; {longest['metric']} {_rule_label(longest)}, "
                        f"{_g(longest.get('runtime_s'), '.1f')} s"))
    runtime += [(f"closed loop {n}" + (" (loaded)" if n in loaded_cl else ""), b["runtime_s"], "-")
                for n, b in cl.items()]
    lines += ["## Wall time (one thread)", "",
              "U5 computes the inputs of a system once, then runs one calculation per metric and rule (R3 once per r); "
              "the longest is listed." + (" Rows marked (loaded) were read with --resume from files of the same "
                                          "provenance; their seconds are the recorded ones."
                                          if loaded_u5 or loaded_cl else ""), "",
              "| part | seconds | inputs; longest calculation |", "|---|---|---|"]
    lines += [f"| {p} | {s:.1f} | {c} |" for p, s, c in runtime]
    lines += ["", f"Script: {time.perf_counter() - t_script:.1f} s; cache hits / misses: {cache.hits} / {cache.misses}; "
                  f"peak RSS {peak_rss_mb():.0f} MB.", ""]
    with open(os.path.join(args.out, "loop_summary.md"), "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {len(u5) - len(loaded_u5)} loop_<system>.json, {len(cl) - len(loaded_cl)} closedloop_<system>.json "
          f"and loop_summary.md to {args.out}" + (f" ({len(loaded_u5)} U5 and {len(loaded_cl)} closed-loop systems "
                                                    f"loaded)" if args.resume else "")
          + f" ({time.perf_counter() - t_script:.1f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
