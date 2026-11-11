"""The M4 summary: verdicts (U0), the QPE-side scheme columns of U4, the schemes on the closed-loop states (U9 part 2),
the closed-loop deep runs' property floors (U11) and the physical units of the decisive runs (U12).

Inputs (``--out``, default results/m4, also the output directory): the JSON of benchmarks/run_m4_scaling.py
(scaling_<s>.json, scaling_fits.json), run_m4_loop.py (loop_<s>.json, closedloop_<s>.json), run_m4_handoff.py
(handoff_<s>.json) and run_m4_properties.py (properties_<s>.json), and their stage cache <out>/cache
(``ft.scaling.Cache``), from which the per-system inputs (``ft.closedloop.system_inputs``) and the scaling records
(``ft.scaling.scan_system``) load; ``--systems`` restricts every part to the listed systems. The inputs are complete
when every system that the producing scripts write by default is present (with ``--quick`` their quick sets, with
``--systems`` the listed systems among them) and scaling_fits.json exists: integration check "inputs complete". U0
keeps one row per question; a question without its inputs is stated as not computed, and a missing system as not
computed for that system. Units: Toffolis, in the tables / C_QPE of the system's selected encoding (F9); Hartree;
dE = 1 mHa.

QPE side of every system (F39-F43)
    As benchmarks/run_m4_handoff.py, with its cache stages: the model spectrum of H' (``ft.acceptance.
    model_spectrum``, N <= 8), E1lb = E_1' - 0.1 mHa from the lowest excited singlet of the ground irrep (of any
    irrep for THC encodings and where the eigenvectors of H' leave their irrep by more than 1e-6; F41), U =
    CISD(H') (``ft.certify.truncated_ci_energy``), certified iff U <= E1lb - 2 dE (F42), rule B's form and n_v
    for a state without a dedicated certificate (``ft.handoff.rule_b_bracket`` on the grid
    g = 2 pi lambda sin(theta_0)/M at M = 2 n_iter: the terminating option with the fewest runs, B_U with more runs
    where B_L does not terminate; B_L, theta_L = E1lb - 5 g, terminates iff E_0' + dE <= theta_L), the final QPE of
    the filtered schemes at a certified U (``ft.amplify.final_qpe_form``), the step filter and the coarse QPE
    (``ft.amplify.filter_cost``, eps_f = 1e-2; ``coarse_qpe_toffolis``, eps_p = 1e-4) at the certified U, at the
    pass edge E1lb - w_L of the E1lb-referenced forms where E_0' <= E1lb - w_L (w_L = 100 mHa;
    "termination_not_certified" unless U <= E1lb - w_L) and at the oracle U = E_0' + 2 dE (lower bounds only, where
    no U is certified). At N = 10 (no model spectrum): E1lb = E_1(H) - 1 mHa, E_0' ~ E_0 + dE_fact and theta_0 from
    it (flag "no_model_spectrum"). The QPE sides of the hand-off benchmark's systems are compared with its JSON, and
    its U9 part 1 rows are re-costed with them, flags included (integration checks). The U1 supplement reports the
    lowest excited level of the ground irrep of any spin of H' (``ft.certify.sector_levels``, F41). Sensitivity of
    Q5 to the classical certificate: U = CISDT(H') (F42's optional certificate, ``truncated_ci_energy(..., 3)``) in
    place of CISD(H') on the hand-off systems, with rule B's form, s* and T of the R1 spin-orbital HF, Psi_1 and
    Psi_2 rows and the first measured iteration's verdict at r = 1 and 3 (reported, never the verdict of record).

U4 scheme columns (F43, F44)
    Per row of scaling_fits.json (system, metric, r, k <= 2 of the R1 records; p on psi_0 of H as the
    repetition columns): mu_k^s = H_s(p_k, c_k) - H_s(p_k+1, c_k+1) and rho_k^s = c_iter(k)/mu_k^s for
    s in {bbht, aa, lt_bbht, lt_bbht_L} (``ft.amplify.scheme_saving``, ``ft.scaling.cost_ratio``), and at the best
    realizable scheme: s* at Psi_k and at Psi_k+1 with its flags (``ft.amplify.state_scheme_costs``,
    ``best_realizable`` over rep, bbht, fp with the Eckart-Temple p_lb of the state, lt_bbht and 2s with the
    certified U, lt_bbht_L and 2s_L where they terminate; rep, bbht, fp with rule B's verification runs),
    mu* = T_s*(k) - T_s*(k+1) + c_iter(k), and "pays" by ``ft.amplify.iteration_verdict``; the same over the set
    extended by the step filter followed by repetition (lt_rep, lt_rep_L; ``ft.amplify.extended_verdict``, a
    sensitivity). The n* brackets of the first measured iteration (F45; the same fits, cost curves, windows and
    margins as the repetition column, ``ft.scaling.break_even_sizes``) under bbht, aa and the best realizable scheme
    beyond the computed sizes, min over {rep, bbht} with B_L's n_v = 2: no certified U, no certified p_lb
    [assumption: CISD's error grows with n while Delta_1 falls], and the E1lb-referenced forms only while
    Delta_1(n) >= w_L + 1 mHa; where that window reaches beyond the largest computed size n0, the bracket holds only
    if no scheme can pay there: c_iter(n) >= c0 min(1, (n/n0)^beta) exceeds the largest H_s(p_k(n)) - 1 of every
    scheme at the window's end (flags "l_window_closed", "l_window_no_break_even", "l_window_unresolved").

U9 part 2 (F43, F44, F38)
    Per closed-loop system, configuration (closedloop_<s>.json) and replicate: T_s(k), k <= 4, under every scheme
    of F43 with the replicate's p~_k (on psi_0 of H: the replicates carry no states; within 0.5% of the weight on
    psi_0' on stretched N2), C(Psi~_k) and C_loop(k) (``ft.amplify.state_scheme_costs`` with its flags; rep, aa,
    bbht with rule B's n_v for CISD(H') as the only certificate, fp without p_lb; "rep_rule_e": M3's rule E). Per
    k: the mean of T_s(k) with its 95% percentile-bootstrap interval (``ft.closedloop.bootstrap_mean``; 2000
    resamples, seed 12345), s* of the means with its flags, and the optimum over the extended set
    (``ft.amplify.best_extended``); per measured iteration k -> k + 1: ``ft.closedloop.scheme_verdict`` (paired
    bootstrap of T_s(k) - T_s(k + 1) per scheme, and the verdict of record T_s*(k)(k) - T_s*(k+1)(k + 1)) and the
    paired verdict at the extended optima (a sensitivity). The deciding cell of a headline system (lagged r = 1, or
    its angle-cap rerun) decides U0's Q4; the deciding cell of a control system (the lagged r = 1 cell of the
    negative and the N = 8 controls) gives its control verdict; the lagged r = 0.5 and 2 cells are the sensitivity
    and the oracle cells the optimistic bound over r <= 3.

U11 closed-loop columns (F46, F47, F49)
    Deep runs (``ft.closedloop.replicate``): R1 spin-orbital, route best (the cheaper of direct_mg and ae_x at
    the planned budget), lagged controller at r = 0.3 and 1, 30 iterations, states kept, 10 replicates, on
    h4_r1.8, lih and n2_r1.098 (``--quick``: h4_r1.8, lih). Per replicate and target of F46 (as
    benchmarks/run_m4_properties.py: NONs, 1-RDM, 2-RDM, cumulant, cumulant norm, the LiH dipole) the property
    error of every iterate against psi_0 (``ft.properties.property_error``), its minimum over k and the last
    iterate's, k_eps per accuracy (the first k after which the error stays <= eps), C_loop(k_eps) at the tighter
    accuracy (``ft.closedloop.closed_loop_costs``), and the smallest F47 bound/|bias| (q and energy forms;
    ``ft.properties.bias_bound``); medians and 16-84% bands over the replicates, next to the exact R1 floor of
    properties_<s>.json.

U12 (F18, F19, F40)
    Logical qubits, Toffolis, physical qubits, code distance, factory, microseconds per Toffoli, seconds per
    run and machines for 7 days (``ft.physical.surface_code_cost``, ``machines_for``; SurfaceCodeModel defaults)
    of the decisive runs at their failure budgets: the n2_r2.4 residual at k = 1 (r = 1, 3; shots at
    p_fail_shot, the scaling record's campaign), one hand-off QPE run (F40: eps_fail/2 for a re-tested run or
    the final QPE of a filtered scheme, eps_fail p_min for an unverified post-selected one; f = 1 as in the
    costing), the coherent (filter +) AA run at the U9 optimum of that system (the whole expected AA sequence as
    one coherent run at eps_fail/2, F43), and the longest property AE run of the best AE rows (p_fail_ae, F48);
    a run that no factory and code distance keep within its budget is reported as having no layout.

U0 (the verdicts)
    One verdict per question Q1-Q6 with its table and its uncertainty and model dependence, built from the
    numbers above: Q1 from U1/U2 (each window statement per gap model: inside iff n(p = 1e-3) lies within the
    window under both the power law and A'/n; the window of a power law that grows with n opens at
    (m/A)^(1/|gamma|)), Q2 from U3/U4 (pays iff rho < 1 per scheme; at s* by ``ft.amplify.iteration_verdict``; n*
    against the windows of M3's rule E and of kernel-aware rule B per gap model, inside under a model iff one of its
    cases has Delta_1(n*) >= m), Q3 from U5 (k* or the first k with p >= 0.9 per system and metric) and U7 (the
    closed-loop step rules at oracle r = 1 and 3 against R1), Q4 from the deciding cells
    (``ft.closedloop.paired_verdict`` under repetition and ``ft.closedloop.scheme_verdict`` at s*) with every lagged
    cell of the headline block and the control cells, Q5 from U8/U9 (s* by ``ft.amplify.best_realizable``, the
    measured iteration by ``iteration_verdict``; M3's T6 rows keep M3's p on psi_0 of H), Q6 from U10/U11 (the best
    rows of ``ft.properties.best_property_row`` at both accuracies, route A at each pre-declared budget and on the
    compiled iterates, ``ft.properties.q6_comparison`` and ``q6_statements``, and the layouts of the best rows'
    longest coherent runs, ``ft.properties.run_layout``). Oracle schemes and the oracle controller are reported,
    never decisive.

Outputs: <out>/summary.json (strict JSON: nan as null, +-inf as "inf"/"-inf"; "checks_failed" lists the failed
integration checks) and <out>/summary.md, which opens with the failed integration checks. Exit status 1 if an
integration check fails; with ``--allow-partial`` the check "inputs complete" alone does not (the missing inputs are
still reported). The summary's own stages (model spectra, CISD, CISDT, property inputs, deep runs) are cached in
<out>/cache under keys that include the SHA-256 digest of this script, so that an edit of the script recomputes them
(``--no-cache`` recomputes them too; the upstream stages of the other benchmarks are only read through the cache).
One thread per process (the thread variables are set before numpy is imported).

Usage:
    python benchmarks/run_m4_summary.py --quick
    python benchmarks/run_m4_summary.py --out results/m4
"""

from __future__ import annotations

import os

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
for _var in THREAD_VARS:                         # one thread per process, before numpy is imported
    os.environ[_var] = "1"

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import pathlib  # noqa: E402
import re  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from thc_engine_cqe.fci_ops import FCISpace, normalize  # noqa: E402
from thc_engine_cqe.ft import acceptance as acc  # noqa: E402
from thc_engine_cqe.ft import amplify, certify, closedloop, handoff, physical, scaling  # noqa: E402
from thc_engine_cqe.ft import properties as P  # noqa: E402
from thc_engine_cqe.ft.primitives import FTParameters  # noqa: E402
from thc_engine_cqe.ft.symmetry import benchmark_system, determinant_irreps  # noqa: E402
from thc_engine_cqe.residual import rdm12  # noqa: E402

EPS_F = 1e-2                    # step-filter error (F43), as the hand-off benchmark
EPS_P = 1e-4                    # coarse-QPE leakage (F43)
IMPURITY_TOL = 1e-6             # eigenvector weight outside the dominant irrep that counts as broken symmetry
DENSE_MAX = 5000                # largest sector with a model spectrum (4900 at N = 8)
K_TABLE = 4                     # hand-off iterates of U9 part 2 (as U6)
U4_SCHEMES = ("bbht", "aa", "lt_bbht", "lt_bbht_L")
N_STAR_SCHEMES = ("bbht", "aa", "best")
BEST_BEYOND = ("rep", "bbht")   # the realizable set beyond the computed sizes (module docstring)
NV_BEYOND = 2                   # B_L's n_v at p_min = 1e-3 (F40), without a certified U
DEEP_SYSTEMS = ("h4_r1.8", "lih", "n2_r1.098")
DEEP_QUICK = ("h4_r1.8", "lih")
DEEP_R = (0.3, 1.0)
DEEP_ITER = 30
DEEP_REPLICATES = 10
DIPOLE_SYSTEMS = ("lih",)       # the polar all-active build of the property set
TINY_ERROR = 1e-9               # iterates of the F47 check: property error above this
U12_SYSTEM = "n2_r2.4"
U12_R = (1.0, 3.0)
MHA = 1e3
U_STAGES = {2: "cisd", 3: "cisdt"}  # excitation level of the classical certificate -> its cache stage (F42)
METRIC_LABEL = {"spin_orbital": "so", "spin_free": "sf"}
RULE_LABEL = {"unit": "R1", "hybrid": "R3", "line_search": "line search", "candidates": "M3 grid"}
CHAIN = re.compile(r"h(\d+)_r([0-9]+(?:\.[0-9]+)?)")     # the H chains of the size axis (F20)
#: SHA-256 of this script: part of the key of every stage the summary caches itself (module docstring)
SCRIPT_DIGEST = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
FAMILIES = ("scaling", "loop", "closedloop", "handoff", "properties")
#: U0 questions -> the input families they read; without the first one the question is not computed
Q_FAMILIES = {"Q1": ("fits", "scaling"), "Q2": ("fits", "scaling"), "Q3": ("loop", "closedloop"),
              "Q4": ("closedloop",), "Q5": ("handoff", "closedloop"), "Q6": ("properties",)}
Q_TABLES = {"Q1": "U1, U2", "Q2": "U3, U4", "Q3": "U5, U7", "Q4": "U6, U7, U9 (part 2)", "Q5": "U8, U9",
            "Q6": "U10, U11"}
INPUTS_CHECK = "inputs complete"
GAP_LABEL = {"power": "the power-law gap", "inverse": "A'/n"}
Q6_LOOPS = ("free", "meas_r1", "meas_r03")
Q6_BUDGETS = (2, 5, 10, 20, 60)
LOOP_LABEL = {"free": "a free loop", "meas_r1": "a measured r = 1 loop", "meas_r03": "a measured r = 0.3 loop"}
STEP_RULE_R = (1.0, 3.0)        # the oracle r of the closed-loop step-rule block (U7)
STATE_FLAGS = ("contaminant_below_theta", "exact_state_loop_costs")   # flags of a hand-off state, not of its scheme
CONTROL_LABEL = {"controls": "negative control", "n8_controls": "N = 8 control"}   # the control blocks (U7)


def peak_rss_mb() -> float:
    """Peak resident memory of this process so far (MB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _num(x) -> float:
    """A strict-JSON number back to a float (null -> nan, "inf"/"-inf" -> +-inf)."""
    if x is None:
        return math.nan
    if isinstance(x, str):
        return {"inf": math.inf, "-inf": -math.inf}[x]
    return float(x)


def _fin(x) -> float | None:
    """A float, or None for nan/inf (for the statistics)."""
    if x is None:
        return None
    v = _num(x) if isinstance(x, str) else float(x)
    return v if math.isfinite(v) else None


def _join(items, empty: str = "none") -> str:
    items = [str(x) for x in items]
    if not items:
        return empty
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _f(x, spec: str = ".3g") -> str:
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    x = float(x)
    if math.isnan(x):
        return "-"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    text = format(x, spec)
    if "e" in text:
        mant, exp = text.split("e")
        text = f"{mant}e{int(exp)}"
    return text


def _rng(v, spec: str = ".3g") -> str:
    """A [lo, hi] range (or any list of numbers) as "lo-hi"; "none" for None or an empty list."""
    vals = [] if v is None else [_num(x) for x in v if x is not None]
    if not vals:
        return "none"
    lo, hi = _f(min(vals), spec), _f(max(vals), spec)
    return lo if lo == hi else f"{lo}-{hi}"


def _ci(mean, ci, spec: str = ".4g") -> str:
    if mean is None:
        return "-"
    if ci is None or ci[0] is None or (isinstance(ci[0], float) and math.isnan(ci[0])):
        return _f(mean, spec)
    return f"{_f(mean, spec)} [{_f(ci[0], spec)}, {_f(ci[1], spec)}]"


def _table(header, rows) -> list[str]:
    return (["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
            + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows] + [""])


def _band(x) -> list[float]:
    return [float(v) for v in np.quantile(np.asarray(x, dtype=float), [0.16, 0.84])]


# ------------------------------------------------------------------ inputs
def discover(directory: str, prefix: str, systems) -> dict:
    """{system: JSON} of <directory>/<prefix>_<system>.json (restricted to ``systems`` if given)."""
    out = {}
    for path in sorted(glob.glob(os.path.join(directory, f"{prefix}_*.json"))):
        name = os.path.basename(path)[len(prefix) + 1:-len(".json")]
        if prefix == "scaling" and name == "fits":
            continue
        if systems is not None and name not in systems:
            continue
        with open(path) as fh:
            out[name] = json.load(fh)
    return out


def _sibling(name: str):
    """benchmarks/<name>.py as a module (for its system sets; the benchmark scripts are not a package)."""
    path = pathlib.Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_m4_summary_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def required_inputs(quick: bool, selected=None) -> dict:
    """{family: [systems]} that the producing scripts write by default (``quick``: their quick sets), restricted to
    ``selected`` (``--systems``): scaling S, O, V (``ft.scaling``), loop U5 and closedloop every closed-loop block
    (benchmarks/run_m4_loop.py), handoff and properties (their scripts' default sets)."""
    loop = _sibling("run_m4_loop")
    hand = _sibling("run_m4_handoff")
    props = _sibling("run_m4_properties")
    cl: list[str] = []
    for names, _, _ in loop.blocks(quick).values():
        cl += [s for s in names if s not in cl]
    sets = {"scaling": scaling.QUICK if quick else scaling.SIZE_AXIS + scaling.OVERLAP_AXIS + scaling.ACTIVE_SPACE_AXIS,
            "loop": loop.U5_QUICK if quick else loop.U5_SYSTEMS, "closedloop": cl,
            "handoff": hand.QUICK_SYSTEMS if quick else hand.DEFAULT_SYSTEMS,
            "properties": props.QUICK_SYSTEMS if quick else props.DEFAULT_SYSTEMS}
    return {f: [s for s in v if selected is None or s in selected] for f, v in sets.items()}


def missing_inputs(found: dict, fits_found: bool, required: dict) -> dict:
    """{family: [missing systems]} of the required inputs (``required_inputs``) and "fits": ["fits"] where
    scaling_fits.json is missing while scaling inputs are required; empty lists where complete."""
    out = {f: [s for s in required.get(f, []) if s not in (found.get(f) or {})] for f in FAMILIES}
    out["fits"] = [] if fits_found or not required.get("scaling") else ["fits"]
    return out


def _file(family: str, system: str) -> str:
    return "scaling_fits.json" if family == "fits" else f"{family}_{system}.json"


def inputs_check(missing: dict) -> dict:
    """The integration check "inputs complete" (module docstring)."""
    files = [_file(f, s) for f in ("fits",) + FAMILIES for s in missing.get(f, [])]
    return {"name": INPUTS_CHECK, "passed": not files,
            "detail": "every required input present" if not files else "missing: " + ", ".join(files)}


class Upstream:
    """The per-system inputs of the other benchmarks, loaded through their cache (module docstring)."""

    def __init__(self, cache: scaling.Cache, params: FTParameters, scan_kw: dict):
        self.cache, self.params, self.scan_kw = cache, params, scan_kw
        self._inputs: dict = {}
        self._scans: dict = {}

    def inputs(self, name: str):
        if name not in self._inputs:
            self._inputs[name] = closedloop.system_inputs(name, cache=self.cache, params=self.params)
        return self._inputs[name]

    def scan(self, name: str) -> scaling.ScalingRecord:
        if name not in self._scans:
            self._scans[name] = scaling.scan_system(name, cache=self.cache, params=self.params, **self.scan_kw)
        return self._scans[name]

    def parts(self, name: str) -> dict:
        """active, irreps, spectrum (H) and the selected QPE: system_inputs at N <= 8, the scaling scan beyond."""
        active = benchmark_system(name)
        if active.norb <= 8:
            inp = self.inputs(name)
            return {"active": inp.active, "irreps": inp.irreps, "spectrum": inp.spectrum, "qpe": inp.qpe}
        from thc_engine_cqe.ft.symmetry import orbital_irreps
        rec = self.scan(name)
        return {"active": active, "irreps": orbital_irreps(active), "spectrum": rec.spectrum, "qpe": rec.qpe}


def scan_settings(scaling_json: dict) -> dict:
    """k_cost, n_steps and r_values of the scaling run (the largest over its records; run_m4_scaling's defaults
    without records), so that ``scan_system`` loads the run's cache entries."""
    recs = [rec for data in scaling_json.values() for rec in data["records"].values()]
    if not recs:
        return {"k_cost": 3, "n_steps": 6, "r_values": (0.3, 1.0, 3.0)}
    rs = sorted({float(r) for rec in recs for r in rec["costs"]})
    k_cost = max(len(c) for rec in recs for c in rec["costs"].values()) - 1
    return {"k_cost": int(k_cost), "n_steps": int(max(rec["n_steps"] for rec in recs)), "r_values": tuple(rs)}


def _own_key(**fields) -> dict:
    """The key of one of the summary's own cache stages: ``fields`` and the digest of this script."""
    return {**fields, "script": SCRIPT_DIGEST}


# ------------------------------------------------------------------ the QPE side (F39-F43)
def _filters(enc, c: float, u: float, e1lb: float, params: FTParameters) -> dict:
    fc = amplify.filter_cost(enc, c, u, e1lb, eps=EPS_F, params=params)
    cc = amplify.coarse_qpe_toffolis(enc, c, u, e1lb, eps_p=EPS_P)
    return {"u": float(u), "degree": int(fc.degree), "c_filt": float(fc.toffolis), "c_coarse": float(cc),
            "filter_qubits": int(fc.logical_qubits)}


def qpe_side(name: str, parts: dict, cache: scaling.Cache, params: FTParameters, *, u_level: int = 2) -> dict:
    """E1lb, U, rule B's form, the filter and coarse-QPE costs of one system (module docstring); ``u_level`` 3 takes
    U = CISDT(H') in place of CISD(H') (the sensitivity of F42's optional certificate)."""
    if u_level not in U_STAGES:
        raise ValueError(f"u_level must be one of {sorted(U_STAGES)}, got {u_level!r}")
    active, irreps, q, sp = parts["active"], parts["irreps"], parts["qpe"], parts["spectrum"]
    enc = q.encoding
    key = _own_key(params=dataclasses.asdict(params), kind=enc.kind, enc_params=scaling.to_json(enc.params))
    model = enc.model(active)
    de = float(q.de_qpe)
    m2 = acc.qpe_points(q)
    flags: list[str] = []
    dim = FCISpace.of(active).size
    levels = None
    if dim <= DENSE_MAX:
        spec = cache.get(name, "model_spectrum", lambda: acc.model_spectrum(active, q, irreps), key=key)
        levels = certify.sector_levels(spec, det_irreps=determinant_irreps(active.norb, tuple(active.nelec), irreps))
        broken = enc.kind == "thc" or levels["irrep_impurity"] > IMPURITY_TOL
        if broken and enc.kind != "thc":
            flags.append("point_group_broken")
        e1_ref = min(spec.e1_singlet, spec.e1_any_irrep) if broken else spec.e1_singlet
        e1lb = certify.e1_lower_bound(e1_ref, model_spectrum=True)
        e0m, c, sin0, grid = float(spec.e0), float(spec.c), float(spec.sin_theta0), float(spec.grid(m2))
        theta0 = float(spec.theta0)
    else:
        flags.append("no_model_spectrum")
        e1lb = certify.e1_lower_bound(float(sp.e1 if sp.e1 is not None else sp.e1_any), model_spectrum=False)
        c = acc.walk_offset(model)
        e0m = float(sp.e0) + float(enc.de_fact)
        x = max(-1.0, min(1.0, (e0m - c) / float(enc.lam)))
        theta0 = math.acos(x)
        sin0 = math.sqrt(max(0.0, 1.0 - x * x))
        grid = 2.0 * math.pi * float(enc.lam) * sin0 / m2
    u = float(cache.get(name, U_STAGES[u_level], lambda: certify.truncated_ci_energy(model, u_level), key=key))
    certified = u <= e1lb - 2.0 * de
    u_cert = u if certified else None
    theta_l = float(acc.acceptance_bracket(None, e1lb, de, grid=grid).theta)
    # rule B's option without a dedicated certificate: the terminating form with the fewest runs (ft.handoff)
    br = handoff.rule_b_bracket(u_cert, e1lb, de, grid=grid, e0_model=e0m)
    rule_b_terminates = br is not None
    if br is None:                       # no form terminates: rep, aa, bbht and fp have no cost
        br = acc.acceptance_bracket(u_cert, e1lb, de, grid=grid)
    filters = {"u": _filters(enc, c, u, e1lb, params) if certified else None,
               "w_l": _filters(enc, c, e1lb - amplify.W_L, e1lb, params) if e0m <= e1lb - amplify.W_L else None,
               "oracle_u": (_filters(enc, c, e0m + 2.0 * de, e1lb, params)
                            if not certified and e0m + 2.0 * de < e1lb else None)}
    c_qpe = float(q.toffolis)
    side = {"system": name, "kind": enc.kind, "params": enc.params, "lam": float(enc.lam), "c_qpe": c_qpe,
            "logical_qubits": int(q.logical_qubits), "n_iter": int(q.iterations), "de": de, "e0_model": e0m,
            "e0": float(sp.e0), "e1": None if sp.e1 is None else float(sp.e1), "e1lb": float(e1lb), "u": u,
            "u_source": U_STAGES[u_level], "u_cisd": u if u_level == 2 else None,
            "u_certified": bool(certified), "walk_offset": c, "sin_theta0": sin0, "theta0": theta0, "m_2n": int(m2),
            "grid": grid, "form": br.form, "n_v": int(br.n_runs), "theta": float(br.theta),
            "distance": float(br.distance), "rule_b_terminates": bool(rule_b_terminates),
            "bracket_flags": list(br.flags), "theta_l": theta_l, "b_l_terminates": bool(e0m + de <= theta_l),
            "u_final": amplify.final_qpe_form(u, e1lb, de, grid=grid) if certified else None, "filters": filters,
            "w_l_certified": bool(certified and u <= e1lb - amplify.W_L), "flags": flags,
            "any_spin_gap": None if levels is None else float(levels["gap_e1_any_spin"]),
            "any_spin_s2": None if levels is None else float(levels["s2_e1_any_spin"]),
            "irrep_impurity": None if levels is None else float(levels["irrep_impurity"]),
            "norb": int(active.norb), "dim": int(dim)}
    side["scheme_inputs"] = scheme_inputs(side, 1.0 / c_qpe)
    return side


def scheme_inputs(side: dict, scale: float) -> dict:
    """The keyword arguments of ``ft.amplify.scheme_costs`` for a state of this system, costs scaled by ``scale``
    (1/C_QPE: C_QPE units)."""
    f = side["filters"]
    out = {"c_filt": None, "c_coarse": None, "c_filt_l": None, "c_coarse_l": None, "n_verify": side["n_v"]}
    if f["u"] is not None:
        out.update(c_filt=f["u"]["c_filt"] * scale, c_coarse=f["u"]["c_coarse"] * scale)
    if f["w_l"] is not None:
        out.update(c_filt_l=f["w_l"]["c_filt"] * scale, c_coarse_l=f["w_l"]["c_coarse"] * scale)
    return out


def _scaled_filter(f: dict | None, scale: float) -> dict | None:
    return None if f is None else {"c_filt": f["c_filt"] * scale, "c_coarse": f["c_coarse"] * scale}


def state_costs(side: dict, p: float, c_prep: float, c_loop: float, c_qpe: float, *, p_lb: float | None = None,
                rule_e: bool = True, flags: dict | None = None) -> dict:
    """T_s of one state under every scheme of F43 (``ft.amplify.state_scheme_costs``, in the units of the arguments):
    rule B's option of the side for rep, aa, bbht and fp (no cost where it does not terminate), the U-referenced forms
    at the certified U with their final QPE, the E1lb-referenced forms where they terminate, the oracle-U bounds where
    no U is certified, the sensitivity keys lt_rep and lt_rep_L, and M3's rule E ("rep_rule_e"); ``flags`` (a dict)
    receives the flags of every scheme."""
    scale = c_qpe / side["c_qpe"]
    f = side["filters"]
    out = amplify.state_scheme_costs(
        min(p, 1.0), c_prep, c_loop, c_qpe, filt_u=_scaled_filter(f["u"], scale),
        filt_wl=_scaled_filter(f["w_l"], scale), filt_oracle_u=_scaled_filter(f["oracle_u"], scale),
        n_verify=side["n_v"], rule_b_terminates=side["rule_b_terminates"], rule_b_flags=side["bracket_flags"],
        u_final=side["u_final"] or "B_U", b_l_terminates=side["b_l_terminates"], e0_model=side["e0_model"],
        e1_lower=side["e1lb"], w_l_certified=side["w_l_certified"], p_lb=p_lb)
    if flags is not None:
        flags.update(out["flags"])
    costs = dict(out["costs"])
    if rule_e:
        costs["rep_rule_e"] = c_loop + (c_prep + c_qpe) / min(p, 1.0)
    return costs


def _agree(a, b) -> bool:
    """Equal, or two real numbers (not bools) equal to 1e-12 relative."""
    if a == b:
        return True
    real = (int, float, np.integer, np.floating)
    if isinstance(a, real) and isinstance(b, real) and not isinstance(a, (bool, np.bool_)) \
            and not isinstance(b, (bool, np.bool_)):
        return abs(float(a) - float(b)) <= 1e-12 * abs(float(b))
    return False


def side_check(side: dict, hand: dict) -> dict:
    """The QPE side against the hand-off benchmark's JSON for the same system (exact agreement expected), with rule B's
    form for a state without a dedicated certificate: B_U with U = CISD(H') where it certifies and needs no more runs
    than B_L, or where B_L does not terminate; B_L otherwise."""
    q = hand["qpe_side"]
    pairs = {"e1lb": (side["e1lb"], q["e1lb"]), "u_cisd": (side["u_cisd"], q["u_cisd"]),
             "grid": (side["grid"], q["grid"]), "c_qpe": (side["c_qpe"], q["c_qpe"]),
             "b_l_terminates": (side["b_l_terminates"], q.get("b_l_terminates")),
             "u_final": (side["u_final"], q.get("u_final")),
             "theta_l_minus_e0_model": (side["theta_l"] - side["e0_model"], q.get("theta_l_minus_e0_model"))}
    for key in ("u", "w_l", "oracle_u"):
        a, b = side["filters"][key], q["filters"][key]
        pairs[f"c_filt_{key}"] = (None if a is None else a["c_filt"], None if b is None else b["c_filt"])
        pairs[f"c_coarse_{key}"] = (None if a is None else a["c_coarse"], None if b is None else b["c_coarse"])
    bad = sorted(k for k, v in pairs.items() if not _agree(*v))
    nv_u, nv_l = q["n_v_u"], q["n_v_l"]
    expected = (("B_U", nv_u) if q["u_certified"] and nv_u is not None
                and (nv_u <= nv_l or not q.get("b_l_terminates", True)) else ("B_L", nv_l))
    nv_ok = (side["form"], side["n_v"]) == expected
    return {"name": f"QPE side of {side['system']} equals the hand-off benchmark's", "passed": not bad and nv_ok,
            "detail": "agree" if not bad and nv_ok else f"differ: {bad}{'' if nv_ok else ', form and n_v'}"}


def _same_cost(a, b) -> bool:
    """Two scheme costs agree: both missing or infinite, or equal to 1e-12 relative."""
    a = None if a is None else _num(a)
    b = None if b is None else _num(b)
    if a is None or b is None or not (math.isfinite(a) and math.isfinite(b)):
        return (a is None or not math.isfinite(a)) and (b is None or not math.isfinite(b))
    return abs(a - b) <= 1e-12 * max(abs(a), abs(b), 1e-300)


def recost_check(side: dict, hand: dict) -> dict:
    """U9 part 1 of the hand-off benchmark re-costed with this QPE side (``state_costs`` on each row's p on psi_0',
    C(Psi_k), C_loop(k) and p_lb): the same T_s and the same flags (less the state's own "contaminant_below_theta" and
    "exact_state_loop_costs") for the E1lb-referenced forms on every row, for the U-referenced filtered forms and the
    filter followed by repetition wherever their U is the classical certificate (or none), and for rep, aa, bbht and
    fp wherever the row also uses the side's own rule-B form and n_v (rows with a dedicated energy certificate are
    skipped)."""
    n, bad = 0, []
    c_qpe = side["c_qpe"]
    for key, rows in hand["u9"].items():
        for row in rows:
            fl: dict = {}
            mine = state_costs(side, float(row["p_model"]), float(row["c_prep"]), float(row["c_loop"]), c_qpe,
                               p_lb=None if row.get("p_lb") is None else _num(row["p_lb"]), rule_e=False, flags=fl)
            same_form = (row["form"], row["n_v"]) == (side["form"], side["n_v"])
            for s, v in row["costs"].items():
                classical = row["sources"].get(s, "none") in ("none", "cisd")
                if s in ("lt", "lt_bbht", "2s", "lt_rep") and not classical:
                    continue
                if s in ("rep", "aa", "bbht", "fp") and not (classical and same_form):
                    continue
                n += 1
                theirs = [f for f in row.get("scheme_flags", {}).get(s, []) if f not in STATE_FLAGS]
                if not _same_cost(mine.get(s), v):
                    bad.append(f"{key} {row['state']} {s}")
                elif theirs != list(fl.get(s, [])):
                    bad.append(f"{key} {row['state']} {s} flags")
    return {"name": f"{side['system']}: U9 part 1 re-costed with the summary's QPE side equals the hand-off "
                    f"benchmark's", "passed": n > 0 and not bad,
            "detail": f"{n} costs" + (f"; differ: {', '.join(bad[:6])}" if bad else " agree")}


def hf_cost_check(name: str, hand: dict) -> dict:
    """U8's rule-B cost and M3's oracle T_E of HF (the values U0's Q5 quotes) against U9 part 1's rep and rule-E
    columns of the same state (both with p on psi_0' of H')."""
    label = f"{name}: U8's rule-B cost of HF equals U9's rep"
    hf = next((u for u in hand["u8"] if u["state"] == "HF" and not u["compiled"]), None)
    rows = [r for key, rws in hand["u9"].items() if hf is not None and key.startswith(hf["record"] + "@")
            for r in rws if r["k"] == 0]
    if hf is None or not rows:
        return {"name": label, "passed": False, "detail": "no HF row in U8 or U9"}
    pairs = [(_num(hf["cost_over_qpe"]), _num(r["costs_over_qpe"]["rep"])) for r in rows]
    pairs += [(_num(hf["cost_oracle_e_over_qpe"]), _num(r["rep_rule_e_over_qpe"])) for r in rows]
    ok = all(abs(a - b) <= 1e-12 * max(abs(a), abs(b)) for a, b in pairs)
    return {"name": label, "passed": ok, "detail": f"{len(rows)} rows" + (" agree" if ok else " differ")}


def cisdt_sensitivity(side_t: dict, side: dict, hand: dict) -> dict:
    """U = CISDT(H') in place of CISD(H') (F42's optional classical certificate; reported, never the verdict of
    record): rule B's form for a state without a dedicated certificate and, on the R1 spin-orbital rows of U9 part 1
    (HF, Psi_1, Psi_2; dedicated energy certificates not re-evaluated), s* and T_s* (C_QPE) per state and the
    verdict of the first measured iteration at r = 1 and 3, next to the hand-off benchmark's values (CISD)."""
    out = {"system": side_t["system"], "u_minus_e0_model": side_t["u"] - side_t["e0_model"],
           "certified": side_t["u_certified"], "form": side_t["form"], "n_v": side_t["n_v"],
           "certified_cisd": side["u_certified"], "form_cisd": side["form"], "n_v_cisd": side["n_v"], "by_r": {}}
    c_qpe = side_t["c_qpe"]
    for key in ("r1_so@r=1.0", "r1_so@r=3.0"):
        rows = {r["k"]: r for r in hand["u9"].get(key, [])}
        if not all(k in rows for k in (0, 1, 2)):
            continue
        costs = {k: state_costs(side_t, float(rows[k]["p_model"]), float(rows[k]["c_prep"]), float(rows[k]["c_loop"]),
                                c_qpe, p_lb=None if rows[k].get("p_lb") is None else _num(rows[k]["p_lb"]),
                                rule_e=False) for k in (0, 1, 2)}
        best = {k: amplify.best_realizable(costs[k]) for k in costs}
        ver = amplify.iteration_verdict(costs[1], costs[2])
        ver_cisd = amplify.iteration_verdict({s: _fin(v) for s, v in rows[1]["costs"].items()},
                                             {s: _fin(v) for s, v in rows[2]["costs"].items()})
        out["by_r"][key.split("=")[1]] = {
            "best": best, "t_best": {k: costs[k][best[k]] / c_qpe for k in costs},
            "best_cisd": {k: rows[k]["best_realizable"] for k in costs},
            "t_best_cisd": {k: _num(rows[k]["costs_over_qpe"][rows[k]["best_realizable"]]) for k in costs},
            "pays": ver["pays_best"], "relative": ver["relative_best"], "pays_cisd": ver_cisd["pays_best"],
            "relative_cisd": ver_cisd["relative_best"]}
    return out


# ------------------------------------------------------------------ U4: the scheme columns and n*
def _p_lb(rec_json: dict, k: int, e1) -> float | None:
    if e1 is None:
        return None
    e, v = _num(rec_json["energies"][k]), _num(rec_json["variances"][k])
    if not (math.isfinite(e) and math.isfinite(v)):
        return None
    lb = handoff.overlap_lower_bound(e, max(v, 0.0), float(e1))
    return lb if math.isfinite(lb) and lb > 0.0 else None


def u4_scheme_rows(fits: dict, scaling_json: dict, sides: dict) -> list[dict]:
    """The scheme columns of every U4 row (module docstring); C_QPE units."""
    rows = []
    for row in fits["U4"]["rows"]:
        side = sides.get(row["system"])
        data = scaling_json.get(row["system"])
        if side is None or data is None:
            continue
        rec = data["records"]["r1_" + row["metric"]]
        k = int(row["k"])
        p, p1 = _num(row["p"]), _num(row["p_next"])
        ck, ck1, c_it = _num(row["c_k"]), _num(row["c_k1"]), _num(row["c_iter"])
        e1 = data["spectrum"].get("e1")
        fl_k, fl_k1 = {}, {}
        ta = state_costs(side, p, ck, 0.0, 1.0, p_lb=_p_lb(rec, k, e1), rule_e=False, flags=fl_k)
        tb = state_costs(side, p1, ck1, c_it, 1.0, p_lb=_p_lb(rec, k + 1, e1), rule_e=False, flags=fl_k1)
        ver = amplify.iteration_verdict(ta, tb)
        ext = amplify.extended_verdict(ta, tb)
        kw = scheme_inputs(side, 1.0 / side["c_qpe"])
        out = {"system": row["system"], "N": row["N"], "metric": row["metric"], "r": row["r"], "k": k, "p": p,
               "p_next": p1, "c_iter": c_it, "mu_rep": _num(row["mu"]), "rho_rep": _num(row["rho"]),
               "pays_rep": bool(row["pays"]), "schemes": {}}
        for s in U4_SCHEMES:
            if ta.get(s) is None or tb.get(s) is None:
                out["schemes"][s] = None
                continue
            cf = kw["c_filt_l"] if s == "lt_bbht_L" else kw["c_filt"]
            sv = amplify.scheme_saving(s, c_qpe=1.0, c_filt=cf if s.startswith("lt") else None)
            mu = float(sv(p, p1, ck, ck1))
            rho = scaling.cost_ratio(c_it, p, p1, ck, ck1, saving=sv)
            out["schemes"][s] = {"mu": mu, "rho": rho, "pays": ver["by_scheme"][s]["pays"]}
        mu_best = None if ver["saving_best"] is None else ver["saving_best"] + c_it
        out["best"] = {"k": ver["best_k"], "k1": ver["best_k1"], "mu": mu_best,
                       "rho": None if mu_best is None else (c_it / mu_best if mu_best > 0 else math.inf),
                       "pays": ver["pays_best"], "relative": ver["relative_best"],
                       "flags_k": list(fl_k.get(ver["best_k"], [])) if ver["best_k"] else [],
                       "flags_k1": list(fl_k1.get(ver["best_k1"], [])) if ver["best_k1"] else []}
        out["extended"] = {"k": ext["best_k"], "k1": ext["best_k1"],
                           "mu": None if ext["saving_best"] is None else ext["saving_best"] + c_it,
                           "relative": ext["relative_best"], "pays": ext["pays_best"]}
        rows.append(out)
    return rows


def with_inside_by_model(by_margin: dict, cases) -> dict:
    """``by_margin`` of an n* bracket with "n_inside_by_model" in each margin: {gap model: the number of the
    bracket's cases of that margin and model whose n* lies inside the window of their own model, ``ft.scaling``'s
    per-case "inside" (Delta_1(n*) >= m)}."""
    out = {}
    for label, m in by_margin.items():
        counts = dict.fromkeys(scaling.GAP_MODELS, 0)
        for c in cases:
            if c.get("margin") == label and c.get("gap_model") in counts and c.get("inside"):
                counts[c["gap_model"]] += 1
        out[label] = dict(m, n_inside_by_model=counts)
    return out


def n_star_schemes(records: list, fits: dict, sides: dict) -> list[dict]:
    """F45 brackets under bbht, aa and the best realizable scheme (module docstring), with the window check; the
    cases of the bracket and of its crossed-corner sensitivity are dropped after their per-model counts inside the
    windows (``with_inside_by_model``)."""
    settings = fits["settings"]
    hooks = {"bbht": amplify.scheme_saving("bbht"), "aa": amplify.scheme_saving("aa"),
             "best": amplify.best_realizable_saving(schemes=BEST_BEYOND, n_verify=NV_BEYOND)}
    savings = {label: hooks[label] for label in N_STAR_SCHEMES}
    gaps = {float(g["R"]): g for g in fits["U2"]["gap"]}
    by_chain = {}
    for rec in records:
        m = CHAIN.fullmatch(rec.name)
        by_chain.setdefault(float(m.group(2)), {})[int(m.group(1))] = rec
    out = []
    for label, sv in savings.items():
        for b in scaling.break_even_sizes(records, saving=sv, scheme=label, sin_theta0=settings["sin_theta0"],
                                          e1_margin=settings["e1_margin"]):
            cases = b.get("cases") or []
            b = {k: v for k, v in b.items() if k != "cases"}
            if "by_margin" in b:
                b["by_margin"] = with_inside_by_model(b["by_margin"], cases)
            if "crossed" in b:
                crossed = b["crossed"]
                b["crossed"] = {k: v for k, v in crossed.items() if k != "cases"}
                if "by_margin" in crossed:
                    b["crossed"]["by_margin"] = with_inside_by_model(crossed["by_margin"], crossed.get("cases") or [])
            if label == "best":
                b["l_window"] = l_window_check(b, gaps.get(float(b["R"])), by_chain.get(float(b["R"]), {}), sides)
            out.append(b)
    return out


def l_window_check(b: dict, gap: dict | None, chain: dict, sides: dict) -> dict:
    """Whether the E1lb-referenced forms, left out beyond the computed sizes, could pay inside their window."""
    thr = amplify.W_L + certify.E1_MARGIN_EXACT
    if gap is None:
        return {"flag": "l_window_unresolved", "n_l": None, "reason": "no gap fit"}
    n0 = int(b["n0"])
    if gap["gamma"] > 0:
        n_pow = (gap["A"] / thr) ** (1.0 / gap["gamma"])
    else:       # a power law that does not close (F21): every n if it grows or stays at or above thr, else none
        n_pow = math.inf if gap["gamma"] < 0 or gap["A"] >= thr else 0.0
    n_l = max(n_pow, gap["A_inv"] / thr)
    if n_l < n0 + 2:
        return {"flag": "l_window_closed", "n_l": n_l}
    if not math.isfinite(n_l):
        return {"flag": "l_window_unresolved", "n_l": n_l, "reason": "the window does not close"}
    rec = chain.get(n0)
    if rec is None:
        return {"flag": "l_window_unresolved", "n_l": n_l, "reason": "no record at n0"}
    key = {"sf": "lbfgs/second_order", "so": "lbfgs_so/second_order"}[b["metric"]]   # the series of alpha_1
    p0 = float(rec.records[key].overlaps[1])
    alphas = [a for a in b["alpha_k"] if a is not None]
    alpha = max(alphas) if alphas else 0.0
    p_lo = min(1.0, p0 * math.exp(-max(alpha, 0.0) * (n_l - n0)))
    side = sides.get(rec.name)
    kw = {} if side is None else scheme_inputs(side, 1.0 / side["c_qpe"])
    h = [1.0 / p_lo, amplify.exponential_search_calls(p_lo)]
    if kw.get("c_filt_l") is not None:
        h.append(amplify.exponential_search_calls(p_lo) * kw["c_filt_l"] + 1.0)
    if kw.get("c_coarse_l") is not None:
        h.append(kw["c_coarse_l"] / p_lo + 1.0)
    mu_bound = max(h) + (NV_BEYOND - 1) - 1.0
    beta_min = min(b["betas"].values())
    c_min = b["c0"] * min(1.0, (n_l / n0) ** beta_min)
    flag = "l_window_no_break_even" if c_min > mu_bound else "l_window_unresolved"
    return {"flag": flag, "n_l": n_l, "p_low": p_lo, "mu_bound": mu_bound, "c_iter_min": c_min}


def _best_or_none(fn, costs: dict) -> str | None:
    try:
        return fn(costs)
    except ValueError:
        return None


# ------------------------------------------------------------------ U9 part 2 and the closed-loop verdicts
def closed_loop_schemes(data: dict, side: dict) -> dict:
    """Schemes on every configuration's replicates (module docstring); C_QPE units."""
    c_qpe = float(data["system"]["c_qpe"])
    out, checks = [], []
    worst_rule_e = 0.0
    for cfg in data["configs"]:
        reps = cfg["replicates"]
        kmax = min(min(len(r["p"]) for r in reps) - 1, K_TABLE)
        fl: dict = {}                    # the schemes' flags (they depend on the QPE side only)
        per_k = []
        for k in range(kmax + 1):
            rows = []
            for rep in reps:
                c = state_costs(side, _num(rep["p"][k]), _num(rep["c_prep"][k]), _num(rep["c_loop"][k]), c_qpe,
                                flags=None if fl else fl)
                ref = _num(rep["handoff"][k]["cost_expected"])
                worst_rule_e = max(worst_rule_e, abs(c["rep_rule_e"] - ref) / ref)
                rows.append({s: None if v is None else v / c_qpe for s, v in c.items()})
            per_k.append(rows)
        stats = []
        for k, rows in enumerate(per_k):
            keys = dict.fromkeys(s for row in rows for s in row)
            means = {}
            for s in keys:
                vals = [row.get(s) for row in rows]
                if all(v is not None and math.isfinite(v) for v in vals):
                    m, lo, hi = closedloop.bootstrap_mean(vals)
                    means[s] = {"mean": m, "ci": [lo, hi]}
            mean_of = {s: v["mean"] for s, v in means.items()}
            best = _best_or_none(amplify.best_realizable, mean_of)
            p = [_num(r["p"][k]) for r in reps]
            stats.append({"k": k, "T": means, "best": best, "flags_best": list(fl.get(best, [])) if best else [],
                          "best_extended": _best_or_none(amplify.best_extended, mean_of),
                          "p_median": float(np.median(p)), "p_band": _band(p)})
        verdicts = []
        for k in range(kmax):
            v = closedloop.scheme_verdict(per_k[k], per_k[k + 1])
            e_k = _best_or_none(amplify.best_extended, v["means_k"])
            e_k1 = _best_or_none(amplify.best_extended, v["means_k1"])
            ext = None
            if e_k is not None and e_k1 is not None:
                ext = {"best_k": e_k, "best_k1": e_k1,
                       **closedloop.paired_verdict([row[e_k] for row in per_k[k]], [row[e_k1] for row in per_k[k + 1]])}
            verdicts.append({"k": k, **v, "flags_best_k": list(fl.get(v["best_k"], [])) if v["best_k"] else [],
                             "flags_best_k1": list(fl.get(v["best_k1"], [])) if v["best_k1"] else [],
                             "extended": ext})
            a_sum = cfg["summary"]["per_k"][k] if k < len(cfg["summary"]["per_k"]) else None
            if a_sum is not None and "diff_mean" in a_sum and "rep_rule_e" in v["by_scheme"]:
                mine = v["by_scheme"]["rep_rule_e"]
                same = (abs(mine["mean"] - a_sum["diff_mean"]) <= 1e-12 * max(1.0, abs(a_sum["diff_mean"]))
                        and all(abs(x - y) <= 1e-12 * max(1.0, abs(y)) for x, y in zip(mine["ci"], a_sum["diff_ci"]))
                        and mine["pays"] == a_sum["pays"])
                checks.append(same)
        out.append({"label": cfg["label"], "block": cfg["block"], "deciding": bool(cfg.get("deciding")),
                    "config": cfg["config"], "n": len(reps), "realizable": cfg["summary"].get("realizable"),
                    "non_realizable": cfg["summary"].get("non_realizable", []),
                    "r_median": [x.get("r_median") for x in cfg["summary"]["per_k"][:kmax + 1]],
                    "stats": stats, "verdicts": verdicts,
                    "flags": cfg.get("flags", []) + (["superseded_by_angle_cap_rerun"]
                                                     if cfg.get("superseded_by_angle_cap_rerun") else [])})
    return {"system": data["system"]["name"], "c_qpe": c_qpe, "configs": out,
            "check_rule_e": {"passed": bool(checks) and all(checks) and worst_rule_e <= 1e-12,
                             "verdicts": len(checks), "worst_relative": worst_rule_e}}


# ------------------------------------------------------------------ U11: the deep runs
def build_targets(name: str, active, irreps, psi0) -> dict:
    """The F46 targets at psi_0, as benchmarks/run_m4_properties.py."""
    dm1, dm2 = rdm12(FCISpace.of(active), psi0)
    out = {"occupations": P.occupation_target(active, dm1), "rdm1": P.rdm1_target(active, irreps),
           "rdm2": P.rdm2_target(active, irreps), "cumulant": P.cumulant_target(active, irreps, dm1),
           "cumulant_norm": P.cumulant_norm_target(active, irreps, dm1, dm2)}
    if name in DIPOLE_SYSTEMS:
        out["dipole"] = P.dipole_target(active)
    return out


def _k_eps(err: np.ndarray, eps: float) -> int | None:
    above = np.flatnonzero(err > eps)
    k = 0 if above.size == 0 else int(above[-1]) + 1
    return k if k < err.size else None


def deep_runs(name: str, up: Upstream, own: scaling.Cache, params: FTParameters, *, n: int, max_iter: int) -> dict:
    """The deep closed-loop runs of one system and their property errors (module docstring)."""
    inputs = up.inputs(name)
    active, sp = inputs.active, inputs.spectrum
    psi0 = normalize(np.asarray(sp.psi0))
    targets = build_targets(name, active, inputs.irreps, psi0)
    inps = own.get(name, "properties_inputs",
                   lambda: {k: P.property_inputs(active, psi0, t) for k, t in targets.items()},
                   key=_own_key(targets=sorted(targets)))
    gap = float(sp.e1 - sp.e0)

    def compute():
        out = {}
        for r in DEEP_R:
            spec = closedloop.LoopSpec(route="best", controller="lagged", r=r)
            recs = closedloop.replicate(inputs, spec, "unit", n=n, max_iter=max_iter, keep_states=True)
            reps = []
            for rec in recs:
                costs = closedloop.closed_loop_costs(inputs, rec, params=params)
                c_loop = np.concatenate([[0.0], np.cumsum([float(c.toffolis) for c in costs])])
                errors, f47 = {}, {}
                for kind, t in targets.items():
                    err = np.array([P.property_error(t, active, s, psi0) for s in rec.states])
                    errors[kind] = err
                    rq, re_ = [], []
                    for k, b in enumerate(err):
                        if b <= TINY_ERROR:
                            continue
                        q = 1.0 - min(float(rec.overlaps[k]), 1.0)
                        rq.append(P.bias_bound(t, active, psi0, q=q, sigma0=inps[kind].sigma0,
                                               ranges=inps[kind].r0) / b)
                        d = max(float(rec.energies[k] - sp.e0), 0.0)
                        re_.append(P.bias_bound(t, active, psi0, energy_error=d, gap=gap, sigma0=inps[kind].sigma0,
                                                ranges=inps[kind].r0) / b)
                    f47[kind] = (min(rq) if rq else None, min(re_) if re_ else None)
                reps.append({"seed": list(rec.seed), "n_steps": rec.n_steps, "converged": bool(rec.converged),
                             "p": np.asarray(rec.overlaps, float), "e_err": np.asarray(rec.energies, float) - sp.e0,
                             "c_loop": c_loop, "r_realized": np.asarray(rec.r_realized, float),
                             "flags": sorted({f for step in rec.flags for f in step}),
                             "non_realizable": list(rec.non_realizable), "errors": errors, "f47": f47})
            out[r] = reps
        return out

    runs = own.get(name, "summary_deep_runs", compute,
                   key=_own_key(n=n, max_iter=max_iter, r=list(DEEP_R), params=dataclasses.asdict(params),
                                targets=sorted(targets)))
    c_qpe = float(inputs.qpe.toffolis)
    summary = {}
    for kind in targets:
        accs = P.ACCURACIES[kind]
        per_r = {}
        for r, reps in runs.items():
            mins = [float(np.min(rep["errors"][kind])) for rep in reps]
            kmin = [int(np.argmin(rep["errors"][kind])) for rep in reps]
            last = [float(rep["errors"][kind][-1]) for rep in reps]
            k_eps = {}
            for eps in accs:
                ks = [_k_eps(rep["errors"][kind], eps) for rep in reps]
                hit = [(k, rep) for k, rep in zip(ks, reps) if k is not None]
                loop_at = [rep["c_loop"][min(k, len(rep["c_loop"]) - 1)] for k, rep in hit]
                k_eps[str(eps)] = {"reached": len(hit) / len(reps),
                                   "k_median": float(np.median([k for k, _ in hit])) if hit else None,
                                   "c_loop_median_over_qpe": float(np.median(loop_at)) / c_qpe if hit else None}
            rq = [rep["f47"][kind][0] for rep in reps if rep["f47"][kind][0] is not None]
            re_ = [rep["f47"][kind][1] for rep in reps if rep["f47"][kind][1] is not None]
            per_r[f"{r:g}"] = {"min_median": float(np.median(mins)), "min_band": _band(mins),
                               "k_min_median": float(np.median(kmin)), "last_median": float(np.median(last)),
                               "last_band": _band(last), "k_eps": k_eps,
                               "f47_q_min": min(rq) if rq else None, "f47_energy_min": min(re_) if re_ else None,
                               "hf_error": float(reps[0]["errors"][kind][0])}
        summary[kind] = {"norm": targets[kind].norm, "accuracies": list(accs), "by_r": per_r}
    loops = {}
    for r, reps in runs.items():
        loops[f"{r:g}"] = {"n": len(reps), "p_last_median": float(np.median([rep["p"][-1] for rep in reps])),
                           "e_err_last_median": float(np.median([rep["e_err"][-1] for rep in reps])),
                           "r1_median": float(np.median([rep["r_realized"][0] for rep in reps])),
                           "c_loop_last_median_over_qpe": float(np.median([rep["c_loop"][-1] for rep in reps])) / c_qpe,
                           "flags": sorted({f for rep in reps for f in rep["flags"]}),
                           "non_realizable": sorted({f for rep in reps for f in rep["non_realizable"]}),
                           "converged": sum(rep["converged"] for rep in reps)}
    return {"system": name, "c_qpe": c_qpe, "targets": summary, "loops": loops, "max_iter": max_iter, "n": n}


# ------------------------------------------------------------------ U12
def _layout(label: str, system: str, logical: int, toffolis: float, budget: float, n_runs: float,
            model: physical.SurfaceCodeModel, *, shot: bool = False, note: str = "") -> dict:
    row = {"run": label, "system": system, "logical_qubits": int(logical), "toffolis": float(toffolis),
           "fail_budget": float(budget), "runs": float(n_runs), "note": note}
    try:
        pc = physical.surface_code_cost(int(logical), max(float(toffolis), 1.0), fail_budget=budget, model=model,
                                        shot=shot)
    except ValueError as exc:
        row.update(error=str(exc))
        return row
    row.update(physical_qubits=pc.physical_qubits, distance=pc.distance, factory=pc.factory,
               us_per_toffoli=pc.us_per_toffoli, seconds=pc.seconds,
               machines_7d=physical.machines_for(math.ceil(max(n_runs, 1.0)), pc))
    return row


def u12_rows(up: Upstream, scaling_json: dict, handoff_json: dict, properties_json: dict, sides: dict,
             params: FTParameters, model: physical.SurfaceCodeModel) -> list[dict]:
    rows = []
    name = U12_SYSTEM
    if name in scaling_json:
        rec = up.scan(name)
        for r in U12_R:
            if r not in rec.costs.get("r1_so", {}) or len(rec.costs["r1_so"][r]) < 2:
                continue
            it = rec.costs["r1_so"][r][1]
            route = next((rc for rc in it.residual if rc.route == it.best), None)
            camp = next((c for c in it.campaigns if c.label == it.best), None)
            if route is None or camp is None or camp.per_run is None:
                continue
            pc = camp.per_run
            per_run = route.toffolis_per_prep if route.route in ("direct_mg", "aux_mg") else route.coherent_toffolis
            rows.append({"run": f"residual at k = 1 (r = {r:g}, {it.best})", "system": name,
                         "logical_qubits": int(route.logical_qubits), "toffolis": float(per_run),
                         "fail_budget": params.p_fail_shot if route.route in ("direct_mg", "aux_mg")
                         else params.p_fail_ae, "runs": float(camp.n_runs), "physical_qubits": pc.physical_qubits,
                         "distance": pc.distance, "factory": pc.factory, "us_per_toffoli": pc.us_per_toffoli,
                         "seconds": pc.seconds, "machines_7d": it.machines_7d,
                         "note": f"C_iter(1)/C_QPE = {float(it.toffolis) / rec.c_qpe:.3g}"})
    side = sides.get(name)
    hand = handoff_json.get(name)
    if side is not None:
        filtered_or_retested = True
        best_row = None
        if hand is not None:
            cands = [row for key, rws in hand["u9"].items() if key == "r1_so@r=1.0" for row in rws
                     if row["best_realizable"] is not None]
            if cands:
                best_row = min(cands, key=lambda row: _num(row["costs"][row["best_realizable"]]))
                s = best_row["best_realizable"]
                filtered_or_retested = s in ("lt", "lt_bbht", "2s", "lt_bbht_L", "2s_L") or best_row["n_v"] >= 2
        kind = "retested" if filtered_or_retested else "unverified"
        budget = acc.failure_budget(kind, f=1.0)
        rows.append(_layout("hand-off QPE run (final, F40 " + kind + ")", name, side["logical_qubits"], side["c_qpe"],
                            budget, 1.0, model, note=f"C_QPE, n_iter = {side['n_iter']:,}"))
        if best_row is not None:
            s = best_row["best_realizable"]
            k, p = int(best_row["k"]), float(best_row["p_model"])
            rec_j = hand["records"][best_row["record"]]
            q_prep = int(_num(rec_j["prep_qubits"][k]))
            extra_q = max(0, q_prep - 2 * side["norb"])
            c_prep = float(best_row["c_prep"])
            if s in ("lt_bbht", "lt_bbht_L", "lt", "bbht", "aa"):
                calls = amplify.grover_calls(p)[0] if s in ("lt", "aa") else amplify.exponential_search_calls(p)
                f = None
                if s == "lt_bbht_L":
                    f = side["filters"]["w_l"]
                elif s in ("lt", "lt_bbht"):
                    src = best_row["sources"].get(s)
                    cert = next((c for c in rec_j.get("certificates", []) if c.get("label") == src), None)
                    if src == "cisd" or cert is None:
                        f = side["filters"]["u"]
                    else:                        # a dedicated energy certificate of the state places the pass edge
                        parts = up.parts(name)
                        f = _filters(parts["qpe"].encoding, side["walk_offset"], _num(cert["upper"]), side["e1lb"],
                                     params)
                per_call = c_prep + (side["c_qpe"] if f is None else f["c_filt"])
                q = (side["logical_qubits"] if f is None else f["filter_qubits"]) + extra_q
                rows.append(_layout(f"coherent {'filter + ' if f is not None else ''}AA run ({s}, {best_row['state']})",
                                    name, q, calls * per_call, acc.failure_budget("retested", f=1.0), 1.0, model,
                                    note=f"{calls:.3g} expected calls of C(Psi) + "
                                         f"{'C_filt' if f is not None else 'C_QPE'} as one coherent run; "
                                         f"T/C_QPE = {_num(best_row['costs'][s]) / side['c_qpe']:.4g}"))
            else:
                rows.append({"run": f"U9 optimum {s} ({best_row['state']}): no coherent AA run", "system": name,
                             "note": "repeated runs"})
    longest = None
    for sname, data in properties_json.items():
        c_qpe = float(data["qpe"]["c_qpe"])
        for kind, blk in data["targets"].items():
            for eps in blk["accuracies"]:
                key = str(eps)
                for lp in ("free", "meas_r1"):
                    b = blk["best"][key][lp]["ae"]
                    if b is None or b["estimator"] != "ae":
                        continue
                    pool = blk["rows"][key]["route_a"] + blk["rows"][key]["route_b"]
                    row = next((x for x in pool if x["route"] == b["route"] and x["loop"] == lp and x["k"] == b["k"]
                                and x["estimator"] == "ae" and x["selectable"]
                                and abs(_num(x["toffolis_over_qpe"]) - b["toffolis_over_qpe"])
                                <= 1e-9 * b["toffolis_over_qpe"]), None)
                    if row is None or not math.isfinite(_num(row["longest_run"])):
                        continue
                    if longest is None or _num(row["longest_run"]) > _num(longest[1]["longest_run"]):
                        longest = (sname, row, c_qpe, kind, eps, lp)
    if longest is not None:
        sname, row, c_qpe, kind, eps, lp = longest
        n_runs = _num(row["toffolis"]) / _num(row["longest_run"])
        rows.append(_layout(f"longest property AE run ({kind} at {eps:g}, route {row['route']}, loop {lp})", sname,
                            int(row["logical_qubits"]), _num(row["longest_run"]), params.p_fail_ae, n_runs, model,
                            note=f"route T/C_QPE = {_num(row['toffolis']) / c_qpe:.4g}; runs counted at the longest"))
    return rows


# ------------------------------------------------------------------ U0
def _u1_rows(fits: dict) -> dict:
    return {r["system"]: r for r in fits.get("U1", [])}


def _tag(flags) -> str:
    """" [flag, ...]" of a scheme's flags, "" without flags."""
    return f" [{', '.join(flags)}]" if flags else ""


def _gap_ends(g: dict, label: str) -> dict:
    """{gap model: [window ends]} of a U2 gap row at the margin ``label``: n_cert = [A'/m, (A/m)^(1/gamma)]
    (``ft.scaling.certification_window``, the inverse model first); for B_L, where the HF overlap falls below F40's
    two-run floor inside the window, the sound two-run window of HF per overlap rate ("n_cert_sound")."""
    inv, power = g["n_cert"][label]
    ends = {"power": [_num(power)], "inverse": [_num(inv)]}
    two = g.get("bl_two_run") or {}
    if label == "BL" and "hf" in (two.get("below_floor") or []):
        sound = two["n_cert_sound"]["hf"]
        ends = {m: [_num(x) for x in sound[m]] for m in ("power", "inverse")}
    return ends


def _gap_starts(g: dict, label: str, margin: float | None = None) -> dict:
    """{gap model: the smallest n of the window} of a U2 gap row at the margin ``label`` (m = ``margin``, else A'/n_cert
    of the inverse model): 0, except under a power law that grows with n (gamma < 0), whose window
    [(m/A)^(1/|gamma|), inf) starts where A n^-gamma reaches m (F21). A constant power law (gamma = 0) holds every n
    or none, which its window end (inf or 0, ``ft.scaling.certification_window``) states."""
    gamma = _num(g["gamma"])
    if not gamma < 0.0:
        return {"power": 0.0, "inverse": 0.0}
    m = _num(margin) if margin is not None else _num(g["A_inv"]) / _num(g["n_cert"][label][0])
    return {"power": (m / _num(g["A"])) ** (1.0 / -gamma), "inverse": 0.0}


#: window statuses that leave the sizes inside or beyond the end of a window (a closing gap)
_END_SIDE = {"inside", "beyond", "straddling its end"}
#: window statuses with every size outside the window
_OUTSIDE = {"beyond", "short of its start"}


def window_status(n_at, ends: dict, starts: dict | None = None) -> dict:
    """{gap model: where the sizes ``n_at`` lie against its window} with the ends ``ends`` ({gap model: [ends]}) and
    the starts ``starts`` ({gap model: start}, 0 where absent, ``_gap_starts``): "inside" iff every n lies within
    [start, min of the ends], "beyond" iff every n exceeds every end, "short of its start" iff every n lies below
    the start, else "straddling its start" or "straddling its end"."""
    lo, hi = min(n_at), max(n_at)
    out = {}
    for m, e in ends.items():
        a = (starts or {}).get(m, 0.0)
        out[m] = ("inside" if a <= lo and hi <= min(e) else "beyond" if lo > max(e) else
                  "short of its start" if hi < a else "straddling its start" if lo < a else "straddling its end")
    return out


def window_position(n_at, ends: dict, starts: dict | None = None) -> str:
    """Where the sizes ``n_at`` (n at p = 1e-3 over the overlap bracket) lie against a window (``window_status``):
    "inside" iff inside under every gap model, "beyond" iff beyond under every model, otherwise per gap model
    ("inside under the power-law gap, beyond under A'/n")."""
    status = window_status(n_at, ends, starts)
    if set(status.values()) == {"inside"}:
        return "inside"
    if set(status.values()) == {"beyond"}:
        return "beyond"
    return ", ".join(f"{st} under {GAP_LABEL[m]}" for m, st in status.items())


def _lies(n_at, ends: dict, name: str, starts: dict | None = None) -> str:
    pos = window_position(n_at, ends, starts)
    if pos in ("inside", "beyond"):
        return f"{pos} the {name} window"
    status = set(window_status(n_at, ends, starts).values())
    if status <= _END_SIDE:
        return f"inside or beyond the {name} window depending on the gap model ({pos})"
    if status <= _OUTSIDE:
        return f"outside the {name} window ({pos})"
    return f"inside or outside the {name} window depending on the gap model ({pos})"


def _window_text(ends: dict, label: str, starts: dict | None = None) -> str:
    """'<range> for <label> (A'/n <end>, power law <end>)' of a window's ends per gap model, with the start of a
    window that opens at n > 0 (a growing power-law gap)."""
    vals = [x for e in ends.values() for x in e]
    a = (starts or {}).get("power", 0.0)
    power = _rng(ends["power"], ".0f") if not a > 0.0 else f"from {_f(a, '.0f')} to {_rng(ends['power'], '.0f')}"
    return f"{_rng(vals, '.0f')} for {label} (A'/n {_rng(ends['inverse'], '.0f')}, power law {power})"


def _rate(row: dict) -> str:
    return f"{_f(row['alpha'], '.4f')}-{_f(row['alpha_aitken'], '.4f')}"


def _rate_ratio(hf: dict, row: dict) -> str:
    """The range of alpha_HF/alpha of a row at matched rate estimates (last successive, Aitken)."""
    vals = [_num(hf[key]) / _num(row[key]) for key in ("alpha", "alpha_aitken")
            if row.get(key) is not None and _num(row[key]) > 0.0]
    return _rng(vals, ".2g")


def verdict_q1(fits: dict, scaling_json: dict) -> dict:
    lines, unc = [], []
    overlap = fits["U2"]["overlap"]
    hf = [r for r in overlap if r["record"] == "hf"]
    gaps = {float(g["R"]): g for g in fits["U2"]["gap"]}
    margins = (fits.get("settings") or {}).get("margins") or {}
    for r in hf:
        R = float(r["R"])
        n_at = [_num(x) for x in r["n_at_p"] if x is not None and math.isfinite(_num(x))]
        p1 = r.get("p1_at_n") or {}
        p1s = "; ".join(f"{m} {_rng(sorted(_num(v) for v in vals if v is not None), '.2g')}"
                        for m, vals in p1.items() if vals)
        text = (f"R = {R:.1f} bohr: p_HF decays by alpha = {_f(r['alpha'], '.4f')} per atom (successive "
                f"{', '.join(_f(a, '.4f') for a in r['successive'])}; Aitken {_f(r['alpha_aitken'], '.4f')}) and "
                f"reaches 1e-3 at n = {_rng(sorted(n_at), '.0f')}"
                + (f", where the free classical step leaves p_1 = {p1s}" if p1s else ""))
        rec = {x["record"]: x for x in overlap if float(x["R"]) == R}
        cl = [rec[k] for k in ("classical_sf", "classical_so") if k in rec]
        r1 = [rec[k] for k in ("r1_sf", "r1_so") if k in rec]
        if cl or r1:
            parts = []
            if cl:
                parts.append("after the free classical step by " + " and ".join(
                    f"{_rate(x)} ({x['record'].split('_')[1]}; alpha_HF/alpha = {_rate_ratio(r, x)})" for x in cl))
            if r1:
                parts.append("after the first measured iteration (R1 Psi_2) by " + " and ".join(
                    f"{_rate(x)} ({x['record'].split('_')[1]}; alpha_HF/alpha = {_rate_ratio(r, x)})" for x in r1))
            text += ("; per atom, from the last successive rate to the Aitken limit, the overlap decays "
                     + ", ".join(parts))
        g = gaps.get(R)
        if g is not None:
            ends = {lab: _gap_ends(g, lab) for lab in ("E", "B", "BL") if lab in g["n_cert"]}
            starts = {lab: _gap_starts(g, lab, margins.get(lab)) for lab in ends}
            same_b = "B" in ends and ends["B"] == ends["E"] and starts["B"] == starts["E"]
            text += ("; the 1-mHa hand-off window is n_cert = "
                     + _window_text(ends["E"], "rule E (m_E)" if not same_b else
                                    "rule E and rule B in the window model (m_E = m_B)", starts["E"])
                     + ("" if same_b or "B" not in ends else ", " + _window_text(ends["B"], "rule B in the window "
                                                                                          "model (m_B)", starts["B"]))
                     + ", " + _window_text(ends["BL"], "kernel-aware B_L with two runs (m_BL)", starts["BL"]))
            one = (g.get("bl_one_run") or {}).get("hf")
            if one:
                text += f", {_rng([_num(x) for v in one.values() for x in v], '.0f')} for HF with one run (D_L(p))"
            if n_at:
                text += (f", so p_HF = 1e-3 lies {_lies(n_at, ends['E'], 'rule-E', starts['E'])}"
                         + ("" if same_b or "B" not in ends else
                            f", {_lies(n_at, ends['B'], 'rule-B', starts['B'])}")
                         + f" and {_lies(n_at, ends['BL'], 'B_L', starts['BL'])}")
        lines.append(text)
        holds = [h for row in overlap if row["R"] == R for h in row["holdouts"]]
        if holds:
            unc.append(f"R = {R:.1f}: holdout errors up to {max(abs(h['error_last']) for h in holds) * 100:.1f}% "
                       f"(last rate) / {max(abs(h['error_aitken']) for h in holds) * 100:.1f}% (Aitken)")
    u1 = _u1_rows(fits)
    n2 = sorted((float(n.split("_r")[1]), n) for n in u1 if n.startswith("n2_r"))
    if n2:
        parts = []
        for _, n in n2:
            recs = scaling_json.get(n, {}).get("records", {})
            p1 = recs.get("lbfgs_so/second_order", {}).get("overlaps", [None, None])[1]
            p2 = recs.get("r1_so", {}).get("overlaps", [None, None, None])[2]
            parts.append(f"{n.split('_r')[1]} A: {_f(u1[n]['p_hf'], '.3f')} / {_f(p1, '.3f')} / {_f(p2, '.3f')}")
        lines.append("N2 CAS(6,6) at fixed N, p_HF / classical Psi_1 (so) / R1 Psi_2 (so): " + "; ".join(parts))
    ns = sorted({n for r in hf for n in r["ns"]})
    if hf:
        head = ("p_HF falls exponentially with the chain length (" + ", ".join(
            f"alpha = {_f(r['alpha'], '.3f')} per atom at R = {float(r['R']):.1f} bohr" for r in hf)
            + f"); p_HF = 1e-3 is reached only beyond the computed sizes (n <= {max(ns)}), at n = "
            + ", ".join(f"{_rng([x for x in r['n_at_p'] if x is not None], '.0f')} (R = {float(r['R']):.1f})"
                        for r in hf) + " in the model bracket")
        lines.insert(0, head)
    unc = ([f"rates fitted on n = {', '.join(map(str, ns))}; beyond the computed sizes the range [last successive "
            f"rate, Aitken limit] is a model bracket over Delta n of tens to hundreds of sites, not a confidence "
            f"interval"] + unc
           + ["each window statement holds under both gap models (power law and A'/n) or is stated per model; the "
              "B_L window is F40's two-run window, capped where the HF overlap falls below its floor",
              "the optional n = 12 overlap holdout is not implemented",
              "model: per-site overlap decay of HF and the iterates (F22), STO-3G linear chains in canonical orbitals "
              "(a model of the overlap catastrophe, not a target)"])
    return {"q": "Q1", "verdict": "; ".join(lines) + ".", "tables": Q_TABLES["Q1"], "uncertainty": "; ".join(unc) + "."}


def window_by_model(m: dict) -> str:
    """Where the n* of a margin lie against that margin's window per gap model: inside under a model iff one of its
    cases has n* inside the window of that model, Delta_1(n*) >= m ("n_inside_by_model", ``with_inside_by_model``;
    without it, the model's smallest n* against the window's end, ``ft.scaling``'s n_star_min_by_model and
    n_cert_by_model, which is the same test for a closing gap): "inside under both gap models", "outside under both
    gap models" or "gap-model-dependent: inside under ..., outside under ..."; "no window" without the per-model
    fields."""
    counts = m.get("n_inside_by_model")
    if counts is not None:
        inside = {g: counts[g] > 0 for g in ("power", "inverse") if g in counts}
    else:
        mins, ends = m.get("n_star_min_by_model") or {}, m.get("n_cert_by_model") or {}
        inside = {g: mins.get(g) is not None and _num(mins[g]) <= _num(ends[g]) for g in ("power", "inverse")
                  if g in ends}
    if not inside:
        return "no window"
    if all(inside.values()):
        return "inside under both gap models"
    if not any(inside.values()):
        return "outside under both gap models"
    return "gap-model-dependent: " + ", ".join(f"{'inside' if v else 'outside'} under {GAP_LABEL[g]}"
                                               for g, v in inside.items())


def _margin_text(m: dict, label: str) -> str:
    return (f"{_rng(m['n_star'], '.0f')} with {label} ({m['n_none']}/{m['n_cases']} none; inside n_cert = "
            f"{_rng(m['n_cert'], '.0f')} in {m['n_inside']} of {m['n_cases']} cases, each against the window of its "
            f"own gap model; {window_by_model(m)})")


def _s_star(best: dict) -> str:
    return f"{best['k']}{_tag(best.get('flags_k'))} -> {best['k1']}{_tag(best.get('flags_k1'))}"


def _window_summary(brackets: list, margin: str) -> str:
    groups: dict = {}
    for b in brackets:
        m = (b.get("by_margin") or {}).get(margin)
        if m is not None:
            groups.setdefault(window_by_model(m), []).append(f"R = {b['R']:.1f} {b['metric']}")
    return "; ".join(f"{k} on {_join(v)}" for k, v in groups.items()) or "no window"


def verdict_q2(fits: dict, u4s: list, nstar: list) -> dict:
    lines = []
    t1 = []
    for row in fits["U3"]["rows"]:
        for m in ("sf", "so"):
            x = row.get(m)
            if x and "1" in x["t_res1"]:
                t1.append(f"{row['system']} {m} {_f(x['t_res1']['1']['over_qpe'], '.3g')}")
    if t1:
        lines.append("T_res(1)/C_QPE of the first measured iteration at r = 1: " + ", ".join(t1))
    ex = [e for e in fits["U3"]["exponents"] if e["quantity"].startswith("t_res1_over_qpe_sf_r1")]
    if ex:
        lines.append("its local exponent in N (sf, r = 1): " + ", ".join(
            f"R = {e['R']:g}: {', '.join(_f(b, '.2f') for b in e['successive'])}" for e in ex))
    k1 = [r for r in u4s if r["k"] == 1 and float(r["r"]) == 1.0]
    pays_rep = [f"{r['system']} {r['metric']}" for r in k1 if r["pays_rep"]]
    pays_best = [f"{r['system']} {r['metric']} ({_s_star(r['best'])})" for r in k1 if r["best"]["pays"]]
    lines.append(f"at r = 1 the first measured iteration pays under repetition on {_join(pays_rep, 'no system')} "
                 f"and at the best realizable scheme on {_join(pays_best, 'no system')} (of {len(k1)} system-metric "
                 f"rows)")
    for r_val in (3.0,):
        k1r = [r for r in u4s if r["k"] == 1 and float(r["r"]) == r_val]
        if k1r:
            rep_ok = [r["system"] + " " + r["metric"] for r in k1r if r["pays_rep"]]
            best_ok = [f"{r['system']} {r['metric']} ({_s_star(r['best'])})" for r in k1r if r["best"]["pays"]]
            lines.append(f"at r = {r_val:g}: under repetition on {_join(rep_ok, 'no system')}, at the best "
                         f"realizable scheme on {_join(best_ok, 'no system')}")
    rep_n = [b for b in fits["U4"]["n_star"] if float(b["r"]) == 1.0]
    best_n = {(b["R"], b["metric"]): b for b in nstar if b["scheme"] == "best" and float(b["r"]) == 1.0}
    for b in rep_n:
        bb = best_n.get((b["R"], b["metric"]))
        bm = b.get("by_margin") or {}
        if b.get("cases"):
            bm = with_inside_by_model(bm, b["cases"])
        text = f"n* (R = {b['R']:.1f}, {b['metric']}, r = 1) under repetition: "
        if "E" in bm:
            text += _margin_text(bm["E"], "m_E of M3's rule E")
            if "BL" in bm:
                text += ", " + _margin_text(bm["BL"], "m_BL of kernel-aware rule B")
            text += (f", {_rng(b['no_window']['n_star'], '.0f')} without the lengthened QPE (not admitted beyond "
                     f"n_cert)")
        else:
            text += f"{_rng(b['no_window']['n_star'], '.0f')}"
        if bb is not None:
            bbm = bb.get("by_margin") or {}
            text += "; at the best realizable scheme: " + ", ".join(
                _margin_text(bbm[k], label) for k, label in (("E", "m_E"), ("BL", "m_BL")) if k in bbm)
            text += f" [{bb['l_window']['flag']}]"
        lines.append(text)
    best_r1 = [b for b in nstar if b["scheme"] == "best" and float(b["r"]) == 1.0]
    hits = [b for b in best_r1 if b["overall"]["n_star"] is not None]
    n_max = max((r["N"] for r in u4s), default=None)
    head = (f"at N <= {n_max} the first measured iteration pays at the best realizable scheme (r = 1) on "
            f"{_join(sorted(set(pays_best)), 'no system')}")
    if best_r1:
        lo = min((b["overall"]["n_star"][0] for b in hits), default=None)
        head += (f"; extrapolated, it pays at the best realizable scheme from n* = {_f(lo, '.0f')} on {len(hits)} "
                 f"of {len(best_r1)} chain brackets (r = 1); against the window of kernel-aware rule B (m_BL): "
                 f"{_window_summary(hits, 'BL')}; against the window of M3's rule E (m_E): "
                 f"{_window_summary(hits, 'E')}" if hits else
                 "; extrapolated, it pays at the best realizable scheme on no chain bracket (r = 1)")
    lines.insert(0, head)
    ext = [r for r in u4s if r["k"] == 1 and r.get("extended")
           and (r["extended"]["k"], r["extended"]["k1"]) != (r["best"]["k"], r["best"]["k1"])]
    ext_text = _join([f"{r['system']} {r['metric']} r = {float(r['r']):g}: {r['extended']['k']} -> "
                      f"{r['extended']['k1']} {'pays' if r['extended']['pays'] else 'does not pay'}"
                      + ("" if r["extended"]["relative"] is None
                         else f" ({_f(100 * r['extended']['relative'], '.3g')}%)")
                      + (" where s* pays" if r["best"]["pays"] and not r["extended"]["pays"] else
                         " where s* does not" if r["extended"]["pays"] and not r["best"]["pays"] else "")
                      for r in ext], "no first measured iteration changes its optimum")
    unc = ("c_iter(1) = T_res(1)/C_QPE of the first measured iteration fitted as a power law in N (F23 exponents on "
           "N >= 6, F45; the R1 record's k = 1 cost, at N = 10 sf the cheapest iteration at the same Psi_1), with beta "
           "in {last, mean, 9} x crossover to N^2 or N^4 beyond n = 20; the preparation terms C(Psi_1)/C_QPE and "
           "C(Psi_2)/C_QPE as power laws on N >= 6, not lengthened with the QPE (F45's c_k = 0 form reported as a "
           "check); alpha_1 from the classical-step series (n <= 10), alpha_2 from the R1 records; n* over rate "
           "(matched corners) x beta x crossover x window x gap model is a model bracket (validated over Delta n = 2 "
           "by the holdouts, unquantified model risk beyond), the crossed rate corners a sensitivity; the model "
           "covers H chains in canonical orbitals with elementwise compilation and the full allowed residual; "
           "localized or truncated residuals, other topologies and larger bases are outside it; a window statement "
           "on n* holds under both gap models or is stated as gap-model-dependent; p on psi_0 of H; beyond the "
           "computed sizes the "
           "best realizable scheme is min(rep, bbht) [assumption: no certified U or p_lb there], with the "
           "E1lb-referenced forms checked inside their window; the free classical step is the k = 0 row; bbht, "
           "lt_bbht and lt_bbht_L follow BBHT's schedule (growth 6/5), whose expected number of calls exceeds 1/p "
           "above p = 0.30; with the step filter followed by repetition admitted (lt_rep, lt_rep_L; a sensitivity, "
           f"never s*) the first measured iteration's optimum changes on {ext_text}; model: F22, F23, F44, F45 on "
           "STO-3G chains")
    return {"q": "Q2", "verdict": "; ".join(lines) + ".", "tables": Q_TABLES["Q2"], "uncertainty": unc + "."}


def _k_ref(run: dict) -> tuple[int | None, str | None, float | None]:
    """(k_ref, its kind "k_star" or "p90", C_loop(k_ref)/C_QPE at r = 1, None beyond the costed iterates) of a U5 run;
    runs without the k_ref keys count at k*."""
    c = run["costs"]["1"]
    k_star = run["k_star"]
    kind = run.get("k_ref_kind", "k_star" if k_star is not None else None)
    k = run.get("k_ref", k_star)
    v = c.get("c_loop_k_ref_over_qpe", c.get("c_loop_k_star_over_qpe"))
    return k, kind, None if v is None else _num(v)


def u5_ranking(runs: list[dict]) -> dict:
    """The rules of one (system, metric) pair at r = 1 ranked by C_loop(k_ref)/C_QPE (module docstring): k_ref = k*
    where some rule reaches 1.6 mHa, else the first k with p >= 0.9; the kinds are never mixed. A rule whose costs end
    before its k_ref is unresolved (with its C_loop at the last costed iterate); the pair is unresolved if such a rule
    could still be the cheapest. Returns {"kind" (None: no rule reaches either), "costs" {rule: (k_ref, C_loop)},
    "unresolved" {rule: (k_ref, k_last, C_loop(k_last))}, "winners" (the cheapest rules), "resolved"}."""
    refs = {RULE_LABEL.get(r["rule"]["name"], r["rule"]["name"]): (r, *_k_ref(r)) for r in runs}
    kinds = {v[2] for v in refs.values()}
    kind = "k_star" if "k_star" in kinds else "p90" if "p90" in kinds else None
    costs, unresolved = {}, {}
    for label, (run, k, knd, v) in refs.items():
        if kind is None or knd != kind:
            continue
        if v is not None:
            costs[label] = (k, v)
        else:
            loop = [_num(x) for x in run["costs"]["1"]["c_loop_over_qpe"]]
            unresolved[label] = (k, len(loop) - 1, loop[-1])
    low = min((v for _, v in costs.values()), default=math.inf)
    winners = tuple(sorted(k for k, (_, v) in costs.items() if v <= low * (1.0 + 1e-9)))
    resolved = bool(costs) and all(c_last >= low for _, _, c_last in unresolved.values())
    return {"kind": kind, "costs": costs, "unresolved": unresolved, "winners": winners if resolved else (),
            "resolved": resolved}


def step_rules_noisy(cl_json: dict, *, target: float, k_max: int = K_TABLE) -> dict:
    """U7's step rules under noise (oracle r = 1, 3; R1 from the headline oracle cells): per system, r and rule the
    mean C_loop(k)/C_QPE for k = 1..k_max with its 95% bootstrap interval and the share of replicates within
    ``target`` of E_0 at some k <= k_max."""
    out = {}
    for name, data in (cl_json or {}).items():
        c_qpe = float(data["system"]["c_qpe"])
        cells = [c for c in data["configs"] if c["block"] == "step_rules"]
        if not cells:
            continue
        cells += [c for c in data["configs"] if c["block"] == "headline" and c["config"]["rule"] == "unit"
                  and c["config"]["controller"] == "oracle" and float(c["config"]["r"]) in STEP_RULE_R]
        per_r: dict = {}
        for c in cells:
            r = float(c["config"]["r"])
            if c["config"]["controller"] != "oracle" or r not in STEP_RULE_R:
                continue
            reps = c["replicates"]
            km = min(k_max, min(len(rep["c_loop"]) for rep in reps) - 1)
            loop = []
            for k in range(1, km + 1):
                m, lo, hi = closedloop.bootstrap_mean([_num(rep["c_loop"][k]) / c_qpe for rep in reps])
                loop.append({"k": k, "mean": m, "ci": [lo, hi]})
            within = [any(_num(e) <= target for e in rep["e_err"][:km + 1]) for rep in reps]
            label = RULE_LABEL.get(c["config"]["rule"], c["config"]["rule"])
            per_r.setdefault(f"{r:g}", {})[label] = {"c_loop": loop, "within": float(np.mean(within)), "n": len(reps)}
        out[name] = per_r
    return out


def verdict_q3(loop_json: dict, cl_json: dict | None = None) -> dict:
    target = FTParameters().target
    lines, ranked, unres = [], [], []
    for name, data in loop_json.items():
        for metric in ("spin_orbital", "spin_free"):
            runs = [r for r in data["runs"]
                    if r["metric"] == metric and r["r_plan"] in (None, 1.0) and "1" in r["costs"]]
            if not runs:
                continue
            tag = f"{name} {METRIC_LABEL.get(metric, metric)}"
            rk = u5_ranking(runs)
            if rk["kind"] is None:
                ends = [f"{RULE_LABEL.get(r['rule']['name'], r['rule']['name'])} "
                        f"{_f(_num(r['e_err'][-1]) * MHA, '.3g')} mHa" for r in runs]
                lines.append(f"{tag}: no rule reaches 1.6 mHa or p >= 0.9 within {runs[0]['n_steps']} steps (final "
                             f"E - E0: {', '.join(ends)})")
                continue
            kl = "k*" if rk["kind"] == "k_star" else "k_p90"
            text = (f"{tag}: C_loop({kl})/C_QPE at r = 1 " + ", ".join(
                f"{k} {_f(v, '.4g')} ({kl} {kk})" for k, (kk, v) in sorted(rk["costs"].items(), key=lambda x: x[1][1])))
            for label, (kk, k_last, c_last) in sorted(rk["unresolved"].items()):
                text += (f"; {label} unresolved ({kl} = {kk} beyond its costed iterates, C_loop({k_last}) = "
                         f"{_f(c_last, '.3g')})")
            if not rk["resolved"]:
                unres.append(tag)
                text += "; the pair is unresolved"
            else:
                ranked.append((rk["kind"], rk["winners"]))
            lines.append(text)
    rho, rho_over, gate = [], 0, [0, 0]
    for name, data in loop_json.items():
        for r in data["runs"]:
            if r["rule"]["name"] == "unit" and r["metric"] == "spin_orbital" and "1" in r["costs"]:
                pr = next((p for p in r["costs"]["1"]["precision"] if p["k"] == 1), None)
                if pr is not None:
                    rho.append(f"{name} {_f(pr.get('rho_armijo'), '.3g')} / {_f(pr.get('rho_parab3'), '.3g')}")
                    vals = [_fin(pr.get(key)) for key in ("rho_armijo", "rho_parab3")]
                    rho_over += all(v is not None and v > 1.0 for v in vals)
            if r["rule"]["name"] == "hybrid":
                gate[0] += sum(1 for g in r.get("gate_log") or [] if g.get("use"))
                gate[1] += len(r.get("gate_log") or [])
    by_kind = {kind: [w for kd, w in ranked if kd == kind] for kind in ("k_star", "p90")}
    r1 = {kind: sum("R1" in w for w in ws) for kind, ws in by_kind.items()}
    per_rule: dict = {}
    for _, w in ranked:
        for rule in w:
            per_rule[rule] = per_rule.get(rule, 0) + 1
    if ranked:
        head = (f"the unit rule R1 reaches k_ref at the lowest measurement cost on {r1['k_star'] + r1['p90']} of "
                f"{len(ranked)} resolved system-metric pairs ({r1['k_star']} of {len(by_kind['k_star'])} at k*, the "
                f"first iterate within 1.6 mHa; {r1['p90']} of {len(by_kind['p90'])} at k_p90, the first with "
                f"p >= 0.9, where no rule reaches k*); the cheapest rule per pair: "
                + ", ".join(f"{rule} {n}" for rule, n in sorted(per_rule.items(), key=lambda kv: (-kv[1], kv[0])))
                + " (a tie counts for each rule)"
                + (f"; unresolved: {_join(unres)} (a rule's costs end before its k_ref below the cheapest C_loop)"
                   if unres else ""))
    else:
        head = "no rule reaches 1.6 mHa or p >= 0.9 at a resolved cost on the U5 systems run"
    lines.insert(0, head)
    if rho:
        lines.insert(1, f"energy-based rules lose at k = 1 wherever rho_E = T_E/T_res,direct > 1 ({rho_over} of "
                        f"{len(rho)} systems)")
        lines.append("energies cost more than the residual they would replace (rho_E = T_E/T_res,direct of the Armijo "
                     "pair / the 3-point fit at k = 1, so): " + ", ".join(rho))
    if gate[1]:
        lines.append(f"R3's planned gate chose the 3-point fit in {gate[0]} of {gate[1]} iterations")
    noisy = step_rules_noisy(cl_json, target=target)
    if noisy:
        parts = []
        for name, per_r in noisy.items():
            for r, rules in per_r.items():
                cheapest = min(rules, key=lambda lab: rules[lab]["c_loop"][-1]["mean"])
                cells = "; ".join(
                    f"{lab} " + " / ".join(_f(x["mean"], ".3g") for x in v["c_loop"])
                    + f" [{_f(v['c_loop'][-1]['ci'][0], '.3g')}, {_f(v['c_loop'][-1]['ci'][1], '.3g')}], "
                    f"{100 * v['within']:.0f}% within 1.6 mHa" for lab, v in rules.items())
                parts.append(f"{name} oracle r = {r}: {cells} (cheapest C_loop(4): {cheapest})")
        lines.append("under noise (U7: closed-loop step rules at oracle r = 1 and 3, R1 from the headline cells; mean "
                     "C_loop(k)/C_QPE for k = 1..4 with the 95% interval at k = 4, and the share of replicates within "
                     "1.6 mHa by k = 4): " + "; ".join(parts))
    else:
        lines.append("under noise: no step-rule block in the closed-loop outputs (U7 not computed)")
    unc = ("U5 is exact emulation (noise-free residuals; energies costed, not drawn); k_ref = k*, the first iterate "
           "within 1.6 mHa, or where no rule reaches it the first k with p >= 0.9, never mixed in one ranking; costs "
           "at r = 1 (U5 lists r = 0.3 and 3); the F29 caps are part of R1; U7's step rules under noise run the oracle "
           "controller (not realizable) with independent Gaussian residual noise, means with 95% percentile-bootstrap "
           "intervals; model: F26-F31, F32-F38")
    return {"q": "Q3", "verdict": "; ".join(lines) + ".", "tables": Q_TABLES["Q3"], "uncertainty": unc + "."}


def _first_iteration(cfg: dict) -> dict | None:
    """The first measured iteration (k = 1 -> 2) of a closed-loop cell: its verdict and T(2) at s*."""
    v1 = next((v for v in cfg["verdicts"] if v["k"] == 1), None)
    st2 = next((s for s in cfg["stats"] if s["k"] == 2), None)
    t2 = None if st2 is None or st2["best"] is None else st2["T"][st2["best"]]["mean"]
    return None if v1 is None else {"v": v1, "t2": t2, "st2": st2}


def _saving_text(v1: dict) -> str:
    """The saving of a measured iteration at s* with the flags of s*, and at the extended optima where they differ."""
    best = v1["best"]
    if best is None:
        return "no realizable scheme at Psi_1 or Psi~_2"
    text = (f"saves {_ci(best['mean'], best['ci'], '.3g')} C_QPE at the best realizable scheme ({v1['best_k']}"
            f"{_tag(v1.get('flags_best_k'))} -> {v1['best_k1']}{_tag(v1.get('flags_best_k1'))}: "
            f"{'pays' if v1['pays'] else 'does not pay'})")
    ext = v1.get("extended")
    if ext is not None and (ext["best_k"], ext["best_k1"]) != (v1["best_k"], v1["best_k1"]):
        text += (f"; with lt_rep and lt_rep_L admitted (sensitivity) {ext['best_k']} -> {ext['best_k1']} saves "
                 f"{_ci(ext['mean'], ext['ci'], '.3g')} ({'pays' if ext['pays'] else 'does not pay'})")
    return text


def _cell_text(name: str, cfg: dict, exact_p2: float | None = None) -> str:
    """The first measured iteration of a deciding cell: under repetition and at s*, p~_2, r_1, the second iteration."""
    fi = _first_iteration(cfg)
    if fi is None:
        return f"{name} ({cfg['label']}): no measured iteration"
    v1 = fi["v"]
    rep = v1["by_scheme"].get("rep_rule_e")
    text = (f"{name} ({cfg['label']}, {cfg['n']} replicates): Psi_1 -> Psi~_2 "
            + (f"saves {_ci(rep['mean'], rep['ci'], '.3g')} C_QPE under repetition "
               f"({'pays' if rep['pays'] else 'does not pay'}); " if rep else "")
            + _saving_text(v1))
    st2 = fi["st2"]
    if st2 is not None:
        text += (f"; median p~_2 = {_f(st2['p_median'], '.3f')} [{_f(st2['p_band'][0], '.3f')}, "
                 f"{_f(st2['p_band'][1], '.3f')}]")
        if exact_p2 is not None:
            text += f" against the exact R1 p_2 = {_f(exact_p2, '.3f')}"
    r1 = cfg["r_median"][1] if len(cfg["r_median"]) > 1 else None
    if r1 is not None:
        text += f"; realized r_1 = {_f(r1, '.2f')}"
    v2 = next((v for v in cfg["verdicts"] if v["k"] == 2), None)
    if v2 is not None and v2["best"] is not None:
        text += (f"; the second measured iteration {'pays' if v2['pays'] else 'does not pay'} at s* "
                 f"({_ci(v2['best']['mean'], v2['best']['ci'], '.3g')})")
    return text


def deciding_cells(cl_s: dict) -> dict:
    """{system: (block, cell)} of the deciding cells: the headline block's first, else a control block's."""
    out = {}
    for name, blk in cl_s.items():
        cells = [c for c in blk["configs"] if c["deciding"]]
        cells.sort(key=lambda c: c["block"] != "headline")
        if cells:
            out[name] = (cells[0]["block"], cells[0])
    return out


def verdict_q4(cl_s: dict, loop_json: dict, scaling_json: dict) -> dict:
    lines, reversed_on, below_on = [], [], []
    dec = deciding_cells(cl_s)
    head_sys = {n: c for n, (b, c) in dec.items() if b == "headline"}
    ctrl_sys = {n: (b, c) for n, (b, c) in dec.items() if b != "headline"}
    for name, cfg in head_sys.items():
        blk = cl_s[name]
        exact_p2 = None
        rec = scaling_json.get(name, {}).get("records", {}).get("r1_so")
        if rec is not None and len(rec["overlaps"]) > 2:
            exact_p2 = _num(rec["overlaps"][2])
        else:
            run = next((r for r in loop_json.get(name, {}).get("runs", []) if r["rule"]["name"] == "unit"
                        and r["metric"] == "spin_orbital"), None)
            if run is not None and len(run["p"]) > 2:
                exact_p2 = _num(run["p"][2])
        text = _cell_text(name, cfg, exact_p2)
        fi_dec = _first_iteration(cfg)
        # the lagged sensitivity cells of the headline block (r = 0.5, 2) and the deciding cell
        lagged = [c for c in blk["configs"] if c["block"] == "headline" and c["config"]["controller"] == "lagged"
                  and c["config"]["rule"] == "unit" and "superseded_by_angle_cap_rerun" not in c["flags"]]
        cells, t2_lag, reverses = [], [], False
        for c in sorted(lagged, key=lambda c: float(c["config"]["r"])):
            fi = _first_iteration(c)
            if fi is None:
                continue
            v1, r1 = fi["v"], c["r_median"][1] if len(c["r_median"]) > 1 else None
            if fi["t2"] is not None:
                t2_lag.append((float(c["config"]["r"]), fi["t2"]))
            best = v1["best"]
            cells.append(f"lagged r = {float(c['config']['r']):g}{' (decides)' if c['deciding'] else ''}: realized "
                         f"r_1 = {_f(r1, '.2f')}, mean T(2) at s* = {_f(fi['t2'], '.4g')}, saving at s* "
                         + ("-" if best is None else f"{_ci(best['mean'], best['ci'], '.3g')} "
                                                     f"({'pays' if v1['pays'] else 'does not pay'})")
                         + (f", non-realizable: {', '.join(c['non_realizable'])}" if c["non_realizable"] else ""))
            if not c["deciding"] and fi_dec is not None and v1["pays"] != fi_dec["v"]["pays"]:
                reversed_on.append(f"{name} (lagged r = {float(c['config']['r']):g})")
                reverses = True
        if cells:
            text += "; lagged r-scan (sensitivity): " + "; ".join(cells)
            if reverses:
                text += "; the sensitivity reverses the verdict"
        orc = [c for c in blk["configs"] if c["config"]["controller"] == "oracle" and c["block"] == "headline"]
        if orc:
            def t2(c, key):
                st = next((s for s in c["stats"] if s["k"] == 2), None)
                return None if st is None or key not in st["T"] else st["T"][key]["mean"]
            best_r = min(orc, key=lambda c: t2(c, "rep_rule_e") if t2(c, "rep_rule_e") is not None else math.inf)
            star = [(float(c["config"]["r"]), _first_iteration(c)) for c in orc]
            star = [(r, fi["t2"]) for r, fi in star if fi is not None and fi["t2"] is not None]
            text += (f"; oracle r-scan (optimistic over r <= 3): mean T(2) least at r = {best_r['config']['r']:g} "
                     f"under repetition" + (f" and at r = {min(star, key=lambda x: x[1])[0]:g} at s* (T(2) = "
                                            f"{_f(min(t for _, t in star), '.4g')})" if star else ""))
            if star and t2_lag and min(t for _, t in t2_lag) < min(t for _, t in star):
                r_low, t_low = min(t2_lag, key=lambda x: x[1])
                text += f"; a lagged cell lies below it (lagged r = {r_low:g}: T(2) = {_f(t_low, '.4g')})"
                below_on.append(name)
        lines.append(text)
    for name, (block, cfg) in ctrl_sys.items():
        lines.append(f"{CONTROL_LABEL.get(block, block)}: " + _cell_text(name, cfg))
    first = {n: _first_iteration(c) for n, c in head_sys.items()}
    rep_pays = sorted(n for n, f in first.items() if f and f["v"]["by_scheme"].get("rep_rule_e", {}).get("pays"))
    best_pays = sorted(n for n, f in first.items() if f and f["v"]["pays"])
    second = {n: next((v for v in c["verdicts"] if v["k"] == 2), None) for n, c in head_sys.items()}
    second_pays = sorted(n for n, v in second.items() if v and v["pays"])
    heads = []
    if first:
        heads.append(f"in the deciding cells (lagged r = 1 of the headline block) the first measured iteration pays "
                     f"under repetition on {_join(rep_pays, 'no system')} and at the best realizable scheme on "
                     f"{_join(best_pays, 'no system')} (of {len(first)}); the second pays at s* on "
                     f"{_join(second_pays, 'no system')}; the lagged sensitivity (r = 0.5, 2) reverses the verdict on "
                     f"{_join(reversed_on, 'no system')}")
    if ctrl_sys:
        cf = {n: _first_iteration(c) for n, (_, c) in ctrl_sys.items()}
        rep_pays = sorted(n for n, f in cf.items() if f and f["v"]["by_scheme"].get("rep_rule_e", {}).get("pays"))
        heads.append(f"in the control cells (lagged r = 1 of the negative and N = 8 controls) it pays under repetition "
                     f"on {_join(rep_pays, 'no system')} "
                     f"and at the best realizable scheme on "
                     f"{_join(sorted(n for n, f in cf.items() if f and f['v']['pays']), 'no system')} (of {len(cf)})")
    lines = heads + lines
    unc = ("replicate statistics (95% percentile bootstrap of the mean paired difference, 2000 resamples); the "
           "lagged r = 1 cell decides (the oracle controller is not realizable); the lagged r = 0.5 and 2 cells are "
           "the sensitivity and the oracle r-scan over r <= 3 the optimistic bound, which a lagged cell can undercut "
           "since the lagged plan's realized r_1 differs from its r (F35); exact propagation (the compiled ablation is "
           "U7's); p~ on psi_0 of H; s* per configuration and iterate by the replicate means, with its flags; "
           "independent Gaussian residual noise (element correlations not modelled); model: F32-F38, F43")
    return {"q": "Q4", "verdict": "; ".join(lines) + "." if lines else "no deciding cell in the closed-loop outputs.",
            "tables": Q_TABLES["Q4"], "uncertainty": unc + "."}


def verdict_q5(handoff_json: dict, cl_s: dict, cisdt: dict | None = None) -> dict:
    lines, pays_r1, pays_r3 = [], [], []
    worst = (0.0, None)
    worst_c = (0.0, None)
    for name, data in handoff_json.items():
        hf = next((u for u in data["u8"] if u["state"] == "HF" and not u["compiled"]), None)
        if hf is not None:
            nv_e = hf.get("n_v_expected")
            lines.append(f"{name} HF: {hf['form']} with {hf['u_label'] or 'no U'}"
                         + (f" (margin {_f(hf['margin'] * MHA, '.3g')} mHa)" if hf["margin"] is not None else "")
                         + f", D = {_f(hf['distance'], '.3g')}, n_v = {hf['n_v']}"
                         + (f" (E {_f(nv_e, '.3g')})" if nv_e is not None and abs(_num(nv_e) - hf["n_v"]) > 1e-12
                            else "")
                         + f", T = {_f(hf['cost_over_qpe'], '.4g')} C_QPE (p on psi_0' of H'; oracle T_E "
                         f"{_f(hf['cost_oracle_e_over_qpe'], '.4g')})")
        for u in data["u8"]:
            tot = u["laws"]["sine_2n"]["n_v_runs"]["eps_id"]
            target = worst_c if u["compiled"] else worst
            if tot is not None and tot > target[0]:
                if u["compiled"]:
                    worst_c = (tot, f"{name} {u['state']}")
                else:
                    worst = (tot, f"{name} {u['state']} ({u['record']})")
        for rr in ("1.0", "3.0"):
            rws = {r["k"]: r for r in data["u9"].get(f"r1_so@r={rr}", [])}
            if 1 not in rws or 2 not in rws:
                continue
            a = {k: _fin(v) for k, v in rws[1]["costs"].items()}
            b = {k: _fin(v) for k, v in rws[2]["costs"].items()}
            ver = amplify.iteration_verdict(a, b)
            c1 = rws[1]["costs_over_qpe"].get(ver["best_k"]) if ver["best_k"] else None
            rel = None if ver["relative_best"] is None else 100 * ver["relative_best"]
            if ver["pays_best"]:
                (pays_r1 if rr == "1.0" else pays_r3).append(name)
            fl1 = (rws[1].get("scheme_flags") or {}).get(ver["best_k"], []) if ver["best_k"] else []
            fl2 = (rws[2].get("scheme_flags") or {}).get(ver["best_k1"], []) if ver["best_k1"] else []
            text = (f"{name} R1 so r = {float(rr):g}: s* = {ver['best_k']}{_tag(fl1)} at Psi_1 (T = {_f(c1, '.4g')}), "
                    f"{ver['best_k1']}{_tag(fl2)} at Psi_2; the first measured iteration "
                    f"{'pays' if ver['pays_best'] else 'does not pay'} at s* ({_f(rel, '.3g')}%)")
            ext = amplify.extended_verdict(a, b)
            if (ext["best_k"], ext["best_k1"]) != (ver["best_k"], ver["best_k1"]):
                text += (f"; with lt_rep and lt_rep_L admitted (sensitivity): {ext['best_k']} -> {ext['best_k1']} ("
                         f"{_f(None if ext['relative_best'] is None else 100 * ext['relative_best'], '.3g')}%)")
            lines.append(text)
        t6 = data.get("m3_t6")
        if t6:
            for fam, loops in t6["families"].items():
                for lp, v in loops.items():
                    rb, mc = v["rule_b"], v["m3_certified"]
                    if lp == "free" or rb is None:
                        continue
                    lines.append(f"{name} M3 T6 {fam}/{lp} (p on psi_0 of H, as M3): rule B k = {rb['k']} at "
                                 f"{_f(rb['cost_over_qpe'], '.4g')} C_QPE against M3's certified "
                                 + ("none" if mc is None else f"k = {mc['k']} at {_f(mc['cost_over_qpe'], '.4g')}"))
    forms = {}
    for name, data in handoff_json.items():
        hf = next((u for u in data["u8"] if u["state"] == "HF" and not u["compiled"]), None)
        if hf is not None:
            forms.setdefault(f"{hf['form']} (n_v = {hf['n_v']})", []).append(name)
    head = ("rule B accepts HF with " + "; ".join(f"{k} on {_join(sorted(v))}" for k, v in sorted(forms.items()))
            + f", every exact state within an identification error of {_f(worst[0], '.1e')} (kernel and logical "
            f"terms); the first measured iteration pays at the best realizable scheme on "
            f"{_join(pays_r1, 'no system')} at r = 1 and on {_join(pays_r3, 'no system')} at r = 3")
    lines.insert(0, head)
    lines.append(f"largest identification error of rule B's choice (sine window, n_v runs, logical term included): "
                 f"{_f(worst[0], '.2e')} ({worst[1]}) on exact states"
                 + (f", {_f(worst_c[0], '.2e')} ({worst_c[1]}) on compiled states" if worst_c[1] else ""))
    for name, (block, cfg) in deciding_cells(cl_s).items():
        st = {s["k"]: s for s in cfg["stats"]}
        if 1 not in st or 2 not in st:
            continue
        b1, b2 = st[1]["best"], st[2]["best"]
        where = "closed loop" if block == "headline" else f"closed loop, {CONTROL_LABEL.get(block, block)}"
        if b1 is None or b2 is None:
            lines.append(f"{name} {where} ({cfg['label']}): no realizable scheme at Psi_1 or Psi~_2")
            continue
        lines.append(f"{name} {where} ({cfg['label']}): s* = {b1}{_tag(st[1].get('flags_best'))} at Psi_1 "
                     f"({_ci(st[1]['T'][b1]['mean'], st[1]['T'][b1]['ci'], '.4g')}), "
                     f"{b2}{_tag(st[2].get('flags_best'))} at Psi~_2 "
                     f"({_ci(st[2]['T'][b2]['mean'], st[2]['T'][b2]['ci'], '.4g')}) C_QPE")
    unc = ("E1lb is assumed (E_1' - 0.1 mHa from the model spectrum of H'); U = CISD(H') or dedicated energies "
           "(z = 3); identification errors are spectrum-weighted for the listed states, the soundness statement is the "
           "F40 bound at p_min = 1e-3 (kernel and logical terms budgeted separately, total <= 2e-3); costs in the "
           "window model (w = 1), U8 and U9 with p on psi_0' of H', M3's T6 rows with p on psi_0 of H as in M3; filter "
           "degrees from the constructed c_f table with F43's GQSP rotations; every scheme without a certified "
           "termination carries its flag; bbht, lt_bbht and lt_bbht_L follow BBHT's schedule (growth 6/5), whose "
           "expected number of calls exceeds 1/p above p = 0.30, so the step filter followed by repetition (lt_rep, "
           "lt_rep_L) is reported as a sensitivity, never s*; oracle schemes (aa, lt) reported, never decisive; model: "
           "F39-F44")
    if cisdt:
        unc += "; " + cisdt_text(cisdt)
    return {"q": "Q5", "verdict": "; ".join(lines) + ".", "tables": Q_TABLES["Q5"], "uncertainty": unc + "."}


def cisdt_text(cisdt: dict) -> str:
    """Q5's sensitivity to F42's optional CISDT(H') certificate (``cisdt_sensitivity``)."""
    gain = sorted(n for n, c in cisdt.items() if c["certified"] and not c["certified_cisd"])
    forms = [f"{n} {c['form_cisd']} ({c['n_v_cisd']}) -> {c['form']} ({c['n_v']})" for n, c in sorted(cisdt.items())
             if (c["form"], c["n_v"]) != (c["form_cisd"], c["n_v_cisd"])]
    hf, pays = [], {}
    for n, c in sorted(cisdt.items()):
        v = c["by_r"].get("1.0")
        if v is not None and (v["best"][0] != v["best_cisd"][0]
                              or abs(v["t_best"][0] - v["t_best_cisd"][0]) > 1e-9 * v["t_best_cisd"][0]):
            hf.append(f"{n} {v['best_cisd'][0]} {_f(v['t_best_cisd'][0], '.4g')} -> {v['best'][0]} "
                      f"{_f(v['t_best'][0], '.4g')}")
        for r, w in c["by_r"].items():
            if w["pays"]:
                pays.setdefault(r, []).append(n)
    text = ("with F42's optional CISDT(H') in place of CISD(H') (reported, never the verdict of record; dedicated "
            f"certificates not re-evaluated) U would be certified also on {_join(gain, 'no further system')}")
    if forms:
        text += f", rule B's form at HF changes on {_join(forms)}"
    if hf:
        text += f", s* and T at HF (r = 1 loop) become {_join(hf)}"
    text += (f", and the first measured iteration would pay at s* on {_join(sorted(pays.get('1.0', [])), 'no system')}"
             f" at r = 1 and on {_join(sorted(pays.get('3.0', [])), 'no system')} at r = 3")
    return text


def best_with_layout(rows, *, loop: str, params: FTParameters = FTParameters()):
    """The cheapest selectable row (AE, ``ft.properties.best_at_budget``) whose longest coherent run has a layout
    (``ft.properties.run_layout``); None if none has."""
    pool = list(rows)
    while True:
        b = P.best_at_budget(pool, loop=loop)
        if b is None or P.run_layout(b, params=params) is not None:
            return b
        pool = [r for r in pool if r is not b]


def q6_layouts(pairs: list, loops, params: FTParameters = FTParameters()) -> dict:
    """The best rows (AE) of the (system, target) pairs without a layout of their longest coherent run, and the tally
    restricted to rows with a layout (F53, F18)."""
    none, tally = [], {lp: {} for lp in loops}
    n_best = 0
    for p in pairs:
        for lp in loops:
            b = P.best_at_budget(p["rows"], loop=lp)
            if b is not None:
                n_best += 1
                if P.run_layout(b, params=params) is None:
                    none.append({"system": p["system"], "target": p["target"], "loop": lp, "route": b.route,
                                 "toffolis": b.longest_run, "logical_qubits": b.logical_qubits})
            w = b if b is None or P.run_layout(b, params=params) is not None else best_with_layout(
                p["rows"], loop=lp, params=params)
            key = "none" if w is None else w.route
            tally[lp][key] = tally[lp].get(key, 0) + 1
    return {"best_rows": n_best, "no_layout": none, "tally": tally}


def _share(tally: dict) -> str:
    return ", ".join(f"{route} {n}" for route, n in sorted(tally.items(), key=lambda kv: (-kv[1], kv[0])))


def q6_loops(properties_json: dict) -> tuple[str, ...]:
    """The loops of Q6 present in every pair's best rows (free, measured r = 1 and r = 0.3)."""
    return tuple(lp for lp in Q6_LOOPS if all(lp in blk["best"][str(eps)] for d in properties_json.values()
                                             for blk in d["targets"].values() for eps in blk["accuracies"]))


def verdict_q6(properties_json: dict, deep: dict, params: FTParameters = FTParameters()) -> dict:
    """Q6 (module docstring): the record tally (route A at its cheapest usable budget per pair, F53) at both
    accuracies, route A at each pre-declared budget and on the compiled iterates (``ft.properties.q6_comparison``,
    ``q6_statements``), the layouts of the best rows, the per-pair cells and the deep runs."""
    loops = q6_loops(properties_json)

    def cell(b) -> str:
        return "-" if b is None else f"{b['route']} {_f(b['toffolis_over_qpe'], '.3g')}"

    tight = P.property_pairs(properties_json)
    loose = P.property_pairs(properties_json, accuracy="loose")
    tal = {"tight": P.route_tally([p["rows"] for p in tight], loops=loops, budgets=Q6_BUDGETS),
           "loose": P.route_tally([p["rows"] for p in loose], loops=loops, budgets=Q6_BUDGETS)}
    changes = []
    loose_of = {(p["system"], p["target"]): p for p in loose}
    for a in tight:
        b = loose_of.get((a["system"], a["target"]))
        if b is None:
            continue
        for lp in loops:
            ra, rb = P.best_at_budget(a["rows"], loop=lp), P.best_at_budget(b["rows"], loop=lp)
            na, nb = "none" if ra is None else ra.route, "none" if rb is None else rb.route
            if na != nb:
                changes.append(f"{a['system']} {a['target']} {lp} ({na} at {a['eps']:g}, {nb} at {b['eps']:g})")
    cmp = P.q6_comparison(properties_json, loops=loops, budgets=Q6_BUDGETS)
    st = P.q6_statements(cmp)
    lay = q6_layouts(tight, loops, params)
    lines, limited = [], []
    for name, data in properties_json.items():
        parts = []
        for kind, blk in data["targets"].items():
            eps_t, eps_l = blk["accuracies"][-1], blk["accuracies"][0]
            bt, bl = blk["best"][str(eps_t)], blk["best"][str(eps_l)]
            fl = blk["floors"]["r1_so"]["min"]
            fs = (blk["floors"].get("compiled_sliced") or {}).get("min")
            if fl > eps_t:
                limited.append(f"{name} {kind}")
            parts.append(f"{kind} at {eps_t:g} / {eps_l:g}: free {cell(bt['free']['ae'])} / "
                         f"{cell(bl['free']['ae'])}, r = 1 loop {cell(bt['meas_r1']['ae'])} / "
                         f"{cell(bl['meas_r1']['ae'])} (R1 floor {_f(fl, '.1e')}"
                         + (f", sliced {_f(fs, '.1e')}" if fs is not None else "")
                         + f"{', bias-limited' if fl > eps_t else ''})")
        lines.append(f"{name}: " + "; ".join(parts))
    n_cases = tal["tight"]["pairs"]
    if n_cases:
        def shares(t) -> str:
            return "; ".join(f"with {LOOP_LABEL.get(lp, lp)}: {_share(t['best'][lp])}" for lp in loops)
        head = [f"at the tighter accuracy of each (system, target) pair ({n_cases} pairs), with route A at its "
                f"cheapest usable budget per pair (the record, F53; this choice of k uses the exact bias, F49), the "
                f"cheapest route is, {shares(tal['tight'])}; at the looser accuracy, {shares(tal['loose'])}; the "
                f"cheapest route changes between the accuracies on {len(changes)} pair-loop cells"
                + (f": {_join(changes)}" if changes else "")]
        head += st["verdict"]
        head.append(f"route A is bias-limited (exact R1 floor above eps) on {_join(limited, 'no pair')}")
        nl = lay["no_layout"]
        head.append(f"{len(nl)} of the {lay['best_rows']} best rows (AE, tighter accuracy) have no surface-code layout "
                    f"of their longest coherent run at p_fail_ae (no factory fails below 3.6e-16 per Toffoli, so no "
                    f"single run above about 2.8e11 Toffolis has one)"
                    + (": " + _join(f"{x['system']} {x['target']} {x['loop']} {x['route']} ({_f(x['toffolis'], '.3g')} "
                                    f"Toffolis on {x['logical_qubits']} qubits)" for x in nl) if nl else "")
                    + "; restricted to rows with a layout the cheapest route is, "
                    + "; ".join(f"with {LOOP_LABEL.get(lp, lp)}: {_share(lay['tally'][lp])}" for lp in loops))
        lines = head + lines
    for name, d in deep.items():
        cells = []
        for kind, t in d["targets"].items():
            eps = t["accuracies"][-1]
            for r, v in t["by_r"].items():
                cells.append(f"{kind} r = {r}: floor {_f(v['min_median'], '.1e')} "
                             f"(eps {eps:g} reached in {100 * v['k_eps'][str(eps)]['reached']:.0f}%)")
        lines.append(f"{name} closed-loop deep runs (median over {d['n']} replicates): " + "; ".join(cells))
    unc = ("route A's usability at a fixed k is an emulation fact (no certificate of its bias, F49): the record "
           "tally picks k per pair with the exact bias, the per-budget tallies fix k for every pair; the A-oracle row "
           "is reported only; AE at kappa = 1 is the reference estimator; route B with the exact model bias of H' and "
           "the property-aware encoding of F52 where F9's is too biased; " + "; ".join(st["uncertainty"])
           + "; the layouts per run at the budgets of F18 (shots and per-element AE runs parallelize over machines, "
           "F19, a single coherent run does not); model: F46-F53")
    return {"q": "Q6", "verdict": "; ".join(lines) + ".", "tables": Q_TABLES["Q6"], "uncertainty": unc + ".",
            "route_comparison": scaling.to_json(cmp), "tallies": scaling.to_json(tal), "accuracy_changes": changes,
            "layouts": scaling.to_json(lay), "loops": list(loops)}


def assemble_u0(inputs: dict, derived: dict, missing: dict, out_label: str) -> list[dict]:
    """The six U0 rows (module docstring). ``inputs``: {"fits", "scaling", "loop", "closedloop", "handoff",
    "properties"} as read; ``derived``: {"u4", "cl_s", "deep", "cisdt"}; ``missing``: :func:`missing_inputs`. A
    question whose first input family is absent is "not computed: no <prefix>_<s>.json in <out>"; a missing system of
    a family it reads is "not computed for <s>"."""
    fits, sj, lj = inputs.get("fits"), inputs.get("scaling") or {}, inputs.get("loop") or {}
    cj, hj, pj = inputs.get("closedloop") or {}, inputs.get("handoff") or {}, inputs.get("properties") or {}
    u4, cl_s = derived.get("u4") or {"rows": [], "n_star": []}, derived.get("cl_s") or {}
    present = {"fits": fits is not None, "scaling": bool(sj), "loop": bool(lj), "closedloop": bool(cj),
               "handoff": bool(hj), "properties": bool(pj)}
    build = {"Q1": lambda: verdict_q1(fits, sj), "Q2": lambda: verdict_q2(fits, u4["rows"], u4["n_star"]),
             "Q3": lambda: verdict_q3(lj, cj), "Q4": lambda: verdict_q4(cl_s, lj, sj),
             "Q5": lambda: verdict_q5(hj, cl_s, derived.get("cisdt")),
             "Q6": lambda: verdict_q6(pj, derived.get("deep") or {})}
    rows = []
    for q, fams in Q_FAMILIES.items():
        first = fams[0]
        if not present[first]:
            what = "scaling_fits.json" if first == "fits" else f"{first}_<s>.json"
            rows.append({"q": q, "verdict": f"not computed: no {what} in {out_label}.", "tables": Q_TABLES[q],
                         "uncertainty": "-"})
            continue
        row = build[q]()
        notes = []
        for f in fams:
            miss = [s for s in missing.get(f, []) if f != "fits"]
            if not present[f]:
                notes.append(f"the part from {f}_<s>.json not computed: no {f}_<s>.json in {out_label}")
            elif miss:
                notes.append(f"not computed for {_join(miss)}: no {', '.join(_file(f, s) for s in miss)} in "
                             f"{out_label}")
        if notes:
            row["verdict"] = row["verdict"].rstrip(".") + "; " + "; ".join(notes) + "."
        rows.append(row)
    return rows


def exit_status(checks: list[dict], *, allow_partial: bool = False) -> int:
    """1 if an integration check fails (with ``allow_partial`` the check "inputs complete" alone does not), else 0."""
    failed = [c["name"] for c in checks if not c["passed"] and not (allow_partial and c["name"] == INPUTS_CHECK)]
    return 1 if failed else 0


# ------------------------------------------------------------------ markdown
def markdown(res: dict, command: str, script_s: float) -> str:
    failed = res.get("checks_failed", [])
    partial = res.get("allow_partial", False)
    lines = ["# M4 summary: U0 verdicts, U4 scheme columns, U9 part 2, U11 closed-loop columns, U12", ""]
    if failed:
        lines += ["FAILED integration checks: " + "; ".join(
            n + (" (permitted by --allow-partial)" if partial and n == INPUTS_CHECK else "") for n in failed), ""]
    else:
        lines += [f"Integration checks: all {len(res['checks'])} pass.", ""]
    inp = res["inputs"]
    read = [f"scaling_<s>.json ({len(inp['scaling'])} systems)" + (" and scaling_fits.json" if inp["fits"] else "")]
    read += [f"{f}_<s>.json ({len(inp[f])})" for f in FAMILIES[1:]]
    miss = [_file(f, s) for f in ("fits",) + FAMILIES for s in (inp.get("missing") or {}).get(f, [])]
    lines += [f"Generated by `{command}` ({script_s:.0f} s, one thread) from {', '.join(read)} in "
              f"{res.get('out', 'the output directory')}" + (f" (missing: {', '.join(miss)})" if miss else "")
              + "; the deep runs computed here. Costs in units of the system's selected C_QPE; dE = 1 mHa; the "
              "verdicts use the realizable schemes and the realizable (lagged) controller only.", ""]
    lines += ["## U0. Verdicts", ""]
    lines += _table(["Q", "verdict", "tables", "uncertainty and model dependence"],
                    [[v["q"], v["verdict"], v["tables"], v["uncertainty"]] for v in res["U0"]])
    q6 = next((v for v in res["U0"] if v["q"] == "Q6" and "tallies" in v), None)
    if q6 is not None:
        rows = []
        for acc_key, label in (("tight", "tighter"), ("loose", "looser")):
            t = q6["tallies"][acc_key]
            for lp in q6["loops"]:
                rows.append([label, lp] + [_share(t["fixed_k"][lp][str(k)]) for k in Q6_BUDGETS]
                            + [_share(t["best"][lp])])
        comp = q6["route_comparison"].get("compiled")
        if comp is not None:
            for lp in q6["loops"]:
                if lp in comp["best"]:
                    rows.append(["tighter, compiled iterates", lp] + [_share(comp["fixed_k"][lp][str(k)])
                                                                      for k in Q6_BUDGETS] + [_share(comp["best"][lp])])
        lines += ["## U0 Q6: the cheapest route with route A at each pre-declared budget", "",
                  f"Over the {q6['tallies']['tight']['pairs']} (system, target) pairs, AE at kappa = 1: routes of the "
                  "best rows with route A at one budget k for every pair, and at its cheapest usable budget per pair "
                  "(the record, F53; this choice of k uses the exact bias, F49); 'none': no selectable row.", ""]
        lines += _table(["accuracy", "loop"] + [f"k = {k}" for k in Q6_BUDGETS] + ["cheapest usable k (record)"], rows)
    noisy = res.get("U7_step_rules") or {}
    if noisy:
        rows = []
        for name, per_r in noisy.items():
            for r, rules in per_r.items():
                for lab, v in rules.items():
                    rows.append([name, r, lab] + [_ci(x["mean"], x["ci"], ".3g") for x in v["c_loop"]]
                                + [f"{100 * v['within']:.0f}%", v["n"]])
        lines += ["## U0 Q3: step rules under noise (U7 cells, oracle controller)", "",
                  "Mean C_loop(k)/C_QPE [95% bootstrap CI] for k = 1..4 and the share of replicates within 1.6 mHa of "
                  "E_0 by k = 4; R1 from the headline oracle cells.", ""]
        lines += _table(["system", "r", "rule"] + [f"C_loop({k})" for k in range(1, K_TABLE + 1)]
                        + ["within 1.6 mHa", "replicates"], rows)
    sides = res["qpe_sides"]
    rows = []
    for name, s in sides.items():
        si = s["scheme_inputs"]
        rows.append([name, s["norb"], s["kind"], _f((s["e1lb"] - s["e0_model"]) * MHA, ".2f"),
                     _f(None if s["any_spin_gap"] is None else s["any_spin_gap"] * MHA, ".2f"),
                     _f(s["any_spin_s2"], ".2g"), _f((s["u_cisd"] - s["e0_model"]) * MHA, ".2f"),
                     "yes" if s["u_certified"] else "no", f"{s['form']} ({s['n_v']})",
                     "yes" if s["b_l_terminates"] else "no", s["u_final"] or "-",
                     _f(si["c_filt"], ".3g"), _f(si["c_coarse"], ".3g"), _f(si["c_filt_l"], ".3g"),
                     _f(si["c_coarse_l"], ".3g"), "-" if s["filters"]["w_l"] is None else
                     ("certified" if s["w_l_certified"] else "termination_not_certified"),
                     _f(s["sin_theta0"], ".4f"),
                     ", ".join(list(s["flags"]) + [f"rule B: {f}" for f in s["bracket_flags"]]
                               + ([] if s["rule_b_terminates"] else ["rule B: not_terminating"])) or "-"])
    lines += ["## QPE side per system (F40-F43) and the U1 supplement", "",
              "E1lb - E0' and CISD(H') - E0' in mHa; 'any spin': the lowest excited level of the ground irrep of any "
              "spin of H' (F41, U1's column) and its <S^2>; rule B's form for a state without a dedicated certificate "
              "(n_v), whether B_L (theta_L = E1lb - 5 g) terminates, and the final QPE of the filtered schemes at the "
              "certified U; c_filt, c_c (C_QPE) at the certified U and at the pass edge E1lb - w_L of the L forms "
              "('-': no certified U, or E0' > E1lb - w_L: not terminating).", ""]
    lines += _table(["system", "N", "kind", "E1lb - E0'", "any spin (mHa)", "<S^2>", "CISD - E0'", "U certified",
                     "rule B (n_v)", "B_L terminates", "final QPE at U", "c_filt(U)", "c_c(U)", "c_filt(w_L)",
                     "c_c(w_L)", "L forms", "sin theta0", "flags"], rows)
    rows = []
    for name, c in res.get("cisdt_sensitivity", {}).items():
        for r, v in c["by_r"].items():
            states = "; ".join(f"{lab} {v['best_cisd'][k]} {_f(v['t_best_cisd'][k], '.4g')} -> {v['best'][k]} "
                               f"{_f(v['t_best'][k], '.4g')}" for k, lab in (("0", "HF"), ("1", "Psi1"), ("2", "Psi2")))
            rows.append([name, _f(c["u_minus_e0_model"] * MHA, ".3f"),
                         f"{'yes' if c['certified_cisd'] else 'no'} -> {'yes' if c['certified'] else 'no'}",
                         f"{c['form_cisd']} ({c['n_v_cisd']}) -> {c['form']} ({c['n_v']})", _f(float(r), "g"), states,
                         f"{_f(None if v['relative_cisd'] is None else 100 * v['relative_cisd'], '.3g')}% -> "
                         f"{_f(None if v['relative'] is None else 100 * v['relative'], '.3g')}%"
                         f" ({'pays' if v['pays'] else 'does not pay'})"])
    if rows:
        lines += ["CISDT sensitivity (F42's optional certificate in place of CISD(H'); reported, never the verdict of "
                  "record): CISDT(H') - E0' (mHa), certification and rule B's form at HF, s* and T_s* (C_QPE) of the "
                  "R1 spin-orbital HF, Psi_1 and Psi_2 rows of U9 part 1 (CISD -> CISDT; dedicated certificates not "
                  "re-evaluated), and the relative saving of the first measured iteration at s*.", ""]
        lines += _table(["system", "CISDT - E0'", "U certified", "rule B (n_v)", "r", "s* and T (CISD -> CISDT)",
                         "first iteration at s*"], rows)
    rows = []
    for r in res["U4_schemes"]["rows"]:
        cells = []
        for s in U4_SCHEMES:
            x = r["schemes"][s]
            cells.append("-" if x is None else f"{_f(x['mu'], '.3g')} / {_f(x['rho'], '.3g')}")
        b, e = r["best"], r.get("extended") or {}
        ext = ("-" if not e or (e["k"], e["k1"]) == (b["k"], b["k1"]) else
               f"{e['k']} -> {e['k1']}: {_f(e['mu'], '.3g')}, {'pays' if e['pays'] else 'does not pay'}")
        rows.append([r["system"], r["metric"], _f(r["r"], "g"), r["k"], _f(r["p"], ".4f"), _f(r["c_iter"], ".3g"),
                     f"{_f(r['mu_rep'], '.3g')} / {_f(r['rho_rep'], '.3g')}"] + cells
                    + [f"{b['k']} -> {b['k1']}", f"{_f(b['mu'], '.3g')} / {_f(b['rho'], '.3g')}",
                       "yes" if b["pays"] else "no",
                       f"{', '.join(b.get('flags_k') or []) or '-'} -> {', '.join(b.get('flags_k1') or []) or '-'}",
                       ext])
    lines += ["## U4: scheme columns (mu_k / rho_k, C_QPE units)", "",
              "Rows of the R1 records (p on psi_0 of H); rho = c_iter(k)/mu_k, the iteration pays iff rho < 1; s* at "
              "Psi_k -> Psi_k+1 with its flags and mu* = T_s*(k) - T_s*(k+1) + c_iter(k) (rule B's verification runs "
              "in rep, bbht and fp); 'extended': the optimum over the set extended by the step filter followed by "
              "repetition (lt_rep, lt_rep_L; a sensitivity, never s*) where it differs from s*: 'k -> k1: mu, pays'.",
              ""]
    lines += _table(["system", "metric", "r", "k", "p_k", "c_iter", "rep"] + list(U4_SCHEMES)
                    + ["s*", "mu* / rho*", "pays at s*", "flags of s*", "extended"], rows)
    rows = []
    for b in res["U4_schemes"]["n_star"]:
        bm = b.get("by_margin") or {}
        nw = b["no_window"]
        rows.append([_f(b["R"], ".1f"), b["metric"], _f(b["r"], "g"), b["scheme"],
                     f"{_rng(nw['n_star'], '.0f')} ({nw['n_none']}/{nw['n_cases']} none)"]
                    + [f"{_rng(bm[k]['n_star'], '.0f')} ({bm[k]['n_none']}/{bm[k]['n_cases']} none; inside "
                       f"{bm[k]['n_inside']}/{bm[k]['n_cases']})" if k in bm else "-" for k in ("E", "B", "BL")]
                    + [" / ".join(_rng(bm[k]["p_k"], ".2g") for k in ("E", "B", "BL") if k in bm) or "-",
                       b.get("l_window", {}).get("flag", "-")])
    lines += ["## U4: break-even size n* under the QPE-side schemes (F45)", "",
              "The first measured iteration of the R1 loop; ranges over the brackets of the extrapolation protocol; "
              "'inside k/N': the cases whose n* lies within the window of their own gap model ((A/m)^(1/gamma) for "
              "the power law, A'/m for A'/n); p_1 and p_2 extrapolated at the same rate (matched corners); a model "
              "bracket of H chains in canonical orbitals with elementwise compilation and the full allowed residual; "
              "'best': min(rep, bbht) beyond the computed sizes, with the window check of the E1lb-referenced forms.",
              ""]
    lines += _table(["R", "metric", "r", "scheme", "n* (no window)", "n* with m_E", "n* with m_B", "n* with m_BL",
                     "p_1(n*) (m_E / m_B / m_BL)", "L window"], rows)
    show = ("rep_rule_e", "rep", "aa", "bbht", "fp", "lt", "lt_bbht", "2s", "lt_bbht_L", "2s_L")
    rows = []
    for name, blk in res["U9_closed_loop"].items():
        for cfg in blk["configs"]:
            for st in cfg["stats"]:
                cells = [_ci(st["T"][s]["mean"], st["T"][s]["ci"], ".4g") if s in st["T"] else
                         (f"({_f(st['T'][s + '@oracle_U']['mean'], '.4g')} oracle U)" if s + "@oracle_U" in st["T"]
                          else "-") for s in show]
                ext = st.get("best_extended")
                rows.append([name, cfg["label"] + (" (decides)" if cfg["deciding"] else ""), st["k"],
                             _f(st["p_median"], ".3f")] + cells
                            + [st["best"] or "-", ", ".join(st.get("flags_best") or []) or "-",
                               "-" if ext is None or ext == st["best"] else
                               f"{ext} {_ci(st['T'][ext]['mean'], st['T'][ext]['ci'], '.4g')}"])
    lines += ["## U9 (part 2): QPE-side schemes on the closed-loop states", "",
              "Mean T_s(k) over the replicates [95% bootstrap CI], C_QPE units; 'rep_rule_e' is M3's rule E, rep, aa, "
              "bbht with rule B's verification runs (CISD(H') the only certificate); s* of the means with its flags, "
              "and the optimum over the set extended by lt_rep and lt_rep_L where it differs (a sensitivity).", ""]
    if not rows:
        lines += ["(no closed-loop inputs)", ""]
    lines += _table(["system", "configuration", "k", "median p~_k"] + list(show) + ["s*", "flags of s*", "extended"],
                    rows)
    cols = ("rep_rule_e",) + tuple(amplify.SCHEMES) + tuple(amplify.SENSITIVITY_SCHEMES)
    rows = []
    for name, blk in res["U9_closed_loop"].items():
        for cfg in blk["configs"]:
            for v in cfg["verdicts"]:
                if v["k"] == 0:
                    continue
                cells = []
                for s in cols:
                    x = v["by_scheme"].get(s)
                    cells.append("-" if x is None
                                 else f"{_ci(x['mean'], x['ci'], '.3g')}{' pays' if x['pays'] else ''}")
                b, e = v["best"], v.get("extended")
                rows.append([name, cfg["label"] + (" (decides)" if cfg["deciding"] else ""),
                             f"Psi{v['k']} -> Psi{v['k'] + 1}"] + cells
                            + [f"{v['best_k']} -> {v['best_k1']}",
                               "-" if b is None else f"{_ci(b['mean'], b['ci'], '.3g')}",
                               "yes" if v["pays"] else "no",
                               f"{', '.join(v.get('flags_best_k') or []) or '-'} -> "
                               f"{', '.join(v.get('flags_best_k1') or []) or '-'}",
                               "-" if e is None or (e["best_k"], e["best_k1"]) == (v["best_k"], v["best_k1"]) else
                               f"{e['best_k']} -> {e['best_k1']}: {_ci(e['mean'], e['ci'], '.3g')}"
                               f"{' pays' if e['pays'] else ''}"])
    lines += ["Savings of the measured iterations, T(k) - T(k + 1) [95% CI] (C_QPE): per scheme (rep_rule_e: M3's rule "
              "E; lt_rep, lt_rep_L: the sensitivity schemes) and at s* (the verdict of record: pays iff the interval "
              "lies above 0), with the flags of s* and the paired verdict at the extended optima where they differ.",
              ""]
    if not rows:
        lines += ["(no closed-loop inputs)", ""]
    lines += _table(["system", "configuration", "iteration"] + [s.replace("rep_rule_e", "rep (rule E)") for s in cols]
                    + ["s*", "saving at s*", "pays at s*", "flags of s*", "extended"], rows)
    rows = []
    for name, d in res["U11_closed_loop"].items():
        exact = res["U11_exact"].get(name, {})
        for kind, t in d["targets"].items():
            cells = []
            for r in ("0.3", "1"):
                v = t["by_r"].get(r)
                if v is None:
                    cells.append("-")
                    continue
                keps = ", ".join(f"{e}: {_f(x['k_median'], '.0f')} ({100 * x['reached']:.0f}%)"
                                 for e, x in v["k_eps"].items())
                cells.append(f"{_f(v['min_median'], '.1e')} [{_f(v['min_band'][0], '.1e')}, "
                             f"{_f(v['min_band'][1], '.1e')}] k {_f(v['k_min_median'], '.0f')}; last "
                             f"{_f(v['last_median'], '.1e')}; k_eps {keps}")
            f47 = min((x for v in t["by_r"].values() for x in (v["f47_q_min"], v["f47_energy_min"]) if x is not None),
                      default=None)
            ex = exact.get(kind)
            rows.append([name, kind, "-" if ex is None else f"{_f(ex['min'], '.1e')} [{ex['k_min']}]"] + cells
                        + [_f(f47, ".3g")])
    lines += ["## U11: closed-loop deep runs (bias floors)", "",
              "Route best, lagged controller, R1 spin-orbital, 30 iterations: the minimum property error over k "
              "(median [16-84%] over the replicates) with its median k, the last iterate's error and k_eps (median, "
              "share of replicates that reach eps); the exact R1 floor [its k] of the properties benchmark; F47: the "
              "smallest bound/|bias| over every iterate (q and energy forms; >= 1: the bound holds).", ""]
    lines += _table(["system", "target", "exact R1 floor", "closed loop r = 0.3", "closed loop r = 1", "F47 min ratio"],
                    rows)
    rows = []
    for name, d in res["U11_closed_loop"].items():
        for r, lp in d["loops"].items():
            rows.append([name, r, lp["n"], _f(lp["r1_median"], ".2f"), _f(lp["p_last_median"], ".4f"),
                         _f(lp["e_err_last_median"] * MHA, ".3g"), _f(lp["c_loop_last_median_over_qpe"], ".3g"),
                         lp["converged"], ", ".join(lp["flags"]) or "-", ", ".join(lp["non_realizable"]) or "-"])
    lines += ["Deep runs: realized r_1, p~ and E - E0 (mHa) at the last iterate, C_loop(30)/C_QPE (medians).", ""]
    lines += _table(["system", "r", "replicates", "r_1", "p~_last", "E - E0", "C_loop", "converged", "flags",
                     "non-realizable"], rows)
    rows = []
    for u in res["U12"]:
        layout = "error" not in u
        note = "; ".join(x for x in (u.get("note", ""), "" if layout else "no layout: " + u["error"]) if x)
        rows.append([u["run"], u["system"], u.get("logical_qubits", "-"), _f(u.get("toffolis"), ".4g"),
                     _f(u.get("fail_budget"), ".2g"), _f(u.get("physical_qubits"), ".4g") if layout else "none",
                     u.get("distance", "-"), u.get("factory", "-"), _f(u.get("us_per_toffoli"), ".4g"),
                     _f(u.get("seconds"), ".4g"), _f(u.get("runs"), ".4g"), _f(u.get("machines_7d"), ".4g"), note])
    lines += ["## U12: physical units of the decisive runs (F18, F19, F40)", "",
              "Per run: logical qubits, Toffolis, the failure budget, physical qubits, code distance d, factory, "
              "microseconds per Toffoli, seconds; runs and machines that finish them within 7 days.", ""]
    lines += _table(["run", "system", "Q", "Toffolis", "budget", "physical qubits", "d", "factory", "us/Toffoli",
                     "seconds", "runs", "machines (7 d)", "note"], rows)
    lines += ["## Integration checks", ""]
    lines += _table(["check", "passed", "detail"], [[c["name"], "yes" if c["passed"] else "NO", c["detail"]]
                                                    for c in res["checks"]])
    lines += ["## Runtime (s, one thread)", ""]
    lines += _table(["part", "seconds"], [[k, _f(v, ".1f")] for k, v in res["timings"].items()])
    return "\n".join(lines)


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help=f"deep runs on {', '.join(DEEP_QUICK)} only; the quick input sets")
    ap.add_argument("--systems", default=None, help="restrict every part to these comma-separated systems")
    ap.add_argument("--out", default="results/m4", help="input and output directory (cache in <out>/cache)")
    ap.add_argument("--no-cache", action="store_true", help="recompute the summary's own cached stages")
    ap.add_argument("--deep-replicates", type=int, default=DEEP_REPLICATES, help="replicates of each deep run")
    ap.add_argument("--allow-partial", action="store_true",
                    help="missing inputs do not fail the run (they are still reported)")
    args = ap.parse_args(argv)
    t_script = time.perf_counter()
    timings: dict[str, float] = {}

    def lap(label: str, t: float) -> float:
        now = time.perf_counter()
        timings[label] = timings.get(label, 0.0) + now - t
        return now

    selected = None if not args.systems else {s.strip() for s in args.systems.split(",") if s.strip()}
    out_label = os.path.relpath(args.out)
    params = FTParameters()
    model = physical.SurfaceCodeModel()
    up_cache = scaling.Cache(os.path.join(args.out, "cache"))
    own = scaling.Cache(os.path.join(args.out, "cache"), refresh=args.no_cache)
    t = time.perf_counter()
    found = {f: discover(args.out, f, selected) for f in FAMILIES}
    scaling_json, loop_json, cl_json = found["scaling"], found["loop"], found["closedloop"]
    handoff_json, properties_json = found["handoff"], found["properties"]
    fits_path = os.path.join(args.out, "scaling_fits.json")
    fits = None
    if os.path.exists(fits_path):
        with open(fits_path) as fh:
            fits = json.load(fh)
    missing = missing_inputs(found, fits is not None, required_inputs(args.quick, selected))
    checks: list[dict] = [inputs_check(missing)]
    up = Upstream(up_cache, params, scan_settings(scaling_json))
    print(f"[inputs] scaling {len(scaling_json)}, loop {len(loop_json)}, closed loop {len(cl_json)}, hand-off "
          f"{len(handoff_json)}, properties {len(properties_json)}; fits {'yes' if fits else 'no'}; "
          f"{checks[0]['detail']}", flush=True)
    t = lap("inputs", t)
    # QPE sides
    names = sorted(set(scaling_json) | set(cl_json))
    sides = {}
    for name in names:
        sides[name] = qpe_side(name, up.parts(name), own, params)
        if name in handoff_json:
            checks.append(side_check(sides[name], handoff_json[name]))
    # the hand-off benchmark's systems: U9 part 1 re-costed, U8 against U9, and the CISDT sensitivity (F42)
    cisdt = {}
    for name, hand in handoff_json.items():
        checks.append(hf_cost_check(name, hand))
        if FCISpace.of(benchmark_system(name)).size > DENSE_MAX:      # no model spectrum of H' (N = 10)
            continue
        parts = up.parts(name)
        if name in sides:
            side = sides[name]
        else:
            side = qpe_side(name, parts, own, params)
            checks.append(side_check(side, hand))
        checks.append(recost_check(side, hand))
        cisdt[name] = cisdt_sensitivity(qpe_side(name, parts, own, params, u_level=3), side, hand)
    print(f"[QPE sides] {len(sides)} systems; CISDT sensitivity on {len(cisdt)}; peak RSS {peak_rss_mb():.0f} MB",
          flush=True)
    t = lap("qpe_sides", t)
    # U4
    u4 = {"rows": [], "n_star": []}
    if fits is not None:
        u4["rows"] = u4_scheme_rows(fits, scaling_json, sides)
        chain = [n for n in scaling_json if CHAIN.fullmatch(n)]
        if chain:
            records = [up.scan(n) for n in chain]
            rep_again = scaling.break_even_sizes(records, sin_theta0=fits["settings"]["sin_theta0"],
                                                 e1_margin=fits["settings"]["e1_margin"])
            checks.append({"name": "U4 repetition brackets recomputed from the cached records equal scaling_fits.json",
                           "passed": rep_again == fits["U4"]["n_star"],
                           "detail": f"{len(rep_again)} brackets"})
            u4["n_star"] = n_star_schemes(records, fits, sides)
    print(f"[U4] {len(u4['rows'])} rows, {len(u4['n_star'])} brackets; peak RSS {peak_rss_mb():.0f} MB", flush=True)
    t = lap("u4", t)
    # U9 part 2
    cl_s = {}
    for name, data in cl_json.items():
        cl_s[name] = closed_loop_schemes(data, sides[name])
        ce = cl_s[name]["check_rule_e"]
        checks.append({"name": f"{name}: rule-E costs and paired verdicts equal the loop benchmark's summary",
                       "passed": ce["passed"], "detail": f"{ce['verdicts']} verdicts, worst relative cost difference "
                                                         f"{ce['worst_relative']:.1e}"})
    noisy = step_rules_noisy(cl_json, target=params.target)
    print(f"[U9 part 2] {len(cl_s)} systems; U7 step rules on {len(noisy)}", flush=True)
    t = lap("u9_closed_loop", t)
    # U11 deep runs
    deep_names = [n for n in (DEEP_QUICK if args.quick else DEEP_SYSTEMS) if selected is None or n in selected]
    deep, exact = {}, {}
    for name in deep_names:
        t1 = time.perf_counter()
        deep[name] = deep_runs(name, up, own, params, n=args.deep_replicates, max_iter=DEEP_ITER)
        if name in properties_json:
            exact[name] = {kind: blk["floors"]["r1_so"] for kind, blk in properties_json[name]["targets"].items()}
            ok = all(abs(deep[name]["targets"][kind]["by_r"][r]["hf_error"]
                         - _num(properties_json[name]["targets"][kind]["errors"]["r1_so"][0]))
                     <= 1e-12 * max(1.0, abs(_num(properties_json[name]["targets"][kind]["errors"]["r1_so"][0])))
                     for kind in deep[name]["targets"] for r in deep[name]["targets"][kind]["by_r"])
            checks.append({"name": f"{name}: deep-run property errors of HF equal the properties benchmark's",
                           "passed": ok, "detail": f"{len(deep[name]['targets'])} targets"})
        print(f"[deep {name}] {time.perf_counter() - t1:.1f} s; peak RSS {peak_rss_mb():.0f} MB", flush=True)
    t = lap("deep_runs", t)
    # U12
    u12 = u12_rows(up, scaling_json, handoff_json, properties_json, sides, params, model)
    t = lap("u12", t)
    # U0
    u0 = assemble_u0({"fits": fits, **found}, {"u4": u4, "cl_s": cl_s, "deep": deep, "cisdt": cisdt}, missing,
                     out_label)
    t = lap("u0", t)
    timings["total"] = time.perf_counter() - t_script
    failed = [c["name"] for c in checks if not c["passed"]]
    res = {"out": out_label,
           "inputs": {"scaling": sorted(scaling_json), "loop": sorted(loop_json), "closedloop": sorted(cl_json),
                      "handoff": sorted(handoff_json), "properties": sorted(properties_json),
                      "fits": fits is not None, "missing": missing},
           "settings": {"eps_f": EPS_F, "eps_p": EPS_P, "w_l": amplify.W_L, "k_table": K_TABLE,
                        "deep": {"systems": deep_names, "r": list(DEEP_R), "iterations": DEEP_ITER,
                                 "replicates": args.deep_replicates}, "params": dataclasses.asdict(params),
                        "script_digest": SCRIPT_DIGEST},
           "U0": u0, "qpe_sides": sides, "cisdt_sensitivity": cisdt, "U4_schemes": u4, "U9_closed_loop": cl_s,
           "U7_step_rules": noisy, "U11_closed_loop": deep, "U11_exact": exact, "U12": u12, "checks": checks,
           "checks_failed": failed, "allow_partial": bool(args.allow_partial), "timings": timings,
           "peak_rss_mb": peak_rss_mb()}
    out = scaling.to_json(res)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(out, fh, allow_nan=False)
    command = ("benchmarks/run_m4_summary.py" + (" --quick" if args.quick else "")
               + (f" --systems {args.systems}" if args.systems else "") + f" --out {out_label}"
               + (" --no-cache" if args.no_cache else "") + (" --allow-partial" if args.allow_partial else ""))
    with open(os.path.join(args.out, "summary.md"), "w") as fh:
        fh.write(markdown(out, command, timings["total"]))
    for v in out["U0"]:
        print(f"[U0 {v['q']}] {v['verdict']}", flush=True)
    tail = "" if not failed else "; FAILED: " + "; ".join(failed)
    print(f"[checks] {len(checks) - len(failed)} of {len(checks)} pass{tail}", flush=True)
    print(f"wrote summary.json and summary.md to {out_label} ({timings['total']:.1f} s; peak RSS "
          f"{peak_rss_mb():.0f} MB)", flush=True)
    return exit_status(checks, allow_partial=args.allow_partial)


if __name__ == "__main__":
    sys.exit(main())
