"""Iteration costs, per-system reports and the tables T1-T9 of M3 (F14, F15, F17, F19).

Tags and formula numbers are those of :mod:`thc_engine_cqe.ft.primitives`. Costs are Toffolis
(one CCZ state each), energies Hartree, times seconds; "C_QPE" is one phase estimation of the
selected block encoding at ``params.de_qpe``.

Optimizer families (:data:`FAMILIES`)
------------------------------------
Exact CQE trajectories from HF (``ft.trajectory.exact_trajectory``, symmetry-projected residual):

    family     run_cqe optimizer, metric                     max_iter
    lbfgs      "lbfgs" (memory 10), "spin_free"              20        Appendix A, golden record
    lbfgs_so   "lbfgs" (memory 10), "spin_orbital"           20        the standard CQE generator
    sd         "sd", "spin_free"                             30        core systems only

"spin_orbital" preconditions with M = (2I + T)/3, (TX)[p,q,r,s] = X[p,s,r,q], so that the
direction is the standard (Mazziotti) spin-orbital ACSE generator (``cqe.run_cqe``). The
measured residual is the spin-free A of every family, with the same element sets,
normalizations and per-shot variances; only the states, directions and steps differ. Every
family has two records: "exact" (step 0 by the exact line search, the measured loop and the
free loop) and "second_order" (step 0 the classical eps_2 of F16 through
``run_cqe(..., first_eps=eps_2)``, then the exact continuation: the classical-first-step
loop, T7 and T8).

Iteration cost (:func:`iteration_costs`, F14)
--------------------------------------------
At iterate k of a record, with Psi_k prepared by the elementwise circuit of ``ft.stateprep``
(C(Psi_k), Q(Psi_k)):

* Residual: ``ft.residual.compare_routes`` on the ``ResidualProblem`` of Psi_k
  (``ft.residual_inputs.build_problem``: the exact A and the aux-route bias at Psi_k; the
  per-shot variances and normalizations computed once per system at iterate 1 of the
  first family), with the walk of the selected block encoding and its factorization bias
  ||A[H'] - A[H]||_m at Psi_k. The best route is ``best_route(costs, kappa=params.kappa)``
  and, separately, at ``params.kappa_practical`` (gradient estimation never competes, F12).
* Line search (F13): the candidates eps_c = ``params.eps_candidates`` along the record's d_k,
  each compiled elementwise (``stateprep.candidate_step_costs``), measured by DF
  basis-rotation grouping with V_E of the split chosen at HF (``energy_variances(...)["best"]``)
  evaluated at Psi_{k+1}; one shot runs C(Psi_k) + C_step(k; eps_c) + C_rot on
  max(Q(Psi_k), Q_step(eps_c), Q_rot) logical qubits (``stateprep.readout_rotation_qubits``).
  Candidates flagged "rounding bound above delta_step" (F3 on small steps) are counted. The
  F13 option of amplitude-estimated energies of H' (``residual.energy_ae`` per candidate on the
  same circuits plus the walk, at kappa) is reported next to it; C_iter uses the sampled
  line search, the hardware rule of M2.
* C_iter(k) = T_res(best) + T_LS, at kappa and at kappa_practical. With
  ``classical_first_step`` the iteration at HF is free (A_0 and eps_2 are classical, F16).
  Without residual inputs only the line search is costed: C_iter is then a lower bound
  (best = "not costed", flag "line search only"). A campaign without a surface-code layout
  at its failure budget (F18) is flagged; with the selected walks no route is infeasible
  over the costed iterates of the benchmark (the largest factorization bias is 0.47 of
  r ||A||_m, n2s k = 16), and every campaign has a layout.

Wall clock (F19) [assumption]
-----------------------------
Every iteration is a sequence of campaigns of independent runs (:class:`Campaign`), each
costed by ``ft.physical`` (layouts cached per logical qubits, Toffolis, budget and shot):

* sampling (direct_mg, aux_mg, and the line search, one campaign per candidate): shots of
  one circuit at budget ``params.p_fail_shot`` with 2d readout cycles; t ceil(n/M) on M
  machines (``physical.campaign_seconds``, whole runs);
* amplitude estimation (ae_product, ae_x, ae_aux): one coherent run per element
  (``RouteCost.run_toffolis``), every run at the microseconds per Toffoli of the layout of
  the longest one (``RouteCost.coherent_toffolis`` on ``RouteCost.logical_qubits``, budget
  ``params.p_fail_ae``), so one machine takes ``RouteCost.toffolis`` Toffolis at that rate
  (F19). The runs are unequal and indivisible (splitting one costs more applications, F11);
  on M machines they follow the LPT schedule, longest run first onto the least-loaded machine
  [cited: Graham, SIAM J. Appl. Math. 17, 416 (1969): within 4/3 - 1/(3M) of the optimal
  makespan]. It is a schedule, so the machine counts below suffice, and any schedule needs at
  least max(W/M, t_max) (W the work, t_max the longest run).

A loop runs its campaigns one after another, each spread over the same M machines: iteration
k + 1 starts from the step of iteration k, and a line search needs the residual's direction.
The machines for a deadline D (7 days) are the smallest M with sum_c T_c(M) <= D
(:func:`machines_for_campaigns`, by bisection [assumption: T_c(M) does not grow with M; true
for identical runs, and no LPT counterexample among 2e5 random instances of 2-12 runs]); None
if a campaign has no layout or the critical path, the longest runs of all campaigns (the
wall clock on unlimited machines), exceeds D. One QPE run is one run (budget
``params.p_fail_qpe``); 1/p QPE repetitions take 1/p runs on average.

Observed (benchmark): the LPT counts equal the smallest M allowed by the lower bound, summed
over the campaigns (t ceil(n/M) for shots, max(W/M, t_max) for AE runs), for 324 of the 404
costed iterations (at most 1.38 times it, h6s k = 13) and for 28 of the 31 loop totals that
have a count (at most 1.05 times; the other 5 have a critical path above 7 days). Sizing M by
the bound of greedy list scheduling in arbitrary order, W/M + (1 - 1/M) t_max [cited: Graham,
Bell Syst. Tech. J. 45, 1563 (1966)], asks for up to 5.6 times as many machines (h6a k = 10:
824 against 146) and 2.6 times for a loop (spin-orbital h6s: 518 against 203); a fluid share
of the work, ceil(sum_c W_c / D), is no schedule: it gives 249 machines to the spin-free h6s
loop, whose critical path is 8.5 days.

Per-system report (:func:`run_system`)
--------------------------------------
QPE scan of all three kinds (``ft.qpe.scan_encodings``) and the selection rule per kind and
overall; the overall selection is the QPE of the hand-off and the walk of the residual
routes. Per family and record: the trajectory, the elementwise and leaf preparation costs,
the iteration costs to k* + 4 (and to k_v - 1 for F17; all iterates if k* is not reached),
the hand-off tables of the three loops (``ft.handoff``, same-irrep and any-irrep E1), the
classical first step (``second_order_step``, compiled replay of step 0), the CQE-only cost
(F17), the replay of the compiled circuits to k* + 2 and the physical costs. ``to_json``
gives a strict-JSON dict (nan as null, +-inf as "inf"/"-inf") and ``markdown_tables``
the tables T1-T9. The hand-off QPE is the cheapest admissible one [assumption: the
alternative a user would run]; on every benchmark system it also has the lowest lambda C_W
of the three per-kind selections (T1), i.e. the cheapest walk per unit of evolution time.

Observed (``benchmarks/run_ft_pipeline.py`` on one thread; docs/m3-results.md)
---------------------------------------------------------------------------
The selected QPE is SF on h4, n2, lih and n2s and THC on h6a, h6s and h8, C_QPE = 4.0e6 (h4)
to 3.1e7 (h8) Toffolis. One measured iteration costs 154-1366 C_QPE at HF, where the line
search is all of it, and grows to 3.0e3-1.6e5 C_QPE by k* + 4. Before k* the line search costs
more than the residual at equilibrium (residual / line search <= 0.57); on h6s the residual
overtakes it from k = 8 (spin-free; 6 spin-orbital), up to 2.1 times before k*, on n2s only in
the last iterations (k >= 13), up to 5.5 times (spin-free) and 7.2 (spin-orbital). Amplitude
estimation beats matchgate shadows from k = 2-4 (ae_x: h4, n2, lih, n2s; ae_aux at k = 9 of h4
sd) or k = 3 (ae_aux: h6a, h6s).
C_loop(k*) of spin-free L-BFGS with the classical first step is 808 (lih) to 6.3e4 (h6a) C_QPE
at equilibrium and 4.1e5 on h6s. The measured loop never pays at equilibrium (the optimal
hand-off is HF under every rule); the classical first step saves 0.02-0.10 C_QPE. On h6s and
n2s the HF hand-off is
not certified; the classical step gives the cheapest certified hand-off on h6s (1.13 and 1.09
C_QPE, spin-free and spin-orbital) and on n2s with the spin-orbital metric (2.52 C_QPE), not
with the spin-free one (1073 C_QPE, against 571 for the measured loop). Runtime and memory of
the benchmark: docs/m3-results.md, section 9.
"""

from __future__ import annotations

import dataclasses
import functools
import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..diagnostics.thc_floors import ExactReference
from ..fci_ops import inner, normalize
from ..integrals import ActiveSpace
from . import residual_inputs as ri
from .circuits import replay_elementwise
from .handoff import LOOPS, RULES, CQEOnlyCost, HandoffRow, cqe_only_cost, handoff_table, optimal_handoff
from .physical import (SECONDS_PER_WEEK, PhysicalCost, SurfaceCodeModel, campaign_seconds,
                       surface_code_cost)
from .primitives import FTParameters
from .qpe import KINDS as QPE_KINDS
from .qpe import BlockEncoding, QPECost, scan_encodings, select_qpe
from .residual import (PrepParams, RouteCost, best_route, compare_routes, crossover_norm, energy_ae,
                       line_search_cost)
from .stateprep import (FLAG_ROUNDING, PrepCost, candidate_step_costs, prep_cost, readout_rotation_qubits,
                        readout_rotation_toffolis)
from .symmetry import benchmark_system, orbital_irreps
from .trajectory import TrajectoryRecord, exact_trajectory, first_iterate_below, second_order_step

__all__ = [
    "FAMILIES",
    "CORE_SYSTEMS",
    "LBFGS_ONLY_SYSTEMS",
    "RECORDS",
    "R_SCAN_SYSTEMS",
    "R_VALUES",
    "layout",
    "lpt_makespan",
    "Campaign",
    "machines_for_campaigns",
    "IterationCost",
    "iteration_costs",
    "SystemReport",
    "default_optimizers",
    "run_system",
    "global_validation",
    "handoff_optima",
    "loop_totals",
    "first_heisenberg_iterate",
    "report_summary",
    "to_json",
    "markdown_tables",
]

#: family -> (run_cqe optimizer, metric, default max_iter) (module docstring).
FAMILIES: dict[str, tuple[str, str, int]] = {
    "lbfgs": ("lbfgs", "spin_free", 20),
    "lbfgs_so": ("lbfgs", "spin_orbital", 20),
    "sd": ("sd", "spin_free", 30),
}
CORE_SYSTEMS = ("h4", "h6a", "n2", "lih")
#: Systems that run the L-BFGS families only [design].
LBFGS_ONLY_SYSTEMS = ("h6s", "n2s", "h8")
RECORDS = ("exact", "second_order")

_AE_ROUTES = ("ae_product", "ae_x", "ae_aux")
_SAMPLING_ROUTES = ("direct_mg", "aux_mg")
_K_ITER = 4        # iteration costs to k* + 4 [design]
_K_TABLE = 2       # T2 and the replay to k* + 2
_K_UNREACHED = 8   # replay and T2 rows when k* is not reached
_N_ROOTS = 12      # singlet FCI roots for E1 (ft.trajectory)
_FLAG_LS_ONLY = "line search only"
_FLAG_NO_LAYOUT = "no layout within the failure budget"
#: T5: the r-scan systems and values [design].
R_SCAN_SYSTEMS = ("h6a", "n2")
R_VALUES = (0.1, 0.3, 1.0)


# ------------------------------------------------------------------------- wall clock (F19)
@functools.lru_cache(maxsize=None)
def _layout(n_logical: int, n_toffoli: int, budget: float, shot: bool, model: SurfaceCodeModel) -> PhysicalCost | None:
    """Cached F18 layout (None if no layout meets the budget)."""
    try:
        return surface_code_cost(n_logical, n_toffoli, fail_budget=budget, model=model, shot=shot)
    except ValueError:
        return None


def layout(n_logical: int, n_toffoli: float, budget: float, *, shot: bool,
           model: SurfaceCodeModel = SurfaceCodeModel()) -> PhysicalCost | None:
    """F18 layout of a circuit, cached per (Q, Toffolis rounded to an integer, budget, shot, model)."""
    return _layout(int(n_logical), int(round(float(n_toffoli))), float(budget), bool(shot), model)


def lpt_makespan(runs_s: Sequence[float], n_machines: int) -> float:
    """Makespan (seconds) of the LPT schedule of indivisible runs on ``n_machines`` identical machines.

    Longest run first, each onto the machine that frees first [cited: Graham, SIAM J. Appl.
    Math. 17, 416 (1969)]; the sum of the runs on one machine, the longest run on as many
    machines as runs.
    """
    m = int(n_machines)
    if m < 1:
        raise ValueError(f"n_machines must be >= 1, got {n_machines!r}")
    runs = sorted((float(t) for t in runs_s), reverse=True)
    if not runs:
        return 0.0
    if m == 1:
        return math.fsum(runs)
    if m >= len(runs):
        return runs[0]
    loads = [0.0] * m
    for t in runs:
        heapq.heapreplace(loads, loads[0] + t)
    return max(loads)


@dataclass(frozen=True)
class Campaign:
    """Independent runs of one estimator, on identical machines (F19; module docstring).

    Attributes:
        label: the estimator ("direct_mg", "ae_x", "line_search", "final_energy", ...).
        n_runs: shots or amplitude-estimation runs (real; inf for an infeasible route).
        per_run: layout of one run; of the longest run for unequal runs (None: no layout).
        work_s: seconds of all runs on one machine.
        identical: every run is the circuit of ``per_run`` (sampling); False for AE runs.
        runs_s: seconds of every run when they differ (AE; longest first, at the rate of
            ``per_run``); empty for identical runs.
    """

    label: str
    n_runs: float
    per_run: PhysicalCost | None
    work_s: float
    identical: bool = True
    runs_s: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.identical and self.per_run is not None and math.isfinite(self.n_runs) \
                and len(self.runs_s) != self.n_runs:
            raise ValueError(f"{self.label}: {len(self.runs_s)} run lengths for {self.n_runs} unequal runs")

    @property
    def longest_s(self) -> float:
        """Seconds of the longest run (inf without a layout)."""
        if self.per_run is None:
            return math.inf
        return self.runs_s[0] if self.runs_s else self.per_run.seconds

    def seconds(self, n_machines: int = 1) -> float:
        """Wall clock on ``n_machines`` machines: t ceil(n/M) for identical runs, the LPT makespan otherwise.

        On one machine both are ``work_s``.
        """
        if self.n_runs == 0:
            return 0.0
        if self.per_run is None or math.isinf(self.n_runs):
            return math.inf
        if self.identical:
            return campaign_seconds(self.n_runs, self.per_run, n_machines)
        return self.work_s if int(n_machines) == 1 else lpt_makespan(self.runs_s, n_machines)


def _sampling_campaign(label: str, shots: float, toffolis_per_shot: float, logical_qubits: int,
                       params: FTParameters, model: SurfaceCodeModel) -> Campaign:
    if not math.isfinite(shots):
        return Campaign(label, math.inf, None, math.inf)
    per = layout(logical_qubits, toffolis_per_shot, params.p_fail_shot, shot=True, model=model)
    work = math.inf if per is None else campaign_seconds(shots, per, 1)
    return Campaign(label, float(shots), per, work)


def _route_campaigns(cost: RouteCost, params: FTParameters, model: SurfaceCodeModel) -> tuple[Campaign, ...]:
    """The campaign of a residual route: shots for the sampling routes, AE runs otherwise."""
    if cost.route in _SAMPLING_ROUTES:
        return (_sampling_campaign(cost.route, cost.n_runs, cost.toffolis_per_prep, cost.logical_qubits,
                                   params, model),)
    if not math.isfinite(cost.toffolis):
        return (Campaign(cost.route, math.inf, None, math.inf, identical=False),)
    per = layout(cost.logical_qubits, cost.coherent_toffolis, params.p_fail_ae, shot=False, model=model)
    if per is None:
        return (Campaign(cost.route, cost.n_runs, None, math.inf, identical=False),)
    rate = per.us_per_toffoli * 1e-6
    runs = tuple(sorted((t * rate for t in cost.run_toffolis), reverse=True))
    return (Campaign(cost.route, cost.n_runs, per, cost.toffolis * rate, identical=False, runs_s=runs),)


def machines_for_campaigns(campaigns: Sequence[Campaign], deadline_s: float = SECONDS_PER_WEEK) -> int | None:
    """Smallest M >= 1 with sum_c T_c(M) <= deadline for campaigns run one after another (F19).

    T_c is :meth:`Campaign.seconds`; the search is a bisection (module docstring). 0 for no
    campaign (nothing to run on a quantum machine), None if no M suffices: a campaign without
    a layout or with infinitely many runs, or a critical path sum_c t_max,c above the deadline.
    """
    active = [c for c in campaigns if c.n_runs > 0]
    if not active:
        return 0
    if any(math.isinf(c.n_runs) or c.per_run is None for c in active):
        return None
    floor = sum(c.longest_s for c in active)        # every campaign needs its longest run on any M
    if floor > deadline_s:
        return None

    def total(m: int) -> float:
        return sum(c.seconds(m) for c in active)

    if total(1) <= deadline_s:
        return 1
    # Upper brackets: one run per machine leaves the floor; list scheduling keeps
    # T_c(M) <= W_c / M + t_max,c [Graham 1966], which meets the deadline at the second M.
    slack = deadline_s - floor
    hi = max(math.ceil(c.n_runs) for c in active)
    if slack > 0:
        hi = min(hi, max(2, math.ceil(sum(c.work_s for c in active) / slack)))
    if total(hi) > deadline_s:
        return None
    lo = 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if total(mid) <= deadline_s:
            hi = mid
        else:
            lo = mid
    return hi


# ------------------------------------------------------------------------- iteration costs (F14)
@dataclass(frozen=True)
class IterationCost:
    """Cost of CQE iteration k (residual at Psi_k and the line search to Psi_{k+1}), F14.

    Attributes:
        k: the iterate.
        residual: every route of ``residual.compare_routes`` (empty for the classical
            iteration 0 and when the residual is not costed).
        best: the cheapest route at ``params.kappa``; "classical" (iteration 0 of the
            classical first step) or "not costed" (line search only).
        line_search: the line-search energies (None for the classical iteration 0).
        toffolis: C_iter(k) at kappa = T_res(best) + T_LS.
        shots: sampling shots: the residual's if its best route samples, plus the line search's.
        seconds_one_machine: wall clock on one machine (F19; inf without a layout).
        machines_7d: machines that finish the iteration within 7 days (None: impossible).
        best_practical: the cheapest route at ``params.kappa_practical``.
        toffolis_practical: C_iter(k) at kappa_practical.
        residual_toffolis, residual_toffolis_practical: T_res of the best route at kappa and
            at kappa_practical.
        line_search_ae: Toffolis of the line search by amplitude-estimated energies of H' (F13
            option, at kappa; not part of C_iter); nan for the classical iteration 0.
        residual_norm_m: ||A_k||_m of the measured elements (Hartree; nan if not costed).
        prep_toffolis: C(Psi_k).
        ls_flagged: line-search candidates with the F3 flag "rounding bound above delta_step".
        campaigns: the campaigns of the best route and the line search (F19).
        flags: "line search only" (C_iter is a lower bound); "no layout within the failure
            budget" (a campaign has no surface-code layout at its budget: its seconds are
            inf and machines_7d None).
    """

    k: int
    residual: tuple[RouteCost, ...]
    best: str
    line_search: RouteCost | None
    toffolis: float
    shots: float
    seconds_one_machine: float
    machines_7d: int | None
    best_practical: str = ""
    toffolis_practical: float = 0.0
    residual_toffolis: float = 0.0
    residual_toffolis_practical: float = 0.0
    line_search_ae: float = math.nan
    residual_norm_m: float = math.nan
    prep_toffolis: float = 0.0
    ls_flagged: int = 0
    campaigns: tuple[Campaign, ...] = ()
    flags: tuple[str, ...] = ()


def _check_record(record: TrajectoryRecord, prep: PrepCost) -> None:
    if len(prep.steps) != record.n_steps or len(prep.cumulative) != len(record.states):
        raise ValueError(f"prep has {len(prep.steps)} steps for a record of {record.n_steps}")


def _line_search(active: ActiveSpace, record: TrajectoryRecord, prep_k: PrepParams, k: int, energy_variance: str,
                 readout_toffolis: int, q_rot: int, walk: BlockEncoding, params: FTParameters,
                 model: SurfaceCodeModel) -> tuple[RouteCost, int, tuple[Campaign, ...], float]:
    """F13 at iterate k: (cost, flagged candidates, one campaign per candidate, Toffolis of the AE option)."""
    cands = candidate_step_costs(record.directions[k], params.eps_candidates, irreps=record.irreps, params=params)
    q = max([prep_k.logical_qubits, q_rot] + [c.logical_qubits for c in cands])
    v_e = ri.energy_variances(active, record.states[k + 1])[energy_variance]
    cost = line_search_cost(v_e, prep_k, [c.toffolis for c in cands], readout_toffolis, params=params,
                            logical_qubits=q)
    shots = cost.params["shots_per_candidate"]
    campaigns = tuple(_sampling_campaign("line_search", shots, prep_k.toffolis + c.toffolis + readout_toffolis, q,
                                         params, model) for c in cands)
    wp = walk.walk_params()
    t_ae = sum(energy_ae(wp, PrepParams(prep_k.toffolis + c.toffolis, max(prep_k.logical_qubits, c.logical_qubits)),
                         eps=params.eps_ls, kappa=params.kappa, norb=active.norb).toffolis for c in cands)
    return cost, sum(FLAG_ROUNDING in c.flags for c in cands), campaigns, float(t_ae)


def iteration_costs(active: ActiveSpace, record: TrajectoryRecord, prep: PrepCost, qpe_walk: BlockEncoding, *,
                    elements: np.ndarray | None, variances: dict | None, normalizations: dict | None,
                    energy_variance: str, readout_toffolis: int, classical_first_step: bool = True,
                    k_max: int | None = None, params: FTParameters = FTParameters(),
                    model: SurfaceCodeModel = SurfaceCodeModel()) -> list[IterationCost]:
    """C_iter(k) for k = 0..k_max along ``record`` (F14; module docstring).

    Args:
        active: the active space of the record.
        record: an exact trajectory (``ft.trajectory.exact_trajectory``) with irreps.
        prep: its elementwise preparation cost (``stateprep.prep_cost``).
        qpe_walk: the block encoding of the walk (``ft.qpe``), with its model integrals for
            the factorization bias.
        elements, variances, normalizations: the measured elements and their per-shot
            variances and LCU normalizations (``ft.residual_inputs``); all None: the residual
            is not costed (line search only, a lower bound on C_iter).
        energy_variance: the DF split of ``residual_inputs.DF_VARIANTS`` chosen at HF (F13).
        readout_toffolis: C_rot of the energy readout (``stateprep.readout_rotation_toffolis``).
        classical_first_step: the iteration at HF is classical and free (F14, F16).
        k_max: last iterate (default the record's last step, K - 1; V_E needs Psi_{k+1}).
        params, model: budgets and the surface-code machine.
    """
    _check_record(record, prep)
    last = record.n_steps - 1 if k_max is None else int(k_max)
    if not 0 <= last <= record.n_steps - 1:
        raise ValueError(f"k_max must lie in 0..{record.n_steps - 1}, got {k_max!r}")
    if energy_variance not in ri.DF_VARIANTS:
        raise ValueError(f"energy_variance must be one of {tuple(ri.DF_VARIANTS)}, got {energy_variance!r}")
    inputs = (elements, variances, normalizations)
    if any(x is None for x in inputs) and not all(x is None for x in inputs):
        raise ValueError("elements, variances and normalizations must be given together (or all None)")
    with_residual = elements is not None
    model_active = qpe_walk.model(active) if with_residual else None
    q_rot = readout_rotation_qubits(active.norb, params=params)
    out = []
    for k in range(last + 1):
        prep_k = PrepParams(prep.cumulative[k], prep.logical_qubits[k])
        if k == 0 and classical_first_step:
            out.append(IterationCost(k=0, residual=(), best="classical", line_search=None, toffolis=0.0,
                                     shots=0.0, seconds_one_machine=0.0, machines_7d=0,
                                     best_practical="classical", prep_toffolis=float(prep_k.toffolis)))
            continue
        ls, flagged, ls_campaigns, ls_ae = _line_search(active, record, prep_k, k, energy_variance, readout_toffolis,
                                                        q_rot, qpe_walk, params, model)
        if with_residual:
            psi = record.states[k]
            problem = ri.build_problem(active, psi, elements, params.kind, variances=variances,
                                       normalizations=normalizations, deltas=params.aux_deltas)
            bias = ri.factorization_bias(active, model_active, psi, elements, params.kind)
            routes = tuple(compare_routes(problem, prep_k, qpe_walk.walk_params(bias), params=params))
            best = best_route(routes, kappa=params.kappa)
            best_p = best_route(routes, kappa=params.kappa_practical)
            res_campaigns = _route_campaigns(best, params, model)
            names, t_res, t_res_p = (best.route, best_p.route), best.toffolis, best_p.toffolis
            res_shots = best.n_prep if best.route in _SAMPLING_ROUTES else 0.0
            norm, flags = problem.norm_a_m, ()
        else:
            routes, res_campaigns, names = (), (), ("not costed", "not costed")
            t_res = t_res_p = res_shots = 0.0
            norm, flags = math.nan, (_FLAG_LS_ONLY,)
        campaigns = res_campaigns + ls_campaigns
        if any(c.per_run is None and 0 < c.n_runs < math.inf for c in campaigns):
            flags += (_FLAG_NO_LAYOUT,)
        out.append(IterationCost(
            k=k, residual=routes, best=names[0], line_search=ls, toffolis=float(t_res + ls.toffolis),
            shots=float(res_shots + ls.n_prep), seconds_one_machine=float(sum(c.seconds(1) for c in campaigns)),
            machines_7d=machines_for_campaigns(campaigns), best_practical=names[1],
            toffolis_practical=float(t_res_p + ls.toffolis), residual_toffolis=float(t_res),
            residual_toffolis_practical=float(t_res_p), line_search_ae=ls_ae, residual_norm_m=float(norm),
            prep_toffolis=float(prep_k.toffolis), ls_flagged=int(flagged), campaigns=campaigns, flags=flags))
    return out


# ------------------------------------------------------------------------- per-system report
@dataclass
class SystemReport:
    """Everything :func:`run_system` computes for one benchmark system (module docstring).

    Attributes:
        name: the system (``ft.symmetry.BENCHMARK_SYSTEMS``).
        params, model: the budgets and the surface-code machine.
        encodings: every scanned ``BlockEncoding``.
        qpe: the selected ``QPECost`` per kind ("df", "thc", "sf"; None if none is admissible)
            and "overall", the QPE of the hand-off and the walk of the residual routes.
        trajectories: family -> {"exact", "second_order"} -> TrajectoryRecord.
        prep: family -> record -> {"elementwise", "leaves"} -> PrepCost.
        iterations: family -> {"measured" (exact record), "classical_first_step"
            (second-order record)} -> list[IterationCost].
        handoff: family -> loop -> {"same_irrep", "any_irrep"} -> list[HandoffRow]; the
            measured and free loops on the exact record, the classical first step on the
            second-order record.
        first_step: family -> the classical first step (T7); one entry per metric.
        cqe_only: family -> loop -> {"cost": CQEOnlyCost, "energy_variance": V_E at Psi_kv,
            "seconds" and "machines_7d" (the loop to k_v plus the final shots), "hf_qpe_seconds"
            (1/p_0 QPE runs), "final_shot" (PhysicalCost of one final-energy shot, None without
            k_v), "lower_bound"} (T8).
        physical: {"qpe": kind -> {budget: PhysicalCost}, "qpe_run": PhysicalCost of the
            hand-off QPE at p_fail_qpe, "shots": family -> [PhysicalCost of a Psi_k shot]}.
        validation: {"replay": family -> {"rows", "rounding"}, "d2e": family -> E''(0) check,
            "global": :func:`global_validation`} (T9).
        crossovers: family -> {"crossover": ||A||*_m of direct_mg against the cheapest AE route at k*,
            "r_scan": rows of the r-scan at k = 2 and k* (h6a and n2)} (T5).
        runtime_s: wall time of :func:`run_system`.
        timings: seconds per component ("build": the active space, irreps and FCI ground state;
            then "qpe", "trajectories", "residual_inputs", "iterations", ... in pipeline order).
        settings: system-level inputs: k* and k_v per record, the residual flag, the energy
            split, C_rot and Q_rot, the variance method, element counts, psi_0 overlaps.
    """

    name: str
    params: FTParameters
    model: SurfaceCodeModel
    encodings: list
    qpe: dict
    trajectories: dict
    prep: dict
    iterations: dict
    handoff: dict
    first_step: dict
    cqe_only: dict
    physical: dict
    validation: dict
    crossovers: dict
    runtime_s: float
    timings: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)


def default_optimizers(name: str, requested: Sequence[str] = tuple(FAMILIES)) -> tuple[str, ...]:
    """The families of ``requested`` that run on ``name``: the L-BFGS families only on h6s, n2s and h8."""
    bad = [f for f in requested if f not in FAMILIES]
    if bad:
        raise ValueError(f"unknown optimizer families {bad}; known: {tuple(FAMILIES)}")
    if name in LBFGS_ONLY_SYSTEMS:
        return tuple(f for f in requested if FAMILIES[f][0] == "lbfgs")
    return tuple(requested)


def _k_last(record: TrajectoryRecord, params: FTParameters) -> int:
    """Last iterate with an iteration cost: k* + 4, and at least k_v - 1 (F17); K - 1 if k* is not reached."""
    k_star = first_iterate_below(record, params.target)
    if k_star is None:
        return record.n_steps - 1
    k_v = first_iterate_below(record, params.e_tol_energy)
    return min(record.n_steps - 1, max(k_star + _K_ITER, (k_v or 0) - 1))


def _k_shown(record: TrajectoryRecord, params: FTParameters) -> int:
    """Last iterate of the per-iterate tables T2 and T9: k* + 2 (8 if k* is not reached), at most K."""
    k_star = first_iterate_below(record, params.target)
    return min(record.n_steps, (_K_UNREACHED if k_star is None else k_star + _K_TABLE))


def _overlap(psi0: np.ndarray, ci: np.ndarray) -> float:
    return abs(complex(inner(psi0, normalize(np.asarray(ci).reshape(psi0.shape))))) ** 2


def _spin_square(active: ActiveSpace, ci: np.ndarray) -> float:
    from pyscf.fci import spin_op

    return float(spin_op.spin_square(np.asarray(ci), active.norb, active.nelec)[0])


def _replay_rows(active: ActiveSpace, record: TrajectoryRecord, prep: PrepCost, psi0: np.ndarray, k_show: int,
                 params: FTParameters, *, rounding: bool) -> dict:
    """Open-loop compiled replay of the first k_show steps against the exact states (T7, T9)."""
    states = replay_elementwise(active, record.directions[:k_show], record.eps[:k_show], irreps=record.irreps,
                                params=params)
    rows = [{"k": k, "c_prep": int(prep.cumulative[k]), "q_prep": int(prep.logical_qubits[k]),
             "error": float(np.linalg.norm(states[k] - record.states[k])), "s2": _spin_square(active, states[k]),
             "p": float(record.overlaps[k]), "p_compiled": _overlap(psi0, states[k]),
             "norm_deficit": float(abs(1.0 - np.linalg.norm(states[k])))} for k in range(k_show + 1)]
    out = {"rows": rows}
    if rounding:
        exact = replay_elementwise(active, record.directions[:k_show], record.eps[:k_show], irreps=record.irreps,
                                   rounded=False, params=params)
        bound = np.concatenate([[0.0], np.cumsum([
            (s.n_double + s.n_single + s.n_diagonal) * math.pi / 2.0 ** s.bits for s in prep.steps[:k_show]])])
        out["rounding"] = [{"k": k, "difference": float(np.linalg.norm(states[k] - exact[k])),
                            "bound": float(bound[k])} for k in range(k_show + 1)]
    return out


def _first_step(active: ActiveSpace, family: str, records: Mapping[str, TrajectoryRecord], prep: PrepCost,
                rows: Sequence[HandoffRow], psi0: np.ndarray, params: FTParameters) -> dict:
    """T7: the classical first step of ``family`` (second-order record), its compiled replay and QPE gain."""
    metric = FAMILIES[family][1]
    classical, exact = records["second_order"], records["exact"]
    so = second_order_step(active, metric=metric, irreps=classical.irreps)
    replay = _replay_rows(active, classical, prep, psi0, 1, params, rounding=False)["rows"][1]
    step0 = prep.steps[0]
    r0, r1 = rows[0], rows[1]

    def ratio(a: float | None, b: float | None) -> float:
        return math.nan if a is None or b is None else a / b

    return {"metric": metric, "eps2": so["eps2"], "eps_exact": so["eps_exact"], "ratio": so["ratio"],
            "energy_lost": so["energy_lost"], "d2e": so["d2e"], "d2e_fd": so["d2e_fd"], "flags": so["flags"],
            "p_hf": float(classical.overlaps[0]), "p1_exact_step": float(exact.overlaps[1]),
            "p1": float(classical.overlaps[1]), "p1_compiled": replay["p_compiled"], "error": replay["error"],
            "s2": replay["s2"], "e1_err": float(classical.errors[1]), "c_step0": int(step0.toffolis),
            "bits0": int(step0.bits), "n_double0": int(step0.n_double), "n_single0": int(step0.n_single),
            "gain_expected": ratio(r0.cost_expected, r1.cost_expected), "certifiable": (r0.certifiable, r1.certifiable),
            "gain_delta": ratio(r0.cost_delta, r1.cost_delta),
            "gain_delta_oracle": ratio(r0.cost_delta_oracle, r1.cost_delta_oracle)}


def _cqe_only(active: ActiveSpace, record: TrajectoryRecord, prep: PrepCost, costs: Sequence[IterationCost],
              loop: str, qpe: QPECost, energy_variance: str, readout_toffolis: int, q_rot: int,
              qpe_run: PhysicalCost | None, params: FTParameters, model: SurfaceCodeModel) -> dict:
    """T8: F17 along ``record`` with its iteration costs, and the wall clock against HF + QPE."""
    k_v = first_iterate_below(record, params.e_tol_energy)
    v_e = ri.energy_variances(active, record.states[k_v])[energy_variance] if k_v is not None else 0.0
    cost = cqe_only_cost(record, prep.cumulative, [c.toffolis for c in costs], [c.shots for c in costs], v_e,
                         readout_toffolis, qpe, params=params, loop=loop)
    out = {"cost": cost, "energy_variance": v_e, "seconds": math.inf, "machines_7d": None,
           "hf_qpe_seconds": math.nan, "final_shot": None,
           "lower_bound": any(_FLAG_LS_ONLY in c.flags for c in costs)}
    if qpe_run is not None:
        out["hf_qpe_seconds"] = qpe_run.seconds / float(record.overlaps[0])
    if k_v is not None:
        final = _sampling_campaign("final_energy", cost.final_shots, prep.cumulative[k_v] + readout_toffolis,
                                   max(int(prep.logical_qubits[k_v]), q_rot), params, model)
        out["final_shot"] = final.per_run
        loop_campaigns = [c for it in costs[:k_v] for c in it.campaigns]
        out["seconds"] = sum(c.seconds(1) for c in loop_campaigns) + final.seconds(1)
        out["machines_7d"] = machines_for_campaigns(loop_campaigns + [final])
    return out


def run_system(name: str, *, optimizers: Sequence[str] | None = None, max_iter: int | None = None,
               residual: bool = True, r_scan: bool | None = None, params: FTParameters = FTParameters(),
               model: SurfaceCodeModel = SurfaceCodeModel()) -> SystemReport:
    """Every M3 cost of the benchmark system ``name`` (module docstring).

    Args:
        name: a key of ``ft.symmetry.BENCHMARK_SYSTEMS``.
        optimizers: families of :data:`FAMILIES`; default :func:`default_optimizers` (all three
            on the core systems, the L-BFGS families on h6s, n2s and h8). Explicit lists are
            run as given; the per-shot variances are taken at iterate 1 of the first of them
            in :data:`FAMILIES` order, so the results do not depend on the list's order.
        max_iter: steps of every trajectory (default the family's).
        residual: cost the residual routes (False: the line search alone, a lower bound on
            C_iter; the default of benchmarks/run_ft_pipeline.py for h8).
        r_scan: run the r-scan of T5 (default: on the systems of :data:`R_SCAN_SYSTEMS`).
        params, model: budgets and the surface-code machine.
    """
    t_start = time.perf_counter()
    timings: dict[str, float] = {}

    def lap(label: str, t0: float) -> float:
        now = time.perf_counter()
        timings[label] = timings.get(label, 0.0) + now - t0
        return now

    families = default_optimizers(name) if optimizers is None else tuple(optimizers)
    default_optimizers(name, families)                       # validates the names
    if not families:
        raise ValueError(f"no optimizer family to run on {name}")
    if max_iter is not None and int(max_iter) < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter!r}")
    t = time.perf_counter()
    active = benchmark_system(name)
    irreps = orbital_irreps(active)
    roots = active.fci(nroots=_N_ROOTS, spin_square=0.0)     # shared by every record (E1, psi_0)
    e0_ref, psi0 = float(roots[0][0]), normalize(np.asarray(roots[1][0]))
    t = lap("build", t)

    reference = ExactReference.build(active)
    encodings = scan_encodings(active, reference=reference, params=params)
    qpe: dict = {}
    for kind in QPE_KINDS + (None,):
        try:
            qpe[kind or "overall"] = select_qpe(encodings, kind=kind, params=params)
        except ValueError:
            qpe[kind or "overall"] = None
    handoff_qpe = qpe["overall"]
    if handoff_qpe is None:
        raise ValueError(f"no admissible block encoding for {name}")
    walk = handoff_qpe.encoding
    t = lap("qpe", t)

    trajectories: dict = {}
    prep: dict = {}
    for fam in families:
        opt, metric, n_default = FAMILIES[fam]
        n_iter = n_default if max_iter is None else int(max_iter)
        trajectories[fam] = {first: exact_trajectory(active, optimizer=opt, metric=metric, max_iter=n_iter,
                                                     system=name, irreps=irreps, n_roots=_N_ROOTS,
                                                     first_step=first, roots=roots)
                             for first in RECORDS}
        prep[fam] = {first: {method: prep_cost(rec.directions, rec.eps, method=method, irreps=irreps, params=params)
                             for method in ("elementwise", "leaves")}
                     for first, rec in trajectories[fam].items()}
    t = lap("trajectories", t)

    primary = trajectories[next(f for f in FAMILIES if f in families)]["exact"]
    energy_split = ri.energy_variances(active, primary.states[0])["best"]
    readout = readout_rotation_toffolis(active.norb, params=params)
    q_rot = readout_rotation_qubits(active.norb, params=params)
    elements = variances = normalizations = None
    settings: dict = {"residual": bool(residual), "energy_split": energy_split, "readout_toffolis": readout,
                      "readout_qubits": q_rot, "families": list(families), "e0": float(e0_ref)}
    if residual:
        elements, _ = ri.allowed_elements(active, params.kind, irreps)
        normalizations = ri.element_normalizations(active, elements, params.kind)
        variances = ri.element_variances(active, primary.states[1], elements, params.kind, params=params)
        settings.update(n_elements=int(len(elements)), variance_method=variances["method"],
                        variance_m=variances["variance_m"], stderr_m=variances["stderr_m"],
                        n_settings=variances["n_settings"], seed=variances["seed"])
    t = lap("residual_inputs", t)

    iterations: dict = {}
    for fam in families:
        iterations[fam] = {}
        for loop, first in (("measured", "exact"), ("classical_first_step", "second_order")):
            rec = trajectories[fam][first]
            iterations[fam][loop] = iteration_costs(
                active, rec, prep[fam][first]["elementwise"], walk, elements=elements, variances=variances,
                normalizations=normalizations, energy_variance=energy_split, readout_toffolis=readout,
                classical_first_step=(loop == "classical_first_step"), k_max=_k_last(rec, params), params=params,
                model=model)
    t = lap("iterations", t)
    crossovers: dict = {}
    if residual:
        for fam in families:
            crossovers[fam] = _crossovers(active, trajectories[fam]["exact"], prep[fam]["exact"]["elementwise"],
                                          iterations[fam]["measured"], walk, elements, variances, normalizations,
                                          handoff_qpe, params,
                                          r_scan=(name in R_SCAN_SYSTEMS) if r_scan is None else bool(r_scan))
    t = lap("crossovers", t)

    handoff: dict = {}
    for fam in families:
        handoff[fam] = {}
        for loop in LOOPS:
            first = "second_order" if loop == "classical_first_step" else "exact"
            rec = trajectories[fam][first]
            costs = None if loop == "free" else [c.toffolis for c in iterations[fam][
                "classical_first_step" if loop == "classical_first_step" else "measured"]]
            handoff[fam][loop] = {e1: handoff_table(rec, prep[fam][first]["elementwise"].cumulative, costs,
                                                    handoff_qpe, loop=loop, use_e1=e1, params=params)
                                  for e1 in ("same_irrep", "any_irrep")}
    t = lap("handoff", t)

    first_step: dict = {}
    for fam in families:
        if any(first_step[f]["metric"] == FAMILIES[fam][1] for f in first_step):
            continue                         # one first step per metric (sd shares lbfgs's)
        first_step[fam] = _first_step(active, fam, trajectories[fam], prep[fam]["second_order"]["elementwise"],
                                      handoff[fam]["classical_first_step"]["same_irrep"], psi0, params)
    t = lap("first_step", t)

    physical: dict = {"qpe": {}, "shots": {}}
    for kind, sel in qpe.items():
        if sel is not None:
            physical["qpe"][kind] = {budget: layout(sel.logical_qubits, sel.toffolis, budget, shot=False, model=model)
                                     for budget in (params.p_fail_qpe, 0.1)}
    physical["qpe_run"] = layout(handoff_qpe.logical_qubits, handoff_qpe.toffolis, params.p_fail_qpe, shot=False,
                                 model=model)
    for fam in families:
        rec, pc = trajectories[fam]["exact"], prep[fam]["exact"]["elementwise"]
        physical["shots"][fam] = [layout(int(pc.logical_qubits[k]), pc.cumulative[k], params.p_fail_shot, shot=True,
                                         model=model) for k in range(_k_shown(rec, params) + 1)]
    cqe_only: dict = {}
    for fam in families:
        cqe_only[fam] = {}
        for loop, first in (("classical_first_step", "second_order"), ("measured", "exact")):
            cqe_only[fam][loop] = _cqe_only(active, trajectories[fam][first], prep[fam][first]["elementwise"],
                                            iterations[fam][loop], loop, handoff_qpe, energy_split, readout, q_rot,
                                            physical["qpe_run"], params, model)
    t = lap("physical_and_cqe_only", t)

    validation: dict = {"replay": {}, "d2e": {}}
    lbfgs_families = [fam for fam in families if FAMILIES[fam][0] == "lbfgs"]
    for fam in lbfgs_families:
        rec = trajectories[fam]["exact"]
        validation["replay"][fam] = _replay_rows(active, rec, prep[fam]["exact"]["elementwise"], psi0,
                                                 _k_shown(rec, params), params, rounding=(fam == lbfgs_families[0]))
    for fam, fs in first_step.items():
        validation["d2e"][fam] = {"d2e": fs["d2e"], "d2e_fd": fs["d2e_fd"],
                                  "relative": abs(fs["d2e"] - fs["d2e_fd"]) / abs(fs["d2e"])}
    t = lap("replay", t)
    validation["global"] = global_validation()
    lap("global_validation", t)

    for fam in families:
        for first, rec in trajectories[fam].items():
            settings.setdefault("k_star", {}).setdefault(fam, {})[first] = first_iterate_below(rec, params.target)
            settings.setdefault("k_v", {}).setdefault(fam, {})[first] = first_iterate_below(rec, params.e_tol_energy)
    return SystemReport(name=name, params=params, model=model, encodings=encodings, qpe=qpe,
                        trajectories=trajectories, prep=prep, iterations=iterations, handoff=handoff,
                        first_step=first_step, cqe_only=cqe_only, physical=physical, validation=validation,
                        crossovers=crossovers, runtime_s=time.perf_counter() - t_start, timings=timings,
                        settings=settings)


def _crossovers(active: ActiveSpace, record: TrajectoryRecord, prep: PrepCost, costs: Sequence[IterationCost],
                walk: BlockEncoding, elements: np.ndarray, variances: dict, normalizations: dict, qpe: QPECost,
                params: FTParameters, *, r_scan: bool) -> dict:
    """T5: ||A||*_m of direct_mg against the cheapest AE route at k*, and the r-scan at k = 2 and k* (``r_scan``)."""
    model_active = walk.model(active)

    def inputs(k: int):
        psi = record.states[k]
        problem = ri.build_problem(active, psi, elements, params.kind, variances=variances,
                                   normalizations=normalizations, deltas=params.aux_deltas)
        bias = ri.factorization_bias(active, model_active, psi, elements, params.kind)
        return problem, PrepParams(prep.cumulative[k], prep.logical_qubits[k]), walk.walk_params(bias)

    k_star = first_iterate_below(record, params.target)
    out: dict = {"k_star": k_star, "crossover": None, "r_scan": []}
    if k_star is not None and k_star < len(costs):
        problem, prep_k, wp = inputs(k_star)
        ae = min((c for c in costs[k_star].residual if c.route in _AE_ROUTES
                  and math.isclose(c.params["kappa"], params.kappa, rel_tol=1e-9)), key=lambda c: c.toffolis)
        out["crossover"] = {"k": k_star, "norm_a_m": problem.norm_a_m, "route": ae.route,
                            "norm_a_m_crossover": crossover_norm(problem, prep_k, wp, "direct_mg", ae.route,
                                                                 params=params)}
    if r_scan:
        for k in sorted({2} | ({k_star} if k_star is not None else set())):
            if k >= len(costs):
                continue
            problem, prep_k, wp = inputs(k)
            t_ls = costs[k].line_search.toffolis
            for r in R_VALUES:
                pr = dataclasses.replace(params, r=r)
                routes = compare_routes(problem, prep_k, wp, params=pr)
                best, best_p = best_route(routes, kappa=pr.kappa), best_route(routes, kappa=pr.kappa_practical)
                ae = min((c for c in routes if c.route in _AE_ROUTES
                          and math.isclose(c.params["kappa"], pr.kappa, rel_tol=1e-9)), key=lambda c: c.toffolis)
                direct = next(c for c in routes if c.route == "direct_mg")
                out["r_scan"].append({"k": k, "r": r, "best": best.route, "best_toffolis": best.toffolis,
                                      "best_practical": best_p.route, "best_practical_toffolis": best_p.toffolis,
                                      "direct_toffolis": direct.toffolis, "best_ae": ae.route,
                                      "best_ae_toffolis": ae.toffolis, "line_search_toffolis": t_ls,
                                      "c_iter_over_qpe": (best.toffolis + t_ls) / qpe.toffolis})
    return out


# OpenFermion 1.8.1 regression cases (df/thc/sf compute_cost_*_test.py): (kind, 2N, lambda), options,
# (C_W, total, qubits); and three cost_estimator cases (Q, Toffolis): the H6 DF QPE run, OpenFermion's
# THC Reiher case and the H6 Psi_6 circuit, costed as a plain run, not a shot
# (tests/test_ft_physical.py::test_matches_openfermion_cost_estimator).
_OF_QPE_CASES = {
    "DF Reiher": (("df", 108, 294.8), {"n_leaves": 360, "n_eigenvectors": 13031, "beta": 16},
                  (21753, 10073183463, 3725)),
    "DF Li": (("df", 152, 1171.2), {"n_leaves": 394, "n_eigenvectors": 20115, "beta": 20},
              (35011, 64410331887, 6405)),
    "THC Reiher": (("thc", 108, 306.3), {"nthc": 350, "beta": 16}, (10912, 5250145120, 2142)),
    "THC Li": (("thc", 152, 1201.5), {"nthc": 450, "beta": 20}, (16923, 31938980976, 2196)),
    "SF Reiher": (("sf", 108, 4258.0), {"n_leaves": 200}, (14184, 94868988984, 3320)),
    "SF Li": (("sf", 152, 3071.8), {"n_leaves": 275}, (24396, 117714920508, 3628)),
}
_OF_PHYSICAL_CASES = ((233, 19_997_593), (2142, 5_250_145_120), (83, 43_427))


def _ae_kappa(t_bits: int, n_offsets: int = 64) -> float:
    """RMS phase error of sine-window phase estimation with M = 2^t - 1 applications, over pi / M (F11).

    The register starts in sum_k sin(pi (k + 1)/(2^t + 1)) |k> (normalized); the outcome law of
    the controlled powers and the inverse QFT is an FFT, averaged over n_offsets phases inside one
    grid cell [derived; Appendix A: 0.974 / 0.993 / 0.998 for M = 63 / 255 / 1023].
    """
    dim, m_apps = 2 ** t_bits, 2 ** t_bits - 1
    k = np.arange(dim)
    amp = np.sin(np.pi * (k + 1) / (dim + 1))
    amp /= np.linalg.norm(amp)
    phi = 2.0 * np.pi * (np.arange(n_offsets) + 0.5) / n_offsets / dim
    prob = np.abs(np.fft.fft(amp * np.exp(1j * np.outer(phi, k)), axis=1)) ** 2 / dim
    err = np.angle(np.exp(1j * (2.0 * np.pi * k[None, :] / dim - phi[:, None])))
    return math.sqrt(float(np.mean(np.sum(prob * err ** 2, axis=1)))) / (math.pi / m_apps)


@functools.lru_cache(maxsize=1)
def global_validation() -> dict:
    """System-independent checks of T9, computed once per process (seconds on one thread).

    * The qubitized-walk model on H4 (``ft.walk.PauliLCUWalk``): the largest deviation of
      <G,psi|W^n|G,psi>, n <= 30, from sum_j p_j T_n(x_j) on HF and the third L-BFGS iterate
      (exact sector spectrum), and the 2R-call Laurent evolution against the dense
      exp(-i tau (H - c0)/lambda) at delta = 1e-4 for tau = 0.5, 3, 7 (error / tail bound).
    * The sine-window amplitude-estimation constant for M = 63, 255, 1023 (:func:`_ae_kappa`).
    * OpenFermion reproduction: ``qpe.openfermion_cost`` on OpenFermion's six regression cases,
      and ``physical.surface_code_cost`` against ``cost_estimator`` on the three cases of
      :data:`_OF_PHYSICAL_CASES` (budget 0.1, d = 7..33).
    """
    import scipy.linalg
    import scipy.special
    from openfermion.resource_estimates.surface_code_compilation.physical_costing import cost_estimator

    from ..cqe import run_cqe
    from ..fci_ops import Hamiltonian
    from ..measure.pauli import QubitState
    from .primitives import jacobi_anger_degree
    from .qpe import openfermion_cost
    from .walk import PauliLCUWalk

    t0 = time.perf_counter()
    a = benchmark_system("h4")
    w = PauliLCUWalk(a)
    ham = Hamiltonian(a)
    sp = ham.space
    dense = np.column_stack([ham.apply(np.eye(sp.size)[i].reshape(sp.shape)).ravel() for i in range(sp.size)])
    dense = 0.5 * (dense + dense.T) + a.ecore * np.eye(sp.size)
    e, v = np.linalg.eigh(dense)
    x = np.clip((e - w.c0) / w.lam, -1.0, 1.0)
    n = np.arange(31)
    states = [normalize(sp.hf()), normalize(np.asarray(run_cqe(a, max_iter=3, optimizer="lbfgs").ci))]

    def qubit(ci: np.ndarray) -> np.ndarray:
        return QubitState.from_ci(a.norb, a.nelec, ci, w.mapping).to_dense()

    moment_error = max(float(np.max(np.abs(w.moments(qubit(ci), 30)
                                           - np.cos(n[:, None] * np.arccos(x)[None, :]) @ ((v.T @ ci.ravel()) ** 2))))
                       for ci in states)
    laurent = []
    gvec = w.prepare(np.eye(1 << w.n_qubits)[0])[:, 0]
    for tau in (0.5, 3.0, 7.0):
        ci = states[1]
        ref_ci = scipy.linalg.expm(-1j * tau * (dense - w.c0 * np.eye(sp.size)) / w.lam) @ ci.ravel()
        ref = w.prepare(qubit(ref_ci.reshape(ci.shape)))
        out = w.laurent_evolution(qubit(ci), tau, 1e-4)
        r_deg = jacobi_anger_degree(tau, 1e-4)
        tail = 2.0 * float(np.sum(np.abs(scipy.special.jv(np.arange(r_deg + 1, r_deg + 200), tau))))
        laurent.append({"tau": tau, "degree": r_deg, "error": float(np.linalg.norm(out - ref)), "tail": tail,
                        "leakage": float(np.linalg.norm(out - np.outer(gvec, gvec @ out)))})
    of_qpe = []
    for case, ((kind, n_so, lam), kw, expected) in _OF_QPE_CASES.items():
        got = openfermion_cost(kind, n_so, lam, 1e-3, chi=10, **kw)[:3]
        of_qpe.append({"case": case, "expected": list(expected), "got": list(got), "match": tuple(got) == expected})
    of_phys = []
    model = SurfaceCodeModel(distances=tuple(range(7, 35, 2)))
    for q, n_tof in _OF_PHYSICAL_CASES:
        cost, prm = cost_estimator(q, n_tof)
        mine = surface_code_cost(q, n_tof, fail_budget=0.1, model=model)
        of_phys.append({"logical_qubits": q, "toffolis": n_tof, "physical_qubits": mine.physical_qubits,
                        "distance": mine.distance, "seconds": mine.seconds,
                        "match": (mine.physical_qubits == cost.physical_qubit_count
                                  and mine.distance == prm.logical_data_qubit_distance
                                  and mine.seconds == cost.duration.total_seconds())})
    return {"walk": {"system": "h4", "n_terms": w.n_terms, "lam": w.lam, "moment_error": moment_error,
                     "laurent": laurent},
            "ae_kappa": [{"t": t, "M": 2 ** t - 1, "kappa": _ae_kappa(t)} for t in (6, 8, 10)],
            "openfermion_qpe": of_qpe, "openfermion_physical": of_phys,
            "seconds": time.perf_counter() - t0}


# ------------------------------------------------------------------------- summaries
def handoff_optima(rows: Sequence[HandoffRow]) -> dict[str, HandoffRow | None]:
    """The optimal hand-off row under every rule of ``ft.handoff.RULES`` (None if no row qualifies)."""
    out: dict[str, HandoffRow | None] = {}
    for rule in RULES:
        try:
            out[rule] = optimal_handoff(rows, rule)
        except ValueError:
            out[rule] = None
    return out


def _direct_toffolis(cost: IterationCost) -> float:
    return next((c.toffolis for c in cost.residual if c.route == "direct_mg"), 0.0)


def loop_totals(costs: Sequence[IterationCost], k_end: int | None, c_qpe: float) -> dict:
    """C_loop(k_end) = sum_{j < k_end} C_iter(j) and its parts (T4); all costed iterations for k_end None.

    Keys: "k_end", "residual_direct" (direct_mg at every iteration), "residual_best" and
    "residual_best_practical", "line_search", "total" and "total_practical" (Toffolis), the same
    over C_QPE ("total_over_qpe", "total_practical_over_qpe"), "line_search_ae" and
    "total_ae_over_qpe" (with the AE line-search option), "shots", "seconds_one_machine",
    "machines_7d" (campaigns of all iterations one after another), "critical_path_s" (the
    longest run of every campaign, the loop's wall clock on unlimited machines) and
    "lower_bound" (line search only).
    """
    n = len(costs) if k_end is None else min(int(k_end), len(costs))
    used = list(costs[:n])
    campaigns = [c for it in used for c in it.campaigns]
    total = sum(it.toffolis for it in used)
    total_p = sum(it.toffolis_practical for it in used)
    return {"k_end": n, "residual_direct": sum(_direct_toffolis(it) for it in used),
            "residual_best": sum(it.residual_toffolis for it in used),
            "residual_best_practical": sum(it.residual_toffolis_practical for it in used),
            "line_search": sum(it.line_search.toffolis for it in used if it.line_search is not None),
            "line_search_ae": sum(it.line_search_ae for it in used if it.line_search is not None),
            "total_ae_over_qpe": sum(it.residual_toffolis + it.line_search_ae for it in used
                                     if it.line_search is not None) / c_qpe,
            "total": total, "total_practical": total_p, "total_over_qpe": total / c_qpe,
            "total_practical_over_qpe": total_p / c_qpe, "shots": sum(it.shots for it in used),
            "seconds_one_machine": sum(c.seconds(1) for c in campaigns),
            "machines_7d": machines_for_campaigns(campaigns),
            "critical_path_s": sum(c.longest_s for c in campaigns if c.n_runs > 0),
            "lower_bound": any(_FLAG_LS_ONLY in it.flags for it in used)}


def first_heisenberg_iterate(costs: Sequence[IterationCost], kappa: float) -> tuple[int, str] | None:
    """(k, route): the first iterate where an AE route at ``kappa`` is cheaper than both sampling routes (T5)."""
    for it in costs:
        if not it.residual:
            continue
        sampling = min(c.toffolis for c in it.residual if c.route in _SAMPLING_ROUTES)
        ae = [c for c in it.residual if c.route in _AE_ROUTES and math.isclose(c.params["kappa"], kappa, rel_tol=1e-9)]
        best = min(ae, key=lambda c: c.toffolis)
        if best.toffolis < sampling:
            return it.k, best.route
    return None


# ------------------------------------------------------------------------- JSON
def _number(x: float) -> float | str | None:
    if math.isnan(x):
        return None
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


# Fields left out of the JSON: CI vectors, directions and model integrals (large arrays).
_SKIP_FIELDS = {"TrajectoryRecord": ("states", "directions"), "BlockEncoding": ("eri_model",)}


def _jsonable(obj):
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return _number(float(obj))
    if isinstance(obj, np.ndarray):
        return [_jsonable(x) for x in obj.tolist()]
    if dataclasses.is_dataclass(obj):
        skip = _SKIP_FIELDS.get(type(obj).__name__, ())
        return {f.name: _jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj) if f.name not in skip}
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    raise TypeError(f"cannot convert {type(obj).__name__} to JSON")


def report_summary(report: SystemReport) -> dict:
    """The headline numbers of a report (golden record and tables): C_QPE, k*, hand-off optima per
    loop and rule, first-step gains, best routes per iterate and loop totals."""
    c_qpe = float(report.qpe["overall"].toffolis)
    out: dict = {"c_qpe": c_qpe, "qpe_kind": report.qpe["overall"].encoding.kind,
                 "c_qpe_kind": {k: (None if q is None else int(q.toffolis)) for k, q in report.qpe.items()},
                 "families": {}}
    for fam in report.trajectories:
        fam_out: dict = {"k_star": report.settings["k_star"][fam], "k_v": report.settings["k_v"][fam],
                         "p": {r: [float(x) for x in rec.overlaps] for r, rec in report.trajectories[fam].items()},
                         "handoff": {}, "best": {}, "best_practical": {}, "totals": {}}
        for loop, tables in report.handoff[fam].items():
            fam_out["handoff"][loop] = {rule: (None if row is None else {"k": row.k, "cost_over_qpe": (
                row.cost_expected if rule.startswith("expected") else
                row.cost_delta if rule == "delta" else row.cost_delta_oracle) / c_qpe})
                for rule, row in handoff_optima(tables["same_irrep"]).items()}
        for loop, costs in report.iterations[fam].items():
            first = "second_order" if loop == "classical_first_step" else "exact"
            fam_out["best"][loop] = [it.best for it in costs]
            fam_out["best_practical"][loop] = [it.best_practical for it in costs]
            fam_out["totals"][loop] = loop_totals(costs, report.settings["k_star"][fam][first], c_qpe)
        if fam in report.first_step:
            fs = report.first_step[fam]
            fam_out["first_step"] = {key: fs[key] for key in ("ratio", "p1", "p1_compiled", "gain_expected",
                                                               "gain_delta", "gain_delta_oracle", "c_step0")}
        out["families"][fam] = fam_out
    return out


def to_json(report: SystemReport) -> dict:
    """A strict-JSON dict of ``report`` (module docstring): nan -> null, +-inf -> "inf" / "-inf",
    arrays -> lists, dataclasses -> dicts; CI vectors, directions and model integrals are left out.
    ``json.loads(json.dumps(to_json(r), allow_nan=False)) == to_json(r)``. The key "summary" holds
    :func:`report_summary`."""
    out = {f.name: _jsonable(getattr(report, f.name)) for f in dataclasses.fields(report)}
    out["summary"] = _jsonable(report_summary(report))
    return out


# ------------------------------------------------------------------------- tables
def _fmt(x, spec: str = ".3g") -> str:
    """x formatted with ``spec``; '-' for None and nan, 'inf' for infinity; exponents as 1.2e7."""
    if x is None:
        return "-"
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)) and not spec.endswith(("e", "g")):
        return format(int(x), spec)
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


def _sci(x) -> str:
    return _fmt(x, ".2g")


def _duration(seconds: float) -> str:
    """Seconds as s, min, h or d (2 significant digits)."""
    if seconds is None or not math.isfinite(seconds):
        return _fmt(seconds)
    for unit, scale in (("d", 86400.0), ("h", 3600.0), ("min", 60.0)):
        if seconds >= scale:
            return f"{_fmt(seconds / scale, '.2g')} {unit}"
    return f"{_fmt(seconds, '.2g')} s"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines + [""]


def _route(costs: Sequence[RouteCost], route: str, kappa: float | None = None) -> RouteCost | None:
    for c in costs:
        if c.route == route and (kappa is None or math.isclose(c.params.get("kappa", kappa), kappa, rel_tol=1e-9)):
            return c
    return None


def _st(c: RouteCost | None) -> str:
    """'n_prep / Toffolis' of a route ('infeasible' or '-')."""
    if c is None:
        return "-"
    if "infeasible" in c.flags:
        return "infeasible"
    return f"{_sci(c.n_prep)} / {_sci(c.toffolis)}"


def _delta(c: RouteCost | None) -> str:
    return "" if c is None or c.params.get("delta") is None else f" [{_fmt(c.params['delta'], 'g')}]"


def _k_label(k: int, k_star: int | None) -> str:
    return f"{k}*" if k == k_star else str(k)


def _t1(reports: Sequence[SystemReport]) -> list[str]:
    rows = []
    for rep in reports:
        for kind in QPE_KINDS:
            sel = rep.qpe.get(kind)
            if sel is None:
                rows.append([rep.name, kind, "none admissible"] + [""] * 11)
                continue
            enc = sel.encoding
            p = enc.params
            if kind == "df":
                choice = f"thr {_fmt(p['threshold'], '.0e')}, L {p['L']}, Xi {p['Lxi']}"
            elif kind == "thc":
                choice = f"M {p['M']}, pen. {p['penalty'] if p['penalty'] == 'auto' else _fmt(p['penalty'], 'g')}"
            else:
                choice = f"L {p['L']}"
            phys = rep.physical["qpe"][kind]
            cells = [f"{_fmt(c.physical_qubits, ',')} / {c.distance} / {_fmt(c.seconds / 60, '.3g')}" if c else "-"
                     for c in (phys[rep.params.p_fail_qpe], phys[0.1])]
            rows.append([rep.name, kind + (" (selected)" if rep.qpe["overall"].encoding is enc else ""), choice,
                         _fmt(enc.lam, ".6g"), _fmt(enc.de_fact * 1e3, ".2g"), _sci(enc.infidelity),
                         str(enc.toffoli_per_step), _fmt(sel.iterations, ","), _fmt(sel.toffolis, ".3g"),
                         str(sel.logical_qubits), _fmt(enc.qr_clamped), _fmt(enc.lam * enc.toffoli_per_step, ".4g"),
                         *cells])
    head = ["system", "kind", "selection", "lambda (Ha)", "dE_fact (mHa)", "infidelity", "C_W", "iterations", "C_QPE",
            "Q", "QR clamp", "lambda C_W", f"budget {_fmt(reports[0].params.p_fail_qpe, 'g')}: qubits / d / min",
            "budget 0.1: qubits / d / min"]
    return ["## T1. QPE (dE = 1 mHa, selection rule per kind)", "",
            "lambda C_W (Ha x Toffolis) sets C_QPE = C_W ceil(pi lambda / (2 dE)); '(selected)' is the cheapest "
            "admissible encoding overall, the QPE of the hand-off and the walk of the residual routes. "
            "Physical: OpenFermion's surface-code model (p = 1e-3, 4 factories), one QPE run.", ""] + _table(head, rows)


def _t2(reports: Sequence[SystemReport]) -> list[str]:
    rows = []
    for rep in reports:
        for fam, recs in rep.trajectories.items():
            rec = recs["exact"]
            el, lv = rep.prep[fam]["exact"]["elementwise"], rep.prep[fam]["exact"]["leaves"]
            k_star = rep.settings["k_star"][fam]["exact"]
            shots = rep.physical["shots"][fam]
            for k in range(_k_shown(rec, rep.params) + 1):
                st = el.steps[k] if k < len(el.steps) else None
                lst = lv.steps[k] if k < len(lv.steps) else None
                rows.append([rep.name, fam, _k_label(k, k_star), _fmt(rec.errors[k] * 1e3, ".3f"),
                             _fmt(rec.overlaps[k], ".4f"), _fmt(rec.residual_norms[k], ".4f"),
                             str(int(el.cumulative[k])), str(int(lv.cumulative[k])), str(int(el.logical_qubits[k])),
                             _fmt(shots[k].seconds, ".3g") if shots[k] else "-",
                             f"{st.n_double} / {st.n_single}" if st else "", str(st.bits) if st else "",
                             str(st.toffolis) if st else "", str(lst.toffolis) if lst else ""])
    head = ["system", "family", "k", "E_k - E_0 (mHa)", "p_k", "norm A_k", "C(Psi_k)", "C(Psi_k) leaves", "Q(Psi_k)",
            "s / shot", "step k: n_d / n_s", "b", "C_step", "C_step leaves"]
    return ["## T2. State preparation (exact records, k <= k* + 2)", "",
            "k* (marked *) is the first iterate within 1.6 mHa of E_0. Step k compiles exp(eps_k D_k), the step "
            "from Psi_k to Psi_{k+1}, elementwise (F3; delta_step = 1e-3) and as DF leaves (F6, order 1). "
            "s / shot: one shot of Psi_k at budget 1e-5 with 2d readout cycles.", ""] + _table(head, rows)


def _t3(reports: Sequence[SystemReport]) -> list[str]:
    rows = []
    for rep in reports:
        c_qpe = float(rep.qpe["overall"].toffolis)
        kp = rep.params.kappa_practical
        for fam, loops in rep.iterations.items():
            k_star = rep.settings["k_star"][fam]["exact"]
            for it in loops["measured"]:
                r = it.residual
                ls = it.line_search
                ls_cell = "-" if ls is None else f"{_sci(ls.n_prep)} / {_sci(ls.toffolis)} [{it.ls_flagged}]"
                ls_ae = "-" if ls is None else _sci(it.line_search_ae)
                if r:
                    aux, aea = _route(r, "aux_mg"), _route(r, "ae_aux", rep.params.kappa)
                    k2 = " / ".join(_sci(_route(r, x, kp).toffolis) for x in _AE_ROUTES)
                    gaw = " / ".join(_sci(c.toffolis) for c in r if c.route == "gaw")
                    cells = [_fmt(it.residual_norm_m, ".3g"), _st(_route(r, "direct_mg")), _st(aux) + _delta(aux),
                             _st(_route(r, "ae_product", rep.params.kappa)), _st(_route(r, "ae_x", rep.params.kappa)),
                             _st(aea) + _delta(aea), k2, gaw]
                else:
                    cells = ["-"] * 8
                rows.append([rep.name, fam, _k_label(it.k, k_star), str(int(it.prep_toffolis)), *cells,
                             f"{it.best} / {it.best_practical}", ls_cell, ls_ae,
                             f"{_fmt(it.toffolis / c_qpe, '.3g')} / {_fmt(it.toffolis_practical / c_qpe, '.3g')}"])
    head = ["system", "family", "k", "C(Psi_k)", "norm A_m", "direct_mg S / T", "aux_mg S / T [delta]",
            "ae_product N / T", "ae_x N / T", "ae_aux N / T [delta]", "kappa 2: product / x / aux T",
            "GAW T, kappa_g 1 / 10 / 100", "best kappa 1 / 2", "line search S / T [flagged]",
            "line search, AE energies T", "C_iter / C_QPE, kappa 1 / 2"]
    return ["## T3. Residual and line search per iteration (measured loop, exact records)", "",
            "S: shots, N: applications of the Hadamard-test circuit, T: Toffolis; AE routes at kappa = 1 unless "
            "marked. r = 0.3 over the full tensor, spin-free symmetry-allowed elements. [delta]: the aux routes' "
            "optimal delta. Line search: 11 candidates at +-1.6 mHa each, S the shots of all 11 (11 S_E); [flagged]: candidates whose F3 rounding "
            "bound exceeds delta_step; AE energies: the F13 option (energies of H' by amplitude estimation on the "
            "walk, kappa = 1), not part of C_iter. GAW (F12) is a band and never selected. C_iter = T(best) + "
            "T(line search).",
            ""] + _table(head, rows)


def _t4(reports: Sequence[SystemReport]) -> list[str]:
    rows = []
    for rep in reports:
        c_qpe = float(rep.qpe["overall"].toffolis)
        for fam, loops in rep.iterations.items():
            for loop, costs in loops.items():
                first = "second_order" if loop == "classical_first_step" else "exact"
                k_star = rep.settings["k_star"][fam][first]
                tot = loop_totals(costs, k_star, c_qpe)
                part = ">= " if k_star is None else ""          # k* not reached: partial sums
                lb = ">= " if tot["lower_bound"] or k_star is None else ""
                res = ["-"] * 3 if tot["lower_bound"] else [part + _sci(tot[x]) for x in (
                    "residual_direct", "residual_best", "residual_best_practical")]
                rows.append([rep.name, fam, loop, _fmt(k_star) if k_star is not None else f"- ({tot['k_end']} costed)",
                             *res, part + _sci(tot["line_search"]), lb + _sci(tot["total"]),
                             f"{lb}{_fmt(tot['total_over_qpe'], '.3g')} / "
                             f"{lb}{_fmt(tot['total_practical_over_qpe'], '.3g')}",
                             lb + _fmt(tot["total_ae_over_qpe"], ".3g"), lb + _duration(tot["seconds_one_machine"]),
                             lb + _duration(tot["critical_path_s"]),
                             "none" if tot["machines_7d"] is None else lb + str(tot["machines_7d"])])
    head = ["system", "family", "loop", "k*", "sum residual, direct only", "sum residual, best kappa 1",
            "best kappa 2", "sum line search", "total (kappa 1)", "total / C_QPE, kappa 1 / 2",
            "total / C_QPE, AE line search", "one machine", "critical path", "machines for 7 days"]
    return ["## T4. Totals to chemical accuracy, C_loop(k*)", "",
            "Sums over the iterations j < k* (Toffolis). 'classical_first_step' costs the record with the classical "
            "eps_2 at HF and C_iter(0) = 0. Wall clock (F19): every campaign of the loop one after another on one "
            "machine; critical path: the longest run of every campaign, the wall clock on unlimited machines; "
            "machines: the smallest number that finishes the loop in 7 days, amplitude-estimation runs in LPT order "
            "('none': the critical path exceeds 7 days). '>=': a lower bound, either line search only (h8 without the "
            "residual) or k* not reached (n2s: the sums run over the costed iterations).",
            ""] + _table(head, rows)


def _t5(reports: Sequence[SystemReport]) -> list[str]:
    rows, scan = [], []
    for rep in reports:
        for fam, loops in rep.iterations.items():
            costs = loops["measured"]
            if not any(it.residual for it in costs):
                continue
            first = [first_heisenberg_iterate(costs, kappa) for kappa in (rep.params.kappa, rep.params.kappa_practical)]
            cr = rep.crossovers.get(fam, {}).get("crossover")
            rows.append([rep.name, fam, *("-" if f is None else f"{f[0]} ({f[1]})" for f in first),
                         _fmt(rep.settings["k_star"][fam]["exact"]),
                         "-" if cr is None else _fmt(cr["norm_a_m"], ".3g"),
                         "-" if cr is None else f"{_fmt(cr['norm_a_m_crossover'], '.3g')} ({cr['route']})"])
            for row in rep.crossovers.get(fam, {}).get("r_scan", []):
                scan.append([rep.name, fam, str(row["k"]), _fmt(row["r"], "g"),
                             f"{row['best']} ({_sci(row['best_toffolis'])})",
                             f"{row['best_practical']} ({_sci(row['best_practical_toffolis'])})",
                             _sci(row["direct_toffolis"]), f"{row['best_ae']} ({_sci(row['best_ae_toffolis'])})",
                             _sci(row["line_search_toffolis"]), _fmt(row["c_iter_over_qpe"], ".3g")])
    head = ["system", "family", "first k: AE < sampling, kappa 1", "kappa 2", "k*", "norm A_m at k*",
            "norm A*_m (direct_mg = best AE) at k*"]
    head2 = ["system", "family", "k", "r", "best kappa 1 (T)", "best kappa 2 (T)", "direct_mg T", "best AE kappa 1 (T)",
             "line search T", "C_iter / C_QPE"]
    return ["## T5. Crossovers and r-scan", "",
            "First iterate of the measured loop at which an amplitude-estimation route is cheaper than both sampling "
            "routes; norm A*_m: the residual norm at which direct_mg and the cheapest AE route at k* cost the same "
            "(A scaled with biases, variances and circuits fixed; residual.crossover_norm): sampling is cheaper "
            "above it. r-scan: the relative residual error r at k = 2 and k*.",
            ""] + _table(head, rows) + (_table(head2, scan) if scan else [])


def _t6(reports: Sequence[SystemReport]) -> list[str]:
    rows, opt = [], []
    for rep in reports:
        c_qpe = float(rep.qpe["overall"].toffolis)
        for fam, loops in rep.handoff.items():
            if FAMILIES[fam][0] != "lbfgs":
                continue
            for loop, tables in loops.items():
                same, other = tables["same_irrep"], tables["any_irrep"]
                costs = rep.iterations[fam].get("measured" if loop == "measured" else "classical_first_step", [])
                # C_loop(k) is a lower bound from the first iteration that costs the line search only.
                lower_from = next((it.k + 1 for it in costs if _FLAG_LS_ONLY in it.flags), math.inf) \
                    if loop != "free" else math.inf
                for r, r_any in zip(same[:4], other[:4]):
                    if loop != "measured" and r.k == 0:
                        continue                       # the HF row is the same in every loop
                    lb = ">= " if r.k >= lower_from else ""
                    rows.append([rep.name, fam, loop if r.k else "all", str(r.k), _fmt(r.p, ".4f"),
                                 f"{_fmt(r.p_lb, '.3f')} / {_fmt(r_any.p_lb, '.3f')}", _fmt(r.certifiable),
                                 _fmt(r.saving_vs_hf / c_qpe, ".3g"), lb + _fmt(r.cost_expected / c_qpe, ".4g"),
                                 f"{lb}{_fmt(None if r.cost_delta is None else r.cost_delta / c_qpe, '.4g')} / "
                                 f"{lb}{_fmt(r.cost_delta_oracle / c_qpe, '.4g')}",
                                 ("<= " if lb else "") + _fmt(r.p0_break_even, ".3g"), ", ".join(r.flags) or "-"])
                best = handoff_optima(same)
                cells = []
                for rule in RULES:
                    row = best[rule]
                    value = None if row is None else (row.cost_expected if rule.startswith("expected") else
                                                      row.cost_delta if rule == "delta" else row.cost_delta_oracle)
                    cells.append("-" if row is None else f"{row.k} ({_fmt(value / c_qpe, '.4g')})")
                breakeven = " / ".join(("<= " if k >= lower_from else "") + _fmt(same[k].p0_break_even, ".3g")
                                       if k < len(same) else "-" for k in (1, 2, 3))
                e = best["expected"]
                if e is None:
                    verdict = "-"
                elif e.k == 0:
                    verdict = "HF"
                else:
                    gain = _fmt(same[0].cost_expected / e.cost_expected, ".4g")
                    verdict = f"Psi_{e.k}, T_E(0)/T_E(k_opt) = {gain}"
                    if not same[0].certifiable:
                        verdict += " (HF not certified)"
                if math.isfinite(lower_from) and len(same) > lower_from:
                    verdict += f" (C_loop a lower bound from k = {lower_from})"
                opt.append([rep.name, fam, loop, str(len(same) - 1), *cells, breakeven, verdict])
    head = ["system", "family", "loop", "k", "p_k", "p_lb same / any irrep", "rule E certified", "saving / C_QPE",
            "T_E / C_QPE", "T_delta / C_QPE, certified / oracle", "p0*(k)", "flags"]
    head2 = ["system", "family", "loop", "rows to k", "k_opt E certified", "k_opt E oracle", "k_opt delta certified",
             "k_opt delta oracle", "p0*(1) / p0*(2) / p0*(3)", "verdict (rule E, certified)"]
    return ["## T6. Hand-off to QPE (L-BFGS families)", "",
            "T_E(k) = C_loop(k) + (C(Psi_k) + C_QPE)/p_k; T_delta(k) = C_loop(k) + R_delta(p) (C(Psi_k) + C_QPE) with "
            "R_delta(p) = ceil(ln 0.01 / ln(1 - p)) at the certified p_lb (Eckart-Temple, same-irrep E1) or the exact "
            "p_k (oracle). Rule E is certified iff E1 - E_k > 3 dE. saving = C_QPE/p_0 - (C(Psi_k) + C_QPE)/p_k, "
            "bounded by C_QPE (1/p_0 - 1); p0*(k) = C_QPE / T_E(k) is the HF overlap below which handing off Psi_k "
            "beats HF. 'measured' and 'free' use the exact records, 'classical_first_step' the second-order "
            "records. '>=': C_loop from line-search-only iteration costs (h8 without the residual), a lower bound. "
            "Rows k <= 3:", ""] + _table(head, rows) + [
            "k_opt per rule (T / C_QPE at k_opt); 'rows to k': the last iterate the loop covers:", ""] + _table(head2, opt)


def _t7(reports: Sequence[SystemReport]) -> list[str]:
    rows = []
    for rep in reports:
        for fs in rep.first_step.values():
            cert = "/".join("yes" if c else "no" for c in fs["certifiable"])
            rows.append([rep.name, fs["metric"], _fmt(fs["ratio"], ".4f"), _fmt(fs["energy_lost"] * 1e3, ".3g"),
                         _fmt(fs["p_hf"], ".4f"), _fmt(fs["p1_exact_step"], ".5f"), _fmt(fs["p1"], ".5f"),
                         _fmt(fs["p1_compiled"], ".5f"), _fmt(fs["error"], ".3g"), _fmt(fs["s2"], ".2g"),
                         f"{fs['c_step0']} ({fs['n_double0']} / {fs['n_single0']}, b = {fs['bits0']})",
                         f"{_fmt(fs['gain_expected'], '.4f')} ({cert})", _fmt(fs["gain_delta"], ".4g"),
                         _fmt(fs["gain_delta_oracle"], ".4g")])
    head = ["system", "metric", "eps_2 / eps*", "E(eps_2) - E(eps*) (mHa)", "p_HF", "p_1 (eps*)", "p_1 (eps_2)",
            "p~_1 compiled", "norm(Psi~_1 - Psi_1)", "<S^2>(Psi~_1)", "C_step(0) (n_d / n_s, b)",
            "gain rule E (certified k = 0/1)", "gain rule delta", "gain rule delta, oracle"]
    return ["## T7. The classical first step (F16)", "",
            "eps_2 = -E'(0)/E''(0) at HF along the masked d_0 (sd shares the spin-free step of lbfgs). p_1 (eps*): "
            "after the exact line search; p_1 (eps_2): after the classical step, exact unitary; p~_1: its compiled "
            "elementwise circuit (open-loop replay, rounded). Gain: T(0)/T(1) of the classical-first-step loop "
            "(C_loop(1) = 0) under rule E and rule delta.", ""] + _table(head, rows)


def _t8(reports: Sequence[SystemReport]) -> list[str]:
    rows = []
    for rep in reports:
        for fam, loops in rep.cqe_only.items():
            for loop, entry in loops.items():
                c: CQEOnlyCost = entry["cost"]
                lb = ">= " if entry["lower_bound"] else ""
                ratio_s = entry["seconds"] / entry["hf_qpe_seconds"] if entry["hf_qpe_seconds"] else math.nan
                cells = ["-"] * 6 if c.k_v is None else [
                    lb + _sci(c.shots), _sci(c.final_shots), lb + _sci(c.toffolis), lb + _fmt(c.ratio_to_hf_qpe, ".3g"),
                    lb + _duration(entry["seconds"]), lb + _fmt(ratio_s, ".3g")]
                rows.append([rep.name, fam, loop, _fmt(c.k_v), _fmt(c.e_err * 1e3, ".3f"), *cells[:5],
                             _duration(entry["hf_qpe_seconds"]), cells[5]])
    head = ["system", "family", "loop", "k_v", "E_kv - E_0 (mHa)", "shots", "final shots", "Toffolis",
            "Toffolis / (C_QPE / p_0)", "wall clock, one machine", "HF + QPE, one machine", "wall-clock ratio"]
    return ["## T8. CQE-only energy (F17)", "",
            "k_v: first iterate within 1 mHa of E_0; then V_E / sigma^2 shots at sigma = 0.3 mHa (exact integrals, DF "
            "grouping and readout rotation). HF + QPE: 1/p_0 QPE runs on average (budget 1e-2 per run). k_v '-': "
            "not reached in the run (n2s), E_kv - E_0 is then the error of the last iterate and F17 has no cost. "
            "'>=': line search only (h8 without the residual).",
            ""] + _table(head, rows)


def _t9(reports: Sequence[SystemReport]) -> list[str]:
    rows, checks = [], []
    for rep in reports:
        for fam, rep_rows in rep.validation["replay"].items():
            k_star = rep.settings["k_star"][fam]["exact"]
            for r in rep_rows["rows"]:
                rows.append([rep.name, fam, _k_label(r["k"], k_star), str(r["c_prep"]), _fmt(r["error"], ".3g"),
                             _fmt(r["s2"], ".2g"), _fmt(r["p"], ".5f"), _fmt(r["p_compiled"], ".5f")])
        fam0 = next(iter(rep.validation["replay"]), None)
        rnd = rep.validation["replay"][fam0].get("rounding", []) if fam0 else []
        worst = max(rnd, key=lambda x: x["difference"] / x["bound"] if x["bound"] else 0.0, default=None)
        deficit = max((r["norm_deficit"] for v in rep.validation["replay"].values() for r in v["rows"]), default=None)
        flagged = [(it.ls_flagged, it.k) for loops in rep.iterations.values() for it in loops["measured"]
                   if it.line_search is not None]
        d2e = "; ".join(f"{fam} {_fmt(v['relative'], '.1e')}" for fam, v in rep.validation["d2e"].items())
        checks.append([rep.name, "-" if worst is None else
                       f"{_fmt(worst['difference'], '.2g')} <= {_fmt(worst['bound'], '.2g')} (k = {worst['k']})",
                       _fmt(deficit, ".1e"), d2e,
                       f"{sum(f for f, _ in flagged)} of {11 * len(flagged)} ({sum(1 for f, _ in flagged if f)} of "
                       f"{len(flagged)} iterations)",
                       _fmt(rep.qpe["overall"].encoding.qr_clamped)])
    g = reports[0].validation["global"]
    w = g["walk"]
    glob = [["walk moments n <= 30, H4 (HF, L-BFGS iterate 3)", f"max deviation {_fmt(w['moment_error'], '.1e')}"]]
    for x in w["laurent"]:
        glob.append([f"Laurent evolution tau = {_fmt(x['tau'], 'g')}, delta = 1e-4 (R = {x['degree']})",
                     f"error {_fmt(x['error'], '.2g')} <= tail {_fmt(x['tail'], '.2g')} "
                     f"(ratio {_fmt(x['error'] / x['tail'], '.2f')}); leakage {_fmt(x['leakage'], '.1e')}"])
    glob += [["sine-window AE constant kappa, M = " + " / ".join(str(x["M"]) for x in g["ae_kappa"]),
              " / ".join(_fmt(x["kappa"], ".3f") for x in g["ae_kappa"])]]
    glob += [["OpenFermion QPE costs (6 regression cases)",
              f"{sum(x['match'] for x in g['openfermion_qpe'])} of {len(g['openfermion_qpe'])} identical"]]
    glob += [["OpenFermion cost_estimator (3 cases: H6 DF QPE, THC Reiher, H6 Psi_6 circuit)",
              f"{sum(x['match'] for x in g['openfermion_physical'])} of {len(g['openfermion_physical'])} identical"]]
    head = ["system", "family", "k", "C(Psi_k)", "norm(Psi~_k - Psi_k)", "<S^2>(Psi~_k)", "p_k", "p~_k"]
    head2 = ["system", "rounding part, worst ratio (difference <= bound)", "norm deficit", "E''(0) vs finite diff.",
             "line-search candidates flagged", "QR clamp (selected)"]
    return ["## T9. Validation", "",
            "Open-loop replay of the compiled elementwise circuits (rounded, F3) of the exact L-BFGS records "
            "(testing.fock oracle): state error, <S^2> (0 for the exact iterates) and overlaps, to k* + 2:", ""] + \
        _table(head, rows) + ["Per system: rounding part of the replay error against sum_j A_j pi / 2^b_j (first "
                              "family), the norm deficit (leakage out of (n_alpha, n_beta)), E''(0) against "
                              "Richardson differences (relative), the F3 rounding flag on line-search candidates:",
                              ""] + _table(head2, checks) + ["Model checks (system independent):", ""] + \
        _table(["check", "result"], glob)


def markdown_tables(reports: Sequence[SystemReport]) -> str:
    """Tables T1-T9 of docs/m3-results.md for the given reports, as markdown."""
    if not reports:
        raise ValueError("need at least one report")
    lines: list[str] = []
    for part in (_t1, _t2, _t3, _t4, _t5, _t6, _t7, _t8, _t9):
        lines += part(reports)
    return "\n".join(lines)
