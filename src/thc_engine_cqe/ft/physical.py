"""Surface-code footprint and wall clock of fault-tolerant circuits (F18, F19).

Every cost of :mod:`thc_engine_cqe.ft` ends as a count of logical qubits Q and Toffolis
N_Tof (one CCZ state each). This module turns a circuit (Q, N_Tof) into physical qubits,
a code distance and seconds with OpenFermion's surface-code model, and a number of runs
of such a circuit into a campaign wall clock. Tags: [OF] OpenFermion 1.8.1,
``resource_estimates/surface_code_compilation/physical_costing.py`` (transcribed here
with floats, reproduced exactly by tests/test_ft_physical.py); [derived], [assumption]
as in the M3 specification.

Layout model (F18) [OF]
-----------------------
A machine has F magic-state factories of one type and a data block at code distance d.
For each known factory (the two-level T factory of OpenFermion's
``_two_level_t_state_factory_1p1000``, only at p = 1e-3, then every AutoCCZ(l1, l2) with
l1 = 5, 7, ..., 23 and l2 = l1 + 2, ..., 39, in OpenFermion's order) and each d in
``SurfaceCodeModel.distances``:

    storage = ceil((1 + routing) Q)                          logical patches
    area    = storage * 2 (d + 1)^2 + F * A_fac              physical qubits
    cycles  = floor(N_Tof / F * rounds_fac)                  factory-limited duration
              (+ 2 d readout cycles for a shot)
    seconds = cycles * t_cycle
    failure = min(1, f_fac N_Tof + 0.1 (100 p)^((d + 1)/2) * storage * cycles)

with A_fac, rounds_fac and f_fac the footprint, cycles per magic state and failure per
state of the factory. Layouts with failure <= budget are kept, and the one with the
smallest area * cycles wins (the first one on ties, as in OpenFermion). Assumptions of
the model [OF]: routing reaches the factories as fast as they run, and the run is
limited by magic-state production (one Toffoli per state).

Differences from ``physical_costing.cost_estimator`` (which this module reproduces
exactly, physical qubits, distance and seconds, for its own settings: p = 1e-3, a 1 us
cycle, F = 4, 50% routing, d = 7..33, budget 0.1):

* Seconds are floats. ``cost_estimator`` multiplies a ``datetime.timedelta`` by the
  qubit count, which overflows (``timedelta.max`` is about 8.6e19 us) near 1e11-1e12
  Toffolis; here 1e13 Toffolis are fine.
* The failure budget, the distances, F, the routing overhead and the cycle time are
  parameters (:class:`SurfaceCodeModel`); the default distances reach d = 43.
* ``shot=True`` adds 2 d cycles of logical readout after the last Toffoli
  [assumption], which also count in the data failure. A Clifford-only shot (N_Tof = 0)
  thus lasts 2 d cycles; it needs no factory, so its area is the data block alone
  (factory "none").
* An optional reaction floor [assumption]: with ``SurfaceCodeModel.reaction_us`` set,
  the Toffoli part of a run lasts at least ceil(toffoli_depth * t_react / t_cycle)
  cycles (Toffolis that depend on measured outcomes cannot outrun the classical
  reaction time), and the readout of a shot follows it:
  cycles = max(rounds, floor) + 2 d. The data failure is evaluated on these cycles.
* N_Tof must be 0 or at least 1: OpenFermion's fluid floor(N_Tof / F * rounds_fac)
  gives a run of zero cycles for 0 < N_Tof < F / rounds_fac.

Observed with the defaults (p = 1e-3, 4 factories): the factory-limited rate is
rounds_fac / F = 26.25 us per Toffoli for AutoCCZ(.., l2 = 21) and 28.75 us for l2 = 23
(:attr:`PhysicalCost.us_per_toffoli`); the H6 QPE run (233 logical qubits, 1.9998e7
Toffolis) needs 956,160 physical qubits at d = 23 and 574.9 s at budget 0.1, and
1,026,160 qubits at d = 25 at 0.01; one H6 Psi_6 shot (83 logical qubits, 43,427
Toffolis) 721,960 qubits at d = 25 and 1.249 s at 1e-5 (spec Appendix A, tested). The
best factories fail with 3.6e-16 (two-level T) and 3.66e-14 (AutoCCZ(23, 39)) per state
at p = 1e-3, so a budget is infeasible beyond about budget / 3.6e-16 Toffolis
(:func:`surface_code_cost` raises; at budget 0.1 the T factory carries 1e13 and 1e14
Toffolis at d = 35 and 37, 46.5 us per Toffoli).

Monotonicity [derived]. Every layout's failure grows with N_Tof and the feasible set
shrinks as the budget tightens, so the optimal area x duration is non-decreasing in
N_Tof and in 1/budget, and for a fixed factory the optimal d is the smallest feasible
one, also non-decreasing. The chosen d itself is monotone in neither, because the
optimum can switch factories (specification section 7 claims both monotonicities; both
are false in OpenFermion's own model, tests/test_ft_physical.py):

* in N_Tof, to a faster factory: 233 logical qubits at budget 0.1 need d = 15 with
  AutoCCZ(9, 15) for 562 Toffolis but d = 13 with AutoCCZ(11, 13) for 1000, a shorter
  run with less data failure (``cost_estimator`` makes the same choice);
* in 1/budget, to a cleaner factory: one shot of 83 logical qubits and 5623 Toffolis
  needs d = 25 with AutoCCZ(13, 23) at budget 3e-5 but d = 23 with AutoCCZ(15, 23) at
  1e-5, whose smaller distillation failure leaves more of the budget to the data block.

Wall clock (F19) [assumption]
-----------------------------
A campaign of n independent runs of one circuit on M identical machines runs whole
runs, each on one machine, one after another (a run is not split; shots and
per-element amplitude-estimation runs parallelize, a single QPE run does not). It
lasts

    T(n, M) = t_run * ceil(n / M)                              (campaign_seconds)

with n rounded up to whole runs (a fractional count from a variance formula needs
ceil(n) runs). For n >= 1 the even (fluid) share t_run max(n / M, 1) is a lower bound,
and T exceeds it by less than one run: by up to 2x when n / M is just above 1 (5 runs
on 4 machines take 2 t_run, not 1.25 t_run), negligibly when n / M >> 1, as for shots.
An infinite n (an infeasible route) lasts forever. A random number of runs with mean n
(e.g. 1/p QPE repetitions) takes n t_run on one machine on average, not T(n, 1). The
machines needed for a deadline D are the smallest M >= 1 with T(n, M) <= D,
M = ceil(n / floor(D / t_run)) (:func:`machines_for`); None if no number suffices
(one run is longer than D, or n is infinite).
"""

from __future__ import annotations

import functools
import math
import numbers
from dataclasses import dataclass

__all__ = [
    "SurfaceCodeModel",
    "PhysicalCost",
    "surface_code_cost",
    "campaign_seconds",
    "machines_for",
]

SECONDS_PER_WEEK = 7 * 86400


@dataclass(frozen=True)
class SurfaceCodeModel:
    """Machine parameters of the surface-code model (F18) [OF defaults].

    Attributes:
        p_phys: physical error rate per operation. OpenFermion's two-level T factory
            exists only at exactly 1e-3; other rates use the AutoCCZ factories alone.
        cycle_us: surface-code cycle time (microseconds).
        n_factories: magic-state factories F running in parallel.
        routing_overhead: extra logical patches for routing, as a fraction of Q.
        distances: odd code distances d of the data block that are scanned.
        reaction_us: classical reaction time (microseconds) of the optional floor on the
            duration (None: no floor).
    """

    p_phys: float = 1e-3
    cycle_us: float = 1.0
    n_factories: int = 4
    routing_overhead: float = 0.5
    distances: tuple[int, ...] = tuple(range(7, 45, 2))
    reaction_us: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.p_phys < 1e-2:
            raise ValueError(f"p_phys must lie in (0, 1e-2) (below threshold), got {self.p_phys!r}")
        if not self.cycle_us > 0.0:
            raise ValueError(f"cycle_us must be positive, got {self.cycle_us!r}")
        f = self.n_factories
        if isinstance(f, bool) or int(f) != f or f < 1:
            raise ValueError(f"n_factories must be an integer >= 1, got {f!r}")
        object.__setattr__(self, "n_factories", int(f))
        if not self.routing_overhead >= 0.0:
            raise ValueError(f"routing_overhead must be non-negative, got {self.routing_overhead!r}")
        ds = tuple(int(d) for d in self.distances)
        if not ds or any(d < 3 or d % 2 == 0 or d != d0 for d, d0 in zip(ds, self.distances)):
            raise ValueError(f"distances must be a non-empty sequence of odd integers >= 3, "
                             f"got {self.distances!r}")
        object.__setattr__(self, "distances", ds)
        if self.reaction_us is not None and not self.reaction_us > 0.0:
            raise ValueError(f"reaction_us must be positive or None, got {self.reaction_us!r}")


@dataclass(frozen=True)
class PhysicalCost:
    """Cheapest layout of one circuit (F18).

    Attributes:
        physical_qubits: data block plus factories.
        seconds: duration of one run.
        distance: code distance d of the data block.
        factory: the factory type ("none" for a circuit without Toffolis).
        failure: logical failure probability of one run (<= the budget).
        us_per_toffoli: seconds * 1e6 / N_Tof, the effective microseconds per Toffoli
            including readout and floors (nan without Toffolis).
    """

    physical_qubits: int
    seconds: float
    distance: int
    factory: str
    failure: float
    us_per_toffoli: float


@dataclass(frozen=True)
class _Factory:
    name: str
    footprint: int        # physical qubits
    rounds: float         # surface-code cycles per magic state
    failure: float        # failure probability per state


def _physical_qubits_per_logical_qubit(d: int) -> int:
    return (d + 1) ** 2 * 2


def _topological_error_per_unit_cell(d: int, p: float) -> float:
    # Expression order as in OpenFermion, so that failures agree to the last bit.
    return 0.1 * (100 * p) ** ((d + 1) / 2)


def _autoccz_factory(l1: int, l2: int, p: float) -> _Factory:
    """OpenFermion's AutoCCZ factory model (``_autoccz_factory_dimensions`` and
    ``_compute_autoccz_distillation_error``), transcribed line by line [OF]."""
    t1_height = 4 * l1 / l2
    t1_width = 8 * l1 / l2
    t1_depth = 5.75 * l1 / l2
    ccz_depth, ccz_height, ccz_width = 5, 6, 3
    storage_width = 2 * l1 / l2
    t1_factories = int(math.ceil((1 / ccz_depth * 8) / (1 / t1_depth)))
    t1_factory_column_height = t1_height * math.ceil(t1_factories / 2)
    width = int(math.ceil(t1_width * 2 + ccz_width + storage_width))
    height = int(math.ceil(max(ccz_height, t1_factory_column_height)))
    depth = max(ccz_depth, t1_depth)

    l0_total = p + 100 * _topological_error_per_unit_cell(l1 // 2, p)
    l1_total = 35 * l0_total ** 3 + 1100 * _topological_error_per_unit_cell(l1, p)
    l2_total = 1000 * _topological_error_per_unit_cell(l2, p) + 28 * l1_total ** 2
    return _Factory(name=f"AutoCCZ(l1={l1}, l2={l2})",
                    footprint=width * height * _physical_qubits_per_logical_qubit(l2),
                    rounds=depth * l2, failure=l2_total)


@functools.lru_cache(maxsize=8)
def _known_factories(p: float) -> tuple[_Factory, ...]:
    """OpenFermion's ``iter_known_factories(p)``, in its order [OF]."""
    out = []
    if p == 0.001:
        out.append(_Factory(name="two-level T (arXiv:1808.06709)",
                            footprint=(12 * 8) * 4 * _physical_qubits_per_logical_qubit(31),
                            rounds=6 * 31, failure=4 * 9 * 10 ** -17))
    for l1 in range(5, 25, 2):
        for l2 in range(l1 + 2, 41, 2):
            out.append(_autoccz_factory(l1, l2, p))
    return tuple(out)


_NO_FACTORY = _Factory(name="none", footprint=0, rounds=0.0, failure=0.0)


def _real(value: float, name: str) -> float:
    """float(value) for a real number; ValueError for complex numbers, strings and non-numbers."""
    if isinstance(value, (str, bytes)) or (isinstance(value, numbers.Complex)
                                           and not isinstance(value, numbers.Real)):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number, got {value!r}") from None


def surface_code_cost(n_logical: int, n_toffoli: float, *, fail_budget: float,
                      model: SurfaceCodeModel = SurfaceCodeModel(), shot: bool = False,
                      toffoli_depth: float | None = None) -> PhysicalCost:
    """Cheapest surface-code layout of a circuit with Q logical qubits and N_Tof Toffolis (F18).

    Args:
        n_logical: Q, logical qubits of the circuit.
        n_toffoli: N_Tof, Toffolis (CCZ states) of one run, 0 or >= 1; a float, so that
            large counts do not overflow.
        fail_budget: largest admissible logical failure probability of one run
            (e.g. FTParameters.p_fail_shot, p_fail_ae, p_fail_qpe).
        model: machine parameters.
        shot: add 2 d cycles of logical readout after the Toffolis (a sampling shot; F19).
        toffoli_depth: Toffolis on the critical path, for the reaction floor
            (default N_Tof: fully sequential). Used only when ``model.reaction_us`` is set.

    Returns:
        The layout with the smallest physical qubits x duration among those within the
        budget (module docstring).

    Raises:
        ValueError: invalid inputs, or no factory and distance meet the budget.
    """
    q_real = _real(n_logical, "n_logical")
    q = int(q_real) if math.isfinite(q_real) else 0
    if q != q_real or q < 1:
        raise ValueError(f"n_logical must be a positive integer, got {n_logical!r}")
    n_tof = _real(n_toffoli, "n_toffoli")
    if not (math.isfinite(n_tof) and (n_tof == 0.0 or n_tof >= 1.0)):
        raise ValueError(f"n_toffoli must be 0 or a finite number >= 1, got {n_toffoli!r}")
    budget = _real(fail_budget, "fail_budget")
    if not 0.0 < budget <= 1.0:
        raise ValueError(f"fail_budget must lie in (0, 1], got {fail_budget!r}")
    floor_cycles = 0
    if model.reaction_us is not None:
        depth = n_tof if toffoli_depth is None else _real(toffoli_depth, "toffoli_depth")
        if not (math.isfinite(depth) and depth >= 0.0):
            raise ValueError(f"toffoli_depth must be finite and non-negative, got {toffoli_depth!r}")
        floor_cycles = math.ceil(depth * model.reaction_us / model.cycle_us)

    storage = int(math.ceil(q * (1 + model.routing_overhead)))
    n_fac = model.n_factories
    factories = _known_factories(model.p_phys) if n_tof > 0.0 else (_NO_FACTORY,)
    best = None
    for fac in factories:
        rounds = int(n_tof / n_fac * fac.rounds)
        distill = fac.failure * n_tof
        fac_area = n_fac * fac.footprint
        for d in model.distances:
            cycles = max(rounds, floor_cycles) + (2 * d if shot else 0)
            failure = min(1.0, _topological_error_per_unit_cell(d, model.p_phys) * storage * cycles
                          + distill)
            if failure > budget:
                continue
            area = storage * _physical_qubits_per_logical_qubit(d) + fac_area
            key = area * cycles           # exact integers: ties go to the first layout
            if best is None or key < best[0]:
                best = (key, area, cycles, d, fac.name, failure)
    if best is None:
        raise ValueError(f"no factory and distance in {model.distances[0]}..{model.distances[-1]} keep "
                         f"{n_tof:.4g} Toffolis on {q} logical qubits within the failure budget "
                         f"{budget:g}")
    _, area, cycles, d, name, failure = best
    seconds = cycles * model.cycle_us / 1e6
    us_per_tof = seconds * 1e6 / n_tof if n_tof > 0.0 else math.nan
    return PhysicalCost(physical_qubits=int(area), seconds=float(seconds), distance=int(d),
                        factory=name, failure=float(failure), us_per_toffoli=float(us_per_tof))


def _check_runs(n_runs: float) -> float:
    n = _real(n_runs, "n_runs")
    if not n >= 0.0:
        raise ValueError(f"n_runs must be non-negative, got {n_runs!r}")
    return n


def campaign_seconds(n_runs: float, per_run: PhysicalCost, n_machines: int = 1) -> float:
    """Wall clock of ``n_runs`` independent runs on ``n_machines`` machines (F19), seconds.

    T = t_run ceil(n / M): whole runs (n rounded up), each on one machine, the machines'
    shares run one after another (module docstring). inf for n = inf (an infeasible
    route), 0 for n = 0.
    """
    n = _check_runs(n_runs)
    m = int(n_machines)
    if m != n_machines or m < 1:
        raise ValueError(f"n_machines must be a positive integer, got {n_machines!r}")
    if math.isinf(n):
        return math.inf
    return per_run.seconds * math.ceil(n / m)


def machines_for(n_runs: float, per_run: PhysicalCost, deadline_s: float = SECONDS_PER_WEEK) -> int | None:
    """Smallest number of machines M >= 1 that finish ``n_runs`` whole runs within ``deadline_s`` (F19).

    M = ceil(n / floor(deadline / t_run)), the runs one machine finishes in time. None if
    no number of machines suffices: one run is longer than the deadline, or n is
    infinite. 1 for n = 0 or a run of zero length.
    """
    n = _check_runs(n_runs)
    deadline = _real(deadline_s, "deadline_s")
    if not deadline > 0.0:
        raise ValueError(f"deadline_s must be positive, got {deadline_s!r}")
    t = per_run.seconds
    if math.isinf(n) or t > deadline:
        return None
    if n == 0.0 or t == 0.0 or math.isinf(deadline):
        return 1
    m = max(1, math.ceil(n / max(1, math.floor(deadline / t))))
    while m > 1 and campaign_seconds(n, per_run, m - 1) <= deadline:
        m -= 1                          # guard the floor and ceil against round-off
    while campaign_seconds(n, per_run, m) > deadline:
        m += 1
    return m
