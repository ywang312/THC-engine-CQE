"""Fault-tolerant costs of the CQE step rules on exact trajectories (F27, F28, F30, F31).

Tags and formula labels are those of docs/m3-cost-model.md (F1-F19, F2a) and of its M4
continuation (F20-F53). Costs are Toffolis, energies Hartree, variances Ha^2; "C_QPE" is one
phase estimation of the selected block encoding. The runs come from ``steprules.run_step_rule``
(the exact CI-space loop with the rules unit, parab3, hybrid, line_search and candidates); its
ledger (``StepRuleResult.events``) lists what each iteration measures.

F27 (precision of an energy that decides or fits a step) [derived; Monte-Carlo calibrated]
-----------------------------------------------------------------------------------------
:func:`energy_precision` returns sigma for a model decrease dE_m (= s0'/2 of F26 with the F33
slope), in units of |dE_m|:

    kind            sigma / |dE_m|                         rule
    parab3          0.47 at L = 0.1 (fixed); table         3-point fit at {0, t, 2t}, interpolation only
    parab3_linear   sqrt(8 L) (0.894 at L = 0.1)           the same fit, linearized at u = eps*/t = 1
    parab2          table (0.167 at L = 0.1)               E(0), E(t) and the measured slope, clip [0.05, 3]t
    armijo          (1 - 2 c1)/(sqrt(2) z) (0.3535 at z = 2) sufficient decrease, two fresh energies

L is the expected fraction of the decrease that the fitted step loses on a parabola, E[(eps^/eps* - 1)^2].
Linearized, the 3-point minimizer has (delta eps/eps*)^2 = sigma^2 u^2 f(u)/(4 dE*^2), f(u) = (u - 3/2)^2
+ 4(1 - u)^2 + (u - 1/2)^2 [derived], i.e. sigma = sqrt(8L)|dE_m| at u = 1; the 2-point fit has
(delta eps/eps*)^2 = 2 sigma^2/dE*^2 [derived]. Both fail at the precisions that matter, where the
fitted curvature is often <= 0: the tables :data:`PRECISION_TABLE` are the Monte-Carlo inverses of
L(sigma) on the parabola E = s eps + c eps^2/2 (eps* = t = 1, 4e5 draws with common random numbers,
:func:`fit_losses`; rebuilt by :func:`build_precision_table` in a slow test), interpolated
log-linearly in (L, sigma) on L in [0.005, 0.3]. [build check: L = 0.101 at 0.47|dE| and 0.264 at
0.894|dE| (parab3); the parab3 inverse at L = 0.1 is 0.468, and the table carries the fixed 0.47
there; parab2 reaches L = 0.1 at 0.167|dE|, while 0.16|dE| [design] loses L = 0.088.]
The Armijo test (c1 = 1e-4) rejects the model step at sigma_A with probability Phi(-z) [derived].

F28 (cost of the energies) [derived]
------------------------------------
:func:`energy_costs`: n sampled DF energies at precision sigma take S_E = V_E/sigma^2 shots each of
C(Psi_k) + C_step(k; eps_i) + C_rot Toffolis on max(Q(Psi_k), Q_step, Q_rot) logical qubits (F13 per
candidate with sigma in place of eps_ls), so T = S_E sum_i (C(Psi_k) + C_step,i + C_rot), evaluated in
the order of ``residual.line_search_cost`` (bit-identical to it at sigma = eps_ls); an
amplitude-estimated energy of H' takes N_E = 2 pi kappa lambda/sigma applications of C(Psi_k) + C_step
+ C_W + Q_A/2 (``residual.energy_ae``, the F13 option). Against the direct residual (F10) n_E energies
cost rho_E = n_E (V_E/V_m) r^2 ||A_k||_m^2/sigma^2 (C(Psi_k) + C_step + C_rot)/C(Psi_k); with ||A_k||_m^2
proportional to E_k - E_0 and sigma proportional to dE_m, rho_E grows as (E_k - E_0)^-1: an energy
check loses as the CQE converges (the energy-precision wall). Every energy of an iteration, the stencil
points Psi_k and Psi_k(2t) and the Armijo pair included, is charged at V_E(Psi_{k+1}) as in M3's F13, so at
small gaps, where V_E at the stencil's end points exceeds V_E(Psi_{k+1}), rho_E(parab3) is a lower estimate
[build check: parab3 with exact energies, spin-orbital: the mean V_E of the stencil {0, 1, 2} is 2.09, 1.45 and
1.05 V_E(Psi_{k+1}) at k = 1, 2, 3 on h4_r4.0, 1.03 to 1.00 on h4_r1.8].

F30 (iteration cost of a step rule) [derived]
---------------------------------------------
:func:`step_rule_iteration_costs` charges the ledger of iteration k:

    C_iter^R(k) = T_res(k; r) + sum_{trials j} T_res(Psi_k(eps_j); r) + sum_{energies i} T_E(Psi_k(eps_i); sigma_i),

C_iter^R(0) = 0 for the classical first step (F16). T_res is M3's F14 residual: the cheapest route
of ``residual.compare_routes`` at kappa on the ResidualProblem of the state, with the selected walk
and its factorization bias, at ``params.r = r``; a trial residual (a "residual" event at eps != 0;
no rule of ``steprules.STEP_RULES`` takes one) is measured on exp(eps_j D_k) Psi_k, prepared by
C(Psi_k) + C_step(k; eps_j). T_E is F28 at the ledger's sigma with V_E of the DF split at Psi_{k+1}
(M3's convention), the AE option reported in ``line_search_ae``. The results are
``pipeline.IterationCost`` objects: the residual routes in ``residual``/``best``, the sampled
energies as one RouteCost ("energy_sampling") in ``line_search`` (None for an iteration without
energies, whose ``line_search_ae`` is 0), the trial residuals inside ``residual_toffolis``, and the
F19 campaigns (the residual's, one per trial, one sampling campaign per energy at the iteration's
largest qubit count), so ``pipeline.loop_totals``, ``pipeline.machines_for_campaigns``,
``handoff.handoff_table`` and ``handoff.optimal_handoff`` apply unchanged. (``loop_totals`` sums
its AE keys over iterations with energies only; "total" is the loop cost.) The unit rule drops the
readout rotation and its qubits from the loop. For "line_search" (the exact line search, charged
as M3's 11 energies at eps_ls) or "candidates" on an exact-line-search record the costs are M3's
F14 C_iter. [build check: on the spin-free h4 classical-first-step record C_iter(k) equals
``pipeline.iteration_costs`` bit for bit at every k, and C_loop(k* = 3) = 2416.996 C_QPE, M3's T4
value; on unit runs C_iter(k) is the best residual route at Psi_k exactly. Spin-orbital R1 with the
F29 caps reaches 1.6 mHa at k* = 3, 7, 11 and 3 on h4_r1.8, h4_r3.5, h6_r3.0 and n2_r1.098
[design].]

F31 (planned hybrid gate) [design]
----------------------------------
:class:`PlannedEnergyGate` (``planned_energy_gate``) is the ``use_energies`` hook of the hybrid rule
R3: at iteration k >= 1 it plans the three energies of the 3-point fit at sigma_P3(k) =
``sigma_parab`` |dE_m| (F27, F28: C_step(k; t), C_step(k; 2t) along the current direction d_k) and
the residual of the iteration (F30 at the planned budget: in exact emulation T_res at the exact
||A_k||_m, r), both before any energy is measured, and uses the fit iff T_E^plan <= theta
T_res^plan. C(Psi_k) is compiled incrementally from the hook's history (``stateprep.
elementwise_step_cost``, as F36), and Psi_k is rebuilt from the start state by exact propagation of
the history (``ft.trajectory.propagate``, the default propagator of ``run_step_rule``, so the
rebuilt state is the run's own); :meth:`PlannedEnergyGate.plan` takes a state directly (a
closed-loop caller passes its realized state). The stencil is planned at its nominal points
{0, t, 2t} (the F29 caps, which shorten it, are not foreseen). The planned V_E is ``energy_variance``:
a constant (Ha^2), or a function of a CI state, evaluated at the planned step exp(t D_k) Psi_k. In exact
emulation (:func:`run_step_rule_system`) it is the function, V_E of the DF split at exp(t D_k) Psi_k: F30
charges the energies at V_E(Psi_{k+1}), and Psi_{k+1} is the planned step itself for an uncapped unit step
(t = 1) and the fitted step near it, so both sides of the switch are planned on the charged trajectory, the
residual at the exact ||A_k||_m. A realizable loop has no exact state: ``ft.closedloop.LoopEnergyGate``
plans with the constant V_E(HF), and against its charge at V_E(Psi~_{k+1}) the effective threshold is
theta V_E(Psi~_{k+1})/V_E(HF). Every decision is logged in ``gate.log`` (k, sigma, the planned V_E,
T_E^plan, T_res^plan, use).

Records (:func:`to_trajectory_record`)
--------------------------------------
A run with ``keep_states=True`` becomes a ``ft.trajectory.TrajectoryRecord`` exactly as
``exact_trajectory`` builds one: the states normalized once more, the rows k < K from the run's
``CQEStep`` (energy, variance, ||A||), the last row from ``residual.compute_residual`` at Psi_K,
p_k against the spectrum's psi_0, E1 and E1_any of the spectrum, the irrep impurity, and
``first_step`` "second_order" for the classical first step ("exact" for first="line_search"). With
first="line_search" the line-search rule's record is M3's ``exact_trajectory`` record bit for bit
(both metrics); with the classical first step it agrees with ``exact_trajectory(first_step=
"second_order")`` to the precision of the line searches [build check: h4, 8 steps: energies within
1.0e-11 Ha, overlaps within 1.7e-10, eps within 4e-6 relative; ``run_cqe`` propagates the forced first
step by its Krylov expansion at tolerance 1e-10, the rule by Taylor at 1e-15, and the late line
searches resolve their flat minima only to sqrt(|E| u/dE) (``ft.trajectory``)]
(tests/test_ft_steprules.py).

Per-system block (:func:`run_step_rule_system`; table U5)
---------------------------------------------------------
Every rule on every metric from HF: the exact CI-space run (``steprules.run_step_rule``; R3 once per r,
since its gate plans at r, with :func:`planned_energy_gate` planning V_E of the DF split at the planned
step, F31), the converted record, k* (E_k - E_0 <= ``params.target``), the first k with p_k >= 0.9 (k_p90)
and the reference iterate k_ref = k*, or k_p90 where k* is not reached, and F30 at every r for
k <= k* + 2 (k <= 20 where k* is not reached; both capped at 12 for N >= 8), with C_loop(k) =
sum_{j<k} C_iter(j), C_loop(k*) and C_loop(k_ref) and, at k = 1 and k_ref - 1, the F27 precisions sigma_A
and sigma_P3 and the F28 ratios rho_E = T_E/T_res,direct of the two Armijo energies (Psi_k and the accepted
step) and of the three energies of the 3-point fit at {0, t, 2t}. The residual inputs are those of the
closed loop (``ft.closedloop.system_inputs``: M3's iterate-1 variances at N <= 6, F24 (i) at N = 8). The
residual problem of every state (``residual_inputs.build_problem`` with the aux-bias curve, the expensive
part) is built once per run, shared by R3's gate and the costing wherever their states agree to the last
bit (the gate rebuilds the run's states by the same propagation), and costed at every r; the result at
each r is :func:`step_rule_iteration_costs` at that r (tests/test_ft_steprules.py).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Callable, Mapping, Sequence

import numpy as np

from ..fci_ops import Hamiltonian, inner, normalize
from ..integrals import ActiveSpace
from ..residual import compute_residual
from ..steprules import StepRule, StepRuleResult, run_step_rule
from . import residual_inputs as ri
from . import scaling
from .physical import SurfaceCodeModel
from .pipeline import IterationCost, _sampling_campaign, machines_for_campaigns, route_campaigns
from .primitives import FTParameters
from .qpe import BlockEncoding
from .residual import PrepParams, RouteCost, WalkParams, best_route, compare_routes, energy_ae
from .stateprep import (FLAG_ROUNDING, PrepCost, candidate_step_costs, elementwise_step_cost, prep_cost,
                        readout_rotation_qubits)
from .symmetry import ci_irrep, determinant_irreps
from .trajectory import TrajectoryRecord, first_iterate_below, propagate

__all__ = [
    "PRECISION_KINDS",
    "SIGMA_P3",
    "PRECISION_TABLE",
    "fit_losses",
    "build_precision_table",
    "energy_precision",
    "energy_costs",
    "PlannedEnergyGate",
    "planned_energy_gate",
    "to_trajectory_record",
    "step_rule_iteration_costs",
    "run_step_rule_system",
]

PRECISION_KINDS = ("parab3", "parab3_linear", "parab2", "armijo")
#: sigma_P3 / |dE_m| of the 3-point fit at the loss 0.1 (F27; the value of record, Monte Carlo 0.468).
SIGMA_P3 = 0.47
_LOSS_REF = 0.1
#: F27: (L, sigma/|dE_m|) of the fitted steps from the Monte-Carlo inverse of L(sigma) (module docstring);
#: the parab3 entry at L = 0.1 is SIGMA_P3 (Monte Carlo: 0.468).
PRECISION_TABLE: dict[str, tuple[tuple[float, float], ...]] = {
    "parab3": ((0.005, 0.181), (0.01, 0.231), (0.02, 0.280), (0.03, 0.313), (0.05, 0.364), (0.07, 0.407),
               (0.1, SIGMA_P3), (0.15, 0.571), (0.2, 0.692), (0.3, 1.05)),
    "parab2": ((0.005, 0.0490), (0.01, 0.0678), (0.02, 0.0922), (0.03, 0.109), (0.05, 0.132), (0.07, 0.148),
               (0.1, 0.167), (0.15, 0.191), (0.2, 0.212), (0.3, 0.247)),
}
_MC_SEED = 20261108
_MC_DRAWS = 400_000
_MC_RATIOS = tuple(round(0.02 + 0.01 * i, 2) for i in range(159))         # sigma/|dE| = 0.02 .. 1.60
# parabola_step's floor and fallback (steprules), and the 2-point fit's clip [0.05, 3] t.
_FLOOR, _FALLBACK, _CEILING = 0.05, 0.25, 3.0
_SAMPLING_ROUTES = ("direct_mg", "aux_mg")
_FLAG_NO_LAYOUT = "no layout within the failure budget"


def _positive(name: str, x, *, allow_inf: bool = False) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number, got {x!r}") from None
    if not (v > 0.0 and (math.isfinite(v) or (allow_inf and v == math.inf))):
        raise ValueError(f"{name} must be positive{'' if allow_inf else ' and finite'}, got {x!r}")
    return v


def _count(name: str, x, least: int) -> int:
    if isinstance(x, (bool, np.bool_)) or int(x) != x or int(x) < least:
        raise ValueError(f"{name} must be an integer >= {least}, got {x!r}")
    return int(x)


# ------------------------------------------------------------------------- F27
def _parab3_steps(e0: np.ndarray, e1: np.ndarray, e2: np.ndarray, t: float = 1.0) -> np.ndarray:
    """``steprules.parabola_step`` (interpolation only) on arrays, operation for operation."""
    c = (e0 - 2.0 * e1 + e2) / t ** 2
    b = (-3.0 * e0 + 4.0 * e1 - e2) / (2.0 * t)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = -b / c
    inside = (c > 0.0) & (x > 0.0) & (x <= 2.0 * t)
    fit = np.where(x >= _FLOOR * t, x, _FLOOR * t)
    best = np.where(np.minimum(e1, e2) < e0, np.where(e1 <= e2, t, 2.0 * t), _FALLBACK * t)
    return np.where(inside, fit, best)


def _parab2_steps(e0: np.ndarray, e1: np.ndarray, slope: float, t: float = 1.0) -> np.ndarray:
    """2-point fit from E(0), E(t) and the slope: eps = -s/c, c = 2(E(t) - E(0) - s t)/t^2, clipped to [0.05, 3]t;
    the far clip 3t for c <= 0."""
    c = 2.0 * (e1 - e0 - slope * t) / t ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        x = -slope / c
    return np.where(c > 0.0, np.clip(x, _FLOOR * t, _CEILING * t), _CEILING * t)


def fit_losses(kind: str, sigma_ratios: Sequence[float], *, n_draws: int = _MC_DRAWS,
               seed: int = _MC_SEED) -> np.ndarray:
    """Expected loss L = E[(eps^/eps* - 1)^2] of a fitted step at sigma = x |dE*| for every x (F27, Monte Carlo).

    The parabola E(eps) = s eps + c eps^2/2 with s = -1, c = 1 (eps* = 1, dE* = -1/2), stencil t = 1,
    independent Gaussian energy errors; the same standard normals for every x (common random numbers).
    ``kind``: "parab3" (``steprules.parabola_step``, interpolation only) or "parab2" (E(0), E(t) and the
    exact slope, clip [0.05, 3]t).
    """
    if kind not in ("parab3", "parab2"):
        raise ValueError(f"kind must be 'parab3' or 'parab2', got {kind!r}")
    x = np.asarray(sigma_ratios, dtype=float).ravel()
    if x.size == 0 or not np.all(np.isfinite(x)) or np.any(x < 0.0):
        raise ValueError("sigma_ratios must be finite and non-negative")
    n = _count("n_draws", n_draws, 1)
    z = np.random.default_rng(_count("seed", seed, 0)).standard_normal((3, n))
    s, c = -1.0, 1.0
    de = 0.5
    grid = np.array([0.0, 1.0, 2.0])
    exact = s * grid + 0.5 * c * grid ** 2
    out = np.empty(x.size)
    for i, ratio in enumerate(x):
        e = exact[:, None] + ratio * de * z
        steps = _parab3_steps(*e) if kind == "parab3" else _parab2_steps(e[0], e[1], s)
        out[i] = float(np.mean((steps - 1.0) ** 2))
    return out


def build_precision_table(*, n_draws: int = _MC_DRAWS,
                          seed: int = _MC_SEED) -> dict[str, tuple[tuple[float, float], ...]]:
    """The Monte-Carlo inverses (L, sigma/|dE_m|) at the losses of :data:`PRECISION_TABLE` (raw, unrounded).

    L(sigma) is evaluated on sigma/|dE| = 0.02 .. 1.60 (step 0.01) by :func:`fit_losses` and inverted
    log-linearly; it is increasing on that grid (checked; ValueError otherwise).
    """
    ratios = np.array(_MC_RATIOS)
    out: dict[str, tuple[tuple[float, float], ...]] = {}
    for kind, table in PRECISION_TABLE.items():
        losses = fit_losses(kind, ratios, n_draws=n_draws, seed=seed)
        if not np.all(np.diff(losses) > 0.0):
            raise ValueError(f"the {kind} loss is not increasing on the sigma grid (too few draws?)")
        targets = np.array([lt for lt, _ in table])
        if targets.min() <= losses[0] or targets.max() >= losses[-1]:
            raise ValueError(f"the {kind} losses do not bracket the table")
        sig = np.exp(np.interp(np.log(targets), np.log(losses), np.log(ratios)))
        out[kind] = tuple((float(lt), float(sv)) for lt, sv in zip(targets, sig))
    return out


def energy_precision(kind: str, de_model: float, *, c1: float = 1e-4, z: float = 2.0, loss: float = 0.1) -> float:
    """F27: the precision sigma (Hartree) of the energies of a step rule at the model decrease ``de_model``.

    ``kind`` is one of :data:`PRECISION_KINDS` (module docstring): "parab3" (0.47|dE_m| at
    ``loss`` = 0.1, the Monte-Carlo table for other losses in [0.005, 0.3]), "parab3_linear" (sqrt(8 loss)
    |dE_m|), "parab2" (the table) and "armijo" ((1 - 2 c1)|dE_m|/(sqrt(2) z); ``loss`` unused). The sign of
    ``de_model`` does not matter; dE_m = 0 gives 0.
    """
    if kind not in PRECISION_KINDS:
        raise ValueError(f"kind must be one of {PRECISION_KINDS}, got {kind!r}")
    try:
        de = abs(float(de_model))
    except (TypeError, ValueError):
        raise ValueError(f"de_model must be a real number, got {de_model!r}") from None
    if not math.isfinite(de):
        raise ValueError(f"de_model must be finite, got {de_model!r}")
    if kind == "armijo":
        c1 = float(c1)
        if not 0.0 < c1 < 0.5:
            raise ValueError(f"c1 must lie in (0, 1/2), got {c1!r}")
        return (1.0 - 2.0 * c1) * de / (math.sqrt(2.0) * _positive("z", z))
    loss = float(loss)
    if not 0.0 < loss < 1.0:
        raise ValueError(f"loss must lie in (0, 1), got {loss!r}")
    if kind == "parab3_linear":
        return math.sqrt(8.0 * loss) * de
    if kind == "parab3" and loss == _LOSS_REF:
        return SIGMA_P3 * de
    table = PRECISION_TABLE[kind]
    ls, ss = np.log([lt for lt, _ in table]), np.log([sv for _, sv in table])
    if not ls[0] <= math.log(loss) <= ls[-1]:
        raise ValueError(f"loss must lie in [{table[0][0]}, {table[-1][0]}] for {kind!r} (the Monte-Carlo table), "
                         f"got {loss!r}")
    return float(np.exp(np.interp(math.log(loss), ls, ss))) * de


# ------------------------------------------------------------------------- F28
def energy_costs(n_energies: int, sigma: float, prep_k: PrepParams, step_toffolis: Sequence[float], *,
                 energy_variance: float, readout_toffolis: int, readout_qubits: int,
                 walk: WalkParams | None = None, kappa: float = 1.0, step_qubits: Sequence[int] | None = None,
                 norb: int | None = None) -> dict:
    """F28: ``n_energies`` energies at precision ``sigma`` (Hartree) of the states exp(eps_i D_k) Psi_k (module
    docstring).

    Args:
        n_energies: number of energies (>= 1).
        sigma: RMS error of each energy (Hartree).
        prep_k: the circuit of Psi_k (C(Psi_k), Q(Psi_k)).
        step_toffolis: C_step(k; eps_i) of each energy (0 for Psi_k itself; ``stateprep.candidate_step_costs``).
        energy_variance: V_E (Ha^2) of the DF split (``residual_inputs.energy_variances``).
        readout_toffolis, readout_qubits: C_rot and Q_rot of the DF readout rotation (``ft.stateprep``).
        walk: the walk of H' for the AE option (None: no AE option).
        kappa: the amplitude-estimation constant of the AE option.
        step_qubits: Q_step of each energy's step; None reports max(Q(Psi_k), Q_rot), a lower bound (flag
            "qubits_lower_bound").
        norb: N, the active orbitals (the AE circuits share the 2N system qubits); None counts them twice
            (``residual.energy_ae``'s upper bound).

    Returns {"sampling": RouteCost "energy_sampling" (n_prep and n_runs: all shots; params: n_energies,
    shots_per_energy, eps = sigma, energy_variance, readout_toffolis, step_toffolis), "ae": RouteCost
    "energy_ae" summed over the energies (``run_toffolis`` per energy; None without ``walk``)}.
    """
    n = _count("n_energies", n_energies, 1)
    sigma = _positive("sigma", sigma)
    if not isinstance(prep_k, PrepParams):
        raise ValueError(f"prep_k must be a PrepParams, got {type(prep_k).__name__}")
    steps = np.asarray(step_toffolis, dtype=float).ravel()
    if steps.shape != (n,) or not np.all(np.isfinite(steps)) or np.any(steps < 0.0):
        raise ValueError(f"step_toffolis must hold {n} finite non-negative Toffoli counts, got {steps.shape}")
    v = float(energy_variance)
    if not (math.isfinite(v) and v >= 0.0):
        raise ValueError(f"energy_variance must be finite and non-negative, got {energy_variance!r}")
    c_rot = _count("readout_toffolis", readout_toffolis, 0)
    q_rot = _count("readout_qubits", readout_qubits, 1)
    if step_qubits is None:
        q_steps = None
        q, flags = max(prep_k.logical_qubits, q_rot), ("qubits_lower_bound",)
    else:
        q_steps = [_count("step_qubits", x, 1) for x in step_qubits]
        if len(q_steps) != n:
            raise ValueError(f"step_qubits must hold {n} qubit counts, got {len(q_steps)}")
        q, flags = max([prep_k.logical_qubits, q_rot] + q_steps), ()
    shots = v / sigma ** 2
    per_shot = prep_k.toffolis + steps + c_rot
    n_shots = shots * steps.size
    total = shots * float(per_shot.sum())
    sampling = RouteCost("energy_sampling", n_shots, total, total / n_shots if n_shots > 0 else math.nan,
                         float(per_shot.max()), n_shots, q,
                         {"n_energies": n, "shots_per_energy": shots, "eps": sigma, "energy_variance": v,
                          "readout_toffolis": c_rot, "step_toffolis": tuple(float(s) for s in steps)}, flags)
    ae = None
    if walk is not None:
        if not isinstance(walk, WalkParams):
            raise ValueError(f"walk must be a WalkParams (BlockEncoding.walk_params()), got {type(walk).__name__}")
        q_ae = q_steps if q_steps is not None else [prep_k.logical_qubits] * n
        costs = [energy_ae(walk, PrepParams(prep_k.toffolis + s, max(prep_k.logical_qubits, qs)), eps=sigma,
                           kappa=kappa, norb=norb) for s, qs in zip(steps.tolist(), q_ae)]
        t_ae = sum(c.toffolis for c in costs)
        n_ae = sum(c.n_prep for c in costs)
        ae = RouteCost("energy_ae", n_ae, t_ae, t_ae / n_ae if n_ae > 0 else math.nan,
                       max(c.coherent_toffolis for c in costs), float(n), max(c.logical_qubits for c in costs),
                       {"eps": sigma, "kappa": float(kappa), "n_energies": n, "walk_kind": walk.kind},
                       tuple(dict.fromkeys([f for c in costs for f in c.flags] + list(flags))),
                       run_toffolis=tuple(c.toffolis for c in costs))
    return {"sampling": sampling, "ae": ae}


# ------------------------------------------------------------------------- the residual of F30
def _residual_problem(active: ActiveSpace, model_active: ActiveSpace, psi: np.ndarray, qpe_walk: BlockEncoding,
                      elements: np.ndarray, variances: dict, normalizations: dict, params: FTParameters,
                      problems: dict | None = None):
    """The ResidualProblem of ``psi`` (exact A, aux-bias curve) and the walk with its factorization bias; the
    r-independent part of F14 (built once and costed at every r).

    ``problems`` memoizes the result by the state's layout and bytes and by ``params.kind`` and
    ``params.aux_deltas`` (the only parameters that enter); it is valid for one set of the other arguments, and a
    hit returns the objects a build of the same bytes returned (ResidualProblem and WalkParams are frozen), so
    results are unchanged."""
    key = None
    if problems is not None:
        a = np.asarray(psi)
        key = (params.kind, tuple(params.aux_deltas), a.shape, a.dtype.str, a.strides, a.tobytes())
        if key in problems:
            return problems[key]
    problem = ri.build_problem(active, psi, elements, params.kind, variances=variances,
                               normalizations=normalizations, deltas=params.aux_deltas)
    bias = ri.factorization_bias(active, model_active, psi, elements, params.kind)
    out = (problem, qpe_walk.walk_params(bias))
    if key is not None:
        problems[key] = out
    return out


def _residual_routes(active: ActiveSpace, model_active: ActiveSpace, psi: np.ndarray, prep: PrepParams,
                     qpe_walk: BlockEncoding, elements: np.ndarray, variances: dict, normalizations: dict,
                     params: FTParameters, problems: dict | None = None) -> tuple[tuple[RouteCost, ...], float]:
    """Every route of F14 at the state ``psi`` (``params.r``), as ``pipeline.iteration_costs``; and ||A||_m."""
    problem, walk = _residual_problem(active, model_active, psi, qpe_walk, elements, variances, normalizations,
                                      params, problems)
    return tuple(compare_routes(problem, prep, walk, params=params)), problem.norm_a_m


def _variance(name: str, x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number, got {x!r}") from None
    if not (math.isfinite(v) and v >= 0.0):
        raise ValueError(f"{name} must be finite and non-negative, got {x!r}")
    return v


def _with_r(params: FTParameters, r) -> FTParameters:
    r = _positive("r", r)
    return params if r == params.r else dataclasses.replace(params, r=r)


def _check_inputs(elements, variances, normalizations) -> None:
    if elements is None or variances is None or normalizations is None:
        raise ValueError("elements, variances and normalizations are required (ft.residual_inputs)")


# ------------------------------------------------------------------------- F31
class PlannedEnergyGate:
    """The F31 gate of the hybrid rule: ``use_energies(k, dE_m, rd, history) -> bool`` (module docstring).

    Attributes:
        theta: the switch threshold.
        energy_variance: the planned V_E (Ha^2), or a function of a CI state evaluated at the planned step
            exp(t D_k) Psi_k (module docstring).
        log: one dict per decision: k, sigma, energy_variance (the planned V_E; None without an energy plan),
            t_energy (T_E^plan), t_residual (T_res^plan), use.
    """

    def __init__(self, active: ActiveSpace, qpe_walk: BlockEncoding, *, irreps: np.ndarray, elements: np.ndarray,
                 variances: dict, normalizations: dict, energy_variance: float | Callable[[np.ndarray], float],
                 readout_toffolis: int, r: float, theta: float = 1.0, sigma_parab: float = SIGMA_P3,
                 parab_t: float = 1.0, ci0: np.ndarray | None = None, params: FTParameters = FTParameters()) -> None:
        _check_inputs(elements, variances, normalizations)
        self.active, self.qpe_walk, self.params = active, qpe_walk, params
        self.irreps = None if irreps is None else np.asarray(irreps, dtype=np.int64)
        if self.irreps is not None and self.irreps.shape != (active.norb,):
            raise ValueError(f"irreps must have shape ({active.norb},), got {self.irreps.shape}")
        self.elements, self.variances, self.normalizations = elements, variances, normalizations
        self.energy_variance = energy_variance if callable(energy_variance) else \
            _variance("energy_variance", energy_variance)
        self.readout_toffolis = _count("readout_toffolis", readout_toffolis, 0)
        self.readout_qubits = readout_rotation_qubits(active.norb, params=params)
        self.params_r = _with_r(params, r)
        self.theta = _positive("theta", theta, allow_inf=True)
        self.sigma_parab = _positive("sigma_parab", sigma_parab)
        self.parab_t = _positive("parab_t", parab_t)
        self._model_active = qpe_walk.model(active)
        self._ham = Hamiltonian(active)
        space = self._ham.space
        if ci0 is not None and np.iscomplexobj(ci0):
            raise ValueError("ci0 must be a real CI vector")
        psi = normalize(space.hf() if ci0 is None else np.asarray(ci0, dtype=float).reshape(space.shape))
        self._states = [psi]
        self._directions: list[np.ndarray] = []
        self._eps: list[float] = []
        self._steps: list = []
        # the residual problems by state (_residual_problem), shared with a costing of the same run
        self._problems: dict | None = None
        self.log: list[dict] = []

    def _advance(self, directions: Sequence[np.ndarray], eps: Sequence[float]) -> tuple[np.ndarray, PrepParams]:
        """Psi_k and its circuit after the steps (directions[j], eps[j]), j < k, reusing the cached prefix."""
        j = 0
        while j < min(len(self._eps), len(eps)) and self._eps[j] == float(eps[j]) \
                and np.array_equal(self._directions[j], directions[j]):
            j += 1
        del self._directions[j:], self._eps[j:], self._steps[j:], self._states[j + 1:]
        for d, e in zip(directions[j:], eps[j:]):
            d = np.array(d, dtype=float, copy=True)
            self._steps.append(elementwise_step_cost(d, float(e), irreps=self.irreps, params=self.params))
            self._states.append(propagate(self._ham, d, self._states[-1], float(e)))
            self._directions.append(d)
            self._eps.append(float(e))
        norb = self.active.norb
        prep = PrepParams(float(sum(s.toffolis for s in self._steps)),
                          max([2 * norb] + [s.logical_qubits for s in self._steps]))
        return self._states[-1], prep

    def planned_variance(self, state: np.ndarray, d_k: np.ndarray) -> float:
        """The planned V_E of the energies at Psi_k = ``state`` along ``d_k``: the constant, or the function at the
        planned step exp(t D_k) Psi_k (``ft.trajectory.propagate``, t = ``parab_t``)."""
        if not callable(self.energy_variance):
            return self.energy_variance
        psi = np.asarray(state, dtype=float).reshape(self._ham.space.shape)
        psi_plan = propagate(self._ham, np.asarray(d_k, dtype=float), psi, self.parab_t)
        return _variance("the planned energy variance", self.energy_variance(psi_plan))

    def plan(self, de_model: float, state: np.ndarray, prep_k: PrepParams, d_k: np.ndarray) -> dict:
        """The planned costs at Psi_k = ``state``: {"sigma", "energy_variance" (the planned V_E; None without an
        energy plan), "energy" (RouteCost), "t_energy", "residual" (the best route), "t_residual"}."""
        sigma = self.sigma_parab * abs(float(de_model))
        t = self.parab_t
        v_plan = None
        if sigma > 0.0 and math.isfinite(sigma):
            cands = candidate_step_costs(np.asarray(d_k, dtype=float), (0.0, t, 2.0 * t), irreps=self.irreps,
                                         params=self.params)
            v_plan = self.planned_variance(state, d_k)
            energy = energy_costs(3, sigma, prep_k, [c.toffolis for c in cands],
                                  energy_variance=v_plan, readout_toffolis=self.readout_toffolis,
                                  readout_qubits=self.readout_qubits,
                                  step_qubits=[c.logical_qubits for c in cands])["sampling"]
            t_energy = energy.toffolis
        else:
            energy, t_energy = None, math.inf
        routes, _ = _residual_routes(self.active, self._model_active, state, prep_k, self.qpe_walk, self.elements,
                                     self.variances, self.normalizations, self.params_r, self._problems)
        best = best_route(routes, kappa=self.params.kappa)
        return {"sigma": sigma, "energy_variance": v_plan, "energy": energy, "t_energy": float(t_energy),
                "residual": best, "t_residual": float(best.toffolis)}

    def decide(self, k: int, plan: Mapping) -> bool:
        """T_E^plan <= theta T_res^plan, logged."""
        use = bool(plan["t_energy"] <= self.theta * plan["t_residual"])
        self.log.append({"k": int(k), "sigma": plan["sigma"], "energy_variance": plan.get("energy_variance"),
                         "t_energy": plan["t_energy"], "t_residual": plan["t_residual"], "use": use})
        return use

    def __call__(self, k: int, de_model: float, rd, history) -> bool:
        directions, eps = history
        if len(directions) != len(eps) + 1 or len(eps) != k:
            raise ValueError(f"history must hold the k + 1 = {k + 1} directions d_0 .. d_k and the k step sizes "
                             f"taken, got {len(directions)} and {len(eps)}")
        state, prep_k = self._advance(directions[:k], eps)
        return self.decide(k, self.plan(de_model, state, prep_k, directions[k]))


def planned_energy_gate(active: ActiveSpace, qpe_walk: BlockEncoding, *, irreps, elements, variances,
                        normalizations, energy_variance: float | Callable[[np.ndarray], float], readout_toffolis: int,
                        r: float, theta: float = 1.0, params: FTParameters = FTParameters(),
                        sigma_parab: float = SIGMA_P3, parab_t: float = 1.0,
                        ci0: np.ndarray | None = None) -> PlannedEnergyGate:
    """F31: the ``use_energies`` hook of the hybrid rule (:class:`PlannedEnergyGate`; module docstring).

    ``energy_variance`` is the planned V_E: a constant (Ha^2), or a function of a CI state (Ha^2) evaluated at the
    planned step exp(t D_k) Psi_k; ``r`` is the residual accuracy of the plan, ``sigma_parab`` and ``parab_t`` those
    of the rule (``StepRule``), ``ci0`` the run's start state (default HF).
    """
    return PlannedEnergyGate(active, qpe_walk, irreps=irreps, elements=elements, variances=variances,
                             normalizations=normalizations, energy_variance=energy_variance,
                             readout_toffolis=readout_toffolis, r=r, theta=theta, sigma_parab=sigma_parab,
                             parab_t=parab_t, ci0=ci0, params=params)


# ------------------------------------------------------------------------- records
def to_trajectory_record(result: StepRuleResult, *, system: str, active: ActiveSpace, irreps: np.ndarray,
                         spectrum: "scaling.Spectrum") -> TrajectoryRecord:
    """The ``TrajectoryRecord`` of a step-rule run with ``keep_states=True`` (module docstring).

    ``spectrum`` (``ft.scaling.spectrum`` of the same build) supplies psi_0, E0, E1 and E1_any; ``irreps`` the
    orbital irreps (the mask of the costs and the irrep impurity). first_step is "second_order" for the
    classical first step, "exact" for first="line_search".
    """
    if not isinstance(result, StepRuleResult):
        raise ValueError(f"result must be a StepRuleResult, got {type(result).__name__}")
    if result.states is None:
        raise ValueError("the run kept no states: run_step_rule(..., keep_states=True)")
    if not isinstance(spectrum, scaling.Spectrum):
        raise ValueError(f"spectrum must be an ft.scaling.Spectrum, got {type(spectrum).__name__}")
    norb, nelec = active.norb, tuple(active.nelec)
    if (spectrum.norb, spectrum.nelec) != (norb, nelec):
        raise ValueError(f"the spectrum belongs to N = {spectrum.norb}, nelec = {spectrum.nelec}, not to the active "
                         f"space (N = {norb}, nelec = {nelec})")
    if spectrum.e1 is None:
        raise ValueError("the spectrum has no E1 (n_roots = 1)")
    ir = np.asarray(irreps)
    if ir.shape != (norb,) or ir.dtype.kind not in "iu":
        raise ValueError(f"irreps must be ({norb},) integers, got {ir.shape} {ir.dtype}")
    ir = ir.astype(int)
    k_steps = result.n_steps
    if len(result.states) != k_steps + 1 or len(result.steps) != k_steps or len(result.directions) != k_steps:
        raise ValueError("inconsistent run: need K steps, directions and K + 1 states")
    ham = Hamiltonian(active)
    states = [normalize(np.asarray(s).reshape(ham.space.shape)) for s in result.states]
    if states[0].size != spectrum.dim:
        raise ValueError(f"the states have {states[0].size} determinants, the spectrum {spectrum.dim}")
    rows = [(s.energy, s.variance, s.residual_norm) for s in result.steps]
    last = compute_residual(active, states[-1], ham)
    rows.append((last.energy, last.variance, last.residual_norm))
    energies, variances, rnorms = (np.array(col, dtype=float) for col in zip(*rows))
    psi0 = spectrum.psi0
    overlaps = np.array([abs(complex(inner(psi0, s))) ** 2 for s in states])
    flags = [f for f in spectrum.flags if f in ("e1_lower_bound", "spin_tol_relaxed")]
    hf_irrep = ci_irrep(states[0], norb, nelec, ir)[0]
    if hf_irrep != spectrum.ground_irrep:
        flags.append("ground_state_not_in_hf_irrep")
    det_ir = determinant_irreps(norb, nelec, ir)
    impurity = max(float(np.sum(np.abs(s[det_ir != hf_irrep]) ** 2)) for s in states)
    return TrajectoryRecord(
        system=system or active.label, optimizer=result.optimizer, norb=norb, nelec=nelec, e0=float(spectrum.e0),
        e1=float(spectrum.e1), e1_any=float(spectrum.e1_any), energies=energies, variances=variances,
        residual_norms=rnorms, overlaps=overlaps, eps=np.array(result.eps, dtype=float),
        directions=[np.asarray(d) for d in result.directions], states=states, irreps=ir, metric=result.metric,
        ground_irrep=spectrum.ground_irrep, irrep_impurity=impurity, converged=bool(result.converged),
        flags=tuple(flags), first_step="second_order" if result.rule.first == "second_order" else "exact")


# ------------------------------------------------------------------------- F30
def step_rule_iteration_costs(active: ActiveSpace, record: TrajectoryRecord, result: StepRuleResult, prep: PrepCost,
                              qpe_walk: BlockEncoding, *, elements, variances, normalizations,
                              energy_variance: str, readout_toffolis: int, r: float,
                              k_max: int | None = None, params: FTParameters = FTParameters(),
                              model: SurfaceCodeModel = SurfaceCodeModel()) -> list[IterationCost]:
    """F30: C_iter^R(k) for k = 0..k_max of the run ``result`` along ``record`` (module docstring).

    Args:
        active: the active space.
        record: the trajectory the costs refer to (states Psi_k, directions d_k): the run's own
            (:func:`to_trajectory_record`) or the matching exact record, with irreps.
        result: the run whose ledger is charged; same number of steps and first-step convention as ``record``.
        prep: the elementwise preparation cost of ``record`` (``stateprep.prep_cost``).
        qpe_walk: the selected block encoding (walk of the residual routes, model integrals for the bias).
        elements, variances, normalizations: the measured elements, their per-shot variances and LCU
            normalizations (``ft.residual_inputs``).
        energy_variance: the DF split of ``residual_inputs.DF_VARIANTS`` (V_E at Psi_{k+1}, F13).
        readout_toffolis: C_rot of the energy readout.
        r: the residual accuracy (F10's r) of T_res.
        k_max: last iterate (default K - 1).
        params, model: budgets (``params.r`` is replaced by ``r``) and the surface-code machine.
    """
    r = _positive("r", r)
    return _step_rule_costs(active, record, result, prep, qpe_walk, elements=elements, variances=variances,
                            normalizations=normalizations, energy_variance=energy_variance,
                            readout_toffolis=readout_toffolis, r_values=(r,), k_max=k_max, params=params,
                            model=model)[r]


def _step_rule_costs(active: ActiveSpace, record: TrajectoryRecord, result: StepRuleResult, prep: PrepCost,
                     qpe_walk: BlockEncoding, *, elements, variances, normalizations, energy_variance: str,
                     readout_toffolis: int, r_values: Sequence[float], k_max: int | None = None,
                     params: FTParameters = FTParameters(), model: SurfaceCodeModel = SurfaceCodeModel(),
                     problems: dict | None = None) -> dict[float, list[IterationCost]]:
    """:func:`step_rule_iteration_costs` at every r of ``r_values`` ({r: costs}): the residual problem of each state
    (Psi_k and every trial state) is built once and costed at each r, the energies once per iteration;
    ``problems`` memoizes :func:`_residual_problem` for these inputs (shared with R3's gate)."""
    _check_inputs(elements, variances, normalizations)
    if not isinstance(result, StepRuleResult):
        raise ValueError(f"result must be a StepRuleResult, got {type(result).__name__}")
    if len(prep.steps) != record.n_steps or len(prep.cumulative) != len(record.states):
        raise ValueError(f"prep has {len(prep.steps)} steps for a record of {record.n_steps}")
    if result.n_steps != record.n_steps:
        raise ValueError(f"the run has {result.n_steps} steps, the record {record.n_steps}")
    classical = result.rule.first == "second_order"
    if classical != (record.first_step == "second_order"):
        raise ValueError(f"the run's first step ({result.rule.first}) does not match the record's "
                         f"({record.first_step})")
    last = record.n_steps - 1 if k_max is None else int(k_max)
    if not 0 <= last <= record.n_steps - 1:
        raise ValueError(f"k_max must lie in 0..{record.n_steps - 1}, got {k_max!r}")
    if energy_variance not in ri.DF_VARIANTS:
        raise ValueError(f"energy_variance must be one of {tuple(ri.DF_VARIANTS)}, got {energy_variance!r}")
    rs = tuple(dict.fromkeys(_positive("r", r) for r in r_values))
    if not rs:
        raise ValueError("r_values must not be empty")
    c_rot = _count("readout_toffolis", readout_toffolis, 0)
    params_r = {r: _with_r(params, r) for r in rs}
    model_active = qpe_walk.model(active)
    q_rot = readout_rotation_qubits(active.norb, params=params)
    ham = Hamiltonian(active)
    out: dict[float, list[IterationCost]] = {r: [] for r in rs}
    for k in range(last + 1):
        prep_k = PrepParams(prep.cumulative[k], prep.logical_qubits[k])
        events = result.iteration_events(k)
        if k == 0 and classical:
            if events:
                raise ValueError("the classical first step measures nothing, but the ledger charges iteration 0")
            for r in rs:
                out[r].append(IterationCost(k=0, residual=(), best="classical", line_search=None, toffolis=0.0,
                                            shots=0.0, seconds_one_machine=0.0, machines_7d=0,
                                            best_practical="classical", prep_toffolis=float(prep_k.toffolis)))
            continue
        main = [e for e in events if e.kind == "residual" and e.eps == 0.0]
        trials = [e for e in events if e.kind == "residual" and e.eps != 0.0]
        energies = [e for e in events if e.kind == "energy"]
        if len(main) != 1 or len(main) + len(trials) + len(energies) != len(events):
            raise ValueError(f"iteration {k}: the ledger must hold one residual at Psi_k, trial residuals and energies")
        psi, d = record.states[k], record.directions[k]
        problem, walk = _residual_problem(active, model_active, psi, qpe_walk, elements, variances, normalizations,
                                          params, problems)
        trial_problems = []
        for ev in trials:
            step = elementwise_step_cost(d, ev.eps, irreps=record.irreps, params=params)
            prep_t = PrepParams(prep_k.toffolis + step.toffolis, max(prep_k.logical_qubits, step.logical_qubits))
            trial_problems.append((prep_t, *_residual_problem(active, model_active, propagate(ham, d, psi, ev.eps),
                                                              qpe_walk, elements, variances, normalizations, params,
                                                              problems)))
        ls, ls_ae, flagged, ls_campaigns = None, 0.0, 0, ()
        if energies:
            sigmas = {e.sigma for e in energies}
            if len(sigmas) != 1 or not all(isinstance(s, float) and s > 0.0 for s in sigmas):
                raise ValueError(f"iteration {k}: the energies of one iteration share one positive precision, "
                                 f"got {sorted(map(str, sigmas))}")
            sigma = sigmas.pop()
            cands = candidate_step_costs(d, [e.eps for e in energies], irreps=record.irreps, params=params)
            v_e = ri.energy_variances(active, record.states[k + 1])[energy_variance]
            ec = energy_costs(len(cands), sigma, prep_k, [c.toffolis for c in cands], energy_variance=v_e,
                              readout_toffolis=c_rot, readout_qubits=q_rot, walk=qpe_walk.walk_params(),
                              kappa=params.kappa, step_qubits=[c.logical_qubits for c in cands], norb=active.norb)
            ls, ls_ae = ec["sampling"], float(ec["ae"].toffolis)
            flagged = sum(FLAG_ROUNDING in c.flags for c in cands)
            per = ls.params["shots_per_energy"]
            ls_campaigns = tuple(_sampling_campaign("energy", per, prep_k.toffolis + c.toffolis + c_rot,
                                                    ls.logical_qubits, params, model) for c in cands)
        t_e = 0.0 if ls is None else ls.toffolis
        for r in rs:
            p_r = params_r[r]
            routes = tuple(compare_routes(problem, prep_k, walk, params=p_r))
            best = best_route(routes, kappa=params.kappa)
            best_p = best_route(routes, kappa=params.kappa_practical)
            t_res, t_res_p = best.toffolis, best_p.toffolis
            shots = best.n_prep if best.route in _SAMPLING_ROUTES else 0.0
            campaigns = tuple(route_campaigns(best, p_r, model))
            for prep_t, problem_t, walk_t in trial_problems:
                r_t = tuple(compare_routes(problem_t, prep_t, walk_t, params=p_r))
                b_t, b_tp = best_route(r_t, kappa=params.kappa), best_route(r_t, kappa=params.kappa_practical)
                t_res, t_res_p = t_res + b_t.toffolis, t_res_p + b_tp.toffolis
                shots += b_t.n_prep if b_t.route in _SAMPLING_ROUTES else 0.0
                campaigns += tuple(route_campaigns(b_t, p_r, model))
            campaigns += ls_campaigns
            flags = (_FLAG_NO_LAYOUT,) if any(c.per_run is None and 0 < c.n_runs < math.inf
                                              for c in campaigns) else ()
            out[r].append(IterationCost(
                k=k, residual=routes, best=best.route, line_search=ls,
                toffolis=float(t_res + t_e), shots=float(shots + (0.0 if ls is None else ls.n_prep)),
                seconds_one_machine=float(sum(c.seconds(1) for c in campaigns)),
                machines_7d=machines_for_campaigns(campaigns), best_practical=best_p.route,
                toffolis_practical=float(t_res_p + t_e), residual_toffolis=float(t_res),
                residual_toffolis_practical=float(t_res_p), line_search_ae=ls_ae, residual_norm_m=float(problem.norm_a_m),
                prep_toffolis=float(prep_k.toffolis), ls_flagged=int(flagged), campaigns=campaigns, flags=flags))
    return out


# ------------------------------------------------------------------------- U5: the per-system block
_K_AFTER = 2                 # costed iterates beyond k*
_K_UNREACHED = 20            # costed iterates where k* is not reached
_K_CAP_N8 = 12               # the cap at N >= 8
_METRICS = ("spin_orbital", "spin_free")
_P_REACHED = 0.9


def _cost_horizon(k_star: int | None, n_steps: int, norb: int, k_cost: int | None) -> int:
    """The last costed iterate: k_cost if given, else k* + 2 (20 if k* is not reached), capped at 12 for N >= 8."""
    if k_cost is not None:
        return min(_count("k_cost", k_cost, 0), n_steps - 1)
    end = _K_UNREACHED if k_star is None else k_star + _K_AFTER
    if norb >= 8:
        end = min(end, _K_CAP_N8)
    return min(end, n_steps - 1)


def _energy_ratios(active: ActiveSpace, record: TrajectoryRecord, result: StepRuleResult, rule, k: int,
                   its: Sequence[IterationCost], prep: PrepCost, energy_split: str, readout_toffolis: int,
                   params: FTParameters) -> dict:
    """F27/F28 at iterate k: sigma_A, sigma_P3 and rho_E of the Armijo pair and of the 3-point fit (module
    docstring)."""
    de = float(result.de_model[k])
    d = record.directions[k]
    prep_k = PrepParams(prep.cumulative[k], prep.logical_qubits[k])
    v_e = ri.energy_variances(active, record.states[k + 1])[energy_split]
    direct = next(c for c in its[k].residual if c.route == "direct_mg")
    q_rot = readout_rotation_qubits(active.norb, params=params)
    out = {"k": int(k), "de_model": de, "sigma_armijo": energy_precision("armijo", de),
           "sigma_parab3": energy_precision("parab3", de), "t_res_direct": float(direct.toffolis)}
    t = rule.parab_t
    for key, sigma, steps in (("armijo", out["sigma_armijo"], (0.0, float(record.eps[k]))),
                              ("parab3", out["sigma_parab3"], (0.0, t, 2.0 * t))):
        if not sigma > 0.0:
            out[f"rho_{key}"] = None
            continue
        cands = candidate_step_costs(d, steps, irreps=record.irreps, params=params)
        ec = energy_costs(len(cands), sigma, prep_k, [c.toffolis for c in cands], energy_variance=v_e,
                          readout_toffolis=readout_toffolis, readout_qubits=q_rot,
                          step_qubits=[c.logical_qubits for c in cands])["sampling"]
        out[f"rho_{key}"] = float(ec.toffolis / direct.toffolis) if direct.toffolis > 0 else math.inf
    return out


def run_step_rule_system(name: str, rules: Sequence, *, metrics=_METRICS, r_values=(0.3, 1.0, 3.0),
                         max_iter: int = 40, k_cost: int | None = None, cache: "scaling.Cache | None" = None,
                         params: FTParameters = FTParameters()) -> dict:
    """U5's block of the system ``name``: exact step-rule runs, records, k* and F30 costs (module docstring).

    Args:
        name: a system of ``ft.symmetry.benchmark_system``.
        rules: ``steprules.StepRule`` objects (e.g. line_search, candidates, unit, hybrid).
        metrics: L-BFGS metrics ("spin_orbital", "spin_free").
        r_values: residual accuracies of the costs (and of R3's plan).
        max_iter: steps per run (40; 30 at N = 8 by the caller).
        k_cost: last costed iterate (default: the horizon of the module docstring).
        cache: an ``ft.scaling.Cache`` for the inputs (stages "spectrum", "qpe", "loop_inputs").
        params: M3's parameters (``kind`` "spin_free"; ``target`` sets k*).

    Returns a JSON-able dict (``ft.scaling.to_json``): "system" {name, norb, nelec, c_qpe, qpe_kind, lam,
    n_iter, e0, e1, e1_any, p_hf, variance_state, energy_split, n_elements} and "runs", one per (metric,
    rule[, r for R3]): {"rule" (the StepRule fields), "metric", "r_plan" (R3's planning r, else None),
    "n_steps", "converged", "k_star", "k_p90", "k_ref" (k*, else k_p90, else None) and "k_ref_kind" ("k_star",
    "p90" or None), "p", "e_err", "eps", "flags", "extra_measurements" (the ledger's events per iteration beyond
    the residual at Psi_k), "prep_cumulative", "cost_horizon", "costs" {f"{r:g}": {"c_iter", "c_loop"
    (k = 0..horizon + 1), "c_iter_over_qpe", "c_loop_over_qpe", "c_loop_k_star_over_qpe" and
    "c_loop_k_ref_over_qpe" (None if k* or k_ref is not costed), "best", "residual_toffolis", "energy_toffolis",
    "machines_7d" (None for an iteration that no machine count finishes within 7 days), "precision"
    (:func:`energy_precision` and rho_E at k = 1 and k_ref - 1)}}, "gate_log" (R3: the gate's log, each entry
    with "t_energy_charged", F30's energies of iteration k at r_plan (0 where the gate chose the unit step, None
    beyond the cost horizon), and "energy_variance_charged", their V_E(Psi_{k+1}) (None without energies))}.
    """
    from . import closedloop                 # at call time: ft.closedloop imports this module

    rules = list(rules)
    if not rules or not all(isinstance(r, StepRule) for r in rules):
        raise ValueError("rules must be a non-empty sequence of steprules.StepRule")
    metrics = tuple(metrics)
    if not metrics or any(m not in _METRICS for m in metrics):
        raise ValueError(f"metrics must be a non-empty subset of {_METRICS}, got {metrics!r}")
    r_values = tuple(_positive("r", r) for r in r_values)
    if not r_values:
        raise ValueError("r_values must not be empty")
    max_iter = _count("max_iter", max_iter, 1)
    inputs = closedloop.system_inputs(name, cache=cache, params=params)
    active, irreps, spec, walk = inputs.active, inputs.irreps, inputs.spectrum, inputs.qpe.encoding
    c_qpe = inputs.c_qpe
    kw = dict(elements=inputs.elements, variances=inputs.variances, normalizations=inputs.normalizations)

    def planned_v_e(psi: np.ndarray) -> float:          # F31 in exact emulation: V_E of the split at the planned step
        return ri.energy_variances(active, psi)[inputs.energy_split]

    out = {"system": {"name": name, "norb": active.norb, "nelec": list(active.nelec), "c_qpe": c_qpe,
                      "qpe_kind": walk.kind, "lam": float(walk.lam), "n_iter": int(inputs.qpe.iterations),
                      "e0": float(spec.e0), "e1": float(spec.e1), "e1_any": float(spec.e1_any),
                      "p_hf": float(spec.p_hf), "variance_state": inputs.variance_state,
                      "energy_split": inputs.energy_split, "n_elements": inputs.n_elements},
           "runs": []}
    for metric in metrics:
        for rule in rules:
            for r_plan in (r_values if rule.name == "hybrid" else (None,)):
                problems: dict = {}                    # the residual problems of this run's states
                gate = None
                if rule.name == "hybrid":
                    gate = planned_energy_gate(active, walk, irreps=irreps, energy_variance=planned_v_e,
                                               readout_toffolis=inputs.readout_toffolis, r=r_plan, theta=rule.theta,
                                               params=params, sigma_parab=rule.sigma_parab, parab_t=rule.parab_t,
                                               **kw)
                    gate._problems = problems
                result = run_step_rule(active, rule, max_iter=max_iter, metric=metric, irreps=irreps,
                                       use_energies=gate, keep_states=True)
                record = to_trajectory_record(result, system=name, active=active, irreps=irreps, spectrum=spec)
                k_star = first_iterate_below(record, params.target)
                reached = np.flatnonzero(record.overlaps >= _P_REACHED)
                k_p90 = int(reached[0]) if reached.size else None
                k_ref, k_ref_kind = ((k_star, "k_star") if k_star is not None
                                     else (k_p90, "p90") if k_p90 is not None else (None, None))
                last = _cost_horizon(k_star, record.n_steps, active.norb, k_cost)
                prep = prep_cost(record.directions, record.eps, irreps=irreps, params=params)
                costs = {}
                by_r = _step_rule_costs(active, record, result, prep, walk, energy_variance=inputs.energy_split,
                                        readout_toffolis=inputs.readout_toffolis,
                                        r_values=r_values if r_plan is None else (r_plan,), k_max=last,
                                        params=params, problems=problems, **kw)
                for r, its in by_r.items():
                    c_iter = [float(c.toffolis) for c in its]
                    c_loop = [0.0] + np.cumsum(c_iter).tolist()
                    ks = sorted({k for k in (1, (k_ref or 0) - 1) if 1 <= k <= last})
                    costs[f"{r:g}"] = {
                        "c_iter": c_iter, "c_loop": c_loop, "c_iter_over_qpe": [c / c_qpe for c in c_iter],
                        "c_loop_over_qpe": [c / c_qpe for c in c_loop],
                        "c_loop_k_star_over_qpe": (c_loop[k_star] / c_qpe
                                                   if k_star is not None and k_star <= last + 1 else None),
                        "c_loop_k_ref_over_qpe": (c_loop[k_ref] / c_qpe
                                                  if k_ref is not None and k_ref <= last + 1 else None),
                        "best": [c.best for c in its], "residual_toffolis": [float(c.residual_toffolis) for c in its],
                        "energy_toffolis": [0.0 if c.line_search is None else float(c.line_search.toffolis)
                                            for c in its],
                        "machines_7d": [None if c.machines_7d is None else int(c.machines_7d) for c in its],
                        "precision": [_energy_ratios(active, record, result, rule, k, its, prep, inputs.energy_split,
                                                     inputs.readout_toffolis, params) for k in ks]}
                gate_log = None
                if gate is not None:
                    its = by_r[r_plan]
                    gate_log = []
                    for g in gate.log:
                        k = g["k"]
                        ls = its[k].line_search if k <= last else None
                        gate_log.append(dict(g, t_energy_charged=(None if k > last else
                                                                  0.0 if ls is None else float(ls.toffolis)),
                                             energy_variance_charged=(None if ls is None
                                                                      else float(ls.params["energy_variance"]))))
                out["runs"].append({
                    "rule": dataclasses.asdict(rule), "metric": metric, "r_plan": r_plan,
                    "n_steps": record.n_steps, "converged": bool(record.converged), "k_star": k_star,
                    "k_p90": k_p90, "k_ref": k_ref, "k_ref_kind": k_ref_kind,
                    "p": record.overlaps.tolist(), "e_err": (record.energies - record.e0).tolist(),
                    "eps": record.eps.tolist(), "flags": [list(f) for f in result.flags],
                    "extra_measurements": [sum(1 for e in result.iteration_events(k)
                                               if not (e.kind == "residual" and e.eps == 0.0))
                                           for k in range(record.n_steps)],
                    "prep_cumulative": [float(x) for x in prep.cumulative], "cost_horizon": last, "costs": costs,
                    "gate_log": gate_log})
    return out
