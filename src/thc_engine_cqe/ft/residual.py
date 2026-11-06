"""Fault-tolerant cost of one CQE iteration: the residual by six routes, and the line search (F10-F14).

Pure functions of numbers and arrays. The current state is described by its circuit
(:class:`PrepParams`: Toffolis C(Psi_k) and logical qubits Q(Psi_k), from
``ft.stateprep``), the Hamiltonian by a qubitized block encoding of a factorized H'
(:class:`WalkParams`: lambda, Toffolis C_W of one controlled walk step, logical qubits
Q_walk; from ``ft.qpe``), and the residual by the per-element inputs of a
:class:`ResidualProblem` (built from the M2 measure layer by ``ft.residual_inputs``).
Every function takes ``params: FTParameters`` (``ft.primitives``); formula tags F1-F19
and the tags [OF], [derived], [cited], [estimate], [assumption] are those of
``ft.primitives``.

Units: Toffolis; A, its norms and biases in Hartree; per-shot variances v_e of the
estimators of A_e in Ha^2. A *preparation* is one application of U_k or U_k^dagger
(or of a circuit that contains it once). Shots and applications are real numbers, not
rounded up, as in M2.

Error target (F10) [derived; equals M2's ``routes.shots_for_relative_error``]
----------------------------------------------------------------------------
Element e of the (symmetry-allowed) element set stands for m_e entries of the full
tensor (``targets.element_multiplicities``), and ||x||_m^2 = sum_e m_e x_e^2 is the
Frobenius norm of the full tensor. An estimate with bias b and per-element variances
Var_e meets the relative RMS target r iff ||b||_m^2 + sum_e m_e Var_e <= r^2 ||A||_m^2.
The variance budget is

    B^2 = r^2 ||A||_m^2 - (||b_route||_m + ||b_fact||_m)^2         (variance_budget)

with the route bias (the aux routes' O(delta^2) term) and the factorization bias
||A[H'] - A[H]||_m of routes that use the block encoding of H' (``WalkParams.fact_bias_m``)
combined by the triangle inequality. A route is infeasible (flag "infeasible", every
count inf) when B^2 <= 1e-12 r^2 ||A||_m^2, the round-off rule of M2.

Sampling routes (F10)
---------------------
* ``direct_mg`` (:func:`sampling_direct`): X_e = [Gamma_e - Gamma_e^dagger, H]/2 on Psi_k,
  read out by signed-permutation matchgate shadows, a Clifford ensemble, so the readout
  costs no Toffolis [cited: Zhao, Rubin, Miyake, PRL 127, 110504 (2021)]. Unbiased,
  exact H. S = sum_e m_e v_e / B^2 shots of T = S C(Psi_k) Toffolis; Q = Q(Psi_k).
* ``aux_mg`` (:func:`sampling_aux`): -<Q_e>/delta on exp(-i delta H') Psi_k, the
  evolution by GQSP on the walk, C_evol(delta) = ``primitives.evolution_toffolis`` (F8).
  The per-shot variance is w_e / delta^2 with w_e independent of delta to O(delta^2) [M2,
  ``measure/routes.py``]. S(delta) = sum_e m_e w_e / (delta^2 B(delta)^2),
  T(delta) = S(delta) (C(Psi_k) + C_evol(delta)), minimized over ``params.aux_deltas``;
  bias ||b(delta)||_m (``ResidualProblem.aux_bias``) plus the factorization bias;
  Q = Q(Psi_k) + Q_walk + 1 - 2N (the evolution's register shares the 2N system qubits).

Per-element amplitude estimation (F11) [derived; kappa verified]
---------------------------------------------------------------
A Hadamard-test circuit 𝒜 with success probability p = (1 + <O_e>/alpha_e)/2 is iterated
with G = 𝒜 S_0 𝒜^dagger S_good, whose reflections alone are controlled; sine-window phase
estimation gives sigma(<O_e>) <= kappa pi alpha_e / M with M applications of G, i.e.
N_e = 2 pi kappa alpha_e / eps_e applications of 𝒜 or 𝒜^dagger (``primitives.ae_applications``;
canonical AE: [cited: Brassard, Hoyer, Mosca, Tapp, Contemp. Math. 305, 53 (2002)]).
Minimizing sum_e N_e under sum_e m_e eps_e^2 = B^2 gives eps_e proportional to
(alpha_e / m_e)^(1/3) and (:func:`ae_allocation`)

    N = sum_e N_e = 2 pi kappa Gamma / B,      Gamma = (sum_e m_e^(1/3) alpha_e^(2/3))^(3/2).

Element e is one coherent run of N_e applications on Q_run,e = Q(Psi_k) + Q_obs +
ceil(log2(N_e / 2)) + 1 logical qubits (phase register, Hadamard-test qubit); each
application costs C(Psi_k) + C_obs,e + Q_run,e / 2, the last term the reflection about
|0> per application [estimate]. T = sum_e N_e (C(Psi_k) + C_obs,e + Q_run,e / 2), and
the longest coherent run is the largest summand. The three observables
(C_Q = 4 Toffolis for spin-orbital, 16 for spin-free Q_e [estimate]):

    route        alpha_e              C_obs,e               Q_obs               bias
    ae_product   2 lambda_Q,e lambda  C_W + C_Q             Q_walk - 2N + 4     factorization
    ae_x         alpha_X,e            2 L_X,e [estimate]    ceil(log2 L_X,e) + chi + 4    none
    ae_aux       lambda_Q,e / delta   C_evol(delta) + C_Q   Q_walk + 1 - 2N + 4 b(delta) + factorization

* ``ae_product`` (b1): the Hadamard test on U_Q U_H (separate ancillas) gives
  Im <Q_e H> / (lambda_Q lambda) and, for real Psi, A_e = -2 Im <Q_e H> [derived; dense
  check on H2 in tests/test_ft_residual_inputs.py].
* ``ae_x`` (b2): the Pauli LCU of X_e, alpha_X = its 1-norm, QROM PREPARE plus unary SELECT
  over its L_X terms [estimate]; no Hamiltonian factorization.
* ``ae_aux`` (b3): AE of <Q_e> on exp(-i delta H') Psi_k, delta optimized over
  ``params.aux_deltas``.

Gradient estimation (F12) [assumption; asymptotics cited: Gilyen, Arunachalam, Wiebe,
SODA 2019 (arXiv:1711.00465); Huggins et al., PRL 129, 240501 (2022)]
-----------------------------------------------------------------------------------
Formulation "aux" (Huggins et al. on {Q_e} measured on exp(-i delta H') Psi_k,
alpha_e = lambda_Q,e / delta), for K elements in n_B equal blocks:

    N = 2 pi kappa_g sqrt(K) sqrt(n_B) (sum_e m_e alpha_e^2)^(1/2) / B,
    T = N (C(Psi_k) + C_evol(delta) + C_probe / n_B),   C_probe = K (2 b_theta + 2),

plus ceil(K / n_B) b_theta logical qubits for the probe angles. kappa_g is unknown; the
route is reported for the band ``params.kappa_g`` and never selected by :func:`best_route`.

Line search and energies (F13)
------------------------------
Each of n_c candidates Psi_{k+1}(eps_c) is measured by DF basis-rotation grouping
[cited: Huggins et al., npj Quantum Inf. 7, 23 (2021)] to precision eps_ls:
S_E = V_E / eps_ls^2 shots per candidate of C(Psi_k) + C_step(k; eps_c) + C_rot Toffolis
(:func:`line_search_cost`; V_E from ``residual_inputs.energy_variances``, C_step and C_rot
from ``ft.stateprep``). :func:`energy_sampling` costs one such energy;
:func:`energy_ae` the amplitude-estimated energy of H' (alpha = lambda, C_obs = C_W).

One iteration (F14)
-------------------
:func:`compare_routes` evaluates every route (AE at ``params.kappa`` and
``params.kappa_practical``, gradient estimation at every ``params.kappa_g``) and
:func:`best_route` picks the cheapest in Toffolis at one kappa. The iteration cost adds
the line search; the classical first step and the loop sums are the pipeline's
(``ft.pipeline``). :func:`crossover_norm` gives the ||A||_m at which two routes cost the
same.

Observed on the exact H4 SD trajectory (1.8 bohr, D2h; 28 allowed spin-free elements,
r = 0.3; C(Psi_k) from ``ft.stateprep``: 0, 160, 1352, 2468, 3622, 4814; DF walk at
threshold 3e-3: lambda = 4.935, C_W = 868; one thread, 4 s with the inputs): direct
sampling is cheapest at iterates 0-2 (0, 5.8e6, 2.0e8 Toffolis) and the X-LCU route from
iterate 3 at kappa = 1 (4.4e8 against 5.2e8 for sampling; 1.4e9 against 4.2e9 at iterate
5), from iterate 4 at kappa = 2; the aux sampling route is last throughout. The line
search (V_E = 0.56-0.61 Ha^2) costs 0.8e9-1.6e10 Toffolis, 11-605 times the cheapest
residual at iterates 1-5. The reflection term Q_run / 2 is at most 10% of an AE
application (ae_product at iterate 1), 3% for ae_x.

Flags: "infeasible" (B = 0), "estimate" (a Toffoli or qubit count above is an estimate),
"assumption" (gradient estimation), "no_aux_bias" (``aux_bias`` None: the aux routes
are costed without their O(delta^2) bias, a lower bound), "qubits_upper_bound"
(:func:`energy_ae` without N).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from ..measure import routes
from ..measure.targets import KINDS
from .primitives import FTParameters, evolution_toffolis

__all__ = [
    "WalkParams",
    "PrepParams",
    "ResidualProblem",
    "RouteCost",
    "ROUTES",
    "variance_budget",
    "ae_allocation",
    "sampling_direct",
    "sampling_aux",
    "ae_product",
    "ae_x_lcu",
    "ae_aux",
    "gradient_estimation",
    "compare_routes",
    "best_route",
    "energy_sampling",
    "line_search_cost",
    "energy_ae",
    "crossover_norm",
]

#: Residual routes, in the order of :func:`compare_routes`.
ROUTES = ("direct_mg", "aux_mg", "ae_product", "ae_x", "ae_aux", "gaw")
# Routes with a kappa_g-free cost, which crossover_norm compares.
_CROSSOVER_ROUTES = ROUTES[:-1]

# Toffolis of a controlled block encoding of Q_e [estimate]: a 3-AND and a controlled Y
# (spin-orbital), an LCU of up to 4 spin terms (spin-free).
_C_Q = {"spin_orbital": 4, "spin_free": 16}
# Round-off rule of routes.shots_for_relative_error: infeasible if B^2 <= 1e-12 r^2 ||A||_m^2.
_BUDGET_RTOL = 1e-12
# crossover_norm: scale range (relative to ||A||_m), grid points per decade, bisection steps.
_CROSS_DECADES = 4
_CROSS_PER_DECADE = 20
_CROSS_BISECT = 60


# ------------------------------------------------------------------------- validation
def _real(name: str, value, *, positive: bool = False) -> float:
    x = float(value)
    if not math.isfinite(x) or x < 0.0 or (positive and x == 0.0):
        raise ValueError(f"{name} must be {'positive' if positive else 'non-negative'} and finite, got {value!r}")
    return x


def _count(name: str, value, least: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < least:
        raise ValueError(f"{name} must be an integer >= {least}, got {value!r}")
    return int(value)


def _frozen_vector(name: str, values, k: int, *, dtype=float, positive: bool = False) -> np.ndarray:
    a = np.array(values, dtype=dtype)
    if a.shape != (k,):
        raise ValueError(f"{name} must have shape ({k},), got {a.shape}")
    if not np.all(np.isfinite(a)) or np.any(a < 0) or (positive and np.any(a == 0)):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'non-negative'}")
    a.setflags(write=False)
    return a


# ------------------------------------------------------------------------- data types
@dataclass(frozen=True)
class WalkParams:
    """A qubitized block encoding of a factorized Hamiltonian H' (from ``ft.qpe.BlockEncoding.walk_params``).

    Attributes:
        lam: normalization lambda of the block encoding (Hartree; OpenFermion's convention).
        toffoli_per_step: C_W, Toffolis of one controlled walk step [OF].
        logical_qubits: Q_walk, OpenFermion's logical qubits minus the QPE control register;
            it includes the 2N system qubits.
        kind: "df", "thc", "sf" (informational).
        fact_bias_m: ||A[H'] - A[H]||_m at the current state (Hartree), the factorization bias
            of every route that uses this encoding (``residual_inputs.factorization_bias``).
    """

    lam: float
    toffoli_per_step: int
    logical_qubits: int
    kind: str = "df"
    fact_bias_m: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "lam", _real("lam", self.lam, positive=True))
        object.__setattr__(self, "toffoli_per_step", _count("toffoli_per_step", self.toffoli_per_step, 0))
        object.__setattr__(self, "logical_qubits", _count("logical_qubits", self.logical_qubits, 1))
        object.__setattr__(self, "fact_bias_m", _real("fact_bias_m", self.fact_bias_m))


@dataclass(frozen=True)
class PrepParams:
    """The circuit of the current state Psi_k (``PrepParams(prep.cumulative[k], prep.logical_qubits[k])``).

    Attributes:
        toffolis: C(Psi_k), Toffolis of one preparation (0 for HF, which is Clifford).
        logical_qubits: Q(Psi_k), including the 2N system qubits.
    """

    toffolis: float
    logical_qubits: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "toffolis", _real("toffolis", self.toffolis))
        object.__setattr__(self, "logical_qubits", _count("logical_qubits", self.logical_qubits, 1))


@dataclass(frozen=True, eq=False)
class ResidualProblem:
    """Per-element inputs of the residual routes at one state (``residual_inputs.build_problem``).

    Attributes:
        kind: "spin_free" or "spin_orbital".
        elements: the (allowed) elements, in the format of ``targets.element_set``: (K, 4)
            orbitals or (K, 4, 2) spin orbitals.
        m: (K,) multiplicities (``targets.element_multiplicities``).
        A: (K,) exact residual on the elements at the state (Hartree).
        v_direct: (K,) per-shot variances of the direct route under matchgate shadows (Ha^2).
        w_aux: (K,) delta^2 times the per-shot variances of the aux route under matchgate
            shadows (dimensionless; independent of delta to O(delta^2)).
        lam_q: (K,) Pauli 1-norms of Q_e, identity excluded (dimensionless).
        alpha_x: (K,) Pauli 1-norms of X_e, identity excluded (Hartree).
        n_terms_x: (K,) numbers of non-identity Pauli terms of X_e (L_X).
        aux_bias: delta -> ||b(delta)||_m, the aux-route bias (Hartree); None costs the aux
            routes without it (flag "no_aux_bias").
        norb: N, the number of active orbitals (2N system qubits, shared by the state and
            the block encodings in the qubit counts). None takes the largest orbital index
            of ``elements`` plus one, which is N for every symmetry-allowed set of N >= 2
            orbitals (pair elements such as (p, q, p, q) are always allowed) and at most N
            for a subset, so the qubit counts err high, never low.

    The arrays are copied and made read-only.
    """

    kind: str
    elements: np.ndarray
    m: np.ndarray
    A: np.ndarray
    v_direct: np.ndarray
    w_aux: np.ndarray
    lam_q: np.ndarray
    alpha_x: np.ndarray
    n_terms_x: np.ndarray
    aux_bias: Callable[[float], float] | None
    norb: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        el = np.array(self.elements)
        shape = (4, 2) if self.kind == "spin_orbital" else (4,)
        if el.ndim != 1 + len(shape) or el.shape[1:] != shape or el.shape[0] == 0 or el.dtype.kind not in "iu":
            raise ValueError(f"{self.kind} elements must be a non-empty integer array of shape (K, "
                             f"{', '.join(map(str, shape))}), got {el.shape} ({el.dtype})")
        orbitals = el[..., 0] if self.kind == "spin_orbital" else el
        if orbitals.min() < 0:
            raise ValueError("element orbital indices must be non-negative")
        least = int(orbitals.max()) + 1
        norb = least if self.norb is None else _count("norb", self.norb, least)
        object.__setattr__(self, "norb", norb)
        el.setflags(write=False)
        object.__setattr__(self, "elements", el)
        k = el.shape[0]
        object.__setattr__(self, "m", _frozen_vector("m", self.m, k, positive=True))
        a = np.array(self.A, dtype=float)
        if a.shape != (k,) or not np.all(np.isfinite(a)):
            raise ValueError(f"A must be a finite real array of shape ({k},)")
        a.setflags(write=False)
        object.__setattr__(self, "A", a)
        for name in ("v_direct", "w_aux", "alpha_x"):
            object.__setattr__(self, name, _frozen_vector(name, getattr(self, name), k))
        object.__setattr__(self, "lam_q", _frozen_vector("lam_q", self.lam_q, k, positive=True))
        nt = np.array(self.n_terms_x)
        if nt.dtype.kind not in "iu" and not np.array_equal(nt, np.round(nt)):
            raise ValueError("n_terms_x must hold integers")
        object.__setattr__(self, "n_terms_x", _frozen_vector("n_terms_x", nt, k, dtype=np.int64))
        if self.aux_bias is not None and not callable(self.aux_bias):
            raise TypeError("aux_bias must be callable (delta -> ||b(delta)||_m) or None")

    @property
    def n_elements(self) -> int:
        """K, the number of elements."""
        return int(self.A.size)

    @property
    def norm_a_m(self) -> float:
        """||A||_m = sqrt(sum_e m_e A_e^2), the Frobenius norm of the full tensor (Hartree)."""
        return math.sqrt(float(np.sum(self.m * self.A ** 2)))


@dataclass(frozen=True)
class RouteCost:
    """Cost of one estimate (the residual by one route, or energies) at one state.

    Attributes:
        route: one of :data:`ROUTES`, or "line_search", "energy_sampling", "energy_ae".
        n_prep: state preparations: shots, or applications of 𝒜 and 𝒜^dagger.
        toffolis: total Toffolis.
        toffolis_per_prep: toffolis / n_prep (nan if infeasible).
        coherent_toffolis: Toffolis of the longest single coherent run (one shot, one AE run,
            or one gradient-estimation block).
        n_runs: independent runs: shots, AE runs (one per element), or blocks.
        logical_qubits: logical qubits of the largest run.
        params: route parameters and diagnostics: ``budget`` (B, Hartree), ``bias_m``,
            ``delta``, ``kappa``, ``kappa_g``, ``n_blocks``, ``c_obs``, ...
        flags: "infeasible", "estimate", "assumption", "no_aux_bias", "qubits_upper_bound".
    """

    route: str
    n_prep: float
    toffolis: float
    toffolis_per_prep: float
    coherent_toffolis: float
    n_runs: float
    logical_qubits: int
    params: dict
    flags: tuple[str, ...] = ()


def _infeasible(route: str, logical_qubits: int, params: dict, flags: tuple[str, ...]) -> RouteCost:
    return RouteCost(route, math.inf, math.inf, math.nan, math.inf, math.inf, int(logical_qubits),
                     dict(params, budget=0.0), tuple(flags) + ("infeasible",))


# ------------------------------------------------------------------------- budgets
def variance_budget(norm_a_m: float, r: float, bias_m: float = 0.0) -> float:
    """Variance budget B = sqrt(r^2 ||A||_m^2 - bias_m^2) (Hartree, F10); 0.0 if infeasible.

    Infeasible means B^2 <= 1e-12 r^2 ||A||_m^2 (the bias alone exhausts the target, or
    does so to round-off), the rule of ``routes.shots_for_relative_error``.
    """
    norm_a_m = _real("norm_a_m", norm_a_m)
    r = _real("r", r, positive=True)
    bias_m = _real("bias_m", bias_m)
    target = (r * norm_a_m) ** 2
    budget = target - bias_m ** 2
    if budget <= _BUDGET_RTOL * target:
        return 0.0
    return math.sqrt(budget)


def ae_allocation(alpha: np.ndarray, m: np.ndarray, budget: float) -> tuple[float, np.ndarray]:
    """Optimal per-element AE errors under sum_e m_e eps_e^2 = B^2 (F11): (Gamma / B, eps_e).

    Minimizes sum_e alpha_e / eps_e (proportional to the applications sum_e N_e): the
    stationarity condition gives eps_e = t (alpha_e / m_e)^(1/3) with
    t = B / (sum_e m_e^(1/3) alpha_e^(2/3))^(1/2), and the minimum is
    Gamma / B = (sum_e m_e^(1/3) alpha_e^(2/3))^(3/2) / B. Elements with alpha_e = 0 get
    eps_e = 0 (they need no applications). Units: eps_e those of alpha_e and B.
    """
    alpha = np.asarray(alpha, dtype=float)
    m = np.asarray(m, dtype=float)
    if alpha.ndim != 1 or m.shape != alpha.shape:
        raise ValueError(f"alpha and m must be 1-D arrays of one shape, got {alpha.shape} and {m.shape}")
    if not (np.all(np.isfinite(alpha)) and np.all(alpha >= 0) and np.all(np.isfinite(m)) and np.all(m > 0)):
        raise ValueError("alpha must be finite and non-negative, m finite and positive")
    budget = _real("budget", budget, positive=True)
    s = float(np.sum(np.cbrt(m) * np.cbrt(alpha) ** 2))
    if s == 0.0:
        return 0.0, np.zeros_like(alpha)
    t = budget / math.sqrt(s)
    return s ** 1.5 / budget, t * np.cbrt(alpha / m)


def _c_evol(walk: WalkParams, delta: float, params: FTParameters) -> int:
    return evolution_toffolis(walk.lam, delta, walk.toffoli_per_step, delta=params.eps_evol, beta=params.beta)


def _aux_bias(problem: ResidualProblem, delta: float) -> float:
    return 0.0 if problem.aux_bias is None else _real("aux_bias(delta)", problem.aux_bias(delta))


def _aux_flags(problem: ResidualProblem) -> tuple[str, ...]:
    return () if problem.aux_bias is not None else ("no_aux_bias",)


def _cheapest(costs: Sequence[RouteCost], route: str, logical_qubits: int, params: dict,
              flags: tuple[str, ...]) -> RouteCost:
    """The cheapest feasible cost over the delta grid (first on ties), or an infeasible one."""
    feasible = [c for c in costs if "infeasible" not in c.flags]
    if not feasible:
        return _infeasible(route, logical_qubits, dict(params, delta=None), flags)
    return min(feasible, key=lambda c: c.toffolis)


# ------------------------------------------------------------------------- sampling routes
def sampling_direct(problem: ResidualProblem, prep: PrepParams, *,
                    params: FTParameters = FTParameters()) -> RouteCost:
    """Direct route with matchgate shadows (F10): S = sum_e m_e v_e / B^2 shots of C(Psi_k) Toffolis.

    S is ``routes.shots_for_relative_error(v_direct, A, r, weights=m)`` (M2's rule, exactly).
    The readout is Clifford, so a shot costs the preparation alone; Q = Q(Psi_k).
    """
    shots = routes.shots_for_relative_error(problem.v_direct, problem.A, params.r, weights=problem.m)
    info = {"r": params.r, "variance_m": float(np.sum(problem.m * problem.v_direct)), "bias_m": 0.0}
    if not math.isfinite(shots):
        return _infeasible("direct_mg", prep.logical_qubits, info, ())
    info["budget"] = variance_budget(problem.norm_a_m, params.r)
    return RouteCost("direct_mg", shots, shots * prep.toffolis, prep.toffolis, prep.toffolis, shots,
                     prep.logical_qubits, info)


def sampling_aux(problem: ResidualProblem, prep: PrepParams, walk: WalkParams, *,
                 params: FTParameters = FTParameters()) -> RouteCost:
    """Aux route with matchgate shadows on exp(-i delta H') Psi_k (F10), delta optimized.

    S(delta) = sum_e m_e w_e / (delta^2 B(delta)^2) shots of C(Psi_k) + C_evol(delta)
    Toffolis, B(delta) with the bias ||b(delta)||_m + ``walk.fact_bias_m``; the delta of
    ``params.aux_deltas`` with the fewest Toffolis is taken. Q = Q(Psi_k) + Q_walk + 1 - 2N.
    """
    w_m = float(np.sum(problem.m * problem.w_aux))
    q = prep.logical_qubits + walk.logical_qubits + 1 - 2 * problem.norb
    flags = _aux_flags(problem)
    info = {"r": params.r, "w_m": w_m, "walk_kind": walk.kind, "fact_bias_m": walk.fact_bias_m}
    costs = []
    for delta in params.aux_deltas:
        aux_b = _aux_bias(problem, delta)
        bias = aux_b + walk.fact_bias_m
        budget = variance_budget(problem.norm_a_m, params.r, bias)
        p = dict(info, delta=delta, aux_bias_m=aux_b, bias_m=bias)
        if budget == 0.0:
            costs.append(_infeasible("aux_mg", q, p, flags))
            continue
        shots = w_m / (delta ** 2 * budget ** 2)
        c_evol = _c_evol(walk, delta, params)
        per = prep.toffolis + c_evol
        p.update(budget=budget, c_evol=c_evol)
        costs.append(RouteCost("aux_mg", shots, shots * per, per, per, shots, q, p, flags))
    return _cheapest(costs, "aux_mg", q, info, flags)


# ------------------------------------------------------------------------- amplitude estimation
def _ae_route(route: str, problem: ResidualProblem, prep: PrepParams, *, alpha: np.ndarray,
              c_obs, q_obs, bias_m: float, kappa: float, params: FTParameters,
              info: dict, flags: tuple[str, ...]) -> RouteCost:
    """Per-element amplitude estimation with the optimal allocation (F11; module docstring)."""
    kappa = _real("kappa", kappa, positive=True)
    k = problem.n_elements
    c_obs = np.broadcast_to(np.asarray(c_obs, dtype=float), (k,))
    q_base = prep.logical_qubits + np.broadcast_to(np.asarray(q_obs, dtype=np.int64), (k,)) + 1
    info = dict(info, kappa=kappa, bias_m=bias_m, c_obs=float(np.mean(c_obs)))
    budget = variance_budget(problem.norm_a_m, params.r, bias_m)
    if budget == 0.0:
        return _infeasible(route, int(q_base.max()), info, flags)
    gamma_over_b, eps = ae_allocation(alpha, problem.m, budget)
    n_e = np.divide(2.0 * math.pi * kappa * alpha, eps, out=np.zeros(k), where=eps > 0)
    register = np.where(n_e > 2.0, np.ceil(np.log2(np.maximum(n_e, 2.0) / 2.0)), 0.0).astype(np.int64)
    q_run = q_base + register
    run = n_e * (prep.toffolis + c_obs + 0.5 * q_run)
    n_total, t_total = float(n_e.sum()), float(run.sum())
    info.update(budget=budget, gamma=gamma_over_b * budget, max_applications=float(n_e.max()))
    return RouteCost(route, n_total, t_total, t_total / n_total if n_total > 0 else math.nan,
                     float(run.max()), float(np.count_nonzero(n_e)), int(q_run.max()), info, flags)


def ae_product(problem: ResidualProblem, prep: PrepParams, walk: WalkParams, *,
               kappa: float | None = None, params: FTParameters = FTParameters()) -> RouteCost:
    """AE of A_e = -2 Im <Q_e H'> through the Hadamard test on U_Q U_H (route b1, F11).

    alpha_e = 2 lambda_Q,e lambda, C_obs = C_W + C_Q, Q_obs = Q_walk - 2N + 4; bias: the
    factorization bias ``walk.fact_bias_m``. ``kappa`` defaults to ``params.kappa``.
    """
    kappa = params.kappa if kappa is None else kappa
    return _ae_route("ae_product", problem, prep, alpha=2.0 * problem.lam_q * walk.lam,
                     c_obs=walk.toffoli_per_step + _C_Q[problem.kind],
                     q_obs=walk.logical_qubits - 2 * problem.norb + 4, bias_m=walk.fact_bias_m,
                     kappa=kappa, params=params, info={"walk_kind": walk.kind}, flags=("estimate",))


def ae_x_lcu(problem: ResidualProblem, prep: PrepParams, *, kappa: float | None = None,
             params: FTParameters = FTParameters()) -> RouteCost:
    """AE of <X_e> through the Pauli LCU of X_e (route b2, F11).

    alpha_e = alpha_X,e, C_obs,e = 2 L_X,e (QROM PREPARE plus unary SELECT) [estimate],
    Q_obs,e = ceil(log2 L_X,e) + chi + 4; unbiased (exact H). ``kappa`` defaults to
    ``params.kappa``.
    """
    kappa = params.kappa if kappa is None else kappa
    l_x = np.maximum(problem.n_terms_x, 1)
    q_obs = np.ceil(np.log2(l_x)).astype(np.int64) + params.chi + 4
    return _ae_route("ae_x", problem, prep, alpha=problem.alpha_x, c_obs=2.0 * problem.n_terms_x,
                     q_obs=q_obs, bias_m=0.0, kappa=kappa, params=params, info={}, flags=("estimate",))


def ae_aux(problem: ResidualProblem, prep: PrepParams, walk: WalkParams, *,
           kappa: float | None = None, params: FTParameters = FTParameters()) -> RouteCost:
    """AE of <Q_e> on exp(-i delta H') Psi_k (route b3, F11), delta optimized.

    alpha_e = lambda_Q,e / delta, C_obs = C_evol(delta) + C_Q, Q_obs = Q_walk + 1 - 2N + 4;
    bias ||b(delta)||_m + ``walk.fact_bias_m``. The delta of ``params.aux_deltas`` with the
    fewest Toffolis is taken. ``kappa`` defaults to ``params.kappa``.
    """
    kappa = params.kappa if kappa is None else kappa
    q_obs = walk.logical_qubits + 1 - 2 * problem.norb + 4
    flags = ("estimate",) + _aux_flags(problem)
    info = {"walk_kind": walk.kind, "fact_bias_m": walk.fact_bias_m}
    costs = []
    for delta in params.aux_deltas:
        aux_b = _aux_bias(problem, delta)
        c_evol = _c_evol(walk, delta, params)
        costs.append(_ae_route("ae_aux", problem, prep, alpha=problem.lam_q / delta,
                               c_obs=c_evol + _C_Q[problem.kind], q_obs=q_obs,
                               bias_m=aux_b + walk.fact_bias_m, kappa=kappa, params=params,
                               info=dict(info, delta=delta, aux_bias_m=aux_b, c_evol=c_evol), flags=flags))
    return _cheapest(costs, "ae_aux", prep.logical_qubits + q_obs + 1, dict(info, kappa=kappa), flags)


def gradient_estimation(problem: ResidualProblem, prep: PrepParams, walk: WalkParams, *, kappa_g: float,
                        n_blocks: int = 1, b_theta: int = 16,
                        params: FTParameters = FTParameters()) -> RouteCost:
    """Gradient-estimation band, formulation "aux" (F12) [assumption], delta optimized.

    N = 2 pi kappa_g sqrt(K) sqrt(n_B) (sum_e m_e alpha_e^2)^(1/2) / B with
    alpha_e = lambda_Q,e / delta; T = N (C(Psi_k) + C_evol(delta) + C_probe / n_B),
    C_probe = K (2 b_theta + 2). Each of the n_B blocks is one coherent run of N / n_B
    preparations; Q = Q(Psi_k) + Q_walk + 1 - 2N + ceil(K / n_B) b_theta.
    """
    kappa_g = _real("kappa_g", kappa_g, positive=True)
    k = problem.n_elements
    n_blocks = _count("n_blocks", n_blocks, 1)
    if n_blocks > k:
        raise ValueError(f"n_blocks must be at most K = {k}, got {n_blocks}")
    b_theta = _count("b_theta", b_theta, 1)
    c_probe = k * (2 * b_theta + 2)
    q = prep.logical_qubits + walk.logical_qubits + 1 - 2 * problem.norb + math.ceil(k / n_blocks) * b_theta
    flags = ("assumption",) + _aux_flags(problem)
    info = {"kappa_g": kappa_g, "n_blocks": n_blocks, "b_theta": b_theta, "c_probe": c_probe,
            "walk_kind": walk.kind, "fact_bias_m": walk.fact_bias_m}
    costs = []
    for delta in params.aux_deltas:
        aux_b = _aux_bias(problem, delta)
        bias = aux_b + walk.fact_bias_m
        budget = variance_budget(problem.norm_a_m, params.r, bias)
        p = dict(info, delta=delta, aux_bias_m=aux_b, bias_m=bias)
        if budget == 0.0:
            costs.append(_infeasible("gaw", q, p, flags))
            continue
        alpha = problem.lam_q / delta
        n = (2.0 * math.pi * kappa_g * math.sqrt(k * n_blocks)
             * math.sqrt(float(np.sum(problem.m * alpha ** 2))) / budget)
        c_evol = _c_evol(walk, delta, params)
        per = prep.toffolis + c_evol + c_probe / n_blocks
        p.update(budget=budget, c_evol=c_evol)
        costs.append(RouteCost("gaw", n, n * per, per, n * per / n_blocks, float(n_blocks), q, p, flags))
    return _cheapest(costs, "gaw", q, info, flags)


# ------------------------------------------------------------------------- comparison
def compare_routes(problem: ResidualProblem, prep: PrepParams, walk: WalkParams, *,
                   params: FTParameters = FTParameters()) -> list[RouteCost]:
    """Every route (F14): direct_mg, aux_mg, then ae_product, ae_x, ae_aux at ``params.kappa``
    and at ``params.kappa_practical``, then gradient estimation at every ``params.kappa_g``."""
    out = [sampling_direct(problem, prep, params=params), sampling_aux(problem, prep, walk, params=params)]
    for kappa in dict.fromkeys((params.kappa, params.kappa_practical)):
        out += [ae_product(problem, prep, walk, kappa=kappa, params=params),
                ae_x_lcu(problem, prep, kappa=kappa, params=params),
                ae_aux(problem, prep, walk, kappa=kappa, params=params)]
    out += [gradient_estimation(problem, prep, walk, kappa_g=kg, params=params) for kg in params.kappa_g]
    return out


def best_route(costs: Sequence[RouteCost], *, exclude: tuple[str, ...] = ("gaw",),
               kappa: float = 1.0) -> RouteCost:
    """The cheapest cost in Toffolis among ``costs`` (first on ties), F14.

    Routes in ``exclude`` are skipped, and so are AE routes evaluated at another kappa
    (``params["kappa"] != kappa``); the sampling routes carry no kappa and always compete.
    Infeasible routes cost inf. ValueError if nothing is left.
    """
    candidates = [c for c in costs if c.route not in exclude and c.params.get("kappa", kappa) == kappa]
    if not candidates:
        raise ValueError(f"no route left to compare (exclude={exclude}, kappa={kappa})")
    return min(candidates, key=lambda c: c.toffolis)


# ------------------------------------------------------------------------- energies (F13)
def energy_sampling(energy_variance: float, per_shot_toffolis: float, logical_qubits: int, *,
                    eps: float) -> RouteCost:
    """One energy to RMS error ``eps`` (Hartree) by sampling: S = V_E / eps^2 shots.

    ``energy_variance`` is the per-shot variance V_E (Ha^2, e.g. DF basis-rotation grouping,
    ``residual_inputs.energy_variances``); each shot costs ``per_shot_toffolis``.
    """
    v = _real("energy_variance", energy_variance)
    eps = _real("eps", eps, positive=True)
    per = _real("per_shot_toffolis", per_shot_toffolis)
    shots = v / eps ** 2
    return RouteCost("energy_sampling", shots, shots * per, per, per, shots,
                     _count("logical_qubits", logical_qubits, 1), {"eps": eps, "energy_variance": v})


def line_search_cost(energy_variance: float, prep: PrepParams, candidate_step_toffolis: Sequence[float],
                     readout_toffolis: int, *, params: FTParameters = FTParameters(),
                     logical_qubits: int | None = None) -> RouteCost:
    """Line-search energies of one iteration (F13).

    Each candidate Psi_{k+1}(eps_c) is measured with S_E = V_E / eps_ls^2 shots (DF
    basis-rotation grouping) of C(Psi_k) + C_step(k; eps_c) + C_rot Toffolis, so
    T = S_E sum_c (C(Psi_k) + C_step,c + C_rot).

    Args:
        energy_variance: V_E (Ha^2) of the fixed DF variant at Psi_{k+1}.
        prep: the circuit of Psi_k.
        candidate_step_toffolis: C_step(k; eps_c) for every candidate (``stateprep.candidate_step_toffolis``).
        readout_toffolis: C_rot of the DF readout rotation (``stateprep.readout_rotation_toffolis``).
        logical_qubits: qubits of the line-search circuits (default ``prep.logical_qubits``;
            pass the maximum over Q(Psi_k), the candidate steps and the readout rotation).
    """
    v = _real("energy_variance", energy_variance)
    steps = np.asarray(candidate_step_toffolis, dtype=float)
    if steps.ndim != 1 or steps.size == 0 or not np.all(np.isfinite(steps)) or np.any(steps < 0):
        raise ValueError("candidate_step_toffolis must be a non-empty 1-D array of non-negative Toffoli counts")
    c_rot = _count("readout_toffolis", readout_toffolis, 0)
    q = prep.logical_qubits if logical_qubits is None else _count("logical_qubits", logical_qubits, 1)
    shots = v / params.eps_ls ** 2
    per_shot = prep.toffolis + steps + c_rot
    n = shots * steps.size
    t = shots * float(per_shot.sum())
    return RouteCost("line_search", n, t, t / n if n > 0 else math.nan, float(per_shot.max()), n, q,
                     {"n_candidates": int(steps.size), "shots_per_candidate": shots, "eps": params.eps_ls,
                      "energy_variance": v, "readout_toffolis": c_rot})


def energy_ae(walk: WalkParams, prep: PrepParams, *, eps: float, kappa: float = 1.0,
              norb: int | None = None) -> RouteCost:
    """Amplitude-estimated energy of H' to RMS error ``eps`` (Hartree), F13 option.

    The Hadamard test on the controlled walk step has alpha = lambda, so
    N = 2 pi kappa lambda / eps applications (F11) of C(Psi_k) + C_W + Q_run / 2 Toffolis,
    one coherent run on Q_run = Q(Psi_k) + Q_walk - 2N + ceil(log2(N / 2)) + 1 qubits
    (phase register and Hadamard-test qubit; ``norb`` = N, the state and the walk share
    the 2N system qubits). ``norb=None`` counts the system register twice (an upper
    bound by 2N, flag "qubits_upper_bound"). The estimate is the energy of H' (bias
    E(H') - E(H), not counted).
    """
    kappa = _real("kappa", kappa, positive=True)
    eps = _real("eps", eps, positive=True)
    if norb is None:
        shared, flags = 0, ("estimate", "qubits_upper_bound")
    else:
        shared, flags = 2 * _count("norb", norb, 1), ("estimate",)
    n = 2.0 * math.pi * kappa * walk.lam / eps
    q_run = (prep.logical_qubits + walk.logical_qubits - shared
             + (math.ceil(math.log2(n / 2.0)) if n > 2.0 else 0) + 1)
    per = prep.toffolis + walk.toffoli_per_step + 0.5 * q_run
    return RouteCost("energy_ae", n, n * per, per, n * per, 1.0, q_run,
                     {"eps": eps, "kappa": kappa, "walk_kind": walk.kind}, flags)


# ------------------------------------------------------------------------- crossovers
def _route_cost(route: str, problem: ResidualProblem, prep: PrepParams, walk: WalkParams,
                params: FTParameters) -> RouteCost:
    if route == "direct_mg":
        return sampling_direct(problem, prep, params=params)
    if route == "aux_mg":
        return sampling_aux(problem, prep, walk, params=params)
    if route == "ae_product":
        return ae_product(problem, prep, walk, params=params)
    if route == "ae_x":
        return ae_x_lcu(problem, prep, params=params)
    if route == "ae_aux":
        return ae_aux(problem, prep, walk, params=params)
    raise ValueError(f"route must be one of {_CROSSOVER_ROUTES}, got {route!r}")


def crossover_norm(problem: ResidualProblem, prep: PrepParams, walk: WalkParams, route_a: str,
                   route_b: str, *, params: FTParameters = FTParameters()) -> float:
    """||A||_m (Hartree) at which ``route_a`` and ``route_b`` cost the same Toffolis; nan if none.

    A is scaled, A -> t A, with every bias, variance and normalization fixed, and AE at
    ``params.kappa``. Without biases the sampling cost falls as t^-2 and the AE costs as
    about t^-1, so a sampling and an AE route cross once, at

        ||A||*_m = V_m C(Psi_k) / (2 pi kappa Gamma r (C(Psi_k) + C_obs))

    for uniform C_obs (C_obs including the reflection cost Q_run / 2). The crossing is
    searched for t in [1e-4, 1e4] on a grid of 20 points per decade and refined by
    bisection in log t; with several crossings (biases), the largest ||A||_m is returned,
    the first one met as the CQE converges.
    """
    for route in (route_a, route_b):
        if route not in _CROSSOVER_ROUTES:
            raise ValueError(f"routes must be among {_CROSSOVER_ROUTES}, got {route!r}")
    if route_a == route_b:
        raise ValueError(f"need two different routes, got {route_a!r} twice")
    norm = problem.norm_a_m
    if norm == 0.0:
        return math.nan

    def gap(log_t: float) -> float:
        """log T_a - log T_b at A -> exp(log_t) A; +-inf if one route is free or infeasible."""
        scaled = dataclasses.replace(problem, A=problem.A * math.exp(log_t))
        ta = _route_cost(route_a, scaled, prep, walk, params).toffolis
        tb = _route_cost(route_b, scaled, prep, walk, params).toffolis
        if ta == tb:                      # a tie, or both free, or both infeasible
            return 0.0 if 0.0 < ta < math.inf else math.nan
        if ta == 0.0 or tb == 0.0 or math.isinf(ta) or math.isinf(tb):
            return math.copysign(math.inf, ta - tb)
        return math.log(ta) - math.log(tb)

    grid = np.linspace(-_CROSS_DECADES, _CROSS_DECADES, 2 * _CROSS_DECADES * _CROSS_PER_DECADE + 1) * math.log(10)
    values = [gap(x) for x in grid]
    for j in range(len(grid) - 1, 0, -1):     # the largest crossing first
        lo, hi = values[j - 1], values[j]
        if hi == 0.0:
            return norm * math.exp(grid[j])
        if not lo * hi < 0.0:             # same sign, nan, or lo == 0 (returned at the next j)
            continue
        a, b, fa = grid[j - 1], grid[j], lo
        for _ in range(_CROSS_BISECT):
            mid = 0.5 * (a + b)
            fm = gap(mid)
            if fm == 0.0:
                return norm * math.exp(mid)
            if math.isnan(fm):
                break
            if (fm > 0.0) == (fa > 0.0):
                a, fa = mid, fm
            else:
                b = mid
        return norm * math.exp(0.5 * (a + b))
    return norm * math.exp(grid[0]) if values[0] == 0.0 else math.nan
