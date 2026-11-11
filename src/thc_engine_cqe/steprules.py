"""Step rules of the measured CQE loop on the exact CI-space harness (F26, F27, F29, F31, F33).

Tags ([derived], [cited], [design], [build check]) and formula labels are those of
docs/m3-cost-model.md (F1-F19, F2a) and of its M4 continuation (F20-F53). Energies are in Hartree;
step sizes eps multiply the unnormalized direction d, as in ``cqe.run_cqe``.

The loop (:func:`run_step_rule`)
--------------------------------
Iteration k at the state Psi_k:

1. Residual. The exact A_k projected on the symmetry-allowed entries (F2a; ``irreps``), or the
   estimate of ``residual_estimator(rd, ci, k, 0.0)``, which may also return the per-element
   variances V_e of its allocation on the measured set (the symmetry-allowed spin-free elements, in
   the order of ``ft.residual_inputs.allowed_elements``). At k = 0 with ``first="second_order"``
   the estimator is not called: A_0 = W is classical at the RHF determinant (M1), so a start state
   ``ci0`` must then be a determinant (a correlated start takes ``first="line_search"``).
2. Direction. d_k from the optimizer (``optimizers``; L-BFGS by default) with g_k = conj(A_k), and
   the slope the optimizer acts on, s0 = -<g_k, d_k> (``optimizers.direction_slope``).
3. Step eps_k by the rule (below), with the safeguards of F29.
4. Propagation Psi_{k+1} = U(d_k, eps_k) Psi_k: by default exp(eps D) by Taylor ``expm_apply`` at
   tolerance 1e-15 (``ft.trajectory.propagate``); ``propagator(d, eps, ci, k)`` replaces it (e.g. a
   compiled circuit) for every state the loop prepares, trial states included.

Rules (:data:`STEP_RULES`)
    unit         eps_k = 1 for k >= 1: the minimizer of the L-BFGS model (F26), capped by F29.
    parab3       energies at {0, t, 2t}, t = ``parab_t`` (capped so that 2t obeys F29), each at
                 sigma_P3 = ``sigma_parab`` |dE_m| (F27); eps_k from :func:`parabola_step`, interpolation
                 only.
    hybrid       R3: parab3 at iteration k iff ``use_energies(k, dE_m, rd, history)``, the F31 gate
                 (``ft.steprules.planned_energy_gate``); unit otherwise (flags "hybrid:parab3",
                 "hybrid:unit").
    line_search  the exact line search along d_k (``cqe.line_search_step`` at eps_max = 1.5/||G_k||,
                 Krylov tolerance 1e-10, exact energies): the oracle step, M3's trajectory model. Its
                 ledger holds the 11 energies at eps_ls by which M3 costs the exact line search (F13, F14).
    candidates   M3's grid: energies at ``candidates`` (default ``FTParameters().eps_candidates``), each
                 at sigma = ``FTParameters().eps_ls``, the lowest estimate taken (the first on ties).
With ``first="second_order"`` (the default) every rule takes the classical second-order step
eps_2 = -E'(0)/E''(0) of F16 at k = 0 (``ft.trajectory.f16``), uncapped; ``first="line_search"``
measures A_0 and takes the exact line search (M3's measured loop). The model rules (unit, parab3,
hybrid) need L-BFGS directions: only for them is the unit step the model step.

F26 (the L-BFGS model step is the unit step) [derived]
------------------------------------------------------
With d_k = H_k g_k, B_k = H_k^-1 and s0 = -<g_k, d_k>, <d_k, B_k d_k> = <g_k, H_k g_k> = -s0, so the
model m(eps) = E_k + eps s0 + eps^2 <d, B d>/2 has its minimizer at eps = 1 and predicts
dE_m = s0/2. The same holds for estimates (d^ = H g^, s0^ = -<g^, H g^>). tests/test_steprules.py
checks <d, B d> = -s0 with B from the BFGS recursion of the stored pairs.

F27 (precision of the energies of a 3-point fit) [derived; Monte-Carlo calibrated]
---------------------------------------------------------------------------------
For energies at {0, t, 2t} with errors of RMS sigma and u = eps*/t, the linearized error of the fitted
minimizer is (delta eps/eps*)^2 = sigma^2 u^2 f(u)/(4 dE*^2), f(u) = (u - 3/2)^2 + 4(1 - u)^2 + (u - 1/2)^2;
a parabola loses the fraction L = (delta eps/eps*)^2 of its decrease, so L = 0.1 needs
sigma = sqrt(8L)|dE_m| = 0.894|dE_m| in the linearization. With the interpolation-only rule of
:func:`parabola_step` the expected loss is 0.10 at sigma_P3 = 0.47|dE_m| (Monte Carlo on the
parabola, 4e5 draws; tests/test_steprules.py::test_f27_monte_carlo): the default ``sigma_parab``.
dE_m = s0'/2 is the model decrease of the unit step with the debiased slope of F33. [build check, 4e5
draws: the linearized ratio is 0.999 / 1.003 / 1.018 at u = 1/2, 1, 2; the expected loss is 0.026,
0.092, 0.101, 0.116 and 0.264 at sigma/|dE| = 0.3, 0.45, 0.47, 0.5 and 0.894.]

F29 (safeguards) [design]
-------------------------
At k >= 1, before any measurement of the iteration's energies:

* growth cap: eps_k <= gamma_s ||s_{k-1}||_F / ||d_k||_F with s_{k-1} = eps_{k-1} d_{k-1} the previous
  accepted step (Frobenius norm of the pair-order tensor), gamma_s = ``growth`` (flag "growth_cap");
* angle cap: eps_k ||G_k|| <= ``angle_cap`` with ||G_k|| from 6 power iterations on the CI space
  (``cqe.generator_norm_estimate``; run_cqe brackets its line search at 1.5/||G_k||): an
  emulation-only safeguard, since ||G_k|| is not measured on hardware (flag "angle_cap", the
  binding events counted by the flags).

Both cap the unit step and the stencil's far point 2t (hence the fitted step); a flag names the cap
that sets the step. The classical step eps_2 at k = 0 and the two M3 references (line_search,
candidates) are not capped. [build check: on exact 20-step unit runs of h4_r{1.8, 3.0, 4.0} and
n2_r{2.0, 2.4}, both metrics, the growth cap at gamma_s = 2 binds in 2-6 of the 19 measured steps,
first at k = 4-12, on the chains mostly after k* (h4_r1.8 spin-orbital: k = 7, 8, 13, 16 with
k* = 3); the angle cap at 3 never binds there.]

F31 (hybrid switch) [design]
----------------------------
The gate is a hook: ``use_energies(k, dE_m, rd, history) -> bool`` with dE_m = s0'/2, ``rd`` the
exact ResidualData of Psi_k and ``history`` = (directions, eps): the directions d_0 .. d_k, the
current one d_k last, and the steps eps_0 .. eps_{k-1} taken so far (read-only views), from which the
planned costs of the iteration are compiled (``ft.steprules.planned_energy_gate``: C(Psi_k) from the
steps taken, C_step(k; t) and C_step(k; 2t) along d_k, 3 energies at sigma_P3 against the planned
residual, threshold ``theta``).

F33 (debiased slope) [derived]
------------------------------
With A^ = A + eta, eta = sum_e eta_e u_e (u_e the unit fill of element e, ``measure.targets.spin_free_tensor``),
E eta_e^2 = V_e independent and H_k independent of eta, E s0^ = s0 - tau with
tau = E<eta, H_k eta> = sum_e V_e <u_e, H_k u_e>, so s0' = s0^ + tau is unbiased (:func:`debiased_slope`, with
``LBFGS.inverse_hessian_apply`` for H_k). The newest L-BFGS pair (y_{k-1} from g^_k) depends on eta_k, so in a
closed loop s0' is an approximation (flag "slope_debias_approx").

Ledger and convergence
----------------------
``StepRuleResult.events`` lists the measurements a costing charges (``ft.steprules``): one "residual"
event at Psi_k per measured iteration, and one "energy" event per energy (eps along d_k, the requested
sigma). The run stops before a step whose predicted lowering is below 1e-10 Ha (the F16 converged-state
rule: s0^2/(2E''(0)) at a classical k = 0, |s0^|/2 of F26 otherwise; ``converged``), or after
``max_iter`` steps.

Parity (symmetry-adapted builds, one thread) [build check; tests/test_steprules.py]: with the caps off
and exact residuals the unit rule equals ``run_cqe(..., optimizer="lbfgs", first_eps=eps_2,
eps_candidates=(1.0,), residual_estimator=mask)`` to <= 1.4e-14 Ha in every energy (h4_r1.8, n2_r2.0,
both metrics, 10 steps; the two propagate by Taylor and Krylov expansions), and the line-search rule
equals ``ft.trajectory.exact_trajectory(..., first_step="second_order")`` to <= 1.9e-11 Ha (h4, both
metrics, 8 steps; the flat late line searches locate their minima only to sqrt(|E| u / dE)). With
``first="line_search"`` the line-search rule is M3's exact record (``exact_trajectory``,
first_step="exact") bit for bit in eps and every energy (h4, n2s, h6s, both metrics, 12 steps). F26 holds
to 9.4e-16 relative on the 44 L-BFGS directions with pairs of 12-step unit runs of h4_r3.0 and n2_r2.0.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Callable, Mapping, NamedTuple

import numpy as np

from .cqe import CQEStep, generator_norm_estimate, line_search_step
from .fci_ops import Hamiltonian, normalize
from .ft.primitives import FTParameters
from .ft.symmetry import element_mask, mask_pair_tensor
from .ft.trajectory import f16, propagate
from .integrals import ActiveSpace
from .measure import targets
from .optimizers import OPTIMIZERS, apply_metric, direction_slope, make_optimizer, real_inner, structure_error
from .residual import ResidualData, compute_residual, descent_generator, spin_free_norm

__all__ = [
    "STEP_RULES",
    "FIRST_STEPS",
    "StepRule",
    "StepEvent",
    "StepRuleResult",
    "run_step_rule",
    "parabola_step",
    "debiased_slope",
    "unit_fills",
]

STEP_RULES = ("unit", "hybrid", "parab3", "line_search", "candidates")
FIRST_STEPS = ("second_order", "line_search")
# Rules whose step is the L-BFGS model step or a fit around it (F26, F27): L-BFGS only, F29 caps apply.
_MODEL_RULES = ("unit", "parab3", "hybrid")
# M3's line search (F13): candidate grid and energy precision.
_M3 = FTParameters()
# The F16 converged-state rule: a predicted lowering below this (Ha) means d is solver noise.
_MIN_LOWERING = 1e-10
# run_cqe's line-search bracket 1.5/||G|| and Krylov tolerance (the line_search rule reproduces it).
_LS_BRACKET = 1.5
_LS_TOL = 1e-10
# Relative tolerance on the structure of a residual estimate (as cqe.run_cqe).
_STRUCTURE_RTOL = 1e-10
# A start state with one amplitude above this (normalized) is a determinant (the classical first step's start).
_DETERMINANT_TOL = 1e-12
# parabola_step: the fitted step is floored at this fraction of t; the fallback when E(0) is lowest.
_PARAB_FLOOR = 0.05
_PARAB_FALLBACK = 0.25
_PARAB_CEILING = 3.0


def _finite_positive(name: str, x, *, allow_inf: bool = False) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number, got {x!r}") from None
    if not (v > 0.0 and (math.isfinite(v) or (allow_inf and v == math.inf))):
        raise ValueError(f"{name} must be positive{'' if allow_inf else ' and finite'}, got {x!r}")
    return v


@dataclass(frozen=True)
class StepRule:
    """A step rule of :func:`run_step_rule` (module docstring).

    Attributes:
        name: one of :data:`STEP_RULES`.
        first: the step at k = 0: "second_order" (eps_2 of F16, classical, no measurement) or
            "line_search" (A_0 measured, exact line search: M3's measured loop).
        growth: gamma_s of the F29 growth cap (> 1), None: off. Applies to the model rules only.
        angle_cap: the F29 angle cap eps ||G|| <= angle_cap (> 0; emulation only), None: off.
        sigma_parab: sigma_P3/|dE_m|, the precision of the parab3 energies (F27; 0.47 at L = 0.1).
        parab_t: the stencil {0, t, 2t} in units of the model step (t = 1: eps = 1 at its centre).
        theta: the F31 switch threshold of the planned-cost gate (``ft.steprules.planned_energy_gate``).
        candidates: the step sizes of "candidates" and the energy grid by which "line_search" is
            costed; empty: ``FTParameters().eps_candidates`` (M3's 0.00625 * 2^j, j < 11). Must be
            empty for the model rules.
    """

    name: str = "unit"
    first: str = "second_order"
    growth: float | None = 2.0
    angle_cap: float | None = 3.0
    sigma_parab: float = 0.47
    parab_t: float = 1.0
    theta: float = 1.0
    candidates: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.name not in STEP_RULES:
            raise ValueError(f"name must be one of {STEP_RULES}, got {self.name!r}")
        if self.first not in FIRST_STEPS:
            raise ValueError(f"first must be one of {FIRST_STEPS}, got {self.first!r}")
        if self.growth is not None:
            g = _finite_positive("growth", self.growth)
            if not g > 1.0:
                raise ValueError(f"growth (gamma_s) must exceed 1 (or be None), got {self.growth!r}")
            object.__setattr__(self, "growth", g)
        if self.angle_cap is not None:
            object.__setattr__(self, "angle_cap", _finite_positive("angle_cap", self.angle_cap))
        object.__setattr__(self, "sigma_parab", _finite_positive("sigma_parab", self.sigma_parab))
        object.__setattr__(self, "parab_t", _finite_positive("parab_t", self.parab_t))
        object.__setattr__(self, "theta", _finite_positive("theta", self.theta, allow_inf=True))
        try:
            cand = tuple(float(c) for c in self.candidates)
        except (TypeError, ValueError):
            raise ValueError(f"candidates must be a sequence of real step sizes, got {self.candidates!r}") from None
        if not all(math.isfinite(c) for c in cand):
            raise ValueError(f"candidates must be finite, got {self.candidates!r}")
        if cand and self.name in _MODEL_RULES:
            raise ValueError(f"candidates are used by the 'candidates' and 'line_search' rules only, not {self.name!r}")
        if not cand and self.name in ("candidates", "line_search"):
            cand = tuple(float(c) for c in _M3.eps_candidates)
        object.__setattr__(self, "candidates", cand)

    @property
    def model_rule(self) -> bool:
        """True for the rules that take the L-BFGS model step or fit around it (unit, parab3, hybrid)."""
        return self.name in _MODEL_RULES


class StepEvent(NamedTuple):
    """One measurement the rule charges (module docstring, "Ledger")."""

    iteration: int
    kind: str                # "residual" | "energy"
    eps: float               # along d_k from Psi_k (0: Psi_k itself)
    sigma: float | None      # energies: the requested precision (Ha); residuals: None
    used_for_step: bool


@dataclass
class StepRuleResult:
    """A run of :func:`run_step_rule`.

    Attributes:
        steps: ``cqe.CQEStep`` of every step taken, with run_cqe's meanings: exact energy, variance,
            ||A|| and slope -Re sum d A of the state before the step, full_slope -||A||^2, eps,
            energy_after, the direction (``generator``), residual_norm_est = ||A^|| and slope_est = s0^
            of the residual the driver acted on (masked or estimated), energy_est / energy_after_est
            where the rule measured E(Psi_k) / the energy of the chosen step, n_energy_evals the
            energies charged in the iteration, optimizer_info.
        events: the measurement ledger (:class:`StepEvent`).
        energies: exact E_0 .. E_K.
        directions, eps: d_k and eps_k of the K steps.
        flags: per step: "first:second_order", "first:line_search", "growth_cap", "angle_cap",
            "hybrid:parab3", "hybrid:unit", "parab:clipped", "parab:floor", "parab:ceiling",
            "slope_debias_approx", "lbfgs_reset", "zero_step".
        states: Psi_0 .. Psi_K (normalized CI arrays) with ``keep_states``, else None.
        optimizer, metric: the optimizer's name and metric.
        rule: the :class:`StepRule`.
        converged: the run stopped on the F16 converged-state rule (module docstring).
        de_model: dE_m = s0'/2 of every step (Ha; F26 with the F33 slope), the classical step's
            predicted lowering -s0^2/(2E''(0)) at a classical k = 0.
    """

    steps: list[CQEStep]
    events: list[StepEvent]
    energies: np.ndarray
    directions: list[np.ndarray]
    eps: list[float]
    flags: list[tuple[str, ...]]
    states: list[np.ndarray] | None
    optimizer: str
    metric: str
    rule: StepRule
    converged: bool
    de_model: list[float] | None = None

    @property
    def n_steps(self) -> int:
        """K, the number of steps taken."""
        return len(self.eps)

    def iteration_events(self, k: int) -> list[StepEvent]:
        """The ledger entries of iteration k."""
        return [e for e in self.events if e.iteration == k]


# ------------------------------------------------------------------------- pure helpers
def parabola_step(e0: float, e1: float, e2: float, t: float, *,
                  interpolate_only: bool = True) -> tuple[float, str]:
    """Step from energies at {0, t, 2t}: the minimizer of the interpolating parabola (F27).

    c = (e0 - 2 e1 + e2)/t^2 and b = (-3 e0 + 4 e1 - e2)/(2t) are the curvature and the slope at 0; the
    fitted minimizer is x = -b/c for c > 0. Returns (eps, status):

    * ``interpolate_only`` (default): x if c > 0 and 0 < x <= 2t, floored at 0.05t ("fit", "floor");
      otherwise the best measured point, t or 2t (the lower, t on ties), or 0.25t if E(0) is the lowest
      ("clipped").
    * otherwise: x clipped to [0.05t, 3t] for c > 0 ("fit", "floor", "ceiling"); for c <= 0 the best
      measured point as above ("clipped").
    """
    t = _finite_positive("t", t)
    vals = [float(e0), float(e1), float(e2)]
    if not all(math.isfinite(v) for v in vals):
        raise ValueError(f"energies must be finite, got {vals}")
    e0, e1, e2 = vals
    c = (e0 - 2.0 * e1 + e2) / t ** 2
    b = (-3.0 * e0 + 4.0 * e1 - e2) / (2.0 * t)

    def best_point() -> tuple[float, str]:
        if min(e1, e2) < e0:
            return (t if e1 <= e2 else 2.0 * t), "clipped"
        return _PARAB_FALLBACK * t, "clipped"

    if not c > 0.0:
        return best_point()
    x = -b / c
    if interpolate_only:
        if not 0.0 < x <= 2.0 * t:
            return best_point()
        return (x, "fit") if x >= _PARAB_FLOOR * t else (_PARAB_FLOOR * t, "floor")
    if x < _PARAB_FLOOR * t:
        return _PARAB_FLOOR * t, "floor"
    if x > _PARAB_CEILING * t:
        return _PARAB_CEILING * t, "ceiling"
    return x, "fit"


def unit_fills(elements: np.ndarray, norb: int) -> np.ndarray:
    """(K, N, N, N, N) unit fills u_e of the spin-free elements: ``targets.spin_free_tensor`` of the e-th unit vector.

    u_e[x] = u_e[P x] = 1 and u_e[T x] = u_e[P T x] = -1 on the orbit of element x = (p, q, r, s)
    (P: (pq, rs) -> (rs, pq), T: (pq, rs) -> (qp, sr)), so A = sum_e A_e u_e and <u_e, u_e> = m_e.
    """
    el = np.asarray(elements)
    n = int(norb)
    targets.spin_free_tensor(np.arange(1.0, el.shape[0] + 1.0), el, n)        # validates distinct orbits
    out = np.zeros((el.shape[0],) + (n,) * 4)
    idx = np.arange(el.shape[0])
    p, q, r, s = el.T
    out[idx, p, q, r, s] = 1.0
    out[idx, r, s, p, q] = 1.0
    out[idx, q, p, s, r] = -1.0
    out[idx, s, r, q, p] = -1.0
    return out


def debiased_slope(s0_hat: float, var_e: np.ndarray, elements: np.ndarray, norb: int,
                   inverse_hessian: Callable[[np.ndarray], np.ndarray]) -> float:
    """F33: s0' = s0^ + tau, tau = sum_e V_e <u_e, H u_e> (module docstring).

    ``var_e`` (K,) are the variances of the element estimates (Ha^2), ``elements`` (K, 4) the spin-free
    elements, ``inverse_hessian`` the operator H of the direction (``LBFGS.inverse_hessian_apply``, or
    the metric M for steepest descent). Unbiased when H does not depend on the estimate.
    """
    v = np.asarray(var_e, dtype=float).ravel()
    el = np.asarray(elements)
    if el.ndim != 2 or el.shape[1] != 4 or v.shape != (el.shape[0],):
        raise ValueError(f"need (K,) variances for (K, 4) elements, got {v.shape} and {el.shape}")
    if not (np.all(np.isfinite(v)) and np.all(v >= 0.0)):
        raise ValueError("element variances must be finite and non-negative")
    targets.spin_free_tensor(np.arange(1.0, el.shape[0] + 1.0), el, int(norb))   # validates distinct orbits
    return float(s0_hat) + _tau(v, el, int(norb), inverse_hessian)


def _tau(var_e: np.ndarray, elements: np.ndarray, norb: int, inverse_hessian: Callable) -> float:
    """sum_e V_e <u_e, H u_e>, one unit fill at a time (the orbit entries of :func:`unit_fills`)."""
    u = np.zeros((norb,) * 4)
    tau = 0.0
    for (p, q, r, s), ve in zip(elements, var_e):
        if not ve > 0.0:
            continue
        idx = ((p, r, q, s), (q, s, p, r), (r, p, s, q), (s, q, r, p))      # x, P x, T x, P T x (axis-wise)
        u[idx[0], idx[1], idx[2], idx[3]] = (1.0, 1.0, -1.0, -1.0)
        tau += ve * real_inner(u, inverse_hessian(u))
        u[idx[0], idx[1], idx[2], idx[3]] = 0.0
    return float(tau)


# ------------------------------------------------------------------------- the driver
def _allowed_spin_free(norb: int, irreps: np.ndarray | None) -> np.ndarray:
    el = targets.element_set(norb, "spin_free")
    return el if irreps is None else el[element_mask(irreps, el, "spin_free")]


def _call_residual_estimator(estimator: Callable, rd: ResidualData, ci: np.ndarray, k: int,
                             n_elements: int) -> tuple[np.ndarray, np.ndarray | None]:
    out = estimator(copy.deepcopy(rd), ci.copy(), k, 0.0)
    var_e = None
    if isinstance(out, tuple):
        if len(out) != 2:
            raise ValueError("residual_estimator must return A_est or (A_est, var_e)")
        out, var_e = out
    a = np.asarray(out)
    if a.shape != rd.A.shape:
        raise ValueError(f"residual_estimator must return a pair-order tensor of shape {rd.A.shape}, got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("residual_estimator returned non-finite values")
    pair, herm = structure_error(a)
    tol = _STRUCTURE_RTOL * float(np.abs(a).max()) if a.size else 0.0
    if pair > tol or herm > tol:
        raise ValueError("residual_estimator must return a tensor with the structure of A "
                         f"(defects {pair:.2e}, {herm:.2e}); see optimizers.symmetrize_residual")
    if var_e is not None:
        var_e = np.asarray(var_e, dtype=float).ravel()
        if var_e.shape != (n_elements,):
            raise ValueError(f"residual_estimator's variances must have shape ({n_elements},) (the allowed "
                             f"spin-free elements), got {var_e.shape}")
        if not (np.all(np.isfinite(var_e)) and np.all(var_e >= 0.0)):
            raise ValueError("residual_estimator's variances must be finite and non-negative")
    return a, var_e


def _call_energy_estimator(estimator: Callable | None, e_exact: float, ci: np.ndarray, sigma: float) -> float:
    if estimator is None:
        return float(e_exact)
    val = float(estimator(float(e_exact), ci.copy(), float(sigma)))
    if not math.isfinite(val):
        raise ValueError(f"energy_estimator returned a non-finite energy ({val})")
    return val


class _Line:
    """States Psi_k(eps) along d_k, prepared by the propagator, with their exact energies (cached by eps)."""

    def __init__(self, ham: Hamiltonian, psi: np.ndarray, d: np.ndarray, k: int, propagator: Callable | None):
        self.ham, self.psi, self.d, self.k, self.propagator = ham, psi, d, k, propagator
        self._states: dict[float, np.ndarray] = {}
        self._energies: dict[float, float] = {}

    def state(self, eps: float) -> np.ndarray:
        eps = float(eps)
        if eps == 0.0:
            return self.psi
        if eps not in self._states:
            if self.propagator is None:
                st = propagate(self.ham, self.d, self.psi, eps)
            else:
                st = np.asarray(self.propagator(self.d, eps, self.psi.copy(), self.k))
                if st.size != self.psi.size or not np.all(np.isfinite(st)):
                    raise ValueError(f"propagator must return a finite CI array of {self.psi.size} entries")
                st = normalize(st.reshape(self.psi.shape))
            self._states[eps] = st
        return self._states[eps]

    def set_state(self, eps: float, state: np.ndarray) -> None:
        self._states[float(eps)] = state

    def prime_energy(self, eps: float, energy: float) -> None:
        self._energies[float(eps)] = float(energy)

    def energy(self, eps: float) -> float:
        eps = float(eps)
        if eps not in self._energies:
            self._energies[eps] = float(self.ham.energy(self.state(eps)))
        return self._energies[eps]


def run_step_rule(active: ActiveSpace, rule: StepRule = StepRule(), *, ci0: np.ndarray | None = None,
                  max_iter: int = 30, optimizer: str = "lbfgs", metric: str = "spin_orbital",
                  optimizer_options: Mapping | None = None, irreps: np.ndarray | None = None,
                  residual_estimator: Callable | None = None,
                  energy_estimator: Callable | None = None,
                  use_energies: Callable | None = None,
                  propagator: Callable | None = None,
                  keep_states: bool = False, callback: Callable | None = None) -> StepRuleResult:
    """The measured CQE loop with the step rule ``rule`` (module docstring).

    Args:
        active: the active space (a symmetry-adapted build for ``irreps``).
        rule: the :class:`StepRule`.
        ci0: start state, a real CI array or its ravel (default the HF determinant); with
            ``first="second_order"`` it must be a determinant (A_0 is classical only at a determinant; ValueError
            otherwise).
        max_iter: at most this many steps.
        optimizer, metric, optimizer_options: the search directions (``optimizers.make_optimizer``);
            the model rules need "lbfgs".
        irreps: orbital irrep ids; the exact residual is projected on the allowed entries (F2a) and the
            measured set is the allowed spin-free elements.
        residual_estimator: (rd, ci, k, eps) -> A_est or (A_est, var_e), called at Psi_k (eps = 0) for
            every measured iteration (k >= 1; also k = 0 with ``first="line_search"``) with copies of the
            exact ResidualData and of the state. A_est must have the structure of A (as for run_cqe);
            var_e (K,) on the allowed spin-free elements turns on the debiased slope (F33).
        energy_estimator: (e_exact, ci, sigma) -> E_est for every energy the rule measures (parab3,
            hybrid, candidates), with the requested precision sigma; default: the exact energy.
        use_energies: the F31 gate of "hybrid", (k, dE_m, rd, history) -> bool with history =
            (directions d_0 .. d_k, eps_0 .. eps_{k-1}): the current direction last, the steps taken so
            far; required by "hybrid".
        propagator: (d, eps, ci, k) -> CI array of U(d, eps) applied to ci, for every state the loop
            prepares (normalized here); default exp(eps D) (``ft.trajectory.propagate``; the exact line
            search keeps its own Krylov state).
        keep_states: keep Psi_0 .. Psi_K.
        callback: callback(step, ci_after) after every step.
    """
    if not isinstance(rule, StepRule):
        raise ValueError(f"rule must be a StepRule, got {rule!r}")
    if int(max_iter) != max_iter or max_iter < 0:
        raise ValueError(f"max_iter must be a non-negative integer, got {max_iter!r}")
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {optimizer!r}")
    if rule.model_rule and optimizer != "lbfgs":
        raise ValueError(f"rule {rule.name!r} takes the L-BFGS model step (F26) and needs optimizer='lbfgs'")
    if rule.name == "hybrid" and use_energies is None:
        raise ValueError("the hybrid rule needs the use_energies gate (F31; ft.steprules.planned_energy_gate)")
    opt = make_optimizer(optimizer, optimizer_options, metric=metric)
    norb = active.norb
    ir = None
    if irreps is not None:
        ir = np.asarray(irreps)
        if ir.shape != (norb,) or ir.dtype.kind not in "iu":
            raise ValueError(f"irreps must be ({norb},) integers, got {ir.shape} {ir.dtype}")
        ir = ir.astype(np.int64)
    ham = Hamiltonian(active)
    space = ham.space
    if ci0 is not None and np.iscomplexobj(ci0):
        raise ValueError("ci0 must be a real CI vector (real states and directions, as in F3)")
    psi = normalize(space.hf() if ci0 is None else np.asarray(ci0, dtype=float).reshape(space.shape))
    if rule.first == "second_order" and ci0 is not None and np.count_nonzero(np.abs(psi) > _DETERMINANT_TOL) != 1:
        raise ValueError("the classical first step (F16) needs a determinant start (A_0 is classical only at a "
                         "determinant); use first='line_search' to measure A_0 at a correlated ci0")
    elements = _allowed_spin_free(norb, ir)
    states = [psi] if keep_states else None
    steps: list[CQEStep] = []
    events: list[StepEvent] = []
    directions: list[np.ndarray] = []
    eps_taken: list[float] = []
    flags_all: list[tuple[str, ...]] = []
    de_models: list[float] = []
    energies: list[float] = []
    converged = False
    prev_step_norm: float | None = None
    zero_dir: np.ndarray | None = None
    rd = compute_residual(active, psi, ham)
    energies.append(rd.energy)
    for k in range(int(max_iter)):
        flags: list[str] = []
        a_mask = rd.A if ir is None else mask_pair_tensor(rd.A, ir)
        classical = k == 0 and rule.first == "second_order"
        var_e = None
        if classical or residual_estimator is None:
            a_est = a_mask
        else:
            a_est, var_e = _call_residual_estimator(residual_estimator, rd, psi, k, elements.shape[0])
        if not classical:
            events.append(StepEvent(k, "residual", 0.0, None, True))
        g = descent_generator(a_est)
        d = opt.direction(g, a_est)
        info = opt.info
        if info and info.get("reset"):
            flags.append("lbfgs_reset")
        s0_hat = direction_slope(d, a_est)
        slope = direction_slope(d, rd.A)
        s0_prime = s0_hat
        if var_e is not None:
            hinv = opt.inverse_hessian_apply if opt.name == "lbfgs" else \
                (lambda x: apply_metric(x, metric)) if opt.name == "sd" else None
            if hinv is not None:
                s0_prime = s0_hat + _tau(var_e, elements, norb, hinv)
                if opt.name == "lbfgs" and opt.n_pairs > 0:
                    flags.append("slope_debias_approx")
        de_model = 0.5 * s0_prime
        n_energy = 0
        e_cur_est = e_after_est = None
        line = _Line(ham, psi, d, k, propagator)
        if classical:
            # F16 at the start state: eps_2 = -E'(0)/E''(0) along the run's own d_0 (bit-equal to f16's d)
            _, d16, slope16, d2e = f16(active, psi, ham, metric, ir)
            if not np.array_equal(d16, d):
                raise RuntimeError("the optimizer's first direction differs from the F16 direction")
            if not d2e > 0.0:
                raise ValueError(f"the second-order first step is undefined (E''(0) = {d2e:.3e} Ha <= 0)")
            lowering = slope16 ** 2 / (2.0 * d2e)
            if lowering < _MIN_LOWERING:
                converged = True
                break
            eps = -slope16 / d2e
            de_model = -lowering
            flags.append("first:second_order")
        else:
            if 0.5 * abs(s0_hat) < _MIN_LOWERING:
                converged = True
                break
            if k == 0:
                flags.append("first:line_search")
            if rule.name == "line_search" or k == 0:
                eps_max = _LS_BRACKET / max(generator_norm_estimate(space, d, psi), 1e-12)
                eps, _, psi_ls = line_search_step(ham, d, psi, eps_max, _LS_TOL)
                if propagator is None:
                    line.set_state(eps, psi_ls)
                grid = rule.candidates if rule.candidates else tuple(float(c) for c in _M3.eps_candidates)
                for c in grid:                      # M3's costing of the exact line search (F13, F14)
                    events.append(StepEvent(k, "energy", c, float(_M3.eps_ls), True))
                n_energy = len(grid)
            elif rule.name == "candidates":
                est = []
                for c in rule.candidates:
                    e_c = _call_energy_estimator(energy_estimator, line.energy(c), line.state(c), _M3.eps_ls)
                    est.append(e_c)
                    events.append(StepEvent(k, "energy", float(c), float(_M3.eps_ls), True))
                i = int(np.argmin(est))
                eps, e_after_est = float(rule.candidates[i]), est[i]
                if 0.0 in rule.candidates:
                    e_cur_est = est[rule.candidates.index(0.0)]
                n_energy = len(rule.candidates)
            else:
                caps = _caps(rule, d, psi, space, prev_step_norm)
                use_parab = rule.name == "parab3"
                if rule.name == "hybrid":
                    history = (tuple(_readonly_view(x) for x in directions + [d]), tuple(eps_taken))
                    use_parab = bool(use_energies(k, de_model, copy.deepcopy(rd), history))
                    flags.append("hybrid:parab3" if use_parab else "hybrid:unit")
                if use_parab:
                    far, binding = _capped(2.0 * rule.parab_t, caps)         # the stencil's far point obeys F29
                    flags += binding
                    t = 0.5 * far
                    sigma = rule.sigma_parab * abs(de_model)
                    line.prime_energy(0.0, rd.energy)
                    ests = []
                    for x in (0.0, t, 2.0 * t):
                        ests.append(_call_energy_estimator(energy_estimator, line.energy(x), line.state(x), sigma))
                        events.append(StepEvent(k, "energy", float(x), float(sigma), True))
                    eps, status = parabola_step(ests[0], ests[1], ests[2], t, interpolate_only=True)
                    if status != "fit":
                        flags.append(f"parab:{status}")
                    e_cur_est = ests[0]
                    if eps == t or eps == 2.0 * t:
                        e_after_est = ests[1] if eps == t else ests[2]
                    n_energy = 3
                else:
                    eps, binding = _capped(1.0, caps)
                    flags += binding
        eps = float(eps)
        if eps == 0.0:
            flags.append("zero_step")
        psi_new = line.state(eps)
        opt.step_taken(eps)
        rd_new = compute_residual(active, psi_new, ham)
        step = CQEStep(k, rd.energy, rd.variance, rd.residual_norm, slope, -float(np.sum(np.abs(rd.A) ** 2)),
                       eps, rd_new.energy, d, None, residual_est=None, residual_norm_est=spin_free_norm(a_est),
                       slope_est=s0_hat, energy_est=e_cur_est, energy_after_est=e_after_est,
                       n_energy_evals=int(n_energy), optimizer_info=info)
        steps.append(step)
        directions.append(d)
        eps_taken.append(eps)
        flags_all.append(tuple(dict.fromkeys(flags)))
        de_models.append(float(de_model))
        step_norm = abs(eps) * float(np.linalg.norm(d))
        if step_norm > 0.0:
            prev_step_norm = step_norm
        if callback is not None:
            callback(step, psi_new)
        stalled = False
        if eps == 0.0:
            # a zero step leaves the state; with no estimator the next iteration would repeat it
            stalled = residual_estimator is None and energy_estimator is None and zero_dir is not None \
                and np.array_equal(zero_dir, d)
            zero_dir = d.copy()
        else:
            zero_dir = None
        psi, rd = psi_new, rd_new
        energies.append(rd.energy)
        if keep_states:
            states.append(psi)
        if stalled:
            break
    return StepRuleResult(steps=steps, events=events, energies=np.array(energies, dtype=float),
                          directions=directions, eps=eps_taken, flags=flags_all, states=states,
                          optimizer=opt.name, metric=opt.metric, rule=rule, converged=converged,
                          de_model=de_models)


def _readonly_view(a: np.ndarray) -> np.ndarray:
    v = a.view()
    v.setflags(write=False)
    return v


def _caps(rule: StepRule, d: np.ndarray, psi: np.ndarray, space, prev_step_norm: float | None) -> dict[str, float]:
    """F29 upper bounds on eps at iteration k >= 1: {"growth_cap": ..., "angle_cap": ...} (the active ones).

    The growth cap needs a previous nonzero step (none before the first step taken)."""
    out: dict[str, float] = {}
    if rule.growth is not None and prev_step_norm is not None:
        out["growth_cap"] = rule.growth * prev_step_norm / max(float(np.linalg.norm(d)), 1e-300)
    if rule.angle_cap is not None:
        out["angle_cap"] = rule.angle_cap / max(generator_norm_estimate(space, d, psi), 1e-300)
    return out


def _capped(eps0: float, caps: Mapping[str, float]) -> tuple[float, list[str]]:
    """min(eps0, caps) and the flags of the caps that set it (those equal to the minimum, if below eps0)."""
    low = min(caps.values(), default=math.inf)
    if not low < eps0:
        return eps0, []
    return low, [name for name, cap in caps.items() if cap == low]
