"""The closed-loop finite-shot fault-tolerant CQE (F32-F38).

Tags ([derived], [cited], [design], [assumption], [build check]) and formula labels are those of
docs/m3-cost-model.md (F1-F19, F2a) and of its M4 continuation (F20-F53). Energies are in Hartree,
variances in Ha^2, costs in Toffolis; ||x||_m^2 = sum_e m_e x_e^2 over the K symmetry-allowed
spin-free elements e with multiplicities m_e (F2a, F10), the Frobenius norm of the full tensor.

F32 (the closed-loop iteration) [definition]
--------------------------------------------
Psi~_0 = HF. At k = 0 the residual A_0 = W is classical and the step is the second-order step eps_2
of F16 (C_iter(0) = 0). For k >= 1: (1) the controller plans the variance budget B_k from the
measured history (:class:`PrecisionController`, F35); (2) the estimator draws A^_k from the route's
law at B_k and returns the per-element variances V^_k known from its allocation
(:class:`FTResidualEstimator`, F33, F34); (3) the optimizer forms d_k from A^_0 .. A^_k (L-BFGS,
memory 10, curvature tolerance 1e-2; spin-orbital metric by default) and the step rule returns eps_k
(``steprules.run_step_rule``, F26-F31); (4) Psi~_{k+1} = U(d_k, eps_k) Psi~_k with U the exact
exponential or the compiled circuit (F37). The exact E_k, sigma_k^2, p~_k = |<psi_0|Psi~_k>|^2,
||A_k||_m, <S^2> and the compile error are recorded (:class:`ClosedLoopRecord`) and never used by a
decision. Not realizable, and flagged as such by the configuration: the oracle controller, the
``line_search`` rule (the exact line search along the measured direction), the F29 angle cap (a
CI-space norm), the F25 slice rule of compiled propagation (a comparison with the exact state; fixed
slices are realizable) and the oracle hand-off columns.

Rules (:data:`RULES`): "unit" (R1), "line_search", "candidates" (M3's grid of 11 steps with
Gaussian energies at ``LoopSpec.eps_ls``) and "hybrid" (R3, gated by :class:`LoopEnergyGate`) run
through ``steprules.run_step_rule`` with the estimators as hooks; a ``steprules.StepRule`` may be
passed instead of a name (e.g. with the angle cap off). "m3_line_search" is M3's trajectory model:
``cqe.run_cqe`` with :meth:`FTResidualEstimator.for_run_cqe` and its exact line search, called
exactly as ``ft.trajectory.exact_trajectory`` calls it, and costed by ``pipeline.iteration_costs``.

Noise-free limit [derived; tested]: with ``LoopSpec(noise=False)`` the estimator returns the masked
exact residual itself (``symmetry.mask_pair_tensor(rd.A, irreps)``, not a refill, which could flip
signed zeros) and zero variances; with the rule "m3_line_search" and the oracle controller at
r = params.r the states, energies and steps are those of ``exact_trajectory`` (first_step
"second_order", and "exact" with ``first_step="exact"``) bit for bit, and :func:`closed_loop_costs`
is ``pipeline.iteration_costs`` on that record, i.e. M3's C_iter(k) bit for bit
(tests/test_ft_closedloop.py, against tests/data/ft_h4_golden.json).

F33 (direct-route estimator) [derived; M2, F10]
-----------------------------------------------
A^_e = A_e + eta_e, eta_e ~ N(0, v_e/S_k) independent, S_k = V_m/B_k^2 shots, V_m = sum_e m_e v_e,
with the per-shot matchgate-shadow variances v_e of ``ft.residual_inputs.element_variances``
(M3's variance model; element correlations, which raise noise floors by up to 1.32x (M2), are not
modelled). The estimate fills the pair-order tensor by ``targets.spin_free_tensor`` on the allowed
set, so forbidden entries are exactly zero (F2a); V^_e = v_e/S_k. The realized relative error
r~_k^2 = ||eta||_m^2/||A_k||_m^2 has mean r_k^2 = (B_k/||A_k||_m)^2 and relative standard deviation
(2/K_eff)^(1/2), K_eff = V_m^2/sum_e m_e^2 v_e^2 [derived: Var sum_e m_e eta_e^2 = 2 sum_e m_e^2
(v_e/S_k)^2 for Gaussian eta].

F34 (AE estimator: the sine-window law) [derived; cited: Berry and Wiseman, PRL 85, 5098 (2000)]
-------------------------------------------------------------------------------------------------
Route ae_x (F11): element e has the LCU normalization alpha_e = alpha_X,e and the route mean
O_e = A_e. The allocation at B_k (``residual.ae_allocation``) gives eps_e = t (alpha_e/m_e)^(1/3),
sum_e m_e eps_e^2 = B_k^2, and N_e = 2 pi kappa alpha_e/eps_e applications of the Hadamard-test
circuit, i.e. D_e = N_e/2 phase points of G (eigenphases +-2 theta_e, 2 theta_e =
arccos(-O_e/alpha_e)). The outcome is phi^_e = 2 theta_e + 2 pi Delta_e/D_e and O^_e =
-alpha_e cos phi^_e, with the error Delta_e (grid units) on the lattice {n - u_e : n in Z, |n| <= 60}
drawn from the continuum sine-window law (:func:`sine_window_law`)

    P(Delta) = 8 cos^2(pi Delta)/(pi^2 (1 - 4 Delta^2)^2) = 2 sinc^2(1/2 - |Delta|)/(1 + 2|Delta|)^2,

renormalized on the truncated lattice (:func:`sine_window_errors`); the second form, with
sinc(x) = sin(pi x)/(pi x), has no 0/0 at |Delta| = 1/2 (P = 1/2 there). The offset is u_e =
frac(D_e 2 theta_e/(2 pi)) ("fixed": the true phase's offset) or u_e ~ U(0, 1) ("random": a known
random phase added to the register). By Poisson summation, since the autocorrelation of
w(xi) = sqrt(2) sin(pi xi) and its first two derivatives vanish at +-1, the lattice mass, mean and
variance are 1, 0 and 1/4 for every u [derived]: RMS phi^ = pi/D_e and F11's kappa = 1 exactly in
the continuum; O^_e is unbiased to first order with Var(O^_e) = (eps_e sin 2 theta_e/kappa)^2 <=
eps_e^2 at kappa = 1. The estimator reports V^_e = (eps_e/kappa)^2 (1 - (O^_e/alpha_e)^2), from the
allocation and the outcome only. Finite registers: kappa(M) = 0.9736, 0.9934, 0.9983 at M = 63, 255,
1023 applications of G (M3's Appendix A). The law is heavy-tailed: P(|Delta| > 5) <= 2.9e-4 for every
u (supremum 2.84e-4; zero at u = 1/2, where the outcome is +-1/2), and the |n| <= 60 truncation lowers
the variance by at most 1.7e-3 [build check: lattice mass 1 and mean 0 to 1e-12, variance 0.25 to
1e-5 at six offsets (sums to +-2e4); the tail and truncation bounds on an offset grid; kappa(M) within
5e-4 of the three values; tests/test_ft_closedloop.py].

F35 (precision controllers) [derived; design]
---------------------------------------------
"oracle": B_k = r ||A_k||_m (M3's convention; not realizable). "lagged": B_1 = r rho_1 ||A_0||_m
(A_0 classical) and B_k = r rho n^_{k-1} for k >= 2, rho_1 = 0.5, rho = 1, with the debiased norm

    n^_j^2 = max(||A^_j||_m^2 - sum_e m_e V^_j,e, f_fl ||A^_j||_m^2),   f_fl = 0.1,

E[||A^||_m^2 - sum_e m_e V^_e] = ||A||_m^2 [derived; the floor, which keeps the next budget positive,
biases n^ upward whenever it binds]. The realized r_k = B_k/||A_k||_m. Norm precision:
sd(n^^2)/||A||_m^2 = (4 r^2/K_1 + 2 r^4/K_2)^(1/2), K_1 = ||A||_m^2 V_m/sum_e m_e^2 A_e^2 v_e,
K_2 = V_m^2/sum_e m_e^2 v_e^2 [derived].

F36 (cost of a realized iteration) [derived]
--------------------------------------------
M3's unbiased routes depend on (r, ||A||_m) only through B = r ||A||_m (F10, F11), so the realized
T_res(k) is the measured route's cost at ``params.r = r_k`` with C(Psi~_k) and Q(Psi~_k) of the
realized steps: direct_mg T = S_k C(Psi~_k); ae_x T = sum_e N_e (C(Psi~_k) + C_obs,e + Q_A,e/2) with
C_obs,e = 2 L_X,e and Q_A,e = Q(Psi~_k) + ceil(log2 L_X,e) + chi + 5 (F11), both from the allocation
at B_k (:func:`closed_loop_costs` evaluates them from the events, so a record without states is
costed; tested equal to ``residual.compare_routes`` at r_k on the realized state). Route "best" is
the cheaper of direct_mg and ae_x at the planned budget, decided before the measurement; the aux
routes are not used in the loop. C(Psi~_k) = sum_{j<k} s_j C_step(d_j, eps_j/s_j) (F3, F25; s_j = 1
for exact propagation), compiled incrementally as the loop runs. Energies (rules with energies) are
F28/F30: S_E = V_E/sigma^2 shots per energy of C(Psi~_k) + C_step(k; eps_i) + C_rot, with V_E of the
DF split at Psi~_{k+1} (M3's convention, F13); M3's grid and the exact line search are charged as
11 energies at ``LoopSpec.eps_ls``. R3's gate (:class:`LoopEnergyGate`) plans the energies before
the step, where no exact state is available, with the constant V_E(HF) of the same split, so against
the charge its effective threshold is theta V_E(Psi~_{k+1})/V_E(HF); the record keeps the gate's log,
and :func:`handoff_inputs` adds the charged T_E to it. The result is a list of
``pipeline.IterationCost``, so ``pipeline.loop_totals`` and ``pipeline.machines_for_campaigns`` apply
unchanged: ``residual`` holds direct_mg and ae_x at B_k, ``best`` names the measured route,
``best_practical`` and the practical Toffolis repeat it at ``params.kappa_practical`` (AE runs only),
and the oracle controller is costed at ``params`` with r = ``LoopSpec.r`` (r_k = r exactly).

F37 (compiled propagation) [derived]
------------------------------------
With ``propagation="compiled"`` Psi~_{k+1} is the F3 elementwise circuit of (d_k, eps_k) applied to
Psi~_k on the Fock-space oracle (N <= 8): s_k successive ``circuits.compiled_step`` calls at
eps_k/s_k (F25), with the slice rule of F25 (the smallest s in {1, 2, 4, 8, 16} with
||Psi~^(s) - exp(eps D) Psi~|| <= eta_s = 0.05 from the current state; s = 16 and the flag
"slice_cap" if none meets it; ``LoopSpec.slices`` fixes s instead) [design: an emulation-side
calibration, not realizable at scale; "slice_rule" in :attr:`ClosedLoopRecord.non_realizable`]. The
next residual is measured on Psi~_{k+1}, so the loop corrects the product-formula error, but the
L-BFGS pair s_k = eps_k d_k refers to a state that exp(eps_k D_k) does not reach (flag
"pair_inconsistency" when the compile error exceeds eta_s).
Compiled states break S^2 (F1, F3): p~_k and <S^2> are those of the compiled state (F41).

F38 (replicates and statistics) [design]
----------------------------------------
Replicate j of every configuration draws from ``SeedSequence(base_seed, spawn_key=(crc32(name), j))
.spawn(3)``, in the order residual, AE-offset, energy (:func:`replicate_streams`; base_seed
20261108); the configuration does not enter the key (common random numbers: configurations that
differ only in r draw the same standard normals, scaled by their budgets; the energy stream does not
move the residual stream). :func:`summarize` gives, per hand-off iterate k, the policy value V(k) =
mean T(k) under repetition (M3's rule E, T(k) = C_loop(k) + (C(Psi~_k) + C_QPE)/p~_k in C_QPE units)
with a 95% percentile-bootstrap interval (2000 resamples, seed 12345), the same for the paired
difference T(k) - T(k+1) (:func:`paired_verdict`: "pays" iff the interval lies above 0), medians and
16-84% bands of p~_k, E_k - E_0 and r_k, P(T(k) < T(1)) with its Wilson interval
(:func:`wilson_interval`), the catastrophic share P(exists k: p~_{k+1} < p~_k/2), and the stopping
rules evaluated post hoc with their regret against the per-replicate optimum (fixed K in
{1, 2, 3, 4}; the cost gate: hand off before the first iteration k >= 1 whose planned cost exceeds
theta C_QPE, theta in {0.3, 1, 3}; :func:`stopping_rules`). The mean decides: 1/p is convex, so noise
that spreads p~ raises E[T] above T at the median. The QPE-side schemes (``ft.amplify``) are applied to
the same replicates through their costs per scheme: :func:`scheme_verdict` takes, per configuration and
iterate, the best realizable scheme of the replicate means (``ft.amplify.best_realizable``) and decides with
:func:`paired_verdict` on the paired T_{s*(k)}(k) - T_{s*(k+1)}(k + 1), the verdict of record of the closed
loop.
"""

from __future__ import annotations

import dataclasses
import math
import statistics
import zlib
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from ..cqe import run_cqe
from ..diagnostics.thc_floors import ExactReference
from ..fci_ops import Hamiltonian, inner, normalize
from ..integrals import ActiveSpace
from ..measure import targets
from ..residual import compute_residual
from ..steprules import STEP_RULES, StepRule, StepRuleResult, run_step_rule
from . import residual_inputs as ri
from . import scaling
from .circuits import compiled_step
from .handoff import handoff_table, optimal_handoff
from .physical import SurfaceCodeModel
from .pipeline import IterationCost, _sampling_campaign, iteration_costs, machines_for_campaigns, route_campaigns
from .primitives import FTParameters
from .qpe import QPECost, scan_encodings, select_qpe
from .residual import PrepParams, RouteCost, ae_allocation
from .stateprep import (FLAG_ROUNDING, candidate_step_costs, elementwise_step_cost, prep_cost,
                        readout_rotation_qubits, readout_rotation_toffolis)
from .steprules import energy_costs
from .symmetry import benchmark_system, ci_irrep, determinant_irreps, mask_pair_tensor, orbital_irreps
from .trajectory import TrajectoryRecord, exact_trajectory, f16, propagate

__all__ = [
    "ROUTES",
    "CONTROLLERS",
    "PROPAGATIONS",
    "RULES",
    "OFFSETS",
    "VARIANCE_STATES",
    "BASE_SEED",
    "SLICE_CHOICES",
    "ETA_SLICE",
    "LoopSpec",
    "SystemInputs",
    "system_inputs",
    "sine_window_law",
    "sine_window_errors",
    "debiased_norm_square",
    "PrecisionController",
    "MeasurementEvent",
    "FTResidualEstimator",
    "FTEnergyEstimator",
    "LoopEnergyGate",
    "ClosedLoopRecord",
    "replicate_streams",
    "run_closed_loop",
    "closed_loop_costs",
    "handoff_inputs",
    "replicate",
    "bootstrap_mean",
    "wilson_interval",
    "stopping_rules",
    "summarize",
    "paired_verdict",
    "scheme_verdict",
    "run_closed_loop_system",
]

ROUTES = ("direct_mg", "ae_x", "best")
CONTROLLERS = ("oracle", "lagged")
PROPAGATIONS = ("exact", "compiled")
RULES = ("m3_line_search", "unit", "line_search", "candidates", "hybrid")
OFFSETS = ("fixed", "random")
VARIANCE_STATES = ("iterate1", "hf")
FIRST_STEPS = ("second_order", "exact")
#: Base seed of the replicate streams (F38).
BASE_SEED = 20261108
#: Slice counts tried by the F25 rule, and its state-error threshold eta_s.
SLICE_CHOICES = (1, 2, 4, 8, 16)
ETA_SLICE = 0.05
# The sine-window lattice of F34 is truncated at |n| <= this.
_LATTICE_N_MAX = 60
# Rows of the lattice sampler per block (memory of the (rows, 2 n_max + 1) probability table).
_SAMPLER_BLOCK = 8192
# Bootstrap defaults of F38.
_N_BOOT = 2000
_BOOT_SEED = 12345
_LEVEL = 0.95
# Stopping rules of F38: fixed hand-off iterates and cost-gate thresholds (C_QPE units).
_FIXED_K = (1, 2, 3, 4)
_GATES = (0.3, 1.0, 3.0)
_FLAG_NO_LAYOUT = "no layout within the failure budget"
# Singlet roots of the spectrum (M3's 12) and the stage-cache format of the inputs.
_N_ROOTS = 12
_SPIN_TOLS = (1e-6, 1e-4)


# ------------------------------------------------------------------------- validation helpers
def _positive(name: str, x, *, allow_inf: bool = False) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number, got {x!r}") from None
    if not (v > 0.0 and (math.isfinite(v) or (allow_inf and v == math.inf))):
        raise ValueError(f"{name} must be positive{'' if allow_inf else ' and finite'}, got {x!r}")
    return v


def _count(name: str, x, least: int) -> int:
    if isinstance(x, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {least}, got {x!r}")
    try:
        ok = int(x) == x and int(x) >= least
    except (TypeError, ValueError, OverflowError):
        ok = False
    if not ok:
        raise ValueError(f"{name} must be an integer >= {least}, got {x!r}")
    return int(x)


def _choice(name: str, x, options: tuple) -> str:
    if x not in options:
        raise ValueError(f"{name} must be one of {options}, got {x!r}")
    return x


def _frozen(a, dtype=float) -> np.ndarray:
    out = np.array(a, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


# ------------------------------------------------------------------------- configuration
@dataclass(frozen=True)
class LoopSpec:
    """A closed-loop configuration (F32-F37; module docstring).

    Attributes:
        route: "direct_mg" (F33), "ae_x" (F34) or "best" (the cheaper of the two at the planned budget).
        r: planned relative RMS error of the residual (F10's r), > 0.
        noise: False gives the masked exact residual (the noise-free limit); the budgets are still planned
            and costed.
        controller: "oracle" (B_k = r ||A_k||_m) or "lagged" (F35).
        rho, rho1: the lagged plan B_k = r rho n^_{k-1} (k >= 2), B_1 = r rho1 ||A_0||_m.
        norm_floor: f_fl of the debiased norm, in (0, 1].
        ae_offsets: "fixed" or "random" phase offsets of the AE law (F34).
        propagation: "exact" (exp(eps D), Taylor at 1e-15) or "compiled" (F37).
        slices: compiled propagation only: None for the F25 rule, an integer s >= 1 for fixed slices.
        eps_ls: precision (Hartree) of the energies of "candidates" and of the charged line search (F13).
        kappa: AE constant of the allocation (the emulated sine-window law has kappa = 1).
    """

    route: str = "direct_mg"
    r: float = 1.0
    noise: bool = True
    controller: str = "lagged"
    rho: float = 1.0
    rho1: float = 0.5
    norm_floor: float = 0.1
    ae_offsets: str = "fixed"
    propagation: str = "exact"
    slices: int | None = None
    eps_ls: float = 1.6e-3
    kappa: float = 1.0

    def __post_init__(self) -> None:
        _choice("route", self.route, ROUTES)
        object.__setattr__(self, "r", _positive("r", self.r))
        if not isinstance(self.noise, (bool, np.bool_)):
            raise ValueError(f"noise must be a bool, got {self.noise!r}")
        object.__setattr__(self, "noise", bool(self.noise))
        _choice("controller", self.controller, CONTROLLERS)
        object.__setattr__(self, "rho", _positive("rho", self.rho))
        object.__setattr__(self, "rho1", _positive("rho1", self.rho1))
        f = _positive("norm_floor", self.norm_floor)
        if f > 1.0:
            raise ValueError(f"norm_floor must lie in (0, 1], got {self.norm_floor!r}")
        object.__setattr__(self, "norm_floor", f)
        _choice("ae_offsets", self.ae_offsets, OFFSETS)
        _choice("propagation", self.propagation, PROPAGATIONS)
        if self.slices is not None:
            if self.propagation != "compiled":
                raise ValueError("slices applies to propagation='compiled' only")
            object.__setattr__(self, "slices", _count("slices", self.slices, 1))
        object.__setattr__(self, "eps_ls", _positive("eps_ls", self.eps_ls))
        object.__setattr__(self, "kappa", _positive("kappa", self.kappa))

    @property
    def realizable(self) -> bool:
        """False for the oracle controller and for compiled propagation with the F25 slice rule (the rule and the
        angle cap are flagged separately, :attr:`ClosedLoopRecord.non_realizable`)."""
        return self.controller != "oracle" and not (self.propagation == "compiled" and self.slices is None)


@dataclass(frozen=True)
class SystemInputs:
    """Everything a closed loop of one system needs, computed once (:func:`system_inputs`).

    Attributes:
        name: the system name (``ft.symmetry.benchmark_system``); it keys the replicate streams.
        active, irreps: the symmetry-adapted build and its orbital irreps.
        spectrum: ``ft.scaling.spectrum`` (E0, psi_0, E1, E1_any, the 12 singlet roots).
        qpe: the selected QPE (F9); ``qpe.encoding.eri_model`` is H'.
        elements, m: the allowed spin-free elements (``ft.residual_inputs.allowed_elements``) and their
            multiplicities.
        variances: per-shot variances (``ft.residual_inputs.element_variances`` format).
        normalizations: LCU normalizations (``ft.residual_inputs.element_normalizations`` format).
        variance_state: "iterate1" (Psi_1 of the exact spin-free L-BFGS record, M3's convention) or "hf" (F24 (i)).
        energy_split: the DF split of ``ft.residual_inputs.DF_VARIANTS`` chosen at HF (F13).
        readout_toffolis, readout_qubits: C_rot and Q_rot of the DF readout rotation (F4).
    """

    name: str
    active: ActiveSpace
    irreps: np.ndarray
    spectrum: "scaling.Spectrum"
    qpe: QPECost
    elements: np.ndarray
    m: np.ndarray
    variances: dict
    normalizations: dict
    variance_state: str
    energy_split: str
    readout_toffolis: int
    readout_qubits: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"name must be a non-empty string, got {self.name!r}")
        if not isinstance(self.active, ActiveSpace):
            raise ValueError(f"active must be an ActiveSpace, got {type(self.active).__name__}")
        norb, nelec = self.active.norb, tuple(self.active.nelec)
        ir = np.asarray(self.irreps)
        if ir.shape != (norb,) or ir.dtype.kind not in "iu":
            raise ValueError(f"irreps must be ({norb},) integers, got {ir.shape} {ir.dtype}")
        object.__setattr__(self, "irreps", _frozen(ir, np.int64))
        if not isinstance(self.spectrum, scaling.Spectrum):
            raise ValueError(f"spectrum must be an ft.scaling.Spectrum, got {type(self.spectrum).__name__}")
        if (self.spectrum.norb, self.spectrum.nelec) != (norb, nelec):
            raise ValueError("the spectrum belongs to another active space")
        if self.spectrum.e1 is None:
            raise ValueError("the spectrum needs E1 (n_roots >= 2)")
        if not isinstance(self.qpe, QPECost) or self.qpe.encoding.eri_model is None:
            raise ValueError("qpe must be an ft.qpe.QPECost whose encoding carries its model integrals")
        el = np.asarray(self.elements)
        if el.ndim != 2 or el.shape[1] != 4 or el.shape[0] == 0 or el.dtype.kind not in "iu":
            raise ValueError(f"elements must be a non-empty (K, 4) integer array, got {el.shape}")
        k = el.shape[0]
        object.__setattr__(self, "elements", _frozen(el, np.int64))
        m = np.asarray(self.m, dtype=float)
        if m.shape != (k,) or not np.all(np.isfinite(m)) or np.any(m <= 0.0):
            raise ValueError(f"m must hold {k} positive multiplicities")
        object.__setattr__(self, "m", _frozen(m))
        var = dict(self.variances)
        for key in ("v_direct", "w_aux"):
            v = np.asarray(var.get(key), dtype=float)
            if v.shape != (k,) or not np.all(np.isfinite(v)) or np.any(v < 0.0):
                raise ValueError(f"variances[{key!r}] must hold {k} finite non-negative values")
            var[key] = _frozen(v)
        object.__setattr__(self, "variances", var)
        nrm = dict(self.normalizations)
        for key in ("lam_q", "alpha_x", "n_terms_x"):
            v = np.asarray(nrm.get(key))
            if v.shape != (k,) or not np.all(np.isfinite(v)) or np.any(v < 0):
                raise ValueError(f"normalizations[{key!r}] must hold {k} finite non-negative values")
            nrm[key] = _frozen(v, np.int64 if key == "n_terms_x" else float)
        object.__setattr__(self, "normalizations", nrm)
        _choice("variance_state", self.variance_state, VARIANCE_STATES)
        _choice("energy_split", self.energy_split, tuple(ri.DF_VARIANTS))
        object.__setattr__(self, "readout_toffolis", _count("readout_toffolis", self.readout_toffolis, 0))
        object.__setattr__(self, "readout_qubits", _count("readout_qubits", self.readout_qubits, 1))

    @property
    def n_elements(self) -> int:
        """K, the number of measured elements."""
        return int(self.elements.shape[0])

    @property
    def variance_m(self) -> float:
        """V_m = sum_e m_e v_e (Ha^2), the numerator of the direct route's shots (F10)."""
        return float(np.sum(self.m * self.variances["v_direct"]))

    @property
    def c_qpe(self) -> float:
        """C_QPE, the Toffolis of the selected QPE (F9)."""
        return float(self.qpe.toffolis)


def _cached(cache, name: str, stage: str, compute: Callable, key: Mapping):
    if cache is None:
        return compute()
    if not isinstance(cache, scaling.Cache):
        raise ValueError(f"cache must be an ft.scaling.Cache or None, got {type(cache).__name__}")
    return cache.get(name, stage, compute, key=key)


def _params_key(params: FTParameters) -> dict:
    return dataclasses.asdict(params)


def _default_variance_state(norb: int) -> str:
    """The default variance state of :func:`system_inputs`: M3's "iterate1" for N <= 6, "hf" (F24 (i)) for N >= 7."""
    return "iterate1" if int(norb) <= 6 else "hf"


def system_inputs(name: str, *, variance_state: str | None = None, cache: "scaling.Cache | None" = None,
                  params: FTParameters = FTParameters()) -> SystemInputs:
    """The :class:`SystemInputs` of the system ``name`` (M3's builds and inputs; F24).

    The spectrum (12 singlet roots, ``ft.scaling.spectrum``), the QPE (``ft.qpe.scan_encodings`` on the
    ``ExactReference`` and ``select_qpe``, F9), the allowed spin-free elements, their multiplicities and LCU
    normalizations, and the per-shot variances at ``variance_state``: "iterate1" (default for N <= 6; Psi_1
    of the exact spin-free L-BFGS record with the exact first line search, the state at which
    ``pipeline.run_system`` takes them, so M3's numbers are reproduced) or "hf" (default for N >= 7; the HF
    determinant, F24 (i)). The DF split is the lowest at HF (F13). With ``cache`` (``ft.scaling.Cache``) the
    stages "spectrum", "qpe" and "loop_inputs" are cached. ``params.kind`` must be "spin_free".
    """
    if params.kind != "spin_free":
        raise ValueError(f"the closed loop measures spin-free elements; params.kind = {params.kind!r}")
    active = benchmark_system(name)
    irreps = orbital_irreps(active)
    norb = active.norb
    state = _default_variance_state(norb) if variance_state is None else variance_state
    _choice("variance_state", state, VARIANCE_STATES)
    spec = _cached(cache, name, "spectrum", lambda: scaling.spectrum(active, irreps, n_roots=_N_ROOTS,
                                                                      spin_tols=_SPIN_TOLS),
                   {"n_roots": _N_ROOTS, "spin_tols": list(_SPIN_TOLS)})

    def qpe() -> QPECost:
        encodings = scan_encodings(active, reference=ExactReference.build(active), params=params)
        return select_qpe(encodings, params=params)

    sel = _cached(cache, name, "qpe", qpe, {"params": _params_key(params)})

    def loop_inputs() -> dict:
        elements, _ = ri.allowed_elements(active, "spin_free", irreps)
        hf = normalize(Hamiltonian(active).space.hf())
        if state == "iterate1":
            rec = exact_trajectory(active, optimizer="lbfgs", metric="spin_free", max_iter=1, system=name,
                                   irreps=irreps, n_roots=spec.n_roots, first_step="exact", roots=spec.roots)
            psi = rec.states[1]
        else:
            psi = hf
        return {"elements": elements,
                "normalizations": ri.element_normalizations(active, elements, "spin_free"),
                "variances": ri.element_variances(active, psi, elements, "spin_free", params=params),
                "energy_split": ri.energy_variances(active, hf)["best"]}

    inp = _cached(cache, name, "loop_inputs", loop_inputs, {"variance_state": state, "params": _params_key(params)})
    m = targets.element_multiplicities(inp["elements"], "spin_free", norb).astype(float)
    return SystemInputs(name=name, active=active, irreps=irreps, spectrum=spec, qpe=sel, elements=inp["elements"],
                        m=m, variances=inp["variances"], normalizations=inp["normalizations"], variance_state=state,
                        energy_split=inp["energy_split"],
                        readout_toffolis=readout_rotation_toffolis(norb, params=params),
                        readout_qubits=readout_rotation_qubits(norb, params=params))


# ------------------------------------------------------------------------- F34: the sine-window law
def sine_window_law(delta: np.ndarray) -> np.ndarray:
    """F34: the continuum sine-window outcome law P(Delta) = 2 sinc^2(1/2 - |Delta|)/(1 + 2|Delta|)^2 (grid units).

    Equal to 8 cos^2(pi Delta)/(pi^2 (1 - 4 Delta^2)^2) and finite at |Delta| = 1/2 (module docstring).
    """
    d = np.abs(np.asarray(delta, dtype=float))
    if not np.all(np.isfinite(d)):
        raise ValueError("delta must be finite")
    return 2.0 * np.sinc(0.5 - d) ** 2 / (1.0 + 2.0 * d) ** 2


def sine_window_errors(offsets: np.ndarray, rng: np.random.Generator, n_max: int = _LATTICE_N_MAX) -> np.ndarray:
    """One error Delta = n - u per offset u in [0, 1] from P(n - u) on |n| <= n_max, renormalized (F34).

    Inverse-CDF sampling with one ``rng.random()`` per offset (in offset order), so the draw consumes exactly
    len(offsets) uniforms of ``rng``.
    """
    u = np.asarray(offsets, dtype=float).ravel()
    if not np.all(np.isfinite(u)) or np.any(u < 0.0) or np.any(u > 1.0):
        raise ValueError("offsets must lie in [0, 1]")
    n_max = _count("n_max", n_max, 1)
    lattice = np.arange(-n_max, n_max + 1, dtype=float)
    draws = rng.random(u.size)
    out = np.empty(u.size)
    for start in range(0, u.size, _SAMPLER_BLOCK):
        sl = slice(start, start + _SAMPLER_BLOCK)
        p = sine_window_law(lattice[None, :] - u[sl, None])
        cdf = np.cumsum(p, axis=1)
        cdf /= cdf[:, -1:]
        idx = np.minimum((cdf < draws[sl, None]).sum(axis=1), lattice.size - 1)
        out[sl] = lattice[idx] - u[sl]
    return out


# ------------------------------------------------------------------------- F35: controllers
def debiased_norm_square(a_hat: np.ndarray, var: np.ndarray, m: np.ndarray) -> float:
    """||A^||_m^2 - sum_e m_e V^_e, unbiased for ||A||_m^2 for independent unbiased element estimates (F35)."""
    a = np.asarray(a_hat, dtype=float).ravel()
    v = np.asarray(var, dtype=float).ravel()
    w = np.asarray(m, dtype=float).ravel()
    if not (a.shape == v.shape == w.shape) or not (np.all(np.isfinite(a)) and np.all(np.isfinite(v))):
        raise ValueError("a_hat, var and m must be finite arrays of one shape")
    return float(np.sum(w * a * a) - np.sum(w * v))


class PrecisionController:
    """F35: the planned variance budget B_k of every measured iteration k >= 1 (module docstring).

    ``norm0`` is ||A_0||_m (the classical residual at HF). :meth:`planned_budget` (k, exact_norm) returns
    B_k (``exact_norm`` = ||A_k||_m is used by the oracle only); :meth:`observe` (k, a_hat, var) records
    the debiased n^_k of the estimate of iteration k and returns it.

    Attributes:
        budgets, norm_estimates: {k: B_k} and {k: n^_k} so far.
    """

    def __init__(self, spec: LoopSpec, inputs: SystemInputs, norm0: float) -> None:
        if not isinstance(spec, LoopSpec):
            raise ValueError(f"spec must be a LoopSpec, got {type(spec).__name__}")
        self.spec = spec
        self.m = np.asarray(inputs.m, dtype=float)
        self.norm0 = _positive("norm0", norm0)
        self.budgets: dict[int, float] = {}
        self.norm_estimates: dict[int, float] = {}

    def planned_budget(self, k: int, exact_norm: float) -> float:
        k = _count("k", k, 1)
        s = self.spec
        if s.controller == "oracle":
            b = s.r * float(exact_norm)
        elif k == 1:
            b = s.r * s.rho1 * self.norm0
        else:
            if k - 1 not in self.norm_estimates:
                raise ValueError(f"the lagged plan of iteration {k} needs the estimate of iteration {k - 1}")
            b = s.r * s.rho * self.norm_estimates[k - 1]
        b = float(b)
        if not (b > 0.0 and math.isfinite(b)):
            raise ValueError(f"the planned budget of iteration {k} is {b!r} (needs a positive finite norm)")
        self.budgets[k] = b
        return b

    def observe(self, k: int, a_hat: np.ndarray, var: np.ndarray) -> float:
        k = _count("k", k, 1)
        raw2 = float(np.sum(self.m * np.asarray(a_hat, dtype=float) ** 2))
        n2 = max(debiased_norm_square(a_hat, var, self.m), self.spec.norm_floor * raw2)
        n_hat = math.sqrt(n2)
        self.norm_estimates[k] = n_hat
        return n_hat


# ------------------------------------------------------------------------- events and estimators
@dataclass(frozen=True)
class MeasurementEvent:
    """One estimate of the loop, the input of its costing (F36).

    Attributes:
        k: the iteration (the estimate refers to Psi~_k, or to Psi~_k(eps) along d_k).
        kind: "residual" or "energy".
        route: "direct_mg" or "ae_x" (residuals), "energy_sampling" (energies).
        eps: offset along d_k (0: Psi~_k itself; energies: the candidate step).
        budget: planned B (residual) or sigma (energy), Hartree.
        r_realized: B/||A_k||_m (residuals; nan for energies).
        shots: S_k = V_m/B^2 (direct_mg), S_E = V_E/sigma^2 (energies), 0 for AE.
        applications: sum_e N_e (ae_x), else 0.
        per_element: N_e of the AE runs (read-only), else None.
        norm_estimate: the debiased n^_k (residuals), else None.
    """

    k: int
    kind: str
    route: str
    eps: float
    budget: float
    r_realized: float
    shots: float
    applications: float
    per_element: np.ndarray | None
    norm_estimate: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "k", _count("k", self.k, 0))
        _choice("kind", self.kind, ("residual", "energy"))
        routes = ("direct_mg", "ae_x") if self.kind == "residual" else ("energy_sampling",)
        _choice("route", self.route, routes)
        eps = float(self.eps)
        if not math.isfinite(eps):
            raise ValueError(f"eps must be finite, got {self.eps!r}")
        object.__setattr__(self, "eps", eps)
        object.__setattr__(self, "budget", _positive("budget", self.budget))
        r = float(self.r_realized)
        if self.kind == "residual" and not r > 0.0:
            raise ValueError(f"r_realized must be positive for a residual, got {self.r_realized!r}")
        if self.kind == "energy" and not math.isnan(r):
            raise ValueError("r_realized is nan for an energy")
        object.__setattr__(self, "r_realized", r)
        for name in ("shots", "applications"):
            v = float(getattr(self, name))
            if not (v >= 0.0 and math.isfinite(v)):
                raise ValueError(f"{name} must be finite and non-negative, got {getattr(self, name)!r}")
            object.__setattr__(self, name, v)
        if self.per_element is not None:
            pe = np.asarray(self.per_element, dtype=float)
            if pe.ndim != 1 or not np.all(np.isfinite(pe)) or np.any(pe < 0.0):
                raise ValueError("per_element must be a finite non-negative 1-D array")
            object.__setattr__(self, "per_element", _frozen(pe))
        if self.norm_estimate is not None:
            v = float(self.norm_estimate)
            if not (v >= 0.0 and math.isfinite(v)):
                raise ValueError(f"norm_estimate must be finite and non-negative, got {self.norm_estimate!r}")
            object.__setattr__(self, "norm_estimate", v)


def _ae_x_plan(inputs: SystemInputs, prep: PrepParams, budget: float, kappa: float,
               params: FTParameters) -> dict:
    """The F11 allocation of ae_x at the budget B and its runs: eps_e, N_e, Q_A,e, the run Toffolis (operation
    order of ``residual.ae_x_lcu``, so the numbers equal it at the same budget)."""
    nrm = inputs.normalizations
    alpha, m = np.asarray(nrm["alpha_x"], dtype=float), inputs.m
    n_terms = np.asarray(nrm["n_terms_x"])
    k = alpha.size
    l_x = np.maximum(n_terms, 1)
    q_obs = np.ceil(np.log2(l_x)).astype(np.int64) + params.chi + 4
    c_obs = np.broadcast_to(np.asarray(2.0 * n_terms, dtype=float), (k,))
    q_circuit = prep.logical_qubits + np.broadcast_to(np.asarray(q_obs, dtype=np.int64), (k,)) + 1
    gamma_over_b, eps = ae_allocation(alpha, m, budget)
    n_e = np.divide(2.0 * math.pi * kappa * alpha, eps, out=np.zeros(k), where=eps > 0)
    register = np.where(n_e > 2.0, np.ceil(np.log2(np.maximum(n_e, 2.0) / 2.0)), 0.0).astype(np.int64)
    run = n_e * (prep.toffolis + c_obs + 0.5 * q_circuit)
    return {"eps": eps, "n_e": n_e, "q_circuit": q_circuit, "q_run": q_circuit + register, "run": run,
            "c_obs": c_obs, "gamma": gamma_over_b * budget, "alpha": alpha}


def _ae_x_cost(inputs: SystemInputs, prep: PrepParams, budget: float, kappa: float,
               params: FTParameters) -> RouteCost:
    """RouteCost of ae_x at the budget B (F11, F36): sum_e N_e (C(Psi~_k) + C_obs,e + Q_A,e/2)."""
    plan = _ae_x_plan(inputs, prep, budget, kappa, params)
    n_e, run, q_run, q_circuit = plan["n_e"], plan["run"], plan["q_run"], plan["q_circuit"]
    n_total, t_total = float(n_e.sum()), float(run.sum())
    runs = n_e > 0
    info = {"kappa": float(kappa), "bias_m": 0.0, "c_obs": float(np.mean(plan["c_obs"])), "budget": float(budget),
            "gamma": float(plan["gamma"]), "max_applications": float(n_e.max())}
    return RouteCost("ae_x", n_total, t_total, t_total / n_total if n_total > 0 else math.nan, float(run.max()),
                     float(np.count_nonzero(runs)), int(q_run[runs].max() if runs.any() else q_circuit.max()), info,
                     ("estimate",), run_toffolis=tuple(run[runs].tolist()))


def _direct_cost(inputs: SystemInputs, prep: PrepParams, budget: float) -> RouteCost:
    """RouteCost of direct_mg at the budget B (F10, F36): S = V_m/B^2 shots of C(Psi~_k) Toffolis."""
    v_m = inputs.variance_m
    shots = v_m / budget ** 2
    info = {"variance_m": v_m, "bias_m": 0.0, "budget": float(budget)}
    return RouteCost("direct_mg", shots, shots * prep.toffolis, prep.toffolis, prep.toffolis, shots,
                     prep.logical_qubits, info)


class FTResidualEstimator:
    """F33, F34: the residual of the measured elements at the planned budget; the hook of
    ``steprules.run_step_rule`` (``__call__``) and of ``cqe.run_cqe`` (:meth:`for_run_cqe`).

    ``__call__(rd, ci, it, eps)`` plans B_k with the controller (``it`` >= 1; the oracle reads the exact
    ||A_k||_m of ``rd``), chooses the route (``LoopSpec.route``; "best": the cheaper of direct_mg and ae_x
    at B_k on the current circuit :attr:`prep`), draws the estimate (module docstring), lets the controller
    observe it, and returns (A_est, V^_e) with A_est the pair-order tensor (``targets.spin_free_tensor``
    of the estimate; the masked exact tensor itself without noise) and V^_e on the allowed elements in the
    order of ``ft.residual_inputs.allowed_elements``. At ``it`` = 0 it returns the masked exact A_0 and zero
    variances (A_0 = W is classical) and records nothing.

    Attributes:
        prep: PrepParams of the current state's circuit, kept current by the driver (``run_closed_loop``).
        events: one :class:`MeasurementEvent` per measurement.
        estimates: the element estimates A^_k (K,), one per event.
        costs: the planned RouteCost of the measured route, one per event (realized = planned).
    """

    def __init__(self, inputs: SystemInputs, spec: LoopSpec, controller: PrecisionController,
                 rng_residual: np.random.Generator, rng_offsets: np.random.Generator) -> None:
        if not isinstance(inputs, SystemInputs):
            raise ValueError(f"inputs must be SystemInputs, got {type(inputs).__name__}")
        if not isinstance(spec, LoopSpec) or not isinstance(controller, PrecisionController):
            raise ValueError("spec must be a LoopSpec and controller a PrecisionController")
        self.inputs, self.spec, self.controller = inputs, spec, controller
        self.rng_residual, self.rng_offsets = rng_residual, rng_offsets
        self.params = FTParameters()
        self.prep = PrepParams(0.0, 2 * inputs.active.norb)
        self._idx = tuple(np.asarray(inputs.elements).T)
        self._m = inputs.m
        self._v = np.asarray(inputs.variances["v_direct"], dtype=float)
        self._vm = inputs.variance_m
        self.events: list[MeasurementEvent] = []
        self.estimates: list[np.ndarray] = []
        self.costs: list[RouteCost] = []

    def _route(self, budget: float) -> str:
        if self.spec.route != "best":
            return self.spec.route
        direct = _direct_cost(self.inputs, self.prep, budget).toffolis
        ae = _ae_x_cost(self.inputs, self.prep, budget, self.spec.kappa, self.params).toffolis
        return "direct_mg" if direct <= ae else "ae_x"

    def __call__(self, rd, ci, it, eps: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        it = _count("it", it, 0)
        a_mask = mask_pair_tensor(rd.A, self.inputs.irreps)
        k_el = self.inputs.n_elements
        if it == 0:
            return a_mask, np.zeros(k_el)
        a = np.ascontiguousarray(np.real(a_mask[self._idx]), dtype=float)
        norm = math.sqrt(float(np.sum(self._m * a * a)))
        budget = self.controller.planned_budget(it, norm)
        route = self._route(budget)
        shots = applications = 0.0
        per_element = None
        if route == "direct_mg":
            cost = _direct_cost(self.inputs, self.prep, budget)
            shots = self._vm / budget ** 2
            var = self._v / shots
            if self.spec.noise:
                a_hat = a + np.sqrt(var) * self.rng_residual.standard_normal(k_el)
        else:
            cost = _ae_x_cost(self.inputs, self.prep, budget, self.spec.kappa, self.params)
            plan = _ae_x_plan(self.inputs, self.prep, budget, self.spec.kappa, self.params)
            per_element = plan["n_e"]
            applications = float(per_element.sum())
            if self.spec.noise:
                a_hat, var = self._ae_draw(a, plan)
            else:
                var = np.zeros(k_el)
        if not self.spec.noise:
            a_hat, var, a_est = a, np.zeros(k_el), a_mask
        else:
            a_est = targets.spin_free_tensor(a_hat, self.inputs.elements, self.inputs.active.norb)
        n_hat = self.controller.observe(it, a_hat, var)
        self.events.append(MeasurementEvent(it, "residual", route, float(eps), budget,
                                            budget / norm if norm > 0.0 else math.inf, shots, applications,
                                            per_element, n_hat))
        self.estimates.append(_frozen(a_hat))
        self.costs.append(cost)
        return a_est, np.asarray(var, dtype=float)

    def _ae_draw(self, a: np.ndarray, plan: Mapping) -> tuple[np.ndarray, np.ndarray]:
        """F34: O^_e = -alpha_e cos(2 theta_e + 2 pi Delta_e/D_e) and V^_e = (eps_e/kappa)^2 (1 - (O^_e/alpha_e)^2)."""
        alpha, eps, n_e = plan["alpha"], plan["eps"], plan["n_e"]
        live = alpha > 0.0
        safe = np.where(live, alpha, 1.0)
        two_theta = np.arccos(np.clip(-a / safe, -1.0, 1.0))
        d_e = 0.5 * n_e
        if self.spec.ae_offsets == "random":
            u = self.rng_offsets.random(a.size)
        else:
            y = d_e * two_theta / (2.0 * math.pi)
            u = y - np.floor(y)
        delta = sine_window_errors(u, self.rng_residual)
        phi = two_theta + np.divide(2.0 * math.pi * delta, d_e, out=np.zeros(a.size), where=d_e > 0.0)
        o_hat = np.where(live, -alpha * np.cos(phi), 0.0)
        ratio = np.where(live, o_hat / safe, 0.0)
        var = (eps / self.spec.kappa) ** 2 * (1.0 - ratio ** 2)
        return o_hat, np.where(live, var, 0.0)

    def for_run_cqe(self) -> Callable:
        """The ``run_cqe`` hook (rd, ci, it) -> A_est; ``it`` = 0 returns the masked exact A_0."""
        def hook(rd, ci, it):
            if it == 0:
                return mask_pair_tensor(rd.A, self.inputs.irreps)
            return self(rd, ci, it, 0.0)[0]
        return hook


class FTEnergyEstimator:
    """Energies of the rules with energies: E^ = E + sigma z, z ~ N(0, 1) from the energy stream (F28).

    ``__call__(e_exact, ci, sigma_or_context)`` takes the requested precision sigma (``steprules``) or a
    ``run_cqe`` context string ("current", "candidate"), which draws at :attr:`eps_ls`. :attr:`sigma`, when
    set, overrides the requested precision (M3's grid at ``LoopSpec.eps_ls``).

    Attributes:
        draws: (sigma, exact energy, estimate) per call.
        events: the energies as :class:`MeasurementEvent` (filled by ``run_closed_loop`` from the step rule's
            ledger, which knows k and eps; shots at V_E of Psi~_{k+1}).
    """

    def __init__(self, inputs: SystemInputs, rng: np.random.Generator) -> None:
        if not isinstance(inputs, SystemInputs):
            raise ValueError(f"inputs must be SystemInputs, got {type(inputs).__name__}")
        self.inputs, self.rng = inputs, rng
        self.eps_ls = FTParameters().eps_ls
        self.sigma: float | None = None
        self.draws: list[tuple[float, float, float]] = []
        self.events: list[MeasurementEvent] = []

    def __call__(self, e_exact: float, ci, sigma_or_context) -> float:
        if isinstance(sigma_or_context, str):
            sigma = self.eps_ls
        else:
            sigma = _positive("sigma", sigma_or_context)
        if self.sigma is not None:
            sigma = self.sigma
        e = float(e_exact) + sigma * float(self.rng.standard_normal())
        self.draws.append((float(sigma), float(e_exact), e))
        return e


class LoopEnergyGate:
    """The F31 gate of the hybrid rule in the closed loop: ``use_energies(k, dE_m, rd, history) -> bool``.

    At iteration k >= 1 the three energies of the 3-point fit at sigma_P3(k) = ``sigma_parab`` |dE_m|
    (dE_m = s0'/2 from the debiased slope of F33) are planned by F28 (``ft.steprules.energy_costs``) on the
    realized circuit C(Psi~_k) (``estimator.prep``) with C_step(k; t) and C_step(k; 2t) along d_k (the last
    direction of ``history``) and the planned V_E ``energy_variance``, a constant: before the step a
    realizable loop has no exact state, and :func:`run_closed_loop` passes V_E of the DF split at HF
    (classical). The energies are charged at V_E(Psi~_{k+1}) (F28, M3's F13), so against the charge the
    effective threshold is theta V_E(Psi~_{k+1})/V_E(HF) (the exact emulation of ``ft.steprules`` plans V_E at
    the planned step instead). The residual of the iteration is its planned (= realized) cost at B_k (the
    estimator's last measurement). The fit is used iff T_E^plan <= theta T_res^plan. Every decision is logged
    (:attr:`log`: k, sigma, energy_variance, t_energy, t_residual, use; the record keeps it, and
    :func:`handoff_inputs` adds the charged T_E).
    """

    def __init__(self, inputs: SystemInputs, estimator: FTResidualEstimator, rule: StepRule, *,
                 energy_variance: float, params: FTParameters = FTParameters()) -> None:
        self.inputs, self.estimator, self.rule, self.params = inputs, estimator, rule, params
        v = float(energy_variance)
        if not (v >= 0.0 and math.isfinite(v)):
            raise ValueError(f"energy_variance must be finite and non-negative, got {energy_variance!r}")
        self.energy_variance = v
        self.log: list[dict] = []

    def __call__(self, k: int, de_model: float, rd, history) -> bool:
        directions, _ = history
        est = self.estimator
        if not est.events or est.events[-1].k != k:
            raise ValueError(f"the gate of iteration {k} needs the residual measured at iteration {k}")
        t_res = float(est.costs[-1].toffolis)
        sigma = self.rule.sigma_parab * abs(float(de_model))
        t = self.rule.parab_t
        if sigma > 0.0 and math.isfinite(sigma):
            cands = candidate_step_costs(np.asarray(directions[-1], dtype=float), (0.0, t, 2.0 * t),
                                         irreps=self.inputs.irreps, params=self.params)
            t_energy = energy_costs(3, sigma, est.prep, [c.toffolis for c in cands],
                                    energy_variance=self.energy_variance,
                                    readout_toffolis=self.inputs.readout_toffolis,
                                    readout_qubits=self.inputs.readout_qubits,
                                    step_qubits=[c.logical_qubits for c in cands])["sampling"].toffolis
        else:
            t_energy = math.inf
        use = bool(t_energy <= self.rule.theta * t_res)
        self.log.append({"k": int(k), "sigma": sigma, "energy_variance": self.energy_variance,
                         "t_energy": float(t_energy), "t_residual": t_res, "use": use})
        return use


# ------------------------------------------------------------------------- compiled propagation (F37)
class _CompiledPropagator:
    """The steprules propagator of F37: s compiled_step calls at eps/s from the current state (F25 rule or fixed s).

    ``log[(k, eps)] = (s, error, flags)`` with error = ||Psi~^(s) - exp(eps D) Psi~|| (both normalized)."""

    def __init__(self, active: ActiveSpace, ham: Hamiltonian, irreps: np.ndarray, slices: int | None,
                 params: FTParameters) -> None:
        self.active, self.ham, self.irreps, self.slices, self.params = active, ham, irreps, slices, params
        self.log: dict[tuple[int, float], tuple[int, float, tuple[str, ...]]] = {}

    def __call__(self, d, eps, ci, k):
        eps = float(eps)
        exact = propagate(self.ham, d, ci, eps)
        choices = (self.slices,) if self.slices is not None else SLICE_CHOICES
        for s in choices:
            psi = ci
            for _ in range(s):
                psi = compiled_step(self.active, psi, d, eps / s if s > 1 else eps, irreps=self.irreps,
                                    rounded=True, params=self.params)
            psi = normalize(np.asarray(psi).reshape(exact.shape))
            err = float(np.linalg.norm(psi - exact))
            if self.slices is not None or err <= ETA_SLICE:
                break
        flags = []
        if self.slices is None and err > ETA_SLICE:
            flags.append("slice_cap")
        if err > ETA_SLICE:
            flags.append("pair_inconsistency")
        self.log[(int(k), eps)] = (int(s), err, tuple(flags))
        return psi


# The propagator log entry of a step it did not prepare (a zero step: the state is kept).
_NO_SLICE = (1, 0.0, ())


class _Circuit:
    """C(Psi~_k) and Q(Psi~_k) of the realized steps, compiled incrementally (F3, F25, F36)."""

    def __init__(self, norb: int, irreps: np.ndarray, params: FTParameters) -> None:
        self.irreps, self.params = irreps, params
        self.cumulative = [0]
        self.qubits = [2 * norb]
        self.slices: list[int] = []

    def add(self, d: np.ndarray, eps: float, s: int) -> None:
        sc = elementwise_step_cost(d, eps / s if s > 1 else eps, irreps=self.irreps, params=self.params)
        self.cumulative.append(self.cumulative[-1] + s * int(sc.toffolis))
        self.qubits.append(max(self.qubits[-1], int(sc.logical_qubits)))
        self.slices.append(int(s))

    @property
    def prep(self) -> PrepParams:
        return PrepParams(float(self.cumulative[-1]), int(self.qubits[-1]))


# ------------------------------------------------------------------------- records
@dataclass
class ClosedLoopRecord:
    """One closed-loop replicate (F32; module docstring).

    Attributes:
        system, spec, rule: the system name, the configuration and the rule's name (:data:`RULES`).
        seed: (base_seed, crc32(name), replicate_index), the key of the replicate's streams.
        optimizer, metric: the optimizer and its metric.
        eps, directions, slices: eps_k, d_k and s_k (1 for exact propagation) of the K steps.
        energies, variances, overlaps, residual_norms, s2: exact E_k, sigma_k^2, p~_k, ||A_k|| (the spin-free
            Frobenius norm, ``CQEStep.residual_norm``) and <S^2> of Psi~_0 .. Psi~_K; never used by a decision.
        compile_errors: ||Psi~_k - exp(eps_{k-1} D_{k-1}) Psi~_{k-1}||, k = 0..K (0 at HF and for exact
            propagation).
        budgets, r_realized, norm_estimates: B_k, B_k/||A_k||_m and n^_k of every measured iteration (the
            residual events, k = 1, 2, ...).
        prep_cumulative, prep_qubits: C(Psi~_k) and Q(Psi~_k), k = 0..K (F36).
        events: the measurements (residual events in iteration order, each followed by its energies).
        flags: per step, the step rule's flags and "slice_cap", "pair_inconsistency" (compiled).
        states: Psi~_0 .. Psi~_K with ``keep_states`` (always for "m3_line_search", whose costing needs them).
        estimates: the element estimates A^_k (K_el,) of the residual events.
        step_rule: the ``steprules.StepRule`` that ran (None for "m3_line_search").
        first_step: "second_order" (the classical first step) or "exact" ("m3_line_search" only).
        converged: the run stopped on its convergence rule.
        gate_log: the decisions of R3's gate (:attr:`LoopEnergyGate.log`; "hybrid" only, else None).
    """

    system: str
    spec: LoopSpec
    rule: str
    seed: tuple[int, int, int]
    optimizer: str
    metric: str
    eps: np.ndarray
    directions: list[np.ndarray]
    slices: list[int]
    energies: np.ndarray
    variances: np.ndarray
    overlaps: np.ndarray
    residual_norms: np.ndarray
    s2: np.ndarray
    compile_errors: np.ndarray
    budgets: np.ndarray
    r_realized: np.ndarray
    norm_estimates: np.ndarray
    prep_cumulative: np.ndarray
    prep_qubits: np.ndarray
    events: list[MeasurementEvent]
    flags: list[tuple[str, ...]]
    states: list[np.ndarray] | None
    estimates: list[np.ndarray] | None = None
    step_rule: StepRule | None = None
    first_step: str = "second_order"
    converged: bool = False
    gate_log: list[dict] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.spec, LoopSpec):
            raise ValueError(f"spec must be a LoopSpec, got {type(self.spec).__name__}")
        _choice("rule", self.rule, RULES)
        _choice("first_step", self.first_step, FIRST_STEPS)
        k_steps = len(self.eps)
        if len(self.directions) != k_steps or len(self.slices) != k_steps or len(self.flags) != k_steps:
            raise ValueError("eps, directions, slices and flags must have one entry per step")
        for name in ("energies", "variances", "overlaps", "residual_norms", "s2", "compile_errors",
                     "prep_cumulative", "prep_qubits"):
            if len(getattr(self, name)) != k_steps + 1:
                raise ValueError(f"{name} must have K + 1 = {k_steps + 1} entries")
        n_res = sum(1 for e in self.events if e.kind == "residual")
        if not (len(self.budgets) == len(self.r_realized) == len(self.norm_estimates) == n_res):
            raise ValueError("budgets, r_realized and norm_estimates need one entry per residual event")
        if self.states is not None and len(self.states) != k_steps + 1:
            raise ValueError(f"states must hold K + 1 = {k_steps + 1} CI arrays")

    @property
    def n_steps(self) -> int:
        """K, the number of steps taken."""
        return len(self.eps)

    @property
    def non_realizable(self) -> tuple[str, ...]:
        """The non-realizable parts of this replicate (F32): "oracle_controller" (B_k from the exact ||A_k||_m),
        "exact_line_search" (the rules "line_search" and "m3_line_search": the step from exact energies),
        "angle_cap" (the F29 CI-space cap bound at some step) and "slice_rule" (compiled propagation with the F25
        slice rule, which compares every slicing with the exact state; fixed slices are realizable); empty for a
        realizable replicate."""
        out = []
        if self.spec.controller == "oracle":
            out.append("oracle_controller")
        if self.rule in ("line_search", "m3_line_search"):
            out.append("exact_line_search")
        if any("angle_cap" in f for f in self.flags):
            out.append("angle_cap")
        if self.spec.propagation == "compiled" and self.spec.slices is None:
            out.append("slice_rule")
        return tuple(out)

    def residual_event(self, k: int) -> MeasurementEvent | None:
        """The residual event of iteration k at Psi~_k (None if iteration k measured none)."""
        return next((e for e in self.events if e.kind == "residual" and e.k == k and e.eps == 0.0), None)

    def to_trajectory_record(self, inputs: SystemInputs) -> TrajectoryRecord:
        """The ``ft.trajectory.TrajectoryRecord`` of the realized trajectory (E0, E1, E1_any of the spectrum).

        first_step is the record's ("second_order" for every rule but "m3_line_search" with
        ``first_step="exact"``); without states the record holds no CI vectors (``handoff.handoff_table``
        needs none) and no irrep impurity.
        """
        if not isinstance(inputs, SystemInputs):
            raise ValueError(f"inputs must be SystemInputs, got {type(inputs).__name__}")
        if inputs.name != self.system:
            raise ValueError(f"the record belongs to {self.system!r}, the inputs to {inputs.name!r}")
        spec = inputs.spectrum
        active, ir = inputs.active, np.asarray(inputs.irreps, dtype=int)
        norb, nelec = active.norb, tuple(active.nelec)
        flags = [f for f in spec.flags if f in ("e1_lower_bound", "spin_tol_relaxed")]
        impurity = None
        states = [] if self.states is None else list(self.states)
        if states:
            hf_irrep = ci_irrep(states[0], norb, nelec, ir)[0]
            det_ir = determinant_irreps(norb, nelec, ir)
            impurity = max(float(np.sum(np.abs(s[det_ir != hf_irrep]) ** 2)) for s in states)
        if spec.hf_irrep != spec.ground_irrep:
            flags.append("ground_state_not_in_hf_irrep")
        return TrajectoryRecord(
            system=self.system, optimizer=self.optimizer, norb=norb, nelec=nelec, e0=float(spec.e0),
            e1=float(spec.e1), e1_any=float(spec.e1_any), energies=np.asarray(self.energies, dtype=float),
            variances=np.asarray(self.variances, dtype=float),
            residual_norms=np.asarray(self.residual_norms, dtype=float),
            overlaps=np.asarray(self.overlaps, dtype=float), eps=np.asarray(self.eps, dtype=float),
            directions=list(self.directions), states=states, irreps=ir, metric=self.metric,
            ground_irrep=spec.ground_irrep, irrep_impurity=impurity, converged=bool(self.converged),
            flags=tuple(flags), first_step=self.first_step)


def replicate_streams(name: str, replicate_index: int,
                      base_seed: int = BASE_SEED) -> tuple[np.random.Generator, np.random.Generator,
                                                           np.random.Generator]:
    """F38: the (residual, AE-offset, energy) generators of replicate ``replicate_index`` of system ``name``:
    ``SeedSequence(base_seed, spawn_key=(zlib.crc32(name.encode()), replicate_index)).spawn(3)``."""
    j = _count("replicate_index", replicate_index, 0)
    seq = np.random.SeedSequence(_count("base_seed", base_seed, 0), spawn_key=(zlib.crc32(name.encode()), j))
    res, off, en = (np.random.default_rng(s) for s in seq.spawn(3))
    return res, off, en


def _spin_square(active: ActiveSpace, ci: np.ndarray) -> float:
    from pyscf.fci import spin_op

    return float(spin_op.spin_square(np.asarray(ci), active.norb, active.nelec)[0])


def _overlap(psi0: np.ndarray, ci: np.ndarray) -> float:
    return abs(complex(inner(psi0, ci))) ** 2


def run_closed_loop(inputs: SystemInputs, spec: LoopSpec, rule: str | StepRule = "unit", *,
                    replicate_index: int, base_seed: int = BASE_SEED, max_iter: int = 5, optimizer: str = "lbfgs",
                    metric: str = "spin_orbital", keep_states: bool = False,
                    first_step: str = "second_order") -> ClosedLoopRecord:
    """One closed-loop replicate (F32-F37; module docstring).

    Args:
        inputs: :func:`system_inputs` of the system.
        spec: the :class:`LoopSpec`.
        rule: a name of :data:`RULES` or a ``steprules.StepRule`` (its ``first`` must be "second_order": the
            closed loop takes the classical first step).
        replicate_index, base_seed: the replicate's streams (:func:`replicate_streams`, F38).
        max_iter: at most this many steps (k = 0 is the classical step).
        optimizer, metric: the search directions (``optimizers``); the model rules need L-BFGS.
        keep_states: keep Psi~_0 .. Psi~_K in the record ("m3_line_search" always keeps them).
        first_step: "second_order", or "exact" for "m3_line_search" with the oracle controller (M3's measured
            loop: the exact line search along the exact masked A_0; noise from k = 1).
    """
    if not isinstance(inputs, SystemInputs):
        raise ValueError(f"inputs must be SystemInputs, got {type(inputs).__name__}")
    if not isinstance(spec, LoopSpec):
        raise ValueError(f"spec must be a LoopSpec, got {type(spec).__name__}")
    _choice("first_step", first_step, FIRST_STEPS)
    if isinstance(rule, StepRule):
        step_rule, rule_name = rule, rule.name
        if step_rule.first != "second_order":
            raise ValueError("the closed loop takes the classical first step: StepRule.first must be 'second_order'")
    else:
        rule_name = _choice("rule", rule, RULES)
        step_rule = None if rule_name == "m3_line_search" else StepRule(rule_name)
    if step_rule is not None and step_rule.name not in STEP_RULES:
        raise ValueError(f"unknown step rule {step_rule.name!r}")
    if first_step == "exact" and (rule_name != "m3_line_search" or spec.controller != "oracle"):
        raise ValueError("first_step='exact' needs the rule 'm3_line_search' and the oracle controller")
    if rule_name == "m3_line_search" and spec.propagation != "exact":
        raise ValueError("'m3_line_search' propagates by run_cqe's Krylov expansion (propagation='exact')")
    max_iter = _count("max_iter", max_iter, 1)
    j = _count("replicate_index", replicate_index, 0)
    base_seed = _count("base_seed", base_seed, 0)
    rng_res, rng_off, rng_en = replicate_streams(inputs.name, j, base_seed)
    params = FTParameters()
    active, irreps = inputs.active, inputs.irreps
    ham = Hamiltonian(active)
    hf = normalize(ham.space.hf())
    rd0 = compute_residual(active, hf, ham)
    a0 = np.real(mask_pair_tensor(rd0.A, irreps)[tuple(np.asarray(inputs.elements).T)])
    controller = PrecisionController(spec, inputs, math.sqrt(float(np.sum(inputs.m * a0 * a0))))
    est = FTResidualEstimator(inputs, spec, controller, rng_res, rng_off)
    eest = FTEnergyEstimator(inputs, rng_en)
    eest.eps_ls = spec.eps_ls
    circuit = _Circuit(active.norb, irreps, params)
    seed = (base_seed, zlib.crc32(inputs.name.encode()), j)
    if rule_name == "m3_line_search":
        out = _run_m3(inputs, spec, est, circuit, ham, max_iter, optimizer, metric, first_step)
    else:
        out = _run_rule(inputs, spec, step_rule, est, eest, circuit, ham, max_iter, optimizer, metric, params)
    states, rows, eps, directions, flags, compile_err, energy_events, converged, opt_name, opt_metric, gate_log = out
    energies, variances, rnorms = (np.array(col, dtype=float) for col in zip(*rows))
    psi0 = inputs.spectrum.psi0
    res_events = est.events
    events: list[MeasurementEvent] = []
    for ev in res_events:
        events.append(ev)
        if ev.eps == 0.0:
            events.extend(e for e in energy_events if e.k == ev.k)
    events.extend(e for e in energy_events if not any(r.k == e.k for r in res_events))
    keep = keep_states or rule_name == "m3_line_search"
    return ClosedLoopRecord(
        system=inputs.name, spec=spec, rule=rule_name, seed=seed, optimizer=opt_name, metric=opt_metric,
        eps=np.array(eps, dtype=float), directions=directions, slices=list(circuit.slices), energies=energies,
        variances=variances, overlaps=np.array([_overlap(psi0, s) for s in states]), residual_norms=rnorms,
        s2=np.array([_spin_square(active, s) for s in states]), compile_errors=np.array(compile_err, dtype=float),
        budgets=np.array([e.budget for e in res_events]), r_realized=np.array([e.r_realized for e in res_events]),
        norm_estimates=np.array([e.norm_estimate for e in res_events], dtype=float),
        prep_cumulative=np.array(circuit.cumulative, dtype=np.int64),
        prep_qubits=np.array(circuit.qubits, dtype=np.int64), events=events, flags=flags,
        states=states if keep else None, estimates=list(est.estimates), step_rule=step_rule, first_step=first_step,
        converged=bool(converged), gate_log=gate_log)


def _run_m3(inputs, spec, est, circuit, ham, max_iter, optimizer, metric, first_step):
    """M3's trajectory model: run_cqe with the hook, called as ``exact_trajectory`` calls it
    (``ft.trajectory._run``)."""
    active = inputs.active
    states = [normalize(ham.space.hf())]

    def callback(step, ci):
        states.append(normalize(np.asarray(ci)))
        circuit.add(np.asarray(step.generator), float(step.eps), 1)
        est.prep = circuit.prep

    run_kw = dict(optimizer=optimizer, optimizer_options=None, metric=metric, e_exact=float(inputs.spectrum.e0),
                  residual_estimator=est.for_run_cqe())
    if first_step == "second_order":
        _, _, slope, d2e = f16(active, normalize(ham.space.hf()), ham, metric, np.asarray(inputs.irreps, dtype=int))
        if not d2e > 0.0 or slope ** 2 / (2.0 * d2e) < 1e-10:
            raise ValueError(f"the second-order first step is undefined at HF (E''(0) = {d2e:.3e} Ha)")
        run_kw["first_eps"] = -slope / d2e
    res = run_cqe(active, max_iter=int(max_iter), keep_generators=True, callback=callback, **run_kw)
    k_steps = len(states) - 1
    taken = res.steps[:k_steps]
    rows = [(s.energy, s.variance, s.residual_norm) for s in taken]
    if len(res.steps) > k_steps:
        last = res.steps[k_steps]
        rows.append((last.energy, last.variance, last.residual_norm))
    else:
        rd = compute_residual(active, states[k_steps], ham)
        rows.append((rd.energy, rd.variance, rd.residual_norm))
    flags = [("first:second_order",) if (k == 0 and first_step == "second_order") else () for k in range(k_steps)]
    return (states, rows, [s.eps for s in taken], [np.asarray(s.generator) for s in taken], flags,
            [0.0] * (k_steps + 1), [], res.converged, res.optimizer, res.metric, None)


def _run_rule(inputs, spec, step_rule, est, eest, circuit, ham, max_iter, optimizer, metric, params):
    """The step rules of ``steprules.run_step_rule`` with the closed-loop hooks."""
    active, irreps = inputs.active, inputs.irreps
    prop = _CompiledPropagator(active, ham, irreps, spec.slices, params) if spec.propagation == "compiled" else None
    gate = None
    if step_rule.name == "hybrid":
        v_e = ri.energy_variances(active, normalize(ham.space.hf()))[inputs.energy_split]
        gate = LoopEnergyGate(inputs, est, step_rule, energy_variance=v_e, params=params)
    if step_rule.name == "candidates":
        eest.sigma = spec.eps_ls

    def callback(step, ci):
        k, eps = int(step.iteration), float(step.eps)
        s = 1 if prop is None else prop.log.get((k, eps), _NO_SLICE)[0]      # a zero step prepares nothing
        circuit.add(np.asarray(step.generator), eps, s)
        est.prep = circuit.prep

    res: StepRuleResult = run_step_rule(active, step_rule, max_iter=max_iter, optimizer=optimizer, metric=metric,
                                        irreps=irreps, residual_estimator=est, energy_estimator=eest,
                                        use_energies=gate, propagator=prop, keep_states=True, callback=callback)
    states = list(res.states)
    k_steps = res.n_steps
    rows = [(s.energy, s.variance, s.residual_norm) for s in res.steps]
    last = compute_residual(active, states[-1], ham)
    rows.append((last.energy, last.variance, last.residual_norm))
    flags, compile_err = [], [0.0]
    for k in range(k_steps):
        f = list(res.flags[k])
        err = 0.0
        if prop is not None:
            _, err, extra = prop.log.get((k, float(res.eps[k])), _NO_SLICE)
            f += list(extra)
        compile_err.append(err)
        flags.append(tuple(dict.fromkeys(f)))
    # energies: the ledger's (k, eps, sigma), the charged precision of M3's grid and line search, and S_E at
    # V_E of Psi~_{k+1}
    energy_events: list[MeasurementEvent] = []
    v_cache: dict[int, float] = {}
    drawn: list[MeasurementEvent] = []
    for ev in res.events:
        if ev.kind != "energy":
            continue
        sigma = spec.eps_ls if step_rule.name in ("candidates", "line_search") else float(ev.sigma)
        if ev.iteration not in v_cache:
            nxt = states[min(ev.iteration + 1, len(states) - 1)]
            v_cache[ev.iteration] = float(ri.energy_variances(active, nxt)[inputs.energy_split])
        me = MeasurementEvent(ev.iteration, "energy", "energy_sampling", float(ev.eps), sigma, math.nan,
                              v_cache[ev.iteration] / sigma ** 2, 0.0, None, None)
        energy_events.append(me)
        if step_rule.name != "line_search":
            drawn.append(me)
    if len(drawn) != len(eest.draws):
        raise RuntimeError(f"{len(eest.draws)} energies drawn for {len(drawn)} charged")
    eest.events = drawn
    return (states, rows, list(res.eps), list(res.directions), flags, compile_err, energy_events, res.converged,
            res.optimizer, res.metric, None if gate is None else [dict(entry) for entry in gate.log])


# ------------------------------------------------------------------------- F36: realized costs
def _params_for(spec: LoopSpec, params: FTParameters, r: float | None = None) -> FTParameters:
    """``params`` with r = r (default ``spec.r``) and eps_ls = ``spec.eps_ls``, replaced only where they differ."""
    r = spec.r if r is None else float(r)
    changes = {}
    if r != params.r:
        changes["r"] = r
    if spec.eps_ls != params.eps_ls:
        changes["eps_ls"] = spec.eps_ls
    return dataclasses.replace(params, **changes) if changes else params


def _m3_costs(inputs: SystemInputs, record: ClosedLoopRecord, params: FTParameters,
              model: SurfaceCodeModel) -> list[IterationCost]:
    """``pipeline.iteration_costs`` of the record (M3's F14), at r = spec.r (oracle) or at r_k per iteration."""
    if record.states is None:
        raise ValueError("the m3_line_search costing needs the record's states")
    trec = record.to_trajectory_record(inputs)
    prep = prep_cost(record.directions, record.eps, method="elementwise", irreps=inputs.irreps, params=params)
    classical = record.first_step == "second_order"
    kw = dict(elements=inputs.elements, variances=inputs.variances, normalizations=inputs.normalizations,
              energy_variance=inputs.energy_split, readout_toffolis=inputs.readout_toffolis,
              classical_first_step=classical, model=model)
    last = record.n_steps - 1
    walk = inputs.qpe.encoding
    if record.spec.controller == "oracle":
        return iteration_costs(inputs.active, trec, prep, walk, k_max=last, params=_params_for(record.spec, params),
                               **kw)
    out = []
    for k in range(last + 1):
        ev = record.residual_event(k)
        if ev is None:
            out.append(iteration_costs(inputs.active, trec, prep, walk, k_max=k,
                                       params=_params_for(record.spec, params), **kw)[k])
        else:
            out.append(iteration_costs(inputs.active, trec, prep, walk, k_max=k,
                                       params=_params_for(record.spec, params, ev.r_realized), **kw)[k])
    return out


def closed_loop_costs(inputs: SystemInputs, record: ClosedLoopRecord, *, params: FTParameters = FTParameters(),
                      model: SurfaceCodeModel = SurfaceCodeModel()) -> list[IterationCost]:
    """F36: C_iter(k), k = 0..K-1, of the realized iterations of ``record`` (module docstring).

    "m3_line_search" is costed by ``pipeline.iteration_costs`` on the converted record (M3's F14: the best
    of every route plus the 11-candidate line search), at ``params`` with r = ``LoopSpec.r`` for the oracle
    controller and at r_k otherwise. The other rules: the measured route at its planned budget (from the
    residual events) on the realized circuit, plus the energies of the ledger (F28, F30), with the F19
    campaigns. k = 0 (the classical first step) is free.
    """
    if not isinstance(inputs, SystemInputs) or not isinstance(record, ClosedLoopRecord):
        raise ValueError("need SystemInputs and a ClosedLoopRecord")
    if record.system != inputs.name:
        raise ValueError(f"the record belongs to {record.system!r}, the inputs to {inputs.name!r}")
    if record.rule == "m3_line_search":
        return _m3_costs(inputs, record, params, model)
    spec = record.spec
    norb = inputs.active.norb
    walk = inputs.qpe.encoding.walk_params()
    c_rot, q_rot = inputs.readout_toffolis, inputs.readout_qubits
    out = []
    for k in range(record.n_steps):
        prep_k = PrepParams(float(record.prep_cumulative[k]), int(record.prep_qubits[k]))
        events = [e for e in record.events if e.k == k]
        if k == 0:
            if events:
                raise ValueError("the classical first step measures nothing, but the record charges iteration 0")
            out.append(IterationCost(k=0, residual=(), best="classical", line_search=None, toffolis=0.0, shots=0.0,
                                     seconds_one_machine=0.0, machines_7d=0, best_practical="classical",
                                     prep_toffolis=float(prep_k.toffolis)))
            continue
        residuals = [e for e in events if e.kind == "residual"]
        main = [e for e in residuals if e.eps == 0.0]
        if len(main) != 1:
            raise ValueError(f"iteration {k}: one residual at Psi~_{k} expected, got {len(main)}")
        d = np.asarray(record.directions[k], dtype=float)
        t_res = t_res_p = shots = 0.0
        campaigns: tuple = ()
        routes: tuple[RouteCost, ...] = ()
        measured_route = main[0].route
        for ev in main + [e for e in residuals if e.eps != 0.0]:
            if ev.eps == 0.0:
                prep_e = prep_k
            else:
                st = elementwise_step_cost(d, ev.eps, irreps=inputs.irreps, params=params)
                prep_e = PrepParams(prep_k.toffolis + st.toffolis, max(prep_k.logical_qubits, st.logical_qubits))
            direct = _direct_cost(inputs, prep_e, ev.budget)
            ae = _ae_x_cost(inputs, prep_e, ev.budget, spec.kappa, params)
            measured = direct if ev.route == "direct_mg" else ae
            practical = measured if ev.route == "direct_mg" else _ae_x_cost(inputs, prep_e, ev.budget,
                                                                           params.kappa_practical, params)
            if ev.eps == 0.0:
                routes = (direct, ae)
            t_res += measured.toffolis
            t_res_p += practical.toffolis
            shots += measured.n_prep if ev.route == "direct_mg" else 0.0
            campaigns += tuple(route_campaigns(measured, params, model))
        energies = [e for e in events if e.kind == "energy"]
        ls, ls_ae, flagged = None, 0.0, 0
        if energies:
            sigmas = {e.budget for e in energies}
            if len(sigmas) != 1:
                raise ValueError(f"iteration {k}: the energies of one iteration share one precision")
            sigma = sigmas.pop()
            v_e = energies[0].shots * sigma ** 2
            cands = candidate_step_costs(d, [e.eps for e in energies], irreps=inputs.irreps, params=params)
            ec = energy_costs(len(cands), sigma, prep_k, [c.toffolis for c in cands], energy_variance=v_e,
                              readout_toffolis=c_rot, readout_qubits=q_rot, walk=walk, kappa=params.kappa,
                              step_qubits=[c.logical_qubits for c in cands], norb=norb)
            ls, ls_ae = ec["sampling"], float(ec["ae"].toffolis)
            flagged = sum(FLAG_ROUNDING in c.flags for c in cands)
            per = ls.params["shots_per_energy"]
            campaigns += tuple(_sampling_campaign("energy", per, prep_k.toffolis + c.toffolis + c_rot,
                                                  ls.logical_qubits, params, model) for c in cands)
        t_e = 0.0 if ls is None else ls.toffolis
        flags = (_FLAG_NO_LAYOUT,) if any(c.per_run is None and 0 < c.n_runs < math.inf for c in campaigns) else ()
        out.append(IterationCost(
            k=k, residual=routes, best=measured_route, line_search=ls, toffolis=float(t_res + t_e),
            shots=float(shots + (0.0 if ls is None else ls.n_prep)),
            seconds_one_machine=float(sum(c.seconds(1) for c in campaigns)),
            machines_7d=machines_for_campaigns(campaigns), best_practical=measured_route,
            toffolis_practical=float(t_res_p + t_e), residual_toffolis=float(t_res),
            residual_toffolis_practical=float(t_res_p), line_search_ae=ls_ae,
            residual_norm_m=float(main[0].budget / main[0].r_realized), prep_toffolis=float(prep_k.toffolis),
            ls_flagged=int(flagged), campaigns=campaigns, flags=flags))
    return out


def handoff_inputs(inputs: SystemInputs, record: ClosedLoopRecord, costs: Sequence[IterationCost]) -> dict:
    """The JSON-able hand-off inputs of one replicate (Toffolis; F36, F38).

    Keys: "seed", "k" (0..K), "p", "e_err", "c_prep" (K + 1 each), "c_iter" (K), "c_loop" (K + 1),
    "r_realized" and "budget" (aligned with k; None where iteration k measured no residual), "flags" (per
    step), "compiled", "compile_errors", "s2", "non_realizable" (:attr:`ClosedLoopRecord.non_realizable`),
    "contaminant_below_theta" (None: the hand-off layer fills it),
    "handoff" (``handoff.handoff_table`` of the converted record under the record's loop, same-irrep E1: k,
    p, cost_expected (rule E), certifiable, cost_delta_oracle), "optima" (k of rule E, None if no row is
    certified, and of "expected_oracle") and "gate_log" (R3: :attr:`ClosedLoopRecord.gate_log`, each entry
    with "t_energy_charged", the energies charged in iteration k (0 where the gate chose the unit step), and
    "energy_variance_charged", their V_E(Psi~_{k+1}) (None without energies); None for the other rules).
    """
    if len(costs) != record.n_steps:
        raise ValueError(f"need C_iter(k) for k = 0..{record.n_steps - 1}, got {len(costs)} costs")
    trec = record.to_trajectory_record(inputs)
    c_iter = [float(c.toffolis) for c in costs]
    loop = "classical_first_step" if record.first_step == "second_order" else "measured"
    rows = handoff_table(trec, np.asarray(record.prep_cumulative, dtype=float), c_iter, inputs.qpe, loop=loop,
                         use_e1="same_irrep")
    optima = {}
    for rule in ("expected", "expected_oracle"):
        try:
            optima[rule] = optimal_handoff(rows, rule).k
        except ValueError:
            optima[rule] = None
    k_all = list(range(record.n_steps + 1))
    r_by_k = {e.k: e for e in record.events if e.kind == "residual" and e.eps == 0.0}
    gate_log = None
    if record.gate_log is not None:
        gate_log = []
        for entry in record.gate_log:
            ls = costs[entry["k"]].line_search
            gate_log.append(dict(entry, t_energy_charged=0.0 if ls is None else float(ls.toffolis),
                                 energy_variance_charged=None if ls is None else float(ls.params["energy_variance"])))
    return {"seed": list(record.seed), "k": k_all, "p": [float(x) for x in record.overlaps],
            "e_err": [float(e - inputs.spectrum.e0) for e in record.energies],
            "c_prep": [float(x) for x in record.prep_cumulative], "c_iter": c_iter,
            "c_loop": [float(r.c_loop) for r in rows],
            "r_realized": [r_by_k[k].r_realized if k in r_by_k else None for k in k_all],
            "budget": [r_by_k[k].budget if k in r_by_k else None for k in k_all],
            "flags": [list(f) for f in record.flags], "compiled": record.spec.propagation == "compiled",
            "compile_errors": [float(x) for x in record.compile_errors], "s2": [float(x) for x in record.s2],
            "non_realizable": list(record.non_realizable), "contaminant_below_theta": None,
            "handoff": [{"k": r.k, "p": r.p, "cost_expected": r.cost_expected, "certifiable": r.certifiable,
                         "cost_delta_oracle": r.cost_delta_oracle} for r in rows],
            "optima": optima, "gate_log": gate_log}


def replicate(inputs: SystemInputs, spec: LoopSpec, rule: str | StepRule, *, n: int, max_iter: int = 5,
              base_seed: int = BASE_SEED, keep_states: bool = False,
              metric: str = "spin_orbital") -> list[ClosedLoopRecord]:
    """F38: replicates j = 0..n-1 of one configuration (:func:`run_closed_loop` with replicate_index j and
    ``base_seed``; L-BFGS with ``metric``)."""
    n = _count("n", n, 1)
    return [run_closed_loop(inputs, spec, rule, replicate_index=j, base_seed=base_seed, max_iter=max_iter,
                            metric=metric, keep_states=keep_states) for j in range(n)]


# ------------------------------------------------------------------------- F38: statistics
def _bootstrap_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def _percentile_interval(values: np.ndarray, idx: np.ndarray, level: float) -> tuple[float, float]:
    means = values[idx].mean(axis=1)
    a = 0.5 * (1.0 - level)
    lo, hi = np.quantile(means, [a, 1.0 - a])
    return float(lo), float(hi)


def _check_level(level) -> float:
    lv = float(level)
    if not 0.0 < lv < 1.0:
        raise ValueError(f"level must lie in (0, 1), got {level!r}")
    return lv


def bootstrap_mean(values: Sequence[float], *, n_boot: int = _N_BOOT, seed: int = _BOOT_SEED,
                   level: float = _LEVEL) -> tuple[float, float, float]:
    """(mean, lo, hi): the mean and its percentile-bootstrap interval (F38: ``n_boot`` resamples with
    replacement from ``np.random.default_rng(seed)``, the (1 -+ level)/2 quantiles of the resampled means);
    lo = hi = nan for fewer than two values."""
    x = np.asarray(values, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("need at least one value")
    level = _check_level(level)
    mean = float(x.mean())
    if x.size < 2:
        return mean, math.nan, math.nan
    lo, hi = _percentile_interval(x, _bootstrap_indices(x.size, _count("n_boot", n_boot, 1),
                                                        _count("seed", seed, 0)), level)
    return mean, lo, hi


def wilson_interval(successes: int, n: int, *, level: float = _LEVEL) -> tuple[float, float]:
    """The Wilson score interval of a binomial share x/n: (c -+ h) with c = (x + z^2/2)/(n + z^2),
    h = z sqrt(x (n - x)/n + z^2/4)/(n + z^2), z = Phi^-1((1 + level)/2)."""
    n = _count("n", n, 1)
    x = _count("successes", successes, 0)
    if x > n:
        raise ValueError(f"successes ({x}) exceed n ({n})")
    z = statistics.NormalDist().inv_cdf(0.5 * (1.0 + _check_level(level)))
    c = (x + 0.5 * z * z) / (n + z * z)
    h = z * math.sqrt(x * (n - x) / n + 0.25 * z * z) / (n + z * z)
    return max(0.0, c - h), min(1.0, c + h)


def paired_verdict(t_k: Sequence[float], t_k1: Sequence[float], *, n_boot: int = _N_BOOT, seed: int = _BOOT_SEED,
                   level: float = _LEVEL) -> dict:
    """The closed-loop decision rule of F38: does iteration k pay, T(k) - T(k + 1) > 0 in mean?

    ``t_k``, ``t_k1``: the paired values of T(k) and T(k + 1) over the replicates (finite, one per
    replicate). Returns {"mean", "ci" ([lo, hi], percentile bootstrap of the mean difference), "pays"
    (iff lo > 0, i.e. the interval lies above 0), "median", "share_positive", "n", "n_boot", "seed",
    "level"}; with a single replicate there is no interval and "pays" is False.
    """
    a = np.asarray(t_k, dtype=float).ravel()
    b = np.asarray(t_k1, dtype=float).ravel()
    if a.shape != b.shape or a.size == 0:
        raise ValueError(f"t_k and t_k1 must be paired non-empty sequences, got {a.shape} and {b.shape}")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("t_k and t_k1 must be finite")
    diff = a - b
    mean, lo, hi = bootstrap_mean(diff, n_boot=n_boot, seed=seed, level=level)
    return {"mean": mean, "ci": [lo, hi], "pays": bool(lo > 0.0), "median": float(np.median(diff)),
            "share_positive": float(np.mean(diff > 0.0)), "n": int(diff.size), "n_boot": int(n_boot),
            "seed": int(seed), "level": float(level)}


def scheme_verdict(costs_k: Sequence[Mapping[str, float | None]], costs_k1: Sequence[Mapping[str, float | None]], *,
                   n_boot: int = _N_BOOT, seed: int = _BOOT_SEED, level: float = _LEVEL) -> dict:
    """The closed-loop decision rule of F38 under the QPE-side schemes of F43 (the verdict of record).

    ``costs_k``, ``costs_k1``: per replicate (paired, same order) the scheme costs {scheme: T_s} at Psi~_k and
    Psi~_{k+1} (``ft.amplify.scheme_costs`` with the replicate's C_loop; None where a scheme has no cost; extra keys
    such as "<scheme>@oracle_U" are carried along). Per key with a finite cost in every replicate at both iterates:
    the replicate means at k and k + 1 and :func:`paired_verdict` of T_s(k) - T_s(k + 1). The verdict of record
    takes, per configuration and iterate, the best realizable scheme of the replicate means,
    s*(k) = ``ft.amplify.best_realizable`` (oracle schemes and bounds never), and decides with
    :func:`paired_verdict` of T_{s*(k)}(k) - T_{s*(k+1)}(k + 1) over the replicates: "pays" iff its interval lies
    above 0.

    Returns {"n", "means_k", "means_k1" ({scheme: mean}), "by_scheme" ({scheme: paired_verdict}), "best_k",
    "best_k1" (None without a realizable scheme), "best" (the paired verdict of record, None without s*), "pays"
    (its "pays"; False without s*)}.
    """
    from .amplify import best_realizable

    a, b = list(costs_k), list(costs_k1)
    if not a or len(a) != len(b):
        raise ValueError(f"costs_k and costs_k1 must be paired non-empty sequences, got {len(a)} and {len(b)}")

    def complete(rows: list) -> dict[str, np.ndarray]:
        keys = dict.fromkeys(key for row in rows for key in row)
        out = {}
        for key in keys:
            vals = [row.get(key) for row in rows]
            if all(v is not None for v in vals):
                arr = np.asarray(vals, dtype=float)
                if np.all(np.isfinite(arr)):
                    out[key] = arr
        return out

    ta, tb = complete(a), complete(b)
    means_k = {key: float(v.mean()) for key, v in ta.items()}
    means_k1 = {key: float(v.mean()) for key, v in tb.items()}
    kw = dict(n_boot=n_boot, seed=seed, level=level)
    by_scheme = {key: paired_verdict(ta[key], tb[key], **kw) for key in ta if key in tb}

    def best(means: dict) -> str | None:
        try:
            return best_realizable(means)
        except ValueError:
            return None

    bk, bk1 = best(means_k), best(means_k1)
    verdict = None if bk is None or bk1 is None else paired_verdict(ta[bk], tb[bk1], **kw)
    return {"n": len(a), "means_k": means_k, "means_k1": means_k1, "by_scheme": by_scheme, "best_k": bk,
            "best_k1": bk1, "best": verdict, "pays": bool(verdict is not None and verdict["pays"])}


def stopping_rules(t: np.ndarray, c_iter: np.ndarray, *, fixed: Sequence[int] = _FIXED_K,
                   gates: Sequence[float] = _GATES) -> dict:
    """F38: post-hoc stopping rules against the per-replicate optimum.

    ``t`` (n, K + 1): T(k) per replicate (C_QPE units); ``c_iter`` (n, K): the planned C_iter(k) (C_QPE
    units; k = 0 is free). "fixed_<K>" hands off Psi~_K; "cost_gate_<theta>" starts at k = 1 and hands off
    Psi~_k before the first iteration k >= 1 whose planned cost exceeds theta (the last iterate if none
    does). Each entry: {"mean" (of T at the rule), "regret" (mean of T(rule) - min_k T(k)), "k_hist"
    (hand-off iterates)}; "oracle": the per-replicate optimum's mean and its iterates.
    """
    t = np.asarray(t, dtype=float)
    c = np.asarray(c_iter, dtype=float)
    if t.ndim != 2 or c.ndim != 2 or c.shape != (t.shape[0], t.shape[1] - 1) or t.shape[1] < 2:
        raise ValueError(f"t must be (n, K + 1) and c_iter (n, K) with K >= 1, got {t.shape} and {c.shape}")
    n, kp1 = t.shape
    best = t.min(axis=1)

    def entry(picks: np.ndarray) -> dict:
        vals = t[np.arange(n), picks]
        return {"mean": float(vals.mean()), "regret": float((vals - best).mean()),
                "k_hist": np.bincount(picks, minlength=kp1).tolist()}

    out: dict = {}
    for k in fixed:
        k = _count("fixed K", k, 0)
        if k < kp1:
            out[f"fixed_{k}"] = entry(np.full(n, k))
    for theta in gates:
        theta = _positive("gate theta", theta)
        picks = np.empty(n, dtype=int)
        for i in range(n):
            k = 1
            while k < kp1 - 1 and c[i, k] <= theta:
                k += 1
            picks[i] = k
        out[f"cost_gate_{theta:g}"] = entry(picks)
    arg = t.argmin(axis=1)
    out["oracle"] = {"mean": float(best.mean()), "regret": 0.0, "k_hist": np.bincount(arg, minlength=kp1).tolist()}
    return out


def _band(x: np.ndarray) -> list[float]:
    return [float(v) for v in np.quantile(x, [0.16, 0.84])]


def summarize(inputs: SystemInputs, records: Sequence[ClosedLoopRecord], costs: Sequence[Sequence[IterationCost]], *,
              n_boot: int = _N_BOOT, seed: int = _BOOT_SEED) -> dict:
    """F38 statistics of one configuration under repetition (M3's rule E; module docstring).

    T(k) = [C_loop(k) + (C(Psi~_k) + C_QPE)/p~_k]/C_QPE (``handoff.handoff_table``'s cost_expected) for the
    hand-off iterates k = 0..K common to every replicate. Returns {"n", "k_max", "c_qpe", "per_k": [{k,
    T_mean, T_ci, T_median, T_band, p_mean, p_median, p_band, e_err_median, e_err_band (Hartree), beats_psi1
    (P(T(k) < T(1))) and beats_psi1_ci (Wilson), r_median and r_band (realized r_k, measured k), c_iter_mean
    (C_QPE units, k < K), diff_mean, diff_ci, pays (T(k) - T(k + 1), k < K)}], "optimum" (per-replicate
    min_k T(k): mean, ci, k_hist), "best_fixed_k", "catastrophic_share" and its Wilson interval,
    "stopping" (:func:`stopping_rules` with the realized C_iter(k) as the planned cost), "flag_counts"
    (replicates with each step flag), "realizable" (no replicate has a :attr:`ClosedLoopRecord.non_realizable`
    part), "non_realizable" (their union, sorted), "rule"}. The bootstrap draws one
    resample matrix (``n_boot`` x n, ``np.random.default_rng(seed)``) shared by every statistic.
    """
    records, costs = list(records), list(costs)
    if not records or len(costs) != len(records):
        raise ValueError(f"need one cost list per record, got {len(records)} records and {len(costs)} cost lists")
    n = len(records)
    c_qpe = inputs.c_qpe
    hand = [handoff_inputs(inputs, r, c) for r, c in zip(records, costs)]
    kp1 = min(len(h["handoff"]) for h in hand)
    t = np.array([[row["cost_expected"] / c_qpe for row in h["handoff"][:kp1]] for h in hand])
    p = np.array([h["p"][:kp1] for h in hand])
    err = np.array([h["e_err"][:kp1] for h in hand])
    citer = np.array([[x / c_qpe for x in h["c_iter"][:kp1 - 1]] for h in hand])
    idx = _bootstrap_indices(n, _count("n_boot", n_boot, 1), _count("seed", seed, 0)) if n >= 2 else None

    def interval(x: np.ndarray) -> list[float]:
        if idx is None or not np.all(np.isfinite(x)):
            return [math.nan, math.nan]
        return list(_percentile_interval(x, idx, _LEVEL))

    per_k = []
    for k in range(kp1):
        row = {"k": k, "T_mean": float(t[:, k].mean()), "T_ci": interval(t[:, k]),
               "T_median": float(np.median(t[:, k])), "T_band": _band(t[:, k]), "p_mean": float(p[:, k].mean()),
               "p_median": float(np.median(p[:, k])), "p_band": _band(p[:, k]),
               "e_err_median": float(np.median(err[:, k])), "e_err_band": _band(err[:, k])}
        if kp1 > 1:
            wins = int(np.sum(t[:, k] < t[:, 1]))
            row["beats_psi1"] = wins / n
            row["beats_psi1_ci"] = list(wilson_interval(wins, n))
        rk = [h["r_realized"][k] for h in hand if h["r_realized"][k] is not None]
        if rk:
            row["r_median"] = float(np.median(rk))
            row["r_band"] = _band(np.asarray(rk))
        if k < kp1 - 1:
            diff = t[:, k] - t[:, k + 1]
            ci = interval(diff)
            row.update(c_iter_mean=float(citer[:, k].mean()), diff_mean=float(diff.mean()), diff_ci=ci,
                       pays=bool(ci[0] > 0.0))
        per_k.append(row)
    best = t.min(axis=1)
    catastrophic = int(np.sum(np.any(p[:, 1:] < 0.5 * p[:, :-1], axis=1)))
    km = int(np.argmin(t.mean(axis=0)))
    flag_counts: dict[str, int] = {}
    for r in records:
        for f in sorted({f for step in r.flags for f in step}):
            flag_counts[f] = flag_counts.get(f, 0) + 1
    return {"n": n, "k_max": kp1 - 1, "c_qpe": c_qpe, "rule": records[0].rule,
            "realizable": not any(r.non_realizable for r in records),
            "non_realizable": sorted({x for r in records for x in r.non_realizable}), "per_k": per_k,
            "optimum": {"mean": float(best.mean()), "ci": interval(best),
                        "k_hist": np.bincount(t.argmin(axis=1), minlength=kp1).tolist()},
            "best_fixed_k": {"k": km, "mean": float(t[:, km].mean()), "ci": interval(t[:, km])},
            "catastrophic_share": catastrophic / n, "catastrophic_ci": list(wilson_interval(catastrophic, n)),
            "stopping": stopping_rules(t, citer) if kp1 > 1 else {}, "flag_counts": flag_counts}


def run_closed_loop_system(name: str, configs: Sequence[tuple], *, n: int, max_iter: int = 5,
                           cache: "scaling.Cache | None" = None, params: FTParameters = FTParameters()) -> dict:
    """The closed-loop block of one system: its inputs, then ``n`` replicates of every configuration.

    ``configs``: (LoopSpec, rule) pairs, or (LoopSpec, rule, metric) triples (default metric spin_orbital).
    Returns a JSON-able dict (``ft.scaling.to_json``): "system" {"name", "c_qpe", "e0", "e1", "e1_any",
    "lam", "n_iter", "p_hf", "qpe_kind", "variance_state", "n_elements"}, "configs" [{"config" (the LoopSpec
    fields, "rule", "metric"), "replicates" (:func:`handoff_inputs` per replicate), "summary"
    (:func:`summarize`)}].
    """
    inputs = system_inputs(name, cache=cache, params=params)
    spec = inputs.spectrum
    out = {"system": {"name": name, "c_qpe": inputs.c_qpe, "e0": spec.e0, "e1": spec.e1, "e1_any": spec.e1_any,
                      "lam": float(inputs.qpe.encoding.lam), "n_iter": int(inputs.qpe.iterations), "p_hf": spec.p_hf,
                      "qpe_kind": inputs.qpe.encoding.kind, "variance_state": inputs.variance_state,
                      "n_elements": inputs.n_elements},
           "configs": []}
    for cfg in configs:
        if len(cfg) == 2:
            loop_spec, rule = cfg
            metric = "spin_orbital"
        elif len(cfg) == 3:
            loop_spec, rule, metric = cfg
        else:
            raise ValueError(f"a configuration is (LoopSpec, rule) or (LoopSpec, rule, metric), got {cfg!r}")
        recs = replicate(inputs, loop_spec, rule, n=n, max_iter=max_iter, metric=metric)
        costs = [closed_loop_costs(inputs, r, params=params) for r in recs]
        rule_name = rule.name if isinstance(rule, StepRule) else rule
        out["configs"].append({"config": dict(dataclasses.asdict(loop_spec), rule=rule_name, metric=metric,
                                              step_rule=None if recs[0].step_rule is None
                                              else dataclasses.asdict(recs[0].step_rule)),
                               "replicates": [handoff_inputs(inputs, r, c) for r, c in zip(recs, costs)],
                               "summary": summarize(inputs, recs, costs)})
    return out
