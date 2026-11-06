"""Fault-tolerant primitives of M3: parameters, phase-gradient rotations, GQSP evolution, AE counts.

Every cost in :mod:`thc_engine_cqe.ft` is a count of Toffolis (one CCZ state each) on
a surface-code machine. An AND gadget costs 1 Toffoli to compute and 0 to uncompute
(measurement-based uncomputation) [cited: Gidney, Quantum 2, 74 (2018)]. Cliffords are
free. Tags: [OF] OpenFermion 1.8.1 source (``openfermion.resource_estimates``);
[derived] derived in the M3 design and re-derived for the specification; [cited]
literature; [estimate] and [assumption] modelling choices. Formula tags F1-F19 are
those of the M3 specification.

Rotations by classical angles (F1, F2)
--------------------------------------
The b-bit phase-gradient register |Phi_b> = 2^{-b/2} sum_x exp(-2 pi i x / 2^b)|x> is an
eigenvector of x -> x + v mod 2^b with eigenvalue exp(+2 pi i v / 2^b) [derived]. A
diagonal phase function that takes the constant phi_c on each of a set of mutually
exclusive indicator cells therefore costs:

* one addition into the gradient register, b - 2 Toffolis
  (:func:`addition_toffolis`) [OF: ``df/compute_cost_df.py``, ``cost4df = 4 n (beta - 2)``];
* the ANDs that compute the cell indicators (counted by the caller);
* nothing for the summand, which the indicators load by CNOT fan-out of the
  constants v_c = :func:`round_angle` (phi_c, b).

Each constant is rounded independently, so the operator-norm error of one addition
is max_c |exp(i phi_c) - exp(2 pi i v_c / 2^b)| <= :func:`rounding_error` <= pi / 2^b. The
unitary is exactly diagonal in the cell basis, so particle number, S_z and the point
group are preserved exactly. A circuit of A additions within an operator-norm budget
delta, with worst-case linear accumulation, needs (F2)

    b = max(b_min, ceil(log2(pi A / delta)))                      (phase_gradient_bits)

bits [derived]. Phase-gradient rotations are the only rotation primitive of M3;
synthesized rotations are out of scope.

Time evolution by GQSP (F8)
---------------------------
A qubitized walk W with normalization lambda has the eigenphases
+-arccos((E - c)/lambda). With x = (E - c)/lambda = cos(theta) and z = exp(i theta),
the Jacobi-Anger expansion gives

    exp(-i tau x) = sum_{k=-inf}^{inf} (-i)^k J_k(tau) z^k,          tau = lambda t,

the same value on both walk eigenvectors of an invariant subspace, so the truncation
to |k| <= R acts as exp(-i tau x) times the identity, up to the tail
2 sum_{j>R} |J_j(tau)| [derived]. GQSP realizes a polynomial of degree d in W with d
calls to controlled-W [cited: Motlagh and Wiebe, PRX Quantum 5, 020368 (2024)]; R of
them replaced by controlled-W^dagger give W^{-R} P(W) with deg P = 2R, the Laurent
polynomial above. The mixed parity of exp(-i tau x) rules out an R + 1 call variant.
Hence [derived]

    R(tau, delta) = min{R >= 0 : 2 sum_{j>R} |J_j(tau)| <= delta}      (jacobi_anger_degree)
    calls         = 2R                                                (gqsp_walk_calls)
    C_evol        = 2R C_W + 3 (2R + 1)(beta - 2)                     (evolution_toffolis)

with three beta-bit phase rotations per GQSP layer (:func:`gqsp_rotation_toffolis`) and
C_W the Toffolis of one controlled walk step (OpenFermion's step cost includes the
control [OF]). Reference values: R(0.79, 1e-4) = 5, R(1.16, 1e-4) = 6,
R(6.81, 1e-6) = 17 (tests/test_ft_primitives.py).

Amplitude estimation (F11)
--------------------------
A Hadamard-test circuit with success probability p = (1 + <O>/alpha)/2 is iterated
with G = A S_0 A^dagger S_good. G has the eigenphases +-2 theta, sin^2 theta = p, and
<O> = -alpha cos 2 theta. Sine-window phase estimation with M applications of G
estimates 2 theta to an RMS error kappa pi / M with kappa -> 1 (simulated: 0.974,
0.993, 0.998 for M = 63, 255, 1023) [derived; kappa verified], so
sigma(<O>) <= kappa pi alpha / M. Each G applies A once and A^dagger once, hence

    N = 2 pi kappa alpha / eps                                       (ae_applications)

applications of A or A^dagger for an RMS error eps.

Parameters
----------
:class:`FTParameters` holds every budget and modelling constant of M3 (specification
section 3); every function of :mod:`thc_engine_cqe.ft` takes
``params: FTParameters = FTParameters()``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.special

from ..measure.targets import KINDS

__all__ = [
    "FTParameters",
    "addition_toffolis",
    "phase_gradient_bits",
    "round_angle",
    "rounding_error",
    "jacobi_anger_degree",
    "gqsp_walk_calls",
    "gqsp_rotation_toffolis",
    "evolution_toffolis",
    "ae_applications",
]

# First line-search candidate of M2's hardware rule; candidates are _LS_EPS_BASE * 2^j
# (benchmarks/run_optimizers.py DEFAULT_CANDIDATES, "pos11" in benchmarks/run_sampled_cqe.py).
_LS_EPS_BASE = 0.00625


@dataclass(frozen=True)
class FTParameters:
    """Budgets and modelling constants of the M3 cost layer (specification section 3).

    Energies are in Hartree, times in units of 1/Hartree, angles in radians.

    Attributes:
        delta_step: operator-norm rounding budget of one compiled CQE step (F2, F3).
            1e-3 adds at most one L-BFGS step [M3 design, memo 1 section 2.4; assumption].
        delta_readout: rounding budget of one readout basis rotation (F4).
        b_min: minimum number of phase-gradient bits (F2).
        chi, beta: coefficient and rotation bits inside OpenFermion's block encodings
            [OF tests: chi = 10, beta = 16]; beta also sets the GQSP rotations (F8).
        eps_evol: GQSP truncation error of exp(-i delta H) (F8).
        r: relative RMS Frobenius error of the residual over the full tensor [M2].
        kind: residual element set, "spin_free" or "spin_orbital".
        kappa, kappa_practical: amplitude-estimation constant, ideal and practical (F11).
        kappa_g: band of the gradient-estimation constant (F12) [assumption].
        n_candidates: line-search candidates eps_c = 0.00625 * 2^j, j < n_candidates
            (:attr:`eps_candidates`; M2 hardware rule).
        eps_ls: precision of every line-search energy (Hartree, F13).
        target: chemical accuracy (Hartree); k* is the first iterate with E_k - E_0 <= target.
        de_qpe: QPE error (Hartree, F9).
        de_fact: largest admissible |E_0(H') - E_0| of a factorized H' (Hartree, F9).
        infid_tol: largest admissible ground-state infidelity of H' (F9).
        delta_conf: failure probability of the fixed-confidence QPE rule (F15).
        e_tol_energy: energy error of the CQE-only route (Hartree, F17).
        sigma_final: statistical error of the final CQE-only energy (Hartree, F17).
        p_fail_shot, p_fail_ae, p_fail_qpe: logical failure budgets of one sampling shot,
            one amplitude-estimation run and one QPE run (F18).
        aux_deltas: evolution times delta of the aux routes (1/Hartree, F10, F11)
            [M3 design, memo 2].
        n_settings, seed: matchgate settings and seed of the Rao-Blackwellized
            per-shot variances for N >= 6 [M2].
    """

    delta_step: float = 1e-3
    delta_readout: float = 1e-5
    b_min: int = 4
    chi: int = 10
    beta: int = 16
    eps_evol: float = 1e-6
    r: float = 0.3
    kind: str = "spin_free"
    kappa: float = 1.0
    kappa_practical: float = 2.0
    kappa_g: tuple[float, ...] = (1.0, 10.0, 100.0)
    n_candidates: int = 11
    eps_ls: float = 1.6e-3
    target: float = 1.6e-3
    de_qpe: float = 1e-3
    de_fact: float = 5e-4
    infid_tol: float = 1e-4
    delta_conf: float = 0.01
    e_tol_energy: float = 1e-3
    sigma_final: float = 3e-4
    p_fail_shot: float = 1e-5
    p_fail_ae: float = 1e-4
    p_fail_qpe: float = 1e-2
    aux_deltas: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2)
    n_settings: int = 5000
    seed: int = 7

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        # b_min and beta >= 3: a phase-gradient addition needs 3 bits (addition_toffolis).
        for name, least in (("b_min", 3), ("beta", 3), ("chi", 1), ("n_candidates", 1),
                            ("n_settings", 1), ("seed", 0)):
            v = getattr(self, name)
            if isinstance(v, bool) or int(v) != v or v < least:
                raise ValueError(f"{name} must be an integer >= {least}, got {v!r}")
            object.__setattr__(self, name, int(v))
        for name in ("delta_step", "delta_readout", "eps_evol", "r", "kappa", "kappa_practical",
                     "eps_ls", "target", "de_qpe", "de_fact", "infid_tol", "e_tol_energy",
                     "sigma_final"):
            if not float(getattr(self, name)) > 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)!r}")
        for name in ("delta_conf", "p_fail_shot", "p_fail_ae", "p_fail_qpe"):
            if not 0.0 < float(getattr(self, name)) < 1.0:
                raise ValueError(f"{name} must lie in (0, 1), got {getattr(self, name)!r}")
        for name in ("kappa_g", "aux_deltas"):
            values = tuple(float(x) for x in getattr(self, name))
            if not values or min(values) <= 0.0:
                raise ValueError(f"{name} must be a non-empty sequence of positive numbers")
            object.__setattr__(self, name, values)

    @property
    def eps_candidates(self) -> tuple[float, ...]:
        """Line-search step sizes 0.00625 * 2^j, j = 0..n_candidates - 1 (F13; 0.00625 to 6.4 by default)."""
        return tuple(_LS_EPS_BASE * 2.0 ** j for j in range(self.n_candidates))


def _check_bits(bits: int, least: int) -> int:
    b = int(bits)
    if b != bits or b < least:
        raise ValueError(f"bits must be an integer >= {least}, got {bits!r}")
    return b


def addition_toffolis(bits: int) -> int:
    """Toffolis of one addition into a ``bits``-bit phase-gradient register: bits - 2 (F1).

    Raises ValueError for bits < 3.
    """
    return _check_bits(bits, 3) - 2


def phase_gradient_bits(n_additions: int, delta: float, b_min: int = 4) -> int:
    """Phase-gradient bits b = max(b_min, ceil(log2(pi A / delta))) for A additions (F2).

    The smallest b >= b_min with A pi / 2^b <= delta: the worst-case sum of the A
    per-addition operator-norm errors pi / 2^b stays within ``delta`` (radians of
    operator norm). A = 0 gives b_min.
    """
    a = int(n_additions)
    if a != n_additions or a < 0:
        raise ValueError(f"n_additions must be a non-negative integer, got {n_additions!r}")
    if not delta > 0.0:
        raise ValueError(f"delta must be positive, got {delta!r}")
    b = _check_bits(b_min, 1)
    if a == 0:
        return b
    b = max(b, math.ceil(math.log2(math.pi * a / delta)))
    # Guard the ceil(log2) against floating-point round-off at exact powers of two.
    while math.pi * a / 2.0 ** b > delta:
        b += 1
    while b > b_min and math.pi * a / 2.0 ** (b - 1) <= delta:
        b -= 1
    return b


def round_angle(theta: float, bits: int) -> int:
    """Phase-gradient summand v = round(2^b theta / 2 pi) mod 2^b of the phase exp(i theta) (F1)."""
    b = _check_bits(bits, 1)
    if not math.isfinite(theta):
        raise ValueError(f"theta must be finite, got {theta!r}")
    return int(round(theta * 2.0 ** b / (2.0 * math.pi))) % (1 << b)


def rounding_error(theta: float, bits: int) -> float:
    """|theta - 2 pi v / 2^b| wrapped to [0, pi] for v = :func:`round_angle` (theta, b); <= pi / 2^b.

    Bounds |exp(i theta) - exp(2 pi i v / 2^b)| (radians).
    """
    b = _check_bits(bits, 1)
    d = theta - 2.0 * math.pi * round_angle(theta, b) / 2.0 ** b
    return abs((d + math.pi) % (2.0 * math.pi) - math.pi)


def jacobi_anger_degree(tau: float, delta: float) -> int:
    """F8 truncation degree R(tau, delta) = min{R >= 0 : 2 sum_{j>R} |J_j(tau)| <= delta}.

    ``tau`` = lambda t is dimensionless (only |tau| matters). The tail is summed
    exactly up to j = ceil(e |tau|) + ceil(log2(1/delta)) + 20; beyond it
    |J_j(tau)| <= (|tau|/2)^j / j! <= 2^-j, so the neglected part is below
    delta 2^-20.
    """
    t = abs(float(tau))
    if not math.isfinite(t):
        raise ValueError(f"tau must be finite, got {tau!r}")
    if not 0.0 < delta:
        raise ValueError(f"delta must be positive, got {delta!r}")
    if t == 0.0:
        return 0
    j_max = math.ceil(math.e * t) + max(0, math.ceil(math.log2(1.0 / delta))) + 20
    a = np.abs(scipy.special.jv(np.arange(j_max + 1), t))
    suffix = np.cumsum(a[::-1])[::-1]                  # suffix[k] = sum_{j >= k} |J_j|
    tails = 2.0 * np.append(suffix[1:], 0.0)           # tails[R] = 2 sum_{j > R} |J_j|
    return int(np.argmax(tails <= delta))


def gqsp_walk_calls(tau: float, delta: float) -> int:
    """Controlled walk calls of the GQSP Laurent polynomial of exp(-i tau x): 2 R(tau, delta) (F8)."""
    return 2 * jacobi_anger_degree(tau, delta)


def gqsp_rotation_toffolis(n_calls: int, bits: int) -> int:
    """Toffolis of the GQSP phase rotations: 3 (n_calls + 1)(bits - 2) (F8).

    n_calls + 1 layers of three ``bits``-bit phase-gradient rotations.
    """
    n = int(n_calls)
    if n != n_calls or n < 0:
        raise ValueError(f"n_calls must be a non-negative integer, got {n_calls!r}")
    return 3 * (n + 1) * addition_toffolis(bits)


def evolution_toffolis(lam: float, t: float, toffoli_per_step: int, *, delta: float = 1e-6,
                       beta: int = 16) -> int:
    """Toffolis of exp(-i t H) by GQSP on a qubitized walk: 2R C_W + 3 (2R + 1)(beta - 2) (F8).

    Args:
        lam: normalization lambda of the block encoding (Hartree).
        t: evolution time (1/Hartree); tau = lam |t|.
        toffoli_per_step: C_W, Toffolis of one controlled walk step.
        delta: truncation error of the Jacobi-Anger series (operator norm).
        beta: bits of the GQSP phase rotations.

    R = 0 (tau so small that exp(-i tau x) is the identity within delta) costs 0.
    """
    if not lam > 0.0:
        raise ValueError(f"lam must be positive, got {lam!r}")
    c_w = int(toffoli_per_step)
    if c_w != toffoli_per_step or c_w < 0:
        raise ValueError(f"toffoli_per_step must be a non-negative integer, got {toffoli_per_step!r}")
    calls = gqsp_walk_calls(lam * t, delta)
    if calls == 0:
        return 0
    return calls * c_w + gqsp_rotation_toffolis(calls, beta)


def ae_applications(alpha: float, eps: float, kappa: float = 1.0) -> float:
    """Applications of a Hadamard-test circuit or its inverse for RMS error eps: 2 pi kappa alpha / eps (F11).

    ``alpha`` is the normalization of the observable (<O> in [-alpha, alpha]) and ``eps``
    the RMS error, both in the units of <O>.
    """
    if not alpha >= 0.0:
        raise ValueError(f"alpha must be non-negative, got {alpha!r}")
    if not eps > 0.0:
        raise ValueError(f"eps must be positive, got {eps!r}")
    if not kappa > 0.0:
        raise ValueError(f"kappa must be positive, got {kappa!r}")
    return 2.0 * math.pi * kappa * alpha / eps
