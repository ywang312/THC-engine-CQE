"""Fault-tolerant cost of preparing the CQE iterates Psi_k = prod_{j<k} exp(eps_j D_j) |HF> (F3-F7).

The unit is the Toffoli (one CCZ state). An AND gadget costs 1 Toffoli to compute and 0
to uncompute [cited: Gidney, Quantum 2, 74 (2018)]. Cliffords are free: Jordan-Wigner
parities, CNOT basis maps and the HF determinant (C(Psi_0) = 0). Every rotation is a
phase-gradient addition of a classical constant (:mod:`thc_engine_cqe.ft.primitives`,
F1): b - 2 Toffolis per addition [OF: ``cost4df = 4 n (beta - 2)``], the ANDs of its
indicator cells, and a rounding error <= pi / 2^b per addition. The gradient register
is a catalyst, prepared once per machine and kept across steps and shots; its b qubits
are part of Q_prep, its preparation is not counted [assumption; OpenFermion's
convention; primitives module docstring]. Tags: [OF] OpenFermion 1.8.1 source;
[verified] reproduced by an explicit circuit in :mod:`thc_engine_cqe.ft.circuits` and
tests/test_ft_circuits.py; [derived], [cited], [estimate] and [assumption] as in the M3
specification. Formula tags F1-F19 are those of the specification.

Primitive counts (:class:`PrimitiveCounts`)
-------------------------------------------
A circuit is counted as n_add additions, n_and ANDs and n_pauli_rot Pauli rotations;
at b bits it costs n_add (b - 2) + n_and Toffolis. The Pauli rotations are the
rotation count of the same circuit in Pauli-based computation, exported for partially
fault-tolerant models (roadmap M6) and not used in any Toffoli count here.

Excitations (:func:`excitation_counts`, F3). ``diagnostics.cost.spin_orbital_excitations``
writes a spin-free generator exactly as a sum of spin-orbital excitations; each
exponential is a Clifford basis map around a multi-controlled R_y, and R_y(phi) on the
target t is S H R_z(phi) H S^dag, i.e. a diagonal phase +-phi/2 on the two cells c AND t
and c AND NOT t of the control indicator c (the second one is c XOR (c AND t), a
Clifford). Per excitation:

    kind      exponential                       ANDs                   additions   Pauli rotations
    double    exp(t tau - conj(t) tau^dag)       2 (3-input control)    1           8
                                                + 1 (cell c AND t)
    single    exp(n_c (t a+_x a_y - h.c.))       1 (control) + 1 (cell)  1           4
    diagonal  exp(T n_P n_Q), T imaginary        1 (cell)               1           3

[derived from the cell structure; the construction itself is verified for doubles and
singles: ``circuits.pg_excitation_unitary``, with the indicators as classical functions
of the basis state, equals ``diagnostics.cost.circuit_unitary(excitation_circuit(exc))``
to 1e-12 unrounded and is within pi / 2^b of it rounded]. The Pauli rotations are the Jordan-Wigner
terms of each generator (8, 4 and 3 mutually commuting strings) [memo 3]. Diagonal terms
occur only for complex generators (their amplitudes are imaginary).

Elementwise step (:func:`elementwise_step_cost`, F3, the default method)
-----------------------------------------------------------------------
For a real direction d (a pair-order tensor, the step is exp(eps D), D = sum d e) and
optional orbital irreps (F2a, :mod:`thc_engine_cqe.ft.symmetry`):

1. X = ``spin_orbital_excitations(mask(d), tol=1e-12)``; mask zeroes the
   symmetry-forbidden entries (no mask without irreps).
2. theta_x = |eps t_x|, the Givens angle of exp(eps (t_x tau_x - h.c.)). The relative
   tolerance commutes with the scaling by eps, so X is expanded once for every step
   size (:func:`candidate_step_toffolis`).
3. Fixed point: A = #kept, b = max(b_min, ceil(log2(pi A / delta_step))) (F2), drop
   every x with theta_x < pi / 2^b (it rounds to the zero phase), repeat. The kept set
   is the prefix of X in decreasing angle, and the iteration ends at the largest b with
   b = F2(#{theta >= pi / 2^b}).
4. C_step = A (b - 2) + 3 n_d + 2 n_s + n_g and Q_prep = 2N + 3b + 8 logical qubits
   (system, gradient register, summand, b - 1 adder carries and the AND ancillas)
   [derived; memo 1, specification S8]. A step without kept excitations costs 0 and
   needs 2N qubits; so does eps = 0, whose expansion is empty (nothing is dropped or
   masked).

Rounding. Each kept excitation is realized as exp(theta~ K) with theta~ the angle
rounded to a multiple of 2 pi / 2^b (both cell constants round symmetrically), so it
errs by <= pi / 2^b in operator norm; a dropped one by <= theta_x. The compiled step is
therefore within ``StepCost.rounding_bound`` = A pi / 2^b + sum_dropped theta_x of the
product of the exact exponentials of X; F3 budgets only the first term, and the flag
"rounding bound above delta_step" marks steps where the dropped angles push the sum
above delta_step.

Product-formula error. The product of the A exponentials differs from exp(eps D) at
second order in eps, and F3 does not budget it: every step that compiles two or more
excitations carries the flag "product-formula error not in budget" (the leaf steps carry
its order-1 Trotter analogue, F6). The product conserves N, S_z and the point group
exactly, but not S^2: a spin-free double is a sum of alpha-alpha, beta-beta and
alpha-beta excitations, which do not commute with S^2 one by one (F1's S^2 statement
holds for one phase function, not for a product of them). ``circuits.replay_elementwise``
measures the error. After the exact L-BFGS step from RHF, ||Psi~_1 - Psi_1|| is 2.2e-4
(LiH) to 2.0e-3 (N2) at equilibrium [memo 1 section 2.4: 1.2e-3 to 2.0e-3, <= 1.8e-3 at
the next step, and closed-loop compiled runs needed no extra step], but 2.7e-2 (H6 at
3 bohr) and 3.1e-2 (N2 at 2.0 A), 27 and 31 times delta_step, and <S^2> of Psi~_1 is
<= 1.5e-5 at equilibrium but 1.5e-3 and 3.9e-3 on the stretched builds, against 0 for
Psi_1 (circuits module docstring; tests/test_ft_circuits.py).

Complex directions raise ValueError: a complex amplitude adds two Z rotations per
excitation and complex states are out of scope (F3).

Orbital rotations and diagonal-Coulomb layers (F4, F5)
------------------------------------------------------
An N x N unitary is N(N-1)/2 = g Givens rotations per spin plus diagonal phases [cited:
Clements et al., Optica 3, 1460 (2016); Kivlichan et al., PRL 120, 110501 (2018);
``circuits.givens_decomposition`` checks the count]. A real Givens rotation acts on
both spins with the same angle; after a Clifford basis change each copy is
exp(i theta s_sigma) with s_sigma in {-1, 0, 1}, so the eight one-hot cells of
(s_alpha, s_beta) share one addition and cost 6 ANDs. A complex one adds the phase
exp(i phi (n_q,alpha + n_q,beta)): one addition and 1 AND [memo 1, derived]
(:func:`orbital_rotation_counts`):

    real:     g additions + 6 g ANDs,   4 g Pauli rotations
    complex:  2 g additions + 7 g ANDs, 6 g Pauli rotations

The DF readout rotation of an energy measurement is real and has its own budget,
b_r = F2(g, delta_readout) (its rounding error is an energy bias of about
2 ||F_l^2|| delta_readout, specification S12): C_rot = g (b_r - 2) + 6 g
(:func:`readout_rotation_toffolis`; 405 Toffolis at N = 6, b_r = 23).

A diagonal-Coulomb layer exp(i tau (sum_p mu_p n_p)^2), n_p = n_p,alpha + n_p,beta, on
n_a orbitals with mu_p != 0 has the cells (n_p, n_q) in {1, 2}^2 per pair (4 ANDs, one
addition) and n_p in {1, 2} per site (1 AND, one addition, which also absorbs the
phases of the adjacent rotation) [memo 1] (:func:`diagonal_layer_counts`, F5):

    n_a (n_a - 1)/2 + n_a additions,  2 n_a (n_a - 1) + n_a ANDs,  n_a (2 n_a + 1) Pauli rotations

Leaf step (:func:`leaf_step_cost`, F6)
--------------------------------------
``factor.generator.factorize_generator(d, rank_tol=1e-10)`` gives
D = sum_k c_k i Qhat_k^2 + Ohat_F with Qhat_k = U(V_k) D_k U(V_k)^dag. The order-1 product
(one-body term first, then the leaves in decreasing |c_k|) merges adjacent rotations into
one, so a step with L leaves has L complex rotations and L layers, and a circuit needs
one final complex rotation [diagnostics.cost, verified there for the CNOT model]. With
n_a,k = ``diagnostics.cost.leaf_active_orbitals`` (|mu| > mu_tol max |mu|):

    A_leaf   = L N(N-1) + sum_k n_a,k (n_a,k + 1)/2
    AND_leaf = 7 L N(N-1)/2 + sum_k n_a,k (2 n_a,k - 1)
    C_step   = A_leaf (b - 2) + AND_leaf,       b = F2(A_leaf, delta_step)

Q_prep = 2N + 3b + 8 as above. Every leaf step carries the flags "complex state" (the
leaf product of a real generator makes the state complex at O(eps^2), and the exact
trajectory it is costed on is not propagated through it) and "order-1 Trotter error not
in budget" (about 1e-2 in state norm at the RHF step [memo 1]). The leaf rotations are
complex: Q_k = (S_k - i A_k)/sqrt2 for real generators.

Cumulative cost (:func:`prep_cost`, F7)
---------------------------------------
C(Psi_0) = 0 and C(Psi_k) = sum_{j<k} C_step(j); Q(Psi_k) = max_{j<k} Q_prep(j), and 2N
for k = 0. A leaf circuit adds its final complex rotation once, at its own budget,
b_f = F2(N(N-1), delta_step) [assumption: the specification leaves b_f open; memo 1 used
the largest b of the trajectory, which makes C(Psi_k) depend on later steps]; it is
``PrepCost.final_rotation`` and is added for every k >= 1 at which a leaf was applied.
With irreps, :func:`prep_cost` masks every direction before compiling it, for both
methods (F2a).

Observed (exact L-BFGS trajectories on the symmetry-adapted builds, delta_step = 1e-3,
spec Appendix A; tests/test_ft_stateprep.py): H4 steps 0-2 cost 160 (10 doubles,
b = 15), 1192 (40 doubles and 24 singles, b = 18) and 1154 Toffolis, C(Psi_3) = 2506;
H6 at 1.0 A 969 (51 doubles, b = 18), then 8442-8526 (b = 21), C(Psi_6) = 43427 on
83 logical qubits; N2 486 (27 doubles, b = 17) and 1045 (55 doubles, b = 18),
C(Psi_3) = 2576. The leaf steps of the same directions cost 3.7k / 19k / 18k Toffolis at
the RHF step and 5.6k / 37k / 25k later. The mask matters on the late N2 steps, whose
forbidden entries are amplified round-off (symmetry module docstring): unmasked, the
forbidden excitations survive the rounding from k = 9 on, and C(Psi_12) is 12854
instead of 11082 Toffolis (k = 9: 1910 instead of 1045; k = 10: 1357 instead of 450;
one thread, round-off dependent).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..diagnostics.cost import Excitation, leaf_active_orbitals, spin_orbital_excitations
from ..factor.generator import factorize_generator
from .primitives import FTParameters, addition_toffolis, phase_gradient_bits
from .symmetry import mask_pair_tensor

__all__ = [
    "METHODS",
    "PrimitiveCounts",
    "excitation_counts",
    "orbital_rotation_counts",
    "diagonal_layer_counts",
    "StepCost",
    "elementwise_excitations",
    "elementwise_step_cost",
    "leaf_step_cost",
    "PrepCost",
    "prep_cost",
    "candidate_step_toffolis",
    "readout_rotation_toffolis",
]

METHODS = ("elementwise", "leaves")

_EXPANSION_TOL = 1e-12          # F3: relative tolerance of the spin-orbital expansion
_LEAF_RANK_TOL = 1e-10          # F6: rank tolerance of the generator factorization
_LEAF_FLAGS = ("complex state", "order-1 Trotter error not in budget")
_FLAG_PRODUCT = "product-formula error not in budget"
_FLAG_ROUNDING = "rounding bound above delta_step"


# ------------------------------------------------------------------------------ counts
@dataclass(frozen=True)
class PrimitiveCounts:
    """Additions, ANDs and Pauli rotations of a phase-gradient circuit (module docstring)."""

    n_add: int = 0
    n_and: int = 0
    n_pauli_rot: int = 0

    def toffolis(self, bits: int) -> int:
        """n_add (bits - 2) + n_and Toffolis with a ``bits``-bit phase-gradient register (F1)."""
        if self.n_add == 0:
            return self.n_and
        return self.n_add * addition_toffolis(bits) + self.n_and

    def __add__(self, other: "PrimitiveCounts") -> "PrimitiveCounts":
        if not isinstance(other, PrimitiveCounts):
            return NotImplemented
        return PrimitiveCounts(self.n_add + other.n_add, self.n_and + other.n_and,
                               self.n_pauli_rot + other.n_pauli_rot)


# (additions, ANDs, Pauli rotations) per excitation kind (F3).
_PER_EXCITATION = {"double": (1, 3, 8), "single": (1, 2, 4), "diagonal": (1, 1, 3)}


def excitation_counts(excitations: Sequence[Excitation]) -> PrimitiveCounts:
    """Counts of one exponential per excitation (F3): double 1 + 3 + 8, single 1 + 2 + 4,
    diagonal 1 + 1 + 3 (additions + ANDs + Pauli rotations)."""
    n_add = n_and = n_rot = 0
    for exc in excitations:
        try:
            a, g, r = _PER_EXCITATION[exc.kind]
        except KeyError:
            raise ValueError(f"unknown excitation kind {exc.kind!r}") from None
        n_add, n_and, n_rot = n_add + a, n_and + g, n_rot + r
    return PrimitiveCounts(n_add, n_and, n_rot)


def _check_count(value: int, name: str) -> int:
    v = int(value)
    if v != value or v < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return v


def orbital_rotation_counts(norb: int, *, complex_: bool) -> PrimitiveCounts:
    """Both-spin orbital rotation U(V) as g = N(N-1)/2 Givens rotations per spin (F4).

    Real: g additions, 6 g ANDs, 4 g Pauli rotations. Complex: 2 g additions, 7 g ANDs,
    6 g Pauli rotations. The final diagonal phases merge into an adjacent layer.
    """
    n = _check_count(norb, "norb")
    g = n * (n - 1) // 2
    if complex_:
        return PrimitiveCounts(2 * g, 7 * g, 6 * g)
    return PrimitiveCounts(g, 6 * g, 4 * g)


def diagonal_layer_counts(n_active: int) -> PrimitiveCounts:
    """exp(i tau (sum_p mu_p n_p)^2) on ``n_active`` orbitals (F5).

    n_a (n_a - 1)/2 + n_a additions, 2 n_a (n_a - 1) + n_a ANDs, n_a (2 n_a + 1) Pauli rotations.
    """
    na = _check_count(n_active, "n_active")
    return PrimitiveCounts(na * (na - 1) // 2 + na, 2 * na * (na - 1) + na, na * (2 * na + 1))


# --------------------------------------------------------------------------- step costs
@dataclass(frozen=True)
class StepCost:
    """Compiled cost of one CQE step exp(eps D) (F3 elementwise, F6 leaves).

    Attributes:
        method: "elementwise" or "leaves".
        step: index j of the step in its trajectory.
        eps: step size (dimensionless; D is the unnormalized direction).
        toffolis: C_step.
        bits: phase-gradient bits b (F2; b_min for a step without additions).
        logical_qubits: Q_prep = 2N + 3b + 8 (2N for an empty step).
        counts: additions, ANDs and Pauli rotations of the step.
        n_double, n_single, n_diagonal: kept excitations by kind (elementwise).
        n_dropped: excitations of the masked expansion that round to zero (elementwise;
            0 for eps = 0).
        n_masked: excitations of the unmasked expansion that the symmetry mask removes
            (elementwise, diagnostic; 0 without irreps and for eps = 0).
        n_leaves: L, leaves compiled (leaves).
        n_active: n_a,k of every compiled leaf (leaves).
        flags: caveats (module docstring): "product-formula error not in budget" (two or
            more kept excitations) and "rounding bound above delta_step" (elementwise);
            "complex state" and "order-1 Trotter error not in budget" (leaves).
        rounding_bound: bound on the operator-norm distance between the compiled step and
            the product of the exact exponentials of its factors: A pi / 2^b plus the
            angles of the dropped excitations (elementwise), A_leaf pi / 2^b (leaves).
    """

    method: str
    step: int
    eps: float
    toffolis: int
    bits: int
    logical_qubits: int
    counts: PrimitiveCounts
    n_double: int = 0
    n_single: int = 0
    n_diagonal: int = 0
    n_dropped: int = 0
    n_masked: int = 0
    n_leaves: int | None = None
    n_active: tuple[int, ...] | None = None
    flags: tuple[str, ...] = ()
    rounding_bound: float = 0.0


def _check_direction(d: np.ndarray, *, real: bool) -> np.ndarray:
    arr = np.asarray(d)
    if arr.ndim != 4 or len(set(arr.shape)) != 1:
        raise ValueError(f"expected an (N, N, N, N) pair-order direction, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("the direction has non-finite entries")
    if real and np.iscomplexobj(arr):
        if np.any(arr.imag != 0):
            raise ValueError("complex directions are out of scope for the elementwise model (F3): "
                             "a complex amplitude adds two Z rotations per excitation")
        arr = np.ascontiguousarray(arr.real)
    return arr


def _check_eps(eps: float) -> float:
    if np.iscomplexobj(eps):
        raise ValueError(f"eps must be a finite real number, got {eps!r}")
    try:
        e = float(eps)
    except (TypeError, ValueError):
        raise ValueError(f"eps must be a finite real number, got {eps!r}") from None
    if e != eps or not math.isfinite(e):
        raise ValueError(f"eps must be a finite real number, got {eps!r}")
    return e


@dataclass(frozen=True)
class _Expansion:
    """Spin-orbital expansion of mask(d), shared by every step size (F3, step 1)."""

    norb: int
    excitations: tuple[Excitation, ...]      # decreasing |amplitude|
    magnitudes: np.ndarray                   # |t_x| of d (eps = 1)
    n_masked: int


def _irrep_product(exc: Excitation, irreps: np.ndarray, norb: int) -> int:
    """Irrep of a+_{c0} a+_{c1} a_{d1} a_{d0}: XOR over all four operators (a control cancels)."""
    x = 0
    for mode in exc.creators + exc.annihilators:
        x ^= int(irreps[mode % norb])
    return x


def _expand(d: np.ndarray, irreps) -> _Expansion:
    d = _check_direction(d, real=True)
    n = d.shape[0]
    n_masked = 0
    if irreps is not None:
        masked = mask_pair_tensor(d, irreps)          # validates irreps against d
        ir = np.asarray(irreps)
        full = spin_orbital_excitations(d, tol=_EXPANSION_TOL)
        n_masked = sum(1 for e in full if _irrep_product(e, ir, n) != 0)
        d = masked
    exc = tuple(spin_orbital_excitations(d, tol=_EXPANSION_TOL))
    mags = np.array([abs(e.amplitude) for e in exc], dtype=float)
    return _Expansion(n, exc, mags, n_masked)


def _compile(exp: _Expansion, eps: float, params: FTParameters) -> tuple[int, int, np.ndarray]:
    """F3 fixed point: (number of kept excitations, bits b, angles theta of all excitations)."""
    theta = abs(eps) * exp.magnitudes
    n_keep = theta.size
    while True:
        b = phase_gradient_bits(n_keep, params.delta_step, params.b_min)
        n_new = int(np.count_nonzero(theta[:n_keep] >= math.pi / 2.0 ** b))
        if n_new == n_keep:
            return n_keep, b, theta
        n_keep = n_new


def _elementwise_cost(exp: _Expansion, eps: float, params: FTParameters, step: int) -> StepCost:
    n_keep, b, theta = _compile(exp, eps, params)
    kept = exp.excitations[:n_keep]
    counts = excitation_counts(kept)
    kinds = [e.kind for e in kept]
    bound = n_keep * math.pi / 2.0 ** b + float(np.sum(theta[n_keep:]))
    flags = (((_FLAG_PRODUCT,) if n_keep >= 2 else ())
             + ((_FLAG_ROUNDING,) if bound > params.delta_step else ()))
    # eps = 0: F3 expands eps * mask(d) = 0, so nothing is dropped or masked.
    n_dropped, n_masked = (len(exp.excitations) - n_keep, exp.n_masked) if eps != 0.0 else (0, 0)
    n = exp.norb
    return StepCost(method="elementwise", step=step, eps=eps, toffolis=counts.toffolis(b), bits=b,
                    logical_qubits=2 * n + 3 * b + 8 if n_keep else 2 * n, counts=counts,
                    n_double=kinds.count("double"), n_single=kinds.count("single"),
                    n_diagonal=kinds.count("diagonal"), n_dropped=n_dropped, n_masked=n_masked,
                    flags=flags, rounding_bound=bound)


def elementwise_excitations(d: np.ndarray, eps: float, *, irreps: np.ndarray | None = None,
                            params: FTParameters = FTParameters()) -> tuple[list[Excitation], int]:
    """Kept excitations of the compiled step exp(eps D) and its phase-gradient bits (F3).

    Returns the excitations of mask(d) that survive the rounding fixed point, with the
    amplitudes eps t_x of the step (Givens angles |eps t_x|), in decreasing angle (the
    order of ``spin_orbital_excitations``), and b.
    """
    e = _check_eps(eps)
    exp = _expand(d, irreps)
    n_keep, b, _ = _compile(exp, e, params)
    return [dataclasses.replace(x, amplitude=e * x.amplitude) for x in exp.excitations[:n_keep]], b


def elementwise_step_cost(d: np.ndarray, eps: float, *, irreps: np.ndarray | None = None,
                          params: FTParameters = FTParameters(), step: int = 0) -> StepCost:
    """Toffolis, bits and logical qubits of the elementwise step exp(eps D) (F3).

    Args:
        d: (N, N, N, N) real pair-order direction (``CQEStep.generator``).
        eps: step size.
        irreps: orbital irreps (``symmetry.orbital_irreps``); the forbidden entries of d
            are zeroed first (F2a). None: no mask.
        params: delta_step and b_min are used.
        step: index recorded in the result.

    Raises:
        ValueError: complex d (out of scope, module docstring) or invalid inputs.
    """
    return _elementwise_cost(_expand(d, irreps), _check_eps(eps), params, _check_count(step, "step"))


def leaf_step_cost(d: np.ndarray, eps: float, *, n_leaves: int | None = None, mu_tol: float = 1e-6,
                   irreps: np.ndarray | None = None, params: FTParameters = FTParameters(),
                   step: int = 0) -> StepCost:
    """Toffolis, bits and logical qubits of the order-1 leaf product for exp(eps D) (F6).

    Args:
        d: (N, N, N, N) pair-order direction (real or complex).
        eps: step size. F6 does not depend on it, except that eps = 0 is the identity
            (no leaves are compiled).
        n_leaves: compile only the top ``n_leaves`` leaves (all by default).
        mu_tol: relative threshold of the active orbitals of a leaf.
        irreps: orbital irreps; mask d first (F2a). None: no mask.
        params: delta_step and b_min are used.
        step: index recorded in the result.

    The final complex rotation of a circuit is not included (:func:`prep_cost`).
    """
    d = _check_direction(d, real=False)
    e = _check_eps(eps)
    j = _check_count(step, "step")
    n = d.shape[0]
    if irreps is not None:
        d = mask_pair_tensor(d, irreps)
    n_active: tuple[int, ...] = ()
    if e != 0.0:
        fac = factorize_generator(d, rank_tol=_LEAF_RANK_TOL)
        size = fac.n_leaves if n_leaves is None else min(_check_count(n_leaves, "n_leaves"), fac.n_leaves)
        n_active = tuple(leaf_active_orbitals(fac, k, mu_tol) for k in range(size))
    counts = PrimitiveCounts()
    rotation = orbital_rotation_counts(n, complex_=True)
    for na in n_active:
        counts = counts + rotation + diagonal_layer_counts(na)
    b = phase_gradient_bits(counts.n_add, params.delta_step, params.b_min)
    return StepCost(method="leaves", step=j, eps=e, toffolis=counts.toffolis(b), bits=b,
                    logical_qubits=2 * n + 3 * b + 8 if n_active else 2 * n, counts=counts,
                    n_leaves=len(n_active), n_active=n_active,
                    flags=_LEAF_FLAGS if n_active else (),
                    rounding_bound=counts.n_add * math.pi / 2.0 ** b)


# --------------------------------------------------------------------------- cumulative
@dataclass(frozen=True)
class PrepCost:
    """State-preparation cost of every iterate of a trajectory (F7).

    Attributes:
        method: "elementwise" or "leaves".
        steps: the K step costs.
        cumulative: (K + 1,) int64, C(Psi_k) in Toffolis; cumulative[0] = 0.
        logical_qubits: (K + 1,) int64, Q(Psi_k); 2N at k = 0.
        final_rotation: Toffolis of the final complex rotation of a leaf circuit, included
            in cumulative[k] for every k >= 1 at which a leaf was applied (0 for elementwise).
    """

    method: str
    steps: tuple[StepCost, ...]
    cumulative: np.ndarray
    logical_qubits: np.ndarray
    final_rotation: int = 0


def _readonly(a: np.ndarray) -> np.ndarray:
    a.flags.writeable = False
    return a


def prep_cost(directions: Sequence[np.ndarray], eps: Sequence[float], *, method: str = "elementwise",
              irreps: np.ndarray | None = None, params: FTParameters = FTParameters()) -> PrepCost:
    """C(Psi_k) and Q(Psi_k) for k = 0..K of the trajectory exp(eps_{K-1} D_{K-1}) ... exp(eps_0 D_0)|HF> (F7).

    ``directions[j]`` and ``eps[j]`` are the direction and step size of step j (for exact
    trajectories ``CQEStep.generator`` and ``CQEStep.eps``). With irreps every direction
    is masked before it is compiled (F2a).
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    directions = list(directions)
    eps = [_check_eps(e) for e in eps]
    if not directions or len(directions) != len(eps):
        raise ValueError(f"need one step size per direction and at least one step "
                         f"(got {len(directions)} directions, {len(eps)} step sizes)")
    norb = _check_direction(directions[0], real=False).shape[0]
    if method == "elementwise":
        steps = tuple(elementwise_step_cost(d, e, irreps=irreps, params=params, step=j)
                      for j, (d, e) in enumerate(zip(directions, eps)))
    else:
        steps = tuple(leaf_step_cost(d, e, irreps=irreps, params=params, step=j)
                      for j, (d, e) in enumerate(zip(directions, eps)))
    cumulative = np.concatenate([[0], np.cumsum([s.toffolis for s in steps])])
    qubits = np.maximum.accumulate([2 * norb] + [s.logical_qubits for s in steps])
    final = 0
    if method == "leaves":
        rotation = orbital_rotation_counts(norb, complex_=True)
        b_f = phase_gradient_bits(rotation.n_add, params.delta_step, params.b_min)
        final = rotation.toffolis(b_f)
        applied = np.concatenate([[0], np.cumsum([s.n_leaves for s in steps])]) > 0
        cumulative = cumulative + final * applied
        qubits = np.where(applied, np.maximum(qubits, 2 * norb + 3 * b_f + 8), qubits)
    return PrepCost(method=method, steps=steps, cumulative=_readonly(cumulative.astype(np.int64)),
                    logical_qubits=_readonly(qubits.astype(np.int64)), final_rotation=int(final))


# ------------------------------------------------------------------------- measurements
def candidate_step_toffolis(d: np.ndarray, eps_candidates: Sequence[float], *,
                            irreps: np.ndarray | None = None,
                            params: FTParameters = FTParameters()) -> np.ndarray:
    """(n_c,) int64 elementwise C_step of exp(eps_c D) for every line-search candidate (F13).

    Equal to ``elementwise_step_cost(d, eps_c, ...).toffolis`` for each candidate; d is
    expanded once.
    """
    exp = _expand(d, irreps)
    return np.array([_elementwise_cost(exp, _check_eps(e), params, 0).toffolis for e in eps_candidates],
                    dtype=np.int64)


def readout_rotation_toffolis(norb: int, *, params: FTParameters = FTParameters()) -> int:
    """Toffolis of the real DF readout rotation: g (b_r - 2) + 6 g, b_r = F2(g, delta_readout) (F4).

    g = N(N-1)/2 Givens rotations per spin, both spins sharing each addition; 405 at N = 6.
    """
    rotation = orbital_rotation_counts(norb, complex_=False)
    return rotation.toffolis(phase_gradient_bits(rotation.n_add, params.delta_readout, params.b_min))
