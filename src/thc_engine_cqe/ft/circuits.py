"""Validation tools for the state-preparation cost model: explicit phase-gradient circuits and replays.

:mod:`thc_engine_cqe.ft.stateprep` counts Toffolis for circuits it never builds. The tools
here build the pieces explicitly on small registers, so that tests/test_ft_circuits.py
can check the counting assumptions against independent references. Qubit layout and
basis order are those of ``diagnostics.cost`` and ``testing.fock``: Jordan-Wigner with
the blocked spin order (spin orbital (p, alpha) is qubit p, (p, beta) is qubit N + p),
basis index bit k = qubit k. Tags as in the M3 specification ([derived], [cited],
[OF]); F1-F19 are its formula tags.

Phase-gradient excitations (:func:`pg_excitation_unitary`, F1, F3) [derived]
--------------------------------------------------------------------------
``diagnostics.cost.excitation_circuit`` compiles exp(t tau - conj(t) tau^dag) as a
Clifford prefix P (the Jordan-Wigner parity ladder, a CZ that carries its sign, and the
CNOT basis map), a multi-controlled R_y(phi) on a target qubit t with
phi = -2 |t| sigma (sigma = <C|tau|D> = +-1), and P^dag, all between the phase gates
p(-arg t) and p(arg t) on one qubit; its unitary is verified against the Fock-space
oracle in tests/test_generator_rank.py. With R_y(phi) = S H R_z(phi) H S^dag,

    U = p(arg t) P^dag (S H)_t  D  (H S^dag)_t P p(-arg t),
    D = exp(+i phi/2) on the cell c AND t,  exp(-i phi/2) on c AND NOT t,  1 elsewhere,

where c is the control indicator: (a, b, c) = (0, 1, 0) of the double's qubits
(2 ANDs), x AND control for a number-conditioned single (1 AND). The cell c AND t is one
more AND, c AND NOT t = c XOR (c AND t) a Clifford, and D is one phase-gradient addition
of the two cell constants (F1). A diagonal term exp(i phi n_a n_b) is D alone, on the
cell a AND b (1 AND). This gives the per-excitation counts of F3 (3 + 1, 2 + 1 and
1 + 1 ANDs + additions). With ``bits`` = b every cell constant is rounded to a multiple
of 2 pi / 2^b (``primitives.round_angle``); because +-phi/2 round symmetrically, the
rounded circuit is exactly the excitation with the rounded Givens angle, within
pi / 2^b of the exact one in operator norm. The phase gates p(-+arg t) are
phase-gradient additions too (one cell each); for real t they are 1 or Z, exact at any
b, and a complex t adds 2 pi / 2^b to the bound (F3 excludes complex directions). Every
factor is diagonal or a basis map of the excitation's own two-dimensional subspaces, so
N and S_z are conserved exactly, rounded or not.

Phase kickback (:func:`kickback_check`, F1) [derived]
-----------------------------------------------------
The gradient state |Phi_b> = 2^{-b/2} sum_x exp(-2 pi i x / 2^b) |x> satisfies
|x + v mod 2^b> summed against it = exp(+2 pi i v / 2^b) |Phi_b>. For mutually exclusive
indicator cells I_c on the system with constants v_c = round_angle(phi_c, b), the
circuit "load v(x) = sum_c v_c I_c(x) into a summand register by CNOT fan-out, add the
summand into the gradient register mod 2^b, unload" maps
|x>|0>|Phi_b> -> exp(2 pi i v(x) / 2^b) |x>|0>|Phi_b>. :func:`kickback_check` simulates
it with explicit summand and gradient registers (the indicators are classical
functions of the system basis state; the AND ancillas and the adder's internals are not
simulated, the adder cost b - 2 is [OF]).

Givens decomposition (:func:`givens_decomposition`, F4) [cited]
---------------------------------------------------------------
Any N x N unitary V is a product of N(N-1)/2 Givens rotations of adjacent rows and a
diagonal of phases [Clements et al., Optica 3, 1460 (2016); Kivlichan et al., PRL 120,
110501 (2018); this is the column-by-column QR elimination]:

    V = G(p_1, p_1 + 1, theta_1, phi_1) ... G(p_K, p_K + 1, theta_K, phi_K) diag(phases),
    G(p, q, theta, phi) = identity except [[cos theta, -e^{i phi} sin theta],
                                           [e^{-i phi} sin theta, cos theta]] on (p, q).

A real orthogonal V gives phi = 0 and phases +-1, i.e. real rotations only. This is the
rotation count g of F4 (per spin).

Compiled elementwise replay (:func:`replay_elementwise`, F3)
-----------------------------------------------------------
The Fock-space oracle ``testing.fock.FockSpace`` (no PySCF CI machinery) prepares HF
and applies each step as the product of its kept excitations
(``stateprep.elementwise_excitations``: masked, rounding fixed point, decreasing angle).
Each excitation exp(theta K), K = (t tau - conj(t) tau^dag)/|t|, is applied exactly
through K^3 = -K:

    v -> v + sin(theta) K v + (1 - cos theta) K (K v),

with theta = |eps t| (``rounded=False``) or theta rounded to b bits (``rounded=True``,
the phase-gradient circuit above). The replay is open loop: the trajectory's directions
are applied to the compiled states, not recomputed. It measures the product-formula
error of the compiled steps, which F3 does not budget, and the rounding error, which it
does: ||Psi~_rounded - Psi~_unrounded|| <= sum_j A_j pi / 2^{b_j}. Every factor is unitary on
the Fock space and conserves N and S_z, so ||Psi~_k|| = 1 up to round-off, and a
deficit would be leakage out of the (n_alpha, n_beta) sector. Directions are real
(F3), so the kept excitations are doubles and singles with real amplitudes. The oracle
is dense in 4^N and refuses N > 7, so H8 cannot be replayed.

Observed (exact L-BFGS trajectories on the symmetry-adapted builds, rounded circuits,
open loop; tests/test_ft_circuits.py): ||Psi~_1 - Psi_1|| after the RHF step is 1.1e-3
(H4), 1.4e-3 (H6 at 1.0 A), 2.0e-3 (N2), 2.2e-4 (LiH), mostly product-formula error
(rounding contributes <= 2.1e-4), and stays below 3.2e-3 up to k* + 2; the compiled
circuits reach 1.6 mHa at the exact k* (3, 6, 3, 2). The stretched builds have a
product-formula error 20-30 times larger at the RHF step, 2.7e-2 (H6 at 3 bohr) and
3.1e-2 (N2 at 2.0 A), and on N2 at 2.0 A the compiled first step reaches the overlap
0.2251 instead of 0.2314.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..diagnostics.cost import Excitation, circuit_unitary
from ..fci_ops import FCISpace
from ..integrals import ActiveSpace
from ..testing.fock import FockSpace
from .primitives import FTParameters, round_angle
from .stateprep import elementwise_excitations

__all__ = [
    "pg_excitation_unitary",
    "kickback_check",
    "givens_decomposition",
    "givens_reconstruct",
    "replay_elementwise",
]

_H = np.array([[1.0, 1.0], [1.0, -1.0]]) / math.sqrt(2.0)
_S = np.diag([1.0, 1.0j])
_KICKBACK_MAX_QUBITS = 12


# ------------------------------------------------------------------ excitation circuits
def _cell_phase(phi: float, bits: int | None) -> float:
    """The cell constant phi (radians), rounded to a multiple of 2 pi / 2^bits unless bits is None."""
    if bits is None:
        return float(phi)
    return 2.0 * math.pi * round_angle(phi, bits) / 2.0 ** bits


def _apply_1q(u: np.ndarray, q: int, m: np.ndarray) -> np.ndarray:
    """(m on qubit q) @ u, for a dense u in the basis with bit k = qubit k."""
    idx = np.arange(u.shape[0])
    lo = idx[((idx >> q) & 1) == 0]
    hi = lo | (1 << q)
    out = np.array(u, dtype=complex)
    out[lo] = m[0, 0] * u[lo] + m[0, 1] * u[hi]
    out[hi] = m[1, 0] * u[lo] + m[1, 1] * u[hi]
    return out


def _jw_sign(exc: Excitation) -> int:
    """sigma = <C| tau |D> for tau = a+_{c0} a+_{c1} a_{d1} a_{d0}, with only the modes C (D) occupied.

    a_j and a+_j carry (-1)^(number of occupied modes below j) (Jordan-Wigner, as in testing.fock).
    """
    occ = set(exc.annihilators)
    sign = 1
    (c0, c1), (d0, d1) = exc.creators, exc.annihilators
    for kind, j in (("a", d0), ("a", d1), ("c", c1), ("c", c0)):      # a_{d0} acts first
        sign *= -1 if sum(1 for m in occ if m < j) % 2 else 1
        if kind == "a":
            occ.remove(j)
        else:
            occ.add(j)
    return sign


def pg_excitation_unitary(exc: Excitation, n_qubits: int, bits: int | None) -> np.ndarray:
    """Dense unitary of the phase-gradient compilation of one excitation (module docstring).

    Args:
        exc: a ``diagnostics.cost.Excitation`` (double, single or diagonal).
        n_qubits: register size; the modes of ``exc`` must lie below it.
        bits: phase-gradient bits b of the cell constants; None leaves them unrounded.

    Returns:
        (2^n, 2^n) complex unitary. Unrounded it equals
        ``diagnostics.cost.circuit_unitary(excitation_circuit(exc), n_qubits)``; rounded it
        is within pi / 2^b of it for real amplitudes (3 pi / 2^b for complex ones).
    """
    n = int(n_qubits)
    if n != n_qubits or n < 1 or min(exc.modes) < 0 or max(exc.modes) >= n:
        raise ValueError(f"excitation on modes {exc.modes} does not fit on {n_qubits!r} qubits")
    if bits is not None and (int(bits) != bits or bits < 1):
        raise ValueError(f"bits must be a positive integer or None, got {bits!r}")
    idx = np.arange(1 << n)

    def occ(q: int) -> np.ndarray:
        return ((idx >> q) & 1).astype(bool)

    t = complex(exc.amplitude)
    if exc.kind == "diagonal":
        a, b = exc.creators
        return np.diag(np.exp(1j * np.where(occ(a) & occ(b), _cell_phase(t.imag, bits), 0.0)))
    if exc.kind == "double":
        (a, b), (c, d) = exc.creators, exc.annihilators
        clifford = [("cx", d, c), ("cx", b, a), ("cx", d, b)]
        control = ~occ(a) & occ(b) & ~occ(c)
        target, phase_qubit = d, a
        lo1, hi1, lo2, hi2 = exc.modes
        string = list(range(lo1 + 1, hi1)) + list(range(lo2 + 1, hi2))
    elif exc.kind == "single":
        x, y, ctl = exc.particle, exc.hole, exc.control
        clifford = [("cx", y, x)]
        control = occ(x) & occ(ctl)
        target, phase_qubit = y, x
        string = [j for j in range(min(x, y) + 1, max(x, y)) if j != ctl]
    else:
        raise ValueError(f"unknown excitation kind {exc.kind!r}")
    if string:
        pq = string[0]
        clifford = [("cx", j, pq) for j in string[1:]] + [("cz", pq, target)] + clifford
    half = -abs(t) * _jw_sign(exc)                 # phi / 2 of the reference R_y(phi)
    beta = float(np.angle(t))
    cells = np.zeros(1 << n)
    cells[control & occ(target)] = _cell_phase(half, bits)
    cells[control & ~occ(target)] = _cell_phase(-half, bits)
    phase_on = occ(phase_qubit)
    u = np.diag(np.exp(1j * np.where(phase_on, _cell_phase(-beta, bits), 0.0)))
    u = circuit_unitary(clifford, n) @ u
    u = _apply_1q(u, target, _H @ _S.conj())
    u = np.exp(1j * cells)[:, None] * u
    u = _apply_1q(u, target, _S @ _H)
    u = circuit_unitary(clifford[::-1], n) @ u
    return np.exp(1j * np.where(phase_on, _cell_phase(beta, bits), 0.0))[:, None] * u


# ------------------------------------------------------------------------ phase kickback
def kickback_check(cells: Sequence[tuple[np.ndarray, float]], n_sys: int, bits: int) -> float:
    """Simulate a phase-gradient addition with explicit summand and gradient registers (F1).

    Args:
        cells: (indicator, phi_c) pairs; indicator is a (2^n_sys,) boolean array over the
            system basis states, the cells must be mutually exclusive, phi_c in radians.
        n_sys: system qubits; n_sys + 2 bits <= 12.
        bits: b, the size of both registers.

    Returns:
        max over system basis states x of || out(x) - exp(2 pi i v(x) / 2^b) |0>|Phi_b> ||,
        with out(x) the register state after load, add and unload from |0>|Phi_b>, and
        v(x) = sum_c round_angle(phi_c, b) I_c(x): the operator-norm error on the subspace
        of clean registers (round-off level when kickback works as F1 states).
    """
    n, b = int(n_sys), int(bits)
    if n != n_sys or b != bits or n < 1 or b < 1 or n + 2 * b > _KICKBACK_MAX_QUBITS:
        raise ValueError(f"need positive integers with n_sys + 2 bits <= {_KICKBACK_MAX_QUBITS}, "
                         f"got n_sys={n_sys!r}, bits={bits!r}")
    dim, m = 1 << n, 1 << b
    loads = []
    covered = np.zeros(dim, dtype=int)
    for indicator, phi in cells:
        ind = np.asarray(indicator, dtype=bool)
        if ind.shape != (dim,):
            raise ValueError(f"indicators must have shape ({dim},), got {ind.shape}")
        covered += ind
        loads.append((ind, round_angle(phi, b)))
    if covered.max(initial=0) > 1:
        raise ValueError("the cells are not mutually exclusive")
    grad = np.exp(-2j * np.pi * np.arange(m) / m) / math.sqrt(m)         # |Phi_b>
    state = np.zeros((dim, m, m), dtype=complex)                           # [system, summand, gradient]
    state[:, 0, :] = grad

    def fan_out(psi: np.ndarray) -> np.ndarray:
        """CNOTs from each cell indicator into the summand bits set in its constant."""
        for ind, v in loads:
            for j in range(b):
                if (v >> j) & 1:
                    psi[ind] = psi[ind][:, np.arange(m) ^ (1 << j), :]
        return psi

    state = fan_out(state)
    added = np.empty_like(state)
    for s in range(m):                                                     # gradient += summand
        added[:, s, :] = np.roll(state[:, s, :], s, axis=1)
    state = fan_out(added)
    v_of_x = np.zeros(dim, dtype=np.int64)
    for ind, v in loads:
        v_of_x[ind] += v
    expected = np.zeros_like(state)
    expected[:, 0, :] = np.exp(2j * np.pi * v_of_x / m)[:, None] * grad[None, :]
    return float(np.max(np.linalg.norm((state - expected).reshape(dim, -1), axis=1)))


# ----------------------------------------------------------------- Givens decomposition
def givens_decomposition(v: np.ndarray) -> tuple[list[tuple[int, int, float, float]], np.ndarray]:
    """N(N-1)/2 adjacent Givens rotations and diagonal phases of a unitary V (module docstring).

    Returns:
        (rotations, phases): rotations (p, p + 1, theta, phi) in product order and the
        (N,) diagonal of the phase matrix, with
        V = G(rotations[0]) ... G(rotations[-1]) diag(phases). For real V every phi is 0
        and the phases are real (+-1).
    """
    a = np.asarray(v)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
        raise ValueError(f"expected a square matrix, got shape {a.shape}")
    n = a.shape[0]
    if np.linalg.norm(a.conj().T @ a - np.eye(n)) > 1e-10:
        raise ValueError("v is not unitary")
    real = not np.iscomplexobj(a)
    w = np.array(a, dtype=float if real else complex)
    rotations: list[tuple[int, int, float, float]] = []
    for j in range(n - 1):
        for q in range(n - 1, j, -1):                 # zero w[q, j] with rows (q - 1, q)
            p = q - 1
            x, y = w[p, j], w[q, j]
            if real:
                theta, phi = math.atan2(y, x), 0.0
            else:
                theta = math.atan2(abs(y), abs(x))
                phi = float(np.angle(x) - np.angle(y))
            c, s = math.cos(theta), math.sin(theta)
            e = 1.0 if real else np.exp(1j * phi)
            rp, rq = w[p].copy(), w[q].copy()
            w[p] = c * rp + e * s * rq                # G^dag on rows (p, q)
            w[q] = -np.conj(e) * s * rp + c * rq
            rotations.append((p, q, theta, phi))
    return rotations, np.diag(w).copy()


def givens_reconstruct(rotations: Sequence[tuple[int, int, float, float]], phases: np.ndarray) -> np.ndarray:
    """V = G(rotations[0]) ... G(rotations[-1]) diag(phases) (:func:`givens_decomposition`)."""
    d = np.asarray(phases)
    real = not np.iscomplexobj(d) and all(phi == 0.0 for *_, phi in rotations)
    out = np.diag(d).astype(float if real else complex)
    for p, q, theta, phi in reversed(rotations):
        c, s = math.cos(theta), math.sin(theta)
        e = 1.0 if real else np.exp(1j * phi)
        rp, rq = out[p].copy(), out[q].copy()
        out[p] = c * rp - e * s * rq
        out[q] = np.conj(e) * s * rp + c * rq
    return out


# --------------------------------------------------------------------- compiled replay
def replay_elementwise(active: ActiveSpace, directions: Sequence[np.ndarray], eps: Sequence[float], *,
                       irreps: np.ndarray | None = None, rounded: bool = True,
                       params: FTParameters = FTParameters()) -> list[np.ndarray]:
    """CI vectors Psi~_0 .. Psi~_K of the compiled elementwise circuits, from HF (module docstring).

    Args:
        active: the system (N <= 7 orbitals: the Fock-space oracle is dense in 4^N).
        directions, eps: real directions and step sizes of the K steps.
        irreps: orbital irreps for the mask (F2a); None: no mask.
        rounded: rotate by the b-bit rounded angles (the compiled circuit) or by the exact ones.
        params: delta_step and b_min (the rounding fixed point of F3).

    Returns:
        K + 1 real CI arrays in PySCF's determinant basis, Psi~_0 = HF.
    """
    directions = list(directions)
    eps = list(eps)
    if len(directions) != len(eps):
        raise ValueError(f"got {len(directions)} directions and {len(eps)} step sizes")
    n, nelec = active.norb, tuple(active.nelec)
    space = FCISpace.of(active)
    fock = FockSpace(n)
    ann = [fock.a(mode % n, mode // n) for mode in range(2 * n)]
    cre = [a.T.tocsr() for a in ann]
    hf = space.hf()
    v = fock.from_ci(hf, nelec)
    states = [hf]
    for d, e in zip(directions, eps):
        kept, b = elementwise_excitations(d, e, irreps=irreps, params=params)
        for x in kept:
            theta = abs(x.amplitude)
            if rounded:
                theta = 2.0 * math.pi * round_angle(theta, b) / 2.0 ** b
            kv = _k_apply(cre, ann, x, v)
            v = v + math.sin(theta) * kv + (1.0 - math.cos(theta)) * _k_apply(cre, ann, x, kv)
        states.append(fock.to_ci(v, nelec))
    return states


def _k_apply(cre: list, ann: list, exc: Excitation, u: np.ndarray) -> np.ndarray:
    """K u = sign(t) (tau - tau^dag) u for a double or single with real amplitude t (sparse JW operators)."""
    (c0, c1), (d0, d1) = exc.creators, exc.annihilators
    tau = cre[c0] @ (cre[c1] @ (ann[d1] @ (ann[d0] @ u)))
    tau_dag = cre[d0] @ (cre[d1] @ (ann[c1] @ (ann[c0] @ u)))
    return math.copysign(1.0, exc.amplitude) * (tau - tau_dag)
