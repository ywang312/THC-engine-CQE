"""Validation tools for the QPE layer: an explicit Pauli-LCU qubitized walk and the QPE outcome law.

Tags and formula numbers are those of :mod:`thc_engine_cqe.ft.primitives`. Energies in Hartree.

The walk (:class:`PauliLCUWalk`) [cited: Babbush et al., PRX 8, 041015 (2018); Low and
Chuang, Quantum 3, 163 (2019)]
--------------------------------------------------------------------------------------
H = c_0 + sum_{l=1}^{L} c_l P_l is the Jordan-Wigner Pauli form of the active-space
Hamiltonian (``measure.pauli.jw_hamiltonian``, blocked qubit order by default), with real c_l,
Hermitian Pauli strings P_l = P(x_l, z_l) and the identity coefficient c_0 = Tr H / 2^{2N}
split off. With lambda = sum_l |c_l| (the Pauli 1-norm, ``factor.pauli_lambda``),

    |G> = sum_l sqrt(|c_l| / lambda) |l>,        SELECT = sum_l |l><l| (x) sign(c_l) P_l,
    R   = 2 |G><G| (x) 1 - 1,                     W = R SELECT,

and (<G| (x) 1) SELECT (|G> (x) 1) = (H - c_0)/lambda. SELECT is Hermitian with
SELECT^2 = 1, so for every eigenvector psi_j of H with x_j = (E_j - c_0)/lambda = cos theta_j
the plane span{|G, psi_j>, W |G, psi_j>} is invariant, W rotates it by theta_j, and
|G, psi_j> = (|w_j^+> + |w_j^->)/sqrt 2 with W |w_j^+-> = exp(+-i theta_j) |w_j^+->. The
ancilla is emulated as an index: states are (L, 2^{2N}) arrays phi[l, b], and
P(x, z)|b> = i^{|x & z|} (-1)^{|b & z|} |b ^ x> (``measure.pauli``). Two consequences are
checked in tests/test_ft_walk.py against the exact spectrum of H in the (n_alpha, n_beta)
sector, for p_j = |<psi_j|psi>|^2:

* Chebyshev moments (:meth:`PauliLCUWalk.moments`):
  <G, psi| W^n |G, psi> = sum_j p_j cos(n theta_j) = sum_j p_j T_n(x_j).
* The Laurent polynomial of F8 (:meth:`PauliLCUWalk.laurent_evolution`): with
  exp(-i tau cos theta) = sum_k (-i)^k J_k(tau) exp(i k theta) (Jacobi-Anger), the same value
  on both rotation eigenvectors of a plane, the truncation to |k| <= R gives

      sum_{k=-R}^{R} (-i)^k J_k(tau) W^k |G, psi> = |G> (x) exp(-i tau (H - c_0)/lambda) |psi> + e,
      ||e|| <= 2 sum_{k>R} |J_k(tau)| ||psi||.

  J_{-k} = (-1)^k J_k makes the coefficient of W^{-k} equal to that of W^k, (-i)^k J_k(tau),
  and W^k + W^{-k} acts as 2 cos(k theta_j) on each plane, so the truncated sum stays exactly
  in |G> (x) (system): e lies in the system register. With R from
  ``primitives.jacobi_anger_degree`` the sum costs R applications of W and R of
  W^dagger = SELECT R, the 2R walk calls of F8. This validates the polynomial that GQSP
  implements, not GQSP's phase finding.

Observed (one thread; H2 and H4 in tests/test_ft_walk.py): H2 at 0.74 A (14 terms on 4 qubits,
lambda = 1.887), H4 (184 terms on 8 qubits, lambda = 7.375) and H6 at 1.0 A (918 terms on
12 qubits, lambda = 17.647); lambda equals ``factor.pauli_lambda`` and c_0 the Fock-space
Tr H / 2^{2N} to 1e-12. The moments for n <= 30 on HF and the third L-BFGS iterate agree
with the spectral sums to <= 1.2e-14 (0.04 s on H4, 6 s on H6). On H4 the Laurent error
is 0.35-0.94 of the tail bound at delta = 1e-4 (R = 4, 9, 15 for tau = 0.5, 3, 7) and
0.51-0.74 at 1e-8, and the amplitude outside |G> is <= 4e-16 (round-off).

QPE outcome law (:func:`qpe_outcome_law`, :func:`window_probability`) [derived]
-------------------------------------------------------------------------------
Textbook phase estimation with t control qubits on W, started from |G> (x) |Psi> with
p_j = |<psi_j|Psi>|^2, returns m in {0, ..., 2^t - 1} with probability

    P(m) = sum_j p_j [F_t(m - 2^t theta_j / 2 pi) + F_t(m + 2^t theta_j / 2 pi)] / 2,
    F_t(y) = sin^2(pi y) / (4^t sin^2(pi y / 2^t))               (Fejer kernel, period 2^t),

and the energy estimate E(m) = c + lambda cos(2 pi m / 2^t). Each F_t sums to 1 over m, so
P sums to sum_j p_j. When the gap E_1 - E_0 is large against the window, the probability
of |E(m) - E_0| <= dE is w p_0 with the window factor w set by where theta_0 falls on the
grid, the same for every state [design]; :func:`window_probability` evaluates it.
Observed on H4 (the setup of the design notes: DF H' at threshold 1e-3, lambda_DF = 4.936,
c = Tr H'/2^{2N}, t = 15, dE = 1 mHa): the ratio is 0.9861933585 at HF and 0.9861933486
from k = 4 on along the L-BFGS iterates, a spread of 9.96e-9, the digits of the design notes
(tests/test_ft_walk.py::test_window_probability_over_p_is_k_independent). The
k-dependence is the excited-state tails, ratio_k - w = sum_{j>=1} p_kj tau_j / p_k0
with single-state window probabilities tau_j <= 3.5e-6 (tested to 1e-12).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import scipy.special

from ..integrals import ActiveSpace
from ..measure.pauli import QubitMapping, QubitState, jw_hamiltonian
from .primitives import jacobi_anger_degree

__all__ = ["PauliLCUWalk", "qpe_outcome_law", "window_probability"]

_IPOW = np.array([1.0, 1.0j, -1.0, -1.0j])
# Largest number of QPE control qubits of qpe_outcome_law (memory about 57 * 2^t bytes).
_MAX_T_BITS = 24


class PauliLCUWalk:
    """Matrix-free qubitized walk W = R SELECT of the JW Pauli LCU of H (module docstring).

    Attributes:
        lam: lambda = sum_l |c_l| (Hartree), identity excluded.
        c0: identity coefficient c_0 = Tr H / 2^{2N} (Hartree, with ecore).
        n_terms: L, the number of non-identity Pauli strings.
        n_qubits: 2N system qubits.
        mapping: the ``measure.pauli.QubitMapping`` of the system qubits.

    States are complex arrays of shape (n_terms, 2^n_qubits); the memory of the
    precomputed SELECT tables is 24 n_terms 2^n_qubits bytes (H4: 184 terms on 8 qubits,
    1.1 MB; H6: 918 terms on 12 qubits, 90 MB).
    """

    def __init__(self, active: ActiveSpace, mapping: QubitMapping | None = None):
        self.mapping = QubitMapping(active.norb) if mapping is None else mapping
        op = jw_hamiltonian(active, self.mapping)
        xs, zs, cs = op.xs, op.zs, op.coeffs.real
        ident = (xs == 0) & (zs == 0)
        self.c0 = float(cs[ident].sum())
        xs, zs, cs = xs[~ident], zs[~ident], cs[~ident]
        self.n_terms = int(cs.size)
        self.n_qubits = int(self.mapping.n_qubits)
        if self.n_terms == 0:
            raise ValueError("the Hamiltonian is a multiple of the identity; there is no walk")
        self.lam = float(np.sum(np.abs(cs)))
        self._g = np.sqrt(np.abs(cs) / self.lam)
        b = np.arange(1 << self.n_qubits, dtype=np.uint64)
        # SELECT|l>|b> = s_l i^{|x&z|} (-1)^{|b&z|} |l>|b^x>; b -> b^x is an involution.
        self._dest = (b[None, :] ^ xs[:, None]).astype(np.intp)
        const = np.sign(cs) * _IPOW[np.bitwise_count(xs & zs) & 3]
        self._phase = const[:, None] * (1.0 - 2.0 * (np.bitwise_count(b[None, :] & zs[:, None]) & 1))

    # ---------------------------------------------------------------- operators
    def _check(self, phi: np.ndarray) -> np.ndarray:
        phi = np.asarray(phi)
        if phi.shape != (self.n_terms, 1 << self.n_qubits):
            raise ValueError(f"walk states have shape ({self.n_terms}, {1 << self.n_qubits}), "
                             f"got {phi.shape}")
        return phi

    def _select(self, phi: np.ndarray) -> np.ndarray:
        return np.take_along_axis(self._phase * phi, self._dest, axis=1)

    def _reflect(self, phi: np.ndarray) -> np.ndarray:
        return 2.0 * np.outer(self._g, self._g @ phi) - phi

    def apply(self, phi: np.ndarray) -> np.ndarray:
        """W phi = R SELECT phi for an (n_terms, 2^n_qubits) array."""
        return self._reflect(self._select(self._check(phi)))

    def apply_dagger(self, phi: np.ndarray) -> np.ndarray:
        """W^dagger phi = SELECT R phi (SELECT and R are Hermitian)."""
        return self._select(self._reflect(self._check(phi)))

    def prepare(self, psi_qubit: np.ndarray | QubitState) -> np.ndarray:
        """|G> (x) |psi> for a dense system state of length 2^n_qubits (or a ``QubitState``)."""
        psi = np.asarray(psi_qubit.to_dense() if hasattr(psi_qubit, "to_dense") else psi_qubit)
        psi = psi.astype(np.complex128).ravel()
        if psi.size != 1 << self.n_qubits:
            raise ValueError(f"system states have 2^{self.n_qubits} entries, got {psi.size}")
        return np.outer(self._g, psi)

    # ------------------------------------------------------------ observables
    def moments(self, psi_qubit: np.ndarray | QubitState, nmax: int) -> np.ndarray:
        """<G, psi| W^n |G, psi> for n = 0..nmax (complex array of nmax + 1 entries)."""
        phi0 = self.prepare(psi_qubit)
        phi = phi0
        out = np.empty(int(nmax) + 1, dtype=np.complex128)
        for n in range(int(nmax) + 1):
            out[n] = np.vdot(phi0, phi)
            if n < nmax:
                phi = self.apply(phi)
        return out

    def laurent_evolution(self, psi_qubit: np.ndarray | QubitState, tau: float, delta: float) -> np.ndarray:
        """sum_{k=-R}^{R} (-i)^k J_k(tau) W^k |G>|psi> with R = R(tau, delta) of F8 (2R walk applications).

        Approximates |G> (x) exp(-i tau (H - c0)/lam)|psi> within 2 sum_{k>R} |J_k(tau)| <= delta
        (times ||psi||); ``tau`` is dimensionless (tau = lam t).
        """
        R = jacobi_anger_degree(tau, delta)
        J = scipy.special.jv(np.arange(R + 1), float(tau))
        phi0 = self.prepare(psi_qubit)
        out = J[0] * phi0
        fwd = bwd = phi0
        for k in range(1, R + 1):
            fwd, bwd = self.apply(fwd), self.apply_dagger(bwd)
            out = out + ((-1j) ** k * J[k]) * (fwd + bwd)
        return out


def _fejer_row(M: int, y: float) -> np.ndarray:
    """F_t(m - y) for m = 0..M-1, M = 2^t, F_t(u) = sin^2(pi u) / (M^2 sin^2(pi u / M)).

    The shift is split into the nearest integer n, reduced modulo the period M in integer
    arithmetic, and the fraction f = y - n in [-1/2, 1/2], so sin^2(pi u) = sin^2(pi f) keeps
    full relative accuracy: reducing m - y in floating point loses ~ M eps in the tails, and a
    fraction near 1 would lose ~ eps / (1 - f) at the peak of a phase just below the grid.
    """
    n = round(y)
    f = y - n
    k = np.mod(np.arange(M, dtype=np.int64) - n + M // 2, M) - M // 2
    den = (M * np.sin(np.pi * (k - f) / M)) ** 2
    out = np.ones(M)            # the limit 1 at u = 0 (f = 0 and k = 0, the only zero of den)
    nz = den != 0.0
    out[nz] = math.sin(math.pi * f) ** 2 / den[nz]
    return out


def qpe_outcome_law(energies: Sequence[float] | np.ndarray, weights: Sequence[float] | np.ndarray, lam: float,
                    c: float, t_bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Outcome law of textbook QPE with ``t_bits`` control qubits on a qubitized walk: (P(m), E(m)).

    Args:
        energies: eigenvalues E_j of H' in the sector of the initial state (Hartree), finite.
        weights: p_j = |<psi_j|Psi>|^2, finite and >= 0 (not renormalized; P sums to their sum).
        lam, c: the walk block-encodes (H' - c)/lam; |E_j - c| <= lam is required.
        t_bits: number of control qubits t, an integer in 1..24; m runs over 0..2^t - 1.

    Returns P(m) of the module docstring (Fejer kernels at +-2^t theta_j / 2 pi) and
    E(m) = c + lam cos(2 pi m / 2^t), both arrays of length 2^t. Cost: peak memory about
    57 bytes per outcome (measured with tracemalloc: 57 MiB at t = 20, 228 MiB at t = 22,
    so about 1 GB at the cap t = 24) and about 0.07 s per eigenstate with nonzero weight
    at t = 20, growing as 2^t (one thread).
    """
    e = np.asarray(energies, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    if e.shape != w.shape:
        raise ValueError(f"energies and weights differ in length ({e.size}, {w.size})")
    if not (np.all(np.isfinite(e)) and np.all(np.isfinite(w))):
        raise ValueError("energies and weights must be finite")
    if np.any(w < 0.0):
        raise ValueError(f"weights must be >= 0, got min {float(w.min()):g}")
    if not (math.isfinite(lam) and lam > 0.0 and math.isfinite(c)):
        raise ValueError(f"lam must be positive and finite and c finite, got lam = {lam!r}, c = {c!r}")
    t = int(t_bits)
    if t != t_bits or not 1 <= t <= _MAX_T_BITS:
        raise ValueError(f"t_bits must be an integer in 1..{_MAX_T_BITS}, got {t_bits!r}")
    x = (e - c) / lam
    if x.size and float(np.max(np.abs(x))) > 1.0 + 1e-12:
        raise ValueError(f"lam = {lam:g} does not bound |E - c| (max {float(np.max(np.abs(e - c))):g})")
    M = 1 << t
    p = np.zeros(M)
    for wj, yj in zip(w, np.arccos(np.clip(x, -1.0, 1.0)) * M / (2.0 * np.pi)):
        if wj != 0.0:
            p += (0.5 * wj) * (_fejer_row(M, float(yj)) + _fejer_row(M, -float(yj)))
    return p, c + lam * np.cos(2.0 * np.pi * np.arange(M) / M)


def window_probability(law: tuple[np.ndarray, np.ndarray], e0: float, de: float) -> float:
    """P(|E(m) - e0| <= de) for ``law`` = (P(m), E(m)) of :func:`qpe_outcome_law`; ``de`` >= 0 (Hartree)."""
    if not de >= 0.0:
        raise ValueError(f"de must be >= 0, got {de!r}")
    p, e = law
    return float(np.sum(np.asarray(p)[np.abs(np.asarray(e) - e0) <= de]))
