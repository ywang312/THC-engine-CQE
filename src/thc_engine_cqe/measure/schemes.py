"""Measurement schemes and their exact per-shot variances, with Monte Carlo validators.

Four schemes estimate the expectation values of Hermitian targets on a qubit
state: qubit-wise-commuting (QWC) Pauli grouping, double-factorized (DF) basis
rotations (energy only), Pauli (random local Clifford) shadows and matchgate
(fermionic Gaussian) shadows. For each scheme this module computes the exact
per-shot variance of the standard unbiased estimator. For targets too large for
the exact pair sums it computes a Rao-Blackwellized Monte Carlo estimate, and it
provides snapshot Monte Carlo validators that simulate the actual measurements.
Conventions (Pauli strings P(x, z), Jordan-Wigner, Majoranas) are those of
:mod:`thc_engine_cqe.measure.pauli`.

Targets, shots and VarianceResult
---------------------------------
A target is a Hermitian :class:`~.pauli.PauliSum` O = sum_j c_j P_j, or a list of
them (a vector target) O_e = sum_j W_ej P_j, with W the real coefficient matrix
of :func:`~.pauli.coefficient_matrix`. Post-processing weights (e.g. -1/delta of
the auxiliary route) are folded into the targets by the caller. The state is a
normalized :class:`~.pauli.QubitState`; complex amplitudes are allowed.

Every scheme defines a single-shot estimator x_e (one measurement setting, one
outcome) with E[x_e] = <O_e>. :class:`VarianceResult` reports per-shot
variances: ``per_target[e]`` = Var(x_e) under the stated shot allocation and
``total`` = sum_e Var(x_e), so the average of S shots has total variance
total / S (for a scalar target ``total`` is the variance). Covariances are
taken in the complex Pauli basis, Cov_jk = <P_j P_k> - <P_j><P_k>, and vector
targets enter through M = W^T W. The identity term of a target is a known
constant: it costs no shots and is excluded from groups and pair sums.

QWC grouping (:func:`qwc_grouping`, :func:`qwc_variance`)
--------------------------------------------------------
Sorted insertion: the non-identity strings of the union basis are ordered by
decreasing weight w_j = sum_e |W_ej| (|c_j| for one target; ties keep the
canonical (x, z) order) and placed first-fit into groups whose members commute
qubit-wise. A group G is measured in one product basis, and each P_j in G is
read off as the parity of the outcome on supp P_j. The group estimator of O_e
is x_{G,e} = sum_{j in G} W_ej s_j, with per-shot variance

    V_{G,e} = sum_{j,k in G} W_ej W_ek Cov_jk,   V_G = sum_e V_{G,e} = sum_{j,k in G} M_jk Cov_jk.

With a fraction f_G of S shots on group G, Var(O_e estimate) = sum_G V_{G,e} / (f_G S).
The optimal allocation f_G = sqrt(V_G) / sum_H sqrt(V_H) minimizes the total,
giving ``total`` = (sum_G sqrt(V_G))^2 and ``per_target[e]`` = sum_G V_{G,e} / f_G.
Uniform allocation gives ``uniform_total`` = n_G sum_G V_G. Groups with V_G = 0
get no shots (their values are deterministic). Within a group every P_j P_k is
again a Pauli string with phase +1, so the covariances are real. Grouping
depends on the qubit order (the Jordan-Wigner strings do).

DF basis rotations (:func:`df_variance`; Huggins et al., npj Quantum Inf. 7, 23 (2021))
-------------------------------------------------------------------------------------
With (pq|rs) ~ sum_l L_l[p,q] L_l[r,s] from :func:`~thc_engine_cqe.factor.double_factorize`
and e_{pq,rs} = E_pq E_rs - delta_qr E_ps,

    H_DF - ecore = H_0 + sum_{l=1}^{L} H_l,   H_0 = sum_pq h'_pq E_pq,   h' = h - 1/2 sum_r (pr|rq),
    H_l = 1/2 F_l^2,   F_l = sum_pq L_l[p,q] E_pq = sum_k mu_lk n_k(U_l),

where n_k(U) is the spin-summed occupation of orbital k of the rotated basis
(L_l = U_l diag(mu_l) U_l^T; h' is diagonalized likewise). h' uses the factorized
ERIs, so H_DF is the model Hamiltonian H[eri_DF]: H_DF = H to round-off for the
untruncated default, and ``info["bias"]`` = <H_DF> - <H> is the truncation bias.
Each fragment is diagonal in its own determinant basis, so one setting per
fragment measures it with variance Var_0 = <H_0^2> - <H_0>^2 and
Var_l = 1/4 (<F_l^4> - <F_l^2>^2) = 1/4 (||F_l^2 psi||^2 - ||F_l psi||^4), evaluated
with ``fci_ops.one_body`` on the CI vector. The L + 1 settings are independent:
``total`` = (sum_{l=0}^{L} sqrt(Var_l))^2 (optimal allocation) and
``uniform_total`` = (L + 1) sum_l Var_l.

Pauli shadows (:func:`pauli_shadow_variance`)
--------------------------------------------
A snapshot measures every qubit in a uniformly random basis B_i in {X, Y, Z}
with outcomes s_i = +-1. The estimator of a string is
x_P = 3^{|P|} prod_{i in supp P} [B_i = P_i] s_i (|P| = weight). The product
x_P x_Q is nonzero only if the bases match P and Q on supp P u supp Q, which
requires P and Q to agree on the overlap (qubit-wise compatible) and has
probability 3^{-|supp P u supp Q|}. Given matching bases, prod s over the
symmetric difference has mean <PQ>, so

    E[x_P x_Q] = 3^{|supp P n supp Q|} <P Q>   (P, Q qubit-wise compatible; else 0),

and Var(x_e) = sum_{j,k} W_ej W_ek E[x_j x_k] - <O_e - c_e0>^2 (c_e0 the
identity coefficient). A single string gives 3^{|P|} - <P>^2 (tested). The
variance depends on the qubit order.

Matchgate shadows (:func:`matchgate_shadow_variance`; Zhao, Rubin, Miyake, PRL 127, 110504 (2021))
--------------------------------------------------------------------------------------------------
A snapshot draws a uniformly random perfect matching pi of the 2n Majoranas
(the signed-permutation matchgate ensemble; the signs only relabel outcomes) and
measures the n commuting pair operators i gamma_a gamma_b, (a, b) in pi. A
monomial gamma_S with |S| = 2k is covered (a product of measured pair operators,
up to a phase) iff S is a union of pairs of pi, which has probability

    lambda_k = (2k-1)!! (2n-2k-1)!! / (2n-1)!! = C(n, k) / C(2n, 2k),

and its estimator is x_S = lambda_k^{-1} [S covered] v_S, with v_S the measured
value of gamma_S (complex for anti-Hermitian monomials). Both S and S' are covered
iff pi splits into matchings of A = S n S', B = S \\ S', C = S' \\ S and of the rest,

    P_both(S, S') = (|A|-1)!! (|B|-1)!! (|C|-1)!! (2n - |A u B u C| - 1)!! / (2n-1)!!

(all sizes even, else 0; (-1)!! = 1), and then conj(v_S) v_S' is the measured
value of gamma_S^dagger gamma_S', so

    E[conj(x_S) x_S'] = (lambda_{|S|/2} lambda_{|S'|/2})^{-1} P_both(S, S') <gamma_S^dagger gamma_S'>.

The module evaluates this in Majorana form: P_j = phi_j gamma_{S_j}
(:func:`~.pauli.pauli_to_majorana`), complex monomial coefficients
a_ej = W_ej phi_j, gamma_S^dagger = (-1)^{|S|/2} gamma_S, gamma_S gamma_S' =
sgn gamma_{S ^ S'} (:func:`~.pauli.majorana_product`) and gamma_T = phi_T P(x_T, z_T)
(:func:`~.pauli.majorana_to_pauli`), so
Var(x_e) = sum_{S,S'} conj(a_eS) a_eS' E[conj(x_S) x_S'] / ... - |<O_e - c_e0>|^2.
P_both is nonzero iff |S n S'| is even, i.e. iff the monomials commute. A single
monomial gives lambda_k^{-1} - |<gamma_S>|^2 (tested). The ensemble is invariant
under permutations of the Majoranas, so the variance does not depend on the
qubit order (tested: blocked vs interleaved).

Rao-Blackwellized estimates for large targets (``method="rb"``)
------------------------------------------------------------
The exact pair sums cost sum_e (terms of O_e)^2 pair evaluations (about 3e9 for
the 840 direct-route X_e of H6, estimate). Sampling only the setting (a matching
pi, or local bases B) and averaging over outcomes exactly gives

    Var(x_e) = E_setting[ <O_{e,setting}^2> ] - <O_e>^2,
    O_{e,pi} = sum_{S covered by pi} a_eS lambda_S^{-1} gamma_S,   O_{e,B} = sum_{P covered by B} W_eP 3^{|P|} P,

where O_{e,setting} is diagonal in the setting's basis (all covered strings
commute) and <O^2> = sum_{j,k covered} W'_ej W'_ek <P_j P_k> with the rescaled
coefficients W'. The identity term is always covered (lambda_0 = 1). The
estimate is the mean over the sampled settings, with standard errors from the
spread over settings; with ``settings=`` an explicit list (e.g. all matchings
or all 3^n bases for tiny n) the mean is the exact average over that list.
Coverage: S is a union of pi-pairs iff swapping the two bits of every pair
leaves S unchanged (:func:`matchgate_covered`, vectorized bit operations); the
RB loop instead enumerates the unions of at most k_max pairs and looks them up,
when that is cheaper (tested to agree).

Monte Carlo validators (``method="mc"``, :func:`df_variance_mc`)
-----------------------------------------------------------------
These simulate actual measurements on the dense state (small n): QWC rotates
the state into each group's product basis and samples outcomes; Pauli shadows
draw random local bases per snapshot and sample the rotated state; matchgate
shadows draw a random matching per snapshot and measure its n pair operators by
sequential projective sampling, then read every covered term off the
post-measurement state as <phi|P_j|phi> (= +-1), independently of the phase
bookkeeping of the exact formula; DF rotates the CI vector into each fragment's
orbital basis (``pyscf.fci.addons.transform_ci``), samples determinants and
evaluates the diagonal fragment operator. Per-target variances are sample
variances with standard errors from the fourth moment; the reported ``total``
and its ``stderr`` combine them as described in each function.

Expectation values
------------------
All exact formulas need <P> for many strings P. For states with
(#x patterns of the support) * 2^n <= 2^23 (H6: 1024 * 4096, observed) a dense
table of <P(x, z)> for all z is built once per state by Walsh-Hadamard
transforms; otherwise :meth:`~.pauli.QubitState.expectations` is called on the
unique strings. Timings are listed in tests/test_schemes.py ("observed", one thread).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp

from .pauli import (PauliSum, QubitState, coefficient_matrix, majorana_product, majorana_to_pauli,
                    pauli_product, pauli_to_majorana)

__all__ = [
    "SCHEMES",
    "VarianceResult",
    "QWCGrouping",
    "qwc_grouping",
    "qwc_variance",
    "pauli_shadow_variance",
    "matchgate_shadow_variance",
    "scheme_variance",
    "df_variance",
    "df_variance_mc",
    "combine_independent",
    "matchgate_lambda",
    "matchgate_p_both",
    "matchgate_covered",
    "pauli_shadow_covered",
    "random_matchings",
    "perfect_matchings",
    "pauli_bases",
    "rotate_to_basis",
]

SCHEMES = ("qwc", "pauli_shadow", "matchgate_shadow")

_U1 = np.uint64(1)
_PAIR_CHUNK = 1 << 21        # pair evaluations per vectorized block
_TABLE_MAX = 1 << 23         # entries of the dense <P(x, z)> table (64 MB of float64)
_NORM_ATOL = 1e-8            # |<psi|psi> - 1| allowed for "normalized" states
_HERM_RTOL = 1e-10           # max |Im W| / max |W| allowed for Hermitian targets
# method="auto": exact pair sums up to this many pairs, else Rao-Blackwellized MC. Pauli shadows
# always use the exact sums: their RB estimate is heavy-tailed (module docstring).
_AUTO_MAX_PAIRS = {"pauli_shadow": np.inf, "matchgate_shadow": 2e7}


# =============================================================== result type
@dataclass(frozen=True)
class VarianceResult:
    """Per-shot variances of an estimator of one target or of a vector of targets.

    Attributes:
        scheme: "qwc", "df", "pauli_shadow", "matchgate_shadow" or "combined".
        method: "exact", "rb" (Rao-Blackwellized Monte Carlo over settings) or
            "mc" (snapshot Monte Carlo).
        total: sum_e Var(x_e) per shot under ``allocation`` (the variance for a
            scalar target); S shots give total / S.
        per_target: (K,) per-shot variances Var(x_e) under the same allocation.
        expectation: (K,) exact <O_e> (including identity terms). For "df" the
            exact <H>; the model's value is in ``info``.
        n_settings: QWC groups, L + 1 DF bases; None for randomized schemes.
        allocation: shot fractions per setting (optimal), or None.
        uniform_total: total under uniform allocation over the settings (QWC, DF).
        per_target_uniform: per-target variances under uniform allocation.
        setting_variances: V_G (QWC, summed over targets) or Var_l (DF).
        stderr: standard error of ``total`` ("rb", "mc").
        per_target_stderr: standard errors of ``per_target`` ("rb", "mc").
        uniform_stderr: standard error of ``uniform_total`` (QWC/DF "mc").
        n_samples: settings ("rb") or snapshots/shots ("mc") used.
        info: scheme-specific extras (documented by each function).
    """

    scheme: str
    method: str
    total: float
    per_target: np.ndarray
    expectation: np.ndarray
    n_settings: int | None = None
    allocation: np.ndarray | None = None
    uniform_total: float | None = None
    per_target_uniform: np.ndarray | None = None
    setting_variances: np.ndarray | None = None
    stderr: float | None = None
    per_target_stderr: np.ndarray | None = None
    uniform_stderr: float | None = None
    n_samples: int | None = None
    info: dict = field(default_factory=dict)

    @property
    def n_targets(self) -> int:
        return int(np.size(self.per_target))

    def shots(self, precision: float) -> float:
        """Shots S with total / S = precision^2 (e.g. 1.6e-3 Ha for the energy)."""
        return float(self.total) / float(precision) ** 2


# ============================================================ target handling
@dataclass(frozen=True)
class _Targets:
    """Union Pauli basis (sorted, unique) and the real coefficient matrix W (K x B, CSR)."""

    n: int
    scalar: bool
    xs: np.ndarray
    zs: np.ndarray
    W: sp.csr_matrix
    id_col: int            # column of the identity string, -1 if absent

    @property
    def K(self) -> int:
        return self.W.shape[0]

    @property
    def B(self) -> int:
        return self.xs.size

    def without_identity(self) -> sp.csr_matrix:
        if self.id_col < 0:
            return self.W
        w = self.W.tocsc(copy=True)
        w[:, self.id_col] = 0.0
        w = w.tocsr()
        w.eliminate_zeros()
        return w


def _prepare(targets) -> _Targets:
    if isinstance(targets, PauliSum):
        ops, scalar = [targets], True
    else:
        ops, scalar = list(targets), False
        if not ops:
            raise ValueError("need at least one target")
        for o in ops:
            if not isinstance(o, PauliSum):
                raise TypeError(f"targets must be PauliSums, got {type(o).__name__}")
    xs, zs, w = coefficient_matrix(ops)
    w = w.tocsr()
    data = w.data
    if data.size:
        scale = float(np.max(np.abs(data)))
        imag = float(np.max(np.abs(data.imag)))
        if imag > _HERM_RTOL * scale:
            raise ValueError(f"targets must be Hermitian (max |Im c| = {imag:.3g}, max |c| = {scale:.3g})")
    wr = sp.csr_matrix((data.real.astype(np.float64), w.indices, w.indptr), shape=w.shape)
    wr.eliminate_zeros()
    used = np.unique(wr.indices)
    if used.size < xs.size:   # drop strings that only carried round-off imaginary parts
        remap = -np.ones(xs.size, dtype=np.int64)
        remap[used] = np.arange(used.size)
        wr = sp.csr_matrix((wr.data, remap[wr.indices], wr.indptr), shape=(wr.shape[0], used.size))
        xs, zs = xs[used], zs[used]
    wr.sort_indices()
    id_col = 0 if xs.size and xs[0] == 0 and zs[0] == 0 else -1
    return _Targets(ops[0].n_qubits, scalar, xs, zs, wr, id_col)


def _check_state(state: QubitState, n: int) -> None:
    if not isinstance(state, QubitState):
        raise TypeError(f"state must be a QubitState, got {type(state).__name__}")
    if state.n_qubits != n:
        raise ValueError(f"state has {state.n_qubits} qubits, targets have {n}")
    nrm2 = float(np.vdot(state.amplitudes, state.amplitudes).real)
    if abs(nrm2 - 1.0) > _NORM_ATOL:
        raise ValueError(f"the state must be normalized (<psi|psi> = {nrm2:.12g})")


def _rng(seed) -> np.random.Generator:
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


# ======================================================= expectation oracle
def _fwht_last_axis(a: np.ndarray) -> np.ndarray:
    """In-place unnormalized Walsh-Hadamard transform along the last axis (length 2^n)."""
    size = a.shape[-1]
    lead = a.shape[:-1]
    h = 1
    while h < size:
        v = a.reshape(*lead, size // (2 * h), 2, h)
        top = v[..., 0, :].copy()
        v[..., 0, :] += v[..., 1, :]
        np.subtract(top, v[..., 1, :], out=v[..., 1, :])
        h *= 2
    return a


class _Expectations:
    """Real <psi|P(x, z)|psi> for many strings (dense table when small, else ``expectations``)."""

    def __init__(self, state: QubitState, max_entries: int = _TABLE_MAX):
        self.state = state
        self.n = n = state.n_qubits
        self.table = None
        b, a = state.basis, state.amplitudes
        d = b.size
        if d == 0 or n > 24 or (1 << n) > max_entries or d > 4096:
            return
        xset = np.unique((b[:, None] ^ b[None, :]).ravel())
        if xset.size * (1 << n) > max_entries:
            return
        dim = 1 << n
        pos = np.full(dim, -1, dtype=np.int64)
        pos[b.astype(np.intp)] = np.arange(d)
        table = np.empty((xset.size, dim), dtype=np.float64)
        zz = np.arange(dim, dtype=np.uint64)
        ipow = np.array([1, 1j, -1, -1j])
        step = max(1, (1 << 20) // dim)
        for s in range(0, xset.size, step):
            xc = xset[s:s + step]
            partner = pos[(b[None, :] ^ xc[:, None]).astype(np.intp)]
            r, c = np.nonzero(partner >= 0)
            blk = np.zeros((xc.size, dim), dtype=np.complex128)
            blk[r, b[c].astype(np.intp)] = np.conj(a[partner[r, c]]) * a[c]
            _fwht_last_axis(blk)
            blk *= ipow[np.bitwise_count(xc[:, None] & zz[None, :]) & 3]
            table[s:s + step] = blk.real
        xrow = np.full(dim, -1, dtype=np.int64)
        xrow[xset.astype(np.intp)] = np.arange(xset.size)
        self.table, self.xrow = table, xrow

    def __call__(self, xs, zs) -> np.ndarray:
        xs = np.asarray(xs, dtype=np.uint64).ravel()
        zs = np.asarray(zs, dtype=np.uint64).ravel()
        if xs.size == 0:
            return np.zeros(0)
        if self.table is not None:
            r = self.xrow[xs.astype(np.intp)]
            hit = r >= 0
            out = np.zeros(xs.size)
            out[hit] = self.table[r[hit], zs[hit].astype(np.intp)]
            return out
        if self.n <= 32:
            key = (xs << np.uint64(32)) | zs
            uk, inv = np.unique(key, return_inverse=True)
            vals = self.state.expectations(uk >> np.uint64(32), uk & np.uint64(0xFFFFFFFF)).real
        else:
            pairs = np.stack([xs, zs], axis=1)
            up, inv = np.unique(pairs, axis=0, return_inverse=True)
            vals = self.state.expectations(up[:, 0], up[:, 1]).real
        return vals[inv.ravel()]


class _Accumulator:
    """Accumulates sum coef * <P(x, z)> (plus constants) into integer slots."""

    def __init__(self, n_slots: int, oracle: _Expectations):
        self.n_slots = int(n_slots)
        self.oracle = oracle
        self.out = np.zeros(self.n_slots)
        self._buf: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self._buf_len = 0
        self.n_terms = 0

    def add(self, slots, xs, zs, coefs) -> None:
        if slots.size == 0:
            return
        self.n_terms += slots.size
        if self.oracle.table is not None:
            self.out += np.bincount(slots, weights=coefs * self.oracle(xs, zs), minlength=self.n_slots)
            return
        self._buf.append(self._reduce(slots, xs, zs, coefs))
        self._buf_len += self._buf[-1][0].size
        if self._buf_len > 4 * _PAIR_CHUNK and len(self._buf) > 1:
            self._buf = [self._reduce(*(np.concatenate(v) for v in zip(*self._buf)))]
            self._buf_len = self._buf[0][0].size

    def add_const(self, slots, values) -> None:
        self.out += np.bincount(slots, weights=values, minlength=self.n_slots)

    @staticmethod
    def _reduce(slots, xs, zs, coefs):
        """Sum the coefficients of equal (slot, x, z)."""
        keys = np.stack([slots.astype(np.uint64), xs, zs], axis=1)
        order = np.lexsort((zs, xs, slots))
        k = keys[order]
        flag = np.ones(k.shape[0], dtype=bool)
        flag[1:] = np.any(k[1:] != k[:-1], axis=1)
        starts = np.flatnonzero(flag)
        c = np.add.reduceat(coefs[order], starts)
        ks = k[starts]
        return ks[:, 0].astype(np.int64), ks[:, 1], ks[:, 2], c

    def result(self) -> np.ndarray:
        if self._buf:
            s, x, z, c = (self._buf[0] if len(self._buf) == 1
                          else self._reduce(*(np.concatenate(v) for v in zip(*self._buf))))
            self.out += np.bincount(s, weights=c * self.oracle(x, z), minlength=self.n_slots)
            self._buf, self._buf_len = [], 0
        return self.out


def _row_pairs(indptr: np.ndarray, chunk: int = _PAIR_CHUNK):
    """Yield (slot, pu, pv): all pairs u <= v of CSR positions within each row, in blocks."""
    indptr = np.asarray(indptr, dtype=np.int64)
    m = np.diff(indptr)
    npairs = m * (m + 1) // 2
    small = np.flatnonzero((npairs > 0) & (npairs <= chunk))
    for mv in np.unique(m[small]):
        rows = small[m[small] == mv]
        tu, tv = np.triu_indices(int(mv))
        per = tu.size
        step = max(1, chunk // per)
        for s in range(0, rows.size, step):
            r = rows[s:s + step]
            start = indptr[r][:, None]
            yield np.repeat(r, per), (start + tu).ravel(), (start + tv).ravel()
    for r in np.flatnonzero(npairs > chunk):
        a, mv = int(indptr[r]), int(m[r])
        bs = max(1, chunk // mv)
        v = np.arange(mv)
        for u0 in range(0, mv, bs):
            u = np.arange(u0, min(mv, u0 + bs))
            uu, vv = np.broadcast_arrays(u[:, None], v[None, :])
            keep = vv >= uu
            yield np.full(int(keep.sum()), r, dtype=np.int64), a + uu[keep], a + vv[keep]


def _pair_count(indptr: np.ndarray) -> float:
    m = np.diff(np.asarray(indptr, dtype=np.int64)).astype(np.float64)
    return float(np.sum(m * (m + 1) / 2))


def _means(tg: _Targets, oracle: _Expectations) -> tuple[np.ndarray, np.ndarray]:
    """(<P_j> for the basis, <O_e>) with the identity included."""
    p = oracle(tg.xs, tg.zs)
    return p, np.asarray(tg.W @ p).ravel()


# ================================================================ QWC grouping
@dataclass(frozen=True)
class QWCGrouping:
    """Sorted-insertion QWC grouping of the union basis of a target.

    Attributes:
        n_qubits: number of qubits.
        xs, zs: the basis strings (sorted, unique; as from ``coefficient_matrix``).
        group: group index of every basis string, -1 for the identity.
        group_x, group_z: per group, the union string (one Pauli letter per
            qubit of the group's support): the measured product basis.
        order: the sorted-insertion order (indices into xs).
    """

    n_qubits: int
    xs: np.ndarray
    zs: np.ndarray
    group: np.ndarray
    group_x: np.ndarray
    group_z: np.ndarray
    order: np.ndarray

    @property
    def n_groups(self) -> int:
        return int(self.group_x.size)

    def members(self, g: int) -> np.ndarray:
        """Indices (into xs) of the strings of group g."""
        return np.flatnonzero(self.group == g)

    def sizes(self) -> np.ndarray:
        return np.bincount(self.group[self.group >= 0], minlength=self.n_groups)


def _qwc_grouping(tg: _Targets) -> QWCGrouping:
    weight = np.asarray(abs(tg.W).sum(axis=0)).ravel()
    cand = np.flatnonzero(~((tg.xs == 0) & (tg.zs == 0)))
    order = cand[np.argsort(-weight[cand], kind="stable")]
    group = np.full(tg.B, -1, dtype=np.int64)
    cap = 64
    gx = np.zeros(cap, dtype=np.uint64)
    gz = np.zeros(cap, dtype=np.uint64)
    ng = 0
    for j in order:
        x, z = tg.xs[j], tg.zs[j]
        s = x | z
        if ng:
            hx, hz = gx[:ng], gz[:ng]
            ok = (((hx ^ x) | (hz ^ z)) & ((hx | hz) & s)) == 0
            g = int(np.argmax(ok)) if ok.any() else -1
        else:
            g = -1
        if g < 0:
            if ng == cap:
                cap *= 2
                gx = np.resize(gx, cap)
                gz = np.resize(gz, cap)
            g = ng
            gx[g] = 0
            gz[g] = 0
            ng += 1
        gx[g] |= x
        gz[g] |= z
        group[j] = g
    return QWCGrouping(tg.n, tg.xs, tg.zs, group, gx[:ng].copy(), gz[:ng].copy(), order)


def qwc_grouping(targets) -> QWCGrouping:
    """Sorted-insertion QWC grouping of a target or vector target (module docstring).

    Every non-identity string of the union basis is in exactly one group, and
    the strings of a group commute qubit-wise (tested).
    """
    return _qwc_grouping(_prepare(targets))


def _qwc_blocks(tg: _Targets, grouping: QWCGrouping):
    """CSR-like blocks of the nonzeros of W sorted by (target e, group G)."""
    w = tg.without_identity().tocoo()
    g = grouping.group[w.col]
    if np.any(g < 0):
        raise ValueError("the grouping does not cover the targets' basis")
    order = np.lexsort((w.col, g, w.row))
    rows, cols, vals, gs = w.row[order], w.col[order], w.data[order], g[order]
    flag = np.ones(rows.size, dtype=bool)
    flag[1:] = (rows[1:] != rows[:-1]) | (gs[1:] != gs[:-1])
    starts = np.flatnonzero(flag)
    indptr = np.append(starts, rows.size)
    return indptr, cols, vals, rows[starts].astype(np.int64), gs[starts].astype(np.int64)


def _check_grouping(grouping: QWCGrouping, tg: _Targets) -> None:
    if (grouping.xs.size != tg.B or not np.array_equal(grouping.xs, tg.xs)
            or not np.array_equal(grouping.zs, tg.zs)):
        raise ValueError("grouping was built for a different set of Pauli strings")


def qwc_variance(targets, state: QubitState, *, grouping: QWCGrouping | None = None,
                 method: str = "exact", n_shots: int = 10000, seed=None) -> VarianceResult:
    """Per-shot variance of QWC-grouped measurement (sorted insertion; module docstring).

    ``method="exact"``: V_{G,e} from the covariances; ``total`` and ``per_target``
    under the optimal allocation f_G = sqrt(V_G) / sum sqrt(V_H), ``uniform_total``
    and ``per_target_uniform`` under f_G = 1/n_G, ``setting_variances`` = V_G,
    ``allocation`` = f_G, ``n_settings`` = number of groups. ``info`` holds the
    grouping, the group sizes and ``group_target_variances`` (sparse V_{G,e} as
    (e, G, value) arrays).

    ``method="mc"``: actual measurements on the dense state (n <= 20). Every group
    is measured ``n_shots`` times in its product basis; V_hat_{G,e} are sample
    variances. The optimal allocation is then the one of the sampled V_hat_G, and
    ``stderr`` (``uniform_stderr``) is the delta-method standard error of
    (sum sqrt(V_hat_G))^2 (of n_G sum V_hat_G). ``info["group_stderr"]`` holds the
    standard errors of V_hat_G and ``info["sample_mean"]``/``info["sample_mean_stderr"]``
    the combined estimate of <O_e> (a uniform-shot campaign) and its error.
    """
    tg = _prepare(targets)
    _check_state(state, tg.n)
    if grouping is None:
        grouping = _qwc_grouping(tg)
    else:
        _check_grouping(grouping, tg)
    oracle = _Expectations(state)
    p, mu = _means(tg, oracle)
    ng = grouping.n_groups
    info = {"grouping": grouping, "group_sizes": grouping.sizes(), "n_terms": int(tg.B - (tg.id_col >= 0))}
    if method == "mc":
        return _qwc_mc(tg, state, grouping, mu, int(n_shots), _rng(seed), info)
    if method != "exact":
        raise ValueError(f"method must be 'exact' or 'mc' for QWC, got {method!r}")
    indptr, cols, vals, brow, bgrp = _qwc_blocks(tg, grouping)
    nb = brow.size
    acc = _Accumulator(nb, oracle)
    for slot, pu, pv in _row_pairs(indptr):
        j, k = cols[pu], cols[pv]
        w = vals[pu] * vals[pv] * np.where(pu == pv, 1.0, 2.0)
        acc.add(slot, tg.xs[j] ^ tg.xs[k], tg.zs[j] ^ tg.zs[k], w)
        acc.add_const(slot, -w * p[j] * p[k])
    vb = acc.result()
    return _allocate_groups("qwc", "exact", tg, mu, vb, brow, bgrp, ng, info)


def _allocate_groups(scheme, method, tg, mu, vb, brow, bgrp, ng, info, extra=None) -> VarianceResult:
    """Optimal and uniform allocations from block variances V_{G,e} (block = (e, G))."""
    vg = np.bincount(bgrp, weights=vb, minlength=ng)
    sq = np.sqrt(np.clip(vg, 0.0, None))
    tot_sq = float(sq.sum())
    f = sq / tot_sq if tot_sq > 0 else np.full(ng, 1.0 / max(ng, 1))
    inv_f = np.divide(1.0, f, out=np.zeros(ng), where=f > 0)
    per = np.bincount(brow, weights=vb * inv_f[bgrp], minlength=tg.K)
    per_u = ng * np.bincount(brow, weights=vb, minlength=tg.K)
    info = dict(info)
    info["group_target_variances"] = (brow, bgrp, vb)
    kw = dict(extra or {})
    return VarianceResult(scheme=scheme, method=method, total=tot_sq ** 2, per_target=per,
                          expectation=mu, n_settings=ng, allocation=f,
                          uniform_total=float(ng * vg.sum()), per_target_uniform=per_u,
                          setting_variances=vg, info=info, **kw)


# ======================================================= dense basis rotations
_H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)
_HSDG = np.array([[1.0, -1.0j], [1.0, 1.0j]], dtype=np.complex128) / np.sqrt(2.0)   # H S^dagger


def rotate_to_basis(psi: np.ndarray, bx: int, bz: int) -> np.ndarray:
    """Dense state rotated so that a computational-basis measurement measures the basis (bx, bz).

    On qubit k the letter is X (bx_k = 1, bz_k = 0), Y (1, 1) or Z (0, 1, or
    unmeasured 0, 0); the gates are H, H S^dagger and I, with V sigma V^dagger = Z.
    A string P on the measured qubits with these letters then has the outcome
    value (-1)^{|b & supp P|} on outcome b. ``psi`` may be a (..., 2^n) array.
    """
    psi = np.asarray(psi, dtype=np.complex128)
    dim = psi.shape[-1]
    n = dim.bit_length() - 1
    out = psi.copy()
    lead = psi.shape[:-1]
    for k in range(n):
        xb, zb = (int(bx) >> k) & 1, (int(bz) >> k) & 1
        if not xb:
            continue
        g = _HSDG if zb else _H
        v = out.reshape(*lead, dim >> (k + 1), 2, 1 << k)
        out = np.einsum("ij,...ajb->...aib", g, v).reshape(*lead, dim)
    return out


def _rotate_rows(psi_rows: np.ndarray, bx: np.ndarray, bz: np.ndarray) -> np.ndarray:
    """Row r rotated to the basis (bx[r], bz[r]) (vectorized over rows)."""
    out = np.array(psi_rows, dtype=np.complex128, copy=True)
    rows, dim = out.shape
    n = dim.bit_length() - 1
    gates = np.stack([np.eye(2, dtype=np.complex128), _H, np.eye(2, dtype=np.complex128), _HSDG])
    for k in range(n):
        code = ((bx >> np.uint64(k)) & _U1).astype(np.int64) + 2 * ((bz >> np.uint64(k)) & _U1).astype(np.int64)
        if not np.any(code & 1):
            continue
        g = gates[code][:, :, :, None, None]              # (rows, 2, 2, 1, 1); identity for Z / unmeasured
        v = out.reshape(rows, dim >> (k + 1), 2, 1 << k)
        v0, v1 = v[:, :, 0, :], v[:, :, 1, :]
        new = np.empty_like(v)
        new[:, :, 0, :] = g[:, 0, 0] * v0 + g[:, 0, 1] * v1
        new[:, :, 1, :] = g[:, 1, 0] * v0 + g[:, 1, 1] * v1
        out = new.reshape(rows, dim)
    return out


def _sample_rows(probs: np.ndarray, counts: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw counts[r] outcomes from row r of probs (inverse CDF); returns (row, outcome)."""
    rows = np.repeat(np.arange(probs.shape[0]), counts)
    cdf = np.cumsum(probs, axis=1)
    cdf /= cdf[:, -1:]
    u = rng.random(rows.size)
    flat = (cdf + np.arange(probs.shape[0])[:, None]).ravel()
    idx = np.searchsorted(flat, rows + u, side="right")
    out = idx - rows * probs.shape[1]
    return rows, np.minimum(out, probs.shape[1] - 1).astype(np.int64)


def _parity(a: np.ndarray) -> np.ndarray:
    return (np.bitwise_count(a) & 1).astype(np.int8)


def _qwc_mc(tg: _Targets, state: QubitState, grouping: QWCGrouping, mu: np.ndarray, n_shots: int,
            rng: np.random.Generator, info: dict) -> VarianceResult:
    if tg.n > 20:
        raise ValueError("Monte Carlo validators need the dense state (n <= 20)")
    if n_shots < 2:
        raise ValueError("n_shots must be >= 2")
    psi = state.to_dense()
    ng = grouping.n_groups
    wn = tg.without_identity().tocsc()
    s_all = tg.xs | tg.zs
    vb_rows, vb_grps, vb_vals, vb_se = [], [], [], []
    vg_hat = np.zeros(ng)
    vg_se = np.zeros(ng)
    mean_est = np.zeros(tg.K)
    mean_var = np.zeros(tg.K)
    for g in range(ng):
        mem = grouping.members(g)
        sub = wn[:, mem].tocsr()
        rows = np.flatnonzero(np.diff(sub.indptr))
        rot = rotate_to_basis(psi, int(grouping.group_x[g]), int(grouping.group_z[g]))
        probs = np.abs(rot) ** 2
        _, b = _sample_rows(probs[None, :], np.array([n_shots]), rng)
        bu = b.astype(np.uint64)
        sign = 1.0 - 2.0 * _parity(bu[:, None] & s_all[mem][None, :])     # (S, |G|)
        x = sign @ sub[rows].toarray().T                                    # (S, rows)
        d = x - x.mean(axis=0)
        v = (d ** 2).sum(axis=0) / (n_shots - 1)
        m4 = (d ** 4).mean(axis=0)
        se = np.sqrt(np.clip(m4 - v ** 2, 0.0, None) / n_shots)
        tot = (d ** 2).sum(axis=1)
        vg_hat[g] = tot.sum() / (n_shots - 1)
        vg_se[g] = tot.std() / np.sqrt(n_shots)
        vb_rows.append(rows)
        vb_grps.append(np.full(rows.size, g))
        vb_vals.append(v)
        vb_se.append(se)
        mean_est[rows] += x.mean(axis=0)
        mean_var[rows] += v / n_shots
    brow = np.concatenate(vb_rows).astype(np.int64)
    bgrp = np.concatenate(vb_grps).astype(np.int64)
    vb = np.concatenate(vb_vals)
    seb = np.concatenate(vb_se)
    if tg.id_col >= 0:
        mean_est += np.asarray(tg.W[:, tg.id_col].toarray()).ravel()
    res = _allocate_groups("qwc", "mc", tg, mu, vb, brow, bgrp, ng, info)
    sq = np.sqrt(np.clip(res.setting_variances, 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(sq > 0, vg_se / sq, 0.0)
    stderr = float(sq.sum() * np.sqrt(np.sum(rel ** 2)))     # d(sum sqrt V)^2 / dV_G = sum sqrt V / sqrt V_G
    f = res.allocation
    inv_f = np.divide(1.0, f, out=np.zeros(ng), where=f > 0)
    per_se = np.sqrt(np.bincount(brow, weights=(seb * inv_f[bgrp]) ** 2, minlength=tg.K))
    new_info = dict(res.info)
    new_info.update(group_stderr=vg_se, sample_mean=mean_est, sample_mean_stderr=np.sqrt(mean_var),
                    group_target_stderr=seb)
    return VarianceResult(scheme="qwc", method="mc", total=res.total, per_target=res.per_target,
                          expectation=mu, n_settings=ng, allocation=f, uniform_total=res.uniform_total,
                          per_target_uniform=res.per_target_uniform, setting_variances=res.setting_variances,
                          stderr=stderr, per_target_stderr=per_se,
                          uniform_stderr=float(ng * np.sqrt(np.sum(vg_se ** 2))),
                          n_samples=n_shots * ng, info=new_info)


# ======================================================== matchgate helpers
def matchgate_lambda(n: int, k) -> np.ndarray | float:
    """lambda_k = C(n, k) / C(2n, 2k): probability that a 2k-Majorana monomial is covered (n qubits)."""
    k_arr = np.asarray(k)
    lam = _lambda_table(int(n))
    out = lam[k_arr]
    return float(out) if out.ndim == 0 else out


@lru_cache(maxsize=None)
def _lambda_table(n: int) -> np.ndarray:
    t = np.array([math.comb(n, k) / math.comb(2 * n, 2 * k) for k in range(n + 1)])
    t.setflags(write=False)
    return t


def _double_factorial_odd(m: int) -> int:
    """(m - 1)!! for even m >= 0 (the number of perfect matchings of m elements); 1 for m = 0."""
    out = 1
    for v in range(m - 1, 0, -2):
        out *= v
    return out


@lru_cache(maxsize=None)
def _p_both_table(n: int) -> np.ndarray:
    """T[a, b, c] = P_both for |A| = a, |B| = b, |C| = c (zero unless all even and a+b+c <= 2n)."""
    m = 2 * n
    t = np.zeros((m + 1, m + 1, m + 1))
    total = _double_factorial_odd(m)
    for a in range(0, m + 1, 2):
        for b in range(0, m + 1 - a, 2):
            for c in range(0, m + 1 - a - b, 2):
                num = (_double_factorial_odd(a) * _double_factorial_odd(b) * _double_factorial_odd(c)
                       * _double_factorial_odd(m - a - b - c))
                t[a, b, c] = float(Fraction(num, total))
    t.setflags(write=False)
    return t


def matchgate_p_both(n: int, s1, s2) -> np.ndarray | float:
    """Probability that a uniformly random perfect matching of 2n Majoranas covers both S1 and S2.

    s1, s2: Majorana bitmasks (broadcasting). Returns P_both of the module
    docstring (0 unless |S1 n S2|, |S1 \\ S2| and |S2 \\ S1| are all even).
    """
    a1, a2 = np.broadcast_arrays(np.asarray(s1, dtype=np.uint64), np.asarray(s2, dtype=np.uint64))
    t = _p_both_table(int(n))
    a = np.bitwise_count(a1 & a2).astype(np.int64)
    b = np.bitwise_count(a1 & ~a2).astype(np.int64)
    c = np.bitwise_count(a2 & ~a1).astype(np.int64)
    if np.any(a + b + c > 2 * n):
        raise ValueError(f"masks have bits beyond the {2 * n} Majoranas")
    out = t[a, b, c]
    return float(out) if out.ndim == 0 else out


def random_matchings(n: int, size: int, rng=None) -> np.ndarray:
    """``size`` uniformly random perfect matchings of the 2n Majoranas, shape (size, n, 2).

    Pairs (a, b) have a < b; a uniformly random permutation paired consecutively
    gives every matching with probability 1/(2n-1)!!.
    """
    rng = _rng(rng)
    perm = rng.permuted(np.tile(np.arange(2 * n), (int(size), 1)), axis=1).reshape(int(size), n, 2)
    return np.sort(perm, axis=2)


def perfect_matchings(m: int) -> np.ndarray:
    """All (m-1)!! perfect matchings of {0, ..., m-1} (m even), shape (count, m/2, 2)."""
    if m % 2:
        raise ValueError("perfect matchings need an even number of elements")

    def rec(items):
        if not items:
            yield []
            return
        a = items[0]
        for i in range(1, len(items)):
            rest = items[1:i] + items[i + 1:]
            for tail in rec(rest):
                yield [(a, items[i])] + tail

    out = np.array(list(rec(list(range(m)))), dtype=np.int64)
    return out.reshape(-1, m // 2, 2)


def matchgate_covered(masks, matchings) -> np.ndarray:
    """Coverage test: S is a union of pairs of the matching iff swapping each pair's two bits fixes S.

    masks: (B,) Majorana bitmasks; matchings: (n, 2) or (R, n, 2) pairs.
    Returns bool (B,) or (R, B). Vectorized bit operations over masks and matchings.
    """
    s = np.asarray(masks, dtype=np.uint64).ravel()
    mt = np.asarray(matchings, dtype=np.int64)
    single = mt.ndim == 2
    mt = mt.reshape(-1, mt.shape[-2], 2)
    swapped = np.broadcast_to(s, (mt.shape[0], s.size)).copy()
    for p in range(mt.shape[1]):
        a = mt[:, p, 0].astype(np.uint64)[:, None]
        b = mt[:, p, 1].astype(np.uint64)[:, None]
        diff = ((s[None, :] >> a) ^ (s[None, :] >> b)) & _U1   # 1 where the two bits differ
        swapped ^= (diff << a) | (diff << b)                     # swap = flip both when they differ
    out = swapped == s[None, :]
    return out[0] if single else out


def pauli_shadow_covered(xs, zs, bx, bz) -> np.ndarray:
    """P(x, z) is covered by local bases (bx, bz) iff its letters equal the bases on its support.

    bx, bz may be scalars or (R,) arrays (one basis per row); returns bool (B,) or (R, B).
    """
    x = np.asarray(xs, dtype=np.uint64).ravel()
    z = np.asarray(zs, dtype=np.uint64).ravel()
    bxa = np.asarray(bx, dtype=np.uint64)
    bza = np.asarray(bz, dtype=np.uint64)
    out = (((x ^ bxa[..., None]) | (z ^ bza[..., None])) & (x | z)) == 0
    return out


def pauli_bases(n: int, size: int | None = None, rng=None) -> tuple[np.ndarray, np.ndarray]:
    """Local bases as (bx, bz) masks: all 3^n (size=None) or ``size`` uniformly random ones."""
    if size is None:
        letters = np.array(list(itertools.product(range(3), repeat=n)), dtype=np.int64).reshape(-1, n)
    else:
        letters = _rng(rng).integers(0, 3, size=(int(size), n))
    # letter 0 = X (1, 0), 1 = Y (1, 1), 2 = Z (0, 1)
    w = (np.uint64(1) << np.arange(n, dtype=np.uint64))
    bx = ((letters <= 1).astype(np.uint64) * w).sum(axis=1).astype(np.uint64)
    bz = ((letters >= 1).astype(np.uint64) * w).sum(axis=1).astype(np.uint64)
    return bx, bz


# =============================================== exact shadow pair sums
def _pauli_shadow_exact(tg: _Targets, oracle: _Expectations, p: np.ndarray) -> tuple[np.ndarray, int]:
    wn = tg.without_identity()
    acc = _Accumulator(tg.K, oracle)
    s_all = tg.xs | tg.zs
    for slot, pu, pv in _row_pairs(wn.indptr):
        j, k = wn.indices[pu], wn.indices[pv]
        xj, zj, xk, zk = tg.xs[j], tg.zs[j], tg.xs[k], tg.zs[k]
        ov = s_all[j] & s_all[k]
        ok = (((xj ^ xk) | (zj ^ zk)) & ov) == 0
        if not ok.any():
            continue
        w = (wn.data[pu] * wn.data[pv] * np.where(pu == pv, 1.0, 2.0))[ok]
        w *= 3.0 ** np.bitwise_count(ov[ok])
        acc.add(slot[ok], (xj ^ xk)[ok], (zj ^ zk)[ok], w)
    second = acc.result()
    mu0 = np.asarray(wn @ p).ravel()
    return second - mu0 ** 2, acc.n_terms


def _matchgate_exact(tg: _Targets, oracle: _Expectations, p: np.ndarray) -> tuple[np.ndarray, int]:
    n = tg.n
    if 2 * n > 64:
        raise ValueError("matchgate shadows need n <= 32 qubits (uint64 Majorana masks)")
    wn = tg.without_identity()
    masks, phi = pauli_to_majorana(tg.xs, tg.zs)      # P_j = phi_j gamma_{S_j}
    masks = np.atleast_1d(masks)
    phi = np.atleast_1d(phi)
    half = (np.bitwise_count(masks) // 2).astype(np.int64)
    lam = _lambda_table(n)
    pb = _p_both_table(n)
    acc = _Accumulator(tg.K, oracle)
    for slot, pu, pv in _row_pairs(wn.indptr):
        j, k = wn.indices[pu], wn.indices[pv]
        sj, sk = masks[j], masks[k]
        a = np.bitwise_count(sj & sk).astype(np.int64)
        ok = (a & 1) == 0
        if not ok.any():
            continue
        j, k, sj, sk, a, slot_ok = j[ok], k[ok], sj[ok], sk[ok], a[ok], slot[ok]
        kj, kk = half[j], half[k]
        weight = pb[a, 2 * kj - a, 2 * kk - a] / (lam[kj] * lam[kk])
        sgn, t = majorana_product(sj, sk)                 # gamma_Sj gamma_Sk = sgn gamma_T
        ph_t, xt, zt = majorana_to_pauli(t)                # gamma_T = ph_t P(x_T, z_T)
        # <gamma_Sj^dag gamma_Sk> = (-1)^{kj} sgn ph_t <P_T>; coefficients a = W phi
        ph = np.conj(phi[j]) * phi[k] * np.where(kj & 1, -1.0, 1.0) * sgn * np.atleast_1d(ph_t)
        if np.any(np.abs(ph.imag) > 1e-12):   # commuting Hermitian P_j P_k: the phase is +-1
            raise RuntimeError("internal error: non-real phase of <P_j P_k> for a commuting pair")
        w = (wn.data[pu] * wn.data[pv] * np.where(pu == pv, 1.0, 2.0))[ok]
        acc.add(slot_ok, np.atleast_1d(xt), np.atleast_1d(zt), w * weight * ph.real)
    second = acc.result()
    mu0 = np.asarray(wn @ p).ravel()
    return second - mu0 ** 2, acc.n_terms


# ==================================================== Rao-Blackwellized MC
class _Welford:
    """Running mean and M2 of vectors (Chan et al.'s pairwise update over batches)."""

    def __init__(self, k: int):
        self.n = 0
        self.mean = np.zeros(k)
        self.m2 = np.zeros(k)

    def update(self, batch: np.ndarray) -> None:
        nb = batch.shape[0]
        if nb == 0:
            return
        bm = batch.mean(axis=0)
        bm2 = ((batch - bm) ** 2).sum(axis=0)
        delta = bm - self.mean
        tot = self.n + nb
        self.mean = self.mean + delta * (nb / tot)
        self.m2 = self.m2 + bm2 + delta ** 2 * (self.n * nb / tot)
        self.n = tot

    def stderr(self) -> np.ndarray:
        if self.n < 2:
            return np.full(self.mean.shape, np.nan)
        return np.sqrt(self.m2 / (self.n - 1) / self.n)


class _RBSetup:
    """Everything the per-setting conditional second moments need."""

    def __init__(self, tg: _Targets, oracle: _Expectations, scheme: str):
        self.tg, self.oracle, self.scheme = tg, oracle, scheme
        n = tg.n
        self.non_id = np.flatnonzero(~((tg.xs == 0) & (tg.zs == 0)))
        if scheme == "matchgate_shadow":
            if 2 * n > 64:
                raise ValueError("matchgate shadows need n <= 32 qubits")
            masks, _ = pauli_to_majorana(tg.xs, tg.zs)
            masks = np.atleast_1d(masks)
            half = (np.bitwise_count(masks) // 2).astype(np.int64)
            scale = 1.0 / _lambda_table(n)[half]
            nm = masks[self.non_id]
            order = np.argsort(nm, kind="stable")
            self.sorted_masks = nm[order]
            self.sorted_cols = self.non_id[order]
            self.masks = masks
            self.kmax = int(half.max()) if half.size else 0
            n_cand = sum(math.comb(n, k) for k in range(1, self.kmax + 1))
            self.enumerate = n_cand <= self.non_id.size
            self.combos = [np.array(list(itertools.combinations(range(n), k)), dtype=np.int64).reshape(-1, k)
                           for k in range(1, self.kmax + 1)] if self.enumerate else []
        elif scheme == "pauli_shadow":
            scale = 3.0 ** np.bitwise_count(tg.xs | tg.zs).astype(np.float64)
        else:
            raise ValueError(f"unknown scheme {scheme!r}")
        self.wp = (tg.W @ sp.diags(scale)).tocsc()
        self.id_cols = np.array([tg.id_col], dtype=np.int64) if tg.id_col >= 0 else np.zeros(0, np.int64)

    def covered(self, setting) -> np.ndarray:
        tg = self.tg
        if self.scheme == "pauli_shadow":
            bx, bz = setting
            nid = self.non_id
            hit = pauli_shadow_covered(tg.xs[nid], tg.zs[nid], bx, bz)
            return np.concatenate([self.id_cols, nid[hit]])
        pairs = np.asarray(setting, dtype=np.int64)
        if self.enumerate:
            pm = (_U1 << pairs[:, 0].astype(np.uint64)) | (_U1 << pairs[:, 1].astype(np.uint64))
            cand = np.concatenate([pm[c].sum(axis=1) for c in self.combos]) if self.combos else np.zeros(0, np.uint64)
            sm = self.sorted_masks
            if sm.size == 0 or cand.size == 0:
                return self.id_cols
            loc = np.minimum(np.searchsorted(sm, cand), sm.size - 1)
            hit = sm[loc] == cand
            return np.concatenate([self.id_cols, np.sort(self.sorted_cols[loc[hit]])])
        hit = matchgate_covered(self.masks[self.non_id], pairs)
        return np.concatenate([self.id_cols, self.non_id[hit]])

    def second_moments(self, cols: np.ndarray) -> np.ndarray:
        """<O_{e,setting}^2> for every target e (K,), given the covered columns."""
        tg = self.tg
        if cols.size == 0:
            return np.zeros(tg.K)
        xc, zc = tg.xs[cols], tg.zs[cols]
        phi, x3, z3 = pauli_product(xc[:, None], zc[:, None], xc[None, :], zc[None, :])
        g = (1.0 - phi) * self.oracle(x3.ravel(), z3.ravel()).reshape(phi.shape)   # phase +-1 (commuting)
        wc = self.wp[:, cols]
        y = wc @ g
        return np.asarray(wc.multiply(y).sum(axis=1)).ravel()


def _shadow_rb(tg: _Targets, state: QubitState, scheme: str, *, n_settings: int, rel_stderr: float | None,
               min_settings: int, batch: int, seed, settings) -> VarianceResult:
    oracle = _Expectations(state)
    p, mu = _means(tg, oracle)
    setup = _RBSetup(tg, oracle, scheme)
    rng = _rng(seed)
    per = _Welford(tg.K)
    tot = _Welford(1)
    n = tg.n

    def run(block):
        vals = np.stack([setup.second_moments(setup.covered(s)) for s in block])
        per.update(vals)
        tot.update(vals.sum(axis=1, keepdims=True))

    if settings is not None:
        block = []
        for s in settings:
            block.append(s)
            if len(block) == batch:
                run(block)
                block = []
        if block:
            run(block)
    else:
        done = 0
        while done < n_settings:
            nb = min(batch, n_settings - done)
            if scheme == "matchgate_shadow":
                block = list(random_matchings(n, nb, rng))
            else:
                bx, bz = pauli_bases(n, nb, rng)
                block = list(zip(bx, bz))
            run(block)
            done += nb
            if rel_stderr is not None and done >= min_settings:
                t = float(tot.mean[0] - np.sum(mu ** 2))
                if t > 0 and float(tot.stderr()[0]) <= rel_stderr * t:
                    break
    var = per.mean - mu ** 2
    total = float(tot.mean[0] - np.sum(mu ** 2))
    return VarianceResult(scheme=scheme, method="rb", total=total, per_target=var, expectation=mu,
                          stderr=float(tot.stderr()[0]), per_target_stderr=per.stderr(), n_samples=per.n,
                          info={"n_terms": int(tg.B - (tg.id_col >= 0)),
                                "enumerated_coverage": getattr(setup, "enumerate", None)})


# ================================================ snapshot Monte Carlo
class _MomentSums:
    """Streaming sums of d = x - mu (per target) and of y = sum_e d_e^2 (per snapshot)."""

    def __init__(self, k: int):
        self.n = 0
        self.s1 = np.zeros(k)
        self.s2 = np.zeros(k)
        self.s4 = np.zeros(k)
        self.y1 = 0.0
        self.y2 = 0.0

    def update(self, d: np.ndarray) -> None:
        self.n += d.shape[0]
        d2 = d * d
        self.s1 += d.sum(axis=0)
        self.s2 += d2.sum(axis=0)
        self.s4 += (d2 * d2).sum(axis=0)
        y = d2.sum(axis=1)
        self.y1 += float(y.sum())
        self.y2 += float((y * y).sum())

    def result(self, scheme: str, mu: np.ndarray, info: dict) -> VarianceResult:
        s = self.n
        mean_d = self.s1 / s
        var = (self.s2 - s * mean_d ** 2) / (s - 1)           # sample variance (around the sample mean)
        m2, m4 = self.s2 / s, self.s4 / s                       # moments around the exact mean
        se = np.sqrt(np.clip(m4 - m2 ** 2, 0.0, None) / s)
        ym = self.y1 / s
        yse = np.sqrt(max(self.y2 / s - ym ** 2, 0.0) / s)
        info = dict(info)
        info.update(sample_mean=mu + mean_d, sample_mean_stderr=np.sqrt(np.clip(var, 0.0, None) / s))
        return VarianceResult(scheme=scheme, method="mc", total=float(var.sum()), per_target=var,
                              expectation=mu, stderr=float(yse), per_target_stderr=se, n_samples=s,
                              info=info)


def _pauli_shadow_mc(tg: _Targets, state: QubitState, mu: np.ndarray, n_snapshots: int, batch: int,
                     rng: np.random.Generator) -> VarianceResult:
    if tg.n > 14:
        raise ValueError("Monte Carlo validators need a small dense state (n <= 14)")
    psi = state.to_dense()
    n = tg.n
    nid = np.flatnonzero(~((tg.xs == 0) & (tg.zs == 0)))
    xs, zs = tg.xs[nid], tg.zs[nid]
    supp = xs | zs
    w3 = 3.0 ** np.bitwise_count(supp).astype(np.float64)
    wt = (tg.W[:, nid].toarray() * w3[None, :]).T                 # (terms, K)
    const = np.asarray(tg.W[:, tg.id_col].toarray()).ravel() if tg.id_col >= 0 else np.zeros(tg.K)
    acc = _MomentSums(tg.K)
    done = 0
    while done < n_snapshots:
        nb = min(batch, n_snapshots - done)
        bx, bz = pauli_bases(n, nb, rng)
        key = (bx << np.uint64(32)) | bz
        uk, inv = np.unique(key, return_inverse=True)
        ubx, ubz = uk >> np.uint64(32), uk & np.uint64(0xFFFFFFFF)
        rot = _rotate_rows(np.broadcast_to(psi, (uk.size, psi.size)), ubx, ubz)
        probs = np.abs(rot) ** 2
        counts = np.bincount(inv, minlength=uk.size)
        rows, outcome = _sample_rows(probs, counts, rng)
        # snapshots in the order of `rows` (sorted by unique basis)
        sbx, sbz = ubx[rows], ubz[rows]
        cov = pauli_shadow_covered(xs, zs, sbx, sbz)                 # (nb, terms)
        sign = 1.0 - 2.0 * _parity(outcome.astype(np.uint64)[:, None] & supp[None, :])
        x = (cov * sign) @ wt + const[None, :]
        acc.update(x - mu[None, :])
        done += nb
    return acc.result("pauli_shadow", mu, {"n_terms": int(nid.size)})


def _apply_pauli_rows(phi: np.ndarray, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Row r of the result is P(x[r], z[r]) phi[r] (P(x, z)|b> = i^{|x&z|} (-1)^{|b&z|} |b ^ x>)."""
    rows, dim = phi.shape
    bp = np.arange(dim, dtype=np.uint64)[None, :]
    src = bp ^ x[:, None]
    sign = 1.0 - 2.0 * _parity(src & z[:, None])
    ipow = np.array([1, 1j, -1, -1j])[np.bitwise_count(x & z) & 3]
    return ipow[:, None] * sign * np.take_along_axis(phi, src.astype(np.intp), axis=1)


def _matchgate_mc(tg: _Targets, state: QubitState, mu: np.ndarray, n_snapshots: int, batch: int,
                  rng: np.random.Generator) -> VarianceResult:
    if tg.n > 14:
        raise ValueError("Monte Carlo validators need a small dense state (n <= 14)")
    psi = state.to_dense()
    n = tg.n
    nid = np.flatnonzero(~((tg.xs == 0) & (tg.zs == 0)))
    xs, zs = tg.xs[nid], tg.zs[nid]
    masks, _ = pauli_to_majorana(xs, zs)
    masks = np.atleast_1d(masks)
    lam = _lambda_table(n)[(np.bitwise_count(masks) // 2).astype(np.int64)]
    wt = (tg.W[:, nid].toarray() / lam[None, :]).T                 # (terms, K)
    const = np.asarray(tg.W[:, tg.id_col].toarray()).ravel() if tg.id_col >= 0 else np.zeros(tg.K)
    acc = _MomentSums(tg.K)
    done = 0
    max_dev = 0.0
    while done < n_snapshots:
        nb = min(batch, n_snapshots - done)
        mt = random_matchings(n, nb, rng)
        phi = np.broadcast_to(psi, (nb, psi.size)).copy()
        for pidx in range(n):
            pm = (_U1 << mt[:, pidx, 0].astype(np.uint64)) | (_U1 << mt[:, pidx, 1].astype(np.uint64))
            ph, px, pz = majorana_to_pauli(pm)                       # gamma_a gamma_b = ph P(x, z), a < b
            coef = 1j * np.atleast_1d(ph)                            # i gamma_a gamma_b = coef P(x, z), +-1
            if np.any(np.abs(coef.imag) > 1e-12):
                raise RuntimeError("internal error: i gamma_a gamma_b is not +-P(x, z)")
            op_phi = coef.real[:, None] * _apply_pauli_rows(phi, np.atleast_1d(px), np.atleast_1d(pz))
            ev = np.einsum("rb,rb->r", np.conj(phi), op_phi).real
            s = np.where(rng.random(nb) < 0.5 * (1.0 + ev), 1.0, -1.0)
            phi = 0.5 * (phi + s[:, None] * op_phi)
            phi /= np.linalg.norm(phi, axis=1, keepdims=True)
        cov = matchgate_covered(masks, mt)                           # (nb, terms)
        vals = np.zeros(cov.shape)
        r, j = np.nonzero(cov)
        step = max(1, (1 << 20) // psi.size)
        for s0 in range(0, r.size, step):
            rr, jj = r[s0:s0 + step], j[s0:s0 + step]
            pj = _apply_pauli_rows(phi[rr], xs[jj], zs[jj])
            v = np.einsum("rb,rb->r", np.conj(phi[rr]), pj)
            max_dev = max(max_dev, float(np.max(np.abs(np.abs(v) - 1.0), initial=0.0)))
            vals[rr, jj] = v.real
        x = vals @ wt + const[None, :]
        acc.update(x - mu[None, :])
        done += nb
    return acc.result("matchgate_shadow", mu, {"n_terms": int(nid.size), "max_eigenvalue_deviation": max_dev})


# =========================================================== public entries
def _shadow_variance(scheme: str, targets, state: QubitState, method: str, n_settings: int,
                     rel_stderr: float | None, min_settings: int, n_snapshots: int, batch: int | None,
                     seed, settings, max_pairs: float) -> VarianceResult:
    tg = _prepare(targets)
    _check_state(state, tg.n)
    if method == "auto":
        limit = _AUTO_MAX_PAIRS[scheme] if max_pairs is None else max_pairs
        method = "exact" if _pair_count(tg.without_identity().indptr) <= limit else "rb"
    if method == "exact":
        oracle = _Expectations(state)
        p, mu = _means(tg, oracle)
        fn = _pauli_shadow_exact if scheme == "pauli_shadow" else _matchgate_exact
        var, npairs = fn(tg, oracle, p)
        return VarianceResult(scheme=scheme, method="exact", total=float(var.sum()), per_target=var,
                              expectation=mu, info={"n_terms": int(tg.B - (tg.id_col >= 0)),
                                                    "n_pairs": int(npairs)})
    if method == "rb":
        return _shadow_rb(tg, state, scheme, n_settings=int(n_settings), rel_stderr=rel_stderr,
                          min_settings=int(min_settings), batch=int(batch or 256), seed=seed,
                          settings=settings)
    if method == "mc":
        mu = _means(tg, _Expectations(state))[1]
        fn = _pauli_shadow_mc if scheme == "pauli_shadow" else _matchgate_mc
        return fn(tg, state, mu, int(n_snapshots), int(batch or 4096), _rng(seed))
    raise ValueError(f"method must be 'auto', 'exact', 'rb' or 'mc', got {method!r}")


def pauli_shadow_variance(targets, state: QubitState, *, method: str = "auto", n_settings: int = 20000,
                          rel_stderr: float | None = None, min_settings: int = 1000,
                          n_snapshots: int = 100000, batch: int | None = None, seed=None,
                          settings: Iterable | None = None,
                          max_pairs: float | None = None) -> VarianceResult:
    """Per-snapshot variance of random-local-Clifford (Pauli) shadows (module docstring).

    Methods:
        "exact": pair sum over qubit-wise compatible term pairs of each target,
            E[x_j x_k] = 3^{|supp_j n supp_k|} <P_j P_k>. ``info["n_pairs"]`` counts them.
        "rb": Rao-Blackwellized Monte Carlo over ``n_settings`` random local bases
            (stops early once stderr <= rel_stderr * total, after ``min_settings``),
            or over the explicit ``settings`` (iterable of (bx, bz) masks; all 3^n
            bases give the exact value). ``stderr``/``per_target_stderr`` are the
            standard errors over settings.
        "mc": ``n_snapshots`` simulated snapshots (random bases, outcomes sampled
            from the rotated dense state); sample variances with standard errors.
        "auto" (default): "exact" if the number of pairs (sum over targets of
            m(m+1)/2 for m terms) is at most ``max_pairs``, else "rb".
    """
    return _shadow_variance("pauli_shadow", targets, state, method, n_settings, rel_stderr, min_settings,
                            n_snapshots, batch, seed, settings, max_pairs)


def matchgate_shadow_variance(targets, state: QubitState, *, method: str = "auto", n_settings: int = 20000,
                              rel_stderr: float | None = None, min_settings: int = 1000,
                              n_snapshots: int = 100000, batch: int | None = None, seed=None,
                              settings: Iterable | None = None,
                              max_pairs: float | None = None) -> VarianceResult:
    """Per-snapshot variance of matchgate (fermionic Gaussian) shadows (module docstring).

    Methods as for :func:`pauli_shadow_variance`, with settings = perfect
    matchings ((n, 2) pair arrays; :func:`perfect_matchings` lists all) and
    "exact" = the Majorana-form pair sum over commuting monomial pairs,
    E[conj(x_S) x_S'] = P_both(S, S') <gamma_S^dag gamma_S'> / (lambda_S lambda_S').
    "mc" measures the pair operators of each random matching by sequential
    projective sampling on the dense state; ``info["max_eigenvalue_deviation"]``
    is max | |<phi|P_j|phi>| - 1 | over the covered terms (a consistency check
    of the coverage test, which should be round-off).
    """
    return _shadow_variance("matchgate_shadow", targets, state, method, n_settings, rel_stderr,
                            min_settings, n_snapshots, batch, seed, settings, max_pairs)


def scheme_variance(targets, state: QubitState, scheme: str, **options) -> VarianceResult:
    """Dispatch to :func:`qwc_variance`, :func:`pauli_shadow_variance` or :func:`matchgate_shadow_variance`."""
    fns = {"qwc": qwc_variance, "pauli_shadow": pauli_shadow_variance,
           "matchgate_shadow": matchgate_shadow_variance}
    if scheme not in fns:
        raise ValueError(f"scheme must be one of {SCHEMES} (DF: df_variance), got {scheme!r}")
    return fns[scheme](targets, state, **options)


def combine_independent(results: Sequence[VarianceResult]) -> VarianceResult:
    """Optimal shot allocation across independent experiments whose estimates are summed.

    For experiments l with per-shot variances v_{l,e} (e.g. the DF-fragment
    auxiliary states, A_hat_e = sum_l (-Q_hat_e^{(l)} / delta)), a fraction f_l of the
    shots on experiment l gives Var_e = sum_l v_{l,e} / f_l per shot. The optimum
    f_l = sqrt(V_l) / sum sqrt(V_m), V_l = sum_e v_{l,e}, gives
    ``total`` = (sum_l sqrt(V_l))^2; ``uniform_total`` = L sum_l V_l. Expectations
    add up. Standard errors (if every input has one) combine by the delta method.
    """
    results = list(results)
    if not results:
        raise ValueError("need at least one result")
    k = results[0].n_targets
    if any(r.n_targets != k for r in results):
        raise ValueError("all results must have the same number of targets")
    v = np.array([r.total for r in results], dtype=float)
    sq = np.sqrt(np.clip(v, 0.0, None))
    f = sq / sq.sum() if sq.sum() > 0 else np.full(v.size, 1.0 / v.size)
    inv_f = np.divide(1.0, f, out=np.zeros(v.size), where=f > 0)
    per = sum(r.per_target * w for r, w in zip(results, inv_f))
    per_u = v.size * sum(r.per_target for r in results)
    stderr = None
    if all(r.stderr is not None for r in results):
        se = np.array([r.stderr for r in results])
        rel = np.divide(se, sq, out=np.zeros(v.size), where=sq > 0)
        stderr = float(sq.sum() * np.sqrt(np.sum(rel ** 2)))
    methods = sorted({r.method for r in results})
    return VarianceResult(scheme="combined", method=methods[0] if len(methods) == 1 else "+".join(methods),
                          total=float(sq.sum() ** 2), per_target=np.asarray(per, dtype=float),
                          expectation=sum(r.expectation for r in results), n_settings=None, allocation=f,
                          uniform_total=float(v.size * v.sum()), per_target_uniform=np.asarray(per_u, float),
                          setting_variances=v, stderr=stderr, info={"parts": results})


# ============================================================ DF basis rotation
def _df_ci(active, state):
    """CI array of a CI vector or a QubitState in the active space's (n_alpha, n_beta) sector."""
    from ..fci_ops import FCISpace

    space = FCISpace.of(active)
    if isinstance(state, QubitState):
        if state.mapping is None:
            raise ValueError("a QubitState needs its mapping to be converted to a CI vector")
        ci = state.to_ci(active.norb, active.nelec, state.mapping)
        lost = abs(float(np.vdot(state.amplitudes, state.amplitudes).real) - float(np.vdot(ci, ci).real))
        if lost > _NORM_ATOL:
            raise ValueError("the state has weight outside the (n_alpha, n_beta) sector of the active space")
    else:
        ci = np.asarray(state)
        if ci.size != space.size:
            raise ValueError(f"CI vector has {ci.size} entries, expected {space.size}")
    ci = ci.reshape(space.shape)
    nrm2 = float(np.vdot(ci, ci).real)
    if abs(nrm2 - 1.0) > _NORM_ATOL:
        raise ValueError(f"the state must be normalized (<psi|psi> = {nrm2:.12g})")
    return space, ci


def _df_fragments(active, factorization, df_options):
    from ..factor import double_factorize

    if factorization is None:
        factorization = double_factorize(active.eri, **df_options)
    elif df_options:
        raise ValueError("pass either a factorization or double_factorize options, not both")
    eri_df = factorization.to_eri()
    hp = np.asarray(active.h1) - 0.5 * np.einsum("prrq->pq", eri_df)
    return factorization, hp, eri_df


def df_variance(active, state, *, factorization=None, **df_options) -> VarianceResult:
    """Per-shot energy variance of DF basis-rotation grouping (module docstring).

    Args:
        active: the :class:`~thc_engine_cqe.integrals.ActiveSpace`.
        state: normalized CI array (real or complex) or a QubitState with a mapping
            (converted with ``to_ci``; it must lie in the (n_alpha, n_beta) sector).
        factorization: a ``DoubleFactorization`` to use; otherwise
            ``factor.double_factorize(active.eri, **df_options)`` (exact by default;
            pass e.g. ``n_leaves`` or ``eig_threshold`` to truncate).

    Returns a VarianceResult with ``n_settings`` = L + 1 (fragment 0 is the
    one-body term), ``setting_variances`` = Var_l, ``allocation`` = the optimal
    f_l, ``expectation`` = [<H>] (exact Hamiltonian), and ``info``:
    ``energy_model`` = <H_DF>, ``bias`` = <H_DF> - <H>, ``fragment_means`` (<H_0>,
    <H_l> = <F_l^2>/2), ``n_leaves``, ``factorization``.
    """
    from ..fci_ops import Hamiltonian, one_body

    space, ci = _df_ci(active, state)
    fac, hp, _ = _df_fragments(active, factorization, df_options)
    leaves = fac.leaves()
    h0 = one_body(space, hp, ci)
    means = [float(np.vdot(ci, h0).real)]
    var = [float(np.vdot(h0, h0).real) - means[0] ** 2]
    for leaf in leaves:
        f1 = one_body(space, leaf, ci)
        f2 = one_body(space, leaf, f1)
        f2m = float(np.vdot(f1, f1).real)               # <F^2> = ||F psi||^2 (F Hermitian)
        means.append(0.5 * f2m)
        var.append(0.25 * (float(np.vdot(f2, f2).real) - f2m ** 2))
    var = np.array(var)
    sq = np.sqrt(np.clip(var, 0.0, None))
    nset = var.size
    f = sq / sq.sum() if sq.sum() > 0 else np.full(nset, 1.0 / nset)
    e_model = float(active.ecore + sum(means))
    e_exact = Hamiltonian(active).energy(ci)
    info = {"energy_model": e_model, "bias": e_model - e_exact, "fragment_means": np.array(means),
            "n_leaves": int(fac.n_leaves), "factorization": fac}
    return VarianceResult(scheme="df", method="exact", total=float(sq.sum() ** 2),
                          per_target=np.array([float(sq.sum() ** 2)]), expectation=np.array([e_exact]),
                          n_settings=nset, allocation=f, uniform_total=float(nset * var.sum()),
                          per_target_uniform=np.array([float(nset * var.sum())]), setting_variances=var,
                          info=info)


def _determinant_occupations(norb: int, nelec) -> np.ndarray:
    """(n_alpha_strings * n_beta_strings, norb) spin-summed occupations in ravelled ci order."""
    from pyscf.fci import cistring

    def occ(ne):
        strs = np.asarray(cistring.make_strings(range(norb), ne), dtype=np.int64)
        return ((strs[:, None] >> np.arange(norb)[None, :]) & 1).astype(np.float64)

    oa, ob = occ(int(nelec[0])), occ(int(nelec[1]))
    return (oa[:, None, :] + ob[None, :, :]).reshape(-1, norb)


def df_variance_mc(active, state, n_shots: int, *, factorization=None, seed=None,
                   **df_options) -> VarianceResult:
    """Monte Carlo validation of :func:`df_variance` by sampling in the rotated bases.

    For every fragment the CI vector is rotated into the fragment's orbital
    basis (``pyscf.fci.addons.transform_ci`` with the eigenvectors of h' or of
    L_l), ``n_shots`` determinants are sampled from |ci'|^2 and the diagonal
    operator is evaluated: sum_k eps_k n_k (fragment 0) or 1/2 (sum_k mu_lk n_k)^2.
    ``setting_variances`` are the sample variances, ``total`` = (sum sqrt V_hat_l)^2
    with its delta-method ``stderr``, ``uniform_total`` = (L + 1) sum V_hat_l with
    ``uniform_stderr``. ``info``: ``setting_stderr``, ``fragment_sample_means``
    and ``fragment_means_exact_distribution`` (the means over the full rotated
    distribution, a check of the rotation convention), ``sample_mean`` (the
    energy estimate of a uniform campaign) and its ``sample_mean_stderr``.
    """
    from pyscf.fci import addons

    space, ci = _df_ci(active, state)
    fac, hp, _ = _df_fragments(active, factorization, df_options)
    rng = _rng(seed)
    n_shots = int(n_shots)
    occ = _determinant_occupations(active.norb, active.nelec)
    eps, u0 = np.linalg.eigh(hp)
    frags = [(u0, eps, False)] + [(u, mu, True) for mu, u in zip(fac.eigenvalues, fac.eigenvectors)]
    v_hat, v_se, means_s, means_d, mean_var = [], [], [], [], []
    for u, ev, square in frags:
        n_k = u.shape[1]
        if n_k < active.norb:      # truncated leaf: complete the basis (the extra orbitals have mu = 0)
            q, _ = np.linalg.qr(np.concatenate([u, np.eye(active.norb)], axis=1))
            comp = q[:, n_k:active.norb]
            comp -= u @ (u.T @ comp)
            comp, _ = np.linalg.qr(comp)
            u = np.concatenate([u, comp], axis=1)
            ev = np.concatenate([ev, np.zeros(active.norb - n_k)])
        rot = addons.transform_ci(ci, active.nelec, u)
        prob = (np.abs(rot) ** 2).ravel()
        lin = occ @ ev
        val = 0.5 * lin ** 2 if square else lin
        means_d.append(float(prob @ val) / float(prob.sum()))
        idx = rng.choice(prob.size, size=n_shots, p=prob / prob.sum())
        x = val[idx]
        d = x - x.mean()
        v = float(d @ d) / (n_shots - 1)
        v_hat.append(v)
        v_se.append(float(np.sqrt(max(np.mean(d ** 4) - v ** 2, 0.0) / n_shots)))
        means_s.append(float(x.mean()))
        mean_var.append(v / n_shots)
    v_hat, v_se = np.array(v_hat), np.array(v_se)
    sq = np.sqrt(np.clip(v_hat, 0.0, None))
    rel = np.divide(v_se, sq, out=np.zeros(sq.size), where=sq > 0)
    nset = v_hat.size
    f = sq / sq.sum()
    from ..fci_ops import Hamiltonian

    e_exact = Hamiltonian(active).energy(ci)
    info = {"setting_stderr": v_se, "fragment_sample_means": np.array(means_s),
            "fragment_means_exact_distribution": np.array(means_d),
            "sample_mean": float(active.ecore + sum(means_s)),
            "sample_mean_stderr": float(np.sqrt(sum(mean_var))), "factorization": fac}
    return VarianceResult(scheme="df", method="mc", total=float(sq.sum() ** 2),
                          per_target=np.array([float(sq.sum() ** 2)]), expectation=np.array([e_exact]),
                          n_settings=nset, allocation=f, uniform_total=float(nset * v_hat.sum()),
                          per_target_uniform=np.array([float(nset * v_hat.sum())]), setting_variances=v_hat,
                          stderr=float(sq.sum() * np.sqrt(np.sum(rel ** 2))),
                          per_target_stderr=None, uniform_stderr=float(nset * np.sqrt(np.sum(v_se ** 2))),
                          n_samples=n_shots * nset, info=info)
