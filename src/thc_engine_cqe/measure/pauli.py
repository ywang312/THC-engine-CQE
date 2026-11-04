"""Pauli strings, the Jordan-Wigner map, Majorana monomials and sparse qubit states.

This module is the qubit-level foundation of the M2 measurement layer. Target
operators (energy, residual estimators), measurement schemes (QWC grouping, DF
basis rotation, Pauli and matchgate shadows) and their exact variances are all
built from the objects defined here. Every convention below is checked in
``tests/test_pauli.py`` against dense matrices, against the Fock-space oracle
:class:`thc_engine_cqe.testing.fock.FockSpace`, against PySCF and against
OpenFermion.

Spin orbitals and qubits
------------------------
N spatial orbitals and n = 2N qubits. A spin orbital is (p, sigma) with
sigma = 0 (alpha) or 1 (beta). A :class:`QubitMapping` fixes the qubit of each
spin orbital:

* ``order="blocked"`` (default): qubit(p, alpha) = p, qubit(p, beta) = N + p.
  This is the mode order of ``testing.fock.FockSpace``.
* ``order="interleaved"``: qubit(p, alpha) = 2p, qubit(p, beta) = 2p + 1.
  This is OpenFermion's spin-orbital order.

The *mode index* of (p, sigma) is p + sigma * N, independent of the mapping
(``QubitMapping.qubit_of_mode`` maps it to the qubit).

Basis states and dense vectors
------------------------------
A basis index b is an unsigned 64-bit bitmask in which bit k is the occupation of qubit k:

    |b> = (a+_0)^{b_0} (a+_1)^{b_1} ... (a+_{n-1})^{b_{n-1}} |vac>

with the creation operators in ascending qubit order. Dense vectors and matrices
(``to_dense``) are indexed by b, so qubit 0 is the least significant bit.
OpenFermion's ``get_sparse_operator`` uses the opposite order: its index is b
with the n bits reversed.

Jordan-Wigner
-------------
a_k = Z_0 ... Z_{k-1} (X_k + i Y_k) / 2, so a_k|b> = (-1)^{sum_{j<k} b_j} |b - e_k>
when b_k = 1. The ladder operators of (p, sigma) are those of qubit(p, sigma). In
Pauli-string form (next section), with e_k = 1 << k, e_<k = e_k - 1 and
e_<=k = 2 e_k - 1:

    a_k  = ( P(e_k, e_<k) + i P(e_k, e_<=k) ) / 2,
    a+_k = ( P(e_k, e_<k) - i P(e_k, e_<=k) ) / 2.

Pauli strings
-------------
    P(x, z) = i^{|x & z|}  prod_k X_k^{x_k} Z_k^{z_k}

Here x and z are uint64 bitmasks and |.| is the popcount. On one qubit,
P(1,0) = X, P(0,1) = Z and P(1,1) = iXZ = Y. So P(x, z) is the tensor product of
X, Y and Z on the qubits in x | z, and it is Hermitian. The defining relations
(``pauli_product``, ``pauli_commutes``, ``qubitwise_commutes``) are:

* Action: P(x, z)|b> = i^{|x&z|} (-1)^{|b&z|} |b ^ x>.
* Product: P(x1,z1) P(x2,z2) = i^phi P(x3, z3) with x3 = x1^x2, z3 = z1^z2 and
  phi = |x1&z1| + |x2&z2| - |x3&z3| + 2|z1&x2| (mod 4).
* Commutation: P1 P2 = (-1)^{|x1&z2| + |z1&x2|} P2 P1.
* Qubit-wise commutation: on every qubit in both supports the two factors are equal.
* Expectation: <psi|P(x,z)|psi> = i^{|x&z|} sum_b conj(psi[b^x]) (-1)^{|b&z|} psi[b].

Majorana operators (qubit order)
--------------------------------
    gamma_{2k}   = a_k + a+_k      = Z_<k X_k = P(e_k, e_<k),
    gamma_{2k+1} = i (a+_k - a_k)  = Z_<k Y_k = P(e_k, e_<=k).

A monomial gamma_S, with S a subset of {0, ..., 2n-1} given as a bitmask m, is the
product in ascending index order. Write a_k = m_{2k}, b_k = m_{2k+1} and
c_k = XOR_{j>k} (a_j ^ b_j). On qubit k the monomial acts as X^{a_k} Y^{b_k} Z^{c_k},
which gives the GF(2) map and the phase:

    gamma_S = i^E P(x, z),  x_k = m_{2k} ^ m_{2k+1},  z_k = m_{2k+1} ^ XOR_{j>k} x_j,
    E = |~a & b & c| + |a & b| - |a & ~b & c|  (mod 4).

The map from monomials to Pauli strings is a bijection, up to the phase
(``majorana_to_pauli``, ``pauli_to_majorana``). Products of monomials follow
gamma_S gamma_T = (-1)^{|T & c(S)|} gamma_{S^T}, with c(S)_t = XOR_{s>t} S_s
(``majorana_product``). Masks are uint64, so the Majorana functions need n <= 32.

Fermionic operators (docs/conventions.md)
-----------------------------------------
* E_pq = sum_sigma a+_{p sigma} a_{q sigma}  (``jw_one_body``).
* e_{pq,rs} = sum_{sigma tau} a+_{p sigma} a+_{r tau} a_{s tau} a_{q sigma}
  = E_pq E_rs - delta_qr E_ps  (``jw_two_body``).
* H = ecore + sum h_pq E_pq + 1/2 sum (pq|rs) e_{pq,rs}  (``jw_hamiltonian``).
* Mazziotti's 2-RDM element 2D^{ij}_{kl} = <a+_i a+_j a_l a_k> is the expectation
  of ``jw_rdm2_element(mapping, i, j, k, l)``.

PySCF CI vectors
----------------
``ci[ia, ib]`` multiplies the determinant made of alpha string ia and beta string
ib, in ``pyscf.fci.cistring`` order. :meth:`QubitState.from_ci` maps the CI vector to

    sum ci[ia, ib] (prod_{p in alpha, asc} a+_{p alpha}) (prod_{q in beta, asc} a+_{q beta}) |vac>.

In blocked order this is the basis index b = alpha_bits | (beta_bits << N) with
sign +1, exactly as ``FockSpace.from_ci``. In any other order, the
determinant-dependent sign of reordering the creation operators into ascending
qubit order is applied. For interleaved order it is
(-1)^{#{(p in alpha, q in beta) : q < p}}. Both orders therefore hold the same
fermionic state.

PySCF's own determinant puts the alpha creators, in descending orbital order, to
the left of the beta creators, also descending (see ``pyscf.fci.addons.cre_a`` and
``cre_b``). It differs from the state above by the global sign
(-1)^{na(na-1)/2 + nb(nb-1)/2}. Pass ``pyscf_sign=True`` to include this sign;
the JW ladder operators then act exactly like PySCF's cre/des functions.

PauliSum canonical form and tolerances
--------------------------------------
A :class:`PauliSum` is immutable. Its terms are unique (x, z) pairs, sorted by x
and then z (as unsigned integers), with nonzero coefficients. Duplicates are
summed with ``numpy.add.reduceat``, whose association order is unspecified; it is
not left to right (observed: 1 + 1e-16 - 1 sums to 1.1e-16, not 0). Which small
terms are removed depends on how the operator is made:

* Constructors and builders (``PauliSum(...)``, ``from_terms``,
  ``from_openfermion``, the ``jw_*`` functions) drop |c| <= tol after summing,
  with an absolute ``tol`` (default ``DEFAULT_TOL = 1e-14``). The cut removes the
  round-off of cancelling O(1) contributions and the Pauli terms generated by
  integrals that vanish only to round-off. ``simplify(tol)`` applies the same cut
  to an existing operator.
* Exact operations drop nothing but exact zeros (e.g. from underflow):
  multiplication and division by scalars (division by a real scalar is
  correctly rounded per component), negation, ``dagger``, ``identity(n, c)``
  and ``take``.
* Operations that can cancel (``+``, ``-``, ``PauliSum.sum``, ``@`` and
  ``multiply``, ``commutator``, ``anticommutator``) drop a coefficient
  c = sum_i t_i only when it vanishes within the round-off of its own
  contributions: |c| <= rtol * sum_i |t_i|, with ``rtol = DEFAULT_TOL`` by
  default. The t_i are the operand coefficients for sums and c_a c_b i^phi for
  products. The rule is scale invariant (operands rescaled by powers of two give
  exactly rescaled results), and a term with a single contribution is never
  dropped. ``hermitian_part`` (``antihermitian_part``) drops the real (imaginary)
  part of c only when it is zero or at most rtol |c|. ``rtol=0`` keeps every
  nonzero coefficient.

Because the terms are sorted by x, terms with the same x are contiguous. These
are the groups over which the expectation kernels vectorize (one partner lookup
per distinct x, then a vectorized sum over all z).

QubitState
----------
A :class:`QubitState` is the sparse vector sum_b psi[b]|b>, with sorted, unique
uint64 basis indices. It need not be normalized and may span several
particle-number sectors.

* ``expectations`` and ``expectation`` return the bare matrix element
  <psi|P|psi> of the formula above. Pass ``normalize=True`` to divide by
  <psi|psi> (a ValueError for the zero state).
* ``apply`` returns op|psi>. It removes the round-off of cancelling
  contributions relative to their own scale (see its docstring), so a
  number-conserving operator does not leak round-off amplitudes out of the
  sector, while small genuine amplitudes are kept (``apply`` of the identity is exact).
* ``+``, ``-`` and ``vdot`` raise a ValueError for two states whose mappings are
  both set and differ (the same basis index means different determinants).

Copies and pickling
-------------------
:class:`PauliSum`, :class:`QubitState` and :class:`QubitMapping` are immutable,
so ``copy.copy`` and ``copy.deepcopy`` return the object itself. Pickling (for
process pools and caches) stores the canonical arrays and restores them read-only,
without re-canonicalizing.

Efficiency
----------
All batch operations are vectorized. Temporaries are bounded by blocks of
2^22 elements:

* Products and commutators of PauliSums loop only over blocks of the shorter operand.
* The JW builders evaluate all ladder-operator products at once.
* ``QubitState.expectations`` performs one partner lookup per distinct x, then
  one sign-matrix product, or a fast Walsh-Hadamard transform when a group has
  many z values.

Measured timings are listed in tests/test_pauli.py. They are "observed" values
for one thread.
"""

from __future__ import annotations

import numbers
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "DEFAULT_TOL",
    "QubitMapping",
    "PauliSum",
    "QubitState",
    "pauli_product",
    "pauli_commutes",
    "qubitwise_commutes",
    "pauli_weight",
    "pauli_label",
    "majorana_to_pauli",
    "pauli_to_majorana",
    "majorana_product",
    "jw_ladder",
    "jw_product",
    "jw_products",
    "jw_one_body",
    "jw_two_body",
    "jw_hamiltonian",
    "jw_rdm2_element",
    "jw_rdm2_elements",
    "coefficient_matrix",
]

DEFAULT_TOL = 1e-14

_U1 = np.uint64(1)
_S32 = np.uint64(32)
_M32 = np.uint64(0xFFFFFFFF)
_ALL64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_IPOW = np.array([1.0, 1.0j, -1.0, -1.0j], dtype=np.complex128)
_CHUNK = 1 << 22            # elements per vectorized block (bounds temporary memory)
_WHT_MAX_QUBITS = 22        # largest n for the dense Walsh-Hadamard path (64 MB)
_WHT_FACTOR = 2.0           # use the WHT when (#points * #queries) > factor * n * 2^n
_INDEX_MAP_MAX_QUBITS = 24  # largest n for the dense basis-position lookup table


# =============================================================== bit utilities
def _as_u64(a) -> np.ndarray:
    """Array of non-negative integer bitmasks as uint64 (accepts Python ints >= 2**63)."""
    arr = np.asarray(a)
    if arr.dtype == np.uint64:
        return arr
    if arr.size == 0:   # e.g. [] or (), which numpy types as float64
        return np.zeros(arr.shape, dtype=np.uint64)
    if arr.dtype.kind == "b":
        return arr.astype(np.uint64)
    if arr.dtype.kind in "iu":
        if arr.dtype.kind == "i" and arr.size and int(arr.min()) < 0:
            raise ValueError("bitmasks must be non-negative")
        return arr.astype(np.uint64)
    if arr.dtype == object:
        return np.array(a, dtype=np.uint64)
    raise TypeError(f"bitmasks must be integers, got dtype {arr.dtype}")


def _low_mask(n: int) -> np.uint64:
    return _ALL64 if n >= 64 else np.uint64((1 << n) - 1)


def _check_n(n) -> int:
    n = int(n)
    if not 0 <= n <= 64:
        raise ValueError(f"n_qubits must be in [0, 64], got {n}")
    return n


def _as_scalar(value) -> complex | None:
    """complex(value) for a number or a 0-d numeric numpy array/scalar; None otherwise."""
    if isinstance(value, numbers.Number):
        return complex(value)
    if (isinstance(value, (np.ndarray, np.generic)) and np.ndim(value) == 0
            and value.dtype.kind in "biufc"):
        return complex(value)
    return None


def _as_int_array(a, what: str) -> np.ndarray:
    """int64 array of integer (or boolean) input; floats are rejected instead of truncated."""
    arr = np.asarray(a)
    if arr.size and arr.dtype.kind not in "biu":
        raise TypeError(f"{what} must be integers, got dtype {arr.dtype}")
    return arr.astype(np.int64)


def _check_bits(n: int, *arrays: np.ndarray) -> None:
    if n >= 64:
        return
    hi = ~_low_mask(n)
    for a in arrays:
        if a.size and np.any(a & hi):
            raise ValueError(f"bitmask has bits outside the {n} qubits")


def _popcount(a: np.ndarray) -> np.ndarray:
    return np.bitwise_count(a)


def _ipow_phase(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """i^{|x & z|} (the phase in the definition of P(x, z))."""
    return _IPOW[_popcount(x & z) & 3]


def _parity_sign(a: np.ndarray) -> np.ndarray:
    """(-1)^{popcount(a)} as float64."""
    return 1.0 - 2.0 * (_popcount(a) & 1)


def _compress_even_bits(m: np.ndarray) -> np.ndarray:
    """Bits 0, 2, 4, ... of m moved to bits 0, 1, 2, ..."""
    m = m & np.uint64(0x5555555555555555)
    m = (m | (m >> np.uint64(1))) & np.uint64(0x3333333333333333)
    m = (m | (m >> np.uint64(2))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    m = (m | (m >> np.uint64(4))) & np.uint64(0x00FF00FF00FF00FF)
    m = (m | (m >> np.uint64(8))) & np.uint64(0x0000FFFF0000FFFF)
    m = (m | (m >> np.uint64(16))) & np.uint64(0x00000000FFFFFFFF)
    return m


def _spread_bits(x: np.ndarray) -> np.ndarray:
    """Bits 0, 1, 2, ... of x (< 2^32) moved to bits 0, 2, 4, ... (inverse of the above)."""
    x = x & np.uint64(0x00000000FFFFFFFF)
    x = (x | (x << np.uint64(16))) & np.uint64(0x0000FFFF0000FFFF)
    x = (x | (x << np.uint64(8))) & np.uint64(0x00FF00FF00FF00FF)
    x = (x | (x << np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    x = (x | (x << np.uint64(2))) & np.uint64(0x3333333333333333)
    x = (x | (x << np.uint64(1))) & np.uint64(0x5555555555555555)
    return x


def _suffix_parity(x: np.ndarray) -> np.ndarray:
    """Bit k of the result is XOR_{j>k} x_j."""
    t = x >> np.uint64(1)
    for s in (1, 2, 4, 8, 16, 32):
        t = t ^ (t >> np.uint64(s))
    return t


def _scalar_or_array(arr: np.ndarray, shape: tuple):
    arr = arr.reshape(shape)
    return arr[()] if shape == () else arr


# ============================================================ Pauli products
def _pp_raw(x1, z1, x2, z2):
    """Unchecked broadcasting core of :func:`pauli_product`; phase as uint8 (mod 4).

    The uint8 arithmetic wraps modulo 256, which preserves the value modulo 4.
    Inputs must be arrays (not numpy scalars) to avoid scalar-overflow warnings.
    """
    x3 = x1 ^ x2
    z3 = z1 ^ z2
    phi = (_popcount(x1 & z1) + _popcount(x2 & z2) - _popcount(x3 & z3)
           + 2 * _popcount(z1 & x2)) & 3
    return phi, x3, z3


def pauli_product(x1, z1, x2, z2):
    """P(x1, z1) P(x2, z2) = i^phi P(x3, z3), vectorized with numpy broadcasting.

    Returns ``(phi, x3, z3)`` with phi in {0, 1, 2, 3} (int64), x3 = x1 ^ x2 and
    z3 = z1 ^ z2 (uint64), where phi = |x1&z1| + |x2&z2| - |x3&z3| + 2|z1&x2| mod 4.
    """
    arrs = np.broadcast_arrays(*(_as_u64(a) for a in (x1, z1, x2, z2)))
    shape = arrs[0].shape
    a = [np.atleast_1d(v).ravel() for v in arrs]
    phi, x3, z3 = _pp_raw(*a)
    return (_scalar_or_array(phi.astype(np.int64), shape), _scalar_or_array(x3, shape),
            _scalar_or_array(z3, shape))


def pauli_commutes(x1, z1, x2, z2):
    """True where P(x1, z1) and P(x2, z2) commute: |x1&z2| + |z1&x2| is even."""
    arrs = np.broadcast_arrays(*(_as_u64(a) for a in (x1, z1, x2, z2)))
    shape = arrs[0].shape
    a = [np.atleast_1d(v).ravel() for v in arrs]
    out = ((_popcount(a[0] & a[3]) + _popcount(a[1] & a[2])) & 1) == 0
    return _scalar_or_array(out, shape)


def qubitwise_commutes(x1, z1, x2, z2):
    """True where the strings agree on every qubit that both act on (QWC compatibility)."""
    arrs = np.broadcast_arrays(*(_as_u64(a) for a in (x1, z1, x2, z2)))
    shape = arrs[0].shape
    x1, z1, x2, z2 = (np.atleast_1d(v).ravel() for v in arrs)
    both = (x1 | z1) & (x2 | z2)
    out = (((x1 ^ x2) | (z1 ^ z2)) & both) == 0
    return _scalar_or_array(out, shape)


def pauli_weight(x, z):
    """Number of non-identity qubits, |x | z| (int64, vectorized)."""
    xa, za = np.broadcast_arrays(_as_u64(x), _as_u64(z))
    shape = xa.shape
    out = _popcount(np.atleast_1d(xa).ravel() | np.atleast_1d(za).ravel()).astype(np.int64)
    return _scalar_or_array(out, shape)


def pauli_label(x: int, z: int) -> str:
    """Sparse label in OpenFermion's QubitOperator syntax, e.g. ``"X0 Y3 Z5"``; ``"I"`` if x = z = 0."""
    x, z = int(x), int(z)
    parts = []
    k = 0
    s = x | z
    while s >> k:
        if (s >> k) & 1:
            xb, zb = (x >> k) & 1, (z >> k) & 1
            parts.append(("X" if not zb else "Y") if xb else "Z")
            parts[-1] += str(k)
        k += 1
    return " ".join(parts) if parts else "I"


def _parse_pauli(spec, n: int) -> tuple[int, int]:
    """(x, z) of a single Pauli string given as (x, z), a label string or OpenFermion term key."""
    letters = {"X": (1, 0), "Y": (1, 1), "Z": (0, 1)}
    if isinstance(spec, str):
        items = []
        for tok in spec.replace(",", " ").split():
            if tok.upper() == "I":
                continue
            items.append((int(tok[1:]), tok[0].upper()))
    elif (isinstance(spec, (tuple, list)) and len(spec) == 2
          and all(isinstance(v, (numbers.Integral, np.integer)) for v in spec)):
        x, z = int(spec[0]), int(spec[1])
        if x < 0 or z < 0 or (n < 64 and ((x >> n) or (z >> n))):
            raise ValueError(f"(x, z) = ({x}, {z}) does not fit in {n} qubits")
        return x, z
    else:
        items = [(int(q), str(c).upper()) for q, c in spec]
    x = z = 0
    seen = set()
    for q, c in items:
        if c not in letters:
            raise ValueError(f"unknown Pauli letter {c!r}")
        if q in seen:
            raise ValueError(f"qubit {q} appears twice in {spec!r}")
        if not 0 <= q < n:
            raise ValueError(f"qubit {q} out of range for {n} qubits")
        seen.add(q)
        xb, zb = letters[c]
        x |= xb << q
        z |= zb << q
    return x, z


# ============================================== canonical-form machinery
def _group_starts(*keys: np.ndarray) -> np.ndarray:
    """Start indices of runs of equal (key_0, key_1, ...) in sorted arrays."""
    size = keys[0].size
    flag = np.zeros(size, dtype=bool)
    if size:
        flag[0] = True
        for k in keys:
            flag[1:] |= k[1:] != k[:-1]
    return np.flatnonzero(flag)


def _sort_and_sum(n: int, xs: np.ndarray, zs: np.ndarray, cs: np.ndarray, ws: np.ndarray | None = None):
    """Sort by (x, z) and sum duplicates; nothing is dropped.

    Returns ``(ux, uz, uc, uw)``: the unique sorted keys, the summed coefficients and,
    if ``ws`` (one weight per input term) is given, the summed weights (else None).
    The sums use numpy's add.reduceat; their association order is unspecified.
    """
    if xs.size == 0:
        return (np.zeros(0, np.uint64), np.zeros(0, np.uint64), np.zeros(0, np.complex128),
                None if ws is None else np.zeros(0))
    if n <= 32:
        key = (xs << _S32) | zs
        order = np.argsort(key, kind="stable")
        ks = key[order]
        starts = _group_starts(ks)
        ux, uz = ks[starts] >> _S32, ks[starts] & _M32
    else:
        order = np.lexsort((zs, xs))
        sx, sz = xs[order], zs[order]
        starts = _group_starts(sx, sz)
        ux, uz = sx[starts], sz[starts]
    uc = np.add.reduceat(cs[order], starts)
    uw = None if ws is None else np.add.reduceat(ws[order], starts)
    return ux, uz, uc, uw


def _keep_mask(uc: np.ndarray, tol: float, rtol: float = 0.0, uw: np.ndarray | None = None) -> np.ndarray:
    """Terms to keep: |c| > tol and, if rtol > 0, |c| > rtol * w (w = sum of |contributions|)."""
    a = np.abs(uc)
    keep = a > tol
    if rtol > 0:
        keep &= a > rtol * uw
    return keep


def _canonicalize(n: int, xs: np.ndarray, zs: np.ndarray, cs: np.ndarray, tol: float,
                  rtol: float = 0.0, ws: np.ndarray | None = None):
    """Sort by (x, z), sum duplicates and drop the terms with |c| <= tol or |c| <= rtol * w.

    w is the sum over the term's contributions of ``ws`` (default |cs|), i.e. the
    scale of the round-off of the summed coefficient.
    """
    if rtol > 0 and ws is None:
        ws = np.abs(cs)
    ux, uz, uc, uw = _sort_and_sum(n, xs, zs, cs, ws if rtol > 0 else None)
    keep = _keep_mask(uc, tol, rtol, uw)
    return ux[keep], uz[keep], uc[keep]


def _canonicalize_rows(n: int, rows: np.ndarray, n_rows: int, xs: np.ndarray, zs: np.ndarray,
                       cs: np.ndarray, tol: float) -> list["PauliSum"]:
    """Canonicalize many operators at once (term t belongs to operator rows[t]; |c| <= tol dropped).

    Terms are sorted by (row, x, z). The current callers (ladder products) have one x
    per row, but the order is correct for any input (tested directly).
    """
    if xs.size == 0:
        return [PauliSum.zero(n) for _ in range(n_rows)]
    if n <= 32:
        key = (xs << _S32) | zs
        order = np.lexsort((key, rows))
        sk, sr = key[order], rows[order]
        starts = _group_starts(sr, sk)
        ux, uz = sk[starts] >> _S32, sk[starts] & _M32
    else:
        order = np.lexsort((zs, xs, rows))
        sx, sz, sr = xs[order], zs[order], rows[order]
        starts = _group_starts(sr, sx, sz)
        ux, uz = sx[starts], sz[starts]
    ur = sr[starts]
    uc = np.add.reduceat(cs[order], starts)
    keep = np.abs(uc) > tol
    ux, uz, uc, ur = ux[keep], uz[keep], uc[keep], ur[keep]
    bounds = np.searchsorted(ur, np.arange(n_rows + 1))
    return [PauliSum._from_canonical(n, ux[a:b], uz[a:b], uc[a:b])
            for a, b in zip(bounds[:-1], bounds[1:])]


# ================================================= character-sum kernels
def _fwht_inplace(a: np.ndarray) -> np.ndarray:
    """Unnormalized Walsh-Hadamard transform: a[z] <- sum_b a[b] (-1)^{|b & z|}."""
    size = a.shape[-1]
    h = 1
    while h < size:
        v = a.reshape(-1, 2, h)
        tmp = v[:, 0, :].copy()
        v[:, 0, :] += v[:, 1, :]
        np.subtract(tmp, v[:, 1, :], out=v[:, 1, :])
        h *= 2
    return a


def _character_sum(points: np.ndarray, weights: np.ndarray, queries: np.ndarray, n: int) -> np.ndarray:
    """out[i] = sum_j weights[j] (-1)^{|points[j] & queries[i]|} (complex).

    Uses a dense fast Walsh-Hadamard transform when that is cheaper than the
    explicit (#queries x #points) sign matrix, otherwise blocked sign matrices.
    """
    npts, nq = points.size, queries.size
    out = np.zeros(nq, dtype=np.complex128)
    if npts == 0 or nq == 0:
        return out
    if n <= _WHT_MAX_QUBITS and npts * nq > _WHT_FACTOR * max(n, 1) * (1 << n):
        dim = 1 << n
        pi = points.astype(np.intp)
        dense = (np.bincount(pi, weights=weights.real, minlength=dim)
                 + 1j * np.bincount(pi, weights=weights.imag, minlength=dim))
        _fwht_inplace(dense)
        return dense[queries.astype(np.intp)]
    w = np.empty((npts, 2))
    w[:, 0] = weights.real
    w[:, 1] = weights.imag
    if not np.any(w[:, 1]):
        w = w[:, :1]
    total = w.sum(axis=0)
    step = max(1, _CHUNK // npts)
    for s in range(0, nq, step):
        q = queries[s:s + step]
        par = _popcount(q[:, None] & points[None, :]) & 1
        if q.size <= 8:   # few queries: explicit +-1 matrix (observed faster)
            r = (1.0 - 2.0 * par) @ w
        else:             # sum = total - 2 * (sum over odd parity); one float cast
            r = total - 2.0 * (par.astype(np.float64) @ w)
        out[s:s + step] = r[:, 0] if r.shape[1] == 1 else r[:, 0] + 1j * r[:, 1]
    return out


def _sum_by_index(b: np.ndarray, a: np.ndarray, w: np.ndarray):
    """Sorted unique basis indices with the summed amplitudes and weights (nothing dropped)."""
    order = np.argsort(b, kind="stable")
    sb = b[order]
    starts = _group_starts(sb)
    return sb[starts], np.add.reduceat(a[order], starts), np.add.reduceat(w[order], starts)


def _x_group_bounds(xs: np.ndarray):
    """(starts, stops) of runs of equal x in a sorted array."""
    starts = _group_starts(xs)
    stops = np.append(starts[1:], xs.size).astype(starts.dtype)
    return starts, stops


# ================================================================ PauliSum
class PauliSum:
    """Immutable linear combination of Pauli strings, sum_t c_t P(x_t, z_t), in canonical form.

    Attributes (read-only numpy arrays):
        n_qubits: number of qubits n (0 <= n <= 64).
        xs, zs:   uint64[n_terms] bitmasks, unique (x, z) pairs sorted by x, then z.
        coeffs:   complex128[n_terms], nonzero (which small terms are cut: module docstring).

    ``PauliSum(n, xs, zs, coeffs, tol=DEFAULT_TOL)`` canonicalizes the input:
    duplicates are summed and the terms with |c| <= tol (absolute) are dropped.
    Arithmetic: ``+``, ``-`` (with PauliSums, or with scalars, which mean
    multiples of I), unary ``-``, multiplication and division by scalars (Python
    or numpy numbers, or 0-d arrays), and ``@`` for the operator product. ``*``
    between two PauliSums is not defined, to avoid confusion with an elementwise
    product. Which terms each operation drops is described in the module
    docstring (scalar scaling drops nothing; sums and products drop only
    coefficients that cancel within the round-off of their contributions).
    Division by a real scalar divides the real and imaginary parts separately,
    so it is correctly rounded (numpy's complex division multiplies by a
    reciprocal: 49 / 49 would give 0.9999999999999999).
    """

    __slots__ = ("_n", "_xs", "_zs", "_cs")
    __array_ufunc__ = None  # numpy scalars and 0-d arrays defer to our reflected operators

    def __init__(self, n_qubits: int, xs=(), zs=(), coeffs=(), *, tol: float = DEFAULT_TOL):
        n = _check_n(n_qubits)
        xs = np.atleast_1d(_as_u64(xs)).ravel()
        zs = np.atleast_1d(_as_u64(zs)).ravel()
        cs = np.atleast_1d(np.asarray(coeffs, dtype=np.complex128)).ravel()
        if cs.size == 1 and xs.size != 1:
            cs = np.full(xs.size, cs[0])
        if not xs.size == zs.size == cs.size:
            raise ValueError(f"xs, zs and coeffs must have equal lengths, got "
                             f"{xs.size}, {zs.size}, {cs.size}")
        _check_bits(n, xs, zs)
        self._set(n, *_canonicalize(n, xs, zs, cs, tol))

    def _set(self, n, xs, zs, cs) -> None:
        for a in (xs, zs, cs):
            a.setflags(write=False)
        object.__setattr__(self, "_n", n)
        object.__setattr__(self, "_xs", xs)
        object.__setattr__(self, "_zs", zs)
        object.__setattr__(self, "_cs", cs)

    def __setattr__(self, name, value):
        raise AttributeError("PauliSum is immutable")

    # immutable: copies are the object itself; pickles restore the canonical arrays
    def __reduce__(self):
        return (type(self)._from_canonical, (self._n, self._xs, self._zs, self._cs))

    def __copy__(self) -> "PauliSum":
        return self

    def __deepcopy__(self, memo) -> "PauliSum":
        return self

    @classmethod
    def _from_canonical(cls, n: int, xs: np.ndarray, zs: np.ndarray, cs: np.ndarray) -> "PauliSum":
        obj = object.__new__(cls)
        obj._set(n, np.ascontiguousarray(xs, dtype=np.uint64),
                 np.ascontiguousarray(zs, dtype=np.uint64),
                 np.ascontiguousarray(cs, dtype=np.complex128))
        return obj

    @classmethod
    def _canon(cls, n, xs, zs, cs, tol=DEFAULT_TOL, rtol=0.0, ws=None) -> "PauliSum":
        return cls._from_canonical(n, *_canonicalize(n, xs, zs, cs, tol, rtol, ws))

    def _filtered(self, keep: np.ndarray, cs: np.ndarray | None = None) -> "PauliSum":
        """Same keys (already canonical) restricted to ``keep``, optionally with new coefficients."""
        cs = self._cs if cs is None else cs
        if keep.all():
            return PauliSum._from_canonical(self._n, self._xs, self._zs, cs)
        return PauliSum._from_canonical(self._n, self._xs[keep], self._zs[keep], cs[keep])

    # ------------------------------------------------------------ constructors
    @classmethod
    def zero(cls, n: int) -> "PauliSum":
        n = _check_n(n)
        return cls._from_canonical(n, np.zeros(0, np.uint64), np.zeros(0, np.uint64),
                                   np.zeros(0, np.complex128))

    @classmethod
    def identity(cls, n: int, c: complex = 1.0) -> "PauliSum":
        """c times the identity on n qubits (exact: only c = 0 gives the zero operator)."""
        n = _check_n(n)
        c = _as_scalar(c)
        if c is None:
            raise TypeError("the identity coefficient must be a number")
        if c == 0:
            return cls.zero(n)
        return cls._from_canonical(n, np.zeros(1, np.uint64), np.zeros(1, np.uint64),
                                   np.array([c], dtype=np.complex128))

    @classmethod
    def from_terms(cls, n: int, terms: Iterable, *, tol: float = DEFAULT_TOL) -> "PauliSum":
        """Build sum_t c_t P_t from ``terms``, an iterable of ``(pauli, c)`` pairs.

        Each ``pauli`` is one of:

        * a pair of integer bitmasks ``(x, z)``;
        * a label string such as ``"X0 Y3 Z5"`` (OpenFermion's syntax; ``""`` or
          ``"I"`` is the identity);
        * a sequence of ``(qubit, letter)`` pairs, e.g. ``((0, "X"), (3, "Z"))``,
          which is the key format of ``openfermion.QubitOperator.terms``.

        A label is a tensor product of X, Y, Z (phase 1), i.e. P(x, z) itself.
        """
        n = _check_n(n)
        xs, zs, cs = [], [], []
        for spec, c in terms:
            x, z = _parse_pauli(spec, n)
            xs.append(x)
            zs.append(z)
            cs.append(complex(c))
        return cls(n, np.array(xs, dtype=np.uint64), np.array(zs, dtype=np.uint64),
                   np.array(cs, dtype=np.complex128), tol=tol)

    @classmethod
    def from_openfermion(cls, qubit_operator, n_qubits: int, *, tol: float = DEFAULT_TOL) -> "PauliSum":
        """Convert an ``openfermion.QubitOperator`` (same qubit labels)."""
        return cls.from_terms(n_qubits, qubit_operator.terms.items(), tol=tol)

    @staticmethod
    def sum(ops: Sequence["PauliSum"], *, rtol: float = DEFAULT_TOL) -> "PauliSum":
        """Sum of many PauliSums with a single canonicalization.

        A summed coefficient c is dropped only if |c| <= rtol * sum |contributions|
        (exact zeros always); ``rtol=0`` keeps every nonzero coefficient.
        """
        ops = list(ops)
        if not ops:
            raise ValueError("PauliSum.sum needs at least one operand")
        n = ops[0].n_qubits
        for op in ops:
            if not isinstance(op, PauliSum):
                raise TypeError(f"PauliSum.sum needs PauliSums, got {type(op).__name__}")
            if op.n_qubits != n:
                raise ValueError("all operands must have the same n_qubits")
        return PauliSum._canon(n, np.concatenate([o._xs for o in ops]),
                               np.concatenate([o._zs for o in ops]),
                               np.concatenate([o._cs for o in ops]), 0.0, rtol)

    # -------------------------------------------------------------- properties
    @property
    def n_qubits(self) -> int:
        return self._n

    @property
    def xs(self) -> np.ndarray:
        return self._xs

    @property
    def zs(self) -> np.ndarray:
        return self._zs

    @property
    def coeffs(self) -> np.ndarray:
        return self._cs

    @property
    def n_terms(self) -> int:
        return int(self._cs.size)

    def __len__(self) -> int:
        return self.n_terms

    def __repr__(self) -> str:
        shown = ", ".join(f"{c:.6g} [{pauli_label(x, z)}]"
                          for x, z, c in zip(self._xs[:4], self._zs[:4], self._cs[:4]))
        more = ", ..." if self.n_terms > 4 else ""
        return f"PauliSum(n_qubits={self._n}, n_terms={self.n_terms}: {shown}{more})"

    def terms(self) -> list[tuple[int, int, complex]]:
        """[(x, z, c), ...] as Python scalars."""
        return [(int(x), int(z), complex(c)) for x, z, c in zip(self._xs, self._zs, self._cs)]

    def labels(self) -> list[str]:
        return [pauli_label(x, z) for x, z in zip(self._xs, self._zs)]

    def to_openfermion(self):
        """``openfermion.QubitOperator`` with the same qubit labels."""
        import openfermion as of

        out = of.QubitOperator()
        for x, z, c in self.terms():
            key = []
            for k in range(self._n):
                xb, zb = (x >> k) & 1, (z >> k) & 1
                if xb or zb:
                    key.append((k, ("Y" if zb else "X") if xb else "Z"))
            out += of.QubitOperator(tuple(key), c)
        return out

    # ------------------------------------------------------------------ lookup
    def coefficient(self, x: int, z: int) -> complex:
        """Coefficient of P(x, z) (0 if absent)."""
        return complex(self.coefficients_of([x], [z])[0])

    def coefficients_of(self, xs, zs) -> np.ndarray:
        """Coefficients of the strings P(xs[i], zs[i]) (0 where absent); vectorized."""
        qx = np.atleast_1d(_as_u64(xs)).ravel()
        qz = np.atleast_1d(_as_u64(zs)).ravel()
        out = np.zeros(qx.size, dtype=np.complex128)
        if self.n_terms == 0 or qx.size == 0:
            return out
        if self._n <= 32 and not (np.any(qx >> _S32) or np.any(qz >> _S32)):
            keys = (self._xs << _S32) | self._zs
            qk = (qx << _S32) | qz
            pos = np.minimum(np.searchsorted(keys, qk), keys.size - 1)
            hit = keys[pos] == qk
        else:
            lo = np.searchsorted(self._xs, qx, side="left")
            hi = np.searchsorted(self._xs, qx, side="right")
            pos = np.zeros(qx.size, dtype=np.intp)
            hit = np.zeros(qx.size, dtype=bool)
            for i in np.flatnonzero(hi > lo):
                j = lo[i] + np.searchsorted(self._zs[lo[i]:hi[i]], qz[i])
                if j < hi[i] and self._zs[j] == qz[i]:
                    pos[i], hit[i] = j, True
        out[hit] = self._cs[pos[hit]]
        return out

    def take(self, indices) -> "PauliSum":
        """The sub-sum of the terms at ``indices`` (indices into xs/zs/coeffs)."""
        idx = np.unique(np.asarray(indices, dtype=np.intp))
        return PauliSum._from_canonical(self._n, self._xs[idx], self._zs[idx], self._cs[idx])

    # -------------------------------------------------------------- arithmetic
    def _check_same(self, other: "PauliSum") -> None:
        if other._n != self._n:
            raise ValueError(f"n_qubits mismatch: {self._n} vs {other._n}")

    def __add__(self, other):
        if isinstance(other, PauliSum):
            self._check_same(other)
            return PauliSum.sum([self, other])
        s = _as_scalar(other)
        if s is None:
            return NotImplemented
        if s == 0:
            return self
        return PauliSum.sum([self, PauliSum.identity(self._n, s)])

    __radd__ = __add__

    def __neg__(self) -> "PauliSum":
        return PauliSum._from_canonical(self._n, self._xs, self._zs, -self._cs)

    def __pos__(self) -> "PauliSum":
        return self

    def __sub__(self, other):
        if isinstance(other, PauliSum):
            return self + (-other)
        s = _as_scalar(other)
        if s is None:
            return NotImplemented
        return self + (-s)

    def __rsub__(self, other):
        s = _as_scalar(other)
        if s is None:
            return NotImplemented
        return (-self) + s

    def _scaled(self, cs: np.ndarray) -> "PauliSum":
        """Same keys with coefficients ``cs``; only exact zeros (underflow) are dropped."""
        return self._filtered(cs != 0, cs)

    def __mul__(self, other):
        if isinstance(other, PauliSum):
            raise TypeError("use '@' for the operator product of PauliSums")
        s = _as_scalar(other)
        if s is None:
            return NotImplemented
        if s == 0:
            return PauliSum.zero(self._n)
        return self._scaled(self._cs * s)

    __rmul__ = __mul__

    def __truediv__(self, other):
        s = _as_scalar(other)
        if s is None:
            return NotImplemented
        if s == 0:
            raise ZeroDivisionError("division of a PauliSum by zero")
        if s.imag == 0:   # numpy's complex division multiplies by a reciprocal; divide each part
            cs = np.empty_like(self._cs)
            cs.real = self._cs.real / s.real
            cs.imag = self._cs.imag / s.real
            return self._scaled(cs)
        return self._scaled(self._cs / s)

    def __matmul__(self, other):
        if isinstance(other, PauliSum):
            return self.multiply(other)
        return NotImplemented

    def _pair_terms(self, other: "PauliSum", select: str | None, rtol: float) -> "PauliSum":
        """Canonicalized sum over term pairs of t = c_a c_b i^phi P(x_a^x_b, z_a^z_b).

        select=None keeps all pairs (product), "anti" only anticommuting pairs,
        "comm" only commuting pairs; the latter two are doubled ([A, B], {A, B}).
        A summed coefficient is dropped if |sum t| <= rtol * sum |t| (exact zeros
        always). The loop runs over blocks of the shorter operand; the blocks keep
        every group (also exact zeros), so the result does not depend on the blocking
        beyond round-off.
        """
        self._check_same(other)
        n = self._n
        na, nb = self.n_terms, other.n_terms
        if na == 0 or nb == 0:
            return PauliSum.zero(n)
        factor = 1.0 if select is None else 2.0
        wa_all, wb_all = np.abs(self._cs), np.abs(other._cs)
        acc, acc_len = [], 0
        short_is_a = na <= nb
        n_short = na if short_is_a else nb
        n_long = nb if short_is_a else na
        step = max(1, _CHUNK // n_long)
        for s in range(0, n_short, step):
            if short_is_a:
                blk = slice(s, s + step)
                xa, za, ca, wa = (self._xs[blk, None], self._zs[blk, None], self._cs[blk, None],
                                  wa_all[blk, None])
                xb, zb, cb, wb = other._xs[None, :], other._zs[None, :], other._cs[None, :], wb_all[None, :]
            else:
                blk = slice(s, s + step)
                xa, za, ca, wa = self._xs[None, :], self._zs[None, :], self._cs[None, :], wa_all[None, :]
                xb, zb, cb, wb = (other._xs[blk, None], other._zs[blk, None], other._cs[blk, None],
                                  wb_all[blk, None])
            phi, x3, z3 = _pp_raw(xa, za, xb, zb)
            c3 = (ca * cb) * _IPOW[phi]
            w3 = wa * wb
            if select is not None:
                anti = ((_popcount(xa & zb) + _popcount(za & xb)) & 1).astype(bool)
                mask = anti if select == "anti" else ~anti
                x3, z3, c3, w3 = x3[mask], z3[mask], factor * c3[mask], factor * w3[mask]
            part = _sort_and_sum(n, x3.ravel(), z3.ravel(), c3.ravel(), w3.ravel())
            acc.append(part)
            acc_len += part[0].size
            if acc_len > 4 * _CHUNK and len(acc) > 1:
                acc = [_sort_and_sum(n, *(np.concatenate(v) for v in zip(*acc)))]
                acc_len = acc[0][0].size
        ux, uz, uc, uw = acc[0] if len(acc) == 1 else _sort_and_sum(n, *(np.concatenate(v) for v in zip(*acc)))
        keep = _keep_mask(uc, 0.0, rtol, uw)
        return PauliSum._from_canonical(n, ux[keep], uz[keep], uc[keep])

    def multiply(self, other: "PauliSum", *, rtol: float = DEFAULT_TOL) -> "PauliSum":
        """Operator product self @ other (vectorized over term pairs).

        A product coefficient c = sum c_a c_b i^phi is dropped only if
        |c| <= rtol * sum |c_a c_b| (exact zeros always; ``rtol=0`` keeps all others).
        """
        return self._pair_terms(other, None, rtol)

    def commutator(self, other: "PauliSum", *, rtol: float = DEFAULT_TOL) -> "PauliSum":
        """[self, other] = 2 sum over anticommuting term pairs of c_a c_b P_a P_b.

        Commuting pairs are skipped rather than cancelled (AB - BA), so they leave no
        round-off. ``rtol`` as in :meth:`multiply`.
        """
        return self._pair_terms(other, "anti", rtol)

    def anticommutator(self, other: "PauliSum", *, rtol: float = DEFAULT_TOL) -> "PauliSum":
        """{self, other} = 2 sum over commuting term pairs of c_a c_b P_a P_b (``rtol`` as in multiply)."""
        return self._pair_terms(other, "comm", rtol)

    def dagger(self) -> "PauliSum":
        """Adjoint: every P(x, z) is Hermitian, so the coefficients are conjugated."""
        return PauliSum._from_canonical(self._n, self._xs, self._zs, np.conj(self._cs))

    def hermitian_part(self, *, rtol: float = DEFAULT_TOL) -> "PauliSum":
        """(A + A^dagger) / 2: the real parts of the coefficients.

        A real part is dropped if |Re c| <= rtol |c| (round-off of an imaginary coefficient).
        """
        re = self._cs.real
        keep = (np.abs(re) > rtol * np.abs(self._cs)) & (re != 0)
        return self._filtered(keep, re.astype(np.complex128))

    def antihermitian_part(self, *, rtol: float = DEFAULT_TOL) -> "PauliSum":
        """(A - A^dagger) / 2: i times the imaginary parts (dropped if |Im c| <= rtol |c|)."""
        im = self._cs.imag
        keep = (np.abs(im) > rtol * np.abs(self._cs)) & (im != 0)
        return self._filtered(keep, 1j * im)

    def is_hermitian(self, atol: float = 1e-12) -> bool:
        return bool(np.all(np.abs(self._cs.imag) <= atol))

    def simplify(self, tol: float = DEFAULT_TOL) -> "PauliSum":
        """Drop the terms with |c| <= tol."""
        keep = np.abs(self._cs) > tol
        return PauliSum._from_canonical(self._n, self._xs[keep], self._zs[keep], self._cs[keep])

    def allclose(self, other: "PauliSum", atol: float = 1e-12) -> bool:
        """True if every coefficient of self - other has modulus <= atol."""
        self._check_same(other)
        d = PauliSum._canon(self._n, np.concatenate([self._xs, other._xs]),
                            np.concatenate([self._zs, other._zs]),
                            np.concatenate([self._cs, -other._cs]), 0.0)
        return bool(np.all(np.abs(d._cs) <= atol))

    # ------------------------------------------------------------- properties
    def weights(self) -> np.ndarray:
        """Number of non-identity qubits of each term (int64)."""
        return _popcount(self._xs | self._zs).astype(np.int64)

    def norm1(self, include_identity: bool = True) -> float:
        """sum_t |c_t| (optionally without the identity term)."""
        if include_identity:
            return float(np.sum(np.abs(self._cs)))
        keep = (self._xs | self._zs) != 0
        return float(np.sum(np.abs(self._cs[keep])))

    def identity_coefficient(self) -> complex:
        """Coefficient of I, i.e. tr(A) / 2^n."""
        return self.coefficient(0, 0)

    # ----------------------------------------------------------- dense forms
    def _x_groups(self):
        starts, stops = _x_group_bounds(self._xs)
        for s, e in zip(starts, stops):
            x = self._xs[s]
            zg = self._zs[s:e]
            yield x, zg, self._cs[s:e] * _ipow_phase(x, zg)

    def to_dense(self) -> np.ndarray:
        """Dense 2^n x 2^n matrix, M[b ^ x, b] = sum c i^{|x&z|} (-1)^{|b&z|} (n <= 14)."""
        n = self._n
        if n > 14:
            raise ValueError(f"to_dense is limited to n <= 14 qubits (got {n}); use to_sparse")
        dim = 1 << n
        out = np.zeros((dim, dim), dtype=np.complex128)
        b = np.arange(dim, dtype=np.uint64)
        cols = b.astype(np.intp)
        for x, zg, cg in self._x_groups():
            out[(b ^ x).astype(np.intp), cols] = _character_sum(zg, cg, b, n)
        return out

    def to_sparse(self):
        """``scipy.sparse.csr_matrix`` of the operator (same index convention as to_dense)."""
        import scipy.sparse as sp

        n = self._n
        if n > 30:
            raise ValueError(f"to_sparse is limited to n <= 30 qubits (got {n})")
        dim = 1 << n
        b = np.arange(dim, dtype=np.uint64)
        rows, cols, vals = [], [], []
        for x, zg, cg in self._x_groups():
            v = _character_sum(zg, cg, b, n)
            nz = np.flatnonzero(v)
            rows.append((b[nz] ^ x).astype(np.intp))
            cols.append(nz)
            vals.append(v[nz])
        if not rows:
            return sp.csr_matrix((dim, dim), dtype=np.complex128)
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(dim, dim))

    def matvec(self, vec: np.ndarray) -> np.ndarray:
        """A @ vec for a dense vector (or a (2^n, k) block) without building the matrix."""
        n = self._n
        v = np.asarray(vec)
        dim = 1 << n
        if v.shape[0] != dim:
            raise ValueError(f"vector length {v.shape[0]} != 2^{n}")
        out = np.zeros(v.shape, dtype=np.complex128)
        b = np.arange(dim, dtype=np.uint64)
        for x, zg, cg in self._x_groups():
            vals = _character_sum(zg, cg, b, n)
            rows = (b ^ x).astype(np.intp)
            if v.ndim == 1:
                out[rows] += vals * v
            else:
                out[rows] += vals[:, None] * v
        return out


def coefficient_matrix(ops: Sequence[PauliSum], basis: tuple | None = None):
    """Common Pauli basis of several operators and their coefficient matrix.

    Returns ``(xs, zs, W)``: the basis strings P(xs[j], zs[j]), which are sorted
    and unique, and ``W``, a complex ``scipy.sparse.csr_matrix`` of shape
    (len(ops), n_basis) with ops[i] = sum_j W[i, j] P(xs[j], zs[j]). ``basis``
    optionally gives the basis strings as ``(xs, zs)``. They are sorted, and they
    must contain every term of every op.
    """
    import scipy.sparse as sp

    ops = list(ops)
    if not ops:
        raise ValueError("coefficient_matrix needs at least one operator")
    n = ops[0].n_qubits
    if any(o.n_qubits != n for o in ops):
        raise ValueError("all operators must have the same n_qubits")
    rows = np.repeat(np.arange(len(ops)), [o.n_terms for o in ops])
    ax = np.concatenate([o.xs for o in ops])
    az = np.concatenate([o.zs for o in ops])
    ac = np.concatenate([o.coeffs for o in ops])
    if basis is None:
        bx, bz, _ = _canonicalize(n, ax, az, np.ones(ax.size, np.complex128), -1.0)
    else:
        gx = np.atleast_1d(_as_u64(basis[0])).ravel()
        gz = np.atleast_1d(_as_u64(basis[1])).ravel()
        if gx.size != gz.size:
            raise ValueError("basis xs and zs must have equal lengths")
        _check_bits(n, gx, gz)
        bx, bz, _ = _canonicalize(n, gx, gz, np.ones(gx.size, np.complex128), -1.0)
    ref = PauliSum._from_canonical(n, bx, bz, np.arange(1, bx.size + 1, dtype=np.complex128))
    col = ref.coefficients_of(ax, az).real.astype(np.intp) - 1
    if np.any(col < 0):
        raise ValueError("basis does not contain every term of the operators")
    w = sp.csr_matrix((ac, (rows, col)), shape=(len(ops), bx.size), dtype=np.complex128)
    return bx, bz, w


# ================================================================ Majoranas
def _majorana_phase_exponent(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """E = |~a & b & c| + |a & b| - |a & ~b & c| (mod 4) as uint8."""
    return (_popcount(~a & b & c) + _popcount(a & b) - _popcount(a & ~b & c)) & 3


def majorana_to_pauli(mask):
    """gamma_S = phase * P(x, z) for Majorana bitmasks S (vectorized; n <= 32).

    Returns ``(phase, x, z)`` with phase in {1, i, -1, -i} (complex128) and
    uint64 bitmasks; see the module docstring for the GF(2) map.
    """
    m0 = _as_u64(mask)
    shape = m0.shape
    m = np.atleast_1d(m0).ravel()
    a = _compress_even_bits(m)
    b = _compress_even_bits(m >> _U1)
    x = a ^ b
    c = _suffix_parity(x)
    z = b ^ c
    phase = _IPOW[_majorana_phase_exponent(a, b, c)]
    return (_scalar_or_array(phase, shape), _scalar_or_array(x, shape), _scalar_or_array(z, shape))


def pauli_to_majorana(x, z):
    """P(x, z) = phase * gamma_S (inverse of :func:`majorana_to_pauli`; vectorized; n <= 32).

    Returns ``(mask, phase)``; phase in {1, i, -1, -i} is the conjugate of the
    phase of :func:`majorana_to_pauli` for the same monomial.
    """
    xa, za = np.broadcast_arrays(_as_u64(x), _as_u64(z))
    shape = xa.shape
    xv = np.atleast_1d(xa).ravel()
    zv = np.atleast_1d(za).ravel()
    if np.any((xv | zv) >> _S32):
        raise ValueError("Majorana masks are uint64: at most 32 qubits")
    c = _suffix_parity(xv)
    b = zv ^ c
    a = xv ^ b
    mask = _spread_bits(a) | (_spread_bits(b) << _U1)
    phase = np.conj(_IPOW[_majorana_phase_exponent(a, b, c)])
    return _scalar_or_array(mask, shape), _scalar_or_array(phase, shape)


def majorana_product(m1, m2):
    """gamma_S gamma_T = sign * gamma_{S ^ T}; returns ``(sign, S ^ T)`` (vectorized).

    sign = (-1)^{|T & c(S)|} with c(S)_t = XOR_{s>t} S_s (float64 +-1).
    """
    s0, t0 = np.broadcast_arrays(_as_u64(m1), _as_u64(m2))
    shape = s0.shape
    s = np.atleast_1d(s0).ravel()
    t = np.atleast_1d(t0).ravel()
    sign = _parity_sign(t & _suffix_parity(s))
    return _scalar_or_array(sign, shape), _scalar_or_array(s ^ t, shape)


# ============================================================ QubitMapping
class QubitMapping:
    """Assignment of the 2N spin orbitals (p, sigma) to qubits.

    ``QubitMapping(norb, order="blocked")`` gives qubit(p, sigma) = p + sigma*N,
    the mode order of ``testing.fock``. ``order="interleaved"`` gives
    qubit(p, sigma) = 2p + sigma, OpenFermion's order.
    """

    __slots__ = ("_norb", "_order", "_qubit", "_mode_of_qubit")

    def __init__(self, norb: int, order: str = "blocked"):
        norb = int(norb)
        if norb < 1 or 2 * norb > 64:
            raise ValueError(f"norb must be in [1, 32], got {norb}")
        if order not in ("blocked", "interleaved"):
            raise ValueError(f"order must be 'blocked' or 'interleaved', got {order!r}")
        p = np.arange(norb)
        if order == "blocked":
            q = np.stack([p, norb + p], axis=1)
        else:
            q = np.stack([2 * p, 2 * p + 1], axis=1)
        q.setflags(write=False)
        inv = np.empty(2 * norb, dtype=np.int64)
        inv[q.T.ravel()] = np.arange(2 * norb)  # mode index = p + sigma * N
        inv.setflags(write=False)
        object.__setattr__(self, "_norb", norb)
        object.__setattr__(self, "_order", order)
        object.__setattr__(self, "_qubit", q)
        object.__setattr__(self, "_mode_of_qubit", inv)

    def __setattr__(self, name, value):
        raise AttributeError("QubitMapping is immutable")

    def __reduce__(self):
        return (type(self), (self._norb, self._order))

    def __copy__(self) -> "QubitMapping":
        return self

    def __deepcopy__(self, memo) -> "QubitMapping":
        return self

    @property
    def norb(self) -> int:
        return self._norb

    @property
    def order(self) -> str:
        return self._order

    @property
    def n_qubits(self) -> int:
        return 2 * self._norb

    def qubit(self, p, sigma):
        """Qubit index of spin orbital (p, sigma); vectorized over arrays.

        p and sigma are integers or integer arrays (booleans count as 0/1; floats
        raise TypeError). A scalar pair gives a Python int.
        """
        p_arr = _as_int_array(p, "orbital indices p")
        s_arr = _as_int_array(sigma, "spin indices sigma")
        if np.any((p_arr < 0) | (p_arr >= self._norb)) or np.any((s_arr != 0) & (s_arr != 1)):
            raise ValueError(f"spin orbital ({p}, {sigma}) out of range for norb={self._norb}")
        out = self._qubit[p_arr, s_arr]
        return int(out) if out.ndim == 0 else out

    @property
    def qubit_of_mode(self) -> np.ndarray:
        """qubit_of_mode[p + sigma*N] = qubit(p, sigma)."""
        return self._qubit.T.ravel()

    @property
    def mode_of_qubit(self) -> np.ndarray:
        """Inverse of :attr:`qubit_of_mode`."""
        return self._mode_of_qubit

    def spin_orbital(self, q: int) -> tuple[int, int]:
        """(p, sigma) of qubit q."""
        m = int(self._mode_of_qubit[int(q)])
        return m % self._norb, m // self._norb

    def __eq__(self, other) -> bool:
        return (isinstance(other, QubitMapping) and other._norb == self._norb
                and other._order == self._order)

    def __hash__(self) -> int:
        return hash((self._norb, self._order))

    def __repr__(self) -> str:
        return f"QubitMapping(norb={self._norb}, order={self._order!r})"

    def convert_basis(self, basis, target: "QubitMapping"):
        """Re-express basis states of this mapping in the qubit order of ``target``.

        Returns ``(basis_target, sign)`` with |b>_self = sign |b'>_target for the same
        fermionic state; sign (int8, +-1) is the parity of the permutation that sorts
        the occupied spin orbitals from this qubit order into the target's.
        """
        if target.norb != self._norb:
            raise ValueError("mappings have different norb")
        b = np.atleast_1d(_as_u64(basis)).ravel()
        n = self.n_qubits
        perm = target.qubit_of_mode[self._mode_of_qubit]  # this qubit k -> target qubit
        out = np.zeros_like(b)
        parity = np.zeros(b.shape, dtype=np.uint8)
        for k in range(n):
            bit = (b >> np.uint64(k)) & _U1
            out |= bit << np.uint64(perm[k])
            # qubits after k (here) that come before k in the target order
            inverted = [k2 for k2 in range(k + 1, n) if perm[k2] < perm[k]]
            if inverted:
                mask = np.uint64(sum(1 << k2 for k2 in inverted))
                parity ^= bit.astype(np.uint8) & (_popcount(b & mask) & 1)
        sign = (1 - 2 * parity.astype(np.int8)).astype(np.int8)
        return out, sign

    def ci_basis(self, nelec) -> tuple[np.ndarray, np.ndarray]:
        """Basis indices and signs of all PySCF determinants (alpha-major, ravelled ci order).

        The determinant (ia, ib) is sign * |b> with the convention of
        :meth:`QubitState.from_ci` (without the PySCF global sign).
        """
        from pyscf.fci import cistring

        n = self._norb
        na, nb = (int(v) for v in nelec)
        sa = np.asarray(cistring.make_strings(range(n), na), dtype=np.uint64)
        sb = np.asarray(cistring.make_strings(range(n), nb), dtype=np.uint64)
        blocked = (sa[:, None] | (sb[None, :] << np.uint64(n))).ravel()
        if self._order == "blocked":
            return blocked, np.ones(blocked.size, dtype=np.int8)
        return QubitMapping(n, "blocked").convert_basis(blocked, self)


def _pyscf_global_sign(nelec) -> int:
    na, nb = (int(v) for v in nelec)
    return -1 if (na * (na - 1) // 2 + nb * (nb - 1) // 2) % 2 else 1


# ============================================================ Jordan-Wigner
def _ladder_terms(qubits: np.ndarray, daggers: np.ndarray, coeffs: np.ndarray):
    """Raw Pauli terms of coeffs[r] * prod_j a^{(daggers[r, j])}_{qubits[r, j]} (operator order).

    qubits, daggers: (B, m); coeffs: (B,). Returns flat (xs, zs, cs) of length B * 2^m,
    not canonicalized; terms of row r occupy the block [r * 2^m, (r + 1) * 2^m).
    """
    qubits = np.asarray(qubits, dtype=np.int64)
    if qubits.ndim != 2:
        raise ValueError("qubits must have shape (B, m)")
    nb, m = qubits.shape
    daggers = np.broadcast_to(np.asarray(daggers, dtype=bool), (nb, m))
    coeffs = np.broadcast_to(np.asarray(coeffs, dtype=np.complex128), (nb,))
    x = np.zeros((nb, 1), dtype=np.uint64)
    z = np.zeros((nb, 1), dtype=np.uint64)
    ph = np.zeros((nb, 1), dtype=np.uint8)
    if nb and m:
        q = qubits.astype(np.uint64)
        ek = _U1 << q
        lo = ek - _U1
        hi = lo | ek
        zero8 = np.zeros(nb, dtype=np.uint8)
        for j in range(m):
            xj = ek[:, j, None, None]                                    # (B, 1, 1)
            zj = np.stack((lo[:, j], hi[:, j]), axis=1)[:, None, :]      # (B, 1, 2)
            # a = (P0 + i P1)/2 (phase exponent 1), a+ = (P0 - i P1)/2 (exponent 3)
            tj = np.where(daggers[:, j], np.uint8(3), np.uint8(1)).astype(np.uint8)
            pj = np.stack((zero8, tj), axis=1)[:, None, :]               # (B, 1, 2)
            phi, x3, z3 = _pp_raw(x[:, :, None], z[:, :, None], xj, zj)
            k2 = 2 * x.shape[1]
            ph = ((ph[:, :, None] + pj + phi) & 3).reshape(nb, k2)
            x = np.broadcast_to(x3, (nb, x.shape[1], 2)).reshape(nb, k2)
            z = z3.reshape(nb, k2)
    cs = (coeffs * 0.5 ** m)[:, None] * _IPOW[ph]
    return x.ravel(), z.ravel(), cs.ravel()


def _ops_to_arrays(mapping: QubitMapping, ops) -> tuple[np.ndarray, np.ndarray]:
    arr = _as_int_array(ops, "ops (p, sigma, dagger)")
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("ops must be (p, sigma, dagger) triples")
    dag = arr[..., 2]
    if np.any((dag != 0) & (dag != 1)):
        raise ValueError("dagger flags must be 0/1 (False/True)")
    qubits = mapping.qubit(arr[..., 0], arr[..., 1])
    return np.asarray(qubits, dtype=np.int64).reshape(arr.shape[:2]), dag.astype(bool)


def jw_ladder(mapping: QubitMapping, p: int, sigma: int, dagger: bool) -> PauliSum:
    """a_{p sigma} (dagger=False) or a+_{p sigma} (dagger=True) as a PauliSum."""
    return jw_product(mapping, [(p, sigma, bool(dagger))])


def jw_product(mapping: QubitMapping, ops: Sequence, coeff: complex = 1.0, *,
               tol: float = DEFAULT_TOL) -> PauliSum:
    """coeff * prod_j a^{(dagger_j)}_{p_j sigma_j} as a PauliSum.

    ``ops`` = [(p, sigma, dagger), ...] in operator order: the leftmost operator
    acts last. An empty ``ops`` gives coeff * I.
    """
    ops = list(ops)
    if not ops:
        return PauliSum(mapping.n_qubits, [0], [0], [coeff], tol=tol)
    return jw_products(mapping, [ops], [coeff], tol=tol)[0]


def jw_products(mapping: QubitMapping, ops_batch, coeffs=None, *,
                tol: float = DEFAULT_TOL) -> list[PauliSum]:
    """Many ladder-operator products at once (vectorized ``jw_product``).

    ``ops_batch`` has shape (K, m, 3): K products of m operators (p, sigma, dagger).
    Returns K PauliSums ``coeffs[k] * prod_j a...`` (coeffs default to 1).
    """
    qubits, daggers = _ops_to_arrays(mapping, ops_batch)
    k = qubits.shape[0]
    if coeffs is None:
        cs = np.ones(k, dtype=np.complex128)
    else:
        cs = np.asarray(coeffs, dtype=np.complex128).reshape(k)
    xs, zs, c = _ladder_terms(qubits, daggers, cs)
    rows = np.repeat(np.arange(k), 1 << qubits.shape[1])
    return _canonicalize_rows(mapping.n_qubits, rows, k, xs, zs, c, tol)


def _one_body_terms(mapping: QubitMapping, f: np.ndarray):
    n = mapping.norb
    f = np.asarray(f)
    if f.shape != (n, n):
        raise ValueError(f"one-body tensor must have shape {(n, n)}, got {f.shape}")
    p, q, s = np.meshgrid(np.arange(n), np.arange(n), np.arange(2), indexing="ij")
    vals = np.broadcast_to(f[:, :, None], (n, n, 2))
    keep = vals != 0
    p, q, s, vals = p[keep], q[keep], s[keep], vals[keep]
    qubits = np.stack([mapping.qubit(p, s), mapping.qubit(q, s)], axis=1)
    daggers = np.array([True, False])
    return _ladder_terms(qubits.reshape(-1, 2), daggers, vals)


def _two_body_terms(mapping: QubitMapping, c: np.ndarray):
    n = mapping.norb
    c = np.asarray(c)
    if c.shape != (n,) * 4:
        raise ValueError(f"two-body tensor must have shape {(n,) * 4}, got {c.shape}")
    idx = np.nonzero(c)
    vals = c[idx]
    p, q, r, s = (np.asarray(i) for i in idx)
    out_q, out_c = [], []
    for sig in (0, 1):
        for tau in (0, 1):
            # a+_{p sig} a+_{r tau} a_{s tau} a_{q sig}; a+a+ (aa) on one mode vanishes
            ok = ~((sig == tau) & ((p == r) | (q == s)))
            qq = np.stack([mapping.qubit(p[ok], sig), mapping.qubit(r[ok], tau),
                           mapping.qubit(s[ok], tau), mapping.qubit(q[ok], sig)], axis=1)
            out_q.append(qq.reshape(-1, 4))
            out_c.append(vals[ok])
    qubits = np.concatenate(out_q)
    daggers = np.array([True, True, False, False])
    return _ladder_terms(qubits, daggers, np.concatenate(out_c))


def jw_one_body(mapping: QubitMapping, f, *, tol: float = DEFAULT_TOL) -> PauliSum:
    """sum_pq f[p,q] E_pq (spin-summed; complex and non-Hermitian f allowed)."""
    return PauliSum._canon(mapping.n_qubits, *_one_body_terms(mapping, f), tol)


def jw_two_body(mapping: QubitMapping, c, *, tol: float = DEFAULT_TOL) -> PauliSum:
    """sum_pqrs c[p,q,r,s] e_{pq,rs}, e_{pq,rs} = sum a+_{p s} a+_{r t} a_{s t} a_{q s} (any c)."""
    return PauliSum._canon(mapping.n_qubits, *_two_body_terms(mapping, c), tol)


def jw_hamiltonian(active, mapping: QubitMapping | None = None, *,
                   tol: float = DEFAULT_TOL, hermiticity_atol: float = 1e-10) -> PauliSum:
    """ecore I + sum h1[p,q] E_pq + 1/2 sum (pq|rs) e_{pq,rs} as a PauliSum with real coefficients.

    ``active`` is an :class:`~thc_engine_cqe.integrals.ActiveSpace` (or any object
    with ``h1``, ``eri``, ``ecore``). The operator is built in one vectorized pass.
    Its coefficients are real up to round-off. If the largest imaginary part
    exceeds ``hermiticity_atol * max(1, max|c|)``, a ValueError is raised, because
    the integrals do not define a Hermitian operator. Otherwise the real part
    (the Hermitian part) is returned.
    """
    h1 = np.asarray(active.h1)
    eri = np.asarray(active.eri)
    n = h1.shape[0]
    mapping = QubitMapping(n) if mapping is None else mapping
    if mapping.norb != n:
        raise ValueError(f"mapping has norb={mapping.norb}, integrals have {n}")
    x1, z1, c1 = _one_body_terms(mapping, h1)
    x2, z2, c2 = _two_body_terms(mapping, 0.5 * eri)
    xs = np.concatenate([np.zeros(1, np.uint64), x1, x2])
    zs = np.concatenate([np.zeros(1, np.uint64), z1, z2])
    cs = np.concatenate([np.array([complex(active.ecore)]), c1, c2])
    op = PauliSum._canon(mapping.n_qubits, xs, zs, cs, tol)
    if op.n_terms:
        imag = float(np.max(np.abs(op.coeffs.imag)))
        scale = max(1.0, float(np.max(np.abs(op.coeffs))))
        if imag > hermiticity_atol * scale:
            raise ValueError(f"the integrals do not define a Hermitian operator "
                             f"(max |Im c| = {imag:.3g})")
    re = op.coeffs.real
    return op._filtered(np.abs(re) > tol, re.astype(np.complex128))   # Hermitian part, same absolute tol


def _rdm2_ops(elements) -> np.ndarray:
    el = _as_int_array(elements, "spin-orbital indices (p, sigma)")
    if el.ndim == 2:
        el = el[None]
    if el.ndim != 3 or el.shape[1:] != (4, 2):
        raise ValueError("elements must be (i, j, k, l) with each index a (p, sigma) pair")
    i, j, k, l = el[:, 0], el[:, 1], el[:, 2], el[:, 3]
    ops = np.empty((el.shape[0], 4, 3), dtype=np.int64)
    for slot, (orb, dag) in enumerate(((i, 1), (j, 1), (l, 0), (k, 0))):
        ops[:, slot, :2] = orb
        ops[:, slot, 2] = dag
    return ops


def jw_rdm2_element(mapping: QubitMapping, i, j, k, l, *, tol: float = DEFAULT_TOL) -> PauliSum:
    """a+_i a+_j a_l a_k for spin-orbital tuples i, j, k, l = (p, sigma).

    Its expectation value is Mazziotti's 2D^{ij}_{kl}.
    """
    return jw_rdm2_elements(mapping, [(i, j, k, l)], tol=tol)[0]


def jw_rdm2_elements(mapping: QubitMapping, elements, *, tol: float = DEFAULT_TOL) -> list[PauliSum]:
    """Vectorized :func:`jw_rdm2_element`. ``elements`` has shape (K, 4, 2): rows (i, j, k, l)."""
    return jw_products(mapping, _rdm2_ops(elements), tol=tol)


# =============================================================== QubitState
class QubitState:
    """Sparse state vector sum_b psi[b] |b>: sorted unique uint64 basis indices and amplitudes.

    ``QubitState(n_qubits, basis, amplitudes, mapping=None, tol=0.0)`` sorts the
    basis, sums duplicates and drops amplitudes with |a| <= tol. ``mapping`` is
    optional bookkeeping (set by :meth:`from_ci`, used as the default of
    :meth:`to_ci`); ``+``, ``-`` and :meth:`vdot` refuse two states with
    different mappings and keep the mapping of whichever operand has one. The
    object is immutable.
    """

    __slots__ = ("_n", "_basis", "_amps", "_mapping", "_index")
    __array_ufunc__ = None

    def __init__(self, n_qubits: int, basis, amplitudes, *, mapping: QubitMapping | None = None,
                 tol: float = 0.0):
        n = _check_n(n_qubits)
        b = np.atleast_1d(_as_u64(basis)).ravel()
        a = np.atleast_1d(np.asarray(amplitudes, dtype=np.complex128)).ravel()
        if b.size != a.size:
            raise ValueError("basis and amplitudes must have equal lengths")
        _check_bits(n, b)
        if mapping is not None and mapping.n_qubits != n:
            raise ValueError("mapping.n_qubits != n_qubits")
        if b.size:
            order = np.argsort(b, kind="stable")
            sb = b[order]
            starts = _group_starts(sb)
            ub = sb[starts]
            ua = np.add.reduceat(a[order], starts)
            keep = np.abs(ua) > tol
            b, a = ub[keep], ua[keep]
        self._set(n, b, a, mapping)

    def _set(self, n, b, a, mapping):
        b = np.ascontiguousarray(b, dtype=np.uint64)
        a = np.ascontiguousarray(a, dtype=np.complex128)
        b.setflags(write=False)
        a.setflags(write=False)
        object.__setattr__(self, "_n", n)
        object.__setattr__(self, "_basis", b)
        object.__setattr__(self, "_amps", a)
        object.__setattr__(self, "_mapping", mapping)
        object.__setattr__(self, "_index", None)

    def __setattr__(self, name, value):
        raise AttributeError("QubitState is immutable")

    # immutable: copies are the object itself; pickles drop the cached position table
    def __reduce__(self):
        return (type(self)._from_canonical, (self._n, self._basis, self._amps, self._mapping))

    def __copy__(self) -> "QubitState":
        return self

    def __deepcopy__(self, memo) -> "QubitState":
        return self

    @classmethod
    def _from_canonical(cls, n, b, a, mapping=None) -> "QubitState":
        obj = object.__new__(cls)
        obj._set(n, b, a, mapping)
        return obj

    def _common_mapping(self, other: "QubitState") -> QubitMapping | None:
        """The mapping of a binary operation; a ValueError if both are set and differ."""
        if other._n != self._n:
            raise ValueError(f"n_qubits mismatch: {self._n} vs {other._n}")
        if self._mapping is not None and other._mapping is not None and self._mapping != other._mapping:
            raise ValueError(f"states use different qubit mappings ({self._mapping} vs {other._mapping}); "
                             f"the same basis index denotes different determinants")
        return self._mapping if self._mapping is not None else other._mapping

    # ------------------------------------------------------------ constructors
    @classmethod
    def from_ci(cls, norb: int, nelec, ci, mapping: QubitMapping | None = None, *,
                pyscf_sign: bool = False) -> "QubitState":
        """Qubit state of a PySCF CI array (n_alpha_strings, n_beta_strings); complex allowed.

        Exact zeros of ``ci`` are not stored. Blocked order: b = alpha_bits | (beta_bits << N)
        with amplitude ci[ia, ib], as in ``FockSpace.from_ci``. Other orders apply the
        determinant-dependent reordering sign, so every order holds the same fermionic state.
        ``pyscf_sign=True`` multiplies by PySCF's global determinant sign
        (-1)^{na(na-1)/2 + nb(nb-1)/2} (see the module docstring).
        """
        mapping = QubitMapping(norb) if mapping is None else mapping
        if mapping.norb != norb:
            raise ValueError(f"mapping has norb={mapping.norb}, expected {norb}")
        basis, sign = mapping.ci_basis(nelec)
        ci = np.asarray(ci)
        if ci.size != basis.size:
            raise ValueError(f"CI vector has {ci.size} entries, expected {basis.size}")
        amps = ci.ravel().astype(np.complex128) * sign
        if pyscf_sign:
            amps = amps * _pyscf_global_sign(nelec)
        return cls(mapping.n_qubits, basis, amps, mapping=mapping)

    def to_ci(self, norb: int, nelec, mapping: QubitMapping | None = None, *,
              pyscf_sign: bool = False) -> np.ndarray:
        """Inverse of :meth:`from_ci` on the (n_alpha, n_beta) sector; other amplitudes are ignored."""
        from pyscf.fci import cistring

        mapping = mapping or self._mapping or QubitMapping(norb)
        if mapping.norb != norb or mapping.n_qubits != self._n:
            raise ValueError(f"{mapping} does not match norb={norb} and {self._n} qubits")
        basis, sign = mapping.ci_basis(nelec)
        amps = self._lookup(basis) * sign
        if pyscf_sign:
            amps = amps * _pyscf_global_sign(nelec)
        shape = (cistring.num_strings(norb, nelec[0]), cistring.num_strings(norb, nelec[1]))
        return amps.reshape(shape)

    @classmethod
    def from_dense(cls, vec, *, tol: float = 0.0, mapping: QubitMapping | None = None) -> "QubitState":
        """Sparse copy of a dense vector of length 2^n (entries with |v| <= tol dropped)."""
        v = np.asarray(vec).ravel()
        if v.size == 0:
            raise ValueError("from_dense needs a non-empty vector of length 2^n")
        n = int(v.size).bit_length() - 1
        if v.size != 1 << n:
            raise ValueError(f"length {v.size} is not a power of two")
        if mapping is not None and mapping.n_qubits != n:
            raise ValueError(f"mapping.n_qubits = {mapping.n_qubits} != {n} qubits of the vector")
        nz = np.flatnonzero(np.abs(v) > tol)
        return cls._from_canonical(n, nz.astype(np.uint64), v[nz].astype(np.complex128), mapping)

    def to_dense(self) -> np.ndarray:
        """Dense vector of length 2^n (n <= 30)."""
        if self._n > 30:
            raise ValueError("to_dense is limited to n <= 30 qubits")
        out = np.zeros(1 << self._n, dtype=np.complex128)
        out[self._basis.astype(np.intp)] = self._amps
        return out

    # -------------------------------------------------------------- properties
    @property
    def n_qubits(self) -> int:
        return self._n

    @property
    def basis(self) -> np.ndarray:
        return self._basis

    @property
    def amplitudes(self) -> np.ndarray:
        return self._amps

    @property
    def mapping(self) -> QubitMapping | None:
        return self._mapping

    @property
    def size(self) -> int:
        """Number of stored (nonzero) amplitudes."""
        return int(self._amps.size)

    def __repr__(self) -> str:
        return f"QubitState(n_qubits={self._n}, size={self.size}, norm={self.norm():.6g})"

    def norm(self) -> float:
        return float(np.linalg.norm(self._amps))

    def normalized(self) -> "QubitState":
        nrm = self.norm()
        if nrm == 0:
            raise ValueError("cannot normalize the zero state")
        return QubitState._from_canonical(self._n, self._basis, self._amps / nrm, self._mapping)

    def _positions(self, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(pos, found): position of each query basis index in self.basis."""
        d = self._basis.size
        if d == 0:
            return np.zeros(queries.shape, np.intp), np.zeros(queries.shape, bool)
        # dense position table (cached) when it is not much larger than the support
        if self._n <= _INDEX_MAP_MAX_QUBITS and (1 << self._n) <= max(1 << 16, 64 * d):
            if self._index is None:
                idx = np.full(1 << self._n, -1, dtype=np.int32 if d < 2**31 else np.int64)
                idx[self._basis.astype(np.intp)] = np.arange(d)
                object.__setattr__(self, "_index", idx)
            pos = self._index[queries.astype(np.intp)]
            found = pos >= 0
            return np.where(found, pos, 0).astype(np.intp), found
        pos = np.minimum(np.searchsorted(self._basis, queries), d - 1)
        return pos, self._basis[pos] == queries

    def _lookup(self, basis: np.ndarray) -> np.ndarray:
        """Amplitudes at the given basis indices (0 where absent)."""
        pos, found = self._positions(np.atleast_1d(_as_u64(basis)).ravel())
        return np.where(found, self._amps[pos], 0.0)

    def vdot(self, other: "QubitState") -> complex:
        """<self|other> (a ValueError if the two mappings are set and differ)."""
        self._common_mapping(other)
        return complex(np.vdot(self._lookup(other._basis), other._amps))

    # ------------------------------------------------------------- arithmetic
    def __add__(self, other):
        if not isinstance(other, QubitState):
            return NotImplemented
        mapping = self._common_mapping(other)
        return QubitState(self._n, np.concatenate([self._basis, other._basis]),
                          np.concatenate([self._amps, other._amps]), mapping=mapping)

    def __neg__(self):
        return QubitState._from_canonical(self._n, self._basis, -self._amps, self._mapping)

    def __sub__(self, other):
        if not isinstance(other, QubitState):
            return NotImplemented
        return self + (-other)

    def __mul__(self, other):
        s = _as_scalar(other)
        if s is None:
            return NotImplemented
        amps = self._amps * s
        keep = amps != 0
        return QubitState._from_canonical(self._n, self._basis[keep], amps[keep], self._mapping)

    __rmul__ = __mul__

    # ----------------------------------------------------------- expectations
    def _norm2_for(self, normalize: bool) -> float:
        """<psi|psi> if ``normalize`` (a ValueError for the zero state), else 1."""
        if not normalize:
            return 1.0
        nrm2 = float(np.vdot(self._amps, self._amps).real)
        if nrm2 == 0:
            raise ValueError("normalized expectation values of the zero state are undefined")
        return nrm2

    def expectations(self, xs, zs, *, normalize: bool = False) -> np.ndarray:
        """<psi|P(xs[i], zs[i])|psi>, complex: the bare matrix elements of the spec's formula.

        ``normalize=True`` divides by <psi|psi> (ValueError for the zero state).
        xs and zs broadcast together. The strings are grouped by x. For each
        distinct x, the pairs (b, b ^ x) inside the support are found once,
        w_b = conj(psi[b ^ x]) psi[b] is formed, and sum_b w_b (-1)^{|b & z|} is
        evaluated for every z of the group, either with one sign-matrix product
        or with a fast Walsh-Hadamard transform.
        """
        xa, za = np.broadcast_arrays(_as_u64(xs), _as_u64(zs))
        shape = xa.shape
        qx = np.atleast_1d(xa).ravel()
        qz = np.atleast_1d(za).ravel()
        _check_bits(self._n, qx, qz)
        nrm2 = self._norm2_for(normalize)
        out = np.zeros(qx.size, dtype=np.complex128)
        d = self._basis.size
        if qx.size and d:
            order = np.argsort(qx, kind="stable")
            sx, sz = qx[order], qz[order]
            starts, stops = _x_group_bounds(sx)
            ux = sx[starts]
            basis, amps = self._basis, self._amps
            step = max(1, _CHUNK // d)
            for c0 in range(0, ux.size, step):
                uxc = ux[c0:c0 + step]
                pos, found = self._positions(basis[None, :] ^ uxc[:, None])
                for r in np.flatnonzero(found.any(axis=1)):
                    g = c0 + r
                    f = found[r]
                    if f.all():
                        bsel, w = basis, np.conj(amps[pos[r]]) * amps
                    else:
                        sel = np.flatnonzero(f)
                        bsel, w = basis[sel], np.conj(amps[pos[r, sel]]) * amps[sel]
                    zg = sz[starts[g]:stops[g]]
                    vals = _character_sum(bsel, w, zg, self._n)
                    out[order[starts[g]:stops[g]]] = _ipow_phase(ux[g], zg) * vals
        if normalize:
            out /= nrm2
        return out.reshape(shape)

    def expectation(self, op: PauliSum, *, normalize: bool = False) -> complex:
        """<psi|op|psi>, the bare matrix element (divided by <psi|psi> if ``normalize``)."""
        if op.n_qubits != self._n:
            raise ValueError(f"operator has {op.n_qubits} qubits, state has {self._n}")
        self._norm2_for(normalize)   # the zero state raises for normalize=True, also for op = 0
        if op.n_terms == 0:
            return 0j
        return complex(np.dot(op.coeffs, self.expectations(op.xs, op.zs, normalize=normalize)))

    def apply(self, op: PauliSum, *, tol: float = DEFAULT_TOL) -> "QubitState":
        """op |psi> as a new sparse state; the support may leave the particle-number sector.

        Round-off from cancellations is removed at two levels, each relative to
        the scale of the contributions that were summed:

        * For the terms that share one x, the sum over z at basis state b,
          v_x(b) = sum_z c_z i^{|x&z|} (-1)^{|b&z|}, is dropped when
          |v_x(b)| <= tol * s_x with s_x = sum_z |c_z|. The value is then zero
          within its round-off, as in the number-violating parts of a
          number-conserving operator.
        * An output amplitude a(b') = sum_x v_x(b) psi(b), with b = b' ^ x, is
          dropped when |a(b')| <= tol * sum_x s_x |psi(b)|.

        Hence each amplitude of the result differs from the exact one by at most
        tol * E(b') plus floating-point round-off, E(b') = sum_x s_x |psi(b' ^ x)|;
        the result is exactly proportional when op or psi is rescaled by a power
        of two; and a single contribution is never dropped (``apply`` of the
        identity returns psi). ``tol=0`` keeps every nonzero amplitude.
        """
        if op.n_qubits != self._n:
            raise ValueError(f"operator has {op.n_qubits} qubits, state has {self._n}")
        basis, amps = self._basis, self._amps
        if op.n_terms == 0 or basis.size == 0:
            return QubitState._from_canonical(self._n, np.zeros(0, np.uint64),
                                              np.zeros(0, np.complex128), self._mapping)
        abs_amps = np.abs(amps)
        new_b, new_a, new_w, acc = [], [], [], 0
        for x, zg, cg in op._x_groups():
            vals = _character_sum(zg, cg, basis, self._n)
            s_x = float(np.sum(np.abs(cg)))
            keep = np.abs(vals) > tol * s_x
            if not keep.any():
                continue
            new_b.append(basis[keep] ^ x)
            new_a.append(vals[keep] * amps[keep])
            new_w.append(s_x * abs_amps[keep])
            acc += new_b[-1].size
            if acc > 4 * _CHUNK and len(new_b) > 1:   # merge duplicates to bound memory (keep zeros)
                b, a, w = _sum_by_index(np.concatenate(new_b), np.concatenate(new_a), np.concatenate(new_w))
                new_b, new_a, new_w, acc = [b], [a], [w], b.size
        if not new_b:
            return QubitState._from_canonical(self._n, np.zeros(0, np.uint64),
                                              np.zeros(0, np.complex128), self._mapping)
        b, a, w = _sum_by_index(np.concatenate(new_b), np.concatenate(new_a), np.concatenate(new_w))
        keep = np.abs(a) > tol * w
        return QubitState._from_canonical(self._n, b[keep], a[keep], self._mapping)
