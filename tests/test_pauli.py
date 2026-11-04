"""Tests of the Pauli / Jordan-Wigner layer (``thc_engine_cqe.measure.pauli``).

Every convention in the module docstring is checked against an independent reference:

* Kronecker products of 2x2 matrices: the Pauli algebra and the Majoranas.
* The Fock-space oracle ``testing.fock.FockSpace``: the JW operators in blocked order.
* PySCF: FCI energies, ``make_rdm12s`` and ``cre_a``/``des_a``/``cre_b``/``des_b``.
* OpenFermion: ``jordan_wigner`` of the ``InteractionOperator`` (interleaved order)
  and its sparse matrices.
"""

import copy
import functools
import itertools
import multiprocessing
import operator
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

from conftest import random_active_space
from thc_engine_cqe import residual, systems
from thc_engine_cqe.fci_ops import FCISpace, Hamiltonian
from thc_engine_cqe.integrals import ActiveSpace
from thc_engine_cqe.measure import pauli as pauli_mod
from thc_engine_cqe.measure.pauli import (DEFAULT_TOL, PauliSum, QubitMapping, QubitState,
                                          coefficient_matrix, jw_hamiltonian, jw_ladder,
                                          jw_one_body, jw_product, jw_products, jw_rdm2_element,
                                          jw_rdm2_elements, jw_two_body, majorana_product,
                                          majorana_to_pauli, pauli_commutes, pauli_label,
                                          pauli_product, pauli_to_majorana, pauli_weight,
                                          qubitwise_commutes)
from thc_engine_cqe.testing.fock import FockSpace

ORDERS = ("blocked", "interleaved")

_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_ONE_QUBIT = {(0, 0): _I2, (1, 0): _X, (0, 1): _Z, (1, 1): _Y}


def popcount(v) -> int:
    return bin(int(v)).count("1")


def one_qubit_factor(x, z, k):
    return _ONE_QUBIT[((int(x) >> k) & 1, (int(z) >> k) & 1)]


def dense_pauli(x, z, n):
    """Independent reference: kron of X/Y/Z factors, qubit 0 = least significant bit."""
    out = np.ones((1, 1), dtype=complex)
    for k in range(n):
        out = np.kron(one_qubit_factor(x, z, k), out)
    return out


def random_paulisum(rng, n, n_terms, complex_=True):
    xs = rng.integers(0, 1 << n, n_terms)
    zs = rng.integers(0, 1 << n, n_terms)
    cs = rng.standard_normal(n_terms) + (1j * rng.standard_normal(n_terms) if complex_ else 0)
    return PauliSum(n, xs, zs, cs)


def assert_canonical(op):
    """The canonical-form invariant checked from the raw arrays: keys sorted by (x, z) as
    unsigned integers and unique, nonzero coefficients, bits inside the n qubits."""
    assert op.xs.dtype == np.uint64 and op.zs.dtype == np.uint64 and op.coeffs.dtype == np.complex128
    keys = list(zip(op.xs.tolist(), op.zs.tolist()))
    assert keys == sorted(set(keys))
    assert np.all(op.coeffs != 0)
    assert all(x >> op.n_qubits == 0 and z >> op.n_qubits == 0 for x, z in keys)


# Symbolic Pauli strings for any n (independent of the module's bit formulas): one code per
# qubit, code = x_k + 2 z_k, i.e. 0 = I, 1 = X, 2 = Z, 3 = Y; P(x, z) is the tensor product.
_CODE_MATS = (_I2, _X, _Z, _Y)


def _single_qubit_table():
    """sigma_a sigma_b = phase[a, b] sigma_{code[a, b]}, read off the 2x2 matrices."""
    phase = np.zeros((4, 4), dtype=complex)
    code = np.zeros((4, 4), dtype=np.int64)
    for a, b in itertools.product(range(4), repeat=2):
        m = _CODE_MATS[a] @ _CODE_MATS[b]
        hits = [(ph, c) for c in range(4) for ph in (1, 1j, -1, -1j) if np.allclose(m, ph * _CODE_MATS[c])]
        assert len(hits) == 1
        phase[a, b], code[a, b] = hits[0]
    return phase, code


_TABLE_PHASE, _TABLE_CODE = _single_qubit_table()


def to_codes(xs, zs, n):
    """(M, n) array of per-qubit codes of the strings P(xs[i], zs[i]) (Python ints, any n <= 64)."""
    return np.array([[((int(x) >> k) & 1) + 2 * ((int(z) >> k) & 1) for k in range(n)]
                     for x, z in zip(xs, zs)], dtype=np.int64).reshape(len(xs), n)


def from_codes(codes):
    """(xs, zs) as lists of Python ints."""
    xs = [sum(1 << k for k, c in enumerate(row) if c in (1, 3)) for row in codes.tolist()]
    zs = [sum(1 << k for k, c in enumerate(row) if c in (2, 3)) for row in codes.tolist()]
    return xs, zs


def table_product(c1, c2):
    """Per-qubit product of code arrays: P1 P2 = phase * P3, phase from the 2x2 table."""
    return _TABLE_PHASE[c1, c2].prod(axis=-1), _TABLE_CODE[c1, c2]


def majorana_codes(masks, n):
    """gamma_S (ascending product) for each mask as (phase, codes), by multiplying the single
    Majoranas gamma_2k = Z_<k X_k and gamma_2k+1 = Z_<k Y_k with the single-qubit table."""
    masks = [int(m) for m in masks]
    codes = np.zeros((len(masks), n), dtype=np.int64)
    phase = np.ones(len(masks), dtype=complex)
    for j in range(2 * n):
        sel = np.array([(m >> j) & 1 == 1 for m in masks], dtype=bool)
        if not sel.any():
            continue
        k = j // 2
        g = np.array([2] * k + [1 if j % 2 == 0 else 3] + [0] * (n - k - 1), dtype=np.int64)
        ph, codes[sel] = table_product(codes[sel], np.broadcast_to(g, (int(sel.sum()), n)))
        phase[sel] *= ph
    return phase, codes


def jw_apply_reference(ops, coeff, basis_state, n_modes):
    """coeff * prod_j a^{(dagger_j)}_{k_j} |b> with Python ints (ops in operator order, rightmost
    first): a_k|b> = (-1)^{sum_{j<k} b_j} |b - e_k>. Returns (b', amplitude) or None."""
    b, amp = int(basis_state), complex(coeff)
    for k, dag in reversed(ops):
        occupied = (b >> k) & 1
        if occupied == int(dag):
            return None
        amp *= (-1) ** bin(b & ((1 << k) - 1)).count("1")
        b ^= 1 << k
    assert b >> n_modes == 0
    return b, amp


def bit_reverse(n):
    idx = np.arange(1 << n)
    rev = np.zeros_like(idx)
    for k in range(n):
        rev |= ((idx >> k) & 1) << (n - 1 - k)
    return rev


@functools.lru_cache(maxsize=None)
def get_system(name):
    """(active space, E_FCI, ground-state CI vector)."""
    act = {
        "h4": lambda: systems.hydrogen_chain(4, spacing=1.8, unit="bohr"),
        "lih_cas": lambda: systems.lih(ncas=5, nelecas=2),   # frozen Li 1s, CAS(2,5)
        "n2": lambda: systems.n2(),                          # CAS(6,6) at 1.098 A
        "random4": lambda: random_active_space(4, (2, 2), seed=11),
    }[name]()
    e, ci = act.fci(spin_square=None if name.startswith("random") else 0.0)
    return act, e, ci


def openfermion_hamiltonian(act, order):
    """jordan_wigner(get_fermion_operator(InteractionOperator)) in the given spin-orbital order."""
    of = pytest.importorskip("openfermion")
    from openfermion.chem.molecular_data import spinorb_from_spatial

    two = np.asarray(act.eri.transpose(0, 2, 3, 1), order="C")   # OpenFermion's (ps|qr) order
    ob, tb = spinorb_from_spatial(act.h1, two)                      # interleaved spin orbitals
    if order == "blocked":
        n = act.norb
        perm = np.array([2 * p + s for s in (0, 1) for p in range(n)])  # blocked mode -> interleaved
        ob = ob[np.ix_(perm, perm)]
        tb = tb[np.ix_(perm, perm, perm, perm)]
    op = of.InteractionOperator(act.ecore, ob, 0.5 * tb)
    return of.jordan_wigner(of.get_fermion_operator(op))


def mazziotti_rdm2(ci, norb, nelec):
    """Spin-orbital 2D^{ij}_{kl} = <a+_i a+_j a_l a_k> from PySCF's make_rdm12s (indices (p, sigma))."""
    from pyscf.fci import direct_spin1

    _, (d2aa, d2ab, d2bb) = direct_spin1.make_rdm12s(ci, norb, nelec)
    # pair order: d2_{st}[p,q,r,s] = <a+_{p s} a+_{r t} a_{s t} a_{q s}>
    blocks = {(0, 0): d2aa, (0, 1): d2ab, (1, 1): d2bb, (1, 0): d2ab.transpose(2, 3, 0, 1)}

    def element(i, j, k, l):
        (pi, si), (pj, sj), (pk, sk), (pl, sl) = i, j, k, l
        if si == sk and sj == sl:
            return blocks[(si, sj)][pi, pk, pj, pl]
        if si == sl and sj == sk:   # a_l a_k = -a_k a_l
            return -blocks[(si, sj)][pi, pl, pj, pk]
        return 0.0

    return element


# =========================================================== Pauli strings
def test_pauli_string_definition_and_action():
    """P(x, z) = i^{|x&z|} X^x Z^z is the tensor product of X (x), Z (z), Y (both) with qubit 0
    the least significant bit; it is Hermitian and P|b> = i^{|x&z|} (-1)^{|b&z|} |b ^ x>."""
    n = 3
    for x in range(8):
        for z in range(8):
            ref = dense_pauli(x, z, n)
            assert np.allclose(PauliSum(n, [x], [z], [1.0]).to_dense(), ref)
            assert np.allclose(ref, ref.conj().T)
            for b in range(8):
                col = np.zeros(8, dtype=complex)
                col[b ^ x] = 1j ** popcount(x & z) * (-1) ** popcount(b & z)
                assert np.allclose(ref[:, b], col)
    assert pauli_label(0b101, 0b110) == "X0 Z1 Y2"
    assert pauli_label(0, 0) == "I"
    assert PauliSum.from_terms(3, [("X0 Z1 Y2", 1.0)]).terms() == [(0b101, 0b110, 1.0)]
    assert pauli_weight(0b101, 0b110) == 3
    assert np.array_equal(pauli_weight([1, 0, 6], [1, 0, 1]), [1, 0, 3])


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_pauli_product_phase_and_commutation_vs_dense(n, rng):
    """P1 P2 = i^phi P(x1^x2, z1^z2), phi = |x1&z1| + |x2&z2| - |x3&z3| + 2|z1&x2| mod 4;
    commutation from the symplectic form; qubit-wise commutation from the single-qubit factors."""
    m = 40
    x1, z1, x2, z2 = (rng.integers(0, 1 << n, m) for _ in range(4))
    phi, x3, z3 = pauli_product(x1, z1, x2, z2)
    comm = pauli_commutes(x1, z1, x2, z2)
    qwc = qubitwise_commutes(x1, z1, x2, z2)
    assert phi.dtype == np.int64 and x3.dtype == np.uint64 and phi.shape == (m,)
    for i in range(m):
        a, b = dense_pauli(x1[i], z1[i], n), dense_pauli(x2[i], z2[i], n)
        assert x3[i] == x1[i] ^ x2[i] and z3[i] == z1[i] ^ z2[i]
        assert np.allclose(a @ b, 1j ** phi[i] * dense_pauli(x3[i], z3[i], n))
        formula = (popcount(x1[i] & z1[i]) + popcount(x2[i] & z2[i]) - popcount(x3[i] & z3[i])
                   + 2 * popcount(z1[i] & x2[i])) % 4
        assert phi[i] == formula
        assert bool(comm[i]) == np.allclose(a @ b, b @ a)
        qwc_ref = all(np.allclose(one_qubit_factor(x1[i], z1[i], k) @ one_qubit_factor(x2[i], z2[i], k),
                                  one_qubit_factor(x2[i], z2[i], k) @ one_qubit_factor(x1[i], z1[i], k))
                      for k in range(n))
        assert bool(qwc[i]) == qwc_ref
    # scalars in, scalars out; broadcasting
    p0, xs0, zs0 = pauli_product(1, 0, 1, 1)   # X Y = i Z
    assert (int(p0), int(xs0), int(zs0)) == (1, 0, 1)
    assert pauli_product(x1[:, None], z1[:, None], x2[None, :], z2[None, :])[0].shape == (m, m)


@pytest.mark.parametrize("n", [32, 33, 64])
def test_pauli_product_and_commutation_large_n_vs_single_qubit_table(n, rng):
    """The product phase, commutation and qubit-wise commutation at 32, 33 and 64 qubits (top bit
    set) agree with per-qubit multiplication by the 2x2 table, which shares no code with the
    module's popcount formulas."""
    m = 200
    x1, z1, x2, z2 = (rng.integers(0, 2 ** 64, m, dtype=np.uint64) >> np.uint64(64 - n) for _ in range(4))
    x1[:50] |= np.uint64(1 << (n - 1))
    z2[25:75] |= np.uint64(1 << (n - 1))
    phi, x3, z3 = pauli_product(x1, z1, x2, z2)
    c1, c2 = to_codes(x1, z1, n), to_codes(x2, z2, n)
    ph, c3 = table_product(c1, c2)
    rx, rz = from_codes(c3)
    assert x3.tolist() == rx and z3.tolist() == rz
    assert np.array_equal(np.array([1, 1j, -1, -1j])[phi], ph)
    differ = (c1 != 0) & (c2 != 0) & (c1 != c2)   # non-identity and different: anticommuting factors
    assert np.array_equal(pauli_commutes(x1, z1, x2, z2), differ.sum(axis=1) % 2 == 0)
    assert np.array_equal(qubitwise_commutes(x1, z1, x2, z2), ~differ.any(axis=1))


# ================================================================ PauliSum
def test_paulisum_canonical_form_and_constructors():
    """Unique (x, z), sorted by x then z, |c| <= tol dropped; immutable; from_terms formats."""
    op = PauliSum(3, [3, 1, 3, 1, 0, 2], [0, 1, 0, 1, 5, 7],
                  [1.0, 2.0, 0.5, -2.0, 1e-15, 3j])
    assert op.terms() == [(2, 7, 3j), (3, 0, 1.5)]   # (1, 1) cancelled, 1e-15 <= DEFAULT_TOL dropped
    keep_small = PauliSum(3, [0, 4], [5, 0], [1e-15, 1.0], tol=0.0)
    assert keep_small.n_terms == 2 and keep_small.simplify(1e-14).n_terms == 1
    assert DEFAULT_TOL == 1e-14
    rng = np.random.default_rng(3)
    big = random_paulisum(rng, 6, 500)
    keys = list(zip(big.xs.tolist(), big.zs.tolist()))
    assert keys == sorted(set(keys))
    assert big.xs.dtype == np.uint64 and big.coeffs.dtype == np.complex128
    for arr in (big.xs, big.zs, big.coeffs):
        with pytest.raises(ValueError):
            arr[0] = 0
    with pytest.raises(AttributeError):
        big.foo = 1
    # from_terms: (x, z) pairs, labels, OpenFermion term keys; repeated terms are summed
    a = PauliSum.from_terms(4, [((0b0011, 0b0010), 1.0), ("X0 Y1", 1.0), (((0, "X"), (1, "Y")), 0.5),
                                ("", 2.0), ("I", 1.0), ("Z3", -1.0)])
    assert a.terms() == [(0, 0, 3.0), (0, 8, -1.0), (3, 2, 2.5)]
    assert PauliSum.identity(4, 2.5).terms() == [(0, 0, 2.5)]
    assert PauliSum.zero(4).n_terms == 0 and len(PauliSum.zero(4)) == 0
    assert PauliSum(4).n_terms == 0 and PauliSum(4, [], [], []).n_terms == 0
    with pytest.raises(TypeError):
        PauliSum(4, [1.5], [0], [1.0])
    with pytest.raises(ValueError):
        PauliSum.from_terms(2, [("X0 Z0", 1.0)])
    with pytest.raises(ValueError):
        PauliSum.from_terms(2, [("X2", 1.0)])
    with pytest.raises(ValueError):
        PauliSum(2, [4], [0], [1.0])
    with pytest.raises(ValueError):
        PauliSum(2, [-1], [0], [1.0])


@pytest.mark.parametrize("n", [3, 5])
def test_paulisum_algebra_vs_dense(n, rng):
    a = random_paulisum(rng, n, 25)
    b = random_paulisum(rng, n, 30)
    da, db = a.to_dense(), b.to_dense()
    eye = np.eye(1 << n)
    checks = {
        "add": (a + b, da + db),
        "sub": (a - b, da - db),
        "neg": (-a, -da),
        "scalar": (2.5j * a, 2.5j * da),
        "rscalar": (a * 2, 2 * da),
        "numpy scalar": (np.float64(2.0) * a, 2 * da),
        "0-d array": (a * np.array(2.0), 2 * da),
        "r 0-d array": (np.array(2.0j) * a, 2j * da),
        "div": (a / 4, da / 4),
        "div 0-d array": (a / np.array(4.0), da / 4),
        "plus identity": (a + 1.5, da + 1.5 * eye),
        "plus 0-d array": (a + np.array(1.5), da + 1.5 * eye),
        "rsub identity": (1.5 - a, 1.5 * eye - da),
        "rsub 0-d array": (np.array(1.5) - a, 1.5 * eye - da),
        "sub 0-d array": (a - np.array(1.5), da - 1.5 * eye),
        "product": (a @ b, da @ db),
        "dagger": (a.dagger(), da.conj().T),
        "commutator": (a.commutator(b), da @ db - db @ da),
        "anticommutator": (a.anticommutator(b), da @ db + db @ da),
        "hermitian part": (a.hermitian_part(), (da + da.conj().T) / 2),
        "antihermitian part": (a.antihermitian_part(), (da - da.conj().T) / 2),
        "sum": (PauliSum.sum([a, b, a]), 2 * da + db),
    }
    for name, (op, ref) in checks.items():
        assert np.allclose(op.to_dense(), ref, atol=1e-12), name
        assert_canonical(op)
    for bad in (np.array([2.0]), "2", None):
        with pytest.raises(TypeError):
            a * bad
        with pytest.raises(TypeError):
            bad * a
    assert a.hermitian_part().is_hermitian() and not a.is_hermitian()
    assert (a.antihermitian_part() * 1j).is_hermitian()
    assert np.isclose(a.norm1(), np.sum(np.abs(a.coeffs)))
    assert np.isclose(a.identity_coefficient(), np.trace(da) / (1 << n))
    w_ref = [sum(1 for k in range(n) if (x >> k) & 1 or (z >> k) & 1) for x, z, _ in a.terms()]
    assert np.array_equal(a.weights(), w_ref)
    assert np.allclose(a.to_sparse().toarray(), da)
    v = rng.standard_normal(1 << n) + 1j * rng.standard_normal(1 << n)
    assert np.allclose(a.matvec(v), da @ v)
    assert np.allclose(a.matvec(np.stack([v, 2 * v], 1)), da @ np.stack([v, 2 * v], 1))
    assert a.allclose(a + 1e-13 * b) and not a.allclose(a + 1e-6 * b)
    with pytest.raises(TypeError):
        a * b
    with pytest.raises(ValueError):
        a + random_paulisum(rng, n + 1, 3)
    # sum() of a list works through __radd__ (0 + op)
    assert sum([a, b]).allclose(a + b)


def test_scalar_operations_are_exact_and_arithmetic_is_scale_invariant(rng):
    """Regression: every operation used to re-apply the absolute cut |c| <= 1e-14, so scaling
    was not invertible, identity(n, 1e-15) was zero and (1e-8 I) @ (1e-8 X) vanished. Now scalar
    scaling is exact (only exact zeros from underflow are dropped), scalars are added to the
    identity coefficient without a cut, and sums and products drop a coefficient only if it
    cancels to within rtol = DEFAULT_TOL of the sum of |contributions|, at every scale."""
    small = PauliSum(4, [1, 2], [0, 0], [1e-3, 2e-3])
    for scaled, ref in ((small * 1e-12, small.coeffs * 1e-12), (small / 1e12, (small.coeffs.real / 1e12) + 0j),
                        (1e-300 * small, 1e-300 * small.coeffs), (small * np.array(1e-20), small.coeffs * 1e-20)):
        assert np.array_equal(scaled.xs, small.xs) and np.array_equal(scaled.coeffs, ref)
    assert np.allclose(((small * 1e-12) / 1e-12).coeffs, small.coeffs, rtol=1e-15, atol=0)
    assert (small * 2e-321).terms() == [(2, 0, 5e-324)]   # 1e-3 * 2e-321 underflows to 0
    assert (small * 0).n_terms == 0 and (0j * small).n_terms == 0
    with pytest.raises(ZeroDivisionError):
        small / 0
    assert (PauliSum(1, [1], [0], [49.0]) / 49).terms() == [(1, 0, 1.0)]   # a true division: 49 * (1/49) != 1
    # exact zeros are dropped even with tol=0 / rtol=0
    assert PauliSum(2, [1, 1], [0, 0], [1.0, -1.0], tol=0).n_terms == 0
    assert PauliSum(2, [1], [0], [1j]).hermitian_part(rtol=0).n_terms == 0
    assert PauliSum(2, [1], [0], [1.0]).antihermitian_part(rtol=0).n_terms == 0
    # the relative threshold itself: |c| <= rtol * sum|t| is dropped, twice that is kept. Here
    # c = 1 - (1 - d) = d exactly and sum|t| = 2 - d, so the boundary is d = 2e-14.
    for d, kept in ((2.0 ** -46, 0), (2.0 ** -45, 1)):   # 1.42e-14 and 2.84e-14
        pair = [PauliSum(1, [1], [0], [1.0]), PauliSum(1, [1], [0], [-(1 - d)])]
        assert PauliSum.sum(pair).n_terms == kept and (pair[0] + pair[1]).n_terms == kept
        lhs = PauliSum.from_terms(1, [("I", 1.0), ("Z0", 1.0)])
        rhs = PauliSum.from_terms(1, [("Z0", 1.0), ("I", -(1 - d))])
        assert (lhs @ rhs).n_terms == 2 * kept                              # I and Z0 both equal d
        assert lhs.multiply(rhs, rtol=0).n_terms == 2
    # identity and scalar addition are exact
    assert PauliSum.identity(2, 1e-15).terms() == [(0, 0, 1e-15)] and PauliSum.identity(2, 0).n_terms == 0
    b1 = PauliSum(3, [1], [0], [1.0])
    assert (b1 + 1e-15).terms() == [(0, 0, 1e-15), (1, 0, 1.0)]
    assert (1e-15 - b1).terms() == [(0, 0, 1e-15), (1, 0, -1.0)]
    a1 = PauliSum(3, [0, 1], [0, 0], [0.5, 2.0])
    assert (a1 + 1e-15).identity_coefficient() == 0.5 + 1e-15 and not a1.allclose(a1 + 1e-15, atol=0)
    assert (a1 + 0) is a1 and (0 + a1) is a1
    # an operator built with tol=0 keeps its small terms through +, sum() and PauliSum.sum
    tiny = PauliSum(3, [1, 2, 3], [0, 0, 0], [1.0, 1e-15, 1e-20], tol=0)
    assert tiny.n_terms == 3
    for same in (tiny + 0, sum([tiny]), PauliSum.sum([tiny]), tiny - PauliSum.zero(3), tiny @ PauliSum.identity(3)):
        assert same.terms() == tiny.terms()
    assert np.array_equal((tiny + tiny).coeffs, 2 * tiny.coeffs)
    assert (PauliSum.identity(2, 1e-8) @ PauliSum(2, [1], [0], [1e-8])).terms() == [(1, 0, 1e-8 * 1e-8)]
    # hermitian parts drop a component only relative to |c|
    assert PauliSum(2, [1], [0], [1e-15], tol=0).hermitian_part().terms() == [(1, 0, 1e-15)]
    assert PauliSum(2, [1], [0], [1e-15 + 1.0j], tol=0).hermitian_part().n_terms == 0
    assert PauliSum(2, [1], [0], [1e-15 + 1.0j], tol=0).hermitian_part(rtol=0).terms() == [(1, 0, 1e-15)]
    # cancellation round-off is still removed, at every scale: 0.1 + 0.2 - 0.3 = 2.8e-17 (not 0)
    zx = PauliSum.from_terms(3, [("Z0", 0.1), ("Z1", 0.2), ("Z2", -0.3)])
    zsum = PauliSum.from_terms(3, [("Z0", 1.0), ("Z1", 1.0), ("Z2", 1.0)])
    for k in (-60, 0, 30):
        s = 2.0 ** k
        parts = [PauliSum(1, [1], [0], [v * s], tol=0) for v in (0.1, 0.2, -0.3)]
        assert PauliSum.sum(parts).n_terms == 0 and (parts[0] + parts[1] + parts[2]).n_terms == 0
        assert PauliSum.sum(parts, rtol=0).n_terms == 1                      # the round-off itself
        prod = (zx * s) @ zsum                                               # I coefficient cancels
        assert prod.identity_coefficient() == 0 and prod.n_terms == 3
        assert (zx * s).multiply(zsum, rtol=0).identity_coefficient() != 0
    # scale invariance: operands rescaled by powers of two give exactly rescaled results
    a, b = random_paulisum(rng, 4, 20), random_paulisum(rng, 4, 25)
    s, t = 2.0 ** -45, 2.0 ** 40   # s * a has coefficients below 1e-14, which the old rule dropped
    cases = {
        "add": ((s * a) + (s * b), (a + b) * s),
        "sub": ((s * a) - (s * b), (a - b) * s),
        "sum": (PauliSum.sum([s * a, s * b, s * a]), PauliSum.sum([a, b, a]) * s),
        "product": ((s * a) @ (t * b), (a @ b) * (s * t)),
        "commutator": ((s * a).commutator(t * b), a.commutator(b) * (s * t)),
        "anticommutator": ((s * a).anticommutator(t * b), a.anticommutator(b) * (s * t)),
        "hermitian part": ((s * a).hermitian_part(), a.hermitian_part() * s),
        "antihermitian part": ((t * a).antihermitian_part(), a.antihermitian_part() * t),
        "plus scalar": ((s * a) + 3 * s, (a + 3) * s),
    }
    for name, (got, ref) in cases.items():
        assert got.n_terms == ref.n_terms > 0, name
        assert np.array_equal(got.xs, ref.xs) and np.array_equal(got.zs, ref.zs), name
        assert np.array_equal(got.coeffs, ref.coeffs), name
        assert_canonical(got)


def test_product_blocking_and_more_than_32_qubits(monkeypatch, rng):
    """Blocked evaluation (loop over blocks of the shorter operand) gives the same product, and
    n > 32 (lexsort path) gives the same algebra as n <= 32 (packed-key path)."""
    a = random_paulisum(rng, 6, 40)
    b = random_paulisum(rng, 6, 90)
    ref = {"ab": a @ b, "ba": b @ a, "comm": a.commutator(b), "anti": a.anticommutator(b)}
    monkeypatch.setattr(pauli_mod, "_CHUNK", 7)
    blocked = {"ab": a @ b, "ba": b @ a, "comm": a.commutator(b), "anti": a.anticommutator(b)}
    for k in ref:
        assert blocked[k].allclose(ref[k], atol=1e-13), k
    monkeypatch.undo()
    shift = 34

    def lift(op):
        return PauliSum(40, op.xs << np.uint64(shift), op.zs << np.uint64(shift), op.coeffs)

    la, lb = lift(a), lift(b)
    assert la.n_qubits == 40 and int(la.xs.max()) >= 1 << shift
    keys = list(zip(la.xs.tolist(), la.zs.tolist()))
    assert keys == sorted(keys)
    for op, small in ((la @ lb, a @ b), (la.commutator(lb), a.commutator(b)), (la + lb, a + b)):
        assert np.array_equal(op.xs >> np.uint64(shift), small.xs)
        assert np.array_equal(op.zs >> np.uint64(shift), small.zs)
        assert np.allclose(op.coeffs, small.coeffs, atol=1e-13)
    assert np.allclose(la.coefficients_of(la.xs, la.zs), la.coeffs)


def test_lookups_and_batch_canonicalization_beyond_32_qubits(rng):
    """n > 32 uses lexsort and per-x searchsorted lookups. Regression for untested paths:
    strings absent from an operator, also those sharing an x with present terms, give 0; the
    batch canonicalization sorts by (x, z) within every row even for mixed x; and jw_products,
    jw_one_body and jw_two_body at 34 qubits are canonical, agree with ladder products built
    with '@', and act on basis states like the Python-int Jordan-Wigner reference."""
    # (1) lookups at n = 40 with several z per x (some x and z above bit 32)
    n = 40
    hi = np.uint64(1 << 39)
    gx = rng.integers(0, 2 ** 40, 6, dtype=np.uint64) | hi
    xs = np.repeat(gx, 5)
    zs = rng.integers(0, 2 ** 40, xs.size, dtype=np.uint64)
    zs[::3] |= hi
    op = PauliSum(n, xs, zs, rng.standard_normal(xs.size) + 1j * rng.standard_normal(xs.size))
    assert_canonical(op)
    table = {(x, z): c for x, z, c in op.terms()}
    qx = np.concatenate([op.xs, op.xs, op.xs, op.xs ^ np.uint64(1 << 36)])
    qz = np.concatenate([op.zs, op.zs ^ np.uint64(1 << 35), op.zs ^ np.uint64(1), op.zs])
    ref = [table.get((int(x), int(z)), 0) for x, z in zip(qx, qz)]
    assert sum(1 for v in ref if v == 0) >= 3 * op.n_terms - 2   # mostly absent strings
    assert np.array_equal(op.coefficients_of(qx, qz), ref)
    # (2) the batch canonicalization itself, with mixed x per row (its contract for any caller)
    rows = rng.integers(0, 4, 300)
    bx = rng.integers(0, 4, 300).astype(np.uint64) << np.uint64(37)   # few x values: duplicates
    bz = rng.integers(0, 4, 300).astype(np.uint64) << np.uint64(35)
    bc = rng.standard_normal(300).astype(complex)
    batch = pauli_mod._canonicalize_rows(n, rows, 5, bx, bz, bc, 0.0)
    assert len(batch) == 5 and batch[4].n_terms == 0
    for r, got in enumerate(batch):
        assert_canonical(got)
        sel = rows == r
        assert got.allclose(PauliSum(n, bx[sel], bz[sel], bc[sel], tol=0), atol=1e-14)
    # (3) JW builders at norb = 17 (34 qubits), both orders
    norb = 17
    for order in ORDERS:
        m = QubitMapping(norb, order)
        orbs = rng.integers(0, norb, (24, 4))
        orbs[:6, 2] = orbs[:6, 0]                                  # some repeated modes
        spins, dags = rng.integers(0, 2, (24, 4)), rng.integers(0, 2, (24, 4))
        spins[:6, 2] = spins[:6, 0]
        batch_ops = np.stack([orbs, spins, dags], axis=-1)
        coeffs = rng.standard_normal(24) + 1j * rng.standard_normal(24)
        prods = jw_products(m, batch_ops, coeffs)
        basis_states = rng.integers(0, 2 ** 34, 40, dtype=np.uint64)
        st = QubitState(2 * norb, basis_states, rng.standard_normal(40))
        for seq, cf, prod in zip(batch_ops.tolist(), coeffs, prods):
            assert_canonical(prod)
            ladder = PauliSum.identity(2 * norb, cf)
            for p, s_, d in seq:
                ladder = ladder @ jw_ladder(m, p, s_, bool(d))
            assert prod.allclose(ladder, atol=1e-15)
            assert np.array_equal(prod.coefficients_of(prod.xs, prod.zs), prod.coeffs)
            # action on basis states vs the Python-int reference
            qops = [(m.qubit(p, s_), bool(d)) for p, s_, d in seq]
            expect = {}
            for b0, a0 in zip(st.basis.tolist(), st.amplitudes.tolist()):
                hit = jw_apply_reference(qops, cf * a0, b0, 2 * norb)
                if hit is not None:
                    expect[hit[0]] = expect.get(hit[0], 0) + hit[1]
            out = st.apply(prod)
            assert sorted(expect) == out.basis.tolist()
            assert np.allclose(out.amplitudes, [expect[b] for b in out.basis.tolist()], atol=1e-14)
        # one- and two-body builders on a sparse random tensor
        f = np.zeros((norb, norb), dtype=complex)
        c = np.zeros((norb,) * 4, dtype=complex)
        for _ in range(6):
            f[tuple(rng.integers(0, norb, 2))] = rng.standard_normal() + 1j * rng.standard_normal()
            c[tuple(rng.integers(0, norb, 4))] = rng.standard_normal() + 1j * rng.standard_normal()
        one, two = jw_one_body(m, f), jw_two_body(m, c)
        assert_canonical(one)
        assert_canonical(two)
        ref_one = PauliSum.sum([f[p, q] * jw_product(m, [(p, s_, True), (q, s_, False)])
                                for p, q in zip(*np.nonzero(f)) for s_ in (0, 1)])
        ref_two = PauliSum.sum([c[p, q, r, s] * jw_product(m, [(p, a_, True), (r, b_, True),
                                                                (s, b_, False), (q, a_, False)])
                                for p, q, r, s in zip(*np.nonzero(c)) for a_ in (0, 1) for b_ in (0, 1)])
        assert one.allclose(ref_one, atol=1e-15) and two.allclose(ref_two, atol=1e-15)


def test_coefficient_lookup_and_coefficient_matrix(rng):
    a = random_paulisum(rng, 5, 20)
    b = random_paulisum(rng, 5, 15)
    x, z, c = a.terms()[3]
    assert a.coefficient(x, z) == c
    absent = next((xx, zz) for xx in range(32) for zz in range(32)
                  if (xx, zz) not in {(t[0], t[1]) for t in a.terms()})
    assert a.coefficient(*absent) == 0
    assert np.allclose(a.coefficients_of(b.xs, b.zs),
                       [a.coefficient(int(xx), int(zz)) for xx, zz in zip(b.xs, b.zs)])
    sub = a.take([5, 1, 1])
    assert sub.terms() == [a.terms()[1], a.terms()[5]]
    xs, zs, w = coefficient_matrix([a, b, a - b])
    keys = list(zip(xs.tolist(), zs.tolist()))
    assert keys == sorted(set(keys)) and w.shape == (3, len(keys))
    for i, op in enumerate([a, b, a - b]):
        rebuilt = PauliSum(5, xs, zs, w[i].toarray().ravel())
        assert rebuilt.allclose(op, atol=1e-14)
    xs2, zs2, w2 = coefficient_matrix([b], basis=(np.arange(32).repeat(32), np.tile(np.arange(32), 32)))
    assert w2.shape == (1, 1024) and np.allclose(w2.toarray().ravel()[xs2 * 32 + zs2],
                                                 b.coefficients_of(xs2, zs2))
    with pytest.raises(ValueError):
        coefficient_matrix([a], basis=(b.xs, b.zs))


def test_openfermion_roundtrip_and_ladder_operator():
    of = pytest.importorskip("openfermion")
    rng = np.random.default_rng(8)
    a = random_paulisum(rng, 5, 30)
    assert PauliSum.from_openfermion(a.to_openfermion(), 5).allclose(a, atol=1e-15)
    m = QubitMapping(3, "interleaved")
    for k in range(6):
        p, s = m.spin_orbital(k)
        for dag in (False, True):
            ref = of.jordan_wigner(of.FermionOperator(((k, int(dag)),)))
            assert jw_ladder(m, p, s, dag).allclose(PauliSum.from_openfermion(ref, 6), atol=1e-15)


# ============================================================== Majoranas
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_majorana_pauli_roundtrip_all_masks(n):
    """Round trip for all 4^n masks; the map is a bijection onto all Pauli strings; the GF(2)
    map x_k = m_2k ^ m_2k+1, z_k = m_2k+1 ^ XOR_{j>k} x_j; phase real iff |S| mod 4 in {0, 1}."""
    masks = np.arange(4 ** n, dtype=np.uint64)
    phase, x, z = majorana_to_pauli(masks)
    back, phase2 = pauli_to_majorana(x, z)
    assert np.array_equal(back, masks)
    assert np.allclose(phase * phase2, 1.0)
    assert len(set(zip(x.tolist(), z.tolist()))) == 4 ** n
    assert int(max(x.max(), z.max())) < 2 ** n
    for mask, xx, zz, ph in zip(masks.tolist(), x.tolist(), z.tolist(), phase):
        xk = [((mask >> (2 * k)) & 1) ^ ((mask >> (2 * k + 1)) & 1) for k in range(n)]
        zk = [((mask >> (2 * k + 1)) & 1) ^ (sum(xk[k + 1:]) % 2) for k in range(n)]
        assert xx == sum(b << k for k, b in enumerate(xk))
        assert zz == sum(b << k for k, b in enumerate(zk))
        size = popcount(mask)
        assert np.isclose(abs(ph), 1) and (abs(ph.imag) < 1e-15) == (size % 4 in (0, 1))
    ph_s, xs, zs = majorana_to_pauli(5)   # scalar in, scalars out
    assert np.ndim(ph_s) == 0 and np.ndim(xs) == 0


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_majorana_monomials_vs_dense_products(n):
    """gamma_2k = a_k + a+_k = Z_<k X_k, gamma_2k+1 = i(a+_k - a_k) = Z_<k Y_k, and every ordered
    monomial equals phase * P(x, z). Even n uses FockSpace's annihilators (independent)."""
    if n % 2 == 0:
        fs = FockSpace(n // 2)
        ann = [fs._ann[k].toarray() for k in range(n)]
    else:
        lower = np.array([[0, 1], [0, 0]], dtype=complex)   # |0><1| annihilates qubit k
        ann = []
        for k in range(n):   # Z_<k |0><1|_k from kron factors (qubit 0 least significant)
            out = np.ones((1, 1), dtype=complex)
            for j in range(n):
                out = np.kron(_Z if j < k else (lower if j == k else _I2), out)
            ann.append(out)
    gam = []
    for k in range(n):
        gam += [ann[k] + ann[k].conj().T, 1j * (ann[k].conj().T - ann[k])]
    for k in range(n):
        assert np.allclose(gam[2 * k], dense_pauli(1 << k, (1 << k) - 1, n))
        assert np.allclose(gam[2 * k + 1], dense_pauli(1 << k, (2 << k) - 1, n))
    for mask in range(4 ** n):
        mono = np.eye(1 << n, dtype=complex)
        for j in range(2 * n):
            if (mask >> j) & 1:
                mono = mono @ gam[j]
        ph, x, z = majorana_to_pauli(mask)
        assert np.allclose(mono, ph * dense_pauli(x, z, n)), mask


@pytest.mark.parametrize("n", [3, 4])
def test_majorana_anticommutation_and_products(n, rng):
    """{gamma_a, gamma_b} = 2 delta_ab, and gamma_S gamma_T = (-1)^{|T & c(S)|} gamma_{S^T}."""
    single = []
    for j in range(2 * n):
        ph, x, z = majorana_to_pauli(1 << j)
        single.append(PauliSum(n, [x], [z], [ph]))
    eye = PauliSum.identity(n)
    for a, b in itertools.product(range(2 * n), repeat=2):
        anti = single[a].anticommutator(single[b])
        assert anti.allclose(2.0 * eye if a == b else PauliSum.zero(n), atol=1e-15)
        assert (single[a] @ single[b] + single[b] @ single[a]).allclose(anti, atol=1e-15)
    s = rng.integers(0, 4 ** n, 300).astype(np.uint64)
    t = rng.integers(0, 4 ** n, 300).astype(np.uint64)
    sign, u = majorana_product(s, t)
    ph_s, xs_, zs_ = majorana_to_pauli(s)
    ph_t, xt, zt = majorana_to_pauli(t)
    ph_u, xu, zu = majorana_to_pauli(u)
    phi, x3, z3 = pauli_product(xs_, zs_, xt, zt)
    assert np.array_equal(u, s ^ t) and np.array_equal(x3, xu) and np.array_equal(z3, zu)
    assert np.allclose(ph_s * ph_t * 1j ** phi, sign * ph_u)


@pytest.mark.parametrize("n", [16, 31, 32])
def test_majorana_bit_tricks_at_large_n(n, rng):
    """Regression for the bit tricks that matter at H6/H8 size and beyond (even-bit compression,
    bit spreading, suffix parity over up to 64-bit masks), which n <= 4 cannot reach: on random
    masks with the top bits set, majorana_to_pauli equals the ordered product of single
    Majoranas multiplied with the 2x2 table, the GF(2) map holds, pauli_to_majorana inverts it,
    and majorana_product's sign satisfies gamma_S gamma_T = sign gamma_{S^T}."""
    top = 2 * n
    masks = (rng.integers(0, 2 ** 64, 240, dtype=np.uint64) >> np.uint64(64 - top)).tolist()
    for i in range(0, 240, 3):
        masks[i] |= 1 << (top - 1)                 # the last Majorana, gamma_{2n-1}
    for i in range(1, 240, 3):
        masks[i] |= 1 << (top - 2)
    full = (1 << top) - 1
    masks += [0, full, 1 << (top - 1), 3 << (top - 2), full & 0x5555555555555555, full & 0xAAAAAAAAAAAAAAAA]
    arr = np.array(masks, dtype=np.uint64)
    phase, x, z = majorana_to_pauli(arr)
    ref_phase, ref_codes = majorana_codes(masks, n)
    rx, rz = from_codes(ref_codes)
    assert x.tolist() == rx and z.tolist() == rz
    assert np.array_equal(phase, ref_phase)
    for mask, xx, zz in zip(masks, rx, rz):                     # the GF(2) map, with Python ints
        xk = [((mask >> (2 * k)) & 1) ^ ((mask >> (2 * k + 1)) & 1) for k in range(n)]
        zk = [((mask >> (2 * k + 1)) & 1) ^ (sum(xk[k + 1:]) & 1) for k in range(n)]
        assert xx == sum(b << k for k, b in enumerate(xk)) and zz == sum(b << k for k, b in enumerate(zk))
    back, phase2 = pauli_to_majorana(x, z)
    assert back.tolist() == masks and np.array_equal(phase * phase2, np.ones(len(masks)))
    # products: gamma_S gamma_T = (phase_S phase_T * table phase) P_U = sign * phase_U P_U
    s_m, t_m = masks[:123], masks[123:]
    sign, u = majorana_product(np.array(s_m, dtype=np.uint64), np.array(t_m, dtype=np.uint64))
    assert u.tolist() == [a ^ b for a, b in zip(s_m, t_m)]
    ph_s, c_s = majorana_codes(s_m, n)
    ph_t, c_t = majorana_codes(t_m, n)
    ph_u, c_u = majorana_codes(u.tolist(), n)
    ph_st, c_st = table_product(c_s, c_t)
    assert np.array_equal(c_st, c_u)
    assert np.array_equal(ph_s * ph_t * ph_st, sign * ph_u)
    assert set(np.unique(sign).tolist()) == {-1.0, 1.0}


# ============================================== mapping and JW vs FockSpace
def test_qubit_mapping():
    blk, itl = QubitMapping(4), QubitMapping(4, "interleaved")
    assert blk.order == "blocked" and blk.n_qubits == 8 and itl.norb == 4
    assert [blk.qubit(p, s) for s in (0, 1) for p in range(4)] == list(range(8))
    assert [itl.qubit(p, s) for p in range(4) for s in (0, 1)] == list(range(8))
    assert np.array_equal(itl.qubit(np.arange(4), 1), [1, 3, 5, 7])
    assert np.array_equal(blk.qubit_of_mode, np.arange(8))
    assert np.array_equal(itl.qubit_of_mode, [0, 2, 4, 6, 1, 3, 5, 7])
    assert np.array_equal(itl.qubit_of_mode[itl.mode_of_qubit], np.arange(8))
    for m in (blk, itl):
        for q in range(8):
            assert m.qubit(*m.spin_orbital(q)) == q
    assert QubitMapping(4) == blk and hash(QubitMapping(4)) == hash(blk) and blk != itl
    with pytest.raises(ValueError):
        QubitMapping(4, "zigzag")
    with pytest.raises(ValueError):
        blk.qubit(4, 0)
    with pytest.raises(ValueError):
        blk.qubit(0, 2)
    # regression: a boolean sigma is the spin 0/1, not a numpy index mask
    assert blk.qubit(1, True) == 5 and type(blk.qubit(1, True)) is int and blk.qubit(True, False) == 1
    assert blk.qubit(np.int64(2), np.uint8(1)) == 6 and itl.qubit(np.int32(3), np.True_) == 7
    assert np.array_equal(blk.qubit(np.array([0, 3]), np.array([True, False])), [4, 3])
    assert blk.qubit([], []).shape == (0,)
    for p, s in ((1, 1.0), (0.5, 0), (np.array([0.0]), 0)):   # floats are rejected, not truncated
        with pytest.raises(TypeError):
            blk.qubit(p, s)
    assert jw_ladder(blk, 1, True, False).allclose(jw_ladder(blk, 1, 1, False), atol=0)
    with pytest.raises(TypeError):
        jw_ladder(blk, 1, 0.5, False)
    with pytest.raises(TypeError):
        jw_rdm2_element(blk, (0, 0.5), (1, 0), (2, 0), (3, 0))
    with pytest.raises(ValueError):
        jw_product(blk, [(0, 0, 2)])


@pytest.mark.parametrize("norb", [2, 3, 4])
def test_jw_ladder_vs_fock_and_car(norb):
    """Blocked-order a_{p sigma}, a+_{p sigma} equal FockSpace's matrices; the canonical
    anticommutation relations hold exactly in the PauliSum algebra."""
    fs = FockSpace(norb)
    m = QubitMapping(norb)
    modes = [(p, s) for s in (0, 1) for p in range(norb)]
    ann = {ps: jw_ladder(m, *ps, False) for ps in modes}
    cre = {ps: jw_ladder(m, *ps, True) for ps in modes}
    for ps in modes:
        assert np.allclose(ann[ps].to_dense(), fs.a(*ps).toarray())
        assert np.allclose(cre[ps].to_dense(), fs.adag(*ps).toarray())
        assert cre[ps].allclose(ann[ps].dagger(), atol=0)
        # a_k = (P(e_k, e_<k) + i P(e_k, e_<=k)) / 2,  e_k = 1 << k
        ek = 1 << m.qubit(*ps)
        assert ann[ps].terms() == [(ek, ek - 1, 0.5), (ek, 2 * ek - 1, 0.5j)]
        assert cre[ps].terms() == [(ek, ek - 1, 0.5), (ek, 2 * ek - 1, -0.5j)]
    eye, zero = PauliSum.identity(2 * norb), PauliSum.zero(2 * norb)
    for i, j in itertools.product(modes, repeat=2):
        assert ann[i].anticommutator(cre[j]).allclose(eye if i == j else zero, atol=1e-15)
        assert ann[i].anticommutator(ann[j]).allclose(zero, atol=1e-15)


@pytest.mark.parametrize("norb", [3, 4])
def test_jw_one_and_two_body_vs_fock(norb, rng):
    """sum f E_pq and sum c e_{pq,rs} for complex, non-Hermitian f and complex c without any
    index symmetry; e_{pq,rs} = E_pq E_rs - delta_qr E_ps in the PauliSum algebra."""
    fs = FockSpace(norb)
    m = QubitMapping(norb)
    f = rng.standard_normal((norb,) * 2) + 1j * rng.standard_normal((norb,) * 2)
    c = rng.standard_normal((norb,) * 4) + 1j * rng.standard_normal((norb,) * 4)
    assert np.allclose(jw_one_body(m, f).to_dense(), fs.one_body(f).toarray(), atol=1e-12)
    assert np.allclose(jw_two_body(m, c).to_dense(), fs.two_body(c).toarray(), atol=1e-11)
    cr = rng.standard_normal((norb,) * 4)
    assert np.allclose(jw_two_body(m, cr).to_dense(), fs.two_body(cr).toarray(), atol=1e-11)
    e_single = np.zeros((norb,) * 4)
    e_single[0, 1, 1, 2 % norb] = 1.0
    E = {(p, q): jw_one_body(m, np.eye(norb)[:, [p]] @ np.eye(norb)[[q], :])
         for p in range(norb) for q in range(norb)}
    assert jw_two_body(m, e_single).allclose(E[0, 1] @ E[1, 2 % norb] - E[0, 2 % norb], atol=1e-15)


def test_jw_hamiltonian_vs_fock(small_random_system):
    """H = ecore + sum h E + 1/2 sum (pq|rs) e_{pq,rs} equals the oracle matrix; real
    coefficients; Hermitian; tr(H)/2^n is the identity coefficient."""
    act = small_random_system
    fs = FockSpace(act.norb)
    h = jw_hamiltonian(act)
    dense = h.to_dense()
    assert np.allclose(dense, fs.hamiltonian(act.h1, act.eri, act.ecore).toarray(), atol=1e-11)
    assert np.all(h.coeffs.imag == 0) and h.is_hermitian(atol=0)
    assert np.allclose(dense, dense.conj().T, atol=1e-13)
    assert np.isclose(h.identity_coefficient(), np.trace(dense) / dense.shape[0])
    # the raw (pre-projection) imaginary parts are pure round-off
    m = QubitMapping(act.norb)
    raw = jw_one_body(m, act.h1) + 0.5 * jw_two_body(m, act.eri)
    assert np.max(np.abs(raw.coeffs.imag), initial=0.0) < 1e-13
    assert (raw + act.ecore).hermitian_part().allclose(h, atol=1e-13)


def test_jw_hamiltonian_rejects_non_hermitian_integrals():
    act = random_active_space(3, (1, 1), seed=2)
    h1 = act.h1.copy()
    h1[0, 1] += 0.3
    with pytest.raises(ValueError, match="Hermitian"):
        jw_hamiltonian(ActiveSpace(h1=h1, eri=act.eri, ecore=0.0, nelec=(1, 1)))


def test_jw_rdm2_element_and_products_vs_fock(rng):
    """a+_i a+_j a_l a_k and arbitrary ladder products equal the oracle's matrix products;
    batch and single versions agree; 2D^{ij}_{kl} = -2D^{ji}_{kl}."""
    norb = 3
    fs = FockSpace(norb)
    m = QubitMapping(norb)
    so = [(p, s) for s in (0, 1) for p in range(norb)]

    def dense_op(ps, dag):
        return (fs.adag(*ps) if dag else fs.a(*ps)).toarray()

    quads = [tuple(so[i] for i in rng.integers(0, 2 * norb, 4)) for _ in range(40)]
    batch = jw_rdm2_elements(m, quads)
    for (i, j, k, l), op in zip(quads, batch):
        ref = dense_op(i, 1) @ dense_op(j, 1) @ dense_op(l, 0) @ dense_op(k, 0)
        assert np.allclose(op.to_dense(), ref)
        assert op.allclose(jw_rdm2_element(m, i, j, k, l), atol=0)
        assert jw_rdm2_element(m, j, i, k, l).allclose(-op, atol=1e-15)
    for length in range(1, 6):
        seqs = [[(*so[q], bool(d)) for q, d in zip(rng.integers(0, 2 * norb, length),
                                                     rng.integers(0, 2, length))] for _ in range(6)]
        coeffs = rng.standard_normal(6) + 1j * rng.standard_normal(6)
        many = jw_products(m, seqs, coeffs)
        for seq, cf, op in zip(seqs, coeffs, many):
            ref = cf * np.eye(fs.dim)
            prod = PauliSum.identity(2 * norb, cf)
            for p, s, d in seq:
                ref = ref @ dense_op((p, s), d)
                prod = prod @ jw_ladder(m, p, s, d)
            assert np.allclose(op.to_dense(), ref)
            assert op.allclose(prod, atol=1e-14)
            assert op.allclose(jw_product(m, seq, cf), atol=0)
    assert jw_product(m, [], 2.0).terms() == [(0, 0, 2.0)]
    assert jw_product(m, [], 1e-15).n_terms == 0 and jw_product(m, [], 1e-15, tol=0).terms() == [(0, 0, 1e-15)]


# ===================================== interleaved order and CI-vector maps
def _order_unitary(norb):
    """U with U[b', b] = sign: |b>_blocked = sign |b'>_interleaved."""
    n = 2 * norb
    b = np.arange(1 << n, dtype=np.uint64)
    b2, sign = QubitMapping(norb).convert_basis(b, QubitMapping(norb, "interleaved"))
    u = np.zeros((1 << n, 1 << n))
    u[b2.astype(np.intp), b.astype(np.intp)] = sign
    return u


def test_interleaved_operators_are_the_conjugated_blocked_ones(rng):
    """Both orders represent the same fermionic operators: O_interleaved = U O_blocked U^T for the
    signed permutation U of QubitMapping.convert_basis."""
    norb = 3
    blk, itl = QubitMapping(norb), QubitMapping(norb, "interleaved")
    u = _order_unitary(norb)
    assert np.allclose(u @ u.T, np.eye(u.shape[0]))
    act = random_active_space(norb, (2, 1), seed=5)
    f = rng.standard_normal((norb,) * 2) + 1j * rng.standard_normal((norb,) * 2)
    c = rng.standard_normal((norb,) * 4) + 1j * rng.standard_normal((norb,) * 4)
    pairs = [(jw_ladder(blk, p, s, d), jw_ladder(itl, p, s, d))
             for p in range(norb) for s in (0, 1) for d in (False, True)]
    pairs += [(jw_one_body(blk, f), jw_one_body(itl, f)), (jw_two_body(blk, c), jw_two_body(itl, c)),
              (jw_hamiltonian(act, blk), jw_hamiltonian(act, itl))]
    for ob, oi in pairs:
        assert np.allclose(u @ ob.to_dense() @ u.T, oi.to_dense(), atol=1e-11)


@pytest.mark.parametrize("nelec", [(2, 2), (3, 1), (1, 2)])
def test_from_ci_both_orders(nelec, rng):
    """Blocked from_ci equals FockSpace.from_ci; interleaved applies the determinant sign
    (-1)^{#{(p in alpha, q in beta): q < p}}, i.e. equals U times the blocked state; to_ci inverts;
    pyscf_sign adds (-1)^{na(na-1)/2 + nb(nb-1)/2}."""
    from pyscf.fci import cistring

    norb = 4
    fs = FockSpace(norb)
    space = FCISpace(norb, nelec)
    u = _order_unitary(norb)
    itl = QubitMapping(norb, "interleaved")
    for complex_ in (False, True):
        ci = space.random(rng, complex_=complex_)
        sb = QubitState.from_ci(norb, nelec, ci)
        si = QubitState.from_ci(norb, nelec, ci, itl)
        assert sb.mapping == QubitMapping(norb) and si.mapping == itl
        assert np.allclose(sb.to_dense(), fs.from_ci(ci, nelec), atol=0)
        assert np.allclose(si.to_dense(), u @ sb.to_dense(), atol=0)
        for st in (sb, si):
            assert np.allclose(st.to_ci(norb, nelec), ci, atol=0)
            assert np.isclose(st.norm(), 1.0)
        g = QubitState.from_ci(norb, nelec, ci, itl, pyscf_sign=True)
        na, nb = nelec
        sign = (-1) ** (na * (na - 1) // 2 + nb * (nb - 1) // 2)
        assert np.allclose(g.to_dense(), sign * si.to_dense(), atol=0)
        assert np.allclose(g.to_ci(norb, nelec, pyscf_sign=True), ci, atol=0)
    # explicit interleaved determinant signs and bit positions
    sa = cistring.make_strings(range(norb), nelec[0])
    sbs = cistring.make_strings(range(norb), nelec[1])
    basis, sign = itl.ci_basis(nelec)
    k = 0
    for a in sa:
        for b in sbs:
            alpha = [p for p in range(norb) if (a >> p) & 1]
            beta = [q for q in range(norb) if (b >> q) & 1]
            inv = sum(1 for p in alpha for q in beta if q < p)
            assert sign[k] == (-1) ** inv
            assert basis[k] == sum(1 << (2 * p) for p in alpha) + sum(1 << (2 * q + 1) for q in beta)
            k += 1


@pytest.mark.parametrize("order", ORDERS)
def test_ladder_operators_reproduce_pyscf_cre_des(order, rng):
    """With pyscf_sign=True, from_ci is PySCF's determinant convention exactly: the JW ladder
    operators act like pyscf.fci.addons.cre_a / des_a / cre_b / des_b (sector-changing)."""
    from pyscf.fci import addons

    norb = 4
    m = QubitMapping(norb, order)
    for nelec in [(2, 1), (1, 2), (2, 2), (3, 2)]:
        ci = FCISpace(norb, nelec).random(rng)
        st = QubitState.from_ci(norb, nelec, ci, m, pyscf_sign=True)
        na, nb = nelec
        ops = [(addons.cre_a, 0, True, (na + 1, nb)), (addons.cre_b, 1, True, (na, nb + 1)),
               (addons.des_a, 0, False, (na - 1, nb)), (addons.des_b, 1, False, (na, nb - 1))]
        for p in range(norb):
            for fn, sigma, dag, new in ops:
                if not (0 <= new[0] <= norb and 0 <= new[1] <= norb):
                    continue
                ref = QubitState.from_ci(norb, new, fn(ci, norb, nelec, p), m, pyscf_sign=True)
                out = st.apply(jw_ladder(m, p, sigma, dag))
                assert (out - ref).norm() < 1e-13, (nelec, p, fn.__name__)


# ======================================================= energies and RDMs
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("name", ["h4", "lih_cas", "n2"])
def test_fci_energy_from_qubit_hamiltonian(name, order):
    """<from_ci(ci_FCI)| jw_hamiltonian |from_ci(ci_FCI)> = E_FCI in both orders (with ecore)."""
    act, e_fci, ci = get_system(name)
    m = QubitMapping(act.norb, order)
    h = jw_hamiltonian(act, m)
    st = QubitState.from_ci(act.norb, act.nelec, ci, m)
    assert np.isclose(st.expectation(h), e_fci, atol=1e-10, rtol=0)
    assert np.isclose(st.vdot(st.apply(h)), e_fci, atol=1e-10, rtol=0)
    assert h.is_hermitian(atol=0)


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("complex_", [False, True])
def test_one_and_two_body_expectations_equal_rdm_contractions(small_random_system, order, complex_, rng):
    """<jw_two_body(c)> = sum c * dm2 and <jw_one_body(f)> = sum f[p,q] dm1[q,p] (PySCF dm1 order),
    for complex non-symmetric c, f and real or complex CI vectors."""
    act = small_random_system
    n = act.norb
    space = FCISpace.of(act)
    m = QubitMapping(n, order)
    ci = space.random(rng, complex_=complex_)
    dm1, dm2 = residual.rdm12(space, ci)
    c = rng.standard_normal((n,) * 4) + 1j * rng.standard_normal((n,) * 4)
    f = rng.standard_normal((n,) * 2) + 1j * rng.standard_normal((n,) * 2)
    st = QubitState.from_ci(n, act.nelec, ci, m)
    assert np.isclose(st.expectation(jw_two_body(m, c)), np.sum(c * dm2), atol=1e-11)
    assert np.isclose(st.expectation(jw_one_body(m, f)), np.einsum("pq,qp->", f, dm1), atol=1e-12)


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("nelec", [(2, 2), (3, 1)])
def test_rdm2_elements_match_pyscf_spin_orbital_rdm(order, nelec, rng):
    """<jw_rdm2_element(i, j, k, l)> equals PySCF's spin-orbital 2-RDM (make_rdm12s, converted to
    Mazziotti order 2D^{ij}_{kl} = <a+_i a+_j a_l a_k>) for all (2N)^4 elements, and the spin sum
    gives the pair-order dm2: sum_{st} 2D^{(p s)(r t)}_{(q s)(s t)} = dm2[p,q,r,s]."""
    norb = 4
    m = QubitMapping(norb, order)
    ci = FCISpace(norb, nelec).random(rng)
    ref = mazziotti_rdm2(ci, norb, nelec)
    so = [(p, s) for s in (0, 1) for p in range(norb)]
    quads = list(itertools.product(so, repeat=4))
    st = QubitState.from_ci(norb, nelec, ci, m)
    ops = jw_rdm2_elements(m, quads)
    vals = np.array([st.expectation(op) for op in ops])
    expect = np.array([ref(*q) for q in quads])
    assert np.abs(expect).max() > 0.1
    assert np.allclose(vals, expect, atol=1e-12)
    # spin sum -> spin-summed pair-order dm2 of residual.rdm12
    d = dict(zip(quads, vals))
    _, dm2 = residual.rdm12(FCISpace(norb, nelec), ci)
    summed = np.zeros((norb,) * 4, dtype=complex)
    for p, q, r, s in itertools.product(range(norb), repeat=4):
        summed[p, q, r, s] = sum(d[(p, a), (r, b), (q, a), (s, b)] for a in (0, 1) for b in (0, 1))
    assert np.allclose(summed, dm2, atol=1e-12)


# ============================================================ OpenFermion
@pytest.mark.parametrize("name", ["h4", "lih_cas", "random4"])
def test_jw_hamiltonian_matches_openfermion_term_by_term(name):
    """jw_hamiltonian equals OpenFermion's jordan_wigner(get_fermion_operator(InteractionOperator))
    term by term: interleaved order directly, blocked order after permuting OpenFermion's spin
    orbitals. The FCI energy with OpenFermion's own sparse matrix (bit-reversed index) checks the
    interleaved determinant signs independently of jw_hamiltonian."""
    of = pytest.importorskip("openfermion")
    act, e_fci, ci = get_system(name)
    n = 2 * act.norb
    for order in ORDERS:
        ref = openfermion_hamiltonian(act, order)
        ours = jw_hamiltonian(act, QubitMapping(act.norb, order))
        theirs = PauliSum.from_openfermion(ref, n, tol=1e-12)
        assert (list(zip(ours.xs.tolist(), ours.zs.tolist()))
                == list(zip(theirs.xs.tolist(), theirs.zs.tolist())))
        assert np.allclose(ours.coeffs, theirs.coeffs, atol=1e-12, rtol=0)
    sparse = of.get_sparse_operator(openfermion_hamiltonian(act, "interleaved"), n_qubits=n)
    v = QubitState.from_ci(act.norb, act.nelec, ci, QubitMapping(act.norb, "interleaved")).to_dense()
    v_of = np.zeros_like(v)
    v_of[bit_reverse(n)] = v
    assert np.isclose(np.vdot(v_of, sparse @ v_of).real, e_fci, atol=1e-10)


# ============================================================== QubitState
def test_qubit_state_basics(rng):
    v = np.zeros(32, dtype=complex)
    v[[3, 9, 17]] = [1.0, 2.0j, -0.5]
    st = QubitState.from_dense(v)
    assert st.n_qubits == 5 and st.size == 3 and np.array_equal(st.basis, [3, 9, 17])
    assert np.allclose(st.to_dense(), v) and np.isclose(st.norm(), np.linalg.norm(v))
    dup = QubitState(5, [9, 3, 9, 17, 30], [1.0, 1.0, 1.0j, -0.5, 0.0])
    assert np.array_equal(dup.basis, [3, 9, 17]) and np.allclose(dup.amplitudes, [1.0, 1 + 1j, -0.5])
    assert np.isclose(st.vdot(dup), np.vdot(v, dup.to_dense()))
    assert np.allclose((st + dup).to_dense(), v + dup.to_dense())
    assert np.allclose((2j * st - dup).to_dense(), 2j * v - dup.to_dense())
    assert np.isclose(st.normalized().norm(), 1.0)
    assert np.allclose((np.array(2.0) * st).to_dense(), 2 * v) and np.allclose((st * np.float32(3)).to_dense(), 3 * v)
    with pytest.raises(TypeError):
        st * np.array([2.0])
    with pytest.raises(ValueError):
        QubitState.from_dense(np.ones(6))
    with pytest.raises(ValueError):
        QubitState(3, [8], [1.0])
    with pytest.raises(AttributeError):
        st.foo = 1
    with pytest.raises(ValueError):
        st.amplitudes[0] = 0
    # from_dense validates the mapping and rejects empty input with a clear message (regression)
    with pytest.raises(ValueError, match="mapping"):
        QubitState.from_dense(np.ones(16), mapping=QubitMapping(4))
    assert QubitState.from_dense(np.ones(16), mapping=QubitMapping(2)).mapping == QubitMapping(2)
    with pytest.raises(ValueError, match="non-empty"):
        QubitState.from_dense(np.zeros(0))
    assert QubitState.from_dense(np.array([2.0])).n_qubits == 0
    # empty operators and the zero state: bare values are 0, normalized ones raise (consistently)
    zero = QubitState(5, [], [])
    assert zero.size == 0 and zero.norm() == 0 and (0.0 * st).size == 0
    assert st.apply(PauliSum.zero(5)).size == 0 and st.expectation(PauliSum.zero(5)) == 0
    assert np.array_equal(zero.expectations([1, 2], [0, 3]), [0, 0])
    assert zero.expectation(PauliSum.identity(5)) == 0 and zero.expectation(PauliSum.zero(5)) == 0
    for call in (lambda: zero.expectations([1], [0], normalize=True),
                 lambda: zero.expectation(PauliSum.identity(5), normalize=True),
                 lambda: zero.expectation(PauliSum.zero(5), normalize=True),
                 lambda: QubitState(5, [3], [0.0], tol=-1).expectation(PauliSum.zero(5), normalize=True)):
        with pytest.raises(ValueError, match="zero state"):
            call()
    with pytest.raises(ValueError):
        st.expectation(PauliSum.identity(4))
    ci_st = QubitState.from_ci(2, (1, 1), np.ones((2, 2)) / 2)
    with pytest.raises(ValueError):
        ci_st.to_ci(3, (1, 1))


def test_states_in_different_qubit_orders_do_not_mix():
    """Regression: +, - and vdot used to accept a blocked and an interleaved state of the same CI
    vector (sb.vdot(si) gave 0 instead of 1, and sb + si was labelled blocked). They now raise;
    a state without a mapping combines with either and passes its partner's mapping on."""
    blk, itl = QubitMapping(2), QubitMapping(2, "interleaved")
    ci = np.array([[0.6, 0.0], [0.0, 0.8]])
    sb, si = QubitState.from_ci(2, (1, 1), ci, blk), QubitState.from_ci(2, (1, 1), ci, itl)
    assert not np.array_equal(sb.basis, si.basis)   # the same determinants at different indices
    for call in (lambda: sb + si, lambda: si - sb, lambda: sb.vdot(si), lambda: si.vdot(sb)):
        with pytest.raises(ValueError, match="mapping"):
            call()
    plain = QubitState(4, sb.basis, sb.amplitudes)
    assert (sb + plain).mapping == blk and (plain + sb).mapping == blk and (plain - si).mapping == itl
    assert (plain + plain).mapping is None
    assert np.isclose(sb.vdot(sb), 1.0) and np.isclose(si.vdot(si), 1.0) and np.isclose(plain.vdot(sb), 1.0)
    with pytest.raises(ValueError, match="n_qubits"):
        sb + QubitState(6, [1], [1.0])


def test_pickle_copy_and_process_pool(rng):
    """Regression: __slots__ plus a raising __setattr__ broke the default pickle and copy
    protocols, so copy.deepcopy failed and ProcessPoolExecutor died with BrokenProcessPool.
    All three classes now round-trip through every pickle protocol (read-only arrays, no cached
    position table), copy and deepcopy return the object itself, and objects cross a spawn
    process pool in both directions."""
    mp = QubitMapping(3, "interleaved")
    op = random_paulisum(rng, 6, 30)
    st = QubitState.from_ci(3, (2, 1), FCISpace(3, (2, 1)).random(rng, complex_=True), mp)
    st.expectations(op.xs, op.zs)   # fills the cached position table
    assert st._index is not None
    for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        op2, st2, mp2 = pickle.loads(pickle.dumps((op, st, mp), protocol=proto))
        assert type(op2) is PauliSum and op2.n_qubits == 6 and op2.terms() == op.terms()
        assert type(st2) is QubitState and st2.mapping == mp and st2._index is None
        assert np.array_equal(st2.basis, st.basis) and np.array_equal(st2.amplitudes, st.amplitudes)
        assert mp2 == mp and mp2.order == "interleaved" and np.array_equal(mp2.mode_of_qubit, mp.mode_of_qubit)
        for arr in (op2.xs, op2.zs, op2.coeffs, st2.basis, st2.amplitudes):
            assert not arr.flags.writeable
        assert np.array_equal(st2.expectations(op.xs, op.zs), st.expectations(op.xs, op.zs))
    for obj in (op, st, mp):
        assert copy.copy(obj) is obj and copy.deepcopy(obj) is obj
    holder = copy.deepcopy({"ops": [op, 2 * op], "state": st, "mapping": mp})
    assert holder["ops"][0] is op and holder["state"] is st and holder["mapping"] is mp
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        norms = list(pool.map(operator.methodcaller("norm1"), [op, 2 * op], timeout=120))
        doubled = pool.submit(operator.mul, op, 2.0).result(timeout=120)
        applied = pool.submit(st.apply, op).result(timeout=120)
    assert norms == [op.norm1(), (2 * op).norm1()]
    assert doubled.terms() == (2.0 * op).terms() and not doubled.coeffs.flags.writeable
    local = st.apply(op)
    assert applied.mapping == mp and np.array_equal(applied.basis, local.basis)
    assert np.allclose(applied.amplitudes, local.amplitudes, rtol=0, atol=1e-14)


@pytest.mark.parametrize("path", ["auto", "direct", "wht", "searchsorted", "small_blocks"])
def test_expectations_vs_dense(path, monkeypatch, rng):
    """expectations(xs, zs) = <psi|P|psi>, the bare matrix element of the spec's formula (divided
    by <psi|psi> with normalize=True), for unnormalized states with dense and sparse supports,
    with many z per x (WHT path) and few (sign-matrix path); both kernels, both lookups and tiny
    blocks agree."""
    if path == "small_blocks":
        monkeypatch.setattr(pauli_mod, "_CHUNK", 64)
    elif path == "direct":
        monkeypatch.setattr(pauli_mod, "_WHT_MAX_QUBITS", -1)
    elif path == "wht":
        monkeypatch.setattr(pauli_mod, "_WHT_FACTOR", 0.0)
    elif path == "searchsorted":
        monkeypatch.setattr(pauli_mod, "_INDEX_MAP_MAX_QUBITS", -1)
    n = 7
    for density in (1.0, 0.3):
        psi = rng.standard_normal(1 << n) + 1j * rng.standard_normal(1 << n)
        psi[rng.random(1 << n) > density] = 0
        st = QubitState.from_dense(psi)
        xs = np.concatenate([rng.integers(0, 1 << n, 50), np.repeat(rng.integers(0, 1 << n, 3), 60)])
        zs = rng.integers(0, 1 << n, xs.size)
        vals = st.expectations(xs, zs)
        ref = np.array([np.vdot(psi, dense_pauli(x, z, n) @ psi) for x, z in zip(xs, zs)])
        nrm2 = np.vdot(psi, psi).real
        assert nrm2 > 10   # far from normalized, so bare and normalized values differ
        assert np.allclose(vals, ref, atol=1e-11)
        assert np.allclose(st.expectations(xs, zs, normalize=True), ref / nrm2, atol=1e-12)
        grid = st.expectations(np.uint64(5), np.arange(1 << n).reshape(8, 16))
        assert grid.shape == (8, 16)
        assert np.isclose(grid[0, 3], np.vdot(psi, dense_pauli(5, 3, n) @ psi))
        op = random_paulisum(rng, n, 40)
        dop = op.to_dense()
        assert np.isclose(st.expectation(op), np.vdot(psi, dop @ psi), atol=1e-10)
        assert np.isclose(st.expectation(op, normalize=True), np.vdot(psi, dop @ psi) / nrm2, atol=1e-12)
        # the matrix element <phi|G|phi> on an applied (unnormalized) state phi = op|psi>
        phi = st.apply(op)
        g = random_paulisum(rng, n, 15)
        dphi = dop @ psi
        assert np.isclose(phi.expectation(g), np.vdot(dphi, g.to_dense() @ dphi), rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("small_blocks", [False, True])
def test_apply_matches_dense_matvec(small_blocks, monkeypatch, rng):
    """apply(op) equals the dense matvec for random complex PauliSums, a Hamiltonian, and ladder
    operators that leave the particle-number sector (also with tiny blocks, which exercises the
    intermediate merging of duplicates)."""
    if small_blocks:
        monkeypatch.setattr(pauli_mod, "_CHUNK", 16)
    norb = 3
    act = random_active_space(norb, (2, 1), seed=9)
    m = QubitMapping(norb)
    ci = FCISpace.of(act).random(rng, complex_=True)
    sector = QubitState.from_ci(norb, act.nelec, ci, m)
    n = 2 * norb
    generic = QubitState.from_dense(rng.standard_normal(1 << n) + 1j * rng.standard_normal(1 << n))
    ops = [random_paulisum(rng, n, 30), jw_hamiltonian(act, m), jw_ladder(m, 1, 0, True),
           jw_ladder(m, 2, 1, False), jw_rdm2_element(m, (0, 0), (1, 1), (2, 0), (2, 1))]
    for st in (sector, generic):
        psi = st.to_dense()
        for op in ops:
            out = st.apply(op)
            assert np.allclose(out.to_dense(), op.to_dense() @ psi, atol=1e-12)
            assert np.allclose(st.apply(op, tol=0).to_dense(), out.to_dense(), atol=1e-12)
            assert np.isclose(st.vdot(out), st.expectation(op, normalize=False), atol=1e-12)
    # H conserves the particle number: no round-off amplitudes outside the sector
    hs = sector.apply(jw_hamiltonian(act, m))
    assert np.isin(hs.basis, m.ci_basis(act.nelec)[0]).all()
    assert not np.isin(sector.apply(ops[0], tol=0).basis, m.ci_basis(act.nelec)[0]).all()
    ref = Hamiltonian(act).apply(ci) + act.ecore * ci
    assert np.allclose(hs.to_ci(norb, act.nelec), ref, atol=1e-12)


@pytest.mark.parametrize("small_blocks", [False, True])
def test_apply_drops_only_round_off(small_blocks, monkeypatch, rng):
    """apply removes only values that vanish within tol of their own contributions. Regression for
    over-truncation, which the O(1) tests above cannot see: amplitudes span 1e-10..1, coefficients
    span 1e-8..1, and three x-groups hold only a near-cancelling pair (c, -(1 - 2^-36) c), so
    |v_x(b)| = 2^-36 |c| at half of the b. Every amplitude agrees with the Kronecker-product
    matvec to within 2 tol E(b'), E(b') = sum_x s_x |psi(b' ^ x)| (the docstring bound tol E(b')
    plus round-off), and every dense entry above that is present. Rescaling op or psi by powers
    of two rescales the result exactly, and apply of (multiples of) I is exact."""
    if small_blocks:
        monkeypatch.setattr(pauli_mod, "_CHUNK", 16)   # intermediate merges of the output
    n, tol = 8, DEFAULT_TOL
    dim = 1 << n
    support = np.flatnonzero(rng.random(dim) < 0.3)   # sparse: some outputs see only small inputs
    psi = np.zeros(dim, dtype=complex)
    psi[support] = 10.0 ** rng.uniform(-10, 0, support.size) * np.exp(2j * np.pi * rng.random(support.size))
    st = QubitState.from_dense(psi)
    gx = rng.choice(np.arange(1, dim - 1), 6, replace=False)   # x != all-ones leaves a free qubit
    xs, zs, cs = [0] * 3, list(rng.choice(dim, 3, replace=False)), list(10.0 ** rng.uniform(-8, 0, 3))
    for x in gx[:3]:                                  # random groups, coefficients 1e-8..1
        xs += [int(x)] * 4
        zs += list(rng.choice(dim, 4, replace=False))
        cs += list(10.0 ** rng.uniform(-8, 0, 4) * np.exp(2j * np.pi * rng.random(4)))
    for x in gx[3:]:                                  # near-cancelling pairs: z2 = z1 ^ w, w & x = 0
        free = [k for k in range(n) if not (int(x) >> k) & 1]
        w = 1 << int(rng.choice(free))
        z1 = int(rng.integers(0, dim))
        c = 10.0 ** rng.uniform(-3, 0) * np.exp(2j * np.pi * rng.random())
        xs += [int(x)] * 2
        zs += [z1, z1 ^ w]
        cs += [c, -(1 - 2.0 ** -36) * c]
    op = PauliSum(n, xs, zs, cs, tol=0)
    assert op.n_terms == 21
    group = {}                                        # M_x = sum_z c_z P(x, z), one entry per column
    s_x = {}
    for x, z, c in op.terms():
        group[x] = group.get(x, 0) + c * dense_pauli(x, z, n)
        s_x[x] = s_x.get(x, 0.0) + abs(c)
    dense = sum(group.values()) @ psi
    b = np.arange(dim)
    scale = sum(s * np.abs(psi[b ^ x]) for x, s in s_x.items())   # E(b')
    out = st.apply(op)
    got = out.to_dense()
    assert np.all(np.abs(got - dense) <= 2 * tol * scale)
    big = np.abs(dense) > 2 * tol * scale
    assert np.isin(np.flatnonzero(big), out.basis).all()
    kept = np.abs(out.amplitudes)
    # small genuine amplitudes, below the old global cut tol * norm1 * max|psi|, are kept
    assert kept.min() < tol * op.norm1() * np.abs(psi).max()
    # the instance detects over-truncation: cuts 1e5 times more aggressive at the first level
    # (v_x(b) <= 1e-9 s_x), at the second level (|a| <= 1e-10 norm1 max|psi|) or the old global
    # second-level rule (|a| <= tol norm1 max|psi|) would each violate the bound somewhere
    first = sum((mx * (np.abs(mx) > 1e-9 * s_x[x])) for x, mx in group.items()) @ psi
    gmax = op.norm1() * np.abs(psi).max()
    for trunc in (first, dense * (np.abs(dense) > 1e-10 * gmax), dense * (np.abs(dense) > tol * gmax)):
        assert np.any(np.abs(trunc - dense) > 2 * tol * scale)
    # exact scaling and the identity
    for s_op, s_psi in ((2.0 ** -40, 1.0), (1.0, 2.0 ** 60), (2.0 ** 30, 2.0 ** -50)):
        scaled = (st * s_psi).apply(op * s_op)
        assert np.array_equal(scaled.basis, out.basis)
        assert np.array_equal(scaled.amplitudes, out.amplitudes * (s_op * s_psi))
    for c in (1.0, 2.0 ** -50, -3.0):
        ident = st.apply(PauliSum.identity(n, c))
        assert np.array_equal(ident.basis, st.basis) and np.array_equal(ident.amplitudes, c * st.amplitudes)
    # tol=0 keeps every nonzero amplitude, and the default drops only what tol=0 shows as round-off
    out0 = st.apply(op, tol=0)
    assert np.isin(out.basis, out0.basis).all()
    assert np.all(np.abs(out0.to_dense() - dense) <= 2 * tol * scale)
    # the second level weighs a contribution by its round-off scale s_x |psi(b)|, not by |v_x(b)|:
    # on |0> + |3>, a(1) = v_X0(0) + v_X1(3) = 2^-20 - 2^-20 (1 - 2^-44) = 2^-64 exactly, where
    # v_X0(0) = 1 - (1 - 2^-20) is itself a near-cancellation with s = 2 - 2^-20
    two = QubitState(2, [0, 3], [1.0, 1.0])
    near = PauliSum(2, [1, 1, 2], [0, 2, 0], [1.0, -(1 - 2.0 ** -20), -(2.0 ** -20) * (1 - 2.0 ** -44)], tol=0)
    assert 1 not in two.apply(near).basis.tolist()
    exact = two.apply(near, tol=0)
    assert exact.amplitudes[exact.basis.tolist().index(1)] == 2.0 ** -64


def test_variance_from_pauli_algebra_matches_fci_ops(rng):
    """<H> and <H^2> - <H>^2 from H @ H, and from |H psi|^2, equal Hamiltonian.energy_and_variance."""
    act, _, _ = get_system("h4")
    ci = FCISpace.of(act).random(rng)
    e_ref, var_ref = Hamiltonian(act).energy_and_variance(ci)
    for order in ORDERS:
        m = QubitMapping(act.norb, order)
        h = jw_hamiltonian(act, m)
        st = QubitState.from_ci(act.norb, act.nelec, ci, m)
        e = st.expectation(h)
        assert np.isclose(e, e_ref, atol=1e-11)
        assert np.isclose(st.expectation(h @ h) - e ** 2, var_ref, atol=1e-10)
        assert np.isclose(st.apply(h).norm() ** 2 - e ** 2, var_ref, atol=1e-10)


# =================================================== efficiency targets
@pytest.fixture(scope="module")
def h8():
    act = systems.hydrogen_chain(8, spacing=1.8, unit="bohr")
    e_fci, ci = act.fci()
    return act, e_fci, ci


@pytest.mark.slow
def test_h8_hamiltonian_build_time_and_size(h8):
    """Target: jw_hamiltonian for H8 (N = 8, 16 qubits) in < 10 s.

    Observed (one thread): about 0.03 s per order, with 2913 Pauli terms in both
    orders. The count does not depend on the order, because the Majorana monomials
    of H do not. With tol=0, several thousand extra terms appear (observed: 3592 to
    3768, depending on the round-off of the integrals). They come from integrals
    that vanish only to round-off (symmetry-forbidden ERIs of about 1e-15), not from
    cancellation. Their largest modulus was observed between 5.1e-16 and 6.6e-16 in
    seven runs with 1 to 4 threads, while the smallest kept coefficient is 1.44e-5.
    The test asserts the separation (round-off below DEFAULT_TOL and below 1e-8 times
    the smallest kept coefficient), not the round-off level itself, which depends on
    the BLAS and CPU.
    """
    act, _, _ = h8
    for order in ORDERS:
        m = QubitMapping(8, order)
        t0 = time.perf_counter()
        h = jw_hamiltonian(act, m)
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.0
        assert h.n_qubits == 16 and h.n_terms == 2913
        smallest = np.min(np.abs(h.coeffs))
        assert h.is_hermitian(atol=0) and smallest > 1e-5
        raw = jw_hamiltonian(act, m, tol=0.0)
        extra = np.sort(np.abs(raw.coeffs))[:raw.n_terms - h.n_terms]
        assert raw.n_terms > h.n_terms
        # the cut lies between the two classes, which are separated by more than 8 decades
        assert extra.max() < DEFAULT_TOL < smallest and extra.max() < 1e-8 * smallest
        assert raw.simplify(DEFAULT_TOL).allclose(h, atol=0)
        # arithmetic applies no absolute cut: raw - h is exactly the round-off terms
        diff = raw - h
        assert diff.n_terms == raw.n_terms - h.n_terms and np.abs(diff.coeffs).max() == extra.max()


@pytest.mark.slow
def test_h8_expectations_time(h8):
    """Target: expectations of 10^4 Pauli strings on an H8 CI state (4900 determinants) in < 10 s.
    Observed (one thread): 10^4 uniformly random strings about 0.5 s; 10^4 products P_i P_j of
    H8 terms (3338 distinct x) about 0.4 s; the 2913 terms of H about 0.07 s."""
    act, e_fci, ci = h8
    m = QubitMapping(8)
    h = jw_hamiltonian(act, m)
    st = QubitState.from_ci(8, act.nelec, ci, m)
    assert ci.size == 4900 and st.size == np.count_nonzero(ci)
    rng = np.random.default_rng(0)
    xr = rng.integers(0, 1 << 16, 10_000)
    zr = rng.integers(0, 1 << 16, 10_000)
    i, j = rng.integers(0, h.n_terms, (2, 10_000))
    _, xp, zp = pauli_product(h.xs[i], h.zs[i], h.xs[j], h.zs[j])
    for xs, zs in ((xr, zr), (xp, zp)):
        t0 = time.perf_counter()
        vals = st.expectations(xs, zs)
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.0
        assert vals.shape == (10_000,) and np.max(np.abs(vals.imag)) < 1e-12
        assert np.max(np.abs(vals)) <= 1 + 1e-12
    t0 = time.perf_counter()
    e = st.expectation(h)
    assert time.perf_counter() - t0 < 10.0
    assert np.isclose(e, e_fci, atol=1e-10)
    # apply stays in the sector and equals the CI-space H (observed: about 0.15 s)
    hs = st.apply(h)
    assert np.isin(hs.basis, m.ci_basis(act.nelec)[0]).all()
    assert np.allclose(hs.to_ci(8, act.nelec), Hamiltonian(act).apply(ci) + act.ecore * ci, atol=1e-12)
    # spot check against the dense formula on a few strings
    psi = st.to_dense()
    b = np.arange(1 << 16)
    for x, z in list(zip(xp[:5], zp[:5])) + list(zip(xr[:3], zr[:3])):
        # float arithmetic: bitwise_count returns uint8, in which 1 - 2 would wrap to 255
        sign = 1.0 - 2.0 * (np.bitwise_count((b & int(z)).astype(np.uint64)) % 2)
        ref = 1j ** popcount(int(x) & int(z)) * np.sum(np.conj(psi[b ^ int(x)]) * sign * psi)
        assert np.isclose(st.expectations(x, z), ref, atol=1e-12)
