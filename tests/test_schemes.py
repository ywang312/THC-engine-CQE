"""Tests of the measurement schemes (``thc_engine_cqe.measure.schemes``).

Every exact variance formula is checked in three independent ways:

* brute force over the whole measurement ensemble on tiny systems: all 3^n
  local bases or all (2n-1)!! perfect matchings, and all outcomes, with dense
  Pauli and Majorana matrices built here from Kronecker products, independent
  of the bit formulas of ``measure.pauli``;
* snapshot Monte Carlo of the actual measurements (the module's validators:
  rotated dense states, sequential projective measurement of Majorana pairs,
  determinants sampled in rotated orbital bases), within 4 standard errors;
* consistency: blocked vs interleaved order, Rao-Blackwellized vs exact, the
  dense expectation table vs ``QubitState.expectations``, small pair blocks.

Targets are built directly with ``measure.pauli`` (not ``measure.targets``):
Q_e = (G_e - G_e^dag)/(2i) and X_e = [G_e - G_e^dag, H]/2 = i[Q_e, H] with
G_e = a+_i a+_j a_l a_k from ``jw_rdm2_elements``.
"""

import contextlib
import functools
import itertools
import math
import time

import numpy as np
import pytest
from pyscf.fci import addons

from thc_engine_cqe import systems
from thc_engine_cqe.factor import double_factorize
from thc_engine_cqe.fci_ops import FCISpace, Hamiltonian, expm_apply, one_body
from thc_engine_cqe.measure import schemes
from thc_engine_cqe.measure.pauli import (PauliSum, QubitMapping, QubitState, jw_hamiltonian,
                                          jw_rdm2_elements, majorana_to_pauli, pauli_to_majorana,
                                          qubitwise_commutes)
from thc_engine_cqe.measure.schemes import (VarianceResult, combine_independent, df_variance,
                                            df_variance_mc, matchgate_covered, matchgate_lambda,
                                            matchgate_p_both, matchgate_shadow_variance,
                                            pauli_bases, pauli_shadow_covered, pauli_shadow_variance,
                                            perfect_matchings, qwc_grouping, qwc_variance,
                                            random_matchings, rotate_to_basis, scheme_variance)

ORDERS = ("blocked", "interleaved")

_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_LETTER = {(1, 0): _X, (1, 1): _Y, (0, 1): _Z}


# ================================================================= helpers
@functools.lru_cache(maxsize=None)
def get_system(name):
    """(active space, E_FCI, ground-state CI)."""
    act = {"h2": lambda: systems.hydrogen_chain(2, 0.74, "angstrom"),
           "h4": lambda: systems.hydrogen_chain(4, 1.8, "bohr")}[name]()
    e, ci = act.fci()
    return act, e, ci


def spin_orbital_elements(norb):
    """All ((i, j), (k, l)) with i < j, k < l (mode order p + sigma N), S_z conserved, (ij) < (kl)."""
    modes = [(p, s) for s in (0, 1) for p in range(norb)]
    pairs = [(a, b) for a in range(2 * norb) for b in range(a + 1, 2 * norb)]
    out = [(modes[i], modes[j], modes[k], modes[l]) for (i, j), (k, l) in itertools.combinations(pairs, 2)
           if modes[i][1] + modes[j][1] == modes[k][1] + modes[l][1]]
    return np.array(out, dtype=np.int64)


def aux_ops(mapping, elements, scale=1.0):
    """scale * Q_e, Q_e = (G - G^dag)/(2i): the coefficients Im(c) of G (exact)."""
    return [PauliSum(mapping.n_qubits, g.xs, g.zs, scale * g.coeffs.imag)
            for g in jw_rdm2_elements(mapping, elements)]


def direct_ops(mapping, elements, ham):
    """X_e = i [Q_e, H] (Hermitian; <X_e> = A_e for real states)."""
    return [q.commutator(ham) * 1j for q in aux_ops(mapping, elements)]


def qstate(name, order, ci=None):
    act, _, ci0 = get_system(name)
    m = QubitMapping(act.norb, order)
    return QubitState.from_ci(act.norb, act.nelec, ci0 if ci is None else ci, m), m


def aux_ci(name, delta):
    """Complex auxiliary state exp(-i delta H) Psi (normalized CI array)."""
    act, _, ci = get_system(name)
    ham = Hamiltonian(act)
    lam = expm_apply(lambda v: -1j * ham.apply(v), ci.astype(complex), delta)
    return lam / np.linalg.norm(lam)


def random_state(n, rng):
    v = rng.standard_normal(1 << n) + 1j * rng.standard_normal(1 << n)
    return QubitState.from_dense(v / np.linalg.norm(v))


def kron_all(factors):
    """factors[k] acts on qubit k (qubit 0 = least significant bit)."""
    out = np.ones((1, 1), dtype=complex)
    for f in factors:
        out = np.kron(f, out)
    return out


def dense_pauli(x, z, n):
    return kron_all([_LETTER.get((int(x) >> k & 1, int(z) >> k & 1), _I2) for k in range(n)])


def dense_majorana(t, n):
    """gamma_{2q} = Z_<q X_q, gamma_{2q+1} = Z_<q Y_q from Kronecker products."""
    q, odd = divmod(t, 2)
    return kron_all([_Z] * q + [_Y if odd else _X] + [_I2] * (n - q - 1))


def dense_monomial(mask, n):
    out = np.eye(1 << n, dtype=complex)
    for t in range(2 * n):
        if mask >> t & 1:
            out = out @ dense_majorana(t, n)
    return out


def majorana_target(n, rng, n_monomials=6, degrees=(2, 4, 6)):
    """Hermitian O = sum_S (a_S gamma_S + conj(a_S) gamma_S^dag) with complex a_S, as a PauliSum."""
    terms = []
    for _ in range(n_monomials):
        k = int(rng.choice([d for d in degrees if d <= 2 * n]))
        mask = sum(1 << int(t) for t in rng.choice(2 * n, size=k, replace=False))
        a = complex(rng.standard_normal(), rng.standard_normal())
        ph, x, z = majorana_to_pauli(mask)                         # gamma_S = ph P(x, z)
        c = a * complex(ph) + np.conj(a * complex(ph))              # a gamma_S + h.c. = 2 Re(a ph) P
        terms.append(((int(x), int(z)), c))
    return PauliSum.from_terms(n, terms)


def assert_within(est, se, ref, k=4.0, atol=1e-12):
    est, se, ref = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (est, se, ref))
    z = np.abs(est - ref) / np.maximum(se, 1e-300)
    assert np.all(np.abs(est - ref) <= k * se + atol), f"max deviation {np.max(z):.2f} standard errors"


# ===================================================== brute-force ensembles
def pauli_shadow_moments(strings, psi, n):
    """E[x_P x_Q] over all 3^n bases and 2^n outcomes, with projector products (dense)."""
    m = len(strings)
    out = np.zeros((m, m))
    for letters in itertools.product(((1, 0), (1, 1), (0, 1)), repeat=n):
        mats = [_LETTER[l] for l in letters]
        for signs in itertools.product((1, -1), repeat=n):
            proj = kron_all([(_I2 + s * mt) / 2 for s, mt in zip(signs, mats)])
            p = float(np.vdot(psi, proj @ psi).real)
            if p < 1e-15:
                continue
            vals = np.zeros(m)
            for i, (x, z) in enumerate(strings):
                v = 1.0
                for k in range(n):
                    pk = (int(x) >> k & 1, int(z) >> k & 1)
                    if pk == (0, 0):
                        continue
                    v *= 3.0 * signs[k] if pk == letters[k] else 0.0
                vals[i] = v
            out += p * np.outer(vals, vals) / 3 ** n
    return out


def matchgate_moments(masks, psi, n):
    """E[conj(x_S) x_S'] over all perfect matchings and outcomes; measured values from post-states."""
    m = len(masks)
    lam = [math.comb(n, bin(s).count("1") // 2) / math.comb(2 * n, bin(s).count("1")) for s in masks]
    gam = [dense_monomial(s, n) for s in masks]
    match = perfect_matchings(2 * n)
    out = np.zeros((m, m), dtype=complex)
    for pairs in match:
        ops = [1j * dense_majorana(a, n) @ dense_majorana(b, n) for a, b in pairs]
        for signs in itertools.product((1, -1), repeat=n):
            proj = np.eye(1 << n, dtype=complex)
            for s, op in zip(signs, ops):
                proj = proj @ (np.eye(1 << n) + s * op) / 2
            phi = proj @ psi
            p = float(np.vdot(phi, phi).real)
            if p < 1e-15:
                continue
            phi /= np.sqrt(p)
            x = np.zeros(m, dtype=complex)
            for i, s in enumerate(masks):
                covered = all((s >> a & 1) == (s >> b & 1) for a, b in pairs)
                if covered:
                    x[i] = np.vdot(phi, gam[i] @ phi) / lam[i]
            out += p * np.outer(np.conj(x), x) / len(match)
    return out


# =========================================================== result object
def test_variance_result_semantics():
    """Scalar targets give one-element arrays; total = sum of per_target; shots = total / eps^2."""
    st, m = qstate("h2", "blocked")
    act, _, _ = get_system("h2")
    h = jw_hamiltonian(act, m)
    r = qwc_variance(h, st)
    assert isinstance(r, VarianceResult) and r.n_targets == 1 and r.method == "exact"
    assert np.isclose(r.total, r.per_target.sum(), rtol=1e-12)
    assert np.isclose(r.expectation[0], get_system("h2")[1], atol=1e-10)
    assert np.isclose(r.shots(1.6e-3), r.total / 1.6e-3 ** 2)
    rv = qwc_variance([h, 2 * h], st)
    assert rv.n_targets == 2 and np.isclose(rv.total, rv.per_target.sum())
    for res in (r, rv):
        assert res.allocation is not None and np.isclose(res.allocation.sum(), 1.0)
    with pytest.raises(ValueError, match="scheme"):
        scheme_variance(h, st, "df")


def test_input_validation(rng):
    st, m = qstate("h2", "blocked")
    h = jw_hamiltonian(get_system("h2")[0], m)
    with pytest.raises(ValueError, match="Hermitian"):
        qwc_variance(h + PauliSum(4, [1], [0], [1j]), st)
    # Hermiticity is checked per target: a small anti-Hermitian target next to a large Hermitian one
    # is rejected (it used to be accepted and silently replaced by its real part, here 0)
    with pytest.raises(ValueError, match="Hermitian.*target 1"):
        qwc_variance([10 * h, PauliSum(4, [3], [0], [1e-11j])], st)
    # ... while round-off imaginary parts are accepted and dropped (relative to the target, or below
    # 1e-14 of the largest coefficient overall)
    noisy = [h + PauliSum(4, [3], [0], [1e-17j], tol=0), PauliSum(4, [3], [0], [1e-16 + 1e-16j], tol=0)]
    r = qwc_variance([h] + noisy, st)
    assert np.isclose(r.per_target[1], r.per_target[0], rtol=1e-12)
    with pytest.raises(ValueError, match="normalized"):
        pauli_shadow_variance(h, st * 2.0)
    with pytest.raises(ValueError, match="qubits"):
        matchgate_shadow_variance(PauliSum(6, [1], [0], [1.0]), st)
    with pytest.raises(ValueError):
        qwc_variance([], st)
    with pytest.raises(ValueError, match="method"):
        pauli_shadow_variance(h, st, method="dense")
    with pytest.raises(ValueError, match="normalized"):
        df_variance(get_system("h2")[0], 2 * get_system("h2")[2])
    # explicit local bases must measure every qubit in X, Y or Z: (bx | bz) == 2^n - 1 (unmeasured
    # qubits used to be accepted and gave -0.06 +- 0 against an exact 2.85)
    for bad in ([(0, 0)] * 10, [(0b0111, 0)], [(0b11111, 0)], [(-1, 15)]):
        with pytest.raises(ValueError, match="every one of the 4 qubits"):
            pauli_shadow_variance(h, st, method="rb", settings=bad)
    # constant targets (identity only, or zero) cost nothing in every scheme and method
    for const in (PauliSum.identity(4, 2.5), PauliSum.zero(4)):
        c0 = const.identity_coefficient().real
        for method in ("exact", "mc"):
            q = qwc_variance(const, st, method=method, n_shots=100, seed=1)
            assert q.total == 0 and q.n_settings == 0 and np.array_equal(q.per_target, [0.0])
            assert np.isclose(q.expectation[0], c0)
            if method == "mc":
                assert q.stderr == 0 and np.array_equal(q.per_target_stderr, [0.0])
                assert np.isclose(q.info["sample_mean"][0], c0)
        for fn in (pauli_shadow_variance, matchgate_shadow_variance):
            for method in ("exact", "rb", "mc"):
                r = fn(const, st, method=method, n_settings=50, n_snapshots=100, seed=1)
                assert r.total == 0 and np.isclose(r.expectation[0], c0)
    mixed = qwc_variance([h, PauliSum.identity(4, 2.5)], st, method="mc", n_shots=1000, seed=2)
    assert mixed.per_target[1] == 0 and mixed.per_target_stderr[1] == 0 and mixed.per_target[0] > 0


# ================================================================= QWC
@pytest.mark.parametrize("order", ORDERS)
def test_qwc_groups_partition_the_terms_and_commute_qubitwise(order):
    """Every non-identity string is in exactly one group, members commute qubit-wise, the union
    string of a group is its measured basis, and the assignment is sorted-insertion first-fit
    (re-derived here by a naive member-by-member implementation) in the order of decreasing weight,
    with round-off ties (consecutive sorted weights within 1e-10 of the largest) in canonical order."""
    act, _, _ = get_system("h4")
    m = QubitMapping(4, order)
    for target in (jw_hamiltonian(act, m), aux_ops(m, spin_orbital_elements(4))):
        g = qwc_grouping(target)
        ident = (g.xs == 0) & (g.zs == 0)
        assert np.all(g.group[ident] == -1) and np.all(g.group[~ident] >= 0)
        assert g.sizes().sum() == np.count_nonzero(~ident)
        for gi in range(g.n_groups):
            mem = g.members(gi)
            xa, xb = np.meshgrid(g.xs[mem], g.xs[mem])
            za, zb = np.meshgrid(g.zs[mem], g.zs[mem])
            assert np.all(qubitwise_commutes(xa, za, xb, zb))
            assert np.bitwise_or.reduce(g.xs[mem]) == g.group_x[gi]
            assert np.bitwise_or.reduce(g.zs[mem]) == g.group_z[gi]
        # sorted insertion: decreasing sum_e |W_ej|, tie classes in canonical order; naive first-fit
        # gives the same groups
        ops = [target] if isinstance(target, PauliSum) else target
        wsum = np.zeros(g.xs.size)
        for op in ops:
            idx = np.searchsorted((g.xs << np.uint64(32)) | g.zs, (op.xs << np.uint64(32)) | op.zs)
            wsum[idx] += np.abs(op.coeffs)
        cand = [j for j in range(g.xs.size) if not ident[j]]
        by_weight = sorted(cand, key=lambda j: -wsum[j])
        tie_class = {by_weight[0]: 0}
        for prev, cur in zip(by_weight, by_weight[1:]):
            tie_class[cur] = tie_class[prev] + int(wsum[prev] - wsum[cur] > 1e-10 * wsum[by_weight[0]])
        assert list(g.order) == sorted(cand, key=lambda j: (tie_class[j], j))
        assert max(tie_class.values()) < len(cand) - 1          # there are ties (alpha/beta partners)
        naive, groups = np.full(g.xs.size, -1), []
        for j in g.order:
            for gi, mem in enumerate(groups):
                if np.all(qubitwise_commutes(g.xs[j], g.zs[j], g.xs[mem], g.zs[mem])):
                    mem.append(j)
                    naive[j] = gi
                    break
            else:
                groups.append([j])
                naive[j] = len(groups) - 1
        assert np.array_equal(naive, g.group)


def test_qwc_grouping_is_insensitive_to_coefficient_round_off(rng):
    """Mathematically equal weights tie, so relative perturbations of every coefficient by 1e-15 or
    1e-13 leave the grouping and the variance unchanged (H4 energy, both orders; 17 H4 aux targets).
    Before the tie rule the order of tied strings followed their round-off: perturbations of 4e-16
    changed the H4 energy variance by up to 22% (37-41 groups instead of 39; observed)."""
    act, _, ci = get_system("h4")
    for order in ORDERS:
        m = QubitMapping(4, order)
        st = QubitState.from_ci(4, act.nelec, ci, m)
        for target in (jw_hamiltonian(act, m), aux_ops(m, spin_orbital_elements(4)[::9], scale=-10.0)):
            ops = [target] if isinstance(target, PauliSum) else target
            ref = qwc_variance(target, st)
            for rel in (1e-15, 1e-15, 1e-15, 1e-13, 1e-13):
                pert = [PauliSum(8, o.xs, o.zs, o.coeffs * (1 + rel * rng.standard_normal(o.n_terms))) for o in ops]
                r = qwc_variance(pert[0] if isinstance(target, PauliSum) else pert, st)
                assert np.array_equal(r.info["grouping"].group, ref.info["grouping"].group)
                assert np.allclose(r.per_target, ref.per_target, rtol=1e-10)


def openfermion_hamiltonian(act, order):
    """H from OpenFermion: jordan_wigner(get_fermion_operator(InteractionOperator)), as a PauliSum."""
    of = pytest.importorskip("openfermion")
    from openfermion.chem.molecular_data import spinorb_from_spatial

    ob, tb = spinorb_from_spatial(act.h1, np.asarray(act.eri.transpose(0, 2, 3, 1), order="C"))
    if order == "blocked":                                  # spinorb_from_spatial is interleaved
        perm = np.array([2 * p + s for s in (0, 1) for p in range(act.norb)])
        ob, tb = ob[np.ix_(perm, perm)], tb[np.ix_(perm, perm, perm, perm)]
    op = of.InteractionOperator(act.ecore, ob, 0.5 * tb)
    return PauliSum.from_openfermion(of.jordan_wigner(of.get_fermion_operator(op)), 2 * act.norb, tol=1e-12)


def test_qwc_grouping_is_the_same_for_openfermions_hamiltonian():
    """H6 at 1.0 A built by OpenFermion's Jordan-Wigner transform has the same strings as
    jw_hamiltonian, with coefficients that differ by round-off (up to 2.1e-15, in 351 and 364 of 919
    terms; observed). The groupings are identical: 204 (blocked) and 312 (interleaved) groups. Before
    the tie rule they differed in both orders (interleaved: 314 groups from jw_hamiltonian, 312 from
    OpenFermion; observed)."""
    act = systems.hydrogen_chain(6, 1.0, "angstrom")
    for order in ORDERS:
        mine = jw_hamiltonian(act, QubitMapping(6, order))
        theirs = openfermion_hamiltonian(act, order)
        assert np.array_equal(mine.xs, theirs.xs) and np.array_equal(mine.zs, theirs.zs)
        assert mine.allclose(theirs, atol=1e-13) and np.any(mine.coeffs != theirs.coeffs)
        a, b = qwc_grouping(mine), qwc_grouping(theirs)
        assert np.array_equal(a.group, b.group) and np.array_equal(a.order, b.order)
        assert a.n_groups == {"blocked": 204, "interleaved": 312}[order]


def test_qwc_single_group_reproduces_the_plain_variance(rng):
    """A target whose strings all commute qubit-wise is one group: V = <O^2> - <O>^2."""
    act, _, ci = get_system("h4")
    m = QubitMapping(4)
    h = jw_hamiltonian(act, m)
    diag = h.take(np.flatnonzero(h.xs == 0))            # Z strings (and I): one group
    st = QubitState.from_ci(4, act.nelec, ci, m)
    r = qwc_variance(diag, st)
    assert r.n_settings == 1
    plain = st.expectation(diag @ diag).real - st.expectation(diag).real ** 2
    assert np.isclose(r.total, plain, rtol=1e-12, atol=1e-13) and np.isclose(r.uniform_total, plain)
    # two targets in one group: per-target plain variances; random complex state
    st2 = random_state(8, rng)
    two = [diag, diag.take(np.arange(1, 6)) * 3.0]
    r2 = qwc_variance(two, st2)
    assert r2.n_settings == 1
    for op, v in zip(two, r2.per_target):
        assert np.isclose(v, st2.expectation(op @ op).real - st2.expectation(op).real ** 2, rtol=1e-12)


def _group_reference(grouping, target_ops, psi, n):
    """Per group, per target: variance from the full outcome distribution of the rotated state."""
    xs, zs = grouping.xs, grouping.zs
    out = np.zeros((grouping.n_groups, len(target_ops)))
    b = np.arange(1 << n, dtype=np.uint64)
    for g in range(grouping.n_groups):
        mem = grouping.members(g)
        prob = np.abs(rotate_to_basis(psi, int(grouping.group_x[g]), int(grouping.group_z[g]))) ** 2
        for e, op in enumerate(target_ops):
            c = op.coefficients_of(xs[mem], zs[mem]).real
            sign = 1.0 - 2.0 * (np.bitwise_count(b[:, None] & (xs[mem] | zs[mem])[None, :]) & 1)
            val = sign @ c
            out[g, e] = prob @ val ** 2 - (prob @ val) ** 2
    return out


def test_rotate_to_basis_matches_kronecker_rotations(rng):
    """rotate_to_basis applies H (X), H S^dag (Y) or I (Z) per qubit; V sigma V^dag = Z."""
    n = 3
    psi = random_state(n, rng).to_dense()
    s_dag = np.diag([1, -1j])
    hd = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    for bx, bz in itertools.product(range(8), repeat=2):
        gates = []
        for k in range(n):
            xb, zb = bx >> k & 1, bz >> k & 1
            gates.append(hd @ s_dag if (xb and zb) else (hd if xb else _I2))
            if xb or zb:
                v = gates[-1]
                assert np.allclose(v @ _LETTER[(xb, zb)] @ v.conj().T, _Z)
        assert np.allclose(rotate_to_basis(psi, bx, bz), kron_all(gates) @ psi)
    # the row-vectorized rotation of the Pauli-shadow validator applies the same gates row by row
    rows = np.stack([random_state(4, rng).to_dense() for _ in range(40)])
    bx, bz = rng.integers(0, 16, (2, 40)).astype(np.uint64)            # X, Y, Z and unmeasured letters
    out = schemes._rotate_rows(rows, bx, bz)
    for r in range(40):
        assert np.allclose(out[r], rotate_to_basis(rows[r], int(bx[r]), int(bz[r])), atol=1e-14)


@pytest.mark.parametrize("order", ORDERS)
def test_qwc_exact_matches_rotated_distributions(order, rng):
    """V_{G,e} from the covariance formula equals the variance over the full outcome distribution of
    each group's rotated state (H4 energy, and the H2 aux targets on a complex auxiliary state)."""
    act, _, ci = get_system("h4")
    m = QubitMapping(4, order)
    h = jw_hamiltonian(act, m)
    st = QubitState.from_ci(4, act.nelec, ci, m)
    r = qwc_variance(h, st)
    ref = _group_reference(r.info["grouping"], [h], st.to_dense(), 8)
    assert np.allclose(r.setting_variances, ref[:, 0], rtol=1e-10, atol=1e-12)
    a2, _, _ = get_system("h2")
    m2 = QubitMapping(2, order)
    ops = aux_ops(m2, spin_orbital_elements(2), scale=-10.0)
    st2 = QubitState.from_ci(2, a2.nelec, aux_ci("h2", 0.1), m2)
    r2 = qwc_variance(ops, st2)
    ref2 = _group_reference(r2.info["grouping"], ops, st2.to_dense(), 4)
    brow, bgrp, vb = r2.info["group_target_variances"]
    dense = np.zeros_like(ref2)
    dense[bgrp, brow] = vb
    assert np.allclose(dense, ref2, rtol=1e-10, atol=1e-12)
    # optimal and uniform per-target variances from the same V_{G,e}
    f = r2.allocation
    assert np.allclose(r2.per_target, (ref2 / f[:, None]).sum(axis=0), rtol=1e-10)
    assert np.allclose(r2.per_target_uniform, r2.n_settings * ref2.sum(axis=0), rtol=1e-10)


def test_optimal_allocation_beats_uniform_and_is_optimal(rng):
    """(sum sqrt V_G)^2 <= n_G sum V_G, and no allocation on the simplex does better."""
    act, _, ci = get_system("h4")
    st, m = qstate("h4", "blocked")
    r = qwc_variance(jw_hamiltonian(act, m), st)
    vg = r.setting_variances
    assert r.total <= r.uniform_total
    assert np.isclose(r.total, np.sum(vg / r.allocation), rtol=1e-12)
    for _ in range(200):
        f = rng.dirichlet(np.ones(vg.size))
        assert np.sum(vg / f) >= r.total * (1 - 1e-12)


def test_qwc_campaign_end_to_end(rng):
    """Simulated S-shot campaigns (H2 energy, optimal allocation): Var(E_hat) * S = total."""
    act, e_fci, ci = get_system("h2")
    st, m = qstate("h2", "blocked")
    h = jw_hamiltonian(act, m)
    r = qwc_variance(h, st)
    g = r.info["grouping"]
    shots, reps = 400, 4000
    s_g = np.maximum(1, np.round(r.allocation * shots)).astype(int)
    est = np.full(reps, h.identity_coefficient().real)
    b = np.arange(16, dtype=np.uint64)
    for gi in range(g.n_groups):
        mem = g.members(gi)
        prob = np.abs(rotate_to_basis(st.to_dense(), int(g.group_x[gi]), int(g.group_z[gi]))) ** 2
        val = (1.0 - 2.0 * (np.bitwise_count(b[:, None] & (g.xs[mem] | g.zs[mem])[None, :]) & 1)) \
            @ h.coefficients_of(g.xs[mem], g.zs[mem]).real
        draws = rng.choice(16, size=(reps, s_g[gi]), p=prob / prob.sum())
        est += val[draws].mean(axis=1)
    predicted = float(np.sum(r.setting_variances / s_g))
    emp = est.var(ddof=1)
    assert abs(emp - predicted) <= 4 * predicted * np.sqrt(2 / (reps - 1))
    assert abs(est.mean() - e_fci) <= 4 * np.sqrt(predicted / reps)
    assert np.isclose(predicted * shots, r.total, rtol=0.05)   # rounding of the allocation only


@pytest.mark.parametrize("order", ORDERS)
def test_qwc_monte_carlo_agrees_with_exact(order):
    """Actual QWC measurements (H4 energy; H2 aux targets on a complex state): every V_hat_G, the
    optimal and uniform totals and the sample means agree with the exact values within 4 sigma."""
    act, _, ci = get_system("h4")
    st, m = qstate("h4", order)
    h = jw_hamiltonian(act, m)
    ex = qwc_variance(h, st)
    mc = qwc_variance(h, st, method="mc", n_shots=20000, seed=11)
    assert mc.method == "mc" and mc.n_settings == ex.n_settings
    assert_within(mc.setting_variances, mc.info["group_stderr"], ex.setting_variances)
    assert_within(mc.total, mc.stderr, ex.total)
    assert_within(mc.uniform_total, mc.uniform_stderr, ex.uniform_total)
    assert_within(mc.info["sample_mean"], mc.info["sample_mean_stderr"], ex.expectation)
    a2, _, _ = get_system("h2")
    m2 = QubitMapping(2, order)
    ops = aux_ops(m2, spin_orbital_elements(2), scale=-10.0)
    st2 = QubitState.from_ci(2, a2.nelec, aux_ci("h2", 0.1), m2)
    ex2 = qwc_variance(ops, st2)
    mc2 = qwc_variance(ops, st2, method="mc", n_shots=20000, seed=12)
    assert_within(mc2.per_target, mc2.per_target_stderr, ex2.per_target)
    assert_within(mc2.total, mc2.stderr, ex2.total)
    assert_within(mc2.info["sample_mean"], mc2.info["sample_mean_stderr"], ex2.expectation)


def _spread_ratio(est, se):
    """SD of the estimates over seeds divided by the mean reported standard error (per column)."""
    est, se = np.asarray(est, dtype=float), np.asarray(se, dtype=float)
    return est.std(axis=0, ddof=1) / se.mean(axis=0)


def _calibration(est, se, ref):
    """(SD / mean stderr per column, SD of the pooled z-scores, scalar?) over seeds (rows).

    Columns whose standard error is 0 for every seed must be exact (|x - <x>| constant per shot, e.g.
    a single string with <P> = 0); they are checked for that and left out of the ratios."""
    est, se = np.asarray(est, dtype=float), np.asarray(se, dtype=float)
    scalar = est.ndim == 1
    est, se = est.reshape(est.shape[0], -1), se.reshape(se.shape[0], -1)
    ref = np.broadcast_to(np.atleast_1d(np.asarray(ref, dtype=float)), est.shape[1:])
    exact = np.all(se == 0, axis=0)
    assert np.allclose(est[:, exact], ref[exact], rtol=1e-12, atol=1e-12)
    est, se, ref = est[:, ~exact], se[:, ~exact], ref[~exact]
    return _spread_ratio(est, se), float(((est - ref) / se).std(ddof=1)), scalar


def test_monte_carlo_standard_errors_are_calibrated():
    """Over 60 seeds (30 for the shadows) the spread of every Monte Carlo estimate matches its
    reported standard error, 0.7 < SD(estimates) / mean(stderr) < 1.4 (and likewise for the pooled
    per-target z-scores), so the 4-sigma checks of the other tests are what they claim. Covered:
    QWC per target and total (17 H4 aux targets on a complex state; per-target errors include the
    dependence of the sampled allocation), the uniform values, the group variances and the sample
    means; DF totals, per-setting variances and the energy (H4); Pauli and matchgate shadows per
    target and total (same aux targets); combine_independent of two independent QWC Monte Carlo
    parts. Observed over 200 seeds on H2 and H4 targets (module docstring): 0.91-1.02 for the
    totals, 0.90-1.05 per target. The sample variances with the plug-in standard errors used before
    gave 0.61 for a symmetric two-point outcome and 0.42-0.78 for the QWC per-target values
    (observed)."""
    act, _, ci = get_system("h4")
    m = QubitMapping(4)
    lam = aux_ci("h4", 0.1)
    st = QubitState.from_ci(4, act.nelec, lam, m)
    st_b = QubitState.from_ci(4, act.nelec, aux_ci("h4", 0.3), m)
    ops = aux_ops(m, spin_orbital_elements(4)[::9], scale=-10.0)
    seeds, few_seeds = range(60), range(30)
    checks = {}

    def collect(name, runs, exact, fields):
        for label, get_est, get_se, ref in fields:
            checks[f"{name} {label}"] = _calibration([get_est(r) for r in runs], [get_se(r) for r in runs], ref)

    ex = qwc_variance(ops, st)
    runs = [qwc_variance(ops, st, method="mc", n_shots=2000, seed=s) for s in seeds]
    collect("qwc", runs, ex, [
        ("total", lambda r: r.total, lambda r: r.stderr, ex.total),
        ("per target", lambda r: r.per_target, lambda r: r.per_target_stderr, ex.per_target),
        ("uniform", lambda r: r.uniform_total, lambda r: r.uniform_stderr, ex.uniform_total),
        ("uniform per target", lambda r: r.per_target_uniform, lambda r: r.info["per_target_uniform_stderr"],
         ex.per_target_uniform),
        ("groups", lambda r: r.setting_variances, lambda r: r.info["group_stderr"], ex.setting_variances),
        ("mean", lambda r: r.info["sample_mean"], lambda r: r.info["sample_mean_stderr"], ex.expectation)])
    exd = df_variance(act, ci)
    runs = [df_variance_mc(act, ci, 2000, seed=100 + s) for s in seeds]
    collect("df", runs, exd, [
        ("total", lambda r: r.total, lambda r: r.stderr, exd.total),
        ("settings", lambda r: r.setting_variances, lambda r: r.info["setting_stderr"], exd.setting_variances),
        ("energy", lambda r: r.info["sample_mean"], lambda r: r.info["sample_mean_stderr"], exd.expectation[0])])
    for fn, n_snap in ((pauli_shadow_variance, 4000), (matchgate_shadow_variance, 1000)):
        exs = fn(ops, st, method="exact")
        runs = [fn(ops, st, method="mc", n_snapshots=n_snap, seed=200 + s) for s in few_seeds]
        collect(fn.__name__, runs, exs, [
            ("total", lambda r: r.total, lambda r: r.stderr, exs.total),
            ("per target", lambda r: r.per_target, lambda r: r.per_target_stderr, exs.per_target),
            ("mean", lambda r: r.info["sample_mean"], lambda r: r.info["sample_mean_stderr"], exs.expectation)])
    exc = combine_independent([qwc_variance(ops, st), qwc_variance(ops, st_b)])
    runs = [combine_independent([qwc_variance(ops, st, method="mc", n_shots=1000, seed=300 + 2 * s),
                                 qwc_variance(ops, st_b, method="mc", n_shots=1000, seed=301 + 2 * s)])
            for s in seeds]
    collect("combined", runs, exc, [("total", lambda r: r.total, lambda r: r.stderr, exc.total),
                                    ("uniform", lambda r: r.uniform_total, lambda r: r.uniform_stderr,
                                     exc.uniform_total)])
    for name, (ratio, z_sd, scalar) in checks.items():
        assert ratio.size and 0.7 < z_sd < 1.4, (name, z_sd)
        if scalar:
            assert 0.7 < ratio[0] < 1.4, (name, ratio)
        else:                       # per column: 30-60 seeds give a sampling spread of 9-13% each
            assert 0.7 < np.median(ratio) < 1.4 and np.all((0.5 < ratio) & (ratio < 2.0)), (name, ratio)


# ======================================================== closed forms
def test_closed_forms_for_single_strings_and_monomials(rng):
    """One Pauli string P with <P> = m: QWC 1 - m^2, Pauli shadows 3^{|P|} - m^2; one Hermitian
    Majorana monomial with |S| = 2k: matchgate shadows 1/lambda_k - m^2."""
    n = 4
    st = random_state(n, rng)
    n_even = 0
    for _ in range(40):
        x, z = (int(v) for v in rng.integers(0, 1 << n, 2))
        if x == z == 0:
            continue
        p = PauliSum(n, [x], [z], [1.0])
        mval = st.expectation(p).real
        w = bin(x | z).count("1")
        assert np.isclose(qwc_variance(p, st).total, 1 - mval ** 2, atol=1e-12)
        assert np.isclose(pauli_shadow_variance(p, st).total, 3 ** w - mval ** 2, atol=1e-10)
        mask, _ = pauli_to_majorana(x, z)
        if bin(int(mask)).count("1") % 2:      # odd monomials are never covered by pairs
            with pytest.raises(ValueError, match="even"):
                matchgate_shadow_variance(p, st)
            continue
        n_even += 1
        k = bin(int(mask)).count("1") // 2
        lam = math.comb(n, k) / math.comb(2 * n, 2 * k)
        assert np.isclose(matchgate_lambda(n, k), lam)
        assert np.isclose(matchgate_shadow_variance(p, st).total, 1 / lam - mval ** 2, rtol=1e-12)
        # a scaled string plus a constant: c^2 (... - m^2), the constant costs nothing
        q = p * 0.3 + 5.0
        assert np.isclose(matchgate_shadow_variance(q, st).total, 0.09 * (1 / lam - mval ** 2), rtol=1e-12)
    assert n_even >= 5


# ====================================================== Pauli shadows
@pytest.mark.parametrize("n", [2, 3])
def test_pauli_shadow_second_moments_by_enumeration(n, rng):
    """E[x_P x_Q] = 3^{|supp P n supp Q|} <PQ> for qubit-wise compatible P, Q, else 0: all bases and
    outcomes enumerated with dense projectors, for every pair of strings; then the module's exact
    variance of a random real combination equals the enumerated one."""
    psi = random_state(n, rng).to_dense()
    strings = [(x, z) for x in range(1 << n) for z in range(1 << n) if x | z]
    mom = pauli_shadow_moments(strings, psi, n)
    for i, (x1, z1) in enumerate(strings):
        for j, (x2, z2) in enumerate(strings):
            if qubitwise_commutes(x1, z1, x2, z2):
                ov = bin((x1 | z1) & (x2 | z2)).count("1")
                ref = 3 ** ov * np.vdot(psi, dense_pauli(x1, z1, n) @ dense_pauli(x2, z2, n) @ psi)
                assert np.isclose(mom[i, j], ref.real, atol=1e-10) and abs(ref.imag) < 1e-12
            else:
                assert abs(mom[i, j]) < 1e-12
    c = rng.standard_normal(len(strings))
    op = PauliSum(n, [s[0] for s in strings], [s[1] for s in strings], c) + 0.7
    mean = sum(ci * np.vdot(psi, dense_pauli(x, z, n) @ psi).real for ci, (x, z) in zip(c, strings))
    st = QubitState.from_dense(psi)
    assert np.isclose(pauli_shadow_variance(op, st).total, c @ mom @ c - mean ** 2, rtol=1e-10)
    # Rao-Blackwellization over all 3^n bases is exact (the 9 bases of n = 2 are too few for a
    # standard error: warning)
    bx, bz = pauli_bases(n)
    with pytest.warns(RuntimeWarning, match="fewer than 10") if n == 2 else contextlib.nullcontext():
        rb = pauli_shadow_variance(op, st, method="rb", settings=zip(bx, bz))
    assert rb.n_samples == 3 ** n and np.isclose(rb.total, c @ mom @ c - mean ** 2, rtol=1e-10)


@pytest.mark.parametrize("order", ORDERS)
def test_pauli_shadow_exact_vs_snapshot_monte_carlo(order):
    """H4 energy and the H2 aux targets (complex state): exact = MC within 4 sigma; the MC means
    reproduce <O> (signs of the Pauli outcomes)."""
    act, _, _ = get_system("h4")
    st, m = qstate("h4", order)
    h = jw_hamiltonian(act, m)
    ex = pauli_shadow_variance(h, st, method="exact")
    mc = pauli_shadow_variance(h, st, method="mc", n_snapshots=100000, seed=21)
    assert_within(mc.total, mc.stderr, ex.total)
    assert_within(mc.info["sample_mean"], mc.info["sample_mean_stderr"], ex.expectation)
    a2, _, _ = get_system("h2")
    m2 = QubitMapping(2, order)
    ops = aux_ops(m2, spin_orbital_elements(2), scale=-10.0)
    st2 = QubitState.from_ci(2, a2.nelec, aux_ci("h2", 0.1), m2)
    ex2 = pauli_shadow_variance(ops, st2)
    mc2 = pauli_shadow_variance(ops, st2, method="mc", n_snapshots=100000, seed=22)
    assert_within(mc2.per_target, mc2.per_target_stderr, ex2.per_target)
    assert_within(mc2.total, mc2.stderr, ex2.total)
    assert_within(mc2.info["sample_mean"], mc2.info["sample_mean_stderr"], ex2.expectation)
    # Rao-Blackwellized estimate over all 81 bases equals the exact value (some targets are covered
    # by fewer than 10 of the 81 bases: warning, infinite per-target standard errors)
    bx, bz = pauli_bases(4)
    with pytest.warns(RuntimeWarning, match="fewer than 10"):
        rb = pauli_shadow_variance(ops, st2, method="rb", settings=zip(bx, bz))
    assert np.allclose(rb.per_target, ex2.per_target, rtol=1e-10, atol=1e-10)


def test_pauli_shadow_monte_carlo_on_a_random_complex_state(rng):
    """Random complex 4-qubit state; a random real combination of about 40 strings with X, Y and Z
    letters, and the eight single-qubit X_k and Y_k: exact = snapshot MC within 4 sigma for every
    per-target variance, the total and every mean. (The real H4/H2 states above do not distinguish
    the X and Y basis rotations. Exchanging them here moves the variance of the combination by 7
    standard errors and the means of the single-qubit targets by 11-56, observed: the shadow of
    Y_k then averages to <X_k>.)"""
    st = random_state(4, rng)
    strings = {(int(x), int(z)) for x, z in rng.integers(0, 16, (40, 2)) if x | z}
    xs, zs = np.array(sorted(strings), dtype=np.uint64).T
    assert np.count_nonzero(xs & zs) > 10                          # Y letters present
    ops = [PauliSum(4, xs, zs, rng.standard_normal(xs.size)) + 0.4]
    ops += [PauliSum(4, [1 << k], [z << k], [1.0]) for z in (0, 1) for k in range(4)]   # X_k, Y_k
    ex = pauli_shadow_variance(ops, st, method="exact")
    mc = pauli_shadow_variance(ops, st, method="mc", n_snapshots=40000, seed=23)
    assert_within(mc.per_target, mc.per_target_stderr, ex.per_target)
    assert_within(mc.total, mc.stderr, ex.total)
    assert_within(mc.info["sample_mean"], mc.info["sample_mean_stderr"], ex.expectation)


# ==================================================== matchgate shadows
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_lambda_and_p_both_by_enumerating_perfect_matchings(n):
    """For 2n <= 8 Majoranas: (2n-1)!! matchings; P(S covered) = lambda_{|S|/2} = C(n,k)/C(2n,2k),
    P(S and S' covered) = P_both(S, S') for every pair of subsets (0 for odd sizes), and the
    bit-swap coverage test equals the definition (S a union of pairs)."""
    m = 2 * n
    match = perfect_matchings(m)
    assert len(match) == math.prod(range(m - 1, 0, -2))
    assert all(sorted(np.ravel(p).tolist()) == list(range(m)) for p in match)
    masks = np.arange(1 << m, dtype=np.uint64)
    cov = matchgate_covered(masks, match)                     # (matchings, subsets)
    ref = np.array([[all((int(s) >> a & 1) == (int(s) >> b & 1) for a, b in p) for s in masks] for p in match])
    assert np.array_equal(cov, ref)
    single = cov.mean(axis=0)
    sizes = np.bitwise_count(masks)
    for s, k2 in zip(range(1 << m), sizes):
        want = 0.0 if k2 % 2 else math.comb(n, k2 // 2) / math.comb(m, k2)
        assert np.isclose(single[s], want, rtol=1e-14, atol=1e-15)
    both = (cov[:, :, None] & cov[:, None, :]).mean(axis=0)
    pb = matchgate_p_both(n, masks[:, None], masks[None, :])
    assert np.allclose(both, pb, rtol=1e-13, atol=1e-15)
    assert np.allclose(np.diag(pb), np.where(sizes % 2, 0, matchgate_lambda(n, sizes // 2)))


@pytest.mark.parametrize("n", [2, 3])
def test_matchgate_second_moments_by_enumeration(n, rng):
    """E[conj(x_S) x_S'] = P_both <gamma_S^dag gamma_S'> / (lambda lambda'): all matchings and all
    outcomes enumerated with dense Majoranas; measured values read off the post-measurement
    states. Then the module's exact (Majorana-form) variance of a target with complex monomial
    coefficients equals the enumerated value."""
    psi = random_state(n, rng).to_dense()
    masks = [s for s in range(1, 1 << (2 * n)) if bin(s).count("1") % 2 == 0]
    mom = matchgate_moments(masks, psi, n)
    for i, s1 in enumerate(masks):
        g1 = dense_monomial(s1, n)
        for j, s2 in enumerate(masks):
            ref = (matchgate_p_both(n, s1, s2) / (matchgate_lambda(n, bin(s1).count("1") // 2)
                                                  * matchgate_lambda(n, bin(s2).count("1") // 2))
                   * np.vdot(psi, g1.conj().T @ dense_monomial(s2, n) @ psi))
            assert np.isclose(mom[i, j], ref, atol=1e-10)
    # Hermitian target with complex monomial coefficients a_S (a_S gamma_S + h.c.)
    a = rng.standard_normal(len(masks)) + 1j * rng.standard_normal(len(masks))
    coef = np.zeros(len(masks), dtype=complex)
    op_dense = np.zeros((1 << n, 1 << n), dtype=complex)
    terms = []
    for i, s in enumerate(masks):
        k = bin(s).count("1") // 2
        # gamma_S^dag = (-1)^k gamma_S: a gamma_S + conj(a) gamma_S^dag = (a + (-1)^k conj(a)) gamma_S
        coef[i] = a[i] + (-1) ** k * np.conj(a[i])
        op_dense += coef[i] * dense_monomial(s, n)
        ph, x, z = majorana_to_pauli(s)
        terms.append(((int(x), int(z)), coef[i] * complex(ph)))
    op = PauliSum.from_terms(n, terms)
    assert np.allclose(op.to_dense(), op_dense) and op.is_hermitian(1e-12)
    mean = np.vdot(psi, op_dense @ psi)
    want = (np.conj(coef) @ mom @ coef - abs(mean) ** 2).real
    st = QubitState.from_dense(psi)
    assert np.isclose(matchgate_shadow_variance(op, st, method="exact").total, want, rtol=1e-10)
    # Rao-Blackwellization over all matchings is exact (3 matchings for n = 2: warning)
    with pytest.warns(RuntimeWarning, match="fewer than 10") if n == 2 else contextlib.nullcontext():
        rb = matchgate_shadow_variance(op, st, method="rb", settings=perfect_matchings(2 * n))
    assert np.isclose(rb.total, want, rtol=1e-10)


def test_matchgate_exact_vs_snapshot_monte_carlo(rng):
    """Sequential projective sampling of the pair operators: H4 energy (blocked), the H2 aux targets
    on a complex state, and a random complex state with a Majorana-form target; exact = MC within
    4 sigma, and every covered term is an eigenvalue +-1 of the post-measurement state."""
    act, _, _ = get_system("h4")
    st, m = qstate("h4", "blocked")
    h = jw_hamiltonian(act, m)
    ex = matchgate_shadow_variance(h, st)
    mc = matchgate_shadow_variance(h, st, method="mc", n_snapshots=20000, seed=31, batch=512)
    assert mc.info["max_eigenvalue_deviation"] < 1e-10
    assert_within(mc.total, mc.stderr, ex.total)
    assert_within(mc.info["sample_mean"], mc.info["sample_mean_stderr"], ex.expectation)
    a2, _, _ = get_system("h2")
    m2 = QubitMapping(2)
    ops = aux_ops(m2, spin_orbital_elements(2), scale=-10.0)
    st2 = QubitState.from_ci(2, a2.nelec, aux_ci("h2", 0.1), m2)
    ex2 = matchgate_shadow_variance(ops, st2)
    mc2 = matchgate_shadow_variance(ops, st2, method="mc", n_snapshots=50000, seed=32)
    assert_within(mc2.per_target, mc2.per_target_stderr, ex2.per_target)
    assert_within(mc2.total, mc2.stderr, ex2.total)
    assert_within(mc2.info["sample_mean"], mc2.info["sample_mean_stderr"], ex2.expectation)
    st3 = random_state(4, rng)
    op3 = majorana_target(4, rng, n_monomials=8)
    ex3 = matchgate_shadow_variance(op3, st3)
    mc3 = matchgate_shadow_variance(op3, st3, method="mc", n_snapshots=50000, seed=33)
    assert_within(mc3.total, mc3.stderr, ex3.total)
    assert_within(mc3.info["sample_mean"], mc3.info["sample_mean_stderr"], ex3.expectation)


def test_matchgate_variance_is_independent_of_the_qubit_order():
    """Blocked and interleaved orders permute the Majoranas: same variances (H4 energy, H4 aux
    targets on a complex state, H4 direct targets), while QWC and Pauli shadows change."""
    act, _, _ = get_system("h4")
    lam = aux_ci("h4", 0.1)
    el = spin_orbital_elements(4)[::5]
    res = {}
    for order in ORDERS:
        m = QubitMapping(4, order)
        h = jw_hamiltonian(act, m)
        st = QubitState.from_ci(4, act.nelec, get_system("h4")[2], m)
        sta = QubitState.from_ci(4, act.nelec, lam, m)
        res[order] = (matchgate_shadow_variance(h, st), matchgate_shadow_variance(aux_ops(m, el), sta),
                      matchgate_shadow_variance(direct_ops(m, el[:6], h), st), qwc_variance(h, st),
                      pauli_shadow_variance(h, st))
    for a, b in zip(res["blocked"][:3], res["interleaved"][:3]):
        assert np.allclose(a.per_target, b.per_target, rtol=1e-10, atol=1e-12)
    assert not np.isclose(res["blocked"][3].total, res["interleaved"][3].total, rtol=1e-3)
    assert not np.isclose(res["blocked"][4].total, res["interleaved"][4].total, rtol=1e-3)


# ================================================ Rao-Blackwellized estimates
def test_rb_coverage_enumeration_equals_bit_swap_test(rng):
    """The RB loop's covered set (unions of <= k_max pairs looked up) equals the bit-swap test; the
    Pauli covered set equals the letter-by-letter definition."""
    act, _, _ = get_system("h4")
    st, m = qstate("h4", "blocked")
    ops = direct_ops(m, spin_orbital_elements(4)[:10], jw_hamiltonian(act, m))
    tg = schemes._prepare(ops)
    setup = schemes._RBSetup(tg, schemes._Expectations(st), "matchgate_shadow")
    assert setup.enumerate
    masks = setup.masks
    for mt in random_matchings(8, 50, rng):
        via_enum = setup.covered(mt)
        bits = np.flatnonzero(matchgate_covered(masks, mt))
        assert bits.size > 0 and np.array_equal(np.sort(via_enum), bits)   # identity column included
    setup.enumerate = False
    for mt in random_matchings(8, 5, rng):
        assert np.array_equal(np.sort(setup.covered(mt)), np.flatnonzero(matchgate_covered(masks, mt)))
    with pytest.raises(ValueError, match="perfect matching"):
        matchgate_shadow_variance(ops, st, method="rb", settings=[np.array([[0, 1], [1, 2]])])
    bx, bz = pauli_bases(8, 50, rng)
    for b1, b2 in zip(bx, bz):
        cov = pauli_shadow_covered(tg.xs, tg.zs, b1, b2)
        ref = [all(((int(x) >> k & 1, int(z) >> k & 1) in ((0, 0), (int(b1) >> k & 1, int(b2) >> k & 1)))
                   for k in range(8)) for x, z in zip(tg.xs, tg.zs)]
        assert np.array_equal(cov, ref)


def test_rao_blackwellized_estimates_match_exact_on_h4():
    """Random settings: matchgate RB of the H4 energy and of H4 direct-route targets X_e, and Pauli
    RB of H4 aux targets, agree with the exact pair sums within 4 standard errors (per target
    and total); the early stop on rel_stderr works."""
    act, _, ci = get_system("h4")
    st, m = qstate("h4", "blocked")
    h = jw_hamiltonian(act, m)
    el = spin_orbital_elements(4)
    xops = direct_ops(m, el[::8], h)
    for target in (h, xops):
        ex = matchgate_shadow_variance(target, st, method="exact")
        rb = matchgate_shadow_variance(target, st, method="rb", n_settings=6000, seed=41)
        assert rb.method == "rb" and rb.n_samples == 6000
        assert_within(rb.per_target, rb.per_target_stderr, ex.per_target)
        assert_within(rb.total, rb.stderr, ex.total)
    sta = QubitState.from_ci(4, act.nelec, aux_ci("h4", 0.1), m)
    qops = aux_ops(m, el[::4])
    ex = pauli_shadow_variance(qops, sta, method="exact")
    rb = pauli_shadow_variance(qops, sta, method="rb", n_settings=20000, seed=42)
    assert_within(rb.total, rb.stderr, ex.total)
    early = matchgate_shadow_variance(xops, st, method="rb", n_settings=50000, rel_stderr=0.05,
                                      min_settings=500, batch=250, seed=43)
    assert early.n_samples < 50000 and early.stderr <= 0.05 * early.total
    # "auto" picks the exact sums for small targets and RB above max_pairs
    assert matchgate_shadow_variance(xops, st).method == "exact"
    with pytest.warns(RuntimeWarning, match="fewer than 10"):
        assert matchgate_shadow_variance(xops, st, max_pairs=10, n_settings=10, seed=1).method == "rb"


def test_rao_blackwellized_vs_snapshot_monte_carlo():
    """RB and snapshot MC estimate the same variance (H4 energy, matchgate): within 4 combined sigma."""
    act, _, _ = get_system("h4")
    st, m = qstate("h4", "interleaved")
    h = jw_hamiltonian(act, m)
    rb = matchgate_shadow_variance(h, st, method="rb", n_settings=6000, seed=51)
    mc = matchgate_shadow_variance(h, st, method="mc", n_snapshots=20000, seed=52, batch=512)
    assert abs(rb.total - mc.total) <= 4 * np.hypot(rb.stderr, mc.stderr)


def test_rb_standard_errors_batches_and_coverage(rng):
    """Over an explicit list of N settings the RB estimate is the plain mean of the per-setting
    values <O_{e,setting}^2> minus <O_e>^2, and its standard errors are their std(ddof=1)/sqrt(N),
    for every batch size (the batch merge). info["n_covering"] counts, per target, the settings that
    cover a non-identity string (here against the bit-swap test). Targets covered fewer than 10 times
    get per_target_stderr = inf with a warning, and so does the total when fewer than 10 settings
    cover anything: before, X0X1 = gamma_1 gamma_2 with no covering setting among 50 gave
    -2.6e-5 +- 0 against an exact 7.0 (observed)."""
    act, _, _ = get_system("h4")
    st, m = qstate("h4", "blocked")
    h = jw_hamiltonian(act, m)
    ops = direct_ops(m, spin_orbital_elements(4)[::40], h) + [h]
    tg = schemes._prepare(ops)
    for scheme, fn in (("matchgate_shadow", matchgate_shadow_variance), ("pauli_shadow", pauli_shadow_variance)):
        setup = schemes._RBSetup(tg, schemes._Expectations(st), scheme)
        if scheme == "matchgate_shadow":
            settings = list(random_matchings(8, 300, rng))
        else:
            settings = list(zip(*pauli_bases(8, 300, rng)))
        vals = np.array([setup.second_moments(setup.covered(s))[0] for s in settings])
        mu = np.array([st.expectation(o).real for o in ops])
        n = len(settings)
        for batch in (1, 3, 256):
            rb = fn(ops, st, method="rb", settings=settings, batch=batch)
            assert rb.n_samples == n and np.all(rb.info["n_covering"] >= 10)
            assert np.allclose(rb.per_target, vals.mean(axis=0) - mu ** 2, rtol=1e-12, atol=1e-12)
            assert np.isclose(rb.total, vals.sum(axis=1).mean() - np.sum(mu ** 2), rtol=1e-12)
            assert np.isclose(rb.stderr, vals.sum(axis=1).std(ddof=1) / np.sqrt(n), rtol=1e-9)
            assert np.allclose(rb.per_target_stderr, vals.std(axis=0, ddof=1) / np.sqrt(n), rtol=1e-9)
        # coverage counts: settings that cover at least one non-identity string of each target
        want = []
        for o in ops:
            keep = (o.xs != 0) | (o.zs != 0)
            if scheme == "matchgate_shadow":
                cov = matchgate_covered(pauli_to_majorana(o.xs[keep], o.zs[keep])[0], np.array(settings))
            else:
                cov = np.array([pauli_shadow_covered(o.xs[keep], o.zs[keep], bx, bz) for bx, bz in settings])
            want.append(int(cov.any(axis=1).sum()))
        assert np.array_equal(rb.info["n_covering"], want)
    # a rarely covered target: gamma_1 gamma_2 is covered with probability 1/15 per matching (n = 8)
    x01 = PauliSum(8, [3], [0], [1.0])
    few = list(random_matchings(8, 30, rng))
    n_hit = int(matchgate_covered(pauli_to_majorana(3, 0)[0], np.array(few)).sum())
    assert n_hit < 10
    with pytest.warns(RuntimeWarning, match="fewer than 10"):
        r1 = matchgate_shadow_variance(x01, st, method="rb", settings=few)
    assert r1.info["n_covering"][0] == n_hit and np.isinf(r1.per_target_stderr[0]) and np.isinf(r1.stderr)
    with pytest.warns(RuntimeWarning, match="fewer than 10"):
        r2 = matchgate_shadow_variance([x01, h], st, method="rb", settings=few)
    assert np.isinf(r2.per_target_stderr[0]) and np.isfinite(r2.per_target_stderr[1]) and np.isfinite(r2.stderr)
    # the early stop on rel_stderr waits for coverage
    r3 = matchgate_shadow_variance(x01, st, method="rb", n_settings=4000, rel_stderr=10.0, min_settings=1,
                                   batch=1, seed=5)
    assert r3.info["n_covering"][0] >= 10 and np.isfinite(r3.stderr) and r3.n_samples < 4000


# ===================================================================== DF
def _fragment_bases(act):
    fac = double_factorize(act.eri)
    hp = act.h1 - 0.5 * np.einsum("prrq->pq", fac.to_eri())
    eps, u0 = np.linalg.eigh(hp)
    return fac, [(u0, eps, False)] + [(u, mu, True) for mu, u in zip(fac.eigenvalues, fac.eigenvectors)]


def test_df_variance_equals_rotated_determinant_distributions():
    """Var_l = 1/4 (<F_l^4> - <F_l^2>^2) (and Var_0 of h'.E) equal the variances of the diagonal
    fragment operators over the determinant distribution of the CI vector rotated into each
    fragment's basis; transform_ci's convention is checked against exp(-K) with U = exp(kappa)."""
    import scipy.linalg

    act, e_fci, ci = get_system("h4")
    res = df_variance(act, ci)
    fac, frags = _fragment_bases(act)
    assert res.n_settings == fac.n_leaves + 1 and abs(res.info["bias"]) < 1e-9
    assert np.isclose(res.expectation[0], e_fci, atol=1e-10)
    space = FCISpace.of(act)
    occ = schemes._determinant_occupations(act.norb, act.nelec)
    for l, (u, ev, square) in enumerate(frags):
        prob = (np.abs(addons.transform_ci(ci, act.nelec, u)) ** 2).ravel()
        lin = occ @ ev
        val = 0.5 * lin ** 2 if square else lin
        assert np.isclose(prob.sum(), 1.0, atol=1e-12)
        assert np.isclose(prob @ val, res.info["fragment_means"][l], atol=1e-10)
        assert np.isclose(prob @ val ** 2 - (prob @ val) ** 2, res.setting_variances[l], rtol=1e-9, atol=1e-12)
    # rotation convention: phi'_k = sum_p phi_p U[p, k]; coefficients <D'|psi> = <D| exp(-K) |psi>
    u = frags[1][0].copy()
    if np.linalg.det(u) < 0:
        u[:, 0] *= -1
    kappa = scipy.linalg.logm(u).real
    assert np.allclose(scipy.linalg.expm(kappa), u, atol=1e-10)
    rotated = expm_apply(lambda v: -one_body(space, kappa, v), ci, 1.0)
    assert np.allclose(np.abs(rotated) ** 2, np.abs(addons.transform_ci(ci, act.nelec, u)) ** 2, atol=1e-10)
    # complex states (auxiliary states exp(-i delta H) Psi) work the same way
    lam = aux_ci("h4", 0.3)
    rc = df_variance(act, lam)
    for l, (u, ev, square) in enumerate(frags[:4]):
        prob = (np.abs(addons.transform_ci(lam, act.nelec, u)) ** 2).ravel()
        val = 0.5 * (occ @ ev) ** 2 if square else occ @ ev
        assert np.isclose(prob @ val ** 2 - (prob @ val) ** 2, rc.setting_variances[l], rtol=1e-9, atol=1e-12)
    assert np.isclose(rc.expectation[0], Hamiltonian(act).energy(lam), atol=1e-12)
    # a QubitState input gives the same result
    st, _ = qstate("h4", "interleaved")
    assert np.isclose(df_variance(act, st).total, res.total, rtol=1e-12)


def test_df_monte_carlo_and_truncation_bias():
    """Sampling determinants in the rotated bases reproduces every Var_l, the optimal and uniform
    totals and the energy within 4 sigma; a truncated factorization reports
    bias = <H[eri_DF]> - <H>, computed here independently."""
    act, e_fci, ci = get_system("h4")
    ex = df_variance(act, ci)
    mc = df_variance_mc(act, ci, 20000, seed=61)
    assert_within(mc.setting_variances, mc.info["setting_stderr"], ex.setting_variances)
    assert_within(mc.total, mc.stderr, ex.total)
    assert_within(mc.uniform_total, mc.uniform_stderr, ex.uniform_total)
    assert_within(mc.info["sample_mean"], mc.info["sample_mean_stderr"], e_fci)
    assert np.allclose(mc.info["fragment_means_exact_distribution"], ex.info["fragment_means"], atol=1e-10)
    tr = df_variance(act, ci, n_leaves=4)
    fac = tr.info["factorization"]
    assert fac.n_leaves == 4 and tr.n_settings == 5
    e_model = Hamiltonian(act.with_eri(fac.to_eri())).energy(ci)
    assert np.isclose(tr.info["energy_model"], e_model, atol=1e-10)
    assert np.isclose(tr.info["bias"], e_model - e_fci, atol=1e-10) and abs(tr.info["bias"]) > 1e-4
    mc_tr = df_variance_mc(act, ci, 20000, n_leaves=4, seed=62)
    assert_within(mc_tr.setting_variances, mc_tr.info["setting_stderr"], tr.setting_variances)
    assert_within(mc_tr.info["sample_mean"], mc_tr.info["sample_mean_stderr"], e_model)
    # truncated leaf ranks (eig_threshold): the sampler completes each leaf's orbitals to an
    # orthonormal basis with mu = 0 on the added orbitals
    tre = df_variance(act, ci, eig_threshold=0.05)
    ranks = [u.shape[1] for u in tre.info["factorization"].eigenvectors]
    assert min(ranks) < act.norb
    mc_e = df_variance_mc(act, ci, 20000, eig_threshold=0.05, seed=64)
    assert np.allclose(mc_e.info["fragment_means_exact_distribution"], tre.info["fragment_means"], atol=1e-10)
    assert_within(mc_e.setting_variances, mc_e.info["setting_stderr"], tre.setting_variances)
    assert_within(mc_e.total, mc_e.stderr, tre.total)


def test_df_shifted_leaves_and_eigendecomposition_factors():
    """shift=True measures 1/2 (F_l - tr L_l)^2 (E~ = E - delta) with T' = h' + sum_l tr(L_l) L_l in
    fragment 0: the same operator (same model energy), a different split. Its variances equal the
    rotated-distribution values and the Monte Carlo ones; first="eigh" factors work as well."""
    act, e_fci, ci = get_system("h4")
    plain = df_variance(act, ci)
    sh = df_variance(act, ci, shift=True)
    assert np.isclose(sh.info["energy_model"], plain.info["energy_model"], atol=1e-10)
    fac = sh.info["factorization"]
    leaves = fac.leaves()
    c = np.einsum("lpp->l", leaves)
    assert np.allclose(sh.info["shifts"], c)
    tprime = act.h1 - 0.5 * np.einsum("prrq->pq", fac.to_eri()) + np.einsum("l,lpq->pq", c, leaves)
    occ = schemes._determinant_occupations(act.norb, act.nelec)
    eps, u0 = np.linalg.eigh(tprime)
    frags = [(u0, occ @ eps)] + [(u, 0.5 * (occ @ mu - cl) ** 2)
                                 for mu, u, cl in zip(fac.eigenvalues, fac.eigenvectors, c)]
    for l, (u, val) in enumerate(frags):
        prob = (np.abs(addons.transform_ci(ci, act.nelec, u)) ** 2).ravel()
        assert np.isclose(prob @ val ** 2 - (prob @ val) ** 2, sh.setting_variances[l], rtol=1e-9, atol=1e-12)
    mc = df_variance_mc(act, ci, 20000, shift=True, seed=63)
    assert_within(mc.setting_variances, mc.info["setting_stderr"], sh.setting_variances)
    assert_within(mc.info["sample_mean"], mc.info["sample_mean_stderr"], e_fci)
    eig = df_variance(act, ci, first="eigh", tol=1e-12)
    assert abs(eig.info["bias"]) < 1e-9 and eig.total != plain.total
    with pytest.raises(ValueError, match="either"):
        df_variance(act, ci, factorization=fac, n_leaves=2)


# ======================================================== combination, internals
def test_combine_independent_experiments():
    """Optimal allocation across experiments: (sum sqrt V_l)^2; per-target sum_l v_le / f_l. The
    uniform values are those of a uniform allocation over all settings of all parts (None for
    shadows); equal shots per part (optimal inside) is info["uniform_over_parts"]. Standard errors of
    the parts combine by the delta method (exact parts count as exact)."""
    st, m = qstate("h2", "blocked")
    ops = aux_ops(m, spin_orbital_elements(2))
    parts = [matchgate_shadow_variance(ops, st), pauli_shadow_variance([o * 2 for o in ops], st)]
    c = combine_independent(parts)
    v = np.array([p.total for p in parts])
    assert np.isclose(c.total, np.sum(np.sqrt(v)) ** 2) and np.isclose(c.total, c.per_target.sum())
    assert c.n_settings is None and c.uniform_total is None and c.per_target_uniform is None
    assert np.isclose(c.info["uniform_over_parts"], 2 * v.sum()) and c.total <= c.info["uniform_over_parts"]
    assert np.allclose(c.info["per_target_uniform_over_parts"], 2 * (parts[0].per_target + parts[1].per_target))
    assert np.allclose(c.expectation, parts[0].expectation + parts[1].expectation)
    assert np.allclose(c.per_target, sum(p.per_target / f for p, f in zip(parts, c.allocation)))
    assert c.stderr is None
    # nested optimal allocations (QWC groups inside parts) equal the global optimum over all groups;
    # the uniform allocation is over all 28 groups: n_tot sum_l sum_G V_lG (per target likewise)
    q = [qwc_variance(ops, st), qwc_variance([o * 3 for o in ops[::-1]], st)]
    cq = combine_independent(q)
    allv = np.concatenate([r.setting_variances for r in q])
    n_tot = q[0].n_settings + q[1].n_settings
    assert cq.n_settings == n_tot == allv.size
    assert np.isclose(cq.total, np.sum(np.sqrt(allv)) ** 2, rtol=1e-12)
    assert np.isclose(cq.uniform_total, n_tot * allv.sum(), rtol=1e-12)
    per_g = np.zeros(len(ops))
    for r in q:
        brow, bgrp, vb = r.info["group_target_variances"]
        per_g += np.bincount(brow, weights=vb, minlength=len(ops))
    assert np.allclose(cq.per_target_uniform, n_tot * per_g, rtol=1e-12)
    assert np.isclose(cq.uniform_total, cq.per_target_uniform.sum())
    # mixed exact and Monte Carlo parts (H2 energy): the exact part contributes no error
    h = jw_hamiltonian(get_system("h2")[0], m)
    ex, mc = qwc_variance(h, st), qwc_variance(h, st, method="mc", n_shots=2000, seed=3)
    cm = combine_independent([ex, mc])
    sq = np.sqrt([ex.total, mc.total])
    assert mc.stderr > 0 and np.isclose(cm.stderr, sq.sum() * mc.stderr / sq[1], rtol=1e-12)
    assert np.isclose(cm.uniform_stderr, (ex.n_settings + mc.n_settings) * mc.uniform_stderr / mc.n_settings)


def test_table_and_fallback_expectations_and_pair_blocks_agree(monkeypatch, rng):
    """The dense <P> table and QubitState.expectations give the same variances; tiny pair blocks
    (big-row path, buffer merges) change nothing."""
    act, _, _ = get_system("h4")
    st, m = qstate("h4", "blocked")
    h = jw_hamiltonian(act, m)
    ops = aux_ops(m, spin_orbital_elements(4)[::3])
    sta = QubitState.from_ci(4, act.nelec, aux_ci("h4", 0.2), m)
    assert schemes._Expectations(st).table is not None
    xs, zs = rng.integers(0, 256, (2, 500)).astype(np.uint64)
    exact = sta.expectations(xs, zs).real
    assert np.allclose(schemes._Expectations(sta)(xs, zs), exact, atol=1e-14)
    ref = [qwc_variance(h, st), pauli_shadow_variance(h, st), matchgate_shadow_variance(h, st),
           qwc_variance(ops, sta), pauli_shadow_variance(ops, sta), matchgate_shadow_variance(ops, sta)]
    monkeypatch.setattr(schemes, "_TABLE_MAX", 0)
    monkeypatch.setattr(schemes, "_PAIR_CHUNK", 257)
    assert schemes._Expectations(st).table is None
    new = [qwc_variance(h, st), pauli_shadow_variance(h, st), matchgate_shadow_variance(h, st),
           qwc_variance(ops, sta), pauli_shadow_variance(ops, sta), matchgate_shadow_variance(ops, sta)]
    for a, b in zip(ref, new):
        assert np.allclose(a.per_target, b.per_target, rtol=1e-11, atol=1e-12)
    assert np.allclose(schemes._Expectations(sta)(xs, zs), exact, atol=1e-14)
    # a dense 12-qubit state cannot have a table (d 2^n > 2^23): rejected before the d^2 pair XORs
    # (which took 1 s and 134 MB; observed); now about 1e-5 s (observed)
    monkeypatch.undo()
    dense = random_state(12, rng)
    t0 = time.perf_counter()
    oracle = schemes._Expectations(dense)
    assert oracle.table is None and time.perf_counter() - t0 < 0.2


def test_h6_aux_route_vector_targets_time():
    """Performance target: aux-route vector variances for the 840 spin-orbital Q_e of H6 (1.0 A)
    exactly, all three Pauli-level schemes, within a few minutes. Observed (one thread, complex
    auxiliary state exp(-0.1 i H) Psi_FCI): 0.3-0.7 s per scheme and order; 757 (blocked) and
    825 (interleaved) QWC groups."""
    act = systems.hydrogen_chain(6, 1.0, "angstrom")
    e, ci = act.fci()
    ham = Hamiltonian(act)
    lam = expm_apply(lambda v: -1j * ham.apply(v), ci.astype(complex), 0.1)
    el = spin_orbital_elements(6)
    assert len(el) == 840
    m = QubitMapping(6)
    ops = aux_ops(m, el, scale=-10.0)
    st = QubitState.from_ci(6, act.nelec, lam / np.linalg.norm(lam), m)
    t0 = time.perf_counter()
    out = {s: scheme_variance(ops, st, s) for s in schemes.SCHEMES}
    assert time.perf_counter() - t0 < 120.0
    assert out["qwc"].n_settings == 757
    for r in out.values():
        assert r.method == "exact" and r.per_target.shape == (840,) and np.all(r.per_target > 0)


def test_h6_energy_variances_against_the_earlier_claims():
    """Per-shot energy variances (Ha^2) for H6/STO-3G at 1.0 A, at FCI and after two exact SD CQE
    iterations (19.57 mHa above FCI; the earlier analysis used a state 6.6 mHa above FCI and
    claimed QWC 28.0, DF 3.2, matchgate shadows 122):

        state      QWC blocked  QWC interl.  DF chol/eigh  shifted DF   Pauli bl/int  matchgate
        FCI        16.18 (204)  28.14 (312)  5.02 / 3.06   1.23 / 1.19  1015 / 1982   121.74
        SD iter 2  15.64 (204)  27.02 (312)  5.44 / 3.35   1.16 / 1.15  1015 / 1980   121.74

    (QWC groups in parentheses; DF has 21 settings; shifted DF: Cholesky / eigh leaves.)

    QWC (interleaved, OpenFermion's order) and matchgate shadows reproduce the claims; DF does
    so with eigendecomposition leaves (first="eigh", unshifted) but not with the pivoted-Cholesky
    default (5.0-5.4); the shifted leaves are 2.6-4.7 times lower than either. The interleaved QWC
    values are those of the round-off-independent sorted insertion (tie classes); ordering tied
    weights by their round-off gave 28.32 and 27.19 (314 groups) for this jw_hamiltonian and 28.14
    and 27.02 (312) for OpenFermion's (observed)."""
    act = systems.hydrogen_chain(6, 1.0, "angstrom")
    e_fci, ci_fci = act.fci()
    from thc_engine_cqe.cqe import run_cqe
    run = run_cqe(act, max_iter=2)
    assert np.isclose((run.energies[-1] - e_fci) * 1e3, 19.5717, atol=1e-3)
    want = {"FCI": (16.1825, 204, 28.1359, 312, 5.0224, 3.0577, 1.2330, 1.1940, 1014.66, 1981.77, 121.737),
            "iter2": (15.6434, 204, 27.0221, 312, 5.4357, 3.3530, 1.1572, 1.1489, 1014.76, 1980.26, 121.740)}
    for label, ci in (("FCI", ci_fci), ("iter2", run.ci)):
        got = []
        mg = []
        for order in ORDERS:
            m = QubitMapping(6, order)
            h = jw_hamiltonian(act, m)
            st = QubitState.from_ci(6, act.nelec, ci, m)
            q = qwc_variance(h, st)
            got += [q.total, q.n_settings]
            mg.append(matchgate_shadow_variance(h, st).total)
            got.append(pauli_shadow_variance(h, st).total)
        ps = got[2], got[5]
        got = [got[0], got[1], got[3], got[4]]
        d = [df_variance(act, ci), df_variance(act, ci, first="eigh", tol=1e-12),
             df_variance(act, ci, shift=True), df_variance(act, ci, first="eigh", tol=1e-12, shift=True)]
        assert all(r.n_settings == 21 for r in d) and all(abs(r.info["bias"]) < 1e-9 for r in d)
        got += [r.total for r in d] + list(ps) + [mg[0]]
        assert np.isclose(mg[0], mg[1], rtol=1e-12)
        assert np.allclose(got, want[label], rtol=2e-4), (label, got)


def test_h8_energy_all_schemes_time():
    """Performance target: all exact energy variances for H8 (16 qubits, 1.8 bohr, FCI) within a few
    minutes. Observed (one thread, blocked order): QWC 0.9 s (626 groups), Pauli shadows 3.2 s
    (9.8e5 compatible pairs), matchgate shadows 4.3 s (2.8e6 commuting pairs), DF 0.03 s
    (31 settings). The interleaved order gives QWC 115.2 (966 groups) and Pauli 9.50e4 (observed)."""
    act = systems.hydrogen_chain(8, 1.8, "bohr")
    e_fci, ci = act.fci()
    m = QubitMapping(8)
    h = jw_hamiltonian(act, m)
    st = QubitState.from_ci(8, act.nelec, ci, m)
    assert schemes._Expectations(st).table is None          # too large: QubitState.expectations path
    t0 = time.perf_counter()
    q, p, g = qwc_variance(h, st), pauli_shadow_variance(h, st), matchgate_shadow_variance(h, st)
    d = df_variance(act, ci)
    assert time.perf_counter() - t0 < 300.0
    assert (q.n_settings, d.n_settings) == (626, 31) and p.method == g.method == "exact"
    assert np.allclose([q.total, p.total, g.total, d.total], [61.6068, 44549.8, 336.743, 9.4051], rtol=2e-4)


def test_h6_direct_route_rao_blackwellized():
    """Direct-route targets X_e = i[Q_e, H] for the 840 H6 elements (1.6e6 Pauli terms) at the SD
    iteration-2 state: matchgate RB with 2000 settings per order. Observed: 2.3-2.4 s per order
    here (relative standard error 0.5%; about 1 s is setup: the coefficient matrix and the dense
    <P> table); 2e4 settings take 13-14 s with relative standard errors of 0.15-0.16% (total) and at
    most 4.0% (per element). The exact pair sum (max_pairs=np.inf; 8.77e8 commuting pairs, 173-174 s
    per order, observed) gives a total of 9636.39 and per-element variances 4.963 to 40.69 (mean
    11.47), identical in both orders (observed); the RB total of each order agrees with it within
    4 standard errors (asserted). The relative standard error of the total is below the 2% target."""
    act = systems.hydrogen_chain(6, 1.0, "angstrom")
    from thc_engine_cqe.cqe import run_cqe
    ci = run_cqe(act, max_iter=2).ci
    el = spin_orbital_elements(6)
    res = {}
    for order in ORDERS:
        m = QubitMapping(6, order)
        xops = direct_ops(m, el, jw_hamiltonian(act, m))
        assert sum(x.n_terms for x in xops) == 1603800
        st = QubitState.from_ci(6, act.nelec, ci, m)
        t0 = time.perf_counter()
        res[order] = matchgate_shadow_variance(xops, st, n_settings=2000, seed=71)
        assert time.perf_counter() - t0 < 120.0
        assert res[order].method == "rb" and res[order].stderr < 0.02 * res[order].total
    a, b = res["blocked"], res["interleaved"]
    assert np.allclose(a.expectation, b.expectation, atol=1e-12)
    assert abs(a.total - b.total) <= 4 * np.hypot(a.stderr, b.stderr)
    for r in (a, b):
        assert abs(r.total - 9636.39) <= 4 * r.stderr         # the exact sum (observed, see above)
    assert 4 < a.per_target.min() and a.per_target.max() < 60
