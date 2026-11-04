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


# ================================================================= QWC
@pytest.mark.parametrize("order", ORDERS)
def test_qwc_groups_partition_the_terms_and_commute_qubitwise(order):
    """Every non-identity string is in exactly one group, members commute qubit-wise, the union
    string of a group is its measured basis, and the assignment is sorted-insertion first-fit
    (re-derived here by a naive member-by-member implementation)."""
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
        # sorted insertion: decreasing sum_e |W_ej|; naive first-fit gives the same groups
        ops = [target] if isinstance(target, PauliSum) else target
        wsum = np.zeros(g.xs.size)
        for op in ops:
            idx = np.searchsorted((g.xs << np.uint64(32)) | g.zs, (op.xs << np.uint64(32)) | op.zs)
            wsum[idx] += np.abs(op.coeffs)
        assert np.all(np.diff(wsum[g.order]) <= 1e-15)
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
    # Rao-Blackwellization over all 3^n bases is exact
    bx, bz = pauli_bases(n)
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
    # Rao-Blackwellized estimate over all 81 bases equals the exact value
    bx, bz = pauli_bases(4)
    rb = pauli_shadow_variance(ops, st2, method="rb", settings=zip(bx, bz))
    assert np.allclose(rb.per_target, ex2.per_target, rtol=1e-10, atol=1e-10)


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
    # Rao-Blackwellization over all matchings is exact
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
    assert matchgate_shadow_variance(xops, st, max_pairs=10, n_settings=10, seed=1).method == "rb"


def test_rao_blackwellized_vs_snapshot_monte_carlo():
    """RB and snapshot MC estimate the same variance (H4 energy, matchgate): within 4 combined sigma."""
    act, _, _ = get_system("h4")
    st, m = qstate("h4", "interleaved")
    h = jw_hamiltonian(act, m)
    rb = matchgate_shadow_variance(h, st, method="rb", n_settings=6000, seed=51)
    mc = matchgate_shadow_variance(h, st, method="mc", n_snapshots=20000, seed=52, batch=512)
    assert abs(rb.total - mc.total) <= 4 * np.hypot(rb.stderr, mc.stderr)


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
    """Optimal allocation across experiments: (sum sqrt V_l)^2; per-target sum_l v_le / f_l."""
    st, m = qstate("h2", "blocked")
    ops = aux_ops(m, spin_orbital_elements(2))
    parts = [matchgate_shadow_variance(ops, st), pauli_shadow_variance([o * 2 for o in ops], st)]
    c = combine_independent(parts)
    v = np.array([p.total for p in parts])
    assert np.isclose(c.total, np.sum(np.sqrt(v)) ** 2) and np.isclose(c.total, c.per_target.sum())
    assert np.isclose(c.uniform_total, 2 * v.sum()) and c.total <= c.uniform_total
    assert np.allclose(c.expectation, parts[0].expectation + parts[1].expectation)
    assert np.allclose(c.per_target, sum(p.per_target / f for p, f in zip(parts, c.allocation)))
    assert c.n_settings is None
    # nested optimal allocations (QWC groups inside parts) equal the global optimum over all groups
    q = [qwc_variance(ops, st), qwc_variance([o * 3 for o in ops[::-1]], st)]
    cq = combine_independent(q)
    allv = np.concatenate([r.setting_variances for r in q])
    assert cq.n_settings == q[0].n_settings + q[1].n_settings
    assert np.isclose(cq.total, np.sum(np.sqrt(allv)) ** 2, rtol=1e-12)


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
        FCI        16.18 (204)  28.32 (314)  5.02 / 3.06   1.23 / 1.19  1015 / 1982   121.74
        SD iter 2  15.64 (204)  27.19 (314)  5.44 / 3.35   1.16 / 1.15  1015 / 1980   121.74

    (QWC groups in parentheses; DF has 21 settings; shifted DF: Cholesky / eigh leaves.)

    QWC (interleaved, OpenFermion's order) and matchgate shadows reproduce the claims; DF does
    so with eigendecomposition leaves (first="eigh", unshifted) but not with the pivoted-Cholesky
    default (5.0-5.4); the shifted leaves are 2.6-4.7 times lower than either."""
    act = systems.hydrogen_chain(6, 1.0, "angstrom")
    e_fci, ci_fci = act.fci()
    from thc_engine_cqe.cqe import run_cqe
    run = run_cqe(act, max_iter=2)
    assert np.isclose((run.energies[-1] - e_fci) * 1e3, 19.5717, atol=1e-3)
    want = {"FCI": (16.1825, 204, 28.3169, 314, 5.0224, 3.0577, 1.2330, 1.1940, 1014.66, 1981.77, 121.737),
            "iter2": (15.6434, 204, 27.1876, 314, 5.4357, 3.3530, 1.1572, 1.1489, 1014.76, 1980.26, 121.740)}
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
    (31 settings). The interleaved order gives QWC 117.6 (967 groups) and Pauli 9.50e4 (observed)."""
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
    assert np.allclose([q.total, p.total, g.total, d.total], [61.607, 44549.8, 336.743, 9.4051], rtol=2e-4)


def test_h6_direct_route_rao_blackwellized():
    """Direct-route targets X_e = i[Q_e, H] for the 840 H6 elements (1.6e6 Pauli terms) at the SD
    iteration-2 state: matchgate RB with 2000 settings per order. Observed: 2.3-2.4 s per order
    here (relative standard error 0.5%; about 1 s is setup: the coefficient matrix and the dense
    <P> table); 2e4 settings take 13.4-13.6 s and give totals 9615 +- 15 (blocked) and 9618 +- 15
    (interleaved), per-element variances 4.9 to 44 (mean 11.4), relative standard errors 0.16%
    (total) and at most 3.9% (per element). The two orders agree (the ensemble is permutation
    invariant) and the relative standard error of the total is below the 2% target."""
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
    assert 9000 < a.total < 10300 and 4 < a.per_target.min() and a.per_target.max() < 60
