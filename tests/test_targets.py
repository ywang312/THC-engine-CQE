"""Tests of the measurement targets (``thc_engine_cqe.measure.targets``).

Every target is checked against a reference that does not use the module's own helpers:

* brute-force orbit enumeration of the (anti)symmetry groups (element sets and counts);
* ``residual.compute_residual`` (spin-free A from PySCF's spin-summed transition RDMs);
* PySCF's spin-resolved transition RDMs ``direct_spin1.trans_rdm12s`` (spin-orbital A at N = 6),
  assembled here into the full (2N)^4 tensor;
* the Fock-space oracle ``testing.fock.FockSpace`` (operators and commutators at N <= 4);
* exp(-i delta H) by eigendecomposition of the CI-space Hamiltonian matrix (aux-route bias).

The CQE iterate used at N = 6 is the H6/STO-3G state at 1.0 A after 2 exact steepest-descent
iterations (``run_cqe(..., max_iter=2)``), 19.6 mHa above FCI (observed).
"""

import itertools
import math
import time

import numpy as np
import pytest

from conftest import random_active_space
from thc_engine_cqe import residual, systems
from thc_engine_cqe.cqe import run_cqe
from thc_engine_cqe.fci_ops import FCISpace, Hamiltonian
from thc_engine_cqe.measure import targets as tg
from thc_engine_cqe.measure.pauli import (PauliSum, QubitMapping, QubitState, coefficient_matrix,
                                          jw_hamiltonian, jw_two_body)
from thc_engine_cqe.testing.fock import FockSpace

ORDERS = ("blocked", "interleaved")


# ============================================================ fixtures / references
@pytest.fixture(scope="module")
def h6():
    """H6 at 1.0 A after 2 exact SD CQE iterations: (active, ci, ResidualData)."""
    act = systems.hydrogen_chain(6, 1.0, "angstrom")
    ci = run_cqe(act, max_iter=2).ci
    return act, ci, residual.compute_residual(act, ci)


@pytest.fixture(scope="module")
def h4():
    act = systems.hydrogen_chain(4, 1.8, "bohr")
    ci = run_cqe(act, max_iter=2).ci
    return act, ci, residual.compute_residual(act, ci)


_TARGET_CACHE = {}


def h6_targets(h6, order, kind):
    """(Q list, X list, build seconds of X) for the full element set, cached per order and kind."""
    key = (order, kind)
    if key not in _TARGET_CACHE:
        act = h6[0]
        m = QubitMapping(act.norb, order)
        el = tg.element_set(act.norb, kind)
        q = tg.aux_targets(el, m, kind)
        t0 = time.perf_counter()
        x = tg.direct_targets(el, act, m, kind)
        _TARGET_CACHE[key] = (q, x, time.perf_counter() - t0)
    return _TARGET_CACHE[key]


def expectations(ops, state):
    """<O_e> of a list of PauliSums (one shared Pauli basis)."""
    xs, zs, w = coefficient_matrix(ops)
    return w @ state.expectations(xs, zs)


def orbits(points, generators):
    """Orbits of a signed group action by breadth-first search.

    generators: functions x -> (y, sign). Returns a list of (members: dict x -> sign relative to
    the first member, forced_zero: bool); forced_zero means some group element maps a member to
    itself with sign -1, so every value on the orbit vanishes.
    """
    seen, out = set(), []
    for x0 in points:
        if x0 in seen:
            continue
        sign, queue, forced = {x0: 1}, [x0], False
        while queue:
            y = queue.pop()
            for g in generators:
                z, s = g(y)
                if z in sign:
                    forced |= sign[z] != sign[y] * s
                else:
                    sign[z] = sign[y] * s
                    queue.append(z)
        seen.update(sign)
        out.append((sign, forced))
    return out


SO_GENERATORS = (lambda x: ((x[1], x[0], x[2], x[3]), -1),     # a+_i a+_j = -a+_j a+_i
                 lambda x: ((x[0], x[1], x[3], x[2]), -1),     # a_l a_k = -a_k a_l
                 lambda x: ((x[2], x[3], x[0], x[1]), -1))     # 2A^{kl}_{ij} = -2A^{ij}_{kl} (real)
SF_GENERATORS = (lambda x: ((x[2], x[3], x[0], x[1]), 1),      # pair exchange
                 lambda x: ((x[1], x[0], x[3], x[2]), -1))     # A[q,p,s,r] = -A[p,q,r,s] (real)


def so_orbits(norb):
    """Free, S_z-conserving orbits of the spin-orbital residual (modes p + sigma N)."""
    nm = 2 * norb
    spin = lambda m: m // norb
    out = []
    for members, forced in orbits(itertools.product(range(nm), repeat=4), SO_GENERATORS):
        x = next(iter(members))
        if not forced and spin(x[0]) + spin(x[1]) == spin(x[2]) + spin(x[3]):
            out.append(members)
    return out


def sf_orbits(norb):
    return [members for members, forced in orbits(itertools.product(range(norb), repeat=4), SF_GENERATORS)
            if not forced]


def modes_of(el, norb):
    return [tuple(int(p + s * norb) for p, s in row) for row in el.tolist()]


def full_spin_orbital_residual(act, ci):
    """2A[m_i, m_j, m_k, m_l] for all spin orbitals from PySCF's spin-resolved transition 2-RDMs.

    T[i,j,k,l] = <Psi| a+_i a+_j a_l a_k H |Psi> is assembled from the blocks
    d2[s,t][p,q,r,s'] = <Psi| a+_{p s} a+_{r t} a_{s' t} a_{q s} |H Psi> (direct spin assignment)
    and, for s != t, the crossed assignment a_l a_k = -a_k a_l; then A = T - T^{(kl)(ij)} (real).
    """
    from pyscf.fci import direct_spin1

    n = act.norb
    ci = ci / np.linalg.norm(ci)
    hci = Hamiltonian(act).apply(ci)
    _, (aa, ab, ba, bb) = direct_spin1.trans_rdm12s(ci, hci, n, act.nelec)
    t = np.zeros((2 * n,) * 4)
    sl = lambda s: slice(s * n, (s + 1) * n)
    for (s, u), blk in {(0, 0): aa, (0, 1): ab, (1, 0): ba, (1, 1): bb}.items():
        t[sl(s), sl(u), sl(s), sl(u)] = blk.transpose(0, 2, 1, 3)
        if s != u:
            t[sl(s), sl(u), sl(u), sl(s)] = -blk.transpose(0, 2, 3, 1)
    return t - t.transpose(2, 3, 0, 1)


def ci_propagator(act):
    """delta -> exp(-i delta H) on CI vectors, by eigendecomposition of the CI-space matrix of H."""
    ham = Hamiltonian(act)
    space = ham.space
    eye = np.eye(space.size)
    hm = np.array([ham.apply(eye[i].reshape(space.shape)).ravel() for i in range(space.size)]).T
    w, v = np.linalg.eigh(0.5 * (hm + hm.T))

    def prop(ci, delta):
        return (v @ (np.exp(-1j * delta * w) * (v.T @ np.asarray(ci).ravel()))).reshape(space.shape)

    return prop


# ================================================================ element sets
@pytest.mark.parametrize("norb", [1, 2, 3, 4, 6])
def test_spin_orbital_elements_match_brute_force_orbits(norb):
    """The rows are canonical (i < j, k < l, (ij) < (kl), S_z conserved, lexicographic order),
    lie in distinct free orbits and cover every free S_z-conserving orbit of the group generated by
    the two antisymmetries and 2A^{kl}_{ij} = -2A^{ij}_{kl}; K = 2 C(C(N,2),2) + C(N^2,2), 840 at N = 6."""
    el = tg.spin_orbital_elements(norb)
    assert el.dtype == np.int64 and el.shape[1:] == (4, 2)
    modes = modes_of(el, norb)
    assert modes == sorted(modes) and len(set(modes)) == len(modes)
    for i, j, k, l in modes:
        assert i < j and k < l and (i, j) < (k, l)
        assert i // norb + j // norb == k // norb + l // norb
    ref = so_orbits(norb)
    assert len(el) == len(ref) == 2 * math.comb(math.comb(norb, 2), 2) + math.comb(norb * norb, 2)
    owner = {x: o for o, members in enumerate(ref) for x in members}
    assert sorted(owner[x] for x in modes) == list(range(len(ref)))
    assert all(len(members) == 8 for members in ref)
    if norb == 6:
        assert len(el) == 840
    if len(el):
        assert np.all(tg.element_multiplicities(el, "spin_orbital", norb) == 8)


@pytest.mark.parametrize("norb", [1, 2, 3, 4, 5, 6])
def test_spin_free_elements_match_brute_force_orbits(norb):
    """One representative (the smallest member) per free orbit of {pair exchange, A[q,p,s,r] = -A[p,q,r,s]};
    count N^2(N^2-1)/4 (60 at N = 4, 315 at N = 6); multiplicities are the orbit sizes."""
    el = tg.spin_free_elements(norb)
    assert el.dtype == np.int64 and el.shape[1:] == (4,)
    reps = [tuple(r) for r in el.tolist()]
    assert reps == sorted(reps)
    ref = sf_orbits(norb)
    assert len(reps) == len(ref) == norb ** 2 * (norb ** 2 - 1) // 4
    by_rep = {min(members): members for members in ref}
    assert set(reps) == set(by_rep)
    mult = tg.element_multiplicities(el, "spin_free", norb) if reps else np.zeros(0)
    assert [len(by_rep[r]) for r in reps] == mult.tolist()
    assert set(mult.tolist()) <= {2, 4}
    assert int(np.sum(mult == 2)) == norb * (norb - 1) // 2
    assert len(reps) == {4: 60, 6: 315}.get(norb, len(reps))


def test_spin_free_tensor_round_trip_and_symmetries(h6, rng):
    """spin_free_tensor(A[reps]) == A for the exact residual of a real state (1e-14), and a fill of random
    values has exactly the pair symmetry, A[q,p,s,r] = -A[p,q,r,s], zeros on the forced orbits, and
    ||A||^2 = sum_e m_e v_e^2."""
    act, ci, rd = h6
    n = act.norb
    el = tg.spin_free_elements(n)
    assert not np.iscomplexobj(rd.A)
    a = rd.A
    filled = tg.spin_free_tensor(a[tuple(el.T)], el, n)
    assert np.abs(filled - a).max() <= 1e-14
    v = rng.standard_normal(len(el))
    t = tg.spin_free_tensor(v, el, n)
    assert np.array_equal(t, t.transpose(2, 3, 0, 1))
    assert np.array_equal(t, -t.transpose(1, 0, 3, 2))
    assert np.array_equal(t[tuple(el.T)], v)
    for p, r in itertools.product(range(n), repeat=2):
        assert t[p, p, r, r] == 0 and t[p, r, r, p] == 0
    mult = tg.element_multiplicities(el, "spin_free", n)
    assert np.isclose(np.sum(t ** 2), np.sum(mult * v ** 2), rtol=1e-14)
    # any member of an orbit works as its representative
    perm = el[:, [2, 3, 0, 1]]
    assert np.array_equal(tg.spin_free_tensor(v, perm, n), t)
    assert np.array_equal(tg.spin_free_tensor(-v, el[:, [1, 0, 3, 2]], n), t)


@pytest.mark.parametrize("norb", [2, 3])
def test_spin_orbital_tensor_and_contract_spin_vs_brute_force(norb, rng):
    """spin_orbital_tensor equals the fill by the brute-force orbits (signs from the BFS), and
    contract_spin equals the explicit loop A[p,q,r,s] = sum_{sigma tau} 2A^{(p sigma)(r tau)}_{(q sigma)(s tau)}
    over the full tensor; contract_spin needs every independent element."""
    el = tg.spin_orbital_elements(norb)
    v = rng.standard_normal(len(el))
    nm = 2 * norb
    ref = np.zeros((nm,) * 4)
    by_member = {}
    for members in so_orbits(norb):
        for x in members:
            by_member[x] = members
    for e, x in enumerate(modes_of(el, norb)):
        members = by_member[x]
        for y, s in members.items():
            ref[y] = s * members[x] * v[e]
    d = tg.spin_orbital_tensor(v, el, norb)
    assert np.array_equal(d, ref)
    assert np.isclose(np.sum(d ** 2), 8 * np.sum(v ** 2))
    loop = np.zeros((norb,) * 4)
    for p, q, r, s in itertools.product(range(norb), repeat=4):
        loop[p, q, r, s] = sum(ref[p + a * norb, r + b * norb, q + a * norb, s + b * norb]
                               for a in (0, 1) for b in (0, 1))
    assert np.allclose(tg.contract_spin(v, el, norb), loop, atol=1e-14, rtol=0)
    with pytest.raises(ValueError, match="misses"):
        tg.contract_spin(v[1:], el[1:], norb)
    # an extra S_z-violating element, a+_{0a} a+_{1a} a_{1b} a_{0a}, is allowed and does not contribute
    extra = np.array([[[0, 0], [1, 0], [0, 0], [1, 1]]])
    both = np.concatenate([el, extra])
    assert np.allclose(tg.contract_spin(np.append(v, 7.0), both, norb), loop, atol=1e-14, rtol=0)


# ========================================================= direct route at N = 6
@pytest.mark.parametrize("order", ORDERS)
def test_direct_spin_free_targets_reproduce_compute_residual(h6, order):
    """<X_e> = A[p,q,r,s] of residual.compute_residual for all 315 spin-free elements of the H6
    iterate (1e-10); the X_e have exactly real Pauli coefficients (Hermitian)."""
    act, ci, rd = h6
    q, x, _ = h6_targets(h6, order, "spin_free")
    el = tg.spin_free_elements(act.norb)
    assert len(x) == len(el) == 315
    assert all(np.all(o.coeffs.imag == 0) for o in x + q)
    st = QubitState.from_ci(act.norb, act.nelec, ci, QubitMapping(act.norb, order))
    vals = expectations(x, st)
    assert np.abs(vals.imag).max() <= 1e-14
    assert np.abs(vals.real - np.real(rd.A)[tuple(el.T)]).max() <= 1e-10
    assert np.abs(tg.spin_free_tensor(vals.real, el, act.norb) - np.real(rd.A)).max() <= 1e-10


@pytest.mark.parametrize("order", ORDERS)
def test_direct_spin_orbital_targets_vs_pyscf_spin_resolved_rdms(h6, order):
    """<X_e> for the 840 spin-orbital elements of the H6 iterate equals the spin-orbital residual from
    PySCF's spin-resolved transition RDMs (1e-10), and its spin contraction equals compute_residual's A."""
    act, ci, rd = h6
    _, x, _ = h6_targets(h6, order, "spin_orbital")
    el = tg.spin_orbital_elements(act.norb)
    assert len(x) == 840
    st = QubitState.from_ci(act.norb, act.nelec, ci, QubitMapping(act.norb, order))
    vals = expectations(x, st)
    full = full_spin_orbital_residual(act, ci)
    ref = np.array([full[m] for m in modes_of(el, act.norb)])
    assert np.abs(ref).max() > 1e-2
    assert np.abs(vals.real - ref).max() <= 1e-10
    assert np.abs(tg.contract_spin(vals.real, el, act.norb) - np.real(rd.A)).max() <= 1e-10
    # the full PySCF tensor is the fill of the independent values
    assert np.abs(tg.spin_orbital_tensor(ref, el, act.norb) - full).max() <= 1e-12


def test_target_structure_sizes_and_build_time(h6):
    """Structure of the H6 targets (blocked order): Q_e have at most 8 (spin-orbital) / 32 (spin-free)
    Pauli terms; the X_e sets have about 1.6e6 and 1.5e6 terms; each set builds in seconds.

    Observed (H6 at 1.0 A, 919-term H, one thread): 840 spin-orbital X_e with 1,603,800 terms
    (mean 1909, max 2448) in 0.4-0.6 s; 315 spin-free X_e with 1,528,224 terms (mean 4852, max 7288)
    in 0.6-0.7 s; both sets span the same 36,474 Pauli strings. Every spin-free X_e is a signed sum
    of spin-orbital ones, so its strings are among theirs (tested).
    """
    sizes = {}
    for kind, q_max, lo, hi in (("spin_orbital", 8, 1.5e6, 1.7e6), ("spin_free", 32, 1.4e6, 1.65e6)):
        q, x, seconds = h6_targets(h6, "blocked", kind)
        assert max(o.n_terms for o in q) == q_max
        total = sum(o.n_terms for o in x)
        assert lo < total < hi
        assert seconds < 30
        assert all(o.identity_coefficient() == 0 for o in x)
        xs, zs, _ = coefficient_matrix(x)
        sizes[kind] = (xs << np.uint64(32)) | zs
    assert np.all(np.isin(sizes["spin_free"], sizes["spin_orbital"]))


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("kind", tg.KINDS)
def test_targets_match_openfermion_at_n6(h6, order, kind):
    """For 4 random H6 elements, Q_e and X_e equal OpenFermion's jordan_wigner of
    (Gamma - Gamma^dagger)/(2i) and i [Q_e, H] (OpenFermion's commutator) term by term (1e-12), with the
    FermionOperator modes labelled by our qubit indices; H is jw_hamiltonian, which test_pauli checks
    against OpenFermion's own qubit Hamiltonian."""
    of = pytest.importorskip("openfermion")
    act = h6[0]
    m = QubitMapping(act.norb, order)
    h = jw_hamiltonian(act, m)
    h_of = h.to_openfermion()
    el = tg.element_set(act.norb, kind)
    pick = el[np.random.default_rng(7).choice(len(el), 4, replace=False)]
    for e, q, x in zip(pick.tolist(), tg.aux_targets(pick, m, kind), tg.direct_targets(pick, act, m, kind)):
        if kind == "spin_orbital":
            (pi, si), (pj, sj), (pk, sk), (pl, sl) = e
            g = of.FermionOperator(((m.qubit(pi, si), 1), (m.qubit(pj, sj), 1), (m.qubit(pl, sl), 0),
                                    (m.qubit(pk, sk), 0)))
        else:
            p, qq, r, s = e
            g = of.FermionOperator()
            for a, b in itertools.product((0, 1), repeat=2):
                g += of.FermionOperator(((m.qubit(p, a), 1), (m.qubit(r, b), 1), (m.qubit(s, b), 0),
                                         (m.qubit(qq, a), 0)))
        q_of = of.jordan_wigner((g - of.hermitian_conjugated(g)) * (-0.5j))
        q_of.compress(1e-14)
        x_of = 1j * of.commutator(q_of, h_of)
        x_of.compress(1e-13)
        assert q.allclose(PauliSum.from_openfermion(q_of, m.n_qubits), atol=1e-14)
        x_ref = PauliSum.from_openfermion(x_of, m.n_qubits, tol=1e-13)
        assert x.simplify(1e-13).n_terms == x_ref.n_terms and x.allclose(x_ref, atol=1e-12)


# ======================================================== Fock-space oracle, N <= 4
def fock_gamma(fs, kind, e):
    if kind == "spin_orbital":
        (pi, si), (pj, sj), (pk, sk), (pl, sl) = e
        return (fs.adag(pi, si) @ fs.adag(pj, sj) @ fs.a(pl, sl) @ fs.a(pk, sk)).toarray()
    return fs.e2(*e).toarray()


@pytest.mark.parametrize("kind", tg.KINDS)
def test_operators_equal_fock_oracle_matrices(kind):
    """Blocked-order Gamma_e, Q_e = (Gamma - Gamma^dagger)/(2i) and X_e = [Gamma - Gamma^dagger, H]/2 equal
    the Fock-oracle matrices for every element (N = 3, random real Hamiltonian; 1e-12, X_e 1e-11)."""
    act = random_active_space(3, (2, 1), seed=5)
    n = act.norb
    fs = FockSpace(n)
    hf = fs.hamiltonian(act.h1, act.eri, act.ecore).toarray()
    m = QubitMapping(n)
    el = tg.element_set(n, kind)
    gam = tg.gamma_operators(el, m, kind)
    q = tg.aux_targets(el, m, kind)
    x = tg.direct_targets(el, act, m, kind)
    for e, g_op, q_op, x_op in zip(el.tolist(), gam, q, x):
        g = fock_gamma(fs, kind, e)
        k = g - g.conj().T
        assert np.abs(g_op.to_dense() - g).max() <= 1e-12
        assert np.abs(q_op.to_dense() - k / 2j).max() <= 1e-12
        assert np.abs(x_op.to_dense() - (k @ hf - hf @ k) / 2).max() <= 1e-11


@pytest.mark.parametrize("order", ORDERS)
def test_direct_targets_vs_fock_commutators_h4(h4, order, rng):
    """<X_e> equals <Psi|[Gamma_e, H]|Psi> from Fock-oracle matrices for all 150 spin-orbital and 60
    spin-free elements at N = 4: the H4 CQE iterate and a random real CI vector (1e-10)."""
    act, ci, _ = h4
    n = act.norb
    fs = FockSpace(n)
    hf = fs.hamiltonian(act.h1, act.eri, act.ecore)
    m = QubitMapping(n, order)
    for psi in (ci, FCISpace(n, act.nelec).random(rng)):
        v = fs.from_ci(psi, act.nelec)
        st = QubitState.from_ci(n, act.nelec, psi, m)
        for kind in tg.KINDS:
            el = tg.element_set(n, kind)
            ref = []
            for e in el.tolist():
                g = fock_gamma(fs, kind, e)
                ref.append(v @ (g @ (hf @ v)) - (hf @ v) @ (g @ v))
            vals = expectations(tg.direct_targets(el, act, m, kind), st)
            assert len(el) == {"spin_orbital": 150, "spin_free": 60}[kind]
            assert np.abs(vals - np.array(ref)).max() <= 1e-10


# ================================================================ aux route
def test_aux_targets_bias_is_second_order(h6):
    """-<Q_e>_Lambda/delta -> A_e with log-log slope 2 +- 0.1 over delta in [1e-3, 1e-1] on the real H6
    iterate, both element sets; Lambda = exp(-i delta H) Psi by eigendecomposition (independent of
    routes.aux_states).

    Observed relative bias ||A_hat - A||/||A||: 3.7e-7 (spin-orbital) and 4.3e-7 (spin-free) at
    delta = 1e-3, 3.7e-3 and 4.3e-3 at delta = 0.1.
    """
    act, ci, rd = h6
    n = act.norb
    prop = ci_propagator(act)
    deltas = np.array([1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
    m = QubitMapping(n)
    states = [QubitState.from_ci(n, act.nelec, prop(ci, d), m) for d in deltas]
    for kind in tg.KINDS:
        el = tg.element_set(n, kind)
        if kind == "spin_free":
            exact = np.real(rd.A)[tuple(el.T)]
        else:
            full = full_spin_orbital_residual(act, ci)
            exact = np.array([full[x] for x in modes_of(el, n)])
        q_ops = tg.aux_targets(el, m, kind)
        bias = []
        for d, st in zip(deltas, states):
            est = tg.aux_estimates(expectations(q_ops, st).real, d)
            bias.append(np.linalg.norm(est - exact))
        slope = np.polyfit(np.log(deltas), np.log(bias), 1)[0]
        assert abs(slope - 2) <= 0.1, (kind, slope, bias)
        rel = np.array(bias) / np.linalg.norm(exact)
        assert 1e-7 < rel[0] < 1e-6 and 1e-3 < rel[-1] < 1e-2


@pytest.mark.parametrize("order", ORDERS)
def test_energy_target_gives_the_energy(h6, order):
    """<H> on the qubit state equals Hamiltonian.energy for the H6 iterate and the FCI energy for the
    FCI vector (1e-11), in both orders."""
    act, ci, rd = h6
    m = QubitMapping(act.norb, order)
    h = tg.energy_target(act, m)
    assert h.allclose(jw_hamiltonian(act, m), atol=0)
    st = QubitState.from_ci(act.norb, act.nelec, ci, m)
    assert abs(st.expectation(h, normalize=True) - Hamiltonian(act).energy(ci)) <= 1e-11
    assert abs(st.expectation(h, normalize=True) - rd.energy) <= 1e-11
    e_fci, ci_fci = act.fci()
    st = QubitState.from_ci(act.norb, act.nelec, ci_fci, m)
    assert abs(st.expectation(h, normalize=True) - e_fci) <= 1e-11


def test_blocked_and_interleaved_give_the_same_expectations(h6):
    """<Q_e> on a complex auxiliary state (delta = 0.3) and <X_e> on Psi agree between the two qubit
    orders for both element sets (1e-13)."""
    act, ci, _ = h6
    n = act.norb
    lam = ci_propagator(act)(ci, 0.3)
    vals = {}
    for order in ORDERS:
        m = QubitMapping(n, order)
        for kind in tg.KINDS:
            q, x, _ = h6_targets(h6, order, kind)
            vals[order, kind, "Q"] = expectations(q, QubitState.from_ci(n, act.nelec, lam, m))
            vals[order, kind, "X"] = expectations(x, QubitState.from_ci(n, act.nelec, ci, m))
    for kind in tg.KINDS:
        for name in ("Q", "X"):
            a, b = vals["blocked", kind, name], vals["interleaved", kind, name]
            assert np.abs(a).max() > 1e-3
            assert np.abs(a - b).max() <= 1e-13


# ============================================================ small checks
def test_spin_free_gamma_equals_jw_two_body():
    """Gamma_e = e_{pq,rs} built from four spin-orbital products equals jw_two_body of a unit tensor."""
    n = 3
    for order in ORDERS:
        m = QubitMapping(n, order)
        el = np.array(list(itertools.product(range(n), repeat=4)))
        for e, g in zip(el, tg.gamma_operators(el, m, "spin_free")):
            c = np.zeros((n,) * 4)
            c[tuple(e)] = 1.0
            assert g.allclose(jw_two_body(m, c), atol=1e-15)


def test_aux_estimates_and_input_validation():
    """Post-processing and the errors raised for malformed input."""
    q = np.array([[0.1, -0.2], [0.3, 0.05]])
    assert np.allclose(tg.aux_estimates(q, 0.5), -(q.sum(axis=0)) / 0.5)
    assert np.allclose(tg.aux_estimates(q[0], -0.25), q[0] / 0.25)
    with pytest.raises(ValueError):
        tg.aux_estimates(q, 0.0)
    with pytest.raises(ValueError):
        tg.element_set(3, "spin")
    m = QubitMapping(2)
    with pytest.raises(ValueError):
        tg.aux_targets(np.zeros((3, 4), dtype=int), m, "spin_orbital")
    with pytest.raises(TypeError):
        tg.aux_targets(np.zeros((3, 4)), m, "spin_free")
    with pytest.raises(ValueError):
        tg.aux_targets(np.array([[0, 0, 0, 2]]), m, "spin_free")
    el = tg.spin_free_elements(2)
    with pytest.raises(ValueError, match="real"):
        tg.spin_free_tensor(np.ones(len(el)) * 1j, el, 2)
    with pytest.raises(ValueError, match="same symmetry orbit"):
        tg.spin_free_tensor(np.ones(2), np.array([[0, 1, 0, 0], [0, 0, 0, 1]]), 2)
    with pytest.raises(ValueError, match="forced to zero"):
        tg.spin_free_tensor(np.ones(1), np.array([[0, 0, 1, 1]]), 2)
    so = tg.spin_orbital_elements(2)
    with pytest.raises(ValueError, match="forced to zero"):
        tg.spin_orbital_tensor(np.ones(1), np.array([[[0, 0], [1, 0], [0, 0], [1, 0]]]), 2)
    with pytest.raises(ValueError, match="same symmetry orbit"):
        tg.spin_orbital_tensor(np.ones(2), np.concatenate([so[:1], so[:1, [1, 0, 2, 3]]]), 2)
    with pytest.raises(ValueError):
        tg.spin_orbital_tensor(np.ones(len(so) + 1), so, 2)
    assert tg.aux_targets(np.zeros((0, 4, 2), dtype=int), m) == []
    assert isinstance(tg.energy_target(random_active_space(2, (1, 1))), PauliSum)
