"""Every operator/RDM convention is checked against the Jordan-Wigner oracle."""

import itertools

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse.linalg

from thc_engine_cqe import fci_ops, residual
from thc_engine_cqe.fci_ops import FCISpace, Hamiltonian
from thc_engine_cqe.testing.fock import FockSpace


def _oracle(system):
    fs = FockSpace(system.norb)
    space = FCISpace.of(system)
    return fs, space


def test_one_body_convention(small_random_system, rng):
    fs, space = _oracle(small_random_system)
    n = space.norb
    f = rng.standard_normal((n, n))
    ci = space.random(rng)
    out = fci_ops.one_body(space, f, ci)
    ref = fs.to_ci(fs.one_body(f) @ fs.from_ci(ci, space.nelec), space.nelec)
    assert np.allclose(out, ref, atol=1e-12)


def test_two_body_ee_convention_non_symmetric(small_random_system, rng):
    """Guards against PySCF's reversed operator order in direct_nosym.contract_2e."""
    fs, space = _oracle(small_random_system)
    n = space.norb
    m = rng.standard_normal((n,) * 4)  # no pair-exchange symmetry
    ci = space.random(rng)
    out = fci_ops.two_body_ee(space, m, ci)
    op = sum(m[p, q, r, s] * (fs.E(p, q) @ fs.E(r, s))
             for p, q, r, s in itertools.product(range(n), repeat=4))
    ref = fs.to_ci(op @ fs.from_ci(ci, space.nelec), space.nelec)
    assert np.allclose(out, ref, atol=1e-11)


def test_two_body_normal_ordered_convention(small_random_system, rng):
    fs, space = _oracle(small_random_system)
    n = space.norb
    c = rng.standard_normal((n,) * 4)
    ci = space.random(rng)
    out = fci_ops.two_body(space, c, ci)
    ref = fs.to_ci(fs.two_body(c) @ fs.from_ci(ci, space.nelec), space.nelec)
    assert np.allclose(out, ref, atol=1e-11)
    # e_{pq,rs} built from explicit a+a+aa products matches E E - delta E
    p, q, r, s = 0, 1, 1, 2 % n
    lhs = fs.e2(p, q, r, s)
    rhs = fs.E(p, q) @ fs.E(r, s) - (fs.E(p, s) if q == r else 0)
    assert abs(lhs - rhs).max() < 1e-12


def test_complex_operators(small_random_system, rng):
    fs, space = _oracle(small_random_system)
    n = space.norb
    c = rng.standard_normal((n,) * 4) + 1j * rng.standard_normal((n,) * 4)
    f = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    ci = space.random(rng, complex_=True)
    v = fs.from_ci(ci, space.nelec)
    assert np.allclose(fci_ops.two_body(space, c, ci), fs.to_ci(fs.two_body(c) @ v, space.nelec), atol=1e-11)
    assert np.allclose(fci_ops.one_body(space, f, ci), fs.to_ci(fs.one_body(f) @ v, space.nelec), atol=1e-11)


def test_hamiltonian_apply_and_energy(small_random_system, rng):
    fs, space = _oracle(small_random_system)
    sysm = small_random_system
    ham = Hamiltonian(sysm)
    H = fs.hamiltonian(sysm.h1, sysm.eri, sysm.ecore)
    for complex_ in (False, True):
        ci = space.random(rng, complex_=complex_)
        v = fs.from_ci(ci, space.nelec)
        ref = fs.to_ci(H @ v, space.nelec) - sysm.ecore * ci
        assert np.allclose(ham.apply(ci), ref, atol=1e-11)
        assert np.isclose(ham.energy(ci), np.vdot(v, H @ v).real, atol=1e-11)


def test_fci_energy_matches_oracle_ground_state(small_random_system):
    sysm = small_random_system
    fs, space = _oracle(sysm)
    e_fci, ci = sysm.fci(spin_square=None)
    H = fs.hamiltonian(sysm.h1, sysm.eri, sysm.ecore)
    # restrict the oracle to the (n_alpha, n_beta) sector
    addr = fs._addresses(space.nelec)
    Hs = H[addr][:, addr].toarray()
    assert np.isclose(e_fci, np.linalg.eigvalsh(Hs)[0], atol=1e-9)
    assert np.isclose(Hamiltonian(sysm).energy(ci), e_fci, atol=1e-9)


@pytest.mark.parametrize("complex_", [False, True])
def test_trans_rdm12_and_cse_matrix(small_random_system, rng, complex_):
    sysm = small_random_system
    fs, space = _oracle(sysm)
    n = space.norb
    bra, ket = space.random(rng, complex_), space.random(rng, complex_)
    vb, vk = fs.from_ci(bra, space.nelec), fs.from_ci(ket, space.nelec)
    dm1, dm2 = residual.trans_rdm12(space, bra, ket)
    for p, q in itertools.product(range(n), repeat=2):
        assert np.isclose(dm1[p, q], np.vdot(vb, fs.E(q, p) @ vk), atol=1e-12)
    for p, q, r, s in itertools.product(range(n), repeat=4):
        assert np.isclose(dm2[p, q, r, s], np.vdot(vb, fs.e2(p, q, r, s) @ vk), atol=1e-12)
    # CSE matrix and ACSE residual against explicit commutators
    ham = Hamiltonian(sysm)
    H = fs.hamiltonian(sysm.h1, sysm.eri, 0.0)
    T = residual.cse_matrix(space, ket, ham.apply(ket))
    A = residual.acse_residual(T)
    for p, q, r, s in itertools.product(range(n), repeat=4):
        e = fs.e2(p, q, r, s)
        assert np.isclose(T[p, q, r, s], np.vdot(vk, e @ (H @ vk)), atol=1e-11)
        assert np.isclose(A[p, q, r, s], np.vdot(vk, (e @ H - H @ e) @ vk), atol=1e-11)


@pytest.mark.parametrize("complex_", [False, True])
def test_descent_generator_slope(small_random_system, rng, complex_):
    """d/deps E(exp(eps G) Psi) at 0 equals -||A||^2 for g = conj(A)."""
    sysm = small_random_system
    fs, space = _oracle(sysm)
    psi = space.random(rng, complex_)
    rd = residual.compute_residual(sysm, psi)
    g = residual.descent_generator(rd.A)
    G = fs.two_body(g).toarray()
    assert np.allclose(G, -G.conj().T, atol=1e-11), "generator must be anti-Hermitian"
    H = fs.hamiltonian(sysm.h1, sysm.eri, sysm.ecore).toarray()
    v = fs.from_ci(psi, space.nelec)
    h = 1e-5
    e = [np.vdot(w, H @ w).real for w in (scipy.linalg.expm(t * G) @ v for t in (-h, h))]
    slope = (e[1] - e[0]) / (2 * h)
    assert np.isclose(slope, -np.sum(np.abs(rd.A) ** 2), rtol=1e-6)


def test_energy_and_variance_identities(small_random_system, rng):
    sysm = small_random_system
    space = FCISpace.of(sysm)
    psi = space.random(rng)
    rd = residual.compute_residual(sysm, psi)
    e_rdm = residual.energy_from_rdms(sysm.h1, sysm.eri, rd.dm1, rd.dm2, sysm.ecore)
    assert np.isclose(e_rdm, rd.energy, atol=1e-11)
    K = residual.reduced_hamiltonian(sysm.h1, sysm.eri, space.n_electrons)
    e_el = rd.energy - sysm.ecore
    assert np.isclose(np.sum(K * rd.dm2), e_el, atol=1e-11)
    assert np.isclose(residual.variance_from_cse(K, rd.T, rd.dm2, e_el), rd.variance, atol=1e-10)


def test_residual_vanishes_at_eigenstate(small_random_system):
    sysm = small_random_system
    e, ci = sysm.fci(spin_square=None)
    rd = residual.compute_residual(sysm, ci)
    assert rd.residual_norm < 1e-7
    assert rd.variance < 1e-10
    assert np.isclose(rd.energy, e, atol=1e-10)


def test_pair_order_symmetries(small_random_system, rng):
    """e_{pq,rs} = e_{rs,pq} so dm2, T-derived A are pair-exchange symmetric."""
    sysm = small_random_system
    space = FCISpace.of(sysm)
    rd = residual.compute_residual(sysm, space.random(rng))
    assert np.allclose(rd.dm2, rd.dm2.transpose(2, 3, 0, 1), atol=1e-12)
    assert np.allclose(rd.A, rd.A.transpose(2, 3, 0, 1), atol=1e-11)
    # real state: A is antisymmetric under the adjoint index map (p q r s) -> (q p s r)
    assert np.allclose(rd.A, -rd.A.transpose(1, 0, 3, 2), atol=1e-11)


def test_mazziotti_order_roundtrip(rng):
    x = rng.standard_normal((3, 3, 3, 3))
    d = residual.pair_to_mazziotti(x)
    assert np.allclose(residual.mazziotti_to_pair(d), x)
    # D[i,j,k,l] = X[i,k,j,l]
    assert np.isclose(d[0, 1, 2, 0], x[0, 2, 1, 0])


def test_expm_apply_and_krylov(small_random_system, rng):
    sysm = small_random_system
    fs, space = _oracle(sysm)
    psi = space.random(rng)
    g = residual.descent_generator(residual.compute_residual(sysm, psi).A)
    G = fs.two_body(g).toarray()
    v = fs.from_ci(psi, space.nelec)
    t = 0.7 / np.linalg.norm(G, 2)

    def mv(x):
        return fci_ops.two_body(space, g, x)

    ref = fs.to_ci(scipy.linalg.expm(t * G) @ v, space.nelec)
    out = fci_ops.expm_apply(mv, psi, t)
    assert np.allclose(out, ref, atol=1e-11)
    prop = fci_ops.KrylovPropagator(mv, psi, eps_max=2 * t, tol=1e-12)
    assert prop.converged
    assert np.allclose(prop.state(t), ref, atol=1e-10)
    # unitarity (G anti-Hermitian)
    assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-12)
