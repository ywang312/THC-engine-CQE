"""Exact double factorization of CQE generators (thc_engine_cqe.factor.generator).

Operator identities are checked two ways: against fci_ops on CI sectors, and
against the independent Jordan-Wigner oracle on the whole Fock space (every
sector at once).
"""

import itertools

import numpy as np
import pytest
import scipy.linalg

from thc_engine_cqe import fci_ops, residual, systems
from thc_engine_cqe.cqe import run_cqe
from thc_engine_cqe.factor.generator import (
    GeneratorFactorization,
    adjoint,
    anti_hermitian_part,
    coupling_matrix,
    factorize_generator,
    hermitian_basis,
    pair_symmetrize,
    projected_generator,
    projection_compressor,
)
from thc_engine_cqe.fci_ops import FCISpace, Hamiltonian
from thc_engine_cqe.testing.fock import FockSpace


# ----------------------------------------------------------------- helpers
def _random_anti_hermitian(n, seed, complex_=True):
    """Full-index anti-Hermitian tensor (adjoint(g) = -g), not pair-symmetric."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n,) * 4)
    if complex_:
        x = x + 1j * rng.standard_normal((n,) * 4)
    return anti_hermitian_part(x)


def _fock_factorized(fs, fac, n_leaves=None):
    """Dense Fock-space matrix of sum_{k<n} c_k i Qhat_k^2 + Ohat_{F_n}, built with the oracle."""
    n = fac.n_leaves if n_leaves is None else n_leaves
    out = fs.one_body(fac.one_body_for(n)).toarray().astype(complex)
    for k in range(n):
        q = fs.one_body(fac.Q[k]).toarray()
        out += 1j * fac.coeffs[k] * (q @ q)
    return out


def _sector(fs, op, nelec):
    """Restrict a Fock-space operator to the (n_alpha, n_beta) sector in PySCF CI order."""
    addr = fs._addresses(nelec)
    dense = op.toarray() if hasattr(op, "toarray") else np.asarray(op)
    return dense[np.ix_(addr, addr)]


def _s_squared(ci, norb, nelec):
    """<S^2> of a (possibly complex) CI vector.

    S^2 is a real symmetric matrix in the determinant basis, so
    <psi|S^2|psi> = <Re psi|S^2|Re psi> + <Im psi|S^2|Im psi>.
    """
    from pyscf.fci import spin_op

    ci = np.asarray(ci)
    val = 0.0
    for part in (np.real(ci), np.imag(ci)):
        part = np.ascontiguousarray(part, dtype=float)
        val += float(np.vdot(part, spin_op.contract_ss(part, norb, nelec)))
    return val / float(np.vdot(ci, ci).real)


def _slope(g, A):
    """dE/deps at eps = 0 along exp(eps sum g e), from the residual: -Re sum g A."""
    return -float(np.real(np.sum(g * A)))


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def h4():
    return systems.hydrogen_chain(4, spacing=1.8, unit="bohr")


@pytest.fixture(scope="module")
def lih():
    return systems.lih()


@pytest.fixture(scope="module")
def physical(h4, lih):
    """name -> (active, ci, ResidualData, g = conj(A)) for real states on exact CQE runs."""
    out = {}
    for name, act, steps in (("H4-HF", h4, 0), ("H4-cqe3", h4, 3),
                             ("LiH-HF", lih, 0), ("LiH-cqe2", lih, 2)):
        space = FCISpace.of(act)
        ci = space.hf() if steps == 0 else run_cqe(act, max_iter=steps).ci
        rd = residual.compute_residual(act, ci)
        out[name] = (act, ci, rd, residual.descent_generator(rd.A))
    return out


PHYSICAL = ["H4-HF", "H4-cqe3", "LiH-HF", "LiH-cqe2"]


# ------------------------------------------------------- tensor-level maps
def test_tensor_maps_match_operators():
    """adjoint / anti_hermitian_part / pair_symmetrize act on operators as documented."""
    n = 3
    fs = FockSpace(n)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n,) * 4) + 1j * rng.standard_normal((n,) * 4)
    G = fs.two_body(x).toarray()
    assert np.allclose(adjoint(adjoint(x)), x)
    assert np.allclose(fs.two_body(adjoint(x)).toarray(), G.conj().T, atol=1e-12)
    assert np.allclose(fs.two_body(anti_hermitian_part(x)).toarray(), 0.5 * (G - G.conj().T), atol=1e-12)
    assert np.allclose(fs.two_body(pair_symmetrize(x)).toarray(), G, atol=1e-12)
    # adjoint commutes with pair exchange, so the anti-Hermitian part stays pair-symmetric
    gs = pair_symmetrize(x)
    assert np.allclose(anti_hermitian_part(gs), pair_symmetrize(anti_hermitian_part(x)))


def test_fock_kernel_is_exactly_pair_antisymmetric():
    """On the full Fock space, g -> sum g e has kernel = pair-antisymmetric tensors (N = 3)."""
    n = 3
    fs = FockSpace(n)
    ops = np.array([fs.e2(p, q, r, s).toarray().ravel()
                    for p, q, r, s in itertools.product(range(n), repeat=4)])
    rank = np.linalg.matrix_rank(ops, tol=1e-9)
    assert rank == n * n * (n * n + 1) // 2  # dimension of the pair-symmetric tensors
    rng = np.random.default_rng(1)
    x = rng.standard_normal((n,) * 4)
    anti = x - x.transpose(2, 3, 0, 1)
    assert abs(fs.two_body(anti)).max() < 1e-12


def test_hermitian_basis_is_orthonormal():
    for n in (1, 2, 4):
        b = hermitian_basis(n)
        assert b.shape == (n * n, n, n)
        assert np.allclose(b, np.conj(b.transpose(0, 2, 1)))  # Hermitian
        gram = np.einsum("apq,bpq->ab", np.conj(b), b)
        assert np.allclose(gram, np.eye(n * n), atol=1e-14)
        ns = n * (n + 1) // 2
        assert np.allclose(b[:ns].imag, 0) and np.allclose(b[:ns], b[:ns].transpose(0, 2, 1))
        assert np.allclose(b[ns:].real, 0) and np.allclose(b[ns:], -b[ns:].transpose(0, 2, 1))


@pytest.mark.parametrize("n", [3, 4])
def test_coupling_matrix_is_real_symmetric_iff_anti_hermitian(n):
    rng = np.random.default_rng(10 + n)
    g = _random_anti_hermitian(n, seed=n)
    gs = pair_symmetrize(g)
    gm = coupling_matrix(gs)
    scale = np.abs(gm).max()
    # anti-Hermitian + pair-symmetric -> real symmetric
    assert np.abs(gm.imag).max() < 1e-13 * scale
    assert np.abs(gm - gm.T).max() < 1e-13 * scale
    # anti-Hermitian but not pair-symmetric -> still real, but not symmetric
    gm_raw = coupling_matrix(g)
    assert np.abs(gm_raw.imag).max() < 1e-13 * scale
    assert np.abs(gm_raw - gm_raw.T).max() > 1e-2 * scale
    # the map is unitary: ||Gm||_F = ||g||_F
    assert np.isclose(np.linalg.norm(gm), np.linalg.norm(gs))
    # Hermitian generators map to purely imaginary Gm, and in general
    # ||Im Gm||_F = ||Hermitian part of gs||_F
    x = rng.standard_normal((n,) * 4) + 1j * rng.standard_normal((n,) * 4)
    xs = pair_symmetrize(x)
    herm = 0.5 * (xs + adjoint(xs))
    assert np.abs(coupling_matrix(herm).real).max() < 1e-13 * np.abs(herm).max() * n * n
    assert np.isclose(np.linalg.norm(coupling_matrix(xs).imag), np.linalg.norm(herm))


# ---------------------------------------------------------- operator identity
@pytest.mark.parametrize("complex_", [True, False], ids=["complex", "real"])
@pytest.mark.parametrize("n", [3, 4])
def test_operator_identity_on_whole_fock_space(n, complex_):
    """sum g e == sum_k c_k i Qhat_k^2 + Ohat_F as 4^N x 4^N matrices (all sectors)."""
    g = _random_anti_hermitian(n, seed=20 + n, complex_=complex_)
    fac = factorize_generator(g)
    fs = FockSpace(n)
    G = fs.two_body(g).toarray()
    assert np.abs(G + G.conj().T).max() < 1e-12  # the input really is anti-Hermitian
    assert np.abs(_fock_factorized(fs, fac) - G).max() < 1e-12 * max(1.0, np.abs(G).max())
    # number- and S_z-conserving (each leaf is a spin-free one-body operator squared)
    Gf = _fock_factorized(fs, fac)
    for op in (fs.number().toarray(), fs.s_z().toarray()):
        assert np.abs(Gf @ op - op @ Gf).max() < 1e-11


SECTORS = {3: [(1, 0), (0, 1), (1, 1), (2, 1), (3, 0), (3, 3)],
           4: [(1, 0), (2, 0), (2, 2), (3, 1), (4, 3)]}


@pytest.mark.parametrize("n,nelec", [(n, s) for n, ss in SECTORS.items() for s in ss],
                         ids=lambda v: str(v))
def test_operator_identity_on_ci_sectors(n, nelec):
    """fac.apply (leaf by leaf) == fci_ops.two_body(g) on many sectors, incl. 1-electron ones."""
    g = _random_anti_hermitian(n, seed=30 + n)
    fac = factorize_generator(g)
    space = FCISpace(n, nelec)
    ci = space.random(np.random.default_rng(sum(nelec)), complex_=True)
    ref = fci_ops.two_body(space, g, ci)
    out = fac.apply(space, ci)
    assert np.abs(out - ref).max() < 1e-12 * max(1.0, np.abs(ref).max())
    if sum(nelec) < 2:  # two-body operators vanish; the squares and F must cancel exactly
        assert np.abs(out).max() < 1e-12


@pytest.mark.parametrize("complex_", [True, False], ids=["complex", "real"])
@pytest.mark.parametrize("n", [3, 4])
def test_leaf_properties(n, complex_):
    g = _random_anti_hermitian(n, seed=40 + n, complex_=complex_)
    gs = pair_symmetrize(g)
    fac = factorize_generator(g)
    K, Q, c = fac.n_leaves, fac.Q, fac.coeffs
    assert fac.is_real == (not complex_)
    assert c.dtype == float and np.all(np.diff(np.abs(c)) <= 1e-12)  # decreasing |c|
    assert np.allclose(Q, np.conj(Q.transpose(0, 2, 1)), atol=1e-15)  # Hermitian
    gram = np.einsum("kpq,lqp->kl", Q, Q)  # tr(Q_k Q_l)
    assert np.allclose(gram, np.eye(K), atol=1e-13)  # orthonormal, unit Frobenius norm
    # weights sum to ||gs||^2; nothing but rounding is discarded
    assert np.isclose(fac.weights().sum(), np.sum(np.abs(gs) ** 2), rtol=1e-12)
    assert fac.discarded_weight < 1e-24 * np.sum(np.abs(gs) ** 2)
    # one-body term: F = -f = -i sum_k c_k Q_k^2, anti-Hermitian
    f = np.einsum("pqqs->ps", gs)
    F = fac.one_body
    assert np.allclose(F, -f, atol=1e-13)
    assert np.allclose(F, -F.conj().T, atol=1e-15)
    # the pair tensor of the leaves is gs; c_k = <T_k, gs> with T_k = i Q_k (x) Q_k
    assert np.allclose(fac.pair_tensor(), gs, atol=1e-13)
    T = 1j * np.einsum("kpq,krs->kpqrs", Q, Q)
    assert np.allclose(np.einsum("kpqrs,pqrs->k", np.conj(T), gs), c, atol=1e-12)
    # leaf_eigh: Q_k = V diag(mu) V^dag
    for k in range(K):
        mu, V = fac.leaf_eigh(k)
        assert np.allclose(V @ np.diag(mu) @ V.conj().T, Q[k], atol=1e-14)
        assert np.allclose(V.conj().T @ V, np.eye(n), atol=1e-14)
    assert fac.captured_fraction(K) == pytest.approx(1.0)
    assert fac.n_leaves_for(1.0) == K
    if complex_:
        assert K == n * n  # a generic complex generator has full rank N^2


def test_rejects_hermitian_input():
    g = _random_anti_hermitian(3, seed=1)
    h = 1j * g  # Hermitian
    with pytest.raises(ValueError, match="not anti-Hermitian"):
        factorize_generator(h)
    with pytest.raises(ValueError, match="not anti-Hermitian"):
        factorize_generator(g + 1e-3 * h)
    # a rounding-level Hermitian part is accepted and dropped
    fac = factorize_generator(g + 1e-13 * h)
    assert np.allclose(fac.pair_tensor(), pair_symmetrize(g), atol=1e-11)
    with pytest.raises(ValueError):
        factorize_generator(np.zeros((2, 2, 2)))


# --------------------------------------------------- real generators: pairing
def _real_generators(physical):
    gens = {"random-N4": _random_anti_hermitian(4, seed=50, complex_=False)}
    for name in ("H4-cqe3", "LiH-cqe2"):
        gens[name] = physical[name][3]
    return gens


def test_real_generator_pairing(physical):
    """Real g: exact (+c, Q), (-c, conj Q) pairs, ||Re Q||^2 = ||Im Q||^2 = 1/2, K <= N(N-1).

    Finding (documented in the module): for real g, Gm = [[0, X], [X^T, 0]] in
    the (real symmetric | imaginary antisymmetric) basis, so the spectrum of Gm
    is exactly +/- paired and at least N eigenvalues vanish.
    """
    for name, g in _real_generators(physical).items():
        assert np.isrealobj(g), name
        n = g.shape[0]
        fac = factorize_generator(g)
        K, c, Q = fac.n_leaves, fac.coeffs, fac.Q
        assert fac.is_real and K % 2 == 0 and K <= n * (n - 1), name
        assert np.array_equal(c[1::2], -c[0::2]), name
        assert np.array_equal(Q[1::2], np.conj(Q[0::2])), name
        assert np.allclose(np.sum(Q.real ** 2, axis=(1, 2)), 0.5, atol=1e-12), name
        assert np.allclose(np.sum(Q.imag ** 2, axis=(1, 2)), 0.5, atol=1e-12), name
        # the generic eigendecomposition of Gm has the same +/- paired spectrum
        gm = coupling_matrix(pair_symmetrize(g))
        ns = n * (n + 1) // 2
        assert np.abs(gm[:ns, :ns]).max() < 1e-14 and np.abs(gm[ns:, ns:]).max() < 1e-14
        lam = np.linalg.eigvalsh(gm.real)
        assert np.allclose(lam, -lam[::-1], atol=1e-13), name
        assert np.sum(np.abs(lam) < 1e-12 * np.abs(lam).max()) >= n, name
        ref = np.sort(np.concatenate([-c, np.zeros(n * n - K)]))
        assert np.allclose(lam, ref, atol=1e-12), name
        # even truncations are real; odd ones split a pair and are complex
        assert np.isrealobj(fac.pair_tensor(2)) and np.isrealobj(fac.one_body_for(2))
        g1 = fac.pair_tensor(1)
        assert np.iscomplexobj(g1) and np.abs(g1.imag).max() > 1e-3 * np.abs(g1).max()


def test_symmetry_degenerate_pairs_and_basis_independent_projection(physical):
    """LiH (C_inf_v, pi orbitals) has exactly degenerate sigma_j. The pairing stays exact,
    complete_group never splits a degenerate group, and the projection onto complete
    groups equals the spectral projector of Gm (so it does not depend on the eigenbasis)."""
    g = physical["LiH-cqe2"][3]
    n = g.shape[0]
    fac = factorize_generator(g)
    sigma = fac.coeffs[0::2]
    rel_gaps = -np.diff(sigma) / sigma[0]
    assert np.sum(rel_gaps < 1e-10) >= 2  # degenerate doublets from symmetry
    assert np.array_equal(fac.Q[1::2], np.conj(fac.Q[0::2]))
    a = np.abs(fac.coeffs)
    gm = coupling_matrix(pair_symmetrize(g)).real
    lam, w = np.linalg.eigh(gm)
    bm = hermitian_basis(n).reshape(n * n, n * n)
    for m in range(1, fac.n_leaves + 1):
        k = fac.complete_group(m)
        if k < fac.n_leaves:
            assert a[k] < (1 - 1e-8) * a[k - 1]  # k sits at a gap
        sel = np.abs(lam) >= a[k - 1] * (1 - 1e-8)
        assert sel.sum() == k
        proj = (w[:, sel] * lam[sel]) @ w[:, sel].T  # spectral part of Gm
        g_spec = (-1j * bm.T @ proj @ bm).reshape((n,) * 4)  # invert Gm = U^T (i g) U
        gl = projected_generator(g, m, complete_groups=True)
        assert np.isrealobj(gl)
        assert np.allclose(gl, g_spec, atol=1e-13)


def test_real_pair_is_anticommutator(physical):
    """Each pair i c (Qhat^2 - conj(Q)hat^2) equals c (Shat Ahat + Ahat Shat), Q = (S - iA)/sqrt2."""
    act, ci, _, g = physical["H4-cqe3"]
    space = FCISpace.of(act)
    fac = factorize_generator(g)
    rng = np.random.default_rng(3)
    psi = space.random(rng, complex_=True)
    for j in range(fac.n_leaves // 2):
        c, q = fac.coeffs[2 * j], fac.Q[2 * j]
        S, A = np.sqrt(2) * q.real, -np.sqrt(2) * q.imag
        assert np.allclose(S, S.T) and np.allclose(A, -A.T)
        assert np.isclose(np.linalg.norm(S), 1.0) and np.isclose(np.linalg.norm(A), 1.0)
        pair = sum(1j * cc * fci_ops.one_body(space, qq, fci_ops.one_body(space, qq, psi))
                   for cc, qq in ((c, q), (-c, np.conj(q))))
        anti = c * (fci_ops.one_body(space, S, fci_ops.one_body(space, A, psi))
                    + fci_ops.one_body(space, A, fci_ops.one_body(space, S, psi)))
        assert np.allclose(pair, anti, atol=1e-12)


# ------------------------------------------------------ physical generators
@pytest.mark.parametrize("name", PHYSICAL)
def test_physical_generator_identity(physical, name):
    """Generators g = conj(A) from H4 (1.8 bohr) and LiH along exact CQE runs."""
    act, ci, rd, g = physical[name]
    n = act.norb
    fac = factorize_generator(g)
    assert fac.is_real
    assert np.isclose(fac.weights().sum(), rd.residual_norm ** 2, rtol=1e-12)
    # CI sectors: the physical one and some others
    for nelec in [act.nelec, (1, 1), (n // 2 + 1, n // 2)]:
        space = FCISpace(n, nelec)
        psi = ci if nelec == act.nelec else space.random(np.random.default_rng(7), complex_=True)
        ref = fci_ops.two_body(space, g, psi)
        assert np.abs(fac.apply(space, psi) - ref).max() < 1e-12 * max(1.0, np.abs(ref).max())
    # whole Fock space against the Jordan-Wigner oracle
    fs = FockSpace(n)
    G = fs.two_body(g)
    if n <= 4:
        assert np.abs(_fock_factorized(fs, fac) - G.toarray()).max() < 1e-12
    else:  # 4096-dim: compare actions on random Fock vectors
        rng = np.random.default_rng(8)
        qs = [fs.one_body(q) for q in fac.Q]
        fo = fs.one_body(fac.one_body)
        for _ in range(2):
            v = rng.standard_normal(fs.dim) + 1j * rng.standard_normal(fs.dim)
            out = fo @ v
            for ck, qo in zip(fac.coeffs, qs):
                out = out + 1j * ck * (qo @ (qo @ v))
            ref = G @ v
            assert np.abs(out - ref).max() < 1e-12 * np.abs(ref).max()


@pytest.mark.parametrize("name", ["H4-HF", "LiH-HF"])
def test_leaves_from_hartree_fock_live_on_ov_block(physical, name):
    """From RHF, A has (vir occ | vir occ) support, so at most 2 n_occ n_vir leaves are
    non-negligible and they vanish on the oo and vv blocks (Brillouin terms ~1e-9 aside)."""
    act, _, _, g = physical[name]
    nocc = act.nelec[0]
    nvir = act.norb - nocc
    fac = factorize_generator(g)
    big = np.abs(fac.coeffs) > 1e-6 * np.abs(fac.coeffs).max()
    assert big.sum() <= 2 * nocc * nvir
    q = fac.Q[big]
    assert np.abs(q[:, :nocc, :nocc]).max() < 1e-6
    assert np.abs(q[:, nocc:, nocc:]).max() < 1e-6


# ------------------------------------------------------ projected generators
def _descent_cases(physical, small_random_system):
    sysm = small_random_system
    space = FCISpace.of(sysm)
    psi = space.random(np.random.default_rng(9), complex_=True)
    cases = {"complex-state": (sysm, psi)}
    act, ci, _, _ = physical["H4-cqe3"]
    cases["H4-cqe3"] = (act, ci)
    return cases


def test_projected_generator_descent(physical, small_random_system):
    """For g = conj(A): slope(g_L) = -||g_L||^2 <= 0, monotone in L, exact at full rank."""
    for label, (act, psi) in _descent_cases(physical, small_random_system).items():
        ham = Hamiltonian(act)
        space = ham.space
        rd = residual.compute_residual(act, psi, ham)
        A = rd.A
        g = residual.descent_generator(A)
        fac = factorize_generator(g)
        K = fac.n_leaves
        hpsi = ham.apply(psi)
        slopes = []
        for n in range(K + 1):
            gl = projected_generator(g, n)
            assert np.allclose(gl, fac.pair_tensor(n), atol=1e-13)
            s = _slope(gl, A)
            slopes.append(s)
            assert np.isclose(s, -np.sum(np.abs(gl) ** 2), rtol=1e-10, atol=1e-14), label
            assert np.isclose(s, -fac.captured_fraction(n) * rd.residual_norm ** 2,
                              rtol=1e-10, atol=1e-14), label
            # orthogonal projection: the remainder is orthogonal to g_L
            assert abs(np.vdot(g - gl, gl)) < 1e-12 * np.sum(np.abs(g) ** 2), label
            # the residual-formula slope is the true slope 2 Re <H psi | G_L psi>
            exact = 2.0 * np.real(np.vdot(hpsi, fci_ops.two_body(space, gl, psi)))
            assert np.isclose(s, exact, rtol=1e-9, atol=1e-13), label
        slopes = np.array(slopes)
        assert np.all(slopes <= 1e-15) and np.all(np.diff(slopes) <= 1e-14), label
        assert np.isclose(slopes[-1], -rd.residual_norm ** 2, rtol=1e-12), label
        assert np.allclose(projected_generator(g, K), g, atol=1e-13 * np.abs(g).max()), label
        assert np.allclose(projected_generator(g, K + 5), g, atol=1e-13 * np.abs(g).max()), label


def test_projected_generator_finite_difference(physical):
    """End to end: E(exp(eps G_L) psi) has slope -||g_L||^2 for an odd (complex) truncation."""
    act, psi, rd, g = physical["H4-cqe3"]
    ham = Hamiltonian(act)
    space = ham.space
    gl = projected_generator(g, 3)
    assert np.iscomplexobj(gl)
    h = 1e-4

    def energy(eps):
        v = fci_ops.expm_apply(lambda x: fci_ops.two_body(space, gl, x), psi.astype(complex), eps)
        return ham.energy(v)

    fd = (energy(h) - energy(-h)) / (2 * h)
    assert np.isclose(fd, -np.sum(np.abs(gl) ** 2), rtol=1e-6)


def test_complete_groups_keep_real_generators_real(physical):
    act, ci, rd, g = physical["H4-cqe3"]
    fac = factorize_generator(g)
    for n in range(fac.n_leaves + 1):
        m = fac.complete_group(n)
        assert m >= n and m % 2 == 0
        gl = projected_generator(g, n, complete_groups=True)
        assert np.isrealobj(gl)
        full = np.einsum("k,kpq,krs->pqrs", 1j * fac.coeffs[:m], fac.Q[:m], fac.Q[:m])
        assert np.abs(full.imag).max() < 1e-14  # only rounding is dropped
        assert np.allclose(gl, full.real, atol=1e-15)


@pytest.mark.parametrize("kw", [{"fraction": 0.9}, {"n_leaves": 2}], ids=["frac0.9", "L2"])
def test_projection_compressor_in_cqe_driver(h4, kw):
    """run_cqe with projected generators: every step descends, nothing is sign-flipped."""
    comp = projection_compressor(**kw)
    res = run_cqe(h4, max_iter=4, compress=comp, keep_generators=True, keep_residuals=True)
    assert len(res.steps) == 4 and len(comp.history) == 4
    assert np.all(np.diff(res.energies) <= 1e-12)
    assert np.isrealobj(res.ci)
    for step, rec in zip(res.steps, comp.history):
        assert np.isrealobj(step.generator)
        expect = projected_generator(np.conj(step.residual), rec["n_leaves"], complete_groups=True)
        assert np.allclose(step.generator, expect, atol=1e-12)  # not sign-flipped
        assert step.slope < 0
        assert np.isclose(step.captured_descent, rec["captured"], rtol=1e-9)
        if "fraction" in kw:
            assert rec["captured"] >= 0.9 - 1e-12
        else:
            assert rec["n_leaves"] == 2


def test_complex_truncation_in_cqe_driver_with_complex_start(h4):
    """Odd truncations of real generators are complex; run_cqe then needs a complex ci0."""
    comp = projection_compressor(n_leaves=1, complete_groups=False)
    ci0 = FCISpace.of(h4).hf().astype(complex)
    res = run_cqe(h4, ci0=ci0, max_iter=3, compress=comp)
    assert np.iscomplexobj(res.ci)
    assert np.abs(res.ci.imag).max() > 1e-3  # the trajectory left the real manifold
    assert np.all(np.diff(res.energies) <= 1e-12)
    for step, rec in zip(res.steps, comp.history):
        assert rec["n_leaves"] == 1 and step.slope < 0
        assert np.isclose(step.captured_descent, rec["captured"], rtol=1e-9)
    # the energy reported by the line search is the energy of the returned state
    assert np.isclose(Hamiltonian(h4).energy(res.ci), res.steps[-1].energy_after, atol=1e-10)


def test_complex_truncation_on_a_real_start_is_exact(h4):
    """Regression for a fixed core bug: KrylovPropagator used to take its dtype from
    the start vector, so from a real ci0 the complex one-leaf projection g_1 was
    propagated as Re(g_1) = g_2 / 2, i.e. along the two-leaf trajectory, while g_1
    was recorded. Now real and complex starts agree, and both differ from the
    two-leaf run."""
    space = FCISpace.of(h4)
    rd = residual.compute_residual(h4, space.hf())
    rdc = residual.compute_residual(h4, space.hf().astype(complex))
    assert np.isrealobj(rd.A) and np.iscomplexobj(rdc.A)
    g = residual.descent_generator(rd.A)
    fac = factorize_generator(g)
    assert np.iscomplexobj(fac.pair_tensor(1)) and np.abs(fac.pair_tensor(1).imag).max() > 1e-3
    assert np.abs(fac.pair_tensor(1).real - 0.5 * fac.pair_tensor(2)).max() < 1e-15
    runs = {}
    for name, ci0, n in [("real", None, 1), ("complex", space.hf().astype(complex), 1),
                         ("two-leaf", None, 2)]:
        comp = projection_compressor(n_leaves=n, complete_groups=False)
        runs[name] = (run_cqe(h4, ci0=ci0, max_iter=3, compress=comp), comp.history)
    res, hist = runs["real"]
    assert np.iscomplexobj(res.ci)
    assert np.allclose(res.energies, runs["complex"][0].energies, atol=1e-8)
    assert np.abs(res.energies - runs["two-leaf"][0].energies).max() > 1e-4
    for step, rec in zip(res.steps, hist):
        assert rec["n_leaves"] == 1 and step.slope < 0
        assert np.isclose(step.captured_descent, rec["captured"], rtol=1e-9)
    assert np.isclose(Hamiltonian(h4).energy(res.ci), res.steps[-1].energy_after, atol=1e-10)


def _krylov_case():
    space = FCISpace(4, (2, 2))
    g = _random_anti_hermitian(4, seed=95)

    def mv(x):
        return fci_ops.two_body(space, g, x)

    psi = space.hf()  # real
    return mv, psi, fci_ops.expm_apply(mv, psi, 0.05)


def test_krylov_complex_start_with_complex_generator():
    """A complex start vector with a complex generator is exact."""
    mv, psi, ref = _krylov_case()
    out = fci_ops.KrylovPropagator(mv, psi.astype(complex), 0.1).state(0.05)
    assert np.allclose(out, ref, atol=1e-10)


@pytest.mark.filterwarnings("error::numpy.exceptions.ComplexWarning")
def test_krylov_real_start_with_complex_generator():
    """A real start vector with a complex generator: the basis is promoted to complex."""
    mv, psi, ref = _krylov_case()
    out = fci_ops.KrylovPropagator(mv, psi, 0.1).state(0.05)
    assert np.allclose(out, ref, atol=1e-10)


def test_truncated_operator_matches_pair_tensor(physical):
    """apply(n) is the operator of pair_tensor(n): kept squares plus their own F_n."""
    act, ci, _, g = physical["H4-cqe3"]
    space = FCISpace.of(act)
    fac = factorize_generator(g)
    psi = space.random(np.random.default_rng(4), complex_=True)
    for n in (0, 1, 2, 5, fac.n_leaves):
        ref = fci_ops.two_body(space, fac.pair_tensor(n), psi)
        assert np.allclose(fac.apply(space, psi, n_leaves=n), ref, atol=1e-13)
        fn = fac.one_body_for(n)
        assert np.allclose(fn, -np.einsum("pqqs->ps", fac.pair_tensor(n)), atol=1e-14)
    assert fac.one_body_for(fac.n_leaves) is fac.one_body


# ------------------------------------------------------------- compilation
@pytest.mark.parametrize("source", ["complex-N3", "H4-cqe3"])
def test_leaf_compiles_as_orbital_rotation_plus_phases(physical, source):
    """Qhat_k = U(V) D U(V)^dag and exp(i t c Qhat^2) = U(V) exp(i t c D^2) U(V)^dag (oracle)."""
    if source == "complex-N3":
        g = _random_anti_hermitian(3, seed=60)
    else:
        g = physical[source][3]
    fac = factorize_generator(g)
    n = fac.norb
    fs = FockSpace(n)
    t = 0.37
    for k in range(min(3, fac.n_leaves)):
        mu, V = fac.leaf_eigh(k)
        kappa = scipy.linalg.logm(V)
        kappa = 0.5 * (kappa - kappa.conj().T)
        assert np.allclose(scipy.linalg.expm(kappa), V, atol=1e-12)
        U = scipy.linalg.expm(fs.one_body(kappa).toarray())  # U(V) on the whole Fock space
        for spin in (0, 1):  # convention: U a+_q U^dag = sum_p V[p,q] a+_p
            for qi in range(n):
                lhs_q = U @ fs.adag(qi, spin).toarray() @ U.conj().T
                rhs_q = sum(V[p, qi] * fs.adag(p, spin).toarray() for p in range(n))
                assert np.allclose(lhs_q, rhs_q, atol=1e-12)
        D = fs.one_body(np.diag(mu)).toarray()
        assert np.allclose(D, np.diag(np.diag(D)))  # diagonal in the occupation basis
        Qo = fs.one_body(fac.Q[k]).toarray()
        assert np.allclose(U @ D @ U.conj().T, Qo, atol=1e-12)
        phase = np.exp(1j * t * fac.coeffs[k] * np.diag(D) ** 2)  # exp(i t c D^2), diagonal
        lhs = scipy.linalg.expm(1j * t * fac.coeffs[k] * (Qo @ Qo))
        assert np.allclose(U @ (phase[:, None] * U.conj().T), lhs, atol=1e-11)
    # the one-body term is the orbital rotation U(exp(t F))
    F = fac.one_body
    assert np.allclose(F, -F.conj().T)
    assert np.allclose(scipy.linalg.expm(t * F) @ scipy.linalg.expm(t * F).conj().T, np.eye(n), atol=1e-13)


# ----------------------------------------------------------------- Trotter
@pytest.mark.parametrize("order", [1, 2])
def test_trotter_matches_explicit_product(order):
    """trotter_apply equals the documented product of exact exponentials (oracle, dense expm)."""
    n, nelec = 3, (2, 1)
    g = _random_anti_hermitian(n, seed=70)
    fac = factorize_generator(g)
    fs = FockSpace(n)
    space = FCISpace(n, nelec)
    psi = space.random(np.random.default_rng(5), complex_=True)
    eps = 0.3
    for n_leaves, n_steps in ((None, 1), (4, 2)):
        L = fac.n_leaves if n_leaves is None else n_leaves
        X = [_sector(fs, fs.one_body(fac.one_body_for(L)), nelec)]
        for k in range(L):
            q = _sector(fs, fs.one_body(fac.Q[k]), nelec)
            X.append(1j * fac.coeffs[k] * (q @ q))
        h = eps / n_steps
        if order == 1:
            step = np.eye(space.size, dtype=complex)
            for x in X:  # X_0 acts first
                step = scipy.linalg.expm(h * x) @ step
        else:
            seq = [(x, 0.5 * h) for x in X[:-1]] + [(X[-1], h)] + [(x, 0.5 * h) for x in X[-2::-1]]
            step = np.eye(space.size, dtype=complex)
            for x, tt in seq:
                step = scipy.linalg.expm(tt * x) @ step
        ref = np.linalg.matrix_power(step, n_steps) @ psi.ravel()
        out = fac.trotter_apply(space, eps, psi, order=order, n_leaves=n_leaves, n_steps=n_steps)
        assert np.allclose(out.ravel(), ref, atol=1e-11)


def _trotter_cases(physical):
    n, nelec = 4, (2, 2)
    space = FCISpace(n, nelec)
    g = _random_anti_hermitian(n, seed=80)
    psi = space.random(np.random.default_rng(6), complex_=True)
    act, ci, _, g_h4 = physical["H4-cqe3"]
    return {"complex-N4": (space, g, psi, 0.01), "H4-cqe3": (FCISpace.of(act), g_h4, ci, 0.1)}


@pytest.mark.parametrize("case", ["complex-N4", "H4-cqe3"])
def test_trotter_local_error_order(physical, case):
    """Halving eps divides the local error by ~4 (order 1) and ~8 (order 2)."""
    space, g, psi, eps = _trotter_cases(physical)[case]
    fac = factorize_generator(g)

    def exact(e):
        return fci_ops.expm_apply(lambda x: fci_ops.two_body(space, g, x), psi.astype(complex), e)

    errs = {}
    for order in (1, 2):
        errs[order] = [np.linalg.norm(fac.trotter_apply(space, e, psi, order=order) - exact(e))
                       for e in (eps, eps / 2)]
    r1 = errs[1][0] / errs[1][1]
    r2 = errs[2][0] / errs[2][1]
    assert 3.6 < r1 < 4.4, (r1, errs)
    assert 7.2 < r2 < 8.8, (r2, errs)
    assert errs[2][0] < errs[1][0]
    # global error with n_steps substeps: ~1/n (order 1), ~1/n^2 (order 2)
    e8 = 8 * eps
    ref = exact(e8)
    g1 = [np.linalg.norm(fac.trotter_apply(space, e8, psi, order=1, n_steps=m) - ref) for m in (8, 16)]
    g2 = [np.linalg.norm(fac.trotter_apply(space, e8, psi, order=2, n_steps=m) - ref) for m in (8, 16)]
    assert 1.8 < g1[0] / g1[1] < 2.2, g1
    assert 3.6 < g2[0] / g2[1] < 4.4, g2


def test_trotter_preserves_norm_and_singlet(h4):
    """Spin-free leaves: a singlet stays a singlet, and the norm is preserved."""
    space = FCISpace.of(h4)
    norb, nelec = h4.norb, h4.nelec
    psi0 = space.hf()
    from pyscf.fci import spin_op

    # helper sanity: agrees with PySCF on real vectors and detects non-singlets
    rnd = space.random(np.random.default_rng(2))
    assert np.isclose(_s_squared(rnd, norb, nelec), spin_op.spin_square0(rnd, norb, nelec)[0])
    assert _s_squared(rnd, norb, nelec) > 0.1
    assert abs(_s_squared(psi0, norb, nelec)) < 1e-12
    fac = factorize_generator(_random_anti_hermitian(norb, seed=90))
    for order, n_leaves in ((1, None), (2, None), (2, 5)):
        out = fac.trotter_apply(space, 0.2, psi0, order=order, n_leaves=n_leaves)
        assert np.iscomplexobj(out) and abs(np.vdot(psi0, out)) < 0.99  # it really moved
        assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-12)
        assert abs(_s_squared(out, norb, nelec)) < 1e-10


# --------------------------------------------------------------- edge cases
def test_zero_generator_and_bookkeeping():
    n = 3
    fac = factorize_generator(np.zeros((n,) * 4))
    assert fac.n_leaves == 0 and fac.captured_fraction(0) == 1.0 and fac.n_leaves_for(0.5) == 0
    space = FCISpace(n, (1, 1))
    psi = space.random(np.random.default_rng(0))
    assert np.allclose(fac.apply(space, psi), 0)
    assert np.allclose(fac.trotter_apply(space, 0.5, psi, order=2), psi)
    assert np.allclose(fac.pair_tensor(), 0)
    # hand-made leaves: |c| = 3, 3, 2, 1, 1 -> weights 9, 9, 4, 1, 1 (total 24)
    q = np.zeros((5, 2, 2), dtype=complex)
    q[:, 0, 0] = 1.0
    hand = GeneratorFactorization(2, np.array([3.0, -3.0, 2.0, 1.0, 1.0 + 1e-12]), q, np.zeros((2, 2)))
    assert [hand.complete_group(m) for m in range(7)] == [0, 2, 2, 3, 5, 5, 5]
    assert np.isclose(hand.captured_fraction(2), 18 / 24)
    assert hand.n_leaves_for(0.75) == 2 and hand.n_leaves_for(0.76) == 3
    assert hand.n_leaves_for(1.0) == 5 and hand.n_leaves_for(0.0) == 0
    with pytest.raises(ValueError):
        hand.n_leaves_for(1.5)
    with pytest.raises(ValueError):
        fac.trotter_apply(space, 0.1, psi, order=3)
    with pytest.raises(ValueError):
        fac.apply(FCISpace(4, (1, 1)), FCISpace(4, (1, 1)).hf())
    with pytest.raises(ValueError):
        projection_compressor()
