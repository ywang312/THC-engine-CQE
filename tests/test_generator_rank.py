"""Generator rank/sparsity diagnostics and the CNOT cost model.

Covers thc_engine_cqe.diagnostics.cost (exact spin-orbital expansion against the
Jordan-Wigner oracle, explicit circuits, two-qubit CNOT counts, cost formulas)
and thc_engine_cqe.diagnostics.generator_rank (leaf/element profiles, orbital
bases, the first-step rank relation, the rank study and compressed CQE) on H4.
"""

import numpy as np
import pytest
import scipy.linalg

from thc_engine_cqe import fci_ops, residual, systems
from thc_engine_cqe.cqe import run_cqe
from thc_engine_cqe.diagnostics import cost
from thc_engine_cqe.diagnostics import generator_rank as gr
from thc_engine_cqe.factor.generator import (
    GeneratorFactorization,
    adjoint,
    anti_hermitian_part,
    factorize_generator,
    pair_symmetrize,
)
from thc_engine_cqe.fci_ops import FCISpace
from thc_engine_cqe.testing.fock import FockSpace


@pytest.fixture(scope="module", autouse=True)
def _single_openmp_thread():
    """PySCF's OpenMP threads oversubscribe cores shared by parallel test runs; restored afterwards."""
    from pyscf import lib

    old = lib.num_threads()
    lib.num_threads(1)
    yield
    lib.num_threads(old)


# ----------------------------------------------------------------- helpers
def _random_anti_hermitian(n, seed, complex_):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n,) * 4)
    if complex_:
        x = x + 1j * rng.standard_normal((n,) * 4)
    return anti_hermitian_part(x)  # deliberately not pair-symmetric


def _fock_excitation(fs, exc):
    """Dense Fock-space matrix of one Excitation, built from the oracle's a, a+."""
    n = fs.norb

    def ad(j):
        return fs.adag(j % n, j // n)

    def an(j):
        return fs.a(j % n, j // n)

    (c0, c1), (d0, d1) = exc.creators, exc.annihilators
    tau = (ad(c0) @ ad(c1) @ an(d1) @ an(d0)).toarray()
    if exc.kind == "diagonal":
        return exc.amplitude * tau
    return exc.amplitude * tau - np.conj(exc.amplitude) * tau.conj().T


@pytest.fixture(scope="module")
def h4():
    return systems.hydrogen_chain(4, spacing=1.8, unit="bohr")


@pytest.fixture(scope="module")
def h4_fci(h4):
    return h4.fci()[0]


@pytest.fixture(scope="module")
def h4_traj(h4, h4_fci):
    """Exact CQE: (result, states) with states[t] the state at iteration t."""
    states = [FCISpace.of(h4).hf()]
    res = run_cqe(h4, max_iter=6, keep_generators=True, keep_residuals=True, e_exact=h4_fci,
                  callback=lambda step, ci: states.append(ci))
    return res, states


@pytest.fixture(scope="module")
def h4_study(h4, h4_fci):
    return gr.rank_study(h4, max_iter=5, e_fci=h4_fci)


# ============================================================ cost: expansion
@pytest.mark.parametrize("complex_", [False, True], ids=["real", "complex"])
@pytest.mark.parametrize("n", [3, 4])
def test_spin_orbital_expansion_is_exact_on_fock_space(n, complex_):
    """sum_X (t tau - h.c.) + sum_Y T n n == sum g e as 4^N x 4^N matrices (all sectors)."""
    g = _random_anti_hermitian(n, seed=7 * n + complex_, complex_=complex_)
    fs = FockSpace(n)
    exc = cost.spin_orbital_excitations(g, tol=1e-14)
    G = fs.two_body(g).toarray()
    R = sum(_fock_excitation(fs, e) for e in exc)
    assert np.abs(R - G).max() < 1e-12 * max(1.0, np.abs(G).max())
    kinds = {e.kind for e in exc}
    assert kinds <= {"double", "single", "diagonal"}
    assert ("diagonal" in kinds) == complex_  # number-number phases need imaginary coefficients
    for e in exc:
        assert e.kind == "diagonal" or e.creators > e.annihilators  # orientation convention
        assert len(e.modes) == {"double": 4, "single": 3, "diagonal": 2}[e.kind]
        if e.kind == "diagonal":
            assert abs(np.real(e.amplitude)) == 0.0
    # every operator appears once (merged), and amplitudes are sorted
    keys = [(e.creators, e.annihilators) for e in exc]
    assert len(set(keys)) == len(keys)
    amps = np.abs([e.amplitude for e in exc])
    assert np.all(np.diff(amps) <= 1e-15)
    # pair-exchange partners describe the same operator
    exc2 = cost.spin_orbital_excitations(pair_symmetrize(g), tol=1e-14)
    assert {(e.creators, e.annihilators): e.amplitude for e in exc2}.keys() == set(keys)


def test_expansion_merges_identical_operators_by_hand():
    """One spin-free orbit -> the expected spin-orbital excitations (N = 4, i,j = 0,1; a,b = 2,3)."""
    n, x = 4, 0.3

    def orbit(p, q, r, s):
        g = np.zeros((n,) * 4)
        g[p, q, r, s] = g[r, s, p, q] = x
        g[q, p, s, r] = g[s, r, q, p] = -x  # adjoint partners: real anti-Hermitian
        return g

    # a != b, i != j: alpha-alpha, beta-beta, alpha-beta, beta-alpha doubles, each 2x
    exc = cost.spin_orbital_excitations(orbit(2, 0, 3, 1))
    got = {(e.creators, e.annihilators): e.amplitude for e in exc}
    assert got == pytest.approx({((2, 3), (0, 1)): 2 * x, ((6, 7), (4, 5)): 2 * x,
                                 ((2, 7), (0, 5)): 2 * x, ((3, 6), (1, 4)): 2 * x})
    assert all(e.kind == "double" for e in exc)
    # |S| = 0, 0, 2, 2 -> 13 + 13 + 17 + 17
    assert sorted(cost.parity_length(e) for e in exc) == [0, 0, 2, 2]
    assert cost.cnot_cost_elementwise(exc, n, "all") == 60
    # linear: same-spin doubles are contiguous AACC blocks (19 each); the opposite-spin
    # ones gather with 6 and 2 swaps into ACAC, which needs one more fSWAP pair
    by = {(e.creators, e.annihilators): e for e in exc}
    assert [cost.gather_swaps(by[k].modes) for k in sorted(by)] == [0, 6, 2, 0]
    assert [cost.role_swaps(by[k]) for k in sorted(by)] == [0, 1, 1, 0]
    assert cost.cnot_cost_elementwise(exc, n, "linear") == 19 + 19 + (19 + 4 * 7) + (19 + 4 * 3)
    # a == b: same-spin terms vanish (a+_a a+_a = 0), two opposite-spin doubles remain
    exc = cost.spin_orbital_excitations(orbit(2, 0, 2, 1))
    got = {(e.creators, e.annihilators): e.amplitude for e in exc}
    assert got == pytest.approx({((2, 6), (0, 5)): 2 * x, ((2, 6), (1, 4)): 2 * x})
    # e_{20,11} = sum n_{1 tau} a+_{2 sigma} a_{0 sigma}: four number-conditioned singles
    exc = cost.spin_orbital_excitations(orbit(2, 0, 1, 1))
    assert all(e.kind == "single" for e in exc) and len(exc) == 4
    assert {(e.control, e.particle, e.hole) for e in exc} == {(1, 2, 0), (5, 2, 0), (1, 6, 4), (5, 6, 4)}
    assert np.allclose(np.abs([e.amplitude for e in exc]), 2 * x)
    # |S'|: (1,2,0) -> {1} minus control = 0; (5,2,0) -> 1; (1,6,4) -> 1; (5,6,4) -> 0
    by = {(e.control, e.particle, e.hole): e for e in exc}
    assert [cost.excitation_cnots(by[k], "all") for k in [(1, 2, 0), (5, 2, 0), (1, 6, 4), (5, 6, 4)]] == [6, 8, 8, 6]
    # linear: gathers 0, 3, 3, 0; control between particle and hole costs one fSWAP pair (+4)
    assert [cost.excitation_cnots(by[k], "linear") for k in [(1, 2, 0), (5, 2, 0), (1, 6, 4), (5, 6, 4)]] == [10, 18, 18, 10]


def test_expansion_rejects_hermitian_input_and_handles_zero():
    g = _random_anti_hermitian(3, seed=1, complex_=True)
    with pytest.raises(ValueError, match="not anti-Hermitian"):
        cost.spin_orbital_excitations(1j * g)
    assert cost.spin_orbital_excitations(np.zeros((3,) * 4)) == []
    with pytest.raises(ValueError):
        cost.spin_orbital_excitations(np.zeros((3, 3, 3)))


# ======================================================= cost: gate counts
def test_two_qubit_cnot_counts():
    """Shende-Bullock-Markov criterion: sanity cases, then the gates the model relies on."""
    rng = np.random.default_rng(0)

    def haar(d):
        q, r = np.linalg.qr(rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)))
        return q * (np.diag(r) / np.abs(np.diag(r)))

    cnot = np.eye(4)[[0, 1, 3, 2]]
    swap = np.eye(4)[[0, 2, 1, 3]]
    cz = np.diag([1.0, 1, 1, -1])
    iswap = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])
    count = cost.two_qubit_cnot_count
    assert [count(np.kron(haar(2), haar(2))), count(cnot), count(cz), count(iswap), count(swap),
            count(haar(4))] == [0, 1, 1, 2, 3, 3]
    # fermionic gates on two adjacent Jordan-Wigner modes (FockSpace(1): modes 0 and 1)
    fs = FockSpace(1)
    a0, a1 = fs.a(0, 0).toarray(), fs.a(0, 1).toarray()
    n0, n1 = a0.T @ a0, a1.T @ a1
    hop = a0.T @ a1
    fswap = np.eye(4) + hop + hop.T - n0 - n1
    assert np.allclose(fswap @ fswap, np.eye(4))
    z = np.diag([1.0, -1.0])
    zz = scipy.linalg.expm(0.37j * np.kron(z, z))
    givens = scipy.linalg.expm(0.53 * (np.exp(0.8j) * hop - np.exp(-0.8j) * hop.conj().T))
    cphase = np.diag([1, 1, 1, np.exp(0.7j)])
    assert count(fswap) == cost.FSWAP_CNOTS == 2
    assert count(zz) == cost.ZZ_CNOTS == 2
    assert count(fswap @ zz) == cost.FSWAP_ZZ_CNOTS == 3
    assert count(givens) == cost.GIVENS_CNOTS == 2
    assert count(cphase) == 2


@pytest.mark.parametrize("n,complex_,stride", [(3, False, 1), (3, True, 1), (4, True, 5)],
                         ids=["N3-real", "N3-complex", "N4-complex"])
def test_explicit_circuits_match_exact_fermionic_unitaries(n, complex_, stride):
    """excitation_circuit == expm(t tau - h.c.) on all 2^(2N) states, with the model's CNOT count.

    The doubles use the explicit 14-CNOT core (the model's default, 13, is the
    published Yordanov et al. circuit); singles (6 + 2|S|) and diagonals (2) match
    the default model.
    """
    g = _random_anti_hermitian(n, seed=100 + n + complex_, complex_=complex_)
    fs = FockSpace(n)
    exc = cost.spin_orbital_excitations(g, tol=1e-14)[::stride]
    assert {e.kind for e in exc} >= {"double", "single"}
    for e in exc:
        e = cost.Excitation(e.kind, e.creators, e.annihilators, 0.9 * e.amplitude)
        gates = cost.excitation_circuit(e)
        U = cost.circuit_unitary(gates, 2 * n)
        assert np.abs(U - scipy.linalg.expm(_fock_excitation(fs, e))).max() < 1e-12, e
        assert cost.circuit_cnots(gates) == cost.excitation_cnots(e, "all", double_core=14), e
        if e.kind != "double":
            assert cost.circuit_cnots(gates) == cost.excitation_cnots(e, "all"), e


def test_cost_formulas_by_hand():
    assert cost.orbital_rotation_cnots(4) == 2 * 6 * 2
    assert cost.orbital_rotation_cnots(4, "linear") == 24
    assert cost.diagonal_layer_cnots(4, 4, "all") == 2 * 28
    assert cost.diagonal_layer_cnots(4, 4, "linear") == 3 * 28
    assert cost.diagonal_layer_cnots(4, 2, "all") == 12
    assert cost.diagonal_layer_cnots(4, 2, "linear") == min(3 * 6 + 2 * 22, 5 * 6)
    assert cost.diagonal_layer_cnots(6, 3, "linear") == 75
    assert [cost.gather_swaps(m) for m in [(0, 1, 2, 3), (0, 2, 5, 7), (1, 3, 4, 6), (5, 9)]] == [0, 6, 2, 3]
    with pytest.raises(ValueError):
        cost.orbital_rotation_cnots(4, "ring")
    # hand-made leaves: Q_0 = diag(1, -1, 0, 0)/sqrt2 (n_a = 2), Q_1 = 1/2 (n_a = 4)
    q = np.zeros((2, 4, 4), dtype=complex)
    q[0] = np.diag([1.0, -1.0, 0, 0]) / np.sqrt(2)
    q[1] = np.eye(4) / 2
    fac = GeneratorFactorization(4, np.array([1.0, -0.5]), q, np.zeros((4, 4)))
    assert [cost.leaf_active_orbitals(fac, k) for k in range(2)] == [2, 4]
    assert cost.cnot_cost_leaves(fac, 0) == 0
    assert cost.cnot_cost_leaves(fac, 1) == 24 + 12 + 24
    assert cost.cnot_cost_leaves(fac, 2) == 2 * 24 + 12 + 56 + 24
    assert cost.cnot_cost_leaves(fac, 2, final_rotation=False) == 2 * 24 + 12 + 56
    assert cost.cnot_cost_leaves(fac, 2, "linear") == 2 * 24 + 30 + 84 + 24
    # second order (Strang): 2L rotations and 2L - 1 diagonal layers (leaf L-1 once, the others twice)
    assert cost.leaf_schedule(3, 1) == ((0, 1.0), (1, 1.0), (2, 1.0))
    assert cost.leaf_schedule(3, 2) == ((0, 0.5), (1, 0.5), (2, 1.0), (1, 0.5), (0, 0.5))
    assert cost.leaf_schedule(1, 2) == ((0, 1.0),) and cost.leaf_schedule(0, 2) == ()
    assert cost.cnot_cost_leaves(fac, 1, order=2) == cost.cnot_cost_leaves(fac, 1) == 24 + 12 + 24
    assert cost.cnot_cost_leaves(fac, 2, order=2) == 4 * 24 + 2 * 12 + 56
    assert cost.cnot_cost_leaves(fac, 2, order=2, final_rotation=False) == 3 * 24 + 2 * 12 + 56
    assert cost.cnot_cost_leaves(fac, 2, "linear", order=2) == 4 * 24 + 2 * 30 + 84
    assert cost.cnot_cost_leaves(fac, 0, order=2) == 0
    for bad in (0, 3):
        with pytest.raises(ValueError):
            cost.cnot_cost_leaves(fac, 2, order=bad)
        with pytest.raises(ValueError):
            cost.leaf_schedule(2, bad)


def test_rhf_leaves_live_on_ov_block_active_orbitals():
    """n_a of every RHF leaf is even and <= 2 min(n_occ, n_vir) (eigenvalues +-s of an ov-block Q)."""
    for act in (systems.hydrogen_chain(4, spacing=1.8, unit="bohr"), systems.lih()):
        rd = residual.compute_residual(act, FCISpace.of(act).hf())
        fac = factorize_generator(residual.descent_generator(rd.A), rank_tol=1e-8)
        no = act.nelec[0]
        na = [cost.leaf_active_orbitals(fac, k) for k in range(fac.n_leaves)]
        assert all(x % 2 == 0 and 0 < x <= 2 * min(no, act.norb - no) for x in na), na


# ================================================== generator_rank: leaves
def test_leaf_spectrum_is_pair_matrix_singular_values(h4_traj):
    """|c_k| are the singular values of M[(pq),(rs)] = gs[p,q,r,s] (all of them; the rest vanish)."""
    res, _ = h4_traj
    n = res.active.norb
    for step in res.steps[:4]:
        g = step.generator
        fac = factorize_generator(g)
        sv = np.linalg.svd(pair_symmetrize(g).reshape(n * n, n * n), compute_uv=False)
        K = fac.n_leaves
        assert np.allclose(np.sort(np.abs(fac.coeffs))[::-1], sv[:K], atol=1e-13)
        assert np.all(sv[K:] < 1e-12 * sv[0])


def test_leaf_profile_is_invariant_under_orbital_rotation(h4, h4_traj):
    res, states = h4_traj
    g = res.steps[2].generator
    ref = gr.leaf_profile(g)
    rotations = dict(gr.orbital_bases(h4, states[2]))
    rotations["random"] = systems.random_orthogonal(4, np.random.default_rng(5))
    assert set(rotations) == {"canonical", "natural", "localized-ov", "localized", "random"}
    for name, u in rotations.items():
        gu = gr.rotate_pair_tensor(g, u)
        assert np.isclose(np.sum(gu ** 2), np.sum(g ** 2))
        prof = gr.leaf_profile(gu)
        assert prof["rank"] == ref["rank"], name
        assert prof["n_leaves"] == ref["n_leaves"] and prof["n_leaves_complete"] == ref["n_leaves_complete"], name
        assert np.allclose(prof["singular_values"], ref["singular_values"], atol=1e-12), name
        assert prof["cnots"] == ref["cnots"], name
    # the inverse rotation recovers g
    u = rotations["random"]
    assert np.allclose(gr.rotate_pair_tensor(gr.rotate_pair_tensor(g, u), u.T), g, atol=1e-14)
    with pytest.raises(ValueError):
        gr.rotate_pair_tensor(g, 2 * np.eye(4))


def test_leaf_profile_fractions(h4_traj):
    res, _ = h4_traj
    prof = gr.leaf_profile(res.steps[1].generator)
    n = [prof["n_leaves"][k] for k in ("0.9", "0.99", "0.999")]
    m = [prof["n_leaves_complete"][k] for k in ("0.9", "0.99", "0.999")]
    assert n == sorted(n) and all(a <= b for a, b in zip(n, m)) and m[-1] <= prof["rank"] <= 12
    sv = np.array(prof["singular_values"])
    w = sv ** 2 / np.sum(sv ** 2)
    for f, k in zip((0.9, 0.99, 0.999), n):
        assert w[:k].sum() >= f - 1e-12 and (k == 0 or w[:k - 1].sum() < f)
    # real generators: complete groups are even (conjugate pairs)
    assert all(x % 2 == 0 for x in m)


# ================================================ generator_rank: elements
def test_element_projection_is_symmetric_orthogonal_and_captures_its_weight(h4_traj):
    res, _ = h4_traj
    step = res.steps[2]
    g, A = step.generator, step.residual
    reps, w = gr.element_weights(g)
    assert np.isclose(w.sum(), np.sum(g ** 2)) and np.all(np.diff(w) <= 0)
    for k in (1, 3, 7, len(reps)):
        gk = gr.project_elements(g, reps[:k])
        assert np.isrealobj(gk)
        assert np.allclose(gk, gk.transpose(2, 3, 0, 1)) and np.allclose(adjoint(gk), -gk)  # symmetries
        assert np.allclose(gr.project_elements(gk, reps[:k]), gk)  # idempotent
        assert abs(np.sum((g - gk) * gk)) < 1e-14  # orthogonal
        slope = -np.sum(gk * A)
        assert np.isclose(slope, -w[:k].sum(), rtol=1e-10)  # captured descent = kept weight
        # number of nonzero orbits kept
        rep = gr.symmetry_orbits(4)
        assert len(np.unique(rep[np.abs(gk.ravel()) > 0])) == k
    # all nonzero orbits (|g| > 1e-10 max|g|) = g up to the dropped rounding noise
    assert np.abs(gr.project_elements(g, reps) - g).max() <= 1e-10 * np.abs(g).max()


def test_element_counts_depend_on_basis(h4, h4_traj):
    res, states = h4_traj
    g = res.steps[2].generator
    bases = gr.orbital_bases(h4, states[2])
    bases["random"] = systems.random_orthogonal(4, np.random.default_rng(6))
    prof = {k: gr.element_profile(g, None if k == "canonical" else u) for k, u in bases.items()}
    assert len({prof[k]["n_unique"] for k in ("canonical", "localized", "random")}) == 3
    assert prof["canonical"]["n_unique"] < prof["localized"]["n_unique"] < prof["random"]["n_unique"]
    assert prof["canonical"]["n_elements"]["0.99"] < prof["localized"]["n_elements"]["0.99"]
    for k, p in prof.items():
        assert np.isclose(p["norm2"], np.sum(g ** 2)), k  # ||g||^2 is basis invariant
        ns = [p["n_elements"][f] for f in ("0.9", "0.99", "0.999")]
        assert ns == sorted(ns) and ns[-1] <= p["n_unique"], k
        assert p["n_excitations_total"] >= p["n_excitations"]["0.999"], k
        assert p["cnots_total"]["all"] >= p["cnots"]["all"]["0.999"], k


def test_orbital_bases(h4, h4_traj):
    res, states = h4_traj
    hf = gr.orbital_bases(h4, states[0])
    assert np.array_equal(hf["canonical"], np.eye(4))
    assert np.allclose(hf["natural"], np.eye(4), atol=1e-12)  # determinant: aligned to canonical
    u = gr.orbital_bases(h4, states[3])["natural"]
    dm1, _ = residual.rdm12(FCISpace.of(h4), states[3])
    d = u.T @ dm1 @ u
    assert np.allclose(u.T @ u, np.eye(4), atol=1e-12)
    assert np.abs(d - np.diag(np.diag(d))).max() < 1e-12 and np.all(np.diff(np.diag(d)) <= 1e-12)
    # Boys orbitals: orthogonal rotation of the active space with a smaller total spread
    loc = hf["localized"]
    assert np.allclose(loc.T @ loc, np.eye(4), atol=1e-12)
    mol = h4.mol
    r = mol.intor("int1e_r")
    rr = mol.intor("int1e_r2")

    def spread(c):
        return float(np.sum(np.einsum("pi,pq,qi->i", c, rr, c)
                            - np.sum(np.einsum("pi,xpq,qi->xi", c, r, c) ** 2, axis=0)))

    c = h4.mo_coeff
    assert spread(c @ loc) < 0.5 * spread(c)
    assert np.isclose(gr.boys_spread(mol, c @ loc), spread(c @ loc), rtol=1e-12)
    assert set(hf) == {"canonical", "natural", "localized-ov", "localized"}
    from thc_engine_cqe.integrals import ActiveSpace
    bare = ActiveSpace(h1=h4.h1, eri=h4.eri, ecore=h4.ecore, nelec=h4.nelec)
    assert set(gr.orbital_bases(bare, states[0])) == {"canonical", "natural"}
    assert gr.localized_orbitals(bare) is None


def _pair_rotation_gain(mol, c, blocks, n_theta=90):
    """Largest decrease of the Boys spread from rotating one pair of orbitals inside a block."""
    base = gr.boys_spread(mol, c)
    gain = 0.0
    for a, b in blocks:
        for i in range(a, b):
            for j in range(i + 1, b):
                for t in np.linspace(0.0, np.pi / 2, n_theta, endpoint=False)[1:]:
                    cc = c.copy()
                    cc[:, i] = np.cos(t) * c[:, i] + np.sin(t) * c[:, j]
                    cc[:, j] = -np.sin(t) * c[:, i] + np.cos(t) * c[:, j]
                    gain = max(gain, base - gr.boys_spread(mol, cc))
    return gain


def test_localized_bases_split_and_mixed(h4):
    """'localized-ov' keeps the HF blocks, 'localized' mixes them; both are pair-rotation minima.

    The canonical occupied pair of H4 has definite inversion parity, so both centroids vanish
    and it is the spread *maximum* among rotations of the pair (a stationary point). That is
    why one Boys run is not trusted and the lowest-spread result of several starts is kept.
    """
    mol, c, no = h4.mol, h4.mo_coeff, h4.nelec[0]
    bases = gr.orbital_bases(h4)
    ov, full = bases["localized-ov"], bases["localized"]
    assert gr.localization_blocks(h4, True) == [(0, 2), (2, 4)] and gr.localization_blocks(h4, False) == [(0, 4)]
    assert np.array_equal(ov[:no, no:], np.zeros((no, 4 - no))) and np.array_equal(ov[no:, :no], np.zeros((4 - no, no)))
    assert np.allclose(ov.T @ ov, np.eye(4), atol=1e-12) and np.allclose(full.T @ full, np.eye(4), atol=1e-12)
    canon = gr.boys_spread(mol, c)
    for u, split in ((ov, True), (full, False)):
        assert _pair_rotation_gain(mol, c @ u, gr.localization_blocks(h4, split)) < 1e-8 * canon
    assert gr.boys_spread(mol, c @ full) < gr.boys_spread(mol, c @ ov) < 0.6 * canon
    # the canonical occupied pair: spread maximal at theta = 0 among all rotations of the pair
    occ = c[:, :no]
    s0 = gr.boys_spread(mol, occ)
    for t in np.linspace(0.05, np.pi / 2 - 0.05, 12):
        rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        assert gr.boys_spread(mol, occ @ rot) < s0 - 1e-3
    assert gr.boys_spread(mol, (c @ ov)[:, :no]) < 0.75 * s0
    # first RHF generator: (ov|ov) + (vo|vo) support survives 'localized-ov', not 'localized'
    g = residual.descent_generator(residual.compute_residual(h4, FCISpace.of(h4).hf()).A)
    o, v = slice(0, no), slice(no, 4)
    mask = np.zeros(g.shape, dtype=bool)
    mask[o, v, o, v] = True
    mask[v, o, v, o] = True
    for name, u in (("canonical", np.eye(4)), ("localized-ov", ov), ("localized", full)):
        gu = gr.rotate_pair_tensor(g, u)
        rel = np.abs(gu[~mask]).max() / np.abs(gu).max()
        assert (rel < 1e-8) == (name != "localized"), (name, rel)
    prof = {k: gr.element_profile(g, u)["n_unique"] for k, u in (("canonical", None), ("localized-ov", ov),
                                                                  ("localized", full))}
    assert prof["canonical"] <= prof["localized-ov"] < prof["localized"], prof
    # deterministic: a second call returns the same rotation
    assert np.array_equal(gr.localized_orbitals(h4, split_ov=True), ov)


# ============================================ first step: the rank relation
def _coulomb_only_generator(act):
    no, n = act.nelec[0], act.norb
    o, v = slice(0, no), slice(no, n)
    g = np.zeros((n,) * 4)
    g[o, v, o, v] = 4 * act.eri[o, v, o, v]
    g[v, o, v, o] = -4 * act.eri[o, v, o, v].transpose(1, 0, 3, 2)
    return g


def test_first_step_rank_relation_h4(h4, h4_study):
    """K_1 = 2 rank(4V - 2X); for H4 r_ov = n_occ n_vir = 4, so K_1 = 8 = 2 r_ov."""
    rel = gr.first_step_rank_relation(h4)
    assert rel["max_dev"] < 1e-13 and rel["off_block"] < 1e-8
    assert rel["K1"] == 2 * rel["rank_W"] == h4_study[0]["leaf_profile"]["rank"] == 8
    assert rel["r_ov"] == rel["n_ov"] == 4 and rel["K1"] <= rel["bound"] == 8
    # the leaf magnitudes are the singular values of W, each twice
    c = np.array(h4_study[0]["leaf_profile"]["singular_values"])
    assert np.allclose(c[0::2], rel["singular_values_W"], atol=1e-13)
    assert np.allclose(c[1::2], rel["singular_values_W"], atol=1e-13)
    # the Coulomb part alone has exactly 2 r_ov leaves
    assert factorize_generator(_coulomb_only_generator(h4), rank_tol=1e-8).n_leaves == 2 * rel["r_ov"]


def test_first_step_rank_can_exceed_twice_the_ov_cholesky_rank():
    """N2/STO-3G CAS(6,6): (ia|jb) has Cholesky rank 8 < 9, but the exchange term restores rank 9."""
    n2 = systems.n2(1.098)
    rel = gr.first_step_rank_relation(n2)
    assert rel["max_dev"] < 1e-13 and rel["off_block"] < 1e-8
    assert rel["r_ov"] == 8 and rel["rank_W"] == 9 and rel["n_ov"] == 9
    assert rel["K1"] == 18 > 2 * rel["r_ov"]
    assert rel["K1"] <= rel["bound"]
    assert factorize_generator(_coulomb_only_generator(n2), rank_tol=1e-8).n_leaves == 2 * rel["r_ov"]


# ============================================================ rank study
def test_rank_study_h4(h4_study, h4_traj):
    res, _ = h4_traj
    assert [r["iteration"] for r in h4_study] == list(range(5))
    err = np.array([r["error"] for r in h4_study])
    assert np.all(err > 0) and np.all(np.diff(err) < 0)
    assert np.allclose(err, res.errors[:5], atol=1e-12)
    assert np.allclose([r["residual_norm"] for r in h4_study], [s.residual_norm for s in res.steps[:5]])
    for r in h4_study:
        lp, ep = r["leaf_profile"], r["element_profile"]
        assert set(ep) == {"canonical", "natural", "localized-ov", "localized"}
        assert lp["rank"] <= 12  # real generators: at most N(N-1) leaves
        assert np.isclose(lp["norm2"], r["residual_norm"] ** 2, rtol=1e-10)
        for p in ep.values():
            assert np.isclose(p["norm2"], r["residual_norm"] ** 2, rtol=1e-10)
        assert np.isclose(np.sum(r["natural_occupations"]), 4.0)
    assert h4_study[0]["element_profile"]["natural"] == h4_study[0]["element_profile"]["canonical"]
    assert h4_study[1]["leaf_profile"]["rank"] == 12
    # observed for H4 (canonical orbitals are symmetry adapted, hence sparse)
    assert h4_study[0]["element_profile"]["canonical"]["n_unique"] == 6
    assert h4_study[1]["element_profile"]["canonical"]["n_unique"] == 28


# ========================================================= compressed CQE
@pytest.fixture(scope="module")
def h4_ref(h4, h4_fci):
    return run_cqe(h4, max_iter=6, e_exact=h4_fci)


@pytest.mark.parametrize("scheme", ["elements", "leaves"])
def test_compressed_full_budget_reproduces_run_cqe(h4, h4_fci, h4_ref, scheme):
    """Keeping everything gives the exact CQE. The match is ~1e-9, not 1e-15: dropping the
    1e-12 Brillouin noise (or any 1e-13 relative change of g) moves the flat line-search
    minimum by ~1e-7 in eps, which shifts later energies by up to ~1e-9."""
    out = gr.compressed_cqe(h4, scheme, 10 ** 6, max_iter=6, e_fci=h4_fci)
    assert np.abs(out.energies - h4_ref.energies).max() < 1e-8
    assert all(abs(r["captured"] - 1) < 1e-12 for r in out.records)
    if scheme == "elements":  # with no truncation at all the trajectory is identical
        exact = gr.compressed_cqe(h4, scheme, 10 ** 6, max_iter=6, e_fci=h4_fci, element_tol=1e-300)
        assert np.abs(exact.energies - h4_ref.energies).max() < 1e-12
    else:
        assert [r["kept"] for r in out.records] == [8] + [12] * 5


@pytest.fixture(scope="module")
def h4_compressed(h4, h4_fci):
    runs = {}
    for scheme, per in (("elements", 3), ("leaves", 2), ("leaves-trotter", 2), ("leaves-trotter", 10 ** 6)):
        runs[(scheme, per)] = gr.compressed_cqe(h4, scheme, per, max_iter=6, e_fci=h4_fci)
    return runs


@pytest.mark.parametrize("key", [("elements", 3), ("leaves", 2), ("leaves-trotter", 2),
                                 ("leaves-trotter", 10 ** 6)], ids=lambda k: f"{k[0]}-{k[1]}")
def test_compressed_energies_decrease_monotonically(h4_compressed, h4, key):
    out = h4_compressed[key]
    e = out.energies
    assert len(e) == 7 and np.all(np.diff(e) <= 1e-12)
    assert e[-1] < e[0] - 0.03  # it really descends (E_HF - E_FCI = 0.062)
    for conn in cost.CONNECTIVITIES:
        cum = out.cum_cnots(conn)
        assert np.all(np.diff(cum) >= 0) and cum[-1] > 0
    for r in out.records:
        assert 0 < r["captured"] <= 1 + 1e-12 and r["slope"] < 0
        assert np.isclose(r["captured"], r["captured_run"], rtol=1e-8)
        if key[0] == "elements":
            assert r["kept"] == min(out.per_iter, r["available"])
        else:  # rounded up to a complete group of degenerate |c|
            assert min(out.per_iter, r["available"]) <= r["kept"] <= r["available"]
    if key[0] == "leaves-trotter":  # the product formula leaves the real manifold
        assert np.iscomplexobj(out.ci) and np.abs(out.ci.imag).max() > 1e-8


def test_compressed_cost_bookkeeping(h4, h4_compressed):
    """step and cumulative CNOTs follow cost.py: leaf rotations merge across steps."""
    rot = cost.orbital_rotation_cnots(4)
    rd = residual.compute_residual(h4, FCISpace.of(h4).hf())
    g0 = residual.descent_generator(rd.A)
    fac0 = factorize_generator(g0, rank_tol=1e-8)
    for scheme in ("leaves", "leaves-trotter"):
        out = h4_compressed[(scheme, 2)]
        r0 = out.records[0]
        assert r0["kept"] == 2 and r0["step_cnots"]["all"] == cost.cnot_cost_leaves(fac0, 2)
        steps = np.array([r["step_cnots"]["all"] for r in out.records if r["eps"] > 0])
        assert out.cum_cnots("all")[-1] == np.sum(steps - rot) + rot  # one final un-rotation
    out = h4_compressed[("elements", 3)]
    reps, _ = gr.element_weights(g0)
    exc = cost.spin_orbital_excitations(gr.project_elements(g0, reps[:3]))
    assert out.records[0]["step_cnots"]["linear"] == cost.cnot_cost_elementwise(exc, 4, "linear")
    assert out.cum_cnots("all")[-1] == sum(r["step_cnots"]["all"] for r in out.records)
    # same generator at iteration 0, so the Trotterized and exact leaf schemes agree there
    a, b = h4_compressed[("leaves", 2)].records[0], h4_compressed[("leaves-trotter", 2)].records[0]
    assert a["captured"] == pytest.approx(b["captured"]) and a["step_cnots"] == b["step_cnots"]
    assert abs(b["energy_after"] - a["energy_after"]) < 1e-3  # first step: small Trotter error


def test_compressed_trotter_order_two_charges_the_strang_circuit(h4, h4_fci, h4_compressed):
    """Regression: trotter_order=2 used to be charged the first-order CNOT count.

    Strang step with L leaves: 2L C_rot + 2 sum_{k<L-1} C_diag(k) + C_diag(L-1). For the
    first H4 step with L = 2 (C_rot = 24, C_diag = 56) that is 264 against 184 for order 1.
    """
    rot = cost.orbital_rotation_cnots(4)
    rd = residual.compute_residual(h4, FCISpace.of(h4).hf())
    fac0 = factorize_generator(residual.descent_generator(rd.A), rank_tol=1e-8)
    diag = [cost.diagonal_layer_cnots(4, cost.leaf_active_orbitals(fac0, k)) for k in range(2)]
    assert diag == [56, 56]
    o1 = h4_compressed[("leaves-trotter", 2)]
    o2 = gr.compressed_cqe(h4, "leaves-trotter", 2, max_iter=3, e_fci=h4_fci, trotter_order=2)
    assert o1.trotter_order == 1 and o2.trotter_order == 2 and o2.to_dict()["trotter_order"] == 2
    assert o1.records[0]["step_cnots"]["all"] == 3 * rot + diag[0] + diag[1] == 184
    assert o2.records[0]["step_cnots"]["all"] == 4 * rot + 2 * diag[0] + diag[1] == 264
    assert o2.records[0]["step_cnots"]["linear"] == cost.cnot_cost_leaves(fac0, 2, "linear", order=2)
    steps = np.array([r["step_cnots"]["all"] for r in o2.records if r["eps"] > 0])
    assert o2.cum_cnots("all")[-1] == np.sum(steps - rot) + rot  # rotations still merge across steps
    assert np.all(np.diff(o2.energies) <= 1e-12) and o2.energies[-1] < o2.energies[0] - 0.03
    # the second-order circuit really is different: smaller Trotter error on the first step
    ex = h4_compressed[("leaves", 2)]
    err1 = abs(o1.records[0]["energy_after"] - ex.records[0]["energy_after"])
    err2 = abs(o2.records[0]["energy_after"] - ex.records[0]["energy_after"])
    assert err2 < err1
    # 'leaves' (exact exponential) charges the circuit of the same order; its energies do not change
    ex2 = gr.compressed_cqe(h4, "leaves", 2, max_iter=6, e_fci=h4_fci, trotter_order=2)
    assert np.allclose(ex2.energies, ex.energies, atol=1e-14, rtol=0)
    assert ex2.records[0]["step_cnots"]["all"] == 264 and ex.records[0]["step_cnots"]["all"] == 184
    # the elementwise model is first order only; other orders are rejected
    with pytest.raises(ValueError, match="first-order"):
        gr.compressed_cqe(h4, "elements", 2, trotter_order=2)
    with pytest.raises(ValueError, match="trotter_order"):
        gr.compressed_cqe(h4, "leaves-trotter", 2, trotter_order=3)


def test_elements_in_rotated_basis(h4, h4_fci, h4_ref):
    u = systems.random_orthogonal(4, np.random.default_rng(8))
    full = gr.compressed_cqe(h4, "elements", 10 ** 6, max_iter=6, e_fci=h4_fci, basis=u,
                             element_tol=1e-300)
    assert np.abs(full.energies - h4_ref.energies).max() < 1e-8  # P_u with everything kept = identity
    assert full.cum_cnots("all")[-1] == sum(r["step_cnots"]["all"] for r in full.records) + 2 * 24
    small = gr.compressed_cqe(h4, "elements", 4, max_iter=4, e_fci=h4_fci, basis=u, basis_label="random")
    assert small.basis == "random" and np.all(np.diff(small.energies) <= 1e-12)
    assert all(r["kept"] == 4 for r in small.records)
    with pytest.raises(ValueError):
        gr.compressed_cqe(h4, "leaves", 2, basis=u)
    with pytest.raises(ValueError):
        gr.compressed_cqe(h4, "nope", 2)


def test_odd_leaf_truncation_runs_on_a_complex_start(h4, h4_fci):
    """complete_groups=False keeps one leaf of a conjugate pair, so the generator is complex.
    compressed_cqe then starts run_cqe from a complex HF vector (the Krylov dtype workaround):
    the reported energies are the energies of the returned complex state."""
    from thc_engine_cqe.fci_ops import Hamiltonian

    out = gr.compressed_cqe(h4, "leaves", 1, max_iter=3, e_fci=h4_fci, complete_groups=False)
    assert [r["kept"] for r in out.records] == [1, 1, 1]
    assert np.all(np.diff(out.energies) <= 1e-12) and out.energies[-1] < out.energies[0] - 1e-3
    assert np.iscomplexobj(out.ci) and np.abs(out.ci.imag).max() > 1e-3
    assert np.isclose(Hamiltonian(h4).energy(out.ci), out.records[-1]["energy_after"], atol=1e-10)


def _rotation_on_ci(space, w, psi):
    """U(W) psi, with U(W) a+_q U(W)^dag = sum_p W[p,q] a+_p, i.e. U(W) = exp(sum kappa E), W = exp(kappa)."""
    kappa = scipy.linalg.logm(w)
    kappa = 0.5 * (kappa - kappa.conj().T)
    return fci_ops.expm_apply(lambda x: fci_ops.one_body(space, kappa, x), psi, 1.0)


def _merged_leaf_circuit(fac, n_leaves, order, eps):
    """The compiled step of cost.py: [('rot', W) | ('diag', k, mu, t)] in application order.

    Unmerged: exp(t Ohat_F) = U(exp(t F)) (t = eps for order 1, eps/2 at both ends for
    order 2) and, for each (k, frac) of cost.leaf_schedule, U(V_k)^dag, exp(i frac eps c_k D_k^2),
    U(V_k). Applying U(A) and then U(B) is the single rotation U(B A).
    """
    f = fac.one_body_for(n_leaves)
    one = 1.0 if order == 1 else 0.5
    seq = [("rot", scipy.linalg.expm(one * eps * f))]
    for k, frac in cost.leaf_schedule(n_leaves, order):
        mu, v = fac.leaf_eigh(k)
        seq += [("rot", v.conj().T), ("diag", k, mu, frac * eps), ("rot", v)]
    if order == 2:
        seq.append(("rot", scipy.linalg.expm(0.5 * eps * f)))
    merged = []
    for item in seq:
        if item[0] == "rot" and merged and merged[-1][0] == "rot":
            merged[-1] = ("rot", item[1] @ merged[-1][1])
        else:
            merged.append(item)
    return merged


@pytest.mark.parametrize("order", [1, 2])
def test_leaf_circuit_layer_counts_reproduce_the_product_formula(h4_traj, order):
    """The merged circuit that cost.cnot_cost_leaves counts equals TrotterStep, for both orders.

    Order 1 has L + 1 rotations (incl. the final un-rotation) and L diagonal layers; order 2
    (Strang) has 2L rotations and 2L - 1 layers. The CNOT estimate is exactly the sum of the
    per-layer costs of this circuit (regression: order 2 used to be charged the order-1 count).
    """
    res, states = h4_traj
    space = FCISpace.of(res.active)
    n = space.norb
    fac = factorize_generator(res.steps[2].generator, rank_tol=1e-8)
    psi_c = space.random(np.random.default_rng(12), complex_=True)
    eps = 0.3
    for L in (1, 2, 5):
        circ = _merged_leaf_circuit(fac, L, order, eps)
        kinds = [c[0] for c in circ]
        n_layers = L if order == 1 else 2 * L - 1
        assert kinds == ["rot", "diag"] * n_layers + ["rot"], (L, kinds)  # alternating
        step = gr.TrotterStep(space, fac, L, order=order)
        for psi in (states[2], psi_c):
            out = np.asarray(psi, dtype=complex)
            for c in circ:
                if c[0] == "rot":
                    out = _rotation_on_ci(space, c[1], out)
                else:
                    _, k, mu, t = c
                    d = np.diag(mu)
                    out = fci_ops.expm_apply(lambda x, d=d, k=k, t=t: (1j * t * fac.coeffs[k]) * fci_ops.one_body(
                        space, d, fci_ops.one_body(space, d, x)), out, 1.0)
            assert np.abs(out - step.apply(eps, psi)).max() < 1e-10, (L, order)
        for conn in cost.CONNECTIVITIES:
            rot = cost.orbital_rotation_cnots(n, conn)
            layers = sum(cost.diagonal_layer_cnots(n, cost.leaf_active_orbitals(fac, c[1]), conn)
                         for c in circ if c[0] == "diag")
            assert cost.cnot_cost_leaves(fac, L, conn, order=order) == kinds.count("rot") * rot + layers
            assert cost.cnot_cost_leaves(fac, L, conn, order=order, final_rotation=False) == \
                (kinds.count("rot") - 1) * rot + layers


def test_trotter_step_matches_generator_factorization(h4_traj):
    """The fast product-formula emulation equals GeneratorFactorization.trotter_apply."""
    res, states = h4_traj
    space = FCISpace.of(res.active)
    fac = factorize_generator(res.steps[2].generator)
    psi_c = space.random(np.random.default_rng(3), complex_=True)
    for order in (1, 2):
        for n_leaves in (None, 3):
            step = gr.TrotterStep(space, fac, n_leaves, order=order)
            for psi in (states[2], psi_c):
                for eps in (0.07, 0.4):
                    ref = fac.trotter_apply(space, eps, psi, order=order, n_leaves=n_leaves)
                    assert np.abs(step.apply(eps, psi) - ref).max() < 1e-11
    with pytest.raises(ValueError):
        gr.TrotterStep(space, fac, 2, order=3)


# ============================================================== benchmark
def _load_benchmark():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "benchmarks" / "run_generator_rank.py"
    spec = importlib.util.spec_from_file_location("run_generator_rank", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_benchmark_smoke_and_headline(tmp_path):
    """benchmarks/run_generator_rank.py on H4 with tiny budgets: files, bases and the computed headline."""
    import json

    bench = _load_benchmark()
    out = tmp_path / "res"
    assert bench.main(["--systems", "h4", "--max-iter", "6", "--cqe-iter", "3", "--elements-per", "2",
                       "--leaves-per", "2", "--out", str(out)]) == 0
    res = json.loads((out / "generator_rank_h4.json").read_text())
    names = [bench._run_name(r) for r in res["runs"]]
    assert names == ["elements[canonical]", "elements[localized-ov]", "elements[localized]", "leaves",
                     "leaves-trotter"]
    assert all(r["trotter_order"] == 1 for r in res["runs"])
    assert set(res["study"][0]["element_profile"]) == {"canonical", "natural", "localized-ov", "localized"}
    # the headline medians are recomputed from the study, not typed in
    head = bench.headline(res)
    it = [r for r in res["study"] if 1 <= r["iteration"] <= 5]
    assert head["one_step"]["leaves"] == np.median([r["leaf_profile"]["cnots"]["all"]["0.99"] for r in it])
    assert head["one_step"]["elements[localized-ov]"] == np.median(
        [r["element_profile"]["localized-ov"]["cnots"]["all"]["0.99"] for r in it])
    assert "elements[natural]" not in head["one_step"]
    ratio = head["one_step"]["leaves"] / head["one_step"]["elements[canonical]"]
    summary = (out / "generator_rank_summary.md").read_text()
    assert "## Headline: leaves vs elements" in summary and f"| {ratio:.2f} |" in summary
    md = (out / "generator_rank_h4.md").read_text()
    assert "elements[localized-ov]" in md and "mixes occupied and virtual" in md
