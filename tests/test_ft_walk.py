"""Tests of ft.walk: the explicit Pauli-LCU qubitized walk, the 2R-call Laurent evolution, the QPE outcome law.

References: the exact spectrum of the dense (n_alpha, n_beta) CI Hamiltonian, the
Jordan-Wigner Fock-space oracle (``testing.fock``) for Tr H, ``factor.pauli_lambda`` for the
Pauli 1-norm, dense matrix exponentials, direct DFT sums for the Fejer kernel, and the QPE
emulation of the design notes [design] for the outcome-law constant.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg
import scipy.special

from thc_engine_cqe import systems
from thc_engine_cqe.cqe import run_cqe
from thc_engine_cqe.factor import pauli_lambda
from thc_engine_cqe.factor.df import double_factorize
from thc_engine_cqe.fci_ops import Hamiltonian, normalize
from thc_engine_cqe.ft import primitives, symmetry, walk
from thc_engine_cqe.measure.pauli import QubitState
from thc_engine_cqe.testing.fock import FockSpace

_CACHE: dict = {}


def _setup(name: str):
    """(active, walk, sector energies E_j, eigenvectors, dense sector H with ecore, [HF, L-BFGS iterate 3])."""
    if name not in _CACHE:
        a = systems.hydrogen_chain(2, 0.74, "angstrom") if name == "h2" else symmetry.benchmark_system(name)
        ham = Hamiltonian(a)
        sp = ham.space
        dense = np.column_stack([ham.apply(np.eye(sp.size)[i].reshape(sp.shape)).ravel() for i in range(sp.size)])
        dense = 0.5 * (dense + dense.T) + a.ecore * np.eye(sp.size)
        e, v = np.linalg.eigh(dense)
        cqe3 = normalize(np.asarray(run_cqe(a, max_iter=3, optimizer="lbfgs").ci))
        _CACHE[name] = (a, walk.PauliLCUWalk(a), e, v, dense, [normalize(sp.hf()), cqe3])
    return _CACHE[name]


def _qubit(a, w, ci) -> np.ndarray:
    return QubitState.from_ci(a.norb, a.nelec, ci, w.mapping).to_dense()


@pytest.mark.parametrize("name", ["h2", "h4"])
def test_lambda_and_identity_coefficient(name):
    """lambda is the Pauli 1-norm (factor.pauli_lambda) and c0 = Tr H / 2^{2N} (Fock-space trace)."""
    a, w, *_ = _setup(name)
    fs = FockSpace(a.norb)
    trace = float(fs.hamiltonian(a.h1, a.eri, a.ecore).diagonal().sum())
    assert abs(w.c0 - trace / fs.dim) < 1e-12
    assert abs(w.lam - pauli_lambda(a.h1, a.eri).lam_total) < 1e-12
    assert w.n_qubits == 2 * a.norb


@pytest.mark.parametrize("name", ["h2", "h4"])
def test_moments_are_chebyshev_moments_of_the_spectrum(name):
    """<G,psi|W^n|G,psi> = sum_j p_j T_n((E_j - c0)/lambda) to 1e-12 for n <= 30, on HF and an L-BFGS iterate.

    The right side uses the exact sector spectrum of the dense CI Hamiltonian; the identity
    holds iff W rotates every plane span{|G,psi_j>, W|G,psi_j>} by arccos((E_j - c0)/lambda).
    """
    a, w, e, v, _, states = _setup(name)
    x = (e - w.c0) / w.lam
    assert np.max(np.abs(x)) <= 1.0
    n = np.arange(31)
    for ci in states:
        p = (v.T @ ci.ravel()) ** 2
        ref = np.cos(n[:, None] * np.arccos(x)[None, :]) @ p
        mom = w.moments(_qubit(a, w, ci), 30)
        assert np.max(np.abs(mom - ref)) < 1e-12


def test_walk_is_unitary_and_block_encodes_h():
    """W^dagger W = 1 on random walk states, and (<G| x 1) W (|G> x |psi>) = (H - c0)|psi> / lambda.

    <G|R = <G|, so projecting W = R SELECT on <G| gives <G|SELECT|G>, the block encoding of
    (H - c0)/lambda; the reference applies the dense CI Hamiltonian and maps to qubits.
    """
    a, w, _, _, dense, states = _setup("h4")
    rng = np.random.default_rng(2)
    shape = (w.n_terms, 1 << w.n_qubits)
    phi = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    assert np.linalg.norm(w.apply_dagger(w.apply(phi)) - phi) < 1e-12 * np.linalg.norm(phi)
    assert abs(np.linalg.norm(w.apply(phi)) - np.linalg.norm(phi)) < 1e-12 * np.linalg.norm(phi)
    gvec = w.prepare(np.eye(1 << w.n_qubits)[0])[:, 0]          # |G> = sum_l sqrt(|c_l|/lambda)|l>
    assert abs(np.linalg.norm(gvec) - 1.0) < 1e-14
    for ci in states:
        projected = gvec @ w.apply(w.prepare(_qubit(a, w, ci)))
        ref = _qubit(a, w, ((dense @ ci.ravel() - w.c0 * ci.ravel()) / w.lam).reshape(ci.shape))
        assert np.linalg.norm(projected - ref) < 1e-12
    with pytest.raises(ValueError):
        w.apply(phi[:, :4])


@pytest.mark.parametrize("tau", [0.5, 3.0, 7.0])
def test_laurent_evolution_within_the_jacobi_anger_tail(tau):
    """sum_{|k|<=R} (-i)^k J_k(tau) W^k |G,psi> = |G> x exp(-i tau (H - c0)/lambda)|psi> within the tail bound.

    The reference propagates with a dense expm in the CI sector; the bound 2 sum_{k>R} |J_k(tau)|
    is summed here with scipy's Bessel functions, R being F8's degree at delta = 1e-4 and 1e-8.
    The symmetric sum W^k + W^-k acts as 2 cos(k theta) on each plane, so nothing leaves |G>:
    the leakage is round-off (<= 1e-12), far inside the bound.
    """
    a, w, _, _, dense, states = _setup("h4")
    ci = states[1]
    psi = _qubit(a, w, ci)
    exact_ci = scipy.linalg.expm(-1j * tau * (dense - w.c0 * np.eye(dense.shape[0])) / w.lam) @ ci.ravel()
    ref = w.prepare(_qubit(a, w, exact_ci.reshape(ci.shape)))
    gvec = w.prepare(np.eye(1 << w.n_qubits)[0])[:, 0]
    for delta in (1e-4, 1e-8):
        R = primitives.jacobi_anger_degree(tau, delta)
        tail = 2.0 * float(np.sum(np.abs(scipy.special.jv(np.arange(R + 1, R + 200), tau))))
        assert tail <= delta
        out = w.laurent_evolution(psi, tau, delta)
        assert np.linalg.norm(out - ref) <= tail
        leak = np.linalg.norm(out - np.outer(gvec, gvec @ out))
        assert leak <= 1e-12


def test_fejer_kernel_against_direct_dft_sums():
    """One eigenphase's law is |(1/M) sum_k exp(2 pi i k (theta/2pi - m/M))|^2, averaged over +-theta.

    Direct DFT sums (M = 16, 64) for random phases, including phases on the grid; the law of
    several eigenstates is the weighted sum and sums to sum_j p_j.
    """
    rng = np.random.default_rng(4)
    for t in (4, 6):
        M = 1 << t
        m = np.arange(M)
        k = np.arange(M)
        for x in list(rng.uniform(-1, 1, 6)) + [np.cos(2 * np.pi * 3 / M), 1.0, -1.0]:
            theta = np.arccos(x)
            ref = 0.0
            for th in (theta, -theta):
                amp = np.exp(2j * np.pi * np.outer(m, -k) / M) @ np.exp(1j * th * k) / M
                ref = ref + 0.5 * np.abs(amp) ** 2
            p, e = walk.qpe_outcome_law([0.3 + 2.0 * x], [1.0], 2.0, 0.3, t)
            assert np.max(np.abs(p - ref)) < 1e-13
            np.testing.assert_allclose(e, 0.3 + 2.0 * np.cos(2 * np.pi * m / M), atol=1e-14)
    energies, weights = rng.uniform(-1.5, 1.5, 5), rng.uniform(0, 1, 5)
    p, _ = walk.qpe_outcome_law(energies, weights, 1.6, -0.1, 10)
    assert abs(p.sum() - weights.sum()) < 1e-12


def test_outcome_law_argument_checks():
    """Invalid inputs raise ValueError instead of returning a law that is not a probability law.

    |E - c| > lambda, mismatched lengths, negative or nan weights (a weight -1 used to give a
    law summing to -1), non-finite energies, lambda or c, t_bits outside 1..24 (t = 30
    allocated 2^30-entry arrays) or non-integral, and a negative window.
    """
    for args in [([2.0], [1.0], 1.0, 0.0, 8), ([0.1, 0.2], [1.0], 1.0, 0.0, 8), ([0.1], [-1.0], 1.0, 0.0, 4),
                 ([0.1], [np.nan], 1.0, 0.0, 4), ([np.inf], [1.0], 1.0, 0.0, 4), ([0.1], [1.0], np.inf, 0.0, 4),
                 ([0.1], [1.0], 1.0, np.nan, 4), ([0.1], [1.0], 1.0, 0.0, 0), ([0.1], [1.0], 1.0, 0.0, 25),
                 ([0.1], [1.0], 1.0, 0.0, 15.5)]:
        with pytest.raises(ValueError):
            walk.qpe_outcome_law(*args)
    law = walk.qpe_outcome_law([0.1], [1.0], 1.0, 0.0, 6)
    assert walk.window_probability(law, 0.1, 0.0) <= walk.window_probability(law, 0.1, 0.05) <= 1.0
    with pytest.raises(ValueError):
        walk.window_probability(law, 0.1, -1e-3)


def test_window_probability_over_p_is_k_independent():
    """H4, DF H' (threshold 1e-3), lambda_DF, c = Tr H'/2^{2N}, t = 15: P(|E(m) - E0'| <= 1 mHa)/p_k is constant.

    The setup is that of the design notes [design]; the ratio reproduces their window factor w = 0.98619336
    and is k-independent to 1e-8 over the L-BFGS iterates k = 0..6 (design notes: spread 9.96e-9). The
    residual k-dependence is exactly the excited-state tails: ratio_k - w =
    sum_{j>=1} p_kj tau_j / p_k0 with w, tau_j the window probabilities of single
    eigenstates (to 1e-12).
    """
    a = symmetry.benchmark_system("h4")
    df = double_factorize(a.eri, first="eigh", tol=0.0, eig_threshold=1e-3, threshold_rule="openfermion")
    lam = df.lam(a.h1, eri_exact=a.eri).lam_total
    model = a.with_eri(df.to_eri())
    c = walk.PauliLCUWalk(model).c0
    ham = Hamiltonian(model)
    sp = ham.space
    dense = np.column_stack([ham.apply(np.eye(sp.size)[i].reshape(sp.shape)).ravel() for i in range(sp.size)])
    e, v = np.linalg.eigh(0.5 * (dense + dense.T))
    e = e + model.ecore
    states = [normalize(sp.hf())]
    run_cqe(a, max_iter=6, optimizer="lbfgs", callback=lambda st, ci: states.append(normalize(np.asarray(ci))))
    t = 15

    def single(j):
        return walk.window_probability(walk.qpe_outcome_law(e[j:j + 1], [1.0], lam, c, t), e[0], 1e-3)

    kappa = single(0)
    tails = np.array([single(j) for j in range(1, e.size)])
    ratios = []
    for psi in states:
        p = (v.T @ psi.ravel()) ** 2
        law = walk.qpe_outcome_law(e, p, lam, c, t)
        r = walk.window_probability(law, e[0], 1e-3) / p[0]
        assert abs(r - kappa - np.sum(p[1:] * tails) / p[0]) < 1e-12
        ratios.append(r)
    ratios = np.array(ratios)
    assert abs(ratios[0] - 0.9861933585) < 1e-9
    assert np.ptp(ratios) <= 1e-8
