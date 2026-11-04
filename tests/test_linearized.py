"""Linearized CQE at an eigenstate (diagnostics.linearized)."""

import numpy as np
import pytest

from thc_engine_cqe import systems
from thc_engine_cqe.cqe import run_cqe
from thc_engine_cqe.diagnostics.linearized import linearize, structured_basis
from thc_engine_cqe.fci_ops import FCISpace, Hamiltonian
from thc_engine_cqe.residual import compute_residual


@pytest.fixture(scope="module", autouse=True)
def _single_openmp_thread():
    from pyscf import lib

    old = lib.num_threads()
    lib.num_threads(1)
    yield
    lib.num_threads(old)


@pytest.fixture(scope="module")
def h4():
    return systems.hydrogen_chain(4, 1.8, "bohr")


@pytest.fixture(scope="module")
def lin_h4(h4):
    return linearize(h4)


@pytest.fixture(scope="module")
def n2():
    return systems.n2(1.098)


@pytest.fixture(scope="module")
def lin_n2(n2):
    return linearize(n2)


def _state(lin, d, t):
    psi = lin.psi0.ravel() + t * (d @ lin.states)
    return (psi / np.linalg.norm(psi)).reshape(lin.psi0.shape)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_structured_basis(n):
    u = structured_basis(n)
    assert u.shape == (n**4, n * n * (n * n - 1) // 4)
    assert np.allclose(u.T @ u, np.eye(u.shape[1]), atol=1e-14)
    x = u.reshape(n, n, n, n, -1)
    assert np.allclose(x, x.transpose(2, 3, 0, 1, 4), atol=1e-15)
    assert np.allclose(x, -x.transpose(1, 0, 3, 2, 4), atol=1e-15)


def test_residual_lies_in_the_structured_subspace(h4):
    a = compute_residual(h4, run_cqe(h4, max_iter=2).ci).A.ravel()
    u = structured_basis(h4.norb)
    assert np.linalg.norm(a - u @ (u.T @ a)) < 1e-13 * np.linalg.norm(a)


def test_spectrum_h4(h4, lin_h4):
    """K is symmetric positive semidefinite; H4 at 1.8 bohr has no null mode
    (observed: lambda from 0.054 to 64.9 over the 19 coupled singlets)."""
    e_fci, _ = h4.fci()
    assert abs(lin_h4.e0 - e_fci) < 1e-10
    assert lin_h4.n_states == 19 and np.all(lin_h4.gaps > 0)
    k = lin_h4.kmatrix()
    assert np.allclose(k, k.T, atol=1e-12)
    assert np.allclose(lin_h4.modes @ np.diag(lin_h4.stiffness) @ lin_h4.modes.T, k, atol=1e-10)
    assert not lin_h4.null_modes().any() and lin_h4.stiffness[0] > 1e-2


def test_first_order_residual(h4, lin_h4):
    """compute_residual(Psi_0 + t d) = t * sum_k d_k w_k M_k + O(t^2)."""
    rng = np.random.default_rng(3)
    d = rng.standard_normal(lin_h4.n_states)
    d /= np.linalg.norm(d)
    errs, rel = [], []
    for t in (2e-3, 1e-3):
        a = compute_residual(h4, _state(lin_h4, d, t)).A
        lin = lin_h4.residual_first_order(t * d)
        errs.append(np.linalg.norm(a - lin))
        rel.append(errs[-1] / np.linalg.norm(lin))
    assert rel[1] < 1e-2
    assert 3.5 < errs[0] / errs[1] < 4.5          # the remainder is second order in t


def test_sd_amplitude_map(h4, lin_h4):
    """One fixed-step SD step maps d to (I - (eps/2) S W) d + O(t^2)."""
    rng = np.random.default_rng(5)
    d = rng.standard_normal(lin_h4.n_states)
    d /= np.linalg.norm(d)
    eps = 0.03
    errs, rel = [], []
    for t in (2e-3, 1e-3):
        res = run_cqe(h4, ci0=_state(lin_h4, d, t), max_iter=1, eps_candidates=(eps,),
                      r_tol=0.0, e_tol=0.0)
        _, d_new = lin_h4.amplitudes(res.ci)
        pred = lin_h4.sd_amplitude_map(eps) @ (t * d)
        errs.append(np.linalg.norm(d_new - pred))
        rel.append(errs[-1] / np.linalg.norm(pred))
    assert rel[1] < 1e-2
    assert 3.5 < errs[0] / errs[1] < 4.5          # the remainder is second order in t


def test_n2_null_modes(n2, lin_n2):
    """N2/STO-3G CAS(6,6) at 1.098 A: B has 5 null vectors among the 174 coupled singlets
    and 3 in the sector the exact CQE from HF populates. Along a null mode ||A|| grows as
    t^2, along the stiffest mode as t; the exact CQE's remaining error sits in null modes
    (observed: 98-99% after 60 L-BFGS steps, E - E_0 = 1.8e-4 Ha)."""
    assert lin_n2.n_states == 174
    assert int(lin_n2.null_modes().sum()) == 5
    traj = []
    res = run_cqe(n2, max_iter=60, optimizer="lbfgs", r_tol=0.0, e_tol=0.0,
                  callback=lambda step, ci: traj.append(np.asarray(ci).copy()))
    for tol in (1e-20, 1e-12):
        sec = linearize(n2, sector="references", references=traj, weight_tol=tol)
        assert int(sec.null_modes().sum()) == 3
    null = lin_n2.null_modes()
    shares = lin_n2.mode_energies(res.ci)
    err = Hamiltonian(n2).energy(res.ci) - lin_n2.e0
    assert 1e-4 < err < 3e-4
    assert shares["per_mode"][null].sum() > 0.9 * err
    for m, ratio in ((int(np.flatnonzero(null)[0]), 4.0), (lin_n2.n_states - 1, 2.0)):
        d = lin_n2.modes[:, m] / np.sqrt(lin_n2.gaps)
        d /= np.linalg.norm(d)
        r = [compute_residual(n2, _state(lin_n2, d, t)).residual_norm for t in (1e-2, 5e-3)]
        assert abs(r[0] / r[1] - ratio) < 0.05


def test_noise_floor_matches_the_linear_recursion(lin_h4):
    """tr C of the Lyapunov solution equals the long-time mean of the simulated linear map."""
    rng = np.random.default_rng(11)
    eps, sigma = 0.03, 1e-3
    f = sigma * structured_basis(lin_h4.active.norb)
    pred = lin_h4.noise_floor(eps, f)
    assert pred["stable"] and not pred["undriven"].any()
    lam = lin_h4.stiffness
    # isotropic noise: equipartition, (eps sigma^2/4) / (1 - eps lambda/4) per mode
    assert np.allclose(pred["per_mode"], (eps * sigma**2 / 4) / (1 - eps * lam / 4), rtol=1e-8)
    pm = 1 - 0.5 * eps * lam
    r = lin_h4.modes.T @ (np.sqrt(lin_h4.gaps)[:, None] * (lin_h4.coupling @ f))
    y = np.zeros(lin_h4.n_states)
    acc = np.zeros_like(y)
    fast = pm < 0.98
    n_burn, n_steps = 500, 40000
    for t in range(n_burn + n_steps):
        y = pm * y - 0.5 * eps * (r @ rng.standard_normal(r.shape[1]))
        if t >= n_burn:
            acc += y**2
    mean = acc / n_steps
    assert abs(mean[fast].sum() / pred["per_mode"][fast].sum() - 1) < 0.05


def test_noise_floor_predicts_the_noisy_cqe(h4, lin_h4):
    """The real (nonlinear) CQE with fixed-step SD and isotropic Gaussian residual noise:
    the time-averaged energy in the fast modes matches the Lyapunov prediction."""
    eps, sigma = 0.03, 3e-3
    u = structured_basis(h4.norb)
    n = h4.norb
    rng = np.random.default_rng(7)

    def noisy(rd, ci, it):
        return rd.A + (sigma * (u @ rng.standard_normal(u.shape[1]))).reshape((n,) * 4)

    pred = lin_h4.noise_floor(eps, sigma * u)
    pm = 1 - 0.5 * eps * lin_h4.stiffness
    fast = pm < 0.98          # relaxation within ~25 steps
    acc = np.zeros(lin_h4.n_states)
    count = [0]
    n_burn = 200

    def cb(step, ci):
        if step.iteration >= n_burn:
            acc[:] += lin_h4.mode_energies(ci)["per_mode"]
            count[0] += 1

    run_cqe(h4, ci0=lin_h4.psi0, max_iter=n_burn + 1500, eps_candidates=(eps,), r_tol=0.0,
            e_tol=0.0, residual_estimator=noisy, callback=cb)
    mean = acc / count[0]
    assert abs(mean[fast].sum() / pred["per_mode"][fast].sum() - 1) < 0.15


def test_bias_offset_predicts_the_biased_fixed_point(h4, lin_h4):
    """A constant residual bias b = B w (w on the five stiffest modes) moves the fixed point
    of fixed-step SD to y* = -w, an energy offset ||w||^2."""
    n = h4.norb
    w_modes = np.zeros(lin_h4.n_states)
    w_modes[-5:] = 1e-3 * np.array([1.0, -2.0, 1.5, 0.5, -1.0])
    w = lin_h4.modes @ w_modes                                   # y coordinates
    bmat = lin_h4.coupling.T * np.sqrt(lin_h4.gaps)[None, :]
    b = (bmat @ w).reshape((n,) * 4)
    pred = lin_h4.bias_offset(b)
    assert np.isclose(pred["energy"], float(w_modes @ w_modes), rtol=1e-8)
    res = run_cqe(h4, ci0=lin_h4.psi0, max_iter=300, eps_candidates=(0.03,), r_tol=0.0,
                  e_tol=0.0, residual_estimator=lambda rd, ci, it: rd.A + b)
    got = lin_h4.mode_energies(res.ci)["per_mode"]
    assert np.allclose(got[-5:], w_modes[-5:] ** 2, rtol=0.03, atol=1e-10)
    assert abs(got.sum() / pred["energy"] - 1) < 0.05
