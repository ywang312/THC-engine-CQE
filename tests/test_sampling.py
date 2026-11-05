"""Finite-shot noise models of the CQE (thc_engine_cqe.sampling).

H4/STO-3G at 1.8 bohr throughout. The per-element variances of the direct route
with matchgate shadows are computed here with measure.targets and
measure.schemes (not with the routes variance functions).
"""

import dataclasses
import math

import numpy as np
import pytest

from thc_engine_cqe import systems
from thc_engine_cqe.cqe import run_cqe
from thc_engine_cqe.diagnostics.linearized import linearize, structured_basis
from thc_engine_cqe.measure import routes, schemes, targets
from thc_engine_cqe.measure.pauli import QubitMapping, QubitState
from thc_engine_cqe.optimizers import structure_error, symmetrize_residual
from thc_engine_cqe.residual import compute_residual
from thc_engine_cqe.sampling import (EnergyNoise, ResidualNoise, ShotLedger, SnapshotResidual,
                                     element_variances, noise_factor)


@pytest.fixture(scope="module", autouse=True)
def _single_openmp_thread():
    """One PySCF OpenMP thread (bitwise reproducible runs); restored afterwards."""
    from pyscf import lib

    old = lib.num_threads()
    lib.num_threads(1)
    yield
    lib.num_threads(old)


@pytest.fixture(scope="module")
def h4():
    return systems.hydrogen_chain(4, 1.8, "bohr")


@pytest.fixture(scope="module")
def fci_h4(h4):
    return h4.fci()


@pytest.fixture(scope="module")
def lin_h4(h4):
    return linearize(h4)


@pytest.fixture(scope="module")
def it2_h4(h4):
    return run_cqe(h4, max_iter=2).ci


def _direct_variances(active, ci, kind):
    m = QubitMapping(active.norb)
    el = targets.element_set(active.norb, kind)
    st = QubitState.from_ci(active.norb, active.nelec, ci, m)
    return schemes.matchgate_shadow_variance(targets.direct_targets(el, active, m, kind), st)


@pytest.fixture(scope="module")
def mg_var(h4, fci_h4):
    """Direct route, matchgate shadows, at FCI: exact per-shot per-element variances.

    Observed: spin-orbital total 426.3 (per element 0.63-13.6), spin-free 671.4 (1.37-54.6).
    """
    return {kind: _direct_variances(h4, fci_h4[1], kind) for kind in targets.KINDS}


def _rd(active, ci):
    return compute_residual(active, ci)


def _structured(a, norb):
    u = structured_basis(norb)
    x = a.ravel()
    return np.linalg.norm(x - u @ (u.T @ x)) <= 1e-12 * max(np.linalg.norm(x), 1e-300)


# --------------------------------------------------------------------------- fill and structure
def test_element_sets_and_fill_columns(h4):
    """The fill of a unit error: spin-free orbits (2 or 4 entries) and the spin contraction
    (H4: 6 vanishing columns, 108 with norm^2 4, 36 with norm^2 8)."""
    n = h4.norb
    so = ResidualNoise(h4, kind="spin_orbital", shots=1.0, per_element_variance=np.ones(150), rng=0)
    sf = ResidualNoise(h4, kind="spin_free", shots=1.0, per_element_variance=np.ones(60), rng=0)
    iso = ResidualNoise(h4, route="isotropic", sigma=1.0, shots=1.0, rng=0)
    assert (so.K, sf.K, iso.K) == (150, 60, 60)
    rng = np.random.default_rng(3)
    v = rng.standard_normal(150)
    assert np.array_equal(so.fill(v), targets.contract_spin(v, so.elements, n))
    w = rng.standard_normal(60)
    assert np.array_equal(sf.fill(w), targets.spin_free_tensor(w, sf.elements, n))
    lso, lsf = so.fill_matrix(), sf.fill_matrix()
    assert np.allclose(lso @ v, so.fill(v).ravel(), atol=1e-14)
    n2 = np.round((lso ** 2).sum(axis=0), 12)
    assert {float(k): int(c) for k, c in zip(*np.unique(n2, return_counts=True))} == {0.0: 6, 4.0: 108, 8.0: 36}
    assert {float(k): int(c) for k, c in zip(*np.unique(np.round((lsf ** 2).sum(axis=0), 12), return_counts=True))} \
        == {2.0: 6, 4.0: 54}
    for e, ((p, sp), (r, sr), (q, sq), (s, ss)) in enumerate(so.elements):
        col = lso[:, e].reshape((n,) * 4)
        if sp != sr:                       # alpha-beta: one orbit; the 2A^{(p a)(q b)}_{(q a)(p b)} vanish
            if p == s and q == r:
                assert not col.any()
            elif p == r and q == s:
                assert col[p, q, p, q] == 2 and col[q, p, q, p] == -2 and np.count_nonzero(col) == 2
            else:
                assert col[p, q, r, s] == col[r, s, p, q] == 1 and col[q, p, s, r] == col[s, r, q, p] == -1
                assert np.count_nonzero(col) == 4
        else:                              # alpha-alpha / beta-beta: the orbit and its exchange partner
            assert col[p, q, r, s] == col[r, s, p, q] == 1 and col[q, p, s, r] == col[s, r, q, p] == -1
            assert col[r, q, p, s] == col[p, s, r, q] == -1 and col[s, p, q, r] == col[q, r, s, p] == 1
            assert np.count_nonzero(col) == 8
    assert np.array_equal(iso.fill_matrix(), structured_basis(n))


def test_estimates_have_the_structure_of_A(h4, it2_h4, mg_var):
    """Every model returns a fresh real tensor in A's symmetry subspace; rd.A is not modified."""
    rd = _rd(h4, it2_h4)
    a0 = rd.A.copy()
    models = [ResidualNoise(h4, kind=k, shots=1e3, per_element_variance=mg_var[k], rng=1) for k in targets.KINDS]
    models += [ResidualNoise(h4, route="aux", kind="spin_orbital", shots=1e4, delta=0.2,
                             per_element_variance=np.full(150, 50.0), rng=2),
               ResidualNoise(h4, route="isotropic", sigma=0.1, shots=4.0, rng=3),
               SnapshotResidual(h4, shots=50, rng=4)]
    for m in models:
        a = m(rd, it2_h4, 0)
        assert a.shape == rd.A.shape and a.dtype == np.float64 and a is not rd.A
        pair, herm = structure_error(a)
        assert pair <= 1e-13 * np.abs(a).max() and herm <= 1e-13 * np.abs(a).max()
        assert _structured(a, h4.norb)
        assert not np.allclose(a, rd.A)
    assert np.array_equal(rd.A, a0)
    inf = ResidualNoise(h4, kind="spin_free", shots=np.inf, per_element_variance=mg_var["spin_free"], rng=5)
    assert np.array_equal(inf(rd, it2_h4, 0), symmetrize_residual(rd.A))
    assert np.abs(inf(rd, it2_h4, 0) - rd.A).max() <= 1e-15 * np.abs(rd.A).max()


# --------------------------------------------------------------------------- Gaussian statistics
def test_gaussian_model_is_unbiased_with_the_stated_covariance(h4, it2_h4, mg_var):
    """Spin-free kind: the representatives of A_est - A are the element errors themselves;
    by sampling, their mean is 0 and their covariance diag(v_e)/S."""
    rd = _rd(h4, it2_h4)
    s = 2e4
    v = mg_var["spin_free"].per_target
    model = ResidualNoise(h4, kind="spin_free", shots=s, per_element_variance=v, rng=11)
    el = model.elements
    m = 4000
    err = np.array([(model(rd, it2_h4, k) - rd.A)[tuple(el.T)] for k in range(m)])
    z = err.mean(axis=0) / np.sqrt(v / s / m)
    assert np.abs(z).max() < 4.5 and abs(np.sqrt(np.mean(z ** 2)) - 1) < 0.3
    ratio = err.var(axis=0) / (v / s)
    assert np.abs(ratio - 1).max() < 5 * math.sqrt(2 / m)
    rho = np.corrcoef(err.T)[~np.eye(60, dtype=bool)]
    assert np.abs(rho).max() < 5 / math.sqrt(m)
    assert model.ledger.n_iterations == m and np.all(model.ledger.residual_shots == s)


def test_aux_route_mean_is_A_plus_bias(h4, it2_h4):
    """Aux route: E[A_est] = A + b(Psi) with b the route bias of the current state."""
    rd = _rd(h4, it2_h4)
    model = ResidualNoise(h4, route="aux", kind="spin_free", shots=1e6, delta=0.3,
                          per_element_variance=np.ones(60), rng=12)
    b = model.bias_tensor(it2_h4, rd.A)
    m = 300
    mean = np.mean([model(rd, it2_h4, k) for k in range(m)], axis=0)
    el = model.elements
    z = (mean - rd.A - b)[tuple(el.T)] / math.sqrt(1.0 / 1e6 / m)
    assert np.abs(z).max() < 4.5
    assert np.linalg.norm(b[tuple(el.T)]) > 10 * math.sqrt(60 / 1e6 / m)     # the bias is resolved


def test_aux_bias_equals_aux_residual_minus_exact_residual(h4, it2_h4):
    """The filled bias -sum_l Im dm2(Lambda_l)/delta - A is the fill of aux_residual - exact_residual,
    for both element kinds (the spin-orbital one through contract_spin) and every method."""
    n = h4.norb
    rd = _rd(h4, it2_h4)
    for method in routes.METHODS:
        tensors = []
        for kind in targets.KINDS:
            el = targets.element_set(n, kind)
            model = ResidualNoise(h4, route="aux", kind=kind, shots=np.inf, delta=0.3, aux_method=method,
                                  per_element_variance=np.zeros(el.shape[0]), rng=0)
            ref = (routes.aux_residual(h4, it2_h4, 0.3, method, el, kind)
                   - routes.exact_residual(h4, it2_h4, el, kind))
            b = model.bias_tensor(it2_h4, rd.A)
            assert np.abs(b - model.fill(ref)).max() < 1e-13 * np.abs(b).max()
            assert np.array_equal(model(rd, it2_h4, 0), symmetrize_residual(rd.A + b))
            tensors.append(b)
        assert np.abs(tensors[0] - tensors[1]).max() < 1e-13 * np.abs(tensors[0]).max()


def test_exact_aux_route_damps_each_excitation_by_sinc(h4, lin_h4):
    """Exact aux route near an eigenstate: b(Psi_0) = 0, and to first order in d (all orders in
    delta) b = sum_k d_k w_k (sinc(delta w_k) - 1) M_k (central differences, 1e-8 relative observed)."""
    model = ResidualNoise(h4, route="aux", kind="spin_free", shots=np.inf, delta=0.6,
                          per_element_variance=np.zeros(60), rng=0)
    psi0 = lin_h4.psi0.ravel()
    assert np.abs(model.bias_tensor(lin_h4.psi0)).max() < 1e-12
    t = 1e-4
    for k in (0, 5, lin_h4.n_states - 1):
        bp = model.bias_tensor((psi0 + t * lin_h4.states[k]).reshape(lin_h4.psi0.shape))
        bm = model.bias_tensor((psi0 - t * lin_h4.states[k]).reshape(lin_h4.psi0.shape))
        w = lin_h4.gaps[k]
        pred = w * (math.sin(0.6 * w) / (0.6 * w) - 1.0) * lin_h4.coupling[k]
        assert np.linalg.norm((bp - bm).ravel() / (2 * t) - pred) < 1e-6 * np.linalg.norm(pred)


# --------------------------------------------------------------------------- noise factor
def test_noise_factor_matches_the_empirical_covariance(h4, lin_h4, it2_h4, mg_var):
    """F = L diag(sqrt(v/S)) (columns: filled unit errors); the filled errors of the model have
    covariance F F^T, checked on the projections that drive the linearized CQE."""
    n = h4.norb
    rd = _rd(h4, it2_h4)
    s = 1e5
    v = mg_var["spin_orbital"].per_target
    model = ResidualNoise(h4, kind="spin_orbital", shots=s, per_element_variance=v, rng=21)
    f = noise_factor(model, n)
    assert f.shape == (n ** 4, 150)
    for e in (0, 29, 40, 149):
        unit = np.zeros(150)
        unit[e] = 1.0
        col = targets.contract_spin(unit, model.elements, n).ravel() * math.sqrt(v[e] / s)
        assert np.allclose(f[:, e], col, rtol=1e-15, atol=0)
    m = 5000
    err = np.array([(model(rd, it2_h4, k) - rd.A).ravel() for k in range(m)])
    proj = lin_h4.modes.T @ (np.sqrt(lin_h4.gaps)[:, None] * lin_h4.coupling)      # (modes, N^4)
    q_emp = np.cov((err @ proj.T).T, bias=True)
    q_pred = (proj @ f) @ (proj @ f).T
    assert np.abs(np.diag(q_emp) / np.diag(q_pred) - 1).max() < 5 * math.sqrt(2 / m)
    assert abs(np.trace(err.T @ err) / m / np.sum(f ** 2) - 1) < 0.02
    with pytest.raises(ValueError):
        noise_factor(model, n + 1)
    iso = ResidualNoise(h4, route="isotropic", sigma=0.2, shots=4.0, rng=0)
    assert np.allclose(noise_factor(iso), 0.1 * structured_basis(n), atol=1e-15)
    assert not noise_factor(ResidualNoise(h4, kind="spin_free", shots=np.inf,
                                          per_element_variance=mg_var["spin_free"], rng=0)).any()


def test_full_covariance_option(h4, it2_h4):
    """per_element_covariance: eta = C^{1/2} z / sqrt(S); a diagonal C reproduces the
    independent-element model draw for draw, and F F^T = L C L^T / S."""
    rd = _rd(h4, it2_h4)
    rng = np.random.default_rng(5)
    v = rng.uniform(1.0, 5.0, 60)
    a = ResidualNoise(h4, kind="spin_free", shots=10.0, per_element_variance=v, rng=7)
    b = ResidualNoise(h4, kind="spin_free", shots=10.0, per_element_covariance=np.diag(v), rng=7)
    for k in range(3):
        assert np.allclose(a(rd, it2_h4, k), b(rd, it2_h4, k), rtol=0, atol=1e-13)
    x = rng.standard_normal((60, 60))
    c = x @ x.T / 60 + np.diag(v)
    model = ResidualNoise(h4, kind="spin_free", shots=10.0, per_element_covariance=c, rng=8)
    f = noise_factor(model)
    lmat = model.fill_matrix()
    assert np.allclose(f @ f.T, lmat @ c @ lmat.T / 10.0, atol=1e-12)
    assert np.allclose(model.variances, np.diag(c))
    eta = np.array([model.sample_element_errors() for _ in range(4000)])
    assert np.abs(np.cov(eta.T, bias=True) * 10.0 - c).max() < 0.15 * np.abs(c).max()
    with pytest.raises(ValueError):
        ResidualNoise(h4, kind="spin_free", shots=1.0, per_element_covariance=-np.eye(60), rng=0)


def test_noise_floor_predicts_the_noisy_cqe_h4(h4, lin_h4, mg_var):
    """Fixed-step SD from FCI with ResidualNoise (direct route, matchgate shadows, spin-orbital
    elements, S = 1e6 shots per iteration): the time-averaged energy of the fast modes
    (1 - eps lambda/2 < 0.98; 16 of 19) matches LinearizedCQE.noise_floor(eps, noise_factor).
    Predicted floor 1.105e-5 Ha (eps = 0.03). Observed ratios of the fast-mode energy: 0.98-1.05
    over 3 seeds x eps in {0.03, 0.05} x S in {1e5, 1e6} x both element kinds; total E - E_FCI
    0.97-1.03 of the predicted total."""
    eps, s = 0.03, 1e6
    model = ResidualNoise(h4, kind="spin_orbital", shots=s, per_element_variance=mg_var["spin_orbital"], rng=1)
    pred = lin_h4.noise_floor(eps, noise_factor(model, h4.norb))
    assert pred["stable"] and not pred["undriven"].any()
    assert abs(pred["total"] / 1.105e-5 - 1) < 1e-3
    fast = 1 - 0.5 * eps * lin_h4.stiffness < 0.98
    acc = np.zeros(lin_h4.n_states)
    count = [0]
    n_burn = 200

    def cb(step, ci):
        if step.iteration >= n_burn:
            acc[:] += lin_h4.mode_energies(ci)["per_mode"]
            count[0] += 1

    run_cqe(h4, ci0=lin_h4.psi0, max_iter=n_burn + 1500, eps_candidates=(eps,), r_tol=0.0, e_tol=0.0,
            residual_estimator=model, callback=cb)
    mean = acc / count[0]
    assert abs(mean[fast].sum() / pred["per_mode"][fast].sum() - 1) < 0.15
    assert model.ledger.n_iterations == n_burn + 1500


def test_aux_bias_fixed_point_matches_bias_offset_h4(h4, lin_h4):
    """Aux route (first-order DF-Trotter, delta = 0.1, untruncated DF), shots = inf, fixed-step SD
    (eps = 0.05) from FCI: the per-mode energies after n steps match the linear prediction
    (y*_m (1 - p_m^n))^2, p_m = 1 - eps lambda_m / 2, with y* = -B^+ b(Psi_0) from bias_offset.
    The three softest modes (lambda <= 0.98) hold 0.03% of the offset (4.02e-6 Ha), so 250
    steps reach 99.99% of it. Observed: 1.014 (total, after 400 and 1500 steps; 1.13 at
    delta = 0.3); with the linear state dependence of the bias, b(Psi) = b(Psi_0) + J d, the
    linear prediction agrees to 1e-4 (3e-3 at delta = 0.3)."""
    eps, n_it = 0.05, 250
    model = ResidualNoise(h4, route="aux", kind="spin_free", shots=np.inf, delta=0.1, aux_method="df-trotter",
                          per_element_variance=np.zeros(60), rng=0)
    pred = lin_h4.bias_offset(model.bias_tensor(lin_h4.psi0))
    assert abs(pred["energy"] / 4.024e-6 - 1) < 1e-3
    pm = 1 - 0.5 * eps * lin_h4.stiffness
    pred_n = ((lin_h4.modes.T @ pred["y"]) * (1 - pm ** n_it)) ** 2
    res = run_cqe(h4, ci0=lin_h4.psi0, max_iter=n_it, eps_candidates=(eps,), r_tol=0.0, e_tol=0.0,
                  residual_estimator=model)
    got = lin_h4.mode_energies(res.ci)
    assert abs(got["per_mode"].sum() / pred_n.sum() - 1) < 0.05
    stiff = slice(lin_h4.n_states - 5, None)
    assert abs(got["per_mode"][stiff].sum() / pred["per_mode"][stiff].sum() - 1) < 0.05
    assert abs(got["outside"]) < 0.05 * pred_n.sum()


def test_exact_aux_route_keeps_the_fixed_point(h4, lin_h4, fci_h4):
    """Exact aux route (delta = 0.3), shots = inf: no offset at first order; the run stays at FCI."""
    model = ResidualNoise(h4, route="aux", kind="spin_orbital", shots=np.inf, delta=0.3,
                          per_element_variance=np.zeros(150), rng=0)
    assert lin_h4.bias_offset(model.bias_tensor(lin_h4.psi0))["energy"] < 1e-20
    res = run_cqe(h4, ci0=lin_h4.psi0, max_iter=40, eps_candidates=(0.05,), r_tol=0.0, e_tol=0.0,
                  residual_estimator=model, e_exact=fci_h4[0])
    assert np.abs(res.errors).max() < 1e-11


# --------------------------------------------------------------------------- snapshots
def test_snapshot_residual_matches_the_gaussian_diagonal(h4, it2_h4):
    """Simulated matchgate snapshots at SD iterate 2: unbiased for the exact residual, per-element
    variances equal to the exact matchgate per-element variances (the Gaussian model's diagonal),
    and correlated elements. Observed with 1e5 snapshots (FCI and iterate 2): variance ratios
    0.98-1.02 per element, off-diagonal correlation coefficients RMS 0.049 (spin-orbital) and
    0.096 (spin-free), max |rho| 0.42 and 0.73."""
    kind = "spin_orbital"
    vr = _direct_variances(h4, it2_h4, kind)
    snap = SnapshotResidual(h4, kind=kind, shots=100, rng=31)
    a = routes.exact_residual(h4, it2_h4, snap.elements, kind)
    r = 30000
    x = snap.snapshot_estimates(it2_h4, r)
    z = (x.mean(axis=0) - a) / np.sqrt(vr.per_target / r)
    assert np.abs(z).max() < 4.5 and abs(np.sqrt(np.mean(z ** 2)) - 1) < 0.25
    dev2 = (x - a) ** 2
    zv = (dev2.mean(axis=0) - vr.per_target) / (dev2.std(axis=0) / math.sqrt(r))
    assert np.abs(zv).max() < 4.5
    tot = dev2.sum(axis=1)
    assert abs(tot.mean() - vr.total) < 4.5 * tot.std() / math.sqrt(r)
    rho = np.corrcoef(x.T)[~np.eye(snap.K, dtype=bool)]
    assert 0.03 < np.sqrt(np.mean(rho ** 2)) < 0.07 and 0.3 < np.abs(rho).max() < 0.55
    assert snap.info["max_eigenvalue_deviation"] < 1e-10


def test_snapshot_estimates_agree_between_element_kinds(h4, it2_h4):
    """The shadow estimator is linear in the target, so the same snapshots give the same filled
    tensor for the spin-orbital and the spin-free element set."""
    a = SnapshotResidual(h4, kind="spin_orbital", shots=2000, rng=41)
    b = SnapshotResidual(h4, kind="spin_free", shots=2000, rng=41)
    ta = a.fill(a.estimate(it2_h4))
    tb = b.fill(b.estimate(it2_h4))
    assert np.abs(ta - tb).max() < 1e-12 * np.abs(ta).max()


def test_snapshot_residual_in_run_cqe(h4, it2_h4):
    """Drop-in residual_estimator: charges the ledger, keeps A's structure, reproducible by seed."""
    runs = []
    for _ in range(2):
        snap = SnapshotResidual(h4, shots=300, rng=np.random.default_rng(51))
        res = run_cqe(h4, ci0=it2_h4, max_iter=3, eps_candidates=(0.05,), r_tol=0.0, e_tol=0.0,
                      residual_estimator=snap, keep_residuals=True)
        assert np.all(snap.ledger.residual_shots == 300) and snap.ledger.n_iterations == 3
        runs.append(res)
    for s1, s2 in zip(runs[0].steps, runs[1].steps):
        assert np.array_equal(s1.residual_est, s2.residual_est) and s1.energy_after == s2.energy_after
    mean, cov = SnapshotResidual(h4, shots=1, rng=52).element_covariance(it2_h4, 2000)
    assert mean.shape == (150,) and cov.shape == (150, 150) and np.allclose(cov, cov.T)


# --------------------------------------------------------------------------- energies and ledger
def test_energy_noise(h4, fci_h4, it2_h4):
    """E_est = E + N(0, V/S): DF variances at the estimated state itself (shift=True), frozen
    variances for the other schemes; unbiased with variance V/S by sampling."""
    e_it2 = compute_residual(h4, it2_h4).energy
    model = EnergyNoise(h4, shots=1e4, rng=61)
    v = schemes.df_variance(h4, it2_h4, shift=True).total
    assert model.variance_mode == "exact" and model.variance_of(it2_h4) == pytest.approx(v, rel=1e-12)
    m = 1000
    draws = np.array([model(e_it2, it2_h4, "candidate") for _ in range(m)]) - e_it2
    assert abs(draws.mean()) < 4.5 * math.sqrt(v / 1e4 / m)
    assert abs(draws.var() / (v / 1e4) - 1) < 5 * math.sqrt(2 / m)
    assert model.ledger.summary()["energy_shots_candidate"] == m * 1e4
    st = QubitState.from_ci(h4.norb, h4.nelec, fci_h4[1], QubitMapping(h4.norb))
    qwc = EnergyNoise(h4, scheme="qwc", shots=1e4, rng=62)
    assert qwc.variance_mode == "frozen"
    assert qwc.frozen_variance == pytest.approx(
        schemes.qwc_variance(targets.energy_target(h4), st).total, rel=1e-12)
    assert EnergyNoise(h4, shots=np.inf, rng=63)(-1.5, it2_h4, "current") == -1.5
    assert EnergyNoise(h4, shots=1.0, per_shot_variance=0.0, rng=64)(-1.5, it2_h4, "current") == -1.5
    with pytest.raises(ValueError):
        model(e_it2, it2_h4, "next")


def test_ledger_counts(h4, mg_var):
    """One shared ledger: per iteration the residual shots, one current-state energy and one
    energy per nonzero candidate (the zero candidate reuses the current estimate, at no cost)."""
    cands = (0.0, -0.025, 0.025, 0.05)
    ledger = ShotLedger()
    rn = ResidualNoise(h4, shots=1e5, per_element_variance=mg_var["spin_orbital"], rng=71, ledger=ledger)
    en = EnergyNoise(h4, shots=1e4, rng=72, ledger=ledger)
    res = run_cqe(h4, max_iter=4, eps_candidates=cands, r_tol=0.0, e_tol=0.0,
                  residual_estimator=rn, energy_estimator=en)
    assert ledger.n_iterations == len(res.steps) == 4
    assert [r["iteration"] for r in ledger.records] == [0, 1, 2, 3]
    assert np.all(ledger.residual_shots == 1e5) and np.all(ledger.energy_shots == 4e4)
    assert all(r["n_current"] == 1 and r["n_candidate"] == 3 for r in ledger.records)
    assert np.array_equal(ledger.cumulative(), np.cumsum(np.full(4, 1.4e5)))
    summ = ledger.summary()
    assert summ["total_shots"] == 5.6e5 and summ["energy_shots_current"] == 4e4
    assert summ["n_energy_estimates_candidate"] == 12 and summ["n_residual_estimates"] == 4
    # exact residuals, noisy energies: one record per "current" call
    en2 = EnergyNoise(h4, shots=10.0, rng=73)
    run_cqe(h4, max_iter=3, eps_candidates=cands, r_tol=0.0, e_tol=0.0, energy_estimator=en2)
    assert en2.ledger.n_iterations == 3 and np.all(en2.ledger.residual_shots == 0)
    assert np.all(en2.ledger.energy_shots == 40.0)
    ledger.reset()
    assert ledger.n_iterations == 0 and ledger.summary()["total_shots"] == 0.0
    with pytest.raises(ValueError):
        ledger.charge_energy(1.0, "bogus")


def test_runs_are_reproducible_with_seeds(h4, mg_var):
    """Equal seeds give bit-identical noisy runs (residual and energy noise); other seeds differ."""
    cands = tuple(0.00625 * 2 ** k for k in range(6))

    def run(seed):
        rs, es = np.random.SeedSequence(seed).spawn(2)
        rn = ResidualNoise(h4, shots=1e4, per_element_variance=mg_var["spin_orbital"], rng=np.random.default_rng(rs))
        en = EnergyNoise(h4, shots=1e5, rng=np.random.default_rng(es))
        res = run_cqe(h4, max_iter=5, eps_candidates=cands, r_tol=0.0, e_tol=0.0, optimizer="lbfgs",
                      residual_estimator=rn, energy_estimator=en, keep_residuals=True)
        return res, rn

    (r1, m1), (r2, _), (r3, _) = run(81), run(81), run(82)
    assert m1.seed["entropy"] == 81
    assert np.array_equal(r1.energies, r2.energies) and np.array_equal(r1.energies_est, r2.energies_est)
    assert all(np.array_equal(a.residual_est, b.residual_est) for a, b in zip(r1.steps, r2.steps))
    assert not np.array_equal(r1.energies_est, r3.energies_est)


# --------------------------------------------------------------------------- variances and validation
def test_element_variances_and_the_default_variance_source(h4, fci_h4, it2_h4, mg_var):
    """ResidualNoise computes its variances at the FCI state by default; element_variances
    wires targets, aux states and schemes (expectations are the exact and aux-route residuals,
    df-fragments combines its L + 1 states with the optimal allocation)."""
    opts = {"method": "rb", "n_settings": 200, "seed": 3}
    model = ResidualNoise(h4, kind="spin_free", shots=1e6, rng=0, variance_options=opts)
    assert model.info["variance_source"] == "computed"
    ref = element_variances(h4, fci_h4[1], route="direct", kind="spin_free", **opts)
    assert np.array_equal(model.variances, ref.per_target)
    assert abs(ref.total / mg_var["spin_free"].total - 1) < 5 * ref.stderr / ref.total
    n = h4.norb
    el = targets.element_set(n, "spin_free")
    r = element_variances(h4, it2_h4, route="direct", kind="spin_free", method="rb", n_settings=20, seed=1)
    assert np.allclose(r.expectation, routes.exact_residual(h4, it2_h4, el, "spin_free"), atol=1e-12)
    ra = element_variances(h4, it2_h4, route="aux", kind="spin_free", delta=0.1, aux_method="df-fragments")
    assert np.allclose(ra.expectation, routes.aux_residual(h4, it2_h4, 0.1, "df-fragments", el, "spin_free"),
                       atol=1e-12)
    parts = ra.info["parts"]
    assert len(parts) == 11 and ra.total == pytest.approx(sum(math.sqrt(p.total) for p in parts) ** 2)


def test_input_validation(h4, mg_var):
    v = mg_var["spin_orbital"].per_target
    with pytest.raises(ValueError):
        ResidualNoise(h4, route="bogus", shots=1.0, per_element_variance=v, rng=0)
    for shots in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            ResidualNoise(h4, shots=shots, per_element_variance=v, rng=0)
    with pytest.raises(ValueError):
        ResidualNoise(h4, route="isotropic", shots=1.0, rng=0)
    with pytest.raises(ValueError):
        ResidualNoise(h4, shots=1.0, per_element_variance=v, sigma=1.0, rng=0)
    with pytest.raises(ValueError):
        ResidualNoise(h4, route="aux", shots=1.0, per_element_variance=v, rng=0)
    with pytest.raises(ValueError):
        ResidualNoise(h4, shots=1.0, per_element_variance=v[:-1], rng=0)
    with pytest.raises(ValueError):
        ResidualNoise(h4, shots=1.0, per_element_variance=-v, rng=0)
    with pytest.raises(ValueError):
        ResidualNoise(h4, shots=1.0, per_element_variance=v, rng=None)
    with pytest.raises(ValueError):
        EnergyNoise(h4, scheme="bogus", shots=1.0, rng=0)
    with pytest.raises(ValueError):
        EnergyNoise(h4, shots=1.0, variance="exact", per_shot_variance=1.0, rng=0)
    with pytest.raises(ValueError):
        SnapshotResidual(h4, shots=10.5, rng=0)
    with pytest.raises(TypeError):
        noise_factor(SnapshotResidual(h4, shots=10, rng=0))
    rd = _rd(h4, h4.fci()[1])
    rd_complex = dataclasses.replace(rd, A=rd.A + 1e-3j * structured_basis(h4.norb)[:, 0].reshape(rd.A.shape))
    model = ResidualNoise(h4, shots=1.0, per_element_variance=v, rng=0)
    with pytest.raises(ValueError):
        model(rd_complex, h4.fci()[1], 0)
