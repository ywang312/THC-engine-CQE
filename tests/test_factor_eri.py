"""ERI factorization layer: Cholesky, double factorization, THC and lambda.

Lambda values are cross-checked against OpenFermion's own implementations
(``resource_estimates.df/thc.compute_lambda``, ``df.factorize``,
``jordan_wigner``). The OpenFermion tests are skipped if it is not installed.
"""

import functools
import importlib
import os

import numpy as np
import pytest

from conftest import random_active_space
from thc_engine_cqe import systems
from thc_engine_cqe.factor.cholesky import cholesky_eri, packed_supermatrix
from thc_engine_cqe.factor.df import (DoubleFactorization, double_factorize,
                                      eigendecomposition_vectors)
from thc_engine_cqe.factor.eri_factor import (DenseERI, ERIFactor, LambdaBreakdown,
                                              one_body_tprime)
from thc_engine_cqe.factor.thc import (THCFactorization, fit_thc, fit_thc_active, grid_guess,
                                       ls_central, random_guess, thc_eri,
                                       thc_objective_and_grad)
from thc_engine_cqe.fci_ops import FCISpace, Hamiltonian, one_body, two_body_ee

_BUILDERS = {
    "h4": lambda: systems.hydrogen_chain(4, 1.8, "bohr"),
    "lih": lambda: systems.lih(),
    "h6": lambda: systems.hydrogen_chain(6, 1.8, "bohr"),
    "random5": lambda: random_active_space(5, (2, 2), seed=7),
}
MOLECULES = ["h4", "lih", "h6"]


@functools.lru_cache(maxsize=None)
def get_system(name):
    return _BUILDERS[name]()


@pytest.fixture(params=list(_BUILDERS))
def system(request):
    return get_system(request.param)


def _rec(L):
    return np.einsum("lpq,lrs->pqrs", L, L)


def _fro(x):
    return float(np.linalg.norm(np.asarray(x).ravel()))


# ----------------------------------------------------------------- Cholesky
@pytest.mark.parametrize("tol", [1e-3, 1e-6, 1e-10])
def test_cholesky_error_bound(system, tol):
    """max |eri - sum L (x) L| < tol, symmetric vectors, PSD residual."""
    eri = system.eri
    n = system.norb
    L = cholesky_eri(eri, tol=tol)
    assert L.ndim == 3 and L.shape[1:] == (n, n)
    assert L.shape[0] <= n * (n + 1) // 2
    assert np.array_equal(L, L.transpose(0, 2, 1))
    assert np.abs(eri - _rec(L)).max() < tol + 1e-14
    w, (p, q) = packed_supermatrix(eri)
    resid = w - np.einsum("la,lb->ab", L[:, p, q], L[:, p, q])
    assert np.linalg.eigvalsh(resid).min() > -1e-12


def test_cholesky_vector_count_and_cap():
    h6 = get_system("h6")
    counts = [cholesky_eri(h6.eri, tol=t).shape[0] for t in (1e-2, 1e-4, 1e-8, 1e-12)]
    assert counts == sorted(counts) and counts[0] < counts[-1]
    full = cholesky_eri(h6.eri, tol=1e-12)
    capped = cholesky_eri(h6.eri, tol=1e-12, max_vectors=3)
    assert capped.shape[0] == 3
    assert np.allclose(capped, full[:3], atol=1e-14)  # same pivot sequence


def test_cholesky_input_checks():
    eri = get_system("h4").eri
    bad = eri.copy()
    bad[0, 1, 2, 3] += 1e-3
    with pytest.raises(ValueError):
        cholesky_eri(bad)
    # an indefinite supermatrix triggers the warning (the bound does not hold)
    L = cholesky_eri(eri, tol=1e-12)
    indefinite = eri - 2.0 * np.einsum("pq,rs->pqrs", L[0], L[0])
    with pytest.warns(RuntimeWarning):
        cholesky_eri(indefinite, tol=1e-10)
    # symmetry_tol = 1e-10 is relative to max(1, max|eri|): the threshold is 1e-10 for
    # these integrals (max|eri| = 0.60) and 100 * 0.60 * 1e-10 = 6e-9 for 100 * eri.
    assert 0.5 < np.abs(eri).max() < 1.0
    for scale, accepted, rejected in ((1.0, 5e-11, 2e-10), (100.0, 5e-9, 2e-8)):
        for delta, ok in ((accepted, True), (rejected, False)):
            bad = scale * eri
            bad[0, 1, 2, 3] += delta
            if ok:
                cholesky_eri(bad, tol=1e-6)
            else:
                with pytest.raises(ValueError):
                    cholesky_eri(bad, tol=1e-6)


# -------------------------------------------------------------------- DF
@pytest.mark.parametrize("first", ["cholesky", "eigh"])
def test_df_exact_with_all_leaves(system, first):
    eri = system.eri
    df = double_factorize(eri, tol=1e-12, first=first)
    assert np.abs(df.to_eri() - eri).max() < 1e-11
    n = system.norb
    for mu, u in zip(df.eigenvalues, df.eigenvectors):
        assert mu.shape == (n,) and np.allclose(u.T @ u, np.eye(n), atol=1e-12)
    if first == "cholesky":
        # leaves are exactly the Cholesky vectors, reordered (largest norm first).
        # Compare as a bijection: degenerate orbitals (LiH pi) give equal norms,
        # so the order among ties is not unique.
        L = cholesky_eri(eri, tol=1e-12)
        lv = df.leaves()
        dist = np.linalg.norm(lv[:, None] - L[None, :], axis=(2, 3))
        match = np.argmin(dist, axis=1)
        assert np.all(dist[np.arange(len(lv)), match] < 1e-12)
        assert sorted(match.tolist()) == list(range(len(L)))
        norms = np.linalg.norm(lv.reshape(len(lv), -1), axis=1)
        assert np.all(np.diff(norms) <= 1e-12 * norms[0])
    assert df.n_eigenvectors == df.n_leaves * n
    assert df.n_params() == df.n_eigenvectors * (n + 1)


def test_df_leaf_operator_identity():
    """1/2 sum (pq|rs) E_pq E_rs = 1/2 sum_l (sum_pq L_l[p,q] E_pq)^2 on CI vectors."""
    act = get_system("h4")
    space = FCISpace.of(act)
    df = double_factorize(act.eri, tol=1e-12)
    ci = space.random(np.random.default_rng(0), complex_=True)
    lhs = two_body_ee(space, act.eri, ci)
    rhs = sum(one_body(space, Ll, one_body(space, Ll, ci)) for Ll in df.leaves())
    assert np.allclose(lhs, rhs, atol=1e-11)


def test_df_leaf_truncation_is_monotone(system):
    """Keeping the n largest leaves: the residual is a sum of PSD terms, so the
    Frobenius error cannot increase with n."""
    eri = system.eri
    full = double_factorize(eri, tol=1e-12)
    norms = np.linalg.norm(full.leaves().reshape(full.n_leaves, -1), axis=1)
    assert np.all(np.diff(norms) <= 1e-12 * norms[0])  # largest first
    errs = [_fro(double_factorize(eri, tol=1e-12, n_leaves=k).to_eri() - eri)
            for k in range(1, full.n_leaves + 1)]
    assert np.all(np.diff(errs) <= 1e-12)
    assert errs[-1] < 1e-10 and errs[0] > 1e-2


def test_df_eigenvalue_truncation(system):
    """Per-leaf thresholding obeys the triangle-inequality bound
    ||V - V'||_F <= ||V - V_chol||_F + sum_l sqrt(||L_l||^4 - ||L'_l||^4)."""
    eri = system.eri
    full = double_factorize(eri, tol=1e-12)
    chol_err = _fro(full.to_eri() - eri)
    counts = []
    for thr in (1e-1, 1e-2, 1e-3, 1e-5):
        df = double_factorize(eri, tol=1e-12, eig_threshold=thr)
        assert all(np.all(np.abs(m) > thr) for m in df.eigenvalues)
        bound = chol_err
        for mu in full.eigenvalues:
            kept = mu[np.abs(mu) > thr]
            bound += np.sqrt(max(np.sum(mu ** 2) ** 2 - np.sum(kept ** 2) ** 2, 0.0))
        assert _fro(df.to_eri() - eri) <= bound + 1e-12
        counts.append(df.n_eigenvectors)
    assert counts == sorted(counts)
    assert _fro(double_factorize(eri, tol=1e-12, eig_threshold=1e-14).to_eri() - eri) < 1e-10


@pytest.mark.parametrize("name", list(_BUILDERS))
@pytest.mark.parametrize("thresh", [1e-2, 1e-3, 1e-5])
def test_df_reproduces_openfermion_factorize(name, thresh):
    """first='eigh', tol=0, threshold_rule='openfermion' == OpenFermion's df.factorize."""
    pytest.importorskip("openfermion")
    try:
        from openfermion.resource_estimates.df.factorize_df import factorize
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"cannot import openfermion df.factorize: {exc!r}")
    eri = get_system(name).eri
    eri_rr, df_factors, _, num_eigenvectors = factorize(eri, thresh)
    ours = double_factorize(eri, first="eigh", tol=0.0, eig_threshold=thresh,
                            threshold_rule="openfermion")
    assert ours.n_leaves == df_factors.shape[2]
    assert ours.n_eigenvectors == num_eigenvectors
    assert np.allclose(ours.to_openfermion(), df_factors, atol=1e-12)
    assert np.abs(ours.to_eri() - eri_rr).max() < 1e-12


@pytest.mark.parametrize("kw", [{"eig_threshold": 10.0}, {"n_leaves": 0},
                                {"eig_threshold": 10.0, "threshold_rule": "openfermion"}])
def test_df_empty_factorization(kw):
    """No leaf survives: the factorization represents the zero tensor and every
    method still works (lambda is then the one-body part alone)."""
    act = get_system("h4")
    n = act.norb
    df = double_factorize(act.eri, **kw)
    assert df.n_leaves == 0 and df.n_eigenvectors == 0 and df.n_params() == 0
    assert np.array_equal(df.to_eri(), np.zeros((n,) * 4))
    assert np.isclose(df.errors(act.eri)["fro"], _fro(act.eri), rtol=1e-14)
    assert df.to_openfermion().shape == (n, n, 0)
    lam = df.lam(act.h1)
    assert lam.lam_two_body == 0.0 and lam.details["t_eri"] == "factorized"
    assert np.isclose(lam.lam_total, np.abs(np.linalg.eigvalsh(act.h1)).sum(), rtol=1e-12)
    assert df.lam(act.h1, act.eri).lam_two_body == 0.0
    assert not np.any(df.apply_to(act).eri)


def test_eigendecomposition_vectors_are_sorted_and_exact():
    eri = get_system("lih").eri
    L = eigendecomposition_vectors(eri, tol=1e-12)
    norms = np.linalg.norm(L.reshape(len(L), -1), axis=1)
    assert np.all(np.diff(norms) <= 1e-12)
    assert np.abs(_rec(L) - eri).max() < 1e-11
    assert np.abs(L - L.transpose(0, 2, 1)).max() < 1e-10


# ------------------------------------------------------------------- THC
def _thc_guesses(name):
    return ["random", "grid"] if name in MOLECULES else ["random"]


@pytest.mark.parametrize("name,guess", [(n, g) for n in _BUILDERS for g in _thc_guesses(n)])
def test_thc_ls_exact_at_full_pair_rank(name, guess):
    """M = N(N+1)/2 generic points span the symmetric pair space: LS-THC is exact."""
    act = get_system(name)
    n = act.norb
    m = n * (n + 1) // 2
    fac = fit_thc(act.eri, m, guess=guess, mol=act.mol, mo_coeff=act.mo_coeff, refine=False)
    assert fac.info["guess"] == guess and fac.info["ls"]["rank"] == m
    assert np.abs(fac.to_eri() - act.eri).max() < 1e-8
    assert np.array_equal(fac.MPQ, fac.MPQ.T)


@pytest.mark.parametrize("name,guess", [(n, g) for n in MOLECULES for g in ("grid", "random")])
def test_thc_ls_error_nonincreasing_in_nthc(name, guess):
    """Nested point sets => ||V - P V P||_F is non-increasing in M; the LS model
    P V P of the (PSD) exact ERIs is PSD."""
    act = get_system(name)
    n = act.norb
    mmax = n * (n + 1) // 2
    if guess == "grid":
        eta_all, ginfo = grid_guess(act.mol, act.mo_coeff, mmax)
    else:
        eta_all = random_guess(n, mmax, seed=0)
    nrm_all = np.linalg.norm(eta_all, axis=1)
    vmax = np.abs(act.eri).max()
    for m in (2, mmax // 2):  # fit_thc's guess is the prefix of the larger guess (unit rows)
        fac = fit_thc(act.eri, m, guess=guess, mol=act.mol, mo_coeff=act.mo_coeff, refine=False)
        assert np.allclose(fac.etaPp, eta_all[:m] / nrm_all[:m, None], atol=1e-14)
        assert np.allclose(fac.info["guess_row_norms"], nrm_all[:m], rtol=1e-14, atol=0)
        model = fac.to_eri().reshape(n * n, n * n)
        assert np.linalg.eigvalsh(model).min() > -1e-12 * vmax
    errs = np.array([ls_central(act.eri, eta_all[:m])[1]["eri_error_fro"]
                     for m in range(1, mmax + 1)])
    assert np.all(np.diff(errs) <= 1e-10 * errs[0])
    assert errs[-1] < 1e-8 < errs[0]
    if guess == "grid":
        assert ginfo["n_grid"] > mmax and len(set(ginfo["grid_indices"].tolist())) == mmax


def test_thc_ls_error_nonincreasing_without_psd():
    """||V - PVP||^2 = ||V||^2 - ||PVP||^2 holds for any symmetric V, so the LS
    error is monotone in nested point sets even for an indefinite V."""
    rng = np.random.default_rng(11)
    n = 4
    L = rng.standard_normal((6, n, n))
    L = L + L.transpose(0, 2, 1)
    signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    v = np.einsum("l,lpq,lrs->pqrs", signs, L, L)  # 8-fold symmetric, indefinite
    vmat = v.reshape(n * n, n * n)
    assert np.linalg.eigvalsh(vmat).min() < -1.0
    eta_all = random_guess(n, n * (n + 1) // 2, seed=4)
    errs = []
    for m in range(1, eta_all.shape[0] + 1):
        z, info = ls_central(v, eta_all[:m])
        pvp = thc_eri(eta_all[:m], z).reshape(n * n, n * n)
        lhs = info["eri_error_fro"] ** 2
        assert np.isclose(lhs, _fro(vmat) ** 2 - _fro(pvp) ** 2, rtol=1e-9, atol=1e-9 * _fro(vmat) ** 2)
        errs.append(info["eri_error_fro"])
    assert np.all(np.diff(errs) <= 1e-10 * errs[0])
    assert errs[-1] < 1e-8 < errs[0]


def test_ls_central_pseudo_inverse():
    """MPQ = B^+ V B^+T equals S^+ B^T V B S^+ (S = B^T B); a duplicated point
    makes B rank deficient without changing the fit."""
    act = get_system("lih")
    n = act.norb
    eta = random_guess(n, 9, seed=3)
    z, info = ls_central(act.eri, eta)
    b = np.einsum("Pp,Pq->pqP", eta, eta).reshape(n * n, -1)
    s_pinv = np.linalg.pinv(b.T @ b)
    z_ref = s_pinv @ b.T @ act.eri.reshape(n * n, n * n) @ b @ s_pinv
    assert np.allclose(z, z_ref, rtol=1e-8, atol=1e-8 * np.abs(z_ref).max())
    eta_dup = np.vstack([eta, eta[4]])
    z_dup, info_dup = ls_central(act.eri, eta_dup)
    assert info_dup["rank"] == info["rank"] == 9
    assert np.isclose(info_dup["eri_error_fro"], info["eri_error_fro"], rtol=1e-9)
    assert np.allclose(thc_eri(eta_dup, z_dup), thc_eri(eta, z), atol=1e-10)


def test_grid_guess_beats_random_guess():
    """The ISDF/QR point selection gives a much better LS fit than random points
    (H6, M = 2N, LS only)."""
    act = get_system("h6")
    m = 2 * act.norb
    e_grid = fit_thc(act.eri, m, guess="grid", mol=act.mol, mo_coeff=act.mo_coeff,
                     refine=False).info["eri_error_fro"]
    e_rand = fit_thc(act.eri, m, guess="random", refine=False).info["eri_error_fro"]
    assert e_grid < 0.2 * e_rand


@pytest.mark.parametrize("penalty", [0.0, 0.37])
@pytest.mark.parametrize("symmetric", [True, False])
def test_thc_gradient_matches_finite_differences(penalty, symmetric):
    rng = np.random.default_rng(5)
    n, m = 4, 6
    eri = get_system("h4").eri if symmetric else rng.standard_normal((n,) * 4)
    x = rng.standard_normal(m * n + m * m)
    f, g = thc_objective_and_grad(x, eri, m, penalty)
    h = 1e-6
    gfd = np.empty_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = h
        gfd[i] = (thc_objective_and_grad(x + e, eri, m, penalty)[0]
                  - thc_objective_and_grad(x - e, eri, m, penalty)[0]) / (2 * h)
    assert np.abs(g - gfd).max() < 1e-7 * np.abs(g).max()


def test_thc_objective_matches_openfermion_numpy_formulas():
    """Transcription of OpenFermion's numpy thc_objective_grad and the
    regularized objective (thc/utils/thc_objectives.py), which cannot be
    imported without jax."""
    act = get_system("h4")
    eri = act.eri
    n, m = act.norb, 6
    rng = np.random.default_rng(2)
    x = rng.standard_normal(m * n + m * m)
    etaPp = x[: n * m].reshape(m, n)
    MPQ = x[n * m:].reshape(m, m)
    CprP = np.einsum("Pp,Pr->prP", etaPp, etaPp)
    Iapprox = np.einsum("pqU,UV,rsV->pqrs", CprP, MPQ, CprP, optimize=True)
    deri = eri - Iapprox
    res = 0.5 * np.sum(deri ** 2)
    dL_dZab = -np.einsum("pqrs,pqA,rsB->AB", deri, CprP, CprP, optimize=True)
    dL_dX_GT = -2 * np.einsum("Tqrs,Gq,Gv,rsv->GT", deri, etaPp, MPQ, CprP, optimize=True)
    dL_dX_GT -= 2 * np.einsum("pqTs,pqu,uG,Gs->GT", deri, CprP, MPQ, etaPp, optimize=True)
    g_of = np.hstack((dL_dX_GT.ravel(), dL_dZab.ravel()))
    f, g = thc_objective_and_grad(x, eri, m)
    assert np.isclose(f, res, rtol=1e-12)
    assert np.allclose(g, g_of, rtol=1e-10, atol=1e-10 * np.abs(g_of).max())
    # regularized objective: res + penalty * lambda_z^2, lambda_z = 0.5 sum |cP M cP|
    SPQ = etaPp.dot(etaPp.T)
    cP = np.diag(np.diag(SPQ))
    lambda_z = 0.5 * np.sum(np.abs(cP.dot(MPQ).dot(cP)))
    f_pen, _ = thc_objective_and_grad(x, eri, m, penalty=0.25)
    assert np.isclose(f_pen, res + 0.25 * lambda_z ** 2, rtol=1e-12)


@pytest.mark.parametrize("penalty", [0.0, 1e-3])
def test_thc_refinement_descends(penalty):
    act = get_system("h4")
    ls = fit_thc(act.eri, 6, mol=act.mol, mo_coeff=act.mo_coeff, refine=False)
    rf = fit_thc(act.eri, 6, mol=act.mol, mo_coeff=act.mo_coeff, refine=True,
                 penalty=penalty, maxiter=300)
    info = rf.info["refine"]
    assert np.isclose(info["f_initial"], ls.objective(act.eri, penalty), rtol=1e-12)
    assert info["f_final"] <= info["f_initial"]
    assert np.isclose(rf.objective(act.eri, penalty), info["f_final"], rtol=1e-12)
    assert np.array_equal(rf.MPQ, rf.MPQ.T)
    if penalty == 0.0:
        assert rf.info["eri_error_fro"] < 0.2 * ls.info["eri_error_fro"]


@pytest.mark.parametrize("guess", ["grid", "random"])
@pytest.mark.parametrize("refine", [False, True])
def test_fit_thc_returns_unit_gauge(guess, refine):
    """fit_thc's factors have unit leaf rows, so MPQ is OpenFermion's normalized
    central tensor and lambda_Z = 1/2 sum |MPQ|. The LS model does not depend on
    the gauge of the guess."""
    act = get_system("h4")
    m = 6
    fac = fit_thc(act.eri, m, guess=guess, mol=act.mol, mo_coeff=act.mo_coeff, refine=refine,
                  maxiter=200)
    assert np.allclose(np.linalg.norm(fac.etaPp, axis=1), 1.0, rtol=0, atol=1e-12)
    assert np.array_equal(fac.MPQ, fac.MPQ.T)
    eta_hat, mpq_n = fac.normalized()
    assert np.allclose(eta_hat, fac.etaPp, atol=1e-12)
    assert np.allclose(mpq_n, fac.MPQ, rtol=1e-12, atol=0)
    lam = fac.lam(act.h1, act.eri)
    assert np.isclose(lam.lam_two_body, 0.5 * np.abs(fac.MPQ).sum(), rtol=1e-12)
    raw = (grid_guess(act.mol, act.mo_coeff, m)[0] if guess == "grid"
           else random_guess(act.norb, m, seed=0))
    assert np.allclose(fac.info["guess_row_norms"], np.linalg.norm(raw, axis=1), rtol=1e-14)
    if not refine:
        z_raw, _ = ls_central(act.eri, raw)
        model_raw = thc_eri(raw, z_raw)
        assert np.abs(fac.to_eri() - model_raw).max() < 1e-10 * np.abs(model_raw).max()
        assert np.isclose(fac.info["lambda_z"], THCFactorization(raw, z_raw).lam(
            act.h1, act.eri).lam_two_body, rtol=1e-8)


def test_grid_guess_gauge_magnitudes():
    """The scales quoted in the thc module docstring (Gauge): for H4 at M = 4 the
    raw grid leaves are short and their LS central tensor is large; in the unit
    gauge it is of order one."""
    act = get_system("h4")
    raw, _ = grid_guess(act.mol, act.mo_coeff, 4)
    z_raw, _ = ls_central(act.eri, raw)
    assert np.linalg.norm(raw, axis=1).max() < 0.3  # observed 0.108
    assert np.abs(z_raw).max() > 1e3  # observed 7.5e3
    fac = fit_thc(act.eri, 4, mol=act.mol, mo_coeff=act.mo_coeff, refine=False)
    assert np.abs(fac.MPQ).max() < 10  # observed 0.74


def test_thc_refinement_converges_in_unit_gauge():
    """Regression for the refinement gauge. Before the fix, the refinement ran in the
    gauge of the raw grid guess. Observed then (stage-3 review and stage-4 checks):
    L-BFGS-B needed 1100-2700 iterations for H4 at M = 4, and for LiH at M = 6 it
    stopped at maxiter = 5000 with f between 0.14 and 0.29, depending on the thread
    count. In the unit gauge both converge."""
    h4 = get_system("h4")
    fac = fit_thc(h4.eri, 4, mol=h4.mol, mo_coeff=h4.mo_coeff)
    rf = fac.info["refine"]
    assert rf["success"] and rf["nit"] < 200, rf  # observed nit = 28, f = 1.302e-3
    lih = get_system("lih")
    fac = fit_thc(lih.eri, 6, mol=lih.mol, mo_coeff=lih.mo_coeff)
    rf = fac.info["refine"]
    assert rf["success"] and rf["f_final"] < 5e-2, rf  # observed f = 2.31e-2, nit = 395-517
    assert rf["f_final"] < 0.05 * rf["f_initial"]


def test_thc_penalty_auto_is_openfermions_rule():
    """penalty='auto' transcribes lbfgsb_opt_thc_l2reg with penalty_param=None
    (thc/utils/thc_factorization.py, which needs jax): penalty = 1/2 ||V - V_THC||^2
    / lambda_Z at the starting point. It lowers lambda_Z well below the LS value."""
    act = get_system("lih")
    m = 12
    ls = fit_thc(act.eri, m, mol=act.mol, mo_coeff=act.mo_coeff, refine=False, penalty="auto")
    etaPp, MPQ = ls.etaPp, ls.MPQ
    SPQ = etaPp.dot(etaPp.T)
    cP = np.diag(np.diag(SPQ))
    lambda_z = np.sum(np.abs(cP.dot(MPQ).dot(cP))) * 0.5
    CprP = np.einsum("Pp,Pr->prP", etaPp, etaPp)
    deri = act.eri - np.einsum("pqU,UV,rsV->pqrs", CprP, MPQ, CprP, optimize=True)
    penalty_param = 0.5 * np.sum(deri ** 2) / lambda_z
    assert ls.info["penalty_rule"] == "auto"
    assert np.isclose(ls.info["penalty"], penalty_param, rtol=1e-10)
    rf = fit_thc(act.eri, m, mol=act.mol, mo_coeff=act.mo_coeff, penalty="auto")
    assert np.isclose(rf.info["penalty"], penalty_param, rtol=1e-10)
    f0 = 0.5 * np.sum(deri ** 2) + penalty_param * lambda_z ** 2
    assert np.isclose(rf.info["refine"]["f_initial"], f0, rtol=1e-10)
    assert rf.info["refine"]["f_final"] < f0
    assert rf.info["lambda_z"] < 0.5 * ls.info["lambda_z"]  # observed 5.06 vs 16.8
    assert fit_thc(act.eri, 4, refine=False).info["penalty_rule"] == "fixed"
    for bad in ("bogus", -1.0, float("nan")):
        with pytest.raises(ValueError):
            fit_thc(act.eri, 4, refine=False, penalty=bad)


def test_thc_lambda_depends_on_representation_at_exact_rank():
    """At M = N(N+1)/2 the LS-THC model of LiH is exact, but its lambda is several
    times the full-DF lambda (quoted in the thc_floors module docstring)."""
    act = get_system("lih")
    n = act.norb
    fac = fit_thc(act.eri, n * (n + 1) // 2, mol=act.mol, mo_coeff=act.mo_coeff, refine=False)
    assert np.abs(fac.to_eri() - act.eri).max() < 1e-8
    lam_thc = fac.lam(act.h1, act.eri).lam_total
    lam_df = double_factorize(act.eri, first="eigh", tol=0.0).lam(act.h1, act.eri).lam_total
    assert lam_thc > 5 * lam_df  # observed 68.9 vs 9.26


def test_thc_operator_identity():
    """(pq|rs)_THC E_pq E_rs = sum_PQ MPQ O_P O_Q with O_P = sum eta_Pp eta_Pq E_pq."""
    act = get_system("h4")
    space = FCISpace.of(act)
    fac = fit_thc(act.eri, 5, guess="random", refine=False)
    ci = space.random(np.random.default_rng(1))
    lhs = two_body_ee(space, fac.to_eri(), ci)
    ops = [np.outer(e, e) for e in fac.etaPp]
    o_ci = [one_body(space, o, ci) for o in ops]
    rhs = sum(fac.MPQ[P, Q] * one_body(space, ops[P], o_ci[Q])
              for P in range(fac.nthc) for Q in range(fac.nthc))
    assert np.allclose(lhs, rhs, atol=1e-11)
    assert np.allclose(thc_eri(fac.etaPp, fac.MPQ),
                       np.einsum("Pp,Pr,Qq,Qs,PQ->prqs", fac.etaPp, fac.etaPp, fac.etaPp,
                                 fac.etaPp, fac.MPQ), atol=1e-13)


@pytest.mark.parametrize("name,mlist", [("h4", [4, 6, 8, 10]), ("h6", [6, 10, 14, 18, 21])])
def test_thc_fci_energy_error_shrinks_with_nthc(name, mlist):
    """FCI energy of the LS-THC Hamiltonian (grid guess, no refinement) approaches
    E_FCI as M grows, and is exact at M = N(N+1)/2. Monotonicity is an empirical
    property of these systems, not a theorem: only the ERI error is guaranteed
    to be non-increasing."""
    act = get_system(name)
    e_fci = act.fci()[0]
    errs = []
    for m in mlist:
        fac = fit_thc(act.eri, m, mol=act.mol, mo_coeff=act.mo_coeff, refine=False)
        errs.append(abs(act.with_eri(fac.to_eri()).fci()[0] - e_fci))
    assert np.all(np.diff(errs) < 0), errs
    assert errs[-1] < 1e-8


def test_fit_thc_options_and_wrappers():
    act = get_system("h4")
    with pytest.raises(ValueError):
        fit_thc(act.eri, 4, guess="grid")  # no mol / mo_coeff
    with pytest.raises(ValueError):
        fit_thc(act.eri, 4, guess="bogus")
    assert fit_thc(act.eri, 4, refine=False).info["guess"] == "random"  # 'auto' without a grid
    fac = fit_thc_active(act, 6, refine=False)
    assert fac.info["guess"] == "grid"
    assert np.isclose(fac.info["lambda"], fac.lam(act.h1, act.eri).lam_total, rtol=1e-12)
    assert fac.n_params() == 6 * act.norb + 6 * 7 // 2


def test_thc_lambda_is_invariant_under_leaf_rescaling():
    act = get_system("h4")
    fac = fit_thc(act.eri, 7, mol=act.mol, mo_coeff=act.mo_coeff, refine=False)
    c = np.linspace(0.3, 2.5, fac.nthc)
    scaled = THCFactorization(fac.etaPp * c[:, None], fac.MPQ / np.outer(c ** 2, c ** 2))
    assert np.allclose(scaled.to_eri(), fac.to_eri(), atol=1e-12)
    a, b = fac.lam(act.h1, act.eri), scaled.lam(act.h1, act.eri)
    assert np.isclose(a.lam_total, b.lam_total, rtol=1e-12)
    eta_hat, mpq_n = fac.normalized()
    assert np.allclose(np.linalg.norm(eta_hat, axis=1), 1.0)
    assert np.isclose(a.lam_two_body, 0.5 * np.abs(mpq_n).sum(), rtol=1e-12)


# ---------------------------------------------------- lambda vs OpenFermion
@pytest.fixture
def openfermion_lambda(monkeypatch):
    """Load OpenFermion's compute_lambda for 'df' or 'thc' without jax.

    The modules import ``pyscf_to_cas`` from a package that only exports it when
    jax is installed. A placeholder is inserted before the import. The loaded
    module's ``pyscf_to_cas`` is then patched to return our integrals, so its
    ``pyscf_mf`` argument is unused.
    """
    pytest.importorskip("openfermion")
    modules = {"df": "openfermion.resource_estimates.df.compute_lambda_df",
               "thc": "openfermion.resource_estimates.thc.compute_lambda_thc"}

    def load(kind, active):
        try:
            import openfermion.resource_estimates.molecule as ofmol
            if not hasattr(ofmol, "pyscf_to_cas"):
                monkeypatch.setattr(ofmol, "pyscf_to_cas", None, raising=False)
            mod = importlib.import_module(modules[kind])
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"cannot load {modules[kind]}: {exc!r}")
        monkeypatch.setattr(mod, "pyscf_to_cas", lambda mf: (
            active.h1, active.eri, active.ecore, active.nelec[0], active.nelec[1]))
        return mod.compute_lambda

    return load


@pytest.mark.parametrize("use_eri_thc_for_t", [False, True])
def test_thc_lambda_matches_openfermion(system, openfermion_lambda, use_eri_thc_for_t):
    compute_lambda = openfermion_lambda("thc", system)
    n = system.norb
    for m, refine in ((n, True), (2 * n, False)):
        fac = fit_thc(system.eri, m, mol=system.mol, mo_coeff=system.mo_coeff, refine=refine,
                      maxiter=200)
        lam_tot, nthc, sqrt_res, res, lam_t, lam_z = compute_lambda(
            None, fac.etaPp, fac.MPQ, use_eri_thc_for_t=use_eri_thc_for_t)
        ours = fac.lam(system.h1, system.eri, use_eri_thc_for_t=use_eri_thc_for_t)
        assert nthc == m == ours.details["nthc"]
        tol = dict(rtol=1e-10, atol=1e-10)
        assert np.isclose(ours.lam_total, lam_tot, **tol)
        assert np.isclose(ours.lam_one_body, lam_t, **tol)
        assert np.isclose(ours.lam_two_body, lam_z, **tol)
        assert np.isclose(ours.details["ls_residual"], res, **tol)
        assert np.isclose(np.sqrt(ours.details["ls_residual"]), sqrt_res, **tol)


def test_df_lambda_matches_openfermion(system, openfermion_lambda):
    compute_lambda = openfermion_lambda("df", system)
    for kw in ({}, {"eig_threshold": 1e-3}, {"n_leaves": 3},
               {"first": "eigh", "tol": 0.0, "eig_threshold": 1e-3, "threshold_rule": "openfermion"}):
        df = double_factorize(system.eri, **kw)
        ours = df.lam(system.h1, system.eri)
        ref = compute_lambda(None, df.to_openfermion())
        assert np.isclose(ours.lam_total, ref, rtol=1e-10, atol=1e-10)
        assert ours.details["t_eri"] == "exact"
    # without eri_exact the factorized ERIs enter T' (exact here: full DF)
    full = double_factorize(system.eri, tol=1e-12)
    assert np.isclose(full.lam(system.h1).lam_total, full.lam(system.h1, system.eri).lam_total,
                      rtol=1e-10)


@pytest.mark.parametrize("name", ["h4", "lih", "random5"])
def test_dense_pauli_lambda_matches_openfermion_jordan_wigner(name):
    """DenseERI.lam is the JW Pauli 1-norm; its one-/two-body split equals the
    1-norm of the degree-2 / degree-4 Majorana terms."""
    of = pytest.importorskip("openfermion")
    from openfermion.chem.molecular_data import spinorb_from_spatial

    act = get_system(name)
    two = np.asarray(act.eri.transpose(0, 2, 3, 1), order="C")  # OpenFermion (ps|qr) order
    ob, tb = spinorb_from_spatial(act.h1, two)
    op = of.InteractionOperator(0.0, ob, 0.5 * tb)
    qubit = of.jordan_wigner(op)
    lam_jw = sum(abs(c) for term, c in qubit.terms.items() if term != ())
    by_degree = {}
    for term, c in of.get_majorana_operator(op).terms.items():
        by_degree[len(term)] = by_degree.get(len(term), 0.0) + abs(c)
    ours = DenseERI(act.eri).lam(act.h1)
    assert np.isclose(ours.lam_total, lam_jw, rtol=1e-12)
    assert np.isclose(ours.lam_one_body, by_degree[2], rtol=1e-12)
    assert np.isclose(ours.lam_two_body, by_degree[4], rtol=1e-12)
    # Coefficient-wise: every degree-2 Majorana term of H comes from sum T'_pq E_pq,
    # so the shifted two-body part 1/2 sum (pq|rs) Et_pq Et_rs has none.
    ob_t, _ = spinorb_from_spatial(one_body_tprime(act.h1, act.eri), np.zeros_like(two))
    maj_t = of.get_majorana_operator(of.InteractionOperator(0.0, ob_t, 0.0 * tb)).terms
    maj_h = of.get_majorana_operator(op).terms
    deg2 = {t for t in set(maj_h) | set(maj_t) if len(t) == 2}
    assert max(abs(maj_h.get(t, 0.0) - maj_t.get(t, 0.0)) for t in deg2) < 1e-12
    # It does have a constant: H = sum T'E + 1/2 sum V Et Et - 1/2 sum (pp|rr), and the
    # identity term of 1/2 sum V Et Et is 1/4 sum_pq (pq|qp).
    const = maj_h.get((), 0.0) - maj_t.get((), 0.0) + 0.5 * np.einsum("pprr->", act.eri)
    assert abs(const - 0.25 * np.einsum("pqqp->", act.eri)) < 1e-12
    assert abs(const) > 0.1


# ------------------------------------------------ one-body correction T'
def test_tprime_operator_identity(system):
    """H - ecore = sum T'_pq E_pq + 1/2 sum (pq|rs) Et_pq Et_rs - 1/2 sum_pr (pp|rr),
    Et = E - 1, on (complex) CI vectors. This identity defines T' in every lambda."""
    space = FCISpace.of(system)
    eri = system.eri
    tp = one_body_tprime(system.h1, eri)
    c0 = np.einsum("pprr->", eri)
    ci = space.random(np.random.default_rng(4), complex_=True)
    # sum V Et Et = sum V E E - sum_rs (sum_p V_pprs) E_rs - sum_pq (sum_r V_pqrr) E_pq + c0
    vtt = (two_body_ee(space, eri, ci) - one_body(space, np.einsum("pprs->rs", eri), ci)
           - one_body(space, np.einsum("pqrr->pq", eri), ci) + c0 * ci)
    rhs = one_body(space, tp, ci) + 0.5 * vtt - 0.5 * c0 * ci
    assert np.allclose(Hamiltonian(system).apply(ci), rhs, atol=1e-11)


def test_shifted_two_body_constant_on_fock_space():
    """Fock-space oracle (all particle numbers): Et_pq = E_pq - delta_pq is traceless,
    tr(Et_pq Et_rs) / tr(1) = delta_ps delta_qr / 2, so 1/2 sum (pq|rs) Et_pq Et_rs has
    the constant (identity) component 1/4 sum_pq (pq|qp) stated in eri_factor's docstring."""
    import scipy.sparse as sp

    from thc_engine_cqe.testing.fock import FockSpace

    act = get_system("h4")
    n = act.norb
    fs = FockSpace(n)
    one = sp.identity(fs.dim, format="csr")
    et = [[fs.E(p, q) - (one if p == q else 0 * one) for q in range(n)] for p in range(n)]
    assert max(abs(et[p][q].diagonal().sum()) for p in range(n) for q in range(n)) < 1e-12
    tr = np.array([[[[(et[p][q] @ et[r][s]).diagonal().sum() / fs.dim for s in range(n)]
                     for r in range(n)] for q in range(n)] for p in range(n)])
    assert np.allclose(tr, 0.5 * np.einsum("ps,qr->pqrs", np.eye(n), np.eye(n)), atol=1e-14)
    shifted = sum(0.5 * act.eri[p, q, r, s] * (et[p][q] @ et[r][s])
                  for p in range(n) for q in range(n) for r in range(n) for s in range(n))
    assert np.isclose(shifted.diagonal().sum() / fs.dim, 0.25 * np.einsum("pqqp->", act.eri),
                      rtol=1e-12)


# --------------------------------------------------------------- interface
def test_interface_and_parameter_counts():
    act = get_system("h4")
    n = act.norb
    with pytest.raises(TypeError):
        ERIFactor()  # abstract
    dense = DenseERI(act.eri)
    assert dense.kind == "dense" and dense.norb == n
    assert dense.n_params() == 55  # npair = 10 -> 10 * 11 / 2 unique (pq|rs)
    assert np.array_equal(dense.to_eri(), act.eri)
    lam = dense.lam(act.h1)
    assert isinstance(lam, LambdaBreakdown)
    assert np.isclose(lam.lam_total, lam.lam_one_body + lam.lam_two_body)
    df = double_factorize(act.eri, eig_threshold=1e-3)
    assert df.kind == "df" and isinstance(df, ERIFactor)
    assert df.n_params() == df.n_eigenvectors * (n + 1)
    thc = fit_thc(act.eri, 5, guess="random", refine=False)
    assert thc.kind == "thc" and thc.nthc == 5 and thc.n_params() == 5 * n + 15
    nonsym = THCFactorization(thc.etaPp, thc.MPQ + np.triu(np.ones((5, 5)), 1))
    assert nonsym.n_params() == 5 * n + 25
    for fac in (dense, df, thc):
        model = fac.apply_to(act)
        assert np.array_equal(model.h1, act.h1) and np.allclose(model.eri, fac.to_eri())
        err = fac.errors(act.eri)
        assert set(err) == {"fro", "rel_fro", "max_abs"}
    rebuilt = DoubleFactorization(df.eigenvalues, df.eigenvectors)
    assert np.allclose(rebuilt.to_eri(), df.to_eri())


# ------------------------------------------- literature regression (FeMoco)
def _femoco_files():
    pytest.importorskip("openfermion")
    import openfermion.resource_estimates.integrals as ints

    base = os.path.dirname(ints.__file__)
    eri_file = os.path.join(base, "eri_reiher.h5")
    thc_file = os.path.join(base, "M_250_beta_16_eta_10.h5")
    if not (os.path.exists(eri_file) and os.path.exists(thc_file)):
        pytest.skip("OpenFermion's FeMoco integral files are not available")
    return eri_file, thc_file


@pytest.mark.slow
def test_femoco_lambdas_match_openfermion_regression_values():
    """Reiher FeMoco Hamiltonian (54 orbitals) shipped with OpenFermion.

    OpenFermion's own tests assert round(lambda_THC) == 294 for its M = 250 THC
    factors (thc/compute_lambda_thc_test.py), and L = 360, Xi = 13031 and
    lambda_DF = 294.8 for DF with threshold 0.00125
    (df/compute_lambda_df_test.py, "Reproduce Reiher et al orbital DF lambda
    from paper").
    """
    import h5py

    eri_file, thc_file = _femoco_files()
    with h5py.File(eri_file, "r") as f:
        eri, h1 = f["eri"][()], f["h0"][()]
    with h5py.File(thc_file, "r") as f:
        etaPp, MPQ = f["etaPp"][()], f["MPQ"][()]
    lam_thc = THCFactorization(etaPp, MPQ).lam(h1, eri)
    assert np.round(lam_thc.lam_total) == 294
    df = double_factorize(eri, first="eigh", tol=0.0, eig_threshold=0.00125,
                          threshold_rule="openfermion")
    assert (df.n_leaves, df.n_eigenvectors) == (360, 13031)
    assert np.isclose(np.round(df.lam(h1, eri).lam_total, 1), 294.8)
