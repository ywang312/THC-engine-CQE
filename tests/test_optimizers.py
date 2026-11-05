"""Search directions (SD, CG, L-BFGS), their metric, and the estimator hooks of cqe.run_cqe (M2).

Iteration counts quoted in docstrings were measured with this code (exact line
search, OMP_NUM_THREADS=1); the assertions keep a margin around them, except
for the 1.6 mHa regression counts of the spin-orbital metric, asserted exactly.

The bit-for-bit comparisons need a deterministic computation. PySCF's
multi-threaded OpenMP reductions are not (observed on H4: with three threads,
repeated identical runs in one process differ within the first three steps), so
this module pins PySCF to one thread whatever OMP_NUM_THREADS says.
"""

import itertools
import math
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg
import scipy.optimize
import scipy.sparse as sp

import thc_engine_cqe.cqe as cqe_module
from thc_engine_cqe import systems
from thc_engine_cqe.cqe import run_cqe
from thc_engine_cqe.factor import projection_compressor
from thc_engine_cqe.fci_ops import FCISpace, Hamiltonian, KrylovPropagator, normalize, two_body
from thc_engine_cqe.optimizers import (LBFGS, METRICS, ConjugateGradient, SteepestDescent,
                                       apply_metric, direction_slope, make_optimizer, real_inner,
                                       structure_error, symmetrize_residual)
from thc_engine_cqe.residual import compute_residual, descent_generator
from thc_engine_cqe.testing.fock import FockSpace

CANDS = (0.025, 0.05, 0.1, 0.2)
STEP_FIELDS = ("iteration", "energy", "variance", "residual_norm", "slope", "full_slope", "eps",
               "energy_after")


@pytest.fixture(scope="module", autouse=True)
def _single_openmp_thread():
    """One PySCF OpenMP thread for the bitwise comparisons (module docstring); restored afterwards."""
    from pyscf import lib

    old = lib.num_threads()
    lib.num_threads(1)
    yield
    lib.num_threads(old)


@pytest.fixture(scope="module")
def h4():
    return systems.hydrogen_chain(4, spacing=1.8, unit="bohr")


@pytest.fixture(scope="module")
def h6():
    return systems.hydrogen_chain(6, spacing=1.8, unit="bohr")


@pytest.fixture(scope="module")
def lih():
    return systems.lih()


@pytest.fixture(scope="module")
def fci_energy(h4, h6, lih):
    return {"h4": h4.fci()[0], "h6": h6.fci()[0], "lih": lih.fci()[0]}


def _same_arrays(x, y):
    if x is None or y is None:
        return x is None and y is None
    return x.dtype == y.dtype and np.array_equal(x, y)


def assert_identical_runs(r1, r2):
    """Bit-for-bit equality of the exact records, the directions and the final state."""
    assert len(r1.steps) == len(r2.steps)
    for a, b in zip(r1.steps, r2.steps):
        assert tuple(getattr(a, f) for f in STEP_FIELDS) == tuple(getattr(b, f) for f in STEP_FIELDS)
        assert _same_arrays(a.generator, b.generator)
        assert _same_arrays(a.residual, b.residual)
        assert a.n_energy_evals == b.n_energy_evals
    assert r1.converged == r2.converged
    assert _same_arrays(r1.ci, r2.ci)


def iterations_to(errors, threshold):
    """First k with E_k - E_FCI < threshold (E_0 = HF), or None."""
    hits = np.flatnonzero(np.asarray(errors) < threshold)
    return int(hits[0]) if hits.size else None


def spin_square(ci, norb, nelec):
    """<psi|S^2|psi> of a (possibly complex) CI vector.

    PySCF's spin_square0 takes a real vector v and returns the unnormalized
    <v|S^2|v>. S^2 is real symmetric, so <psi|S^2|psi> = <Re|S^2|Re> + <Im|S^2|Im>.
    """
    from pyscf.fci import spin_op

    ci = np.asarray(ci)
    return sum(spin_op.spin_square0(np.ascontiguousarray(part), norb, nelec)[0]
               for part in (ci.real, ci.imag) if np.any(part))


# --------------------------------------------------------------- frozen M1 driver
# The numerical path of cqe.run_cqe, cqe._krylov_step, cqe.line_search and
# cqe.generator_norm_estimate at the end of M1 (commit de21460), copied so that
# "the defaults reproduce M1 bit for bit" is checked against M1 itself rather
# than against the extended driver.

def _m1_generator_norm_estimate(space, g, psi, iters=6):
    rng = np.random.default_rng(11)
    x = np.asarray(psi) + 0.1 * rng.standard_normal(space.shape)
    x = x / np.linalg.norm(x)
    est = 0.0
    for _ in range(iters):
        y = two_body(space, g, x)
        est = float(np.linalg.norm(y))
        if est == 0.0:
            return 0.0
        x = y / est
    return est


def _m1_line_search(energy_of_eps, eps_max, n_grid=64, candidates=None):
    if candidates is not None:
        cand = np.asarray(candidates, dtype=float).ravel()
        vals = np.array([energy_of_eps(e) for e in cand])
        i = int(np.argmin(vals))
        return float(cand[i]), float(vals[i])
    grid = np.linspace(0.0, eps_max, n_grid + 1)
    vals = np.array([energy_of_eps(e) for e in grid])
    i = int(np.argmin(vals))
    lo = grid[max(i - 1, 0)]
    hi = grid[min(i + 1, n_grid)]
    if hi <= lo:
        return float(grid[i]), float(vals[i])
    res = scipy.optimize.minimize_scalar(energy_of_eps, bounds=(lo, hi), method="bounded",
                                         options={"xatol": 1e-10 * max(eps_max, 1e-12)})
    if res.fun <= vals[i]:
        return float(res.x), float(res.fun)
    return float(grid[i]), float(vals[i])


def _m1_krylov_step(ham, g, psi, eps_max, tol, candidates=None):
    space = ham.space

    def matvec(v):
        return two_body(space, g, v)

    if candidates is not None:
        eps_max = float(np.max(candidates))
    for _ in range(6):
        prop = KrylovPropagator(matvec, psi, eps_max=eps_max, tol=tol)
        V = prop.basis()
        HV = np.column_stack([ham.apply(V[:, j].reshape(space.shape)).ravel() for j in range(V.shape[1])])
        hproj = V.conj().T @ HV

        def energy(eps):
            y = prop.coefficients(eps)
            return float((np.vdot(y, hproj @ y) / np.vdot(y, y)).real) + ham.ecore

        eps, e = _m1_line_search(energy, eps_max, candidates=candidates)
        if candidates is not None or eps < 0.98 * eps_max:
            return eps, e, normalize(prop.state(eps))
        eps_max *= 2.0
    return eps, e, normalize(prop.state(eps))


def _m1_run_cqe(active, ci0=None, max_iter=30, r_tol=1e-7, e_tol=1e-11, compress=None,
                krylov_tol=1e-10, eps_candidates=None):
    """M1 run_cqe; returns (records, ci, converged) with records like CQEStep (generator, residual kept)."""
    ham = Hamiltonian(active)
    space = ham.space
    psi = space.hf() if ci0 is None else normalize(np.asarray(ci0).reshape(space.shape))
    records, converged = [], False
    for it in range(max_iter):
        rd = compute_residual(active, psi, ham)
        A = rd.A
        g_full = descent_generator(A)
        full_slope = -float(np.sum(np.abs(A) ** 2))
        g = g_full if compress is None else np.asarray(compress(g_full, rd, it))
        slope = -float(np.real(np.sum(g * A)))
        if slope > 0:
            g, slope = -g, -slope
        if rd.residual_norm < r_tol or abs(slope) < 1e-300:
            converged = rd.residual_norm < r_tol
            records.append((it, rd.energy, rd.variance, rd.residual_norm, slope, full_slope, 0.0,
                            rd.energy, g, A))
            break
        gnorm = _m1_generator_norm_estimate(space, g, psi)
        eps_max = 1.5 / max(gnorm, 1e-12)
        eps, e_new, psi_new = _m1_krylov_step(ham, g, psi, eps_max, krylov_tol, eps_candidates)
        records.append((it, rd.energy, rd.variance, rd.residual_norm, slope, full_slope, eps, e_new, g, A))
        converged_e = abs(rd.energy - e_new) < e_tol
        psi = psi_new
        if converged_e:
            converged = True
            break
    return records, psi, converged


M1_CASES = {
    "default": dict(max_iter=40),
    "r_tol-stop": dict(max_iter=40, r_tol=0.02),
    "e_tol-stop": dict(max_iter=60, e_tol=1e-5),
    "candidates": dict(max_iter=8, eps_candidates=CANDS),
    "compressed": dict(max_iter=4, compress=lambda: projection_compressor(n_leaves=4)),
    "complex-odd": dict(max_iter=3, complex_start=True,
                        compress=lambda: projection_compressor(n_leaves=3, complete_groups=False)),
}


@pytest.mark.parametrize("case", list(M1_CASES))
def test_defaults_reproduce_m1_bit_for_bit(h4, case):
    """Default arguments give the M1 trajectory bit for bit: every recorded number,
    every generator and residual, the final state and the stopping decision.
    The cases cover the r_tol and e_tol stops, eps_candidates, a compressor, and a
    complex state (odd leaf truncation from a complex HF vector)."""
    kw = dict(M1_CASES[case])
    if kw.pop("complex_start", False):
        kw["ci0"] = FCISpace.of(h4).hf().astype(complex)
    make = kw.pop("compress", None)
    kw_ref = dict(kw, compress=make() if make else None)
    kw_new = dict(kw, compress=make() if make else None)
    records, ci, converged = _m1_run_cqe(h4, **kw_ref)
    res = run_cqe(h4, keep_generators=True, keep_residuals=True, **kw_new)
    assert len(res.steps) == len(records)
    for s, r in zip(res.steps, records):
        assert tuple(getattr(s, f) for f in STEP_FIELDS) == r[:8]
        assert _same_arrays(s.generator, r[8]) and _same_arrays(s.residual, r[9])
        assert s.residual_norm_est is None and s.slope_est is None and s.energy_est is None
        assert s.optimizer_info is None
    assert res.converged == converged
    assert _same_arrays(res.ci, ci)


def test_defaults_reproduce_m1_lih(lih):
    """Same check on LiH (15 iterations of the exact line search)."""
    records, ci, _ = _m1_run_cqe(lih, max_iter=15)
    res = run_cqe(lih, max_iter=15, keep_generators=True, keep_residuals=True)
    for s, r in zip(res.steps, records):
        assert tuple(getattr(s, f) for f in STEP_FIELDS) == r[:8]
        assert _same_arrays(s.generator, r[8])
    assert len(res.steps) == len(records) and _same_arrays(res.ci, ci)


def test_compressor_sign_flip_reproduces_m1(h4):
    """Regression for the compressor sign flip, which M2 rewrote (it now tests the
    slope the driver acts on). A compressor returning -g is never a descent
    direction; the flip turns it back into g exactly, so the run equals the
    default run and the frozen M1 driver bit for bit."""
    neg = lambda g, rd, it: -g                                   # noqa: E731
    records, ci, converged = _m1_run_cqe(h4, max_iter=6, compress=neg)
    res = run_cqe(h4, max_iter=6, keep_generators=True, keep_residuals=True, compress=neg)
    assert len(res.steps) == len(records) == 6
    for s, r in zip(res.steps, records):
        assert tuple(getattr(s, f) for f in STEP_FIELDS) == r[:8]
        assert _same_arrays(s.generator, r[8]) and _same_arrays(s.residual, r[9])
    assert res.converged == converged and _same_arrays(res.ci, ci)
    assert_identical_runs(run_cqe(h4, max_iter=6, keep_generators=True, keep_residuals=True), res)


def test_compressor_sign_flip_uses_the_estimated_slope(h4):
    """With A_est = -A and compress = -g, the compressor returns -conj(A_est) = conj(A),
    whose estimated slope -Re sum conj(A) A_est = +||A||^2 is positive. The driver flips
    it to conj(A_est): downhill for A_est, uphill for the exact A. A flip decided on the
    exact slope, or no flip, would keep conj(A)."""
    res = run_cqe(h4, max_iter=2, keep_generators=True, keep_residuals=True,
                  compress=lambda g, rd, it: -g, residual_estimator=lambda rd, ci, it: -rd.A)
    assert len(res.steps) == 2 and not res.converged
    for s in res.steps:
        assert _same_arrays(s.generator, np.conj(s.residual_est))
        assert np.isclose(s.slope_est, s.full_slope, rtol=1e-12)
        assert np.isclose(s.slope, -s.full_slope, rtol=1e-12)


def test_sd_equals_default(h4):
    ref = run_cqe(h4, max_iter=12, keep_generators=True, keep_residuals=True)
    for opts in (None, {}):
        res = run_cqe(h4, max_iter=12, optimizer="sd", optimizer_options=opts,
                      keep_generators=True, keep_residuals=True)
        assert_identical_runs(ref, res)
    assert res.optimizer == "sd"


@pytest.mark.parametrize("cands", [None, CANDS], ids=["exact", "candidates"])
@pytest.mark.parametrize("opt", ["cg", "lbfgs"])
def test_first_step_equals_sd(h4, opt, cands):
    sd = run_cqe(h4, max_iter=1, keep_generators=True, keep_residuals=True, eps_candidates=cands)
    res = run_cqe(h4, max_iter=1, optimizer=opt, keep_generators=True, keep_residuals=True,
                  eps_candidates=cands)
    assert_identical_runs(sd, res)
    assert res.optimizer == opt


def test_cg_with_restart_1_is_sd(h4):
    """restart=1 resets the CG cycle at every iteration, so CG is then steepest descent."""
    sd = run_cqe(h4, max_iter=10, keep_generators=True, keep_residuals=True)
    cg = run_cqe(h4, max_iter=10, optimizer="cg", optimizer_options={"restart": 1},
                 keep_generators=True, keep_residuals=True)
    assert_identical_runs(sd, cg)
    assert [s.optimizer_info["restart"] for s in cg.steps] == ["first"] + ["periodic"] * 9


@pytest.mark.parametrize("opt, opts", [("cg", None), ("cg", {"beta": "fr"}), ("lbfgs", None),
                                       ("lbfgs", {"memory": 2})])
@pytest.mark.parametrize("system", ["h4", "h6"])
def test_monotone_with_exact_line_search(request, system, opt, opts):
    act = request.getfixturevalue(system)
    res = run_cqe(act, max_iter=15, optimizer=opt, optimizer_options=opts)
    assert np.all(np.diff(res.energies) <= 1e-12)
    assert all(s.slope < 0 for s in res.steps)
    # the exact slope is the one the driver acted on (no estimator)
    assert all(s.slope_est is None for s in res.steps)


@pytest.mark.parametrize("opt", ["cg", "lbfgs"])
@pytest.mark.parametrize("start", ["real", "complex"])
def test_directions_keep_structure_and_singlet(h6, opt, start):
    """Every CG/L-BFGS direction is pair-symmetric and anti-Hermitian to 1e-12
    (relative), and <S^2> stays 0. The complex start is psi_2 + 0.3i psi_HF
    (psi_2: two SD steps), a complex singlet with complex residuals."""
    space = FCISpace.of(h6)
    ci0 = None
    if start == "complex":
        psi2 = run_cqe(h6, max_iter=2).ci
        ci0 = normalize(psi2 + 0.3j * space.hf())
    states = []
    res = run_cqe(h6, ci0=ci0, max_iter=8, optimizer=opt, keep_generators=True,
                  callback=lambda step, ci: states.append(ci))
    for s in res.steps:
        d = s.generator
        assert np.iscomplexobj(d) == (start == "complex")
        pair, herm = structure_error(d)
        scale = np.abs(d).max()
        assert pair <= 1e-12 * scale and herm <= 1e-12 * scale
    for ci in states:
        assert abs(spin_square(ci, h6.norb, h6.nelec)) < 1e-10
    assert np.all(np.diff(res.energies) <= 1e-12)


@pytest.mark.parametrize("opt", ["sd", "cg", "lbfgs"])
def test_exact_line_search_makes_next_gradient_orthogonal(h4, lih, opt):
    """The previous search line continues straight in the new chart, so after an
    exact line search the slope of d_k at Psi_{k+1}, -Re sum d_k A_{k+1}, vanishes:
    below 1e-3 of the initial slope (observed <= 3.2e-5 over 12 iterations, largest
    for L-BFGS on H4)."""
    for act in (h4, lih):
        res = run_cqe(act, max_iter=12, optimizer=opt, keep_generators=True, keep_residuals=True)
        for a, b in zip(res.steps[:-1], res.steps[1:]):
            assert abs(direction_slope(a.generator, b.residual)) < 1e-3 * abs(a.slope)


@pytest.mark.parametrize("system, threshold, measured", [
    ("h4", 1e-5, {"sd": 35, "cg": 9, "lbfgs": 9}),
    ("h6", 1e-4, {"sd": None, "cg": 17, "lbfgs": 15}),
    ("lih", 1e-5, {"sd": 51, "cg": 10, "lbfgs": 9}),
])
def test_cg_and_lbfgs_converge_much_faster_than_sd(request, fci_energy, system, threshold, measured):
    """Iterations to E - E_FCI < threshold (exact line search, measured):

    * H4 (1.8 bohr), 1e-5: SD 35, CG 9, L-BFGS 9. To 1e-8: CG 22, L-BFGS 15, SD not
      within 60 (4.6e-6 after 60).
    * H6 (1.8 bohr), 1e-4: SD not within 60 (1.45e-4 after 60), CG 17, L-BFGS 15.
      To 1e-3: SD 18, CG 7, L-BFGS 7.
    * LiH, 1e-5: SD 51, CG 10, L-BFGS 9.

    Asserted: CG and L-BFGS within 1.4x their measured counts, and SD still above
    the threshold after 2.5x the larger of them.
    """
    act = request.getfixturevalue(system)
    e_fci = fci_energy[system]
    slowest = max(measured["cg"], measured["lbfgs"])
    for opt in ("cg", "lbfgs"):
        cap = math.ceil(1.4 * measured[opt])
        res = run_cqe(act, max_iter=cap, optimizer=opt, e_exact=e_fci)
        n = iterations_to(res.errors, threshold)
        assert n is not None and n <= cap, (opt, n)
    sd = run_cqe(act, max_iter=math.floor(2.5 * slowest), e_exact=e_fci)
    assert iterations_to(sd.errors, threshold) is None


# ---------------------------------------------------------------- estimator hooks
@pytest.mark.parametrize("opt", ["sd", "cg", "lbfgs"])
def test_identity_residual_estimator_reproduces_exact_run(h4, opt):
    """Including the e_tol stop (measured: CG stops after 31 steps, L-BFGS after 19)."""
    ref = run_cqe(h4, max_iter=40, optimizer=opt, keep_generators=True, keep_residuals=True)
    res = run_cqe(h4, max_iter=40, optimizer=opt, keep_generators=True, keep_residuals=True,
                  residual_estimator=lambda rd, ci, it: rd.A)
    assert_identical_runs(ref, res)
    assert ref.converged == (opt != "sd") and (len(ref.steps) < 40) == (opt != "sd")
    for s in res.steps:
        assert s.residual_norm_est == s.residual_norm and s.slope_est == s.slope
        assert _same_arrays(s.residual_est, s.residual)


@pytest.mark.parametrize("opt", ["sd", "cg", "lbfgs"])
def test_uphill_estimated_direction_is_not_convergence(h4, opt):
    """Regression: with A_est = -A every direction is uphill for the exact energy, so
    the exact line search returns eps = 0 and |E - E_new| ~ 0 < e_tol. The run used to
    stop after one step with converged=True; such steps no longer count as e_tol
    convergence, and the run ends at max_iter unconverged."""
    res = run_cqe(h4, max_iter=4, optimizer=opt, residual_estimator=lambda rd, ci, it: -rd.A)
    assert len(res.steps) == 4 and not res.converged
    for s in res.steps:
        assert s.slope_est < 0 < s.slope and s.eps == 0.0
        assert abs(s.energy_after - s.energy) < 1e-12


@pytest.mark.parametrize("opt", ["cg", "lbfgs"])
def test_cg_and_lbfgs_checks_use_the_estimate(h4, opt):
    """Regression for the descent checks of CG and L-BFGS: at iteration 1 the estimate
    is A_est = -A (exact otherwise). After the exact first step <g_1, g_0> ~ 0, so the
    CG direction g_1 + beta d_0 has beta ~ ||A_1||^2 / ||A_0||^2 > 0, and the L-BFGS
    direction is H g_1 with H positive definite. Both are descent directions for A_est
    and uphill for the exact A: checked on the exact A, CG would restart and L-BFGS
    would reset. The exact line search then makes no step, and the run goes on."""
    res = run_cqe(h4, max_iter=4, optimizer=opt,
                  residual_estimator=lambda rd, ci, it: -rd.A if it == 1 else rd.A)
    s1 = res.steps[1]
    assert s1.slope_est < 0 < s1.slope and s1.eps < 1e-8
    if opt == "cg":
        assert s1.optimizer_info["restart"] is None and s1.optimizer_info["beta"] > 0
    else:
        assert s1.optimizer_info["pair"] == "stored" and not s1.optimizer_info["reset"]
        assert s1.optimizer_info["n_pairs"] == 1
    assert len(res.steps) == 4 and not res.converged and res.steps[2].eps > 0


@pytest.mark.parametrize("opt", ["cg", "lbfgs"])
def test_optimizer_receives_the_estimate(h4, opt, monkeypatch):
    """The optimizer gets direction(g, A) with A = A_est (the estimator's output) and
    g = conj(A_est), not the exact residual."""
    rng = np.random.default_rng(5)
    returned, seen = [], []

    def noisy(rd, ci, it):
        a = symmetrize_residual(rd.A + 0.05 * np.abs(rd.A).max() * rng.standard_normal(rd.A.shape))
        returned.append(a.copy())
        return a

    make = cqe_module.make_optimizer

    def spy_make(name, options, **kwargs):
        opt_obj = make(name, options, **kwargs)
        direction = opt_obj.direction

        def spy(g, A):
            seen.append((np.array(g), np.array(A)))
            return direction(g, A)

        opt_obj.direction = spy
        return opt_obj

    monkeypatch.setattr(cqe_module, "make_optimizer", spy_make)
    res = run_cqe(h4, max_iter=4, optimizer=opt, residual_estimator=noisy, keep_residuals=True)
    assert len(seen) == len(returned) == len(res.steps) == 4
    for (g, a), a_est, s in zip(seen, returned, res.steps):
        assert np.array_equal(a, a_est) and np.array_equal(g, np.conj(a_est))
        assert not np.allclose(a, s.residual)


def test_vanishing_estimated_slope_stops_the_run(h4):
    """The run stops when the slope it acts on, -Re sum d A_est, vanishes, whatever the
    exact slope. A_est keeps the elements of A with an index 0 and the compressor
    returns conj(A) on the others; both masks are invariant under the index maps of the
    residual structure, so -Re sum d A_est = 0 exactly while -Re sum d A < 0."""
    idx = np.indices((h4.norb,) * 4)
    mask = np.any(idx == 0, axis=0)
    res = run_cqe(h4, max_iter=3, keep_residuals=True,
                  residual_estimator=lambda rd, ci, it: np.where(mask, rd.A, 0.0),
                  compress=lambda g, rd, it: np.where(mask, 0.0, np.conj(rd.A)))
    (step,) = res.steps
    assert step.slope_est == 0.0 and step.slope < -1e-3 and step.eps == 0.0
    assert step.residual_norm_est > 0.1 and not res.converged


@pytest.mark.parametrize("opt", ["cg", "lbfgs"])
def test_editing_records_does_not_change_the_run(h4, opt):
    """Regression: CQEStep.generator was the optimizer's own history array, so a callback
    scaling it in place changed the later CG/L-BFGS directions (observed before the fix:
    energies off by 5e-3 after 6 H4 steps). The optimizers now keep copies."""
    kw = dict(max_iter=6, optimizer=opt, keep_generators=True, keep_residuals=True,
              residual_estimator=lambda rd, ci, it: rd.A)
    ref = run_cqe(h4, **kw)

    def vandal(step, ci):
        step.generator *= 0.5
        step.residual *= 0.5
        step.residual_est *= 0.5

    res = run_cqe(h4, callback=vandal, **kw)
    for a, b in zip(ref.steps, res.steps):
        assert tuple(getattr(a, f) for f in STEP_FIELDS) == tuple(getattr(b, f) for f in STEP_FIELDS)
        assert np.array_equal(a.generator, 2.0 * b.generator)
    assert len(ref.steps) == len(res.steps) and _same_arrays(ref.ci, res.ci)


@pytest.mark.parametrize("opt", ["sd", "cg", "lbfgs"])
def test_identity_energy_estimator_reproduces_candidate_run(h4, opt):
    ref = run_cqe(h4, max_iter=10, optimizer=opt, eps_candidates=CANDS, keep_generators=True,
                  keep_residuals=True)
    res = run_cqe(h4, max_iter=10, optimizer=opt, eps_candidates=CANDS, keep_generators=True,
                  keep_residuals=True, energy_estimator=lambda e, ci, ctx: e)
    assert_identical_runs(ref, res)
    for s in res.steps:
        assert s.energy_est == s.energy and s.energy_after_est == s.energy_after
    assert np.array_equal(res.energies_est, res.energies)
    both = run_cqe(h4, max_iter=10, optimizer=opt, eps_candidates=CANDS, keep_generators=True,
                   keep_residuals=True, energy_estimator=lambda e, ci, ctx: e,
                   residual_estimator=lambda rd, ci, it: rd.A)
    assert_identical_runs(ref, both)


def test_energy_estimator_requires_candidates(h4):
    with pytest.raises(ValueError, match="eps_candidates"):
        run_cqe(h4, max_iter=2, energy_estimator=lambda e, ci, ctx: e)
    with pytest.raises(ValueError):
        run_cqe(h4, max_iter=2, energy_estimator=lambda e, ci, ctx: e, eps_candidates=())


@pytest.mark.parametrize("opt", ["cg", "lbfgs"])
def test_compress_needs_sd(h4, opt):
    with pytest.raises(NotImplementedError):
        run_cqe(h4, max_iter=2, optimizer=opt, compress=projection_compressor(n_leaves=4))


def test_bad_optimizer_arguments(h4):
    with pytest.raises(ValueError, match="optimizer must be one of"):
        run_cqe(h4, max_iter=1, optimizer="newton")
    with pytest.raises(ValueError, match="unknown optimizer_options"):
        run_cqe(h4, max_iter=1, optimizer="sd", optimizer_options={"memory": 3})
    with pytest.raises(ValueError, match="unknown optimizer_options"):
        run_cqe(h4, max_iter=1, optimizer="cg", optimizer_options={"memory": 3})
    with pytest.raises(ValueError):
        make_optimizer("cg", {"beta": "hs"})
    with pytest.raises(ValueError):
        make_optimizer("cg", {"restart": 0})
    with pytest.raises(ValueError):
        make_optimizer("lbfgs", {"memory": 0})


def test_residual_estimator_drives_every_decision(h4):
    """A_est = A/2: the direction is conj(A_est), slope_est = -||A_est||^2 = full_slope/4,
    the exact slope is full_slope/2; ||A_est|| < r_tol stops the run although ||A|| is large."""
    res = run_cqe(h4, max_iter=3, keep_generators=True, keep_residuals=True,
                  residual_estimator=lambda rd, ci, it: 0.5 * rd.A)
    for s in res.steps:
        assert _same_arrays(s.generator, np.conj(0.5 * s.residual))
        assert np.isclose(s.residual_norm_est, 0.5 * s.residual_norm, rtol=1e-14)
        assert np.isclose(s.slope_est, 0.25 * s.full_slope, rtol=1e-14)
        assert np.isclose(s.slope, 0.5 * s.full_slope, rtol=1e-14)
        assert np.isclose(s.captured_descent, 0.5, rtol=1e-14)
    tiny = run_cqe(h4, max_iter=5, residual_estimator=lambda rd, ci, it: 1e-9 * rd.A)
    (step,) = tiny.steps
    assert tiny.converged and step.eps == 0.0 and step.residual_norm > 0.1
    assert step.residual_norm_est < 1e-7


def test_residual_estimator_arguments_and_copies(h4):
    """The estimator gets copies of the exact ResidualData and the normalized state plus the
    iteration index; mutating them does not affect the run."""
    seen = []

    def vandal(rd, ci, it):
        seen.append(it)
        assert np.isclose(np.linalg.norm(ci), 1.0, atol=1e-12)
        assert np.allclose(compute_residual(h4, ci).A, rd.A, atol=1e-12)
        out = rd.A.copy()
        rd.A[...] = 0.0
        rd.T[...] = 0.0
        ci[...] = 0.0
        return out

    ref = run_cqe(h4, max_iter=4, keep_generators=True, keep_residuals=True)
    res = run_cqe(h4, max_iter=4, keep_generators=True, keep_residuals=True, residual_estimator=vandal)
    assert_identical_runs(ref, res)
    assert seen == [0, 1, 2, 3]


def test_residual_estimate_must_have_residual_structure(h4):
    rng = np.random.default_rng(3)
    noise = rng.standard_normal((h4.norb,) * 4) * 1e-3
    with pytest.raises(ValueError, match="structure of A"):
        run_cqe(h4, max_iter=1, residual_estimator=lambda rd, ci, it: rd.A + noise)
    with pytest.raises(ValueError, match="shape"):
        run_cqe(h4, max_iter=1, residual_estimator=lambda rd, ci, it: rd.A[0])
    with pytest.raises(ValueError, match="non-finite"):
        run_cqe(h4, max_iter=1, residual_estimator=lambda rd, ci, it: rd.A * np.nan)
    res = run_cqe(h4, max_iter=3, optimizer="lbfgs",
                  residual_estimator=lambda rd, ci, it: symmetrize_residual(rd.A + noise))
    assert len(res.steps) == 3 and np.all(np.diff(res.energies) < 0)


def test_symmetrize_residual_is_the_orthogonal_projector(h4, rng):
    n = h4.norb
    x = rng.standard_normal((n,) * 4) + 1j * rng.standard_normal((n,) * 4)
    p = symmetrize_residual(x)
    assert max(structure_error(p)) < 1e-15
    assert np.allclose(symmetrize_residual(p), p, atol=1e-15)       # idempotent
    y = rng.standard_normal((n,) * 4) + 1j * rng.standard_normal((n,) * 4)
    assert np.isclose(real_inner(p, y), real_inner(x, symmetrize_residual(y)), atol=1e-12)  # symmetric
    A = compute_residual(h4, run_cqe(h4, max_iter=2).ci).A
    assert np.array_equal(symmetrize_residual(A), A)


def test_energy_estimator_contexts_and_states(h4):
    """Per iteration: one "current" call for the state before the step, then one
    "candidate" call per candidate in order, each with a normalized state whose
    exact energy is the first argument. The terminal iteration only calls "current"."""
    ham = Hamiltonian(h4)
    log = []

    def est(e, ci, ctx):
        assert np.isclose(np.linalg.norm(ci), 1.0, atol=1e-12)
        assert np.isclose(ham.energy(ci), e, atol=1e-10)
        log.append(ctx)
        return e

    res = run_cqe(h4, max_iter=3, eps_candidates=CANDS, energy_estimator=est)
    assert log == (["current"] + ["candidate"] * len(CANDS)) * 3
    log.clear()
    res = run_cqe(h4, max_iter=5, eps_candidates=CANDS, energy_estimator=est,
                  residual_estimator=lambda rd, ci, it: rd.A if it < 2 else 1e-9 * rd.A)
    assert log == (["current"] + ["candidate"] * len(CANDS)) * 2 + ["current"]
    assert res.converged and res.steps[-1].eps == 0.0
    assert res.steps[-1].energy_after_est == res.steps[-1].energy_est


def test_energy_estimator_drives_candidate_choice_and_e_tol(h4):
    """The candidate with the lowest *estimated* energy is taken, even when its exact
    energy is not the lowest, and e_tol compares estimated energies."""
    n_cand = len(CANDS)
    calls = {"candidate": 0}

    def favor_largest(e, ci, ctx):
        if ctx == "current":
            return e
        k = calls["candidate"] % n_cand
        calls["candidate"] += 1
        return e if k == n_cand - 1 else e + 1e3

    res = run_cqe(h4, max_iter=3, eps_candidates=CANDS, energy_estimator=favor_largest)
    ref = run_cqe(h4, max_iter=1, eps_candidates=CANDS)
    assert all(s.eps == max(CANDS) for s in res.steps)
    assert ref.steps[0].eps != max(CANDS)          # the exact choice differs
    assert all(s.energy_after_est == s.energy_after for s in res.steps)
    flat = run_cqe(h4, max_iter=5, eps_candidates=CANDS, energy_estimator=lambda e, ci, ctx: 0.0)
    assert flat.converged and len(flat.steps) == 1
    assert flat.steps[0].energy_after < flat.steps[0].energy - 1e-3   # exact energy did drop
    with pytest.raises(ValueError, match="non-finite"):
        run_cqe(h4, max_iter=1, eps_candidates=CANDS, energy_estimator=lambda e, ci, ctx: np.nan)
    with pytest.raises(ValueError, match="no energy estimates"):
        _ = ref.energies_est


def test_energy_evaluation_counts(h4):
    exact = run_cqe(h4, max_iter=4, r_tol=1e-7)
    assert all(s.n_energy_evals > 65 for s in exact.steps)      # 65 grid points + Brent
    cand = run_cqe(h4, max_iter=4, eps_candidates=CANDS)
    assert [s.n_energy_evals for s in cand.steps] == [len(CANDS)] * 4
    assert list(cand.cumulative_energy_evals) == [0, 4, 8, 12, 16]
    stop = run_cqe(h4, max_iter=4, r_tol=10.0)
    assert stop.steps[0].n_energy_evals == 0 and stop.converged


# ------------------------------------------------- optimizer math on model problems
def _quadratic(n, complex_, seed):
    rng = np.random.default_rng(seed)
    if complex_:
        x = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        m = x @ x.conj().T + 0.5 * np.eye(n)
        b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    else:
        x = rng.standard_normal((n, n))
        m = x @ x.T + 0.5 * np.eye(n)
        b = rng.standard_normal(n)
    return m, b


def _minimize_quadratic(opt, m, b, steps):
    """f(x) = Re<x, Mx>/2 - Re<b, x>: grad = Mx - b, g = -grad, A = conj(g), exact line search.

    Returns (||g_0||, ||g_steps||, x_steps)."""
    x = np.zeros_like(b)
    g0 = np.linalg.norm(b)
    for _ in range(steps):
        g = b - m @ x
        d = opt.direction(g, np.conj(g))
        eps = real_inner(g, d) / real_inner(d, m @ d)
        x = x + eps * d
        opt.step_taken(eps)
    return g0, np.linalg.norm(b - m @ x), x


@pytest.mark.parametrize("complex_", [False, True], ids=["real", "complex"])
@pytest.mark.parametrize("make", [lambda: ConjugateGradient(), lambda: ConjugateGradient(beta="fr"),
                                  lambda: LBFGS(), lambda: LBFGS(memory=2)],
                         ids=["cg-pr+", "cg-fr", "lbfgs", "lbfgs-m2"])
def test_quadratic_finite_termination(make, complex_):
    """With exact line searches on a strictly convex quadratic, CG (either beta rule)
    terminates in at most as many steps as the real Hessian has distinct eigenvalues:
    n = 8 here, also for complex Hermitian M, whose eigenvalues are doubly degenerate as
    a real-linear map. L-BFGS with H_0 = gamma I (any gamma > 0, any memory) then gives
    directions parallel to CG's: exact line searches make g_k orthogonal to every stored
    s_i and to every y_i but the newest, so the two-loop recursion returns
    gamma (g_k - rho <y, g_k> s) for the newest pair, the Hestenes-Stiefel direction.
    Steepest descent is far from converged after the same 8 steps."""
    n = 8
    m, b = _quadratic(n, complex_, seed=4)
    g0, gn, x = _minimize_quadratic(make(), m, b, n)
    assert gn < 1e-9 * g0
    assert np.allclose(x, np.linalg.solve(m, b), atol=1e-8)
    g0, gn_sd, _ = _minimize_quadratic(SteepestDescent(), m, b, n)
    assert gn_sd > 1e-3 * g0


def test_two_loop_equals_dense_bfgs_update():
    """The two-loop direction equals H g with H from the BFGS inverse update
    H <- (1 - rho s y^T) H (1 - rho y s^T) + rho s s^T applied to H_0 = gamma I
    (gamma = <s, y>/<y, y> of the newest pair) for the stored pairs, oldest first."""
    rng = np.random.default_rng(8)
    n = 10
    m, _ = _quadratic(n, False, seed=9)
    pairs = []
    for _ in range(4):                       # secant pairs of a convex quadratic: <s, y> > 0
        s = rng.standard_normal(n)
        y = m @ s
        pairs.append((s, y, 1.0 / real_inner(s, y)))
    opt = LBFGS(memory=4)
    opt._pairs = list(pairs)
    g = rng.standard_normal(n)
    d, gamma = opt._two_loop(g)
    s_new, y_new, _ = pairs[-1]
    assert np.isclose(gamma, (s_new @ y_new) / (y_new @ y_new))
    h = gamma * np.eye(n)
    for s, y, rho in pairs:
        v = np.eye(n) - rho * np.outer(y, s)
        h = v.T @ h @ v + rho * np.outer(s, s)
    assert np.allclose(d, h @ g, rtol=1e-12, atol=1e-12)


def test_lbfgs_pair_bookkeeping_and_fallback():
    """s = eps d and y = -(g_new - g_old) are stored; eps = 0 (s = 0) and negative
    curvature are skipped; a non-descent result clears the memory and returns g."""
    opt = LBFGS(memory=3)
    g0 = np.array([1.0, 0.0, 0.0])
    d0 = opt.direction(g0, g0)
    assert d0 is g0 and opt.info["n_pairs"] == 0
    opt.step_taken(0.5)
    g1 = np.array([0.2, 0.3, 0.0])
    opt.direction(g1, g1)
    (s, y, rho), = opt._pairs
    assert np.array_equal(s, 0.5 * g0) and np.array_equal(y, -(g1 - g0))
    assert np.isclose(rho, 1.0 / real_inner(s, y)) and opt.info["pair"] == "stored"
    opt.step_taken(0.0)                     # no step: s = 0 is skipped
    opt.direction(np.array([0.1, 0.1, 0.1]), np.array([0.1, 0.1, 0.1]))
    assert opt.info["pair"] == "skipped" and opt.n_pairs == 1
    # Force a non-descent result: an A inconsistent with g (slope >= 0) resets the memory.
    g3 = np.array([0.0, 0.2, 0.1])
    d3 = opt.direction(g3, -g3)
    assert d3 is g3 and opt.info["reset"] and opt.n_pairs == 0 and opt.n_resets == 1
    # Negative curvature <s, y> < 0 is skipped.
    opt2 = LBFGS()
    opt2.direction(g0, g0)
    opt2.step_taken(1.0)
    opt2.direction(2.0 * g0, 2.0 * g0)      # y = -(2 g0 - g0) = -g0, s = g0: <s, y> < 0
    assert opt2.info["pair"] == "skipped" and opt2.n_pairs == 0 and opt2.n_skipped == 1


def test_cg_beta_rules_and_restarts():
    g0 = np.array([1.0, 1.0, 0.0])
    g1 = np.array([0.6, -0.2, 0.5])          # <g1, g1> = 0.65, <g1, g0> = 0.4, <g0, g0> = 2
    pr = ConjugateGradient()
    pr.direction(g0, g0)
    d1 = pr.direction(g1, g1)
    assert np.isclose(pr.info["beta"], 0.125) and np.allclose(d1, g1 + 0.125 * g0)
    fr = ConjugateGradient(beta="fr")
    fr.direction(g0, g0)
    d1 = fr.direction(g1, g1)
    assert np.isclose(fr.info["beta"], 0.325) and np.allclose(d1, g1 + 0.325 * g0)
    clip = ConjugateGradient()
    clip.direction(g0, g0)
    clip.direction(np.array([0.1, 0.1, 0.0]), np.array([0.1, 0.1, 0.0]))   # PR < 0: clipped
    assert clip.info == {"beta": 0.0, "restart": None}
    per = ConjugateGradient(restart=2)
    reasons = []
    for k in range(5):
        gk = np.array([1.0, 0.5 ** k, 0.25])
        per.direction(gk, gk)
        reasons.append(per.info["restart"])
    assert reasons == ["first", None, "periodic", None, "periodic"]
    slope = ConjugateGradient(beta="fr")
    slope.direction(g0, g0)
    d = slope.direction(g1, -g1)            # inconsistent A: slope >= 0 forces d = g
    assert d is g1 and slope.info["restart"] == "slope" and slope.n_restarts == 1


def test_lbfgs_curvature_test_is_relative():
    """A pair with 0 < <s, y> <= curvature_tol ||s|| ||y|| is skipped (default 1e-12),
    curvature_tol = 0 stores it, and <s, y> above the threshold is stored. Here s = d_0 =
    (1, 0, 0) and y = (delta, 1, 0): <s, y> = delta, ||s|| ||y|| = sqrt(1 + delta^2)."""
    for delta, tol, fate in ((1e-13, 1e-12, "skipped"), (1e-13, 0.0, "stored"),
                             (1e-11, 1e-12, "stored")):
        opt = LBFGS(curvature_tol=tol)
        g0 = np.array([1.0, 0.0, 0.0])
        opt.direction(g0, g0)
        opt.step_taken(1.0)
        g1 = g0 - np.array([delta, 1.0, 0.0])          # y = -(g1 - g0) = (delta, 1, 0)
        opt.direction(g1, g1)
        assert opt.info["pair"] == fate, (delta, tol)
        assert opt.n_skipped == (fate == "skipped")


# -------------------------------------------- driver-level audits of the recorded directions
def _real_coords(x):
    """Pair-order tensor -> real vector with <u, v> = Re sum conj(u) v as the dot product."""
    x = np.asarray(x)
    return np.concatenate([x.real.ravel(), x.imag.ravel()]) if np.iscomplexobj(x) else x.ravel()


def _audit_cg(steps, beta_rule, m=None, field="residual"):
    """CG directions rebuilt from the records: d_k = z_k + beta d_{k-1}, g_k = conj(A_k),
    z_k = m g_k (m: dense metric matrix on the real coordinates, None for M = I),
    beta = 0 (d = z) when the slope -<g_k, d> is not negative. ``field``: the recorded
    residual the driver acted on ("residual", or "residual_est" with a residual_estimator)."""
    dirs, g_prev, z_prev, d_prev = [], None, None, None
    for st in steps:
        g = _real_coords(np.conj(getattr(st, field)))
        z = g if m is None else m @ g
        d = z
        if d_prev is not None:
            if beta_rule == "fr":
                beta = (z @ g) / (z_prev @ g_prev)
            else:
                beta = max(0.0, z @ (g - g_prev) / (z_prev @ g_prev))
            if -(g @ (z + beta * d_prev)) < 0:
                d = z + beta * d_prev
        dirs.append(d)
        g_prev, z_prev, d_prev = g, z, d
    return dirs


def _audit_lbfgs(steps, memory, curvature_tol=1e-12, m=None):
    """L-BFGS directions rebuilt from the records with an explicit inverse-Hessian matrix:
    pairs s = eps_{k-1} d_{k-1} and y = -(g_k - g_{k-1}) (skipped when <s, y> <=
    curvature_tol ||s|| ||y||, oldest dropped beyond ``memory``), H_0 = gamma m with gamma
    = <s, y>/<y, m y> of the newest pair (m: dense metric matrix on the real coordinates, None
    for the identity), the BFGS update H <- (1 - rho s y^T) H (1 - rho y s^T) + rho s s^T oldest
    first, d = H g, and d = m g with an emptied memory if d is not descent.
    Returns (directions, number of pairs used)."""
    pairs, dirs, n_pairs, g_prev = [], [], [], None
    for k, st in enumerate(steps):
        g = _real_coords(np.conj(st.residual))
        mm = np.eye(g.size) if m is None else m
        if k > 0:
            s = steps[k - 1].eps * _real_coords(steps[k - 1].generator)
            y = -(g - g_prev)
            if s @ y > curvature_tol * np.linalg.norm(s) * np.linalg.norm(y):
                pairs = (pairs + [(s, y)])[-memory:]
        d = mm @ g
        if pairs:
            s_new, y_new = pairs[-1]
            h = (s_new @ y_new) / (y_new @ mm @ y_new) * mm
            for s, y in pairs:
                rho = 1.0 / (s @ y)
                hy = h @ y
                h = (h - rho * (np.outer(s, hy) + np.outer(hy, s))
                     + (rho + rho ** 2 * (y @ hy)) * np.outer(s, s))
            d = h @ g
            if not -(g @ d) < 0:
                pairs, d = [], mm @ g
        dirs.append(d)
        n_pairs.append(len(pairs))
        g_prev = g
    return dirs, n_pairs


@pytest.mark.parametrize("opt, opts, start", [
    ("cg", None, "real"), ("cg", {"beta": "fr"}, "real"), ("cg", None, "complex"),
    ("lbfgs", None, "real"), ("lbfgs", {"memory": 3}, "real"), ("lbfgs", {"memory": 3}, "complex"),
])
def test_recorded_directions_match_an_independent_implementation(h4, opt, opts, start):
    """Every recorded CG / L-BFGS direction of a run equals the one rebuilt from the
    records (residuals, previous directions and step sizes eps) by an independent
    implementation, to 1e-8 relative. For L-BFGS this pins the secant pair s_k = eps_k d_k
    of the step actually taken, the gradient difference y_k, the curvature skip, the
    memory limit and the descent fallback (memory 3 drops pairs within 12 steps)."""
    ci0 = None
    if start == "complex":
        ci0 = normalize(run_cqe(h4, max_iter=2).ci + 0.3j * FCISpace.of(h4).hf())
    res = run_cqe(h4, ci0=ci0, max_iter=12, optimizer=opt, optimizer_options=opts,
                  keep_generators=True, keep_residuals=True)
    assert len(res.steps) == 12
    if opt == "cg":
        dirs = _audit_cg(res.steps, (opts or {}).get("beta", "pr+"))
    else:
        dirs, n_pairs = _audit_lbfgs(res.steps, (opts or {}).get("memory", 10))
        assert n_pairs == [s.optimizer_info["n_pairs"] for s in res.steps]
        assert max(n_pairs) == min(len(res.steps) - 1, (opts or {}).get("memory", 10))
    for d, s in zip(dirs, res.steps):
        rec = _real_coords(s.generator)
        assert np.linalg.norm(d - rec) <= 1e-8 * np.linalg.norm(rec)


def _dense(space, apply):
    """Matrix of a linear map on the CI space (columns: images of the basis vectors)."""
    eye = np.eye(space.size)
    return np.column_stack([np.asarray(apply(eye[j].reshape(space.shape))).ravel()
                            for j in range(space.size)])


@pytest.mark.parametrize("start", ["real", "complex"])
def test_chart_transfer_formula(h4, start):
    """Module docstring of optimizers: along u, the derivative of the chart-k energy
    E(exp(S + t U) Psi_k) at t = 0 equals the chart-(k+1) derivative <[H, U]> plus
    <[H, [S, U]]>/2 + O(s^2), expectations in Psi_{k+1} = exp(S) Psi_k (S: a step s along
    g = conj(A); u: a random direction). Exact Frechet derivatives of expm on H4 (36
    determinants). Measured remainder/correction: 3.5e-3 (real) and 8.3e-3 (complex)
    at s = 0.01, and the remainder shrinks by 0.25-0.26 when s halves; with the opposite
    sign of the correction the error would be 2|correction|."""
    ham = Hamiltonian(h4)
    space = ham.space
    hmat = _dense(space, ham.apply)
    psi = run_cqe(h4, max_iter=2).ci.ravel()
    if start == "complex":
        psi = normalize(psi + 0.3j * space.hf().ravel())
    A = compute_residual(h4, psi.reshape(space.shape), ham).A
    rng = np.random.default_rng(3)
    n = h4.norb
    u = np.conj(symmetrize_residual(rng.standard_normal((n,) * 4) + 1j * rng.standard_normal((n,) * 4)))
    umat = _dense(space, lambda v: two_body(space, u, v))
    errors = []
    for s in (0.01, 0.005):
        smat = _dense(space, lambda v: two_body(space, s * np.conj(A), v))
        expm_s, frechet = scipy.linalg.expm_frechet(smat, umat)
        psi1 = expm_s @ psi
        chart_k = 2.0 * np.real(np.vdot(psi1, hmat @ (frechet @ psi)))
        chart_k1 = np.real(np.vdot(psi1, (hmat @ umat - umat @ hmat) @ psi1))
        su = smat @ umat - umat @ smat
        correction = 0.5 * np.real(np.vdot(psi1, (hmat @ su - su @ hmat) @ psi1))
        err = abs(chart_k - (chart_k1 + correction))
        assert err < 0.02 * abs(correction)
        errors.append(err)
    assert 0.2 < errors[1] / errors[0] < 0.32


def test_spin_square_helper(h6):
    """Regression for the <S^2> check: on (D + i HF)/sqrt(2), with D the open-shell
    determinant alpha {0,1,3} beta {0,1,2} (<S^2> = 1) and the closed-shell HF determinant
    (<S^2> = 0), <S^2> = 1/2. The check used to weight each part by its squared norm
    as well, which gives 1/4 here."""
    from pyscf.fci import cistring

    space = FCISpace.of(h6)
    norb, nelec = h6.norb, h6.nelec
    assert nelec == (3, 3)
    det = np.zeros(space.shape)
    det[cistring.str2addr(norb, 3, 0b1011), cistring.str2addr(norb, 3, 0b0111)] = 1.0
    hf = space.hf()
    assert np.isclose(spin_square(det, norb, nelec), 1.0, atol=1e-12)
    assert abs(spin_square(hf, norb, nelec)) < 1e-12
    assert np.isclose(spin_square((det + 1j * hf) / np.sqrt(2), norb, nelec), 0.5, atol=1e-12)
    assert np.isclose(spin_square(1j * det, norb, nelec), 1.0, atol=1e-12)


# ================================================== the spin-orbital metric (preconditioner)
# optimizers module docstring: M = (2 I + T)/3 with (T x)[p,q,r,s] = x[p,s,r,q]; for singlets
# sum (M g) e is the standard spin-orbital ACSE generator sum conj(2A^{ij}_{kl}) a+_i a+_j a_l a_k.

BENCH_CANDS = tuple(0.00625 * 2 ** k for k in range(11))     # benchmarks/run_optimizers.py default


def _tmap(x):
    return np.asarray(x).transpose(0, 3, 2, 1)


def _project_spin(ci, norb, nelec, keep, drop):
    """Projection of a real CI vector onto <S^2> = keep, removing the S(S+1) values in ``drop``."""
    from pyscf.fci import spin_op

    x = np.array(ci, dtype=float)
    for ss in drop:
        x = (spin_op.contract_ss(x, norb, nelec) - ss * x) / (keep - ss)
    return normalize(x)


def _frob(x, y):
    return complex(x.conj().multiply(y).sum())


def _fnorm(x):
    return float(np.sqrt(abs(_frob(x, x))))


@pytest.fixture(scope="module")
def h4_fock(h4):
    """Fock-space oracle of H4 (8 modes, 256 states): H, the pair annihilators B_ij = a_j a_i
    (so a+_i a+_j a_l a_k = B_ij^dag B_kl), every e_{pq,rs} and S^2 = S_- S_+ + S_z + S_z^2."""
    n = h4.norb
    nm = 2 * n
    fs = FockSpace(n)
    ann = [fs.a(k % n, k // n) for k in range(nm)]               # mode k = p + sigma N
    pairs = [(ann[j] @ ann[i]).tocsr() for i in range(nm) for j in range(nm)]
    s_plus = sum(fs.adag(p, 0) @ fs.a(p, 1) for p in range(n))
    sz = fs.s_z()
    return SimpleNamespace(fs=fs, nm=nm, ham=fs.hamiltonian(h4.h1, h4.eri, h4.ecore).tocsr(),
                           pairs=pairs, stack=sp.vstack(pairs).tocsr(),
                           e2={idx: fs.e2(*idx) for idx in itertools.product(range(n), repeat=4)},
                           s2=(s_plus.T @ s_plus + sz + sz @ sz).tocsr(),
                           s_plus=s_plus.tocsr(), s_minus=s_plus.T.tocsr(), s_z=sz.tocsr())


def _spin_rank_parts(fo, x):
    """(x0, x1, x2): the spin-tensor rank-0, 1 and 2 parts of a two-body operator x (ranks <= 2).
    L(x) = [S_z, [S_z, x]] + ([S_+, [S_-, x]] + [S_-, [S_+, x]])/2 is k(k+1) on rank k, so
    L x = 2 x1 + 6 x2 and L^2 x = 4 x1 + 36 x2."""
    def comm(a, b):
        return a @ b - b @ a

    def casimir(y):
        return (comm(fo.s_z, comm(fo.s_z, y))
                + 0.5 * (comm(fo.s_plus, comm(fo.s_minus, y)) + comm(fo.s_minus, comm(fo.s_plus, y))))

    lx = casimir(x)
    x1 = (6.0 * lx - casimir(lx)) / 8.0
    x2 = (lx - 2.0 * x1) / 6.0
    return x - x1 - x2, x1, x2


def _fock_residuals(fo, v):
    """(2A, A) of the Fock vector v from their definitions, with no residual code:
    2A[i,j,k,l] = <v|[a+_i a+_j a_l a_k, H]|v> over all (2N)^4 mode tuples and
    A[p,q,r,s] = <v|[e_{pq,rs}, H]|v>."""
    phi = fo.ham @ v
    bv = np.stack([b @ v for b in fo.pairs])
    bphi = np.stack([b @ phi for b in fo.pairs])
    a_so = (bv.conj() @ bphi.T - bphi.conj() @ bv.T).reshape((fo.nm,) * 4)
    a_sf = np.zeros((fo.nm // 2,) * 4, dtype=np.result_type(v, float))
    for idx, op in fo.e2.items():
        a_sf[idx] = np.vdot(v, op @ phi) - np.vdot(phi, op @ v)
    return a_so, a_sf


def _spin_orbital_operator(fo, coef):
    """sum_{ijkl} coef[i,j,k,l] a+_i a+_j a_l a_k on the whole Fock space (sparse)."""
    c = sp.csr_matrix(np.asarray(coef).reshape(fo.nm ** 2, fo.nm ** 2))
    kron = sp.kron(c, sp.identity(fo.fs.dim, format="csr"), format="csr")
    return (fo.stack.conj().T @ kron @ fo.stack).tocsr()


def test_spin_orbital_generator_identity_on_fock_space(h4, h4_fock):
    """The identity of the optimizers module docstring with its scale, as an operator identity
    on the whole Fock space of H4: G_so = sum_{ijkl} conj(2A^{ij}_{kl}) a+_i a+_j a_l a_k (the
    full sum over spin orbitals) equals sum ((2g + T g)/c) e_{pq,rs}, g = conj(A), with c = 3,
    for HF, a CQE iterate (two SD steps), a random real singlet and a complex singlet. 2A and A
    are built on the Fock space from their definitions (no residual code). Measured: the
    least-squares c is 3 to 1.3e-15 and the relative mismatch at c = 3 is <= 2.2e-15. The
    library path, apply_metric(conj(compute_residual(...).A), "spin_orbital"), gives the same
    operator; G_so commutes with S^2, and its slope -sum |2A|^2 is -<g, M g>."""
    fo = h4_fock
    space = FCISpace.of(h4)
    psi2 = run_cqe(h4, max_iter=2).ci
    states = {"hf": space.hf(), "sd-iterate": psi2,
              "random-singlet": _project_spin(space.random(np.random.default_rng(7)), h4.norb, h4.nelec,
                                              keep=0.0, drop=(2.0, 6.0)),
              "complex-singlet": normalize(psi2 + 0.3j * space.hf())}
    for label, ci in states.items():
        v = fo.fs.from_ci(ci, h4.nelec)
        v = v / np.linalg.norm(v)
        assert abs(np.vdot(v, fo.s2 @ v)) < 1e-12, label
        a_so, a_sf = _fock_residuals(fo, v)
        g = np.conj(a_sf)
        g_so = _spin_orbital_operator(fo, np.conj(a_so))
        x = fo.fs.two_body(2.0 * g + _tmap(g))
        c = _frob(x, x).real / _frob(x, g_so).real
        scale = _fnorm(g_so)
        assert abs(c - 3.0) < 1e-12, (label, c)
        assert _fnorm(g_so - x / 3.0) < 1e-12 * scale, label
        lib = fo.fs.two_body(apply_metric(descent_generator(compute_residual(h4, ci).A), "spin_orbital"))
        assert _fnorm(g_so - lib) < 1e-12 * scale, label
        assert _fnorm(g_so @ fo.s2 - fo.s2 @ g_so) < 1e-12 * scale, label
        assert np.isclose(-np.sum(np.abs(a_so) ** 2), -real_inner(g, apply_metric(g, "spin_orbital")),
                          rtol=1e-12, atol=0.0), label


def test_fixed_spin_orbital_steps_match_the_fock_space_cqe(h4, h4_fock):
    """The scale matters only for fixed step sizes: three steps of eps = 0.15 (the only
    candidate) with metric="spin_orbital" equal the standard spin-orbital CQE run on the Fock
    space, psi <- exp(eps G_so) psi with G_so = sum conj(2A^{ij}_{kl}) a+_i a+_j a_l a_k rebuilt
    from 2A at every step: same energies and states to 1e-9 (the Krylov tolerance is 1e-10)."""
    import scipy.sparse.linalg as spla

    fo = h4_fock
    eps = 0.15
    states = []
    res = run_cqe(h4, max_iter=3, eps_candidates=(eps,), metric="spin_orbital",
                  callback=lambda step, ci: states.append(ci))
    v = fo.fs.from_ci(FCISpace.of(h4).hf(), h4.nelec)
    for step, ci in zip(res.steps, states):
        a_so, _ = _fock_residuals(fo, v)
        v = spla.expm_multiply(eps * _spin_orbital_operator(fo, np.conj(a_so)), v)
        v = v / np.linalg.norm(v)
        assert step.eps == eps
        assert abs(np.vdot(v, fo.ham @ v).real - step.energy_after) < 1e-9
        w = fo.fs.from_ci(ci, h4.nelec)
        assert abs(abs(np.vdot(v, w)) - 1.0) < 1e-9
    assert len(states) == 3 and res.energies[-1] < res.energies[0] - 0.05


def test_spin_orbital_identity_needs_a_singlet(h4_fock):
    """For non-singlet states the identity fails: for a random triplet (M_S = 0) and a random
    doublet of H4 the spin-orbital generator G_so differs from sum (M g) e by 46% and 73%
    (relative Frobenius norm, measured). sum (M g) e is then the spin average of G_so: the
    remainder has no spin-tensor rank-0 part, and it is orthogonal to every e_{pq,rs}
    (measured <= 1e-16 relative), so sum (M g) e is the Frobenius projection of G_so onto the
    spin-free operators. The remainder is pure rank 2 for the triplet (rank 1 drops out at
    M_S = 0: <S 0; 1 0|S 0> = 0) and pure rank 1 for the doublet (rank 2 needs S >= 1). G_so does
    not commute with S^2 (relative commutator 1.38 and 1.35), sum (M g) e does, and the slope
    of G_so is steeper (-5.00 against -2.97 and -3.43 against -1.87)."""
    fo = h4_fock
    rng = np.random.default_rng(8)
    triplet = _project_spin(FCISpace(4, (2, 2)).random(rng), 4, (2, 2), keep=2.0, drop=(0.0, 6.0))
    doublet = _project_spin(FCISpace(4, (2, 1)).random(rng), 4, (2, 1), keep=0.75, drop=(3.75,))
    for label, ci, nelec, ss in (("triplet", triplet, (2, 2), 2.0), ("doublet", doublet, (2, 1), 0.75)):
        v = fo.fs.from_ci(ci, nelec)
        v = v / np.linalg.norm(v)
        assert np.isclose(np.vdot(v, fo.s2 @ v).real, ss, atol=1e-10)
        a_so, a_sf = _fock_residuals(fo, v)
        g = np.conj(a_sf)
        g_so = _spin_orbital_operator(fo, np.conj(a_so))
        g_sf = fo.fs.two_body(apply_metric(g, "spin_orbital"))
        rem = g_so - g_sf
        scale = _fnorm(g_so)
        assert _fnorm(rem) > 0.3 * scale, label
        overlap = max(abs(_frob(op, rem)) / _fnorm(op) for op in fo.e2.values() if op.nnz)
        assert overlap < 1e-12 * scale, label
        r0, r1, r2 = _spin_rank_parts(fo, rem)
        present, absent = (r2, r1) if label == "triplet" else (r1, r2)
        assert _fnorm(r0) < 1e-12 * scale and _fnorm(absent) < 1e-12 * scale, label
        assert abs(_fnorm(present) - _fnorm(rem)) < 1e-12 * scale, label
        assert _fnorm(g_so @ fo.s2 - fo.s2 @ g_so) > 0.3 * scale, label
        assert _fnorm(g_sf @ fo.s2 - fo.s2 @ g_sf) < 1e-12 * _fnorm(g_sf), label
        assert -np.sum(np.abs(a_so) ** 2) < -1.2 * real_inner(g, apply_metric(g, "spin_orbital")), label


def _contract_spin_dense(c):
    """c[p,q,r,s] = sum_{sigma tau} C[(p sigma), (r tau), (q sigma), (s tau)] of a dense (2N)^4 tensor
    (modes p + sigma N), any dtype; measure.targets.contract_spin does this for real element values."""
    nm = c.shape[0]
    n = nm // 2
    return np.einsum("apbraqbs->pqrs", c.reshape(2, n, 2, n, 2, n, 2, n))


def test_metric_direction_of_an_estimate_is_the_spin_average(h4, h4_fock):
    """optimizers module docstring ("Estimated residuals"): for any coefficient tensor C that is
    antisymmetric in (i, j) and in (k, l), the spin average (rank-0 part) of
    sum C_{ijkl} a+_i a+_j a_l a_k is sum (M c) e with c the spin contraction of C. So for a
    spin-orbital residual estimate 2A_est with A_est = contract_spin(2A_est), the
    "spin_orbital" direction sum (M conj(A_est)) e is the spin average of the estimated generator
    sum conj(2A_est) a+a+aa, with no state involved: here random estimates with the structure of
    measure.targets (real, S_z-conserving), and generic complex ones with S_z-violating entries
    (anti-Hermitian or not). The raw estimated generator has rank-1 and rank-2 parts and does
    not commute with S^2 (measured: relative commutator 1.66-1.80; rank-0 agreement 1.7e-16)."""
    from thc_engine_cqe.measure.targets import contract_spin, spin_orbital_elements, spin_orbital_tensor

    fo = h4_fock
    n, nm = h4.norb, h4_fock.nm
    rng = np.random.default_rng(2026)
    els = spin_orbital_elements(n)
    vals = rng.standard_normal(len(els))
    structured = spin_orbital_tensor(vals, els, n)
    assert np.allclose(_contract_spin_dense(structured), contract_spin(vals, els, n), rtol=0, atol=1e-14)
    generic = rng.standard_normal((nm,) * 4) + 1j * rng.standard_normal((nm,) * 4)
    generic = (generic - generic.transpose(1, 0, 2, 3)) / 2
    generic = (generic - generic.transpose(0, 1, 3, 2)) / 2
    anti_hermitian = (generic - np.conj(generic.transpose(2, 3, 0, 1))) / 2
    for label, a_so in (("targets", structured), ("generic", generic), ("anti-Hermitian", anti_hermitian)):
        g_est = np.conj(_contract_spin_dense(a_so))
        g_raw = _spin_orbital_operator(fo, np.conj(a_so))
        r0, r1, r2 = _spin_rank_parts(fo, g_raw)
        scale = _fnorm(g_raw)
        assert _fnorm(r0 - fo.fs.two_body(apply_metric(g_est, "spin_orbital"))) < 1e-12 * scale, label
        assert _fnorm(r1) > 0.1 * scale and _fnorm(r2) > 0.1 * scale, label
        assert _fnorm(g_raw @ fo.s2 - fo.s2 @ g_raw) > 0.5 * scale, label


def _structured_basis(n, complex_):
    """Orthonormal basis (columns, in _real_coords) of the structured space V of (n,)^4 tensors."""
    dim = n ** 4
    units = [np.eye(dim)[k].reshape((n,) * 4) for k in range(dim)]
    if complex_:
        units = [u.astype(complex) for u in units] + [1j * u for u in units]
    cols = np.column_stack([_real_coords(symmetrize_residual(u)) for u in units])
    u, s, _ = np.linalg.svd(cols, full_matrices=False)
    return u[:, s > 1e-8]


def _from_coords(c, n, complex_):
    c = np.asarray(c)
    dim = n ** 4
    x = c[:dim] + 1j * c[dim:] if complex_ else c
    return x.reshape((n,) * 4)


@pytest.mark.parametrize("complex_", [False, True], ids=["real", "complex"])
def test_spin_orbital_metric_is_spd_on_the_structured_space(complex_, rng):
    """M = (2I + T)/3 maps the structured space V (pair symmetry, x[q,p,s,r] = -conj(x[p,q,r,s]);
    N = 4: real dimension N^2 (N^2 - 1)/4 = 60 for real tensors, plus N^2 (N^2 + 3)/4 = 76
    imaginary directions for complex ones) to itself, commutes with the projector
    symmetrize_residual, is symmetric in <u, v> = Re sum conj(u) v, and its eigenvalues on V are
    3/c = 1 (T = +1) and 1/c = 1/3 (T = -1), both present, so it is positive definite. T commutes
    with the pair exchange, and with eta: x -> conj(x[q,p,s,r]) on pair-symmetric tensors but not
    on general ones (module docstring)."""
    n = 4
    basis = _structured_basis(n, complex_)
    assert basis.shape[1] == (136 if complex_ else 60)
    images = np.column_stack([_real_coords(apply_metric(_from_coords(basis[:, j], n, complex_), "spin_orbital"))
                              for j in range(basis.shape[1])])
    assert np.abs(images - basis @ (basis.T @ images)).max() < 1e-14         # M V = V
    for _ in range(3):                                                        # exact structure kept
        x = rng.standard_normal((n,) * 4) + (1j * rng.standard_normal((n,) * 4) if complex_ else 0.0)
        mx = apply_metric(symmetrize_residual(x), "spin_orbital")
        assert max(structure_error(mx)) <= 1e-15 * np.abs(mx).max()
    mv = basis.T @ images
    assert np.abs(mv - mv.T).max() < 1e-15                                    # symmetric
    w = np.linalg.eigvalsh(mv)
    tv = basis.T @ np.column_stack([_real_coords(_tmap(_from_coords(basis[:, j], n, complex_)))
                                    for j in range(basis.shape[1])])
    t = np.linalg.eigvalsh(0.5 * (tv + tv.T))
    n_plus, n_minus = int(np.sum(t > 0)), int(np.sum(t < 0))
    assert n_plus > 0 and n_minus > 0 and n_plus + n_minus == basis.shape[1]
    assert np.allclose(np.sort(w), np.r_[np.full(n_minus, 1.0 / 3.0), np.ones(n_plus)], atol=1e-14)
    x = rng.standard_normal((n,) * 4) + (1j * rng.standard_normal((n,) * 4) if complex_ else 0.0)
    assert np.allclose(apply_metric(symmetrize_residual(x), "spin_orbital"),
                       symmetrize_residual(apply_metric(x, "spin_orbital")), atol=1e-15)

    def eta(y):
        return np.conj(y.transpose(1, 0, 3, 2))

    xp = 0.5 * (x + x.transpose(2, 3, 0, 1))                                   # pair-symmetric
    assert np.array_equal(_tmap(x.transpose(2, 3, 0, 1)), _tmap(x).transpose(2, 3, 0, 1))
    assert np.array_equal(_tmap(eta(xp)), eta(_tmap(xp)))
    assert np.abs(_tmap(eta(x)) - eta(_tmap(x))).max() > 0.1 * np.abs(x).max()


@pytest.mark.parametrize("complex_", [False, True], ids=["real", "complex"])
def test_spin_orbital_metric_is_the_spin_orbital_amplitude_metric(complex_, rng):
    """M^-1 is the Gram matrix of the e_{pq,rs} in the Euclidean inner product of spin-orbital
    amplitudes: the antisymmetrized spin-orbital coefficients C of sum c e_{pq,rs}
    (C_{(p s)(r t)(q s)(s t)} = c[p,q,r,s] before antisymmetrization in (ij) and (kl)) have
    sum |C|^2 = <c, M^-1 c> for every structured c, with no state involved. So "spin_orbital"
    steepest descent is steepest descent in the metric of the spin-orbital amplitudes, which
    fixes the scale: for a singlet, C(M g) = conj(2A) and sum |2A|^2 = <g, M g>."""
    n = 4
    for _ in range(3):
        x = rng.standard_normal((n,) * 4) + (1j * rng.standard_normal((n,) * 4) if complex_ else 0.0)
        c = symmetrize_residual(x)
        c0 = np.zeros((2, n, 2, n, 2, n, 2, n), dtype=c.dtype)        # [s_i, p, s_j, r, s_k, q, s_l, s]
        for s1, s2 in itertools.product(range(2), repeat=2):
            c0[s1, :, s2, :, s1, :, s2, :] = c.transpose(0, 2, 1, 3)
        c0 = c0.reshape((2 * n,) * 4)
        amp = (c0 - c0.transpose(1, 0, 2, 3) - c0.transpose(0, 1, 3, 2) + c0.transpose(1, 0, 3, 2)) / 4.0
        m_inv_c = (c + _tmap(c)) / 2.0 + 3.0 * (c - _tmap(c)) / 2.0         # M^-1: 1 on T = +1, 3 on T = -1
        assert np.allclose(apply_metric(m_inv_c, "spin_orbital"), c, atol=1e-14)
        assert np.isclose(np.sum(np.abs(amp) ** 2), real_inner(c, m_inv_c), rtol=1e-13, atol=0.0)


@pytest.mark.parametrize("cands", [None, BENCH_CANDS], ids=["exact", "candidates"])
def test_spin_orbital_sd_is_the_metric_compressed_run(h4, cands):
    """SD with metric="spin_orbital" equals the default driver with compress = M, bit for bit,
    exact line search and fixed candidates; its captured descent <g, M g>/<g, g> is in [1/3, 1]."""
    ref = run_cqe(h4, max_iter=10, eps_candidates=cands, keep_generators=True, keep_residuals=True,
                  compress=lambda g, rd, it: apply_metric(g, "spin_orbital"))
    res = run_cqe(h4, max_iter=10, eps_candidates=cands, keep_generators=True, keep_residuals=True,
                  metric="spin_orbital")
    assert_identical_runs(ref, res)
    assert res.metric == "spin_orbital" and ref.metric == "spin_free" and res.optimizer == "sd"
    assert all(s.optimizer_info == {"metric": "spin_orbital"} for s in res.steps)
    assert all(1.0 / 3.0 - 1e-12 <= s.captured_descent <= 1.0 + 1e-12 for s in res.steps)
    assert all(s.slope < 0 for s in res.steps)


@pytest.mark.parametrize("cands", [None, CANDS], ids=["exact", "candidates"])
@pytest.mark.parametrize("opt", ["cg", "lbfgs"])
def test_first_spin_orbital_step_equals_sd(h4, opt, cands):
    sd = run_cqe(h4, max_iter=1, keep_generators=True, keep_residuals=True, eps_candidates=cands,
                 metric="spin_orbital")
    res = run_cqe(h4, max_iter=1, optimizer=opt, keep_generators=True, keep_residuals=True,
                  eps_candidates=cands, metric="spin_orbital")
    assert_identical_runs(sd, res)
    assert res.metric == "spin_orbital" and res.steps[0].optimizer_info["metric"] == "spin_orbital"


def test_spin_orbital_sd_reproduces_the_h6_claims():
    """H6/STO-3G at 1.0 A (the geometry of the earlier claims): steepest descent along the
    spin-orbital generator is 16.995 and 6.5577 mHa above FCI after steps 1 and 2, against
    37.956 and 19.572 mHa for the spin-free SD (measured 16.99528, 6.55773, 37.95612, 19.57169)."""
    h6a = systems.hydrogen_chain(6, 1.0, "angstrom")
    e_fci = h6a.fci()[0]
    so = run_cqe(h6a, max_iter=2, metric="spin_orbital", e_exact=e_fci).errors
    sf = run_cqe(h6a, max_iter=2, e_exact=e_fci).errors
    assert abs(so[2] - 6.5577e-3) < 1e-6 and abs(so[1] - 16.995e-3) < 1e-6
    assert abs(sf[2] - 19.572e-3) < 1e-6 and abs(sf[1] - 37.956e-3) < 1e-6


@pytest.mark.parametrize("opt", ["sd", "cg", "lbfgs"])
@pytest.mark.parametrize("start", ["real", "complex"])
def test_spin_orbital_directions_keep_structure_and_singlet(h6, opt, start):
    """With metric="spin_orbital", every direction is pair-symmetric and anti-Hermitian to 1e-12
    (relative), a descent direction (exact slope < 0), and <S^2> stays 0; energies decrease."""
    space = FCISpace.of(h6)
    ci0 = None
    if start == "complex":
        ci0 = normalize(run_cqe(h6, max_iter=2).ci + 0.3j * space.hf())
    states = []
    res = run_cqe(h6, ci0=ci0, max_iter=8, optimizer=opt, metric="spin_orbital", keep_generators=True,
                  keep_residuals=True, callback=lambda step, ci: states.append(ci))
    assert len(res.steps) == 8 and res.metric == "spin_orbital"
    for s in res.steps:
        d = s.generator
        assert np.iscomplexobj(d) == (start == "complex")
        pair, herm = structure_error(d)
        scale = np.abs(d).max()
        assert pair <= 1e-12 * scale and herm <= 1e-12 * scale
        assert s.slope < 0 and direction_slope(d, s.residual) == s.slope
        assert s.optimizer_info["metric"] == "spin_orbital"
    for ci in states:
        assert abs(spin_square(ci, h6.norb, h6.nelec)) < 1e-10
    assert np.all(np.diff(res.energies) <= 1e-12)


@pytest.mark.parametrize("system, counts", [
    ("h4", {"cg": (3, 7), "lbfgs": (3, 7)}),
    ("h6", {"cg": (4, 38), "lbfgs": (4, 36)}),
])
def test_preconditioned_cg_and_lbfgs_step_counts(request, fci_energy, system, counts):
    """Regression: steps to E - E_FCI < 1.6 mHa and < 1e-5 Ha with metric="spin_orbital", exact
    line search: H4 CG 3/7, L-BFGS 3/7 (spin-free 3/9, 3/9); H6 CG 4/38, L-BFGS 4/36 (spin-free
    6/51, 6/47). The 1.6 mHa counts are asserted exactly, the 1e-5 counts within one step."""
    act = request.getfixturevalue(system)
    for opt, (n_chem, n_tight) in counts.items():
        res = run_cqe(act, max_iter=n_tight + 1, optimizer=opt, metric="spin_orbital",
                      e_exact=fci_energy[system])
        assert iterations_to(res.errors, 1.6e-3) == n_chem, opt
        n = iterations_to(res.errors, 1e-5)
        assert n is not None and abs(n - n_tight) <= 1, (opt, n)


def _dense_spin_orbital_metric(n, complex_):
    """M as a dense matrix on _real_coords, built from an index permutation (not apply_metric)."""
    perm = np.arange(n ** 4).reshape((n,) * 4).transpose(0, 3, 2, 1).ravel()
    eye = np.eye(n ** 4)
    m = (2.0 * eye + eye[perm]) / 3.0
    return scipy.linalg.block_diag(m, m) if complex_ else m


class _NoisyResidual:
    """residual_estimator: symmetrize_residual(A + sigma N(0, 1)), seeded with (seed, iteration)."""

    def __init__(self, sigma, seed):
        self.sigma, self.seed = sigma, seed

    def __call__(self, rd, ci, it):
        noise = np.random.default_rng([self.seed, it]).standard_normal(rd.A.shape)
        return symmetrize_residual(rd.A + self.sigma * noise)


class _NoisyEnergy:
    """energy_estimator: E + sigma N(0, 1), seeded with (seed, call number)."""

    def __init__(self, sigma, seed):
        self.sigma, self.seed, self.calls = sigma, seed, 0

    def __call__(self, e, ci, context):
        self.calls += 1
        return e + self.sigma * np.random.default_rng([self.seed, self.calls]).standard_normal()


# Candidates with a negative and a zero step, as in a noisy (hardware-like) run.
NOISY_CANDS = (-0.05, 0.0) + tuple(0.0125 * 2 ** k for k in range(9))


@pytest.mark.parametrize("opt, opts, start", [
    ("cg", None, "real"), ("cg", {"beta": "fr"}, "real"), ("cg", None, "complex"),
    ("cg", {"beta": "fr"}, "noisy"),
    ("lbfgs", None, "real"), ("lbfgs", {"memory": 3}, "complex"),
])
def test_recorded_metric_directions_match_an_independent_implementation(h4, opt, opts, start):
    """With metric="spin_orbital", every recorded direction equals the one rebuilt from the
    records by the dense audits above with the metric matrix m: preconditioned CG
    (z = m g, beta from z, restarts d = z) and L-BFGS with H_0 = gamma m, gamma = <s, y>/<y, m y>
    and the unpreconditioned secant pairs, to 1e-8 relative.

    After an exact line search with the exact residual the conjugate direction is a descent
    direction (to the accuracy of the search), and no benchmark run restarts, so the CG slope
    restart is exercised with noise. "noisy": Fletcher-Reeves CG driven by a
    residual estimate (A + 0.03 N(0, 1), symmetrized) and energy estimates (0.003 N(0, 1)) over
    candidates with a negative and a zero step; measured, its conjugate direction was uphill for
    the estimated slope at steps 7 and 22 (by 0.40 and 0.17 <g, M g>), where the restart
    direction is z = M conj(A_est), not g = conj(A_est), bit for bit."""
    ci0, kw = None, {}
    if start == "complex":
        ci0 = normalize(run_cqe(h4, max_iter=2).ci + 0.3j * FCISpace.of(h4).hf())
    n_steps = 25 if start == "noisy" else 12
    if start == "noisy":
        kw = dict(e_tol=0.0, eps_candidates=NOISY_CANDS, residual_estimator=_NoisyResidual(0.03, 11),
                  energy_estimator=_NoisyEnergy(0.003, 111))
    res = run_cqe(h4, ci0=ci0, max_iter=n_steps, optimizer=opt, optimizer_options=opts, metric="spin_orbital",
                  keep_generators=True, keep_residuals=True, **kw)
    assert len(res.steps) == n_steps
    m = _dense_spin_orbital_metric(h4.norb, start == "complex")
    if opt == "cg":
        dirs = _audit_cg(res.steps, (opts or {}).get("beta", "pr+"), m=m,
                         field="residual_est" if start == "noisy" else "residual")
    else:
        dirs, n_pairs = _audit_lbfgs(res.steps, (opts or {}).get("memory", 10), m=m)
        assert n_pairs == [s.optimizer_info["n_pairs"] for s in res.steps]
    for d, s in zip(dirs, res.steps):
        rec = _real_coords(s.generator)
        assert np.linalg.norm(d - rec) <= 1e-8 * np.linalg.norm(rec)
    if start == "noisy":
        restarts = [s for s in res.steps if s.optimizer_info["restart"] == "slope"]
        assert restarts
        for s in restarts:
            g = np.conj(s.residual_est)
            assert np.array_equal(s.generator, apply_metric(g, "spin_orbital"))
            assert np.linalg.norm(s.generator - g) > 0.1 * np.linalg.norm(g)


def _metric_quadratic(k_eigs, seed):
    """f(theta) = <theta, H theta>/2 - <b, theta> on the real structured space V of N = 3 tensors
    (dimension 18), with H = M^(-1/2) K M^(-1/2) in V-coordinates (M: the spin-orbital metric on V)
    and K symmetric with the eigenvalues k_eigs (cycled over V). Returns (basis, H, b)."""
    n = 3
    basis = _structured_basis(n, False)
    r = basis.shape[1]
    mv = basis.T @ np.column_stack([apply_metric(basis[:, j].reshape((n,) * 4), "spin_orbital").ravel()
                                    for j in range(r)])
    w, u = np.linalg.eigh(mv)
    m_inv_half = u @ np.diag(w ** -0.5) @ u.T
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((r, r)))
    k = q @ np.diag(np.resize(np.asarray(k_eigs, dtype=float), r)) @ q.T
    return basis, m_inv_half @ k @ m_inv_half, rng.standard_normal(r)


def _run_metric_quadratic(opt, basis, h, b, steps):
    """Exact line searches on the quadratic, the optimizer acting on (3,)^4 tensors in V.
    Returns (||g_0||, ||g_steps||, directions in V-coordinates)."""
    x = np.zeros_like(b)
    dirs = []
    for _ in range(steps):
        g = b - h @ x
        gt = (basis @ g).reshape((3,) * 4)
        d = basis.T @ np.asarray(opt.direction(gt, gt)).ravel()
        eps = (g @ d) / (d @ h @ d)
        x = x + eps * d
        opt.step_taken(eps)
        dirs.append(d)
    return np.linalg.norm(b), np.linalg.norm(b - h @ x), dirs


@pytest.mark.parametrize("make", [lambda m: ConjugateGradient(metric=m),
                                  lambda m: ConjugateGradient(beta="fr", metric=m),
                                  lambda m: LBFGS(metric=m), lambda m: LBFGS(memory=2, metric=m)],
                         ids=["cg-pr+", "cg-fr", "lbfgs", "lbfgs-m2"])
def test_metric_preconditions_a_model_quadratic(make):
    """On a quadratic whose preconditioned Hessian M^(1/2) H M^(1/2) = K has 3 distinct
    eigenvalues, preconditioned CG (either beta rule) and L-BFGS with H_0 = gamma M terminate
    in 3 exact line searches, with parallel directions (L-BFGS = preconditioned CG on
    quadratics), while the unpreconditioned versions are far from converged (H itself has 18
    distinct eigenvalues). With K = I (H = M^-1) preconditioned SD, CG and L-BFGS all reach the
    minimum in one step (the first direction M g is the Newton step)."""
    basis, h, b = _metric_quadratic((1.0, 2.5, 7.0), seed=3)
    g0, gn, dirs = _run_metric_quadratic(make("spin_orbital"), basis, h, b, 3)
    assert gn < 1e-9 * g0
    _, _, ref = _run_metric_quadratic(ConjugateGradient(metric="spin_orbital"), basis, h, b, 3)
    for d, e in zip(dirs, ref):
        assert abs(d @ e) >= (1 - 1e-10) * np.linalg.norm(d) * np.linalg.norm(e)
    _, gn_plain, _ = _run_metric_quadratic(make("spin_free"), basis, h, b, 3)
    assert gn_plain > 1e-2 * g0
    basis, h, b = _metric_quadratic((1.0,), seed=4)
    for opt in (make("spin_orbital"), SteepestDescent(metric="spin_orbital")):
        g0, gn, _ = _run_metric_quadratic(opt, basis, h, b, 1)
        assert gn < 1e-12 * g0
    _, gn_plain, _ = _run_metric_quadratic(SteepestDescent(), basis, h, b, 1)
    assert gn_plain > 1e-2 * g0


def test_metric_restart_and_reset_return_m_g():
    """With metric="spin_orbital" the CG slope restart and the L-BFGS reset return z = M g, not g.
    The driver passes A consistent with g = conj(A); there the L-BFGS matrix H is symmetric
    positive definite (H_0 = gamma M, gamma > 0, stored pairs <s, y> > 0), so its reset only guards
    against round-off (LBFGS docstring). Both branches are forced here with an A inconsistent
    with g (slope -Re sum d A = <g, d> > 0), as in test_lbfgs_pair_bookkeeping_and_fallback."""
    rng = np.random.default_rng(5)
    g0, r1, g2 = (symmetrize_residual(rng.standard_normal((3,) * 4)) for _ in range(3))
    g1 = 0.3 * g0 + 0.1 * r1
    so = "spin_orbital"
    for g in (g1, g2):                        # M g != g: the test tells z from g
        assert np.linalg.norm(apply_metric(g, so) - g) > 0.1 * np.linalg.norm(g)
    cg = ConjugateGradient(beta="fr", metric=so)
    cg.direction(g0, g0)
    d = cg.direction(g1, -g1)                 # the conjugate direction is uphill for this A
    assert np.array_equal(d, apply_metric(g1, so)) and cg.n_restarts == 1
    assert cg.info == {"beta": 0.0, "restart": "slope", "metric": so}
    lb = LBFGS(metric=so)
    lb.direction(g0, g0)
    lb.step_taken(0.1)
    d1 = lb.direction(g1, g1)                 # consistent A: a pair is stored, d1 = H g1 is descent
    assert lb.info["pair"] == "stored" and not lb.info["reset"] and real_inner(g1, d1) > 0
    lb.step_taken(0.1)
    d = lb.direction(g2, -g2)                 # H g2 is uphill for this A: reset
    assert np.array_equal(d, apply_metric(g2, so))
    assert lb.n_resets == 1 and lb.n_pairs == 0
    assert lb.info["reset"] is True and lb.info["gamma"] is None and lb.info["n_pairs"] == 0
    assert lb.info["metric"] == so


def test_metric_arguments(h4):
    assert METRICS == ("spin_free", "spin_orbital")
    x = np.zeros((h4.norb,) * 4)
    assert apply_metric(x) is x and apply_metric(x, "spin_free") is x
    y = np.random.default_rng(3).standard_normal((3,) * 4)       # no symmetry: the documented formula
    my = apply_metric(y, "spin_orbital")
    assert all(math.isclose(my[p, q, r, s], (2.0 * y[p, q, r, s] + y[p, s, r, q]) / 3.0, rel_tol=1e-15)
               for p, q, r, s in itertools.product(range(3), repeat=4))
    with pytest.raises(ValueError, match="metric must be one of"):
        run_cqe(h4, max_iter=1, metric="mazziotti")
    with pytest.raises(ValueError, match="metric must be one of"):
        make_optimizer("cg", metric="spin-orbital")
    with pytest.raises(ValueError, match="metric must be one of"):
        LBFGS(metric=None)
    with pytest.raises(ValueError, match="not an optimizer option"):
        run_cqe(h4, max_iter=1, optimizer="cg", optimizer_options={"metric": "spin_orbital"})
    with pytest.raises(ValueError, match="pair-order tensors"):
        apply_metric(np.zeros(5), "spin_orbital")
    with pytest.raises(NotImplementedError, match="metric='spin_free'"):
        run_cqe(h4, max_iter=1, metric="spin_orbital", compress=projection_compressor(n_leaves=4))
    for opt in ("cg", "lbfgs"):
        with pytest.raises(NotImplementedError):
            run_cqe(h4, max_iter=1, optimizer=opt, metric="spin_orbital", compress=lambda g, rd, it: g)
    assert run_cqe(h4, max_iter=1).metric == "spin_free"
    for name in ("sd", "cg", "lbfgs"):
        o = make_optimizer(name, metric="spin_orbital")
        assert o.metric == "spin_orbital" and make_optimizer(name).metric == "spin_free"


# --------------------------------------------- the default metric reproduces M2 bit for bit
# The CG and L-BFGS direction code at the end of M2 (commit 5a89cba), before the metric
# existed, copied so that "the default metric reproduces M2 bit for bit" is checked against
# the M2 code itself (the driver is shared; the frozen M1 driver above covers steepest descent).

class _M2ConjugateGradient:
    name, metric = "cg", "spin_free"          # metric: read by the driver

    def __init__(self, beta="pr+", restart=None):
        self.beta_rule, self.restart = beta, restart
        self._g_prev = self._d_prev = None
        self._cycle = 0
        self.info = None

    def step_taken(self, eps):
        pass

    def direction(self, g, A):
        g = np.asarray(g)
        beta, reason = 0.0, None
        if self._d_prev is None:
            reason = "first"
        elif self.restart is not None and self._cycle >= self.restart:
            reason = "periodic"
        else:
            gg_prev = real_inner(self._g_prev, self._g_prev)
            if gg_prev <= 0.0:
                reason = "zero"
            elif self.beta_rule == "fr":
                beta = real_inner(g, g) / gg_prev
            else:
                beta = max(0.0, real_inner(g, g - self._g_prev) / gg_prev)
        if reason is None:
            d = g + beta * self._d_prev
            if not direction_slope(d, A) < 0.0:
                reason, beta, d = "slope", 0.0, g
        else:
            d = g
        self._cycle = self._cycle + 1 if reason is None else 1
        self._g_prev, self._d_prev = g.copy(), d.copy()
        self.info = {"beta": float(beta), "restart": reason}
        return d


class _M2LBFGS:
    name, metric = "lbfgs", "spin_free"

    def __init__(self, memory=10, curvature_tol=1e-12):
        self.memory, self.curvature_tol = memory, curvature_tol
        self._pairs, self._g_prev, self._d_prev, self._eps_prev = [], None, None, None
        self.info = None

    def step_taken(self, eps):
        self._eps_prev = float(eps)

    def _two_loop(self, g):
        q = np.array(g, dtype=np.result_type(g, *(s for s, _, _ in self._pairs)), copy=True)
        alphas = []
        for s, y, rho in reversed(self._pairs):
            a = rho * real_inner(s, q)
            q = q - a * y
            alphas.append(a)
        s, y, _ = self._pairs[-1]
        gamma = real_inner(s, y) / real_inner(y, y)
        r = gamma * q
        for (s, y, rho), a in zip(self._pairs, reversed(alphas)):
            b = rho * real_inner(y, r)
            r = r + (a - b) * s
        return r, gamma

    def direction(self, g, A):
        g = np.asarray(g)
        info = {"n_pairs": 0, "pair": None, "reset": False, "gamma": None}
        if self._g_prev is not None and self._eps_prev is not None:
            s = self._eps_prev * self._d_prev
            y = -(g - self._g_prev)
            sy = real_inner(s, y)
            if sy > self.curvature_tol * np.linalg.norm(s) * np.linalg.norm(y):
                self._pairs.append((s, y, 1.0 / sy))
                if len(self._pairs) > self.memory:
                    self._pairs.pop(0)
                info["pair"] = "stored"
            else:
                info["pair"] = "skipped"
        if self._pairs:
            d, gamma = self._two_loop(g)
            info["gamma"] = float(gamma)
            if not direction_slope(d, A) < 0.0:
                self._pairs.clear()
                info.update(reset=True, gamma=None)
                d = g
        else:
            d = g
        info["n_pairs"] = len(self._pairs)
        self._g_prev, self._d_prev, self._eps_prev = g.copy(), d.copy(), None
        self.info = info
        return d


@pytest.mark.parametrize("opt, opts, start, cands", [
    ("cg", None, "real", None), ("cg", {"beta": "fr"}, "real", None), ("cg", {"restart": 3}, "real", None),
    ("cg", None, "real", CANDS), ("cg", None, "complex", None),
    ("lbfgs", None, "real", None), ("lbfgs", {"memory": 3}, "real", None), ("lbfgs", None, "real", CANDS),
    ("lbfgs", None, "complex", None),
])
def test_default_metric_reproduces_m2_optimizers_bit_for_bit(h4, monkeypatch, opt, opts, start, cands):
    """Every recorded number, direction, residual, optimizer_info and the final state of a
    default-metric CG / L-BFGS run equal those of the same run with the frozen M2 optimizers."""
    ci0 = None
    if start == "complex":
        ci0 = normalize(run_cqe(h4, max_iter=2).ci + 0.3j * FCISpace.of(h4).hf())
    kw = dict(ci0=ci0, max_iter=14, optimizer=opt, optimizer_options=opts, eps_candidates=cands,
              keep_generators=True, keep_residuals=True)
    res = run_cqe(h4, **kw)
    frozen = {"cg": _M2ConjugateGradient, "lbfgs": _M2LBFGS}
    monkeypatch.setattr(cqe_module, "make_optimizer",
                        lambda name, options=None, **unused: frozen[name](**(options or {})))
    ref = run_cqe(h4, **kw)
    assert_identical_runs(ref, res)
    assert [s.optimizer_info for s in ref.steps] == [s.optimizer_info for s in res.steps]
    assert res.metric == "spin_free"


# Energies (float.hex) of default runs on H4 (1.8 bohr) recorded with the M2 code (commit
# 5a89cba, one thread) before the metric was added; "cands" is the benchmark's default
# candidate set 0.00625 * 2^k, k = 0..10.
M2_ENERGIES = {
    ("sd", None): [
        "-0x1.0e84d6b471c53p+1", "-0x1.13b33857c3173p+1", "-0x1.1546a66e4a10fp+1", "-0x1.15e68316b8751p+1",
        "-0x1.162ca28db4437p+1", "-0x1.164e4a0a21a39p+1", "-0x1.165f26c39d28fp+1", "-0x1.1667a5b9a0fe1p+1",
        "-0x1.166c24f45461fp+1", "-0x1.166e6a2bc5e99p+1", "-0x1.166fe70ee389dp+1"],
    ("cg", None): [
        "-0x1.0e84d6b471c53p+1", "-0x1.13b33857c3173p+1", "-0x1.15f46811d708dp+1", "-0x1.164c718bae489p+1",
        "-0x1.166671adfc95fp+1", "-0x1.1671b76279b29p+1", "-0x1.16726c9ced8afp+1", "-0x1.1672daf49c6c1p+1",
        "-0x1.1672f627c5405p+1", "-0x1.1673a2f097e71p+1", "-0x1.1673abbefd213p+1", "-0x1.1673aebda6fc1p+1",
        "-0x1.1673b25bcbe7dp+1"],
    ("lbfgs", None): [
        "-0x1.0e84d6b471c53p+1", "-0x1.13b33857c3173p+1", "-0x1.15f46811d085fp+1", "-0x1.164c7441bd2d5p+1",
        "-0x1.16666fb8b9e37p+1", "-0x1.1671b95af715dp+1", "-0x1.16726a0f051cfp+1", "-0x1.1672d302590adp+1",
        "-0x1.1672ff7c71b3bp+1", "-0x1.1673a94b16265p+1", "-0x1.1673b288b27b7p+1", "-0x1.1673b64fee42bp+1",
        "-0x1.1673b7367f9b9p+1"],
    ("cg", "cands"): [
        "-0x1.0e84d6b471c53p+1", "-0x1.13a945c682873p+1", "-0x1.15e305c103643p+1", "-0x1.16490f38c6fefp+1",
        "-0x1.1663dc9839801p+1", "-0x1.166ef075987a9p+1", "-0x1.1670e1ebe9a53p+1", "-0x1.167266b8e1e11p+1",
        "-0x1.1672afaf349b9p+1", "-0x1.1672f6960268dp+1", "-0x1.16731371f448dp+1"],
    ("lbfgs", "cands"): [
        "-0x1.0e84d6b471c53p+1", "-0x1.13a945c682873p+1", "-0x1.15cd2823ff9d1p+1", "-0x1.1643b8f20eb67p+1",
        "-0x1.16602f60bcb85p+1", "-0x1.166e7209d8169p+1", "-0x1.1671ebb150d37p+1", "-0x1.1672a8adf286dp+1",
        "-0x1.1672db5306d5fp+1", "-0x1.1673228a8129dp+1", "-0x1.16737a4f6da7dp+1"],
}


@pytest.mark.parametrize("opt, mode", list(M2_ENERGIES))
def test_default_runs_reproduce_stored_m2_energies(h4, opt, mode):
    """Default-metric runs reproduce the energies recorded with the M2 code, a reference
    that is not computed by the current code (the frozen-code tests above share its driver).

    They were equal bit for bit on the machine that recorded them. Rounding is machine
    dependent (BLAS kernels; on a second machine E_HF differed by 2 ulp), and the exact
    line search (bounded Brent, xatol = 1e-10 eps_max) turns that into energy differences
    of up to 8.6e-10 Ha in these runs, while the candidate runs differed by at most
    5.5e-14 Ha. On that machine the M2 code and the current code again agreed bit for bit.
    The tolerances, 1e-8 Ha (exact search) and 1e-12 Ha (candidates), stay far below the
    smallest changes of the default directions tried on these runs: CG -> L-BFGS moves the
    energies by 1.1e-6 Ha, PR+ -> FR by 2.6e-6 and L-BFGS memory 10 -> 3 by 2.8e-6.
    """
    stored = np.array([float.fromhex(x) for x in M2_ENERGIES[(opt, mode)]])
    res = run_cqe(h4, max_iter=len(stored) - 1, optimizer=opt,
                  eps_candidates=BENCH_CANDS if mode == "cands" else None)
    assert res.energies.shape == stored.shape
    assert np.allclose(res.energies, stored, rtol=0.0, atol=1e-12 if mode == "cands" else 1e-8)
