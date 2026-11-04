"""Search directions (SD, CG, L-BFGS) and the estimator hooks of cqe.run_cqe (M2 workflow A2).

Iteration counts quoted in docstrings were measured with this code (exact line
search, OMP_NUM_THREADS=1); the assertions keep a margin around them.

The bit-for-bit comparisons need a deterministic computation. PySCF's
multi-threaded OpenMP reductions are not (observed on H4: with three threads,
repeated identical runs in one process differ within the first three steps), so
this module pins PySCF to one thread whatever OMP_NUM_THREADS says.
"""

import math

import numpy as np
import pytest
import scipy.linalg
import scipy.optimize

import thc_engine_cqe.cqe as cqe_module
from thc_engine_cqe import systems
from thc_engine_cqe.cqe import run_cqe
from thc_engine_cqe.factor import projection_compressor
from thc_engine_cqe.fci_ops import FCISpace, Hamiltonian, KrylovPropagator, normalize, two_body
from thc_engine_cqe.optimizers import (LBFGS, ConjugateGradient, SteepestDescent, direction_slope,
                                       make_optimizer, real_inner, structure_error,
                                       symmetrize_residual)
from thc_engine_cqe.residual import compute_residual, descent_generator

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

    def spy_make(name, options):
        opt_obj = make(name, options)
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


def _audit_cg(steps, beta_rule):
    """CG directions rebuilt from the records: d_k = g_k + beta d_{k-1}, g_k = conj(A_k),
    beta = 0 when the slope -<g_k, d> is not negative."""
    dirs, g_prev, d_prev = [], None, None
    for st in steps:
        g = _real_coords(np.conj(st.residual))
        d = g
        if d_prev is not None:
            if beta_rule == "fr":
                beta = (g @ g) / (g_prev @ g_prev)
            else:
                beta = max(0.0, g @ (g - g_prev) / (g_prev @ g_prev))
            if -(g @ (g + beta * d_prev)) < 0:
                d = g + beta * d_prev
        dirs.append(d)
        g_prev, d_prev = g, d
    return dirs


def _audit_lbfgs(steps, memory, curvature_tol=1e-12):
    """L-BFGS directions rebuilt from the records with an explicit inverse-Hessian matrix:
    pairs s = eps_{k-1} d_{k-1} and y = -(g_k - g_{k-1}) (skipped when <s, y> <=
    curvature_tol ||s|| ||y||, oldest dropped beyond ``memory``), H_0 = gamma I with gamma
    = <s, y>/<y, y> of the newest pair, the BFGS update H <- (1 - rho s y^T) H (1 - rho y s^T)
    + rho s s^T oldest first, d = H g, and d = g with an emptied memory if d is not descent.
    Returns (directions, number of pairs used)."""
    pairs, dirs, n_pairs, g_prev = [], [], [], None
    for k, st in enumerate(steps):
        g = _real_coords(np.conj(st.residual))
        if k > 0:
            s = steps[k - 1].eps * _real_coords(steps[k - 1].generator)
            y = -(g - g_prev)
            if s @ y > curvature_tol * np.linalg.norm(s) * np.linalg.norm(y):
                pairs = (pairs + [(s, y)])[-memory:]
        d = g
        if pairs:
            s_new, y_new = pairs[-1]
            h = (s_new @ y_new) / (y_new @ y_new) * np.eye(g.size)
            for s, y in pairs:
                rho = 1.0 / (s @ y)
                hy = h @ y
                h = (h - rho * (np.outer(s, hy) + np.outer(hy, s))
                     + (rho + rho ** 2 * (y @ hy)) * np.outer(s, s))
            d = h @ g
            if not -(g @ d) < 0:
                pairs, d = [], g
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
