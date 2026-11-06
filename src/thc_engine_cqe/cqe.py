"""Exact (noise-free, emulated) contracted quantum eigensolver.

Each iteration computes the exact ACSE residual A of the current state, forms a
search direction d and moves along Psi(eps) = exp(eps D) Psi,
D = sum d[p,q,r,s] e_{pq,rs}, with an exact line search. The default direction
is steepest descent, d = g = conj(A), optionally compressed by a user callback;
``optimizer="cg"`` or ``"lbfgs"`` selects the conjugate-gradient or L-BFGS
directions of :mod:`thc_engine_cqe.optimizers`, and ``metric="spin_orbital"``
preconditions any of them with the metric of the standard spin-orbital ACSE
generator (for singlets, steepest descent then is the standard spin-orbital
CQE). This is the reference trajectory against which compressed and noisy
variants are compared. ``run_cqe(..., eps_candidates=...)`` replaces the exact
line search by a choice among a few fixed step sizes, as on hardware.

Estimator hooks. ``residual_estimator`` and ``energy_estimator`` substitute
estimates (for example from a finite-shot noise model) for the exact residual
and energies in every decision of the driver. The exact quantities are still
computed and recorded next to the estimates, so one run gives both the
trajectory the estimates produce and its exact energies.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import scipy.optimize

from .fci_ops import FCISpace, Hamiltonian, KrylovPropagator, inner, normalize, two_body
from .integrals import ActiveSpace
from .optimizers import direction_slope, make_optimizer, structure_error
from .residual import ResidualData, compute_residual, descent_generator, spin_free_norm

__all__ = ["CQEStep", "CQEResult", "run_cqe", "line_search", "generator_norm_estimate"]

# compress(g, residual_data, iteration) -> compressed pair-order coefficients
Compressor = Callable[[np.ndarray, ResidualData, int], np.ndarray]
# residual_estimator(residual_data, ci, iteration) -> estimated pair-order residual A_est
ResidualEstimator = Callable[[ResidualData, np.ndarray, int], np.ndarray]
# energy_estimator(exact_energy, ci, context) -> estimated energy; context "current" or "candidate"
EnergyEstimator = Callable[[float, np.ndarray, str], float]

# Relative tolerance on the pair symmetry and anti-Hermiticity of A_est (max-norm).
_STRUCTURE_RTOL = 1e-10


@dataclass
class CQEStep:
    """One CQE iteration. Without estimators every ``*_est`` field is None.

    Exact fields (always): energy, variance, residual_norm = ||A||, slope =
    -Re sum d A (exact dE/deps along the direction used), full_slope = -||A||^2,
    eps, energy_after, and with keep_* the direction d (``generator``) and the
    exact A (``residual``). With a residual_estimator: residual_norm_est =
    ||A_est||, slope_est = -Re sum d A_est (the slope the driver acted on) and,
    with keep_residuals, residual_est = A_est. With an energy_estimator:
    energy_est (state before the step, context "current") and energy_after_est
    (the chosen candidate; equal to energy_est when no step is taken).
    n_energy_evals counts the energies E(eps) the line search evaluated in this
    step (exact search: grid and refinement points of every bracket; candidates:
    one per candidate); the energy of the current state is not counted.
    optimizer_info is the optimizer's diagnostics dict (None for "sd" with the
    default metric; it carries "metric" when the metric is not the default).
    """

    iteration: int
    energy: float            # energy of the state *before* this step's update
    variance: float
    residual_norm: float     # ||A|| of the state before the update
    slope: float             # dE/deps at eps = 0 along the generator used
    full_slope: float        # -||A||^2, the slope of the uncompressed generator
    eps: float               # step taken
    energy_after: float
    generator: Optional[np.ndarray] = None
    residual: Optional[np.ndarray] = None
    residual_est: Optional[np.ndarray] = None
    residual_norm_est: Optional[float] = None
    slope_est: Optional[float] = None
    energy_est: Optional[float] = None
    energy_after_est: Optional[float] = None
    n_energy_evals: int = 0
    optimizer_info: Optional[dict] = None

    @property
    def captured_descent(self) -> float:
        """slope / full_slope: fraction of the steepest-descent rate retained.

        full_slope is always that of the unpreconditioned d = g. For "cg" and
        "lbfgs" this ratio of exact slopes can exceed 1 (observed up to 2.6 in
        benchmarks/run_optimizers.py); for "sd" with metric "spin_orbital" and
        the exact residual it is <g, M g>/<g, g>, between 1/3 and 1. With a
        residual_estimator it is <g, d>/<g, g> for the direction d built from
        A_est (for "sd" with metric "spin_orbital", <g, M g_est>/<g, g> with
        g_est = conj(A_est)), which has neither bound and can be negative.
        """
        return self.slope / self.full_slope if self.full_slope != 0 else 0.0


@dataclass
class CQEResult:
    active: ActiveSpace
    steps: list[CQEStep] = field(default_factory=list)
    ci: Optional[np.ndarray] = None
    e_exact: Optional[float] = None
    converged: bool = False
    optimizer: str = "sd"
    metric: str = "spin_free"

    @property
    def energies(self) -> np.ndarray:
        """Energy before each step, followed by the final energy."""
        if not self.steps:
            return np.array([])
        return np.array([s.energy for s in self.steps] + [self.steps[-1].energy_after])

    @property
    def errors(self) -> np.ndarray:
        if self.e_exact is None:
            raise ValueError("no exact reference energy stored")
        return self.energies - self.e_exact

    @property
    def energies_est(self) -> np.ndarray:
        """Estimated energies aligned with :attr:`energies` (runs with an energy_estimator)."""
        if not self.steps:
            return np.array([])
        if any(s.energy_est is None for s in self.steps):
            raise ValueError("no energy estimates recorded (run without energy_estimator)")
        return np.array([s.energy_est for s in self.steps] + [self.steps[-1].energy_after_est])

    @property
    def cumulative_energy_evals(self) -> np.ndarray:
        """Line-search energy evaluations used to reach each entry of :attr:`energies` (0 first)."""
        if not self.steps:
            return np.array([], dtype=int)
        return np.concatenate([[0], np.cumsum([s.n_energy_evals for s in self.steps])]).astype(int)


def generator_norm_estimate(space: FCISpace, g: np.ndarray, psi: np.ndarray, iters: int = 6) -> float:
    """Power-iteration estimate of ||G|| on the CI space (starting near psi)."""
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


def _candidate_steps(candidates: Sequence[float]) -> np.ndarray:
    """Fixed step sizes: a non-empty sequence of finite reals.

    Zero (stay at the current state) and negative values (a step against the
    direction) are allowed. A negative step pays off when the direction is
    uphill for the true energy, which happens with noisy residual estimates;
    with the exact residual every direction the driver uses is a descent
    direction, so the exact line search stays on eps >= 0.
    """
    cand = np.asarray(candidates, dtype=float).ravel()
    if cand.size == 0 or not np.all(np.isfinite(cand)):
        raise ValueError("candidates must be a non-empty sequence of finite step sizes")
    return cand


def line_search(energy_of_eps: Callable[[float], float], eps_max: float, n_grid: int = 64,
                candidates: Sequence[float] | None = None) -> tuple[float, float]:
    """Minimize E(eps) on [0, eps_max]: coarse grid, then bounded Brent refinement.

    The search is one-sided: the grid starts at 0 and the Brent refinement is
    bounded by the neighbouring grid points, so the result is never negative.
    The driver only searches along descent directions (dE/deps < 0 at 0), whose
    minimum lies at eps > 0.

    With ``candidates``, E is evaluated only at those fixed step sizes and the
    lowest is returned (the first on ties); ``eps_max`` and ``n_grid`` are then
    ignored. This mimics a resource-friendly hardware CQE, where each energy
    costs a set of measurements: one energy per candidate and no refinement.
    The lowest candidate is taken even if it raises the energy, as such a run
    would. Candidates may be zero or negative (:func:`_candidate_steps`).
    """
    if candidates is not None:
        cand = _candidate_steps(candidates)
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


def _estimate_energy(energy_estimator: EnergyEstimator, e_exact: float, ci: np.ndarray,
                     context: str) -> float:
    val = float(energy_estimator(e_exact, ci.copy(), context))
    if not np.isfinite(val):
        raise ValueError(f"energy_estimator returned a non-finite energy ({val}) for context {context!r}")
    return val


def _estimate_residual(residual_estimator: ResidualEstimator, rd: ResidualData, ci: np.ndarray,
                       it: int) -> np.ndarray:
    a = np.asarray(residual_estimator(copy.deepcopy(rd), ci.copy(), it))
    if a.shape != rd.A.shape:
        raise ValueError(f"residual_estimator must return a pair-order tensor of shape {rd.A.shape}, "
                         f"got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("residual_estimator returned non-finite values")
    pair, herm = structure_error(a)
    tol = _STRUCTURE_RTOL * float(np.abs(a).max()) if a.size else 0.0
    if pair > tol or herm > tol:
        raise ValueError("residual_estimator must return a tensor with the structure of A: "
                         "A[p,q,r,s] = A[r,s,p,q] and A[q,p,s,r] = -conj(A[p,q,r,s]) "
                         f"(defects {pair:.2e}, {herm:.2e}); see optimizers.symmetrize_residual")
    return a


def _choose_candidate(energy: Callable[[float], float], prop: KrylovPropagator,
                      candidates: Sequence[float], energy_estimator: EnergyEstimator,
                      psi: np.ndarray, e_current: float, e_current_est: float):
    """Fixed-candidate step chosen by estimated energies: (eps, E, psi, E_est), E exact.

    A zero candidate is the current state: it reuses the current energy and its
    estimate, without a new estimator call.
    """
    cand = _candidate_steps(candidates)
    exact, states, est = [], [], []
    for e in cand:
        if e == 0.0:
            exact.append(e_current)
            states.append(psi)
            est.append(e_current_est)
            continue
        x = energy(float(e))
        st = normalize(prop.state(float(e)))
        exact.append(x)
        states.append(st)
        est.append(_estimate_energy(energy_estimator, float(x), st, "candidate"))
    i = int(np.argmin(est))
    return float(cand[i]), float(exact[i]), states[i], float(est[i])


def _krylov_step(ham: Hamiltonian, g: np.ndarray, psi: np.ndarray, eps_max: float,
                 tol: float, candidates: Sequence[float] | None = None,
                 energy_estimator: EnergyEstimator | None = None,
                 e_current: float | None = None, e_current_est: float | None = None):
    """Line search along exp(eps G) psi: (eps, E, psi_new, n_evals, E_est or None).

    With ``energy_estimator`` (which requires ``candidates``) the candidate
    with the lowest estimated energy is taken; E is its exact energy. A zero
    candidate is the current state ``psi`` with energy ``e_current`` (and
    estimate ``e_current_est``); it is not counted as an evaluation.
    """
    space = ham.space

    def matvec(v):
        return two_body(space, g, v)

    check_eps = None
    if candidates is not None:
        # One Krylov space, accurate up to the largest |candidate|, serves all of them.
        cand = _candidate_steps(candidates)
        eps_max = float(np.max(np.abs(cand)))
        if cand.min() < 0:
            check_eps = (eps_max, -eps_max)
    n_evals = 0
    for _ in range(6):
        prop = KrylovPropagator(matvec, psi, eps_max=eps_max, tol=tol, check_eps=check_eps)
        V = prop.basis()
        HV = np.column_stack([ham.apply(V[:, j].reshape(space.shape)).ravel() for j in range(V.shape[1])])
        hproj = V.conj().T @ HV

        def energy(eps):
            nonlocal n_evals
            n_evals += 1
            y = prop.coefficients(eps)
            return float((np.vdot(y, hproj @ y) / np.vdot(y, y)).real) + ham.ecore

        if energy_estimator is not None:
            eps, e, psi_new, e_est = _choose_candidate(energy, prop, candidates, energy_estimator,
                                                       psi, e_current, e_current_est)
            return eps, e, psi_new, n_evals, e_est
        if candidates is not None:
            def cand_energy(eps):
                return e_current if (eps == 0.0 and e_current is not None) else energy(eps)

            eps, e = line_search(cand_energy, eps_max, candidates=candidates)
            psi_new = psi if (eps == 0.0 and e_current is not None) else normalize(prop.state(eps))
            return eps, e, psi_new, n_evals, None
        eps, e = line_search(energy, eps_max)
        if eps < 0.98 * eps_max:
            return eps, e, normalize(prop.state(eps)), n_evals, None
        eps_max *= 2.0  # the minimum is on the edge of the bracket: enlarge it
    return eps, e, normalize(prop.state(eps)), n_evals, None


def run_cqe(active: ActiveSpace, ci0: np.ndarray | None = None, max_iter: int = 30,
            r_tol: float = 1e-7, e_tol: float = 1e-11, compress: Compressor | None = None,
            keep_generators: bool = False, keep_residuals: bool = False,
            krylov_tol: float = 1e-10, e_exact: float | None = None,
            callback: Callable[[CQEStep, np.ndarray], None] | None = None,
            eps_candidates: Sequence[float] | None = None, optimizer: str = "sd",
            optimizer_options: Mapping | None = None,
            residual_estimator: ResidualEstimator | None = None,
            energy_estimator: EnergyEstimator | None = None,
            metric: str = "spin_free", first_eps: float | None = None) -> CQEResult:
    """Run the exact CQE from ``ci0`` (default: the Hartree-Fock determinant).

    With the default arguments this is the M1 steepest-descent CQE, bit for bit.

    Args:
        compress: optional callback returning compressed generator coefficients
            (steepest descent with the default metric only; NotImplementedError
            otherwise). If the compressed generator is not a descent direction
            its sign is flipped; if its slope vanishes the run stops. With a
            residual_estimator it receives g = conj(A_est) and the exact
            ResidualData. To compress the spin-orbital direction, compose:
            compress=lambda g, rd, it: f(optimizers.apply_metric(g, "spin_orbital"), rd, it).
        e_exact: reference energy stored on the result (e.g. FCI) for errors.
        callback: called as callback(step, ci_after) after every iteration.
        eps_candidates: fixed step sizes. If given, each step evaluates the energy
            only at these and takes the lowest (:func:`line_search`), instead of
            the exact line search. Steps can then raise the energy. eps multiplies
            the unnormalized direction d (g = conj(A) for steepest descent).
            Observed steps of the exact line search on the systems of
            benchmarks/run_optimizers.py (H4, H6, H8, H6 at 3 bohr, LiH, N2 at
            1.098 and 2.0 A): first step 0.05-0.29 (d = g for every optimizer);
            later steps, 10-90% ranges per system, 0.017-0.12 for steepest
            descent (roughly independent of ||A||), 0.026-0.32 for "cg" and
            0.27-6.9 for "lbfgs", with isolated L-BFGS steps above 100. The set
            must reach below the smallest steps a run needs. Observed on H8 with
            steepest descent: with (0.025, 0.05, 0.1, 0.2) the energy rises from
            step 6 on and ends 0.74 Ha above FCI after 60 steps; adding 0.0125
            keeps it monotone. The benchmark's default 0.00625 * 2^k,
            k = 0..10, served all three optimizers on those systems.
            With metric "spin_orbital" the directions are shorter (M has the
            eigenvalues 1 and 1/3) and the steps longer. Observed on the same
            systems plus H6 at 1.0 A (two benchmark runs, on different
            machines): first step 0.12-0.38 (d = M g for every optimizer);
            later steps, 10-90% ranges per system, 0.035-0.35 for "sd",
            0.06-0.8 for "cg" and 0.5-8.5 for "lbfgs" (15-20 of 427 L-BFGS steps
            above 6.4, the largest 1.9e3 on N2; spin-free L-BFGS has 15 of 431).
            The default set still contains every exact "sd" and "cg" step of
            both metrics.
            Candidates may be zero or negative. Zero keeps the current state; its
            energy (and estimate) is the current one, not evaluated again and not
            counted in n_energy_evals. A negative eps steps against d, which can
            lower the energy when a noisy estimated direction is uphill for the
            true energy. A zero step never counts as e_tol convergence; without
            estimators, a second consecutive zero step along the same direction
            stops the run unconverged, since every later step would repeat it.
        optimizer: the search direction (:mod:`thc_engine_cqe.optimizers`):
            "sd" (steepest descent d = g = conj(A), the default; d = M g with a
            metric), "cg" (nonlinear conjugate gradient, Polak-Ribiere+ by
            default) or "lbfgs" (L-BFGS in the local coordinates). Each call
            starts a fresh optimizer; the first "cg" and "lbfgs" direction is
            the "sd" direction of the same metric. The line search is the same
            for all.
        optimizer_options: keyword arguments of the optimizer: "cg" takes
            beta ("pr+" | "fr") and restart; "lbfgs" takes memory (default 10)
            and curvature_tol (default 1e-2); "sd" takes none.
        metric: the metric (preconditioner) M of every optimizer
            (optimizers.METRICS, optimizers.apply_metric). "spin_free"
            (default): M = I, the M1/M2 directions bit for bit.
            "spin_orbital": M = (2 I + T)/3 with (T X)[p,q,r,s] = X[p,s,r,q],
            symmetric positive definite on the residual's structured space
            (eigenvalues 1 and 1/3). For singlet states, sum (M g) e is exactly
            the standard spin-orbital ACSE generator (Smart & Mazziotti,
            PRL 126, 070504 (2021))
            sum_{ijkl} conj(2A^{ij}_{kl}) a+_i a+_j a_l a_k,
            2A^{ij}_{kl} = <[a+_i a+_j a_l a_k, H]>, in the full-sum
            convention: i, j, k, l run independently over all 2N spin orbitals,
            with no prefactor (the convention of the Mazziotti group's HQCA
            code; optimizers module docstring). "sd" is then the standard
            spin-orbital CQE, and eps_candidates multiply that full-sum
            generator; a generator summed over i < j, k < l only is a quarter
            of it and needs 4 eps for the same step. "cg" uses z = M g
            (preconditioned Polak-Ribiere+ or Fletcher-Reeves, restarts d = z);
            "lbfgs" uses H_0 = gamma M with gamma = <s, y>/<y, M y> and d = M g
            with an empty memory. For non-singlet states sum (M g) e is the spin
            average of the spin-orbital generator, not the generator itself.
            The same holds with a residual_estimator: if A_est is the spin
            contraction (measure.targets.contract_spin) of a spin-orbital
            estimate 2A_est, sum (M conj(A_est)) e is the spin-averaged
            (S^2-conserving) projection of the spin-orbital generator built
            from 2A_est, for any state and estimation error, not that raw
            estimated generator, whose non-singlet error components would
            change <S^2>. The metric is recorded in CQEResult.metric and, when
            not the default, in optimizer_info.
        residual_estimator: optional callable (rd, ci, iteration) -> A_est, the
            estimated pair-order residual of the current state. rd is a copy of
            the exact ResidualData and ci a copy of the normalized current CI
            vector. A_est must have A's shape and structure, pair-symmetric with
            A_est[q,p,s,r] = -conj(A_est[p,q,r,s]) to 1e-10 relative to
            max |A_est| (see optimizers.symmetrize_residual); otherwise
            ValueError. Every decision then uses A_est: the direction (g =
            conj(A_est) and the optimizer history), the slope -Re sum d A_est
            and its sign check, the generator norm that sets the line-search
            bracket, and the convergence test ||A_est|| < r_tol. The exact
            residual is still computed and recorded (residual_norm, slope,
            full_slope, residual); the estimates are recorded in
            residual_norm_est, slope_est and residual_est. Without an
            energy_estimator the line search (exact or over eps_candidates) and
            the e_tol test use exact energies, so the exact line search is an
            oracle along the estimated direction. A step along a direction that
            is not a descent direction of the exact energy (exact slope >= 0;
            the exact search then returns eps = 0 or a negligible step, observed
            down to a few 1e-9) never counts as e_tol convergence: the run goes
            on with the next estimate. A noisy direction nearly orthogonal to
            the exact gradient also gives a tiny energy change and can still
            meet e_tol, so for noisy estimates use e_tol = 0 and an r_tol below
            the noise floor of ||A_est||.
        energy_estimator: optional callable (e_exact, ci, context) -> E_est.
            Requires eps_candidates (ValueError otherwise). It is called with
            context "current" for the state before each step (once per
            iteration, also when the run then stops) and with "candidate" for
            each normalized candidate state exp(eps D) Psi, in the order of
            eps_candidates. The candidate with the lowest estimate is taken
            (the first on ties), and the e_tol test compares estimates,
            |E_est(current) - E_est(chosen)| < e_tol. Exact energies stay in
            energy / energy_after; the estimates are recorded in energy_est /
            energy_after_est.
        first_eps: optional step size of iteration 0, taken instead of its line
            search (exact or over eps_candidates): Psi_1 = exp(first_eps D_0)
            Psi_0 whatever its energy, with one energy evaluation (also through
            energy_estimator, as the only candidate, if one is given). Later
            iterations search as usual. The optimizer records the step like any
            other, so "cg" and "lbfgs" keep the step-0 pair (s_0 = first_eps d_0,
            y_0 = g_1 - g_0): this is the run that starts the measured loop after
            a classically computed first step, e.g. the second-order step of
            ft.trajectory.second_order_step. A restart from Psi_1 would instead
            lose the step-0 pair. With first_eps equal to the exact first step
            the run equals the plain run up to the Krylov tolerance (the two
            build different Krylov spaces). Must be finite.

    Per iteration the hooks are called in this order: residual_estimator,
    energy_estimator("current"), then (unless the run stops) one
    energy_estimator("candidate") call per candidate.
    """
    opt = make_optimizer(optimizer, optimizer_options, metric=metric)
    if compress is not None and opt.name != "sd":
        raise NotImplementedError("compress is implemented only for optimizer='sd' (out of scope for M2)")
    if compress is not None and opt.metric != "spin_free":
        raise NotImplementedError("compress is implemented only for metric='spin_free'; to compress the "
                                  "spin-orbital direction, apply optimizers.apply_metric(g, 'spin_orbital') "
                                  "inside the compressor")
    if energy_estimator is not None:
        if eps_candidates is None:
            raise ValueError("energy_estimator requires eps_candidates: the exact line search "
                             "has no estimated-energy version")
        _candidate_steps(eps_candidates)
    if first_eps is not None:
        first_eps = float(first_eps)
        if not np.isfinite(first_eps):
            raise ValueError(f"first_eps must be finite, got {first_eps}")
    ham = Hamiltonian(active)
    space = ham.space
    psi = space.hf() if ci0 is None else normalize(np.asarray(ci0).reshape(space.shape))
    result = CQEResult(active=active, e_exact=e_exact, optimizer=opt.name, metric=opt.metric)
    with_res_est = residual_estimator is not None
    zero_step_dir = None
    for it in range(max_iter):
        rd = compute_residual(active, psi, ham)
        A = rd.A
        A_est = A if not with_res_est else _estimate_residual(residual_estimator, rd, psi, it)
        e_cur_est = None
        if energy_estimator is not None:
            e_cur_est = _estimate_energy(energy_estimator, rd.energy, psi, "current")
        g_full = descent_generator(A_est)
        full_slope = -float(np.sum(np.abs(A) ** 2))
        if compress is not None:
            g = np.asarray(compress(g_full, rd, it))
        else:
            g = opt.direction(g_full, A_est)
        slope_est = direction_slope(g, A_est)
        if compress is not None and slope_est > 0:
            g, slope_est = -g, -slope_est
        if with_res_est:
            slope, r_est = direction_slope(g, A), spin_free_norm(A_est)
        else:
            slope, r_est = slope_est, rd.residual_norm
        est_fields = {"optimizer_info": opt.info, "energy_est": e_cur_est}
        if with_res_est:
            est_fields.update(residual_norm_est=r_est, slope_est=slope_est,
                              residual_est=A_est if keep_residuals else None)
        if r_est < r_tol or abs(slope_est) < 1e-300:
            result.converged = r_est < r_tol
            result.steps.append(CQEStep(it, rd.energy, rd.variance, rd.residual_norm, slope,
                                        full_slope, 0.0, rd.energy,
                                        g if keep_generators else None,
                                        A if keep_residuals else None,
                                        energy_after_est=e_cur_est, **est_fields))
            break
        gnorm = generator_norm_estimate(space, g, psi)
        # Initial line-search bracket [0, 1.5/||G||]. E(eps) = <psi|exp(eps G)^dag H
        # exp(eps G)|psi> is a sum of sinusoids in eps whose angular frequencies are
        # differences of eigenvalues of -iG, so at most 2||G||. With 1.5 just under
        # pi/2, even the fastest one completes less than half a period inside the
        # bracket, so it has at most one minimum there; the bracket also bounds the
        # Krylov space. It is conservative: the optimum is typically at 0.5-30% of it,
        # and _krylov_step doubles it if the minimum lands on the edge.
        eps_max = 1.5 / max(gnorm, 1e-12)
        cands = eps_candidates if (first_eps is None or it > 0) else (first_eps,)
        eps, e_new, psi_new, n_evals, e_new_est = _krylov_step(ham, g, psi, eps_max, krylov_tol,
                                                               cands, energy_estimator,
                                                               rd.energy, e_cur_est)
        opt.step_taken(eps)
        step = CQEStep(it, rd.energy, rd.variance, rd.residual_norm, slope, full_slope, eps,
                       e_new, g if keep_generators else None, A if keep_residuals else None,
                       energy_after_est=e_new_est, n_energy_evals=n_evals, **est_fields)
        result.steps.append(step)
        if callback is not None:
            callback(step, psi_new)
        if energy_estimator is None:
            converged_e = abs(rd.energy - e_new) < e_tol
            if with_res_est and not slope < 0.0:
                # The estimated direction is uphill (or flat) for the exact energy; the exact
                # search then returns eps = 0 or a negligible step (Brent can settle at a tiny
                # eps whose energy rounds to E). A small energy change is no evidence of
                # convergence here. Without estimates, and with a residual_estimator that
                # returns rd.A, the exact slope is negative at every line search, so those
                # runs are unaffected.
                converged_e = False
        else:
            converged_e = abs(e_cur_est - e_new_est) < e_tol
        stalled = False
        if eps_candidates is not None and eps == 0.0:
            # A zero candidate was chosen: the state is unchanged, which says nothing
            # about convergence. Without estimators the run is deterministic, so a second
            # zero step along the same direction would repeat forever: stop (unconverged).
            converged_e = False
            if not (with_res_est or energy_estimator is not None):
                stalled = zero_step_dir is not None and np.array_equal(zero_step_dir, g)
                zero_step_dir = g.copy()
        else:
            zero_step_dir = None
        psi = psi_new
        if converged_e:
            result.converged = True
            break
        if stalled:
            break
    result.ci = psi
    return result
