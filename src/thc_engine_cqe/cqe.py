"""Exact (noise-free, emulated) contracted quantum eigensolver.

Each iteration computes the exact ACSE residual A of the current state, forms
the generator G = sum g[p,q,r,s] e_{pq,rs} (g = conj(A), optionally compressed
by a user callback), and moves along Psi(eps) = exp(eps G) Psi with an exact
line search. This is the reference trajectory against which compressed and
noisy variants are compared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import scipy.optimize

from .fci_ops import FCISpace, Hamiltonian, KrylovPropagator, inner, normalize, two_body
from .integrals import ActiveSpace
from .residual import ResidualData, compute_residual, descent_generator

__all__ = ["CQEStep", "CQEResult", "run_cqe", "line_search", "generator_norm_estimate"]

# compress(g, residual_data, iteration) -> compressed pair-order coefficients
Compressor = Callable[[np.ndarray, ResidualData, int], np.ndarray]


@dataclass
class CQEStep:
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

    @property
    def captured_descent(self) -> float:
        """slope / full_slope: fraction of the steepest-descent rate retained."""
        return self.slope / self.full_slope if self.full_slope != 0 else 0.0


@dataclass
class CQEResult:
    active: ActiveSpace
    steps: list[CQEStep] = field(default_factory=list)
    ci: Optional[np.ndarray] = None
    e_exact: Optional[float] = None
    converged: bool = False

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

# to do: this line_search should also provide a numerical option to select the lowest energy from a few fixed eps to mimic resource-friendly hardware execution
def line_search(energy_of_eps: Callable[[float], float], eps_max: float, n_grid: int = 64) -> tuple[float, float]:
    """Minimize E(eps) on [0, eps_max]: coarse grid, then bounded Brent refinement."""
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


def _krylov_step(ham: Hamiltonian, g: np.ndarray, psi: np.ndarray, eps_max: float,
                 tol: float) -> tuple[float, float, np.ndarray]:
    space = ham.space

    def matvec(v):
        return two_body(space, g, v)

    for _ in range(6):
        prop = KrylovPropagator(matvec, psi, eps_max=eps_max, tol=tol)
        V = prop.basis()
        HV = np.column_stack([ham.apply(V[:, j].reshape(space.shape)).ravel() for j in range(V.shape[1])])
        hproj = V.conj().T @ HV

        def energy(eps):
            y = prop.coefficients(eps)
            return float((np.vdot(y, hproj @ y) / np.vdot(y, y)).real) + ham.ecore

        eps, e = line_search(energy, eps_max)
        if eps < 0.98 * eps_max:
            return eps, e, normalize(prop.state(eps))
        eps_max *= 2.0
    return eps, e, normalize(prop.state(eps))


def run_cqe(active: ActiveSpace, ci0: np.ndarray | None = None, max_iter: int = 30,
            r_tol: float = 1e-7, e_tol: float = 1e-11, compress: Compressor | None = None,
            keep_generators: bool = False, keep_residuals: bool = False,
            krylov_tol: float = 1e-10, e_exact: float | None = None,
            callback: Callable[[CQEStep, np.ndarray], None] | None = None) -> CQEResult:
    """Run the exact CQE from ``ci0`` (default: the Hartree-Fock determinant).

    Args:
        compress: optional callback returning compressed generator coefficients.
            If the compressed generator is not a descent direction its sign is
            flipped; if its slope vanishes the run stops.
        e_exact: reference energy stored on the result (e.g. FCI) for errors.
        callback: called as callback(step, ci_after) after every iteration.
    """
    ham = Hamiltonian(active)
    space = ham.space
    psi = space.hf() if ci0 is None else normalize(np.asarray(ci0).reshape(space.shape))
    result = CQEResult(active=active, e_exact=e_exact)
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
            result.converged = rd.residual_norm < r_tol
            result.steps.append(CQEStep(it, rd.energy, rd.variance, rd.residual_norm, slope,
                                        full_slope, 0.0, rd.energy,
                                        g if keep_generators else None,
                                        A if keep_residuals else None))
            break
        gnorm = generator_norm_estimate(space, g, psi)
        # what is this 1.5 below for??
        eps_max = 1.5 / max(gnorm, 1e-12)
        eps, e_new, psi_new = _krylov_step(ham, g, psi, eps_max, krylov_tol)
        step = CQEStep(it, rd.energy, rd.variance, rd.residual_norm, slope, full_slope, eps,
                       e_new, g if keep_generators else None, A if keep_residuals else None)
        result.steps.append(step)
        if callback is not None:
            callback(step, psi_new)
        converged_e = abs(rd.energy - e_new) < e_tol
        psi = psi_new
        if converged_e:
            result.converged = True
            break
    result.ci = psi
    return result
