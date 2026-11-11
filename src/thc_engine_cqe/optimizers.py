"""Search directions for the CQE: steepest descent, conjugate gradient and L-BFGS, with a metric.

Local coordinates. At the iterate Psi_k the CQE parameterizes nearby states as

    Psi(theta) = exp(Theta) Psi_k,   Theta = sum_pqrs theta[p,q,r,s] e_{pq,rs},

with theta in the real vector space V of pair-order tensors that have the
structure of conj(A):

    pair symmetry       theta[p,q,r,s] = theta[r,s,p,q]
    anti-Hermiticity    theta[q,p,s,r] = -conj(theta[p,q,r,s])   (Theta^dag = -Theta)

V carries the real inner product <u, v> = Re sum conj(u) v (:func:`real_inner`).
Along a direction d the slope of the energy at theta = 0 is

    dE/deps = <[H, D]> = -Re sum d A = -<g, d>,   g = conj(A),

so the energy gradient is grad = -conj(A) = -g and steepest descent is d = g
(docs/conventions.md). Every direction below is a combination of g's, M g's
(next paragraph) and earlier directions with real coefficients, so it stays in V
(tests/test_optimizers.py checks pair symmetry and anti-Hermiticity of every
direction to 1e-12 relative).

Metric. Every optimizer takes ``metric`` (one of METRICS), a symmetric
positive-definite operator M on V that turns gradients into directions
(:func:`apply_metric`):

    "spin_free"     M = I, the default: the M1/M2 directions, bit for bit;
    "spin_orbital"  M = (2 I + T)/3,  (T x)[p,q,r,s] = x[p,s,r,q].

T exchanges the two annihilated orbitals (e_{pq,rs} -> e_{ps,rq}). It is an
involutive permutation of the entries, so M is symmetric in <u, v>, with the
eigenvalues 1 on T = +1 (the singlet-pair channel) and 1/3 on T = -1 (the
triplet-pair channel). T commutes with the pair exchange and, on pair-symmetric
tensors only, with x -> conj(x[q,p,s,r]); hence it commutes with the projector
onto V (:func:`symmetrize_residual`), and M maps V to itself (tested). For a
singlet state, D = sum (M g) e is exactly the standard spin-orbital ACSE
generator of Mazziotti and co-workers (e.g. Smart & Mazziotti, PRL 126, 070504
(2021)), in the full-sum convention (scale below):

    sum_{ijkl} conj(2A^{ij}_{kl}) a+_i a+_j a_l a_k = sum_pqrs [M g]_pqrs e_{pq,rs},
    2A^{ij}_{kl} = <[a+_i a+_j a_l a_k, H]>,

where i, j, k, l run independently over all 2N spin orbitals (real states:
conj(2A) = 2A). Derivation, for spin orbitals (p, sigma) and the blocks
B^{sigma tau}[p,q,r,s] = 2A^{(p sigma)(r tau)}_{(q sigma)(s tau)}, whose sum over
sigma, tau is A. H is spin-free, so [Gamma, H] transforms like Gamma under spin
rotations, and in a singlet only the spin-scalar parts survive:
B^{alpha alpha} = B^{beta beta} = (A - T A)/6 and
B^{alpha beta} = B^{beta alpha} = (2A + T A)/6. The S_z-conserving terms of the
full sum are these four blocks plus the exchange-type terms
a+_{p alpha} a+_{r beta} a_{s alpha} a_{q beta} (and alpha <-> beta), which equal
the opposite-spin blocks after a_l a_k = -a_k a_l and so double them. Comparing the
coefficients of the independent products a+_{p alpha} a+_{r beta} a_{s beta} a_{q alpha}
and of the antisymmetrized same-spin products with those of sum c e gives
c = (2A + T A)/3 = M A, the unique pair-symmetric solution; T commutes with
complex conjugation, so the coefficients conj(2A) give M g. tests/test_optimizers.py
checks the operator identity, including the factor 1/3, on the Fock space of H4,
with 2A built from its definition, and fixed-step runs against
psi <- exp(eps G_so) psi there. The slopes agree: -<g, M g> = -sum_{ijkl} |2A^{ij}_{kl}|^2.
Equivalently, M^-1 is the Gram matrix of the e_{pq,rs} in the Euclidean inner
product of spin-orbital amplitudes: the antisymmetrized spin-orbital coefficients
C(c) of sum c e have sum |C(c)|^2 = <c, M^-1 c> for every c in V (a triplet-pair
amplitude multiplies three M_S components, a singlet-pair amplitude one; tested).
So "spin_orbital" is steepest descent in the metric of the spin-orbital
amplitudes, restricted to spin-free directions, which for a singlet restricts
nothing. With metric "spin_orbital", steepest descent is therefore the standard
spin-orbital CQE.

Scale. The sum above runs over all (2N)^4 tuples with no prefactor (the
full-sum convention); fixed candidate step sizes eps multiply that generator. A
generator summed over i < j, k < l only is a quarter of it (c = 12 in
M = (2 I + T)/c), so its step sizes are 4 eps. The full-sum convention is the one
of the Mazziotti group's code HQCA (github.com/damazz/HQCA, commit 5733356):
hqca/acse/_quant_A_acse.py measures the full (2N)^4 residual tensor as
Im 2D(exp(i delta H) Psi)/delta, with the 2-RDM unnormalized (no 1/p! prefactor,
hqca/tools/rdm/_spin_rdm.py), and turns every element above a truncation
threshold into one term of the generator, with no restriction to i < j, k < l
and no prefactor; the Euler step (hqca/acse/_euler_acse.py) gives
exp(eps sum 2A^{ik}_{jl} a+_i a+_k a_l a_j), and hqca/core/primitives/_pauli_strings.py
applies exp(i theta P) exactly (as Rz(-2 theta)). The generator equation of the
PRL itself was not checked.

Non-singlet states. The identity then fails, by an amount that depends on the
state: e.g. by 0.46 and 0.73 (relative Frobenius norm on the Fock space) for the
random M_S = 0 triplet and the random doublet of tests/test_optimizers.py. For
every state, sum (M g) e is the spin average of the spin-orbital generator: its
rank-0 part, which is also its Frobenius-orthogonal projection onto the
spin-free two-body operators. The remainder has spin-tensor rank 1 (M_S != 0, or
mixed spins) and rank 2 (S >= 1); on the random H4 states checked (two triplets
with M_S = 0, one with M_S = 1, two doublets and a spin-contaminated vector) it
was pure rank 2 for the M_S = 0 triplets and pure rank 1 for the doublets. It
does not commute with S^2, so the standard step can change <S^2>, and the
standard slope -sum |2A^{ij}_{kl}|^2 is steeper than -<g, M g>. "spin_orbital"
stays a spin-free (S^2-conserving) descent direction for every state.

Estimated residuals. More generally, for every coefficient tensor C that is
antisymmetric in (i, j) and in (k, l), the spin average of
sum C_{ijkl} a+_i a+_j a_l a_k is sum (M c) e with the spin contraction
c[p,q,r,s] = sum_{sigma tau} C_{(p sigma)(r tau)(q sigma)(s tau)}
(:func:`thc_engine_cqe.measure.targets.contract_spin`): the average is the
orthogonal projection of C onto the amplitudes C(c') of the spin-free operators,
whose Gram matrix is M^-1 (above). So if a residual estimate A_est is the spin
contraction of a spin-orbital estimate 2A_est, "spin_orbital" steps along the
spin average of the estimated spin-orbital generator
sum conj(2A_est^{ij}_{kl}) a+_i a+_j a_l a_k, whatever the state and the
estimation error, not along that raw estimated generator, whose non-singlet
error components would change <S^2> (tested on H4, with S_z-violating and
complex errors).

How the optimizers use M (the standard preconditioned forms, e.g. Nocedal &
Wright, Numerical Optimization, sections 5.1 and 7.2; in the coordinates
M^(-1/2) theta they are the unpreconditioned methods):

* steepest descent: d = M g;
* CG: z = M g and d_k = z_k + beta_k d_{k-1}, beta from z (:class:`ConjugateGradient`);
* L-BFGS: H_0 = gamma M with gamma = <s, y>/<y, M y>, and d = M g with an empty
  memory; the pairs keep the unpreconditioned gradients (:class:`LBFGS`).

The driver (:func:`thc_engine_cqe.cqe.run_cqe`) calls ``direction(g, A)`` once
per iteration, with A the residual it acts on (exact, or the estimate A_est of a
``residual_estimator``) and g = conj(A), and ``step_taken(eps)`` after the line
search; the step is Psi_{k+1} = exp(eps_k D_k) Psi_k. Directions are not
rescaled: with ``eps_candidates`` the candidates multiply d.

Charts. The previous search line continues as a straight line in the chart of
the new iterate: exp((eps_k + t) D_k) Psi_k = exp(t D_k) Psi_{k+1}. The previous
direction d_k is therefore exactly a tangent vector at Psi_{k+1}, and after an
exact line search <g_{k+1}, d_k> = 0 up to the accuracy of the search (tested:
below 1e-3 <g_k, d_k>). Comparing gradients of different iterates
(Polak-Ribiere beta, the L-BFGS pairs y_k) is not exact. The chart of Psi_k at
theta = s = eps_k d_k and the chart of Psi_{k+1} at 0 describe the same state,
but exp(S + t U) != exp(t U) exp(S) for non-commuting operators. Since
d/dt exp(S + t U) at t = 0 is (U + [S, U]/2 + O(s^2)) exp(S), the derivative of
the chart-k energy at s along u equals the derivative of the chart-(k+1) energy
at 0 along u plus <[H, [S, U]]>/2 + O(s^2) (expectation values in Psi_{k+1};
tests/test_optimizers.py checks the sign and the O(s^2) remainder with exact
Frechet derivatives on H4). This term is of first order in the step, like y_k
itself, and vanishes at an eigenstate, where <[H, X]> = 0 for every X. CG with
Polak-Ribiere and L-BFGS treat the local coordinates of successive iterates as
one coordinate system (the "locally parameterized" quasi-Newton CQE of Smart,
Boyn & Mazziotti, JCTC 18, 5286 (2022)).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

__all__ = [
    "OPTIMIZERS",
    "METRICS",
    "real_inner",
    "direction_slope",
    "apply_metric",
    "symmetrize_residual",
    "structure_error",
    "SearchDirection",
    "SteepestDescent",
    "ConjugateGradient",
    "LBFGS",
    "make_optimizer",
]

OPTIMIZERS = ("sd", "cg", "lbfgs")
METRICS = ("spin_free", "spin_orbital")


def _check_metric(metric: str) -> str:
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    return metric


def apply_metric(x: np.ndarray, metric: str = "spin_free") -> np.ndarray:
    """M x for the metric of the search directions (module docstring).

    "spin_free": M = I; returns ``x`` itself (no copy), so default directions
    are unchanged bit for bit. "spin_orbital": M = (2 I + T)/3 with
    (T x)[p,q,r,s] = x[p,s,r,q], a new array; ``x`` must be a pair-order
    tensor (N, N, N, N), real or complex. For g = conj(A) of a singlet state,
    sum (M g) e is the spin-orbital ACSE generator sum conj(2A^{ij}_{kl}) a+_i a+_j a_l a_k,
    summed over all spin orbitals i, j, k, l.
    """
    if _check_metric(metric) == "spin_free":
        return x
    x = np.asarray(x)
    if x.ndim != 4 or len(set(x.shape)) != 1:
        raise ValueError(f"the spin-orbital metric acts on pair-order tensors (N, N, N, N), got shape {x.shape}")
    return (2.0 * x + x.transpose(0, 3, 2, 1)) / 3.0


def real_inner(u: np.ndarray, v: np.ndarray) -> float:
    """<u, v> = Re sum conj(u) v, the real inner product on pair-order tensors."""
    return float(np.real(np.vdot(np.asarray(u), np.asarray(v))))


def direction_slope(d: np.ndarray, A: np.ndarray) -> float:
    """dE/deps at eps = 0 along exp(eps D), D = sum d e: -Re sum d A (= -<conj(A), d>).

    This is the expression the CQE driver uses for its slope and sign checks.
    """
    return -float(np.real(np.sum(d * A)))


def structure_error(x: np.ndarray) -> tuple[float, float]:
    """(max |x - x[r,s,p,q]|, max |x + conj(x[q,p,s,r])|): pair-symmetry and anti-Hermiticity defects."""
    x = np.asarray(x)
    pair = float(np.abs(x - x.transpose(2, 3, 0, 1)).max()) if x.size else 0.0
    herm = float(np.abs(x + np.conj(x.transpose(1, 0, 3, 2))).max()) if x.size else 0.0
    return pair, herm


def symmetrize_residual(x: np.ndarray) -> np.ndarray:
    """Orthogonal projection (in <u, v> = Re sum conj(u) v) onto tensors with the structure of A.

    P = (1 + tau)(1 - eta)/4 with the commuting involutions
    tau(x)[p,q,r,s] = x[r,s,p,q] and eta(x)[p,q,r,s] = conj(x[q,p,s,r]); the
    result is pair-symmetric with x[q,p,s,r] = -conj(x[p,q,r,s]). Use it to give
    an elementwise residual estimate the structure that
    :func:`thc_engine_cqe.cqe.run_cqe` requires of ``residual_estimator`` output.
    """
    x = np.asarray(x)
    xp = 0.5 * (x + x.transpose(2, 3, 0, 1))
    return 0.5 * (xp - np.conj(xp.transpose(1, 0, 3, 2)))


class SearchDirection:
    """Base class of the CQE search directions.

    ``direction(g, A)`` returns the generator coefficients d of the next step
    (a new array, or g itself). Optimizers with a history store copies of g
    and d, so a caller may modify g or the returned d in place (for example a
    recorded ``CQEStep.generator``) without changing later directions.
    ``metric`` (METRICS, default "spin_free") is the preconditioner M of the
    module docstring. ``info`` holds diagnostics of the last direction (a dict,
    or None for steepest descent with the default metric); it carries
    "metric" whenever the metric is not the default. ``reset()`` forgets all
    history.
    """

    name = ""

    def __init__(self, metric: str = "spin_free") -> None:
        self.metric = _check_metric(metric)
        self.info: dict | None = None

    def reset(self) -> None:
        self.info = None

    def _metric_info(self, info: dict | None) -> dict | None:
        """``info`` with the metric added when it is not the default (default records unchanged)."""
        if self.metric == "spin_free":
            return info
        return {**(info or {}), "metric": self.metric}

    def direction(self, g: np.ndarray, A: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def step_taken(self, eps: float) -> None:
        """Record the step size eps_k taken along the last direction."""


class SteepestDescent(SearchDirection):
    """d = M g: g = conj(A) itself with the default metric (the M1 path, unchanged).

    With metric "spin_orbital", d = M g is, for singlet states, the standard
    spin-orbital ACSE generator (module docstring); ``info`` = {"metric": ...}.
    """

    name = "sd"

    def direction(self, g: np.ndarray, A: np.ndarray) -> np.ndarray:
        self.info = self._metric_info(None)
        return apply_metric(g, self.metric)


class ConjugateGradient(SearchDirection):
    """Nonlinear conjugate gradient, preconditioned by the metric: d_k = z_k + beta_k d_{k-1}, z = M g.

    Args:
        beta: "pr+" (default), Polak-Ribiere+
            beta = max(0, <z_k, g_k - g_{k-1}> / <z_{k-1}, g_{k-1}>), or "fr",
            Fletcher-Reeves beta = <z_k, g_k> / <z_{k-1}, g_{k-1}>. With the
            default metric z = g, the unpreconditioned rules.
        restart: periodic restart (beta = 0). A cycle starts at every restart
            and holds at most ``restart`` directions, so ``restart=1`` is
            steepest descent and ``restart=2`` alternates z and one conjugate
            step. Default None: no periodic restart.
        metric: "spin_free" (default, M = I) or "spin_orbital" (module docstring).

    Whatever the rule, d_k = z_k (a restart) on the first call and whenever the
    slope -Re sum d_k A_k (:func:`direction_slope`) of the conjugate direction
    is not negative; z_k itself is a descent direction, slope -<g_k, M g_k> < 0.
    Fletcher-Reeves compares only norms, so it does not depend on identifying
    the charts of successive iterates; Polak-Ribiere compares g_k with g_{k-1}
    (see the module docstring). ``info`` = {"beta": beta used, "restart": None
    or "first" / "periodic" / "slope" / "zero" (zero previous g)}, plus "metric"
    for a non-default metric.
    """

    name = "cg"
    BETA_RULES = ("pr+", "fr")

    def __init__(self, beta: str = "pr+", restart: int | None = None, metric: str = "spin_free") -> None:
        super().__init__(metric)
        if beta not in self.BETA_RULES:
            raise ValueError(f"beta must be one of {self.BETA_RULES}, got {beta!r}")
        if restart is not None and (int(restart) != restart or restart < 1):
            raise ValueError("restart must be a positive integer or None")
        self.beta_rule = beta
        self.restart = None if restart is None else int(restart)
        self.reset()

    def reset(self) -> None:
        super().reset()
        self._g_prev: np.ndarray | None = None
        self._z_prev: np.ndarray | None = None
        self._d_prev: np.ndarray | None = None
        self._cycle = 0
        self.n_restarts = 0

    def direction(self, g: np.ndarray, A: np.ndarray) -> np.ndarray:
        g = np.asarray(g)
        z = apply_metric(g, self.metric)          # g itself for the default metric
        beta = 0.0
        reason: str | None = None
        if self._d_prev is None:
            reason = "first"
        elif self.restart is not None and self._cycle >= self.restart:
            reason = "periodic"
        else:
            zg_prev = real_inner(self._z_prev, self._g_prev)
            if zg_prev <= 0.0:
                reason = "zero"
            elif self.beta_rule == "fr":
                beta = real_inner(z, g) / zg_prev
            else:
                beta = max(0.0, real_inner(z, g - self._g_prev) / zg_prev)
        if reason is None:
            d = z + beta * self._d_prev
            if not direction_slope(d, A) < 0.0:
                reason, beta, d = "slope", 0.0, z
        else:
            d = z
        if reason is None:
            self._cycle += 1
        else:
            self._cycle = 1
            if reason != "first":
                self.n_restarts += 1
        # Copies: the caller (and the user, through CQEStep.generator) owns g and d.
        self._g_prev, self._d_prev = g.copy(), d.copy()
        self._z_prev = self._g_prev if z is g else z.copy()
        self.info = self._metric_info({"beta": float(beta), "restart": reason})
        return d


class LBFGS(SearchDirection):
    """Limited-memory BFGS in the local coordinates (two-loop recursion), H_0 = gamma M.

    Pairs: s_k = eps_k d_k (the step actually taken, in the chart of Psi_k) and
    y_k = grad_{k+1} - grad_k = -(g_{k+1} - g_k), formed when the next gradient
    arrives; the metric does not enter them. A pair is skipped (not stored) when
    <s, y> <= curvature_tol ||s|| ||y||, which includes eps_k = 0. The default
    curvature_tol = 1e-2 rejects pairs whose s and y are nearly orthogonal. With
    exact residuals that does not happen on the benchmark systems (60-step runs of
    H4-H8, LiH and N2 are bit for bit the same with 1e-12); with estimated
    residuals such pairs are noise-dominated, and 1e-12 let them drive gamma
    towards zero and the direction up to hundreds of times |g| (observed in the
    finite-shot benchmark: single steps raising E by up to 3.4 Ha on LiH). The two-loop
    recursion (Nocedal & Wright, Algorithm 7.4) with the last ``memory`` pairs
    and H_0 = gamma M, gamma = <s, y> / <y, M y> of the newest pair (M = I and
    gamma = <s, y> / <y, y> for the default metric), gives d = H g = -H grad;
    with an empty memory (first call, or after a reset) d = M g. If d is not a
    descent direction (slope -Re sum d A >= 0), the memory is cleared and
    d = M g. The driver passes A consistent with g = conj(A), and H is symmetric
    positive definite (H_0 = gamma M with gamma > 0, and every stored pair has
    <s, y> > 0), so the slope -<g, H g> is negative for every g != 0: there this
    reset only guards against round-off. A direct call with an A inconsistent
    with g reaches it (tested). ``metric``: "spin_free" (default) or
    "spin_orbital" (module docstring).

    Approximation: the pairs combine the chart of Psi_k (s_k) with gradients of
    two different iterates (y_k), i.e. the local coordinates of successive
    iterates are treated as one coordinate system, the "locally parameterized"
    quasi-Newton CQE of Smart, Boyn & Mazziotti, JCTC 18, 5286 (2022); see the
    module docstring. ``info`` = {"n_pairs": pairs used, "pair": None /
    "stored" / "skipped" (fate of the newest pair), "reset": bool, "gamma"},
    plus "metric" for a non-default metric.
    """

    name = "lbfgs"

    def __init__(self, memory: int = 10, curvature_tol: float = 1e-2, metric: str = "spin_free") -> None:
        super().__init__(metric)
        if int(memory) != memory or memory < 1:
            raise ValueError("memory must be a positive integer")
        if not curvature_tol >= 0.0:
            raise ValueError("curvature_tol must be >= 0")
        self.memory = int(memory)
        self.curvature_tol = float(curvature_tol)
        self.reset()

    def reset(self) -> None:
        super().reset()
        self._pairs: list[tuple[np.ndarray, np.ndarray, float]] = []
        self._g_prev: np.ndarray | None = None
        self._d_prev: np.ndarray | None = None
        self._eps_prev: float | None = None
        self.n_resets = 0
        self.n_skipped = 0

    @property
    def n_pairs(self) -> int:
        return len(self._pairs)

    def step_taken(self, eps: float) -> None:
        self._eps_prev = float(eps)

    def _two_loop(self, g: np.ndarray) -> tuple[np.ndarray, float]:
        q = np.array(g, dtype=np.result_type(g, *(s for s, _, _ in self._pairs)), copy=True)
        alphas = []
        for s, y, rho in reversed(self._pairs):
            a = rho * real_inner(s, q)
            q = q - a * y
            alphas.append(a)
        s, y, _ = self._pairs[-1]
        gamma = real_inner(s, y) / real_inner(y, apply_metric(y, self.metric))
        r = gamma * apply_metric(q, self.metric)
        for (s, y, rho), a in zip(self._pairs, reversed(alphas)):
            b = rho * real_inner(y, r)
            r = r + (a - b) * s
        return r, gamma

    def inverse_hessian_apply(self, g: np.ndarray) -> np.ndarray:
        """H_k g for the current memory: the two-loop recursion (H_0 = gamma M) with the stored pairs,
        M g with an empty memory. A new array; the optimizer's state is not changed.

        After ``direction(g_k, A_k)`` the memory holds the pairs that produced d_k = H_k g_k
        (the newest one, (s_{k-1}, y_{k-1}), formed in that call), so this applies the
        inverse-Hessian operator of d_k to any tensor, e.g. the element fills of a residual
        estimate (``steprules.debiased_slope``). H_k is symmetric positive definite in
        :func:`real_inner` (class docstring).
        """
        g = np.asarray(g)
        if self._pairs:
            return self._two_loop(g)[0]
        return np.array(apply_metric(g, self.metric), copy=True)

    def direction(self, g: np.ndarray, A: np.ndarray) -> np.ndarray:
        g = np.asarray(g)
        info: dict = {"n_pairs": 0, "pair": None, "reset": False, "gamma": None}
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
                self.n_skipped += 1
                info["pair"] = "skipped"
        if self._pairs:
            d, gamma = self._two_loop(g)
            info["gamma"] = float(gamma)
            if not direction_slope(d, A) < 0.0:
                self._pairs.clear()
                self.n_resets += 1
                info.update(reset=True, gamma=None)
                d = apply_metric(g, self.metric)
        else:
            d = apply_metric(g, self.metric)       # g itself for the default metric
        info["n_pairs"] = len(self._pairs)
        # Copies: the caller (and the user, through CQEStep.generator) owns g and d.
        self._g_prev, self._d_prev, self._eps_prev = g.copy(), d.copy(), None
        self.info = self._metric_info(info)
        return d


_CLASSES = {"sd": SteepestDescent, "cg": ConjugateGradient, "lbfgs": LBFGS}
_OPTIONS = {"sd": (), "cg": ("beta", "restart"), "lbfgs": ("memory", "curvature_tol")}


def make_optimizer(name: str = "sd", options: Mapping | None = None,
                   metric: str = "spin_free") -> SearchDirection:
    """A fresh search-direction object: name in OPTIMIZERS, options as keyword arguments.

    "sd" takes no options; "cg" takes beta ("pr+" | "fr") and restart;
    "lbfgs" takes memory (default 10) and curvature_tol (default 1e-2).
    ``metric`` (METRICS) is a separate argument, not an option: "spin_free"
    (default) or "spin_orbital" (module docstring).
    """
    if name not in _CLASSES:
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {name!r}")
    _check_metric(metric)
    opts = dict(options or {})
    if "metric" in opts:
        raise ValueError("metric is not an optimizer option: pass metric= to run_cqe or make_optimizer")
    unknown = sorted(set(opts) - set(_OPTIONS[name]))
    if unknown:
        raise ValueError(f"unknown optimizer_options for {name!r}: {unknown}; "
                         f"allowed: {list(_OPTIONS[name])}")
    return _CLASSES[name](metric=metric, **opts)
