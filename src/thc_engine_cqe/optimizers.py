"""Search directions for the CQE: steepest descent, conjugate gradient and L-BFGS.

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
(docs/conventions.md). Every direction below is a combination of g's and earlier
directions with real coefficients, so it stays in V (tests/test_optimizers.py
checks pair symmetry and anti-Hermiticity of every direction to 1e-12 relative).

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
    "real_inner",
    "direction_slope",
    "symmetrize_residual",
    "structure_error",
    "SearchDirection",
    "SteepestDescent",
    "ConjugateGradient",
    "LBFGS",
    "make_optimizer",
]

OPTIMIZERS = ("sd", "cg", "lbfgs")


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
    ``info`` holds diagnostics of the last direction (a dict, or None for
    steepest descent). ``reset()`` forgets all history.
    """

    name = ""

    def __init__(self) -> None:
        self.info: dict | None = None

    def reset(self) -> None:
        self.info = None

    def direction(self, g: np.ndarray, A: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def step_taken(self, eps: float) -> None:
        """Record the step size eps_k taken along the last direction."""


class SteepestDescent(SearchDirection):
    """d = g = conj(A). Returns g itself, so the driver's M1 path is unchanged."""

    name = "sd"

    def direction(self, g: np.ndarray, A: np.ndarray) -> np.ndarray:
        return g


class ConjugateGradient(SearchDirection):
    """Nonlinear conjugate gradient, d_k = g_k + beta_k d_{k-1}.

    Args:
        beta: "pr+" (default), Polak-Ribiere+
            beta = max(0, <g_k, g_k - g_{k-1}> / <g_{k-1}, g_{k-1}>), or "fr",
            Fletcher-Reeves beta = <g_k, g_k> / <g_{k-1}, g_{k-1}>.
        restart: periodic restart (beta = 0). A cycle starts at every restart
            and holds at most ``restart`` directions, so ``restart=1`` is
            steepest descent and ``restart=2`` alternates g and one conjugate
            step. Default None: no periodic restart.

    Whatever the rule, d_k = g_k (a restart) on the first call and whenever the
    slope -Re sum d_k A_k (:func:`direction_slope`) of the conjugate direction
    is not negative. Fletcher-Reeves compares only norms, so it does not depend
    on identifying the charts of successive iterates; Polak-Ribiere compares
    g_k with g_{k-1} (see the module docstring). ``info`` = {"beta": beta used,
    "restart": None or "first" / "periodic" / "slope" / "zero" (zero previous g)}.
    """

    name = "cg"
    BETA_RULES = ("pr+", "fr")

    def __init__(self, beta: str = "pr+", restart: int | None = None) -> None:
        super().__init__()
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
        self._d_prev: np.ndarray | None = None
        self._cycle = 0
        self.n_restarts = 0

    def direction(self, g: np.ndarray, A: np.ndarray) -> np.ndarray:
        g = np.asarray(g)
        beta = 0.0
        reason: str | None = None
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
        if reason is None:
            self._cycle += 1
        else:
            self._cycle = 1
            if reason != "first":
                self.n_restarts += 1
        # Copies: the caller (and the user, through CQEStep.generator) owns g and d.
        self._g_prev, self._d_prev = g.copy(), d.copy()
        self.info = {"beta": float(beta), "restart": reason}
        return d


class LBFGS(SearchDirection):
    """Limited-memory BFGS in the local coordinates (two-loop recursion).

    Pairs: s_k = eps_k d_k (the step actually taken, in the chart of Psi_k) and
    y_k = grad_{k+1} - grad_k = -(g_{k+1} - g_k), formed when the next gradient
    arrives. A pair is skipped (not stored) when <s, y> <= curvature_tol ||s|| ||y||,
    which includes eps_k = 0. The two-loop recursion (Nocedal & Wright,
    Algorithm 7.4) with the last ``memory`` pairs and H_0 = gamma I,
    gamma = <s, y> / <y, y> of the newest pair, gives d = H g = -H grad; with an
    empty memory (first call, or after a reset) d = g. If d is not a descent
    direction (slope -Re sum d A >= 0), the memory is cleared and d = g.

    Approximation: the pairs combine the chart of Psi_k (s_k) with gradients of
    two different iterates (y_k), i.e. the local coordinates of successive
    iterates are treated as one coordinate system, the "locally parameterized"
    quasi-Newton CQE of Smart, Boyn & Mazziotti, JCTC 18, 5286 (2022); see the
    module docstring. ``info`` = {"n_pairs": pairs used, "pair": None /
    "stored" / "skipped" (fate of the newest pair), "reset": bool, "gamma"}.
    """

    name = "lbfgs"

    def __init__(self, memory: int = 10, curvature_tol: float = 1e-12) -> None:
        super().__init__()
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
        gamma = real_inner(s, y) / real_inner(y, y)
        r = gamma * q
        for (s, y, rho), a in zip(self._pairs, reversed(alphas)):
            b = rho * real_inner(y, r)
            r = r + (a - b) * s
        return r, gamma

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
                d = g
        else:
            d = g
        info["n_pairs"] = len(self._pairs)
        # Copies: the caller (and the user, through CQEStep.generator) owns g and d.
        self._g_prev, self._d_prev, self._eps_prev = g.copy(), d.copy(), None
        self.info = info
        return d


_CLASSES = {"sd": SteepestDescent, "cg": ConjugateGradient, "lbfgs": LBFGS}
_OPTIONS = {"sd": (), "cg": ("beta", "restart"), "lbfgs": ("memory", "curvature_tol")}


def make_optimizer(name: str = "sd", options: Mapping | None = None) -> SearchDirection:
    """A fresh search-direction object: name in OPTIMIZERS, options as keyword arguments.

    "sd" takes no options; "cg" takes beta ("pr+" | "fr") and restart;
    "lbfgs" takes memory (default 10) and curvature_tol (default 1e-12).
    """
    if name not in _CLASSES:
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {name!r}")
    opts = dict(options or {})
    unknown = sorted(set(opts) - set(_OPTIONS[name]))
    if unknown:
        raise ValueError(f"unknown optimizer_options for {name!r}: {unknown}; "
                         f"allowed: {list(_OPTIONS[name])}")
    return _CLASSES[name](**opts)
