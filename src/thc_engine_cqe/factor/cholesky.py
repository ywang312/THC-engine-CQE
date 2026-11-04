"""Pivoted (modified) Cholesky decomposition of the ERI supermatrix.

The supermatrix V[(pq), (rs)] = (pq|rs) of real orbitals is symmetric positive
semidefinite. Pivoted Cholesky writes it as

    (pq|rs) ~= sum_l L_l[p, q] L_l[r, s],      L_l = L_l^T,

choosing as the next pivot the pair (pq) with the largest remaining diagonal
R[(pq), (pq)], where R = V - sum_l vec(L_l) vec(L_l)^T is the residual.

Error bound: R stays positive semidefinite, so |R[(pq), (rs)]| <=
sqrt(R[(pq),(pq)] R[(rs),(rs)]). Stopping once the largest remaining diagonal
is below ``tol`` therefore guarantees max_pqrs |(pq|rs) - sum_l L_l L_l| < tol
(up to round-off). tests/test_factor_eri.py checks this bound.

Symmetry: because (pq|rs) = (qp|rs), the algorithm works on the packed pairs
p <= q only (N (N + 1) / 2 rows instead of N^2). Unpacking makes every L_l
exactly symmetric.
"""

from __future__ import annotations

import warnings

import numpy as np

from .eri_factor import check_eri

__all__ = ["cholesky_eri", "packed_supermatrix"]


def packed_supermatrix(eri: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """W[a, b] = (p_a q_a | p_b q_b) over packed pairs p <= q (``numpy.triu_indices``)."""
    n = eri.shape[0]
    p, q = np.triu_indices(n)
    w = eri[p[:, None], q[:, None], p[None, :], q[None, :]]
    return np.ascontiguousarray(w), (p, q)


def cholesky_eri(eri: np.ndarray, tol: float = 1e-10, max_vectors: int | None = None,
                 symmetry_tol: float = 1e-10) -> np.ndarray:
    """Pivoted Cholesky vectors of (pq|rs).

    Args:
        eri: (N, N, N, N) real tensor (pq|rs) with 8-fold symmetry. It must be
            positive semidefinite as a supermatrix, which exact ERIs are.
        tol: stop when the largest remaining diagonal element is below ``tol``.
            The value is absolute, in the units of ``eri``. The effective
            threshold is never below 100 * machine-eps * max diag, so that
            no pivot is taken on pure round-off.
        max_vectors: optional cap on the number of vectors.
        symmetry_tol: maximum allowed violation of (pq|rs) = (qp|rs) = (rs|pq),
            relative to max(1, max |eri|). It is therefore absolute for integrals
            below 1 in magnitude.

    Returns:
        L with shape (n_vec, N, N), each L[l] symmetric, such that
        ``einsum('lpq,lrs->pqrs', L, L)`` approximates ``eri``. Unless
        ``max_vectors`` stops the loop first, the max-abs error is below the
        effective threshold max(tol, 100 eps max diag).

    Raises:
        ValueError: if ``eri`` lacks the required symmetry.

    Warns:
        RuntimeWarning: if the remaining diagonal becomes significantly
            negative, which means the supermatrix is not positive semidefinite
            (e.g. an approximate ERI model) and the error bound does not hold.
    """
    eri = check_eri(eri)
    n = eri.shape[0]
    scale = float(np.max(np.abs(eri))) if eri.size else 0.0
    asym = max(float(np.max(np.abs(eri - eri.transpose(1, 0, 2, 3)))),
               float(np.max(np.abs(eri - eri.transpose(2, 3, 0, 1)))))
    if asym > symmetry_tol * max(scale, 1.0):
        raise ValueError(f"eri violates (pq|rs) = (qp|rs) = (rs|pq) by {asym:.3e}")

    w, (pi, qi) = packed_supermatrix(eri)
    npair = w.shape[0]
    d = np.diag(w).copy()
    dmax = float(d.max()) if npair else 0.0
    floor = max(float(tol), 100.0 * np.finfo(float).eps * max(dmax, 0.0))
    nmax = npair if max_vectors is None else max(0, min(int(max_vectors), npair))

    lp = np.zeros((nmax, npair))
    k = 0
    while k < nmax:
        a = int(np.argmax(d))
        if not d[a] >= floor:  # also stops on NaN
            break
        col = w[:, a] - lp[:k].T @ lp[:k, a]
        lp[k] = col / np.sqrt(d[a])
        d -= lp[k] ** 2
        d[a] = 0.0  # exact in exact arithmetic; avoids re-pivoting on round-off
        k += 1

    if npair and float(d.min()) < -floor:
        warnings.warn(f"ERI supermatrix is not positive semidefinite (remaining diagonal "
                      f"{float(d.min()):.3e}); the Cholesky error bound does not apply",
                      RuntimeWarning, stacklevel=2)

    out = np.zeros((k, n, n))
    out[:, pi, qi] = lp[:k]
    out[:, qi, pi] = lp[:k]
    return out
