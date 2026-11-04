"""Double factorization (DF) of the ERIs.

First factorization: symmetric vectors L_l with (pq|rs) ~= sum_l L_l[p,q] L_l[r,s],
from pivoted Cholesky (:func:`~thc_engine_cqe.factor.cholesky.cholesky_eri`,
the default) or from the eigendecomposition of the supermatrix (``first='eigh'``,
OpenFermion's ``factorize``).

Second factorization: each leaf is diagonalized, L_l = U_l diag(mu_l) U_l^T, and
only some eigenpairs are kept. The two-body operator becomes

    1/2 sum_pqrs (pq|rs) E_pq E_rs ~= 1/2 sum_l ( sum_k mu_lk n~_lk )^2,

with n~_lk = sum_pq U_l[p,k] U_l[q,k] E_pq the spin-summed number operator of
orbital k of the rotated basis U_l. Each leaf is therefore one orbital rotation
plus a diagonal (Ising) quadratic form.

Truncation, applied in this order:

1. Leaves are sorted by decreasing Frobenius norm ||L_l||_F. Ties within a relative
   1e-12 keep their input order, so eigendecomposition vectors stay in
   eigenvalue order. ``n_leaves`` keeps the first n.
2. ``eig_threshold`` drops eigenpairs inside each leaf.
   * ``threshold_rule='absolute'``: keep k iff |mu_lk| > eig_threshold.
   * ``threshold_rule='openfermion'``: keep k iff (sum_j |mu_lj|) |mu_lk| > eig_threshold,
     and stop at the first leaf that keeps nothing. This is exactly
     ``openfermion.resource_estimates.df.factorize``. Together with
     ``first='eigh', tol=0.0`` it reproduces OpenFermion's factors (tested).
   Leaves that keep no eigenpair are dropped. If no leaf survives, the result
   is an empty factorization whose ERIs are zero (tested).

Lambda (:meth:`DoubleFactorization.lam`) follows OpenFermion's
``df.compute_lambda`` (von Burg et al., PRR 3, 033055 (2021); Lee et al.,
PRX Quantum 2, 030305 (2021)):

    lambda = sum_k |eig_k(T')| + 1/4 sum_l ( sum_k |mu_lk| )^2,

with T'_pq = h_pq - 1/2 sum_r (pr|rq) + sum_r (rr|pq) evaluated with the exact
ERIs (see :mod:`~thc_engine_cqe.factor.eri_factor`). The 1/4 is the convention of
those papers. It comes from writing each leaf in the shifted form
-1/2 sum_{k sigma} mu_lk Z_{k sigma}, plus the factor 1/2 gained by block-encoding
its square through a reflection. The formula is taken from OpenFermion and is not
re-derived by any test here. The tests check that the numbers agree with
OpenFermion's code.
"""

from __future__ import annotations

import functools
from typing import Sequence

import numpy as np

from .cholesky import cholesky_eri
from .eri_factor import ERIFactor, LambdaBreakdown, check_eri, lambda_one_body

__all__ = ["DoubleFactorization", "double_factorize", "eigendecomposition_vectors"]

_RULES = ("absolute", "openfermion")


def eigendecomposition_vectors(eri: np.ndarray, tol: float = 0.0) -> np.ndarray:
    """First-stage vectors L_l = sqrt(w_l) reshape(u_l) for supermatrix eigenpairs w_l > tol.

    Eigenvalues are taken in decreasing order. This is OpenFermion's
    ``utils.eigendecomp(eri.reshape(N^2, N^2), tol)``, reshaped to (n_vec, N, N).
    For pair-symmetric ERIs the vectors with w_l well above round-off are
    symmetric matrices.
    """
    eri = check_eri(eri)
    n = eri.shape[0]
    w, v = np.linalg.eigh(eri.reshape(n * n, n * n))
    w, v = w[::-1], v[:, ::-1]
    idx = np.where(w > tol)[0]
    vecs = v[:, idx] * np.sqrt(w[idx])
    return np.ascontiguousarray(vecs.T.reshape(len(idx), n, n))


def _sort_by_norm(L: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
    """Stable order by decreasing ||L_l||_F; norms within rtol * max are ties.

    Ties keep their input order (stable sort with a tolerant comparison), so an
    input that is already sorted up to round-off, such as eigendecomposition
    vectors with degenerate eigenvalues, is left unchanged.
    """
    norms = np.sqrt(np.einsum("lpq,lpq->l", L, L))
    if norms.size == 0 or norms.max() == 0:
        return np.arange(len(norms))
    tol = rtol * float(norms.max())

    def cmp(i: int, j: int) -> int:
        d = norms[j] - norms[i]
        return 0 if abs(d) <= tol else (1 if d > 0 else -1)

    return np.array(sorted(range(len(norms)), key=functools.cmp_to_key(cmp)), dtype=int)


class DoubleFactorization(ERIFactor):
    """Leaves L_l = U_l diag(mu_l) U_l^T with (pq|rs) ~= sum_l L_l[p,q] L_l[r,s].

    Attributes:
        eigenvalues: list of (n_l,) arrays mu_l.
        eigenvectors: list of (N, n_l) arrays U_l (orthonormal columns).
        info: provenance (first factorization, tolerances, truncation).

    ``n_params`` counts the stored reals, sum_l n_l (N + 1). The DF cost
    parameters of Lee et al. are ``n_leaves`` (their L) and
    ``n_eigenvectors`` (their Xi = sum_l n_l).
    """

    kind = "df"

    def __init__(self, eigenvalues: Sequence[np.ndarray], eigenvectors: Sequence[np.ndarray],
                 norb: int | None = None, info: dict | None = None):
        mus = [np.asarray(m, dtype=float).reshape(-1) for m in eigenvalues]
        us = [np.asarray(u, dtype=float) for u in eigenvectors]
        if len(mus) != len(us):
            raise ValueError("eigenvalues and eigenvectors must have the same number of leaves")
        if norb is None:
            if not us:
                raise ValueError("norb is required for an empty factorization")
            norb = us[0].shape[0]
        for m, u in zip(mus, us):
            if u.ndim != 2 or u.shape != (norb, m.size):
                raise ValueError(f"leaf eigenvectors must have shape (N, n_l) = ({norb}, {m.size}), "
                                 f"got {u.shape}")
        self.eigenvalues = mus
        self.eigenvectors = us
        self.norb = int(norb)
        self.info = dict(info or {})

    # -------------------------------------------------------------- builders
    @classmethod
    def from_vectors(cls, L: np.ndarray, *, n_leaves: int | None = None,
                     eig_threshold: float | None = None, threshold_rule: str = "absolute",
                     info: dict | None = None) -> "DoubleFactorization":
        """Second factorization of given first-stage vectors L (n_vec, N, N).

        See the module docstring for the truncation rules. Each L_l is
        diagonalized with ``numpy.linalg.eigh``, which reads the lower
        triangle, as OpenFermion does.
        """
        L = np.asarray(L, dtype=float)
        if L.ndim != 3 or L.shape[1] != L.shape[2]:
            raise ValueError(f"L must have shape (n_vec, N, N), got {L.shape}")
        if threshold_rule not in _RULES:
            raise ValueError(f"threshold_rule must be one of {_RULES}")
        n = L.shape[1]
        order = _sort_by_norm(L)
        if n_leaves is not None:
            order = order[:max(0, int(n_leaves))]
        mus, us = [], []
        stopped_at = None
        for l in order:
            e, v = np.linalg.eigh(L[l])
            if eig_threshold is None:
                keep = np.ones(e.size, dtype=bool)
            elif threshold_rule == "absolute":
                keep = np.abs(e) > eig_threshold
            else:  # openfermion
                keep = np.sum(np.abs(e)) * np.abs(e) > eig_threshold
                if not keep.any():
                    stopped_at = int(len(mus))
                    break
            if keep.any():
                mus.append(e[keep])
                us.append(v[:, keep])
        meta = {"n_vectors_in": int(L.shape[0]), "n_leaves_requested": n_leaves,
                "eig_threshold": eig_threshold, "threshold_rule": threshold_rule,
                "openfermion_stop": stopped_at}
        meta.update(info or {})
        return cls(mus, us, norb=n, info=meta)

    # ------------------------------------------------------------ properties
    @property
    def n_leaves(self) -> int:
        return len(self.eigenvalues)

    @property
    def n_eigenvectors(self) -> int:
        """Xi = total number of retained eigenpairs."""
        return int(sum(m.size for m in self.eigenvalues))

    def leaves(self) -> np.ndarray:
        """(n_leaves, N, N) array of L_l = U_l diag(mu_l) U_l^T."""
        out = np.zeros((self.n_leaves, self.norb, self.norb))
        for l, (m, u) in enumerate(zip(self.eigenvalues, self.eigenvectors)):
            out[l] = (u * m) @ u.T
        return out

    def leaf_one_norms(self) -> np.ndarray:
        """sum_k |mu_lk| for every leaf."""
        return np.array([np.sum(np.abs(m)) for m in self.eigenvalues])

    def to_openfermion(self) -> np.ndarray:
        """Leaves in OpenFermion's layout (N, N, n_leaves), as taken by ``df.compute_lambda``."""
        return np.ascontiguousarray(self.leaves().transpose(1, 2, 0))

    # ------------------------------------------------------------ interface
    def to_eri(self) -> np.ndarray:
        """sum_l L_l[p,q] L_l[r,s]; the zero tensor when no leaf is kept."""
        n = self.norb
        lv = self.leaves().reshape(self.n_leaves, n * n)
        return (lv.T @ lv).reshape(n, n, n, n)

    def n_params(self) -> int:
        return self.n_eigenvectors * (self.norb + 1)

    def lam(self, h1: np.ndarray, eri_exact: np.ndarray | None = None) -> LambdaBreakdown:
        """OpenFermion's DF lambda: sum |eig(T')| + 1/4 sum_l (sum_k |mu_lk|)^2.

        T' uses ``eri_exact`` when given (OpenFermion's convention), otherwise
        the factorized ERIs. ``details`` holds ``lambda_T``, ``lambda_F``,
        ``t_eigenvalues``, ``leaf_one_norms``, ``n_leaves``,
        ``n_eigenvectors`` and ``t_eri`` ('exact' or 'factorized').
        """
        eri_t = self.to_eri() if eri_exact is None else check_eri(eri_exact, "eri_exact")
        lam_t, t = lambda_one_body(h1, eri_t)
        norms = self.leaf_one_norms()
        lam_f = float(0.25 * np.sum(norms ** 2))
        return LambdaBreakdown(
            lam_total=lam_t + lam_f, lam_one_body=lam_t, lam_two_body=lam_f,
            details={"lambda_T": lam_t, "lambda_F": lam_f, "t_eigenvalues": t,
                     "leaf_one_norms": norms, "n_leaves": self.n_leaves,
                     "n_eigenvectors": self.n_eigenvectors,
                     "t_eri": "factorized" if eri_exact is None else "exact"})


def double_factorize(eri: np.ndarray, *, tol: float = 1e-10, first: str = "cholesky",
                     n_leaves: int | None = None, eig_threshold: float | None = None,
                     threshold_rule: str = "absolute",
                     max_vectors: int | None = None) -> DoubleFactorization:
    """Double factorization of (pq|rs).

    Args:
        eri: (N, N, N, N) real ERIs, 8-fold symmetric, positive semidefinite.
        tol: first-stage tolerance. For ``first='cholesky'`` it is the max
            remaining diagonal (:func:`cholesky_eri`); for ``first='eigh'``
            supermatrix eigenvalues <= tol are dropped (OpenFermion uses 0.0).
        first: ``'cholesky'`` (default) or ``'eigh'``.
        n_leaves: keep only the n leaves of largest Frobenius norm.
        eig_threshold, threshold_rule: second-stage truncation (module docstring).
        max_vectors: cap on the number of Cholesky vectors.

    Without truncation (``n_leaves=None, eig_threshold=None``) the result
    reproduces the first-stage approximation exactly (tested).
    """
    eri = check_eri(eri)
    if first == "cholesky":
        L = cholesky_eri(eri, tol=tol, max_vectors=max_vectors)
    elif first == "eigh":
        L = eigendecomposition_vectors(eri, tol=tol)
        if max_vectors is not None:
            L = L[:max_vectors]
    else:
        raise ValueError("first must be 'cholesky' or 'eigh'")
    return DoubleFactorization.from_vectors(
        L, n_leaves=n_leaves, eig_threshold=eig_threshold, threshold_rule=threshold_rule,
        info={"first": first, "first_tol": tol, "max_vectors": max_vectors})
