"""Common interface for factorized electron-repulsion integrals (ERIs).

Every factorization approximates the chemist-notation tensor of N real orbitals,

    eri[p, q, r, s] = (pq|rs),

which enters H = ecore + sum_pq h_pq E_pq + 1/2 sum_pqrs (pq|rs) e_{pq,rs}
(docs/conventions.md). A factorization is turned into a model Hamiltonian with
``active.with_eri(factor.to_eri())``.

Block-encoding 1-norms (lambda)
-------------------------------
The lambda values follow the conventions of OpenFermion's ``resource_estimates``
package (Lee et al., PRX Quantum 2, 030305 (2021)). They start from the exact
operator identity (checked on CI vectors in tests/test_factor_eri.py)

    H - ecore = sum_pq T'_pq E_pq + 1/2 sum_pqrs (pq|rs) Et_pq Et_rs - 1/2 sum_pr (pp|rr),

    Et_pq = E_pq - delta_pq   (= sum_sigma (a+_{p sigma} a_{q sigma} - delta_pq / 2)),
    T'_pq = h_pq - 1/2 sum_r (pr|rq) + sum_r (rr|pq).

Each Et_pq is traceless and bilinear in Majorana operators. For real 8-fold-
symmetric (pq|rs), the shifted two-body part 1/2 sum (pq|rs) Et_pq Et_rs has no
degree-2 Majorana terms. It consists of degree-4 terms plus the constant
1/4 sum_pq (pq|qp), which comes from tr(Et_pq Et_rs) / tr(1) = delta_ps delta_qr / 2
over the full Fock space. Every degree-2 term of H therefore comes from
sum T'_pq E_pq. The degree-2 coefficients are checked one by one against
OpenFermion's Majorana representation, and the constant against both that
representation and the Fock-space oracle. Lambda excludes constants. DF and THC
write the shifted two-body part as sums of products of such one-body operators.
The price of the shift is the Coulomb-like term sum_r (rr|pq) in T'. In the
eigenbasis of T', sum_pq T'_pq E_pq = sum_{k sigma} t_k n_{k sigma}
= const - 1/2 sum_{k sigma} t_k Z_{k sigma}, so the one-body 1-norm is
lambda_T = sum_k |t_k| (the factor 1/2 cancels against the spin sum).

OpenFermion evaluates T' with the *exact* ERIs and only the two-body part with
the factorized ones. Every ``lam`` method therefore takes the exact ERIs as
``eri_exact``. If it is omitted, the factorized ERIs are used for T' as well,
and ``details['t_eri']`` records which choice was made.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "LambdaBreakdown",
    "ERIFactor",
    "DenseERI",
    "one_body_tprime",
    "lambda_one_body",
    "pauli_lambda",
    "eri_errors",
    "check_eri",
]


@dataclass
class LambdaBreakdown:
    """1-norm of a block encoding, split into one- and two-body parts.

    ``lam_total = lam_one_body + lam_two_body``. ``details`` holds
    method-specific intermediate values (documented by each ``lam`` method).
    """

    lam_total: float
    lam_one_body: float
    lam_two_body: float
    details: dict = field(default_factory=dict)


def check_eri(eri: np.ndarray, name: str = "eri") -> np.ndarray:
    """Return ``eri`` as a real float array of shape (N, N, N, N), or raise."""
    eri = np.asarray(eri)
    if eri.ndim != 4 or len(set(eri.shape)) != 1:
        raise ValueError(f"{name} must have shape (N, N, N, N), got {eri.shape}")
    if np.iscomplexobj(eri):
        raise ValueError(f"{name} must be real (complex orbitals are not supported)")
    return np.asarray(eri, dtype=float)


def one_body_tprime(h1: np.ndarray, eri: np.ndarray,
                    eri_coulomb: np.ndarray | None = None) -> np.ndarray:
    """T'_pq = h_pq - 1/2 sum_r (pr|rq) + sum_r (rr|pq).

    This is exactly OpenFermion's
    ``h1 - 0.5 * einsum("illj->ij", eri) + einsum("llij->ij", eri_coulomb)``.
    ``eri_coulomb`` defaults to ``eri``; passing the factorized tensor here
    reproduces OpenFermion's THC option ``use_eri_thc_for_t=True``.
    """
    eri_c = eri if eri_coulomb is None else eri_coulomb
    return (np.asarray(h1, dtype=float) - 0.5 * np.einsum("illj->ij", eri)
            + np.einsum("llij->ij", eri_c))


def lambda_one_body(h1: np.ndarray, eri: np.ndarray,
                    eri_coulomb: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """(lambda_T, eigenvalues of T') with lambda_T = sum_k |t_k|.

    Like OpenFermion, this uses ``numpy.linalg.eigvalsh`` (lower triangle of T'),
    which is exact for real 8-fold-symmetric integrals because T' is symmetric.
    """
    t = np.linalg.eigvalsh(one_body_tprime(h1, eri, eri_coulomb))
    return float(np.sum(np.abs(t))), t


def pauli_lambda(h1: np.ndarray, eri: np.ndarray,
                 eri_exact: np.ndarray | None = None) -> LambdaBreakdown:
    """1-norm of the Jordan-Wigner (Pauli) representation of H, identity excluded.

    Closed form (Koridon et al., Phys. Rev. Research 3, 033127 (2021)), checked
    against ``openfermion.jordan_wigner`` in tests/test_factor_eri.py:

        lambda_1 = sum_pq |T'_pq|                       (elementwise, not eigenvalues)
        lambda_2 = 1/4 sum_pqrs |(pq|rs)|
                   + 1/2 sum_{p>r, s>q} |(pq|rs) - (ps|rq)|

    T' uses ``eri_exact`` if given (as for DF/THC), otherwise ``eri``. The
    two-body part always uses ``eri``. If the two tensors differ, the result
    mixes them, as the DF/THC lambdas do. It is then not the Pauli 1-norm of
    a single Hamiltonian.
    """
    eri = check_eri(eri)
    eri_t = eri if eri_exact is None else check_eri(eri_exact, "eri_exact")
    n = eri.shape[0]
    tp = one_body_tprime(h1, eri_t)
    lam1 = float(np.sum(np.abs(tp)))
    exch = eri - eri.transpose(0, 3, 2, 1)  # (pq|rs) - (ps|rq)
    idx = np.arange(n)
    mask = (idx[:, None, None, None] > idx[None, None, :, None]) & \
           (idx[None, None, None, :] > idx[None, :, None, None])  # p > r and s > q
    lam2_coulomb = 0.25 * float(np.sum(np.abs(eri)))
    lam2_exchange = 0.5 * float(np.sum(np.abs(exch[mask])))
    lam2 = lam2_coulomb + lam2_exchange
    return LambdaBreakdown(
        lam_total=lam1 + lam2, lam_one_body=lam1, lam_two_body=lam2,
        details={
            "method": "pauli (Jordan-Wigner) 1-norm",
            "lambda_2_coulomb": lam2_coulomb,
            "lambda_2_exchange": lam2_exchange,
            "t_eri": "factorized" if eri_exact is None else "exact",
        })


def eri_errors(eri_approx: np.ndarray, eri_exact: np.ndarray) -> dict:
    """Frobenius norm, relative Frobenius norm and max-abs error of an ERI model."""
    d = np.asarray(eri_approx) - np.asarray(eri_exact)
    fro = float(np.linalg.norm(d.ravel()))
    ref = float(np.linalg.norm(np.asarray(eri_exact).ravel()))
    return {"fro": fro, "rel_fro": fro / ref if ref > 0 else float("nan"),
            "max_abs": float(np.max(np.abs(d))) if d.size else 0.0}


class ERIFactor(ABC):
    """Abstract factorized ERI tensor.

    Attributes:
        kind: short identifier (``'dense'``, ``'df'``, ``'thc'``).
        norb: number of spatial orbitals N.
    """

    kind: str = "abstract"
    norb: int

    @abstractmethod
    def to_eri(self) -> np.ndarray:
        """Dense (N, N, N, N) tensor (pq|rs) represented by the factorization."""

    @abstractmethod
    def n_params(self) -> int:
        """Number of real numbers stored by the representation (see subclasses)."""

    @abstractmethod
    def lam(self, h1: np.ndarray, eri_exact: np.ndarray | None = None) -> LambdaBreakdown:
        """Block-encoding 1-norm; ``eri_exact`` is used for the one-body term T'."""

    # ------------------------------------------------------------- helpers
    def errors(self, eri_exact: np.ndarray) -> dict:
        """:func:`eri_errors` of :meth:`to_eri` against ``eri_exact``."""
        return eri_errors(self.to_eri(), eri_exact)

    def apply_to(self, active, label: str | None = None):
        """The :class:`~thc_engine_cqe.integrals.ActiveSpace` with these ERIs."""
        if active.norb != self.norb:
            raise ValueError(f"factor has {self.norb} orbitals, active space {active.norb}")
        lab = f"{active.label} [{self.kind}]" if label is None else label
        return active.with_eri(self.to_eri(), label=lab)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(kind={self.kind!r}, norb={self.norb}, n_params={self.n_params()})"


class DenseERI(ERIFactor):
    """The unfactorized tensor (reference point for parameter counts and lambda).

    ``n_params`` counts the unique elements of a real 8-fold-symmetric tensor,
    M (M + 1) / 2 with M = N (N + 1) / 2. ``lam`` is the Jordan-Wigner Pauli
    1-norm (:func:`pauli_lambda`), the natural lambda without factorization.
    """

    kind = "dense"

    def __init__(self, eri: np.ndarray):
        self.eri = check_eri(eri)
        self.norb = self.eri.shape[0]

    def to_eri(self) -> np.ndarray:
        return self.eri.copy()

    def n_params(self) -> int:
        npair = self.norb * (self.norb + 1) // 2
        return npair * (npair + 1) // 2

    def lam(self, h1: np.ndarray, eri_exact: np.ndarray | None = None) -> LambdaBreakdown:
        return pauli_lambda(h1, self.eri, eri_exact)
