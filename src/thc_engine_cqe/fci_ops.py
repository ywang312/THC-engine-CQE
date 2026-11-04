"""Operators on PySCF CI vectors (exact emulation of small CQE runs).

A CI vector is an array of shape (n_alpha_strings, n_beta_strings) in PySCF's
determinant basis (fixed n_alpha, n_beta). Real and complex vectors are both
supported; complex inputs are handled by splitting into real and imaginary
parts because PySCF's kernels are real.

Verified conventions (see tests/test_conventions.py, which checks every one of
these against an independent Jordan-Wigner implementation):

* ``one_body(f)``      applies  sum_pq f[p,q] E_pq
* ``two_body_ee(m)``   applies  sum_pqrs m[p,q,r,s] E_pq E_rs
* ``two_body(c)``      applies  sum_pqrs c[p,q,r,s] e_{pq,rs},
                       e_{pq,rs} = E_pq E_rs - delta_qr E_ps
* ``Hamiltonian.apply`` applies the electronic Hamiltonian (no ecore).

Note: PySCF's ``direct_nosym.contract_2e(m)`` computes sum m[p,q,r,s] E_rs E_pq
(the reverse operator order); :func:`two_body_ee` compensates for this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import scipy.linalg

from .integrals import ActiveSpace

__all__ = [
    "FCISpace",
    "Hamiltonian",
    "one_body",
    "two_body_ee",
    "two_body",
    "expm_apply",
    "KrylovPropagator",
    "inner",
    "normalize",
]


def inner(a: np.ndarray, b: np.ndarray) -> complex | float:
    """<a|b> for CI arrays (conjugates a)."""
    return np.vdot(np.asarray(a).ravel(), np.asarray(b).ravel())


def normalize(ci: np.ndarray) -> np.ndarray:
    return ci / np.sqrt(inner(ci, ci).real)


@dataclass(frozen=True)
class FCISpace:
    """Determinant space for ``norb`` spatial orbitals and ``nelec`` electrons."""

    norb: int
    nelec: tuple[int, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "nelec", tuple(int(x) for x in self.nelec))
        from pyscf.fci import cistring

        na = cistring.num_strings(self.norb, self.nelec[0])
        nb = cistring.num_strings(self.norb, self.nelec[1])
        object.__setattr__(self, "_shape", (na, nb))
        link_a = cistring.gen_linkstr_index(range(self.norb), self.nelec[0])
        link_b = cistring.gen_linkstr_index(range(self.norb), self.nelec[1])
        object.__setattr__(self, "_link_nosym", (link_a, link_b))

    @classmethod
    def of(cls, active: ActiveSpace) -> "FCISpace":
        return cls(active.norb, active.nelec)

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape  # type: ignore[attr-defined]

    @property
    def size(self) -> int:
        return self.shape[0] * self.shape[1]

    @property
    def n_electrons(self) -> int:
        return self.nelec[0] + self.nelec[1]

    def hf(self) -> np.ndarray:
        """The determinant occupying the lowest orbitals (PySCF address 0)."""
        ci = np.zeros(self.shape)
        ci[0, 0] = 1.0
        return ci

    def random(self, rng: np.random.Generator, complex_: bool = False) -> np.ndarray:
        ci = rng.standard_normal(self.shape)
        if complex_:
            ci = ci + 1j * rng.standard_normal(self.shape)
        return normalize(ci)


def _split_apply(fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 op: np.ndarray, ci: np.ndarray) -> np.ndarray:
    """Apply a real-kernel linear map fn(op, ci) to complex op and/or ci."""
    op_c, ci_c = np.iscomplexobj(op), np.iscomplexobj(ci)
    if not op_c and not ci_c:
        return fn(np.ascontiguousarray(op, dtype=float), np.ascontiguousarray(ci, dtype=float))
    opr = np.ascontiguousarray(np.real(op), dtype=float)
    opi = np.ascontiguousarray(np.imag(op), dtype=float) if op_c else None
    cir = np.ascontiguousarray(np.real(ci), dtype=float)
    cii = np.ascontiguousarray(np.imag(ci), dtype=float) if ci_c else None
    out = fn(opr, cir).astype(complex)
    if cii is not None:
        out += 1j * fn(opr, cii)
    if opi is not None:
        out += 1j * fn(opi, cir)
        if cii is not None:
            out -= fn(opi, cii)
    return out


def one_body(space: FCISpace, f: np.ndarray, ci: np.ndarray) -> np.ndarray:
    """sum_pq f[p,q] E_pq |ci>."""
    from pyscf.fci import direct_nosym

    def kernel(fr, cr):
        return np.asarray(direct_nosym.contract_1e(fr, cr, space.norb, space.nelec,
                                                   link_index=space._link_nosym))

    return _split_apply(kernel, np.asarray(f), np.asarray(ci).reshape(space.shape))


def two_body_ee(space: FCISpace, m: np.ndarray, ci: np.ndarray) -> np.ndarray:
    """sum_pqrs m[p,q,r,s] E_pq E_rs |ci> (no normal ordering)."""
    from pyscf.fci import direct_nosym

    # direct_nosym.contract_2e(x) = sum x[p,q,r,s] E_rs E_pq, so pass x = m^(rs,pq).
    mt = np.ascontiguousarray(np.asarray(m).transpose(2, 3, 0, 1))

    def kernel(mr, cr):
        return np.asarray(direct_nosym.contract_2e(mr, cr, space.norb, space.nelec,
                                                   link_index=space._link_nosym))

    return _split_apply(kernel, mt, np.asarray(ci).reshape(space.shape))


def normal_order_correction(c: np.ndarray) -> np.ndarray:
    """f[p,s] = sum_q c[p,q,q,s], so that sum c e_{pq,rs} = sum c E_pq E_rs - sum f E_ps."""
    return np.einsum("pqqs->ps", c)


def two_body(space: FCISpace, c: np.ndarray, ci: np.ndarray) -> np.ndarray:
    """sum_pqrs c[p,q,r,s] e_{pq,rs} |ci> with e_{pq,rs} = E_pq E_rs - delta_qr E_ps."""
    return two_body_ee(space, c, ci) - one_body(space, normal_order_correction(c), ci)


class Hamiltonian:
    """Electronic Hamiltonian of an :class:`ActiveSpace` acting on CI vectors."""

    def __init__(self, active: ActiveSpace):
        from pyscf.fci import cistring, direct_spin1

        self.active = active
        self.space = FCISpace.of(active)
        self.ecore = active.ecore
        n, ne = active.norb, active.nelec
        self._h2e = direct_spin1.absorb_h1e(active.h1, active.eri, n, ne, 0.5)
        self._link = (cistring.gen_linkstr_index_trilidx(range(n), ne[0]),
                      cistring.gen_linkstr_index_trilidx(range(n), ne[1]))

    def apply(self, ci: np.ndarray) -> np.ndarray:
        """H_el |ci> (electronic part, without ecore)."""
        from pyscf.fci import direct_spin1

        space = self.space

        def kernel(h, cr):
            return np.asarray(direct_spin1.contract_2e(h, cr, space.norb, space.nelec,
                                                       link_index=self._link))

        return _split_apply(kernel, self._h2e, np.asarray(ci).reshape(space.shape))

    def energy(self, ci: np.ndarray) -> float:
        """<ci|H|ci>/<ci|ci> including ecore."""
        hci = self.apply(ci)
        return float((inner(ci, hci) / inner(ci, ci)).real) + self.ecore

    def energy_and_variance(self, ci: np.ndarray) -> tuple[float, float]:
        """(E including ecore, <H^2> - <H>^2) for a normalized-or-not ci."""
        nrm = inner(ci, ci).real
        hci = self.apply(ci)
        e_el = float(inner(ci, hci).real / nrm)
        h2 = float(inner(hci, hci).real / nrm)
        return e_el + self.ecore, max(h2 - e_el * e_el, 0.0)


def _estimate_norm(matvec: Callable[[np.ndarray], np.ndarray], v: np.ndarray,
                   iters: int = 8, seed: int = 7) -> float:
    """Crude spectral-norm estimate by power iteration on op^dagger op ~ -op^2."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(v.shape) + (1j * rng.standard_normal(v.shape) if np.iscomplexobj(v) else 0)
    x = x / np.linalg.norm(x)
    est = 0.0
    for _ in range(iters):
        y = matvec(x)
        est = np.linalg.norm(y)
        if est == 0:
            return 0.0
        x = y / est
    return float(est)


def expm_apply(matvec: Callable[[np.ndarray], np.ndarray], v: np.ndarray, t: float,
               norm_est: float | None = None, tol: float = 1e-13,
               max_terms: int = 60) -> np.ndarray:
    """exp(t * Op) v by Taylor series with substepping.

    ``Op`` is given by ``matvec``. Each substep h satisfies |h| * ||Op|| <= 1/2,
    and the series is summed until the next term is below ``tol * ||v||``.
    """
    v = np.asarray(v)
    if t == 0:
        return v.copy()
    if norm_est is None:
        norm_est = _estimate_norm(matvec, v)
    nsteps = max(1, int(np.ceil(abs(t) * norm_est * 2.0 * 1.1)))
    h = t / nsteps
    out = v.astype(np.result_type(v, float), copy=True)
    vnorm = np.linalg.norm(v)
    for _ in range(nsteps):
        term = out
        acc = out.copy()
        for k in range(1, max_terms + 1):
            term = (h / k) * matvec(term)
            acc = acc + term
            if np.linalg.norm(term) <= tol * vnorm:
                break
        else:
            raise RuntimeError("expm_apply: Taylor series did not converge; "
                               "norm estimate too small?")
        out = acc
    return out


class KrylovPropagator:
    """exp(eps * Op) psi for many eps from one Krylov (Arnoldi) space.

    Builds V (orthonormal, m columns) and the projection Hm = V^dag Op V, then
    exp(eps Op) psi ~ ||psi|| V expm(eps Hm) e_1. Accuracy is monitored with the
    standard a-posteriori estimate eps * h_{m+1,m} |e_m^T expm(eps Hm) e_1|,
    and the space is enlarged until it is below ``tol`` at ``eps_max``.

    The basis is promoted to complex as soon as ``matvec`` returns a complex
    vector, so a complex operator acting on a real ``psi`` is propagated exactly
    (the state becomes complex).

    ``check_eps`` replaces the single accuracy check at ``eps_max`` by checks at
    every listed eps (e.g. ``(eps_max, -eps_max)`` for steps of either sign).
    """

    def __init__(self, matvec: Callable[[np.ndarray], np.ndarray], psi: np.ndarray,
                 eps_max: float, tol: float = 1e-11, m_init: int = 12, m_max: int = 120,
                 check_eps: Sequence[float] | None = None):
        self.shape = np.asarray(psi).shape
        self.beta = float(np.linalg.norm(psi))
        dtype = np.result_type(psi, float)
        vs = [np.asarray(psi, dtype=dtype).ravel() / self.beta]
        hcols: list[np.ndarray] = []
        self._matvec = matvec
        self.eps_max = float(eps_max)
        self.tol = tol
        h_next = 0.0
        m = 0
        target = m_init
        while True:
            while m < target:
                w = np.asarray(matvec(vs[m].reshape(self.shape))).ravel()
                if np.iscomplexobj(w) and not np.iscomplexobj(vs[0]):
                    dtype = np.result_type(dtype, w)
                    vs = [v.astype(dtype) for v in vs]
                w = w.astype(np.result_type(dtype, w), copy=False)
                col = np.zeros(m + 2, dtype=np.result_type(w, float))
                for _ in range(2):  # re-orthogonalize twice for stability
                    for j in range(m + 1):
                        cj = np.vdot(vs[j], w)
                        col[j] += cj
                        w = w - cj * vs[j]
                h_next = float(np.linalg.norm(w))
                col[m + 1] = h_next
                hcols.append(col)
                m += 1
                if h_next < 1e-14:  # invariant subspace: exact
                    break
                vs.append(w / h_next)
            hm = np.zeros((m, m), dtype=np.result_type(*hcols))
            for j, col in enumerate(hcols):
                k = min(j + 2, m)
                hm[:k, j] = col[:k]
            self._V = np.array(vs[:m]).T
            self._Hm = hm
            self._h_next = h_next
            if check_eps is None:
                err = self.error_estimate(self.eps_max)
            else:
                err = max(self.error_estimate(float(e)) for e in check_eps)
            if err <= tol or h_next < 1e-14 or m >= m_max:
                self.converged = err <= tol or h_next < 1e-14
                break
            target = min(m_max, 2 * m)
        self.m = m

    def _coeffs(self, eps: float) -> np.ndarray:
        e1 = np.zeros(self._Hm.shape[0], dtype=self._Hm.dtype)
        e1[0] = 1.0
        return scipy.linalg.expm(eps * self._Hm) @ e1

    def error_estimate(self, eps: float) -> float:
        y = self._coeffs(eps)
        return float(abs(eps) * self._h_next * abs(y[-1]) * self.beta)

    def basis(self) -> np.ndarray:
        """Orthonormal Krylov basis as a (dim, m) matrix."""
        return self._V

    def coefficients(self, eps: float) -> np.ndarray:
        """y(eps) such that exp(eps Op) psi ~ basis() @ y(eps)."""
        return self.beta * self._coeffs(eps)

    def state(self, eps: float) -> np.ndarray:
        return (self._V @ self.coefficients(eps)).reshape(self.shape)
