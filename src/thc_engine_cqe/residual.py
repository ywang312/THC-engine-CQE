"""Reduced density matrices, the CSE matrix and the ACSE residual.

All two-index-pair tensors here use PySCF's *pair order* (see docs/conventions.md):

    X[p, q, r, s]  <->  the spin-free operator e_{pq,rs}
                        = sum_{sigma tau} a+_{p sigma} a+_{r tau} a_{s tau} a_{q sigma}

so that, for a state Psi,

    dm2[p,q,r,s] = <Psi| e_{pq,rs} |Psi>                (spin-summed 2-RDM)
    T[p,q,r,s]   = <Psi| e_{pq,rs} H |Psi>              (CSE matrix)
    A[p,q,r,s]   = <Psi| [e_{pq,rs}, H] |Psi>
                 = T[p,q,r,s] - conj(T[q,p,s,r])        (ACSE residual)

Mazziotti's spin-orbital notation 2D^{ij}_{kl} = <a+_i a+_j a_l a_k> corresponds
to the spin sum of pair order with (i, k) = (p, q) and (j, l) = (r, s); use
:func:`pair_to_mazziotti` / :func:`mazziotti_to_pair` to convert.

The CQE update is Psi <- exp(eps * G) Psi with the generator
G = sum_pqrs g[p,q,r,s] e_{pq,rs} and g = conj(A) (:func:`descent_generator`),
which gives dE/deps = -||A||^2 at eps = 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fci_ops import FCISpace, Hamiltonian, inner
from .integrals import ActiveSpace

__all__ = [
    "rdm12",
    "trans_rdm12",
    "cse_matrix",
    "acse_residual",
    "descent_generator",
    "reduced_hamiltonian",
    "energy_from_rdms",
    "variance_from_cse",
    "ResidualData",
    "compute_residual",
    "pair_to_mazziotti",
    "mazziotti_to_pair",
    "spin_free_norm",
]


def _real_trans_rdm12(bra: np.ndarray, ket: np.ndarray, space: FCISpace):
    from pyscf.fci import direct_spin1

    dm1, dm2 = direct_spin1.trans_rdm12(np.ascontiguousarray(bra, dtype=float),
                                        np.ascontiguousarray(ket, dtype=float),
                                        space.norb, space.nelec)
    return np.asarray(dm1), np.asarray(dm2)


def trans_rdm12(space: FCISpace, bra: np.ndarray, ket: np.ndarray):
    """Spin-summed transition RDMs between CI vectors (complex allowed).

    Returns (dm1, dm2) with dm1[p,q] = <bra|E_qp|ket> (PySCF's transposed
    convention) and dm2[p,q,r,s] = <bra|e_{pq,rs}|ket>.
    """
    bra = np.asarray(bra).reshape(space.shape)
    ket = np.asarray(ket).reshape(space.shape)
    if not np.iscomplexobj(bra) and not np.iscomplexobj(ket):
        return _real_trans_rdm12(bra, ket, space)
    br, bi = bra.real, bra.imag
    kr, ki = ket.real, ket.imag
    # <bra| = <br| - i <bi| (conjugation), operators are real.
    d1rr, d2rr = _real_trans_rdm12(br, kr, space)
    d1ii, d2ii = _real_trans_rdm12(bi, ki, space)
    d1ri, d2ri = _real_trans_rdm12(br, ki, space)
    d1ir, d2ir = _real_trans_rdm12(bi, kr, space)
    dm1 = (d1rr + d1ii) + 1j * (d1ri - d1ir)
    dm2 = (d2rr + d2ii) + 1j * (d2ri - d2ir)
    return dm1, dm2


def rdm12(space: FCISpace, ci: np.ndarray):
    """Spin-summed (dm1, dm2) of a normalized CI vector, PySCF conventions.

    dm1[p,q] = <E_qp>, dm2[p,q,r,s] = <e_{pq,rs}>. For complex states the
    imaginary parts are kept (dm1 is then Hermitian rather than symmetric).
    """
    nrm = inner(ci, ci).real
    dm1, dm2 = trans_rdm12(space, ci, ci)
    return dm1 / nrm, dm2 / nrm


def cse_matrix(space: FCISpace, ci: np.ndarray, hci: np.ndarray) -> np.ndarray:
    """T[p,q,r,s] = <Psi| e_{pq,rs} H |Psi> for normalized Psi, given hci = H|Psi>."""
    nrm = inner(ci, ci).real
    return trans_rdm12(space, ci, hci)[1] / nrm


def acse_residual(T: np.ndarray) -> np.ndarray:
    """A[p,q,r,s] = <[e_{pq,rs}, H]> = T[p,q,r,s] - conj(T[q,p,s,r])."""
    return T - np.conj(T.transpose(1, 0, 3, 2))


def descent_generator(A: np.ndarray) -> np.ndarray:
    """Coefficients g of G = sum g e_{pq,rs} with dE/deps = -||A||^2 for exp(eps G)."""
    return np.conj(A)


def reduced_hamiltonian(h1: np.ndarray, eri: np.ndarray, n_electrons: int) -> np.ndarray:
    """Spin-free reduced Hamiltonian K (pair order) with E_el = sum K * dm2.

    K[p,q,r,s] = 1/2 (pq|rs) + (h[p,q] d_rs + d_pq h[r,s]) / (2 (N - 1)),
    valid on the N-electron space (uses E_pq = sum_r e_{pq,rr} / (N - 1)).
    """
    if n_electrons < 2:
        raise ValueError("the reduced Hamiltonian needs at least two electrons")
    n = h1.shape[0]
    eye = np.eye(n)
    k = 0.5 * np.asarray(eri, dtype=float).copy()
    fac = 1.0 / (2.0 * (n_electrons - 1))
    k += fac * (np.einsum("pq,rs->pqrs", h1, eye) + np.einsum("pq,rs->pqrs", eye, h1))
    return k


def energy_from_rdms(h1: np.ndarray, eri: np.ndarray, dm1: np.ndarray, dm2: np.ndarray,
                     ecore: float = 0.0) -> float:
    """E = ecore + sum_pq h[p,q] <E_pq> + 1/2 sum (pq|rs) dm2[p,q,r,s] (PySCF dm1 order)."""
    e = np.einsum("pq,qp->", h1, dm1) + 0.5 * np.einsum("pqrs,pqrs->", eri, dm2)
    return float(np.real(e)) + ecore


def variance_from_cse(K: np.ndarray, T: np.ndarray, dm2: np.ndarray, e_el: float) -> float:
    """sigma^2 = sum K (T - E_el dm2) with electronic energy E_el (no ecore)."""
    return float(np.real(np.einsum("pqrs,pqrs->", K, T - e_el * dm2)))


def spin_free_norm(x: np.ndarray) -> float:
    """Frobenius norm of a pair-order tensor."""
    return float(np.sqrt(np.sum(np.abs(x) ** 2)))


def pair_to_mazziotti(x: np.ndarray) -> np.ndarray:
    """Pair order X[p,q,r,s] -> Mazziotti order D[i,j,k,l] (creators i,j; annihilators l,k)."""
    return np.asarray(x).transpose(0, 2, 1, 3)


def mazziotti_to_pair(d: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pair_to_mazziotti` (the map is an involution)."""
    return np.asarray(d).transpose(0, 2, 1, 3)


@dataclass
class ResidualData:
    """Everything one CQE iteration needs about the current state."""

    energy: float          # total energy (with ecore)
    variance: float        # <H^2> - <H>^2 (direct)
    dm1: np.ndarray
    dm2: np.ndarray
    T: np.ndarray          # CSE matrix (pair order)
    A: np.ndarray          # ACSE residual (pair order)

    @property
    def residual_norm(self) -> float:
        return spin_free_norm(self.A)


def compute_residual(active: ActiveSpace, ci: np.ndarray, ham: Hamiltonian | None = None) -> ResidualData:
    """Exact energy, variance, RDMs, CSE matrix and ACSE residual of ``ci``."""
    ham = ham or Hamiltonian(active)
    space = ham.space
    ci = np.asarray(ci).reshape(space.shape)
    nrm = inner(ci, ci).real
    hci = ham.apply(ci)
    e_el = float(inner(ci, hci).real / nrm)
    var = max(float(inner(hci, hci).real / nrm) - e_el * e_el, 0.0)
    dm1, dm2 = rdm12(space, ci)
    T = cse_matrix(space, ci, hci)
    A = acse_residual(T)
    return ResidualData(energy=e_el + active.ecore, variance=var, dm1=dm1, dm2=dm2, T=T, A=A)
