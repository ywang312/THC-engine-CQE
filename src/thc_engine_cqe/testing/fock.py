"""Independent Jordan-Wigner Fock-space reference (tests only; tiny systems).

Nothing here uses PySCF's CI machinery, so it serves as an oracle for the
conventions used in :mod:`thc_engine_cqe.fci_ops` and
:mod:`thc_engine_cqe.residual`.

Mode ordering: spatial orbital p with spin alpha is mode p, with spin beta it is
mode N + p. Basis index i has bit k set iff mode k is occupied, and denotes
(a+_0)^{n_0} (a+_1)^{n_1} ... |vac>, so that
a+_j |n> = (-1)^{sum_{k<j} n_k} |n + e_j>.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

__all__ = ["FockSpace"]


def _popcount(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.int64)
    c = np.zeros_like(x)
    while np.any(x):
        c += x & 1
        x >>= 1
    return c


class FockSpace:
    """Fermionic Fock space of ``2 * norb`` spin orbitals (dimension 4**norb)."""

    def __init__(self, norb: int):
        if norb > 7:
            raise ValueError("FockSpace is a test oracle; keep norb <= 7")
        self.norb = norb
        self.nmodes = 2 * norb
        self.dim = 1 << self.nmodes
        idx = np.arange(self.dim, dtype=np.int64)
        self._ann = []
        for j in range(self.nmodes):
            occ = (idx >> j) & 1
            src = idx[occ == 1]
            dst = src ^ (1 << j)
            sign = (-1.0) ** _popcount(src & ((1 << j) - 1))
            self._ann.append(sp.csr_matrix((sign, (dst, src)), shape=(self.dim, self.dim)))
        self._E = {}

    # ------------------------------------------------------------- operators
    def mode(self, p: int, spin: int) -> int:
        return p + spin * self.norb

    def a(self, p: int, spin: int) -> sp.csr_matrix:
        return self._ann[self.mode(p, spin)]

    def adag(self, p: int, spin: int) -> sp.csr_matrix:
        return self._ann[self.mode(p, spin)].T.tocsr()

    def E(self, p: int, q: int) -> sp.csr_matrix:
        """Spin-summed excitation operator E_pq = sum_s a+_{ps} a_{qs}."""
        key = (p, q)
        if key not in self._E:
            self._E[key] = (self.adag(p, 0) @ self.a(q, 0) + self.adag(p, 1) @ self.a(q, 1)).tocsr()
        return self._E[key]

    def e2(self, p: int, q: int, r: int, s: int) -> sp.csr_matrix:
        """Normal-ordered spin-free e_{pq,rs} = sum_{st} a+_{p s} a+_{r t} a_{s t} a_{q s}."""
        out = sp.csr_matrix((self.dim, self.dim))
        for s1 in (0, 1):
            for s2 in (0, 1):
                out = out + (self.adag(p, s1) @ self.adag(r, s2) @ self.a(s, s2) @ self.a(q, s1))
        return out.tocsr()

    def number(self) -> sp.csr_matrix:
        return sum(self.E(p, p) for p in range(self.norb)).tocsr()

    def s_z(self) -> sp.csr_matrix:
        n_a = sum(self.adag(p, 0) @ self.a(p, 0) for p in range(self.norb))
        n_b = sum(self.adag(p, 1) @ self.a(p, 1) for p in range(self.norb))
        return (0.5 * (n_a - n_b)).tocsr()

    def one_body(self, f: np.ndarray) -> sp.csr_matrix:
        n = self.norb
        out = sp.csr_matrix((self.dim, self.dim), dtype=np.result_type(f, float))
        for p in range(n):
            for q in range(n):
                if f[p, q] != 0:
                    out = out + f[p, q] * self.E(p, q)
        return out.tocsr()

    def two_body(self, c: np.ndarray) -> sp.csr_matrix:
        """sum_pqrs c[p,q,r,s] e_{pq,rs} (built as E_pq E_rs - delta_qr E_ps)."""
        n = self.norb
        out = sp.csr_matrix((self.dim, self.dim), dtype=np.result_type(c, float))
        for p in range(n):
            for q in range(n):
                for r in range(n):
                    for s in range(n):
                        v = c[p, q, r, s]
                        if v == 0:
                            continue
                        term = self.E(p, q) @ self.E(r, s)
                        if q == r:
                            term = term - self.E(p, s)
                        out = out + v * term
        return out.tocsr()

    def hamiltonian(self, h1: np.ndarray, eri: np.ndarray, ecore: float = 0.0) -> sp.csr_matrix:
        h = self.one_body(h1) + 0.5 * self.two_body(eri)
        return (h + ecore * sp.identity(self.dim)).tocsr()

    # ----------------------------------------------------------- CI mapping
    def _addresses(self, nelec):
        from pyscf.fci import cistring

        na, nb = nelec
        sa = np.asarray(cistring.make_strings(range(self.norb), na), dtype=np.int64)
        sb = np.asarray(cistring.make_strings(range(self.norb), nb), dtype=np.int64)
        return (sa[:, None] | (sb[None, :] << self.norb)).ravel()

    def from_ci(self, ci: np.ndarray, nelec) -> np.ndarray:
        """Embed a PySCF CI array (n_alpha_strings, n_beta_strings) in Fock space.

        The two conventions agree up to a global sign that depends only on
        (n_alpha, n_beta), which is irrelevant for every quantity compared.
        """
        addr = self._addresses(nelec)
        v = np.zeros(self.dim, dtype=np.result_type(ci, float))
        v[addr] = np.asarray(ci).ravel()
        return v

    def to_ci(self, v: np.ndarray, nelec) -> np.ndarray:
        from pyscf.fci import cistring

        addr = self._addresses(nelec)
        na = cistring.num_strings(self.norb, nelec[0])
        nb = cistring.num_strings(self.norb, nelec[1])
        return np.asarray(v)[addr].reshape(na, nb)

    def leakage(self, v: np.ndarray, nelec) -> float:
        """Norm^2 of v outside the (n_alpha, n_beta) sector."""
        addr = self._addresses(nelec)
        inside = np.vdot(v[addr], v[addr]).real
        return float(np.vdot(v, v).real - inside)
