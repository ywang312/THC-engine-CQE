"""Active-space Hamiltonians.

All quantities refer to a set of N real, orthonormal spatial orbitals.

    h1[p, q]        = <p|h|q>, including the mean field of frozen core orbitals
    eri[p, q, r, s] = (pq|rs) in chemist notation (8-fold symmetric for real orbitals)
    ecore           = nuclear repulsion + frozen-core energy

    H = ecore + sum_pq h1[p,q] E_pq + 1/2 sum_pqrs eri[p,q,r,s] e_{pq,rs}

with E_pq = sum_sigma a+_{p sigma} a_{q sigma} and the normal-ordered spin-free
two-body operator e_{pq,rs} = E_pq E_rs - delta_qr E_ps
= sum_{sigma tau} a+_{p sigma} a+_{r tau} a_{s tau} a_{q sigma}.
See docs/conventions.md.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

__all__ = ["ActiveSpace"]


@dataclass(frozen=True, eq=False)
class ActiveSpace:
    """A spin-free active-space Hamiltonian with a fixed electron count.

    Attributes:
        h1: (N, N) one-electron integrals (real symmetric).
        eri: (N, N, N, N) two-electron integrals (pq|rs).
        ecore: constant energy (nuclear repulsion plus frozen core).
        nelec: (n_alpha, n_beta) active electrons.
        mo_coeff: optional (n_ao, N) coefficients of the active orbitals.
        mol: optional ``pyscf.gto.Mole`` the orbitals live on (needed for
            grid-based THC guesses and orbital localization).
        label: free-form description used in reports.
    """

    h1: np.ndarray
    eri: np.ndarray
    ecore: float
    nelec: tuple[int, int]
    mo_coeff: np.ndarray | None = None
    mol: object | None = None
    label: str = ""

    def __post_init__(self) -> None:
        h1 = np.asarray(self.h1)
        eri = np.asarray(self.eri)
        n = h1.shape[0]
        if h1.shape != (n, n):
            raise ValueError(f"h1 must be square, got {h1.shape}")
        if eri.shape != (n, n, n, n):
            raise ValueError(f"eri must have shape {(n,) * 4}, got {eri.shape}")
        if np.iscomplexobj(h1) or np.iscomplexobj(eri):
            raise ValueError("only real orbitals/integrals are supported")
        na, nb = (int(x) for x in self.nelec)
        if not (0 <= na <= n and 0 <= nb <= n):
            raise ValueError(f"nelec={self.nelec} does not fit in {n} orbitals")
        object.__setattr__(self, "h1", np.ascontiguousarray(h1, dtype=float))
        object.__setattr__(self, "eri", np.ascontiguousarray(eri, dtype=float))
        object.__setattr__(self, "ecore", float(self.ecore))
        object.__setattr__(self, "nelec", (na, nb))

    # ------------------------------------------------------------------ basics
    @property
    def norb(self) -> int:
        return self.h1.shape[0]

    @property
    def n_electrons(self) -> int:
        return self.nelec[0] + self.nelec[1]

    def check_symmetry(self, atol: float = 1e-10) -> None:
        """Raise if h1/eri violate the symmetries of real orbitals."""
        h1, eri = self.h1, self.eri
        errs = {
            "h1 - h1.T": np.abs(h1 - h1.T).max(),
            "(pq|rs)-(qp|rs)": np.abs(eri - eri.transpose(1, 0, 2, 3)).max(),
            "(pq|rs)-(pq|sr)": np.abs(eri - eri.transpose(0, 1, 3, 2)).max(),
            "(pq|rs)-(rs|pq)": np.abs(eri - eri.transpose(2, 3, 0, 1)).max(),
        }
        bad = {k: v for k, v in errs.items() if v > atol}
        if bad:
            raise ValueError(f"integral symmetry violated: {bad}")

    def with_eri(self, eri: np.ndarray, label: str | None = None) -> "ActiveSpace":
        """Same system with replaced two-electron integrals (e.g. a THC model)."""
        return replace(self, eri=np.asarray(eri), label=self.label if label is None else label)

    def rotated(self, u: np.ndarray, label: str | None = None) -> "ActiveSpace":
        """Express the Hamiltonian in rotated orbitals phi'_a = sum_p phi_p u[p, a].

        ``u`` must be real orthogonal. The new integrals are
        h1' = u^T h1 u and (ab|cd)' = sum u[p,a] u[q,b] u[r,c] u[s,d] (pq|rs).
        """
        u = np.asarray(u, dtype=float)
        n = self.norb
        if u.shape != (n, n) or not np.allclose(u.T @ u, np.eye(n), atol=1e-10):
            raise ValueError("u must be a real orthogonal (N, N) matrix")
        h1 = u.T @ self.h1 @ u
        eri = np.einsum("pqrs,pa,qb,rc,sd->abcd", self.eri, u, u, u, u, optimize=True)
        mo = None if self.mo_coeff is None else self.mo_coeff @ u
        return replace(self, h1=h1, eri=eri, mo_coeff=mo,
                       label=self.label if label is None else label)

    # ------------------------------------------------------------- reference
    def fci(self, nroots: int = 1, spin_square: float | None = 0.0, **kwargs):
        """Exact (FCI) solution in the active space.

        Returns ``(e_total, ci)`` where ``e_total`` includes ``ecore`` and ``ci``
        is a PySCF CI array of shape (n_alpha_strings, n_beta_strings).
        ``spin_square`` fixes <S^2> (default: singlet when n_alpha == n_beta;
        pass ``None`` to disable).
        """
        from pyscf import fci

        solver = fci.direct_spin1.FCI()
        solver.conv_tol = kwargs.pop("conv_tol", 1e-12)
        solver.max_cycle = kwargs.pop("max_cycle", 500)
        solver.nroots = nroots
        if spin_square is not None and self.nelec[0] == self.nelec[1]:
            solver = fci.addons.fix_spin_(solver, ss=spin_square)
        e, ci = solver.kernel(self.h1, self.eri, self.norb, self.nelec,
                              ecore=self.ecore, **kwargs)
        return e, ci

    # ------------------------------------------------------------ constructors
    @classmethod
    def from_pyscf(cls, mf, ncas: int | None = None, nelecas=None,
                   mo_coeff: np.ndarray | None = None, label: str = "") -> "ActiveSpace":
        """Build a CASCI-style active space from a converged (restricted) SCF.

        Args:
            mf: converged ``pyscf.scf`` RHF/ROHF object with real orbitals.
            ncas: number of active orbitals (default: all MOs).
            nelecas: active electrons, int or (n_alpha, n_beta) (default: all).
            mo_coeff: orbitals to use instead of ``mf.mo_coeff`` (e.g. CASSCF or
                natural orbitals). Core orbitals are taken from the first columns.
        """
        from pyscf import ao2mo, mcscf

        mol = mf.mol
        nmo = (mf.mo_coeff if mo_coeff is None else mo_coeff).shape[1]
        if ncas is None:
            ncas = nmo
        if nelecas is None:
            nelecas = mol.nelec
        mc = mcscf.CASCI(mf, ncas, nelecas)
        if mo_coeff is not None:
            mc.mo_coeff = mo_coeff
        h1, ecore = mc.get_h1eff()
        eri = ao2mo.restore(1, mc.get_h2eff(), ncas)
        ncore = mc.ncore
        mo_cas = mc.mo_coeff[:, ncore:ncore + ncas]
        na, nb = mc.nelecas
        return cls(h1=np.asarray(h1), eri=np.asarray(eri), ecore=float(ecore),
                   nelec=(int(na), int(nb)), mo_coeff=np.asarray(mo_cas), mol=mol,
                   label=label)

    @classmethod
    def from_fcidump(cls, path: str, label: str = "") -> "ActiveSpace":
        """Read an FCIDUMP file (spatial orbitals, real integrals)."""
        from pyscf import ao2mo
        from pyscf.tools import fcidump

        d = fcidump.read(path)
        n = int(d["NORB"])
        ne = int(d["NELEC"])
        ms2 = int(d.get("MS2", 0))
        na, nb = (ne + ms2) // 2, (ne - ms2) // 2
        eri = ao2mo.restore(1, np.asarray(d["H2"]), n)
        return cls(h1=np.asarray(d["H1"]), eri=eri, ecore=float(d["ECORE"]),
                   nelec=(na, nb), label=label or str(path))

    def to_fcidump(self, path: str, tol: float = 1e-15) -> None:
        """Write the active space as an FCIDUMP file."""
        from pyscf.tools import fcidump

        fcidump.from_integrals(path, self.h1, self.eri, self.norb, self.nelec,
                               nuc=self.ecore, ms=self.nelec[0] - self.nelec[1], tol=tol)
