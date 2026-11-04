"""Small benchmark systems used in tests and benchmarks.

Every builder runs a restricted Hartree-Fock calculation with PySCF and
returns an :class:`~thc_engine_cqe.integrals.ActiveSpace` in the canonical
RHF orbitals.
"""

from __future__ import annotations

import numpy as np

from .integrals import ActiveSpace

__all__ = ["hydrogen_chain", "hydrogen_rectangle", "lih", "n2", "molecule"]


def _rhf(mol, conv_tol: float = 1e-12, conv_tol_grad: float = 1e-9):
    """Tightly converged RHF (the orbital gradient sets how well Brillouin holds)."""
    from pyscf import scf

    mf = scf.RHF(mol)
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = conv_tol_grad
    mf.max_cycle = 200
    mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        # Second-order solver as a fallback for stretched geometries.
        mf = mf.newton()
        mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"RHF did not converge for {mol.atom}")
    return mf


def molecule(atom, basis: str = "sto-3g", unit: str = "angstrom", charge: int = 0,
             spin: int = 0, ncas: int | None = None, nelecas=None, label: str = "",
             symmetry: bool = False) -> ActiveSpace:
    """Generic builder: geometry -> RHF -> active space."""
    from pyscf import gto

    mol = gto.M(atom=atom, basis=basis, unit=unit, charge=charge, spin=spin,
                symmetry=symmetry, verbose=0)
    mf = _rhf(mol)
    return ActiveSpace.from_pyscf(mf, ncas=ncas, nelecas=nelecas, label=label)


def hydrogen_chain(n: int, spacing: float = 1.0, unit: str = "angstrom",
                   basis: str = "sto-3g") -> ActiveSpace:
    """Linear, equally spaced H_n (all orbitals and electrons active)."""
    atom = [("H", (0.0, 0.0, i * spacing)) for i in range(n)]
    return molecule(atom, basis=basis, unit=unit, spin=n % 2,
                    label=f"H{n} chain, {spacing:g} {unit}, {basis}")


def hydrogen_rectangle(a: float, b: float, unit: str = "angstrom",
                       basis: str = "sto-3g") -> ActiveSpace:
    """Rectangular H4 with sides a x b (a == b gives the square)."""
    atom = [("H", (0.0, 0.0, 0.0)), ("H", (a, 0.0, 0.0)),
            ("H", (a, b, 0.0)), ("H", (0.0, b, 0.0))]
    return molecule(atom, basis=basis, unit=unit,
                    label=f"H4 rectangle {a:g}x{b:g} {unit}, {basis}")


def lih(r: float = 1.595, basis: str = "sto-3g", ncas: int | None = None,
        nelecas=None) -> ActiveSpace:
    """LiH at bond length r (angstrom)."""
    return molecule([("Li", (0, 0, 0)), ("H", (0, 0, r))], basis=basis,
                    ncas=ncas, nelecas=nelecas, label=f"LiH r={r:g} A, {basis}")


def n2(r: float = 1.098, basis: str = "sto-3g", ncas: int = 6, nelecas=6) -> ActiveSpace:
    """N2 at bond length r (angstrom) with a CAS(nelecas, ncas) active space."""
    return molecule([("N", (0, 0, 0)), ("N", (0, 0, r))], basis=basis,
                    ncas=ncas, nelecas=nelecas,
                    label=f"N2 r={r:g} A, {basis}, CAS({nelecas},{ncas})")


def random_orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-ish random real orthogonal matrix (for tests)."""
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))
