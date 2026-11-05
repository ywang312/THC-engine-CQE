"""Small benchmark systems used in tests and benchmarks.

Every builder runs a restricted Hartree-Fock calculation with PySCF and
returns an :class:`~thc_engine_cqe.integrals.ActiveSpace` in the canonical
RHF orbitals.
"""

from __future__ import annotations

import numpy as np

from .integrals import ActiveSpace

__all__ = ["hydrogen_chain", "hydrogen_rectangle", "lih", "n2", "molecule", "canonical_orbitals"]


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


def canonical_orbitals(mf, degeneracy_tol: float = 1e-6) -> np.ndarray:
    """MO coefficients of a symmetry-adapted RHF with a reproducible order and sign.

    A symmetry-adapted RHF fixes the orientation of degenerate orbitals (each member of
    a pi pair belongs to its own irrep of the abelian computational group), but not their
    order, which follows round-off in their equal energies, nor the signs of the
    eigenvectors. Within every run of orbital energies closer than ``degeneracy_tol``
    (Hartree) the orbitals are sorted by the index of their largest AO coefficient, so
    pi_x comes before pi_y (PySCF orders p functions x, y, z), then by irrep id. Every
    orbital's sign is then chosen so that its largest AO coefficient (the first one
    within 1e-8 of the maximum) is positive.

    Orbital order and orientation within a degenerate set do not change energies or any
    other physical quantity, but they do change basis-dependent ones: QWC groupings and
    Pauli-shadow variances, pivoted-Cholesky leaves, THC fits and element-wise sparsity.
    Signs change only the signs of integrals and Pauli coefficients.
    """
    from pyscf import symm

    mo = np.array(mf.mo_coeff, dtype=float, copy=True)
    energies = np.asarray(mf.mo_energy, dtype=float)
    mol = mf.mol
    irreps = np.asarray(symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo, check=False))
    dominant = np.argmax(np.abs(mo), axis=0)
    order: list[int] = []
    start = 0
    while start < len(energies):
        stop = start + 1
        while stop < len(energies) and abs(energies[stop] - energies[start]) < degeneracy_tol:
            stop += 1
        order.extend(sorted(range(start, stop), key=lambda k: (int(dominant[k]), int(irreps[k]), k)))
        start = stop
    mo = mo[:, order]
    for k in range(mo.shape[1]):
        a = np.abs(mo[:, k])
        first = int(np.flatnonzero(a >= a.max() - 1e-8)[0])
        if mo[first, k] < 0:
            mo[:, k] = -mo[:, k]
    return mo


def molecule(atom, basis: str = "sto-3g", unit: str = "angstrom", charge: int = 0,
             spin: int = 0, ncas: int | None = None, nelecas=None, label: str = "",
             symmetry: bool | str = False) -> ActiveSpace:
    """Generic builder: geometry -> RHF -> active space.

    ``symmetry`` is passed to PySCF (False, True or a point group such as "D2h"). With
    symmetry the active space is built from :func:`canonical_orbitals`, so degenerate
    orbitals have a reproducible orientation, order and sign; without it (the default)
    the canonical RHF orbitals are used as returned, and degenerate orbitals come in
    whatever orientation round-off selects.
    """
    from pyscf import gto

    mol = gto.M(atom=atom, basis=basis, unit=unit, charge=charge, spin=spin,
                symmetry=symmetry, verbose=0)
    mf = _rhf(mol)
    mo = canonical_orbitals(mf) if mol.symmetry else None
    return ActiveSpace.from_pyscf(mf, ncas=ncas, nelecas=nelecas, mo_coeff=mo, label=label)


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
    """N2 at bond length r (angstrom) with a CAS(nelecas, ncas) active space.

    The RHF is symmetry-adapted (D2h) and the orbitals are made reproducible by
    :func:`canonical_orbitals`: the degenerate pi_u and pi_g pairs come out axis-aligned,
    pi_x before pi_y, with fixed signs. Without symmetry their orientation followed
    round-off and changed with the BLAS thread count, and so did basis-dependent
    quantities such as QWC and Pauli-shadow variances (docs/m2-results.md). Energies are
    unchanged (E_FCI agrees with the unsymmetrized build to 1e-13 at 1.098 and 2.0 A).
    """
    return molecule([("N", (0, 0, 0)), ("N", (0, 0, r))], basis=basis,
                    ncas=ncas, nelecas=nelecas, symmetry="D2h",
                    label=f"N2 r={r:g} A, {basis}, CAS({nelecas},{ncas})")


def random_orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-ish random real orthogonal matrix (for tests)."""
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))
