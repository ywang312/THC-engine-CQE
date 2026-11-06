"""Benchmark systems on symmetry-adapted orbitals, orbital irreps and point-group masks (F2a).

Benchmark systems (:data:`BENCHMARK_SYSTEMS`, :func:`benchmark_system`)
---------------------------------------------------------------------
The seven M3 systems, STO-3G, as symmetry-adapted RHF builds of ``systems.molecule``
(``symmetry`` is passed through to PySCF, and the active space comes from
``systems.canonical_orbitals``: degenerate sets in a fixed order, pi_x before pi_y,
with fixed signs):

    name  system                     point group  active space
    h4    H4 chain, 1.8 bohr         D2h          all (4 orbitals, 4 electrons)
    h6a   H6 chain, 1.0 A            D2h          all (6, 6)
    h6s   H6 chain, 3.0 bohr         D2h          all (6, 6)
    h8    H8 chain, 1.8 bohr         D2h          all (8, 8)
    lih   LiH, 1.595 A               C2v          all (6, 4)
    n2    N2, 1.098 A                D2h          CAS(6,6)
    n2s   N2, 2.0 A                  D2h          CAS(6,6)

They describe the same molecules as the current builders (``systems.hydrogen_chain``,
``systems.lih``, ``systems.n2``) and give the same E_HF and E_FCI as plain
(unsymmetrized) RHF orbitals of the same geometry, to 1e-10 Ha (tested for all seven;
observed on one thread <= 7.1e-14 for n2 and n2s and <= 7.1e-15 for the others;
``systems.n2`` is itself a D2h build, so for N2 the plain RHF is the independent
reference). Only basis-dependent quantities differ, because orbitals of a
symmetry-adapted RHF each belong to one irrep: the degenerate pi pairs of N2 come out
axis-aligned (pi_x in B3u and B2g, pi_y in B2u and B3g; tested on the AO
coefficients). ``benchmark_system(name, symmetry=False)`` gives the plain RHF orbitals
of the same geometry, which for N2 have an arbitrary pi orientation.

Orbital irreps (:func:`orbital_irreps`)
---------------------------------------
PySCF's irrep ids of the active orbitals (``symm.label_orb_symm``, which rejects
orbitals that are not symmetry-pure to 1e-9). Only the abelian groups D2h and its
subgroups are accepted: their irreps are one-dimensional and real, and PySCF numbers
them so that the irrep of a direct product is the XOR of the ids
(``symm.param.IRREP_ID_TABLE``; D2h: Ag 0, B1g 1, B2g 2, B3g 3, Au 4, B1u 5, B2u 6,
B3u 7; C2v: A1 0, A2 1, B1 2, B2 3) [cited: PySCF documentation of
``label_orb_symm``; checked against the character tables in tests/test_ft_symmetry.py].
Linear molecules must therefore be built in D2h or C2v, not in Dooh or Coov.
Observed (canonical order): H chains Ag, B1u, Ag, B1u, ...; LiH A1, A1, A1, B1, B2, A1;
N2 at 1.098 A B3u, B2u, Ag, B2g, B3g, B1u; N2 at 2.0 A Ag, B3u, B2u, B2g, B3g, B1u.

Selection rule (F2a) [derived]
------------------------------
Every symmetry operation R of an abelian group maps an orbital to chi_p(R) phi_p with
chi_p(R) = +-1, so R a+_p R^dagger = chi_p(R) a+_p and R a_p R^dagger = chi_p(R) a_p for
either spin, and e_{pq,rs} (or a+_i a+_j a_l a_k) transforms as the product of its four
orbital irreps. H is totally symmetric. For a state Psi of one irrep Gamma,
<Psi|O|Psi> vanishes unless O transforms as Gamma x Gamma = totally symmetric. Hence

    element (p, q, r, s) allowed  <=>  irrep(p) XOR irrep(q) XOR irrep(r) XOR irrep(s) == 0,

and every forbidden residual element vanishes exactly at every symmetry-pure state
(:func:`pair_mask`, :func:`element_mask`; spin-orbital elements use their orbital
indices). The CQE generator built from the residual is then totally symmetric, so the
iterates keep the irrep of the HF determinant (totally symmetric for these closed
shells). Tested: along 5 exact steepest-descent steps of h4, h6a, lih and n2, forbidden
elements are <= 1e-10 max|A|, and the states have the HF irrep with weight > 1 - 1e-8
(:func:`ci_irrep`), as have the L-BFGS iterates up to k* + 2.

In floating point the forbidden entries carry round-off, and exact runs amplify it
(docs/m2-results.md, section 2). Observed along 20 exact L-BFGS steps (one thread; the
values follow the round-off history, which depends on the BLAS thread count): for N2
the forbidden entries grow by 10-150x per step from the third step on and reach max|A|
at step 11 (n2 and n2s); the weight outside the HF irrep reaches 1.8e-7 (n2, step 10)
and 1.9e-4 (n2s), against <= 6.4e-13 for the chains, LiH and h8. Projecting the
residual, ``run_cqe(..., residual_estimator=lambda rd, ci, it: mask_pair_tensor(rd.A,
irreps))``, keeps both at round-off on all seven systems (forbidden entries
<= 6.3e-15 Ha and 1 - weight <= 3.7e-15; tested on n2s). For the same reason
:func:`mask_pair_tensor` is applied to directions before compilation (F2a, F3).

Allowed elements, allowed / all (spin-free; spin-orbital), tested:

    h4   28 / 60     (70 / 150)        lih  105 / 315   (264 / 840)
    h6a  153 / 315   (408 / 840)       n2   33 / 315    (84 / 840)
    h8   496 / 1008  (1364 / 2772)

The counts depend only on the multiset of orbital irreps (an orbital permutation maps
element orbits to element orbits), so h6s and n2s have those of h6a and n2.

Determinant irreps (:func:`ci_irrep`)
-------------------------------------
A determinant transforms as the XOR of the irreps of its occupied spin orbitals, i.e.
irrep(alpha string) XOR irrep(beta string). :func:`ci_irrep` returns the irrep that
holds the largest weight sum |c|^2 of a CI vector and that weight (fraction of the
norm); tested against ``pyscf.fci.addons.symmetrize_wfn``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .. import systems
from ..integrals import ActiveSpace
from ..measure.targets import KINDS

__all__ = [
    "BENCHMARK_SYSTEMS",
    "benchmark_system",
    "orbital_irreps",
    "pair_mask",
    "element_mask",
    "mask_pair_tensor",
    "ci_irrep",
]

# Abelian point groups of PySCF whose irrep ids multiply by XOR (D2h and its subgroups).
_ABELIAN_GROUPS = ("D2h", "C2h", "C2v", "D2", "Cs", "Ci", "C2", "C1")


def _chain(n: int, spacing: float) -> tuple:
    return tuple(("H", (0.0, 0.0, i * spacing)) for i in range(n))


def _diatomic(a: str, b: str, r: float) -> tuple:
    return ((a, (0.0, 0.0, 0.0)), (b, (0.0, 0.0, r)))


def _entry(atom: tuple, unit: str, symmetry: str, description: str, ncas: int | None = None,
           nelecas: int | None = None) -> dict:
    return {"atom": atom, "unit": unit, "basis": "sto-3g", "symmetry": symmetry, "ncas": ncas,
            "nelecas": nelecas, "description": description}


# Geometries (in "unit") as in systems.hydrogen_chain / systems.lih / systems.n2.
BENCHMARK_SYSTEMS: dict[str, dict] = {
    "h4": _entry(_chain(4, 1.8), "bohr", "D2h", "H4 chain, 1.8 bohr, STO-3G"),
    "h6a": _entry(_chain(6, 1.0), "angstrom", "D2h", "H6 chain, 1.0 A, STO-3G"),
    "h6s": _entry(_chain(6, 3.0), "bohr", "D2h", "H6 chain, 3.0 bohr, STO-3G"),
    "h8": _entry(_chain(8, 1.8), "bohr", "D2h", "H8 chain, 1.8 bohr, STO-3G"),
    "lih": _entry(_diatomic("Li", "H", 1.595), "angstrom", "C2v", "LiH, 1.595 A, STO-3G"),
    "n2": _entry(_diatomic("N", "N", 1.098), "angstrom", "D2h", "N2, 1.098 A, STO-3G, CAS(6,6)",
                 ncas=6, nelecas=6),
    "n2s": _entry(_diatomic("N", "N", 2.0), "angstrom", "D2h", "N2, 2.0 A, STO-3G, CAS(6,6)",
                  ncas=6, nelecas=6),
}


def benchmark_system(name: str, *, symmetry: bool = True) -> ActiveSpace:
    """Active space of the benchmark system ``name`` (a key of :data:`BENCHMARK_SYSTEMS`).

    With ``symmetry`` (the default) the RHF is symmetry-adapted in the system's point
    group (D2h, or C2v for LiH) and the orbitals are ``systems.canonical_orbitals``;
    without it, the plain RHF orbitals of the same geometry. The label is ``name``.
    """
    try:
        spec = BENCHMARK_SYSTEMS[name]
    except KeyError:
        raise ValueError(f"unknown benchmark system {name!r}; known: {sorted(BENCHMARK_SYSTEMS)}") from None
    return systems.molecule(list(spec["atom"]), basis=spec["basis"], unit=spec["unit"],
                            ncas=spec["ncas"], nelecas=spec["nelecas"], label=name,
                            symmetry=spec["symmetry"] if symmetry else False)


def orbital_irreps(active: ActiveSpace) -> np.ndarray:
    """(N,) int64 PySCF irrep ids of the active orbitals of a symmetry-adapted build.

    Raises ValueError if ``active`` carries no symmetry-adapted molecule (``active.mol``
    with ``symmetry`` on), if its point group is not D2h or a subgroup, or if an
    orbital is not symmetry-pure.
    """
    mol = active.mol
    if mol is None or not getattr(mol, "symmetry", False):
        raise ValueError("orbital irreps need a symmetry-adapted build (active.mol.symmetry is off); "
                         "use benchmark_system(...) or systems.molecule(..., symmetry='D2h')")
    if mol.groupname not in _ABELIAN_GROUPS:
        raise ValueError(f"point group {mol.groupname} is not D2h or one of its subgroups; build "
                         "linear molecules with symmetry='D2h' or 'C2v'")
    if active.mo_coeff is None:
        raise ValueError("active.mo_coeff is needed to label the orbitals")
    from pyscf import symm

    ids = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, active.mo_coeff)
    return np.asarray(ids, dtype=np.int64)


def _check_irreps(irreps: np.ndarray | Sequence[int]) -> np.ndarray:
    ir = np.asarray(irreps)
    if ir.ndim != 1 or ir.size == 0:
        raise ValueError(f"irreps must be a non-empty 1-D array, got shape {ir.shape}")
    if ir.dtype.kind not in "iu":
        raise TypeError(f"irreps must be integers, got dtype {ir.dtype}")
    ir = ir.astype(np.int64)
    if np.any((ir < 0) | (ir > 7)):
        raise ValueError(f"irrep ids of D2h and its subgroups lie in 0..7, got {ir.tolist()}")
    return ir


def pair_mask(irreps: np.ndarray | Sequence[int]) -> np.ndarray:
    """(N, N, N, N) bool: True where irrep(p) XOR irrep(q) XOR irrep(r) XOR irrep(s) == 0 (F2a).

    The allowed entries of every pair-order tensor of e_{pq,rs} (residual, direction).
    """
    ir = _check_irreps(irreps)
    x = (ir[:, None, None, None] ^ ir[None, :, None, None]
         ^ ir[None, None, :, None] ^ ir[None, None, None, :])
    return x == 0


def element_mask(irreps: np.ndarray | Sequence[int], elements: np.ndarray, kind: str) -> np.ndarray:
    """(K,) bool: the symmetry-allowed rows of an element array (F2a).

    ``elements`` has the format of ``measure.targets.element_set``: (K, 4) orbitals
    (p, q, r, s) for ``kind="spin_free"``, (K, 4, 2) spin orbitals (p, sigma) for
    ``kind="spin_orbital"`` (the orbital indices decide).
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    ir = _check_irreps(irreps)
    el = np.asarray(elements)
    if el.size == 0:
        return np.zeros(0, dtype=bool)
    if el.dtype.kind not in "iu":
        raise TypeError(f"elements must be integers, got dtype {el.dtype}")
    if kind == "spin_orbital":
        if el.ndim != 3 or el.shape[1:] != (4, 2):
            raise ValueError(f"spin-orbital elements must have shape (K, 4, 2), got {el.shape}")
        orbs = el[..., 0]
    else:
        if el.ndim != 2 or el.shape[1] != 4:
            raise ValueError(f"spin-free elements must have shape (K, 4), got {el.shape}")
        orbs = el
    if np.any((orbs < 0) | (orbs >= ir.size)):
        raise ValueError(f"orbital index out of range for {ir.size} orbitals")
    g = ir[orbs]
    return (g[:, 0] ^ g[:, 1] ^ g[:, 2] ^ g[:, 3]) == 0


def mask_pair_tensor(t: np.ndarray, irreps: np.ndarray | Sequence[int]) -> np.ndarray:
    """Copy of the pair-order tensor ``t`` (N, N, N, N) with the forbidden entries set to zero (F2a)."""
    arr = np.asarray(t)
    mask = pair_mask(irreps)
    if arr.shape != mask.shape:
        raise ValueError(f"t must have shape {mask.shape}, got {arr.shape}")
    return np.where(mask, arr, np.zeros((), dtype=arr.dtype))


def _string_irreps(norb: int, n_elec: int, ir: np.ndarray) -> np.ndarray:
    """Irrep (XOR of occupied orbital irreps) of every PySCF occupation string, in address order."""
    from pyscf.fci import cistring

    strings = np.asarray(cistring.make_strings(range(norb), n_elec), dtype=np.int64)
    out = np.zeros(strings.size, dtype=np.int64)
    for p in range(norb):
        out ^= np.where((strings >> p) & 1, ir[p], 0)
    return out


def ci_irrep(ci: np.ndarray, norb: int, nelec: int | tuple[int, int],
             irreps: np.ndarray | Sequence[int]) -> tuple[int, float]:
    """Dominant irrep of a CI vector and its weight sum |c|^2 / sum |c|^2 over its determinants.

    ``ci`` is a PySCF CI array (n_alpha_strings, n_beta_strings), or its ravel, for
    ``norb`` orbitals with irrep ids ``irreps``; ``nelec`` is (n_alpha, n_beta) or a total
    (split as in PySCF, n_alpha = (n + n % 2) // 2 >= n_beta). A symmetry-pure state has
    weight 1.
    """
    ir = _check_irreps(irreps)
    if ir.size != norb:
        raise ValueError(f"irreps has {ir.size} entries for norb={norb}")
    if np.ndim(nelec) == 0:
        n = int(nelec)
        nelec = ((n + n % 2) // 2, n - (n + n % 2) // 2)
    na, nb = (int(x) for x in nelec)
    ga, gb = _string_irreps(norb, na, ir), _string_irreps(norb, nb, ir)
    c = np.asarray(ci)
    if c.shape not in ((ga.size, gb.size), (ga.size * gb.size,)):
        raise ValueError(f"ci has shape {c.shape}, expected ({ga.size}, {gb.size}) alpha x beta strings "
                         f"or its ravel")
    w = np.abs(c.reshape(ga.size, gb.size)) ** 2
    total = float(w.sum())
    if not total > 0.0:
        raise ValueError("ci must be nonzero")
    weights = np.bincount((ga[:, None] ^ gb[None, :]).ravel(), weights=w.ravel(), minlength=8)
    g = int(np.argmax(weights))
    return g, float(weights[g] / total)
