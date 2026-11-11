"""Tests of the parametric system names of ``thc_engine_cqe.ft.symmetry`` (F20) and of the CAS rule of
``integrals.ActiveSpace.from_pyscf``.

References: the M3 entries of ``BENCHMARK_SYSTEMS`` (the aliases must build them bit for bit), a direct
``systems.molecule`` build of the same geometry and active space, the determinant count C(8,5)^2 of
CAS(10,8), the HF weight of the CAS(10,8) FCI ground state at 1.098 A (0.9173 [design]), PySCF's own
core count (``mcscf.CASCI(...).ncore``) for the CAS rule, and the point-group selection rule for the
square H4, whose B1g ground state is orthogonal to the Ag HF determinant.
"""

import math
import warnings

import numpy as np
import pytest

from thc_engine_cqe import systems
from thc_engine_cqe.fci_ops import FCISpace
from thc_engine_cqe.ft import symmetry as sy
from thc_engine_cqe.ft.trajectory import exact_trajectory
from thc_engine_cqe.integrals import ActiveSpace

ENTRY_KEYS = {"atom", "unit", "basis", "symmetry", "ncas", "nelecas", "description"}


def _assert_same_build(a: ActiveSpace, b: ActiveSpace) -> None:
    for x, y in ((a.h1, b.h1), (a.eri, b.eri), (a.mo_coeff, b.mo_coeff),
                 (sy.orbital_irreps(a), sy.orbital_irreps(b))):
        assert x.shape == y.shape and np.array_equal(x, y)
    assert a.ecore == b.ecore and a.nelec == b.nelec


@pytest.mark.parametrize("m3", ["h4", "h6s", "n2", "n2s", pytest.param("h8", marks=pytest.mark.slow)])
def test_parametric_names_match_m3_builds(m3):
    """Each alias builds the M3 system bit for bit (h1, eri, ecore, mo_coeff, irreps); the entries agree key by
    key except the description, and the M3 name keeps its label."""
    par = sy.ALIASES[m3]
    spec, m3_spec = sy.system_spec(par), sy.BENCHMARK_SYSTEMS[m3]
    assert set(spec) == ENTRY_KEYS == set(m3_spec)
    assert all(spec[k] == m3_spec[k] for k in ENTRY_KEYS - {"description"})
    a, b = sy.benchmark_system(m3), sy.benchmark_system(par)
    _assert_same_build(a, b)
    assert (a.label, b.label) == (m3, par)


def test_aliases_cover_the_m3_names_with_a_pattern():
    """The five aliases name M3 systems; h6a and lih have no parametric name; the parametric names are not
    M3 keys (they take the fallback path)."""
    assert set(sy.ALIASES) == {"h4", "h6s", "h8", "n2", "n2s"} <= set(sy.BENCHMARK_SYSTEMS)
    assert not set(sy.ALIASES.values()) & set(sy.BENCHMARK_SYSTEMS)
    for name in ("h6a", "lih"):
        assert name not in sy.ALIASES


@pytest.mark.parametrize("name, atom, unit, ncas, nelecas", [
    ("h6_r2.5", [("H", (0.0, 0.0, 2.5 * i)) for i in range(6)], "bohr", None, None),
    ("h10_r4.0", [("H", (0.0, 0.0, 4.0 * i)) for i in range(10)], "bohr", None, None),
    ("n2_r2.4", [("N", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 2.4))], "angstrom", 6, 6),
    ("n2v_r2.8", [("N", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 2.8))], "angstrom", 8, 10),
    ("h4rect_1.8x2.2", [("H", (0.0, 0.0, 0.0)), ("H", (1.8, 0.0, 0.0)), ("H", (1.8, 2.2, 0.0)),
                        ("H", (0.0, 2.2, 0.0))], "bohr", None, None),
])
def test_system_spec_patterns(name, atom, unit, ncas, nelecas):
    """Geometry (i R on z for chains, N2 on z, the rectangle in the xy plane), unit, active space, STO-3G and
    D2h of every pattern."""
    spec = sy.system_spec(name)
    assert set(spec) == ENTRY_KEYS
    assert [(s, tuple(map(float, xyz))) for s, xyz in spec["atom"]] == [(s, xyz) for s, xyz in atom]
    assert (spec["unit"], spec["basis"], spec["symmetry"]) == (unit, "sto-3g", "D2h")
    assert (spec["ncas"], spec["nelecas"]) == (ncas, nelecas)


def test_parametric_build_equals_a_direct_build():
    """benchmark_system on a parametric name is systems.molecule on its entry (D2h, canonical orbitals), with the
    name as label; symmetry=False gives the plain RHF of the same geometry (same E_FCI)."""
    name = "n2_r2.4"
    a = sy.benchmark_system(name)
    ref = systems.molecule([("N", (0, 0, 0)), ("N", (0, 0, 2.4))], basis="sto-3g", unit="angstrom", ncas=6,
                           nelecas=6, symmetry="D2h")
    _assert_same_build(a, ref)
    assert a.label == name and a.mol.groupname == "D2h"
    plain = sy.benchmark_system(name, symmetry=False)
    assert not plain.mol.symmetry
    assert plain.fci()[0] == pytest.approx(a.fci()[0], abs=1e-9)


def test_n2v_builds():
    """n2v_r1.098: CAS(10,8) with ncore 2, N = 8, nelec (5, 5), C(8,5)^2 = 3136 determinants and
    p_HF = 0.9173 +- 1e-4 [design]; orbital irreps are those of D2h."""
    a = sy.benchmark_system("n2v_r1.098")
    assert (a.norb, a.nelec) == (8, (5, 5))
    space = FCISpace.of(a)
    assert space.size == math.comb(8, 5) ** 2 == 3136
    ir = sy.orbital_irreps(a)
    assert ir.shape == (8,) and set(ir.tolist()) <= set(range(8))
    _, ci = a.fci()
    p_hf = float(abs(np.vdot(space.hf().ravel(), ci.ravel())) ** 2 / np.vdot(ci, ci).real)
    assert p_hf == pytest.approx(0.9173, abs=1e-4)


def test_cas_beyond_the_orbitals_raises():
    """N2 in STO-3G has 10 MOs: CAS(8,8) needs ncore = 3 and 3 + 8 > 10 orbitals -> ValueError (PySCF would
    slice a 7-orbital active space); CAS(8,7) and CAS(14,10) fit and build."""
    with pytest.raises(ValueError, match=r"ncore \+ ncas = 3 \+ 8 = 11 exceeds the 10 molecular orbitals"):
        systems.n2(1.098, ncas=8, nelecas=8)
    from pyscf import gto, scf
    mol = gto.M(atom=[("N", (0, 0, 0)), ("N", (0, 0, 1.098))], basis="sto-3g", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    with pytest.raises(ValueError, match="exceeds the 10 molecular orbitals"):
        ActiveSpace.from_pyscf(mf, ncas=11, nelecas=14)
    for ncas, nelecas in ((7, 8), (10, 14)):
        act = ActiveSpace.from_pyscf(mf, ncas=ncas, nelecas=nelecas)
        assert act.norb == ncas and sum(act.nelec) == nelecas and act.eri.shape == (ncas,) * 4


def test_h4_square_ground_state_flag():
    """h4rect_1.8x1.8: the square's ground state is B1g (D2h id 1), the HF determinant Ag, so the exact
    trajectory carries "ground_state_not_in_hf_irrep" and every overlap with psi_0 vanishes (p_HF = 0 up to
    the solver's round-off: the CQE keeps the HF irrep)."""
    a = sy.benchmark_system("h4rect_1.8x1.8")
    ir = sy.orbital_irreps(a)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rec = exact_trajectory(a, max_iter=2, irreps=ir, n_roots=12)
    assert "ground_state_not_in_hf_irrep" in rec.flags
    assert rec.ground_irrep == 1
    assert sy.ci_irrep(FCISpace.of(a).hf(), a.norb, a.nelec, ir)[0] == 0
    assert np.max(rec.overlaps) < 1e-20


@pytest.mark.parametrize("name", ["h4", "h6a", "lih", "n2s", "h5_r1.0", "h0_r1.0", "h04_r1.8", "h4_r0",
                                  "h4_r0.0", "h4_r", "h4_r1.8x", "h4_rinf", "n2_r", "n2v_r-1.0", "n2_r1e1",
                                  "h4rect_1.8", "h4rect_0x1.8", "N2_r1.098", " h4_r1.8", "foo"])
def test_unknown_names_raise(name):
    """system_spec accepts only the four patterns (not the M3 names, odd chains, zero lengths or other
    spellings); benchmark_system raises ValueError for everything that is neither an M3 key nor a pattern."""
    with pytest.raises(ValueError):
        sy.system_spec(name)
    if name not in sy.BENCHMARK_SYSTEMS:
        with pytest.raises(ValueError, match="unknown benchmark system"):
            sy.benchmark_system(name)


@pytest.mark.parametrize("bad", [None, 4, ("h4_r1.8",)])
def test_non_string_names_raise(bad):
    with pytest.raises(ValueError):
        sy.system_spec(bad)
    with pytest.raises(ValueError):
        sy.benchmark_system(bad)
