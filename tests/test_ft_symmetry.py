"""Tests of the M3 symmetry layer (``thc_engine_cqe.ft.symmetry``): benchmark builds on symmetry-adapted
orbitals, orbital irreps, the point-group masks of F2a and determinant irreps.

References independent of the module: the current builders of ``systems`` (plain RHF for the chains
and LiH) and plain-RHF builds of N2 (``systems.n2`` is itself the D2h build), the AO labels of PySCF,
PySCF's character tables (not the XOR rule), the allowed-element
counts of the M3 specification (Appendix A, from the critic's ``sym_check.py``), the exact residual of
``residual.compute_residual`` and of ``routes.exact_residual`` (PySCF transition RDMs), neither of
which knows about symmetry, and ``pyscf.fci.addons.symmetrize_wfn``.
"""

import itertools

import numpy as np
import pytest

from thc_engine_cqe import systems
from thc_engine_cqe.cqe import run_cqe
from thc_engine_cqe.fci_ops import Hamiltonian
from thc_engine_cqe.ft import symmetry as sy
from thc_engine_cqe.measure import routes as rt
from thc_engine_cqe.measure import targets as tg

CURRENT_BUILDERS = {
    "h4": lambda: systems.hydrogen_chain(4, 1.8, "bohr"),
    "h6a": lambda: systems.hydrogen_chain(6, 1.0, "angstrom"),
    "h6s": lambda: systems.hydrogen_chain(6, 3.0, "bohr"),
    "h8": lambda: systems.hydrogen_chain(8, 1.8, "bohr"),
    "lih": lambda: systems.lih(),
    "n2": lambda: systems.n2(1.098),
    "n2s": lambda: systems.n2(2.0),
}

# systems.n2 is itself a D2h build on canonical_orbitals, the same build as benchmark_system("n2"), so
# for N2 the independent reference is the plain RHF of the same geometry (what systems.n2 built before
# it was symmetry-adapted; its pi pairs have an arbitrary orientation).
PLAIN_RHF_N2 = {
    "n2": lambda: systems.molecule([("N", (0, 0, 0)), ("N", (0, 0, 1.098))], ncas=6, nelecas=6),
    "n2s": lambda: systems.molecule([("N", (0, 0, 0)), ("N", (0, 0, 2.0))], ncas=6, nelecas=6),
}

# Appendix A of the M3 specification: allowed / all elements, spin-free and spin-orbital.
APPENDIX_A_COUNTS = {
    "h4": ((28, 60), (70, 150)),
    "h6a": ((153, 315), (408, 840)),
    "lih": ((105, 315), (264, 840)),
    "n2": ((33, 315), (84, 840)),
    "h8": ((496, 1008), (1364, 2772)),
}

# Appendix A: L-BFGS reaches 1.6 mHa after 3 / 6 / 3 steps.
K_STAR = {"h4": 3, "h6a": 6, "n2": 3}


@pytest.fixture(scope="module")
def builds():
    """Symmetry-adapted builds of the four core systems, with their irreps."""
    out = {}
    for name in ("h4", "h6a", "lih", "n2"):
        act = sy.benchmark_system(name)
        out[name] = (act, sy.orbital_irreps(act))
    return out


def _characters(group: str) -> dict:
    """irrep id -> character vector, from PySCF's character table (not from the ids)."""
    from pyscf.symm import param

    ids = param.IRREP_ID_TABLE[group]
    return {ids[row[0]]: np.array(row[1:]) for row in param.CHARACTER_TABLE[group]}


@pytest.mark.parametrize("name", list(CURRENT_BUILDERS))
def test_symmetric_builds_reproduce_the_current_builders(name):
    """E_HF (the active-space HF determinant) and E_FCI of the symmetry-adapted build equal those of
    plain-RHF builds of the same geometry to 1e-10 Ha: the current builder for the chains and LiH, and for
    n2 and n2s a plain systems.molecule build (systems.n2 is now the same D2h build, so it only checks the
    geometry and the active space). The build is labelled and in the point group of BENCHMARK_SYSTEMS."""
    act = sy.benchmark_system(name)
    assert act.label == name
    assert act.mol.groupname == sy.BENCHMARK_SYSTEMS[name]["symmetry"]
    refs = [CURRENT_BUILDERS[name]()] + ([PLAIN_RHF_N2[name]()] if name in PLAIN_RHF_N2 else [])
    assert not refs[-1].mol.symmetry                          # at least one reference is unsymmetrized
    h_sym = Hamiltonian(act)
    e_hf, e_fci = h_sym.energy(h_sym.space.hf()), act.fci()[0]
    for ref in refs:
        assert (act.norb, act.nelec) == (ref.norb, ref.nelec)
        h_ref = Hamiltonian(ref)
        assert abs(e_hf - h_ref.energy(h_ref.space.hf())) <= 1e-10
        assert abs(e_fci - ref.fci()[0]) <= 1e-10


def test_unsymmetrized_build_has_no_irreps():
    """benchmark_system(..., symmetry=False) is the plain RHF of the same geometry (same E_FCI), and
    orbital_irreps rejects it, as it rejects unknown names and non-abelian (Dooh) builds."""
    plain, sym = sy.benchmark_system("h4", symmetry=False), sy.benchmark_system("h4")
    assert not plain.mol.symmetry
    assert abs(plain.fci()[0] - sym.fci()[0]) <= 1e-10
    with pytest.raises(ValueError):
        sy.orbital_irreps(plain)
    with pytest.raises(ValueError):
        sy.benchmark_system("h5")
    dooh = systems.molecule([("H", (0, 0, 0)), ("H", (0, 0, 0.74))], symmetry=True)
    assert dooh.mol.groupname == "Dooh"
    with pytest.raises(ValueError):
        sy.orbital_irreps(dooh)


@pytest.mark.parametrize("name", ["n2", "n2s"])
def test_n2_pi_orbitals_are_axis_aligned(name):
    """Each N2 active orbital lives on the AOs of its irrep's axis, read from the AO labels: the pi_x
    orbitals (B3u, B2g) on p_x only, the pi_y orbitals (B2u, B3g) on p_y only, sigma (Ag, B1u) on s and
    p_z only, to 1e-12; and pi_x precedes pi_y in each degenerate pair (B3u before B2u, B2g before B3g)."""
    act = sy.benchmark_system(name)
    ir = sy.orbital_irreps(act)
    labels = act.mol.ao_labels()
    kind = np.array([next((c for c in ("px", "py", "pz") if c in s), "s") for s in labels])
    allowed = {7: {"px"}, 2: {"px"}, 6: {"py"}, 3: {"py"}, 0: {"s", "pz"}, 5: {"s", "pz"}}
    assert sorted(ir.tolist()) == [0, 2, 3, 5, 6, 7]
    for k, g in enumerate(ir):
        off = ~np.isin(kind, list(allowed[int(g)]))
        assert np.max(np.abs(act.mo_coeff[off, k])) <= 1e-12
    pos = {int(g): k for k, g in enumerate(ir)}
    assert pos[7] < pos[6] and pos[2] < pos[3]


def test_orbital_irreps_of_the_core_systems(builds):
    """Irreps in the canonical orbital order (observed when the module was written; a regression of the
    orbital order of systems.canonical_orbitals that the other M3 tests rely on)."""
    assert builds["h4"][1].tolist() == [0, 5, 0, 5]
    assert builds["h6a"][1].tolist() == [0, 5, 0, 5, 0, 5]
    assert builds["lih"][1].tolist() == [0, 0, 0, 2, 3, 0]
    assert builds["n2"][1].tolist() == [7, 6, 0, 2, 3, 5]
    assert builds["lih"][0].mol.groupname == "C2v"


@pytest.mark.parametrize("group", ["D2h", "C2v"])
def test_xor_rule_is_the_group_product(group):
    """For all quadruples of irreps of D2h and C2v, XOR of the ids is 0 iff the product of the four
    character vectors of PySCF's character table is the trivial character; and pair_mask on a random
    irrep assignment equals that character criterion entry by entry."""
    chars = _characters(group)
    ids = sorted(chars)
    for quad in itertools.product(ids, repeat=4):
        prod = np.prod([chars[g] for g in quad], axis=0)
        assert (quad[0] ^ quad[1] ^ quad[2] ^ quad[3] == 0) == bool(np.all(prod == 1))
    rng = np.random.default_rng(3)
    ir = rng.choice(ids, size=5)
    mask = sy.pair_mask(ir)
    for p, q, r, s in itertools.product(range(5), repeat=4):
        prod = chars[ir[p]] * chars[ir[q]] * chars[ir[r]] * chars[ir[s]]
        assert mask[p, q, r, s] == bool(np.all(prod == 1))


@pytest.mark.parametrize("name", list(APPENDIX_A_COUNTS))
def test_allowed_element_counts_match_appendix_a(name):
    """Allowed / all elements of targets.element_set, spin-free and spin-orbital, equal Appendix A
    (computed by the critic's script); the spin-free mask is pair_mask at the element rows and the
    spin-orbital mask the XOR of the rows' orbital indices."""
    act = sy.benchmark_system(name)
    ir = sy.orbital_irreps(act)
    (sf_ok, sf_all), (so_ok, so_all) = APPENDIX_A_COUNTS[name]
    el_sf = tg.element_set(act.norb, "spin_free")
    m_sf = sy.element_mask(ir, el_sf, "spin_free")
    assert (int(m_sf.sum()), m_sf.size) == (sf_ok, sf_all)
    assert np.array_equal(m_sf, sy.pair_mask(ir)[tuple(el_sf.T)])
    el_so = tg.element_set(act.norb, "spin_orbital")
    m_so = sy.element_mask(ir, el_so, "spin_orbital")
    assert (int(m_so.sum()), m_so.size) == (so_ok, so_all)
    assert np.array_equal(m_so, sy.pair_mask(ir)[tuple(el_so[..., 0].T)])


@pytest.mark.parametrize("name", ["h4", "h6a", "lih", "n2"])
def test_forbidden_residual_elements_vanish_along_sd(builds, name):
    """Along 5 exact steepest-descent steps, the forbidden entries of the pair-order residual
    (residual.compute_residual, which knows nothing of symmetry) are <= 1e-10 max|A|, the states keep the
    HF irrep (totally symmetric) with weight > 1 - 1e-8, and at the last state the spin-orbital residual
    from PySCF transition RDMs (routes.exact_residual) is <= 1e-10 max|A| on the forbidden rows, while
    the allowed rows carry the whole norm."""
    act, ir = builds[name]
    pm = sy.pair_mask(ir)
    states = []
    res = run_cqe(act, max_iter=5, keep_residuals=True, callback=lambda st, ci: states.append(ci.copy()))
    assert len(res.steps) == 5
    for st in res.steps:
        assert np.max(np.abs(st.residual[~pm])) <= 1e-10 * np.max(np.abs(st.residual))
    for ci in states:
        g, w = sy.ci_irrep(ci, act.norb, act.nelec, ir)
        assert g == 0 and w > 1 - 1e-8
    el = tg.element_set(act.norb, "spin_orbital")
    a_so = rt.exact_residual(act, res.ci, el, kind="spin_orbital")
    m = sy.element_mask(ir, el, "spin_orbital")
    assert np.max(np.abs(a_so[~m])) <= 1e-10 * np.max(np.abs(a_so))
    assert np.sum(a_so[m] ** 2) == pytest.approx(np.sum(a_so ** 2), rel=1e-12)


@pytest.mark.parametrize("name", list(K_STAR))
def test_lbfgs_states_are_symmetry_pure(name):
    """The exact L-BFGS iterates of Appendix A, up to k* + 2 steps, stay in the totally symmetric irrep
    with weight > 1 - 1e-8, as exact arithmetic requires (a totally symmetric generator maps each irrep
    block to itself); ci_irrep itself is checked against PySCF in
    test_ci_irrep_matches_pyscf_symmetrize_wfn."""
    act = sy.benchmark_system(name)
    ir = sy.orbital_irreps(act)
    states = []
    run_cqe(act, max_iter=K_STAR[name] + 2, optimizer="lbfgs",
            callback=lambda st, ci: states.append(ci.copy()))
    assert len(states) == K_STAR[name] + 2
    for ci in states:
        g, w = sy.ci_irrep(ci, act.norb, act.nelec, ir)
        assert g == 0 and w > 1 - 1e-8


def test_masked_residual_keeps_long_lbfgs_runs_pure():
    """Exact L-BFGS amplifies the round-off in the forbidden entries (on N2 at 2.0 A it reached 2e-4 of
    weight outside the HF irrep within 20 steps when this test was written). Projecting the residual with
    mask_pair_tensor through run_cqe's residual_estimator hook keeps every iterate pure to 1e-12 and the
    forbidden residual at round-off, which is why directions are masked before compilation (F2a)."""
    act = sy.benchmark_system("n2s")
    ir = sy.orbital_irreps(act)
    pm = sy.pair_mask(ir)
    states = []
    res = run_cqe(act, max_iter=20, optimizer="lbfgs", keep_residuals=True,
                  residual_estimator=lambda rd, ci, it: sy.mask_pair_tensor(rd.A, ir),
                  callback=lambda st, ci: states.append(ci.copy()))
    assert len(states) == 20
    for st, ci in zip(res.steps, states):
        assert np.max(np.abs(st.residual[~pm])) <= 1e-13
        g, w = sy.ci_irrep(ci, act.norb, act.nelec, ir)
        assert g == 0 and w > 1 - 1e-12


def test_ci_irrep_matches_pyscf_symmetrize_wfn(builds):
    """On mixtures of FCI roots of different irreps (N2), ci_irrep returns the irrep of largest weight and
    that weight, which equals <P_k ci, ci>^2 / ||ci||^2 with P_k ci the normalized projection of PySCF's
    symmetrize_wfn; single determinants have the XOR of their occupied irreps (HF: 0; the excitation
    p -> q from HF: irrep(p) XOR irrep(q))."""
    from pyscf.fci import addons

    act, ir = builds["n2"]
    n, nelec = act.norb, act.nelec
    _, roots = act.fci(nroots=8)
    rng = np.random.default_rng(2)
    for _ in range(5):
        ci = sum(rng.standard_normal() * c for c in roots)
        g, w = sy.ci_irrep(ci, n, nelec, ir)
        with np.errstate(invalid="ignore", divide="ignore"):    # irreps without weight: 0/0
            proj = [addons.symmetrize_wfn(ci, n, nelec, ir, k) for k in range(8)]
        weights = np.nan_to_num([np.vdot(p, ci) ** 2 / np.sum(ci ** 2) for p in proj])
        assert np.count_nonzero(weights > 1e-6) >= 2                  # a genuine mixture
        assert g == int(np.argmax(weights))
        assert w == pytest.approx(max(weights), abs=1e-12)
        assert sum(weights) == pytest.approx(1.0, abs=1e-12)
    hf = np.zeros((roots[0].shape))
    hf[0, 0] = 1.0
    assert sy.ci_irrep(hf, n, nelec, ir) == (0, 1.0)
    assert sy.ci_irrep(1j * hf, n, sum(nelec), ir) == (0, 1.0)
    from pyscf.fci import cistring

    strings = cistring.make_strings(range(n), nelec[0])
    for p, q in [(2, 3), (1, 4), (0, 5)]:                     # occupied p -> virtual q, alpha spin
        target = (1 << 0 | 1 << 1 | 1 << 2) ^ (1 << p) ^ (1 << q)
        det = np.zeros_like(hf)
        det[int(np.flatnonzero(strings == target)[0]), 0] = 1.0
        assert sy.ci_irrep(det, n, nelec, ir) == (int(ir[p] ^ ir[q]), 1.0)
    with pytest.raises(ValueError):
        sy.ci_irrep(np.zeros_like(hf), n, nelec, ir)
    with pytest.raises(ValueError):
        sy.ci_irrep(hf[:-1], n, nelec, ir)


def test_ci_irrep_splits_an_odd_electron_count_as_pyscf():
    """A total electron count splits as PySCF's fci.addons._unpack_nelec does (5 -> (3, 2)): a 4 x 6 CI
    array of 4 orbitals with ids (0, 1, 2, 3) is accepted, and the determinant alpha {1, 2, 3},
    beta {0, 1} (addresses from cistring.str2addr) has the irrep (1 ^ 2 ^ 3) ^ (0 ^ 1) = 1, as with the
    explicit (3, 2); the transposed 6 x 4 array, the layout of the split (2, 3), is rejected."""
    from pyscf.fci import addons, cistring

    assert addons._unpack_nelec(5) == (3, 2)
    ir = np.array([0, 1, 2, 3])
    ci = np.zeros((cistring.num_strings(4, 3), cistring.num_strings(4, 2)))
    ci[cistring.str2addr(4, 3, 0b1110), cistring.str2addr(4, 2, 0b0011)] = 1.0
    assert ci.shape == (4, 6)
    assert sy.ci_irrep(ci, 4, 5, ir) == (1, 1.0) == sy.ci_irrep(ci, 4, (3, 2), ir)
    assert sy.ci_irrep(ci.ravel(), 4, 5, ir) == (1, 1.0)
    with pytest.raises(ValueError):
        sy.ci_irrep(ci.T, 4, 5, ir)


def test_mask_helpers_and_validation(builds):
    """mask_pair_tensor zeroes exactly the entries pair_mask forbids, keeps the dtype (complex directions)
    and leaves its input untouched; malformed inputs raise."""
    _, ir = builds["n2"]
    rng = np.random.default_rng(4)
    t = rng.standard_normal((6,) * 4) + 1j * rng.standard_normal((6,) * 4)
    t0 = t.copy()
    out = sy.mask_pair_tensor(t, ir)
    pm = sy.pair_mask(ir)
    assert out.dtype == t.dtype and np.array_equal(t, t0)
    assert np.array_equal(out[pm], t[pm]) and not np.any(out[~pm])
    assert sy.pair_mask(np.zeros(6, dtype=int)).all()                 # C1: nothing is forbidden
    with pytest.raises(ValueError):
        sy.mask_pair_tensor(t[:5], ir)
    with pytest.raises(ValueError):
        sy.element_mask(ir, tg.element_set(6, "spin_free"), "spin")
    with pytest.raises(ValueError):
        sy.element_mask(ir, tg.element_set(6, "spin_orbital"), "spin_free")
    with pytest.raises(ValueError):
        sy.element_mask(ir[:4], tg.element_set(6, "spin_free"), "spin_free")
    with pytest.raises(TypeError):
        sy.pair_mask(np.array([0.0, 5.0]))
    with pytest.raises(ValueError):
        sy.pair_mask(np.array([0, 8]))
    assert sy.element_mask(ir, np.zeros((0, 4), dtype=int), "spin_free").shape == (0,)
