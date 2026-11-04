import numpy as np
import pytest

from thc_engine_cqe import systems
from thc_engine_cqe.cqe import run_cqe
from thc_engine_cqe.fci_ops import FCISpace, Hamiltonian
from thc_engine_cqe.integrals import ActiveSpace
from thc_engine_cqe.residual import compute_residual


@pytest.fixture(scope="module")
def h4():
    return systems.hydrogen_chain(4, spacing=1.8, unit="bohr")


def test_active_space_basics(h4):
    h4.check_symmetry()
    assert h4.norb == 4 and h4.nelec == (2, 2)
    e_fci, _ = h4.fci()
    from pyscf import fci, gto, scf

    mol = gto.M(atom=[("H", (0, 0, 1.8 * i)) for i in range(4)], basis="sto-3g", unit="bohr", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-11)
    e_ref = fci.FCI(mf).kernel()[0]
    assert np.isclose(e_fci, e_ref, atol=1e-8)


def test_cas_active_space_energy():
    """CASCI energy from the ActiveSpace matches PySCF's CASCI."""
    from pyscf import gto, mcscf, scf

    mol = gto.M(atom="Li 0 0 0; H 0 0 1.595", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-11)
    act = ActiveSpace.from_pyscf(mf, ncas=4, nelecas=2)
    e_act, _ = act.fci()
    e_ref = mcscf.CASCI(mf, 4, 2).run().e_tot
    assert np.isclose(e_act, e_ref, atol=1e-8)


def test_rotation_invariance(h4):
    u = systems.random_orthogonal(h4.norb, np.random.default_rng(3))
    rot = h4.rotated(u)
    rot.check_symmetry()
    assert np.isclose(rot.fci()[0], h4.fci()[0], atol=1e-9)


def test_fcidump_roundtrip(h4, tmp_path):
    path = str(tmp_path / "h4.fcidump")
    h4.to_fcidump(path)
    back = ActiveSpace.from_fcidump(path)
    assert back.nelec == h4.nelec
    assert np.allclose(back.h1, h4.h1, atol=1e-12)
    assert np.allclose(back.eri, h4.eri, atol=1e-12)
    assert np.isclose(back.ecore, h4.ecore)


def test_exact_cqe_converges_h4(h4):
    """Steepest descent with exact line search: monotone, but with a slow
    zig-zag tail (about 8e-6 Ha above FCI after 40 steps for H4 at 1.8 bohr)."""
    e_fci, _ = h4.fci()
    res = run_cqe(h4, max_iter=40, e_exact=e_fci)
    e = res.energies
    assert np.all(np.diff(e) <= 1e-12), "energy must decrease monotonically"
    assert e[-1] - e_fci < 2e-5
    assert res.steps[-1].residual_norm < res.steps[0].residual_norm / 100
    # every step uses the full steepest-descent rate
    assert all(abs(s.captured_descent - 1) < 1e-12 for s in res.steps)
    # the CQE state stays a singlet (spin-free generators)
    from pyscf.fci import spin_op

    ss, _ = spin_op.spin_square0(res.ci, h4.norb, h4.nelec)
    assert abs(ss) < 1e-8


def test_cqe_step_energy_is_exact(h4):
    """The Krylov line-search energy equals the energy of the returned state."""
    seen = []

    def cb(step, ci):
        seen.append((step.energy_after, Hamiltonian(h4).energy(ci)))

    run_cqe(h4, max_iter=3, callback=cb)
    for e_ls, e_state in seen:
        assert np.isclose(e_ls, e_state, atol=1e-10)


def test_first_residual_from_hf_is_ov_ov_block(h4):
    """From RHF, A is nonzero only on (vir occ | vir occ)-type index blocks."""
    space = FCISpace.of(h4)
    rd = compute_residual(h4, space.hf())
    nocc = h4.nelec[0]
    occ, vir = slice(0, nocc), slice(nocc, h4.norb)
    mask = np.zeros_like(rd.A, dtype=bool)
    # e_{pq,rs} with p,r virtual and q,s occupied (excitations) and its adjoint
    mask[vir, occ, vir, occ] = True
    mask[occ, vir, occ, vir] = True
    # Off-block elements are Brillouin (orbital-gradient) terms: zero up to the
    # SCF gradient tolerance (1e-9).
    assert np.abs(rd.A[~mask]).max() < 1e-8
    assert np.abs(rd.A[mask]).max() > 1e-3
