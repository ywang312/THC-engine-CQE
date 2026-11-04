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


def test_fixed_eps_candidates(h4):
    """eps_candidates: every step takes the lowest energy among the fixed step sizes."""
    from thc_engine_cqe.fci_ops import expm_apply, normalize, two_body

    cands = (0.025, 0.05, 0.1, 0.2)
    space = FCISpace.of(h4)
    ham = Hamiltonian(h4)
    states = []
    res = run_cqe(h4, max_iter=6, eps_candidates=cands, keep_generators=True,
                  callback=lambda step, ci: states.append(ci))
    psi = space.hf()
    for step, nxt in zip(res.steps, states):
        # independent check: the energy of exp(eps G) psi at every candidate
        es = [ham.energy(normalize(expm_apply(lambda v: two_body(space, step.generator, v), psi, e)))
              for e in cands]
        assert step.eps == cands[int(np.argmin(es))]
        assert np.isclose(step.energy_after, min(es), atol=1e-10)
        assert np.isclose(ham.energy(nxt), step.energy_after, atol=1e-10)
        psi = nxt
    e_fci, _ = h4.fci()
    assert res.energies[-1] - e_fci < 0.1 * (res.energies[0] - e_fci)


def test_fixed_eps_candidates_can_overshoot(h4):
    """With only an overshooting step size the step is still taken, as on hardware
    (E(1.0) - E(0) = +1.86 Ha along the first H4 generator)."""
    res = run_cqe(h4, max_iter=1, eps_candidates=(1.0,))
    (step,) = res.steps
    assert step.eps == 1.0 and step.energy_after > step.energy + 1.0


def test_line_search_candidates_validation():
    """Candidates must be a non-empty sequence of finite reals; zero and negative
    step sizes are allowed (the first of tied minima wins)."""
    from thc_engine_cqe.cqe import line_search

    for bad in [(), (0.1, np.nan), (np.inf,), (0.1, -np.inf)]:
        with pytest.raises(ValueError):
            line_search(lambda e: e, 1.0, candidates=bad)
    eps, e = line_search(lambda x: (x - 0.3) ** 2, 1.0, candidates=(0.1, 0.25, 0.5))
    assert eps == 0.25 and np.isclose(e, 0.05**2)
    eps, e = line_search(lambda x: (x + 0.2) ** 2, 1.0, candidates=(0.1, -0.2, 0.0))
    assert eps == -0.2 and e == 0.0
    eps, _ = line_search(lambda x: x**2, 1.0, candidates=(0.5, 0.0, -0.5))
    assert eps == 0.0
    eps, _ = line_search(lambda x: abs(x), 1.0, candidates=(-0.1, 0.1))
    assert eps == -0.1                              # tie: the first candidate


def _energy_along(h4, g, psi, eps):
    from thc_engine_cqe.fci_ops import expm_apply, normalize, two_body

    space = FCISpace.of(h4)
    return Hamiltonian(h4).energy(normalize(expm_apply(lambda v: two_body(space, g, v), psi, eps)))


def test_negative_candidate_recovers_from_an_uphill_direction(h4):
    """With an uphill estimated direction (A_est = -A), only the negative candidates
    lower the energy; the step takes the best of them and propagates it exactly."""
    cands = (-0.2, -0.1, -0.05, 0.05, 0.1)
    res = run_cqe(h4, max_iter=1, eps_candidates=cands, keep_generators=True,
                  residual_estimator=lambda rd, ci, it: -rd.A)
    (step,) = res.steps
    psi = FCISpace.of(h4).hf()
    es = [_energy_along(h4, step.generator, psi, e) for e in cands]
    assert step.slope > 0                           # uphill for the exact energy
    assert step.eps == cands[int(np.argmin(es))] and step.eps < 0
    assert step.energy_after < step.energy - 1e-3
    assert np.isclose(step.energy_after, min(es), atol=1e-10)
    assert np.isclose(Hamiltonian(h4).energy(res.ci), step.energy_after, atol=1e-10)


def test_negative_candidates_match_the_mirrored_positive_ones(h4):
    """Stepping by -eps along d is stepping by +eps along -d: same states and energies,
    which also checks the Krylov accuracy at negative eps."""
    cands = (0.025, 0.05, 0.1, 0.2)
    ref = run_cqe(h4, max_iter=4, eps_candidates=cands)
    mir = run_cqe(h4, max_iter=4, eps_candidates=tuple(-c for c in cands),
                  residual_estimator=lambda rd, ci, it: -rd.A)
    assert [s.eps for s in mir.steps] == [-s.eps for s in ref.steps]
    assert np.allclose(mir.energies, ref.energies, atol=1e-10)
    assert abs(abs(np.vdot(mir.ci.ravel(), ref.ci.ravel())) - 1) < 1e-10


def test_zero_candidate_keeps_the_state_and_is_not_convergence(h4):
    """A zero candidate is the current state: its energy is the current energy (not
    re-evaluated, not counted), and choosing it is not e_tol convergence. Without
    estimators a repeated zero step stops the run, unconverged."""
    # 5.0 overshoots along the first H4 generator, so 0 wins
    res = run_cqe(h4, max_iter=10, eps_candidates=(0.0, 5.0))
    assert len(res.steps) == 2 and not res.converged
    for s in res.steps:
        assert s.eps == 0.0 and s.energy_after == s.energy and s.n_energy_evals == 1
    assert np.array_equal(res.ci, FCISpace.of(h4).hf())
    for opt in ("cg", "lbfgs"):
        r = run_cqe(h4, max_iter=10, eps_candidates=(0.0, 5.0), optimizer=opt)
        assert len(r.steps) == 2 and not r.converged and all(s.eps == 0.0 for s in r.steps)
    # with an energy_estimator the zero candidate reuses the "current" estimate
    log = []

    def est(e, ci, ctx):
        log.append(ctx)
        return e

    r = run_cqe(h4, max_iter=3, eps_candidates=(0.0, 5.0), energy_estimator=est)
    assert log == ["current", "candidate"] * 3       # no call for the zero candidate
    assert len(r.steps) == 3 and not r.converged
    assert all(s.eps == 0.0 and s.energy_after_est == s.energy_est for s in r.steps)
    # a usable candidate set with 0 included still converges normally
    ok = run_cqe(h4, max_iter=40, eps_candidates=(0.0, 0.025, 0.05, 0.1, 0.2), r_tol=0.0, e_tol=1e-10)
    e_fci, _ = h4.fci()
    assert ok.energies[-1] - e_fci < 1e-4 and all(s.eps > 0 for s in ok.steps[:5])
