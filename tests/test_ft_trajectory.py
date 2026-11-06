"""Tests of ft.trajectory: exact CQE trajectories, overlaps, E1 and the classical first step (F16).

Reference trajectory numbers are the critic's full-precision values (synth-m3,
``traj_check_lbfgs.json``, spec Appendix A), computed with its own script on symmetric
builds made directly with ``systems.molecule`` and without the residual projection; the
spectrum references come from PySCF's symmetry-adapted FCI solver and dense
diagonalization; F16 is checked against JW Fock-space matrices (``testing.fock``).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg

from thc_engine_cqe.fci_ops import FCISpace, Hamiltonian, normalize, two_body
from thc_engine_cqe.ft import symmetry, trajectory
from thc_engine_cqe.optimizers import apply_metric
from thc_engine_cqe.residual import compute_residual, descent_generator
from thc_engine_cqe.testing.fock import FockSpace

# (E_{k+1} - E_0, p_{k+1}, ||A_k||) after L-BFGS step k (Appendix A; critic's JSON, full precision).
REFERENCE = {
    "h4": [(0.021504277878737987, 0.9802267643657638, 0.8584626920670166),
           (0.0038899458295142963, 0.9959988972047659, 0.6114978604421792),
           (0.0012029461227172078, 0.9981052824995057, 0.21573782206439898)],
    "h6a": [(0.03795612439536722, 0.9576347051805821, 1.0993212512524289),
            (0.012014066938412782, 0.9849616228138302, 0.7423906510797204),
            (0.005235538049297439, 0.9906911858991058, 0.4218510424421071),
            (0.0029912099755398103, 0.9950295255905398, 0.23967334614588914),
            (0.0018766310075433168, 0.9974391285918158, 0.20771935449463547),
            (0.001110740484043582, 0.9987186020371953, 0.1526546835367351)],
    "n2": [(0.03520878551135809, 0.9635000865170638, 1.8954622654606657),
           (0.01467844041741273, 0.9813004202637063, 0.6637845296143801),
           (0.0011465443159579536, 0.9993742385694582, 0.403824789764888)],
}
K_STAR = {"h4": 3, "h6a": 6, "n2": 3}                       # Appendix A: 1.6 mHa after 3 / 6 / 3 steps
P_HF = {"h4": 0.9455, "h6a": 0.9026, "n2": 0.9257, "lih": 0.9743}   # Appendix A, 4 digits
GAPS = {"h4": (0.5657, 0.5633), "h6a": (0.3909, 0.3909), "n2": (0.7115, 0.4117),
        "lih": (0.1332, 0.1332)}                            # E1 - E0 same irrep / any irrep, 4 digits
RATIOS = {"h4": 1.0159, "h6a": 1.0220, "n2": 0.9805, "lih": 1.0192}  # eps2 / eps*, Appendix A

_CACHE: dict = {}


def _system(name: str):
    if name not in _CACHE:
        a = symmetry.benchmark_system(name)
        _CACHE[name] = (a, symmetry.orbital_irreps(a))
    return _CACHE[name]


def _record(name: str, **kw) -> trajectory.TrajectoryRecord:
    key = (name, tuple(sorted(kw.items())))
    if key not in _CACHE:
        a, ir = _system(name)
        _CACHE[key] = trajectory.exact_trajectory(a, system=name, irreps=ir, **kw)
    return _CACHE[key]


def _check_against_reference(name: str) -> None:
    rec = _record(name)
    rows = REFERENCE[name]
    for k, (err, p, rnorm) in enumerate(rows):
        assert abs(rec.errors[k + 1] - err) <= 1e-6
        assert abs(rec.overlaps[k + 1] - p) <= 1e-6
        assert abs(rec.residual_norms[k] - rnorm) <= 1e-6
    assert trajectory.first_iterate_below(rec, 1.6e-3) == K_STAR[name]


def test_h4_trajectory_reproduces_appendix_a():
    """h4 L-BFGS: E_k - E_0 and p_k after steps 0-2 to 1e-6, ||A_k|| to 1e-6, k* = 3.

    The reference is the critic's independent run (unmasked residual, its own build call),
    so the agreement also shows that the symmetry projection leaves the trajectory unchanged.
    """
    _check_against_reference("h4")


@pytest.mark.slow
@pytest.mark.parametrize("name", ["h6a", "n2"])
def test_h6a_n2_trajectories_reproduce_appendix_a(name):
    """h6a (k* = 6) and n2 (k* = 3) against the critic's full-precision values (as for h4).

    n2 was built before ``systems.canonical_orbitals`` fixed the pi orientation; energies and
    overlaps are invariant under that orbital rotation, so the numbers must still agree.
    """
    _check_against_reference(name)


def test_record_structure_and_states():
    """Lengths, normalization, and E_k / sigma_k^2 / ||A_k|| recomputed from the stored states.

    The recomputation (``residual.compute_residual`` on Psi_k) is independent of how the
    driver recorded the step; p_0 is checked against the HF weight of the ground state of the
    dense CI Hamiltonian and against Appendix A.
    """
    rec = _record("h4")
    a, ir = _system("h4")
    K = rec.n_steps
    assert len(rec.states) == len(rec.energies) == len(rec.variances) == len(rec.overlaps) == K + 1
    assert len(rec.directions) == len(rec.eps) == K and K >= 3
    assert rec.optimizer == "lbfgs" and rec.metric == "spin_free" and rec.system == "h4"
    for k, psi in enumerate(rec.states):
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-12
        rd = compute_residual(a, psi)
        assert abs(rd.energy - rec.energies[k]) < 1e-10
        assert abs(rd.variance - rec.variances[k]) < 1e-10
        assert abs(rd.residual_norm - rec.residual_norms[k]) < 1e-8
    ham = Hamiltonian(a)
    sp = ham.space
    dense = np.column_stack([ham.apply(np.eye(sp.size)[i].reshape(sp.shape)).ravel() for i in range(sp.size)])
    w, v = np.linalg.eigh(0.5 * (dense + dense.T))
    assert abs(w[0] + a.ecore - rec.e0) < 1e-10
    assert abs(v[0, 0] ** 2 - rec.overlaps[0]) < 1e-10
    assert abs(rec.overlaps[0] - P_HF["h4"]) < 5e-5
    assert rec.irrep_impurity < 1e-12 and rec.ground_irrep == 0 and rec.flags == ()


def test_first_iterate_below():
    """first_iterate_below against a direct scan of E_k - E_0, including the never-reached case."""
    rec = _record("h4")
    for thr in (1.0, 0.03, 0.004, 1.6e-3, 1e-12):
        expected = next((k for k, e in enumerate(rec.energies - rec.e0) if e <= thr), None)
        assert trajectory.first_iterate_below(rec, thr) == expected
    assert trajectory.first_iterate_below(rec, -1.0) is None


def _symmetric_fci_gap(a, irreps, wfnsym: int) -> float:
    """E1 - E0 of the singlets of irrep ``wfnsym`` by PySCF's symmetry-adapted solver."""
    from pyscf import fci

    solver = fci.addons.fix_spin_(fci.direct_spin1_symm.FCI(a.mol), shift=1.0, ss=0.0)
    solver.conv_tol = 1e-12
    solver.nroots = 3
    solver.wfnsym = int(wfnsym)
    e, _ = solver.kernel(a.h1, a.eri, a.norb, a.nelec, ecore=a.ecore, orbsym=np.asarray(irreps))
    return float(e[1] - e[0])


def _dense_singlet_gap(a) -> float:
    """E1 - E0 among singlets of the dense (n_alpha, n_beta) CI Hamiltonian (any irrep)."""
    from pyscf.fci import spin_op

    ham = Hamiltonian(a)
    sp = ham.space
    dense = np.column_stack([ham.apply(np.eye(sp.size)[i].reshape(sp.shape)).ravel() for i in range(sp.size)])
    w, v = np.linalg.eigh(0.5 * (dense + dense.T))
    singlets = [w[j] for j in range(min(sp.size, 40))
                if abs(spin_op.spin_square0(v[:, j].reshape(sp.shape), a.norb, a.nelec)[0]) < 1e-6]
    return float(singlets[1] - singlets[0])


@pytest.mark.parametrize("name", ["h4", "h6a", "n2", "lih"])
def test_e1_same_and_any_irrep(name):
    """E1 (same irrep) and E1_any against PySCF's symmetry-adapted FCI and dense diagonalization.

    The same-irrep reference is ``direct_spin1_symm`` in the irrep of psi_0 (a different
    solver, which never mixes irreps); E1_any is the second singlet of the dense CI matrix.
    Both agree to 1e-8 Ha and with Appendix A to its 4 printed digits. The N2 value
    (0.7115 against 0.4117) is the S15 correction.
    """
    a, ir = _system(name)
    rec = trajectory.exact_trajectory(a, system=name, irreps=ir, max_iter=0)
    assert rec.n_steps == 0 and len(rec.states) == 1
    assert abs((rec.e1 - rec.e0) - _symmetric_fci_gap(a, ir, rec.ground_irrep)) < 1e-8
    assert abs((rec.e1_any - rec.e0) - _dense_singlet_gap(a)) < 1e-8
    same, any_ = GAPS[name]
    assert abs(rec.e1 - rec.e0 - same) < 5e-5 and abs(rec.e1_any - rec.e0 - any_) < 5e-5


def test_e1_degenerate_roots_are_tested_as_a_subspace():
    """A same-irrep root hidden in a degenerate mixture is still found (module docstring).

    Hand-made 3 x 3 CI arrays with determinant irreps a XOR b: a triple of degenerate roots
    spans pure states of irreps 0, 1 and 2, rotated so that no root has dominant irrep 0
    (weights 1/3, 1/2, 1/6 and 1/3, 0, 2/3). The dominant-irrep rule would skip the triple;
    the subspace test finds irrep 0 there, irrep 3 at the next root and no irrep 5. In a
    near-degenerate group (0, 4e-7, 8e-7 Ha) whose only irrep-0 root is the top one, E1 is
    the group's lowest energy, the lower bound of the module docstring.
    """
    det_ir = np.arange(3)[:, None] ^ np.arange(3)[None, :]

    def unit(i, j):
        u = np.zeros((3, 3))
        u[i, j] = 1.0
        return u

    pure = [unit(1, 1), unit(0, 1), unit(0, 2)]          # irreps 0, 1, 2
    q = np.array([[1 / np.sqrt(3), 1 / np.sqrt(2), 1 / np.sqrt(6)],
                  [1 / np.sqrt(3), -1 / np.sqrt(2), 1 / np.sqrt(6)],
                  [1 / np.sqrt(3), 0.0, -2 / np.sqrt(6)]])
    triple = [sum(q[r, i] * pure[i] for i in range(3)) for r in range(3)]
    roots = [unit(0, 0)] + triple + [unit(1, 2)]          # irreps 0, mixed triple, 3
    energies = np.array([-1.0, 0.0, 0.0, 0.0, 0.5])
    dominant = [int(np.argmax(np.bincount(det_ir.ravel(), weights=(r ** 2).ravel(), minlength=4)))
                for r in triple]
    assert 0 not in dominant
    assert trajectory._same_irrep_e1(energies, roots, det_ir, 0) == 0.0
    assert trajectory._same_irrep_e1(energies, roots, det_ir, 3) == 0.5
    assert trajectory._same_irrep_e1(energies, roots, det_ir, 5) is None
    near = [unit(0, 0), unit(0, 1), unit(0, 2), unit(1, 1)]          # irreps 0 | 1, 2, 0
    assert trajectory._same_irrep_e1(np.array([-1.0, 0.0, 4e-7, 8e-7]), near, det_ir, 0) == 0.0
    assert trajectory._same_irrep_e1(np.array([-1.0, 0.0, 4e-7, 2e-6]), near, det_ir, 0) == 2e-6


def test_symmetry_projection_removes_irrep_leakage_on_n2s():
    """n2s, 20 L-BFGS steps: the projected residual keeps Psi_k in the HF irrep to round-off.

    Without the projection, exact L-BFGS amplifies the round-off of the forbidden entries
    (ft.symmetry: weight outside the HF irrep up to 2e-4); the purity is measured from the
    determinant irreps of the stored states, independently of the driver.
    """
    masked = _record("n2s", n_roots=2)
    plain = _record("n2s", n_roots=2, mask_residual=False)
    assert masked.irrep_impurity < 1e-12
    assert plain.irrep_impurity > 1e-6
    a, ir = _system("n2s")
    for psi in masked.states:
        assert symmetry.ci_irrep(psi, a.norb, a.nelec, ir)[1] > 1.0 - 1e-12
    np.testing.assert_allclose(masked.energies[:6], plain.energies[:6], atol=1e-8)


def test_spin_orbital_metric_trajectory():
    """metric="spin_orbital" reaches the driver: the first direction is M conj(A_0), recorded as such.

    The reference direction is ``optimizers.apply_metric`` of the HF descent generator,
    recomputed here; the energies decrease monotonically (exact line search).
    """
    a, ir = _system("h4")
    rec = trajectory.exact_trajectory(a, system="h4", irreps=ir, n_roots=3, max_iter=4, metric="spin_orbital")
    assert rec.metric == "spin_orbital" and rec.n_steps == 4
    rd = compute_residual(a, FCISpace.of(a).hf())
    np.testing.assert_allclose(rec.directions[0], apply_metric(descent_generator(rd.A), "spin_orbital"),
                               atol=1e-14)
    assert np.all(np.diff(rec.energies) < 0)


def _fock_second_derivative(a, d: np.ndarray, ci: np.ndarray) -> tuple[float, float]:
    """<[[H, G], G]> and a dense-expm central difference of E(eps), in the JW Fock space."""
    fs = FockSpace(a.norb)
    H = fs.hamiltonian(a.h1, a.eri, a.ecore).toarray()
    G = fs.two_body(d).toarray()
    psi = fs.from_ci(ci, a.nelec)
    comm = H @ G - G @ H
    exact = float(np.real(np.vdot(psi, (comm @ G - G @ comm) @ psi)))

    def energy(eps: float) -> float:
        v = scipy.linalg.expm(eps * G) @ psi
        return float(np.real(np.vdot(v, H @ v)))

    h = 1e-3
    fd = (energy(h) - 2 * energy(0.0) + energy(-h)) / h ** 2
    return exact, fd


@pytest.mark.parametrize("name", ["h4", "lih"])
def test_second_order_step(name):
    """F16: eps2/eps* within 3% (Appendix A ratio to 1e-3), E''(0) against finite differences to 1e-6.

    h4: E''(0) and a central difference are recomputed from dense JW Fock-space matrices
    (``testing.fock``, no PySCF CI code). lih: against the module's Richardson difference and
    the energy loss E(eps2) - E(eps*) of the spec (<= 3.4e-5 Ha). eps* is the trajectory's
    first exact line-search step.
    """
    a, ir = _system(name)
    out = trajectory.second_order_step(a)
    assert out["flags"] == () and out["lowering"] == pytest.approx(out["slope"] ** 2 / (2 * out["d2e"]), rel=1e-14)
    assert abs(out["ratio"] - 1.0) <= 0.03
    assert abs(out["ratio"] - RATIOS[name]) <= 1e-3
    assert abs(out["d2e"] - out["d2e_fd"]) <= 1e-6 * abs(out["d2e"])
    assert 0.0 <= out["energy_lost"] <= 3.4e-5
    rd = compute_residual(a, FCISpace.of(a).hf())
    assert abs(out["slope"] + float(np.sum(rd.A ** 2))) < 1e-12
    assert abs(out["eps2"] - (-out["slope"] / out["d2e"])) < 1e-15
    if name == "h4":
        exact, fd = _fock_second_derivative(a, out["direction"], FCISpace.of(a).hf())
        assert abs(out["d2e"] - exact) <= 1e-10 * abs(exact)
        assert abs(out["d2e"] - fd) <= 1e-5 * abs(exact)
        rec = _record("h4")
        assert abs(out["eps_exact"] - rec.eps[0]) <= 1e-12
        masked = trajectory.second_order_step(a, irreps=ir)
        assert masked["eps_exact"] == rec.eps[0] and abs(masked["eps2"] - out["eps2"]) <= 1e-14 * out["eps2"]
    ham = Hamiltonian(a)
    assert abs(ham.energy(out["state"]) - out["energy_eps2"]) < 1e-12
    assert abs(np.linalg.norm(out["state"]) - 1.0) < 1e-12


@pytest.mark.slow
@pytest.mark.parametrize("name", ["h6a", "n2"])
def test_second_order_step_slow(name):
    """F16 on h6a and n2: ratio within 3% (Appendix A to 1e-3), E''(0) against the Richardson difference."""
    a, _ = _system(name)
    out = trajectory.second_order_step(a)
    assert abs(out["ratio"] - 1.0) <= 0.03
    assert abs(out["ratio"] - RATIOS[name]) <= 1e-3
    assert abs(out["d2e"] - out["d2e_fd"]) <= 1e-6 * abs(out["d2e"])
    assert 0.0 <= out["energy_lost"] <= 3.4e-5


def test_second_order_step_metric_and_state():
    """With metric="spin_orbital" the slope is -<conj(A), M conj(A)> and the step uses that direction.

    The slope is recomputed from apply_metric; the state is exp(eps2 G) HF, checked by a
    dense expm of the CI-space generator.
    """
    a, _ = _system("h4")
    out = trajectory.second_order_step(a, metric="spin_orbital")
    rd = compute_residual(a, FCISpace.of(a).hf())
    g = descent_generator(rd.A)
    m = apply_metric(g, "spin_orbital")
    assert abs(out["slope"] + float(np.real(np.sum(m * rd.A)))) < 1e-12
    assert abs(out["ratio"] - 1.0) <= 0.05
    sp = FCISpace.of(a)
    G = np.column_stack([two_body(sp, m, np.eye(sp.size)[i].reshape(sp.shape)).ravel() for i in range(sp.size)])
    ref = normalize(scipy.linalg.expm(out["eps2"] * G) @ sp.hf().ravel())
    assert abs(abs(np.vdot(ref, out["state"].ravel())) - 1.0) < 1e-12


def test_second_order_step_masks_direction_and_line_search():
    """With irreps, d = conj(mask(A)) and eps* is the exact line search along that d, also away from HF.

    At HF the forbidden entries of A are round-off, so the state here is HF plus a random
    admixture (HF-irrep weight 0.969), where masking changes the step by a factor 2. The
    direction is recomputed from compute_residual and mask_pair_tensor; eps* is the minimum
    of <psi|exp(eps G)^T H exp(eps G)|psi> along the returned direction, found by a bounded
    scalar minimizer with dense JW Fock-space matrices (testing.fock), to 1e-6 (relative).
    """
    import scipy.optimize

    a, ir = _system("h4")
    sp = FCISpace.of(a)
    rng = np.random.default_rng(11)
    psi = normalize(sp.hf() + 0.3 * rng.standard_normal(sp.shape) / np.sqrt(sp.size))
    out = trajectory.second_order_step(a, psi, irreps=ir)
    plain = trajectory.second_order_step(a, psi)
    A = compute_residual(a, psi).A
    masked = symmetry.mask_pair_tensor(A, ir)
    assert np.abs(A - masked).max() > 1e-2
    np.testing.assert_allclose(out["direction"], descent_generator(masked), rtol=0.0, atol=1e-14)
    fs = FockSpace(a.norb)
    H = fs.hamiltonian(a.h1, a.eri, a.ecore).toarray()
    v = fs.from_ci(psi, a.nelec)
    G = fs.two_body(out["direction"]).toarray()

    def energy(eps: float) -> float:
        x = scipy.linalg.expm(eps * G) @ v
        return float(np.real(np.vdot(x, H @ x)))

    ref = scipy.optimize.minimize_scalar(energy, bounds=(0.3 * out["eps2"], 3.0 * out["eps2"]), method="bounded",
                                         options={"xatol": 1e-12}).x
    assert abs(out["eps_exact"] / ref - 1.0) <= 1e-6
    assert abs(plain["eps_exact"] / out["eps_exact"] - 1.0) > 0.1


def test_second_order_step_at_a_converged_state():
    """At the FCI ground state the direction is solver noise: eps2, eps*, ratio are nan, flag "converged_state".

    The predicted lowering E'(0)^2 / (2 E''(0)) is far below 1e-10 Ha there (the exact line
    search then returns round-off; before this rule the ratio came out as 0.49), while HF
    is unflagged.
    """
    a, _ = _system("h4")
    _, ci = a.fci()
    out = trajectory.second_order_step(a, np.asarray(ci))
    assert out["flags"] == ("converged_state",) and out["lowering"] < 1e-10
    for key in ("eps2", "eps_exact", "ratio", "d2e_fd", "energy_eps2", "energy_exact", "energy_lost"):
        assert np.isnan(out[key])
    assert out["state"] is None and np.isfinite(out["d2e"])


def test_second_order_first_step_record():
    """first_step="second_order": row 1 is the classical step, recomputed in the JW Fock space.

    E'(0) = <[H, G]> and E''(0) = <[[H, G], G]> at HF come from dense Fock-space matrices
    (testing.fock), eps_2 = -E'(0)/E''(0), Psi_1 = expm(eps_2 G) HF, and E_1, sigma_1^2, p_1
    from the dense H and the dense ground state (p_1 to 1e-8, the accuracy of the
    record's Davidson psi_0); none of it uses fci_ops. The values are
    those of the reviewer's run (E_1 - E_0 = 21.515 mHa, p_1 = 0.98039, against 21.504 mHa
    and 0.98023 after the exact step). L-BFGS rows k >= 2 and steps j >= 1 are the exact
    trajectory's (flag), and d_0 is the exact run's first direction bit for bit (the same
    masked residual and metric; without the mask it would differ by the round-off of the
    forbidden entries, <= 7.5e-16 here).
    """
    a, ir = _system("h4")
    exact = _record("h4")
    rec = trajectory.exact_trajectory(a, system="h4", irreps=ir, n_roots=3, max_iter=5, first_step="second_order")
    assert rec.first_step == "second_order" and rec.flags == ("steps_from_exact_trajectory",)
    assert rec.n_steps == 5 and len(rec.states) == 6
    assert np.array_equal(rec.directions[0], exact.directions[0])
    fs = FockSpace(a.norb)
    H = fs.hamiltonian(a.h1, a.eri, a.ecore).toarray()
    G = fs.two_body(rec.directions[0]).toarray()
    hf = fs.from_ci(FCISpace.of(a).hf(), a.nelec)
    comm = H @ G - G @ H
    eps2 = -float(np.real(np.vdot(hf, comm @ hf))) / float(np.real(np.vdot(hf, (comm @ G - G @ comm) @ hf)))
    assert abs(rec.eps[0] - eps2) <= 1e-12 * eps2
    psi1 = scipy.linalg.expm(eps2 * G) @ hf
    assert abs(abs(np.vdot(psi1, fs.from_ci(rec.states[1], a.nelec))) - 1.0) < 1e-12
    e1 = float(np.real(np.vdot(psi1, H @ psi1)))
    assert abs(rec.energies[1] - e1) < 1e-10
    assert abs(rec.variances[1] - (float(np.real(np.vdot(H @ psi1, H @ psi1))) - e1 ** 2)) < 1e-9
    ham = Hamiltonian(a)
    sp = ham.space
    dense = np.column_stack([ham.apply(np.eye(sp.size)[i].reshape(sp.shape)).ravel() for i in range(sp.size)])
    _, v = np.linalg.eigh(0.5 * (dense + dense.T))
    psi0 = fs.from_ci(v[:, 0].reshape(sp.shape), a.nelec)
    # psi_0 of the record is PySCF's Davidson root, whose overlaps agree with the dense
    # eigenvector to a few 1e-9 (3.0e-9 here).
    assert abs(rec.overlaps[1] - abs(np.vdot(psi0, psi1)) ** 2) < 1e-8
    assert abs(rec.errors[1] - 21.515e-3) < 1e-6 and abs(rec.overlaps[1] - 0.98039) < 1e-5
    assert abs(rec.residual_norms[1] - compute_residual(a, rec.states[1]).residual_norm) < 1e-12
    for k in range(2, 6):
        assert np.array_equal(rec.states[k], exact.states[k]) and rec.energies[k] == exact.energies[k]
    np.testing.assert_array_equal(rec.eps[1:], exact.eps[1:5])
    with pytest.raises(ValueError):
        trajectory.exact_trajectory(a, irreps=ir, n_roots=2, max_iter=1, first_step="classical")


def test_second_order_first_step_sd_is_a_continuation():
    """With the memoryless "sd" the later steps are run_cqe from the classical Psi_1 itself (no flag).

    The reference reruns run_cqe from the record's Psi_1 with the same mask; steps, energies
    and states must agree.
    """
    from thc_engine_cqe.cqe import run_cqe

    a, ir = _system("h4")
    rec = trajectory.exact_trajectory(a, irreps=ir, n_roots=3, max_iter=3, optimizer="sd", first_step="second_order")
    assert rec.flags == () and rec.n_steps == 3 and rec.optimizer == "sd"
    so = trajectory.second_order_step(a, irreps=ir)
    assert rec.eps[0] == so["eps2"] and abs(abs(np.vdot(rec.states[1], so["state"])) - 1.0) < 1e-14
    states = []
    res = run_cqe(a, ci0=rec.states[1], max_iter=2, optimizer="sd", keep_generators=True,
                  residual_estimator=lambda rd, ci, it: symmetry.mask_pair_tensor(rd.A, ir),
                  callback=lambda st, ci: states.append(normalize(np.asarray(ci))))
    np.testing.assert_allclose(rec.eps[1:], [s.eps for s in res.steps], rtol=1e-12)
    np.testing.assert_allclose(rec.energies[1:3], [s.energy for s in res.steps], atol=1e-12)
    for k in (2, 3):
        assert abs(abs(np.vdot(rec.states[k], states[k - 2])) - 1.0) < 1e-12
