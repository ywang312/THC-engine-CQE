"""Tests of the M3 validation circuits (``thc_engine_cqe.ft.circuits``): phase-gradient excitations,
phase kickback, Givens decompositions and the compiled elementwise replay.

References independent of the module: M1's verified excitation circuits
(``diagnostics.cost.excitation_circuit`` / ``circuit_unitary``, checked against the Fock-space oracle in
tests/test_generator_rank.py), the defining identity of the phase-gradient state (evaluated with numpy),
random unitaries from QR decompositions, the exact CQE trajectory and FCI states (PySCF), sparse-matrix
exponentials (``scipy.sparse.linalg.expm_multiply``) of the excitation operators built on the
Jordan-Wigner Fock space, and PySCF's S^2 (``fci.spin_op``).
"""

import math

import numpy as np
import pytest
import scipy.sparse.linalg as spla
from pyscf.fci import spin_op

from thc_engine_cqe.cqe import run_cqe
from thc_engine_cqe.diagnostics.cost import Excitation, circuit_unitary, excitation_circuit
from thc_engine_cqe.fci_ops import FCISpace, Hamiltonian
from thc_engine_cqe.ft import circuits as cc
from thc_engine_cqe.ft import stateprep as sp
from thc_engine_cqe.ft import symmetry as sym
from thc_engine_cqe.testing.fock import FockSpace

N_QUBITS = 8                      # 4 spatial orbitals, blocked spin order


def random_excitations(kind: str, count: int, seed: int) -> list[Excitation]:
    """S_z-conserving excitations on 8 qubits, as spin-free generators produce them, random amplitudes."""
    rng = np.random.default_rng(seed)
    half = N_QUBITS // 2
    out = []
    while len(out) < count:
        if kind == "double":
            modes = sorted(int(m) for m in rng.choice(N_QUBITS, 4, replace=False))
            cre = tuple(sorted(int(m) for m in rng.choice(modes, 2, replace=False)))
            ann = tuple(sorted(set(modes) - set(cre)))
        else:
            c, x, y = (int(m) for m in rng.choice(N_QUBITS, 3, replace=False))
            cre, ann = tuple(sorted((c, x))), tuple(sorted((c, y)))
        if sum(m < half for m in cre) != sum(m < half for m in ann):
            continue
        if not cre > ann:
            cre, ann = ann, cre
        out.append(Excitation(kind, cre, ann, float(rng.uniform(-1.5, 1.5))))
    return out


def sector_labels(n_qubits: int) -> np.ndarray:
    """(n_alpha, n_beta) of every basis state, encoded as one integer."""
    idx = np.arange(1 << n_qubits)
    half = n_qubits // 2
    pop = np.array([bin(i & ((1 << half) - 1)).count("1") for i in idx])
    return 16 * pop + np.array([bin(i >> half).count("1") for i in idx])


def rounded_angle(theta: float, bits: int) -> float:
    return 2.0 * math.pi * round(2.0 ** bits * theta / (2.0 * math.pi)) / 2.0 ** bits


# ------------------------------------------------------------- phase-gradient excitations
@pytest.mark.parametrize("kind", ["double", "single"])
def test_pg_excitation_unitaries_match_verified_circuits(kind):
    """20 random S_z-conserving excitations of each kind on 8 qubits: the phase-gradient compilation
    (Clifford maps, AND cells, one addition of the cell constants) equals M1's verified excitation
    circuit to 1e-12 unrounded; rounded to b = 6, 10, 16 bits it errs by at most pi / 2^b and equals the
    verified circuit at the rounded Givens angle (so rounding only changes the angle); N and S_z blocks
    are exact (off-block entries <= 1e-13), rounded or not."""
    labels = sector_labels(N_QUBITS)
    off_block = labels[:, None] != labels[None, :]
    for exc in random_excitations(kind, 20, seed=17 if kind == "double" else 23):
        ref = circuit_unitary(excitation_circuit(exc), N_QUBITS)
        pg = cc.pg_excitation_unitary(exc, N_QUBITS, None)
        assert np.abs(pg - ref).max() < 1e-12
        assert np.abs(pg[off_block]).max() < 1e-13
        for bits in (6, 10, 16):
            pg = cc.pg_excitation_unitary(exc, N_QUBITS, bits)
            assert np.linalg.norm(pg - ref, 2) <= math.pi / 2.0 ** bits + 1e-12
            at = Excitation(kind, exc.creators, exc.annihilators,
                            math.copysign(rounded_angle(abs(exc.amplitude), bits), exc.amplitude))
            assert np.abs(pg - circuit_unitary(excitation_circuit(at), N_QUBITS)).max() < 1e-12
            assert np.abs(pg[off_block]).max() < 1e-13


def test_pg_complex_amplitudes_and_diagonal_terms():
    """A complex amplitude adds two rounded phase additions (bound 3 pi / 2^b), and a diagonal term
    exp(i phi n_a n_b) is one cell (bound pi / 2^b); both equal M1's circuits unrounded. Invalid inputs
    raise."""
    cases = [(Excitation("double", (5, 6), (1, 2), 0.3 * np.exp(0.7j)), 3),
             (Excitation("single", (3, 6), (3, 7), -0.8 * np.exp(-2.1j)), 3),
             (Excitation("diagonal", (1, 5), (1, 5), 0.37j), 1)]
    for exc, n_add in cases:
        ref = circuit_unitary(excitation_circuit(exc), N_QUBITS)
        assert np.abs(cc.pg_excitation_unitary(exc, N_QUBITS, None) - ref).max() < 1e-12
        for bits in (6, 10, 16):
            err = np.linalg.norm(cc.pg_excitation_unitary(exc, N_QUBITS, bits) - ref, 2)
            assert err <= n_add * math.pi / 2.0 ** bits + 1e-12
    with pytest.raises(ValueError):
        cc.pg_excitation_unitary(cases[0][0], 6, None)
    with pytest.raises(ValueError):
        cc.pg_excitation_unitary(cases[0][0], N_QUBITS, 0)
    with pytest.raises(ValueError):                         # the sign sigma = +1 needs sorted modes
        cc.pg_excitation_unitary(Excitation("double", (6, 5), (1, 2), 0.3), N_QUBITS, None)


# --------------------------------------------------------------------------- kickback
@pytest.mark.parametrize("bits", [3, 4, 5])
def test_kickback_with_explicit_registers(bits):
    """Loading cell constants by fan-out, adding them into |Phi_b> mod 2^b and unloading returns clean
    registers and the phase exp(+2 pi i v / 2^b) on every system basis state (error <= 1e-12), for random
    mutually exclusive cells on 12 - 2b system qubits, random and exactly representable constants. The
    sign rests on the identity (x -> x + v) |Phi_b> = exp(+2 pi i v / 2^b) |Phi_b>, checked here with numpy."""
    m = 2 ** bits
    grad = np.exp(-2j * np.pi * np.arange(m) / m) / math.sqrt(m)
    for v in range(m):
        assert np.allclose(np.roll(grad, v), np.exp(2j * np.pi * v / m) * grad, atol=1e-14)
    n_sys = 12 - 2 * bits
    rng = np.random.default_rng(bits)
    for trial in range(4):
        label = rng.integers(-1, 3, size=2 ** n_sys)              # -1: in no cell
        phases = rng.uniform(-math.pi, math.pi, 3) if trial < 2 else 2 * np.pi * rng.integers(0, m, 3) / m
        cells = [(label == c, float(phases[c])) for c in range(3)]
        assert cc.kickback_check(cells, n_sys, bits) < 1e-12
    with pytest.raises(ValueError):
        cc.kickback_check([(np.ones(2 ** n_sys, bool), 0.1), (np.ones(2 ** n_sys, bool), 0.2)], n_sys, bits)
    with pytest.raises(ValueError):
        cc.kickback_check([], n_sys + 1, bits)


# ---------------------------------------------------------------------- Givens rotations
@pytest.mark.parametrize("n", [4, 6])
@pytest.mark.parametrize("complex_", [False, True])
def test_givens_decomposition_reconstructs(n, complex_):
    """Random real orthogonal and complex unitary V (QR of Gaussian matrices) are reconstructed to 1e-12
    from exactly N(N-1)/2 adjacent Givens rotations and diagonal phases; real V gives phi = 0 and real
    +-1 phases, and the identity gives zero angles."""
    rng = np.random.default_rng(n + 10 * complex_)
    a = rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if complex_ else 0.0)
    v, _ = np.linalg.qr(a)
    rotations, phases = cc.givens_decomposition(v)
    assert len(rotations) == n * (n - 1) // 2
    assert all(q == p + 1 for p, q, _, _ in rotations)
    assert np.abs(cc.givens_reconstruct(rotations, phases) - v).max() < 1e-12
    assert np.allclose(np.abs(phases), 1.0, atol=1e-12)
    if not complex_:
        assert all(phi == 0.0 for *_, phi in rotations) and not np.iscomplexobj(phases)
        assert np.allclose(np.abs(phases), 1.0) and np.all(np.isin(np.round(phases), (-1.0, 1.0)))
    rotations, phases = cc.givens_decomposition(np.eye(n))
    assert all(theta == 0.0 for _, _, theta, _ in rotations) and np.all(phases == 1.0)
    with pytest.raises(ValueError):
        cc.givens_decomposition(2.0 * np.eye(n))


# -------------------------------------------------------------------------------- replay
_TRAJECTORIES: dict = {}


def trajectory(name: str, n_steps: int) -> dict:
    key = (name, n_steps)
    if key not in _TRAJECTORIES:
        act = sym.benchmark_system(name)
        e0, ci0 = act.fci()
        states = []
        res = run_cqe(act, max_iter=n_steps, e_exact=e0, keep_generators=True, optimizer="lbfgs",
                      callback=lambda st, ci: states.append(ci.copy()))
        steps = [s for s in res.steps if s.eps != 0.0]
        _TRAJECTORIES[key] = {"active": act, "irreps": sym.orbital_irreps(act), "e0": e0, "ci0": ci0,
                              "states": states, "d": [s.generator for s in steps],
                              "eps": [s.eps for s in steps]}
    return _TRAJECTORIES[key]


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(np.vdot(a.ravel(), b.ravel())) ** 2)


def expm_product(act, directions, eps, irreps, rounded: bool) -> np.ndarray:
    """CI vector of HF followed by expm_multiply of theta sign(t) (tau - tau^dag), tau built from the
    Fock-space annihilators, for every kept excitation in the order of stateprep.elementwise_excitations
    (theta = |eps t|, rounded to b bits if asked)."""
    n, nelec = act.norb, tuple(act.nelec)
    fock = FockSpace(n)
    a = [fock.a(m % n, m // n) for m in range(2 * n)]
    v = fock.from_ci(FCISpace.of(act).hf(), nelec)
    for d, e in zip(directions, eps):
        kept, b = sp.elementwise_excitations(d, e, irreps=irreps)
        for x in kept:
            (c0, c1), (d0, d1) = x.creators, x.annihilators
            tau = a[c0].T @ a[c1].T @ a[d1] @ a[d0]
            theta = rounded_angle(abs(x.amplitude), b) if rounded else abs(x.amplitude)
            v = spla.expm_multiply(math.copysign(theta, x.amplitude) * (tau - tau.T), v)
    return fock.to_ci(v, nelec)


def spin_weight(c: np.ndarray, norb: int, nelec: tuple[int, int], s: int) -> float:
    """||P_S c||^2, P_S the projector on total spin S in the M_S = 0 block: the product over the other
    S' of (S^2 - S'(S'+1)) / (S(S+1) - S'(S'+1)), with S^2 applied by PySCF's spin_op.contract_ss."""
    out = c
    for t in range(min(sum(nelec), 2 * norb - sum(nelec)) // 2 + 1):
        if t != s:
            out = (spin_op.contract_ss(out, norb, nelec) - t * (t + 1) * out) / (s * (s + 1) - t * (t + 1))
    return float(np.linalg.norm(out) ** 2)


@pytest.mark.parametrize("name", ["h4", "h6a"])
def test_replay_first_three_steps(name):
    """Compiled replay of the first three exact L-BFGS steps: the rounded and unrounded circuits differ by
    at most sum_j A_j pi / 2^{b_j}; every state keeps unit norm in its (n_alpha, n_beta) sector (no leakage,
    1e-12); the unrounded first step is within 5e-3 of the exact Psi_1 ([design]: 1.2e-3 to 2.0e-3; the
    elementwise product error); and the ground-state overlap after it is within 1e-3 of the exact one."""
    tr = trajectory(name, 3)
    act, ir = tr["active"], tr["irreps"]
    rounded = cc.replay_elementwise(act, tr["d"], tr["eps"], irreps=ir)
    exact_angles = cc.replay_elementwise(act, tr["d"], tr["eps"], irreps=ir, rounded=False)
    prep = sp.prep_cost(tr["d"], tr["eps"], irreps=ir)
    bound = np.cumsum([0.0] + [(s.n_double + s.n_single + s.n_diagonal) * math.pi / 2.0 ** s.bits
                               for s in prep.steps])
    assert len(rounded) == 4 and np.array_equal(rounded[0], exact_angles[0])
    for k in range(4):
        assert np.linalg.norm(rounded[k] - exact_angles[k]) <= bound[k] + 1e-12
        assert abs(np.linalg.norm(rounded[k]) - 1.0) < 1e-12
    assert np.linalg.norm(exact_angles[1] - tr["states"][0]) <= 5e-3
    assert abs(overlap(tr["ci0"], rounded[1]) - overlap(tr["ci0"], tr["states"][0])) <= 1e-3


def test_replay_equals_product_of_sparse_exponentials():
    """The replay's closed-form rotation v + sin(theta) K v + (1 - cos theta) K^2 v equals expm_multiply of
    the excitation operators eps (t tau - t tau^dag), built from the Fock-space annihilators, applied in
    the same order (the first two H4 steps, rounded and unrounded angles)."""
    tr = trajectory("h4", 2)
    act, ir = tr["active"], tr["irreps"]
    for rounded in (False, True):
        ref = expm_product(act, tr["d"], tr["eps"], ir, rounded)
        replay = cc.replay_elementwise(act, tr["d"], tr["eps"], irreps=ir, rounded=rounded)[-1]
        assert np.abs(ref - replay).max() < 1e-12


def test_compiled_replay_reaches_chemical_accuracy_at_k_star():
    """Open loop (the exact directions applied to the compiled states), the rounded elementwise circuits
    of the core systems reach 1.6 mHa at the same k* as the exact L-BFGS trajectory (3 / 6 / 3 / 2 for
    h4 / h6a / n2 / lih), with ground-state overlaps within 1e-3 of the exact ones at every k <= k*.
    Energies from the CI-space Hamiltonian (PySCF), which the Fock-space replay does not use."""
    for name, k_star in (("h4", 3), ("h6a", 6), ("n2", 3), ("lih", 2)):
        tr = trajectory(name, k_star)
        act = tr["active"]
        ham = Hamiltonian(act)
        replay = cc.replay_elementwise(act, tr["d"], tr["eps"], irreps=tr["irreps"])
        errors = [ham.energy(ci) - tr["e0"] for ci in replay]
        assert errors[k_star] <= 1.6e-3 < errors[k_star - 1]
        for k in range(1, k_star + 1):
            assert abs(overlap(tr["ci0"], replay[k]) - overlap(tr["ci0"], tr["states"][k - 1])) <= 1e-3


@pytest.mark.parametrize("name, error, p_compiled, s2, quintet", [
    ("h4", 1.1242e-3, 0.98013, 1.99e-6, 0.263), ("h6a", 1.3954e-3, 0.95745, 1.03e-6, 0.088),
    ("n2", 1.9785e-3, 0.96324, 1.47e-5, 0.626), ("lih", 2.1828e-4, 0.99399, 1.47e-10, 0.0),
    ("h8", 1.2661e-3, 0.94525, 1.17e-6, 0.121), ("h6s", 2.7407e-2, 0.87015, 1.545e-3, 0.343),
    ("n2s", 3.1296e-2, 0.22507, 3.925e-3, 0.668)])
def test_first_step_product_error_and_spin_contamination(name, error, p_compiled, s2, quintet):
    """The compiled RHF step on all seven systems (H8 too: testing.fock allows 8 orbitals). The replay equals
    expm_multiply of the kept excitations at the rounded angles (an independent reference) to 1e-12. Against
    the exact Psi_1 of run_cqe, the regression values of the circuits module docstring, which an independent
    replay (its own expansion, character mask and F3) reproduced [build check]: the state error (rel 1e-3), the
    ground-state overlap of the compiled state (1e-5), <S^2> from PySCF's spin_square (2%; 0 for the exact
    Psi_1) and the quintet share of ||Psi~_1 - Psi_1||^2 (projectors from PySCF's S^2; 2e-3; the rest is
    singlet, apart from 0.2% triplet for LiH). The stretched builds (h6s, n2s) have a product-formula error
    20 and 16 times that of the equilibrium build of the same molecule (h6a, n2), 27 and 31 times
    delta_step, and the product breaks S^2: the elementwise cost model budgets neither."""
    tr = trajectory(name, 1)
    act, ir = tr["active"], tr["irreps"]
    n, nelec = act.norb, tuple(act.nelec)
    replay = cc.replay_elementwise(act, tr["d"], tr["eps"], irreps=ir)
    assert np.abs(replay[1] - expm_product(act, tr["d"], tr["eps"], ir, rounded=True)).max() < 1e-12
    err = replay[1] - tr["states"][0]
    assert np.linalg.norm(err) == pytest.approx(error, rel=1e-3)
    assert overlap(tr["ci0"], replay[1]) == pytest.approx(p_compiled, abs=1e-5)
    assert abs(spin_op.spin_square(tr["states"][0], n, nelec)[0]) <= 1e-12
    assert spin_op.spin_square(replay[1], n, nelec)[0] == pytest.approx(s2, rel=2e-2)
    weights = [spin_weight(err, n, nelec, k) for k in range(min(sum(nelec), 2 * n - sum(nelec)) // 2 + 1)]
    assert sum(weights) == pytest.approx(np.linalg.norm(err) ** 2, rel=1e-9)
    assert weights[2] / sum(weights) == pytest.approx(quintet, abs=2e-3)
    assert weights[1] / sum(weights) <= (3e-3 if name == "lih" else 1e-3)


def test_replay_validation():
    """Mismatched inputs and complex directions (out of scope, F3) raise; no steps gives HF."""
    tr = trajectory("h4", 2)
    act = tr["active"]
    assert len(cc.replay_elementwise(act, [], [])) == 1
    with pytest.raises(ValueError):
        cc.replay_elementwise(act, tr["d"], tr["eps"][:1])
    with pytest.raises(ValueError):
        cc.replay_elementwise(act, [tr["d"][0] * (1 + 1e-3j)], [0.1])
