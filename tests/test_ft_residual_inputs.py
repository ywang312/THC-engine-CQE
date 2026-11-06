"""Tests of the residual-cost inputs (``thc_engine_cqe.ft.residual_inputs``) built from the M2 layer.

References independent of the module: the symmetry-blind full residual tensor of
``residual.compute_residual`` (it knows neither element sets nor multiplicities), dense
diagonalization of the CI Hamiltonian for the exact ground state, the Pauli coefficients
and dense matrices of the M2 targets, ``scipy.linalg.expm`` on the qubit space, dense
Pauli-LCU block encodings and Hadamard tests built here from the Pauli terms, and the
independently computed numbers of the M3 specification (Appendix A, the critic's
``var_check.py``) and of memo 2.
"""

import math

import numpy as np
import pytest
import scipy.linalg

from thc_engine_cqe import systems
from thc_engine_cqe.cqe import run_cqe
from thc_engine_cqe.factor import double_factorize
from thc_engine_cqe.fci_ops import FCISpace, Hamiltonian
from thc_engine_cqe.ft import residual_inputs as ri
from thc_engine_cqe.ft import symmetry as sy
from thc_engine_cqe.ft.primitives import FTParameters
from thc_engine_cqe.measure import routes, schemes, targets
from thc_engine_cqe.measure.pauli import PauliSum, QubitMapping, QubitState, jw_hamiltonian
from thc_engine_cqe.residual import compute_residual


@pytest.fixture(scope="module")
def h4():
    """Symmetric H4 (1.8 bohr), its irreps, the exact SD iterates Psi_1..Psi_5, the allowed spin-free
    elements, their normalizations and their variances at iterate 1 (computed once, as the pipeline
    does)."""
    act = sy.benchmark_system("h4")
    ir = sy.orbital_irreps(act)
    states = []
    run_cqe(act, max_iter=5, optimizer="sd", callback=lambda st, ci: states.append(ci.copy()))
    el, _ = ri.allowed_elements(act, "spin_free", ir)
    return {"act": act, "irreps": ir, "states": states, "el": el,
            "norms": ri.element_normalizations(act, el, "spin_free"),
            "var": ri.element_variances(act, states[0], el, "spin_free")}


def _m(act, el, kind="spin_free"):
    return targets.element_multiplicities(el, kind, act.norb).astype(float)


def _ground_state(act) -> np.ndarray:
    """Lowest eigenvector of the CI Hamiltonian by dense diagonalization (machine precision)."""
    space = FCISpace(act.norb, act.nelec)
    ham = Hamiltonian(act)
    eye = np.eye(space.size)
    hmat = np.column_stack([ham.apply(eye[i].reshape(space.shape)).ravel() for i in range(space.size)])
    w, v = np.linalg.eigh(0.5 * (hmat + hmat.T))
    return v[:, 0].reshape(space.shape)


# ----------------------------------------------------------------------- elements
def test_allowed_elements_follow_the_irreps(h4):
    """allowed_elements keeps the rows of targets.element_set whose irrep product is totally symmetric
    (Appendix A: 28 of 60 spin-free and 70 of 150 spin-orbital H4 elements), the forbidden ones vanish
    in the symmetry-blind residual of the CQE iterates, and irreps=None keeps everything with a warning."""
    act, ir = h4["act"], h4["irreps"]
    for kind, (n_allowed, n_all) in (("spin_free", (28, 60)), ("spin_orbital", (70, 150))):
        el, mask = ri.allowed_elements(act, kind, ir)
        full = targets.element_set(act.norb, kind)
        assert (len(el), mask.size, int(mask.sum())) == (n_allowed, n_all, n_allowed)
        assert np.array_equal(el, full[mask])
        for ci in h4["states"]:
            a = routes.exact_residual(act, ci, full, kind)
            assert np.max(np.abs(a[~mask])) <= 1e-10 * np.max(np.abs(a))
    with pytest.warns(UserWarning, match="every element is kept"):
        el, mask = ri.allowed_elements(act, "spin_free")
    assert mask.all() and len(el) == 60


# ----------------------------------------------------------------------- normalizations
def test_lambda_q_is_one_half_for_every_spin_orbital_element(h4):
    """Every spin-orbital Q_e = (Gamma_e - Gamma_e^dagger)/(2i) has Pauli 1-norm 1/2 (16 strings of
    modulus 1/16 in Gamma_e, 8 kept; or 8 of 1/8 with a repeated mode, 4 kept), on all 150 H4 elements."""
    act = h4["act"]
    el = targets.element_set(act.norb, "spin_orbital")
    norms = ri.element_normalizations(act, el, "spin_orbital")
    assert np.allclose(norms["lam_q"], 0.5, rtol=0, atol=1e-14)


def test_alpha_x_is_the_pauli_one_norm_and_bounds_the_sector_norm(h4):
    """alpha_X and L_X are the 1-norm and the number of non-identity Pauli terms of direct_targets'
    X_e (from their coefficients), lambda_Q that of aux_targets' Q_e; neither depends on the qubit
    order; and ||X_e|| on the (2, 2) sector, from the dense matrix, is at most alpha_X (here well
    below it: the 1-norm is loose, specification section 11)."""
    act, el, norms = h4["act"], h4["el"], h4["norms"]
    mapping = QubitMapping(act.norb)
    xs = targets.direct_targets(el, act, mapping, "spin_free")
    qs = targets.aux_targets(el, mapping, "spin_free")
    for e, (x, q) in enumerate(zip(xs, qs)):
        keep = (x.xs | x.zs) != 0
        assert norms["alpha_x"][e] == pytest.approx(np.abs(x.coeffs[keep]).sum(), rel=1e-14)
        assert norms["n_terms_x"][e] == keep.sum()
        assert norms["lam_q"][e] == pytest.approx(np.abs(q.coeffs[(q.xs | q.zs) != 0]).sum(), rel=1e-14)
    inter = ri.element_normalizations(act, el, "spin_free", QubitMapping(act.norb, "interleaved"))
    assert inter["order"] == "interleaved"
    for key in ("lam_q", "alpha_x"):
        assert np.allclose(inter[key], norms[key], rtol=1e-13, atol=0)
    assert np.array_equal(inter["n_terms_x"], norms["n_terms_x"])
    basis = np.arange(1 << mapping.n_qubits)
    n_alpha = sum((basis >> mapping.qubit(p, 0)) & 1 for p in range(act.norb))
    n_beta = sum((basis >> mapping.qubit(p, 1)) & 1 for p in range(act.norb))
    sector = basis[(n_alpha == act.nelec[0]) & (n_beta == act.nelec[1])]
    ratios = np.array([np.linalg.norm(x.to_dense()[np.ix_(sector, sector)], 2) for x in xs]) / norms["alpha_x"]
    assert np.all(ratios > 0) and np.all(ratios <= 1 + 1e-12) and ratios.max() < 0.9


# ----------------------------------------------------------------------- variances
def test_element_variances_exact_branch(h4):
    """N <= 4: exact matchgate pair sums (no standard errors); w_aux is 0.01 x the aux-route variance at
    delta = 0.1. V_m = sum m v at iterate 1 agrees within 1% with the critic's independent value at SD
    iterate 2 (Appendix A: 1220) and W_m with memo 2's 816 (iterate 5): the variances barely move."""
    act, el, var = h4["act"], h4["el"], h4["var"]
    m = _m(act, el)
    assert var["method"] == "exact" and var["n_settings"] is None and var["delta_aux"] == 0.1
    assert np.all(var["stderr"] == 0) and var["stderr_total"] == 0.0
    av = routes.aux_residual_variance(act, h4["states"][0], 0.1, "exact", el, "spin_free", "matchgate_shadow")
    assert np.allclose(var["w_aux"], 0.01 * av.per_target, rtol=1e-14, atol=0)
    assert np.sum(m * var["v_direct"]) == pytest.approx(1220, rel=0.01)
    assert np.sum(m * var["w_aux"]) == pytest.approx(816, rel=0.01)


def test_element_variances_rb_branch():
    """N > 4: Rao-Blackwellized matchgate variances with params.n_settings and params.seed, with
    standard errors; at 200 settings V_m of H6 (1.0 A) agrees with the critic's 5000-setting value at
    SD iterate 2 (Appendix A: 2.515e4) within two standard errors, the standard error of V_m bounded
    by sum_e m_e sigma_e (valid for any correlation between the elements' estimates; observed
    V_m = 24465, bound 2714)."""
    act = sy.benchmark_system("h6a")
    states = []
    run_cqe(act, max_iter=1, optimizer="sd", callback=lambda st, ci: states.append(ci.copy()))
    el, _ = ri.allowed_elements(act, "spin_free", sy.orbital_irreps(act))
    var = ri.element_variances(act, states[0], el, "spin_free", params=FTParameters(n_settings=200, seed=3))
    assert (var["method"], var["n_settings"], var["seed"]) == ("rb", 200, 3)
    assert np.all(np.isfinite(var["stderr"])) and 0 < var["stderr_total"] < math.inf
    m = _m(act, el)
    assert abs(np.sum(m * var["v_direct"]) - 2.515e4) <= 2 * (np.sum(m * var["stderr"]) + 33.0)


# ----------------------------------------------------------------------- biases
def test_aux_bias_curve_is_quadratic_and_matches_the_route(h4):
    """||b(delta)||_m of the exact aux route is the weighted norm of routes.aux_residual - exact_residual,
    and grows as delta^2 at small delta (log-log slopes in [1.8, 2.2] for delta <= 0.1, also relative to
    ||A||_m; M2: b = (delta^2/6) <ad_H^3(Gamma)> + O(delta^4))."""
    act, el = h4["act"], h4["el"]
    ci = h4["states"][1]
    m = _m(act, el)
    deltas = np.array([0.0125, 0.025, 0.05, 0.1])
    curve = ri.aux_bias_curve(act, ci, el, "spin_free", deltas)
    a = routes.exact_residual(act, ci, el, "spin_free")
    for d, b in zip(deltas, curve):
        diff = routes.aux_residual(act, ci, d, "exact", el, "spin_free") - a
        assert b == pytest.approx(math.sqrt(np.sum(m * diff ** 2)), rel=1e-13)
    rel = curve / math.sqrt(np.sum(m * a ** 2))
    slopes = np.diff(np.log(rel)) / np.diff(np.log(deltas))
    assert np.all((slopes >= 1.8) & (slopes <= 2.2))


def test_factorization_bias_at_the_ground_state_and_linearity(h4):
    """At the exact ground state Psi_0 of H (dense diagonalization) A[H](Psi_0) = 0, so the
    factorization bias is ||A[H'](Psi_0)||_m, here equal to the Frobenius norm of the full tensor of
    residual.compute_residual for H' (a 3-leaf DF model); on the allowed set too, since the DF model
    keeps the point group. A is linear in H, so the bias at a CQE iterate scales as |t| under
    V -> t V (V = H' - H)."""
    act, el = h4["act"], h4["el"]
    model = act.with_eri(double_factorize(act.eri, n_leaves=3).to_eri())
    psi0 = _ground_state(act)
    assert np.linalg.norm(compute_residual(act, psi0).A) <= 1e-12
    full = np.linalg.norm(compute_residual(model, psi0).A)
    assert full > 1e-3
    every = targets.element_set(act.norb, "spin_free")
    assert ri.factorization_bias(act, model, psi0, every, "spin_free") == pytest.approx(full, rel=1e-10)
    assert ri.factorization_bias(act, model, psi0, el, "spin_free") == pytest.approx(full, rel=1e-10)
    ci = h4["states"][2]
    v = model.eri - act.eri
    b1 = ri.factorization_bias(act, model, ci, el, "spin_free")
    for t in (0.5, 2.0, -1.0):
        bt = ri.factorization_bias(act, act.with_eri(act.eri + t * v), ci, el, "spin_free")
        assert bt == pytest.approx(abs(t) * b1, rel=1e-10)
    with pytest.raises(ValueError):
        ri.factorization_bias(act, sy.benchmark_system("h6a"), ci, el, "spin_free")


def test_aux_bias_curve_interpolation():
    """AuxBiasCurve interpolates linearly in log-log, so it reproduces a power law exactly inside and
    outside the grid; a one-point grid extends with delta^2; zeros give ~0; bad grids raise."""
    d = np.array([0.1, 0.3, 0.9])
    curve = ri.AuxBiasCurve(d, 0.7 * d ** 2)
    for x in (0.05, 0.1, 0.2, 0.5, 0.9, 1.5):
        assert curve(x) == pytest.approx(0.7 * x ** 2, rel=1e-12)
    cubic = ri.AuxBiasCurve(d[::-1], 2.0 * d[::-1] ** 3)          # unsorted input is sorted
    assert cubic(0.2) == pytest.approx(2.0 * 0.2 ** 3, rel=1e-12)
    assert ri.AuxBiasCurve([0.2], [0.04])(0.4) == pytest.approx(0.16, rel=1e-12)
    assert ri.AuxBiasCurve(d, [0.0, 0.0, 0.0])(0.5) < 1e-300
    for bad in (([0.1, -0.2], [1.0, 1.0]), ([0.1, 0.1], [1.0, 2.0]), ([0.1, 0.2], [1.0, -1.0]), ([], [])):
        with pytest.raises(ValueError):
            ri.AuxBiasCurve(*bad)
    with pytest.raises(ValueError):
        curve(0.0)


# ----------------------------------------------------------------------- problem
def test_norm_on_the_allowed_set_is_the_full_tensor_norm(h4):
    """||A||_m over the allowed spin-free elements equals the Frobenius norm of the full N^4 tensor of
    residual.compute_residual (no element sets, no multiplicities) along the SD iterates; for the
    spin-orbital set the allowed elements carry the norm of all 150."""
    act, ir = h4["act"], h4["irreps"]
    so_allowed, _ = ri.allowed_elements(act, "spin_orbital", ir)
    so_all = targets.element_set(act.norb, "spin_orbital")
    for ci in h4["states"][:4]:
        problem = ri.build_problem(act, ci, h4["el"], "spin_free", variances=h4["var"],
                                   normalizations=h4["norms"], deltas=(0.1, 0.5))
        assert problem.norm_a_m == pytest.approx(np.linalg.norm(compute_residual(act, ci).A), rel=1e-10)
        norms = [math.sqrt(np.sum(_m(act, el, "spin_orbital") * routes.exact_residual(act, ci, el) ** 2))
                 for el in (so_allowed, so_all)]
        assert norms[0] == pytest.approx(norms[1], rel=1e-10)


def test_build_problem_wiring(h4):
    """build_problem: A is routes.exact_residual on the elements, m their multiplicities, the variances
    and normalizations pass through, N is the active space's, and aux_bias reproduces aux_bias_curve
    on its grid (default: FTParameters().aux_deltas)."""
    act, el, ci = h4["act"], h4["el"], h4["states"][3]
    problem = ri.build_problem(act, ci, el, "spin_free", variances=h4["var"], normalizations=h4["norms"])
    assert problem.kind == "spin_free" and problem.norb == act.norb and np.array_equal(problem.elements, el)
    assert np.array_equal(problem.A, routes.exact_residual(act, ci, el, "spin_free"))
    assert np.array_equal(problem.m, _m(act, el))
    for key in ("v_direct", "w_aux"):
        assert np.array_equal(getattr(problem, key), h4["var"][key])
    for key in ("lam_q", "alpha_x", "n_terms_x"):
        assert np.array_equal(getattr(problem, key), h4["norms"][key])
    grid = FTParameters().aux_deltas
    ref = ri.aux_bias_curve(act, ci, el, "spin_free", grid)
    assert np.allclose([problem.aux_bias(d) for d in grid], ref, rtol=1e-12, atol=0)


def test_energy_variances_pick_the_lowest_df_split(h4):
    """The four DF splits are schemes.df_variance totals and the lowest is chosen; at H4 SD iterate 1
    the shifted splits are about 3x lower than the unshifted ones (S6), eigh_shift lowest."""
    act, ci = h4["act"], h4["states"][0]
    out = ri.energy_variances(act, ci)
    for name, (opts, shift) in ri.DF_VARIANTS.items():
        assert out[name] == pytest.approx(schemes.df_variance(act, ci, shift=shift, **opts).total, rel=1e-12)
    assert out["best"] == min(ri.DF_VARIANTS, key=out.__getitem__) == "eigh_shift"
    assert out["variance"] == out["eigh_shift"]
    assert out["cholesky"] / out["cholesky_shift"] > 3 and out["eigh"] / out["eigh_shift"] > 2.5


# ----------------------------------------------------------------------- slow
@pytest.mark.slow
def test_direct_variances_are_state_independent(h4):
    """M2 found the per-element matchgate variances state-independent to about 1%, which is why they
    are computed once per system: on H4 (exact pair sums) v_e at SD iterates 1 and 4 agree within 3%
    element by element, and the iterate-1 values are routes.direct_residual_variance's."""
    act, el, var = h4["act"], h4["el"], h4["var"]
    ref = routes.direct_residual_variance(act, h4["states"][0], el, "spin_free", "matchgate_shadow",
                                          method="exact")
    assert np.allclose(var["v_direct"], ref.per_target, rtol=1e-13, atol=0)
    v4 = ri.element_variances(act, h4["states"][3], el, "spin_free")["v_direct"]
    assert np.max(np.abs(v4 / var["v_direct"] - 1)) <= 0.03
    m = _m(act, el)
    assert np.sum(m * v4) == pytest.approx(np.sum(m * var["v_direct"]), rel=0.03)


def _lcu(op: PauliSum):
    """Dense Pauli LCU of ``op`` without its identity term: PREPARE (real orthogonal, first column
    sqrt(|c_j| / alpha)), the SELECT blocks sign(c_j) P_j (identity on padding) and alpha."""
    terms = [(x, z, c) for x, z, c in op.terms() if (x | z) != 0]
    c = np.array([t[2] for t in terms])
    assert np.allclose(c.imag, 0.0)
    alpha = float(np.abs(c).sum())
    dim = 1 << max(1, math.ceil(math.log2(len(terms))))
    v = np.zeros(dim)
    v[:len(terms)] = np.sqrt(np.abs(c.real) / alpha)
    u = -v.copy()
    u[0] += 1.0
    prep = np.eye(dim) - 2.0 * np.outer(u, u) / (u @ u) if u @ u > 1e-30 else np.eye(dim)   # prep e_0 = v
    eye = np.eye(1 << op.n_qubits)
    blocks = [np.sign(cj.real) * PauliSum(op.n_qubits, [x], [z], [1.0]).to_dense() for x, z, cj in terms]
    return prep, blocks + [eye] * (dim - len(terms)), alpha


def _apply_lcu(lcu, state: np.ndarray) -> np.ndarray:
    """(PREP^T x I) SELECT (PREP x I) on a state of shape (ancilla dim, 2^n)."""
    prep, blocks, _ = lcu
    s = prep @ state
    return prep.T @ np.stack([b @ s[j] for j, b in enumerate(blocks)])


def _hadamard_p0(apply_u, phi: np.ndarray, imaginary: bool) -> float:
    """P(0) of the Hadamard test: H on the control, controlled U, S^dagger (for Im) and H."""
    return float(np.linalg.norm(phi + (-1j if imaginary else 1.0) * apply_u(phi)) ** 2 / 4)


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["spin_free", "spin_orbital"])
def test_dense_block_encodings_reproduce_the_residual(kind):
    """H2 (4 qubits), a random real state, every element: dense Pauli-LCU block encodings built from
    the Pauli terms give <0|U_X|0> = X_e / alpha_X with the module's alpha_X, and the Hadamard tests of
    the three AE observables reproduce the residual to 1e-10: b1 Im<0,Psi|U_Q U_H|0,Psi> with separate
    ancillas (A_e = -2 lambda_Q lambda_H (2p - 1)), b2 Re<U_X> (A_e = alpha_X (2p - 1)), and b3 Re<U_Q> on
    expm(-i delta H) Psi (scipy), which gives the aux estimate -<Q_e>/delta of routes.aux_residual."""
    act = systems.molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))], symmetry="D2h", label="h2")
    space = FCISpace(act.norb, act.nelec)
    ci = np.random.default_rng(5).standard_normal(space.shape)
    ci /= np.linalg.norm(ci)
    mapping = QubitMapping(act.norb)
    psi = QubitState.from_ci(act.norb, act.nelec, ci, mapping).to_dense()
    ham = jw_hamiltonian(act, mapping)
    lcu_h = _lcu(ham)
    el = targets.element_set(act.norb, kind)
    norms = ri.element_normalizations(act, el, kind, mapping)
    a = routes.exact_residual(act, ci, el, kind)
    delta = 0.1
    lam_psi = scipy.linalg.expm(-1j * delta * ham.to_dense()) @ psi
    a_aux = routes.aux_residual(act, ci, delta, "exact", el, kind)
    tol = 1e-10 * max(1.0, np.max(np.abs(a)))
    xs = targets.direct_targets(el, act, mapping, kind)
    for e, (q, x) in enumerate(zip(targets.aux_targets(el, mapping, kind), xs)):
        lcu_q, lcu_x = _lcu(q), _lcu(x)
        assert lcu_q[2] == pytest.approx(norms["lam_q"][e], rel=1e-14)
        assert lcu_x[2] == pytest.approx(norms["alpha_x"][e], rel=1e-14)
        dim_x, n = len(lcu_x[1]), psi.size
        block = np.column_stack([_apply_lcu(lcu_x, np.outer(np.eye(dim_x)[0], np.eye(n)[s]))[0] for s in range(n)])
        assert np.allclose(block, x.to_dense() / lcu_x[2], rtol=0, atol=1e-12)
        # b2: X-LCU
        phi_x = np.outer(np.eye(dim_x)[0], psi)
        p = _hadamard_p0(lambda s: _apply_lcu(lcu_x, s), phi_x, imaginary=False)
        assert abs(lcu_x[2] * (2 * p - 1) - a[e]) <= tol
        # b1: product U_Q U_H, separate ancillas; state shape (dim_q, dim_h, 2^n)
        dim_q, dim_h = len(lcu_q[1]), len(lcu_h[1])
        phi = np.zeros((dim_q, dim_h, n), dtype=complex)
        phi[0, 0] = psi

        def product(s):
            s = np.stack([_apply_lcu(lcu_h, s[i]) for i in range(dim_q)])
            return np.stack([_apply_lcu(lcu_q, s[:, j]) for j in range(dim_h)], axis=1)

        p = _hadamard_p0(product, phi, imaginary=True)
        assert abs(-2 * lcu_q[2] * lcu_h[2] * (2 * p - 1) - a[e]) <= tol
        # b3: aux observable on exp(-i delta H) Psi
        phi_q = np.outer(np.eye(dim_q)[0], lam_psi)
        p = _hadamard_p0(lambda s: _apply_lcu(lcu_q, s), phi_q, imaginary=False)
        assert abs(-lcu_q[2] * (2 * p - 1) / delta - a_aux[e]) <= tol
