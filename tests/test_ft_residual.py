"""Tests of the M3 residual-cost routes (``thc_engine_cqe.ft.residual``, F10-F14).

The routes are pure functions of numbers and arrays, so most tests use synthetic problems
(random per-element inputs on the H4 element sets) and compare with references that do not
use the module: the formulas of the M3 specification evaluated by hand in the test
(including the GQSP degree, recomputed here from Bessel functions rather than taken from
``ft.primitives``), ``scipy.optimize`` for the optimal AE allocation, M2's shot rule
``routes.shots_for_relative_error``, and a direct simulation of sine-window phase estimation
for the amplitude-estimation constant.
"""

import dataclasses
import math

import numpy as np
import pytest
import scipy.optimize
import scipy.special

from thc_engine_cqe.ft import primitives as pm
from thc_engine_cqe.ft import residual as rs
from thc_engine_cqe.measure import routes, targets

NORB = 4
PREP = rs.PrepParams(toffolis=2506.0, logical_qubits=70)
# H4 DF block encoding at threshold 3e-3 (specification Appendix A): lambda, C_W, and the
# 188 logical qubits of the QPE minus its 2 ceil(log2(7753)) - 1 = 25 control qubits.
WALK = rs.WalkParams(lam=4.935, toffoli_per_step=868, logical_qubits=163, kind="df")
C_Q = {"spin_free": 16, "spin_orbital": 4}       # F11 [estimate]


def make_problem(k: int = 12, *, kind: str = "spin_free", seed: int = 0, aux_bias=None,
                 uniform: bool = False) -> rs.ResidualProblem:
    """A ResidualProblem on the first k H4 elements with random per-element inputs.

    The magnitudes are those of H4 (memo 2): A ~ 0.1 Ha, v ~ 5-20 Ha^2, lambda_Q ~ 1-2,
    alpha_X ~ 2-8 Ha, L_X ~ 200-900. ``uniform`` makes alpha_X, L_X and m uniform.
    """
    rng = np.random.default_rng(seed)
    el = targets.element_set(NORB, kind)[:k]
    m = targets.element_multiplicities(el, kind, NORB).astype(float)
    alpha_x, n_terms = rng.uniform(2.0, 8.0, k), rng.integers(200, 900, k)
    if uniform:
        m, alpha_x, n_terms = np.full(k, 4.0), np.full(k, 3.0), np.full(k, 500)
    return rs.ResidualProblem(kind=kind, elements=el, m=m, A=0.1 * rng.standard_normal(k),
                              v_direct=rng.uniform(5.0, 20.0, k), w_aux=rng.uniform(2.0, 10.0, k),
                              lam_q=rng.uniform(1.0, 2.0, k), alpha_x=alpha_x, n_terms_x=n_terms,
                              aux_bias=aux_bias, norb=NORB)


def quadratic_bias(beta: float, problem: rs.ResidualProblem):
    """b(delta) = beta delta^2 ||A||_m, the leading order of the aux bias (memo 2: beta 0.16-1.06)."""
    norm = problem.norm_a_m
    return lambda delta: beta * delta ** 2 * norm


def evolution_ref(lam: float, t: float, c_w: int, eps: float = 1e-6, beta: int = 16) -> int:
    """F8 from its definition: R = min{R : 2 sum_{j>R} |J_j(lam t)| <= eps}, 2R C_W + 3(2R+1)(beta-2)."""
    a = np.abs(scipy.special.jv(np.arange(400), lam * t))
    r = next(r for r in range(399) if 2.0 * a[r + 1:].sum() <= eps)
    return 0 if r == 0 else 2 * r * c_w + 3 * (2 * r + 1) * (beta - 2)


def budget_ref(problem: rs.ResidualProblem, r: float, bias: float = 0.0) -> float | None:
    """B of F10, or None when the bias exhausts the target (M2's 1e-12 round-off rule)."""
    target = (r * problem.norm_a_m) ** 2
    b2 = target - bias ** 2
    return None if b2 <= 1e-12 * target else math.sqrt(b2)


def ae_ref(alpha, m, budget, kappa, c_prep, c_obs, q0):
    """F11 by hand: eps_e = t (alpha_e/m_e)^(1/3), N_e = 2 pi kappa alpha_e / eps_e,
    Q_run = q0 + ceil(log2(N_e/2)) + 1, T = sum_e N_e (c_prep + c_obs,e + Q_run,e / 2).
    Returns (sum N_e, T, max_e run, max Q_run)."""
    s = np.sum(np.cbrt(m) * np.cbrt(alpha) ** 2)
    eps = budget / math.sqrt(s) * np.cbrt(alpha / m)
    n = 2 * math.pi * kappa * alpha / eps
    q_run = q0 + np.ceil(np.log2(n / 2)) + 1
    run = n * (c_prep + c_obs + q_run / 2)
    return n.sum(), run.sum(), run.max(), int(q_run.max())


# ----------------------------------------------------------------------- F11 allocation and kappa
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ae_allocation_matches_slsqp(seed):
    """ae_allocation gives min sum_e alpha_e/eps_e under sum_e m_e eps_e^2 = B^2: its Gamma/B matches
    scipy's SLSQP minimum to 1e-6 relative and its eps_e the SLSQP minimizer to 1e-4 (a numerical
    optimizer, independent of the closed form), and the eps_e satisfy the constraint exactly."""
    rng = np.random.default_rng(seed)
    k = 7
    alpha, m, budget = rng.uniform(0.5, 5.0, k), rng.choice([2.0, 4.0, 8.0], k), rng.uniform(0.01, 0.5)
    g_over_b, eps = rs.ae_allocation(alpha, m, budget)
    x0 = np.full(k, budget / math.sqrt(m.sum()))
    res = scipy.optimize.minimize(lambda x: np.sum(alpha / x), x0, method="SLSQP",
                                  jac=lambda x: -alpha / x ** 2,
                                  constraints=[{"type": "eq", "fun": lambda x: np.sum(m * x ** 2) - budget ** 2,
                                                "jac": lambda x: 2 * m * x}],
                                  bounds=[(1e-9, None)] * k, options={"ftol": 1e-15, "maxiter": 1000})
    assert res.success
    assert g_over_b == pytest.approx(res.fun, rel=1e-6)
    assert np.allclose(eps, res.x, rtol=1e-4)
    assert np.sum(m * eps ** 2) == pytest.approx(budget ** 2, rel=1e-12)
    assert np.sum(alpha / eps) == pytest.approx(g_over_b, rel=1e-12)


def test_ae_uniform_case_is_lambda_over_eps_per_element():
    """With uniform alpha and m, the AE route needs N = 2 pi kappa alpha K^(3/2) / (r ||A||_el)
    applications (||A||_el the unweighted norm), the "lambda/eps per element" count at
    eps = r ||A||_el / sqrt(K) (memo 2, section 1.4): a closed form, evaluated by hand."""
    p = make_problem(12, uniform=True)
    params = pm.FTParameters()
    for kappa in (1.0, 2.0):
        c = rs.ae_x_lcu(p, PREP, kappa=kappa, params=params)
        expected = 2 * math.pi * kappa * 3.0 * 12 ** 1.5 / (params.r * np.linalg.norm(p.A))
        assert c.n_prep == pytest.approx(expected, rel=1e-12)


def test_ae_constant_from_sine_window_phase_estimation():
    """Sine-window phase estimation with M = 2^t - 1 applications of G: the RMS error of the phase
    estimate over the uniform phase offset is kappa pi / M with kappa in [0.96, 1.0] for M = 63 and
    255 (Appendix A: 0.974 / 0.993). The outcome law is simulated directly (register amplitudes
    sin(pi (k+1)/(2^t+1)), controlled powers of G, inverse QFT), independent of the cost code; the
    count N = 2M applications of the Hadamard-test circuit then matches primitives.ae_applications
    at eps = pi alpha / M."""
    offsets = (np.arange(64) + 0.5) / 64          # phase = 2 pi (j + u) / 2^t, j irrelevant
    for t in (6, 8):
        dim, m_apps = 2 ** t, 2 ** t - 1
        k = np.arange(dim)
        amp = np.sin(np.pi * (k + 1) / (dim + 1))
        amp /= np.linalg.norm(amp)
        phi = 2 * np.pi * offsets / dim
        prob = np.abs(np.fft.fft(amp * np.exp(1j * np.outer(phi, k)), axis=1)) ** 2 / dim
        assert np.allclose(prob.sum(axis=1), 1.0, atol=1e-12)
        err = np.angle(np.exp(1j * (2 * np.pi * k[None, :] / dim - phi[:, None])))
        ratio = math.sqrt(np.mean(np.sum(prob * err ** 2, axis=1))) / (math.pi / m_apps)
        assert 0.96 <= ratio <= 1.0
        alpha = 2.7
        assert pm.ae_applications(alpha, math.pi * alpha / m_apps) == pytest.approx(2 * m_apps, rel=1e-12)


# ----------------------------------------------------------------------- F10 sampling routes
@pytest.mark.parametrize("kind", ["spin_free", "spin_orbital"])
def test_sampling_direct_equals_m2_shot_rule(kind):
    """direct_mg: n_prep is exactly routes.shots_for_relative_error(v, A, r, weights=m) (M2's rule),
    each shot costs C(Psi_k) Toffolis (Clifford readout), on Q(Psi_k) qubits."""
    p = make_problem(10, kind=kind, seed=3)
    params = pm.FTParameters(r=0.3)
    c = rs.sampling_direct(p, PREP, params=params)
    assert c.n_prep == routes.shots_for_relative_error(p.v_direct, p.A, 0.3, weights=p.m)
    assert c.toffolis == c.n_prep * PREP.toffolis
    assert c.toffolis_per_prep == c.coherent_toffolis == PREP.toffolis
    assert c.n_runs == c.n_prep and c.logical_qubits == PREP.logical_qubits and c.flags == ()
    assert c.params["budget"] == pytest.approx(0.3 * p.norm_a_m, rel=1e-15)


def test_sampling_aux_minimizes_over_the_delta_grid():
    """aux_mg: S(delta) = W_m / (delta^2 B(delta)^2) shots of C(Psi_k) + C_evol(delta), B(delta) with
    the aux bias plus the factorization bias (triangle inequality), minimized over aux_deltas; the
    reference evaluates F10 and F8 by hand for every delta; Q = Q(Psi_k) + Q_walk + 1 - 2N."""
    p0 = make_problem(12, seed=4)
    p = dataclasses.replace(p0, aux_bias=quadratic_bias(0.4, p0))
    walk = dataclasses.replace(WALK, fact_bias_m=0.02 * p.norm_a_m)
    params = pm.FTParameters()
    c = rs.sampling_aux(p, PREP, walk, params=params)
    w_m = np.sum(p.m * p.w_aux)
    rows = []
    for d in params.aux_deltas:
        b = budget_ref(p, params.r, p.aux_bias(d) + walk.fact_bias_m)
        if b is not None:
            shots = w_m / (d ** 2 * b ** 2)
            rows.append((shots * (PREP.toffolis + evolution_ref(walk.lam, d, walk.toffoli_per_step)), shots, d))
    t_best, s_best, d_best = min(rows)
    assert c.toffolis == pytest.approx(t_best, rel=1e-12)
    assert c.n_prep == pytest.approx(s_best, rel=1e-12)
    assert c.params["delta"] == d_best
    assert c.params["c_evol"] == evolution_ref(walk.lam, d_best, walk.toffoli_per_step)
    assert c.logical_qubits == PREP.logical_qubits + walk.logical_qubits + 1 - 2 * NORB
    assert c.flags == ()


# ----------------------------------------------------------------------- F11 amplitude estimation
def test_ae_routes_follow_f11():
    """ae_product (alpha = 2 lambda_Q lambda, C_obs = C_W + C_Q), ae_x (alpha_X, 2 L_X) and ae_aux
    (lambda_Q / delta, C_evol(delta) + C_Q) at one delta: applications, Toffolis (with the reflection
    cost Q_run / 2 per application), longest run, runs and qubits of F11, evaluated by hand; the
    biases enter the budget (factorization for b1 and b3, b(delta) for b3, none for b2)."""
    p0 = make_problem(9, seed=5)
    p = dataclasses.replace(p0, aux_bias=quadratic_bias(0.3, p0))
    walk = dataclasses.replace(WALK, fact_bias_m=0.01 * p.norm_a_m)
    delta = 0.5
    params = pm.FTParameters(aux_deltas=(delta,))
    c_evol = evolution_ref(walk.lam, delta, walk.toffoli_per_step)
    cases = {
        "ae_product": (rs.ae_product(p, PREP, walk, kappa=1.5, params=params), 2 * p.lam_q * walk.lam,
                       walk.toffoli_per_step + C_Q["spin_free"], walk.logical_qubits - 2 * NORB + 4,
                       walk.fact_bias_m),
        "ae_x": (rs.ae_x_lcu(p, PREP, kappa=1.5, params=params), p.alpha_x, 2.0 * p.n_terms_x,
                 np.ceil(np.log2(p.n_terms_x)) + params.chi + 4, 0.0),
        "ae_aux": (rs.ae_aux(p, PREP, walk, kappa=1.5, params=params), p.lam_q / delta,
                   c_evol + C_Q["spin_free"], walk.logical_qubits + 1 - 2 * NORB + 4,
                   p.aux_bias(delta) + walk.fact_bias_m),
    }
    for route, (c, alpha, c_obs, q_obs, bias) in cases.items():
        budget = budget_ref(p, params.r, bias)
        n, t, run, q = ae_ref(alpha, p.m, budget, 1.5, PREP.toffolis, c_obs, PREP.logical_qubits + q_obs)
        assert c.route == route
        assert c.n_prep == pytest.approx(n, rel=1e-12)
        assert c.toffolis == pytest.approx(t, rel=1e-12)
        assert c.coherent_toffolis == pytest.approx(run, rel=1e-12)
        assert c.toffolis_per_prep == pytest.approx(t / n, rel=1e-12)
        assert c.n_runs == p.n_elements and c.logical_qubits == q
        assert c.params["kappa"] == 1.5 and c.params["budget"] == pytest.approx(budget, rel=1e-12)
        assert "estimate" in c.flags
    assert cases["ae_aux"][0].params["delta"] == delta


def test_kappa_scales_the_applications():
    """compare_routes evaluates each AE route at kappa and kappa_practical; N is proportional to kappa
    (F11), so the applications double from kappa = 1 to 2, and the qubits grow by the extra phase bit."""
    p = make_problem(8, seed=6)
    costs = rs.compare_routes(p, PREP, WALK, params=pm.FTParameters(kappa=1.0, kappa_practical=2.0))
    for route in ("ae_product", "ae_x", "ae_aux"):
        c1, c2 = (c for c in costs if c.route == route)
        assert (c1.params["kappa"], c2.params["kappa"]) == (1.0, 2.0)
        assert c2.n_prep == pytest.approx(2 * c1.n_prep, rel=1e-12)
        assert c2.logical_qubits == c1.logical_qubits + 1


# ----------------------------------------------------------------------- F12 gradient estimation
@pytest.mark.parametrize("n_blocks", [1, 3])
def test_gradient_estimation_follows_f12(n_blocks):
    """gaw: N = 2 pi kappa_g sqrt(K n_B) (sum_e m_e alpha_e^2)^(1/2) / B with alpha_e = lambda_Q,e/delta,
    T = N (C(Psi_k) + C_evol + K (2 b_theta + 2) / n_B), n_B coherent blocks, ceil(K/n_B) b_theta extra
    qubits over the aux route, evaluated by hand; flagged as an assumption."""
    p0 = make_problem(9, seed=7)
    p = dataclasses.replace(p0, aux_bias=quadratic_bias(0.3, p0))
    delta, k = 0.6, p.n_elements
    params = pm.FTParameters(aux_deltas=(delta,))
    c = rs.gradient_estimation(p, PREP, WALK, kappa_g=10.0, n_blocks=n_blocks, b_theta=12, params=params)
    budget = budget_ref(p, params.r, p.aux_bias(delta))
    n = 2 * math.pi * 10.0 * math.sqrt(k * n_blocks) * math.sqrt(np.sum(p.m * (p.lam_q / delta) ** 2)) / budget
    per = PREP.toffolis + evolution_ref(WALK.lam, delta, WALK.toffoli_per_step) + k * (2 * 12 + 2) / n_blocks
    assert c.n_prep == pytest.approx(n, rel=1e-12)
    assert c.toffolis == pytest.approx(n * per, rel=1e-12)
    assert c.coherent_toffolis == pytest.approx(n * per / n_blocks, rel=1e-12)
    assert c.n_runs == n_blocks
    assert c.logical_qubits == PREP.logical_qubits + WALK.logical_qubits + 1 - 2 * NORB + math.ceil(k / n_blocks) * 12
    assert "assumption" in c.flags and c.params["kappa_g"] == 10.0
    with pytest.raises(ValueError):
        rs.gradient_estimation(p, PREP, WALK, kappa_g=1.0, n_blocks=k + 1)


# ----------------------------------------------------------------------- biases, r, C(Psi_k)
def test_infeasible_bias_gives_the_flag_and_infinite_cost():
    """A factorization bias above r ||A||_m makes every route that uses H' infeasible (flag, inf
    applications and Toffolis); the direct route and the X-LCU route, which use the exact H, are
    unaffected. An aux bias above the target at every delta makes the aux routes infeasible."""
    p = make_problem(10, seed=8)
    params = pm.FTParameters(r=0.3)
    walk = dataclasses.replace(WALK, fact_bias_m=0.3 * p.norm_a_m * 1.01)
    costs = rs.compare_routes(p, PREP, walk, params=params)
    for c in costs:
        if c.route in ("direct_mg", "ae_x"):
            assert "infeasible" not in c.flags and math.isfinite(c.toffolis)
        else:
            assert "infeasible" in c.flags
            assert c.toffolis == c.n_prep == math.inf and c.params["budget"] == 0.0
    assert rs.best_route(costs).route in ("direct_mg", "ae_x")
    huge = dataclasses.replace(p, aux_bias=lambda d: 10.0 * p.norm_a_m)
    for c in (rs.sampling_aux(huge, PREP, WALK, params=params), rs.ae_aux(huge, PREP, WALK, params=params)):
        assert "infeasible" in c.flags and c.toffolis == math.inf and c.params["delta"] is None
    assert rs.variance_budget(1.0, 0.3, 0.3) == 0.0
    assert rs.variance_budget(1.0, 0.3, 0.3 * (1 - 1e-14)) == 0.0          # M2's round-off rule
    assert rs.variance_budget(2.0, 0.3, 0.36) == pytest.approx(0.48, rel=1e-14)


def test_costs_fall_with_r_and_rise_with_the_state_cost():
    """Every feasible route gets cheaper as the target r loosens and dearer as C(Psi_k) grows
    (F10-F12: shots and applications fall with B, every per-preparation cost contains C(Psi_k))."""
    p0 = make_problem(10, seed=9)
    p = dataclasses.replace(p0, aux_bias=quadratic_bias(0.3, p0))
    walk = dataclasses.replace(WALK, fact_bias_m=0.01 * p.norm_a_m)
    by_r = [rs.compare_routes(p, PREP, walk, params=pm.FTParameters(r=r)) for r in (0.1, 0.3, 1.0)]
    by_c = [rs.compare_routes(p, rs.PrepParams(c, 70), walk) for c in (0.0, 1e3, 1e5)]
    for series in (by_r, by_c[::-1]):
        for costs in zip(*series):
            t = [c.toffolis for c in costs]
            assert all(math.isfinite(x) for x in t) and t[0] > t[1] > t[2]
    assert by_c[0][0].toffolis == 0.0                # HF is Clifford: direct sampling costs no Toffolis


def test_crossover_norm_matches_the_closed_form():
    """Without biases T_direct = V_m C / (r^2 ||A||^2) and T_AE = 2 pi kappa Gamma (C + C_obs) / (r ||A||),
    so they cross at ||A||* = V_m C / (2 pi kappa Gamma r (C + C_obs)); C_obs includes the reflection
    cost Q_run/2 at the crossing (uniform for uniform alpha and m). crossover_norm (a grid search and
    bisection on the routes) agrees to 1e-9, and both routes cost the same there. At C(Psi_k) = 0
    sampling is free, so there is no crossing (nan)."""
    p = make_problem(12, seed=10, uniform=True)
    params = pm.FTParameters(r=0.3, kappa=1.0)
    a_star = rs.crossover_norm(p, PREP, WALK, "direct_mg", "ae_x", params=params)
    at = dataclasses.replace(p, A=p.A * a_star / p.norm_a_m)
    ae = rs.ae_x_lcu(at, PREP, params=params)
    c_obs = ae.toffolis_per_prep - PREP.toffolis                  # 2 L_X + Q_run / 2
    assert c_obs == pytest.approx(2 * 500 + ae.logical_qubits / 2, rel=1e-12)
    v_m, gamma = np.sum(p.m * p.v_direct), (np.sum(np.cbrt(p.m) * np.cbrt(p.alpha_x) ** 2)) ** 1.5
    closed = v_m * PREP.toffolis / (2 * math.pi * gamma * params.r * (PREP.toffolis + c_obs))
    assert a_star == pytest.approx(closed, rel=1e-9)
    assert rs.sampling_direct(at, PREP, params=params).toffolis == pytest.approx(ae.toffolis, rel=1e-9)
    hf = rs.PrepParams(0.0, 70)
    assert math.isnan(rs.crossover_norm(p, hf, WALK, "direct_mg", "ae_x", params=params))
    with pytest.raises(ValueError):
        rs.crossover_norm(p, PREP, WALK, "direct_mg", "gaw")


# ----------------------------------------------------------------------- comparison
def test_compare_routes_and_best_route():
    """compare_routes returns the routes in the documented order (sampling, AE at kappa and at
    kappa_practical, GAW at each kappa_g); best_route takes the cheapest at one kappa, never GAW by
    default, and lets the kappa-free sampling routes compete at every kappa."""
    p0 = make_problem(10, seed=11)
    p = dataclasses.replace(p0, aux_bias=quadratic_bias(0.3, p0))
    params = pm.FTParameters()
    costs = rs.compare_routes(p, PREP, WALK, params=params)
    assert [c.route for c in costs] == ["direct_mg", "aux_mg"] + ["ae_product", "ae_x", "ae_aux"] * 2 + ["gaw"] * 3
    assert [c.params["kappa_g"] for c in costs[-3:]] == list(params.kappa_g)
    for kappa in (1.0, 2.0):
        pool = [c for c in costs if c.route != "gaw" and c.params.get("kappa", kappa) == kappa]
        assert rs.best_route(costs, kappa=kappa) is min(pool, key=lambda c: c.toffolis)
    everything = rs.best_route(costs, exclude=())
    assert everything.toffolis == min(c.toffolis for c in costs if c.params.get("kappa", 1.0) == 1.0)
    with pytest.raises(ValueError):
        rs.best_route(costs, exclude=rs.ROUTES)


# ----------------------------------------------------------------------- F13 energies
def test_line_search_cost_follows_f13():
    """n_c candidates, each measured with V_E / eps_ls^2 shots of C(Psi_k) + C_step,c + C_rot: for
    uniform candidates T = n_c V_E / eps^2 (C(Psi_k) + C_step + C_rot), and in general the sum over
    the candidates; the longest shot is the dearest candidate."""
    params = pm.FTParameters(eps_ls=1.6e-3)
    c = rs.line_search_cost(0.61, PREP, [1154.0] * 11, 405, params=params)
    assert c.toffolis == pytest.approx(11 * 0.61 / 1.6e-3 ** 2 * (PREP.toffolis + 1154 + 405), rel=1e-12)
    assert c.n_prep == pytest.approx(11 * 0.61 / 1.6e-3 ** 2, rel=1e-12)
    steps = [160.0, 1192.0, 1154.0]
    c = rs.line_search_cost(0.61, PREP, steps, 405, params=params, logical_qubits=90)
    assert c.toffolis == pytest.approx(0.61 / 1.6e-3 ** 2 * sum(PREP.toffolis + s + 405 for s in steps), rel=1e-12)
    assert c.coherent_toffolis == PREP.toffolis + 1192 + 405
    assert c.logical_qubits == 90 and c.params["n_candidates"] == 3
    with pytest.raises(ValueError):
        rs.line_search_cost(0.61, PREP, [], 405)


def test_energy_routes():
    """energy_sampling: V_E / eps^2 shots of the given cost. energy_ae: N = 2 pi kappa lambda / eps
    applications of C(Psi_k) + C_W + Q_run/2 in one run (F11 with alpha = lambda); with N known the
    state and the walk share the 2N system qubits, without it the count is an upper bound (flagged)."""
    c = rs.energy_sampling(0.61, 3000.0, 70, eps=1.6e-3)
    assert c.n_prep == pytest.approx(0.61 / 1.6e-3 ** 2, rel=1e-14) and c.toffolis == pytest.approx(c.n_prep * 3000.0)
    e = rs.energy_ae(WALK, PREP, eps=1e-3, kappa=1.0, norb=NORB)
    n = 2 * math.pi * WALK.lam / 1e-3
    q = PREP.logical_qubits + WALK.logical_qubits - 2 * NORB + math.ceil(math.log2(n / 2)) + 1
    assert e.n_prep == pytest.approx(n, rel=1e-14) and e.logical_qubits == q and e.n_runs == 1
    assert e.toffolis == pytest.approx(n * (PREP.toffolis + WALK.toffoli_per_step + q / 2), rel=1e-12)
    assert e.coherent_toffolis == e.toffolis
    upper = rs.energy_ae(WALK, PREP, eps=1e-3)
    assert upper.logical_qubits == q + 2 * NORB and "qubits_upper_bound" in upper.flags


# ----------------------------------------------------------------------- data types
def test_problem_validation_and_immutability():
    """ResidualProblem copies its arrays read-only, infers N from the elements when not given (never
    below the largest orbital index + 1) and rejects malformed inputs; WalkParams and PrepParams
    reject non-physical values."""
    p = make_problem(6, seed=12)
    a = np.array(p.A)
    q = dataclasses.replace(p, A=a, norb=None)
    a[0] = 99.0
    assert q.A[0] != 99.0 and not q.A.flags.writeable and not q.elements.flags.writeable
    assert q.norb == int(q.elements.max()) + 1
    with pytest.raises(ValueError):
        dataclasses.replace(p, norb=1)                       # below the largest index + 1
    with pytest.raises(ValueError):
        dataclasses.replace(p, v_direct=-p.v_direct)
    with pytest.raises(ValueError):
        dataclasses.replace(p, m=np.zeros(6))
    with pytest.raises(ValueError):
        dataclasses.replace(p, A=p.A[:-1])
    with pytest.raises(ValueError):
        dataclasses.replace(p, elements=targets.element_set(NORB, "spin_orbital")[:6])
    with pytest.raises(ValueError):
        dataclasses.replace(p, kind="pauli")
    with pytest.raises(TypeError):
        dataclasses.replace(p, aux_bias=0.1)
    so = make_problem(6, kind="spin_orbital", seed=12)
    assert dataclasses.replace(so, norb=None).norb == int(so.elements[..., 0].max()) + 1
    for bad in ({"lam": 0.0}, {"toffoli_per_step": 1.5}, {"logical_qubits": 0}, {"fact_bias_m": -1.0}):
        with pytest.raises(ValueError):
            dataclasses.replace(WALK, **bad)
    with pytest.raises(ValueError):
        rs.PrepParams(toffolis=-1.0, logical_qubits=10)
    with pytest.raises(ValueError):
        rs.ae_allocation(np.ones(3), np.ones(2), 0.1)


def test_missing_aux_bias_is_flagged_as_a_lower_bound():
    """Without an aux-bias curve the aux routes are costed bias-free, which can only lower their cost
    (a lower bound), and they say so with the flag "no_aux_bias"."""
    p0 = make_problem(10, seed=13)
    p = dataclasses.replace(p0, aux_bias=quadratic_bias(0.5, p0))
    for f in (rs.sampling_aux, rs.ae_aux):
        free, biased = f(p0, PREP, WALK), f(p, PREP, WALK)
        assert "no_aux_bias" in free.flags and "no_aux_bias" not in biased.flags
        assert free.toffolis <= biased.toffolis
