"""Tests of the closed-loop finite-shot fault-tolerant CQE (``thc_engine_cqe.ft.closedloop``; F25, F32-F38).

References: M3 itself for the noise-free limit (``ft.trajectory.exact_trajectory`` and ``pipeline.iteration_costs``
bit for bit, and the golden record tests/data/ft_h4_golden.json); ``steprules.run_step_rule`` with its default
(masked exact) residual for the unit rule; Monte Carlo against the closed forms of F33-F35 (element means and
variances, the realized r~^2 and its relative spread (2/K_eff)^(1/2), the debiased norm and its precision); the
Poisson-summation identities of the sine-window lattice law summed to +-2e4 and the finite-register sine-window law,
built here by direct summation over the register (kappa(M) and the continuum limit); the F11 allocation by hand;
``residual.compare_routes`` at r_k on the realized state and the F10/F11 cost formulas by hand (F36);
``stateprep.prep_cost`` and ``ft.trajectory.propagate`` for the compiled loop; hand-computed hand-off costs, shares
and regrets; the coverage of the percentile bootstrap on synthetic Gaussian sets and the Wilson interval's closed
form; ``circuits.replay_elementwise`` for the compiled step (an independent loop over the same excitations that
carries the Fock-space vector between steps); and the [design] headline cells (slow).
"""

import dataclasses
import json
import math
import os
import statistics
import warnings
import zlib

import numpy as np
import pytest

from thc_engine_cqe.fci_ops import FCISpace, Hamiltonian, normalize
from thc_engine_cqe.ft import circuits
from thc_engine_cqe.ft import closedloop as cl
from thc_engine_cqe.ft import pipeline
from thc_engine_cqe.ft import residual_inputs as ri
from thc_engine_cqe.ft import scaling
from thc_engine_cqe.ft import symmetry as sy
from thc_engine_cqe.ft.circuits import compiled_step
from thc_engine_cqe.ft.handoff import handoff_table
from thc_engine_cqe.ft.primitives import FTParameters
from thc_engine_cqe.ft.residual import PrepParams, compare_routes
from thc_engine_cqe.ft.stateprep import candidate_step_costs, elementwise_step_cost, prep_cost
from thc_engine_cqe.ft.trajectory import exact_trajectory, propagate
from thc_engine_cqe.measure import targets
from thc_engine_cqe.residual import compute_residual
from thc_engine_cqe.steprules import StepRule, run_step_rule

M3 = FTParameters()
GOLDEN = os.path.join(os.path.dirname(__file__), "data", "ft_h4_golden.json")


def _z_family(n_tests: int, z: float = 3.0) -> float:
    """The two-sided threshold whose family-wise error over ``n_tests`` independent tests is that of one z-sigma
    test (Sidak): every per-element check of a K-element estimator is held to "3 standard errors" family-wise."""
    alpha = math.erfc(z / math.sqrt(2.0))
    return statistics.NormalDist().inv_cdf(1.0 - 0.5 * (1.0 - (1.0 - alpha) ** (1.0 / n_tests)))


@pytest.fixture(scope="module")
def h4():
    """SystemInputs of h4 (M3's inputs: iterate-1 variances of the exact spin-free record)."""
    return cl.system_inputs("h4")


@pytest.fixture(scope="module")
def psi1(h4):
    """Psi_1 of the classical first step (spin-orbital), its residual data, the measured A_e and ||A_1||_m."""
    act, ir = h4.active, h4.irreps
    run = run_step_rule(act, StepRule("unit"), max_iter=1, irreps=ir, keep_states=True)
    psi = run.states[1]
    rd = compute_residual(act, psi, Hamiltonian(act))
    a = np.real(sy.mask_pair_tensor(rd.A, ir)[tuple(h4.elements.T)])
    return {"psi": psi, "rd": rd, "a": a, "norm": math.sqrt(float(np.sum(h4.m * a * a)))}


@pytest.fixture(scope="module")
def records():
    """Exact spin-free L-BFGS records (4 steps, masked residual) of h4 and n2s."""
    out = {}
    for name in ("h4", "n2s"):
        act = sy.benchmark_system(name)
        ir = sy.orbital_irreps(act)
        with warnings.catch_warnings():             # E1 is not needed: two roots suffice
            warnings.simplefilter("ignore", RuntimeWarning)
            rec = exact_trajectory(act, max_iter=4, irreps=ir, n_roots=2)
        out[name] = (act, ir, rec)
    return out


def _estimator(inputs, spec, norm0, seed=7):
    ctrl = cl.PrecisionController(spec, inputs, norm0)
    rng_res, rng_off, _ = cl.replicate_streams(inputs.name, seed)
    return cl.FTResidualEstimator(inputs, spec, ctrl, rng_res, rng_off), ctrl


# ------------------------------------------------------------------------- F32: the noise-free limit
@pytest.mark.parametrize("first, loop", [("second_order", "classical_first_step"), ("exact", "measured")])
def test_noise_free_limit_is_m3(h4, first, loop):
    """noise=False, oracle at M3's r, "m3_line_search", spin-free: states, energies, eps, overlaps, variances and
    directions are exact_trajectory's (np.array_equal), and closed_loop_costs is pipeline.iteration_costs on that
    record, i.e. M3's C_iter(k) of tests/data/ft_h4_golden.json (toffolis ==) for k = 0..7."""
    act, ir, spec_ = h4.active, h4.irreps, h4.spectrum
    spec = cl.LoopSpec(r=M3.r, noise=False, controller="oracle")
    rec = cl.run_closed_loop(h4, spec, "m3_line_search", replicate_index=0, max_iter=20, metric="spin_free",
                             first_step=first)
    ref = exact_trajectory(act, optimizer="lbfgs", metric="spin_free", max_iter=20, system="h4", irreps=ir,
                           n_roots=spec_.n_roots, first_step=first, roots=spec_.roots)
    assert rec.n_steps == ref.n_steps
    for name in ("energies", "eps", "overlaps", "variances", "residual_norms"):
        assert np.array_equal(getattr(rec, name), getattr(ref, name)), name
    assert len(rec.states) == len(ref.states)
    assert all(np.array_equal(a, b) for a, b in zip(rec.states, ref.states))
    assert all(np.array_equal(a, b) for a, b in zip(rec.directions, ref.directions))
    assert rec.converged and [e.k for e in rec.events] == list(range(1, rec.n_steps))   # A_0 is classical
    assert all(e.r_realized == pytest.approx(M3.r, rel=1e-14) and e.shots > 0.0 for e in rec.events)

    k_cost = 8
    rec8 = cl.run_closed_loop(h4, spec, "m3_line_search", replicate_index=0, max_iter=k_cost, metric="spin_free",
                              first_step=first)
    costs = cl.closed_loop_costs(h4, rec8)
    with open(GOLDEN) as fh:
        gold = json.load(fh)["families"]["lbfgs"]["c_iter"][loop]
    assert [c.toffolis for c in costs] == gold
    ref8 = exact_trajectory(act, optimizer="lbfgs", metric="spin_free", max_iter=k_cost, system="h4", irreps=ir,
                            n_roots=spec_.n_roots, first_step=first, roots=spec_.roots)
    prep = prep_cost(ref8.directions, ref8.eps, irreps=ir)
    m3 = pipeline.iteration_costs(act, ref8, prep, h4.qpe.encoding, elements=h4.elements, variances=h4.variances,
                                  normalizations=h4.normalizations, energy_variance=h4.energy_split,
                                  readout_toffolis=h4.readout_toffolis, classical_first_step=first == "second_order",
                                  k_max=k_cost - 1)
    assert [c.toffolis for c in m3] == [c.toffolis for c in costs]
    assert [c.best for c in m3] == [c.best for c in costs]
    assert np.array_equal(rec8.prep_cumulative, prep.cumulative[:k_cost + 1])


@pytest.mark.parametrize("metric", ["spin_orbital", "spin_free"])
@pytest.mark.parametrize("controller", ["oracle", "lagged"])
def test_unit_noise_free_equals_steprules(h4, metric, controller):
    """Without noise the closed loop's unit rule (either controller: the budgets are planned and costed, the
    estimate is the masked exact residual) is steprules.run_step_rule with its default residual: energies to
    1e-13 Ha, steps and states to rounding."""
    act, ir = h4.active, h4.irreps
    rec = cl.run_closed_loop(h4, cl.LoopSpec(noise=False, controller=controller), "unit", replicate_index=0,
                             max_iter=8, metric=metric, keep_states=True)
    ref = run_step_rule(act, StepRule("unit"), max_iter=8, metric=metric, irreps=ir, keep_states=True)
    assert rec.n_steps == ref.n_steps == len(ref.eps)
    assert np.max(np.abs(rec.energies - ref.energies)) <= 1e-13
    assert np.max(np.abs(rec.eps - np.asarray(ref.eps))) <= 1e-12
    assert max(float(np.linalg.norm(a - b)) for a, b in zip(rec.states, ref.states)) <= 1e-12
    assert "first:second_order" in rec.flags[0] and rec.compile_errors.tolist() == [0.0] * (rec.n_steps + 1)
    res = [e for e in rec.events if e.kind == "residual"]
    assert [e.k for e in res] == list(range(1, rec.n_steps))          # none at the classical k = 0
    assert np.array_equal(rec.budgets, [e.budget for e in res])


# ------------------------------------------------------------------------- F33: the direct route
def test_direct_estimator_statistics(h4, psi1):
    """F33 by Monte Carlo (1e4 draws at Psi_1, oracle r = 1): eta_e = A^_e - A_e has mean 0 and variance v_e/S_k,
    S_k = V_m/B_k^2, per element within 3 standard errors (family-wise over the K elements); the realized r~^2 =
    ||eta||_m^2/||A||_m^2 has mean (B_k/||A_k||_m)^2 (3 SE) and relative sd (2/K_eff)^(1/2) within 10%; the
    returned V^_e is v_e/S_k; the filled tensor carries A^_e on the allowed set and exact zeros elsewhere."""
    a, norm = psi1["a"], psi1["norm"]
    n = 10000
    spec = cl.LoopSpec(controller="oracle", r=1.0)
    est, ctrl = _estimator(h4, spec, norm0=1.0)
    v = h4.variances["v_direct"]
    for _ in range(n):
        a_est, var = est(psi1["rd"], psi1["psi"], 1)
    ev = est.events[-1]
    budget = 1.0 * norm
    assert ev.budget == budget and ev.route == "direct_mg" and ev.shots == pytest.approx(h4.variance_m / budget ** 2,
                                                                                            rel=1e-14)
    assert np.allclose(var, v / ev.shots, rtol=1e-14, atol=0.0)
    allowed = sy.mask_pair_tensor(np.ones_like(psi1["rd"].A), h4.irreps) != 0
    assert np.all(a_est[~allowed] == 0.0)
    assert np.array_equal(np.real(a_est[tuple(h4.elements.T)]), est.estimates[-1])

    eta = np.array(est.estimates) - a
    sigma2 = v / ev.shots
    z_crit = _z_family(a.size)
    z_mean = eta.mean(axis=0) / np.sqrt(sigma2 / n)
    z_var = (eta.var(axis=0, ddof=1) - sigma2) / (sigma2 * math.sqrt(2.0 / (n - 1)))
    assert np.max(np.abs(z_mean)) <= z_crit and np.max(np.abs(z_var)) <= z_crit

    r2 = np.sum(h4.m * eta ** 2, axis=1) / norm ** 2
    assert abs(r2.mean() - (budget / norm) ** 2) <= 3.0 * r2.std(ddof=1) / math.sqrt(n)
    k_eff = h4.variance_m ** 2 / float(np.sum(h4.m ** 2 * v ** 2))
    assert r2.std(ddof=1) / r2.mean() == pytest.approx(math.sqrt(2.0 / k_eff), rel=0.1)


# ------------------------------------------------------------------------- F34: the sine-window law
def _finite_sine_law(dim: int, offsets: np.ndarray):
    """The sine-window register of ``dim`` points (M = dim - 1 applications of G), amplitudes proportional to
    sin(pi (x + 1)/(dim + 1)), the true phase 2 pi u/dim: outcome probabilities (m, u) by direct summation over the
    register, and the errors m - u wrapped to [-dim/2, dim/2) in grid units."""
    x = np.arange(dim)
    amp = np.sin(np.pi * (x + 1) / (dim + 1))
    amp /= np.linalg.norm(amp)
    dft = np.exp(-2j * np.pi * np.outer(x, x) / dim)
    phase = np.exp(2j * np.pi * np.outer(x, offsets) / dim)
    prob = np.abs(dft @ (amp[:, None] * phase)) ** 2 / dim
    err = (x[:, None] - offsets[None, :] + 0.5 * dim) % dim - 0.5 * dim
    return prob, err


@pytest.mark.parametrize("u", [0.0, 0.1, 0.25, 0.37, 0.5, 0.8])
def test_sine_window_law_identities(u):
    """The lattice law P(n - u) (sum over |n| <= 2e4): mass 1 and mean 0 to 1e-12, variance 1/4 to 1e-5 for every
    offset (Poisson summation); P equals 8 cos^2(pi D)/(pi^2 (1 - 4 D^2)^2) wherever that form is defined and 1/2
    at |D| = 1/2."""
    d = np.arange(-20000, 20001) - u
    p = cl.sine_window_law(d)
    assert abs(p.sum() - 1.0) <= 1e-12
    assert abs(float(np.sum(d * p))) <= 1e-12
    assert abs(float(np.sum(d * d * p)) - 0.25) <= 1e-5
    ok = np.abs(np.abs(d) - 0.5) > 1e-6
    closed = 8.0 * np.cos(np.pi * d[ok]) ** 2 / (np.pi ** 2 * (1.0 - 4.0 * d[ok] ** 2) ** 2)
    assert np.allclose(p[ok], closed, rtol=1e-10, atol=1e-24)
    assert cl.sine_window_law(np.array([0.5, -0.5]))[0] == pytest.approx(0.5, rel=1e-15)
    with pytest.raises(ValueError):
        cl.sine_window_law(np.array([np.nan]))


def test_sine_window_tails():
    """The heavy tails of F34 on 401 offsets u in [0, 1]: sup_u P(|D| > 5) = 2.84e-4 (<= 2.9e-4), zero at u = 1/2;
    the |n| <= 60 truncation of the sampler lowers the variance by at most 1.7e-3 (0 at u = 1/2)."""
    d0 = np.arange(-20000, 20001)
    tails, deficits = [], []
    for u in np.linspace(0.0, 1.0, 401):
        p = cl.sine_window_law(d0 - u)
        tails.append(float(p[np.abs(d0 - u) > 5].sum()))
        deficits.append(0.25 - _lattice_moments(u)[3])
    assert 2.8e-4 <= max(tails) <= 2.9e-4 and tails[200] <= 1e-30
    assert 0.0 <= min(deficits) + 1e-15 and max(deficits) <= 1.7e-3 and abs(deficits[200]) <= 1e-15


def test_sine_window_finite_register():
    """The finite register (direct summation): kappa(M) = RMS phase error/(pi/M) averaged over the offset is
    0.9736, 0.9934, 0.9983 at M = 63, 255, 1023 within 5e-4 (M3's Appendix A), and the outcome law converges to
    the continuum law of sine_window_law: |P_D(m - u) - P(m - u)| <= 1/D within 5.5 grid units, D = 256, 1024."""
    for t, kappa in ((6, 0.9736), (8, 0.9934), (10, 0.9983)):
        dim = 2 ** t
        offsets = np.arange(40) / 40.0
        prob, err = _finite_sine_law(dim, offsets)
        assert np.max(np.abs(prob.sum(axis=0) - 1.0)) <= 1e-12
        rms = math.sqrt(float(np.mean(np.sum(prob * err ** 2, axis=0)))) * 2.0 * math.pi / dim
        assert rms / (math.pi / (dim - 1)) == pytest.approx(kappa, abs=5e-4)
    for dim in (256, 1024):
        offsets = np.array([0.0, 0.1, 0.25, 0.37, 0.5, 0.8])
        prob, err = _finite_sine_law(dim, offsets)
        near = np.abs(err) <= 5.5
        assert np.max(np.abs(prob[near] - cl.sine_window_law(err[near]))) <= 1.0 / dim


def _lattice_moments(u: float, n_max: int = 60):
    lat = np.arange(-n_max, n_max + 1) - u
    p = cl.sine_window_law(lat)
    p = p / p.sum()
    return lat, p, float(np.sum(lat * p)), float(np.sum(lat * lat * p))


@pytest.mark.parametrize("mode", ["fixed", "random"])
def test_sine_window_sampler(mode):
    """sine_window_errors, 1e5 draws: mean 0 and variance 1/4 within 3 standard errors (fixed offsets 0, 0.37, 0.8
    and uniformly random offsets), equal within 3 SE to the moments of the truncated lattice; the frequencies of the
    lattice points |n| <= 4 within 4 SE of P (fixed offsets); one uniform consumed per draw."""
    n = 100000
    gen = np.random.default_rng(99)
    cases = [np.full(n, u) for u in (0.0, 0.37, 0.8)] if mode == "fixed" else [gen.random(n)]
    for offsets in cases:
        rng = np.random.default_rng(2026)
        x = cl.sine_window_errors(offsets, rng)
        ref = np.random.default_rng(2026)
        ref.random(n)
        assert rng.random() == ref.random()                       # exactly n uniforms consumed
        se_mean = x.std(ddof=1) / math.sqrt(n)
        se_var = math.sqrt(float(np.var(x ** 2, ddof=1)) / n)
        assert abs(x.mean()) <= 3.0 * se_mean
        assert abs(x.var(ddof=1) - 0.25) <= 3.0 * se_var
        if mode == "fixed":
            lat, p, mean_t, var_t = _lattice_moments(float(offsets[0]))
            assert abs(x.mean() - mean_t) <= 3.0 * se_mean and abs(x.var(ddof=1) - var_t) <= 3.0 * se_var
            assert var_t == pytest.approx(0.25, abs=2e-3)
            central = np.abs(lat) <= 4.5
            freq = np.array([np.mean(np.isclose(x, val, atol=1e-9)) for val in lat[central]])
            assert np.all(np.abs(freq - p[central]) <= 4.0 * np.sqrt(p[central] * (1.0 - p[central]) / n) + 1e-12)
            assert np.all(np.isin(np.round(x + offsets[0], 9), np.arange(-60, 61)))
    with pytest.raises(ValueError):
        cl.sine_window_errors(np.array([1.5]), np.random.default_rng(0))


def test_ae_estimator_allocation(h4, psi1):
    """F34/F11: the AE allocation at B_k has sum_e m_e eps_e^2 = B_k^2 to 1e-12 and eps_e = t (alpha_e/m_e)^(1/3)
    (N_e = 2 pi kappa alpha_e/eps_e from the event); the estimate is O^_e = -alpha_e cos(2 theta_e + 2 pi
    Delta_e/D_e) with the residual stream's lattice draw at the fixed offset u_e = frac(D_e 2 theta_e/2 pi), and
    V^_e = (eps_e/kappa)^2 (1 - (O^_e/alpha_e)^2); Var(O^_e) = (eps_e sin 2 theta_e)^2 exactly on the truncated
    lattice up to its tail deficit (1%, fixed offsets and averaged over random ones) and by Monte Carlo (1e4 draws
    per element, fixed and random offsets) within 3 standard errors (family-wise over the elements). The law is
    heavy-tailed (fourth moment ~ n_max/pi^2), so the standard error of a sample variance is the exact one,
    ((mu_4 - sigma^4)/n)^(1/2) from the lattice law, not its sample estimate."""
    a, norm = psi1["a"], psi1["norm"]
    spec = cl.LoopSpec(route="ae_x", controller="oracle", r=1.0)
    est, _ = _estimator(h4, spec, norm0=1.0, seed=3)
    rng_copy = np.random.default_rng()
    rng_copy.bit_generator.state = est.rng_residual.bit_generator.state
    a_est, var = est(psi1["rd"], psi1["psi"], 1)
    ev = est.events[-1]
    assert ev.route == "ae_x" and ev.shots == 0.0 and ev.budget == norm
    alpha, m = np.asarray(h4.normalizations["alpha_x"]), h4.m
    assert np.all(alpha > 0.0)
    eps = 2.0 * math.pi * spec.kappa * alpha / ev.per_element
    assert float(np.sum(m * eps ** 2)) == pytest.approx(ev.budget ** 2, rel=1e-12)
    t = ev.budget / math.sqrt(float(np.sum(np.cbrt(m) * np.cbrt(alpha) ** 2)))
    assert np.allclose(eps, t * np.cbrt(alpha / m), rtol=1e-12, atol=0.0)
    assert ev.applications == pytest.approx(float(ev.per_element.sum()), rel=1e-15)

    two_theta = np.arccos(-a / alpha)
    d_e = 0.5 * ev.per_element
    y = d_e * two_theta / (2.0 * math.pi)
    delta = cl.sine_window_errors(y - np.floor(y), rng_copy)
    o_hat = -alpha * np.cos(two_theta + 2.0 * math.pi * delta / d_e)
    assert np.allclose(est.estimates[-1], o_hat, rtol=0.0, atol=1e-15 * float(alpha.max()))
    assert np.allclose(var, eps ** 2 * (1.0 - (est.estimates[-1] / alpha) ** 2), rtol=1e-13, atol=0.0)
    assert np.allclose(np.real(a_est[tuple(h4.elements.T)]), est.estimates[-1], rtol=0.0, atol=0.0)
    # random offsets (F38): one uniform per element from the AE-offset stream, the lattice draw from the residual stream
    spec_r = cl.LoopSpec(route="ae_x", ae_offsets="random", controller="oracle", r=1.0)
    est_r, _ = _estimator(h4, spec_r, norm0=1.0, seed=3)
    res_copy, off_copy = np.random.default_rng(), np.random.default_rng()
    res_copy.bit_generator.state = est_r.rng_residual.bit_generator.state
    off_copy.bit_generator.state = est_r.rng_offsets.bit_generator.state
    est_r(psi1["rd"], psi1["psi"], 1)
    d_r = 0.5 * est_r.events[-1].per_element
    u_r = off_copy.random(a.size)
    o_r = -alpha * np.cos(two_theta + 2.0 * math.pi * cl.sine_window_errors(u_r, res_copy) / d_r)
    assert np.allclose(est_r.estimates[-1], o_r, rtol=0.0, atol=1e-15 * float(alpha.max()))
    assert res_copy.random() == est_r.rng_residual.random() and off_copy.random() == est_r.rng_offsets.random()

    target = (eps * np.sin(two_theta)) ** 2
    u_fixed = y - np.floor(y)

    def lattice_moments(e, u):
        lat, p, _, _ = _lattice_moments(u)
        o = -alpha[e] * np.cos(two_theta[e] + 2.0 * math.pi * lat / d_e[e])
        mean = float(np.sum(p * o))
        return float(np.sum(p * (o - mean) ** 2)), float(np.sum(p * (o - mean) ** 4))

    grid = (np.arange(32) + 0.5) / 32
    exact = {"fixed": np.array([lattice_moments(e, u_fixed[e]) for e in range(a.size)]),
             "random": np.array([np.mean([lattice_moments(e, u) for u in grid], axis=0) for e in range(a.size)])}
    for mode in exact:
        assert np.allclose(exact[mode][:, 0], target, rtol=1e-2, atol=0.0), mode

    n = 10000
    z_crit = _z_family(2 * a.size)
    gen = np.random.default_rng(5)
    for mode in ("fixed", "random"):
        u = np.repeat(u_fixed, n) if mode == "fixed" else gen.random(n * a.size)
        x = cl.sine_window_errors(u, gen).reshape(a.size, n)
        o = -alpha[:, None] * np.cos(two_theta[:, None] + 2.0 * math.pi * x / d_e[:, None])
        var_e, mu4 = exact[mode][:, 0], exact[mode][:, 1]
        se = np.sqrt((mu4 - var_e ** 2) / n)
        assert np.all(np.abs(o.var(axis=1, ddof=1) - target) <= z_crit * se), mode


# ------------------------------------------------------------------------- F35: controllers
def test_controllers(h4, psi1):
    """F35: the oracle plans B_k = r ||A_k||_m exactly; lagged B_1 = r rho_1 ||A_0||_m and B_k = r rho n^_{k-1}
    (k >= 2) in a run; n^_k^2 = max(||A^||_m^2 - sum m V^, f_fl ||A^||_m^2) from the run's own estimates;
    ||A^||_m^2 - sum_e m_e V^_e (floor off) is unbiased for ||A||_m^2 (Monte Carlo, 3 SE; direct and AE routes)
    with the precision (4 r^2/K_1 + 2 r^4/K_2)^(1/2) of F35 (direct route, within 10%)."""
    a, norm = psi1["a"], psi1["norm"]
    for r in (0.5, 1.0, 3.0):
        ctrl = cl.PrecisionController(cl.LoopSpec(controller="oracle", r=r), h4, 2.0)
        assert ctrl.planned_budget(1, norm) == r * norm and ctrl.planned_budget(3, 0.7) == r * 0.7
        lag = cl.PrecisionController(cl.LoopSpec(controller="lagged", r=r, rho1=0.4, rho=1.5), h4, 2.0)
        assert lag.planned_budget(1, 123.0) == r * 0.4 * 2.0
        with pytest.raises(ValueError, match="needs the estimate"):
            lag.planned_budget(2, 1.0)
    act, ir = h4.active, h4.irreps
    rd0 = compute_residual(act, normalize(Hamiltonian(act).space.hf()), Hamiltonian(act))
    a0 = np.real(sy.mask_pair_tensor(rd0.A, ir)[tuple(h4.elements.T)])
    norm0 = math.sqrt(float(np.sum(h4.m * a0 * a0)))
    spec = cl.LoopSpec(controller="lagged", r=1.0)
    rec = cl.run_closed_loop(h4, spec, "unit", replicate_index=5, max_iter=5)
    assert rec.budgets[0] == pytest.approx(spec.r * spec.rho1 * norm0, rel=1e-14)
    for k in range(1, len(rec.budgets)):
        assert rec.budgets[k] == spec.r * spec.rho * rec.norm_estimates[k - 1]
    for ev, a_hat in zip([e for e in rec.events if e.kind == "residual"], rec.estimates):
        raw = float(np.sum(h4.m * a_hat ** 2))
        n2 = max(raw - float(np.sum(h4.m * h4.variances["v_direct"])) / ev.shots, spec.norm_floor * raw)
        assert ev.norm_estimate ** 2 == pytest.approx(n2, rel=1e-12)
        assert ev.norm_estimate ** 2 >= spec.norm_floor * raw * (1.0 - 1e-12)

    n = 6000
    v = h4.variances["v_direct"]
    for route in ("direct_mg", "ae_x"):
        est, ctrl = _estimator(h4, cl.LoopSpec(controller="oracle", r=1.0, route=route, norm_floor=1.0), 1.0,
                               seed=11)
        vals, floored = [], 0
        for _ in range(n):
            _, var = est(psi1["rd"], psi1["psi"], 1)
            vals.append(cl.debiased_norm_square(est.estimates[-1], var, h4.m))
            raw = float(np.sum(h4.m * est.estimates[-1] ** 2))
            assert est.events[-1].norm_estimate == pytest.approx(math.sqrt(raw), rel=1e-12)   # floor 1 binds always
        vals = np.array(vals)
        assert abs(vals.mean() - norm ** 2) <= 3.0 * vals.std(ddof=1) / math.sqrt(n), route
        if route == "direct_mg":
            k1 = norm ** 2 * h4.variance_m / float(np.sum(h4.m ** 2 * a ** 2 * v))
            k2 = h4.variance_m ** 2 / float(np.sum(h4.m ** 2 * v ** 2))
            assert vals.std(ddof=1) / norm ** 2 == pytest.approx(math.sqrt(4.0 / k1 + 2.0 / k2), rel=0.1)
    with pytest.raises(ValueError):
        cl.debiased_norm_square(np.ones(3), np.ones(2), np.ones(3))


# ------------------------------------------------------------------------- F38: statistics
def test_summarize_statistics(h4):
    """F38: (a) the 95% percentile bootstrap (2000 resamples, seed 12345) of the mean of n = 100 paired differences
    T(1) - T(2) ~ N(mu, sigma^2) covers mu in [0.92, 0.98] of 200 synthetic sets; (b) wilson_interval equals its
    closed form; (c) stopping_rules on a hand-worked case; (d) summarize on replicates with planted overlaps: T(k) =
    [C_loop(k) + (C(Psi~_k) + C_QPE)/p~_k]/C_QPE by hand, the catastrophic share, the fixed-K and cost-gate
    regrets, P(T(k) < T(1)) with its Wilson interval, the shared bootstrap and the medians."""
    gen = np.random.default_rng(31)
    mu, sigma, covered = 0.3, 1.0, 0
    for _ in range(200):
        diff = mu + sigma * gen.standard_normal(100)
        v = cl.paired_verdict(diff, np.zeros(100))
        covered += v["ci"][0] <= mu <= v["ci"][1]
    assert 0.92 <= covered / 200 <= 0.98

    z = statistics.NormalDist().inv_cdf(0.975)
    for x, n in ((0, 10), (3, 10), (10, 10), (57, 100), (1, 1)):
        ph = x / n
        c = (ph + z * z / (2 * n)) / (1 + z * z / n)
        h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        lo, hi = cl.wilson_interval(x, n)
        assert lo == pytest.approx(max(0.0, c - h), abs=1e-14) and hi == pytest.approx(min(1.0, c + h), abs=1e-14)
    with pytest.raises(ValueError):
        cl.wilson_interval(4, 3)

    # (c) three replicates, K = 3; minima 3, 2, 1 (mean 2)
    t = np.array([[5.0, 3.0, 4.0, 6.0],
                  [5.0, 4.0, 2.0, 3.0],
                  [5.0, 2.5, 3.0, 1.0]])
    c_iter = np.array([[0.0, 0.2, 0.5],
                       [0.0, 0.5, 2.0],
                       [0.0, 2.0, 0.1]])
    s = cl.stopping_rules(t, c_iter)
    assert s["oracle"]["mean"] == 2.0 and s["oracle"]["k_hist"] == [0, 1, 1, 1]
    assert s["fixed_1"]["mean"] == pytest.approx(9.5 / 3) and s["fixed_1"]["regret"] == pytest.approx(3.5 / 3)
    assert s["fixed_2"]["regret"] == pytest.approx(3.0 / 3) and s["fixed_3"]["regret"] == pytest.approx(4.0 / 3)
    assert "fixed_4" not in s
    # gate 0.3: row 0 runs iteration 1 (0.2) and stops before 2 (0.5): Psi_2; rows 1, 2 stop before 1: Psi_1
    assert s["cost_gate_0.3"]["k_hist"] == [0, 2, 1, 0]
    assert s["cost_gate_0.3"]["regret"] == pytest.approx(((4 - 3) + (4 - 2) + (2.5 - 1)) / 3)
    # gate 1: row 0 runs both (Psi_3), row 1 stops before 2 (2.0 > 1): Psi_2, row 2 before 1: Psi_1
    assert s["cost_gate_1"]["k_hist"] == [0, 1, 1, 1]
    assert s["cost_gate_1"]["regret"] == pytest.approx(((6 - 3) + 0 + (2.5 - 1)) / 3)
    # gate 3: every row runs to Psi_3
    assert s["cost_gate_3"]["k_hist"] == [0, 0, 0, 3]
    assert s["cost_gate_3"]["regret"] == pytest.approx(s["fixed_3"]["regret"])

    # (d) 12 replicates of h4 (lagged r = 1, 4 steps); overlaps planted on four of them
    recs = cl.replicate(h4, cl.LoopSpec(controller="lagged", r=1.0), "unit", n=12, max_iter=4)
    planted = []
    for j, rec in enumerate(recs):
        p = np.array(rec.overlaps)
        if j in (1, 4):
            p[2] = 0.49 * p[1]                       # a catastrophic drop at k = 1 -> 2
        if j == 7:
            p[3] = 0.51 * p[2]                       # a drop that is not catastrophic
        if j == 9:
            p[1:4] = [0.99, 0.6, 0.8]                # p~_2 < p~_1 but not below half
        planted.append(dataclasses.replace(rec, overlaps=p))
    costs = [cl.closed_loop_costs(h4, rec) for rec in planted]
    summ = cl.summarize(h4, planted, costs)
    c_qpe = h4.c_qpe
    kp1 = planted[0].n_steps + 1
    tt = np.empty((12, kp1))
    for i, (rec, cs) in enumerate(zip(planted, costs)):
        citer = [c.toffolis for c in cs]
        for k in range(kp1):
            c_loop = sum(citer[1:k])                 # C_iter(0) = 0: the classical first step
            tt[i, k] = (c_loop + (rec.prep_cumulative[k] + c_qpe) / rec.overlaps[k]) / c_qpe
    capped = any("angle_cap" in f for r in planted for f in r.flags)
    assert summ["n"] == 12 and summ["k_max"] == kp1 - 1 and summ["rule"] == "unit"
    assert summ["realizable"] == (not capped) and summ["non_realizable"] == (["angle_cap"] if capped else [])
    for k, row in enumerate(summ["per_k"]):
        assert row["T_mean"] == pytest.approx(tt[:, k].mean(), rel=1e-12)
        assert row["T_median"] == pytest.approx(float(np.median(tt[:, k])), rel=1e-12)
        wins = int(np.sum(tt[:, k] < tt[:, 1]))
        assert row["beats_psi1"] == wins / 12 and row["beats_psi1_ci"] == list(cl.wilson_interval(wins, 12))
        mean, lo, hi = cl.bootstrap_mean(tt[:, k])
        assert row["T_ci"] == pytest.approx([lo, hi], rel=1e-12)
        if k < kp1 - 1:
            assert row["diff_mean"] == pytest.approx(float(np.mean(tt[:, k] - tt[:, k + 1])), rel=1e-12, abs=1e-12)
            assert row["pays"] == (row["diff_ci"][0] > 0.0)
        assert row["p_median"] == pytest.approx(float(np.median([r.overlaps[k] for r in planted])), rel=1e-14)
    assert summ["catastrophic_share"] == pytest.approx(2.0 / 12)
    assert summ["catastrophic_ci"] == list(cl.wilson_interval(2, 12))
    best = tt.min(axis=1)
    assert summ["optimum"]["mean"] == pytest.approx(best.mean(), rel=1e-12)
    for kk in (1, 2, 3):
        assert summ["stopping"][f"fixed_{kk}"]["regret"] == pytest.approx(float(np.mean(tt[:, kk] - best)),
                                                                          rel=1e-12, abs=1e-12)
    for theta in (0.3, 1.0, 3.0):
        picks = []
        for i, cs in enumerate(costs):
            k = 1
            while k < kp1 - 1 and cs[k].toffolis / c_qpe <= theta:
                k += 1
            picks.append(tt[i, k] - best[i])
        assert summ["stopping"][f"cost_gate_{theta:g}"]["regret"] == pytest.approx(float(np.mean(picks)),
                                                                                   rel=1e-12, abs=1e-12)
    assert summ["flag_counts"]["first:second_order"] == 12


def test_paired_verdict():
    """F38's rule on constructed cases: "pays" iff the percentile-bootstrap interval of the mean difference lies
    above 0 (a positive mean whose interval reaches 0 does not pay; an interval touching 0 does not pay); the
    interval is the (2.5%, 97.5%) quantile pair of the means of 2000 resamples of default_rng(12345)."""
    n = 40
    v = cl.paired_verdict(np.full(n, 2.0), np.full(n, 1.5))
    assert v["pays"] and v["mean"] == 0.5 and v["ci"] == [0.5, 0.5] and v["share_positive"] == 1.0
    assert not cl.paired_verdict(np.full(n, 1.0), np.full(n, 1.5))["pays"]
    v = cl.paired_verdict(np.array([1.0, -0.8] * 10), np.zeros(20))         # mean 0.1, standard error 0.2
    assert v["mean"] == pytest.approx(0.1) and v["ci"][0] < 0.0 < v["ci"][1] and not v["pays"]
    gen = np.random.default_rng(8)
    strong = 1.0 + 0.5 * gen.standard_normal(100)
    v = cl.paired_verdict(strong + 3.0, np.full(100, 3.0))
    assert v["ci"][0] > 0.0 and v["pays"]
    idx = np.random.default_rng(12345).integers(0, 100, size=(2000, 100))
    means = np.sort((strong + 3.0 - 3.0)[idx].mean(axis=1))
    for q, got in zip((0.025, 0.975), v["ci"]):
        pos = q * (means.size - 1)
        lo_i = int(math.floor(pos))
        ref = means[lo_i] + (pos - lo_i) * (means[min(lo_i + 1, means.size - 1)] - means[lo_i])
        assert got == pytest.approx(ref, rel=1e-12)
    touching = cl.paired_verdict([0.0, 0.0, 0.0, 0.0, 1.0], [0.0] * 5)
    assert touching["ci"][0] == 0.0 and not touching["pays"]
    single = cl.paired_verdict([1.0], [0.0])
    assert not single["pays"] and all(math.isnan(x) for x in single["ci"])
    with pytest.raises(ValueError):
        cl.paired_verdict([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        cl.paired_verdict([1.0, np.inf], [1.0, 1.0])


def test_scheme_verdict():
    """F38 under the QPE-side schemes on constructed replicates: per scheme with a finite cost in every replicate,
    paired_verdict of T_s(k) - T_s(k + 1) exactly; s*(k) and s*(k + 1) are best_realizable of the replicate means
    (an oracle scheme or bound never, a scheme missing in one replicate never), so the verdict of record pairs
    T_{s*(k)}(k) with T_{s*(k+1)}(k + 1) even where s* changes; a mean dominated by one replicate decides s*."""
    from thc_engine_cqe.ft import amplify as amp

    gen = np.random.default_rng(21)
    n = 30
    rep_k, rep_k1 = 4.0 + 0.2 * gen.standard_normal(n), 3.6 + 0.2 * gen.standard_normal(n)
    l_k, l_k1 = 1.60 + 0.01 * gen.standard_normal(n), 1.45 + 0.01 * gen.standard_normal(n)
    s_k1 = 1.40 + 0.01 * gen.standard_normal(n)                  # 2s_L is the cheapest after the iteration
    costs_k = [{"rep": a, "bbht": a + 0.5, "lt_bbht_L": b, "2s_L": b + 0.3, "aa": 0.1, "lt@oracle_U": 0.05,
                "fp": None} for a, b in zip(rep_k, l_k)]
    costs_k1 = [{"rep": a, "bbht": a + 0.5, "lt_bbht_L": b, "2s_L": c, "aa": 0.1, "lt@oracle_U": 0.05, "fp": None}
                for a, b, c in zip(rep_k1, l_k1, s_k1)]
    costs_k[3]["2s"] = 0.5                                        # one replicate only: never a complete scheme
    v = cl.scheme_verdict(costs_k, costs_k1)
    assert v["n"] == n and v["best_k"] == "lt_bbht_L" and v["best_k1"] == "2s_L"
    assert set(v["by_scheme"]) == {"rep", "bbht", "lt_bbht_L", "2s_L", "aa", "lt@oracle_U"}
    assert v["by_scheme"]["rep"] == cl.paired_verdict(rep_k, rep_k1)
    assert v["means_k"]["lt_bbht_L"] == pytest.approx(float(np.mean(l_k)), rel=1e-15)
    assert v["best"] == cl.paired_verdict(l_k, s_k1) and v["pays"] and v["best"]["mean"] == pytest.approx(0.2, abs=0.01)
    assert amp.best_realizable(v["means_k1"]) == v["best_k1"]
    # an iteration that does not pay at the best realizable scheme although it pays under repetition
    worse = [dict(c, lt_bbht_L=b + 0.05, **{"2s_L": b + 0.06}) for c, b in zip(costs_k1, l_k)]
    v = cl.scheme_verdict(costs_k, worse)
    assert v["by_scheme"]["rep"]["pays"] and v["best_k1"] == "lt_bbht_L" and not v["pays"]
    assert v["best"] == cl.paired_verdict(l_k, l_k + 0.05)
    # one heavy replicate moves the mean and with it s*
    heavy = [dict(c) for c in costs_k]
    heavy[0]["lt_bbht_L"] = 1e3
    assert cl.scheme_verdict(heavy, costs_k1)["best_k"] == "2s_L"
    # no realizable scheme with a cost in every replicate: no verdict of record
    v = cl.scheme_verdict([{"aa": 1.0}, {"aa": 2.0}], [{"aa": 0.5}, {"aa": 0.5}])
    assert v["best"] is None and not v["pays"] and v["by_scheme"]["aa"]["pays"]
    with pytest.raises(ValueError):
        cl.scheme_verdict(costs_k, costs_k1[:-1])
    with pytest.raises(ValueError):
        cl.scheme_verdict([], [])


# ------------------------------------------------------------------------- F36: realized costs
def _routes_at(h4, psi, prep, r):
    """M3's F14 routes at psi (ft.residual_inputs and ft.residual), at r."""
    problem = ri.build_problem(h4.active, psi, h4.elements, "spin_free", variances=h4.variances,
                               normalizations=h4.normalizations, deltas=M3.aux_deltas)
    walk = h4.qpe.encoding.walk_params()
    routes = compare_routes(problem, prep, walk, params=FTParameters(r=r))
    direct = next(c for c in routes if c.route == "direct_mg")
    ae = next(c for c in routes if c.route == "ae_x" and c.params["kappa"] == M3.kappa)
    return direct, ae


@pytest.mark.parametrize("route", ["direct_mg", "ae_x", "best"])
def test_realized_cost_identity(h4, route):
    """F36: direct route T = S_k C(Psi~_k) exactly; AE route T = sum_e N_e (C + C_obs,e + Q_A,e/2) with C_obs,e =
    2 L_X,e and Q_A,e = Q(Psi~_k) + ceil(log2 L_X,e) + chi + 5 from the allocation at B_k; both equal
    compare_routes at r_k = B_k/||A_k||_m on the realized state (1e-12); "best" measures the cheaper of the two at
    the plan; C(Psi~_k), Q(Psi~_k) are stateprep.prep_cost of the realized steps."""
    spec = cl.LoopSpec(route=route, controller="lagged", r=1.0)
    rec = cl.run_closed_loop(h4, spec, "unit", replicate_index=2, max_iter=4, keep_states=True)
    costs = cl.closed_loop_costs(h4, rec)
    prep = prep_cost(rec.directions, rec.eps, irreps=h4.irreps)
    assert np.array_equal(rec.prep_cumulative, prep.cumulative)
    for k in range(1, rec.n_steps + 1):
        q = max(2 * h4.active.norb, max(elementwise_step_cost(d, e, irreps=h4.irreps).logical_qubits
                                        for d, e in zip(rec.directions[:k], rec.eps[:k])))
        assert rec.prep_qubits[k] == q
    n_terms = np.asarray(h4.normalizations["n_terms_x"])
    for k in range(1, rec.n_steps):
        ev = rec.residual_event(k)
        c = costs[k]
        prep_k = PrepParams(float(rec.prep_cumulative[k]), int(rec.prep_qubits[k]))
        direct, ae = _routes_at(h4, rec.states[k], prep_k, ev.r_realized)
        assert c.residual[0].route == "direct_mg" and c.residual[1].route == "ae_x"
        assert c.residual[0].toffolis == pytest.approx(direct.toffolis, rel=1e-12)
        assert c.residual[1].toffolis == pytest.approx(ae.toffolis, rel=1e-12)
        assert c.residual[1].logical_qubits == ae.logical_qubits
        assert c.best == ev.route and c.toffolis == c.residual_toffolis
        if route == "best":
            assert ev.route == ("direct_mg" if direct.toffolis <= ae.toffolis else "ae_x")
        if ev.route == "direct_mg":
            assert c.residual_toffolis == ev.shots * float(rec.prep_cumulative[k])
            assert c.shots == ev.shots
        else:
            q_a = prep_k.logical_qubits + np.ceil(np.log2(np.maximum(n_terms, 1))) + M3.chi + 5
            ref = float(np.sum(ev.per_element * (prep_k.toffolis + 2.0 * n_terms + 0.5 * q_a)))
            assert c.residual_toffolis == pytest.approx(ref, rel=1e-12)
            assert c.residual_toffolis_practical == pytest.approx(M3.kappa_practical * ref, rel=1e-12)
        rd = compute_residual(h4.active, rec.states[k], Hamiltonian(h4.active))
        a_k = np.real(sy.mask_pair_tensor(rd.A, h4.irreps)[tuple(h4.elements.T)])
        assert c.residual_norm_m == pytest.approx(math.sqrt(float(np.sum(h4.m * a_k * a_k))), rel=1e-12)


def test_costs_feed_pipeline(h4):
    """The cost lists are pipeline.IterationCost: loop_totals and machines_for_campaigns accept them, k = 0 is free;
    the candidates rule charges 11 sampled energies per measured iteration at eps_ls with S_E = V_E(Psi~_{k+1})/
    eps_ls^2 shots each, its residual stream untouched; handoff_inputs is JSON-able and its rows are handoff_table's
    of the converted record."""
    c_qpe = h4.c_qpe
    for rule in ("unit", "candidates"):
        rec = cl.run_closed_loop(h4, cl.LoopSpec(controller="lagged", r=1.0), rule, replicate_index=1, max_iter=4,
                                 keep_states=True)
        costs = cl.closed_loop_costs(h4, rec)
        assert len(costs) == rec.n_steps
        c0 = costs[0]
        assert (c0.k, c0.toffolis, c0.best, c0.campaigns, c0.machines_7d) == (0, 0.0, "classical", (), 0)
        tot = pipeline.loop_totals(costs, None, c_qpe)
        assert tot["total"] == pytest.approx(sum(c.toffolis for c in costs), rel=1e-14)
        assert tot["total_over_qpe"] == pytest.approx(tot["total"] / c_qpe, rel=1e-14)
        assert tot["machines_7d"] == pipeline.machines_for_campaigns([x for c in costs for x in c.campaigns])
        for c in costs[1:]:
            assert c.machines_7d == pipeline.machines_for_campaigns(c.campaigns) and c.toffolis > 0.0
        if rule == "candidates":
            grid = M3.eps_candidates
            v_split = {}
            for c in costs[1:]:
                en = [e for e in rec.events if e.kind == "energy" and e.k == c.k]
                assert len(en) == len(grid) and all(e.budget == M3.eps_ls for e in en)
                assert sorted(e.eps for e in en) == pytest.approx(sorted(grid))
                v = ri.energy_variances(h4.active, rec.states[c.k + 1])[h4.energy_split]
                v_split[c.k] = v
                assert all(e.shots == pytest.approx(v / M3.eps_ls ** 2, rel=1e-14) for e in en)
                assert c.line_search.route == "energy_sampling" and c.line_search_ae > 0.0
                assert c.toffolis == pytest.approx(c.residual_toffolis + c.line_search.toffolis, rel=1e-14)
            assert tot["line_search"] == pytest.approx(sum(c.line_search.toffolis for c in costs[1:]), rel=1e-14)
        else:
            assert all(c.line_search is None for c in costs)
        h = cl.handoff_inputs(h4, rec, costs)
        json.dumps(scaling.to_json(h), allow_nan=False)
        trec = rec.to_trajectory_record(h4)
        rows = handoff_table(trec, rec.prep_cumulative.astype(float), [c.toffolis for c in costs], h4.qpe)
        assert [r.cost_expected for r in rows] == [x["cost_expected"] for x in h["handoff"]]
        assert h["c_loop"] == pytest.approx([0.0] + list(np.cumsum([0.0] + [c.toffolis for c in costs[1:]])))
        assert h["k"] == list(range(rec.n_steps + 1)) and h["r_realized"][0] is None
        assert h["r_realized"][1:rec.n_steps] == [e.r_realized for e in rec.events if e.kind == "residual"]
        assert trec.first_step == "second_order" and np.array_equal(trec.overlaps, rec.overlaps)
    with pytest.raises(ValueError, match="need C_iter"):
        cl.handoff_inputs(h4, rec, costs[:-1])


# ------------------------------------------------------------------------- F38: common random numbers
def test_common_random_numbers(h4, psi1):
    """F38: the streams are SeedSequence(20261108, spawn_key=(crc32(name), j)).spawn(3) (residual, AE-offset,
    energy); configurations that differ only in r draw identical standard normals at k = 1, scaled by their
    budgets; a rule with energies (candidates) draws the same residual at k = 1; replicates and systems differ."""
    j = 4
    seq = np.random.SeedSequence(20261108, spawn_key=(zlib.crc32(b"h4"), j)).spawn(3)
    mine = cl.replicate_streams("h4", j)
    for s, g in zip(seq, mine):
        assert np.random.default_rng(s).random() == g.random()
    v = h4.variances["v_direct"]
    z = {}
    for r in (1.0, 3.0):
        rec = cl.run_closed_loop(h4, cl.LoopSpec(controller="oracle", r=r), "unit", replicate_index=j, max_iter=3)
        assert rec.seed == (20261108, zlib.crc32(b"h4"), j)
        ev = rec.residual_event(1)
        assert ev.budget == pytest.approx(r * psi1["norm"], rel=1e-13)
        z[r] = (rec.estimates[0] - psi1["a"]) / np.sqrt(v / ev.shots)
    assert np.allclose(z[1.0], z[3.0], rtol=0.0, atol=1e-9)
    unit = cl.run_closed_loop(h4, cl.LoopSpec(controller="oracle", r=1.0), "unit", replicate_index=j, max_iter=2)
    cand = cl.run_closed_loop(h4, cl.LoopSpec(controller="oracle", r=1.0), "candidates", replicate_index=j,
                              max_iter=2)
    assert any(e.kind == "energy" for e in cand.events)
    assert np.array_equal(unit.estimates[0], cand.estimates[0])
    other = cl.run_closed_loop(h4, cl.LoopSpec(controller="oracle", r=1.0), "unit", replicate_index=j + 1,
                               max_iter=2)
    assert not np.array_equal(unit.estimates[0], other.estimates[0])
    assert cl.replicate_streams("h4", 0)[0].random() != cl.replicate_streams("h4_r1.8", 0)[0].random()
    reps = cl.replicate(h4, cl.LoopSpec(controller="oracle", r=1.0), "unit", n=3, max_iter=2)
    assert [r.seed[2] for r in reps] == [0, 1, 2]
    again = cl.run_closed_loop(h4, cl.LoopSpec(controller="oracle", r=1.0), "unit", replicate_index=2, max_iter=2)
    assert np.array_equal(reps[2].energies, again.energies)


# ------------------------------------------------------------------------- F37 / F25: compiled propagation
@pytest.mark.parametrize("name", ["h4", "n2s"])
@pytest.mark.parametrize("rounded", [True, False])
def test_compiled_step_equals_replay(records, name, rounded):
    """K successive compiled_step calls from HF give the K + 1 states of replay_elementwise to 1e-12."""
    act, ir, rec = records[name]
    replay = circuits.replay_elementwise(act, rec.directions, rec.eps, irreps=ir, rounded=rounded)
    ci = FCISpace.of(act).hf()
    for k, (d, eps) in enumerate(zip(rec.directions, rec.eps)):
        ci = circuits.compiled_step(act, ci, d, eps, irreps=ir, rounded=rounded)
        assert ci.shape == replay[k + 1].shape
        assert np.max(np.abs(ci - replay[k + 1])) <= 1e-12
    assert abs(np.linalg.norm(ci) - 1.0) <= 1e-12


@pytest.mark.parametrize("name", ["h4", "n2s"])
def test_sliced_compiled_step_equals_replay_of_slices(records, name):
    """s slices of step 0 from HF (s calls at eps/s from the current state) equal the replay of
    [d]*s at [eps/s]*s (F25), and the slicing error falls with s on n2s."""
    act, ir, rec = records[name]
    d, eps = rec.directions[0], rec.eps[0]
    errors = []
    for s in (1, 2, 4):
        replay = circuits.replay_elementwise(act, [d] * s, [eps / s] * s, irreps=ir)
        ci = FCISpace.of(act).hf()
        for _ in range(s):
            ci = circuits.compiled_step(act, ci.ravel(), d, eps / s, irreps=ir)
        assert np.max(np.abs(ci - replay[-1])) <= 1e-12
        errors.append(float(np.linalg.norm(ci - rec.states[1])))
    if name == "n2s":
        assert errors[0] > errors[1] > errors[2]


def test_compiled_step_validates_the_state(records):
    act, ir, rec = records["h4"]
    with pytest.raises(ValueError, match="ci has shape"):
        circuits.compiled_step(act, np.zeros(7), rec.directions[0], rec.eps[0], irreps=ir)


@pytest.mark.parametrize("slices", [None, 2])
def test_compiled_loop(h4, slices):
    """F37 with the F25 rule (or fixed s): Psi~_{k+1} is s_k compiled_step calls at eps_k/s_k from Psi~_k; the
    compile error is ||Psi~_{k+1} - exp(eps_k D_k) Psi~_k||; under the rule s_k is the smallest of {1, 2, 4, 8, 16}
    with error <= 0.05 ("slice_cap" otherwise); "pair_inconsistency" iff the error exceeds 0.05; C(Psi~_{k+1}) -
    C(Psi~_k) = s_k C_step(d_k, eps_k/s_k); p~_k and <S^2> are those of the compiled states."""
    act, ir = h4.active, h4.irreps
    ham = Hamiltonian(act)
    spec = cl.LoopSpec(controller="oracle", r=1.0, propagation="compiled", slices=slices)
    rec = cl.run_closed_loop(h4, spec, "unit", replicate_index=1, max_iter=4, keep_states=True)
    assert rec.compile_errors[0] == 0.0 and len(rec.slices) == rec.n_steps
    for k in range(rec.n_steps):
        d, eps, s = rec.directions[k], float(rec.eps[k]), rec.slices[k]
        assert s in cl.SLICE_CHOICES and (slices is None or s == slices)

        def compiled(n_slices):
            psi = rec.states[k]
            for _ in range(n_slices):
                psi = compiled_step(act, psi, d, eps / n_slices, irreps=ir)
            return normalize(np.asarray(psi).reshape(rec.states[k].shape))

        exact = propagate(ham, d, rec.states[k], eps)
        assert np.max(np.abs(rec.states[k + 1] - compiled(s))) <= 1e-12
        err = float(np.linalg.norm(rec.states[k + 1] - exact))
        assert rec.compile_errors[k + 1] == pytest.approx(err, rel=1e-9, abs=1e-14)
        if slices is None:
            assert err <= cl.ETA_SLICE or s == 16
            for smaller in (x for x in cl.SLICE_CHOICES if x < s):
                assert float(np.linalg.norm(compiled(smaller) - exact)) > cl.ETA_SLICE
        assert ("pair_inconsistency" in rec.flags[k]) == (err > cl.ETA_SLICE)
        step = elementwise_step_cost(d, eps / s, irreps=ir)
        assert rec.prep_cumulative[k + 1] - rec.prep_cumulative[k] == s * step.toffolis
    psi0 = h4.spectrum.psi0
    assert rec.overlaps == pytest.approx([abs(np.vdot(psi0, s)) ** 2 for s in rec.states], rel=1e-12)
    assert np.all(rec.s2 >= -1e-12)
    h = cl.handoff_inputs(h4, rec, cl.closed_loop_costs(h4, rec))
    assert h["compiled"] and h["compile_errors"] == pytest.approx(list(rec.compile_errors))


# ------------------------------------------------------------------------- rules with energies
def test_loop_energy_gate(h4, psi1):
    """F31 in the closed loop, by hand (h4, oracle r = 1, the residual of iteration 1 measured at Psi_1 on the circuit
    of the classical step, dE_m = -1 mHa along d_0): sigma = 0.47|dE_m|; T_E^plan = V_E/sigma^2 sum_{x in 0, 1, 2}
    (C(Psi_1) + C_step(d_0; x) + C_rot) with the planned V_E (here V_E(HF)); T_res^plan = V_m/(r ||A_1||_m)^2
    C(Psi_1) (direct_mg at the plan); the fit is used iff T_E <= theta T_res: not at theta = 0.999 T_E/T_res, at
    1.001 T_E/T_res; the log carries the planned V_E; a gate without the iteration's residual is refused."""
    act, ir = h4.active, h4.irreps
    run = run_step_rule(act, StepRule("unit"), max_iter=1, irreps=ir)
    d0, eps0 = np.asarray(run.directions[0]), float(run.eps[0])
    step0 = elementwise_step_cost(d0, eps0, irreps=ir)
    prep1 = PrepParams(float(step0.toffolis), max(2 * act.norb, int(step0.logical_qubits)))
    rd0 = compute_residual(act, normalize(Hamiltonian(act).space.hf()), Hamiltonian(act))
    a0 = np.real(sy.mask_pair_tensor(rd0.A, ir)[tuple(h4.elements.T)])
    est, _ = _estimator(h4, cl.LoopSpec(controller="oracle", r=1.0), math.sqrt(float(np.sum(h4.m * a0 * a0))), seed=0)
    est.prep = prep1
    est(psi1["rd"], psi1["psi"], 1)
    v_hf = ri.energy_variances(act, FCISpace.of(act).hf())[h4.energy_split]
    de_m, sigma = -1e-3, 0.47e-3
    steps = [c.toffolis for c in candidate_step_costs(d0, (0.0, 1.0, 2.0), irreps=ir)]
    t_e = v_hf / sigma ** 2 * sum(prep1.toffolis + st + h4.readout_toffolis for st in steps)
    t_res = h4.variance_m / psi1["norm"] ** 2 * prep1.toffolis
    for factor, want in ((0.999, False), (1.001, True)):
        gate = cl.LoopEnergyGate(h4, est, StepRule("hybrid", theta=factor * t_e / t_res), energy_variance=v_hf)
        assert gate(1, de_m, None, ((d0,), ())) is want
        (log,) = gate.log
        assert (log["k"], log["use"], log["energy_variance"]) == (1, want, v_hf)
        assert log["sigma"] == pytest.approx(sigma, rel=1e-15)
        assert log["t_energy"] == pytest.approx(t_e, rel=1e-12) and log["t_residual"] == pytest.approx(t_res, rel=1e-12)
    with pytest.raises(ValueError, match="residual measured"):
        gate(2, de_m, None, ((d0, d0), (eps0,)))


def test_hybrid_gate_in_the_loop(h4):
    """R3 in the closed loop (F31 on planned costs): at theta = 1e-9 the gate never opens and the run is the unit
    rule's (same draws, energies np.array_equal); at theta = 1e6 every measured iteration fits three energies at
    {0, t_k, 2 t_k} (t_k <= t: the growth cap may shrink the stencil) with one precision, charged as F28's sampled
    energies; a StepRule whose first step is not the classical one is refused."""
    spec = cl.LoopSpec(controller="oracle", r=1.0)
    unit = cl.run_closed_loop(h4, spec, "unit", replicate_index=0, max_iter=4)
    closed = cl.run_closed_loop(h4, spec, StepRule("hybrid", theta=1e-9), replicate_index=0, max_iter=4)
    assert closed.rule == "hybrid" and not any(e.kind == "energy" for e in closed.events)
    assert np.array_equal(closed.energies, unit.energies) and np.array_equal(closed.eps, unit.eps)
    rule = StepRule("hybrid", theta=1e6)
    rec = cl.run_closed_loop(h4, spec, rule, replicate_index=0, max_iter=4, keep_states=True)
    costs = cl.closed_loop_costs(h4, rec)
    assert rec.n_steps >= 2
    for k in range(1, rec.n_steps):
        en = sorted((e for e in rec.events if e.kind == "energy" and e.k == k), key=lambda e: e.eps)
        assert len(en) == 3 and len({e.budget for e in en}) == 1
        t_k = en[1].eps
        assert en[0].eps == 0.0 and 0.0 < t_k <= rule.parab_t and en[2].eps == pytest.approx(2.0 * t_k, rel=1e-14)
        assert costs[k].line_search is not None and costs[k].line_search.route == "energy_sampling"
        assert costs[k].toffolis == pytest.approx(costs[k].residual_toffolis + costs[k].line_search.toffolis,
                                                  rel=1e-14)
    # the realizable gate plans every iteration at the constant V_E(HF); the hand-off inputs log the charge next to it
    v_hf = ri.energy_variances(h4.active, FCISpace.of(h4.active).hf())[h4.energy_split]
    assert closed.gate_log is not None and [g["k"] for g in closed.gate_log] == list(range(1, closed.n_steps))
    assert unit.gate_log is None and cl.handoff_inputs(h4, unit, cl.closed_loop_costs(h4, unit))["gate_log"] is None
    log = cl.handoff_inputs(h4, rec, costs)["gate_log"]
    assert [g["k"] for g in log] == list(range(1, rec.n_steps)) and all(g["use"] for g in log)
    for g in log:
        k = g["k"]
        assert g["energy_variance"] == v_hf and g["t_energy_charged"] == costs[k].line_search.toffolis
        v_next = ri.energy_variances(h4.active, rec.states[k + 1])[h4.energy_split]
        assert g["energy_variance_charged"] == pytest.approx(v_next, rel=1e-12)
        assert g["t_energy"] == pytest.approx(g["t_energy_charged"] * v_hf / v_next, rel=0.05)
    json.dumps(scaling.to_json(log), allow_nan=False)
    with pytest.raises(ValueError):
        cl.run_closed_loop(h4, cl.LoopSpec(), StepRule("unit", first="line_search"), replicate_index=0)


def test_energy_estimator_noise_law(h4, monkeypatch):
    """F28, F38: FTEnergyEstimator returns E + sigma z with z the next standard normal of its stream, one per call: at
    the requested sigma, at eps_ls for run_cqe's context strings, at its fixed sigma when set; draws records (sigma, E,
    E^). In the loop (h4, candidates, oracle r = 1, replicate 3, 3 steps) the 22 energies are drawn at eps_ls = 1.6 mHa
    from the replicate's energy stream, in order."""
    est = cl.FTEnergyEstimator(h4, np.random.default_rng(5))
    z = np.random.default_rng(5).standard_normal(8)
    got = [est(-1.0, None, 1e-3) for _ in range(4)]
    assert got == pytest.approx(-1.0 + 1e-3 * z[:4], rel=0.0, abs=1e-15)
    est.sigma = 2e-3
    assert est(-1.0, None, 1e-3) == pytest.approx(-1.0 + 2e-3 * z[4], rel=0.0, abs=1e-15)
    est.sigma = None
    for i, context in ((5, "current"), (6, "candidate")):
        assert est(-2.0, None, context) == pytest.approx(-2.0 + est.eps_ls * z[i], rel=0.0, abs=1e-15)
    assert [d[0] for d in est.draws] == [1e-3] * 4 + [2e-3, est.eps_ls, est.eps_ls]
    assert [d[1] for d in est.draws] == [-1.0] * 5 + [-2.0] * 2 and [d[2] for d in est.draws[:4]] == got
    with pytest.raises(ValueError):
        est(-1.0, None, 0.0)
    made = []

    class Recording(cl.FTEnergyEstimator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(self)

    monkeypatch.setattr(cl, "FTEnergyEstimator", Recording)
    rec = cl.run_closed_loop(h4, cl.LoopSpec(controller="oracle", r=1.0), "candidates", replicate_index=3, max_iter=3)
    (eest,) = made
    sig, e, e_hat = (np.array(col) for col in zip(*eest.draws))
    n = len(eest.draws)
    assert n == 11 * (rec.n_steps - 1) == sum(ev.kind == "energy" for ev in rec.events) == 22
    assert np.all(sig == 1.6e-3)
    z_loop = cl.replicate_streams("h4", 3)[2].standard_normal(n)
    assert np.allclose((e_hat - e) / sig, z_loop, rtol=0.0, atol=1e-10)


# ------------------------------------------------------------------------- validation
def test_default_variance_state(h4):
    """F24: system_inputs takes M3's iterate-1 variances up to N = 6 and the HF state (F24 (i)) from N = 7."""
    assert [cl._default_variance_state(n) for n in (2, 4, 6)] == ["iterate1"] * 3
    assert [cl._default_variance_state(n) for n in (7, 8, 10)] == ["hf"] * 3
    assert h4.variance_state == "iterate1"


@pytest.mark.slow
def test_n8_inputs_use_hf_variances():
    """At N = 8 (h8_r3.0) system_inputs measures the allowed spin-free elements of ri.allowed_elements with the
    per-shot variances at the HF determinant (F24 (i)): v_e of ri.element_variances at HF, V_m = sum_e m_e v_e."""
    inp = cl.system_inputs("h8_r3.0")
    act, ir = inp.active, inp.irreps
    assert act.norb == 8 and inp.variance_state == "hf"
    el, _ = ri.allowed_elements(act, "spin_free", ir)
    assert np.array_equal(inp.elements, el) and inp.n_elements == len(el)
    var = ri.element_variances(act, normalize(FCISpace.of(act).hf()), el, "spin_free")
    assert np.array_equal(inp.variances["v_direct"], var["v_direct"])
    m = targets.element_multiplicities(el, "spin_free", act.norb)
    assert inp.variance_m == pytest.approx(float(np.sum(m * var["v_direct"])), rel=1e-12)


def test_loop_spec_validation():
    assert cl.LoopSpec().realizable and not cl.LoopSpec(controller="oracle").realizable
    assert not cl.LoopSpec(propagation="compiled").realizable
    assert cl.LoopSpec(propagation="compiled", slices=2).realizable
    bad = [dict(route="aux_mg"), dict(r=0.0), dict(r=-1.0), dict(r=math.inf), dict(noise=1), dict(controller="pilot"),
           dict(rho=0.0), dict(rho1=-1.0), dict(norm_floor=0.0), dict(norm_floor=1.5), dict(ae_offsets="zero"),
           dict(propagation="trotter"), dict(slices=2), dict(propagation="compiled", slices=0),
           dict(propagation="compiled", slices=1.5), dict(eps_ls=0.0), dict(kappa=0.0)]
    for kw in bad:
        with pytest.raises(ValueError):
            cl.LoopSpec(**kw)
    assert cl.LoopSpec(propagation="compiled", slices=4).slices == 4


def test_record_and_event_validation(h4):
    ev = cl.MeasurementEvent(1, "residual", "direct_mg", 0.0, 0.1, 1.0, 10.0, 0.0, None, 0.2)
    for kw in (dict(kind="trial"), dict(route="ae_aux"), dict(budget=0.0), dict(r_realized=0.0), dict(shots=-1.0),
               dict(eps=math.nan), dict(per_element=np.array([-1.0])), dict(norm_estimate=-0.1), dict(k=-1)):
        with pytest.raises(ValueError):
            dataclasses.replace(ev, **kw)
    with pytest.raises(ValueError):
        cl.MeasurementEvent(1, "energy", "energy_sampling", 0.0, 1e-3, 1.0, 10.0, 0.0, None, None)
    rec = cl.run_closed_loop(h4, cl.LoopSpec(), "unit", replicate_index=0, max_iter=3)
    for kw in (dict(rule="armijo"), dict(eps=rec.eps[:-1]), dict(overlaps=rec.overlaps[:-1]),
               dict(budgets=rec.budgets[:-1]), dict(first_step="third"), dict(states=[rec.energies])):
        with pytest.raises(ValueError):
            dataclasses.replace(rec, **kw)
    with pytest.raises(ValueError):
        dataclasses.replace(h4, irreps=h4.irreps[:-1])
    with pytest.raises(ValueError):
        dataclasses.replace(h4, m=-h4.m)
    with pytest.raises(ValueError):
        dataclasses.replace(h4, variance_state="psi2")
    with pytest.raises(ValueError, match="first_step='exact'"):
        cl.run_closed_loop(h4, cl.LoopSpec(controller="oracle"), "unit", replicate_index=0, first_step="exact")
    with pytest.raises(ValueError, match="propagation"):
        cl.run_closed_loop(h4, cl.LoopSpec(propagation="compiled"), "m3_line_search", replicate_index=0)
    with pytest.raises(ValueError):
        cl.run_closed_loop(h4, cl.LoopSpec(), "unit", replicate_index=-1)
    with pytest.raises(ValueError, match="measures spin-free"):
        cl.system_inputs("h4", params=FTParameters(kind="spin_orbital"))


def test_non_realizable_parts(h4):
    """F32: the oracle controller, the exact line search ("line_search", "m3_line_search"), a binding angle cap and the
    F25 slice rule of compiled propagation (it compares every slicing with the exact state) are named per replicate
    and in the summary; the lagged unit loop without the angle cap is realizable, with exact propagation or with fixed
    compiled slices."""
    lag = cl.LoopSpec(controller="lagged", r=1.0)
    no_cap = StepRule("unit", angle_cap=None)
    cases = [(cl.LoopSpec(controller="oracle"), "unit", ("oracle_controller",)),
             (lag, "line_search", ("exact_line_search",)),
             (cl.LoopSpec(controller="oracle", noise=False), "m3_line_search",
              ("oracle_controller", "exact_line_search")),
             (lag, no_cap, ()),
             (lag, StepRule("unit", angle_cap=1e-3), ("angle_cap",)),
             (cl.LoopSpec(controller="lagged", r=1.0, propagation="compiled"), no_cap, ("slice_rule",)),
             (cl.LoopSpec(controller="lagged", r=1.0, propagation="compiled", slices=2), no_cap, ()),
             (cl.LoopSpec(controller="oracle", r=1.0, propagation="compiled"), no_cap,
              ("oracle_controller", "slice_rule"))]
    for spec, rule, expected in cases:
        rec = cl.run_closed_loop(h4, spec, rule, replicate_index=0, max_iter=3)
        assert rec.non_realizable == expected, (spec.controller, rule)
        summ = cl.summarize(h4, [rec, rec], [cl.closed_loop_costs(h4, rec)] * 2)
        assert summ["realizable"] == (not expected) and summ["non_realizable"] == sorted(expected)
        assert cl.handoff_inputs(h4, rec, cl.closed_loop_costs(h4, rec))["non_realizable"] == list(expected)


def test_run_closed_loop_system(h4, tmp_path):
    """The per-system block: inputs, n replicates per configuration, JSON-able, with the summary of each."""
    cache = scaling.Cache(tmp_path / "cache")
    out = cl.run_closed_loop_system("h4", [(cl.LoopSpec(controller="lagged", r=1.0), "unit"),
                                           (cl.LoopSpec(controller="oracle", r=3.0), "unit", "spin_free")],
                                    n=2, max_iter=3, cache=cache)
    assert out["system"]["c_qpe"] == h4.c_qpe and out["system"]["p_hf"] == h4.spectrum.p_hf
    assert [c["config"]["metric"] for c in out["configs"]] == ["spin_orbital", "spin_free"]
    assert all(len(c["replicates"]) == 2 and c["summary"]["n"] == 2 for c in out["configs"])
    json.dumps(scaling.to_json(out), allow_nan=False)
    again = cl.system_inputs("h4", cache=cache)
    assert np.array_equal(again.variances["v_direct"], h4.variances["v_direct"])


# ------------------------------------------------------------------------- [design] headline cells
@pytest.mark.slow
def test_headline_regression():
    """[design] The closed-loop first measured step (R1 spin-orbital, direct_mg, exact propagation, T_E(k) of M3's rule
    E in C_QPE units): n2_r2.4 oracle r = 1 (100 replicates) mean T_E(2) in [3.85, 4.06], median p~_2 in [0.27,
    0.30]; oracle r = 3 in [3.50, 3.78]; lagged r = 1 in [3.45, 3.80] with P(T_E(2) < T_E(1)) >= 0.9; n2_r2.0
    oracle r = 3 in [2.20, 2.32]; h4_r1.8 oracle r = 1 (20 replicates): T_E(1) is the per-replicate optimum in
    >= 19."""
    cells = [("n2_r2.4", "oracle", 1.0, 100, (3.85, 4.06)), ("n2_r2.4", "oracle", 3.0, 100, (3.50, 3.78)),
             ("n2_r2.4", "lagged", 1.0, 100, (3.45, 3.80)), ("n2_r2.0", "oracle", 3.0, 100, (2.20, 2.32)),
             ("h4_r1.8", "oracle", 1.0, 20, None)]
    inputs = {}
    for name, ctrl, r, n, window in cells:
        if name not in inputs:
            inputs[name] = cl.system_inputs(name)
        inp = inputs[name]
        recs = cl.replicate(inp, cl.LoopSpec(controller=ctrl, r=r), "unit", n=n)
        costs = [cl.closed_loop_costs(inp, x) for x in recs]
        s = cl.summarize(inp, recs, costs)
        t2 = s["per_k"][2]
        if window is not None:
            assert window[0] <= t2["T_mean"] <= window[1], (name, ctrl, r, t2["T_mean"])
        if (name, ctrl, r) == ("n2_r2.4", "oracle", 1.0):
            assert 0.27 <= t2["p_median"] <= 0.30
        if (name, ctrl) == ("n2_r2.4", "lagged"):
            assert t2["beats_psi1"] >= 0.9
        if name == "h4_r1.8":
            assert s["optimum"]["k_hist"][1] >= 19
