"""Tests of the step-rule driver (``thc_engine_cqe.steprules``; F26, F27, F29, F31, F33) and of the public helpers
it uses (``cqe.line_search_step``, ``optimizers.LBFGS.inverse_hessian_apply``).

References: ``cqe.run_cqe`` itself (one exact line-search step; the unit rule as one fixed candidate),
``ft.trajectory.exact_trajectory`` (the line-search rule), dense BFGS matrices of the stored pairs (F26),
closed forms of the parabola fits, Monte Carlo draws of the fits and of the residual noise (F27, F33), and
synthetic planned costs for the hybrid gate (F31).
"""

import math

import numpy as np
import pytest
import scipy.stats

from thc_engine_cqe import cqe, optimizers, steprules
from thc_engine_cqe.fci_ops import FCISpace, Hamiltonian, normalize
from thc_engine_cqe.ft import symmetry as sy
from thc_engine_cqe.ft import residual_inputs as ri
from thc_engine_cqe.ft import trajectory
from thc_engine_cqe.ft.primitives import FTParameters
from thc_engine_cqe.ft.trajectory import propagate
from thc_engine_cqe.measure import targets
from thc_engine_cqe.residual import compute_residual, descent_generator
from thc_engine_cqe.steprules import StepRule, run_step_rule

M3 = FTParameters()
NO_CAPS = {"growth": None, "angle_cap": None}


@pytest.fixture(scope="module")
def built():
    """Symmetry-adapted builds with irreps, built once."""
    cache = {}

    def get(name):
        if name not in cache:
            act = sy.benchmark_system(name)
            cache[name] = (act, sy.orbital_irreps(act))
        return cache[name]

    return get


def _mask(ir):
    return lambda rd, ci, it: sy.mask_pair_tensor(rd.A, ir)


@pytest.fixture(scope="module")
def h4(built):
    return built("h4")


@pytest.mark.parametrize("metric", ["spin_free", "spin_orbital"])
def test_line_search_step_is_the_search_of_run_cqe(h4, metric):
    """line_search_step at eps_max = 1.5/||G|| and tol 1e-10 is run_cqe's first exact search, bit for bit."""
    act, ir = h4
    ham = Hamiltonian(act)
    psi = normalize(FCISpace.of(act).hf())
    res = cqe.run_cqe(act, max_iter=1, metric=metric, residual_estimator=_mask(ir))
    d = optimizers.apply_metric(descent_generator(sy.mask_pair_tensor(compute_residual(act, psi, ham).A, ir)), metric)
    eps_max = 1.5 / max(cqe.generator_norm_estimate(ham.space, d, psi), 1e-12)
    eps, e, psi1 = cqe.line_search_step(ham, d, psi, eps_max, 1e-10)
    assert eps == res.steps[0].eps and e == res.steps[0].energy_after
    assert np.array_equal(psi1, res.ci)


def test_inverse_hessian_apply(h4):
    """No pairs: M g (a copy). With pairs: H_k g_k is the direction d_k; H_k is symmetric positive definite in
    <u, v> = Re sum conj(u) v; calling it changes neither the memory nor the next direction."""
    act, ir = h4
    rng = np.random.default_rng(5)

    def rand():
        return optimizers.symmetrize_residual(rng.standard_normal((4,) * 4))

    g0 = rand()
    so = optimizers.make_optimizer("lbfgs", metric="spin_orbital")
    out = so.inverse_hessian_apply(g0)
    assert np.allclose(out, optimizers.apply_metric(g0, "spin_orbital")) and out is not g0
    out = optimizers.make_optimizer("lbfgs").inverse_hessian_apply(g0)
    assert np.array_equal(out, g0) and not np.shares_memory(out, g0)
    # an exact masked L-BFGS run (exact first step, then unit steps) supplies pairs with <s, y> > 0
    opt = optimizers.make_optimizer("lbfgs", metric="spin_orbital")
    twin = optimizers.make_optimizer("lbfgs", metric="spin_orbital")
    ham = Hamiltonian(act)
    psi = normalize(FCISpace.of(act).hf())
    for it in range(4):
        a = sy.mask_pair_tensor(compute_residual(act, psi, ham).A, ir)
        g = descent_generator(a)
        d, d_twin = opt.direction(g, a), twin.direction(g, a)
        if it > 0:
            assert opt.n_pairs > 0
            h_g = opt.inverse_hessian_apply(g)
            assert np.allclose(h_g, d, rtol=0, atol=1e-14 * np.abs(d).max())
            u, v = rand(), rand()
            hu, hv = opt.inverse_hessian_apply(u), opt.inverse_hessian_apply(v)
            assert optimizers.real_inner(u, hv) == pytest.approx(optimizers.real_inner(hu, v), rel=1e-10)
            assert optimizers.real_inner(u, hu) > 0
        np.testing.assert_array_equal(d, d_twin)          # the extra applications changed nothing
        if it == 0:
            eps_max = 1.5 / cqe.generator_norm_estimate(ham.space, d, psi)
            eps, _, psi = cqe.line_search_step(ham, d, psi, eps_max, 1e-10)
        else:
            eps = 1.0
            psi = propagate(ham, d, psi, eps)
        opt.step_taken(eps)
        twin.step_taken(eps)


# ------------------------------------------------------------------------- parity with M3's driver
@pytest.mark.parametrize("name", ["h4_r1.8", "n2_r2.0"])
@pytest.mark.parametrize("metric", ["spin_free", "spin_orbital"])
def test_unit_rule_equals_run_cqe(built, name, metric):
    """The unit rule with the caps off and exact (masked) residuals is run_cqe with the classical first step and
    one fixed candidate eps = 1: every energy to 1e-12 Ha over 10 steps (observed <= 1.4e-14)."""
    act, ir = built(name)
    eps2 = trajectory.second_order_step(act, metric=metric, irreps=ir)["eps2"]
    ref = cqe.run_cqe(act, max_iter=10, optimizer="lbfgs", metric=metric, first_eps=eps2, eps_candidates=(1.0,),
                      residual_estimator=_mask(ir))
    res = run_step_rule(act, StepRule("unit", **NO_CAPS), max_iter=10, metric=metric, irreps=ir)
    assert res.n_steps == 10 and len(ref.energies) == 11
    assert res.eps[0] == eps2 and all(e == 1.0 for e in res.eps[1:])
    assert np.max(np.abs(res.energies - ref.energies)) <= 1e-12
    assert res.optimizer == "lbfgs" and res.metric == metric
    for a, b in zip(res.steps, ref.steps):
        assert a.slope == pytest.approx(b.slope, rel=1e-9, abs=1e-12)
        assert a.residual_norm == pytest.approx(b.residual_norm, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("metric", ["spin_free", "spin_orbital"])
def test_line_search_rule_equals_exact_trajectory(built, metric):
    """The line-search rule with the classical first step is exact_trajectory(first_step="second_order"):
    energies to 1e-10 Ha over 8 steps (observed <= 2e-11; the steps of the flat late line searches agree only
    to the line search's round-off, sqrt(|E| u / dE))."""
    act, ir = built("h4")
    rec = trajectory.exact_trajectory(act, max_iter=8, irreps=ir, metric=metric, first_step="second_order")
    res = run_step_rule(act, StepRule("line_search"), max_iter=8, metric=metric, irreps=ir, keep_states=True)
    assert res.n_steps == rec.n_steps == 8
    assert res.eps[0] == rec.eps[0]
    assert np.max(np.abs(res.energies - rec.energies)) <= 1e-10
    for a, b in zip(rec.states, res.states):
        assert abs(np.vdot(a.ravel(), b.ravel())) ** 2 == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize("name", ["h4", "n2s"])
@pytest.mark.parametrize("metric", ["spin_free", "spin_orbital"])
def test_measured_loop_equals_m3_exact_record(built, name, metric):
    """first="line_search" with the line-search rule is M3's exact record (exact_trajectory, first_step="exact"),
    bit for bit in eps and every energy over 12 steps; the stored states agree after exact_trajectory's extra
    normalization."""
    act, ir = built(name)
    rec = trajectory.exact_trajectory(act, max_iter=12, irreps=ir, metric=metric, n_roots=12)
    res = run_step_rule(act, StepRule("line_search", first="line_search"), max_iter=12, metric=metric, irreps=ir,
                        keep_states=True)
    assert res.n_steps == rec.n_steps == 12
    assert res.eps == rec.eps.tolist()
    assert np.array_equal(res.energies, rec.energies)
    assert all(np.array_equal(normalize(b), a) for a, b in zip(rec.states, res.states))
    assert [e.iteration for e in res.events if e.kind == "residual"] == list(range(12))


# ------------------------------------------------------------------------- F26
@pytest.mark.parametrize("name", ["h4_r3.0", "n2_r2.0"])
@pytest.mark.parametrize("metric", ["spin_free", "spin_orbital"])
def test_model_step_is_unit(built, monkeypatch, name, metric):
    """F26: for every L-BFGS direction with stored pairs, <d, B d> = -s0 = <g, d> to 1e-12 relative, with B the
    dense BFGS matrix of the stored pairs (B_0 = M^-1/gamma, M^-1 = 2I - T for the spin-orbital metric)."""
    act, ir = built(name)
    seen = []
    original = optimizers.LBFGS.direction

    def spy(self, g, a):
        d = original(self, g, a)
        if self._pairs:
            seen.append((np.array(g), np.array(d), [(s.copy(), y.copy()) for s, y, _ in self._pairs], self.metric))
        return d

    monkeypatch.setattr(optimizers.LBFGS, "direction", spy)
    run_step_rule(act, StepRule("unit"), max_iter=12, metric=metric, irreps=ir)
    assert len(seen) >= 8
    n = act.norb
    eye = np.eye(n ** 4)
    tmat = eye.reshape((n,) * 4 + (n ** 4,)).transpose(0, 3, 2, 1, 4).reshape(n ** 4, n ** 4)   # (T x) = x[p,s,r,q]
    minv = eye if metric == "spin_free" else 2.0 * eye - tmat
    for g, d, pairs, m in seen:
        s_new, y_new = pairs[-1]
        gamma = optimizers.real_inner(s_new, y_new) / optimizers.real_inner(y_new, optimizers.apply_metric(y_new, m))
        b = minv / gamma
        for s, y in pairs:
            s, y = s.ravel(), y.ravel()
            bs = b @ s
            b = b - np.outer(bs, bs) / (s @ bs) + np.outer(y, y) / (y @ s)
        dbd = d.ravel() @ b @ d.ravel()
        gd = optimizers.real_inner(g, d)
        assert dbd == pytest.approx(gd, rel=1e-12)


# ------------------------------------------------------------------------- F29
def test_growth_cap_binds_and_flags(built):
    """The growth cap: every step obeys ||s_k|| <= gamma_s ||s_{k-1}|| (k >= 1); a flagged step lies on the cap
    (eps < 1 and ||s_k|| = gamma_s ||s_{k-1}||), an unflagged one is the unit step; the k = 0 step is the uncapped
    classical step. It binds on h4_r4.0 (spin-orbital) at gamma_s = 2 and more often at 1.2."""
    act, ir = built("h4_r4.0")
    eps2 = trajectory.second_order_step(act, metric="spin_orbital", irreps=ir)["eps2"]
    for growth in (2.0, 1.2):
        res = run_step_rule(act, StepRule("unit", growth=growth, angle_cap=None), max_iter=14, irreps=ir)
        assert res.eps[0] == eps2 and "growth_cap" not in res.flags[0] and "first:second_order" in res.flags[0]
        norms = [abs(e) * np.linalg.norm(d) for e, d in zip(res.eps, res.directions)]
        flagged = [k for k, f in enumerate(res.flags) if "growth_cap" in f]
        assert flagged and all(k >= 1 for k in flagged)
        for k in range(1, res.n_steps):
            assert norms[k] <= growth * norms[k - 1] * (1 + 1e-12)
            if k in flagged:
                assert res.eps[k] < 1.0 and norms[k] == pytest.approx(growth * norms[k - 1], rel=1e-12)
            else:
                assert res.eps[k] == 1.0
    loose = run_step_rule(act, StepRule("unit", growth=2.0, angle_cap=None), max_iter=14, irreps=ir)
    tight = run_step_rule(act, StepRule("unit", growth=1.2, angle_cap=None), max_iter=14, irreps=ir)
    assert sum("growth_cap" in f for f in tight.flags) > sum("growth_cap" in f for f in loose.flags)


def test_angle_cap_flags(built):
    """A tight angle cap (eps ||G_k|| <= 0.2 on h4_r4.0, where the unit steps reach 1.1 at k = 1) binds: flagged
    steps have eps_k ||G_k|| = 0.2 with ||G_k|| the power estimate at Psi_k; unflagged steps are unit steps with
    ||G_k|| <= 0.2; the classical k = 0 step (angle 2.3) is not capped; with both caps on, a flag names the cap that
    sets the step."""
    act, ir = built("h4_r4.0")
    cap = 0.2
    res = run_step_rule(act, StepRule("unit", growth=None, angle_cap=cap), max_iter=8, irreps=ir, keep_states=True)
    space = Hamiltonian(act).space
    assert res.eps[0] * cqe.generator_norm_estimate(space, res.directions[0], res.states[0]) > 2.0
    flagged = [k for k, f in enumerate(res.flags) if "angle_cap" in f]
    assert flagged and 0 not in flagged and len(flagged) < res.n_steps - 1
    for k in range(1, res.n_steps):
        gnorm = cqe.generator_norm_estimate(space, res.directions[k], res.states[k])
        if k in flagged:
            assert res.eps[k] * gnorm == pytest.approx(cap, rel=1e-12)
        else:
            assert res.eps[k] == 1.0 and gnorm <= cap
    both = run_step_rule(act, StepRule("unit", growth=1.5, angle_cap=cap), max_iter=8, irreps=ir, keep_states=True)
    for k in range(1, both.n_steps):
        caps = {"angle_cap": cap / cqe.generator_norm_estimate(space, both.directions[k], both.states[k]),
                "growth_cap": 1.5 * abs(both.eps[k - 1]) * np.linalg.norm(both.directions[k - 1])
                / np.linalg.norm(both.directions[k])}
        names = [n for n in ("growth_cap", "angle_cap") if n in both.flags[k]]
        if names:
            assert both.eps[k] == pytest.approx(min(caps.values()), rel=1e-12)
            assert names == [min(caps, key=caps.get)]
        else:
            assert both.eps[k] == 1.0 and min(caps.values()) >= 1.0


# ------------------------------------------------------------------------- F27: the 3-point fit
def test_parabola_step():
    """The exact minimizer of a parabola from {0, t, 2t}; the interpolation-only rule (minimizer outside (0, 2t]
    or a concave fit -> the best measured point, 0.25t when E(0) is the lowest; floor 0.05t); the extrapolating
    variant clips to [0.05t, 3t]."""
    def energies(eps_star, t, c=0.8, e0=-1.3):
        f = lambda x: e0 + 0.5 * c * (x - eps_star) ** 2       # noqa: E731
        return f(0.0), f(t), f(2.0 * t)

    for eps_star, t in ((0.7, 0.5), (1.0, 1.0), (1.9, 1.0), (0.2, 0.15)):
        eps, status = steprules.parabola_step(*energies(eps_star, t), t)
        assert status == "fit" and eps == pytest.approx(eps_star, rel=1e-12)
    assert steprules.parabola_step(*energies(2.6, 1.0), 1.0) == (2.0, "clipped")          # beyond the stencil
    assert steprules.parabola_step(*energies(-0.5, 1.0), 1.0) == (0.25, "clipped")        # E(0) is the lowest
    for es, t in (((1.0, 0.5, 0.9), 1.0), ((0.0, -0.2, -0.1), 0.4), ((0.3, -0.4, -0.3), 2.0)):
        a2, a1, _ = np.polyfit([0.0, t, 2.0 * t], es, 2)                                  # independent fit
        eps, status = steprules.parabola_step(*es, t)
        assert status == "fit" and eps == pytest.approx(-a1 / (2.0 * a2), rel=1e-10)
    concave = (0.0, -0.1, -0.5)                                                              # c < 0, 2t lowest
    assert steprules.parabola_step(*concave, 1.0) == (2.0, "clipped")
    assert steprules.parabola_step(0.0, -0.5, -0.1, 1.0)[1] == "fit"
    assert steprules.parabola_step(0.0, -0.5, -0.5, 1.0) == pytest.approx((1.5, "fit"))
    eps, status = steprules.parabola_step(*energies(0.01, 1.0), 1.0)
    assert (eps, status) == (0.05, "floor")
    assert steprules.parabola_step(*energies(2.6, 1.0), 1.0, interpolate_only=False) == pytest.approx((2.6, "fit"))
    assert steprules.parabola_step(*energies(4.0, 1.0), 1.0, interpolate_only=False) == (3.0, "ceiling")
    assert steprules.parabola_step(*concave, 1.0, interpolate_only=False) == (2.0, "clipped")
    with pytest.raises(ValueError):
        steprules.parabola_step(0.0, 1.0, 2.0, 0.0)
    with pytest.raises(ValueError):
        steprules.parabola_step(0.0, math.nan, 2.0, 1.0)


@pytest.mark.slow
def test_f27_monte_carlo():
    """F27 on the parabola E = s eps + c eps^2/2 (eps* = 1, dE* = -1/2), 4e5 draws: the linearized error of the
    3-point minimizer, sigma^2 u^2 f(u)/(4 dE^2), within 5% at u = eps*/t in {1/2, 1, 2} (sigma = 0.01|dE|, no
    clip); the expected loss of parabola_step (interpolation only) at sigma = 0.47|dE| is 0.10 +- 0.01; the Armijo
    test with two fresh energies at sigma_A = (1 - 2 c1)|dE|/(sqrt 2 z) rejects the model step with probability
    Phi(-z) = 0.0228 (z = 2) within 3 standard errors; the 2-point fit (E(0), E(t), exact slope) loses 0.10 at
    sigma_P2 ~ 0.16|dE|."""
    rng = np.random.default_rng(20261108)
    n = 400_000
    s, c, de = -1.0, 1.0, -0.5

    def e(x):
        return s * x + 0.5 * c * x * x

    def f(u):
        return (u - 1.5) ** 2 + 4.0 * (1.0 - u) ** 2 + (u - 0.5) ** 2

    for u in (0.5, 1.0, 2.0):
        t, sig = 1.0 / u, 0.01 * abs(de)
        e0, e1, e2 = (e(x) + sig * rng.standard_normal(n) for x in (0.0, t, 2.0 * t))
        cc = (e0 - 2.0 * e1 + e2) / t ** 2
        bb = (-3.0 * e0 + 4.0 * e1 - e2) / (2.0 * t)
        sim = np.mean((-bb / cc - 1.0) ** 2)
        assert sim / (sig ** 2 * u ** 2 * f(u) / (4.0 * de ** 2)) == pytest.approx(1.0, abs=0.05)
    sig = 0.47 * abs(de)
    draws = e(np.array([0.0, 1.0, 2.0]))[:, None] + sig * rng.standard_normal((3, n))
    steps = np.array([steprules.parabola_step(a, b, cc, 1.0)[0] for a, b, cc in draws.T])
    loss = (e(steps) - e(1.0)).mean() / abs(de)
    assert loss == pytest.approx(0.10, abs=0.01)
    c1, z = 1e-4, 2.0
    sig_a = (1.0 - 2.0 * c1) * abs(de) / (math.sqrt(2.0) * z)
    d_hat = (e(1.0) + sig_a * rng.standard_normal(n)) - (e(0.0) + sig_a * rng.standard_normal(n))
    rej = np.mean(d_hat > c1 * 1.0 * s)
    p = scipy.stats.norm.cdf(-z)
    assert abs(rej - p) <= 3.0 * math.sqrt(p * (1.0 - p) / n)
    sig2 = 0.16 * abs(de)
    dd = (e(1.0) + sig2 * rng.standard_normal(n)) - (e(0.0) + sig2 * rng.standard_normal(n))
    c2 = 2.0 * (dd - s) / 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        x2 = np.where(c2 > 0, np.clip(-s / c2, 0.05, 3.0), 3.0)
    assert (e(x2) - e(1.0)).mean() / abs(de) == pytest.approx(0.10, abs=0.02)


# ------------------------------------------------------------------------- ledger and rules
def test_event_ledger(built):
    """Residual events: one per iteration k >= 1 (at Psi_k), none at k = 0 with the classical first step, one at
    k = 0 with first="line_search"; the unit rule has no energies; parab3 (and hybrid on its parab3 iterations)
    measures E at {0, t, 2t} with sigma = 0.47|dE_m|, dE_m = s0/2; line_search and candidates charge M3's 11 energies
    at eps_ls (F13); n_energy_evals counts the iteration's energy events."""
    act, ir = built("h4_r3.0")
    unit = run_step_rule(act, StepRule("unit"), max_iter=5, irreps=ir)
    assert [(e.iteration, e.kind, e.eps, e.sigma, e.used_for_step) for e in unit.events] == \
        [(k, "residual", 0.0, None, True) for k in range(1, 5)]
    assert all(s.n_energy_evals == 0 for s in unit.steps)
    measured = run_step_rule(act, StepRule("unit", first="line_search"), max_iter=4, irreps=ir)
    assert [e.iteration for e in measured.events if e.kind == "residual"] == [0, 1, 2, 3]
    assert len(measured.iteration_events(0)) == 1 + M3.n_candidates and "first:line_search" in measured.flags[0]
    parab = run_step_rule(act, StepRule("parab3"), max_iter=5, irreps=ir)
    gate = lambda k, de_m, rd, history: k % 2 == 1                        # noqa: E731
    hybrid = run_step_rule(act, StepRule("hybrid"), max_iter=5, irreps=ir, use_energies=gate)
    for res, parab_iters in ((parab, (1, 2, 3, 4)), (hybrid, (1, 3))):
        for k in range(1, 5):
            ev = res.iteration_events(k)
            assert ev[0] == steprules.StepEvent(k, "residual", 0.0, None, True)
            energies = [e for e in ev if e.kind == "energy"]
            assert res.steps[k].n_energy_evals == len(energies)
            if k in parab_iters:
                de_m = 0.5 * res.steps[k].slope_est
                assert res.de_model[k] == pytest.approx(de_m, rel=1e-14)
                assert [e.eps for e in energies][0] == 0.0 and len(energies) == 3
                assert energies[2].eps == pytest.approx(2.0 * energies[1].eps, rel=1e-15)
                assert all(e.sigma == pytest.approx(0.47 * abs(de_m), rel=1e-14) for e in energies)
                assert res.steps[k].energy_est == pytest.approx(res.steps[k].energy, abs=1e-12)
            else:
                assert not energies and res.eps[k] <= 1.0
    assert hybrid.flags[1][0] == "hybrid:parab3" and "hybrid:unit" in hybrid.flags[2]
    for rule in ("line_search", "candidates"):
        res = run_step_rule(act, StepRule(rule), max_iter=3, irreps=ir)
        for k in (1, 2):
            energies = [e for e in res.iteration_events(k) if e.kind == "energy"]
            assert [e.eps for e in energies] == list(M3.eps_candidates)
            assert all(e.sigma == M3.eps_ls for e in energies) and res.steps[k].n_energy_evals == 11


def test_candidates_rule_takes_the_lowest_estimate(built):
    """candidates: the step is the candidate with the lowest estimated energy (exact: the lowest exact energy along
    d_k); with an energy estimator that adds a known offset per candidate the choice follows the estimates; the
    estimator receives the requested sigma (eps_ls)."""
    act, ir = built("h4_r3.0")
    res = run_step_rule(act, StepRule("candidates"), max_iter=3, irreps=ir, keep_states=True)
    ham = Hamiltonian(act)
    for k in (1, 2):
        e_c = [ham.energy(propagate(ham, res.directions[k], res.states[k], c)) for c in M3.eps_candidates]
        assert res.eps[k] == M3.eps_candidates[int(np.argmin(e_c))]
    sigmas = []

    def biased(e_exact, ci, sigma):
        sigmas.append(sigma)
        return e_exact - 10.0 * float(np.sum(np.abs(ci) ** 3)) if len(sigmas) % 11 == 4 else e_exact

    res_b = run_step_rule(act, StepRule("candidates"), max_iter=2, irreps=ir, energy_estimator=biased)
    assert res_b.eps[1] == M3.eps_candidates[3] and set(sigmas) == {M3.eps_ls}
    assert res_b.steps[1].energy_after_est < res_b.steps[1].energy_after


def test_parab3_rule_with_exact_energies(built):
    """parab3 with exact energies: the step is parabola_step of the exact E at {0, t, 2t} along d_k (t = 1 unless
    capped), and on h4_r3.0 it lowers the energy at every step."""
    act, ir = built("h4_r3.0")
    res = run_step_rule(act, StepRule("parab3"), max_iter=6, irreps=ir, keep_states=True)
    ham = Hamiltonian(act)
    for k in range(1, res.n_steps):
        ev = [e for e in res.iteration_events(k) if e.kind == "energy"]
        t = ev[1].eps
        es = [ham.energy(res.states[k]) if x == 0.0 else ham.energy(propagate(ham, res.directions[k], res.states[k], x))
              for x in (0.0, t, 2.0 * t)]
        eps, status = steprules.parabola_step(*es, t)
        assert res.eps[k] == pytest.approx(eps, rel=1e-10)
        assert (status != "fit") == any(f.startswith("parab:") for f in res.flags[k])
    assert np.all(np.diff(res.energies) < 0)


def test_parab3_stencil(built):
    """parab3 with the caps off measures its three energies at exactly {0, t, 2t}, t = parab_t (1 and 0.5), at every
    iteration k >= 1."""
    act, ir = built("h4_r3.0")
    for t in (1.0, 0.5):
        res = run_step_rule(act, StepRule("parab3", parab_t=t, **NO_CAPS), max_iter=4, irreps=ir)
        assert res.n_steps == 4
        for k in (1, 2, 3):
            assert [e.eps for e in res.iteration_events(k) if e.kind == "energy"] == [0.0, t, 2.0 * t]


@pytest.mark.parametrize("name, growth", [("h4_r4.0", 1.2), ("n2_r2.4", 2.0)])
def test_parab3_stencil_obeys_the_growth_cap(built, name, growth):
    """F29 caps the stencil's far point 2t (t = 1): with c = gamma_s ||s_{k-1}||_F/||d_k||_F (s_{k-1} the previous
    nonzero step), an iteration flagged "growth_cap" measures at {0, c/2, c} with c < 2 and fits eps_k <= c; an
    unflagged one at {0, 1, 2} with c >= 2; the cap binds at least once in 8 steps."""
    act, ir = built(name)
    res = run_step_rule(act, StepRule("parab3", growth=growth, angle_cap=None), max_iter=8, irreps=ir)
    prev, capped = None, 0
    for k in range(res.n_steps):
        stencil = [e.eps for e in res.iteration_events(k) if e.kind == "energy"]
        if k >= 1:
            c = growth * prev / np.linalg.norm(res.directions[k])
            if "growth_cap" in res.flags[k]:
                capped += 1
                assert stencil == pytest.approx([0.0, 0.5 * c, c], rel=1e-12) and c < 2.0
                assert res.eps[k] <= c * (1.0 + 1e-12)
            else:
                assert stencil == [0.0, 1.0, 2.0] and c >= 2.0
        step = abs(res.eps[k]) * np.linalg.norm(res.directions[k])
        prev = step if step > 0.0 else prev
    assert capped >= 1


def test_parab3_stencil_obeys_the_angle_cap(built):
    """F29's angle cap on the stencil (h4_r4.0, cap 0.2, growth cap off): a flagged iteration measures at {0, c/2, c}
    with c = 0.2/||G_k|| < 2 (||G_k||: the power estimate at Psi_k) and fits eps_k <= c; an unflagged one at {0, 1, 2}
    with c >= 2; the cap binds at least once."""
    act, ir = built("h4_r4.0")
    res = run_step_rule(act, StepRule("parab3", growth=None, angle_cap=0.2), max_iter=6, irreps=ir, keep_states=True)
    space = Hamiltonian(act).space
    flagged = 0
    for k in range(1, res.n_steps):
        stencil = [e.eps for e in res.iteration_events(k) if e.kind == "energy"]
        c = 0.2 / cqe.generator_norm_estimate(space, res.directions[k], res.states[k])
        if "angle_cap" in res.flags[k]:
            flagged += 1
            assert stencil == pytest.approx([0.0, 0.5 * c, c], rel=1e-12) and c < 2.0
            assert res.eps[k] <= c * (1.0 + 1e-12)
        else:
            assert stencil == [0.0, 1.0, 2.0] and c >= 2.0
    assert flagged >= 1


def test_hybrid_gate_switches_at_theta(built):
    """F31 through the hook: on synthetic planned costs, T_E = 3 V_E/sigma_P3^2 (sigma_P3 = 0.47|dE_m|) and
    T_res = V_m/(r^2 ||A_k||^2) (shots of unit cost), the hybrid rule measures energies iff T_E <= theta T_res; the
    hook receives k, dE_m = s0/2, the exact ResidualData of Psi_k and the history: the directions d_0 .. d_k (the
    current one last, read-only) and the k steps taken."""
    act, ir = built("h4_r4.0")
    v_e, v_m, r = 0.4, 400.0, 1.0
    calls = []

    def ratio(de_m, norm):
        t_e = 3.0 * v_e / (0.47 * de_m) ** 2
        t_res = v_m / (r ** 2 * norm ** 2)
        return t_e / t_res

    for theta in (0.3, 1.0, 3.0):
        calls.clear()

        def gate(k, de_m, rd, history):
            directions, eps = history
            assert len(directions) == k + 1 and len(eps) == k
            assert not any(x.flags.writeable for x in directions)
            calls.append((k, de_m, sy.mask_pair_tensor(rd.A, ir), [x.copy() for x in directions], list(eps)))
            return ratio(de_m, np.linalg.norm(calls[-1][2])) <= theta

        res = run_step_rule(act, StepRule("hybrid", theta=theta), max_iter=8, irreps=ir, use_energies=gate)
        assert [c[0] for c in calls] == list(range(1, res.n_steps))
        for k, de_m, a, directions, eps in calls:
            assert de_m == pytest.approx(0.5 * res.steps[k].slope_est, rel=1e-14)
            assert all(np.array_equal(x, y) for x, y in zip(directions, res.directions[:k + 1]))
            assert eps == res.eps[:k]
            want = ratio(de_m, np.linalg.norm(a)) <= theta
            assert ("hybrid:parab3" in res.flags[k]) == want
            assert ("hybrid:unit" in res.flags[k]) == (not want)
            assert res.steps[k].n_energy_evals == (3 if want else 0)


def test_step_rule_validation(built):
    for kw in ({"name": "secant"}, {"first": "exact"}, {"growth": 1.0}, {"growth": 0.5}, {"growth": math.inf},
               {"angle_cap": 0.0}, {"angle_cap": -1.0}, {"sigma_parab": 0.0}, {"parab_t": -1.0},
               {"theta": 0.0}, {"theta": math.nan}, {"name": "unit", "candidates": (1.0,)},
               {"name": "candidates", "candidates": (0.1, math.inf)}, {"name": "candidates", "candidates": ("a",)}):
        with pytest.raises(ValueError):
            StepRule(**kw)
    assert StepRule("candidates").candidates == M3.eps_candidates
    assert StepRule("line_search").candidates == M3.eps_candidates
    assert StepRule("candidates", candidates=(0.1, 0.2)).candidates == (0.1, 0.2)
    assert StepRule(theta=math.inf).theta == math.inf and StepRule().candidates == ()
    act, ir = built("h4")
    with pytest.raises(ValueError, match="use_energies"):
        run_step_rule(act, StepRule("hybrid"), irreps=ir)
    for opt in ("sd", "cg"):
        with pytest.raises(ValueError, match="lbfgs"):
            run_step_rule(act, StepRule("unit"), optimizer=opt, irreps=ir)
        run_step_rule(act, StepRule("line_search"), optimizer=opt, irreps=ir, max_iter=2)
    with pytest.raises(ValueError, match="irreps"):
        run_step_rule(act, irreps=np.zeros(3, dtype=int))
    with pytest.raises(ValueError):
        run_step_rule(act, "unit", irreps=ir)
    with pytest.raises(ValueError, match="max_iter"):
        run_step_rule(act, max_iter=-1)
    with pytest.raises(ValueError, match="shape"):
        run_step_rule(act, irreps=ir, max_iter=2, residual_estimator=lambda rd, ci, k, eps: rd.A[:2])
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="structure"):
        run_step_rule(act, irreps=ir, max_iter=2,
                      residual_estimator=lambda rd, ci, k, eps: rd.A + rng.standard_normal(rd.A.shape))
    with pytest.raises(ValueError, match="variances"):
        run_step_rule(act, irreps=ir, max_iter=2, residual_estimator=lambda rd, ci, k, eps: (rd.A, np.ones(3)))
    with pytest.raises(ValueError, match="non-finite"):
        run_step_rule(act, StepRule("parab3"), irreps=ir, max_iter=2, energy_estimator=lambda e, ci, s: math.nan)
    assert run_step_rule(act, irreps=ir, max_iter=0).n_steps == 0
    with pytest.raises(ValueError, match="real"):
        run_step_rule(act, irreps=ir, ci0=FCISpace.of(act).hf() * 1j)
    # a correlated start: the classical first step (F16) needs a determinant, the measured first step does not
    start = run_step_rule(act, irreps=ir, max_iter=1, keep_states=True)
    assert np.count_nonzero(np.abs(start.states[1]) > 1e-12) > 1
    with pytest.raises(ValueError, match="determinant"):
        run_step_rule(act, irreps=ir, max_iter=1, ci0=start.states[1].ravel())
    cont = run_step_rule(act, StepRule(first="line_search"), irreps=ir, max_iter=1, ci0=start.states[1].ravel())
    assert cont.energies[0] == pytest.approx(start.energies[1], abs=1e-12)
    assert [e.iteration for e in cont.events if e.kind == "residual"] == [0] and "first:line_search" in cont.flags[0]
    # a determinant start (HF given explicitly) keeps the classical first step of the default start
    explicit = run_step_rule(act, irreps=ir, max_iter=2, ci0=FCISpace.of(act).hf().ravel())
    default = run_step_rule(act, irreps=ir, max_iter=2)
    assert explicit.eps == default.eps and np.array_equal(explicit.energies, default.energies)
    assert "first:second_order" in explicit.flags[0] and not explicit.iteration_events(0)


def test_estimator_and_propagator_hooks(built):
    """The residual estimator is called at Psi_k for k >= 1 only (classical k = 0) with eps = 0 and copies of the
    state; returning the masked exact residual reproduces the default run; a propagator prepares every state
    (here the default exponential plus a check of its arguments)."""
    act, ir = built("h4_r3.0")
    ham = Hamiltonian(act)
    calls, props = [], []

    def est(rd, ci, k, eps):
        calls.append((k, eps))
        ci[...] = 0.0                                   # a copy: the driver's state is untouched
        return sy.mask_pair_tensor(rd.A, ir)

    def prop(d, eps, ci, k):
        props.append((k, eps))
        return propagate(ham, d, ci, eps)

    base = run_step_rule(act, StepRule("parab3"), max_iter=5, irreps=ir)
    res = run_step_rule(act, StepRule("parab3"), max_iter=5, irreps=ir, residual_estimator=est, propagator=prop)
    assert calls == [(k, 0.0) for k in range(1, 5)]
    assert np.max(np.abs(res.energies - base.energies)) <= 1e-12 and res.eps == pytest.approx(base.eps, rel=1e-10)
    assert props[0] == (0, base.eps[0]) and {k for k, _ in props} == set(range(5))
    for k in range(1, 5):                               # the stencil {t, 2t}, then the chosen step if new
        ks = [e for kk, e in props if kk == k]
        assert ks[:2] == [ks[0], 2.0 * ks[0]] and res.eps[k] in ks


def test_compiled_propagator_hook(built):
    """A compiled propagator (ft.circuits.compiled_step from the current state) drives the loop: the K states are
    the open-loop replay of the run's own directions and steps from HF (to 1e-12), the residuals are taken on the
    compiled states, and the compiled trajectory departs from the exact one."""
    from thc_engine_cqe.ft import circuits

    act, ir = built("n2s")
    prop = lambda d, eps, ci, k: circuits.compiled_step(act, ci, d, eps, irreps=ir)      # noqa: E731
    res = run_step_rule(act, StepRule("unit"), max_iter=4, irreps=ir, propagator=prop, keep_states=True)
    replay = circuits.replay_elementwise(act, res.directions, res.eps, irreps=ir)
    assert all(np.max(np.abs(a - b)) <= 1e-12 for a, b in zip(res.states, replay))
    ham = Hamiltonian(act)
    assert all(st.energy == pytest.approx(ham.energy(x), abs=1e-12) for st, x in zip(res.steps, res.states))
    exact = run_step_rule(act, StepRule("unit"), max_iter=4, irreps=ir)
    assert np.max(np.abs(exact.energies - res.energies)) > 1e-4


# ------------------------------------------------------------------------- F33
@pytest.mark.parametrize("metric", ["spin_free", "spin_orbital"])
def test_debiased_slope(built, metric):
    """F33 with the inverse Hessian frozen (the pairs of an exact unit run up to iteration 3, fixed before eta):
    over 2e4 draws eta_e ~ N(0, V_e) on the allowed elements, the mean of s0^ = -<g + eta, H(g + eta)> plus
    tau = debiased_slope(0, V, ...) equals s0 = -<g, H g> within 3 standard errors, while the raw bias -tau is
    resolved (|tau| > 10 standard errors)."""
    act, ir = built("h4_r3.0")
    res = run_step_rule(act, StepRule("unit"), max_iter=4, metric=metric, irreps=ir, keep_states=True)
    ham = Hamiltonian(act)
    opt = optimizers.make_optimizer("lbfgs", metric=metric)
    for k in range(3):
        a = sy.mask_pair_tensor(compute_residual(act, res.states[k], ham).A, ir)
        d = opt.direction(descent_generator(a), a)
        np.testing.assert_allclose(d, res.directions[k], rtol=0, atol=1e-14 * np.abs(d).max())
        opt.step_taken(res.eps[k])
    a3 = sy.mask_pair_tensor(compute_residual(act, res.states[3], ham).A, ir)
    g = descent_generator(a3)
    d3 = opt.direction(g, a3)
    hinv = opt.inverse_hessian_apply
    el, _ = ri.allowed_elements(act, "spin_free", ir)
    m = targets.element_multiplicities(el, "spin_free", act.norb).astype(float)
    a_el = a3[tuple(el.T)]
    var = np.full(len(el), float(np.sum(m * a_el ** 2)) / float(np.sum(m)))       # r = 1, uniform
    s0 = -optimizers.real_inner(g, d3)
    tau = steprules.debiased_slope(0.0, var, el, act.norb, hinv)
    assert steprules.debiased_slope(-1.5, var, el, act.norb, hinv) == pytest.approx(tau - 1.5, rel=1e-14)
    fills = steprules.unit_fills(el, act.norb)
    for j in (0, 5, len(el) - 1):
        onehot = np.zeros(len(el))
        onehot[j] = 1.0
        np.testing.assert_array_equal(fills[j], targets.spin_free_tensor(onehot, el, act.norb))
    # dense H on the element span: s0^ = -(a + eta)^T Hm (a + eta), Hm_ef = <u_e, H u_f>
    hu = np.array([hinv(u).ravel() for u in fills])
    hm = fills.reshape(len(el), -1) @ hu.T
    assert np.allclose(hm, hm.T, rtol=0, atol=1e-12 * np.abs(hm).max())
    assert tau == pytest.approx(float(np.sum(var * np.diag(hm))), rel=1e-12)
    assert -(a_el @ hm @ a_el) == pytest.approx(s0, rel=1e-12)
    rng = np.random.default_rng(11)
    eta = rng.standard_normal((20_000, len(el))) * np.sqrt(var)
    x = a_el + eta
    s0_hat = -np.einsum("ne,ef,nf->n", x, hm, x)
    se = s0_hat.std(ddof=1) / math.sqrt(len(s0_hat))
    assert abs(s0_hat.mean() + tau - s0) <= 3.0 * se
    assert abs(tau) > 10.0 * se
    # one draw through the tensors: the same s0^ as the element form
    eta_t = np.tensordot(eta[0], fills, axes=1)
    g_hat = descent_generator(a3 + eta_t)
    assert -optimizers.real_inner(g_hat, hinv(g_hat)) == pytest.approx(s0_hat[0], rel=1e-10)


def test_debiased_slope_flags_in_the_loop(built):
    """With an estimator that returns variances, the loop debiases s0 (dE_m = s0'/2) and flags "slope_debias_approx"
    where L-BFGS holds pairs; zero variances leave s0 unchanged."""
    act, ir = built("h4")
    el, _ = ri.allowed_elements(act, "spin_free", ir)
    zero = run_step_rule(act, irreps=ir, max_iter=4,
                         residual_estimator=lambda rd, ci, k, eps: (sy.mask_pair_tensor(rd.A, ir), np.zeros(len(el))))
    base = run_step_rule(act, irreps=ir, max_iter=4)
    assert zero.de_model == pytest.approx(base.de_model, rel=1e-14)
    assert all("slope_debias_approx" in f for f in zero.flags[1:])
    pos = run_step_rule(act, irreps=ir, max_iter=4,
                        residual_estimator=lambda rd, ci, k, eps: (sy.mask_pair_tensor(rd.A, ir),
                                                                   np.full(len(el), 1e-4)))
    for k in range(1, 4):
        assert pos.de_model[k] > 0.5 * pos.steps[k].slope_est          # tau > 0 for SPD H
