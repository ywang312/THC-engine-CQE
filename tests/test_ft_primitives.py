"""Tests of the M3 primitives (``thc_engine_cqe.ft.primitives``): parameters, phase-gradient bits and
rounding (F1, F2), the Jacobi-Anger degree and GQSP costs (F8), and amplitude-estimation counts (F11).

References independent of the module: the reference numbers of the M3 specification (Appendix A,
recomputed by the critic's ``misc_check.py``), brute-force evaluations of the defining inequalities,
complex exponentials, and the truncated Jacobi-Anger series summed with ``scipy.special.jv`` against
``numpy.exp``.
"""

import dataclasses
import math

import numpy as np
import pytest
import scipy.special

from thc_engine_cqe.ft import primitives as pr
from thc_engine_cqe.ft.primitives import FTParameters

# Specification section 3: every field and its default.
SPEC_DEFAULTS = {
    "delta_step": 1e-3, "delta_readout": 1e-5, "b_min": 4, "chi": 10, "beta": 16, "eps_evol": 1e-6,
    "r": 0.3, "kind": "spin_free", "kappa": 1.0, "kappa_practical": 2.0, "kappa_g": (1.0, 10.0, 100.0),
    "n_candidates": 11, "eps_ls": 1.6e-3, "target": 1.6e-3, "de_qpe": 1e-3, "de_fact": 5e-4,
    "infid_tol": 1e-4, "delta_conf": 0.01, "e_tol_energy": 1e-3, "sigma_final": 3e-4,
    "p_fail_shot": 1e-5, "p_fail_ae": 1e-4, "p_fail_qpe": 1e-2,
    "aux_deltas": (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2),
    "n_settings": 5000, "seed": 7,
}


def _ja_degree_brute(tau: float, delta: float) -> int:
    """R by the definition, summing |J_j| over j = R+1 .. R+60 term by term (as misc_check.py)."""
    for r in range(400):
        if 2.0 * sum(abs(scipy.special.jv(j, tau)) for j in range(r + 1, r + 61)) <= delta:
            return r
    raise AssertionError("no degree below 400")


def test_parameters_have_the_spec_defaults_and_are_frozen():
    """Every field of specification section 3 exists with its default (the binding table, typed in
    here), the dataclass is frozen, and the line-search candidates are 0.00625 * 2^j, j = 0..10 (the M2
    hardware rule, benchmarks/run_optimizers.py)."""
    p = FTParameters()
    assert {f.name for f in dataclasses.fields(p)} == set(SPEC_DEFAULTS)
    for name, value in SPEC_DEFAULTS.items():
        assert getattr(p, name) == value, name
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.delta_step = 1e-2
    assert p.eps_candidates == tuple(0.00625 * 2 ** j for j in range(11))
    assert p.eps_candidates[0] == 0.00625 and p.eps_candidates[-1] == 6.4
    q = dataclasses.replace(p, aux_deltas=[0.1, 0.2], n_candidates=3, kind="spin_orbital")
    assert q.aux_deltas == (0.1, 0.2) and q.eps_candidates == (0.00625, 0.0125, 0.025)
    assert hash(q) != hash(p)


@pytest.mark.parametrize("bad", [
    {"kind": "spin"}, {"delta_step": 0.0}, {"r": -0.3}, {"b_min": 2}, {"beta": 2}, {"b_min": 4.5},
    {"n_candidates": 0}, {"p_fail_shot": 1.0}, {"delta_conf": 0.0}, {"aux_deltas": ()},
    {"kappa_g": (1.0, -1.0)}, {"seed": -1},
])
def test_parameters_reject_invalid_values(bad):
    """Invalid budgets, kinds and bit counts raise at construction (they would make every cost
    meaningless downstream)."""
    with pytest.raises(ValueError):
        FTParameters(**bad)


def test_phase_gradient_bits_reference_values():
    """b = 15 / 18 / 21 for A = 10 / 64 / 393 additions at delta = 1e-3 (specification section 7; the
    H4 and H6 step counts of Appendix A), and on a grid b is the smallest integer >= b_min with
    A pi / 2^b <= delta, found here by brute force."""
    assert [pr.phase_gradient_bits(a, 1e-3) for a in (10, 64, 393)] == [15, 18, 21]
    assert pr.phase_gradient_bits(0, 1e-3) == 4
    assert pr.phase_gradient_bits(1, 1.0, b_min=4) == 4            # pi / 2^2 <= 1, floored at b_min
    for a in (1, 2, 3, 7, 100, 1000, 12345):
        for delta in (1.0, 1e-2, 1e-3, 1e-5, 1e-8):
            for b_min in (3, 4, 10):
                ref = next(b for b in range(b_min, 200) if a * math.pi / 2.0 ** b <= delta)
                assert pr.phase_gradient_bits(a, delta, b_min) == ref
    with pytest.raises(ValueError):
        pr.phase_gradient_bits(-1, 1e-3)
    with pytest.raises(ValueError):
        pr.phase_gradient_bits(10, 0.0)


def test_addition_toffolis():
    """b - 2 Toffolis per phase-gradient addition (OpenFermion's cost4df = 4 n (beta - 2)); fewer than
    3 bits raise (specification section 6.1)."""
    assert pr.addition_toffolis(16) == 14
    assert pr.addition_toffolis(3) == 1
    for bad in (2, 0, 3.5):
        with pytest.raises(ValueError):
            pr.addition_toffolis(bad)


def test_rounding_error_is_bounded_by_pi_over_2b():
    """On random angles, including negative and large ones, v lies in [0, 2^b), the wrapped angle error
    is <= pi / 2^b, and it equals the distance of the two phases on the unit circle computed with
    complex exponentials (2 arcsin(|exp(i theta) - exp(2 pi i v / 2^b)| / 2))."""
    rng = np.random.default_rng(11)
    thetas = np.concatenate([rng.uniform(-50.0, 50.0, 400), [0.0, math.pi, -math.pi, 2 * math.pi]])
    for b in (1, 3, 6, 10, 16, 21):
        for theta in thetas:
            v = pr.round_angle(float(theta), b)
            assert 0 <= v < 2 ** b
            err = pr.rounding_error(float(theta), b)
            assert err <= math.pi / 2 ** b * (1 + 1e-12)
            chord = abs(np.exp(1j * theta) - np.exp(2j * math.pi * v / 2 ** b))
            assert err == pytest.approx(2 * math.asin(min(chord / 2, 1.0)), abs=1e-12)
    assert pr.round_angle(2 * math.pi * 5 / 2 ** 6, 6) == 5
    assert pr.round_angle(-2 * math.pi / 2 ** 6, 6) == 63
    with pytest.raises(ValueError):
        pr.round_angle(float("nan"), 6)


def test_jacobi_anger_reference_degrees():
    """R(tau, delta) of Appendix A: 3, 4, 5, 6 at tau = 0.2, 0.5, 0.79, 1.16 (delta = 1e-4) and 14, 17
    at tau = 4.87, 6.81 (delta = 1e-6), and agreement with a brute-force evaluation of the definition
    (term-by-term Bessel sums, as the critic's misc_check.py) on random (tau, delta)."""
    ref = {(0.2, 1e-4): 3, (0.5, 1e-4): 4, (0.79, 1e-4): 5, (1.16, 1e-4): 6, (4.87, 1e-6): 14,
           (6.81, 1e-6): 17}
    for (tau, delta), r in ref.items():
        assert pr.jacobi_anger_degree(tau, delta) == r
        assert pr.jacobi_anger_degree(-tau, delta) == r
    assert pr.jacobi_anger_degree(0.0, 1e-6) == 0
    rng = np.random.default_rng(5)
    for _ in range(40):
        tau, delta = rng.uniform(0.0, 25.0), 10.0 ** rng.uniform(-13, -1)
        assert pr.jacobi_anger_degree(tau, delta) == _ja_degree_brute(tau, delta)
    with pytest.raises(ValueError):
        pr.jacobi_anger_degree(1.0, 0.0)


def test_jacobi_anger_degree_is_monotone():
    """R is non-decreasing in tau (grid 0..30) and in 1/delta (1e-2..1e-14): more time or more
    precision never needs fewer walk calls."""
    taus = np.linspace(0.0, 30.0, 301)
    deltas = 10.0 ** -np.arange(2, 15)
    table = np.array([[pr.jacobi_anger_degree(t, d) for t in taus] for d in deltas])
    assert np.all(np.diff(table, axis=1) >= 0)
    assert np.all(np.diff(table, axis=0) >= 0)


@pytest.mark.parametrize("tau,delta", [(0.79, 1e-4), (3.0, 1e-6), (6.81, 1e-6), (20.0, 1e-10)])
def test_truncated_jacobi_anger_series_meets_the_budget(tau, delta):
    """The Laurent polynomial sum_{|k|<=R} (-i)^k J_k(tau) z^k at R = R(tau, delta) equals
    exp(-i tau cos theta) within delta on a theta grid (numpy.exp as the reference): the truncation that
    F8 costs is the evolution operator on every walk eigenvalue."""
    r = pr.jacobi_anger_degree(tau, delta)
    theta = np.linspace(0.0, 2 * np.pi, 1001)
    k = np.arange(-r, r + 1)
    series = (((-1j) ** k * scipy.special.jv(k, tau))[None, :] * np.exp(1j * np.outer(theta, k))).sum(axis=1)
    assert np.max(np.abs(series - np.exp(-1j * tau * np.cos(theta)))) <= delta


def test_gqsp_and_evolution_costs():
    """gqsp_walk_calls == 2R, the rotation count 3 (n + 1)(b - 2), and C_evol = 2R C_W + 3 (2R + 1)(beta - 2)
    written out from F8, e.g. H6 (lambda_DF = 9.726, C_W = 1309) at t = 0.7: R = 17, 45976 Toffolis."""
    for tau, delta in [(0.79, 1e-4), (6.81, 1e-6), (12.0, 1e-8)]:
        assert pr.gqsp_walk_calls(tau, delta) == 2 * pr.jacobi_anger_degree(tau, delta)
    assert pr.gqsp_rotation_toffolis(0, 16) == 42
    assert pr.gqsp_rotation_toffolis(34, 16) == 3 * 35 * 14
    assert pr.evolution_toffolis(9.726, 0.7, 1309) == 34 * 1309 + 3 * 35 * 14 == 45976
    for lam, t, c_w, delta, beta in [(4.935, 0.1, 868, 1e-6, 16), (9.4, 1.2, 1420, 1e-4, 12)]:
        r = pr.jacobi_anger_degree(lam * t, delta)
        expected = 2 * r * c_w + 3 * (2 * r + 1) * (beta - 2)
        assert pr.evolution_toffolis(lam, t, c_w, delta=delta, beta=beta) == expected
    assert pr.evolution_toffolis(9.726, 0.0, 1309) == 0                  # R = 0: the identity
    assert pr.evolution_toffolis(9.726, -0.7, 1309) == 45976              # only |t| matters
    with pytest.raises(ValueError):
        pr.evolution_toffolis(0.0, 0.7, 1309)
    with pytest.raises(ValueError):
        pr.gqsp_rotation_toffolis(-1, 16)


def test_ae_applications():
    """N = 2 pi kappa alpha / eps (F11): linear in alpha and kappa, inverse in eps."""
    assert pr.ae_applications(1.0, 1.0) == pytest.approx(2 * math.pi)
    assert pr.ae_applications(9.726, 1.6e-3, kappa=2.0) == pytest.approx(4 * math.pi * 9.726 / 1.6e-3)
    assert pr.ae_applications(0.0, 1e-3) == 0.0
    for bad in [(1.0, 0.0, 1.0), (-1.0, 1e-3, 1.0), (1.0, 1e-3, 0.0)]:
        with pytest.raises(ValueError):
            pr.ae_applications(*bad)
