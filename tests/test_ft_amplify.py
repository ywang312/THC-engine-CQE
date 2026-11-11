"""Tests of ft.amplify: AA counts, step filters, two-stage QPE, scheme costs, savings and break-even (F43, F44).

References: a full scan over the Grover iteration count; Monte Carlo of the exponential search;
an explicit simulation of the Yoder-Low-Chuang phase sequence in the two-dimensional
good/bad subspace; dense-grid evaluation of the constructed filter polynomial; a brute-force
scan of the coarse-QPE length with kernels from plain FFTs; the defining formulas of the scheme
table, ``ft.handoff.handoff_table`` for the repetition column, and reference numbers of the M4 cost
model (the scheme table, the filter degrees) [build check] with their tolerances.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from thc_engine_cqe.diagnostics.thc_floors import ExactReference
from thc_engine_cqe.ft import acceptance as acc
from thc_engine_cqe.ft import amplify as amp
from thc_engine_cqe.ft import certify, handoff
from thc_engine_cqe.ft import qpe as ftqpe
from thc_engine_cqe.ft import symmetry
from thc_engine_cqe.ft.primitives import FTParameters
from thc_engine_cqe.ft.trajectory import TrajectoryRecord, second_order_step

DE = 1e-3
_CACHE: dict = {}

# The hand-off scheme table of the M4 cost model (C_QPE; "rep" with F40's verification runs; None: no certified U).
_SCHEME_TABLE = {
    ("n2_r2.0", "HF"): (3.939, 3.001, 3.514, 1.654, 1.765, 2.622, 1.441, 1.934),
    ("n2_r2.0", "Psi1"): (2.515, 2.515, 3.200, 1.548, 1.697, 2.036, 1.401, 1.597),
    ("n2_r2.0", "Psi2"): (2.259, 2.259, 3.232, 1.537, 1.749, 1.964, 1.455, 1.580),
    ("n2_r2.4", "HF"): (9.928, 5.113, 6.301, 11.17, 14.10, 42.74, 1.665, 3.119),
    ("n2_r2.4", "Psi1"): (5.705, 4.054, 4.768, 8.549, 10.31, 23.00, 1.473, 2.117),
    ("n2_r2.4", "Psi2"): (4.643, 4.064, 4.471, 8.503, 9.511, 17.86, 1.477, 1.901),
    ("n2_r2.8", "HF"): (13.80, 6.121, 7.633, None, None, None, 1.832, 4.038),
    ("n2_r2.8", "Psi1"): (7.072, 4.323, 5.261, None, None, None, 1.535, 2.442),
    ("n2_r2.8", "Psi2"): (5.327, 4.097, 4.680, None, None, None, 1.501, 2.061),
}
_SCHEME_KEYS = ("rep", "aa", "bbht", "lt", "lt_bbht", "2s", "lt_bbht_L", "2s_L")


# ------------------------------------------------------------------ amplitude amplification
def _grover_scan(a: float) -> tuple[float, int]:
    phi = math.asin(math.sqrt(a))
    best = (math.inf, 0)
    m = 0
    while (2 * m + 1) * phi < math.pi:
        s = math.sin((2 * m + 1) * phi) ** 2
        if s > 0:
            best = min(best, ((2 * m + 1) / s, m))
        m += 1
    return best


def test_grover_brute_force():
    """N*(a) and m* equal a full scan over m with (2m + 1) phi < pi, on fixed and random a."""
    rng = np.random.default_rng(3)
    for a in [0.999, 0.9, 0.5, 0.4, 0.32, 0.3, 0.25, 0.2, 0.1, 0.0955, 0.05, 0.01, 1e-3, 1e-4,
              *rng.uniform(1e-4, 1.0, 40), *np.exp(rng.uniform(math.log(1e-6), math.log(1e-2), 20))]:
        n, m = amp.grover_calls(a)
        n_ref, m_ref = _grover_scan(a)
        assert n == pytest.approx(n_ref, rel=1e-13) and m == m_ref
    assert amp.grover_calls(1.0) == (1.0, 0)


def test_grover_values():
    """N* sqrt(a) = 1.3801 at a = 1e-6 (1e-4); N*(0.25) = 3 (m = 1); N*(0.32) = 3.125 (m = 0); tan x = 2x at
    x = 1.165561 with x/sin^2 x = 1.380050; N* < 1/a iff a < 0.32; the bump on (0.25, 0.32)."""
    n, _ = amp.grover_calls(1e-6)
    assert abs(n * 1e-3 - 1.3801) < 1e-4
    assert amp.grover_calls(0.25)[1] == 1 and amp.grover_calls(0.25)[0] == pytest.approx(3.0, rel=1e-12)
    assert amp.grover_calls(0.32) == (pytest.approx(3.125, rel=1e-12), 0)
    x = 1.165561
    assert abs(math.tan(x) - 2 * x) < 1e-5 and abs(x / math.sin(x) ** 2 - 1.380050) < 1e-6
    for a in np.linspace(0.01, 0.99, 99):
        n, m = amp.grover_calls(a)
        assert (n < 1.0 / a - 1e-12) == (a < 0.32 - 1e-9)
    assert amp.grover_calls(0.25)[0] < amp.grover_calls(0.3)[0] < amp.grover_calls(0.32)[0]


def _bbht_monte_carlo(a: float, n_runs: int, rng: np.random.Generator) -> np.ndarray:
    phi = math.asin(math.sqrt(a))
    cost = np.zeros(n_runs)
    alive = np.ones(n_runs, dtype=bool)
    r = 0
    while alive.any():
        m = math.ceil(1.2 ** r - 1e-12)
        idx = np.flatnonzero(alive)
        j = rng.integers(0, m, idx.size)
        cost[idx] += 2 * j + 1
        win = rng.random(idx.size) < np.sin((2 * j + 1) * phi) ** 2
        alive[idx[win]] = False
        r += 1
    return cost


@pytest.mark.parametrize("a", [0.01, 0.1])
def test_exponential_search_monte_carlo(a):
    """E[N] of the exponential search (growth 6/5) against 1e5 simulated runs within 3 standard errors."""
    rng = np.random.default_rng(11)
    cost = _bbht_monte_carlo(a, 100_000, rng)
    ref = amp.exponential_search_calls(a)
    assert abs(cost.mean() - ref) < 3 * cost.std() / math.sqrt(cost.size)


def test_exponential_search_values():
    """E_BBHT sqrt(a) in [1.8, 2.9] for a <= 0.1 [design; build check 1.796 at a = 0.1 to 2.899 at a = 5e-6, so
    [1.79, 2.90]]; E(1) = 1; the stage law P(M) closed form."""
    vals = [amp.exponential_search_calls(a) * math.sqrt(a) for a in np.geomspace(1e-8, 0.1, 400)]
    assert 1.79 <= min(vals) <= 1.80 and 2.89 <= max(vals) <= 2.90
    assert amp.exponential_search_calls(1.0) == 1.0
    for a in (0.3, 0.02, 1e-5):
        phi = math.asin(math.sqrt(a))
        for m in (1, 5, 64, 65, 300):
            direct = float(np.mean(np.sin((2 * np.arange(m) + 1) * phi) ** 2))
            assert amp._stage_success(m, phi) == pytest.approx(direct, rel=1e-10, abs=1e-14)
    with pytest.raises(ValueError):
        amp.exponential_search_calls(0.1, growth=2.0)
    with pytest.raises(ValueError):
        amp.exponential_search_calls(0.0)


def _ylc_simulation(a: float, n_calls: int, delta: float) -> float:
    """Success probability of the YLC sequence by explicit generalized reflections in span{good, bad}."""
    l = (n_calls - 1) // 2
    gamma = 1.0 / math.cosh(math.acosh(1.0 / delta) / n_calls)
    s = np.array([math.sqrt(a), math.sqrt(1 - a)], dtype=complex)
    alphas = [2 * math.atan2(1.0, math.tan(2 * math.pi * j / n_calls) * math.sqrt(1 - gamma ** 2))
              for j in range(1, l + 1)]
    betas = [-alphas[l - j] for j in range(1, l + 1)]
    psi = s.copy()
    p_s = np.outer(s, s.conj())
    p_t = np.array([[1, 0], [0, 0]], dtype=complex)
    for j in range(l):
        psi = (np.eye(2) - (1 - np.exp(-1j * alphas[j])) * p_s) @ ((np.eye(2) - (1 - np.exp(1j * betas[j])) * p_t) @ psi)
    return float(abs(psi[0]) ** 2)


def test_fixed_point_closed_form():
    """The closed form equals the explicit phase sequence to 1e-12; success >= 1 - delta^2 for every a >= a_lb."""
    for n_calls in (1, 3, 11, 51):
        for delta in (0.3, 0.05, 0.01):
            for a in (0.9, 0.7, 0.2, 0.02, 2e-3, 0.0):
                assert amp.fixed_point_success(a, n_calls, delta) == pytest.approx(
                    _ylc_simulation(a, n_calls, delta), abs=1e-12)
    for a_lb in (0.5, 0.2, 0.01, 1e-3, 1e-4):
        for delta in (0.1, 0.01):
            n = amp.fixed_point_calls(a_lb, delta)
            assert n % 2 == 1 and n >= math.acosh(1 / delta) / math.atanh(math.sqrt(a_lb)) - 1e-9
            assert min(amp.fixed_point_success(a, n, delta) for a in np.geomspace(a_lb, 1.0, 200)) >= 1 - delta ** 2 - 1e-12
            if n >= 3:                                         # L - 2 misses the guarantee at a_lb
                assert amp.fixed_point_success(a_lb, n - 2, delta) < 1 - delta ** 2
    assert amp.fixed_point_calls(1.0, 0.01) == 1
    with pytest.raises(ValueError):
        amp.fixed_point_success(0.5, 4, 0.01)


# ------------------------------------------------------------------ step filters
def _filter_values(coef: np.ndarray, theta: np.ndarray) -> np.ndarray:
    k = np.arange(coef.size)
    return np.cos(np.outer(theta, k)) @ coef


@pytest.mark.parametrize("eps,band", [(1e-2, (8.8, 10.0)), (1e-3, (17.5, 18.4))])
def test_step_filter_degree(eps, band):
    """d dtheta in [8.8, 10.0] at eps_f = 1e-2 and [17.5, 18.4] at 1e-3 (dtheta = 0.02, 0.05) [build check: 8.92 /
    9.85 and 18.20 / 18.20; the bound 18.2 of a grid-only check rises with the exact band edges]; the filter meets
    its band conditions on a grid 16x denser than the construction's (|g| <= 1 + 1e-5); d non-increasing in dtheta;
    degree d - 1 misses eps."""
    th = math.pi / 2 + 0.17
    for dth in (0.02, 0.05):
        d, coef = amp.step_filter_polynomial(th, th - dth, eps)
        assert d == amp.step_filter_degree(th, th - dth, eps) and coef.size == d + 1
        assert band[0] <= d * dth <= band[1]
        n = 16 * (1 << max(14, int(math.ceil(math.log2(64.0 / dth))) + 3))
        theta = np.linspace(0.0, math.pi, n // 2 + 1)
        g = _filter_values(coef, theta)
        assert np.max(np.abs(g)) <= 1.0 + 1.1e-6
        assert np.max(np.abs(1.0 - g[theta >= th])) <= eps * (1 + 1e-3)
        assert np.max(np.abs(g[theta <= th - dth])) <= eps * (1 + 1e-3)
        cons = amp._StepConstruction(th, th - dth)
        assert cons.error(d - 1)[0] > eps
    degrees = [amp.step_filter_degree(th, th - dth, 1e-2) for dth in (0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.2, 0.4)]
    assert all(x >= y for x, y in zip(degrees, degrees[1:]))
    pins = {(1e-2, 0.02): 446, (1e-2, 0.05): 197, (1e-3, 0.02): 910, (1e-3, 0.05): 364}       # [build check]
    for dth in (0.02, 0.05):
        assert amp.step_filter_degree(th, th - dth, eps) == pins[eps, dth]


def test_filter_band_edges_need_the_exact_check():
    """At eps_f = 1e-3, dtheta = 0.05 a grid-only check accepts d = 354, whose error at the exact pass edge is 11%
    above eps_f; with the edges the construction needs 364. At 1e-2 the edge check changes nothing (d = 197)."""
    th = math.pi / 2 + 0.17
    for eps, d_grid, d_edge in ((1e-3, 354, 364), (1e-2, 197, 197)):
        cons = amp._StepConstruction(th, th - 0.05)

        def grid_error(d):
            best = math.inf
            for coef in cons.coefs:
                g, _ = cons._values(coef, d)
                best = min(best, max(float(np.max(np.abs(1 - g[cons.pass_band]))), float(np.max(np.abs(g[cons.stop_band])))))
            return best

        assert grid_error(d_grid) <= eps < grid_error(d_grid - 1)
        assert amp.step_filter_degree(th, th - 0.05, eps) == d_edge
        if d_edge > d_grid:
            assert 1.05 * eps < cons.error(d_grid)[0] < 1.15 * eps


def test_filter_degree_from_the_table():
    """filter_degree = ceil(c_f(eps rounded down)/dtheta) for dtheta <= 0.1, the construction beyond; c_f(1e-2) = 9.999,
    c_f(1e-3) = 18.483 (1.01 x 9.9 and 1.01 x 18.3); at dtheta = 0.03, 1e-3 the construction's 607 is bounded (617)."""
    assert len(amp.FILTER_TABLE) == 33
    assert [e for e, _ in amp.FILTER_TABLE] == pytest.approx([10 ** (-j / 4) for j in range(8, 41)], rel=1e-14)
    cf = dict(amp.FILTER_TABLE)
    assert cf[1e-2] == pytest.approx(9.999, rel=1e-12) and cf[1e-3] == pytest.approx(18.483, rel=1e-12)
    assert all(x[1] < y[1] for x, y in zip(amp.FILTER_TABLE, amp.FILTER_TABLE[1:]))
    th = math.pi / 2 + 0.17
    assert amp.filter_degree(th, th - 0.03, 1e-3) == 617 >= amp.step_filter_degree(th, th - 0.03, 1e-3) == 607
    assert amp.filter_degree(th, th - 0.05, 1e-2) == math.ceil(9.999 / 0.05)
    assert amp.filter_degree(th, th - 0.05, 2e-2) == amp.filter_degree(th, th - 0.05, 1e-2)      # rounded down
    assert amp.filter_degree(th, th - 0.05, 0.007) == math.ceil(cf[10 ** (-9 / 4)] / 0.05 - 1e-12)
    assert amp.filter_degree(th, th - 0.3, 1e-2) == amp.step_filter_degree(th, th - 0.3, 1e-2)  # dtheta > 0.1
    with pytest.raises(ValueError):
        amp.filter_degree(th, th + 0.01, 1e-2)


@pytest.mark.slow
def test_filter_table():
    """build_filter_table() reproduces FILTER_TABLE; at 12 random (theta_pass, dtheta, eps_f) in [pi/2, pi/2 + 0.8] x
    [0.005, 0.1] x [1e-8, 1e-2]: filter_degree >= the construction at eps_f and <= 1.13 x the construction at eps_f
    rounded down to the grid."""
    table = amp.build_filter_table()
    assert len(table) == len(amp.FILTER_TABLE)
    for (e, c), (e_ref, c_ref) in zip(table, amp.FILTER_TABLE):
        assert e == pytest.approx(e_ref, rel=1e-14) and c == pytest.approx(c_ref, rel=1e-12)
    rng = np.random.default_rng(2026)
    for _ in range(12):
        th = rng.uniform(math.pi / 2, math.pi / 2 + 0.8)
        dth = math.exp(rng.uniform(math.log(0.005), math.log(0.1)))
        eps = 10 ** rng.uniform(-8, -2)
        d = amp.filter_degree(th, th - dth, eps)
        assert d >= amp.step_filter_degree(th, th - dth, eps)
        eps_grid = 10 ** (-math.ceil(-4 * math.log10(eps) - 1e-9) / 4)
        assert d <= 1.13 * amp.step_filter_degree(th, th - dth, eps_grid)


def _n2(name: str) -> dict:
    """Selected SF QPE of N2 (SF L = 17 / 16), the model spectrum, CISD(H'), E1lb = E_1' - 0.1 mHa."""
    if name not in _CACHE:
        rank = {"n2s": 17, "n2_r2.4": 16}[name]
        a = symmetry.benchmark_system(name)
        ir = symmetry.orbital_irreps(a)
        enc = ftqpe.sf_encoding(a, rank, reference=ExactReference.build(a))
        q = ftqpe.qpe_cost(enc, DE)
        spec = acc.model_spectrum(a, q, ir)
        _CACHE[name] = {"active": a, "irreps": ir, "qpe": q, "spec": spec,
                        "u": certify.truncated_ci_energy(enc.model(a), 2),
                        "e1lb": certify.e1_lower_bound(spec.e1_singlet, model_spectrum=True)}
    return _CACHE[name]


def test_filter_cost_ratio():
    """C_filt/C_QPE within 10% of (4 c_f/pi) sin(theta_bar) dE/(E1lb - U) (n2s: U = CISD(H') and E1lb - W_L); the
    F8 count 2d C_W + 3(2d + 1)(beta - 2) on Q_walk + 1 qubits."""
    s = _n2("n2s")
    enc, spec, q = s["qpe"].encoding, s["spec"], s["qpe"]
    params = FTParameters()
    for u in (s["u"], s["e1lb"] - amp.W_L):
        f = amp.filter_cost(enc, spec.c, u, s["e1lb"])
        tp = math.acos((u - spec.c) / spec.lam)
        ts = math.acos((s["e1lb"] - spec.c) / spec.lam)
        assert f.degree == amp.filter_degree(tp, ts, 1e-2)
        assert f.toffolis == 2 * f.degree * enc.toffoli_per_step + 3 * (2 * f.degree + 1) * (params.beta - 2)
        assert f.logical_qubits == enc.ancilla_walk + 1 and f.eps == 1e-2
        closed = 4 * 9.999 / math.pi * math.sin(0.5 * (tp + ts)) * DE / (s["e1lb"] - u)
        assert f.toffolis / q.toffolis == pytest.approx(closed, rel=0.1)
    with pytest.raises(ValueError):
        amp.filter_cost(enc, spec.c, s["e1lb"], s["u"])
    with pytest.raises(ValueError):
        amp.FilterCost(degree=0, toffolis=1.0, logical_qubits=1, u_upper=0.0, e1_lower=1.0, eps=1e-2)


# ------------------------------------------------------------------ two-stage QPE
def _plain_kernel(theta: float, m: int) -> np.ndarray:
    """One branch |A(m|theta)|^2 of the sine window by a plain FFT of a_x exp(i x theta)."""
    a = acc.phase_window(m, "sine")
    amp_ = np.fft.fft(a * np.exp(1j * theta * np.arange(m)))
    return np.abs(amp_) ** 2 / m


def _coarse_brute(m: int, th_u: float, th_1: float, eps: float, n_sub: int = 32) -> bool:
    """Every level within one grid spacing beyond each edge and every cut between grid phases, by explicit masks."""
    step = 2 * math.pi / m
    ph = 2 * np.pi * np.arange(m) / m
    folded = np.where(ph <= np.pi, ph, 2 * np.pi - ph)
    t = np.arange(n_sub + 1) / max(n_sub, 1)            # n_sub = 0: the two edge levels only
    lo = np.array([_plain_kernel(th_1 - x * step, m) for x in t])
    hi = np.array([_plain_kernel(th_u + x * step, m) for x in t])
    masks = (folded[None, :] >= np.unique(folded)[:, None]).astype(float)       # one accepted set per cut
    leak = (lo @ masks.T).max(axis=0)
    keep = (hi @ masks.T).min(axis=0)
    return bool(np.any((leak <= eps) & (keep >= 1 - eps)))


def test_coarse_edge_levels_are_a_lottery():
    """With only the two levels exactly at E1lb and U the condition holds at M = 182, at no M from 183 to 246 and at
    about a third of the lengths up to 697 [build check: 175 of 516] (theta_U = pi/2 + 0.17, dtheta = 0.1,
    eps_p = 1e-4): the sine-window lattice law vanishes at half-integer distances; the robust condition (every level
    beyond each edge) needs 704."""
    th, dth, eps = math.pi / 2 + 0.17, 0.1, 1e-4

    def edge_only(m):
        return _coarse_brute(m, th, th - dth, eps, n_sub=0)

    assert edge_only(182) and not any(edge_only(m) for m in range(4, 182))
    assert not any(edge_only(m) for m in range(183, 247))
    assert 150 < sum(edge_only(m) for m in range(182, 698)) < 200
    assert amp.coarse_qpe_points(th, th - dth, eps) == 704


def test_two_stage_qpe():
    """coarse_qpe_points equals a brute-force scan of M (every M below fails, M_c passes) with every level beyond each
    edge (32 sub-offsets, plain FFT kernels, every cut); C_c = (M_c/2) C_W; C_c/C_QPE within 10% of
    4 sin(theta_0) D_p dE/(E1lb - U), D_p = 6.0 at eps_p = 1e-4 (n2s, U = E1lb - W_L)."""
    th = math.pi / 2 + 0.17
    for dth, eps in ((0.1, 1e-4), (0.3, 1e-3)):
        m_c = amp.coarse_qpe_points(th, th - dth, eps)
        assert _coarse_brute(m_c, th, th - dth, eps)
        assert not any(_coarse_brute(m, th, th - dth, eps) for m in range(4, m_c))
    # the continuum path (M > 4096) agrees with the finite-register law at its answer
    m_big = amp.coarse_qpe_points(th, th - 0.0133, 1e-4)
    assert m_big > 4096 and amp._coarse_ok_exact(m_big, th, th - 0.0133, 1e-4)
    s = _n2("n2s")
    enc, spec, q = s["qpe"].encoding, s["spec"], s["qpe"]
    for u in (s["e1lb"] - amp.W_L, s["u"]):
        c_c = amp.coarse_qpe_toffolis(enc, spec.c, u, s["e1lb"])
        tu, t1 = math.acos((u - spec.c) / spec.lam), math.acos((s["e1lb"] - spec.c) / spec.lam)
        assert c_c == 0.5 * amp.coarse_qpe_points(tu, t1, 1e-4) * enc.toffoli_per_step
        closed = 4 * spec.sin_theta0 * 6.0 * DE / (s["e1lb"] - u)
        assert c_c / q.toffolis == pytest.approx(closed, rel=0.1)
    assert abs(amp._tail_distance(1e-4) - 6.0) < 0.05
    with pytest.raises(ValueError):
        amp.coarse_qpe_points(th, th - 0.1, 1e-4, window="rect")


def test_coarse_length_from_the_continuum_is_minimal():
    """Above M = 4096 the coarse length comes from the continuum lattice law, and it is the smallest length at which
    the finite-register law passes: n2_r2.4's E1lb-referenced edges (theta_U = 1.722595071783865, theta_1 =
    1.7092810578657096, dtheta = 0.013314, eps_p = 1e-4) give M_c = 5216, where the finite-register condition holds
    (the code's check and the brute force over every cut with plain FFT kernels) and at no M in [5096, 5216)."""
    tu, t1 = 1.722595071783865, 1.7092810578657096
    assert amp.coarse_qpe_points(tu, t1, 1e-4) == 5216
    assert amp._coarse_ok_exact(5216, tu, t1, 1e-4) and _coarse_brute(5216, tu, t1, 1e-4)
    assert not _coarse_brute(5215, tu, t1, 1e-4)
    assert not any(amp._coarse_ok_exact(m, tu, t1, 1e-4) for m in range(5096, 5216))


# ------------------------------------------------------------------ scheme costs
def test_scheme_costs():
    """The scheme table [build check, 1e-3 relative]: p = 0.11201276 gives rep 8.928, aa 4.113, bbht 5.301 and one more
    each with n_verify = 2; p = 0.21252473 gives 4.705, 3.054, 3.768; with n2_r2.0's c_filt = 0.21779, c_coarse =
    0.41194, c_filt_l = 0.12543, c_coarse_l = 0.23724: p = 0.25390 gives lt 1.654, lt_bbht 1.765, 2s 2.622,
    lt_bbht_L 1.441, 2s_L 1.934 and p = 0.39761 gives 1.548, 1.697, 2.036, 1.401, 1.597; the formulas with C(Psi)
    and C_loop."""
    for p, ref in ((0.11201276, (8.928, 4.113, 5.301)), (0.21252473, (4.705, 3.054, 3.768))):
        for nv in (1, 2):
            c = amp.scheme_costs(p, 0.0, 0.0, 1.0, n_verify=nv)
            for key, val in zip(("rep", "aa", "bbht"), ref):
                assert c[key] == pytest.approx(val + nv - 1, rel=1e-3)
            assert c["fp"] is None and c["lt"] is None and c["2s"] is None and c["lt_bbht_L"] is None
    kw = dict(c_filt=0.21779, c_coarse=0.41194, c_filt_l=0.12543, c_coarse_l=0.23724)
    for p, ref in ((0.25390, (1.654, 1.765, 2.622, 1.441, 1.934)), (0.39761, (1.548, 1.697, 2.036, 1.401, 1.597))):
        c = amp.scheme_costs(p, 0.0, 0.0, 1.0, **kw)
        for key, val in zip(("lt", "lt_bbht", "2s", "lt_bbht_L", "2s_L"), ref):
            assert c[key] == pytest.approx(val, rel=1e-3)
    # general formulas
    p, cp, cl, cq, plb = 0.3, 0.05, 2.0, 1.0, 0.2
    c = amp.scheme_costs(p, cp, cl, cq, p_lb=plb, n_verify=3, **kw)
    n_star, e_b = amp.grover_calls(p)[0], amp.exponential_search_calls(p)
    assert c["rep"] == pytest.approx(cl + (cp + cq) / p + 2 * cq)
    assert c["aa"] == pytest.approx(cl + n_star * (cp + cq) + 2 * cq)
    assert c["bbht"] == pytest.approx(cl + e_b * (cp + cq) + 2 * cq)
    assert c["fp"] == pytest.approx(cl + amp.fixed_point_calls(plb, 0.01) * (cp + cq) + 2 * cq)
    assert c["lt"] == pytest.approx(cl + n_star * (cp + kw["c_filt"]) + cq)
    assert c["lt_bbht_L"] == pytest.approx(cl + e_b * (cp + kw["c_filt_l"]) + cq)
    assert c["2s_L"] == pytest.approx(cl + (cp + kw["c_coarse_l"]) / p + cq)
    assert amp.scheme_costs(p, cp, cl, cq, p_lb=0.0)["fp"] is None
    assert amp.scheme_costs(p, cp, math.inf, cq)["rep"] == math.inf
    for bad in (dict(p=0.0), dict(p=1.5), dict(c_qpe=0.0), dict(c_prep=-1.0), dict(n_verify=0)):
        args = dict(p=0.3, c_prep=0.0, c_loop=0.0, c_qpe=1.0, n_verify=1) | bad
        with pytest.raises(ValueError):
            amp.scheme_costs(args["p"], args["c_prep"], args["c_loop"], args["c_qpe"], n_verify=args["n_verify"])


def test_best_realizable():
    """Never aa, lt, an "@oracle_U" key, a sensitivity key, a scheme without a cost (missing certificate, fp without
    p_lb, a non-terminating form); lt_bbht_L on every row of the scheme table; h4_r4.0's HF row (no certified U, the
    oracle-U bounds below every realizable cost) gives rep; with every U-referenced scheme missing and its oracle-U
    bound cheapest, the choice stays among the realizable schemes with a cost."""
    for row, vals in _SCHEME_TABLE.items():
        costs = dict(zip(_SCHEME_KEYS, vals)) | {"fp": None}
        assert amp.best_realizable(costs) == "lt_bbht_L", row
    h4_r40_hf = {"rep": 3.30371, "aa": 3.30371, "bbht": 4.18490, "fp": None, "lt": None, "lt_bbht": None, "2s": None,
                 "lt_bbht_L": None, "2s_L": None, "lt@oracle_U": 2.22660, "lt_bbht@oracle_U": 2.69574,
                 "2s@oracle_U": 2.91666}
    assert amp.best_realizable(h4_r40_hf) == "rep"
    rng = np.random.default_rng(5)
    for _ in range(200):
        costs = {s: (None if rng.random() < 0.3 else float(rng.uniform(1, 10))) for s in amp.SCHEMES}
        costs["rep"] = float(rng.uniform(1, 10))
        costs["lt@oracle_U"] = 0.5
        costs["aa"] = costs["lt"] = 0.1
        costs["lt_rep"] = costs["lt_rep_L"] = 0.05
        best = amp.best_realizable(costs)
        assert best in amp.REALIZABLE and best not in ("aa", "lt")
        ref = min((s for s in amp.REALIZABLE if costs.get(s) is not None), key=lambda s: costs[s])
        assert costs[best] == costs[ref]
        assert amp.best_extended(costs) == "lt_rep"                   # the sensitivity optimum
        for s in amp.ORACLE_U_SCHEMES:                                # the oracle-U bounds never stand in
            costs[s] = None
            costs[f"{s}@oracle_U"] = 0.01
        best = amp.best_realizable(costs)
        assert best not in amp.ORACLE_U_SCHEMES and costs[best] is not None and best in amp.REALIZABLE
    with pytest.raises(ValueError):
        amp.best_realizable({"aa": 1.0, "lt": 1.0, "lt_rep": 0.5})
    with pytest.raises(ValueError):
        amp.best_extended({"aa": 1.0, "lt_bbht@oracle_U": 0.5})


def _bbht_by_stages(a: float, growth: float = 1.2) -> float:
    """E[N] of the exponential search from its stages, the success of stage r the mean of sin^2((2j + 1) phi) over
    j < M_r = ceil(growth^r) by direct sums (no closed form)."""
    phi = math.asin(math.sqrt(a))
    total, alive, r = 0.0, 1.0, 0
    while alive > 1e-16:
        m = math.ceil(growth ** r - 1e-12)
        total += alive * m
        alive *= 1.0 - float(np.mean(np.sin((2 * np.arange(m) + 1) * phi) ** 2))
        r += 1
    return total


def test_state_scheme_costs_without_a_certified_u():
    """No certified U, the oracle U's filter 0.1 and coarse QPE 0.2 C_QPE, p = 0.4340820466, n_v = 2, no
    preparation or loop cost: lt_bbht and 2s have no cost ("no_certificate"); the bounds lt_bbht@oracle_U = 0.1 E_BBHT
    + 1 = 1.318490 (E_BBHT = 3.184901 from the stages), 2s@oracle_U = 0.2/p + 1 = 1.460742 ("oracle_U"); rep = 1/p + 1
    = 3.303712, bbht = E_BBHT + 1 = 4.184901; s* = rep (an oracle-U leak would make it lt_bbht)."""
    p = 0.4340820466
    e_b = _bbht_by_stages(p)
    assert e_b == pytest.approx(3.184901, abs=1e-6)
    out = amp.state_scheme_costs(p, 0.0, 0.0, 1.0, filt_oracle_u={"c_filt": 0.1, "c_coarse": 0.2}, n_verify=2,
                                 rule_b_flags=("no_certificate", "termination_not_certified"))
    c, fl = out["costs"], out["flags"]
    assert c["lt_bbht"] is None and c["2s"] is None and c["lt"] is None and c["lt_rep"] is None
    assert fl["lt_bbht"] == fl["2s"] == ["no_certificate"]
    assert c["lt_bbht@oracle_U"] == pytest.approx(0.1 * e_b + 1.0, rel=1e-12) == pytest.approx(1.318490, abs=1e-6)
    assert c["2s@oracle_U"] == pytest.approx(0.2 / p + 1.0, rel=1e-14) == pytest.approx(1.460742, abs=1e-6)
    assert c["lt@oracle_U"] == pytest.approx(0.1 * _grover_scan(p)[0] + 1.0, rel=1e-12)
    assert fl["lt_bbht@oracle_U"] == ["oracle_U"]
    assert c["rep"] == pytest.approx(1.0 / p + 1.0, rel=1e-14) == pytest.approx(3.303712, abs=1e-6)
    assert c["bbht"] == pytest.approx(e_b + 1.0, rel=1e-12) == pytest.approx(4.184901, abs=1e-6)
    assert c["aa"] == pytest.approx(_grover_scan(p)[0] + 1.0, rel=1e-12)
    assert fl["rep"] == ["no_certificate", "termination_not_certified"] and fl["fp"] == ["no_overlap_bound"]
    assert c["lt_bbht_L"] is None and fl["lt_bbht_L"] == ["not_terminating_L"]
    assert out["best_realizable"] == "rep" == amp.best_realizable(c) and out["best_extended"] == "rep"
    # with a certified U the oracle-U bounds are not reported
    out = amp.state_scheme_costs(p, 0.0, 0.0, 1.0, filt_u={"c_filt": 0.3, "c_coarse": 0.5},
                                 filt_oracle_u={"c_filt": 0.1, "c_coarse": 0.2})
    assert not any(s.endswith("@oracle_U") for s in out["costs"])
    assert out["costs"]["lt_bbht"] == pytest.approx(0.3 * e_b + 1.0, rel=1e-12)


def test_state_scheme_costs_terminate_the_l_forms_at_w_l():
    """lt_bbht_L, 2s_L and lt_rep_L need E_0' <= E1lb - W_L (F43): E1lb - E_0' = 0.05 Ha gives no cost and the flag
    "not_terminating_L" although the w_L filter is priced; 0.15 Ha gives E_BBHT(p) C_filt + C_QPE, C_c/p + C_QPE,
    C_filt/p + C_QPE, flagged "termination_not_certified" unless the termination is certified."""
    p, wl = 0.2, {"c_filt": 0.13, "c_coarse": 0.22}
    e1lb = -1.0
    out = amp.state_scheme_costs(p, 0.0, 0.0, 1.0, filt_wl=wl, e0_model=e1lb - 0.05, e1_lower=e1lb)
    for s in ("lt_bbht_L", "2s_L", "lt_rep_L"):
        assert out["costs"][s] is None and out["flags"][s] == ["not_terminating_L"]
    for certified in (False, True):
        out = amp.state_scheme_costs(p, 0.0, 0.0, 1.0, filt_wl=wl, e0_model=e1lb - 0.15, e1_lower=e1lb,
                                     w_l_certified=certified)
        c = out["costs"]
        assert c["lt_bbht_L"] == pytest.approx(_bbht_by_stages(p) * 0.13 + 1.0, rel=1e-12)
        assert c["2s_L"] == pytest.approx(0.22 / p + 1.0, rel=1e-14)
        assert c["lt_rep_L"] == pytest.approx(0.13 / p + 1.0, rel=1e-14)
        assert out["flags"]["lt_bbht_L"] == ([] if certified else ["termination_not_certified"])
    out = amp.state_scheme_costs(p, 0.0, 0.0, 1.0, filt_wl=None, e0_model=e1lb - 0.15, e1_lower=e1lb)
    assert out["costs"]["2s_L"] is None and out["flags"]["2s_L"] == ["not_terminating_L"]
    with pytest.raises(ValueError):
        amp.state_scheme_costs(p, 0.0, 0.0, 1.0, filt_wl=wl, e0_model=-1.2)
    with pytest.raises(ValueError):
        amp.state_scheme_costs(p, 0.0, 0.0, 1.0, filt_wl={"c_filt": 0.1})
    with pytest.raises(ValueError):
        amp.state_scheme_costs(p, 0.0, 0.0, 1.0, n_verify=0.5)
    with pytest.raises(ValueError):
        amp.state_scheme_costs(p, 0.0, 0.0, 1.0, u_final="B_X")


# QPE sides of the quick summary run (run_m4_summary.qpe_side; n2_r2.0 and n2_r2.4 equal the hand-off
# benchmark's n2s and n2_r2.4 sides) in Hartree: E_0', E1lb, U = CISD(H') (certified or not), g
_QUICK_SIDES = {
    "n2_r2.0": (-107.43725564659529, -107.24544871965084, -107.30304025157412, True, 0.0019703283838425657),
    "n2_r2.4": (-107.43398323034428, -107.20321992235951, -107.20829683512028, True, 0.001971344288983838),
    "h4_r3.0": (-1.9708684409051023, -1.8320092597167348, -1.951414740531688, True, 0.001958956675216257),
    "h4_r4.0": (-1.8879033412879438, -1.8597645127300868, -1.8486637995499209, False, 0.0019684973903430214),
}


def test_state_scheme_costs_flags_on_the_quick_sides():
    """Per-scheme flags of F43 on the quick QPE sides, rule B's form from ft.handoff.rule_b_bracket: n2_r2.4 (B_L,
    n_v = 2, CISD certified with D_U = 2.07 but not used, L forms uncertified) rep "no_certificate",
    "termination_not_certified", lt_bbht none, lt_bbht_L "termination_not_certified"; n2_r2.0 (B_U, one run) rep
    none, lt_bbht_L "termination_not_certified"; h4_r3.0 (CISD below E1lb - W_L) lt_bbht_L none; h4_r4.0 (no U,
    E1lb - E_0' = 28 mHa) rep "no_certificate", "termination_not_certified", lt_bbht "no_certificate", lt_bbht_L
    "not_terminating_L"; the final QPE of the U-referenced forms is B_U at D_U >= 1.3, else B_L
    ("termination_not_certified"; no cost where B_L does not terminate); a non-terminating rule B leaves rep, aa,
    bbht, fp without cost."""
    de = 1e-3
    expect = {"n2_r2.0": ("B_U", 1, [], [], ["termination_not_certified"]),
              "n2_r2.4": ("B_L", 2, ["no_certificate", "termination_not_certified"], [], ["termination_not_certified"]),
              "h4_r3.0": ("B_U", 1, [], [], []),
              "h4_r4.0": ("B_L", 2, ["no_certificate", "termination_not_certified"], ["no_certificate"],
                          ["not_terminating_L"])}
    for name, (e0, e1lb, u, certified, g) in _QUICK_SIDES.items():
        br = handoff.rule_b_bracket(u if certified else None, e1lb, de, grid=g, e0_model=e0)
        form, n_v, f_rep, f_lt, f_l = expect[name]
        assert (br.form, br.n_runs, list(br.flags)) == (form, n_v, f_rep), name
        filt = {"c_filt": 0.2, "c_coarse": 0.4}
        out = amp.state_scheme_costs(0.3, 0.0, 0.0, 1.0, filt_u=filt if certified else None, filt_wl=filt,
                                     n_verify=br.n_runs, rule_b_flags=br.flags,
                                     u_final=amp.final_qpe_form(u, e1lb, de, grid=g) if certified else "B_U",
                                     e0_model=e0, e1_lower=e1lb,
                                     w_l_certified=certified and u <= e1lb - amp.W_L)
        fl = out["flags"]
        assert fl["rep"] == fl["bbht"] == f_rep and fl["lt_bbht"] == fl["2s"] == f_lt and fl["lt_bbht_L"] == f_l, name
        assert out["flags"][out["best_realizable"]] in (f_rep, f_lt, f_l)
    assert amp.final_qpe_form(-1.0 - 2.0 * de, -1.0, de, grid=2e-3) == "B_L"          # D_U = 0.5
    assert amp.final_qpe_form(-1.0 - 3.7e-3, -1.0, de, grid=2e-3) == "B_U"           # D_U = 1.35
    assert amp.final_qpe_form(-1.0 - 2.0 * de, -1.0, de, grid=None) == "B_U"
    filt = {"c_filt": 0.2, "c_coarse": 0.4}
    out = amp.state_scheme_costs(0.3, 0.0, 0.0, 1.0, filt_u=filt, u_final="B_L")
    assert out["costs"]["lt_bbht"] is not None and out["flags"]["lt_bbht"] == ["termination_not_certified"]
    out = amp.state_scheme_costs(0.3, 0.0, 0.0, 1.0, filt_u=filt, u_final="B_L", b_l_terminates=False,
                                 rule_b_terminates=False)
    assert out["costs"]["lt_bbht"] is None and out["flags"]["2s"] == ["not_terminating"]
    assert all(out["costs"][s] is None and out["flags"][s] == ["not_terminating"] for s in ("rep", "aa", "bbht", "fp"))
    assert out["best_realizable"] is None


def test_state_scheme_costs_against_the_scheme_table():
    """rep, aa, bbht, fp carry rule B's surcharge and expected runs, the U-referenced forms the certificate's cost:
    T = C_loop + extra + H_s + (n_v - 1) C_QPE and C_loop + C_U + H_s against scheme_costs and the formulas of F43;
    lt_rep = C_loop + C_U + (c + C_filt)/p + C_QPE, lt_rep_L = C_loop + (c + C_filt(W_L))/p + C_QPE."""
    rng = np.random.default_rng(21)
    for _ in range(50):
        p, c, loop, cq = float(rng.uniform(0.02, 1.0)), float(rng.uniform(0, 0.1)), float(rng.uniform(0, 3)), 1.0
        nv, extra_b, extra_u = float(rng.uniform(1.0, 3.0)), float(rng.uniform(0, 1)), float(rng.uniform(0, 1))
        fu = {"c_filt": float(rng.uniform(0.05, 3)), "c_coarse": float(rng.uniform(0.05, 5))}
        fl = {"c_filt": float(rng.uniform(0.05, 0.3)), "c_coarse": float(rng.uniform(0.05, 0.5))}
        plb = float(rng.uniform(0.01, p))
        out = amp.state_scheme_costs(p, c, loop, cq, filt_u=fu, filt_wl=fl, n_verify=nv, rule_b_extra=extra_b,
                                     u_extra=extra_u, p_lb=plb)
        t = out["costs"]
        ref_b = amp.scheme_costs(p, c, loop + extra_b, cq, p_lb=plb)
        ref_u = amp.scheme_costs(p, c, loop + extra_u, cq, c_filt=fu["c_filt"], c_coarse=fu["c_coarse"])
        ref_l = amp.scheme_costs(p, c, loop, cq, c_filt_l=fl["c_filt"], c_coarse_l=fl["c_coarse"])
        for s in ("rep", "aa", "bbht", "fp"):
            assert t[s] == pytest.approx(ref_b[s] + (nv - 1.0) * cq, rel=1e-14)
        for s in ("lt", "lt_bbht", "2s"):
            assert t[s] == pytest.approx(ref_u[s], rel=1e-14)
        for s in ("lt_bbht_L", "2s_L"):
            assert t[s] == pytest.approx(ref_l[s], rel=1e-14)
        assert t["rep"] == pytest.approx(loop + extra_b + (c + cq) / p + (nv - 1) * cq, rel=1e-13)
        assert t["lt_rep"] == pytest.approx(loop + extra_u + (c + fu["c_filt"]) / p + cq, rel=1e-13)
        assert t["lt_rep_L"] == pytest.approx(loop + (c + fl["c_filt"]) / p + cq, rel=1e-13)
        assert out["best_realizable"] == amp.best_realizable(t) and out["best_extended"] == amp.best_extended(t)
        assert set(out["flags"]) == set(t) and list(t)[:len(amp.SCHEMES)] == list(amp.SCHEMES)
        amp.SchemeRow(k=1, p=p, c_prep=c, c_loop=loop, costs=t, best_realizable=out["best_realizable"])


def test_filter_then_repeat_sensitivity():
    """The cold h4_r3.5 spin-free r = 1, k = 1 cell (C_QPE units; U = CISD(H') certified, B_U with two runs, L forms
    not terminating): p = 0.9178435172 -> 0.9687861467, C(Psi) 5.5496359e-5 -> 4.6894424e-4, C_iter = 0.0640356497,
    C_filt(U) = 0.5171712673, C_c(U) = 0.8096696857. Over the scheme set of record s* = lt_bbht at both iterates,
    T = 1.651979 -> 1.619557, the iteration pays (+0.032422); with the step filter followed by repetition: lt_rep,
    T = 1.563524 -> 1.598354, it does not pay (-0.034830). E_BBHT(p) p = 1.0027, 1.5785, 1.1565 at p = 0.30, 0.5,
    0.918 (BBHT's schedule overshoots 1/p above 0.30)."""
    for p, ref in ((0.30, 1.0027), (0.5, 1.5785), (0.918, 1.1565)):
        assert amp.exponential_search_calls(p) * p == pytest.approx(ref, abs=1e-4)
        assert _bbht_by_stages(p) * p == pytest.approx(ref, abs=1e-4)
    p, p1, ck, ck1, c_it = 0.9178435172, 0.9687861467, 5.5496359e-5, 4.6894424e-4, 0.0640356497
    fu = {"c_filt": 0.5171712673, "c_coarse": 0.8096696857}
    kw = dict(filt_u=fu, n_verify=2, u_final="B_U")
    a = amp.state_scheme_costs(p, ck, 0.0, 1.0, **kw)["costs"]
    b = amp.state_scheme_costs(p1, ck1, c_it, 1.0, **kw)["costs"]
    t_k = _bbht_by_stages(p) * (ck + fu["c_filt"]) + 1.0
    t_k1 = c_it + _bbht_by_stages(p1) * (ck1 + fu["c_filt"]) + 1.0
    assert (t_k, t_k1) == (pytest.approx(1.651979, abs=1e-6), pytest.approx(1.619557, abs=1e-6))
    ver = amp.iteration_verdict(a, b)
    assert (ver["best_k"], ver["best_k1"]) == ("lt_bbht", "lt_bbht") and ver["pays_best"]
    assert ver["saving_best"] == pytest.approx(t_k - t_k1, rel=1e-10) == pytest.approx(0.032422, abs=1e-6)
    r_k, r_k1 = (ck + fu["c_filt"]) / p + 1.0, c_it + (ck1 + fu["c_filt"]) / p1 + 1.0
    assert (r_k, r_k1) == (pytest.approx(1.563524, abs=1e-6), pytest.approx(1.598354, abs=1e-6))
    ext = amp.extended_verdict(a, b)
    assert (ext["best_k"], ext["best_k1"]) == ("lt_rep", "lt_rep") and not ext["pays_best"]
    assert ext["saving_best"] == pytest.approx(r_k - r_k1, rel=1e-10) == pytest.approx(-0.034830, abs=1e-6)
    assert ext["relative_best"] == pytest.approx((r_k - r_k1) / r_k, rel=1e-10)
    assert amp.extended_verdict({"aa": 1.0}, b)["best_k"] is None


def _record(p, e) -> TrajectoryRecord:
    k = len(p) - 1
    return TrajectoryRecord(system="t", optimizer="lbfgs", norb=4, nelec=(2, 2), e0=-1.0, e1=-0.4, e1_any=-0.5,
                            energies=np.asarray(e, float), variances=np.full(k + 1, 1e-3),
                            residual_norms=np.zeros(k + 1), overlaps=np.asarray(p, float), eps=np.zeros(k),
                            directions=[np.zeros((4,) * 4)] * k, states=[np.zeros((6, 6))] * (k + 1), irreps=None)


def test_rep_column_equals_m3():
    """scheme_table's "rep" (n_v = 1) equals HandoffRow.cost_expected on every row and loop; the per-row inputs,
    the oracle-U keys and their flag; fp takes the rows' Eckart-Temple p_lb."""
    rng = np.random.default_rng(9)
    c_qpe = 2e7
    for _ in range(5):
        p = np.sort(rng.uniform(0.05, 1.0, 7))
        rec = _record(p, -1.0 + 0.3 * (1 - p))
        prep = np.concatenate([[0.0], np.cumsum(rng.uniform(1e2, 1e5, 6))])
        iters = rng.uniform(0.0, 3e6, 6)
        for loop in handoff.LOOPS:
            rows = handoff.handoff_table(rec, prep, iters, c_qpe, loop=loop)
            table = amp.scheme_table(rows, c_qpe, c_filt_oracle=0.1 * c_qpe, c_coarse_oracle=0.2 * c_qpe)
            for r, t in zip(rows, table):
                assert t.costs["rep"] == pytest.approx(r.cost_expected, rel=1e-14)
                assert t.k == r.k and t.p == r.p and t.c_loop == r.c_loop and t.c_prep == r.c_prep
                assert "oracle_U" in t.flags and t.costs["lt@oracle_U"] is not None and t.costs["lt"] is None
                assert t.best_realizable == amp.best_realizable(t.costs)
                if r.p_lb > 0:
                    assert t.costs["fp"] == pytest.approx(
                        r.c_loop + amp.fixed_point_calls(min(r.p_lb, 1.0), 0.01) * (r.c_prep + c_qpe))
    rows = [{"k": 0, "p": 0.25, "c_prep": 0.0, "c_loop": 0.0, "flags": ("termination_not_certified",)},
            {"k": 1, "p": 0.4, "c_prep": 10.0, "c_loop": 5.0}]
    table = amp.scheme_table(rows, 100.0, c_filt=[20.0, None], c_filt_l=12.0, n_verify=[2, 1], flags=("x",))
    assert table[0].costs["lt_bbht"] is not None and table[1].costs["lt_bbht"] is None
    assert table[0].costs["rep"] == pytest.approx(100 / 0.25 + 100) and table[1].costs["rep"] == pytest.approx(5 + 110 / 0.4)
    assert table[0].flags == ("termination_not_certified", "x") and table[1].flags == ("x",)
    with pytest.raises(ValueError):
        amp.scheme_table(rows, 100.0, c_filt=[1.0])
    with pytest.raises(ValueError):
        amp.SchemeRow(k=0, p=0.5, c_prep=0.0, c_loop=0.0, costs={"rep": 1.0}, best_realizable="aa")


# ------------------------------------------------------------------ savings and break-even (F44)
def test_saving_and_break_even():
    """saving_budget: C_QPE(1/p0 - 1), C_QPE(N*(p0) - 1), (N*(p0) - 1) C_filt, (E_BBHT(p0) - 1) C_filt, C_c(1/p0 - 1);
    break_even_overlap solves H_s(p0, 0) - H_s(p1, c1) = C_it, falls with C_it and approaches C_QPE/C_it (rep),
    (1.38 C_QPE/C_it)^2 (aa), (1.38 C_filt/C_it)^2 (lt), (kappa_B C_filt/C_it)^2 (lt_bbht_L) for C_it >> C_QPE;
    scheme_saving is H_s(p_k, c_k) - H_s(p_k1, c_k1)."""
    cf, cc = 0.125, 0.24
    p0 = 0.112
    assert amp.saving_budget("rep", p0, c_qpe=1.0) == pytest.approx(1 / p0 - 1)
    assert amp.saving_budget("aa", p0, c_qpe=1.0) == pytest.approx(amp.grover_calls(p0)[0] - 1)
    assert amp.saving_budget("lt", p0, c_qpe=1.0, c_filt=cf) == pytest.approx((amp.grover_calls(p0)[0] - 1) * cf)
    assert amp.saving_budget("lt_bbht_L", p0, c_qpe=1.0, c_filt=cf) == pytest.approx(
        (amp.exponential_search_calls(p0) - 1) * cf)
    assert amp.saving_budget("2s_L", p0, c_qpe=1.0, c_coarse=cc) == pytest.approx(cc * (1 / p0 - 1))
    assert amp.saving_budget("bbht", 1.0, c_qpe=1.0) == 0.0
    p1, c1 = 0.6, 0.002
    for scheme, kw in (("rep", {}), ("aa", {}), ("bbht", {}), ("lt", {"c_filt": cf}), ("lt_bbht_L", {"c_filt": cf}),
                       ("2s", {"c_coarse": cc}), ("fp", {})):
        prev = 1.0
        for c_it in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
            p_star = amp.break_even_overlap(scheme, c_it, p1, c_qpe=1.0, c_prep1=c1, **kw)
            assert 0 < p_star <= prev
            mu = amp.scheme_saving(scheme, c_qpe=1.0, **kw)(p_star, p1, 0.0, c1)
            assert mu == pytest.approx(c_it, rel=1e-6, abs=1e-9) or mu >= c_it
            assert amp.scheme_saving(scheme, c_qpe=1.0, **kw)(p_star * 1.01, p1, 0.0, c1) < c_it or scheme in ("aa", "fp")
            prev = p_star
    big = 1e4
    assert amp.break_even_overlap("rep", big, p1, c_qpe=1.0) == pytest.approx(1 / big, rel=1e-3)
    assert amp.break_even_overlap("aa", big, p1, c_qpe=1.0) == pytest.approx((1.38005 / big) ** 2, rel=2e-3)
    assert amp.break_even_overlap("lt", big * cf, p1, c_qpe=1.0, c_filt=cf) == pytest.approx((1.38005 / big) ** 2,
                                                                                            rel=2e-3)
    p_l = amp.break_even_overlap("lt_bbht_L", big * cf, p1, c_qpe=1.0, c_filt=cf)
    kappa = amp.exponential_search_calls(p_l) * math.sqrt(p_l)
    assert 1.8 <= kappa <= 2.9 and p_l == pytest.approx((kappa / big) ** 2, rel=2e-3)
    with pytest.raises(ValueError):
        amp.saving_budget("lt", 0.3, c_qpe=1.0)
    with pytest.raises(ValueError):
        amp.break_even_overlap("rep", 0.0, 0.5, c_qpe=1.0)
    with pytest.raises(ValueError):
        amp.scheme_saving("nope")


def test_scheme_saving_accepts_arrays():
    """scheme_saving on arrays (broadcast) equals the scalar calls elementwise, repeated calls give the same values
    (the kept values of H_s), invalid entries raise as for scalars; the ft.scaling hook uses the array path."""
    from thc_engine_cqe.ft import scaling

    pk = np.geomspace(1e-20, 0.9, 41)
    pk1 = np.minimum(1.0, 1.7 * pk)
    for scheme, kw in (("rep", {}), ("aa", {}), ("bbht", {}), ("lt_bbht_L", {"c_filt": 0.13}),
                       ("2s", {"c_coarse": 0.4})):
        sv = amp.scheme_saving(scheme, c_qpe=1.0, **kw)
        arr = sv(pk, pk1, 0.0, 0.01)
        ref = np.array([amp.scheme_saving(scheme, c_qpe=1.0, **kw)(a, b, 0.0, 0.01) for a, b in zip(pk, pk1)])
        assert arr.shape == pk.shape and np.array_equal(arr, ref)
        assert np.array_equal(sv(pk, pk1, 0.0, 0.01), arr)
        assert sv(float(pk[3]), float(pk1[3]), 0.0, 0.01) == arr[3]
        assert np.array_equal(scaling._saving_values(sv, pk, pk1), sv(pk, pk1, 0.0, 0.0))
    with pytest.raises(ValueError):
        amp.scheme_saving("rep")(np.array([0.5, 0.0]), np.array([0.6, 0.7]))
    with pytest.raises(ValueError):
        amp.scheme_saving("rep")(0.5, 0.6, -1.0, 0.0)


def test_best_realizable_saving():
    """mu* = min_s H_s(p_k, c_k) - min_s H_s(p_k1, c_k1) over the realizable schemes with a cost, each with F40's
    verification runs (rep, bbht), checked against scheme_costs and best_realizable by hand; one scheme alone is
    scheme_saving (n_v = 1); the minimizing scheme may switch between the iterates; c_iter < mu* iff the best
    realizable T falls (iteration_verdict); fp never has a cost; inputs validated."""
    kw = dict(c_filt_l=0.13, c_coarse_l=0.23, n_verify=2)
    sv = amp.best_realizable_saving(c_qpe=1.0, **kw)
    rng = np.random.default_rng(17)
    switched = 0
    for _ in range(200):
        pk = float(10 ** rng.uniform(-4, -0.05))
        pk1 = min(1.0, pk * float(rng.uniform(1.0, 4.0)))
        ck, ck1 = float(rng.uniform(0, 0.01)), float(rng.uniform(0, 0.02))
        a = amp.scheme_costs(pk, ck, 0.0, 1.0, **kw)
        b = amp.scheme_costs(pk1, ck1, 0.0, 1.0, **kw)
        sa, sb = amp.best_realizable(a), amp.best_realizable(b)
        switched += sa != sb
        assert sv(pk, pk1, ck, ck1) == pytest.approx(a[sa] - b[sb], rel=1e-14, abs=1e-14)
        c_it = float(rng.uniform(0.0, 2.0))
        ver = amp.iteration_verdict(a, {s: None if v is None else v + c_it for s, v in b.items()})
        assert ver["best_k"] == sa and ver["best_k1"] == sb
        assert ver["pays_best"] == (c_it < sv(pk, pk1, ck, ck1))
    assert switched > 0                                     # 2s_L wins near p = 0.6-0.7, lt_bbht_L elsewhere
    for scheme in ("rep", "bbht"):
        one = amp.best_realizable_saving(schemes=(scheme,), c_qpe=1.0)
        assert one(0.1, 0.3, 0.0, 0.001) == pytest.approx(amp.scheme_saving(scheme, c_qpe=1.0)(0.1, 0.3, 0.0, 0.001),
                                                          rel=1e-14)
    pk = np.geomspace(1e-12, 0.5, 9)
    assert np.allclose(sv(pk, 2 * pk), [sv(float(a), float(2 * a)) for a in pk], rtol=1e-14, atol=0.0)
    for bad in (dict(schemes=()), dict(schemes=("aa",)), dict(schemes=("fp",)), dict(schemes=("lt_bbht",)),
                dict(n_verify=0)):
        with pytest.raises(ValueError):
            amp.best_realizable_saving(c_qpe=1.0, **bad)


def test_saving_hooks_mark_qpe_scaling():
    """The hooks say whether H_s scales with C_QPE at fixed C(Psi): rep, aa, bbht and fp do, the filtered schemes
    (C_QPE once) do not; best_realizable_saving marks a set of one kind only (lt_bbht_L without c_filt_l has no cost).
    The F45 scan of ft.scaling reads the mark: beyond the window lt_bbht_L's n* is its n* without one (C_QPE cancels
    in mu), while lengthened it would come earlier; bbht's n* is that of the lengthened QPE."""
    from thc_engine_cqe.ft import scaling

    assert all(amp.scheme_saving(s).qpe_scaled is True for s in ("rep", "aa", "bbht", "fp"))
    assert all(amp.scheme_saving(s, c_filt=0.1, c_coarse=0.2).qpe_scaled is False
               for s in ("lt", "lt_bbht", "lt_bbht_L", "2s", "2s_L"))
    assert amp.best_realizable_saving(schemes=("rep", "bbht"), n_verify=2).qpe_scaled is True
    assert amp.best_realizable_saving(schemes=("rep", "lt_bbht_L")).qpe_scaled is True
    assert amp.best_realizable_saving(schemes=("lt_bbht_L", "2s_L"), c_filt_l=0.1, c_coarse_l=0.2).qpe_scaled is False
    assert not hasattr(amp.best_realizable_saving(schemes=("rep", "lt_bbht_L"), c_filt_l=0.1), "qpe_scaled")
    ns = np.array([4, 6, 8, 10])
    fk = scaling.fit_log_overlap(ns, np.exp(-0.09 - 0.0591 * ns))
    rates = [0.054 - 0.01 * 0.5 ** (n / 2) for n in ns[1:]]
    fk1 = scaling.fit_log_overlap(ns, np.exp(np.concatenate([[-0.05], -0.05 - 2.0 * np.cumsum(rates)])))
    args, prep = (fk, fk1, 166.0, 10, 9.4), ((1e-3, 10, 2.2), (1e-2, 10, 2.3))
    win = dict(gap=scaling.fit_gap([8, 10], [0.017, 0.0141]), margin=3e-3)
    lt = amp.scheme_saving("lt_bbht_L", c_filt=0.125)
    plain = scaling.break_even_size(*args, saving=lt, prep=prep)
    assert plain is not None and scaling.break_even_size(*args, saving=lt, prep=prep, **win) == plain
    assert scaling.break_even_size(*args, saving=lt, prep=prep, qpe_scaled=True, **win) < plain
    bbht = amp.scheme_saving("bbht")
    lengthened = scaling.break_even_size(*args, saving=bbht, prep=prep, qpe_scaled=True, **win)
    assert scaling.break_even_size(*args, saving=bbht, prep=prep, **win) == lengthened
    assert lengthened != scaling.break_even_size(*args, saving=bbht, prep=prep)


def test_iteration_verdict():
    """F44's rule: pays under s iff T_s(k + 1) < T_s(k), for every key with a cost at both iterates (oracle keys
    reported, never in the verdict of record); the verdict of record compares the best realizable T before and after
    the iteration, with s* re-chosen at each iterate; agreement with cost_ratio < 1 of the same scheme; inf and None."""
    from thc_engine_cqe.ft import scaling

    pk, pk1, ck, ck1, c_it = 0.11, 0.21, 0.0, 1.5e-4, 0.42
    kw = dict(c_filt_l=0.133, c_coarse_l=0.219, n_verify=2)
    a = amp.scheme_costs(pk, ck, 0.0, 1.0, **kw)
    b = amp.scheme_costs(pk1, ck1, c_it, 1.0, **kw)
    ver = amp.iteration_verdict(a, b)
    for s in ("rep", "aa", "bbht", "lt_bbht_L", "2s_L"):
        v = ver["by_scheme"][s]
        assert v["saving"] == pytest.approx(a[s] - b[s], rel=1e-14)
        assert v["relative"] == pytest.approx(v["saving"] / a[s])
        hook = amp.scheme_saving(s, c_qpe=1.0, c_filt=kw["c_filt_l"] if s == "lt_bbht_L" else None,
                                 c_coarse=kw["c_coarse_l"] if s == "2s_L" else None)
        rho = scaling.cost_ratio(c_it, pk, pk1, ck, ck1, saving=hook)
        assert v["pays"] == (rho < 1.0)
    assert "fp" not in ver["by_scheme"] and "lt" not in ver["by_scheme"]
    assert ver["best_k"] == amp.best_realizable(a) and ver["best_k1"] == amp.best_realizable(b)
    assert ver["saving_best"] == pytest.approx(a[ver["best_k"]] - b[ver["best_k1"]], rel=1e-14)
    assert ver["pays_best"] == (ver["saving_best"] > 0)
    # the best realizable scheme switches: rep is cheapest before (p = 0.95), 2s_L after
    before = {"rep": 1.05, "bbht": 1.2, "2s_L": 1.3, "aa": 0.5, "lt@oracle_U": 0.1}
    after = {"rep": 1.4, "bbht": 1.25, "2s_L": 1.0, "aa": 0.6, "lt@oracle_U": 0.05}
    ver = amp.iteration_verdict(before, after)
    assert (ver["best_k"], ver["best_k1"]) == ("rep", "2s_L") and ver["pays_best"]
    assert ver["saving_best"] == pytest.approx(0.05) and not ver["by_scheme"]["rep"]["pays"]
    assert ver["by_scheme"]["lt@oracle_U"]["pays"] and ver["by_scheme"]["aa"]["saving"] == pytest.approx(-0.1)
    ver = amp.iteration_verdict({"rep": math.inf, "bbht": None}, {"rep": 3.0, "bbht": 2.0})
    assert ver["by_scheme"]["rep"]["saving"] == math.inf and ver["by_scheme"]["rep"]["pays"]
    assert "bbht" not in ver["by_scheme"] and ver["best_k"] == "rep" and ver["best_k1"] == "bbht" and ver["pays_best"]
    ver = amp.iteration_verdict({"aa": 1.0}, {"rep": 1.0})
    assert ver["by_scheme"] == {} and ver["best_k"] is None and ver["pays_best"] is False
    ver = amp.iteration_verdict({"rep": math.inf}, {"rep": math.inf})
    assert ver["by_scheme"] == {} and ver["saving_best"] is None and not ver["pays_best"]


# ------------------------------------------------------------------ the scheme table from the model
@pytest.mark.slow
def test_scheme_table_from_the_model():
    """n2_r2.0 and n2_r2.4, HF and the classical spin-orbital Psi_1, p_0 on H': the scheme table's repetition and AA
    columns to 1e-3 (F40's n_v from acceptance_bracket: B_U with one run on n2_r2.0, B_L with two on n2_r2.4); the
    filtered columns from filter_cost and coarse_qpe_toffolis: the walk part 2d C_W of C_filt equals the closed form
    (4 c_f/pi) sin(theta_bar) dE/(E1lb - U) behind the table to 0.5%, and C_filt adds F43's GQSP rotations
    3(2d + 1)(beta - 2), 5.1-5.2% of C_filt here, which the table omits (lt, lt_bbht +2 to +5.6%, lt_bbht_L +1.7 to
    +2.4% [build check]); the coarse lengths are 0.92 of the closed form (2s, 2s_L -3 to -8%); lt_bbht_L the best
    realizable scheme on every row."""
    for name, key in (("n2s", "n2_r2.0"), ("n2_r2.4", "n2_r2.4")):
        s = _n2(name)
        spec, q, enc = s["spec"], s["qpe"], s["qpe"].encoding
        br = acc.acceptance_bracket(s["u"], s["e1lb"], DE, grid=spec.grid(acc.qpe_points(q)))
        assert (br.form, br.n_runs) == (("B_U", 1) if name == "n2s" else ("B_L", 2))
        step = second_order_step(s["active"], metric="spin_orbital", irreps=s["irreps"])
        c_q = float(q.toffolis)
        cf = amp.filter_cost(enc, spec.c, s["u"], s["e1lb"]).toffolis / c_q
        cc = amp.coarse_qpe_toffolis(enc, spec.c, s["u"], s["e1lb"]) / c_q
        cfl = amp.filter_cost(enc, spec.c, s["e1lb"] - amp.W_L, s["e1lb"]).toffolis / c_q
        ccl = amp.coarse_qpe_toffolis(enc, spec.c, s["e1lb"] - amp.W_L, s["e1lb"]) / c_q
        hf = np.zeros(spec.shape)
        hf[0, 0] = 1.0
        for state, ci in (("HF", hf), ("Psi1", step["state"])):
            p = float(spec.weights(ci)[0])
            c = amp.scheme_costs(p, 0.0, 0.0, 1.0, c_filt=cf, c_coarse=cc, c_filt_l=cfl, c_coarse_l=ccl,
                                 n_verify=br.n_runs)
            ref = dict(zip(_SCHEME_KEYS, _SCHEME_TABLE[(key, state)]))
            for k in ("rep", "aa", "bbht"):
                assert c[k] == pytest.approx(ref[k], rel=1e-3)
            for k in ("lt", "lt_bbht", "lt_bbht_L"):
                assert 0.0 < c[k] / ref[k] - 1.0 < 0.06
            for k in ("2s", "2s_L"):
                assert -0.1 < c[k] / ref[k] - 1.0 < 0.0
            assert amp.best_realizable(c) == "lt_bbht_L"
        for u in (s["u"], s["e1lb"] - amp.W_L):
            f = amp.filter_cost(enc, spec.c, u, s["e1lb"])
            tp, ts = math.acos((u - spec.c) / spec.lam), math.acos((s["e1lb"] - spec.c) / spec.lam)
            closed = 4 * 9.999 / math.pi * math.sin(0.5 * (tp + ts)) * DE / (s["e1lb"] - u)
            assert 2 * f.degree * enc.toffoli_per_step / c_q == pytest.approx(closed, rel=5e-3)
            assert 0.05 < 3 * (2 * f.degree + 1) * 14 / f.toffolis < 0.053
