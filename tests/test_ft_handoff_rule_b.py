"""Tests of the rule-B additions to ft.handoff (F40, F42): rule_b_table, optimal_rule_b, runtime_quantile.

References: M3's ``handoff_table`` rows (the oracle T_E), the cost formula of F40 evaluated per option
with ``ft.acceptance.acceptance_bracket`` and a brute-force minimum over the options, M3's rule-E
certification (``rule_e_certifiable``), the defining property of R_delta, and M3's T6 numbers under
rule B for h6s and n2s [design] with the selected encodings (h6s THC M = 12, n2s SF L = 17), their
model spectra, CISD(H') and the elementwise state preparation. The tests of M3's hand-off
(tests/test_ft_handoff.py) are unchanged.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from thc_engine_cqe.diagnostics.thc_floors import ExactReference
from thc_engine_cqe.ft import acceptance as acc
from thc_engine_cqe.ft import certify, handoff, stateprep, symmetry
from thc_engine_cqe.ft import qpe as ftqpe
from thc_engine_cqe.ft.trajectory import TrajectoryRecord, exact_trajectory

C_QPE = 2.0e7
DE = 1e-3


def _record(p, e, var, e0=-1.0, e1=-0.4, e1_any=-0.5) -> TrajectoryRecord:
    k = len(p) - 1
    return TrajectoryRecord(system="test", optimizer="lbfgs", norb=4, nelec=(2, 2), e0=e0, e1=e1, e1_any=e1_any,
                            energies=np.asarray(e, float), variances=np.asarray(var, float),
                            residual_norms=np.zeros(k + 1), overlaps=np.asarray(p, float), eps=np.zeros(k),
                            directions=[np.zeros((4,) * 4)] * k, states=[np.zeros((6, 6))] * (k + 1), irreps=None)


def _random_case(rng, k: int = 8):
    p = np.sort(rng.uniform(0.05, 1.0, k + 1))
    e = -1.0 + (1.0 - p) * 0.7
    prep = np.concatenate([[0.0], np.cumsum(rng.uniform(1e2, 1e5, k))])
    iters = rng.uniform(1e5, 3e7, k)
    return _record(p, e, (1.0 - p) * 0.05), prep, iters


def _certificates(rec, rng, *, free: bool = True) -> list[certify.EnergyCertificate]:
    """A free classical U, a costly sampled U of a late iterate, a loop energy and an oracle one (free=False: the
    sampled and the oracle one only)."""
    e1lb = rec.e1 - 1e-4
    classical = certify.EnergyCertificate("classical", float(rng.uniform(rec.e0, e1lb - 0.05)), None, 0.0, 0.0, 0.0,
                                          "cisd")
    sampled = certify.EnergyCertificate("sampled", float(rng.uniform(rec.e0, e1lb - 0.002)), int(rng.integers(1, 6)),
                                        float(rng.uniform(1e5, 3e7)), 1e3, 0.0, "sampled")
    loop = certify.EnergyCertificate("loop_energy", float(rec.e0 + 1e-3), 2, 0.0, 0.0, 0.0, "loop")
    oracle = certify.EnergyCertificate("oracle", float(rec.e0), None, 0.0, 0.0, 0.0, "E0")
    return [classical, sampled, loop, oracle] if free else [sampled, oracle]


def _brute_options(certs, e1lb: float, grid) -> list:
    """Every (certificate or None, n_v, theta, form) option of F40 by hand: B_L without U; per certificate the
    bracket's form and, with the kernels (grid given), B_U at U + dE with verification_runs(D_U) runs."""
    opts = [(None, acc.acceptance_bracket(None, e1lb, DE, grid=grid))]
    for c in certs:
        if c.source == "oracle":
            continue
        br = acc.acceptance_bracket(c.upper, e1lb, DE, grid=grid)
        opts.append((c, br))
        if grid is not None and c.upper + DE < e1lb:
            try:
                n_u = acc.verification_runs((e1lb - c.upper - DE) / grid)
            except ValueError:
                continue
            opts.append((c, acc.Bracket(theta=c.upper + DE, form="B_U", lower=c.upper + DE, upper=e1lb - 5 * grid,
                                        certified=c.upper <= e1lb - 2 * DE, distance=(e1lb - c.upper - DE) / grid,
                                        n_runs=n_u)))
    return opts


@pytest.mark.parametrize("grid", [None, 1.97e-3, 1e-4])
def test_rule_b_is_oracle_plus_certificate(grid):
    """T_B = M3's oracle T_E + C_U + (n_v - 1) C_QPE (+ C_loop(j_U) - C_loop(k) for a certificate from a later
    iterate), and the row's option is the cheapest of every (certificate or none) option that terminates, B_U at
    U + dE with its own n_v included; oracle certificates are never used. Without the free certificates the costly
    sampled one is taken on some rows (its C_U and the loop to its iterate enter)."""
    rng = np.random.default_rng(0)
    costly = 0
    for free in (True, False):
        for _ in range(6):
            rec, prep, iters = _random_case(rng)
            certs = _certificates(rec, rng, free=free)
            e1lb = rec.e1 - 1e-4
            opts = _brute_options(certs, e1lb, grid)
            for loop in handoff.LOOPS:
                m3 = handoff.handoff_table(rec, prep, iters, C_QPE, loop=loop)
                rows = handoff.rule_b_table(rec, prep, iters, C_QPE, certificates=certs, loop=loop, e1_lower=e1lb,
                                            grid=grid)
                assert [r.k for r in rows] == [r.k for r in m3]
                c_loop = [r.c_loop for r in m3]
                for rb, r in zip(rows, m3):
                    assert rb.cost_oracle_e == r.cost_expected and rb.p == r.p and rb.c_loop == r.c_loop
                    cert = rb.certificate
                    j_u = 0 if cert is None else cert.loop_iterate
                    c_u = 0.0 if cert is None else cert.toffolis
                    expect = r.cost_expected + (c_loop[max(r.k, j_u)] - c_loop[r.k]) + c_u + (rb.n_runs - 1) * C_QPE
                    assert rb.cost == pytest.approx(expect, rel=1e-13)
                    assert rb.n_runs_expected == rb.n_runs and rb.certificate_cost == c_u
                    assert cert is None or cert.source != "oracle"
                    costly += cert is not None and cert.toffolis > 0.0
                    options = []
                    for c, br in opts:
                        j = 0 if c is None else c.loop_iterate
                        if j > m3[-1].k or not (br.form == "B_U" or rec.e0 + DE <= br.theta):
                            continue
                        options.append(c_loop[max(r.k, j)] + (0.0 if c is None else c.toffolis)
                                       + (r.c_prep + C_QPE) / r.p + (br.n_runs - 1) * C_QPE)
                    assert rb.cost == pytest.approx(min(options), rel=1e-13)
                best = handoff.optimal_rule_b(rows)
                assert best.cost == min(r.cost for r in rows)
                assert (handoff.optimal_rule_b(rows, "bracketed_oracle").k
                        == handoff.optimal_handoff(m3, "expected_oracle").k)
    assert costly > 0 if grid is not None else costly == 0       # window model: B_L needs no more runs than B_U


def _rec(p, e, e0=-1.0, e1=-0.993) -> TrajectoryRecord:
    return _record(p, e, np.full(len(p), 1e-4), e0=e0, e1=e1, e1_any=e1)


def test_b_u_with_more_runs_where_b_l_does_not_terminate():
    """Gap 7 mHa: B_L's theta = E1lb - 5g lies 2.95 mHa below E_0' (3.95 mHa below the termination edge
    E_0' + dE), so B_L does not terminate; the certified U = E_0 + 1 mHa gives D_U = (6.9 - 1 - 1)/1.97 = 2.487 and
    three runs (acceptance_bracket picks B_L, the form with fewer runs). The row takes B_U with three runs:
    T_B(0) = C_QPE/0.25 + 2 C_QPE = 6 C_QPE, certified, without "not_terminating" or "no_certificate";
    rule_b_bracket gives the same form. Without U the row costs inf."""
    g, c_qpe = 1.97e-3, 1e7
    r = _rec([0.25, 0.4], [-0.98, -0.99])
    e1lb = r.e1 - 1e-4
    assert acc.verification_runs((e1lb - (r.e0 + 1e-3) - DE) / g) == 3
    assert acc.acceptance_bracket(r.e0 + 1e-3, e1lb, DE, grid=g).form == "B_L"
    cert = certify.EnergyCertificate("classical", r.e0 + 1e-3, None, 0.0, 0.0, 0.0, "cisd")
    rows = handoff.rule_b_table(r, [0.0, 1e3], None, c_qpe, certificates=[cert], loop="free", e1_lower=e1lb, grid=g,
                                e0_model=r.e0)
    row = rows[0]
    assert (row.form, row.n_runs, row.certificate) == ("B_U", 3, cert) and row.certified
    assert row.cost == pytest.approx(c_qpe / 0.25 + 2 * c_qpe, rel=1e-13)
    assert row.theta == pytest.approx(r.e0 + 2e-3, abs=1e-15)
    assert "not_terminating" not in row.flags and "no_certificate" not in row.flags
    br = handoff.rule_b_bracket(r.e0 + 1e-3, e1lb, DE, grid=g, e0_model=r.e0)
    assert (br.form, br.n_runs, br.certified) == ("B_U", 3, True) and br.theta == row.theta
    none = handoff.rule_b_table(r, [0.0, 1e3], None, c_qpe, certificates=[], loop="free", e1_lower=e1lb, grid=g,
                                e0_model=r.e0)
    assert math.isinf(none[0].cost) and "not_terminating" in none[0].flags
    assert handoff.rule_b_bracket(None, e1lb, DE, grid=g, e0_model=r.e0) is None
    # where B_L terminates it keeps its two runs, against B_U's three
    br = handoff.rule_b_bracket(r.e0 + 1e-3, e1lb, DE, grid=g, e0_model=r.e0 - 0.01)
    assert (br.form, br.n_runs) == ("B_L", 2)
    rows = handoff.rule_b_table(r, [0.0, 1e3], None, c_qpe, certificates=[cert], loop="free", e1_lower=e1lb, grid=g,
                                e0_model=r.e0 - 0.01)
    assert (rows[0].form, rows[0].n_runs) == ("B_L", 2) and rows[0].cost == pytest.approx(5e7, rel=1e-13)


def test_b_l_termination_edge():
    """B_L terminates iff E_0' + dE <= theta_L = E1lb - 5g: E_0' = theta_L - 0.5 mHa costs inf ("not_terminating"),
    theta_L - 1.5 mHa costs C_QPE/0.3 + C_QPE (two runs)."""
    g = 1.97e-3
    r = _rec([0.3, 0.5], [-0.99, -0.995])
    e1lb = -0.98
    theta_l = e1lb - 5 * g
    rows = handoff.rule_b_table(r, [0.0, 1e3], None, C_QPE, certificates=[], loop="free", e1_lower=e1lb, grid=g,
                                e0_model=theta_l - 0.5e-3)
    assert math.isinf(rows[0].cost) and "not_terminating" in rows[0].flags
    rows = handoff.rule_b_table(r, [0.0, 1e3], None, C_QPE, certificates=[], loop="free", e1_lower=e1lb, grid=g,
                                e0_model=theta_l - 1.5e-3)
    assert rows[0].cost == pytest.approx(C_QPE / 0.3 + C_QPE, rel=1e-13) and rows[0].theta == theta_l


def test_certificate_beyond_the_rows_is_ignored():
    """A certificate of iterate 5 on a table whose rows end at k = 1 (classical first step without iteration costs)
    leaves every row's cost and certificate as without it (its loop C_loop(5) is not covered)."""
    g = 1.97e-3
    r = _rec([0.3, 0.5, 0.7], [-0.99, -0.995, -0.998], e1=-0.9)
    late = certify.EnergyCertificate("sampled", -0.997, 5, 1e5, 10.0, 0.0, "late")
    kw = dict(certificates=[], loop="classical_first_step", e1_lower=r.e1 - 1e-4, grid=g)
    base = handoff.rule_b_table(r, [0.0, 1e3, 2e3], None, C_QPE, **kw)
    rows = handoff.rule_b_table(r, [0.0, 1e3, 2e3], None, C_QPE, **(kw | {"certificates": [late]}))
    assert [x.k for x in rows] == [0, 1]
    assert [x.cost for x in rows] == [x.cost for x in base] and all(x.certificate is None for x in rows)


def test_e1lb_fallback_value():
    """Without e1_lower (and grid None) E1lb = E_1(H) - 1 mHa: B_L at E1lb - dE = -0.95 - 2 mHa, flag
    "e1lb_from_h"."""
    r = _rec([0.3, 0.5], [-0.99, -0.995], e1=-0.95)
    rows = handoff.rule_b_table(r, [0.0, 1e3], None, C_QPE, certificates=[], loop="free", grid=None)
    assert rows[0].theta == pytest.approx(-0.952, abs=1e-15) and "e1lb_from_h" in rows[0].flags


def test_certificate_of_the_last_row_iterate_is_used():
    """A sampled certificate of Psi_1 (C_U = 0.2 C_QPE, D_U = 45.1: one run) with the classical-first-step loop (rows
    k <= 1) beats B_L's two runs on both rows: T_B(k) = (C(Psi_k) + C_QPE)/p_k + 0.2 C_QPE, i.e. 3.5333e7 and
    2.2002e7 (j_U = k_max = 1 is covered; B_L's second run, C_QPE = 1e7, would cost 8e6 more than C_U)."""
    g, c_qpe = 1.97e-3, 1e7
    r = _rec([0.3, 0.5], [-0.95, -0.97], e1=-0.9)
    e1lb = r.e1 - 1e-4
    cert = certify.EnergyCertificate("sampled", r.e0 + 0.01, 1, 0.2 * c_qpe, 100.0, 0.0, "sampled E(Psi_1)")
    assert acc.verification_runs((e1lb - cert.upper - DE) / g) == 1
    rows = handoff.rule_b_table(r, [0.0, 1e3], None, c_qpe, certificates=[cert], loop="classical_first_step",
                                e1_lower=e1lb, grid=g, e0_model=r.e0)
    assert [x.k for x in rows] == [0, 1]
    for x, ref in zip(rows, (1e7 / 0.3 + 2e6, (1e3 + 1e7) / 0.5 + 2e6)):
        assert x.certificate is cert and (x.form, x.n_runs) == ("B_U", 1)
        assert x.cost == pytest.approx(ref, rel=1e-13)


def _phi(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _t1() -> float:
    """min{D : verification_runs(D) = 1} by bisection of verification_runs."""
    lo, hi = 26.02, 26.05
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if acc.verification_runs(mid) == 1:
            hi = mid
        else:
            lo = mid
    return hi


def test_dedicated_certificates_cost_their_expected_runs():
    """A dedicated energy is a random bound U^ ~ N(U, sigma^2) (F42): its B_U option costs C_U + C_QPE P(n_v(U^) >= 2)
    with B_L's two runs as the fallback, P(D(U^) < t_1) = Phi((t_1 - D(U)) g / sigma).

    Synthetic, sigma = 5 mHa at D(U) = t_1: C_U + 0.5 C_QPE. The cold inputs of n2_r2.8 exact_so Psi_3 (U, E1lb, g of
    the selected SF encoding, C_U = 0.564 C_QPE, sigma = (E1lb - 2 dE - U)/3 = 17.1 mHa, D(U) = 26.52): C_U + 0.478
    C_QPE > B_L's 1 C_QPE, so B_L is taken, T_B = T_E + C_QPE; n2_r2.4 exact_so Psi_3 (C_U = 0.306 C_QPE, D(U) =
    33.41): B_U stays at T_E + (0.306 + 0.25) C_QPE with the planned n_v = 1 and the expected 1.25. Where B_L does
    not terminate, inadmissible bounds repeat the certificate: C_U/P(admissible) + C_QPE (E[n_v | admissible] - 1)."""
    c_qpe = 1e7
    t1 = _t1()
    for name, e1lb, g, e0, u, c_u, form in (
            ("n2_r2.8", -107.19223289333748, 0.0019716553327535076, -107.43657402476404, -107.24552503147541,
             0.5639086680416813, "B_L"),
            ("n2_r2.4", -107.20321992235951, 0.001971344288983838, -107.43398323034428, -107.27008724301838,
             0.30575537077576376, "B_U"),
            ("synthetic", -107.0, 0.002, -107.3, -107.0 - DE - t1 * 0.002, 0.2, "B_U")):
        sigma = 5e-3 if name == "synthetic" else (e1lb - 2 * DE - u) / 3.0
        d_plan = (e1lb - u - DE) / g
        p_two = _phi((t1 - d_plan) * g / sigma)
        if name == "synthetic":
            assert p_two == pytest.approx(0.5, abs=1e-6)
        r = _rec([0.7, 0.75], [e0 + 0.01, e0 + 0.008], e0=e0, e1=e1lb + 1e-4)
        cert = certify.EnergyCertificate("sampled", u, 0, c_u * c_qpe, 1e4, 0.0, "sampled E(Psi_0)", sigma=sigma)
        rows = handoff.rule_b_table(r, [0.0, 1e3], None, c_qpe, certificates=[cert], loop="free", e1_lower=e1lb,
                                    grid=g, e0_model=e0)
        row = rows[0]
        assert row.form == form, name
        if form == "B_L":
            assert row.certificate is None and row.cost == pytest.approx(row.cost_oracle_e + c_qpe, rel=1e-13)
            assert c_u + p_two > 1.0
        else:
            assert row.certificate is cert and row.n_runs == 1
            assert row.n_runs_expected == pytest.approx(1.0 + p_two, rel=1e-6)
            assert row.cost == pytest.approx(row.cost_oracle_e + (c_u + p_two) * c_qpe, rel=1e-6)
            assert row.certificate_cost == pytest.approx(c_u * c_qpe, rel=1e-15)
        if name == "n2_r2.8":
            assert p_two == pytest.approx(0.478, abs=1e-3)
        if name == "n2_r2.4":
            assert p_two == pytest.approx(0.250, abs=1e-3)
    # B_L does not terminate (gap 7 mHa): the certificate is repeated until its bound is admissible
    g = 1.97e-3
    r = _rec([0.25, 0.4], [-0.98, -0.99])
    e1lb = r.e1 - 1e-4
    u, sigma = r.e0 + 0.4e-3, 1.5e-3                     # P(admissible) = Phi(3), D(U) = 2.79: three runs
    cert = certify.EnergyCertificate("sampled", u, 0, 0.1 * c_qpe, 1e4, 0.0, "sampled E(Psi_0)", sigma=sigma)
    ex = acc.expected_verification_runs(u, sigma, e1lb, DE, grid=g)
    assert ex.fallback is None and 0.99 < ex.admissible < 1.0
    rows = handoff.rule_b_table(r, [0.0, 1e3], None, c_qpe, certificates=[cert], loop="free", e1_lower=e1lb, grid=g,
                                e0_model=r.e0)
    assert rows[0].form == "B_U" and rows[0].certificate is cert
    assert rows[0].certificate_cost == pytest.approx(0.1 * c_qpe / ex.admissible, rel=1e-13)
    assert rows[0].cost == pytest.approx(c_qpe / 0.25 + 0.1 * c_qpe / ex.admissible + (ex.runs - 1) * c_qpe, rel=1e-13)
    # in the window model the bound's spread changes nothing (one run for every form)
    rows = handoff.rule_b_table(r, [0.0, 1e3], None, c_qpe, certificates=[cert], loop="free", e1_lower=e1lb,
                                grid=None, e0_model=r.e0 - 0.05)
    assert rows[0].n_runs_expected == 1.0


def test_rule_b_reduces_to_rule_e():
    """Window model with U = E_k + dE for row k: theta = E_k + 2 dE, n_v = 1, T_B = T_E, and B_U is certified iff
    E1 - E_k >= 3 dE (M3's rule_e_certifiable away from the edge); otherwise B_L at E1lb - dE, flagged."""
    rng = np.random.default_rng(4)
    n_cert = 0
    for _ in range(8):
        p = np.sort(rng.uniform(0.3, 1.0, 9))
        e = -1.0 + (1.0 - p) * 0.02
        rec = _record(p, e, (1 - p) * 0.01, e1=-0.99)
        prep = np.concatenate([[0.0], np.cumsum(rng.uniform(1e2, 1e5, 8))])
        for k in range(9):
            u = float(rec.energies[k]) + DE
            cert = certify.EnergyCertificate("classical", u, None, 0.0, 0.0, 0.0, "rule E")
            rows = handoff.rule_b_table(rec, prep, None, C_QPE, certificates=[cert], loop="free", e1_lower=rec.e1,
                                        grid=None)
            m3 = handoff.handoff_table(rec, prep, None, C_QPE, loop="free")
            rb, r = rows[k], m3[k]
            assert rb.cost == pytest.approx(r.cost_expected, rel=1e-15) and rb.n_runs == 1
            gap = rec.e1 - rec.energies[k]
            if abs(gap - 3 * DE) > 1e-12:
                assert rb.certified == r.certifiable == handoff.rule_e_certifiable(rec.energies[k], rec.e1, DE)
            if rb.certified:
                n_cert += 1
                assert rb.form == "B_U" and rb.theta == pytest.approx(rec.energies[k] + 2 * DE, abs=1e-15)
                assert rb.certificate is cert
            else:
                assert rb.form == "B_L" and rb.theta == pytest.approx(rec.e1 - DE, abs=1e-15)
                assert "termination_not_certified" in rb.flags and "window_model" in rb.flags
    assert n_cert > 10


def test_runtime_quantile():
    """R_delta(q) = runs_for_confidence(q, delta): (1 - q)^n <= delta < (1 - q)^(n - 1); a geometric runtime exceeds
    it with probability <= delta (Monte Carlo, 2e5 hand-offs)."""
    rng = np.random.default_rng(8)
    for q, delta in ((0.3, 0.01), (0.112, 0.05), (0.02, 1e-3), (0.9, 0.5), (1.0, 0.01)):
        n = handoff.runtime_quantile(q, delta)
        assert n == handoff.runs_for_confidence(q, delta)
        assert (1 - q) ** n <= delta + 1e-15
        if n > 1:
            assert (1 - q) ** (n - 1) > delta
        if q < 1.0:
            runs = rng.geometric(q, 200_000)
            assert np.mean(runs > n) <= delta + 3 * math.sqrt(delta * (1 - delta) / runs.size)
    with pytest.raises(ValueError):
        handoff.runtime_quantile(0.0, 0.01)


def test_rule_b_termination_flags_and_inputs():
    """B_L does not terminate when E_0' + dE > E1lb - D_L g (a small gap): the row costs inf ("not_terminating")
    and optimal_rule_b skips it; e1_lower=None takes E_1 - 1 mHa ("e1lb_from_h"); e0_model moves the check;
    stats attach their identification errors; M3's names are unchanged."""
    p = [0.3, 0.5, 0.7]
    rec = _record(p, [-0.99, -0.995, -0.998], [1e-3, 5e-4, 1e-4], e1=-0.992)       # gap 8 mHa
    prep = [0.0, 1e3, 2e3]
    g = 1.97e-3
    rows = handoff.rule_b_table(rec, prep, None, C_QPE, certificates=[], loop="free", e1_lower=rec.e1 - 1e-4, grid=g)
    assert all(math.isinf(r.cost) and "not_terminating" in r.flags and not r.certified for r in rows)
    with pytest.raises(ValueError):
        handoff.optimal_rule_b(rows)
    assert handoff.optimal_rule_b(rows, "bracketed_oracle").k == 2
    # a certified U rescues the rows: B_U terminates
    cert = certify.EnergyCertificate("classical", -0.9995, None, 0.0, 0.0, 0.0, "u")
    rows = handoff.rule_b_table(rec, prep, None, C_QPE, certificates=[cert], loop="free", e1_lower=rec.e1 - 1e-4,
                                grid=g)
    assert all(r.form == "B_U" and math.isfinite(r.cost) for r in rows)
    # a lower E_0' makes B_L terminate
    rows = handoff.rule_b_table(rec, prep, None, C_QPE, certificates=[], loop="free", e1_lower=rec.e1 - 1e-4, grid=g,
                                e0_model=-1.02)
    assert all(r.form == "B_L" and math.isfinite(r.cost) and "termination_not_certified" in r.flags for r in rows)
    rows = handoff.rule_b_table(rec, prep, None, C_QPE, certificates=[cert], loop="free")
    assert all("e1lb_from_h" in r.flags for r in rows)
    stats = [acc.AcceptanceStats(accept=0.3, ground=0.29, identification_error=x, energy_error=0.0, k0=0.97,
                                 logical=0.0) for x in (1e-4, 2e-4, 3e-4)]
    rows = handoff.rule_b_table(rec, prep, None, C_QPE, certificates=[cert], loop="free", stats=stats)
    assert [r.identification_error for r in rows] == [1e-4, 2e-4, 3e-4]
    with pytest.raises(ValueError):
        handoff.rule_b_table(rec, prep, None, C_QPE, certificates=[cert], loop="free", stats=stats[:2])
    with pytest.raises(ValueError):
        handoff.rule_b_table(rec, prep, None, C_QPE, certificates=[1.0], loop="free")
    with pytest.raises(ValueError):
        handoff.optimal_rule_b(rows, "expected")
    with pytest.raises(ValueError):
        handoff.RuleBRow(k=0, p=0.5, e_err=0.1, c_prep=0.0, c_loop=0.0, form="B_U", certified=True, certificate=None,
                         theta=-1.0, n_runs=1, cost=1.0, cost_oracle_e=1.0)
    with pytest.raises(ValueError):
        handoff.RuleBRow(k=0, p=0.5, e_err=0.1, c_prep=0.0, c_loop=0.0, form="B_X", certified=True, certificate=None,
                         theta=-1.0, n_runs=1, cost=1.0, cost_oracle_e=1.0)
    assert handoff.RULES == ("expected", "expected_oracle", "delta", "delta_oracle")
    assert handoff.RULES_B == ("bracketed", "bracketed_oracle")


# ------------------------------------------------------------------ M3's T6 under rule B
def _t6_system(name: str):
    a = symmetry.benchmark_system(name)
    ir = symmetry.orbital_irreps(a)
    ref = ExactReference.build(a)
    enc = (ftqpe.thc_encoding(a, 12, penalty=0.0, reference=ref) if name == "h6s"
           else ftqpe.sf_encoding(a, 17, reference=ref))
    q = ftqpe.qpe_cost(enc, DE)
    spec = acc.model_spectrum(a, q, ir)
    e1 = spec.e1_any_irrep if name == "h6s" else spec.e1_singlet          # THC: the any-irrep singlet E_1 (F41)
    e1lb = certify.e1_lower_bound(e1, model_spectrum=True)
    u = certify.truncated_ci_energy(enc.model(a), 2)
    return a, ir, q, spec, e1lb, u


def test_rule_b_on_m3_t6():
    """M3's T6 under rule B [design, 1e-3 relative]: h6s measured loops (both metrics): HF at 2.744 C_QPE (CISD, B_U
    with D_U = 25.4, so n_v = 2: 1.744 + 1); n2s spin-free classical-first-step loop: HF at 3.956; n2s spin-orbital
    classical-first-step loop: Psi_1 at 2.525 (B_U, D_U = 28.7, n_v = 1).

    The measured loops cost C_iter = 100 C_QPE per iteration (M3's are 516-570 C_QPE): the HF optimum does not
    depend on it; the classical-first-step loop covers k <= 1 without iteration costs.
    """
    a, ir, q, spec, e1lb, u = _t6_system("h6s")
    grid = spec.grid(acc.qpe_points(q))
    certs = [certify.EnergyCertificate("classical", u, None, 0.0, 0.0, 0.0, "cisd")]
    assert (e1lb - u - DE) / grid == pytest.approx(25.4, abs=0.3)
    roots = a.fci(nroots=12, spin_square=0.0)
    for metric in ("spin_free", "spin_orbital"):
        rec = exact_trajectory(a, optimizer="lbfgs", metric=metric, max_iter=4, irreps=ir, roots=roots)
        prep = stateprep.prep_cost(rec.directions, rec.eps, irreps=ir).cumulative
        iters = [100.0 * q.toffolis] * rec.n_steps
        rows = handoff.rule_b_table(rec, prep, iters, q, certificates=certs, loop="measured", e1_lower=e1lb,
                                    grid=grid, e0_model=spec.e0)
        best = handoff.optimal_rule_b(rows)
        assert best.k == 0 and best.form == "B_U" and best.n_runs == 2 and best.certified
        assert best.cost / q.toffolis == pytest.approx(2.744, rel=1e-3)
    a, ir, q, spec, e1lb, u = _t6_system("n2s")
    grid = spec.grid(acc.qpe_points(q))
    certs = [certify.EnergyCertificate("classical", u, None, 0.0, 0.0, 0.0, "cisd")]
    assert (e1lb - u - DE) / grid == pytest.approx(28.7, abs=0.1)
    roots = a.fci(nroots=12, spin_square=0.0)
    for metric, k_ref, t_ref in (("spin_free", 0, 3.956), ("spin_orbital", 1, 2.525)):
        rec = exact_trajectory(a, optimizer="lbfgs", metric=metric, max_iter=1, irreps=ir, roots=roots,
                               first_step="second_order")
        prep = stateprep.prep_cost(rec.directions, rec.eps, irreps=ir).cumulative
        rows = handoff.rule_b_table(rec, prep, None, q, certificates=certs, loop="classical_first_step",
                                    e1_lower=e1lb, grid=grid, e0_model=spec.e0)
        best = handoff.optimal_rule_b(rows)
        assert (best.k, best.form, best.n_runs) == (k_ref, "B_U", 1) and best.certified
        assert best.cost / q.toffolis == pytest.approx(t_ref, rel=1e-3)
        assert best.cost == best.cost_oracle_e
