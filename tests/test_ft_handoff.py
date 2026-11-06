"""Tests of ft.handoff: the CQE -> QPE hand-off (F15) and the CQE-only energy (F17).

The references are the defining formulas evaluated directly on the inputs (the totals
T_E, T_d, the marginal condition), the number of runs from its definition in exact rationals
(and 60+ digit decimal arithmetic for large R) on the exact binary inputs, the exact
spectrum of the dense H4 CI Hamiltonian for the overlap certificate, and hand-built
trajectories.
"""

from __future__ import annotations

import dataclasses
import decimal
import fractions
import math

import numpy as np
import pytest

from thc_engine_cqe.fci_ops import Hamiltonian
from thc_engine_cqe.ft import handoff, symmetry
from thc_engine_cqe.ft.primitives import FTParameters
from thc_engine_cqe.ft.qpe import BlockEncoding, QPECost
from thc_engine_cqe.ft.trajectory import TrajectoryRecord

C_QPE = 2.0e7


def _record(p, e, var, e0=-1.0, e1=-0.4, e1_any=-0.5) -> TrajectoryRecord:
    """A hand-built trajectory with overlaps p_k, energies e_k and variances var_k."""
    K = len(p) - 1
    return TrajectoryRecord(system="test", optimizer="lbfgs", norb=4, nelec=(2, 2), e0=e0, e1=e1, e1_any=e1_any,
                            energies=np.asarray(e, float), variances=np.asarray(var, float),
                            residual_norms=np.zeros(K + 1), overlaps=np.asarray(p, float), eps=np.zeros(K),
                            directions=[np.zeros((4,) * 4)] * K, states=[np.zeros((6, 6))] * (K + 1), irreps=None)


def _random_case(rng, K: int = 8):
    p = np.sort(rng.uniform(0.5, 1.0, K + 1))
    e = -1.0 + (1.0 - p) * 0.3
    var = (1.0 - p) * 0.05
    prep = np.concatenate([[0.0], np.cumsum(rng.uniform(1e2, 1e5, K))])
    iters = rng.uniform(0.0, 3e6, K) * rng.choice([1e-3, 1.0], K)
    return _record(p, e, var), prep, iters


def test_hf_row_and_loops():
    """At k = 0: cost_expected = C_QPE/p_0 and saving = 0 for every loop; C_loop follows the loop rule.

    C_loop(k) is recomputed as the partial sums of the iteration costs with C_iter(0) zeroed
    for the classical first step and all zeroed for the free loop.
    """
    rng = np.random.default_rng(0)
    rec, prep, iters = _random_case(rng)
    for loop in handoff.LOOPS:
        rows = handoff.handoff_table(rec, prep, iters, C_QPE, loop=loop)
        assert len(rows) == len(rec.energies)
        assert rows[0].cost_expected == pytest.approx(C_QPE / rec.overlaps[0], rel=1e-15)
        assert rows[0].saving_vs_hf == 0.0 and rows[0].c_loop == 0.0 and rows[0].c_prep == 0.0
        c = iters.copy()
        if loop == "classical_first_step":
            c[0] = 0.0
        if loop == "free":
            c[:] = 0.0
        np.testing.assert_allclose([r.c_loop for r in rows], np.concatenate([[0.0], np.cumsum(c)]), rtol=1e-14)
        for r in rows:
            run = prep[r.k] + C_QPE
            assert r.cost_expected == pytest.approx(r.c_loop + run / r.p, rel=1e-14)
            assert r.runs_expected == pytest.approx(1.0 / r.p, rel=1e-15)
            assert r.saving_vs_hf == pytest.approx(C_QPE / rec.overlaps[0] - run / r.p, rel=1e-12, abs=1e-6)
            assert r.saving_vs_hf <= C_QPE * (1.0 / rec.overlaps[0] - 1.0) + 1e-6       # the budget
            assert r.cost_delta_oracle == pytest.approx(r.c_loop + r.runs_delta_oracle * run, rel=1e-14)
    assert handoff.handoff_table(rec, prep, None, C_QPE, loop="free")[-1].c_loop == 0.0
    with pytest.raises(ValueError):
        handoff.handoff_table(rec, prep, None, C_QPE, loop="measured")
    with pytest.raises(ValueError):
        handoff.handoff_table(rec, prep, iters, C_QPE, loop="cheap")


def _runs_reference(p: float, d: float) -> int:
    """min{R >= 1 : (1 - p)^R <= d} for the exact binary values of p and d.

    ceil(ln d / ln(1 - p)) in decimal arithmetic with 60 digits beyond those of p; where the
    quotient is an integer to that precision, (1 - p)^R <= d is decided in exact rationals.
    """
    with decimal.localcontext() as ctx:
        ctx.prec = 60 + max(0, int(-math.log10(p)))
        q = 1 - decimal.Decimal(p)
        if q <= decimal.Decimal(d):
            return 1
        ratio = decimal.Decimal(d).ln() / q.ln()
        r = int(ratio.to_integral_value(rounding=decimal.ROUND_CEILING))
        near = int(ratio.to_integral_value(rounding=decimal.ROUND_HALF_EVEN))
        if abs(ratio - near) < decimal.Decimal(10) ** (-50):
            r = near if (1 - fractions.Fraction(p)) ** near <= fractions.Fraction(d) else near + 1
    return max(1, r)


def _satisfies_definition(R: int, p: float, d: float) -> bool:
    """(1 - p)^R <= d < (1 - p)^(R - 1) in exact rationals on the binary values of p and d."""
    q, dd = 1 - fractions.Fraction(p), fractions.Fraction(d)
    return q ** R <= dd and (R == 1 or q ** (R - 1) > dd)


def test_runs_for_confidence_against_exact_arithmetic():
    """runs_for_confidence is min{R : (1 - p)^R <= delta} for the exact binary values of p and delta.

    References: the defining inequality in exact rationals where R <= 3000 (no logarithm
    involved), high-precision decimal logarithms above. Cases: 2000 random (p, delta) with p in
    [1e-4, 1]; small p (1e-9, 1e-12), where 1 - p rounds in double precision (a (1 - p) ** R
    criterion gives 130 runs too many at p = 1e-9); p = 1 - delta^(1/n) and its neighbours
    within 2 ulps, quotients ln(delta)/ln(1 - p) within rounding of an integer, where the float
    ceiling alone is one run short in some cases (counted, so that the exact path is
    exercised); exact ties ((1/2)^2 = 1/4, (1/2)^30 = 2^-30, 3/4 = 3/4); (0.99, 0.01) needs 2
    runs, since 1 - 0.99 = 0.0100000000000000089 > 0.01 in binary; relative agreement 1e-12
    above 2^50 runs (p = 1e-17, 1e-200), where the float ceiling is used.
    """
    rng = np.random.default_rng(1)
    cases = [(float(p), float(d)) for p, d in zip(rng.uniform(1e-4, 1.0, 2000), 10 ** rng.uniform(-6, -0.5, 2000))]
    for p, d in cases + [(1e-9, 0.01), (1e-12, 0.01), (1e-9, 1e-6)]:
        R = handoff.runs_for_confidence(p, d)
        assert _satisfies_definition(R, p, d) if R <= 3000 else R == _runs_reference(p, d)
    short = 0
    for d in (0.5, 0.125, 0.01, 0.001):
        for n in range(2, 81):
            p = math.nextafter(math.nextafter(1.0 - d ** (1.0 / n), 0.0), 0.0)
            for _ in range(5):
                R = handoff.runs_for_confidence(p, d)
                assert _satisfies_definition(R, p, d)
                short += math.ceil(math.log(d) / math.log1p(-p)) < R
                p = math.nextafter(p, 1.0)
    assert short > 0
    for p in (1e-17, 1e-200):
        R = handoff.runs_for_confidence(p, 0.01)
        assert abs(R / _runs_reference(p, 0.01) - 1.0) < 1e-12
    ties = [(0.5, 0.25), (0.5, 0.125), (0.5, 2.0 ** -30), (0.25, 0.75)]
    assert [handoff.runs_for_confidence(p, d) for p, d in ties] == [2, 3, 30, 1]
    assert handoff.runs_for_confidence(0.99, 0.01) == 2 == _runs_reference(0.99, 0.01)
    assert handoff.runs_for_confidence(1.0, 0.01) == 1 and handoff.runs_for_confidence(0.995, 0.01) == 1
    assert handoff.runs_for_confidence(0.9455) == 2          # H4 HF overlap: 2 runs [design]
    for bad in (0.0, -0.1, 1.1, math.nan):
        with pytest.raises(ValueError):
            handoff.runs_for_confidence(bad)
    with pytest.raises(ValueError):
        handoff.runs_for_confidence(0.5, 1.0)
    with pytest.raises(ValueError):
        handoff.runs_for_confidence(1e-310)


def test_break_even_overlap():
    """At p_HF = p0*(k) the HF hand-off C_QPE / p_HF equals T_E(k), for every k and loop."""
    rng = np.random.default_rng(2)
    rec, prep, iters = _random_case(rng)
    for loop in handoff.LOOPS:
        for r in handoff.handoff_table(rec, prep, iters, C_QPE, loop=loop):
            assert C_QPE / r.p0_break_even == pytest.approx(r.cost_expected, rel=1e-14)


def test_marginal_condition_property():
    """Random trajectories: T_E(k+1) < T_E(k) iff C_iter(k) < (C_k + C_QPE)/p_k - (C_{k+1} + C_QPE)/p_{k+1}.

    The right side is evaluated directly on the inputs (measured loop, so C_iter(k) enters
    C_loop(k+1) undiminished); 300 random cases with iteration costs spread over the
    break-even scale, so both outcomes occur.
    """
    rng = np.random.default_rng(3)
    seen = set()
    for _ in range(300):
        rec, prep, iters = _random_case(rng)
        rows = handoff.handoff_table(rec, prep, iters, C_QPE, loop="measured")
        p = rec.overlaps
        for k in range(len(rows) - 1):
            margin = (prep[k] + C_QPE) / p[k] - (prep[k + 1] + C_QPE) / p[k + 1]
            pays = iters[k] < margin
            seen.add(pays)
            if abs(iters[k] - margin) > 1e-6 * C_QPE:
                assert (rows[k + 1].cost_expected < rows[k].cost_expected) == pays
    assert seen == {True, False}


def test_infinite_iteration_costs_and_coverage():
    """C_iter = +inf (an infeasible route) makes T_E, T_d infinite from the next row on; coverage per loop.

    Rows up to k = j keep their finite costs; nan and negative costs raise. The classical
    first step covers k <= 1 without any measured cost ([] or None) and ignores C_iter(0);
    the measured loop with [] has only the HF row.
    """
    rng = np.random.default_rng(6)
    rec, prep, iters = _random_case(rng, K=5)
    iters = iters.copy()
    iters[2] = math.inf
    rows = handoff.handoff_table(rec, prep, iters, C_QPE, loop="measured")
    assert len(rows) == 6
    for r in rows:
        finite = r.k <= 2
        assert math.isfinite(r.cost_expected) == finite and math.isfinite(r.cost_delta_oracle) == finite
        assert (r.p0_break_even > 0.0) == finite
        if r.cost_delta is not None:
            assert math.isfinite(r.cost_delta) == finite
        assert math.isfinite(r.saving_vs_hf)
    for bad in (math.nan, -1.0):
        broken = iters.copy()
        broken[1] = bad
        with pytest.raises(ValueError):
            handoff.handoff_table(rec, prep, broken, C_QPE, loop="measured")
    for given in ([], None, [math.nan]):
        rows = handoff.handoff_table(rec, prep, given, C_QPE, loop="classical_first_step")
        assert [r.k for r in rows] == [0, 1] and rows[1].c_loop == 0.0
    rows = handoff.handoff_table(rec, prep, [5.0, 7.0], C_QPE, loop="classical_first_step")
    assert [r.c_loop for r in rows] == [0.0, 0.0, 7.0]
    assert [r.k for r in handoff.handoff_table(rec, prep, [], C_QPE, loop="measured")] == [0]


def test_first_step_flags_and_overlap_roundoff():
    """Row flags mark a loop applied to a record of the other first step; overlaps above 1 read as 1.

    The classical-first-step loop on an exact-line-search record flags rows k >= 1, the
    measured loop on a classical-step record likewise, the matching pairs and the free loop
    flag nothing. With p_0 = 1 + 1e-15 (round-off) the HF saving is exactly 0.
    """
    rng = np.random.default_rng(7)
    rec, prep, iters = _random_case(rng, K=3)
    rows = handoff.handoff_table(rec, prep, iters, C_QPE)
    assert rows[0].flags == () and all(r.flags == ("exact_first_step",) for r in rows[1:])
    assert all(r.flags == () for r in handoff.handoff_table(rec, prep, iters, C_QPE, loop="free"))
    assert all(r.flags == () for r in handoff.handoff_table(rec, prep, iters, C_QPE, loop="measured"))
    classical = dataclasses.replace(rec, first_step="second_order")
    rows = handoff.handoff_table(classical, prep, iters, C_QPE)
    assert [r.flags for r in rows] == [(), (), (), ()]
    rows = handoff.handoff_table(classical, prep, iters, C_QPE, loop="measured")
    assert rows[0].flags == () and all(r.flags == ("second_order_first_step",) for r in rows[1:])
    above = _record([1.0 + 1e-15, 1.0], [-1.0, -1.0], [0.0, 0.0])
    row0 = handoff.handoff_table(above, [0.0, 1.0], None, C_QPE, loop="free")[0]
    assert row0.p == 1.0 and row0.saving_vs_hf == 0.0 and row0.cost_expected == C_QPE


def _h4_spectrum():
    a = symmetry.benchmark_system("h4")
    ham = Hamiltonian(a)
    sp = ham.space
    dense = np.column_stack([ham.apply(np.eye(sp.size)[i].reshape(sp.shape)).ravel() for i in range(sp.size)])
    e, v = np.linalg.eigh(0.5 * (dense + dense.T) + a.ecore * np.eye(sp.size))
    return e, v


def test_overlap_lower_bound_on_h4_states():
    """Eckart-Temple: p_lb <= p on random H4 states with E < E1, equality for two-level states, nan for E >= E1.

    E1 is the first excited eigenvalue of the dense CI Hamiltonian (any spin and irrep, the
    right bound for unconstrained random states); p, E and sigma^2 are computed from the
    eigen-decomposition, independently of the trajectory code.
    """
    e, v = _h4_spectrum()
    rng = np.random.default_rng(4)
    checked = 0
    for _ in range(400):
        c = v[:, 0] + rng.uniform(0.02, 1.0) * rng.standard_normal(e.size) / np.sqrt(e.size)
        c /= np.linalg.norm(c)
        w = (v.T @ c) ** 2
        energy = float(w @ e)
        var = float(w @ (e - energy) ** 2)
        lb = handoff.overlap_lower_bound(energy, var, e[1])
        if energy < e[1]:
            assert lb <= w[0] + 1e-12
            checked += 1
        else:
            assert math.isnan(lb)
    assert checked > 100
    for p in (0.2, 0.6, 0.9455, 0.999):
        w = np.zeros(e.size)
        w[0], w[1] = p, 1.0 - p
        energy = float(w @ e)
        var = float(w @ (e - energy) ** 2)
        assert handoff.overlap_lower_bound(energy, var, e[1]) == pytest.approx(p, rel=1e-10)
    assert handoff.overlap_lower_bound(-1.0, 0.0, -0.5) == 1.0
    with pytest.raises(ValueError):
        handoff.overlap_lower_bound(-1.0, -0.1, -0.5)


def test_certificates_in_the_table():
    """p_lb, R_d(p_lb), T_d and rule-E certification per row, with the same-irrep or any-irrep E1.

    Rows with E_k >= E1 have p_lb = nan and no certified cost; R_d with the exact p_k (oracle)
    is always present; certification is E1 - E_k > 3 dE with dE from the QPECost. The inputs
    are synthetic (not a physical state), so only the wiring of the formulas is checked here.
    """
    p = np.array([0.30, 0.60, 0.95, 0.999])
    e = np.array([-0.30, -0.45, -0.90, -0.995])
    var = np.array([0.04, 0.03, 0.002, 1e-5])
    rec = _record(p, e, var, e0=-1.0, e1=-0.40, e1_any=-0.50)
    prep = np.array([0.0, 1e3, 3e3, 6e3])
    rows = handoff.handoff_table(rec, prep, [0.0, 5e5, 5e5], C_QPE)
    assert math.isnan(rows[0].p_lb) and rows[0].runs_delta is None and rows[0].cost_delta is None
    assert not rows[0].certifiable
    for r in rows[1:]:
        assert r.p_lb == pytest.approx(1.0 / (1.0 + var[r.k] / (-0.40 - e[r.k]) ** 2), rel=1e-14)
        assert r.runs_delta == handoff.runs_for_confidence(r.p_lb, 0.01)
        assert r.cost_delta == pytest.approx(r.c_loop + r.runs_delta * (prep[r.k] + C_QPE), rel=1e-14)
        assert r.runs_delta_oracle == handoff.runs_for_confidence(r.p, 0.01)
        assert r.certifiable == (-0.40 - e[r.k] > 3e-3)
    any_rows = handoff.handoff_table(rec, prep, [0.0, 5e5, 5e5], C_QPE, use_e1="any_irrep")
    assert math.isnan(any_rows[1].p_lb) and any_rows[2].p_lb < rows[2].p_lb
    enc = BlockEncoding(kind="df", params={}, lam=1.0, lam_one_body=0.5, lam_two_body=0.5, de_fact=0.0,
                        infidelity=0.0, toffoli_per_step=1000, ancilla_qpe=100, ancilla_walk=80,
                        qr_clamped=False, chi=10, beta=16)
    coarse = QPECost(encoding=enc, de_qpe=0.2, iterations=int(C_QPE // 1000), toffolis=int(C_QPE), logical_qubits=100)
    assert [r.certifiable for r in handoff.handoff_table(rec, prep, [0.0, 5e5, 5e5], coarse)] == [
        False, False, False, False]
    with pytest.raises(ValueError):
        handoff.handoff_table(rec, prep, [0.0, 5e5, 5e5], C_QPE, use_e1="lowest")


def test_optimal_handoff():
    """optimal_handoff: certified-only minima for "expected" and "delta", all rows for the oracle rules.

    Random free-loop tables are checked against the minima recomputed over the qualifying
    rows. A hand-built record whose HF energy lies above E1 (rule E not certified there)
    with the cheapest T_E at k = 0 (expensive measured iterations, as h6s and n2s): "expected"
    must return the cheapest certified row (k = 1), "expected_oracle" k = 0. A table without
    certified rows raises for "expected" and "delta".
    """
    rng = np.random.default_rng(5)
    rec, prep, iters = _random_case(rng)
    rows = handoff.handoff_table(rec, prep, iters, C_QPE, loop="free")
    certified_e = [r for r in rows if r.certifiable]
    assert handoff.optimal_handoff(rows).cost_expected == min(r.cost_expected for r in certified_e)
    assert handoff.optimal_handoff(rows, "expected_oracle").cost_expected == min(r.cost_expected for r in rows)
    certified = [r for r in rows if r.cost_delta is not None]
    if certified:
        assert handoff.optimal_handoff(rows, "delta").cost_delta == min(r.cost_delta for r in certified)
    assert handoff.optimal_handoff(rows, "delta_oracle").cost_delta_oracle == min(r.cost_delta_oracle for r in rows)
    above = _record([0.30, 0.90, 0.99], [-0.20, -0.95, -0.99], [0.04, 0.003, 1e-4], e0=-1.0, e1=-0.40)
    rows = handoff.handoff_table(above, [0.0, 1e3, 2e3], [1e9, 1e9], C_QPE, loop="measured")
    assert [r.certifiable for r in rows] == [False, True, True]
    assert min(rows, key=lambda r: r.cost_expected).k == 0
    assert handoff.optimal_handoff(rows).k == 1 and handoff.optimal_handoff(rows, "expected_oracle").k == 0
    uncertified = handoff.handoff_table(_record([0.3, 0.4], [-0.3, -0.35], [0.05, 0.05]), [0.0, 10.0],
                                        [0.0], C_QPE)
    for rule in ("expected", "delta"):
        with pytest.raises(ValueError):
            handoff.optimal_handoff(uncertified, rule)
    assert handoff.optimal_handoff(uncertified, "expected_oracle").k == 1
    with pytest.raises(ValueError):
        handoff.optimal_handoff(rows, "median")


def test_cqe_only_cost():
    """F17 recomputed by hand: k_v, loop shots and Toffolis to k_v, S_fin = V_E / sigma_final^2, ratio to C_QPE/p_0."""
    p = np.array([0.94, 0.98, 0.996, 0.9981, 0.9997])
    e = -1.0 + np.array([0.062, 0.0215, 0.0039, 0.0012, 0.0004])
    rec = _record(p, e, np.full(5, 1e-3))
    prep = np.array([0.0, 160.0, 1352.0, 2506.0, 3600.0])
    iters = np.array([5e6, 7e8, 9e9, 2e10])
    shots = np.array([1e5, 2e6, 3e7, 4e8])
    par = FTParameters()
    out = handoff.cqe_only_cost(rec, prep, iters, shots, 0.59, 405, C_QPE, params=par)
    k_v = 4                                                  # first E_k - E_0 <= 1 mHa
    s_fin = 0.59 / 3e-4 ** 2
    tof = iters[1:k_v].sum() + s_fin * (prep[k_v] + 405)
    assert out.k_v == k_v and out.e_err == pytest.approx(4e-4, abs=1e-15)
    assert out.final_shots == pytest.approx(s_fin, rel=1e-14)
    assert out.shots == pytest.approx(shots[1:k_v].sum() + s_fin, rel=1e-14)
    assert out.toffolis == pytest.approx(tof, rel=1e-14)
    assert out.ratio_to_hf_qpe == pytest.approx(tof / (C_QPE / p[0]), rel=1e-14)
    assert out.flags == ("exact_first_step",)
    measured = handoff.cqe_only_cost(rec, prep, iters, shots, 0.59, 405, C_QPE, loop="measured")
    assert measured.toffolis == pytest.approx(tof + iters[0], rel=1e-14) and measured.flags == ()
    never = handoff.cqe_only_cost(rec, prep, iters, shots, 0.59, 405, C_QPE, params=FTParameters(e_tol_energy=1e-6))
    assert never.k_v is None and math.isinf(never.toffolis) and never.e_err == pytest.approx(4e-4, abs=1e-15)
    with pytest.raises(ValueError):
        handoff.cqe_only_cost(rec, prep, iters[:2], shots, 0.59, 405, C_QPE)


def test_certificates_on_exact_trajectories():
    """p_lb along exact trajectories: the design values for H4 (0.76, 0.92, 0.98, 1.00 at k = 0-3) and N2.

    With the same-irrep E1 of F15 the N2 HF certificate is 0.59 instead of 0.26 (any irrep),
    and rule E becomes certifiable or stays so. The trajectories come from
    ft.trajectory.exact_trajectory; the reference values are those of the design notes [design].
    """
    from thc_engine_cqe.ft.trajectory import exact_trajectory

    a = symmetry.benchmark_system("h4")
    rec = exact_trajectory(a, system="h4", irreps=symmetry.orbital_irreps(a), max_iter=3)
    prep = np.zeros(4)
    for use_e1 in ("same_irrep", "any_irrep"):
        rows = handoff.handoff_table(rec, prep, None, C_QPE, loop="free", use_e1=use_e1)
        np.testing.assert_allclose([r.p_lb for r in rows], [0.76, 0.92, 0.98, 1.00], atol=6e-3)
        assert all(r.certifiable for r in rows)
    n2 = symmetry.benchmark_system("n2")
    hf = exact_trajectory(n2, system="n2", irreps=symmetry.orbital_irreps(n2), max_iter=0)
    same = handoff.handoff_table(hf, [0.0], None, C_QPE, loop="free")[0]
    any_ = handoff.handoff_table(hf, [0.0], None, C_QPE, loop="free", use_e1="any_irrep")[0]
    assert abs(same.p_lb - 0.59) < 5e-3 and abs(any_.p_lb - 0.26) < 5e-3
    assert same.runs_delta == handoff.runs_for_confidence(same.p_lb) and same.runs_delta < any_.runs_delta
