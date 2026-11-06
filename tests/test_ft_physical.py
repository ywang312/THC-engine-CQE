"""Tests of the surface-code layer (``thc_engine_cqe.ft.physical``, F18, F19).

References independent of the module: OpenFermion's ``physical_costing.cost_estimator`` (the source
F18 transcribes; tests that need it start with importorskip), the physical numbers of the M3
specification (Appendix A, recomputed by the critic's ``phys_check.py`` with OpenFermion's own
``AlgorithmParameters.estimate_cost``), and closed forms of the cycle counts and of the wall-clock
model evaluated here.
"""

import math

import numpy as np
import pytest

from thc_engine_cqe.ft import physical as ph

OF_MODEL = ph.SurfaceCodeModel(distances=tuple(range(7, 35, 2)))      # cost_estimator's settings


def _cost_estimator():
    pytest.importorskip("openfermion")
    from openfermion.resource_estimates.surface_code_compilation import physical_costing

    return physical_costing.cost_estimator


def test_matches_openfermion_cost_estimator():
    """With OpenFermion's settings (budget 0.1, d = 7..33) the layout equals cost_estimator's exactly
    (physical qubits, distance, seconds and failure) for the three cases of the specification and 60
    random (Q, N_Tof), also at p = 1e-4 and 5e-4, where only the AutoCCZ factories exist."""
    cost_estimator = _cost_estimator()
    rng = np.random.default_rng(11)
    cases = [(233, 19_997_593, 1e-3), (2142, 5_250_145_120, 1e-3), (83, 43_427, 1e-3)]
    cases += [(int(rng.integers(4, 4000)), int(10 ** rng.uniform(1, 10.5)), p)
              for p in (1e-3, 1e-4, 5e-4) for _ in range(20)]
    compared = 0
    for q, n, p in cases:
        cost, params = cost_estimator(q, n, physical_error_rate=p)
        if cost is None:                 # no layout within d <= 33 (OpenFermion returns None)
            continue
        model = ph.SurfaceCodeModel(p_phys=p, distances=tuple(range(7, 35, 2)))
        mine = ph.surface_code_cost(q, n, fail_budget=0.1, model=model)
        assert mine.physical_qubits == cost.physical_qubit_count
        assert mine.distance == params.logical_data_qubit_distance
        assert mine.seconds == cost.duration.total_seconds()
        assert mine.failure == cost.algorithm_failure_probability
        compared += 1
    assert compared >= 55


def test_appendix_a_physical_numbers():
    """Appendix A: the H6 QPE run (233 logical qubits, 1.9998e7 Toffolis) needs 956,160 qubits, d = 23,
    574.9 s at budget 0.1 and 1,026,160 qubits, d = 25 at 0.01; one H6 Psi_6 shot (83 qubits, 43,427
    Toffolis, plus 2d readout cycles) 721,960 qubits, d = 25, 1.249 s at 1e-5 and 696,960 qubits, d = 23
    at 1e-4. The factory-limited rate is 5 l2 / 4 us per Toffoli: 28.75 (l2 = 23) and 26.25 (l2 = 21,
    the H4 QPE run of 6.73e6 Toffolis on 188 qubits at budget 0.1)."""
    qpe = ph.surface_code_cost(233, 19_997_593, fail_budget=0.1)
    assert (qpe.physical_qubits, qpe.distance) == (956_160, 23)
    assert qpe.seconds == pytest.approx(574.93, abs=5e-3) and qpe.failure <= 0.1
    assert qpe.us_per_toffoli == pytest.approx(28.75, rel=1e-8)
    qpe = ph.surface_code_cost(233, 19_997_593, fail_budget=0.01)
    assert (qpe.physical_qubits, qpe.distance) == (1_026_160, 25) and qpe.seconds == pytest.approx(574.93, abs=5e-3)
    shot = ph.surface_code_cost(83, 43_427, fail_budget=1e-5, shot=True)
    assert (shot.physical_qubits, shot.distance) == (721_960, 25)
    assert shot.seconds == pytest.approx(1.249, abs=5e-4) and shot.failure <= 1e-5
    shot = ph.surface_code_cost(83, 43_427, fail_budget=1e-4, shot=True)
    assert (shot.physical_qubits, shot.distance) == (696_960, 23)
    h4 = ph.surface_code_cost(188, 6.73e6, fail_budget=0.1)
    assert h4.factory.endswith("l2=21)") and h4.us_per_toffoli == pytest.approx(26.25, rel=1e-6)


def test_no_overflow_at_large_toffoli_counts():
    """OpenFermion's cost_estimator overflows (timedelta x qubits) at 1e12 Toffolis on 233 qubits; F18
    with float seconds gives 1e12 Toffolis (AutoCCZ(19, 33), d = 35, 5 x 33 / 4 = 41.25 us per Toffoli),
    1e13 and 1e14 (the two-level T factory, 6 x 31 / 4 = 46.5 us, d = 35 and 37), and raises ValueError
    only when no layout meets the budget."""
    cost_estimator = _cost_estimator()
    with pytest.raises(OverflowError):
        cost_estimator(233, 10 ** 12)
    for n, d, factory, us in ((1e12, 35, "AutoCCZ(l1=19, l2=33)", 41.25), (1e13, 35, "two-level T", 46.5),
                              (1e14, 37, "two-level T", 46.5)):
        c = ph.surface_code_cost(233, n, fail_budget=0.1)
        assert c.distance == d and c.factory.startswith(factory)
        assert math.isfinite(c.seconds) and c.us_per_toffoli == pytest.approx(us, rel=1e-9)
    with pytest.raises(ValueError):
        ph.surface_code_cost(233, 3e14, fail_budget=0.1)    # 3.6e-16 per state x 3e14 > 0.1


def test_objective_is_monotone_in_toffolis_and_budget():
    """Every layout fails more often with more Toffolis, and a tighter budget only removes layouts, so
    the optimal physical qubits x seconds is non-decreasing in N_Tof and in 1/budget, and once no layout
    is feasible none is for larger N_Tof or tighter budgets (theorems of the model; checked on grids).
    Every chosen layout meets its budget."""

    def objectives(points):
        out = []
        for q, n, budget in points:
            try:
                c = ph.surface_code_cost(q, n, fail_budget=budget)
            except ValueError:
                out.append(math.inf)
                continue
            assert c.failure <= budget
            out.append(c.physical_qubits * c.seconds)
        return out

    for q in (20, 83, 233, 2000):
        for budget in (0.1, 1e-2, 1e-5):
            obj = objectives([(q, 10 ** e, budget) for e in np.arange(1.0, 15.01, 0.25)])
            assert all(x <= y for x, y in zip(obj, obj[1:])) and math.isfinite(obj[0]) and obj[-1] == math.inf
        for n in (1e2, 1e5, 1e8, 1e11):
            obj = objectives([(q, n, b) for b in (0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-12)])
            assert all(x <= y for x, y in zip(obj, obj[1:])) and math.isfinite(obj[0])


def test_distance_is_not_monotone_in_toffolis_and_openfermion_agrees():
    """The specification's 'd is non-decreasing in N_Tof' does not hold for the area x time optimum:
    233 qubits at budget 0.1 take d = 15 (AutoCCZ(9, 15)) for 562 Toffolis and d = 13 (the faster
    AutoCCZ(11, 13), shorter run, less data failure) for 1000. OpenFermion's cost_estimator makes the
    same choice, so this is the model, not the transcription."""
    cost_estimator = _cost_estimator()
    a = ph.surface_code_cost(233, 562, fail_budget=0.1, model=OF_MODEL)
    b = ph.surface_code_cost(233, 1000, fail_budget=0.1, model=OF_MODEL)
    assert (a.distance, a.factory) == (15, "AutoCCZ(l1=9, l2=15)")
    assert (b.distance, b.factory) == (13, "AutoCCZ(l1=11, l2=13)")
    for n, d in ((562, 15), (1000, 13)):
        assert cost_estimator(233, n)[1].logical_data_qubit_distance == d


def test_reaction_floor_binds_when_set():
    """With a reaction time t_r the run lasts at least ceil(depth t_r / t_cycle) cycles: for the H6 shot
    with t_r = 100 us and the sequential depth 43,427 that is 4.3427 s (instead of 1.249 s), and the
    floor is inactive for a shallow depth or without reaction_us."""
    free = ph.surface_code_cost(83, 43_427, fail_budget=1e-5, shot=True)
    model = ph.SurfaceCodeModel(reaction_us=100.0)
    floored = ph.surface_code_cost(83, 43_427, fail_budget=1e-5, shot=True, model=model)
    assert floored.seconds == pytest.approx(43_427 * 100e-6, rel=1e-12) and floored.failure <= 1e-5
    assert floored.seconds > free.seconds and floored.distance >= free.distance
    shallow = ph.surface_code_cost(83, 43_427, fail_budget=1e-5, shot=True, model=model, toffoli_depth=100)
    assert shallow == free
    with pytest.raises(ValueError):
        ph.surface_code_cost(83, 43_427, fail_budget=1e-5, model=model, toffoli_depth=-1.0)


@pytest.mark.parametrize("cycle_us", [1.0, 2.5])
def test_clifford_only_shot_lasts_two_d_cycles(cycle_us):
    """A shot without Toffolis needs no factory and lasts 2d cycles of readout, at the smallest odd d whose
    data failure 0.1 (100 p)^((d+1)/2) ceil(1.5 Q) 2d meets the budget (closed form evaluated here); the
    area is the data block alone. A circuit without Toffolis and without readout takes no time."""
    model = ph.SurfaceCodeModel(cycle_us=cycle_us)
    for q, budget in ((83, 1e-5), (12, 1e-2), (500, 1e-8)):
        c = ph.surface_code_cost(q, 0, fail_budget=budget, shot=True, model=model)
        storage = math.ceil(1.5 * q)
        d = next(d for d in range(7, 45, 2) if 0.1 * 0.1 ** ((d + 1) / 2) * storage * 2 * d <= budget)
        assert c.distance == d and c.factory == "none" and math.isnan(c.us_per_toffoli)
        assert c.seconds == pytest.approx(2 * d * cycle_us * 1e-6, rel=1e-12)
        assert c.physical_qubits == storage * 2 * (d + 1) ** 2
    assert ph.surface_code_cost(83, 0, fail_budget=1e-5).seconds == 0.0


def test_campaign_seconds_and_machines():
    """T(n, M) = t_run max(n / M, min(n, 1)) (F19, fluid share, a run is not split), inf for inf runs; the
    machines for a deadline are the smallest M with T(n, M) <= D, and none suffice when one run exceeds
    D or the runs are infinite."""
    run = ph.surface_code_cost(83, 43_427, fail_budget=1e-5, shot=True)
    t = run.seconds
    for n, m in ((1e6, 1), (1e6, 7), (0.3, 1), (0.3, 5), (2.5, 4), (0.0, 3)):
        assert ph.campaign_seconds(n, run, m) == pytest.approx(t * max(n / m, min(n, 1.0)), rel=1e-15)
    assert ph.campaign_seconds(math.inf, run, 10) == math.inf
    week = 7 * 86400
    for n in (1.0, 1e5, 4.84e5, 4.85e5, 1e9, 3.7e12):
        m = ph.machines_for(n, run)
        assert ph.campaign_seconds(n, run, m) <= week
        assert m == 1 or ph.campaign_seconds(n, run, m - 1) > week
    assert ph.machines_for(1e5, run, deadline_s=t * 1e5 / 3.0) == 3
    long_run = ph.surface_code_cost(233, 1e10, fail_budget=0.1)            # 3.9 days per run
    assert 2 * 86400 < long_run.seconds < 4 * 86400
    assert ph.machines_for(10, long_run) == math.ceil(10 * long_run.seconds / week) == 6
    assert ph.machines_for(1, long_run) == 1 and ph.machines_for(0.5, long_run, deadline_s=2 * 86400) == 1
    with pytest.raises(ValueError):
        ph.machines_for(1, long_run, deadline_s=86400.0)
    with pytest.raises(ValueError):
        ph.machines_for(math.inf, run)
    for bad in (-1.0, math.nan):
        with pytest.raises(ValueError):
            ph.campaign_seconds(bad, run)
    with pytest.raises(ValueError):
        ph.campaign_seconds(10, run, 0)


def test_input_validation():
    """Invalid machines and circuits raise ValueError instead of giving a meaningless layout."""
    for bad in ({"p_phys": 0.0}, {"p_phys": 0.02}, {"cycle_us": 0.0}, {"n_factories": 0},
                {"n_factories": 2.5}, {"routing_overhead": -0.1}, {"distances": ()},
                {"distances": (7, 8)}, {"distances": (1,)}, {"reaction_us": 0.0}):
        with pytest.raises(ValueError):
            ph.SurfaceCodeModel(**bad)
    assert ph.SurfaceCodeModel(distances=[7, 9]).distances == (7, 9)
    for args, kwargs in (((0, 100), {"fail_budget": 0.1}), ((10.5, 100), {"fail_budget": 0.1}),
                         ((10, -1), {"fail_budget": 0.1}), ((10, math.inf), {"fail_budget": 0.1}),
                         ((10, 100), {"fail_budget": 0.0}), ((10, 100), {"fail_budget": 1.5})):
        with pytest.raises(ValueError):
            ph.surface_code_cost(*args, **kwargs)
