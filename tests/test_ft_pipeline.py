"""Tests of ft.pipeline: iteration costs (F14), wall clock (F19), the report, its JSON and tables, the golden H4 record.

References independent of the module: F14 sums evaluated in this file on a hand-built residual
input (unit per-shot variances and normalizations) from the defining formulas, with A from
``routes.exact_residual``, V_E from ``schemes.df_variance``, the candidate steps from
``stateprep.elementwise_step_cost`` and the shot layouts from ``physical.surface_code_cost``;
closed-form schedules, exhaustive optimal schedules and scans over the machine count for the
campaign wall clock; the defining sums of F15 and F17 on the report's own inputs (cumulative
C_iter of each loop, ``physical.surface_code_cost``, the StepCost counts) for the hand-off,
CQE-only and table checks; ``json`` itself for the round trip; and the golden record
tests/data/ft_h4_golden.json, recorded once from ``run_system("h4")`` and checked by hand
against Appendix A of docs/m3-cost-model.md and the ``ft`` module docstrings (QPE, trajectories,
first step, hand-off; test_h4_reproduces_the_golden_record).
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import os

import numpy as np
import pytest

from thc_engine_cqe.ft import pipeline, physical, residual_inputs, stateprep, symmetry, trajectory
from thc_engine_cqe.ft.primitives import FTParameters
from thc_engine_cqe.ft.qpe import BlockEncoding
from thc_engine_cqe.ft.residual import RouteCost
from thc_engine_cqe.measure import routes, schemes, targets

GOLDEN = os.path.join(os.path.dirname(__file__), "data", "ft_h4_golden.json")
P = FTParameters()
_CACHE: dict = {}


def _h4():
    """H4 with a 3-step exact L-BFGS record (symmetry-projected), its elementwise prep and irreps."""
    if "h4" not in _CACHE:
        a = symmetry.benchmark_system("h4")
        ir = symmetry.orbital_irreps(a)
        rec = trajectory.exact_trajectory(a, system="h4", irreps=ir, n_roots=3, max_iter=3)
        _CACHE["h4"] = (a, ir, rec, stateprep.prep_cost(rec.directions, rec.eps, irreps=ir))
    return _CACHE["h4"]


def _walk(a) -> BlockEncoding:
    """A hand-built block encoding: Appendix A's H4 DF walk numbers, model ERIs 0.999 x the exact ones."""
    return BlockEncoding(kind="df", params={}, lam=4.935, lam_one_body=1.0, lam_two_body=3.935, de_fact=0.0,
                         infidelity=0.0, toffoli_per_step=868, ancilla_qpe=188, ancilla_walk=163, qr_clamped=True,
                         chi=10, beta=16, eri_model=0.999 * a.eri)


def _unit_inputs(a, ir):
    """The 28 allowed spin-free H4 elements with unit per-shot variances, lambda_Q = 1, alpha_X = 2, L_X = 10."""
    full = targets.element_set(a.norb, "spin_free")
    el = full[symmetry.element_mask(ir, full, "spin_free")]
    k = len(el)
    variances = {"v_direct": np.ones(k), "w_aux": np.ones(k)}
    norms = {"lam_q": np.ones(k), "alpha_x": 2.0 * np.ones(k), "n_terms_x": 10 * np.ones(k, dtype=np.int64)}
    return el, variances, norms


def test_iteration_costs_follow_f14_on_a_hand_built_input():
    """F14 on H4 with unit per-shot variances and normalizations, recomputed term by term.

    At iterate k of a 3-step L-BFGS record (measured loop, kappa = 1):
    - direct_mg: S = sum_e m_e v_e / (r^2 sum_e m_e A_e^2) shots of C(Psi_k) (A from
      routes.exact_residual at Psi_k, m from targets), i.e. the right state and circuit;
    - the walk's factorization bias is ||A[0.999 V] - A[V]||_m at Psi_k (ae_product's bias);
    - line search: S_E = V_E(Psi_{k+1}) / eps_ls^2 per candidate (V_E of the HF-chosen split from
      schemes.df_variance), T = S_E sum_c (C(Psi_k) + C_step(eps_c) + C_rot), C_rot = 150 (g = 6,
      b_r = 21), on max(Q(Psi_k), Q_step, Q_rot = 79) qubits;
    - the AE line-search option: sum_c N (C(Psi_k) + C_step + C_W + Q_A/2), N = 2 pi lambda / eps_ls,
      Q_A = max(Q(Psi_k), Q_step) + Q_walk - 2N + 1 (F11, F13);
    - C_iter = min over the non-GAW routes at kappa (and at kappa_practical) + T_LS;
    - seconds on one machine = ceil(S) t_shot(direct) + sum_c ceil(S_E) t_shot(c) when direct_mg
      is best, with the shot layouts of physical.surface_code_cost at 1e-5;
    - the classical first step makes iteration 0 free (zero Toffolis, no route, no campaign).
    """
    a, ir, rec, prep = _h4()
    el, variances, norms = _unit_inputs(a, ir)
    walk = _walk(a)
    split = "eigh_shift"
    m = targets.element_multiplicities(el, "spin_free", a.norb).astype(float)
    costs = pipeline.iteration_costs(a, rec, prep, walk, elements=el, variances=variances, normalizations=norms,
                                     energy_variance=split, readout_toffolis=stateprep.readout_rotation_toffolis(4),
                                     classical_first_step=False, params=P)
    assert [c.k for c in costs] == [0, 1, 2] and all(c.flags == () for c in costs)
    for c in costs:
        psi = rec.states[c.k]
        A = routes.exact_residual(a, psi, el, "spin_free")
        direct = next(r for r in c.residual if r.route == "direct_mg")
        shots = float(np.sum(m)) / (P.r ** 2 * float(np.sum(m * A ** 2)))
        assert direct.n_prep == pytest.approx(shots, rel=1e-12)
        assert direct.toffolis == pytest.approx(shots * prep.cumulative[c.k], rel=1e-12, abs=1e-9)
        assert c.prep_toffolis == prep.cumulative[c.k]
        assert c.residual_norm_m == pytest.approx(math.sqrt(float(np.sum(m * A ** 2))), rel=1e-12)
        model = routes.exact_residual(a.with_eri(walk.eri_model), psi, el, "spin_free")
        bias = math.sqrt(float(np.sum(m * (model - A) ** 2)))
        assert next(r for r in c.residual if r.route == "ae_product").params["bias_m"] == pytest.approx(bias, rel=1e-10)
        opts, shift = {"first": "eigh"}, True
        v_e = schemes.df_variance(a, rec.states[c.k + 1] / np.linalg.norm(rec.states[c.k + 1]), shift=shift,
                                  **opts).total
        steps = [stateprep.elementwise_step_cost(rec.directions[c.k], e, irreps=ir) for e in P.eps_candidates]
        s_e = v_e / P.eps_ls ** 2
        t_ls = s_e * sum(prep.cumulative[c.k] + s.toffolis + 150 for s in steps)
        assert c.line_search.toffolis == pytest.approx(t_ls, rel=1e-10)
        q_ls = max([int(prep.logical_qubits[c.k]), 79] + [s.logical_qubits for s in steps])
        assert c.line_search.logical_qubits == q_ls
        assert c.ls_flagged == sum("rounding bound above delta_step" in s.flags for s in steps)
        n_ae = 2.0 * math.pi * 4.935 / P.eps_ls                    # F11 with alpha = lambda, kappa = 1
        q_a = [max(int(prep.logical_qubits[c.k]), s.logical_qubits) + 163 - 2 * 4 + 1 for s in steps]
        t_ae = sum(n_ae * (prep.cumulative[c.k] + s.toffolis + 868 + 0.5 * q) for s, q in zip(steps, q_a))
        assert c.line_search_ae == pytest.approx(t_ae, rel=1e-12)
        pool = [r for r in c.residual if r.route != "gaw" and r.params.get("kappa", 1.0) == 1.0]
        pool_p = [r for r in c.residual if r.route != "gaw" and r.params.get("kappa", 2.0) == 2.0]
        best, best_p = min(pool, key=lambda r: r.toffolis), min(pool_p, key=lambda r: r.toffolis)
        assert (c.best, c.best_practical) == (best.route, best_p.route)
        assert c.toffolis == pytest.approx(best.toffolis + t_ls, rel=1e-10)
        assert c.toffolis_practical == pytest.approx(best_p.toffolis + t_ls, rel=1e-10)
        if c.best == "direct_mg":
            q = int(prep.logical_qubits[c.k])
            t = math.ceil(shots) * physical.surface_code_cost(q, int(prep.cumulative[c.k]), fail_budget=1e-5,
                                                               shot=True).seconds
            q_ls = c.line_search.logical_qubits
            t += sum(math.ceil(s_e) * physical.surface_code_cost(
                q_ls, int(prep.cumulative[c.k] + s.toffolis + 150), fail_budget=1e-5, shot=True).seconds for s in steps)
            assert c.seconds_one_machine == pytest.approx(t, rel=1e-12)
            assert c.shots == pytest.approx(shots + 11 * s_e, rel=1e-12)
    assert costs[0].best == "direct_mg" and costs[0].residual[0].toffolis == 0.0
    classical = pipeline.iteration_costs(a, rec, prep, walk, elements=el, variances=variances, normalizations=norms,
                                         energy_variance=split, readout_toffolis=150, k_max=1, params=P)
    c0 = classical[0]
    assert len(classical) == 2 and c0.best == "classical" and c0.toffolis == 0.0 and c0.residual == ()
    assert c0.line_search is None and c0.campaigns == () and c0.machines_7d == 0
    assert classical[1].toffolis == pytest.approx(costs[1].toffolis, rel=1e-14)


def test_iteration_costs_without_residual_and_validation():
    """Without residual inputs only the line search is costed (best "not costed", flag): C_iter equals the line
    search of the full run, a lower bound, strict from k = 1 (at HF the direct route costs no Toffolis);
    mixed or invalid inputs raise ValueError."""
    a, ir, rec, prep = _h4()
    el, variances, norms = _unit_inputs(a, ir)
    walk = _walk(a)
    full = pipeline.iteration_costs(a, rec, prep, walk, elements=el, variances=variances, normalizations=norms,
                                    energy_variance="eigh_shift", readout_toffolis=150, classical_first_step=False)
    bare = pipeline.iteration_costs(a, rec, prep, walk, elements=None, variances=None, normalizations=None,
                                    energy_variance="eigh_shift", readout_toffolis=150, classical_first_step=False)
    for f, b in zip(full, bare):
        assert b.best == "not costed" and b.flags == ("line search only",) and b.residual == ()
        assert b.toffolis == pytest.approx(f.line_search.toffolis, rel=1e-14)
        assert b.toffolis < f.toffolis or (b.k == 0 and b.toffolis == f.toffolis)
    kw = dict(energy_variance="eigh_shift", readout_toffolis=150)
    with pytest.raises(ValueError):
        pipeline.iteration_costs(a, rec, prep, walk, elements=el, variances=None, normalizations=norms, **kw)
    with pytest.raises(ValueError):
        pipeline.iteration_costs(a, rec, prep, walk, elements=None, variances=None, normalizations=None,
                                 energy_variance="pivoted", readout_toffolis=150)
    with pytest.raises(ValueError):
        pipeline.iteration_costs(a, rec, prep, walk, elements=None, variances=None, normalizations=None, k_max=3, **kw)
    short = stateprep.prep_cost(rec.directions[:2], rec.eps[:2], irreps=ir)
    with pytest.raises(ValueError):
        pipeline.iteration_costs(a, rec, short, walk, elements=None, variances=None, normalizations=None, **kw)


def _cost(seconds: float) -> physical.PhysicalCost:
    return physical.PhysicalCost(physical_qubits=1, seconds=seconds, distance=7, factory="x", failure=0.0,
                                 us_per_toffoli=1.0)


def _ae(runs: tuple[float, ...]) -> pipeline.Campaign:
    """An amplitude-estimation campaign of the given run lengths (s), longest first."""
    runs = tuple(sorted(runs, reverse=True))
    return pipeline.Campaign("ae", len(runs), _cost(runs[0]), sum(runs), identical=False, runs_s=runs)


def test_lpt_makespan_against_exhaustive_schedules():
    """lpt_makespan is the LPT schedule (longest run first onto the least-loaded machine), compared with a
    direct list implementation; on random instances of 2-7 runs and 1-3 machines it lies between the optimal
    makespan (exhaustive search over all assignments) and Graham's 1969 LPT bound (4/3 - 1/(3M)) OPT, and above
    max(W/M, t_max); one machine gives the sum, M >= n the longest run. The classic tight case 3,3,2,2,2 on 2
    machines gives 7 against the optimum 6."""
    rng = np.random.default_rng(3)
    for _ in range(300):
        runs = list(rng.integers(1, 30, size=int(rng.integers(2, 8))).astype(float))
        for m in (1, 2, 3):
            loads = [0.0] * m
            for t in sorted(runs, reverse=True):
                loads[loads.index(min(loads))] += t
            got = pipeline.lpt_makespan(runs, m)
            assert got == max(loads)
            opt = min(max(sum(t for t, a in zip(runs, assign) if a == j) for j in range(m))
                      for assign in itertools.product(range(m), repeat=len(runs)))
            assert max(sum(runs) / m, max(runs), opt) <= got <= (4.0 / 3.0 - 1.0 / (3 * m)) * opt + 1e-12
    assert pipeline.lpt_makespan([3, 3, 2, 2, 2], 2) == 7.0
    assert pipeline.lpt_makespan([2.0, 5.0], 1) == 7.0 and pipeline.lpt_makespan([2.0, 5.0], 5) == 5.0
    assert pipeline.lpt_makespan([], 3) == 0.0
    with pytest.raises(ValueError):
        pipeline.lpt_makespan([1.0], 0)


def test_campaign_schedules_and_machines():
    """F19 schedules against closed forms: identical runs take t ceil(n/M); unequal (AE) runs follow LPT (their sum
    on one machine, the longest run on as many machines as runs); campaigns run one after another.
    machines_for_campaigns is the smallest M meeting the deadline (checked by scanning M), 0 for nothing to run,
    None when the longest runs of the campaigns (the critical path) exceed the deadline, a campaign has no
    layout or infinitely many runs; for one campaign of identical runs it equals physical.machines_for."""
    shots = pipeline.Campaign("shots", 10.5, _cost(2.0), 22.0)
    ae = _ae((30.0, 30.0, 20.0, 20.0))
    assert shots.seconds(1) == 22.0 and shots.seconds(4) == 2.0 * math.ceil(10.5 / 4)
    assert ae.seconds(1) == 100.0 and ae.seconds(2) == 50.0 and ae.seconds(3) == 40.0 and ae.seconds(4) == 30.0
    assert ae.longest_s == 30.0
    for deadline in (130.0, 60.0, 57.0, 52.0, 40.0, 32.0):
        m = pipeline.machines_for_campaigns([shots, ae], deadline)
        feasible = [k for k in range(1, 200) if shots.seconds(k) + ae.seconds(k) <= deadline]
        assert m == (feasible[0] if feasible else None)
    assert pipeline.machines_for_campaigns([]) == 0
    assert pipeline.machines_for_campaigns([pipeline.Campaign("x", 0.0, _cost(1e9), 0.0)]) == 0
    assert pipeline.machines_for_campaigns([pipeline.Campaign("x", 3.0, _cost(10.0), 30.0)], 9.0) is None
    assert pipeline.machines_for_campaigns([pipeline.Campaign("x", math.inf, None, math.inf)]) is None
    assert pipeline.machines_for_campaigns([pipeline.Campaign("x", 3.0, None, math.inf)]) is None
    assert pipeline.machines_for_campaigns([shots, ae], 31.0) is None       # 2 + 30 > 31 on any M
    for n, t, deadline in ((10.5, 2.0, 7.0), (1e6, 1.25, 7 * 86400.0), (3.0, 5.0, 4.0), (7.0, 3.0, 9.0)):
        one = pipeline.Campaign("shots", n, _cost(t), t * math.ceil(n))
        assert pipeline.machines_for_campaigns([one], deadline) == physical.machines_for(n, _cost(t), deadline)
    with pytest.raises(ValueError):
        pipeline.Campaign("ae", 3, _cost(1.0), 3.0, identical=False, runs_s=(1.0, 1.0))


def test_machines_for_several_unequal_campaigns():
    """Two AE campaigns of 10 runs of 3 s, deadline 10 s: 3 ceil(10/M) each, so 10 machines (one run each); the
    search bracket must use the sum of the campaigns' longest runs (6 s), not the largest (3 s), which would
    bracket at 9 machines and report None. Random mixes of shot and AE campaigns agree with a scan over M."""
    ae = _ae((3.0,) * 10)
    assert pipeline.machines_for_campaigns([ae, ae], 10.0) == 10
    assert pipeline.machines_for_campaigns([ae, ae], 6.0) == 10
    assert pipeline.machines_for_campaigns([ae, ae], 5.9) is None
    rng = np.random.default_rng(5)
    for _ in range(200):
        cs = [pipeline.Campaign("shots", float(n), _cost(t), t * math.ceil(n))
              for n, t in zip(rng.uniform(0.5, 40, 2), rng.uniform(0.1, 2.0, 2))]
        cs += [_ae(tuple(rng.uniform(0.1, 5.0, int(rng.integers(1, 12))))) for _ in range(int(rng.integers(1, 3)))]
        deadline = float(rng.uniform(2.0, 60.0))
        feasible = [k for k in range(1, 400) if sum(c.seconds(k) for c in cs) <= deadline]
        assert pipeline.machines_for_campaigns(cs, deadline) == (feasible[0] if feasible else None)


def test_loop_totals_and_first_heisenberg_iterate():
    """loop_totals sums C_iter(j) and its parts over j < k_end (hand-built iterations, kappa 1 and 2), and
    first_heisenberg_iterate returns the first k where an AE route at kappa beats both sampling routes."""
    def route(name, t, kappa=None):
        params = {} if kappa is None else {"kappa": kappa}
        return RouteCost(name, 1.0, t, t, t, 1.0, 10, params)

    def it(k, direct, aux, ae1, ae2, ls):
        res = (route("direct_mg", direct), route("aux_mg", aux), route("ae_x", ae1, 1.0), route("ae_x", ae2, 2.0))
        best, best_p = min(res[:3], key=lambda r: r.toffolis), min(res[:2] + res[3:], key=lambda r: r.toffolis)
        return pipeline.IterationCost(k=k, residual=res, best=best.route, line_search=route("line_search", ls),
                                      toffolis=best.toffolis + ls, shots=0.0, seconds_one_machine=0.0, machines_7d=1,
                                      best_practical=best_p.route, toffolis_practical=best_p.toffolis + ls,
                                      residual_toffolis=best.toffolis, residual_toffolis_practical=best_p.toffolis)

    costs = [it(0, 1.0, 5.0, 3.0, 6.0, 10.0), it(1, 4.0, 9.0, 3.0, 6.0, 20.0), it(2, 9.0, 8.0, 5.0, 7.0, 30.0)]
    tot = pipeline.loop_totals(costs, 2, c_qpe=10.0)
    assert tot["k_end"] == 2 and tot["residual_direct"] == 5.0 and tot["residual_best"] == 4.0
    assert tot["residual_best_practical"] == 5.0 and tot["line_search"] == 30.0
    assert tot["total"] == 34.0 and tot["total_over_qpe"] == 3.4 and tot["total_practical_over_qpe"] == 3.5
    assert pipeline.loop_totals(costs, None, 10.0)["total"] == 34.0 + 35.0
    assert pipeline.first_heisenberg_iterate(costs, 1.0) == (1, "ae_x")
    assert pipeline.first_heisenberg_iterate(costs, 2.0) == (2, "ae_x")


def test_run_system_argument_checks():
    """Invalid arguments raise ValueError before any computation; by default the stretched systems and h8 run
    the L-BFGS families only [design], the core systems all three."""
    assert pipeline.default_optimizers("h6s") == ("lbfgs", "lbfgs_so")
    assert pipeline.default_optimizers("h8", ("sd",)) == ()
    assert pipeline.default_optimizers("n2") == ("lbfgs", "lbfgs_so", "sd")
    with pytest.raises(ValueError):
        pipeline.run_system("h4", optimizers=("bfgs",))
    with pytest.raises(ValueError):
        pipeline.run_system("h4", max_iter=0)
    with pytest.raises(ValueError):
        pipeline.run_system("h5")


def _small_report() -> pipeline.SystemReport:
    """run_system("h4") with the spin-free L-BFGS family, 6 steps and the r-scan of T5 (cached)."""
    pytest.importorskip("openfermion")
    if "report" not in _CACHE:
        _CACHE["report"] = pipeline.run_system("h4", optimizers=("lbfgs",), max_iter=6, r_scan=True)
    return _CACHE["report"]


def test_to_json_is_strict_json_and_round_trips():
    """to_json gives a dict that json encodes with allow_nan=False and decodes to an equal dict; numbers survive
    the round trip (C_QPE, energies, iteration Toffolis), nan becomes null and inf "inf", and the CI
    vectors and model integrals are left out."""
    rep = _small_report()
    js = pipeline.to_json(rep)
    back = json.loads(json.dumps(js, allow_nan=False))
    assert back == js
    assert back["qpe"]["overall"]["toffolis"] == rep.qpe["overall"].toffolis
    rec = rep.trajectories["lbfgs"]["exact"]
    assert back["trajectories"]["lbfgs"]["exact"]["energies"] == [float(x) for x in rec.energies]
    assert "states" not in back["trajectories"]["lbfgs"]["exact"]
    assert all("eri_model" not in e for e in back["encodings"])
    iters = rep.iterations["lbfgs"]["measured"]
    assert [x["toffolis"] for x in back["iterations"]["lbfgs"]["measured"]] == [c.toffolis for c in iters]
    assert pipeline._jsonable({"a": math.nan, "b": (math.inf, -math.inf), "c": np.arange(2)}) == \
        {"a": None, "b": ["inf", "-inf"], "c": [0, 1]}
    with pytest.raises(TypeError):
        pipeline._jsonable(object())


def test_markdown_tables_structure():
    """T1-T9 are present, in order, each with a table; k* is marked on its row (k = 3 for H4 L-BFGS) in T2, T3 and
    T9; T1 marks the selected encoding; every table row has the header's number of cells."""
    rep = _small_report()
    md = pipeline.markdown_tables([rep])
    heads = [line for line in md.splitlines() if line.startswith("## T")]
    assert [h.split(".")[0] for h in heads] == [f"## T{i}" for i in range(1, 10)]
    assert "(selected)" in md
    assert rep.settings["k_star"]["lbfgs"]["exact"] == 3
    sections = md.split("## T")[1:]
    for n in (2, 3, 9):
        rows = [line for line in sections[n - 1].splitlines() if line.startswith("| h4 | lbfgs |")]
        assert any(line.split(" | ")[2] == "3*" for line in rows)
    lines = md.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("|---"):
            n_cells = lines[i - 1].count(" | ") + 1
            j = i + 1
            while j < len(lines) and lines[j].startswith("| "):
                assert lines[j].count(" | ") + 1 == n_cells, lines[j]
                j += 1
    with pytest.raises(ValueError):
        pipeline.markdown_tables([])


def test_iteration_costs_flag_campaigns_without_a_layout():
    """With a shot budget of 1e-40 no surface-code layout exists (F18: the d = 43 storage error alone exceeds it):
    every iteration carries the flag, its one-machine seconds are inf and machines_7d is None, and so is the
    loop's machine count; the Toffolis are unchanged. With the default budgets nothing is flagged."""
    a, ir, rec, prep = _h4()
    el, variances, norms = _unit_inputs(a, ir)
    kw = dict(elements=el, variances=variances, normalizations=norms, energy_variance="eigh_shift",
              readout_toffolis=150, classical_first_step=False)
    tight = pipeline.iteration_costs(a, rec, prep, _walk(a), params=FTParameters(p_fail_shot=1e-40), **kw)
    plain = pipeline.iteration_costs(a, rec, prep, _walk(a), **kw)
    for t, c in zip(tight, plain):
        assert t.flags == ("no layout within the failure budget",) and c.flags == ()
        assert math.isinf(t.seconds_one_machine) and t.machines_7d is None and math.isfinite(c.seconds_one_machine)
        assert t.toffolis == c.toffolis
    assert pipeline.loop_totals(tight, None, 1.0)["machines_7d"] is None


def test_iteration_coverage_rules():
    """The last costed iterate is k* + 4, at least k_v - 1 (F17 needs C_iter(j), j < k_v), and K - 1 when k* is
    never reached (all iterates). The thresholds are moved with FTParameters so that each branch binds on one
    14-step H4 record, whose E_k - E_0 come from first_iterate_below (k* = 3; within 1e-5 Ha first at k = 9)."""
    a = symmetry.benchmark_system("h4")
    rec = trajectory.exact_trajectory(a, system="h4", irreps=symmetry.orbital_irreps(a), n_roots=3, max_iter=14)
    assert rec.n_steps == 14 and trajectory.first_iterate_below(rec, 1.6e-3) == 3
    assert pipeline._k_last(rec, FTParameters()) == 3 + 4
    never = FTParameters(target=1e-15, e_tol_energy=1e-15)
    assert trajectory.first_iterate_below(rec, 1e-15) is None
    assert pipeline._k_last(rec, never) == rec.n_steps - 1
    late = FTParameters(e_tol_energy=1e-5)
    k_v = trajectory.first_iterate_below(rec, 1e-5)
    assert k_v == 9 and k_v - 1 > 3 + 4
    assert pipeline._k_last(rec, late) == k_v - 1


def test_handoff_rows_use_the_loops_own_iteration_costs():
    """C_loop(k) of every hand-off row is the cumulative sum of its own loop's C_iter (F15): the measured loop on
    the exact record, the classical first step on the second-order record with C_iter(0) = 0, the free loop
    0; p_k is the overlap of the loop's record (any-irrep tables alike)."""
    rep = _small_report()
    its = rep.iterations["lbfgs"]
    for loop in ("measured", "classical_first_step", "free"):
        rec = rep.trajectories["lbfgs"]["second_order" if loop == "classical_first_step" else "exact"]
        for e1 in ("same_irrep", "any_irrep"):
            rows = rep.handoff["lbfgs"][loop][e1]
            if loop == "free":
                expected = np.zeros(len(rows))
            else:
                c = [it.toffolis for it in its[loop]]
                if loop == "classical_first_step":
                    c[0] = 0.0
                assert len(rows) == len(c) + 1
                expected = np.concatenate([[0.0], np.cumsum(c)])
            np.testing.assert_allclose([r.c_loop for r in rows], expected, rtol=1e-12, atol=0.0)
            np.testing.assert_allclose([r.p for r in rows], np.minimum(rec.overlaps[:len(rows)], 1.0), rtol=1e-14)


def test_cqe_only_wall_clock_and_the_qpe_run():
    """T8: the hand-off QPE run is surface_code_cost at p_fail_qpe; HF + QPE takes 1/p_0 runs; one final-energy
    shot runs C(Psi_kv) + C_rot on max(Q(Psi_kv), Q_rot) qubits at p_fail_shot, and the CQE-only wall clock is the
    loop's iterations before k_v plus ceil(S_fin) such shots."""
    rep = _small_report()
    qpe = rep.qpe["overall"]
    run = physical.surface_code_cost(qpe.logical_qubits, qpe.toffolis, fail_budget=rep.params.p_fail_qpe)
    assert rep.physical["qpe_run"] == run
    q_rot = stateprep.readout_rotation_qubits(4)
    assert q_rot == 79 and rep.settings["readout_qubits"] == q_rot
    for loop, first in (("measured", "exact"), ("classical_first_step", "second_order")):
        e = rep.cqe_only["lbfgs"][loop]
        rec = rep.trajectories["lbfgs"][first]
        assert e["hf_qpe_seconds"] == pytest.approx(run.seconds / rec.overlaps[0], rel=1e-12)
        k_v = e["cost"].k_v
        pc = rep.prep["lbfgs"][first]["elementwise"]
        shot = physical.surface_code_cost(max(int(pc.logical_qubits[k_v]), q_rot),
                                          int(pc.cumulative[k_v]) + rep.settings["readout_toffolis"],
                                          fail_budget=rep.params.p_fail_shot, shot=True)
        assert e["final_shot"] == shot and int(pc.logical_qubits[k_v]) < q_rot
        loop_s = sum(it.seconds_one_machine for it in rep.iterations["lbfgs"][loop][:k_v])
        assert e["seconds"] == pytest.approx(loop_s + math.ceil(e["cost"].final_shots) * shot.seconds, rel=1e-12)


def _rows(md: str, n: int) -> list[list[str]]:
    section = md.split(f"## T{n}.")[1].split("\n## T")[0]
    return [line[2:-2].split(" | ") for line in section.splitlines() if line.startswith("| h4 | ")]


def test_table_cells_show_the_report_values():
    """Cells of T4, T6, T7, T8 and T9 against the report objects (loop_totals, HandoffRow, the records,
    CQEOnlyCost, StepCost): T4 total / C_QPE at kappa 1 and 2; T6 p_lb same / any irrep; T7 p_1 after eps* and
    after eps_2; the T9 rounding bound sum_j A_j pi / 2^b_j with A_j the additions of step j (F3), which bounds
    the replay difference. Lower bounds are marked: a record whose k* is not reached gets '>=' on every T4 sum,
    and an F17 without k_v prints '-' for its costs."""
    rep = _small_report()
    md = pipeline.markdown_tables([rep])
    c_qpe = float(rep.qpe["overall"].toffolis)
    f = pipeline._fmt
    t4 = _rows(md, 4)
    assert len(t4) == 2
    for cells in t4:
        first = "second_order" if cells[2] == "classical_first_step" else "exact"
        tot = pipeline.loop_totals(rep.iterations["lbfgs"][cells[2]], rep.settings["k_star"]["lbfgs"][first], c_qpe)
        assert cells[9] == f"{f(tot['total_over_qpe'], '.3g')} / {f(tot['total_practical_over_qpe'], '.3g')}"
        assert not any(c.startswith(">=") for c in cells)
    t6 = [c for c in _rows(md, 6) if len(c) == 12]
    assert len(t6) == 10
    for cells in t6:
        loop, k = ("measured" if cells[2] == "all" else cells[2]), int(cells[3])
        same, other = (rep.handoff["lbfgs"][loop][e1][k] for e1 in ("same_irrep", "any_irrep"))
        assert cells[5] == f"{f(same.p_lb, '.3f')} / {f(other.p_lb, '.3f')}"
    (t7,) = _rows(md, 7)
    recs = rep.trajectories["lbfgs"]
    assert t7[5:7] == [f(recs["exact"].overlaps[1], ".5f"), f(recs["second_order"].overlaps[1], ".5f")]
    t8 = _rows(md, 8)
    for cells in t8:
        c = rep.cqe_only["lbfgs"][cells[2]]["cost"]
        assert cells[3] == str(c.k_v) and cells[5:7] == [pipeline._sci(c.shots), pipeline._sci(c.final_shots)]
    rnd = rep.validation["replay"]["lbfgs"]["rounding"]
    steps = rep.prep["lbfgs"]["exact"]["elementwise"].steps
    bound = np.concatenate([[0.0], np.cumsum([s.counts.n_add * math.pi / 2.0 ** s.bits for s in steps])])
    np.testing.assert_allclose([r["bound"] for r in rnd], bound[:len(rnd)], rtol=1e-12)
    assert all(r["difference"] <= r["bound"] + 1e-12 for r in rnd)
    # k* and k_v not reached (as on n2s): partial sums are lower bounds, F17 has no cost.
    settings = dict(rep.settings, k_star={"lbfgs": {"exact": None, "second_order": None}})
    unreached = {loop: dict(v, cost=dataclasses.replace(v["cost"], k_v=None, shots=math.inf, toffolis=math.inf,
                                                         ratio_to_hf_qpe=math.inf))
                 for loop, v in rep.cqe_only["lbfgs"].items()}
    md2 = pipeline.markdown_tables([dataclasses.replace(rep, settings=settings, cqe_only={"lbfgs": unreached})])
    for cells in _rows(md2, 4):
        n = len(rep.iterations["lbfgs"][cells[2]])
        assert cells[3] == f"- ({n} costed)" and all(c.startswith(">= ") for c in cells[4:13])
        assert cells[9].count(">= ") == 2
    for cells in _rows(md2, 8):
        assert cells[3] == "-" and cells[5:10] == ["-"] * 5 and cells[11] == "-" and cells[10] != "-"


def test_r_scan_and_crossover_rows():
    """T5 on h4 (r-scan switched on): every r-scan row uses the line search of its own iterate, C_iter/C_QPE =
    (T_best + T_LS)/C_QPE, and at the default r = 0.3 the scan reproduces the iteration's best route and residual
    Toffolis; r = 0.1 costs more than r = 0.3; the crossover route is the cheapest AE route at kappa = 1 of the
    iteration at k*."""
    rep = _small_report()
    costs = rep.iterations["lbfgs"]["measured"]
    rows = rep.crossovers["lbfgs"]["r_scan"]
    assert sorted({r["k"] for r in rows}) == [2, 3] and sorted({r["r"] for r in rows}) == [0.1, 0.3, 1.0]
    c_qpe = float(rep.qpe["overall"].toffolis)
    for r in rows:
        t_ls = costs[r["k"]].line_search.toffolis
        assert r["line_search_toffolis"] == t_ls
        assert r["c_iter_over_qpe"] == pytest.approx((r["best_toffolis"] + t_ls) / c_qpe, rel=1e-12)
        if r["r"] == 0.3:
            assert r["best"] == costs[r["k"]].best
            assert r["best_toffolis"] == pytest.approx(costs[r["k"]].residual_toffolis, rel=1e-12)
    by = {(r["k"], r["r"]): r["best_toffolis"] for r in rows}
    assert by[(2, 0.1)] > by[(2, 0.3)] > by[(2, 1.0)]
    cr = rep.crossovers["lbfgs"]["crossover"]
    ae1 = [c for c in costs[3].residual if c.route in ("ae_product", "ae_x", "ae_aux") and c.params["kappa"] == 1.0]
    assert cr["k"] == 3 and cr["route"] == min(ae1, key=lambda c: c.toffolis).route


@pytest.mark.slow
def test_energy_split_and_variances_follow_the_rules():
    """F13: the DF split of V_E is the lowest at HF, then fixed; on h6a the lowest at HF (cholesky_shift) differs
    from the lowest at Psi_1 (eigh_shift), so the rule is observable (h4 picks eigh_shift at both). The per-shot
    variances come from Psi_1 of the first family in FAMILIES order, whatever the order requested. References:
    residual_inputs.energy_variances at the two states, element_variances at Psi_1 of the spin-free record."""
    pytest.importorskip("openfermion")
    rep = pipeline.run_system("h6a", optimizers=("lbfgs_so", "lbfgs"), max_iter=1)
    a = symmetry.benchmark_system("h6a")
    rec = rep.trajectories["lbfgs"]["exact"]
    at_hf, at_1 = residual_inputs.energy_variances(a, rec.states[0]), residual_inputs.energy_variances(a, rec.states[1])
    assert rep.settings["energy_split"] == at_hf["best"] == "cholesky_shift"
    assert at_1["best"] == "eigh_shift"
    el, _ = residual_inputs.allowed_elements(a, "spin_free", symmetry.orbital_irreps(a))
    ref = residual_inputs.element_variances(a, rec.states[1], el, "spin_free")
    assert rep.settings["variance_m"] == ref["variance_m"]
    so = residual_inputs.element_variances(a, rep.trajectories["lbfgs_so"]["exact"].states[1], el, "spin_free")
    assert so["variance_m"] != ref["variance_m"]


# ------------------------------------------------------------------------- golden record
def golden_view(rep: pipeline.SystemReport) -> dict:
    """The quantities of the golden record: C_QPE (overall, DF, SF), p_k of every record, k* and k_v, the
    hand-off optimum per loop and rule, the first-step gains, the best route and C_iter per iterate, the loop
    totals and the CQE-only Toffolis (THC is left out: its fits are non-convex)."""
    s = pipeline._jsonable(pipeline.report_summary(rep))
    s["c_qpe_kind"].pop("thc", None)
    for fam, f in s["families"].items():
        f["c_iter"] = {loop: [c.toffolis for c in costs] for loop, costs in rep.iterations[fam].items()}
        f["cqe_only"] = {loop: pipeline._jsonable(v["cost"].toffolis) for loop, v in rep.cqe_only[fam].items()}
    return s


def _compare(ref, got, path: str = "") -> None:
    if isinstance(ref, dict):
        assert set(ref) == set(got), path
        for k in ref:
            _compare(ref[k], got[k], f"{path}/{k}")
    elif isinstance(ref, list):
        assert len(ref) == len(got), path
        for i, (r, g) in enumerate(zip(ref, got)):
            _compare(r, g, f"{path}[{i}]")
    elif isinstance(ref, float):
        assert got == pytest.approx(ref, rel=1e-6, abs=1e-12), path
    else:
        assert ref == got, path


@pytest.mark.slow
def test_h4_reproduces_the_golden_record():
    """run_system("h4") (lbfgs, lbfgs_so, sd) reproduces tests/data/ft_h4_golden.json to rtol 1e-6.

    The record was written once from the final code and checked by hand: C_QPE of DF (6,728,736) is Appendix
    A's, SF (3,969,408) the qpe module's table; p_k of the exact L-BFGS record are Appendix A's (0.9455, 0.9802,
    0.9960, 0.9981); k* = 3 (L-BFGS, both metrics) and 5 (SD); eps_2/eps* = 1.0159; the hand-off verdicts
    (HF for the measured loop, Psi_1 for the classical first step) follow from T_E of the defining formula.
    Re-record with ``python tests/test_ft_pipeline.py`` after a deliberate change.
    """
    pytest.importorskip("openfermion")
    with open(GOLDEN) as fh:
        ref = json.load(fh)
    _compare(ref, golden_view(pipeline.run_system("h4")))


if __name__ == "__main__":
    with open(GOLDEN, "w") as fh:
        json.dump(golden_view(pipeline.run_system("h4")), fh, indent=1, allow_nan=False)
        fh.write("\n")
