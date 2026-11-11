"""Tests of the step-rule cost layer (``thc_engine_cqe.ft.steprules``; F27, F28, F30, F31) and of the record conversion.

References: M3's ``pipeline.iteration_costs`` on the same exact record (the line-search rule is charged as M3's
11 energies at eps_ls, F14) and M3's T4 loop total, ``residual.compare_routes``/``best_route`` at each state
(the unit rule), ``residual.energy_sampling``, ``residual.energy_ae`` and ``residual.line_search_cost`` (F28),
``ft.trajectory.exact_trajectory`` (the conversion, bit for bit with the measured first step), closed forms of
F27, the scalar ``steprules.parabola_step`` (the vectorized Monte Carlo), an independent recomputation of the
F31 plan, and the R1 iteration counts [design].
"""

import json
import math

import numpy as np
import pytest

from thc_engine_cqe.diagnostics.thc_floors import ExactReference
from thc_engine_cqe.fci_ops import FCISpace, Hamiltonian
from thc_engine_cqe.ft import pipeline
from thc_engine_cqe.ft import residual_inputs as ri
from thc_engine_cqe.ft import scaling
from thc_engine_cqe.ft import steprules as fsr
from thc_engine_cqe.ft import symmetry as sy
from thc_engine_cqe.ft.handoff import handoff_table, optimal_handoff
from thc_engine_cqe.ft.primitives import FTParameters
from thc_engine_cqe.ft.qpe import scan_encodings, select_qpe
from thc_engine_cqe.ft.residual import (PrepParams, ResidualProblem, WalkParams, best_route, compare_routes,
                                        energy_ae, energy_sampling, line_search_cost, sampling_direct)
from thc_engine_cqe.ft.stateprep import (candidate_step_costs, prep_cost, readout_rotation_qubits,
                                         readout_rotation_toffolis)
from thc_engine_cqe.ft.trajectory import exact_trajectory, first_iterate_below, propagate
from thc_engine_cqe.steprules import StepEvent, StepRule, parabola_step, run_step_rule

M3 = FTParameters()
_HANDOFF_FLOATS = ("p", "c_prep", "c_loop", "runs_expected", "cost_expected", "p_lb", "p0_break_even",
                   "cost_delta_oracle")


@pytest.fixture(scope="module")
def h4m3():
    """h4 with M3's inputs (pipeline.run_system): the selected QPE, residual inputs at iterate 1 of the exact
    spin-free record, the DF split at HF, the readout rotation, the spectrum and both spin-free records."""
    act = sy.benchmark_system("h4")
    ir = sy.orbital_irreps(act)
    spec = scaling.spectrum(act, ir)
    kw = dict(irreps=ir, max_iter=8, system="h4", roots=spec.roots, n_roots=spec.n_roots)
    exact = exact_trajectory(act, **kw)
    second = exact_trajectory(act, first_step="second_order", **kw)
    qpe = select_qpe(scan_encodings(act, reference=ExactReference.build(act)))
    el, _ = ri.allowed_elements(act, "spin_free", ir)
    inputs = dict(elements=el, variances=ri.element_variances(act, exact.states[1], el, "spin_free"),
                  normalizations=ri.element_normalizations(act, el, "spin_free"),
                  energy_variance=ri.energy_variances(act, exact.states[0])["best"],
                  readout_toffolis=readout_rotation_toffolis(act.norb))
    return dict(act=act, ir=ir, spec=spec, exact=exact, second=second, qpe=qpe, walk=qpe.encoding, inputs=inputs)


def _routes_at(act, walk, psi, prep_k, inputs, r):
    """Every F14 route at the state psi, recomputed from ft.residual_inputs and ft.residual."""
    el = inputs["elements"]
    problem = ri.build_problem(act, psi, el, "spin_free", variances=inputs["variances"],
                               normalizations=inputs["normalizations"], deltas=M3.aux_deltas)
    bias = ri.factorization_bias(act, walk.model(act), psi, el, "spin_free")
    return compare_routes(problem, prep_k, walk.walk_params(bias), params=FTParameters(r=r))


# ------------------------------------------------------------------------- F30
def test_m3_costing_reproduced(h4m3):
    """The line-search rule (charged as M3's 11 candidates at eps_ls) on h4's spin-free classical-first-step exact
    record equals pipeline.iteration_costs bit for bit (toffolis ==), and C_loop(k* = 3) = 2416.996 C_QPE, M3's T4
    value (results/ft_h4.json: 2416.995810433715); with the measured first step it is M3's measured loop."""
    act, ir, rec, walk = h4m3["act"], h4m3["ir"], h4m3["second"], h4m3["walk"]
    run = run_step_rule(act, StepRule("line_search"), max_iter=rec.n_steps, metric="spin_free", irreps=ir)
    prep = prep_cost(rec.directions, rec.eps, irreps=ir)
    m3 = pipeline.iteration_costs(act, rec, prep, walk, classical_first_step=True, k_max=6, **h4m3["inputs"])
    mine = fsr.step_rule_iteration_costs(act, rec, run, prep, walk, r=M3.r, k_max=6, **h4m3["inputs"])
    assert len(mine) == len(m3) == 7
    assert mine[0].best == "classical" and mine[0].toffolis == 0.0 and mine[0].line_search is None
    for a, b in zip(mine[1:], m3[1:]):
        assert a.toffolis == b.toffolis and a.toffolis_practical == b.toffolis_practical
        assert (a.best, a.best_practical, a.residual_toffolis, a.residual_norm_m, a.prep_toffolis) == \
            (b.best, b.best_practical, b.residual_toffolis, b.residual_norm_m, b.prep_toffolis)
        assert a.line_search.toffolis == b.line_search.toffolis and a.line_search.n_prep == b.line_search.n_prep
        assert a.line_search.logical_qubits == b.line_search.logical_qubits
        assert (a.shots, a.seconds_one_machine, a.machines_7d, a.line_search_ae, a.ls_flagged) == \
            (b.shots, b.seconds_one_machine, b.machines_7d, b.line_search_ae, b.ls_flagged)
        assert a.line_search.params["n_energies"] == 11 and a.line_search.params["eps"] == M3.eps_ls
        assert len(a.campaigns) == len(b.campaigns) == 12
    c_qpe = float(h4m3["qpe"].toffolis)
    k_star = first_iterate_below(rec, M3.target)
    assert k_star == 3
    total = pipeline.loop_totals(mine, k_star, c_qpe)
    assert total["total_over_qpe"] == pytest.approx(2416.995810433715, rel=1e-12)
    assert total == pipeline.loop_totals(m3, k_star, c_qpe)
    # the measured loop: first="line_search" charges A_0 and the line search at HF, as M3's measured loop
    measured = run_step_rule(act, StepRule("line_search", first="line_search"), max_iter=8, metric="spin_free",
                             irreps=ir)
    ex = h4m3["exact"]
    prep_ex = prep_cost(ex.directions, ex.eps, irreps=ir)
    m3m = pipeline.iteration_costs(act, ex, prep_ex, walk, classical_first_step=False, k_max=3, **h4m3["inputs"])
    mine_m = fsr.step_rule_iteration_costs(act, ex, measured, prep_ex, walk, r=M3.r, k_max=3, **h4m3["inputs"])
    assert [a.toffolis for a in mine_m] == [b.toffolis for b in m3m] and mine_m[0].toffolis > 0.0
    assert [a.machines_7d for a in mine_m] == [b.machines_7d for b in m3m]


def test_costs_at_several_r(h4m3):
    """The F30 costing at several r (one residual problem per state for every r) equals M3's
    pipeline.iteration_costs at each r on the line-search rule (11 energies at eps_ls; the residual routes in strict
    JSON, every cost and count; F30 names its energies "energy_sampling", M3 "line_search") and the single-r call
    for the candidates rule (strict JSON of every IterationCost)."""
    act, ir, rec, walk = h4m3["act"], h4m3["ir"], h4m3["second"], h4m3["walk"]
    run = run_step_rule(act, StepRule("line_search"), max_iter=rec.n_steps, metric="spin_free", irreps=ir)
    prep = prep_cost(rec.directions, rec.eps, irreps=ir)
    rs = (0.3, 1.0, 3.0)
    many = fsr._step_rule_costs(act, rec, run, prep, walk, r_values=rs, k_max=3, **h4m3["inputs"])
    assert list(many) == list(rs)
    for r in rs:
        m3 = pipeline.iteration_costs(act, rec, prep, walk, classical_first_step=True, k_max=3,
                                      params=FTParameters(r=r), **h4m3["inputs"])
        for a, b in zip(many[r], m3):
            assert scaling.to_json(a.residual) == scaling.to_json(b.residual)
            assert (a.toffolis, a.toffolis_practical, a.best, a.best_practical, a.residual_toffolis, a.shots,
                    a.seconds_one_machine, a.machines_7d, a.residual_norm_m, a.prep_toffolis, len(a.campaigns)) == \
                (b.toffolis, b.toffolis_practical, b.best, b.best_practical, b.residual_toffolis, b.shots,
                 b.seconds_one_machine, b.machines_7d, b.residual_norm_m, b.prep_toffolis, len(b.campaigns))
            if a.k > 0:
                assert (a.line_search.toffolis, a.line_search.n_prep, a.line_search_ae) == \
                    (b.line_search.toffolis, b.line_search.n_prep, b.line_search_ae)
    cand = run_step_rule(act, StepRule("candidates"), max_iter=3, metric="spin_orbital", irreps=ir, keep_states=True)
    crec = fsr.to_trajectory_record(cand, system="h4", active=act, irreps=ir, spectrum=h4m3["spec"])
    cprep = prep_cost(crec.directions, crec.eps, irreps=ir)
    many_c = fsr._step_rule_costs(act, crec, cand, cprep, walk, r_values=(1.0, 0.3), **h4m3["inputs"])
    for r in (1.0, 0.3):
        one = fsr.step_rule_iteration_costs(act, crec, cand, cprep, walk, r=r, **h4m3["inputs"])
        assert scaling.to_json(many_c[r]) == scaling.to_json(one)
    assert many_c[0.3][1].toffolis > many_c[1.0][1].toffolis
    with pytest.raises(ValueError):
        fsr._step_rule_costs(act, rec, run, prep, walk, r_values=(), **h4m3["inputs"])


@pytest.mark.parametrize("r", [0.3, 1.0])
def test_unit_cost_is_the_residual(h4m3, r):
    """Unit steps (R1, spin-orbital, classical first step): C_iter(0) = 0 and C_iter(k) = the best route of
    compare_routes at Psi_k exactly (no energies: line_search None, the AE option 0); the residual campaign is the
    iteration's only campaign; pipeline.loop_totals and machines_for_campaigns accept the costs."""
    act, ir, walk, spec = h4m3["act"], h4m3["ir"], h4m3["walk"], h4m3["spec"]
    run = run_step_rule(act, StepRule("unit"), max_iter=6, irreps=ir, keep_states=True)
    rec = fsr.to_trajectory_record(run, system="h4", active=act, irreps=ir, spectrum=spec)
    prep = prep_cost(rec.directions, rec.eps, irreps=ir)
    costs = fsr.step_rule_iteration_costs(act, rec, run, prep, walk, r=r, **h4m3["inputs"])
    assert len(costs) == rec.n_steps and costs[0].toffolis == 0.0 and costs[0].best == "classical"
    for k in range(1, rec.n_steps):
        prep_k = PrepParams(prep.cumulative[k], prep.logical_qubits[k])
        routes = _routes_at(act, walk, rec.states[k], prep_k, h4m3["inputs"], r)
        best = best_route(routes, kappa=1.0)
        c = costs[k]
        assert c.toffolis == best.toffolis == c.residual_toffolis and c.best == best.route
        assert c.toffolis_practical == best_route(routes, kappa=2.0).toffolis
        assert c.line_search is None and c.line_search_ae == 0.0 and c.ls_flagged == 0
        assert len(c.campaigns) == 1 and c.campaigns[0].label == best.route
    tot = pipeline.loop_totals(costs, None, float(h4m3["qpe"].toffolis))
    assert tot["total"] == pytest.approx(sum(c.toffolis for c in costs), rel=1e-15) and tot["line_search"] == 0
    assert pipeline.machines_for_campaigns([x for c in costs for x in c.campaigns]) >= 1


def test_energy_rules_are_charged_by_their_ledger(h4m3):
    """parab3: the three energies at {0, t, 2t} along d_k at sigma = 0.47|dE_m| cost F28 at V_E of Psi_{k+1}; the
    iteration adds them to the residual; one sampling campaign per energy."""
    act, ir, walk, spec = h4m3["act"], h4m3["ir"], h4m3["walk"], h4m3["spec"]
    inputs = h4m3["inputs"]
    run = run_step_rule(act, StepRule("parab3"), max_iter=4, irreps=ir, keep_states=True)
    rec = fsr.to_trajectory_record(run, system="h4", active=act, irreps=ir, spectrum=spec)
    prep = prep_cost(rec.directions, rec.eps, irreps=ir)
    costs = fsr.step_rule_iteration_costs(act, rec, run, prep, walk, r=1.0, **inputs)
    q_rot = readout_rotation_qubits(act.norb)
    for k in range(1, rec.n_steps):
        ev = [e for e in run.iteration_events(k) if e.kind == "energy"]
        sigma = ev[0].sigma
        assert sigma == pytest.approx(0.47 * abs(run.de_model[k]), rel=1e-14)
        prep_k = PrepParams(prep.cumulative[k], prep.logical_qubits[k])
        cands = candidate_step_costs(rec.directions[k], [e.eps for e in ev], irreps=ir)
        v_e = ri.energy_variances(act, rec.states[k + 1])[inputs["energy_variance"]]
        q = max([prep_k.logical_qubits, q_rot] + [c.logical_qubits for c in cands])
        want = sum(energy_sampling(v_e, prep_k.toffolis + c.toffolis + inputs["readout_toffolis"], q,
                                   eps=sigma).toffolis for c in cands)
        c = costs[k]
        assert c.line_search.toffolis == pytest.approx(want, rel=1e-13)
        assert c.toffolis == pytest.approx(c.residual_toffolis + want, rel=1e-13)
        assert [x.label for x in c.campaigns] == [c.best] + ["energy"] * 3


# ------------------------------------------------------------------------- F27
def test_energy_precision_values():
    """F27 times |dE_m| (any sign): 0.894 (parab3_linear, L = 0.1; sqrt(8L) exactly), 0.47 (parab3, fixed),
    0.354 (armijo, z = 2, c1 = 1e-4: 0.3535 to four digits), 0.167 (parab2: the Monte-Carlo inverse; 0.16 [design]
    gives L = 0.088); the tables interpolate log-linearly, exactly at their nodes."""
    de = -2.5e-3
    assert fsr.energy_precision("parab3_linear", de) == pytest.approx(math.sqrt(0.8) * 2.5e-3, rel=1e-15)
    assert fsr.energy_precision("parab3_linear", de) / 2.5e-3 == pytest.approx(0.894, abs=5e-4)
    assert fsr.energy_precision("parab3", de) == 0.47 * 2.5e-3
    assert fsr.energy_precision("armijo", de) == pytest.approx((1 - 2e-4) / (2 * math.sqrt(2)) * 2.5e-3, rel=1e-15)
    assert fsr.energy_precision("armijo", de) / 2.5e-3 == pytest.approx(0.354, abs=6e-4)   # 0.35348 (c1 -> 0: 0.35355)
    assert fsr.energy_precision("armijo", de, z=3.0, c1=0.1) == pytest.approx(0.8 / (3 * math.sqrt(2)) * 2.5e-3)
    assert fsr.energy_precision("parab2", de) / 2.5e-3 == pytest.approx(0.167, rel=1e-12)
    assert fsr.energy_precision("parab2", de) / 2.5e-3 == pytest.approx(0.16, rel=0.05)
    assert fsr.energy_precision("parab3", 0.0) == 0.0
    for kind, table in fsr.PRECISION_TABLE.items():
        ls, ss = zip(*table)
        assert np.all(np.diff(ls) > 0) and np.all(np.diff(ss) > 0)
        for loss, ratio in table:
            assert fsr.energy_precision(kind, 1.0, loss=loss) == pytest.approx(ratio, rel=1e-14)
        mid = math.sqrt(ls[2] * ls[3])
        assert fsr.energy_precision(kind, 1.0, loss=mid) == pytest.approx(math.sqrt(ss[2] * ss[3]), rel=1e-12)
    assert fsr.energy_precision("parab3", 1.0, loss=0.05) == pytest.approx(0.364, rel=1e-12)
    for kw in ({"kind": "secant"}, {"kind": "parab3", "loss": 0.0}, {"kind": "parab3", "loss": 0.5},
               {"kind": "parab2", "loss": 1e-3}, {"kind": "armijo", "c1": 0.5}, {"kind": "armijo", "z": 0.0}):
        kw = dict(kw)
        kind = kw.pop("kind")
        with pytest.raises(ValueError):
            fsr.energy_precision(kind, 1e-3, **kw)
    with pytest.raises(ValueError):
        fsr.energy_precision("parab3", math.nan)


def test_fit_losses_use_parabola_step():
    """The vectorized 3-point rule of the Monte Carlo is steprules.parabola_step (interpolation only) draw for
    draw; at sigma -> 0 the losses vanish and the linearized ratios hold (sigma^2/8 for parab3, 2 sigma^2 for
    parab2, in units of dE*^2)."""
    rng = np.random.default_rng(5)
    e = np.array([0.0, -0.5, 0.0])[:, None] + 0.4 * rng.standard_normal((3, 3000))
    vec = fsr._parab3_steps(*e)
    assert np.array_equal(vec, [parabola_step(a, b, c, 1.0)[0] for a, b, c in e.T])
    small = fsr.fit_losses("parab3", [0.0, 0.02], n_draws=200_000)
    assert small[0] == 0.0 and small[1] == pytest.approx(0.02 ** 2 / 8, rel=0.02)
    small2 = fsr.fit_losses("parab2", [0.0, 0.01], n_draws=200_000)
    assert small2[0] == 0.0 and small2[1] == pytest.approx(2 * 0.01 ** 2, rel=0.02)
    with pytest.raises(ValueError):
        fsr.fit_losses("armijo", [0.1])


@pytest.mark.slow
def test_precision_table_monte_carlo():
    """build_precision_table (4e5 draws) reproduces PRECISION_TABLE within 1% (the parab3 entry at L = 0.1 is the
    fixed 0.47 against the Monte-Carlo 0.468); the loss at sigma_P3 = 0.47 is 0.10 +- 0.01, at the linearized
    0.894 it is 0.26, and parab2 at 0.16 [design] loses 0.088."""
    table = fsr.build_precision_table()
    for kind, rows in fsr.PRECISION_TABLE.items():
        for (loss, pinned), (loss_mc, sig) in zip(rows, table[kind]):
            assert loss == loss_mc and sig == pytest.approx(pinned, rel=0.01)
    l3 = fsr.fit_losses("parab3", [0.47, 0.894])
    assert l3[0] == pytest.approx(0.10, abs=0.01) and l3[1] == pytest.approx(0.264, abs=0.01)
    assert fsr.fit_losses("parab2", [0.16])[0] == pytest.approx(0.088, abs=0.003)


# ------------------------------------------------------------------------- F28
def test_energy_costs_follow_f28(h4m3):
    """Sampled energies: T = (V_E/sigma^2) sum_i (C(Psi_k) + C_step,i + C_rot) = the sum of energy_sampling per
    energy, on max(Q(Psi_k), Q_step, Q_rot); at sigma = eps_ls with M3's candidates it is line_search_cost bit
    for bit. AE: the sum of energy_ae per energy (C(Psi_k) + C_step,i, max(Q(Psi_k), Q_step,i)). rho_E against the
    direct residual is n_E (V_E/V_m) r^2 ||A||_m^2/sigma^2 (C + C_step + C_rot)/C."""
    prep = PrepParams(1000.0, 20)
    steps, qs = [0, 150, 300], [8, 30, 31]
    walk = WalkParams(lam=5.0, toffoli_per_step=400, logical_qubits=50, kind="sf")
    v_e, sigma, c_rot, q_rot = 0.6, 2e-4, 150, 26
    out = fsr.energy_costs(3, sigma, prep, steps, energy_variance=v_e, readout_toffolis=c_rot, readout_qubits=q_rot,
                           walk=walk, kappa=1.0, step_qubits=qs, norb=4)
    s, a = out["sampling"], out["ae"]
    shots = v_e / sigma ** 2
    assert s.toffolis == pytest.approx(shots * sum(1000.0 + x + c_rot for x in steps), rel=1e-15)
    assert s.toffolis == pytest.approx(sum(energy_sampling(v_e, 1000.0 + x + c_rot, 31, eps=sigma).toffolis
                                           for x in steps), rel=1e-14)
    assert (s.n_prep, s.logical_qubits, s.coherent_toffolis) == (pytest.approx(3 * shots), 31, 1450.0)
    assert s.params["shots_per_energy"] == pytest.approx(shots) and s.params["n_energies"] == 3
    ae = [energy_ae(walk, PrepParams(1000.0 + x, max(20, q)), eps=sigma, kappa=1.0, norb=4) for x, q in zip(steps, qs)]
    assert a.toffolis == sum(c.toffolis for c in ae) and a.run_toffolis == tuple(c.toffolis for c in ae)
    assert a.n_prep == pytest.approx(3 * 2 * math.pi * 5.0 / sigma) and a.n_runs == 3
    assert a.logical_qubits == max(c.logical_qubits for c in ae)
    low = fsr.energy_costs(3, sigma, prep, steps, energy_variance=v_e, readout_toffolis=c_rot, readout_qubits=q_rot)
    assert low["ae"] is None and "qubits_lower_bound" in low["sampling"].flags
    assert low["sampling"].logical_qubits == 26 and low["sampling"].toffolis == s.toffolis
    # M3's line search along an h4 direction
    cands = candidate_step_costs(h4m3["second"].directions[2], M3.eps_candidates, irreps=h4m3["ir"])
    q = max([20, q_rot] + [c.logical_qubits for c in cands])
    ref = line_search_cost(v_e, prep, [c.toffolis for c in cands], c_rot, logical_qubits=q)
    mine = fsr.energy_costs(11, M3.eps_ls, prep, [c.toffolis for c in cands], energy_variance=v_e,
                            readout_toffolis=c_rot, readout_qubits=q_rot,
                            step_qubits=[c.logical_qubits for c in cands])["sampling"]
    assert (mine.toffolis, mine.n_prep, mine.coherent_toffolis, mine.logical_qubits) == \
        (ref.toffolis, ref.n_prep, ref.coherent_toffolis, ref.logical_qubits)
    # rho_E against the direct route
    m, a_e, v = np.array([1.0, 2.0, 2.0]), np.array([0.01, -0.02, 0.005]), np.array([3.0, 1.0, 2.0])
    problem = ResidualProblem("spin_free", np.array([[0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]]), m, a_e, v,
                              np.ones(3), np.ones(3), np.ones(3), np.ones(3), None, norb=2)
    r = 0.5
    direct = sampling_direct(problem, prep, params=FTParameters(r=r))
    rho = 3 * (v_e / float(np.sum(m * v))) * r ** 2 * problem.norm_a_m ** 2 / sigma ** 2 \
        * (1000.0 + np.mean(steps) + c_rot) / 1000.0
    assert s.toffolis / direct.toffolis == pytest.approx(rho, rel=1e-12)
    for args, kw in (((0, sigma, prep, []), {}), ((3, 0.0, prep, steps), {}), ((3, sigma, prep, [0, 1]), {}),
                     ((3, sigma, prep, [0, -1, 2]), {}), ((3, sigma, prep, steps), {"step_qubits": [1, 2]}),
                     ((3, sigma, (1000.0, 20), steps), {})):
        with pytest.raises(ValueError):
            fsr.energy_costs(*args, energy_variance=v_e, readout_toffolis=c_rot, readout_qubits=q_rot, **kw)


# ------------------------------------------------------------------------- F31
def test_planned_gate(h4m3):
    """The F31 gate on h4: every logged plan equals an independent recomputation (C(Psi_k) from prep_cost of the
    steps taken, the three energies at {0, 1, 2} along d_k at 0.47|dE_m| with the planned V_E, the best residual
    route at the run's own Psi_k at r), the decision is T_E <= theta T_res and the run's flags follow it;
    theta = inf always fits, a tiny theta never does."""
    act, ir, walk = h4m3["act"], h4m3["ir"], h4m3["walk"]
    inputs = dict(h4m3["inputs"])
    split = inputs.pop("energy_variance")
    v_plan = ri.energy_variances(act, FCISpace.of(act).hf())[split]
    for theta in (1.0, math.inf, 1e-9):
        gate = fsr.planned_energy_gate(act, walk, irreps=ir, energy_variance=v_plan, r=1.0, theta=theta, **inputs)
        run = run_step_rule(act, StepRule("hybrid", theta=theta), max_iter=5, irreps=ir, use_energies=gate,
                            keep_states=True)
        assert [x["k"] for x in gate.log] == list(range(1, run.n_steps))
        for entry in gate.log:
            k = entry["k"]
            prep = prep_cost(run.directions[:k], run.eps[:k], irreps=ir)
            prep_k = PrepParams(prep.cumulative[k], prep.logical_qubits[k])
            sigma = 0.47 * abs(run.de_model[k])
            cands = candidate_step_costs(run.directions[k], (0.0, 1.0, 2.0), irreps=ir)
            t_e = fsr.energy_costs(3, sigma, prep_k, [c.toffolis for c in cands], energy_variance=v_plan,
                                   readout_toffolis=inputs["readout_toffolis"],
                                   readout_qubits=readout_rotation_qubits(act.norb),
                                   step_qubits=[c.logical_qubits for c in cands])["sampling"].toffolis
            t_res = best_route(_routes_at(act, walk, run.states[k], prep_k, inputs, 1.0), kappa=1.0).toffolis
            assert entry["sigma"] == sigma and entry["t_energy"] == t_e and entry["t_residual"] == t_res
            assert entry["use"] == (t_e <= theta * t_res)
            assert ("hybrid:parab3" in run.flags[k]) == entry["use"]
            plan = gate.plan(run.de_model[k], run.states[k], prep_k, run.directions[k])
            assert (plan["t_energy"], plan["t_residual"]) == (t_e, t_res)
        if theta == math.inf:
            assert all(x["use"] for x in gate.log)
        if theta == 1e-9:
            assert not any(x["use"] for x in gate.log)
    with pytest.raises(ValueError, match="history"):
        gate(2, -1e-3, None, ((np.zeros((4,) * 4),), (0.1,)))


# ------------------------------------------------------------------------- records
def _assert_same_record(a, b):
    for f in ("energies", "variances", "residual_norms", "overlaps", "eps"):
        assert np.array_equal(getattr(a, f), getattr(b, f)), f
    assert all(np.array_equal(x, y) for x, y in zip(a.states, b.states)) and len(a.states) == len(b.states)
    assert all(np.array_equal(x, y) for x, y in zip(a.directions, b.directions))
    for f in ("system", "optimizer", "metric", "norb", "nelec", "e0", "e1", "e1_any", "ground_irrep", "irrep_impurity",
              "flags", "first_step"):
        assert getattr(a, f) == getattr(b, f), f
    assert np.array_equal(a.irreps, b.irreps)


@pytest.mark.parametrize("metric", ["spin_free", "spin_orbital"])
def test_record_conversion_is_m3s_measured_record(h4m3, metric):
    """With the measured first step the line-search rule's converted record is exact_trajectory's (first_step
    "exact") bit for bit: energies, variances, ||A||, overlaps, eps, states, directions, E1, flags."""
    act, ir, spec = h4m3["act"], h4m3["ir"], h4m3["spec"]
    run = run_step_rule(act, StepRule("line_search", first="line_search"), max_iter=8, metric=metric, irreps=ir,
                        keep_states=True)
    rec = fsr.to_trajectory_record(run, system="h4", active=act, irreps=ir, spectrum=spec)
    ref = exact_trajectory(act, irreps=ir, max_iter=8, system="h4", metric=metric, roots=spec.roots,
                           n_roots=spec.n_roots)
    _assert_same_record(rec, ref)


def test_record_conversion_feeds_handoff(h4m3):
    """The converted classical-first-step record feeds handoff_table like exact_trajectory(first_step=
    "second_order"): with one cost list, every float field agrees to 1e-9 relative; the two differences, E_k - E_0
    and saving_vs_hf = C_QPE/p_0 - (C(Psi_k) + C_QPE)/p_k, to 1e-10 Ha and to 1e-9 of C_QPE/p_0 (the records differ
    by the precision of the line searches, the parity tolerance of the line-search rule); the integer and boolean
    fields and the optimal rows exactly."""
    act, ir, spec, walk = h4m3["act"], h4m3["ir"], h4m3["spec"], h4m3["walk"]
    ref = h4m3["second"]
    run = run_step_rule(act, StepRule("line_search"), max_iter=8, metric="spin_free", irreps=ir, keep_states=True)
    rec = fsr.to_trajectory_record(run, system="h4", active=act, irreps=ir, spectrum=spec)
    assert rec.first_step == "second_order" and rec.e1 == ref.e1 and rec.e1_any == ref.e1_any
    assert np.max(np.abs(rec.energies - ref.energies)) <= 1e-10
    prep_a, prep_b = prep_cost(rec.directions, rec.eps, irreps=ir), prep_cost(ref.directions, ref.eps, irreps=ir)
    assert np.array_equal(prep_a.cumulative, prep_b.cumulative)
    costs = [c.toffolis for c in pipeline.iteration_costs(act, ref, prep_b, walk, **h4m3["inputs"])]
    for loop in ("classical_first_step", "free"):
        rows_a = handoff_table(rec, prep_a.cumulative, costs, h4m3["qpe"], loop=loop)
        rows_b = handoff_table(ref, prep_b.cumulative, costs, h4m3["qpe"], loop=loop)
        assert len(rows_a) == len(rows_b) == 9
        for a, b in zip(rows_a, rows_b):
            for f in _HANDOFF_FLOATS:
                assert getattr(a, f) == pytest.approx(getattr(b, f), rel=1e-9), (a.k, f)
            assert a.e_err == pytest.approx(b.e_err, abs=1e-10)
            assert a.saving_vs_hf == pytest.approx(b.saving_vs_hf, abs=1e-9 * rows_b[0].cost_expected)
            assert (a.k, a.runs_delta, a.runs_delta_oracle, a.certifiable, a.flags) == \
                (b.k, b.runs_delta, b.runs_delta_oracle, b.certifiable, b.flags)
        for rule in ("expected", "expected_oracle"):
            assert optimal_handoff(rows_a, rule).k == optimal_handoff(rows_b, rule).k


def test_conversion_and_cost_validation(h4m3):
    act, ir, spec, walk = h4m3["act"], h4m3["ir"], h4m3["spec"], h4m3["walk"]
    run = run_step_rule(act, StepRule("unit"), max_iter=3, irreps=ir)
    with pytest.raises(ValueError, match="keep_states"):
        fsr.to_trajectory_record(run, system="h4", active=act, irreps=ir, spectrum=spec)
    run = run_step_rule(act, StepRule("unit"), max_iter=3, irreps=ir, keep_states=True)
    other = scaling.spectrum(sy.benchmark_system("lih"), sy.orbital_irreps(sy.benchmark_system("lih")), n_roots=2)
    with pytest.raises(ValueError, match="spectrum"):
        fsr.to_trajectory_record(run, system="h4", active=act, irreps=ir, spectrum=other)
    with pytest.raises(ValueError, match="irreps"):
        fsr.to_trajectory_record(run, system="h4", active=act, irreps=ir[:2], spectrum=spec)
    rec = fsr.to_trajectory_record(run, system="h4", active=act, irreps=ir, spectrum=spec)
    prep = prep_cost(rec.directions, rec.eps, irreps=ir)
    kw = dict(r=1.0, **h4m3["inputs"])
    fsr.step_rule_iteration_costs(act, rec, run, prep, walk, k_max=0, **kw)
    with pytest.raises(ValueError, match="k_max"):
        fsr.step_rule_iteration_costs(act, rec, run, prep, walk, k_max=3, **kw)
    measured = run_step_rule(act, StepRule("line_search", first="line_search"), max_iter=8, metric="spin_free",
                             irreps=ir)
    second = h4m3["second"]
    with pytest.raises(ValueError, match="first step"):
        fsr.step_rule_iteration_costs(act, second, measured, prep_cost(second.directions, second.eps, irreps=ir),
                                      walk, **kw)
    short = run_step_rule(act, StepRule("unit"), max_iter=2, irreps=ir)
    with pytest.raises(ValueError, match="steps"):
        fsr.step_rule_iteration_costs(act, rec, short, prep, walk, **kw)
    with pytest.raises(ValueError, match="energy_variance"):
        fsr.step_rule_iteration_costs(act, rec, run, prep, walk, **dict(kw, energy_variance="dense"))
    with pytest.raises(ValueError, match="required"):
        fsr.step_rule_iteration_costs(act, rec, run, prep, walk, **dict(kw, variances=None))
    with pytest.raises(ValueError, match="r must be positive"):
        fsr.step_rule_iteration_costs(act, rec, run, prep, walk, **dict(kw, r=0.0))
    run.events.append(StepEvent(0, "residual", 0.0, None, True))
    with pytest.raises(ValueError, match="classical first step"):
        fsr.step_rule_iteration_costs(act, rec, run, prep, walk, **kw)


# ------------------------------------------------------------------------- U5 block
def _capture_runs(monkeypatch) -> list:
    """The StepRuleResult of every run_step_rule call that ft.steprules makes (in order)."""
    runs = []
    original = fsr.run_step_rule

    def spy(*args, **kwargs):
        runs.append(original(*args, **kwargs))
        return runs[-1]

    monkeypatch.setattr(fsr, "run_step_rule", spy)
    return runs


def test_run_step_rule_system(h4m3, monkeypatch):
    """U5's block on h4 (spin-free, r = 0.3, 8 steps): the line-search rule's C_loop(k* = 3) is M3's T4 value
    2416.996 C_QPE with 11 charged energies per measured iteration; R1's C_iter(k) is its residual alone; R3's gate
    (planned at r) never opens on h4 and every logged decision is T_E^plan <= theta T_res^plan, so its costs are
    R1's; the horizon is k* + 2 and k_ref = k*; sigma_A and sigma_P3 are F27's closed forms; rho_E = T_E/T_res,direct
    at k = 1 equals F28's closed form (V_E(Psi_2)/sigma^2) sum_i (C(Psi_1) + C_step(1; eps_i) + C_rot)/((V_m/(r
    ||A_1||_m)^2) C(Psi_1)) for the 3-point fit of R1 (sigma_P3, eps_i in {0, 1, 2}) and for the Armijo pair of the
    line search (sigma_A, eps_i in {0, eps_1}, eps_1 != 1); the stencils requested at k = 1 and k* - 1 = 2 are
    (0, eps_k) and (0, t, 2t) along d_k; the block is strict JSON."""
    act, ir = h4m3["act"], h4m3["ir"]
    runs, stencils = [], []
    original_run, original_steps = fsr.run_step_rule, fsr.candidate_step_costs

    def run_spy(*args, **kwargs):
        runs.append(None)                            # the stencils below are those of the run started last
        runs[-1] = original_run(*args, **kwargs)
        return runs[-1]

    def step_spy(d, eps, **kw):
        stencils.append((len(runs) - 1, np.array(d, copy=True), tuple(float(x) for x in eps)))
        return original_steps(d, eps, **kw)

    monkeypatch.setattr(fsr, "run_step_rule", run_spy)

    monkeypatch.setattr(fsr, "candidate_step_costs", step_spy)
    rules = [StepRule("line_search"), StepRule("unit"), StepRule("hybrid")]
    out = fsr.run_step_rule_system("h4", rules, metrics=("spin_free",), r_values=(0.3,), max_iter=8)
    monkeypatch.setattr(fsr, "candidate_step_costs", original_steps)
    json.dumps(scaling.to_json(out), allow_nan=False)
    assert out["system"]["c_qpe"] == h4m3["qpe"].toffolis and out["system"]["variance_state"] == "iterate1"
    ls, unit, hyb = out["runs"]
    assert [r["rule"]["name"] for r in out["runs"]] == ["line_search", "unit", "hybrid"] and hyb["r_plan"] == 0.3
    assert ls["k_star"] == unit["k_star"] == 3 and ls["cost_horizon"] == unit["cost_horizon"] == 5
    assert all((r["k_ref"], r["k_ref_kind"]) == (3, "k_star") for r in out["runs"])
    c_ls, c_unit, c_hyb = ls["costs"]["0.3"], unit["costs"]["0.3"], hyb["costs"]["0.3"]
    assert c_ls["c_loop_k_star_over_qpe"] == pytest.approx(2416.995810433715, rel=1e-6)
    assert all(c["c_loop_k_ref_over_qpe"] == c["c_loop_k_star_over_qpe"] for c in (c_ls, c_unit, c_hyb))
    assert ls["extra_measurements"][1:] == [11] * (ls["n_steps"] - 1) and ls["extra_measurements"][0] == 0
    assert c_unit["c_iter"] == c_unit["residual_toffolis"] and c_unit["energy_toffolis"] == [0.0] * 6
    assert c_unit["c_loop"] == pytest.approx([0.0] + list(np.cumsum(c_unit["c_iter"])), rel=1e-15)
    assert all(not g["use"] and g["t_energy"] > g["t_residual"] for g in hyb["gate_log"])
    assert [g["k"] for g in hyb["gate_log"]] == list(range(1, hyb["n_steps"]))
    assert all(g["t_energy_charged"] == (0.0 if g["k"] <= hyb["cost_horizon"] else None) for g in hyb["gate_log"])
    assert c_hyb["c_iter"] == c_unit["c_iter"] and hyb["e_err"] == unit["e_err"]
    pre = c_unit["precision"][0]
    assert pre["k"] == 1 and [p["k"] for p in c_unit["precision"]] == [1, 2]
    assert pre["sigma_armijo"] == pytest.approx((1.0 - 2e-4) * abs(pre["de_model"]) / (2.0 * math.sqrt(2.0)),
                                                rel=1e-15)
    assert pre["sigma_parab3"] == pytest.approx(0.47 * abs(pre["de_model"]), rel=1e-15)
    run_ls, run_unit = runs[0], runs[1]
    assert run_unit.rule.name == "unit" and pre["de_model"] == run_unit.de_model[1]
    el, var, nrm = (h4m3["inputs"][key] for key in ("elements", "variances", "normalizations"))
    c_rot = h4m3["inputs"]["readout_toffolis"]

    def rho_closed(run, k, sigma, eps_points):
        rec = fsr.to_trajectory_record(run, system="h4", active=act, irreps=ir, spectrum=h4m3["spec"])
        c_k = float(prep_cost(rec.directions, rec.eps, irreps=ir).cumulative[k])
        steps = [c.toffolis for c in candidate_step_costs(rec.directions[k], eps_points, irreps=ir)]
        v_e = ri.energy_variances(act, rec.states[k + 1])[h4m3["inputs"]["energy_variance"]]
        problem = ri.build_problem(act, rec.states[k], el, "spin_free", variances=var, normalizations=nrm,
                                   deltas=M3.aux_deltas)
        v_m, norm = float(np.sum(problem.m * problem.v_direct)), problem.norm_a_m
        return (v_e / sigma ** 2) * sum(c_k + st + c_rot for st in steps) / ((v_m / (0.3 * norm) ** 2) * c_k)

    rec_unit = fsr.to_trajectory_record(run_unit, system="h4", active=act, irreps=ir, spectrum=h4m3["spec"])
    assert unit["e_err"] == pytest.approx(list(rec_unit.errors), abs=1e-15)
    assert pre["rho_parab3"] == pytest.approx(rho_closed(run_unit, 1, pre["sigma_parab3"], (0.0, 1.0, 2.0)),
                                              rel=1e-12)
    pre_ls = c_ls["precision"][0]
    eps1 = float(run_ls.eps[1])
    assert run_ls.rule.name == "line_search" and pre_ls["k"] == 1 and abs(eps1 - 1.0) > 0.1
    sigma_a = (1.0 - 2e-4) * abs(run_ls.de_model[1]) / (2.0 * math.sqrt(2.0))
    assert pre_ls["sigma_armijo"] == pytest.approx(sigma_a, rel=1e-15)
    assert pre_ls["rho_armijo"] == pytest.approx(rho_closed(run_ls, 1, sigma_a, (0.0, eps1)), rel=1e-12)
    # the F28 stencils of the precision entries: (0, eps_k) and (0, 1, 2) along d_k at k = 1 and k* - 1 = 2
    for i, run in enumerate((run_ls, run_unit)):
        for k in (1, 2):
            asked = sorted(eps for j, d, eps in stencils
                           if j == i and len(eps) <= 3 and np.array_equal(d, run.directions[k]))
            want = sorted([(0.0, float(run.eps[k])), (0.0, 1.0, 2.0)])
            assert asked == want, (run.rule.name, k, asked)
    with pytest.raises(ValueError):
        fsr.run_step_rule_system("h4", [], metrics=("spin_free",))
    with pytest.raises(ValueError):
        fsr.run_step_rule_system("h4", rules, metrics=("lowdin",))


def test_u5_serializes_iterations_without_a_machine_count(monkeypatch):
    """An iteration that no machine count finishes within 7 days (pipeline.machines_for_campaigns returns None: a
    coherent run beyond the deadline, or a campaign without a layout) is stored as null in U5's block, which stays
    strict JSON; the classical k = 0 runs nothing on a quantum machine (0)."""
    seen = []

    def no_machine(campaigns, *args, **kwargs):
        seen.append(len(campaigns))
        return None

    monkeypatch.setattr(fsr, "machines_for_campaigns", no_machine)
    out = fsr.run_step_rule_system("h4", [StepRule("unit")], metrics=("spin_free",), r_values=(0.3,), max_iter=4)
    run = out["runs"][0]
    machines = run["costs"]["0.3"]["machines_7d"]
    assert len(machines) == run["cost_horizon"] + 1 >= 3 and len(seen) == run["cost_horizon"]
    assert machines[0] == 0 and all(m is None for m in machines[1:])
    back = json.loads(json.dumps(scaling.to_json(out), allow_nan=False))
    assert back["runs"][0]["costs"]["0.3"]["machines_7d"] == [0] + [None] * (len(machines) - 1)


@pytest.mark.slow
def test_u5_null_machine_counts_on_a_stretched_chain(monkeypatch):
    """h6_r4.0, spin-orbital R1 at r = 0.3, costed to k = 17: the late iterations' cheapest route runs single coherent
    AE runs beyond 7 days, so machines_for_campaigns finds no machine count; U5 stores null exactly where the
    campaigns' critical path (the sum of their longest single runs) exceeds 7 days, the count elsewhere."""
    calls = []
    original = fsr.machines_for_campaigns

    def spy(campaigns, *args, **kwargs):
        calls.append((tuple(campaigns), original(campaigns, *args, **kwargs)))
        return calls[-1][1]

    monkeypatch.setattr(fsr, "machines_for_campaigns", spy)
    out = fsr.run_step_rule_system("h6_r4.0", [StepRule("unit")], metrics=("spin_orbital",), r_values=(0.3,),
                                   max_iter=40, k_cost=17)
    machines = out["runs"][0]["costs"]["0.3"]["machines_7d"]
    assert len(machines) == 18 and len(calls) == 17 and machines[17] is None
    week = 7 * 24 * 3600.0
    for k, (campaigns, m) in enumerate(calls, start=1):
        active = [c for c in campaigns if c.n_runs > 0]
        impossible = any(c.per_run is None for c in active) or sum(c.longest_s for c in active) > week
        assert (machines[k] is None) == impossible and machines[k] == m, k
    json.dumps(scaling.to_json(out), allow_nan=False)


@pytest.fixture(scope="module")
def h4r4(tmp_path_factory):
    """h4_r4.0 (a small-gap chain: R3's gate opens there, and k* = 6 lies beyond a 4-step run): the closed-loop
    inputs of U5 and one input cache for its blocks."""
    from thc_engine_cqe.ft import closedloop as cl

    cache = scaling.Cache(tmp_path_factory.mktemp("u5_cache"))
    return dict(cache=cache, inputs=cl.system_inputs("h4_r4.0", cache=cache))


def test_u5_gate_plans_at_the_planned_step(h4r4, monkeypatch):
    """F31 in U5's exact emulation (h4_r4.0, spin-orbital, R3 planned at r = 0.3 and at r = 1, 4 steps): every logged
    plan equals an independent recomputation from the run's own states: sigma_P3 = 0.47|dE_m|, C(Psi_k) from prep_cost
    of the steps taken, C_step(k; 0, 1, 2) along d_k, V_E of the DF split at the planned step exp(D_k) Psi_k, and the
    best residual route at Psi_k at r; the decision is T_E^plan <= T_res^plan: the fit at k = 1, 2 and the unit step at
    k = 3 for r = 0.3, and at r = 1 the fit at k = 1, where V_E(HF) would plan T_E above T_res and keep the unit step;
    next to every plan the charged T_E (U5's energy column) and, where the fit ran, V_E(Psi_{k+1})."""
    runs = _capture_runs(monkeypatch)
    inp = h4r4["inputs"]
    out = fsr.run_step_rule_system("h4_r4.0", [StepRule("hybrid")], metrics=("spin_orbital",), r_values=(0.3, 1.0),
                                   max_iter=4, cache=h4r4["cache"])
    json.dumps(scaling.to_json(out), allow_nan=False)
    act, ir, split, walk = inp.active, inp.irreps, inp.energy_split, inp.qpe.encoding
    ham = Hamiltonian(act)
    v_hf = ri.energy_variances(act, FCISpace.of(act).hf())[split]
    routes_in = dict(elements=inp.elements, variances=inp.variances, normalizations=inp.normalizations)
    q_rot = readout_rotation_qubits(act.norb)
    hf_would_differ = []
    assert [r["r_plan"] for r in out["runs"]] == [0.3, 1.0] and len(runs) == 2
    for run_json, res in zip(out["runs"], runs):
        r = run_json["r_plan"]
        log = run_json["gate_log"]
        assert [g["k"] for g in log] == list(range(1, res.n_steps)) == [1, 2, 3]
        energy_col = run_json["costs"][f"{r:g}"]["energy_toffolis"]
        for g in log:
            k = g["k"]
            prep = prep_cost(res.directions[:k], res.eps[:k], irreps=ir)
            prep_k = PrepParams(prep.cumulative[k], prep.logical_qubits[k])
            cands = candidate_step_costs(res.directions[k], (0.0, 1.0, 2.0), irreps=ir)
            sigma = 0.47 * abs(res.de_model[k])
            v_plan = ri.energy_variances(act, propagate(ham, res.directions[k], res.states[k], 1.0))[split]

            def t_e(v):
                return fsr.energy_costs(3, sigma, prep_k, [c.toffolis for c in cands], energy_variance=v,
                                        readout_toffolis=inp.readout_toffolis, readout_qubits=q_rot,
                                        step_qubits=[c.logical_qubits for c in cands])["sampling"].toffolis

            t_res = best_route(_routes_at(act, walk, res.states[k], prep_k, routes_in, r), kappa=1.0).toffolis
            assert g["sigma"] == sigma and g["energy_variance"] == v_plan
            assert g["t_energy"] == t_e(v_plan) and g["t_residual"] == t_res
            assert g["use"] == (t_e(v_plan) <= t_res) == ("hybrid:parab3" in res.flags[k])
            hf_would_differ.append((r, k, (t_e(v_hf) <= t_res) != g["use"]))
            assert g["t_energy_charged"] == energy_col[k] and (g["t_energy_charged"] > 0.0) == g["use"]
            if g["use"]:
                v_next = ri.energy_variances(act, res.states[k + 1])[split]
                assert g["energy_variance_charged"] == pytest.approx(v_next, rel=1e-12)
            else:
                assert g["energy_variance_charged"] is None
        decisions = [g["use"] for g in log]
        assert decisions == ([True, True, False] if r == 0.3 else [True, False, False])
    assert (1.0, 1, True) in hf_would_differ


def test_u5_shared_residual_problems_change_nothing(h4r4, monkeypatch):
    """The residual problems that R3's gate and the costing share (by the bytes of the state) leave U5's block
    unchanged, and fewer problems are built (h4_r4.0, spin-orbital, R3 at r = 0.3, 5 steps)."""
    builds = []
    original_build = ri.build_problem

    def counting(*args, **kwargs):
        builds.append(1)
        return original_build(*args, **kwargs)

    monkeypatch.setattr(ri, "build_problem", counting)
    kw = dict(metrics=("spin_orbital",), r_values=(0.3,), max_iter=5, cache=h4r4["cache"])
    shared = fsr.run_step_rule_system("h4_r4.0", [StepRule("hybrid")], **kw)
    n_shared = len(builds)
    plain = fsr._residual_problem
    monkeypatch.setattr(fsr, "_residual_problem", lambda *args, **kwargs: plain(*args[:8]))
    builds.clear()
    alone = fsr.run_step_rule_system("h4_r4.0", [StepRule("hybrid")], **kw)
    assert scaling.to_json(shared) == scaling.to_json(alone)
    run = alone["runs"][0]
    assert len(builds) == (run["n_steps"] - 1) + run["cost_horizon"] and n_shared < len(builds)


def test_u5_reference_iterate_falls_back_to_p90(h4r4):
    """Where k* is not reached (h4_r4.0, spin-orbital R1, 4 steps; k* = 6) k_ref is the first k with p_k >= 0.9:
    C_loop(k_ref) = sum_{j < k_ref} C_iter(j) at every r, the precision entries at k = 1 and k_ref - 1, and the k*
    entries stay empty."""
    out = fsr.run_step_rule_system("h4_r4.0", [StepRule("unit")], metrics=("spin_orbital",), r_values=(0.3, 1.0, 3.0),
                                   max_iter=4, cache=h4r4["cache"])
    run = out["runs"][0]
    k_p90 = next(k for k, p in enumerate(run["p"]) if p >= 0.9)
    assert run["k_star"] is None and run["k_p90"] == k_p90 >= 1
    assert (run["k_ref"], run["k_ref_kind"]) == (k_p90, "p90")
    c_qpe = out["system"]["c_qpe"]
    for r in ("0.3", "1", "3"):
        c = run["costs"][r]
        assert c["c_loop_k_star_over_qpe"] is None
        assert c["c_loop_k_ref_over_qpe"] == pytest.approx(sum(c["c_iter"][:k_p90]) / c_qpe, rel=1e-14)
        assert [p["k"] for p in c["precision"]] == sorted({1, k_p90 - 1} & set(range(1, run["cost_horizon"] + 1)))


@pytest.mark.slow
def test_u5_reference_iterate_n2():
    """Stretched N2 (n2_r2.0, spin-orbital R1 at r = 1, 10 steps costed to k = 7): no rule reaches 1.6 mHa there, so
    k_ref is the first k with p_k >= 0.9 (k_p90 = 7), C_loop(k_ref) = sum_{j < 7} C_iter(j), and the precision entries
    lie at k = 1 and k_ref - 1 = 6."""
    out = fsr.run_step_rule_system("n2_r2.0", [StepRule("unit")], metrics=("spin_orbital",), r_values=(1.0,),
                                   max_iter=10, k_cost=7)
    run = out["runs"][0]
    k_p90 = next(k for k, p in enumerate(run["p"]) if p >= 0.9)
    assert run["k_star"] is None and run["k_ref"] == run["k_p90"] == k_p90 == 7 and run["k_ref_kind"] == "p90"
    c = run["costs"]["1"]
    assert c["c_loop_k_ref_over_qpe"] == pytest.approx(sum(c["c_iter"][:7]) / out["system"]["c_qpe"], rel=1e-14)
    assert [p["k"] for p in c["precision"]] == [1, 6]


def test_cost_horizon():
    """U5's costed iterates: k* + 2, 20 where k* is not reached, both capped at 12 for N >= 8 and at the last
    iterate K - 1; an explicit k_cost overrides (capped at K - 1)."""
    def expected(k_star, n_steps, norb, k_cost):
        if k_cost is not None:
            return min(k_cost, n_steps - 1)
        end = 20 if k_star is None else k_star + 2
        return min(min(end, 12) if norb >= 8 else end, n_steps - 1)

    cases = [(None, 31, 8, None), (26, 31, 8, None), (6, 31, 8, None), (5, 31, 8, None), (10, 31, 8, None),
             (None, 41, 6, None), (None, 41, 6, 3), (30, 41, 6, None), (3, 4, 4, None), (None, 31, 10, None),
             (None, 31, 7, None), (2, 41, 6, 50)]
    for case in cases:
        assert fsr._cost_horizon(*case) == expected(*case), case
    assert [fsr._cost_horizon(*c) for c in cases[:7]] == [12, 12, 8, 7, 12, 20, 3]


# ------------------------------------------------------------------------- R1 counts
@pytest.mark.slow
@pytest.mark.parametrize("name, k_star", [("h4_r1.8", 3), ("h4_r3.5", 7), ("h6_r3.0", 11), ("n2_r1.098", 3)])
def test_r1_iteration_counts(name, k_star):
    """Spin-orbital R1 (unit steps after the classical first step, growth cap 2, angle cap 3) reaches 1.6 mHa at
    k* = 3 (h4_r1.8), 7 (h4_r3.5), 11 (h6_r3.0), 3 (n2_r1.098) [design]."""
    act = sy.benchmark_system(name)
    ir = sy.orbital_irreps(act)
    spec = scaling.spectrum(act, ir, n_roots=2)
    run = run_step_rule(act, StepRule("unit"), max_iter=k_star + 2, irreps=ir, keep_states=True)
    rec = fsr.to_trajectory_record(run, system=name, active=act, irreps=ir, spectrum=spec)
    assert first_iterate_below(rec, M3.target) == k_star
