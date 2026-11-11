"""Tests of benchmarks/run_m4_properties.py's helpers: U10a's marks, the compiled-ablation cell of U10b, the
HF-QPE field steps, U11's floors and its F47 check with the cumulant norm's all-orders form, and the loop costs
along the compiled record.

References: hand-built strict-JSON rows (``ft.scaling.to_json`` of ``ft.properties.PropertyRow``); F51's closed
forms with synthetic stencil biases; the definitions of the floor columns by brute force; direct loops over the
record iterates with ``ft.properties.bias_bound``; M3's residual routes (``ft.residual.sampling_direct``,
``ae_x_lcu``) at the exact residual of each state.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from thc_engine_cqe import steprules
from thc_engine_cqe.fci_ops import FCISpace, normalize
from thc_engine_cqe.ft import properties as P
from thc_engine_cqe.ft import scaling, symmetry
from thc_engine_cqe.ft import steprules as ftsteprules
from thc_engine_cqe.ft.primitives import FTParameters
from thc_engine_cqe.ft.qpe import QPECost
from thc_engine_cqe.ft.residual import PrepParams, ResidualProblem, ae_x_lcu, sampling_direct
from thc_engine_cqe.ft.stateprep import prep_cost
from thc_engine_cqe.measure import routes
from thc_engine_cqe.residual import rdm12

_MOD: dict = {}


def _bench():
    if "mod" not in _MOD:
        path = Path(__file__).resolve().parents[1] / "benchmarks" / "run_m4_properties.py"
        spec = importlib.util.spec_from_file_location("run_m4_properties", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MOD["mod"] = mod
    return _MOD["mod"]


class _NoCache:
    """The ``get`` of ``ft.scaling.Cache`` without storage."""

    def get(self, name, stage, compute, key=None):
        return compute()


def _json_row(route, k, toffolis, *, loop="free", bias=1e-4, flags=(), eps=1e-3) -> dict:
    row = P.PropertyRow(system="s", target="rdm1", eps=eps, route=route, estimator="ae", loop=loop, k=k, bias=bias,
                        toffolis=toffolis, loop_toffolis=0.0, per_prep=1.0, n_prep=1.0, longest_run=1.0,
                        logical_qubits=10, flags=tuple(flags))
    out = scaling.to_json(row)
    out["toffolis_over_qpe"] = scaling.to_json(toffolis / 4.0)
    out["selectable"] = bool(row.selectable)
    return out


def test_u10a_marks_and_the_compiled_cell():
    b = _bench()
    eps = 1e-3
    assert b._a_marks(_json_row("A", 60, 8.0, flags=("record_ended",)), eps) == ""          # free, converged
    assert b._a_marks(_json_row("A", 60, 8.0, loop="meas_r1", flags=("record_ended", "not_usable")), eps) == " e"
    assert b._a_marks(_json_row("A", 5, 8.0, bias=2e-3, flags=("not_usable",)), eps) == " x"
    assert b._a_marks(_json_row("A", 60, 8.0, loop="meas_r1", bias=2e-3, flags=("not_usable", "record_ended")),
                      eps) == " x e"
    # free loop past a record that did not converge (usable bias, yet not usable)
    assert b._a_marks(_json_row("A", 60, 8.0, flags=("record_ended", "not_usable")), eps) == " e"
    assert b._a_marks(_json_row("A", 60, 8.0, bias=2e-3, flags=("not_usable", "record_ended")), eps) == " x"
    rows = [_json_row("A", 5, 8.0, bias=2e-3, flags=("not_usable",)), _json_row("A", 10, 12.0),
            _json_row("A", 20, 9.0, loop="meas_r1"), _json_row("A-oracle", 7, 1.0, flags=("oracle_k",)),
            _json_row("Bk", 3, 0.5)]
    assert b._best_a(rows, "free", 4.0) == "3 [10]"                     # route A only, the usable budget
    assert b._best_a(rows, "meas_r1", 4.0) == "2.25 [20]"
    assert b._best_a(rows[:1], "free", 4.0) == "x" and b._best_a(None, "free", 4.0) == "-"


def test_best_rows_carry_the_layout_of_their_longest_run():
    """The best selectable row (F53, Toffolis) keeps its place without a layout; its JSON entry carries the layout of
    the longest coherent run (``ft.physical.surface_code_cost`` at p_fail_ae: 3.95e11 Toffolis on 261 qubits has none,
    the best factory failing at 3.6e-16 per Toffoli) and U10b's layout cell reads "none" or "d<distance>"."""
    from thc_engine_cqe.ft import physical

    b = _bench()
    params = FTParameters()

    def row(route, toffolis, run, q=261, est="ae", flags=()):
        return P.PropertyRow(system="s", target="cumulant_norm", eps=1e-3, route=route, estimator=est, loop="free",
                             k=4, bias=1e-4, toffolis=toffolis, loop_toffolis=0.0, per_prep=1.0, n_prep=1.0,
                             longest_run=run, logical_qubits=q, flags=tuple(flags))

    long_bk, short_a = row("Bk", 4e11, 3.95e11), row("A", 9e11, 2.0e9, q=120)
    best = b._best([short_a, long_bk], "free", "ae", 2.0, params)
    assert best["route"] == "Bk" and best["toffolis_over_qpe"] == 2e11 and best["layout"] is None
    assert best["longest_run"] == 3.95e11 and best["logical_qubits"] == 261
    assert b._layout_cell(best) == "none"
    ref = physical.surface_code_cost(120, 2.0e9, fail_budget=params.p_fail_ae)
    best_a = b._best([short_a, row("Bk", 4e11, 3.95e11, flags=("not_usable",))], "free", "ae", 2.0, params)
    assert best_a["route"] == "A" and best_a["layout"]["distance"] == ref.distance
    assert best_a["layout"]["physical_qubits"] == ref.physical_qubits
    assert b._layout_cell(best_a) == f"d{ref.distance}"
    assert b._layout_cell(None) == "-"
    assert b._layout_cell({"flags": ["no_layout"]}) == "none" and b._layout_cell({"flags": []}) == "-"
    assert b._best([row("Bk", 4e11, 3.95e11, flags=("not_usable",))], "free", "ae", 2.0, params) is None


def test_hf_steps_report_the_refined_field_step():
    """The field steps of the HF-QPE rows: b from F51's stationarity with the model bias added linearly, F51's
    leading-order h, and the largest h whose stencil bias (here 1.03 and 1.10 times the leading order) is within b."""
    b = _bench()
    qpe = QPECost(encoding=None, de_qpe=1e-3, iterations=1000, toffolis=10 ** 7, logical_qubits=100)
    d3, d5, b_m = 581.571, 6.96513e6, 4.7146e-4
    scale = {"central": 1.03, "five_point": 1.10}

    def stencil(h, st):
        return scale[st] * (d3 * h ** 2 / 6.0 if st == "central" else -d5 * h ** 4 / 30.0)

    row = P.PropertyRow(system="s", target="dipole", eps=1e-3, route="HF-QPE", estimator="qpe", loop="free", k=0,
                        bias=1e-4, toffolis=1.0, loop_toffolis=0.0, per_prep=1.0, n_prep=4.0, longest_run=1.0,
                        logical_qubits=10, flags=("scheme:rep", "stencil:five_point"))
    for eps in (4e-3, 1e-3):
        out = b.hf_steps(qpe, eps, {"d3": d3, "d5": d5}, b_m, stencil, [row])
        assert set(out) == {"central", "five_point"}
        assert out["five_point"]["used"] and not out["central"]["used"]
        for st, v in out.items():
            n, c, k = (2, 6.0, d3) if st == "central" else (4, 30.0, d5)
            alloc = max(np.roots((3.0, 4.0 * b_m, b_m ** 2 - eps ** 2) if st == "central"
                                 else (5.0, 6.0 * b_m, b_m ** 2 - eps ** 2)).real)
            h0 = (c * alloc / k) ** (1.0 / n)
            h_star = h0 / scale[st] ** (1.0 / n)                       # |stencil(h*)| = b
            assert math.isclose(v["stencil_bias"], alloc, rel_tol=1e-10) and v["fixed_bias"] == b_m
            assert math.isclose(v["bias"], b_m + alloc, rel_tol=1e-10)
            assert math.isclose(v["h_leading"], h0, rel_tol=1e-10)
            assert h_star * (1.0 - 1e-3) <= v["h"] <= h_star
            assert math.isclose(abs(v["stencil_bias_at_leading"]), scale[st] * alloc, rel_tol=1e-9)
            assert v["stencil_bias_at_h"] == stencil(v["h"], st) and abs(v["stencil_bias_at_h"]) <= alloc
            assert v["feasible"] and not v["no_field_step"]
    # without d5 only the central stencil; a model bias at eps is infeasible
    out = b.hf_steps(qpe, 1e-3, {"d3": d3, "d5": None}, 1e-3, None, [])
    assert set(out) == {"central"} and not out["central"]["feasible"] and not out["central"]["no_field_step"]
    assert out["central"]["stencil_bias_at_leading"] is None and out["central"]["stencil_bias_at_h"] is None
    # a stencil bias above b at every h: no admissible step (h nan), reported, not raised
    out = b.hf_steps(qpe, 1e-3, {"d3": d3, "d5": None}, b_m, lambda h, st: 1.0, [])
    v = out["central"]
    assert v["no_field_step"] and not v["feasible"] and math.isnan(v["h"]) and v["stencil_bias_at_h"] == 1.0
    alloc = max(np.roots((3.0, 4.0 * b_m, b_m ** 2 - 1e-6)).real)
    assert math.isclose(v["stencil_bias"], alloc, rel_tol=1e-10)
    assert math.isclose(v["h_leading"], (6.0 * alloc / d3) ** 0.5, rel_tol=1e-10)


def test_floors_and_k_eps():
    """U11's floor columns: the minimum error, its k and energy error, and k_eps, the first k from which every later
    error is <= eps (None if the last error exceeds eps), against the definition by brute force."""
    b = _bench()
    rng = np.random.default_rng(5)
    for trial in range(40):
        n = int(rng.integers(1, 12))
        err = 10.0 ** rng.uniform(-4, 0, n)
        if trial % 3 == 0:
            err[int(rng.integers(n))] = 1e-3                            # an error equal to eps counts as reached
        energies = rng.uniform(-1.0, 0.0, n)
        accs = (1e-1, 1e-2, 1e-3)
        out = b.floors(err, energies, -1.5, accs)
        k_min = min(range(n), key=lambda k: (err[k], k))
        assert out["min"] == err[k_min] and out["k_min"] == k_min and out["last"] == err[-1]
        assert out["e_err_at_min"] == energies[k_min] + 1.5
        for eps in accs:
            reached = [k for k in range(n) if all(err[j] <= eps for j in range(k, n))]
            assert out["k_eps"][str(eps)] == (reached[0] if reached else None)


def _h4_r1(n: int = 20):
    key = ("h4", n)
    if key not in _MOD:
        a = symmetry.benchmark_system("h4_r1.8")
        ir = symmetry.orbital_irreps(a)
        res = steprules.run_step_rule(a, steprules.StepRule("unit"), max_iter=n, metric="spin_orbital", irreps=ir,
                                      keep_states=True)
        rec = ftsteprules.to_trajectory_record(res, system="h4_r1.8", active=a, irreps=ir,
                                               spectrum=scaling.spectrum(a, ir))
        _MOD[key] = (a, ir, rec)
    return _MOD[key]


def test_f47_check_with_the_all_orders_form():
    """U11's F47 check: the smallest bound/error over the iterates for both forms; for the cumulant norm also the
    all-orders form, never below the linearized one."""
    b = _bench()
    a, ir, rec = _h4_r1()
    psi0 = normalize(np.asarray(scaling.spectrum(a, ir).psi0))
    dm1, dm2 = rdm12(FCISpace.of(a), psi0)
    targets = {"cumulant_norm": P.cumulant_norm_target(a, ir, dm1, dm2), "cumulant": P.cumulant_target(a, ir, dm1),
               "rdm1": P.rdm1_target(a, ir)}

    def inputs(t):
        k = t.n_elements
        return P.PropertyInputs(target=t, v_shadow=np.zeros(k), alpha=np.zeros(k), n_terms=np.zeros(k),
                                sigma0=t.spreads(a, psi0), r0=t.ranges(a, psi0))

    inps = {k: inputs(t) for k, t in targets.items()}
    tn = targets["cumulant_norm"]
    errs = np.array([P.property_error(tn, a, s, psi0) for s in rec.states])
    gap = float(rec.e1 - rec.e0)
    parts = (inps["cumulant"], inps["rdm1"])
    out = b.f47_check(tn, a, psi0, inps["cumulant_norm"], rec, errs, gap, parts=parts)
    rq, re_, aq, ae = [], [], [], []
    for k, e in enumerate(errs):
        if e <= b.TINY_ERROR:
            continue
        q = 1.0 - min(float(rec.overlaps[k]), 1.0)
        d = max(float(rec.energies[k] - rec.e0), 0.0)
        sig, rng = inps["cumulant_norm"].sigma0, inps["cumulant_norm"].r0
        rq.append(P.bias_bound(tn, a, psi0, q=q, sigma0=sig, ranges=rng) / e)
        re_.append(P.bias_bound(tn, a, psi0, energy_error=d, gap=gap, sigma0=sig, ranges=rng) / e)
        aq.append(P.bias_bound(tn, a, psi0, q=q, cumulant_parts=parts) / e)
        ae.append(P.bias_bound(tn, a, psi0, energy_error=d, gap=gap, cumulant_parts=parts) / e)
    assert out["n_rows"] == len(rq) > 10 and out["order"] == "first"
    assert out["q_form_min_ratio"] == min(rq) and out["energy_form_min_ratio"] == min(re_)
    assert out["q_form_min_ratio_all_orders"] == min(aq) and out["energy_form_min_ratio_all_orders"] == min(ae)
    assert min(aq) >= min(rq) >= 1.0 and min(ae) >= min(re_) >= 1.0
    plain = b.f47_check(tn, a, psi0, inps["cumulant_norm"], rec, errs, gap)
    assert set(plain) == {"n_rows", "q_form_min_ratio", "energy_form_min_ratio"}


@pytest.mark.slow
def test_compiled_residual_costs_follow_the_compiled_record():
    """T_res(j) along the compiled record: M3's routes at the exact residual of each state with its own preparation
    cost; on the exact record's states and costs it reproduces the exact record's loop costs."""
    from thc_engine_cqe.ft import closedloop

    b = _bench()
    params = FTParameters()
    a, ir, rec = _h4_r1(12)
    inputs = closedloop.system_inputs("h4_r1.8", params=params)
    prep = prep_cost(rec.directions, rec.eps, irreps=ir, params=params)
    exact = b.residual_costs("h4_r1.8", inputs, {"r1_so": rec}, {"r1_so": prep}, _NoCache(), params)["r1_so"]
    same = b.compiled_residual_costs("h4_r1.8", inputs, {"states": rec.states, "cumulative": prep.cumulative,
                                                         "qubits": prep.logical_qubits}, _NoCache(), params)
    for lp in b.MEASURED_LOOPS:
        assert same[lp] == exact[lp] and len(same[lp]) == rec.n_steps and same[lp][0] == 0.0
    comp = b.compiled_records("h4_r1.8", inputs, rec, _NoCache(), params)["compiled_sliced"]
    tres = b.compiled_residual_costs("h4_r1.8", inputs, comp, _NoCache(), params)
    j = 3
    res_a = routes.exact_residual(a, comp["states"][j], inputs.elements, "spin_free")
    var, nrm = inputs.variances, inputs.normalizations
    prob = ResidualProblem(kind="spin_free", elements=inputs.elements, m=inputs.m, A=res_a, v_direct=var["v_direct"],
                           w_aux=var["w_aux"], lam_q=nrm["lam_q"], alpha_x=nrm["alpha_x"], n_terms_x=nrm["n_terms_x"],
                           aux_bias=None, norb=a.norb)
    pp = PrepParams(float(comp["cumulative"][j]), int(comp["qubits"][j]))
    for lp, r in b.MEASURED_LOOPS.items():
        pr = dataclasses.replace(params, r=r)
        expect = min(sampling_direct(prob, pp, params=pr).toffolis, ae_x_lcu(prob, pp, kappa=1.0, params=pr).toffolis)
        assert math.isclose(tres[lp][j], expect, rel_tol=1e-12)
    # the compiled states are not the exact ones (the loop costs are those of Psi~_j)
    assert max(float(np.linalg.norm(np.asarray(x).ravel() - np.asarray(y).ravel()))
               for x, y in zip(comp["states"], rec.states)) > 1e-6
