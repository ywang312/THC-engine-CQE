"""Tests of benchmarks/run_m4_handoff.py: U8 rows, U9 scheme rows, the compiled record and one system end to end.

References: closed forms of F40, F41 and F43 evaluated here (the identification bound, the contaminant term
q_c/(p_0 K_0^n) with K_0 from ``ft.acceptance.eigen_kernel``, the exponential search from its stages, repetition
1/p + n_v - 1), a synthetic model spectrum of two orbitals with one electron per spin (|00>, |11> and the
open-shell singlet are singlets, (|01> - |10>)/sqrt 2 the M_s = 0 triplet in PySCF's convention), and the
numbers of the n2_r2.4 hand-off [build check]: p_0 on H' 0.11201276 against 0.11246346 on H, its w_L filter
1,218,282 and coarse QPE 2,002,944 Toffolis at C_QPE = 9,157,632.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import types

import numpy as np
import pytest

from thc_engine_cqe.ft import acceptance as acc
from thc_engine_cqe.ft import amplify, certify, handoff, scaling, symmetry
from thc_engine_cqe.ft.primitives import FTParameters
from thc_engine_cqe.ft.trajectory import exact_trajectory

_PATH = pathlib.Path(__file__).resolve().parents[1] / "benchmarks" / "run_m4_handoff.py"
_SPEC = importlib.util.spec_from_file_location("run_m4_handoff", _PATH)
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)

DE = 1e-3
C_QPE = 1e7


def _bbht_by_stages(a: float, growth: float = 1.2) -> float:
    """E[N] of the exponential search, stage r succeeding with the mean of sin^2((2j + 1) phi), j < ceil(growth^r)."""
    phi = math.asin(math.sqrt(a))
    total, alive, r = 0.0, 1.0, 0
    while alive > 1e-16:
        m = math.ceil(growth ** r - 1e-12)
        total += alive * m
        alive *= 1.0 - float(np.mean(np.sin((2 * np.arange(m) + 1) * phi) ** 2))
        r += 1
    return total


# ------------------------------------------------------------------ a synthetic model spectrum
_E = np.array([-1.0, -0.995, -0.80, -0.50])           # psi_0, the triplet, E_1 (singlet), a high singlet
_S2 = np.array([0.0, 2.0, 0.0, 0.0])
_R = 1.0 / math.sqrt(2.0)
_V = np.array([[1, 0, 0, 0], [0, _R, _R, 0], [0, -_R, _R, 0], [0, 0, 0, 1]], dtype=float)   # columns: eigenvectors


def _spectrum() -> acc.ModelSpectrum:
    lam, c = 2.0, -0.5
    x0 = (_E[0] - c) / lam
    return acc.ModelSpectrum(energies=_E, vectors=_V, s2=_S2, irreps=np.zeros(4, dtype=np.int64), shape=(2, 2),
                             lam=lam, c=c, sin_theta0=math.sqrt(1.0 - x0 * x0), e0=float(_E[0]),
                             e1_singlet=float(_E[2]), e1_any_spin=float(_E[1]), e1_any_irrep=float(_E[2]))


def _state(weights) -> np.ndarray:
    return (_V @ np.sqrt(np.asarray(weights, dtype=float))).reshape(2, 2)


def _side(spec: acc.ModelSpectrum, e1lb: float) -> dict:
    m2 = 2 * math.ceil(math.pi * spec.lam / (2 * DE))
    g = spec.grid(m2)
    return {"spec": spec, "grid": g, "e1lb": e1lb, "de": DE, "qpe": types.SimpleNamespace(toffolis=C_QPE), "m2": m2,
            "m4": 2 * m2, "u": None, "u_certified": False, "u_final": None, "theta_l": e1lb - 5 * g,
            "b_l_terminates": True, "w_l_certified": False, "filters": {"u": None, "w_l": None, "oracle_u": None},
            "filter_cache": {}}


def _rule_b_row(side: dict, p: float, k: int = 1) -> handoff.RuleBRow:
    theta = side["theta_l"]
    return handoff.RuleBRow(k=k, p=p, e_err=0.1, c_prep=0.0, c_loop=0.0, form="B_L", certified=False, certificate=None,
                            theta=theta, n_runs=2, cost=C_QPE / p + C_QPE, cost_oracle_e=C_QPE / p,
                            flags=("no_certificate", "termination_not_certified"),
                            bracket_flags=("no_certificate", "termination_not_certified"))


_INPUTS = types.SimpleNamespace(spectrum=types.SimpleNamespace(e0=-1.0, e1=-0.8),
                                active=types.SimpleNamespace(norb=2, nelec=(1, 1)))


def _k0(spec, side, theta: float) -> float:
    m = side["m2"]
    grid = spec.c + spec.lam * np.cos(2 * np.pi * np.arange(m) / m)
    return float(acc.eigen_kernel(spec.e0, spec.lam, spec.c, acc.phase_window(m))[grid < theta].sum())


def test_u8_bounds_of_a_compiled_state_with_a_contaminant():
    """The design-floor bound [(1 - p_min)/p_min] tau_W^n_v assumes every populated excited level at or above E1lb:
    a compiled state with weight q = 1e-4 on the triplet 5 mHa above E_0' (below theta) has none ("bound_premise"
    False, "n/a (q_c)" in the markdown), its state bound is [(1 - p_0)/p_0] (tau_W(5)/K_0)^2 + q/(p_0 K_0^2), above
    the kernel term; without the contaminant both bounds apply; with E1lb above a populated singlet as well, every
    populated level below E1lb counts as accepted: [(1 - p_0)/p_0] (tau_W/K_0)^2 + (q + p_1)/(p_0 K_0^2), and a state
    whose only level below E1lb is that singlet (q_c = 0) is marked "n/a (below E1lb)"."""
    spec = _spectrum()
    side = _side(spec, spec.e1_singlet - 1e-4)
    q, p0 = 1e-4, 0.3
    ci = _state([p0, q, 1.0 - p0 - q, 0.0])
    rb = _rule_b_row(side, p0)
    row = bench.u8_row("Psi1", "compiled_so", 1, ci, rb, side, _INPUTS, compiled=True, p_h=0.31)
    k0 = _k0(spec, side, rb.theta)
    tau = acc.tail_mass(5.0, window="sine")
    assert row["distance"] == pytest.approx(5.0, abs=1e-9) and row["p"] == 0.31 and row["p_model"] == pytest.approx(p0)
    assert row["contaminant"] == pytest.approx(q, rel=1e-12) and not row["bound_premise"] and row["bound_p_min"] is None
    ref = (1 - p0) / p0 * (tau / k0) ** 2 + q / (p0 * k0 ** 2)
    assert row["bound_state"] == pytest.approx(ref, rel=1e-9)
    assert row["contaminant_budget"] == pytest.approx(q / (p0 * k0 ** 2), rel=1e-9)
    assert row["bound_state"] >= row["laws"]["sine_2n"]["n_v_runs"]["kernel"] > q / p0 * 0.99
    assert "exact_state_loop_costs" in row["flags"]
    exact = bench.u8_row("Psi1", "r1_so", 1, _state([p0, 0.0, 1.0 - p0, 0.0]), rb, side, _INPUTS, compiled=False,
                         p_h=0.31)
    assert exact["bound_premise"] and exact["bound_p_min"] == pytest.approx(999.0 * tau ** 2, rel=1e-12)
    assert exact["bound_state"] == pytest.approx((1 - p0) / p0 * (tau / k0) ** 2, rel=1e-9)
    assert exact["bound_state"] >= exact["laws"]["sine_2n"]["n_v_runs"]["kernel"]
    high = _side(spec, float(_E[3]) - 1e-4)                         # E1lb above the populated singlet E_1
    rb_h = _rule_b_row(high, p0)
    both = bench.u8_row("Psi1", "compiled_so", 1, ci, rb_h, high, _INPUTS, compiled=True, p_h=0.31)
    k0_h, tau_h = _k0(spec, high, rb_h.theta), acc.tail_mass(5.0, window="sine")
    assert both["bound_p_min"] is None and not both["bound_premise"]
    assert both["excited_below_e1lb"] == pytest.approx(1.0 - p0, rel=1e-12)
    ref = (1 - p0) / p0 * (tau_h / k0_h) ** 2 + (1.0 - p0) / (p0 * k0_h ** 2)
    assert both["bound_state"] == pytest.approx(ref, rel=1e-9)
    assert both["bound_state"] >= both["laws"]["sine_2n"]["n_v_runs"]["kernel"]
    # the markdown cell: the bound, or why its premise fails (no contaminant below theta: a singlet below E1lb)
    singlet = bench.u8_row("Psi1", "compiled_so", 1, _state([p0, 0.0, 1.0 - p0, 0.0]), rb_h, high, _INPUTS,
                           compiled=True, p_h=0.31)
    assert singlet["contaminant"] < 1e-30 and not singlet["bound_premise"]
    assert singlet["bound_state"] == pytest.approx(ref, rel=1e-9)
    assert bench.bound_cell(row) == bench.bound_cell(both) == "n/a (q_c)"
    assert bench.bound_cell(singlet) == "n/a (below E1lb)"
    assert bench.bound_cell(exact) == bench._f(999.0 * tau ** 2, ".2e")


def test_u8_row_at_a_distance_beyond_the_tail_register():
    """A B_U threshold more than M/2 - 1 = 2047 grid units below E1lb (here 3.4 Ha, D = 3398.5, at g = 1e-3) takes
    tau_W at 1000 grid units, which bounds the non-increasing tail beyond it (as verification_runs), instead of
    failing."""
    spec = _spectrum()
    side = dict(_side(spec, 2.4), grid=1e-3)
    rb = handoff.RuleBRow(k=1, p=1.0, e_err=0.1, c_prep=0.0, c_loop=0.0, form="B_U", certified=True,
                          certificate=certify.EnergyCertificate("classical", -0.9995, None, 0.0, 0.0, 0.0, "u"),
                          theta=-0.9985, n_runs=1, cost=C_QPE, cost_oracle_e=C_QPE)
    row = bench.u8_row("Psi1", "r1_so", 1, _state([1.0, 0.0, 0.0, 0.0]), rb, side, _INPUTS, compiled=False, p_h=1.0)
    assert row["distance"] > 2047 and row["tau"] == acc.tail_mass(1000.0, window="sine")
    assert row["bound_premise"] and row["bound_p_min"] == pytest.approx(999.0 * row["tau"], rel=1e-12)


def test_scheme_rows_of_a_compiled_state():
    """Compiled rows (F41, F43): every scheme accepted below theta carries "contaminant_below_theta" and
    q_c/(p_0 K_0^n) (n = n_v for rep, aa, bbht, fp at rule B's theta), the row's flags are those of s* (rep here:
    1/p + 1 = 4.333 C_QPE against bbht's E_BBHT(0.3) + 1), with "exact_state_loop_costs"; fp's Eckart-Temple bound
    refers to the lowest populated excited level (the triplet, less 0.1 mHa): none for an energy above it, the bound
    with it below."""
    spec = _spectrum()
    side = _side(spec, spec.e1_singlet - 1e-4)
    q, p0 = 1e-4, 0.3
    ci = _state([p0, q, 1.0 - p0 - q, 0.0])
    rb = _rule_b_row(side, p0)
    rec = types.SimpleNamespace(energies=[-0.9, -0.9], variances=[1e-6, 1e-6], e1=-0.8)
    row = bench.scheme_row("Psi1", "compiled_so", 1, ci, rb, [0.0, 0.0], rec, [], side, FTParameters(), p_h=0.31,
                           compiled=True)
    c = row["costs_over_qpe"]
    assert c["rep"] == pytest.approx(1.0 / p0 + 1.0, rel=1e-12)
    assert c["bbht"] == pytest.approx(_bbht_by_stages(p0) + 1.0, rel=1e-10) and c["bbht"] > c["rep"]
    assert row["best_realizable"] == "rep" and row["flags"] == row["scheme_flags"]["rep"]
    assert row["flags"] == ["no_certificate", "termination_not_certified", "contaminant_below_theta",
                            "exact_state_loop_costs"]
    k0 = _k0(spec, side, rb.theta)
    assert row["contaminant"]["q_c"]["rep"] == pytest.approx(q, rel=1e-12)
    assert row["contaminant"]["terms"]["rep"] == pytest.approx(q / (p0 * k0 ** 2), rel=1e-9)
    assert row["p_lb"] is None and c["fp"] is None and row["scheme_flags"]["fp"][0] == "no_overlap_bound"
    assert "contaminant_below_theta" in row["row_flags"] and row["p_model"] == pytest.approx(p0) and row["p"] == 0.31
    low = types.SimpleNamespace(energies=[-0.999, -0.999], variances=[1e-6, 1e-6], e1=-0.8)
    row = bench.scheme_row("Psi1", "compiled_so", 1, ci, rb, [0.0, 0.0], low, [], side, FTParameters(), p_h=0.31,
                           compiled=True)
    assert row["p_lb"] == pytest.approx(handoff.overlap_lower_bound(-0.999, 1e-6, -0.995 - 1e-4), rel=1e-14)
    exact = bench.scheme_row("Psi1", "r1_so", 1, _state([p0, 0.0, 1 - p0, 0.0]), rb, [0.0, 0.0], low, [], side,
                             FTParameters(), p_h=0.31)
    assert exact["p_lb"] == pytest.approx(handoff.overlap_lower_bound(-0.999, 1e-6, -0.8), rel=1e-14)
    assert "contaminant" not in exact and exact["flags"] == ["no_certificate", "termination_not_certified"]


def test_scheme_row_takes_p_on_h_prime():
    """U9's p is the weight on psi_0' of H' (spec.weights), not the record's overlap on H: n2_r2.4 HF with
    p = 0.11201276 on H' and 0.11246346 on H, B_L with two runs and its w_L filter: rep = 1/p + 1 = 9.927554 and
    lt_bbht_L = E_BBHT(p) C_filt(w_L)/C_QPE + 1 = 1.705232 C_QPE; s* lt_bbht_L, flagged "termination_not_certified"."""
    p, p_h, c_qpe = 0.11201276, 0.11246346, 9157632.0
    spec = types.SimpleNamespace(weights=lambda ci: np.array([p, 1.0 - p]), e0=-107.43398323034428)
    e1lb = -107.20321992235951
    side = {"spec": spec, "e1lb": e1lb, "de": DE, "grid": 0.001971344288983838,
            "qpe": types.SimpleNamespace(toffolis=c_qpe), "w_l_certified": False, "b_l_terminates": True,
            "filters": {"u": None, "w_l": {"c_filt": 1218282.0, "c_coarse": 2002944.0}, "oracle_u": None}}
    rb = handoff.RuleBRow(k=0, p=p_h, e_err=0.777, c_prep=0.0, c_loop=0.0, form="B_L", certified=False,
                          certificate=None, theta=e1lb - 5 * side["grid"], n_runs=2, cost=c_qpe / p_h + c_qpe,
                          cost_oracle_e=c_qpe / p_h, flags=("no_certificate", "termination_not_certified"),
                          bracket_flags=("no_certificate", "termination_not_certified"))
    rec = types.SimpleNamespace(energies=[-106.66], variances=[0.5], e1=-107.2)
    row = bench.scheme_row("HF", "r1_so", 0, None, rb, [0.0], rec, [], side, FTParameters(), p_h=p_h)
    c = row["costs_over_qpe"]
    assert row["p_model"] == p and row["p"] == p_h
    assert c["rep"] == pytest.approx(1.0 / p + 1.0, rel=1e-12) == pytest.approx(9.927554, abs=1e-6)
    ref = _bbht_by_stages(p) * 1218282.0 / c_qpe + 1.0
    assert c["lt_bbht_L"] == pytest.approx(ref, rel=1e-10) == pytest.approx(1.705232, abs=1e-6)
    assert c["2s_L"] == pytest.approx(2002944.0 / c_qpe / p + 1.0, rel=1e-12)
    assert c["lt_rep_L"] == pytest.approx(1218282.0 / c_qpe / p + 1.0, rel=1e-12)
    assert row["best_realizable"] == "lt_bbht_L" and row["flags"] == ["termination_not_certified"]
    assert row["rep_rule_e_over_qpe"] == pytest.approx(1.0 / p, rel=1e-12)
    assert c["lt_bbht"] is None and row["scheme_flags"]["lt_bbht"] == ["no_certificate"]


def test_scheme_row_final_qpe_of_the_u_referenced_forms():
    """The U-referenced forms (lt, lt_bbht, 2s, lt_rep) end with one QPE accepted by B_U at U + dE where D_U >= 1.3
    (no flag), else by B_L ("termination_not_certified"), and have no cost where that B_L does not terminate
    ("not_terminating"; the U is certified, so not "no_certificate"); with the certificate's filter C_filt = 0.2 and
    C_c = 0.5 C_QPE: lt_bbht = E_BBHT(p) C_filt + C_QPE, lt_rep = C_filt/p + C_QPE, 2s = C_c/p + C_QPE."""
    spec = _spectrum()
    side = _side(spec, spec.e1_singlet - 1e-4)
    g, e1lb, p0 = side["grid"], side["e1lb"], 0.3
    ci = _state([p0, 0.0, 1.0 - p0, 0.0])
    rec = types.SimpleNamespace(energies=[-0.9, -0.9], variances=[1e-6, 1e-6], e1=-0.8)
    rb = _rule_b_row(side, p0)
    u_keys = ("lt", "lt_bbht", "2s", "lt_rep")
    for d_u, b_l, ref_flags in ((2.0, True, []), (1.0, True, ["termination_not_certified"]),
                                (1.0, False, ["not_terminating"])):
        u = e1lb - DE - d_u * g
        assert u <= e1lb - 2 * DE                                   # a certified U
        cert = certify.EnergyCertificate("classical", u, None, 0.0, 0.0, 0.0, "cisd")
        sd = dict(side, b_l_terminates=b_l,
                  filter_cache={round(u, 12): {"c_filt": 0.2 * C_QPE, "c_coarse": 0.5 * C_QPE}})
        row = bench.scheme_row("Psi1", "r1_so", 1, ci, rb, [0.0, 0.0], rec, [cert], sd, FTParameters(), p_h=0.31)
        c, fl = row["costs_over_qpe"], row["scheme_flags"]
        assert all(fl[s] == ref_flags for s in u_keys), (d_u, b_l, {s: fl[s] for s in u_keys})
        assert not any(s.endswith("@oracle_U") for s in c)
        if b_l:
            assert c["lt_bbht"] == pytest.approx(_bbht_by_stages(p0) * 0.2 + 1.0, rel=1e-10)
            assert c["lt_rep"] == pytest.approx(0.2 / p0 + 1.0, rel=1e-12)
            assert c["2s"] == pytest.approx(0.5 / p0 + 1.0, rel=1e-12)
            assert row["sources"]["lt_bbht"] == "cisd"
        else:
            assert all(c[s] is None for s in u_keys)


def test_compiled_record_flags(tmp_path):
    """The compiled record (R1 spin-orbital steps compiled elementwise from HF) carries "compiled_elementwise" and
    "exact_state_loop_costs" (its loop costs are those of the exact states); its HF is the record's."""
    a = symmetry.benchmark_system("h4")
    ir = symmetry.orbital_irreps(a)
    rec = exact_trajectory(a, optimizer="lbfgs", metric="spin_orbital", max_iter=3, irreps=ir,
                           first_step="second_order")
    e0, ci0 = a.fci()
    inputs = types.SimpleNamespace(active=a, irreps=ir, spectrum=types.SimpleNamespace(psi0=np.asarray(ci0)))
    comp = bench.compiled_record("h4", inputs, rec, scaling.Cache(tmp_path), FTParameters())
    assert {"compiled_elementwise", "exact_state_loop_costs"} <= set(comp.flags)
    assert comp.energies[0] == pytest.approx(rec.energies[0], abs=1e-10)
    assert comp.overlaps[0] == pytest.approx(rec.overlaps[0], abs=1e-10) and np.all(comp.overlaps <= 1 + 1e-12)


@pytest.mark.slow
def test_handoff_rows_of_n2_r2_4(tmp_path):
    """n2_r2.4 end to end (QPE side, records, U8, U9 part 1): U8's T_B uses p on H' and equals U9's rep (F40) of the
    same row (HF of the R1 record: 1/0.112013 + 1 = 9.928 C_QPE); every U9 row's flags are those of its s*; the
    dedicated sampled certificate of the exact Psi_3 costs its expected runs (1.25, T_B 1021.11 C_QPE); compiled rows
    with a contaminant have no design-floor bound and a state bound above their kernel term."""
    params = FTParameters()
    cache = scaling.Cache(tmp_path / "cache")
    inputs = bench.closedloop.system_inputs("n2_r2.4", cache=cache, params=params)
    side = bench.qpe_side("n2_r2.4", inputs, cache, params)
    res = bench.state_path("n2_r2.4", inputs, side, cache, params, {})
    u9 = {(key.split("@r=")[0], float(key.split("@r=")[1]), row["state"]): row
          for key, rows in res["u9"].items() for row in rows}
    for u in res["u8"]:
        r = 1.0 if u["record"].startswith(("r1", "compiled")) else params.r
        row = u9[(u["record"], r, u["state"])]
        assert u["cost_over_qpe"] == pytest.approx(row["costs_over_qpe"]["rep"], rel=1e-9)
        assert u["p_model"] == pytest.approx(row["p_model"], rel=1e-14)
    hf = next(u for u in res["u8"] if u["state"] == "HF")
    assert hf["cost_over_qpe"] == pytest.approx(1.0 / hf["p_model"] + 1.0, rel=1e-9) == pytest.approx(9.928, abs=1e-3)
    for row in u9.values():
        assert row["best_realizable"] == amplify.best_realizable(row["costs"])
        assert row["flags"] == row["scheme_flags"][row["best_realizable"]]
    psi3 = next(u for u in res["u8"] if u["record"] == "exact_so" and u["state"] == "Psi3")
    assert psi3["u_source"] == "sampled" and psi3["n_v"] == 1
    assert psi3["n_v_expected"] == pytest.approx(1.25, abs=2e-3) and psi3["cost_over_qpe"] == pytest.approx(1021.11,
                                                                                                          abs=0.01)
    assert "exact_state_loop_costs" in res["records"]["compiled_so"]["flags"]
    compiled = [u for u in res["u8"] if u["compiled"] and u["contaminant"] > 1e-12]
    assert compiled
    for u in compiled:
        assert u["bound_p_min"] is None and u["bound_state"] >= u["laws"]["sine_2n"]["n_v_runs"]["kernel"]
    sp = inputs.spectrum
    result = {"system": "n2_r2.4", "spectrum": {"flags": list(sp.flags)}, "qpe_side": bench.side_summary(side, inputs),
              "m3_t6": None, "timings": {}, "runtime_s": 0.0, "peak_rss_mb": 0.0, **res}
    md = bench.markdown([result], "test", 0.0)
    assert "n/a (q_c)" in md and "flags of s*" in md and "lt_rep_L" in md
