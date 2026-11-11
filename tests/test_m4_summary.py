"""Tests of benchmarks/run_m4_summary.py: the scheme costs of a state, k_eps, the U0 verdict logic (Q1 windows per
gap model, a growing power-law gap included, Q2 against the windows of M3's rule E and kernel-aware rule B, Q3 at
k_ref and under noise, Q4 with the lagged sensitivity and the control cells, Q5 per r, without a realizable scheme and
with M3's T6 convention, Q6 at both accuracies, per pre-declared budget and with the run layouts), the QPE-side check,
the failure budgets of U12, the missing-inputs rows, the "inputs complete" check with the quick and the full input
sets, the exit status and the script digest in the keys of the summary's own cache stages.

The script is loaded from its file (benchmarks/ is not a package); the inputs are small synthetic dicts shaped like
the JSON of the M4 benchmarks. Expected values are closed forms: BBHT's expected calls from its stage success
probability 1/2 - sin(4 M phi)/(4 M sin 2 phi) [cited: Boyer, Brassard, Hoyer, Tapp, Fortsch. Phys. 46, 493 (1998),
Lemma 2], the schemes' costs of F43, the window ends (A/m)^(1/gamma) and A'/m of F21, and counts by hand.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import math
import types
from pathlib import Path

import numpy as np
import pytest

from thc_engine_cqe.ft import properties as P
from thc_engine_cqe.ft import scaling
from thc_engine_cqe.ft.primitives import FTParameters

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "run_m4_summary.py"
_MOD: dict = {}


def _summary():
    if "mod" not in _MOD:
        spec = importlib.util.spec_from_file_location("run_m4_summary", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MOD["mod"] = mod
    return _MOD["mod"]


def _bbht(a: float, growth: float = 1.2) -> float:
    """E[N] of the exponential search: sum_r M_r prod_{s<r}(1 - P(M_s)), M_r = ceil(growth^r), with the stage
    success P(M) = 1/2 - sin(4 M phi)/(4 M sin 2 phi), sin^2 phi = a."""
    phi = math.asin(math.sqrt(a))
    total, alive, r = 0.0, 1.0, 0
    while alive > 1e-16:
        m = math.ceil(growth ** r - 1e-12)
        total += alive * m
        alive *= 0.5 + math.sin(4 * m * phi) / (4 * m * math.sin(2 * phi))
        r += 1
    return total


def _side(**kw) -> dict:
    """A QPE side (``qpe_side``'s keys that the scheme costs read): C_QPE 1e7 Toffolis, B_L with n_v = 2, the
    E1lb-referenced filter (C_filt 1.33e6, C_c 2.19e6) not certified, no certified U."""
    side = {"system": "x", "c_qpe": 1e7, "n_v": 2, "form": "B_L", "rule_b_terminates": True,
            "bracket_flags": ["termination_not_certified", "no_certificate"], "u_final": None, "b_l_terminates": True,
            "e0_model": -1.0, "e1lb": -0.8, "w_l_certified": False, "u_certified": False,
            "filters": {"u": None, "w_l": {"c_filt": 1.33e6, "c_coarse": 2.19e6}, "oracle_u": None}}
    side.update(kw)
    return side


# ------------------------------------------------------------------------- scheme costs of a state
def test_state_costs_follow_the_scheme_closed_forms():
    """At p = 0.21252, C(Psi) = C_loop = 0, in C_QPE units (the side's Toffolis scaled by 1/C_QPE): lt_bbht_L =
    E_BBHT(p) c_filt(w_L) + 1, rep = 1/p + (n_v - 1), bbht = E_BBHT(p) + (n_v - 1), 2s_L = c_c(w_L)/p + 1, lt_rep_L =
    c_filt(w_L)/p + 1 and M3's rule E 1/p; the L forms carry "termination_not_certified", rep and bbht rule B's
    bracket flags; without terminating rule B, rep, aa, bbht and fp have no cost ("not_terminating")."""
    s = _summary()
    p = 0.21252
    e = _bbht(p)
    assert e == pytest.approx(3.768386, rel=1e-6)
    fl = {}
    c = s.state_costs(_side(), p, 0.0, 0.0, 1.0, flags=fl)
    assert c["lt_bbht_L"] == pytest.approx(1.0 + 0.133 * e, rel=1e-9)
    assert c["rep"] == pytest.approx(1.0 / p + 1.0, rel=1e-12)
    assert c["bbht"] == pytest.approx(e + 1.0, rel=1e-9)
    assert c["2s_L"] == pytest.approx(0.219 / p + 1.0, rel=1e-12)
    assert c["lt_rep_L"] == pytest.approx(0.133 / p + 1.0, rel=1e-12)
    assert c["rep_rule_e"] == pytest.approx(1.0 / p, rel=1e-12)
    assert c["fp"] is None and c["lt"] is None and c["lt_bbht"] is None and c["2s"] is None and c["lt_rep"] is None
    assert fl["lt_bbht_L"] == ["termination_not_certified"] and fl["rep"] == ["termination_not_certified",
                                                                              "no_certificate"]
    assert fl["lt"] == ["no_certificate"] and fl["fp"] == ["no_overlap_bound"]
    # in Toffolis: the same numbers times C_QPE
    c_t = s.state_costs(_side(), p, 0.0, 0.0, 1e7)
    assert c_t["lt_bbht_L"] == pytest.approx(1e7 * (1.0 + 0.133 * e), rel=1e-9)
    # with a state cost and a loop cost: T = C_loop + (C + C_QPE)/p + (n_v - 1) C_QPE under repetition
    c2 = s.state_costs(_side(), 0.5, 0.2, 0.7, 1.0)
    assert c2["rep"] == pytest.approx(0.7 + 1.2 / 0.5 + 1.0, rel=1e-12)
    assert c2["rep_rule_e"] == pytest.approx(0.7 + 1.2 / 0.5, rel=1e-12)
    fl3 = {}
    c3 = s.state_costs(_side(rule_b_terminates=False), p, 0.0, 0.0, 1.0, rule_e=False, flags=fl3)
    assert all(c3[k] is None and fl3[k] == ["not_terminating"] for k in ("rep", "aa", "bbht", "fp"))
    assert "rep_rule_e" not in c3 and c3["lt_bbht_L"] == pytest.approx(1.0 + 0.133 * e, rel=1e-9)
    # B_L without a terminating final QPE: the L forms need E_0' <= E1lb - w_L
    c4 = s.state_costs(_side(e0_model=-0.85), p, 0.0, 0.0, 1.0)
    assert c4["lt_bbht_L"] is None and c4["2s_L"] is None


def test_k_eps_is_the_first_k_after_which_the_error_stays_below_eps():
    s = _summary()
    assert s._k_eps(np.array([0.5, 2e-3, 5e-4, 2e-3, 4e-4, 3e-4]), 1e-3) == 4
    assert s._k_eps(np.array([5e-4, 4e-4]), 1e-3) == 0
    assert s._k_eps(np.array([0.5, 4e-4, 2e-3]), 1e-3) is None


# ------------------------------------------------------------------------- the QPE side against the hand-off's
def _hand_side(**kw) -> dict:
    q = {"e1lb": -0.8, "u_cisd": -0.95, "grid": 2e-3, "c_qpe": 1e7, "b_l_terminates": True, "u_final": "B_U",
         "theta_l_minus_e0_model": 0.19, "u_certified": True, "n_v_u": 1, "n_v_l": 2,
         "filters": {"u": {"c_filt": 1.0, "c_coarse": 2.0}, "w_l": None, "oracle_u": None}}
    q.update(kw)
    return {"qpe_side": q}


def _qside(**kw) -> dict:
    side = {"system": "x", "e1lb": -0.8, "u_cisd": -0.95, "grid": 2e-3, "c_qpe": 1e7, "b_l_terminates": True,
            "u_final": "B_U", "theta_l": -0.81, "e0_model": -1.0, "form": "B_U", "n_v": 1,
            "filters": {"u": {"c_filt": 1.0, "c_coarse": 2.0}, "w_l": None, "oracle_u": None}}
    side.update(kw)
    return side


def test_side_check_expects_rule_b_with_more_runs_where_b_l_does_not_terminate():
    """Rule B's form without a dedicated certificate: B_U with the certified U where it needs no more runs than B_L,
    or where B_L does not terminate, else B_L; the side's B_L termination, the final QPE at U and theta_L - E_0'
    must equal the hand-off benchmark's."""
    s = _summary()
    assert s.side_check(_qside(), _hand_side())["passed"]
    assert not s.side_check(_qside(form="B_L", n_v=2), _hand_side())["passed"]
    # B_U needs 3 runs against B_L's 2: B_L, unless B_L does not terminate
    assert s.side_check(_qside(form="B_L", n_v=2), _hand_side(n_v_u=3))["passed"]
    assert not s.side_check(_qside(form="B_U", n_v=3), _hand_side(n_v_u=3))["passed"]
    assert s.side_check(_qside(form="B_U", n_v=3, b_l_terminates=False),
                        _hand_side(n_v_u=3, b_l_terminates=False))["passed"]
    assert not s.side_check(_qside(form="B_L", n_v=2, b_l_terminates=False),
                            _hand_side(n_v_u=3, b_l_terminates=False))["passed"]
    for key, value in (("u_final", "B_L"), ("theta_l_minus_e0_model", 0.18), ("b_l_terminates", False)):
        out = s.side_check(_qside(), _hand_side(**{key: value}))
        assert not out["passed"] and key in out["detail"]
    assert s.side_check(_qside(theta_l=-0.81 + 1e-16), _hand_side())["passed"]       # 1e-12 relative


def test_recost_and_hf_checks_compare_costs_and_flags():
    """U9 part 1 re-costed with the summary's side: the same T_s (Toffolis) and the same scheme flags, less the state's
    own flags ("contaminant_below_theta", "exact_state_loop_costs"); U-referenced schemes of a dedicated certificate
    are skipped. U8's rule-B cost and oracle T_E of HF must equal U9's rep and rule-E columns."""
    s = _summary()
    p, e = 0.21252, _bbht(0.21252)
    q = 1e7
    row = {"state": "HF", "k": 0, "p_model": p, "c_prep": 0.0, "c_loop": 0.0, "p_lb": None, "form": "B_L", "n_v": 2,
           "sources": {"rep": "none", "bbht": "none"},
           "costs": {"rep": q * (1.0 / p + 1.0), "bbht": q * (e + 1.0), "lt_bbht_L": q * (1.0 + 0.133 * e),
                     "lt_rep_L": q * (0.133 / p + 1.0), "2s_L": q * (0.219 / p + 1.0), "lt_rep": None},
           "scheme_flags": {"rep": ["termination_not_certified", "no_certificate"],
                            "bbht": ["termination_not_certified", "no_certificate"],
                            "lt_bbht_L": ["termination_not_certified", "exact_state_loop_costs"],
                            "lt_rep_L": ["termination_not_certified"], "2s_L": ["termination_not_certified"],
                            "lt_rep": ["no_certificate"]}}
    hand = {"u9": {"r1_so@r=1.0": [row]}}
    out = s.recost_check(_side(), hand)
    assert out["passed"] and out["detail"] == "6 costs agree"
    bad = json.loads(json.dumps(hand))
    bad["u9"]["r1_so@r=1.0"][0]["scheme_flags"]["lt_bbht_L"] = []
    out = s.recost_check(_side(), bad)
    assert not out["passed"] and "lt_bbht_L flags" in out["detail"]
    bad = json.loads(json.dumps(hand))
    bad["u9"]["r1_so@r=1.0"][0]["costs"]["lt_rep_L"] *= 1.0 + 1e-9
    assert "lt_rep_L" in s.recost_check(_side(), bad)["detail"]
    # a dedicated certificate behind the filter followed by repetition: skipped
    bad["u9"]["r1_so@r=1.0"][0]["costs"]["lt_rep_L"] = row["costs"]["lt_rep_L"]
    bad["u9"]["r1_so@r=1.0"][0]["costs"]["lt_rep"] = 3.0
    bad["u9"]["r1_so@r=1.0"][0]["sources"]["lt_rep"] = "ae_1"
    assert s.recost_check(_side(), bad)["detail"] == "5 costs agree"
    hf = {"u8": [{"state": "HF", "compiled": False, "record": "r1_so", "cost_over_qpe": 1.0 / p + 1.0,
                  "cost_oracle_e_over_qpe": 1.0 / p}],
          "u9": {"r1_so@r=1.0": [{"k": 0, "costs_over_qpe": {"rep": 1.0 / p + 1.0}, "rep_rule_e_over_qpe": 1.0 / p}],
                 "exact_so@r=0.3": [{"k": 0, "costs_over_qpe": {"rep": 0.0}, "rep_rule_e_over_qpe": 0.0}]}}
    assert s.hf_cost_check("x", hf)["passed"] and s.hf_cost_check("x", hf)["detail"] == "1 rows agree"
    hf["u9"]["r1_so@r=1.0"][0]["rep_rule_e_over_qpe"] = 1.0 / p + 1e-6
    assert not s.hf_cost_check("x", hf)["passed"]
    assert not s.hf_cost_check("x", {"u8": [], "u9": {}})["passed"]


def test_l_window_check_reads_the_classical_step_series():
    """The window check of the E1lb-referenced forms starts at p_1(n0) of the classical-step record of the metric
    (lbfgs/second_order for sf), which exists at every size, the R1 records only up to N = 8."""
    s = _summary()
    rec = types.SimpleNamespace(name="h10_r3.0", records={"lbfgs/second_order": types.SimpleNamespace(
        overlaps=[0.4, 0.3])})
    gap = {"A": 2.0, "gamma": 0.8, "A_inv": 2.2}
    b = {"n0": 10, "metric": "sf", "alpha_k": [0.04, 0.05], "betas": {"last": 9.0, "fixed_9": 9.0}, "c0": 200.0}
    out = s.l_window_check(b, gap, {10: rec}, {})
    thr = 0.1 + 1e-3
    n_l = max((2.0 / thr) ** (1 / 0.8), 2.2 / thr)
    assert out["n_l"] == pytest.approx(n_l, rel=1e-12)
    assert out["p_low"] == pytest.approx(0.3 * math.exp(-0.05 * (n_l - 10)), rel=1e-12)
    assert out["flag"] == "l_window_no_break_even" and out["c_iter_min"] == pytest.approx(200.0, rel=1e-12)
    # a power law that does not close: a growing gap (gamma < 0) keeps the window open beyond every n; a constant
    # gap (gamma = 0) below w_L + 1 mHa has no window under the power law, which leaves A'/n's end A'/thr = 4.95
    grow = s.l_window_check(b, {"A": 2.5e-4, "gamma": -1.0, "A_inv": 0.025}, {10: rec}, {})
    assert grow == {"flag": "l_window_unresolved", "n_l": math.inf, "reason": "the window does not close"}
    flat = s.l_window_check(b, {"A": 0.05, "gamma": 0.0, "A_inv": 0.5}, {10: rec}, {})
    assert flat == {"flag": "l_window_closed", "n_l": pytest.approx(0.5 / thr, rel=1e-12)}
    assert s.l_window_check(b, {"A": 0.2, "gamma": 0.0, "A_inv": 2.0}, {10: rec}, {})["n_l"] == math.inf


def test_n_star_brackets_drop_their_cases(monkeypatch):
    """The brackets of the summary keep their summaries and drop the per-case lists, the crossed-corner
    sensitivity's included."""
    s = _summary()

    def fake(records, *, saving, scheme, sin_theta0, e1_margin):
        return [{"R": 3.0, "metric": "sf", "r": 1.0, "scheme": scheme, "cases": [1, 2], "overall": {"n_star": None},
                 "crossed": {"cases": [3], "overall": {"n_star": None}}}]

    monkeypatch.setattr(scaling, "break_even_sizes", fake)
    fits = {"settings": {"sin_theta0": 1.0, "e1_margin": 1e-3}, "U2": {"gap": []}}
    out = s.n_star_schemes([types.SimpleNamespace(name="h4_r3.0")], fits, {})
    assert [b["scheme"] for b in out] == ["bbht", "aa", "best"]
    assert all("cases" not in b and "cases" not in b["crossed"] and b["crossed"]["overall"] == {"n_star": None}
               for b in out)
    assert out[2]["l_window"]["flag"] == "l_window_unresolved"


def test_markdown_opens_with_the_failed_checks_and_lists_every_scheme():
    """summary.md opens with the failed integration checks and names the families read; the U4 tables carry the
    flags of s*, the extended optimum and the n* cases per margin; the savings table of U9 part 2 lists M3's rule E,
    every scheme of F43 and the sensitivity schemes; Q6's per-budget table gives each accuracy its own tally."""
    s = _summary()
    q6 = {"q": "Q6", "verdict": "v6.", "tables": "U10, U11", "uncertainty": "u6.", "loops": ["free"],
          "route_comparison": {"compiled": None},
          "tallies": {"tight": {"pairs": 2, "best": {"free": {"A": 2}},
                                "fixed_k": {"free": {"2": {"Bk": 2}, "5": {"Bk": 2}, "10": {"A": 1, "Bk": 1},
                                                     "20": {"A": 2}, "60": {"A": 2}}}},
                      "loose": {"pairs": 2, "best": {"free": {"A": 1, "B0": 1}},
                                "fixed_k": {"free": {"2": {"B0": 2}, "5": {"B0": 2}, "10": {"B0": 2},
                                                     "20": {"A": 1, "B0": 1}, "60": {"B0": 1, "A": 1}}}}}}
    ver = {"k": 1, "by_scheme": {"lt_rep": {"mean": 0.1, "ci": [0.05, 0.15], "pays": True}}, "best_k": "rep",
           "best_k1": "rep", "best": {"mean": -0.1, "ci": [-0.2, -0.05]}, "pays": False, "flags_best_k": ["f1"],
           "flags_best_k1": [], "extended": {"best_k": "lt_rep", "best_k1": "lt_rep", "mean": 0.1, "ci": [0.05, 0.15],
                                             "pays": True}}
    margin = {"n_star": [400, 900], "n_none": 2, "n_cases": 36, "n_inside": 5, "p_k": [1e-8, 1e-20]}
    res = {"checks": [{"name": "inputs complete", "passed": False, "detail": "missing: loop_a.json"},
                      {"name": "b", "passed": True, "detail": "-"}],
           "checks_failed": ["inputs complete"], "allow_partial": True, "out": "results/m4",
           "inputs": {"scaling": ["x"], "loop": [], "closedloop": ["c"], "handoff": [], "properties": [], "fits": True,
                      "missing": {"loop": ["a"]}},
           "U0": [{"q": "Q1", "verdict": "v.", "tables": "U1, U2", "uncertainty": "u."}, q6], "qpe_sides": {},
           "U4_schemes": {"rows": [{"system": "x", "metric": "sf", "r": 1.0, "k": 1, "p": 0.5, "c_iter": 0.1,
                                    "mu_rep": 0.2, "rho_rep": 0.5, "schemes": {k: None for k in s.U4_SCHEMES},
                                    "best": {"k": "rep", "k1": "bbht", "mu": 0.3, "rho": 0.3, "pays": True,
                                             "flags_k": ["f1"], "flags_k1": []},
                                    "extended": {"k": "lt_rep", "k1": "lt_rep", "mu": 0.4, "pays": True}}],
                           "n_star": [{"R": 3.0, "metric": "sf", "r": 1.0, "scheme": "best",
                                       "no_window": {"n_star": [500, 1000], "n_none": 0, "n_cases": 18},
                                       "by_margin": {"E": margin, "BL": margin}, "l_window": {"flag": "f"}}]},
           "U9_closed_loop": {"c": {"configs": [{"label": "lagged r = 1", "deciding": True, "stats": [],
                                                 "verdicts": [ver]}]}},
           "U11_closed_loop": {}, "U11_exact": {}, "U12": [], "timings": {"total": 1.0}}
    md = s.markdown(res, "cmd", 1.0)
    lines = md.splitlines()
    assert lines[2] == "FAILED integration checks: inputs complete (permitted by --allow-partial)"
    assert ("from scaling_<s>.json (1 systems) and scaling_fits.json, loop_<s>.json (0), closedloop_<s>.json (1), "
            "handoff_<s>.json (0), properties_<s>.json (0) in results/m4 (missing: loop_a.json)") in md
    head = next(x for x in lines if x.startswith("| system | configuration | iteration |"))
    cols = [c.strip() for c in head.strip("|").split("|")]
    assert cols[3:-5] == ["rep (rule E)", "rep", "aa", "bbht", "fp", "lt", "lt_bbht", "2s", "lt_bbht_L", "2s_L",
                          "lt_rep", "lt_rep_L"]
    row = next(x for x in lines if x.startswith("| c | lagged r = 1 (decides) | Psi1 -> Psi2 |"))
    assert "| 0.1 [0.05, 0.15] pays |" in row
    assert row.rstrip().endswith("| f1 -> - | lt_rep -> lt_rep: 0.1 [0.05, 0.15] pays |")
    u4 = next(x for x in lines if x.startswith("| x | sf | 1 | 1 |"))
    assert u4.rstrip().endswith("| rep -> bbht | 0.3 / 0.3 | yes | f1 -> - | lt_rep -> lt_rep: 0.4, pays |")
    nstar = next(x for x in lines if x.startswith("| 3.0 | sf | 1 | best |"))
    assert ("| 500-1000 (0/18 none) | 400-900 (2/36 none; inside 5/36) | - | 400-900 (2/36 none; inside 5/36) | "
            "1e-20-1e-8 / 1e-20-1e-8 | f |") in nstar
    assert "| p_1(n*) (m_E / m_B / m_BL) |" in md and "'inside k/N'" in md
    # Q6 per budget k = 2, 5, 10, 20, 60 and the record (routes by count, then name)
    assert "| tighter | free | Bk 2 | Bk 2 | A 1, Bk 1 | A 2 | A 2 | A 2 |" in lines
    assert "| looser | free | B0 2 | B0 2 | B0 2 | A 1, B0 1 | A 1, B0 1 | A 1, B0 1 |" in lines
    assert "Over the 2 (system, target) pairs" in md and not any(x.startswith("| tighter, compiled") for x in lines)
    assert md.count("(no closed-loop inputs)") == 1             # the configuration has savings but no T_s(k) rows
    res["U9_closed_loop"] = {}
    assert s.markdown(res, "cmd", 1.0).count("(no closed-loop inputs)") == 2


# ------------------------------------------------------------------------- Q1: windows per gap model
def _overlap_row(R, record, alpha, aitken, n_at=None, **kw) -> dict:
    row = {"R": R, "record": record, "k": 0, "ns": [4, 6, 8], "p": [0.7, 0.57, 0.46], "successive": [alpha, alpha],
           "alpha": alpha, "alpha_aitken": aitken, "holdouts": [], "flags": [], "n_at_p": n_at or [100.0, 90.0]}
    row.update(kw)
    return row


def _gap_row(R, A, gamma, A_inv, margins=(("E", 3e-3), ("B", 3e-3), ("BL", 12e-3))) -> dict:
    return {"R": R, "A": A, "gamma": gamma, "A_inv": A_inv,
            "n_cert": {lab: [A_inv / m, (A / m) ** (1.0 / gamma)] for lab, m in margins}}


def test_q1_states_each_window_per_gap_model_and_the_iterates_rates():
    """'inside' iff the largest n(p = 1e-3) lies within the window under both gap models, 'beyond' iff the smallest
    lies beyond it under both, otherwise per model: at R = 3.0 n = 64.3-66.6 against m_BL's window 55.5 (A'/n) to
    92.1 (power law), so inside under the power-law gap and beyond under A'/n; rule E's 222-529 holds both; at
    R = 4.0 n = 29 lies beyond B_L's 11.3-12.4 under both. The classical-step and R1 Psi_2 rates enter per R."""
    s = _summary()
    A3, g3 = 0.43330958891278015, 0.7928951201909031
    A3_inv = 0.6665498623331274
    assert (A3 / 0.012) ** (1 / g3) == pytest.approx(92.144, abs=1e-3) and A3_inv / 0.012 == pytest.approx(55.546,
                                                                                                         abs=1e-3)
    fits = {"U1": [], "U2": {
        "overlap": [_overlap_row(3.0, "hf", 0.1049, 0.1091, [66.55, 64.32], p1_at_n={}),
                    _overlap_row(3.0, "classical_sf", 0.0392, 0.0434), _overlap_row(3.0, "classical_so", 0.0284,
                                                                                   0.0334),
                    _overlap_row(3.0, "r1_sf", 0.0254, 0.0328), _overlap_row(3.0, "r1_so", 0.0324, 0.0441),
                    _overlap_row(4.0, "hf", 0.2395, 0.2452, [29.41, 28.91], p1_at_n={})],
        "gap": [_gap_row(3.0, A3, g3, A3_inv), _gap_row(4.0, 0.08819, 0.79219, 0.13586)]}}
    v = s.verdict_q1(fits, {})
    text = v["verdict"]
    r3, r4 = text.split("R = 3.0 bohr:")[1].split("R = 4.0 bohr:")
    assert "lies inside the rule-E window" in r3
    assert ("inside or beyond the B_L window depending on the gap model (inside under the power-law gap, beyond "
            "under A'/n)") in r3
    assert "56-92 for kernel-aware B_L with two runs (m_BL) (A'/n 56, power law 92)" in r3
    assert "222-529 for rule E and rule B in the window model (m_E = m_B) (A'/n 222, power law 529)" in r3
    # the rates after the classical step and the first measured iteration, with HF's rate over each (matched rates)
    assert "0.0392-0.0434 (sf; alpha_HF/alpha = 2.5-2.7) and 0.0284-0.0334 (so; alpha_HF/alpha = 3.3-3.7)" in r3
    assert "0.0254-0.0328 (sf; alpha_HF/alpha = 3.3-4.1) and 0.0324-0.0441 (so; alpha_HF/alpha = 2.5-3.2)" in r3
    assert "lies inside the rule-E window and beyond the B_L window" in r4
    assert s.window_position([64.3, 66.6], {"power": [92.1], "inverse": [55.5]}) == ("inside under the power-law gap, "
                                                                                    "beyond under A'/n")
    assert s.window_position([64.3, 66.6], {"power": [529.4], "inverse": [222.2]}) == "inside"
    assert s.window_position([29.0], {"power": [12.4], "inverse": [11.3]}) == "beyond"
    assert s.window_position([60.0, 70.0], {"power": [92.0], "inverse": [65.0]}) == ("inside under the power-law "
                                                                                    "gap, straddling its end under "
                                                                                    "A'/n")
    # per-rate ends of one model (the capped windows): beyond only past the largest end
    assert s.window_position([100, 105], {"power": [200.0], "inverse": [95.0, 110.0]}) == ("inside under the "
                                                                                          "power-law gap, straddling "
                                                                                          "its end under A'/n")
    # the B_L window of HF capped below the two-run overlap floor (per rate)
    g = _gap_row(2.5, 1.0, 0.8, 1.32)
    g["bl_two_run"] = {"below_floor": ["hf"], "n_cert_sound": {"hf": {"inverse": [110.0, 110.0],
                                                                      "power": [174.6, 168.3]}}}
    assert s._gap_ends(g, "BL") == {"power": [174.6, 168.3], "inverse": [110.0, 110.0]}
    assert s.window_position([113.5, 127.4], s._gap_ends(g, "BL")) == ("inside under the power-law gap, beyond "
                                                                      "under A'/n")
    assert s.window_position([113.5, 170.0], s._gap_ends(g, "BL")) == ("straddling its end under the power-law "
                                                                      "gap, beyond under A'/n")


def _growing_gap():
    """F21's fit through Delta_1 = 2 and 2.5 mHa at n = 8 and 10: the power law Delta_1 = 2.5e-4 n (gamma = -1)
    grows with n, A'/n = 0.025/n falls."""
    g = scaling.fit_gap([8, 10], [0.002, 0.0025])
    assert g.gamma == pytest.approx(-1.0, rel=1e-12) and g.A == pytest.approx(2.5e-4, rel=1e-12)
    assert g.A_inv == pytest.approx(0.025, rel=1e-12)
    return g


def test_q1_window_of_a_growing_power_law_gap():
    """A power law that grows with n has the window [m/2.5e-4, inf): 12 for m_E = 3 mHa and 48 for m_BL = 12 mHa,
    where A'/n's windows end at 0.025/m = 8.3 and 2.1. At n(p = 1e-3) = 10 the power-law gap is 2.5 mHa < m_E, so
    p_HF = 1e-3 lies outside both windows (short of the power law's start, beyond A'/n's end), not inside the
    power-law window that ends at inf; at n = 13-14 it lies inside under the power law."""
    s = _summary()
    grow = _growing_gap()
    margins = {"E": 3e-3, "B": 3e-3, "BL": 12e-3}
    g = {"R": 3.0, "A": grow.A, "gamma": grow.gamma, "A_inv": grow.A_inv,
         "n_cert": {lab: list(scaling.certification_window(grow, m)) for lab, m in margins.items()}}
    assert g["n_cert"]["E"] == [pytest.approx(0.025 / 3e-3, rel=1e-12), math.inf]
    fits = {"settings": {"margins": margins}, "U1": [],
            "U2": {"overlap": [_overlap_row(3.0, "hf", 0.69, 0.69, [10.0, 10.0], p1_at_n={})], "gap": [g]}}
    text = s.verdict_q1(fits, {})["verdict"]
    assert ("so p_HF = 1e-3 lies outside the rule-E window (short of its start under the power-law gap, beyond under "
            "A'/n) and outside the B_L window (short of its start under the power-law gap, beyond under A'/n)") in text
    assert "inside under the power-law gap" not in text
    assert "8-inf for rule E and rule B in the window model (m_E = m_B) (A'/n 8, power law from 12 to inf)" in text
    fits["U2"]["overlap"][0]["n_at_p"] = [13.0, 14.0]
    text = s.verdict_q1(fits, {})["verdict"]
    assert ("lies inside or beyond the rule-E window depending on the gap model (inside under the power-law gap, "
            "beyond under A'/n)") in text
    starts = s._gap_starts(g, "E", 3e-3)
    assert starts == {"power": pytest.approx(12.0, rel=1e-12), "inverse": 0.0}
    assert s._gap_starts(g, "E") == {"power": pytest.approx(12.0, rel=1e-12), "inverse": 0.0}   # m = A'/n_cert
    assert s._gap_starts(_gap_row(3.0, 0.4333, 0.7929, 0.6665), "E", 3e-3) == {"power": 0.0, "inverse": 0.0}
    ends = s._gap_ends(g, "E")
    assert s.window_position([10.0], ends, starts) == "short of its start under the power-law gap, beyond under A'/n"
    assert s.window_position([13.0, 14.0], ends, starts) == "inside under the power-law gap, beyond under A'/n"
    assert s.window_position([11.0, 13.0], ends, starts) == ("straddling its start under the power-law gap, beyond "
                                                             "under A'/n")


# ------------------------------------------------------------------------- Q2: n* against the windows per margin
def _margin(n_star, n_cert_inv, n_cert_pow, min_pow, min_inv, n_inside, n_cases=36) -> dict:
    return {"n_star": n_star, "n_none": 0, "n_cases": n_cases, "n_inside": n_inside, "p_k": [1e-8, 1e-20],
            "n_cert": [n_cert_inv, n_cert_pow], "n_cert_by_model": {"power": n_cert_pow, "inverse": n_cert_inv},
            "n_star_min_by_model": {"power": min_pow, "inverse": min_inv}}


def test_q2_windows_per_gap_model_and_the_paying_rows():
    """By margin and gap model: n* 394-1290 against m_E's window 222 (A'/n) / 529 (power law) is gap-model-dependent,
    m_BL's 360-1248 against 56 / 92 is outside under both; the per-bracket text reports both margins (M3's rule E and
    kernel-aware rule B), at the best realizable scheme each with its own numbers (m_E 500-2956, inside under the
    power law from n* = 500 <= 529; m_BL 600-2900, outside under both), and the head states each margin's window.
    The r = 1 rows that pay at s* are the k = 1 rows (with the flags of s*), never k = 0."""
    s = _summary()
    e = _margin([394, 1290], 222.0, 529.0, 400, 394, 6)
    bl = _margin([360, 1248], 55.5, 92.1, 370, 360, 0)
    assert s.window_by_model(e) == "gap-model-dependent: inside under the power-law gap, outside under A'/n"
    assert s.window_by_model(bl) == "outside under both gap models"
    assert s.window_by_model(_margin([100, 300], 222.0, 529.0, 120, 100, 30)) == "inside under both gap models"
    assert s.window_by_model(_margin(None, 222.0, 529.0, None, None, 0)) == "outside under both gap models"
    rep = {"R": 3.0, "metric": "sf", "r": 1.0, "scheme": "rep", "overall": {"n_star": [360, 1290]},
           "no_window": {"n_star": [392, 1226]}, "by_margin": {"E": e, "BL": bl}}
    best = dict(rep, scheme="best", by_margin={"E": _margin([500, 2956], 222.0, 529.0, 500, 698, 1),
                                               "BL": _margin([600, 2900], 55.5, 92.1, 610, 600, 0)},
                overall={"n_star": [500, 2956]}, l_window={"flag": "l_window_closed"})

    def u4(system, k, pays, flags=()):
        return {"system": system, "metric": "sf", "r": 1.0, "k": k, "N": 4, "pays_rep": pays,
                "best": {"k": "rep", "k1": "lt_bbht_L", "pays": pays, "flags_k": [], "flags_k1": list(flags)},
                "extended": {"k": "rep", "k1": "lt_bbht_L", "pays": pays, "relative": 0.01}}

    fits = {"U3": {"rows": [], "exponents": []}, "U4": {"n_star": [rep]}}
    v = s.verdict_q2(fits, [u4("a", 0, True), u4("b", 1, False)], [best])
    assert "pays at the best realizable scheme (r = 1) on no system" in v["verdict"]
    v = s.verdict_q2(fits, [u4("a", 0, True), u4("b", 1, True, ["termination_not_certified"])], [best])
    assert "(r = 1) on b sf (rep -> lt_bbht_L [termination_not_certified])" in v["verdict"]
    text = v["verdict"]
    assert ("394-1290 with m_E of M3's rule E (0/36 none; inside n_cert = 222-529 in 6 of 36 cases, each against the "
            "window of its own gap model; gap-model-dependent: inside under the power-law gap, outside under "
            "A'/n)") in text
    assert ("360-1248 with m_BL of kernel-aware rule B (0/36 none; inside n_cert = 56-92 in 0 of 36 cases, each "
            "against the window of its own gap model; outside under both gap models)") in text
    assert "392-1226 without the lengthened QPE" in text and "[l_window_closed]" in text
    assert ("at the best realizable scheme: 500-2956 with m_E (0/36 none; inside n_cert = 222-529 in 1 of 36 cases, "
            "each against the window of its own gap model; gap-model-dependent: inside under the power-law gap, "
            "outside under A'/n), 600-2900 with m_BL (0/36 none; inside n_cert = 56-92 in 0 of 36 cases, each against "
            "the window of its own gap model; outside under both gap models) [l_window_closed]") in text
    assert ("it pays at the best realizable scheme from n* = 500 on 1 of 1 chain brackets (r = 1); against the window "
            "of kernel-aware rule B (m_BL): outside under both gap models on R = 3.0 sf; against the window of M3's "
            "rule E (m_E): gap-model-dependent: inside under the power-law gap, outside under A'/n on R = 3.0 "
            "sf") in text
    assert "BBHT's schedule (growth 6/5)" in v["uncertainty"] and "lt_rep, lt_rep_L" in v["uncertainty"]


def _growing_bracket(c0: float) -> dict:
    """F45's bracket of the first measured iteration (p_1 = e^-0.06 n, p_2 = e^-0.03 n, c_iter = c0 (n/10)^~10) with
    the growing gap of ``_growing_gap`` and m_E = 3 mHa."""
    ns = np.array([4, 6, 8, 10])
    fk, fk1 = scaling.fit_log_overlap(ns, np.exp(-0.06 * ns)), scaling.fit_log_overlap(ns, np.exp(-0.03 * ns))
    cost = scaling.fit_power_law([6, 8, 10], [c0 / 60.0, c0 / 6.0, c0])
    return scaling.break_even_bracket(fk, fk1, cost, c0, 10, gap=_growing_gap(), margins={"E": 3e-3})


def test_q2_window_of_a_growing_power_law_gap(monkeypatch):
    """Under a growing power-law gap the window end is inf, but a case is inside only where Delta_1(n*) >= m: at
    c0 = 1e-3 every n* is 10, where 2.5e-4 n* = 2.5 mHa < m_E and 0.025/n* = 2.5 mHa < m_E, so no case is inside and
    the bracket lies outside under both gap models; at c0 = 10 every power-law case is inside (n* >= 152) and no A'/n
    case. Counted from the cases in the repetition brackets (Q2's text) and in n_star_schemes' brackets (the head)."""
    s = _summary()

    def gap(c):
        return 2.5e-4 * c["n_star"] if c["gap_model"] == "power" else 0.025 / c["n_star"]

    words = {}
    for c0 in (1e-3, 10.0):
        br = _growing_bracket(c0)
        cases = [c for c in br["cases"] if c["margin"] == "E"]
        want = {g: sum(1 for c in cases if c["gap_model"] == g and c["n_star"] is not None and gap(c) >= 3e-3)
                for g in ("power", "inverse")}
        rep = scaling.to_json(dict(br, R=3.0, metric="sf", r=1.0, scheme="rep"))
        text = s.verdict_q2({"U3": {"rows": [], "exponents": []}, "U4": {"n_star": [rep]}}, [], [])["verdict"]
        words[c0] = (br, want, text)
    br, want, text = words[1e-3]
    assert want == {"power": 0, "inverse": 0}
    assert ("10 with m_E of M3's rule E (0/36 none; inside n_cert = 8-inf in 0 of 36 cases, each against the window of "
            "its own gap model; outside under both gap models)") in text
    assert s.with_inside_by_model(br["by_margin"], br["cases"])["E"]["n_inside_by_model"] == want
    br, want, text = words[10.0]
    assert want == {"power": 18, "inverse": 0}
    assert ("in 18 of 36 cases, each against the window of its own gap model; gap-model-dependent: inside under the "
            "power-law gap, outside under A'/n)") in text
    assert s.with_inside_by_model(br["by_margin"], br["cases"])["E"]["n_inside_by_model"] == want
    # the best realizable scheme's brackets keep the per-model counts after their cases are dropped
    monkeypatch.setattr(scaling, "break_even_sizes", lambda records, *, saving, scheme, sin_theta0, e1_margin: [
        dict(_growing_bracket(1e-3), R=3.0, metric="sf", r=1.0, scheme=scheme)])
    fits = {"settings": {"sin_theta0": 1.0, "e1_margin": 1e-3}, "U2": {"gap": []}}
    out = s.n_star_schemes([types.SimpleNamespace(name="h10_r3.0")], fits, {})
    assert all("cases" not in b and b["by_margin"]["E"]["n_inside_by_model"] == {"power": 0, "inverse": 0}
               and b["crossed"]["by_margin"]["E"]["n_inside_by_model"] == {"power": 0, "inverse": 0} for b in out)
    head = s.verdict_q2({"U3": {"rows": [], "exponents": []}, "U4": {"n_star": []}}, [],
                        [scaling.to_json(b) for b in out])["verdict"]
    assert "against the window of M3's rule E (m_E): outside under both gap models on R = 3.0 sf" in head


# ------------------------------------------------------------------------- Q3: k_ref and the noisy step rules
def _u5_run(rule, metric, k_star, k_p90, c_iter, *, r_plan=None, kind=None) -> dict:
    c_iter = list(c_iter)
    loop = [float(sum(c_iter[:k])) for k in range(len(c_iter) + 1)]
    k_ref = k_star if k_star is not None else k_p90
    kind = kind or ("k_star" if k_star is not None else "p90" if k_p90 is not None else None)
    ref = None if k_ref is None or k_ref > len(c_iter) else loop[k_ref]
    return {"rule": {"name": rule}, "metric": metric, "r_plan": r_plan, "n_steps": 40, "k_star": k_star,
            "k_p90": k_p90, "k_ref": k_ref, "k_ref_kind": kind, "e_err": [0.1, 0.01], "gate_log": [],
            "costs": {"1": {"c_iter_over_qpe": c_iter, "c_loop_over_qpe": loop,
                            "c_loop_k_star_over_qpe": ref if kind == "k_star" else None,
                            "c_loop_k_ref_over_qpe": ref, "precision": []}}}


def test_q3_ranks_at_k_ref_and_flags_unresolved_pairs():
    """k_ref = k*, else the first k with p >= 0.9: a rule that reaches p >= 0.9 at k = 7 within its costs (k <= 21) is
    ranked by C_loop(7); one whose k_p90 = 37 lies beyond its costs is unresolved, and the pair with it is unresolved
    while its C_loop(21) undercuts the best resolved rule; k* and k_p90 rankings are never mixed."""
    s = _summary()
    unit = [1.0] * 21
    runs = [_u5_run("unit", "spin_orbital", None, 7, unit), _u5_run("line_search", "spin_orbital", None, 9,
                                                                    [50.0] * 21)]
    rk = s.u5_ranking(runs)
    assert rk["kind"] == "p90" and rk["costs"] == {"R1": (7, 7.0), "line search": (9, 450.0)}
    assert rk["winners"] == ("R1",) and rk["resolved"]
    runs2 = [_u5_run("unit", "spin_free", None, 37, unit), _u5_run("candidates", "spin_free", None, 16, [100.0] * 21)]
    rk2 = s.u5_ranking(runs2)
    assert rk2["unresolved"] == {"R1": (37, 21, 21.0)} and not rk2["resolved"] and rk2["winners"] == ()
    # a rule at k* outranks every p90 entry: only the k* kind is ranked
    runs3 = [_u5_run("unit", "spin_orbital", 3, 2, [1.0] * 6), _u5_run("line_search", "spin_orbital", None, 4,
                                                                       [0.1] * 21)]
    assert s.u5_ranking(runs3)["costs"] == {"R1": (3, 3.0)}
    # an unresolved rule whose partial cost is already dearer cannot win: resolved
    runs4 = [_u5_run("unit", "spin_free", None, 37, [10.0] * 21), _u5_run("candidates", "spin_free", None, 16,
                                                                         [1.0] * 21)]
    assert s.u5_ranking(runs4)["resolved"] and s.u5_ranking(runs4)["winners"] == ("M3 grid",)
    loop = {"a": {"runs": runs + runs2}}
    v = s.verdict_q3(loop, {})
    assert v["tables"] == "U5, U7"
    assert "R1 reaches k_ref at the lowest measurement cost on 1 of 1 resolved system-metric pairs" in v["verdict"]
    assert "0 of 0 at k*" in v["verdict"] and "1 of 1 at k_p90" in v["verdict"]
    assert "unresolved: a sf" in v["verdict"] and "R1 unresolved (k_p90 = 37 beyond" in v["verdict"]
    assert "no step-rule block" in v["verdict"]


def _cl_rep(p, c_prep, c_loop, *, e_err=None, c_qpe=1.0) -> dict:
    return {"p": list(p), "c_prep": list(c_prep), "c_loop": list(c_loop), "e_err": e_err or [0.1] * len(p),
            "handoff": [{"cost_expected": cl + (cp + c_qpe) / pk} for pk, cp, cl in zip(p, c_prep, c_loop)]}


def test_q3_noisy_step_rules_from_the_closed_loop():
    """U7's step rules: per rule at oracle r = 1 the mean C_loop(k)/C_QPE for k = 1..4 and the share of replicates
    within 1.6 mHa of E_0 by k = 4; R1 from the headline oracle cell."""
    s = _summary()
    c_qpe = 1e6

    def cfg(block, rule, r, loops, errs):
        return {"block": block, "label": f"{rule} oracle r = {r:g}", "deciding": False,
                "config": {"controller": "oracle", "r": r, "rule": rule},
                "replicates": [_cl_rep([0.5] * 5, [0.0] * 5, [x * c_qpe for x in lp], e_err=e, c_qpe=c_qpe)
                               for lp, e in zip(loops, errs)]}
    ok, far = [0.1, 0.01, 1e-3, 1e-3, 1e-3], [0.1] * 5
    data = {"system": {"name": "n", "c_qpe": c_qpe},
            "configs": [cfg("step_rules", "line_search", 1.0, [[0, 10, 20, 30, 40], [0, 12, 22, 32, 42]], [ok, far]),
                        cfg("headline", "unit", 1.0, [[0, 1, 2, 3, 4], [0, 1, 2, 3, 6]], [far, far]),
                        cfg("headline", "unit", 2.0, [[0, 1, 2, 3, 4]] * 2, [far, far])]}
    out = s.step_rules_noisy({"n": data}, target=1.6e-3)
    assert set(out["n"]) == {"1"}
    ls, r1 = out["n"]["1"]["line search"], out["n"]["1"]["R1"]
    assert [x["mean"] for x in ls["c_loop"]] == pytest.approx([11.0, 21.0, 31.0, 41.0])
    assert [x["mean"] for x in r1["c_loop"]] == pytest.approx([1.0, 2.0, 3.0, 5.0])
    assert ls["within"] == 0.5 and r1["within"] == 0.0 and ls["n"] == 2
    v = s.verdict_q3({}, {"n": data})
    assert "n oracle r = 1:" in v["verdict"] and "(cheapest C_loop(4): R1)" in v["verdict"]
    assert "50% within 1.6 mHa" in v["verdict"]


# ------------------------------------------------------------------------- Q4: lagged cells and controls
def _cl_config(block, controller, r, savings, *, deciding=False, non_realizable=(), r1=1.0) -> dict:
    """A closed-loop configuration with one replicate per entry of ``savings``: under repetition with n_v = 1 (C_QPE
    units) T(1) = 2 (p_1 = 0.5, C(Psi_1) = C_loop(1) = 0) and T(2) = 1/p_2 = 2 - saving (C(Psi_2) = C_loop(2) = 0),
    so the first measured iteration saves ``saving``; rep is s* at both states (E_BBHT(p) p > 1 for p > 0.30). The
    realized r is ``r1`` at the first measured iteration (k = 1) and 2 ``r1`` at the second."""
    reps = [_cl_rep([0.25, 0.5, 1.0 / (2.0 - sv), 0.5, 0.5], [0.0] * 5, [0.0, 0.0, 0.0, 1.0, 1.0]) for sv in savings]
    return {"label": f"{controller} r = {r:g}", "block": block, "deciding": deciding, "flags": [],
            "config": {"controller": controller, "r": r, "rule": "unit", "route": "direct_mg", "propagation": "exact"},
            "replicates": reps,
            "summary": {"per_k": [{"r_median": None}, {"r_median": r1}, {"r_median": 2.0 * r1}, {}, {}],
                        "realizable": not non_realizable, "non_realizable": list(non_realizable)}}


def test_q4_reports_the_lagged_sensitivity_the_oracle_bound_and_the_controls():
    """The deciding lagged r = 1 cell decides at s*; the lagged r = 2 cell (a sensitivity, angle cap binding) pays
    where the deciding one does not ("the sensitivity reverses the verdict") and lies below every oracle cell ("a
    lagged cell lies below it"); the scan reports each cell's realized r at its first measured iteration (k = 1, not
    k = 2) and leaves out the lagged r = 1 cell superseded by its angle-cap rerun (which pays); an oracle cell that pays
    never counts; a control system decides with its own lagged r = 1 cell and is reported as a control."""
    s = _summary()
    side = _side(n_v=1, bracket_flags=[], filters={"u": None, "w_l": None, "oracle_u": None})
    superseded = _cl_config("headline", "lagged", 1.0, [0.5, 0.6, 0.55])
    superseded["superseded_by_angle_cap_rerun"] = True
    head = {"system": {"name": "h", "c_qpe": 1.0}, "configs": [
        _cl_config("headline", "oracle", 1.0, [0.2, 0.3, 0.25]),
        _cl_config("headline", "oracle", 3.0, [0.1, 0.12, 0.11]),
        _cl_config("headline", "lagged", 0.5, [-0.5, -0.4, -0.45]),
        _cl_config("headline", "lagged", 1.0, [-0.1, -0.12, -0.11], deciding=True),
        _cl_config("headline", "lagged", 2.0, [0.5, 0.6, 0.55], non_realizable=("angle_cap",), r1=5.4), superseded]}
    ctrl = {"system": {"name": "c", "c_qpe": 1.0}, "configs": [
        _cl_config("controls", "oracle", 1.0, [0.3, 0.3, 0.3]),
        _cl_config("controls", "lagged", 1.0, [-0.2, -0.25, -0.3], deciding=True)]}
    cl_s = {"h": s.closed_loop_schemes(head, side), "c": s.closed_loop_schemes(ctrl, side)}
    dec = cl_s["h"]["configs"][3]
    v1 = next(v for v in dec["verdicts"] if v["k"] == 1)
    assert v1["best_k"] == "rep" and v1["best_k1"] == "rep" and not v1["pays"]
    assert v1["best"]["mean"] == pytest.approx(-0.11, abs=1e-12)
    v = s.verdict_q4(cl_s, {}, {})
    text = v["verdict"]
    assert ("in the deciding cells (lagged r = 1 of the headline block) the first measured iteration pays under "
            "repetition on no system and at the best realizable scheme on no system (of 1)") in text
    assert "reverses the verdict on h (lagged r = 2)" in text and "the sensitivity reverses the verdict" in text
    assert "lagged r = 2: realized r_1 = 5.40" in text and "non-realizable: angle_cap" in text
    assert "realized r_1 = 10.80" not in text and "realized r_1 = 2.00" not in text       # k = 2's realized r
    assert text.count("lagged r = 1 (decides): realized r_1 = 1.00") == 1 and "lagged r = 1: realized" not in text
    assert "h (lagged r = 1)" not in text                                                  # no reversal by it
    assert "oracle r-scan (optimistic over r <= 3)" in text and "a lagged cell lies below it (lagged r = 2" in text
    assert ("in the control cells (lagged r = 1 of the negative and N = 8 controls) it pays under repetition on no "
            "system and at the best realizable scheme on no system (of 1)") in text
    assert "negative control: c (lagged r = 1, 3 replicates)" in text
    # the oracle cell pays but never decides; a deciding cell that pays counts
    head2 = dict(head, configs=[dict(c) for c in head["configs"]])
    head2["configs"][3] = _cl_config("headline", "lagged", 1.0, [0.2, 0.25, 0.3], deciding=True)
    cl2 = {"h": s.closed_loop_schemes(head2, side)}
    t2 = s.verdict_q4(cl2, {}, {})["verdict"]
    assert "pays under repetition on h and at the best realizable scheme on h (of 1)" in t2
    assert "reverses the verdict on h (lagged r = 0.5)" in t2
    # the verdict of record is the one at s*: with the E1lb-referenced filter (c_filt(w_L) = 0.05 C_QPE) s* is
    # lt_bbht_L, and Psi_1 (p = 0.5) -> Psi~_2 (p = 0.6, C_loop(2) = 0.15) pays under repetition, 2 - (0.15 + 1/0.6),
    # but not at s*, 0.05 (E_BBHT(0.5) - E_BBHT(0.6)) - 0.15 < 0
    rep_saving = 2.0 - (0.15 + 1.0 / 0.6)
    star_saving = 0.05 * (_bbht(0.5) - _bbht(0.6)) - 0.15
    assert rep_saving > 0.0 > star_saving
    side_l = _side(n_v=1, c_qpe=1.0, bracket_flags=[], e0_model=-1.0, e1lb=-0.8,
                   filters={"u": None, "w_l": {"c_filt": 0.05, "c_coarse": 0.5}, "oracle_u": None})
    cfg = _cl_config("headline", "lagged", 1.0, [0.0, 0.0, 0.0], deciding=True)
    for rep in cfg["replicates"]:
        rep["p"][2], rep["c_loop"][2] = 0.6, 0.15
        rep["handoff"][2]["cost_expected"] = 0.15 + 1.0 / 0.6
    cl3 = {"h": s.closed_loop_schemes({"system": {"name": "h", "c_qpe": 1.0}, "configs": [cfg]}, side_l)}
    v1 = next(v for v in cl3["h"]["configs"][0]["verdicts"] if v["k"] == 1)
    assert (v1["best_k"], v1["best_k1"]) == ("lt_bbht_L", "lt_bbht_L")
    assert v1["best"]["mean"] == pytest.approx(star_saving, rel=1e-9)
    assert v1["by_scheme"]["rep_rule_e"]["mean"] == pytest.approx(rep_saving, rel=1e-9)
    t3 = s.verdict_q4(cl3, {}, {})["verdict"]
    assert "pays under repetition on h and at the best realizable scheme on no system (of 1)" in t3
    assert "(lt_bbht_L [termination_not_certified] -> lt_bbht_L [termination_not_certified]: does not pay)" in t3


def test_q5_without_a_realizable_scheme_at_a_closed_loop_state():
    """A deciding cell whose states have no realizable scheme (B_L not terminating, no certified U, no L forms) is
    reported as such instead of failing."""
    s = _summary()
    side = _side(n_v=1, rule_b_terminates=False, filters={"u": None, "w_l": None, "oracle_u": None})
    data = {"system": {"name": "h", "c_qpe": 1.0},
            "configs": [_cl_config("headline", "lagged", 1.0, [-0.1, -0.1, -0.1], deciding=True)]}
    cl_s = {"h": s.closed_loop_schemes(data, side)}
    assert cl_s["h"]["configs"][0]["stats"][1]["best"] is None
    v = s.verdict_q5({}, cl_s)
    assert "h closed loop (lagged r = 1): no realizable scheme at Psi_1 or Psi~_2" in v["verdict"]
    assert "no realizable scheme" in s.verdict_q4(cl_s, {}, {})["verdict"]


def _u9_pair(t1: float, t2: float) -> list[dict]:
    """U9 part 1 rows of Psi_1 and Psi_2 (k = 1, 2) with repetition only, T in C_QPE units."""
    return [{"k": k, "costs": {"rep": t}, "costs_over_qpe": {"rep": t}, "scheme_flags": {"rep": []}}
            for k, t in ((1, t1), (2, t2))]


def test_q5_lists_the_paying_systems_per_r():
    """Q5's head lists the systems whose first measured iteration pays at s* (U9 part 1, R1 spin-orbital) at r = 1
    and at r = 3 separately: system a pays at r = 3 only (T 2 -> 1.5, saving 25%), b at neither. U8's HF cost is
    stated with p on psi_0' of H', M3's T6 rows (measured loops) with M3's p on psi_0 of H."""
    s = _summary()

    def data(r1, r3):
        hf = {"state": "HF", "compiled": False, "form": "B_L", "u_label": None, "margin": None, "distance": 5.0,
              "n_v": 2, "n_v_expected": 2, "cost_over_qpe": 9.9, "cost_oracle_e_over_qpe": 8.9, "record": "r1_so",
              "laws": {"sine_2n": {"n_v_runs": {"eps_id": 4e-4}}}}
        return {"u8": [hf], "u9": {"r1_so@r=1.0": _u9_pair(*r1), "r1_so@r=3.0": _u9_pair(*r3)}}

    a = data((2.0, 2.5), (2.0, 1.5))
    a["m3_t6"] = {"families": {"lbfgs": {
        "measured": {"rule_b": {"k": 0, "cost_over_qpe": 3.956}, "m3_certified": {"k": 2, "cost_over_qpe": 570.8}},
        "free": {"rule_b": {"k": 16, "cost_over_qpe": 1.012}, "m3_certified": {"k": 16, "cost_over_qpe": 1.012}}}}}
    v = s.verdict_q5({"a": a, "b": data((2.0, 2.5), (2.0, 2.5))}, {})
    text = v["verdict"]
    assert "pays at the best realizable scheme on no system at r = 1 and on a at r = 3" in text
    assert ("a R1 so r = 3: s* = rep at Psi_1 (T = 2), rep at Psi_2; the first measured iteration pays at s* "
            "(25%)") in text
    assert "a R1 so r = 1: s* = rep at Psi_1 (T = 2), rep at Psi_2; the first measured iteration does not pay" in text
    assert "a HF: B_L with no U, D = 5, n_v = 2, T = 9.9 C_QPE (p on psi_0' of H'; oracle T_E 8.9)" in text
    assert ("a M3 T6 lbfgs/measured (p on psi_0 of H, as M3): rule B k = 0 at 3.956 C_QPE against M3's certified "
            "k = 2 at 570.8") in text
    assert "lbfgs/free" not in text
    assert "U8 and U9 with p on psi_0' of H', M3's T6 rows with p on psi_0 of H as in M3" in v["uncertainty"]


def test_u12_runs_at_their_failure_budgets():
    """U12's hand-off QPE run is laid out at F40's budget of a re-tested run, eps_fail/2 = 5e-4, where the U9 optimum
    is a filtered scheme or verified (n_v = 2), and at eps_fail p_min = 1e-6 for an unverified post-selected one (rep
    with n_v = 1); the
    coherent filter + AA run of lt_bbht_L carries BBHT's expected calls E_BBHT(p) of C(Psi) + c_filt(w_L) on the
    filter's qubits plus the preparation's ancillas, at 5e-4; repetition has no coherent AA run."""
    from thc_engine_cqe.ft import physical

    s = _summary()
    p, c_prep = 0.21252, 2.0e5
    side = _side(system="n2_r2.4", logical_qubits=185, n_iter=11924, norb=6,
                 filters={"u": None, "w_l": {"c_filt": 1.33e6, "c_coarse": 2.19e6, "filter_qubits": 218},
                          "oracle_u": None})

    def hand(best, n_v):
        return {"u9": {"r1_so@r=1.0": [{"best_realizable": best, "costs": {"rep": 5e7, "lt_bbht_L": 1.5e7},
                                         "n_v": n_v, "k": 1, "p_model": p, "record": "r1_so", "c_prep": c_prep,
                                         "state": "Psi1", "sources": {}}]},
                "records": {"r1_so": {"prep_qubits": [12, 14, 14]}}}

    model = physical.SurfaceCodeModel()
    rows = s.u12_rows(None, {}, {"n2_r2.4": hand("lt_bbht_L", 2)}, {}, {"n2_r2.4": side}, FTParameters(), model)
    qpe, aa = rows
    assert qpe["fail_budget"] == pytest.approx(1e-3 / 2, rel=1e-12) and qpe["toffolis"] == 1e7
    assert qpe["run"] == "hand-off QPE run (final, F40 retested)" and qpe["logical_qubits"] == 185
    assert aa["toffolis"] == pytest.approx(_bbht(p) * (c_prep + 1.33e6), rel=1e-9)
    assert aa["fail_budget"] == pytest.approx(5e-4, rel=1e-12) and aa["logical_qubits"] == 218 + 14 - 12
    rows = s.u12_rows(None, {}, {"n2_r2.4": hand("rep", 1)}, {}, {"n2_r2.4": dict(side, n_v=1)}, FTParameters(),
                      model)
    assert rows[0]["fail_budget"] == pytest.approx(1e-3 * 1e-3, rel=1e-12)
    assert rows[0]["run"] == "hand-off QPE run (final, F40 unverified)"
    assert rows[1]["run"] == "U9 optimum rep (Psi1): no coherent AA run"
    # repetition with rule B's verification run (n_v = 2) re-tests the state
    rows = s.u12_rows(None, {}, {"n2_r2.4": hand("rep", 2)}, {}, {"n2_r2.4": side}, FTParameters(), model)
    assert rows[0]["fail_budget"] == pytest.approx(1e-3 / 2, rel=1e-12)


# ------------------------------------------------------------------------- Q6
def _prow(system, target, eps, route, k, toffolis, *, loop, run=1e8, q=200, flags=(), bias=1e-5) -> P.PropertyRow:
    return P.PropertyRow(system=system, target=target, eps=eps, route=route, estimator="ae", loop=loop, k=k,
                         bias=bias, toffolis=toffolis, loop_toffolis=0.0, per_prep=1.0, n_prep=1.0, longest_run=run,
                         logical_qubits=q, flags=tuple(flags))


def _rjson(row: P.PropertyRow) -> dict:
    out = scaling.to_json(row)
    out["toffolis_over_qpe"] = scaling.to_json(row.toffolis)
    out["selectable"] = bool(row.selectable)
    return out


def _target_block(system, *, tight_a_k, loose_route_a: bool, bk_run=1e8) -> dict:
    """One rdm1 target at eps = 1e-3 (tighter) and 1e-2 (looser): at the tighter accuracy route A (1 Toffoli) is
    usable only at k = tight_a_k against Bk (5 Toffolis, longest run ``bk_run`` on 261 qubits) and B0 (8); at the looser
    accuracy route A wins at every budget if ``loose_route_a``, else Bk."""
    blk = {"accuracies": [1e-2, 1e-3], "floors": {"r1_so": {"min": 1e-6}, "compiled_sliced": {"min": 2e-3}},
           "rows": {}, "best": {}}
    for eps, kind in ((1e-3, "tight"), (1e-2, "loose")):
        ra, rb, best = [], [], {}
        for lp in ("free", "meas_r1", "meas_r03"):
            for k in (2, 5, 10, 20, 60):
                usable = (k == tight_a_k) if kind == "tight" else loose_route_a
                ra.append(_prow(system, "rdm1", eps, "A", k, 1.0 if usable else 10.0, loop=lp,
                                flags=() if usable else ("not_usable",), bias=1e-5 if usable else 1.0))
            rb.append(_prow(system, "rdm1", eps, "Bk", 3, 5.0, loop=lp, run=bk_run, q=261))
            rb.append(_prow(system, "rdm1", eps, "B0", 1, 8.0, loop=lp))
            pool = [r for r in ra + rb if r.loop == lp and r.selectable]
            b = min(pool, key=lambda r: r.toffolis)
            best[lp] = {"ae": {"route": b.route, "estimator": "ae", "k": b.k, "toffolis_over_qpe": b.toffolis,
                               "bias": b.bias, "flags": list(b.flags)}, "shadow": None}
        blk["rows"][str(eps)] = {"route_a": [_rjson(r) for r in ra], "route_b": [_rjson(r) for r in rb],
                                 "route_a_compiled": None}
        blk["best"][str(eps)] = best
    return blk


def test_q6_tallies_both_accuracies_each_budget_and_the_layouts():
    """The record tally (route A at its cheapest usable budget per pair) is taken at the tighter accuracy: route A on
    both pairs; at single budgets it wins on one pair at k = 10 and one at k = 60 only; at the looser accuracy route A
    wins on pair 1 and Bk on pair 2, which the list of route changes names (per loop). At the looser accuracy pair 2's
    best row (Bk, longest run 3.95e11 Toffolis on 261 qubits) has no layout at p_fail_ae: it is counted, and the tally
    restricted to rows with a layout takes the next row (B0). The compiled iterates' sliced floor above eps on both
    pairs makes route A bias-limited there, which Q6 states with both floors."""
    s = _summary()
    pj = {"p1": {"system": "p1", "targets": {"rdm1": _target_block("p1", tight_a_k=10, loose_route_a=True)}},
          "p2": {"system": "p2", "targets": {"rdm1": _target_block("p2", tight_a_k=60, loose_route_a=False,
                                                                   bk_run=3.95e11)}}}
    v = s.verdict_q6(pj, {})
    t = v["tallies"]
    for lp in ("free", "meas_r1", "meas_r03"):
        assert t["tight"]["best"][lp] == {"A": 2}
        assert {k: x.get("A", 0) for k, x in t["tight"]["fixed_k"][lp].items()} == {"2": 0, "5": 0, "10": 1, "20": 0,
                                                                                     "60": 1}
        assert t["loose"]["best"][lp] == {"A": 1, "Bk": 1}
    assert v["loops"] == ["free", "meas_r1", "meas_r03"]
    assert sorted(v["accuracy_changes"]) == sorted(f"p2 rdm1 {lp} (A at 0.001, Bk at 0.01)"
                                                   for lp in ("free", "meas_r1", "meas_r03"))
    text = v["verdict"]
    assert "with a free loop: A 2; with a measured r = 1 loop: A 2; with a measured r = 0.3 loop: A 2" in text
    assert "at the looser accuracy, with a free loop: A 1, Bk 1" in text
    assert "single pre-declared budget k = 2 / 5 / 10 / 20 / 60" in text
    assert "0 / 0 / 1 / 0 / 1 with a free loop" in text
    assert "0 / 0 / 1 / 0 / 1 with a measured r = 0.3 loop (2 with the cheapest usable budget per pair" in text
    # layouts: at the tighter accuracy the best rows are route A (laid out); the looser accuracy's Bk row has none
    lay = s.q6_layouts(P.property_pairs(pj, accuracy="loose"), ("free",))
    assert lay["best_rows"] == 2 and [(x["system"], x["route"]) for x in lay["no_layout"]] == [("p2", "Bk")]
    assert lay["tally"]["free"] == {"A": 1, "B0": 1}
    assert "0 of the 6 best rows (AE, tighter accuracy) have no surface-code layout" in text
    assert "the record tally picks k per pair with the exact bias" in v["uncertainty"]
    # compiled iterates: the F25-sliced floor (2e-3) lies above the tighter eps while the exact R1 floor (1e-6) does
    # not, on both pairs; each pair's cell shows both floors; the slice rule bounds the state error only
    assert "route A is bias-limited on the compiled iterates although the exact R1 floor reaches eps on 2 pairs" in text
    assert "p1: rdm1 at 0.001 / 0.01: free A 1 / A 1, r = 1 loop A 1 / A 1 (R1 floor 1.0e-6, sliced 2.0e-3)" in text
    assert "p2: rdm1 at 0.001 / 0.01: free A 1 / Bk 5, r = 1 loop A 1 / Bk 5 (R1 floor 1.0e-6, sliced 2.0e-3)" in text
    assert ("F25's slice rule (eta_s = 0.05, s <= 16) controls a sliced step's state error, not the property error of "
            "the compiled iterate") in v["uncertainty"]


# ------------------------------------------------------------------------- inputs, U0 rows, checks, exit status
def test_missing_inputs_give_not_computed_rows_and_a_failed_check():
    """With the loop and closed-loop JSON missing, U0 keeps six rows: Q3 and Q4 'not computed: no <prefix>_<s>.json
    in <out>', Q5 states its closed-loop part as not computed, and "inputs complete" fails listing the files; a
    single missing system is 'not computed for <s>'."""
    s = _summary()
    required = {"scaling": [], "loop": ["a", "b"], "closedloop": ["a"], "handoff": ["h"], "properties": []}
    found = {"scaling": {}, "loop": {}, "closedloop": {}, "handoff": {"h": {}}, "properties": {}}
    missing = s.missing_inputs(found, False, required)
    assert missing == {"scaling": [], "loop": ["a", "b"], "closedloop": ["a"], "handoff": [], "properties": [],
                       "fits": []}
    chk = s.inputs_check(missing)
    assert chk["name"] == "inputs complete" and not chk["passed"]
    assert chk["detail"] == "missing: loop_a.json, loop_b.json, closedloop_a.json"
    assert s.missing_inputs(found, False, dict(required, scaling=["x"]))["fits"] == ["fits"]
    assert s.inputs_check(s.missing_inputs({"loop": {"a": 1}}, True, {"loop": ["a"]}))["passed"]
    hand = {"u8": [], "u9": {}}
    rows = s.assemble_u0({"fits": None, "scaling": {}, "loop": {}, "closedloop": {}, "handoff": {"h": hand},
                          "properties": {}}, {}, missing, "results/m4")
    assert [r["q"] for r in rows] == ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
    by_q = {r["q"]: r["verdict"] for r in rows}
    assert by_q["Q1"] == by_q["Q2"] == "not computed: no scaling_fits.json in results/m4."
    assert by_q["Q3"] == "not computed: no loop_<s>.json in results/m4."
    assert by_q["Q4"] == "not computed: no closedloop_<s>.json in results/m4."
    assert by_q["Q6"] == "not computed: no properties_<s>.json in results/m4."
    assert "the part from closedloop_<s>.json not computed: no closedloop_<s>.json in results/m4" in by_q["Q5"]
    assert {r["q"]: r["tables"] for r in rows}["Q3"] == "U5, U7"
    # one of two loop systems missing: Q3 is computed on the other and names the missing one
    loop = {"a": {"runs": [_u5_run("unit", "spin_orbital", 3, 2, [1.0] * 6)]}}
    rows2 = s.assemble_u0({"fits": None, "loop": loop}, {}, {"loop": ["b"]}, "out")
    q3 = next(r for r in rows2 if r["q"] == "Q3")["verdict"]
    assert q3.startswith("the unit rule R1") and "not computed for b: no loop_b.json in out" in q3


def test_required_inputs_follow_the_producing_scripts():
    """The required sets are the producing scripts' default (quick) system sets, restricted by --systems."""
    s = _summary()
    quick = s.required_inputs(True)
    assert quick["scaling"] == list(scaling.QUICK) and quick["closedloop"] == ["n2_r2.4", "h4_r4.0"]
    assert quick["loop"] == ["h4_r1.8", "h4_r4.0", "n2_r2.0", "n2_r2.4"]
    assert quick["handoff"] == ["h4", "n2s", "n2_r2.4", "n2_r2.8"] and quick["properties"] == ["h4_r1.8", "lih",
                                                                                              "n2_r2.0"]
    full = s.required_inputs(False)
    assert len(full["scaling"]) == 25 and set(full["closedloop"]) == {
        "n2_r2.0", "n2_r2.4", "n2_r2.8", "h4_r4.0", "h6_r3.0", "h4_r1.8", "n2_r1.098", "n2v_r2.4", "n2v_r2.8",
        "h8_r3.0"}
    assert len(full["loop"]) == 17 and len(full["handoff"]) == 21 and full["properties"] == list(scaling.PROPERTIES)
    assert s.required_inputs(False, {"n2_r2.4", "zz"}) == {"scaling": ["n2_r2.4"], "loop": ["n2_r2.4"],
                                                           "closedloop": ["n2_r2.4"], "handoff": ["n2_r2.4"],
                                                           "properties": ["n2_r2.4"]}


def test_exit_status_and_the_failed_checks_at_the_top():
    """Exit status 1 iff a check fails; with --allow-partial the check "inputs complete" alone does not fail the run.
    A run without its inputs writes six 'not computed' rows and opens summary.md with the failed checks."""
    s = _summary()
    ok, bad = {"name": "a", "passed": True}, {"name": "b", "passed": False}
    inc = {"name": "inputs complete", "passed": False}
    assert s.exit_status([ok]) == 0 and s.exit_status([ok, bad]) == 1 and s.exit_status([inc]) == 1
    assert s.exit_status([inc], allow_partial=True) == 0 and s.exit_status([inc, bad], allow_partial=True) == 1


def test_main_without_inputs(tmp_path):
    s = _summary()
    out = tmp_path / "m4"
    out.mkdir()
    argv = ["--quick", "--systems", "h4_r3.0", "--out", str(out)]
    assert s.main(argv) == 1
    md = (out / "summary.md").read_text().splitlines()
    assert md[2] == "FAILED integration checks: inputs complete"
    res = json.loads((out / "summary.json").read_text())
    assert res["checks_failed"] == ["inputs complete"]
    assert res["inputs"]["missing"]["scaling"] == ["h4_r3.0"] and res["inputs"]["missing"]["fits"] == ["fits"]
    assert [r["q"] for r in res["U0"]] == ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
    assert all(r["verdict"].startswith("not computed: no ") for r in res["U0"])
    assert s.main(argv + ["--allow-partial"]) == 0
    md = (out / "summary.md").read_text().splitlines()
    assert md[2] == "FAILED integration checks: inputs complete (permitted by --allow-partial)"
    assert "--allow-partial" in (out / "summary.md").read_text()


def test_main_requires_the_quick_sets_under_quick(tmp_path):
    """--quick requires the producing scripts' quick sets: h4_r2.5 (in the full size axis and U5's default set, in no
    quick set) is required without --quick (scaling, loop and the fits), not with it."""
    s = _summary()
    loop = s._sibling("run_m4_loop")
    assert "h4_r2.5" in scaling.SIZE_AXIS and "h4_r2.5" in loop.U5_SYSTEMS
    assert "h4_r2.5" not in scaling.QUICK and "h4_r2.5" not in loop.U5_QUICK
    out = tmp_path / "m4"
    out.mkdir()
    assert s.main(["--quick", "--systems", "h4_r2.5", "--out", str(out)]) == 0
    res = json.loads((out / "summary.json").read_text())
    assert res["checks_failed"] == [] and not any(res["inputs"]["missing"].values())
    assert s.main(["--systems", "h4_r2.5", "--out", str(out)]) == 1
    res = json.loads((out / "summary.json").read_text())
    assert res["checks_failed"] == ["inputs complete"]
    assert res["inputs"]["missing"] == {"scaling": ["h4_r2.5"], "loop": ["h4_r2.5"], "closedloop": [], "handoff": [],
                                        "properties": [], "fits": ["fits"]}


# ------------------------------------------------------------------------- the summary's own cache stages
class _Recorder(scaling.Cache):
    """A cache that records the key of every stage and stops at ``stop``."""

    def __init__(self, root, stop: str):
        super().__init__(root)
        self.keys: dict = {}
        self.stop = stop

    def get(self, name, stage, compute, *, key=None):
        self.keys[stage] = key
        if stage == self.stop:
            raise LookupError(stage)
        return compute()


def test_own_stages_are_keyed_by_the_script_digest(tmp_path):
    """Every stage the summary caches itself (model spectrum, CISD, property inputs, deep runs) carries the SHA-256
    of the script in its key, so that an edit of the script recomputes it."""
    from thc_engine_cqe.diagnostics.thc_floors import ExactReference
    from thc_engine_cqe.ft import qpe as ftqpe
    from thc_engine_cqe.ft import symmetry

    s = _summary()
    digest = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert s.SCRIPT_DIGEST == digest and s._own_key(a=1) == {"a": 1, "script": digest}
    active = symmetry.benchmark_system("h4")
    irreps = symmetry.orbital_irreps(active)
    spec = scaling.spectrum(active, irreps, n_roots=2)
    q = ftqpe.qpe_cost(ftqpe.sf_encoding(active, 7, reference=ExactReference.build(active)), 1e-3)
    parts = {"active": active, "irreps": irreps, "qpe": q, "spectrum": spec}
    rec = _Recorder(tmp_path / "c1", stop="")
    s.qpe_side("h4", parts, rec, FTParameters())
    assert set(rec.keys) == {"model_spectrum", "cisd"} and all(k["script"] == digest for k in rec.keys.values())
    up = types.SimpleNamespace(inputs=lambda name: types.SimpleNamespace(active=active, spectrum=spec, irreps=irreps,
                                                                         qpe=q))
    rec2 = _Recorder(tmp_path / "c2", stop="summary_deep_runs")
    with pytest.raises(LookupError):
        s.deep_runs("h4", up, rec2, FTParameters(), n=1, max_iter=1)
    assert set(rec2.keys) == {"properties_inputs", "summary_deep_runs"}
    assert all(k["script"] == digest for k in rec2.keys.values())
    assert dataclasses.asdict(FTParameters()) == rec2.keys["summary_deep_runs"]["params"]
