"""Tests of benchmarks/run_m4_loop.py (U5, U6, U7).

The pre-declared deciding cell of the closed loop (lagged r = 1 of the headline block) and the rule that reruns it
with the emulation-only angle cap off if the cap binds in any replicate (the rerun decides, the original is kept and
marked superseded); U5's calculations (one per system, metric and rule, R3 once per planning r, each timed) and the
per-system outputs, which a later failure keeps, their provenance and --resume; and the U5, U6 and U7 markdown (the
k = 1 and k_ref - 1 columns of U5, rho_E per r, the superseded and non-realizable marks of U6 and U7). The script is
loaded from its file; its expensive parts are replaced by recording stubs where only the orchestration is under
test.
"""

import dataclasses
import hashlib
import importlib.util
import json
import pathlib

import pytest

from thc_engine_cqe.ft import closedloop as cl
from thc_engine_cqe.ft import scaling
from thc_engine_cqe.ft import steprules as fsr
from thc_engine_cqe.ft.closedloop import LoopSpec
from thc_engine_cqe.ft.primitives import FTParameters
from thc_engine_cqe.steprules import StepRule

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "benchmarks" / "run_m4_loop.py"


@pytest.fixture
def bench():
    """benchmarks/run_m4_loop.py as a module."""
    spec = importlib.util.spec_from_file_location("run_m4_loop", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------- the deciding cell
def test_deciding_cell_is_predeclared(bench):
    """The headline block (100 replicates on n2_r2.0, n2_r2.4, n2_r2.8, h4_r4.0) holds exactly one lagged r = 1 cell,
    the pre-declared deciding cell: R1 (unit) spin-orbital, direct_mg, exact propagation, with noise."""
    systems, configs, n = bench.blocks(False)["headline"]
    assert n == 100 and systems == ("n2_r2.0", "n2_r2.4", "n2_r2.8", "h4_r4.0")
    cells = [c for c in configs if (c[0].controller, c[0].r) == ("lagged", 1.0)]
    assert len(cells) == 1
    spec, rule, metric = cells[0]
    assert (rule, metric, spec.route, spec.propagation, spec.noise) == ("unit", "spin_orbital", "direct_mg", "exact",
                                                                       True)
    assert bench.DECIDING == ("lagged", 1.0)
    quick_systems, quick_configs, quick_n = bench.blocks(True)["headline"]
    assert quick_n == 20 and quick_configs == configs and set(quick_systems) <= set(systems)


def test_angle_cap_rerun_decides(bench, tmp_path, monkeypatch):
    """With the angle cap forced to bind in the deciding cell (cap 0.05 on h4_r4.0, 3 replicates) the cell is rerun
    with the cap off: the rerun ("angle cap off lagged r = 1", flag "angle_cap_rerun", realizable) is the only deciding
    configuration; the original stays, not deciding, marked superseded (the cap bound in all 3 replicates; angle_cap
    among its non-realizable parts), and U6 labels it "(superseded)"; the oracle cell never decides. A compiled cell
    with the F25 slice rule is non-realizable ("slice_rule"), which U7's flags column shows. closedloop_<s>.json is
    the returned block."""
    monkeypatch.setattr(bench, "blocks", lambda quick: {
        "headline": (("h4_r4.0",), [(LoopSpec(controller="oracle", r=1.0), "unit", "spin_orbital"),
                                    (LoopSpec(controller="lagged", r=1.0), StepRule("unit", angle_cap=0.05),
                                     "spin_orbital")], 3),
        "compiled": (("h4_r4.0",), [(LoopSpec(controller="lagged", r=1.0, propagation="compiled"),
                                     StepRule("unit", angle_cap=None), "spin_orbital")], 2)})
    out_all = bench.run_closed_loop(None, False, None, scaling.Cache(tmp_path / "cache"), FTParameters(),
                                    out_dir=str(tmp_path))
    out = out_all["h4_r4.0"]
    assert json.loads((tmp_path / "closedloop_h4_r4.0.json").read_text()) == out
    deciding = [c for c in out["configs"] if c["deciding"]]
    assert len(deciding) == 1
    rerun = deciding[0]
    assert rerun["flags"] == ["angle_cap_rerun"] and rerun["label"] == "angle cap off lagged r = 1"
    assert rerun["block"] == "headline" and rerun["n"] == 3
    assert rerun["config"]["step_rule"]["angle_cap"] is None
    assert rerun["summary"]["realizable"] and "angle_cap" not in rerun["summary"]["flag_counts"]
    original = [c for c in out["configs"] if c.get("superseded_by_angle_cap_rerun")]
    assert len(original) == 1 and not original[0]["deciding"] and original[0]["label"] == "lagged r = 1"
    assert original[0]["summary"]["flag_counts"]["angle_cap"] == 3
    assert "angle_cap" in original[0]["summary"]["non_realizable"]
    oracle = [c for c in out["configs"] if c["config"]["controller"] == "oracle"]
    assert len(oracle) == 1 and not oracle[0]["deciding"]
    compiled = [c for c in out["configs"] if c["block"] == "compiled"]
    assert len(compiled) == 1 and not compiled[0]["deciding"]
    assert compiled[0]["summary"]["non_realizable"] == ["slice_rule"] and not compiled[0]["summary"]["realizable"]
    lines = bench.closed_loop_tables(out_all)
    u6 = lines[:lines.index("## U7. Closed-loop ablations")]
    u7 = lines[lines.index("## U7. Closed-loop ablations"):]
    assert sum("| lagged r = 1 (superseded) |" in row for row in u6) == 1
    assert sum("| angle cap off lagged r = 1 (decides) |" in row for row in u6) == 1
    assert sum("| oracle r = 1 |" in row for row in u6) == 1
    (row,) = [x for x in u7 if x.startswith("| compiled | h4_r4.0 | compiled angle cap off lagged r = 1 |")]
    assert row.rstrip(" |").endswith("non-realizable: slice_rule")


def test_control_blocks_have_their_own_deciding_cell(bench, tmp_path, monkeypatch):
    """A control system decides with the lagged r = 1 cell of its block (R1 spin-orbital, direct_mg, exact
    propagation) under the headline's rerun rule: with the angle cap forced to bind (cap 0.05 on h4_r4.0 as a
    control, 2 replicates) the cap-off rerun is its only deciding cell and the original is superseded; the oracle
    cell, a lagged r = 1 cell on another route and the cells of the other blocks never decide. The predicate holds
    in the headline and both control blocks only, each of which has one such cell."""
    full = bench.blocks(False)
    for block in ("headline", "controls", "n8_controls"):
        assert sum(bench.is_deciding(block, *c) for c in full[block][1]) == 1
    monkeypatch.setattr(bench, "blocks", lambda quick: {
        "controls": (("h4_r4.0",), [(LoopSpec(controller="oracle", r=1.0), "unit", "spin_orbital"),
                                    (LoopSpec(controller="lagged", r=1.0), StepRule("unit", angle_cap=0.05),
                                     "spin_orbital"),
                                    (LoopSpec(route="ae_x", controller="lagged", r=1.0), "unit", "spin_orbital")], 2),
        "routes": (("h4_r4.0",), [(LoopSpec(controller="lagged", r=1.0), "unit", "spin_orbital")], 2)})
    out = bench.run_closed_loop(None, False, None, scaling.Cache(tmp_path / "cache"), FTParameters())["h4_r4.0"]
    deciding = [c for c in out["configs"] if c["deciding"]]
    assert len(deciding) == 1
    (rerun,) = deciding
    assert rerun["block"] == "controls" and rerun["flags"] == ["angle_cap_rerun"]
    assert rerun["label"] == "angle cap off lagged r = 1" and rerun["config"]["step_rule"]["angle_cap"] is None
    (original,) = [c for c in out["configs"] if c.get("superseded_by_angle_cap_rerun")]
    assert original["block"] == "controls" and original["summary"]["flag_counts"]["angle_cap"] > 0
    assert not any(c["deciding"] for c in out["configs"] if c["block"] == "routes" or c["config"]["route"] == "ae_x"
                   or c["config"]["controller"] == "oracle")
    unit = StepRule("unit")
    lagged = LoopSpec(controller="lagged", r=1.0)
    for block in ("headline", "controls", "n8_controls"):
        assert bench.is_deciding(block, lagged, "unit", "spin_orbital")
        assert bench.is_deciding(block, lagged, unit, "spin_orbital")
        assert not bench.is_deciding(block, lagged, "unit", "spin_free")
        assert not bench.is_deciding(block, lagged, "hybrid", "spin_orbital")
        assert not bench.is_deciding(block, LoopSpec(controller="lagged", r=2.0), "unit", "spin_orbital")
        assert not bench.is_deciding(block, LoopSpec(controller="oracle", r=1.0), "unit", "spin_orbital")
        assert not bench.is_deciding(block, LoopSpec(controller="lagged", r=1.0, propagation="compiled"), "unit",
                                     "spin_orbital")
        assert not bench.is_deciding(block, LoopSpec(controller="lagged", r=1.0, route="best"), "unit",
                                     "spin_orbital")
    for block in ("routes", "step_rules", "compiled", "metric"):
        assert not bench.is_deciding(block, lagged, "unit", "spin_orbital")


# ------------------------------------------------------------------------- U5 orchestration
def _u5_stub(calls, fail_on=(), caches=None):
    """A recording stand-in of ft.steprules.run_step_rule_system: one run per call, shaped like its output."""
    def stub(name, rules, *, metrics, r_values, max_iter, cache, params):
        if name in fail_on:
            raise RuntimeError(f"{name} failed")
        (rule,), (metric,) = rules, metrics
        calls.append((name, metric, rule.name, tuple(r_values), max_iter))
        if caches is not None:
            caches.append(cache)
        r_plan = r_values[0] if rule.name == "hybrid" else None
        return {"system": {"name": name},
                "runs": [{"rule": dataclasses.asdict(rule), "metric": metric, "r_plan": r_plan, "k_star": 3,
                          "tag": len(calls)}]}
    return stub


def test_u5_one_calculation_per_metric_rule_and_r(bench, tmp_path, monkeypatch):
    """run_u5 computes a system's inputs once (refreshed with --no-cache, then read from the cache by every
    calculation) and makes one run_step_rule_system call per metric and rule and one per planning r for R3 (40
    steps, 30 at N = 8), keeps the order of the runs (per metric: line_search, candidates, unit, hybrid at r = 0.3, 1,
    3), records every calculation's seconds and the inputs' seconds, and writes loop_<s>.json as each system
    finishes."""
    calls, caches, inputs = [], [], []
    monkeypatch.setattr(fsr, "run_step_rule_system", _u5_stub(calls, caches=caches))
    monkeypatch.setattr(cl, "system_inputs", lambda name, *, cache, params: inputs.append((name, cache)))
    refreshing = scaling.Cache(tmp_path / "cache", refresh=True)
    out = bench.run_u5(["h4_r1.8", "h8_r3.0"], refreshing, FTParameters(), out_dir=str(tmp_path))
    assert inputs == [("h4_r1.8", refreshing), ("h8_r3.0", refreshing)]
    assert len(caches) == 24 and all(not c.refresh and c.root == refreshing.root for c in caches)
    r_all = (0.3, 1.0, 3.0)
    order = [("line_search", r_all), ("candidates", r_all), ("unit", r_all), ("hybrid", (0.3,)), ("hybrid", (1.0,)),
             ("hybrid", (3.0,))]
    want = [(name, metric, rule, rs, steps) for name, steps in (("h4_r1.8", 40), ("h8_r3.0", 30))
            for metric in ("spin_orbital", "spin_free") for rule, rs in order]
    assert calls == want
    for name in ("h4_r1.8", "h8_r3.0"):
        block = out[name]
        runs = block["runs"]
        assert [(r["metric"], r["rule"]["name"], r["r_plan"]) for r in runs] == \
            [(m, rule, rs[0] if rule == "hybrid" else None) for n, m, rule, rs, _steps in want if n == name]
        assert [r["tag"] for r in runs] == sorted(r["tag"] for r in runs)
        assert all(r["runtime_s"] >= 0.0 for r in runs) and block["inputs_runtime_s"] >= 0.0
        assert block["runtime_s"] >= block["inputs_runtime_s"] + sum(r["runtime_s"] for r in runs)
        assert json.loads((tmp_path / f"loop_{name}.json").read_text()) == block


def test_finished_systems_survive_a_later_failure(bench, tmp_path, monkeypatch):
    """A failing U5 or closed-loop calculation propagates, and the systems finished before it keep their
    loop_<s>.json or closedloop_<s>.json."""
    calls = []
    monkeypatch.setattr(fsr, "run_step_rule_system", _u5_stub(calls, fail_on=("h4_r4.0",)))
    monkeypatch.setattr(cl, "system_inputs", lambda name, *, cache, params: None)
    with pytest.raises(RuntimeError, match="h4_r4.0 failed"):
        bench.main(["--systems", "h4_r1.8,h4_r4.0", "--skip-closed-loop", "--out", str(tmp_path / "u5")])
    assert (tmp_path / "u5" / "loop_h4_r1.8.json").exists() and not (tmp_path / "u5" / "loop_h4_r4.0.json").exists()

    def loop_stub(name, configs, *, n, max_iter, cache, params):
        if name == "h4_r4.0":
            raise RuntimeError(f"{name} failed")
        return {"system": {"name": name},
                "configs": [{"config": dict(dataclasses.asdict(spec), rule=rule, metric=metric, step_rule=None),
                             "summary": {"flag_counts": {}}} for spec, rule, metric in configs]}

    monkeypatch.setattr(cl, "run_closed_loop_system", loop_stub)
    with pytest.raises(RuntimeError, match="h4_r4.0 failed"):
        bench.main(["--quick", "--skip-u5", "--out", str(tmp_path / "cl")])
    first = json.loads((tmp_path / "cl" / "closedloop_n2_r2.4.json").read_text())
    assert [c["label"] for c in first["configs"]] == ["oracle r = 1", "oracle r = 2", "oracle r = 3", "lagged r = 0.5",
                                                       "lagged r = 1", "lagged r = 2"]
    assert not (tmp_path / "cl" / "closedloop_h4_r4.0.json").exists()


# ------------------------------------------------------------------------- provenance and --resume
def _u5_full_stub(calls):
    """A recording stand-in of ft.steprules.run_step_rule_system whose runs carry every field U5's table reads."""
    def stub(name, rules, *, metrics, r_values, max_iter, cache, params):
        (rule,), (metric,) = rules, metrics
        calls.append(name)
        r_plan = r_values[0] if rule.name == "hybrid" else None
        rs = ["0.3", "1", "3"] if r_plan is None else [f"{r_plan:g}"]
        run = dict(_u5_run(rule.name, r_plan, 3, 3, "k_star", rs), metric=metric, rule=dataclasses.asdict(rule))
        return {"system": {"name": name}, "runs": [run]}
    return stub


def _cl_stub(calls):
    """A recording stand-in of ft.closedloop.run_closed_loop_system with the summary fields U6 reads."""
    def stub(name, configs, *, n, max_iter, cache, params):
        calls.append(name)
        return {"system": {"name": name},
                "configs": [{"config": dict(dataclasses.asdict(spec), rule=rule, metric=metric, step_rule=None),
                             "summary": {"flag_counts": {}, "per_k": [{"T_mean": 2.0, "T_ci": [1.9, 2.1]}],
                                         "catastrophic_share": 0.0, "n": n, "stopping": {}}}
                            for spec, rule, metric in configs]}
    return stub


def test_resume_loads_files_of_the_same_provenance(bench, tmp_path, monkeypatch, capsys):
    """Every loop_<s>.json carries its provenance (the package source digest and the SHA-256 of the script's text).
    With --resume a loop_<s>.json of the same provenance is loaded and not recomputed (file unchanged) while one with
    another package digest is recomputed (saying so) and overwritten; loop_summary.md lists both systems, the loaded
    one marked "(loaded)" with its recorded seconds. Without --resume both are recomputed."""
    calls = []
    monkeypatch.setattr(fsr, "run_step_rule_system", _u5_full_stub(calls))
    monkeypatch.setattr(cl, "system_inputs", lambda name, *, cache, params: None)
    want = {"package": scaling.source_digest(), "script": hashlib.sha256(SCRIPT.read_bytes()).hexdigest()}
    first = bench.run_u5(["h4_r1.8", "h4_r4.0"], None, FTParameters(), out_dir=str(tmp_path))
    assert calls == ["h4_r1.8"] * 12 + ["h4_r4.0"] * 12
    for name in ("h4_r1.8", "h4_r4.0"):
        assert json.loads((tmp_path / f"loop_{name}.json").read_text())["provenance"] == want
    out = tmp_path / "out"
    out.mkdir()
    kept = dict(first["h4_r1.8"], runtime_s=4321.5)
    (out / "loop_h4_r1.8.json").write_text(json.dumps(kept))
    stale = dict(first["h4_r4.0"], runtime_s=9999.0, provenance=dict(want, package="0" * 64))
    (out / "loop_h4_r4.0.json").write_text(json.dumps(stale))
    before = (out / "loop_h4_r1.8.json").read_bytes()
    calls.clear()
    capsys.readouterr()
    assert bench.main(["--systems", "h4_r1.8,h4_r4.0", "--skip-closed-loop", "--resume", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert calls == ["h4_r4.0"] * 12
    assert "[resume] loop_h4_r1.8.json: same provenance; loaded" in printed
    assert "[resume] loop_h4_r4.0.json: different package; recomputing" in printed
    assert (out / "loop_h4_r1.8.json").read_bytes() == before
    redone = json.loads((out / "loop_h4_r4.0.json").read_text())
    assert redone["provenance"] == want and redone["runtime_s"] != 9999.0
    md = (out / "loop_summary.md").read_text()
    assert "--resume" in md.splitlines()[2]
    assert "| U5 h4_r1.8 (loaded) | 4321.5 |" in md and "| U5 h4_r4.0 | " in md
    assert sum(row.startswith("| h4_r1.8 | ") for row in md.splitlines()) == 12
    assert sum(row.startswith("| h4_r4.0 | ") for row in md.splitlines()) == 12
    calls.clear()
    assert bench.main(["--systems", "h4_r1.8,h4_r4.0", "--skip-closed-loop", "--out", str(out)]) == 0
    assert calls == ["h4_r1.8"] * 12 + ["h4_r4.0"] * 12
    assert "(loaded)" not in (out / "loop_summary.md").read_text()


def test_resume_of_the_closed_loop_compares_its_settings(bench, tmp_path, monkeypatch, capsys):
    """A closedloop_<s>.json carries the package and script digests and the settings that fix its blocks (--quick,
    --replicates). With --resume a file of this run's provenance is loaded, one written with other settings (a full
    run's file in a --quick run) or without provenance is recomputed; U6 and the wall-time table list every system,
    the loaded one with its recorded seconds."""
    calls = []
    monkeypatch.setattr(cl, "run_closed_loop_system", _cl_stub(calls))
    digests = {"package": scaling.source_digest(), "script": hashlib.sha256(SCRIPT.read_bytes()).hexdigest()}
    quick = dict(digests, settings={"quick": True, "replicates": None})
    first = bench.run_closed_loop(None, True, None, None, FTParameters(), out_dir=str(tmp_path))
    assert calls == ["n2_r2.4", "h4_r4.0"]
    assert all(json.loads((tmp_path / f"closedloop_{n}.json").read_text())["provenance"] == quick
               for n in ("n2_r2.4", "h4_r4.0"))
    assert bench.provenance("closed_loop", quick=False, replicates=7)["settings"] == {"quick": False, "replicates": 7}
    out = tmp_path / "out"
    out.mkdir()
    (out / "closedloop_n2_r2.4.json").write_text(json.dumps(dict(first["n2_r2.4"], runtime_s=777.0)))
    (out / "closedloop_h4_r4.0.json").write_text(json.dumps(dict(first["h4_r4.0"], runtime_s=555.0, provenance=dict(
        digests, settings={"quick": False, "replicates": None}))))
    calls.clear()
    capsys.readouterr()
    assert bench.main(["--quick", "--skip-u5", "--resume", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert calls == ["h4_r4.0"]
    assert "[resume] closedloop_h4_r4.0.json: different settings; recomputing" in printed
    assert json.loads((out / "closedloop_h4_r4.0.json").read_text())["provenance"] == quick
    md = (out / "loop_summary.md").read_text()
    assert "| closed loop n2_r2.4 (loaded) | 777.0 |" in md and "| closed loop h4_r4.0 | " in md
    u6 = md.split("## U6.")[1].split("##")[0].splitlines()
    assert sum(row.startswith("| n2_r2.4 | ") for row in u6) == 6 and sum(row.startswith("| h4_r4.0 | ")
                                                                           for row in u6) == 6
    bare = {k: v for k, v in first["n2_r2.4"].items() if k != "provenance"}
    (out / "closedloop_n2_r2.4.json").write_text(json.dumps(bare))
    calls.clear()
    assert bench.main(["--quick", "--skip-u5", "--resume", "--out", str(out)]) == 0
    assert calls == ["n2_r2.4"] and "closedloop_n2_r2.4.json: no provenance; recomputing" in capsys.readouterr().out


# ------------------------------------------------------------------------- the U5 table
def _precision(k, de, rho):
    return {"k": k, "de_model": de, "sigma_armijo": 0.354 * de, "sigma_parab3": 0.47 * de, "t_res_direct": 1.0,
            "rho_armijo": rho, "rho_parab3": 2.0 * rho}


def _u5_run(rule, r_plan, k_star, k_p90, kind, rs):
    k_ref = k_star if kind == "k_star" else k_p90
    costs = {}
    for i, r in enumerate(rs):
        costs[r] = {"c_iter_over_qpe": [0.0, 1.0 + i, 2.0, 3.0], "c_loop_k_star_over_qpe": 50.0 + i if k_star else None,
                    "c_loop_k_ref_over_qpe": 50.0 + i, "precision": [_precision(1, 1e-3, 0.25 * (i + 1)),
                                                                     _precision(k_ref - 1, 1e-4, 4.0 * (i + 1))]}
    return {"rule": {"name": rule}, "metric": "spin_orbital", "r_plan": r_plan, "k_star": k_star, "k_p90": k_p90,
            "k_ref": k_ref, "k_ref_kind": kind, "extra_measurements": [0, 0, 0, 0, 0], "cost_horizon": 4,
            "costs": costs}


def test_u5_table(bench):
    """U5's markdown: C_loop(k_ref), marked (k_p90) where k* is not reached; sigma_A, sigma_P3 (mHa) and rho_E of the
    Armijo pair and of the 3-point fit at k = 1 and at k_ref - 1, rho_E per r as 0.3 / 1 / 3 and at the planning r
    for R3 rows."""
    u5 = {"sys": {"runs": [_u5_run("unit", None, 3, 0, "k_star", ["0.3", "1", "3"]),
                           _u5_run("unit", None, None, 7, "p90", ["0.3", "1", "3"]),
                           _u5_run("hybrid", 1.0, None, 7, "p90", ["1"])]}}
    lines = bench.u5_table(u5)
    head = [x.strip() for x in lines[4].strip("|").split("|")]
    assert head[8:] == ["C_loop(k_ref), r = 0.3 / 1 / 3", "sigma_A, sigma_P3 (k = 1)", "rho_E Armijo (k = 1)",
                        "rho_E parab3 (k = 1)", "k_ref - 1", "sigma_A, sigma_P3 (k_ref - 1)",
                        "rho_E Armijo (k_ref - 1)", "rho_E parab3 (k_ref - 1)"]
    rows = [[x.strip() for x in row.strip("|").split("|")] for row in lines[6:]]
    assert len(rows) == 3 and all(len(row) == len(head) for row in rows)
    unit, unit_p90, hybrid = rows
    assert unit[8:] == ["50 / 51 / 52", "0.354, 0.47", "0.25 / 0.5 / 0.75", "0.5 / 1 / 1.5", "2", "0.0354, 0.047",
                        "4 / 8 / 12", "8 / 16 / 24"]
    assert unit_p90[3] == "- (7)" and unit_p90[8] == "50 / 51 / 52 (k_p90)" and unit_p90[12] == "6 (k_p90)"
    assert hybrid[2] == "hybrid (r = 1)" and hybrid[5:8] == ["-", "1 / 2 / 3", "-"]
    assert hybrid[8:] == ["- / 50 / - (k_p90)", "0.354, 0.47", "0.25", "0.5", "6 (k_p90)", "0.0354, 0.047", "4", "8"]


def test_reference_rows_without_an_interval(bench):
    """The reference rows of loop_summary.md take the strict-JSON forms of a closed-loop cell: without a bootstrap
    interval (null bounds, one replicate) the CI row reads "no" and the 100-replicate-interval row still compares the
    mean; an infinite mean ("inf") reads "no"; matching cells read "yes"."""
    def cell(controller, t_mean, t_ci, n, beats=1.0):
        per_k = [{"T_mean": 9.0}, {"T_mean": 5.0}, {"T_mean": t_mean, "T_ci": t_ci, "p_median": 0.284,
                                                     "p_band": [0.27, 0.30], "beats_psi1": beats}]
        return {"block": "headline", "label": f"{controller} r = 1",
                "config": {"rule": "unit", "controller": controller, "r": 1.0},
                "summary": {"per_k": per_k, "n": n}}

    one = {"n2_r2.4": {"configs": [cell("oracle", 3.95, [None, None], 1), cell("lagged", "inf", [None, None], 1)]}}
    rows = bench.reference_checks({}, one)
    body = [r for r in rows if r.startswith("| n2_r2.4")]
    assert len(body) == 4
    assert body[0].endswith("| no (reference inside the CI) |") and body[1].endswith("| yes (inside the 16-84% band) |")
    assert body[2].endswith("| yes |") and body[3].endswith("| no |")              # the interval check needs no CI
    good = {"n2_r2.4": {"configs": [cell("oracle", 3.95, [3.9, 4.0], 20), cell("lagged", 3.6, [3.5, 3.7], 20)]}}
    body = [r for r in bench.reference_checks({}, good) if r.startswith("| n2_r2.4")]
    assert body[0].endswith("| yes (reference inside the CI) |") and body[2].endswith("| yes |")
    assert body[3].endswith("| yes |")
