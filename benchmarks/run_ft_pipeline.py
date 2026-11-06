"""The M3 fault-tolerant cost pipeline on the benchmark systems: tables T1-T9 and the Appendix A checks.

For every system of ``--systems`` this script runs ``thc_engine_cqe.ft.pipeline.run_system``:
the QPE scan and selection (DF, THC, SF), exact CQE trajectories of the optimizer families of
``--optimizers`` ("lbfgs": spin-free L-BFGS, "lbfgs_so": L-BFGS with the spin-orbital metric,
i.e. the standard CQE generator, "sd": spin-free steepest descent), each with the exact and the
classical (second-order) first step, their state-preparation costs, the residual and
line-search cost of every iteration to k* + 4, the hand-off to QPE, the classical first step,
the CQE-only energy, the compiled replay and the physical costs (``ft.pipeline`` module
docstring). As designed [design], the stretched systems h6s and n2s and h8 run the L-BFGS
families only, and h8 costs the line search but not the residual unless ``--h8-residual`` is
given (its X_e targets exceed the runtime budget); its measured-loop costs are then lower
bounds.

It then checks the reference numbers of Appendix A (docs/m3-cost-model.md) against the run
(element counts, V_m at SD iterate 2, the L-BFGS trajectories and their compiled costs, the
spectrum, the first step, the QPE selection, the physical model and the primitives) and writes
a table: quantity, design value, reproduced, match. The "design value" column holds the values
the design specification printed; its n2 V_m on the allowed elements (3703, an RB estimate on an
earlier orbital labelling) does not reproduce, and Appendix A lists the exact pair sum this
script computes (3753.0) instead. A value matches when it agrees to the digits printed in
that column (half a unit of the last digit); integers must be equal; Rao-Blackwellized V_m
estimates must lie within 3 standard errors (Appendix A's RB values used the same seed, so the
distance from an exact pair sum, where one is known, is printed as well).

Outputs: <out>/ft_<system>.json (``pipeline.to_json``) per system and <out>/ft_summary.md
(T1-T9, the Appendix A table, wall time per system and component, peak memory).

Threads. ``--threads n`` sets PySCF's OpenMP threads and, through OMP_NUM_THREADS /
OPENBLAS_NUM_THREADS / MKL_NUM_THREADS, numpy's BLAS threads; numpy is imported only after the
arguments are parsed (as in benchmarks/run_optimizers.py), so the pin takes effect when the
file runs as a script.

Usage:
    python benchmarks/run_ft_pipeline.py --systems h4,h6a,n2,lih,h6s,n2s,h8 \\
        --optimizers lbfgs,lbfgs_so,sd --threads 1 --out results/
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
DEFAULT_SYSTEMS = "h4,h6a,n2,lih,h6s,n2s,h8"
DEFAULT_OPTIMIZERS = "lbfgs,lbfgs_so,sd"

# Appendix A (docs/m3-cost-model.md) with the values the design specification printed (strings keep the
# printed precision); n2's allowed V_m is the design value 3703, which Appendix A corrects to 3753.0.
APPENDIX_COUNTS = {"h4": ("28 / 60", "70 / 150"), "h6a": ("153 / 315", "408 / 840"),
                   "lih": ("105 / 315", "264 / 840"), "n2": ("33 / 315", "84 / 840"),
                   "h8": ("496 / 1008", "1364 / 2772")}
APPENDIX_VM = {"h4": ("1220", "2402"), "h6a": ("2.515e4", "5.144e4"), "n2": ("3703", "2.947e4")}
# Exact pair sums (Ha^2) at SD iterate 2 computed once outside the benchmark through measure.routes alone
# (unweighted X_e, method="exact"): h6a allowed elements, 1.12e9 commuting pairs, 269 s [build check];
# they check the Rao-Blackwellized estimates independently of their seed.
EXACT_VM = {("h6a", "allowed"): 25308.28}
APPENDIX_STEPS = {   # E - E_FCI after the step (mHa), p after, n_d / n_s, b, C_step, cumulative
    "h4": [("21.504", "0.9802", "10 / 0", "15", "160", "160"), ("3.890", "0.9960", "40 / 24", "18", "1192", "1352"),
           ("1.203", "0.9981", "38 / 24", "18", "1154", "2506")],
    "h6a": [("37.956", "0.9576", "51 / 0", "18", "969", "969"), ("12.014", "0.9850", "273 / 120", "21", "8526", "9495"),
            ("5.236", "0.9907", "271 / 120", "21", "8482", "17977"),
            ("2.991", "0.9950", "271 / 120", "21", "8482", "26459"),
            ("1.877", "0.9974", "273 / 120", "21", "8526", "34985"),
            ("1.111", "0.9987", "273 / 116", "21", "8442", "43427")],
    "n2": [("35.209", "0.9635", "27 / 0", "17", "486", "486"), ("14.678", "0.9813", "55 / 0", "18", "1045", "1531"),
           ("1.147", "0.9994", "55 / 0", "18", "1045", "2576")],
}
APPENDIX_P_HF = {"h4": "0.9455", "h6a": "0.9026", "n2": "0.9257", "lih": "0.9743"}
APPENDIX_NORM_A = {"h4": ("0.8585", "0.6115", "0.2157", "0.1256"), "h6a": ("1.0993", "0.7424", "0.4219", "0.2397"),
                   "n2": ("1.8955", "0.6638", "0.4038", "0.1168")}
APPENDIX_RATIO = {"h4": "1.0159", "h6a": "1.0220", "n2": "0.9805", "lih": "1.0192"}
APPENDIX_GAP = {"h4": ("0.5657", "0.5633"), "h6a": ("0.3909", "0.3909"), "n2": ("0.7115", "0.4117"),
                "lih": ("0.1332", "0.1332")}
APPENDIX_QPE = {  # threshold, L, Xi, lambda_DF, Toffolis / step, total, logical qubits
    "h4": ("3e-3", "7", "27", "4.935", "868", "6.73e6", "188"),
    "h6a": ("3e-3", "11", "63", "9.726", "1309", "2.00e7", "233"),
    "n2": ("3e-3", "20", "82", "9.423", "1420", "2.10e7", "235"),
    "lih": ("3e-4", "18", "68", "9.259", "1366", "1.99e7", "235"),
}
APPENDIX_THC_H6A = ("15", "auto", "9.125", "1035", "1.48e7")
APPENDIX_JA = (((0.2, 1e-4), "3"), ((0.5, 1e-4), "4"), ((0.79, 1e-4), "5"), ((1.16, 1e-4), "6"),
               ((4.87, 1e-6), "14"), ((6.81, 1e-6), "17"))
APPENDIX_BITS = ((10, "15"), (64, "18"), (393, "21"))
APPENDIX_KAPPA = ("0.974", "0.993", "0.998")


def pin_threads(n: int | None) -> dict:
    """Pin PySCF's OpenMP threads and numpy's BLAS threads to ``n`` (None: inherit); the settings in effect."""
    blas_pinned = None
    if n is not None:
        blas_pinned = "numpy" not in sys.modules
        if blas_pinned:
            for var in THREAD_VARS:
                os.environ[var] = str(n)
    from pyscf import lib

    if n is not None:
        lib.num_threads(n)
    return {"threads": n, "blas_pinned": blas_pinned, "pyscf_threads": lib.num_threads(),
            "cpu_count": os.cpu_count(), **{var: os.environ.get(var) for var in THREAD_VARS}}


def _g(x: float) -> str:
    """x in the shortest form, exponents as 1e-5."""
    text = format(x, "g")
    if "e" in text:
        mant, exp = text.split("e")
        text = f"{mant}e{int(exp)}"
    return text


def peak_rss_mb() -> float:
    """Peak resident memory of this process so far (MB; ru_maxrss is in kB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ----------------------------------------------------------------- Appendix A
def matches(expected: str, value) -> bool:
    """True if ``value`` agrees with the printed ``expected`` to half a unit of its last digit.

    "a / b" compares both parts; an integer count must equal the printed integer, a real value
    printed as an integer (e.g. V_m = 1220) must lie within 0.5 of it; "6.73e6" compares the
    significant digits.
    """
    if " / " in expected:
        parts = expected.split(" / ")
        return len(parts) == len(value) and all(matches(e, v) for e, v in zip(parts, value))
    if isinstance(value, str):
        return value == expected
    if "e" in expected:
        mant, exp = expected.split("e")
        decimals = len(mant.split(".")[1]) if "." in mant else 0
        unit = 10.0 ** (int(exp) - decimals)
    elif "." in expected:
        unit = 10.0 ** -len(expected.split(".")[1])
    elif isinstance(value, int) or hasattr(value, "dtype") and value.dtype.kind in "iu":
        return int(value) == int(expected)
    else:
        unit = 1.0
    return abs(float(value) - float(expected)) <= 0.5 * unit * (1 + 1e-9)


def _show(value, expected: str) -> str:
    """``value`` printed with the precision of ``expected``."""
    if " / " in expected:
        return " / ".join(_show(v, e) for v, e in zip(value, expected.split(" / ")))
    if isinstance(value, str):
        return value
    if "e" in expected:
        mant = expected.split("e")[0]
        decimals = len(mant.split(".")[1]) if "." in mant else 0
        text = format(float(value), f".{decimals}e")
        m, e = text.split("e")
        return f"{m}e{int(e)}"
    if "." in expected:
        return format(float(value), f".{len(expected.split('.')[1])}f")
    return str(int(value)) if float(value) == int(value) else format(float(value), ".6g")


def appendix_a(reports: dict, params) -> tuple[list[tuple], float]:
    """(rows (quantity, design value, reproduced, match), seconds) for the systems in ``reports``."""
    from thc_engine_cqe.ft import physical, primitives, residual_inputs as ri, stateprep, symmetry
    from thc_engine_cqe.ft.pipeline import global_validation
    from thc_engine_cqe.measure import targets

    t0 = time.perf_counter()
    rows: list[tuple] = []

    def add(quantity: str, expected: str, value, verdict: str | None = None) -> None:
        rows.append((quantity, expected, _show(value, expected),
                     verdict or ("yes" if matches(expected, value) else "no")))

    for name, (sf, so) in APPENDIX_COUNTS.items():
        act = symmetry.benchmark_system(name)
        ir = symmetry.orbital_irreps(act)
        for kind, exp in (("spin_free", sf), ("spin_orbital", so)):
            el, mask = ri.allowed_elements(act, kind, ir)
            add(f"{name} allowed / all elements, {kind}", exp, (len(el), len(mask)))
    for name, (allowed, every) in APPENDIX_VM.items():
        rep = reports.get(name)
        if rep is None or "sd" not in rep.trajectories:
            continue
        act = symmetry.benchmark_system(name)
        ir = symmetry.orbital_irreps(act)
        psi = rep.trajectories["sd"]["exact"].states[2]
        for label, el, exp in (("allowed", ri.allowed_elements(act, "spin_free", ir)[0], allowed),
                               ("all", targets.element_set(act.norb, "spin_free"), every)):
            var = ri.element_variances(act, psi, el, "spin_free", params=params)
            verdict = None
            if var["method"] == "exact":
                method = "exact"
                if name == "n2" and label == "allowed":
                    verdict = ("no: the design value is an RB estimate (seed 7) on the earlier orbital labelling; "
                               "the exact pair sum is the reproduced one (docs/m3-cost-model.md, Appendix A)")
            else:
                method = f"RB {var['n_settings']}, +-{var['stderr_m']:.3g}"
                z = (var["variance_m"] - float(exp)) / var["stderr_m"]
                verdict = f"{'yes' if abs(z) <= 3.0 else 'no'} ({z:+.1f} sigma, same seed as Appendix A"
                exact = EXACT_VM.get((name, label))
                if exact is not None:
                    z_exact = (var["variance_m"] - exact) / var["stderr_m"]
                    verdict += f"; {z_exact:+.1f} sigma from the exact pair sum {exact:,.0f}"
                verdict += ")"
            add(f"{name} V_m at SD iterate 2, {label} elements ({method})", exp, var["variance_m"], verdict)
    for name, steps in APPENDIX_STEPS.items():
        rep = reports.get(name)
        if rep is None or "lbfgs" not in rep.trajectories:
            continue
        rec = rep.trajectories["lbfgs"]["exact"]
        pc = rep.prep["lbfgs"]["exact"]["elementwise"]
        for k, (err, p, ndns, b, c, cum) in enumerate(steps):
            st = pc.steps[k]
            add(f"{name} L-BFGS step {k}: E - E_FCI after (mHa)", err, rec.errors[k + 1] * 1e3)
            add(f"{name} L-BFGS step {k}: p after", p, rec.overlaps[k + 1])
            add(f"{name} L-BFGS step {k}: n_d / n_s, b, C_step, C(Psi_{k + 1})", f"{ndns} / {b} / {c} / {cum}",
                (st.n_double, st.n_single, st.bits, st.toffolis, int(pc.cumulative[k + 1])))
    for name, exp in APPENDIX_P_HF.items():
        if name in reports and "lbfgs" in reports[name].trajectories:
            add(f"{name} HF overlap p_0", exp, reports[name].trajectories["lbfgs"]["exact"].overlaps[0])
    for name, exp in APPENDIX_NORM_A.items():
        if name in reports and "lbfgs" in reports[name].trajectories:
            rec = reports[name].trajectories["lbfgs"]["exact"]
            add(f"{name} norm A before steps 0-3", " / ".join(exp), tuple(rec.residual_norms[:4]))
    for name, exp in APPENDIX_RATIO.items():
        if name in reports and "lbfgs" in reports[name].first_step:
            add(f"{name} eps_2 / eps*", exp, reports[name].first_step["lbfgs"]["ratio"])
    for name, (same, other) in APPENDIX_GAP.items():
        if name in reports and "lbfgs" in reports[name].trajectories:
            rec = reports[name].trajectories["lbfgs"]["exact"]
            add(f"{name} E1 - E0 (Ha), same / any irrep", f"{same} / {other}", (rec.e1 - rec.e0, rec.e1_any - rec.e0))
    for name, (thr, L, xi, lam, cw, total, q) in APPENDIX_QPE.items():
        rep = reports.get(name)
        if rep is None or rep.qpe.get("df") is None:
            continue
        sel = rep.qpe["df"]
        enc = sel.encoding
        add(f"{name} DF selection: threshold / L / Xi", f"{thr} / {L} / {xi}",
            (format(enc.params["threshold"], ".0e").replace("e-0", "e-"), enc.params["L"], enc.params["Lxi"]))
        add(f"{name} DF lambda / C_W / Q", f"{lam} / {cw} / {q}", (enc.lam, enc.toffoli_per_step, sel.logical_qubits))
        add(f"{name} DF C_QPE (1 mHa)", total, sel.toffolis)
    rep = reports.get("h6a")
    if rep is not None and rep.qpe.get("thc") is not None:
        sel = rep.qpe["thc"]
        enc = sel.encoding
        m, pen, lam, cw, total = APPENDIX_THC_H6A
        add("h6a THC selection: M / penalty / lambda / C_W", f"{m} / {pen} / {lam} / {cw}",
            (enc.params["M"], str(enc.params["penalty"]), enc.lam, enc.toffoli_per_step))
        add("h6a THC C_QPE", total, sel.toffolis)
    for (q, n, budget), exp in (((233, 19_997_593, 0.1), "956160 / 23 / 574.9"),
                                ((233, 19_997_593, 0.01), "1026160 / 25 / 574.9"),
                                ((83, 43_427, 1e-5), "721960 / 25 / 1.249"),
                                ((83, 43_427, 1e-4), "696960 / 23")):
        c = physical.surface_code_cost(q, n, fail_budget=budget, shot=(q == 83))
        label = "shot" if q == 83 else "QPE"
        value = (c.physical_qubits, c.distance, c.seconds) if exp.count("/") == 2 else (c.physical_qubits, c.distance)
        add(f"physical {label} Q = {q}, {n:,} Toffolis, budget {_g(budget)}: qubits / d / s", exp, value)
    rates = sorted({round(physical.surface_code_cost(q, n, fail_budget=0.1).us_per_toffoli, 4)
                    for q, n in ((188, 6_728_736), (233, 19_997_593))})
    add("physical us per Toffoli (H4, H6 DF QPE)", "26.25 / 28.75", tuple(rates))
    for (tau, delta), exp in APPENDIX_JA:
        add(f"Jacobi-Anger degree R({_g(tau)}, {_g(delta)})", exp, primitives.jacobi_anger_degree(tau, delta))
    for n_add, exp in APPENDIX_BITS:
        add(f"phase-gradient bits for A = {n_add} at delta = 1e-3", exp, primitives.phase_gradient_bits(n_add, 1e-3))
    add("readout rotation C_rot (N = 6)", "405", stateprep.readout_rotation_toffolis(6))
    kappas = [x["kappa"] for x in global_validation()["ae_kappa"]]
    add("sine-window AE kappa, M = 63 / 255 / 1023", " / ".join(APPENDIX_KAPPA), tuple(kappas))
    return rows, time.perf_counter() - t0


# ----------------------------------------------------------------- summary
def runtime_table(reports: dict, extra: dict, peak: dict) -> list[str]:
    comps = list(dict.fromkeys(c for rep in reports.values() for c in rep.timings))   # pipeline order
    lines = ["## Runtime", "",
             "Wall time (s) per system and component of ``run_system`` on one thread, and the peak resident "
             "memory of the process after the system (MB).", "",
             "| system | families | total | " + " | ".join(comps) + " | peak RSS (MB) |",
             "|---|---|---|" + "---|" * len(comps) + "---|"]
    for name, rep in reports.items():
        lines.append(f"| {name} | {', '.join(rep.settings['families'])} | {rep.runtime_s:.1f} | "
                     + " | ".join(f"{rep.timings.get(c, 0.0):.1f}" for c in comps) + f" | {peak[name]:.0f} |")
    lines.append("")
    total = sum(rep.runtime_s for rep in reports.values())
    lines.append(f"Systems: {total:.1f} s. Appendix A checks: {extra['appendix_a']:.1f} s. "
                 f"Whole script: {extra['script']:.1f} s. Peak RSS: {max(peak.values(), default=0.0):.0f} MB.")
    lines.append("")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", default=DEFAULT_SYSTEMS, help="comma-separated subset of " + DEFAULT_SYSTEMS)
    ap.add_argument("--optimizers", default=DEFAULT_OPTIMIZERS,
                    help="comma-separated optimizer families (lbfgs, lbfgs_so, sd); h6s, n2s and h8 run the "
                         "L-BFGS families only")
    ap.add_argument("--h8-residual", action=argparse.BooleanOptionalAction, default=False,
                    help="cost the residual routes on h8 too (default: line search only)")
    ap.add_argument("--threads", type=int, default=None,
                    help="PySCF OpenMP and numpy BLAS threads (default: inherit)")
    ap.add_argument("--out", default="results/", help="output directory")
    args = ap.parse_args(argv)
    t_script = time.perf_counter()
    threads = pin_threads(args.threads)
    # imported after pin_threads, because they import numpy (module docstring)
    from thc_engine_cqe.ft import pipeline
    from thc_engine_cqe.ft.primitives import FTParameters
    from thc_engine_cqe.ft.symmetry import BENCHMARK_SYSTEMS

    names = [s.strip() for s in args.systems.split(",") if s.strip()]
    requested = [s.strip() for s in args.optimizers.split(",") if s.strip()]
    bad = [n for n in names if n not in BENCHMARK_SYSTEMS]
    if bad:
        raise SystemExit(f"unknown systems {bad}; choose from {sorted(BENCHMARK_SYSTEMS)}")
    os.makedirs(args.out, exist_ok=True)
    params = FTParameters()
    reports, peak = {}, {}
    for name in names:
        families = pipeline.default_optimizers(name, requested)
        residual = args.h8_residual or name != "h8"
        print(f"[{name}] families {', '.join(families)}; residual {'on' if residual else 'off (line search only)'}",
              flush=True)
        rep = pipeline.run_system(name, optimizers=families, residual=residual, params=params)
        reports[name] = rep
        peak[name] = peak_rss_mb()
        data = pipeline.to_json(rep)
        data["threads"] = threads
        with open(os.path.join(args.out, f"ft_{name}.json"), "w") as fh:
            json.dump(data, fh, allow_nan=False)
        k_star = "; ".join(f"{fam} k* {rep.settings['k_star'][fam]['exact']}" for fam in families)
        print(f"[{name}] {rep.runtime_s:.1f} s ({', '.join(f'{k} {v:.1f}' for k, v in rep.timings.items())}); "
              f"C_QPE {rep.qpe['overall'].toffolis:.4g} ({rep.qpe['overall'].encoding.kind}); {k_star}; "
              f"peak RSS {peak[name]:.0f} MB", flush=True)
    rows, t_app = appendix_a(reports, params)
    n_match = sum(r[3].startswith("yes") for r in rows)
    print(f"[appendix A] {n_match} of {len(rows)} match ({t_app:.1f} s)", flush=True)
    extra = {"appendix_a": t_app, "script": time.perf_counter() - t_script}
    command = (f"benchmarks/run_ft_pipeline.py --systems {','.join(names)} --optimizers {','.join(requested)}"
               + ("" if args.threads is None else f" --threads {args.threads}") + f" --out {args.out}"
               + (" --h8-residual" if args.h8_residual else ""))
    lines = ["# M3 fault-tolerant cost pipeline: tables T1-T9", "",
             f"Generated by `{command}`. Threads: {threads}.", ""]
    lines.append(pipeline.markdown_tables(list(reports.values())))
    lines += ["## Appendix A regressions", "",
              f"{n_match} of {len(rows)} quantities match their design values to the printed digits "
              "(Rao-Blackwellized V_m: within 3 standard errors).", "",
              "| quantity | design value | reproduced | match |", "|---|---|---|---|"]
    lines += [f"| {q} | {e} | {v} | {m} |" for q, e, v, m in rows]
    lines.append("")
    lines += runtime_table(reports, extra, peak)
    with open(os.path.join(args.out, "ft_summary.md"), "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {len(names)} JSON files and ft_summary.md to {args.out} ({extra['script']:.1f} s, peak RSS "
          f"{max(peak.values(), default=0.0):.0f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
