"""The M4 scaling study: tables U1-U4 (repetition columns) of the size, overlap and active-space axes.

For every system of ``--systems`` this script runs ``thc_engine_cqe.ft.scaling.scan_system``, the per-system
protocol of the scaling study (``ft.scaling`` module docstring): the singlet spectrum (12 roots; 2 at N = 10,
extended to 12 if root 1 is not in the ground irrep), the QPE scan and selection (F9), the exact L-BFGS records
(spin-free and spin-orbital, exact and classical first step; second-order only at N = 10), the R1 unit-step
records (both metrics; spin-orbital only at N = 10), the residual inputs (M3's iterate-1 variances at N <= 6, the
HF determinant at N = 8, F24 (i), a 100-element subset at HF at N = 10, F24 (ii)), the iteration costs for
k <= 3 at r = 0.3, 1 and 3 (M3's F14 on the exact records, F30 of the unit rule on the R1 records; the
closed-form direct route at N = 10), the cheapest measured iteration (k = 1 of the unit loop) and the compiled
overlaps (F25, k <= 3, s = 1, 2, 4, and the slice rule over s <= 16 from its own compiled state; N <= 8). Then
``scaling_report`` fits the overlaps (F22), gaps and windows (F21, with the one-run B_L window solved jointly with
F22 and the overlap floor of the two-run window) and cost components (F23) per bond length and brackets the
break-even size (F45) under repetition: alpha_1 from the classical-step series, alpha_2 from the R1 records, the
cost points from the R1 records or, where none is run (spin-free at N = 10), the cheapest iteration at the same
Psi_1, the preparation terms C(Psi_j)/C_QPE as power laws, the window of each case's own gap model, F45's c_k = 0
form as a check and the crossed rate corners as a sensitivity. Every value beyond the computed sizes is a model
statement (a model bracket of H chains in canonical orbitals with elementwise compilation and the full allowed
residual).

Systems: the size axis (H_n, n = 4..10, R = 2.5..4.0 bohr), the overlap axis (N2 CAS(6,6), 1.098..2.8 A) and the
active-space axis (N2 CAS(10,8)); ``--quick``: h4_r{3.0,4.0}, h6_r{3.0,4.0}, h8_r{3.0,4.0}, n2_r{2.0,2.4,2.8}
(rate fits on n = 4, 6, 8 at R = 3.0 and 4.0).

Stages and the cache. Every stage of a system (``ft.scaling.SCAN_STAGES``: spectrum, qpe, residual_inputs,
records, costs, compiled) is cached under <out>/cache (``ft.scaling.Cache``: one pickle per system, stage and
digest of the build, the package sources, the dependency versions and the stage's parameters; atomic writes).
``--stage <name>`` computes that stage (and, through the cache, the stages it needs) for the systems and exits,
so a cold run can be split into processes that each stay below about 10 minutes; a final run without
``--stage`` assembles the tables from the cache. ``--no-cache`` recomputes and overwrites every entry. At
N <= 8 the first three stages are those of ``ft.closedloop.system_inputs`` (shared with
benchmarks/run_m4_loop.py).

Outputs (``--out``, default results/m4): scaling_<system>.json (``ft.scaling.record_summary``: spectrum, QPE,
per record the energies, overlaps, eps, prep cumulative, k*, rule-E k_cert and the costs per r, no states; the
cheapest iteration and the compiled overlaps), scaling_fits.json (``ft.scaling.scaling_report``) and
scaling_summary.md (U1-U4 with U2's B_L side column and U4's F45 inputs and n* per window, the reference-number
checks, wall time per system and stage). Strict JSON: nan as null, +-inf as "inf"/"-inf".

Threads: OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and MKL_NUM_THREADS are set to 1 before numpy is imported.

Usage:
    python benchmarks/run_m4_scaling.py --quick
    python benchmarks/run_m4_scaling.py --systems h8_r3.0,h8_r4.0 --stage costs
    python benchmarks/run_m4_scaling.py --out results/m4
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from decimal import Decimal  # noqa: E402

# Reference numbers: the printed strings keep their precision; a value matches when it agrees to half a unit of
# the last printed digit ([design] values, re-checked by this run).
REF_P_HF = {  # [design]
    **{f"h4_r{r}": v for r, v in zip(("1.8", "2.5", "3.0", "3.5", "4.0"), (".946", ".836", ".701", ".553", ".434"))},
    **{f"h6_r{r}": v for r, v in zip(("1.8", "2.5", "3.0", "3.5", "4.0"), (".916", ".756", ".573", ".396", ".272"))},
    **{f"h8_r{r}": v for r, v in zip(("1.8", "2.5", "3.0", "3.5", "4.0"), (".886", ".680", ".465", ".280", ".168"))},
    **{f"h10_r{r}": v for r, v in zip(("1.8", "2.5", "3.0", "3.5", "4.0"), (".856", ".610", ".375", ".197", ".104"))},
    **{f"n2_r{r}": v for r, v in zip(("1.098", "1.6", "2.0", "2.4", "2.8"), (".926", ".657", ".253", ".112", ".0785"))},
    **{f"n2v_r{r}": v for r, v in zip(("1.098", "1.6", "2.0", "2.4", "2.8"),
                                       (".9173", ".6490", ".2464", ".1112", ".0782"))},
}
REF_GAP_MHA = {  # [design] same-irrep Delta_1
    **{f"h4_r{r}": v for r, v in zip(("2.5", "3.0", "3.5", "4.0"), ("266.3", "139.0", "65.3", "28.2"))},
    **{f"h6_r{r}": v for r, v in zip(("2.5", "3.0", "3.5", "4.0"), ("200.0", "104.7", "49.3", "21.3"))},
    **{f"h8_r{r}": v for r, v in zip(("2.5", "3.0", "3.5", "4.0"), ("159.3", "83.3", "39.2", "17.0"))},
    **{f"h10_r{r}": v for r, v in zip(("2.5", "3.0", "3.5", "4.0"), ("132.1", "69.0", "32.5", "14.1"))},
    **{f"n2_r{r}": v for r, v in zip(("1.098", "1.6", "2.0", "2.4", "2.8"),
                                      ("711.5", "218.5", "191.7", "229.9", "243.3"))},
}
REF_C_QPE = {"h4_r3.0": "3.03e6", "h4_r4.0": "2.80e6", "h6_r3.0": "9.74e6", "h8_r3.0": "1.87e7",  # [design]
             "n2_r2.0": "9.49e6", "n2_r2.4": "9.16e6", "n2_r2.8": "8.44e6", "h10_r3.0": "3.01e7"}
REF_CHEAPEST = {("n2_r2.4", "so"): "0.43", ("n2_r2.8", "so"): "0.41", ("h6_r3.0", "sf"): "1.8",  # [design], r = 1
                ("h8_r3.0", "sf"): "27", ("h10_r3.0", "sf"): "209"}
# [design] c_res(1) = T_res(1)/C_QPE at r = 1 of the first closed-loop step (spin-orbital, M3's iterate-1 variances):
# the design values above used variances at the HF determinant (F24 (i)) at every N, the protocol keeps M3's
# iterate-1 variances at N <= 6.
REF_CHEAPEST_V = {("n2_r2.4", "so"): "0.423", ("n2_r2.0", "so"): "0.515", ("n2_r2.8", "so"): "0.398"}
REF_N2_R24 = {"kind": "sf", "lam": "7.59", "n_iter": "11924", "c_qpe": "9157632",  # [design]
              "e_err_hf": "777.01", "e_err_psi1_so": "301.43", "e_err_psi2_ls_so": "232.16",
              "e_err_psi2_unit_so": "252.43"}
REF_RATES_R30 = ("0.1007", "0.1049")  # [design] HF successive rates at R = 3.0 bohr (n = 4, 6, 8)


def matches(expected: str, value) -> bool:
    """True if ``value`` agrees with the printed ``expected`` to half a unit of its last digit."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return False
    if isinstance(value, str):
        return value == expected
    d = Decimal(expected)
    half = Decimal(1).scaleb(d.as_tuple().exponent) / 2
    return abs(Decimal(repr(float(value))) - d) <= half


def peak_rss_mb() -> float:
    """Peak resident memory of this process so far (MB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def reference_checks(records: dict, report: dict) -> list[tuple[str, str, str, bool]]:
    """(quantity, reference, reproduced, match) for every reference number whose system was run."""
    rows = []

    def add(quantity: str, expected: str, value, shown: str | None = None) -> None:
        rows.append((quantity, expected, shown if shown is not None else ("-" if value is None else f"{value:.6g}"),
                     matches(expected, value)))

    for name, rec in records.items():
        if name in REF_P_HF:
            add(f"{name} p_HF [design]", REF_P_HF[name], rec.spectrum.p_hf)
        if name in REF_GAP_MHA:
            add(f"{name} Delta_1 (mHa) [design]", REF_GAP_MHA[name], 1e3 * rec.spectrum.gap)
        if name in REF_C_QPE:
            add(f"{name} C_QPE [design]", REF_C_QPE[name], rec.c_qpe)
        for metric in ("sf", "so"):
            for refs, tag in ((REF_CHEAPEST, "[design, HF-state variances]"),
                              (REF_CHEAPEST_V, "[design, iterate-1 variances]")):
                ref = refs.get((name, metric))
                if ref is not None and metric in rec.cheapest:
                    add(f"{name} T_res(1)/C_QPE, {metric}, r = 1 {tag}", ref,
                        rec.cheapest[metric]["by_r"]["1"]["over_qpe"])
    if "n2_r2.4" in records:
        rec = records["n2_r2.4"]
        enc = rec.qpe.encoding
        add("n2_r2.4 QPE kind [design]", REF_N2_R24["kind"], enc.kind, enc.kind)
        add("n2_r2.4 lambda (Ha) [design]", REF_N2_R24["lam"], float(enc.lam))
        add("n2_r2.4 n_iter [design]", REF_N2_R24["n_iter"], int(rec.qpe.iterations), str(rec.qpe.iterations))
        add("n2_r2.4 C_QPE [design]", REF_N2_R24["c_qpe"], int(rec.qpe.toffolis), str(rec.qpe.toffolis))
        so, r1 = rec.records["lbfgs_so/second_order"], rec.records["r1_so"]
        for key, value in (("e_err_hf", so.errors[0]), ("e_err_psi1_so", so.errors[1]), ("e_err_psi2_ls_so", so.errors[2]),
                           ("e_err_psi2_unit_so", r1.errors[2])):
            add(f"n2_r2.4 {key.replace('e_err_', 'E - E0 of ')} (mHa) [design]", REF_N2_R24[key], 1e3 * float(value))
    hf30 = next((r for r in report["U2"]["overlap"] if r["record"] == "hf" and r["R"] == 3.0), None)
    if hf30 is not None and hf30["ns"][:3] == [4, 6, 8]:
        for n, ref, value in zip((6, 8), REF_RATES_R30, hf30["successive"]):
            add(f"R = 3.0 HF successive rate alpha^({n}) [design]", ref, value)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="the quick system set (ft.scaling.QUICK)")
    ap.add_argument("--systems", default=None, help="comma-separated systems (default: the size, overlap and "
                                                    "active-space axes, or the quick set)")
    ap.add_argument("--out", default="results/m4", help="output directory (the cache goes to <out>/cache)")
    ap.add_argument("--no-cache", action="store_true", help="recompute and overwrite every cache entry")
    ap.add_argument("--stage", default=None, help="compute one stage of ft.scaling.SCAN_STAGES and exit")
    ap.add_argument("--k-cost", type=int, default=3, help="last costed iterate (default 3)")
    ap.add_argument("--n-steps", type=int, default=6, help="steps of every record (default 6)")
    args = ap.parse_args(argv)
    t_script = time.perf_counter()
    from thc_engine_cqe.ft import scaling

    if args.systems:
        names = [s.strip() for s in args.systems.split(",") if s.strip()]
    elif args.quick:
        names = list(scaling.QUICK)
    else:
        names = list(scaling.SIZE_AXIS + scaling.OVERLAP_AXIS + scaling.ACTIVE_SPACE_AXIS)
    if args.stage is not None and args.stage not in scaling.SCAN_STAGES:
        raise SystemExit(f"--stage must be one of {scaling.SCAN_STAGES}")
    os.makedirs(args.out, exist_ok=True)
    cache = scaling.Cache(os.path.join(args.out, "cache"), refresh=args.no_cache)
    kw = dict(k_cost=args.k_cost, n_steps=args.n_steps, r_values=(0.3, 1.0, 3.0))
    if args.stage is not None:
        for name in names:
            t = time.perf_counter()
            timings = scaling.scan_stage(name, args.stage, cache=cache, **kw)
            print(f"[{name}] stage {args.stage}: {time.perf_counter() - t:.1f} s "
                  f"({', '.join(f'{k} {v:.1f}' for k, v in timings.items())}); peak RSS {peak_rss_mb():.0f} MB",
                  flush=True)
        return 0
    records, runtime, peak = {}, {}, {}
    for name in names:
        t = time.perf_counter()
        rec = scaling.scan_system(name, cache=cache, **kw)
        runtime[name] = time.perf_counter() - t
        peak[name] = peak_rss_mb()
        records[name] = rec
        data = scaling.record_summary(rec)
        data["runtime_s"] = runtime[name]
        with open(os.path.join(args.out, f"scaling_{name}.json"), "w") as fh:
            json.dump(data, fh, allow_nan=False)
        cheap = "; ".join(f"{m} T_res(1)/C_QPE {c['by_r']['1']['over_qpe']:.3g}" for m, c in rec.cheapest.items())
        print(f"[{name}] {runtime[name]:.1f} s ({', '.join(f'{k} {v:.1f}' for k, v in rec.timings.items())}); "
              f"p_HF {rec.spectrum.p_hf:.4f}; Delta_1 {1e3 * rec.spectrum.gap:.2f} mHa; C_QPE {rec.c_qpe:.4g} "
              f"({rec.qpe.encoding.kind}); {cheap}; peak RSS {peak[name]:.0f} MB", flush=True)
    t = time.perf_counter()
    report = scaling.scaling_report(list(records.values()))
    report["runtime_s"] = runtime
    with open(os.path.join(args.out, "scaling_fits.json"), "w") as fh:
        json.dump(report, fh, allow_nan=False)
    checks = reference_checks(records, report)
    n_match = sum(c[3] for c in checks)
    command = "benchmarks/run_m4_scaling.py" + (" --quick" if args.quick else "") + \
        (f" --systems {','.join(names)}" if args.systems else "") + f" --out {os.path.relpath(args.out)}" + \
        (" --no-cache" if args.no_cache else "")
    lines = ["# M4 scaling study: tables U1-U4 (repetition columns)", "",
             f"Generated by `{command}` (one thread). Systems: {', '.join(names)}.", "",
             scaling.markdown_tables(report),
             "## Reference numbers", "",
             f"{n_match} of {len(checks)} reproduce to the printed digits.", "",
             "| quantity | reference | reproduced | match |", "|---|---|---|---|"]
    lines += [f"| {q} | {e} | {v} | {'yes' if m else 'no'} |" for q, e, v, m in checks]
    lines += ["", "## Wall time (one thread)", "", "| system | seconds | stages (s) | peak RSS (MB) |", "|---|---|---|---|"]
    lines += [f"| {name} | {runtime[name]:.1f} | {', '.join(f'{k} {v:.1f}' for k, v in records[name].timings.items())} "
              f"| {peak[name]:.0f} |" for name in names]
    lines += ["", f"Report and tables: {time.perf_counter() - t:.1f} s; script: {time.perf_counter() - t_script:.1f} s; "
                  f"cache hits / misses: {cache.hits} / {cache.misses}.", ""]
    with open(os.path.join(args.out, "scaling_summary.md"), "w") as fh:
        fh.write("\n".join(lines))
    print(f"[reference] {n_match} of {len(checks)} match; wrote {len(names)} scaling_<system>.json, "
          f"scaling_fits.json and scaling_summary.md to {args.out} ({time.perf_counter() - t_script:.1f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
