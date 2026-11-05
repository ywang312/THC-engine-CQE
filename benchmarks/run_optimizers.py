"""Steepest descent vs conjugate gradient vs L-BFGS for the exact CQE (milestone 2), per metric.

For each system, optimizer and metric this script runs ``cqe.run_cqe`` from the
HF determinant for at most ``--max-iter`` iterations (default 60) in two
line-search modes. ``--metrics`` (default "spin_free,spin_orbital") selects the
metrics of ``run_cqe(..., metric=...)``: "spin_free" is the M2 direction, and
"spin_orbital" preconditions every optimizer with M = (2 I + T)/3,
(T X)[p,q,r,s] = X[p,s,r,q], for which steepest descent is the standard
spin-orbital ACSE generator (``thc_engine_cqe.optimizers``). The modes:

* ``exact``: the exact Krylov line search, an emulator oracle. Its energy
  evaluations are the grid and refinement points of every bracket
  (``CQEStep.n_energy_evals``; observed 71-102 per iteration, mean 79-89).
* ``candidates``: fixed step sizes (``--candidates``), one energy per candidate
  and iteration, the lowest taken even if it raises the energy, as on
  hardware. The default set 0.00625 * 2^k, k = 0..10 (0.00625 to 6.4), serves
  all three optimizers. Observed with the exact line search on these systems:
  after the first step (eps 0.05-0.3, d = g for all three) the median eps is
  0.02-0.09 for steepest descent, 0.04-0.14 for CG and 1.2-1.9 for L-BFGS,
  whose directions are scaled like quasi-Newton steps (10-90% range 0.27-6.9).
  Isolated exact L-BFGS steps are much longer (observed: 136 on H4 at step 12,
  4.8e-6 Ha above FCI; up to 23 on N2 and 8.2 on LiH); a set capped at 6.4
  truncates them. Observed: with 0.025 as the smallest candidate, steepest
  descent and CG diverge on H8. Every candidate overshoots from step 6 (steepest
  descent) or step 27 (CG) on, the energy then climbs for several steps, and
  both runs end about 0.7 Ha above FCI after 60 steps. The set must reach below
  the smallest exact steps. With metric "spin_orbital" the directions are
  shorter (M has the eigenvalues 1 and 1/3) and the exact steps longer: first
  step 0.12-0.38, then median eps 0.049-0.26 for steepest descent (10-90% range
  per system 0.035-0.35, every step 0.034-0.77), 0.09-0.34 for CG (0.06-0.8,
  every step 0.037-2.5) and 1.2-2.0 for L-BFGS (0.5-8.5; 15-20 of 427 steps
  above 6.4, up to 1.9e3 on N2), in two runs on different machines. The default
  set therefore still spans every exact steepest-descent and CG step of both
  metrics (observed on h4, h6, h6a, h6s, lih, n2, n2s, h8, 60 steps).

It reports the iterations and line-search energy evaluations needed to reach
E - E_FCI below each threshold (default 1e-3, 1.6e-3, 1e-5 and 1e-8 Ha; E_0 is
the HF energy, so 'iterations' counts CQE steps), and for the final state the
error, ||A|| and sigma^2 = <H^2> - <H>^2. The energy of the current state is
not counted as an evaluation. A step-size table gives, per run, the first eps,
the median and 10-90% range of the later ones, how many exact-search steps fall
outside the candidate range, and how often the candidate runs chose the smallest
or the largest candidate (a sign that the set does not span the steps needed).

Outputs: <out>/optimizers_<sys>.json per system and <out>/optimizers_summary.md.

Threads. ``--threads n`` sets PySCF's OpenMP threads and, through
OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / MKL_NUM_THREADS, numpy's BLAS threads.
BLAS libraries read those variables when numpy is first imported, so this file
imports numpy only after parsing its arguments; the BLAS pin takes effect only
if numpy is not loaded yet, as when the file runs as a script
(``settings["threads_env"]`` records what was in effect). Both matter for
bitwise reproducibility: observed on H8 with PySCF at one thread but OpenBLAS
at four, steepest descent needed 4004 instead of 3981 energy evaluations to
reach 1e-3, and errors differed by up to 2.3e-9. Across machines (different
BLAS kernels) the rounding differs and long runs amplify it. Observed between
two machines, one thread each: E_HF and E_FCI agreed to 3e-13 Ha, but the
error trajectories differed by up to 7e-6 Ha (exact search) and 2e-5 Ha
(candidates) on the H chains and LiH, 6e-5 Ha on N2 and 1.5e-2 Ha on N2 at
2 A. Three of the 96 runs moved by one step: one stopping step and two
1e-5 Ha counts (none of the 1.6 mHa counts).

Usage:
    python benchmarks/run_optimizers.py --systems h4,h6,h6a,h6s,lih,n2,n2s,h8 --threads 1 --out results/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# numpy (and everything that imports it) is imported inside the functions, after
# --threads has set the BLAS thread variables; see the module docstring.

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
DEFAULT_CANDIDATES = "0.00625,0.0125,0.025,0.05,0.1,0.2,0.4,0.8,1.6,3.2,6.4"
DEFAULT_THRESHOLDS = "1e-3,1.6e-3,1e-5,1e-8"
DEFAULT_METRICS = "spin_free,spin_orbital"
LABELS = {"sd": "SD", "cg": "CG", "lbfgs": "L-BFGS"}
METRIC_TAGS = {"spin_free": "", "spin_orbital": " (so)"}


def build_system(name: str):
    from thc_engine_cqe import systems

    table = {
        "h4": lambda: systems.hydrogen_chain(4, 1.8, "bohr"),
        "h6": lambda: systems.hydrogen_chain(6, 1.8, "bohr"),
        "h6a": lambda: systems.hydrogen_chain(6, 1.0, "angstrom"),
        "h6s": lambda: systems.hydrogen_chain(6, 3.0, "bohr"),
        "h8": lambda: systems.hydrogen_chain(8, 1.8, "bohr"),
        "lih": lambda: systems.lih(),
        "n2": lambda: systems.n2(1.098),
        "n2s": lambda: systems.n2(2.0),
    }
    if name not in table:
        raise SystemExit(f"unknown system {name!r}; choose from {sorted(table)}")
    return table[name]()


def _floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def reach(errors, cum_evals, threshold: float) -> dict | None:
    """First CQE step k with E_k - E_FCI < threshold, and the evaluations used by then."""
    import numpy as np

    hits = np.flatnonzero(np.asarray(errors) < threshold)
    if not hits.size:
        return None
    k = int(hits[0])
    return {"iterations": k, "energy_evals": int(cum_evals[k])}


def optimizer_options(name: str, args) -> dict | None:
    if name == "cg":
        opts = {"beta": args.cg_beta}
        if args.cg_restart is not None:
            opts["restart"] = args.cg_restart
        return opts
    if name == "lbfgs":
        return {"memory": args.lbfgs_memory}
    return None


def _restarts(res) -> int:
    """CG restarts after the first step (periodic or slope), or L-BFGS memory resets."""
    n = 0
    for s in res.steps:
        info = s.optimizer_info or {}
        if info.get("restart") not in (None, "first") or info.get("reset"):
            n += 1
    return n


def step_sizes(eps, candidates) -> dict:
    """Step-size statistics of one run: first step, then the later nonzero steps.

    A run that stops on r_tol records a final step with eps = 0; zero steps are
    counted ('zero') and left out of the statistics. ``outside`` counts later steps
    below the smallest / above the largest of ``candidates`` (the default set when
    the run used the exact search); ``at_edges`` counts steps equal to the smallest /
    largest candidate (meaningful for candidate runs).
    """
    import numpy as np

    eps = np.asarray(eps, dtype=float)
    cand = np.asarray(candidates, dtype=float)
    later = eps[1:][eps[1:] != 0.0]
    out = {"first": float(eps[0]) if eps.size else None, "n_later": int(later.size),
           "zero": int(np.sum(eps == 0.0))}
    if later.size:
        p10, p50, p90 = (float(x) for x in np.percentile(later, [10, 50, 90]))
        out.update(median=p50, p10=p10, p90=p90, min=float(later.min()), max=float(later.max()))
    else:
        out.update(median=None, p10=None, p90=None, min=None, max=None)
    if cand.size:
        lo, hi = float(cand.min()), float(cand.max())
        out["outside"] = [int(np.sum(later < lo)), int(np.sum(later > hi))]
        out["at_edges"] = [int(np.sum(eps == lo)), int(np.sum(eps == hi))]
    return out


def run_one(act, e_fci: float, opt: str, metric: str, mode: str, args) -> dict:
    import numpy as np

    from thc_engine_cqe.cqe import run_cqe
    from thc_engine_cqe.residual import compute_residual

    t = time.time()
    cands = args.candidates if mode == "candidates" else None
    opts = optimizer_options(opt, args)
    res = run_cqe(act, max_iter=args.max_iter, e_exact=e_fci, optimizer=opt, optimizer_options=opts,
                  eps_candidates=cands, r_tol=args.r_tol, e_tol=args.e_tol, metric=metric)
    final = compute_residual(act, res.ci)
    errors = [float(x) for x in res.errors]
    cum = [int(x) for x in res.cumulative_energy_evals]
    eps = [float(s.eps) for s in res.steps]
    return {
        "mode": mode,
        "optimizer": opt,
        "metric": res.metric,
        "options": opts,
        "candidates": list(cands) if cands is not None else None,
        "steps": len(res.steps),
        "converged": bool(res.converged),
        "errors": errors,
        "cum_energy_evals": cum,
        "residual_norms": [float(s.residual_norm) for s in res.steps],
        "eps": eps,
        "step_sizes": step_sizes(eps, args.candidates or []),
        "captured_descent": [float(s.captured_descent) for s in res.steps],
        "monotone": bool(np.all(np.diff(res.energies) <= 1e-12)),
        "restarts": _restarts(res),
        "reach": {f"{t_:g}": reach(errors, cum, t_) for t_ in args.thresholds},
        "final_error": errors[-1],
        "final_residual_norm": float(final.residual_norm),
        "final_variance": float(final.variance),
        "seconds": time.time() - t,
    }


def run_system(name: str, args) -> dict:
    from thc_engine_cqe.fci_ops import FCISpace, Hamiltonian

    t0 = time.time()
    act = build_system(name)
    e_fci = float(act.fci()[0])
    e_hf = float(Hamiltonian(act).energy(FCISpace.of(act).hf()))
    settings = {k: v for k, v in vars(args).items() if k != "func"}
    out = {"system": name, "label": act.label, "norb": act.norb, "nelec": list(act.nelec),
           "e_fci": e_fci, "e_hf": e_hf, "settings": settings, "runs": []}
    print(f"[{name}] {act.label}: N={act.norb}, nelec={act.nelec}, E_HF-E_FCI={e_hf - e_fci:.3e}", flush=True)
    for mode in args.modes:
        for opt in args.optimizers:
            for metric in args.metrics:
                r = run_one(act, e_fci, opt, metric, mode, args)
                out["runs"].append(r)
                cells = " ".join(f"{k}:{v['iterations'] if v else '-'}" for k, v in r["reach"].items())
                print(f"[{name}] {mode:10s} {opt:5s} {metric:12s} steps={r['steps']:3d} "
                      f"final={r['final_error']:.2e} |A|={r['final_residual_norm']:.1e} "
                      f"var={r['final_variance']:.1e} | {cells} ({r['seconds']:.1f} s)", flush=True)
    out["seconds"] = time.time() - t0
    return out


# ----------------------------------------------------------------- markdown
def _cell(v: dict | None) -> str:
    return "-" if v is None else f"{v['iterations']} ({v['energy_evals']})"


def _num(x, fmt: str) -> str:
    return "-" if x is None else format(x, fmt)


def _label(opt: str, metric: str) -> str:
    return LABELS.get(opt, opt) + METRIC_TAGS.get(metric, f" ({metric})")


def _pivot_thresholds(th: list[str]) -> list[str]:
    """The chemical-accuracy and tight thresholds of the pivot tables (1.6 mHa and 1e-5 if run)."""
    pick = [t for t in ("0.0016", "1e-05") if t in th]
    return pick if len(pick) == 2 else [th[0], th[-1]]


def markdown_summary(results: list[dict], args) -> str:
    th = [f"{t:g}" for t in args.thresholds]
    pairs = [(o, m) for o in args.optimizers for m in args.metrics]
    lines = ["# CQE optimizers: steepest descent, conjugate gradient, L-BFGS, per metric", ""]
    lines.append(f"Exact CQE from HF, at most {args.max_iter} iterations, r_tol = {args.r_tol:g}, "
                 f"e_tol = {args.e_tol:g}. CG: beta = {args.cg_beta}, restart = {args.cg_restart}; "
                 f"L-BFGS: memory = {args.lbfgs_memory}. Metrics: {', '.join(args.metrics)}. "
                 "'spin_orbital' (marked '(so)') preconditions every optimizer with M = (2I + T)/3, "
                 "(T X)[p,q,r,s] = X[p,s,r,q]; its steepest descent steps along the standard "
                 "spin-orbital ACSE generator (exactly, for singlets), CG uses z = M g and L-BFGS "
                 "H_0 = gamma M. Each threshold cell gives the CQE steps to E - E_FCI < threshold and, "
                 "in parentheses, the line-search energy evaluations used by then; '-' means not "
                 "reached. 'resets' counts CG restarts after the first step or L-BFGS memory resets. "
                 "sigma^2 and ||A|| are those of the final state.")
    lines.append("")
    a, b = _pivot_thresholds(th)
    for mode in args.modes:
        title = "exact line search" if mode == "exact" else "fixed candidates"
        lines.append(f"## Steps to E - E_FCI < {a} / < {b} Ha ({title})")
        lines.append("")
        lines.append("| system | " + " | ".join(_label(o, m) for o, m in pairs) + " |")
        lines.append("|---|" + "---|" * len(pairs))
        for res in results:
            runs = {(r["optimizer"], r["metric"]): r for r in res["runs"] if r["mode"] == mode}
            cells = []
            for key in pairs:
                r = runs.get(key)
                if r is None:
                    cells.append("")
                    continue
                ra, rb = r["reach"][a], r["reach"][b]
                cells.append(f"{ra['iterations'] if ra else '-'} / {rb['iterations'] if rb else '-'}")
            lines.append(f"| {res['system']} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append(f"Final E - E_FCI after at most {args.max_iter} steps ({title}):")
        lines.append("")
        lines.append("| system | " + " | ".join(_label(o, m) for o, m in pairs) + " |")
        lines.append("|---|" + "---|" * len(pairs))
        for res in results:
            runs = {(r["optimizer"], r["metric"]): r for r in res["runs"] if r["mode"] == mode}
            lines.append(f"| {res['system']} | " + " | ".join(
                f"{runs[k]['final_error']:.1e}" if k in runs else "" for k in pairs) + " |")
        lines.append("")
    for mode in args.modes:
        if mode == "exact":
            lines.append("## Exact line search")
            lines.append("")
            lines.append("Energy evaluations are those of the emulator's line search (grid and "
                         "refinement points of every bracket).")
        else:
            lines.append("## Fixed step-size candidates")
            lines.append("")
            lines.append(f"Candidates eps = {', '.join(f'{c:g}' for c in args.candidates)} multiply "
                         f"the direction d; {len(args.candidates)} energies per step.")
        lines.append("")
        lines.append("| system | optimizer | metric | " + " | ".join(f"< {t}" for t in th)
                     + " | steps | final E - E_FCI | final norm A | final sigma^2 | resets | monotone |")
        lines.append("|---|---|---|" + "---|" * (len(th) + 6))
        for res in results:
            for r in res["runs"]:
                if r["mode"] != mode:
                    continue
                cells = " | ".join(_cell(r["reach"][t]) for t in th)
                lines.append(f"| {res['system']} | {r['optimizer']} | {r['metric']} | {cells} | {r['steps']} | "
                             f"{r['final_error']:.2e} | {r['final_residual_norm']:.1e} | "
                             f"{r['final_variance']:.1e} | {r['restarts']} | {'yes' if r['monotone'] else 'no'} |")
        lines.append("")
    lines.append("## Step sizes")
    lines.append("")
    if args.candidates:
        lo, hi = min(args.candidates), max(args.candidates)
        lines.append(f"eps of every run: the first step, then the later nonzero steps (median, 10-90% "
                     f"range, min-max). 'outside' counts later exact-search steps below {lo:g} / above "
                     f"{hi:g}, the ends of the candidate set; 'at ends' counts steps of the candidate "
                     f"runs at the smallest / largest candidate.")
    else:
        lines.append("eps of every run: the first step, then the later nonzero steps (median, 10-90% "
                     "range, min-max).")
    lines.append("")
    lines.append("| system | optimizer | metric | mode | first | median | 10-90% | min-max | outside | at ends |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for res in results:
        for r in res["runs"]:
            st = r["step_sizes"]
            rng_ = (f"{_num(st['p10'], '.3g')}-{_num(st['p90'], '.3g')}" if st["p10"] is not None else "-")
            mm = (f"{_num(st['min'], '.3g')}-{_num(st['max'], '.3g')}" if st["min"] is not None else "-")
            outside = "/".join(str(x) for x in st["outside"]) if r["mode"] == "exact" and "outside" in st else ""
            edges = "/".join(str(x) for x in st["at_edges"]) if r["mode"] == "candidates" and "at_edges" in st else ""
            lines.append(f"| {res['system']} | {r['optimizer']} | {r['metric']} | {r['mode']} | "
                         f"{_num(st['first'], '.3g')} | {_num(st['median'], '.3g')} | {rng_} | {mm} | "
                         f"{outside} | {edges} |")
    lines.append("")
    return "\n".join(lines)


def pin_threads(n: int | None) -> dict:
    """Pin PySCF's OpenMP threads and numpy's BLAS threads to ``n`` (None: inherit).

    Returns the thread settings in effect, for the JSON. The BLAS pin is set through
    the environment and works only if numpy is not loaded yet ("blas_pinned").
    """
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", default="h4,h6,h6a,h6s,lih,n2,n2s,h8",
                    help="comma-separated subset of h4,h6,h6a,h6s,lih,n2,n2s,h8")
    ap.add_argument("--optimizers", default="sd,cg,lbfgs", help="comma-separated subset of sd,cg,lbfgs")
    ap.add_argument("--metrics", default=DEFAULT_METRICS,
                    help="comma-separated subset of spin_free,spin_orbital (run_cqe's metric)")
    ap.add_argument("--max-iter", type=int, default=60)
    ap.add_argument("--thresholds", type=_floats, default=_floats(DEFAULT_THRESHOLDS),
                    help="E - E_FCI thresholds in Ha (comma-separated)")
    ap.add_argument("--candidates", default=DEFAULT_CANDIDATES,
                    help="fixed step sizes of the 'candidates' mode (comma-separated), or 'none'")
    ap.add_argument("--cg-beta", default="pr+", choices=("pr+", "fr"))
    ap.add_argument("--cg-restart", type=int, default=None)
    ap.add_argument("--lbfgs-memory", type=int, default=10)
    ap.add_argument("--r-tol", type=float, default=1e-7)
    ap.add_argument("--e-tol", type=float, default=1e-11)
    ap.add_argument("--threads", type=int, default=None,
                    help="PySCF OpenMP and numpy BLAS threads (default: inherit)")
    ap.add_argument("--out", default="results/", help="output directory")
    args = ap.parse_args(argv)
    args.optimizers = [s.strip() for s in args.optimizers.split(",") if s.strip()]
    args.metrics = [s.strip() for s in args.metrics.split(",") if s.strip()]
    if args.candidates.strip().lower() == "none":
        args.candidates, args.modes = None, ["exact"]
    else:
        args.candidates, args.modes = _floats(args.candidates), ["exact", "candidates"]
    args.threads_env = pin_threads(args.threads)
    # imported only after pin_threads, because it imports numpy (module docstring)
    from thc_engine_cqe.optimizers import METRICS

    bad = [m for m in args.metrics if m not in METRICS]
    if bad or not args.metrics:
        raise SystemExit(f"unknown metrics {bad}; choose from {METRICS}")
    os.makedirs(args.out, exist_ok=True)
    names = [s.strip() for s in args.systems.split(",") if s.strip()]
    results = []
    for name in names:
        res = run_system(name, args)
        results.append(res)
        with open(os.path.join(args.out, f"optimizers_{name}.json"), "w") as fh:
            json.dump(res, fh, indent=1)
        print(f"[{name}] done in {res['seconds']:.1f} s", flush=True)
    with open(os.path.join(args.out, "optimizers_summary.md"), "w") as fh:
        fh.write(markdown_summary(results, args) + "\n")
    print(f"wrote {os.path.join(args.out, 'optimizers_summary.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
