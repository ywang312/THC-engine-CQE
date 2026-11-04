"""Steepest descent vs conjugate gradient vs L-BFGS for the exact CQE (milestone 2).

For each system and optimizer this script runs ``cqe.run_cqe`` from the HF
determinant for at most ``--max-iter`` iterations (default 60) in two
line-search modes:

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
  the smallest exact steps.

It reports the iterations and line-search energy evaluations needed to reach
E - E_FCI below each threshold (default 1e-3, 1.6e-3, 1e-5 and 1e-8 Ha; E_0 is
the HF energy, so 'iterations' counts CQE steps), and for the final state the
error, ||A|| and sigma^2 = <H^2> - <H>^2. The energy of the current state is
not counted as an evaluation.

Outputs: <out>/optimizers_<sys>.json per system and <out>/optimizers_summary.md.

Threads. ``--threads n`` sets PySCF's OpenMP threads and, through
OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / MKL_NUM_THREADS, numpy's BLAS threads.
BLAS libraries read those variables when numpy is first imported, so this file
imports numpy only after parsing its arguments; the BLAS pin takes effect only
if numpy is not loaded yet, as when the file runs as a script
(``settings["threads_env"]`` records what was in effect). Both matter for
bitwise reproducibility: observed on H8 with PySCF at one thread but OpenBLAS
at four, steepest descent needed 4004 instead of 3981 energy evaluations to
reach 1e-3, and errors differed by up to 2.3e-9.

Usage:
    python benchmarks/run_optimizers.py --systems h4,h6,h6s,lih,n2,n2s,h8 --threads 1 --out results/
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


def build_system(name: str):
    from thc_engine_cqe import systems

    table = {
        "h4": lambda: systems.hydrogen_chain(4, 1.8, "bohr"),
        "h6": lambda: systems.hydrogen_chain(6, 1.8, "bohr"),
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


def run_one(act, e_fci: float, opt: str, mode: str, args) -> dict:
    import numpy as np

    from thc_engine_cqe.cqe import run_cqe
    from thc_engine_cqe.residual import compute_residual

    t = time.time()
    cands = args.candidates if mode == "candidates" else None
    opts = optimizer_options(opt, args)
    res = run_cqe(act, max_iter=args.max_iter, e_exact=e_fci, optimizer=opt, optimizer_options=opts,
                  eps_candidates=cands, r_tol=args.r_tol, e_tol=args.e_tol)
    final = compute_residual(act, res.ci)
    errors = [float(x) for x in res.errors]
    cum = [int(x) for x in res.cumulative_energy_evals]
    return {
        "mode": mode,
        "optimizer": opt,
        "options": opts,
        "candidates": list(cands) if cands is not None else None,
        "steps": len(res.steps),
        "converged": bool(res.converged),
        "errors": errors,
        "cum_energy_evals": cum,
        "residual_norms": [float(s.residual_norm) for s in res.steps],
        "eps": [float(s.eps) for s in res.steps],
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
            r = run_one(act, e_fci, opt, mode, args)
            out["runs"].append(r)
            cells = " ".join(f"{k}:{v['iterations'] if v else '-'}" for k, v in r["reach"].items())
            print(f"[{name}] {mode:10s} {opt:5s} steps={r['steps']:3d} final={r['final_error']:.2e} "
                  f"|A|={r['final_residual_norm']:.1e} var={r['final_variance']:.1e} | {cells} "
                  f"({r['seconds']:.1f} s)", flush=True)
    out["seconds"] = time.time() - t0
    return out


# ----------------------------------------------------------------- markdown
def _cell(v: dict | None) -> str:
    return "-" if v is None else f"{v['iterations']} ({v['energy_evals']})"


def markdown_summary(results: list[dict], args) -> str:
    th = [f"{t:g}" for t in args.thresholds]
    lines = ["# CQE optimizers: steepest descent, conjugate gradient, L-BFGS", ""]
    lines.append(f"Exact CQE from HF, at most {args.max_iter} iterations, r_tol = {args.r_tol:g}, "
                 f"e_tol = {args.e_tol:g}. CG: beta = {args.cg_beta}, restart = {args.cg_restart}; "
                 f"L-BFGS: memory = {args.lbfgs_memory}. Each threshold cell gives the CQE steps to "
                 "E - E_FCI < threshold and, in parentheses, the line-search energy evaluations "
                 "used by then; '-' means not reached. 'resets' counts CG restarts after the first "
                 "step or L-BFGS memory resets. sigma^2 and ||A|| are those of the final state.")
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
        lines.append("| system | optimizer | " + " | ".join(f"< {t}" for t in th)
                     + " | steps | final E - E_FCI | final norm A | final sigma^2 | resets | monotone |")
        lines.append("|---|---|" + "---|" * (len(th) + 6))
        for res in results:
            for r in res["runs"]:
                if r["mode"] != mode:
                    continue
                cells = " | ".join(_cell(r["reach"][t]) for t in th)
                lines.append(f"| {res['system']} | {r['optimizer']} | {cells} | {r['steps']} | "
                             f"{r['final_error']:.2e} | {r['final_residual_norm']:.1e} | "
                             f"{r['final_variance']:.1e} | {r['restarts']} | {'yes' if r['monotone'] else 'no'} |")
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
    ap.add_argument("--systems", default="h4,h6,h6s,lih,n2,n2s,h8",
                    help="comma-separated subset of h4,h6,h6s,lih,n2,n2s,h8")
    ap.add_argument("--optimizers", default="sd,cg,lbfgs", help="comma-separated subset of sd,cg,lbfgs")
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
    if args.candidates.strip().lower() == "none":
        args.candidates, args.modes = None, ["exact"]
    else:
        args.candidates, args.modes = _floats(args.candidates), ["exact", "candidates"]
    args.threads_env = pin_threads(args.threads)
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
