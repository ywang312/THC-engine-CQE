"""Stiffness spectra of the linearized CQE and where the exact CQE's remaining error sits (M2).

For each system this script linearizes the CQE at the FCI ground state
(``diagnostics.linearize``, all coupled singlets; see its module docstring) and
reports the spectrum of K: the smallest and largest stiffness and the number of
null modes (lambda <= 1e-10 lambda_max), which are contaminations the residual
does not see at first order. It also linearizes in the sector the exact CQE
from HF actually populates, i.e. the states with weight > 1e-20 along the
trajectories of all three optimizers.

The exact CQE then runs from HF with steepest descent, CG and L-BFGS for
``--max-iter`` steps (default 60). The final E - E_0 is split into modes:

- energy in null modes;
- energy in soft modes, lambda < 1e-3 lambda_max (null modes excluded);
- "outside": higher orders and states outside the linearized sector.

``--long-iters n`` adds L-BFGS runs of n steps that record the null-mode energy
along the way (e.g. 2000 for N2).

Outputs: <out>/linearized_<sys>.json per system and <out>/linearized_summary.md.

Usage:
    python benchmarks/run_linearized.py --systems h4,h6,h6a,h6s,lih,n2,n2s --threads 1 --out results/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
OPTIMIZERS = ("sd", "cg", "lbfgs")


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


def pin_threads(n: int | None) -> dict:
    """Pin PySCF's OpenMP and numpy's BLAS threads (the BLAS pin needs numpy not yet loaded)."""
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
            **{var: os.environ.get(var) for var in THREAD_VARS}}


def split_energy(lin, ci, soft_rtol: float) -> dict:
    import numpy as np

    me = lin.mode_energies(ci)
    lam = lin.stiffness
    null = lin.null_modes()
    soft = (~null) & (lam < soft_rtol * lam.max())
    total = me["sector"] + me["outside"]
    return {"error": float(total), "null": float(me["per_mode"][null].sum()),
            "soft": float(me["per_mode"][soft].sum()),
            "stiff": float(me["per_mode"][~(null | soft)].sum()), "outside": me["outside"],
            "n_soft": int(soft.sum()), "largest_mode_share": float(np.max(me["per_mode"]) / total)
            if total > 0 else 0.0}


def run_system(name: str, args) -> dict:
    import numpy as np

    from thc_engine_cqe.cqe import run_cqe
    from thc_engine_cqe.diagnostics.linearized import linearize

    t0 = time.time()
    act = build_system(name)
    lin = linearize(act)
    lam = lin.stiffness
    out = {"system": name, "label": act.label, "norb": act.norb, "nelec": list(act.nelec),
           "e0": lin.e0, "coupled": {"n_states": lin.n_states, "n_null": int(lin.null_modes().sum()),
                                     "lambda_min_nonnull": float(lam[~lin.null_modes()].min()),
                                     "lambda_max": float(lam.max()),
                                     "lambda_smallest": [float(x) for x in lam[:8]]},
           "runs": []}
    trajectories = []
    finals = {}
    for opt in OPTIMIZERS:
        states = []
        res = run_cqe(act, max_iter=args.max_iter, optimizer=opt, e_exact=lin.e0,
                      callback=lambda step, ci: states.append(np.asarray(ci).copy()))
        trajectories.extend(states)
        finals[opt] = res
    sec = linearize(act, sector="references", references=trajectories, weight_tol=1e-20)
    out["trajectory_sector"] = {"n_states": sec.n_states, "n_null": int(sec.null_modes().sum()),
                                "lambda_min_nonnull": float(sec.stiffness[~sec.null_modes()].min())}
    for opt, res in finals.items():
        split = split_energy(lin, res.ci, args.soft_rtol)
        out["runs"].append({"optimizer": opt, "steps": len(res.steps), "final_error": float(res.errors[-1]),
                            "split": split})
    if args.long_iters:
        rec = []
        null = lin.null_modes()

        def cb(step, ci):
            it = step.iteration + 1
            if it in args.long_marks or it == args.long_iters:
                me = lin.mode_energies(ci)
                rec.append({"iteration": it, "error": me["sector"] + me["outside"],
                            "null": float(me["per_mode"][null].sum())})

        run_cqe(act, max_iter=args.long_iters, optimizer="lbfgs", r_tol=0.0, e_tol=0.0, callback=cb)
        out["long_lbfgs"] = rec
    out["seconds"] = time.time() - t0
    return out


def markdown_summary(results: list[dict], args) -> str:
    lines = ["# Linearized CQE at the FCI ground state", "",
             "K = W^{1/2} S W^{1/2} over all singlets coupled to the ground state by a two-body operator "
             "(diagnostics/linearized.py). Null: lambda <= 1e-10 lambda_max. The trajectory sector is the "
             "set of states with weight > 1e-20 along the exact SD, CG and L-BFGS runs from HF (it keeps "
             "the spatial symmetry of HF).", "",
             "| system | coupled singlets | null | smallest non-null lambda | lambda_max | trajectory sector | null in sector |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        c, s = r["coupled"], r["trajectory_sector"]
        lines.append(f"| {r['system']} | {c['n_states']} | {c['n_null']} | {c['lambda_min_nonnull']:.2e} | "
                     f"{c['lambda_max']:.1f} | {s['n_states']} | {s['n_null']} |")
    lines += ["", f"## Where the remaining error sits after {args.max_iter} exact steps from HF", "",
              f"Soft: non-null modes with lambda < {args.soft_rtol:g} lambda_max. Outside: higher orders "
              "and states outside the linearized sector. Energies in Ha.", "",
              "| system | optimizer | E - E_0 | null modes | soft modes | stiff modes | outside |",
              "|---|---|---|---|---|---|---|"]
    for r in results:
        for run in r["runs"]:
            sp = run["split"]
            lines.append(f"| {r['system']} | {run['optimizer']} | {run['final_error']:.2e} | {sp['null']:.2e} | "
                         f"{sp['soft']:.2e} ({sp['n_soft']}) | {sp['stiff']:.2e} | {sp['outside']:.1e} |")
    long = [r for r in results if r.get("long_lbfgs")]
    if long:
        lines += ["", "## Long L-BFGS runs: energy in null modes", "",
                  "| system | iteration | E - E_0 | in null modes |", "|---|---|---|---|"]
        for r in long:
            for rec in r["long_lbfgs"]:
                lines.append(f"| {r['system']} | {rec['iteration']} | {rec['error']:.3e} | {rec['null']:.3e} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", default="h4,h6,h6a,h6s,lih,n2,n2s")
    ap.add_argument("--max-iter", type=int, default=60)
    ap.add_argument("--soft-rtol", type=float, default=1e-3)
    ap.add_argument("--long-iters", type=int, default=0)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default="results/")
    args = ap.parse_args(argv)
    args.long_marks = (10, 30, 100, 300, 1000)
    args.threads_env = pin_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    results = []
    for name in [s.strip() for s in args.systems.split(",") if s.strip()]:
        res = run_system(name, args)
        results.append(res)
        with open(os.path.join(args.out, f"linearized_{name}.json"), "w") as fh:
            json.dump(res, fh, indent=1)
        c = res["coupled"]
        print(f"[{name}] coupled={c['n_states']} null={c['n_null']} lam_min={c['lambda_min_nonnull']:.2e} "
              f"sector={res['trajectory_sector']['n_states']} null_sector={res['trajectory_sector']['n_null']} "
              f"({res['seconds']:.1f} s)", flush=True)
    with open(os.path.join(args.out, "linearized_summary.md"), "w") as fh:
        fh.write(markdown_summary(results, args) + "\n")
    print(f"wrote {os.path.join(args.out, 'linearized_summary.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
