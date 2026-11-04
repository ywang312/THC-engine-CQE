#!/usr/bin/env python
"""THC floors benchmark: how the THC rank controls the quantities the CQE uses.

For each system, this fits THC models over a sweep of ranks M (plus DF comparison
rows) and records the integral error, lambda, the energy errors and the
residual/variance floors of
:func:`thc_engine_cqe.diagnostics.thc_floors.thc_floor_scan`. The definitions are in
that module's docstring. It writes ``<out>/thc_floors_<system>.json`` and prints a
markdown table for each system.

Examples::

    python benchmarks/run_thc_floors.py                          # h4, h6, lih; auto ranks
    python benchmarks/run_thc_floors.py --systems h4 --ranks 4,8,10 --no-refine
    python benchmarks/run_thc_floors.py --systems n2,h8 --out /tmp/floors

Systems: h4/h6/h8 = linear H_n at 1.8 bohr (STO-3G, all orbitals active),
lih = LiH at 1.595 angstrom (STO-3G, all orbitals active), n2 = N2 at 1.098 angstrom
(STO-3G, CAS(6,6)).

``--ranks auto`` sweeps M/N in {1, 1.5, 2, 3, 4, ...} below N(N+1)/2 and then adds
M = N(N+1)/2 (:func:`~thc_engine_cqe.diagnostics.thc_floors.auto_ranks`).

THC lambda and FT costs: the fits are unregularized by default (``--penalty 0``),
unlike OpenFermion's THC pipeline. Their lambda is not a reliable cost estimate; see
the "THC lambda and the fit penalty" section of
:mod:`thc_engine_cqe.diagnostics.thc_floors`. ``--penalty auto`` uses OpenFermion's
automatic penalty rule.

Reproducibility (observations on one machine, not tests). With several OpenMP
threads, PySCF's SCF gives orbitals, and hence integrals, that differ in the last
bits from run to run. This script therefore builds each system with one OpenMP
thread (:func:`build_system`). Everything after that may use threads. On this
machine, with that setting, every THC and DF fit (lambda, ERI errors, refinement
objective) was bitwise identical for OMP_NUM_THREADS = 1, 2 and 4. The FCI-based
floors agreed to about 1e-12 relative, so only round-off-level entries of the table
changed.

The THC rows differ in how they react when the integrals change slightly. That
happens with another machine, BLAS, or PySCF or scipy version, or with integrals
built with several threads:

* A refinement (L-BFGS-B) that converges to a minimum with an objective well above
  zero is insensitive. With the defaults, those rows (h4 M=4 and 6, h6 M=6,
  lih M=6) agreed to all printed digits when the integrals varied in their last
  bits.
* A refinement that stops at ``--maxiter`` is marked with a dagger, and the legend
  gives its iteration count. Its ERI error is stable to a few percent, but its
  floors are not. For lih M=12, the delta E_FCI of four runs with integrals that
  varied in their last bits ranged from -2.9e-3 to -3.7e-4 Ha, and the
  unregularized lambda from 15 to 57. Fits that are equally good in the Frobenius
  norm can have quite different energy errors, so such a row shows the size of a
  floor at that rank, not its value.
* A rank with enough parameters to fit the ERIs almost exactly (h4 M=8, h6 M=18)
  lets the objective creep towards zero. L-BFGS-B then stops at ``--maxiter``, or
  on its relative-reduction test, which for an objective below 1 is a test on the
  absolute decrease. Such rows have tiny floors (below 1e-4 Ha here), but even the
  leading digit of those floors can change.

A small ``--penalty`` of 1e-7 or 1e-6 kept lambda at most 1.4 times the DF value
for lih and h6 at M = 9 to 18. The ERI error changed by less than a factor of two,
and often decreased. A penalty of 1e-5 raised the ERI error of lih M=18 eightyfold.
No penalty made the floors of unconverged rows robust. The JSON records the thread
settings. ``--no-refine`` (least-squares fits only) is robust.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

from thc_engine_cqe import systems
from thc_engine_cqe.diagnostics.thc_floors import (DEFAULT_DF_THRESHOLDS, FT_DEFAULTS,
                                                    ExactReference, auto_ranks,
                                                    format_markdown, thc_floor_scan)

SYSTEMS = {
    "h4": lambda: systems.hydrogen_chain(4, 1.8, "bohr"),
    "h6": lambda: systems.hydrogen_chain(6, 1.8, "bohr"),
    "h8": lambda: systems.hydrogen_chain(8, 1.8, "bohr"),
    "lih": lambda: systems.lih(),
    "n2": lambda: systems.n2(r=1.098, basis="sto-3g", ncas=6, nelecas=6),
}


def parse_ranks(text: str) -> list[int] | None:
    """``'auto'`` -> None (per-system sweep), otherwise a comma-separated list of ints."""
    text = text.strip().lower()
    if text == "auto":
        return None
    try:
        ranks = [int(x) for x in text.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--ranks must be 'auto' or a comma list of ints: {exc}")
    if not ranks or min(ranks) < 1:
        raise argparse.ArgumentTypeError("--ranks needs at least one positive integer")
    return ranks


def parse_penalty(text: str) -> float | str:
    """``'auto'`` (OpenFermion's rule) or a non-negative float."""
    text = text.strip().lower()
    if text == "auto":
        return "auto"
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--penalty must be 'auto' or a number: {exc}")
    if not value >= 0.0:
        raise argparse.ArgumentTypeError("--penalty must be non-negative")
    return value


def parse_systems(text: str) -> list[str]:
    names = [x.strip().lower() for x in text.split(",") if x.strip()]
    bad = [x for x in names if x not in SYSTEMS]
    if bad or not names:
        raise argparse.ArgumentTypeError(f"unknown system(s) {bad}; choose from {sorted(SYSTEMS)}")
    return names


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def _environment() -> dict:
    import pyscf
    import scipy

    try:
        import openfermion
        of_version = openfermion.__version__
    except Exception:
        of_version = None
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pyscf": pyscf.__version__, "openfermion": of_version,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "cpu_count": os.cpu_count(), "integral_threads": 1}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--systems", type=parse_systems, default=parse_systems("h4,h6,lih"),
                   help=f"comma list from {sorted(SYSTEMS)} (default: h4,h6,lih)")
    p.add_argument("--ranks", type=parse_ranks, default=None,
                   help="comma list of THC ranks M, or 'auto' (default)")
    p.add_argument("--no-refine", action="store_true", help="least-squares THC fits only")
    p.add_argument("--no-df", action="store_true", help="skip the double-factorization rows")
    p.add_argument("--no-ft", action="store_true", help="skip OpenFermion's THC cost model")
    p.add_argument("--guess", choices=("grid", "random"), default="grid",
                   help="THC leaf guess (grid falls back to random without a molecule)")
    p.add_argument("--seed", type=int, default=0, help="seed of the random guess")
    p.add_argument("--maxiter", type=int, default=None,
                   help="L-BFGS-B iteration limit (default: fit_thc's 5000)")
    p.add_argument("--penalty", type=parse_penalty, default=0.0,
                   help="lambda penalty of the THC refinement: a number, or 'auto' for "
                        "OpenFermion's rule (default: 0, unregularized)")
    p.add_argument("--out", default="results", help="output directory (default: results/)")
    p.add_argument("--quiet", action="store_true", help="no progress messages on stderr")
    return p


def build_system(name: str):
    """The active space of ``name``, built with one OpenMP thread.

    PySCF's multi-threaded SCF gives orbitals that differ in the last bits from run
    to run (module docstring), and fits that stopped at ``--maxiter`` amplify that.
    These systems are small, so building them single-threaded costs little.
    """
    from pyscf import lib

    with lib.with_omp_threads(1):
        return SYSTEMS[name]()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    fit_kwargs = {"penalty": args.penalty}
    if args.maxiter is not None:
        fit_kwargs["maxiter"] = args.maxiter
    env = _environment()
    for name in args.systems:
        t0 = time.perf_counter()
        active = build_system(name)
        ranks = auto_ranks(active.norb) if args.ranks is None else list(args.ranks)
        ref = ExactReference.build(active)

        def progress(row, _name=name):
            if args.quiet:
                return
            details = row.get("details") or {}
            note = ""
            if row.get("kind") == "thc" and details.get("refine_success") is False:
                note = f" (refinement not converged: nit = {details.get('refine_nit')})"
            print(f"  [{_name}] {row['label']}: {row['seconds']:.2f} s{note}", file=sys.stderr,
                  flush=True)

        rows = thc_floor_scan(active, ranks, guess=args.guess, refine=not args.no_refine,
                              seed=args.seed, include_df=not args.no_df, ft_cost=not args.no_ft,
                              fit_kwargs=fit_kwargs, reference=ref, progress=progress)
        elapsed = time.perf_counter() - t0
        payload = {
            "system": name, "label": active.label, "norb": active.norb,
            "nelec": list(active.nelec), "e_fci": ref.energy, "e_hf": ref.e_hf,
            "res_hf": ref.res_hf, "s2": ref.state.s2,
            "settings": {"ranks": ranks, "guess": args.guess, "refine": not args.no_refine,
                         "seed": args.seed, "include_df": not args.no_df,
                         "df_thresholds": list(DEFAULT_DF_THRESHOLDS),
                         "ft_cost": not args.no_ft, "ft_params": dict(FT_DEFAULTS),
                         "fit_kwargs": fit_kwargs},
            "environment": env, "elapsed_s": elapsed, "rows": rows,
        }
        path = os.path.join(args.out, f"thc_floors_{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1, default=_json_default)
        print(f"## {name}: {active.label} (N = {active.norb}, nelec = {active.nelec})\n")
        print(format_markdown(rows))
        print(f"\nWrote {path} ({elapsed:.1f} s).\n", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
