"""The finite-shot CQE: noisy residuals and energies, floors, optimizers and step candidates (M2, workflow C).

Every run uses ``cqe.run_cqe`` with the estimator hooks of :mod:`thc_engine_cqe.sampling`; the exact
energies and residuals are recorded next to the estimates, so every error below is the exact
E - E_FCI of the state the noisy run produced. Systems (STO-3G): h4 (H4 at 1.8 bohr), h6a (H6 at 1.0 A),
lih (LiH at 1.595 A), n2 (N2 CAS(6,6) at 1.098 A); h6 (H6 at 1.8 bohr) is optional. Errors and energies
in Ha, shots are counts (one shot = one measured circuit; a matchgate snapshot is one shot).

Noise models (all frozen at the FCI ground state, the linearization point of the theory)
---------------------------------------------------------------------------------------
* Residual: the direct route A_e = <X_e> with matchgate shadows on the 840 (N = 6) or 150 (H4)
  independent spin-orbital elements (``sampling.ResidualNoise``, route "direct"), in two Gaussian forms:
  - "corr": correlated Gaussian with the per-shot element covariance C of real snapshots,
    ``SnapshotResidual.element_covariance(c_fci, n, mean=0)`` (n = 2e5 snapshots for H4, 2e4 for N = 6,
    ``--cov-snapshots``); this is the model of E2, E3 and E5;
  - "indep": independent elements with the exact (H4) or Rao-Blackwellized (N = 6, 2e4 matchings, seeded)
    per-element variances of ``sampling.element_variances``.
  The sampling build observed that real snapshots give 1.32-1.41 times the independent-element floor on
  H4 and that the correlated Gaussian reproduces the snapshot floors; E4 repeats that test.
  The aux route (E1 only) uses independent elements with ``element_variances(route="aux")``.
* Energy (E3): DF basis rotations with shifted leaves (``EnergyNoise`` scheme "df", ``shift=True``);
  untruncated pivoted-Cholesky leaves, except N2, which uses the leaves of the supermatrix
  eigendecomposition (``first="eigh"``) because RHF returns its degenerate pi orbitals in an arbitrary
  orientation and the Cholesky pivots, hence the Cholesky DF variance, depend on it (workflow B; the
  eigh-leaf variance, the matchgate variances and all physical quantities do not). The per-shot variance
  V(Psi) is evaluated once per CQE iteration at the current state and reused for that iteration's
  candidates (``IterationEnergyNoise``; the library's ``variance="exact"`` evaluates it at every
  candidate state, which costs 12-23 variance evaluations per iteration). E3 checks this approximation
  against ``variance="exact"`` on paired runs (identical random streams).
* Shots: S_res per residual estimate (one per iteration, all elements), S_E per energy estimate (the
  current state once per iteration, plus every nonzero candidate; a zero candidate reuses the current
  estimate and is free). One ``ShotLedger`` per run; "shots to reach and stay below 1.6 mHa" is
  ``ledger.cumulative()`` at the first iteration after which every exact error stays below 1.6 mHa
  until the end of the run.
* Seeds: every run draws from ``SeedSequence([--seed, crc32(key)])``; the key holds the experiment, the
  system and the seed index but not the shots, the optimizer or the candidate set (common random
  numbers: paired comparisons). Statistics over seeds: median and the 16-84% band (linear
  interpolation of the order statistics; failures count as +inf).

Experiments
-----------
E1  Floor versus residual shots: fixed-step SD from FCI (eps = 0.03 for H4, 0.02 otherwise; stable for
    eps < 4 / lambda_max) with the direct route. Correlated model at S in {1e4, ..., 1e9}, independent
    model at S in {1e4, 1e6}; ``--e1-burn`` steps of burn-in, then the exact E - E_FCI and the
    second-order mode energies y_m^2 (``diagnostics.linearized``) are averaged over ``--e1-window`` steps.
    Theory: ``LinearizedCQE.noise_floor(eps, noise_factor(model))`` (stationary), and the
    transient-aware expectation of the same window from FCI, floor_m (1 - <p_m^{2t}>_window),
    p_m = 1 - eps lambda_m / 2. Modes are split by stiffness: fast |p_m| < 0.98 (relaxed after the
    burn-in), slow (the rest), null (lambda <= 1e-10 lambda_max; N2 only; undriven at first order).
    Shots per iteration for floors of 1.6 mHa and 0.1 mHa from the stationary total, from the window
    expectation and from the simulated floors (log-log interpolation).
    Aux route: "exact" exp(-i delta H) at the delta that minimizes the bias-aware shots in workflow B's
    tables (0.6), S = 1e6. Its residual is sum_k d_k w_k sinc(delta w_k) M_k (sampling.py), so the
    SD map becomes y <- (I - (eps/2) K D) y + noise, D = diag(sinc(delta w)); the prediction solves that
    Lyapunov equation (stationary and the same window), and the plain noise_floor (D = 1) is shown for
    comparison. If the map is unstable (sinc < 0 for strongly coupled high excitations), delta = 0.3 is
    run as well. "df-trotter" and "df-fragments" at delta = 0.1, S = inf: the energy after
    ``--e1-bias-steps`` steps against ``LinearizedCQE.bias_offset`` (stationary) and its transient
    sum_m (y*_m (1 - p_m^n))^2.
E2  Optimizers with noisy residuals and exact energies (the oracle Krylov line search along the noisy
    direction): SD, CG (PR+), L-BFGS (memory 10) from HF, 60 iterations, S_res in {1e5, 1e6, 1e7, 1e8}
    and the exact run (S = inf). Final and best error, first iteration below 1.6 mHa, reach-and-stay
    iteration, CG restarts, L-BFGS resets and skipped pairs, uphill steps (exact slope >= 0, for which
    the oracle search returns a void step) and void steps (eps < 1e-6).
E3  Hardware-like runs: noisy residual (corr) plus noisy DF energies, fixed step candidates multiplying
    the direction, 40 iterations from HF, SD, CG, L-BFGS, at six (S_res, S_E) points: the cross through
    (1e6, 1e6) with S_res in {1e5, 1e7} and S_E in {1e5, 1e7}, and the corner (1e5, 1e7), where the
    residual noise dominates and the energies resolve uphill directions best. Candidate sets ("pos"
    means eps > 0):
      pos11    0.00625 * 2^k, k = 0..10 (11 energy estimates per iteration; workflow A's default);
      pos11+0  pos11 and 0 (11 estimates: the zero candidate is free);
      eq11     0, +-{0.0125, 0.05, 0.2, 0.8, 3.2} (10 estimates: the same cost as pos11, a factor-4 grid);
      dbl23    0, +-pos11 (22 estimates: twice the cost).
    For every iteration the exact energies and the estimates of all candidates are recorded (a wrapper
    of the energy estimator), which gives how often zero and negative candidates are chosen, how often
    the noisy direction is uphill for the exact energy, how often the exact-best candidate is negative,
    and the gain of each zero or negative choice: E_exact(chosen) - E_exact(the positive candidate the
    same estimates would have chosen), i.e. what that step gained over a positive-only set in the same
    state (negative = it helped).
E4  H4: real matchgate snapshots (``SnapshotResidual``) against the correlated and independent Gaussian
    models at the same shots: the fixed-step SD floor from FCI (eps = 0.03, 2000 snapshots per
    iteration) and convergence from HF with the oracle line search (SD and L-BFGS, 5000 snapshots).
E5  N2: the energy of the noisy runs split by mode class of the linearized CQE at FCI: the HF symmetry
    sector (the block of K that contains the HF determinant; K is block diagonal by symmetry) and the
    other coupled states, each split into null (lambda <= 1e-10 lambda_max), soft (< 1e-3 lambda_max)
    and stiff modes, plus "outside" (E - E_FCI minus the sum of y_m^2: the energy outside the coupled
    singlets, zero up to round-off, because E - E_0 = sum_k w_k d_k^2 exactly). Runs: E2's N2
    runs and long (150-iteration) SD and L-BFGS runs with the oracle line search at S_res = 1e6, 1e7.
Summary: per system, the cheapest E3 configuration (optimizer, candidate set, S_res, S_E) whose median
    shots to reach and stay below 1.6 mHa is lowest among the configurations in which every seed got
    there, with the exact-CQE iteration counts of the same optimizers (exact line search, and the pos11
    candidates without noise).

Reductions (full run within about 90 minutes on one thread): 5 seeds; E1 burn-in 150 + window 400
steps (aux "exact" 150 + 250, bias runs 200 steps); E1 independent model only at 1e4 and 1e6; E3 on six
grid points and 40 iterations (with noisy directions every iteration builds a Krylov space for the
largest candidate, 6.4: observed 40 / 52-56 / 23 ms per SD / CG / L-BFGS iteration for H6 at 1.0 A, so
the 3 x 3 grid with 60 iterations would take about 85 minutes alone); E4 burn-in 100 + window 200 steps;
the residual variances and covariances are frozen at FCI (along the exact SD trajectory the matchgate
totals change by <= 0.4%, sampling.py). ``--quick`` runs h4 and n2 with 2 seeds, short runs and small
grids (< 3 min) and only checks the plumbing. ``--e1-shots``, ``--e2-shots``, ``--e3-grid``,
``--e3-iter`` and similar options override the preset.

Threads: ``--threads n`` pins PySCF's OpenMP threads and, through OMP_NUM_THREADS / OPENBLAS_NUM_THREADS /
MKL_NUM_THREADS, numpy's BLAS threads; numpy is imported only after the arguments are parsed (as in
benchmarks/run_optimizers.py), so the pin takes effect when this file runs as a script.

Outputs: <out>/sampled_<system>.json, <out>/sampled_summary.md and <out>/sampled_summary.json; cached
variances and covariances in <out>/cache (``--no-cache``). ``--summary-only`` rebuilds the markdown from
the JSON files of ``--systems`` (its runtime total is the sum of the per-system run times). Every random
stream is keyed by the experiment, the system and the seed index, so a system run on its own
(``--systems lih``) gives exactly the runs of the same system inside a full run, and the systems can be
run in separate invocations into one ``--out`` and summarized together. ``--e3-sets`` restricts E3 to
some candidate sets (e.g. a finer (S_res, S_E) grid for pos11 and pos11+0 with ``--e3-grid``); a cell of
such a run equals the same cell of the full grid.

Usage:
    python benchmarks/run_sampled_cqe.py --threads 1 --out results/
    python benchmarks/run_sampled_cqe.py --quick --threads 1 --out /tmp/quick
    python benchmarks/run_sampled_cqe.py --experiments e1,e2 --systems h4,n2 --seeds 5 --threads 1
    python benchmarks/run_sampled_cqe.py --summary-only --out results/
    python benchmarks/run_sampled_cqe.py --threads 1 --experiments e3 --e3-sets pos11,pos11+0 \\
        --e3-grid 3e5:1e4,3e5:3e4,1e6:1e4,1e6:3e4 --out results/refine
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
import zlib

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
EXPERIMENTS = ("e1", "e2", "e3", "e4", "e5")
DEFAULT_SYSTEMS = "h4,h6a,lih,n2"
TARGET = 1.6e-3                 # Ha, chemical accuracy
FINE = 1e-4                     # Ha, the second floor target of E1
POS11 = tuple(0.00625 * 2 ** k for k in range(11))
MAG5 = (0.0125, 0.05, 0.2, 0.8, 3.2)
CANDIDATE_SETS = {
    "pos11": POS11,
    "pos11+0": (0.0,) + POS11,
    "eq11": (0.0,) + MAG5 + tuple(-x for x in MAG5),
    "dbl23": (0.0,) + POS11 + tuple(-x for x in POS11),
}
OPTIMIZERS = ("sd", "cg", "lbfgs")
NULL_RTOL = 1e-10               # null modes: lambda <= NULL_RTOL * lambda_max (LinearizedCQE.null_modes)
SOFT_RTOL = 1e-3                # soft modes: lambda < SOFT_RTOL * lambda_max (docs/m2-results.md)
FAST_P = 0.98                   # fast modes: |1 - eps lambda / 2| < FAST_P

SYSTEMS = {
    "h4": {"builder": ("hydrogen_chain", (4, 1.8, "bohr")), "eps": 0.03, "df_first": "cholesky"},
    "h6a": {"builder": ("hydrogen_chain", (6, 1.0, "angstrom")), "eps": 0.02, "df_first": "cholesky"},
    "h6": {"builder": ("hydrogen_chain", (6, 1.8, "bohr")), "eps": 0.02, "df_first": "cholesky"},
    "lih": {"builder": ("lih", ()), "eps": 0.02, "df_first": "cholesky"},
    "n2": {"builder": ("n2", (1.098,)), "eps": 0.02, "df_first": "eigh"},
}

FULL = {
    "seeds": 5,
    "cov_snapshots_small": 200000, "cov_snapshots": 20000, "rb_settings": 20000,
    "e1_shots": (1e4, 1e5, 1e6, 1e7, 1e8, 1e9), "e1_indep_shots": (1e4, 1e6),
    "e1_burn": 150, "e1_window": 400,
    "e1_aux_delta": 0.6, "e1_aux_fallback": 0.3, "e1_aux_shots": 1e6, "e1_aux_burn": 150, "e1_aux_window": 250,
    "e1_bias_delta": 0.1, "e1_bias_steps": 200,
    "e2_shots": (1e5, 1e6, 1e7, 1e8), "e2_iter": 60,
    "e3_grid": ((1e5, 1e6), (1e5, 1e7), (1e6, 1e5), (1e6, 1e6), (1e6, 1e7), (1e7, 1e6)), "e3_iter": 40,
    "e3_check_seeds": 3,
    "e4_floor_shots": 2000, "e4_eps": 0.03, "e4_burn": 100, "e4_window": 200,
    "e4_conv_shots": 5000, "e4_conv_iter": 15,
    "e5_shots": (1e6, 1e7), "e5_iter": 150,
}
QUICK = {
    "seeds": 2,
    "cov_snapshots_small": 5000, "cov_snapshots": 2000, "rb_settings": 2000,
    "e1_shots": (1e4, 1e6), "e1_indep_shots": (1e6,),
    "e1_burn": 30, "e1_window": 60,
    "e1_aux_delta": 0.6, "e1_aux_fallback": 0.3, "e1_aux_shots": 1e6, "e1_aux_burn": 30, "e1_aux_window": 40,
    "e1_bias_delta": 0.1, "e1_bias_steps": 40,
    "e2_shots": (1e6,), "e2_iter": 15,
    "e3_grid": ((1e6, 1e6),), "e3_iter": 15, "e3_check_seeds": 1,
    "e4_floor_shots": 500, "e4_eps": 0.03, "e4_burn": 20, "e4_window": 40,
    "e4_conv_shots": 1000, "e4_conv_iter": 6,
    "e5_shots": (1e6,), "e5_iter": 30,
}
QUICK_SYSTEMS = "h4,n2"

# numpy, scipy.linalg and the package modules are bound by _import_numerics(), after --threads pinned BLAS.
np = None
sla = None
L = None   # namespace of package objects


# ====================================================================== setup helpers
def pin_threads(n: int | None) -> dict:
    """Pin PySCF's OpenMP threads and numpy's BLAS threads (the BLAS pin needs numpy not yet loaded)."""
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


def _import_numerics() -> None:
    global np, sla, L
    import types

    import numpy
    import scipy.linalg

    from thc_engine_cqe import sampling, systems
    from thc_engine_cqe.cqe import run_cqe
    from thc_engine_cqe.diagnostics.linearized import linearize
    from thc_engine_cqe.factor import double_factorize
    from thc_engine_cqe.fci_ops import FCISpace, Hamiltonian
    from thc_engine_cqe.measure import schemes
    from thc_engine_cqe.residual import compute_residual

    np = numpy
    sla = scipy.linalg
    L = types.SimpleNamespace(sampling=sampling, systems=systems, run_cqe=run_cqe, linearize=linearize,
                              double_factorize=double_factorize, FCISpace=FCISpace, Hamiltonian=Hamiltonian,
                              schemes=schemes, compute_residual=compute_residual)


def build_system(name: str):
    if name not in SYSTEMS:
        raise SystemExit(f"unknown system {name!r}; choose from {sorted(SYSTEMS)}")
    fn, a = SYSTEMS[name]["builder"]
    return getattr(L.systems, fn)(*a)


def _names(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


def seed_seq(base: int, *key):
    """SeedSequence([base, crc32(key)]): a reproducible stream per run key."""
    return np.random.SeedSequence([int(base), zlib.crc32("/".join(str(k) for k in key).encode())])


def quantile(v, q: float) -> float:
    """Linear-interpolation quantile that allows +inf entries (inf if it interpolates towards one)."""
    x = np.sort(np.asarray(v, dtype=float))
    if x.size == 0:
        return math.nan
    h = (x.size - 1) * q
    lo, hi = int(math.floor(h)), int(math.ceil(h))
    if lo == hi or x[lo] == x[hi]:
        return float(x[lo])
    if not math.isfinite(x[hi]):
        return math.inf
    return float(x[lo] + (h - lo) * (x[hi] - x[lo]))


def band(values) -> dict:
    """Median and the 16-84% band over seeds (None -> +inf), with the number of finite values."""
    v = [math.inf if x is None else float(x) for x in values]
    return {"median": quantile(v, 0.5), "q16": quantile(v, 0.16), "q84": quantile(v, 0.84),
            "n": len(v), "n_ok": int(sum(math.isfinite(x) for x in v))}


def reach_index(err, thr: float):
    """First index j into the energies with error < thr (None if never)."""
    hit = np.flatnonzero(np.asarray(err) < thr)
    return int(hit[0]) if hit.size else None


def stay_index(err, thr: float):
    """Smallest j such that every error from index j to the end is < thr (None if the last is not)."""
    err = np.asarray(err)
    j = err.size
    while j > 0 and err[j - 1] < thr:
        j -= 1
    return None if j == err.size else int(j)


def shots_at(cum, j):
    """Shots spent to reach energies[j] (cum = ShotLedger.cumulative())."""
    if j is None:
        return None
    return 0.0 if j == 0 else float(cum[j - 1])


def _cache_path(args, name: str) -> str | None:
    if args.no_cache:
        return None
    d = os.path.join(args.out, "cache")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def cached(args, name: str, fn):
    """np.save/np.load cache of an array-valued computation (returns (array, from_cache))."""
    path = _cache_path(args, name)
    if path is not None and os.path.exists(path):
        return np.load(path), True
    arr = np.asarray(fn())
    if path is not None:
        np.save(path, arr)
    return arr, False


def clone_residual(template, shots: float, rng, ledger):
    """A ResidualNoise that shares the template's variances (and C^{1/2}), with its own shots, Generator, ledger.

    The template's K x K square root of the covariance (an eigendecomposition per construction) is
    computed once per system; the clone draws exactly what a fresh ResidualNoise with the same
    arguments would draw (checked at setup, ``clone_check``).
    """
    s = float(shots)
    if not s > 0:
        raise ValueError("shots must be positive")
    m = copy.copy(template)
    m.shots = s
    m.rng = rng
    seq = getattr(rng.bit_generator, "seed_seq", None)
    m.seed = None if seq is None else {"entropy": seq.entropy, "spawn_key": tuple(seq.spawn_key)}
    m.ledger = ledger
    return m


# ====================================================================== energy estimators
def make_iteration_energy_noise():
    """The EnergyNoise subclass with one variance evaluation per CQE iteration (module docstring)."""

    class IterationEnergyNoise(L.sampling.EnergyNoise):
        """DF energy noise whose per-shot variance V is evaluated at the current state of each iteration
        and reused for that iteration's candidates. One standard normal per call, as EnergyNoise."""

        def __init__(self, active, **kw):
            super().__init__(active, scheme="df", variance="exact", **kw)
            self.current_variance = None
            self.n_variance_evals = 0

        def __call__(self, e_exact, ci, context):
            if context not in ("current", "candidate"):
                raise ValueError(f"unknown context {context!r}")
            self.ledger.charge_energy(self.shots, context)
            if math.isinf(self.shots):
                return float(e_exact)
            if context == "current" or self.current_variance is None:
                self.current_variance = self.variance_of(ci)
                self.n_variance_evals += 1
            return float(e_exact) + math.sqrt(self.current_variance / self.shots) * float(self.rng.standard_normal())

    return IterationEnergyNoise


class RecordingEnergy:
    """Wraps an energy estimator; records (exact, estimate) of the current state and of every candidate."""

    def __init__(self, model):
        self.model = model
        self.records: list[dict] = []

    def __call__(self, e_exact, ci, context):
        est = float(self.model(e_exact, ci, context))
        if context == "current":
            self.records.append({"current": (float(e_exact), est), "cand": []})
        else:
            self.records[-1]["cand"].append((float(e_exact), est))
        return est


class LinearTwin:
    """Residual-estimator wrapper that drives the linearized CQE with the same residual errors (E1, E4).

    Each call returns the wrapped model's A_est unchanged and advances the linear mode amplitudes
    u <- p u - (eps/2) U^T W^{1/2} M (A_est - A) (fixed-step SD at the linearization point, c_0 = 1),
    so after step t, u holds the linear theory's y of the same noise path. Comparing the CQE's mode
    energies with u^2 separates nonlinear effects from the statistical scatter of a finite window.
    Valid for unbiased routes (direct; snapshots).
    """

    def __init__(self, model, gm, p, eps: float):
        self.model = model
        self.gm = gm
        self.p = p
        self.eps = float(eps)
        self.u = np.zeros(gm.shape[0])

    def __call__(self, rd, ci, it):
        a = self.model(rd, ci, it)
        err = np.real(np.asarray(a) - np.asarray(rd.A)).ravel()
        self.u = self.p * self.u - 0.5 * self.eps * (self.gm @ err)
        return a


# ====================================================================== mode bookkeeping
class ModeBasis:
    """Orthonormal mode basis of the linearized CQE at FCI with named classes of modes.

    ``energies(ci, e)`` returns y_m^2 summed per class, the sum over all modes and "outside" = (e - e0) -
    that sum. With d_k the exact overlaps, sum_m y_m^2 = sum_k w_k d_k^2 is the exact energy of the coupled
    states (E - E_0 = sum_k w_k d_k^2 for a normalized state), so "outside" is the energy in the other
    states (uncoupled singlets, other spin sectors): zero up to round-off for spin-free generators.
    """

    def __init__(self, lin, e0: float, modes, classes: dict):
        self.states = lin.states
        self.psi0 = lin.psi0.ravel()
        self.wh = np.sqrt(lin.gaps)
        self.modes = modes
        self.e0 = float(e0)
        self.classes = {k: np.asarray(v, bool) for k, v in classes.items()}

    def y2(self, ci):
        v = np.asarray(ci).ravel().real
        v = v / np.linalg.norm(v)
        d = self.states @ v
        if float(self.psi0 @ v) < 0:
            d = -d
        return (self.modes.T @ (self.wh * d)) ** 2

    def energies(self, ci, e: float) -> dict:
        y2 = self.y2(ci)
        out = {k: float(y2[m].sum()) for k, m in self.classes.items()}
        out["modes"] = float(y2.sum())
        out["total"] = float(e) - self.e0
        out["outside"] = out["total"] - out["modes"]
        return out


class WindowMean:
    """Callback: averages ModeBasis.energies over the steps with iteration >= start."""

    def __init__(self, basis: ModeBasis, start: int, twin: LinearTwin | None = None):
        self.basis = basis
        self.start = int(start)
        self.twin = twin
        self.sums: dict = {}
        self.count = 0
        self.last = None

    def __call__(self, step, ci):
        if step.iteration < self.start:
            return
        e = self.basis.energies(ci, step.energy_after)
        if self.twin is not None:
            u2 = self.twin.u ** 2
            for k, m in self.basis.classes.items():
                e[f"twin_{k}"] = float(u2[m].sum())
            e["twin_total"] = float(u2.sum())
        for k, v in e.items():
            self.sums[k] = self.sums.get(k, 0.0) + v
        self.count += 1
        self.last = e

    def means(self) -> dict:
        return {k: v / max(self.count, 1) for k, v in self.sums.items()}


class Trajectory:
    """Callback: ModeBasis.energies after every step (E5)."""

    def __init__(self, basis: ModeBasis, ci0=None, e0_state=None):
        self.basis = basis
        self.rows = [] if ci0 is None else [basis.energies(ci0, e0_state)]

    def __call__(self, step, ci):
        self.rows.append(self.basis.energies(ci, step.energy_after))


def window_factor(p, start: int, length: int):
    """1 - mean_{t=start+1..start+length} p^{2t}: the transient factor of a window from y_0 = 0."""
    p2 = np.asarray(p, dtype=float) ** 2
    out = np.ones_like(p2)
    pos = p2 > 0
    lg = np.log(p2[pos])                            # <= 0 for stable modes
    num = np.exp((start + 1) * lg) * np.expm1(length * lg)
    den = length * np.expm1(lg)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(den != 0, num / den, 1.0)
    out[pos] = 1.0 - mean
    return out


def lyapunov_window(P, Q, start: int, length: int):
    """X_t = P X_{t-1} P^T + Q from X_0 = 0: (mean of diag X_t over the window, diag of the stationary X or None)."""
    n = P.shape[0]
    X = np.zeros((n, n))
    acc = np.zeros(n)
    for t in range(1, start + length + 1):
        X = P @ X @ P.T + Q
        if t > start:
            acc += np.diag(X)
    rho = float(np.max(np.abs(np.linalg.eigvals(P))))
    stat = None
    if rho < 1.0 - 1e-13:
        stat = np.diag(sla.solve_discrete_lyapunov(P, Q)).copy()
    return acc / length, stat, rho


# ====================================================================== system setup
class Ctx:
    """Everything a system's experiments share (built by setup_system)."""


def setup_system(name: str, args) -> Ctx:
    t0 = time.perf_counter()
    S = L.sampling
    c = Ctx()
    c.name = name
    c.act = act = build_system(name)
    c.eps = SYSTEMS[name]["eps"]
    e_fci, ci_fci = act.fci()
    c.e_fci, c.ci_fci = float(e_fci), np.asarray(ci_fci)
    space = L.FCISpace.of(act)
    c.hf = space.hf()
    c.e_hf = float(L.Hamiltonian(act).energy(c.hf))
    c.lin = lin = L.linearize(act)
    lam = lin.stiffness
    lmax = float(lam.max())
    c.null = lam <= NULL_RTOL * lmax
    c.soft = (lam < SOFT_RTOL * lmax) & ~c.null
    info = {"system": name, "label": act.label, "norb": act.norb, "nelec": list(act.nelec),
            "n_det": int(space.size), "e_fci": c.e_fci, "e_hf": c.e_hf, "e0_lin_minus_fci": lin.e0 - c.e_fci,
            "n_coupled": int(lin.n_states), "n_null": int(c.null.sum()), "n_soft": int(c.soft.sum()),
            "lambda_min_nonnull": float(lam[~c.null].min()), "lambda_max": lmax, "eps": c.eps,
            "eps_stability": 4.0 / lmax}
    # residual variances (direct route, matchgate shadows, at FCI), both element kinds
    big = act.norb >= 6
    ts = time.perf_counter()
    c.var = {}
    for kind in ("spin_orbital", "spin_free"):
        opts = {"seed": zlib.crc32(f"var/{name}/{kind}".encode())}
        if big:
            opts["n_settings"] = args.rb_settings
        tag = f"var_{name}_{kind}_{args.rb_settings if big else 'exact'}.npy"
        v, hit = cached(args, tag, lambda: S.element_variances(act, c.ci_fci, route="direct",
                                                                 scheme="matchgate_shadow", kind=kind,
                                                                 **opts).per_target)
        c.var[kind] = v
        info[f"var_{kind}_total"] = float(v.sum())
        info[f"var_{kind}_cached"] = hit
    info["seconds_variances"] = time.perf_counter() - ts
    # element covariance of real snapshots at FCI (spin-orbital)
    ts = time.perf_counter()
    n_snap = args.cov_snapshots_small if act.norb <= 4 else args.cov_snapshots
    K = c.var["spin_orbital"].size

    def _cov():
        sr = S.SnapshotResidual(act, kind="spin_orbital", shots=1,
                                rng=np.random.default_rng(seed_seq(args.seed, "cov", name)))
        return sr.element_covariance(c.ci_fci, n_snap, mean=np.zeros(K))[1]

    c.cov, hit = cached(args, f"cov_{name}_{n_snap}_{args.seed}.npy", _cov)
    info.update(cov_snapshots=n_snap, cov_cached=hit, cov_trace=float(np.trace(c.cov)),
                seconds_covariance=time.perf_counter() - ts)
    # templates (shots = 1) and their noise factors
    c.tmpl = {
        "corr": S.ResidualNoise(act, kind="spin_orbital", shots=1.0, per_element_covariance=c.cov, rng=0),
        "indep": S.ResidualNoise(act, kind="spin_orbital", shots=1.0, per_element_variance=c.var["spin_orbital"],
                                 rng=0),
        "indep_sf": S.ResidualNoise(act, kind="spin_free", shots=1.0, per_element_variance=c.var["spin_free"],
                                    rng=0),
    }
    c.F1 = {k: S.noise_factor(m) for k, m in c.tmpl.items()}
    # the clone reproduces a fresh model bit for bit
    rd = L.compute_residual(act, c.hf)
    a = clone_residual(c.tmpl["corr"], 1e6, np.random.default_rng(7), S.ShotLedger())(rd, c.hf, 0)
    b = S.ResidualNoise(act, kind="spin_orbital", shots=1e6, per_element_covariance=c.cov,
                        rng=np.random.default_rng(7))(rd, c.hf, 0)
    info["clone_check_max_abs_diff"] = float(np.abs(a - b).max())
    # DF energy factorization and variances
    c.df_first = SYSTEMS[name]["df_first"]
    c.df = {f: L.double_factorize(np.asarray(act.eri), first=f) for f in ("cholesky", "eigh")}
    for f, fac in c.df.items():
        vf = float(L.schemes.df_variance(act, c.ci_fci, factorization=fac, shift=True).total)
        vh = float(L.schemes.df_variance(act, c.hf, factorization=fac, shift=True).total)
        info[f"df_{f}_shift_fci"] = vf
        info[f"df_{f}_shift_hf"] = vh
    info["df_first"] = c.df_first
    info["n_leaves"] = int(len(c.df[c.df_first].eigenvalues))
    # mode classes for E1 (global modes) and the HF symmetry sector (E5)
    p = 1.0 - 0.5 * c.eps * lam
    c.p = p
    c.fast = (np.abs(p) < FAST_P) & ~c.null
    c.slow = ~c.fast & ~c.null
    info.update(n_fast=int(c.fast.sum()), n_slow=int(c.slow.sum()))
    c.basis_e1 = ModeBasis(lin, lin.e0, lin.modes, {"fast": c.fast, "slow": c.slow, "null": c.null})
    c.gm = lin.modes.T @ (np.sqrt(lin.gaps)[:, None] * lin.coupling)      # (modes, N^4) for LinearTwin
    c.basis_sector, sector_info = sector_basis(lin, c.hf)
    info["sector"] = sector_info
    info["seconds_setup"] = time.perf_counter() - t0
    c.info = info
    return c


def sector_basis(lin, hf):
    """Mode basis split by the HF symmetry sector: the K block (coupling-graph component) holding the HF weight."""
    from scipy.sparse.csgraph import connected_components

    S_ = lin.coupling @ lin.coupling.T
    adj = np.abs(S_) > 1e-10 * np.abs(S_).max()
    ncomp, lab = connected_components(adj, directed=False)
    w = (lin.states @ (hf.ravel() / np.linalg.norm(hf))) ** 2
    comp_w = np.array([w[lab == k].sum() for k in range(ncomp)])
    in_comp = np.flatnonzero(comp_w > 1e-12 * comp_w.max())
    ins = np.isin(lab, in_comp)
    K = lin.kmatrix()
    lmax = float(lin.stiffness.max())
    n = lin.n_states
    U = np.zeros((n, n))
    lam = np.zeros(n)
    sector = np.zeros(n, bool)
    col = 0
    for mask, flag in ((ins, True), (~ins, False)):
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        lk, uk = np.linalg.eigh(K[np.ix_(idx, idx)])
        U[np.ix_(idx, np.arange(col, col + idx.size))] = uk
        lam[col:col + idx.size] = np.clip(lk, 0.0, None)
        sector[col:col + idx.size] = flag
        col += idx.size
    null = lam <= NULL_RTOL * lmax
    soft = (lam < SOFT_RTOL * lmax) & ~null
    stiff = ~null & ~soft
    classes = {"in_null": sector & null, "in_soft": sector & soft, "in_stiff": sector & stiff,
               "out_null": ~sector & null, "out_soft": ~sector & soft, "out_stiff": ~sector & stiff}
    cross = float(np.abs(K[np.ix_(ins, ~ins)]).max()) if (~ins).any() else 0.0
    info = {"n_components": int(ncomp), "component_sizes": [int(x) for x in np.bincount(lab)],
            "hf_weight_per_component": [float(x) for x in comp_w], "n_in": int(ins.sum()), "n_out": int((~ins).sum()),
            "cross_block_max_abs_K": cross, "K_max": float(np.abs(K).max()),
            **{f"n_{k}": int(v.sum()) for k, v in classes.items()}}
    return ModeBasis(lin, lin.e0, U, classes), info


# ====================================================================== E1
def e1_theory(c, F1, start: int, length: int) -> dict:
    """Stationary noise_floor and window expectation at S = 1 (scale by 1/S), per class."""
    pr = c.lin.noise_floor(c.eps, F1)
    per = pr["per_mode"]
    win = per * window_factor(c.p, start, length)
    out = {"stable": bool(pr["stable"]), "n_undriven": int(pr["undriven"].sum())}
    for label, arr in (("stationary", per), ("window", win)):
        out[label] = {"total": float(arr.sum()), "fast": float(arr[c.fast].sum()),
                      "slow": float(arr[c.slow].sum()), "null": float(arr[c.null].sum()),
                      "soft": float(arr[c.soft].sum())}
    return out


def e1_direct(c, args, model: str, shots_list, start: int, length: int) -> dict:
    th = e1_theory(c, c.F1[model], start, length)
    rows = []
    for S_ in shots_list:
        t = time.perf_counter()
        per_seed = []
        for i in range(args.seeds):
            rng = np.random.default_rng(seed_seq(args.seed, "e1", c.name, model, i))
            m = clone_residual(c.tmpl[model], S_, rng, L.sampling.ShotLedger())
            twin = LinearTwin(m, c.gm, c.p, c.eps)
            acc = WindowMean(c.basis_e1, start, twin)
            L.run_cqe(c.act, ci0=c.lin.psi0, max_iter=start + length, eps_candidates=(c.eps,), r_tol=0.0,
                      e_tol=0.0, residual_estimator=twin, callback=acc)
            per_seed.append(acc.means())
        row = {"shots": S_, "per_seed": per_seed, "seconds": time.perf_counter() - t}
        for k in ("total", "fast", "slow", "null", "outside", "twin_total", "twin_fast", "twin_slow"):
            row[k] = band([s[k] for s in per_seed])
        for k in ("total", "fast", "slow"):
            tw = "twin_" + ("total" if k == "total" else k)
            row[f"twin_ratio_{k}"] = band([s[k] / s[tw] if s[tw] > 0 else math.nan for s in per_seed])
        for k in ("total", "fast", "slow"):
            exp_ = th["window"][k] / S_
            row[f"expected_{k}"] = exp_
            row[f"ratio_{k}"] = band([s[k] / exp_ for s in per_seed]) if exp_ > 0 else None
        rows.append(row)
        print(f"[{c.name}] E1 {model:5s} S={S_:.0e}: sim {row['total']['median']:.3e} / expected "
              f"{row['expected_total']:.3e} (fast ratio {row['ratio_fast']['median']:.3f}) "
              f"[{row['seconds']:.1f} s]", flush=True)
    return {"model": model, "theory_S1": th, "rows": rows, "burn": start, "window": length}


def floor_shots(c, res: dict) -> dict:
    """Shots per iteration for floors of TARGET and FINE: theory (stationary, window) and simulated."""
    th = res["theory_S1"]
    out = {}
    for tgt in (TARGET, FINE):
        d = {"stationary": th["stationary"]["total"] / tgt, "stationary_fast": th["stationary"]["fast"] / tgt,
             "window": th["window"]["total"] / tgt, "simulated": None}
        rows = sorted(res.get("rows", []), key=lambda r: r["shots"])
        for a, b in zip(rows, rows[1:]):
            fa, fb = a["total"]["median"], b["total"]["median"]
            if fa >= tgt > fb > 0:
                la, lb = math.log(a["shots"]), math.log(b["shots"])
                d["simulated"] = math.exp(la + (math.log(tgt) - math.log(fa)) * (lb - la)
                                          / (math.log(fb) - math.log(fa)))
                break
        out[f"{tgt:g}"] = d
    return out


def e1_aux_exact(c, args, delta: float) -> dict:
    """Exact aux route at delta: simulated floor vs the sinc-damped Lyapunov prediction."""
    S = L.sampling
    act, lin = c.act, c.lin
    t = time.perf_counter()
    v = S.element_variances(act, c.ci_fci, route="aux", scheme="matchgate_shadow", kind="spin_orbital",
                            delta=delta, aux_method="exact",
                            seed=zlib.crc32(f"auxvar/{c.name}/{delta}".encode())).per_target
    m1 = S.ResidualNoise(act, route="aux", kind="spin_orbital", shots=1.0, delta=delta, aux_method="exact",
                         per_element_variance=v, rng=0)
    F1 = S.noise_factor(m1)
    s = np.sinc(delta * lin.gaps / np.pi)
    Kmat = lin.kmatrix()
    U = lin.modes
    n = lin.n_states
    eps = c.eps
    R = np.sqrt(lin.gaps)[:, None] * (lin.coupling @ F1)
    Q = (0.5 * eps) ** 2 * (R @ R.T)
    P = np.eye(n) - 0.5 * eps * Kmat * s[None, :]
    # null modes of K are neither driven nor coupled back (their rows of U^T K D U vanish): drop them
    live = np.ix_(~c.null, ~c.null)
    Pm, Qm = (U.T @ P @ U)[live], (U.T @ Q @ U)[live]
    start, length = args.e1_aux_burn, args.e1_aux_window
    win_l, stat_l, rho = lyapunov_window(Pm, Qm, start, length)
    win = np.zeros(n)
    win[~c.null] = win_l
    stat = None
    if stat_l is not None:
        stat = np.zeros(n)
        stat[~c.null] = stat_l
    # check: without the damping the same solver reproduces noise_floor
    P0 = np.eye(n) - 0.5 * eps * Kmat
    _, stat0, _ = lyapunov_window((U.T @ P0 @ U)[live], Qm, 0, 1)
    plain = lin.noise_floor(eps, F1)
    eig = np.linalg.eigvals(Kmat * s[None, :])
    out = {"method": "exact", "delta": delta, "shots": args.e1_aux_shots, "burn": start, "window": length,
           "var_total": float(v.sum()), "sinc_min": float(s.min()), "rho": rho,
           "min_eig_KD": float(eig.real.min()), "plain_stationary_S1": float(plain["total"]),
           "plain_check_rel": (float(stat0.sum() / plain["total"] - 1) if stat0 is not None else None),
           "damped_stationary_S1": None if stat is None else float(stat.sum()),
           "damped_window_S1": float(win.sum()) if rho < 1.0 + 1e-12 else None,
           "damped_window_fast_S1": float(win[c.fast].sum()) if rho < 1.0 + 1e-12 else None}
    S_ = args.e1_aux_shots
    per_seed = []
    for i in range(args.seeds):
        rng = np.random.default_rng(seed_seq(args.seed, "e1aux", c.name, delta, i))
        m = S.ResidualNoise(act, route="aux", kind="spin_orbital", shots=S_, delta=delta, aux_method="exact",
                            per_element_variance=v, rng=rng)
        acc = WindowMean(c.basis_e1, start)
        maxe = [0.0]

        def cb(step, ci, acc=acc, maxe=maxe):
            maxe[0] = max(maxe[0], step.energy_after - c.e_fci)
            acc(step, ci)

        L.run_cqe(act, ci0=lin.psi0, max_iter=start + length, eps_candidates=(eps,), r_tol=0.0, e_tol=0.0,
                  residual_estimator=m, callback=cb)
        d = acc.means()
        d["max_error"] = maxe[0]
        per_seed.append(d)
    out["per_seed"] = per_seed
    for k in ("total", "fast", "slow", "outside", "max_error"):
        out[k] = band([d[k] for d in per_seed])
    if out["damped_window_S1"] is not None:
        out["expected_total"] = out["damped_window_S1"] / S_
        out["ratio_total"] = band([d["total"] / out["expected_total"] for d in per_seed])
        out["expected_fast"] = out["damped_window_fast_S1"] / S_
        out["ratio_fast"] = band([d["fast"] / out["expected_fast"] for d in per_seed])
    out["seconds"] = time.perf_counter() - t
    print(f"[{c.name}] E1 aux exact delta={delta}: rho={rho:.4f}, sim {out['total']['median']:.3e}, expected "
          f"{out.get('expected_total', float('nan')):.3e}, plain {out['plain_stationary_S1'] / S_:.3e} "
          f"[{out['seconds']:.1f} s]", flush=True)
    return out


def e1_aux_bias(c, args, method: str) -> dict:
    """df-trotter / df-fragments at delta, S = inf: energy after n steps vs bias_offset and its transient."""
    S = L.sampling
    act, lin = c.act, c.lin
    t = time.perf_counter()
    delta, n = args.e1_bias_delta, args.e1_bias_steps
    K = c.var["spin_orbital"].size
    m = S.ResidualNoise(act, route="aux", kind="spin_orbital", shots=math.inf, delta=delta, aux_method=method,
                        per_element_variance=np.zeros(K), rng=0)
    b = m.bias_tensor(lin.psi0)
    pr = lin.bias_offset(b)
    ystar = lin.modes.T @ pr["y"]
    trans = (ystar * (1.0 - c.p ** n)) ** 2
    res = L.run_cqe(act, ci0=lin.psi0, max_iter=n, eps_candidates=(c.eps,), r_tol=0.0, e_tol=0.0,
                    residual_estimator=m)
    got = c.basis_e1.energies(res.ci, res.energies[-1])
    out = {"method": method, "delta": delta, "steps": n, "bias_norm": float(np.linalg.norm(b)),
           "offset_stationary": float(pr["energy"]),
           "offset_stationary_soft_share": float(pr["per_mode"][c.soft].sum() / max(pr["energy"], 1e-300)),
           "offset_stationary_fast": float(pr["per_mode"][c.fast].sum()),
           "transient_total": float(trans.sum()), "transient_fast": float(trans[c.fast].sum()),
           "transient_slow": float(trans[c.slow].sum()),
           "simulated": got, "ratio_total": got["total"] / max(float(trans.sum()), 1e-300),
           "ratio_fast": got["fast"] / max(float(trans[c.fast].sum()), 1e-300),
           "seconds": time.perf_counter() - t}
    print(f"[{c.name}] E1 bias {method} delta={delta}: sim {got['total']:.3e} vs transient {trans.sum():.3e} "
          f"(stationary {pr['energy']:.3e}) [{out['seconds']:.1f} s]", flush=True)
    return out


def run_e1(c, args) -> dict:
    t = time.perf_counter()
    out = {"eps": c.eps, "direct": {}, "aux_exact": [], "aux_bias": []}
    out["direct"]["corr"] = e1_direct(c, args, "corr", args.e1_shots, args.e1_burn, args.e1_window)
    out["direct"]["indep"] = e1_direct(c, args, "indep", args.e1_indep_shots, args.e1_burn, args.e1_window)
    out["direct"]["indep_sf"] = {"model": "indep_sf", "theory_S1": e1_theory(c, c.F1["indep_sf"], args.e1_burn,
                                                                               args.e1_window), "rows": []}
    for k, r in out["direct"].items():
        r["floor_shots"] = floor_shots(c, r)
    a = e1_aux_exact(c, args, args.e1_aux_delta)
    out["aux_exact"].append(a)
    if a["rho"] > 1.0 + 1e-12 and args.e1_aux_fallback:
        out["aux_exact"].append(e1_aux_exact(c, args, args.e1_aux_fallback))
    for method in ("df-trotter", "df-fragments"):
        out["aux_bias"].append(e1_aux_bias(c, args, method))
    out["seconds"] = time.perf_counter() - t
    return out


# ====================================================================== E2
def opt_counts(res) -> dict:
    cg = lb_reset = lb_skip = 0
    for s in res.steps:
        info = s.optimizer_info or {}
        if info.get("restart") not in (None, "first"):
            cg += 1
        if info.get("reset"):
            lb_reset += 1
        if info.get("pair") == "skipped":
            lb_skip += 1
    return {"cg_restarts": cg, "lbfgs_resets": lb_reset, "lbfgs_skips": lb_skip}


def run_metrics(res, cum=None) -> dict:
    err = np.asarray(res.errors)
    j_reach, j_stay = reach_index(err, TARGET), stay_index(err, TARGET)
    out = {"errors": [float(x) for x in err], "eps": [float(s.eps) for s in res.steps],
           "final": float(err[-1]), "best": float(err.min()), "reach": j_reach, "stay": j_stay,
           "reach_fine": reach_index(err, FINE),
           "uphill": int(sum(1 for s in res.steps if not s.slope < 0.0)),
           "void": int(sum(1 for s in res.steps if abs(s.eps) < 1e-6)), "n_steps": len(res.steps),
           **opt_counts(res)}
    if cum is not None:
        out["shots_reach"] = shots_at(cum, j_reach)
        out["shots_stay"] = shots_at(cum, j_stay)
        out["shots_total"] = float(cum[-1]) if len(cum) else 0.0
    return out


def summarize_runs(runs: list[dict]) -> dict:
    keys = ("final", "best", "reach", "stay", "shots_reach", "shots_stay", "uphill", "void", "cg_restarts",
            "lbfgs_resets", "lbfgs_skips")
    return {k: band([r.get(k) for r in runs]) for k in keys if k in runs[0]}


def run_e2(c, args, tracker_basis=None) -> dict:
    t0 = time.perf_counter()
    out = {"iterations": args.e2_iter, "rows": [], "trajectories": {}}
    for opt in OPTIMIZERS:
        for S_ in (math.inf,) + tuple(args.e2_shots):
            t = time.perf_counter()
            runs = []
            seeds = [0] if math.isinf(S_) else range(args.seeds)
            for i in seeds:
                led = L.sampling.ShotLedger()
                est = None
                if not math.isinf(S_):
                    est = clone_residual(c.tmpl["corr"], S_, np.random.default_rng(seed_seq(args.seed, "e2", c.name, i)),
                                         led)
                tr = None if tracker_basis is None else Trajectory(tracker_basis, c.hf, c.e_hf)
                res = L.run_cqe(c.act, max_iter=args.e2_iter, e_exact=c.e_fci, optimizer=opt, r_tol=0.0, e_tol=0.0,
                                residual_estimator=est, callback=tr)
                m = run_metrics(res)
                if not math.isinf(S_):
                    m["shots_reach"] = None if m["reach"] is None else S_ * m["reach"]
                    m["shots_stay"] = None if m["stay"] is None else S_ * m["stay"]
                m["seed"] = i
                runs.append(m)
                if tr is not None:
                    out["trajectories"].setdefault(f"{opt}/{S_:g}", []).append(tr.rows)
            row = {"optimizer": opt, "shots": S_, "runs": runs, "stats": summarize_runs(runs),
                   "seconds": time.perf_counter() - t}
            out["rows"].append(row)
            st = row["stats"]
            print(f"[{c.name}] E2 {opt:5s} S={S_:.0e}: final {st['final']['median']:.2e} best "
                  f"{st['best']['median']:.2e} stay {st['stay']['median']} ({st['stay']['n_ok']}/{st['stay']['n']}) "
                  f"[{row['seconds']:.1f} s]", flush=True)
    out["seconds"] = time.perf_counter() - t0
    return out


# ====================================================================== E3
def candidate_analysis(cands, rec: RecordingEnergy, res) -> dict:
    """Per-iteration choice statistics of a fixed-candidate run with a RecordingEnergy (module docstring)."""
    cands = np.asarray(cands, dtype=float)
    nz = cands != 0.0
    pos = cands > 0.0
    n = len(res.steps)
    chosen_sign = np.zeros(n, dtype=int)
    best_sign = np.zeros(n, dtype=int)
    uphill = np.zeros(n, dtype=bool)
    gain = np.zeros(n)
    regret = np.zeros(n)
    mismatch = 0
    for k, step in enumerate(res.steps):
        r = rec.records[k]
        ex = np.empty(cands.size)
        es = np.empty(cands.size)
        ex[~nz], es[~nz] = r["current"]
        ce = np.asarray(r["cand"], dtype=float).reshape(-1, 2)
        ex[nz], es[nz] = ce[:, 0], ce[:, 1]
        i = int(np.argmin(es))
        if cands[i] != step.eps:
            mismatch += 1
        j = int(np.argmin(ex))
        ip = np.flatnonzero(pos)[int(np.argmin(es[pos]))]
        chosen_sign[k] = int(np.sign(cands[i]))
        best_sign[k] = int(np.sign(cands[j]))
        uphill[k] = not step.slope < 0.0
        gain[k] = ex[i] - ex[ip]
        regret[k] = ex[i] - ex[j]
    half = slice(n // 2, None)

    def frac(mask, sl=slice(None)):
        m = np.asarray(mask)[sl]
        return float(m.mean()) if m.size else 0.0

    nonpos = chosen_sign <= 0
    out = {"chosen_sign": chosen_sign.tolist(), "uphill": uphill.astype(int).tolist(), "mismatch": mismatch,
           "best_sign": best_sign.tolist(), "gain": gain.tolist()}
    for label, sl in (("all", slice(None)), ("late", half)):
        g = gain[sl]
        npos = nonpos[sl]
        out[label] = {"zero": frac(chosen_sign == 0, sl), "neg": frac(chosen_sign < 0, sl),
                      "uphill": frac(uphill, sl), "best_neg": frac(best_sign < 0, sl),
                      "best_zero": frac(best_sign == 0, sl),
                      "uphill_and_nonpos": frac(uphill & nonpos, sl),
                      "n_nonpos": int(npos.sum()), "gain_sum": float(g[npos].sum()),
                      "n_helped": int((g[npos] < 0).sum()), "n_hurt": int((g[npos] > 0).sum()),
                      "regret_mean": float(regret[sl].mean()) if regret[sl].size else 0.0}
    return out


def e3_one(c, args, opt, set_name, S_res, S_E, i, energy_mode="iteration") -> dict:
    S = L.sampling
    cands = CANDIDATE_SETS[set_name]
    t = time.perf_counter()
    led = S.ShotLedger()
    if math.isinf(S_res) and math.isinf(S_E):
        res = L.run_cqe(c.act, max_iter=args.e3_iter, e_exact=c.e_fci, optimizer=opt, eps_candidates=cands,
                        r_tol=0.0, e_tol=0.0)
        m = run_metrics(res)
        m["seconds"] = time.perf_counter() - t
        return m
    ss = seed_seq(args.seed, "e3", c.name, i).spawn(2)
    est = clone_residual(c.tmpl["corr"], S_res, np.random.default_rng(ss[0]), led)
    fac = c.df[c.df_first]
    if energy_mode == "iteration":
        en = c.IterationEnergyNoise(c.act, shots=S_E, factorization=fac, shift=True,
                                    rng=np.random.default_rng(ss[1]), ledger=led)
    else:
        en = S.EnergyNoise(c.act, scheme="df", shots=S_E, variance="exact", factorization=fac, shift=True,
                           rng=np.random.default_rng(ss[1]), ledger=led)
    rec = RecordingEnergy(en)
    res = L.run_cqe(c.act, max_iter=args.e3_iter, e_exact=c.e_fci, optimizer=opt, eps_candidates=cands,
                    r_tol=0.0, e_tol=0.0, residual_estimator=est, energy_estimator=rec)
    m = run_metrics(res, led.cumulative())
    m["candidates"] = candidate_analysis(cands, rec, res)
    m["ledger"] = led.summary()
    m["seed"] = i
    m["seconds"] = time.perf_counter() - t
    return m


def run_e3(c, args) -> dict:
    t0 = time.perf_counter()
    c.IterationEnergyNoise = make_iteration_energy_noise()
    out = {"iterations": args.e3_iter, "exact": [], "cells": []}
    for opt in OPTIMIZERS:
        for set_name in args.e3_sets:
            m = e3_one(c, args, opt, set_name, math.inf, math.inf, 0)
            m.update(optimizer=opt, set=set_name)
            out["exact"].append(m)
    print(f"[{c.name}] E3 exact references: " + ", ".join(
        f"{m['optimizer']}/{m['set']}:{m['reach']}" for m in out["exact"]), flush=True)
    for S_res, S_E in args.e3_grid:
        t = time.perf_counter()
        for opt in OPTIMIZERS:
            for set_name in args.e3_sets:
                runs = [e3_one(c, args, opt, set_name, S_res, S_E, i) for i in range(args.seeds)]
                cell = {"optimizer": opt, "set": set_name, "S_res": S_res, "S_E": S_E, "runs": runs,
                        "stats": summarize_runs(runs)}
                for label in ("all", "late"):
                    cell[f"usage_{label}"] = {
                        k: float(np.mean([r["candidates"][label][k] for r in runs]))
                        for k in ("zero", "neg", "uphill", "best_neg", "best_zero", "uphill_and_nonpos",
                                  "regret_mean")}
                    cell[f"usage_{label}"].update({
                        k: int(sum(r["candidates"][label][k] for r in runs))
                        for k in ("n_nonpos", "n_helped", "n_hurt")})
                    cell[f"usage_{label}"]["gain_sum"] = float(sum(r["candidates"][label]["gain_sum"]
                                                                   for r in runs))
                out["cells"].append(cell)
        print(f"[{c.name}] E3 S_res={S_res:.0e} S_E={S_E:.0e} done [{time.perf_counter() - t:.1f} s]: " + "; ".join(
            f"{x['optimizer']}/{x['set']} stay {_short(x['stats']['shots_stay']['median'])} "
            f"({x['stats']['shots_stay']['n_ok']}) fin {x['stats']['final']['median']:.1e}"
            for x in out["cells"] if x["S_res"] == S_res and x["S_E"] == S_E), flush=True)
    # check of the per-iteration energy variance against EnergyNoise(variance="exact")
    t = time.perf_counter()
    chk = []
    S_res, S_E = args.e3_check_shots
    for i in range(args.e3_check_seeds):
        a = e3_one(c, args, "lbfgs", "pos11", S_res, S_E, i, "iteration")
        b = e3_one(c, args, "lbfgs", "pos11", S_res, S_E, i, "exact")
        chk.append({"seed": i, "final_iteration": a["final"], "final_exact": b["final"],
                    "stay_iteration": a["stay"], "stay_exact": b["stay"],
                    "max_abs_error_diff": float(np.max(np.abs(np.subtract(a["errors"], b["errors"])))),
                    "seconds_iteration": a["seconds"], "seconds_exact": b["seconds"]})
    out["variance_check"] = {"optimizer": "lbfgs", "set": "pos11", "S_res": S_res, "S_E": S_E, "runs": chk,
                             "seconds": time.perf_counter() - t}
    out["seconds"] = time.perf_counter() - t0
    return out


def _short(x) -> str:
    return "–" if x is None or not math.isfinite(x) else f"{x:.2e}"


# ====================================================================== E4
def run_e4(c, args) -> dict:
    S = L.sampling
    t0 = time.perf_counter()
    act, lin = c.act, c.lin
    out = {"floor": {}, "convergence": {}}
    eps = args.e4_eps
    p = 1.0 - 0.5 * eps * lin.stiffness
    fast = (np.abs(p) < FAST_P) & ~c.null
    basis = ModeBasis(lin, lin.e0, lin.modes, {"fast": fast, "slow": ~fast & ~c.null, "null": c.null})
    start, length = args.e4_burn, args.e4_window
    S_ = args.e4_floor_shots
    theory = {}
    for model in ("corr", "indep"):
        pr = lin.noise_floor(eps, c.F1[model])
        win = pr["per_mode"] * window_factor(p, start, length)
        theory[model] = {"stationary": float(pr["per_mode"].sum()) / S_, "window": float(win.sum()) / S_,
                         "window_fast": float(win[fast].sum()) / S_}
    out["floor"]["theory"] = theory
    out["floor"].update(shots=S_, eps=eps, burn=start, window=length)
    for model in ("snapshot", "corr", "indep"):
        t = time.perf_counter()
        per_seed = []
        for i in range(args.seeds):
            rng = np.random.default_rng(seed_seq(args.seed, "e4floor", model, i))
            if model == "snapshot":
                m = S.SnapshotResidual(act, kind="spin_orbital", shots=int(S_), rng=rng)
            else:
                m = clone_residual(c.tmpl[model], S_, rng, S.ShotLedger())
            twin = LinearTwin(m, c.gm, p, eps)
            acc = WindowMean(basis, start, twin)
            L.run_cqe(act, ci0=lin.psi0, max_iter=start + length, eps_candidates=(eps,), r_tol=0.0, e_tol=0.0,
                      residual_estimator=twin, callback=acc)
            per_seed.append(acc.means())
        row = {"per_seed": per_seed, "seconds": time.perf_counter() - t}
        for k in ("total", "fast", "slow", "outside", "twin_total", "twin_fast"):
            row[k] = band([d[k] for d in per_seed])
        row["twin_ratio_total"] = band([d["total"] / d["twin_total"] for d in per_seed])
        row["twin_ratio_fast"] = band([d["fast"] / d["twin_fast"] for d in per_seed])
        for th in ("corr", "indep"):
            row[f"ratio_{th}"] = band([d["total"] / theory[th]["window"] for d in per_seed])
            row[f"ratio_fast_{th}"] = band([d["fast"] / theory[th]["window_fast"] for d in per_seed])
        out["floor"][model] = row
        print(f"[{c.name}] E4 floor {model:8s}: {row['total']['median']:.3e} (corr theory "
              f"{theory['corr']['window']:.3e}, ratio {row['ratio_corr']['median']:.3f}) [{row['seconds']:.1f} s]",
              flush=True)
    S_ = args.e4_conv_shots
    for model in ("snapshot", "corr", "indep"):
        for opt in ("sd", "lbfgs"):
            t = time.perf_counter()
            runs = []
            for i in range(args.seeds):
                rng = np.random.default_rng(seed_seq(args.seed, "e4conv", model, i))
                led = S.ShotLedger()
                if model == "snapshot":
                    m = S.SnapshotResidual(act, kind="spin_orbital", shots=int(S_), rng=rng, ledger=led)
                else:
                    m = clone_residual(c.tmpl[model], S_, rng, led)
                res = L.run_cqe(act, max_iter=args.e4_conv_iter, e_exact=c.e_fci, optimizer=opt, r_tol=0.0,
                                e_tol=0.0, residual_estimator=m)
                r = run_metrics(res)
                r["seed"] = i
                runs.append(r)
            row = {"runs": runs, "stats": summarize_runs(runs), "seconds": time.perf_counter() - t}
            out["convergence"][f"{model}/{opt}"] = row
            st = row["stats"]
            print(f"[{c.name}] E4 conv {model:8s} {opt:5s}: final {st['final']['median']:.2e} best "
                  f"{st['best']['median']:.2e} [{row['seconds']:.1f} s]", flush=True)
    out["convergence_settings"] = {"shots": S_, "iterations": args.e4_conv_iter}
    out["seconds"] = time.perf_counter() - t0
    return out


# ====================================================================== E5
def run_e5(c, args, e2: dict | None) -> dict:
    t0 = time.perf_counter()
    out = {"sector": c.info["sector"], "long": {}, "e2": None}
    if e2 is not None:
        out["e2"] = e2.get("trajectories")
    for opt in ("sd", "lbfgs"):
        for S_ in (math.inf,) + tuple(args.e5_shots):
            t = time.perf_counter()
            rows = []
            seeds = [0] if math.isinf(S_) else range(args.seeds)
            for i in seeds:
                est = None
                if not math.isinf(S_):
                    est = clone_residual(c.tmpl["corr"], S_, np.random.default_rng(seed_seq(args.seed, "e5", c.name, i)),
                                         L.sampling.ShotLedger())
                tr = Trajectory(c.basis_sector, c.hf, c.e_hf)
                L.run_cqe(c.act, max_iter=args.e5_iter, e_exact=c.e_fci, optimizer=opt, r_tol=0.0, e_tol=0.0,
                          residual_estimator=est, callback=tr)
                rows.append(tr.rows)
            out["long"][f"{opt}/{S_:g}"] = rows
            print(f"[{c.name}] E5 {opt} S={S_:.0e}: final total {np.median([r[-1]['total'] for r in rows]):.2e}, "
                  f"in-null {np.median([r[-1]['in_null'] for r in rows]):.2e}, out "
                  f"{np.median([r[-1]['out_null'] + r[-1]['out_soft'] + r[-1]['out_stiff'] for r in rows]):.2e} "
                  f"[{time.perf_counter() - t:.1f} s]", flush=True)
    out["seconds"] = time.perf_counter() - t0
    return out


# ====================================================================== markdown
def _g(x, digits: int = 3) -> str:
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    x = float(x)
    if math.isnan(x):
        return "–"
    if not math.isfinite(x):
        return "inf"
    if x == 0:
        return "0"
    if 1e-2 <= abs(x) < 1e4:
        dec = max(0, digits - 1 - int(math.floor(math.log10(abs(x)))))
        return f"{x:.{dec}f}"
    return f"{x:.{max(digits - 1, 1)}e}".replace("e+0", "e").replace("e-0", "e-").replace("e+", "e")


def _b(st: dict | None, digits: int = 3, kind: str = "g") -> str:
    """median [q16, q84] of a band dict."""
    if st is None:
        return "–"
    f = _it if kind == "it" else (lambda v: _g(v, digits))
    if st["n"] == 1:
        return f(st["median"])
    return f"{f(st['median'])} [{f(st['q16'])}, {f(st['q84'])}]"


def _it(x) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    return f"{x:g}" if isinstance(x, float) else str(x)


def _ok(st: dict | None) -> str:
    return "–" if st is None else f"{st['n_ok']}/{st['n']}"


def _S(x) -> str:
    return "inf" if x is None or not math.isfinite(float(x)) else f"{float(x):.0e}".replace("e+0", "e")


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def md_setup(results: dict) -> list[str]:
    lines = ["## Systems and noise models", "",
             "Linearized CQE at FCI (coupled singlets; null: lambda <= 1e-10 lambda_max; soft: lambda < 1e-3 "
             "lambda_max), the fixed SD step of E1 and its stability limit 4/lambda_max, the per-shot residual "
             "noise (direct route, matchgate shadows, spin-orbital elements: sum_e v_e for independent elements, "
             "tr C of the snapshot covariance), the predicted floor ratio corr / indep (same eps), and the shifted "
             "DF energy variance (leaves used in E3; HF / FCI ratio). The clone check is the max |difference| "
             "between a shared-template model and a fresh ResidualNoise on the same stream.", "",
             "| system | N | dets | E_HF - E_FCI (mHa) | coupled | null | soft | fast / slow at eps | lambda_min (non-null) "
             "| lambda_max | eps (4/lambda_max) | sum v (indep) | tr C (corr; snapshots) | floor corr/indep | DF leaves | "
             "V_DF at FCI (Ha^2) | V_DF HF/FCI | clone check | setup (s) |",
             "|" + "---|" * 19]
    for name, r in results.items():
        s = r["setup"]
        e1 = r.get("e1")
        ratio = "–"
        if e1:
            a = e1["direct"]["corr"]["theory_S1"]["stationary"]["total"]
            b = e1["direct"]["indep"]["theory_S1"]["stationary"]["total"]
            ratio = _g(a / b)
        f = s["df_first"]
        lines.append(
            f"| {name} | {s['norb']} | {s['n_det']} | {_g(1e3 * (s['e_hf'] - s['e_fci']))} | {s['n_coupled']} | "
            f"{s['n_null']} | {s['n_soft']} | {s['n_fast']} / {s['n_slow']} | {_g(s['lambda_min_nonnull'])} | "
            f"{_g(s['lambda_max'])} | {s['eps']} ({_g(s['eps_stability'])}) | {_g(s['var_spin_orbital_total'])} | "
            f"{_g(s['cov_trace'])} ({_g(float(s['cov_snapshots']))}) | {ratio} | {f} | {_g(s[f'df_{f}_shift_fci'])} | "
            f"{_g(s[f'df_{f}_shift_hf'] / s[f'df_{f}_shift_fci'])} | {_g(s['clone_check_max_abs_diff'])} | "
            f"{_g(s['seconds_setup'])} |")
    lines.append("")
    return lines


def md_e1(results: dict) -> list[str]:
    lines = ["## E1: floor versus residual shots (fixed-step SD from FCI, direct route, matchgate shadows)", ""]
    any_ = next((r["e1"] for r in results.values() if r.get("e1")), None)
    if any_ is None:
        return []
    d = any_["direct"]["corr"]
    lines += [f"Burn-in {d['burn']} steps, then the time average over {d['window']} steps. 'expected' is the "
              "transient-aware linear prediction for that window (from FCI); 'stationary' is noise_floor. "
              "Ratios are simulated / expected per seed (median [16-84%]). fast: |1 - eps lambda/2| < 0.98; slow: "
              "the other non-null modes; null: lambda <= 1e-10 lambda_max (undriven at first order, expected 0); "
              "outside: E - E_FCI minus the sum of y_m^2, the energy in states outside the coupled singlets "
              "(E - E_0 = sum_k w_k d_k^2 holds exactly for a normalized state, so it is zero up to round-off while "
              "the state stays a singlet). Mode energies use the exact overlaps d_k, so 'ratio' tests the linearized "
              "dynamics, not the energy expansion. Common random numbers across S (the same z for every S), so in "
              "the linear regime the ratios do not depend on S.", "",
              "### Theory: floors and shots per iteration", "",
              "Floor x S (Ha x shots; divide by S). S for a floor: from the stationary total, from its fast modes, "
              "from the window expectation, and from the simulated floors (log-log interpolation of the medians; "
              "'–' outside the grid or not simulated). indep_sf: spin-free elements (theory only).", "",
              "| system | model | eps | stationary total | stationary fast / slow (soft) | window total | "
              "S(1.6 mHa): stationary / fast / window / simulated | S(0.1 mHa): stationary / fast / window / simulated |",
              "|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        e1 = r.get("e1")
        if not e1:
            continue
        for model in ("corr", "indep", "indep_sf"):
            x = e1["direct"].get(model)
            if not x:
                continue
            th = x["theory_S1"]
            fs = x["floor_shots"]
            a, b = fs[f"{TARGET:g}"], fs[f"{FINE:g}"]
            lines.append(
                f"| {name} | {model} | {e1['eps']} | {_g(th['stationary']['total'])} | {_g(th['stationary']['fast'])} / "
                f"{_g(th['stationary']['slow'])} ({_g(th['stationary']['soft'])}) | {_g(th['window']['total'])} | "
                f"{_g(a['stationary'])} / {_g(a['stationary_fast'])} / {_g(a['window'])} / {_g(a['simulated'])} | "
                f"{_g(b['stationary'])} / {_g(b['stationary_fast'])} / {_g(b['window'])} / {_g(b['simulated'])} |")
    lines += ["", "### Simulation against theory", "",
              "ratio: simulated / expected (the transient-aware mean of the window). twin: simulated / the linear "
              "CQE driven by the same residual errors (LinearTwin: the linear theory on the same noise path), which "
              "removes the statistical scatter of a finite window and leaves the nonlinear effects.", "",
              "| system | model | S | expected total | simulated total | ratio total | twin total | expected fast | "
              "ratio fast | twin fast | expected slow | ratio slow | twin slow | null (sim) | outside (sim) | s |",
              "|" + "---|" * 16]
    for name, r in results.items():
        e1 = r.get("e1")
        if not e1:
            continue
        for model in ("corr", "indep"):
            for row in e1["direct"][model]["rows"]:
                lines.append(
                    f"| {name} | {model} | {_S(row['shots'])} | {_g(row['expected_total'])} | {_b(row['total'])} | "
                    f"{_b(row['ratio_total'])} | {_b(row.get('twin_ratio_total'), 4)} | {_g(row['expected_fast'])} | "
                    f"{_b(row['ratio_fast'])} | {_b(row.get('twin_ratio_fast'), 4)} | {_g(row['expected_slow'])} | "
                    f"{_b(row['ratio_slow'])} | {_b(row.get('twin_ratio_slow'), 4)} | {_b(row['null'])} | "
                    f"{_b(row['outside'])} | {_g(row['seconds'])} |")
    lines += ["", "### Aux route", "",
              "Exact exp(-i delta H), independent elements (matchgate shadows), S per iteration as given: the "
              "residual of excitation k is damped by sinc(delta w_k), so the SD map is I - (eps/2) K diag(sinc). "
              "rho: spectral radius of that map (> 1: unstable); min eig: smallest eigenvalue of K diag(sinc). "
              "'plain' is noise_floor without the damping (stationary); 'damped' solves the Lyapunov equation of "
              "the damped map (stationary, and the window expectation). max error: the largest E - E_FCI of the run.",
              "",
              "| system | delta | S | sum v | min sinc | min eig K D | rho | plain stationary | damped stationary | "
              "damped window | simulated | ratio total | ratio fast | max error | s |",
              "|" + "---|" * 15]
    for name, r in results.items():
        e1 = r.get("e1")
        if not e1:
            continue
        for a in e1["aux_exact"]:
            S_ = a["shots"]
            lines.append(
                f"| {name} | {a['delta']} | {_S(S_)} | {_g(a['var_total'])} | {_g(a['sinc_min'])} | "
                f"{_g(a['min_eig_KD'])} | {_g(a['rho'], 6)} | {_g(a['plain_stationary_S1'] / S_)} | "
                f"{_g(None if a['damped_stationary_S1'] is None else a['damped_stationary_S1'] / S_)} | "
                f"{_g(None if a['damped_window_S1'] is None else a['damped_window_S1'] / S_)} | {_b(a['total'])} | "
                f"{_b(a.get('ratio_total'))} | {_b(a.get('ratio_fast'))} | {_b(a['max_error'])} | {_g(a['seconds'])} |")
    lines += ["", "DF methods at S = inf (bias only): ||b|| is the Frobenius norm of the filled bias tensor at FCI; "
              "the stationary offset is bias_offset (||B^+ b||^2) with the share of its soft modes; the transient is "
              "sum_m (y*_m (1 - p_m^n))^2 after n steps; ratios are simulated / transient (total and fast modes).", "",
              "| system | method | delta | steps n | norm b | stationary offset | soft share | transient after n | "
              "simulated after n | ratio total | ratio fast | s |",
              "|" + "---|" * 12]
    for name, r in results.items():
        e1 = r.get("e1")
        if not e1:
            continue
        for a in e1["aux_bias"]:
            lines.append(
                f"| {name} | {a['method']} | {a['delta']} | {a['steps']} | {_g(a['bias_norm'])} | "
                f"{_g(a['offset_stationary'])} | {_pct(a['offset_stationary_soft_share'])} | {_g(a['transient_total'])} | "
                f"{_g(a['simulated']['total'])} | {_g(a['ratio_total'])} | {_g(a['ratio_fast'])} | {_g(a['seconds'])} |")
    lines.append("")
    return lines


def md_e2(results: dict) -> list[str]:
    lines = ["## E2: optimizers with noisy residuals and exact energies (oracle line search)", ""]
    first = next((r["e2"] for r in results.values() if r.get("e2")), None)
    if first is None:
        return []
    lines += [f"From HF, {first['iterations']} iterations, correlated Gaussian residual noise, exact Krylov line "
              "search along the noisy direction. Iterations count CQE steps (0 = HF); 'stay' is the first iteration "
              "after which E - E_FCI stays below 1.6 mHa to the end; residual shots = S x iterations. uphill: steps "
              "whose noisy direction has exact slope >= 0 (the oracle then returns a void step); void: |eps| < 1e-6. "
              "Restarts/resets/skips: median per run. S = inf is the exact CQE (one run).", "",
              "| system | optimizer | S_res | final error | best error | first < 1.6 mHa (n ok) | stay (n ok) | "
              "residual shots to stay | uphill | void | CG restarts | L-BFGS resets / skips | s |",
              "|" + "---|" * 13]
    for name, r in results.items():
        e2 = r.get("e2")
        if not e2:
            continue
        for row in e2["rows"]:
            st = row["stats"]
            sh = st.get("shots_stay")
            lines.append(
                f"| {name} | {row['optimizer']} | {_S(row['shots'])} | {_b(st['final'])} | {_b(st['best'])} | "
                f"{_b(st['reach'], kind='it')} ({_ok(st['reach'])}) | {_b(st['stay'], kind='it')} ({_ok(st['stay'])}) | "
                f"{_b(sh) if sh else '–'} | {_g(st['uphill']['median'])} | {_g(st['void']['median'])} | "
                f"{_g(st['cg_restarts']['median']) if row['optimizer'] == 'cg' else '–'} | "
                + (f"{_g(st['lbfgs_resets']['median'])} / {_g(st['lbfgs_skips']['median'])}"
                   if row["optimizer"] == "lbfgs" else "–") + f" | {_g(row['seconds'])} |")
    lines.append("")
    return lines


def _sets_present(results) -> list[str]:
    """The candidate sets of the E3 cells in the results, in the order of CANDIDATE_SETS."""
    have = {x["set"] for r in results.values() if r.get("e3") for x in r["e3"]["cells"]}
    return [s for s in CANDIDATE_SETS if s in have] or list(CANDIDATE_SETS)


def _cell(results, name, opt, set_name, S_res, S_E):
    for x in results[name]["e3"]["cells"]:
        if x["optimizer"] == opt and x["set"] == set_name and x["S_res"] == S_res and x["S_E"] == S_E:
            return x
    return None


def md_e3(results: dict) -> list[str]:
    first = next((r["e3"] for r in results.values() if r.get("e3")), None)
    if first is None:
        return []
    sets = _sets_present(results)
    lines = ["## E3: noisy residuals and noisy DF energies with fixed step candidates", "",
             f"From HF, {first['iterations']} iterations. Candidate sets: pos11 = 0.00625 * 2^k (k = 0..10); "
             "pos11+0 = pos11 and 0; eq11 = 0, +-{0.0125, 0.05, 0.2, 0.8, 3.2} (equal energy cost: 10 estimates "
             "against 11); dbl23 = 0, +-pos11 (22 estimates). Shots per iteration: S_res + (1 + nonzero candidates) "
             "S_E. Total shots to reach and stay below 1.6 mHa (median [16-84%], failures = inf; n ok of n seeds).", ""]
    lines += ["### Exact references (no noise, same candidate sets)", "",
              "| system | optimizer | " + " | ".join(f"{s}: first < 1.6 mHa / final" for s in sets) + " |",
              "|---|---|" + "---|" * len(sets)]
    for name, r in results.items():
        e3 = r.get("e3")
        if not e3:
            continue
        for opt in OPTIMIZERS:
            cells = []
            for s in sets:
                m = next((x for x in e3["exact"] if x["optimizer"] == opt and x["set"] == s), None)
                cells.append("–" if m is None else f"{_it(m['reach'])} / {_g(m['final'])}")
            lines.append(f"| {name} | {opt} | " + " | ".join(cells) + " |")
    lines += ["", "### Candidate sets compared", "",
              "Pairs of sets over all cells (system, optimizer, S_res, S_E): cells where the first set needs fewer "
              "/ the same / more median shots to reach and stay (both reached), where only one reached, and the "
              "geometric mean of the ratio of median final errors (first / second). Seeds: the same seed of the "
              "two cells (common random numbers; pos11 and pos11+0 share every draw, so their runs are identical "
              "until zero is first chosen), counted by which set ends with the lower final error.", "",
              "| comparison | fewer shots | same | more shots | only first reached | only second reached | neither | "
              "final error ratio (geo. mean) | seeds: first lower / equal / higher final | cells |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for a, b in (("pos11+0", "pos11"), ("eq11", "pos11"), ("dbl23", "pos11+0"), ("dbl23", "pos11"),
                 ("eq11", "pos11+0")):
        cnt = {"fewer": 0, "tie": 0, "more": 0, "only_a": 0, "only_b": 0, "neither": 0}
        seeds = {"lower": 0, "equal": 0, "higher": 0}
        logs = []
        n = 0
        for name, r in results.items():
            e3 = r.get("e3")
            if not e3:
                continue
            for x in e3["cells"]:
                if x["set"] != a:
                    continue
                y = _cell(results, name, x["optimizer"], b, x["S_res"], x["S_E"])
                if y is None:
                    continue
                n += 1
                sa, sb = x["stats"]["shots_stay"]["median"], y["stats"]["shots_stay"]["median"]
                fa, fb = math.isfinite(sa), math.isfinite(sb)
                if fa and fb:
                    cnt["fewer" if sa < sb else ("tie" if sa == sb else "more")] += 1
                elif fa:
                    cnt["only_a"] += 1
                elif fb:
                    cnt["only_b"] += 1
                else:
                    cnt["neither"] += 1
                ea, eb = x["stats"]["final"]["median"], y["stats"]["final"]["median"]
                if ea > 0 and eb > 0:
                    logs.append(math.log(ea / eb))
                fy = {ry["seed"]: ry["final"] for ry in y["runs"]}
                for rx in x["runs"]:
                    if rx["seed"] in fy:
                        d = rx["final"] - fy[rx["seed"]]
                        seeds["lower" if d < 0 else ("equal" if d == 0 else "higher")] += 1
        if n == 0:
            continue
        gm = math.exp(sum(logs) / len(logs)) if logs else None
        lines.append(f"| {a} vs {b} | {cnt['fewer']} | {cnt['tie']} | {cnt['more']} | {cnt['only_a']} | "
                     f"{cnt['only_b']} | {cnt['neither']} | {_g(gm)} | {seeds['lower']} / {seeds['equal']} / "
                     f"{seeds['higher']} | {n} |")
    lines += ["", "### How often zero and negative candidates are chosen, and what they gain", "",
              "Fractions of iterations (all, and in the second half of the runs, 'late'), averaged over seeds, "
              "systems and the (S_res, S_E) grid. uphill: the noisy direction has exact slope >= 0. best neg / best "
              "zero: the exact-best candidate of the set is negative / zero. gain: E_exact(chosen) - E_exact(the "
              "positive candidate the same estimates would have picked), summed over the zero or negative choices "
              "of the late half (negative = the choice helped), as mean per such choice (mHa) and the counts that "
              "helped / hurt. The means are dominated by rare large events (a noise-corrupted L-BFGS direction can "
              "make every positive candidate raise the energy by up to Ha; the zero candidate refuses such steps), "
              "so the counts and the split below are the robust statistics.", "",
              "| optimizer | set | zero chosen | neg chosen | uphill | best neg | best zero | late: zero | late: neg | "
              "late: uphill | late: best neg | late gain per choice (mHa) | late helped / hurt | late regret (mHa) |",
              "|" + "---|" * 14]
    for opt in OPTIMIZERS:
        for s in sets:
            cs = [x for r in results.values() if r.get("e3") for x in r["e3"]["cells"]
                  if x["optimizer"] == opt and x["set"] == s]
            if not cs:
                continue

            def avg(key, label="all"):
                return float(np.mean([x[f"usage_{label}"][key] for x in cs]))

            npos = sum(x["usage_late"]["n_nonpos"] for x in cs)
            gsum = sum(x["usage_late"]["gain_sum"] for x in cs)
            helped = sum(x["usage_late"]["n_helped"] for x in cs)
            hurt = sum(x["usage_late"]["n_hurt"] for x in cs)
            lines.append(
                f"| {opt} | {s} | {_pct(avg('zero'))} | {_pct(avg('neg'))} | {_pct(avg('uphill'))} | "
                f"{_pct(avg('best_neg'))} | {_pct(avg('best_zero'))} | {_pct(avg('zero', 'late'))} | "
                f"{_pct(avg('neg', 'late'))} | {_pct(avg('uphill', 'late'))} | {_pct(avg('best_neg', 'late'))} | "
                f"{_g(1e3 * gsum / npos) if npos else '–'} | {helped} / {hurt} | {_g(1e3 * avg('regret_mean', 'late'))} |")
    lines += ["", "Usage by shots (sets with zero and negative candidates; averaged over systems and optimizers):", "",
              "| set | S_res | S_E | zero chosen | neg chosen | uphill | best neg | best zero | late gain per choice (mHa) | "
              "late helped / hurt |", "|" + "---|" * 10]
    grid = sorted({(x["S_res"], x["S_E"]) for r in results.values() if r.get("e3") for x in r["e3"]["cells"]})
    for s in sets[1:]:
        for sr, se in grid:
            cs = [x for r in results.values() if r.get("e3") for x in r["e3"]["cells"]
                  if x["set"] == s and x["S_res"] == sr and x["S_E"] == se]
            if not cs:
                continue
            u = {k: float(np.mean([x["usage_all"][k] for x in cs])) for k in ("zero", "neg", "uphill", "best_neg",
                                                                               "best_zero")}
            npos = sum(x["usage_late"]["n_nonpos"] for x in cs)
            gsum = sum(x["usage_late"]["gain_sum"] for x in cs)
            lines.append(f"| {s} | {_S(sr)} | {_S(se)} | {_pct(u['zero'])} | {_pct(u['neg'])} | {_pct(u['uphill'])} | "
                         f"{_pct(u['best_neg'])} | {_pct(u['best_zero'])} | {_g(1e3 * gsum / npos) if npos else '–'} | "
                         f"{sum(x['usage_late']['n_helped'] for x in cs)} / {sum(x['usage_late']['n_hurt'] for x in cs)} |")
    lines += _md_split(results, sets)
    lines += ["", "### All cells", "",
              "Per cell: total shots to reach and stay below 1.6 mHa, median (n ok), and the median final error.", "",
              "| system | optimizer | S_res | S_E | " + " | ".join(f"{s}: shots (n ok) / final" for s in sets) + " |",
              "|---|---|---|---|" + "---|" * len(sets)]
    for name, r in results.items():
        e3 = r.get("e3")
        if not e3:
            continue
        keys = sorted({(x["optimizer"], x["S_res"], x["S_E"]) for x in e3["cells"]},
                      key=lambda k: (OPTIMIZERS.index(k[0]), k[1], k[2]))
        for opt, sr, se in keys:
            cells = []
            for s in sets:
                x = _cell(results, name, opt, s, sr, se)
                if x is None:
                    cells.append("–")
                    continue
                st = x["stats"]
                cells.append(f"{_g(st['shots_stay']['median'])} ({st['shots_stay']['n_ok']}) / {_g(st['final']['median'])}")
            lines.append(f"| {name} | {opt} | {_S(sr)} | {_S(se)} | " + " | ".join(cells) + " |")
    lines += ["", "### Check of the per-iteration energy variance", "",
              "Paired runs (same random streams) with the variance evaluated once per iteration (this benchmark) and "
              "at every candidate state (EnergyNoise variance='exact').", "",
              "| system | optimizer / set | S_res | S_E | seed | final (iteration) | final (exact) | stay (iteration / exact) | "
              "max abs error difference | s (iteration / exact) |", "|" + "---|" * 10]
    for name, r in results.items():
        e3 = r.get("e3")
        if not e3:
            continue
        vc = e3["variance_check"]
        for x in vc["runs"]:
            lines.append(f"| {name} | {vc['optimizer']} / {vc['set']} | {_S(vc['S_res'])} | {_S(vc['S_E'])} | {x['seed']} | "
                         f"{_g(x['final_iteration'])} | {_g(x['final_exact'])} | {_it(x['stay_iteration'])} / "
                         f"{_it(x['stay_exact'])} | {_g(x['max_abs_error_diff'])} | {_g(x['seconds_iteration'])} / "
                         f"{_g(x['seconds_exact'])} |")
    lines.append("")
    return lines


def _md_split(results: dict, sets) -> list[str]:
    """Zero and negative choices split (all iterations; needs the per-iteration gains of candidate_analysis)."""
    rows = []
    for opt in OPTIMIZERS:
        for s in sets:
            acc = {"zero": [], "neg": [], "zero_up": 0, "neg_up": 0, "n": 0}
            for r in results.values():
                for x in (r.get("e3") or {}).get("cells", []):
                    if x["optimizer"] != opt or x["set"] != s:
                        continue
                    for run in x["runs"]:
                        ca = run["candidates"]
                        if "gain" not in ca:
                            continue
                        for sg, g, up in zip(ca["chosen_sign"], ca["gain"], ca["uphill"]):
                            acc["n"] += 1
                            if sg == 0:
                                acc["zero"].append(g)
                                acc["zero_up"] += up
                            elif sg < 0:
                                acc["neg"].append(g)
                                acc["neg_up"] += up
            if acc["n"] and (acc["zero"] or acc["neg"]):
                rows.append((opt, s, acc))
    if not rows:
        return []
    lines = ["", "Zero and negative choices split (all iterations, all systems and grid points): n chosen (share of "
             "iterations), helped / hurt (gain < 0 / > 0), median and mean gain per choice (mHa), and how many of "
             "the choices were made on uphill directions.", "",
             "| optimizer | set | iterations | zero: n | helped / hurt | median / mean gain | on uphill | negative: n | "
             "helped / hurt | median / mean gain | on uphill |", "|" + "---|" * 11]
    for opt, s, a in rows:
        cells = []
        for k in ("zero", "neg"):
            g = np.asarray(a[k], dtype=float)
            if g.size == 0:
                cells.append("0 | – | – | –")
                continue
            cells.append(f"{g.size} ({_pct(g.size / a['n'])}) | {int((g < 0).sum())} / {int((g > 0).sum())} | "
                         f"{_g(1e3 * float(np.median(g)))} / {_g(1e3 * float(g.mean()))} | {a[k + '_up']}")
        lines.append(f"| {opt} | {s} | {a['n']} | " + " | ".join(cells) + " |")
    return lines


def md_e4(results: dict) -> list[str]:
    r = results.get("h4", {}).get("e4")
    if not r:
        return []
    f = r["floor"]
    lines = ["## E4 (H4): real matchgate snapshots against the Gaussian models", "",
             f"Floor: fixed-step SD from FCI, eps = {f['eps']}, {f['shots']} snapshots per iteration, burn-in "
             f"{f['burn']} + window {f['window']} steps. Linear predictions for that window: correlated "
             f"{_g(f['theory']['corr']['window'])} (stationary {_g(f['theory']['corr']['stationary'])}), independent "
             f"{_g(f['theory']['indep']['window'])} (stationary {_g(f['theory']['indep']['stationary'])}) Ha.", "",
             "twin: simulated / the linear CQE driven by the same residual errors (for the snapshots: the actual "
             "snapshot errors), which tests the nonlinearity path by path.", "",
             "| model | simulated E - E_FCI | ratio to corr theory | ratio to indep theory | fast-mode ratio to corr | "
             "twin total | twin fast | outside | s |", "|---|---|---|---|---|---|---|---|---|"]
    for model in ("snapshot", "corr", "indep"):
        x = f[model]
        lines.append(f"| {model} | {_b(x['total'])} | {_b(x['ratio_corr'])} | {_b(x['ratio_indep'])} | "
                     f"{_b(x['ratio_fast_corr'])} | {_b(x.get('twin_ratio_total'), 4)} | "
                     f"{_b(x.get('twin_ratio_fast'), 4)} | {_b(x['outside'])} | {_g(x['seconds'])} |")
    cs = r["convergence_settings"]
    lines += ["", f"Convergence from HF, oracle line search, {cs['shots']} snapshots (or Gaussian shots) per "
              f"iteration, {cs['iterations']} iterations.", "",
              "| model | optimizer | final error | best error | first < 1.6 mHa (n ok) | stay (n ok) | uphill | s |",
              "|---|---|---|---|---|---|---|---|"]
    for key, x in r["convergence"].items():
        model, opt = key.split("/")
        st = x["stats"]
        lines.append(f"| {model} | {opt} | {_b(st['final'])} | {_b(st['best'])} | {_b(st['reach'], kind='it')} "
                     f"({_ok(st['reach'])}) | {_b(st['stay'], kind='it')} ({_ok(st['stay'])}) | "
                     f"{_g(st['uphill']['median'])} | {_g(x['seconds'])} |")
    lines.append("")
    return lines


E5_MARKS = (0, 3, 10, 30, 60, 100, 150)


def md_e5(results: dict) -> list[str]:
    r = results.get("n2", {}).get("e5")
    if not r:
        return []
    sec = r["sector"]
    lines = ["## E5 (N2): energy per mode class along noisy runs", "",
             f"K is block diagonal by symmetry: {sec['n_components']} blocks of the coupling graph "
             f"(sizes {sec['component_sizes']}); the HF symmetry sector is the block holding the HF determinant "
             f"({sec['n_in']} coupled states: {sec['n_in_null']} null, {sec['n_in_soft']} soft, {sec['n_in_stiff']} "
             f"stiff modes), the other {sec['n_out']} coupled states hold {sec['n_out_null']} null, {sec['n_out_soft']} "
             f"soft and {sec['n_out_stiff']} stiff modes (largest cross-block |K| {_g(sec['cross_block_max_abs_K'])} of "
             f"{_g(sec['K_max'])}). Energies in Ha (median over seeds) of y_m^2 per class (exact overlaps, so the "
             "classes sum to E - E_FCI); outside = E - E_FCI minus their sum (states outside the coupled singlets, "
             "zero up to round-off). Oracle line search, correlated Gaussian residual noise.", "",
             "| run | S_res | iteration | E - E_FCI | in: null | in: soft | in: stiff | out: null | out: soft | "
             "out: stiff | outside |", "|" + "---|" * 11]

    def rows_for(label, runs, S_):
        for it in E5_MARKS:
            if it >= len(runs[0]):
                continue
            med = {k: float(np.median([run[it][k] for run in runs]))
                   for k in ("total", "in_null", "in_soft", "in_stiff", "out_null", "out_soft", "out_stiff", "outside")}
            lines.append(f"| {label} | {S_} | {it} | {_g(med['total'])} | {_g(med['in_null'])} | {_g(med['in_soft'])} | "
                         f"{_g(med['in_stiff'])} | {_g(med['out_null'])} | {_g(med['out_soft'])} | "
                         f"{_g(med['out_stiff'])} | {_g(med['outside'])} |")

    for key, runs in r["long"].items():
        opt, S_ = key.split("/")
        rows_for(f"{opt} (long)", runs, "inf" if S_ == "inf" else _S(float(S_)))
    if r.get("e2"):
        for key, runs in r["e2"].items():
            opt, S_ = key.split("/")
            if opt == "cg":
                rows_for(f"{opt} (E2)", runs, "inf" if S_ == "inf" else _S(float(S_)))
    lines.append("")
    return lines


def cheapest(results: dict) -> dict:
    """Per system: the E3 cell with the lowest median shots to reach and stay below 1.6 mHa (all seeds ok)."""
    out = {}
    for name, r in results.items():
        e3 = r.get("e3")
        if not e3:
            continue
        best, by_set, by_opt = None, {}, {}
        for x in e3["cells"]:
            st = x["stats"]["shots_stay"]
            if st["n_ok"] < st["n"]:
                continue
            v = st["median"]
            if best is None or v < best["stats"]["shots_stay"]["median"]:
                best = x
            if x["set"] not in by_set or v < by_set[x["set"]]["stats"]["shots_stay"]["median"]:
                by_set[x["set"]] = x
            if x["optimizer"] not in by_opt or v < by_opt[x["optimizer"]]["stats"]["shots_stay"]["median"]:
                by_opt[x["optimizer"]] = x
        out[name] = {"best": best, "by_set": by_set, "by_opt": by_opt}
    return out


def _cfg(x) -> str:
    if x is None:
        return "none reached in all seeds"
    st = x["stats"]
    return (f"{x['optimizer']}, {x['set']}, S_res {_S(x['S_res'])}, S_E {_S(x['S_E'])}: "
            f"{_b(st['shots_stay'])} shots, stay at iteration {_b(st['stay'], kind='it')}")


def md_summary(results: dict) -> list[str]:
    ch = cheapest(results)
    if not ch:
        return []
    lines = ["## Summary: the cheapest configuration to reach and stay below 1.6 mHa", "",
             "Route and scheme: direct route with matchgate shadows (correlated Gaussian of real snapshots, "
             "spin-orbital elements) and shifted-DF energies, as in E3. Among the E3 cells in which every seed "
             "reached and stayed below 1.6 mHa, the one with the lowest median total shots; per candidate set and "
             "per optimizer the best cell as well. Exact CQE iterations to < 1.6 mHa: exact line search (E2, S = inf) "
             "and the pos11 candidates without noise (E3), SD / CG / L-BFGS.", "",
             "| system | exact LS: SD / CG / L-BFGS | pos11 no noise: SD / CG / L-BFGS | cheapest | residual share | "
             "best per candidate set | best per optimizer |", "|---|---|---|---|---|---|---|"]
    for name, x in ch.items():
        r = results[name]
        ex = "–"
        if r.get("e2"):
            ex = " / ".join(_it(next(row for row in r["e2"]["rows"] if row["optimizer"] == o and
                                     not math.isfinite(row["shots"]))["runs"][0]["reach"]) for o in OPTIMIZERS)
        pc = " / ".join(_it(next(m for m in r["e3"]["exact"] if m["optimizer"] == o and m["set"] == "pos11")["reach"])
                        for o in OPTIMIZERS)
        b = x["best"]
        share = "–"
        if b is not None:
            fr = [run["ledger"]["residual_shots"] / run["ledger"]["total_shots"] for run in b["runs"]]
            share = _pct(float(np.median(fr)))
        sets = "<br>".join(f"{s}: {_cfg(x['by_set'].get(s))}" for s in _sets_present(results))
        opts = "<br>".join(f"{o}: {_cfg(x['by_opt'].get(o))}" for o in OPTIMIZERS)
        lines.append(f"| {name} | {ex} | {pc} | {_cfg(b)} | {share} | {sets} | {opts} |")
    lines.append("")
    return lines


def md_timing(results: dict, total: float) -> list[str]:
    lines = ["## Runtime (s, one thread)", "", "| system | setup | e1 | e2 | e3 | e4 | e5 | total |",
             "|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        t = r.get("timing", {})
        lines.append(f"| {name} | " + " | ".join(_g(t.get(k)) for k in ("setup", "e1", "e2", "e3", "e4", "e5", "total"))
                     + " |")
    lines.append(f"| all | | | | | | | {_g(total)} |")
    lines.append("")
    return lines


def markdown(results: dict, settings: dict, total: float) -> str:
    lines = ["# The finite-shot CQE (M2, workflow C)", "",
             "Generated by `benchmarks/run_sampled_cqe.py` (conventions, models and reductions: its module "
             f"docstring). Systems: {', '.join(results)}; seeds per setting: {settings['seeds']}; base seed "
             f"{settings['seed']}; threads: {settings.get('threads_env', {}).get('threads')}. Errors are exact "
             "E - E_FCI (Ha) of the states the noisy runs produced; cells give median [16-84%] over seeds.", ""]
    lines += md_setup(results)
    lines += md_summary(results)
    lines += md_e1(results)
    lines += md_e2(results)
    lines += md_e3(results)
    lines += md_e4(results)
    lines += md_e5(results)
    lines += md_timing(results, total)
    return "\n".join(lines)


# ====================================================================== driver
def _json_default(x):
    if np is not None:
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            x = float(x)
        elif isinstance(x, np.ndarray):
            return x.tolist()
        elif isinstance(x, np.bool_):
            return bool(x)
    if isinstance(x, float):
        return x if math.isfinite(x) else str(x)
    if isinstance(x, (set, tuple)):
        return list(x)
    return str(x)


def _sanitize(obj):
    """Replace non-finite floats by strings ("inf", "nan") so the JSON is standard."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else ("inf" if obj > 0 else ("-inf" if obj < 0 else "nan"))
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def _restore(obj):
    if isinstance(obj, str) and obj in ("inf", "-inf", "nan"):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _restore(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore(v) for v in obj]
    return obj


def run_system(name: str, args) -> dict:
    t0 = time.perf_counter()
    c = setup_system(name, args)
    s = c.info
    print(f"[{name}] {s['label']}: N={s['norb']}, dets={s['n_det']}, coupled={s['n_coupled']} (null {s['n_null']}, "
          f"soft {s['n_soft']}), lambda_max={s['lambda_max']:.1f}, sum v={s['var_spin_orbital_total']:.4g}, "
          f"tr C={s['cov_trace']:.4g}, clone diff={s['clone_check_max_abs_diff']:.1e} "
          f"[{s['seconds_setup']:.1f} s]", flush=True)
    out = {"system": name, "setup": s, "timing": {"setup": s["seconds_setup"]}}
    if "e1" in args.experiments:
        out["e1"] = run_e1(c, args)
        out["timing"]["e1"] = out["e1"]["seconds"]
    tracker = c.basis_sector if name == "n2" and "e5" in args.experiments else None
    if "e2" in args.experiments:
        out["e2"] = run_e2(c, args, tracker)
        out["timing"]["e2"] = out["e2"]["seconds"]
    if "e3" in args.experiments:
        out["e3"] = run_e3(c, args)
        out["timing"]["e3"] = out["e3"]["seconds"]
    if "e4" in args.experiments and name == "h4":
        out["e4"] = run_e4(c, args)
        out["timing"]["e4"] = out["e4"]["seconds"]
    if "e5" in args.experiments and name == "n2":
        out["e5"] = run_e5(c, args, out.get("e2"))
        out["timing"]["e5"] = out["e5"]["seconds"]
        if out.get("e2"):
            out["e2"].pop("trajectories", None)     # stored under e5
    out["timing"]["total"] = time.perf_counter() - t0
    return out


def write_outputs(results: dict, args, settings: dict, total: float) -> None:
    md = markdown(results, settings, total)
    with open(os.path.join(args.out, "sampled_summary.md"), "w") as fh:
        fh.write(md + "\n")
    summ = {"settings": settings, "seconds": total,
            "timing": {k: v.get("timing") for k, v in results.items()},
            "cheapest": {k: {kk: (None if vv is None else {q: vv[q] for q in ("optimizer", "set", "S_res", "S_E", "stats")})
                             for kk, vv in (("best", v["best"]),)} for k, v in cheapest(results).items()}}
    with open(os.path.join(args.out, "sampled_summary.json"), "w") as fh:
        json.dump(_sanitize(summ), fh, indent=1, default=_json_default)
    print(f"wrote {os.path.join(args.out, 'sampled_summary.md')} ({total:.1f} s)", flush=True)


def summary_only(args) -> int:
    results = {}
    for name in args.systems:
        path = os.path.join(args.out, f"sampled_{name}.json")
        if not os.path.exists(path):
            print(f"[{name}] skipped: {path} not found")
            continue
        with open(path) as fh:
            results[name] = _restore(json.load(fh))
    if not results:
        raise SystemExit(f"no sampled_<system>.json files in {args.out}")
    settings = dict(next(iter(results.values())).get("settings", {}), summary_only=True)
    # The systems may come from separate invocations: the total is the sum of the per-system run times.
    total = float(sum(r["timing"]["total"] for r in results.values()))
    write_outputs(results, args, settings, total)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", default=DEFAULT_SYSTEMS, help="comma-separated subset of " + ",".join(SYSTEMS))
    ap.add_argument("--experiments", default=",".join(EXPERIMENTS), help="comma-separated subset of e1..e5")
    ap.add_argument("--seeds", type=int, default=None, help="seeds per setting (default 5; --quick: 2)")
    ap.add_argument("--seed", type=int, default=2026, help="base seed of every random stream")
    ap.add_argument("--quick", action="store_true", help=f"systems {QUICK_SYSTEMS}, 2 seeds, short runs (< 3 min)")
    ap.add_argument("--e3-check-shots", default="1e6,1e6", help="S_res,S_E of the E3 energy-variance check")
    ap.add_argument("--e3-sets", default=",".join(CANDIDATE_SETS),
                    help="E3 candidate sets, comma-separated subset of " + ",".join(CANDIDATE_SETS))
    ap.add_argument("--e3-grid", default=None,
                    help="E3 (S_res:S_E) points, comma-separated, e.g. 1e6:1e6,1e5:1e7 (default: the preset)")
    for opt_name, help_ in (("e1-shots", "E1 shots (correlated model)"), ("e2-shots", "E2 residual shots"),
                            ("e5-shots", "E5 residual shots")):
        ap.add_argument(f"--{opt_name}", default=None, help=f"{help_} (comma-separated; default: the preset)")
    for opt_name in ("e1-window", "e1-burn", "e2-iter", "e3-iter", "e5-iter"):
        ap.add_argument(f"--{opt_name}", type=int, default=None, help="override the preset")
    ap.add_argument("--no-cache", action="store_true", help="do not read or write <out>/cache")
    ap.add_argument("--summary-only", action="store_true", help="rebuild the markdown from the JSON files in --out")
    ap.add_argument("--threads", type=int, default=None, help="PySCF OpenMP and numpy BLAS threads")
    ap.add_argument("--out", default="results/", help="output directory")
    args = ap.parse_args(argv)
    preset = QUICK if args.quick else FULL
    user_seeds = args.seeds
    overrides = {k: getattr(args, k) for k in ("e1_shots", "e2_shots", "e5_shots",
                                                "e1_window", "e1_burn", "e2_iter", "e3_iter", "e5_iter")}
    grid_override = args.e3_grid
    for k, v in preset.items():
        setattr(args, k, v)
    for k, v in overrides.items():
        if v is not None:
            setattr(args, k, tuple(float(x) for x in _names(v)) if isinstance(v, str) else v)
    if grid_override is not None:
        args.e3_grid = tuple(tuple(float(y) for y in x.split(":")) for x in _names(grid_override))
        overrides["e3_grid"] = args.e3_grid
    args.overrides = {k: v for k, v in overrides.items() if v is not None}
    args.seeds = preset["seeds"] if user_seeds is None else int(user_seeds)
    if args.quick and args.systems == DEFAULT_SYSTEMS:
        args.systems = QUICK_SYSTEMS
    args.systems = _names(args.systems)
    args.experiments = [e.lower() for e in _names(args.experiments)]
    bad = [e for e in args.experiments if e not in EXPERIMENTS]
    if bad:
        raise SystemExit(f"unknown experiments {bad}; choose from {EXPERIMENTS}")
    args.e3_check_shots = tuple(float(x) for x in _names(args.e3_check_shots))
    args.e3_sets = _names(args.e3_sets)
    bad = [x for x in args.e3_sets if x not in CANDIDATE_SETS]
    if bad:
        raise SystemExit(f"unknown candidate sets {bad}; choose from {list(CANDIDATE_SETS)}")
    args.threads_env = pin_threads(args.threads)
    _import_numerics()
    os.makedirs(args.out, exist_ok=True)
    if args.summary_only:
        return summary_only(args)
    settings = {k: v for k, v in vars(args).items()}
    t0 = time.perf_counter()
    results = {}
    for name in args.systems:
        res = run_system(name, args)
        res["settings"] = settings
        results[name] = res
        with open(os.path.join(args.out, f"sampled_{name}.json"), "w") as fh:
            json.dump(_sanitize(res), fh, default=_json_default)
        t = res["timing"]
        print(f"[{name}] done in {t['total']:.1f} s (" + ", ".join(f"{k} {v:.1f}" for k, v in t.items() if k != "total")
              + ")", flush=True)
        write_outputs(results, args, settings, time.perf_counter() - t0)
    write_outputs(results, args, settings, time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
