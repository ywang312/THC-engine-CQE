# THC engine for the CQE

This package provides tensor hypercontraction (THC) and double-factorization (DF)
tools for the **contracted quantum eigensolver** (CQE), the ACSE-based solver in
which Ψ ← exp(εA)Ψ is driven by the two-body residual of the anti-Hermitian
contracted Schrödinger equation.

## What it does

Each part has a different job:

- **Double factorization:** compiles CQE generators into orbital rotations plus
  diagonal phases, and measures energies in a few rotated bases. This is what
  matters on NISQ hardware.
- **THC:** compresses the Hamiltonian for block encodings (residual
  interferometers, QPE) and for large active spaces. This is what matters on
  fault-tolerant hardware.
- **Shared core:** an exact, noise-free CQE harness. Every factorization is judged
  by its effect on CQE quantities (residual, energy, σ²), not only by integral errors.

## Status: milestone 1 (complete)

| Component | Module | State |
|---|---|---|
| Active-space Hamiltonians (PySCF, FCIDUMP) | `integrals`, `systems` | done |
| Exact CI-space operators, Krylov propagator | `fci_ops` | done |
| RDMs, CSE matrix T, ACSE residual A, σ² identity | `residual` | done |
| Exact steepest-descent CQE | `cqe` | done |
| Cholesky / DF / THC factorizations, λ (OpenFermion conventions) | `factor` | done |
| Exact DF of CQE generators, projected (compressed) generators | `factor.generator` | done |
| CNOT cost model for compiled generators | `diagnostics.cost` | done (estimates) |
| Generator-rank and THC-floor studies | `diagnostics` | done |

Results and conclusions of M1 are in [docs/m1-results.md](docs/m1-results.md).

## Status: milestone 2 (complete)

M2 asks what the CQE costs when the residual and the energies come from
measurements, and replaces steepest descent with better optimizers.

| Component | Module | State |
|---|---|---|
| Jordan–Wigner layer (blocked and interleaved orders), Pauli sums, qubit states | `measure.pauli` | done |
| Residual element sets and measurement targets (aux route Q_e, direct route X_e) | `measure.targets` | done |
| Residual routes: aux exp(−iδH) (exact, DF-Trotter, DF fragments) and direct; shots for a target error | `measure.routes` | done |
| Exact per-shot variances: QWC, DF basis rotation, Pauli and matchgate shadows | `measure.schemes` | done |
| CG and L-BFGS in local coordinates; spin-free or spin-orbital (standard CQE) metric | `optimizers`, `cqe` | done |
| Estimated residuals and energies, fixed / zero / negative step candidates | `cqe` | done |
| Finite-shot noise models (Gaussian, correlated, snapshots) and a shot ledger | `sampling` | done |
| Linearized CQE: stiffness, null modes, symmetry blocks, noise floor, bias offset | `diagnostics.linearized` | done |

Results and conclusions of M2 are in [docs/m2-results.md](docs/m2-results.md).

## Status: milestone 3 (complete)

M3 asks what the CQE costs on a fault-tolerant machine, and what it is worth as a state
preparer for QPE. `thc_engine_cqe.ft` is a resource-estimation layer (Toffolis, logical
qubits, surface-code footprint and wall clock), not a compiler.

| Component | Module | State |
|---|---|---|
| Budgets, phase-gradient rotations, GQSP time evolution, amplitude estimation | `ft.primitives` | done |
| Symmetry-adapted benchmark systems, orbital irreps, point-group masks | `ft.symmetry` | done |
| State preparation of the CQE iterates: elementwise and DF-leaf steps, cumulative costs | `ft.stateprep` | done |
| Validation circuits: phase-gradient excitations, Givens rotations, compiled replay | `ft.circuits` | done |
| Surface-code footprint and wall clock (OpenFermion's model) | `ft.physical` | done |
| Residual and energy measurement: matchgate sampling, per-element amplitude estimation, gradient estimation, line search | `ft.residual`, `ft.residual_inputs` | done |
| DF, THC and SF block encodings and QPE costs; a qubitized-walk emulator | `ft.qpe`, `ft.walk` | done |
| Exact trajectories with overlaps and E₁, the classical second-order first step; the hand-off to QPE; the CQE-only energy | `ft.trajectory`, `ft.handoff` | done |
| Iteration costs, machine schedules, per-system reports, tables T1–T9 | `ft.pipeline` | done |

Results and conclusions of M3 are in [docs/m3-results.md](docs/m3-results.md); the cost
model (parameters, formulas F1–F19, reference numbers) is in
[docs/m3-cost-model.md](docs/m3-cost-model.md).

## Status: milestone 4 (complete)

M4 asks where M3's verdict, that no measured CQE iteration pays as a state preparer for
QPE at equilibrium, changes: on overlap-limited systems (hydrogen chains to n = 10,
stretched N2), with step rules that measure no energies, in a closed-loop finite-shot
fault-tolerant CQE, under an E1-referenced hand-off rule and QPE-side amplification
schemes, and for ground-state properties against QPE.

| Component | Module | State |
|---|---|---|
| Unit-step, line-search, grid and gated 3-point-fit step rules with safeguards | `steprules` | done |
| Fault-tolerant costs of the step rules, the energy-precision wall | `ft.steprules` | done |
| Closed-loop finite-shot CQE: estimators, controllers, replicates, paired verdicts | `ft.closedloop` | done |
| Overlap, gap and cost scaling, break-even sizes, compiled overlaps, stage cache | `ft.scaling` | done |
| Windowed QPE outcome laws, model spectra of H′, the E1-referenced acceptance rule B | `ft.acceptance` | done |
| Sector levels, truncated-CI bounds, contaminant weights, energy certificates | `ft.certify` | done |
| QPE-side schemes: amplitude amplification, step filters, two-stage QPE | `ft.amplify` | done |
| Ground-state properties: the CQE iterate against QPE-projected copies and Hellmann–Feynman QPE | `ft.properties` | done |

In short (costs in units of one QPE, C_QPE; r is the relative error of the measured
residual):

- At equilibrium M3's verdict holds at r ≤ 1 and in the closed loop (the one exception,
  h4's spin-free unit step at r = 3 on a noise-free trajectory, gains 0.0163%).
- Without the line search the cheapest measured iteration, a unit step after the free
  classical step, costs 0.0184–519 C_QPE at r = 1. At the best realizable QPE-side scheme
  it pays on exact trajectories at r ≤ 1 only on H4 at 3.5–4.0 bohr and H6 at 4.0 bohr,
  and in the closed loop's deciding cells only on H4 at 4.0 bohr and, narrowly, on N2 at
  2.8 Å. These pays are fragile: on H4 at 3.5 bohr, the one certified hand-off among them,
  the gain comes mostly from the overshoot of the exponential search; elsewhere they rest
  on hand-offs whose termination is not certified, and at 4.0 bohr the iteration mostly
  repairs an overshooting free step. The second measured iteration pays at the best scheme
  on no exact trajectory and in no deciding cell.
- On N2 at 2.0–2.8 Å the large lever is the QPE side: a step filter with exponential
  search, again without certified termination, cuts the HF hand-off from 3.939–12.8 C_QPE
  (M3's uncertified oracle rule E) to 1.466–1.885. Rule B, given an assumed lower bound on
  E₁, gives every exact hand-off state a realizable acceptance at M3's oracle cost plus
  one C_QPE per verification run (on nine stretched systems without a certificate and with
  termination not certified); on M3's systems it buys soundness, not a cheaper hand-off.
- Beyond the computed sizes (n ≤ 10) the chains' overlap catastrophe outgrows the cost of
  the first measured iteration only from n* = 86 (repetition) or 166 (best scheme) at
  r = 1, at overlaps ≤ 1.5e-5 and outside the sizes at which the kernel-aware form of
  rule B can still certify a 1-mHa hand-off: a model extrapolation.
- For ground-state properties the CQE iterate is the cheapest estimator on most (system,
  target) pairs only with a free loop.

Results and conclusions of M4 are in [docs/m4-results.md](docs/m4-results.md); the cost
model (parameters, formulas F20–F53, reference numbers) is in
[docs/m4-cost-model.md](docs/m4-cost-model.md).

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e ".[test,ft]"
.venv/bin/python -m pytest
```

Requires Python ≥ 3.10, NumPy, SciPy, PySCF and h5py. The `ft` extra installs
OpenFermion. The λ, `df.factorize` and THC-cost cross-checks compare against it,
and M3 takes its QPE costs from OpenFermion's resource estimates (`ft.qpe`), so
`ft.pipeline` and the M3 benchmark need it; tests that need it are skipped
without it. The whole suite takes about 4 minutes on one thread;
`-m "not slow"` skips the slowest checks.

## Quick start

```python
from thc_engine_cqe import systems
from thc_engine_cqe.cqe import run_cqe
from thc_engine_cqe.factor import double_factorize, fit_thc_active, projection_compressor

h4 = systems.hydrogen_chain(4, spacing=1.8, unit="bohr")
e_fci, _ = h4.fci()
res = run_cqe(h4, max_iter=20, e_exact=e_fci)
print(res.errors)                      # E - E_FCI per iteration
print([s.residual_norm for s in res.steps])

# CQE with each generator projected onto its top double-factorized leaves
res4 = run_cqe(h4, max_iter=20, e_exact=e_fci, compress=projection_compressor(n_leaves=4))

# L-BFGS preconditioned with the spin-orbital ACSE generator (the standard CQE metric)
res_lb = run_cqe(h4, max_iter=20, e_exact=e_fci, optimizer="lbfgs", metric="spin_orbital")

# Shots for a 30% error of the full residual tensor: direct route, matchgate shadows
from thc_engine_cqe.measure import routes, targets
el = targets.element_set(h4.norb, "spin_free")
var = routes.direct_residual_variance(h4, res.ci, el, "spin_free", "matchgate_shadow")
a = routes.exact_residual(h4, res.ci, el, "spin_free")
w = targets.element_multiplicities(el, "spin_free", h4.norb)
print(routes.shots_for_relative_error(var.per_target, a, 0.3, weights=w))

# Model Hamiltonians from factorized integrals
thc = fit_thc_active(h4, nthc=8)       # THC rank M = 8
df = double_factorize(h4.eri, eig_threshold=1e-3, first="eigh", tol=0.0,
                      threshold_rule="openfermion")
print(thc.lam(h4.h1, eri_exact=h4.eri).lam_total, df.lam(h4.h1, eri_exact=h4.eri).lam_total)
h4_thc = thc.apply_to(h4)              # ActiveSpace with the THC integrals
```

M3: the fault-tolerant costs of one benchmark system, end to end (needs the `ft` extra):

```python
from thc_engine_cqe.ft import pipeline

rep = pipeline.run_system("h4", optimizers=["lbfgs"])   # about 5 s on one thread
s = pipeline.report_summary(rep)
print(s["qpe_kind"], s["c_qpe"])       # selected (cheapest admissible) QPE encoding, its Toffolis
fam = s["families"]["lbfgs"]
print(fam["k_star"]["exact"])          # first L-BFGS iterate within 1.6 mHa
print(fam["handoff"]["classical_first_step"]["expected"])  # optimal hand-off, in units of C_QPE
print(pipeline.markdown_tables([rep]))                     # tables T1-T9 for this system
```

## Benchmarks

M1:

```bash
.venv/bin/python benchmarks/run_thc_floors.py --systems h4,h6,h8,lih,n2 --out results/
.venv/bin/python benchmarks/run_generator_rank.py --systems h4,h6,h6s,lih,n2,n2s,h8 --threads 1 --out results/
```

M2:

```bash
.venv/bin/python benchmarks/run_optimizers.py --threads 1 --out results/
.venv/bin/python benchmarks/run_linearized.py --threads 1 --out results/
.venv/bin/python benchmarks/run_measurement.py --threads 1 --out results/
.venv/bin/python benchmarks/run_sampled_cqe.py --threads 1 --out results/
```

M3 (needs the `ft` extra):

```bash
.venv/bin/python benchmarks/run_ft_pipeline.py --threads 1 --out results/
```

M4 (needs the `ft` extra; the summary reads the other four outputs and runs last):

```bash
.venv/bin/python benchmarks/run_m4_scaling.py --out results/m4
.venv/bin/python benchmarks/run_m4_loop.py --out results/m4
.venv/bin/python benchmarks/run_m4_handoff.py --m3-results results --out results/m4
.venv/bin/python benchmarks/run_m4_properties.py --out results/m4
.venv/bin/python benchmarks/run_m4_summary.py --out results/m4
```

Each writes JSON and markdown tables to `results/` (not tracked). Approximate
single-thread runtimes:

| Benchmark | Runtime |
|---|---|
| `run_thc_floors.py` | about 20 s |
| `run_generator_rank.py` | about 3 minutes |
| `run_optimizers.py` | about 3 minutes |
| `run_linearized.py` | about 20 s (add `--long-iters 2000` for the long N2 runs) |
| `run_measurement.py` | about 1 hour (H6 alone half of it); `--quick` < 2 minutes |
| `run_sampled_cqe.py` | about 65 minutes; `--quick` < 3 minutes |
| `run_ft_pipeline.py` | about 5–6 minutes; `--systems h4` about 15 s |
| `run_m4_scaling.py` | about 75 minutes |
| `run_m4_loop.py` | about 75 minutes |
| `run_m4_handoff.py` | about 25 minutes |
| `run_m4_properties.py` | about 5 minutes |
| `run_m4_summary.py` | about 6 minutes |

`--threads 1` pins the BLAS and PySCF threads, so the numbers are reproducible.

The M4 scripts write to `results/m4/` and cache their stages in `results/m4/cache`, so an
interrupted run resumes: `run_m4_scaling.py --systems <s> --stage <stage>` computes one
stage per process, and `run_m4_loop.py --resume` reloads the finished systems.
`--m3-results results` reads M3's `results/ft_<system>.json`; without it the hand-off
benchmark recomputes M3's records. Each script has `--quick` for a small system set. The
M4 scripts take no `--threads`: they set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and
MKL_NUM_THREADS to 1 themselves. Their runtimes above were measured on one thread at a
load of about 2.

## Conventions

See [docs/conventions.md](docs/conventions.md). In short, two-body tensors use
PySCF's pair order, so `X[p,q,r,s]` labels
e\_{pq,rs} = Σ a†\_{pσ} a†\_{rτ} a\_{sτ} a\_{qσ}. They carry no factor of ½, and the
descent generator is g = conj(A). Every convention is tested against an
independent Jordan–Wigner implementation.

## License

Apache-2.0 (see `LICENSE`).
