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

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e ".[test,ft]"
.venv/bin/python -m pytest
```

Requires Python ≥ 3.10, NumPy, SciPy, PySCF and h5py. The `ft` extra installs
OpenFermion, which the λ, `df.factorize` and THC-cost cross-checks compare
against (they are skipped without it); the suite takes about a minute.

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

# Model Hamiltonians from factorized integrals
thc = fit_thc_active(h4, nthc=8)       # THC rank M = 8
df = double_factorize(h4.eri, eig_threshold=1e-3, first="eigh", tol=0.0,
                      threshold_rule="openfermion")
print(thc.lam(h4.h1, eri_exact=h4.eri).lam_total, df.lam(h4.h1, eri_exact=h4.eri).lam_total)
h4_thc = thc.apply_to(h4)              # ActiveSpace with the THC integrals
```

## Benchmarks

```bash
.venv/bin/python benchmarks/run_thc_floors.py --systems h4,h6,h8,lih,n2 --out results/
.venv/bin/python benchmarks/run_generator_rank.py --systems h4,h6,h6s,lih,n2,n2s,h8 --threads 1 --out results/
```

Each writes JSON and markdown tables to `results/` (not tracked). The first takes
about 20 s, the second about 3 minutes, on 4 cores.

## Conventions

See [docs/conventions.md](docs/conventions.md). In short, two-body tensors use
PySCF's pair order, so `X[p,q,r,s]` labels
e_{pq,rs} = Σ a†_{pσ} a†_{rτ} a_{sτ} a_{qσ}. They carry no factor of ½, and the
descent generator is g = conj(A). Every convention is tested against an
independent Jordan–Wigner implementation.

## License

Apache-2.0 (see `LICENSE`).
