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

## Status: milestone 1 (in progress)

| Component | Module | State |
|---|---|---|
| Active-space Hamiltonians (PySCF, FCIDUMP) | `integrals`, `systems` | done |
| Exact CI-space operators, Krylov propagator | `fci_ops` | done |
| RDMs, CSE matrix T, ACSE residual A, σ² identity | `residual` | done |
| Exact steepest-descent CQE | `cqe` | done |
| Cholesky / DF / THC factorizations, λ | `factor` | in progress |
| Exact DF of CQE generators | `factor.generator` | in progress |
| Generator-rank and THC-floor diagnostics | `diagnostics` | in progress |

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e ".[test]"
.venv/bin/python -m pytest
```

Requires Python ≥ 3.10, NumPy, SciPy, PySCF and h5py.

## Quick start

```python
from thc_engine_cqe import systems
from thc_engine_cqe.cqe import run_cqe

h4 = systems.hydrogen_chain(4, spacing=1.8, unit="bohr")
e_fci, _ = h4.fci()
res = run_cqe(h4, max_iter=20, e_exact=e_fci)
print(res.errors)                      # E - E_FCI per iteration
print([s.residual_norm for s in res.steps])
```

## Conventions

See [docs/conventions.md](docs/conventions.md). In short, two-body tensors use
PySCF's pair order, so `X[p,q,r,s]` labels
e_{pq,rs} = Σ a†_{pσ} a†_{rτ} a_{sτ} a_{qσ}. They carry no factor of ½, and the
descent generator is g = conj(A). Every convention is tested against an
independent Jordan–Wigner implementation.

## License

Apache-2.0 (see `LICENSE`).
