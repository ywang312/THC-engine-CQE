"""THC engine for the contracted quantum eigensolver (CQE).

Core modules:
    integrals   ActiveSpace: active-space Hamiltonians (PySCF, FCIDUMP)
    systems     small benchmark molecules
    fci_ops     exact operators on CI vectors (H, one-/two-body, exponentials)
    residual    RDMs, CSE matrix T, ACSE residual A, variance identities
    cqe         exact (emulated) CQE driver
    factor      ERI factorizations (Cholesky, DF, THC) and generator factorization
    diagnostics M1 studies (generator rank, THC floors, cost model)

Conventions are documented in docs/conventions.md.
"""

from .integrals import ActiveSpace

__version__ = "0.1.0.dev0"
__all__ = ["ActiveSpace", "__version__"]
