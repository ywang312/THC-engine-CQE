"""Factorizations of electron-repulsion integrals and of CQE generators.

    eri_factor  common interface (to_eri, n_params, lambda) and the one-body T' term
    cholesky    pivoted Cholesky of the (pq|rs) supermatrix
    df          double factorization (orbital rotations + diagonal Coulomb leaves)
    thc         least-squares tensor hypercontraction and its block-encoding 1-norm
    generator   exact double factorization of anti-Hermitian CQE generators

Lambda values follow OpenFermion's ``resource_estimates`` conventions and are
tested against its ``compute_lambda`` functions (see docs/conventions.md).
"""

from .cholesky import cholesky_eri
from .df import DoubleFactorization, double_factorize
from .eri_factor import DenseERI, ERIFactor, LambdaBreakdown, one_body_tprime, pauli_lambda
from .generator import (GeneratorFactorization, factorize_generator, projected_generator,
                        projection_compressor)
from .thc import THCFactorization, fit_thc, fit_thc_active

__all__ = [
    "ERIFactor",
    "DenseERI",
    "LambdaBreakdown",
    "one_body_tprime",
    "pauli_lambda",
    "cholesky_eri",
    "DoubleFactorization",
    "double_factorize",
    "THCFactorization",
    "fit_thc",
    "fit_thc_active",
    "GeneratorFactorization",
    "factorize_generator",
    "projected_generator",
    "projection_compressor",
]
