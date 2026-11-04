"""CQE studies.

    generator_rank  rank/sparsity of CQE generators along exact trajectories, and
                    compressed CQE (elements vs double-factorized leaves) vs CNOTs
    cost            CNOT cost model for compiled generators (estimates)
    thc_floors      THC/DF rank vs integral error, lambda, and the CQE residual,
                    variance and energy floors the model Hamiltonian induces
    linearized      the CQE linearized at an eigenstate: stiffness modes, null
                    modes the residual cannot see, noise floors and bias offsets

The command-line drivers are ``benchmarks/run_generator_rank.py`` and
``benchmarks/run_thc_floors.py``.
"""

from .generator_rank import compressed_cqe, rank_study
from .linearized import LinearizedCQE, linearize
from .thc_floors import model_floors, thc_floor_scan

__all__ = ["rank_study", "compressed_cqe", "thc_floor_scan", "model_floors", "LinearizedCQE",
           "linearize"]
