"""Measurement layer of M2: the qubit-level description of the CQE.

    pauli    Pauli strings (PauliSum), the Jordan-Wigner map, Majorana monomials,
             sparse qubit states (QubitState)

The conventions are documented in the docstring of :mod:`.pauli`.
"""

from .pauli import (
    DEFAULT_TOL,
    PauliSum,
    QubitMapping,
    QubitState,
    coefficient_matrix,
    jw_hamiltonian,
    jw_ladder,
    jw_one_body,
    jw_product,
    jw_products,
    jw_rdm2_element,
    jw_rdm2_elements,
    jw_two_body,
    majorana_product,
    majorana_to_pauli,
    pauli_commutes,
    pauli_label,
    pauli_product,
    pauli_to_majorana,
    pauli_weight,
    qubitwise_commutes,
)

__all__ = [
    "DEFAULT_TOL",
    "PauliSum",
    "QubitMapping",
    "QubitState",
    "coefficient_matrix",
    "jw_hamiltonian",
    "jw_ladder",
    "jw_one_body",
    "jw_product",
    "jw_products",
    "jw_rdm2_element",
    "jw_rdm2_elements",
    "jw_two_body",
    "majorana_product",
    "majorana_to_pauli",
    "pauli_commutes",
    "pauli_label",
    "pauli_product",
    "pauli_to_majorana",
    "pauli_weight",
    "qubitwise_commutes",
]
