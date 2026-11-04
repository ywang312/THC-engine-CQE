"""Measurement layer of M2: the qubit-level description of the CQE.

    pauli    Pauli strings (PauliSum), the Jordan-Wigner map, Majorana monomials,
             sparse qubit states (QubitState)
    targets  measurement targets: element sets of the residual, the energy target and
             the auxiliary-state (Q_e) and direct (X_e) residual operators
    routes   residual routes: auxiliary states (exact, DF-Trotter, DF fragments),
             infinite-shot estimates (bias), shot counts

The conventions are documented in the docstring of :mod:`.pauli`; the targets and
routes in those of :mod:`.targets` and :mod:`.routes`.
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
from .routes import (
    METHODS,
    aux_residual,
    aux_residual_variance,
    aux_states,
    df_fragments,
    direct_residual_variance,
    exact_residual,
    shots_for_relative_error,
)
from .targets import (
    KINDS,
    aux_estimates,
    aux_targets,
    contract_spin,
    direct_targets,
    element_multiplicities,
    element_set,
    energy_target,
    gamma_operators,
    spin_free_elements,
    spin_free_tensor,
    spin_orbital_elements,
    spin_orbital_tensor,
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
    # targets
    "KINDS",
    "spin_orbital_elements",
    "spin_free_elements",
    "element_set",
    "element_multiplicities",
    "spin_free_tensor",
    "spin_orbital_tensor",
    "contract_spin",
    "gamma_operators",
    "energy_target",
    "aux_targets",
    "direct_targets",
    "aux_estimates",
    # routes
    "METHODS",
    "df_fragments",
    "aux_states",
    "aux_residual",
    "exact_residual",
    "shots_for_relative_error",
    "aux_residual_variance",
    "direct_residual_variance",
]
