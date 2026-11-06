"""Fault-tolerant cost layer of M3: resource estimates for the CQE and the CQE -> QPE hand-off.

A resource-estimation layer built from validated cost models, not a fault-tolerant
compiler. The unit is the Toffoli (one CCZ state); Cliffords are free. Every module
states its cost model, with references and tags ([OF], [derived], [cited], [estimate],
[assumption], [design], [build check]), in its docstring; the tag legend, the formula
labels F1-F19 and F2a and the whole model are in docs/m3-cost-model.md.

    primitives       FTParameters (budgets and constants), phase-gradient rotations
                     (F1, F2), GQSP time evolution (F8), amplitude estimation (F11)
    symmetry         benchmark systems on symmetry-adapted RHF orbitals, orbital
                     irreps, point-group masks of the residual and the directions (F2a)
    stateprep        state preparation of the CQE iterates: elementwise and DF-leaf
                     steps, orbital rotations, cumulative costs (F3-F7)
    circuits         validation tools: phase-gradient excitation unitaries, kickback,
                     Givens decompositions, compiled elementwise replay
    physical         surface-code footprint and wall clock (F18, F19)
    residual         residual-estimation routes: matchgate sampling, amplitude
                     estimation, gradient estimation, line search (F10-F14)
    residual_inputs  per-element variances, normalizations and biases from the M2 layer
    qpe              OpenFermion DF, THC and SF block encodings and QPE costs (F9)
    walk             a Pauli-LCU qubitized-walk emulator and the QPE outcome law
    trajectory       exact CQE trajectories with states, overlaps and E_1 (F16)
    handoff          the CQE as a QPE state preparer (F15, F17)
    pipeline         iteration costs, per-system reports, tables

The submodules are imported explicitly (``from thc_engine_cqe.ft import primitives``);
nothing is re-exported here. The command-line driver is ``benchmarks/run_ft_pipeline.py``
and the results are in docs/m3-results.md.
"""

__all__ = [
    "primitives",
    "symmetry",
    "stateprep",
    "circuits",
    "physical",
    "residual",
    "residual_inputs",
    "qpe",
    "walk",
    "trajectory",
    "handoff",
    "pipeline",
]
