"""Fault-tolerant cost layer: resource estimates for the CQE and the CQE -> QPE hand-off (M3, M4).

A resource-estimation layer built from validated cost models, not a fault-tolerant
compiler. The unit is the Toffoli (one CCZ state); Cliffords are free. Every module
states its cost model, with references and tags ([OF], [derived], [cited], [estimate],
[assumption], [design], [build check]), in its docstring; the tag legend, the formula
labels F1-F19 and F2a and the whole M3 model are in docs/m3-cost-model.md, and the M4
modules continue the numbering with F20-F53.

M3 (the CQE as a QPE state preparer at equilibrium):

    primitives       FTParameters (budgets and constants), phase-gradient rotations
                     (F1, F2), GQSP time evolution (F8), amplitude estimation (F11)
    symmetry         benchmark systems on symmetry-adapted RHF orbitals, orbital
                     irreps, point-group masks of the residual and the directions (F2a);
                     the parametric system names of M4 (F20)
    stateprep        state preparation of the CQE iterates: elementwise and DF-leaf
                     steps, orbital rotations, cumulative costs (F3-F7)
    circuits         validation tools: phase-gradient excitation unitaries, kickback,
                     Givens decompositions, compiled elementwise replay and steps
    physical         surface-code footprint and wall clock (F18, F19)
    residual         residual-estimation routes: matchgate sampling, amplitude
                     estimation, gradient estimation, line search (F10-F14)
    residual_inputs  per-element variances, normalizations and biases from the M2 layer;
                     HF-state and subset variance totals (F24)
    qpe              OpenFermion DF, THC and SF block encodings and QPE costs (F9)
    walk             a Pauli-LCU qubitized-walk emulator and the QPE outcome law
    trajectory       exact CQE trajectories with states, overlaps and E_1 (F16)
    handoff          the CQE as a QPE state preparer (F15, F17); the bracketed rule B (F40)
    pipeline         iteration costs, per-system reports, tables

M4 (overlap-limited systems, cheap step rules, the closed loop, the E1-referenced
hand-off, QPE-side schemes and properties):

    scaling          system sets, spectra, overlap, gap and cost fits, break-even overlap
                     and size, the cheapest measured iteration, compiled overlaps, the
                     per-system scan and its stage cache (F20-F25, F44, F45)
    steprules        costs of the step rules on exact trajectories, the energy-precision
                     wall and the hybrid switch (F27, F28, F30, F31); the rules themselves
                     are ``thc_engine_cqe.steprules`` (F26, F29)
    closedloop       the closed-loop finite-shot CQE: estimators, controllers, exact and
                     compiled propagation, realized costs, replicate statistics and the
                     paired decision rule (F32-F38)
    acceptance       windowed QPE outcome laws, tails, the model spectrum of H' and the
                     rule-B acceptance bracket with verification runs (F39, F40)
    certify          sector levels, variational truncated-CI bounds, contaminant weights
                     and energy certificates (F41, F42)
    amplify          amplitude amplification, step filters, two-stage QPE, scheme costs
                     with their flags, the best realizable scheme and what a better state
                     is worth (F43, F44)
    properties       property targets, bias bounds, estimators, routes A and B,
                     Hellmann-Feynman QPE, the property-aware encoding, the layouts of the
                     property runs and the route comparison over property pairs (F46-F53)

The submodules are imported explicitly (``from thc_engine_cqe.ft import primitives``);
nothing is re-exported here. The command-line drivers are ``benchmarks/run_ft_pipeline.py``
(M3; results in docs/m3-results.md) and ``benchmarks/run_m4_{scaling,loop,handoff,
properties,summary}.py`` (M4).
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
    "scaling",
    "steprules",
    "closedloop",
    "acceptance",
    "certify",
    "amplify",
    "properties",
]
