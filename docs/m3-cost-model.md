# M3 cost model

`thc_engine_cqe.ft` estimates what the CQE and the CQE → QPE hand-off cost on a surface-code
machine. It is a resource-estimation layer built from stated and tested cost models, not a
fault-tolerant compiler: it counts the gates of circuits it does not build (`ft.circuits` and
`ft.walk` build small instances explicitly, to check the counting). The unit of cost is the
Toffoli, one CCZ state. An AND gadget costs 1 Toffoli to compute and 0 to uncompute [cited:
Gidney, Quantum 2, 74 (2018)]. Cliffords are free: Jordan–Wigner parities, CNOT basis maps, the
HF determinant (C(Ψ₀) = 0) and the signed-permutation matchgate readout [cited: Zhao, Rubin,
Miyake, PRL 127, 110504 (2021)]. Logical qubits and Toffolis become physical qubits and seconds
through OpenFermion's surface-code model (F18). Hamiltonian factorizations (DF: double
factorization, THC: tensor hypercontraction, SF: single factorization) enter only through block
encodings of the factorized Hamiltonian H′ (F9): QPE, exp(−iδH′) and amplitude estimation. The
direct sampling route, the line-search energies and the state preparation need none: matchgate
shadows, DF basis-rotation grouping of the exact H and elementwise circuits all use the exact
Hamiltonian.

This document is the model as implemented in `thc_engine_cqe.ft`. Every formula names the
function that implements it (`ft.module.function` is short for
`thc_engine_cqe.ft.module.function`); the module docstrings state the same models with their
derivations. The model was designed before the build, and [m3-results.md](m3-results.md),
section 11, lists what changed from that design specification and why. Each change is folded
in below in its corrected reading, followed by
"(changed from the design specification; m3-results.md §11)". The results of the benchmark
(`benchmarks/run_ft_pipeline.py`) are in m3-results.md; its tables T1–T9 are cited by number.

Units: Toffolis; energies, λ, ‖A‖ and biases in Ha; per-shot variances in Ha²; evolution times δ
and t in 1/Ha; wall clock in seconds. Shot and application counts are real numbers, not rounded
up (as in M2); the wall clock rounds them up to whole runs (F19). N is the number of active
(spatial) orbitals, so a state has 2N system qubits; application counts carry subscripts (N_e,
N_A, N_GE, N_E).

Tags, used here, in m3-results.md and in the `ft` docstrings:

- [OF]: OpenFermion's resource-estimate functions or conventions (OpenFermion 1.8.1,
  `openfermion.resource_estimates`);
- [derived]: derived for this package and tested;
- [cited]: from the cited literature;
- [estimate]: a model choice, not a bound;
- [assumption]: a stated assumption;
- [design]: taken from the M3 design notes (not in the repository) and re-checked during the
  build;
- [build check]: verified once during the build by a script that is not part of the repository.

## Parameters

`ft.primitives.FTParameters` is one frozen dataclass. The functions that apply its budgets (in
`ft.stateprep`, `ft.circuits`, `ft.residual`, `ft.residual_inputs`, `ft.qpe`, `ft.handoff` and
`ft.pipeline`) take `params: FTParameters = FTParameters()`; the others need no budget or take
the constants they use as explicit arguments (e.g. `ft.primitives.phase_gradient_bits`,
`ft.qpe.qpe_cost`, `ft.residual.energy_ae`), and `ft.physical` takes a `SurfaceCodeModel` and
explicit failure budgets. Invalid values raise at construction
(tests/test_ft_primitives.py::test_parameters_have_the_spec_defaults_and_are_frozen,
::test_parameters_reject_invalid_values).

| field | default | meaning |
|---|---|---|
| `delta_step` | 1e-3 | operator-norm rounding budget of one compiled CQE step (F2, F3); adds at most one L-BFGS step [design; assumption] |
| `delta_readout` | 1e-5 | rounding budget of one readout basis rotation (F4) [design] |
| `b_min` | 4 | minimum phase-gradient bits (F2); at least 3 |
| `chi`, `beta` | 10, 16 | coefficient and rotation bits of OpenFermion's block encodings [OF: its tests]; `beta` also sets the GQSP rotations (F8) |
| `eps_evol` | 1e-6 | truncation error of exp(−iδH′) by GQSP (F8) [design] |
| `r` | 0.3 | relative RMS Frobenius error of the residual over the full tensor (M2's convention, docs/m2-results.md section 3) |
| `kind` | `"spin_free"` | residual element set (`"spin_free"` or `"spin_orbital"`) |
| `kappa`, `kappa_practical` | 1.0, 2.0 | amplitude-estimation constant: 1 from the sine-window simulation (F11), 2 with a practical margin [estimate] |
| `kappa_g` | (1, 10, 100) | band of the gradient-estimation constant (F12) [assumption] |
| `n_candidates`, `eps_ls` | 11, 1.6e-3 Ha | line search (F13): ε_c = 0.00625·2^j, j = 0…10, i.e. 0.00625 to 6.4 (`FTParameters.eps_candidates`), each energy to ±1.6 mHa: M2's hardware rule (docs/m2-results.md sections 1 and 4) |
| `target` | 1.6e-3 Ha | chemical accuracy; k* is the first iterate with E_k − E₀ ≤ target |
| `de_qpe`, `de_fact`, `infid_tol` | 1e-3 Ha, 5e-4 Ha, 1e-4 | QPE error and the selection rule of the block encoding (F9) [design] |
| `delta_conf` | 0.01 | failure probability of the fixed-confidence hand-off rule (F15) |
| `e_tol_energy`, `sigma_final` | 1e-3 Ha, 3e-4 Ha | CQE-only energy (F17) [design] |
| `p_fail_shot`, `p_fail_ae`, `p_fail_qpe` | 1e-5, 1e-4, 1e-2 | logical failure budget of one sampling shot, one amplitude-estimation run and one QPE run (F18) [design] |
| `aux_deltas` | (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2) 1/Ha | δ grid of the aux routes (F10–F12); the grid of the design notes' scripts, which has no 0.9 (the design specification's "0.2, …, 1.0" would include it) [design] (changed from the design specification; m3-results.md §11) |
| `n_settings`, `seed` | 5000, 7 | Rao–Blackwellized matchgate variances, used when the exact pair sums would exceed 5e7 term pairs (F10) (changed from the design specification; m3-results.md §11) |

The failure budgets follow from the bias a failed run leaves: shots that fail with probability p
bias ⟨O⟩ by at most 2p‖O − c‖, about 2e-2 Ha for the energy at p = 1e-3 [design], hence the shot
budget 1e-5 (about 2e-4 Ha). OpenFermion's budget of 0.1 is reported for comparison (T1).

Surface-code machine, `ft.physical.SurfaceCodeModel` [OF defaults, except where noted]:

| field | default | meaning |
|---|---|---|
| `p_phys` | 1e-3 | physical error rate (OpenFermion's two-level T factory exists only at exactly 1e-3) |
| `cycle_us` | 1.0 µs | surface-code cycle |
| `n_factories` | 4 | magic-state factories F |
| `routing_overhead` | 0.5 | extra logical patches for routing, as a fraction of Q |
| `distances` | 7, 9, …, 43 | odd code distances scanned (OpenFermion: 7…33) |
| `reaction_us` | None | optional classical reaction time of the duration floor (F18) [assumption] |

Other defaults of the code:

- F3: relative tolerance of the spin-orbital expansion 1e-12. F6: rank tolerance of the
  generator factorization 1e-10, active-orbital threshold μ_tol = 1e-6 (relative to max|μ|).
- F9: DF thresholds 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5, 1e-6; THC ranks 4, 6, 8, 10
  (N = 4), 6, 9, 12, 15, 18, 21 (N = 6) and 8, 12, 16, 20, 24, 32 (N = 8), each with the penalties
  0 and "auto" [design]; SF ranks 1 to the rank of the ERI supermatrix; OpenFermion's cost in two
  passes, the first with stps = 20000.
- F10: exact matchgate pair sums up to 5e7 term pairs; w_e evaluated at δ = 0.1; a route is
  infeasible when B² ≤ 1e-12 r²‖A‖_m².
- F11: C_Q = 4 Toffolis for a spin-orbital and 16 for a spin-free Q_e [estimate]. F12: b_θ = 16,
  n_B = 1.
- F13: DF energy splits within 1e-9 (relative) of the lowest tie, and the tie goes to a shifted
  split.
- F15, F16: 12 singlet FCI roots; roots within 1e-6 Ha form one degenerate group; a predicted
  lowering E′(0)²/(2E″(0)) below 1e-10 Ha marks a converged state; finite differences at
  h = 2e-3 and 1e-3.
- F14, F19: iteration costs to k* + 4 (and at least to k_v − 1), per-iterate tables to k* + 2 (to
  k = 8 if k* is not reached); deadline 7 days = 604,800 s (`ft.physical.SECONDS_PER_WEEK`).
- Trajectories (`ft.pipeline.FAMILIES`): `lbfgs` (spin-free L-BFGS, memory 10, 20 steps),
  `lbfgs_so` (the same with the spin-orbital metric) and `sd` (spin-free steepest descent,
  30 steps; core systems only), each with the exact and the classical first step (F14, F16).
  Systems: `ft.symmetry.BENCHMARK_SYSTEMS` (STO-3G; h4, h6a, h6s, h8, lih, n2, n2s).

## Cost model

**F1 (rotation by a classical angle).** [derived]; the adder cost [OF]; the register is
[assumption]. Code: `ft.primitives.addition_toffolis`, `ft.primitives.round_angle`,
`ft.primitives.rounding_error`.

- The b-bit phase-gradient register |Φ_b⟩ = 2^{−b/2} Σ_x e^{−2πix/2^b}|x⟩ is an eigenvector of
  x → x + v mod 2^b with eigenvalue e^{+2πiv/2^b}
  (tests/test_ft_circuits.py::test_kickback_with_explicit_registers, explicit summand and
  gradient registers, b = 3, 4, 5).
- A diagonal phase function that takes the constant φ_c on each of a set of mutually exclusive
  indicator cells costs one addition into the register, b − 2 Toffolis [OF: `cost4df = 4n(β − 2)`
  in `df/compute_cost_df.py`], plus the ANDs of the indicators (counted by the caller). The
  indicators load the summand v_c = round(2^b φ_c/2π) mod 2^b by CNOT fan-out, at no cost.
- Each constant is rounded on its own, so one addition errs by at most π/2^b in operator norm
  (tests/test_ft_primitives.py::test_rounding_error_is_bounded_by_pi_over_2b).
- The unitary is exactly diagonal in the cell basis, so the particle number, S_z and the point
  group are conserved exactly. S² is conserved by one phase function whose spin-mirrored cells
  share their constants, but not by the product of such functions that an elementwise step
  compiles (F3): a spin-free double is a sum of αα, ββ and αβ excitations, which do not commute
  with S² one by one. The compiled RHF step has ⟨S²⟩ = 1.5e-3 (h6s) and 3.9e-3 (n2s), and at
  most 1.5e-5 at equilibrium,
  against 0 for the exact Ψ₁
  (tests/test_ft_circuits.py::test_first_step_product_error_and_spin_contamination)
  (changed from the design specification; m3-results.md §11).
- The register is a catalyst [assumption; OpenFermion's convention]: it is prepared once per
  machine and kept across additions, steps and shots, since an addition leaves it unchanged; its
  top b′ qubits are |Φ_b′⟩, so one register of the largest b serves every smaller b. Its b qubits
  count in the logical qubits of every circuit (Q_prep, F3); its one-time preparation (b
  single-qubit rotations) does not count (changed from the design specification; m3-results.md §11).
- Phase-gradient additions are the only rotation primitive; synthesized rotations are out of
  scope.

**F2 (bits).** [derived]. Code: `ft.primitives.phase_gradient_bits`.

- A circuit of A additions within the operator-norm budget δ (worst-case linear accumulation of
  the per-addition errors) needs b = max(b_min, ⌈log₂(πA/δ)⌉) bits, the smallest b ≥ b_min with
  Aπ/2^b ≤ δ; A = 0 gives b_min.
- b = 15 / 18 / 21 for A = 10 / 64 / 393 at δ = 1e-3
  (tests/test_ft_primitives.py::test_phase_gradient_bits_reference_values, also against a
  brute-force search).

**F2a (symmetry).** [derived]. Code: `ft.symmetry.benchmark_system`,
`ft.symmetry.orbital_irreps`, `ft.symmetry.pair_mask`, `ft.symmetry.element_mask`,
`ft.symmetry.mask_pair_tensor`.

- The orbital irreps come from a symmetry-adapted RHF in an abelian group: D2h for the hydrogen
  chains and N2, C2v for LiH (only D2h and its subgroups are accepted). They are PySCF's irrep
  ids, with which the irrep of a direct product is the XOR of the ids
  (tests/test_ft_symmetry.py::test_xor_rule_is_the_group_product, against the character tables).
  The symmetric builds give the E_HF and E_FCI of plain RHF orbitals of the same geometry to
  1e-10 Ha (tests/test_ft_symmetry.py::test_symmetric_builds_reproduce_the_current_builders).
- An element (p, q, r, s) is allowed iff irrep(p) ⊕ irrep(q) ⊕ irrep(r) ⊕ irrep(s) = 0;
  spin-orbital elements use their orbital indices. The rule is exact: H is totally symmetric and
  the iterates keep the totally symmetric irrep of the closed-shell HF determinant, so every
  forbidden element vanishes at every iterate
  (tests/test_ft_symmetry.py::test_forbidden_residual_elements_vanish_along_sd: ≤ 1e-10 max|A|
  along 5 SD steps). The allowed counts are in Appendix A.
- The mask is used three times. The residual is measured on the allowed elements only (F10–F12;
  `ft.residual_inputs.allowed_elements`). Every direction is masked before it is compiled (F3,
  F6). And the exact trajectories project the residual on the allowed entries before every step
  of the driver,
  `run_cqe(..., residual_estimator=lambda rd, ci, it: mask_pair_tensor(rd.A, irreps))`
  (`ft.trajectory.exact_trajectory`), which keeps the iterates symmetry-pure. Exact
  L-BFGS amplifies the round-off of the forbidden entries (10–150× per step on N2, up to max|A| at
  step 11), and without the projection the weight outside the HF irrep reaches 1.8e-7 (n2) and
  1.9e-4 (n2s) in 20 steps; n2s then ends at 7.31 instead of 5.71 mHa (`ft.symmetry`,
  `ft.trajectory`). With the projection the 20 n2s iterates stay in the HF irrep to 1e-12, without
  it they do not (tests/test_ft_trajectory.py::test_symmetry_projection_removes_irrep_leakage_on_n2s,
  tests/test_ft_symmetry.py::test_masked_residual_keeps_long_lbfgs_runs_pure)
  (changed from the design specification; m3-results.md §11).

**F3 (elementwise step, default).** [derived]. Code: `ft.stateprep.elementwise_step_cost`
(`ft.stateprep.elementwise_excitations`, `ft.stateprep.candidate_step_costs`,
`ft.stateprep.excitation_counts`).

For step j with the real direction d_j (a pair-order tensor; the step is exp(ε_j D_j),
D_j = Σ d_j e_{pq,rs}) and the step size ε_j:

1. X = `spin_orbital_excitations(ε_j·mask(d_j), tol=1e-12)` (`diagnostics.cost`), the exact
   spin-orbital expansion of the masked generator into excitations t_x τ_x − h.c., with
   τ_x = a†_{c0}a†_{c1}a_{d1}a_{d0} a normal-ordered two-body operator (a double, a
   number-conditioned single or a diagonal) and t_x its amplitude. The relative tolerance commutes
   with the scaling by ε, so d_j is expanded once for every step size.
2. θ_x = |ε_j t_x|, the Givens angle of exp(ε_j(t_x τ_x − h.c.)).
3. Fixed point: A = #kept, b = F2(A, δ_step), drop every x with θ_x < π/2^b (it rounds to the
   zero phase), repeat. The kept set is the prefix of X in decreasing angle, and the iteration
   ends at the largest b with b = F2(#{x : θ_x ≥ π/2^b}).
4. Count the kept doubles n_d, number-conditioned singles n_s and diagonals n_g
   (A = n_d + n_s + n_g).

Then C_step = A(b − 2) + 3n_d + 2n_s + n_g Toffolis on Q_prep = 2N + 3b + 8 logical qubits.

- Per excitation: a double costs 2 ANDs for its 3-input control, 1 AND for the cell c ∧ t and one
  addition; a single 1 + 1 ANDs and one addition; a diagonal 1 AND and one addition. The counts
  follow from the cell structure of M1's verified excitation circuits: R_y(φ) on the target t is
  S H R_z(φ) H S†, a phase ±φ/2 on the cells c ∧ t and c ∧ ¬t = c ⊕ (c ∧ t). Built explicitly,
  the phase-gradient circuit equals M1's excitation unitary to 1e-12 unrounded and is within
  π/2^b of it rounded, for 20 random doubles and 20 singles on 8 qubits
  (tests/test_ft_circuits.py::test_pg_excitation_unitaries_match_verified_circuits). Diagonals
  occur only for complex generators.
- Q_prep = 2N + b + b + (b − 1) + 9 counts the 2N system qubits, the gradient register (b), the
  summand (b), b − 1 adder carries and 9 AND ancillas [derived]. A step without kept excitations
  costs 0 on 2N qubits.
- The step also records n_rot = 8n_d + 4n_s + 3n_g Pauli rotations, the Jordan–Wigner terms of
  its generators, for partially fault-tolerant models (`PrimitiveCounts.n_pauli_rot`) [design];
  they enter no Toffoli count.
- Complex directions raise ValueError: a complex amplitude adds two Z rotations per excitation,
  and complex states are out of scope.
- Rounding. A kept excitation is realized with its angle rounded to a multiple of 2π/2^b, within
  π/2^b in operator norm; a dropped one errs by θ_x. The compiled step is therefore within
  `StepCost.rounding_bound` = Aπ/2^b + Σ_dropped θ_x of the product of the exact exponentials of
  X, and F3 budgets only the first term. For small ε the dropped angles push the bound above
  δ_step, and the flag "rounding bound above delta_step" marks such steps and line-search
  candidates: the h6a L-BFGS step 3 at ε = 0.00625 compiles to the identity with a bound of
  1.73e-3 (tests/test_ft_stateprep.py::test_candidate_step_costs_give_qubits_and_flags), and
  4–31 candidates per system are flagged along the measured loops (T9)
  (changed from the design specification; m3-results.md §11).
- Product-formula error. The product of the A exponentials differs from exp(εD) at second order
  in ε. F3 does not budget it (flag "product-formula error not in budget" for A ≥ 2), and the
  product breaks S² (F1). The open-loop replay `ft.circuits.replay_elementwise` measures it:
  ‖Ψ̃₁ − Ψ₁‖ after the RHF step is 2.2e-4 (lih) to 2.0e-3 (n2) at equilibrium, 2.7e-2 (h6s) and
  3.1e-2 (n2s) on the stretched builds
  (tests/test_ft_circuits.py::test_first_step_product_error_and_spin_contamination).
- The counts equal a literal implementation of steps 1–4 over 12 L-BFGS steps of h4, h6a, n2 and
  lih, masked and unmasked
  (tests/test_ft_stateprep.py::test_elementwise_cost_equals_literal_f3_along_trajectories), and
  Appendix A's (tests/test_ft_stateprep.py::test_appendix_a_elementwise_regressions).

**F4 (orbital rotation).** The rotation count [cited]; the cells [derived]; the readout layout
[estimate]. Code: `ft.stateprep.orbital_rotation_counts`, `ft.stateprep.readout_rotation_toffolis`,
`ft.stateprep.readout_rotation_qubits`.

- An N × N unitary is g = N(N − 1)/2 Givens rotations per spin plus diagonal phases [cited:
  Clements et al., Optica 3, 1460 (2016); Kivlichan et al., PRL 120, 110501 (2018)];
  tests/test_ft_circuits.py::test_givens_decomposition_reconstructs checks the count.
- A real Givens rotation acts on both spins with one angle. After a Clifford basis change each
  copy is exp(iθ s_σ), s_σ ∈ {−1, 0, 1}, so the eight one-hot cells of (s_α, s_β) share one
  addition and cost 6 ANDs: g additions + 6g ANDs (4g Pauli rotations). A complex rotation adds
  the phase exp(iφ(n_qα + n_qβ)), one addition and 1 AND: 2g additions + 7g ANDs (6g Pauli
  rotations). The final diagonal phases merge into an adjacent layer.
- The DF readout rotation of an energy measurement is real and has its own budget:
  C_rot = g(b_r − 2) + 6g with b_r = F2(g, δ_readout), on Q_rot = 2N + 3b_r + 8 logical qubits,
  the layout of Q_prep at the readout's bits [estimate: a Givens cell needs at most the ANDs of a
  double]. N = 4, 6, 8: b_r = 21, 23, 24; C_rot = 150, 405, 784; Q_rot = 79, 89, 96
  (tests/test_ft_stateprep.py::test_readout_rotation_toffolis, ::test_readout_rotation_qubits).
  Its rounding error is an energy bias of about 2‖F_l²‖δ_readout, F_l the one-body operator of
  DF leaf l [design], hence the tighter budget.

**F5 (diagonal Coulomb layer).** [design]. Code: `ft.stateprep.diagonal_layer_counts`.

- exp(iτ(Σ_p μ_p n_p)²), n_p = n_pα + n_pβ, on the n_a orbitals with |μ_p| > μ_tol·max|μ|: the
  cells (n_p, n_q) ∈ {1, 2}² of every pair (4 ANDs, one addition) and n_p ∈ {1, 2} of every site
  (1 AND, one addition, which also absorbs the phases of the adjacent rotation):
  n_a(n_a − 1)/2 + n_a additions, 2n_a(n_a − 1) + n_a ANDs, n_a(2n_a + 1) Pauli rotations
  (tests/test_ft_stateprep.py::test_primitive_count_rules).

**F6 (leaf step, order 1).** [design]; the final rotation's budget [assumption]. Code:
`ft.stateprep.leaf_step_cost`, `ft.stateprep.prep_cost`.

- `factor.generator.factorize_generator(mask(d), rank_tol=1e-10)` writes
  D = Σ_k c_k iQ̂_k² + Ô_F with Q̂_k = U(V_k) D_k U(V_k)†. The order-1 product (the one-body term
  first, then the leaves in decreasing |c_k|) merges adjacent rotations, so a step with L leaves
  has L complex rotations and L layers, and a circuit one final complex rotation (the circuit
  structure of `diagnostics.cost`, M1). With n_a,k = `diagnostics.cost.leaf_active_orbitals`:

  - A_leaf = L·N(N − 1) + Σ_k n_a,k(n_a,k + 1)/2 additions;
  - AND_leaf = 7L·N(N − 1)/2 + Σ_k n_a,k(2n_a,k − 1);
  - C_step = A_leaf(b − 2) + AND_leaf with b = F2(A_leaf, δ_step), on Q_prep = 2N + 3b + 8.

- Every leaf step carries the flags "complex state" (the leaf product of a real generator makes
  the state complex at O(ε²), and the exact trajectory it is costed on is not propagated through
  it) and "order-1 Trotter error not in budget" (about 1e-2 in state norm at the RHF step
  [design]).
- The final complex rotation has its own budget, b_f = F2(N(N − 1), δ_step):
  N(N − 1)(b_f − 2) + 7N(N − 1)/2 Toffolis, 210 at N = 4 (b_f = 16) and 555 at N = 6 (b_f = 17)
  (the 555: tests/test_ft_stateprep.py::test_prep_cost_cumulative_sums_and_qubits; both are
  C(Ψ₁) − C_step(0) of the leaf columns of T2, e.g. 3938 − 3728 for h4). The design left b_f
  open; the largest b of the trajectory would make C(Ψ_k) depend on later steps
  (changed from the design specification; m3-results.md §11).
- The RHF step has 2 rank(W) leaves (M1), and the cost equals F6 evaluated from the generator
  leaves (tests/test_ft_stateprep.py::test_leaf_step_cost_matches_f6_and_the_rhf_rank).

**F7 (cumulative).** [derived]. Code: `ft.stateprep.prep_cost`.

- C(Ψ₀) = 0 (HF is Clifford), C(Ψ_k) = Σ_{j<k} C_step(j); a leaf circuit adds its final rotation
  (F6) for every k ≥ 1 at which a leaf was applied.
- Q(Ψ_k) = max(2N, max_{j<k} Q_prep(j)), with 2N + 3b_f + 8 for the final leaf rotation.
- With irreps every direction is masked before it is compiled, for both methods (F2a).
- Tests: tests/test_ft_stateprep.py::test_prep_cost_cumulative_sums_and_qubits; the L-BFGS
  values of Appendix A (C(Ψ₃) = 2506 for h4, C(Ψ₆) = 43,427 on 83 qubits for h6a).

**F8 (GQSP time evolution).** [derived]; the GQSP theorem [cited: Motlagh and Wiebe, PRX
Quantum 5, 020368 (2024)]; the rotation count [estimate]. Code:
`ft.primitives.jacobi_anger_degree`, `ft.primitives.gqsp_walk_calls`,
`ft.primitives.gqsp_rotation_toffolis`, `ft.primitives.evolution_toffolis`,
`ft.qpe.evolution_cost`.

- W is a qubitized walk with normalization λ, eigenphases ±arccos((E − c)/λ) and cost C_W per
  controlled step (OpenFermion's step cost includes the control [OF]). With x = (E − c)/λ = cos θ
  and z = e^{iθ}, e^{−iτx} = Σ_k (−i)^k J_k(τ) z^k, τ = λ|t|.
- The truncation to |k| ≤ R takes the same value on both walk eigenvectors of an invariant
  subspace, so it acts as e^{−iτx} times the identity there, up to the tail 2Σ_{j>R}|J_j(τ)|.
  GQSP realizes a degree-d polynomial in W with d calls to controlled-W; R of them replaced by
  controlled-W† give W^{−R}P(W) with deg P = 2R, the Laurent polynomial. The mixed parity of
  e^{−iτx} rules out an R + 1 call variant.
- Cost: R(τ, ε) = min{R ≥ 0 : 2Σ_{j>R}|J_j(τ)| ≤ ε} with ε = eps_evol; 2R controlled walk calls;
  C_evol = 2R·C_W + 3(2R + 1)(β − 2) Toffolis, three β-bit phase rotations per GQSP layer; on
  Q_evol = Q_walk + 1 logical qubits (the GQSP control).
- The rotation term is a conservative model choice [estimate]: in the Motlagh–Wiebe
  parametrization a degree-d polynomial has 2d + 3 angles, d fewer than 3(d + 1), so it
  overcounts by d(β − 2), about (β − 2)/C_W ≈ 1% of C_evol: 476 of 45,976 Toffolis for H6
  (λ = 9.726 Ha, C_W = 1309, t = 0.7 Ha⁻¹, R = 17; `ft.primitives`)
  (changed from the design specification; m3-results.md §11).
- R = 0, e^{−iτx} within ε of the identity, costs nothing, not one layer of 3(β − 2) = 42
  Toffolis (changed from the design specification; m3-results.md §11).
- Tests: tests/test_ft_primitives.py::test_gqsp_and_evolution_costs (45,976 and R = 0),
  ::test_truncated_jacobi_anger_series_meets_the_budget,
  ::test_jacobi_anger_reference_degrees. The 2R-call Laurent polynomial on an explicit Pauli-LCU
  walk of H4 stays within the tail bound, with no amplitude outside |G⟩ beyond round-off
  (tests/test_ft_walk.py::test_laurent_evolution_within_the_jacobi_anger_tail); this validates the
  polynomial that GQSP implements, not GQSP's phase finding.

**F9 (QPE).** [OF; cited: Babbush et al., PRX 8, 041015 (2018); Berry et al., Quantum 3, 208
(2019); von Burg et al., PRR 3, 033055 (2021); Lee et al., PRX Quantum 2, 030305 (2021)]; the QR
clamp [derived]; the selection rule [design]. Code: `ft.qpe.qpe_cost`, `ft.qpe.select_qpe`,
`ft.qpe.scan_encodings`, `ft.qpe.openfermion_cost`, `ft.qpe.openfermion_qr_clamp`.

- Phase estimation to ΔE = de_qpe takes n_iter = ⌈πλ/(2ΔE)⌉ walk steps, C_QPE = C_W·n_iter
  Toffolis: a rule for the error scale, not a confidence interval.
- Block encodings (`ft.qpe.df_encoding`, `ft.qpe.thc_encoding`, `ft.qpe.sf_encoding`): DF by
  `factor.double_factorize(eri, first="eigh", tol=0, eig_threshold=thr,
  threshold_rule="openfermion")`, which reproduces OpenFermion's `df.factorize` (L leaves, Ξ
  eigenvectors); THC by `factor.fit_thc_active(active, M, penalty=0 or "auto")` (grid guess,
  L-BFGS-B refinement; "auto" is OpenFermion's λ-penalty rule); SF by the factors
  L_P = √w_P v_P of the first L eigenpairs (w_P, v_P) of the ERI supermatrix, in decreasing
  eigenvalue (`factor.df.eigendecomposition_vectors`, OpenFermion's `utils.eigendecomp`).
- C_W is OpenFermion's `compute_cost` of the kind, in two passes as in its tests (stps = 20000,
  then the first pass's step cost), with χ = 10 and β = 16 (SF has no β); it is the cost of one
  controlled walk step.
- λ follows the repository conventions (docs/conventions.md), with the one-body operator
  T′_pq = h_pq − ½Σ_r(pr|rq) + Σ_r(rr|pq) from the exact ERIs (`factor.eri_factor.one_body_tprime`):
  DF λ = Σ|eig(T′)| + ¼Σ_l(Σ_k|μ_lk|)², μ_lk the eigenvalues of DF leaf l; THC λ is OpenFermion's
  `thc.compute_lambda`; SF
  λ = Σ_pq|T′_pq| + ¼Σ_P(Σ_pq|L_P[p,q]|)² (`ft.qpe.sf_lambda`, OpenFermion's `sf.compute_lambda`).
- Q_QPE is OpenFermion's `ancilla_cost`; Q_walk (`BlockEncoding.ancilla_walk`, the walk of F8 and
  F10–F13) is Q_QPE without the phase-estimation control register of 2⌈log₂(n_iter + 1)⌉ − 1
  qubits (DF, THC) or 2⌈log₂ n_iter⌉ − 1 (SF).
- QR clamp. OpenFermion's `utils.QR(L, M)` minimizes the QROM cost L/2^k + M(2^k − 1) over the
  integers next to k_QR = ½log₂(L/M) and calls `sys.exit` when L < M, which happens for every
  small active space here (DF's output width nβ/2 = 64 exceeds Ξ + n/2 = 31 for H4, with n = 2N
  spin orbitals). For L < M the cost
  increases in k ≥ 0, so the integer optimum is k = 0 with cost ⌈L⌉ (plain unary iteration). The
  clamp substitutes this rule inside a context manager, calls OpenFermion's `QR` for L ≥ M, and
  records a flag (`BlockEncoding.qr_clamped`).
- Each encoding carries its factorization error, de_fact = E₀(H′) − E₀(H), and the ground-state
  infidelity 1 − |⟨Ψ₀(H′)|Ψ₀(H)⟩|² (`diagnostics.thc_floors.model_floors`).
- Selection rule. Among the scanned encodings with |de_fact| ≤ 0.5 mHa and infidelity ≤ 1e-4,
  take the one with the fewest QPE Toffolis at ΔE = 1 mHa, per kind and overall; ties go to fewer
  logical qubits, then to the scan order. The total error is then ≤ 1.5 mHa.
- The QPE of the hand-off (F15, F17) and the walk of the residual routes (F10–F13) are the
  cheapest admissible encoding overall, not the DF one: SF on h4, n2, lih and n2s, THC on h6a,
  h6s and h8 (T1). On every benchmark system it also has the lowest λ·C_W of the three per-kind
  selections, and the DF selection's C_QPE is 1.26–1.76× that of the selected encoding
  (changed from the design specification; m3-results.md §11). So the DF choice of h6a in
  Appendix A (threshold 3e-3, L = 11, Ξ = 63) is the choice of `select_qpe(..., kind="df")`;
  overall THC with M = 15 wins, with 1.48e7 Toffolis against 1.82e7 (SF) and 2.00e7 (DF)
  (changed from the design specification; m3-results.md §11).
- Tests: tests/test_ft_qpe.py::test_openfermion_reference_tuples (OpenFermion's six regression
  tuples), ::test_qpe_cost_matches_openfermion_at_several_errors,
  ::test_qr_clamp_restores_and_flags, ::test_qr_clamp_value_is_the_integer_optimum,
  ::test_selection_rule_brute_force, ::test_control_register_at_powers_of_two,
  ::test_sf_lambda_matches_openfermion_on_random_factors.

**F10 (sampling routes).** [derived; equals M2's `routes.shots_for_relative_error`]; the aux
route's qubits [estimate]. Code: `ft.residual.variance_budget`, `ft.residual.sampling_direct`,
`ft.residual.sampling_aux`; inputs from `ft.residual_inputs`.

- Element e of the allowed spin-free set (F2a) stands for m_e entries of the full tensor
  (`targets.element_multiplicities`), and ‖x‖_m² = Σ_e m_e x_e² is the full-tensor Frobenius
  norm (tests/test_ft_residual_inputs.py::test_norm_on_the_allowed_set_is_the_full_tensor_norm).
  An estimate with bias b and variances Var_e meets the relative RMS target r iff
  ‖b‖_m² + Σ_e m_e Var_e ≤ r²‖A‖_m². The variance budget is
  B² = r²‖A‖_m² − (‖b_route‖_m + ‖b_fact‖_m)², the two biases combined by the triangle
  inequality. A route with B² ≤ 1e-12 r²‖A‖_m² is infeasible (M2's round-off rule): flag
  "infeasible", infinite cost.
- Direct route, `direct_mg`: X_e = [Γ_e − Γ_e†, H]/2 on Ψ_k, read out by signed-permutation
  matchgate shadows (Clifford); unbiased, with the exact H. S = Σ_e m_e v_e/B² shots,
  T = S·C(Ψ_k) Toffolis, Q = Q(Ψ_k).
- Aux route, `aux_mg`: −⟨Q_e⟩/δ on e^{−iδH′}Ψ_k, the evolution by GQSP on the walk of the
  selected encoding (F8, F9). The per-shot variance is w_e/δ², with w_e independent of δ to O(δ²)
  (M2). The bias is the exact O(δ²) aux bias ‖b(δ)‖_m on the δ grid, interpolated log–log
  (`ft.residual_inputs.aux_bias_curve`), plus the factorization bias ‖A[H′] − A[H]‖_m at Ψ_k
  (`ft.residual_inputs.factorization_bias`). S(δ) = Σ_e m_e w_e/(δ²B(δ)²),
  T(δ) = S(δ)·(C(Ψ_k) + C_evol(δ)), minimized over `aux_deltas`;
  Q = Q(Ψ_k) + Q_walk + 1 − 2N [estimate], the evolution sharing the 2N system qubits.
- Per-shot variances (`ft.residual_inputs.element_variances`, from M2's `measure` layer): v_e is
  `routes.direct_residual_variance` under matchgate shadows, by exact pair sums when the pair
  count Σ_e L_e(L_e + 1)/2 (L_e = L_X,e of F11, the number of non-identity Pauli terms of X_e)
  is at most 5e7 and Rao–Blackwellized over 5000 matchings (seed 7)
  above: exact for H4 (7.6e6 pairs) and N2 (2.8e7), RB for LiH (3.5e8) and H6 (2.2e9). The design
  had exact sums for N ≤ 4 only, which would give N2 a seed-dependent RB estimate (3730 ± 50 Ha²
  at seed 7 against the exact 3751.1 Ha² at SD iterate 1, `ft.residual_inputs`)
  (tests/test_ft_residual_inputs.py::test_n2_variances_use_exact_pair_sums)
  (changed from the design specification; m3-results.md §11).
  w_e = δ²·`routes.aux_residual_variance` at δ = 0.1. Both are computed once per system, at
  iterate 1 of the first family, and reused for every iterate: on H4, v_e at SD iterates 1 and 4
  agree within 3% element by element
  (tests/test_ft_residual_inputs.py::test_direct_variances_are_state_independent, slow).
- The qubit counts take N from `ResidualProblem.norb`, the active orbitals, whose 2N system
  qubits the state and the block encodings share
  (changed from the design specification; m3-results.md §11).
- Tests: tests/test_ft_residual.py::test_sampling_direct_equals_m2_shot_rule,
  ::test_sampling_aux_minimizes_over_the_delta_grid,
  ::test_infeasible_bias_gives_the_flag_and_infinite_cost.

**F11 (per-element amplitude estimation).** [derived; κ simulated]; canonical AE [cited:
Brassard, Høyer, Mosca, Tapp, Contemp. Math. 305, 53 (2002)]; C_Q, C_obs and Q_obs [estimate].
Code: `ft.residual.ae_allocation`, `ft.residual.ae_product`, `ft.residual.ae_x_lcu`,
`ft.residual.ae_aux`, `ft.primitives.ae_applications`.

- A Hadamard-test circuit 𝒜 on Q_A,e = Q(Ψ_k) + Q_obs,e + 1 logical qubits (the last one the
  Hadamard-test qubit) has the success probability p = (1 + ⟨O_e⟩/α_e)/2 and is iterated with
  G = 𝒜S₀𝒜†S_good, whose reflections alone are controlled. G has the eigenphases ±2θ with
  sin²θ = p. Sine-window phase estimation with M_G applications of G estimates 2θ to an RMS error
  κπ/M_G with κ → 1 (0.974, 0.993, 0.998 for M_G = 63, 255, 1023,
  tests/test_ft_residual.py::test_ae_constant_from_sine_window_phase_estimation), so
  σ(⟨O_e⟩) ≤ κπα_e/M_G and N_e = 2πκα_e/ε_e applications of 𝒜 or 𝒜† for an RMS error ε_e.
- Allocation. Minimizing Σ_e N_e under Σ_e m_e ε_e² = B² gives ε_e = t(α_e/m_e)^{1/3},
  t = B/(Σ_e m_e^{1/3}α_e^{2/3})^{1/2}, and N_A = Σ_e N_e = 2πκΓ/B with
  Γ = (Σ_e m_e^{1/3} α_e^{2/3})^{3/2}; an element with α_e = 0 needs no applications
  (tests/test_ft_residual.py::test_ae_allocation_matches_slsqp). B is that of F10, with the
  route's bias.
- Element e is one coherent run of N_e applications on Q_run,e = Q_A,e + ⌈log₂(N_e/2)⌉ logical
  qubits (the phase register; none for N_e ≤ 2). An application costs C(Ψ_k) + C_obs,e + Q_A,e/2;
  the last term is the controlled reflection S₀ about |0⟩ on the qubits of 𝒜, a multi-controlled
  Z of about Q_A,e Toffolis once per G, i.e. per two applications [estimate]. So
  T = Σ_e N_e(C(Ψ_k) + C_obs,e + Q_A,e/2); its summands are the runs (`RouteCost.run_toffolis`),
  the longest coherent run is the largest summand, and the route's qubits are the largest
  Q_run,e. The reflection acts on 𝒜's qubits, not on the phase register, and it counts in T, in
  the longest run and in the wall clock (F19). On the benchmark records
  (`results/ft_<system>.json`, `iterations`) it is at most 14% of an application (ae_product, h4
  at HF on the SF walk, C_W = 384: 67 of 467 Toffolis; at most 1.1% for ae_aux), and counting the
  phase register in it would raise T by at most 1.25% at κ = 1 and 1.35% at κ = 2 (the same
  iteration); with the DF walk of the `ft.residual` example the two shares are 9.6% and 0.7%
  (changed from the design specification; m3-results.md §11).
- The three observables, with C_Q = 4 Toffolis for a spin-orbital and 16 for a spin-free Q_e:

  | route | α_e | C_obs,e | Q_obs,e | bias |
  |---|---|---|---|---|
  | `ae_product` (b1) | 2λ_Q,e·λ | C_W + C_Q | Q_walk − 2N + 4 | factorization |
  | `ae_x` (b2; X-LCU, the Pauli LCU of X_e) | α_X,e | 2L_X,e (QROM PREPARE + unary SELECT) | ⌈log₂ L_X,e⌉ + χ + 4 | none (exact H) |
  | `ae_aux` (b3) | λ_Q,e/δ | C_evol(δ) + C_Q | Q_walk + 1 − 2N + 4 | ‖b(δ)‖_m + factorization |

  `ae_aux` takes the δ of `aux_deltas` with the fewest Toffolis. λ_Q,e and α_X,e are the Pauli
  1-norms, identity excluded, of Q_e = (Γ_e − Γ_e†)/(2i) and X_e = i[Q_e, H], and L_X,e is the
  number of non-identity Pauli terms of X_e (`ft.residual_inputs.element_normalizations`); every
  spin-orbital Q_e has λ_Q = ½ [derived]. For b1 the Hadamard test on U_Q U_H gives
  Im⟨Q_e H⟩/(λ_Q λ), and A_e = −2 Im⟨Q_e H⟩ for real Ψ [derived]. Dense Pauli-LCU block encodings
  of H2 reproduce A_e through the Hadamard tests of b1, b2 and b3 to 1e-10
  (tests/test_ft_residual_inputs.py::test_dense_block_encodings_reproduce_the_residual).
- Every AE route is costed at κ = 1 and at κ_practical = 2 (`ft.residual.compare_routes`).
- Tests: tests/test_ft_residual.py::test_ae_routes_follow_f11,
  ::test_kappa_scales_the_applications, ::test_element_with_zero_alpha_needs_no_applications.

**F12 (gradient-estimation band).** [assumption; asymptotics cited: Gilyén, Arunachalam, Wiebe,
SODA 2019 (arXiv:1711.00465); Huggins et al., PRL 129, 240501 (2022)]; the qubits [estimate].
Code: `ft.residual.gradient_estimation` (route `gaw`, "GAW" in T3, after Gilyén, Arunachalam and
Wiebe).

- Formulation "aux": Huggins et al. on {Q_e} measured on e^{−iδH′}Ψ_k, α_e = λ_Q,e/δ, for K
  elements in n_B equal blocks:
  N_GE = 2πκ_g √K √n_B (Σ_e m_e α_e²)^{1/2}/B preparations and
  T = N_GE(C(Ψ_k) + C_evol(δ) + C_probe/n_B), with C_probe = K(2b_θ + 2) and B from the bias
  ‖b(δ)‖_m + factorization; minimized over `aux_deltas`. Each block is one coherent run of
  N_GE/n_B preparations, on
  Q = Q(Ψ_k) + Q_walk + 1 − 2N + ⌈K/n_B⌉b_θ logical qubits (b_θ = 16 bits per probe angle, at
  least 3; n_B = 1 by default).
- C_obs is C_evol(δ) alone: the Q_e enter only through the probe Π_e exp(−ix_e Q_e/λ_Q,e), whose
  cost C_probe contains them [design], so the C_Q of b3 has no counterpart here; with it a query
  would cost 16 Toffolis more, at most 0.17% on the benchmark (h4 at HF, δ = 0.4: 9514 Toffolis
  per query; `results/ft_h4.json`)
  (changed from the design specification; m3-results.md §11).
- κ_g is unknown: the route is reported for κ_g ∈ {1, 10, 100} and never selected
  (`ft.residual.best_route` excludes it).
- Test: tests/test_ft_residual.py::test_gradient_estimation_follows_f12.

**F13 (line search).** [derived]; DF basis-rotation grouping [cited: Huggins et al., npj Quantum
Inf. 7, 23 (2021)]; the AE option [estimate]. Code: `ft.residual.line_search_cost`,
`ft.stateprep.candidate_step_costs`, `ft.residual_inputs.energy_variances`,
`ft.residual.energy_ae`; assembled per iteration by `ft.pipeline.iteration_costs`.

- The n_c = 11 candidates Ψ_{k+1}(ε_c), ε_c = 0.00625·2^j along d_k, are compiled elementwise one
  by one (F3) and measured by DF basis-rotation grouping to eps_ls = 1.6 mHa: S_E = V_E/eps_ls²
  shots per candidate, each of C(Ψ_k) + C_step(k; ε_c) + C_rot Toffolis, so
  T_LS = S_E Σ_c (C(Ψ_k) + C_step(k; ε_c) + C_rot). The candidate step costs are not stored:
  `ft.stateprep.candidate_step_costs(d_k, FTParameters().eps_candidates, irreps=...)` gives them,
  and the JSON keeps their mean shot cost, C(Ψ_k) + C_rot + Σ_c C_step(k; ε_c)/n_c
  (`line_search.toffolis_per_prep`).
- The shots run on max(Q(Ψ_k), max_c Q_prep(ε_c), Q_rot) logical qubits, since a candidate step
  or the readout rotation can need more than Ψ_k: 8 qubits for the h4 HF state against 61 for
  its candidate steps (`ft.residual.line_search_cost`); Q_rot = 89 against Q(Ψ₆) = 83 for h6a
  (`ft.stateprep.readout_rotation_qubits`)
  (changed from the design specification; m3-results.md §11). On every benchmark system the
  readout rotation sets them: Q_rot = 79 (N = 4), 89 (N = 6) and 96 (N = 8) lies above every
  Q(Ψ_k) (at most 70, 83 and 93) and every candidate step, so every costed line search runs on
  Q_rot qubits (`results/ft_<system>.json`, `line_search.logical_qubits`).
- V_E is the per-shot variance of DF basis-rotation grouping (`schemes.df_variance`) in one of
  M2's four splits: leaves from pivoted Cholesky or from the supermatrix eigendecomposition, each
  unshifted or shifted to E_pq − δ_pq. The split with the lowest V_E at HF is kept along the
  trajectory (tests/test_ft_pipeline.py::test_energy_split_and_variances_follow_the_rules, slow);
  splits within 1e-9 (relative) of the lowest tie, and the tie goes to a shifted split. N2's two
  Cholesky splits tie to the last bit at HF (0.573 Ha²), and the shifted one is 14–20% lower
  along the trajectories (`ft.residual_inputs`;
  tests/test_ft_residual_inputs.py::test_energy_variance_tie_goes_to_the_shifted_split)
  (changed from the design specification; m3-results.md §11). The chosen split is evaluated at
  Ψ_{k+1} of every iteration, not once per system: the shifted Cholesky split of h6a has 0.99 Ha²
  at HF and 1.23 Ha² at SD iterate 1 (`ft.residual_inputs`;
  tests/test_ft_pipeline.py::test_iteration_costs_follow_f14_on_a_hand_built_input)
  (changed from the design specification; m3-results.md §11).
- Option, reported next to the sampled line search and not part of C_iter: amplitude-estimated
  energies of H′ (F11 with α = λ and C_obs = C_W, `ft.residual.energy_ae`). Per candidate
  N_E = 2πκλ/eps_ls applications of C(Ψ_k) + C_step(k; ε_c) + C_W + Q_A/2 Toffolis, with
  Q_A = Q + Q_walk − 2N + 1 the reflected qubits (Q the larger of Q(Ψ_k) and the candidate's
  Q_prep), in one coherent run on Q_run = Q_A + ⌈log₂(N_E/2)⌉ logical qubits [estimate], b1's
  count without the +4 of U_Q (changed from the design specification; m3-results.md §11). Its
  bias, ⟨Ψ_{k+1}(ε_c)|H′ − H|Ψ_{k+1}(ε_c)⟩ for each candidate (not the ground-state de_fact of
  F9), is not counted.
- Tests: tests/test_ft_residual.py::test_line_search_cost_follows_f13, ::test_energy_routes;
  tests/test_ft_residual_inputs.py::test_energy_variances_pick_the_lowest_df_split;
  tests/test_ft_stateprep.py::test_candidate_step_costs_give_qubits_and_flags.

**F14 (iteration, totals).** Definitions; the classical first step [derived]. Code:
`ft.pipeline.iteration_costs`, `ft.residual.compare_routes`, `ft.residual.best_route`,
`ft.pipeline.loop_totals`.

- C_iter(k) = T_res(k) + T_LS(k): T_res is the cheapest of direct_mg, aux_mg, ae_product, ae_x
  and ae_aux at κ = 1 (and, reported separately, at κ_practical = 2), on the residual inputs of
  Ψ_k (the exact A and aux bias at Ψ_k) with the walk of the selected encoding (F9) and its
  factorization bias at Ψ_k; gradient estimation never competes, an infeasible route costs ∞.
  With the selected walks no route is infeasible over the costed iterates (the largest
  factorization bias is 0.47 r‖A‖_m, n2s k = 16; `ft.pipeline`).
- Classical first step, the default for the hand-off: C_iter(0) = 0, since A₀ = W (M1) and its
  step ε₂ (F16) are classical.
- The classical-first-step loop is costed on its own record: step 0 at ε₂ along d₀, then the
  exact continuation, `run_cqe(..., first_eps=ε₂)`
  (`ft.trajectory.exact_trajectory(..., first_step="second_order")`), with the optimizer's
  memory of step 0 kept (changed from the design specification; m3-results.md §11). Its iterates
  k ≥ 2 differ from those after the exact first step by up to 30 mHa and 0.11 in p on n2s,
  0.45 mHa on h6s, and at most 0.012 mHa (L-BFGS) and 0.33 mHa (SD) at equilibrium
  (`results/ft_<system>.json`, `trajectories`; `ft.trajectory`). A plain L-BFGS restart from Ψ₁,
  which loses the step-0 pair, lies 3.4–7.6 mHa higher at k = 2 on h4, h6a, n2 and h8 (0.24 mHa
  on lih; `ft.trajectory`)
  (tests/test_ft_trajectory.py::test_second_order_first_step_is_the_first_eps_continuation,
  ::test_second_order_first_step_keeps_the_step0_memory).
- C_loop(k) = Σ_{j<k} C_iter(j); the total to chemical accuracy is C_loop(k*). Iteration costs
  run to k* + 4 (and at least to k_v − 1, for F17); if k* is not reached (n2s), every iteration
  of the record is costed and the per-iterate tables stop at k = 8
  (changed from the design specification; m3-results.md §11).
- Without residual inputs (h8 by default) only the line search is costed: C_iter is then a lower
  bound (flag "line search only").
- Crossovers (T5): `ft.residual.crossover_norm` gives the ‖A‖_m at which two routes cost the same
  (A scaled, biases fixed); without biases sampling and AE cross at
  ‖A‖*_m = V_m C(Ψ_k)/(2πκΓr(C(Ψ_k) + C_obs)) for a uniform C_obs that includes the reflection
  (tests/test_ft_residual.py::test_crossover_norm_matches_the_closed_form).
- Tests: tests/test_ft_pipeline.py::test_iteration_costs_follow_f14_on_a_hand_built_input,
  ::test_iteration_coverage_rules, ::test_loop_totals_and_first_heisenberg_iterate;
  tests/test_ft_residual.py::test_compare_routes_and_best_route.

**F15 (hand-off).** [derived]; the certificate's attribution [cited: Eckart, Phys. Rev. 36, 878
(1930), volume not checked against the source; Temple (1928), reference not checked]. Code:
`ft.handoff.handoff_table`, `ft.handoff.optimal_handoff`, `ft.handoff.overlap_lower_bound`,
`ft.handoff.rule_e_certifiable`, `ft.handoff.runs_for_confidence`; E₁ from
`ft.trajectory.exact_trajectory`.

- QPE on the walk returns an outcome within ΔE of E₀ with probability w·p_k, p_k = |⟨ψ₀|Ψ_k⟩|²,
  where the window factor w (not the AE constant κ of F11) is set by where the ground-state
  phase falls on the QPE grid and is nearly state-independent. w and the logical success
  probability of a run (1 − p_fail_qpe) rescale every QPE term alike; both are set to 1. On H4
  with a DF H′ (threshold 1e-3), t = 15 control qubits and ΔE = 1 mHa, w = 0.98619 (the window
  probability over p_k), and it varies by less than
  1e-8 along the L-BFGS iterates
  (tests/test_ft_walk.py::test_window_probability_over_p_is_k_independent); with the exact H the
  spread is 1.21e-8 [build check], so the 1e-8 holds in that DF setup only
  (changed from the design specification; m3-results.md §11).
- Handing off Ψ_k costs T_E(k) = C_loop(k) + (C(Ψ_k) + C_QPE)/p_k (rule E, expected cost) or
  T_δ(k) = C_loop(k) + R_δ(p_lb,k)·(C(Ψ_k) + C_QPE) (rule δ), with
  R_δ(p) = ⌈ln δ_conf/ln(1 − p)⌉ (1 if p ≥ 1 − δ_conf), δ_conf = delta_conf = 0.01 (not the
  evolution time δ of F10–F12), and C_QPE the QPE of the selected encoding (F9). A
  C_iter(j) = ∞ makes T_E and T_δ infinite from k = j + 1 on.
- saving(k) = C_QPE/p₀ − (C(Ψ_k) + C_QPE)/p_k ≤ C_QPE(1/p₀ − 1), the budget of all savings. The
  break-even HF overlap is p₀*(k) = C_QPE/T_E(k): handing off Ψ_k beats HF iff p_HF < p₀*(k).
  Step k pays, T_E(k + 1) < T_E(k), iff
  C_iter(k) < (C(Ψ_k) + C_QPE)/p_k − (C(Ψ_{k+1}) + C_QPE)/p_{k+1}
  (tests/test_ft_handoff.py::test_marginal_condition_property).
- Certificate: p_k ≥ p_lb = 1/(1 + σ_k²/(E₁ − E_k)²) for E_k < E₁ (nan otherwise), with
  σ_k² = ⟨H²⟩ − ⟨H⟩² and E₁ a lower bound on the first excited energy of the state's sector. With
  q = 1 − p_k, q(E₁ − E_k) ≤ p_k(E_k − E₀), and Cauchy–Schwarz gives
  σ_k² ≥ p_k(E_k − E₀)²/q ≥ q(E₁ − E_k)²/p_k. Exact for two-level states
  (tests/test_ft_handoff.py::test_overlap_lower_bound_on_h4_states).
- Rule E accepts the first outcome below E_k + 2ΔE and is certified iff E₁ − E_k > 3ΔE; rule δ
  makes R_δ(p_lb) runs and keeps the lowest outcome. The optima (`ft.handoff.optimal_handoff`),
  written k_opt: "expected" minimizes T_E over the certified rows only (k_opt of rule E),
  "delta" the certified T_δ, and "expected_oracle" and "delta_oracle" minimize over every row,
  the latter with the exact p_k, as if the ground-state outcomes could be recognized or p_k were known; ties
  go to the smaller k (changed from the design specification; m3-results.md §11). HF of h6s and
  n2s is not certified: E_HF − E₀ = 0.282 and 0.566 Ha (T2) against E₁ − E₀ = 0.105 and
  0.192 Ha (`ft.trajectory`).
- E₁ is the lowest excited singlet in the irrep of ψ₀, since the circuits preserve the point
  group exactly (F1–F3); the any-irrep E₁ is reported alongside. From the 12 lowest singlet FCI
  roots, a group of roots whose successive energies differ by at most 1e-6 Ha is tested as a
  subspace: it contains a state of irrep g iff the largest eigenvalue of CᵀP_gC exceeds ½ (C the
  group's CI vectors, P_g the projector on the determinants of irrep g), and E₁ is the lowest
  energy of the first such group. The solver returns degenerate roots of different irreps as
  mixtures, so a per-root rule can skip a symmetric state and overestimate E₁, which would
  invalidate the certificate (changed from the design specification; m3-results.md §11). If no
  root qualifies, E₁ is the highest root, a lower bound (flag "e1_lower_bound").
- Loops: "measured" (every iteration costs), "classical_first_step" (C_iter(0) = 0, on the
  classical-step record, F14) and "free" (C_loop ≡ 0, the bound for generators that cost
  nothing, e.g. replayed classical-ACSE generators).
- Tests: tests/test_ft_handoff.py::test_hf_row_and_loops,
  ::test_runs_for_confidence_against_exact_arithmetic, ::test_break_even_overlap,
  ::test_certificates_in_the_table, ::test_optimal_handoff,
  ::test_infinite_iteration_costs_and_coverage;
  tests/test_ft_trajectory.py::test_e1_degenerate_roots_are_tested_as_a_subspace,
  ::test_e1_same_and_any_irrep.

**F16 (classical first step).** [derived]. Code: `ft.trajectory.second_order_step`; used by
`ft.trajectory.exact_trajectory(..., first_step="second_order")`.

- At the RHF determinant A₀ is known in closed form, A_ovov = W = 4(ia|jb) − 2(ib|ja) (M1,
  docs/m1-results.md section 3), so the first direction d₀ = M conj(mask(A₀)) needs no
  measurement (M the metric: the identity, or (2I + T)/3 with (Tx)[p,q,r,s] = x[p,s,r,q] for the
  spin-orbital metric, `optimizers.apply_metric`). Along
  Ψ(ε) = exp(εG)Ψ, G = Σ d e_{pq,rs} (anti-Hermitian):
  E′(0) = ⟨[H, G]⟩ = −Re Σ d A, and
  E″(0) = ⟨[[H, G], G]⟩ = 2⟨φ|H|φ⟩ + 2 Re⟨HΨ|Gφ⟩ with φ = GΨ, a CID-type contraction computed
  classically.
- The step is ε₂ = −E′(0)/E″(0). For the spin-free metric E′(0) = −‖A₀‖², the design's
  ε₀ = ‖A₀‖²/E″(0); for the spin-orbital metric E′(0) = −⟨g, Mg⟩ with g = conj(A₀)
  (changed from the design specification; m3-results.md §11).
- The step is undefined for E″(0) ≤ 0 (flag "nonpositive_curvature") and meaningless at a
  converged state, where d is solver noise: when the predicted lowering E′(0)²/(2E″(0)) is below
  1e-10 Ha (flag "converged_state"), ε₂, ε* and their ratio are nan. At the H4 FCI state the
  ratio came out as 0.49 before this rule (`ft.trajectory`)
  (changed from the design specification; m3-results.md §11).
- ε₂ is compared with the exact line search ε* of `run_cqe` (same state, metric and mask), and
  E″(0) with a Richardson-extrapolated central second difference (h = 2e-3 and 1e-3).
- Tests: tests/test_ft_trajectory.py::test_second_order_step (h4, lih: ε₂/ε* within 3% and to
  Appendix A to 1e-3; E″(0) against dense Fock-space matrices), ::test_second_order_step_slow
  (h6a, n2), ::test_second_order_step_metric_and_state,
  ::test_second_order_step_at_a_converged_state.

**F17 (CQE-only energy).** [design]. Code: `ft.handoff.cqe_only_cost`; wall clock in
`ft.pipeline`.

- k_v is the first iterate with E_k − E₀ ≤ e_tol_energy = 1 mHa. The final energy with the exact
  integrals takes S_fin = V_E/σ_final² shots (σ_final = 0.3 mHa; V_E of the split chosen at HF,
  F13, at Ψ_kv) of C(Ψ_kv) + C_rot Toffolis on max(Q(Ψ_kv), Q_rot) qubits, so
  C = C_loop(k_v) + S_fin·(C(Ψ_kv) + C_rot), with the loop convention of the hand-off (classical
  first step by default), against the HF hand-off C_QPE/p₀. It is a lower bound: finite-shot
  trajectories stop on noise floors (M2). If k_v is not reached the cost is infinite.
- Its wall clock is the loop's campaigns to k_v plus the final shots (F19), against 1/p₀ QPE runs.
- Tests: tests/test_ft_handoff.py::test_cqe_only_cost;
  tests/test_ft_pipeline.py::test_cqe_only_wall_clock_and_the_qpe_run.

**F18 (physical).** [OF: `resource_estimates/surface_code_compilation/physical_costing.py`,
transcribed with floats]; the readout of a shot and the reaction floor [assumption]. Code:
`ft.physical.surface_code_cost`, `ft.physical.SurfaceCodeModel`.

- For each known factory (OpenFermion's two-level T factory, only at p = 1e-3, then every
  AutoCCZ(l1, l2) with l1 = 5, 7, …, 23 and l2 = l1 + 2, …, 39, in OpenFermion's order) and each
  d in `distances`:
  storage = ⌈(1 + routing)·Q⌉ logical patches;
  area = storage·2(d + 1)² + F·A_fac physical qubits;
  cycles = ⌊N_Tof/F·rounds_fac⌋, plus 2d readout cycles for a shot;
  seconds = cycles·t_cycle;
  failure = min(1, f_fac·N_Tof + 0.1(100p)^{(d+1)/2}·storage·cycles),
  with A_fac, rounds_fac and f_fac the factory's footprint, cycles per state and failure per
  state. Layouts with failure ≤ budget are kept, and the smallest area × cycles wins (the first
  on ties, as in OpenFermion). The model assumes [OF] that routing reaches the factories as fast
  as they run and that the run is limited by magic-state production, one Toffoli per state.
- It reproduces OpenFermion's `cost_estimator` exactly (physical qubits, distance, seconds,
  failure) at its settings: p = 1e-3, 1 µs, F = 4, 50% routing, d = 7…33, budget 0.1
  (tests/test_ft_physical.py::test_matches_openfermion_cost_estimator). It differs in that the
  seconds are floats (`cost_estimator`'s timedelta × qubits product overflows near 1e11–1e12
  Toffolis; here 1e13 are fine, tests/test_ft_physical.py::test_no_overflow_at_large_toffoli_counts),
  the budget, distances, F, routing and cycle time are parameters, a shot adds 2d cycles of
  logical readout, which also count in the data failure, and a Clifford-only shot (N_Tof = 0)
  lasts 2d cycles on the data block alone (factory "none";
  tests/test_ft_physical.py::test_clifford_only_shot_lasts_two_d_cycles). With `reaction_us`
  set, the Toffoli part lasts at least ⌈toffoli_depth·t_react/t_cycle⌉ cycles, and a shot's
  readout follows it (tests/test_ft_physical.py::test_reaction_floor_binds_when_set).
- Monotonicity. Every layout's failure grows with N_Tof and the feasible set shrinks as the
  budget tightens, so the optimal area × duration is non-decreasing in N_Tof and in 1/budget
  (tests/test_ft_physical.py::test_objective_is_monotone_in_toffolis_and_budget). The chosen d is
  monotone in neither, because the optimum can switch factories: 233 logical qubits at budget 0.1
  need d = 15 with AutoCCZ(9, 15) for 562 Toffolis but d = 13 with AutoCCZ(11, 13) for 1000; one
  shot of 83 logical qubits and 5623 Toffolis needs d = 25 at budget 3e-5 and d = 23 at 1e-5, and
  OpenFermion's own objects make the same choices
  (tests/test_ft_physical.py::test_distance_is_not_monotone_in_toffolis_or_budget_and_openfermion_agrees)
  (changed from the design specification; m3-results.md §11).
- Rates at p = 1e-3, F = 4 and 1 µs cycles: the factory-limited rounds_fac/F of AutoCCZ(l1, l2)
  is max(5, 5.75·l1/l2)·l2/4 µs per Toffoli, i.e. 26.25 µs for l2 = 21 with l1 ≤ 17 and 28.75 µs
  for l2 = 23 with l1 ≤ 19 (27.31 µs for AutoCCZ(19, 21), 30.19 µs for AutoCCZ(21, 23)), and
  46.5 µs for the two-level T factory. The best factories fail with 3.6e-16 (two-level T) and
  3.66e-14 (AutoCCZ(23, 39)) per state, so a budget is infeasible beyond about budget/3.6e-16
  Toffolis (`ft.physical`).

**F19 (wall clock).** [assumption]; LPT [cited: Graham, SIAM J. Appl. Math. 17, 416 (1969)].
Code: `ft.physical.campaign_seconds`, `ft.physical.machines_for`, `ft.pipeline.Campaign`,
`ft.pipeline.lpt_makespan`, `ft.pipeline.machines_for_campaigns`.

- A campaign is n independent runs of one estimator on M identical machines; a run is not split.
- Sampling (direct_mg, aux_mg, each line-search candidate, the final energy of F17): shots of one
  circuit at the budget p_fail_shot with 2d readout cycles. T(n, M) = t_run⌈n/M⌉, n rounded up to
  whole runs: 5 shots on 4 machines take 2t_run, not 1.25t_run
  (tests/test_ft_physical.py::test_campaign_seconds_and_machines)
  (changed from the design specification; m3-results.md §11).
- Amplitude estimation (ae_product, ae_x, ae_aux): one coherent run per element
  (`RouteCost.run_toffolis`), every run at the µs per Toffoli of the layout of the longest one
  (its Toffolis on the route's logical qubits, budget p_fail_ae). One machine takes T_res at that
  rate, the reflection of F11 included
  (changed from the design specification; m3-results.md §11). On M machines the unequal,
  indivisible runs follow the LPT schedule, longest run first onto the least-loaded machine,
  within 4/3 − 1/(3M) of the optimal makespan
  (tests/test_ft_pipeline.py::test_lpt_makespan_against_exhaustive_schedules).
- QPE: one run at the budget p_fail_qpe; a single run does not parallelize, and 1/p repetitions
  take 1/p runs on average.
- A loop runs its campaigns one after another, each spread over the same M machines: iteration
  k + 1 starts from the step of iteration k, and a line search needs the residual's direction.
  The machines for the deadline D = 7 days are the smallest M with Σ_c T_c(M) ≤ D, found by
  bisection [assumption: T_c(M) does not grow with M; true for identical runs, and no LPT
  counterexample among 2e5 random instances of 2–12 runs, `ft.pipeline`]. The count is None when
  a campaign has no layout or when the critical path, the sum of the longest runs of all
  campaigns (the wall clock on unlimited machines), exceeds D, and 0 for the free classical
  iteration (changed from the design specification; m3-results.md §11). For one campaign of
  identical runs it is ⌈n/⌊D/t_run⌋⌉ (`ft.physical.machines_for`): ten runs of 3.9 days need
  10 machines for a week, not 6 (tests/test_ft_physical.py::test_campaign_seconds_and_machines)
  (changed from the design specification; m3-results.md §11).
- The LPT counts equal the smallest M that the lower bound of every schedule allows (summed over
  the campaigns: t⌈n/M⌉ for shots, which is exact, and max(W/M, t_max) for AE runs) for 324 of
  the 404 costed iterations (at most 1.38× it, h6s k = 13) and for 28 of the 31 loop totals that
  have a count (at most 1.05×); the other 5 have none, since their critical path exceeds 7 days
  (`results/ft_<system>.json`, `machines_7d` and `campaigns`). Greedy
  list scheduling in arbitrary order, W/M + (1 − 1/M)t_max [cited: Graham, Bell Syst. Tech. J.
  45, 1563 (1966)], asks for up to 2.6× as many machines for a loop, and a fluid share
  ⌈Σ_c W_c/D⌉ is no schedule: it gives 249 machines to the spin-free h6s loop, whose critical
  path is 8.5 days (`ft.pipeline`; T4)
  (changed from the design specification; m3-results.md §11).
- Tests: tests/test_ft_pipeline.py::test_campaign_schedules_and_machines,
  ::test_machines_for_several_unequal_campaigns,
  ::test_iteration_costs_flag_campaigns_without_a_layout.

## Scope and open questions

What the benchmark run settled about the open questions is in m3-results.md, section 10.

**Out of scope for M3:**

- Gate-level circuits: compilation of PREPARE, SELECT and QROM, and cross-checks with Qualtran or
  PennyLane. M3 takes OpenFermion's block-encoding costs [OF] and checks only the polynomial of
  the walk (F8, `ft.walk`). Left for a later milestone: a controlled THC block encoding in a
  Hadamard test, and an explicit SELECT for iD.
- QSP state preparation of exp(εD) (sparse or DF block encodings of iD). It needs the explicit
  SELECT of a later milestone; the design's QSP-sparse numbers are unvalidated estimates,
  possibly 1.5–2× optimistic [design].
- Partially fault-tolerant architectures with analog rotations (STAR) and similar models,
  probabilistic error cancellation (PEC), angle-dependent injection (a later milestone).
  `PrimitiveCounts.n_pauli_rot` exports the Pauli-rotation counts they need (F3–F5).
- Synthesized rotations (Ross–Selinger, repeat-until-success, mixed fallbacks): their T counts
  and their leakage out of N and S_z.
- Closed-loop compiled CQE: rounded or Trotterized steps inside the line search, compressed
  L-BFGS directions, finite-shot fault-tolerant runs. M3 costs exact trajectories and checks
  their compiled circuits by the open-loop replay (`ft.circuits.replay_elementwise`). The
  design's closed-loop results (δ_step = 1e-3 adds at most one L-BFGS step; compressing a step to
  99% elementwise adds none) are carried as assumptions [design; assumption].
- Complex-state propagation of leaf circuits (F6 flags it), and measuring Im A.
- Hamiltonian preprocessing: BLISS, SCDF, DFTHC, spectrum amplification.
- Gradient estimation beyond the formula of F12: emulating it, or determining κ_g.
- Other readouts and estimators: number-conserving fermionic shadows; Pauli and QWC shadows on
  fault-tolerant hardware (dominated in M2); DF-Trotter aux routes.
- Early-fault-tolerant phase estimation (QCELS and similar), the variance-based CQE (H²
  residuals), excited states.
- Classical ACSE generators, which need a 3-RDM reconstruction; the free loop of F15 is only
  their bound.
- Systems beyond N = 8 orbitals, and the H8 residual: h8 costs its line search alone unless
  `benchmarks/run_ft_pipeline.py --h8-residual` is given, so its loop costs are lower bounds.

**Open questions:**

1. The X-LCU cost C_obs = 2L_X [estimate] (ae_x: the Pauli LCU of X_e) decides the ae_x rows.
   The 1-norm α_X is loose (‖X_e‖ on the H4 sector is 0.16–0.50 of α_X, `ft.residual_inputs`),
   so structured block encodings, left for a later milestone, could lower these rows 2–6×.
2. The aux-route bias is counted fully against the r budget, so the aux_mg and ae_aux rows are
   conservative in that respect; in the design a biased residual with δ ≤ 0.8 barely slowed the
   CQE [design]. The aux route is also unstable for δ·w_max ≥ π (M2), which part of the selected
   ae_aux rows violate (m3-results.md section 10, item 2).
3. The line-search model (11 candidates at ±1.6 mHa) dominates the iterations before k* at
   equilibrium. Cheaper step rules (L-BFGS unit steps, fits through 3 energies, the classical
   second-order step beyond k = 0) and amplitude-estimated energies (F13) are a question for the
   next milestone.
4. The δ_step = 1e-3 budget is empirical [design]. It assumes frozen, bit-identical compiled
   circuits for every shot, and it does not cover the product-formula error of F3: after the RHF
   step it is 2.7e-2 (h6s) and 3.1e-2 (n2s), 27 and 31 δ_step and 20 and 16 times the
   equilibrium build of the same molecule (h6a, n2), and it reaches 9.5e-2 along the n2s replay
   (F3; T9).
5. The hand-off conclusions hold for equilibrium molecules (p_HF = 0.89–0.97 here). The break-even
   overlaps p₀*(k) say where a measured loop could pay; overlap-limited systems need a scaling
   study, of which h6s and n2s are a first look.
6. THC fits are non-convex, and λ depends on the penalty and on the fit's convergence: the THC
   rows are as good as `factor.fit_thc_active`.
7. The trajectories are exact (no shot noise in the loop), and r = 0.3 is M2's convention, while
   the finite-shot CQE converges with much less accurate residuals (docs/m2-results.md,
   section 4).

## Appendix A: reference numbers

The regression targets of the design, recomputed during the build on the symmetry-adapted builds
(D2h; C2v for LiH), exact trajectories, δ_step = 1e-3. Each number is pinned by the test named,
or checked by the benchmark only: the Appendix A table of `benchmarks/run_ft_pipeline.py`
(results/ft_summary.md; m3-results.md, section 8), which compares the numbers below with the
run, except Q(Ψ₆) = 83, the n2 comparison of symmetric and plain orbitals and the comparison of
E₁ with PySCF's symmetry-adapted FCI, which only the tests pin (k* follows from the compared
energies). One value is corrected: n2's V_m on the allowed elements, 3753.0 Ha² (exact) instead
of 3703 (changed from the design specification; m3-results.md §11); the benchmark's table
compares with the design value 3703 and reports that row as its one mismatch (96 of 97), so
against this appendix all 97 compared quantities reproduce.

**Allowed elements (F2a)**, allowed / all. Pinned by
tests/test_ft_symmetry.py::test_allowed_element_counts_match_appendix_a.

| system | spin-free | spin-orbital |
|---|---|---|
| h4 | 28 / 60 | 70 / 150 |
| h6a | 153 / 315 | 408 / 840 |
| lih | 105 / 315 | 264 / 840 |
| n2 | 33 / 315 | 84 / 840 |
| h8 | 496 / 1008 | 1364 / 2772 |

**V_m** = Σ_e m_e v_e (Ha²; direct route, matchgate shadows, spin-free) at SD iterate 2:

| system | allowed | all | method | pinned by |
|---|---|---|---|---|
| h4 | 1220 | 2402 | exact pair sums | tests/test_ft_residual_inputs.py::test_vm_at_sd_iterate_2_matches_appendix_a (slow) |
| h6a | 2.515e4 | 5.144e4 | RB, 5000 settings, seed 7 | benchmark only |
| n2 | 3753.0 | 2.947e4 | exact pair sums (allowed); RB, 5000 settings, seed 7 (all) | benchmark only |

- n2, allowed: the design value 3703 was an RB estimate (seed 7) on an earlier orbital
  labelling. With 2.8e7 term pairs the set is below the 5e7 limit of F10, and its exact pair sum
  is 3753.0 Ha² (changed from the design specification; m3-results.md §11). At SD iterate 1 the
  exact value, 3751.08 Ha², is pinned by
  tests/test_ft_residual_inputs.py::test_n2_variances_use_exact_pair_sums (slow).
- The RB values are reproduced with the same seed (n2, all elements: 2.941e4 ± 159 Ha², −0.4σ),
  so they are no independent check. The exact pair sum of the h6a allowed set at SD iterate 2 is
  25,308 Ha² [build check], against which Appendix A's 2.515e4 is 0.6% (1.4σ) low; at SD
  iterate 1 the RB estimate is pinned within 3σ of the exact 25,288.53 Ha² by
  tests/test_ft_residual_inputs.py::test_element_variances_rb_branch.

**L-BFGS trajectories** (spin-free, memory 10): step k; E − E_FCI and p after the step;
elementwise n_d / n_s, b and C_step of the step; cumulative C(Ψ_{k+1}).

| system | k | E − E_FCI after (mHa) | p after | n_d / n_s | b | C_step | C(Ψ_{k+1}) |
|---|---|---|---|---|---|---|---|
| h4 | 0 | 21.504 | 0.9802 | 10 / 0 | 15 | 160 | 160 |
| h4 | 1 | 3.890 | 0.9960 | 40 / 24 | 18 | 1192 | 1352 |
| h4 | 2 | **1.203** | 0.9981 | 38 / 24 | 18 | 1154 | 2506 |
| h6a | 0 | 37.956 | 0.9576 | 51 / 0 | 18 | 969 | 969 |
| h6a | 1 | 12.014 | 0.9850 | 273 / 120 | 21 | 8526 | 9495 |
| h6a | 2 | 5.236 | 0.9907 | 271 / 120 | 21 | 8482 | 17977 |
| h6a | 3 | 2.991 | 0.9950 | 271 / 120 | 21 | 8482 | 26459 |
| h6a | 4 | 1.877 | 0.9974 | 273 / 120 | 21 | 8526 | 34985 |
| h6a | 5 | **1.111** | 0.9987 | 273 / 116 | 21 | 8442 | 43427 |
| n2 | 0 | 35.209 | 0.9635 | 27 / 0 | 17 | 486 | 486 |
| n2 | 1 | 14.678 | 0.9813 | 55 / 0 | 18 | 1045 | 1531 |
| n2 | 2 | **1.147** | 0.9994 | 55 / 0 | 18 | 1045 | 2576 |

Bold: the step that reaches 1.6 mHa (k* = 3, 6, 3). The counts, bits and costs, and Q(Ψ₆) = 83
for h6a, are pinned by tests/test_ft_stateprep.py::test_appendix_a_elementwise_regressions (which
also checks the energies to 1e-3 mHa and the overlaps to 1e-4); the energies, overlaps and k* to
1e-6 by tests/test_ft_trajectory.py::test_h4_trajectory_reproduces_appendix_a (h4) and
::test_h6a_n2_trajectories_reproduce_appendix_a (h6a, n2; slow).

**Other trajectory and spectrum numbers.**

- HF overlaps: h4 0.9455, h6a 0.9026, n2 0.9257, lih 0.9743. Pinned by
  tests/test_ft_trajectory.py::test_e1_same_and_any_irrep.
- ‖A‖ before steps 0–3 (Ha, full tensor):

  | system | k = 0 | 1 | 2 | 3 |
  |---|---|---|---|---|
  | h4 | 0.8585 | 0.6115 | 0.2157 | 0.1256 |
  | h6a | 1.0993 | 0.7424 | 0.4219 | 0.2397 |
  | n2 | 1.8955 | 0.6638 | 0.4038 | 0.1168 |

  Pinned by tests/test_ft_trajectory.py::test_h4_trajectory_reproduces_appendix_a (h4) and
  ::test_h6a_n2_trajectories_reproduce_appendix_a (h6a, n2; slow).
- Classical first step, ε₂/ε*: h4 1.0159, h6a 1.0220, n2 0.9805, lih 1.0192. Pinned to 1e-3 by
  tests/test_ft_trajectory.py::test_second_order_step (h4, lih) and ::test_second_order_step_slow
  (h6a, n2; slow).
- E₁ − E₀ (Ha), same irrep / any irrep: h4 0.5657 / 0.5633; h6a 0.3909 / 0.3909; n2 0.7115 /
  0.4117; lih 0.1332 / 0.1332. Pinned by tests/test_ft_trajectory.py::test_e1_same_and_any_irrep
  (also against PySCF's symmetry-adapted FCI to 1e-8 Ha).

**QPE** (ΔE = 1 mHa, selection rule restricted to DF, `select_qpe(..., kind="df")`):

| system | threshold | L | Ξ | λ_DF (Ha) | Toffolis / step | total | logical qubits |
|---|---|---|---|---|---|---|---|
| h4 | 3e-3 | 7 | 27 | 4.935 | 868 | 6.73e6 | 188 |
| h6a | 3e-3 | 11 | 63 | 9.726 | 1309 | 2.00e7 | 233 |
| n2 | 3e-3 | 20 | 82 | 9.423 | 1420 | 2.10e7 | 235 |
| lih | 3e-4 | 18 | 68 | 9.259 | 1366 | 1.99e7 | 235 |

- Pinned by tests/test_ft_qpe.py::test_select_qpe_df_reproduces_appendix_a (threshold, L, Ξ,
  Toffolis per step and qubits exactly, λ to 1e-3 Ha, the total to 0.5%). These are the DF
  selections; the QPE of the hand-off is the cheapest admissible encoding overall, SF on h4, n2
  and lih and THC on h6a (T1) (changed from the design specification; m3-results.md §11).
- THC, h6a: M = 15 with the auto penalty, λ = 9.125 Ha, 1035 Toffolis per step, 1.48e7 in total.
  Pinned by tests/test_ft_qpe.py::test_thc_h6a_appendix_a; the scan selects it,
  tests/test_ft_qpe.py::test_thc_scan_h6a_selects_m15 (slow).
- n2: DF λ and step cost agree to 2% between the symmetry-adapted and plain RHF orbitals
  (observed: λ to 1.3e-14 relative, the same C_W = 1420, L = 20, Ξ = 82). Pinned by
  tests/test_ft_qpe.py::test_n2_df_on_symmetric_and_plain_orbitals.

**Physical** (OpenFermion's model, p = 1e-3, 4 factories). Pinned by
tests/test_ft_physical.py::test_appendix_a_physical_numbers.

- H6 DF QPE (the DF selection above: 233 logical qubits, 19,997,593 Toffolis): 956,160 physical
  qubits, d = 23, 574.9 s at budget 0.1; 1,026,160 physical qubits, d = 25, 574.9 s at 0.01.
- H6 Ψ₆ shot (83 logical qubits, 43,427 Toffolis, plus 2d readout cycles): 721,960 physical
  qubits, d = 25, 1.249 s at 1e-5; 696,960 physical qubits, d = 23 at 1e-4.
- 26.25–28.75 µs per Toffoli (the H4 and H6 DF QPE runs at budget 0.1, 188 qubits / 6.73e6
  Toffolis and 233 / 2.00e7: AutoCCZ with l2 = 21 and 23, F18).

**Primitives.**

- Jacobi–Anger degrees: R(0.2, 1e-4) = 3, R(0.5, 1e-4) = 4, R(0.79, 1e-4) = 5, R(1.16, 1e-4) = 6,
  R(4.87, 1e-6) = 14, R(6.81, 1e-6) = 17. Pinned by
  tests/test_ft_primitives.py::test_jacobi_anger_reference_degrees.
- Sine-window AE: RMS/(π/M_G) = 0.974 / 0.993 / 0.998 for M_G = 63 / 255 / 1023. Pinned by
  tests/test_ft_residual.py::test_ae_constant_from_sine_window_phase_estimation.
- Phase-gradient bits b = 15 / 18 / 21 for A = 10 / 64 / 393 additions at δ = 1e-3. Pinned by
  tests/test_ft_primitives.py::test_phase_gradient_bits_reference_values.
- Readout rotation C_rot = 405 Toffolis at N = 6 (b_r = 23). Pinned by
  tests/test_ft_stateprep.py::test_readout_rotation_toffolis.
