# M4 cost model

M4 asks where the CQE pays as a state preparer for fault-tolerant QPE: on overlap-limited systems, with
step rules that measure no energies, in a closed loop of finite-shot fault-tolerant iterations, under a
hand-off rule that does not tie certification to the energy of the hand-off state, under QPE-side
amplification schemes that change what a better state is worth, and for ground-state properties. It
extends the resource-estimation layer of [m3-cost-model.md](m3-cost-model.md) (F1–F19, F2a) and reuses
its models unchanged: the elementwise compilation and its bits (F2, F2a, F3, F7), the readout rotation
(F4), GQSP evolution and the QPE selection (F8, F9), the residual routes (F10–F12), the line search and
its energies (F13), iteration costs and totals (F14), the hand-off rules E and δ with the Eckart–Temple
certificate (F15), the classical first step (F16), and the surface-code costing and wall clock (F18,
F19). New are five blocks: systems and overlap-limited scaling (F20–F25), cheap step rules (F26–F31), a
closed-loop finite-shot fault-tolerant CQE (F32–F38), an E1-referenced hand-off with windowed QPE,
certificates, QPE-side schemes and the break-even overlap and size (F39–F45), and ground-state properties
against QPE (F46–F53). They live in `steprules` (`thc_engine_cqe.steprules`), `ft.steprules`,
`ft.closedloop`, `ft.scaling`, `ft.acceptance`, `ft.certify`, `ft.amplify` and `ft.properties`; the
changes to `ft.symmetry`, `ft.circuits`, `ft.handoff`, `ft.pipeline`, `ft.residual_inputs` and
`ft.trajectory` are additive, and M3's outputs are unchanged [build check: M3's golden h4 test and
`benchmarks/run_ft_pipeline.py` on h4 and n2s reproduce M3's records bit for bit].

The unit of cost is M3's: the Toffoli, one CCZ state, with logical qubits beside it. "C_QPE units" divide by
the system's selected C_QPE (F9, at ΔE = `de_qpe` = 1 mHa), so that handing off HF by repetition costs 1/p_HF.
Shot, preparation and call counts are real numbers (M3). Energies are in Ha (mHa where marked), dipoles and
field strengths in atomic units, bond lengths in bohr for the hydrogen chains and in Å for N2 (F20). N is the
number of active spatial orbitals; on the hydrogen chains the number of atoms n equals N. ψ₀, E₀ and E₁ belong
to the active-space H (E₁ the lowest excited singlet in the irrep of ψ₀, M3's F15), Δ₁ = E₁ − E₀; ψ₀′, E₀′ and
E₁′ to the selected factorized H′ (F9). An overlap p_k = |⟨ψ₀|Ψ_k⟩|² is the weight on ψ₀ of H unless it is
marked "on H′"; Ψ̃_k is a compiled or closed-loop state and p̃_k its weight; E1lb is the assumed lower bound
on the excited levels a hand-off state populates (F41); ΔE = `de_qpe`. Two symbols differ from M3: b_rot
denotes M3's rotation bits β (β = 16), since β denotes power-law exponents here (F23), and ‖D_k‖ is the
CI-space operator norm of the step generator D_k (M3's G of F16). T_E(k) is M3's rule-E hand-off cost (F15),
T_en(k) the Toffolis of the energies a step rule measures in iteration k (F28; M3's T_LS of F13 for the line
search). θ without a subscript is rule B's energy threshold (F40); θ_j, θ₀, θ_pass, θ_stop and θ̄ are walk
phases (F39, F43), 2θ_e an AE phase (F34), θ_R3 R3's gate ratio (F31) and Θ a cost gate (F38).

Every formula names the functions that implement it, as in M3: `ft.module.function` is short for
`thc_engine_cqe.ft.module.function`, `steprules.x` is `thc_engine_cqe.steprules.x`, and other package
modules are named by their path in `thc_engine_cqe` (`cqe`, `optimizers`, `fci_ops`, `integrals`,
`measure`, `diagnostics`); benchmark helpers are named with their script. The module docstrings state
the same models with their derivations. The results of the benchmarks (`benchmarks/run_m4_scaling.py`,
`run_m4_loop.py`, `run_m4_handoff.py`, `run_m4_properties.py` and `run_m4_summary.py`) are in
[m4-results.md](m4-results.md), whose tables U0–U12 are cited by label and whose sections by §. Numbers
marked "quick run" come from the `--quick` benchmarks and numbers marked "full run" from a full run of
the benchmark chain; both are examples, and m4-results.md holds the values of record.

The model was designed before the build, and m4-results.md, §13, lists what changed from that design
specification and why. Each change is folded in below in its implemented form, followed by
"(changed from the design specification; m4-results.md §13)".

Tags, with M3's meanings:

- [OF]: OpenFermion's resource-estimate functions or conventions (OpenFermion 1.8.1);
- [derived]: derived for this package and tested;
- [cited]: from the cited literature;
- [estimate]: a model choice, not a bound;
- [assumption]: a stated assumption;
- [design]: taken from the M4 design notes (not in the repository) and re-checked during the build;
- [build check]: verified once during the build by a script that is not part of the repository.

## Parameters

M3's `ft.primitives.FTParameters` are unchanged (docs/m3-cost-model.md, Parameters). M4 passes the
residual accuracy explicitly, every M4 cost taking r (or `r_values`) in place of `FTParameters.r`, and
measures the spin-free element set (`FTParameters.kind = "spin_free"`, which the scan and the closed loop
require); the spin-orbital metric of L-BFGS is a preconditioner, not an element set. The M4 parameters
are keyword defaults and module constants:

**Systems and scaling (F20–F25).**

| parameter | value | code |
|---|---|---|
| axes | H chains `h<n>_r<R>`, n = 4, 6, 8, 10, R = 2.5, 3.0, 3.5, 4.0 bohr; N2 CAS(6,6) `n2_r<R>`, R = 1.098, 1.6, 2.0, 2.4, 2.8 Å; N2 CAS(10,8) `n2v_r<R>`, R = 1.098, 2.0, 2.4, 2.8 Å | `ft.scaling.SIZE_AXIS`, `ft.scaling.OVERLAP_AXIS`, `ft.scaling.ACTIVE_SPACE_AXIS` |
| other sets | EQUILIBRIUM, CLOSED_LOOP (11 systems), PROPERTIES (13), HOLDOUT (h12, not run), QUICK | `ft.scaling.EQUILIBRIUM` … `ft.scaling.QUICK` |
| scan | 6-step records, costs for k ≤ 3 at r = 0.3, 1, 3; dense inputs at N ≤ 8 | `ft.scaling.scan_system(k_cost=3, n_steps=6)`, `ft.scaling._N_DENSE_MAX` |
| spectrum | 12 singlet roots (2 at N = 10); ⟨S²⟩ tolerance 1e-6, then 1e-4 | `ft.scaling.spectrum(n_roots=12, spin_tols=(1e-6, 1e-4))` |
| overlap fits | window n ∈ (6, 8, 10); rates last and Aitken; target p = 1e-3 | `ft.scaling.fit_log_overlap(window=(6, 8, 10))`, `ft.scaling.RATES`, `ft.scaling._P_TARGET` |
| cost exponents | N ≥ 6 | `ft.scaling._N_COST_MIN` |
| slice rule | s ∈ (1, 2, 4, 8, 16), η_s = 0.05 | `ft.scaling.SLICE_RULE`, `ft.scaling.ETA_SLICE`, `ft.closedloop.SLICE_CHOICES` |

**Residual accuracy.**

| parameter | value | code |
|---|---|---|
| r of exact-trajectory costs | 0.3, 1, 3 (rule B in U8 at r = 1) | `ft.steprules.run_step_rule_system(r_values=(0.3, 1.0, 3.0))`; `benchmarks/run_m4_handoff.py` (`R_RULE_B`) |
| r of the closed loop | 1; U6: oracle 1, 2, 3, lagged 0.5, 1, 2 (lagged 1 decides); deep runs lagged 0.3, 1 | `ft.closedloop.LoopSpec.r`; `benchmarks/run_m4_loop.py` (`blocks`); `benchmarks/run_m4_summary.py` (`DEEP_R`) |
| r of the property loops | 1 and 0.3 | `benchmarks/run_m4_properties.py` (`MEASURED_LOOPS`) |
| κ | 1 in T_res; κ_practical = 2 reported | `FTParameters.kappa`, `FTParameters.kappa_practical` |
| variance state | iterate 1 for N ≤ 6, HF for N ≥ 7; a 100-element subset (seed 11) at N = 10 | `ft.closedloop.system_inputs(variance_state=None)`; `ft.residual_inputs.variance_totals_subset(k_subset=100, seed=11)` |

**Optimizer and step rules (F26–F31).**

| parameter | value | code |
|---|---|---|
| L-BFGS | memory 10, curvature tolerance 1e-2 | `optimizers.LBFGS(memory=10, curvature_tol=1e-2)` |
| metric, steps | spin-orbital, 30 steps; U5 both metrics, 40 steps (30 at N = 8) | `steprules.run_step_rule(metric="spin_orbital", max_iter=30)`, `ft.steprules.run_step_rule_system(max_iter=40)` |
| rule, first step | "unit" (R1); classical step ε₂ (F16) | `steprules.StepRule` (`name`, `first`) |
| caps | growth γ_s = 2, angle 3 (model rules, k ≥ 1) | `steprules.StepRule.growth`, `steprules.StepRule.angle_cap` |
| 3-point fit, R3 switch | σ_P3 = 0.47\|ΔE_m\| on {0, t, 2t}, t = 1; θ_R3 = 1 | `steprules.StepRule.sigma_parab`, `steprules.StepRule.parab_t`, `steprules.StepRule.theta` |
| references | M3's 11-step grid at eps_ls = 1.6 mHa; exact line search on [0, 1.5/‖D_k‖] | `FTParameters.eps_candidates`, `FTParameters.eps_ls`; `steprules._LS_BRACKET` |
| energy precisions | c₁ = 1e-4, z = 2, loss ℓ = 0.1 | `ft.steprules.energy_precision` |
| U5 horizon | k* + 2, 20 where k* is not reached, ≤ 12 at N ≥ 8; k_p90 at p ≥ 0.9 | `ft.steprules._K_AFTER`, `ft.steprules._K_UNREACHED`, `ft.steprules._K_CAP_N8`, `ft.steprules._P_REACHED` |

**Closed loop (F32–F38).**

| parameter | value | code |
|---|---|---|
| iterations | 5 (U6, U7 show k ≤ 4); deep runs 30 | `ft.closedloop.run_closed_loop(max_iter=5)`; `benchmarks/run_m4_summary.py` (`DEEP_ITER`) |
| route, propagation | direct_mg (default), ae_x, best; exact (default) or compiled with the slice rule or a fixed s | `ft.closedloop.LoopSpec` (`route`, `propagation`, `slices`) |
| controller | lagged (default) or oracle; ν₁ = 0.5, ν = 1, f_fl = 0.1 | `ft.closedloop.LoopSpec` (`controller`, `rho1`, `rho`, `norm_floor`) |
| AE estimator | κ = 1, fixed offsets, lattice \|n\| ≤ 60 | `ft.closedloop.LoopSpec` (`kappa`, `ae_offsets`), `ft.closedloop._LATTICE_N_MAX` |
| streams, bootstrap | base seed 20261108; 2000 resamples, seed 12345, level 0.95 | `ft.closedloop.BASE_SEED`, `ft.closedloop._N_BOOT`, `ft.closedloop._BOOT_SEED`, `ft.closedloop._LEVEL` |
| stopping rules | fixed iterates 1–4; cost gates Θ ∈ {0.3, 1, 3} C_QPE | `ft.closedloop._FIXED_K`, `ft.closedloop._GATES` |
| replicates | headline 100; other blocks 20; N = 8 controls and deep runs 10 | `benchmarks/run_m4_loop.py` (`blocks`); `benchmarks/run_m4_summary.py` (`DEEP_REPLICATES`) |

**Hand-off (F39–F45).**

| parameter | value | code |
|---|---|---|
| QPE error, cost model | ΔE = 1 mHa; M3's window model (w = K₀ = 1) | `FTParameters.de_qpe` |
| phase points per C_QPE | M = 2n_iter (sine); Kaiser(α = 3) at 4n_iter in U8 | `ft.acceptance.qpe_points(qpe, multiple=2)` |
| design floor, kernel target, output failure | p_min = ε_id = ε_fail = 1e-3 | `ft.acceptance.P_MIN`, `ft.acceptance.EPS_ID`, `ft.acceptance.EPS_FAIL` |
| B_L top margin | D_L = 5 grid units, n_v = 2 | `ft.acceptance.acceptance_bracket(d_l=5)` |
| margins beyond the computed sizes | sin θ₀ = 1, E₁ − E1lb = 1 mHa: m_E = m_B = 3 mHa, m_BL = 12 mHa | `ft.scaling.scaling_report(sin_theta0=1.0, e1_margin=1e-3)` |
| E1lb | E₁(H′) − 0.1 mHa with the model spectrum (N ≤ 8), else E₁(H) − 1 mHa | `ft.certify.E1_MARGIN_MODEL`, `ft.certify.E1_MARGIN_EXACT` |
| certificates | CISD(H′); dedicated sampled and AE energies at z = 3 | `ft.certify.truncated_ci_energy(max_level=2)`, `ft.certify.energy_certificates(z=3.0)` |
| per-run failure budgets | 1e-6 (unverified), 5e-4 (retested), f = 1 | `ft.acceptance.failure_budget` |
| amplification | exponential search growth 6/5; fixed point δ = 0.01 | `ft.amplify.exponential_search_calls(growth=1.2)`, `ft.amplify.fixed_point_calls(delta=0.01)` |
| filters | ε_f = 1e-2; coarse ε_p = 1e-4; w_L = 100 mHa; final QPE B_U iff D_U ≥ 1.3 | `ft.amplify.filter_cost(eps=1e-2)`, `ft.amplify.coarse_qpe_toffolis(eps_p=1e-4)`, `ft.amplify.W_L`, `ft.amplify.FINAL_D_MIN` |
| break-even scan | even n ≤ 3000; overlap floor 1e-300; β ∈ {last, mean, 9}; crossovers none, (20, 2), (20, 4) | `ft.scaling._N_MAX`, `ft.scaling.P_FLOOR`, `ft.scaling._BETA_FIXED`, `ft.scaling._CROSSOVERS` |

**Properties (F46–F53).**

| parameter | value | code |
|---|---|---|
| targets, norms, accuracies ε | NONs (max) 1e-2, 1e-3; 1-RDM (fro) 1e-2, 1e-3; 2-RDM (fro) 1e-1, 1e-2; cumulant (fro) 1e-1, 1e-2; cumulant norm (scalar) 1e-2, 1e-3; LiH dipole (scalar) 4e-3, 1e-3 a.u. | `ft.properties.ACCURACIES`, `ft.properties.TARGET_NORMS` |
| route A | budgets k ∈ {2, 5, 10, 20, 60}; R1 and exact-line-search spin-orbital records, 60 steps | `ft.properties.route_a_rows(budgets=(2, 5, 10, 20, 60))`; `benchmarks/run_m4_properties.py` (`N_STEPS`) |
| route B | leakage share 0.1 of ε; ε_f ≤ 1e-2 | `ft.properties.ground_copy(share=0.1)`, `ft.properties._EPS_F_CAP` |
| estimators | AE κ = 1 (reference); gradient estimation κ_g ∈ {1, 10, 100} | `FTParameters.kappa`, `FTParameters.kappa_g` |
| HF-QPE | 41 fields on [−0.02, 0.02] a.u., degree 8; ≤ 12 halvings and 12 bisections of h | `ft.properties.finite_field_derivatives(h_max=0.02, n_points=41, degree=8)`, `ft.properties._H_HALVINGS` |

**Physical.** `ft.physical.SurfaceCodeModel` with M3's defaults (F18). Shots and matchgate snapshots run at
`FTParameters.p_fail_shot` = 1e-5 with 2d readout cycles; amplitude-estimation runs, the coherent runs of a
property estimate and the HF-QPE energies at `FTParameters.p_fail_ae` = 1e-4 (`ft.properties.run_layout`);
hand-off and filter runs at F40's per-run budgets, 1e-6 (unverified) and 5e-4 (retested), so that logical
failures add at most ε_fail = 1e-3 to the hand-off's identification error, beside the kernel term ε_id ≤ 1e-3
at p₀ ≥ p_min (F40: ≤ 2e-3 in all); M3's `p_fail_qpe` = 1e-2 stays in M3's tables. The machine counts keep
M3's 7-day deadline. U12 (m4-results.md §10; `benchmarks/run_m4_summary.py`, `u12_rows`) lays out the decisive
runs at these budgets (hand-off budgets at f = 1): on n2_r2.4 the residual at k = 1 of the R1 spin-orbital
scan record at r = 1 and 3 (its best route's runs, shots at p_fail_shot, AE at p_fail_ae, with the record's
campaign), one hand-off QPE run (5e-4 where U9's optimum s* on that record at r = 1 is a filtered scheme or
has n_v ≥ 2, else 1e-6) and, where s* amplifies (aa, bbht, lt, lt_bbht, lt_bbht_L), its whole expected AA
sequence as one coherent run at 5e-4; and the longest property AE run of the best AE rows over the property
systems (p_fail_ae). A run that no factory and code distance keep within its budget is reported without a
layout.

## Cost model

### Systems and scaling (F20-F25)

**F20 (system names, builds and the per-system scan).** [design]; the CAS rule [derived]. Code:
`ft.symmetry.system_spec`, `ft.symmetry.benchmark_system`, `ft.symmetry.ALIASES`,
`integrals.ActiveSpace.from_pyscf`; `ft.scaling.scan_system`, `ft.scaling.scan_stage`, `ft.scaling.Cache`,
`ft.scaling.record_summary`; `benchmarks/run_m4_scaling.py`.

- Names outside `ft.symmetry.BENCHMARK_SYSTEMS` are built on M3's path (F2a: STO-3G, D2h symmetry-adapted
  RHF singlet, `systems.canonical_orbitals`; label = name): `h<n>_r<R>`, linear H_n with atoms at iR on z
  (bohr), all active; `n2_r<R>`, N2 at (0, 0, 0) and (0, 0, R) (Å) in CAS(6,6); `n2v_r<R>`, the same in
  CAS(10,8) with ncore 2; `h4rect_<a>x<b>`, H4 at (0, 0, 0), (a, 0, 0), (a, b, 0), (0, b, 0) (bohr). R, a
  and b must be plain positive decimals and n an even integer ≥ 2 without a leading zero (closed-shell
  singlets); signs, exponents, `inf`, other spellings and the M3 names raise ValueError in `system_spec`,
  and `benchmark_system` resolves the M3 names first, on their unchanged path (changed from the design
  specification; m4-results.md §13). The aliases h4 = h4_r1.8, h6s = h6_r3.0, h8 = h8_r1.8,
  n2 = n2_r1.098 and n2s = n2_r2.0 build bit-identical integrals, orbitals and irreps.
- CAS rule: ncore = (n_elec − nelecas)/2 with ncore + ncas ≤ n_MO, else ValueError (PySCF's CASCI would
  truncate silently); N2 CAS(8,8) is impossible in STO-3G. The chain orbitals alternate Ag and B1u, so the
  allowed count K of F2a depends on N alone (Appendix A). The chains are a controllable model of the
  overlap catastrophe, not targets; the square `h4rect_<a>x<a>` (B1g ground state, p_HF < 1e-24) tests the
  flag "ground_state_not_in_hf_irrep" (F21).
- `scan_system(name, cache=None, k_cost=3, n_steps=6, r_values=(0.3, 1, 3))`, on the spin-free elements,
  runs the stages of `ft.scaling.SCAN_STAGES`: the spectrum (F21); the QPE scan and selection (F9); the
  residual inputs (F24); 6-step records, exact L-BFGS with the spin-free and spin-orbital metric and the
  exact or second-order first step (F14, F16) and the R1 records r1_sf, r1_so (`steprules.run_step_rule`
  with `steprules.StepRule("unit")`, `ft.steprules.to_trajectory_record`), at N = 10 only the two
  second-order records and r1_so; the costs for k ≤ 3 at every r, F14 on the exact and F30 on the R1
  records, with the cheapest iteration per metric (F24); the compiled overlaps (F25, N ≤ 8). An iterate's
  residual problem is built once and costed at every r (`ft.residual.compare_routes` depends on r only
  through B), equal to `ft.pipeline.iteration_costs` and `ft.steprules.step_rule_iteration_costs` at
  each r field for field. At N ≤ 8 the first three stages are shared with `ft.closedloop.system_inputs`;
  an N = 10 record carries "no_model_spectrum", "direct_route_only", "subset_variances" and
  "no_compiled_overlaps" (`ft.scaling.SCAN_FLAGS`). The benchmark scans the 25 systems of the three axes;
  `record_summary`, `ft.scaling.scaling_report` and `ft.scaling.markdown_tables` give
  `scaling_<system>.json` and U1–U4, in strict JSON (`ft.scaling.to_json`: nan → null, ±inf → "inf").
- Stage cache, `Cache(root, refresh=False)`: one pickle per system, stage and digest, the SHA-256 of the
  build entry, every source file of the package (`ft.scaling.source_digest`), the numpy, scipy, pyscf and
  openfermion versions (`ft.scaling.dependency_versions`), the stage, a format number and the stage's
  parameters (changed from the design specification; m4-results.md §13); writes are atomic, `--no-cache`
  refreshes, `--stage` computes one stage per process.
- Tests: tests/test_ft_symmetry_names.py::test_parametric_names_match_m3_builds (h8 slow),
  ::test_aliases_cover_the_m3_names_with_a_pattern, ::test_system_spec_patterns,
  ::test_parametric_build_equals_a_direct_build, ::test_n2v_builds, ::test_cas_beyond_the_orbitals_raises,
  ::test_h4_square_ground_state_flag, ::test_unknown_names_raise, ::test_non_string_names_raise;
  tests/test_ft_symmetry.py::test_orbital_irreps_of_the_core_systems,
  ::test_allowed_element_counts_match_appendix_a; tests/test_ft_scaling.py::test_system_sets,
  ::test_scan_system_and_cache, ::test_scan_large_n_path, ::test_costs_by_r_equal_m3_and_f30,
  ::test_cache_roundtrip_and_selection, ::test_cache_writes_are_atomic, ::test_cache_digest_rules,
  ::test_to_json_is_strict.

**F21 (spectrum, gap and the hand-off window).** [derived]; the gap models [estimate]. Code:
`ft.scaling.spectrum`, `ft.scaling.Spectrum`, `ft.scaling.fit_gap`, `ft.scaling.GapFit`,
`ft.scaling.acceptance_margins`, `ft.scaling.certification_window`, `ft.scaling.d_l_one_run`,
`ft.scaling.two_run_overlap_floor`, `ft.scaling.certification_window_overlap`; U1, U2 by
`ft.scaling.scaling_report`.

- Spectrum: `integrals.ActiveSpace.fci(nroots=n_roots, spin_square=0, spin_tol=tol)` at 1e-6, M3's call
  (so `exact_trajectory(..., roots=spectrum.roots)` reproduces M3's records bit for bit), then 1e-4 (flag
  "spin_tol_relaxed"; only n2v_r2.4 of the 25 systems, ⟨S²⟩ up to 3.0e-5 [build check]; RuntimeError if
  neither admits n_roots singlets). N = 10 takes 2 roots, 12 with the flag "roots_extended" if root 1 is
  not in the ground irrep. E₁ is M3's same-irrep rule (`ft.trajectory.same_irrep_e1`, flag
  "e1_lower_bound"; None with one root), E₁,any is root 1, Δ₁ = E₁ − E₀ (`Spectrum.gap`, `Spectrum.gap_any`):
  the two gaps coincide on every chain at R ≥ 2.5 bohr [design] and differ at 1.8 bohr and on N2 at 1.098
  and 1.6 Å. "ground_state_not_in_hf_irrep" marks a ground irrep other than HF's (`ft.symmetry.ci_irrep`).
- Margins, `acceptance_margins(ΔE, sin θ₀, E₁ − E1lb, d_l=5)`, the smallest gap that admits a sound 1-mHa
  hand-off of the most favourable state: rule E m_E = 3ΔE (F15 with E_k → E₀); rule B in the window model
  m_B = 2ΔE + (E₁ − E1lb) (B_U of F40 with U → E₀); B_L m_BL = ΔE(1 + 2D_L sin θ₀) + (E₁ − E1lb), its
  threshold lying D_L = 5 grid units g ≈ 2ΔE sin θ₀ below E1lb (F39, F40). `scaling_report(records,
  sin_theta0=1.0, e1_margin=1e-3)` takes sin θ₀ = 1, the upper bound (no extrapolated size has a walk),
  and E₁ − E1lb = 1 mHa (F41 without a model spectrum): m_E = m_B = 3 mHa and m_BL = 12 mHa in U2 and F45;
  the design's m_BL ≈ 11.4 mHa (sin θ₀ ≈ 0.94) gives a 5% (A′/n) to 6% (power law) longer n_cert(m_BL)
  (changed from the design specification; m4-results.md §13).
- Gap models, `fit_gap(ns, gaps)` at fixed R through the two largest sizes with a same-irrep E₁:
  Δ₁ = An^−γ, and A′/n with A′ = n_maxΔ₁(n_max) [assumption: n·Δ₁ does not decrease beyond n_max; it rises
  over n = 4–10 (U2), and the gap of the stretched chain, a spin-½ Heisenberg chain at large R, closes as
  1/n; cited: des Cloizeaux and Pearson, Phys. Rev. 128, 2131 (1962)]. `certification_window(fit, m)`:
  n_cert(m) = [A′/m, (A/m)^{1/γ}], the largest n with Δ₁(n) ≥ m per model (γ ≤ 0: ∞ for a growing gap or a
  constant one ≥ m, 0 for a constant one below m). Beyond n_cert(m) F45 lengthens the hand-off QPE by
  f(n) = max(1, m/Δ₁(n)) (F9's 1/ΔE law).
- One run of B_L: D_L(p) = min{D ∈ ℕ : τ_W(D) ≤ ε_id p/(1 − p)} (`d_l_one_run`, F40; 0 at p = 1).
  `certification_window_overlap` solves Δ₁(n) ≥ ΔE(1 + 2D_L(p(n)) sin θ₀) + (E₁ − E1lb) jointly with F22's
  p(n) [derived]: D_L ≤ d iff p ≥ q_d = min_{d′≤d} τ_W(d′)/(ε_id + τ_W(d′)), so with a decaying overlap the
  level d holds on n ∈ (u_{d−1}, u_d], u_d = n_last + (ln p(n_last) − ln q_d)/α, and the window ends where a
  closing gap meets its level's margin or where D_L steps up; with a closing gap a constant overlap holds one
  level, D_L(p(n_last)), and a growing one ends at its first admitted level; a gap that does not close (γ ≤ 0)
  is resolved level by level (the window ends at u_d of the largest admitted level) and, with an overlap that
  does not decay, gives ∞ (the gap reaches its large-n margin) or 0; D_L > 1000 raises ValueError. The two-run
  window n_cert(m_BL) holds for every state with p_k(n) ≥ τ_W(5)²/(ε_id + τ_W(5)²) = 3.37e-5
  (`two_run_overlap_floor`); U2's side table gives the one-run windows, p at the ends of the two-run window
  and that window capped at the floor. D_L(p) and the floor, like F40's n_v, take K₀ = 1 (F40).
- U1: N, determinants, p_HF, E_HF − E₀, Δ₁ (same / any irrep), the selected encoding, C_QPE, Q_QPE, rule-E
  k_cert on the four exact records, flags; the lowest same-irrep level of any spin of H′ (N ≤ 8) is in the
  U1 supplement of `benchmarks/run_m4_summary.py` (`ft.certify.sector_levels`, F41) (changed from the
  design specification; m4-results.md §13).
- Tests: tests/test_ft_scaling.py::test_spectrum_gaps_and_overlaps, ::test_spectrum_reproduces_m3_records,
  ::test_spectrum_flags_and_few_roots, ::test_spectrum_validation, ::test_spin_tolerance_fallback (slow),
  ::test_gap_window, ::test_one_run_margin, ::test_one_run_window_jointly_with_overlap,
  ::test_one_run_window_growing_gap, ::test_u2_fit_window_p1_and_one_run_column, ::test_regression_pins
  (slow), ::test_scaling_report_fits (slow).

**F22 (per-site overlap model).** [derived] for matrix-product states, [assumption] for HF and the CQE
iterates. Code: `ft.scaling.successive_rates`, `ft.scaling.fit_log_overlap`, `ft.scaling.RateFit`,
`ft.scaling.extrapolate_overlap`, `ft.scaling.size_at_overlap`; U2.

- For two uniform matrix-product states of an open chain (bulk tensors A^σ, B^σ repeated every m atoms,
  boundaries absorbed in l and r), each normalized so that its own transfer matrix Σ_σ Ā^σ ⊗ A^σ,
  Σ_σ B̄^σ ⊗ B^σ has leading eigenvalue 1, ⟨ψ₀|Φ⟩ = lᵀE^{n/m}r with the mixed transfer matrix
  E = Σ_σ Ā^σ ⊗ B^σ; if its leading eigenvalue λ₁ is non-degenerate and l, r overlap its eigenvectors,
  ln p(n) = a − αn + O(|λ₂/λ₁|^{n/m}) with α = −(2/m) ln|λ₁| ≥ 0 (|λ₁| ≤ 1 by Cauchy–Schwarz) [derived]. HF
  and the CQE iterates (finite products of exponentials of extensive two-body generators on a gapped
  determinant) are assumed to be such states [assumption]: the orthogonality catastrophe of approximate
  many-body states [cited: W. Kohn, Rev. Mod. Phys. 71, 1253 (1999), the "exponential wall"]. Rates
  are per atom, p is the weight on ψ₀ of H.
- `fit_log_overlap(ns, ps, window=(6, 8, 10))`: successive rates α^(n_i) = [ln p(n_{i−1}) − ln p(n_i)]/(n_i −
  n_{i−1}) (`successive_rates`), drift δ = α^(n_last) − α^(n_prev), and the Aitken limit
  α^A = α^(n_last) + δq/(1 − q), q = δ/δ_prev, if 0 < q < 1 (exact for geometric drifts), else α^(n_last) + δ
  (flag "aitken_fallback", also with two rates, as for n = 4, 6, 8); one rate gives α^A = α^(n_last) (flag
  "no_drift"). The least-squares (a, α) on the window is reported only.
- Extrapolation: p(n) = p(n_last)e^{−α(n − n_last)} at both ends of the bracket [α^(n_last), α^A]
  (`ft.scaling.RATES`), capped at 1; `size_at_overlap` inverts it (∞ if never reached). Every value beyond
  n = 10 is a model statement and its range a model bracket, validated over Δn = 2 by the holdout, with an
  unquantified model risk over Δn ≥ 20, never a confidence interval; it covers H chains in canonical
  orbitals with elementwise compilation and the full allowed residual.
- U2 series per R: HF; the classical Ψ₁ of both metrics; the R1 Ψ₂ of r1_so and, for F45's spin-free α₂, of
  r1_sf at n ≤ 8 (changed from the design specification; m4-results.md §13); with p, the rates, (a, α), α^A,
  the holdout errors, n(p = 1e-3) and the classical p₁ there. A holdout is every n with three smaller sizes
  (n = 10): error = p_extrapolated(n)/p_computed(n) − 1 at both rates, the fit taken on the smaller sizes
  alone (`ft.scaling.scaling_report`). The optional n = 12 holdout (`ft.scaling.HOLDOUT`) is not run (no
  `--holdout-h12`) (changed from the design specification; m4-results.md §13).
- Tests: tests/test_ft_scaling.py::test_rate_fits_synthetic, ::test_rate_fit_validation,
  ::test_rates_on_chains, ::test_holdout_h10_from_h8 (slow), ::test_u2_fit_window_p1_and_one_run_column,
  ::test_scaling_report_fits (slow).

**F23 (cost components against N).** [estimate]. Code: `ft.scaling.fit_power_law`, `ft.scaling.PowerFit`;
U3.

- y(N) = y_last(N/N_last)^β with the successive exponents β^(N_i) = ln(y_i/y_{i−1})/ln(N_i/N_{i−1}), their
  last value β_last and mean β_mean, at fixed R on the chains with N ≥ 6 (the selected encoding switches
  from SF to THC between N = 4 and 6), for C_QPE, λ, C_W, K, V_m, W_m, C(Ψ₁), C(Ψ₂), ‖A₁‖_m and
  T_res(1)/C_QPE at r = 0.3, 1, 3 per metric on its R1 record (sf at N = 10 on lbfgs/second_order, whose Ψ₁
  is R1's; the exponent row names the records).
- Where direct_mg is the best route at both sizes (every row of U3), T_res(1)/C_QPE =
  V_mC(Ψ₁)/(r²‖A₁‖²_mC_QPE), so β(T_res(1)/C_QPE) = β(V_m) + β(C(Ψ₁)) − 2β(‖A₁‖_m) − β(C_QPE) at every r
  [derived]. V_m grows fastest: a three-body term of X_e on 2N modes has the matchgate-shadow factor
  C(4N, 6)/C(2N, 3) → 64N³/15 [derived; the shadow norm cited: Zhao, Rubin, Miyake, PRL 127, 110504
  (2021)], which with β(K) ≈ 4.1 accounts for β ≈ 7 of the observed ≈ 8 [estimate]. Since the variance
  state changes along N (F24), β(V_m) over N = 6 → 8 compares two states (1.4% moves it by 0.05).
- C_iter is never fitted (it mixes components and route switches). For R1, C_iter(1) = T_res(1) (F30), so
  F45's c_iter(1) is the power law of T_res(1)/C_QPE on N ≥ 6 anchored at the last cost point (the R1
  record's, or the cheapest iteration at the same Ψ₁, F24), with β ∈ {β_last, β_mean, 9} and, beyond
  n_c = 20, a crossover to β_loc ∈ {2, 4} [estimate; no data beyond n = 10]; C(Ψ₁)/C_QPE and C(Ψ₂)/C_QPE are
  power laws anchored at their last size.
- Tests: tests/test_ft_scaling.py::test_power_law_fit, ::test_scaling_report_tables,
  ::test_scaling_report_fits (slow), ::test_n_star_inputs_from_records, ::test_regression_pins (slow);
  tests/test_ft_acceptance.py::test_model_spectrum_reference_numbers.

**F24 (residual inputs at N ≥ 8; the cheapest measured iteration).** [estimate]; the subset estimator
[derived]. Code: `ft.closedloop.system_inputs` (variance state), `ft.residual_inputs.element_variances`,
`ft.residual_inputs.variance_totals_subset`, `ft.scaling.cheapest_iteration`.

- Variance state (`ft.closedloop._default_variance_state`): "iterate1" for N ≤ 6, M3's v_e, w_e at Ψ₁ of
  the exact spin-free L-BFGS record with the exact first line search, so M3's numbers are reproduced
  (F10); "hf" for N ≥ 7: (i) at N = 8 `element_variances` at HF on every allowed element (Rao–Blackwellized
  above F10's 5e7 term pairs), (ii) at N = 10 subset totals at HF. The DF split is the lowest at HF (F13);
  LCU normalizations α_e exist at N ≤ 8 only.
- (i): the totals depend weakly on the state, within 0.2–1.4% (V_m) and 0.1% (W_m) against Ψ₁ and Ψ₃ on
  h4_r3.0, h4_r4.0, h6_r3.0, h6_r4.0, up to 22% per element [design]; the AE routes use α_e, so the HF
  state enters the sampling routes only.
- (ii), `variance_totals_subset(..., k_subset=100, seed=11)`: a uniform subset S of K_s = 100 of the K
  elements, V̂_m = (K/K_s)Σ_{e∈S} m_e v_e with the finite-population standard error
  (K/K_s)√K_s sd(m_e v_e)√(1 − K_s/K), the same for Ŵ_m; unbiased over uniform subsets, the subset's own
  Rao–Blackwellized error reported apart (`stderr_m_estimator`). The routes receive uniform
  v_e = V̂_m/Σ_e m_e and w_e = Ŵ_m/Σ_e m_e, which reproduce direct_mg and aux_mg exactly (both see the
  totals only) and are no per-element estimates (h10_r3.0, full run: sampling error 2.2%).
- At N = 10 every residual is costed by the closed-form direct route, F10 without biases:
  S = V̂_m/(r²‖A_k‖²_m) preparations of C(Ψ_k) with the exact A_k (`measure.routes.exact_residual`), equal
  to M3's direct_mg, ∝ r⁻², infinite only at ‖A_k‖_m = 0.
- `cheapest_iteration(..., k=1, r_values=(0.3, 1, 3), routes="all")`: the residual of iterate k, at k = 1
  after the classical step R1's whole C_iter(1) (F30); "all" is every route of `ft.residual.compare_routes`
  with the aux-bias curve and the factorization bias, best at κ = 1 (`ft.residual.best_route`), "direct"
  the closed form. It feeds U3, U4's c_iter(k) and F45; (i) also gives the N = 8 closed-loop inputs.
  T_res(1)/C_QPE at r = 1 (U3) is 0.515 / 0.423 / 0.398 on n2_r2.0 / 2.4 / 2.8 (so), by direct_mg; the
  design's 0.43 and 0.41 for n2_r2.4 and n2_r2.8 used HF-state variances at every N, while M3's iterate-1
  variances are kept at N ≤ 6 (changed from the design specification; m4-results.md §13).
- Tests: tests/test_ft_scaling.py::test_hf_variances, ::test_subset_totals_formula,
  ::test_subset_totals_full_set_and_mean, ::test_subset_totals_unbiased (slow),
  ::test_subset_totals_validation, ::test_cheapest_iteration_routes, ::test_costs_by_r_equal_m3_and_f30,
  ::test_scan_large_n_path; tests/test_ft_closedloop.py::test_default_variance_state,
  ::test_n8_inputs_use_hf_variances (slow).

**F25 (sliced elementwise compilation).** [derived; the 1/s law measured]; the slice rule [design]. Code:
`ft.scaling.compiled_overlaps`, `ft.circuits.compiled_step`, `ft.stateprep.elementwise_step_cost`.

- F3's step is a first-order product formula, Π_a e^{εX_a} = exp(εD + (ε²/2)Σ_{a<b}[X_a, X_b] + O(ε³)),
  whose O(ε²) error F3 flags and does not budget. s consecutive F3 circuits at ε/s err by O(ε²/s)
  [derived] at the cost s·C_step(d, ε/s) on one slice's Q_prep; each slice has its own F2 bits and
  δ_step, so the rounding adds up to sδ_step plus the dropped angles.
- `compiled_step(active, Ψ̃, d, ε, irreps=..., rounded=True)` is the loop body of
  `ft.circuits.replay_elementwise` (kept excitations in decreasing angle, applied exactly with b-bit
  rounded angles on the Fock-space oracle, N ≤ 8); s slices of a step from HF equal
  `replay_elementwise(active, [d]*s, [ε/s]*s)`. `compiled_overlaps(active, record, psi0, k_max=3,
  slices=(1, 2, 4))` gives the open-loop compiled iterates of a record per s, with p̃_k, ‖Ψ̃_k − Ψ_k‖, the
  step error ‖Ψ̃_k − e^{ε_{k−1}D_{k−1}}Ψ̃_{k−1}‖, ⟨S²⟩, the norm deficit and the costs (r1_so and the two
  second-order records, k ≤ 3; `scaling_<system>.json` only).
- Slice rule [design: an emulation-side calibration at N ≤ 8, not realizable, since it compares every
  slicing with the exact e^{εD}Ψ̃ in CI space]: from the rule's own current state, the smallest s in
  `ft.scaling.SLICE_RULE` = (1, 2, 4, 8, 16) with ‖Ψ̃^(s) − e^{ε_kD_k}Ψ̃_k‖ ≤ η_s = 0.05
  (`ft.scaling.ETA_SLICE`), else s = 16 with the flag "slice_cap"; it drives the closed loop's compiled
  propagation (F37) and the sliced property iterates (F49).
- The classical step overshoots on the stretched chains (ε₂/ε* = 1.598 for h8_r4.0 sf) and compiles
  poorly (‖Ψ̃₁ − Ψ₁‖ = 0.49–0.81, ⟨S²⟩ up to 0.87 at s = 1 and R = 4.0, quick run [design: 0.16–0.81 at
  R ≥ 3.5]): the rule takes s = 16 and still misses η_s on h6_r4.0 and h8_r4.0 (so, "slice_cap"). It
  bounds the step error, not the error against the exact record, which accumulates (0.051, 0.078, 0.081 at
  k = 1, 2, 3 on h6_r4.0 r1_so). The R1 spin-orbital steps on N2 at 2.0–2.8 Å need no slicing (quick run).
- Tests: tests/test_ft_scaling.py::test_slicing_law, ::test_slice_rule_continues_to_16,
  ::test_slice_rule_follows_its_own_state, ::test_scan_system_and_cache;
  tests/test_ft_closedloop.py::test_compiled_step_equals_replay,
  ::test_sliced_compiled_step_equals_replay_of_slices, ::test_compiled_step_validates_the_state,
  ::test_compiled_loop.

### Step rules (F26-F31)

**F26 (the L-BFGS model step is the unit step).** [derived]; the quasi-Newton model and its two-loop recursion
[cited: Nocedal and Wright, Numerical Optimization, 2nd ed., Springer (2006), Algorithm 7.4], applied in the
local coordinates of the iterate as in the quasi-Newton CQE [cited: Smart, Boyn, Mazziotti, J. Chem. Theory
Comput. 18, 5286 (2022)]; model steps of SD and CG directions [cited: Barzilai and Borwein, IMA J. Numer.
Anal. 8, 141 (1988)]. Code: `steprules.run_step_rule` (rule "unit"), `steprules.StepRule`,
`steprules.StepRuleResult`, `optimizers.LBFGS.inverse_hessian_apply`.

- The loop: `run_step_rule` is the measured CQE on the exact CI-space harness, from HF by default. At
  iterate k the residual A_k masked to the allowed entries (F2a), or the estimate of the hook
  `residual_estimator` (with A's structure to 1e-10, optionally with variances V_e); d_k = H_kg_k,
  g_k = conj(Â_k), of M2's L-BFGS (memory 10, curvature tolerance 1e-2, H₀ = γM, M the spin-free or
  spin-orbital metric); the rule's ε_k; Ψ_{k+1} = exp(ε_kD_k)Ψ_k by `ft.trajectory.propagate` (Taylor series
  at 1e-15) or by the hook `propagator` for every state the loop prepares (F37).
- Rules (`steprules.STEP_RULES`): "unit" (R1, ε_k = 1 under F29), "parab3" (F27), "hybrid" (R3, F31),
  "line_search" (the exact line search along d_k, `cqe.line_search_step` on [0, 1.5/‖D_k‖] at Krylov
  tolerance 1e-10: the oracle step, M3's trajectory model) and "candidates" (M3's grid of F13, each energy
  at eps_ls = 1.6 mHa, the lowest estimate taken, the first on ties). At k = 0 every rule takes, with
  `first="second_order"` (default), the classical step ε₂ = −E′(0)/E″(0) of F16 along its d₀, unmeasured
  and uncapped (E″(0) ≤ 0 and a start that is not a determinant raise ValueError); `first="line_search"`
  measures A₀ and searches exactly: M3's measured loop.
- The model step: with the slope s₀ = dE/dε|₀ = −⟨g_k, d_k⟩ (`optimizers.direction_slope`),
  ⟨d_k, H_k⁻¹d_k⟩ = ⟨g_k, H_kg_k⟩ = −s₀, so the model m(ε) = E_k + εs₀ + ε²⟨d_k, H_k⁻¹d_k⟩/2 has its
  minimizer at ε_QN = 1 and predicts ΔE_m = s₀/2. Only the symmetry and positive definiteness of H_k
  enter, so this holds for estimates and for an empty memory (d = Mg). At k = 1 the memory holds the
  classical step's pair if it passes the curvature test; after an L-BFGS reset (flag "lbfgs_reset") the
  unit step is the unscaled metric step, bounded only by F29. A model step other than 1 needs SD or CG
  directions (Barzilai–Borwein): unit, parab3 and hybrid require `optimizer="lbfgs"` (ValueError).
- `StepRuleResult.de_model`, a field beyond the design's result (changed from the design specification;
  m4-results.md §13), holds ΔE_m = ŝ₀′/2 (the debiased slope of F33 where the estimator returns variances,
  flag "slope_debias_approx" while L-BFGS holds pairs; ŝ₀ otherwise; −E′(0)²/(2E″(0)) at a classical k = 0);
  U5's σ_A, σ_P3 and ρ_E read it (F27, F28). The run stops before a step whose predicted lowering is below
  1e-10 Ha (F16's rule; `StepRuleResult.converged`), after a zero step that repeats the previous one (without
  estimators), or after `max_iter`.
- Equivalences (Appendix A): the unit rule with the caps off and the masked exact residual is
  `cqe.run_cqe(..., optimizer="lbfgs", first_eps=ε₂, eps_candidates=(1.0,))`; the line-search rule with
  the classical first step is `ft.trajectory.exact_trajectory(..., first_step="second_order")` up to the
  round-off of the flat late line searches, and with `first="line_search"` M3's exact record bit for bit.
- Tests: tests/test_steprules.py::test_model_step_is_unit, ::test_unit_rule_equals_run_cqe,
  ::test_line_search_rule_equals_exact_trajectory, ::test_measured_loop_equals_m3_exact_record,
  ::test_line_search_step_is_the_search_of_run_cqe, ::test_inverse_hessian_apply,
  ::test_step_rule_validation, ::test_estimator_and_propagator_hooks, ::test_compiled_propagator_hook,
  ::test_debiased_slope_flags_in_the_loop; tests/test_ft_steprules.py::test_r1_iteration_counts (slow).

**F27 (precision of an energy that decides or fits a step).** [derived]; σ_P3 = 0.47|ΔE_m| from a
Monte-Carlo calibration [design], recomputed by a slow test; the sufficient-decrease test [cited:
Armijo, Pacific J. Math. 16, 1 (1966)]. Code: `ft.steprules.energy_precision`,
`ft.steprules.PRECISION_TABLE`, `ft.steprules.fit_losses`, `ft.steprules.build_precision_table`,
`steprules.parabola_step`.

- The 3-point fit of Ê₀, Ê₁, Ê₂ at Ψ_k(0), Ψ_k(t), Ψ_k(2t), Ψ_k(ε) = exp(εD_k)Ψ_k: c = (Ê₀ − 2Ê₁ + Ê₂)/t²,
  b = (−3Ê₀ + 4Ê₁ − Ê₂)/(2t), x = −b/c. `parabola_step` (interpolation only) takes x if c > 0 and
  0 < x ≤ 2t, floored at 0.05t (status "floor"), else the lower measured point (t on ties), or 0.25t if Ê₀
  is the lowest ("clipped") [design]; the extrapolating variant (used by no rule) clips to [0.05t, 3t]. A
  status other than "fit" becomes the flag `parab:<status>`; non-finite energies or t ≤ 0 raise
  ValueError. parab3 (and hybrid where its gate opens) measures the energies at
  σ_P3(k) = 0.47|ΔE_m(k)| on {0, t, 2t}, t = `parab_t` = 1, so that the model step sits at the centre
  (t = cap/2 where F29 caps the far point), through the hook `energy_estimator` (exact by default).
- Linearized [derived]: for independent errors of RMS σ and u = ε*/t, (δε/ε*)² = σ²u²f(u)/(4ΔE\*²),
  f(u) = (u − 3/2)² + 4(1 − u)² + (u − 1/2)²; a step ε̂ loses the fraction ℓ = (ε̂/ε* − 1)² of the decrease,
  so at u = 1 ℓ = 0.1 needs σ = √(8ℓ)|ΔE_m| = 0.894|ΔE_m| ("parab3_linear").
- Calibrated: at these precisions the fitted curvature is often ≤ 0 and the clip binds (the loss is 0.264
  at 0.894|ΔE|). `fit_losses` computes ℓ(σ) through `parabola_step` on an exact parabola with 4e5 draws and
  common random numbers, and `build_precision_table` inverts it into `PRECISION_TABLE`
  (ℓ ∈ [0.005, 0.3], log-linear interpolation, ValueError outside); σ_P3 = 0.47|ΔE_m| at ℓ = 0.1 is fixed
  (`SIGMA_P3`; the Monte-Carlo inverse is 0.468).
- Armijo [derived], reported, not run: energies at 0 and at the model step, accepted iff Ê(1) − Ê(0) ≤ c₁ŝ₀;
  on a parabola it rejects the model step with probability Φ(−(1 − 2c₁)|ΔE_m|/(√2σ)), so a false rejection
  Φ(−z) needs σ_A = (1 − 2c₁)|ΔE_m|/(√2z) = 0.3535|ΔE_m| at z = 2, c₁ = 1e-4. 2-point fit [derived], reported,
  not run: E(0), E(t) and the slope, ε = −s/c clipped to [0.05t, 3t] (3t where the fitted c ≤ 0); linearized,
  (δε/ε*)² = 2σ²u⁴/ΔE*², so at u = 1 ℓ = 0.1 would need σ = √(ℓ/2)|ΔE_m| = 0.224|ΔE_m|, but the clip and
  non-positive fitted curvatures bind as for the 3-point fit, and the Monte-Carlo inverse (`PRECISION_TABLE`)
  gives σ_P2 = 0.167|ΔE_m| (the design's 0.16|ΔE_m| loses 0.088), returned by `energy_precision("parab2")`
  only, while U5 reports σ_A and σ_P3 (changed from the design specification; m4-results.md §13).
- `energy_precision(kind, ΔE_m; c1=1e-4, z=2, loss=0.1)`: kinds "parab3", "parab3_linear", "parab2",
  "armijo"; the sign of ΔE_m does not matter, ΔE_m = 0 gives 0. U5 reports σ_A and σ_P3 at k = 1 and
  k_ref − 1 (F30).
- Tests: tests/test_steprules.py::test_parabola_step, ::test_parab3_rule_with_exact_energies,
  ::test_parab3_stencil, ::test_event_ledger, ::test_f27_monte_carlo (slow);
  tests/test_ft_steprules.py::test_energy_precision_values, ::test_fit_losses_use_parabola_step,
  ::test_precision_table_monte_carlo (slow).

**F28 (cost of the energies; the energy-precision wall).** [derived]. Code: `ft.steprules.energy_costs`
(`ft.residual.energy_sampling`, `ft.residual.energy_ae`, `ft.stateprep.candidate_step_costs`); ρ_E in
`ft.steprules.run_step_rule_system`.

- Sampled: n energies of Ψ_k(ε_i) at precision σ take S_E = V_E/σ² shots each of
  C(Ψ_k) + C_step(k; ε_i) + C_rot, T_en = S_E Σ_i(C(Ψ_k) + C_step(k; ε_i) + C_rot), on
  max(Q(Ψ_k), max_i Q_prep(ε_i), Q_rot) qubits: F13 with σ in place of eps_ls, bit-identical to
  `ft.residual.line_search_cost` at σ = eps_ls; each energy is one F19 sampling campaign. `energy_costs`
  takes `step_qubits` and `norb` beyond the design's signature; without `step_qubits` it reports
  max(Q(Ψ_k), Q_rot) (flag "qubits_lower_bound") (changed from the design specification; m4-results.md §13).
- Every energy of iteration k, stencil end points and Armijo pair included, is charged at V_E(Ψ_{k+1}) of the
  split chosen at HF (M3's F13 convention), which makes ρ_E(parab3) a lower estimate where V_E at the
  stencil's points exceeds V_E(Ψ_{k+1}), as at small gaps (mean V_E of {0, 1, 2}: 2.09, 1.45, 1.05
  V_E(Ψ_{k+1}) at k = 1, 2, 3 on h4_r4.0, spin-orbital; 1.03 to 1.00 on h4_r1.8 [build check]).
- AE option (F13's): per energy N_E = 2πκλ/σ applications of C(Ψ_k) + C_step(k; ε_i) + C_W + Q_A/2,
  Q_A = max(Q(Ψ_k), Q_prep(ε_i)) + Q_walk − 2N + 1, one coherent run on Q_A + ⌈log₂(N_E/2)⌉ qubits
  (without `norb` the system register counts twice, flag "qubits_upper_bound"); it estimates ⟨H′⟩ with an
  uncounted bias and is reported as `line_search_ae`, outside C_iter.
- Against the direct residual (F10): ρ_E = T_en/T_res,direct =
  n_E(V_E/V_m)(r²‖A_k‖²_m/σ²)(C(Ψ_k) + C̄_step + C_rot)/C(Ψ_k). In the quadratic regime
  ‖A_k‖²_m ≈ 2λ̄_k(E_k − E₀) (λ̄_k a Rayleigh quotient of the local Hessian) and σ = cω_k(E_k − E₀) with
  c = σ/|ΔE_m| (F27) and ω_k = |ΔE_m,k|/(E_k − E₀), so ρ_E ∝ r²/(E_k − E₀) at fixed λ̄_k and ω_k: every
  energy check loses against the residual as the CQE converges, the more so at loose r (the
  energy-precision wall). ρ_E = ∞ if T_res,direct = 0, None if σ = 0.
- Against QPE [derived from F9 and F13]: with C_QPE = C_Wπλ/(2ΔE), one AE energy costs
  ≥ 4κ(ΔE/σ)(1 + C(Ψ_k)/C_W) C_QPE and one Armijo energy ≥ 11.3κ(ΔE/|ΔE_m|)(1 + C(Ψ_k)/C_W) C_QPE: below
  |ΔE_m| = 11.3κΔE an amplitude-estimated decision energy costs more than the QPE it prepares for.
- U5 reports ρ_E of the Armijo pair (at 0 and the step taken) and of the 3-point fit at k = 1 and
  k_ref − 1 against direct_mg at r = 0.3, 1, 3 (R3 at its planning r).
- Tests: tests/test_ft_steprules.py::test_energy_costs_follow_f28,
  ::test_energy_rules_are_charged_by_their_ledger, ::test_run_step_rule_system.

**F29 (safeguards).** [design]. Code: `steprules.StepRule` (`growth`, `angle_cap`), `steprules.run_step_rule`.

- Growth cap: ε_k ≤ γ_s|ε_{k−1}|‖d_{k−1}‖_F/‖d_k‖_F, γ_s = `growth` = 2 (> 1; None: off), flag "growth_cap",
  against the previous nonzero step (at k = 1 ε₂d₀; a zero step, flag "zero_step", leaves the reference
  unchanged, where the design's previous accepted step would cap every later step at 0) (changed from the
  design specification; m4-results.md §13). Angle cap: ε_k‖D_k‖ ≤ `angle_cap` = 3 (None: off), ‖D_k‖ from 6
  power iterations on the CI space (`cqe.generator_norm_estimate`), an estimate from below, so ε_k‖D_k‖ ≤ 3
  holds only approximately; flag "angle_cap": an emulation safeguard, since ‖D_k‖ is not measured on hardware
  (the closed loop counts a binding cap as non-realizable, F32, and reruns a deciding cell without it, F38).
- Both act at k ≥ 1 before any energy is measured, on the unit step (ε_k = min(1, caps)) and on the
  stencil's far point (2t = min(2·`parab_t`, caps)), so the fitted step obeys them too; the flag names the
  binding cap (both on a tie). They act on the model rules only: ε₂ and the references line_search and
  candidates stay uncapped, so that the references remain M3's trajectory models (changed from the design
  specification; m4-results.md §13).
- Binding [build check]: on exact 20-step unit runs of h4_r{1.8, 3.0, 4.0} and n2_r{2.0, 2.4}, both metrics,
  the growth cap binds in 2–6 of 19 steps, first at k = 4–12, on the chains mostly after k*; the angle cap
  never binds there, and once in the 1146 steps the caps act on in the quick U5 run (unit and hybrid, k ≥ 1),
  on R3's stencil (n2_r2.4 spin-free, R3 planned at r = 3, k = 36), whose far point 2t is capped before the
  unit step is.
- Tests: tests/test_steprules.py::test_growth_cap_binds_and_flags, ::test_angle_cap_flags,
  ::test_parab3_stencil_obeys_the_growth_cap, ::test_parab3_stencil_obeys_the_angle_cap,
  ::test_step_rule_validation; tests/test_m4_loop_benchmark.py::test_angle_cap_rerun_decides.

**F30 (iteration cost of a step rule).** [derived]; it replaces T_LS of F14 for a rule R. Code:
`ft.steprules.step_rule_iteration_costs`, `ft.steprules.to_trajectory_record`,
`ft.steprules.run_step_rule_system`; U5 by `benchmarks/run_m4_loop.py`.

- The ledger `StepRuleResult.events` (`steprules.StepEvent`: iteration, kind, ε, σ, used_for_step): one
  residual at Ψ_k per measured iteration and one entry per energy: none for unit, three for parab3 (and
  hybrid where its gate opens), eleven on M3's grid at eps_ls for candidates and for line_search, which is
  charged as M3 charges an exact line search (F13, F14; the oracle search's own energies are not).
- C_iter^R(k) = T_res(k; r) + Σ_{trials j} T_res(Ψ_k(ε_j); r) + T_en(k), C_iter^R(0) = 0 for the classical
  first step (its ledger must be empty; ValueError), C_loop^R(k) = Σ_{j<k} C_iter^R(j). T_res(k; r) is
  F14's residual (the cheapest route at κ = 1 of `ft.residual.compare_routes`, with the system's fixed
  variances, the selected walk and its factorization bias, at `params.r` = r) at C(Ψ_k), Q(Ψ_k) of the
  record; a trial residual at ε_j ≠ 0 costs C(Ψ_k) + C_step(k; ε_j) per preparation (supported, taken by
  no rule); T_en(k) is F28 at the iteration's one σ (ValueError otherwise). R1 costs its residual alone, in
  one campaign without a readout rotation, so the Q_rot that set the qubits of every M3 line search (F4)
  leaves the loop; line_search on its own record is F14's C_iter bit for bit.
- Output: one `ft.pipeline.IterationCost` per k (the energies as one RouteCost "energy_sampling", `ls_flagged`
  for steps with F3's "rounding bound above delta_step", the F19 campaigns), so `ft.pipeline.loop_totals`,
  `ft.pipeline.machines_for_campaigns`, `ft.handoff.handoff_table` and `ft.handoff.optimal_handoff` apply
  unchanged; "no layout within the failure budget" or a critical path beyond 7 days (late h6_r4.0
  iterations of spin-orbital R1 at r = 0.3) gives `machines_7d` None.
- Records: `to_trajectory_record` turns a run with `keep_states=True` into an
  `ft.trajectory.TrajectoryRecord` (p_k against the spectrum's ψ₀, E₁ of both irreps, the irrep impurity);
  run and record must agree (ValueError). With `first="line_search"` it is M3's exact record bit for bit;
  with the classical first step it differs from `exact_trajectory(first_step="second_order")` by the
  precision of the line searches (≤ 1.0e-11 Ha [build check]), so the hand-off rows agree to 1e-9
  relative except E_k − E₀ (1e-10 Ha) and saving_vs_hf (1e-9 C_QPE/p₀) (changed from the design
  specification; m4-results.md §13).
- U5 (`run_step_rule_system`; m4-results.md §5), on h4_r{1.8, 2.5, 3.0, 3.5, 4.0}, h6_r{1.8, 2.5, 3.0, 3.5,
  4.0}, h8_r{1.8, 3.0} and the overlap axis (`benchmarks/run_m4_loop.py`, `U5_SYSTEMS`, `U5_QUICK`): per
  system, metric and rule from HF with the classical first step, 40 steps (30 at N = 8), R3 once per
  planning r, on the inputs of `ft.closedloop.system_inputs` (F32); k*
  (E_k − E₀ ≤ 1.6 mHa), k_p90 (p_k ≥ 0.9), k_ref = k* or else k_p90; costs at r = 0.3, 1, 3 to k* + 2 (to 20
  where k* is not reached, at most 12 at N ≥ 8). Columns: k* (k_p90); the ledger's extra events per costed
  iteration; C_iter(1..3)/C_QPE; C_loop(k_ref)/C_QPE, marked (k_p90) where k_ref = k_p90 (changed from the
  design specification; m4-results.md §13); σ_A, σ_P3 and ρ_E at k = 1 and k_ref − 1.
- Tests: tests/test_ft_steprules.py::test_m3_costing_reproduced, ::test_costs_at_several_r,
  ::test_unit_cost_is_the_residual, ::test_energy_rules_are_charged_by_their_ledger,
  ::test_record_conversion_is_m3s_measured_record, ::test_record_conversion_feeds_handoff,
  ::test_conversion_and_cost_validation, ::test_run_step_rule_system,
  ::test_u5_serializes_iterations_without_a_machine_count,
  ::test_u5_null_machine_counts_on_a_stretched_chain (slow), ::test_u5_reference_iterate_falls_back_to_p90,
  ::test_u5_reference_iterate_n2 (slow), ::test_cost_horizon, ::test_u5_shared_residual_problems_change_nothing;
  tests/test_steprules.py::test_event_ledger, ::test_candidates_rule_takes_the_lowest_estimate;
  tests/test_m4_loop_benchmark.py::test_u5_one_calculation_per_metric_rule_and_r, ::test_u5_table,
  ::test_finished_systems_survive_a_later_failure.

**F31 (hybrid switch R3).** [design]. Code: `steprules.run_step_rule` (rule "hybrid", hook `use_energies`),
`ft.steprules.planned_energy_gate`, `ft.steprules.PlannedEnergyGate`; in the closed loop
`ft.closedloop.LoopEnergyGate` (F36).

- At k ≥ 1, before any energy of the iteration is measured, the hybrid rule calls
  `use_energies(k, ΔE_m, rd, history)` (ΔE_m = ŝ₀′/2, rd a copy of the exact ResidualData of Ψ_k,
  history = (d₀ … d_k, ε₀ … ε_{k−1}) as read-only views, the current direction last) (changed from the
  design specification; m4-results.md §13): True takes the 3-point fit (flag "hybrid:parab3"), False the
  unit step ("hybrid:unit"), both under F29; without the hook ValueError.
- The gate (`PlannedEnergyGate`) plans T_en^plan, F28's three energies at σ_P3(k) on the nominal stencil
  {0, t, 2t} (the step costs along d_k by `ft.stateprep.candidate_step_costs`, C(Ψ_k) and Q(Ψ_k) compiled
  incrementally from the history by `ft.stateprep.elementwise_step_cost`), and T_res^plan, F30's residual
  at the planned budget; it fits iff T_en^plan ≤ θ_R3T_res^plan, θ_R3 = `theta` = 1 (θ_R3 ∈ (0, ∞]; σ = 0
  plans ∞), so an opened iteration costs at most (1 + θ_R3)T_res^plan at planned costs. Every decision is
  logged (k, σ, the planned V_E, T_en^plan, T_res^plan, use). The gate rebuilds Ψ_k from the history and the
  start state (HF or `ci0`); `PlannedEnergyGate.plan` takes a state directly.
- The planned V_E is a constant (Ha²) or a function of the state, evaluated at the planned step
  exp(tD_k)Ψ_k, and `planned_energy_gate` also takes `sigma_parab`, `parab_t` and `ci0` (changed from the
  design specification; m4-results.md §13). U5 plans at the planned step, near the Ψ_{k+1} at which F30
  charges; the realizable closed loop plans with the constant V_E(HF) (F36). The plan does not foresee the
  F29 caps, which shorten the stencil, nor where the fit lands.
- U5 runs R3 once per r ∈ {0.3, 1, 3}, since the gate plans at r, and costs it at that r; `gate_log` adds
  the charged T_en (`t_energy_charged`, 0 where the unit step was kept) and its V_E. U7's step-rule block
  runs R3 in the closed loop (F36).
- Tests: tests/test_steprules.py::test_hybrid_gate_switches_at_theta, ::test_event_ledger;
  tests/test_ft_steprules.py::test_planned_gate, ::test_u5_gate_plans_at_the_planned_step,
  ::test_u5_shared_residual_problems_change_nothing, ::test_run_step_rule_system;
  tests/test_ft_closedloop.py::test_loop_energy_gate, ::test_hybrid_gate_in_the_loop.

### The closed loop (F32-F38)

**F32 (the closed-loop iteration).** Definitions; the noise-free limit [derived]; the iteration is the CQE on
the ACSE residual, in the spin-orbital metric of the standard CQE and with quasi-Newton directions [cited:
Smart and Mazziotti, PRL 126, 070504 (2021); Smart, Boyn, Mazziotti, J. Chem. Theory Comput. 18, 5286 (2022)].
Code: `ft.closedloop.run_closed_loop`, `ft.closedloop.LoopSpec`, `ft.closedloop.system_inputs`,
`ft.closedloop.SystemInputs`, `ft.closedloop.ClosedLoopRecord`; the loop body is `steprules.run_step_rule`
(F26–F31) with the estimators of F33–F35 as its hooks.

- Inputs, once per system (`system_inputs(name)`, cached in `ft.scaling.Cache`): the symmetry-adapted
  build (F2a), 12 singlet roots (F21), the selected QPE and its H′ (F9), the K allowed spin-free elements
  with m_e and LCU normalizations (F11), the per-shot variances at the variance state of F24, the DF split
  lowest at HF (F13), C_rot and Q_rot (F4). Spin-free elements only (ValueError for
  `params.kind = "spin_orbital"`).
- The iteration: Ψ̃₀ = HF; at k = 0, A₀ = W is classical (M1) and the step is ε₂ of F16, unmeasured and
  uncapped, C_iter(0) = 0. At k ≥ 1: (1) the controller plans the budget B_k from the measured history
  (F35); (2) the estimator draws Â_k from the route's law at B_k and returns the per-element variances V̂_k
  (F33, F34); (3) L-BFGS (spin-orbital metric by default) forms d_k from Â₀ … Â_k (Â₀ = A₀) and the rule
  returns ε_k (F26–F31); (4) Ψ̃_{k+1} = exp(ε_kD_k)Ψ̃_k, or the compiled circuit (F37). A run takes
  `max_iter` = 5 steps unless F26's converged-state rule stops it.
- Rules (`ft.closedloop.RULES`): "unit" (R1 with F29's caps), "line_search" (the exact line search along
  the measured d_k: the oracle step), "candidates" (M3's grid, energies drawn at `LoopSpec.eps_ls` =
  1.6 mHa) and "hybrid" (R3, F36), through `steprules.run_step_rule`; a `steprules.StepRule` with the
  classical first step may be passed (ValueError otherwise). "m3_line_search" is M3's trajectory model,
  `cqe.run_cqe` with `FTResidualEstimator.for_run_cqe()` as `ft.trajectory.exact_trajectory` calls it,
  exact propagation only (ValueError otherwise), costed by F14. Under noise the growth cap binds in 17–56
  of the 100 replicates of each headline deciding cell (U6), and the exact line search along a noisy d_k
  ends at ε = 0 (flag "zero_step") in 6–9 of 20 replicates at oracle r = 3 on n2_r2.0 and n2_r2.4 (U7).
- `ClosedLoopRecord` keeps, without using them in any decision, the exact E_k, σ_k², p̃_k (ψ₀ of H), ‖A_k‖,
  ⟨S²⟩ and compile errors, B_k, r_k = B_k/‖A_k‖_m and n̂_k, C(Ψ̃_k), Q(Ψ̃_k), the events and flags, and the
  states with `keep_states` (always for m3_line_search). Its non-realizable parts
  (`ClosedLoopRecord.non_realizable`; their union and "realizable" in the summary) are
  "oracle_controller", "exact_line_search" (line_search, m3_line_search), "angle_cap" (where the cap
  bound) and "slice_rule" (compiled propagation with F25's rule; fixed slices are realizable) (changed
  from the design specification; m4-results.md §13); `LoopSpec.realizable` is False for the oracle
  controller and the slice rule. The oracle hand-off quantities keep their own names (F36).
- Noise-free limit [derived]: `LoopSpec(noise=False)` returns the masked exact residual itself
  (`ft.symmetry.mask_pair_tensor`) with zero variances, the budgets still planned and costed. With
  m3_line_search, the oracle controller at r = 0.3 and the spin-free metric, states, energies, ε_k,
  overlaps, variances, norms and directions equal `exact_trajectory`'s under `np.array_equal` with either
  first step, and `closed_loop_costs` reproduces M3's C_iter(k) of h4, k = 0…7, bit for bit in both loops
  (tests/data/ft_h4_golden.json); the noise-free unit rule is `steprules.run_step_rule` with its masked
  exact residual (energies to 1e-13 Ha), for either controller and metric.
- Interface beyond the design: `run_closed_loop(..., first_step="exact")` (M3's measured loop, with
  m3_line_search and the oracle controller only), so that the noise-free limit covers both M3 records;
  `replicate(..., metric=...)` and (LoopSpec, rule, metric) triples in `run_closed_loop_system`; the record
  fields `estimates`, `step_rule`, `first_step`, `converged` and `gate_log`; the hand-off keys of F36
  (changed from the design specification; m4-results.md §13).
- Tests: tests/test_ft_closedloop.py::test_noise_free_limit_is_m3, ::test_unit_noise_free_equals_steprules,
  ::test_non_realizable_parts, ::test_loop_spec_validation, ::test_record_and_event_validation,
  ::test_default_variance_state, ::test_n8_inputs_use_hf_variances (slow), ::test_run_closed_loop_system.

**F33 (direct-route estimator).** [derived]; independent Gaussian noise with fixed per-shot variances is
M3's variance model (F10) [assumption]. Code: `ft.closedloop.FTResidualEstimator` (route direct_mg); the
debiased slope in `steprules.run_step_rule` (`steprules.debiased_slope`, `steprules.unit_fills`).

- Â_e = A_e + η_e, η_e = (v_e/S_k)^{1/2}z_e (z_e standard normals of the residual stream, F38), S_k = V_m/B_k²
  shots, A_e the masked exact residual of Ψ̃_k; the estimate fills the pair tensor
  (`measure.targets.spin_free_tensor`) with exact zeros off the allowed set and returns V̂_e = v_e/S_k. The
  cost is S_k shots of C(Ψ̃_k) on Q(Ψ̃_k) qubits (F10, F36).
- Realized error [derived]: r̃_k² = ‖η‖_m²/‖A_k‖_m² has the mean r_k² = (B_k/‖A_k‖_m)² and the relative
  standard deviation (2/K_eff)^{1/2}, K_eff = V_m²/Σ_e m_e²v_e² (25.5 on N2 CAS(6,6) at 2.0–2.8 Å, 22.1 on
  h4_r4.0: 28–30% scatter of one residual's r̃_k² [build check]).
- Debiased slope (ΔE_m = ŝ₀′/2 of F26, F27, F31): ŝ₀′ = ŝ₀ + Σ_e V̂_e⟨u_e, H_ku_e⟩, u_e the unit fill of
  element e and H_k the L-BFGS inverse Hessian (`optimizers.LBFGS.inverse_hessian_apply`) [derived: the
  expectation of ⟨η, H_kη⟩ for H_k independent of η]. The newest pair depends on η_k, so in the loop ŝ₀′
  is an approximation, flagged "slope_debias_approx" wherever L-BFGS holds pairs and variances are returned
  (every replicate of U6 and U7). Only R3 uses it; R1, the controllers and the stopping rule (|ŝ₀|) do not.
- Model limits: the element correlations of matchgate shadows (raising M2's noise floors by up to 1.32×)
  are not modelled [design: they left T_E(2) unchanged within the replicate intervals]; v_e is fixed at one
  state per system (F24); the shots' logical failures are costed (F18, F19), not emulated.
- Tests: tests/test_ft_closedloop.py::test_direct_estimator_statistics;
  tests/test_steprules.py::test_debiased_slope, ::test_debiased_slope_flags_in_the_loop.

**F34 (AE estimator: the sine-window law).** [derived]; the sine window [cited: Berry and Wiseman, PRL 85,
5098 (2000)]; the continuum law as the outcome law of a finite run [estimate]. Code:
`ft.closedloop.sine_window_law`, `ft.closedloop.sine_window_errors`, `ft.closedloop.FTResidualEstimator`
(route ae_x), `ft.residual.ae_allocation`.

- Route ae_x (F11): α_e = α_X,e, mean O_e = A_e; F11's allocation at B_k gives ε_e and N_e = 2πκα_e/ε_e
  (κ = `LoopSpec.kappa` = 1), i.e. D_e = N_e/2 applications of G with eigenphases ±2θ_e,
  2θ_e = arccos(−O_e/α_e). Outcome: φ̂_e = 2θ_e + 2πΔ_e/D_e, Ô_e = −α_e cos φ̂_e, with Δ_e on the lattice
  {n − u_e : |n| ≤ 60} drawn by inverse CDF (one uniform of the residual stream per element) from
  P(Δ) = 8cos²(πΔ)/(π²(1 − 4Δ²)²) = 2 sinc²(½ − |Δ|)/(1 + 2|Δ|)², renormalized; offsets "fixed"
  (u_e = frac(D_e·2θ_e/2π), the true phase's offset) or "random" (u_e ~ U(0, 1) from the AE-offset stream).
- Moments [derived]: P = |ŵ|² for w(ξ) = √2 sin(πξ) on [0, 1], whose autocorrelation is supported on
  [−1, 1] and vanishes at ±1 with its first two derivatives; by Poisson summation the lattice mass is 1,
  the mean 0 and the variance ¼ for every u. So the RMS phase error is π/D_e = ε_e/(κα_e) (F11's κ = 1
  in the continuum), Ô_e is unbiased to first order, and Var(Ô_e) = (ε_e sin 2θ_e/κ)².
- Returned variance V̂_e = (ε_e/κ)²(1 − (Ô_e/α_e)²), the plug-in from the allocation and the outcome
  (changed from the design specification; m4-results.md §13): ‖Â‖_m² − Σ_e m_eV̂_e (F35) is then biased
  by −Σ_e m_e(ε_e/κ)²(A_e/α_e)² to second order, 7e-5 (h4) and 3e-4 (n2_r2.4) of ‖A₁‖_m² at r = 1
  [build check]; α_e = 0 returns Ô_e = V̂_e = 0.
- Tails [derived]: P(Δ) ≈ cos²(πu)/(2π²Δ⁴) for |Δ| ≫ 1, so sup_u P(|Δ| > 5) = 2.84e-4 (zero at u = ½), against
  the design's 0.4–2.5e-4 over five sampled offsets (changed from the design specification; m4-results.md
  §13); the |n| ≤ 60 truncation lowers the variance by at most 1.7e-3. A finite register of D applications of
  G has κ(D) = 0.9736, 0.9934, 0.9983 at D = 63, 255, 1023 (M3's F11; RMS phase error κπ/D), and on registers
  of 256 and 1024 points its outcome law lies within 1/256 and 1/1024 of P for |Δ| ≤ 5.5; the emulation's
  continuum law (κ = 1) is on the high side of κ(D) < 1 [estimate], and the runs' logical failures are costed,
  not emulated.
- Cost T = Σ_e N_e(C(Ψ̃_k) + C_obs,e + Q_A,e/2) (F11, F36): on n2_r2.4 at k = 1 ae_x costs 5.09 / 1.70
  C_QPE at r = 1 / 3 against 0.42 / 0.047 for direct_mg, so route "best" measures direct_mg on stretched
  N2 at k = 1, while the deep runs on h4_r1.8 switch to ae_x from k = 3–5 [build check].
- Tests: tests/test_ft_closedloop.py::test_sine_window_law_identities, ::test_sine_window_tails,
  ::test_sine_window_finite_register, ::test_sine_window_sampler, ::test_ae_estimator_allocation.

**F35 (precision controllers).** [derived]; ν₁ = 0.5, ν = 1 and f_fl = 0.1 [design]. Code:
`ft.closedloop.PrecisionController`, `ft.closedloop.debiased_norm_square`.

- "oracle": B_k = r‖A_k‖\_m, M3's convention, from the exact residual (not realizable). "lagged"
  (default): B₁ = rν₁‖A₀‖\_m with the classical A₀ = W, B_k = rνn̂_{k−1} for k ≥ 2, with
  n̂_j² = max(‖Â_j‖\_m² − Σ_e m_eV̂_{j,e}, f_fl‖Â_j‖\_m²) [derived: E[‖Â‖\_m² − Σ_e m_eV̂_e] = ‖A‖\_m², exact
  for the direct route and to second order for the AE route (F34); the floor keeps the budget positive and
  biases n̂ upward where it binds]. ν₁, ν and f_fl are `LoopSpec.rho1`, `LoopSpec.rho` and
  `LoopSpec.norm_floor` (ν avoids the cost ratio ρ of F44).
- r₁ = rν₁‖A₀‖_m/‖A₁‖_m is deterministic (Ψ̃₁ is the classical step's): 3.00r, 2.71r, 2.54r on n2_r2.0, 2.4,
  2.8 and 0.46r on h4_r4.0 (spin-orbital) [build check; U6], the design's 0.46r–3.0r. The deciding lagged
  r = 1 cell thus measures the first iteration near the oracle r = 3 cell on stretched N2 (on n2_r2.0 it
  reproduces that cell's mean T_E(2) to four digits under common random numbers, F38) and at r ≈ 0.5 on
  h4_r4.0.
- Norm precision [derived] (direct route, floor off): sd(n̂²)/‖A‖_m² = (4r²/K₁ + 2r⁴/K₂)^{1/2},
  K₁ = ‖A‖_m²V_m/Σ_e m_e²A_e²v_e, K₂ = K_eff (F33): 0.48–0.51 at r = 1 (design 0.45–0.55) but 2.8–2.9 at r = 3
  at the classical Ψ̃₁ [build check]. At r₁ ≈ 2.5–3 n̂₁ is noise-dominated: the floor binds at k = 1 in 52
  and 54 of 100 replicates on n2_r2.0 and n2_r2.4 [build check], and the median realized r₂ falls to
  0.82–1.38 on N2 at lagged r = 1 (U6).
- A planned budget must be positive and finite (ValueError, e.g. the oracle at ‖A_k‖_m = 0), and the
  lagged plan of k ≥ 2 needs the previous estimate (ValueError). In U6 the oracle r-scan (r = 1, 2, 3) is
  the optimistic bound and the lagged cells (r = 0.5, 1, 2) the realizable plan, of which r = 1 decides.
- Tests: tests/test_ft_closedloop.py::test_controllers.

**F36 (cost of a realized iteration).** [derived]; the planned V_E of the loop's energy gate [estimate].
Code: `ft.closedloop.closed_loop_costs`, `ft.closedloop.handoff_inputs`, `ft.closedloop.LoopEnergyGate`,
`ft.closedloop.FTEnergyEstimator`.

- Residual: M3's unbiased routes depend on (r, ‖A‖_m) only through B = r‖A‖_m (F10, F11), so T_res(k) is
  the measured route at B_k, i.e. at `params.r` = r_k on the realized state: direct_mg S_kC(Ψ̃_k), ae_x
  Σ_e N_e(C(Ψ̃_k) + C_obs,e + Q_A,e/2) with C_obs,e = 2L_X,e and Q_A,e = Q(Ψ̃_k) + ⌈log₂ L_X,e⌉ + χ + 5.
  Both are evaluated from the measurement events (a record without states is costed), equal
  `ft.residual.compare_routes` at r_k to 1e-12, and are reported at B_k with the measured one as `best`
  (AE repeated at κ_practical = 2). Route "best" is the cheaper of the two at B_k on the current circuit,
  chosen before the measurement at the defaults of FTParameters(). The aux routes and gradient estimation
  are not in the loop.
- Circuit: C(Ψ̃_k) = Σ_{j<k} s_jC_step(d_j, ε_j/s_j) (F3, F25; s_j = 1 for exact propagation),
  Q(Ψ̃_k) = max(2N, max_{j<k} Q_prep(j)) (F7), compiled incrementally at δ_step = 1e-3 as the loop runs.
- Energies: F28 at V_E(Ψ̃_{k+1}), one σ per iteration (ValueError otherwise); line_search and candidates
  are charged as 11 energies at 1.6 mHa (line_search charged, not drawn; candidates drawn as E + σz from
  the energy stream, `FTEnergyEstimator`), R3's stencil at σ_P3. R3's gate (`LoopEnergyGate`) is F31's
  switch with T_en^plan at the constant V_E(HF), since a realizable loop has no exact state before the
  step, and T_res^plan the iteration's residual at B_k (planned = realized); against the charge its
  effective threshold is θ_R3V_E(Ψ̃_{k+1})/V_E(HF), with V_E(HF)/V_E(Ψ̃_{k+1}) = 0.75–0.81 on h4 and
  1.95–3.29 on n2_r2.4 at k = 1–3 [build check: oracle r = 1, replicate 0]. It needs the iteration's
  residual (ValueError) and logs every decision in `ClosedLoopRecord.gate_log`.
- Output: one `ft.pipeline.IterationCost` per k (k = 0 free, "classical"; for k ≥ 1 toffolis = T_res + T_en,
  the F19 campaigns, `machines_7d`, "no layout within the failure budget"; no readout rotation for the
  unit rule), so `ft.pipeline.loop_totals` and `ft.pipeline.machines_for_campaigns` apply. m3_line_search
  is costed by `ft.pipeline.iteration_costs` on the converted record (F14), at r = `LoopSpec.r` for the
  oracle controller and at r_k otherwise.
- `handoff_inputs`, one JSON row per replicate: k, p̃_k, E_k − E₀, C(Ψ̃_k), C_iter(k), C_loop(k), r_k, B_k,
  flags, compile errors, ⟨S²⟩, the non-realizable parts, the gate's charged T_en, and M3's rows of
  `ft.handoff.handoff_table` on `ClosedLoopRecord.to_trajectory_record` (loop "classical_first_step",
  same-irrep E₁; the optima "expected" and "expected_oracle"). The 11 energies at the fixed eps_ls make
  line_search and candidates cost 449–485 C_QPE at k = 1 on n2_r2.0 and n2_r2.4 (U7), against 0.51 and 0.42
  for R1 at r = 1: F28's ρ_E ≈ 10³ at σ = eps_ls, which falls as ‖A_k‖²_m on convergence, unlike the
  energy-precision wall of energies at σ ∝ |ΔE_m|.
- Tests: tests/test_ft_closedloop.py::test_realized_cost_identity, ::test_costs_feed_pipeline,
  ::test_loop_energy_gate, ::test_hybrid_gate_in_the_loop, ::test_energy_estimator_noise_law,
  ::test_noise_free_limit_is_m3.

**F37 (compiled propagation in the loop).** [derived]; the slice rule [design]. Code:
`ft.closedloop.run_closed_loop` with `LoopSpec(propagation="compiled")` (`ft.closedloop.SLICE_CHOICES`,
`ft.closedloop.ETA_SLICE`), `ft.circuits.compiled_step`.

- Ψ̃_{k+1} is s_k successive `compiled_step` calls at ε_k/s_k from Ψ̃_k, normalized (F25, N ≤ 8), for every
  state the loop prepares (R3's stencil and the candidates included; the exact line search keeps its own
  Krylov states, and its step is compiled); s_k is fixed by `LoopSpec.slices` (realizable) or set by F25's
  rule (the default; "slice_rule", F32). m3_line_search propagates by Krylov only (ValueError).
- The next residual is measured on Ψ̃_{k+1}, so the loop corrects the step's product-formula and rounding
  errors, but the L-BFGS pair (ε_kd_k, ĝ_k − ĝ_{k+1}) refers to a state that exp(ε_kD_k) does not reach:
  flag "pair_inconsistency" where the step's compile error exceeds η_s.
- Recorded: `compile_errors` (‖Ψ̃_{k+1} − exp(ε_kD_k)Ψ̃_k‖; 0 for exact propagation) and `slices`; the cost
  increment s_kC_step(d_k, ε_k/s_k) (F36). Compiled states break S² (F1, F3): p̃_k and ⟨S²⟩ are the compiled
  state's, and its hand-off carries F41's contaminant term ("contaminant_below_theta" is left to the
  hand-off layer). U7's compiled block runs F25's rule (largest compile error per cell 0.036–0.049).
- Tests: tests/test_ft_closedloop.py::test_compiled_loop, ::test_compiled_step_equals_replay,
  ::test_sliced_compiled_step_equals_replay_of_slices, ::test_compiled_step_validates_the_state,
  ::test_non_realizable_parts.

**F38 (replicates and statistics).** [design]; the percentile bootstrap [cited: Efron and Tibshirani, An
Introduction to the Bootstrap, Chapman & Hall (1993)]; the score interval [cited: Wilson, J. Am. Stat.
Assoc. 22, 209 (1927)]. Code: `ft.closedloop.replicate_streams`, `ft.closedloop.replicate`,
`ft.closedloop.summarize`, `ft.closedloop.bootstrap_mean`, `ft.closedloop.wilson_interval`,
`ft.closedloop.stopping_rules`, `ft.closedloop.paired_verdict`, `ft.closedloop.scheme_verdict`,
`ft.closedloop.run_closed_loop_system`; U6 and U7 by `benchmarks/run_m4_loop.py`.

- Streams (common random numbers): replicate j of a system draws from
  `SeedSequence(20261108, spawn_key=(zlib.crc32(name.encode()), j)).spawn(3)`: residual (F33's normals,
  F34's uniforms), AE offset, energy (F36). The configuration is not in the key, so cells that differ in r
  draw the same normals at k = 1, scaled by their budgets, a rule with energies draws R1's k = 1 residual,
  and the cells of one system are paired replicate by replicate.
- Per replicate, T_E(k) = [C_loop(k) + (C(Ψ̃_k) + C_QPE)/p̃_k]/C_QPE, M3's rule-E expected cost in C_QPE units
  (U6's T_rep), C_loop(k) = Σ_{1≤j<k} C_iter(j), at every k whether or not rule E certifies Ψ̃_k. It is
  repetition as if the ground-state outcomes were recognized (M3's "expected_oracle"): rule E certifies no
  early iterate of stretched N2 (n2_r2.4: E − E₀ = 777, 301 and 252 mHa at HF, the classical Ψ₁ and the unit
  Ψ₂ of the exact records, against E₁ − E₀ = 230 mHa; in the quick run no replicate of its headline cells is
  certified at k ≤ 2, and 3 and 7 of 20 at k = 4 and 5 in the deciding lagged r = 1 cell). The certified
  counterparts are F40's T_B and F43's T_s on the same replicates (U9 part 2). p̃_k is on ψ₀ of H and C_QPE
  that of H′, whose ground states' weights agree within 0.5% on the stretched N2 states [design].
- `summarize`, per configuration over the common hand-off iterates: the mean of T_E(k) with its 95%
  percentile-bootstrap interval (2000 resamples, `np.random.default_rng(12345)`, one resample matrix for every
  statistic; nan with fewer than two replicates or a non-finite value); medians and 16–84% bands of T_E(k),
  p̃_k, E_k − E₀ and r_k; P(T_E(k) < T_E(1)) with its Wilson interval; the mean C_iter(k); the paired
  T_E(k) − T_E(k + 1) with its interval and "pays"; the per-replicate optimum and the best fixed k; the
  catastrophic share P(∃k: p̃_{k+1} < p̃_k/2) with its Wilson interval; flag counts; "realizable". The mean
  decides: 1/p is convex, so noise that spreads p̃ raises the mean T_E above T_E at the mean p̃ (Jensen), and
  a heavy lower tail of p̃ far above T_E at the median p̃ (h4_r4.0 at oracle r = 3, U6: mean T_E(2) = 63.5
  [9.1, 165] C_QPE at median p̃₂ = 0.684, catastrophic share 0.35).
- Stopping rules (`stopping_rules`, post hoc; regret = mean of T_E(rule) − min_k T_E(k)): a fixed hand-off
  iterate k_h ∈ {1, 2, 3, 4} (U6's "K = 2") and the cost gates Θ ∈ {0.3, 1, 3} C_QPE, which hand off Ψ̃_k
  before the first iteration k ≥ 1 whose cost exceeds Θ (the last iterate if none does). The gate reads
  the realized C_iter(k) as the planned cost (changed from the design specification; m4-results.md §13),
  which equals it for R1 (B_k and C(Ψ̃_k) are fixed before the measurement), while the rules with energies
  charge V_E(Ψ̃_{k+1}), known only after the step, so their gate regrets (U7) are post hoc in this respect.
- `paired_verdict`: iteration k pays iff the 95% percentile-bootstrap interval of the mean paired
  T_E(k) − T_E(k + 1) lies above 0 (touching 0 or one replicate: does not pay; non-finite values raise
  ValueError). `scheme_verdict`, the verdict of record under the QPE-side schemes (F43): the paired verdict
  of T_s(k) − T_s(k + 1) per scheme finite in every replicate, and s*(k) = `ft.amplify.best_realizable` of
  the replicate means (never an oracle scheme), pairing T_{s*(k)}(k) with T_{s*(k+1)}(k + 1).
  `benchmarks/run_m4_summary.py` applies both to every configuration (U9's closed-loop rows, U0's Q4) and
  checks its rule-E costs and verdicts against `summarize` to 1e-12.
- Configurations (`benchmarks/run_m4_loop.py`; R1 spin-orbital, direct_mg, exact propagation, 5
  iterations, unless noted):

  | block | systems | configurations | replicates |
  |---|---|---|---|
  | headline (U6) | n2_r2.0, n2_r2.4, n2_r2.8, h4_r4.0 | oracle r = 1, 2, 3; lagged r = 0.5, 1, 2 | 100 |
  | controls | h6_r3.0, h4_r1.8, n2_r1.098 | oracle r = 1, 3; lagged r = 1 | 20 |
  | routes | n2_r2.4, h4_r1.8 | ae_x, fixed and random offsets; oracle r = 1, 3 | 20 |
  | step rules | n2_r2.0, n2_r2.4 | line_search, candidates, hybrid; oracle r = 1, 3 | 20 |
  | compiled | the headline systems | F25 slices; oracle r = 1, 3; lagged r = 1 | 20 |
  | metric | n2_r2.0 | spin-free metric; oracle r = 1, 3 | 20 |
  | N = 8 controls | n2v_r2.4, n2v_r2.8, h8_r3.0 | lagged r = 1; oracle r = 3 | 10 |

  `--quick` runs the headline cells of n2_r2.4 and h4_r4.0 at 20 replicates; the deep runs of U11
  (`benchmarks/run_m4_summary.py`) take route best, lagged r = 0.3 and 1, 30 iterations and 10
  replicates with states on h4_r1.8, lih and n2_r1.098.
- Deciding cells: the lagged r = 1 cell of the headline block decides U0's Q4, that of each control block
  the control's verdict; the oracle r-scan is the optimistic bound, the lagged r = 0.5 and 2 cells the
  sensitivity, the compiled cells an ablation. If the angle cap binds in any replicate of a deciding cell,
  the cell is rerun with `StepRule("unit", angle_cap=None)` on the same replicates, the rerun decides (flag
  "angle_cap_rerun") and the original stays, marked superseded; in U6 and U7 the cap binds in no deciding
  cell. `closedloop_<system>.json` is written as each system finishes, with its provenance; `--resume`
  loads a file of the same provenance.
- Tests: tests/test_ft_closedloop.py::test_common_random_numbers, ::test_summarize_statistics,
  ::test_paired_verdict, ::test_scheme_verdict, ::test_run_closed_loop_system, ::test_headline_regression
  (slow); tests/test_m4_loop_benchmark.py::test_deciding_cell_is_predeclared, ::test_angle_cap_rerun_decides,
  ::test_control_blocks_have_their_own_deciding_cell, ::test_finished_systems_survive_a_later_failure,
  ::test_resume_of_the_closed_loop_compares_its_settings, ::test_reference_rows_without_an_interval;
  tests/test_m4_summary.py::test_q4_reports_the_lagged_sensitivity_the_oracle_bound_and_the_controls.

### The hand-off: windowed QPE, rule B, certificates, QPE-side schemes, break-even overlap and size (F39-F45)

**F39 (windowed QPE outcome law).** [derived]; the sine window [cited: Berry and Wiseman, PRL 85, 5098
(2000)]; (A2) and (A3) [assumption]. Code: `ft.acceptance.phase_window`, `ft.acceptance.walk_offset`,
`ft.acceptance.qpe_points`, `ft.acceptance.eigen_kernel`, `ft.acceptance.windowed_outcome_law`,
`ft.acceptance.model_spectrum`, `ft.acceptance.ModelSpectrum`.

- An M-point control register Σ_x a_x|x⟩ (a_x real, Σ_x a_x² = 1), controlled W^x and the inverse QFT return
  m with amplitude A(m|φ) = M^{−1/2}Σ_x a_x e^{ix(φ − 2πm/M)} on a walk eigenvector of phase φ. An eigenstate
  ψ_j of H′ splits into the walk eigenvectors at ±θ_j, θ_j = arccos((E_j − c)/λ), with weight ½ each:
  K_j(m) = ½(|A(m|θ_j)|² + |A(m|−θ_j)|²), P(m) = Σ_j p_jK_j(m), E(m) = c + λ cos(2πm/M), p_j = |⟨ψ_j|Ψ⟩|². For a
  real window the mass of K_j on an outcome set symmetric under m → −m (every energy window) is that of one
  branch. Weights ≤ 1e-14 are left out; a λ that does not bound |E_j − c| raises ValueError.
- Windows: "rect", the Fejér kernel of `ft.walk.qpe_outcome_law` (equal to it at M = 2^t to 1e-10);
  "sine", a_x ∝ sin(π(x + 1)/(M + 1)), the RMS-optimal register state [cited]; "kaiser",
  a_x ∝ I₀(πα(1 − (2x/(M − 1) − 1)²)^{1/2}) (α = 3 in U8).
- (A2) c is the identity coefficient of the Jordan–Wigner form of H′ [assumption: exact for the Pauli LCU of
  `ft.walk`; another offset moves θ₀ only], c = Tr H′/2^{2N} = ecore + Σ_p h_pp + ½Σ_pr (pp|rr) − ¼Σ_pr (pr|rp)
  [derived]. (A3) One C_QPE buys M = 2n_iter phase points, since a controlled walk step kicks back 2θ
  (`qpe_points(qpe, multiple=2)`) [assumption, consistent with F9's n_iter = ⌈πλ/(2ΔE)⌉ [OF]]: the sine
  window's RMS energy error at 2n_iter is then ΔE sin θ₀ (Appendix A); under the other reading every QPE
  length below doubles. The grid is g = 2πλ sin θ₀/M (`ModelSpectrum.grid`), 2ΔE sin θ₀ at M = 2n_iter;
  distances in grid units use g at θ₀ (the exact phase distance from E₀′ to E₁′ is 0.997–0.998 of
  (E₁′ − E₀′)/g on stretched N2).
- Model spectrum: every eigenpair of the selected H′ in the (n_α, n_β) sector by dense diagonalization
  (N ≤ 8, `max_dim=5000`; none at N = 10, F41), with ⟨S²⟩ and the dominant irrep of every eigenvector
  (degenerate levels, within 1e-10 Ha, rotated to diagonalize the irrep, then S²). An SF truncation that
  cuts through a degenerate ERI eigenspace breaks the point group (n2_r2.4's SF L = 16: forbidden integrals
  up to 1.3e-5 Ha, eigenvector irrep weights down to 0.9945); the kernels do not depend on the labels, the
  choice of E1lb does (F41).
- Costs keep M3's window model for the success probability of a run (an outcome within ΔE of E_j with
  probability wp_j, w = 1, so K₀ = 1; F15) and for the QPE length; the windowed law enters the costs through
  its tails τ_W, which set n_v (F40: the verification runs (n̄_v − 1)C_QPE, the choice between B_U and B_L and
  the expected runs of a dedicated certificate) and the coarse length M_c (F43), and beyond the costs the
  identification errors and their bounds (U8). K₀ = 1 fails where θ lies within about 1.5 grid units of E₀′
  (F40).
- Tests: tests/test_ft_acceptance.py::test_rect_window_equals_ft_walk, ::test_kernels_normalized,
  ::test_kernels_against_direct_dft_sums, ::test_sine_rms_is_de_sin_theta, ::test_qpe_points_and_grid,
  ::test_model_spectrum_against_fci, ::test_model_spectrum_reference_numbers, ::test_model_spectrum_limits,
  ::test_argument_validation, ::test_dense_sector_against_fci_ops_at_n8 (slow), ::test_selected_encodings
  (slow).

**F40 (rule B: the bracketed threshold, verification runs, logical failures).** [derived]; the design floor
of n_v [assumption]. Code: `ft.acceptance.tail_mass`, `ft.acceptance.fejer_tail_bound`,
`ft.acceptance.verification_runs`, `ft.acceptance.verification_run_threshold`,
`ft.acceptance.expected_verification_runs`, `ft.acceptance.failure_budget`,
`ft.acceptance.identification_bound`, `ft.acceptance.acceptance_bracket`, `ft.acceptance.acceptance_stats`;
`ft.handoff.rule_b_table`, `ft.handoff.RuleBRow`, `ft.handoff.rule_b_bracket`, `ft.handoff.optimal_rule_b`,
`ft.handoff.runtime_quantile`; U8 by `benchmarks/run_m4_handoff.py`.

- Rule: fix θ and n_v ≥ 1; repeat QPE runs on Ψ until an outcome has E(m) < θ, then run n_v − 1 verification
  QPEs on the post-measurement state; accept if all return E(m) < θ, else restart from Ψ.
- Soundness [derived]: QPE maps every walk eigenvector to itself and A_θ = {m : E(m) < θ} is symmetric, so an
  attempt passes with P_acc = Σ_j p_jK_j(A_θ)^{n_v}, and the accepted state carries ε_id(θ, n_v) = Σ_{j≥1}
  p_jK_j^{n_v}/Σ_j p_jK_j^{n_v} outside the walk eigenvectors of ψ₀′ (`acceptance_stats`, exact on the sector
  of H′, with P_acc, p₀K₀^n and the energy error P(|E(m) − E₀′| > ΔE); p₀ is the state's weight on ψ₀′). If
  every populated excited level lies at or above E1lb and below the mirror energy 2c − θ (a level above it
  puts up to 2τ_W of its kernel below θ), ε_id ≤ [(1 − p₀)/p₀][τ_W(D)/K₀]^{n_v}, D = (E1lb − θ)/g
  (`identification_bound`); since this grows as 1/p₀, soundness is stated for every state with p₀ ≥ p_min. In
  the window model ε_id = 0 for θ ≤ E1lb − ΔE. Rule B certifies the identification of ψ₀′, not an energy
  window: an outcome of ψ₀′ lies outside ±ΔE of E₀′ with probability 0–0.49 over phase offsets (mean 0.29)
  [build check: F34's lattice law at sin θ₀ = 0.986], 0.396 in U8's n2_r2.4 rows (quick run), so ΔE is an RMS
  scale (ΔE sin θ₀, F39).
- Tail [derived]: τ_W(D) is the supremum over phase offsets of one branch's tail beyond grid distance D, i.e.
  of T(x) on [D, D + 1) (T(x) − T(x + 1) = K(x) ≥ 0), decided near the main lobe and at the steps of n_v by
  the jump point (a lattice point at distance exactly D). `tail_mass` evaluates the M = 4096 register at 64
  equispaced and 33 refined offsets and the jump point and, for the sine window, takes the larger of that and
  the continuum supremum of F34's lattice law,
  Σ_{k≥0} P(x + k) = (8cos²(πx)/π²)[(ψ₁(x − ½) + ψ₁(x + ½))/16 − 1/(8(x − ½))] (ψ₁ the trigamma function),
  which is taken to bound the longer hand-off registers (M = 2n_iter ≈ 2.4e4) too (changed from the design
  specification; m4-results.md §13) [build check: at M = 8192, 16384 and 23848 the finite-register suprema
  over 64 offsets lie at 0.997–1.000 of the value used for D = 1.3–28.7; at D = 1.30 the M = 4096 value with
  the jump point is 9.352e-3 against the continuum 9.389e-3, and at D = 26.03 the continuum 1.0055e-6 needs
  two runs where the M = 4096 law, 9.989e-7, needs one]. Kaiser(3) has τ_W(5) = 1.6e-8; the rect (Fejér) mass
  beyond D, both sides, is ≤ (1 + 1/D)/(2D) (`fejer_tail_bound`).
- Verification runs: n_v = min{n ≥ 1 : [(1 − p_min)/p_min]τ_W(D)^n ≤ ε_id} at p_min = ε_id = 1e-3
  [assumption: the hand-off is designed for p₀ ≥ p_min]; it needs no p₀, so it is realizable. τ_W is a step
  function beyond the main lobe, with thresholds t_n = min{D : n_v(D) ≤ n} (`verification_run_threshold`,
  Appendix A); for the rect window t₁ = ∞ and t₂ = 101.5. Beyond D = 1000 τ_W(1000) bounds the tail;
  ValueError if no n ≤ 64 suffices.
- K₀ = 1 [estimate: M3's window model]: the n_v rule, the B_L floors below, F21's D_L(p), F43's 9.6e-4 and the
  cost T_B set K₀ = 1. By the soundness bound n_v runs keep ε_id ≤ 1e-3/K₀^{n_v} at p₀ = p_min, and an attempt
  passes with p₀K₀^{n_v}. With the sine window the smallest K₀ over phase offsets is 0.51 at θ = E₀′ + ΔE (a
  certificate U → E₀′, or B_L at its termination edge), 0.91 at E₀′ + 2ΔE (rule E at a converged E_k), 0.995
  at E₀′ + 3ΔE and ≥ 0.9989 from 2.5 grid units above E₀′ [build check: F34's lattice law at sin θ₀ = 0.986].
  The thresholds of the table below lie ≥ 18 mHa above E₀′, but B_U at U = CISD(H′) lies 1.01 mHa (0.53g)
  above E₀′ on LiH, K₀ = 0.529, and 2.04 mHa (1.07g) on h4, K₀ = 0.996 [build check]: there the costing
  understates the QPE terms accepted at θ by up to 1/K₀ (1.9× on LiH), while the kernel term of HF stays below
  1e-9.
- Forms (`acceptance_bracket`, chosen without E₀′): B_U, θ = U + ΔE with a certificate U ≥ E₀′ (F42),
  D_U = (E1lb − U − ΔE)/g and n_v(D_U) runs, terminating and certified iff U ≤ E1lb − 2ΔE; M3's rule E is B_U
  with U = E_k + ΔE and n_v = 1 (T_B = T_E in the window model; B_U certifies it iff E₁ − E_k ≥ 3ΔE with
  E1lb = E₁, M3's E₁ − E_k > 3ΔE of F15, `ft.handoff.rule_e_certifiable`, with the edge included). B_L: no U,
  θ_L = E1lb − D_Lg, D_L = 5, n_v = 2, sound to ε_id ≤ 1e-3 at K₀ = 1 for every state with
  p₀ ≥ τ_W(5)²/(ε_id + τ_W(5)²) = 3.37e-5 (one run: p₀ ≥ 0.155); it terminates iff E₀′ + ΔE ≤ θ_L, checked in
  the model, and a chosen B_L carries "termination_not_certified" (a U certifying it would give D_U ≥ 5 and
  B_U). The smaller n_v decides, B_U on ties. Flags of an `ft.acceptance.Bracket`: "no_certificate",
  "termination_not_certified", "u_outside_bracket" (U + ΔE ≥ E1lb, or more than 64 runs) and "window_model"
  (`grid=None`: n_v = 1, B_U at U + ΔE if U ≤ E1lb − 2ΔE, else B_L at E1lb − ΔE).
- `rule_b_table` (on F15's rows): the options are B_L without U and, per non-oracle certificate, its bracket
  and, where that is B_L, B_U at U + ΔE with its own larger n_v; a row takes the cheapest terminating option
  (ties to certified termination, then B_U, then no certificate), costs ∞ where nothing terminates (flag
  "not_terminating"), and ignores certificates of iterates beyond the rows (changed from the design
  specification; m4-results.md §13). `rule_b_bracket` makes the same choice by n_v for a free classical U;
  E1lb defaults to E₁(H) − 1 mHa (flag "e1lb_from_h", F41).
- Cost, in M3's window model (K₀ = 1):
  T_B(k) = C_loop(max(k, j_U)) + C̄_U + (C(Ψ_k) + C_QPE)/p_k + (n̄_v − 1)C_QPE, F15's oracle T_E(k)
  (`RuleBRow.cost_oracle_e`) plus the certificate and the verification runs
  (j_U = `EnergyCertificate.loop_iterate`). With K₀ < 1 and no accepted excited weight an attempt passes with
  p_kK₀^{n_v}, and the hand-off term would be (C(Ψ_k) + C_QPE)/(p_kK₀^{n_v}) + C_QPEΣ_{i=1}^{n_v−1}K₀^{−i}. A
  dedicated sampled or AE energy (F42) is a random bound Û = U + δ, δ ~ N(0, σ²), and n_v is fixed after Û is
  known (`expected_verification_runs`): B_U with n_v(D(Û)) where Û ≤ E1lb − 2ΔE, B_L's n_l where that is fewer
  or Û is inadmissible, so n̄_v = 1 + Σ_{n=1}^{n_l−1} Φ((max(ΔE/g, t_n) − D(U))g/σ); where B_L does not
  terminate, inadmissible bounds repeat the certificate, C̄_U = C_U/P(Û ≤ E1lb − 2ΔE),
  n̄_v = E[n_v | admissible]. `RuleBRow` keeps the planned U, θ and n_v and reports n̄_v (`n_runs_expected`)
  and C̄_U (`certificate_cost`) (changed from the design specification; m4-results.md §13).
- `optimal_rule_b`: "bracketed", the cheapest finite T_B (flagged rows included), and "bracketed_oracle",
  M3's oracle T_E; ties to the smaller k. The runtime is a quantile, not a certificate: attempts succeed with
  probability ≥ p₀K₀^{n_v}, and a deadline met with probability 1 − δ needs R_δ(q) attempts
  (`runtime_quantile`, F15's `runs_for_confidence`).
- Logical failures [derived]: a failed run (probability p_run) returns an arbitrary outcome and state,
  accepted with probability f ≤ 1 (f = 1 − θ₀/π for a uniformly random register, the share of outcomes below
  E₀′; 0.446 on n2_r2.4). Post-selection enriches the accepted failures of an unverified run to f
  p_run/(P_acc + f p_run). A retested run's predecessors survive only through the excited part it accepts (≤
  τ_W), which removes the enrichment, and its own failure adds f p_run, so the two share the budget. The
  output failure is ε_fail = 1e-3 (`failure_budget`): p_run = ε_fail/(2f) for a "retested" run (the last of
  n_v ≥ 2, the coarse runs, the final QPE of a filtered scheme) and ε_fail p_min/f, with p_min in place of the
  unknown P_acc, for an "unverified" one (n_v = 1 with rep, aa, bbht or fp), i.e. 5e-4 and 1e-6 at f = 1 in
  the costing and U12; U8's failure term takes the state's f, so that f p_run = 5e-4 and 1e-6 there
  (p_run = 1.12e-3 and 2.24e-6 on n2_r2.4). A coherent AA sequence is laid out as one run at the retested
  budget (U12) [assumption: amplification brings its acceptance probability to O(1), so its failures are not
  enriched by 1/p₀; bbht's first stages, which accept with probability ≈ p₀, are not separated]. The kernel
  term (≤ ε_id at p₀ ≥ p_min) and the failure term are budgeted apart, so the total is ≤ 2e-3; M3's
  p_fail_qpe = 1e-2 stays in M3's tables.
- Identification errors (kernel terms of these states, not bounds; selected SF encodings,
  E1lb = E₁′ − 0.1 mHa, B_U at U = CISD(H′), B_L at D_L = 5; Ψ₁ the classical spin-orbital step, Ψ₂ the
  noise-free unit-step iterate; p₀ on H′):

  | state (p₀) | B_U sine 2n_iter | B_U Kaiser(3) 4n_iter | B_U rect 2n_iter | B_L sine, 1 run | B_L sine, 2 runs | θ = E₁′ − ΔE, sine |
  |---|---|---|---|---|---|---|
  | n2_r2.0 HF (0.2539) | 7.5e-7 | 2.3e-9 | 1.25e-3 | 1.1e-4 | 9.6e-9 | 0.180 |
  | n2_r2.0 Ψ₂ (0.4730) | 4.9e-7 | 1.5e-9 | 6.9e-4 | – | – | 0.126 |
  | n2_r2.4 HF (0.1120) | 2.38e-4 | 3.76e-8 | 9.8e-2 | 1.6e-5 | 1.0e-10 | 1.9e-3 |
  | n2_r2.4 Ψ₁ (0.2125) | 1.9e-4 | 3.0e-8 | 8.1e-2 | 1.3e-5 | 8.5e-11 | 1.5e-3 |
  | n2_r2.4 Ψ₂ (0.2810) | 1.5e-4 | 2.3e-8 | 6.4e-2 | – | – | 1.2e-3 |
  | n2_r2.8 HF (0.0781) | no U | no U | no U | 1.6e-4 | 1.0e-8 | 0.322 |
  | n2_r2.8 Ψ₁ (0.1647) | no U | no U | no U | 1.3e-4 | 8.5e-9 | 0.281 |
  | h4_r4.0 HF (0.4341) | no U | no U | no U | 4.9e-6 | 6.9e-10 | 1.5e-3 |

  The literal E1-referenced threshold E₁′ − ΔE is unsound (E₁′'s main lobe straddles it), and so is the rect
  window at these overlaps. At U8's budgets (f p_run = 1e-6 for one run, 5e-4 for the last of n_v ≥ 2) the
  failure term adds ε_fail/2 = 5e-4 to the two-run entries and ε_fail p_min/(P_acc + ε_fail p_min) ≈ ε_fail
  p_min/P_acc to the one-run entries of B_U and B_L, 2.1e-6 (n2_r2.0 Ψ₂) to 1.3e-5 (n2_r2.8 HF) [build check];
  down to p₀ = p_min the bracket's n_v keeps the exact ε_id ≤ 1e-3, while one B_L run would not. The literal
  entry of h4_r4.0 is not pinned [build check].
- U8 (m4-results.md §7), per state (HF, the classical Ψ₁, Ψ₂ and Ψ₃ of the exact and R1 records of both
  metrics, the compiled Ψ̃₁–Ψ̃₃; F30's R1 costs at r = 1, M3's F14 costs at r = 0.3 on the exact records):
  form, U source, margin E1lb − U − 2ΔE, D, n_v and n̄_v, C_U and C̄_U, T_B, ε_id (sine and Kaiser(3), one and
  n_v runs), the failure term, the design-floor bound [(1 − p_min)/p_min]τ_W(D)^{n_v}, the state's bound (F41)
  and M3's rule-E verdict. It costs with p on ψ₀′ of H′, so that its T_B equals U9's rep; M3's T6 under rule
  B keeps p on H.
- Tests: tests/test_ft_acceptance.py::test_fejer_tail_bound, ::test_tail_mass_against_the_closed_form,
  ::test_verification_runs, ::test_tail_mass_is_the_supremum_with_the_jump_point,
  ::test_verification_run_thresholds, ::test_expected_verification_runs,
  ::test_expected_runs_where_a_run_count_suffices_nowhere, ::test_identification_errors,
  ::test_identification_error_table, ::test_acceptance_matches_monte_carlo, ::test_identification_bound_holds,
  ::test_identification_bound_closed_form, ::test_small_weights_enter_the_identification_error,
  ::test_bracket_forms, ::test_logical_failure_term, ::test_energy_error_and_stats_identities;
  tests/test_ft_handoff_rule_b.py::test_rule_b_is_oracle_plus_certificate,
  ::test_b_u_with_more_runs_where_b_l_does_not_terminate, ::test_b_l_termination_edge,
  ::test_certificate_beyond_the_rows_is_ignored, ::test_e1lb_fallback_value,
  ::test_certificate_of_the_last_row_iterate_is_used, ::test_dedicated_certificates_cost_their_expected_runs,
  ::test_rule_b_reduces_to_rule_e, ::test_runtime_quantile, ::test_rule_b_termination_flags_and_inputs,
  ::test_rule_b_on_m3_t6; tests/test_m4_handoff_benchmark.py::test_u8_row_at_a_distance_beyond_the_tail_register,
  ::test_handoff_rows_of_n2_r2_4 (slow).

**F41 (the sector of E1lb; contaminants of compiled states).** [derived]; E1lb [assumption]; THC's breaking
of the point group [design]. Code: `ft.certify.e1_lower_bound`, `ft.certify.sector_levels`,
`ft.certify.contaminant_weight`, `ft.acceptance.acceptance_stats` (`contaminant=`); the E1lb choice, the
bounds and the compiled rows of U8 and U9 in `benchmarks/run_m4_handoff.py`.

- Rule B (F40) and the filters (F43) are sound if every populated eigenstate of H′ (j ≥ 1) lies at or above
  E1lb; p_j = 0 outside the sector that the preparation and the walk conserve. `sector_levels` reads the
  candidates off the model spectrum: `e1_singlet` (the lowest excited singlet of ψ₀′'s irrep),
  `e1_any_spin`, `e1_any_irrep`, `e1_any`, each with gap, ⟨S²⟩ and irrep, and with `det_irreps` the irrep
  weights and the impurity max_j(1 − weight).
- HF and the exact iterates conserve (N, S_z, S², irrep), so E1lb refers to the same-irrep singlet E₁ (M3's
  E₁ on H′). Where H′ breaks the point group it refers to the any-irrep singlet: for THC [design: at the
  1e-5 level] and, in the benchmarks, for every encoding whose eigenvectors leave their dominant irrep by
  more than 1e-6 (flag "point_group_broken"; min(e1_singlet, e1_any_irrep)) (changed from the design
  specification; m4-results.md §13), e.g. n2_r2.4's SF L = 16 (impurity 5.5e-3; both singlets 230.863 mHa).
  E1lb = E₁(H′) − 0.1 mHa where the model spectrum exists (N ≤ 8; `ft.certify.E1_MARGIN_MODEL`), else
  E₁(H) − 1 mHa (`E1_MARGIN_EXACT`; flag "e1lb_from_h") [assumption: |E₁(H′) − E₁(H)| ≤ 1 mHa; observed
  +0.75, +0.56, −0.06 mHa on n2_r2.8, n2_r2.4, n2s]. As in M3 no lower bound on E₁ is computed: every
  certificate is conditional on E1lb.
- Compiled elementwise iterates (F1, F3) break S². Either E1lb refers to the lowest excited level of the
  irrep of any S, or the contaminant weight q_c = Σ_j p_j over the eigenstates below θ whose ⟨S²⟩ differs from
  ψ₀′'s by more than 1e-4 (`contaminant_weight`) is budgeted, ε_id ← ε_id + q_c/(p₀K₀^n) for n runs
  (`acceptance_stats(contaminant=q_c)`; the design's q_c/(p₀K₀) is the one-run case) (changed from the design
  specification; m4-results.md §13). The kernel term already contains the contaminants exactly; q_c is
  reported beside it. The any-spin option fails where it is needed: the lowest same-irrep level of any spin
  is a quintet 21.4, 4.35 and 1.21 mHa above E₀′ on n2s, n2_r2.4 and n2_r2.8, so a spin-broken state of
  stretched N2 has no E1lb with a margin, and only the contaminant budget remains.
- U8's bounds need every populated excited level at or above E1lb. Where a state puts more than 1e-12 on
  excited levels below E1lb, the design-floor bound is not reported ("n/a (q_c)" for contaminants below θ,
  "n/a (below E1lb)" for weight between θ and E1lb or on singlets of other irreps), and the state's bound
  adds that weight w_<, accepted with K_j ≤ 1: [(1 − p₀)/p₀][τ_W(D)/K₀]^{n_v} + w_</(p₀K₀^{n_v}) (changed from the
  design specification; m4-results.md §13); on n2_r2.4's compiled rows it lies above the exact kernel term.
- Compiled rows of U8 and U9 are the R1 spin-orbital steps 0–2 compiled elementwise from HF in one slice
  (`ft.circuits.replay_elementwise`), with the loop costs of the exact R1 states (flags
  "compiled_elementwise", "exact_state_loop_costs"); every U9 scheme accepted below θ with q_c > 1e-12
  carries "contaminant_below_theta" and q_c/(p₀K₀^n) at its own threshold.
- Tests: tests/test_ft_certify.py::test_sector_levels, ::test_sector_levels_any_irrep_n2,
  ::test_sector_levels_on_model_and_irrep_weights, ::test_e1_lower_bound, ::test_contaminant_weight_compiled_n2s;
  tests/test_ft_acceptance.py::test_e1_any_irrep_against_fci, ::test_contaminant_term_uses_the_k0_power;
  tests/test_ft_handoff_rule_b.py::test_e1lb_fallback_value;
  tests/test_m4_handoff_benchmark.py::test_u8_bounds_of_a_compiled_state_with_a_contaminant,
  ::test_scheme_rows_of_a_compiled_state, ::test_compiled_record_flags.

**F42 (certificates U and their costs).** [derived; F13, F17]; the z-scaling of the AE error [estimate].
Code: `ft.certify.truncated_ci_energy`, `ft.certify.excitation_levels`, `ft.certify.EnergyCertificate`,
`ft.certify.energy_certificates`.

- A certificate is an upper bound U ≥ E₀′ with its cost beyond the loop and the iterate j_U up to which the
  loop must run (`EnergyCertificate.loop_iterate`): classical, E_CISD(H′) (E_CISDT(H′) optional), free,
  j_U = 0; sampled, E_j + m_j/2 at the cost S_j(C(Ψ_j) + C_rot), S_j = (2z)²V_E(Ψ_j)/m_j² shots, j_U = j; ae,
  E_j + m_j/2 at N_j = 4πκλz/m_j applications of C(Ψ_j) + C_W + Q_A/2 [estimate], j_U = j; loop_energy,
  Ê_j + zε_j, no cost beyond C_loop(j + 1), j_U = j + 1; oracle, E₀′, reported only. Here
  m_j = E1lb − 2ΔE − E_j and σ = m_j/(2z), z = 3, so that the bound lies in [E_j, E1lb − 2ΔE] whenever
  |Ê_j − E_j| ≤ zσ: F13's DF-grouped estimator (`ft.residual.energy_sampling`) or its AE option
  (`ft.residual.energy_ae`, κ = 1). A certificate is returned only if U ≤ E1lb − 2ΔE (m_j ≤ 0 or a NaN
  variance gives none; AE certificates need Q(Ψ_j)); the loop-energy row needs a rule that measures energies
  (R1 does not) and is conservative by one iteration. The dedicated energies are random bounds carrying σ
  (`EnergyCertificate.sigma`), costed by F40's expected runs (changed from the design specification;
  m4-results.md §13). E_j is the record's energy of H; ⟨Ψ_j|H′ − H|Ψ_j⟩ is not budgeted [assumption: below the
  ΔE margins, since |de_fact| ≤ 0.5 mHa (F9)].
- Truncated CI [derived]: E_CI(l) is the lowest eigenvalue of P_lHP_l, P_l the projector on the determinants
  of the (n_α, n_β) sector at excitation level ≤ l relative to HF (l = 2 CISD, 3 CISDT), and
  E_min ≤ E_CI(l) ≤ E_HF with E_min the lowest level of the sector of any spin and irrep, which E₀′ (index 0
  of F39's model spectrum) is by construction. The level depends on the spatial occupations, which S₊ and S₋
  conserve, so P_l commutes with S_z, S² and the point group, and the minimum over all spins and irreps is a
  variational bound on E_min whatever its spin, hence on the singlet ψ₀′ wherever it is the sector's ground
  state (every system of U8 and U9 [build check: ⟨S²⟩ = 0 at index 0, full run]). Lanczos with full
  reorthogonalization from HF plus a seeded random vector, to the residual 1e-10 Ha (RuntimeError after 1000
  iterations), returns the Rayleigh quotient of the Ritz vector, an upper bound at any tolerance; it equals
  dense solves to 1e-9 Ha and reaches N = 10. On stretched N2 the CISD minimum is a quintet and the CISDT
  minimum the septet of six open shells; on n2_r2.4 the singlet CISD lies above E₁′ (Appendix A).
- With U = CISD(H′): n2_r2.0 B_U (margin 55.59 mHa, D_U = 28.7, one run); n2_r2.4 margin 3.08 mHa, D_U = 2.07
  (three runs, so B_L with two is taken); n2_r2.8 none. CISDT(H′) is reported, not used; it would certify
  n2_r2.8 and give n2_r2.4 B_U with one run, a sensitivity of Q5. U8 and U9 take CISD(H′) and the
  dedicated energies of Ψ₀–Ψ₃; M3's T6 under rule B those of every iterate.
- Tests: tests/test_ft_certify.py::test_excitation_levels, ::test_lanczos_equals_dense_solve,
  ::test_truncated_ci_is_variational_before_convergence, ::test_cisd_values, ::test_truncated_ci_table,
  ::test_truncated_ci_validation, ::test_certificate_costs_follow_f42, ::test_energy_certificate_validation,
  ::test_cisd_at_n10 (slow); tests/test_ft_acceptance.py::test_model_spectrum_reference_numbers;
  tests/test_ft_handoff_rule_b.py::test_dedicated_certificates_cost_their_expected_runs,
  ::test_certificate_of_the_last_row_iterate_is_used.

**F43 (QPE-side schemes).** [derived; cited as noted]; the filter table and the two-stage QPE [design]; the
comparator and the reflections [estimate]. Code: `ft.amplify.grover_calls`,
`ft.amplify.exponential_search_calls`, `ft.amplify.fixed_point_calls`, `ft.amplify.step_filter_polynomial`,
`ft.amplify.step_filter_degree`, `ft.amplify.FILTER_TABLE`, `ft.amplify.filter_degree`,
`ft.amplify.filter_cost`, `ft.amplify.coarse_qpe_points`, `ft.amplify.coarse_qpe_toffolis`,
`ft.amplify.scheme_costs`, `ft.amplify.final_qpe_form`, `ft.amplify.state_scheme_costs`,
`ft.amplify.scheme_table`, `ft.amplify.best_realizable`, `ft.amplify.best_extended`; U9 by
`benchmarks/run_m4_handoff.py` (part 1) and `benchmarks/run_m4_summary.py` (part 2).

- A call of 𝒜 = (coherent QPE with the comparator flag E(m) < θ)·Prep(Ψ) costs C(Ψ) + C_QPE (the comparator,
  about 2t Toffolis, and the reflections, about Q, neglected [estimate]); its success probability is a = p in
  M3's window model, φ = asin√a. An AA sequence is one coherent run of N(C(Ψ) + C_QPE) Toffolis, which does
  not parallelize (F19) and carries F40's retested budget.
- Known a [cited: Brassard, Høyer, Mosca, Tapp, Contemp. Math. 305, 53 (2002)]: repetition with m Grover
  iterations, N*(a) = min_m (2m + 1)/sin²((2m + 1)φ) expected calls (`grover_calls`, over the integers next to
  (1.165561/φ − 1)/2 and m = 0, 1, with (2m + 1)φ < π); x/sin²x is least at tan x = 2x, so N\*√a → 1.380050
  [derived]; N* < 1/a iff a < (3 − √3)/4 = 0.317, where the m = 1 branch 3/sin²(3φ) = 3/(a(3 − 4a)²) meets 1/a
  [derived; tests/test_ft_amplify.py::test_grover_values checks it on a 0.01 grid], and on (0.25, 0.317) N*
  rises with a to 3.155 (N*(0.25) = 3, N*(0.32) = 1/0.32 = 3.125). An oracle: m needs a.
- Unknown a, exponential search [cited: Boyer, Brassard, Høyer, Tapp, Fortschr. Phys. 46, 493 (1998)]: stage
  r draws j uniformly from [0, M_r), M_r = ⌈(6/5)^r⌉, and makes 2j + 1 calls, succeeding with
  P(M) = ½ − sin(4Mφ)/(4M sin 2φ) [derived], so E_BBHT(a) = Σ_r M_r Π_{s<r}(1 − P(M_s)), summed to survival
  1e-16 (`exponential_search_calls`; E_BBHT(1) = 1). Above p = 0.299 it overshoots, E_BBHT(p) > 1/p.
- Certified a_lb, fixed point [cited: Yoder, Low, Chuang, PRL 113, 210501 (2014)]: the smallest odd
  L = 2l + 1 with L ≥ acosh(1/δ)/atanh(√a_lb), for which P_L(a) = 1 − δ²T_L(T_{1/L}(1/δ)√(1 − a))² ≥ 1 − δ² at
  every a ≥ a_lb (`fixed_point_calls`, δ = 0.01).
- Step filter [cited: Lin and Tong, Quantum 4, 372 (2020); GQSP as F8]: an even trigonometric polynomial g of
  degree d in the walk phase θ with |g| ≤ 1, |1 − g| ≤ ε_f on [θ_pass, π] (energies ≤ U) and |g| ≤ ε_f on
  [0, θ_stop] (energies ≥ E1lb), Δθ = θ_pass − θ_stop. `step_filter_polynomial` Fourier-truncates a
  Gaussian-smoothed step centred in the band (width optimized over 28 values in [0.05, 0.5]Δθ), scaled to
  max|g| ≤ 1, checks the band errors on a dense periodic grid and exactly at both band edges, and bisects d
  up to the grid's Nyquist limit; the edge check and six widths below 0.08Δθ are additions (changed from the
  design specification; m4-results.md §13), since a grid-only check misses the pass-edge error by up to 11%
  at ε_f = 1e-3. Between grid points |g| exceeds 1 by ≤ 1.1e-6. Degree [design]: d = ⌈c_f(ε_f)/Δθ⌉
  (`filter_degree`), c_f = 1.01 × the largest constructed dΔθ over two pass edges and two widths, tabulated
  on ε_f = 10^{−j/4}, j = 8…40 (`FILTER_TABLE`, `ft.amplify.build_filter_table`), the construction itself for
  Δθ > 0.1 or ε_f < 1e-10; the table bounds the construction from above, by ≤ 13% at 1e-2.
- Filter cost (`filter_cost`, ε_f = 1e-2): C_filt = 2dC_W + 3(2d + 1)(b_rot − 2) on Q_walk + 1 qubits, F8's
  count of 2d controlled walk calls for the Laurent polynomial g = Σ_{|j|≤d} g_jW^j in GQSP; the walk part is
  (4c_f/π) sin θ̄ ΔE/(E1lb − U) C_QPE, θ̄ = (θ_pass + θ_stop)/2, to 0.5%, and the rotations, which the design's
  scheme table leaves out, add 3(b_rot − 2)/C_W ≈ 5.1–5.2% (changed from the design specification;
  m4-results.md §13). The count is conservative by up to 2× in the walk part against (A3) [estimate]: (A3)'s
  controlled step applies W or W† and kicks back 2θ, with which a real polynomial of degree d in x = cos θ
  takes d steps, as in the construction of QETU [cited: Dong, Lin, Tong, PRX Quantum 3, 040305 (2022)], and
  which the coarse QPE's C_c = (M_c/2)C_W already assumes; the model is thus biased against the step-filter
  schemes, whose C_filt/C_c = 0.59–0.64 would fall to about half. The leakage is ε_f² per unit excited weight.
- Two-stage QPE [design]: a coarse sine-window QPE of M_c points, C_c = (M_c/2)C_W, repeated until its outcome
  lies below a cut between U and E1lb, then one full QPE. M_c (`coarse_qpe_points`) is the smallest length at
  which every level at or above E1lb passes the best cut with probability ≤ ε_p and every level at or below
  U with ≥ 1 − ε_p, the levels taken at 32 sub-offsets per grid spacing beyond each edge (finite-register law
  up to M = 4096, F34's continuum law above; D_p = 6.0 at ε_p = 1e-4). The design's condition on the two edge
  levels alone is a lattice lottery, since the lattice law vanishes at half-integer distances (changed from
  the design specification; m4-results.md §13). C_c/C_QPE ≈ 4 sin θ₀D_pΔE/(E1lb − U) within −8%/+0%.
- E1lb-referenced forms [derived; B_L's status]: in lt, lt_bbht and 2s, U only places the pass edge
  (termination), soundness comes from the stop edge at E1lb; lt_bbht_L and 2s_L put the filter or coarse
  QPE between E1lb − w_L and E1lb, w_L = 100 mHa (`ft.amplify.W_L`), terminate iff E₀′ ≤ E1lb − w_L (else no
  cost, "not_terminating_L") and are certified only with a certified U ≤ E1lb − w_L
  ("termination_not_certified").
- Acceptance: rep, aa, bbht and fp accept by the state's rule-B option (F40: expected verification runs,
  certificate cost and certified iterate, with its flags). The filtered schemes need no verification run
  [derived: the suppressions multiply, [(1 − p_min)/p_min]ε_f²τ_W(1.3)/(1 − ε_f)² = 9.6e-4 at ε_f = 1e-2, the
  same with ε_p = 1e-4 for 2s]; their final QPE is B_U at U + ΔE where the certified U has D_U ≥ 1.3
  (`final_qpe_form`), else B_L ("termination_not_certified"; no cost where B_L does not terminate,
  "not_terminating").
- Cost at iterate k (c_k = C(Ψ_k), C_loop = C_loop(k), V = (n̄_v − 1)C_QPE plus rule B's surcharge
  C_loop(max(k, j_U)) − C_loop(k) + C̄_U, C_U the cost of the U_k of the U-referenced forms):

  | scheme | T_s(k) | status |
  |---|---|---|
  | rep | C_loop + (c_k + C_QPE)/p + V | realizable (B_U; B_L flagged) |
  | aa | C_loop + N*(p)(c_k + C_QPE) + V | oracle (p known) |
  | bbht | C_loop + E_BBHT(p)(c_k + C_QPE) + V | realizable |
  | fp | C_loop + L(p_lb, δ)(c_k + C_QPE) + V | realizable given p_lb > 0 (exact inputs, uncosted) |
  | lt | C_loop + C_U + N*(p)(c_k + C_filt(U_k)) + C_QPE | oracle; certified U |
  | lt_bbht | C_loop + C_U + E_BBHT(p)(c_k + C_filt(U_k)) + C_QPE | realizable; certified U |
  | 2s | C_loop + C_U + (c_k + C_c(U_k))/p + C_QPE | realizable; certified U |
  | lt_bbht_L | C_loop + E_BBHT(p)(c_k + C_filt(w_L)) + C_QPE | realizable (termination_not_certified) |
  | 2s_L | C_loop + (c_k + C_c(w_L))/p + C_QPE | realizable (termination_not_certified) |
  | lt_rep (sensitivity) | C_loop + C_U + (c_k + C_filt(U_k))/p + C_QPE | certified U; never s* |
  | lt_rep_L (sensitivity) | C_loop + (c_k + C_filt(w_L))/p + C_QPE | where the L forms terminate; never s* |

  `scheme_costs` is the plain table (integer n_v, no surcharge; at n_v = 1 its rep is M3's T_E),
  `scheme_table` its rows over hand-off rows, `state_scheme_costs` a state's row with the flags of every
  scheme ("not_terminating", "not_terminating_L", "no_certificate", "no_overlap_bound",
  "termination_not_certified", "oracle_U"). Without a certified U, lt, lt_bbht and 2s are reported at the
  oracle U = E₀′ + 2ΔE without a certificate cost (`<scheme>@oracle_U`), never in s*: since C_filt and C_c
  grow with U (∝ 1/(E1lb − U)), these rows bound the cost under every certified U ≥ E₀′ + 2ΔE from below, not
  under a certificate closer to E₀′.
- U_k is, per scheme, the cheapest of CISD(H′) and the dedicated sampled and AE energies of Ψ_k, each with its
  own filter and coarse QPE, certificate cost and final-QPE form (changed from the design specification;
  m4-results.md §13). fp uses F15's Eckart–Temple p_lb = 1/(1 + σ_k²/(E₁ − E_k)²) from the record's exact E_k,
  σ_k² and same-irrep E₁ of H (`ft.handoff.overlap_lower_bound`), as M3's rule δ does: the energy and ⟨H²⟩
  measurements a hardware bound needs are not costed and E1lb does not replace E₁, so fp is realizable only
  given those inputs; for compiled states the lowest populated excited level of H′ of any spin and irrep
  (weight > 1e-14), less 0.1 mHa, stands for E₁, and fp has no cost where the state's energy lies above it
  (changed from the design specification; m4-results.md §13).
- s* = `best_realizable`, the argmin over rep, bbht, fp, lt_bbht, 2s, lt_bbht_L, 2s_L with a cost, ties in
  that order (never aa, lt, an oracle-U bound or a sensitivity key); each scheme keeps its own flags, and a
  U9 row's flags are those of s* (changed from the design specification; m4-results.md §13). `best_extended`
  adds lt_rep and lt_rep_L, the filter followed by plain repetition: where E_BBHT(p) > 1/p (p ≳ 0.3) they
  undercut lt_bbht and lt_bbht_L, and with C_filt/C_c = 0.59–0.64 [build check] also 2s and 2s_L; they
  report the sensitivity of s* to the scheme set and never decide (changed from the design specification;
  m4-results.md §13).
- U9 (m4-results.md §8): per record (exact_so, exact_sf with M3's F14 loop costs at r = 0.3; r1_so, r1_sf,
  compiled_so with F30's at r = 0.3, 1, 3) and state Ψ₀–Ψ₃, T_s/C_QPE under every scheme (p on ψ₀′ of H′)
  beside rep under M3's rule E ("rep_rule_e": C_loop + (C(Ψ_k) + C_QPE)/p, no verification run and no
  certificate surcharge), s* with its flags, the extended optimum, and the savings T_s(k) − T_s(k + 1),
  k = 1, 2, per scheme (rep_rule_e included) and at s* (F44; `benchmarks/run_m4_handoff.py`, `savings`); part 2
  the same on the closed-loop replicates (p̃ on ψ₀ of H, CISD(H′) the only certificate) with bootstrap
  intervals (F38).
- Tests: tests/test_ft_amplify.py::test_grover_brute_force, ::test_grover_values,
  ::test_exponential_search_monte_carlo, ::test_exponential_search_values, ::test_fixed_point_closed_form,
  ::test_step_filter_degree, ::test_filter_band_edges_need_the_exact_check, ::test_filter_degree_from_the_table,
  ::test_filter_cost_ratio, ::test_coarse_edge_levels_are_a_lottery, ::test_two_stage_qpe,
  ::test_coarse_length_from_the_continuum_is_minimal, ::test_scheme_costs, ::test_best_realizable,
  ::test_state_scheme_costs_without_a_certified_u, ::test_state_scheme_costs_terminate_the_l_forms_at_w_l,
  ::test_state_scheme_costs_flags_on_the_quick_sides, ::test_state_scheme_costs_against_the_scheme_table,
  ::test_filter_then_repeat_sensitivity, ::test_rep_column_equals_m3, ::test_filter_table (slow),
  ::test_scheme_table_from_the_model (slow); tests/test_m4_handoff_benchmark.py::test_scheme_rows_of_a_compiled_state,
  ::test_scheme_row_takes_p_on_h_prime, ::test_scheme_row_final_qpe_of_the_u_referenced_forms,
  ::test_handoff_rows_of_n2_r2_4 (slow).

**F44 (what a better state is worth: marginal condition and break-even overlap).** [derived; generalizes
F15]. Code: `ft.scaling.marginal_saving`, `ft.scaling.break_even_overlap`, `ft.scaling.cost_ratio`;
`ft.amplify.scheme_saving`, `ft.amplify.best_realizable_saving`, `ft.amplify.iteration_verdict`,
`ft.amplify.extended_verdict`, `ft.amplify.saving_budget`, `ft.amplify.break_even_overlap`.

- With T_s(k) = C_loop(k) + H_s(p_k, c_k), H_s the hand-off cost of F43 beyond the loop, iteration k pays under
  s iff C_iter(k) < μ_k^s = H_s(p_k, c_k) − H_s(p_{k+1}, c_{k+1}); the cost ratio is ρ_k^s = C_iter(k)/μ_k^s (∞ if
  μ_k^s ≤ 0). `scheme_saving(s)` returns μ^s as a function of (p_k, p_{k+1}, c_k, c_{k+1}) (arrays broadcast; fp
  with p_lb = p, the most a certified bound can be), the `saving` hook of `ft.scaling.cost_ratio`.
- Repetition in C_QPE units (`marginal_saving`): μ_k = (1 + c_k)/p_k − (1 + c_{k+1})/p_{k+1}; with the gain
  G_k = p_{k+1}/p_k, iteration k pays iff p_k < p_k* = [(1 + c_k) − (1 + c_{k+1})/G_k]/c_iter(k)
  (`ft.scaling.break_even_overlap`; c_iter = 0 gives ∞ if the bracket is positive, else 0), a function of the
  gain and the costs only.
- At the best realizable scheme (`iteration_verdict`, the verdict of record "pays_best") iteration k pays iff
  min_s T_s(k + 1) < min_s T_s(k), s* re-chosen at each iterate, i.e. iff C_iter(k) < μ*\_k =
  min_s H_s(p_k, c_k) − min_s H_s(p_{k+1}, c_{k+1}) (`best_realizable_saving`, with F40's (n_v − 1)C_QPE in rep
  and bbht; fp has no cost there, since an overlap alone certifies no p_lb). The verification runs enter rep,
  bbht and fp but not the filtered schemes, so they do not cancel between schemes. `iteration_verdict` also
  compares every scheme with a cost at both iterates (oracle schemes reported, never decisive); an infinite
  T_s(k) with a finite T_s(k + 1) pays, two infinite costs are not compared; `extended_verdict` is the same at
  `best_extended`.
- Budget (`saving_budget`), the most a perfect state can save, H_s(p₀, 0) − H_s(1, 0): C_QPE(1/p₀ − 1) (rep),
  C_QPE(N*(p₀) − 1) ≈ C_QPE(1.38/√p₀ − 1) (aa), C_QPE(E_BBHT(p₀) − 1) (bbht), (N*(p₀) − 1)C_filt (lt),
  (E_BBHT(p₀) − 1)C_filt (lt_bbht, lt_bbht_L with C_filt(w_L)), C_c(1/p₀ − 1) (2s, 2s_L); F40's verification
  runs cancel within a scheme.
- Break-even overlap (`ft.amplify.break_even_overlap`) of an iteration of cost C_it that lifts an HF-like
  state (c = 0) to (p₁, c₁): the largest p₀ with H_s(p₀, 0) − H_s(p₁, c₁) ≥ C_it, by stepping down from p₁ in
  factors of 1.1, then bisection, since N* is not monotone and E_BBHT ripples (changed from the design
  specification; m4-results.md §13). For C_it ≫ C_QPE, p₀* ≈ C_QPE/C_it (rep), (1.38C_QPE/C_it)² (aa),
  (1.38C_filt/C_it)² (lt) and (κ_BC_filt/C_it)², κ_B = E_BBHT√p ∈ [1.8, 2.9] (lt_bbht, lt_bbht_L): under the
  amplified schemes a measured iteration needs a far smaller p₀ to pay. fp's L jumps from 1 at p = 1 to 3 just
  below it, so its p₀* against a perfect state is ≈ 1 for C_it ≤ 2C_QPE.
- U9b gives p₀* per scheme for C_it ∈ {0.1, 1, 10, 100, 1000} C_QPE against a perfect state and the R1
  spin-orbital Ψ₂. U4 (m4-results.md §4) gives, per R1 record, metric, r and k ≤ 2, p_k, G_k, c_iter(k), μ_k,
  ρ_k and p_k* under repetition (p on ψ₀ of H), as an optimistic side column ρ of the exact-line-search step
  of the classical-step record (its p_k → p_{k+1}, C(Ψ_k) and C(Ψ_{k+1}), charged its residual T_res(k) alone,
  the line search free), and μ, ρ and the verdict under the schemes (`benchmarks/run_m4_summary.py`).
- Tests: tests/test_ft_scaling.py::test_break_even_overlap_matches_marginal_condition,
  ::test_scaling_report_tables; tests/test_ft_amplify.py::test_saving_and_break_even,
  ::test_scheme_saving_accepts_arrays, ::test_best_realizable_saving, ::test_iteration_verdict,
  ::test_filter_then_repeat_sensitivity.

**F45 (break-even size).** [derived; the inputs are estimates]. Code: `ft.scaling.break_even_size`,
`ft.scaling.break_even_bracket`, `ft.scaling.break_even_sizes`; U4's n* by `ft.scaling.scaling_report`
(repetition) and `benchmarks/run_m4_summary.py` (schemes).

- Model: p_k(n), p_{k+1}(n) from F22 (in logarithms, capped at 1), c_iter(k; n) = c₀(n/n₀)^β (F23; with a
  crossover (n_c, β_loc) continued as (n/n_c)^{β_loc} beyond n_c), the preparation terms
  c_j(n) = C(Ψ_j)(n)/C_QPE(n) = c_{j0}(n/n_{j0})^{β_j} (`prep`), and the hand-off QPE lengthened by
  f(n) = max(1, m/Δ₁(n)) beyond the window (F21): ρ_k(n) = c_iter(k; n)/μ_k(n), n*\_k = min{n ≥ n₀ : ρ_k(n) < 1}
  over even n (closed-shell chains), None if no n ≤ 3000 qualifies, and the scan ends where p_k or p_{k+1}
  falls below `P_FLOOR` = 1e-300 (changed from the design specification; m4-results.md §13).
- Lengthening [derived], in C_QPE(n) units: under repetition μ_k = (f + c_k)/p_k − (f + c_{k+1})/p_{k+1}, so the
  preparation terms are not lengthened; a scheme whose H_s scales with C_QPE at fixed C(Ψ) (rep, aa, bbht, fp
  and minima over them, F40's runs included) gives μ_k = f·μ_k^s(p_k, p_{k+1}, c_k/f, c_{k+1}/f), and the filtered
  schemes, which carry C_QPE once and whose C_filt and C_c do not depend on ΔE, μ_k = μ_k^s(p_k, p_{k+1}, c_k,
  c_{k+1}) at every n. The F44 hooks carry `qpe_scaled` (the keyword overrides it); a mixed minimum is exact
  under neither form (changed from the design specification; m4-results.md §13).
- For c_k ≪ 1, with a_k = ln p_k(n_last) + α_kn_last, ln ρ_k = ln c₀ + β ln(n/n₀) − ln
  f(n) − (α_kn − a_k) − ln(1 − e^{(a_k − a_{k+1}) − (α_k − α_{k+1})n}): ρ_k falls only beyond n_turn ≈ β/α_k,
  and there is no n* if α_{k+1} ≥ α_k and a_{k+1} ≤ a_k, since then G_k = p_{k+1}/p_k ≤ 1 at every n (no
  gain); a bounded gain (α_{k+1} = α_k, a_{k+1} > a_k) still crosses, μ_k growing as e^{α_kn} against the
  power law of c_iter. This, the design's form, is reported as a check ("n_star_c0") beside n* with the
  preparation terms, which grow faster with N than C_QPE and need not be small at n* (changed from the design
  specification; m4-results.md §13).
- Bracket (`break_even_bracket`): rate ∈ {last, Aitken} × β ∈ {last, mean, 9} × crossover ∈ {none, (20, 2),
  (20, 4)} × window ∈ {none, m_E, m_B, m_BL} × gap model ∈ {power law, A′/n}, 126 cases, with p_k and p_{k+1}
  extrapolated at the same rate (matched corners) and the crossed corners reported apart as a sensitivity
  ("crossed") (changed from the design specification; m4-results.md §13). A windowed case is "inside" iff
  Δ₁(n*) ≥ m in its own gap model (changed from the design specification; m4-results.md §13). A summary gives
  the range of n* with p and c at both ends, the cases without n*, the c_k = 0 check and "inside k/N";
  "overall" ranges over the windowed cases, since F21 rules out the unlengthened QPE beyond n_cert, with the
  no-window cases in their own column, and "by_margin" adds the per-model window ends and minima,
  inside_any and inside_all (changed from the design specification; m4-results.md §13).
- U4's inputs: the first measured iteration of the R1 loop, k = 1 (classical Ψ₁ → unit Ψ₂), per R, metric and
  r: α₁ from the classical-step series, α₂ from the R1 record, each from its own last size; c_iter(1) from the
  R1 record's k = 1 cost or, without an R1 record (sf at N = 10), the cheapest iteration at the same Ψ₁ (F24),
  anchored at the last point; F21's margins. k = 2 would need fits of Ψ₃, which the protocol does not record
  (changed from the design specification; m4-results.md §13).
- Schemes beyond the computed sizes (`benchmarks/run_m4_summary.py`): bbht, aa and "best" = min(rep, bbht)
  with B_L's n_v = 2, since no certified U or p_lb exists there [assumption: CISD's error grows with n while
  Δ₁ falls]; the E1lb-referenced forms only while Δ₁(n) ≥ w_L + 1 mHa, and where that window reaches beyond
  n₀ the bracket holds only if no scheme can pay inside it (flags "l_window_closed",
  "l_window_no_break_even", "l_window_unresolved") (changed from the design specification; m4-results.md
  §13). Every value beyond the computed sizes is a model statement and its range a model bracket.
- Tests: tests/test_ft_scaling.py::test_break_even_size_brute_force, ::test_break_even_size_rates,
  ::test_break_even_size_preparation_terms, ::test_break_even_size_preparation_window,
  ::test_break_even_size_underflow, ::test_break_even_size_schemes_beyond_window,
  ::test_break_even_size_finds_and_misses, ::test_break_even_bracket_cases_and_summaries,
  ::test_inside_window_per_gap_model, ::test_inside_window_open_gap, ::test_inside_window_regression_values,
  ::test_break_even_bracket_crossed_corners, ::test_no_break_even_without_gain, ::test_n_star_inputs_from_records,
  ::test_scaling_report_fits (slow); tests/test_ft_amplify.py::test_saving_hooks_mark_qpe_scaling.

### Properties (F46-F53)

**F46 (property targets, elements and norms).** [derived; conventions of docs/conventions.md]; the accuracy
rule [assumption]. Code: `ft.properties.PropertyTarget`, `ft.properties.rdm1_target`,
`ft.properties.rdm2_target`, `ft.properties.natural_orbitals`, `ft.properties.occupation_target`,
`ft.properties.cumulant`, `ft.properties.cumulant_target`, `ft.properties.cumulant_norm_target`,
`ft.properties.dipole_target`, `ft.properties.property_values`, `ft.properties.property_error`;
`ft.properties.TARGET_NORMS`, `ft.properties.ACCURACIES`.

- The target state is ψ₀, the lowest singlet of the HF irrep of the active-space H; γ[p,q] = ⟨E_qp⟩
  (spin-summed, tr γ = n_e) and D[p,q,r,s] = ⟨e_{pq,rs}⟩ (pair order, no ½), the `dm1` and `dm2` of
  `residual.rdm12`. States are real (an imaginary part above 1e-13 raises ValueError).
- A target is K Hermitian element operators O_e = Σ f_e[p,q]E_pq + Σ c_e[p,q,r,s]e_{pq,rs} (+ a constant)
  with multiplicities m_e and a norm: fro, (Σ_e m_e x_e²)^{1/2}; max; scalar. The kinds (accuracies in
  Parameters): natural occupations, N̂_i = Σ_pq v_ipv_iqE_qp at the natural orbitals v_i of γ₀ = γ(ψ₀) (elements
  (i, i)); 1-RDM, (E_pq + E_qp)/2 for p ≤ q of one irrep (m = 2 off the diagonal); 2-RDM,
  (e_{pq,rs} + e_{qp,sr})/2 on the orbits of {1, P, T, PT}, P: (pq,rs) → (rs,pq), T: (pq,rs) → (qp,sr), m the
  orbit size; cumulant, O^Δ_e on the 2-RDM orbits; cumulant norm, Σ_e m_e(Δ_e/‖Δ‖_F)O^Δ_e at (γ₀, D₀); dipole,
  μ_z = Σ_A Z_Az_A − Σ_pq r_z[p,q]E_pq. P and T have sign +1 on real states, so Σ_e m_e x_e² is the squared
  Frobenius norm of the full tensor ((N⁴ + 3N²)/4 orbits, Σ m_e = N⁴); F2a's mask leaves the counts of
  Appendix A, and since the forbidden elements vanish on ψ₀'s sector the masked targets carry the whole
  tensors.
- Natural orbitals: γ₀ is diagonalized block by block (connected components of |γ₀[p,q]| > 1e-10) and the
  occupations are sorted descending, so degenerate occupations of different irreps are never mixed (LiH's
  π pair, 1.5246e-3 twice, keeps its C2v orbitals) (changed from the design specification; m4-results.md
  §13). N̂_i is first-order exact only for perturbations that do not couple degenerate natural orbitals;
  the CQE iterates keep the HF irrep, so route A is unaffected.
- Cumulant (spin-summed singlet): Δ[p,q,r,s] = D[p,q,r,s] − γ[p,q]γ[r,s] + ½γ[p,s]γ[r,q] [derived from the
  spin-orbital cumulant with γ_α = γ_β = γ/2; the decompositions of Mazziotti, Chem. Phys. Lett. 289, 419
  (1998), and of Kutzelnigg and Mukherjee, J. Chem. Phys. 110, 2800 (1999), not re-checked against the
  papers]; Δ = 0 for a closed-shell determinant. The element operators are linearized at γ₀,
  O^Δ_e = O^D_e − γ₀[r,s]Ê_pq − γ₀[p,q]Ê_rs + ½(γ₀[r,q]Ê_ps + γ₀[p,s]Ê_rq), Ê_ab = (E_ab + E_ba)/2, averaged
  with e's Hermitian partner, so that their expectations change as Δ_e to first order (the occupations
  likewise); `cumulant_norm_target` raises ValueError where ‖Δ₀‖_F ≤ 1e-12.
- Dipole: the z component of `int1e_r` at the origin in the MO basis plus the nuclear constant, for
  all-active builds only (a frozen core would add an unimplemented constant; ValueError); the benchmark
  computes it on LiH (STO-3G, 6 orbitals, 4 electrons; the chains' dipoles vanish by symmetry).
- Each element has a Jordan–Wigner Pauli sum (`PropertyTarget.operators`) for the estimators and a CI-space
  operator (`PropertyTarget.expectations`), equal to 1e-12 and reproducing `residual.rdm12`; `property_values`
  gives the nonlinear functionals (RDM elements, sorted eigenvalues of γ, Δ, ‖Δ‖_F, μ) and `property_error`
  their difference in the target's norm, which against ψ₀ is the bias of every U10 and U11 cell. An estimate meets ε iff bias² + variance ≤ ε² in
  the target's norm (F10's rule with an absolute target), RMS over the elements without a union bound
  [assumption]. U10 and U11 rows are system × target (× ε) of `ft.scaling.PROPERTIES`, the targets built at
  ψ₀ (`benchmarks/run_m4_properties.py`, `build_targets`).
- Tests: tests/test_ft_properties.py::test_targets_reproduce_rdm12, ::test_two_constructions_agree,
  ::test_rdm2_orbits, ::test_cumulant_vanishes_for_hf_and_trace_rules, ::test_dipole_matches_pyscf,
  ::test_target_validation.

**F47 (bias of an approximate state).** [derived]. Code: `ft.properties.bias_bound`,
`ft.properties.PropertyTarget.spreads`, `ft.properties.PropertyTarget.ranges`,
`ft.properties.model_bias_bound`; U11's check `f47_check` in `benchmarks/run_m4_properties.py`.

- For a real Ψ = √pψ₀ + √qχ with χ ⊥ ψ₀, ⟨O⟩_Ψ − ⟨O⟩₀ = 2√(pq)⟨ψ₀|(O − ⟨O⟩₀)|χ⟩ + q(⟨χ|O|χ⟩ − ⟨O⟩₀), so
  |b_O| ≤ 2σ₀√(q(1 − q)) + q·r_O (q form), σ₀² = ⟨O²⟩₀ − ⟨O⟩₀² and r_O = ‖O − ⟨O⟩₀‖ on the (n_α, n_β)
  sector, and |b_O| ≤ 2σ₀√(d/Δ₁) + r_O·d/Δ₁ (energy form), with q ≤ d/Δ₁ (Eckart), d = E_Ψ − E₀ and Δ₁ the
  gap to the lowest level the state populates. A property error is first order in √q, the energy error
  second order.
- σ₀ by `spreads`; r_O by `ranges`, the largest |eigenvalue − ⟨O_e⟩| on the sector (ARPACK to 1e-10, dense
  up to 64 determinants). Tensors, with x = √(q(1 − q)), y = q (or x = √(d/Δ₁), y = d/Δ₁):
  2x‖σ₀‖_m + y‖r_O‖_m (fro, the triangle inequality), max_e(2xσ₀,e + y·r_O,e) (max), the element (scalar).
  Exactly one of q and (d, Δ₁) is given (ValueError otherwise); q ∈ [−1e-12, 1 + 1e-12] is clipped and
  d ≥ −1e-12 read as 0.
- Nonlinear targets: for the occupations, the cumulant and its norm F47 bounds the property error to first
  order in √q only. For ‖Δ‖_F the maximal error exceeds the bound at q ≳ 0.1 and stays below it at
  q ≤ 1e-2, which covers its cost uses (route B's leaked weights and the default model bias are ≪ 1e-2).
  All-orders form (`bias_bound(..., cumulant_parts=(cumulant target, 1-RDM target))`) [derived]: with
  γ = γ₀ + δγ and D = D₀ + δD, Δ_Ψ − Δ₀ = L + Q, L the changes of the linearized cumulant elements and
  Q = −δγ⊗δγ + ½(δγ⊗δγ)^X ((A⊗B)[pqrs] = A[pq]B[rs], Y^X[pqrs] = Y[psrq]), ‖Q‖_F ≤ (3/2)‖δγ‖_F², so
  |‖Δ_Ψ‖_F − ‖Δ₀‖_F| ≤ B_cum + (3/2)B_1² with F47's bounds of the cumulant and 1-RDM targets, on real states of
  ψ₀'s sector with the masked targets (on every real state with unmasked ones); the default stays linearized
  and U11 reports both (changed from the design specification;
  m4-results.md §13).
- Uses: route A (U11), route B's leakage at q = the leaked weight (F50), and the model bias of H′ at
  q = 1 − F, F = |⟨ψ₀′|ψ₀⟩|² (`model_bias_bound`), which F50 and F51 charge only without the exact model
  bias; it is loose, 4–24 times the exact model bias over LiH's six targets on SF L = 16 [build check].
- U11's check (`f47_check`): per target, on every iterate of the R1 and exact-line-search records with an
  error above 1e-9 (`TINY_ERROR`), at q = 1 − p_k, d = E_k − E₀, Δ₁ = E₁ − E₀, the smallest bound/error of
  each form ("F47 min ratio (q, energy)"; ≥ 1 means the bound holds on every iterate); the cumulant norm is
  marked first order and gets the all-orders ratios.
- Tests: tests/test_ft_properties.py::test_bias_bound_on_records, ::test_cumulant_norm_bound_to_all_orders,
  ::test_bias_bound_scalar_and_max_forms, ::test_ranges_are_sector_norms, ::test_model_bias_bound_helper,
  ::test_model_bias_bound_on_scanned_encodings (slow), ::test_cumulant_norm_bound_adversarial (slow);
  tests/test_m4_properties_benchmark.py::test_f47_check_with_the_all_orders_form.

**F48 (property estimators).** [cited: Zhao, Rubin, Miyake, PRL 127, 110504 (2021) for matchgate shadows];
per-element AE as F11 and gradient estimation as F12, with their [estimate]s; the crossover [derived].
Code: `ft.properties.property_inputs`, `ft.properties.PropertyInputs`, `ft.properties.shadow_cost`,
`ft.properties.ae_cost`, `ft.properties.gradient_cost`, `ft.properties.crossover_accuracy`.

- Statistical budget B = (ε² − b²)^{1/2}, b the row's bias; infeasible (infinite Toffolis, flag
  "infeasible") where B² ≤ 1e-12ε² (F10's round-off rule) or b = ∞, which is how route-A rows above their
  usable bias pass (F49). The inputs (`property_inputs`) are evaluated at ψ₀ and used for every measured
  state [design]: v_e, the exact per-snapshot matchgate-shadow variance of O_e
  (`measure.schemes.matchgate_shadow_variance`, method "auto": exact pair sums up to 2e7 pairs, Monte Carlo
  above; every benchmark target is exact); α_e and L_e, the 1-norm and term count of O_e's Jordan–Wigner
  form; σ₀,e and r_O,e (F47), with the target norms `PropertyInputs.sigma_norm` and `range_norm`.
- Shadows (`shadow_cost`): S = Σ_e m_ev_e/B² (fro) or max_e v_e/B² (max, scalar) snapshots of C_copy Toffolis
  (route A: C(Ψ_k); route B: F50's destructive copy) on Q_copy qubits, each a run at p_fail_shot; with b = 0
  and B = r‖A‖_m it is M3's direct route to 1e-12.
- Per-element AE (`ae_cost`; F11 with the Pauli LCU of O_e): C_obs,e = 2L_e, Q_obs,e = ⌈log₂ L_e⌉ + χ + 4
  [estimate], Q_A,e = Q_U + Q_obs,e + 1; fro: F11's allocation, N_A = 2πκΓ/B; max and scalar: ε_e = B per
  element, N_A = 2πκΣ_e α_e/B; T = Σ_e N_e(C_U + C_obs,e + Q_A,e/2), one coherent run per element at
  p_fail_ae. κ = 1, and AE is the reference estimator; it equals M3's `ft.residual.ae_x_lcu` to 1e-12.
- Gradient estimation (`gradient_cost`; a band, never in F53's best rows): N_GE = 2πκ_g√K(Σ_e m_eα_e²)^{1/2}/B
  (fro), 2πκ_g√K max_e α_e/B (max), 2πκ_gα/B (scalar); T = N_GE(C_U + K(2b_θ + 2)), b_θ = 16, on
  Q_U + Kb_θ + max_e Q_obs,e qubits [estimate: it undercounts multi-term O_e]; κ_g ∈ {1, 10, 100} (rows
  "ge1", "ge10", "ge100", flags "estimate" and "assumption"); at K = 1 and κ_g = κ it needs AE's
  preparations.
- Crossover (`crossover_accuracy`) [derived]: S ∝ B⁻² and N_A ∝ B⁻¹, so shadows and AE need the same
  preparations at B* = V_m/(2πκΓ) (fro) or max_e v_e/(2πκΣ_e α_e) (max, scalar), and AE fewer below it (U10b's
  column B*); on every pair of the benchmark runs B* ≥ 0.22, above every accuracy of the study, so AE needs
  fewer preparations wherever both are costed on the same preparation.
- Tests: tests/test_ft_properties.py::test_shadow_cost_and_crossover, ::test_ae_cost_uses_f11,
  ::test_gradient_cost_k1_equals_ae, ::test_property_inputs_on_h4, ::test_matchgate_variances_match_snapshots
  (slow).

**F49 (route A).** [derived]; the records and loops [design]. Code: `ft.properties.route_a_rows`,
`ft.properties.PropertyRow`, `ft.properties.compiled_iterates`, `ft.properties.sliced_prep_cost`;
`benchmarks/run_m4_properties.py` (`build_records`, `residual_costs`, `compiled_records`,
`compiled_residual_costs`, `floors`).

- At a budget k fixed before the run, T_A(P, ε; k) = C_loop(k) + T_meas(P, B_A(k); C_U = C(Ψ_k),
  Q_U = Q(Ψ_k)) with B_A(k)² = ε² − ‖b_A(k)‖², ‖b_A(k)‖ the exact property error of Ψ_k (F46), for
  k ∈ {2, 5, 10, 20, 60}, per estimator and loop. A row is usable iff ‖b_A(k)‖ < ε; otherwise it is costed
  at b = ∞ and carries "not_usable" (reported, never selected). A budget past the record (k > k_end) uses
  the last iterate and carries "record_ended"; it stays selectable only with the free loop on a converged
  record, where C(Ψ_kend) is the realized cost, and is "not_usable" with a measured loop (whose cost
  C_loop(k_end) is only a lower bound) or past a record that did not converge (changed from the design
  specification; m4-results.md §13).
- The A-oracle row is the minimum over k at the exact bias (route "A-oracle", flag "oracle_k"), reported
  only; where no k is finite it is the iterate with the smallest error, "not_usable". Route A's estimate
  carries no error certificate (F47 needs E_Ψk − E₀ and E1lb), so a fixed-budget row's usability is a fact
  of the emulation, and U0's Q6 states the comparison per fixed k (F53).
- Loops: "free" (C_loop ≡ 0), "meas_r1" (r = 1) and "meas_r03" (r = 0.3) with C_loop(k) = Σ_{j<k} T_res(j),
  T_res(0) = 0 (the classical first step) and T_res(j) = min(direct_mg, ae_x) at κ = 1 at Ψ_j
  (`ft.residual.sampling_direct`, `ft.residual.ae_x_lcu`) from the exact residual, the system's iterate-1
  variances and LCU normalizations (`ft.closedloop.system_inputs`). Records (`build_records`; 60 steps,
  spin-orbital metric, classical first step): r1_so (`steprules.run_step_rule` with
  `StepRule("unit")`), the reference, and ls_so (`ft.trajectory.exact_trajectory`), the comparison
  (route_a_ls, best_ls); a converged record ends there.
- Compiled iterates (N ≤ 6): Ψ̃_k by `ft.circuits.replay_elementwise`, a step j in `sliced_steps` run as s
  F3 circuits at ε_j/s (`compiled_iterates`) costed s·C_step(d_j, ε_j/s) (`sliced_prep_cost`, F25); the
  records compiled_s1 (no slicing) and compiled_sliced (F25's rule on steps 0–2, flag
  `slice_cap:step<j>`). The compiled ablation (U10b's "A compiled (free / r=1)") is `route_a_rows` on
  compiled_sliced, with the errors of Ψ̃_k, the sliced C(Ψ̃_k) and the loop costs at Ψ̃_j
  (`compiled_residual_costs`). F25's η_s bounds a sliced step's state error, not the property error of the
  compiled iterate.
- Tables: U10a, T_A(k)/C_QPE (AE) per loop and budget, marked 'x' (bias ≥ ε) or 'e' (a budget past the record
  that is not usable: a measured loop, or the free loop past a record that did not converge), with A-oracle
  [k]; U10b, route A's minimum bias (R1 exact, compiled s = 1, compiled sliced) and the compiled ablation; U11
  (`floors`), per target and record the smallest error over k ≤ 60, its k, its energy error and k_ε (the first
  k after which the error stays ≤ ε), with the deep-run columns (F38).
- Tests: tests/test_ft_properties.py::test_route_a_fixed_budgets, ::test_route_a_budgets_beyond_the_record,
  ::test_compiled_iterates_equal_replay, ::test_compiled_floors (slow), ::test_best_property_row_rules;
  tests/test_m4_properties_benchmark.py::test_u10a_marks_and_the_compiled_cell, ::test_floors_and_k_eps,
  ::test_compiled_residual_costs_follow_the_compiled_record (slow).

**F50 (route B copies).** [derived]; N_ex [cited: Brassard, Høyer, Mosca, Tapp, Contemp. Math. 305, 53
(2002), Theorem 4]; a = p [assumption]; the estimate of a [estimate]. Code: `ft.properties.ground_copy`,
`ft.properties.GroundCopy`, `ft.properties.exact_aa_calls`, `ft.properties.route_b_rows`,
`ft.properties.model_property_errors`; F43's `ft.amplify.filter_cost`, `ft.amplify.exponential_search_calls`,
`ft.amplify.W_L`; F42's `ft.certify.energy_certificates`, `ft.certify.truncated_ci_energy`.

- A copy filters a start Ψ_s of overlap p (HF, Ψ₁ or an iterate Ψ_k with its C_loop(k)) into ψ₀′ of the H′
  the rows run on (F52), by F43's step filter between a pass edge and the stop edge E1lb (F41); overlaps
  on H stand for those on H′ [assumption; within 0.5% on the checked states]. Pass edges per start:
  CISD(H′) where it lies below E1lb; the start's dedicated AE certificate (F42, z = 3, κ = 1), its Toffolis
  added once; E1lb − w_L, F43's E1lb-referenced form ("termination_not_certified"; where E₀′ > E1lb − w_L
  the copy is "not_terminating" and dropped).
- Filter error ε_f = min(1e-2, share·ε√p/(2σ₀√(1 − p))), share = 0.1 (1e-2 at p = 1 or σ₀ = 0), which puts the
  first-order leak bias 2σ₀ε_f√((1 − p)/p) at share·ε. The success probability lies in
  [p(1 − ε_f)², p + ε_f²(1 − p)]; the costs take a = p [assumption: ψ₀′ lies deep in the pass band; the lower
  end would raise them by at most (1 − ε_f)⁻² ≤ 1.02 and move N_ex at a = 0.254 from 3 to 5], under which
  alone the design's "a = p up to O(ε_f²)" holds (changed from the design specification; m4-results.md
  §13). Leaked weight w_f ≤ ε_f²(1 − p)/(p(1 − ε_f)²), leak bound b_leak = 2σ₀√(w_f(1 − w_f)) + w_f·r_O (F47;
  without r_O first order only, flag "leak_first_order"). Both copies run on
  Q_U = Q_walk + 1 + (Q(Ψ_s) − 2N) qubits.
- Destructive copy (shadows): C_copy = min(1/a, E_BBHT(a))(C_s + C_filt). Coherent copy (AE, GE), the cheaper
  per row of the ratio estimator R = ⟨ΠO⟩/⟨Π⟩ (Π the filter's success flag), C_U = (2/a)(C_s + C_filt)
  [derived: δR = (δ⟨ΠO⟩ − Rδ⟨Π⟩)/a, so with half the budget on each term every element's AE of ⟨ΠO⟩
  (normalization α_O) runs to aB/2, 2/a times its length at B; the factor is charged on C_U, not on C_obs and
  Q_A/2, and the one estimate of ⟨Π⟩, which serves every element, is not charged; it overcounts by up to 2× at
  a ≈ 1] with the bias b_m + b_leak, and exact amplitude amplification on an estimate â,
  N_ex(a) = 2⌈π/(4 asin√a) − ½⌉ + 1 calls (N_ex(1) = 1), C_U = N_ex(C_s + C_filt) [cited]: N_ex = 3 on
  [0.25, 1), 5 on [sin²(π/10), ¼) and 7 on [sin²(π/14), sin²(π/10)) [derived], so the ratio estimator is
  cheaper above a = 2/3.
- â comes from one AE of the flag to δa [estimate]: the sine window resolves the Grover phase 2θ_a
  (sin²θ_a = a) to RMS κπ/D after D iterates, and δa = √(a(1 − a))δ(2θ_a), so it takes D = πκ√(a(1 − a))/δa
  iterates, i.e. 2πκ√(a(1 − a))/δa calls of C_s + C_filt in F11's count (𝒜 and 𝒜† per iterate, as N_ex counts
  them). The implementation charges πκ√(a(1 − a))/δa calls, half of that count, once per row (in
  loop_toffolis), since the estimate serves every element. An error δa leaks
  w_AA = sin²((N_ex/2)δa/√(a(1 − a))) onto the flag's bad branch; δa is the largest value (≤ a/2) with
  2σ₀√w_AA + w_AA·r_O ≤ share·ε, i.e. √w_AA = ξ with 2σ₀ξ + ξ²r_O = share·ε, so the estimate takes
  πκN_ex/(2 asin ξ) iterates, independent of a below the cap (charged as 7.85e4 calls at N_ex = 5, ε = 1e-3,
  σ₀ = 0.5 and r_O = 0; 1.57e5 in F11's count). The exact-AA bias is b_m + F47(q = min(1, w_f + w_AA))
  (`GroundCopy.exact_aa_leak_bound`) [design: the flag sectors are orthogonal, so the AA leakage biases a
  system observable only linearly, and F47's cross term is conservative]: each leakage fills its own share,
  and their sum enters the budget (changed from the design specification; m4-results.md §13).
- Model bias b_m of H′: the exact property error of ψ₀′ (`model_property_errors`, ψ₀′ from
  `diagnostics.thc_floors.model_floors(..., return_state=True)`) where the caller passes it, as the
  benchmark does for the encoding the rows run on; otherwise F47 at q = 1 − F (flag "model_bias_bound"). The
  design charges the F47 bound, which alone would make route B infeasible at 1e-3 for LiH's 1-RDM and
  dipole (changed from the design specification; m4-results.md §13); the benchmark keeps it as a
  sensitivity (`model_bias_sensitivity`, F53).
- Rows (`route_b_rows`): per route, estimator and loop the cheapest (start, edge, copy option), with
  T = C_loop(k) + certificate Toffolis + (exact AA: the estimate of a) + T_meas(P, B; C_U, Q_U) and
  B² = ε² − (b_m + leak)²; B0 starts at k ≤ 1 (HF, Ψ₁), Bk at every k, so with T_res(0) = 0 B0 carries no
  loop cost and Bk is never dearer than B0. Flags "edge:<U_classical, U_ae, U_sampled or E1lb>",
  "destructive", "coherent:ratio" or "coherent:exact_aa", "termination_not_certified", "model_bias_bound",
  "leak_first_order"; without a terminating copy the row is infinite and "not_usable" (and "infeasible"
  where every option is). On exact spectra of H′ (the constructed filter on HF of h4, lih and n2s at
  ε_f = 1e-2, 1e-3) the success probability lies in its interval, the leaked weight is 0.014–0.21 of its
  bound and the property errors stay below F47 [build check].
- Tests: tests/test_ft_properties.py::test_copy_counts, ::test_route_b_rows_add_the_estimate_once,
  ::test_route_b_rows_pass_edges_and_termination, ::test_route_b_rows_b0_start_and_exact_aa,
  ::test_default_model_bias_is_f47_at_the_infidelity, ::test_filter_leakage_bound_on_exact_spectra (slow).

**F51 (Hellmann–Feynman QPE for scalar properties).** [derived; the theorem is standard]; the refinement
of the field step [design]; λ and the projections on H′ + hO [estimate]. Code:
`ft.properties.hellmann_feynman_cost`, `ft.properties.STENCILS`, `ft.properties.finite_field_derivatives`,
`ft.properties.FiniteFieldStencil`, `ft.properties.route_b_rows` (`hf_derivatives=`, `hf_stencil_bias=`);
`benchmarks/run_m4_properties.py` (`hf_steps`).

- ⟨ψ₀′|O|ψ₀′⟩ = dE₀′(h)/dh at h = 0 for H′ + hO, O a one-body scalar. Stencils (`STENCILS`): central (bias
  |E‴|h²/6, noise σ_E/(√2h), 2 energies) and five-point (bias |E⁽⁵⁾|h⁴/30, noise √130σ_E/(12h), 4 energies).
  One QPE run reaches RMS σ_E at C_QPE(σ_E) = C_QPE·ΔE/σ_E Toffolis (F9's 1/ΔE law; conservative by sin θ₀,
  F39).
- The model bias b_m (exact, or F47 at q = 1 − F; F50) is systematic and adds to the stencil bias b
  linearly: (b_m + b)² + σ_d² ≤ ε², σ_d the stencil noise. Since C_QPE(σ_E) ∝ 1/(hσ_d) with h ∝ b^{1/2} or
  b^{1/4}, the optimum [derived] is 3b² + 4b_mb + b_m² = ε², h = (6b/|E‴|)^{1/2}, σ_E = √2hσ_d (central) and
  5b² + 6b_mb + b_m² = ε², h = (30b/|E⁽⁵⁾|)^{1/4}, σ_E = 12hσ_d/√130 (five-point), with
  σ_d = (ε² − (b_m + b)²)^{1/2}; b_m = 0 gives b = ε/√3 and ε/√5. Infeasible iff b_m ≥ ε or σ_d² ≤ 1e-12ε².
  The design has no model-bias term (changed from the design specification; m4-results.md §13).
- Field step: the leading-order h is kept if its actual stencil bias on the emulated H + hO
  (`FiniteFieldStencil`, FCI energies at ±h, ±2h, minus ⟨O⟩₀) is within b; otherwise h is halved, at most 12
  times, then bisected 12 times to the largest admissible h (flag "stencil_bias_checked"). A stencil without
  an admissible step is infeasible (`no_field_step:<stencil>`), and the rows use the other stencil or none,
  with routes B0 and Bk unchanged (changed from the design specification; m4-results.md §13). E‴ and E⁽⁵⁾
  (`finite_field_derivatives`): FCI of H + hO on 41 fields of [−0.02, 0.02] a.u., a degree-8 fit, d_n = n!c_n.
  Three approximations [estimate]: the derivatives and stencil biases of H stand for those of H′, the growth
  of λ by |h|α_O is not counted, and the filters and coarse QPE on H′ + hO are costed once on H′, with its λ,
  c and the phases of U and E1lb, independent of h (changed from the design specification; m4-results.md §13).
- Schemes per energy, T = n_E·min_s T_s [derived]: rep (C_s + C_QPE(σ_E))/p + (n_v − 1)C_QPE(σ_E); lt_bbht and
  lt_bbht_L E_BBHT(p)(C_s + C_filt) + C_QPE(σ_E); 2s and 2s_L (C_s + C_c)/p + C_QPE(σ_E); the U forms at
  CISD(H′) where it lies below E1lb, the _L forms at E1lb − w_L where E₀′ ≤ E1lb − w_L, ε_f = 1e-2,
  ε_p = 1e-4, ties in the order rep, lt_bbht, lt_bbht_L, 2s, 2s_L. The pre-projections do not depend on the
  start: the iterates' energy certificates feed route B but not HF-QPE (changed from the design
  specification; m4-results.md §13). n_v is rule B's (F40) at ΔE = σ_E on the long QPE's grid
  g = 2σ_E sin θ₀, probed per stencil at p = 1: B_U at a certified CISD(H′) with n_v = 1 beyond D_U = 26.04
  grid units, or B_L with two; a U 20 mHa below E1lb lies about 500 grid units away on the long grid (about
  10 on a 1-mHa grid: n_v = 1 against 2).
- Rows (scalar targets with derivatives; the LiH dipole): per loop the cheapest start k (with its
  C_loop(k)) and stencil (central only without E⁽⁵⁾), estimator "qpe", competing with every estimator in
  F53; per_prep one energy, n_prep = n_E, longest run C_QPE(σ_E) + C(Ψ_k) on Q_QPE qubits, bias b_m + b. Flags
  `scheme:<s>`, `stencil:<st>`, `n_verify:<n_v>`, "stencil_bias_checked", "termination_not_certified" (rep
  without a certified U, and the _L forms), "model_bias_bound", `no_field_step:<stencil>`. `hf_steps` in
  the JSON records per stencil b, both h and their stencil biases; LiH's numbers are in Appendix A.
- Tests: tests/test_ft_properties.py::test_hellmann_feynman_optimum, ::test_hellmann_feynman_fixed_bias,
  ::test_hellmann_feynman_step_refinement, ::test_hellmann_feynman_without_an_admissible_field_step,
  ::test_route_b_hellmann_feynman_rows_on_lih, ::test_hf_qpe_rows_without_an_admissible_field_step,
  ::test_hf_qpe_acceptance_on_the_long_qpe, ::test_default_model_bias_is_f47_at_the_infidelity,
  ::test_finite_field_bias_on_lih (slow); tests/test_m4_properties_benchmark.py::test_hf_steps_report_the_refined_field_step.

**F52 (property-aware encoding selection).** [design]. Code: `ft.properties.select_for_property`,
`ft.properties.model_property_errors`; `benchmarks/run_m4_properties.py` (`scan_states`, `qpe_side`, the
selection in `run_one`).

- F9 bounds the energy only: its infidelity limit 1e-4 gives b_fact ≤ 2σ₀·1e-2 by F47 (first order), within
  ε/2 only where σ₀ ≤ 25ε; on LiH σ₀ exceeds 25ε for every target and accuracy (0.30 for the NONs to 9.1
  for the 2-RDM, 1.14 a.u. for the dipole). b_fact(P) of every encoding of F9's scan
  (`ft.qpe.scan_encodings`) is the exact error of its ψ₀′ in the target's norm (`model_property_errors`;
  `scan_states` caches the states).
- Rule (`select_for_property`): among the scanned encodings with |de_fact| ≤ 0.5 mHa and b_fact(P) ≤ ε/2,
  the cheapest QPE at ΔE = 1 mHa (ties: fewer logical qubits, then the scan order); if none, the cheapest
  with b_fact(P) ≤ ε/2 regardless of de_fact; if none either, ValueError, and the benchmark keeps F9's
  encoding (flag "no_property_aware_encoding"). F9's infidelity condition is not part of the rule, so at
  loose accuracies the choice can be cheaper than F9's (quick run: LiH's NONs at 1e-2 and cumulant at 0.1
  on SF L = 15, infidelity 1.07e-4, 0.950 C_QPE).
- The benchmark runs route B and HF-QPE on the property-aware encoding of each (target, ε), with that H′'s
  own model spectrum, E1lb, CISD(H′), E₀′, walk offset and b_fact (`qpe_side`; flags
  "property_aware_encoding" and `encoding:<kind><parameters>`), keeps F9's rows (route_b_f9) and costs in
  F9's C_QPE units. The design costs the routes on F9's encoding, with which n2_r2.0's tighter accuracies
  are reached by no route (changed from the design specification; m4-results.md §13).
- Tests: tests/test_ft_properties.py::test_select_for_property_rules, ::test_property_selection_on_n2s (slow).

**F53 (comparison rule and physical cost).** [design; F18, F19]; the minimum factory failure [OF]. Code:
`ft.properties.best_property_row`, `ft.properties.PropertyRow`, `ft.properties.PROPERTY_ROUTES`,
`ft.properties.run_layout`, `ft.properties.best_at_budget`, `ft.properties.route_tally`,
`ft.properties.property_pairs`, `ft.properties.row_from_json`, `ft.properties.compiled_bias_limited`,
`ft.properties.model_bias_sensitivity`, `ft.properties.q6_comparison`, `ft.properties.q6_statements`.

- A `PropertyRow` (route "A", "A-oracle", "B0", "Bk" or "HF-QPE"; estimator "shadow", "ae", "ge<κ_g>" or
  "qpe"; loop, k, bias, Toffolis with the loop, loop Toffolis, per_prep, n_prep, longest_run, qubits, flags)
  is selectable iff its cost is finite, it is not the A-oracle row and it carries none of "not_usable",
  "not_terminating", "infeasible" and "oracle_k"; "termination_not_certified" and "no_layout" rows stay
  selectable.
- `best_property_row(rows, loop, estimator)` returns the cheapest selectable row (HF-QPE rows compete with
  every estimator; ties to the first row; ValueError with none, reported as none). Per (P, ε) the
  candidates are route A at the fixed budgets, B0, Bk and HF-QPE; AE at κ = 1 is the reference, the shadow
  rows are reported ("best free (shadows)") and gradient estimation is a band. U10b's "best free / meas r=1
  / meas r=0.3 (AE)" reads route [k] and cost; best_ls repeats it on the exact-line-search record.
- Physical cost (`run_layout`): a row's longest coherent run on its qubits by
  `ft.physical.surface_code_cost` (F18), a shadow snapshot at p_fail_shot with its 2d-cycle readout, every
  other run (one AE element, a GE sequence, one HF-QPE energy with its start) at p_fail_ae; shots and
  per-element AE runs parallelize over machines (F19), single coherent runs do not. No factory fails below
  3.6e-16 per Toffoli [OF], so no run beyond 2.78e11 Toffolis (2.78e10 for a snapshot) has a layout: such
  rows carry "no_layout" and stay selectable, since F53 compares Toffolis, and U10b's layout column reads
  `d<distance>` or "none" (changed from the design specification; m4-results.md §13). HF-QPE energies are
  laid out at p_fail_ae as single coherent property runs, not at F40's per-run budgets, which changes the
  layout column only (changed from the design specification; m4-results.md §13).
- U0's Q6 (`benchmarks/run_m4_summary.py`): `property_pairs` reads the (system, target) pairs; `route_tally`
  counts the best rows' routes per loop, route A at its cheapest usable budget per pair ("best") and at each
  single pre-declared k ("fixed_k", `best_at_budget`), "none" where nothing is selectable; `q6_comparison` adds the tally on the F25-sliced compiled iterates, the
  pairs bias-limited only there (`compiled_bias_limited`), F25's step errors and caps, and the cells resting
  on the exact model bias (`model_bias_sensitivity`); `q6_statements` words them.
- Tests: tests/test_ft_properties.py::test_best_property_row_rules, ::test_run_layout_and_the_no_layout_flag,
  ::test_route_tally_and_q6_comparison, ::test_route_a_fixed_budgets;
  tests/test_m4_properties_benchmark.py::test_best_rows_carry_the_layout_of_their_longest_run,
  ::test_u10a_marks_and_the_compiled_cell;
  tests/test_m4_summary.py::test_q6_tallies_both_accuracies_each_budget_and_the_layouts.

## Scope and open questions

What the benchmark run settled about the open questions is in m4-results.md, §12.

**Out of scope for M4:**

- Circuit-level checks of the QPE side: PREPARE/SELECT/QROM, GQSP phase finding, the filter phase sequences,
  the comparator of coherent QPE and the AA reflections (neglected [estimate]), Qualtran cross-checks;
  STAR-type partial fault tolerance; THC scale-up beyond N = 10. Hand-off costs stay in M3's window model
  (K₀ = 1) except through n_v and M_c (F40, F43); the windowed laws also give the identification errors and
  their bounds.
- In the closed loop: the aux routes (M3's costed rows are unchanged), ae_product and gradient estimation;
  correlated matchgate noise; pilot and sequential controllers; the free matchgate energy side channel;
  heavy-tail robustness (median of means); replicates at N = 10 and compiled propagation beyond N = 8;
  emulated logical failures and the finite-register AE law; a model of the noise floors beyond the
  replicate statistics.
- Step rules: the gated secant (trial residuals are costed by F30, but no rule takes one); Armijo and the
  2-point fit as running rules; trust regions, damped unit steps, Barzilai–Borwein and CG model steps;
  noise-tolerant quasi-Newton updates beyond the curvature test; F16-type curvature measurements beyond
  k = 0; a hardware estimate of ‖D_k‖ for the angle cap.
- Certificates beyond CISD(H′) and dedicated energies (CISDT(H′) is a sensitivity only): moment-LP and
  Clopper–Pearson overlap bounds, pilot runs, sequential tests, DMRG/SCI variational bounds; lower bounds on
  E₁, so E1lb stays an assumption as in M3; the energy shift ⟨Ψ_j|H′ − H|Ψ_j⟩ of dedicated certificates; loop
  energies as certificates (implemented, supplied by no costed rule). Spin filters before QPE and
  spin-adapted compilation; excited states; early-fault-tolerant phase estimation.
- Identification errors beyond N = 8; break-even sizes for the second measured iteration and, beyond the
  computed sizes, for the U-referenced, fp and E1lb-referenced schemes, where no certificate exists.
- Systems: localized orbitals and truncated residuals, H rings, R = 5.0 bohr, larger bases, N2 CAS(8,8)
  (impossible in STO-3G), a CI-space compiled replay beyond N = 8, H12 (the holdout is defined, not run).
  N = 10 has no model spectrum of H′, no compiled overlaps, no exact-first-step or spin-free R1 records and
  no LCU normalizations, hence the direct route only; every statement beyond n = 10, C_QPE included, is a
  model statement of F21–F23 and F45, and the locality crossover has no data. A realizable slicing
  criterion: F25's rule needs the exact step in CI space; fixed slices are realizable but neither
  calibrated beyond N = 8 nor run in U7.
- Properties: spin correlations and the unpaired-electron index; Hellmann–Feynman QPE for tensors and
  two-body scalars (the cumulant norm); κ_g of gradient estimation (a band); structured observable block
  encodings; closed-loop property studies beyond the three deep-run systems; complex states and frozen-core
  dipoles; compiled route-A iterates beyond N = 6; union-bound confidence over the elements of a tensor;
  circuit-level constructions of the copies (costed by formula only).
- Classical ACSE generators (3-RDM reconstruction): the free loop remains their bound.

**Open questions:**

1. The overlap model: the successive rates still drift at n = 10 (HF at R = 3.0: 0.1007, 0.1049, 0.1078 per
   atom; the R1 Ψ₂ Aitken limit reaches 4× the last rate at R = 2.5). Whether HF and the CQE iterates behave
   as matrix-product states with one size-independent rate over Δn ≥ 20 is untested, so n(p = 1e-3) and the
   F45 brackets carry an unquantified model risk.
2. The gap model: n·Δ₁ still rises at n = 10 (1065 to 1321 mHa at R = 2.5, 113 to 141 mHa at R = 4.0) and γ
   comes from n = 8 and 10 alone; n_cert and F45's lengthened QPE depend on whether the gap closes as 1/n or
   as n^−0.84.
3. E1lb is assumed: every rule-B certificate, the soundness of the filters, the termination of the
   E1lb-referenced forms and of route B, and HF-QPE's B_U rest on E1lb ≤ E₁; beyond the computed sizes m_BL
   also rests on sin θ₀ = 1 and E1lb = E₁(H) − 1 mHa, the one-run window on F22's p. A computed lower bound
   would make them unconditional.
4. The cost exponents come from two intervals (N = 6 → 8 → 10) and drift (T_res(1)/C_QPE, sf: 9.55 to 9.03
   at R = 2.5, 9.40 to 9.17 at R = 3.0; C_QPE 2.27 to 2.43 at R = 2.5, with non-convex THC fits); whether β
   falls toward the locality bracket beyond n = 20 is open.
5. Inputs at one state: F24 (i) rests on HF-state variances validated at N = 4–6 (≤ 1.4% in V_m, up to 22%
   per element), F24 (ii) adds a 2.2% sampling error at N = 10 and gives no per-element variances, and the
   property estimators take their inputs at ψ₀ for every measured state.
6. Slicing: η_s bounds one step's state error, not the overlap, energy or property error; the error against
   the exact trajectory accumulates (0.051 to 0.081 over k = 1–3 on h6_r4.0 r1_so), and the overshooting
   classical steps miss η_s even at s = 16. Whether a realizable criterion (a commutator bound of the
   product formula) or a fixed s reproduces the compiled ablation is open.
7. Energies charged at V_E(Ψ_{k+1}) understate the stencil end points (mean 2.09 V_E(Ψ_{k+1}) at k = 1 on
   h4_r4.0, spin-orbital): per-point variances would raise ρ_E(parab3) and R3's energy costs on small-gap
   systems and close its gate earlier.
8. R3's plan foresees neither the F29 caps nor where the fit lands (planned/charged T_en up to 1.6–2.3 on
   stretched N2, quick U5 run), and the realizable gate plans with V_E(HF), over-planning 2–3.3× on n2_r2.4
   and under-planning (0.75–0.81) on h4; a realizable state-dependent plan is open, and U7 measures R3 only
   with the oracle controller.
9. σ_P3 = 0.47|ΔE_m| is calibrated on an exact parabola with Gaussian errors and the exact ΔE_m;
   non-parabolic lines (stretched N2, ω_k = 0.03–0.41 in the quick run) and the noise of the debiased slope,
   whose residual bias in the loop is not quantified, are outside the calibration.
10. On stretched N2 no rule reaches 1.6 mHa within 40 steps, and U5 compares the rules at k_p90; whether
    that ranking carries over to the hand-off cost is a question for F44, U8 and U9.
11. The growth cap binds in 5–12 of the 28–39 measured steps of each quick-run R1 run and 2–5 times inside
    the cost horizon on stretched N2, so R1's iteration counts there depend on it; U5 has no uncapped R1 row.
12. The energy-precision wall makes energy checks lose near convergence at fixed r, but ρ_E ∝ r² falls below
    1 at r = 0.3 on small-gap and stretched systems (quick run, k = 1: ρ_E(parab3) = 0.05 on h4_r4.0, 0.33 on
    n2_r2.0, 0.18 on n2_r2.4): the residual accuracy at which a hybrid loop is cheapest is open.
13. The lagged plan realizes r₁ = 3.00r, 2.71r, 2.54r on stretched N2 but 0.46r on h4_r4.0; whether a
    realizable plan tracks the per-system optimum of the oracle r-scan is open.
14. The second budget is noise-dominated: at r₁ = 2.5–3, sd(n̂²)/‖A₁‖_m² ≈ 2.8, the floor binds at k = 1 in
    52–54 of 100 replicates on n2_r2.0 and n2_r2.4, and B₂ follows a noisy, often floored estimate (median
    r₂ 0.82–1.38); its share in T_E(3) and in the deep runs is not separated from the trajectory's.
15. Heavy-tailed small-gap cells (h4_r4.0 at oracle r = 2, 3: catastrophic shares 0.18, 0.35; mean T_E(2) =
    3.5 [1.5, 6.4] and 63.5 [9.1, 165] C_QPE) are not resolved by 100 replicates.
16. Correlated matchgate noise: the design's statement that element correlations leave T_E(2) unchanged
    within the replicate intervals is not re-checked in M4.
17. Overlaps on H against H′: the closed-loop p̃_k and route B's start overlaps are weights on ψ₀ of H, while
    the hand-off QPE and the copies act on H′ (agreement within 0.5% on stretched N2 is a design number,
    not re-checked on the closed-loop states); route B takes a = p.
18. (A3) reads one C_QPE as M = 2n_iter phase points; under the other reading every hand-off QPE length
    doubles, and g, D, n_v and the coarse lengths change, while under (A3) the step filter would take d walk
    steps, not F8's 2d (F43). The windowed ε_id are phase-offset lotteries, not monotone in M: the bounds use
    the supremum over offsets, U8 the actual offsets. The costing's K₀ = 1 understates every QPE term whose
    threshold lies within about 1.5 grid units of E₀′ (LiH's B_U at CISD(H′): K₀ = 0.529, F40).
19. Thin certificate margins: CISD(H′) leaves 3.08 mHa on n2_r2.4 and certifies nothing on n2_r2.8, while
    CISDT(H′), a septet 9.24 and 2.46 mHa above E₀′, would certify both; whether a high-spin truncated-CI
    minimum is an acceptable production certificate, and how the margins scale with size, is open.
20. Compiled states on stretched N2 populate a quintet 4.35 (n2_r2.4) and 1.21 mHa (n2_r2.8) above E₀′: no
    E1lb of any spin with a margin exists, and only the contaminant budget (q_c/p₀ of order 1e-4) keeps rule
    B sound.
21. Exponential search overshoots 1/p above p ≈ 0.3 (E_BBHT(p)p = 1.58 at p = 0.5), where lt_rep and
    lt_rep_L are cheaper and flip the h4_r3.5 verdict as a sensitivity; whether a realizable choice between
    repetition and exponential search, from a certified overlap bound, should enter the scheme set of record
    is open. The filter degrees come from a smoothed-step construction with a 1% margin, not from a minimax
    polynomial.
22. Beyond the computed sizes the realizable set reduces to rep and bbht, so the n* under the best
    realizable scheme depends on that restriction, and a saving hook that mixes QPE-scaled and filtered
    schemes is exact under neither form of the lengthening.
23. Route B and HF-QPE charge the exact model bias b_fact, a fact of the emulation like route A's usability
    at a pre-declared k: F47 at q = 1 − F is 4–24× larger on LiH's SF L = 16 and changes the best route on 6
    quick-run cells, and no certifiable bound as tight exists; on hardware route A's k would be fixed
    without knowing its bias.
24. F52's literal rule has no infidelity condition (LiH SF L = 15, infidelity 1.07e-4, 0.950 C_QPE): should it
    keep F9's infidelity ≤ 1e-4? The exact-AA leakage allocation, each leakage its own share, is a model
    choice; capping ε_f to leave room for w_AA was not explored, and the estimate of a is charged at half
    F11's count (F50).
25. HF-QPE assumes that the finite-field stencil bias falls below b as h is halved, takes the derivatives
    and stencil biases of H for those of H′, does not count the growth of λ by |h|α_O, and costs the filters
    and coarse QPE on H′ + hO as on H′.
26. The linearized NON operators are first-order exact only without coupling of degenerate natural
    orbitals, and the all-orders cumulant-norm bound is proven for real states only.
27. Best rows without a surface-code layout stay selectable (5 of the 48 best AE rows of the quick run at the
    tighter accuracy), and which failure budget an HF-QPE energy should carry (p_fail_ae, or F40's 1e-6 and
    5e-4) is open.

## Appendix A: reference numbers

Numbers pinned by the test named, or checked by a benchmark's reference table only: the "Reference
numbers" of `benchmarks/run_m4_scaling.py` (scaling_summary.md) and of `benchmarks/run_m4_loop.py`
(loop_summary.md), which compare them with the run. [design] marks the design's values, recomputed and
re-checked during the build. The tables are grouped by block.

**Systems and scaling (F20–F25).**

| quantity | value | pinned by |
|---|---|---|
| p_HF of h4_r3.0; Δ₁ of h6_r3.0 (same and any irrep); Δ₁ of h4_r1.8 (same / any) | 0.7012 ± 1e-4; 104.67 mHa; 565.72 / 563.34 mHa [design] | tests/test_ft_scaling.py::test_spectrum_gaps_and_overlaps |
| h4rect_1.8x1.8 | ground irrep B1g (PySCF id 1), HF determinant Ag, p_HF < 1e-24, flag "ground_state_not_in_hf_irrep" | tests/test_ft_scaling.py::test_spectrum_flags_and_few_roots; tests/test_ft_symmetry_names.py::test_h4_square_ground_state_flag |
| n2v_r1.098; n2v_r2.4 | N = 8, nelec (5, 5), 3136 determinants, p_HF = 0.9173 ± 1e-4; "spin_tol_relaxed" (1e-4), p_HF = 0.1112 [design] | tests/test_ft_symmetry_names.py::test_n2v_builds; tests/test_ft_scaling.py::test_spin_tolerance_fallback (slow) |
| N2 CAS(8,8) in STO-3G | ValueError (3 + 8 > 10 orbitals); CAS(8,7) and CAS(14,10) build | tests/test_ft_symmetry_names.py::test_cas_beyond_the_orbitals_raises |
| aliases h4, h6s, n2, n2s, h8 | bit-identical h1, eri, ecore, mo_coeff and irreps to M3's builds | tests/test_ft_symmetry_names.py::test_parametric_names_match_m3_builds (h8 slow) |
| allowed spin-free elements K of the chains | 28, 153, 496 at N = 4, 6, 8; 1225 at N = 10 [design] | tests/test_ft_symmetry.py::test_allowed_element_counts_match_appendix_a (N ≤ 8); N = 10 benchmark only (U3) |
| HF successive rates at R = 3.0 (n = 4, 6, 8) | 0.1007, 0.1049 per atom [design] | tests/test_ft_scaling.py::test_rates_on_chains; benchmark |
| H10 holdout | p_HF and the classical p₁ (sf, so) of H10 within 3% of the extrapolation from n = 4, 6, 8 at R = 3.0 and 4.0, both rates [design] | tests/test_ft_scaling.py::test_holdout_h10_from_h8 (slow) |
| h8_r4.0 and h10_r3.0 spectra; ε₂/ε* of the sf classical step of h8_r4.0; C_QPE of h8_r3.0 | p_HF 0.1685, Δ₁ 16.98 mHa; p_HF 0.3746, Δ₁ 69.04 mHa; 1.598; THC M = 16, 18,661,951 Toffolis [design] | tests/test_ft_scaling.py::test_regression_pins (slow) |
| selected QPE of n2_r2.4 | SF L = 16, λ = 7.5908 Ha, n_iter = 11,924, C_QPE = 9,157,632 Toffolis [design] | tests/test_ft_acceptance.py::test_model_spectrum_reference_numbers; benchmark |
| gap fit through 138.96, 104.67, 83.32 mHa at n = 4, 6, 8 | γ = 0.792972, A = 433.386 mHa, A′ = 666.56 mHa, n_cert(3 mHa) = [222.187, 529.197] | tests/test_ft_scaling.py::test_gap_window |
| m_BL at sin θ₀ = 0.94, E₁ − E1lb = 1 mHa, D_L = 5 | 11.4 mHa | tests/test_ft_scaling.py::test_gap_window |
| one-run B_L margin D_L(p) | 5, 6, 7, 9, 13, 27 grid units at p = 0.25, 0.1, 0.078, 0.03, 0.01, 1e-3; 0 at p = 1 | tests/test_ft_scaling.py::test_one_run_margin |
| two-run overlap floor | τ_W(5)²/(ε_id + τ_W(5)²) = 3.369e-5 (τ_W(5) = 1.8355e-4) | tests/test_ft_scaling.py::test_one_run_margin |
| one-run window, R = 3.0 HF-like inputs (p(10) = 0.37463, rates 0.1007, 0.1078; gap through 83.32, 69.04 mHa) | n = 37 / 35 (power law; last / Aitken), 34 / 32 (A′/n) | tests/test_ft_scaling.py::test_one_run_window_jointly_with_overlap |
| one-run window, gap growing as n^0.817 | ends at n = 126.51 (D_L 237 to 238); 83.70 at p(10) = 0.01 (D_L 169 to 170) | tests/test_ft_scaling.py::test_one_run_window_growing_gap |
| V_m at HF and at Ψ₁ (exact pair sums), h4_r4.0; V_m(HF)/V_m(Ψ₁), h4_r3.0 | 418.7 and 420.5 Ha²; 480.7/478.1 [design] | tests/test_ft_scaling.py::test_hf_variances; ::test_scan_large_n_path |
| subset totals | unbiased: the 28 leave-one-out subsets of h4_r4.0 average to V_m exactly; 20 seeds on h6_r3.0 within 2 standard errors | tests/test_ft_scaling.py::test_subset_totals_formula, ::test_subset_totals_full_set_and_mean, ::test_subset_totals_unbiased (slow) |
| closed-form direct route | M3's direct_mg exactly; T(r = 0.3) = 100T(r = 3) | tests/test_ft_scaling.py::test_cheapest_iteration_routes |
| T_res(1)/C_QPE at r = 1 (classical-step loop) | n2_r2.0 / 2.4 / 2.8 so 0.515 / 0.423 / 0.398; h6_r3.0 sf 1.81, h8_r3.0 sf 27.0, h10_r3.0 sf 209 [design] | benchmark (the HF-state variants 0.43, 0.41 of n2_r2.4, n2_r2.8 are its two misses) |
| p_HF, Δ₁ and C_QPE of the scanned systems; E − E₀ of n2_r2.4 | the design values of the reference table; HF 777.01, classical Ψ₁ so 301.43, exact-line-search Ψ₂ 232.16, unit Ψ₂ 252.43 mHa [design] | benchmark |
| slicing law, h4_r4.0 exact-line-search first step (ε* = 0.522) | error ratio s = 1 → 2 in [1.8, 2.2] (2.08), \|p̃₁/p₁ − 1\| ≤ 1% at s = 2 (0.55%); errors 0.128 / 0.062 / 0.030 / 0.015 at s = 1 / 2 / 4 / 8 [design; the four errors not pinned] | tests/test_ft_scaling.py::test_slicing_law |
| slice rule on the classical spin-orbital first step; along h4_r4.0's record | s = 16 with step error 0.0332 (h4_r4.0), 0.0507 and "slice_cap" (h6_r4.0) [build check]; s = 16, 8, 1 at k = 1, 2, 3 | tests/test_ft_scaling.py::test_slice_rule_continues_to_16; ::test_slice_rule_follows_its_own_state |

**Step rules (F26–F31).**

| quantity | value | pinned by |
|---|---|---|
| F26 identity ⟨d, H_k⁻¹d⟩ = −s₀ on 12-step unit runs (h4_r3.0, n2_r2.0, both metrics) | 1e-12 relative (9.4e-16 on 44 directions [build check]) | tests/test_steprules.py::test_model_step_is_unit |
| unit rule (caps off, masked exact residual) against `cqe.run_cqe(first_eps=ε₂, eps_candidates=(1.0,))`, 10 steps of h4_r1.8 and n2_r2.0 | every energy to 1e-12 Ha (1.4e-14 [build check]) | tests/test_steprules.py::test_unit_rule_equals_run_cqe |
| line-search rule, classical first step, against `exact_trajectory(first_step="second_order")`, h4, 8 steps; with `first="line_search"` against M3's record (h4, n2s, 12 steps) | 1e-10 Ha (1.9e-11 [build check]); bit for bit in ε and every energy | tests/test_steprules.py::test_line_search_rule_equals_exact_trajectory; ::test_measured_loop_equals_m3_exact_record |
| σ/\|ΔE_m\| of F27: parab3_linear (ℓ = 0.1), parab3, armijo (z = 2, c₁ = 1e-4), parab2 | 0.894 (√0.8), 0.47, 0.3535, 0.167 | tests/test_ft_steprules.py::test_energy_precision_values |
| losses of the 3-point fit at 0.47 and 0.894\|ΔE\|; of the 2-point fit at 0.16\|ΔE\|; table rebuild (4e5 draws) | 0.10 ± 0.01 and 0.264 ± 0.01; 0.088 ± 0.003; within 1% | tests/test_ft_steprules.py::test_precision_table_monte_carlo (slow) |
| linearized 3-point error ratio at u ∈ {½, 1, 2}; Armijo false rejection at σ_A | within 5%; Φ(−2) = 0.0228 within 3 standard errors | tests/test_steprules.py::test_f27_monte_carlo (slow) |
| C_loop(k* = 3)/C_QPE of the line-search rule on h4 (sf, classical first step, r = 0.3), C_iter(k) = `ft.pipeline.iteration_costs` | 2416.996 (M3's T4) | tests/test_ft_steprules.py::test_m3_costing_reproduced; ::test_run_step_rule_system |
| k* of spin-orbital R1 | 3 (h4_r1.8), 7 (h4_r3.5), 11 (h6_r3.0), 3 (n2_r1.098) [design] | tests/test_ft_steprules.py::test_r1_iteration_counts (slow); benchmark (h4_r1.8 in quick mode) |
| R3 gate on h4_r4.0 (so, 4 steps, V_E at exp(tD_k)Ψ_k), k = 1, 2, 3 | fit, fit, unit at r = 0.3; fit, unit, unit at r = 1 (V_E(HF) would keep the unit step at r = 1, k = 1) | tests/test_ft_steprules.py::test_u5_gate_plans_at_the_planned_step |
| R3 on h4 (sf, r = 0.3, 8 steps) | the gate never opens; C_iter equals R1's; k* = 3 | tests/test_ft_steprules.py::test_run_step_rule_system |
| converted classical-first-step record against `exact_trajectory(first_step="second_order")`, h4 | energies within 1e-10 Ha; hand-off rows to 1e-9 relative, E_k − E₀ to 1e-10 Ha, saving_vs_hf to 1e-9 C_QPE/p₀ | tests/test_ft_steprules.py::test_record_conversion_feeds_handoff |
| k_ref of spin-orbital R1 on n2_r2.0 (r = 1, 10 steps) | k* not reached; k_ref = k_p90 = 7 | tests/test_ft_steprules.py::test_u5_reference_iterate_n2 (slow) |

**Closed loop (F32–F38).** T_E(k) in C_QPE units (M3's rule E at every k, certified or not, F38); 100
replicates unless noted.

| quantity | value | pinned by |
|---|---|---|
| n2_r2.4, oracle r = 1: mean T_E(2); median p̃₂ | 3.955 [3.894, 4.018], window [3.85, 4.06] (design 3.958 ± 0.030); 0.288, window [0.27, 0.30] (design 0.284) | tests/test_ft_closedloop.py::test_headline_regression (slow); benchmark |
| n2_r2.4, oracle r = 3: mean T_E(2) | 3.603 [3.496, 3.714], window [3.50, 3.78] (design 3.643 ± 0.047) | tests/test_ft_closedloop.py::test_headline_regression (slow); benchmark |
| n2_r2.4, lagged r = 1 (deciding): mean T_E(2); P(T_E(2) < T_E(1)) | 3.601 [3.496, 3.711], window [3.45, 3.80]; 0.96 (≥ 0.9 required) | tests/test_ft_closedloop.py::test_headline_regression (slow); benchmark |
| n2_r2.0, oracle r = 3: mean T_E(2) | 2.257 [2.221, 2.291], window [2.20, 2.32] (design 2.259 ± 0.018) | tests/test_ft_closedloop.py::test_headline_regression (slow) |
| h4_r1.8, oracle r = 1, 20 replicates: per-replicate optimum at T_E(1) | 20 of 20 (≥ 19 required) | tests/test_ft_closedloop.py::test_headline_regression (slow) |
| n2_r2.0 oracle r = 1; n2_r2.8 oracle r = 1 / 3; median p̃₂ at oracle r = 3 of n2_r2.0, n2_r2.4 | design T_E(2) 2.638 ± 0.010, 4.691 ± 0.046 / 4.327 ± 0.068 and p̃₂ 0.471, 0.235 / 0.235; 0.453, 0.279 (inside the 16–84% bands) | benchmark |
| sine-window lattice law P(n − u), every offset u | mass 1 and mean 0 to 1e-12, variance ¼ to 1e-5 | tests/test_ft_closedloop.py::test_sine_window_law_identities |
| finite-register sine-window constant κ(D), D applications of G | 0.9736, 0.9934, 0.9983 at D = 63, 255, 1023 (within 5e-4) | tests/test_ft_closedloop.py::test_sine_window_finite_register |
| sup_u P(\|Δ\| > 5); variance deficit of the \|n\| ≤ 60 lattice | 2.84e-4 (in [2.8e-4, 2.9e-4]), 0 at u = ½; ≤ 1.7e-3 | tests/test_ft_closedloop.py::test_sine_window_tails |
| noise-free limit: h4 C_iter(k), k = 0…7 (sf L-BFGS, m3_line_search, oracle at r = 0.3, both first steps) | M3's tests/data/ft_h4_golden.json, Toffolis equal; states, energies, ε np.array_equal to `exact_trajectory` | tests/test_ft_closedloop.py::test_noise_free_limit_is_m3 |
| percentile-bootstrap coverage (100 Gaussian differences, 2000 resamples) | 0.92–0.98 over 200 synthetic sets | tests/test_ft_closedloop.py::test_summarize_statistics |
| realized-r spread (2/K_eff)^{1/2} and norm precision (4r²/K₁ + 2r⁴/K₂)^{1/2} at Ψ₁ of h4, r = 1 | within 10% of Monte Carlo | tests/test_ft_closedloop.py::test_direct_estimator_statistics, ::test_controllers |

**Hand-off (F39–F45).** p₀ on H′ unless noted; C_QPE units.

| quantity | value | pinned by |
|---|---|---|
| n2_r2.4 spectrum of H′ | E₁′ − E₀′ = 230.863 mHa, E₀′ − E₀ = −0.368 mHa (PySCF FCI of H′ to 1e-8 Ha) | tests/test_ft_acceptance.py::test_model_spectrum_against_fci |
| n2_r2.4 sin θ₀; sine RMS at M = 2n_iter | 0.9857; 0.9856 mHa over 8 offsets (half at 4n_iter) | tests/test_ft_acceptance.py::test_sine_rms_is_de_sin_theta |
| CISD(H′) − E₀′; E₁′ − E₀′ | 134.215 / 225.686 / 293.495 mHa (n2_r2.0 / 2.4 / 2.8); 191.907 (n2_r2.0), 244.441 (n2_r2.8) | tests/test_ft_acceptance.py::test_model_spectrum_reference_numbers; tests/test_ft_certify.py::test_cisd_values |
| CISD(H) − E₀ | 133.608 / 224.810 / 292.418 mHa | tests/test_ft_certify.py::test_cisd_values |
| truncated CI of H′ | singlet CISD 159.641 / 263.969 / 331.066; CISDT (septet) 54.752 / 9.241 / 2.455 mHa; h6s CISD 54.506, CISDT 51.399 (singlets) | tests/test_ft_certify.py::test_truncated_ci_table |
| p₀ of HF / Ψ₁ / unit Ψ₂ | n2_r2.0 0.25390 / 0.39761 / 0.47300; n2_r2.4 0.11201 / 0.21252 / 0.28103; n2_r2.8 0.07815 / 0.16468 / 0.22942; h4_r4.0 HF 0.4341 | tests/test_ft_acceptance.py::test_identification_error_table |
| identification errors | F40's table, two digits (5%), except h4_r4.0's literal entry | tests/test_ft_acceptance.py::test_identification_error_table, ::test_identification_errors |
| sine tail τ_W | τ_W(5) = 1.8355e-4 (closed form 1.84e-4), τ_W(28.7) = 7.30e-7; τ_W(1.3) = 9.389e-3 with the jump point (M = 4096 there: 9.352e-3) | tests/test_ft_acceptance.py::test_verification_runs, ::test_tail_mass_against_the_closed_form, ::test_tail_mass_is_the_supremum_with_the_jump_point |
| n_v and its thresholds | n_v = 1, 2, 3 at D = 28.7, 5, 2.07; t₁ = 26.036, t₂ = 3.032, t₃ = 1.295 | tests/test_ft_acceptance.py::test_verification_runs, ::test_verification_run_thresholds |
| B_L overlap floors; rect window | 0.155 (one run), 3.4e-5 (two runs); t₁ = ∞, verification_runs(5000, rect) = 2 | tests/test_ft_acceptance.py::test_verification_runs; ::test_expected_runs_where_a_run_count_suffices_nowhere |
| logical failures | f = 0.446 (n2_r2.4); p_run = 1e-6 (unverified), 5e-4 (retested) at f = 1 | tests/test_ft_acceptance.py::test_logical_failure_term |
| rule-B forms | n2_r2.0 B_U, D_U = 28.7, n_v = 1; n2_r2.4 B_L, n_v = 2 (B_U at D_U = 2.07 needs 3); n2_r2.8 B_L, "u_outside_bracket" | tests/test_ft_acceptance.py::test_bracket_forms |
| expected runs of dedicated certificates | n2_r2.8 D(U) = 26.52, σ = 17.1 mHa: P(second run) = 0.478, B_L taken; n2_r2.4 D(U) = 33.41: E[n_v] = 1.25 | tests/test_ft_handoff_rule_b.py::test_dedicated_certificates_cost_their_expected_runs; tests/test_ft_acceptance.py::test_expected_verification_runs |
| n2_r2.4 end to end (U8) | R1 HF T_B = 9.928 (= U9's rep); exact_so Ψ₃ with its sampled certificate, E[n_v] = 1.25, T_B = 1021.11 | tests/test_m4_handoff_benchmark.py::test_handoff_rows_of_n2_r2_4 (slow) |
| M3's T6 under rule B (p on H) | h6s measured HF 2.744 (B_U, D_U = 25.4, n_v = 2); n2s sf classical-first-step HF 3.956; n2s so classical-first-step Ψ₁ 2.525 (B_U, D_U = 28.7, n_v = 1) | tests/test_ft_handoff_rule_b.py::test_rule_b_on_m3_t6 |
| sector levels, exact H; of H′ | n2s singlet E₁ 191.7, same-irrep quintet 20.9 mHa; h6s 104.7, triplet 88.6 mHa; quintets of H′ 21.395 / 4.346 / 1.210 mHa (n2s / n2_r2.4 / n2_r2.8), E₁(H′) − E₁(H) = −0.06 / +0.56 / +0.75 mHa | tests/test_ft_certify.py::test_sector_levels; ::test_sector_levels_on_model_and_irrep_weights |
| any-irrep singlet E₁ | n2 (exact H) 411.7 against 711.5 mHa; h4 (SF L = 7) 563.347 against 565.666 mHa | tests/test_ft_certify.py::test_sector_levels_any_irrep_n2; tests/test_ft_acceptance.py::test_e1_any_irrep_against_fci |
| contaminant of the compiled n2s Ψ̃₁ | q_c/p₀ = 1.2e-4 ± 20% below CISD(H′) + ΔE | tests/test_ft_certify.py::test_contaminant_weight_compiled_n2s |
| known-overlap AA | N\*√a = 1.3801 at a = 1e-6; N*(0.25) = 3 (m = 1), N*(0.32) = 3.125 (m = 0); x/sin²x = 1.380050 at x = 1.165561 | tests/test_ft_amplify.py::test_grover_values |
| exponential search | E_BBHT√a ∈ [1.796, 2.899] for a ≤ 0.1; E_BBHT(p)p = 1.0027 / 1.5785 / 1.1565 at p = 0.30 / 0.5 / 0.918 | tests/test_ft_amplify.py::test_exponential_search_values, ::test_filter_then_repeat_sensitivity |
| step-filter degrees | d = 446 / 197 (ε_f = 1e-2, Δθ = 0.02 / 0.05), 910 / 364 (1e-3); grid-only check 354 against 364; c_f(1e-2) = 9.999, c_f(1e-3) = 18.483; table 617 ≥ construction 607 at Δθ = 0.03, 1e-3 | tests/test_ft_amplify.py::test_step_filter_degree, ::test_filter_band_edges_need_the_exact_check, ::test_filter_degree_from_the_table |
| coarse QPE lengths | edge levels only: M = 182 passes, none of 183–246; robust M_c = 704 (θ_pass = π/2 + 0.17, Δθ = 0.1, ε_p = 1e-4); n2_r2.4 E1lb-referenced edges M_c = 5216; D_p = 6.0 | tests/test_ft_amplify.py::test_coarse_edge_levels_are_a_lottery, ::test_coarse_length_from_the_continuum_is_minimal, ::test_two_stage_qpe |
| scheme table (design inputs) | p = 0.11201276: rep 8.928, aa 4.113, bbht 5.301 (+1 each at n_v = 2); p = 0.21252473: 4.705, 3.054, 3.768; n2_r2.0 filters, p = 0.25390: lt 1.654, lt_bbht 1.765, 2s 2.622, lt_bbht_L 1.441, 2s_L 1.934; p = 0.39761: 1.548, 1.697, 2.036, 1.401, 1.597 | tests/test_ft_amplify.py::test_scheme_costs |
| best realizable scheme | lt_bbht_L on every row of the design's table; h4_r4.0 HF (no certified U) rep at 3.303712, bbht 4.184901 (n_v = 2) | tests/test_ft_amplify.py::test_best_realizable, ::test_state_scheme_costs_without_a_certified_u |
| n2_r2.4 HF row of U9 | p = 0.11201276 on H′ (0.11246346 on H); rep 9.927554, lt_bbht_L 1.705232, s* lt_bbht_L | tests/test_m4_handoff_benchmark.py::test_scheme_row_takes_p_on_h_prime |
| filter-then-repeat sensitivity (h4_r3.5 sf r = 1 k = 1) | lt_bbht 1.651979 → 1.619557 (pays); lt_rep 1.563524 → 1.598354 (does not) | tests/test_ft_amplify.py::test_filter_then_repeat_sensitivity |
| scheme table from the model against the design's | lt, lt_bbht +2 to +5.6%, lt_bbht_L +1.7 to +2.4%, 2s and 2s_L −3 to −8%; rotations 5.1–5.2% of C_filt; walk part = closed form to 0.5% | tests/test_ft_amplify.py::test_scheme_table_from_the_model (slow) |
| U4 repetition row h4_r3.0 sf r = 1 k = 1 | μ₁ = 0.0367, ρ₁ = 1.457 (pinned 1.45 ± 0.02), p₁* = 0.650 | tests/test_ft_scaling.py::test_scaling_report_tables |
| F44 marginal condition | T_E(1) < T_E(0) iff ρ₀ < 1 iff p₀ < p₀* on 300 random cases | tests/test_ft_scaling.py::test_break_even_overlap_matches_marginal_condition |
| F45 reference sizes | n* = 682 (last) / 634 (Aitken); 938 without preparation terms, none with them, 2534 with c₂ at c₁'s exponent; n* = 92 ahead of the 1e-300 floor | tests/test_ft_scaling.py::test_break_even_size_rates, ::test_break_even_size_preparation_terms, ::test_break_even_size_underflow |
| F45 windows on H-chain inputs (sf, r = 1, m_E) | R = 3.0: 1/36 inside (n* = 410 against (A/m)^{1/γ} = 414.21; minima 410 power, 396 A′/n); R = 2.5: 6/36 inside | tests/test_ft_scaling.py::test_inside_window_regression_values |

**Properties (F46–F53).**

| quantity | value | pinned by |
|---|---|---|
| 2-RDM orbits | {1, P, T, PT}: 76 (N = 4), 351 (N = 6), Σ m_e = N⁴; masked (F2a): 44 (h4), 189 (h6s), 141 (lih, C2v), 69 (n2, D2h) | tests/test_ft_properties.py::test_rdm2_orbits |
| LiH dipole μ_z (origin 0) | −1.8177157 a.u. (ψ₀), −1.9110889 a.u. (HF), to 1e-6; PySCF's `dip_moment` to 1e-10 | tests/test_ft_properties.py::test_dipole_matches_pyscf |
| V_m of the h4 2-RDM at ψ₀ (44 elements, exact pair sums) | 1511 (relative 1e-3) | tests/test_ft_properties.py::test_property_inputs_on_h4 |
| N_ex at a = 0.434, 0.254, 0.2125, 0.112, 0.078 | 3, 3, 5, 5, 7; N_ex(0.25) = 3, N_ex(0.2499) = 5, N_ex(sin²(π/10)) = 5, N_ex(1) = 1 | tests/test_ft_properties.py::test_copy_counts |
| LiH E‴ of the dipole field (21 fields); HF-QPE optimum at ε = 4e-3 a.u. without model bias, from HF (p = 0.974), central | 581.6 a.u. (in 560–600); h = 4.88e-3 ± 0.02e-3 a.u., T = 91.1 ± 5 C_QPE | tests/test_ft_properties.py::test_hellmann_feynman_optimum |
| LiH HF-QPE with b_m = 4.7146e-4 a.u. (E‴ = 581.571, E⁽⁵⁾ = 6.96513e6) | 4e-3 central: b = 2.0004e-3, h = 4.5429e-3, C_QPE(σ_E) = 49.49 C_QPE; 1e-3 five-point: b = 2.0247e-4, h = 5.4343e-3, 236.66 C_QPE | tests/test_ft_properties.py::test_hellmann_feynman_fixed_bias |
| LiH dipole b_fact of SF L = 16; HF-QPE row at 4e-3 | 4.71e-4 ± 0.05e-4 a.u.; 95–105 C_QPE | tests/test_ft_properties.py::test_route_b_hellmann_feynman_rows_on_lih |
| LiH central stencil bias at h = 1e-3; stencil bias at the leading-order h | 9.69e-5 a.u. (1%); > 1.08b (five-point, 1e-3), > 1.005b (central, 4e-3) | tests/test_ft_properties.py::test_finite_field_bias_on_lih (slow) |
| F47 scalar case (σ₀ = 1.144, r_O = 8.674 a.u., q = 0.3) | 3.650694 | tests/test_ft_properties.py::test_bias_bound_scalar_and_max_forms |
| LiH dipole F47 model-bias bound at the infidelity of SF L = 16 | 3.3e-3 a.u. > 1e-3: HF-QPE infeasible at 1e-3 without b_fact | tests/test_ft_properties.py::test_default_model_bias_is_f47_at_the_infidelity |
| n2s property-aware encoding for NONs at 1e-3 | F9's SF L = 17 has b_fact > 0.5e-3 (9.7e-4); F52 takes SF L = 18 at 1.00–1.05 C_QPE | tests/test_ft_properties.py::test_property_selection_on_n2s (slow) |
| compiled-iterate distances (exact line search, so, classical first step) | 1.8e-3 at k = 10 (h4), 2.9e-4 at k = 14 (lih), within 10%; 8 slices on steps 0–1: < 1.4e-3 at 1.6–1.9 × C | tests/test_ft_properties.py::test_compiled_floors (slow) |
| smallest factory failure; layout limit at p_fail_ae | 3.6e-16 per Toffoli; 3.95e11 Toffolis on 261 qubits has no layout, 2.5e11 has one | tests/test_ft_properties.py::test_run_layout_and_the_no_layout_flag |
| cumulant-norm error, h4 at q = 0.5 (BFGS-maximized) | 3.648 against the linearized bound 3.431 (± 0.01), below the all-orders bound | tests/test_ft_properties.py::test_cumulant_norm_bound_adversarial (slow) |
| h4 (= h4_r1.8) R1 spin-orbital record | converges at k_end = 29 < 60 | tests/test_ft_properties.py::test_route_a_budgets_beyond_the_record |
| HF-QPE verification runs on the long QPE's grid | B_L without U: n_v = 2; U 20 mHa below E1lb: n_v = 1 on the long grid, 2 on a 1-mHa grid | tests/test_ft_properties.py::test_hf_qpe_acceptance_on_the_long_qpe |
| matchgate variances against snapshots (h4, six elements per target) | \|z\| ≤ 4 at 2e4 snapshots | tests/test_ft_properties.py::test_matchgate_variances_match_snapshots (slow) |
