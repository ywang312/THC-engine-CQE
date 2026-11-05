# Milestone 1 results

All numbers come from noise-free CI-space emulation, produced by
`benchmarks/run_thc_floors.py` and `benchmarks/run_generator_rank.py` with the
commands in the README. The systems use STO-3G:

- H_n chains at 1.8 bohr; `h6s` is H6 at 3.0 bohr.
- LiH at 1.595 Å.
- N2 CAS(6,6) at 1.098 Å; `n2s` is the same at 2.0 Å.

> **N2 rows updated in M2.** `systems.n2` now builds symmetry-adapted (D2h)
> orbitals with the degenerate π pairs axis-aligned (see docs/m2-results.md). Before,
> their orientation was set by round-off and changed with the BLAS thread count.
> Energies, leaf counts, ranks, DF floors and λ do not depend on it; element counts,
> CNOT estimates of elementwise compilation and THC fits do. In the symmetry-adapted
> basis the N2 generators are sparser: 14 [30] instead of 24 [59] canonical elements
> at 99% (n2s: 16 [35] instead of 25 [57]), and elementwise compilation is 4.4× cheaper
> per step than leaves instead of 2.2×. The N2 rows below are the regenerated values.

CNOT counts are model estimates (`diagnostics.cost`) for all-to-all connectivity.

## 1. With a factorized Hamiltonian, measure the energy with the full H

Suppose the residual is built from a model Hamiltonian H′ = H + V, with THC or
truncated-DF integrals. The CQE then converges to Ψ′, the ground state of H′,
and two energies can be reported:

- **E₀′ − E₀**, the energy of H′ at its own ground state. This error is first order in V.
- **⟨Ψ′|H|Ψ′⟩ − E₀ ≥ 0**, the energy of the exact H at the same state. This error is second order.

Selected rows; † marks a THC refinement that stopped at the iteration limit:

| system | model | max\|ΔV\| | E₀′ − E₀ | ⟨Ψ′\|H\|Ψ′⟩ − E₀ | ‖A[H](Ψ′)‖ |
|---|---|---|---|---|---|
| H6 | DF, L = 11, Ξ = 63 | 2.2e-4 | +2.3e-4 | 1.3e-7 | 1.8e-3 |
| H6 | THC, M = 2N † | 1.4e-4 | −2.6e-4 | 3.0e-7 | 4.4e-3 |
| H8 | DF, L = 15, Ξ = 114 | 3.0e-4 | +6.2e-4 | 3.9e-7 | 4.3e-3 |
| H8 | THC, M = 2N † | 1.8e-4 | −4.6e-4 | 3.3e-7 | 4.5e-3 |
| LiH | DF, L = 18, Ξ = 60 | 9.7e-4 | +1.0e-3 | 1.9e-6 | 3.7e-3 |
| N2 | DF, L = 18, Ξ = 72 | 2.1e-3 | −5.5e-3 | 2.1e-6 | 6.5e-3 |

The DF rows were re-checked with the core code alone (FCI of H′, then
⟨Ψ′|H|Ψ′⟩ and the residual of H). They agree to the printed digits.

This has two consequences for a NISQ CQE:

- **Energy.** A truncated factorization can drive the residual, meaning both
  the measurement bases and the generator. The energy should then be measured
  with the full Hamiltonian, which gives the second-order error. Measuring it
  with the same H′ gives the first-order error, 500–3000× larger in the rows above.
- **Residual floor.** The exact-H residual cannot drop below
  ‖A[H](Ψ′)‖ ≈ ‖A[H′](Ψ₀)‖, which is 3–30 × max|ΔV| here. This floor matters only
  once the shot noise on the residual is below it.

## 2. THC versus DF

- **Parameter count.** At matched integral error THC needs far fewer
  parameters. On H8, THC at M = 16 uses 264 parameters for max|ΔV| = 1.8e-4. DF
  needs Ξ = 114–117 eigenvectors, about 1000 parameters, for 3e-4 to 1e-4.
- **Fitting.** The THC fit is non-convex. The refinement originally ran in the
  grid-guess gauge, where the Hessian blocks differ in scale by 10⁸–10¹⁰, and
  it stalled. It now uses unit-norm leaves. Even so, most intermediate ranks stop
  at the 5000-iteration limit, and their floors can change by factors of several
  when the integrals change at the round-off level.
- **λ.** Without the λ penalty, THC λ is erratic:
  - LiH at exact rank (M = 21) has λ = 68.9, against 9.26 for DF.
  - H8 at M = 36 has λ = 804, against 17.6 at M = 32.

  OpenFermion's penalty rule (`fit_thc(..., penalty="auto")`) lowers λ, at the
  cost of a larger integral error.
- **N2.** THC struggles here: M = 2N leaves max|ΔV| = 4.6e-2, and M = 3N reaches
  1.4e-3. DF with L = 18 reaches 2.1e-3.
- **OpenFermion agreement.** λ and Toffoli counts reproduce OpenFermion: FeMoco
  THC λ = 294, DF λ = 294.8, and the `compute_cost` reference values. The
  fault-tolerant pipeline therefore works, but for N ≤ 8 its numbers only check
  that pipeline.

For a NISQ CQE, DF is the practical choice. THC is for the fault-tolerant
block encoding, with the λ penalty on.

## 3. CQE generators are not low rank

- **First step from RHF.** A_ovov = W = 4(ia|jb) − 2(ib|ja) to machine precision,
  so the first generator has K₁ = 2 rank(W) leaves. The Cholesky rank r_ov of
  (ia|jb) alone does not bound K₁; for N2, r_ov = 8 but K₁ = 18, because the
  exchange term adds rank.
- **Later steps.** From the second step on, the leaf rank saturates at or near
  N(N−1), the maximum for a real generator. Capturing 99% of ‖g‖², and so 99% of
  the steepest-descent slope, takes 50–85% of the leaves.
- **Basis independence.** The leaf magnitudes |c_k| are the singular values of
  g reshaped to N²×N², so none of this depends on the orbital basis.

Medians over iterations 1–5 of the exact CQE:

| system | N | K₁ | r_ov | leaf rank | N(N−1) | leaves @99% | canonical elements @99% [spin-orbital excitations] |
|---|---|---|---|---|---|---|---|
| h4 | 4 | 8 | 4 | 12 | 12 | 8 | 11 [27] |
| h6 | 6 | 18 | 9 | 30 | 30 | 18 | 39 [103] |
| h6s | 6 | 18 | 9 | 30 | 30 | 20 | 76 [204] |
| lih | 6 | 16 | 8 | 28 | 30 | 14 | 18 [39] |
| n2 | 6 | 18 | 8 | 26 | 30 | 18 | 14 [30] |
| n2s | 6 | 18 | 8 | 26 | 30 | 22 | 16 [35] |
| h8 | 8 | 32 | 16 | 56 | 56 | 28 | 88 [241] |

## 4. Compiling a step: elements beat leaves

| system | one step @99%: leaves | elements, canonical | leaves / elements | to 1.6e-3 Ha: leaves | elements, canonical | leaves / elements |
|---|---|---|---|---|---|---|
| h4 | 664 | 315 | 2.1 | 984 | 338 | 2.9 |
| h6 | 3516 | 1507 | 2.3 | 5052 | 1800 | 2.8 |
| h6s | 3900 | 2978 | 1.3 | – | – | – |
| lih | 2268 | 538 | 4.2 | 580 | 101 | 5.7 |
| n2 | 2452 | 554 | 4.4 | 3836 | 878 | 4.4 |
| n2s | 2916 | 647 | 4.5 | – | – | – |
| h8 | 9968 | 4401 | 2.3 | – | – | – |

- **Canonical elements win.** Compiling element by element in canonical
  orbitals needs 1.3–4.5× fewer CNOTs per step at 99%. It needs 2.8–5.7× fewer to
  reach chemical accuracy, on the four systems where any run gets there.
- **Localization hurts.** Localizing the orbitals makes the elementwise scheme
  more expensive. Boys localization within the occupied and within the virtual
  orbitals costs 1.4–3.1× canonical, and whole-space Boys costs 1.6–14×.
- **Trotter error is small.** Trotterizing the kept leaves, which is the
  circuit the leaf CNOT count describes, changes little at these step sizes.

Caveats:

1. The canonical orbitals of these small, symmetric molecules are
   point-group adapted, which makes elementwise generators sparse. The leaf
   cost model exploits no symmetry.
2. All counts are estimates. They assume no gate cancellation, a naive
   2-CNOT ZZ term, and the published 13-CNOT double-excitation core; the
   circuit verified here uses 14.
3. The elementwise runs use the exact exponential of the kept sum. On H4, a
   review check found that the first-order product moves the trajectory by at
   most 1e-5.
4. The cost of measuring the residual is not modeled. It is likely the
   dominant NISQ cost of the CQE.

## 5. Steepest descent limits the tail

Even without compression, the exact steepest-descent CQE is still above
chemical accuracy after 15 steps for:

- stretched H6 (4.6e-3 Ha),
- stretched N2 (4.8e-2 Ha),
- H8 (5.5e-3 Ha).

No compressed run reaches chemical accuracy within 20 steps on these systems.
The optimizer causes this, not the compression.

## Implications for the next milestone

1. **Model the residual measurement.** Count shots for A using DF-grouped
   measurements and direct shadows of Ψ, with the energy measured in the full
   Hamiltonian (section 1).
2. **Replace steepest descent** with a quasi-Newton or momentum update before
   comparing compression schemes on harder systems.
3. **Keep generator DF as an option**, not the default compilation. Revisit it
   for larger, less symmetric active spaces, where elementwise sparsity is lost.
4. **On the fault-tolerant track**, use THC with the λ penalty and compare λ at
   matched floors rather than matched integral error.
