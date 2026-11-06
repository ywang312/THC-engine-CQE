# Milestone 2 results

M2 asks what the CQE costs when the residual and the energies are estimated
from measurements, and replaces steepest descent with better optimizers. All
numbers come from CI-space emulation with STO-3G and the systems of
[m1-results.md](m1-results.md):

- H_n chains at 1.8 bohr; `h6s` is H6 at 3.0 bohr and `h6a` is H6 at 1.0 Å, the
  geometry of the earlier claims.
- LiH at 1.595 Å.
- N2 CAS(6,6) at 1.098 Å; `n2s` is the same at 2.0 Å.

Every table can be regenerated with the benchmark scripts named in its section,
run with `--threads 1`.

**Main results.**

1. **Optimizers (section 1).**
   - The standard CQE steps along the spin-orbital ACSE generator. It is the
     spin-free generator preconditioned by M = (2I + T)/3, and it reaches
     chemical accuracy 1.6–2.4× faster than the spin-free steepest descent of M1.
   - CG and L-BFGS preconditioned with the same metric are a further 1.5–3×
     faster on H6, H6s and H8.
2. **Why the CQE stalls (section 2).** The exact CQE's slow tails sit in soft
   modes of its linearization. N2 has exact null modes, which hold its
   1.8e-4 Ha plateau.
3. **Measurement cost (section 3).**
   - *Energy.* Shifted DF grouping costs 2.3e5–8.3e5 shots per ±1.6 mHa energy.
   - *Residual.* The direct route with matchgate shadows is the cheapest
     unbiased estimator; the spin-free elements need about half the shots of the
     spin-orbital ones.
   - *Earlier claims.* All 17 of the earlier H6 claims hold at the state they
     were made for, iteration 2 of the spin-orbital ACSE.
4. **The finite-shot CQE (section 4).**
   - *Floors.* Noise floors follow the linearized theory (1/S).
   - *Optimizers.* CG is the most robust under residual noise.
   - *Zero candidate.* It pays only for CG.
   - *Negative candidates.* They do not pay (1.3–1.4× worse at equal cost).
   - *Two levers that do pay.* Fewer candidates (0.63× the shots), and projecting
     the residual estimate onto the symmetry-allowed entries, which costs no
     shots and cuts the floor to 0.14–0.56×.
   - *Cost of the whole run.* The cheapest validated runs reach and stay below
     1.6 mHa with 1e7–3e8 shots in total.

## 1. Optimizers and the spin-orbital metric

`run_cqe(..., optimizer="cg" | "lbfgs", metric="spin_free" | "spin_orbital")`
builds the direction in the local coordinates of the current iterate. The
gradient there is −conj(A), with the inner product Re Σ conj(u)v. CG uses
Polak–Ribière+; L-BFGS keeps memory 10, as in the locally parameterized
quasi-Newton CQE of Smart, Boyn & Mazziotti (JCTC 18, 5286 (2022)).

**The metric.** The two generators differ:

- **M1 harness.** It steps along the spin-free generator Σ g[p,q,r,s] e_{pq,rs},
  with g = conj(A).
- **The CQE of Mazziotti and co-workers.** It steps along the spin-orbital
  generator Σ_{ijkl} conj(²A^{ij}_{kl}) a†_i a†_j a_l a_k (full sum).
- **The relation.** For singlets the spin-orbital generator is the spin-free one
  with amplitude M g:
  - M = (2I + T)/3, with (TX)[p,q,r,s] = X[p,s,r,q];
  - verified on the Fock space to 1e-15;
  - with this scale, our ε is the Euler step of the group's HQCA code.
- **Why it preconditions.** M is symmetric positive definite on the residual's
  subspace: eigenvalue 1 on singlet-coupled pairs, 1/3 on triplet-coupled pairs.
  So `metric="spin_orbital"` preconditions every optimizer:
  - SD takes d = M g, the standard CQE;
  - CG is preconditioned PR+;
  - L-BFGS uses H₀ = γM.

Steps to E − E_FCI below 1.6 mHa / 1e-5 Ha, exact CQE from HF, at most 60 steps
(`benchmarks/run_optimizers.py`; "so" = spin-orbital metric):

| system | SD | SD (so) | CG | CG (so) | L-BFGS | L-BFGS (so) |
|---|---|---|---|---|---|---|
| H4 | 5 / 35 | 3 / 18 | 3 / 9 | 3 / 7 | 3 / 9 | 3 / 7 |
| H6 | 14 / – | 6 / – | 6 / 51 | 4 / 38 | 6 / 47 | 4 / 36 |
| H6 (1.0 Å) | 14 / – | 7 / – | 6 / 54 | 4 / 41 | 6 / 51 | 4 / 38 |
| H6s | 47 / – | 30 / – | 13 / – | 10 / – | 12 / – | 10 / – |
| LiH | 2 / 51 | 2 / 29 | 2 / 10 | 2 / 8 | 2 / 9 | 2 / 8 |
| N2 | 7 / – | 3 / – | 3 / – | 3 / – | 3 / – | 3 / – |
| N2s | – / – | – / – | – / – | – / – | – / – | – / – |
| H8 | 36 / – | 15 / – | 9 / – | 6 / – | 9 / – | 6 / – |

Final E − E_FCI after 60 steps:

| system | SD | SD (so) | CG | CG (so) | L-BFGS | L-BFGS (so) |
|---|---|---|---|---|---|---|
| H6 | 1.5e-4 | 9.1e-5 | 7.0e-6 | 3.3e-6 | 5.9e-6 | 3.0e-6 |
| H6s | 1.4e-3 | 1.0e-3 | 7.6e-5 | 4.4e-5 | 7.3e-5 | 3.9e-5 |
| N2 | 3.3e-4 | 3.4e-4 | 1.8e-4 | 1.8e-4 | 1.8e-4 | 1.8e-4 |
| N2s | 8.3e-3 | 7.9e-3 | 3.3e-3 | 3.5e-3 | 5.5e-3 | 3.7e-3 |
| H8 | 7.2e-4 | 3.7e-4 | 4.8e-5 | 2.6e-5 | 4.7e-5 | 2.5e-5 |

- **The standard CQE beats spin-free SD.** It reaches chemical accuracy 1.6–2.4×
  faster (H6 6 against 14 steps, H8 15 against 36, N2 3 against 7; LiH needs 2
  either way). Its first step alone recovers more than twice as much: one step
  from HF leaves H6 (1.0 Å) 17.0 mHa above FCI, against 38.0 with the spin-free
  generator.
- **Preconditioned CG and L-BFGS beat the standard CQE.** With the
  spin-orbital metric they need 1.5–3× fewer steps than it to chemical accuracy
  on H6, H6s and H8 (H6s: 10 against 30), and 2.6× (H4) and 3.6× (LiH) fewer to
  1e-5 Ha. Without the metric they tie with it on H6 and N2. The metric saves
  CG and L-BFGS 1.2–1.5× in steps (H6: 4 against 6 to 1.6 mHa, 36 against 47
  to 1e-5 Ha) and halves their final error.
- **N2 is plateau-limited.** Every optimizer stops at about 1.8e-4 Ha (section 2).
- **Fixed candidates.** The ordering holds with the hardware-like line search:
  the lowest energy among the 11 candidates 0.00625·2^k, i.e. 11 energy
  evaluations per step. Steps to 1.6 mHa on H8:

  | SD | SD (so) | CG | CG (so) | L-BFGS | L-BFGS (so) |
  |---|---|---|---|---|---|
  | 24 | 9 | 10 | 7 | 9 | 6 |

  All 96 benchmark runs, both line searches and both metrics, decrease the
  energy monotonically.
- **Step sizes.** Exact-search step sizes, 10–90% range per system:

  | optimizer | spin-free | spin-orbital |
  |---|---|---|
  | SD | 0.017–0.12 | 0.035–0.35 |
  | CG | 0.026–0.32 | 0.061–0.71 |
  | L-BFGS | about 0.3–7 | 0.5–9 |

  3–5% of L-BFGS steps exceed 6.4, so a candidate set must span 0.03–7 or
  more.
- **Defaults.** `metric="spin_free"` reproduces M1 bit for bit. Every direction
  keeps pair symmetry, anti-Hermiticity and S² = 0.
- **L-BFGS curvature tolerance.** The default is now 1e-2: pairs with
  ⟨s, y⟩ ≤ 1e-2 ‖s‖‖y‖ are skipped. Exact runs are bit for bit the same as with
  the old 1e-12; under noise the old value let noise-dominated pairs blow up the
  direction (section 4).
- **Negative and zero candidates.** Candidates may be zero (stay) or negative
  (step against d). A zero step never counts as convergence. Section 4 measures
  when they help under noise.

## 2. Why the exact CQE stalls: null and soft modes of the linearized CQE

Near the ground state Ψ₀, write Ψ = c₀Ψ₀ + Σ_k d_kΨ_k over the other singlets.
To first order, the residual is A = Σ_k d_k (E_k − E₀) M_k, where M_k is the
anti-Hermitian part of the transition 2-RDM ⟨Ψ₀|e_{pq,rs}|Ψ_k⟩. A
fixed-step SD step maps y = W^{1/2}d by I − (ε/2)K, with:

- W = diag(E_k − E₀);
- K = W^{1/2} S W^{1/2} = BᵀB and S_kj = ⟨M_k, M_j⟩.

In these coordinates E − E₀ = Σ y². The eigenvalues λ of K are the stiffnesses
of the CQE modes. A null vector of B is a contamination the residual does not
see at first order (`diagnostics/linearized.py`; `benchmarks/run_linearized.py`).
K is block diagonal by spatial symmetry, and an exact CQE from HF stays in the
block that holds the HF determinant's weight (`linearize(..., sector="hf_block")`):

| system | coupled singlets | null modes | smallest non-null λ | λ_max | HF symmetry block | null modes in it | weight outside it, 60 exact steps |
|---|---|---|---|---|---|---|---|
| H4 | 19 | 0 | 5.4e-2 | 64.9 | 11 | 0 | 6e-12 |
| H6 | 174 | 0 | 1.9e-6 | 135 | 91 | 0 | 2e-7 |
| H6 (1.0 Å) | 174 | 0 | 2.7e-6 | 125 | 91 | 0 | 4e-7 |
| H6s | 174 | 0 | 1.1e-5 | 66 | 91 | 0 | 1e-5 |
| LiH | 104 | 0 | 1.4e-6 | 122 | 30 | 0 | 3e-10 |
| N2 | 174 | 5 | 2.8e-6 | 154 | 17 | 3 | 4e-7 |
| N2s | 174 | 5 | 2.5e-7 | 95 | 17 | 3 | 3e-4 |

- **Round-off leaves the symmetry block.** In exact arithmetic the CQE keeps the
  point-group symmetry of HF. In floating point, contamination of other irreps
  is amplified whenever a step exceeds 4/λ for their stiff modes, so the exact
  runs carry 1e-12 to 3e-4 of weight outside the HF block after 60 steps (last
  column; maximum over SD, CG and L-BFGS). An earlier version of this table
  counted the states with weight above 1e-20 along the runs as the "sector
  reached from HF" (149 states for N2); that count measured amplified round-off,
  not the reachable sector. Section 4 shows that under shot noise this symmetry
  breaking becomes a real cost.
- **N2 orbitals.** `systems.n2` now builds symmetry-adapted (D2h) orbitals with
  the degenerate π pairs axis-aligned (`systems.canonical_orbitals`). Before,
  their orientation was set by round-off and changed with the BLAS thread count;
  none of the numbers in this section depend on it, but measurement costs do
  (section 3).

- **N2 has exact null modes.** Along a null mode ‖A‖ grows as t², not t: halving
  t divides it by 4.00, against 2.00 along the stiffest mode (tested). Such
  contamination can leave only through higher-order terms.
- **Where N2's remaining error sits.** After 60 exact L-BFGS steps it is
  1.83e-4 Ha, and 1.82e-4 of it lies in the null modes. That energy is set in
  the first ten steps and then barely moves for a hundred more: 1.82e-4 after
  100 L-BFGS or CG steps. Only higher-order terms lower it after that, slowly and
  in a way that depends on round-off: after 2000 steps 1.67e-4 (L-BFGS) and
  6.8e-5 (CG), against 1.33e-4 for L-BFGS with the earlier, arbitrarily oriented
  π orbitals. Under steepest descent it even grows, from 1.5e-4 to 1.8e-4, while
  the other modes decay. This is the N2 plateau reported by the optimizer
  benchmark.
- **The other systems have soft modes.** With condition numbers λ_max/λ_min of
  10⁶–10⁸, their slow tails sit in the soft modes (λ < 10⁻³ λ_max) after
  60 CG/L-BFGS steps: 81–93% of the remaining error for H6, H6 (1.0 Å), H6s and
  LiH. These systems still converge: H6 reaches 2e-9 and LiH 2e-12 after
  2000 L-BFGS steps.
- **Two predictions for estimated residuals.** Both are tested on the real,
  nonlinear CQE.
  - *Noise floor.* With unbiased Gaussian residual errors, fixed-step SD
    settles at a mean energy floor given by a Lyapunov equation. For isotropic
    noise every mode with λ > 0 holds (εσ²/4)/(1 − ελ/4) (equipartition), and
    null modes are neither excited nor damped. On H4, the measured fast-mode
    energy is 0.97–1.01 of the prediction over three seeds and two step sizes.
  - *Bias offset.* A biased estimator moves the fixed point by y* = −B⁺b, and
    soft modes amplify that offset.

## 3. Measurement cost of the energy and the residual

`benchmarks/run_measurement.py` computes exact per-shot variances along exact CQE
trajectories, from the qubit layer in `measure/`:

- **Qubit layer.** Jordan–Wigner in two orders, blocked (qubit p + σN) and
  interleaved (2p + σ, OpenFermion's) (`measure/pauli.py`).
- **Element sets** (`measure/targets.py`). The residual is measured element by
  element over 840 spin-orbital or 315 spin-free independent elements at N = 6.
- **Residual routes** (`measure/routes.py`).
  - *Aux route.* Â_e = −⟨Q_e⟩/δ on Λ = exp(−iδH)Ψ, with Q_e = (Γ_e − Γ_e†)/(2i).
    Its bias is (δ²/6)⟨ad³_H Γ_e⟩ + O(δ⁴). Λ is exact, a DF-Trotter product, or
    a sum over the L + 1 DF fragments (L + 1 states).
  - *Direct route.* X_e = [Γ_e − Γ_e†, H]/2, unbiased.
- **Schemes** (`measure/schemes.py`).
  - QWC: sorted insertion with the optimal shot allocation.
  - DF basis rotation: L + 1 settings, leaves from pivoted Cholesky or from the
    eigendecomposition of the ERI supermatrix (eigh), optionally shifted to
    E_pq − δ_pq.
  - Pauli shadows and matchgate (fermionic Gaussian) shadows, from exact pair
    sums, or Rao–Blackwellized where those are too large.
- **Shots for a relative Frobenius error r:** S = Σ_e m_e v_e / (r²‖A‖² − ‖b‖²),
  with element multiplicities m_e and route bias b.

**Energy.** Per-shot variance (Ha²) at FCI. The last column is the shots for a
±1.6 mHa estimate with the best scheme.

| system | QWC blocked / interleaved | DF eigh | DF shifted, Cholesky / eigh | matchgate | Pauli blocked | shots, best |
|---|---|---|---|---|---|---|
| H4 | 2.80 / 4.02 | 1.61 | 0.614 / **0.597** | 33.1 | 31.8 | 2.3e5 |
| H6 (1.0 Å) | 16.2 / 28.1 | 3.06 | 1.23 / **1.19** | 122 | 1015 | 4.7e5 |
| H6 (1.5 Å) | 15.2 / 25.1 | 1.29 | 0.794 / **0.731** | 72.5 | 1302 | 2.9e5 |
| H6 | 16.6 / 28.3 | 3.30 | 1.27 / **1.23** | 131 | 996 | 4.8e5 |
| LiH | 1.05 / 0.852 | 1.40 | **0.606** / 0.905 | 155 | 269 | 2.4e5 |
| N2 | 4.13 / 3.00 | 0.928 | **0.844** / 0.906 | 224 | 462 | 3.3e5 |
| H8 | 61.6 / 115 | 5.33 | 2.30 / **2.13** | 337 | 4.5e4 | 8.3e5 |

- **Shifted DF grouping is the cheapest energy measurement** on every system.
- **QWC needs more.** Its best order costs 1.4× (LiH) to 29× (H8) more. The
  Jordan–Wigner order alone changes QWC by up to 1.9×.
- **Shadows cost far more.** Matchgate shadows need 55–270× more; Pauli shadows
  53× (H4) to 2×10⁴× (H8).
- **The DF split is not unique.** Cholesky and eigh leaves differ by up to 1.8×.
  The shift lowers the variance 1.8–2.7× on the hydrogen chains.

**Residual.** Shots for 30% relative error over the full tensor. The aux route
uses the best δ of {0.01, 0.03, 0.1, 0.3, 0.6} for the bias-aware count; values
are spin-orbital elements unless marked sf.

| system | M1 SD iterate (E − E_FCI, mHa) | direct, matchgate, sf / spin-orbital | aux exact, matchgate [δ] | aux exact, QWC interleaved | aux DF-Trotter, matchgate | direct, Pauli shadows |
|---|---|---|---|---|---|---|
| H4 | 2 (9.19) | 2.9e5 / 5.2e5 | 1.3e6 [0.6] | 2.8e6 | 3.5e7 | 4.8e6 |
| H4 | 7 (0.373) | 5.7e6 / 1.3e7 | 3.4e7 [0.6] | 7.2e7 | 1.4e9 | 1.2e8 |
| H6 (1.0 Å) | 2 (19.6) | 2.9e6 / 6.1e6 | 9.4e6 [0.6] | 3.9e7 | 8.7e7 | 3.3e8 |
| H6 (1.0 Å) | 7 (3.81) | 1.7e7 / 3.6e7 | 5.0e7 [0.6] | 2.1e8 | 1.7e9 | – |
| LiH | 2 (1.32) | 8.6e7 / 1.8e8 | 6.6e8 [0.3] | 2.6e9 | 4.8e10 | – |
| LiH | 7 (0.622) | 2.3e8 / 5.0e8 | 8.1e8 [0.6] | 3.2e9 | 1.4e11 | – |
| N2 | 2 (18.4) | 1.5e6 / 3.3e6 | 1.4e7 [0.6] | 5.8e7 | 2.6e8 | – |
| N2 | 7 (1.49) | 2.6e7 / 6.4e7 | 1.8e8 [0.6] | 7.6e8 | 6.6e9 | – |

- **Cheapest residual: direct route with matchgate shadows** (unbiased), at
  every state of every system.
- **Spin-free elements need half the shots.** They need 0.41–0.56× those of the
  spin-orbital set for the same relative error. They are also the elements the
  spin-free generator uses.
- **Other routes cost more:**
  - the exact aux route, 1.4–5.6× more, with its bias capping δ;
  - QWC tomography of the aux targets, another 2–4× on top;
  - Pauli shadows on the direct route, 9–53×.
- **DF aux routes are dominated.** DF-Trotter and DF-fragment states add biases
  that do not vanish with A: 5.0% and 5.3% at δ = 0.1 on the H6 claims state.
  They cost 14–270× (DF-Trotter) and 10³–10⁵× (DF fragments) more, and become
  unreachable as ‖A‖ shrinks.
- **The residual cost grows as the CQE converges.** Shots scale as 1/‖A‖², so
  the same relative error costs 3–25× more at iterate 7 than at iterate 2. LiH
  is the costly case: small ‖A‖ and twice the per-element variance of H6.
  Section 4 shows that a CQE iteration does not need 30% accuracy.
- **Only symmetry-allowed elements need measuring.** The point-group-forbidden
  elements vanish exactly, and this benchmark still measures them. The M3
  design computed the variance of the allowed elements alone: 0.51 (H4), 0.49
  (H6) and 0.126 (N2) of the total at SD iterate 2. M3 uses the allowed
  elements only.

**The earlier H6 claims.** All 17 claims hold at the state they were made for,
the claims' "CQE iteration 2, about 6.6 mHa above FCI". That state is iteration
2 of steepest descent with the spin-orbital ACSE generator (section 1; 6.558 mHa,
‖A‖₈₄₀ = 0.0760); the shot counts imply ‖A‖₈₄₀ = 0.0755–0.0761. It is not an
iterate of our spin-free SD, where iteration 2 is 19.6 mHa above FCI and the
shot counts are 3× smaller (shots ∝ 1/‖A‖²).

| claim (H6, STO-3G, 1.0 Å unless noted) | claimed | ours | ratio |
|---|---|---|---|
| energy variance, QWC (interleaved order) | 28.0 | 27.6 | 0.99 |
| energy variance, DF (eigh leaves, unshifted) | 3.2 | 3.16 | 0.99 |
| energy variance, matchgate shadows | 122 | 122 | 1.00 |
| the same at 1.5 Å: QWC / DF / matchgate | 24.7 / 1.44 / 72.5 | 24.7 / 1.44 / 72.5 | 1.00 / 1.00 / 1.00 |
| aux route, Var(Im ²D) per element: matchgate / QWC with 854 settings | 5.4 / 36 | 5.32 / 35.2 | 0.99 / 0.98 |
| aux route, relative bias at δ = 0.1 / 0.3 / 0.6 | 0.2–1% / 2–8% / 6–29% | 0.95% / 8.2% / 28.5% | inside / 1.03× the top / inside |
| direct route, per-element variance min / mean / max | 7 / 12–16 / 41 | 4.96 / 11.5 / 40.6 | partly |
| shots for 30%: aux + QWC, δ = 0.1 | 5.9e9 | 5.68e9 | 0.96 |
| shots for 30%: aux + shadows, δ = 0.1 / 0.3 | 8.7e8 / 9.7e7 | 8.60e8 / 9.56e7 | 0.99 / 0.99 |
| shots for 30%: direct shadows | 2.6e7 | 1.85e7 | 0.71 |
| shots per line-search energy, DF, ±1.6 mHa | 1.2e6 | 1.23e6 | 1.03 |

- **Direct-route statistic.** The claimed 7 / 12–16 / 41 are the statistics of
  the 98 elements with |A_e| > 10⁻³ (98.9% of ‖A‖²): 7.01 / 16.0 / 40.6. The
  claimed shot count extrapolates that mean to all 840 elements (2.59e7).
- **Which variant each claim used.**
  - QWC: the interleaved order. Sorted insertion leaves the order inside exact
    ties open; random tie orders give 27.9–30.3 and, for the aux targets,
    839–864 groups, which contains the claimed 854.
  - DF: unshifted eigh leaves.
  - The relative bias depends on the trajectory: 0.25% at δ = 0.1 on the
    spin-free SD iterate with the same energy (5.86 mHa).

**Reproducibility.**
- **Orbital orientation.** The N2 numbers above use the symmetry-adapted
  orbitals of section 2. Before, QWC and Pauli-shadow variances changed by
  35–57% between runs with the BLAS thread count; now the tables are identical
  with 1 and 2 threads.
- **Cholesky pivots.** Pivoted Cholesky breaks exact symmetry ties by index
  (`factor/cholesky.py`). Before, the order of norm-tied leaves, and with it
  the DF-Trotter product of LiH and N2, followed round-off.

## 4. The finite-shot CQE

`benchmarks/run_sampled_cqe.py` runs the CQE with estimated residuals and
energies (noise models in `sampling.py`). Systems: H4, H6 (1.0 Å), LiH and N2.
Unless noted there are 5 seeds per setting, with medians and 16–84% bands; the
run takes about 65 minutes on one thread.

- **Residual noise.** Gaussian, with the exact per-shot covariance of matchgate
  snapshots over the 840 spin-orbital elements of the direct route
  ("correlated"), or with independent elements.
- **Energy noise.** Gaussian with the per-shot variance of shifted DF grouping,
  evaluated at each state.
- **Check against real snapshots (E4, H4).** The floor with real matchgate
  snapshots is 0.98 [0.97, 1.13] of the correlated model and 1.29 of the
  independent one.

**Noise floors (E1).** Fixed-step SD from FCI, compared with the linearized CQE
of section 2.

- **The floor scales as 1/S** and matches the Lyapunov prediction. The fast-mode
  ratio is 0.99–1.03 with correlated noise and 0.97–1.01 with independent noise.
  Each run is shadowed by a linear twin driven by the same errors; the real CQE
  stays within 1.000–1.002 of it for S ≥ 1e6.
- **Soft modes do not equilibrate (N = 6).** The stationary prediction overstates
  the floor by 18–27%; the transient-aware prediction matches.
- **Correlations between elements raise the floor**, by 1.03× (LiH) to 1.32×
  (H4) over independent noise.
- **Residual shots per iteration for a 1.6 mHa floor:** H4 9.0e3, H6 1.07e5,
  LiH 1.72e5, N2 7.0e4. A 0.1 mHa floor costs 16× more.
- **The CQE does not need 30% residual accuracy.** At S_res = 1e6 the H6
  residual is off by 74–137% relative, yet the runs reach chemical accuracy.
- **The aux route has a higher floor.** With exact exp(−iδH) at δ = 0.6 its floor
  is 1.3–1.8× that of the direct route at equal S. Predicting it needs the sinc
  damping of each mode, sinc(δ w_k). LiH diverges at δ = 0.6, where some
  sinc(δ w_k) < 0, so δ must also satisfy δ w_max < π.
- **N2's null modes are excited only at second order** (∝ 1/S²).

**Optimizers under residual noise (E2, exact line search on the noisy direction).**

- **CG has the lowest median final error** in 15 of 16 (system, S_res) cells
  (L-BFGS in the other). In 12 of the 16 the best optimizer beats both others in
  every seed. Example, H6 at S_res = 1e6:
  CG 3.2e-4, L-BFGS 4.4e-4, SD 5.5e-4 Ha.
- **Noise costs extra iterations even when it is small.** Write j\* for the
  first iteration after which at most 5% of the seeds are above 1.6 mHa in
  every 20-iteration window. At S_res = 1e7–1e8, j\* exceeds the exact
  iteration count: H6 SD 19 / 18 against 14, N2 SD 16 against 7. The cause is
  symmetry breaking (see "Symmetry breaking and a free projection" below).

**Your question: zero and negative step candidates (E3).**

*Setup.*
- Noisy residuals and noisy shifted-DF energies; fixed candidates; 40
  iterations from HF.
- Six (S_res, S_E) points from (1e5, 1e6) to (1e7, 1e6).
- Candidate sets:
  - pos11: 0.00625·2^k, k = 0…10;
  - pos11+0: pos11 plus zero;
  - pos5+0: zero and five positive steps a factor 4 apart;
  - eq11: zero and ±{0.0125, 0.05, 0.2, 0.8, 3.2}, the same energy cost as
    pos11;
  - dbl23: zero and ±pos11, double the cost.
- One statistic throughout: the geometric mean of paired ratios of the late
  error (mean over iterations 21–40), first set over second, with a 95%
  bootstrap interval over the 20 (system, seed) clusters.

| comparison | all optimizers | SD | CG | L-BFGS |
|---|---|---|---|---|
| pos11+0 vs pos11 | 0.97 [0.94, 1.01] | 1.05 [1.00, 1.11] | 0.82 [0.76, 0.90] | 1.05 [1.00, 1.10] |
| pos11+0 vs pos11, S_res = 1e5 | 0.91 [0.88, 0.95] | 1.10 [1.05, 1.14] | 0.68 [0.62, 0.76] | 1.02 [0.95, 1.07] |
| pos5+0 vs pos11+0 | 1.07 [1.03, 1.12] | 1.17 [1.08, 1.28] | 1.01 [0.94, 1.09] | 1.05 [0.94, 1.17] |
| eq11 vs pos5+0 (same positive magnitudes) | 1.34 [1.22, 1.51] | 1.48 [1.25, 1.75] | 1.24 [1.12, 1.42] | 1.32 [1.20, 1.45] |
| eq11 vs pos11 (same cost) | 1.40 [1.24, 1.59] | 1.82 [1.62, 2.08] | 1.03 [0.88, 1.23] | 1.44 [1.27, 1.63] |
| dbl23 vs pos11+0, S_res = 1e5 | 0.99 [0.94, 1.04] | 1.12 [1.01, 1.28] | 0.77 [0.70, 0.84] | 1.13 [0.99, 1.26] |
| dbl23 vs pos11+0, S_res ≥ 1e6 | 1.47 [1.32, 1.64] | 1.75 [1.56, 1.97] | 1.16 [1.02, 1.30] | 1.57 [1.36, 1.84] |

- **Zero pays only for CG.** 11–13% of CG's noisy directions point uphill at
  S_res = 1e5–1e6, and zero refuses them: it helped in all 270 such choices,
  median −0.52 mHa.
  - CG's late error drops to 0.68× at S_res = 1e5 and to 0.90× [0.84, 0.98] at
    S_res ≥ 1e6.
  - For SD, whose directions are uphill in at most 0.5% of iterations, zero is
    chosen through noise alone and costs 5–10%.
  - For L-BFGS it is neutral, now that the curvature tolerance is 1e-2. With the
    old 1e-12 it seemed to help (0.62×), because it refused steps that raised
    the energy by up to 3.4 Ha; that only hid the bug.
- **Negative steps do not pay.**
  - At the same positive magnitudes they make the late error 1.34× worse, and
    the shots to converge 1.82× more.
  - At the same energy cost the late error is 1.40× worse.
  - They help only CG with very noisy residuals and the doubled set (0.77× at
    S_res = 1e5), at twice the energy cost; at that S_res no N = 6 cell reaches
    the target with either set.
- **Why negatives lose.** A negative step can only help on an uphill direction,
  but 72–95% of negative choices happen on downhill directions, and 86–92% of
  those hurt.

  | optimizer | negative chosen | negative is the exact best |
  |---|---|---|
  | SD | 13–16% | 0.0–0.1% |
  | CG | 13–18% | 1.7–4.8% |
  | L-BFGS | 30–33% | 2.5–3.0% |

  Near the floor all candidates have nearly the same exact energy, so the
  minimum of 11–23 noisy estimates is decided by noise.
- **What pays instead: fewer candidates.** pos5+0 uses 6 estimates per
  iteration instead of 12. Its late error is 1.07× worse, but it needs 0.63× the
  shots to converge, and it appears in the cheapest validated configuration of
  three of the four systems.

**Cheapest validated configurations.** These are the cells that reach and stay
below 1.6 mHa with the fewest total shots (residual plus energies).

- **Selection.** Screened at 40 iterations, then rerun with 10 fresh seeds for
  120 iterations. P_late is the share of seed-iterations above 1.6 mHa in
  iterations 61–120.
- **Earlier screening winners did not survive.** They depended on the
  40-iteration horizon, and bootstrapping the selection shows a winner's curse:
  the screening winner is picked in only 18–86% of resamples, and LiH's failed
  validation.

| system | optimizer, set, (S_res, S_E) | total shots [95% over seeds] | j\* | P_late | residual share | with the projection |
|---|---|---|---|---|---|---|
| H4 | SD, pos5+0, (1e6, 1e5) | 1.12e7 [1.12e7, 1.25e8] | 7 | 1.3% | 63% | 1.12e7 |
| H6 | L-BFGS, pos5+0, (1e7, 1e6) | 2.72e8 [2.08e8, 2.72e8] | 17 | 0.0% | 63% | 2.24e8 |
| LiH | L-BFGS, pos11, (1e6, 3e5) | 6.02e7 [4.73e7, 9.89e7] | 14 | 0.2% | 23% | 6.02e7 |
| N2 | L-BFGS, pos5+0, (1e6, 3e5) | 3.36e7 [2.24e7, 3.36e7] | 12 | 0.5% | 36% | 2.80e7 |

- **H4's winner is fragile.** L-BFGS with pos11 at the same point costs 1.26e7
  shots with P_late 0%.
- **These are upper bounds.** The grid is coarse (S_res = 1e5, 1e6, 1e7).
  Certifying the final energy to ±1.6 mHa would cost a further 2.3e5–4.7e5 shots
  (shifted DF, section 3).

**Symmetry breaking and a free projection (E5, E6).**

- **Noise breaks the point-group symmetry.** Estimated residuals do not respect
  it, so noisy runs populate states outside the HF symmetry block of section 2.
  For N2 after 150 iterations: about 1e-5 Ha at S_res = 1e6 (SD 1.3e-5, L-BFGS
  5.3e-5).
- **Noise erases L-BFGS's soft-mode advantage on N2.** Its soft-mode energy is
  1.9e-4 Ha at S_res = 1e6 against 5e-7 in the exact run. The null-mode plateau
  (1.5e-4 Ha) is unchanged.
- **The projection.** The exact residual of a totally symmetric state vanishes
  on the symmetry-forbidden entries. Projecting the estimate there,
  A_est → mask·A_est with the mask from the Z2 gradings of the integrals, is
  unbiased and costs no shots.
- **It lowers the noise floor:**

  | system | projected / plain floor (theory) | simulated, S = 1e6 |
  |---|---|---|
  | H4 | 0.56 | 0.55 |
  | H6 | 0.51 | 0.51 |
  | LiH | 0.41 | 0.41 |
  | N2 | 0.145 | 0.144 |

  The D2h and C2v gradings keep Δ-type components of LiH and N2; a full D∞h or
  C∞v projection would go lower (0.34 and 0.08 for the HF block alone).
- **It recovers most of the noise-induced iterations.** j\* at
  S_res = 1e6 / 1e7 / 1e8, against the exact count:

  | system | optimizer | exact | plain | projected |
  |---|---|---|---|---|
  | N2 | SD | 7 | 19 / 16 / 16 | 8 / 7 / 7 |
  | N2 | CG | 3 | 5 / 4 / 4 | 4 / 3 / 3 |
  | N2 | L-BFGS | 3 | 6 / 4 / 4 | 4 / 3 / 3 |
  | H6 | CG | 6 | 10 / 7 / 7 | 8 / 7 / 6 |
  | H6 | L-BFGS | 6 | 11 / 7 / 7 | 9 / 7 / 6 |
  | LiH | CG | 2 | 6 / 3 / 2 | 4 / 3 / 2 |

  M3 measures only the allowed elements in the first place (section 3).

**Notes.**

- **L-BFGS curvature tolerance.** With the old 1e-12, 6 of 15 LiH runs (pos11)
  took steps that raised the energy by more than 10 mHa, up to 3.4 Ha. With the
  new default 1e-2, none of the 1980 screening runs of any optimizer did; the
  largest rise was 7 mHa.
- **An unused energy estimate.** `run_cqe` requests the current-state energy
  estimate every iteration, but nothing uses it when the candidates exclude zero
  and e_tol = 0. The benchmark therefore does not charge it, and counts 11
  estimates per iteration for pos11, 12 for pos11+0 and 6 for pos5+0.
  Skipping the call in the library would change the published random streams,
  so it is left for a later milestone.
- **Limitations.**
  - Residual variances are frozen at their FCI values.
  - The energy DF leaves carry 3–7% more variance than the cheapest split
    (section 3).
  - j\* still worsens with the horizon for cells with 2–6% late exceedance.
