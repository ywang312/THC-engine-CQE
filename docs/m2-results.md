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

## 1. Optimizers: conjugate gradient and L-BFGS

`run_cqe(..., optimizer="cg" | "lbfgs")` builds the direction in the local
coordinates of the current iterate. The gradient there is −conj(A), with the
inner product Re Σ conj(u)v. CG uses Polak–Ribière+; L-BFGS keeps memory 10, as
in the locally parameterized quasi-Newton CQE of Smart, Boyn & Mazziotti
(JCTC 18, 5286 (2022)). Steps to reach E − E_FCI below a threshold, exact CQE
from HF, at most 60 steps (`benchmarks/run_optimizers.py`):

| system | SD < 1.6 mHa | CG | L-BFGS | SD < 1e-5 | CG | L-BFGS | final error after 60: SD / CG / L-BFGS |
|---|---|---|---|---|---|---|---|
| H4 | 5 | 3 | 3 | 35 | 9 | 9 | 4.6e-6 / 1e-11 / 2e-13 |
| H6 | 14 | 6 | 6 | – | 51 | 47 | 1.5e-4 / 7.1e-6 / 5.9e-6 |
| H6s | 47 | 13 | 12 | – | – | – | 1.4e-3 / 7.6e-5 / 7.2e-5 |
| LiH | 2 | 2 | 2 | 51 | 10 | 9 | 5.8e-6 / 1.1e-7 / 1.4e-7 |
| N2 | 7 | 3 | 3 | – | – | – | 3.3e-4 / 1.8e-4 / 1.8e-4 |
| N2s | – | – | – | – | – | – | 8.5e-3 / 3.3e-3 / 5.2e-3 |
| H8 | 36 | 9 | 9 | – | – | – | 7.2e-4 / 4.8e-5 / 4.6e-5 |

- **Speedup.** CG and L-BFGS reach chemical accuracy in up to 4× fewer steps
  than steepest descent: 2.3× on H6 and N2, 4× on H6s and H8, and no gain on
  LiH, which needs only 2. They reach 1e-5 Ha in 4–6× fewer steps where SD gets
  there at all.
- **Defaults.** Default runs reproduce M1 bit for bit, and every direction keeps
  pair symmetry, anti-Hermiticity and S² = 0.
- **Fixed candidates.** The same ordering holds with the hardware-like line
  search, the lowest energy among the 11 fixed candidates 0.00625·2^k. To
  1.6 mHa on H8: 24 / 10 / 9 steps, or 264 / 110 / 99 energy evaluations.
- **Step sizes.** L-BFGS directions are scaled like quasi-Newton steps. Their
  exact-search step sizes are 0.27–6.9 (10–90% range), against 0.017–0.12 for
  SD, so a candidate set must span both.
- **Negative and zero candidates.** Since this milestone, candidates may be zero
  (stay) or negative (step against d). A zero step never counts as convergence.
  Section 4 measures when these help under noise.

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

(workflow C: pending)
