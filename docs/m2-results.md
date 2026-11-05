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

(workflow B: in progress)

## 4. The finite-shot CQE

(workflow C: pending)
