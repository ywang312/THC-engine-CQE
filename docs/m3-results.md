# Milestone 3 results

M3 asks what the CQE costs on a fault-tolerant machine, and what it is worth as a state
preparer for QPE. `thc_engine_cqe.ft` is a resource-estimation layer built from validated cost
models, not a compiler: the unit is the Toffoli (one CCZ state) and Cliffords are free. The
cost model as implemented is in [m3-cost-model.md](m3-cost-model.md); the formula labels
F1–F19 and F2a, the parameters, "Appendix A" (its reference numbers) and the tags ([OF],
[derived], [cited], [estimate], [assumption], [design], [build check]) used below refer to
it. Every model is also stated in its module docstring (`ft/__init__.py` lists the modules),
and section 11 lists what changed from the design specification. The systems are the
STO-3G builds of [m2-results.md](m2-results.md), on symmetry-adapted orbitals (D2h; C2v for
LiH; `ft.symmetry`):

- core: `h4` (H4 chain, 1.8 bohr), `h6a` (H6, 1.0 Å), `n2` (N2 CAS(6,6), 1.098 Å), `lih`
  (LiH, 1.595 Å);
- overlap-limited: `h6s` (H6, 3.0 bohr), `n2s` (N2 CAS(6,6), 2.0 Å);
- scaling: `h8` (H8, 1.8 bohr), costed without the residual: only its line search is costed,
  so its loop costs are lower bounds (marked ≥).

The states are exact CQE trajectories from HF with the residual projected onto the
symmetry-allowed entries, for three optimizer families:

- `lbfgs`: spin-free L-BFGS (memory 10, 20 steps), the primary family (Appendix A;
  `tests/test_ft_pipeline.py::test_h4_reproduces_the_golden_record`);
- `lbfgs_so`: the same with `metric="spin_orbital"`, i.e. along the standard (Mazziotti)
  spin-orbital CQE generator; its measured residual is the same spin-free A, with the same
  elements and per-shot variances;
- `sd`: spin-free steepest descent (30 steps), core systems only.

Every family has two records: step 0 by the exact line search (the measured and free loops),
or by the classical second-order step ε₂ of F16 followed by the exact continuation
(`run_cqe(..., first_eps=ε₂)`; the classical-first-step loop). C_QPE is one phase estimation
to 1 mHa with the cheapest admissible block encoding of the system (T1). The residual is
costed at r = 0.3 relative error over the symmetry-allowed spin-free elements, the line search
as 11 candidate energies at ±1.6 mHa (M2's hardware rule). Every table except the inputs of T3
(section 3) and those of sections 7 and 11 is copied from `results/ft_summary.md`, written (with
`results/ft_<system>.json`) by

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python benchmarks/run_ft_pipeline.py --threads 1 --out results/

whose defaults are `--systems h4,h6a,n2,lih,h6s,n2s,h8 --optimizers lbfgs,lbfgs_so,sd`. On one
thread it takes 280–348 s, with a peak resident memory of about 550 MB (section 9). One system
alone, e.g. `--systems h4 --out <dir>`, takes 13 s and 516 MB, and its table rows equal those
below apart from the runtime row [build check]; `--h8-residual` also costs the residual of h8. The tests that
pin the model are `tests/test_ft_*.py` (`python -m pytest tests/test_ft_*.py -m "not slow"`
skips the slow ones).

**Notation** (defined in m3-cost-model.md). C(Ψ_k) and Q(Ψ_k) are the Toffolis and logical
qubits of the circuit that prepares the iterate Ψ_k from HF (F7). Iteration k costs
C_iter(k) = T_res(k) + T_LS(k): the residual at Ψ_k by its cheapest route plus the line search
over the candidates Ψ_{k+1}(ε_c) (F14); C_loop(k) = Σ_{j<k} C_iter(j). k* is the first iterate
within 1.6 mHa of E₀ and k_v the first within 1 mHa; p_k = |⟨ψ₀|Ψ_k⟩|² is the ground-state
overlap, p₀ = p_HF. Handing Ψ_k to QPE (F15) costs T_E(k) = C_loop(k) + (C(Ψ_k) + C_QPE)/p_k
under rule E (QPE runs on Ψ_k until one returns an outcome below E_k + 2ΔE; expected cost) and
T_δ(k) = C_loop(k) + R_δ·(C(Ψ_k) + C_QPE) under rule δ (R_δ runs, the lowest outcome kept,
failure probability 0.01). The certified rules rest on a certificate: rule E needs
E₁ − E_k > 3ΔE, so that no excited-state outcome is accepted, and rule δ takes R_δ from the
Eckart–Temple lower bound p_lb ≤ p_k; the oracle rules drop it (rule E over every k, rule δ with
the exact p_k). These are the four rules; k_opt is the optimal hand-off under one of them.
p₀*(k) = C_QPE/T_E(k) is the HF overlap below which handing off Ψ_k beats HF.
The loops differ in what the iterations cost: "measured" costs every iteration;
"classical_first_step" takes step 0 from the classical ε₂ of F16 (C_iter(0) = 0) and measures
the rest of its record; "free" sets C_loop ≡ 0, the bound for generators that cost nothing. RB:
Rao–Blackwellized matchgate variances (F10).

**In short.** At equilibrium one measured CQE iteration costs 154–1366 C_QPE already at HF,
where the line search is all of it, while a better initial state can save at most
C_QPE(1/p_HF − 1) = 0.026–0.128 C_QPE: no measured iteration pays as state preparation for QPE.
The classical first step is free and saves 0.020–0.095 C_QPE. The CQE alone, converged to 1 mHa
with its energy measured to 0.3 mHa, costs 2.55e3–7.3e5 times the HF hand-off. On the stretched
h6s and n2s, where the HF hand-off is not certified, the classical first step gives a certified
hand-off at 1.09–2.5 C_QPE, except on spin-free n2s (1073 C_QPE).

**Main results.**

1. **QPE (section 1).** The cheapest admissible block encoding is SF on h4, n2, lih and n2s
   and THC on h6a, h6s and h8: C_QPE = 4.0e6 (h4) to 3.1e7 (h8) Toffolis, one run of
   1.9–15 min on 0.82–1.08 million physical qubits (budget 1e-2). THC beats DF on h6a, h6s and
   h8 (λ·C_W 26–37% lower) and loses on h4 (by 1% in λ·C_W, at M = 10) and on n2, lih and n2s,
   where its cheapest admissible fit has M = 21 and λ = 20–69 Ha.
2. **State preparation (section 2).** With spin-free L-BFGS, the first iterate within
   1.6 mHa costs C(Ψ_k*) = 2506 (h4), 43,427 (h6a), 2576 (n2), 4648 (lih), 94,623 (h6s) and
   245,530 (h8) Toffolis on 70–90 logical qubits, 0.072–8.3 s per shot. DF-leaf circuits cost
   4.5–27× more.
3. **One iteration (section 3).** A measured iteration costs 154–1366 C_QPE at HF and grows to
   3.0e3–1.6e5 C_QPE by k* + 4. Before k* the line search costs more than the residual at
   equilibrium (residual/line search ≤ 0.57); amplitude-estimated line-search energies would
   cost 0.065–0.31 of it. Matchgate shadows are the cheapest residual route up to k = 1–3,
   per-element amplitude estimation after (mostly X-LCU on h4, n2, lih and n2s, the aux route
   on h6a and h6s).
4. **Totals (section 4).** Reaching chemical accuracy with spin-free L-BFGS and the classical
   first step costs C_loop(k*) = 808 (lih) to 6.3e4 (h6a) C_QPE at equilibrium, 3.1 (h4) to
   340 days (h6a) on one machine, or 1 to 51 machines for 7 days; 4.1e5 C_QPE on h6s, which no
   number of machines finishes in 7 days, and ≥ 3.2e5 on h8.
5. **Hand-off at equilibrium (section 5).** A measured iteration never pays: the optimal
   hand-off is HF (k_opt = 0) under all four rules, and one measured iteration would pay only
   below an HF overlap of p₀*(1) = 7.3e-4 to 6.5e-3. The classical first step pays on every
   equilibrium system, T_E(1) = 1.006–1.057 C_QPE against C_QPE/p_HF = 1.026–1.128, a saving
   of 0.020–0.095 C_QPE; free generators would save at most 0.12 C_QPE.
6. **Stretched systems (section 5).** Rule E is not certified at HF on h6s and n2s. On h6s the
   classical step gives a certified hand-off at T_E(1) = 1.13 (spin-free) and 1.09
   (spin-orbital) C_QPE, against 517–554 C_QPE for the cheapest certified measured hand-off.
   On n2s rule E certifies Ψ_k only for E_k − E₀ < E₁ − E₀ − 3ΔE = 0.1887 Ha. The measured
   loop's Ψ₂ (0.1834 Ha) passes, at 571 C_QPE; after the spin-free classical step Ψ₂ misses by
   1.1 mHa (0.1898 Ha), so its first certified hand-off is Ψ₃ at 1073 C_QPE. The spin-orbital
   classical step certifies Ψ₁ (0.1735 Ha) at 2.52 C_QPE.
7. **Spin-orbital L-BFGS (section 7).** It needs fewer steps on h6a (4 against 6), h6s (10
   against 12) and h8 (6 against 9), and its C_loop(k*) with the classical first step is 0.40
   (h6a) to 0.99 (h4, n2) of the spin-free one; its classical first step saves more
   (0.020–0.095 against 0.020–0.071 C_QPE).
8. **CQE-only energy (section 6).** Converging to 1 mHa and measuring the energy to 0.3 mHa
   costs 2.55e3–7.3e5 times the HF hand-off to QPE, 14 days to 13 years on one machine (at least
   13.5 to 24 years on h8, line search only) against 2–17 min.
9. **Validation (section 8).** The run reproduces all 97 Appendix A quantities that the
   benchmark compares (the RB variances within 0.4σ, same seed). The benchmark's own table
   compares with the values the design specification printed and reports one mismatch: N2's
   V_m on the allowed elements, an RB estimate there (3703) and the exact pair sum here
   (3753.0 Ha², the value Appendix A lists). The compiled circuits stay within 3.1e-3 of the
   exact states to k* + 2 at equilibrium, but drift to 4.5e-2 (h6s) and 9.5e-2 (n2s), with ⟨S²⟩
   up to 1.5e-3 and 9.7e-3.

**Reading the tables.** In T2 row k is the state Ψ_k (E_k, p_k, ‖A_k‖, C(Ψ_k), Q(Ψ_k)) and the
step from Ψ_k to Ψ_{k+1}; in T3 row k is iteration k, the residual measured on Ψ_k and the line
search over the candidates Ψ_{k+1}(ε_c); in T6 row k is the hand-off of Ψ_k after C_loop(k).
Circuit costs (C(Ψ_k), C_step, C_W) and counts such as T1's iterations are exact integers. T1
gives λ to 6 significant digits, C_QPE to 3, and ΔE_fact and the infidelity to 2; other Toffoli
and shot counts, the durations of T4 and T8 and ⟨S²⟩ have 2 significant digits, probabilities
and overlaps 3–5 decimals, and most remaining reals 3–4 significant digits. Significant-digit
formats drop trailing zeros (4e6 is 4.0e6); some derived numbers need the unrounded values of
`results/ft_<system>.json`. Columns marked "/ C_QPE" are in units of the system's selected C_QPE
(T1). Marks: `3*` is k*; `>=` a lower bound; `-` not reached or undefined; `[0.6]` after an aux
route the optimal δ (1/Ha); `[2]` after the line search the flagged candidates. T3's "line search
S" counts the shots of all 11 candidates, 11·S_E (F13), and T8's "shots" are those of the loop to
k_v plus the final shots.

## 1. QPE (T1)

lambda C_W (Ha x Toffolis) sets C_QPE = C_W ceil(pi lambda / (2 dE)); '(selected)' is the cheapest admissible encoding overall, the QPE of the hand-off and the walk of the residual routes. Physical: OpenFermion's surface-code model (p = 1e-3, 4 factories), one QPE run.

| system | kind | selection | lambda (Ha) | dE_fact (mHa) | infidelity | C_W | iterations | C_QPE | Q | QR clamp | lambda C_W | budget 0.01: qubits / d / min | budget 0.1: qubits / d / min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| h4 | df | thr 3e-3, L 7, Xi 27 | 4.93464 | 0.27 | 6.7e-7 | 868 | 7,752 | 6.73e6 | 188 | yes | 4283 | 877,824 / 23 / 3.22 | 737,616 / 21 / 2.94 |
| h4 | thc | M 10, pen. 0 | 5.86438 | 3.3e-11 | 0 | 739 | 9,212 | 6.81e6 | 180 | no | 4334 | 864,000 / 23 / 3.26 | 726,000 / 21 / 2.98 |
| h4 | sf (selected) | L 7 | 6.58038 | 0.074 | 2.9e-9 | 384 | 10,337 | 3.97e6 | 156 | yes | 2527 | 822,528 / 23 / 1.9 | 691,152 / 21 / 1.74 |
| h6a | df | thr 3e-3, L 11, Xi 63 | 9.7256 | 0.2 | 1.5e-7 | 1309 | 15,277 | 2e7 | 233 | yes | 1.273e4 | 1,026,160 / 25 / 9.58 | 956,160 / 23 / 9.58 |
| h6a | thc (selected) | M 15, pen. auto | 9.12472 | -0.13 | 1.5e-9 | 1035 | 14,334 | 1.48e7 | 217 | no | 9444 | 993,712 / 25 / 7.11 | 905,392 / 25 / 6.49 |
| h6a | sf | L 11 | 17.2252 | 0.11 | 4.9e-9 | 672 | 27,058 | 1.82e7 | 184 | yes | 1.158e4 | 926,112 / 25 / 8.71 | 824,736 / 25 / 8.71 |
| n2 | df | thr 3e-3, L 20, Xi 82 | 9.42256 | 0.27 | 4.6e-7 | 1420 | 14,801 | 2.1e7 | 235 | yes | 1.338e4 | 1,030,216 / 25 / 10.1 | 959,616 / 23 / 10.1 |
| n2 | thc | M 21, pen. 0 | 22.5834 | 1.4e-11 | 0 | 1141 | 35,474 | 4.05e7 | 226 | no | 2.577e4 | 1,084,512 / 27 / 19.4 | 943,488 / 23 / 19.4 |
| n2 | sf (selected) | L 20 | 9.71614 | 1.4e-11 | 0 | 828 | 15,263 | 1.26e7 | 185 | no | 8045 | 928,816 / 25 / 6.06 | 784,896 / 23 / 5.53 |
| lih | df | thr 3e-4, L 18, Xi 68 | 9.2592 | 0.042 | 1.3e-8 | 1366 | 14,545 | 1.99e7 | 235 | yes | 1.265e4 | 1,030,216 / 25 / 9.52 | 959,616 / 23 / 9.52 |
| lih | thc | M 21, pen. 0 | 68.9189 | 0 | 0 | 1141 | 108,258 | 1.24e8 | 228 | no | 7.864e4 | 1,141,952 / 27 / 64.3 | 1,015,344 / 25 / 59.2 |
| lih | sf (selected) | L 16 | 13.0786 | 0.12 | 2.1e-6 | 768 | 20,544 | 1.58e7 | 187 | no | 1.004e4 | 932,872 / 25 / 7.56 | 844,552 / 25 / 6.9 |
| h6s | df | thr 1e-3, L 11, Xi 61 | 7.57122 | 0.13 | 2.5e-7 | 1303 | 11,893 | 1.55e7 | 233 | yes | 9865 | 1,026,160 / 25 / 7.43 | 854,784 / 23 / 7.43 |
| h6s | thc (selected) | M 12, pen. 0 | 6.11959 | -0.026 | 1.2e-8 | 1013 | 9,613 | 9.74e6 | 217 | no | 6199 | 993,712 / 25 / 4.67 | 840,192 / 23 / 4.26 |
| h6s | sf | L 11 | 14.1392 | 0.0026 | 5.2e-11 | 672 | 22,210 | 1.49e7 | 184 | yes | 9502 | 926,112 / 25 / 7.15 | 837,792 / 25 / 6.53 |
| n2s | df | thr 3e-3, L 17, Xi 74 | 7.7162 | 0.016 | 5.4e-6 | 1378 | 12,121 | 1.67e7 | 235 | yes | 1.063e4 | 1,030,216 / 25 / 8 | 928,840 / 25 / 8 |
| n2s | thc | M 21, pen. 0 | 20.0891 | 1.4e-11 | 0 | 1141 | 31,556 | 3.6e7 | 224 | no | 2.292e4 | 1,007,232 / 25 / 17.3 | 940,032 / 23 / 17.3 |
| n2s | sf (selected) | L 17 | 7.77897 | -0.23 | 3.5e-6 | 777 | 12,220 | 9.49e6 | 185 | no | 6044 | 873,216 / 23 / 4.55 | 784,896 / 23 / 4.15 |
| h8 | df | thr 1e-3, L 15, Xi 117 | 16.6561 | 0.26 | 1.4e-7 | 1743 | 26,164 | 4.56e7 | 271 | yes | 2.903e4 | 1,191,136 / 27 / 21.9 | 1,021,824 / 23 / 21.9 |
| h8 | thc (selected) | M 24, pen. auto | 14.0836 | -0.025 | 1.2e-9 | 1415 | 22,123 | 3.13e7 | 261 | no | 1.993e4 | 1,082,944 / 25 / 15 | 1,004,544 / 23 / 15 |
| h8 | sf | L 15 | 36.6987 | 0.22 | 1.2e-8 | 829 | 57,647 | 4.78e7 | 185 | no | 3.042e4 | 988,864 / 27 / 22.9 | 873,216 / 23 / 22.9 |

- **Selection.** Per kind, the encoding with the fewest QPE Toffolis among those with
  |ΔE_fact| ≤ 0.5 mHa and ground-state infidelity 1 − |⟨Ψ₀(H′)|Ψ₀(H)⟩|² ≤ 1e-4 (F9);
  "(selected)" is the cheapest overall, the QPE of every hand-off below and the walk of the
  residual routes. It also has the lowest λ·C_W of the three kinds on every system.
- **SF.** Its walk step is 0.44–0.58 of DF's in OpenFermion's models (C_W 384–829 against
  868–1743 Toffolis), so SF beats DF in λ·C_W wherever λ_SF/λ_DF stays below
  C_W(DF)/C_W(SF) = 1.7–2.3: on n2 (λ 9.716 against 9.423 Ha), n2s (7.779 against 7.716), h4
  (6.58 against 4.935), lih (13.08 against 9.259), h6a (1.77× DF's λ, C_W ratio 1.95) and h6s
  (1.87×, 1.94), but not on h8 (2.20×, 2.10; λ·C_W 30,420 against 29,030). It is selected on
  h4, n2, lih and n2s; on h6a and h6s it loses only to THC (λ·C_W 11,580 and 9502 against 9444
  and 6199).
- **Where THC beats DF.** In λ·C_W by 26% on h6a (9444 against 12,730), 37% on h6s (6199
  against 9865) and 31% on h8 (19,930 against 29,030), and in C_QPE by the same factors
  (1.5e7 against 2.0e7, 9.7e6 against 1.5e7, 3.1e7 against 4.6e7). On h4 the full-rank THC fit
  (M = 10) is 1% worse than DF in λ·C_W (4334 against 4283). On n2 and lih THC passes the
  selection rule only at M = 21, and on n2s its cheapest admissible rank is M = 21 (M = 18 passes
  too, at 3.94e7 Toffolis), with λ = 22.6, 68.9 and 20.1 Ha; there it costs 1.9–6.2× DF.
- **Physical.** One run of the selected QPE at budget 1e-2 needs 822,528 (h4) to 1,082,944
  (h8) physical qubits at d = 23–25 and lasts 1.9 (h4) to 15 min (h8); at budget 0.1 the
  distance drops by 0–2.
- **QR clamp.** OpenFermion's QROM rule needed the clamp for every selected DF encoding and for
  SF with L ≤ 11 (h4, h6a, h6s), never for THC.

## 2. State preparation (T2)

**T2** (state preparation; exact records, k ≤ k* + 2, to k = 8 where k* is not reached).

k* (marked *) is the first iterate within 1.6 mHa of E_0. Step k compiles exp(eps_k D_k), the step from Psi_k to Psi_{k+1}, elementwise (F3; delta_step = 1e-3) and as DF leaves (F6, order 1). s / shot: one shot of Psi_k at budget 1e-5 with 2d readout cycles.

| system | family | k | E_k - E_0 (mHa) | p_k | norm A_k | C(Psi_k) | C(Psi_k) leaves | Q(Psi_k) | s / shot | step k: n_d / n_s | b | C_step | C_step leaves |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| h4 | lbfgs | 0 | 61.982 | 0.9455 | 0.8585 | 0 | 0 | 8 | 2.6e-5 | 10 / 0 | 15 | 160 | 3728 |
| h4 | lbfgs | 1 | 21.504 | 0.9802 | 0.6115 | 160 | 3938 | 61 | 0.00424 | 40 / 24 | 18 | 1192 | 5592 |
| h4 | lbfgs | 2 | 3.890 | 0.9960 | 0.2157 | 1352 | 9530 | 70 | 0.0355 | 38 / 24 | 18 | 1154 | 5592 |
| h4 | lbfgs | 3* | 1.203 | 0.9981 | 0.1256 | 2506 | 15122 | 70 | 0.0721 | 40 / 24 | 18 | 1192 | 5592 |
| h4 | lbfgs | 4 | 0.410 | 0.9997 | 0.0702 | 3698 | 20714 | 70 | 0.106 | 40 / 24 | 18 | 1192 | 5592 |
| h4 | lbfgs | 5 | 0.066 | 1.0000 | 0.0295 | 4890 | 26306 | 70 | 0.141 | 38 / 24 | 18 | 1154 | 5592 |
| h4 | lbfgs_so | 0 | 61.982 | 0.9455 | 0.8585 | 0 | 0 | 8 | 2.6e-5 | 10 / 0 | 15 | 160 | 3728 |
| h4 | lbfgs_so | 1 | 8.536 | 0.9911 | 0.3110 | 160 | 3938 | 61 | 0.00424 | 36 / 24 | 18 | 1116 | 5592 |
| h4 | lbfgs_so | 2 | 1.619 | 0.9981 | 0.1345 | 1276 | 9530 | 70 | 0.0335 | 40 / 24 | 18 | 1192 | 5592 |
| h4 | lbfgs_so | 3* | 0.344 | 0.9996 | 0.0925 | 2468 | 15122 | 70 | 0.071 | 40 / 24 | 18 | 1192 | 5592 |
| h4 | lbfgs_so | 4 | 0.090 | 0.9999 | 0.0416 | 3660 | 20714 | 70 | 0.105 | 40 / 24 | 18 | 1192 | 5592 |
| h4 | lbfgs_so | 5 | 0.031 | 1.0000 | 0.0121 | 4852 | 26306 | 70 | 0.14 | 34 / 24 | 18 | 1078 | 5592 |
| h4 | sd | 0 | 61.982 | 0.9455 | 0.8585 | 0 | 0 | 8 | 2.6e-5 | 10 / 0 | 15 | 160 | 3728 |
| h4 | sd | 1 | 21.504 | 0.9802 | 0.6115 | 160 | 3938 | 61 | 0.00424 | 40 / 24 | 18 | 1192 | 5592 |
| h4 | sd | 2 | 9.193 | 0.9896 | 0.3022 | 1352 | 9530 | 70 | 0.0355 | 36 / 24 | 18 | 1116 | 5592 |
| h4 | sd | 3 | 4.314 | 0.9951 | 0.2534 | 2468 | 15122 | 70 | 0.071 | 38 / 24 | 18 | 1154 | 5592 |
| h4 | sd | 4 | 2.174 | 0.9973 | 0.1377 | 3622 | 20714 | 70 | 0.104 | 40 / 24 | 18 | 1192 | 5592 |
| h4 | sd | 5* | 1.147 | 0.9986 | 0.1241 | 4814 | 26306 | 70 | 0.138 | 38 / 24 | 18 | 1154 | 5592 |
| h4 | sd | 6 | 0.632 | 0.9992 | 0.0715 | 5968 | 31898 | 70 | 0.172 | 40 / 22 | 18 | 1156 | 5592 |
| h4 | sd | 7 | 0.373 | 0.9995 | 0.0686 | 7124 | 37490 | 70 | 0.205 | 38 / 24 | 18 | 1154 | 5592 |
| h6a | lbfgs | 0 | 100.534 | 0.9026 | 1.0993 | 0 | 0 | 12 | 2.6e-5 | 51 / 0 | 18 | 969 | 19374 |
| h6a | lbfgs | 1 | 37.956 | 0.9576 | 0.7424 | 969 | 19929 | 74 | 0.0255 | 273 / 120 | 21 | 8526 | 37260 |
| h6a | lbfgs | 2 | 12.014 | 0.9850 | 0.4219 | 9495 | 57189 | 83 | 0.273 | 271 / 120 | 21 | 8482 | 37260 |
| h6a | lbfgs | 3 | 5.236 | 0.9907 | 0.2397 | 17977 | 94449 | 83 | 0.517 | 271 / 120 | 21 | 8482 | 37260 |
| h6a | lbfgs | 4 | 2.991 | 0.9950 | 0.2077 | 26459 | 131709 | 83 | 0.761 | 273 / 120 | 21 | 8526 | 37260 |
| h6a | lbfgs | 5 | 1.877 | 0.9974 | 0.1527 | 34985 | 168969 | 83 | 1.01 | 273 / 116 | 21 | 8442 | 37260 |
| h6a | lbfgs | 6* | 1.111 | 0.9987 | 0.1364 | 43427 | 206229 | 83 | 1.25 | 269 / 120 | 21 | 8438 | 37260 |
| h6a | lbfgs | 7 | 0.764 | 0.9993 | 0.0949 | 51865 | 243489 | 83 | 1.49 | 271 / 120 | 21 | 8482 | 37260 |
| h6a | lbfgs | 8 | 0.441 | 0.9997 | 0.0578 | 60347 | 280749 | 83 | 1.74 | 271 / 120 | 21 | 8482 | 37260 |
| h6a | lbfgs_so | 0 | 100.534 | 0.9026 | 1.0993 | 0 | 0 | 12 | 2.6e-5 | 51 / 0 | 18 | 969 | 19374 |
| h6a | lbfgs_so | 1 | 16.995 | 0.9769 | 0.4069 | 969 | 19929 | 74 | 0.0255 | 273 / 120 | 21 | 8526 | 37260 |
| h6a | lbfgs_so | 2 | 5.184 | 0.9914 | 0.2358 | 9495 | 57189 | 83 | 0.273 | 273 / 120 | 21 | 8526 | 37260 |
| h6a | lbfgs_so | 3 | 2.588 | 0.9961 | 0.2368 | 18021 | 94449 | 83 | 0.518 | 273 / 120 | 21 | 8526 | 37260 |
| h6a | lbfgs_so | 4* | 1.297 | 0.9985 | 0.1371 | 26547 | 131709 | 83 | 0.763 | 273 / 120 | 21 | 8526 | 37260 |
| h6a | lbfgs_so | 5 | 0.705 | 0.9993 | 0.0987 | 35073 | 168969 | 83 | 1.01 | 273 / 118 | 21 | 8484 | 37260 |
| h6a | lbfgs_so | 6 | 0.458 | 0.9997 | 0.0484 | 43557 | 206229 | 83 | 1.25 | 273 / 118 | 21 | 8484 | 37260 |
| h6a | sd | 0 | 100.534 | 0.9026 | 1.0993 | 0 | 0 | 12 | 2.6e-5 | 51 / 0 | 18 | 969 | 19374 |
| h6a | sd | 1 | 37.956 | 0.9576 | 0.7424 | 969 | 19929 | 74 | 0.0255 | 273 / 120 | 21 | 8526 | 37260 |
| h6a | sd | 2 | 19.572 | 0.9738 | 0.4428 | 9495 | 57189 | 83 | 0.273 | 273 / 120 | 21 | 8526 | 37260 |
| h6a | sd | 3 | 11.989 | 0.9820 | 0.3699 | 18021 | 94449 | 83 | 0.518 | 265 / 120 | 21 | 8350 | 37260 |
| h6a | sd | 4 | 7.954 | 0.9878 | 0.2673 | 26371 | 131709 | 83 | 0.758 | 271 / 118 | 21 | 8440 | 37260 |
| h6a | sd | 5 | 5.861 | 0.9900 | 0.2479 | 34811 | 168969 | 83 | 1 | 265 / 118 | 21 | 8308 | 37260 |
| h6a | sd | 6 | 4.626 | 0.9922 | 0.1929 | 43119 | 206229 | 83 | 1.24 | 271 / 120 | 21 | 8482 | 37260 |
| h6a | sd | 7 | 3.809 | 0.9935 | 0.1856 | 51601 | 243489 | 83 | 1.48 | 271 / 118 | 21 | 8440 | 37260 |
| h6a | sd | 8 | 3.210 | 0.9945 | 0.1562 | 60041 | 280749 | 83 | 1.73 | 264 / 118 | 21 | 8286 | 37260 |
| h6a | sd | 9 | 2.773 | 0.9953 | 0.1500 | 68327 | 318009 | 83 | 2.14 | 271 / 120 | 21 | 8482 | 37260 |
| h6a | sd | 10 | 2.422 | 0.9959 | 0.1300 | 76809 | 355269 | 83 | 2.4 | 269 / 120 | 21 | 8438 | 37260 |
| h6a | sd | 11 | 2.131 | 0.9964 | 0.1264 | 85247 | 392529 | 83 | 2.66 | 261 / 120 | 21 | 8262 | 37260 |
| h6a | sd | 12 | 1.884 | 0.9968 | 0.1107 | 93509 | 429789 | 83 | 2.92 | 265 / 118 | 21 | 8308 | 37260 |
| h6a | sd | 13 | 1.673 | 0.9972 | 0.1081 | 101817 | 467049 | 83 | 3.44 | 261 / 118 | 21 | 8220 | 37260 |
| h6a | sd | 14* | 1.492 | 0.9975 | 0.0950 | 110037 | 504309 | 83 | 3.44 | 261 / 116 | 21 | 8178 | 37260 |
| h6a | sd | 15 | 1.336 | 0.9978 | 0.0931 | 118215 | 541569 | 83 | 3.69 | 252 / 118 | 21 | 8022 | 37260 |
| h6a | sd | 16 | 1.201 | 0.9981 | 0.0821 | 126237 | 578829 | 83 | 3.94 | 259 / 116 | 21 | 8134 | 37260 |
| n2 | lbfgs | 0 | 126.040 | 0.9257 | 1.8955 | 0 | 0 | 12 | 2.6e-5 | 27 / 0 | 17 | 486 | 17826 |
| n2 | lbfgs | 1 | 35.209 | 0.9635 | 0.6638 | 486 | 18381 | 71 | 0.0128 | 55 / 0 | 18 | 1045 | 25290 |
| n2 | lbfgs | 2 | 14.678 | 0.9813 | 0.4038 | 1531 | 43671 | 74 | 0.0402 | 55 / 0 | 18 | 1045 | 25290 |
| n2 | lbfgs | 3* | 1.147 | 0.9994 | 0.1168 | 2576 | 68961 | 74 | 0.0741 | 55 / 0 | 18 | 1045 | 25290 |
| n2 | lbfgs | 4 | 0.399 | 0.9998 | 0.0242 | 3621 | 94251 | 74 | 0.104 | 53 / 0 | 18 | 1007 | 25290 |
| n2 | lbfgs | 5 | 0.349 | 0.9999 | 0.0101 | 4628 | 119541 | 74 | 0.133 | 51 / 0 | 18 | 969 | 25290 |
| n2 | lbfgs_so | 0 | 126.040 | 0.9257 | 1.8955 | 0 | 0 | 12 | 2.6e-5 | 27 / 0 | 17 | 486 | 15960 |
| n2 | lbfgs_so | 1 | 10.893 | 0.9904 | 0.4454 | 486 | 16515 | 71 | 0.0128 | 55 / 0 | 18 | 1045 | 25290 |
| n2 | lbfgs_so | 2 | 2.234 | 0.9979 | 0.1640 | 1531 | 41805 | 74 | 0.0402 | 55 / 0 | 18 | 1045 | 25290 |
| n2 | lbfgs_so | 3* | 0.748 | 0.9995 | 0.0870 | 2576 | 67095 | 74 | 0.0741 | 51 / 0 | 18 | 969 | 25290 |
| n2 | lbfgs_so | 4 | 0.384 | 0.9998 | 0.0116 | 3545 | 92385 | 74 | 0.102 | 43 / 0 | 18 | 817 | 25290 |
| n2 | lbfgs_so | 5 | 0.367 | 0.9999 | 0.0054 | 4362 | 117675 | 74 | 0.125 | 47 / 0 | 18 | 893 | 25290 |
| n2 | sd | 0 | 126.040 | 0.9257 | 1.8955 | 0 | 0 | 12 | 2.6e-5 | 27 / 0 | 17 | 486 | 17826 |
| n2 | sd | 1 | 35.209 | 0.9635 | 0.6638 | 486 | 18381 | 71 | 0.0128 | 55 / 0 | 18 | 1045 | 25290 |
| n2 | sd | 2 | 18.446 | 0.9798 | 0.4776 | 1531 | 43671 | 74 | 0.0402 | 55 / 0 | 18 | 1045 | 25290 |
| n2 | sd | 3 | 10.749 | 0.9873 | 0.3399 | 2576 | 68961 | 74 | 0.0741 | 55 / 0 | 18 | 1045 | 25290 |
| n2 | sd | 4 | 6.339 | 0.9932 | 0.2748 | 3621 | 94251 | 74 | 0.104 | 53 / 0 | 18 | 1007 | 25290 |
| n2 | sd | 5 | 3.800 | 0.9957 | 0.1953 | 4628 | 119541 | 74 | 0.133 | 55 / 0 | 18 | 1045 | 25290 |
| n2 | sd | 6 | 2.336 | 0.9977 | 0.1583 | 5673 | 144831 | 74 | 0.163 | 51 / 0 | 18 | 969 | 25290 |
| n2 | sd | 7* | 1.494 | 0.9985 | 0.1122 | 6642 | 170121 | 74 | 0.191 | 55 / 0 | 18 | 1045 | 25290 |
| n2 | sd | 8 | 1.009 | 0.9991 | 0.0910 | 7687 | 195411 | 74 | 0.221 | 51 / 0 | 18 | 969 | 25290 |
| n2 | sd | 9 | 0.731 | 0.9994 | 0.0644 | 8656 | 220701 | 74 | 0.249 | 53 / 0 | 18 | 1007 | 25290 |
| lih | lbfgs | 0 | 20.378 | 0.9743 | 0.3999 | 0 | 0 | 12 | 2.6e-5 | 26 / 0 | 17 | 468 | 13048 |
| lih | lbfgs | 1 | 3.455 | 0.9940 | 0.3068 | 468 | 13603 | 71 | 0.0123 | 100 / 104 | 20 | 4180 | 26628 |
| lih | lbfgs | 2* | 1.011 | 0.9957 | 0.1143 | 4648 | 40231 | 80 | 0.134 | 100 / 86 | 20 | 3820 | 26628 |
| lih | lbfgs | 3 | 0.797 | 0.9963 | 0.0825 | 8468 | 66859 | 80 | 0.244 | 106 / 94 | 20 | 4106 | 26628 |
| lih | lbfgs | 4 | 0.508 | 0.9978 | 0.0940 | 12574 | 93487 | 80 | 0.362 | 110 / 94 | 20 | 4190 | 26628 |
| lih | lbfgs_so | 0 | 20.378 | 0.9743 | 0.3999 | 0 | 0 | 12 | 2.6e-5 | 26 / 0 | 17 | 468 | 13048 |
| lih | lbfgs_so | 1 | 3.409 | 0.9940 | 0.3029 | 468 | 13603 | 71 | 0.0123 | 90 / 84 | 20 | 3570 | 26628 |
| lih | lbfgs_so | 2* | 0.958 | 0.9956 | 0.0845 | 4038 | 40231 | 80 | 0.116 | 94 / 92 | 20 | 3814 | 26628 |
| lih | lbfgs_so | 3 | 0.659 | 0.9966 | 0.0851 | 7852 | 66859 | 80 | 0.226 | 96 / 96 | 20 | 3936 | 26628 |
| lih | lbfgs_so | 4 | 0.361 | 0.9981 | 0.0663 | 11788 | 93487 | 80 | 0.339 | 94 / 92 | 20 | 3814 | 26628 |
| lih | sd | 0 | 20.378 | 0.9743 | 0.3999 | 0 | 0 | 12 | 2.6e-5 | 26 / 0 | 17 | 468 | 13048 |
| lih | sd | 1 | 3.455 | 0.9940 | 0.3068 | 468 | 13603 | 71 | 0.0123 | 100 / 100 | 20 | 4100 | 26628 |
| lih | sd | 2* | 1.316 | 0.9951 | 0.1121 | 4568 | 40231 | 80 | 0.131 | 98 / 80 | 20 | 3658 | 26628 |
| lih | sd | 3 | 1.088 | 0.9957 | 0.0982 | 8226 | 66859 | 80 | 0.237 | 97 / 76 | 20 | 3557 | 26628 |
| lih | sd | 4 | 0.931 | 0.9961 | 0.0855 | 11783 | 93487 | 80 | 0.339 | 78 / 62 | 19 | 2738 | 26628 |
| h6s | lbfgs | 0 | 282.214 | 0.5733 | 1.1519 | 0 | 0 | 12 | 2.6e-5 | 51 / 0 | 18 | 969 | 19374 |
| h6s | lbfgs | 1 | 84.900 | 0.8722 | 0.8124 | 969 | 19929 | 74 | 0.0255 | 273 / 120 | 21 | 8526 | 37260 |
| h6s | lbfgs | 2 | 18.266 | 0.9470 | 0.3413 | 9495 | 57189 | 83 | 0.273 | 273 / 120 | 21 | 8526 | 37260 |
| h6s | lbfgs | 3 | 12.031 | 0.9568 | 0.2804 | 18021 | 94449 | 83 | 0.518 | 273 / 120 | 21 | 8526 | 37260 |
| h6s | lbfgs | 4 | 8.277 | 0.9629 | 0.1883 | 26547 | 131709 | 83 | 0.763 | 271 / 120 | 21 | 8482 | 37260 |
| h6s | lbfgs | 5 | 6.566 | 0.9676 | 0.2134 | 35029 | 168969 | 83 | 1.01 | 271 / 120 | 21 | 8482 | 37260 |
| h6s | lbfgs | 6 | 4.272 | 0.9770 | 0.1107 | 43511 | 206229 | 83 | 1.25 | 273 / 120 | 21 | 8526 | 37260 |
| h6s | lbfgs | 7 | 3.037 | 0.9827 | 0.0940 | 52037 | 243489 | 83 | 1.5 | 271 / 120 | 21 | 8482 | 37260 |
| h6s | lbfgs | 8 | 2.561 | 0.9845 | 0.0740 | 60519 | 280749 | 83 | 1.74 | 273 / 120 | 21 | 8526 | 37260 |
| h6s | lbfgs | 9 | 2.262 | 0.9859 | 0.0421 | 69045 | 318009 | 83 | 2.16 | 273 / 120 | 21 | 8526 | 37260 |
| h6s | lbfgs | 10 | 1.919 | 0.9893 | 0.0425 | 77571 | 355269 | 83 | 2.42 | 273 / 120 | 21 | 8526 | 37260 |
| h6s | lbfgs | 11 | 1.694 | 0.9914 | 0.0390 | 86097 | 392529 | 83 | 2.69 | 273 / 120 | 21 | 8526 | 37260 |
| h6s | lbfgs | 12* | 1.396 | 0.9944 | 0.0393 | 94623 | 429789 | 83 | 2.96 | 273 / 120 | 21 | 8526 | 37260 |
| h6s | lbfgs | 13 | 1.137 | 0.9961 | 0.0380 | 103149 | 467049 | 83 | 3.48 | 271 / 120 | 21 | 8482 | 37260 |
| h6s | lbfgs | 14 | 0.970 | 0.9973 | 0.0302 | 111631 | 504309 | 83 | 3.49 | 273 / 120 | 21 | 8526 | 37260 |
| h6s | lbfgs_so | 0 | 282.214 | 0.5733 | 1.1519 | 0 | 0 | 12 | 2.6e-5 | 51 / 0 | 18 | 969 | 19374 |
| h6s | lbfgs_so | 1 | 39.371 | 0.9135 | 0.5429 | 969 | 19929 | 74 | 0.0255 | 273 / 120 | 21 | 8526 | 37260 |
| h6s | lbfgs_so | 2 | 16.199 | 0.9425 | 0.3110 | 9495 | 57189 | 83 | 0.273 | 271 / 120 | 21 | 8482 | 37260 |
| h6s | lbfgs_so | 3 | 8.102 | 0.9633 | 0.2402 | 17977 | 94449 | 83 | 0.517 | 273 / 120 | 21 | 8526 | 37260 |
| h6s | lbfgs_so | 4 | 5.455 | 0.9720 | 0.1837 | 26503 | 131709 | 83 | 0.762 | 273 / 120 | 21 | 8526 | 37260 |
| h6s | lbfgs_so | 5 | 3.758 | 0.9776 | 0.1486 | 35029 | 168969 | 83 | 1.01 | 273 / 120 | 21 | 8526 | 37260 |
| h6s | lbfgs_so | 6 | 2.811 | 0.9813 | 0.0770 | 43555 | 206229 | 83 | 1.25 | 273 / 120 | 21 | 8526 | 37260 |
| h6s | lbfgs_so | 7 | 2.288 | 0.9853 | 0.0476 | 52081 | 243489 | 83 | 1.5 | 273 / 120 | 21 | 8526 | 37260 |
| h6s | lbfgs_so | 8 | 1.942 | 0.9892 | 0.0548 | 60607 | 280749 | 83 | 1.74 | 273 / 120 | 21 | 8526 | 37260 |
| h6s | lbfgs_so | 9 | 1.759 | 0.9910 | 0.0508 | 69133 | 318009 | 83 | 2.16 | 270 / 120 | 21 | 8460 | 37260 |
| h6s | lbfgs_so | 10* | 1.453 | 0.9936 | 0.0363 | 77593 | 355269 | 83 | 2.42 | 273 / 120 | 21 | 8526 | 37260 |
| h6s | lbfgs_so | 11 | 1.185 | 0.9962 | 0.0384 | 86119 | 392529 | 83 | 2.69 | 273 / 120 | 21 | 8526 | 37260 |
| h6s | lbfgs_so | 12 | 1.024 | 0.9974 | 0.0294 | 94645 | 429789 | 83 | 2.96 | 273 / 120 | 21 | 8526 | 37260 |
| n2s | lbfgs | 0 | 565.520 | 0.2528 | 3.1965 | 0 | 0 | 12 | 2.6e-5 | 27 / 0 | 17 | 486 | 17826 |
| n2s | lbfgs | 1 | 212.223 | 0.2314 | 0.4754 | 486 | 18381 | 71 | 0.0128 | 55 / 0 | 18 | 1045 | 25290 |
| n2s | lbfgs | 2 | 183.366 | 0.3178 | 0.4912 | 1531 | 43671 | 74 | 0.0402 | 55 / 0 | 18 | 1045 | 25290 |
| n2s | lbfgs | 3 | 102.134 | 0.6862 | 0.5292 | 2576 | 68961 | 74 | 0.0741 | 55 / 0 | 18 | 1045 | 25290 |
| n2s | lbfgs | 4 | 61.245 | 0.8417 | 0.3967 | 3621 | 94251 | 74 | 0.104 | 55 / 0 | 18 | 1045 | 25290 |
| n2s | lbfgs | 5 | 34.551 | 0.8609 | 0.2227 | 4666 | 119541 | 74 | 0.134 | 55 / 0 | 18 | 1045 | 25290 |
| n2s | lbfgs | 6 | 22.841 | 0.9167 | 0.2103 | 5711 | 144831 | 74 | 0.164 | 55 / 0 | 18 | 1045 | 25290 |
| n2s | lbfgs | 7 | 16.350 | 0.9318 | 0.0814 | 6756 | 170121 | 74 | 0.194 | 55 / 0 | 18 | 1045 | 25290 |
| n2s | lbfgs | 8 | 13.789 | 0.9418 | 0.0456 | 7801 | 195411 | 74 | 0.224 | 55 / 0 | 18 | 1045 | 25290 |
| n2s | lbfgs_so | 0 | 565.520 | 0.2528 | 3.1965 | 0 | 0 | 12 | 2.6e-5 | 27 / 0 | 17 | 486 | 15960 |
| n2s | lbfgs_so | 1 | 155.050 | 0.4250 | 0.4593 | 486 | 16515 | 71 | 0.0128 | 51 / 0 | 18 | 969 | 25290 |
| n2s | lbfgs_so | 2 | 124.191 | 0.5789 | 0.4935 | 1455 | 41805 | 74 | 0.0382 | 55 / 0 | 18 | 1045 | 25290 |
| n2s | lbfgs_so | 3 | 62.175 | 0.8355 | 0.3176 | 2500 | 67095 | 74 | 0.0719 | 55 / 0 | 18 | 1045 | 25290 |
| n2s | lbfgs_so | 4 | 24.608 | 0.9183 | 0.1823 | 3545 | 92385 | 74 | 0.102 | 55 / 0 | 18 | 1045 | 25290 |
| n2s | lbfgs_so | 5 | 19.711 | 0.9262 | 0.0920 | 4590 | 117675 | 74 | 0.132 | 55 / 0 | 18 | 1045 | 25290 |
| n2s | lbfgs_so | 6 | 17.952 | 0.9390 | 0.0760 | 5635 | 142965 | 74 | 0.162 | 51 / 0 | 18 | 969 | 25290 |
| n2s | lbfgs_so | 7 | 15.651 | 0.9456 | 0.0519 | 6604 | 168255 | 74 | 0.19 | 55 / 0 | 18 | 1045 | 25290 |
| n2s | lbfgs_so | 8 | 14.682 | 0.9549 | 0.0360 | 7649 | 193545 | 74 | 0.22 | 55 / 0 | 18 | 1045 | 25290 |
| h8 | lbfgs | 0 | 122.480 | 0.8864 | 1.3099 | 0 | 0 | 16 | 2.6e-5 | 168 / 0 | 20 | 3528 | 74880 |
| h8 | lbfgs | 1 | 47.515 | 0.9454 | 0.8383 | 3528 | 75972 | 84 | 0.101 | 998 / 336 | 22 | 30346 | 131040 |
| h8 | lbfgs | 2 | 18.864 | 0.9750 | 0.5517 | 33874 | 207012 | 90 | 0.974 | 998 / 336 | 22 | 30346 | 131040 |
| h8 | lbfgs | 3 | 10.279 | 0.9826 | 0.4755 | 64220 | 338052 | 90 | 1.85 | 998 / 334 | 22 | 30302 | 131040 |
| h8 | lbfgs | 4 | 7.585 | 0.9862 | 0.3752 | 94522 | 469092 | 90 | 2.95 | 996 / 336 | 22 | 30300 | 131040 |
| h8 | lbfgs | 5 | 5.274 | 0.9906 | 0.3194 | 124822 | 600132 | 90 | 3.9 | 992 / 332 | 22 | 30120 | 131040 |
| h8 | lbfgs | 6 | 3.796 | 0.9937 | 0.2250 | 154942 | 731172 | 90 | 5.23 | 992 / 334 | 22 | 30164 | 131040 |
| h8 | lbfgs | 7 | 2.884 | 0.9954 | 0.2347 | 185106 | 862212 | 90 | 6.25 | 996 / 332 | 22 | 30212 | 131040 |
| h8 | lbfgs | 8 | 2.010 | 0.9969 | 0.1507 | 215318 | 993252 | 90 | 7.27 | 996 / 332 | 22 | 30212 | 131040 |
| h8 | lbfgs | 9* | 1.363 | 0.9985 | 0.1410 | 245530 | 1124292 | 90 | 8.29 | 992 / 332 | 22 | 30120 | 131040 |
| h8 | lbfgs | 10 | 1.037 | 0.9991 | 0.0991 | 275650 | 1255332 | 90 | 9.3 | 984 / 336 | 22 | 30024 | 131040 |
| h8 | lbfgs | 11 | 0.860 | 0.9993 | 0.0900 | 305674 | 1386372 | 90 | 10.3 | 1000 / 336 | 23 | 31728 | 131040 |
| h8 | lbfgs_so | 0 | 122.480 | 0.8864 | 1.3099 | 0 | 0 | 16 | 2.6e-5 | 168 / 0 | 20 | 3528 | 74880 |
| h8 | lbfgs_so | 1 | 22.770 | 0.9671 | 0.4771 | 3528 | 75972 | 84 | 0.101 | 994 / 334 | 22 | 30210 | 131040 |
| h8 | lbfgs_so | 2 | 8.160 | 0.9853 | 0.3158 | 33738 | 207012 | 90 | 0.97 | 996 / 334 | 22 | 30256 | 131040 |
| h8 | lbfgs_so | 3 | 5.324 | 0.9909 | 0.3782 | 63994 | 338052 | 90 | 1.84 | 996 / 336 | 22 | 30300 | 131040 |
| h8 | lbfgs_so | 4 | 3.112 | 0.9953 | 0.2356 | 94294 | 469092 | 90 | 2.95 | 992 / 330 | 22 | 30076 | 131040 |
| h8 | lbfgs_so | 5 | 2.077 | 0.9973 | 0.1900 | 124370 | 600132 | 90 | 3.89 | 996 / 334 | 22 | 30256 | 131040 |
| h8 | lbfgs_so | 6* | 1.407 | 0.9984 | 0.1476 | 154626 | 731172 | 90 | 5.22 | 992 / 336 | 22 | 30208 | 131040 |
| h8 | lbfgs_so | 7 | 1.067 | 0.9990 | 0.0868 | 184834 | 862212 | 90 | 6.24 | 984 / 336 | 22 | 30024 | 131040 |
| h8 | lbfgs_so | 8 | 0.875 | 0.9993 | 0.0854 | 214858 | 993252 | 90 | 7.25 | 992 / 330 | 22 | 30076 | 131040 |

- **The cost of Ψ_k\*.** Spin-free L-BFGS (k\* marked \*): h4 2506 Toffolis on 70 logical
  qubits (0.072 s per shot at budget 1e-5), h6a 43,427 on 83 (1.25 s), n2 2576 on 74
  (0.074 s), lih 4648 on 80 (0.13 s), h6s 94,623 on 83 (2.96 s), h8 245,530 on 90 (8.3 s).
  n2s does not reach 1.6 mHa in 20 steps. The spin-orbital family reaches it with
  2468 / 26,547 / 2576 / 4038 / 77,593 / 154,626 Toffolis, steepest descent with 4814 (h4),
  110,037 (h6a), 6642 (n2) and 4568 (lih).
- **Steps.** A step costs 160 (the RHF step of h4) to 8526 Toffolis (h6a, h6s) on four to six
  orbitals and 3.0e4 on h8 (992 doubles and 332 singles at b = 22); after the RHF step the
  per-step cost is nearly constant, so C(Ψ_k) grows linearly in k.
- **Leaves.** The DF-leaf circuits of the same steps cost 4.5× (h6s) to 27× (n2) more at k\*,
  and they are order-1 Trotter products of complex rotations (flags of F6).
- **Shots.** A shot of HF is Clifford (2d = 26 µs); one of Ψ_k\* takes 0.07–8.3 s.

## 3. One CQE iteration (T3, T5)

**T3** (measured loop, exact records; the classical-first-step records are in
`results/ft_<system>.json`, `iterations.<family>.classical_first_step`).

S: shots, N: applications of the Hadamard-test circuit, T: Toffolis; AE routes at kappa = 1 unless marked. r = 0.3 over the full tensor, spin-free symmetry-allowed elements. [delta]: the aux routes' optimal delta. Line search: 11 candidates at +-1.6 mHa each, S the shots of all 11 (11 S_E); [flagged]: candidates whose F3 rounding bound exceeds delta_step; AE energies: the F13 option (energies of H' by amplitude estimation on the walk, kappa = 1), not part of C_iter. GAW (F12) is a band and never selected. C_iter = T(best) + T(line search).

| system | family | k | C(Psi_k) | norm A_m | direct_mg S / T | aux_mg S / T [delta] | ae_product N / T | ae_x N / T | ae_aux N / T [delta] | kappa 2: product / x / aux T | GAW T, kappa_g 1 / 10 / 100 | best kappa 1 / 2 | line search S / T [flagged] | line search, AE energies T | C_iter / C_QPE, kappa 1 / 2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| h4 | lbfgs | 0 | 0 | 0.858 | 1.8e4 / 0 | 5.3e4 / 5.9e8 [0.6] | 1.3e5 / 6.2e7 | 3.2e4 / 4.8e7 | 2.6e4 / 2.3e8 [0.4] | 1.2e8 / 9.5e7 / 4.5e8 | 5.2e7 / 5.2e8 / 5.2e9 | direct_mg / direct_mg | 2.6e6 / 8.1e8 [0] | 1.8e8 | 203 / 203 |
| h4 | lbfgs | 1 | 160 | 0.611 | 3.6e4 / 5.8e6 | 9.7e4 / 1.1e9 [0.6] | 1.9e5 / 1.2e8 | 4.5e4 / 7.5e7 | 2.8e4 / 3.2e8 [0.6] | 2.4e8 / 1.5e8 / 6.5e8 | 7.3e7 / 7.3e8 / 7.3e9 | direct_mg / direct_mg | 2.7e6 / 3.1e9 [2] | 4.2e8 | 785 / 785 |
| h4 | lbfgs | 2 | 1352 | 0.216 | 2.9e5 / 3.9e8 | 1e6 / 1.2e10 [0.5] | 5.3e5 / 9.8e8 | 1.3e5 / 3.7e8 | 1.1e5 / 1.1e9 [0.4] | 2e9 / 7.3e8 / 2.1e9 | 2.4e8 / 2.4e9 / 2.4e10 | ae_x / direct_mg | 2.6e6 / 6.1e9 [2] | 7.6e8 | 1.63e3 / 1.64e3 |
| h4 | lbfgs | 3* | 2506 | 0.126 | 8.6e5 / 2.1e9 | 7.1e5 / 1.3e10 [1.2] | 9.1e5 / 2.7e9 | 2.2e5 / 8.8e8 | 8.2e4 / 1.4e9 [1] | 5.5e9 / 1.8e9 / 2.8e9 | 3.1e8 / 3.1e9 / 3.1e10 | ae_x / ae_x | 2.6e6 / 9.2e9 [0] | 1.1e9 | 2.55e3 / 2.77e3 |
| h4 | lbfgs | 4 | 3698 | 0.0702 | 2.7e6 / 1e10 | 1.6e7 / 2e11 [0.4] | 1.6e6 / 6.8e9 | 3.9e5 / 2e9 | 3.6e5 / 4.5e9 [0.4] | 1.4e10 / 4.1e9 / 9e9 | 1e9 / 1e10 / 1e11 | ae_x / ae_x | 2.6e6 / 1.2e10 [0] | 1.4e9 | 3.48e3 / 4e3 |
| h4 | lbfgs | 5 | 4890 | 0.0295 | 1.6e7 / 7.6e10 | 3.8e7 / 6.1e11 [0.6] | 3.9e6 / 2.1e10 | 9.4e5 / 6e9 | 5.6e5 / 9.1e9 [0.6] | 4.2e10 / 1.2e10 / 1.8e10 | 2e9 / 2e10 / 2e11 | ae_x / ae_x | 2.6e6 / 1.5e10 [0] | 1.7e9 | 5.25e3 / 6.76e3 |
| h4 | lbfgs | 6 | 6044 | 0.0155 | 5.6e7 / 3.4e11 | 1.9e8 / 3.2e12 [0.6] | 7.4e6 / 4.8e10 | 1.8e6 / 1.3e10 | 1.3e6 / 2.1e10 [0.5] | 9.7e10 / 2.7e10 / 4.2e10 | 4.7e9 / 4.7e10 / 4.7e11 | ae_x / ae_x | 2.6e6 / 1.7e10 [1] | 2e9 | 7.72e3 / 1.11e4 |
| h4 | lbfgs | 7 | 7198 | 0.0077 | 2.3e8 / 1.6e12 | 2e9 / 3.1e13 [0.4] | 1.5e7 / 1.2e11 | 3.6e6 / 3.1e10 | 4e6 / 6.4e10 [0.4] | 2.3e11 / 6.2e10 / 1.3e11 | 1.4e10 / 1.4e11 / 1.4e12 | ae_x / ae_x | 2.6e6 / 2e10 [1] | 2.3e9 | 1.3e4 / 2.08e4 |
| h4 | lbfgs_so | 0 | 0 | 0.858 | 1.8e4 / 0 | 5.3e4 / 5.9e8 [0.6] | 1.3e5 / 6.2e7 | 3.2e4 / 4.8e7 | 2.6e4 / 2.3e8 [0.4] | 1.2e8 / 9.5e7 / 4.5e8 | 5.2e7 / 5.2e8 / 5.2e9 | direct_mg / direct_mg | 2.8e6 / 8.6e8 [0] | 1.8e8 | 216 / 216 |
| h4 | lbfgs_so | 1 | 160 | 0.311 | 1.4e5 / 2.2e7 | 6.9e5 / 6.1e9 [0.4] | 3.7e5 / 2.4e8 | 8.9e4 / 1.5e8 | 7.6e4 / 6.7e8 [0.4] | 4.8e8 / 3e8 / 1.3e9 | 1.5e8 / 1.5e9 / 1.5e10 | direct_mg / direct_mg | 2.6e6 / 3.1e9 [1] | 4.3e8 | 778 / 778 |
| h4 | lbfgs_so | 2 | 1276 | 0.135 | 7.5e5 / 9.5e8 | 3.2e6 / 3.6e10 [0.5] | 8.5e5 / 1.5e9 | 2.1e5 / 5.7e8 | 1.8e5 / 1.8e9 [0.4] | 3e9 / 1.1e9 / 3.5e9 | 4e8 / 4e9 / 4e10 | ae_x / direct_mg | 2.6e6 / 5.8e9 [1] | 7.4e8 | 1.62e3 / 1.71e3 |
| h4 | lbfgs_so | 3* | 2468 | 0.0925 | 1.6e6 / 3.9e9 | 2.5e6 / 3.8e10 [0.8] | 1.2e6 / 3.7e9 | 3e5 / 1.2e9 | 1.5e5 / 2.2e9 [0.7] | 7.3e9 / 2.4e9 / 4.4e9 | 4.9e8 / 4.9e9 / 4.9e10 | ae_x / ae_x | 2.6e6 / 8.9e9 [0] | 1.1e9 | 2.54e3 / 2.84e3 |
| h4 | lbfgs_so | 4 | 3660 | 0.0416 | 7.8e6 / 2.9e10 | 2e7 / 3e11 [0.6] | 2.7e6 / 1.1e10 | 6.6e5 / 3.4e9 | 4.1e5 / 6.1e9 [0.6] | 2.3e10 / 6.9e9 / 1.2e10 | 1.3e9 / 1.3e10 / 1.3e11 | ae_x / ae_x | 2.6e6 / 1.2e10 [1] | 1.4e9 | 3.79e3 / 4.66e3 |
| h4 | lbfgs_so | 5 | 4852 | 0.0121 | 9.2e7 / 4.4e11 | 5.5e8 / 7.4e12 [0.4] | 9.4e6 / 5e10 | 2.3e6 / 1.4e10 | 2.1e6 / 2.9e10 [0.4] | 1e11 / 2.9e10 / 5.8e10 | 6.4e9 / 6.4e10 / 6.4e11 | ae_x / ae_x | 2.6e6 / 1.4e10 [0] | 1.7e9 | 7.24e3 / 1.09e4 |
| h4 | lbfgs_so | 6 | 5930 | 0.0064 | 3.3e8 / 2e12 | 3.3e9 / 4.5e13 [0.3] | 1.8e7 / 1.2e11 | 4.3e6 / 3.2e10 | 5.3e6 / 7.2e10 [0.3] | 2.3e11 / 6.4e10 / 1.4e11 | 1.6e10 / 1.6e11 / 1.6e12 | ae_x / ae_x | 2.6e6 / 1.7e10 [1] | 1.9e9 | 1.23e4 / 2.04e4 |
| h4 | lbfgs_so | 7 | 7010 | 0.0075 | 2.4e8 / 1.7e12 | 1.6e9 / 2.5e13 [0.4] | 1.5e7 / 1.2e11 | 3.7e6 / 3.1e10 | 3.6e6 / 5.7e10 [0.4] | 2.3e11 / 6.3e10 / 1.1e11 | 1.3e10 / 1.3e11 / 1.3e12 | ae_x / ae_x | 2.6e6 / 2e10 [0] | 2.4e9 | 1.31e4 / 2.1e4 |
| h4 | sd | 0 | 0 | 0.858 | 1.8e4 / 0 | 5.3e4 / 5.9e8 [0.6] | 1.3e5 / 6.2e7 | 3.2e4 / 4.8e7 | 2.6e4 / 2.3e8 [0.4] | 1.2e8 / 9.5e7 / 4.5e8 | 5.2e7 / 5.2e8 / 5.2e9 | direct_mg / direct_mg | 2.6e6 / 8.1e8 [0] | 1.8e8 | 203 / 203 |
| h4 | sd | 1 | 160 | 0.611 | 3.6e4 / 5.8e6 | 9.7e4 / 1.1e9 [0.6] | 1.9e5 / 1.2e8 | 4.5e4 / 7.5e7 | 2.8e4 / 3.2e8 [0.6] | 2.4e8 / 1.5e8 / 6.5e8 | 7.3e7 / 7.3e8 / 7.3e9 | direct_mg / direct_mg | 2.4e6 / 3.5e9 [0] | 5e8 | 883 / 883 |
| h4 | sd | 2 | 1352 | 0.302 | 1.5e5 / 2e8 | 2.8e5 / 3.7e9 [0.7] | 3.8e5 / 7e8 | 9.1e4 / 2.6e8 | 4.8e4 / 6.4e8 [0.7] | 1.4e9 / 5.2e8 / 1.3e9 | 1.4e8 / 1.4e9 / 1.4e10 | direct_mg / direct_mg | 2.5e6 / 6.7e9 [0] | 8.4e8 | 1.73e3 / 1.73e3 |
| h4 | sd | 3 | 2468 | 0.253 | 2.1e5 / 5.2e8 | 5.4e5 / 7.3e9 [0.6] | 4.5e5 / 1.3e9 | 1.1e5 / 4.3e8 | 6.7e4 / 9.2e8 [0.6] | 2.7e9 / 8.7e8 / 1.8e9 | 2e8 / 2e9 / 2e10 | ae_x / direct_mg | 2.5e6 / 9.6e9 [0] | 1.2e9 | 2.52e3 / 2.54e3 |
| h4 | sd | 4 | 3622 | 0.138 | 7.1e5 / 2.6e9 | 1.3e6 / 2.1e10 [0.7] | 8.3e5 / 3.4e9 | 2e5 / 1e9 | 1e5 / 1.6e9 [0.7] | 6.8e9 / 2.1e9 / 3.3e9 | 3.6e8 / 3.6e9 / 3.6e10 | ae_x / ae_x | 2.6e6 / 1.2e10 [1] | 1.5e9 | 3.39e3 / 3.65e3 |
| h4 | sd | 5* | 4814 | 0.124 | 8.8e5 / 4.2e9 | 2.2e6 / 3.5e10 [0.6] | 9.2e5 / 4.9e9 | 2.2e5 / 1.4e9 | 1.3e5 / 2.2e9 [0.6] | 9.8e9 / 2.8e9 / 4.3e9 | 4.8e8 / 4.8e9 / 4.8e10 | ae_x / ae_x | 2.5e6 / 1.6e10 [0] | 1.8e9 | 4.28e3 / 4.63e3 |
| h4 | sd | 6 | 5968 | 0.0715 | 2.6e6 / 1.6e10 | 5.6e6 / 1e11 [0.7] | 1.6e6 / 1e10 | 3.9e5 / 2.9e9 | 2.1e5 / 3.9e9 [0.7] | 2.1e10 / 5.8e9 / 7.8e9 | 8.5e8 / 8.5e9 / 8.5e10 | ae_x / ae_x | 2.6e6 / 1.8e10 [1] | 2.1e9 | 5.36e3 / 6.09e3 |
| h4 | sd | 7 | 7124 | 0.0686 | 2.9e6 / 2e10 | 5.8e6 / 1.1e11 [0.7] | 1.7e6 / 1.3e10 | 4e5 / 3.5e9 | 2.2e5 / 4.2e9 [0.7] | 2.5e10 / 7e9 / 8.4e9 | 9.2e8 / 9.2e9 / 9.2e10 | ae_x / ae_x | 2.6e6 / 2.1e10 [0] | 2.5e9 | 6.29e3 / 7.17e3 |
| h4 | sd | 8 | 8278 | 0.0434 | 7.2e6 / 5.9e10 | 1.8e7 / 3.5e11 [0.6] | 2.6e6 / 2.3e10 | 6.4e5 / 6.2e9 | 3.9e5 / 7.6e9 [0.6] | 4.6e10 / 1.2e10 / 1.5e10 | 1.7e9 / 1.7e10 / 1.7e11 | ae_x / ae_x | 2.6e6 / 2.4e10 [0] | 2.8e9 | 7.7e3 / 9.26e3 |
| h4 | sd | 9 | 9432 | 0.0412 | 8e6 / 7.5e10 | 1.2e7 / 2.6e11 [0.8] | 2.8e6 / 2.8e10 | 6.7e5 / 7.3e9 | 3.1e5 / 6.9e9 [0.8] | 5.5e10 / 1.5e10 / 1.4e10 | 1.5e9 / 1.5e10 / 1.5e11 | ae_aux / ae_aux | 2.6e6 / 2.7e10 [0] | 3.1e9 | 8.63e3 / 1.04e4 |
| h6a | lbfgs | 0 | 0 | 1.1 | 2.3e5 / 0 | 4.4e5 / 1.4e10 [0.6] | 2.1e6 / 2.4e9 | 9e5 / 9.5e9 | 2.6e5 / 7.7e9 [0.5] | 4.9e9 / 1.9e10 / 1.5e10 | 7.8e8 / 7.8e9 / 7.8e10 | direct_mg / direct_mg | 5.3e6 / 7.3e9 [0] | 8.4e8 | 490 / 490 |
| h6a | lbfgs | 1 | 969 | 0.742 | 5.1e5 / 4.9e8 | 9.8e5 / 3.3e10 [0.6] | 3.1e6 / 6.7e9 | 1.3e6 / 1.5e10 | 3.8e5 / 1.2e10 [0.5] | 1.3e10 / 3.1e10 / 2.4e10 | 1.2e9 / 1.2e10 / 1.2e11 | direct_mg / direct_mg | 5.5e6 / 4.7e10 [1] | 3.7e9 | 3.2e3 / 3.2e3 |
| h6a | lbfgs | 2 | 9495 | 0.422 | 1.6e6 / 1.5e10 | 3.2e6 / 1.3e11 [0.6] | 5.5e6 / 5.9e10 | 2.4e6 / 4.7e10 | 6.8e5 / 2.7e10 [0.5] | 1.2e11 / 9.5e10 / 5.4e10 | 2.6e9 / 2.6e10 / 2.6e11 | direct_mg / direct_mg | 5.5e6 / 9.5e10 [1] | 7.1e9 | 7.38e3 / 7.38e3 |
| h6a | lbfgs | 3 | 17977 | 0.24 | 4.9e6 / 8.7e10 | 8.6e6 / 4.3e11 [0.6] | 9.7e6 / 1.9e11 | 4.1e6 / 1.2e11 | 1.1e6 / 5.3e10 [0.6] | 3.7e11 / 2.4e11 / 1.1e11 | 5.1e9 / 5.1e10 / 5.1e11 | ae_aux / direct_mg | 5.4e6 / 1.4e11 [1] | 1e10 | 1.29e4 / 1.52e4 |
| h6a | lbfgs | 4 | 26459 | 0.208 | 6.5e6 / 1.7e11 | 7.8e6 / 5.1e11 [0.8] | 1.1e7 / 3.1e11 | 4.8e6 / 1.8e11 | 1e6 / 6.5e10 [0.7] | 6.2e11 / 3.5e11 / 1.3e11 | 6.1e9 / 6.1e10 / 6.1e11 | ae_aux / ae_aux | 5.3e6 / 1.8e11 [1] | 1.4e10 | 1.66e4 / 2.1e4 |
| h6a | lbfgs | 5 | 34985 | 0.153 | 1.2e7 / 4.2e11 | 2.8e7 / 1.8e12 [0.5] | 1.5e7 / 5.5e11 | 6.5e6 / 3e11 | 1.9e6 / 1.2e11 [0.5] | 1.1e12 / 5.9e11 / 2.5e11 | 1.2e10 / 1.2e11 / 1.2e12 | ae_aux / ae_aux | 5.3e6 / 2.2e11 [1] | 1.7e10 | 2.33e4 / 3.17e4 |
| h6a | lbfgs | 6* | 43427 | 0.136 | 1.5e7 / 6.5e11 | 2.8e7 / 2.1e12 [0.6] | 1.7e7 / 7.6e11 | 7.3e6 / 3.9e11 | 1.9e6 / 1.4e11 [0.6] | 1.5e12 / 7.9e11 / 2.9e11 | 1.3e10 / 1.3e11 / 1.3e12 | ae_aux / ae_aux | 5.3e6 / 2.7e11 [1] | 2e10 | 2.76e4 / 3.73e4 |
| h6a | lbfgs | 7 | 51865 | 0.0949 | 3.1e7 / 1.6e12 | 6e7 / 5.1e12 [0.6] | 2.4e7 / 1.3e12 | 1e7 / 6.5e11 | 2.8e6 / 2.4e11 [0.6] | 2.6e12 / 1.3e12 / 4.7e11 | 2.2e10 / 2.2e11 / 2.2e12 | ae_aux / ae_aux | 5.3e6 / 3.1e11 [1] | 2.3e10 | 3.66e4 / 5.26e4 |
| h6a | lbfgs | 8 | 60347 | 0.0578 | 8.4e7 / 5e12 | 2e8 / 1.8e13 [0.5] | 4e7 / 2.5e12 | 1.7e7 / 1.2e12 | 5.1e6 / 4.6e11 [0.5] | 4.9e12 / 2.4e12 / 9.2e11 | 4.2e10 / 4.2e11 / 4.2e12 | ae_aux / ae_aux | 5.3e6 / 3.5e11 [1] | 2.7e10 | 5.5e4 / 8.61e4 |
| h6a | lbfgs | 9 | 68829 | 0.0467 | 1.3e8 / 8.8e12 | 2.5e8 / 2.5e13 [0.6] | 5e7 / 3.5e12 | 2.1e7 / 1.7e12 | 5.7e6 / 5.8e11 [0.6] | 7e12 / 3.4e12 / 1.2e12 | 5.3e10 / 5.3e11 / 5.3e12 | ae_aux / ae_aux | 5.3e6 / 4e11 [1] | 3e10 | 6.58e4 / 1.05e5 |
| h6a | lbfgs | 10 | 77225 | 0.0266 | 3.9e8 / 3e13 | 1.5e9 / 1.5e14 [0.4] | 8.7e7 / 6.8e12 | 3.7e7 / 3.3e12 | 1.4e7 / 1.4e12 [0.4] | 1.4e13 / 6.6e12 / 2.9e12 | 1.3e11 / 1.3e12 / 1.3e13 | ae_aux / ae_aux | 5.3e6 / 4.4e11 [1] | 3.3e10 | 1.27e5 / 2.24e5 |
| h6a | lbfgs_so | 0 | 0 | 1.1 | 2.3e5 / 0 | 4.4e5 / 1.4e10 [0.6] | 2.1e6 / 2.4e9 | 9e5 / 9.5e9 | 2.6e5 / 7.7e9 [0.5] | 4.9e9 / 1.9e10 / 1.5e10 | 7.8e8 / 7.8e9 / 7.8e10 | direct_mg / direct_mg | 5.6e6 / 7.7e9 [0] | 8.4e8 | 520 / 520 |
| h6a | lbfgs_so | 1 | 969 | 0.407 | 1.7e6 / 1.6e9 | 5.7e6 / 1.5e11 [0.4] | 5.7e6 / 1.2e10 | 2.4e6 / 2.8e10 | 8.6e5 / 2.3e10 [0.4] | 2.5e10 / 5.6e10 / 4.6e10 | 2.4e9 / 2.4e10 / 2.4e11 | direct_mg / direct_mg | 5.4e6 / 4.6e10 [2] | 3.7e9 | 3.24e3 / 3.24e3 |
| h6a | lbfgs_so | 2 | 9495 | 0.236 | 5e6 / 4.8e10 | 1.5e7 / 6e11 [0.5] | 9.9e6 / 1.1e11 | 4.2e6 / 8.5e10 | 1.5e6 / 5.3e10 [0.4] | 2.1e11 / 1.7e11 / 1.1e11 | 5.2e9 / 5.2e10 / 5.2e11 | direct_mg / direct_mg | 5.3e6 / 9.1e10 [1] | 7.1e9 | 9.37e3 / 9.37e3 |
| h6a | lbfgs_so | 3 | 18021 | 0.237 | 5e6 / 9e10 | 8.3e6 / 4.2e11 [0.6] | 9.8e6 / 1.9e11 | 4.2e6 / 1.2e11 | 1e6 / 5.3e10 [0.6] | 3.8e11 / 2.4e11 / 1.1e11 | 5e9 / 5e10 / 5e11 | ae_aux / direct_mg | 5.3e6 / 1.3e11 [1] | 1e10 | 1.26e4 / 1.51e4 |
| h6a | lbfgs_so | 4* | 26547 | 0.137 | 1.5e7 / 3.9e11 | 4.2e7 / 2.4e12 [0.5] | 1.7e7 / 4.7e11 | 7.3e6 / 2.7e11 | 2.5e6 / 1.3e11 [0.4] | 9.4e11 / 5.4e11 / 2.7e11 | 1.3e10 / 1.3e11 / 1.3e12 | ae_aux / ae_aux | 5.3e6 / 1.8e11 [2] | 1.3e10 | 2.09e4 / 2.99e4 |
| h6a | lbfgs_so | 5 | 35073 | 0.0987 | 2.9e7 / 1e12 | 4.8e7 / 3.2e12 [0.6] | 2.4e7 / 8.5e11 | 1e7 / 4.6e11 | 2.5e6 / 1.7e11 [0.6] | 1.7e12 / 9.2e11 / 3.4e11 | 1.6e10 / 1.6e11 / 1.6e12 | ae_aux / ae_aux | 5.3e6 / 2.2e11 [1] | 1.7e10 | 2.63e4 / 3.76e4 |
| h6a | lbfgs_so | 6 | 43557 | 0.0484 | 1.2e8 / 5.2e12 | 4.6e8 / 3.2e13 [0.4] | 4.8e7 / 2.1e12 | 2.1e7 / 1.1e12 | 7.8e6 / 5.4e11 [0.4] | 4.3e12 / 2.2e12 / 1.1e12 | 5e10 / 5e11 / 5e12 | ae_aux / ae_aux | 5.3e6 / 2.6e11 [1] | 2e10 | 5.42e4 / 9.07e4 |
| h6a | lbfgs_so | 7 | 52041 | 0.0333 | 2.5e8 / 1.3e13 | 9.1e8 / 7.1e13 [0.4] | 7e7 / 3.7e12 | 3e7 / 1.9e12 | 1.1e7 / 8.5e11 [0.4] | 7.4e12 / 3.7e12 / 1.7e12 | 7.8e10 / 7.8e11 / 7.8e12 | ae_aux / ae_aux | 5.3e6 / 3.1e11 [1] | 2.3e10 | 7.8e4 / 1.35e5 |
| h6a | lbfgs_so | 8 | 60479 | 0.0387 | 1.9e8 / 1.1e13 | 4.5e8 / 4.1e13 [0.5] | 6e7 / 3.7e12 | 2.6e7 / 1.8e12 | 7.7e6 / 7e11 [0.5] | 7.4e12 / 3.7e12 / 1.4e12 | 6.4e10 / 6.4e11 / 6.4e12 | ae_aux / ae_aux | 5.3e6 / 3.5e11 [0] | 2.7e10 | 7.09e4 / 1.18e5 |
| h6a | sd | 0 | 0 | 1.1 | 2.3e5 / 0 | 4.4e5 / 1.4e10 [0.6] | 2.1e6 / 2.4e9 | 9e5 / 9.5e9 | 2.6e5 / 7.7e9 [0.5] | 4.9e9 / 1.9e10 / 1.5e10 | 7.8e8 / 7.8e9 / 7.8e10 | direct_mg / direct_mg | 5.3e6 / 7.3e9 [0] | 8.4e8 | 490 / 490 |
| h6a | sd | 1 | 969 | 0.742 | 5.1e5 / 4.9e8 | 9.8e5 / 3.3e10 [0.6] | 3.1e6 / 6.7e9 | 1.3e6 / 1.5e10 | 3.8e5 / 1.2e10 [0.5] | 1.3e10 / 3.1e10 / 2.4e10 | 1.2e9 / 1.2e10 / 1.2e11 | direct_mg / direct_mg | 5e6 / 4.9e10 [0] | 4.2e9 | 3.34e3 / 3.34e3 |
| h6a | sd | 2 | 9495 | 0.443 | 1.4e6 / 1.4e10 | 2.4e6 / 1e11 [0.6] | 5.2e6 / 5.6e10 | 2.2e6 / 4.5e10 | 5.6e5 / 2.4e10 [0.6] | 1.1e11 / 9e10 / 4.7e10 | 2.3e9 / 2.3e10 / 2.3e11 | direct_mg / direct_mg | 5.2e6 / 9.5e10 [0] | 7.5e9 | 7.33e3 / 7.33e3 |
| h6a | sd | 3 | 18021 | 0.37 | 2e6 / 3.7e10 | 3.9e6 / 2e11 [0.6] | 6.3e6 / 1.2e11 | 2.7e6 / 7.7e10 | 7.2e5 / 3.6e10 [0.6] | 2.4e11 / 1.5e11 / 7.2e10 | 3.4e9 / 3.4e10 / 3.4e11 | ae_aux / direct_mg | 5.2e6 / 1.4e11 [0] | 1.1e10 | 1.18e4 / 1.18e4 |
| h6a | sd | 4 | 26371 | 0.267 | 3.9e6 / 1e11 | 5e6 / 3.1e11 [0.7] | 8.7e6 / 2.4e11 | 3.7e6 / 1.4e11 | 8.1e5 / 5.1e10 [0.7] | 4.8e11 / 2.7e11 / 1e11 | 4.8e9 / 4.8e10 / 4.8e11 | ae_aux / ae_aux | 5.2e6 / 1.8e11 [0] | 1.4e10 | 1.58e4 / 1.92e4 |
| h6a | sd | 5 | 34811 | 0.248 | 4.5e6 / 1.6e11 | 4.4e6 / 3.2e11 [0.8] | 9.4e6 / 3.4e11 | 4e6 / 1.8e11 | 7.5e5 / 5.6e10 [0.8] | 6.7e11 / 3.6e11 / 1.1e11 | 5.1e9 / 5.1e10 / 5.1e11 | ae_aux / ae_aux | 5.2e6 / 2.3e11 [0] | 1.7e10 | 1.91e4 / 2.28e4 |
| h6a | sd | 6 | 43119 | 0.193 | 7.5e6 / 3.2e11 | 4e6 / 3.7e11 [1.2] | 1.2e7 / 5.3e11 | 5.2e6 / 2.8e11 | 7.6e5 / 6.7e10 [1] | 1.1e12 / 5.5e11 / 1.3e11 | 6.1e9 / 6.1e10 / 6.1e11 | ae_aux / ae_aux | 5.2e6 / 2.7e11 [0] | 2.1e10 | 2.28e4 / 2.73e4 |
| h6a | sd | 7 | 51601 | 0.186 | 8.1e6 / 4.2e11 | 7.5e6 / 6.7e11 [0.8] | 1.3e7 / 6.6e11 | 5.4e6 / 3.3e11 | 9.9e5 / 8.9e10 [0.8] | 1.3e12 / 6.7e11 / 1.8e11 | 8.2e9 / 8.2e10 / 8.2e11 | ae_aux / ae_aux | 5.2e6 / 3.2e11 [0] | 2.4e10 | 2.74e4 / 3.34e4 |
| h6a | sd | 8 | 60041 | 0.156 | 1.1e7 / 6.9e11 | 9.6e6 / 1e12 [1] | 1.5e7 / 9.1e11 | 6.4e6 / 4.5e11 | 1.2e6 / 1.1e11 [0.8] | 1.8e12 / 9e11 / 2.3e11 | 1e10 / 1e11 / 1e12 | ae_aux / ae_aux | 5.2e6 / 3.6e11 [1] | 2.7e10 | 3.2e4 / 3.97e4 |
| h6a | sd | 9 | 68327 | 0.15 | 1.2e7 / 8.5e11 | 1.4e7 / 1.6e12 [0.8] | 1.5e7 / 1.1e12 | 6.6e6 / 5.2e11 | 1.4e6 / 1.5e11 [0.8] | 2.2e12 / 1e12 / 3e11 | 1.3e10 / 1.3e11 / 1.3e12 | ae_aux / ae_aux | 5.3e6 / 4.1e11 [0] | 3.1e10 | 3.73e4 / 4.72e4 |
| h6a | sd | 10 | 76809 | 0.13 | 1.7e7 / 1.3e12 | 1.7e7 / 2e12 [0.8] | 1.8e7 / 1.4e12 | 7.6e6 / 6.7e11 | 1.5e6 / 1.7e11 [0.8] | 2.8e12 / 1.3e12 / 3.5e11 | 1.6e10 / 1.6e11 / 1.6e12 | ae_aux / ae_aux | 5.3e6 / 4.5e11 [0] | 3.4e10 | 4.2e4 / 5.37e4 |
| h6a | sd | 11 | 85247 | 0.126 | 1.7e7 / 1.5e12 | 2.3e7 / 2.8e12 [0.7] | 1.8e7 / 1.6e12 | 7.9e6 / 7.5e11 | 1.7e6 / 2.1e11 [0.7] | 3.2e12 / 1.5e12 / 4.2e11 | 1.9e10 / 1.9e11 / 1.9e12 | ae_aux / ae_aux | 5.3e6 / 4.9e11 [1] | 3.7e10 | 4.76e4 / 6.18e4 |
| h6a | sd | 12 | 93509 | 0.111 | 2.3e7 / 2.1e12 | 2.6e7 / 3.5e12 [0.8] | 2.1e7 / 2e12 | 9e6 / 9.4e11 | 1.9e6 / 2.5e11 [0.8] | 4e12 / 1.9e12 / 4.9e11 | 2.2e10 / 2.2e11 / 2.2e12 | ae_aux / ae_aux | 5.3e6 / 5.4e11 [0] | 4.1e10 | 5.27e4 / 6.93e4 |
| h6a | sd | 13 | 101817 | 0.108 | 2.4e7 / 2.4e12 | 3.3e7 / 4.6e12 [0.7] | 2.1e7 / 2.2e12 | 9.2e6 / 1e12 | 2.1e6 / 2.9e11 [0.7] | 4.4e12 / 2.1e12 / 5.8e11 | 2.6e10 / 2.6e11 / 2.6e12 | ae_aux / ae_aux | 5.3e6 / 5.8e11 [1] | 4.4e10 | 5.87e4 / 7.81e4 |
| h6a | sd | 14* | 110037 | 0.095 | 3.1e7 / 3.4e12 | 4e7 / 5.9e12 [0.7] | 2.4e7 / 2.7e12 | 1e7 / 1.3e12 | 2.3e6 / 3.4e11 [0.7] | 5.4e12 / 2.5e12 / 6.7e11 | 3e10 / 3e11 / 3e12 | ae_aux / ae_aux | 5.3e6 / 6.2e11 [0] | 4.7e10 | 6.47e4 / 8.73e4 |
| h6a | sd | 15 | 118215 | 0.0931 | 3.2e7 / 3.8e12 | 4.7e7 / 7.3e12 [0.7] | 2.5e7 / 3e12 | 1.1e7 / 1.4e12 | 2.5e6 / 3.8e11 [0.7] | 6e12 / 2.8e12 / 7.7e11 | 3.4e10 / 3.4e11 / 3.4e12 | ae_aux / ae_aux | 5.3e6 / 6.7e11 [0] | 5e10 | 7.1e4 / 9.69e4 |
| h6a | sd | 16 | 126237 | 0.0821 | 4.1e7 / 5.2e12 | 5.7e7 / 9.3e12 [0.7] | 2.8e7 / 3.6e12 | 1.2e7 / 1.7e12 | 2.7e6 / 4.5e11 [0.7] | 7.2e12 / 3.3e12 / 8.9e11 | 4e10 / 4e11 / 4e12 | ae_aux / ae_aux | 5.3e6 / 7.1e11 [0] | 5.3e10 | 7.79e4 / 1.08e5 |
| h6a | sd | 17 | 134371 | 0.0808 | 4.3e7 / 5.8e12 | 6.7e7 / 1.1e13 [0.7] | 2.9e7 / 3.9e12 | 1.2e7 / 1.8e12 | 3e6 / 5.1e11 [0.7] | 7.8e12 / 3.6e12 / 1e12 | 4.5e10 / 4.5e11 / 4.5e12 | ae_aux / ae_aux | 5.3e6 / 7.6e11 [0] | 5.7e10 | 8.5e4 / 1.19e5 |
| h6a | sd | 18 | 142481 | 0.0714 | 5.5e7 / 7.8e12 | 8.2e7 / 1.5e13 [0.7] | 3.3e7 / 4.7e12 | 1.4e7 / 2.1e12 | 3.3e6 / 5.9e11 [0.7] | 9.3e12 / 4.3e12 / 1.2e12 | 5.2e10 / 5.2e11 / 5.2e12 | ae_aux / ae_aux | 5.3e6 / 8e11 [0] | 6e10 | 9.33e4 / 1.33e5 |
| n2 | lbfgs | 0 | 0 | 1.9 | 1.2e4 / 0 | 5.1e4 / 1.2e9 [0.4] | 1.1e5 / 1e8 | 1.5e4 / 3.9e7 | 1.6e4 / 3.6e8 [0.4] | 2e8 / 7.9e7 / 7.2e8 | 7.5e7 / 7.5e8 / 7.5e9 | direct_mg / direct_mg | 3.6e6 / 3.2e9 [0] | 6e8 | 250 / 250 |
| n2 | lbfgs | 1 | 486 | 0.664 | 9.5e4 / 4.6e7 | 1.7e5 / 5e9 [0.7] | 3.2e5 / 4.6e8 | 4.2e4 / 1.3e8 | 3.4e4 / 8.6e8 [0.5] | 9.1e8 / 2.7e8 / 1.7e9 | 1.8e8 / 1.8e9 / 1.8e10 | direct_mg / direct_mg | 3.3e6 / 5.8e9 [1] | 9.6e8 | 463 / 463 |
| n2 | lbfgs | 2 | 1531 | 0.404 | 2.6e5 / 3.9e8 | 3.9e5 / 1.2e10 [0.7] | 5.2e5 / 1.3e9 | 7e4 / 2.9e8 | 4.4e4 / 1.4e9 [0.7] | 2.6e9 / 5.9e8 / 2.8e9 | 2.8e8 / 2.8e9 / 2.8e10 | ae_x / direct_mg | 3.6e6 / 9.9e9 [2] | 1.4e9 | 809 / 817 |
| n2 | lbfgs | 3* | 2576 | 0.117 | 3.1e6 / 7.9e9 | 1.6e7 / 3.9e11 [0.4] | 1.8e6 / 6.3e9 | 2.4e5 / 1.3e9 | 2.8e5 / 7e9 [0.4] | 1.3e10 / 2.5e9 / 1.4e10 | 1.5e9 / 1.5e10 / 1.5e11 | ae_x / ae_x | 3.6e6 / 1.4e10 [1] | 1.8e9 | 1.18e3 / 1.28e3 |
| n2 | lbfgs | 4 | 3621 | 0.0242 | 7.1e7 / 2.6e11 | 5.6e8 / 1.3e13 [0.3] | 8.7e6 / 4e10 | 1.2e6 / 7.4e9 | 1.7e6 / 3.8e10 [0.3] | 7.9e10 / 1.5e10 / 7.6e10 | 7.9e9 / 7.9e10 / 7.9e11 | ae_x / ae_x | 3.6e6 / 1.6e10 [2] | 2.2e9 | 1.87e3 / 2.46e3 |
| n2 | lbfgs | 5 | 4628 | 0.0101 | 4.1e8 / 1.9e12 | 5.5e9 / 1.3e14 [0.3] | 2.1e7 / 1.2e11 | 2.8e6 / 2e10 | 5.7e6 / 1.2e11 [0.2] | 2.3e11 / 4.1e10 / 2.3e11 | 2.4e10 / 2.4e11 / 2.4e12 | ae_x / ae_x | 3.6e6 / 1.9e10 [2] | 2.5e9 | 3.15e3 / 4.75e3 |
| n2 | lbfgs | 6 | 5597 | 0.00557 | 1.3e9 / 7.5e12 | 4.5e10 / 9.7e14 [0.2] | 3.8e7 / 2.5e11 | 5.1e6 / 4.2e10 | 1.5e7 / 3.2e11 [0.2] | 4.9e11 / 8.4e10 / 6.4e11 | 6.7e10 / 6.7e11 / 6.7e12 | ae_x / ae_x | 3.5e6 / 2.3e10 [1] | 2.9e9 | 5.12e3 / 8.44e3 |
| n2 | lbfgs | 7 | 6642 | 0.0167 | 1.5e8 / 9.9e11 | 1.1e9 / 2.9e13 [0.3] | 1.3e7 / 9.5e10 | 1.7e6 / 1.6e10 | 2.4e6 / 6.1e10 [0.3] | 1.9e11 / 3.1e10 / 1.2e11 | 1.3e10 / 1.3e11 / 1.3e12 | ae_x / ae_x | 3.6e6 / 2.8e10 [0] | 3.4e9 | 3.44e3 / 4.68e3 |
| n2 | lbfgs_so | 0 | 0 | 1.9 | 1.2e4 / 0 | 5.1e4 / 1.2e9 [0.4] | 1.1e5 / 1e8 | 1.5e4 / 3.9e7 | 1.6e4 / 3.6e8 [0.4] | 2e8 / 7.9e7 / 7.2e8 | 7.5e7 / 7.5e8 / 7.5e9 | direct_mg / direct_mg | 3.7e6 / 3.3e9 [0] | 6e8 | 259 / 259 |
| n2 | lbfgs_so | 1 | 486 | 0.445 | 2.1e5 / 1e8 | 7.3e5 / 1.8e10 [0.5] | 4.7e5 / 6.8e8 | 6.3e4 / 2e8 | 6e4 / 1.5e9 [0.5] | 1.4e9 / 4e8 / 3e9 | 3.1e8 / 3.1e9 / 3.1e10 | direct_mg / direct_mg | 3.4e6 / 6.1e9 [0] | 9.7e8 | 491 / 491 |
| n2 | lbfgs_so | 2 | 1531 | 0.164 | 1.5e6 / 2.4e9 | 4.5e6 / 1.2e11 [0.5] | 1.3e6 / 3.2e9 | 1.7e5 / 7.2e8 | 1.5e5 / 3.9e9 [0.5] | 6.4e9 / 1.4e9 / 7.8e9 | 8e8 / 8e9 / 8e10 | ae_x / ae_x | 3.5e6 / 9.1e9 [2] | 1.3e9 | 775 / 833 |
| n2 | lbfgs_so | 3* | 2576 | 0.087 | 5.5e6 / 1.4e10 | 1.6e7 / 4.3e11 [0.5] | 2.4e6 / 8.5e9 | 3.2e5 / 1.7e9 | 2.8e5 / 7.6e9 [0.5] | 1.7e10 / 3.4e9 / 1.5e10 | 1.6e9 / 1.6e10 / 1.6e11 | ae_x / ae_x | 3.6e6 / 1.3e10 [0] | 1.7e9 | 1.14e3 / 1.28e3 |
| n2 | lbfgs_so | 4 | 3545 | 0.0116 | 3.1e8 / 1.1e12 | 5.1e9 / 9.9e13 [0.2] | 1.8e7 / 8.1e10 | 2.4e6 / 1.5e10 | 5e6 / 9.8e10 [0.2] | 1.6e11 / 3e10 / 2e11 | 2e10 / 2e11 / 2e12 | ae_x / ae_x | 3.5e6 / 1.5e10 [1] | 2e9 | 2.41e3 / 3.6e3 |
| n2 | lbfgs_so | 5 | 4362 | 0.00535 | 1.5e9 / 6.4e12 | 4.9e10 / 9.8e14 [0.2] | 3.9e7 / 2.1e11 | 5.3e6 / 3.7e10 | 1.6e7 / 3.1e11 [0.2] | 4.2e11 / 7.4e10 / 6.3e11 | 6.6e10 / 6.6e11 / 6.6e12 | ae_x / ae_x | 3.6e6 / 1.8e10 [1] | 2.4e9 | 4.38e3 / 7.32e3 |
| n2 | lbfgs_so | 6 | 5255 | 0.00645 | 1e9 / 5.3e12 | 2.4e10 / 5.1e14 [0.2] | 3.2e7 / 2e11 | 4.4e6 / 3.5e10 | 1.1e7 / 2.3e11 [0.2] | 4e11 / 6.9e10 / 4.6e11 | 4.8e10 / 4.8e11 / 4.8e12 | ae_x / ae_x | 3.6e6 / 2.1e10 [2] | 2.7e9 | 4.44e3 / 7.18e3 |
| n2 | lbfgs_so | 7 | 6300 | 0.0222 | 8.4e7 / 5.3e11 | 8.4e8 / 2.1e13 [0.3] | 9.4e6 / 6.8e10 | 1.3e6 / 1.1e10 | 2e6 / 5.2e10 [0.3] | 1.4e11 / 2.3e10 / 1e11 | 1.1e10 / 1.1e11 / 1.1e12 | ae_x / ae_x | 3.6e6 / 2.6e10 [2] | 3.3e9 | 2.98e3 / 3.89e3 |
| n2 | sd | 0 | 0 | 1.9 | 1.2e4 / 0 | 5.1e4 / 1.2e9 [0.4] | 1.1e5 / 1e8 | 1.5e4 / 3.9e7 | 1.6e4 / 3.6e8 [0.4] | 2e8 / 7.9e7 / 7.2e8 | 7.5e7 / 7.5e8 / 7.5e9 | direct_mg / direct_mg | 3.6e6 / 3.2e9 [0] | 6e8 | 250 / 250 |
| n2 | sd | 1 | 486 | 0.664 | 9.5e4 / 4.6e7 | 1.7e5 / 5e9 [0.7] | 3.2e5 / 4.6e8 | 4.2e4 / 1.3e8 | 3.4e4 / 8.6e8 [0.5] | 9.1e8 / 2.7e8 / 1.7e9 | 1.8e8 / 1.8e9 / 1.8e10 | direct_mg / direct_mg | 3.1e6 / 6e9 [0] | 1e9 | 478 / 478 |
| n2 | sd | 2 | 1531 | 0.478 | 1.8e5 / 2.8e8 | 6.3e5 / 1.6e10 [0.5] | 4.4e5 / 1.1e9 | 5.9e4 / 2.5e8 | 5.6e4 / 1.5e9 [0.5] | 2.2e9 / 5e8 / 2.9e9 | 3e8 / 3e9 / 3e10 | ae_x / direct_mg | 3.5e6 / 1e10 [0] | 1.5e9 | 825 / 828 |
| n2 | sd | 3 | 2576 | 0.34 | 3.6e5 / 9.3e8 | 7.5e5 / 2.3e10 [0.6] | 6.2e5 / 2.2e9 | 8.3e4 / 4.4e8 | 6.8e4 / 1.8e9 [0.5] | 4.4e9 / 8.7e8 / 3.7e9 | 3.8e8 / 3.8e9 / 3.8e10 | ae_x / ae_x | 3.3e6 / 1.3e10 [0] | 1.9e9 | 1.07e3 / 1.11e3 |
| n2 | sd | 4 | 3621 | 0.275 | 5.5e5 / 2e9 | 1.7e6 / 4.9e10 [0.5] | 7.6e5 / 3.5e9 | 1e5 / 6.5e8 | 9.3e4 / 2.6e9 [0.5] | 7e9 / 1.3e9 / 5.2e9 | 5.4e8 / 5.4e9 / 5.4e10 | ae_x / ae_x | 3.5e6 / 1.7e10 [1] | 2.3e9 | 1.42e3 / 1.47e3 |
| n2 | sd | 5 | 4628 | 0.195 | 1.1e6 / 5.1e9 | 2.4e6 / 7.9e10 [0.6] | 1.1e6 / 6e9 | 1.4e5 / 1.1e9 | 1.2e5 / 3.5e9 [0.5] | 1.2e10 / 2.1e9 / 7e9 | 7.2e8 / 7.2e9 / 7.2e10 | ae_x / ae_x | 3.4e6 / 2.1e10 [0] | 2.8e9 | 1.71e3 / 1.79e3 |
| n2 | sd | 6 | 5673 | 0.158 | 1.7e6 / 9.4e9 | 4.9e6 / 1.5e11 [0.5] | 1.3e6 / 8.8e9 | 1.8e5 / 1.5e9 | 1.6e5 / 4.7e9 [0.5] | 1.8e10 / 3e9 / 9.4e9 | 9.7e8 / 9.7e9 / 9.7e10 | ae_x / ae_x | 3.5e6 / 2.4e10 [1] | 3.2e9 | 2.05e3 / 2.17e3 |
| n2 | sd | 7* | 6642 | 0.112 | 3.3e6 / 2.2e10 | 8.4e6 / 2.9e11 [0.6] | 1.9e6 / 1.4e10 | 2.5e5 / 2.3e9 | 2.2e5 / 6.7e9 [0.5] | 2.8e10 / 4.7e9 / 1.3e10 | 1.4e9 / 1.4e10 / 1.4e11 | ae_x / ae_x | 3.5e6 / 2.8e10 [1] | 3.6e9 | 2.38e3 / 2.57e3 |
| n2 | sd | 8 | 7687 | 0.091 | 5e6 / 3.9e10 | 1.2e7 / 4.4e11 [0.6] | 2.3e6 / 2e10 | 3.1e5 / 3.2e9 | 2.6e5 / 8.4e9 [0.5] | 4e10 / 6.4e9 / 1.7e10 | 1.7e9 / 1.7e10 / 1.7e11 | ae_x / ae_x | 3.5e6 / 3.2e10 [0] | 4e9 | 2.76e3 / 3.01e3 |
| n2 | sd | 9 | 8656 | 0.0644 | 1e7 / 8.7e10 | 3.3e7 / 1.1e12 [0.5] | 3.3e6 / 3.1e10 | 4.4e5 / 5e9 | 4.1e5 / 1.3e10 [0.5] | 6.3e10 / 9.9e9 / 2.7e10 | 2.8e9 / 2.8e10 / 2.8e11 | ae_x / ae_x | 3.5e6 / 3.5e10 [0] | 4.4e9 | 3.17e3 / 3.56e3 |
| n2 | sd | 10 | 9663 | 0.0524 | 1.5e7 / 1.5e11 | 3.4e7 / 1.3e12 [0.6] | 4e6 / 4.2e10 | 5.4e5 / 6.6e9 | 4.4e5 / 1.5e10 [0.5] | 8.5e10 / 1.3e10 / 3e10 | 3.1e9 / 3.1e10 / 3.1e11 | ae_x / ae_x | 3.5e6 / 3.8e10 [0] | 4.8e9 | 3.56e3 / 4.09e3 |
| n2 | sd | 11 | 10518 | 0.0372 | 3e7 / 3.2e11 | 1.5e8 / 4.9e12 [0.4] | 5.6e6 / 6.5e10 | 7.6e5 / 1e10 | 8.6e5 / 2.9e10 [0.4] | 1.3e11 / 2e10 / 5.7e10 | 5.9e9 / 5.9e10 / 5.9e11 | ae_x / ae_x | 3.5e6 / 4.2e10 [0] | 5.2e9 | 4.1e3 / 4.89e3 |
| lih | lbfgs | 0 | 0 | 0.4 | 2.6e6 / 0 | 3.1e6 / 8.7e10 [0.5] | 4.2e6 / 3.7e9 | 6.4e5 / 3.3e9 | 4.6e5 / 1.1e10 [0.4] | 7.3e9 / 6.7e9 / 2.2e10 | 1.4e9 / 1.4e10 / 1.4e11 | direct_mg / direct_mg | 2.9e6 / 2.4e9 [1] | 7.5e8 | 155 / 155 |
| lih | lbfgs | 1 | 468 | 0.307 | 4.5e6 / 2.1e9 | 3.7e7 / 6.2e11 [0.2] | 5.5e6 / 7.5e9 | 8.3e5 / 4.8e9 | 1.4e6 / 2.4e10 [0.2] | 1.5e10 / 9.5e9 / 4.8e10 | 3.1e9 / 3.1e10 / 3.1e11 | direct_mg / direct_mg | 2.7e6 / 1.1e10 [0] | 2.5e9 | 812 / 812 |
| lih | lbfgs | 2* | 4648 | 0.114 | 3.2e7 / 1.5e11 | 9.9e7 / 2.6e12 [0.3] | 1.5e7 / 8.2e10 | 2.2e6 / 2.2e10 | 2.3e6 / 6e10 [0.3] | 1.6e11 / 4.4e10 / 1.2e11 | 7.2e9 / 7.2e10 / 7.2e11 | ae_x / ae_x | 2.6e6 / 1.9e10 [0] | 4.5e9 | 2.62e3 / 4.02e3 |
| lih | lbfgs | 3 | 8468 | 0.0825 | 6.2e7 / 5.2e11 | 1.8e8 / 5.4e12 [0.3] | 2.1e7 / 1.9e11 | 3.1e6 / 4.3e10 | 3.1e6 / 9.3e10 [0.3] | 3.9e11 / 8.5e10 / 1.9e11 | 1.1e10 / 1.1e11 / 1.1e12 | ae_x / ae_x | 2.6e6 / 2.7e10 [3] | 6.2e9 | 4.43e3 / 7.13e3 |
| lih | lbfgs | 4 | 12574 | 0.094 | 4.8e7 / 6e11 | 2.7e8 / 7.9e12 [0.2] | 1.8e7 / 2.4e11 | 2.7e6 / 4.8e10 | 3.8e6 / 1.1e11 [0.2] | 4.9e11 / 9.7e10 / 2.2e11 | 1.3e10 / 1.3e11 / 1.3e12 | ae_x / ae_x | 2.6e6 / 4.1e10 [1] | 9e9 | 5.64e3 / 8.71e3 |
| lih | lbfgs | 5 | 16764 | 0.0756 | 7.3e7 / 1.2e12 | 4.6e8 / 1.5e13 [0.2] | 2.3e7 / 4e11 | 3.4e6 / 7.4e10 | 4.9e6 / 1.6e11 [0.2] | 8e11 / 1.5e11 / 3.3e11 | 1.9e10 / 1.9e11 / 1.9e12 | ae_x / ae_x | 2.6e6 / 5e10 [3] | 1.1e10 | 7.88e3 / 1.26e4 |
| lih | lbfgs | 6 | 20866 | 0.0493 | 1.7e8 / 3.6e12 | 1.1e9 / 3.9e13 [0.2] | 3.5e7 / 7.6e11 | 5.2e6 / 1.4e11 | 7.5e6 / 2.8e11 [0.2] | 1.5e12 / 2.7e11 / 5.6e11 | 3.3e10 / 3.3e11 / 3.3e12 | ae_x / ae_x | 2.6e6 / 6.1e10 [2] | 1.3e10 | 1.24e4 / 2.1e4 |
| lih | lbfgs_so | 0 | 0 | 0.4 | 2.6e6 / 0 | 3.1e6 / 8.7e10 [0.5] | 4.2e6 / 3.7e9 | 6.4e5 / 3.3e9 | 4.6e5 / 1.1e10 [0.4] | 7.3e9 / 6.7e9 / 2.2e10 | 1.4e9 / 1.4e10 / 1.4e11 | direct_mg / direct_mg | 2.9e6 / 2.4e9 [0] | 7.4e8 | 154 / 154 |
| lih | lbfgs_so | 1 | 468 | 0.303 | 4.6e6 / 2.1e9 | 3.9e7 / 6.6e11 [0.2] | 5.6e6 / 7.6e9 | 8.4e5 / 4.8e9 | 1.4e6 / 2.4e10 [0.2] | 1.5e10 / 9.6e9 / 4.9e10 | 3.1e9 / 3.1e10 / 3.1e11 | direct_mg / direct_mg | 2.6e6 / 8.8e9 [3] | 2.1e9 | 694 / 694 |
| lih | lbfgs_so | 2* | 4038 | 0.0845 | 5.9e7 / 2.4e11 | 3.5e8 / 7.1e12 [0.2] | 2e7 / 1e11 | 3e6 / 2.8e10 | 4.3e6 / 8.8e10 [0.2] | 2e11 / 5.6e10 / 1.8e11 | 1.1e10 / 1.1e11 / 1.1e12 | ae_x / ae_x | 2.6e6 / 1.6e10 [2] | 3.8e9 | 2.79e3 / 4.57e3 |
| lih | lbfgs_so | 3 | 7852 | 0.0851 | 5.8e7 / 4.6e11 | 3.8e8 / 9.1e12 [0.2] | 2e7 / 1.8e11 | 3e6 / 3.9e10 | 4.5e6 / 1.1e11 [0.2] | 3.5e11 / 7.9e10 / 2.2e11 | 1.3e10 / 1.3e11 / 1.3e12 | ae_x / ae_x | 2.6e6 / 2.7e10 [3] | 6e9 | 4.17e3 / 6.67e3 |
| lih | lbfgs_so | 4 | 11788 | 0.0663 | 9.5e7 / 1.1e12 | 2.3e8 / 8.1e12 [0.4] | 2.6e7 / 3.3e11 | 3.9e6 / 6.6e10 | 3.7e6 / 1.2e11 [0.3] | 6.5e11 / 1.3e11 / 2.5e11 | 1.4e10 / 1.4e11 / 1.4e12 | ae_x / ae_x | 2.6e6 / 3.7e10 [2] | 8.3e9 | 6.51e3 / 1.07e4 |
| lih | lbfgs_so | 5 | 15602 | 0.0721 | 8.1e7 / 1.3e12 | 2.7e8 / 9.8e12 [0.3] | 2.4e7 / 3.9e11 | 3.5e6 / 7.4e10 | 3.8e6 / 1.4e11 [0.3] | 7.8e11 / 1.5e11 / 2.8e11 | 1.6e10 / 1.6e11 / 1.6e12 | ae_x / ae_x | 2.6e6 / 4.7e10 [2] | 1e10 | 7.65e3 / 1.23e4 |
| lih | lbfgs_so | 6 | 19454 | 0.0274 | 5.6e8 / 1.1e13 | 3.1e9 / 1.2e14 [0.3] | 6.6e7 / 1.3e12 | 9.3e6 / 2.3e11 | 1.4e7 / 5.1e11 [0.2] | 2.7e12 / 4.6e11 / 1e12 | 6e10 / 6e11 / 6e12 | ae_x / ae_x | 2.6e6 / 5.6e10 [2] | 1.2e10 | 1.81e4 / 3.28e4 |
| lih | sd | 0 | 0 | 0.4 | 2.6e6 / 0 | 3.1e6 / 8.7e10 [0.5] | 4.2e6 / 3.7e9 | 6.4e5 / 3.3e9 | 4.6e5 / 1.1e10 [0.4] | 7.3e9 / 6.7e9 / 2.2e10 | 1.4e9 / 1.4e10 / 1.4e11 | direct_mg / direct_mg | 2.9e6 / 2.4e9 [1] | 7.5e8 | 155 / 155 |
| lih | sd | 1 | 468 | 0.307 | 4.5e6 / 2.1e9 | 3.7e7 / 6.2e11 [0.2] | 5.5e6 / 7.5e9 | 8.3e5 / 4.8e9 | 1.4e6 / 2.4e10 [0.2] | 1.5e10 / 9.5e9 / 4.8e10 | 3.1e9 / 3.1e10 / 3.1e11 | direct_mg / direct_mg | 2.7e6 / 1.3e10 [0] | 3.1e9 | 987 / 987 |
| lih | sd | 2* | 4568 | 0.112 | 3.3e7 / 1.5e11 | 1e8 / 2.6e12 [0.3] | 1.5e7 / 8.3e10 | 2.3e6 / 2.2e10 | 2.3e6 / 6e10 [0.3] | 1.7e11 / 4.5e10 / 1.2e11 | 7.3e9 / 7.3e10 / 7.3e11 | ae_x / ae_x | 2.6e6 / 2.3e10 [1] | 5.3e9 | 2.87e3 / 4.29e3 |
| lih | sd | 3 | 8226 | 0.0982 | 4.4e7 / 3.6e11 | 6.9e7 / 2.2e12 [0.4] | 1.7e7 / 1.6e11 | 2.6e6 / 3.5e10 | 1.9e6 / 6.3e10 [0.4] | 3.2e11 / 7e10 / 1.3e11 | 7.4e9 / 7.4e10 / 7.4e11 | ae_x / ae_x | 2.6e6 / 3.3e10 [2] | 7.3e9 | 4.29e3 / 6.51e3 |
| lih | sd | 4 | 11783 | 0.0855 | 5.7e7 / 6.8e11 | 9.1e7 / 3.3e12 [0.4] | 2e7 / 2.5e11 | 3e6 / 5.1e10 | 2.2e6 / 8e10 [0.4] | 5.1e11 / 1e11 / 1.6e11 | 9.4e9 / 9.4e10 / 9.4e11 | ae_x / ae_x | 2.6e6 / 4.1e10 [1] | 9.2e9 | 5.81e3 / 9.04e3 |
| lih | sd | 5 | 14521 | 0.0807 | 6.4e7 / 9.4e11 | 1.1e8 / 4.2e12 [0.4] | 2.1e7 / 3.3e11 | 3.2e6 / 6.3e10 | 2.4e6 / 9.4e10 [0.4] | 6.5e11 / 1.3e11 / 1.9e11 | 1.1e10 / 1.1e11 / 1.1e12 | ae_x / ae_x | 2.6e6 / 4.9e10 [0] | 1.1e10 | 7.05e3 / 1.1e4 |
| lih | sd | 6 | 17221 | 0.072 | 8.1e7 / 1.4e12 | 1.4e8 / 5.7e12 [0.4] | 2.4e7 / 4.3e11 | 3.6e6 / 8e10 | 2.7e6 / 1.1e11 [0.4] | 8.6e11 / 1.6e11 / 2.3e11 | 1.3e10 / 1.3e11 / 1.3e12 | ae_x / ae_x | 2.6e6 / 5.5e10 [2] | 1.2e10 | 8.53e3 / 1.36e4 |
| h6s | lbfgs | 0 | 0 | 1.15 | 1.2e5 / 0 | 1e5 / 3.8e9 [1.2] | 1.4e6 / 1.5e9 | 7.5e5 / 7.9e9 | 1.2e5 / 4.1e9 [1] | 3e9 / 1.6e10 / 8.3e9 | 4.1e8 / 4.1e9 / 4.1e10 | direct_mg / direct_mg | 3.9e6 / 5.4e9 [0] | 5.6e8 | 553 / 553 |
| h6s | lbfgs | 1 | 969 | 0.812 | 2.5e5 / 2.4e8 | 3.6e5 / 1.1e10 [0.8] | 1.9e6 / 4.1e9 | 1.1e6 / 1.2e10 | 2.2e5 / 6.6e9 [0.8] | 8.2e9 / 2.5e10 / 1.3e10 | 6.7e8 / 6.7e9 / 6.7e10 | direct_mg / direct_mg | 3.3e6 / 3.2e10 [0] | 2.8e9 | 3.34e3 / 3.34e3 |
| h6s | lbfgs | 2 | 9495 | 0.341 | 1.4e6 / 1.3e10 | 2.1e6 / 8.3e10 [0.8] | 4.6e6 / 4.9e10 | 2.5e6 / 5.1e10 | 5.3e5 / 2.1e10 [0.8] | 9.7e10 / 1e11 / 4.1e10 | 2e9 / 2e10 / 2e11 | direct_mg / direct_mg | 3.2e6 / 5.9e10 [0] | 5e9 | 7.43e3 / 7.43e3 |
| h6s | lbfgs | 3 | 18021 | 0.28 | 2.1e6 / 3.7e10 | 2.1e6 / 1.1e11 [1] | 5.6e6 / 1.1e11 | 3.1e6 / 8.8e10 | 5.3e5 / 2.8e10 [1] | 2.1e11 / 1.8e11 / 5.5e10 | 2.6e9 / 2.6e10 / 2.6e11 | ae_aux / direct_mg | 3.2e6 / 8.5e10 [0] | 7.3e9 | 1.16e4 / 1.26e4 |
| h6s | lbfgs | 4 | 26547 | 0.188 | 4.6e6 / 1.2e11 | 5.1e6 / 3.1e11 [1] | 8.3e6 / 2.3e11 | 4.6e6 / 1.7e11 | 8.1e5 / 4.9e10 [1] | 4.6e11 / 3.4e11 / 9.9e10 | 4.6e9 / 4.6e10 / 4.6e11 | ae_aux / ae_aux | 3.1e6 / 1.1e11 [0] | 9.5e9 | 1.64e4 / 2.15e4 |
| h6s | lbfgs | 5 | 35029 | 0.213 | 3.6e6 / 1.2e11 | 3.1e6 / 2.2e11 [1.2] | 7.3e6 / 2.6e11 | 4e6 / 1.8e11 | 6.7e5 / 4.6e10 [1] | 5.3e11 / 3.7e11 / 9.2e10 | 4.3e9 / 4.3e10 / 4.3e11 | ae_aux / ae_aux | 3e6 / 1.3e11 [0] | 1.2e10 | 1.83e4 / 2.3e4 |
| h6s | lbfgs | 6 | 43511 | 0.111 | 1.3e7 / 5.8e11 | 2.2e7 / 1.6e12 [0.8] | 1.4e7 / 6.3e11 | 7.8e6 / 4.2e11 | 1.7e6 / 1.2e11 [0.8] | 1.3e12 / 8.4e11 / 2.5e11 | 1.1e10 / 1.1e11 / 1.1e12 | ae_aux / ae_aux | 3.1e6 / 1.6e11 [0] | 1.4e10 | 2.9e4 / 4.17e4 |
| h6s | lbfgs | 7 | 52037 | 0.094 | 1.8e7 / 9.5e11 | 2.9e7 / 2.3e12 [0.8] | 1.7e7 / 8.8e11 | 9.2e6 / 5.8e11 | 1.9e6 / 1.6e11 [0.8] | 1.8e12 / 1.2e12 / 3.2e11 | 1.5e10 / 1.5e11 / 1.5e12 | ae_aux / ae_aux | 3.1e6 / 1.9e11 [0] | 1.6e10 | 3.53e4 / 5.15e4 |
| h6s | lbfgs | 8 | 60519 | 0.074 | 3e7 / 1.8e12 | 3.8e7 / 3.6e12 [1] | 2.1e7 / 1.3e12 | 1.2e7 / 8.3e11 | 2.2e6 / 2.1e11 [1] | 2.6e12 / 1.7e12 / 4.2e11 | 1.9e10 / 1.9e11 / 1.9e12 | ae_aux / ae_aux | 3e6 / 2.1e11 [1] | 1.8e10 | 4.29e4 / 6.44e4 |
| h6s | lbfgs | 9 | 69045 | 0.0421 | 9.1e7 / 6.3e12 | 1.6e8 / 1.6e13 [0.8] | 3.7e7 / 2.6e12 | 2.1e7 / 1.6e12 | 4.5e6 / 4.5e11 [0.8] | 5.2e12 / 3.3e12 / 9e11 | 4.1e10 / 4.1e11 / 4.1e12 | ae_aux / ae_aux | 3e6 / 2.3e11 [0] | 2e10 | 6.99e4 / 1.16e5 |
| h6s | lbfgs | 10 | 77571 | 0.0425 | 9e7 / 7e12 | 1.5e8 / 1.7e13 [0.8] | 3.7e7 / 2.9e12 | 2e7 / 1.8e12 | 4.5e6 / 4.8e11 [0.8] | 5.8e12 / 3.6e12 / 9.6e11 | 4.4e10 / 4.4e11 / 4.4e12 | ae_aux / ae_aux | 3e6 / 2.6e11 [0] | 2.3e10 | 7.64e4 / 1.26e5 |
| h6s | lbfgs | 11 | 86097 | 0.039 | 1.1e8 / 9.2e12 | 2.3e8 / 2.6e13 [0.7] | 4e7 / 3.5e12 | 2.2e7 / 2.1e12 | 5.4e6 / 6.2e11 [0.7] | 7e12 / 4.3e12 / 1.2e12 | 5.6e10 / 5.6e11 / 5.6e12 | ae_aux / ae_aux | 3.1e6 / 2.9e11 [0] | 2.5e10 | 9.3e4 / 1.56e5 |
| h6s | lbfgs | 12* | 94623 | 0.0393 | 1e8 / 9.9e12 | 1.6e8 / 2e13 [0.8] | 4e7 / 3.8e12 | 2.2e7 / 2.3e12 | 4.6e6 / 5.7e11 [0.8] | 7.6e12 / 4.6e12 / 1.1e12 | 5.1e10 / 5.1e11 / 5.1e12 | ae_aux / ae_aux | 3e6 / 3.1e11 [0] | 2.7e10 | 9.09e4 / 1.5e5 |
| h6s | lbfgs | 13 | 103149 | 0.038 | 1.1e8 / 1.2e13 | 2e8 / 2.6e13 [0.8] | 4.1e7 / 4.3e12 | 2.3e7 / 2.6e12 | 5.1e6 / 6.8e11 [0.8] | 8.6e12 / 5.2e12 / 1.4e12 | 6.1e10 / 6.1e11 / 6.1e12 | ae_aux / ae_aux | 3e6 / 3.4e11 [0] | 3e10 | 1.04e5 / 1.74e5 |
| h6s | lbfgs | 14 | 111631 | 0.0302 | 1.8e8 / 2e13 | 3.6e8 / 5.1e13 [0.8] | 5.2e7 / 5.8e12 | 2.9e7 / 3.5e12 | 6.9e6 / 9.7e11 [0.8] | 1.2e13 / 7e12 / 1.9e12 | 8.7e10 / 8.7e11 / 8.7e12 | ae_aux / ae_aux | 3e6 / 3.6e11 [0] | 3.2e10 | 1.37e5 / 2.37e5 |
| h6s | lbfgs | 15 | 120157 | 0.0318 | 1.6e8 / 1.9e13 | 2.1e8 / 3.3e13 [1] | 4.9e7 / 6e12 | 2.7e7 / 3.6e12 | 5.3e6 / 8.2e11 [1] | 1.2e13 / 7.1e12 / 1.6e12 | 7.3e10 / 7.3e11 / 7.3e12 | ae_aux / ae_aux | 3e6 / 3.9e11 [0] | 3.4e10 | 1.24e5 / 2.07e5 |
| h6s | lbfgs | 16 | 128683 | 0.0258 | 2.4e8 / 3.1e13 | 3.9e8 / 6.2e13 [0.8] | 6e7 / 7.9e12 | 3.4e7 / 4.7e12 | 7.1e6 / 1.1e12 [0.8] | 1.6e13 / 9.3e12 / 2.3e12 | 1e11 / 1e12 / 1e13 | ae_aux / ae_aux | 3e6 / 4.1e11 [0] | 3.6e10 | 1.58e5 / 2.74e5 |
| h6s | lbfgs_so | 0 | 0 | 1.15 | 1.2e5 / 0 | 1e5 / 3.8e9 [1.2] | 1.4e6 / 1.5e9 | 7.5e5 / 7.9e9 | 1.2e5 / 4.1e9 [1] | 3e9 / 1.6e10 / 8.3e9 | 4.1e8 / 4.1e9 / 4.1e10 | direct_mg / direct_mg | 3.7e6 / 5e9 [0] | 5.6e8 | 516 / 516 |
| h6s | lbfgs_so | 1 | 969 | 0.543 | 5.5e5 / 5.3e8 | 6.3e5 / 2.2e10 [1] | 2.9e6 / 6.1e9 | 1.6e6 / 1.8e10 | 3.2e5 / 9.7e9 [0.8] | 1.2e10 / 3.7e10 / 1.9e10 | 9.8e8 / 9.8e9 / 9.8e10 | direct_mg / direct_mg | 3.2e6 / 3.1e10 [0] | 2.8e9 | 3.25e3 / 3.25e3 |
| h6s | lbfgs_so | 2 | 9495 | 0.311 | 1.7e6 / 1.6e10 | 2.1e6 / 9.3e10 [1] | 5e6 / 5.3e10 | 2.8e6 / 5.6e10 | 5.6e5 / 2.2e10 [0.8] | 1.1e11 / 1.1e11 / 4.4e10 | 2.2e9 / 2.2e10 / 2.2e11 | direct_mg / direct_mg | 3.1e6 / 5.7e10 [0] | 5e9 | 7.53e3 / 7.53e3 |
| h6s | lbfgs_so | 3 | 17977 | 0.24 | 2.8e6 / 5.1e10 | 3.4e6 / 1.7e11 [1] | 6.5e6 / 1.2e11 | 3.6e6 / 1e11 | 6.6e5 / 3.4e10 [1] | 2.5e11 / 2.1e11 / 6.9e10 | 3.3e9 / 3.3e10 / 3.3e11 | ae_aux / direct_mg | 3.1e6 / 8.3e10 [0] | 7.2e9 | 1.21e4 / 1.38e4 |
| h6s | lbfgs_so | 4 | 26503 | 0.184 | 4.8e6 / 1.3e11 | 5.9e6 / 3.5e11 [1] | 8.5e6 / 2.3e11 | 4.7e6 / 1.7e11 | 8.8e5 / 5.3e10 [1] | 4.7e11 / 3.5e11 / 1.1e11 | 5e9 / 5e10 / 5e11 | ae_aux / ae_aux | 3.1e6 / 1.1e11 [0] | 9.4e9 | 1.65e4 / 2.2e4 |
| h6s | lbfgs_so | 5 | 35029 | 0.149 | 7.3e6 / 2.6e11 | 7.4e6 / 5.1e11 [1] | 1e7 / 3.8e11 | 5.8e6 / 2.7e11 | 9.9e5 / 6.8e10 [1] | 7.6e11 / 5.3e11 / 1.4e11 | 6.3e9 / 6.3e10 / 6.3e11 | ae_aux / ae_aux | 3e6 / 1.3e11 [0] | 1.2e10 | 2.06e4 / 2.75e4 |
| h6s | lbfgs_so | 6 | 43555 | 0.077 | 2.7e7 / 1.2e12 | 3.2e7 / 2.4e12 [1] | 2e7 / 9e11 | 1.1e7 / 6.1e11 | 2e6 / 1.6e11 [1] | 1.8e12 / 1.2e12 / 3.1e11 | 1.5e10 / 1.5e11 / 1.5e12 | ae_aux / ae_aux | 3.1e6 / 1.6e11 [1] | 1.4e10 | 3.22e4 / 4.84e4 |
| h6s | lbfgs_so | 7 | 52081 | 0.0476 | 7.2e7 / 3.7e12 | 1.1e8 / 9.4e12 [0.8] | 3.3e7 / 1.7e12 | 1.8e7 / 1.1e12 | 3.9e6 / 3.2e11 [0.8] | 3.5e12 / 2.3e12 / 6.3e11 | 2.9e10 / 2.9e11 / 2.9e12 | ae_aux / ae_aux | 3e6 / 1.8e11 [0] | 1.6e10 | 5.13e4 / 8.39e4 |
| h6s | lbfgs_so | 8 | 60607 | 0.0548 | 5.4e7 / 3.3e12 | 7.7e7 / 7.2e12 [1] | 2.8e7 / 1.8e12 | 1.6e7 / 1.1e12 | 3.3e6 / 3e11 [0.8] | 3.5e12 / 2.2e12 / 5.9e11 | 2.7e10 / 2.7e11 / 2.7e12 | ae_aux / ae_aux | 3e6 / 2.1e11 [0] | 1.8e10 | 5.18e4 / 8.21e4 |
| h6s | lbfgs_so | 9 | 69133 | 0.0508 | 6.3e7 / 4.3e12 | 1.1e8 / 1e13 [0.8] | 3.1e7 / 2.2e12 | 1.7e7 / 1.4e12 | 3.7e6 / 3.7e11 [0.8] | 4.3e12 / 2.7e12 / 7.3e11 | 3.3e10 / 3.3e11 / 3.3e12 | ae_aux / ae_aux | 3.1e6 / 2.4e11 [1] | 2e10 | 6.19e4 / 9.96e4 |
| h6s | lbfgs_so | 10* | 77593 | 0.0363 | 1.2e8 / 9.5e12 | 2.4e8 / 2.5e13 [0.8] | 4.3e7 / 3.4e12 | 2.4e7 / 2.1e12 | 5.5e6 / 6e11 [0.8] | 6.8e12 / 4.2e12 / 1.2e12 | 5.4e10 / 5.4e11 / 5.4e12 | ae_aux / ae_aux | 3e6 / 2.6e11 [0] | 2.3e10 | 8.81e4 / 1.49e5 |
| h6s | lbfgs_so | 11 | 86119 | 0.0384 | 1.1e8 / 9.5e12 | 2.7e8 / 3e13 [0.7] | 4.1e7 / 3.5e12 | 2.2e7 / 2.2e12 | 5.9e6 / 6.7e11 [0.7] | 7.1e12 / 4.3e12 / 1.3e12 | 6.1e10 / 6.1e11 / 6.1e12 | ae_aux / ae_aux | 3e6 / 2.9e11 [0] | 2.5e10 | 9.85e4 / 1.68e5 |
| h6s | lbfgs_so | 12 | 94645 | 0.0294 | 1.9e8 / 1.8e13 | 4.3e8 / 5.3e13 [0.7] | 5.3e7 / 5.1e12 | 2.9e7 / 3.1e12 | 7.5e6 / 9.2e11 [0.7] | 1e13 / 6.2e12 / 1.8e12 | 8.3e10 / 8.3e11 / 8.3e12 | ae_aux / ae_aux | 3e6 / 3.1e11 [0] | 2.7e10 | 1.27e5 / 2.21e5 |
| h6s | lbfgs_so | 13 | 103171 | 0.0225 | 3.2e8 / 3.3e13 | 6.7e8 / 8.9e13 [0.8] | 6.9e7 / 7.2e12 | 3.8e7 / 4.4e12 | 9.4e6 / 1.2e12 [0.7] | 1.4e13 / 8.7e12 / 2.5e12 | 1.1e11 / 1.1e12 / 1.1e13 | ae_aux / ae_aux | 3e6 / 3.4e11 [1] | 3e10 | 1.62e5 / 2.88e5 |
| h6s | lbfgs_so | 14 | 111697 | 0.0229 | 3.1e8 / 3.5e13 | 5.5e8 / 7.7e13 [0.8] | 6.8e7 / 7.7e12 | 3.8e7 / 4.6e12 | 8.5e6 / 1.2e12 [0.8] | 1.5e13 / 9.2e12 / 2.4e12 | 1.1e11 / 1.1e12 / 1.1e13 | ae_aux / ae_aux | 3e6 / 3.6e11 [0] | 3.2e10 | 1.6e5 / 2.83e5 |
| n2s | lbfgs | 0 | 0 | 3.2 | 3.1e3 / 0 | 9.6e3 / 2.2e8 [0.6] | 5.3e4 / 4.6e7 | 9.5e3 / 2.5e7 | 6.9e3 / 1.6e8 [0.6] | 9.2e7 / 5e7 / 3.2e8 | 3.3e7 / 3.3e8 / 3.3e9 | direct_mg / direct_mg | 2e6 / 1.8e9 [0] | 4.6e8 | 188 / 188 |
| n2s | lbfgs | 1 | 486 | 0.475 | 1.4e5 / 6.8e7 | 1.6e5 / 5e9 [1] | 3.5e5 / 4.9e8 | 6.4e4 / 2e8 | 3.1e4 / 8.4e8 [0.8] | 9.8e8 / 4e8 / 1.7e9 | 1.7e8 / 1.7e9 / 1.7e10 | direct_mg / direct_mg | 2.1e6 / 3.5e9 [1] | 7.3e8 | 380 / 380 |
| n2s | lbfgs | 2 | 1531 | 0.491 | 1.3e5 / 2e8 | 1.5e5 / 4.8e9 [1] | 3.4e5 / 8.3e8 | 6.2e4 / 2.6e8 | 3e4 / 8.4e8 [0.8] | 1.7e9 / 5.2e8 / 1.7e9 | 1.7e8 / 1.7e9 / 1.7e10 | direct_mg / direct_mg | 2.6e6 / 7.3e9 [1] | 1.1e9 | 788 / 788 |
| n2s | lbfgs | 3 | 2576 | 0.529 | 1.1e5 / 2.9e8 | 3.2e5 / 8e9 [0.6] | 3.2e5 / 1.1e9 | 5.8e4 / 3e8 | 4e4 / 1e9 [0.6] | 2.2e9 / 6e8 / 2e9 | 2.1e8 / 2.1e9 / 2.1e10 | direct_mg / direct_mg | 2.2e6 / 8.7e9 [0] | 1.5e9 | 948 / 948 |
| n2s | lbfgs | 4 | 3621 | 0.397 | 2e5 / 7.3e8 | 5.7e5 / 1.5e10 [0.6] | 4.2e5 / 1.9e9 | 7.7e4 / 4.8e8 | 5.3e4 / 1.4e9 [0.6] | 3.8e9 / 9.7e8 / 2.8e9 | 2.9e8 / 2.9e9 / 2.9e10 | ae_x / direct_mg | 1.8e6 / 9e9 [0] | 1.9e9 | 1e3 / 1.03e3 |
| n2s | lbfgs | 5 | 4666 | 0.223 | 6.4e5 / 3e9 | 7.7e5 / 2.8e10 [1] | 7.5e5 / 4.2e9 | 1.4e5 / 1e9 | 6.7e4 / 2.1e9 [0.8] | 8.4e9 / 2e9 / 4.2e9 | 4.3e8 / 4.3e9 / 4.3e10 | ae_x / ae_x | 1.7e6 / 1e10 [0] | 2.2e9 | 1.2e3 / 1.31e3 |
| n2s | lbfgs | 6 | 5711 | 0.21 | 7.2e5 / 4.1e9 | 1.5e6 / 4.7e10 [0.7] | 8e5 / 5.3e9 | 1.4e5 / 1.2e9 | 8.7e4 / 2.7e9 [0.7] | 1.1e10 / 2.4e9 / 5.3e9 | 5.5e8 / 5.5e9 / 5.5e10 | ae_x / ae_x | 1.6e6 / 1.1e10 [0] | 2.6e9 | 1.31e3 / 1.44e3 |
| n2s | lbfgs | 7 | 6756 | 0.0814 | 4.8e6 / 3.2e10 | 1.5e7 / 4.3e11 [0.6] | 2.1e6 / 1.6e10 | 3.7e5 / 3.5e9 | 2.7e5 / 8e9 [0.6] | 3.2e10 / 7.1e9 / 1.6e10 | 1.6e9 / 1.6e10 / 1.6e11 | ae_x / ae_x | 1.5e6 / 1.2e10 [0] | 2.9e9 | 1.64e3 / 2.02e3 |
| n2s | lbfgs | 8 | 7801 | 0.0456 | 1.5e7 / 1.2e11 | 3.5e7 / 1.1e12 [0.7] | 3.7e6 / 3.2e10 | 6.7e5 / 7e9 | 4.2e5 / 1.4e10 [0.7] | 6.4e10 / 1.4e10 / 2.7e10 | 2.8e9 / 2.8e10 / 2.8e11 | ae_x / ae_x | 1.4e6 / 1.3e10 [1] | 3.2e9 | 2.12e3 / 2.86e3 |
| n2s | lbfgs | 9 | 8846 | 0.0467 | 1.5e7 / 1.3e11 | 4.4e7 / 1.4e12 [0.6] | 3.6e6 / 3.5e10 | 6.5e5 / 7.5e9 | 4.7e5 / 1.5e10 [0.6] | 7e10 / 1.5e10 / 3e10 | 3.1e9 / 3.1e10 / 3.1e11 | ae_x / ae_x | 1.5e6 / 1.5e10 [1] | 3.6e9 | 2.36e3 / 3.15e3 |
| n2s | lbfgs | 10 | 9891 | 0.0549 | 1.1e7 / 1e11 | 6.3e7 / 1.8e12 [0.4] | 3.1e6 / 3.3e10 | 5.6e5 / 7e9 | 5.6e5 / 1.6e10 [0.4] | 6.6e10 / 1.4e10 / 3.1e10 | 3.2e9 / 3.2e10 / 3.2e11 | ae_x / ae_x | 1.4e6 / 1.5e10 [0] | 3.9e9 | 2.36e3 / 3.09e3 |
| n2s | lbfgs | 11 | 10936 | 0.0309 | 3.3e7 / 3.6e11 | 1.6e8 / 5.3e12 [0.5] | 5.5e6 / 6.5e10 | 9.9e5 / 1.3e10 | 9.8e5 / 2.8e10 [0.4] | 1.3e11 / 2.7e10 / 5.7e10 | 5.8e9 / 5.8e10 / 5.8e11 | ae_x / ae_x | 1.3e6 / 1.6e10 [0] | 4.3e9 | 3.13e3 / 4.55e3 |
| n2s | lbfgs | 12 | 11981 | 0.0375 | 2.2e7 / 2.7e11 | 4.2e7 / 1.6e12 [0.8] | 4.5e6 / 5.8e10 | 8.1e5 / 1.2e10 | 4.8e5 / 1.8e10 [0.7] | 1.2e11 / 2.4e10 / 3.5e10 | 3.6e9 / 3.6e10 / 3.6e11 | ae_x / ae_x | 1.3e6 / 1.7e10 [0] | 4.6e9 | 3.04e3 / 4.29e3 |
| n2s | lbfgs | 13 | 13026 | 0.0235 | 5.7e7 / 7.5e11 | 2.4e8 / 8.4e12 [0.5] | 7.2e6 / 1e11 | 1.3e6 / 2e10 | 1.1e6 / 3.8e10 [0.5] | 2e11 / 4.1e10 / 7.6e10 | 7.8e9 / 7.8e10 / 7.8e11 | ae_x / ae_x | 1.3e6 / 1.9e10 [2] | 5e9 | 4.1e3 / 6.25e3 |
| n2s | lbfgs | 14 | 14071 | 0.0312 | 3.2e7 / 4.6e11 | 1.4e8 / 5e12 [0.5] | 5.4e6 / 8.1e10 | 9.8e5 / 1.6e10 | 8.4e5 / 3e10 [0.5] | 1.6e11 / 3.3e10 / 6e10 | 6.1e9 / 6.1e10 / 6.1e11 | ae_x / ae_x | 1.2e6 / 1.9e10 [1] | 5.3e9 | 3.69e3 / 5.41e3 |
| n2s | lbfgs | 15 | 15116 | 0.065 | 7.5e6 / 1.1e11 | 2.1e7 / 7.9e11 [0.6] | 2.6e6 / 4.1e10 | 4.7e5 / 8.3e9 | 3.2e5 / 1.2e10 [0.6] | 8.3e10 / 1.7e10 / 2.5e10 | 2.5e9 / 2.5e10 / 2.5e11 | ae_x / ae_x | 1.2e6 / 1.9e10 [0] | 5.7e9 | 2.9e3 / 3.78e3 |
| n2s | lbfgs | 16 | 16161 | 0.0287 | 3.8e7 / 6.2e11 | 3.4e8 / 1.2e13 [0.4] | 5.9e6 / 1e11 | 1.1e6 / 2e10 | 1.3e6 / 4.5e10 [0.4] | 2e11 / 4e10 / 9e10 | 9.2e9 / 9.2e10 / 9.2e11 | ae_x / ae_x | 1.2e6 / 2.1e10 [0] | 6e9 | 4.28e3 / 6.38e3 |
| n2s | lbfgs | 17 | 17168 | 0.0141 | 1.6e8 / 2.7e12 | 2.2e9 / 7.3e13 [0.3] | 1.2e7 / 2.2e11 | 2.2e6 / 4.3e10 | 3.3e6 / 1.1e11 [0.3] | 4.4e11 / 8.6e10 / 2.2e11 | 2.3e10 / 2.3e11 / 2.3e12 | ae_x / ae_x | 1.2e6 / 2.1e10 [1] | 6.3e9 | 6.78e3 / 1.13e4 |
| n2s | lbfgs | 18 | 18213 | 0.0068 | 6.9e8 / 1.2e13 | 2.4e10 / 7.6e14 [0.2] | 2.6e7 / 4.9e11 | 4.5e6 / 9.4e10 | 1.1e7 / 3.5e11 [0.2] | 9.9e11 / 1.9e11 / 6.9e11 | 7.1e10 / 7.1e11 / 7.1e12 | ae_x / ae_x | 1.2e6 / 2.3e10 [2] | 6.7e9 | 1.23e4 / 2.21e4 |
| n2s | lbfgs | 19 | 19258 | 0.00512 | 1.2e9 / 2.3e13 | 1.2e11 / 3.5e15 [0.1] | 3.5e7 / 7.1e11 | 6e6 / 1.3e11 | 2.5e7 / 7.2e11 [0.1] | 1.4e12 / 2.6e11 / 1.4e12 | 1.5e11 / 1.5e12 / 1.5e13 | ae_x / ae_x | 1.2e6 / 2.4e10 [0] | 7.1e9 | 1.63e4 / 3e4 |
| n2s | lbfgs_so | 0 | 0 | 3.2 | 3.1e3 / 0 | 9.6e3 / 2.2e8 [0.6] | 5.3e4 / 4.6e7 | 9.5e3 / 2.5e7 | 6.9e3 / 1.6e8 [0.6] | 9.2e7 / 5e7 / 3.2e8 | 3.3e7 / 3.3e8 / 3.3e9 | direct_mg / direct_mg | 2e6 / 1.8e9 [0] | 4.6e8 | 186 / 186 |
| n2s | lbfgs_so | 1 | 486 | 0.459 | 1.5e5 / 7.3e7 | 3.7e5 / 9.2e9 [0.7] | 3.7e5 / 5.1e8 | 6.6e4 / 2.1e8 | 4.4e4 / 1e9 [0.6] | 1e9 / 4.2e8 / 2.1e9 | 2.2e8 / 2.2e9 / 2.2e10 | direct_mg / direct_mg | 2.1e6 / 3.6e9 [0] | 7.3e8 | 385 / 385 |
| n2s | lbfgs_so | 2 | 1455 | 0.493 | 1.3e5 / 1.9e8 | 2.3e5 / 6.5e9 [0.8] | 3.4e5 / 8e8 | 6.2e4 / 2.5e8 | 3.5e4 / 9.3e8 [0.7] | 1.6e9 / 5.1e8 / 1.9e9 | 1.9e8 / 1.9e9 / 1.9e10 | direct_mg / direct_mg | 2.3e6 / 6.5e9 [0] | 1.1e9 | 708 / 708 |
| n2s | lbfgs_so | 3 | 2500 | 0.318 | 3.1e5 / 7.8e8 | 8.5e5 / 2.2e10 [0.6] | 5.3e5 / 1.8e9 | 9.6e4 / 5e8 | 6.5e4 / 1.7e9 [0.6] | 3.6e9 / 9.9e8 / 3.3e9 | 3.4e8 / 3.4e9 / 3.4e10 | ae_x / direct_mg | 1.5e6 / 5.9e9 [0] | 1.5e9 | 673 / 704 |
| n2s | lbfgs_so | 4 | 3545 | 0.182 | 9.5e5 / 3.4e9 | 7.1e5 / 2.7e10 [1.2] | 9.2e5 / 4.1e9 | 1.7e5 / 1e9 | 6.4e4 / 2.2e9 [1] | 8.2e9 / 2.1e9 / 4.5e9 | 4.6e8 / 4.6e9 / 4.6e10 | ae_x / ae_x | 1.5e6 / 7.7e9 [0] | 1.8e9 | 923 / 1.03e3 |
| n2s | lbfgs_so | 5 | 4590 | 0.092 | 3.7e6 / 1.7e10 | 8.4e6 / 2.5e11 [0.7] | 1.8e6 / 1e10 | 3.3e5 / 2.4e9 | 2e5 / 6e9 [0.7] | 2e10 / 4.8e9 / 1.2e10 | 1.2e9 / 1.2e10 / 1.2e11 | ae_x / ae_x | 1.5e6 / 8.9e9 [1] | 2.2e9 | 1.2e3 / 1.45e3 |
| n2s | lbfgs_so | 6 | 5635 | 0.076 | 5.5e6 / 3.1e10 | 1.7e7 / 4.7e11 [0.6] | 2.2e6 / 1.4e10 | 4e5 / 3.3e9 | 2.9e5 / 8.2e9 [0.6] | 2.9e10 / 6.7e9 / 1.6e10 | 1.7e9 / 1.7e10 / 1.7e11 | ae_x / ae_x | 1.6e6 / 1.1e10 [0] | 2.5e9 | 1.52e3 / 1.87e3 |
| n2s | lbfgs_so | 7 | 6604 | 0.0519 | 1.2e7 / 7.8e10 | 7e7 / 1.7e12 [0.4] | 3.2e6 / 2.4e10 | 5.9e5 / 5.4e9 | 5.9e5 / 1.5e10 [0.4] | 4.9e10 / 1.1e10 / 2.9e10 | 3e9 / 3e10 / 3e11 | ae_x / ae_x | 1.6e6 / 1.3e10 [0] | 2.9e9 | 1.9e3 / 2.47e3 |
| n2s | lbfgs_so | 8 | 7649 | 0.036 | 2.4e7 / 1.9e11 | 1.9e8 / 4.8e12 [0.4] | 4.7e6 / 4e10 | 8.5e5 / 8.7e9 | 9.6e5 / 2.5e10 [0.4] | 8e10 / 1.7e10 / 5e10 | 5.1e9 / 5.1e10 / 5.1e11 | ae_x / ae_x | 1.6e6 / 1.4e10 [2] | 3.2e9 | 2.39e3 / 3.31e3 |
| n2s | lbfgs_so | 9 | 8694 | 0.0385 | 2.1e7 / 1.9e11 | 1.6e8 / 4.2e12 [0.4] | 4.4e6 / 4.2e10 | 7.9e5 / 9e9 | 8.8e5 / 2.4e10 [0.4] | 8.4e10 / 1.8e10 / 4.7e10 | 4.9e9 / 4.9e10 / 4.9e11 | ae_x / ae_x | 1.5e6 / 1.5e10 [2] | 3.5e9 | 2.48e3 / 3.43e3 |
| n2s | lbfgs_so | 10 | 9739 | 0.0557 | 1e7 / 9.9e10 | 6e7 / 1.7e12 [0.4] | 3e6 / 3.2e10 | 5.5e5 / 6.8e9 | 5.4e5 / 1.5e10 [0.4] | 6.4e10 / 1.4e10 / 3e10 | 3.1e9 / 3.1e10 / 3.1e11 | ae_x / ae_x | 1.4e6 / 1.5e10 [0] | 3.9e9 | 2.32e3 / 3.03e3 |
| n2s | lbfgs_so | 11 | 10784 | 0.0326 | 3e7 / 3.2e11 | 1.9e8 / 5.4e12 [0.4] | 5.2e6 / 6e10 | 9.3e5 / 1.3e10 | 9.6e5 / 2.8e10 [0.4] | 1.2e11 / 2.5e10 / 5.6e10 | 5.7e9 / 5.7e10 / 5.7e11 | ae_x / ae_x | 1.3e6 / 1.6e10 [1] | 4.2e9 | 3.01e3 / 4.33e3 |
| n2s | lbfgs_so | 12 | 11829 | 0.0398 | 2e7 / 2.4e11 | 8.7e7 / 2.9e12 [0.5] | 4.2e6 / 5.4e10 | 7.7e5 / 1.1e10 | 6.6e5 / 2.2e10 [0.5] | 1.1e11 / 2.2e10 / 4.4e10 | 4.5e9 / 4.5e10 / 4.5e11 | ae_x / ae_x | 1.3e6 / 1.8e10 [0] | 4.6e9 | 3.02e3 / 4.19e3 |
| n2s | lbfgs_so | 13 | 12874 | 0.0559 | 1e7 / 1.3e11 | 3.9e7 / 1.4e12 [0.6] | 3e6 / 4.1e10 | 5.5e5 / 8.5e9 | 4.5e5 / 1.5e10 [0.5] | 8.3e10 / 1.7e10 / 3.1e10 | 3.2e9 / 3.2e10 / 3.2e11 | ae_x / ae_x | 1.3e6 / 1.8e10 [1] | 4.9e9 | 2.78e3 / 3.67e3 |
| n2s | lbfgs_so | 14 | 13919 | 0.0375 | 2.2e7 / 3.1e11 | 9.6e7 / 3.4e12 [0.5] | 4.5e6 / 6.6e10 | 8.1e5 / 1.3e10 | 6.9e5 / 2.4e10 [0.5] | 1.3e11 / 2.7e10 / 4.9e10 | 5e9 / 5e10 / 5e11 | ae_x / ae_x | 1.3e6 / 1.9e10 [0] | 5.3e9 | 3.44e3 / 4.86e3 |
| n2s | lbfgs_so | 15 | 14964 | 0.0121 | 2.2e8 / 3.2e12 | 3.5e9 / 1.1e14 [0.3] | 1.4e7 / 2.2e11 | 2.5e6 / 4.4e10 | 4.2e6 / 1.3e11 [0.3] | 4.5e11 / 8.9e10 / 2.6e11 | 2.7e10 / 2.7e11 / 2.7e12 | ae_x / ae_x | 1.3e6 / 2.1e10 [0] | 5.6e9 | 6.84e3 / 1.15e4 |
| n2s | lbfgs_so | 16 | 16009 | 0.0102 | 3.1e8 / 4.9e12 | 8.2e9 / 2.4e14 [0.2] | 1.7e7 / 2.9e11 | 3e6 / 5.6e10 | 6.4e6 / 1.9e11 [0.2] | 5.7e11 / 1.1e11 / 3.7e11 | 3.8e10 / 3.8e11 / 3.8e12 | ae_x / ae_x | 1.3e6 / 2.2e10 [1] | 6e9 | 8.21e3 / 1.41e4 |
| n2s | lbfgs_so | 17 | 17054 | 0.0154 | 1.3e8 / 2.3e12 | 2.3e9 / 7.5e13 [0.3] | 1.1e7 / 2e11 | 2e6 / 3.9e10 | 3.3e6 / 1.1e11 [0.3] | 3.9e11 / 7.8e10 / 2.2e11 | 2.3e10 / 2.3e11 / 2.3e12 | ae_x / ae_x | 1.3e6 / 2.3e10 [0] | 6.3e9 | 6.53e3 / 1.06e4 |
| n2s | lbfgs_so | 18 | 18099 | 0.0183 | 9.4e7 / 1.7e12 | 1.2e9 / 4e13 [0.3] | 9.2e6 / 1.7e11 | 1.7e6 / 3.5e10 | 2.4e6 / 8.3e10 [0.3] | 3.5e11 / 6.9e10 / 1.7e11 | 1.7e10 / 1.7e11 / 1.7e12 | ae_x / ae_x | 1.3e6 / 2.5e10 [0] | 6.7e9 | 6.22e3 / 9.85e3 |
| n2s | lbfgs_so | 19 | 19144 | 0.0291 | 3.7e7 / 7.2e11 | 2.3e8 / 8.4e12 [0.4] | 5.8e6 / 1.2e11 | 1e6 / 2.3e10 | 1.1e6 / 4e10 [0.4] | 2.3e11 / 4.6e10 / 7.9e10 | 8.1e9 / 8.1e10 / 8.1e11 | ae_x / ae_x | 1.3e6 / 2.6e10 [0] | 7.1e9 | 5.12e3 / 7.53e3 |
| h8 | lbfgs | 0 | 0 | - | - | - | - | - | - | - | - | not costed / not costed | 9.6e6 / 4.1e10 [0] | 3.1e9 | 1.32e3 / 1.32e3 |
| h8 | lbfgs | 1 | 3528 | - | - | - | - | - | - | - | - | not costed / not costed | 9.6e6 / 3e11 [0] | 2e10 | 9.74e3 / 9.74e3 |
| h8 | lbfgs | 2 | 33874 | - | - | - | - | - | - | - | - | not costed / not costed | 9.5e6 / 6e11 [0] | 3.9e10 | 1.91e4 / 1.91e4 |
| h8 | lbfgs | 3 | 64220 | - | - | - | - | - | - | - | - | not costed / not costed | 9.5e6 / 8.8e11 [0] | 5.7e10 | 2.8e4 / 2.8e4 |
| h8 | lbfgs | 4 | 94522 | - | - | - | - | - | - | - | - | not costed / not costed | 9.4e6 / 1.1e12 [1] | 7.3e10 | 3.59e4 / 3.59e4 |
| h8 | lbfgs | 5 | 124822 | - | - | - | - | - | - | - | - | not costed / not costed | 9.3e6 / 1.4e12 [1] | 9.2e10 | 4.47e4 / 4.47e4 |
| h8 | lbfgs | 6 | 154942 | - | - | - | - | - | - | - | - | not costed / not costed | 9.2e6 / 1.7e12 [2] | 1.1e11 | 5.28e4 / 5.28e4 |
| h8 | lbfgs | 7 | 185106 | - | - | - | - | - | - | - | - | not costed / not costed | 9.1e6 / 1.9e12 [1] | 1.3e11 | 6.13e4 / 6.13e4 |
| h8 | lbfgs | 8 | 215318 | - | - | - | - | - | - | - | - | not costed / not costed | 9.1e6 / 2.2e12 [1] | 1.5e11 | 6.96e4 / 6.96e4 |
| h8 | lbfgs | 9* | 245530 | - | - | - | - | - | - | - | - | not costed / not costed | 9.1e6 / 2.5e12 [1] | 1.7e11 | 7.9e4 / 7.9e4 |
| h8 | lbfgs | 10 | 275650 | - | - | - | - | - | - | - | - | not costed / not costed | 9.2e6 / 2.7e12 [1] | 1.8e11 | 8.74e4 / 8.74e4 |
| h8 | lbfgs | 11 | 305674 | - | - | - | - | - | - | - | - | not costed / not costed | 9.2e6 / 3e12 [1] | 2e11 | 9.63e4 / 9.63e4 |
| h8 | lbfgs | 12 | 337402 | - | - | - | - | - | - | - | - | not costed / not costed | 9.2e6 / 3.3e12 [1] | 2.2e11 | 1.06e5 / 1.06e5 |
| h8 | lbfgs | 13 | 367478 | - | - | - | - | - | - | - | - | not costed / not costed | 9.2e6 / 3.6e12 [1] | 2.4e11 | 1.14e5 / 1.14e5 |
| h8 | lbfgs_so | 0 | 0 | - | - | - | - | - | - | - | - | not costed / not costed | 9.9e6 / 4.3e10 [0] | 3.1e9 | 1.37e3 / 1.37e3 |
| h8 | lbfgs_so | 1 | 3528 | - | - | - | - | - | - | - | - | not costed / not costed | 9.3e6 / 3e11 [0] | 2e10 | 9.5e3 / 9.5e3 |
| h8 | lbfgs_so | 2 | 33738 | - | - | - | - | - | - | - | - | not costed / not costed | 9.3e6 / 5.8e11 [0] | 3.8e10 | 1.84e4 / 1.84e4 |
| h8 | lbfgs_so | 3 | 63994 | - | - | - | - | - | - | - | - | not costed / not costed | 9.2e6 / 8.3e11 [0] | 5.6e10 | 2.66e4 / 2.66e4 |
| h8 | lbfgs_so | 4 | 94294 | - | - | - | - | - | - | - | - | not costed / not costed | 9.2e6 / 1.1e12 [2] | 7.3e10 | 3.48e4 / 3.48e4 |
| h8 | lbfgs_so | 5 | 124370 | - | - | - | - | - | - | - | - | not costed / not costed | 9.2e6 / 1.4e12 [2] | 9.1e10 | 4.35e4 / 4.35e4 |
| h8 | lbfgs_so | 6* | 154626 | - | - | - | - | - | - | - | - | not costed / not costed | 9.2e6 / 1.6e12 [2] | 1.1e11 | 5.24e4 / 5.24e4 |
| h8 | lbfgs_so | 7 | 184834 | - | - | - | - | - | - | - | - | not costed / not costed | 9.2e6 / 1.9e12 [1] | 1.3e11 | 6.07e4 / 6.07e4 |
| h8 | lbfgs_so | 8 | 214858 | - | - | - | - | - | - | - | - | not costed / not costed | 9.2e6 / 2.2e12 [0] | 1.5e11 | 6.98e4 / 6.98e4 |
| h8 | lbfgs_so | 9 | 244934 | - | - | - | - | - | - | - | - | not costed / not costed | 9.2e6 / 2.4e12 [2] | 1.6e11 | 7.79e4 / 7.79e4 |
| h8 | lbfgs_so | 10 | 275073 | - | - | - | - | - | - | - | - | not costed / not costed | 9.2e6 / 2.7e12 [2] | 1.8e11 | 8.75e4 / 8.75e4 |

- **At HF** the direct route costs no Toffolis (HF is Clifford), so C_iter(0) is the line
  search: 154 (lih) to 553 (h6s) C_QPE with the residual, 1322 (spin-free) and 1366
  (spin-orbital) C_QPE on h8, line search only.
- **The line search dominates early.** Before k\* the residual costs at most 0.57 of the line
  search on every equilibrium record (0.03–0.10 on h4 and n2); after k\* it takes over on the
  L-BFGS records of lih (at k\*), n2 and h6a (k\* + 2; k\* + 1 for the spin-orbital n2
  classical-first-step record) and h4 (k\* + 2 to k\* + 4), and of the SD records only on lih
  (k\* + 1). On h6s the residual overtakes the line search from k = 8 (spin-free) or k = 6–7
  (spin-orbital), up to 2.1× before k\*; on n2s only in the last iterations (k ≥ 13), up to
  5.5× (spin-free) and 7.2× (spin-orbital, classical first step). The classical-first-step
  figures are not in T3 (`iterations.<family>.classical_first_step` in `results/ft_n2.json` and
  `results/ft_n2s.json`). Amplitude-estimated energies
  (the F13 option, T3's "AE energies" column) would cost 0.065 (h8) to 0.31 (lih at HF) of the
  sampled line search in T3, and up to 0.313 on the spin-orbital classical-first-step record of
  n2s (at k = 13; `results/ft_n2s.json`).
- **Routes.** Matchgate shadows of the direct route are the cheapest residual estimator up to
  k = 1 (L-BFGS on h4, n2 and lih at κ = 1, SD on n2 and lih), 2 (h6a, h6s, SD on h4,
  spin-orbital n2s) or 3 (spin-free n2s); then per-element amplitude estimation takes over: the
  X-LCU route (ae_x) on h4, n2, lih and n2s (ae_aux, δ = 0.8, at k = 9 of h4 SD), the aux
  route (ae_aux, δ = 0.4–1.0) on h6a and h6s. At κ = 2 the switch comes one iterate later on
  most records (T5). The aux sampling route is never the cheapest.
- **Gradient estimation** at κ_g = 1 would cost 0.09 (h6a, h6s) to 1.8 (n2) times the best
  route from k = 2 on, at κ_g = 10 0.9–18 times; κ_g is unknown, so the band is never selected
  (F12).
- **Small candidates.** The F3 rounding flag marks 0–3 of the 11 line-search candidates per
  iteration (T9 counts them per system).
- **Steps beyond the grid.** On n2s four steps exceed the largest candidate, 6.4: ε = 7.17 at
  k = 14 of the spin-free exact record, 11.4 at k = 2 of its classical-first-step record, and
  6.61 and 6.94 at k = 2 and 17 of the spin-orbital classical-first-step record. Their
  line-search rows assume the grid. Every other step before k\* lies within it (0.00625–6.4).
- **Feasibility.** With the selected walks no route is infeasible over the costed iterates: the
  factorization bias stays below 0.47 r‖A‖_m (n2s, k = 16; 0.36 on lih, at most 0.17
  elsewhere). With DF at threshold 3e-3, Appendix A's encoding, every walk-based route would
  become infeasible near convergence, first at h4 k = 7 and n2 k = 6 (bias 3.1e-3 Ha against
  r‖A‖_m = 2.3e-3 Ha at h4 k = 7, 2.6e-3 against 1.7e-3 Ha at n2 k = 6; a separate run
  [build check]: `ft.residual_inputs.factorization_bias` of that DF model, built by
  `ft.qpe.df_encoding(active, 3e-3, ...)`, along the spin-free L-BFGS iterates).

**Inputs of T3** (`results/ft_<system>.json`: `settings`, and `line_search.params` of the
iterations). V_m = Σ_e m_e v_e of the direct route, computed at iterate 1 of spin-free L-BFGS and
reused at every iterate (F10); V_E of the DF split chosen at HF, evaluated at Ψ_{k+1} of
iteration k (F13; the column gives Ψ₁); C_rot and Q_rot of the readout rotation (F4).

| system | allowed elements | V_m (Ha²) | DF split | V_E at Ψ₁ (Ha²) | C_rot / Q_rot |
|---|---|---|---|---|---|
| h4 | 28 | 1216.7, exact | eigh_shift | 0.610 | 150 / 79 |
| h6a | 153 | 2.513e4 ± 113, RB | cholesky_shift | 1.230 | 405 / 89 |
| n2 | 33 | 3751.1, exact | cholesky_shift | 0.827 | 405 / 89 |
| lih | 105 | 3.778e4 ± 396, RB | cholesky_shift | 0.672 | 405 / 89 |
| h6s | 153 | 1.459e4 ± 55, RB | cholesky_shift | 0.912 | 405 / 89 |
| n2s | 33 | 2849.0, exact | cholesky_shift | 0.466 | 405 / 89 |
| h8 | 496, residual not costed | - | eigh_shift | 2.234 | 784 / 96 |

**Worked example** (h4, spin-free L-BFGS, measured loop, k = 1; T2, T3 and the table above).
With ‖A₁‖_m = 0.6115 Ha and r = 0.3 the variance budget is B = 0.1834 Ha, so direct_mg takes
S = V_m/B² = 3.615e4 shots of C(Ψ₁) = 160 Toffolis, T = 5.78e6 (F10). The line search measures
the 11 candidates with V_E(Ψ₂) = 0.6395 Ha² to 1.6 mHa, S_E = 2.498e5 shots each
(11·S_E = 2.75e6, T3's "line search S"). Their steps cost C_step(1; ε_c) = 112, 128, 410, 570,
894, 1042, 1116 and 4 × 1192 Toffolis, 9040 in all (`ft.stateprep.candidate_step_costs`; not
stored [build check]), so T_LS = S_E·(11·(160 + 150) + 9040) = 3.110e9 (F13) and
C_iter(1) = 3.116e9 Toffolis = 785.0 C_QPE, with C_QPE = 3,969,408 (T3). With
C_iter(0) = 202.9 C_QPE and p₁ = 0.98023, rule E gives
T_E(1) = 202.9 + (160 + C_QPE)/(p₁·C_QPE) = 203.9 C_QPE (T6).

**T5** (crossovers, measured loop; r-scan).

First iterate of the measured loop at which an amplitude-estimation route is cheaper than both sampling routes; norm A*_m: the residual norm at which direct_mg and the cheapest AE route at k* cost the same (A scaled with biases, variances and circuits fixed; residual.crossover_norm): sampling is cheaper above it. r-scan: the relative residual error r at k = 2 and k*.

| system | family | first k: AE < sampling, kappa 1 | kappa 2 | k* | norm A_m at k* | norm A*_m (direct_mg = best AE) at k* |
|---|---|---|---|---|---|---|
| h4 | lbfgs | 2 (ae_x) | 3 (ae_x) | 3 | 0.126 | 0.305 (ae_x) |
| h4 | lbfgs_so | 2 (ae_x) | 3 (ae_x) | 3 | 0.0925 | 0.303 (ae_x) |
| h4 | sd | 3 (ae_x) | 4 (ae_x) | 5 | 0.124 | 0.372 (ae_x) |
| h6a | lbfgs | 3 (ae_aux) | 4 (ae_aux) | 6 | 0.136 | 1.2 (ae_aux) |
| h6a | lbfgs_so | 3 (ae_aux) | 4 (ae_aux) | 4 | 0.137 | 0.854 (ae_aux) |
| h6a | sd | 3 (ae_aux) | 4 (ae_aux) | 14 | 0.095 | 1.81 (ae_aux) |
| n2 | lbfgs | 2 (ae_x) | 3 (ae_x) | 3 | 0.117 | 0.724 (ae_x) |
| n2 | lbfgs_so | 2 (ae_x) | 2 (ae_x) | 3 | 0.087 | 0.724 (ae_x) |
| n2 | sd | 2 (ae_x) | 3 (ae_x) | 7 | 0.112 | 1.05 (ae_x) |
| lih | lbfgs | 2 (ae_x) | 2 (ae_x) | 2 | 0.114 | 0.772 (ae_x) |
| lih | lbfgs_so | 2 (ae_x) | 2 (ae_x) | 2 | 0.0845 | 0.715 (ae_x) |
| lih | sd | 2 (ae_x) | 2 (ae_x) | 2 | 0.112 | 0.765 (ae_x) |
| h6s | lbfgs | 3 (ae_aux) | 4 (ae_aux) | 12 | 0.0393 | 1.09 (ae_aux) |
| h6s | lbfgs_so | 3 (ae_aux) | 4 (ae_aux) | 10 | 0.0363 | 1.02 (ae_aux) |
| n2s | lbfgs | 4 (ae_x) | 5 (ae_x) | - | - | - |
| n2s | lbfgs_so | 3 (ae_x) | 4 (ae_x) | - | - | - |

| system | family | k | r | best kappa 1 (T) | best kappa 2 (T) | direct_mg T | best AE kappa 1 (T) | line search T | C_iter / C_QPE |
|---|---|---|---|---|---|---|---|---|---|
| h6a | lbfgs | 2 | 0.1 | ae_aux (1.2e11) | direct_mg (1.3e11) | 1.3e11 | ae_aux (1.2e11) | 9.5e10 | 1.43e4 |
| h6a | lbfgs | 2 | 0.3 | direct_mg (1.5e10) | direct_mg (1.5e10) | 1.5e10 | ae_aux (2.7e10) | 9.5e10 | 7.38e3 |
| h6a | lbfgs | 2 | 1 | direct_mg (1.3e9) | direct_mg (1.3e9) | 1.3e9 | ae_aux (5.5e9) | 9.5e10 | 6.46e3 |
| h6a | lbfgs | 6 | 0.1 | ae_aux (7.1e11) | ae_aux (1.4e12) | 5.9e12 | ae_aux (7.1e11) | 2.7e11 | 6.61e4 |
| h6a | lbfgs | 6 | 0.3 | ae_aux (1.4e11) | ae_aux (2.9e11) | 6.5e11 | ae_aux (1.4e11) | 2.7e11 | 2.76e4 |
| h6a | lbfgs | 6 | 1 | ae_aux (2.6e10) | ae_aux (5.1e10) | 5.9e10 | ae_aux (2.6e10) | 2.7e11 | 1.96e4 |
| h6a | lbfgs_so | 2 | 0.1 | ae_aux (2.5e11) | direct_mg (4.3e11) | 4.3e11 | ae_aux (2.5e11) | 9.1e10 | 2.31e4 |
| h6a | lbfgs_so | 2 | 0.3 | direct_mg (4.8e10) | direct_mg (4.8e10) | 4.8e10 | ae_aux (5.3e10) | 9.1e10 | 9.37e3 |
| h6a | lbfgs_so | 2 | 1 | direct_mg (4.3e9) | direct_mg (4.3e9) | 4.3e9 | ae_aux (1e10) | 9.1e10 | 6.45e3 |
| h6a | lbfgs_so | 4 | 0.1 | ae_aux (6.8e11) | ae_aux (1.4e12) | 3.5e12 | ae_aux (6.8e11) | 1.8e11 | 5.74e4 |
| h6a | lbfgs_so | 4 | 0.3 | ae_aux (1.3e11) | ae_aux (2.7e11) | 3.9e11 | ae_aux (1.3e11) | 1.8e11 | 2.09e4 |
| h6a | lbfgs_so | 4 | 1 | ae_aux (2.4e10) | direct_mg (3.5e10) | 3.5e10 | ae_aux (2.4e10) | 1.8e11 | 1.35e4 |
| h6a | sd | 2 | 0.1 | ae_aux (1.1e11) | direct_mg (1.2e11) | 1.2e11 | ae_aux (1.1e11) | 9.5e10 | 1.36e4 |
| h6a | sd | 2 | 0.3 | direct_mg (1.4e10) | direct_mg (1.4e10) | 1.4e10 | ae_aux (2.4e10) | 9.5e10 | 7.33e3 |
| h6a | sd | 2 | 1 | direct_mg (1.2e9) | direct_mg (1.2e9) | 1.2e9 | ae_aux (4.9e9) | 9.5e10 | 6.5e3 |
| h6a | sd | 14 | 0.1 | ae_aux (1.7e12) | ae_aux (3.5e12) | 3.1e13 | ae_aux (1.7e12) | 6.2e11 | 1.6e5 |
| h6a | sd | 14 | 0.3 | ae_aux (3.4e11) | ae_aux (6.7e11) | 3.4e12 | ae_aux (3.4e11) | 6.2e11 | 6.47e4 |
| h6a | sd | 14 | 1 | ae_aux (5.7e10) | ae_aux (1.1e11) | 3.1e11 | ae_aux (5.7e10) | 6.2e11 | 4.6e4 |
| n2 | lbfgs | 2 | 0.1 | ae_x (8.8e8) | ae_x (1.8e9) | 3.5e9 | ae_x (8.8e8) | 9.9e9 | 856 |
| n2 | lbfgs | 2 | 0.3 | ae_x (2.9e8) | direct_mg (3.9e8) | 3.9e8 | ae_x (2.9e8) | 9.9e9 | 809 |
| n2 | lbfgs | 2 | 1 | direct_mg (3.5e7) | direct_mg (3.5e7) | 3.5e7 | ae_x (8.8e7) | 9.9e9 | 789 |
| n2 | lbfgs | 3 | 0.1 | ae_x (3.8e9) | ae_x (7.6e9) | 7.1e10 | ae_x (3.8e9) | 1.4e10 | 1.38e3 |
| n2 | lbfgs | 3 | 0.3 | ae_x (1.3e9) | ae_x (2.5e9) | 7.9e9 | ae_x (1.3e9) | 1.4e10 | 1.18e3 |
| n2 | lbfgs | 3 | 1 | ae_x (3.8e8) | direct_mg (7.1e8) | 7.1e8 | ae_x (3.8e8) | 1.4e10 | 1.11e3 |
| n2 | lbfgs_so | 2 | 0.1 | ae_x (2.2e9) | ae_x (4.3e9) | 2.1e10 | ae_x (2.2e9) | 9.1e9 | 890 |
| n2 | lbfgs_so | 2 | 0.3 | ae_x (7.2e8) | ae_x (1.4e9) | 2.4e9 | ae_x (7.2e8) | 9.1e9 | 775 |
| n2 | lbfgs_so | 2 | 1 | direct_mg (2.1e8) | direct_mg (2.1e8) | 2.1e8 | ae_x (2.2e8) | 9.1e9 | 735 |
| n2 | lbfgs_so | 3 | 0.1 | ae_x (5.1e9) | ae_x (1e10) | 1.3e11 | ae_x (5.1e9) | 1.3e10 | 1.41e3 |
| n2 | lbfgs_so | 3 | 0.3 | ae_x (1.7e9) | ae_x (3.4e9) | 1.4e10 | ae_x (1.7e9) | 1.3e10 | 1.14e3 |
| n2 | lbfgs_so | 3 | 1 | ae_x (5.1e8) | ae_x (1e9) | 1.3e9 | ae_x (5.1e8) | 1.3e10 | 1.05e3 |
| n2 | sd | 2 | 0.1 | ae_x (7.5e8) | ae_x (1.5e9) | 2.5e9 | ae_x (7.5e8) | 1e10 | 864 |
| n2 | sd | 2 | 0.3 | ae_x (2.5e8) | direct_mg (2.8e8) | 2.8e8 | ae_x (2.5e8) | 1e10 | 825 |
| n2 | sd | 2 | 1 | direct_mg (2.5e7) | direct_mg (2.5e7) | 2.5e7 | ae_x (7.5e7) | 1e10 | 807 |
| n2 | sd | 7 | 0.1 | ae_x (7e9) | ae_x (1.4e10) | 2e11 | ae_x (7e9) | 2.8e10 | 2.75e3 |
| n2 | sd | 7 | 0.3 | ae_x (2.3e9) | ae_x (4.7e9) | 2.2e10 | ae_x (2.3e9) | 2.8e10 | 2.38e3 |
| n2 | sd | 7 | 1 | ae_x (7e8) | ae_x (1.4e9) | 2e9 | ae_x (7e8) | 2.8e10 | 2.25e3 |

- **Crossover.** At k\* the residual norm ‖A‖_m is 0.04–0.14 Ha, while sampling would beat the
  best AE route only above ‖A‖\*_m = 0.30–1.81 Ha.
- **r-scan.** At k = 2, loosening r from 0.3 to 1.0 lowers C_iter by at most 1.45× (h6a
  spin-free 7.38e3 → 6.46e3 C_QPE, spin-orbital 9.37e3 → 6.45e3; n2 809 → 789). Tightening to
  r = 0.1 doubles it on h6a (1.43e4, 2.2× the r = 1.0 value; 3.6× spin-orbital, 2.1× SD),
  where the residual (ae_aux, 1.2e11 Toffolis) then costs more than the line search (9.5e10);
  on n2 it changes little (856, 1.09×). At k\* the residual weighs more: h6a 6.61e4 / 2.76e4 /
  1.96e4 C_QPE for r = 0.1 / 0.3 / 1.0, n2 1.38e3 / 1.18e3 / 1.11e3. The loose r = 1.0 moves
  n2's best route at k = 2 back to sampling (ae_x → direct_mg); h6a samples there at r = 0.3
  already. M2 found the noisy H6 CQE converging with residuals off by 74–137% (S_res = 1e6),
  so r = 1.0 need not slow the optimizer.

## 4. Totals to chemical accuracy and wall clock (T4)

**T4** (totals to chemical accuracy, C_loop(k*); every family, measured and classical-first-step
loops).

Sums over the iterations j < k* (Toffolis). 'classical_first_step' costs the record with the classical eps_2 at HF and C_iter(0) = 0. Wall clock (F19): every campaign of the loop one after another on one machine; critical path: the longest run of every campaign, the wall clock on unlimited machines; machines: the smallest number that finishes the loop in 7 days, amplitude-estimation runs in LPT order ('none': the critical path exceeds 7 days). '>=': a lower bound, either line search only (h8 without the residual) or k* not reached (n2s: the sums run over the costed iterations).

| system | family | loop | k* | sum residual, direct only | sum residual, best kappa 1 | best kappa 2 | sum line search | total (kappa 1) | total / C_QPE, kappa 1 / 2 | total / C_QPE, AE line search | one machine | critical path | machines for 7 days |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| h4 | lbfgs | measured | 3 | 4e8 | 3.7e8 | 4e8 | 1e10 | 1e10 | 2.62e3 / 2.63e3 | 436 | 3.4 d | 12 min | 1 |
| h4 | lbfgs | classical_first_step | 3 | 4e8 | 3.7e8 | 4e8 | 9.2e9 | 9.6e9 | 2.42e3 / 2.42e3 | 391 | 3.1 d | 12 min | 1 |
| h4 | lbfgs_so | measured | 3 | 9.8e8 | 6e8 | 9.8e8 | 9.8e9 | 1e10 | 2.61e3 / 2.71e3 | 490 | 3.4 d | 18 min | 1 |
| h4 | lbfgs_so | classical_first_step | 3 | 9.8e8 | 5.9e8 | 9.8e8 | 8.9e9 | 9.5e9 | 2.38e3 / 2.48e3 | 443 | 3.1 d | 18 min | 1 |
| h4 | sd | measured | 5 | 3.3e9 | 1.7e9 | 2.8e9 | 3.3e10 | 3.5e10 | 8.72e3 / 9e3 | 1.47e3 | 12 d | 46 min | 2 |
| h4 | sd | classical_first_step | 5 | 3.5e9 | 1.7e9 | 2.9e9 | 3.2e10 | 3.4e10 | 8.54e3 / 8.82e3 | 1.44e3 | 11 d | 47 min | 2 |
| h6a | lbfgs | measured | 6 | 6.9e11 | 2.6e11 | 4.8e11 | 6.9e11 | 9.5e11 | 6.38e4 / 7.89e4 | 2.1e4 | 3.4e2 d | 20 h | 51 |
| h6a | lbfgs | classical_first_step | 6 | 6.9e11 | 2.6e11 | 4.8e11 | 6.8e11 | 9.4e11 | 6.33e4 / 7.85e4 | 2.09e4 | 3.4e2 d | 20 h | 51 |
| h6a | lbfgs_so | measured | 4 | 1.4e11 | 1e11 | 1.4e11 | 2.8e11 | 3.8e11 | 2.58e4 / 2.82e4 | 8.35e3 | 1.3e2 d | 4.3 h | 20 |
| h6a | lbfgs_so | classical_first_step | 4 | 1.4e11 | 1e11 | 1.4e11 | 2.7e11 | 3.7e11 | 2.52e4 / 2.77e4 | 8.29e3 | 1.3e2 d | 4.3 h | 19 |
| h6a | sd | measured | 14 | 9.9e12 | 1.5e12 | 2.9e12 | 4.1e12 | 5.6e12 | 3.78e5 / 4.75e5 | 1.22e5 | 2.2e3 d | 5.3 d | 848 |
| h6a | sd | classical_first_step | 14 | 1e13 | 1.5e12 | 3e12 | 4.1e12 | 5.6e12 | 3.79e5 / 4.77e5 | 1.23e5 | 2.2e3 d | 5.3 d | 869 |
| n2 | lbfgs | measured | 3 | 4.4e8 | 3.4e8 | 4.4e8 | 1.9e10 | 1.9e10 | 1.52e3 / 1.53e3 | 258 | 6.3 d | 10 min | 1 |
| n2 | lbfgs | classical_first_step | 3 | 4.4e8 | 3.4e8 | 4.4e8 | 1.6e10 | 1.6e10 | 1.28e3 / 1.29e3 | 211 | 5.3 d | 10 min | 1 |
| n2 | lbfgs_so | measured | 3 | 2.5e9 | 8.3e8 | 1.6e9 | 1.8e10 | 1.9e10 | 1.53e3 / 1.58e3 | 293 | 6.3 d | 25 min | 1 |
| n2 | lbfgs_so | classical_first_step | 3 | 2.4e9 | 8.3e8 | 1.5e9 | 1.5e10 | 1.6e10 | 1.26e3 / 1.32e3 | 245 | 5.3 d | 24 min | 1 |
| n2 | sd | measured | 7 | 1.8e10 | 3.9e9 | 7.6e9 | 9.5e10 | 9.9e10 | 7.81e3 / 8.09e3 | 1.36e3 | 33 d | 2.2 h | 5 |
| n2 | sd | classical_first_step | 7 | 1.8e10 | 3.9e9 | 7.6e9 | 9.1e10 | 9.4e10 | 7.48e3 / 7.77e3 | 1.3e3 | 32 d | 2.1 h | 5 |
| lih | lbfgs | measured | 2 | 2.1e9 | 2.1e9 | 2.1e9 | 1.3e10 | 1.5e10 | 967 / 967 | 340 | 4.9 d | 1.5 s | 1 |
| lih | lbfgs | classical_first_step | 2 | 2e9 | 2e9 | 2e9 | 1.1e10 | 1.3e10 | 808 / 808 | 288 | 4.2 d | 1.3 s | 1 |
| lih | lbfgs_so | measured | 2 | 2.1e9 | 2.1e9 | 2.1e9 | 1.1e10 | 1.3e10 | 848 / 848 | 319 | 4.3 d | 1.3 s | 1 |
| lih | lbfgs_so | classical_first_step | 2 | 2.1e9 | 2.1e9 | 2.1e9 | 8.8e9 | 1.1e10 | 690 / 690 | 267 | 3.6 d | 1.1 s | 1 |
| lih | sd | measured | 2 | 2.1e9 | 2.1e9 | 2.1e9 | 1.6e10 | 1.8e10 | 1.14e3 / 1.14e3 | 379 | 5.9 d | 1.9 s | 1 |
| lih | sd | classical_first_step | 2 | 2e9 | 2e9 | 2e9 | 1.4e10 | 1.6e10 | 983 / 983 | 326 | 5.1 d | 1.6 s | 1 |
| h6s | lbfgs | measured | 12 | 2.6e13 | 2.2e12 | 4.3e12 | 1.8e12 | 3.9e12 | 4.04e5 / 6.24e5 | 2.39e5 | 1.7e3 d | 8.5 d | none |
| h6s | lbfgs | classical_first_step | 12 | 2.6e13 | 2.2e12 | 4.3e12 | 1.8e12 | 3.9e12 | 4.05e5 / 6.27e5 | 2.41e5 | 1.7e3 d | 8.5 d | none |
| h6s | lbfgs_so | measured | 10 | 1.3e13 | 1.3e12 | 2.6e12 | 1.2e12 | 2.5e12 | 2.58e5 / 3.88e5 | 1.45e5 | 1.1e3 d | 4.9 d | 203 |
| h6s | lbfgs_so | classical_first_step | 10 | 1.2e13 | 1.3e12 | 2.6e12 | 1.2e12 | 2.5e12 | 2.55e5 / 3.85e5 | 1.43e5 | 1.1e3 d | 4.9 d | 197 |
| n2s | lbfgs | measured | - (20 costed) | >= 4.1e13 | >= 3.9e11 | >= 7.7e11 | >= 2.9e11 | >= 6.7e11 | >= 7.08e4 / >= 1.11e5 | >= 4.85e4 | >= 2.9e2 d | >= 8.8 d | none |
| n2s | lbfgs | classical_first_step | - (20 costed) | >= 4.9e13 | >= 4.3e11 | >= 8.6e11 | >= 2.8e11 | >= 7.1e11 | >= 7.45e4 / >= 1.2e5 | >= 5.33e4 | >= 3.1e2 d | >= 9.9 d | none |
| n2s | lbfgs_so | measured | - (20 costed) | >= 1.4e13 | >= 2.8e11 | >= 5.6e11 | >= 2.9e11 | >= 5.7e11 | >= 5.98e4 / >= 8.93e4 | >= 3.73e4 | >= 2.3e2 d | >= 6.1 d | >= 110 |
| n2s | lbfgs_so | classical_first_step | - (20 costed) | >= 7.7e13 | >= 5.5e11 | >= 1.1e12 | >= 2.6e11 | >= 8.1e11 | >= 8.51e4 / >= 1.43e5 | >= 6.57e4 | >= 3.7e2 d | >= 13 d | none |
| h8 | lbfgs | measured | 9 | - | - | - | 1e13 | >= 1e13 | >= 3.23e5 / >= 3.23e5 | >= 2.13e4 | >= 3.9e3 d | >= 6.6 min | >= 551 |
| h8 | lbfgs | classical_first_step | 9 | - | - | - | 1e13 | >= 1e13 | >= 3.21e5 / >= 3.21e5 | >= 2.12e4 | >= 3.8e3 d | >= 6.6 min | >= 548 |
| h8 | lbfgs_so | measured | 6 | - | - | - | 4.2e12 | >= 4.2e12 | >= 1.34e5 / >= 1.34e5 | >= 8.96e3 | >= 1.5e3 d | >= 2.7 min | >= 221 |
| h8 | lbfgs_so | classical_first_step | 6 | - | - | - | 4.2e12 | >= 4.2e12 | >= 1.35e5 / >= 1.35e5 | >= 8.98e3 | >= 1.6e3 d | >= 2.7 min | >= 223 |

- **Totals.** C_loop(k\*) with the classical first step: spin-free L-BFGS 2.42e3 (h4), 6.33e4
  (h6a), 1.28e3 (n2), 808 (lih), 4.05e5 (h6s) and ≥ 3.21e5 (h8) C_QPE; steepest descent
  8.54e3, 3.79e5, 7.48e3 and 983 on the core systems. At equilibrium the classical first step
  saves 1% (h6a) to 16% (n2, lih) of the measured loop with spin-free L-BFGS (0.5% on h8, lower
  bounds); with steepest descent on h6a and spin-orbital L-BFGS on h8 it costs 0.2–0.3% more
  (3.79e5 against 3.78e5 C_QPE; ≥ 1.35e5 against ≥ 1.34e5). On h6s it costs 0.3% more with
  spin-free L-BFGS (4.052e5 against 4.041e5 C_QPE; the spin-orbital step saves 1%) and on n2s
  5% more over the 20 costed iterations (42% with the spin-orbital metric): its record follows a
  different path after the step, whose iterations cost more. κ = 2 adds 0–24% to spin-free
  L-BFGS at equilibrium (h6a: 6.33e4 to 7.85e4), 0–10% to spin-orbital L-BFGS and 0–26% to
  steepest descent (h6a: 3.79e5 to 4.77e5), and 51–55% on h6s.
- **With amplitude-estimated line-search energies** the totals fall to 0.16–0.39 of these at
  equilibrium, 0.56–0.59 on h6s and 0.62–0.77 on n2s (over its 20 costed iterations), and to
  0.07 on h8, where the line search is all of the cost; they are still 211 (n2) to 2.4e5 (h6s)
  C_QPE.
- **Wall clock** (spin-free L-BFGS, classical first step, unless noted). One machine needs 3.1
  (h4), 5.3 (n2) and 4.2 days (lih), 340 days (h6a), 4.8 years (h6s) and ≥ 10 years (h8, line
  search alone). Within 7 days: 1 machine for h4, n2 and lih, 51 for h6a (19–20 with the
  spin-orbital family), 197–203 for h6s with the spin-orbital family and none with spin-free
  L-BFGS: the longest run of each of its campaigns, mostly amplitude-estimation runs that cannot
  be split, adds up to a critical path of 8.5 days. Steepest descent needs 11 days and
  2 machines (h4), 32 days and 5 machines (n2), 5.1 days and 1 machine (lih), and 5.9 years and
  869 machines (h6a; 848 for the measured loop) (T4).
- **Schedule.** Shots of one circuit take t⌈n/M⌉ on M machines; the amplitude-estimation runs
  of a residual (one per element, of unequal lengths) follow the LPT schedule, longest run first
  onto the least-loaded machine. Any schedule needs at least t⌈n/M⌉ for shots and
  max(W/M, t_max) for AE runs, summed over the campaigns. The LPT machine count is the smallest
  M that this bound allows for 28 of the 31 loop totals that have a count (at most 1.05× it; the
  other 5 have none, their critical path exceeding 7 days), and for 324 of the 404 costed
  iterations (at most 1.38×). Sizing M by Graham's bound for list scheduling in arbitrary order,
  W/M + (1 − 1/M) t_max, would ask for up to 2.6× as many machines (spin-orbital h6s: 518
  against 203); a fluid share of the work, ⌈ΣW / 7 d⌉, would give 249 machines to the spin-free
  h6s loop, which no schedule finishes in 7 days.

## 5. The hand-off to QPE (T6, T7)

**T6** (L-BFGS families; steepest descent has the same measured and classical-first-step optima
as spin-free L-BFGS under all four rules, `summary.families.sd.handoff` in
`results/ft_<system>.json`).

T_E(k) = C_loop(k) + (C(Psi_k) + C_QPE)/p_k; T_delta(k) = C_loop(k) + R_delta(p) (C(Psi_k) + C_QPE) with R_delta(p) = ceil(ln 0.01 / ln(1 - p)) at the certified p_lb (Eckart-Temple, same-irrep E1) or the exact p_k (oracle). Rule E is certified iff E1 - E_k > 3 dE. saving = C_QPE/p_0 - (C(Psi_k) + C_QPE)/p_k, bounded by C_QPE (1/p_0 - 1); p0*(k) = C_QPE / T_E(k) is the HF overlap below which handing off Psi_k beats HF. 'measured' and 'free' use the exact records, 'classical_first_step' the second-order records. '>=': C_loop from line-search-only iteration costs (h8 without the residual), a lower bound. Rows k <= 3:

| system | family | loop | k | p_k | p_lb same / any irrep | rule E certified | saving / C_QPE | T_E / C_QPE | T_delta / C_QPE, certified / oracle | p0*(k) | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| h4 | lbfgs | all | 0 | 0.9455 | 0.763 / 0.761 | yes | 0 | 1.058 | 4 / 2 | 0.946 | - |
| h4 | lbfgs | measured | 1 | 0.9802 | 0.918 / 0.918 | yes | 0.0374 | 203.9 | 204.9 / 204.9 | 0.0049 | - |
| h4 | lbfgs | measured | 2 | 0.9960 | 0.984 / 0.984 | yes | 0.0532 | 988.9 | 989.9 / 988.9 | 0.00101 | - |
| h4 | lbfgs | measured | 3 | 0.9981 | 0.997 / 0.997 | yes | 0.0551 | 2622 | 2622 / 2622 | 0.000381 | - |
| h4 | lbfgs | classical_first_step | 1 | 0.9804 | 0.917 / 0.917 | yes | 0.0376 | 1.02 | 2 / 2 | 0.98 | - |
| h4 | lbfgs | classical_first_step | 2 | 0.9960 | 0.985 / 0.984 | yes | 0.0532 | 783.8 | 784.8 / 783.8 | 0.00128 | - |
| h4 | lbfgs | classical_first_step | 3 | 0.9981 | 0.997 / 0.997 | yes | 0.0551 | 2418 | 2418 / 2418 | 0.000414 | - |
| h4 | lbfgs | free | 1 | 0.9802 | 0.918 / 0.918 | yes | 0.0374 | 1.02 | 2 / 2 | 0.98 | - |
| h4 | lbfgs | free | 2 | 0.9960 | 0.984 / 0.984 | yes | 0.0532 | 1.004 | 2.001 / 1 | 0.996 | - |
| h4 | lbfgs | free | 3 | 0.9981 | 0.997 / 0.997 | yes | 0.0551 | 1.003 | 1.001 / 1.001 | 0.997 | - |
| h4 | lbfgs_so | all | 0 | 0.9455 | 0.763 / 0.761 | yes | 0 | 1.058 | 4 / 2 | 0.946 | - |
| h4 | lbfgs_so | measured | 1 | 0.9911 | 0.966 / 0.965 | yes | 0.0486 | 216.7 | 217.7 / 216.7 | 0.00461 | - |
| h4 | lbfgs_so | measured | 2 | 0.9981 | 0.993 / 0.993 | yes | 0.0554 | 994.8 | 994.8 / 994.8 | 0.00101 | - |
| h4 | lbfgs_so | measured | 3 | 0.9996 | 0.999 / 0.999 | yes | 0.0566 | 2612 | 2612 / 2612 | 0.000383 | - |
| h4 | lbfgs_so | classical_first_step | 1 | 0.9914 | 0.964 / 0.964 | yes | 0.0489 | 1.009 | 2 / 1 | 0.991 | - |
| h4 | lbfgs_so | classical_first_step | 2 | 0.9981 | 0.993 / 0.993 | yes | 0.0554 | 780.2 | 780.2 / 780.2 | 0.00128 | - |
| h4 | lbfgs_so | classical_first_step | 3 | 0.9996 | 0.999 / 0.999 | yes | 0.0566 | 2385 | 2385 / 2385 | 0.000419 | - |
| h4 | lbfgs_so | free | 1 | 0.9911 | 0.966 / 0.965 | yes | 0.0486 | 1.009 | 2 / 1 | 0.991 | - |
| h4 | lbfgs_so | free | 2 | 0.9981 | 0.993 / 0.993 | yes | 0.0554 | 1.002 | 1 / 1 | 0.998 | - |
| h4 | lbfgs_so | free | 3 | 0.9996 | 0.999 / 0.999 | yes | 0.0566 | 1.001 | 1.001 / 1.001 | 0.999 | - |
| h6a | lbfgs | all | 0 | 0.9026 | 0.409 / 0.409 | yes | 0 | 1.108 | 9 / 2 | 0.903 | - |
| h6a | lbfgs | measured | 1 | 0.9576 | 0.750 / 0.750 | yes | 0.0636 | 490.6 | 493.5 / 491.5 | 0.00204 | - |
| h6a | lbfgs | measured | 2 | 0.9850 | 0.915 / 0.915 | yes | 0.092 | 3690 | 3691 / 3691 | 0.000271 | - |
| h6a | lbfgs | measured | 3 | 0.9907 | 0.968 / 0.968 | yes | 0.0973 | 1.107e4 | 1.107e4 / 1.107e4 | 9.04e-5 | - |
| h6a | lbfgs | classical_first_step | 1 | 0.9580 | 0.747 / 0.747 | yes | 0.064 | 1.044 | 4 / 2 | 0.958 | - |
| h6a | lbfgs | classical_first_step | 2 | 0.9849 | 0.916 / 0.916 | yes | 0.0919 | 3192 | 3193 / 3193 | 0.000313 | - |
| h6a | lbfgs | classical_first_step | 3 | 0.9906 | 0.969 / 0.969 | yes | 0.0972 | 1.055e4 | 1.055e4 / 1.055e4 | 9.48e-5 | - |
| h6a | lbfgs | free | 1 | 0.9576 | 0.750 / 0.750 | yes | 0.0636 | 1.044 | 4 / 2 | 0.958 | - |
| h6a | lbfgs | free | 2 | 0.9850 | 0.915 / 0.915 | yes | 0.092 | 1.016 | 2.001 / 2.001 | 0.984 | - |
| h6a | lbfgs | free | 3 | 0.9907 | 0.968 / 0.968 | yes | 0.0973 | 1.011 | 2.002 / 1.001 | 0.989 | - |
| h6a | lbfgs_so | all | 0 | 0.9026 | 0.409 / 0.409 | yes | 0 | 1.108 | 9 / 2 | 0.903 | - |
| h6a | lbfgs_so | measured | 1 | 0.9769 | 0.880 / 0.880 | yes | 0.0842 | 520.5 | 522.5 / 521.5 | 0.00192 | - |
| h6a | lbfgs_so | measured | 2 | 0.9914 | 0.963 / 0.963 | yes | 0.0986 | 3761 | 3762 / 3761 | 0.000266 | - |
| h6a | lbfgs_so | measured | 3 | 0.9961 | 0.980 / 0.980 | yes | 0.103 | 1.313e4 | 1.314e4 / 1.313e4 | 7.61e-5 | - |
| h6a | lbfgs_so | classical_first_step | 1 | 0.9774 | 0.874 / 0.874 | yes | 0.0848 | 1.023 | 3 / 2 | 0.977 | - |
| h6a | lbfgs_so | classical_first_step | 2 | 0.9914 | 0.963 / 0.963 | yes | 0.0986 | 3241 | 3242 / 3241 | 0.000309 | - |
| h6a | lbfgs_so | classical_first_step | 3 | 0.9960 | 0.981 / 0.981 | yes | 0.103 | 1.257e4 | 1.257e4 / 1.257e4 | 7.95e-5 | - |
| h6a | lbfgs_so | free | 1 | 0.9769 | 0.880 / 0.880 | yes | 0.0842 | 1.024 | 3 / 2 | 0.977 | - |
| h6a | lbfgs_so | free | 2 | 0.9914 | 0.963 / 0.963 | yes | 0.0986 | 1.009 | 2.001 / 1.001 | 0.991 | - |
| h6a | lbfgs_so | free | 3 | 0.9961 | 0.980 / 0.980 | yes | 0.103 | 1.005 | 2.002 / 1.001 | 0.995 | - |
| n2 | lbfgs | all | 0 | 0.9257 | 0.593 / 0.257 | yes | 0 | 1.08 | 6 / 2 | 0.926 | - |
| n2 | lbfgs | measured | 1 | 0.9635 | 0.924 / 0.790 | yes | 0.0424 | 251.5 | 252.5 / 252.5 | 0.00398 | - |
| n2 | lbfgs | measured | 2 | 0.9813 | 0.975 / 0.926 | yes | 0.0611 | 714.6 | 715.6 / 715.6 | 0.0014 | - |
| n2 | lbfgs | measured | 3 | 0.9994 | 0.995 / 0.985 | yes | 0.0794 | 1524 | 1524 / 1524 | 0.000656 | - |
| n2 | lbfgs | classical_first_step | 1 | 0.9636 | 0.924 / 0.790 | yes | 0.0424 | 1.038 | 2 / 2 | 0.964 | - |
| n2 | lbfgs | classical_first_step | 2 | 0.9812 | 0.975 / 0.927 | yes | 0.061 | 466 | 467 / 467 | 0.00215 | - |
| n2 | lbfgs | classical_first_step | 3 | 0.9994 | 0.995 / 0.985 | yes | 0.0795 | 1279 | 1279 / 1279 | 0.000782 | - |
| n2 | lbfgs | free | 1 | 0.9635 | 0.924 / 0.790 | yes | 0.0424 | 1.038 | 2 / 2 | 0.963 | - |
| n2 | lbfgs | free | 2 | 0.9813 | 0.975 / 0.926 | yes | 0.0611 | 1.019 | 2 / 2 | 0.981 | - |
| n2 | lbfgs | free | 3 | 0.9994 | 0.995 / 0.985 | yes | 0.0794 | 1.001 | 1 / 1 | 0.999 | - |
| n2 | lbfgs_so | all | 0 | 0.9257 | 0.593 / 0.257 | yes | 0 | 1.08 | 6 / 2 | 0.926 | - |
| n2 | lbfgs_so | measured | 1 | 0.9904 | 0.969 / 0.912 | yes | 0.0705 | 260 | 261 / 260 | 0.00385 | - |
| n2 | lbfgs_so | measured | 2 | 0.9979 | 0.993 / 0.980 | yes | 0.078 | 751.5 | 751.5 / 751.5 | 0.00133 | - |
| n2 | lbfgs_so | measured | 3 | 0.9995 | 0.997 / 0.990 | yes | 0.0796 | 1527 | 1527 / 1527 | 0.000655 | - |
| n2 | lbfgs_so | classical_first_step | 1 | 0.9901 | 0.970 / 0.915 | yes | 0.0703 | 1.01 | 2 / 1 | 0.99 | - |
| n2 | lbfgs_so | classical_first_step | 2 | 0.9979 | 0.993 / 0.980 | yes | 0.078 | 493.9 | 493.9 / 493.9 | 0.00202 | - |
| n2 | lbfgs_so | classical_first_step | 3 | 0.9995 | 0.997 / 0.990 | yes | 0.0796 | 1264 | 1264 / 1264 | 0.000791 | - |
| n2 | lbfgs_so | free | 1 | 0.9904 | 0.969 / 0.912 | yes | 0.0705 | 1.01 | 2 / 1 | 0.99 | - |
| n2 | lbfgs_so | free | 2 | 0.9979 | 0.993 / 0.980 | yes | 0.078 | 1.002 | 1 / 1 | 0.998 | - |
| n2 | lbfgs_so | free | 3 | 0.9995 | 0.997 / 0.990 | yes | 0.0796 | 1.001 | 1 / 1 | 0.999 | - |
| lih | lbfgs | all | 0 | 0.9743 | 0.389 / 0.389 | yes | 0 | 1.026 | 10 / 2 | 0.974 | - |
| lih | lbfgs | measured | 1 | 0.9940 | 0.598 / 0.598 | yes | 0.0203 | 156.2 | 161.1 / 156.1 | 0.0064 | - |
| lih | lbfgs | measured | 2 | 0.9957 | 0.956 / 0.956 | yes | 0.0218 | 967.8 | 968.8 / 967.8 | 0.00103 | - |
| lih | lbfgs | measured | 3 | 0.9963 | 0.972 / 0.972 | yes | 0.0221 | 3586 | 3587 / 3586 | 0.000279 | - |
| lih | lbfgs | classical_first_step | 1 | 0.9941 | 0.589 / 0.589 | yes | 0.0204 | 1.006 | 6 / 1 | 0.994 | - |
| lih | lbfgs | classical_first_step | 2 | 0.9958 | 0.955 / 0.955 | yes | 0.0218 | 809.5 | 810.5 / 809.5 | 0.00124 | - |
| lih | lbfgs | classical_first_step | 3 | 0.9963 | 0.972 / 0.972 | yes | 0.0221 | 3419 | 3420 / 3419 | 0.000292 | - |
| lih | lbfgs | free | 1 | 0.9940 | 0.598 / 0.598 | yes | 0.0203 | 1.006 | 6 / 1 | 0.994 | - |
| lih | lbfgs | free | 2 | 0.9957 | 0.956 / 0.956 | yes | 0.0218 | 1.005 | 2.001 / 1 | 0.995 | - |
| lih | lbfgs | free | 3 | 0.9963 | 0.972 / 0.972 | yes | 0.0221 | 1.004 | 2.001 / 1.001 | 0.996 | - |
| lih | lbfgs_so | all | 0 | 0.9743 | 0.389 / 0.389 | yes | 0 | 1.026 | 10 / 2 | 0.974 | - |
| lih | lbfgs_so | measured | 1 | 0.9940 | 0.599 / 0.599 | yes | 0.0203 | 154.9 | 159.9 / 154.9 | 0.00645 | - |
| lih | lbfgs_so | measured | 2 | 0.9956 | 0.962 / 0.962 | yes | 0.0217 | 849.2 | 850.2 / 849.2 | 0.00118 | - |
| lih | lbfgs_so | measured | 3 | 0.9966 | 0.971 / 0.971 | yes | 0.0224 | 3637 | 3638 / 3637 | 0.000275 | - |
| lih | lbfgs_so | classical_first_step | 1 | 0.9941 | 0.590 / 0.590 | yes | 0.0204 | 1.006 | 6 / 1 | 0.994 | - |
| lih | lbfgs_so | classical_first_step | 2 | 0.9956 | 0.962 / 0.962 | yes | 0.0217 | 690.6 | 691.6 / 690.6 | 0.00145 | - |
| lih | lbfgs_so | classical_first_step | 3 | 0.9966 | 0.972 / 0.972 | yes | 0.0224 | 3471 | 3472 / 3471 | 0.000288 | - |
| lih | lbfgs_so | free | 1 | 0.9940 | 0.599 / 0.599 | yes | 0.0203 | 1.006 | 6 / 1 | 0.994 | - |
| lih | lbfgs_so | free | 2 | 0.9956 | 0.962 / 0.962 | yes | 0.0217 | 1.005 | 2.001 / 1 | 0.995 | - |
| lih | lbfgs_so | free | 3 | 0.9966 | 0.971 / 0.971 | yes | 0.0224 | 1.004 | 2.001 / 1 | 0.996 | - |
| h6s | lbfgs | all | 0 | 0.5733 | - / - | no | 0 | 1.744 | - / 6 | 0.573 | - |
| h6s | lbfgs | measured | 1 | 0.8722 | 0.006 / 0.006 | yes | 0.598 | 554.1 | 1331 / 556 | 0.0018 | - |
| h6s | lbfgs | measured | 2 | 0.9470 | 0.339 / 0.339 | yes | 0.687 | 3893 | 3904 / 3894 | 0.000257 | - |
| h6s | lbfgs | measured | 3 | 0.9568 | 0.505 / 0.505 | yes | 0.697 | 1.132e4 | 1.133e4 / 1.132e4 | 8.83e-5 | - |
| h6s | lbfgs | classical_first_step | 1 | 0.8831 | 0.003 / 0.003 | yes | 0.612 | 1.132 | 1392 / 3 | 0.883 | - |
| h6s | lbfgs | classical_first_step | 2 | 0.9466 | 0.349 / 0.349 | yes | 0.687 | 3366 | 3376 / 3367 | 0.000297 | - |
| h6s | lbfgs | classical_first_step | 3 | 0.9553 | 0.534 / 0.534 | yes | 0.696 | 1.078e4 | 1.079e4 / 1.078e4 | 9.28e-5 | - |
| h6s | lbfgs | free | 1 | 0.8722 | 0.006 / 0.006 | yes | 0.598 | 1.147 | 778.1 / 3 | 0.872 | - |
| h6s | lbfgs | free | 2 | 0.9470 | 0.339 / 0.339 | yes | 0.687 | 1.057 | 12.01 / 2.002 | 0.946 | - |
| h6s | lbfgs | free | 3 | 0.9568 | 0.505 / 0.505 | yes | 0.697 | 1.047 | 7.013 / 2.004 | 0.955 | - |
| h6s | lbfgs_so | all | 0 | 0.5733 | - / - | no | 0 | 1.744 | - / 6 | 0.573 | - |
| h6s | lbfgs_so | measured | 1 | 0.9135 | 0.130 / 0.130 | yes | 0.65 | 517.1 | 549 / 518 | 0.00193 | - |
| h6s | lbfgs_so | measured | 2 | 0.9425 | 0.394 / 0.394 | yes | 0.682 | 3763 | 3772 / 3764 | 0.000266 | - |
| h6s | lbfgs_so | measured | 3 | 0.9633 | 0.640 / 0.640 | yes | 0.704 | 1.129e4 | 1.13e4 / 1.13e4 | 8.85e-5 | - |
| h6s | lbfgs_so | classical_first_step | 1 | 0.9165 | 0.062 / 0.062 | yes | 0.653 | 1.091 | 73.01 / 2 | 0.916 | - |
| h6s | lbfgs_so | classical_first_step | 2 | 0.9431 | 0.383 / 0.383 | yes | 0.683 | 3178 | 3187 / 3179 | 0.000315 | - |
| h6s | lbfgs_so | classical_first_step | 3 | 0.9597 | 0.667 / 0.667 | yes | 0.7 | 1.045e4 | 1.045e4 / 1.045e4 | 9.57e-5 | - |
| h6s | lbfgs_so | free | 1 | 0.9135 | 0.130 / 0.130 | yes | 0.65 | 1.095 | 33 / 2 | 0.913 | - |
| h6s | lbfgs_so | free | 2 | 0.9425 | 0.394 / 0.394 | yes | 0.682 | 1.062 | 10.01 / 2.002 | 0.942 | - |
| h6s | lbfgs_so | free | 3 | 0.9633 | 0.640 / 0.640 | yes | 0.704 | 1.04 | 5.009 / 2.004 | 0.962 | - |
| n2s | lbfgs | all | 0 | 0.2528 | - / - | no | 0 | 3.956 | - / 16 | 0.253 | - |
| n2s | lbfgs | measured | 1 | 0.2314 | - / - | no | -0.365 | 192.1 | - / 205.8 | 0.0052 | - |
| n2s | lbfgs | measured | 2 | 0.3178 | 0.001 / 0.001 | yes | 0.808 | 570.8 | 4018 / 580.7 | 0.00175 | - |
| n2s | lbfgs | measured | 3 | 0.6862 | 0.093 / 0.093 | yes | 2.5 | 1357 | 1403 / 1359 | 0.000737 | - |
| n2s | lbfgs | classical_first_step | 1 | 0.2433 | - / - | no | -0.155 | 4.111 | - / 17 | 0.243 | - |
| n2s | lbfgs | classical_first_step | 2 | 0.2726 | 0.000 / 0.000 | no | 0.287 | 378.5 | 6.431e4 / 389.8 | 0.00264 | - |
| n2s | lbfgs | classical_first_step | 3 | 0.7965 | 0.195 / 0.195 | yes | 2.7 | 1073 | 1094 / 1075 | 0.000932 | - |
| n2s | lbfgs | free | 1 | 0.2314 | - / - | no | -0.365 | 4.321 | - / 18 | 0.231 | - |
| n2s | lbfgs | free | 2 | 0.3178 | 0.001 / 0.001 | yes | 0.808 | 3.148 | 3451 / 13 | 0.318 | - |
| n2s | lbfgs | free | 3 | 0.6862 | 0.093 / 0.093 | yes | 2.5 | 1.458 | 48.01 / 4.001 | 0.686 | - |
| n2s | lbfgs_so | all | 0 | 0.2528 | - / - | no | 0 | 3.956 | - / 16 | 0.253 | - |
| n2s | lbfgs_so | measured | 1 | 0.4250 | 0.025 / 0.025 | yes | 1.6 | 188.7 | 371.3 / 195.3 | 0.0053 | - |
| n2s | lbfgs_so | measured | 2 | 0.5789 | 0.066 / 0.066 | yes | 2.23 | 573.5 | 639.8 / 577.8 | 0.00174 | - |
| n2s | lbfgs_so | measured | 3 | 0.8355 | 0.222 / 0.222 | yes | 2.76 | 1281 | 1299 / 1283 | 0.000781 | - |
| n2s | lbfgs_so | classical_first_step | 1 | 0.3961 | 0.007 / 0.007 | yes | 1.43 | 2.525 | 685 / 10 | 0.396 | - |
| n2s | lbfgs_so | classical_first_step | 2 | 0.5452 | 0.060 / 0.060 | yes | 2.12 | 384.2 | 457.4 / 388.4 | 0.0026 | - |
| n2s | lbfgs_so | classical_first_step | 3 | 0.8674 | 0.323 / 0.323 | yes | 2.8 | 1007 | 1018 / 1009 | 0.000993 | - |
| n2s | lbfgs_so | free | 1 | 0.4250 | 0.025 / 0.025 | yes | 1.6 | 2.353 | 185 / 9 | 0.425 | - |
| n2s | lbfgs_so | free | 2 | 0.5789 | 0.066 / 0.066 | yes | 2.23 | 1.728 | 68.01 / 6.001 | 0.579 | - |
| n2s | lbfgs_so | free | 3 | 0.8355 | 0.222 / 0.222 | yes | 2.76 | 1.197 | 19.01 / 3.001 | 0.835 | - |
| h8 | lbfgs | all | 0 | 0.8864 | 0.232 / 0.232 | yes | 0 | 1.128 | 18 / 3 | 0.886 | - |
| h8 | lbfgs | measured | 1 | 0.9454 | 0.619 / 0.619 | yes | 0.0703 | >= 1323 | >= 1327 / >= 1324 | <= 0.000756 | - |
| h8 | lbfgs | measured | 2 | 0.9750 | 0.840 / 0.840 | yes | 0.101 | >= 1.106e4 | >= 1.106e4 / >= 1.106e4 | <= 9.04e-5 | - |
| h8 | lbfgs | measured | 3 | 0.9826 | 0.916 / 0.916 | yes | 0.108 | >= 3.018e4 | >= 3.018e4 / >= 3.018e4 | <= 3.31e-5 | - |
| h8 | lbfgs | classical_first_step | 1 | 0.9458 | 0.616 / 0.616 | yes | 0.0708 | 1.057 | 5.001 / 2 | 0.946 | - |
| h8 | lbfgs | classical_first_step | 2 | 0.9749 | 0.841 / 0.841 | yes | 0.101 | >= 9778 | >= 9780 / >= 9779 | <= 0.000102 | - |
| h8 | lbfgs | classical_first_step | 3 | 0.9826 | 0.917 / 0.917 | yes | 0.108 | >= 2.874e4 | >= 2.874e4 / >= 2.874e4 | <= 3.48e-5 | - |
| h8 | lbfgs | free | 1 | 0.9454 | 0.619 / 0.619 | yes | 0.0703 | 1.058 | 5.001 / 2 | 0.945 | - |
| h8 | lbfgs | free | 2 | 0.9750 | 0.840 / 0.840 | yes | 0.101 | 1.027 | 3.003 / 2.002 | 0.974 | - |
| h8 | lbfgs | free | 3 | 0.9826 | 0.916 / 0.916 | yes | 0.108 | 1.02 | 2.004 / 2.004 | 0.981 | - |
| h8 | lbfgs_so | all | 0 | 0.8864 | 0.232 / 0.232 | yes | 0 | 1.128 | 18 / 3 | 0.886 | - |
| h8 | lbfgs_so | measured | 1 | 0.9671 | 0.799 / 0.799 | yes | 0.0941 | >= 1367 | >= 1369 / >= 1368 | <= 0.000732 | - |
| h8 | lbfgs_so | measured | 2 | 0.9853 | 0.927 / 0.927 | yes | 0.112 | >= 1.086e4 | >= 1.086e4 / >= 1.086e4 | <= 9.2e-5 | - |
| h8 | lbfgs_so | measured | 3 | 0.9909 | 0.950 / 0.950 | yes | 0.117 | >= 2.93e4 | >= 2.93e4 / >= 2.93e4 | <= 3.41e-5 | - |
| h8 | lbfgs_so | classical_first_step | 1 | 0.9677 | 0.790 / 0.790 | yes | 0.0947 | 1.033 | 3 / 2 | 0.968 | - |
| h8 | lbfgs_so | classical_first_step | 2 | 0.9853 | 0.927 / 0.927 | yes | 0.112 | >= 9660 | >= 9661 / >= 9661 | <= 0.000104 | - |
| h8 | lbfgs_so | classical_first_step | 3 | 0.9908 | 0.951 / 0.951 | yes | 0.117 | >= 2.846e4 | >= 2.846e4 / >= 2.846e4 | <= 3.51e-5 | - |
| h8 | lbfgs_so | free | 1 | 0.9671 | 0.799 / 0.799 | yes | 0.0941 | 1.034 | 3 / 2 | 0.967 | - |
| h8 | lbfgs_so | free | 2 | 0.9853 | 0.927 / 0.927 | yes | 0.112 | 1.016 | 2.002 / 2.002 | 0.984 | - |
| h8 | lbfgs_so | free | 3 | 0.9909 | 0.950 / 0.950 | yes | 0.117 | 1.011 | 2.004 / 1.002 | 0.989 | - |

**T6, optima per rule.**

k_opt per rule (T / C_QPE at k_opt); 'rows to k': the last iterate the loop covers:

| system | family | loop | rows to k | k_opt E certified | k_opt E oracle | k_opt delta certified | k_opt delta oracle | p0*(1) / p0*(2) / p0*(3) | verdict (rule E, certified) |
|---|---|---|---|---|---|---|---|---|---|
| h4 | lbfgs | measured | 8 | 0 (1.058) | 0 (1.058) | 0 (4) | 0 (2) | 0.0049 / 0.00101 / 0.000381 | HF |
| h4 | lbfgs | classical_first_step | 8 | 1 (1.02) | 1 (1.02) | 1 (2) | 0 (2) | 0.98 / 0.00128 / 0.000414 | Psi_1, T_E(0)/T_E(k_opt) = 1.037 |
| h4 | lbfgs | free | 18 | 4 (1.001) | 4 (1.001) | 3 (1.001) | 2 (1) | 0.98 / 0.996 / 0.997 | Psi_4, T_E(0)/T_E(k_opt) = 1.056 |
| h4 | lbfgs_so | measured | 8 | 0 (1.058) | 0 (1.058) | 0 (4) | 0 (2) | 0.00461 / 0.00101 / 0.000383 | HF |
| h4 | lbfgs_so | classical_first_step | 8 | 1 (1.009) | 1 (1.009) | 1 (2) | 1 (1) | 0.991 / 0.00128 / 0.000419 | Psi_1, T_E(0)/T_E(k_opt) = 1.048 |
| h4 | lbfgs_so | free | 15 | 4 (1.001) | 4 (1.001) | 2 (1) | 1 (1) | 0.991 / 0.998 / 0.999 | Psi_4, T_E(0)/T_E(k_opt) = 1.057 |
| h6a | lbfgs | measured | 11 | 0 (1.108) | 0 (1.108) | 0 (9) | 0 (2) | 0.00204 / 0.000271 / 9.04e-5 | HF |
| h6a | lbfgs | classical_first_step | 11 | 1 (1.044) | 1 (1.044) | 1 (4) | 0 (2) | 0.958 / 0.000313 / 9.48e-5 | Psi_1, T_E(0)/T_E(k_opt) = 1.061 |
| h6a | lbfgs | free | 20 | 6 (1.004) | 6 (1.004) | 7 (1.003) | 3 (1.001) | 0.958 / 0.984 / 0.989 | Psi_6, T_E(0)/T_E(k_opt) = 1.103 |
| h6a | lbfgs_so | measured | 9 | 0 (1.108) | 0 (1.108) | 0 (9) | 0 (2) | 0.00192 / 0.000266 / 7.61e-5 | HF |
| h6a | lbfgs_so | classical_first_step | 9 | 1 (1.023) | 1 (1.023) | 1 (3) | 0 (2) | 0.977 / 0.000309 / 7.95e-5 | Psi_1, T_E(0)/T_E(k_opt) = 1.083 |
| h6a | lbfgs_so | free | 20 | 5 (1.003) | 5 (1.003) | 5 (1.002) | 2 (1.001) | 0.977 / 0.991 / 0.995 | Psi_5, T_E(0)/T_E(k_opt) = 1.105 |
| n2 | lbfgs | measured | 8 | 0 (1.08) | 0 (1.08) | 0 (6) | 0 (2) | 0.00398 / 0.0014 / 0.000656 | HF |
| n2 | lbfgs | classical_first_step | 8 | 1 (1.038) | 1 (1.038) | 1 (2) | 0 (2) | 0.964 / 0.00215 / 0.000782 | Psi_1, T_E(0)/T_E(k_opt) = 1.041 |
| n2 | lbfgs | free | 20 | 4 (1) | 4 (1) | 3 (1) | 3 (1) | 0.963 / 0.981 / 0.999 | Psi_4, T_E(0)/T_E(k_opt) = 1.08 |
| n2 | lbfgs_so | measured | 8 | 0 (1.08) | 0 (1.08) | 0 (6) | 0 (2) | 0.00385 / 0.00133 / 0.000655 | HF |
| n2 | lbfgs_so | classical_first_step | 8 | 1 (1.01) | 1 (1.01) | 1 (2) | 1 (1) | 0.99 / 0.00202 / 0.000791 | Psi_1, T_E(0)/T_E(k_opt) = 1.07 |
| n2 | lbfgs_so | free | 20 | 4 (1) | 4 (1) | 2 (1) | 1 (1) | 0.99 / 0.998 / 0.999 | Psi_4, T_E(0)/T_E(k_opt) = 1.08 |
| lih | lbfgs | measured | 7 | 0 (1.026) | 0 (1.026) | 0 (10) | 0 (2) | 0.0064 / 0.00103 / 0.000279 | HF |
| lih | lbfgs | classical_first_step | 7 | 1 (1.006) | 1 (1.006) | 1 (6) | 1 (1) | 0.994 / 0.00124 / 0.000292 | Psi_1, T_E(0)/T_E(k_opt) = 1.02 |
| lih | lbfgs | free | 20 | 7 (1.002) | 7 (1.002) | 7 (1.002) | 1 (1) | 0.994 / 0.995 / 0.996 | Psi_7, T_E(0)/T_E(k_opt) = 1.025 |
| lih | lbfgs_so | measured | 7 | 0 (1.026) | 0 (1.026) | 0 (10) | 0 (2) | 0.00645 / 0.00118 / 0.000275 | HF |
| lih | lbfgs_so | classical_first_step | 7 | 1 (1.006) | 1 (1.006) | 1 (6) | 1 (1) | 0.994 / 0.00145 / 0.000288 | Psi_1, T_E(0)/T_E(k_opt) = 1.02 |
| lih | lbfgs_so | free | 20 | 7 (1.002) | 7 (1.002) | 6 (1.001) | 1 (1) | 0.994 / 0.995 / 0.996 | Psi_7, T_E(0)/T_E(k_opt) = 1.025 |
| h6s | lbfgs | measured | 17 | 1 (554.1) | 0 (1.744) | 1 (1331) | 0 (6) | 0.0018 / 0.000257 / 8.83e-5 | Psi_1, T_E(0)/T_E(k_opt) = 0.003148 (HF not certified) |
| h6s | lbfgs | classical_first_step | 17 | 1 (1.132) | 1 (1.132) | 1 (1392) | 1 (3) | 0.883 / 0.000297 / 9.28e-5 | Psi_1, T_E(0)/T_E(k_opt) = 1.54 (HF not certified) |
| h6s | lbfgs | free | 20 | 14 (1.014) | 14 (1.014) | 10 (2.016) | 11 (1.009) | 0.872 / 0.946 / 0.955 | Psi_14, T_E(0)/T_E(k_opt) = 1.72 (HF not certified) |
| h6s | lbfgs_so | measured | 15 | 1 (517.1) | 0 (1.744) | 1 (549) | 0 (6) | 0.00193 / 0.000266 / 8.85e-5 | Psi_1, T_E(0)/T_E(k_opt) = 0.003373 (HF not certified) |
| h6s | lbfgs_so | classical_first_step | 15 | 1 (1.091) | 1 (1.091) | 1 (73.01) | 1 (2) | 0.916 / 0.000315 / 9.57e-5 | Psi_1, T_E(0)/T_E(k_opt) = 1.599 (HF not certified) |
| h6s | lbfgs_so | free | 20 | 12 (1.012) | 12 (1.012) | 7 (2.011) | 9 (1.007) | 0.913 / 0.942 / 0.962 | Psi_12, T_E(0)/T_E(k_opt) = 1.723 (HF not certified) |
| n2s | lbfgs | measured | 20 | 2 (570.8) | 0 (3.956) | 3 (1403) | 0 (16) | 0.0052 / 0.00175 / 0.000737 | Psi_2, T_E(0)/T_E(k_opt) = 0.006931 (HF not certified) |
| n2s | lbfgs | classical_first_step | 20 | 3 (1073) | 0 (3.956) | 3 (1094) | 0 (16) | 0.243 / 0.00264 / 0.000932 | Psi_3, T_E(0)/T_E(k_opt) = 0.003687 (HF not certified) |
| n2s | lbfgs | free | 20 | 16 (1.012) | 16 (1.012) | 7 (3.002) | 6 (2.001) | 0.231 / 0.318 / 0.686 | Psi_16, T_E(0)/T_E(k_opt) = 3.909 (HF not certified) |
| n2s | lbfgs_so | measured | 20 | 1 (188.7) | 0 (3.956) | 1 (371.3) | 0 (16) | 0.0053 / 0.00174 / 0.000781 | Psi_1, T_E(0)/T_E(k_opt) = 0.02097 (HF not certified) |
| n2s | lbfgs_so | classical_first_step | 20 | 1 (2.525) | 1 (2.525) | 2 (457.4) | 1 (10) | 0.396 / 0.0026 / 0.000993 | Psi_1, T_E(0)/T_E(k_opt) = 1.567 (HF not certified) |
| n2s | lbfgs_so | free | 20 | 17 (1.013) | 17 (1.013) | 18 (4.008) | 4 (2.001) | 0.425 / 0.579 / 0.835 | Psi_17, T_E(0)/T_E(k_opt) = 3.907 (HF not certified) |
| h8 | lbfgs | measured | 14 | 0 (1.128) | 0 (1.128) | 0 (18) | 0 (3) | <= 0.000756 / <= 9.04e-5 / <= 3.31e-5 | HF (C_loop a lower bound from k = 1) |
| h8 | lbfgs | classical_first_step | 14 | 1 (1.057) | 1 (1.057) | 1 (5.001) | 1 (2) | 0.946 / <= 0.000102 / <= 3.48e-5 | Psi_1, T_E(0)/T_E(k_opt) = 1.067 (C_loop a lower bound from k = 2) |
| h8 | lbfgs | free | 20 | 9 (1.009) | 9 (1.009) | 13 (1.012) | 5 (1.004) | 0.945 / 0.974 / 0.981 | Psi_9, T_E(0)/T_E(k_opt) = 1.118 |
| h8 | lbfgs_so | measured | 11 | 0 (1.128) | 0 (1.128) | 0 (18) | 0 (3) | <= 0.000732 / <= 9.2e-5 / <= 3.41e-5 | HF (C_loop a lower bound from k = 1) |
| h8 | lbfgs_so | classical_first_step | 11 | 1 (1.033) | 1 (1.033) | 1 (3) | 1 (2) | 0.968 / <= 0.000104 / <= 3.51e-5 | Psi_1, T_E(0)/T_E(k_opt) = 1.092 (C_loop a lower bound from k = 2) |
| h8 | lbfgs_so | free | 20 | 6 (1.007) | 6 (1.007) | 11 (1.01) | 3 (1.002) | 0.967 / 0.984 / 0.989 | Psi_6, T_E(0)/T_E(k_opt) = 1.121 |

- **Equilibrium, measured loop: HF.** k_opt = 0 under all four rules on h4, h6a, n2, lih and
  h8.
  One measured iteration would pay only for an HF overlap below p₀\*(1) = 7.3e-4 (h8) to
  6.5e-3 (lih); two or three below 3.3e-5 to 1.4e-3. The saving of Ψ_k on the QPE side,
  C_QPE/p₀ − (C(Ψ_k) + C_QPE)/p_k, is bounded by C_QPE(1/p₀ − 1) = 0.026–0.128 C_QPE, while
  one measured iteration costs at least 153.9 C_QPE (lih, spin-orbital, at HF; 154 in T3).
- **Equilibrium, classical first step: Ψ₁.** Rule E is certified everywhere, and T_E(1) =
  1.006 (lih) to 1.057 C_QPE (h8) against T_E(0) = 1.026–1.128: a gain T_E(0)/T_E(1) of
  1.020–1.067 (spin-free) and 1.020–1.092 (spin-orbital). Under the certified rule δ the
  classical step cuts T_δ from 4–18 to 2–6 C_QPE (h6a: 9 to 4), because p_lb rises.
- **Free loop.** With free generators the L-BFGS optimum moves to k_opt = 4–9 at equilibrium
  (7–13 with steepest descent; `summary.families.sd.handoff.free` in
  `results/ft_<system>.json`), with T_E(k_opt) = 1.000–1.009 C_QPE: free generators recover
  nearly the whole budget (T_E(0)/T_E(k_opt) = 1.025 on lih to 1.121 on h8, L-BFGS).
- **h6s.** E_HF − E₀ = 0.282 Ha (T2) exceeds E₁ − E₀ = 0.105 Ha (`ft.trajectory` docstring),
  so rule E is not certified at HF; the oracle T_E(0) is 1.744 C_QPE, and no certified δ-rule
  run exists (p_lb undefined). The classical step reaches the certified Ψ₁ at T_E(1) = 1.132
  (spin-free) and 1.091 C_QPE (spin-orbital), the cheapest certified hand-off; the measured
  loop needs one measured iteration for it (554 and 517 C_QPE). Free generators would reach
  1.014 C_QPE at k = 14.
- **n2s.** E_HF − E₀ = 0.566 Ha (T2) against E₁ − E₀ = 0.192 Ha, so rule E certifies Ψ_k only
  for E_k − E₀ < 0.1887 Ha. The spin-free classical step (ε₂/ε\* = 0.80) leaves Ψ₁ and Ψ₂
  0.224 and 0.190 Ha above E₀, both uncertified, so its certified hand-off costs 1073 C_QPE at
  k = 3; the measured loop's Ψ₂ lies 0.183 Ha above E₀ and is certified at 571 C_QPE
  (`trajectories` in `results/ft_n2s.json`). Both first steps lower the overlap from 0.2528, to
  0.2433 (classical) and 0.2314 (exact), so the overlap does not decide it. The spin-orbital
  classical step certifies Ψ₁ (0.1735 Ha above E₀) at 2.525 C_QPE. The free loop reaches
  1.012–1.013 C_QPE at k = 16–17, a factor 3.9 below the uncertified HF hand-off.
- **Irrep of E₁.** F15 certifies with the same-irrep E₁; the any-irrep E₁ would weaken the
  certificate only for N2: p_lb(HF) = 0.257 against 0.593 with the same-irrep E₁, i.e. R_δ = 16
  against 6 runs.

**T7** (the classical first step, F16).

eps_2 = -E'(0)/E''(0) at HF along the masked d_0 (sd shares the spin-free step of lbfgs). p_1 (eps*): after the exact line search; p_1 (eps_2): after the classical step, exact unitary; p~_1: its compiled elementwise circuit (open-loop replay, rounded). Gain: T(0)/T(1) of the classical-first-step loop (C_loop(1) = 0) under rule E and rule delta.

| system | metric | eps_2 / eps* | E(eps_2) - E(eps*) (mHa) | p_HF | p_1 (eps*) | p_1 (eps_2) | p~_1 compiled | norm(Psi~_1 - Psi_1) | <S^2>(Psi~_1) | C_step(0) (n_d / n_s, b) | gain rule E (certified k = 0/1) | gain rule delta | gain rule delta, oracle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| h4 | spin_free | 1.0159 | 0.0103 | 0.9455 | 0.98023 | 0.98039 | 0.98029 | 0.00118 | 2.1e-6 | 160 (10 / 0, b = 15) | 1.0368 (yes/yes) | 2 | 1 |
| h4 | spin_orbital | 1.0251 | 0.0338 | 0.9455 | 0.99114 | 0.99139 | 0.99135 | 0.00153 | 4.8e-6 | 160 (10 / 0, b = 15) | 1.0484 (yes/yes) | 2 | 2 |
| h6a | spin_free | 1.0220 | 0.0303 | 0.9026 | 0.95763 | 0.95801 | 0.95782 | 0.00149 | 1.2e-6 | 969 (51 / 0, b = 18) | 1.0613 (yes/yes) | 2.25 | 0.9999 |
| h6a | spin_orbital | 1.0294 | 0.0724 | 0.9026 | 0.97685 | 0.97744 | 0.97732 | 0.0028 | 8.7e-6 | 969 (51 / 0, b = 18) | 1.0828 (yes/yes) | 3 | 0.9999 |
| n2 | spin_free | 0.9805 | 0.0338 | 0.9257 | 0.96350 | 0.96357 | 0.96333 | 0.00189 | 1.4e-5 | 486 (27 / 0, b = 17) | 1.0409 (yes/yes) | 3 | 1 |
| n2 | spin_orbital | 0.9785 | 0.0515 | 0.9257 | 0.99038 | 0.99012 | 0.99005 | 0.00277 | 3.7e-6 | 486 (27 / 0, b = 17) | 1.0696 (yes/yes) | 3 | 2 |
| lih | spin_free | 1.0192 | 0.00628 | 0.9743 | 0.99401 | 0.99411 | 0.99409 | 0.000236 | 1.7e-10 | 468 (26 / 0, b = 17) | 1.0203 (yes/yes) | 1.667 | 2 |
| lih | spin_orbital | 1.0193 | 0.00636 | 0.9743 | 0.99404 | 0.99414 | 0.99413 | 0.000237 | 1.7e-10 | 468 (26 / 0, b = 17) | 1.0203 (yes/yes) | 1.667 | 2 |
| h6s | spin_free | 1.1371 | 3.68 | 0.5733 | 0.87223 | 0.88310 | 0.88195 | 0.0397 | 0.0032 | 969 (51 / 0, b = 18) | 1.5403 (no/yes) | - | 2 |
| h6s | spin_orbital | 1.2028 | 10.3 | 0.5733 | 0.91355 | 0.91650 | 0.91461 | 0.0713 | 0.0029 | 969 (51 / 0, b = 18) | 1.5985 (no/yes) | - | 3 |
| n2s | spin_free | 0.7955 | 11.9 | 0.2528 | 0.23144 | 0.24327 | 0.23971 | 0.0161 | 0.0011 | 486 (27 / 0, b = 17) | 0.9623 (no/no) | - | 0.9411 |
| n2s | spin_orbital | 0.7619 | 18.5 | 0.2528 | 0.42503 | 0.39609 | 0.40270 | 0.0174 | 0.00035 | 486 (27 / 0, b = 17) | 1.5669 (no/yes) | - | 1.6 |
| h8 | spin_free | 1.0216 | 0.0352 | 0.8864 | 0.94542 | 0.94581 | 0.94563 | 0.00135 | 1.4e-6 | 3528 (168 / 0, b = 20) | 1.0669 (yes/yes) | 3.6 | 1.5 |
| h8 | spin_orbital | 1.0258 | 0.0666 | 0.8864 | 0.96710 | 0.96772 | 0.96761 | 0.00241 | 7.5e-6 | 3528 (168 / 0, b = 20) | 1.0916 (yes/yes) | 5.999 | 1.5 |

- ε₂/ε\* is 0.98–1.03 at equilibrium (both metrics), and the classical step loses at most
  0.072 mHa against the exact line search; on the stretched systems it loses 3.7–18.5 mHa
  (ε₂/ε\* = 0.76–1.20).
- The compiled circuit of the classical step stays within 2.8e-3 of Ψ₁ at equilibrium
  (⟨S²⟩ ≤ 1.4e-5, p̃₁ within 2.4e-4 of p₁), but errs by 1.6e-2 to 7.1e-2 on h6s and n2s with
  ⟨S²⟩ = 3.5e-4 to 3.2e-3: the product-formula error of F3, not in the δ_step budget.
- The step costs 160 (h4) to 3528 Toffolis (h8), negligible against C_QPE.

## 6. The CQE energy alone (T8)

**T8** (the CQE-only energy, F17).

k_v: first iterate within 1 mHa of E_0; then V_E / sigma^2 shots at sigma = 0.3 mHa (exact integrals, DF grouping and readout rotation). HF + QPE: 1/p_0 QPE runs on average (budget 1e-2 per run). k_v '-': not reached in the run (n2s), E_kv - E_0 is then the error of the last iterate and F17 has no cost. '>=': line search only (h8 without the residual).

| system | family | loop | k_v | E_kv - E_0 (mHa) | shots | final shots | Toffolis | Toffolis / (C_QPE / p_0) | wall clock, one machine | HF + QPE, one machine | wall-clock ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| h4 | lbfgs | classical_first_step | 4 | 0.409 | 1.5e7 | 6.6e6 | 4.5e10 | 1.08e4 | 15 d | 2 min | 1.07e4 |
| h4 | lbfgs | measured | 4 | 0.410 | 1.7e7 | 6.6e6 | 4.6e10 | 1.09e4 | 15 d | 2 min | 1.09e4 |
| h4 | lbfgs_so | classical_first_step | 3 | 0.345 | 1.2e7 | 6.6e6 | 2.7e10 | 6.38e3 | 8.8 d | 2 min | 6.33e3 |
| h4 | lbfgs_so | measured | 3 | 0.344 | 1.5e7 | 6.6e6 | 2.8e10 | 6.6e3 | 9.1 d | 2 min | 6.53e3 |
| h4 | sd | classical_first_step | 6 | 0.602 | 1.9e7 | 6.6e6 | 9.1e10 | 2.17e4 | 31 d | 2 min | 2.18e4 |
| h4 | sd | measured | 6 | 0.632 | 2.2e7 | 6.6e6 | 9.2e10 | 2.19e4 | 31 d | 2 min | 2.2e4 |
| h6a | lbfgs | classical_first_step | 7 | 0.764 | 4.8e7 | 1.4e7 | 2.1e12 | 1.25e5 | 7.4e2 d | 7.9 min | 1.34e5 |
| h6a | lbfgs | measured | 7 | 0.764 | 5.3e7 | 1.4e7 | 2.1e12 | 1.26e5 | 7.4e2 d | 7.9 min | 1.35e5 |
| h6a | lbfgs_so | classical_first_step | 5 | 0.705 | 4.2e7 | 1.4e7 | 1.2e12 | 7.1e4 | 4.1e2 d | 7.9 min | 7.57e4 |
| h6a | lbfgs_so | measured | 5 | 0.705 | 4.8e7 | 1.4e7 | 1.2e12 | 7.16e4 | 4.2e2 d | 7.9 min | 7.63e4 |
| h6a | sd | classical_first_step | 18 | 0.976 | 1e8 | 1.4e7 | 1.2e13 | 7.31e5 | 4.8e3 d | 7.9 min | 8.72e5 |
| h6a | sd | measured | 18 | 0.983 | 1.1e8 | 1.4e7 | 1.2e13 | 7.3e5 | 4.8e3 d | 7.9 min | 8.7e5 |
| n2 | lbfgs | classical_first_step | 4 | 0.397 | 2e7 | 9.2e6 | 6.8e10 | 5e3 | 23 d | 6.5 min | 5.01e3 |
| n2 | lbfgs | measured | 4 | 0.399 | 2.3e7 | 9.2e6 | 7.1e10 | 5.22e3 | 24 d | 6.5 min | 5.22e3 |
| n2 | lbfgs_so | classical_first_step | 3 | 0.749 | 1.6e7 | 9.2e6 | 4.3e10 | 3.14e3 | 14 d | 6.5 min | 3.15e3 |
| n2 | lbfgs_so | measured | 3 | 0.748 | 2e7 | 9.1e6 | 4.7e10 | 3.41e3 | 15 d | 6.5 min | 3.39e3 |
| n2 | sd | classical_first_step | 8 | 0.981 | 3.3e7 | 8.9e6 | 2e11 | 1.44e4 | 66 d | 6.5 min | 1.45e4 |
| n2 | sd | measured | 9 | 0.731 | 4e7 | 9.1e6 | 2.5e11 | 1.8e4 | 83 d | 6.5 min | 1.82e4 |
| lih | lbfgs | classical_first_step | 3 | 0.796 | 1.6e7 | 6.7e6 | 1.1e11 | 7e3 | 40 d | 7.8 min | 7.46e3 |
| lih | lbfgs | measured | 3 | 0.797 | 2.2e7 | 6.7e6 | 1.2e11 | 7.17e3 | 41 d | 7.8 min | 7.62e3 |
| lih | lbfgs_so | classical_first_step | 2 | 0.961 | 1.4e7 | 6.9e6 | 4.1e10 | 2.55e3 | 14 d | 7.8 min | 2.54e3 |
| lih | lbfgs_so | measured | 2 | 0.958 | 2e7 | 6.9e6 | 4.4e10 | 2.71e3 | 14 d | 7.8 min | 2.68e3 |
| lih | sd | classical_first_step | 4 | 0.893 | 1.9e7 | 6.8e6 | 2.1e11 | 1.32e4 | 78 d | 7.8 min | 1.44e4 |
| lih | sd | measured | 4 | 0.931 | 2.5e7 | 6.8e6 | 2.1e11 | 1.32e4 | 78 d | 7.8 min | 1.44e4 |
| h6s | lbfgs | classical_first_step | 15 | 0.861 | 5.3e7 | 7.9e6 | 8.1e12 | 4.8e5 | 3.7e3 d | 8.1 min | 6.48e5 |
| h6s | lbfgs | measured | 14 | 0.970 | 5.4e7 | 7.9e6 | 6.7e12 | 3.95e5 | 3e3 d | 8.1 min | 5.24e5 |
| h6s | lbfgs_so | classical_first_step | 13 | 0.913 | 4.6e7 | 7.9e6 | 6.3e12 | 3.73e5 | 2.8e3 d | 8.1 min | 5.02e5 |
| h6s | lbfgs_so | measured | 13 | 0.902 | 5.1e7 | 7.9e6 | 6.4e12 | 3.75e5 | 2.9e3 d | 8.1 min | 5.05e5 |
| n2s | lbfgs | classical_first_step | - | 5.186 | - | - | - | - | - | 18 min | - |
| n2s | lbfgs | measured | - | 5.710 | - | - | - | - | - | 18 min | - |
| n2s | lbfgs_so | classical_first_step | - | 2.611 | - | - | - | - | - | 18 min | - |
| n2s | lbfgs_so | measured | - | 7.567 | - | - | - | - | - | 18 min | - |
| h8 | lbfgs | classical_first_step | 11 | 0.861 | >= 1.2e8 | 2.4e7 | >= 2.3e13 | >= 6.37e5 | >= 8.7e3 d | 17 min | >= 7.4e5 |
| h8 | lbfgs | measured | 11 | 0.860 | >= 1.3e8 | 2.4e7 | >= 2.3e13 | >= 6.39e5 | >= 8.7e3 d | 17 min | >= 7.42e5 |
| h8 | lbfgs_so | classical_first_step | 8 | 0.874 | >= 8.8e7 | 2.4e7 | >= 1.3e13 | >= 3.66e5 | >= 5e3 d | 17 min | >= 4.22e5 |
| h8 | lbfgs_so | measured | 8 | 0.875 | >= 9.8e7 | 2.4e7 | >= 1.3e13 | >= 3.64e5 | >= 4.9e3 d | 17 min | >= 4.19e5 |

- Reaching 1 mHa (k_v) and measuring the energy to σ = 0.3 mHa costs 2.55e3 (lih,
  spin-orbital) to 7.3e5 (h6a, steepest descent) times the HF hand-off C_QPE/p₀, and ≥ 3.6e5
  on h8. The final energy alone needs 6.6e6–2.4e7 shots.
- On one machine this takes 14 days (lih, spin-orbital) to 13 years (h6a, steepest
  descent), and at least 13.5 (spin-orbital) to 24 years (spin-free) on h8, where only the line
  search is costed, against 2–17 min for HF + QPE. n2s does not reach 1 mHa in 20 steps.

## 7. Spin-free against spin-orbital L-BFGS

From T2, T4, T6 and T7 (k\*, C(Ψ_k\*) in Toffolis, C_loop(k\*) of the classical-first-step
loop, and the certified optimum k_opt of rule E with its T_E in C_QPE):

| system | k\* sf / so | C(Ψ_k\*) sf / so | C_loop(k\*) sf / so | classical step: k_opt, T_E sf / so | free loop: k_opt, T_E sf / so |
|---|---|---|---|---|---|
| h4 | 3 / 3 | 2506 / 2468 | 2.42e3 / 2.38e3 | 1, 1.02 / 1, 1.009 | 4, 1.001 / 4, 1.001 |
| h6a | 6 / 4 | 43,427 / 26,547 | 6.33e4 / 2.52e4 | 1, 1.044 / 1, 1.023 | 6, 1.004 / 5, 1.003 |
| n2 | 3 / 3 | 2576 / 2576 | 1.28e3 / 1.26e3 | 1, 1.038 / 1, 1.01 | 4, 1.000 / 4, 1.000 |
| lih | 2 / 2 | 4648 / 4038 | 808 / 690 | 1, 1.006 / 1, 1.006 | 7, 1.002 / 7, 1.002 |
| h6s | 12 / 10 | 94,623 / 77,593 | 4.05e5 / 2.55e5 | 1, 1.132 / 1, 1.091 | 14, 1.014 / 12, 1.012 |
| n2s | – / – | – | – | 3, 1073 / 1, 2.525 | 16, 1.012 / 17, 1.013 |
| h8 | 9 / 6 | 245,530 / 154,626 | ≥ 3.21e5 / ≥ 1.35e5 | 1, 1.057 / 1, 1.033 | 9, 1.009 / 6, 1.007 |

- **Steps and state preparation.** The standard CQE generator needs fewer L-BFGS steps on h6a,
  h6s and h8 and the same elsewhere; with the same per-step cost, C(Ψ_k\*) falls in proportion
  (0.61× on h6a, 0.63× on h8).
- **Per-iteration cost.** It is similar at k = 1 (T3: h6a C_iter(1) = 3.24e3 against 3.20e3
  C_QPE) and 1.27× the spin-free cost at k = 2 on h6a (9.37e3 against 7.38e3); the totals fall
  with the number of steps, to 0.40× (h6a), 0.42× (h8) and 0.63× (h6s).
- **Hand-off.** Its first step lowers the energy more (h6a: 17.0 against 38.0 mHa after the
  exact first step, T2), so its classical first step saves more (h6a T_E(1) = 1.023 against 1.044
  C_QPE) and certifies Ψ₁ on n2s. At equilibrium the verdicts are the same for both metrics:
  HF for the measured loop, Ψ₁ for the classical first step.

## 8. Validation (T9) and the Appendix A regressions

**T9** (validation: replay).

Open-loop replay of the compiled elementwise circuits (rounded, F3) of the exact L-BFGS records (testing.fock oracle): state error, <S^2> (0 for the exact iterates) and overlaps, to k* + 2:

| system | family | k | C(Psi_k) | norm(Psi~_k - Psi_k) | <S^2>(Psi~_k) | p_k | p~_k |
|---|---|---|---|---|---|---|---|
| h4 | lbfgs | 0 | 0 | 0 | 0 | 0.94554 | 0.94554 |
| h4 | lbfgs | 1 | 160 | 0.00112 | 2e-6 | 0.98023 | 0.98013 |
| h4 | lbfgs | 2 | 1352 | 0.00117 | 2e-6 | 0.99600 | 0.99598 |
| h4 | lbfgs | 3* | 2506 | 0.00125 | 1.9e-6 | 0.99811 | 0.99811 |
| h4 | lbfgs | 4 | 3698 | 0.00126 | 1.7e-6 | 0.99969 | 0.99969 |
| h4 | lbfgs | 5 | 4890 | 0.00127 | 1.6e-6 | 0.99996 | 0.99997 |
| h4 | lbfgs_so | 0 | 0 | 0 | 0 | 0.94554 | 0.94554 |
| h4 | lbfgs_so | 1 | 160 | 0.00154 | 4.1e-6 | 0.99114 | 0.99109 |
| h4 | lbfgs_so | 2 | 1276 | 0.00176 | 3.7e-6 | 0.99809 | 0.99811 |
| h4 | lbfgs_so | 3* | 2468 | 0.0018 | 3.8e-6 | 0.99961 | 0.99959 |
| h4 | lbfgs_so | 4 | 3660 | 0.0018 | 3.8e-6 | 0.99994 | 0.99993 |
| h4 | lbfgs_so | 5 | 4852 | 0.00179 | 3.8e-6 | 0.99998 | 0.99998 |
| h6a | lbfgs | 0 | 0 | 0 | 0 | 0.90259 | 0.90259 |
| h6a | lbfgs | 1 | 969 | 0.0014 | 1e-6 | 0.95763 | 0.95745 |
| h6a | lbfgs | 2 | 9495 | 0.00188 | 4.4e-6 | 0.98496 | 0.98494 |
| h6a | lbfgs | 3 | 17977 | 0.00207 | 5e-6 | 0.99069 | 0.99071 |
| h6a | lbfgs | 4 | 26459 | 0.00213 | 5e-6 | 0.99503 | 0.99504 |
| h6a | lbfgs | 5 | 34985 | 0.00218 | 5.1e-6 | 0.99744 | 0.99747 |
| h6a | lbfgs | 6* | 43427 | 0.00219 | 5.2e-6 | 0.99872 | 0.99873 |
| h6a | lbfgs | 7 | 51865 | 0.0022 | 5.2e-6 | 0.99927 | 0.99927 |
| h6a | lbfgs | 8 | 60347 | 0.00222 | 5.3e-6 | 0.99970 | 0.99969 |
| h6a | lbfgs_so | 0 | 0 | 0 | 0 | 0.90259 | 0.90259 |
| h6a | lbfgs_so | 1 | 969 | 0.00257 | 7.2e-6 | 0.97685 | 0.97673 |
| h6a | lbfgs_so | 2 | 9495 | 0.00301 | 8.6e-6 | 0.99137 | 0.99145 |
| h6a | lbfgs_so | 3 | 18021 | 0.00308 | 8.4e-6 | 0.99605 | 0.99608 |
| h6a | lbfgs_so | 4* | 26547 | 0.0031 | 8.5e-6 | 0.99852 | 0.99851 |
| h6a | lbfgs_so | 5 | 35073 | 0.00312 | 8.5e-6 | 0.99933 | 0.99931 |
| h6a | lbfgs_so | 6 | 43557 | 0.00313 | 8.5e-6 | 0.99967 | 0.99964 |
| n2 | lbfgs | 0 | 0 | 0 | 0 | 0.92569 | 0.92569 |
| n2 | lbfgs | 1 | 486 | 0.00198 | 1.5e-5 | 0.96350 | 0.96324 |
| n2 | lbfgs | 2 | 1531 | 0.00272 | 1.1e-5 | 0.98130 | 0.98125 |
| n2 | lbfgs | 3* | 2576 | 0.00306 | 1.5e-5 | 0.99937 | 0.99938 |
| n2 | lbfgs | 4 | 3621 | 0.00312 | 1.6e-5 | 0.99982 | 0.99982 |
| n2 | lbfgs | 5 | 4628 | 0.0031 | 1.5e-5 | 0.99986 | 0.99985 |
| n2 | lbfgs_so | 0 | 0 | 0 | 0 | 0.92569 | 0.92569 |
| n2 | lbfgs_so | 1 | 486 | 0.00295 | 3.6e-6 | 0.99038 | 0.99030 |
| n2 | lbfgs_so | 2 | 1531 | 0.00306 | 3.3e-6 | 0.99785 | 0.99796 |
| n2 | lbfgs_so | 3* | 2576 | 0.00305 | 3.2e-6 | 0.99955 | 0.99959 |
| n2 | lbfgs_so | 4 | 3545 | 0.00304 | 3.2e-6 | 0.99984 | 0.99983 |
| n2 | lbfgs_so | 5 | 4362 | 0.00303 | 3.2e-6 | 0.99986 | 0.99984 |
| lih | lbfgs | 0 | 0 | 0 | 0 | 0.97434 | 0.97434 |
| lih | lbfgs | 1 | 468 | 0.000218 | 1.5e-10 | 0.99401 | 0.99399 |
| lih | lbfgs | 2* | 4648 | 0.000225 | 2e-10 | 0.99574 | 0.99573 |
| lih | lbfgs | 3 | 8468 | 0.000231 | 2e-10 | 0.99629 | 0.99628 |
| lih | lbfgs | 4 | 12574 | 0.000235 | 7.6e-10 | 0.99777 | 0.99776 |
| lih | lbfgs_so | 0 | 0 | 0 | 0 | 0.97434 | 0.97434 |
| lih | lbfgs_so | 1 | 468 | 0.000246 | 9.2e-9 | 0.99404 | 0.99402 |
| lih | lbfgs_so | 2* | 4038 | 0.000267 | 1.1e-8 | 0.99561 | 0.99560 |
| lih | lbfgs_so | 3 | 7852 | 0.000269 | 1.4e-8 | 0.99663 | 0.99661 |
| lih | lbfgs_so | 4 | 11788 | 0.000275 | 1.4e-8 | 0.99811 | 0.99810 |
| h6s | lbfgs | 0 | 0 | 0 | 0 | 0.57328 | 0.57328 |
| h6s | lbfgs | 1 | 969 | 0.0274 | 0.0015 | 0.87223 | 0.87015 |
| h6s | lbfgs | 2 | 9495 | 0.0299 | 0.0014 | 0.94699 | 0.94842 |
| h6s | lbfgs | 3 | 18021 | 0.0301 | 0.0014 | 0.95677 | 0.95800 |
| h6s | lbfgs | 4 | 26547 | 0.0302 | 0.0014 | 0.96286 | 0.96326 |
| h6s | lbfgs | 5 | 35029 | 0.0301 | 0.0014 | 0.96764 | 0.96823 |
| h6s | lbfgs | 6 | 43511 | 0.0301 | 0.0014 | 0.97699 | 0.97831 |
| h6s | lbfgs | 7 | 52037 | 0.0302 | 0.0014 | 0.98270 | 0.98342 |
| h6s | lbfgs | 8 | 60519 | 0.0302 | 0.0014 | 0.98452 | 0.98513 |
| h6s | lbfgs | 9 | 69045 | 0.0302 | 0.0014 | 0.98590 | 0.98663 |
| h6s | lbfgs | 10 | 77571 | 0.0302 | 0.0014 | 0.98932 | 0.98988 |
| h6s | lbfgs | 11 | 86097 | 0.0302 | 0.0014 | 0.99138 | 0.99176 |
| h6s | lbfgs | 12* | 94623 | 0.0303 | 0.0014 | 0.99444 | 0.99456 |
| h6s | lbfgs | 13 | 103149 | 0.0302 | 0.0014 | 0.99606 | 0.99552 |
| h6s | lbfgs | 14 | 111631 | 0.0302 | 0.0014 | 0.99732 | 0.99656 |
| h6s | lbfgs_so | 0 | 0 | 0 | 0 | 0.57328 | 0.57328 |
| h6s | lbfgs_so | 1 | 969 | 0.0422 | 0.001 | 0.91355 | 0.90924 |
| h6s | lbfgs_so | 2 | 9495 | 0.0435 | 0.00094 | 0.94252 | 0.94315 |
| h6s | lbfgs_so | 3 | 17977 | 0.0442 | 0.00098 | 0.96332 | 0.96324 |
| h6s | lbfgs_so | 4 | 26503 | 0.0444 | 0.001 | 0.97202 | 0.97158 |
| h6s | lbfgs_so | 5 | 35029 | 0.0444 | 0.001 | 0.97762 | 0.97765 |
| h6s | lbfgs_so | 6 | 43555 | 0.0444 | 0.001 | 0.98134 | 0.98137 |
| h6s | lbfgs_so | 7 | 52081 | 0.0445 | 0.001 | 0.98534 | 0.98482 |
| h6s | lbfgs_so | 8 | 60607 | 0.0445 | 0.001 | 0.98920 | 0.98901 |
| h6s | lbfgs_so | 9 | 69133 | 0.0445 | 0.001 | 0.99105 | 0.99090 |
| h6s | lbfgs_so | 10* | 77593 | 0.0446 | 0.001 | 0.99364 | 0.99283 |
| h6s | lbfgs_so | 11 | 86119 | 0.0445 | 0.001 | 0.99621 | 0.99498 |
| h6s | lbfgs_so | 12 | 94645 | 0.0445 | 0.001 | 0.99735 | 0.99576 |
| n2s | lbfgs | 0 | 0 | 0 | 0 | 0.25278 | 0.25278 |
| n2s | lbfgs | 1 | 486 | 0.0313 | 0.0039 | 0.23144 | 0.22507 |
| n2s | lbfgs | 2 | 1531 | 0.0353 | 0.0042 | 0.31776 | 0.31317 |
| n2s | lbfgs | 3 | 2576 | 0.0654 | 0.0067 | 0.68623 | 0.66424 |
| n2s | lbfgs | 4 | 3621 | 0.087 | 0.0097 | 0.84174 | 0.86057 |
| n2s | lbfgs | 5 | 4666 | 0.0866 | 0.0065 | 0.86092 | 0.86980 |
| n2s | lbfgs | 6 | 5711 | 0.0951 | 0.008 | 0.91666 | 0.93025 |
| n2s | lbfgs | 7 | 6756 | 0.0943 | 0.0078 | 0.93182 | 0.94840 |
| n2s | lbfgs | 8 | 7801 | 0.0918 | 0.0078 | 0.94179 | 0.95445 |
| n2s | lbfgs_so | 0 | 0 | 0 | 0 | 0.25278 | 0.25278 |
| n2s | lbfgs_so | 1 | 486 | 0.0382 | 0.0016 | 0.42503 | 0.44168 |
| n2s | lbfgs_so | 2 | 1455 | 0.0426 | 0.0016 | 0.57887 | 0.59797 |
| n2s | lbfgs_so | 3 | 2500 | 0.0648 | 0.0011 | 0.83553 | 0.83231 |
| n2s | lbfgs_so | 4 | 3545 | 0.0878 | 0.0011 | 0.91829 | 0.92929 |
| n2s | lbfgs_so | 5 | 4590 | 0.0886 | 0.0014 | 0.92623 | 0.93501 |
| n2s | lbfgs_so | 6 | 5635 | 0.0894 | 0.0013 | 0.93904 | 0.94813 |
| n2s | lbfgs_so | 7 | 6604 | 0.0889 | 0.0013 | 0.94556 | 0.95137 |
| n2s | lbfgs_so | 8 | 7649 | 0.089 | 0.0013 | 0.95490 | 0.96111 |
| h8 | lbfgs | 0 | 0 | 0 | 0 | 0.88638 | 0.88638 |
| h8 | lbfgs | 1 | 3528 | 0.00127 | 1.2e-6 | 0.94542 | 0.94525 |
| h8 | lbfgs | 2 | 33874 | 0.00171 | 1.1e-6 | 0.97498 | 0.97493 |
| h8 | lbfgs | 3 | 64220 | 0.00204 | 1.6e-6 | 0.98265 | 0.98258 |
| h8 | lbfgs | 4 | 94522 | 0.00205 | 1.5e-6 | 0.98615 | 0.98609 |
| h8 | lbfgs | 5 | 124822 | 0.00207 | 1.7e-6 | 0.99062 | 0.99058 |
| h8 | lbfgs | 6 | 154942 | 0.0021 | 1.6e-6 | 0.99374 | 0.99373 |
| h8 | lbfgs | 7 | 185106 | 0.00211 | 1.6e-6 | 0.99538 | 0.99538 |
| h8 | lbfgs | 8 | 215318 | 0.00211 | 1.6e-6 | 0.99690 | 0.99687 |
| h8 | lbfgs | 9* | 245530 | 0.00216 | 1.6e-6 | 0.99846 | 0.99843 |
| h8 | lbfgs | 10 | 275650 | 0.00218 | 1.7e-6 | 0.99907 | 0.99907 |
| h8 | lbfgs | 11 | 305674 | 0.00218 | 1.7e-6 | 0.99930 | 0.99929 |
| h8 | lbfgs_so | 0 | 0 | 0 | 0 | 0.88638 | 0.88638 |
| h8 | lbfgs_so | 1 | 3528 | 0.00224 | 6.5e-6 | 0.96710 | 0.96699 |
| h8 | lbfgs_so | 2 | 33738 | 0.0027 | 7.9e-6 | 0.98530 | 0.98540 |
| h8 | lbfgs_so | 3 | 63994 | 0.00275 | 8e-6 | 0.99085 | 0.99091 |
| h8 | lbfgs_so | 4 | 94294 | 0.00279 | 8e-6 | 0.99535 | 0.99538 |
| h8 | lbfgs_so | 5 | 124370 | 0.0028 | 8e-6 | 0.99733 | 0.99734 |
| h8 | lbfgs_so | 6* | 154626 | 0.00281 | 8e-6 | 0.99842 | 0.99842 |
| h8 | lbfgs_so | 7 | 184834 | 0.00282 | 8e-6 | 0.99901 | 0.99901 |
| h8 | lbfgs_so | 8 | 214858 | 0.00282 | 8.1e-6 | 0.99933 | 0.99931 |

**T9** (validation: per system).

Per system: rounding part of the replay error against sum_j A_j pi / 2^b_j (first family), the norm deficit (leakage out of (n_alpha, n_beta)), E''(0) against Richardson differences (relative), the F3 rounding flag on line-search candidates:

| system | rounding part, worst ratio (difference <= bound) | norm deficit | E''(0) vs finite diff. | line-search candidates flagged | QR clamp (selected) |
|---|---|---|---|---|---|
| h4 | 0.00021 <= 0.00096 (k = 1) | 4.4e-16 | lbfgs 6.1e-10; lbfgs_so 5.0e-10 | 12 of 286 (10 of 26 iterations) | yes |
| h6a | 4.7e-5 <= 0.00061 (k = 1) | 2.1e-15 | lbfgs 1.6e-11; lbfgs_so 3.4e-11 | 22 of 429 (20 of 39 iterations) | no |
| n2 | 7.3e-5 <= 0.00065 (k = 1) | 1.6e-15 | lbfgs 5.5e-11; lbfgs_so 8.6e-11 | 20 of 308 (14 of 28 iterations) | no |
| lih | 7e-5 <= 0.00062 (k = 1) | 8.9e-16 | lbfgs 2.1e-9; lbfgs_so 1.7e-8 | 31 of 231 (16 of 21 iterations) | no |
| h6s | 4.5e-5 <= 0.00061 (k = 1) | 8.9e-16 | lbfgs 3.6e-11; lbfgs_so 1.5e-9 | 4 of 352 (4 of 32 iterations) | no |
| n2s | 5.3e-5 <= 0.00065 (k = 1) | 1.0e-15 | lbfgs 8.1e-10; lbfgs_so 1.2e-11 | 18 of 440 (14 of 40 iterations) | no |
| h8 | 2.1e-5 <= 0.0005 (k = 1) | 6.4e-15 | lbfgs 1.6e-10; lbfgs_so 4.0e-11 | 22 of 275 (16 of 25 iterations) | no |

**T9** (validation: model checks).

Model checks (system independent):

| check | result |
|---|---|
| walk moments n <= 30, H4 (HF, L-BFGS iterate 3) | max deviation 9.6e-15 |
| Laurent evolution tau = 0.5, delta = 1e-4 (R = 4) | error 1.6e-5 <= tail 1.7e-5 (ratio 0.94); leakage 1.5e-16 |
| Laurent evolution tau = 3, delta = 1e-4 (R = 9) | error 2.4e-5 <= tail 3e-5 (ratio 0.79); leakage 2.4e-16 |
| Laurent evolution tau = 7, delta = 1e-4 (R = 15) | error 1e-5 <= tail 2.9e-5 (ratio 0.35); leakage 2.7e-16 |
| sine-window AE constant kappa, M = 63 / 255 / 1023 | 0.974 / 0.993 / 0.998 |
| OpenFermion QPE costs (6 regression cases) | 6 of 6 identical |
| OpenFermion cost_estimator (3 cases: H6 DF QPE, THC Reiher, H6 Psi_6 circuit) | 3 of 3 identical |

- **Replay.** At equilibrium the compiled circuits stay within 3.1e-3 of the exact iterates to
  k\* + 2, and ⟨S²⟩ stays ≤ 1.6e-5. On h6s and n2s the error grows to 4.5e-2 and 9.5e-2,
  ⟨S²⟩ to 1.5e-3 and 9.7e-3, and the compiled overlaps differ from the exact ones by up to
  4.3e-3 (h6s) and 2.2e-2 (n2s). The rounding part stays below its bound Σ_j A_j π/2^b_j on
  every system (worst ratio 0.22, H4).
- **Model checks.** The walk moments, the 2R-call Laurent evolution, the AE constant and the
  OpenFermion reproductions all hold.

**The benchmark's Appendix A table** (its column "design value" holds the values the design
specification printed; against Appendix A of m3-cost-model.md all 97 rows reproduce).

96 of 97 quantities match their design values to the printed digits (Rao-Blackwellized V_m: within 3 standard errors).

| quantity | design value | reproduced | match |
|---|---|---|---|
| h4 allowed / all elements, spin_free | 28 / 60 | 28 / 60 | yes |
| h4 allowed / all elements, spin_orbital | 70 / 150 | 70 / 150 | yes |
| h6a allowed / all elements, spin_free | 153 / 315 | 153 / 315 | yes |
| h6a allowed / all elements, spin_orbital | 408 / 840 | 408 / 840 | yes |
| lih allowed / all elements, spin_free | 105 / 315 | 105 / 315 | yes |
| lih allowed / all elements, spin_orbital | 264 / 840 | 264 / 840 | yes |
| n2 allowed / all elements, spin_free | 33 / 315 | 33 / 315 | yes |
| n2 allowed / all elements, spin_orbital | 84 / 840 | 84 / 840 | yes |
| h8 allowed / all elements, spin_free | 496 / 1008 | 496 / 1008 | yes |
| h8 allowed / all elements, spin_orbital | 1364 / 2772 | 1364 / 2772 | yes |
| h4 V_m at SD iterate 2, allowed elements (exact) | 1220 | 1220.17 | yes |
| h4 V_m at SD iterate 2, all elements (exact) | 2402 | 2401.88 | yes |
| h6a V_m at SD iterate 2, allowed elements (RB 5000, +-113) | 2.515e4 | 2.515e4 | yes (-0.0 sigma, same seed as Appendix A; -1.4 sigma from the exact pair sum 25,308) |
| h6a V_m at SD iterate 2, all elements (RB 5000, +-190) | 5.144e4 | 5.144e4 | yes (-0.0 sigma, same seed as Appendix A) |
| n2 V_m at SD iterate 2, allowed elements (exact) | 3703 | 3753.04 | no: the design value is an RB estimate (seed 7) on the earlier orbital labelling; the exact pair sum is the reproduced one (docs/m3-cost-model.md, Appendix A) |
| n2 V_m at SD iterate 2, all elements (RB 5000, +-159) | 2.947e4 | 2.941e4 | yes (-0.4 sigma, same seed as Appendix A) |
| h4 L-BFGS step 0: E - E_FCI after (mHa) | 21.504 | 21.504 | yes |
| h4 L-BFGS step 0: p after | 0.9802 | 0.9802 | yes |
| h4 L-BFGS step 0: n_d / n_s, b, C_step, C(Psi_1) | 10 / 0 / 15 / 160 / 160 | 10 / 0 / 15 / 160 / 160 | yes |
| h4 L-BFGS step 1: E - E_FCI after (mHa) | 3.890 | 3.890 | yes |
| h4 L-BFGS step 1: p after | 0.9960 | 0.9960 | yes |
| h4 L-BFGS step 1: n_d / n_s, b, C_step, C(Psi_2) | 40 / 24 / 18 / 1192 / 1352 | 40 / 24 / 18 / 1192 / 1352 | yes |
| h4 L-BFGS step 2: E - E_FCI after (mHa) | 1.203 | 1.203 | yes |
| h4 L-BFGS step 2: p after | 0.9981 | 0.9981 | yes |
| h4 L-BFGS step 2: n_d / n_s, b, C_step, C(Psi_3) | 38 / 24 / 18 / 1154 / 2506 | 38 / 24 / 18 / 1154 / 2506 | yes |
| h6a L-BFGS step 0: E - E_FCI after (mHa) | 37.956 | 37.956 | yes |
| h6a L-BFGS step 0: p after | 0.9576 | 0.9576 | yes |
| h6a L-BFGS step 0: n_d / n_s, b, C_step, C(Psi_1) | 51 / 0 / 18 / 969 / 969 | 51 / 0 / 18 / 969 / 969 | yes |
| h6a L-BFGS step 1: E - E_FCI after (mHa) | 12.014 | 12.014 | yes |
| h6a L-BFGS step 1: p after | 0.9850 | 0.9850 | yes |
| h6a L-BFGS step 1: n_d / n_s, b, C_step, C(Psi_2) | 273 / 120 / 21 / 8526 / 9495 | 273 / 120 / 21 / 8526 / 9495 | yes |
| h6a L-BFGS step 2: E - E_FCI after (mHa) | 5.236 | 5.236 | yes |
| h6a L-BFGS step 2: p after | 0.9907 | 0.9907 | yes |
| h6a L-BFGS step 2: n_d / n_s, b, C_step, C(Psi_3) | 271 / 120 / 21 / 8482 / 17977 | 271 / 120 / 21 / 8482 / 17977 | yes |
| h6a L-BFGS step 3: E - E_FCI after (mHa) | 2.991 | 2.991 | yes |
| h6a L-BFGS step 3: p after | 0.9950 | 0.9950 | yes |
| h6a L-BFGS step 3: n_d / n_s, b, C_step, C(Psi_4) | 271 / 120 / 21 / 8482 / 26459 | 271 / 120 / 21 / 8482 / 26459 | yes |
| h6a L-BFGS step 4: E - E_FCI after (mHa) | 1.877 | 1.877 | yes |
| h6a L-BFGS step 4: p after | 0.9974 | 0.9974 | yes |
| h6a L-BFGS step 4: n_d / n_s, b, C_step, C(Psi_5) | 273 / 120 / 21 / 8526 / 34985 | 273 / 120 / 21 / 8526 / 34985 | yes |
| h6a L-BFGS step 5: E - E_FCI after (mHa) | 1.111 | 1.111 | yes |
| h6a L-BFGS step 5: p after | 0.9987 | 0.9987 | yes |
| h6a L-BFGS step 5: n_d / n_s, b, C_step, C(Psi_6) | 273 / 116 / 21 / 8442 / 43427 | 273 / 116 / 21 / 8442 / 43427 | yes |
| n2 L-BFGS step 0: E - E_FCI after (mHa) | 35.209 | 35.209 | yes |
| n2 L-BFGS step 0: p after | 0.9635 | 0.9635 | yes |
| n2 L-BFGS step 0: n_d / n_s, b, C_step, C(Psi_1) | 27 / 0 / 17 / 486 / 486 | 27 / 0 / 17 / 486 / 486 | yes |
| n2 L-BFGS step 1: E - E_FCI after (mHa) | 14.678 | 14.678 | yes |
| n2 L-BFGS step 1: p after | 0.9813 | 0.9813 | yes |
| n2 L-BFGS step 1: n_d / n_s, b, C_step, C(Psi_2) | 55 / 0 / 18 / 1045 / 1531 | 55 / 0 / 18 / 1045 / 1531 | yes |
| n2 L-BFGS step 2: E - E_FCI after (mHa) | 1.147 | 1.147 | yes |
| n2 L-BFGS step 2: p after | 0.9994 | 0.9994 | yes |
| n2 L-BFGS step 2: n_d / n_s, b, C_step, C(Psi_3) | 55 / 0 / 18 / 1045 / 2576 | 55 / 0 / 18 / 1045 / 2576 | yes |
| h4 HF overlap p_0 | 0.9455 | 0.9455 | yes |
| h6a HF overlap p_0 | 0.9026 | 0.9026 | yes |
| n2 HF overlap p_0 | 0.9257 | 0.9257 | yes |
| lih HF overlap p_0 | 0.9743 | 0.9743 | yes |
| h4 norm A before steps 0-3 | 0.8585 / 0.6115 / 0.2157 / 0.1256 | 0.8585 / 0.6115 / 0.2157 / 0.1256 | yes |
| h6a norm A before steps 0-3 | 1.0993 / 0.7424 / 0.4219 / 0.2397 | 1.0993 / 0.7424 / 0.4219 / 0.2397 | yes |
| n2 norm A before steps 0-3 | 1.8955 / 0.6638 / 0.4038 / 0.1168 | 1.8955 / 0.6638 / 0.4038 / 0.1168 | yes |
| h4 eps_2 / eps* | 1.0159 | 1.0159 | yes |
| h6a eps_2 / eps* | 1.0220 | 1.0220 | yes |
| n2 eps_2 / eps* | 0.9805 | 0.9805 | yes |
| lih eps_2 / eps* | 1.0192 | 1.0192 | yes |
| h4 E1 - E0 (Ha), same / any irrep | 0.5657 / 0.5633 | 0.5657 / 0.5633 | yes |
| h6a E1 - E0 (Ha), same / any irrep | 0.3909 / 0.3909 | 0.3909 / 0.3909 | yes |
| n2 E1 - E0 (Ha), same / any irrep | 0.7115 / 0.4117 | 0.7115 / 0.4117 | yes |
| lih E1 - E0 (Ha), same / any irrep | 0.1332 / 0.1332 | 0.1332 / 0.1332 | yes |
| h4 DF selection: threshold / L / Xi | 3e-3 / 7 / 27 | 3e-3 / 7 / 27 | yes |
| h4 DF lambda / C_W / Q | 4.935 / 868 / 188 | 4.935 / 868 / 188 | yes |
| h4 DF C_QPE (1 mHa) | 6.73e6 | 6.73e6 | yes |
| h6a DF selection: threshold / L / Xi | 3e-3 / 11 / 63 | 3e-3 / 11 / 63 | yes |
| h6a DF lambda / C_W / Q | 9.726 / 1309 / 233 | 9.726 / 1309 / 233 | yes |
| h6a DF C_QPE (1 mHa) | 2.00e7 | 2.00e7 | yes |
| n2 DF selection: threshold / L / Xi | 3e-3 / 20 / 82 | 3e-3 / 20 / 82 | yes |
| n2 DF lambda / C_W / Q | 9.423 / 1420 / 235 | 9.423 / 1420 / 235 | yes |
| n2 DF C_QPE (1 mHa) | 2.10e7 | 2.10e7 | yes |
| lih DF selection: threshold / L / Xi | 3e-4 / 18 / 68 | 3e-4 / 18 / 68 | yes |
| lih DF lambda / C_W / Q | 9.259 / 1366 / 235 | 9.259 / 1366 / 235 | yes |
| lih DF C_QPE (1 mHa) | 1.99e7 | 1.99e7 | yes |
| h6a THC selection: M / penalty / lambda / C_W | 15 / auto / 9.125 / 1035 | 15 / auto / 9.125 / 1035 | yes |
| h6a THC C_QPE | 1.48e7 | 1.48e7 | yes |
| physical QPE Q = 233, 19,997,593 Toffolis, budget 0.1: qubits / d / s | 956160 / 23 / 574.9 | 956160 / 23 / 574.9 | yes |
| physical QPE Q = 233, 19,997,593 Toffolis, budget 0.01: qubits / d / s | 1026160 / 25 / 574.9 | 1026160 / 25 / 574.9 | yes |
| physical shot Q = 83, 43,427 Toffolis, budget 1e-5: qubits / d / s | 721960 / 25 / 1.249 | 721960 / 25 / 1.249 | yes |
| physical shot Q = 83, 43,427 Toffolis, budget 0.0001: qubits / d / s | 696960 / 23 | 696960 / 23 | yes |
| physical us per Toffoli (H4, H6 DF QPE) | 26.25 / 28.75 | 26.25 / 28.75 | yes |
| Jacobi-Anger degree R(0.2, 0.0001) | 3 | 3 | yes |
| Jacobi-Anger degree R(0.5, 0.0001) | 4 | 4 | yes |
| Jacobi-Anger degree R(0.79, 0.0001) | 5 | 5 | yes |
| Jacobi-Anger degree R(1.16, 0.0001) | 6 | 6 | yes |
| Jacobi-Anger degree R(4.87, 1e-6) | 14 | 14 | yes |
| Jacobi-Anger degree R(6.81, 1e-6) | 17 | 17 | yes |
| phase-gradient bits for A = 10 at delta = 1e-3 | 15 | 15 | yes |
| phase-gradient bits for A = 64 at delta = 1e-3 | 18 | 18 | yes |
| phase-gradient bits for A = 393 at delta = 1e-3 | 21 | 21 | yes |
| readout rotation C_rot (N = 6) | 405 | 405 | yes |
| sine-window AE kappa, M = 63 / 255 / 1023 | 0.974 / 0.993 / 0.998 | 0.974 / 0.993 / 0.998 | yes |

- The "design value" column holds the design specification's printed values, and "match" means
  agreement to the printed digits for exact quantities and within 3 standard errors for the
  Rao–Blackwellized V_m (all within 0.4 here). The one mismatch,
  N2's V_m on the allowed elements, is an error there: 3703 is an RB estimate (seed 7) on the
  earlier orbital labelling, while N2's 2.8e7 term pairs lie below the 5e7 limit for exact
  pair sums, so the benchmark computes the exact pair sum, V_m = 3753.0 Ha², the value
  Appendix A now lists (the exact branch at SD iterate 1:
  `tests/test_ft_residual_inputs.py::test_n2_variances_use_exact_pair_sums`). Every
  Rao–Blackwellized V_m lies within 0.4 standard errors of the design value, which used the
  same seed, so this is no independent check. The exact pair sum on the 153 allowed h6a
  elements at SD iterate 2 is 25,308 Ha² (a separate run [build check]:
  `measure.routes.direct_residual_variance(..., "matchgate_shadow", method="exact")` alone,
  1.1e9 commuting pairs, about 5 min on one thread; the benchmark script keeps it as
  `EXACT_VM`). The RB estimate of the table, 2.515e4 ± 113 (the value the design specification
  printed, from the same seed), lies 1.4σ and 0.6% below it.

## 9. Runtime

Wall time (s) per system and component of ``run_system`` on one thread, and the peak resident memory of the process after the system (MB).

| system | families | total | build | qpe | trajectories | residual_inputs | iterations | crossovers | handoff | first_step | physical_and_cqe_only | replay | global_validation | peak RSS (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| h4 | lbfgs, lbfgs_so, sd | 11.9 | 0.1 | 2.9 | 1.1 | 1.7 | 5.4 | 0.4 | 0.0 | 0.0 | 0.0 | 0.1 | 0.2 | 517 |
| h6a | lbfgs, lbfgs_so, sd | 52.3 | 1.3 | 7.1 | 2.0 | 3.0 | 33.2 | 4.5 | 0.0 | 0.1 | 0.1 | 1.1 | 0.0 | 517 |
| n2 | lbfgs, lbfgs_so, sd | 36.7 | 1.1 | 6.4 | 1.3 | 4.1 | 20.1 | 3.4 | 0.0 | 0.1 | 0.1 | 0.1 | 0.0 | 537 |
| lih | lbfgs, lbfgs_so, sd | 22.8 | 0.6 | 4.4 | 1.4 | 1.8 | 13.3 | 1.0 | 0.0 | 0.0 | 0.1 | 0.3 | 0.0 | 537 |
| h6s | lbfgs, lbfgs_so | 41.6 | 2.4 | 6.2 | 1.3 | 3.0 | 25.1 | 1.3 | 0.0 | 0.1 | 0.1 | 2.1 | 0.0 | 537 |
| n2s | lbfgs, lbfgs_so | 40.0 | 1.4 | 7.3 | 0.8 | 4.1 | 26.2 | 0.0 | 0.0 | 0.1 | 0.0 | 0.2 | 0.0 | 551 |
| h8 | lbfgs, lbfgs_so | 121.9 | 5.1 | 15.7 | 13.3 | 0.1 | 4.4 | 0.0 | 0.0 | 1.6 | 0.3 | 81.5 | 0.0 | 551 |

Systems: 327.3 s. Appendix A checks: 19.8 s. Whole script: 348.2 s. Peak RSS: 551 MB.

- **Budget.** The script takes 348 s on one thread (327 s for the seven systems, 20 s for the
  Appendix A checks), inside the design budget of 20 min and its 12 min target [design], with a
  peak resident memory of 551 MB (limit 2 GB). The iteration costs dominate (per iterate the
  exact residual and the aux-bias curve on 11 δ values, 0.5 s on h6a); on h8 the compiled
  replay (30 compiled steps on 65,536 Fock states, 82 s) does. The 12 singlet FCI roots are
  solved once per system ("build", 5.1 s on h8) and shared by its four to six records. The
  component budgets of the design [design] hold: the
  QPE scan takes 2.9–15.7 s per system (≤ 25 s) and the residual inputs 1.7–4.1 s where the
  residual is costed (≤ 10 s).
- **The spin-orbital family** adds 15–84 s, depending on the run: the same script with
  `--threads 1 --optimizers lbfgs,sd` and its other defaults (a separate run [build check])
  takes 265 s (246 s for the systems), peak memory 542 MB, against 280–348 s for the full runs.
  Per system, with / without `lbfgs_so` (the run of the table above): h4 11.9 / 9.0 s, h6a
  52.3 / 43.8, n2 36.7 / 30.7, lih 22.8 / 17.4, h6s 41.6 / 28.9, n2s 40.0 / 27.5, h8
  121.9 / 88.2. These are single runs, and repeated runs of this code on the shared machine
  differ per system by up to 40% of the fastest (eight runs of h4, 8.5–11.9 s; six of n2,
  30.6–38.2 s; one of them next to another job; the fastest whole run took 280 s, peak 546 MB)
  [build check], so the family's share is a rough figure. The budget holds with the family, so
  its residual costs are not restricted to the core systems.

## 10. Caveats

The open questions of [m3-cost-model.md](m3-cost-model.md) ("Scope and open questions"), with
what this run says about them:

1. **The X-LCU cost C_obs = 2 L_X [estimate]** decides the ae_x rows, the best residual route
   on h4, n2, lih and n2s from k = 2–4. ‖X_e‖ on the sector is 0.16–0.50 of α_X on H4
   (`ft.residual_inputs`), so structured block encodings could lower these rows 2–6×. On h6a
   and h6s ae_x costs 1.8–4.4× the selected route over the 122 costed iterations from k = 2 on
   (both loops, every family; direct_mg at k = 2, ae_aux after; `results/ft_h6a.json`,
   `results/ft_h6s.json`), so ae_x rows 2× cheaper overall, e.g. through an X block encoding
   with half the normalization α_X (which halves every N_e), would take over 2 of them (h6a,
   spin-orbital, k = 2), 3× cheaper 35, and 4.4× cheaper all. A cheaper C_obs alone gains less,
   since every application also prepares Ψ_k: at h6a k = 2 (spin-orbital, measured loop)
   C(Ψ₂) = 9495 of the 20,079 Toffolis of an application, and halving C_obs leaves ae_x at
   least 1.3× dearer than direct_mg (≥ 6.2e10 against 4.8e10 Toffolis). The iteration cost
   would follow only where the residual dominates (after k\*, or on h6s and n2s).
2. **The aux-route bias** is counted fully against r. It decides the ae_aux rows of h6a and h6s,
   which are therefore upper bounds in that respect. M2 found the aux route unstable for
   δ·w_max ≥ π (LiH diverged at δ = 0.6). With w_max = 5.02 Ha on h6a and 3.18 Ha on h6s (the
   largest singlet gap in the HF irrep, from a full diagonalization in that irrep
   [build check]), 32 of the 60 ae_aux rows selected on h6a (δ = 0.4–1.0) and 17 of the 52 on
   h6s (δ = 0.7–1.0) lie at δ ≥ π/w_max = 0.63 and 0.99; 30 of the 32 on h6a are steepest-descent
   rows. Restricting δ below π/w_max raises C_loop(k\*) on h6a by 0.08% (spin-free L-BFGS), 0
   (spin-orbital) and 1.9% (steepest descent, whose 7-day machine counts rise from 848 / 869 to
   1088 / 1181, measured / classical first step), and on h6s by 0.17–0.32% (spin-free, which
   still no number of machines finishes in 7 days) and 0.53–0.72% (spin-orbital; 203 / 197 →
   208 / 200 machines). These come from separate runs [build check] of `ft.pipeline.run_system`,
   the function the benchmark script calls per system (the script has no option for the grid):
   `run_system("h6a", params=FTParameters(aux_deltas=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)),
   r_scan=False)`, and on h6s the same with the grid up to 0.8.
3. **The line-search model** (11 candidates at ±1.6 mHa) is most of every iteration before k\*
   at equilibrium. Amplitude-estimated energies would cut it to 0.065–0.31 (T3), cheaper step
   rules (L-BFGS unit steps, fits) further; neither changes a verdict: with amplitude-estimated
   energies the cheapest iteration, h4 at HF, still costs 45 C_QPE (1.8e8 Toffolis, T3), and
   without a line search an iteration costs its residual alone: nothing at HF, where the
   classical step applies, and at least 1.4 C_QPE from k = 1 on (h4; 3.6 on n2, 32 on h6a;
   `results/ft_<system>.json`), against savings ≤ 0.128 C_QPE.
4. **δ_step = 1e-3 is empirical.** The product-formula error of the elementwise circuits is not
   budgeted: 2e-4 to 3.1e-3 at equilibrium but 3e-2 to 9.5e-2 on h6s and n2s (T9). Its part
   outside the singlet sector is a quintet at the RHF step
   (`tests/test_ft_circuits.py::test_first_step_product_error_and_spin_contamination`), of
   weight ⟨S²⟩/6; counted the same way along the replay (⟨S²⟩/(6‖Ψ̃_k − Ψ_k‖²) from T9), it is
   two thirds of the RHF-step error on n2s (67% at k = 1, 56% at k = 2) but only 15–26% at
   k ≥ 3, where the error peaks (2–18% spin-orbital); on h6s 25–34% (spin-free) and 8–10%
   (spin-orbital).
5. **The hand-off conclusions** hold for equilibrium molecules with p_HF = 0.89–0.97: a measured
   loop could pay only below p_HF = 7.3e-4 to 6.5e-3 (the break-even overlaps p₀*(1)). h6s and
   n2s are only a first look at overlap-limited systems, where certification decides: the
   classical first step gives a cheap certified hand-off on h6s (both metrics) and on n2s with
   the spin-orbital metric (2.5 C_QPE), but not on spin-free n2s (1073 C_QPE, against 571 for
   the measured loop). A scaling study is needed.
6. **THC fits are non-convex.** The selected THC encodings of h6s (M = 12) and h8 (M = 24) come
   from fits whose L-BFGS-B refinement did not converge (`refine_success` false in
   `results/ft_h6s.json` and `results/ft_h8.json`; they still meet the selection rule), and
   lih, n2 and n2s select M = 21 with λ = 20–69 Ha (on n2s M = 18 passes too, at 3.94e7
   Toffolis).

Beyond those: the trajectories are exact (no shot noise, no closed-loop compiled CQE), the
residual target r = 0.3 is M2's convention while the noisy CQE converges with much less
accurate residuals (M2, section 4), H8's residual is not costed, and the wall clock assumes
OpenFermion's factory-limited surface-code model (p = 1e-3, 4 factories, 1 µs cycles). The
per-shot residual variances v_e and w_e are taken at iterate 1 and reused at every iterate (v_e
agrees within 3% element by element on H4 at SD iterates 1 and 4; F10), and the V_E of Ψ_{k+1}
stands for all 11 line-search candidates (F13).

## 11. Changes from the design specification

The cost model in [m3-cost-model.md](m3-cost-model.md) is the corrected one; this table records
what the design specification said and why it changed. Every item was found while building
and testing `ft`, and its evidence is the test that pins it, the docstring that
states it, a table or result file of this run, or a separate check [build check].

| item | design specification | corrected reading | evidence |
|---|---|---|---|
| physical layer: code distance | the code distance d is non-decreasing in N_Tof and in 1/budget | false for both; the optimal area × duration is monotone in both (`tests/test_ft_physical.py::test_objective_is_monotone_in_toffolis_and_budget`) | 233 logical qubits at budget 0.1: 562 Toffolis need d = 15 with AutoCCZ(9, 15), 1000 need d = 13 with AutoCCZ(11, 13); a shot of 83 qubits and 5623 Toffolis: d = 25 at budget 3e-5, d = 23 at 1e-5; OpenFermion's own `cost_estimator` and `estimate_cost` agree (`tests/test_ft_physical.py::test_distance_is_not_monotone_in_toffolis_or_budget_and_openfermion_agrees`) |
| F1: S² | the phase function conserves S² when spin-mirrored cells share constants | true for one phase function, not for the elementwise product of F3 | ⟨S²⟩ of the compiled RHF step is 3.9e-3 (n2s) and 1.5e-3 (h6s) against 0 for the exact Ψ₁ (`tests/test_ft_circuits.py::test_first_step_product_error_and_spin_contamination`); 3.2e-3 (h6s) and 1.1e-3 (n2s) after the classical step (T7) |
| F3: small steps | the rounding fixed point keeps the step within δ_step | for small ε the dropped angles exceed δ_step; flagged per candidate | the h6a L-BFGS step 3 at ε = 0.00625 compiles to the identity with a rounding bound of 1.73e-3 (`tests/test_ft_stateprep.py::test_candidate_step_costs_give_qubits_and_flags`); T9: 4–31 flagged candidates per system along the measured loops |
| F8: GQSP rotations | 3(2R + 1)(β − 2) Toffolis of GQSP rotations | an upper bound: the Motlagh–Wiebe form needs 2d + 3 angles for degree d | overcount 2R(β − 2) = 476 of 45,976 Toffolis (1.0%) for H6 at t = 0.7 (`ft.primitives` module docstring; the 45,976: `tests/test_ft_primitives.py::test_gqsp_and_evolution_costs`) |
| F11 / F19: reflection | T contains Q_run/2 per application; the longest run and the AE time of F19 do not | the reflection S₀ acts on the Q_A = Q(Ψ_k) + Q_obs + 1 qubits of the Hadamard-test circuit, and counts in T, in the longest run and in F19 | `tests/test_ft_residual.py::test_ae_routes_follow_f11`; on the benchmark the reflection term is ≤ 14% of an application (ae_product, h4 at HF on the SF walk: 67 of 467 Toffolis), and dropping the phase register from it changes T by ≤ 1.25% at κ = 1 and 1.35% at κ = 2 (`results/ft_<system>.json`, `iterations`); 9.6% and 0.7% with the DF walk of the `ft.residual` module docstring |
| F12: C_obs | C_obs of the "aux" formulation left implicit | C_obs = C_evol(δ); the Q_e enter through C_probe | adding C_Q would add 16 Toffolis per query, ≤ 0.17% on the benchmark (h4 at HF, δ = 0.4: 9514 Toffolis per query; `results/ft_h4.json`); the formula: `tests/test_ft_residual.py::test_gradient_estimation_follows_f12` |
| F13: DF split | the DF split is "the lowest of four at HF" | ties within 1e-9 relative go to the shifted split | for N2 the Cholesky splits tie to the last bit at HF (0.573 Ha²); the shifted one is 14–20% lower along the trajectories (`tests/test_ft_residual_inputs.py::test_energy_variance_tie_goes_to_the_shifted_split`; the values: `ft.residual_inputs` module docstring) |
| parameters and `residual_inputs.element_variances`: exact pair sums | exact matchgate pair sums for N ≤ 4, RB above | exact when Σ_e L_e(L_e + 1)/2 ≤ 5e7 pairs | N2 (2.8e7 pairs) is then exact: V_m = 3751.1 at SD iterate 1 (`tests/test_ft_residual_inputs.py::test_n2_variances_use_exact_pair_sums`), in 3.5 s, where RB at seed 7 gives 3730 ± 50 (`ft.residual_inputs` module docstring) |
| Appendix A: V_m | n2: 3703 (allowed) at SD iterate 2; h6a: 2.515e4 (RB) | n2: 3753.0 (exact pair sums); h6a: the exact pair sum is 25,308 | the n2 value is an RB estimate on the earlier orbital labelling, and the benchmark reproduces 3753.0 (benchmark only); the h6a RB estimate is 0.6% (1.4σ) below the exact pair sum, a separate run [build check] (section 8 and its Appendix A table) |
| residual API: `ResidualProblem` | `ResidualProblem` has no orbital count | `norb` added: Q_obs and the aux-route qubits need 2N | Q_obs = Q_walk − 2N + 4 (`ft.residual` module docstring; `tests/test_ft_residual_inputs.py::test_build_problem_wiring`) |
| F13: qubits | the line search runs on Q(Ψ_k) | on max(Q(Ψ_k), Q_step(ε_c), Q_rot) | H4 at k = 0: 8 qubits for Ψ_0 but 61 for the candidate step (`ft.residual.line_search_cost`); Q_rot = 89 > Q(Ψ_6) = 83 for H6 (the 89: `tests/test_ft_stateprep.py::test_readout_rotation_qubits`; the 83: T2; `ft.stateprep.candidate_step_costs`, `ft.stateprep.readout_rotation_qubits`) |
| F6: final leaf rotation | the final leaf rotation's bits b_f left open | b_f = F2(N(N − 1), δ_step), its own budget | the design notes used the trajectory's largest b [design], which makes C(Ψ_k) depend on later steps; the rotation costs 210 Toffolis at N = 4 (b_f = 16) and 555 at N = 6 (b_f = 17) (`ft.stateprep.prep_cost` with `method="leaves"`; the 555: `tests/test_ft_stateprep.py::test_prep_cost_cumulative_sums_and_qubits`) |
| parameters: `aux_deltas` | (0.05, 0.1, 0.2, …, 1.0, 1.2) | (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2), the grid of the design notes' scripts, which has no 0.9 (the specification's "0.2, …, 1.0" would include it) | the grid of the design notes' cost and r-scan scripts [design]; `ft.primitives.FTParameters` (`tests/test_ft_primitives.py::test_parameters_have_the_spec_defaults_and_are_frozen`) |
| F8: R = 0 | R = 0 costs one GQSP layer, 3(β − 2) = 42 | R = 0 (the identity within δ) costs nothing | `tests/test_ft_primitives.py::test_gqsp_and_evolution_costs` |
| trajectory API: E₁ | E₁ = the first root after E₀ whose `ci_irrep` equals that of ψ₀ | degenerate roots are tested as a subspace (largest eigenvalue of CᵀP_gC > ½) | the solver returns degenerate roots of different irreps as mixtures; the per-root rule can skip a symmetric state and overestimate E₁, which would invalidate the certificate (`tests/test_ft_trajectory.py::test_e1_degenerate_roots_are_tested_as_a_subspace`); the same E₁ as the per-root rule on all seven systems [build check] |
| F15: rule E | the optimum k_opt of rule E over all rows | over the certified rows (E₁ − E_k > 3ΔE) only; "expected_oracle" over all rows | HF of h6s and n2s is not certified (E_HF − E₀ = 0.282 and 0.566 Ha against E₁ − E₀ = 0.105 and 0.192 Ha); T6: certified k_opt = 1 and 2 at 554 and 571 C_QPE, oracle k_opt = 0 at 1.74 and 3.96 C_QPE (`tests/test_ft_handoff.py::test_optimal_handoff`) |
| F14 / F16 and the benchmark states | ε₀ = ε₂ (classical) and "exact CQE iterates" | the classical-first-step loop uses its own record: step 0 at ε₂, then the exact continuation (`run_cqe(..., first_eps=ε₂)`) | the exact iterates after the exact first step differ from the continuation by up to 30 mHa and 0.11 in p (n2s), 0.45 mHa (h6s), ≤ 0.012 mHa (L-BFGS) and ≤ 0.33 mHa (SD) at equilibrium (the "exact" and "second_order" trajectories of `results/ft_<system>.json`; `ft.trajectory` module docstring); a restart from Ψ₁ instead lies 3.4–7.6 mHa higher at k = 2 on h4, h6a, n2 and h8 (0.24 mHa on lih) (`ft.trajectory` module docstring; `tests/test_ft_trajectory.py::test_second_order_first_step_keeps_the_step0_memory`) |
| F16: metric | ε₀ = ‖A₀‖²/E''(0) | ε₂ = −E'(0)/E''(0) with E'(0) = −Re Σ d A, i.e. −⟨g, Mg⟩ for the spin-orbital metric | `ft.trajectory.second_order_step` (`tests/test_ft_trajectory.py::test_second_order_step_metric_and_state`); T7 lists both metrics |
| F16: converged states | ε₂/ε* is reported for any state | flagged "converged_state" when the predicted lowering E'²/(2E'') < 1e-10 Ha | at the H4 FCI state the ratio came out as 0.49 (`tests/test_ft_trajectory.py::test_second_order_step_at_a_converged_state`) |
| QPE test: selection on h6a | `select_qpe` on h6a picks DF threshold 3e-3 | only with `kind="df"`; overall THC M = 15 wins | T1: THC 1.48e7 against SF 1.82e7 and DF 2.00e7 Toffolis (`tests/test_ft_qpe.py::test_select_qpe_df_reproduces_appendix_a`, `tests/test_ft_qpe.py::test_thc_scan_h6a_selects_m15`) |
| walk test: QPE outcome law | window probability / p_k k-independent to 1e-8 on H4 | only in the setup of the design notes (DF H′ at threshold 1e-3) [design] | spread 9.96e-9 there (`tests/test_ft_walk.py::test_window_probability_over_p_is_k_independent`), 1.21e-8 with the exact H [build check] |
| benchmark states: symmetry | the trajectories are the exact CQE iterates from HF | the residual is projected onto the symmetry-allowed entries before every step of the driver (F2a), which keeps the iterates symmetry-pure | exact L-BFGS amplifies the round-off of the forbidden entries 10–150× per step on N2, up to max\|A\| at step 11; without the projection the weight outside the HF irrep reaches 1.8e-7 (n2) and 1.9e-4 (n2s), and n2s ends its 20 steps at 7.31 instead of 5.71 mHa (`ft.symmetry` and `ft.trajectory` module docstrings; `tests/test_ft_trajectory.py::test_symmetry_projection_removes_irrep_leakage_on_n2s`, `tests/test_ft_symmetry.py::test_masked_residual_keeps_long_lbfgs_runs_pure`) |
| F1: phase-gradient register | the preparation of the phase-gradient register is not mentioned | \|Φ_b⟩ is prepared once per machine and reused across steps and shots [assumption; OpenFermion's convention]: its qubits count in Q_prep, its preparation does not | `ft.primitives` and `ft.stateprep` module docstrings |
| F13: AE-energy qubits | no qubit count for amplitude-estimated energies | Q_run = max(Q(Ψ_k), Q_prep(ε_c)) + Q_walk − 2N + 1 + ⌈log₂(N_E/2)⌉ [estimate], with N_E = 2πκλ/eps_ls applications and N the active orbitals: the ae_product count without the +4 of U_Q | `tests/test_ft_residual.py::test_energy_routes`; it enters only T3's "AE energies" column |
| pipeline API: hand-off QPE | C_QPE and the residual walk of the DF encoding (Appendix A) | the cheapest admissible encoding overall (SF on h4, n2, lih and n2s, THC on h6a, h6s and h8), which also has the lowest λ·C_W | it is the QPE one would run; DF's C_QPE is 1.26–1.76× the selected one (T1) |
| physical API: `machines_for` (F19) | `machines_for` returns an int; the campaign is a fluid share | whole runs, t ⌈n/M⌉; `machines_for` returns None when no number of machines suffices | 5 shots on 4 machines take 2 t_run, not 1.25 t_run; ten runs of 3.9 days need 10 machines per week, not 6 (`tests/test_ft_physical.py::test_campaign_seconds_and_machines`) |
| F19: loops | one machine count per campaign | the loop's campaigns run one after another; the unequal, indivisible AE runs follow the LPT schedule (longest first onto the least-loaded machine) | T4: the spin-free h6s loop has a critical path of 8.5 days (the longest run of each campaign), so no number of machines finishes it in 7 days; the LPT counts are within 1.05× of the lower bound of every schedule (t⌈n/M⌉ for shots, max(W/M, t_max) for AE runs) for every loop total that has a count, where the list-scheduling bound W/M + (1 − 1/M) t_max asked for up to 2.6× more machines and a fluid share ⌈ΣW/D⌉ is no schedule (249 machines for that h6s loop) (`ft.pipeline` module docstring; `tests/test_ft_pipeline.py::test_lpt_makespan_against_exhaustive_schedules`, `tests/test_ft_pipeline.py::test_machines_for_several_unequal_campaigns`) |
| pipeline API: `energy_variance` | one energy variance for `iteration_costs` | the DF split chosen at HF, evaluated at Ψ_{k+1} of every iteration (F13) | F13 evaluates V_E at Ψ_{k+1}; one number cannot (H6, the shifted Cholesky split: 0.99 Ha² at HF, 1.23 Ha² at SD iterate 1; `ft.residual_inputs` module docstring); the split chosen at HF: `tests/test_ft_pipeline.py::test_energy_split_and_variances_follow_the_rules`; the evaluation at Ψ_{k+1}: `tests/test_ft_pipeline.py::test_iteration_costs_follow_f14_on_a_hand_built_input` |
| pipeline API: `machines_7d` | an int | int or None (no number of machines suffices); 0 for the free classical iteration | as `ft.physical.machines_for` (`tests/test_ft_physical.py::test_campaign_seconds_and_machines`, `tests/test_ft_pipeline.py::test_campaign_schedules_and_machines`) |
| benchmark: unreached k* | iteration costs to k* + 4, T2 to k* + 2 | if k* is not reached (n2s), every iteration is costed and the per-iterate tables stop at k = 8 | n2s ends 2.6–7.6 mHa above E₀ after 20 steps (T8; `tests/test_ft_pipeline.py::test_iteration_coverage_rules`) |

