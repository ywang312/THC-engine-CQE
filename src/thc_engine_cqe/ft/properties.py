"""Ground-state properties from the CQE iterate (route A) and from QPE-projected copies (route B) (F46-F53).

Tags and formulas F1-F19 as in :mod:`thc_engine_cqe.ft.primitives` (docs/m3-cost-model.md); F40 in
:mod:`thc_engine_cqe.ft.acceptance`, F42 in :mod:`thc_engine_cqe.ft.certify`, F43 in
:mod:`thc_engine_cqe.ft.amplify`. Costs are Toffolis, energies Hartree; properties are in their own
units (RDM elements and occupations in electrons, the dipole in atomic units). The target state is
psi_0, the lowest singlet of the HF irrep of the exact active-space H (M3's sector); route A
delivers an iterate Psi_k (exact, or compiled: Psi~_k), route B the ground state psi_0' of the
block-encoded H' of the selected QPE encoding (F9) up to the filter leakage.

Property targets (F46) [derived; conventions of docs/conventions.md]
-------------------------------------------------------------------
gamma[p,q] = <E_qp> (spin-summed; tr gamma = n_e), D[p,q,r,s] = <e_{pq,rs}> (pair order, no 1/2;
sum_pq D[p,p,q,q] = n_e (n_e - 1)), real states throughout (as M3). A :class:`PropertyTarget` is a set
of K Hermitian element operators O_e = sum f_e[p,q] E_pq + sum c_e[p,q,r,s] e_{pq,rs} (+ a constant),
with multiplicities m_e and a norm:

    kind           elements                                     operator O_e                     norm
    rdm1           p <= q, irrep(p) = irrep(q); m = 2 (p != q)  (E_pq + E_qp)/2                  fro
    rdm2           orbits of {1, P, T, PT}, F2a mask; m = orbit  (e_{pq,rs} + e_{qp,sr})/2         fro
    occupations    the N natural orbitals v_i of gamma_0        N_i = sum_pq v_ip v_iq E_qp      max
    cumulant       the rdm2 orbits                              O^Delta_e (linearized at gamma_0) fro
    cumulant_norm  one                                          sum_e m_e (Delta_e/||Delta||) O^Delta_e  scalar
    dipole         one (all-active builds)                      sum_A Z_A z_A - sum_pq r_z[p,q] E_pq     scalar

with P: (pq,rs) -> (rs,pq) and T: (pq,rs) -> (qp,sr), both of sign +1 on real states, so every
element of an orbit has the same expectation and sum_e m_e x_e^2 is the Frobenius norm of the full
N^4 tensor; there are (N^4 + 3 N^2)/4 orbits (76 at N = 4, 351 at N = 6) with sum_e m_e = N^4
[build check]. After the mask of F2a (irrep(p) x irrep(q) x irrep(r) x irrep(s) totally symmetric)
44 (H4), 189 (H6), 141 (LiH, C2v) and 69 (N2 CAS(6,6), D2h) remain [build check]. The natural
orbitals diagonalize gamma_0 block by block (its symmetry blocks: the connected components of
|gamma_0| > 1e-10), so degenerate occupations of different irreps are never mixed; they are sorted by
occupation, descending. The linearized occupations N_i are first-order exact only for perturbations
that do not couple degenerate natural orbitals: LiH's two pi occupations (1.5246e-3, of different C2v
irreps) shift at first order under a symmetry-breaking perturbation of psi_0, while symmetry-conserving
states (every CQE iterate) keep them uncoupled [build check]. The spin-summed singlet cumulant is

    Delta[p,q,r,s] = D[p,q,r,s] - gamma[p,q] gamma[r,s] + 1/2 gamma[p,s] gamma[r,q]

[derived from the spin-orbital cumulant with gamma_alpha = gamma_beta = gamma/2; the decomposition
cited from memory: Kutzelnigg and Mukherjee, J. Chem. Phys. 110, 2800 (1999), unverified]; it
vanishes for a closed-shell determinant. Its element operators are linearized at gamma_0:
O^Delta_e = O^D_e - gamma[r,s] E^_pq - gamma[p,q] E^_rs + 1/2 (gamma[r,q] E^_ps + gamma[p,s] E^_rq),
E^_ab = (E_ab + E_ba)/2, averaged with the same expression of e's Hermitian partner (q,p,s,r), so
<O^Delta_e>_Psi - <O^Delta_e>_0 is Delta_e(Psi) - Delta_e(psi_0) to first order (the expectations of a
"cumulant" target are those of the linearized operators, not Delta itself). The natural occupations
are linearized the same way (N_i at the natural orbitals of psi_0); :func:`property_values` returns
the nonlinear functionals (the sorted eigenvalues of gamma, Delta, ||Delta||_F, mu), and
:func:`property_error` their difference in the target's norm: fro sqrt(sum_e m_e d_e^2), max
max_e |d_e|, scalar |d|. An estimate meets eps iff bias^2 + variance <= eps^2 (F10's rule with an
absolute target); targets are RMS, without a union bound over elements [assumption].

The two constructions of every element, JW Pauli sums (:meth:`PropertyTarget.operators`,
``measure.pauli``) and CI-space operators (:meth:`PropertyTarget.expectations`, ``fci_ops``), agree to
<= 1e-12 and reproduce ``residual.rdm12`` (h4, lih; tests/test_ft_properties.py). The LiH dipole
(origin 0) is -1.8177157 a.u. at the FCI state and -1.9110889 a.u. at HF, equal to PySCF's
``dip_moment`` to 1e-10 [build check].

Bias of an approximate state (F47) [derived]
-------------------------------------------
For Psi = sqrt(p) psi_0 + sqrt(q) chi with chi orthogonal to psi_0,
<O>_Psi - <O>_0 = 2 sqrt(pq) Re<psi_0|O|chi> + q (<chi|O|chi> - <O>_0), so

    |b_O| <= 2 sigma_0(O) sqrt(q (1 - q)) + q r_O,          (the q form)
    |b_O| <= 2 sigma_0(O) sqrt(d/Delta_1) + r_O d/Delta_1,   (the energy form, q <= d/Delta_1 by Eckart)

sigma_0^2 = <O^2>_0 - <O>_0^2 (:meth:`PropertyTarget.spreads`), r_O = ||O - <O>_0|| on the
(n_alpha, n_beta) sector (:meth:`PropertyTarget.ranges`, Lanczos to 1e-10 relative), d = E_Psi - E_0
and Delta_1 the gap to the lowest level the state populates (the same-irrep singlet E_1 for exact
iterates). A property error is first order in sqrt(q), the energy error second order. Per element; for
tensors by the triangle inequality, ||b||_m <= 2 s ||sigma_0||_m + t ||r||_m (fro) or the largest
element bound (max) (:func:`bias_bound`). Used for route A, for route B's leakage (q the leaked
weight) and for the model bias of H' (q = 1 - F, F = |<psi_0'|psi_0>|^2 = 1 - the encoding's
infidelity). [build check: both forms hold on every row of the h4 and lih spin-orbital records with
the classical first step (exact line search and unit steps, 20 steps, the six targets), the smallest
bound/|bias| per element 1.06 (LiH 2-RDM); and for the model bias of every scanned encoding of h4 and
n2s.] The model-bias bound is loose: at the infidelity 2.1e-6 of LiH's selected SF L = 16 it is 5.7e-3
(1-RDM) and 3.4e-3 a.u. (dipole), against exact model biases of 2.4e-4 and 4.7e-4 a.u. [build check],
so route B costs with the exact model bias wherever the emulation has psi_0'.

For the nonlinear targets (the sorted NONs, the cumulant and its norm) F47 bounds the error of the
linearized element operators, i.e. the property error to first order in sqrt(q). For the norm ||Delta||_F
the second-order terms (the norm's curvature, the gamma-quadratic part of Delta) are not covered: for chi
in psi_0's symmetry sector (the excited eigenstates of H with HF overlap) the maximal error exceeds the
bound at q >~ 0.1 and stays below it at q <= 1e-2, which covers its uses in the costs (route B's leaked
weight w_f + w_AA << 1e-2 and the default model bias at the infidelity) [build check: bound/maximal
error 1.011, 1.027, 0.983, 0.941 (h4) and 1.004, 1.007, 0.910, 0.730 (lih) at q = 1e-3, 1e-2, 0.1, 0.5];
U11's F47 check of the norm on the records is first order. To all orders
(``bias_bound(..., cumulant_parts=(cumulant, rdm1))``): with gamma = gamma_0 + dg and D = D_0 + dD,
Delta_Psi - Delta_0 = L + Q, where L = dD - gamma_0 (x) dg - dg (x) gamma_0 + 1/2 (gamma_0 (x) dg +
dg (x) gamma_0)^X holds the changes of the linearized cumulant elements exactly and
Q = -dg (x) dg + 1/2 (dg (x) dg)^X, with (A (x) B)[pqrs] = A[pq] B[rs] and Y^X[pqrs] = Y[psrq]. Since
||dg (x) dg||_F = ||(dg (x) dg)^X||_F = ||dg||_F^2, ||Q||_F <= 3/2 ||dg||_F^2, and the reverse triangle
inequality gives

    | ||Delta_Psi||_F - ||Delta_0||_F | <= ||L||_F + ||Q||_F <= B_cum + 3/2 B_1^2,

B_cum and B_1 F47's bounds of the cumulant tensor and 1-RDM targets, exact for these linear operators
[derived]. With the F2a-masked targets this holds on real states of psi_0's symmetry sector, where the
masked elements carry the whole tensors; unmasked targets (every irrep 0) extend it to every real state.
[build check: 2.7-9.3 (h4) and 4.4-15.9 (lih) times the maximal errors above.]

Estimators (F48) [cited: Zhao, Rubin, Miyake, PRL 127, 110504 (2021) for matchgate shadows; F11; F12]
-----------------------------------------------------------------------------------------------------
With the statistical budget B = (eps^2 - b^2)^{1/2} (infeasible when B^2 <= 1e-12 eps^2):

* shadows (:func:`shadow_cost`): S = sum_e m_e v_e / B^2 (fro) or max_e v_e / B^2 (max, scalar)
  snapshots of C_copy Toffolis, each a run on Q_copy logical qubits at p_fail_shot (F18 with 2d
  readout cycles); v_e the exact per-snapshot matchgate variances
  (``measure.schemes.matchgate_shadow_variance``; :func:`property_inputs`);
* per-element amplitude estimation (:func:`ae_cost`; F11 with the Pauli LCU of O_e: alpha_e its
  1-norm without the identity, L_e terms, C_obs,e = 2 L_e, Q_obs,e = ceil(log2 L_e) + chi + 4
  [estimate], as M3's ae_x): N_A = 2 pi kappa Gamma / B, Gamma = (sum_e m_e^(1/3) alpha_e^(2/3))^(3/2)
  (fro; ``ft.residual.ae_allocation``), or eps_e = B for every element (max, scalar);
  T = sum_e N_e (C_U + C_obs,e + Q_A,e / 2), Q_A,e = Q_U + Q_obs,e + 1, one coherent run per element
  at p_fail_ae on Q_A,e + ceil(log2(N_e/2)) qubits;
* gradient estimation (:func:`gradient_cost`; a band, never selected): N_GE = 2 pi kappa_g sqrt(K)
  (sum_e m_e alpha_e^2)^(1/2) / B (fro), 2 pi kappa_g sqrt(K) max_e alpha_e / B (max), 2 pi kappa_g
  alpha / B (scalar), T = N_GE (C_U + K (2 b_theta + 2)) on Q_U + K b_theta + max_e Q_obs,e qubits
  [estimate: undercounts multi-term O_e; F12's probe count]; at K = 1 and kappa_g = kappa it needs AE's
  preparations;
* crossover (:func:`crossover_accuracy`): shadows and AE need the same preparations at
  B* = V_m/(2 pi kappa Gamma) (fro) or max_e v_e/(2 pi kappa sum_e alpha_e) (max, scalar); AE wins
  below it [derived].

At psi_0 [build check]: V_m(2-RDM) = 1511 (h4, 44 elements), 1.04e4 (lih, 141), 5733 (n2s, 69); the
largest 2-RDM element variance 40 (h4) and 77 (lih); B* = 0.43 (1-RDM), 0.28 (2-RDM), 0.24 (cumulant),
0.28 (NONs), 0.72 (cumulant norm) on h4, and 2.5 a.u. for the LiH dipole. The exact pair sums agree
with 2e4 emulated snapshots (|z| <= 4, six elements of every h4 target).

Route A (F49) [derived]
----------------------
At a budget k fixed before the run, T_A(P, eps; k) = C_loop(k) + T_meas(P, B_A(k); C_U = C(Psi_k)) with
B_A(k)^2 = eps^2 - ||b_A(k)||^2, for k in {2, 5, 10, 20, 60} (:func:`route_a_rows`); a row is usable
iff its exact bias ||b_A(k)|| < eps (flag "not_usable" otherwise, reported and never selected). A
budget k beyond the record (K < k) uses its last iterate Psi_K (flag "record_ended"): with the free
loop and a converged record C(Psi_K) is the realized cost (the loop stops at convergence); a measured
loop with the pre-declared budget would run k - K more measured iterations, so C_loop(K) only bounds its
cost from below, and such a row (as every row past a record that did not converge) is also
"not_usable". The minimum over k with the exact bias is the oracle row "A-oracle" (flag "oracle_k"):
reported, never selected. Route A's estimate carries no error certificate (F47 needs E_Psik - E_0 and
E1lb), so a fixed-budget row's usability is an emulation fact. Loops: "free" (C_loop = 0), and
measured loops whose residual costs T_res(j) (j = 0..K-1, T_res(0) = 0 after the classical first step)
the caller passes per label (``residual_costs``; F49: min(direct_mg, ae_x) at kappa = 1 and r = 1 or 0.3
along the record): C_loop(k) = sum_{j<k} T_res(j). Compiled iterates Psi~_k come from
``ft.circuits.replay_elementwise`` with s first-order slices of the steps in ``sliced_steps``
(:func:`compiled_iterates`; F25's sliced cost by :func:`sliced_prep_cost`). [build check: on the
spin-orbital exact-line-search records with the classical first step, ||Psi~_k - Psi_k|| = 1.77e-3 at
k = 10 (h4) and 2.85e-4 at k = 14 (lih); 8 slices on steps 0-1 lower h4's to 1.31e-3 at 1.74 x
C(Psi~_10).]

Route B copies (F50) [derived]
-----------------------------
A start Psi_s (HF, Psi_1 or an iterate Psi_k with its C_loop) with overlap p and the step filter of
F43 (``ft.amplify.filter_cost``) between a pass edge and the stop edge E1lb (:func:`ground_copy`): the
pass edge is a certified U_s (CISD(H'), or the iterate's energy with its F42 certificate cost) or
E1lb - W_L (the E1lb-referenced form: flag "termination_not_certified"; with E_0' > E1lb - W_L the copy
is "not_terminating" and never selected); :func:`route_b_rows` takes the edge that gives the cheaper
row. The filter error is

    eps_f = min(1e-2, share eps sqrt(p) / (2 sigma_0 sqrt(1 - p))),   share = 0.1,

sigma_0 the target's ground-state spread ((sum_e m_e sigma_0,e^2)^(1/2) for fro, max_e sigma_0,e
for max). The success probability is a in [p (1 - eps_f)^2, p + eps_f^2 (1 - p)]; the costs take
a = p [assumption: the ground state lies deep in the pass band, where |1 - g| << eps_f; the lower end
would raise them by at most (1 - eps_f)^-2 <= 1.02]. Leaked weight and bias (F47 with q = w):

    w_f <= eps_f^2 (1 - p) / (p (1 - eps_f)^2),   b_leak <= 2 sigma_0 sqrt(w_f) + w_f r_O.

* destructive copy (shadows): C_copy = min(1/a, E_BBHT(a)) (C_s + C_filt);
* coherent copy (AE, GE): C_U = min{2/a, N_ex(a)} (C_s + C_filt) on Q_U = Q_walk + 1 + (Q(Psi_s) - 2N)
  qubits: the ratio estimator <Pi O>/<Pi> (Pi the filter's success flag; <= 2/a [derived;
  overcounts by up to 2x at a ~ 1]) or exact amplitude amplification of the filtered state on an
  estimate a^, N_ex(a) = 2 ceil(pi/(4 asin sqrt(a)) - 1/2) + 1 calls [cited: Brassard, Hoyer, Mosca,
  Tapp, Contemp. Math. 305, 53 (2002), Theorem 4]: N_ex = 3 on [0.25, 1) (the ratio estimator is
  cheaper above a = 2/3), 5 on [sin^2(pi/10), 1/4) = [0.0955, 0.25), 7 on [0.0495, 0.0955) [derived].
  a^ comes from one amplitude estimation of the flag to delta_a, at about pi kappa sqrt(a(1 - a))/delta_a
  calls of C_s + C_filt [estimate], added once to the row (it serves every element). The mismatch
  leaks w_AA = sin^2((N_ex/2) delta_a / sqrt(a(1 - a))) of the amplified state onto the flag's bad
  branch; delta_a is the largest with 2 sigma_0 sqrt(w_AA) + w_AA r_O <= share eps, and the exact-AA
  copy's leak bound is F47 with q = w_f + w_AA (``GroundCopy.exact_aa_leak_bound``), larger than the
  ratio estimator's, which the row's budget then carries [design; the flag sectors are orthogonal, so
  the AA leakage biases a system observable only linearly (w r_O) and F47's cross term is conservative].
  :func:`route_b_rows` takes the cheaper of the two options for the whole row (the estimate included);
  ``GroundCopy.coherent`` records the cheaper per copy (min{2/a, N_ex}).
* model bias of H': the exact property error of psi_0' when the caller has it (``model_bias``;
  :func:`model_property_errors`), else F47 with q = 1 - F (flag "model_bias_bound").

On exact spectra of H' [build check: the constructed filter of ``ft.amplify.step_filter_polynomial``
applied to HF of h4, lih and n2s at eps_f = 1e-2 and 1e-3]: the success probability lies in its
interval, the leaked weight is 0.014-0.21 of its bound, the property errors of the filtered state stay
below F47, and adversarial stop-band signs reach 0.95 (h4 NONs), 0.56 (LiH dipole) and 0.55 (n2s
NONs) of the first-order bound 2 sigma_0 eps_f ((1 - p)/p)^(1/2).

Hellmann-Feynman QPE (F51) [derived; the theorem is standard]
------------------------------------------------------------
<psi_0'|O|psi_0'> = dE_0'(h)/dh for H' + hO (:func:`hellmann_feynman_cost`). Central stencil: bias
|E'''| h^2/6, noise sigma_E/(sqrt(2) h), 2 energies; five-point: bias |E^(5)| h^4/30, noise
sqrt(130) sigma_E/(12 h), 4 energies. One QPE run reaches RMS sigma_E at C_QPE dE/sigma_E Toffolis
(F9's 1/dE law; conservative by sin(theta_0), F39) and is accepted by rule B after the best realizable
pre-projection of F43: T = n_E min_s T_s over rep ((C_s + L)/p + (n_v - 1) L, n_v of F40 on the long
QPE's grid), lt_bbht and lt_bbht_L (E_BBHT(p)(C_s + C_filt) + L), 2s and 2s_L ((C_s + C_c)/p + L),
L = C_QPE dE/sigma_E, the U forms with a certified U and the _L forms where they terminate (filters and
coarse QPE on H' + hO costed as on H' [estimate]). The optimum is b = eps/sqrt(3), h =
(6 eps/(sqrt(3)|E'''|))^(1/2), sigma_E = (2/sqrt(3)) h eps (central), b = eps/sqrt(5), h =
(30 eps/(sqrt(5)|E^(5)|))^(1/4) (five-point); under repetition T = sqrt(3) C_QPE dE/(h eps p) + 2 C_s/p
(central, n_v = 1), and with a pre-projection the 1/p moves to the projection term at the same h. E'''
from finite-field FCI (:func:`finite_field_derivatives`); the growth of lambda by |h| alpha_O is not
counted [estimate]. [build check: LiH, 41 fields on [-0.02, 0.02] a.u., degree 8: E''' = 581.6,
E^(5) = 6.97e6 a.u., dE/dh = -1.817719 against <mu> = -1.8177157; the central bias at h = 1e-3 is
9.685e-5 a.u. (E''' h^2/6 to 0.1%); at eps = 4e-3 a.u. h = 4.88e-3 a.u. and T = 91.1 C_QPE under
repetition from HF (p = 0.974).]

The model bias b_m of H' is systematic, like the stencil's, and adds to it linearly (as in route B):
the estimate meets (b_m + b)^2 + sigma_d^2 <= eps^2, and the stencil bias maximizes h(b) sigma_d(b),
i.e. 3 b^2 + 4 b_m b + b_m^2 = eps^2 (central, h ~ b^(1/2)) and 5 b^2 + 6 b_m b + b_m^2 = eps^2
(five-point, h ~ b^(1/4)):

    b = ((b_m^2 + 3 eps^2)^(1/2) - 2 b_m)/3,   b = ((4 b_m^2 + 5 eps^2)^(1/2) - 3 b_m)/5,

sigma_d = (eps^2 - (b_m + b)^2)^(1/2), bias b_m + b; b_m = 0 gives the optimum above, and the estimate
is infeasible iff b_m >= eps [derived]. F51's leading-order h is then refined against the finite-field
values of the emulated model [design]: :class:`FiniteFieldStencil` evaluates the stencil on the FCI
energies of H + hO at +-h (and +-2h) by the path of :func:`finite_field_derivatives`, and
:func:`hellmann_feynman_cost` shrinks h by bisection (12 steps) until the actual stencil bias is within
the allocated b, so b stays a bound on the stencil's bias at the chosen h; a stencil whose bias still
exceeds b at h/2^12 (e.g. where the convergence noise of the energies, ~conv_tol/h, reaches b) has no
admissible step at eps, and the HF-QPE rows use the other stencil or are infeasible (flag
"no_field_step:<stencil>"). [build check: LiH dipole with b_m = 4.71e-4 a.u. (SF L = 16): at eps =
1e-3 a.u. (five-point) b = 2.025e-4, and the leading-order h = 5.434e-3 a.u. carries an actual bias of
1.105 b; the refined h = 5.304e-3 a.u. raises n_E C_QPE dE/sigma_E from 946.6 to 969.9 C_QPE. At 4e-3
a.u. (central) b = 2.000e-3, the leading-order h = 4.543e-3 a.u. carries 1.011 b, and the refined
h = 4.517e-3 a.u. gives 99.55 C_QPE (98.99 at the leading order).]

Property-aware encoding selection (F52) [design]
-----------------------------------------------
F9 bounds the energy only (its infidelity <= 1e-4 guarantees b_fact <= 0.02 sigma_0 by F47). For a
property P at eps (:func:`select_for_property`): among the scanned encodings with |de_fact| <= 0.5 mHa
the cheapest QPE whose exact b_fact(P) <= eps/2; if there is none, the cheapest scanned encoding with
b_fact(P) <= eps/2 regardless of F9. [build check: on n2s the selected SF L = 17 has b_fact = 9.68e-4
(NONs), 1.99e-3 (1-RDM) and 1.19e-2 (2-RDM); the property-aware choice at 1e-3 (1e-2 for the 2-RDM)
is SF L = 18, +2.58% C_QPE, with 1.47e-4, 2.94e-4 and 1.68e-3.]

Comparison (F53) [design; F18, F19]
----------------------------------
For each (P, eps): the cheapest of route A at the fixed budgets (per loop; unusable rows reported, the
A-oracle row reported only), route B from HF or Psi_1 (B0), route B from the best Psi_k with its loop
(Bk) and Hellmann-Feynman QPE (scalars), per estimator, AE at kappa = 1 the reference
(:func:`best_property_row`; :class:`PropertyRow`). Shots and per-element AE runs parallelize over
machines (F19); single coherent runs (one AE element, one AA sequence, one long QPE) do not. Every row's
longest coherent run is laid out by ``ft.physical.surface_code_cost`` (F18) at its budget, p_fail_ae (a
shadow snapshot at p_fail_shot with its readout; :func:`run_layout`): no factory fails below 3.6e-16 per
Toffoli [OF], so a single run at p_fail_ae = 1e-4 has no layout beyond 2.8e11 Toffolis, and such rows
carry the flag "no_layout"; they stay selectable (the comparison is in Toffolis) and are reported. Over the
(system, target) pairs of the benchmark's outputs (:func:`q6_comparison`, :func:`q6_statements`): the
routes of the best rows with route A at its cheapest usable budget and at each single pre-declared
budget (:func:`route_tally`, :func:`best_at_budget`; route A's usability is an emulation fact, F49), the
same with route A on the F25-sliced compiled iterates, the pairs bias-limited only on the compiled
iterates (:func:`compiled_bias_limited`), and the best-row cells that rest on the exact model bias
b_fact, i.e. change when routes B and HF-QPE charge F47 at q = 1 - F (:func:`model_bias_sensitivity`).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..fci_ops import FCISpace, normalize, one_body, two_body
from ..integrals import ActiveSpace
from ..measure.pauli import PauliSum, QubitMapping, QubitState, jw_one_body, jw_two_body
from .amplify import W_L, FilterCost, coarse_qpe_toffolis, exponential_search_calls, filter_cost
from .physical import PhysicalCost, SurfaceCodeModel, surface_code_cost
from .primitives import FTParameters
from .qpe import BlockEncoding, QPECost, qpe_cost
from .residual import PrepParams, RouteCost, ae_allocation
from .stateprep import PrepCost, prep_cost
from .trajectory import TrajectoryRecord

__all__ = [
    "TARGET_KINDS",
    "NORMS",
    "TARGET_NORMS",
    "ACCURACIES",
    "PROPERTY_ROUTES",
    "PropertyTarget",
    "rdm1_target",
    "rdm2_target",
    "occupation_target",
    "cumulant_target",
    "cumulant_norm_target",
    "dipole_target",
    "natural_orbitals",
    "cumulant",
    "property_values",
    "property_error",
    "bias_bound",
    "PropertyInputs",
    "property_inputs",
    "shadow_cost",
    "ae_cost",
    "gradient_cost",
    "crossover_accuracy",
    "exact_aa_calls",
    "GroundCopy",
    "ground_copy",
    "STENCILS",
    "hellmann_feynman_cost",
    "finite_field_derivatives",
    "FiniteFieldStencil",
    "model_property_errors",
    "select_for_property",
    "PropertyRow",
    "route_a_rows",
    "route_b_rows",
    "best_property_row",
    "run_layout",
    "model_bias_bound",
    "row_from_json",
    "best_at_budget",
    "route_tally",
    "property_pairs",
    "compiled_bias_limited",
    "model_bias_sensitivity",
    "q6_comparison",
    "q6_statements",
    "compiled_iterates",
    "sliced_prep_cost",
]

TARGET_KINDS = ("occupations", "rdm1", "rdm2", "cumulant", "cumulant_norm", "dipole")
NORMS = ("fro", "max", "scalar")
#: The norm of each target kind (F46).
TARGET_NORMS = {"occupations": "max", "rdm1": "fro", "rdm2": "fro", "cumulant": "fro",
                "cumulant_norm": "scalar", "dipole": "scalar"}
#: The accuracies of the property study, by target kind (electrons; the dipole in a.u.).
ACCURACIES = {"occupations": (1e-2, 1e-3), "rdm1": (1e-2, 1e-3), "rdm2": (1e-1, 1e-2),
              "cumulant": (1e-1, 1e-2), "cumulant_norm": (1e-2, 1e-3), "dipole": (4e-3, 1e-3)}
#: Routes of a :class:`PropertyRow` (F53).
PROPERTY_ROUTES = ("A", "A-oracle", "B0", "Bk", "HF-QPE")
# Budget round-off rule of F10 (ft.residual): infeasible if B^2 <= 1e-12 eps^2.
_BUDGET_RTOL = 1e-12
# Natural-orbital blocks: |gamma[p,q]| above this couples p and q.
_BLOCK_TOL = 1e-10
# Lanczos tolerance of the sector ranges r_O; sectors up to this size are diagonalized densely.
_RANGE_TOL = 1e-10
_RANGE_DENSE = 64
# The filter error cap and the leakage share of F50; the exact-AA estimate's delta_a cap (fraction of a).
_EPS_F_CAP = 1e-2
_DELTA_A_CAP = 0.5
# F51: the coarse-QPE leakage of the 2s forms (F43).
_EPS_P = 1e-4
# Overlaps above 1 - this are read as 1; weights and energy errors down to -this are round-off zeros (F47).
_ONE_TOL = 1e-15
_ROUND_OFF = 1e-12


# ------------------------------------------------------------------ validation helpers
def _finite(value, name: str) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return v


def _positive(value, name: str) -> float:
    v = _finite(value, name)
    if not v > 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return v


def _nonnegative(value, name: str) -> float:
    v = _finite(value, name)
    if v < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return v


def _cost(value, name: str) -> float:
    """A non-negative cost, +inf allowed (an infeasible estimate)."""
    v = float(value)
    if math.isnan(v) or v < 0.0:
        raise ValueError(f"{name} must be non-negative (inf allowed), got {value!r}")
    return v


def _int_at_least(value, low: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) < low:
        raise ValueError(f"{name} must be an integer >= {low}, got {value!r}")
    return int(value)


def _overlap(value, name: str = "p") -> float:
    v = _finite(value, name)
    if not 0.0 < v <= 1.0 + 1e-12:
        raise ValueError(f"{name} must lie in (0, 1], got {value!r}")
    return min(v, 1.0)


def _frozen(a: np.ndarray) -> np.ndarray:
    a = np.array(a)
    a.setflags(write=False)
    return a


def _real_ci(ci) -> np.ndarray:
    a = np.asarray(ci)
    if np.iscomplexobj(a):
        if np.max(np.abs(a.imag), initial=0.0) > 1e-13 * max(1.0, float(np.max(np.abs(a)))):
            raise ValueError("property states must be real (the targets assume real states)")
        a = a.real
    return np.asarray(a, dtype=float)


def _state(active: ActiveSpace, ci) -> tuple[FCISpace, np.ndarray]:
    space = FCISpace.of(active)
    a = _real_ci(ci)
    if a.size != space.size:
        raise ValueError(f"the CI vector has {a.size} entries, the sector {space.size}")
    return space, normalize(a.reshape(space.shape))


def _rdms(active: ActiveSpace, ci) -> tuple[np.ndarray, np.ndarray]:
    from ..residual import rdm12

    space, psi = _state(active, ci)
    dm1, dm2 = rdm12(space, psi)
    return np.asarray(np.real(dm1), dtype=float), np.asarray(np.real(dm2), dtype=float)


# ------------------------------------------------------------------ targets (F46)
@dataclass(frozen=True, eq=False)
class PropertyTarget:
    """K Hermitian element operators with multiplicities and a norm (F46, module docstring).

    Attributes:
        kind: one of :data:`TARGET_KINDS`.
        norm: one of :data:`NORMS` ("scalar" iff K = 1 and no element indices).
        elements: (K, 2) or (K, 4) orbital indices (natural-orbital indices (i, i) for
            "occupations"), shape (0,) for scalars.
        m: (K,) positive multiplicities; sum_e m_e x_e^2 = ||x||_F^2 of the full tensor for "fro".
        one_body: per element an (N, N) f (O_e contains sum f[p,q] E_pq) or None.
        two_body: per element an (N, N, N, N) c (O_e contains sum c[p,q,r,s] e_{pq,rs}) or None.
        constant: added to every element's operator (the nuclear dipole; 0 otherwise).
    """

    kind: str
    norm: str
    elements: np.ndarray
    m: np.ndarray
    one_body: tuple
    two_body: tuple
    constant: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in TARGET_KINDS:
            raise ValueError(f"kind must be one of {TARGET_KINDS}, got {self.kind!r}")
        if self.norm not in NORMS:
            raise ValueError(f"norm must be one of {NORMS}, got {self.norm!r}")
        m = np.asarray(self.m, dtype=float)
        if m.ndim != 1 or m.size == 0 or not np.all(np.isfinite(m)) or np.any(m <= 0.0):
            raise ValueError("m must be a non-empty 1-D array of positive multiplicities")
        k = m.size
        el = np.asarray(self.elements)
        if self.norm == "scalar":
            if k != 1 or el.size != 0:
                raise ValueError("a scalar target has one element (m = [m_0]) and no element indices")
            el = np.zeros((0,), dtype=np.int64)
        else:
            if el.dtype.kind not in "iu" or el.ndim != 2 or el.shape[0] != k or el.shape[1] not in (2, 4):
                raise ValueError(f"elements must be integer (K, 2) or (K, 4) with K = {k}, got {el.shape}")
            el = el.astype(np.int64)
        ob, tb = tuple(self.one_body), tuple(self.two_body)
        if len(ob) != k or len(tb) != k:
            raise ValueError(f"one_body and two_body need one entry per element (K = {k})")
        n = None
        fixed_ob, fixed_tb = [], []
        for e, (f, c) in enumerate(zip(ob, tb)):
            if f is None and c is None:
                raise ValueError(f"element {e} has neither a one-body nor a two-body part")
            if f is not None:
                f = np.asarray(f, dtype=float)
                if f.ndim != 2 or f.shape[0] != f.shape[1] or not np.all(np.isfinite(f)):
                    raise ValueError(f"one_body[{e}] must be a finite square matrix")
                n = f.shape[0] if n is None else n
                if f.shape != (n, n):
                    raise ValueError(f"one_body[{e}] has shape {f.shape}, expected {(n, n)}")
                f = _frozen(f)
            if c is not None:
                c = np.asarray(c, dtype=float)
                if c.ndim != 4 or not np.all(np.isfinite(c)):
                    raise ValueError(f"two_body[{e}] must be a finite (N, N, N, N) array")
                n = c.shape[0] if n is None else n
                if c.shape != (n,) * 4:
                    raise ValueError(f"two_body[{e}] has shape {c.shape}, expected {(n,) * 4}")
                c = _frozen(c)
            fixed_ob.append(f)
            fixed_tb.append(c)
        if el.size and (el.min() < 0 or el.max() >= n):
            raise ValueError(f"element indices must lie in 0..{n - 1}")
        object.__setattr__(self, "elements", _frozen(el))
        object.__setattr__(self, "m", _frozen(m))
        object.__setattr__(self, "one_body", tuple(fixed_ob))
        object.__setattr__(self, "two_body", tuple(fixed_tb))
        object.__setattr__(self, "constant", _finite(self.constant, "constant"))

    @property
    def n_elements(self) -> int:
        """K."""
        return int(self.m.size)

    @property
    def norb(self) -> int:
        """N, the orbitals of the operators."""
        for f, c in zip(self.one_body, self.two_body):
            if f is not None:
                return int(f.shape[0])
            return int(c.shape[0])
        raise AssertionError("unreachable")                 # pragma: no cover

    def _check_active(self, active: ActiveSpace) -> None:
        if active.norb != self.norb:
            raise ValueError(f"the target has {self.norb} orbitals, the active space {active.norb}")

    def _apply(self, space: FCISpace, e: int, psi: np.ndarray) -> np.ndarray:
        """(O_e - constant) |psi>."""
        f, c = self.one_body[e], self.two_body[e]
        out = np.zeros(space.shape)
        if f is not None:
            out = out + np.asarray(one_body(space, f, psi))
        if c is not None:
            out = out + np.asarray(two_body(space, c, psi))
        return out

    def operators(self, mapping: QubitMapping | None = None) -> list[PauliSum]:
        """The JW Pauli sums of the elements (``measure.pauli``; blocked order by default)."""
        mp = QubitMapping(self.norb) if mapping is None else mapping
        if mp.norb != self.norb:
            raise ValueError(f"mapping has {mp.norb} orbitals, the target {self.norb}")
        out = []
        for f, c in zip(self.one_body, self.two_body):
            parts = []
            if f is not None:
                parts.append(jw_one_body(mp, f))
            if c is not None:
                parts.append(jw_two_body(mp, c))
            if self.constant != 0.0:
                parts.append(PauliSum.identity(mp.n_qubits, self.constant))
            op = parts[0]
            for x in parts[1:]:
                op = op + x
            out.append(op)
        return out

    def expectations(self, active: ActiveSpace, ci) -> np.ndarray:
        """(K,) <O_e> in the normalized state ``ci`` (CI space, ``fci_ops``)."""
        self._check_active(active)
        space, psi = _state(active, ci)
        return np.array([float(np.vdot(psi, self._apply(space, e, psi)).real) + self.constant
                         for e in range(self.n_elements)])

    def spreads(self, active: ActiveSpace, ci) -> np.ndarray:
        """(K,) sigma_0(O_e) = (<O_e^2> - <O_e>^2)^(1/2) in the normalized state ``ci``."""
        self._check_active(active)
        space, psi = _state(active, ci)
        out = np.empty(self.n_elements)
        for e in range(self.n_elements):
            v = self._apply(space, e, psi)
            mean = float(np.vdot(psi, v).real)
            out[e] = math.sqrt(max(float(np.vdot(v, v).real) - mean * mean, 0.0))
        return out

    def ranges(self, active: ActiveSpace, ci) -> np.ndarray:
        """(K,) r_O = ||O_e - <O_e>|| on the (n_alpha, n_beta) sector, <O_e> in ``ci`` (F47).

        The largest |eigenvalue - <O_e>| of O_e restricted to the sector: Lanczos (ARPACK, largest
        magnitude, relative tolerance 1e-10) on the shifted operator, dense for sectors of <= 64
        determinants.
        """
        import scipy.sparse.linalg as spla

        self._check_active(active)
        space, psi = _state(active, ci)
        n = space.size
        out = np.empty(self.n_elements)
        rng = np.random.default_rng(20261108)
        v0 = rng.standard_normal(n)
        for e in range(self.n_elements):
            mean = float(np.vdot(psi, self._apply(space, e, psi)).real)

            def matvec(x, e=e, mean=mean):
                x = np.asarray(x, dtype=float).reshape(space.shape)
                return (self._apply(space, e, x) - mean * x).ravel()

            if n <= _RANGE_DENSE:
                mat = np.column_stack([matvec(col) for col in np.eye(n)])
                w = np.linalg.eigvalsh(0.5 * (mat + mat.T))
                out[e] = float(np.max(np.abs(w)))
                continue
            op = spla.LinearOperator((n, n), matvec=matvec, dtype=float)
            w = spla.eigsh(op, k=1, which="LM", tol=_RANGE_TOL, v0=v0, return_eigenvectors=False)
            out[e] = float(np.max(np.abs(w)))
        return out


def _rdm1_elements(irreps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = irreps.size
    el = [(p, q) for p in range(n) for q in range(p, n) if irreps[p] == irreps[q]]
    m = [1.0 if p == q else 2.0 for p, q in el]
    return np.array(el, dtype=np.int64).reshape(-1, 2), np.array(m)


def _rdm2_orbits(irreps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Representatives (lexicographic minimum) and sizes of the {1, P, T, PT} orbits allowed by F2a."""
    n = irreps.size
    seen: set = set()
    el, m = [], []
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    if irreps[p] ^ irreps[q] ^ irreps[r] ^ irreps[s]:
                        continue
                    orbit = {(p, q, r, s), (r, s, p, q), (q, p, s, r), (s, r, q, p)}
                    rep = min(orbit)
                    if rep in seen:
                        continue
                    seen.add(rep)
                    el.append(rep)
                    m.append(float(len(orbit)))
    return np.array(el, dtype=np.int64).reshape(-1, 4), np.array(m)


def _check_irreps(active: ActiveSpace, irreps) -> np.ndarray:
    ir = np.asarray(irreps)
    if ir.dtype.kind not in "iu" or ir.shape != (active.norb,):
        raise ValueError(f"irreps must be ({active.norb},) integers, got {ir.dtype} {ir.shape}")
    ir = ir.astype(np.int64)
    if np.any((ir < 0) | (ir > 7)):
        raise ValueError("irrep ids of D2h and its subgroups lie in 0..7")
    return ir


def _check_dm1(active: ActiveSpace, dm1) -> np.ndarray:
    g = np.asarray(dm1, dtype=float)
    if g.shape != (active.norb, active.norb) or not np.all(np.isfinite(g)):
        raise ValueError(f"dm1 must be a finite ({active.norb}, {active.norb}) matrix")
    return 0.5 * (g + g.T)


def _rdm2_operator(n: int, p: int, q: int, r: int, s: int) -> np.ndarray:
    c = np.zeros((n, n, n, n))
    c[p, q, r, s] += 0.5
    c[q, p, s, r] += 0.5
    return c


def rdm1_target(active: ActiveSpace, irreps) -> PropertyTarget:
    """The 1-RDM: O_pq = (E_pq + E_qp)/2 for p <= q of one irrep, m = 2 off the diagonal (F46)."""
    ir = _check_irreps(active, irreps)
    n = active.norb
    el, m = _rdm1_elements(ir)
    fs = []
    for p, q in el:
        f = np.zeros((n, n))
        f[p, q] += 0.5
        f[q, p] += 0.5
        fs.append(f)
    return PropertyTarget("rdm1", "fro", el, m, tuple(fs), (None,) * len(fs))


def rdm2_target(active: ActiveSpace, irreps) -> PropertyTarget:
    """The 2-RDM: O_e = (e_{pq,rs} + e_{qp,sr})/2 on the F2a-allowed {1, P, T, PT} orbits, m = orbit size (F46)."""
    ir = _check_irreps(active, irreps)
    n = active.norb
    el, m = _rdm2_orbits(ir)
    cs = tuple(_rdm2_operator(n, *map(int, e)) for e in el)
    return PropertyTarget("rdm2", "fro", el, m, (None,) * len(cs), cs)


def natural_orbitals(dm1) -> tuple[np.ndarray, np.ndarray]:
    """(occupations n_i, orbitals v[:, i]) of a 1-RDM, block by block, sorted by occupation (descending).

    The blocks are the connected components of |gamma[p,q]| > 1e-10 (the symmetry blocks of a
    symmetric state), so degenerate occupations of different blocks are never mixed.
    """
    from scipy.sparse.csgraph import connected_components

    g = np.asarray(dm1, dtype=float)
    if g.ndim != 2 or g.shape[0] != g.shape[1] or not np.all(np.isfinite(g)):
        raise ValueError("dm1 must be a finite square matrix")
    g = 0.5 * (g + g.T)
    n = g.shape[0]
    adj = np.abs(g) > _BLOCK_TOL
    n_blocks, label = connected_components(adj, directed=False)
    occ = np.empty(n)
    vec = np.zeros((n, n))
    col = 0
    for b in range(n_blocks):
        idx = np.flatnonzero(label == b)
        w, v = np.linalg.eigh(g[np.ix_(idx, idx)])
        for j in range(idx.size):
            occ[col] = w[j]
            vec[idx, col] = v[:, j]
            col += 1
    order = np.argsort(-occ, kind="stable")
    return occ[order], vec[:, order]


def occupation_target(active: ActiveSpace, dm1) -> PropertyTarget:
    """Natural occupations, linearized at the natural orbitals v_i of ``dm1``: N_i = sum_pq v_ip v_iq E_qp (F46)."""
    g = _check_dm1(active, dm1)
    _, v = natural_orbitals(g)
    n = active.norb
    el = np.repeat(np.arange(n, dtype=np.int64)[:, None], 2, axis=1)
    fs = tuple(np.outer(v[:, i], v[:, i]) for i in range(n))
    return PropertyTarget("occupations", "max", el, np.ones(n), fs, (None,) * n)


def _cumulant_parts(n: int, dm1: np.ndarray, e) -> tuple[np.ndarray, np.ndarray]:
    """(f, c) of the linearized cumulant operator O^Delta_e, averaged with e's Hermitian partner."""
    p, q, r, s = (int(x) for x in e)
    c = _rdm2_operator(n, p, q, r, s)
    f = np.zeros((n, n))

    def add(i, j, x):                                       # x E^_ij, E^_ij = (E_ij + E_ji)/2
        f[i, j] += 0.5 * x
        f[j, i] += 0.5 * x

    for pp, qq, rr, ss in ((p, q, r, s), (q, p, s, r)):
        add(pp, qq, -0.5 * dm1[rr, ss])
        add(rr, ss, -0.5 * dm1[pp, qq])
        add(pp, ss, 0.25 * dm1[rr, qq])
        add(rr, qq, 0.25 * dm1[pp, ss])
    return f, c


def cumulant(dm1, dm2) -> np.ndarray:
    """Delta[p,q,r,s] = D[p,q,r,s] - gamma[p,q] gamma[r,s] + 1/2 gamma[p,s] gamma[r,q] (spin-summed singlet, F46)."""
    g = np.asarray(dm1, dtype=float)
    d = np.asarray(dm2, dtype=float)
    return d - np.einsum("pq,rs->pqrs", g, g) + 0.5 * np.einsum("ps,rq->pqrs", g, g)


def cumulant_target(active: ActiveSpace, irreps, dm1) -> PropertyTarget:
    """The cumulant on the rdm2 orbits, element operators linearized at ``dm1`` (F46)."""
    ir = _check_irreps(active, irreps)
    g = _check_dm1(active, dm1)
    n = active.norb
    el, m = _rdm2_orbits(ir)
    parts = [_cumulant_parts(n, g, e) for e in el]
    return PropertyTarget("cumulant", "fro", el, m, tuple(f for f, _ in parts), tuple(c for _, c in parts))


def cumulant_norm_target(active: ActiveSpace, irreps, dm1, dm2) -> PropertyTarget:
    """||Delta||_F linearized at (dm1, dm2): sum_e m_e (Delta_e/||Delta||_F) O^Delta_e (F46).

    F47 on this operator bounds the norm's error to first order only (it fails at q >~ 0.1);
    ``bias_bound(..., cumulant_parts=(cumulant target, rdm1 target))`` bounds it to all orders (module
    docstring). Raises ValueError if Delta(dm1, dm2) vanishes (a determinant): the norm has no derivative
    there.
    """
    ir = _check_irreps(active, irreps)
    g = _check_dm1(active, dm1)
    d = np.asarray(dm2, dtype=float)
    n = active.norb
    if d.shape != (n,) * 4 or not np.all(np.isfinite(d)):
        raise ValueError(f"dm2 must be a finite {(n,) * 4} array")
    delta = cumulant(g, d)
    norm = float(np.linalg.norm(delta))
    if not norm > 1e-12:
        raise ValueError("the cumulant vanishes at the reference: its norm cannot be linearized")
    el, m = _rdm2_orbits(ir)
    f_tot = np.zeros((n, n))
    c_tot = np.zeros((n,) * 4)
    for e, mm in zip(el, m):
        f, c = _cumulant_parts(n, g, e)
        w = mm * delta[tuple(e)] / norm
        f_tot += w * f
        c_tot += w * c
    return PropertyTarget("cumulant_norm", "scalar", np.zeros((0,), dtype=np.int64), np.ones(1),
                          (f_tot,), (c_tot,))


def dipole_target(active: ActiveSpace, axis: int = 2) -> PropertyTarget:
    """mu_axis = sum_A Z_A R_A,axis - sum_pq r[p,q] E_pq (a.u., origin 0; F46); all-active builds only.

    Raises ValueError unless ``active`` carries its molecule and orbitals, every electron is active
    and every orbital is active (a frozen core would add a constant that is not implemented).
    """
    axis = _int_at_least(axis, 0, "axis")
    if axis > 2:
        raise ValueError(f"axis must be 0, 1 or 2, got {axis}")
    mol, c = active.mol, active.mo_coeff
    if mol is None or c is None:
        raise ValueError("the dipole needs active.mol and active.mo_coeff")
    if active.n_electrons != mol.nelectron or c.shape[1] != mol.nao_nr() or c.shape[1] != active.norb:
        raise ValueError("the dipole target needs an all-active build (no frozen core, every orbital active)")
    with mol.with_common_orig((0.0, 0.0, 0.0)):
        r = mol.intor("int1e_r")[axis]
    r_mo = c.T @ r @ c
    nuc = float(np.dot(mol.atom_charges(), mol.atom_coords()[:, axis]))
    return PropertyTarget("dipole", "scalar", np.zeros((0,), dtype=np.int64), np.ones(1),
                          (-0.5 * (r_mo + r_mo.T),), (None,), constant=nuc)


def property_values(target: PropertyTarget, active: ActiveSpace, ci) -> np.ndarray:
    """The nonlinear functional of the target in ``ci``: RDM elements, sorted NONs, Delta, ||Delta||_F or mu (F46)."""
    target._check_active(active)
    dm1, dm2 = _rdms(active, ci)
    el = target.elements
    if target.kind == "rdm1":
        return dm1[el[:, 0], el[:, 1]].copy()
    if target.kind == "rdm2":
        return dm2[el[:, 0], el[:, 1], el[:, 2], el[:, 3]].copy()
    if target.kind == "occupations":
        return np.sort(np.linalg.eigvalsh(0.5 * (dm1 + dm1.T)))[::-1].copy()
    if target.kind == "cumulant":
        delta = cumulant(dm1, dm2)
        return delta[el[:, 0], el[:, 1], el[:, 2], el[:, 3]].copy()
    if target.kind == "cumulant_norm":
        return np.array([float(np.linalg.norm(cumulant(dm1, dm2)))])
    # dipole (and any one-body scalar): constant + sum_pq f[p,q] <E_pq>, <E_pq> = dm1[q,p]
    f = target.one_body[0]
    return np.array([target.constant + float(np.sum(f * dm1.T))])


def _norm(target: PropertyTarget, d: np.ndarray) -> float:
    d = np.asarray(d, dtype=float)
    if target.norm == "fro":
        return float(math.sqrt(float(np.sum(target.m * d * d))))
    return float(np.max(np.abs(d)))


def property_error(target: PropertyTarget, active: ActiveSpace, ci, ci_ref) -> float:
    """||P(ci) - P(ci_ref)|| in the target's norm (fro, max or scalar; F46)."""
    return _norm(target, property_values(target, active, ci) - property_values(target, active, ci_ref))


def _f47_norm(target: PropertyTarget, s: float, t: float, sig: np.ndarray, rng: np.ndarray) -> float:
    """F47 per element in the target's norm: 2 s ||sigma||_m + t ||r||_m (fro), max_e (2 s sigma_e + t r_e)."""
    k = target.n_elements
    if sig.shape != (k,) or rng.shape != (k,) or np.any(sig < 0) or np.any(rng < 0):
        raise ValueError(f"sigma0 and ranges must be non-negative ({k},) arrays")
    if target.norm == "fro":
        return float(2.0 * s * math.sqrt(float(np.sum(target.m * sig ** 2)))
                     + t * math.sqrt(float(np.sum(target.m * rng ** 2))))
    return float(np.max(2.0 * s * sig + t * rng))


def _part_terms(part, active: ActiveSpace, ci_ref) -> tuple[PropertyTarget, np.ndarray, np.ndarray]:
    """(target, sigma_0, r_O) of a PropertyInputs (r0 computed if absent) or a PropertyTarget at ``ci_ref``."""
    if isinstance(part, PropertyInputs):
        t = part.target
        sig = np.asarray(part.sigma0, dtype=float)
        rng = t.ranges(active, ci_ref) if part.r0 is None else np.asarray(part.r0, dtype=float)
        return t, sig, rng
    if isinstance(part, PropertyTarget):
        return part, part.spreads(active, ci_ref), part.ranges(active, ci_ref)
    raise ValueError(f"cumulant_parts entries must be PropertyInputs or PropertyTarget, got {type(part).__name__}")


def bias_bound(target: PropertyTarget, active: ActiveSpace, ci_ref, *, q: float | None = None,
               energy_error: float | None = None, gap: float | None = None,
               sigma0: np.ndarray | None = None, ranges: np.ndarray | None = None,
               cumulant_parts: Sequence | None = None) -> float:
    """F47's bound on ||<O>_Psi - <O>_ref|| in the target's norm (module docstring).

    Either ``q`` (the weight of Psi outside psi_ref, q form) or ``energy_error`` d = E_Psi - E_ref and
    ``gap`` Delta_1 (energy form, q <= d/Delta_1). ``sigma0`` and ``ranges`` are the per-element
    sigma_0 and r_O at ``ci_ref`` if the caller has them (:meth:`PropertyTarget.spreads`,
    :meth:`PropertyTarget.ranges`); otherwise they are computed.

    For the nonlinear targets this bounds the error of the linearized operators, i.e. the property error to
    first order (module docstring). ``cumulant_parts``: for a "cumulant_norm" target, the cumulant tensor
    and the 1-RDM targets at ``ci_ref`` (each a :class:`PropertyInputs`, whose sigma0 and r0 are used, or
    a :class:`PropertyTarget`); the result is then B_cum + 3/2 B_1^2, a bound on
    | ||Delta_Psi||_F - ||Delta_ref||_F | to all orders (module docstring), and ``sigma0``/``ranges`` are
    not used.
    """
    if (q is None) == (energy_error is None):
        raise ValueError("give either q or energy_error (with gap)")
    if q is not None:
        if gap is not None:
            raise ValueError("gap belongs to the energy form")
        qq = _finite(q, "q")
        if not -_ROUND_OFF <= qq <= 1.0 + _ROUND_OFF:
            raise ValueError(f"q must lie in [0, 1], got {q!r}")
        qq = min(max(qq, 0.0), 1.0)
        s, t = math.sqrt(qq * (1.0 - qq)), qq
    else:
        if gap is None:
            raise ValueError("the energy form needs the gap")
        d = _finite(energy_error, "energy_error")
        if d < -_ROUND_OFF:
            raise ValueError(f"energy_error must be non-negative (E_Psi >= E_ref), got {energy_error!r}")
        x = max(d, 0.0) / _positive(gap, "gap")
        s, t = math.sqrt(x), x
    if cumulant_parts is not None:
        if target.kind != "cumulant_norm":
            raise ValueError(f"cumulant_parts belong to a 'cumulant_norm' target, got {target.kind!r}")
        parts = tuple(cumulant_parts)
        if len(parts) != 2:
            raise ValueError("cumulant_parts must be (the cumulant tensor target, the 1-RDM target)")
        (tc, sc, rc), (t1, s1, r1) = (_part_terms(x, active, ci_ref) for x in parts)
        if tc.kind != "cumulant" or t1.kind != "rdm1":
            raise ValueError(f"cumulant_parts must be ('cumulant', 'rdm1') targets, got ({tc.kind!r}, {t1.kind!r})")
        if tc.norb != target.norb or t1.norb != target.norb:
            raise ValueError("cumulant_parts must act on the target's orbitals")
        return _f47_norm(tc, s, t, sc, rc) + 1.5 * _f47_norm(t1, s, t, s1, r1) ** 2
    sig = target.spreads(active, ci_ref) if sigma0 is None else np.asarray(sigma0, dtype=float)
    rng = target.ranges(active, ci_ref) if ranges is None else np.asarray(ranges, dtype=float)
    return _f47_norm(target, s, t, sig, rng)


# ------------------------------------------------------------------ estimators (F48)
@dataclass(frozen=True, eq=False)
class PropertyInputs:
    """Per-element inputs of the estimator costs (F48; :func:`property_inputs`).

    Attributes:
        target: the property target.
        v_shadow: (K,) per-snapshot matchgate-shadow variances.
        alpha: (K,) Pauli 1-norms of O_e without the identity.
        n_terms: (K,) non-identity Pauli terms L_e.
        sigma0: (K,) ground-state spreads sigma_0(O_e).
        r0: (K,) ranges r_O,e at the same state (F47's second-order term), or None.
        method: the variance method of ``measure.schemes`` ("exact", "rb", "mc").
    """

    target: PropertyTarget
    v_shadow: np.ndarray
    alpha: np.ndarray
    n_terms: np.ndarray
    sigma0: np.ndarray
    r0: np.ndarray | None = None
    method: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.target, PropertyTarget):
            raise ValueError(f"target must be a PropertyTarget, got {type(self.target).__name__}")
        k = self.target.n_elements
        for name in ("v_shadow", "alpha", "n_terms", "sigma0") + (() if self.r0 is None else ("r0",)):
            a = np.asarray(getattr(self, name), dtype=float)
            if a.shape != (k,) or not np.all(np.isfinite(a)) or np.any(a < 0.0):
                raise ValueError(f"{name} must be a finite non-negative ({k},) array")
            object.__setattr__(self, name, _frozen(a))
        if not np.array_equal(self.n_terms, np.round(self.n_terms)):
            raise ValueError("n_terms must hold integers")

    @property
    def sigma_norm(self) -> float:
        """sigma_0 in the target's norm: (sum_e m_e sigma_0,e^2)^(1/2) (fro) or max_e sigma_0,e."""
        return _norm(self.target, self.sigma0)

    @property
    def range_norm(self) -> float | None:
        """r_O in the target's norm (None without ``r0``)."""
        return None if self.r0 is None else _norm(self.target, self.r0)


def property_inputs(active: ActiveSpace, ci, target: PropertyTarget, *, method: str = "auto",
                    ranges: bool = True) -> PropertyInputs:
    """Matchgate variances, Pauli 1-norms, term counts, spreads (and ranges) of ``target`` at ``ci`` (F48).

    ``method`` goes to ``measure.schemes.matchgate_shadow_variance`` ("auto": exact pair sums up to
    2e7 pairs, else Rao-Blackwellized Monte Carlo).
    """
    from ..measure import schemes

    target._check_active(active)
    _, psi = _state(active, ci)
    mp = QubitMapping(active.norb)
    ops = target.operators(mp)
    state = QubitState.from_ci(active.norb, active.nelec, psi, mp)
    res = schemes.matchgate_shadow_variance(ops, state, method=method)
    alpha, n_terms = [], []
    for op in ops:
        ident = (op.xs == 0) & (op.zs == 0)
        alpha.append(float(np.sum(np.abs(op.coeffs[~ident]))))
        n_terms.append(int(np.count_nonzero(~ident)))
    return PropertyInputs(target=target, v_shadow=np.atleast_1d(np.asarray(res.per_target, dtype=float)),
                          alpha=np.array(alpha), n_terms=np.array(n_terms, dtype=float),
                          sigma0=target.spreads(active, psi),
                          r0=target.ranges(active, psi) if ranges else None, method=res.method)


def _budget(eps: float, bias: float) -> float:
    """B = (eps^2 - b^2)^(1/2); 0 if B^2 <= 1e-12 eps^2 (F48, F10's round-off rule)."""
    e = _positive(eps, "eps")
    b = _cost(bias, "bias")
    if not math.isfinite(b):
        return 0.0
    b2 = e * e - b * b
    return math.sqrt(b2) if b2 > _BUDGET_RTOL * e * e else 0.0


def _infeasible(route: str, q: int, info: dict, flags: tuple[str, ...] = ()) -> RouteCost:
    return RouteCost(route, math.inf, math.inf, math.nan, math.inf, math.inf, int(q), dict(info, budget=0.0),
                     tuple(flags) + ("infeasible",))


def _check_inputs(inputs) -> PropertyInputs:
    if not isinstance(inputs, PropertyInputs):
        raise ValueError(f"inputs must be PropertyInputs, got {type(inputs).__name__}")
    return inputs


def _check_prep(prep) -> PrepParams:
    if not isinstance(prep, PrepParams):
        raise ValueError(f"prep must be a PrepParams, got {type(prep).__name__}")
    return prep


def shadow_cost(inputs: PropertyInputs, eps: float, bias: float, prep: PrepParams, *,
                params: FTParameters = FTParameters()) -> RouteCost:
    """Matchgate shadows (F48): S = sum_e m_e v_e/B^2 (fro) or max_e v_e/B^2 snapshots of C_copy = prep.toffolis."""
    inputs, prep = _check_inputs(inputs), _check_prep(prep)
    t = inputs.target
    var = float(np.sum(t.m * inputs.v_shadow)) if t.norm == "fro" else float(np.max(inputs.v_shadow))
    info = {"eps": float(eps), "bias": float(bias), "variance": var, "p_fail": params.p_fail_shot}
    b = _budget(eps, bias)
    if b == 0.0:
        return _infeasible("shadow", prep.logical_qubits, info)
    shots = var / (b * b)
    info["budget"] = b
    return RouteCost("shadow", shots, shots * prep.toffolis, prep.toffolis, prep.toffolis, shots,
                     prep.logical_qubits, info)


def _obs_qubits(inputs: PropertyInputs, params: FTParameters) -> np.ndarray:
    """Q_obs,e = ceil(log2 L_e) + chi + 4 (the Pauli LCU of O_e, as M3's ae_x) [estimate]."""
    l = np.maximum(inputs.n_terms, 1.0)
    return np.ceil(np.log2(l)).astype(np.int64) + params.chi + 4


def ae_cost(inputs: PropertyInputs, eps: float, bias: float, prep: PrepParams, *, kappa: float = 1.0,
            params: FTParameters = FTParameters()) -> RouteCost:
    """Per-element amplitude estimation with the Pauli LCU of O_e (F48, F11; module docstring).

    fro: the optimal allocation of ``ft.residual.ae_allocation`` (N_A = 2 pi kappa Gamma/B); max and
    scalar: eps_e = B for every element. C_U = prep.toffolis, Q_U = prep.logical_qubits.
    """
    inputs, prep = _check_inputs(inputs), _check_prep(prep)
    kappa = _positive(kappa, "kappa")
    t = inputs.target
    k = t.n_elements
    alpha = np.asarray(inputs.alpha, dtype=float)
    c_obs = 2.0 * inputs.n_terms
    q_circuit = prep.logical_qubits + _obs_qubits(inputs, params) + 1
    info = {"eps": float(eps), "bias": float(bias), "kappa": kappa, "p_fail": params.p_fail_ae}
    b = _budget(eps, bias)
    if b == 0.0:
        return _infeasible("ae", int(q_circuit.max()), info, ("estimate",))
    if t.norm == "fro":
        gamma_over_b, eps_e = ae_allocation(alpha, t.m, b)
        info["gamma"] = gamma_over_b * b
    else:
        eps_e = np.where(alpha > 0.0, b, 0.0)
    n_e = np.divide(2.0 * math.pi * kappa * alpha, eps_e, out=np.zeros(k), where=eps_e > 0)
    register = np.where(n_e > 2.0, np.ceil(np.log2(np.maximum(n_e, 2.0) / 2.0)), 0.0).astype(np.int64)
    q_run = q_circuit + register
    run = n_e * (prep.toffolis + c_obs + 0.5 * q_circuit)
    runs = n_e > 0
    n_total, t_total = float(n_e.sum()), float(run.sum())
    info.update(budget=b, max_applications=float(n_e.max()))
    return RouteCost("ae", n_total, t_total, t_total / n_total if n_total > 0 else math.nan,
                     float(run.max()), float(np.count_nonzero(runs)),
                     int(q_run[runs].max() if runs.any() else q_circuit.max()), info, ("estimate",),
                     run_toffolis=tuple(run[runs].tolist()))


def gradient_cost(inputs: PropertyInputs, eps: float, bias: float, prep: PrepParams, *, kappa_g: float,
                  b_theta: int = 16, params: FTParameters = FTParameters()) -> RouteCost:
    """Gradient estimation (F48; a band, never selected): N_GE queries of C_U + K (2 b_theta + 2) Toffolis."""
    inputs, prep = _check_inputs(inputs), _check_prep(prep)
    kappa_g = _positive(kappa_g, "kappa_g")
    b_theta = _int_at_least(b_theta, 3, "b_theta")
    t = inputs.target
    k = t.n_elements
    alpha = np.asarray(inputs.alpha, dtype=float)
    q = int(prep.logical_qubits + k * b_theta + int(_obs_qubits(inputs, params).max()))
    info = {"eps": float(eps), "bias": float(bias), "kappa_g": kappa_g, "b_theta": b_theta}
    flags = ("estimate", "assumption")
    b = _budget(eps, bias)
    if b == 0.0:
        return _infeasible("gaw", q, info, flags)
    if t.norm == "fro":
        n = 2.0 * math.pi * kappa_g * math.sqrt(k) * math.sqrt(float(np.sum(t.m * alpha ** 2))) / b
    elif t.norm == "max":
        n = 2.0 * math.pi * kappa_g * math.sqrt(k) * float(np.max(alpha)) / b
    else:
        n = 2.0 * math.pi * kappa_g * float(alpha[0]) / b
    per = prep.toffolis + k * (2 * b_theta + 2)
    info.update(budget=b, c_probe=k * (2 * b_theta + 2))
    return RouteCost("gaw", n, n * per, per, n * per, 1.0, q, info, flags)


def crossover_accuracy(inputs: PropertyInputs, *, kappa: float = 1.0) -> float:
    """B* at which shadows and per-element AE need the same preparations (F48): AE is cheaper below it."""
    inputs = _check_inputs(inputs)
    kappa = _positive(kappa, "kappa")
    t = inputs.target
    alpha = np.asarray(inputs.alpha, dtype=float)
    if t.norm == "fro":
        gamma = float(np.sum(np.cbrt(t.m) * np.cbrt(alpha) ** 2)) ** 1.5
        return float(np.sum(t.m * inputs.v_shadow)) / (2.0 * math.pi * kappa * gamma)
    return float(np.max(inputs.v_shadow)) / (2.0 * math.pi * kappa * float(np.sum(alpha)))


# ------------------------------------------------------------------ route B copies (F50)
def exact_aa_calls(a: float) -> int:
    """N_ex(a) = 2 ceil(pi/(4 asin sqrt(a)) - 1/2) + 1 calls of exact AA with known a (F50); N_ex(1) = 1."""
    a = _overlap(a, "a")
    if a >= 1.0 - _ONE_TOL:
        return 1
    m = max(0, math.ceil(math.pi / (4.0 * math.asin(math.sqrt(a))) - 0.5 - 1e-12))
    return 2 * m + 1


@dataclass(frozen=True)
class GroundCopy:
    """A filtered ground-state copy of H' from a start state (F50, module docstring).

    Attributes:
        start: label of the start state.
        p: its overlap (the success probability a, module docstring).
        u_upper: the pass edge (a certified U, or E1lb - W_L).
        e1_lower: the stop edge E1lb.
        eps_f: the filter error.
        filter: the step filter (``ft.amplify.FilterCost``).
        copy_toffolis: C_copy = min(1/a, E_BBHT(a)) (C_s + C_filt) (destructive copy).
        unitary_toffolis: C_U = min{2/a, N_ex(a)} (C_s + C_filt) (coherent copy, the cheaper per copy).
        logical_qubits: Q_U = Q_walk + 1 + (Q(Psi_s) - 2N).
        leak_bound: F47 with q = w_f (the filter's leakage; destructive copy and ratio estimator).
        model_bias: the model bias of H' in the target's norm (exact, or F47 with q = 1 - F).
        coherent: "ratio" or "exact_aa", the cheaper coherent copy per copy.
        aa_calls: N_ex(a).
        a_estimate_toffolis: the one-time estimate of a for exact AA (0 for "ratio").
        flags: "termination_not_certified", "not_terminating", "model_bias_bound", "leak_first_order", ...
        ratio_toffolis, exact_aa_toffolis: (2/a) (C_s + C_filt) and N_ex (C_s + C_filt).
        exact_aa_estimate_toffolis: the estimate of a for exact AA, whichever option is cheaper per copy.
        exact_aa_leak_bound: F47 with q = w_f + w_AA (the exact-AA coherent copy).
        delta_a: the precision of a^.
    """

    start: str
    p: float
    u_upper: float
    e1_lower: float
    eps_f: float
    filter: FilterCost
    copy_toffolis: float
    unitary_toffolis: float
    logical_qubits: int
    leak_bound: float
    model_bias: float
    coherent: str
    aa_calls: int
    a_estimate_toffolis: float
    flags: tuple[str, ...]
    ratio_toffolis: float = math.nan
    exact_aa_toffolis: float = math.nan
    exact_aa_estimate_toffolis: float = math.nan
    exact_aa_leak_bound: float = math.nan
    delta_a: float = math.nan

    def __post_init__(self) -> None:
        if not isinstance(self.start, str):
            raise ValueError(f"start must be a label, got {self.start!r}")
        object.__setattr__(self, "p", _overlap(self.p))
        u = _finite(self.u_upper, "u_upper")
        e1 = _finite(self.e1_lower, "e1_lower")
        if not u < e1:
            raise ValueError(f"the pass edge must lie below E1lb, got U = {u!r}, E1lb = {e1!r}")
        if not isinstance(self.filter, FilterCost):
            raise ValueError("filter must be a FilterCost")
        if not 0.0 < float(self.eps_f) < 1.0:
            raise ValueError(f"eps_f must lie in (0, 1), got {self.eps_f!r}")
        for name in ("copy_toffolis", "unitary_toffolis", "leak_bound", "model_bias", "a_estimate_toffolis"):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        _int_at_least(self.logical_qubits, 1, "logical_qubits")
        _int_at_least(self.aa_calls, 1, "aa_calls")
        if self.coherent not in ("ratio", "exact_aa"):
            raise ValueError(f"coherent must be 'ratio' or 'exact_aa', got {self.coherent!r}")
        if self.coherent == "ratio" and self.a_estimate_toffolis != 0.0:
            raise ValueError("the ratio estimator needs no estimate of a")
        object.__setattr__(self, "flags", tuple(self.flags))

    @property
    def terminating(self) -> bool:
        return "not_terminating" not in self.flags


def _norb_of(enc: BlockEncoding, norb: int | None) -> int:
    if norb is not None:
        return _int_at_least(norb, 1, "norb")
    if enc.eri_model is None:
        raise ValueError("norb is needed when the encoding carries no model integrals")
    return int(np.asarray(enc.eri_model).shape[0])


def _f47(sigma: float, r: float | None, q: float) -> float:
    return 2.0 * sigma * math.sqrt(max(q * (1.0 - q), 0.0)) + (0.0 if r is None else q * r)


def ground_copy(enc: BlockEncoding, c: float, *, p: float, start_toffolis: float, start_qubits: int,
                u_upper: float | None, e1_lower: float, eps: float, sigma0: float, share: float = 0.1,
                params: FTParameters = FTParameters(), start: str = "", norb: int | None = None,
                r_o: float | None = None, e0_model: float | None = None, model_bias: float | None = None,
                kappa: float = 1.0) -> GroundCopy:
    """One filtered copy of psi_0' from a start state of overlap ``p`` (F50, module docstring).

    Args:
        enc, c: the encoding of H' and its walk offset (``ft.acceptance.walk_offset``).
        p, start_toffolis, start_qubits: the start's overlap, C(Psi_s) and Q(Psi_s).
        u_upper: a certified U (the pass edge), or None for the E1lb-referenced edge E1lb - W_L.
        e1_lower: E1lb, the stop edge.
        eps, sigma0: the property accuracy and the target's ground-state spread (its norm).
        share: the leakage share of eps.
        norb: N (default: from the encoding's model integrals).
        r_o: the target's range r_O (its norm) for F47's second-order term; None: first order only
            (flag "leak_first_order").
        e0_model: E_0' for the termination check of the E1lb-referenced edge (None: unchecked).
        model_bias: the exact model bias of H' for the target; None: F47 with q = enc.infidelity.
        kappa: the AE constant of the estimate of a.
    """
    if not isinstance(enc, BlockEncoding):
        raise ValueError(f"enc must be a BlockEncoding, got {type(enc).__name__}")
    p = _overlap(p)
    c_s = _nonnegative(start_toffolis, "start_toffolis")
    q_s = _int_at_least(start_qubits, 1, "start_qubits")
    e1 = _finite(e1_lower, "e1_lower")
    eps = _positive(eps, "eps")
    sigma = _nonnegative(sigma0, "sigma0")
    share = _positive(share, "share")
    kappa = _positive(kappa, "kappa")
    r = None if r_o is None else _nonnegative(r_o, "r_o")
    n = _norb_of(enc, norb)
    flags: list[str] = []
    if u_upper is None:
        u = e1 - W_L
        flags.append("termination_not_certified")
    else:
        u = _finite(u_upper, "u_upper")
        if not u < e1:
            raise ValueError(f"u_upper must lie below e1_lower, got {u!r} >= {e1!r}")
    if e0_model is not None and _finite(e0_model, "e0_model") > u:
        flags.append("not_terminating")
    if r is None:
        flags.append("leak_first_order")
    if p >= 1.0 - _ONE_TOL or sigma == 0.0:
        eps_f = _EPS_F_CAP
    else:
        eps_f = min(_EPS_F_CAP, share * eps * math.sqrt(p) / (2.0 * sigma * math.sqrt(1.0 - p)))
    filt = filter_cost(enc, c, u, e1, eps=eps_f, params=params)
    per_call = c_s + filt.toffolis
    w_f = eps_f ** 2 * (1.0 - p) / (p * (1.0 - eps_f) ** 2)
    leak = _f47(sigma, r, w_f)
    if model_bias is None:
        mb = _f47(sigma, r, float(enc.infidelity))
        flags.append("model_bias_bound")
    else:
        mb = _nonnegative(model_bias, "model_bias")
    a = p
    copy = min(1.0 / a, exponential_search_calls(a)) * per_call
    n_ex = exact_aa_calls(a)
    ratio = (2.0 / a) * per_call
    exact = n_ex * per_call
    # exact AA on an estimate a^: the largest delta_a with 2 sigma sqrt(w_AA) + w_AA r <= share eps
    target = share * eps
    if r is not None and r > 0.0:
        s_max = (-2.0 * sigma + math.sqrt(4.0 * sigma * sigma + 4.0 * r * target)) / (2.0 * r)
    elif sigma > 0.0:
        s_max = target / (2.0 * sigma)
    else:
        s_max = 1.0
    s_max = min(s_max, 1.0)
    spread = math.sqrt(max(a * (1.0 - a), 0.0))
    if spread > 0.0:
        delta_a = min(2.0 * math.asin(s_max) * spread / n_ex, _DELTA_A_CAP * a)
        w_aa = math.sin(0.5 * n_ex * delta_a / spread) ** 2
        n_est = math.pi * kappa * spread / delta_a
    else:                                                   # a = 1: nothing to amplify or estimate
        delta_a, w_aa, n_est = 0.0, 0.0, 0.0
    exact_leak = _f47(sigma, r, min(1.0, w_f + w_aa))
    est = n_est * per_call
    coherent = "exact_aa" if exact < ratio else "ratio"
    return GroundCopy(start=str(start), p=p, u_upper=u, e1_lower=e1, eps_f=eps_f, filter=filt,
                      copy_toffolis=copy, unitary_toffolis=min(ratio, exact),
                      logical_qubits=int(enc.ancilla_walk) + 1 + q_s - 2 * n, leak_bound=leak, model_bias=mb,
                      coherent=coherent, aa_calls=n_ex,
                      a_estimate_toffolis=est if coherent == "exact_aa" else 0.0, flags=tuple(flags),
                      ratio_toffolis=ratio, exact_aa_toffolis=exact, exact_aa_estimate_toffolis=est,
                      exact_aa_leak_bound=exact_leak, delta_a=delta_a)


# ------------------------------------------------------------------ Hellmann-Feynman QPE (F51)
STENCILS = ("central", "five_point")
# Bisection steps of the finite-field refinement of h, and the halvings allowed to bracket it.
_H_BISECTIONS = 12
_H_HALVINGS = 12


def _stencil_step(stencil: str, b: float, k: float) -> float:
    """F51's leading-order h for the stencil bias b: (6 b/|E'''|)^(1/2) (central), (30 b/|E^(5)|)^(1/4) (five-point)."""
    return math.sqrt(6.0 * b / k) if stencil == "central" else (30.0 * b / k) ** 0.25


def _stencil_noise(stencil: str, h: float) -> float:
    """sigma_d / sigma_E of the stencil at h: 1/(sqrt(2) h) (central), sqrt(130)/(12 h) (five-point)."""
    return 1.0 / (math.sqrt(2.0) * h) if stencil == "central" else math.sqrt(130.0) / (12.0 * h)


def _stencil_split(stencil: str, eps: float, fixed: float) -> float:
    """The stencil bias b maximizing h(b) (eps^2 - (fixed + b)^2)^(1/2) (F51 with a fixed bias; 0 < b < eps - fixed).

    Central (h ~ b^(1/2)): 3 b^2 + 4 b_m b + b_m^2 - eps^2 = 0; five-point (h ~ b^(1/4)):
    5 b^2 + 6 b_m b + b_m^2 - eps^2 = 0 [derived]; b_m = 0 gives eps/sqrt(3) and eps/sqrt(5).
    """
    if fixed == 0.0:
        return eps / math.sqrt(3.0 if stencil == "central" else 5.0)
    if stencil == "central":
        return (math.sqrt(fixed * fixed + 3.0 * eps * eps) - 2.0 * fixed) / 3.0
    return (math.sqrt(4.0 * fixed * fixed + 5.0 * eps * eps) - 3.0 * fixed) / 5.0


def _refine_step(h0: float, b: float, stencil: str, stencil_bias) -> tuple[float, float, int]:
    """The largest h <= h0 (to 2^-12 of the bracket) with |stencil_bias(h, stencil)| <= b, by halving and bisection.

    Returns (h, the signed stencil bias at h, the number of evaluations); if 12 halvings of h0 do not meet b
    there is no admissible step: (nan, the stencil bias at h0/2^12, the number of evaluations).
    """
    def bias(h: float) -> float:
        return _finite(stencil_bias(h, stencil), "stencil_bias(h)")

    n_eval = 1
    v0 = bias(h0)
    if abs(v0) <= b:
        return h0, v0, n_eval
    hi, lo = h0, 0.5 * h0
    v_lo = bias(lo)
    n_eval += 1
    halvings = 1
    while abs(v_lo) > b:
        if halvings >= _H_HALVINGS:
            return math.nan, v_lo, n_eval
        hi, lo = lo, 0.5 * lo
        v_lo = bias(lo)
        n_eval += 1
        halvings += 1
    for _ in range(_H_BISECTIONS):
        mid = 0.5 * (lo + hi)
        v_mid = bias(mid)
        n_eval += 1
        if abs(v_mid) <= b:
            lo, v_lo = mid, v_mid
        else:
            hi = mid
    return lo, v_lo, n_eval


def hellmann_feynman_cost(qpe: QPECost, eps: float, *, d3: float, p: float = 1.0, start_toffolis: float = 0.0,
                          d5: float | None = None, stencil: str = "central",
                          projections: Mapping[str, float] | None = None, n_verify: int = 1,
                          fixed_bias: float = 0.0, stencil_bias=None) -> dict:
    """F51: the cheapest Hellmann-Feynman QPE estimate of a scalar to RMS ``eps`` (module docstring).

    Args:
        qpe: the QPE of H' (C_QPE, dE).
        eps: the accuracy (property units).
        d3, d5: E''' and E^(5) of E_0'(h) (property and energy units: Hartree per unit^n).
        p, start_toffolis: the start's overlap and C(Psi_s).
        stencil: "central" (2 energies) or "five_point" (4 energies).
        projections: per-preparation Toffolis of the pre-projection schemes, keys "c_filt" (lt_bbht),
            "c_coarse" (2s), "c_filt_l" (lt_bbht_L), "c_coarse_l" (2s_L); None: repetition only.
        n_verify: n_v of rule B for the repeated long QPE (F40 on its grid).
        fixed_bias: b_m >= 0, a systematic bias outside the stencil (the model bias of H'), added linearly:
            the stencil bias b maximizes h(b) sigma_d(b) under (b_m + b)^2 + sigma_d^2 = eps^2 (module
            docstring); the estimate is infeasible (T = inf) iff b_m >= eps (or sigma_d^2 <= 1e-12 eps^2,
            F10's rule).
        stencil_bias: None, or a callable (h, stencil) -> the signed stencil bias D(h) - dE_0/dh(0) of the
            finite-field energies the derivatives describe (:class:`FiniteFieldStencil`): the leading-order h
            is then shrunk by bisection until |stencil_bias(h)| <= b (module docstring); if it still exceeds
            b at h_leading/2^12, the stencil has no admissible field step and the estimate is infeasible.

    Returns:
        {"stencil", "h", "bias" (b_m + b, the bias of the estimate), "stencil_bias" (b, the allocation),
        "fixed_bias" (b_m), "h_leading" (F51's leading-order h), "stencil_bias_checked" (the signed
        finite-field stencil bias at h, at h_leading/2^12 without an admissible step, None without
        ``stencil_bias``), "bias_evaluations", "no_field_step" (no admissible h; h is nan), "sigma_e",
        "n_energies", "qpe_toffolis" (one long QPE run, L), "schemes" {scheme: Toffolis per energy}, "scheme"
        (the cheapest), "toffolis" (T), "toffolis_rep" (T under repetition), "c_qpe_units" (T / C_QPE),
        "feasible"}.
    """
    if not isinstance(qpe, QPECost):
        raise ValueError(f"qpe must be a QPECost, got {type(qpe).__name__}")
    eps = _positive(eps, "eps")
    p = _overlap(p)
    c_s = _nonnegative(start_toffolis, "start_toffolis")
    nv = _int_at_least(n_verify, 1, "n_verify")
    b_m = _nonnegative(fixed_bias, "fixed_bias")
    if stencil == "central":
        k = abs(_finite(d3, "d3"))
        if not k > 0.0:
            raise ValueError("the central stencil needs a non-zero E'''")
        n_e = 2
    elif stencil == "five_point":
        if d5 is None:
            raise ValueError("the five-point stencil needs d5 = E^(5)")
        k = abs(_finite(d5, "d5"))
        if not k > 0.0:
            raise ValueError("the five-point stencil needs a non-zero E^(5)")
        n_e = 4
    else:
        raise ValueError(f"stencil must be 'central' or 'five_point', got {stencil!r}")
    proj = {} if projections is None else dict(projections)
    unknown = set(proj) - {"c_filt", "c_coarse", "c_filt_l", "c_coarse_l"}
    if unknown:
        raise ValueError(f"unknown projection keys {sorted(unknown)}")
    out = {"stencil": stencil, "fixed_bias": b_m, "n_energies": n_e, "stencil_bias_checked": None,
           "bias_evaluations": 0, "no_field_step": False}
    infeasible = dict(sigma_e=0.0, qpe_toffolis=math.inf, schemes={"rep": math.inf}, scheme="rep",
                      toffolis=math.inf, toffolis_rep=math.inf, c_qpe_units=math.inf, feasible=False)
    if b_m >= eps:
        return dict(out, h=math.nan, h_leading=math.nan, bias=b_m, stencil_bias=math.nan, **infeasible)
    b = _stencil_split(stencil, eps, b_m)
    h0 = _stencil_step(stencil, b, k)
    h = h0
    if stencil_bias is not None:
        h, actual, n_eval = _refine_step(h0, b, stencil, stencil_bias)
        out.update(stencil_bias_checked=actual, bias_evaluations=n_eval)
        if math.isnan(h):
            out["no_field_step"] = True
            return dict(out, h=h, h_leading=h0, bias=b_m + b, stencil_bias=b, **infeasible)
    total_bias = b_m + b
    var = eps * eps - total_bias * total_bias
    if not var > _BUDGET_RTOL * eps * eps:
        return dict(out, h=h, h_leading=h0, bias=total_bias, stencil_bias=b, **infeasible)
    sigma_e = math.sqrt(var) / _stencil_noise(stencil, h)
    length = float(qpe.toffolis) * qpe.de_qpe / sigma_e
    schemes = {"rep": (c_s + length) / p + (nv - 1) * length}
    n_bbht = exponential_search_calls(p)
    for key, scheme in (("c_filt", "lt_bbht"), ("c_filt_l", "lt_bbht_L")):
        if proj.get(key) is not None:
            schemes[scheme] = n_bbht * (c_s + _nonnegative(proj[key], key)) + length
    for key, scheme in (("c_coarse", "2s"), ("c_coarse_l", "2s_L")):
        if proj.get(key) is not None:
            schemes[scheme] = (c_s + _nonnegative(proj[key], key)) / p + length
    best = min(schemes, key=lambda s: (schemes[s], list(schemes).index(s)))
    total = n_e * schemes[best]
    return dict(out, h=h, h_leading=h0, bias=total_bias, stencil_bias=b, sigma_e=sigma_e, qpe_toffolis=length,
                schemes=schemes, scheme=best, toffolis=total, toffolis_rep=n_e * schemes["rep"],
                c_qpe_units=total / float(qpe.toffolis), feasible=True)


def _check_one_body_scalar(target: PropertyTarget, active: ActiveSpace, name: str) -> None:
    if target.norm != "scalar" or target.two_body[0] is not None or target.one_body[0] is None:
        raise ValueError(f"{name} needs a one-body scalar target (e.g. the dipole)")
    target._check_active(active)


def _field_ground(active: ActiveSpace, target: PropertyTarget, h: float, conv_tol: float):
    """(E_0(h), ci) of H + h O by FCI: h1 + h f and ecore + h constant (the one-body scalar O of ``target``)."""
    import dataclasses

    a = dataclasses.replace(active, h1=active.h1 + h * target.one_body[0], ecore=active.ecore + h * target.constant)
    e, ci = a.fci(nroots=1, conv_tol=conv_tol)
    return float(np.atleast_1d(e)[0]), ci


def finite_field_derivatives(active: ActiveSpace, target: PropertyTarget, *, h_max: float = 0.02,
                             n_points: int = 41, degree: int = 8, conv_tol: float = 1e-12) -> dict:
    """E_0(h) of H + h O by FCI on a symmetric field grid and its derivatives from a polynomial fit (F51).

    ``target`` must be a one-body scalar (the dipole): O = constant + sum f[p,q] E_pq, so H + h O has
    h1 + h f and ecore + h constant. Returns {"fields", "energies", "coefficients", "d1" .. "d5"} with
    d_n = n! c_n (d1 = <O> by Hellmann-Feynman).
    """
    _check_one_body_scalar(target, active, "finite_field_derivatives")
    h_max = _positive(h_max, "h_max")
    degree = _int_at_least(degree, 5, "degree")
    n_points = _int_at_least(n_points, degree + 2, "n_points")
    fields = np.linspace(-h_max, h_max, n_points)
    energies = np.empty(n_points)
    for i, h in enumerate(fields):
        energies[i] = _field_ground(active, target, h, conv_tol)[0]
    coef = np.polynomial.polynomial.polyfit(fields, energies, degree)
    out = {"fields": fields, "energies": energies, "coefficients": coef}
    for k in range(1, 6):
        out[f"d{k}"] = float(math.factorial(k) * coef[k])
    return out


class FiniteFieldStencil:
    """The stencil biases of F51 on the FCI energies of H + h O (the path of :func:`finite_field_derivatives`).

    ``bias(h, stencil)`` = D(h) - <O>_0 with D the central ((E(h) - E(-h))/(2h)) or five-point
    ((-E(2h) + 8E(h) - 8E(-h) + E(-2h))/(12h)) difference and <O>_0 the expectation in the FCI ground state at
    h = 0 (dE_0/dh(0) by Hellmann-Feynman); energies are computed once per field. An instance is the
    ``stencil_bias`` callable of :func:`hellmann_feynman_cost`.
    """

    def __init__(self, active: ActiveSpace, target: PropertyTarget, *, conv_tol: float = 1e-12) -> None:
        _check_one_body_scalar(target, active, "FiniteFieldStencil")
        self.active, self.target = active, target
        self.conv_tol = _positive(conv_tol, "conv_tol")
        e0, ci0 = _field_ground(active, target, 0.0, self.conv_tol)
        self._energies = {0.0: e0}
        self.expectation = float(property_values(target, active, ci0)[0])

    def energy(self, h: float) -> float:
        """E_0(h) (cached per field)."""
        h = _finite(h, "h")
        if h not in self._energies:
            self._energies[h] = _field_ground(self.active, self.target, h, self.conv_tol)[0]
        return self._energies[h]

    @property
    def n_solves(self) -> int:
        """FCI solves so far (h = 0 included)."""
        return len(self._energies)

    def derivative(self, h: float, stencil: str) -> float:
        """The stencil's estimate D(h) of dE_0/dh at 0."""
        h = _positive(h, "h")
        if stencil == "central":
            return (self.energy(h) - self.energy(-h)) / (2.0 * h)
        if stencil == "five_point":
            return (-self.energy(2.0 * h) + 8.0 * self.energy(h) - 8.0 * self.energy(-h)
                    + self.energy(-2.0 * h)) / (12.0 * h)
        raise ValueError(f"stencil must be 'central' or 'five_point', got {stencil!r}")

    def bias(self, h: float, stencil: str) -> float:
        """D(h) - <O>_0, signed."""
        return self.derivative(h, stencil) - self.expectation

    __call__ = bias


def model_property_errors(active: ActiveSpace, encodings: Sequence[BlockEncoding], target: PropertyTarget,
                          ci_ref, *, reference=None) -> np.ndarray:
    """b_fact(P) of every encoding: the target's error of psi_0' (the ground state of H') against ``ci_ref`` (F52).

    psi_0' from ``diagnostics.thc_floors.model_floors(..., return_state=True)`` (``reference``: an
    ``ExactReference`` of ``active``, built if None).
    """
    from ..diagnostics.thc_floors import ExactReference, model_floors

    ref = ExactReference.build(active) if reference is None else reference
    out = []
    for enc in encodings:
        if enc.eri_model is None:
            raise ValueError("every encoding needs its model integrals")
        _, psi_m = model_floors(ref, enc.eri_model, return_state=True)
        out.append(property_error(target, active, psi_m, ci_ref))
    return np.array(out)


def select_for_property(encodings: Sequence[BlockEncoding], model_errors: Sequence[float], eps: float, *,
                        params: FTParameters = FTParameters()) -> QPECost:
    """F52: the cheapest QPE with |de_fact| <= params.de_fact and b_fact(P) <= eps/2, else with b_fact(P) <= eps/2.

    ``model_errors[i]`` is the exact b_fact(P) of ``encodings[i]`` (:func:`model_property_errors`).
    Raises ValueError if no scanned encoding has b_fact(P) <= eps/2.
    """
    encs = list(encodings)
    errs = [float(x) for x in model_errors]
    if len(encs) != len(errs) or not encs:
        raise ValueError("need one model error per encoding (and at least one encoding)")
    eps = _positive(eps, "eps")
    ok = [e for e, b in zip(encs, errs) if b <= 0.5 * eps]
    first = [e for e in ok if abs(e.de_fact) <= params.de_fact]
    pool = first or ok
    if not pool:
        raise ValueError(f"no scanned encoding has b_fact(P) <= eps/2 = {0.5 * eps:g}")
    costs = [qpe_cost(e, params.de_qpe) for e in pool]
    return min(costs, key=lambda q: (q.toffolis, q.logical_qubits))


# ------------------------------------------------------------------ rows, routes A and B (F49, F50, F53)
@dataclass(frozen=True)
class PropertyRow:
    """One way to measure a property to eps (F53).

    Attributes:
        system, target: labels (``target`` e.g. "rdm1").
        eps: the accuracy.
        route: one of :data:`PROPERTY_ROUTES`.
        estimator: "shadow", "ae", "ge<kappa_g>" (the gradient-estimation band) or "qpe" (HF-QPE).
        loop: "free" or the label of a measured loop.
        k: the iterate (route A) or the start iterate (routes B, HF-QPE).
        bias: the bias that sets the statistical budget (target norm).
        toffolis: the total, loop included (inf: infeasible).
        loop_toffolis: C_loop(k) (plus certificate and one-time estimate costs for route B).
        per_prep: Toffolis per preparation; n_prep: preparations (HF-QPE: per_prep is the Toffolis of one
            energy, n_prep the energies); longest_run: Toffolis of the longest coherent run (HF-QPE: one
            long QPE on a prepared start); logical_qubits: of the largest run.
        flags: "not_usable" (route A: bias >= eps, or a budget beyond the record in a measured loop or past
            a record that did not converge), "oracle_k", "record_ended", "termination_not_certified",
            "not_terminating", "infeasible", "no_field_step:<stencil>" (HF-QPE: no admissible field step),
            "no_layout" (no surface-code layout of the longest coherent run at its failure budget,
            :func:`run_layout`; reported, the row stays selectable: F53 compares Toffolis), ...
    """

    system: str
    target: str
    eps: float
    route: str
    estimator: str
    loop: str
    k: int
    bias: float
    toffolis: float
    loop_toffolis: float
    per_prep: float
    n_prep: float
    longest_run: float
    logical_qubits: int
    flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.route not in PROPERTY_ROUTES:
            raise ValueError(f"route must be one of {PROPERTY_ROUTES}, got {self.route!r}")
        if not (self.estimator in ("shadow", "ae", "qpe") or self.estimator.startswith("ge")):
            raise ValueError(f"estimator must be 'shadow', 'ae', 'qpe' or 'ge<kappa_g>', got {self.estimator!r}")
        if not isinstance(self.loop, str) or not self.loop:
            raise ValueError("loop must be a non-empty label")
        object.__setattr__(self, "eps", _positive(self.eps, "eps"))
        object.__setattr__(self, "k", _int_at_least(self.k, 0, "k"))
        b = float(self.bias)
        if math.isnan(b) or b < 0.0:
            raise ValueError(f"bias must be non-negative (inf allowed), got {self.bias!r}")
        for name in ("toffolis", "loop_toffolis", "n_prep", "longest_run"):
            object.__setattr__(self, name, _cost(getattr(self, name), name))
        pp = float(self.per_prep)
        if not (math.isnan(pp) or pp >= 0.0):
            raise ValueError(f"per_prep must be non-negative or nan, got {self.per_prep!r}")
        _int_at_least(self.logical_qubits, 0, "logical_qubits")
        object.__setattr__(self, "flags", tuple(self.flags))

    @property
    def selectable(self) -> bool:
        """A finite cost, not the oracle row, usable and terminating (F53)."""
        return (math.isfinite(self.toffolis) and self.route != "A-oracle"
                and not {"not_usable", "not_terminating", "infeasible", "oracle_k"} & set(self.flags))


def _prep_arrays(prep) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(prep, PrepCost):
        return np.asarray(prep.cumulative, dtype=float), np.asarray(prep.logical_qubits, dtype=np.int64)
    cum, qub = prep
    return np.asarray(cum, dtype=float), np.asarray(qub, dtype=np.int64)


def _loop_costs(residual_costs: Mapping[str, Sequence[float]], loops: Sequence[str], n_states: int) -> dict:
    out = {}
    for lp in loops:
        if lp == "free":
            out[lp] = np.zeros(n_states)
            continue
        if lp not in residual_costs:
            raise ValueError(f"no residual costs for the loop {lp!r}")
        t = np.asarray(residual_costs[lp], dtype=float)
        if t.ndim != 1 or t.size < n_states - 1 or np.any(np.isnan(t)) or np.any(t < 0):
            raise ValueError(f"residual_costs[{lp!r}] must hold T_res(j) >= 0 for j = 0..{n_states - 2}")
        out[lp] = np.concatenate([[0.0], np.cumsum(t[:n_states - 1])])
    return out


def _estimators(params: FTParameters) -> list[str]:
    return ["shadow", "ae"] + [f"ge{g:g}" for g in params.kappa_g]


def _measure(est: str, inputs: PropertyInputs, eps: float, bias: float, prep: PrepParams,
             params: FTParameters) -> RouteCost:
    if est == "shadow":
        return shadow_cost(inputs, eps, bias, prep, params=params)
    if est == "ae":
        return ae_cost(inputs, eps, bias, prep, kappa=params.kappa, params=params)
    return gradient_cost(inputs, eps, bias, prep, kappa_g=float(est[2:]), params=params)


def _row(system: str, inputs: PropertyInputs, eps: float, route: str, est: str, loop: str, k: int, bias: float,
         extra: float, cost: RouteCost, flags: Sequence[str], params: FTParameters) -> PropertyRow:
    total = extra + cost.toffolis
    fl = tuple(flags) + tuple(f for f in cost.flags if f == "infeasible")
    return _layout_flag(PropertyRow(system=system, target=inputs.target.kind, eps=eps, route=route, estimator=est,
                                    loop=loop, k=k, bias=bias, toffolis=total, loop_toffolis=extra,
                                    per_prep=cost.toffolis_per_prep, n_prep=cost.n_prep,
                                    longest_run=cost.coherent_toffolis, logical_qubits=cost.logical_qubits, flags=fl),
                        params)


def _layout_flag(row: PropertyRow, params: FTParameters) -> PropertyRow:
    """The row with the flag "no_layout" where its longest coherent run has no layout (:func:`run_layout`)."""
    if math.isfinite(row.toffolis) and math.isfinite(row.longest_run) and row.logical_qubits >= 1 \
            and run_layout(row, params=params) is None:
        return dataclasses.replace(row, flags=tuple(row.flags) + ("no_layout",))
    return row


def route_a_rows(record: TrajectoryRecord, prep, residual_costs: Mapping[str, Sequence[float]],
                 errors: Sequence[float], inputs: PropertyInputs, eps: float, *,
                 loops: Sequence[str] = ("free", "meas_r1", "meas_r03"),
                 budgets: Sequence[int] = (2, 5, 10, 20, 60), system: str | None = None,
                 params: FTParameters = FTParameters()) -> list[PropertyRow]:
    """Route A at the fixed budgets and the A-oracle row, per loop and estimator (F49, module docstring).

    Args:
        record: the trajectory (Psi_0 .. Psi_K).
        prep: its ``stateprep.PrepCost`` (or a pair of arrays C(Psi_k), Q(Psi_k)), k = 0..K.
        residual_costs: per measured loop label, T_res(j) for j = 0..K-1 (Toffolis).
        errors: ||b_A(k)|| for k = 0..K, the exact property errors in the target's norm.
        inputs: the target's estimator inputs; eps: the accuracy.
        loops: "free" and labels of ``residual_costs``; budgets: the fixed k.
    """
    eps = _positive(eps, "eps")
    inputs = _check_inputs(inputs)
    n_states = record.n_steps + 1
    err = np.asarray(errors, dtype=float)
    if err.shape != (n_states,) or np.any(np.isnan(err)) or np.any(err < 0):
        raise ValueError(f"errors must hold ||b_A(k)|| >= 0 for k = 0..{n_states - 1}")
    cum, qub = _prep_arrays(prep)
    if cum.size < n_states or qub.size < n_states:
        raise ValueError(f"prep must cover k = 0..{n_states - 1}")
    budgets = [_int_at_least(b, 0, "budget") for b in budgets]
    sysname = record.system if system is None else system
    cl = _loop_costs(residual_costs, loops, n_states)
    rows: list[PropertyRow] = []
    for est in _estimators(params):
        meas = [_measure(est, inputs, eps, float(err[k]) if err[k] < eps else math.inf,
                         PrepParams(cum[k], int(qub[k])), params) for k in range(n_states)]
        for lp in loops:
            for b in budgets:
                kk = min(b, n_states - 1)
                flags = [] if err[kk] < eps else ["not_usable"]
                if b > n_states - 1:
                    flags.append("record_ended")
                    # C(Psi_K) is the cost of the free loop only if the record ended by converging; a measured
                    # loop's C_loop(K) only bounds the pre-declared loop's cost from below
                    if (lp != "free" or not record.converged) and "not_usable" not in flags:
                        flags.append("not_usable")
                rows.append(_row(sysname, inputs, eps, "A", est, lp, b, float(err[kk]), float(cl[lp][kk]),
                                 meas[kk], flags, params))
            totals = np.array([cl[lp][k] + meas[k].toffolis for k in range(n_states)])
            if np.any(np.isfinite(totals)):
                kb = int(np.argmin(totals))
                rows.append(_row(sysname, inputs, eps, "A-oracle", est, lp, kb, float(err[kb]),
                                 float(cl[lp][kb]), meas[kb], ["oracle_k"], params))
            else:
                kb = int(np.argmin(err))
                rows.append(_row(sysname, inputs, eps, "A-oracle", est, lp, kb, float(err[kb]),
                                 float(cl[lp][kb]), meas[kb], ["oracle_k", "not_usable"], params))
    return rows


def _walk_sin(enc: BlockEncoding, c: float, energy: float) -> float:
    x = (energy - c) / enc.lam
    return math.sqrt(max(0.0, 1.0 - min(1.0, x * x)))


def route_b_rows(record: TrajectoryRecord, prep, residual_costs: Mapping[str, Sequence[float]],
                 inputs: PropertyInputs, eps: float, enc: BlockEncoding, c: float, *,
                 u_classical: float | None, e1_lower: float, starts: str = "all",
                 loops: Sequence[str] | None = None, system: str | None = None, e0_model: float | None = None,
                 model_bias: float | None = None, certificates: Sequence | None = None,
                 hf_derivatives: Mapping[str, float] | None = None, qpe: QPECost | None = None,
                 share: float = 0.1, params: FTParameters = FTParameters(),
                 hf_stencil_bias=None) -> list[PropertyRow]:
    """Route B (B0 and Bk) per estimator and loop, and Hellmann-Feynman QPE for scalars (F50, F51, F53).

    For every start Psi_k (k = 0, 1 for B0; every k for Bk with ``starts="all"``) and every pass edge
    (``u_classical``, the start's energy certificate, E1lb - W_L) the copy of :func:`ground_copy`; per
    estimator the cheapest start and edge (and coherent option) with C_loop(k) and the certificate cost.

    Args:
        record, prep, residual_costs: as :func:`route_a_rows` (overlaps p_k on H stand for those on H').
        inputs, eps: the target's estimator inputs (with ``r0``) and the accuracy.
        enc, c: the encoding of H' and its walk offset.
        u_classical: CISD(H') (or None); e1_lower: E1lb.
        starts: "all" (B0 and Bk) or "b0".
        loops: default "free" and every label of ``residual_costs``.
        e0_model: E_0' (default record.e0 + enc.de_fact) for the termination of the E1lb edge.
        model_bias: the exact model bias of H' for the target (default F47 with q = 1 - F).
        certificates: ``ft.certify.EnergyCertificate`` objects of the record (sampled or AE energies,
            used at their iterate j); None: the AE certificates of every iterate
            (``ft.certify.energy_certificates`` with the walk of ``enc``).
        hf_derivatives: {"d3": E''', optionally "d5"} for Hellmann-Feynman rows (scalar targets).
        qpe: the QPE of H' (default ``qpe_cost(enc, params.de_qpe)``).
        hf_stencil_bias: the ``stencil_bias`` callable of :func:`hellmann_feynman_cost` (a
            :class:`FiniteFieldStencil`) that refines the field step of the HF-QPE rows; None: F51's
            leading-order step.
    """
    from .acceptance import acceptance_bracket
    from .certify import energy_certificates

    eps = _positive(eps, "eps")
    inputs = _check_inputs(inputs)
    if starts not in ("all", "b0"):
        raise ValueError(f"starts must be 'all' or 'b0', got {starts!r}")
    e1 = _finite(e1_lower, "e1_lower")
    c = _finite(c, "c")
    n_states = record.n_steps + 1
    cum, qub = _prep_arrays(prep)
    if cum.size < n_states or qub.size < n_states:
        raise ValueError(f"prep must cover k = 0..{n_states - 1}")
    loops = ("free",) + tuple(residual_costs) if loops is None else tuple(loops)
    cl = _loop_costs(residual_costs, loops, n_states)
    sysname = record.system if system is None else system
    e0m = float(record.e0) + float(enc.de_fact) if e0_model is None else _finite(e0_model, "e0_model")
    qpe = qpe_cost(enc, params.de_qpe) if qpe is None else qpe
    norb = int(record.norb)
    sigma = inputs.sigma_norm
    r_o = inputs.range_norm
    ks = range(n_states) if starts == "all" else range(min(2, n_states))
    u_cl = None if u_classical is None else _finite(u_classical, "u_classical")
    if certificates is None:
        certificates = energy_certificates(record, cum[:n_states], e1_lower=e1, de=params.de_qpe, walk=enc,
                                           prep_qubits=qub[:n_states], e0_model=e0m, kappa=params.kappa)
    cert_by_j: dict[int, list] = {}
    for cert in certificates:
        if cert.source in ("sampled", "ae") and cert.j is not None:
            cert_by_j.setdefault(int(cert.j), []).append(cert)
    # copies per start: (k, edge label, extra one-time cost, GroundCopy)
    copies = []
    for k in ks:
        p = float(record.overlaps[k])
        if not p > 0.0:
            continue
        common = dict(p=min(p, 1.0), start_toffolis=float(cum[k]), start_qubits=int(qub[k]), e1_lower=e1,
                      eps=eps, sigma0=sigma, share=share, params=params, start=f"Psi_{k}", norb=norb, r_o=r_o,
                      e0_model=e0m, model_bias=model_bias, kappa=params.kappa)
        edges = []
        if u_cl is not None and u_cl < e1:
            edges.append(("U_classical", u_cl, 0.0))
        for cert in cert_by_j.get(k, []):
            if cert.upper < e1:
                edges.append((f"U_{cert.source}", float(cert.upper), float(cert.toffolis)))
        edges.append(("E1lb", None, 0.0))
        for label, u, extra in edges:
            gc = ground_copy(enc, c, u_upper=u, **common)
            if gc.terminating:
                copies.append((k, label, extra, gc))
    rows: list[PropertyRow] = []
    estimators = _estimators(params)
    for route in (("B0", "Bk") if starts == "all" else ("B0",)):
        allowed = [x for x in copies if route == "Bk" or x[0] <= 1]
        for est in estimators:
            # measurement cost per copy option (independent of the loop)
            options = []
            for k, label, extra, gc in allowed:
                if est == "shadow":
                    bias = gc.model_bias + gc.leak_bound
                    cost = _measure(est, inputs, eps, bias, PrepParams(gc.copy_toffolis, gc.logical_qubits),
                                    params)
                    options.append((k, label, extra, gc, bias, cost, "destructive"))
                    continue
                bias = gc.model_bias + gc.leak_bound
                cost = _measure(est, inputs, eps, bias, PrepParams(gc.ratio_toffolis, gc.logical_qubits),
                                params)
                options.append((k, label, extra, gc, bias, cost, "coherent:ratio"))
                bias_x = gc.model_bias + gc.exact_aa_leak_bound
                cost_x = _measure(est, inputs, eps, bias_x, PrepParams(gc.exact_aa_toffolis, gc.logical_qubits),
                                  params)
                options.append((k, label, extra + gc.exact_aa_estimate_toffolis, gc, bias_x, cost_x,
                                "coherent:exact_aa"))
            for lp in loops:
                best = None
                for k, label, extra, gc, bias, cost, mode in options:
                    total = float(cl[lp][k]) + extra + cost.toffolis
                    if best is None or total < best[0]:
                        best = (total, k, label, extra, gc, bias, cost, mode)
                if best is None or not math.isfinite(best[0]):
                    k0 = 0
                    rows.append(PropertyRow(system=sysname, target=inputs.target.kind, eps=eps, route=route,
                                            estimator=est, loop=lp, k=k0, bias=math.inf, toffolis=math.inf,
                                            loop_toffolis=float(cl[lp][k0]), per_prep=math.nan, n_prep=math.inf,
                                            longest_run=math.inf, logical_qubits=0,
                                            flags=("not_usable",) if best is None else ("not_usable", "infeasible")))
                    continue
                total, k, label, extra, gc, bias, cost, mode = best
                flags = [f for f in gc.flags if f != "termination_not_certified" or label == "E1lb"]
                flags += [f"edge:{label}", mode]
                rows.append(_row(sysname, inputs, eps, route, est, lp, k, bias, float(cl[lp][k]) + extra, cost,
                                 flags, params))
    if hf_derivatives is not None and inputs.target.norm == "scalar":
        rows += _hf_rows(record, cum, qub, cl, loops, inputs, eps, enc, c, qpe, u_cl, e1, e0m, ks,
                         hf_derivatives, model_bias, sysname, acceptance_bracket, params, hf_stencil_bias)
    return rows


def _hf_rows(record, cum, qub, cl, loops, inputs, eps, enc, c, qpe, u_cl, e1, e0m, ks, derivs, model_bias,
             sysname, acceptance_bracket, params, stencil_bias=None) -> list[PropertyRow]:
    """HF-QPE rows per loop: the cheapest start with the start-independent pre-projections (F51).

    The model bias b_m of H' is a fixed systematic bias of the estimate (``hellmann_feynman_cost(...,
    fixed_bias=b_m)``): the rows carry bias = b_m + b and meet (b_m + b)^2 + sigma_d^2 = eps^2. A stencil
    without an admissible field step (``stencil_bias``) is dropped, and every row carries the flag
    "no_field_step:<stencil>"; the rows are infeasible if no stencil is left.
    """
    d3 = derivs.get("d3")
    if d3 is None:
        raise ValueError("hf_derivatives needs 'd3'")
    mb = _f47(inputs.sigma_norm, inputs.range_norm, float(enc.infidelity)) if model_bias is None else float(model_bias)
    flags_mb = ("model_bias_bound",) if model_bias is None else ()

    def infeasible(extra: tuple[str, ...] = ()) -> list[PropertyRow]:
        return [PropertyRow(system=sysname, target=inputs.target.kind, eps=eps, route="HF-QPE", estimator="qpe",
                            loop=lp, k=0, bias=mb, toffolis=math.inf, loop_toffolis=0.0, per_prep=math.nan,
                            n_prep=math.inf, longest_run=math.inf, logical_qubits=0,
                            flags=("not_usable", "infeasible") + flags_mb + extra) for lp in loops]

    if mb >= eps:
        return infeasible()
    out = []
    projections: dict[str, float] = {}
    if u_cl is not None and u_cl < e1:
        projections["c_filt"] = filter_cost(enc, c, u_cl, e1, eps=1e-2, params=params).toffolis
        projections["c_coarse"] = coarse_qpe_toffolis(enc, c, u_cl, e1, eps_p=_EPS_P)
    if e0m <= e1 - W_L:
        projections["c_filt_l"] = filter_cost(enc, c, e1 - W_L, e1, eps=1e-2, params=params).toffolis
        projections["c_coarse_l"] = coarse_qpe_toffolis(enc, c, e1 - W_L, e1, eps_p=_EPS_P)
    u_cert = u_cl if (u_cl is not None and u_cl < e1) else None
    stencils = ["central"] + (["five_point"] if derivs.get("d5") is not None else [])
    common = dict(d3=d3, d5=derivs.get("d5"), fixed_bias=mb, stencil_bias=stencil_bias)
    n_verify = {}
    no_step: list[str] = []
    for st in stencils:
        probe = hellmann_feynman_cost(qpe, eps, stencil=st, p=1.0, **common)
        if not probe["feasible"]:
            if probe["no_field_step"]:
                no_step.append(f"no_field_step:{st}")
            continue
        grid = 2.0 * probe["sigma_e"] * _walk_sin(enc, c, e0m)       # the long QPE's grid (F39, A3)
        n_verify[st] = acceptance_bracket(u_cert, e1, probe["sigma_e"], grid=grid).n_runs if grid > 0 else 1
    stencils = [st for st in stencils if st in n_verify]
    if not stencils:                    # no admissible field step, or sigma_d^2 <= 1e-12 eps^2 (F10's rule)
        return infeasible(tuple(no_step))
    flags = list(flags_mb) + no_step
    for lp in loops:
        best = None
        for k in ks:
            p = float(record.overlaps[k])
            if not p > 0.0:
                continue
            for st in stencils:
                res = hellmann_feynman_cost(qpe, eps, stencil=st, p=min(p, 1.0), start_toffolis=float(cum[k]),
                                            projections=projections, n_verify=n_verify[st], **common)
                total = float(cl[lp][k]) + res["toffolis"]
                if best is None or total < best[0]:
                    best = (total, k, res)
        if best is None:
            continue
        total, k, res = best
        n_v = n_verify[res["stencil"]]
        fl = flags + [f"scheme:{res['scheme']}", f"stencil:{res['stencil']}", f"n_verify:{n_v}"]
        if res["stencil_bias_checked"] is not None:
            fl.append("stencil_bias_checked")
        if (res["scheme"] == "rep" and u_cert is None) or res["scheme"] in ("lt_bbht_L", "2s_L"):
            fl.append("termination_not_certified")
        out.append(_layout_flag(PropertyRow(
            system=sysname, target=inputs.target.kind, eps=eps, route="HF-QPE", estimator="qpe", loop=lp, k=k,
            bias=res["bias"], toffolis=total, loop_toffolis=float(cl[lp][k]), per_prep=res["schemes"][res["scheme"]],
            n_prep=float(res["n_energies"]), longest_run=res["qpe_toffolis"] + float(cum[k]),
            logical_qubits=int(qpe.logical_qubits), flags=tuple(fl)), params))
    return out


def best_property_row(rows: Sequence[PropertyRow], *, loop: str, estimator: str = "ae") -> PropertyRow:
    """F53: the cheapest selectable row of the loop and estimator (HF-QPE rows compete with every estimator).

    Never the A-oracle row, an unusable or non-terminating row, or an infeasible one; ties go to the
    first row. Raises ValueError if none is left.
    """
    pool = [r for r in rows if r.loop == loop and (r.estimator == estimator or r.route == "HF-QPE") and r.selectable]
    if not pool:
        raise ValueError(f"no selectable row for loop {loop!r} and estimator {estimator!r}")
    return min(pool, key=lambda r: r.toffolis)


def run_layout(row: PropertyRow, *, params: FTParameters = FTParameters(),
               model: SurfaceCodeModel = SurfaceCodeModel()) -> PhysicalCost | None:
    """The surface-code layout of the row's longest coherent run (F53; ``ft.physical.surface_code_cost``, F18).

    The run's Toffolis (``longest_run``) on the row's logical qubits: a shadow snapshot at p_fail_shot with its
    2d-cycle readout (F19), every other run (AE per element, gradient estimation, an HF-QPE energy) at p_fail_ae,
    the budget of a single coherent run, which does not parallelize over machines (F19). None where no factory
    and code distance keep the run within its budget, or where the row has no finite run.
    """
    run, q = float(row.longest_run), int(row.logical_qubits)
    if not (math.isfinite(run) and q >= 1):
        return None
    shot = row.estimator == "shadow"
    try:
        return surface_code_cost(q, max(run, 1.0), fail_budget=params.p_fail_shot if shot else params.p_fail_ae,
                                 model=model, shot=shot)
    except ValueError:
        return None


def model_bias_bound(inputs: PropertyInputs, infidelity: float) -> float:
    """F47 with q = 1 - F (the encoding's infidelity) in the target's norm, as :func:`ground_copy` and the HF-QPE
    rows charge it without the exact model bias: 2 sigma_0 (q (1 - q))^(1/2) + q r_O (first order without r0)."""
    inputs = _check_inputs(inputs)
    q = _finite(infidelity, "infidelity")
    if not -_ROUND_OFF <= q <= 1.0 + _ROUND_OFF:
        raise ValueError(f"infidelity must lie in [0, 1], got {infidelity!r}")
    return _f47(inputs.sigma_norm, inputs.range_norm, min(max(q, 0.0), 1.0))


# ------------------------------------------------------------------ comparisons over (system, target) pairs (F53)
def _json_number(x) -> float:
    """A strict-JSON number (``ft.scaling.to_json``) back to a float: null -> nan, "inf"/"-inf" -> +-inf."""
    if x is None:
        return math.nan
    if isinstance(x, str):
        return {"inf": math.inf, "-inf": -math.inf}[x]
    return float(x)


def row_from_json(row: Mapping) -> PropertyRow:
    """A :class:`PropertyRow` from its strict-JSON form (``ft.scaling.to_json``; extra keys are ignored)."""
    return PropertyRow(system=str(row["system"]), target=str(row["target"]), eps=_json_number(row["eps"]),
                       route=str(row["route"]), estimator=str(row["estimator"]), loop=str(row["loop"]),
                       k=int(row["k"]), bias=_json_number(row["bias"]), toffolis=_json_number(row["toffolis"]),
                       loop_toffolis=_json_number(row["loop_toffolis"]), per_prep=_json_number(row["per_prep"]),
                       n_prep=_json_number(row["n_prep"]), longest_run=_json_number(row["longest_run"]),
                       logical_qubits=int(row["logical_qubits"]), flags=tuple(row["flags"]))


def best_at_budget(rows: Sequence[PropertyRow], *, loop: str, estimator: str = "ae",
                   budget: int | None = None) -> PropertyRow | None:
    """F53 with route A at one pre-declared budget: :func:`best_property_row` over ``rows`` with the route-A rows
    restricted to k = ``budget`` (None: every budget; routes B and HF-QPE unrestricted); None if nothing is
    selectable."""
    pool = [r for r in rows if budget is None or r.route != "A" or r.k == budget]
    try:
        return best_property_row(pool, loop=loop, estimator=estimator)
    except ValueError:
        return None


def route_tally(pairs: Iterable[Sequence[PropertyRow]], *, loops: Sequence[str] = ("free", "meas_r1"),
                budgets: Sequence[int] = (2, 5, 10, 20, 60), estimator: str = "ae") -> dict:
    """The routes of the best rows over (system, target) pairs (each a row list), per loop (F53).

    Returns {"pairs": n, "best": {loop: {route: count}}, "fixed_k": {loop: {k: {route: count}}}}: "best"
    with route A at its cheapest usable budget per pair, "fixed_k" with route A at the single pre-declared
    budget k for every pair (route A's usability is an emulation fact, F49); route "none" where no row is
    selectable.
    """
    budgets = [_int_at_least(b, 0, "budget") for b in budgets]
    out = {"pairs": 0, "best": {lp: {} for lp in loops}, "fixed_k": {lp: {b: {} for b in budgets} for lp in loops}}

    def count(d: dict, row: PropertyRow | None) -> None:
        key = "none" if row is None else row.route
        d[key] = d.get(key, 0) + 1

    for rows in pairs:
        rows = list(rows)
        out["pairs"] += 1
        for lp in loops:
            count(out["best"][lp], best_at_budget(rows, loop=lp, estimator=estimator))
            for b in budgets:
                count(out["fixed_k"][lp][b], best_at_budget(rows, loop=lp, estimator=estimator, budget=b))
    return out


def property_pairs(properties_json: Mapping[str, Mapping], *, accuracy: str = "tight",
                   route_a: str = "route_a") -> list[dict]:
    """The (system, target) pairs of property-benchmark outputs at their tighter (or looser) accuracy.

    ``properties_json``: {system: the strict JSON of benchmarks/run_m4_properties.py}. Per pair {"system",
    "target", "eps", "rows" (PropertyRows: route A from rows[eps][route_a], routes B and HF-QPE from
    rows[eps]["route_b"]), "floors" {record: minimum error over k}, "best" and "best_f47" (the JSON best rows
    with the exact model bias and with F47's model-bias bound, None if absent)}; pairs without ``route_a``
    rows (no compiled records) are skipped.
    """
    if accuracy not in ("tight", "loose"):
        raise ValueError(f"accuracy must be 'tight' or 'loose', got {accuracy!r}")
    out = []
    for name, data in properties_json.items():
        for kind, blk in data["targets"].items():
            eps = float(blk["accuracies"][-1 if accuracy == "tight" else 0])
            key = str(blk["accuracies"][-1 if accuracy == "tight" else 0])
            rows = blk["rows"][key]
            if rows.get(route_a) is None:
                continue
            sens = (blk.get("model_bias_sensitivity") or {}).get(key)
            out.append({"system": str(name), "target": str(kind), "eps": eps,
                        "rows": [row_from_json(r) for r in rows[route_a]] + [row_from_json(r) for r in rows["route_b"]],
                        "floors": {lab: _json_number(f["min"]) for lab, f in blk["floors"].items()},
                        "best": blk.get("best", {}).get(key), "best_f47": None if sens is None else sens["best"]})
    return out


def compiled_bias_limited(pairs: Iterable[Mapping]) -> list[dict]:
    """The pairs whose exact R1 floor reaches eps while the F25-sliced compiled floor does not (route A bias-limited
    on the iterates the charged circuit prepares), with the unsliced compiled floor alongside."""
    out = []
    for p in pairs:
        fl = p["floors"]
        if "compiled_sliced" in fl and fl["r1_so"] <= p["eps"] < fl["compiled_sliced"]:
            out.append({"system": p["system"], "target": p["target"], "eps": p["eps"], "r1_so": fl["r1_so"],
                        "compiled_s1": fl.get("compiled_s1", math.nan), "compiled_sliced": fl["compiled_sliced"]})
    return out


def model_bias_sensitivity(pairs: Iterable[Mapping], *, loops: Sequence[str] = ("free", "meas_r1"),
                           estimator: str = "ae", rtol: float = 1e-6) -> list[dict]:
    """The best-row cells that rest on the exact model bias b_fact: those whose route or cost changes when routes B
    and HF-QPE charge F47's bound at q = 1 - F instead (the pairs' "best" and "best_f47" JSON rows).

    Each {"system", "target", "eps", "loop", "change" ("route": another route or none is selected; "cost": the
    same route at another cost), "ratio" (the F47 cost over the exact one), "exact" and "f47" {route,
    toffolis_over_qpe}}; costs equal to ``rtol`` count as unchanged.
    """
    out = []
    for p in pairs:
        if p.get("best") is None or p.get("best_f47") is None:
            continue
        for lp in loops:
            a, b = p["best"][lp][estimator], p["best_f47"][lp][estimator]
            ca = math.inf if a is None else _json_number(a["toffolis_over_qpe"])
            cb = math.inf if b is None else _json_number(b["toffolis_over_qpe"])
            ra = "none" if a is None else a["route"]
            rb = "none" if b is None else b["route"]
            same = ra == rb and (ca == cb or (math.isfinite(ca) and math.isfinite(cb)
                                              and abs(cb - ca) <= rtol * max(abs(ca), abs(cb))))
            if not same:
                ratio = cb / ca if math.isfinite(ca) and ca > 0 else math.inf
                out.append({"system": p["system"], "target": p["target"], "eps": p["eps"], "loop": lp,
                            "change": "route" if ra != rb else "cost", "ratio": ratio,
                            "exact": {"route": ra, "toffolis_over_qpe": ca},
                            "f47": {"route": rb, "toffolis_over_qpe": cb}})
    return out


def q6_comparison(properties_json: Mapping[str, Mapping], *, loops: Sequence[str] = ("free", "meas_r1"),
                  budgets: Sequence[int] = (2, 5, 10, 20, 60)) -> dict:
    """The route comparison over the (system, target) pairs at their tighter accuracy (F49, F53).

    Returns {"exact": :func:`route_tally` with route A on the exact R1 iterates, "compiled": the same with route
    A on the F25-sliced compiled iterates (rows[eps]["route_a_compiled"]; None without them),
    "compiled_bias_limited": :func:`compiled_bias_limited`, "slices": per system the sliced steps with their
    slices s, state errors ||Psi~^(s) - exp(eps D) Psi~|| and the steps capped at s = 16 (flag "slice_cap"),
    "largest_slice_error" (system, step, error) over the steps that met eta_s, "slice_caps" [(system, step,
    error)] of the capped steps, "model_bias_sensitivity": :func:`model_bias_sensitivity`}.
    """
    exact = property_pairs(properties_json)
    comp = property_pairs(properties_json, route_a="route_a_compiled")
    slices = {}
    for name, data in properties_json.items():
        rec = (data.get("records") or {}).get("compiled_sliced")
        if rec and rec.get("slice_errors") is not None:
            steps = [j for j, x in enumerate(rec["slice_errors"]) if x is not None]
            capped = sorted(int(f[len("slice_cap:step"):]) for f in (rec.get("flags") or [])
                            if f.startswith("slice_cap:step"))
            slices[str(name)] = {"steps": steps, "slices": [int(rec["slices"][j]) for j in steps],
                                 "errors": [_json_number(rec["slice_errors"][j]) for j in steps], "capped": capped}
    largest, caps = None, []
    for name, v in slices.items():
        for j, e in zip(v["steps"], v["errors"]):
            if not math.isfinite(e):
                continue
            if j in v["capped"]:
                caps.append((name, j, e))
            elif largest is None or e > largest[2]:
                largest = (name, j, e)
    return {"exact": route_tally([p["rows"] for p in exact], loops=loops, budgets=budgets),
            "compiled": route_tally([p["rows"] for p in comp], loops=loops, budgets=budgets) if comp else None,
            "compiled_bias_limited": compiled_bias_limited(exact), "slices": slices,
            "largest_slice_error": largest, "slice_caps": caps,
            "model_bias_sensitivity": model_bias_sensitivity(exact, loops=loops)}


def _route_count(tally: Mapping, route: str = "A") -> int:
    return int(tally.get(route, 0))


def q6_statements(comparison: Mapping) -> dict:
    """The sentences of U0's Q6 on route A at single budgets, on compiled iterates and on the exact model bias.

    ``comparison``: :func:`q6_comparison`. Returns {"verdict": [...], "uncertainty": [...]}.
    """
    def fmt(x: float) -> str:
        if not math.isfinite(x):
            return "inf"
        text = format(x, ".3g")
        if "e" in text:
            mant, exp = text.split("e")
            text = f"{mant}e{int(exp)}"
        return text

    def count(n: int, noun: str) -> str:
        return f"{n} {noun}" + ("" if n == 1 else "s")

    def cell(x: Mapping) -> str:
        return "none" if x["route"] == "none" else f"{x['route']} {fmt(x['toffolis_over_qpe'])}"

    ex = comparison["exact"]
    n = ex["pairs"]
    loops = list(ex["best"])
    budgets = list(next(iter(ex["fixed_k"].values())))
    label = {"free": "a free loop", "meas_r1": "a measured r = 1 loop", "meas_r03": "a measured r = 0.3 loop"}
    per_k = []
    for lp in loops:
        counts = [_route_count(ex["fixed_k"][lp][b]) for b in budgets]
        best = _route_count(ex["best"][lp])
        same = [str(b) for b, c in zip(budgets, counts) if c == best]
        per_k.append(f"{' / '.join(map(str, counts))} with {label.get(lp, lp)} ({best} with the cheapest usable "
                     f"budget per pair, " + (f"reproduced by k = {' and '.join(same)})" if same
                                             else "reproduced by no single budget)"))
    verdict = [f"with route A at a single pre-declared budget k = {' / '.join(map(str, budgets))} for every pair, "
               f"the cheapest route is route A on " + " and on ".join(per_k) + f" of the {count(n, 'pair')}"]
    comp = comparison.get("compiled")
    lim = comparison.get("compiled_bias_limited") or []
    if comp is not None:
        parts = [f"{_route_count(comp['best'][lp])} with {label.get(lp, lp)}" for lp in loops if lp in comp["best"]]
        verdict.append(f"with route A on the compiled iterates (F25 slices on steps 0-2, their sliced cost in "
                       f"C(Psi~_k)) it is the cheapest on {' and '.join(parts)} of the {count(comp['pairs'], 'pair')}")
    if lim:
        cells = ", ".join(f"{x['system']} {x['target']} ({fmt(x['r1_so'])} / {fmt(x['compiled_s1'])} / "
                          f"{fmt(x['compiled_sliced'])})" for x in lim)
        verdict.append(f"route A is bias-limited on the compiled iterates although the exact R1 floor reaches eps on "
                       f"{count(len(lim), 'pair')} (floors R1 / compiled s = 1 / compiled sliced): {cells}")
    else:
        verdict.append("no pair is bias-limited on the compiled iterates while the exact R1 floor reaches eps")
    unc = []
    big = comparison.get("largest_slice_error")
    detail = []
    if big is not None:
        sl = (comparison.get("slices") or {}).get(big[0]) or {"steps": [], "slices": []}
        s_big = dict(zip(sl["steps"], sl["slices"])).get(big[1])
        detail.append(f"up to {fmt(big[2])} where it meets eta_s, {big[0]}'s step {big[1]}"
                      + ("" if s_big is None else f" at s = {s_big}"))
    detail += [f"{fmt(e)} at the cap s = 16, {name}'s step {j}" for name, j, e in comparison.get("slice_caps") or []]
    unc.append("F25's slice rule (eta_s = 0.05, s <= 16) controls a sliced step's state error"
               + (f" ({'; '.join(detail)})" if detail else "") + ", not the property error of the compiled iterate")
    sens = comparison.get("model_bias_sensitivity") or []
    routes = [x for x in sens if x.get("change", "route") == "route"]
    costs = [x for x in sens if x.get("change") == "cost"]
    text = ("route B and HF-QPE charge the exact model bias b_fact of H' (F52's selection rests on it); with F47's "
            "bound at q = 1 - F instead (a sensitivity, never decisive) ")
    if routes:
        cells = ", ".join(f"{x['system']} {x['target']} {x['loop']} ({cell(x['exact'])} -> {cell(x['f47'])})"
                          for x in routes)
        text += (f"the best route changes on {count(len(routes), 'cell')}, whose route-B usability is an emulation "
                 f"fact like route A's: {cells}")
    else:
        text += "no best route changes"
    if costs:
        top = max(costs, key=lambda x: x["ratio"])
        text += (f"; {count(len(costs), 'more cell')} keep{'s its' if len(costs) == 1 else ' their'} route at another "
                 f"cost (up to x{fmt(top['ratio'])}, {top['system']} {top['target']} {top['loop']})")
    unc.append(text)
    return {"verdict": verdict, "uncertainty": unc}


# ------------------------------------------------------------------ compiled iterates (F49, F25)
def _sliced_steps(record: TrajectoryRecord, slices: int, sliced_steps) -> tuple[list, list, list]:
    s = _int_at_least(slices, 1, "slices")
    chosen = set(range(record.n_steps)) if sliced_steps is None else {int(j) for j in sliced_steps}
    dirs, eps, index = [], [], [0]
    for j, (d, e) in enumerate(zip(record.directions, record.eps)):
        n = s if j in chosen else 1
        dirs += [d] * n
        eps += [float(e) / n] * n
        index.append(index[-1] + n)
    return dirs, eps, index


def compiled_iterates(active: ActiveSpace, record: TrajectoryRecord, *, slices: int = 1,
                      sliced_steps: Sequence[int] | None = (0, 1, 2),
                      params: FTParameters = FTParameters()) -> list[np.ndarray]:
    """Psi~_0 .. Psi~_K of the record's steps compiled elementwise from HF, s slices on ``sliced_steps`` (F49).

    Step j in ``sliced_steps`` (None: every step) runs as s successive F3 circuits at eps_j/s
    (``ft.circuits.replay_elementwise``, with the record's irreps); one state per original iterate.
    """
    from .circuits import replay_elementwise

    dirs, eps, index = _sliced_steps(record, slices, sliced_steps)
    states = replay_elementwise(active, dirs, eps, irreps=record.irreps, params=params)
    return [np.asarray(states[i]) for i in index]


def sliced_prep_cost(record: TrajectoryRecord, *, slices: int = 1, sliced_steps: Sequence[int] | None = (0, 1, 2),
                     params: FTParameters = FTParameters()) -> tuple[np.ndarray, np.ndarray]:
    """(C(Psi~_k), Q(Psi~_k)) for k = 0..K of :func:`compiled_iterates` (F25: s C_step(d, eps/s) per sliced step)."""
    dirs, eps, index = _sliced_steps(record, slices, sliced_steps)
    pc = prep_cost(dirs, eps, method="elementwise", irreps=record.irreps, params=params)
    idx = np.asarray(index)
    return np.asarray(pc.cumulative)[idx].copy(), np.asarray(pc.logical_qubits)[idx].copy()
