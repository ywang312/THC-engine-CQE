# Milestone 4 results

M3 found that no measured CQE iteration pays as a state preparer for QPE at equilibrium: there one measured
iteration costs 154–1366 C_QPE already at HF, where the 11-candidate line search is all of it, while a better
initial state can save at most C_QPE(1/p_HF − 1) = 0.026–0.128 C_QPE, and only the free classical second-order
first step (F16) pays (M3 T6). M4 asks where that verdict changes: how the overlaps fall with chain length and
stretching and how far the gap allows a 1-mHa hand-off (Q1); what the cheapest measured iteration costs, how
it scales and whether a break-even size exists (Q2); which step rule keeps the iteration count at the lowest
measurement cost (Q3); whether a realizable closed-loop finite-shot fault-tolerant CQE keeps the exact
trajectory's overlap gains and pays (Q4); what an E1-referenced hand-off rule (rule B) certifies, how sound it
is under real QPE kernels, and what the QPE-side schemes do to the value of a better state (Q5); and whether
the CQE iterate or QPE-projected copies are the cheaper source of ground-state properties (Q6). U0
(`results/m4/summary.md`) states one verdict per question under the design's decision rules: on an exact
trajectory iteration k pays under scheme s iff T_s(k + 1) < T_s(k); in the closed loop iff the 95% bootstrap
interval of the mean paired saving lies above zero in the pre-declared cell (lagged controller at r = 1,
direct_mg, unit steps, exact propagation); the verdict of record uses the best realizable QPE-side scheme s*,
and oracle schemes and the oracle controller never decide.

The cost model as implemented is [m4-cost-model.md](m4-cost-model.md) (formulas F20–F53, parameters, Appendix
A); F1–F19 are M3's ([m3-cost-model.md](m3-cost-model.md)), and M3's tables are cited as M3 T1–T9. The tags
are M3's: [OF], [derived], [cited], [estimate], [assumption], [design], [build check]; after a number,
[derived] marks a value computed from the outputs named beside it rather than copied from them. §13 lists what
changed from the design specification.

**Systems** (F20; STO-3G, D2h symmetry-adapted RHF, canonical orbitals; R in bohr for hydrogen chains, Å for
N2; N active orbitals, N = n on the chains): the size axis `h<n>_r<R>` (n = 4–10, R = 2.5–4.0 bohr; U1–U4),
the overlap axis N2 CAS(6,6) `n2_r<R>` (R = 1.098–2.8 Å; U1–U9) and the active-space axis N2 CAS(10,8)
`n2v_r<R>` (U1–U4; "stretched N2" below is N2 at R = 2.0–2.8 Å in either active space); the equilibrium
references h4_r1.8, h6_r1.8, h8_r1.8 (U5); M3's systems h4, h6a, h6s, h8, lih, n2 and n2s, of which h4, h6s,
h8, n2 and n2s are built bit for bit by the aliases h4_r1.8, h6_r3.0, h8_r1.8, n2_r1.098 and n2_r2.0 (h6a is
H6 at 1.0 Å, not h6_r1.8); the hand-off set of U8 and U9 (19 distinct systems, §7); the property set of U10
and U11 (13 systems, §9); and the closed loop (headline n2_r2.0, n2_r2.4, n2_r2.8 and the small-gap chain
h4_r4.0; controls h4_r1.8, n2_r1.098, h6_r3.0 and, at N = 8, n2v_r2.4, n2v_r2.8, h8_r3.0; deep runs on
h4_r1.8, lih, n2_r1.098). The chains are a controllable model of the overlap catastrophe, not targets (1D
chains are classically easy).

**Records and loops.** The exact records are M3's L-BFGS trajectories from HF along the spin-free (sf) or
spin-orbital (so, the standard CQE generator) metric, with the exact first line search or the classical step
ε₂ (F16). R1 is the unit step (F26) after the classical step, which measures nothing beyond the residual; R3
adds a planned gate to a 3-point energy fit (F31); "line search" is the exact line search charged as M3's 11
candidate energies, "M3 grid" the lowest of the 11 (F13, F14). On exact trajectories the residual's relative
accuracy r (0.3, M3's value, 1 and 3) enters only the cost. The closed loop (F32–F38) draws every residual
with finite-shot noise at a budget planned by the oracle controller (not realizable) or the lagged one (F35).

**How the tables were produced.** From the repository root, on one thread (the scripts set the thread
variables), with M3's `results/ft_<system>.json` in `results/`:

    python benchmarks/run_m4_scaling.py --systems <s> --stage <stage> --out results/m4   # every system and stage
    python benchmarks/run_m4_scaling.py --out results/m4                                  # U1-U4 from the cache
    python benchmarks/run_m4_loop.py --resume --out results/m4                            # U5-U7
    python benchmarks/run_m4_handoff.py --m3-results results --out results/m4             # U8, U9 part 1
    python benchmarks/run_m4_properties.py --out results/m4                               # U10, U11
    python benchmarks/run_m4_summary.py --out results/m4                                  # U0, U4 schemes, U9 part 2, U11 deep runs, U12

The stages (`ft.scaling.SCAN_STAGES`: spectrum, qpe, residual_inputs, records, costs, compiled) run as
separate processes, so that none exceeds about 10 minutes. The five benchmarks take about 3.0 h on one thread
(peak 1470 MB): scaling stages 73 min, loop benchmark 73.0 min (U5) and 194 s (closed loop), hand-off 1497 s,
properties 285 s, summary 371 s (§11). Each script writes `<table>_<system>.json` and a markdown summary to
`results/m4` (strict JSON: nan as null, ±inf as "inf"/"-inf"); the tables below are condensed from
`scaling_summary.md`, `loop_summary.md`, `handoff_summary.md`, `properties_summary.md` and `summary.md` there,
which hold the full tables. All 78 integration checks pass, and M3's outputs are reproduced bit for bit (§11).

**Notation** (m4-cost-model.md). Costs are in units of C_QPE, one QPE to ΔE = 1 mHa with the selected encoding
H′ (F9), unless in Toffolis; T_res(k) is the residual at Ψ_k by its cheapest route (direct_mg: direct route,
matchgate shadows; ae_x: per-element amplitude estimation), c_iter(k) = C_iter(k)/C_QPE, C_loop(k) = Σ_{j<k}
C_iter(j), c_k = C(Ψ_k)/C_QPE with C(Ψ_k) the elementwise circuit of Ψ_k. p_k is the weight on ψ₀ of H ("on
H′": on ψ₀′); Ψ̃_k, p̃_k compiled or closed-loop states; G_k = p_{k+1}/p_k; Δ₁ = E₁ − E₀; n_cert(m) the size
up to which Δ₁ exceeds the margin m (F21). Rule E is M3's (certified iff E₁ − E_k > 3ΔE); rule B (F40) accepts
outcomes below θ, B_U at U + ΔE with a certified upper bound U on E₀′ (CISD(H′), F42) or B_L at E1lb − 5g (g
the QPE grid), accepting after n_v runs below θ, n_v − 1 of them verification runs (cost T_B, F40), E1lb being
the assumed lower bound on the excited levels a state populates (F41); ε_id is the identification error.
Schemes (F43): rep (repetition; "rep E" M3's rule E), aa (AA, known overlap, an oracle), bbht (exponential
search), fp (fixed-point AA), lt (filter and AA, an oracle), lt_bbht and 2s (step filter with exponential
search, two-stage QPE, at U), lt_bbht_L and 2s_L (the same at E1lb − w_L, w_L = 100 mHa), lt_rep (filter then
repetition, a sensitivity); s* the best realizable scheme, T_s(k) the cost of handing off Ψ_k under s after
C_loop(k), μ_k = T_s(k) − T_s(k + 1) + c_iter(k) the hand-off saving of the better state (F44),
ρ_k = c_iter(k)/μ_k (∞ if μ_k ≤ 0; pays iff ρ_k < 1), n* the break-even size (F45), p₀* the break-even HF
overlap (F44). Route A estimates a property on the CQE iterate at a budget k, route B on QPE-projected copies
from HF or Ψ₁ (B0) or a later iterate (Bk), HF-QPE by Hellmann–Feynman QPE (F46–F53).

**In short.** At equilibrium M3's verdict holds in M4's settings at r ≤ 1 and in the closed loop; the one
exception, the exact spin-free unit step on h4 at r = 3, gains 1.66e-4 C_QPE (0.0163%) on a noise-free
trajectory (§5). Without the line search the cheapest measured iteration costs 0.0184–519 C_QPE at r = 1; at
M3's r = 0.3 it is 89–166 times cheaper than M3's iteration k = 1 on n2, h6s and n2s (M3 T3 [derived]). At the
best realizable QPE-side scheme it nevertheless pays only on short stretched chains and, narrowly, on N2 at
2.8 Å: on exact trajectories at r = 1 on h4_r3.5, h4_r4.0 and h6_r4.0 (by 1.96–30.8%), in the closed loop on
h4_r4.0 and on n2_r2.8 (+0.0212 [0.0151, 0.027] C_QPE, 1.35%). These pays are fragile. They come mostly from
BBHT's overshoot (h4_r3.5, the one certified hand-off among them) or from repairing the overshoot of the free
classical step (h4_r4.0, h6_r4.0; no damped free step was run); outside h4_r3.5 they go through hand-offs
whose termination is not certified; and most of them do not survive the slices their compiled steps need, a
filter about 25% cheaper or a neighbouring r cell (Main results 2 and 4). The second measured iteration pays
at s* on no exact trajectory and in no deciding cell. The QPE side is the largest lever on N2 at every bond
length, on h6s and at equilibrium on h4, h6a, h8 and n2, through the scheme rather than the acceptance rule:
on stretched N2 the E1lb-referenced filter with exponential search, whose termination no certified U
guarantees, cuts the HF hand-off from 3.939–12.8 C_QPE (M3's uncertified oracle rule E) to 1.466–1.885, which
leaves a better state little to save. On LiH and the six stretched chains of the hand-off set the classical
step (and, where it pays, the measured iteration) carries the savings (§8). Rule B, conditional on the assumed
E1lb, gives every exact state a realizable acceptance (at HF by B_L, without a certificate and with
termination not certified, on nine stretched systems), but on M3's stretched systems it does not lower the
cheapest certified hand-off by a measured or classical-step loop (h6s 1.091 → 2.091, n2s 2.525). Along the
chains, at r = 1, the overlap catastrophe outgrows the residual's cost only from n* = 86 (repetition) or 166
(best scheme), at overlaps ≤ 1.5e-5 and outside the kernel-aware hand-off window: a model statement far beyond
n = 10. For properties the CQE iterate is the cheapest estimator on most pairs only with a free loop (37 of 66
at the tighter accuracy, against 11 with a measured r = 1 loop); with a measured loop QPE-projected copies of
a later iterate win on 53 of the 66 pairs in Toffolis, but most of those single coherent runs have no
surface-code layout or exceed 7 days (§10).

**Main results.**

1. **Q1, overlap scaling (§1, §2).** p_HF decays exponentially with chain length, at 0.0546–0.0620,
   0.1078–0.1149, 0.1760–0.1824 and 0.2431–0.2493 per atom (last successive rate to Aitken limit) at R =
   2.5–4.0 bohr, reaching 1e-3 at n = 113.5–127.4, 61.6–65.0, 39.0–40.0 and 28.6–29.1; the free classical step
   decays 1.7–4.1 (sf) and 1.9–3.9 (so) times more slowly [derived from U2]. The 1-mHa windows are n_cert =
   440–912 to 47–62 for rule E but 110–175 to 12 for kernel-aware B_L, so p_HF = 1e-3 lies inside rule E's
   window at every R and beyond B_L's at R ≥ 3.5 bohr (at 2.5 and 3.0 bohr inside it under the power-law gap
   and beyond it under A′/n); rule E certifies HF on only 4 of the 25 scanned systems. Uncertainty: model
   brackets validated only by the n = 10 holdout (HF: +0.37% to +0.73% at the last rate, −0.09% to −0.42% at
   the Aitken limit; all series +0.28% to +2.77% and −2.36% to +0.45%, the R1 Ψ₂ brackets at R ≥ 3.0 bohr
   overestimating p(10) at both ends).
2. **Q2, cost scaling and break-even (§3, §4).** The cheapest measured iteration costs
   T_res(1)/C_QPE = 0.0184–519 at r = 1 (∝ r⁻²), growing as N^8.39–9.55 on the chains (V_m 8.02–8.29, C_QPE
   2.12–2.43). On exact trajectories (U4: the R1 records of both metrics, p on H) it pays at s* on 2, 5 and 12
   of 46 system-metric rows at r = 0.3, 1 and 3; at r = 1 on h4_r3.5 sf/so (1.96%/9.7%), h4_r4.0 sf/so
   (13.4%/30.8%) and h6_r4.0 so (17.4%), the last three by repetition with B_L (no certificate, termination
   not certified): the design's expectation that none pays at N ≤ 10 is refuted, fragile in mechanism rather
   than in size (h4_r3.5 sf pays only through BBHT's overshoot and fails with filter-then-repeat admitted; on
   h4_r4.0 and h6_r4.0 the gain mostly repairs the overshoot of the free classical step (so: p₁ = 0.4995 and
   0.4047 against 0.9412 and 0.8672 after an exact line search), against which no damped free step was run;
   h4_r3.5 sf, h4_r4.0 sf and h6_r4.0 so would fail with the 4–16 slices their steps need [derived]; and
   h4_r3.5 so, the one pay on a certified hand-off that also holds under repetition (ρ₁ = 0.509), shrinks to
   0.0452% with filter-then-repeat). On N2 at 2.4–2.8 Å it pays under repetition but loses 16.0–25.0% at s*
   (r = 1); at s* never on CAS(10,8) or on N2 at equilibrium (on M3's equilibrium systems only h4 sf at r = 3
   pays, by 0.0163%; §8); the second iteration never pays at s* (under repetition only on N2 at 2.0–2.8 Å at
   r = 3); the free classical step pays on 40 of 46 rows. Extrapolated at r = 1, n* ≥ 86 (repetition,
   p₁(n*) ≤ 1.5e-5) and ≥ 166 (best scheme), none spin-orbital at R ≤ 3.5 bohr, always outside the
   kernel-aware window: a model bracket 76–2988 sites beyond n = 10.
3. **Q3, step rules (§5, §6).** The two unit-step rules reach the reference iterate at the lowest measurement
   cost at r = 1: with ties counted for each, R3 on 25 and R1 on 24 of 30 resolved pairs, the two coinciding
   on 20. R1 cuts C_loop(k*) to 0.0038–0.87 of the exact line search's on the 26 pairs where both reach k*,
   for 0–3 (so) and 0–6 (sf) extra iterations (R3 within these ranges except on h6_r4.0: 0.957 and four extra
   iterations so, k* not reached sf); at r = 0.3 the gain reverses on the hardest H6 pairs. The
   energy-precision wall holds at r = 1 (the Armijo pair costs more than the residual at k = 1 on 13 of 17
   systems and 5.36–569 times near k*; at M3's r = 0.3, where ρ_E is 0.09 times as large, the 3-point fit is
   cheaper than the residual at k = 1 on 24 of the 34 R1 rows), and under noise the fixed-precision energy
   rules (line search, M3's grid) cost 449–485 C_QPE in the first measured iteration and 757–908 in the
   second. At equilibrium one R1 iteration at r = 1 costs 0.127–0.474 (h4) and 0.339–0.79 (n2) against at most
   0.058 and 0.080 C_QPE to gain: M3's verdict holds at r ≤ 1 (at r = 3 h4's spin-free unit step pays by
   0.0163%). Model dependence: deterministic emulation over a finite costed horizon, κ = 1.
4. **Q4, closed loop (§6).** In the deciding cells (lagged r = 1, 100 replicates) the first measured iteration
   pays under repetition on all four headline systems (0.268–1.73 C_QPE) but at s* only on n2_r2.8 (0.0212
   [0.0151, 0.027], lt_bbht_L, termination not certified) and h4_r4.0 (0.844 [0.841, 0.846], repetition with
   B_L), losing on n2_r2.0 (−0.054 [−0.0545, −0.0535]) and n2_r2.4 (−0.0111 [−0.0153, −0.007]); the second
   loses everywhere (−0.686 to −1.72 at s*), and every control loses at s*. The loop keeps the first step's
   overlap gain (median p̃₂ within 0.012 of exact), not the later ones; the best first accuracy is r₁ ≥ 3 on
   stretched N2 and r₁ ≈ 1 on h4_r4.0. The s* verdicts are fragile: the lagged r = 0.5 cell reverses n2_r2.8
   (−0.166 [−0.17, −0.161]), the lagged r = 2 cell reverses n2_r2.4 (+0.0276 [0.0232, 0.0318], a cell that is
   not realizable because the angle cap bound once), and compiled propagation reverses h4_r4.0. Model
   dependence: M3's independent Gaussian noise, variances at one state, exact propagation; the N2 margins at
   s* (at most 0.054 C_QPE) are smaller than what the filter costing and the certificate move in T_s*, and
   both modelling choices favour the n2_r2.8 pay, which a filter about 25% cheaper would reverse (§6, §12).
   Against the design, its N2 savings at oracle r = 3 are reproduced (under repetition with verification runs
   10.60 / 19.16 / 24.77% against its 10 / 19 / 25% at 2.0 / 2.4 / 2.8 Å; under lt_bbht_L with margins as
   narrow as it expected), except that its near miss at 2.8 Å is a narrow pay; that the second iteration does
   not pay is confirmed, that none pays on CAS(10,8) holds at s* but not under repetition, and that none pays
   on the chains at N ≤ 10 is refuted on h4_r4.0.
5. **Q5, hand-off and schemes (§7, §8).** Rule B accepts HF on all 19 distinct hand-off systems (B_U with
   CISD(H′) on 10, B_L with two runs on 9 stretched ones, without a certificate and with termination not
   certified) at M3's oracle T_E plus one C_QPE per verification run (1.026–13.8 C_QPE), covering the 83 of
   209 exact states rule E does not certify. On M3's T6 it changes only h6s and n2s: the certified optima of
   the measured loops and of n2s's spin-free classical step fall from 554.1/517.1 and 570.8/188.7/1073 to
   2.744 and 3.956 C_QPE, but h6s's other optima rise by one C_QPE, so the cheapest certified hand-off by a
   measured or classical-step loop does not fall (h6s 1.091 → 2.091, n2s 2.525). On stretched N2 lt_bbht_L,
   whose termination is not certified, cuts the HF hand-off from 3.939–12.8 C_QPE (M3's uncertified oracle
   rule E with p on H′, as in U9; 3.956 on n2s with M3's p on H, as in M3 T6; 3.939–13.8 under rule B) to
   1.466–1.885 (n2_r2.4: 8.928 → 1.705 → Ψ₁ 1.502, design 8.93 → 1.67 → 1.47); at equilibrium the scheme saves
   0.030–0.074 on h4, h6a, h8 and n2 (LiH keeps repetition); on the stretched chains s* costs more than M3's
   oracle rule E on five of six. The design's ordering (scheme, then classical step, then measured iteration)
   is confirmed on N2 at every bond length, on h6s and at equilibrium on h4, h6a, h8 and n2, but fails on LiH
   and on the six stretched chains, where the classical step and, on h4_r3.5, h4_r4.0 and h6_r4.0, the
   measured iteration carry the savings. An iteration of 1000 C_QPE reaching the ground state pays only below
   p₀* = 0.000999 (rep), 8.39e-6 (bbht), 1.9e-6 (aa), 1.48e-7 (lt_bbht_L) on n2_r2.4. Model dependence:
   assumed E1lb (N ≤ 8), K₀ = 1 (LiH understated up to 1.9×); CISDT(H′) flips the r = 3 N2 verdicts.
6. **Q6, properties (§9).** With a free loop route A is cheapest on 37 (tighter) and 50 (looser) of 66 pairs,
   and where its exact floor reaches ε route B costs 4.99–300× route A's oracle row; with a measured loop Bk
   wins on 53 of 66 tighter pairs in Toffolis, route A keeps 11 (r = 1) and 9 (r = 0.3), at any single
   pre-declared budget at most 5. As single coherent runs 37 of the 135 best tighter Bk rows have no layout
   and 75 of the 98 laid out exceed 7 days (route A: 3 of 57), so only 23 meet M3's deadline, although among
   the laid-out rows Bk still wins 40 pairs at either measured r (B0 2 more at r = 0.3; §9, §10). Route A is
   bias-limited on 23 pairs, and on 26 more when compiled. HF-QPE gives the LiH dipole for 99.6 and 970 C_QPE,
   16–58× below AE on filtered copies, above free-loop route A (34.5, 221); at 1e-3 a.u. it and route B rely
   on charging the exact model bias of H′ (with F47's bound the measured-loop cells go to route A at 8.05e5
   and 2.68e6 C_QPE). Model dependence: route A's usability at a budget and route B's exact model bias are
   emulation facts (F47's bound changes 11 cells).
7. **Soundness of the hand-off (§7).** On every exact state the identification error of rule B's form is at
   most 5.00e-4 (the logical budget; kernel ≤ 1.24e-6), while the rect window (above 1e-3 on 27 of the 121
   exact states with a certified U) and the literal threshold E₁′ − ΔE (on 129 of the 209 exact states, up to
   0.706) are unsound, as the design expected. Compiled states lose the design-floor bound on 46 of the 57
   compiled states (52 of the 63 compiled rows of the 21 labels), since their elementwise circuits populate
   levels below E1lb; on stretched N2 these are spin contaminants below θ that verification runs accept as
   readily as ψ₀′ (total up to 1.15e-3, within the 2e-3 budget).
8. **Physical units (§10).** On n2_r2.4 the first residual is 7965 (r = 1) or 885 (r = 3) shots of 12.8 ms on
   5.682e5 physical qubits, the hand-off QPE one 309.1 s run on 1.14e6, the lt_bbht_L filter and AA sequence,
   laid out as one run, 143.5 s on 1.118e6: Toffoli counts, not wall clock, decide Q4 and Q5. Property AE runs
   are single coherent runs: 39 of the 198 best tighter rows have no layout, and 80 of the 159 with one run
   longer than 7 days.

**Reading the tables.** Cells are copied from the outputs at their precision (costs and most reals 3–4
significant digits, overlaps 3–5 decimals), mostly with the outputs' ASCII minus and e-notation; the prose
writes "−". Costs are in C_QPE unless in Toffolis, energies and levels in mHa above E₀ (H) or E₀′ (H′). A
saving in per cent is 1 − T(k + 1)/T(k). "a-b" in an n* or n(p) column is a model bracket, never a confidence
interval; "[a, b]" after a mean or a paired saving is its 95% percentile-bootstrap interval (2000 resamples),
after a median the 16–84% band; "pays" marks an interval above zero; "X -> Y" in an s* column is the optimum
at Ψ_k and at Ψ_{k+1}. "-": not reached, not run, not terminating or undefined; "inf": no gain; "(k_p90)": the
reference iterate is the first with p ≥ 0.9; "[k]": a route-A budget or route-B start; "none": no break-even
below n = 3000, no route or no layout. Flags: termination_not_certified (the E1lb-referenced forms or B_L
terminate in the emulation, but no certified U guarantees it), no_certificate (no certified U or p_lb),
point_group_broken (E1lb over every irrep), spin_tol_relaxed (the ⟨S²⟩ tolerance of the spectrum was relaxed
to 1e-4), slice_cap (F25's slices miss η_s at s = 16), aitken_fallback (the drift of the rates grows),
slope_debias_approx (F33's debiased slope is approximate) and no_layout (no surface-code layout of a run).

## 1. Systems and spectra (U1)

On the chains, length and stretching drive the overlap and the gap apart. From n = 4 to 10, p_HF falls from
0.8359 to 0.6097 at R = 2.5 bohr and from 0.4341 to 0.1036 at 4.0 bohr, E_HF − E₀ grows linearly (123.4 to
295.3 and 358.7 to 884.8 mHa) and Δ₁ shrinks (266.28 to 132.09 and 28.24 to 14.07 mHa). M3's rule E therefore
certifies HF on only 4 of the 25 scanned systems (h4_r2.5, h6_r2.5 and N2 at 1.098 Å in both active spaces),
elsewhere only after 1–6 exact L-BFGS steps, and on h10_r4.0 no iterate of the six-step records passes. On N2
CAS(6,6) stretching alone takes p_HF from 0.9257 to 0.07849 at N = 6, with Δ₁ = 191.73–711.49 mHa; CAS(10,8)
gives nearly the same overlaps and gaps at N = 8 (p_HF 0.9173–0.07817, Δ₁ 195.72–696.30 mHa) with λ =
23.317–28.870 Ha against 7.577–9.716 Ha, so the active-space axis changes N and λ at nearly fixed p_HF and Δ₁,
as the design intended. The selected QPE is SF on H4 and N2 and THC on H6–H10, at C_QPE = 2.802e6 (h4_r4.0) to
4.598e7 (n2v_r1.098) Toffolis.

**U1** (the 25 scanned systems; `results/m4/scaling_summary.md`, U1, with the flags). Energies in mHa; Δ₁ with
E₁ the lowest excited singlet in the irrep of ψ₀ (M3's rule) and in any irrep; QPE: the selected encoding
(F9), λ in Ha, C_W in Toffolis; k_cert: the first iterate with E₁ − E_k > 3ΔE on the four exact six-step
L-BFGS records (sf, so; exact first line search "ex" or classical step "cl"); '-': none up to k = 6, or not
run (the exact-first-step records at N = 10).

| system | N | dets | p_HF | E_HF − E₀ | Δ₁ same / any | QPE (kind, λ, C_W) | C_QPE | Q_QPE | k_cert (sf ex / sf cl / so ex / so cl) |
|---|---|---|---|---|---|---|---|---|---|
| h4_r2.5 | 4 | 36 | 0.8359 | 123.4 | 266.28 / 266.28 | sf, 5.473, 384 | 3.301e+06 | 156 | 0 / 0 / 0 / 0 |
| h6_r2.5 | 6 | 400 | 0.7557 | 181.0 | 200.04 / 200.04 | thc, 7.105, 1013 | 1.131e+07 | 217 | 0 / 0 / 0 / 0 |
| h8_r2.5 | 8 | 4900 | 0.6801 | 238.2 | 159.27 / 159.27 | thc, 10.392, 1331 | 2.173e+07 | 258 | 1 / 1 / 1 / 1 |
| h10_r2.5 | 10 | 63504 | 0.6097 | 295.3 | 132.09 / 132.09 | thc, 13.895, 1713 | 3.739e+07 | 297 | - / 1 / - / 1 |
| h4_r3.0 | 4 | 36 | 0.7012 | 191.7 | 138.96 / 138.96 | sf, 5.023, 384 | 3.03e+06 | 154 | 1 / 1 / 1 / 1 |
| h6_r3.0 | 6 | 400 | 0.5733 | 282.2 | 104.67 / 104.67 | thc, 6.120, 1013 | 9.738e+06 | 217 | 1 / 1 / 1 / 1 |
| h8_r3.0 | 8 | 4900 | 0.4648 | 372.4 | 83.32 / 83.32 | thc, 8.926, 1331 | 1.866e+07 | 258 | 2 / 2 / 1 / 1 |
| h10_r3.0 | 10 | 63504 | 0.3746 | 462.5 | 69.04 / 69.04 | thc, 11.875, 1612 | 3.007e+07 | 296 | - / 2 / - / 2 |
| h4_r3.5 | 4 | 36 | 0.5535 | 274.4 | 65.26 / 65.26 | sf, 4.779, 384 | 2.883e+06 | 154 | 1 / 2 / 1 / 2 |
| h6_r3.5 | 6 | 400 | 0.3961 | 406.6 | 49.26 / 49.26 | thc, 5.457, 1013 | 8.683e+06 | 217 | 2 / 2 / 2 / 2 |
| h8_r3.5 | 8 | 4900 | 0.2805 | 538.5 | 39.22 / 39.22 | thc, 7.895, 1331 | 1.651e+07 | 258 | 3 / 3 / 2 / 2 |
| h10_r3.5 | 10 | 63504 | 0.1973 | 670.4 | 32.50 / 32.50 | thc, 10.489, 1612 | 2.656e+07 | 296 | - / 5 / - / 3 |
| h4_r4.0 | 4 | 36 | 0.4341 | 358.7 | 28.24 / 28.24 | sf, 4.645, 384 | 2.802e+06 | 154 | 2 / 3 / 1 / 2 |
| h6_r4.0 | 6 | 400 | 0.272 | 534.1 | 21.33 / 21.33 | thc, 4.997, 1013 | 7.952e+06 | 215 | 3 / 5 / 3 / 3 |
| h8_r4.0 | 8 | 4900 | 0.1685 | 709.5 | 16.98 / 16.98 | thc, 7.207, 1331 | 1.507e+07 | 258 | 6 / - / 4 / 5 |
| h10_r4.0 | 10 | 63504 | 0.1036 | 884.8 | 14.07 / 14.07 | thc, 9.540, 1612 | 2.416e+07 | 294 | - / - / - / - |
| n2_r1.098 | 6 | 400 | 0.9257 | 126.0 | 711.49 / 411.71 | sf, 9.716, 828 | 1.264e+07 | 185 | 0 / 0 / 0 / 0 |
| n2_r1.6 | 6 | 400 | 0.6567 | 328.5 | 218.53 / 187.19 | sf, 8.247, 797 | 1.033e+07 | 185 | 1 / 1 / 1 / 1 |
| n2_r2.0 | 6 | 400 | 0.2528 | 565.5 | 191.73 / 191.73 | sf, 7.779, 777 | 9.495e+06 | 185 | 2 / 3 / 1 / 1 |
| n2_r2.4 | 6 | 400 | 0.1125 | 777.0 | 229.93 / 229.93 | sf, 7.591, 768 | 9.158e+06 | 185 | 3 / 3 / 2 / 3 |
| n2_r2.8 | 6 | 400 | 0.07849 | 912.0 | 243.25 / 243.25 | sf, 7.577, 709 | 8.439e+06 | 182 | 5 / 3 / 2 / 3 |
| n2v_r1.098 | 8 | 3136 | 0.9173 | 156.7 | 696.30 / 348.41 | sf, 28.870, 1014 | 4.598e+07 | 259 | 0 / 0 / 0 / 0 |
| n2v_r2.0 | 8 | 3136 | 0.2464 | 583.6 | 195.72 / 195.72 | sf, 25.033, 920 | 3.618e+07 | 259 | 3 / 3 / 1 / 1 |
| n2v_r2.4 | 8 | 3136 | 0.1112 | 784.7 | 232.48 / 232.48 | sf, 24.014, 920 | 3.47e+07 | 259 | 5 / 6 / 4 / 3 |
| n2v_r2.8 | 8 | 3136 | 0.07817 | 914.8 | 244.11 / 244.11 | sf, 23.317, 914 | 3.348e+07 | 259 | 6 / 6 / 5 / 5 |

- **Builds and flags.** 12 singlet roots at the ⟨S²⟩ tolerance 1e-6 (1e-4 only on n2v_r2.4); at N = 10 two
  roots, root 1 in the ground irrep on every H10. The two gaps coincide on every chain and on N2 from 2.0 Å.
- **The window closes from both sides.** At fixed R, E_HF − E₀ grows by 57.1–57.6 mHa per added H₂ unit at 2.5
  bohr and by 175.3–175.4 mHa at 4.0 bohr [derived] and exceeds Δ₁ already at n = 4 for R ≥ 3.0 bohr; rule E,
  tied to the energy of the hand-off state, then needs an iterate within Δ₁ − 3ΔE of E₀, 11.07 mHa on h10_r4.0
  [derived]: the regime rule B addresses (§7). On N2 the singlet gap does not close: Δ₁ falls to 191.73 mHa at
  2.0 Å and rises again to 243.25 mHa at 2.8 Å, while the lowest quintet of the ground irrep approaches E₀′
  (1.21 mHa at 2.8 Å; U1 supplement).
- **M3's systems.** The aliases reproduce M3 T1 and T2 (h6_r3.0 = h6s: THC, λ = 6.120 Ha, C_QPE = 9.738e6,
  p_HF = 0.5733; n2_r1.098 = n2: SF, 9.716, 1.264e7; n2_r2.0 = n2s: SF, 7.779, 9.495e6; M3: 6.11959, 9.74e6,
  0.5733; 9.71614, 1.26e7; 7.77897, 9.49e6), and their k_cert reproduce M3 T6's certification facts. h4_r1.8,
  h6_r1.8 and h8_r1.8 are not scanned (§13; their spectra and QPE are in `loop_<s>.json`).
- **Uncertainty.** None statistical (FCI in the active space, OpenFermion's QPE costs); 70 of the design's 72
  reference numbers reproduce (§11).

**U1 supplement** (the QPE side per system; `results/m4/summary.md`, "QPE side per system (F40-F43) and the U1
supplement", with c_filt, c_c and sin θ₀). It carries the design's U1 column "lowest same-irrep level of any
spin", which needs the model spectrum of H′ (N ≤ 8; §13). Levels in mHa above E₀′; "any spin": the lowest
excited level of the ground irrep of any spin, with its ⟨S²⟩; E1lb = E₁(H′) − 0.1 mHa, or E₁(H) − 1 mHa
without a model spectrum (F41); "U certified": CISD(H′) ≤ E1lb − 2ΔE (F42); rule B (n_v): the form for a state
without a dedicated certificate (F40); "L forms" (lt_bbht_L, 2s_L): '-' not terminating (E₀′ > E1lb − w_L),
"certified" with a certified U ≤ E1lb − w_L, else termination_not_certified.

| system | E1lb − E₀′ | any spin (⟨S²⟩) | CISD(H′) − E₀′ | U certified | rule B (n_v) | L forms |
|---|---|---|---|---|---|---|
| h4_r1.8 | 565.57 | 491.40 (2) | 1.04 | yes | B_U (1) | certified |
| h4_r2.5 | 266.17 | 222.66 (2) | 6.93 | yes | B_U (1) | certified |
| h4_r3.0 | 138.86 | 113.25 (2) | 19.45 | yes | B_U (1) | certified |
| h4_r3.5 | 65.16 | 52.32 (2) | 38.25 | yes | B_U (2) | - |
| h4_r4.0 | 28.14 | 22.44 (2) | 39.24 | no | B_L (2) | - |
| h6_r2.5 | 199.96 | 173.89 (2) | 20.86 | yes | B_U (1) | certified |
| h6_r3.0 | 104.58 | 88.57 (2) | 54.50 | yes | B_U (2) | termination_not_certified |
| h6_r3.5 | 49.15 | 40.99 (2) | 108.25 | no | B_L (2) | - |
| h6_r4.0 | 21.23 | 17.60 (2) | 170.30 | no | B_L (2) | - |
| h8_r2.5 | 159.27 | 141.01 (2) | 39.51 | yes | B_U (1) | certified |
| h8_r3.0 | 83.24 | 71.86 (2) | 97.34 | no | B_L (2) | - |
| h8_r3.5 | 39.12 | 33.29 (2) | 185.95 | no | B_L (2) | - |
| h8_r4.0 | 16.88 | 14.30 (2) | 287.13 | no | B_L (2) | - |
| h10_r2.5 | 131.10 | - | 61.45 | yes | B_U (1) | termination_not_certified |
| h10_r3.0 | 68.15 | - | 144.98 | no | B_L (2) | - |
| h10_r3.5 | 31.99 | - | 269.23 | no | B_L (2) | - |
| h10_r4.0 | 13.12 | - | 408.61 | no | B_L (2) | - |
| n2_r1.098 | 711.39 | 618.54 (6) | 7.97 | yes | B_U (1) | certified |
| n2_r1.6 | 218.55 | 107.30 (6) | 64.33 | yes | B_U (1) | certified |
| n2_r2.0 | 191.81 | 21.39 (6) | 134.22 | yes | B_U (1) | termination_not_certified |
| n2_r2.4 | 230.76 | 4.35 (6) | 225.69 | yes | B_L (2) | termination_not_certified |
| n2_r2.8 | 244.34 | 1.21 (6) | 293.50 | no | B_L (2) | termination_not_certified |
| n2v_r1.098 | 696.16 | 626.20 (6) | 12.33 | yes | B_U (1) | certified |
| n2v_r2.0 | 196.32 | 25.84 (6) | 150.95 | yes | B_U (2) | termination_not_certified |
| n2v_r2.4 | 232.51 | 6.33 (6) | 232.05 | no | B_L (2) | termination_not_certified |
| n2v_r2.8 | 243.95 | 1.34 (6) | 294.99 | no | B_L (2) | termination_not_certified |

- **Levels of other spin and certificates.** The lowest level of the ground irrep is a triplet on the chains
  and a quintet on N2, 4.35 mHa (2.4 Å) and 1.21 mHa (2.8 Å) above E₀′ against the singlet E1lb − E₀′ = 230.76
  and 244.34 mHa: E1lb bounds only the levels a spin-pure singlet can populate (§7). CISD(H′) − E₀′ grows with
  n and R (1.04 to 408.61 mHa) while the gap closes: a certified U exists on 14 of the 26 systems, and without
  one rule B falls back to B_L with two runs. On n2_r2.4 the U's margin is only 3.077 mHa (D_U = 2.07), at
  which B_U needs three runs, so rule B takes B_L with two there too; n2_r2.4 and n2v_r2.4 carry
  point_group_broken, so E1lb is taken over every irrep, as for THC.
- **Realizable schemes on the chains.** The E1lb-referenced forms terminate only where E1lb − E₀′ ≥ w_L = 100
  mHa, which excludes 10 of the 16 chains (h4_r3.5–4.0, h6_r3.5–4.0, h8_r3.0–4.0, h10_r3.0–4.0). There the
  realizable set is rep, bbht, fp with a certified p_lb and, with a certified U (h4_r3.5 only), lt_bbht and
  2s, confirming the design's expectation for chains with Δ₁ < w_L; §4 finds s* ∈ {rep, bbht, lt_bbht, 2s} on
  them.

## 2. Overlap scaling (U2)

On the chains every overlap decays exponentially with n at fixed R, at per-atom rates that still drift upward
at n = 10: p_HF reaches 1e-3 at n = 113.5–127.4, 61.6–65.0, 39.0–40.0 and 28.6–29.1 at R = 2.5, 3.0, 3.5 and
4.0 bohr (between the last successive rate and the Aitken limit), model brackets far beyond n ≤ 10. The free
classical step decays 1.7–4.1 (sf) and 1.9–3.9 (so) times more slowly (α_HF/α at the last successive rate and
at the Aitken limit [derived from U2]) and leaves p₁ = 0.0199–0.181 (sf) and 0.0232–0.174 (so) there; the
spin-orbital first measured iteration decays faster than its classical Ψ₁ at R ≤ 3.5 bohr, so its gain falls
at the largest sizes (G₁ = 0.9919 < 1 on h10_r3.0). The gap closes as Δ₁ ∝ n^−0.84 while nΔ₁ still rises: the
1-mHa windows are n_cert(m_E) = n_cert(m_B) = 440–912 down to 47–62, but 110–175 down to 12 for kernel-aware
B_L with two runs, so p_HF = 1e-3 lies inside the rule-E window at every R and beyond the B_L window at
R ≥ 3.5 bohr (at every R under the A′/n gap). On N2 at N = 6, p_HF falls from 0.926 to 0.078 between 1.098 and
2.8 Å.

**U2** (overlap scaling, F22; `results/m4/scaling_summary.md`, U2). Successive rates α^(n_i) = [ln p(n_{i−1})
− ln p(n_i)]/(n_i − n_{i−1}); (a, α) the least-squares fit on n = 6, 8, 10, reported only; α^A the Aitken
limit; the extrapolation bracket is [α_last, α^A]. Holdout: p(10) predicted from n = 4, 6, 8 at each end of
the bracket (relative error); n(p = 1e-3) at α_last and α^A; "p_1 there": the classical-step series (sf; so)
at the same n and rate. k = 1: the classical Ψ₁; k = 2: the R1 Ψ₂ (r1_sf not run at N = 10).

| R | record | k | n | p | successive | a, alpha | alpha^A | holdout error (last / Aitken) | n(p = 1e-3) | p_1 there |
|---|---|---|---|---|---|---|---|---|---|---|
| 2.5 | hf | 0 | 4 / 6 / 8 / 10 | 0.8359 / 0.7557 / 0.6801 / 0.6097 | 0.0504, 0.0528, 0.0546 | +0.043, 0.0537 | 0.0620 | n=10: +0.37% / -0.09% | 127.4-113.5 | sf 0.0345-0.0199; so 0.0852-0.0667 |
| 2.5 | classical_sf | 1 | 4 / 6 / 8 / 10 | 0.9589 / 0.9139 / 0.8683 / 0.8226 | 0.0240, 0.0256, 0.0270 | +0.068, 0.0263 | 0.0360 | n=10: +0.28% / -0.04% | 258.5-196.7 | - |
| 2.5 | classical_so | 1 | 4 / 6 / 8 / 10 | 0.9791 / 0.9501 / 0.9167 / 0.8809 | 0.0151, 0.0179, 0.0199 | +0.063, 0.0189 | 0.0249 | n=10: +0.40% / -0.16% | 350.8-282.0 | - |
| 2.5 | r1_sf | 2 | 4 / 6 / 8 | 0.9898 / 0.9666 / 0.9365 | 0.0119, 0.0158 | +0.061, 0.0158 | 0.0197 | - | 441.5-355.6 | - |
| 2.5 | r1_so | 2 | 4 / 6 / 8 / 10 | 0.992 / 0.9722 / 0.9429 / 0.9056 | 0.0100, 0.0153, 0.0202 | +0.080, 0.0178 | 0.0807 | n=10: +0.98% / -0.08% | 347.1-94.4 | - |
| 3 | hf | 0 | 4 / 6 / 8 / 10 | 0.7012 / 0.5733 / 0.4648 / 0.3746 | 0.1007, 0.1049, 0.1078 | +0.083, 0.1064 | 0.1149 | n=10: +0.59% / -0.24% | 65.0-61.6 | sf 0.0755-0.07; so 0.143-0.119 |
| 3 | classical_sf | 1 | 4 / 6 / 8 / 10 | 0.947 / 0.8831 / 0.8166 / 0.7511 | 0.0349, 0.0392, 0.0418 | +0.119, 0.0405 | 0.0460 | n=10: +0.53% / -0.33% | 168.4-153.9 | - |
| 3 | classical_so | 1 | 4 / 6 / 8 / 10 | 0.9604 / 0.9165 / 0.8659 / 0.8129 | 0.0234, 0.0284, 0.0316 | +0.094, 0.0300 | 0.0373 | n=10: +0.64% / -0.36% | 222.1-189.6 | - |
| 3 | r1_sf | 2 | 4 / 6 / 8 | 0.9815 / 0.947 / 0.9002 | 0.0179, 0.0254 | +0.098, 0.0254 | 0.0328 | - | 276.3-215.1 | - |
| 3 | r1_so | 2 | 4 / 6 / 8 / 10 | 0.9807 / 0.9409 / 0.8818 / 0.8063 | 0.0207, 0.0324, 0.0447 | +0.175, 0.0386 | 0.0571 | n=10: +2.49% / +0.13% | 159.6-127.3 | - |
| 3.5 | hf | 0 | 4 / 6 / 8 / 10 | 0.5535 / 0.3961 / 0.2805 / 0.1973 | 0.1673, 0.1726, 0.1760 | +0.121, 0.1743 | 0.1824 | n=10: +0.69% / -0.37% | 40.0-39.0 | sf 0.181-0.17; so 0.174-0.154 |
| 3.5 | classical_sf | 1 | 4 / 6 / 8 / 10 | 0.9178 / 0.8586 / 0.7897 / 0.7203 | 0.0333, 0.0419, 0.0460 | +0.113, 0.0439 | 0.0499 | n=10: +0.83% / -0.88% | 153.1-141.9 | - |
| 3.5 | classical_so | 1 | 4 / 6 / 8 / 10 | 0.8722 / 0.8135 / 0.7485 / 0.6832 | 0.0348, 0.0416, 0.0456 | +0.057, 0.0436 | 0.0514 | n=10: +0.80% / -0.56% | 153.1-137.1 | - |
| 3.5 | r1_sf | 2 | 4 / 6 / 8 | 0.9688 / 0.917 / 0.8482 | 0.0275, 0.0390 | +0.147, 0.0390 | 0.0505 | - | 180.9-141.4 | - |
| 3.5 | r1_so | 2 | 4 / 6 / 8 / 10 | 0.9637 / 0.9134 / 0.8462 / 0.7628 | 0.0268, 0.0382, 0.0519 | +0.184, 0.0450 | 0.0656 | n=10: +2.77% / +0.45% | 137.9-111.2 | - |
| 4 | hf | 0 | 4 / 6 / 8 / 10 | 0.4341 / 0.272 / 0.1685 / 0.1036 | 0.2337, 0.2395, 0.2431 | +0.147, 0.2413 | 0.2493 | n=10: +0.73% / -0.42% | 29.1-28.6 | sf 0.171-0.16; so 0.0247-0.0232 |
| 4 | classical_sf | 1 | 4 / 6 / 8 / 10 | 0.7001 / 0.6571 / 0.593 / 0.5269 | 0.0317, 0.0513, 0.0591 | -0.086, 0.0552 | 0.0641 | n=10: +1.56% / -2.36% | 116.1-107.7 | - |
| 4 | classical_so | 1 | 4 / 6 / 8 / 10 | 0.4995 / 0.4047 / 0.3209 / 0.2516 | 0.1051, 0.1160, 0.1217 | -0.190, 0.1188 | 0.1280 | n=10: +1.14% / -1.03% | 55.4-53.2 | - |
| 4 | r1_sf | 2 | 4 / 6 / 8 | 0.933 / 0.8622 / 0.7794 | 0.0394, 0.0505 | +0.155, 0.0505 | 0.0616 | - | 139.8-116.0 | - |
| 4 | r1_so | 2 | 4 / 6 / 8 / 10 | 0.9457 / 0.8772 / 0.7998 / 0.7146 | 0.0376, 0.0462, 0.0563 | +0.180, 0.0512 | 0.0665 | n=10: +2.05% / +0.32% | 126.7-108.9 | - |

**U2, gap and windows** (F21; m_E = m_B = 3.00 mHa, m_BL = 12.00 mHa at sin θ₀ = 1, E₁ − E1lb = 1 mHa and
F40's verification run). Δ₁ = An^−γ through the two largest sizes, A′ = 10·Δ₁(10), n_cert(m) = [A′/m,
(A/m)^(1/γ)], a model bracket.

| R | Delta_1 at n (mHa) | n Delta_1 (mHa) | A (mHa), gamma | A' (mHa) | n_cert(m_E) | n_cert(m_B) | n_cert(m_BL) |
|---|---|---|---|---|---|---|---|
| 2.5 | 266.28 / 200.04 / 159.27 / 132.09 | 1065 / 1200 / 1274 / 1321 | 911, 0.839 | 1321 | 440-912 | 440-912 | 110-175 |
| 3 | 138.96 / 104.67 / 83.32 / 69.04 | 556 / 628 / 667 / 690 | 480, 0.842 | 690 | 230-414 | 230-414 | 58-80 |
| 3.5 | 65.26 / 49.26 / 39.22 / 32.50 | 261 / 296 / 314 / 325 | 226, 0.843 | 325 | 108-169 | 108-169 | 27-33 |
| 4 | 28.24 / 21.33 / 16.98 / 14.07 | 113 / 128 / 136 / 141 | 98, 0.844 | 141 | 47-62 | 47-62 | 12-12 |

**U2, one run of B_L** (F21, F40): the one-run window solves Δ₁(n) ≥ ΔE(1 + 2D_L(p(n)) sin θ₀) + (E₁ − E1lb)
jointly with F22's p(n) per gap model and rate (last / Aitken); p at the two ends of the two-run window; and
that window restricted to p ≥ 3.37e-5, the overlap down to which the two-run B_L is sound (the classical_so
rows are in `scaling_summary.md`).

| R | state | one run, power (last / Aitken) | one run, A'/n (last / Aitken) | one-run n_cert(m_BL(p)) | p at A'/m_BL | p at (A/m_BL)^(1/gamma) | two-run n_cert(m_BL) with p >= floor |
|---|---|---|---|---|---|---|---|
| 2.5 | hf | 76.4 / 70.5 | 66.0 / 60.5 | 61-76 | 0.00258 / 0.00124 | 7.59e-05 / 2.27e-05 | 110-175 (capped at power, aitken: 168) |
| 2.5 | classical_sf | 108.4 / 95.2 | 91.3 / 82.6 | 83-108 | 0.0551 / 0.0225 | 0.00963 / 0.00221 | 110-175 |
| 3 | hf | 37.5 / 35.8 | 34.5 / 33.0 | 33-38 | 0.00223 / 0.00159 | 0.0002 / 0.000122 | 58-80 |
| 3 | classical_sf | 60.3 / 56.8 | 49.3 / 49.3 | 49-60 | 0.103 / 0.0842 | 0.0405 / 0.0301 | 58-80 |
| 3.5 | hf | 19.3 / 19.0 | 18.1 / 18.1 | 18-19 | 0.00976 / 0.00876 | 0.00369 / 0.0032 | 27-33 |
| 3.5 | classical_sf | 32.6 / 32.6 | 30.6 / 29.0 | 29-33 | 0.328 / 0.307 | 0.255 / 0.233 | 27-33 |
| 4 | hf | 10.1 / 10.1 | 10.0 / 10.0 | 10-10 | 0.0682 / 0.0675 | 0.0626 / 0.0618 | 12-12 |
| 4 | classical_sf | 15.0 / 15.0 | 14.1 / 14.1 | 14-15 | 0.476 / 0.472 | 0.466 / 0.461 | 12-12 |

- **HF and holdouts.** The successive rates rise with n at every R, so the per-site decay is not yet geometric
  at n = 10. The n = 10 holdout from n ≤ 8 errs by +0.37% to +0.73% (last rate) and −0.09% to −0.42% (Aitken),
  so the bracket contains p_HF(10) at every R, and the rates reproduce the design's α^(6) = 0.1007 and α^(8) =
  0.1049 at R = 3.0 bohr. For r1_so at R ≥ 3.0 bohr both ends overestimate p(10) (+2.05% to +2.77% and +0.13%
  to +0.45%; aitken_fallback), so those brackets likely overestimate p₂. Without the n = 12 holdout (§13) the
  model, F22's per-site decay ([derived] for matrix-product states, an [assumption] for HF and the iterates),
  is validated over Δn = 2, while n(p = 1e-3) lies 18.6–431.5 sites beyond n = 10 [derived].
- **The free classical step and the first measured iteration.** At 4.0 bohr the classical step overshoots
  (ε₂/ε* = 1.872–1.968 so and 1.598–1.725 sf, against 1.060–1.093 at 2.5 bohr [derived from the records of
  `scaling_<s>.json`]; spin-orbital p₁ = 0.4995 on h4_r4.0 where the exact line search reaches 0.9412), and on
  N2 it undershoots more as R grows (0.978–0.983 at 1.098 Å, 0.647–0.707 at 2.8 Å, both active spaces; M3 T7:
  0.9785–0.9805 on n2, 0.7619–0.7955 on n2s). The spin-orbital R1 Ψ₂ decays faster than its Ψ₁ at R ≤ 3.5 bohr
  (last rates 0.0202, 0.0447, 0.0519 against 0.0199, 0.0316, 0.0456), so G₁ = p₂/p₁ peaks at n = 6–8 and falls
  by n = 10 (1.021, 1.027, 1.018 and 0.9919 at n = 4–10 and R = 3.0, U4: at n = 10 the unit step lowers the
  overlap) and F45 finds no spin-orbital n* there (§4); at 4.0 bohr G₁ grows (1.894 to 2.84): the iteration
  repairs the overshoot.
- **Gap and windows.** nΔ₁ rises over n = 4–10 (1065 to 1321 mHa at 2.5 bohr, 113 to 141 at 4.0), so A′/n is
  the faster-closing model and the asymptote of the stretched chain in its Heisenberg limit (gap ∝ 1/n; F21
  [cited]); that R = 2.5–4.0 bohr is already in that regime is an [assumption], not checked. The windows
  reproduce the design's power-law n_cert(3ΔE) = 62 / 169 / 414 / 912; m_BL = 12 mHa makes n_cert(m_BL) 5–6%
  shorter than at the design's 11.4 mHa (§13).
- **Where the HF hand-off ends.** Rule E's window is that of a state at E₀ and outlasts the overlap. Against
  the two-run B_L window p_HF = 1e-3 falls inside only under the power-law gap at 2.5 bohr (end 175, capped at
  168 where p_HF crosses the floor 3.37e-5) and 3.0 bohr (80); one B_L run closes earlier still (HF: 61–76,
  33–38, 18–19, 10). Under kernel-aware rule B the gap, not the overlap, ends the HF hand-off on the stretched
  chains.
- **N2 at fixed N.** p_HF / classical so Ψ₁ / R1 so Ψ₂ are 0.926 / 0.990 / 0.997 at 1.098 Å, 0.657 / 0.829 /
  0.877 at 1.6, 0.253 / 0.396 / 0.471 at 2.0, 0.112 / 0.213 / 0.282 at 2.4 and 0.078 / 0.165 / 0.230 at 2.8 Å
  (U0, Q1). The spin-free classical step lowers p at 2.0–2.8 Å (to 0.2433, 0.09651, 0.06531; CAS(10,8) 0.2363,
  0.09508, 0.06484), the sign M3 found on n2s (M3 T6), now at 2.4–2.8 Å and in CAS(10,8) too.
- **Compiled states.** U2's overlaps are those of the exact unitaries e^{εD}. At s = 1 the elementwise
  classical step errs by ‖Ψ̃₁ − Ψ₁‖ = 0.0091–0.0182 at 2.5 bohr, 0.0397–0.0748 at 3.0, 0.159–0.278 at 3.5 and
  0.488–0.805 at 4.0 (0.00167–0.0241 on N2), with ⟨S²⟩ up to 0.875 (`scaling_<s>.json`, "compiled"); F25's
  slice rule (η_s = 0.05) takes s = 1–2 at k = 1 for R ≤ 3.0 bohr and on N2, 4–8 at 3.5 and 16 at 4.0 bohr,
  and misses η_s even at s = 16 with the spin-orbital step on h6_r4.0 and h8_r4.0 (slice_cap; step errors
  0.0507 and 0.0559; the spin-free step reaches it). The stretched-chain overlaps of U2 and U4 are realizable
  only with s-fold slicing at s times the step's Toffolis, which U3's C(Ψ_k) do not include.

## 3. Cost components (U3)

The cheapest measured iteration is R1's first unit step after the free classical step; its cost is the
residual at Ψ₁ alone (F30). At r = 1 it costs T_res(1)/C_QPE = 0.0184 (h4_r4.0, so) to 519 (h10_r2.5, so), and
since direct_mg is the cheapest route on every N ≤ 8 row at every r (at N = 10 only the direct route is
costed), it scales as r⁻² (0.205–5.77e3 at r = 0.3, 0.00205–57.7 at r = 3). On the chains it grows with local
exponents 8.39–9.55 in N (N = 6 → 8 → 10), mostly through V_m (8.02–8.29) against 2.12–2.43 for C_QPE: above 1
C_QPE from N = 6 on every chain except h6_r4.0 so, 9.65–519 C_QPE at N = 8–10. On N2 it stays at 0.336–0.79
C_QPE (CAS(6,6)) and 1.5–3.69 (CAS(10,8)). Without the line search the iteration at Ψ₁ is two orders of
magnitude cheaper than M3's at r = 0.3: 3.77 against 463 C_QPE on n2, 20.1 against 3.34e3 on h6s, 4.26 against
380 on n2s (spin-free; M3 T3, k = 1).

**U3** (cost components; C_QPE, λ and C_W are U1's; `results/m4/scaling_summary.md`, U3, gives every r). K
allowed spin-free elements; V_m, W_m in Ha² (F10); C(Ψ_k) in Toffolis; ‖A₁‖\_m in Ha. C(Ψ₁) is the classical
step's elementwise circuit (s = 1), C(Ψ₂) the R1 Ψ₂ of each metric (at N = 10 sf the exact-line-search Ψ₂ of
the classical-step record). The last column is at r = 1, every entry direct_mg (× 1/0.09 at r = 0.3, × 1/9 at
r = 3).

| system | N | K | V_m (method) | W_m | C(Ψ₁) | C(Ψ₂) sf / so (record) | ‖A₁‖\_m sf / so | T_res(1)/C_QPE, r = 1: sf / so |
|---|---|---|---|---|---|---|---|---|
| h4_r2.5 | 4 | 28 | 607.5 (exact) | 817.7 | 160 | 1352 (r1_sf) / 1352 (r1_so) | 0.6787 / 0.3805 | 0.0639 / 0.203 |
| h6_r2.5 | 6 | 153 | 1.696e+04 (rb) | 1.118e+04 | 969 | 9451 (r1_sf) / 9495 (r1_so) | 0.8374 / 0.5151 | 2.07 / 5.48 |
| h8_r2.5 | 8 | 496 | 1.817e+05 (rb) | 6.848e+04 | 3528 | 33828 (r1_sf) / 35256 (r1_so) | 0.9555 / 0.6238 | 32.3 / 75.8 |
| h10_r2.5 | 10 | 1225 | 1.103e+06 (subset) | 2.775e+05 | 9174 | 93079 (lbfgs/second_order) / 93031 (r1_so) | 1.0568 / 0.7218 | 242 / 519 |
| h4_r3.0 | 4 | 28 | 478.1 (exact) | 818.2 | 160 | 1352 (r1_sf) / 1352 (r1_so) | 0.6867 / 0.4618 | 0.0535 / 0.118 |
| h6_r3.0 | 6 | 153 | 1.459e+04 (rb) | 1.118e+04 | 969 | 9495 (r1_sf) / 9495 (r1_so) | 0.8963 / 0.6655 | 1.81 / 3.28 |
| h8_r3.0 | 8 | 496 | 1.583e+05 (rb) | 6.848e+04 | 3528 | 35256 (r1_sf) / 33874 (r1_so) | 1.0523 / 0.8525 | 27 / 41.2 |
| h10_r3.0 | 10 | 1225 | 9.621e+05 (subset) | 2.775e+05 | 9174 | 93079 (lbfgs/second_order) / 92979 (r1_so) | 1.1848 / 1.0312 | 209 / 276 |
| h4_r3.5 | 4 | 28 | 431.4 (exact) | 817.9 | 160 | 1352 (r1_sf) / 1352 (r1_so) | 0.6114 / 0.6589 | 0.064 / 0.0551 |
| h6_r3.5 | 6 | 153 | 1.362e+04 (rb) | 1.118e+04 | 969 | 9451 (r1_sf) / 9495 (r1_so) | 0.8463 / 0.9004 | 2.12 / 1.88 |
| h8_r3.5 | 8 | 496 | 1.471e+05 (rb) | 6.848e+04 | 3528 | 35256 (r1_sf) / 35256 (r1_so) | 1.0476 / 1.1572 | 28.6 / 23.5 |
| h10_r3.5 | 10 | 1225 | 8.882e+05 (subset) | 2.775e+05 | 9174 | 93079 (lbfgs/second_order) / 92981 (r1_so) | 1.2262 / 1.4184 | 204 / 152 |
| h4_r4.0 | 4 | 28 | 420.5 (exact) | 817.3 | 160 | 1352 (r1_sf) / 1352 (r1_so) | 0.8819 / 1.1419 | 0.0309 / 0.0184 |
| h6_r4.0 | 6 | 153 | 1.33e+04 (rb) | 1.118e+04 | 969 | 9495 (r1_sf) / 9495 (r1_so) | 1.0212 / 1.4960 | 1.55 / 0.724 |
| h8_r4.0 | 8 | 496 | 1.418e+05 (rb) | 6.848e+04 | 3528 | 35256 (r1_sf) / 35256 (r1_so) | 1.2011 / 1.8544 | 23 / 9.65 |
| h10_r4.0 | 10 | 1225 | 8.491e+05 (subset) | 2.775e+05 | 9174 | 93031 (lbfgs/second_order) / 93079 (r1_so) | 1.3928 / 2.2155 | 166 / 65.7 |
| n2_r1.098 | 6 | 33 | 3751 (exact) | 2118 | 486 | 1531 (r1_sf) / 1531 (r1_so) | 0.6520 / 0.4274 | 0.339 / 0.79 |
| n2_r1.6 | 6 | 33 | 2330 (exact) | 2116 | 486 | 1531 (r1_sf) / 1531 (r1_so) | 0.5712 / 0.4292 | 0.336 / 0.595 |
| n2_r2.0 | 6 | 33 | 2849 (exact) | 2114 | 486 | 1379 (r1_sf) / 1455 (r1_so) | 0.6166 / 0.5323 | 0.384 / 0.515 |
| n2_r2.4 | 6 | 33 | 3434 (exact) | 2113 | 486 | 1493 (r1_sf) / 1455 (r1_so) | 0.7169 / 0.6566 | 0.355 / 0.423 |
| n2_r2.8 | 6 | 33 | 3923 (exact) | 2112 | 486 | 1417 (r1_sf) / 1417 (r1_so) | 0.8030 / 0.7530 | 0.35 / 0.398 |
| n2v_r1.098 | 8 | 128 | 5.672e+04 (rb) | 1.657e+04 | 969 | 7087 (r1_sf) / 7129 (r1_so) | 0.8937 / 0.6895 | 1.5 / 2.51 |
| n2v_r2.0 | 8 | 128 | 5.366e+04 (rb) | 1.657e+04 | 969 | 7045 (r1_sf) / 7129 (r1_so) | 0.7171 / 0.6243 | 2.79 / 3.69 |
| n2v_r2.4 | 8 | 128 | 6.07e+04 (rb) | 1.657e+04 | 969 | 7045 (r1_sf) / 7129 (r1_so) | 0.7612 / 0.6808 | 2.93 / 3.66 |
| n2v_r2.8 | 8 | 128 | 6.732e+04 (rb) | 1.657e+04 | 969 | 7003 (r1_sf) / 7129 (r1_so) | 0.8238 / 0.7490 | 2.87 / 3.47 |

**U3, successive exponents in N** (chains, N = 6 → 8 and 8 → 10 at fixed R; N = 4 is SF, N ≥ 6 THC, so N = 4
is not used; the exponents of T_res(1)/C_QPE are the same at every r).

| quantity | R = 2.5 | R = 3.0 | R = 3.5 | R = 4.0 |
|---|---|---|---|---|
| C_QPE | 2.27, 2.43 | 2.26, 2.14 | 2.23, 2.13 | 2.22, 2.12 |
| λ | 1.32, 1.30 | 1.31, 1.28 | 1.28, 1.27 | 1.27, 1.26 |
| C_W | 0.95, 1.13 | 0.95, 0.86 | 0.95, 0.86 | 0.95, 0.86 |
| K | 4.09, 4.05 | 4.09, 4.05 | 4.09, 4.05 | 4.09, 4.05 |
| V_m | 8.24, 8.08 | 8.29, 8.09 | 8.27, 8.06 | 8.23, 8.02 |
| W_m | 6.30, 6.27 | 6.30, 6.27 | 6.30, 6.27 | 6.30, 6.27 |
| C(Ψ₁) | 4.49, 4.28 | 4.49, 4.28 | 4.49, 4.28 | 4.49, 4.28 |
| C(Ψ₂) sf / so | 4.43, 4.54 / 4.56, 4.35 | 4.56, 4.35 / 4.42, 4.53 | 4.58, 4.35 / 4.56, 4.35 | 4.56, 4.35 / 4.56, 4.35 |
| ‖A₁‖\_m sf / so | 0.46, 0.45 / 0.67, 0.65 | 0.56, 0.53 / 0.86, 0.85 | 0.74, 0.71 / 0.87, 0.91 | 0.56, 0.66 / 0.75, 0.80 |
| T_res(1)/C_QPE sf / so | 9.55, 9.03 / 9.13, 8.62 | 9.40, 9.17 / 8.80, 8.53 | 9.04, 8.80 / 8.78, 8.39 | 9.37, 8.86 / 9.00, 8.59 |

- **Route.** With direct_mg best at both sizes, T_res(1)/C_QPE = V_mC(Ψ₁)/(r²‖A₁‖²_mC_QPE) (F10, F23), and the
  exponents add up to the printed digits (8.09 + 4.28 − 2·0.53 − 2.14 = 9.17 at R = 3.0, sf, N = 8 → 10
  [derived]). Every other route is far behind at k = 1 (h8_r3.0 sf, r = 1: direct_mg 5.04e8 Toffolis, aux_mg
  3.12e9, ae_x 1.12e11; `cheapest.sf.by_r` of `scaling_h8_r3.0.json`); over the N ≤ 8 rows the next route
  costs at least 5.71 times direct_mg at r = 1, 5.44 times at r = 3 and 1.89 times at r = 0.3 [derived from
  `cheapest.<metric>.by_r`]. This is M3's large-residual regime, where matchgate shadows win (M3 T3). At
  N = 10 only the direct route is costed.
- **What grows.** K as N^4.05–4.09, V_m as N^8.02–8.29, W_m as N^6.27–6.30, the circuits as N^4.28–4.58,
  ‖A₁‖\_m weakly and C_QPE as N^2.12–2.43. The matchgate-shadow factor of a three-body term on 2N modes tends
  to 64N³/15, which with β(K) ≈ 4.1 accounts for β ≈ 7 of the observed ≈ 8 [estimate, F23]. These are local
  exponents over N = 6–10, and F45 extrapolates them as such (§4).
- **Stretching and N2.** The spin-orbital ‖A₁‖\_m grows strongly with R (h10: 0.7218 to 2.2155) because the
  overshooting classical step leaves a larger residual (§2), so T_res(1)/C_QPE falls from 519 to 65.7: the
  cheapest stretched-chain iterations follow a poor free step. CAS(6,6) allows K = 33 elements with exact V_m
  and C(Ψ₁) = 486; CAS(10,8) has K = 128 and a C_QPE 3.6–4.0 times larger [derived from U1].
- **Variance states (model dependence, F24).** At N ≤ 6 the variances are M3's at Ψ₁, at N = 8 at HF, at N =
  10 from a 100-element subset at HF whose sampling error `results/m4` does not record (F24: 2.2% on
  h10_r3.0), so the N = 6 → 8 exponent of V_m compares two states.
- **Reference numbers and M3.** The design's cheapest-iteration values are reproduced for h6_r3.0, h8_r3.0 and
  h10_r3.0 sf (1.8, 27, 209). On N2 so the run gives 0.515, 0.423 and 0.398 at 2.0, 2.4 and 2.8 Å with M3's
  iterate-1 variances, the values of the design's own closed-loop check, against 0.43 and 0.41 at 2.4 and
  2.8 Å in its reference table, which used HF-state variances (§11, §13). M3's iteration k = 1 at r = 0.3 (at
  its Ψ₁, after the exact first line search; its first measured iteration is k = 0 at HF) was dominated by its
  own line search (C_iter(1) = 463, 3.34e3 and 380 C_QPE on n2, h6s and n2s; M3 T3); U3 gives 3.77, 20.1 and
  4.26 (sf; 8.78, 36.4, 5.72 so), the residual-only floors M3 anticipated (M3 caveat 3: at least 3.6 C_QPE on
  n2), and 0.339, 1.81 and 0.384 at r = 1.

## 4. The break-even map (U4)

The verdicts of this section are for exact trajectories (noise-free unit steps, r entering only the cost),
with p on ψ₀ of H; the closed loop decides under noise (§6), and U9 part 1 repeats the hand-off systems with p
on ψ₀′ of H′ (§8; the printed savings coincide on the chains and differ slightly on N2, e.g. n2_r2.4 so at r =
3: 0.0799% here, 0.0997% there).

The free classical step pays at the best realizable scheme on 40 of the 46 system-metric rows (0.242–49.7%),
not with the spin-free metric on N2 at 2.0–2.8 Å, where it lowers the overlap. The first measured iteration
(classical Ψ₁ → unit Ψ₂) pays at s* on 2, 5 and 12 of the 46 rows at r = 0.3, 1 and 3; at r = 1 on h4_r3.5
(sf, so), h4_r4.0 (sf, so) and h6_r4.0 (so), short chains at R ≥ 3.5 bohr with Δ₁ ≤ 65.26 mHa, saving
1.96–30.8%. That refutes the design's expectation that no measured iteration pays on the chains at N ≤ 10
under any scheme: under repetition on four chain rows (h4_r3.5 so, h4_r4.0 sf and so, h6_r4.0 so;
ρ₁ = 0.0195–0.545) and at s* on five. The refutation is fragile in its mechanism rather than small in size:
h4_r3.5 sf pays only through BBHT's overshoot at p ≈ 0.9 (h4_r3.5 so, on the same certified hand-off, also
pays under repetition), h4_r4.0 and h6_r4.0 pay by repetition with B_L, which has no certificate and whose
termination is not certified, and with the slicing their compiled circuits need h4_r3.5 sf, h4_r4.0 sf and
h6_r4.0 so would no longer pay, while h4_r3.5 so and h4_r4.0 so would [derived below]. On N2 CAS(6,6) at
2.4–2.8 Å the iteration pays under repetition at r = 1 (ρ₁ = 0.234–0.625) but loses 16.0–25.0% at s*
(lt_bbht_L); at s* it never pays on CAS(10,8) or at N2's equilibrium, and the second measured iteration never
pays at s*. Extrapolated along the chains (F45, r = 1), break-even needs n* ≥ 86 under repetition
(p₁(n*) ≤ 1.5e-5) and n* ≥ 166 at the best realizable scheme (p₁(n*) ≤ 5.4e-10), outside the kernel-aware
window in every case, and none exists with the spin-orbital metric at R ≤ 3.5 bohr.

**U4, first measured iteration** (k = 1, r = 1; the N ≤ 6 chains, the R = 4.0 bohr chains and N2 CAS(6,6);
`results/m4/scaling_summary.md`, U4, gives every system, k ≤ 2 and r under repetition with p₁* and the
optimistic ρ_LS, and `results/m4/summary.md`, "U4: scheme columns", every scheme; the N = 8, 10 rows at R ≤
3.5 bohr have ρ* = 154–1.83e4, and inf on h10_r3.0 so, where G₁ < 1; CAS(10,8) ρ* = 45.2–2.73e3). Units C_QPE.
p₁ is the classical Ψ₁ on ψ₀ of H, G₁ = p₂/p₁, c_iter(1) = T_res(1)/C_QPE (F30); μ₁ = (1 + c₁)/p₁ − (1 +
c₂)/p₂ and ρ₁ = c_iter(1)/μ₁ are the repetition values (F44), and the iteration pays iff ρ < 1. s* is the best
realizable scheme at Ψ₁ and at Ψ₂ (F43, F44), μ* = T_s*(1) − T_s*(2) + c_iter(1) with rule B's verification
runs in rep, bbht and fp, and the saving at s* is 1 − T_s*(2)/T_s*(1) (`summary.json`,
`U4_schemes.rows[].best.relative`). s* carries termination_not_certified on every lt_bbht_L row, and
no_certificate with termination_not_certified on the rep and bbht rows of h8_r3.0, h10_r3.0 and every chain at
R ≥ 3.5 bohr except h4_r3.5.

| system | metric | p₁ | G₁ | c_iter(1) | μ₁ / ρ₁ (rep) | s* | μ* / ρ* | saving at s* (%) |
|---|---|---|---|---|---|---|---|---|
| h4_r2.5 | sf | 0.9589 | 1.032 | 0.0639 | 0.0322 / 1.99 | rep -> rep | 0.0322 / 1.99 | −3.04 |
| h4_r2.5 | so | 0.9791 | 1.013 | 0.203 | 0.0129 / 15.8 | rep -> rep | 0.0129 / 15.8 | −18.7 |
| h6_r2.5 | sf | 0.9139 | 1.058 | 2.07 | 0.0588 / 35.3 | lt_bbht -> rep | 0.0564 / 36.8 | −185 |
| h6_r2.5 | so | 0.9501 | 1.023 | 5.48 | 0.0232 / 236 | rep -> rep | 0.0232 / 236 | −518 |
| h4_r3.0 | sf | 0.947 | 1.036 | 0.0535 | 0.0367 / 1.46 | rep -> rep | 0.0367 / 1.46 | −1.59 |
| h4_r3.0 | so | 0.9604 | 1.021 | 0.118 | 0.0212 / 5.58 | rep -> rep | 0.0212 / 5.58 | −9.33 |
| h6_r3.0 | sf | 0.8831 | 1.072 | 1.81 | 0.0755 / 23.9 | lt_bbht_L -> lt_bbht_L | 0.04 / 45.2 | −149 |
| h6_r3.0 | so | 0.9165 | 1.027 | 3.28 | 0.0274 / 120 | lt_bbht_L -> lt_bbht_L | 0.0125 / 262 | −281 |
| h4_r3.5 | sf | 0.9178 | 1.056 | 0.064 | 0.0569 / 1.13 | lt_bbht -> lt_bbht | 0.0965 / 0.664 | 1.96 |
| h4_r3.5 | so | 0.8722 | 1.105 | 0.0551 | 0.108 / 0.509 | lt_bbht -> lt_bbht | 0.229 / 0.241 | 9.7 |
| h6_r3.5 | sf | 0.8586 | 1.068 | 2.12 | 0.0731 / 29.1 | rep -> rep | 0.0731 / 29.1 | −94.7 |
| h6_r3.5 | so | 0.8135 | 1.123 | 1.88 | 0.133 / 14.1 | rep -> rep | 0.133 / 14.1 | −78.1 |
| h4_r4.0 | sf | 0.7001 | 1.333 | 0.0309 | 0.356 / 0.0867 | rep -> rep | 0.356 / 0.0867 | 13.4 |
| h4_r4.0 | so | 0.4995 | 1.894 | 0.0184 | 0.944 / 0.0195 | rep -> rep | 0.944 / 0.0195 | 30.8 |
| h6_r4.0 | sf | 0.6571 | 1.312 | 1.55 | 0.361 / 4.31 | rep -> rep | 0.361 / 4.31 | −47.3 |
| h6_r4.0 | so | 0.4047 | 2.167 | 0.724 | 1.33 / 0.545 | rep -> rep | 1.33 / 0.545 | 17.4 |
| h8_r4.0 | sf | 0.593 | 1.314 | 23 | 0.401 / 57.4 | rep -> rep | 0.401 / 57.4 | −841 |
| h8_r4.0 | so | 0.3209 | 2.492 | 9.65 | 1.86 / 5.18 | rep -> rep | 1.86 / 5.18 | −189 |
| h10_r4.0 | so | 0.2516 | 2.84 | 65.7 | 2.57 / 25.6 | bbht -> rep | 2.12 / 31 | −1.4e3 |
| n2_r1.098 | sf | 0.9636 | 1.012 | 0.339 | 0.0127 / 26.7 | lt_bbht -> lt_bbht | 0.000516 / 658 | −33.2 |
| n2_r1.098 | so | 0.9901 | 1.007 | 0.79 | 0.00724 / 109 | rep -> rep | 0.00724 / 109 | −77.5 |
| n2_r1.6 | sf | 0.7057 | 1.038 | 0.336 | 0.0511 / 6.57 | 2s -> 2s | 0.0071 / 47.3 | −27.4 |
| n2_r1.6 | so | 0.8289 | 1.058 | 0.595 | 0.066 / 9.02 | lt_bbht -> lt_bbht | 0.0298 / 20 | −48.8 |
| n2_r2.0 | sf | 0.2433 | 1.06 | 0.384 | 0.232 / 1.65 | lt_bbht_L -> lt_bbht_L | 0.00937 / 40.9 | −25.4 |
| n2_r2.0 | so | 0.3961 | 1.19 | 0.515 | 0.403 / 1.28 | lt_bbht_L -> lt_bbht_L | 0.00351 / 147 | −35.9 |
| n2_r2.4 | sf | 0.09651 | 1.058 | 0.355 | 0.567 / 0.625 | lt_bbht_L -> lt_bbht_L | 0.0258 / 13.7 | −18.5 |
| n2_r2.4 | so | 0.2131 | 1.322 | 0.423 | 1.14 / 0.37 | lt_bbht_L -> lt_bbht_L | 0.0482 / 8.78 | −25 |
| n2_r2.8 | sf | 0.06531 | 1.051 | 0.35 | 0.738 / 0.475 | lt_bbht_L -> lt_bbht_L | 0.0306 / 11.5 | −16 |
| n2_r2.8 | so | 0.1653 | 1.393 | 0.398 | 1.7 / 0.234 | lt_bbht_L -> lt_bbht_L | 0.0808 / 4.93 | −20.3 |

- **The free classical step (k = 0)** costs no iteration and pays wherever it raises the overlap: on the
  chains it saves 4.84–16.1% at 2.5 bohr to 27.2–49.7% at 3.5 and 9.12–48.4% at 4.0, on N2 CAS(6,6) (so)
  1.23–16.7% (`U4_schemes`, k = 0); the spin-free step lowers p on N2 at 2.0–2.8 Å (G₀ = 0.8295–0.9624) and
  loses 0.457–5.98%, as M3 found on n2s.
- **Where the first measured iteration pays (r = 1).** On h4_r3.5, h4_r4.0 and h6_r4.0,
  c_iter(1) = 0.0184–0.724 lies below μ* = 0.0965–1.33: a four- or six-orbital residual is cheap (U3), and
  p₁ = 0.4047–0.9178 leaves a large gain (G₁ = 1.056–2.167). On h4_r4.0 and h6_r4.0 the gain mostly repairs
  the overshooting free step (§2; the exact line search reaches 0.9412 and 0.8672); no damped or otherwise
  improved free first step was run, so on these rows the run does not show that a measured iteration beats the
  best free state. With N the cost wins: at the same R, h8_r4.0 so has ρ* = 5.18 and h10_r4.0 so ρ* = 31. At
  r = 0.3 it pays only on h4_r4.0 (0.538% sf, 24.6% so); at r = 3 also on h4_r2.5 sf, h4_r3.0, h6_r4.0 sf,
  h8_r4.0 so (19.2%) and N2 so at 2.4 and 2.8 Å, rows that pair the noise-free trajectory with a residual at
  300% relative error, which only the closed loop can test (§6). With the exact-line-search Ψ₂ at zero step
  cost (ρ_LS) the r = 1 verdicts change only on N2 at 2.0 Å (ρ_LS = 0.868 sf and 0.745 so against ρ₁ = 1.65
  and 1.28).
- **Scheme dependence.** On h4_r3.5 s* is lt_bbht at both iterates, and above p = 0.30 BBHT's expected calls
  exceed 1/p (F43): at p₁ = 0.8722–0.9178 the schedule overpays and a better state recovers part of the excess
  (μ* = 0.0965 and 0.229 against μ₁ = 0.0569 and 0.108), which the lt_rep sensitivity removes on sf (−2.23%)
  and shrinks to 0.0452% on so, where the iteration also pays under repetition (ρ₁ = 0.509). On h4_r4.0 and
  h6_r4.0 no certified U exists and no E1lb-referenced form terminates (§1), so s* is repetition with B_L's
  two runs; on the ten chains with E1lb − E₀′ < w_L, s* is rep, bbht, lt_bbht or 2s at every k and r, and fp
  is never selected.
- **Compiled states.** The s = 1 circuits that U3 costs miss Ψ₁ by 0.488–0.783 on h4_r4.0 and h6_r4.0 (§2:
  0.488–0.805 over the 4.0-bohr chains); F25 takes s = 16, at C(Ψ̃₁) = 2560 (h4) and 15,504 (h6) Toffolis
  against 160 and 969, and on h6_r4.0 so even s = 16 leaves a step error of 0.0507 > η_s (slice_cap). For
  direct_mg T_res(1) ∝ C(Ψ₁) (F10), so c_iter(1) would be 16 times larger [derived] (at least that on h6_r4.0
  so): 0.295 on h4_r4.0 so, still below μ₁ = 0.944, but 0.494 on h4_r4.0 sf (μ₁ = 0.356) and at least 11.6 on
  h6_r4.0 so (μ₁ = 1.33), which would no longer pay. On h4_r3.5 F25 takes s = 4 (C(Ψ̃₁) = 640), which would
  make c_iter(1) 0.256 (sf) and 0.221 (so) against μ* = 0.0965 and 0.229 [derived]: the spin-free pay would
  go, the spin-orbital one would remain. In the closed loop, which measures h4_r4.0's first residual at
  r₁ = 0.46, the slices reverse the verdict (§6).
- **N2: the QPE side removes the gain.** At s* = lt_bbht_L the filter costs C_filt(w_L) = 0.133 C_QPE per call
  (§8), so μ* = 0.0258–0.0808 against μ₁ = 0.567–1.7 under repetition at 2.4–2.8 Å (0.00351–0.00937 against
  0.232–0.403 at 2.0 Å). At r = 3 (c_iter(1) = 0.0389–0.0572) the spin-orbital iteration saves 0.0799% at
  2.4 Å and 2.33% at 2.8 Å and loses 3.77% at 2.0 Å, the 0.0799% being smaller than the slack of the filter
  cost (F43: the tabulated degree exceeds the construction by up to 13%); the closed loop decides the design's
  corresponding expectation (§6).
- **Equilibrium, CAS(10,8) and k = 2.** On n2_r1.098 (M3's n2) the cheap step rule narrows M3's gap without
  closing it (ρ₁ = 26.7 and 109, ρ* = 658 and 109, where M3's first measured iteration on n2 cost 250 (sf) and
  259 C_QPE (so) against a saving of at most C_QPE(1/p_HF − 1) = 0.080; M3 T3, T6). CAS(10,8) never pays at s*
  (ρ* = 45.2–2.73e3 at r = 1): its residual costs 1.5–3.69 C_QPE for gains close to CAS(6,6)'s. The second
  measured iteration never pays at s* (best −437%, −39.3% and −4.39% at r = 0.3, 1, 3, all on h4_r4.0 sf):
  where the first pays the second gains nothing (G₂ = 0.9885–1.005), and on stretched N2 (CAS(6,6), 2.0–2.8 Å)
  it gains more (G₂ = 1.227–1.622) at c_iter(2) = 2.88–4.72, 7.5–11.8 times c_iter(1): its residual is
  measured on Ψ₂, whose circuit is 2.84–3.07 times C(Ψ₁), and is smaller, which a residual at fixed relative
  accuracy pays as 1/‖A_k‖²_m (F23; ‖A₁‖²_m/‖A₂‖²_m = 2.65–4.06) [derived from the records of
  `scaling_<s>.json`].

**U4, break-even size: inputs** (F45, first measured iteration, r = 1; `scaling_summary.md`, U4). c_iter(1) is
anchored at n₀ = 10 (sf: the cheapest iteration at the same Ψ₁); α₁ from the classical-step series, α₂ from
the R1 record, each from its own last size; c₁, c₂ = C(Ψ_j)/C_QPE as power laws on N ≥ 6.

| R | metric | c₀ (C_QPE) | β successive (N = 6, 8, 10) | α₁ bracket | α₂ bracket | preparation terms |
|---|---|---|---|---|---|---|
| 2.5 | sf | 242 | 9.55, 9.03 | 0.0270-0.0360 | 0.0158-0.0197 (n ≤ 8) | c_1 0.000245 at 10 (beta 1.85); c_2 0.00156 at 8 (beta 2.16) |
| 2.5 | so | 519 | 9.13, 8.62 | 0.0199-0.0249 | 0.0202-0.0807 | c_1 0.000245 at 10 (beta 1.85); c_2 0.00249 at 10 (beta 1.92) |
| 3 | sf | 209 | 9.40, 9.17 | 0.0418-0.0460 | 0.0254-0.0328 (n ≤ 8) | c_1 0.000305 at 10 (beta 2.14); c_2 0.00189 at 8 (beta 2.30) |
| 3 | so | 276 | 8.80, 8.53 | 0.0316-0.0373 | 0.0447-0.0571 | c_1 0.000305 at 10 (beta 2.14); c_2 0.00309 at 10 (beta 2.39) |
| 3.5 | sf | 204 | 9.04, 8.80 | 0.0460-0.0499 | 0.0390-0.0505 (n ≤ 8) | c_1 0.000345 at 10 (beta 2.15); c_2 0.00214 at 8 (beta 2.34) |
| 3.5 | so | 152 | 8.78, 8.39 | 0.0456-0.0514 | 0.0519-0.0656 | c_1 0.000345 at 10 (beta 2.15); c_2 0.0035 at 10 (beta 2.21) |
| 4 | sf | 166 | 9.37, 8.86 | 0.0591-0.0641 | 0.0505-0.0616 (n ≤ 8) | c_1 0.00038 at 10 (beta 2.17); c_2 0.00234 at 8 (beta 2.34) |
| 4 | so | 65.7 | 9.00, 8.59 | 0.1217-0.1280 | 0.0563-0.0665 | c_1 0.00038 at 10 (beta 2.17); c_2 0.00385 at 10 (beta 2.24) |

**U4, break-even size n*** (r = 1). Ranges over rate (matched corners) × β ∈ {last, mean, 9} × crossover ∈
{none, N² or N⁴ beyond n = 20} × gap model ∈ {power law, A′/n}, 36 cases per window; "(k/36 none)" counts
cases without n* below n = 3000, "[k/36]" those inside the window of their own gap model. "best" is min(rep,
bbht) beyond the computed sizes (it equals bbht in every case), with the window check of the E1lb-referenced
forms (`summary.md`, "U4: break-even size n* under the QPE-side schemes").

| R | metric | n_cert(m_E) / n_cert(m_BL) | rep, m_E | rep, m_BL | p₁(n*), rep, m_BL | best, m_E | best, m_BL | p₁(n*), best, m_BL |
|---|---|---|---|---|---|---|---|---|
| 2.5 | sf | 440-912 / 110-175 | 500-1954 [9/36] | 464-1924 [0/36] | 6.7e-08-2.9e-23 | 964-2974 (8/36 none) [0/36] | 898-2998 (7/36 none) [0/36] | 1.8e-47-1.1e-14 |
| 3 | sf | 230-414 / 58-80 | 366-1126 [5/36] | 340-1112 [0/36] | 1.9e-07-7.4e-21 | 700-2474 [0/36] | 654-2460 [0/36] | 2.5e-45-1e-13 |
| 3.5 | sf | 108-169 / 27-33 | 358-962 (18/36 none) [0/36] | 322-940 (18/36 none) [0/36] | 4.2e-07-1.9e-19 | 718-2122 (18/36 none) [0/36] | 632-2098 (18/36 none) [0/36] | 1.4e-42-2.7e-13 |
| 4 | sf | 47-62 / 12-12 | 250-758 [0/36] | 218-678 [0/36] | 8.5e-07-3.8e-18 | 496-2260 [0/36] | 440-1544 [0/36] | 2.3e-40-4.9e-12 |
| 4 | so | 47-62 / 12-12 | 98-266 [0/36] | 86-250 [0/36] | 1.5e-05-5.2e-14 | 190-624 [0/36] | 166-600 [0/36] | 1.7e-32-5.4e-10 |
| 2.5, 3, 3.5 | so | as above | none (36/36 none) | none (36/36 none) | - | none (36/36 none) | none (36/36 none) | - |

- **Which brackets have an n\*.** Under repetition n* exists for the spin-free metric at every R, but at 3.5
  bohr in only half of the bracket (the 18 of 36 cases at the Aitken corner, where α₂ = 0.0505 exceeds
  α₁ = 0.0499, have none below n = 3000; at the best scheme also 7–8 of 36 at 2.5 bohr), and for the
  spin-orbital one only at 4.0 bohr: at R ≤ 3.5 its R1 Ψ₂ decays faster than the classical Ψ₁ at both ends of
  the bracket (α₂ > α₁, §2), so G₁ = p₂/p₁ falls below 1 at a finite n and μ₁ turns negative before the gain
  can outgrow the polynomial cost (F45; a gain that merely stayed bounded, α₂ = α₁, would still give an n*).
  n* falls with R and with r (sf, m_E, r = 0.3 / 1 / 3: 570–2058 / 500–1954 / 436–1858 at 2.5 bohr, 292–764 /
  250–758 / 212–756 at 4.0).
- **What n\* means.** At n* the overlaps are tiny (p₁(n*) ≤ 1.5e-5 under repetition), so handing off Ψ₁ there
  costs 1/p₁ ≥ 6.7e4 C_QPE [derived]: n* marks where the exponential overlap catastrophe outgrows the
  polynomial residual cost, not an operating point. The preparation terms are not small there (c₂(n*) =
  0.473–226), and the design's form without them shifts the range ends by a few per cent.
- **Schemes and windows.** At the best realizable scheme n* roughly doubles, since under amplitude
  amplification a better state saves in proportion to 1/√p rather than 1/p (F44); the oracle aa gives slightly
  larger n* than bbht. No case lies inside the kernel-aware window. Under repetition 9/36 (2.5 bohr) and 5/36
  (3.0 bohr) spin-free cases lie inside the rule-E window, all under the power-law gap with a crossover beyond
  n = 20, at p₁(n*) = 6.2e-12–4.1e-8 [derived from `scaling_fits.json`]; rule E would not certify those
  hand-off states, whose energies lie far above E₀.
- **Against the design.** The design expected n* ≈ 630–2100 under repetition with the measured exponents and
  250–720 with an N² crossover, against n_cert(3ΔE) ≈ 62–912; the run gives 632–1954, 250–756 (N²) and 350–992
  (N⁴) for the spin-free chains at r = 1 with m_E [derived] and n_cert = 62–912: reproduced. That n* lies
  beyond the window holds against the kernel-aware window everywhere and at the best scheme against every
  window at r ≤ 1, but fails for 14 of the 72 spin-free rule-E cases under repetition at R = 2.5–3.0 bohr. The
  spin-orbital metric, not treated separately by the design, has no n* at R ≤ 3.5 bohr and 98–266 at 4.0.
- **Model dependence.** Every n* lies 76–2988 sites beyond n = 10 [derived], with the overlap model validated
  over Δn = 2 (§2), local cost exponents (N = 6–10) and an [estimate] crossover beyond n = 20; the R1 Ψ₂
  brackets at R ≥ 3.0 bohr overestimate p₂ at the holdout, which favours the spin-orbital n* at 4.0 bohr. The
  model covers H chains in canonical orbitals with elementwise compilation and the full allowed residual.

## 5. Step rules in exact emulation (U5)

In exact emulation the two unit-step rules are the cheapest way to the reference iterate at r = 1: with ties
counted for each rule, R3 (R1 with the planned gate) is cheapest on 25 and the unit step R1, the L-BFGS model
step that measures nothing beyond the residual (F26, F30), on 24 of the 30 resolved system-metric pairs; the
two coincide on 20, R1 alone wins 4, R3 alone 5, M3's grid one pair, and 4 pairs are unresolved (U0, Q3).
Dropping the 11 line-search energies of every iteration cuts C_loop(k*) at r = 1 to 0.0038 (h4_r1.8 sf) to
0.87 (h6_r4.0 so) of the exact line search's on the 26 pairs where both reach k* [derived], for 0–3
(spin-orbital) and 0–6 (spin-free) extra iterations. The gain shrinks with stretching and size, because the
late iterations are residual-dominated, and at M3's r = 0.3 it reverses on the three hardest H6 pairs.
Energies that decide or fit a step cost more than the residual they accompany at k = 1 on 13 of the 17 systems
and near k* on every pair that reaches it (ρ_E = 5.36–569 at r = 1): the energy-precision wall of F28. R3's
planned gate opens the 3-point fit in 222 of 3491 iterations, so R3 is R1 on most pairs.

**U5** (step rules on exact trajectories; `results/m4/loop_summary.md`, U5, has all 204 rows, r = 0.3 and 3
and the precisions at k_ref − 1; records and ledgers in `loop_<s>.json`). 17 systems (h4 and h6 at R = 1.8,
2.5, 3.0, 3.5, 4.0 bohr, h8 at 1.8 and 3.0 bohr, N2 CAS(6,6) at 1.098–2.8 Å), both metrics, from HF with the
classical step at k = 0, 40 steps (30 at N = 8), exact masked residuals and propagation; energies costed
(F28), not drawn. line search and M3 grid are uncapped (F13, F14), R1 and R3 carry F29's caps, R3 is shown for
its r = 1 run. k* is the first iterate within 1.6 mHa of E₀, k_p90 the first with p ≥ 0.9, and k_ref = k*, or
k_p90 where no rule reaches k*. Costs to k_ref + 2 (k = 20 where k* is not reached, ≤ 12 at N = 8), the
residual by the cheapest route at κ = 1 with per-shot variances at iterate 1 (N ≤ 6) or HF (N = 8). σ_A, σ_P3:
precisions of the Armijo pair (z = 2) and the 3-point fit (F27); ρ_E = T_en/T_res of the Armijo pair against
direct_mg (F28). '-': not reached or beyond the costed iterations (on h6_r4.0 sf R3 does not reach k*).

| system | metric | k* (k_p90): line search / M3 grid / R1 / R3 | C_loop(k_ref) / C_QPE, r = 1: line search / M3 grid / R1 / R3 | R1 C_iter(1) / C_QPE, r = 0.3 / 1 / 3 | R1 at k = 1: σ_A, σ_P3 (mHa); ρ_E, r = 1 |
|---|---|---|---|---|---|
| h4_r1.8 | so | 3 (0) / 3 (0) / 3 (0) / 3 (0) | 2.26e+03 / 2.28e+03 / 24 / 24 | 5.27 / 0.474 / 0.0527 | 2.32, 3.09; 104 |
| h4_r1.8 | sf | 3 (0) / 4 (0) / 3 (0) / 3 (0) | 2.33e+03 / 4.6e+03 / 8.82 / 8.82 | 1.41 / 0.127 / 0.0141 | 4.8, 6.38; 97.1 |
| h4_r2.5 | so | 4 (1) / 4 (1) / 4 (1) / 4 (1) | 4.78e+03 / 4.73e+03 / 55.7 / 55.7 | 2.26 / 0.203 / 0.0226 | 4.5, 5.99; 62.3 |
| h4_r2.5 | sf | 5 (1) / 5 (1) / 5 (1) / 5 (1) | 7.75e+03 / 7.93e+03 / 107 / 107 | 0.71 / 0.0639 / 0.0071 | 10.5, 14; 38.7 |
| h4_r3.0 | so | 5 (1) / 5 (1) / 5 (1) / 5 (1) | 6.37e+03 / 6.44e+03 / 114 / 114 | 1.32 / 0.118 / 0.0132 | 7.92, 10.5; 26.8 |
| h4_r3.0 | sf | 5 (1) / 6 (1) / 6 (1) / 6 (1) | 6.33e+03 / 9.44e+03 / 223 / 223 | 0.595 / 0.0535 / 0.00595 | 15.2, 20.2; 16.8 |
| h4_r3.5 | so | 6 (2) / 7 (2) / 7 (2) / 7 (2) | 6.25e+03 / 8.44e+03 / 1.12e+03 / 1.12e+03 | 0.613 / 0.0551 / 0.00613 | 26.2, 34.8; 3.97 |
| h4_r3.5 | sf | 6 (1) / 6 (1) / 8 (1) / 8 (1) | 6.02e+03 / 6.12e+03 / 952 / 952 | 0.712 / 0.064 / 0.00712 | 18.7, 24.8; 6.73 |
| h4_r4.0 | so | 6 (2) / 6 (2) / 6 (2) / 6 (2) | 3.16e+03 / 3.29e+03 / 239 / 239 | 0.205 / 0.0184 / 0.00205 | 104, 139; 0.502 |
| h4_r4.0 | sf | 6 (2) / 7 (2) / 7 (2) / 7 (2) | 3.87e+03 / 5.31e+03 / 321 / 321 | 0.343 / 0.0309 / 0.00343 | 60.6, 80.6; 1.49 |
| h6_r1.8 | so | 4 (0) / 4 (0) / 4 (0) / 4 (0) | 1.87e+04 / 1.96e+04 / 1.17e+03 / 1.17e+03 | 108 / 9.74 / 1.08 | 3.57, 4.74; 7.51 |
| h6_r1.8 | sf | 6 (0) / 6 (0) / 6 (0) / 6 (0) | 4.78e+04 / 4.77e+04 / 2.75e+03 / 2.75e+03 | 33.7 / 3.04 / 0.337 | 6.2, 8.24; 7.89 |
| h6_r2.5 | so | 6 (1) / 6 (1) / 6 (1) / 6 (1) | 5.32e+04 / 5.26e+04 / 4.39e+03 / 4.39e+03 | 60.9 / 5.48 / 0.609 | 7.39, 9.83; 3.3 |
| h6_r2.5 | sf | 8 (1) / 8 (1) / 9 (1) / 9 (1) | 9.9e+04 / 9.87e+04 / 9.16e+03 / 9.16e+03 | 23 / 2.07 / 0.23 | 14, 18.7; 2.53 |
| h6_r3.0 | so | 10 (1) / 11 (1) / 11 (1) / 11 (1) | 1.49e+05 / 1.84e+05 / 3.47e+04 / 3.47e+04 | 36.4 / 3.28 / 0.364 | 13.4, 17.8; 1.42 |
| h6_r3.0 | sf | 12 (2) / 12 (2) / 14 (2) / 14 (2) | 2.23e+05 / 2.18e+05 / 6.03e+04 / 6.03e+04 | 20.1 / 1.81 / 0.201 | 22.5, 29.9; 0.982 |
| h6_r3.5 | so | 14 (2) / 14 (2) / 15 (2) / 15 (2) | 2.44e+05 / 2.5e+05 / 9.75e+04 / 8.82e+04 | 20.8 / 1.88 / 0.208 | 36, 47.9; 0.263 |
| h6_r3.5 | sf | 25 (2) / 27 (2) / 31 (2) / 29 (2) | 9.04e+05 / 1.04e+06 / 5.85e+05 / 5.34e+05 | 23.6 / 2.12 / 0.236 | 29.3, 38.9; 0.369 |
| h6_r4.0 | so | 22 (6) / 22 (6) / 25 (6) / 26 (6) | 5.19e+05 / 5.12e+05 / 4.54e+05 / 4.97e+05 | 8.05 / 0.724 / 0.0805 | 142, 188; 0.029 |
| h6_r4.0 | sf | 37 (5) / 40 (5) / 40 (5) / - (5) | 1.82e+06 / 2.16e+06 / 1.55e+06 / 320 (k_p90) | 17.3 / 1.55 / 0.173 | 69.7, 92.6; 0.083 |
| h8_r1.8 | so | 6 (1) / 6 (1) / 6 (1) / 6 (1) | 1.56e+05 / 1.61e+05 / 2.66e+04 / 2.66e+04 | 1.4e+03 / 126 / 14 | 4.65, 6.18; 1.03 |
| h8_r1.8 | sf | 9 (1) / 9 (1) / 11 (1) / 11 (1) | 3.64e+05 / 3.68e+05 / 7.63e+04 / 7.6e+04 | 462 / 41.6 / 4.62 | 7.67, 10.2; 1.12 |
| h8_r3.0 | so | 22 (3) / 23 (3) / 26 (3) / 25 (3) | - / - / - / - | 458 / 41.2 / 4.58 | 19.4, 25.8; 0.14 |
| h8_r3.0 | sf | 29 (3) / 30 (3) / - (2) / - (3) | - / - / 27 (k_p90) / 999 (k_p90) | 300 / 27 / 3 | 29, 38.5; 0.112 |
| n2_r1.098 | so | 3 (0) / 3 (0) / 3 (0) / 3 (0) | 1.22e+03 / 1.23e+03 / 18.8 / 18.8 | 8.78 / 0.79 / 0.0878 | 2.55, 3.4; 34.8 |
| n2_r1.098 | sf | 3 (0) / 4 (0) / 4 (0) / 4 (0) | 1.25e+03 / 2.36e+03 / 17.4 / 17.4 | 3.77 / 0.339 / 0.0377 | 3.55, 4.72; 40.9 |
| n2_r1.6 | so | 7 (2) / 7 (3) / 7 (3) / 7 (3) | 8.99e+03 / 8.84e+03 / 134 / 134 | 6.62 / 0.595 / 0.0662 | 4.8, 6.38; 14.9 |
| n2_r1.6 | sf | 7 (3) / 7 (4) / 8 (4) / 8 (4) | 9.19e+03 / 9.17e+03 / 260 / 260 | 3.73 / 0.336 / 0.0373 | 4.41, 5.87; 30.4 |
| n2_r2.0 | so | - (5) / - (6) / - (7) / - (7) | 2.69e+03 (k_p90) / 4.11e+03 (k_p90) / 53.4 (k_p90) / 57.9 (k_p90) | 5.72 / 0.515 / 0.0572 | 8.13, 10.8; 3.84 |
| n2_r2.0 | sf | - (6) / - (8) / - (9) / - (8) | 3.98e+03 (k_p90) / 6.81e+03 (k_p90) / 116 (k_p90) / 122 (k_p90) | 4.26 / 0.384 / 0.0426 | 5.53, 7.36; 11.1 |
| n2_r2.4 | so | - (16) / - (14) / - (21) / - (17) | 1.74e+04 (k_p90) / 1.54e+04 (k_p90) / 3.62e+03 (k_p90) / 2.24e+03 (k_p90) | 4.7 / 0.423 / 0.047 | 11.5, 15.3; 2.09 |
| n2_r2.4 | sf | - (21) / - (16) / - (37) / - (32) | 3.23e+04 (k_p90) / 1.82e+04 (k_p90) / - / - | 3.94 / 0.355 / 0.0394 | 7.2, 9.58; 6.6 |
| n2_r2.8 | so | - (-) / - (17) / - (-) / - (-) | - / 1.63e+04 (k_p90) / - / - | 4.43 / 0.398 / 0.0443 | 14, 18.6; 1.51 |
| n2_r2.8 | sf | - (29) / - (32) / - (29) / - (-) | - / - / - / - | 3.89 / 0.35 / 0.0389 | 8.7, 11.6; 4.54 |

- **Ranking at r = 1** (C_loop(k_ref)/C_QPE to four digits in U0, Q3). R1 is cheapest, ties included, on 22 of
  the 26 pairs resolved at k* and 2 of the 4 at k_p90; R3 is cheaper on h4_r4.0 so, h6_r3.5 (both metrics),
  h8_r1.8 sf and n2_r2.4 so, and on n2_r2.8 so only M3's grid reaches p ≥ 0.9 within 40 steps. Unresolved,
  because a rule's costed iterations end before its k_ref with a partial sum below the cheapest total, are
  h8_r3.0 (both metrics), n2_r2.4 sf and n2_r2.8 sf. Line search and M3 grid pay 11 energies at ±1.6 mHa per
  iteration, C_iter(1) = 268–1e4 C_QPE at r = 1; R1 pays its residual alone (direct_mg on every row, ∝ r⁻²).
- **Iteration counts.** Unit steps reach k* in as many iterations as the exact line search on 9 of 14
  spin-orbital and 3 of 13 spin-free pairs, with at most 4 and 6 more and on average 0.71 and 1.54 more
  [derived], fewer than the design's estimate of about 1.65 and 2.4; the spin-orbital metric needs fewer R1
  iterations than the spin-free one on 15 of the 17 systems, as in M3 §7.
- **Where R1's gain ends.** R1's C_loop(k*) relative to the line search's at r = 1 grows from 0.0038–0.035 on
  h4 at R ≤ 3.0 bohr to 0.85–0.87 on h6_r4.0 [derived]: the line search dominates the early iterations, the
  residual the late ones (M3 §3), where R1's extra iterations fall; at r = 0.3 the exact line search (h6_r3.5
  sf, h6_r4.0 sf) or M3's grid (h6_r4.0 so) is cheaper than R1.
- **The energy-precision wall (F28).** Decision and fit energies are set by the model decrease |ΔE_m| = |ŝ₀|/2
  (F26, F27): σ_A = 2.32–142 mHa at k = 1 but 0.0141–1.55 mHa at k_ref − 1. The Armijo pair costs ρ_E =
  0.029–104 times the direct residual at k = 1 (above 1 on 13 of 17 systems, below only where σ_A = 19.4–142
  mHa) and 5.36–569 times at k_ref − 1 on all 26 pairs where R1 reaches k*; the 3-point fit costs 0.85–1.09
  times as much [derived]. ρ_E grows as r², and amplitude-estimated energies do not remove the wall: one AE
  energy at precision σ costs at least 4κΔE/σ C_QPE (F28), 2.6–284 C_QPE at the σ_A of k_ref − 1 [derived].
- **R3 and the safeguards.** The planned gate opens in 146 of 1167 iterations at r = 0.3, 61 of 1163 at r = 1
  and 15 of 1161 at r = 3 (143 times on stretched N2), saving an iteration on some pairs and costing on
  others. The growth cap binds in 292 of the 1161 R1 steps (69 of the 352 before k_ref), with no uncapped R1
  run to compare; the angle cap never binds on R1 and binds in 6 of 4652 model-rule steps (0.13%, against the
  design's estimate of 0.3%).
- **Comparison with M3.** U5's line-search rows at r = 0.3 are M3's classical-first-step loops
  (C_loop(k*) = 2.42e3 / 2.38e3 on h4_r1.8, 1.28e3 / 1.26e3 on n2_r1.098, 4.05e5 / 2.55e5 C_QPE on h6_r3.0,
  sf / so: M3 T4's values); on h8_r1.8, whose residual M3 did not cost, the loop costs 1.37e5–2.7e5 (so) and
  3.26e5–6.07e5 C_QPE (sf), against M3 T4's lower bounds ≥ 1.35e5 and ≥ 3.21e5. M3's statement that a cheaper
  step rule does not change the equilibrium verdict (M3 §10, caveat 3) holds at r ≤ 1: at r = 1 R1's first
  iteration alone costs 0.127–0.474 (h4) and 0.339–0.79 C_QPE (n2), against the at most
  C_QPE(1/p_HF − 1) = 0.058 and 0.080 C_QPE any better state could save [derived]. At r = 3 it fails,
  marginally, on h4 with the spin-free metric: the unit iteration (0.0141 C_QPE) pays at s* = rep by 1.66e-4
  C_QPE (0.0163%; U9 part 1, §8), on a noise-free trajectory at 300% residual error that no closed-loop cell
  tests (no closed-loop cell runs h4 with the spin-free metric, which enters only n2_r2.0's two metric cells
  of U7); the exact n2_r1.098 records and the closed-loop controls (§6) do not pay.
- **Uncertainty and model dependence.** U5 is deterministic and carries the model dependence of F14 and F30
  (M3's residual-route estimates, κ = 1, variances fixed at one state per system); every energy of iteration k
  is charged at V_E(Ψ_{k+1}) of the DF split chosen at HF (F28), a lower estimate for the 3-point fit (§12,
  item 7). The ranking holds over the costed horizon only, hence the four unresolved pairs.

## 6. The closed loop (U6, U7)

In the pre-declared deciding cells (lagged controller at r = 1, R1 spin-orbital, direct_mg, exact propagation,
100 replicates) the first measured iteration Ψ₁ → Ψ̃₂ pays under repetition on all four headline systems, by
0.268 (n2_r2.0) to 1.73 C_QPE (n2_r2.8). At the best realizable scheme s*, the verdict of record, it pays only
on n2_r2.8 (0.0212 [0.0151, 0.027] C_QPE, 1.35% of T_s*(Ψ₁), lt_bbht_L with termination not certified) and on
h4_r4.0 (0.844 [0.841, 0.846] C_QPE, repetition with B_L, flagged no_certificate and
termination_not_certified), and loses on n2_r2.0 (−0.054 [−0.0545, −0.0535]) and n2_r2.4 (−0.0111 [−0.0153,
−0.007]). The second measured iteration pays in no deciding cell under any scheme (−0.686 to −1.72 C_QPE at
s*). The loop keeps the exact trajectory's overlap gain of the first measured step (median p̃₂ within 0.012 of
the exact R1 p₂) but not of the later ones (median p̃₄ on stretched N2 0.512, 0.329 and 0.259 against the
exact R1's 0.787, 0.620 and 0.546). On stretched N2 the first iteration is best at the loosest residual (r₁ ≥
3), on h4_r4.0 at r₁ ≈ 1, where r = 2–3 is catastrophic. The lagged sensitivity reverses the s* verdict on
n2_r2.4 (r = 2) and n2_r2.8 (r = 0.5); no ablation rescues an energy-based rule, the AE route or the compiled
h4_r4.0 cell, and no control pays at s*.

**U6** (closed loop, 15 of the 24 headline cells; `results/m4/loop_summary.md`, U6, has every cell with
T_rep(3) and the regrets of the stopping rules; `results/m4/closedloop_<s>.json`). R1 spin-orbital from HF
with the classical step, direct_mg with independent Gaussian noise at the planned budget (F33), exact
propagation, 5 iterations, 100 replicates with common random numbers (F38). The oracle controller plans B_k =
r‖A_k‖\_m (not realizable), the lagged one B₁ = 0.5r‖A₀‖\_m and B_k = rn̂_{k−1} (F35). T_rep(k) = [C_loop(k) +
(C(Ψ̃_k) + C_QPE)/p̃_k]/C_QPE per replicate is M3's rule-E cost without its certification condition, p̃_k on
ψ₀ of H; T(1) − T(2) is the mean paired saving of the first measured iteration; r₁ / r₂ the median realized
r_k = B_k/‖A_k‖\_m; catastrophic: the share of replicates with p̃_{k+1} < p̃_k/2 at some k. T_rep(0) = 3.956,
8.892, 12.74 and 2.304 (HF).

| system | configuration | T_rep(1) | T_rep(2) [95% CI] | T(1) − T(2) [CI] | P(T(2) < T(1)) | median p̃₂ [16–84%] | r₁ / r₂ | catastrophic | n |
|---|---|---|---|---|---|---|---|---|---|
| n2_r2.0 | oracle r = 1 | 2.525 | 2.64 [2.621, 2.658] | -0.115 [-0.133, -0.0964] | 0.11 | 0.472 [0.448, 0.491] | 1.00 / 1.00 | 0.00 | 100 |
| n2_r2.0 | oracle r = 3 | 2.525 | 2.257 [2.221, 2.291] | 0.268 [0.234, 0.304] pays | 0.94 | 0.459 [0.413, 0.491] | 3.00 / 3.00 | 0.06 | 100 |
| n2_r2.0 | lagged r = 1 (decides) | 2.525 | 2.257 [2.221, 2.291] | 0.268 [0.234, 0.304] pays | 0.94 | 0.459 [0.413, 0.491] | 3.00 / 0.82 | 0.00 | 100 |
| n2_r2.4 | oracle r = 1 | 4.694 | 3.955 [3.894, 4.018] | 0.739 [0.676, 0.8] pays | 0.97 | 0.288 [0.264, 0.311] | 1.00 / 1.00 | 0.00 | 100 |
| n2_r2.4 | oracle r = 3 | 4.694 | 3.603 [3.496, 3.714] | 1.09 [0.98, 1.2] pays | 0.95 | 0.290 [0.246, 0.328] | 3.00 / 3.00 | 0.05 | 100 |
| n2_r2.4 | lagged r = 1 (decides) | 4.694 | 3.601 [3.496, 3.711] | 1.09 [0.983, 1.2] pays | 0.96 | 0.291 [0.248, 0.328] | 2.71 / 1.11 | 0.00 | 100 |
| n2_r2.8 | oracle r = 1 | 6.049 | 4.716 [4.629, 4.804] | 1.33 [1.25, 1.42] pays | 0.99 | 0.234 [0.211, 0.256] | 1.00 / 1.00 | 0.00 | 100 |
| n2_r2.8 | oracle r = 3 | 6.049 | 4.303 [4.168, 4.443] | 1.75 [1.61, 1.88] pays | 0.98 | 0.243 [0.205, 0.279] | 3.00 / 3.00 | 0.00 | 100 |
| n2_r2.8 | lagged r = 1 (decides) | 6.049 | 4.317 [4.185, 4.453] | 1.73 [1.6, 1.86] pays | 0.99 | 0.241 [0.205, 0.282] | 2.54 / 1.38 | 0.01 | 100 |
| h4_r4.0 | oracle r = 1 | 2.002 | 1.155 [1.14, 1.171] | 0.848 [0.832, 0.863] pays | 1.00 | 0.898 [0.848, 0.934] | 1.00 / 1.00 | 0.04 | 100 |
| h4_r4.0 | oracle r = 2 | 2.002 | 3.505 [1.529, 6.354] | -1.5 [-4.35, 0.473] | 0.82 | 0.782 [0.472, 0.905] | 2.00 / 2.00 | 0.18 | 100 |
| h4_r4.0 | oracle r = 3 | 2.002 | 63.46 [9.092, 165] | -61.5 [-163, -7.09] | 0.71 | 0.684 [0.144, 0.894] | 3.00 / 3.00 | 0.35 | 100 |
| h4_r4.0 | lagged r = 0.5 | 2.002 | 1.414 [1.413, 1.415] | 0.588 [0.587, 0.589] pays | 1.00 | 0.944 [0.938, 0.947] | 0.23 / 1.37 | 0.00 | 100 |
| h4_r4.0 | lagged r = 1 (decides) | 2.002 | 1.159 [1.156, 1.161] | 0.844 [0.841, 0.846] pays | 1.00 | 0.936 [0.925, 0.944] | 0.46 / 2.29 | 0.08 | 100 |
| h4_r4.0 | lagged r = 2 | 2.002 | 1.142 [1.131, 1.154] | 0.86 [0.848, 0.871] pays | 1.00 | 0.903 [0.863, 0.936] | 0.91 / 3.21 | 0.34 | 100 |

**U6 under the QPE-side schemes** (the savings T_s(k) − T_s(k + 1) of U9 part 2 on the same replicates,
`results/m4/summary.md`; the deciding and oracle r = 3 cells and the two lagged cells that reverse a verdict).
CISD(H′) is the only certificate; "rep" is repetition with rule B's verification runs, whose savings equal
M3's rule E in every row; aa needs the overlap and is reported only; s* is re-chosen at each iterate from the
replicate means. s* carries termination_not_certified on every N2 row (lt_bbht_L) and no_certificate with
termination_not_certified on every h4_r4.0 row (repetition or bbht with B_L; '-': no cost).

| system | configuration | iteration | rep | aa | bbht | lt_bbht_L | s* | saving at s* | pays at s* |
|---|---|---|---|---|---|---|---|---|---|
| n2_r2.0 | lagged r = 1 (decides) | Ψ₁ → Ψ̃₂ | 0.268 [0.234, 0.304] pays | 0.268 [0.234, 0.304] pays | -0.0313 [-0.0349, -0.0272] | -0.054 [-0.0545, -0.0535] | lt_bbht_L -> lt_bbht_L | -0.054 [-0.0545, -0.0535] | no |
| n2_r2.0 | lagged r = 1 (decides) | Ψ̃₂ → Ψ̃₃ | -1.58 [-1.76, -1.4] | -1.58 [-1.76, -1.4] | -1.7 [-1.88, -1.52] | -1.72 [-1.89, -1.54] | lt_bbht_L -> lt_bbht_L | -1.72 [-1.89, -1.54] | no |
| n2_r2.0 | oracle r = 3 | Ψ₁ → Ψ̃₂ | 0.268 [0.234, 0.304] pays | 0.268 [0.234, 0.304] pays | -0.0314 [-0.0349, -0.0272] | -0.0541 [-0.0546, -0.0536] | lt_bbht_L -> lt_bbht_L | -0.0541 [-0.0546, -0.0536] | no |
| n2_r2.4 | lagged r = 1 (decides) | Ψ₁ → Ψ̃₂ | 1.09 [0.983, 1.2] pays | -0.0319 [-0.0484, -0.0135] | 0.294 [0.262, 0.325] pays | -0.0111 [-0.0153, -0.007] | lt_bbht_L -> lt_bbht_L | -0.0111 [-0.0153, -0.007] | no |
| n2_r2.4 | lagged r = 1 (decides) | Ψ̃₂ → Ψ̃₃ | -1.44 [-1.63, -1.27] | -1.64 [-1.84, -1.46] | -1.62 [-1.81, -1.44] | -1.67 [-1.87, -1.49] | lt_bbht_L -> lt_bbht_L | -1.67 [-1.87, -1.49] | no |
| n2_r2.4 | oracle r = 3 | Ψ₁ → Ψ̃₂ | 1.09 [0.98, 1.2] pays | -0.0227 [-0.0389, -0.00449] | 0.301 [0.269, 0.332] pays | -0.00106 [-0.00538, 0.00307] | lt_bbht_L -> lt_bbht_L | -0.00106 [-0.00538, 0.00307] | no |
| n2_r2.4 | lagged r = 2 | Ψ₁ → Ψ̃₂ | 1.01 [0.897, 1.11] pays | -0.00575 [-0.0174, 0.00578] | 0.304 [0.271, 0.336] pays | 0.0276 [0.0232, 0.0318] pays | lt_bbht_L -> lt_bbht_L | 0.0276 [0.0232, 0.0318] | yes |
| n2_r2.8 | lagged r = 1 (decides) | Ψ₁ → Ψ̃₂ | 1.73 [1.6, 1.86] pays | 0.201 [0.186, 0.216] pays | 0.563 [0.518, 0.607] pays | 0.0212 [0.0151, 0.027] pays | lt_bbht_L -> lt_bbht_L | 0.0212 [0.0151, 0.027] | yes |
| n2_r2.8 | lagged r = 1 (decides) | Ψ̃₂ → Ψ̃₃ | -1.45 [-1.61, -1.28] | -1.66 [-1.84, -1.47] | -1.6 [-1.78, -1.42] | -1.66 [-1.85, -1.48] | lt_bbht_L -> lt_bbht_L | -1.66 [-1.85, -1.48] | no |
| n2_r2.8 | oracle r = 3 | Ψ₁ → Ψ̃₂ | 1.75 [1.61, 1.88] pays | 0.219 [0.203, 0.233] pays | 0.58 [0.534, 0.624] pays | 0.0386 [0.0324, 0.0445] pays | lt_bbht_L -> lt_bbht_L | 0.0386 [0.0324, 0.0445] | yes |
| n2_r2.8 | lagged r = 0.5 | Ψ₁ → Ψ̃₂ | 1.5 [1.4, 1.61] pays | 0.0274 [0.0167, 0.0376] pays | 0.367 [0.333, 0.401] pays | -0.166 [-0.17, -0.161] | lt_bbht_L -> lt_bbht_L | -0.166 [-0.17, -0.161] | no |
| h4_r4.0 | lagged r = 1 (decides) | Ψ₁ → Ψ̃₂ | 0.844 [0.841, 0.846] pays | 0.844 [0.841, 0.846] pays | 1.88 [1.87, 1.89] pays | - | rep -> rep | 0.844 [0.841, 0.846] | yes |
| h4_r4.0 | lagged r = 1 (decides) | Ψ̃₂ → Ψ̃₃ | -0.686 [-0.955, -0.467] | -0.547 [-0.671, -0.437] | -1.13 [-1.3, -0.964] | - | rep -> rep | -0.686 [-0.955, -0.467] | no |
| h4_r4.0 | oracle r = 3 | Ψ₁ → Ψ̃₂ | -61.5 [-163, -7.09] | -2.11 [-4.47, -0.476] | -4.04 [-8.94, -0.77] | - | rep -> bbht | -5.19 [-10.1, -1.93] | no |

- **Under repetition** the first measured iteration pays in the deciding cells with P(T(2) < T(1)) =
  0.94–1.00, saving 10.60–42.13% of T_rep(1) [derived], and in 21 of the 24 headline cells. M3's rule E
  certifies Ψ̃₂ in 66 (n2_r2.0), 0 (n2_r2.4, n2_r2.8) and 58 (h4_r4.0) of the 100 replicates, so its column is
  an oracle cost there; repetition with rule B's runs, realizable on every replicate, saves the same.
- **At s*** the margins on stretched N2 are narrow: −3.79%, −0.74% and +1.35% of T_s*(Ψ₁) = 1.425, 1.501 and
  1.568 C_QPE [derived from U9 part 2], a loss at 2.0 and 2.4 Å and a narrow pay at 2.8 Å, which the lagged
  r = 0.5 cell reverses; bbht pays on n2_r2.4 and n2_r2.8 but is dearer than lt_bbht_L at both iterates. On
  h4_r4.0, where the E1lb-referenced filter does not terminate, s* is repetition, and the iteration pays
  because it lifts the overlap from 0.499 to a median p̃₂ = 0.936 at a mean C_iter(1) of 0.0883 C_QPE. Beside
  the scheme and the classical step the measured iteration is a small term (n2_r2.4: HF 8.892 C_QPE under rule
  E, 1.704 under lt_bbht_L, Ψ₁ 1.501, Ψ̃₂ 1.512; §8).
- **Later iterations.** The second is measured at a median r₂ = 0.82–1.38 on N2 (2.29 on h4_r4.0) and costs
  1.67–1.72 C_QPE there, almost the whole loss. The loop keeps the first step's gain (median p̃₂ = 0.459,
  0.291, 0.241, 0.936 against the exact 0.471, 0.282, 0.230, 0.946) but not the later ones (p̃₄ = 0.512,
  0.329, 0.259, 0.930 against 0.787, 0.620, 0.546, 0.958; U9 part 2); on h4_r4.0 the second budget, planned
  before the large first step, is too loose and p̃₃ falls below p̃₂.
- **Which r.** Under rule E the oracle scan's T_rep(2) falls from r = 1 to 3 on the three N2, since C_iter(1)
  falls as r⁻² while the median p̃₂ moves by at most 0.013; at s* its minimum is at r = 3 (1.479, 1.502,
  1.529), and the lagged r = 2 cells, at r₁ = 6.00, 5.43 and 5.08, lie lower still (1.438, 1.473, 1.503; U0,
  Q4): on stretched N2 the first iteration keeps much of its overlap gain at r₁ = 2.5–6 (median
  p̃₂ = 0.421–0.459, 0.278–0.291 and 0.241–0.243 against p₁ = 0.396, 0.213 and 0.165) while its cost falls as
  r⁻², so the loosest residual is the cheapest. On h4_r4.0 the optimum is r₁ ≈ 1, while at oracle r = 2 and 3
  noisy unit steps on the smallest closed-loop gap (Δ₁ = 28.24 mHa) are catastrophic in 18 and 35 of 100
  replicates and the mean T_rep(2), which decides because 1/p is convex (F38), lies far above its value at the
  median p̃₂.
- **Controller and sensitivity.** The lagged plan realizes r₁ = 3.00r, 2.71r, 2.54r and 0.46r (F35), so the
  deciding cell sits near the oracle r = 3 cell on N2 and near r = 0.5 on h4_r4.0. The lagged r = 2 cell of
  n2_r2.4 pays at s* (0.0276 [0.0232, 0.0318]) but is not realizable (the angle cap bound once), the r = 0.5
  cell of n2_r2.8 does not (−0.166 [−0.17, −0.161]); n2_r2.0 loses at s* at all three lagged r, h4_r4.0 pays
  at all three (0.588, 0.844, 0.86). The angle cap binds in no deciding cell (F38).
- **Exact records, references and M3.** On the exact R1 records (§4) the first iteration pays at s* at r = 3
  on n2_r2.4 and n2_r2.8, not on n2_r2.0; the closed loop agrees except on n2_r2.4, which sits on the boundary
  (−0.00106 [−0.00538, 0.00307] at oracle r = 3). All 16 closed-loop reference rows of the design reproduce
  (§11). On n2_r2.0, M3's n2s, the loop starts from the spin-orbital classical Ψ₁ that M3 certified at 2.525
  C_QPE (M3 T6; M3's cheapest certified measured hand-off cost 188.7). The first measured iteration costs
  0.0571 C_QPE and takes Ψ̃₂ to 2.257 [2.221, 2.291] C_QPE under repetition with rule B's B_U at the free
  CISD(H′) (n_v = 1), equal to rule E's oracle cost (rule E itself certifies Ψ̃₂ on 66 of the 100 replicates):
  0.268 C_QPE (10.6%) below M3's cheapest certified hand-off of n2s by a measured or classical-step loop
  [derived].

**U7** (closed-loop ablations and controls, selected rows; `results/m4/loop_summary.md`, U7, holds all 49
cells with their flags). As U6 unless noted: 20 replicates (10 for the N = 8 controls), oracle controller
except the lagged cells. line_search is the exact line search along the measured d_k, charged 11 energies
(F36); hybrid is R3 with the gate at the constant V_E(HF); ae_x the X-LCU amplitude-estimation route (F34);
compiled cells propagate by F25's slices (F37; not realizable), with largest compile errors of 0.0361–0.0494;
spin_free switches the metric. The controls decide in their lagged r = 1 cells.

| block | system | configuration | T_rep(1) | T_rep(2) [95% CI] | T(1) − T(2) [CI] | median p̃₂ [16–84%] | r₁ / r₂ | catastrophic | mean C_iter(1) / C_iter(2) |
|---|---|---|---|---|---|---|---|---|---|
| metric | n2_r2.0 | spin_free oracle r = 1 | 4.111 | 4.381 [4.329, 4.437] | -0.27 [-0.326, -0.218] | 0.251 [0.241, 0.258] | 1.00 / 1.00 | 0.00 | 0.384 / 2.76 |
| routes | n2_r2.4 | ae_x fixed oracle r = 1 | 4.694 | 8.635 [8.529, 8.735] | -3.94 [-4.04, -3.84] | 0.284 [0.264, 0.295] | 1.00 / 1.00 | 0.00 | 5.09 / 12.6 |
| step_rules | n2_r2.4 | line_search oracle r = 1 | 4.694 | 481.1 [473.1, 489.8] | -476 [-485, -468] | 0.315 [0.297, 0.346] | 1.00 / 1.00 | 0.00 | 478 / 908 |
| step_rules | n2_r2.4 | hybrid oracle r = 3 | 4.694 | 3.587 [3.377, 3.803] | 1.11 [0.891, 1.32] pays | 0.275 [0.260, 0.328] | 3.00 / 3.00 | 0.05 | 0.0478 / 0.229 |
| compiled | n2_r2.8 | compiled lagged r = 1 | 5.827 | 4.045 [3.731, 4.401] | 1.78 [1.43, 2.1] pays | 0.262 [0.221, 0.306] | 2.40 / 1.44 | 0.00 | 0.0618 / 1.89 |
| compiled | h4_r4.0 | compiled lagged r = 1 | 1.994 | 2.478 [2.474, 2.481] | -0.484 [-0.487, -0.48] | 0.940 [0.934, 0.949] | 0.46 / 2.14 | 0.10 | 1.41 / 1.11 |
| controls | h6_r3.0 | lagged r = 1 (decides) | 1.091 | 5.45 [5.447, 5.453] | -4.36 [-4.36, -4.36] | 0.933 [0.927, 0.937] | 0.87 / 1.27 | 0.00 | 4.38 / 35.6 |
| controls | h4_r1.8 | lagged r = 1 (decides) | 1.009 | 1.273 [1.272, 1.275] | -0.264 [-0.266, -0.263] | 0.994 [0.992, 0.997] | 1.33 / 0.58 | 0.00 | 0.266 / 9.72 |
| controls | n2_r1.098 | lagged r = 1 (decides) | 1.01 | 1.179 [1.173, 1.187] | -0.169 [-0.177, -0.163] | 0.988 [0.979, 0.990] | 2.22 / 0.42 | 0.00 | 0.161 / 5.39 |
| n8_controls | n2v_r2.4 | lagged r = 1 (decides) | 4.769 | 4.023 [3.904, 4.148] | 0.747 [0.622, 0.866] pays | 0.286 [0.271, 0.305] | 2.65 / 0.65 | 0.00 | 0.521 / 33.2 |
| n8_controls | n2v_r2.8 | lagged r = 1 (decides) | 6.113 | 4.801 [4.577, 5.018] | 1.31 [1.1, 1.54] pays | 0.228 [0.215, 0.258] | 2.58 / 0.86 | 0.00 | 0.523 / 28.9 |
| n8_controls | h8_r3.0 | lagged r = 1 (decides) | 1.155 | 68.69 [68.69, 68.7] | -67.5 [-67.5, -67.5] | 0.878 [0.875, 0.885] | 0.78 / 1.16 | 0.00 | 67.6 / 425 |

- **Ablations.** Line search and M3's grid cost 449–485 C_QPE in the first measured iteration and 757–908 in
  the second, so T_rep(2) = 453.1–487 against T_rep(1) = 2.525 and 4.694: M3's line-search cost carried into
  the loop (eleven energies at the fixed 1.6 mHa, F36), about 10³ times the residual, not F28's
  energy-precision wall, which concerns energies at σ ∝ |ΔE_m| (the hybrid cells, whose gated fits use such
  energies, cost 0.0478–0.515 C_QPE in the first measured iteration). R3 matches R1 within the intervals, and
  no rule brings a replicate within 1.6 mHa by k = 4 (U0, Q3). ae_x costs 5.09 and 1.7 C_QPE at k = 1 on
  n2_r2.4 (r = 1, 3) against 0.423 and 0.047 for direct_mg; amplitude estimation becomes the cheaper route
  only from k = 2–10 on U5's records, as in M3 T3. On N2 the compiled lagged cells keep the exact verdicts (s*
  savings −0.0544, −0.0071 and 0.0212 [0.0051, 0.0346] C_QPE); on h4_r4.0 F25's slices raise the classical
  step's circuit from 160 to 2560 Toffolis, the first iteration costs 1.41 C_QPE against 0.0883, and it loses
  (−0.484 [−0.487, −0.48], at s* too). With the spin-free metric the classical step lowers the overlap on
  n2_r2.0 (M3 T7's loss on n2s) and the first iteration loses: the N2 results above need the spin-orbital
  metric.
- **Controls.** On h4_r1.8, n2_r1.098 and h6_r3.0 (M3's h4, n2, h6s) the first iteration loses in every cell
  under every scheme (lagged r = 1: −0.264, −0.169, −4.36 C_QPE; −4.37 at s* on h6_r3.0): the classical Ψ₁
  already sits at T_rep(1) = 1.009, 1.01 and 1.091, leaving at most 0.009–0.091 C_QPE for any later state
  [derived], less than the first iteration costs even at oracle r = 3, so M3's verdicts hold in the closed
  loop: at equilibrium on h4 and n2, and on the stretched h6s, where the classical Ψ₁ remains the optimum. No
  closed-loop cell runs h4 with the spin-free metric (it enters only n2_r2.0's two metric cells of U7), so the
  one equilibrium exception on the exact trajectories (h4 spin-free at r = 3, §5) has no closed-loop
  counterpart. h8_r3.0 loses by 67.5 C_QPE; N2 CAS(10,8) at 2.4 and 2.8 Å pays under repetition (0.747, 1.31)
  but not at s* (−0.469, −0.438; §8), its first iteration costing 0.521 and 0.523 C_QPE.
- **Against the design.** For stretched N2 CAS(6,6) at oracle r = 3 the design expected the first measured
  iteration to save 10 / 19 / 25% of Ψ₁'s hand-off under repetition with verification runs (10 / 23 / 29%
  under M3's rule E), −1 / 6 / 11% under bbht, about 10 / −0.2 / 5% under known-overlap AA and −3.9 / −0.3 /
  +2.2% under lt_bbht_L at 2.0 / 2.4 / 2.8 Å. The run gives 10.60 / 19.16 / 24.77% (10.60 / 23.24 / 28.87%),
  −0.98 / 6.32 / 11.04%, 10.60 / −0.56 / 5.08% and −3.80 [−3.83, −3.76] / −0.07 [−0.36, 0.20] / +2.46 [2.07,
  2.84]% [derived: saving / T_s(Ψ₁)]: confirmed (AA at 2.4 Å just below the range), with margins as narrow as
  the design expected; but at 2.8 Å the interval lies above zero in the oracle r = 3 cell and in the deciding
  cell (−3.79 / −0.74 / +1.35%), so there the design's near miss is a narrow pay. That the second iteration
  does not pay is confirmed; that none pays on N2 CAS(10,8) holds at s* but not under repetition; that none
  pays on the chains at N ≤ 10 is refuted on h4_r4.0 and confirmed on h4_r1.8, h6_r3.0 and h8_r3.0. That the
  scheme saves most and the classical step next holds on stretched N2 but not on h4_r4.0, where the realizable
  scheme saves nothing at HF (3.304 C_QPE against rule E's oracle 2.304) and the measured iteration saves most
  (3.002 → 2.159 at s*).
- **Uncertainty and model dependence.** The intervals of the s* savings are 0.0005–0.012 C_QPE wide on N2, so
  every deciding sign is resolved by the replicates, but not by the model. Under lt_bbht_L a better state
  saves about in proportion to C_filt (T_s = C_loop + E_BBHT(p)(c_k + C_filt) + C_QPE, c_k ≪ C_filt; F43),
  whose walk part F43 counts conservatively by up to 2×: on n2_r2.8 μ* = 0.0830 C_QPE (the saving 0.0212 plus
  the mean C_iter(1) 0.0618) would fall to about 0.044 at half the walk count, below C_iter(1), and a filter
  about 25% cheaper would already turn the pay into a loss, while on n2_r2.0 and n2_r2.4 a cheaper filter
  deepens the loss [derived from the U9 part 2 means and C_filt(w_L) = 0.1335 C_QPE]. The noise model is M3's
  (independent Gaussian errors, variances fixed at iterate 1 or HF, F33; matchgate correlations, which raised
  M2's noise floors by up to 1.32×, not modelled), every deciding cell propagates exactly, and p̃_k is on ψ₀
  of H while the schemes act on H′ (F38). The verdicts hold for the ten closed-loop systems and five
  iterations only.

## 7. The hand-off under rule B (U8)

Rule B (F40) gives every exact state of the 19 distinct hand-off systems a realizable acceptance whose
identification error is bounded under the windowed QPE kernels of F39 for every p₀ ≥ p_min, conditional on the
assumed E1lb (F41), including the 83 of 209 exact states that M3's rule E does not certify (87 of 231 over the
21 labels of `handoff_summary.md`, where n2_r1.098 and n2_r2.0 repeat n2 and n2s). At HF it takes B_U with the
free certificate U = CISD(H′) on 10 systems (two runs on h4_r3.5, h6s and n2v_r2.0) and B_L (θ_L = E1lb − 5g,
two runs, termination not certified) on the nine stretched systems where CISD(H′) certifies no U or would need
three runs, at M3's oracle T_E plus one C_QPE per verification run (T_B(HF) = 1.026–13.8 C_QPE). On every
exact state the identification error of the chosen form is at most 5.00e-4, the logical-failure budget (the
kernel term is at most 1.24e-6). On M3's T6 rule B moves no equilibrium verdict. On h6s and n2s it certifies
HF, so the certified optima of the measured loops fall from 554.1/517.1 and 570.8/188.7 C_QPE to 2.744 and
3.956; but on h6s every other optimum rises by the verification run (classical step 1.132 → 2.132 and
1.091 → 2.091, free loops 1.014 → 2.014 and 1.012 → 2.012), so the cheapest certified hand-off of M3's
stretched systems by a measured or classical-step loop does not fall (1.091 → 2.091 C_QPE on h6s, 2.525
unchanged on n2s): on M3's systems rule B buys soundness under real kernels, not a cheaper hand-off. Compiled
states lose the design-floor bound on 52 of the 63 compiled rows of the 21 labels (46 of the 57 compiled
states of the 19 systems), since the elementwise circuits populate levels below E1lb; on stretched N2 these
are spin contaminants below θ, which pass the verification runs (total up to 1.15e-3).

**QPE side and certificates** (F39–F42; `results/m4/handoff_summary.md`, "QPE side and certificates", which
adds the encoding, C_QPE, λ, C_W, n_iter, sin θ₀ and E₀′ − E₀). n2_r1.098 and n2_r2.0 repeat n2 and n2s row
for row and are left out here and in §8. Energies in mHa above E₀′; g = 2ΔE sin θ₀, the grid of the
sine-window QPE at M = 2n_iter; E1lb = E₁′ − 0.1 mHa, referred to the any-irrep singlet on the THC systems
(h6a, h6s, h8, h6_r3.5, h6_r4.0, h8_r3.0, h8_r4.0) and on n2_r2.4 and n2v_r2.4 (point_group_broken, F41); "any
spin": the lowest excited level of the ground irrep of any S; margin = E1lb − U − 2ΔE and D_U = (E1lb − U −
ΔE)/g at U = CISD(H′) ('no U': CISD(H′) > E1lb − 2ΔE); n_v by F40 at p_min = ε_id = 1e-3, n_v(B_L) at D_L = 5.

| system | g (mHa) | E1lb - E0' | any spin | CISD - E0' | CISDT - E0' | margin | D_U | n_v(B_U) | n_v(B_L) | flags |
|---|---|---|---|---|---|---|---|---|---|---|
| h4 | 1.909 | 565.566 | 491.404 | 1.037 | 0.989 | 562.529 | 295 | 1 | 2 | - |
| h6a | 1.895 | 390.865 | 351.204 | 4.685 | 4.125 | 384.180 | 203 | 1 | 2 | - |
| h6s | 1.936 | 104.576 | 88.565 | 54.495 | 51.389 | 48.081 | 25.4 | 2 | 2 | - |
| h8 | 1.904 | 344.409 | 316.355 | 7.808 | 6.690 | 334.601 | 176 | 1 | 2 | - |
| lih | 1.916 | 132.968 | 115.924 | 0.013 | 0.003 | 130.955 | 68.9 | 1 | 2 | - |
| n2 | 1.912 | 711.393 | 618.544 | 7.970 | 7.970 | 701.423 | 367 | 1 | 2 | - |
| n2s | 1.970 | 191.807 | 21.395 | 134.215 | 54.752 | 55.592 | 28.7 | 1 | 2 | - |
| n2_r1.6 | 1.962 | 218.547 | 107.302 | 64.327 | 61.430 | 152.220 | 78.1 | 1 | 2 | - |
| n2_r2.4 | 1.971 | 230.763 | 4.346 | 225.686 | 9.241 | 3.077 | 2.07 | 3 | 2 | point_group_broken |
| n2_r2.8 | 1.972 | 244.341 | 1.210 | 293.495 | 2.455 | no U | - | - | 2 | - |
| h4_r3.5 | 1.965 | 65.160 | 52.316 | 38.254 | 25.410 | 24.906 | 13.2 | 2 | 2 | - |
| h4_r4.0 | 1.968 | 28.139 | 22.440 | 39.240 | 10.852 | no U | - | - | 2 | - |
| h6_r3.5 | 1.939 | 49.155 | 40.990 | 108.248 | 61.517 | no U | - | - | 2 | - |
| h6_r4.0 | 1.938 | 21.229 | 17.600 | 170.299 | 61.037 | no U | - | - | 2 | - |
| h8_r3.0 | 1.945 | 83.237 | 71.858 | 97.341 | 89.594 | no U | - | - | 2 | - |
| h8_r4.0 | 1.947 | 16.883 | 14.302 | 287.132 | 176.815 | no U | - | - | 2 | - |
| n2v_r2.0 | 1.978 | 196.322 | 25.843 | 150.945 | 68.136 | 43.376 | 22.4 | 2 | 2 | - |
| n2v_r2.4 | 1.978 | 232.508 | 6.333 | 232.051 | 14.754 | no U | - | - | 2 | point_group_broken, spin_tol_relaxed |
| n2v_r2.8 | 1.977 | 243.951 | 1.340 | 294.992 | 3.213 | no U | - | - | 2 | - |

- **Certificates and runs.** CISD(H′) lies 0.013 (lih) to 294.992 mHa (n2v_r2.8) above E₀′ and certifies U at
  D_U = 68.9–367 grid units at equilibrium and on n2_r1.6, 22.4–28.7 on h6s, n2s and n2v_r2.0, 13.2 on h4_r3.5
  and 2.07 on n2_r2.4 (margin 3.077 mHa). τ_W of the sine window is a step function of the distance: one run
  needs D ≥ t₁ = 26.036 grid units, two D ≥ t₂ = 3.032, three D ≥ t₃ = 1.295 (F40), so h4_r3.5, h6s and
  n2v_r2.0 need two runs where n2s (28.7) needs one, and on n2_r2.4 B_L with two runs beats B_U with three.
  B_L's threshold lies 7.149 mHa above E₀′ at the closest (h8_r4.0), so it terminates everywhere.
- **L forms and sectors.** The E1lb-referenced schemes terminate everywhere except on the six stretched chains
  (E1lb − E₀′ = 16.883–83.237 mHa < w_L; §8). On stretched N2 the lowest same-irrep level of any spin is a
  quintet 21.395, 4.346 and 1.210 mHa above E₀′ (n2s, n2_r2.4, n2_r2.8), so a spin-broken state has no E1lb
  with a margin there.

**U8 at HF** (`results/m4/handoff_summary.md`, "U8: rule B per state", 294 rows: HF, the classical Ψ₁, the R1
Ψ₂ and Ψ₃ of both metrics, Ψ₂ and Ψ₃ of the exact-line-search records and the compiled Ψ̃₁–Ψ̃₃, with F30's R1
loop costs at r = 1 and M3's F14 costs at r = 0.3). p on ψ₀′ of H′, so that T_B equals U9's rep (M3's T6 below
keeps p on H); E − E₀ in mHa of H (rule E certifies HF on h4, h6a, h8, lih and n2); T_B in C_QPE (window
model, K₀ = 1, verification runs and certificate included). ε_id terms of the accepted state: the sine kernel
at M = 2n_iter with n_v runs, the logical failure term at F40's budgets and their total; "bound" = [(1 −
p_min)/p_min] τ_W(D)^{n_v}, the guarantee for every state with p₀ ≥ p_min; "state bound" = [(1 −
p₀)/p₀][τ_W(D)/K₀]^{n_v}. Every B_L row carries no_certificate and termination_not_certified.

| system | p | E - E0 | form | n_v | T_B | sine n_v | logical | total | bound | state bound |
|---|---|---|---|---|---|---|---|---|---|---|
| h4 | 0.9455 | 61.98 | B_U | 1 | 1.058 | 1.03e-11 | 1.1e-6 | 1.06e-6 | 6.98e-7 | 4.04e-11 |
| h6a | 0.9026 | 100.53 | B_U | 1 | 1.108 | 3.72e-11 | 1.1e-6 | 1.11e-6 | 2.06e-6 | 2.23e-10 |
| h6s | 0.5733 | 282.21 | B_U | 2 | 2.744 | 2.16e-14 | 5.0e-4 | 5.00e-4 | 1.04e-9 | 7.73e-13 |
| h8 | 0.8864 | 122.48 | B_U | 1 | 1.128 | 1.87e-11 | 1.1e-6 | 1.13e-6 | 3.14e-6 | 4.03e-10 |
| lih | 0.9744 | 20.38 | B_U | 1 | 1.026 | 3.81e-10 | 1.9e-6 | 1.94e-6 | 5.27e-5 | 2.62e-9 |
| n2 | 0.9257 | 126.04 | B_U | 1 | 1.08 | 3.76e-12 | 1.1e-6 | 1.08e-6 | 3.76e-7 | 3.02e-11 |
| n2s | 0.2539 | 565.52 | B_U | 1 | 3.939 | 7.45e-7 | 3.9e-6 | 4.68e-6 | 7.29e-4 | 2.15e-6 |
| n2_r1.6 | 0.6553 | 328.55 | B_U | 1 | 1.526 | 6.66e-9 | 1.5e-6 | 1.53e-6 | 3.50e-5 | 1.85e-8 |
| n2_r2.4 | 0.1120 | 777.01 | B_L | 2 | 9.928 | 1.05e-10 | 5.0e-4 | 5.00e-4 | 3.37e-5 | 2.67e-7 |
| n2_r2.8 | 0.0781 | 911.99 | B_L | 2 | 13.8 | 1.04e-8 | 5.0e-4 | 5.00e-4 | 3.37e-5 | 3.97e-7 |
| h4_r3.5 | 0.5535 | 274.42 | B_U | 2 | 2.807 | 9.81e-15 | 5.0e-4 | 5.00e-4 | 4.71e-8 | 3.81e-11 |
| h4_r4.0 | 0.4341 | 358.67 | B_L | 2 | 3.304 | 6.89e-10 | 5.0e-4 | 5.00e-4 | 3.37e-5 | 4.39e-8 |
| h6_r3.5 | 0.3961 | 406.56 | B_L | 2 | 3.525 | 8.57e-11 | 5.0e-4 | 5.00e-4 | 3.37e-5 | 5.14e-8 |
| h6_r4.0 | 0.2720 | 534.10 | B_L | 2 | 4.676 | 8.58e-12 | 5.0e-4 | 5.00e-4 | 3.37e-5 | 9.02e-8 |
| h8_r3.0 | 0.4649 | 372.40 | B_L | 2 | 3.151 | 6.89e-10 | 5.0e-4 | 5.00e-4 | 3.37e-5 | 3.88e-8 |
| h8_r4.0 | 0.1685 | 709.45 | B_L | 2 | 6.935 | 2.24e-9 | 5.0e-4 | 5.00e-4 | 3.37e-5 | 1.66e-7 |
| n2v_r2.0 | 0.2453 | 583.61 | B_U | 2 | 5.077 | 9.24e-13 | 5.0e-4 | 5.00e-4 | 2.20e-9 | 6.77e-12 |
| n2v_r2.4 | 0.1111 | 784.69 | B_L | 2 | 10 | 3.09e-11 | 5.0e-4 | 5.00e-4 | 3.37e-5 | 2.70e-7 |
| n2v_r2.8 | 0.0782 | 914.84 | B_L | 2 | 13.78 | 2.06e-8 | 5.0e-4 | 5.00e-4 | 3.37e-5 | 3.97e-7 |

- **Forms and costs over the 294 rows of the 21 labels** (266 states of the 19 systems). 126 take B_U with one
  run, 42 B_U with two, 125 B_L with two, and one a dedicated certificate (n2_r2.4's exact spin-orbital Ψ₃,
  T_B = 1021); elsewhere rule B, which takes the cheapest option per state (F40), finds no admissible
  dedicated energy of F42 that costs less than the verification run it would save. At HF, T_B = M3's oracle
  T_E + (n̄_v − 1)C_QPE (h6s 2.744 = 1.744 + 1); on the stretched chains the second run is 14–36% of T_B(HF)
  [derived], the price of soundness under real kernels.
- **Identification errors of the chosen form.** The kernel term is at most 1.24e-6 (one run) and 5.49e-8 (two
  runs) on exact states; the logical term, 5.0e-4 = ε_fail/2 for every two-run form, sets the largest total.
  The bound is 3.37e-5 for B_L ([(1 − p_min)/p_min]τ_W(5)², numerically equal to F40's overlap floor because
  p_min = ε_id) and at most 7.29e-4 for B_U. Rule B certifies which eigenstate is accepted, not an energy
  window (an outcome of ψ₀′ falls outside ±ΔE of E₀′ with probability 2.6e-4 to 0.414). The costs set K₀ = 1;
  on lih B_U's threshold lies 0.53g above E₀′ (K₀ = 0.529), so the accepted QPE terms are understated by up to
  1.9× (F40).

**Identification errors of the forms side by side, at HF** (spectrum-weighted kernel terms, not bounds;
`handoff_summary.md`, second U8 table, all 294 states). B_U at θ = CISD(H′) + ΔE with one run: sine and rect
at 2n_iter, Kaiser(3) at 4n_iter; B_L at E1lb − 5g, sine, one and two runs; "literal": θ = E₁′ − ΔE, sine.

| system | p | B_U sine | B_U Kaiser(3) 4n | B_U rect | B_L 1 run | B_L 2 runs | literal |
|---|---|---|---|---|---|---|---|
| h4 | 0.9455 | 1.03e-11 | 2.45e-12 | 3.29e-6 | 9.28e-7 | 9.17e-11 | 0.0013 |
| h6a | 0.9026 | 3.72e-11 | 8.56e-12 | 9.86e-6 | 2.23e-6 | 2.62e-10 | 0.00058 |
| h6s | 0.5733 | 3.10e-8 | 1.64e-10 | 1.74e-4 | 3.27e-6 | 2.48e-10 | 0.00927 |
| h8 | 0.8864 | 1.87e-11 | 7.54e-12 | 2.02e-5 | 1.85e-7 | 1.46e-12 | 0.0098 |
| lih | 0.9744 | 3.81e-10 | 7.31e-12 | 7.06e-6 | 4.36e-7 | 3.37e-11 | 0.0012 |
| n2 | 0.9257 | 3.76e-12 | 2.19e-12 | 6.15e-6 | 1.72e-6 | 3.18e-10 | 0.00108 |
| n2s | 0.2539 | 7.45e-7 | 2.33e-9 | 1.25e-3 | 1.08e-4 | 9.58e-9 | 0.18 |
| n2_r1.6 | 0.6553 | 6.66e-9 | 4.40e-11 | 4.25e-5 | 3.15e-5 | 5.31e-9 | 0.0122 |
| n2_r2.4 | 0.1120 | 2.38e-4 | 3.76e-8 | 9.80e-2 | 1.57e-5 | 1.05e-10 | 0.0019 |
| n2_r2.8 | 0.0781 | no U | no U | no U | 1.62e-4 | 1.04e-8 | 0.322 |
| h4_r3.5 | 0.5535 | 1.70e-8 | 1.83e-10 | 3.48e-4 | 3.06e-7 | 3.31e-12 | 4.15e-5 |
| h4_r4.0 | 0.4341 | no U | no U | no U | 4.90e-6 | 6.89e-10 | 0.00149 |
| h6_r3.5 | 0.3961 | no U | no U | no U | 2.15e-6 | 8.57e-11 | 0.00036 |
| h6_r4.0 | 0.2720 | no U | no U | no U | 7.61e-7 | 8.58e-12 | 0.000103 |
| h8_r3.0 | 0.4649 | no U | no U | no U | 6.44e-6 | 6.89e-10 | 0.00157 |
| h8_r4.0 | 0.1685 | no U | no U | no U | 1.49e-5 | 2.24e-9 | 0.0049 |
| n2v_r2.0 | 0.2453 | 1.17e-6 | 4.03e-9 | 3.08e-3 | 1.27e-4 | 1.12e-8 | 0.0271 |
| n2v_r2.4 | 0.1111 | no U | no U | no U | 5.23e-6 | 3.09e-11 | 0.526 |
| n2v_r2.8 | 0.0782 | no U | no U | no U | 2.24e-4 | 2.06e-8 | 0.198 |

- **Over the 231 exact-state rows of the 21 labels** (209 states of the 19 systems) B_U with one sine run
  stays at or below 5.33e-4 on the 143 rows with a certified U (121 states), Kaiser(3) at or below 8.23e-8,
  B_L at or below 5.91e-4 with one run and 5.49e-8 with two; the rect (Fejér) window exceeds 1e-3 on 32 of the
  143 (27 of 121; up to 0.193), and the literal threshold on 144 of the 231 (129 of 209; up to 0.706), because
  E₁′'s main lobe straddles it. One B_L run would have sufficed for every listed state, but as a bound it
  holds only for p₀ ≥ 0.155; the second run extends it to p₀ ≥ 3.37e-5 without knowing p₀, which is what makes
  the form realizable.

**Compiled states** (R1 spin-orbital steps 0–2 compiled elementwise in one slice, with the exact states' loop
costs; six of the 18 stretched-N2 rows, all 63 compiled rows in `handoff_summary.md`). No design-floor bound
applies ("n/a (q_c)").

| system | state | p | E - E0 | form | n_v | T_B | sine n_v | logical | total | q_c | state bound |
|---|---|---|---|---|---|---|---|---|---|---|---|
| n2s | Ψ̃₁ | 0.4042 | 173.62 | B_U | 1 | 2.474 | 1.20e-4 | 2.5e-6 | 1.22e-4 | 4.82e-5 | 1.21e-4 |
| n2s | Ψ̃₃ | 0.6226 | 104.11 | B_U | 1 | 6.473 | 6.86e-5 | 1.6e-6 | 7.03e-5 | 4.26e-5 | 6.93e-5 |
| n2_r2.4 | Ψ̃₁ | 0.2191 | 301.90 | B_L | 2 | 5.565 | 3.03e-4 | 5.0e-4 | 8.03e-4 | 6.63e-5 | 3.03e-4 |
| n2_r2.4 | Ψ̃₃ | 0.4394 | 202.47 | B_L | 2 | 8.235 | 1.44e-4 | 5.0e-4 | 6.44e-4 | 6.33e-5 | 1.44e-4 |
| n2v_r2.8 | Ψ̃₂ | 0.2333 | 317.82 | B_L | 2 | 8.76 | 4.69e-4 | 5.0e-4 | 9.69e-4 | 1.09e-4 | 4.69e-4 |
| n2v_r2.8 | Ψ̃₃ | 0.3523 | 282.35 | B_L | 2 | 27.39 | 6.47e-4 | 5.0e-4 | 1.15e-3 | 2.28e-4 | 6.47e-4 |

- **Contaminants.** The elementwise circuits break S² and populate levels of other spin below θ (q_c = 4.26e-5
  to 2.28e-4 on stretched N2, at most 4.19e-11 elsewhere), which a verification run accepts as readily as ψ₀′:
  the total reaches 6.44e-4 to 9.69e-4 on N2 from 2.4 Å and 1.15e-3 on n2v_r2.8 Ψ̃₃. A compiled stretched-N2
  iterate needs F41's contaminant budget, or circuits that conserve S², not more runs.

**M3's T6 under rule B** (`handoff_summary.md`, "U8: M3's T6 under rule B"; k_opt with T/C_QPE; p on ψ₀ of H,
as in M3 T6; certificates CISD(H′) and the dedicated energies of every iterate). The 30 rows of h4, h6a, h8,
lih and n2 do not change (B_U with one run, D_U = 68.9–367). The rows that change:

| system | family | loop | rows to k | M3 certified | M3 oracle | rule B | form | n_v | U | D |
|---|---|---|---|---|---|---|---|---|---|---|
| h6s | lbfgs | measured | 17 | 1 (554.1) | 0 (1.744) | 0 (2.744) | B_U | 2 | cisd | 25.4 |
| h6s | lbfgs | classical_first_step | 17 | 1 (1.132) | 1 (1.132) | 1 (2.132) | B_U | 2 | cisd | 25.4 |
| h6s | lbfgs | free | 20 | 14 (1.014) | 14 (1.014) | 14 (2.014) | B_U | 2 | cisd | 25.4 |
| h6s | lbfgs_so | measured | 15 | 1 (517.1) | 0 (1.744) | 0 (2.744) | B_U | 2 | cisd | 25.4 |
| h6s | lbfgs_so | classical_first_step | 15 | 1 (1.091) | 1 (1.091) | 1 (2.091) | B_U | 2 | cisd | 25.4 |
| h6s | lbfgs_so | free | 20 | 12 (1.012) | 12 (1.012) | 12 (2.012) | B_U | 2 | cisd | 25.4 |
| n2s | lbfgs | measured | 20 | 2 (570.8) | 0 (3.956) | 0 (3.956) | B_U | 1 | cisd | 28.7 |
| n2s | lbfgs | classical_first_step | 20 | 3 (1073) | 0 (3.956) | 0 (3.956) | B_U | 1 | cisd | 28.7 |
| n2s | lbfgs | free | 20 | 16 (1.012) | 16 (1.012) | 16 (1.012) | B_U | 1 | cisd | 28.7 |
| n2s | lbfgs_so | measured | 20 | 1 (188.7) | 0 (3.956) | 0 (3.956) | B_U | 1 | cisd | 28.7 |
| n2s | lbfgs_so | classical_first_step | 20 | 1 (2.525) | 1 (2.525) | 1 (2.525) | B_U | 1 | cisd | 28.7 |
| n2s | lbfgs_so | free | 20 | 17 (1.013) | 17 (1.013) | 17 (1.013) | B_U | 1 | cisd | 28.7 |

- **h6s and n2s.** On h6s CISD(H′) lies 54.495 mHa above E₀′ against E1lb − E₀′ = 104.576 mHa, so B_U
  certifies HF, which M3's rule E cannot (E_HF − E₀ = 282.21 mHa > Δ₁): rule B removes the measured iteration
  M3's certification needed, at one C_QPE more than M3's oracle T_E since D_U = 25.4 < t₁ (M3's certified
  1.132 and 1.091 at Ψ₁ used the record's exact energy and ignored the kernel tails). On n2s D_U = 28.7 ≥ t₁,
  rule B certifies HF at M3's oracle 3.956 C_QPE, and the spin-free classical step no longer pays (T_E(1) =
  4.111 > 3.956). The free loops keep their k_opt, and the benchmark reproduces M3's T6 rows with zero
  difference (§11).

**Conditions and model dependence.** Every certificate assumes E1lb = E₁′ − 0.1 mHa from the model spectrum of
H′ (N ≤ 8; no lower bound on E₁ is computed, as in M3, and N = 10 is not in U8); the costs keep M3's window
model (K₀ = 1) and (A3); n_v is designed for p₀ ≥ p_min = 1e-3 at ε_id = 1e-3, kernel and logical terms
budgeted apart (total ≤ 2e-3). With CISDT(H′) as the certificate (a sensitivity) the form at HF would change
on h4_r4.0 (B_L (2) → B_U (2)), h6s and n2v_r2.0 (B_U (2) → B_U (1)) and n2_r2.4, n2_r2.8, n2v_r2.4 and
n2v_r2.8 (B_L (2) → B_U (1)).

**Against the design.** The design expected rule B with θ = U + ΔE (CISD) or a 5-grid-unit top margin to be
sound to ε_id ≤ 1e-3 on the checked systems and on every N ≤ 8 system with Δ₁ ≥ 11 mHa, the literal threshold
E1lb − ΔE (tested as E₁′ − ΔE, 0.1 mHa higher) not to be, and B_L to bound the error with one run only for
p₀ ≥ 0.155 and with the verification run for every p₀ ≥ 3.4e-5. The run confirms this for exact states on all
19 systems (E₁′ − E₀′ = 16.983–711.493 mHa); it does not cover compiled stretched-N2 states, which populate
levels of other spin below θ, so that no design-floor bound applies to them (their kernel terms, contaminants
included, stay below 1e-3, at most 6.47e-4, and their totals within the 2e-3 budget, at most 1.15e-3). For
Q5(a) rule B accepts every exact state at M3's oracle cost plus one C_QPE per verification run (with a
certified U at HF on 10 of the 19 systems, by B_L without a certificate on the other nine) and changes M3's T6
on h6s and n2s only, lowering the measured loops' optima and raising h6s's others by one C_QPE; for Q5(b) the
sine and Kaiser kernels stay below 1e-3 on every exact state, the rect window and the literal threshold do
not.

## 8. QPE-side schemes (U9)

The QPE-side scheme moves the hand-off cost more than anything the CQE does. On stretched N2 the
E1lb-referenced step filter with exponential search (lt_bbht_L, F43), whose termination no certified U
guarantees there, is s* at HF, Ψ₁ and Ψ₂ in both active spaces and cuts the HF hand-off from M3's oracle rule
E, 3.939–12.8 C_QPE (which certifies none of these HF states; 3.939–13.8 under rule B), to 1.466–1.885
(CAS(6,6)), a factor 2.7–6.8 [derived]; on n2_r2.4 the chain is HF 8.928 → HF 1.705 → classical Ψ₁ 1.502
C_QPE. At equilibrium the cheap CISD-referenced filter (lt_bbht) saves 0.030–0.074 C_QPE at HF on h4, h6a, h8
and n2 (LiH keeps repetition), and Ψ₁ under repetition (1.006–1.033) is the optimum. On the six stretched
chains the E1lb-referenced forms do not terminate, and s* falls back to repetition or bbht with B_L (2s and
lt_bbht on h4_r3.5), above M3's uncertified oracle T_E on five of them. At s* the first measured iteration of
the exact spin-orbital R1 trajectory pays at r = 1 on h4_r3.5 (9.7%), h4_r4.0 (30.8%) and h6_r4.0 (17.4%), at
r = 3 also on h8_r4.0, n2_r2.4 and n2_r2.8; the second pays on no exact trajectory. Amplification also shrinks
what a better state is worth: an iteration of 1000 C_QPE that reaches the exact ground state pays below an HF
overlap of 0.000999 under repetition, but only below 1.9e-6 under known-overlap AA and 1.48e-7 under lt_bbht_L
(n2_r2.4; U9b).

**The HF hand-off under the schemes** (U9 part 1; `results/m4/handoff_summary.md`, "U9 (part 1)", 588 rows:
the exact and R1 records of both metrics and the compiled R1 states, Ψ₀–Ψ₃, with fp and lt_rep_L). T_s/C_QPE,
p on ψ₀′ of H′; "rep E" is M3's rule E (oracle T_E, no verification run); rep, aa and bbht use rule B's
acceptance (expected runs and certificate, F40); lt, lt_bbht and 2s need a certified U, and "(x oracle U)" is
their lower bound at U = E₀′ + 2ΔE where none is certified; aa and lt are oracles (lt in the overlap as well);
'-': no cost (not terminating). fp, which needs F15's p_lb from exact inputs, is s* on no row. s* carries
termination_not_certified where it is lt_bbht_L or 2s_L, and no_certificate with termination_not_certified on
h4_r4.0, h6_r3.5, h6_r4.0, h8_r3.0 and h8_r4.0. HF rows of the R1 spin-orbital record:

| system | p | rep E | rep | aa | bbht | lt | lt_bbht | 2s | lt_bbht_L | 2s_L | best |
|---|---|---|---|---|---|---|---|---|---|---|---|
| h4 | 0.9455 | 1.058 | 1.058 | 1.058 | 1.145 | 1.026 | 1.028 | 1.04 | 1.158 | 1.229 | lt_bbht |
| h6a | 0.9026 | 1.108 | 1.108 | 1.108 | 1.34 | 1.036 | 1.044 | 1.061 | 1.17 | 1.235 | lt_bbht |
| h8 | 0.8864 | 1.128 | 1.128 | 1.128 | 1.437 | 1.042 | 1.054 | 1.071 | 1.181 | 1.239 | lt_bbht |
| lih | 0.9744 | 1.026 | 1.026 | 1.026 | 1.058 | 1.099 | 1.103 | 1.164 | 1.136 | 1.218 | rep |
| n2 | 0.9257 | 1.08 | 1.08 | 1.08 | 1.224 | 1.02 | 1.023 | 1.033 | 1.16 | 1.234 | lt_bbht |
| n2_r1.6 | 0.6553 | 1.526 | 1.526 | 1.526 | 2.856 | 1.131 | 1.245 | 1.216 | 1.377 | 1.333 | 2s |
| n2s | 0.2539 | 3.939 | 3.939 | 3.001 | 3.514 | 1.692 | 1.81 | 2.494 | 1.466 | 1.863 | lt_bbht_L |
| n2_r2.4 | 0.1120 | 8.928 | 9.928 | 5.113 | 6.301 | 11.77 | 14.89 | 39.47 | 1.705 | 2.953 | lt_bbht_L |
| n2_r2.8 | 0.0781 | 12.8 | 13.8 | 6.121 | 7.633 | - (1.283 oracle U) | - (1.366 oracle U) | - (2.156 oracle U) | 1.885 | 3.799 | lt_bbht_L |
| n2v_r2.0 | 0.2453 | 4.077 | 5.077 | 4.001 | 4.557 | 1.872 | 2.033 | 2.964 | 1.469 | 1.893 | lt_bbht_L |
| n2v_r2.4 | 0.1111 | 9.001 | 10 | 5.135 | 6.327 | - (1.236 oracle U) | - (1.305 oracle U) | - (1.859 oracle U) | 1.702 | 2.97 | lt_bbht_L |
| n2v_r2.8 | 0.0782 | 12.78 | 13.78 | 6.119 | 7.628 | - (1.279 oracle U) | - (1.361 oracle U) | - (2.156 oracle U) | 1.874 | 3.8 | lt_bbht_L |
| h6s | 0.5733 | 1.744 | 2.744 | 2.744 | 4.071 | 1.449 | 1.79 | 1.75 | 1.395 | 1.376 | 2s_L |
| h4_r3.5 | 0.5535 | 1.807 | 2.807 | 2.807 | 4.103 | 1.934 | 2.605 | 2.463 | - | - | 2s |
| h4_r4.0 | 0.4341 | 2.304 | 3.304 | 3.304 | 4.185 | - (2.227 oracle U) | - (2.696 oracle U) | - (2.917 oracle U) | - | - | rep |
| h6_r3.5 | 0.3961 | 2.525 | 3.525 | 3.525 | 4.201 | - (1.69 oracle U) | - (1.874 oracle U) | - (2.147 oracle U) | - | - | rep |
| h6_r4.0 | 0.2720 | 3.676 | 4.676 | 4.017 | 4.435 | - (3.017 oracle U) | - (3.297 oracle U) | - (5.096 oracle U) | - | - | bbht |
| h8_r3.0 | 0.4649 | 2.151 | 3.151 | 3.151 | 4.175 | - (1.339 oracle U) | - (1.5 oracle U) | - (1.571 oracle U) | - | - | rep |
| h8_r4.0 | 0.1685 | 5.935 | 6.935 | 4.291 | 5.211 | - (3.828 oracle U) | - (4.618 oracle U) | - (9.573 oracle U) | - | - | bbht |

- **Stretched N2.** lt_bbht_L costs E_BBHT(p) calls of a filter of c_filt(w_L) = 0.1318–0.1335 C_QPE plus one
  QPE, against 3.939–13.8 C_QPE under repetition; oracle AA (3.001–6.121) and bbht (3.514–7.633) are dearer.
  The U-referenced filters do worse at the CISD certificate, which on n2_r2.4 passes only 5.077 mHa below E1lb
  [derived]; their oracle-U bounds (1.236–1.366 C_QPE on n2_r2.8 and n2v_r2.4–2.8) show what a certificate
  near E₀′ would be worth, and with CISDT(H′) (a sensitivity) s* at HF becomes lt_bbht at 1.318 (n2_r2.4) and
  1.366 (n2_r2.8).
- **Equilibrium, h6s, n2_r1.6.** CISD(H′) lies 1.037–7.970 mHa above E₀′ on h4, h6a, h8 and n2, so lt_bbht is
  s* at 1.023–1.054 C_QPE against 1.058–1.128 under repetition. lih keeps repetition (1.026): at p_HF = 0.9744
  exponential search overshoots (E_BBHT(p) > 1/p above p = 0.299), and its CISD-referenced filter, whose band
  from U (0.013 mHa above E₀′) to E1lb is only 132.955 mHa wide, needs degree 944 (c_filt(U) = 0.09693 C_QPE,
  against 0.01845–0.03733 on h4, h6a, h8 and n2) [derived from the QPE-side and filter tables]. On h6s and
  n2_r1.6, 2s_L (1.376) and 2s (1.216) are s*.
- **Stretched chains.** Without a certified U and with no terminating L form, s* is repetition or bbht with
  B_L's two runs (no_certificate, termination_not_certified), 2s on h4_r3.5; it costs more than M3's
  uncertified oracle rule E on five of the six and less only on h8_r4.0 (5.211 against 5.935). The oracle-U
  bounds of the realizable lt_bbht (1.5–4.618 C_QPE) would undercut s* by up to 1.651 C_QPE, and those of lt,
  which also needs the overlap, by up to 1.835 [derived]: a certificate near E₀′ is the missing input.

**Filter and coarse-QPE costs** (F43, ε_f = 1e-2; `handoff_summary.md`, "Filter and coarse-QPE costs"). Where
the L forms terminate, the w_L filter costs c_filt = 0.1258–0.1378 C_QPE and the coarse QPE c_c =
0.2118–0.2192 C_QPE (degrees 594–2479); both scale as ΔE/(E1lb − U) and do not grow with the system at fixed
w_L. The U-referenced filters follow the distance from U to E1lb, from c_filt(U) = 0.01845 C_QPE on n2 (degree
134) to 2.619 on n2_r2.4 (degree 14807). The filter count is conservative by up to 2× against (A3), which
biases the model against the filtered schemes (F43).

**The first measured iteration at s\* on exact trajectories** (U9 part 1, `handoff_summary.md`, "Savings of a
measured iteration", 462 rows; `summary.md`, CISDT sensitivity). s* with T_s* (C_QPE) at HF, at the classical
Ψ₁ and at the unit-step Ψ₂ of the R1 spin-orbital record for C_iter(1) at r = 1 and 3; the saving of the first
measured iteration at s*, [min_s T_s(1) − min_s T_s(2)]/min_s T_s(1), for that record, for the spin-free R1
record (the compiled states' savings are in the full table); positive where the iteration pays (F44).

| system | s* at HF | s* at Ψ₁ | s* at Ψ₂, r = 1 | so, r = 1 | s* at Ψ₂, r = 3 | so, r = 3 | sf, r = 1 | sf, r = 3 |
|---|---|---|---|---|---|---|---|---|
| h4 | lt_bbht 1.028 | rep 1.009 | rep 1.477 | -46.4% | rep 1.055 | -4.59% | -11.1% | 0.0163% |
| h6a | lt_bbht 1.044 | rep 1.023 | rep 10.2 | -897% | rep 2.031 | -98.5% | -274% | -28.9% |
| h8 | lt_bbht 1.054 | rep 1.033 | rep 126.9 | -1.22e4% | rep 15 | -1.35e3% | -3.99e3% | -442% |
| lih | rep 1.026 | rep 1.006 | rep 12.76 | -1.17e3% | rep 2.311 | -130% | -1.14e3% | -126% |
| n2 | lt_bbht 1.023 | rep 1.01 | rep 1.793 | -77.5% | rep 1.091 | -7.97% | -33.2% | -3.65% |
| n2_r1.6 | 2s 1.216 | lt_bbht 1.159 | lt_bbht 1.725 | -48.8% | lt_bbht 1.195 | -3.12% | -27.4% | -2.52% |
| n2s | lt_bbht_L 1.466 | lt_bbht_L 1.425 | lt_bbht_L 1.936 | -35.9% | lt_bbht_L 1.478 | -3.77% | -25.4% | -2.26% |
| n2_r2.4 | lt_bbht_L 1.705 | lt_bbht_L 1.502 | lt_bbht_L 1.876 | -24.9% | lt_bbht_L 1.5 | 0.0997% | -18.5% | -0.75% |
| n2_r2.8 | lt_bbht_L 1.885 | lt_bbht_L 1.569 | lt_bbht_L 1.886 | -20.2% | lt_bbht_L 1.532 | 2.36% | -16% | -0.416% |
| n2v_r2.0 | lt_bbht_L 1.469 | lt_bbht_L 1.423 | lt_bbht_L 5.107 | -259% | lt_bbht_L 1.829 | -28.6% | -189% | -20.5% |
| n2v_r2.4 | lt_bbht_L 1.702 | lt_bbht_L 1.5 | lt_bbht_L 5.11 | -241% | lt_bbht_L 1.859 | -24% | -164% | -17% |
| n2v_r2.8 | lt_bbht_L 1.874 | lt_bbht_L 1.564 | lt_bbht_L 4.96 | -217% | lt_bbht_L 1.873 | -19.8% | -143% | -14.7% |
| h6s | 2s_L 1.376 | lt_bbht_L 1.163 | lt_bbht_L 4.428 | -281% | lt_bbht_L 1.515 | -30.2% | -149% | -13.5% |
| h4_r3.5 | 2s 2.463 | lt_bbht 1.792 | lt_bbht 1.618 | 9.7% | lt_bbht 1.569 | 12.4% | 1.96% | 5.41% |
| h4_r4.0 | rep 3.304 | rep 3.002 | rep 2.076 | 30.8% | rep 2.06 | 31.4% | 13.4% | 14.5% |
| h6_r3.5 | rep 3.525 | rep 2.229 | rep 3.971 | -78.1% | rep 2.304 | -3.36% | -94.7% | -7.52% |
| h6_r4.0 | bbht 4.435 | rep 3.471 | rep 2.866 | 17.4% | rep 2.222 | 36% | -47.3% | 7.46% |
| h8_r3.0 | rep 3.151 | rep 2.155 | rep 43.32 | -1.91e3% | rep 6.712 | -211% | -1.21e3% | -130% |
| h8_r4.0 | bbht 5.211 | rep 4.116 | rep 11.91 | -189% | rep 3.326 | 19.2% | -841% | -80.2% |

- **Where it pays.** At r = 1 on h4_r3.5 (both metrics; lt_bbht at a certified CISD U), h4_r4.0 (both metrics)
  and h6_r4.0 (spin-orbital only), the last two under repetition with B_L; at r = 3 also on h6_r4.0 spin-free
  (7.46%), h8_r4.0 (19.2%, the only N = 8 case), n2_r2.4 (0.0997%) and n2_r2.8 (2.36%) under lt_bbht_L and,
  spin-free, on h4 by 1.66e-4 C_QPE (0.0163%), the only equilibrium case. At M3's r = 0.3 only h4_r4.0 pays,
  with M3's line-search costs nothing does, and the second iteration pays on no record. Compiled one-slice
  states pay on h4_r3.5, h4_r4.0 and (r = 3) n2_r2.8 at smaller savings, not on h6_r4.0, h8_r4.0 or n2_r2.4
  (§2). Admitting lt_rep flips h4_r3.5 spin-free and compiled at r = 1 (to −2.23%, −2.4%); CISDT(H′) flips the
  r = 3 verdicts of n2_r2.4 and n2_r2.8 (to −2.06%, −0.883%).
- **The classical step.** The spin-orbital classical step pays at s* on all 19 systems (it lowers T_s* by
  0.013–1.296 C_QPE [derived]); the spin-free one does not on stretched N2 in either active space, where its
  Ψ₁ lowers the overlap (n2_r2.4: p = 0.0961 against 0.1120) and costs 1.473–1.999 C_QPE at s* against
  1.466–1.885 for HF, M3's spin-free n2s loss (M3 T7).
- **What each lever saves** (n2_r2.4, R1 so, r = 1): the scheme cuts the HF hand-off from 8.928 to 1.705
  C_QPE, the classical step saves 0.203 more, and the measured iteration then loses 0.374 at r = 1 and gains
  0.0015 at r = 3 [derived]. The ordering scheme > classical step > measured iteration holds on N2 at every
  bond length, on h6s, n2_r1.6 and at equilibrium on h4, h6a, h8 and n2; it fails on lih and on the six
  stretched chains, where the classical step and, on h4_r3.5, h4_r4.0 and h6_r4.0, the measured iteration
  carry the savings.

**The closed-loop states** (U9 part 2; `summary.md`, "U9 (part 2)", with every scheme, configuration and later
iterate; mean T_s over the replicates with 95% intervals). p̃ on ψ₀ of H, CISD(H′) the only certificate, rep,
aa and bbht with rule B's verification runs. Each system's deciding cell (lagged r = 1, direct_mg, unit steps,
exact propagation; 100, 20 and 10 replicates as in §6): s* with its mean T at HF, Ψ₁ and the realized Ψ̃₂, the
paired saving of the first measured iteration at s*, and the optimistic oracle r = 3 cell at s*. Savings under
repetition are in U6 and U7.

| system | median p̃₂ | HF at s* | Ψ₁ at s* | Ψ̃₂ at s* | saving at s*, k = 1 | pays | oracle r = 3 at s*, k = 1 |
|---|---|---|---|---|---|---|---|
| n2_r2.0 | 0.459 | lt_bbht_L 1.467 [1.467, 1.467] | lt_bbht_L 1.425 [1.425, 1.425] | lt_bbht_L 1.479 [1.478, 1.479] | -0.054 [-0.0545, -0.0535] | no | -0.0541 [-0.0546, -0.0536] |
| n2_r2.4 | 0.291 | lt_bbht_L 1.704 [1.704, 1.704] | lt_bbht_L 1.501 [1.501, 1.501] | lt_bbht_L 1.512 [1.508, 1.516] | -0.0111 [-0.0153, -0.007] | no | -0.00106 [-0.00538, 0.00307] |
| n2_r2.8 | 0.241 | lt_bbht_L 1.883 [1.883, 1.883] | lt_bbht_L 1.568 [1.568, 1.568] | lt_bbht_L 1.547 [1.541, 1.553] | 0.0212 [0.0151, 0.027] | yes | 0.0386 [0.0324, 0.0445] |
| h4_r4.0 | 0.936 | rep 3.304 [3.304, 3.304] | rep 3.002 [3.002, 3.002] | rep 2.159 [2.156, 2.161] | 0.844 [0.841, 0.846] | yes | -5.19 [-10.1, -1.93] |
| h4_r1.8 | 0.994 | lt_bbht 1.028 [1.028, 1.028] | rep 1.009 [1.009, 1.009] | rep 1.273 [1.272, 1.275] | -0.264 [-0.266, -0.263] | no | -0.0706 [-0.0719, -0.0696] |
| n2_r1.098 | 0.988 | lt_bbht 1.023 [1.023, 1.023] | rep 1.01 [1.01, 1.01] | rep 1.179 [1.173, 1.187] | -0.169 [-0.177, -0.163] | no | -0.0982 [-0.1, -0.0972] |
| h6_r3.0 | 0.933 | 2s_L 1.376 [1.376, 1.376] | lt_bbht_L 1.163 [1.163, 1.163] | lt_bbht_L 5.531 [5.53, 5.533] | -4.37 [-4.37, -4.37] | no | -0.458 [-0.49, -0.429] |
| h8_r3.0 | 0.878 | rep 3.151 [3.151, 3.151] | rep 2.155 [2.155, 2.155] | rep 69.69 [69.69, 69.7] | -67.5 [-67.5, -67.5] | no | -4.78 [-4.86, -4.72] |
| n2v_r2.4 | 0.286 | lt_bbht_L 1.702 [1.702, 1.702] | lt_bbht_L 1.5 [1.5, 1.5] | lt_bbht_L 1.969 [1.964, 1.973] | -0.469 [-0.473, -0.465] | no | -0.354 [-0.358, -0.35] |
| n2v_r2.8 | 0.228 | lt_bbht_L 1.874 [1.874, 1.874] | lt_bbht_L 1.564 [1.564, 1.564] | lt_bbht_L 2.002 [1.992, 2.011] | -0.438 [-0.447, -0.428] | no | -0.3 [-0.309, -0.29] |

- **The first iteration at s*** pays in the deciding cells only on n2_r2.8 (lt_bbht_L, termination not
  certified) and h4_r4.0 (repetition with B_L), and misses on n2_r2.0, n2_r2.4 and every control; the second
  loses at s* in every deciding cell, by 0.686 (h4_r4.0) to 425 C_QPE (h8_r3.0). Outside the deciding cells it
  pays at s* in some sensitivity and compiled cells of n2_r2.4, n2_r2.8 and h4_r4.0 (`summary.md`), never on
  n2_r2.0 or in the step-rule, route or spin-free cells. The per-scheme oracle r = 3 savings, the design's
  optimistic case, are evaluated in §6. On N2 CAS(10,8) the iteration loses at s* (−31.26% and −28.01% of
  T_s*(Ψ₁) [derived]), as on the exact trajectories (−241% and −217% at r = 1). The n2_r2.4 chain (8.892,
  1.704, 1.501, 1.512) follows U9 part 1 to the difference between p on H and on H′.

**Break-even overlap** (U9b, F44; `handoff_summary.md`, "U9b", 306 rows). p₀* is the HF overlap below which an
iteration of cost C_it, lifting an HF-like state to (p₁, C(Ψ₁)), pays under scheme s: against the perfect
state (p₁ = 1, c₁ = 0), the most a state can be worth, and against the R1 Ψ₂, the state the first measured
iteration reaches. n2_r2.4, C_it in C_QPE:

| scheme | perfect, 0.1 | perfect, 1 | perfect, 10 | perfect, 100 | perfect, 1000 | Ψ₂, 0.1 | Ψ₂, 1 | Ψ₂, 10 | Ψ₂, 100 | Ψ₂, 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| rep | 0.909 | 0.5 | 0.0909 | 0.0099 | 0.000999 | 0.273 | 0.219 | 0.0738 | 0.00966 | 0.000996 |
| aa | 0.909 | 0.5 | 0.0157 | 0.000187 | 1.9e-6 | 0.192 | 0.115 | 0.0112 | 0.000179 | 1.89e-6 |
| bbht | 0.959 | 0.809 | 0.0359 | 0.000776 | 8.39e-6 | 0.256 | 0.155 | 0.0264 | 0.000742 | 8.35e-6 |
| fp | 1 | 1 | 0.28 | 0.00286 | 2.81e-5 | 0.28 | 0.28 | 0.0739 | 0.00236 | 2.76e-5 |
| lt | 0.963 | 0.724 | 0.0891 | 0.00124 | 1.3e-5 | 0.207 | 0.155 | 0.0414 | 0.00112 | 1.29e-5 |
| lt_bbht | 0.983 | 0.895 | 0.132 | 0.00441 | 5.7e-5 | 0.271 | 0.211 | 0.0685 | 0.00397 | 5.63e-5 |
| 2s | 0.977 | 0.812 | 0.301 | 0.0413 | 0.00429 | 0.279 | 0.264 | 0.17 | 0.0374 | 0.00424 |
| lt_bbht_L | 0.842 | 0.0531 | 0.00132 | 1.48e-5 | 1.48e-7 | 0.173 | 0.0363 | 0.00125 | 1.47e-5 | 1.48e-7 |
| 2s_L | 0.686 | 0.179 | 0.0214 | 0.00218 | 0.000219 | 0.249 | 0.123 | 0.0203 | 0.00217 | 0.000219 |

- **Limits.** Under repetition p₀* = 1/(1 + C_it/C_QPE) against a perfect state, M3's p₀* = C_QPE/T_E scaling;
  F44's limits for C_it ≫ C_QPE hold, (1.38C_QPE/C_it)² for known-overlap AA and (κ_B C_filt/C_it)² for
  lt_bbht_L. The cheaper a scheme makes the hand-off, the cheaper a measured iteration must be: on n2_r2.4 the
  R1 Ψ₂ pays from the classical Ψ₁ under repetition at C_it = 1 (p₀* = 0.219) but under lt_bbht_L not even at
  0.1 (0.173), matching the verdicts above. M3 found that a measured iteration of 154–1366 C_QPE would pay
  under rule E only below p₀*(1) = 7.3e-4 to 6.5e-3 (M3 T6); under the amplified schemes the same cost needs a
  starting overlap 119–6750 times smaller [derived: 8.39e-6 (bbht), 1.9e-6 (aa), 1.48e-7 (lt_bbht_L) against
  0.000999 at 1000 C_QPE].

**Conditions and model dependence.** Costs use M3's window model (a call succeeds with a = p; comparator and
reflections neglected, F43 [estimate]); an AA sequence is laid out as one coherent run (F40, F19, U12),
conservative for exponential search, which measures between its stages. The filter cost includes GQSP
rotations that add 5.1–5.2% (§13). BBHT's schedule (growth 6/5) overshoots 1/p above p = 0.299, where
filter-then-repeat would undercut it but never decides. Compiled rows carry the exact states' loop costs and
F41's contaminant term; E1lb, (A3) and p_min are as in §7.

**Against the design.** The design expected the QPE-side scheme to give the largest saving and the free
classical step the next (n2_r2.4: HF 8.93 → HF 1.67 → Ψ₁ 1.47 C_QPE), and the E1lb-referenced forms not to
terminate on chains with Δ₁ < w_L. Both are confirmed (8.928 → 1.705 → 1.502, 2.4% and 1.9% above the design's
1.665 and 1.473 [derived], about the filter rotations its scheme table left out, §13; the fallback holds on
the six stretched chains), but the ordering itself fails on lih and on those chains. The expectation that no
measured iteration pays on the chains at N ≤ 10 is refuted, on hand-offs whose termination is not certified
except on h4_r3.5 and h4. For Q5(c), handing off at s* costs, for HF and Ψ₁: 1.023–1.054 and 1.006–1.033 C_QPE
at equilibrium, 1.466–1.885 and 1.423–1.569 on stretched N2, 2.463–5.211 and 1.792–4.116 on the stretched
chains; the realized Ψ̃₂ of the deciding closed-loop cells costs 1.179–1.273 at equilibrium, 1.479–2.002 on
stretched N2 and 2.159 on h4_r4.0.

## 9. Properties (U10, U11)

Which estimator of a ground-state property is cheapest depends first on whether the CQE loop is free. With a
free loop route A, amplitude estimation on the CQE iterate itself, is the cheapest route on 37 of the 66
(system, target) pairs at the tighter accuracy and on 50 at the looser one, and wherever its exact bias floor
reaches the target it beats route B on every cell, by 4.99–300× at the oracle k. With a measured loop the
QPE-projected copy from a later iterate (Bk) is the cheapest route in Toffolis (F53) on 53 of the 66 pairs at
the tighter accuracy (r = 1 and 0.3 alike); route A keeps 11 (r = 1) and 9 (r = 0.3), mostly H4 and LiH's
two-body targets, and at any single pre-declared measured budget at most 5. As single coherent runs most of
these Bk rows do not fit M3's machine model: 37 of the 135 best tighter Bk rows have no surface-code layout
and 75 of the 98 laid out run longer than 7 days (route A: 3 of 57; §10), so that only 23 meet M3's 7-day
deadline, although among the laid-out rows Bk still wins 40 pairs at either measured r and B0 2 more at
r = 0.3 (U0, Q6). Route A is bias-limited (exact floor over 60 R1 iterations above ε) on 23 pairs at the
tighter accuracy (stretched H6 except its cumulant norm, N2 at 2.0 and 2.4 Å, N2's cumulant norm at
equilibrium) and on its compiled iterates on 26 more. Hellmann–Feynman QPE gives the LiH dipole for 99.6 (4e-3
a.u.) and 970 C_QPE (1e-3 a.u.), 16–58× below AE on filtered copies and above route A with a free loop (34.5
and 221); at 1e-3 a.u. HF-QPE and route B rely on charging the exact model bias of H′ (b_fact = 4.715e-4 a.u.,
about half the target): with F47's bound instead, the measured-loop cells fall to route A at 8.05e5 (r = 1)
and 2.68e6 C_QPE (r = 0.3). No route-A estimate carries an error certificate: its usability at a given k is a
fact of the emulation (F49), which is why U0 also states the comparison at each pre-declared budget.

**Conditions.** The property set (N = 4–6; `properties_summary.md`, "Systems"). Targets and accuracies (F46):
NONs (max norm) and the 1-RDM at 1e-2 and 1e-3, the 2-RDM and the cumulant at 0.1 and 1e-2, the cumulant norm
at 1e-2 and 1e-3, LiH's dipole at 4e-3 and 1e-3 a.u.: 66 pairs, 132 cells, "looser" and "tighter". Route A
(F49) runs on the R1 spin-orbital record (60 steps) at budgets k ∈ {2, 5, 10, 20, 60} fixed before the run,
with a free loop and measured loops at r = 1 and 0.3. Route B (F50) filters HF or Ψ₁ (B0) or the best iterate
with its loop cost (Bk) into ψ₀′ (F43's step filter) and takes the cheaper coherent copy, the ratio estimator
or exact AA on an estimated a; HF-QPE (F51) serves the dipole. AE at κ = 1 is the reference estimator (shadows
reported, gradient estimation never selected, F48). Route B and HF-QPE charge the exact model bias b_fact of
their H′, on F52's property-aware encoding where F9's is too biased; costs are in F9's C_QPE units.

**U10b** (routes compared, selected rows; `results/m4/properties_summary.md`, U10b, has every cell with the
route-B encoding, shadow, compiled and layout columns, and U10a the fixed-budget rows). Route-A minimum bias
over k ≤ 60 on R1 exact / compiled (one slice) / compiled (F25 slices on steps 0–2); B0 and Bk with AE
[start]; HF-QPE (scalars); and the best selectable row per loop, route [k] and cost (AE).

| system | target | eps | A min bias (R1 / s=1 / sliced) | B0 AE [start] | Bk AE free | Bk AE meas r=1 | HF-QPE | best free (AE) | best meas r=1 (AE) | best meas r=0.3 (AE) |
|---|---|---|---|---|---|---|---|---|---|---|
| h4_r1.8 | rdm1 | 0.001 | 1.5e-5 / 7.9e-4 / 7.9e-4 | 1.15e4 [1] | 5.46e3 [6] | 6.47e3 [6] | - | A [60] 656 | Bk [6] 6.47e3 | Bk [5] 8.48e3 |
| h4_r4.0 | rdm1 | 0.001 | 1.6e-5 / 3.9e-2 / 1.8e-2 | inf [0] | 1.96e5 [31] | 3.57e5 [30] | - | A [60] 1.4e3 | Bk [30] 3.57e5 | Bk [18] 4.41e5 |
| h6a | rdm1 | 0.001 | 1.8e-4 / 1.4e-3 / 1.4e-3 | 6.08e4 [1] | 3.37e4 [37] | 4.57e4 [4] | - | A [60] 1.05e4 | Bk [4] 4.57e4 | Bk [3] 5.11e4 |
| h6_r3.0 | rdm1 | 0.001 | 3.6e-3 / 5.9e-2 / 1.8e-2 | 2.71e5 [1] | 1.67e5 [46] | 2.5e5 [4] | - | Bk [46] 1.67e5 | Bk [4] 2.5e5 | Bk [3] 2.51e5 |
| h6_r4.0 | rdm2 | 0.01 | 2.1e-1 / 3.2e0 / 3.6e-1 | inf [0] | 2.75e7 [60] | 3.83e7 [41] | - | Bk [60] 2.75e7 | Bk [41] 3.83e7 | Bk [17] 4.31e7 |
| lih | rdm2 | 0.01 | 5.7e-5 / 2.1e-3 / 2.1e-3 | 1.52e6 [1] | 7e5 [15] | 7.31e5 [10] | - | A [20] 9.84e3 | A [10] 7.93e4 | A [10] 1.42e5 |
| lih | dipole | 0.004 | 1.4e-6 / 7.0e-5 / 7.0e-5 | 5.77e3 [1] | 3.41e3 [10] | 5.77e3 [1] | 99.6 | A [10] 34.5 | HF-QPE [1] 99.7 | HF-QPE [1] 99.7 |
| lih | dipole | 0.001 | 1.4e-6 / 7.0e-5 / 7.0e-5 | 3.41e4 [1] | 1.56e4 [22] | 3.22e4 [7] | 970 | A [20] 221 | HF-QPE [1] 970 | HF-QPE [1] 970 |
| n2_r1.098 | rdm1 | 0.001 | 5.1e-4 / 1.2e-3 / 1.2e-3 | 7.24e3 [1] | 4.76e3 [4] | 4.81e3 [4] | - | Bk [4] 4.76e3 | Bk [4] 4.81e3 | Bk [4] 4.95e3 |
| n2_r2.0 | occupations | 0.001 | 1.7e-2 / 2.6e-2 / 2.6e-2 | 8.23e4 [1] | 2.27e4 [49] | 2.63e4 [13] | - | Bk [49] 2.27e4 | Bk [13] 2.63e4 | Bk [13] 3.05e4 |
| n2_r2.4 | rdm1 | 0.001 | 1.1e-2 / 6.8e-2 / 6.8e-2 | 4.22e5 [1] | 6.46e4 [40] | 7.56e4 [25] | - | Bk [40] 6.46e4 | Bk [25] 7.56e4 | Bk [25] 8.64e4 |

**U0's Q6 tally** (`summary.md`, "U0 Q6"; 66 pairs, AE at κ = 1): the routes of the best rows with route A at
one budget k for every pair, and at its cheapest usable budget per pair (the record, which chooses k with the
exact bias). On the compiled iterates route A is the cheapest on 13, 5 and 4 pairs at the tighter accuracy
with the free, r = 1 and r = 0.3 loops (full tally in `summary.md`).

| accuracy | loop | k = 2 | k = 5 | k = 10 | k = 20 | k = 60 | cheapest usable k (record) |
|---|---|---|---|---|---|---|---|
| tighter | free | Bk 65, HF-QPE 1 | Bk 62, A 3, HF-QPE 1 | Bk 51, A 14, HF-QPE 1 | Bk 38, A 28 | A 37, Bk 29 | A 37, Bk 29 |
| tighter | meas_r1 | Bk 64, B0 1, HF-QPE 1 | Bk 61, A 3, B0 1, HF-QPE 1 | Bk 59, A 5, B0 1, HF-QPE 1 | Bk 60, A 4, B0 1, HF-QPE 1 | Bk 64, B0 1, HF-QPE 1 | Bk 53, A 11, B0 1, HF-QPE 1 |
| tighter | meas_r03 | Bk 62, B0 3, HF-QPE 1 | Bk 59, A 3, B0 3, HF-QPE 1 | Bk 59, A 3, B0 3, HF-QPE 1 | Bk 59, A 3, B0 3, HF-QPE 1 | Bk 62, B0 3, HF-QPE 1 | Bk 53, A 9, B0 3, HF-QPE 1 |
| looser | free | Bk 62, B0 3, HF-QPE 1 | Bk 49, A 16, HF-QPE 1 | A 33, Bk 33 | A 38, Bk 28 | A 50, Bk 16 | A 50, Bk 16 |
| looser | meas_r1 | Bk 57, B0 8, HF-QPE 1 | Bk 48, A 10, B0 7, HF-QPE 1 | Bk 51, B0 8, A 6, HF-QPE 1 | Bk 57, B0 8, HF-QPE 1 | Bk 57, B0 8, HF-QPE 1 | Bk 43, A 15, B0 7, HF-QPE 1 |
| looser | meas_r03 | Bk 54, B0 11, HF-QPE 1 | Bk 49, B0 9, A 7, HF-QPE 1 | Bk 53, B0 11, A 1, HF-QPE 1 | Bk 54, B0 11, HF-QPE 1 | Bk 54, B0 11, HF-QPE 1 | Bk 48, B0 9, A 8, HF-QPE 1 |

- **Free loop.** The exact R1 floor reaches ε on 96 of the 132 cells; there route B costs 4.99 (h6a 1-RDM at
  1e-3) to 300 times (LiH NONs at 1e-2) the A-oracle row, median 56, and 3.0–186 times route A at its cheapest
  usable pre-declared budget [derived from `properties_<s>.json`]. At the tighter accuracy the floor reaches ε
  on 43 pairs, 6 of them only between the pre-declared budgets (n2_r1.098's 1-RDM, 2-RDM and cumulant, which
  hover at ε on the record's energy plateau, and h6_r3.0–4.0's cumulant norm), where Bk wins; hence route A's
  37, which k = 60 reproduces.
- **Measured loops.** A measured loop costs Σ_{j<k} T_res(j), and T_res(j) grows as the residual norm falls,
  since it is measured to the relative accuracy r (LiH NONs at 1e-2, r = 1: route A 2.64e3 C_QPE at k = 5,
  9.33e7 at k = 60; U10a). Route B needs the loop only to the start it filters, at overlaps of 0.75 to
  0.999999 (median 0.987 over the 242 best Bk rows). Route A survives a measured r = 1 loop on 11 tighter
  pairs (8 of the 15 of H4 at 1.8–3.5 bohr, LiH's 2-RDM and cumulant, n2_r1.098's NONs; 229–7.93e4 C_QPE);
  with one budget for every pair it is the cheapest on 0 / 3 / 5 / 4 / 0 pairs (r = 1) at k = 2 / 5 / 10 / 20
  / 60: a realizable route A must fix k without knowing its bias.
- **Route B's copies.** Every best route-B row (262) uses the ratio estimator; exact AA is the cheaper copy
  only for B0 starts of n2_r2.0 and n2_r2.4 (54 rows), where Bk is cheaper still by 1.88–7.79×. On h4_r4.0,
  h6_r3.5 and h6_r4.0, where Δ₁ < w_L and CISD(H′) lies above E1lb, no copy from HF or Ψ₁ terminates, and
  route B exists only from an iterate whose own certificate lies below E1lb.
- **Hellmann–Feynman QPE.** On LiH dE/dh = −1.817719 a.u. against ⟨μ⟩ = −1.817716 at ψ₀. With b_fact =
  4.715e-4 a.u. added linearly the stencils need b = 2.000e-3 (4e-3) and 2.025e-4 (1e-3), and the field steps
  are refined to 0.9999b and 0.9996b of stencil bias (`properties_summary.md`, "Finite field"); the rows cost
  99.55 and 969.9 C_QPE with the free loop and 99.74 and 970.2 with a measured one (lt_bbht from Ψ₁).
- **Shadows and encodings.** The crossover budget at which matchgate shadows and AE need the same preparations
  (B*, F48) is 0.224–2.5, above every accuracy, so AE needs fewer preparations than matchgate shadows, which
  win no cell. F9's encoding is kept on 120 of the 132 cells; n2_r2.0's and n2_r2.4's tighter cells run on
  F52's SF L = 18 (1.0258 C_QPE) and L = 17 (1.0121), without which nine of the ten would have no route, and
  LiH's looser NONs and cumulant on the cheaper L = 15 (0.9502).
- **Uncertainty and model dependence.** U10 is deterministic and rests on models. Route A's usability is
  chosen per pair with the exact bias, which no hardware run can do. Route B and HF-QPE charge the exact
  b_fact; with F47's bound at q = 1 − F instead (a sensitivity) the best row changes on 11 tighter cells (8 of
  LiH, e.g. the dipole with a measured loop, HF-QPE 970 → A 8.05e5 at r = 1; n2_r2.0's 1-RDM, Bk → none), 130
  more keep their route at up to 3.92× the cost, and at the looser accuracy n2_r2.4's NONs, 1-RDM, 2-RDM and
  cumulant lose their AE route with every loop (12 cells) [derived from `properties_n2_r2.4.json`]. Further
  assumptions: a = p, overlaps on H for H′, estimator inputs at ψ₀, AE at κ = 1, RMS targets (F46, F50). 39 of
  the 198 best tighter AE rows have no surface-code layout (§10) but stay selectable, since F53 compares
  Toffolis.

U11 locates the limit of route A. On the exact R1 record its floors lie at 9.0e-7 to 1.1e-4 on the H4 chains
and at 9.9e-7 to 1.0e-3 on h6_r1.8, h6a and lih, but at 3.1e-4 to 0.46 on stretched H6 and N2, whose 60-step
records end 0.053 to 10.2 mHa above E₀, and at 3.2e-4 to 9.0e-3 on N2 at equilibrium, whose records reach an
energy plateau 0.18 mHa above E₀. Compiled circuits raise the floors 6.6–506 times on h4_r1.8, h6_r1.8, h6a
and LiH (1.7–2.5 times on N2 at equilibrium, whose exact floor sits on the plateau), a finite-shot closed loop
7.5–123 times on h4_r1.8 and LiH. F47 bounds every error on the exact records and on the closed-loop iterates.

**U11** (bias floors, the 1-RDM rows of eight systems; `properties_summary.md`, U11, has every target, the
all-orders cumulant-norm ratios and the F25 slices). Minimum property error over k ≤ 60 [its k] and k_ε (the
first k after which the error stays ≤ ε) per record; F47: the smallest bound/|bias| of the q and energy forms
on R1 / exact line search (≥ 1: the bound holds on every iterate).

| system | target | R1 exact | M3 exact line search | compiled s = 1 | compiled sliced | F47 min ratio (q, energy): R1 / LS |
|---|---|---|---|---|---|---|
| h4_r1.8 | rdm1 | 1.5e-5 [28] (0.01: 5, 0.001: 23) | 3.3e-7 [16] (0.01: 4, 0.001: 8) | 7.9e-4 [25] (0.01: 5, 0.001: 23) | 7.9e-4 [25] (0.01: 5, 0.001: 23) | 1.68, 2.29 / 1.61, 2.14 |
| h4_r4.0 | rdm1 | 1.6e-5 [36] (0.01: 28, 0.001: 30) | 2.8e-7 [27] (0.01: 18, 0.001: 21) | 3.9e-2 [15] (0.01: >60, 0.001: >60) | 1.8e-2 [18] (0.01: >60, 0.001: >60) | 1.81, 7.89 / 1.81, 7.21 |
| h6_r1.8 | rdm1 | 1.6e-4 [49] (0.01: 7, 0.001: 36) | 7.6e-5 [48] (0.01: 6, 0.001: 33) | 1.1e-3 [49] (0.01: 7, 0.001: >60) | 1.1e-3 [49] (0.01: 7, 0.001: >60) | 4.23, 5.43 / 4.11, 5.44 |
| h6_r3.0 | rdm1 | 3.6e-3 [52] (0.01: 32, 0.001: >60) | 2.8e-3 [60] (0.01: 28, 0.001: >60) | 5.9e-2 [38] (0.01: >60, 0.001: >60) | 1.8e-2 [57] (0.01: >60, 0.001: >60) | 4.26, 6.68 / 4.6, 6.74 |
| h6_r4.0 | rdm1 | 2.6e-2 [60] (0.01: >60, 0.001: >60) | 1.4e-2 [60] (0.01: >60, 0.001: >60) | 2.3e-1 [56] (0.01: >60, 0.001: >60) | 4.4e-2 [60] (0.01: >60, 0.001: >60) | 2.5, 15.5 / 2.5, 14.9 |
| lih | rdm1 | 6.9e-6 [59] (0.01: 10, 0.001: 12) | 1.4e-5 [36] (0.01: 8, 0.001: 13) | 4.9e-4 [16] (0.01: 10, 0.001: 14) | 4.9e-4 [16] (0.01: 10, 0.001: 14) | 2.28, 2.71 / 2.23, 2.96 |
| n2_r1.098 | rdm1 | 5.1e-4 [4] (0.01: 3, 0.001: >60) | 8.1e-4 [12] (0.01: 3, 0.001: 51) | 1.2e-3 [4] (0.01: 3, 0.001: >60) | 1.2e-3 [4] (0.01: 3, 0.001: >60) | 3.35, 4.46 / 3.35, 4.46 |
| n2_r2.4 | rdm1 | 1.1e-2 [35] (0.01: >60, 0.001: >60) | 6.4e-3 [22] (0.01: >60, 0.001: >60) | 6.8e-2 [5] (0.01: >60, 0.001: >60) | 6.8e-2 [5] (0.01: >60, 0.001: >60) | 1.73, 4.61 / 1.73, 4.61 |

- **Exact floors.** The R1 record converges within 60 steps only on the H4 chains; on h6_r3.0–4.0 it ends
  0.0529 to 0.299 mHa and on n2_r2.0 and n2_r2.4 5.68 and 10.2 mHa above E₀, with 2-RDM floors of 2.6e-2 to
  0.46, and N2 at equilibrium stalls on a plateau (0.182 mHa at k = 60), where the cumulant norm's floor,
  1.1e-3, lies above 1e-3; the exact-line-search record stalls there too (0.1785 mHa at k = 60, records of
  `properties_n2_r1.098.json`), so the plateau is not the unit step's or its growth cap's, and the run does
  not identify its cause. A property error is first order in √q where the energy error is second order (F47):
  h6_r4.0's NONs miss 1e-2 with the energy 0.30 mHa above E₀. The exact-line-search record has the lower floor
  on 49 of the 66 pairs, and R1 needs more iterations on 27 of the 37 pairs where both reach the tighter
  accuracy: unit steps save energies and pay in iterations, which in a measured loop are what costs.
- **Compiled iterates.** Unsliced compiled floors are 6.6–506 times the exact ones at equilibrium on H4, H6
  and LiH, and 1.7–2.5 times on n2_r1.098 [derived from the floors of `properties_<s>.json`]; F25's slices
  (stretched chains only) lower them but leave them 1.4 to 3.9e4 times the exact ones, and route A on the
  compiled iterates is bias-limited on 26 more pairs (listed in U0's Q6 row). F25 bounds a sliced step's state
  error, not the property error, on which the compiled tally rests.
- **F47** holds on every iterate of both records for every target (smallest bound/|error| 1.12 in the q form,
  the first-order cumulant norm of n2_r2.4, 1.38 otherwise, 1.82 in the energy form; all-orders cumulant-norm
  ratios 3.98–302).

**U11, closed-loop deep runs** (selected rows; `summary.md`, "U11: closed-loop deep runs", has every target).
Route best, lagged controller, R1 spin-orbital, 30 iterations, 10 replicates: the minimum error over k (median
[16–84%]) with its median k, the last iterate's error and k_ε (median over the replicates that reach ε, with
their share); the exact R1 floor [its k]; F47's smallest bound/|bias| over every iterate.

| system | target | exact R1 floor | closed loop r = 0.3 | closed loop r = 1 | F47 min ratio |
|---|---|---|---|---|---|
| h4_r1.8 | occupations | 1.7e-6 [28] | 1.9e-5 [1.4e-5, 3.5e-5] k 21; last 1.2e-4; k_eps 0.01: 3 (100%), 0.001: 7 (90%) | 7.9e-5 [3.4e-5, 1.5e-4] k 28; last 2.4e-4; k_eps 0.01: 7 (100%), 0.001: 19 (90%) | 1.12 |
| h4_r1.8 | rdm1 | 1.5e-5 [28] | 1.1e-4 [8.3e-5, 1.5e-4] k 20; last 6.2e-4; k_eps 0.01: 5 (100%), 0.001: 14 (80%) | 4.4e-4 [2.1e-4, 6.8e-4] k 26; last 6.9e-4; k_eps 0.01: 10 (100%), 0.001: 30 (80%) | 1.87 |
| lih | rdm1 | 6.9e-6 [59] | 8.5e-5 [5.1e-5, 1.3e-4] k 27; last 1.9e-4; k_eps 0.01: 11 (100%), 0.001: 20 (100%) | 5.7e-4 [3.6e-4, 9.7e-4] k 28; last 1.1e-3; k_eps 0.01: 20 (100%), 0.001: 29 (40%) | 2.19 |
| lih | dipole | 1.4e-6 [55] | 1.5e-5 [5.4e-6, 4.4e-5] k 23; last 1.7e-4; k_eps 0.004: 12 (100%), 0.001: 20 (100%) | 1.8e-4 [6.2e-5, 4.2e-4] k 26; last 4.8e-4; k_eps 0.004: 22 (100%), 0.001: 28 (80%) | 1.62 |
| n2_r1.098 | rdm1 | 5.1e-4 [4] | 6.2e-4 [4.6e-4, 8.9e-4] k 23; last 1.1e-3; k_eps 0.01: 4 (100%), 0.001: 30 (10%) | 1.1e-3 [6.5e-4, 1.4e-3] k 20; last 1.9e-3; k_eps 0.01: 11 (100%), 0.001: - (0%) | 1.99 |
| n2_r1.098 | rdm2 | 7.7e-3 [28] | 1.0e-2 [9.8e-3, 1.1e-2] k 28; last 1.1e-2; k_eps 0.1: 4 (100%), 0.01: - (0%) | 1.3e-2 [1.2e-2, 1.4e-2] k 28; last 1.5e-2; k_eps 0.1: 11 (100%), 0.01: - (0%) | 3.01 |

At the last iterate the deep runs' medians lie 0.000482 to 0.343 mHa above E₀ at p̃ ≥ 0.9999, with realized r₁
= 0.19–0.67 at nominal r = 0.3 and 0.65–2.22 at nominal 1 and C_loop(30) = 3.23e4 to 1.94e7 C_QPE
(`summary.md`); no replicate converged in 30 iterations.

- **Noise floors and cost.** On h4_r1.8 and LiH the median floors are 7.5–47 times the exact R1 floors at
  r = 0.3 and 30–123 times at r = 1 [derived], and the tighter accuracy is still reached by 80–100% of the
  replicates at r = 0.3; on n2_r1.098 the loop sits on the exact plateau: the NONs reach the tighter accuracy
  in all replicates at r = 0.3 and in 40% at r = 1, the 1-RDM in 10% at r = 0.3, and no other target reaches
  it. The loop's own cost to k_ε at the tighter accuracy (1.55e3 to 3.23e6 C_QPE, the median over the
  replicates that reach ε, with k_ε known only in emulation; `U11_closed_loop` of `summary.json`) exceeds the
  cheapest measured-loop route of U10 on 22 of the 25 cells that some replicate reaches, and on the other 7
  (n2_r1.098) no replicate reaches ε; at the looser accuracy it costs 12.5 to 1.75e5 C_QPE and exceeds U10's
  route on 16 of the 32 cells [derived]. The LiH dipole at 1e-3 takes 2.73e6 (r = 0.3) and 6.99e5 C_QPE
  (r = 1) of loop against 970 for HF-QPE. Three equilibrium systems, 10 replicates each.

**Against the design.** The design expected route A with a free loop and the oracle k to be 3–550× cheaper
than route B wherever its floor reaches the target: confirmed in direction and narrowed in range (4.99–300×).
It expected exact AA on an estimated a to lower that low end: refuted, since exact AA is never the copy of a
best row (the best route-B rows start at p ≥ 0.75, B0 from Ψ₁ at p₁ = 0.977–0.994). It expected route B to win
where route A is bias-limited (stretched H6, N2 at 2.0 Å): confirmed on a larger set (N2 at 2.4 Å and N2's
cumulant norm join; 26 more pairs when compiled); and where the loop must be measured on strongly correlated
N = 6 systems: confirmed and broader at the tighter accuracy, where route B takes every target of every N = 6
system but LiH and n2_r1.098, the equilibrium H6 chains included (at the looser accuracy route A keeps their
cumulant with a measured loop). HF-QPE wins the LiH dipole against AE on filtered copies (16–58×), but a
free-loop route A is cheaper still.

**Against M3.** M3 costed no property; its CQE-only energy cost 2.55e3–7.3e5 times the HF hand-off (M3 T8).
The property counterpart, route A on a measured loop, is the cheapest route on 11 of 66 pairs at r = 1
(229–7.93e4 C_QPE), and its cost rises with k as M3's measured iterations did. M3's replay (M3 T9) left the
product-formula error of the compiled steps outside the δ_step budget; U11 shows that error setting route A's
floor on the compiled iterates.

## 10. Physical units (U12)

The decisive hand-off runs are short and fit on one machine: on n2_r2.4 the residual of the first measured
iteration is 7965 shots (r = 1) or 885 (r = 3) of 12.8 ms each on 5.682e5 physical qubits, the hand-off QPE
one 309.1 s run on 1.14e6 physical qubits at d = 31, and the filter and AA sequence of the best realizable
hand-off, laid out as one run, 143.5 s on 1.118e6 physical qubits, each well inside 7 days. What decides Q4
and Q5 is therefore the Toffoli count against the hand-off saving (§6–§8), not the wall clock. The property
estimates are different: their AE runs are single coherent runs, the longest best row with a free or r = 1
loop (h6_r4.0's cumulant norm at 1e-3, 6.93e12 Toffolis; 7.38e12 with the r = 0.3 loop) has no surface-code
layout, and at the tighter accuracy 39 of the 198 best AE rows have none, while 80 of the 159 with one run
longer than 7 days.

**U12** (physical units of the decisive runs; `results/m4/summary.md`, "U12").

| run | system | Q | Toffolis | budget | physical qubits | d | factory | us/Toffoli | seconds | runs | machines (7 d) | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| residual at k = 1 (r = 1, direct_mg) | n2_r2.4 | 71 | 486 | 1e-5 | 5.682e5 | 21 | AutoCCZ(l1=13, l2=21) | 26.34 | 0.0128 | 7965 | 1 | C_iter(1)/C_QPE = 0.423 |
| residual at k = 1 (r = 3, direct_mg) | n2_r2.4 | 71 | 486 | 1e-5 | 5.682e5 | 21 | AutoCCZ(l1=13, l2=21) | 26.34 | 0.0128 | 885 | 1 | C_iter(1)/C_QPE = 0.047 |
| hand-off QPE run (final, F40 retested) | n2_r2.4 | 185 | 9.158e6 | 0.0005 | 1.14e6 | 31 | AutoCCZ(l1=15, l2=27) | 33.75 | 309.1 | 1 | 1 | C_QPE, n_iter = 11,924 |
| coherent filter + AA run (lt_bbht_L, Psi1) | n2_r2.4 | 218 | 4.593e6 | 0.0005 | 1.118e6 | 27 | AutoCCZ(l1=15, l2=25) | 31.25 | 143.5 | 1 | 1 | 3.77 expected calls of C(Psi) + C_filt as one coherent run; T/C_QPE = 1.502 |
| longest property AE run (cumulant_norm at 0.001, route Bk, loop meas_r1) | h6_r4.0 | 304 | 6.93e12 | 0.0001 | none | - | - | - | - | 1.023 | - | route T/C_QPE = 8.916e+05; runs counted at the longest; no layout: no factory and distance in 7..43 keep 6.93e+12 Toffolis on 304 logical qubits within the failure budget 0.0001 |

- **The residual at k = 1.** Each shot prepares the compiled classical Ψ₁ (486 Toffolis on 71 logical qubits;
  the matchgate readout adds Cliffords only) at p_fail_shot = 1e-5: 12.8 ms on 5.682e5 physical qubits at d =
  21. The planned campaign, 7965.35 shots at r = 1 and 885.04 at r = 3, takes 102 s and 11.3 s on one machine
  [derived], and the shots parallelize (F19); in Toffolis it is the c_iter(1) = 0.423 and 0.047 C_QPE that a
  hand-off saving must exceed.
- **The hand-off QPE run.** One C_QPE of n2_r2.4's SF L = 16 encoding (9.158e6 Toffolis, n_iter = 11,924) on
  185 logical qubits at F40's retested budget 5e-4: 1,140,096 physical qubits, d = 31, 309.1 s. M3 laid out
  n2s's QPE (9.49e6 Toffolis, same 185 logical qubits) at 1e-2 on 873,216 physical qubits at d = 23 in 4.55
  min (M3 T1): the 20 times tighter budget raises d by 8 and the footprint by 31% [derived].
- **The coherent filter and AA run.** U9's s* on n2_r2.4 is lt_bbht_L on Ψ₁ at T = 1.502 C_QPE (§8): 3.77
  expected calls of C(Ψ₁) + C_filt, laid out as one coherent run of 4.593e6 Toffolis on 218 logical qubits
  (1.118e6 physical qubits, d = 27, 143.5 s; F40's convention: exponential search measures between its stages,
  so its actual coherent runs are shorter), followed by the final QPE; with half the QPE's Toffolis it runs at
  a lower distance, so its 33 extra logical qubits need fewer physical qubits.
- **The property runs** (`layout` of the best rows in `properties_<s>.json`; F53, M3's 7-day deadline). No
  factory of OpenFermion's model fails below 3.6e-16 per Toffoli, so no run above about 2.8e11 Toffolis meets
  p_fail_ae = 1e-4: the longest best run of the free and r = 1 loops (U12: 6.93e12 Toffolis on 304 logical
  qubits, 1.023 runs of that length, 8.916e5 C_QPE for the row) has no layout at any distance up to 43. At the
  tighter accuracy 39 of the 198 best AE rows have none (37 Bk, 2 B0; 2.99e11 to 7.38e12 Toffolis), and of the
  159 with a layout 80 run longer than 7 days, which no number of machines shortens: 75 of the 98 laid-out Bk
  rows (median 2.74e6 s), both laid-out B0 rows and 3 of the 57 route-A rows (route A: 1.85e3 to 8.09e5 s,
  median 2.27e4 s). At the looser accuracy 2 of the 198 have no layout and 46 of the 196 laid out run longer
  than 7 days, no route-A row among them. The LiH dipole by free-loop route A is one AE run of 5.444e8
  Toffolis (4e-3; 21,095 s on 1,333,248 physical qubits) or 3.483e9 (1e-3; 161,965 s), by HF-QPE energies of
  7.853e8 and 3.826e9 Toffolis (30,431 s and 177,887 s), each energy a separate coherent run.

**Conditions.** OpenFermion's surface-code model with M3's defaults (p = 1e-3, 4 factories, 1 µs cycles; F18)
and M3's 7-day deadline (F19); shots at p_fail_shot = 1e-5, the hand-off QPE and the filter and AA run at
F40's retested budget 5e-4, property AE runs and HF-QPE energies at p_fail_ae = 1e-4. The residual's shot
count is the plan at the exact Ψ₁; the coherent AA run counts the expected calls. The no-layout verdict is a
fact of the factory model (it would change with a better factory or with AE runs split at a loss of Heisenberg
scaling, neither costed here); F53 compares Toffolis and does not depend on it.

## 11. Validation, regressions and runtime

All 78 integration checks of the summary pass. The scaling benchmark reproduces 70 of the design's 72
reference numbers and the loop benchmark all 20; the two misses are design values of the cheapest iteration
computed with HF-state variances, whose counterparts with M3's iterate-1 variances, the protocol's at N ≤ 6,
are reproduced. The design's hand-off references reproduce to their printed digits except the costs of the
filtered and two-stage schemes, which move by −7.66% to +5.57% through the implemented filter and coarse-QPE
costing (§13). M3's outputs are unchanged, all 1324 tests pass (400 of them M4's), and a second full run gives
the same per-system outputs (the summary's own outputs were not compared; see Reproducibility). The five
benchmarks take about 3.0 h on one thread, with a peak resident memory of 1470 MB.

**Integration checks** (`results/m4/summary.md`, "Integration checks"; `summary.json`: `checks_failed` = []).
Each compares a number the summary recomputes with the one the producing benchmark wrote, and the script exits
with status 0. They catch wiring and bookkeeping errors between the benchmarks, not model errors, which the
tests and the reference numbers address.

| check | checks | detail |
|---|---|---|
| inputs complete | 1 | every required input present |
| QPE side of a hand-off system (model spectrum of H′, E1lb, CISD(H′), rule B's form and n_v, filter and coarse-QPE costs) equals the hand-off benchmark's | 21 | agree |
| U8's rule-B cost of HF equals U9's rep (F40's verification runs) | 21 | 3 rows each |
| U9 part 1, re-costed with the summary's QPE side (flags included), equals the hand-off benchmark's | 21 | 413–616 costs per system, 10,751 in all [derived] |
| U4 repetition brackets recomputed from the cached records equal `scaling_fits.json` | 1 | 24 brackets |
| rule-E costs and paired verdicts recomputed from the replicates equal the loop benchmark's summary | 10 | 8–76 verdicts per system, 292 in all [derived]; worst relative cost difference 0.0e+00 |
| deep-run property errors of HF equal the properties benchmark's | 3 | 5 (h4_r1.8), 6 (lih) and 5 (n2_r1.098) targets |

**Reference numbers.** The scaling benchmark's (`scaling_summary.md`, "Reference numbers") reproduce 70 of 72
to the printed digits: p_HF of the 25 scanned systems, Δ₁ of the 16 chains and the five N2 CAS(6,6)
geometries, C_QPE of eight systems, the cheapest iteration of h6_r3.0, h8_r3.0 and h10_r3.0 (sf; 1.80662,
27.0289, 209.111) and of N2 CAS(6,6) at 2.0–2.8 Å (so), the n2_r2.4 QPE (SF, λ = 7.59075 Ha, n_iter = 11924,
C_QPE = 9157632) and E − E₀ of its four reference states, and the successive HF rates at R = 3.0 bohr
(0.100726, 0.10489). The design also quotes α^(10) = 0.1079 there, which the table does not check; the run
gives 0.1078 (U2). The two misses:

| quantity | design | reproduced | reading |
|---|---|---|---|
| n2_r2.4 T_res(1)/C_QPE, so, r = 1 | 0.43 | 0.422725 | the design's value used HF-state variances at N = 6; with M3's iterate-1 variances, the protocol's at N ≤ 6 and those of the design's own closed-loop check, its 0.423 is reproduced |
| n2_r2.8 T_res(1)/C_QPE, so, r = 1 | 0.41 | 0.398468 | the same; 0.398 is reproduced |

The loop benchmark's (`loop_summary.md`, "Reference numbers") all reproduce: k* of four spin-orbital R1
records and 16 checks of the closed-loop first measured iteration against the design's independent
implementation (Gaussian direct-route noise at the oracle budget, unit steps, no growth cap, 100 replicates).
The run draws other random numbers, so the comparison is statistical: every design mean lies inside the run's
95% bootstrap interval and every design median inside the run's 16–84% band.

| system | oracle r | mean T_rep(2): design | run [95% CI] | median p₂: design | run [16–84%] |
|---|---|---|---|---|---|
| n2_r2.0 | 1 | 2.638 ± 0.01 | 2.64 [2.621, 2.658] | 0.471 | 0.472 [0.448, 0.491] |
| n2_r2.0 | 3 | 2.259 ± 0.018 | 2.257 [2.221, 2.291] | 0.453 | 0.459 [0.413, 0.491] |
| n2_r2.4 | 1 | 3.958 ± 0.03 | 3.955 [3.894, 4.018] | 0.284 | 0.288 [0.264, 0.311] |
| n2_r2.4 | 3 | 3.643 ± 0.047 | 3.603 [3.496, 3.714] | 0.279 | 0.290 [0.246, 0.328] |
| n2_r2.8 | 1 | 4.691 ± 0.046 | 4.716 [4.629, 4.804] | 0.235 | 0.234 [0.211, 0.256] |
| n2_r2.8 | 3 | 4.327 ± 0.068 | 4.303 [4.168, 4.443] | 0.235 | 0.243 [0.205, 0.279] |

The design's acceptance intervals hold too (e.g. n2_r2.4, oracle r = 1 and 3: 3.955 in [3.85, 4.06] and 3.603
in [3.5, 3.78]; its deciding cell 3.601 in [3.45, 3.8] with P(T(2) < T(1)) = 0.96, design ≥ 0.9), and the
first iteration costs 0.5146 / 0.05718, 0.4227 / 0.04697 and 0.3985 / 0.04427 C_QPE on n2_r2.0, 2.4 and 2.8 at
oracle r = 1 / 3, against the design's 0.515 / 0.057, 0.423 / 0.047 and 0.398 / 0.044. The hand-off and
property benchmarks print no reference table; their design values against the outputs (`handoff_summary.md`,
`handoff_<s>.json`, `ft.acceptance`, `properties_summary.md`), also pinned by the tests:

| quantity | design | run |
|---|---|---|
| E₁′ − E₀′ (mHa), n2_r2.0 / 2.4 / 2.8 | 191.907 / 230.863 / 244.441 | 191.907 / 230.863 / 244.441 |
| p on ψ₀′ of HF / Ψ₁ / exact-LS Ψ₂ / unit Ψ₂, n2_r2.0 | 0.25390 / 0.39761 / 0.54684 / 0.47300 | 0.2539 / 0.3976 / 0.5468 / 0.4730 |
| ε_id of HF, n2_r2.4: B_U, sine at 2n_iter / Kaiser(3) at 4n_iter | 2.38e-4 / 3.76e-8 | 2.38e-4 / 3.76e-8 |
| ε_id of HF, n2_r2.8: B_L, one / two runs; literal θ | 1.6e-4 / 1.0e-8; 0.322 | 1.62e-4 / 1.04e-8; 0.322 |
| τ_W(5); n_v thresholds t₁ / t₂ / t₃ (grid units) | 1.84e-4; ≈ 26 / 3.0 / 1.3 | 1.835456e-4; 26.036 / 3.032 / 1.295 |
| M3's T6 under rule B: h6s measured, HF / n2s spin-free classical first step, HF / n2s spin-orbital classical first step, Ψ₁ | 2.744 / 3.956 / 2.525 | 2.744 / 3.956 / 2.525 |
| schemes on HF and Ψ₁ of N2 at 2.0–2.8 Å: rep (M3's rule E and F40), aa, bbht | e.g. n2_r2.4 HF: 8.928 / 9.928 / 5.113 / 6.301 | 8.928 / 9.928 / 5.113 / 6.301; all 24 cells within 0.03% [derived] |
| the same, lt and lt_bbht (2.0, 2.4 Å) | e.g. n2_r2.4 HF: 11.17 / 14.10 | 11.77 / 14.89; +2.05% to +5.57% over the 8 cells [derived] |
| the same, lt_bbht_L | e.g. n2_r2.4 HF: 1.665 | 1.705; +1.69% to +2.92% over the 6 cells [derived] |
| the same, 2s and 2s_L | e.g. n2_r2.4 HF: 42.74 / 3.119 | 39.47 / 2.953; −7.66% to −2.85% over the 10 cells [derived] |

The overlaps on H′ agree with the design's to its five decimals, CISD(H′) − E₀′ of N2 at 2.0–2.8 Å and the LiH
dipole (−1.817716 a.u. at ψ₀, −1.911089 at HF) to its printed digits, and s* is lt_bbht_L on all six N2 states
of the design's scheme table, as in the design. The tests pin the remaining reference numbers within their
stated tolerances (among them CISD(H) − E₀ of N2, ε₂/ε* = 1.598 of h8_r4.0 sf, C_QPE(h8_r3.0) = 18,661,951,
the n = 10 holdout within 3%, the contaminant term q_c/p₀ of the compiled n2s Ψ̃₁, 1.2e-4 ± 20%, and the
HF-QPE optimum of the LiH dipole without model bias, T = 91 ± 5 C_QPE). In the outputs the F47 bias bound
holds in all 264 checks (§9) and the windowed identification errors of the exact states stay below ε_id = 1e-3
in every form of record (§7).

**Regressions against M3.**

- **Bit identity.** `benchmarks/run_ft_pipeline.py --systems h4,n2s --threads 1`, run with the M4 package,
  reproduces M3's `results/ft_h4.json` and `results/ft_n2s.json` with 0 differences in every key except
  `runtime_s`, `timings` and `validation.global.seconds` (49.1 s, peak 516 MB; its Appendix A table 44 of 44)
  [build check]; the 17 tests of `tests/test_ft_pipeline.py`, the golden h4 test among them, pass (31 s)
  [build check]. M4 changes M3's modules only additively and edits no M3 test file.
- **M3's records inside M4.** The hand-off benchmark recomputes M3's exact records of h4, h6a, h6s, h8, lih,
  n2 and n2s from M3's settings with differences of 0.0 in energies, overlaps and preparation Toffolis, and
  the oracle rule-E column of rule B equals M3's T6 rows (`m3_t6` of `handoff_<s>.json`).
- **Loop costs and encodings.** U5's line-search rows at r = 0.3 reproduce M3 T4's classical-first-step totals
  (`tests/test_ft_steprules.py` pins the h4 total, 2416.996 C_QPE, to 1e-12 relative), and the aliases build
  M3's systems bit for bit and select M3 T1's encodings of h4, h6s, n2 and n2s (§1).

**Tests.** `pytest --collect-only` finds 1324 tests, 1277 fast and 47 slow, and all pass. M4 adds 14 files
with 400 tests (34 slow): `tests/test_ft_symmetry_names.py` (F20), `test_steprules.py` and
`test_ft_steprules.py` (F26–F31, F33), `test_ft_closedloop.py` (F32–F38), `test_ft_scaling.py` (F21–F25, F45),
`test_ft_acceptance.py`, `test_ft_certify.py`, `test_ft_amplify.py` and `test_ft_handoff_rule_b.py` (F39–F44),
`test_ft_properties.py` (F46–F53), and 46 tests of the benchmark scripts and the summary. The 366 fast M4
tests take 182 s on one thread on an otherwise idle machine [build check].

**Reproducibility.** A second, independent full run [build check] with the same numerical code (an earlier
summary script, and without the layout fields of the property rows and the deciding flag of the control cells)
gives identical JSON apart from runtimes and those fields: `scaling_<s>.json` and `scaling_fits.json` 26 of
26, `loop_<s>.json` 17 of 17, `closedloop_<s>.json` 10 of 10, `handoff_<s>.json` 21 of 21,
`properties_<s>.json` 13 of 13. The closed-loop statistics are deterministic given the design's seeds
(`SeedSequence(20261108, ...)` per replicate, bootstrap seed 12345), so their intervals are the sampling
uncertainty of the replicate means, not run-to-run noise.

**Runtime** (one thread; other processes ran alongside, load about 2).

| benchmark | wall time | design estimate | longest calculation | peak RSS |
|---|---|---|---|---|
| `run_m4_scaling.py`, one process per system and stage (25 systems, 146 stages) | 73 min: h4 12–29 s, h6 30–37 s, n2 28–43 s, h8 308–457 s, n2v 255–329 s, h10 252–487 s per system (the stages' own timings); assembly from the cache 10.3 s (`scaling_summary.md`) | ≈ 70 min idle, ≈ 95 min at load 3–4 | 363 s (the QPE scan of an H10) | 254 MB (assembly) |
| `run_m4_loop.py` | U5 73.0 min (17 systems; h8_r1.8 and h8_r3.0 49.2 min [derived]); closed loop 194 s (10 systems, 73 configurations) | ≈ 40 min, of which ≈ 18 min for the closed-loop blocks | 233.7 s (U5 h8_r1.8, spin-orbital R3 at r = 0.3) | 194 MB (the resumed process) |
| `run_m4_handoff.py` | 1497 s (21 systems; the six N = 8 systems 1196.9 s, 80.0% [derived]) | ≈ 10 min | 314.0 s (h8, the whole system) | 1470 MB |
| `run_m4_properties.py` | 285 s (13 systems) | ≈ 10 min | 40.1 s (h6_r1.8, the whole system) | 456 MB |
| `run_m4_summary.py` | 371.5 s: QPE sides 211.1, U4 120.5, U9 part 2 13.8, deep runs 24.5 | ≈ 10 min (deep runs ≈ 3 min) | – | 1227 MB |

The five benchmarks take about 3.0 h, the upper end of the design's 2.5–3 h, and no single calculation exceeds
the design's limit of about 10 minutes (the longest: 363 s). The loop benchmark takes about twice its estimate
(U5 on the two H8 chains alone 49.2 min) and the hand-off benchmark 2.5 times (80.0% on the six N = 8
systems); the others stay within theirs. The run was made in pieces: the stage cache keys carry the SHA-256
digest of the package source, and `--resume` reuses a `loop_<s>.json` or `closedloop_<s>.json` only when its
package and script digests equal the running ones (all 27 carry the same two; 11 of the 17 U5 systems were
loaded this way, their recorded seconds marked "(loaded)" in `loop_summary.md`). M3's benchmark took 280–348 s
with a peak of 551 MB (M3 §9); M4's benchmarks take roughly 30–40 times as long, mostly on the N = 8 systems
and the N = 10 QPE scans.

## 12. Caveats

The open questions of [m4-cost-model.md](m4-cost-model.md) ("Scope and open questions"), numbered as there,
with what this run says about them: items 1, 2, 4 and 22 qualify the statements beyond n = 10, 5 and 6 the
inputs and compiled states, 7–12 the step rules, 13–17 the closed loop, 3 and 18–21 the hand-off, 23–27 the
properties.

1. **The overlap model (F22)** is validated over Δn = 2 only (no n = 12 holdout, §13), while p_HF reaches 1e-3
   at n = 28.6–127.4; the n = 10 holdout errs by +0.28% to +2.77% (last rate) and −2.36% to +0.45% (Aitken),
   and for the R1 Ψ₂ series at 2.5 bohr the Aitken limit (0.0807) is four times the last rate (0.0202), so its
   n(p = 1e-3) spans 94.4–347.1 sites. Every n > 10 statement (U2, U4, U0's Q1 and Q2) is a model bracket with
   an unquantified model risk.
2. **The gap model (F21).** n·Δ₁ still rises at n = 10 and γ rests on n = 8 and 10; the A′/n end is the
   stretched chain's Heisenberg limit (gap ∝ 1/n, F21 [cited]), but that R = 2.5–4.0 bohr is already in that
   regime is an [assumption], not checked, and F45's lengthened hand-off QPE depends on which end holds.
3. **E1lb is assumed (F41)**, E₁(H′) − 0.1 mHa at N ≤ 8 and E₁(H) − 1 mHa at N = 10, and no lower bound on E₁
   is computed: every rule-B certificate, filter stop edge, L-form termination, route-B pass edge and HF-QPE
   acceptance, hence U8, U9, U4's scheme columns and U0's Q5, is conditional on it, as M3's certified rules
   were on E₁.
4. **The cost exponents (F23)** come from two intervals and drift (U3), on THC fits that are non-convex (the
   selected fits of eight of the twelve scanned THC chains, h6_r2.5–3.0, h8_r2.5–3.5 and h10_r2.5–3.5, and of
   M3's h8 report refine_success false); whether β falls toward the locality bracket beyond n = 20 is
   untested.
5. **Inputs at one state (F24).** Variances at iterate 1 (N ≤ 6), at HF (N = 8; within 1.4% in V_m at N = 4–6,
   up to 22% per element) and from a 100-element subset (N = 10, sampling error unrecorded, direct route
   only); property estimator inputs at ψ₀ (§3, §9).
6. **Slicing (F25)** bounds one step's state error, not the overlap, energy or property error: on h6_r4.0 (so)
   the error accumulates to 0.0507, 0.0778 and 0.0810 at k = 1–3, and its classical step misses η_s even at
   s = 16. The rule compares with the exact step and is not realizable (F32); every compiled cell of U7 (hence
   of U9 part 2) and the sliced columns of U10 and U11 rest on it, while the compiled rows of U8 and U9 part 1
   use one fixed slice; fixed slices are not calibrated beyond N = 8.
7. **Energies charged at V_E(Ψ_{k+1}) (F28).** On small gaps the stencil's end points have larger variances
   (2.09, 1.45 and 1.05 V_E(Ψ_{k+1}) at k = 1–3 on h4_r4.0, so [build check]), so ρ_E(parab3) and R3's energy
   costs are lower estimates there.
8. **R3's plans (F31).** U5's planned T_en lies within 0.962–1.14 of the charge except on stretched N2
   (0.79–4.49), where the plan foresees neither the caps nor where the fit lands [derived from the gate logs];
   the closed loop's gate, at V_E(HF), over-plans by a median 1.91–2.84 [derived].
9. **σ_P3 = 0.47|ΔE_m| (F27)** is calibrated on exact parabolas; on stretched N2 ω_k = |ΔE_m,k|/(E_k − E₀)
   spans 4.34e-4 to 10.9 over U5's gate decisions, against 0.00266–1.32 elsewhere [derived from the gate logs
   of the R3 rows of `loop_<s>.json` within each row's costed horizon, |ΔE_m,k| = σ/0.47 and E_k − E₀ the
   record's e_err], far outside the calibrated regime, and the residual bias of the debiased slope
   (slope_debias_approx) is not quantified.
10. **Stretched N2 within 40 steps.** No rule reaches 1.6 mHa at 2.0–2.8 Å, and 7 of the 12 rows of n2_r2.8 do
    not reach p = 0.9; U5 ranks there at k_p90 (§5), and the hand-off verdicts come from U6, U8 and U9.
11. **The growth cap (F29)** binds on 292 of the 1161 R1 steps of U5 (69 of 352 before k_ref), and U5 has no
    uncapped R1 row. The angle cap binds only in non-deciding cells (h4_r4.0 at oracle r = 2 and 3, n2_r2.4's
    lagged r = 2 and two compiled oracle r = 3 cells).
12. **The energy-precision wall (F28)** is lower against a tight residual: at k = 1, ρ_E(parab3) < 1 on 24 of
    the 34 R1 rows at r = 0.3, 7 at r = 1 and 2 at r = 3 [derived]; at which r a hybrid loop is cheapest the
    run does not resolve.
13. **The lagged plan (F35)** realizes r₁ / r₂ = 3.00 / 0.82, 2.71 / 1.11, 2.54 / 1.38 (stretched N2) and 0.46
    / 2.29 (h4_r4.0), near the oracle optimum on every headline system, but the verdict of record is that of
    this one pre-declared setting (§6).
14. **The second budget** follows a noise-dominated estimate of ‖A₁‖\_m (the floor binds at k = 1 in about
    half the replicates on n2_r2.0 and n2_r2.4, F35 [build check]); its share in T(3) is not separated.
15. **Heavy tails.** h4_r4.0 at oracle r = 2 and 3 (catastrophic shares 0.18 and 0.35) and at lagged r = 2
    (0.34) is not resolved by 100 replicates; the deciding cells are (shares 0–0.08, stretched N2 ≤ 0.10).
16. **Correlated matchgate noise** is not modelled (up to 1.32× in M2's noise floors), and the design's
    statement that it leaves T_rep(2) unchanged within the intervals is not re-checked.
17. **Overlaps on H against H′.** The closed loop's p̃_k and route B's starts are weights on ψ₀ of H, the QPE
    acts on H′: they differ by up to 0.48% on spin-orbital R1 states (within the design's 0.5%) and 0.68% on
    spin-free ones [derived], with a persistent sign, so the difference largely cancels in U9 part 2's
    savings.
18. **(A3) and K₀ = 1 (F39, F40).** Under the other reading of (A3) every hand-off QPE doubles. K₀ = 1
    understates QPE terms accepted within about 1.5 grid units of E₀′: LiH's B_U lies 0.53g above E₀′ (K₀ =
    0.529 [build check]), so LiH's rule-B, U9 and HF-QPE costs understate their accepted terms by up to 1.9×;
    h4 lies at 1.07g (K₀ = 0.996). The windowed ε_id are phase-offset lotteries; the bounds take the supremum.
19. **Thin certificate margins.** CISD(H′) leaves 3.077 mHa on n2_r2.4 and certifies nothing on eight
    stretched systems. CISDT(H′), a sensitivity, would lower the HF hand-off at s* on stretched N2 by
    7.0–27.5% [derived] and turn the two exact-trajectory first iterations that pay at r = 3 into losses
    (n2_r2.4 0.0997% → −2.06%, n2_r2.8 2.36% → −0.883%; §8). Whether a truncated-CI minimum is an acceptable
    certificate, and how the margins scale with size, is open; the verdicts of record use CISD(H′).
20. **Spin contamination of compiled states.** On stretched N2 a quintet lies 21.395, 4.346 and 1.210 mHa
    above E₀′, so compiled states have no E1lb of any spin with a margin, and rule B stays sound on them
    through the contaminant budget alone (§7); the headline cells propagate exactly.
21. **Exponential search overshoots 1/p above p = 0.299 (F43)**: with lt_rep and lt_rep_L the optimum differs
    from s* on 304 of the 924 rows of U9 part 1 and flips 2 of its 462 verdicts (h4_r3.5 at r = 1) [derived];
    whether a choice between repetition and exponential search belongs in the scheme set of record is open.
    The filter degrees come from a smoothed step with a 1% margin, not a minimax polynomial.
22. **Beyond the computed sizes** only rep and bbht with B_L remain (no certificate assumed, L forms while Δ₁
    ≥ w_L + 1 mHa), so U4's "best" n* rests on that restriction (§4).
23. **The exact model bias.** Route B and HF-QPE charge b_fact, an emulation fact like route A's usability at
    a pre-declared k; F47's bound instead changes 11 tighter cells and removes 12 looser AE cells of n2_r2.4
    (§9). No certifiable bound as tight as b_fact exists.
24. **F52's rule** has no infidelity condition: LiH's NONs at 1e-2 and cumulant at 0.1 run on SF L = 15
    (infidelity 1.07e-4 against F9's limit of 1e-4) with b_fact 1.27e-3 and 1.22e-2 against 1.22e-4 and
    4.15e-3 on F9's L = 16. The exact-AA leakage allocation is a model choice (F50).
25. **HF-QPE (F51)** takes the derivatives and stencil biases of H for those of H′, ignores the growth of λ by
    |h|α_O and costs the filters and coarse QPE on H′; LiH's refined steps carry 0.9999b and 0.9996b of
    stencil bias.
26. **NONs and the cumulant bound (F46, F47).** The linearized NON operators are first-order exact only
    without coupling of degenerate natural orbitals (LiH's π pair is degenerate), and the all-orders
    cumulant-norm bound is proven for real states only; on the records F47 holds in all 264 checks.
27. **Layouts (F53).** 39 of the 198 best tighter AE rows (2 of 198 looser) have no layout and stay selectable
    (§10); which failure budget an HF-QPE energy should carry is open.

Beyond those, the conditions of the whole study:

- **Exact and realizable.** U1–U4, U8, U9 part 1, U10 and U11 (exact and compiled columns) are noise-free
  emulations where r enters only the costs; only the closed loop (U6, U7, U9 part 2, U11's deep runs) lets
  finite shots move the state. The oracle controller, the line-search rule, F25's slice rule, the angle cap,
  the oracle schemes and the emulation's exact quantities (route A's usability at k, b_fact, fp's inputs) are
  not realizable; none decides a verdict of U0 except route A's usability and b_fact in Q6, as U0 states.
- **The noise model.** Independent Gaussian residual noise at variances taken at one state, F34's sine-window
  law for the AE route, no emulated logical failures, 100 / 20 / 10 replicates: the intervals are the sampling
  uncertainty of this model, not of a hardware run.
- **The verdicts near zero.** The s* savings of the deciding cells on stretched N2 (−0.054, −0.0111, +0.0212
  C_QPE) are resolved by the replicates but not by the model: the certificate (item 19) and the filter costing
  (§13, F43) move T_s* itself by more, and lt_bbht_L's termination is certified by no U. Both modelling
  choices favour the one N2 pay: a filter about 25% cheaper, within F43's factor of up to 2 in the walk part,
  would turn n2_r2.8's +0.0212 into a loss (§6), and CISDT(H′) turns the comparable exact-trajectory pays at
  r = 3 into losses (item 19). h4_r4.0's paying verdict rests on repetition with B_L (no certificate,
  termination not certified) and on an iteration that mostly repairs the overshoot of the free classical step,
  against which no damped free step was run (§4), and is reversed by the slices of its compiled step (§6).
- **Systems and cost models.** STO-3G throughout; every statement beyond the emulated sizes is a model
  statement of F21–F23 and F45 with its bracket; OpenFermion's QPE and surface-code models, M3's window model
  (K₀ = 1), (A3), and filter, coarse-QPE and AA-reflection costs by formula, without circuit-level checks.

## 13. Changes from the design specification

The cost model in [m4-cost-model.md](m4-cost-model.md) is the implemented one; this table records, for every
change from the design specification, what the design said, what is implemented, why, and which numbers of
this run it moves. Each item is marked in m4-cost-model.md by "(changed from the design specification;
m4-results.md §13)"; the last row is a change of the tables. The evidence is the output that holds the number,
the test or docstring that pins the rule, or a separate check [build check]; a counterfactual number is
derived from the outputs [derived] or such a check.

| formula | design specification | implemented | why | effect on numbers |
|---|---|---|---|---|
| F20: names | parametric names; R parsed with float(); ValueError for any other name | R, a, b plain positive decimals, n even ≥ 2 without a leading zero; other spellings raise ValueError, M3's names resolve first | closed-shell RHF singlets; patterns disjoint from M3's names | none |
| F20: stage cache | digest of the build spec, module sources and stage key | also the numpy, scipy, pyscf and openfermion versions, system, stage and a format number; `--no-cache`; atomic writes | no reuse across library versions or stages | none (U1–U4 assembled from 146 cached stages) |
| F21: margins | m_BL ≈ 11.4 mHa (sin θ₀ ≈ 0.94) | sin θ₀ = 1, E₁ − E1lb = 1 mHa at every size: m_E = m_B = 3, m_BL = 12 mHa | no extrapolated size has a walk; 1 is the upper bound, so m_BL is not underestimated | n_cert(m_BL) 5–6% shorter [derived] (R = 3.0 bohr: 58–80 against 61–86) |
| F21: U1 | U1 carries the lowest same-irrep level of any spin | in the U1 supplement of the summary (`ft.certify.sector_levels`) | it needs the model spectrum of H′ | none (§1) |
| F22: U2 series | fits for HF, classical Ψ₁ (sf, so), R1 Ψ₂ (so) | adds the R1 Ψ₂ of r1_sf (n ≤ 8) | F45's spin-free n* needs α₂ | one more series per R |
| F22: holdout | optional n = 12 holdout (`--holdout-h12`) | not run (`ft.scaling.HOLDOUT` defined, no flag) | optional | overlap model validated over Δn = 2 only (§12, item 1) |
| F24: cheapest iteration | n2_r2.4 so 0.43, n2_r2.8 so 0.41 | 0.422725, 0.398468 (0.514627 on n2_r2.0) with iterate-1 variances at N ≤ 6 | the design used HF-state variances | the two misses of the reference table (§11) |
| F26, F28, F31, F32: interfaces | the listed result fields, keywords and hook arguments; the classical first step only in the closed loop | added `de_model` (ΔE_m), `step_qubits` and `norb`, the direction d_k in the hook history, `first_step="exact"` and `replicate(metric=...)` | U5's precisions need ΔE_m, F28's qubit counts N and the step qubits, the plan d_k; the noise-free limit reproduces both M3 records | none (additive) |
| F27: 2-point fit | σ_P2 ≈ 0.16\|ΔE_m\|; σ_A, σ_P2 in U5 | σ_P2 = 0.167\|ΔE_m\| (Monte-Carlo, 4e5 draws), not run; U5 reports σ_A and σ_P3 | 0.16 loses 0.088; the design's columns carry σ_P3 | none |
| F29: scope of the caps | caps on every rule | model rules at k ≥ 1 only; ε₂ and the references uncapped | the references reproduce M3's trajectories (to 1e-10 Ha) | U5's line-search rows are M3's (§11); R1, R3 unaffected |
| F29: growth reference | the previous accepted step | the previous nonzero step; none before the first | a zero step would cap every later step at 0 | none |
| F30: U5 reference iterate | C_loop(k*), precisions at k* − 1 | k_ref = k* or k_p90, marked; precisions at k_ref − 1 | k* is undefined on stretched N2 | N2 rows at 2.0–2.8 Å and h8_r3.0's sf R1/R3 rows at k_p90 (e.g. n2_r2.0 so, R1 at r = 0.3 / 1 / 3: 307 / 53.4 / 5.93) |
| F30: record conversion | 1e-9 relative | 1e-9 relative except E_k − E₀ (1e-10 Ha) and saving_vs_hf (absolute) | line searches differ by ≤ 1.0e-11 Ha | none (a test tolerance) |
| F31: planned V_E | one constant V_E | V_E at the planned step in U5; V_E(HF) in the closed loop's gate | F30 charges V_E(Ψ_{k+1}); V_E(HF) over-plans by 1.91–2.84 [derived] | with V_E(HF): h4_r4.0 sf r = 0.3 k* = 7 at 1434 (U5: 6, 451); n2_r2.4 so r = 0.3 k_p90 = 24 (U5: 17) [build check]; R1 unchanged |
| F32: non-realizable parts | oracle controller, line search, angle cap, oracle columns | also "slice_rule" (F25's rule); fixed slices realizable | F25 compares with the exact step in CI space | all 12 compiled U7 cells non-realizable; no deciding cell has a non-realizable part |
| F34: returned variances | variances from the allocation | plug-in V̂_e = (ε_e/κ)²(1 − (Ô_e/α_e)²) | Var(Ô_e) depends on the unknown θ_e; bias 7e-5–3e-4 of ‖A₁‖²_m [build check] | only where ae_x is measured: no U6 cell |
| F34: tails | P(\|Δ\| > 5) = 0.4–2.5e-4 | supremum 2.84e-4 (≤ 2.9e-4) | the design sampled five offsets | none |
| F38: cost gate | planned cost | the realized C_iter(k) | equal for R1 | none for R1 cells |
| F40: tail | τ_W(D) on the M = 4096 register at 64 offsets | also the jump point and refined offsets, and for the sine window the continuum lattice law | the offsets miss the jump point; M = 4096 underestimates longer registers | τ_W(5) = 1.834077e-4 → 1.835456e-4 [build check]; B_L bound 3.37e-5 (not 3.36e-5); no n_v changes |
| F40: choice of form | the smaller n_v, B_U on ties | the cheapest terminating option (B_L, each certificate's bracket, B_U at U + ΔE); non-terminating rows cost ∞ | a costly certificate must be weighed against a verification run | B_L terminates on all 21 systems; n2_r2.8 exact_so Ψ₃ takes B_L (T_B = 1141) |
| F40, F42: dedicated certificates | planned U, deterministic n_v | expected n_v over Û ~ N(U, σ²), σ = m_j/(2z); U9's schemes take the same runs | the realized bound sets D | n2_r2.4 exact_so Ψ₃: E[n_v] = 1.2506, T_B = 1021.11 (1020.86 at the planned n_v [derived]) |
| F41: point group | SF and DF conserve the point group | min(e1_singlet, e1_any_irrep) where eigenvectors leave their irrep by > 1e-6 | n2_r2.4's SF L = 16 breaks D2h (impurity 5.5e-3) | n2_r2.4, n2v_r2.4 flagged; singlets coincide, no number moves |
| F41: contaminant term | q_c/(p₀K₀) | q_c/(p₀K₀ⁿ) for n runs | n runs pass ψ₀′ with K₀ⁿ | ≤ 1.1e-8 relative [derived] |
| F41: U8's bounds | design-floor bound; + q_c/(p₀K₀) when compiled | the design-floor bound only with ≤ 1e-12 weight below E1lb ("n/a (q_c)", "n/a (below E1lb)") | its derivation needs every populated level at or above E1lb | 52 of 63 compiled rows lose it (24 n/a (q_c): 21 on stretched N2 and 3 on h8_r3.0 with q_c ≤ 4.2e-11; 28 n/a (below E1lb)) |
| F43: filter cost | scheme table without GQSP rotations | `filter_cost` includes 3(2d + 1)(b_rot − 2) | F8's GQSP count | lt +2.05% to +5.39%, lt_bbht +2.40% to +5.57%, lt_bbht_L +1.69% to +2.92% on the design's rows [derived]; s* unchanged |
| F43: filter construction | dense-grid check, the design's widths | also the band edges; six narrower widths; Nyquist cap | the grid check misses the pass-edge error by up to 11% at ε_f = 1e-3 | degrees +2–3% at 1e-3, none at 1e-2: the hand-off schemes do not move |
| F43: two-stage QPE | the smallest M_c passing both edges | every level beyond each edge at the best cut; register law to M = 4096 | the two-edge condition is a lattice lottery | M_c ≈ 0.92 × the closed form; 2s, 2s_L 2.85–7.66% below the design's rows [derived] |
| F43: flags | per scheme row | per scheme; a U9 row carries those of s*; compiled rows add two flags | the verdict carries its own scheme's status | no cost changes |
| F43: filter then repeat | none | sensitivity keys lt_rep, lt_rep_L; never s* | E_BBHT(p) > 1/p above p = 0.299 | flips 2 of 462 U9 verdicts (§12, item 21) |
| F43: U_k | min(U_classical, certified energy of Ψ_k) | the cheapest certificate per scheme, each with its own filter and cost | the lowest U is not always the cheapest hand-off | dedicated U_k in 277 cells; s* uses none [derived] |
| F43: fp on compiled states | Eckart–Temple with E₁ | the lowest populated level of H′ less 0.1 mHa replaces E₁ | compiled states populate levels below E₁ | fp uncosted on 150 of the 252 compiled cells of the 21 labels (on 48 of them the exact state's fp is costed); never s* |
| F44: break-even overlap | the root of H_s(p₀, 0) − H_s(p₁, c₁) = C_it | the largest p₀ with ≥ C_it (downward scan, bisection) | the root is not unique | none for monotone H_s |
| F45: preparation terms | c_k ≪ 1, dropped | c_k(n) as power laws on N ≥ 6; c_k = 0 reported | C(Ψ_j) outgrows C_QPE | e.g. 340–1126 against 342–1168 (R = 3.0, sf) |
| F45: lengthening | all of C_QPE(n) rescaled beyond the window | only the hand-off QPE of QPE-scaled schemes | preparation and filters do not depend on ΔE | windowed n* moves; no filtered hook is used |
| F45: scan and iterations | n* ≤ 3000 for the measured iterations | even n, stopping below p = 1e-300; k = 1 only | closed shells, underflow; k = 2 needs fits of Ψ₃, which the protocol does not record | none on the reported n*; no k = 2 brackets |
| F45: rate corners | n* over each fit's rate bracket | matched corners; crossed corners a sensitivity | one extrapolation model for both series | e.g. R = 2.5 so: 774–2118 crossed, none matched |
| F45: inside the window | inside n_cert(m) | inside iff Δ₁(n*) ≥ m in its own gap model | the window of the model that produced n* | U4's [k/36] counts |
| F45: overall range | over all cases | over the windowed cases; no-window column apart | F21 rules out the unlengthened QPE beyond n_cert | R = 3.0 sf: 340–1126 against 340–1130 |
| F45: schemes beyond the sizes | the best realizable scheme | min(rep, bbht) with B_L; L forms while Δ₁ ≥ w_L + 1 mHa | no certificate at unbuilt sizes [assumption] | U4's "best" n* (§12, item 22) |
| F46: natural orbitals | eig γ at the exact natural orbitals | block-by-block diagonalization of γ₀ | degenerate occupations of different irreps (LiH's π pair) | NON errors unaffected |
| F47: cumulant norm | the linearized bound for every target | also the all-orders bound B_cum + (3/2)B_1² | the linearized bound fails at q ≳ 0.1 | U11 adds ratios 3.98–302 |
| F49: budgets past the record | usable iff bias < ε | past k_end usable only with a free loop on a converged record | C_loop(k_end) only bounds a measured loop | 400 route-A rows not usable [derived] |
| F50: success probability | a = p up to O(ε_f²) | a = p as an assumption, interval stated | the lower end differs at O(ε_f) | none (≤ 2% at the lower end) |
| F50: AA leakage | AA and filter leakage share the budget | the AA leakage alone fills the share | otherwise exact AA is never feasible | exact AA costed (N_ex = 3 or 5 calls against the ratio estimator's 2/a at the property set's start overlaps, at most 3.6× fewer [derived from F50's N_ex(a)]; 1.7–3.6× on the B0 starts of n2_r2.0 and n2_r2.4, the only starts where exact AA is the cheaper copy, §9); never a best row |
| F50, F51: model bias | F47 at q = 1 − F | the exact b_fact; F47 a sensitivity | F47 is 4–24× b_fact on LiH and blocks route B at 1e-3 | F47 would change 11 tighter and 12 looser cells (§12, item 23) |
| F51: model bias in HF-QPE | none: b = ε/√3, ε/√5 | b_m added linearly to the stencil bias | b_m is systematic | LiH dipole b = 2.000e-3 and 2.025e-4 a.u.; 98.99 and 946.6 C_QPE at the leading-order step [build check] |
| F51: field step | the leading-order h | refined against the emulated stencil bias | leading-order h carries 1.011b and 1.105b | h = 4.517e-3, 5.304e-3 a.u.; 99.55–970.25 C_QPE |
| F51: pre-projections | U forms with a certified U, possibly the start's | independent of the start (CISD(H′) or E1lb − w_L) | computed once per row set | none on LiH |
| F51: projections at the field | filters built on H′ + hO | costed once on H′ [estimate] | a factorization per field point | not evaluated (at first order the levels move by at most \|h\|r_O ≤ 0.092 Ha, with fields up to 1.06e-2 a.u. and r_O = 8.67 a.u. [derived]) |
| F52: encoding of the routes | routes costed on F9's encoding | route B and HF-QPE on F52's encoding per (target, ε) | F9's bias exhausts the tighter N2 accuracies | 12 of 132 cells change encoding; 9 would have no route on F9's (§9) |
| F53: layouts | layouts at the parameter budgets | the longest coherent run laid out; "no_layout" rows stay selectable | no factory fails below 3.6e-16 per Toffoli | 39 of 198 tighter, 2 of 198 looser rows flagged [derived] |
| F53: HF-QPE budget | F40's budgets | p_fail_ae = 1e-4 | one coherent run of a property estimate | layout only (none at 1e-6 [build check]) |
| U1: rows | the size, overlap and active-space axes and the equilibrium references | the 25 scanned systems | the design's equilibrium-reference set (h4_r1.8, h6_r1.8, h8_r1.8) serves only U5 (`loop_<s>.json`); the same systems enter other tables as members of other sets (h4_r1.8 and h8_r1.8 as M3's h4 and h8 in U8 and U9, h4_r1.8 as a closed-loop control, h4_r1.8 and h6_r1.8 in the property set) | none |
