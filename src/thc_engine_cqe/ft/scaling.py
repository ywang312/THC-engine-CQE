"""The M4 scaling study: systems, spectra, overlap, gap and cost fits, break-even overlap and size, the
cheapest measured iteration, compiled overlaps, the per-system scan, the stage cache and tables U1-U4
(F20-F25, F44 under repetition, F45).

Tags and formula labels are those of docs/m3-cost-model.md (F1-F19, F2a) and of its M4
continuation (F20-F53). Energies are in Hartree, costs in Toffolis; "C_QPE units" divide by the
system's selected C_QPE (F9); "per atom" rates are per chain site (n = N for the H chains).

System sets (F20)
-----------------
Names follow ``ft.symmetry.system_spec``: ``h<n>_r<R>`` (linear H_n, R in bohr), ``n2_r<R>``
(N2 CAS(6,6), R in angstrom), ``n2v_r<R>`` (N2 CAS(10,8)); the M3 names (``h6a``, ``lih``) keep
their labels.

    SIZE_AXIS          h<n>_r<R>, n = 4, 6, 8, 10, R = 2.5, 3.0, 3.5, 4.0 bohr (by R, then n)
    OVERLAP_AXIS       n2_r<R>, R = 1.098, 1.6, 2.0, 2.4, 2.8 A (N = 6)
    ACTIVE_SPACE_AXIS  n2v_r<R>, R = 1.098, 2.0, 2.4, 2.8 A (N = 8)
    EQUILIBRIUM        h4_r1.8, h6_r1.8, h8_r1.8
    CLOSED_LOOP        n2_r2.0, n2_r2.4, n2_r2.8 (headline); h4_r4.0, h6_r3.0, h4_r1.8, n2_r1.098
                       (controls); n2v_r2.4, n2v_r2.8, h8_r3.0 (N = 8 controls); lih (deep runs)
    PROPERTIES         h4_r{1.8, 3.0, 3.5, 4.0}, h6_r1.8, h6a, h6_r{3.0, 3.5, 4.0}, lih,
                       n2_r{1.098, 2.0, 2.4}
    HOLDOUT            h12_r3.0, h12_r4.0 (optional overlap holdout)
    QUICK              h4_r{3.0, 4.0}, h6_r{3.0, 4.0}, h8_r{3.0, 4.0}, n2_r{2.0, 2.4, 2.8}

Spectrum (:func:`spectrum`, F21) [derived]
------------------------------------------
``active.fci(nroots=n_roots, spin_square=0.0, spin_tol=tol)`` for the tolerances of
``spin_tols`` in turn: the first (1e-6, ``ActiveSpace.fci``'s default) is M3's call, so the
roots of an M3 system are those of ``ft.pipeline.run_system`` and ``exact_trajectory(...,
roots=spectrum.roots)`` reproduces M3's records bit for bit; a relaxed tolerance carries the flag
"spin_tol_relaxed". psi_0 is the normalized first root, E1_any the second, and E1 (same irrep)
M3's rule (``ft.trajectory.same_irrep_e1``: the first root group above psi_0 that holds a state
of the irrep of psi_0; if none of the roots qualifies, the highest root, a lower bound, flag
"e1_lower_bound"). With one root (overlaps only) E1 is None and E1_any nan. p_HF =
|<psi_0|HF>|^2 and E_HF are evaluated as in ``exact_trajectory`` (its overlaps[0] and
energies[0], bit for bit). The flag "ground_state_not_in_hf_irrep" marks a ground state outside
the irrep of the HF determinant (p_HF = 0 up to round-off, e.g. the square H4 rectangle, B1g).

Observed (one thread) [build check; tests/test_ft_scaling.py]: Delta_1 = E1 - E0 = 104.67 mHa
on h6_r3.0 (same and any irrep), 565.72 / 563.34 mHa on h4_r1.8 (same / any irrep), p_HF =
0.70122 on h4_r3.0, 0.11246 on n2_r2.4 and 0.11118 on n2v_r2.4. n2v_r2.4 (3136 determinants,
91 s for 12 roots) needs the relaxed tolerance: no penalty shift brings all 12 roots within 1e-6
of <S^2> = 0, and the accepted roots reach 3.0e-5. h4_r1.8, h4_r3.0, h6_r3.0, n2_r2.4, h6a, lih
and n2s do not; the square h4rect_1.8x1.8 has a B1g ground state (p_HF = 1.6e-29).

F22 (per-site overlap model) [derived for matrix-product states; assumption for HF and the iterates]
----------------------------------------------------------------------------------------------------
For two states of an open chain written as matrix-product states on a unit cell of m atoms,
<psi_0|Phi> = l^T E^(n/m) r with the mixed transfer matrix E, so ln p(n) = a - alpha n +
O(|lambda_2/lambda_1|^(n/m)), alpha = -(2/m) ln|lambda_1(E)| [derived]. HF and the CQE iterates (finite
products of exponentials of extensive two-body generators on a gapped determinant) are assumed to be such
states [assumption]; this is the orthogonality catastrophe of approximate many-body states [cited from
memory: W. Kohn, Rev. Mod. Phys. 71, 1253 (1999); unverified]. Estimators per R, record and k
(:func:`fit_log_overlap`): the successive rates alpha^(n) = [ln p(n - 2) - ln p(n)]/2
(:func:`successive_rates`), the drift delta = alpha^(n_last) - alpha^(n_prev), the Aitken limit alpha^A =
alpha^(n_last) + delta q/(1 - q) with q = delta/delta_prev if q lies in (0, 1), else alpha^(n_last) + delta
(flag "aitken_fallback"), the least-squares (a, alpha) on n in (6, 8, 10), and the extrapolation p(n) =
p(n_last) e^(-alpha (n - n_last)) at the two rates of the bracket [alpha^(n_last), alpha^A]
(:func:`extrapolate_overlap`, capped at 1; :func:`size_at_overlap` inverts it). Beyond n = 10 (12 with a
holdout) every statement is a model statement: the bracket is a "model bracket", never a confidence
interval. [build check: the HF rates at R = 3.0 bohr are 0.1007 and 0.1049 per atom (n = 4, 6, 8); p_HF and
the classical p_1 (both metrics) of H10 lie within 3% of the extrapolation from n = 4, 6, 8 at R = 3.0 and
4.0 at both rates; tests/test_ft_scaling.py.]

F21 (gap and the hand-off window) [derived; the fits estimate]
--------------------------------------------------------------
Delta_1 = E1 - E0 with M3's same-irrep E1. :func:`acceptance_margins`: the smallest gap that admits a sound
1-mHa hand-off is m_E = 3 dE (rule E), m_B = 2 dE + (E1 - E1lb) (rule B in the window model) and m_BL =
dE (1 + 2 D_L sin(theta_0)) + (E1 - E1lb) (rule B_L, sine window at M = 2 n_iter, grid 2 dE sin(theta_0),
D_L = 5 grid units with F40's verification run). :func:`fit_gap` fits Delta_1 = A n^-gamma through the two largest
sizes and A'/n with A' = n_max Delta_1(n_max) [assumption: n Delta_1 does not decrease beyond n_max];
:func:`certification_window` returns the bracket [A'/m, (A/m)^(1/gamma)] of the last size with Delta_1(n) >= m.
Beyond it the hand-off QPE is lengthened to dE Delta_1(n)/m, i.e. C_QPE(n) -> C_QPE(n) m/Delta_1(n) (F9's 1/dE
law; F45). The report takes sin(theta_0) = 1 (its upper bound, so m_BL is not underestimated) and E1 - E1lb = 1
mHa (E1lb = E1(H) - 1 mHa where no model spectrum is computed, F41) unless told otherwise.

The two-run B_L window is sound for every input state with p_k(n) >= :func:`two_run_overlap_floor` =
tau_W(5)^2/(eps_id + tau_W(5)^2) = 3.37e-5 (F40). Without the verification run the margin depends on the overlap,
D_L = D_L(p) = min{D : tau_W(D) <= eps_id p/(1 - p)} (:func:`d_l_one_run`; 5, 6, 7, 9, 13, 27 at p = 0.25, 0.1,
0.078, 0.03, 0.01, 1e-3), and the window is solved jointly with F22 (:func:`certification_window_overlap`: the
largest n with Delta_1(n) >= dE (1 + 2 D_L(p(n)) sin(theta_0)) + (E1 - E1lb)). U2 reports it as a side column for HF
and the classical Psi_1 over gap model x rate, with p at both ends of the two-run window and that window capped
where p falls below the floor. A gap that does not close (gamma <= 0) is admitted from some n on within each level
of D_L, and with a decaying overlap the one-run window still ends, at the upper edge of the last admitted level.
[build check: 37 / 35 (power law, last / Aitken) and 34 / 32 (A'/n) on R = 3.0 HF-like inputs, 126.5 with a gap
growing as n^0.817; tests/test_ft_scaling.py.]

F23 (cost components against N) [estimate]
------------------------------------------
:func:`fit_power_law`: y(N) = y_last (N/N_last)^beta with the successive exponents beta^(N) =
ln(y_N/y_{N-2})/ln(N/(N - 2)), fitted at fixed R on N >= 6 (the selected encoding switches from SF to THC
between N = 4 and 6) for C_QPE, lambda, C_W, K, V_m, W_m, C(Psi_1), C(Psi_2), ||A_1||_m and T_res(1)/C_QPE.
C_iter is never fitted directly (it mixes components and route switches); F45's c_iter(1) of the first measured
iteration is its residual alone, T_res(1)/C_QPE, one of the fitted components. Beyond n_c = 20 a crossover to
beta_loc in {2, 4} brackets locality [estimate; no data beyond n = 10].

F44 under repetition (marginal saving and break-even overlap) [derived]
-----------------------------------------------------------------------
With T(k) = C_loop(k) + (C(Psi_k) + C_QPE)/p_k (M3's rule E), iteration k pays iff C_iter(k) < mu_k, in C_QPE
units mu_k = (1 + c_k)/p_k - (1 + c_{k+1})/p_{k+1} (:func:`marginal_saving`), c_k = C(Psi_k)/C_QPE; the cost
ratio is rho_k = c_iter(k)/mu_k (:func:`cost_ratio`; another scheme's mu plugs in), and with the gain G_k =
p_{k+1}/p_k iteration k pays iff p_k < p_k* = [(1 + c_k) - (1 + c_{k+1})/G_k]/c_iter(k)
(:func:`break_even_overlap`), which depends on the gain and the costs only. [build check: T(k + 1) < T(k) of
``handoff.handoff_table`` iff rho_k < 1 iff p_k < p_k* on 300 random cases.]

F45 (break-even size) [derived; the inputs are estimates]
---------------------------------------------------------
:func:`break_even_size`: with p_k(n) from F22 (in logarithms; the scan ends where p falls below
:data:`P_FLOOR` = 1e-300, no break-even beyond), c_iter(k; n) = c_0 (n/n_0)^beta (F23, with or without the
crossover), the preparation terms c_j(n) = C(Psi_j)(n)/C_QPE(n) = c_j0 (n/n_j0)^beta_j (``prep``; F23 power laws)
and the hand-off QPE lengthened by f(n) = max(1, m/Delta_1(n)) beyond the window (F21), rho_k(n) =
c_iter(k; n)/mu_k(n) and n*_k = min{n >= n_0 : rho_k(n) < 1} over even n, None below n = 3000. In C_QPE(n) units
the lengthening enters the hand-off term only: under repetition mu_k = (f + c_k)/p_k - (f + c_{k+1})/p_{k+1}, and a
scheme whose H_s scales with C_QPE (rep, aa, bbht, fp and minima over them, with F40's (n_v - 1) C_QPE) gives mu_k =
f mu_k^s(p_k, p_{k+1}, c_k/f, c_{k+1}/f) exactly; the filtered schemes (lt, lt_bbht, lt_bbht_L, 2s, 2s_L) carry
C_QPE once, so it cancels in mu, and their C_filt or C_c does not depend on dE: mu_k = mu_k^s(p_k, p_{k+1}, c_k,
c_{k+1}) at every n (``qpe_scaled=False``). With c_k << 1, the closed form ln rho_k = ln c_0 + beta ln(n/n_0) -
ln f(n) - (alpha_k n - a_k) - ln(1 - e^((a_k - a_{k+1}) - (alpha_k - alpha_{k+1}) n)) shows that rho_k falls only
beyond n_turn ~ beta/alpha_k and that there is no n* if alpha_{k+1} >= alpha_k and a_{k+1} <= a_k (bounded gain);
it is reported as a check ("n_star_c0") next to n* with the preparation terms, which grow faster with N than
C_QPE (U3's exponents) and need not be small at n*. :func:`break_even_bracket` takes the range over rate in {last,
Aitken} x beta in {last, mean, 9} x crossover in {none, (20, 2), (20, 4)} x window in {none, m_E, m_B, m_BL} x gap
model in {power, 1/n}, with p_k, c_k and c_{k+1} at n* at both ends; p_k and p_{k+1} are extrapolated at the same
rate (matched corners of the two F22 brackets), and the crossed corners, (last, Aitken) and (Aitken, last), are
reported as a sensitivity outside the bracket. Each windowed case is inside its window iff Delta_1(n*) >= m in its
own gap model (n* <= the window end of that model for a closing gap); "overall" is the range over the windowed
cases (F21 rules out the unlengthened QPE beyond n_cert), the no-window cases its own column. [build check: equal
to a pure-Python scan of mu on synthetic fits, with and without the preparation terms and the window; no n*
without gain.]

The cheapest measured iteration (F24, F30) [derived from F10-F14]
-----------------------------------------------------------------
:func:`cheapest_iteration`: the residual of iteration k (k = 1 of the unit loop after the classical step: the
unit rule measures nothing else, F30) at every r, with every route of ``residual.compare_routes`` ("all": the
aux-bias curve and the factorization bias of the selected walk, the best at kappa) or the closed-form direct
route of F24 (ii) ("direct": T = V_m C(Psi_k)/B^2, B = r ||A_k||_m, F10 without biases, no LCU
normalizations). [build check: on h4_r3.0 the closed form equals M3's direct_mg route exactly; T_res(1)/C_QPE
of the spin-free unit loop is 0.595 / 0.0535 / 0.00595 at r = 0.3 / 1 / 3.]

F25 (sliced elementwise compilation) [derived; the 1/s law measured]
--------------------------------------------------------------------
The F3 circuit of a step is an ordered product of excitation exponentials, a first-order product formula:
||Psi~ - exp(eps D) Psi|| = O(eps^2). :func:`compiled_overlaps` compiles every step of an open-loop record
as s consecutive F3 circuits at eps/s (``circuits.compiled_step`` from the current compiled state; s
C_step(d, eps/s) Toffolis) for s in (1, 2, 4), k <= 3, N <= 8 (Fock-space oracle), and reports p~_k, the
error against the exact record, the error of the last step from the compiled state (the F25 criterion),
<S^2> and the cost. The F25 rule follows its own compiled trajectory: the smallest s in {1, 2, 4, 8, 16} with step
error <= eta_s = 0.05 from its current state, s = 16 with the flag "slice_cap" if none meets it (as
``ft.closedloop``'s compiled propagation) [design: an emulation-side calibration, not realizable at scale].
[build check: on the h4_r4.0 exact-line-search step (eps* = 0.522) the error halves from s = 1 to 2 (ratio in
[1.8, 2.2]) and p~_1 lies within 1% of p_1 at s = 2; the overshooting classical spin-orbital steps of h4_r4.0 and
h6_r4.0 need s = 16 (step error 0.033, and 0.051 with "slice_cap").]

Per-system scan (:func:`scan_system`)
-------------------------------------
The protocol per system, each stage cached (:data:`SCAN_STAGES`; :func:`scan_stage` computes one):

    stage            N <= 8                                       N = 10
    spectrum         12 singlet roots (``ft.closedloop.system_inputs``)  2 roots; 12 if root 1 is not in the
                                                                 ground irrep (flag "roots_extended")
    qpe              ``scan_encodings`` + ``select_qpe`` (F9)      the same
    residual_inputs  M3's iterate-1 variances (N <= 6); the HF    a 100-element subset at HF, seed 11
                     determinant (N = 8, F24 (i))                 (F24 (ii)); no LCU normalizations
    records          exact L-BFGS sf/so x exact/second_order;     second_order exact records; R1 so
                     R1 unit runs sf and so (``steprules``, the   (flags "no_model_spectrum",
                     F29 caps), 6 steps                           "direct_route_only", "subset_variances")
    costs            k <= 3 at r = 0.3, 1, 3: M3's F14 on the     the closed-form direct route
                     exact records, F30 on the R1 records; the
                     cheapest iteration per metric
    compiled         F25 on r1_so and the two second-order        none (flag "no_compiled_overlaps")
                     records, k <= 3, s = 1, 2, 4 and the rule

The residual problem of an iterate (``residual_inputs.build_problem`` with the aux-bias curve, 4 s at N = 8)
is built once and costed at every r (``compare_routes`` is r-dependent only through B); the costs equal
``pipeline.iteration_costs`` (exact records) and ``ft.steprules.step_rule_iteration_costs`` (R1 records) at
each r field for field (tests/test_ft_scaling.py). :func:`record_summary` is the JSON of one system.

Report (:func:`scaling_report`, :func:`markdown_tables`; tables U1-U4)
---------------------------------------------------------------------
U1: N, determinants, p_HF, E_HF - E_0, Delta_1 same / any irrep, the selected encoding, C_QPE, Q_QPE, rule-E
k_cert on the four exact records, flags. U2: per R and series (HF; the classical Psi_1 sf, so; the R1 Psi_2
sf, so) the overlaps, successive rates, the fit, alpha^A, the holdouts (every n with three smaller sizes,
from those), n(p = 1e-3) at both rates and p_1 there; per R the gap fit and n_cert brackets for m_E, m_B,
m_BL, and the one-run B_L side column with the floor of the two-run window (F21). U3: C_QPE, lambda, C_W, K, V_m
(method), W_m, C(Psi_1), C(Psi_2) (with the record of its Psi_2), ||A_1||_m, T_res(1)/C_QPE at every r with the
route; successive exponents per R on N >= 6. U4: per system, metric (R1 records), r and k <= 2: p_k, G_k,
c_iter(k), mu_k, rho_k, p_k* and the optimistic rho with the exact-line-search Psi_{k+1} of the classical-step
record; per R, metric and r the F45 bracket of the first measured iteration (k = 1, classical Psi_1 -> unit Psi_2)
under repetition: alpha_1 from the classical-step series (every size), alpha_2 from the R1 record, the cost points
T_res(1)/C_QPE on N >= 6 from the R1 record or, where it is not run (sf at N = 10), the cheapest iteration at the
same Psi_1, anchored at the last of them, and the preparation terms on N >= 6 (each fit from its own last size).
The QPE-side schemes (bbht, aa, lt_bbht, lt_bbht_L, the best realizable scheme) are applied to the same records by
the hand-off layer (``ft.amplify.scheme_saving`` and ``best_realizable_saving`` plug into :func:`cost_ratio`,
:func:`break_even_size` and :func:`break_even_sizes`, U4's n* brackets under a scheme). Every value beyond the
computed sizes is a model statement and its range a model bracket (validated over Delta n = 2 by the holdouts,
unquantified model risk beyond); the model covers H chains in canonical orbitals with elementwise compilation and
the full allowed residual.

Stage cache (:class:`Cache`)
----------------------------
One pickle per (system name, stage, digest) under the cache root (results/m4/cache for the
benchmarks). The digest is the SHA-256 of the canonical JSON of

* the build parameters of the system (``ft.symmetry.BENCHMARK_SYSTEMS[name]`` or
  ``ft.symmetry.system_spec(name)``; None for a name that is neither, e.g. a fit over systems),
* the source text of every module of the package (:func:`source_digest`: every ``*.py`` below
  the package directory, by relative path), so that any code change invalidates every entry,
* the versions of numpy, scipy, pyscf and openfermion (:func:`dependency_versions`),
* the system name, the stage name, a format number, and ``key`` (e.g. the FTParameters fields a
  stage uses).

A write goes to a temporary file in the cache directory and is renamed onto the entry with
``os.replace`` (atomic on POSIX), so a concurrent reader sees the old entry or the new one,
never a partial file. An entry that fails to unpickle is recomputed and overwritten.
``Cache(root, refresh=True)`` (the benchmarks' ``--no-cache``) ignores existing entries and
overwrites them. Entries are never deleted; the cold runs start from an empty cache directory.

Strict JSON (:func:`to_json`)
----------------------------
As ``ft.pipeline.to_json``: nan -> null, +-inf -> "inf" / "-inf" (readers map the strings back),
arrays -> lists, dataclasses -> dicts, mapping keys -> str; CI vectors, directions and model
integrals are left out (``TrajectoryRecord.states`` and ``.directions``, ``BlockEncoding.eri_model``,
``Spectrum.psi0`` and the CI vectors of ``Spectrum.roots``, ``StepRuleResult.states`` and
``.directions``, ``CQEStep.generator``, ``.residual`` and ``.residual_est``); named tuples become dicts. ``json.dumps(to_json(x), allow_nan=False)`` succeeds
for every supported x.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import pathlib
import pickle
import re
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..fci_ops import FCISpace, Hamiltonian, inner, normalize
from ..integrals import ActiveSpace
from ..measure import routes as measure_routes
from ..measure import targets
from ..steprules import StepRule, run_step_rule
from . import residual_inputs as ri
from .acceptance import EPS_ID, tail_mass
from .circuits import compiled_step
from .handoff import rule_e_certifiable
from .physical import SurfaceCodeModel
from .pipeline import IterationCost, _line_search, machines_for_campaigns, route_campaigns
from .primitives import FTParameters
from .qpe import QPECost, qpe_cost, scan_encodings, select_qpe
from .residual import PrepParams, RouteCost, best_route, compare_routes, variance_budget
from .stateprep import PrepCost, elementwise_step_cost, prep_cost, readout_rotation_qubits, readout_rotation_toffolis
from .symmetry import BENCHMARK_SYSTEMS, benchmark_system, ci_irrep, determinant_irreps, orbital_irreps, system_spec
from .trajectory import TrajectoryRecord, exact_trajectory, first_iterate_below, propagate, same_irrep_e1

__all__ = [
    "SIZE_AXIS",
    "OVERLAP_AXIS",
    "ACTIVE_SPACE_AXIS",
    "EQUILIBRIUM",
    "CLOSED_LOOP",
    "PROPERTIES",
    "HOLDOUT",
    "QUICK",
    "SPECTRUM_FLAGS",
    "Spectrum",
    "spectrum",
    "Cache",
    "source_digest",
    "dependency_versions",
    "to_json",
    "RATES",
    "RATE_FLAGS",
    "RateFit",
    "successive_rates",
    "fit_log_overlap",
    "extrapolate_overlap",
    "size_at_overlap",
    "GAP_MODELS",
    "GapFit",
    "fit_gap",
    "acceptance_margins",
    "certification_window",
    "d_l_one_run",
    "two_run_overlap_floor",
    "certification_window_overlap",
    "PowerFit",
    "fit_power_law",
    "marginal_saving",
    "break_even_overlap",
    "cost_ratio",
    "P_FLOOR",
    "break_even_size",
    "break_even_bracket",
    "break_even_sizes",
    "cheapest_iteration",
    "compiled_overlaps",
    "RECORD_KEYS",
    "SCAN_STAGES",
    "SCAN_FLAGS",
    "ScalingRecord",
    "scan_system",
    "scan_stage",
    "record_summary",
    "scaling_report",
    "markdown_tables",
]


# ------------------------------------------------------------------------- system sets (F20)
def _chains(ns, rs) -> tuple[str, ...]:
    return tuple(f"h{n}_r{r}" for r in rs for n in ns)


SIZE_AXIS: tuple[str, ...] = _chains((4, 6, 8, 10), ("2.5", "3.0", "3.5", "4.0"))
OVERLAP_AXIS: tuple[str, ...] = tuple(f"n2_r{r}" for r in ("1.098", "1.6", "2.0", "2.4", "2.8"))
ACTIVE_SPACE_AXIS: tuple[str, ...] = tuple(f"n2v_r{r}" for r in ("1.098", "2.0", "2.4", "2.8"))
EQUILIBRIUM: tuple[str, ...] = ("h4_r1.8", "h6_r1.8", "h8_r1.8")
CLOSED_LOOP: tuple[str, ...] = ("n2_r2.0", "n2_r2.4", "n2_r2.8", "h4_r4.0", "h6_r3.0", "h4_r1.8", "n2_r1.098",
                                "n2v_r2.4", "n2v_r2.8", "h8_r3.0", "lih")
PROPERTIES: tuple[str, ...] = ("h4_r1.8", "h4_r3.0", "h4_r3.5", "h4_r4.0", "h6_r1.8", "h6a", "h6_r3.0", "h6_r3.5",
                               "h6_r4.0", "lih", "n2_r1.098", "n2_r2.0", "n2_r2.4")
HOLDOUT: tuple[str, ...] = ("h12_r3.0", "h12_r4.0")
QUICK: tuple[str, ...] = ("h4_r3.0", "h4_r4.0", "h6_r3.0", "h6_r4.0", "h8_r3.0", "h8_r4.0", "n2_r2.0", "n2_r2.4",
                          "n2_r2.8")


# ------------------------------------------------------------------------- spectrum (F21)
#: Flags of :class:`Spectrum` (module docstring).
SPECTRUM_FLAGS = ("spin_tol_relaxed", "e1_lower_bound", "ground_state_not_in_hf_irrep")
# Tolerances of the consistency checks of Spectrum (Hartree; weights).
_ENERGY_TOL = 1e-9
_NORM_TOL = 1e-10


def _readonly(a: np.ndarray) -> np.ndarray:
    a = np.array(a, dtype=float, copy=True)
    a.setflags(write=False)
    return a


def _finite(name: str, x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number, got {x!r}") from None
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {x!r}")
    return v


def _irrep_id(name: str, x) -> int:
    if isinstance(x, (bool, np.bool_)) or int(x) != x or not 0 <= int(x) <= 7:
        raise ValueError(f"{name} must be a PySCF irrep id in 0..7, got {x!r}")
    return int(x)


@dataclass(frozen=True, eq=False)
class Spectrum:
    """The low singlet spectrum of an active space and the HF overlap (F21; module docstring).

    Attributes:
        name: system label.
        norb, nelec: active orbitals and (n_alpha, n_beta).
        dim: number of determinants.
        e0: exact ground-state energy (Hartree, with ecore).
        psi0: normalized ground state (PySCF CI array; read-only).
        e_hf: energy of the HF determinant.
        p_hf: |<psi_0|HF>|^2.
        e1: lowest excited singlet of the irrep of psi_0 (M3's rule; the highest root with the flag
            "e1_lower_bound" when no root qualifies); None with a single root.
        e1_any: lowest excited singlet of any irrep (the second root); nan with a single root.
        roots: (energies, CI vectors) of the n_roots singlet roots, the ``roots`` argument of
            ``ft.trajectory.exact_trajectory`` (with ``n_roots=len(roots[0])``); read-only.
        ground_irrep, hf_irrep: irrep ids of psi_0 and of the HF determinant.
        spin_tol: the <S^2> tolerance the roots were accepted at.
        flags: a subset of :data:`SPECTRUM_FLAGS`.
    """

    name: str
    norb: int
    nelec: tuple[int, int]
    dim: int
    e0: float
    psi0: np.ndarray
    e_hf: float
    p_hf: float
    e1: float | None
    e1_any: float
    roots: tuple
    ground_irrep: int
    hf_irrep: int
    spin_tol: float
    flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise ValueError(f"name must be a string, got {self.name!r}")
        if isinstance(self.norb, (bool, np.bool_)) or int(self.norb) != self.norb or self.norb < 1:
            raise ValueError(f"norb must be a positive integer, got {self.norb!r}")
        object.__setattr__(self, "norb", int(self.norb))
        try:
            nelec = tuple(int(x) for x in self.nelec)
        except (TypeError, ValueError):
            raise ValueError(f"nelec must be (n_alpha, n_beta), got {self.nelec!r}") from None
        if len(nelec) != 2 or any(x != y for x, y in zip(nelec, self.nelec)) \
                or not all(0 <= x <= self.norb for x in nelec):
            raise ValueError(f"nelec must be two integers in 0..norb = {self.norb}, got {self.nelec!r}")
        object.__setattr__(self, "nelec", nelec)
        if isinstance(self.dim, (bool, np.bool_)) or int(self.dim) != self.dim or self.dim < 1:
            raise ValueError(f"dim must be a positive integer, got {self.dim!r}")
        object.__setattr__(self, "dim", int(self.dim))
        e0 = _finite("e0", self.e0)
        object.__setattr__(self, "e0", e0)
        if np.iscomplexobj(self.psi0):
            raise ValueError("psi0 must be a real CI vector")
        psi0 = _readonly(self.psi0)
        if psi0.size != self.dim or not np.all(np.isfinite(psi0)):
            raise ValueError(f"psi0 must be a finite CI array of dim = {self.dim} entries, got {psi0.shape}")
        if abs(float(np.linalg.norm(psi0)) - 1.0) > _NORM_TOL:
            raise ValueError(f"psi0 must be normalized (norm {np.linalg.norm(psi0):.12g})")
        object.__setattr__(self, "psi0", psi0)
        e_hf = _finite("e_hf", self.e_hf)
        if e_hf < e0 - _ENERGY_TOL:
            raise ValueError(f"e_hf = {e_hf!r} lies below e0 = {e0!r} (not variational)")
        object.__setattr__(self, "e_hf", e_hf)
        p_hf = _finite("p_hf", self.p_hf)
        if not -_NORM_TOL <= p_hf <= 1.0 + _NORM_TOL:
            raise ValueError(f"p_hf must lie in [0, 1], got {self.p_hf!r}")
        object.__setattr__(self, "p_hf", p_hf)
        try:
            energies, cis = self.roots
        except (TypeError, ValueError):
            raise ValueError("roots must be a pair (energies, CI vectors)") from None
        energies = _readonly(np.asarray(energies, dtype=float).ravel())
        cis = tuple(_readonly(c) for c in cis)
        n = energies.size
        if n < 1 or len(cis) != n or not np.all(np.isfinite(energies)):
            raise ValueError(f"roots must hold n >= 1 finite energies and as many CI vectors, got {n} and {len(cis)}")
        if any(c.size != self.dim for c in cis):
            raise ValueError(f"every root must have dim = {self.dim} entries")
        if energies[0] != e0 or np.any(np.diff(energies) < -_ENERGY_TOL):
            raise ValueError("roots must be in ascending order with roots[0][0] == e0")
        object.__setattr__(self, "roots", (energies, cis))
        if self.e1 is None:
            if n > 1:
                raise ValueError("e1 is None only for a single root")
        else:
            e1 = _finite("e1", self.e1)
            if e1 < e0 - _ENERGY_TOL:
                raise ValueError(f"e1 = {e1!r} lies below e0 = {e0!r}")
            object.__setattr__(self, "e1", e1)
        e1_any = float(self.e1_any)
        if n > 1:
            if e1_any != energies[1]:
                raise ValueError("e1_any must be the second root")
        elif not math.isnan(e1_any):
            raise ValueError("e1_any must be nan for a single root")
        object.__setattr__(self, "e1_any", e1_any)
        object.__setattr__(self, "ground_irrep", _irrep_id("ground_irrep", self.ground_irrep))
        object.__setattr__(self, "hf_irrep", _irrep_id("hf_irrep", self.hf_irrep))
        tol = _finite("spin_tol", self.spin_tol)
        if not tol > 0.0:
            raise ValueError(f"spin_tol must be positive, got {self.spin_tol!r}")
        object.__setattr__(self, "spin_tol", tol)
        flags = tuple(self.flags)
        if any(f not in SPECTRUM_FLAGS for f in flags) or len(set(flags)) != len(flags):
            raise ValueError(f"flags must be distinct entries of {SPECTRUM_FLAGS}, got {self.flags!r}")
        if ("ground_state_not_in_hf_irrep" in flags) != (self.ground_irrep != self.hf_irrep):
            raise ValueError("the flag ground_state_not_in_hf_irrep must mark ground_irrep != hf_irrep")
        object.__setattr__(self, "flags", flags)

    @property
    def n_roots(self) -> int:
        """Number of singlet roots computed."""
        return int(self.roots[0].size)

    @property
    def gap(self) -> float:
        """Delta_1 = E1 - E0 (same irrep; nan without E1)."""
        return math.nan if self.e1 is None else self.e1 - self.e0

    @property
    def gap_any(self) -> float:
        """E1_any - E0 (nan with a single root)."""
        return self.e1_any - self.e0


def _spin_tols(spin_tols) -> tuple[float, ...]:
    try:
        tols = tuple(float(t) for t in spin_tols)
    except (TypeError, ValueError):
        raise ValueError(f"spin_tols must be a sequence of positive tolerances, got {spin_tols!r}") from None
    if not tols or not all(t > 0.0 and math.isfinite(t) for t in tols) or any(b <= a for a, b in zip(tols, tols[1:])):
        raise ValueError(f"spin_tols must be increasing positive tolerances, got {spin_tols!r}")
    return tols


def spectrum(active: ActiveSpace, irreps: np.ndarray, *, n_roots: int = 12, spin_tols=(1e-6, 1e-4)) -> Spectrum:
    """The singlet roots, psi_0, E1 (same and any irrep), E_HF and p_HF of ``active`` (F21; module docstring).

    Args:
        active: the active space (a symmetry-adapted build of ``ft.symmetry.benchmark_system``).
        irreps: its orbital irrep ids (``ft.symmetry.orbital_irreps``).
        n_roots: singlet roots to compute (>= 1; E1 needs >= 2).
        spin_tols: increasing <S^2> tolerances tried in turn (``ActiveSpace.fci``'s ``spin_tol``); a
            tolerance beyond the first sets the flag "spin_tol_relaxed".

    Raises RuntimeError (from ``ActiveSpace.fci``) if no tolerance admits n_roots singlet roots.
    """
    if isinstance(n_roots, (bool, np.bool_)) or int(n_roots) != n_roots or n_roots < 1:
        raise ValueError(f"n_roots must be a positive integer, got {n_roots!r}")
    n_roots = int(n_roots)
    tols = _spin_tols(spin_tols)
    norb, nelec = active.norb, tuple(active.nelec)
    ir = np.asarray(irreps)
    if ir.shape != (norb,) or ir.dtype.kind not in "iu":
        raise ValueError(f"irreps must be ({norb},) integers, got {ir.shape} {ir.dtype}")
    ir = ir.astype(np.int64)
    flags: list[str] = []
    error: RuntimeError | None = None
    for i, tol in enumerate(tols):
        try:
            out = active.fci(nroots=n_roots, spin_square=0.0, spin_tol=tol)
        except RuntimeError as exc:
            error = exc
            continue
        if i > 0:
            flags.append("spin_tol_relaxed")
        break
    else:
        raise RuntimeError(f"{active.label or 'active space'}: no {n_roots} singlet roots within <S^2> tolerances "
                           f"{tols}") from error
    if n_roots == 1:
        es, cis = np.array([float(out[0])]), [np.asarray(out[1])]
    else:
        es, cis = np.asarray(out[0], dtype=float), [np.asarray(c) for c in out[1]]
    space = FCISpace(norb, nelec)
    psi0 = normalize(np.asarray(cis[0]))
    hf = normalize(space.hf())
    ground = ci_irrep(psi0, norb, nelec, ir)[0]
    hf_irrep = ci_irrep(hf, norb, nelec, ir)[0]
    e1: float | None = None
    e1_any = math.nan
    if n_roots > 1:
        e1_any = float(es[1])
        e1 = same_irrep_e1(es, cis, determinant_irreps(norb, nelec, ir), ground)
        if e1 is None:
            e1 = float(es[-1])
            flags.append("e1_lower_bound")
    if ground != hf_irrep:
        flags.append("ground_state_not_in_hf_irrep")
    e_hf = Hamiltonian(active).energy_and_variance(hf)[0]
    return Spectrum(name=active.label, norb=norb, nelec=nelec, dim=int(psi0.size), e0=float(es[0]), psi0=psi0,
                    e_hf=float(e_hf), p_hf=float(abs(complex(inner(psi0, hf))) ** 2), e1=e1, e1_any=e1_any,
                    roots=(es, cis), ground_irrep=ground, hf_irrep=hf_irrep, spin_tol=tol, flags=tuple(flags))


# ------------------------------------------------------------------------- strict JSON
# Fields left out of the JSON: CI vectors, directions and model integrals (large arrays).
_SKIP_FIELDS = {"TrajectoryRecord": ("states", "directions"), "BlockEncoding": ("eri_model",),
                "StepRuleResult": ("states", "directions"), "CQEStep": ("generator", "residual", "residual_est"),
                "Spectrum": ("psi0", "roots")}


def _number(x: float) -> float | str | None:
    if math.isnan(x):
        return None
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def to_json(obj) -> Any:
    """A strict-JSON form of ``obj`` (module docstring): nan -> None, +-inf -> "inf" / "-inf", arrays -> lists,
    dataclasses -> dicts without their CI vectors, directions and model integrals (a Spectrum keeps the root
    energies as "root_energies"), named tuples -> dicts, other tuples -> lists, mapping keys -> str.

    Raises TypeError for anything else (complex numbers, callables, ...).
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return _number(float(obj))
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            raise TypeError("cannot convert a complex array to JSON")
        return [to_json(x) for x in obj.tolist()]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        skip = _SKIP_FIELDS.get(type(obj).__name__, ())
        out = {f.name: to_json(getattr(obj, f.name)) for f in dataclasses.fields(obj) if f.name not in skip}
        if isinstance(obj, Spectrum):
            out["root_energies"] = to_json(obj.roots[0])
        return out
    if isinstance(obj, Mapping):
        return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return {name: to_json(v) for name, v in zip(obj._fields, obj)}
    if isinstance(obj, (list, tuple)):
        return [to_json(x) for x in obj]
    raise TypeError(f"cannot convert {type(obj).__name__} to JSON")


# ------------------------------------------------------------------------- the stage cache
_CACHE_FORMAT = 1
_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
_DEPENDENCIES = ("numpy", "scipy", "pyscf", "openfermion")


def _tree_digest(root: pathlib.Path) -> str:
    """SHA-256 over the relative path and the bytes of every ``*.py`` file below ``root`` (``__pycache__`` excluded),
    in sorted path order."""
    root = pathlib.Path(root)
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.relative_to(root).parts):
        h.update(path.relative_to(root).as_posix().encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


@functools.lru_cache(maxsize=1)
def source_digest() -> str:
    """The digest of the package's source text (every ``*.py`` of ``thc_engine_cqe``; computed once per process)."""
    return _tree_digest(pathlib.Path(__file__).resolve().parent.parent)


@functools.lru_cache(maxsize=1)
def dependency_versions() -> dict[str, str | None]:
    """Installed versions of numpy, scipy, pyscf and openfermion (None if not installed)."""
    out: dict[str, str | None] = {}
    for dist in _DEPENDENCIES:
        try:
            out[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            out[dist] = None
    return out


def _build_spec(name: str) -> dict | None:
    if name in BENCHMARK_SYSTEMS:
        return BENCHMARK_SYSTEMS[name]
    try:
        return system_spec(name)
    except ValueError:
        return None


def _part(label: str, value) -> str:
    if not isinstance(value, str) or not _PART.fullmatch(value):
        raise ValueError(f"{label} must be a non-empty string of letters, digits and ._+- (not starting with one "
                         f"of ._+-), got {value!r}")
    return value


class Cache:
    """Pickled results per (system, stage, digest) with atomic writes (module docstring).

    Attributes:
        root: the cache directory (created on the first write).
        refresh: ignore existing entries and overwrite them (``--no-cache``).
        hits, misses: entries loaded and computed by :meth:`get` so far.
    """

    def __init__(self, root: str | os.PathLike, *, refresh: bool = False) -> None:
        self.root = pathlib.Path(root)
        self.refresh = bool(refresh)
        self.hits = 0
        self.misses = 0

    def __repr__(self) -> str:
        return f"Cache({str(self.root)!r}, refresh={self.refresh})"

    def digest(self, name: str, stage: str, *, key: Mapping | None = None) -> str:
        """The SHA-256 digest of the entry (module docstring)."""
        _part("name", name)
        _part("stage", stage)
        if key is not None and not isinstance(key, Mapping):
            raise ValueError(f"key must be a mapping or None, got {type(key).__name__}")
        payload = {"format": _CACHE_FORMAT, "name": name, "stage": stage, "build": to_json(_build_spec(name)),
                   "source": source_digest(), "versions": dependency_versions(),
                   "key": None if key is None else to_json(key)}
        text = json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()

    def path(self, name: str, stage: str, *, key: Mapping | None = None) -> pathlib.Path:
        """The file of the entry: ``<root>/<name>__<stage>__<digest>.pkl``."""
        return self.root / f"{name}__{stage}__{self.digest(name, stage, key=key)}.pkl"

    def get(self, name: str, stage: str, compute: Callable[[], Any], *, key: Mapping | None = None) -> Any:
        """The cached value of (name, stage, key), or ``compute()``, written atomically."""
        if not callable(compute):
            raise ValueError("compute must be a callable without arguments")
        path = self.path(name, stage, key=key)
        if not self.refresh and path.is_file():
            try:
                with open(path, "rb") as fh:
                    value = pickle.load(fh)
            except Exception:                           # a corrupt or stale pickle: recompute
                pass
            else:
                self.hits += 1
                return value
        value = compute()
        self._write(path, value)
        self.misses += 1
        return value

    def _write(self, path: pathlib.Path, value: Any) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise


# ------------------------------------------------------------------------- F22: per-site overlap model
#: Rates of the F22 bracket: the last successive rate and the Aitken limit.
RATES = ("last", "aitken")
#: Flags of :class:`RateFit`: "aitken_fallback" (fewer than three rates, or q outside (0, 1): alpha^A = alpha_last
#: + drift), "no_drift" (one rate: alpha^A = alpha_last).
RATE_FLAGS = ("aitken_fallback", "no_drift")
_P_TOL = 1e-10                  # round-off allowed above p = 1
_CONSISTENCY_RTOL = 1e-9


def _sizes(ns, what: str = "sizes") -> np.ndarray:
    try:
        n = np.asarray(ns)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a sequence of integers, got {ns!r}") from None
    if n.ndim != 1 or n.dtype.kind not in "iuf":
        raise ValueError(f"{what} must be a 1-D sequence of integers, got {ns!r}")
    if n.dtype.kind == "f":
        if not np.all(np.isfinite(n)) or np.any(n != np.round(n)):
            raise ValueError(f"{what} must be integers, got {ns!r}")
    return n.astype(np.int64)


def _points(ns, ys, what: str, *, overlaps: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(n, y) sorted by n: >= 2 distinct positive integer sizes and as many finite positive values."""
    n = _sizes(ns)
    try:
        y = np.asarray(ys, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be real numbers, got {ys!r}") from None
    if y.shape != n.shape or n.size < 2:
        raise ValueError(f"need at least two sizes and as many {what}, got {n.shape} and {y.shape}")
    order = np.argsort(n, kind="stable")
    n, y = n[order], y[order]
    if n[0] < 1 or np.any(np.diff(n) <= 0):
        raise ValueError(f"sizes must be distinct positive integers, got {ns!r}")
    if not np.all(np.isfinite(y)) or np.any(y <= 0.0):
        raise ValueError(f"{what} must be finite and positive, got {ys!r}")
    if overlaps and np.any(y > 1.0 + _P_TOL):
        raise ValueError(f"overlaps must lie in (0, 1], got {ys!r}")
    return n, (np.minimum(y, 1.0) if overlaps else y)


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=_CONSISTENCY_RTOL, abs_tol=1e-15)


def successive_rates(ns: Sequence[int], ps: Sequence[float]) -> np.ndarray:
    """F22: the successive rates alpha^(n_i) = [ln p(n_{i-1}) - ln p(n_i)]/(n_i - n_{i-1}) (per atom).

    ``ns`` are chain lengths (sorted here), ``ps`` overlaps in (0, 1]; for the chains of :data:`SIZE_AXIS`
    (step 2) this is [ln p(n - 2) - ln p(n)]/2. Returns the len(ns) - 1 rates in increasing n.
    """
    n, p = _points(ns, ps, "overlaps", overlaps=True)
    lp = np.log(p)
    return (lp[:-1] - lp[1:]) / np.diff(n)


@dataclass(frozen=True)
class RateFit:
    """The F22 fit of ln p_k against n at one R and record (module docstring).

    Attributes:
        R: bond length (bohr; nan if not a chain).
        record: label of the record (e.g. "hf", "classical_sf", "r1_so").
        k: the iterate.
        ns, p: the sizes and overlaps, increasing n.
        successive: alpha^(n) for n = ns[1:] (per atom).
        a, alpha: least-squares ln p = a - alpha n on the fit window.
        drift: alpha^(n_last) - alpha^(n_prev) (nan with one rate).
        alpha_aitken: the Aitken limit alpha^A (or its fallback, see ``flags``).
        flags: a subset of :data:`RATE_FLAGS`.
    """

    R: float
    record: str
    k: int
    ns: tuple[int, ...]
    p: tuple[float, ...]
    successive: tuple[float, ...]
    a: float
    alpha: float
    drift: float
    alpha_aitken: float
    flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        r = float(self.R)
        if math.isinf(r):
            raise ValueError(f"R must be finite or nan, got {self.R!r}")
        object.__setattr__(self, "R", r)
        if not isinstance(self.record, str):
            raise ValueError(f"record must be a string, got {self.record!r}")
        if isinstance(self.k, (bool, np.bool_)) or int(self.k) != self.k or self.k < 0:
            raise ValueError(f"k must be a non-negative integer, got {self.k!r}")
        object.__setattr__(self, "k", int(self.k))
        n, p = _points(self.ns, self.p, "overlaps", overlaps=True)
        if tuple(int(x) for x in n) != tuple(self.ns):
            raise ValueError(f"ns must be increasing, got {self.ns!r}")
        object.__setattr__(self, "ns", tuple(int(x) for x in n))
        object.__setattr__(self, "p", tuple(float(x) for x in p))
        succ = np.asarray(self.successive, dtype=float)
        ref = (np.log(p[:-1]) - np.log(p[1:])) / np.diff(n)
        if succ.shape != ref.shape or not np.allclose(succ, ref, rtol=_CONSISTENCY_RTOL, atol=1e-14):
            raise ValueError("successive must be the successive rates of (ns, p)")
        object.__setattr__(self, "successive", tuple(float(x) for x in succ))
        for name in ("a", "alpha", "alpha_aitken"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        drift = float(self.drift)
        if (succ.size >= 2) != math.isfinite(drift):
            raise ValueError("drift must be finite with two or more rates and nan with one")
        if succ.size >= 2 and not _close(drift, float(succ[-1] - succ[-2])):
            raise ValueError("drift must be the difference of the last two rates")
        object.__setattr__(self, "drift", drift)
        flags = tuple(self.flags)
        if any(f not in RATE_FLAGS for f in flags) or len(set(flags)) != len(flags):
            raise ValueError(f"flags must be distinct entries of {RATE_FLAGS}, got {self.flags!r}")
        object.__setattr__(self, "flags", flags)

    @property
    def alpha_last(self) -> float:
        """The last successive rate alpha^(n_last)."""
        return self.successive[-1]

    def rate(self, rate: str = "last") -> float:
        """alpha of the extrapolation: "last" (alpha^(n_last)) or "aitken" (alpha^A)."""
        if rate not in RATES:
            raise ValueError(f"rate must be one of {RATES}, got {rate!r}")
        return self.alpha_last if rate == "last" else self.alpha_aitken


def _window(window) -> tuple[int, ...]:
    w = _sizes(window, "window")
    if w.size == 0:
        raise ValueError("window must not be empty")
    return tuple(int(x) for x in w)


def fit_log_overlap(ns: Sequence[int], ps: Sequence[float], *, window=(6, 8, 10), R: float = math.nan,
                    record: str = "", k: int = 0) -> RateFit:
    """F22 at one R, record and k: successive rates, drift, Aitken limit and the least-squares (a, alpha).

    (a, alpha) fit ln p = a - alpha n on the sizes of ``ns`` inside ``window`` (at least two). The drift is
    delta = alpha^(n_last) - alpha^(n_prev); with three or more rates q = delta/delta_prev and alpha^A =
    alpha^(n_last) + delta q/(1 - q) if q lies in (0, 1), else alpha^(n_last) + delta (flag
    "aitken_fallback", also with exactly two rates); with one rate alpha^A = alpha^(n_last) (flag "no_drift").
    """
    n, p = _points(ns, ps, "overlaps", overlaps=True)
    sel = np.isin(n, _window(window))
    if int(sel.sum()) < 2:
        raise ValueError(f"the fit window {tuple(window)} holds fewer than two of the sizes {tuple(n.tolist())}")
    lp = np.log(p)
    x = n[sel].astype(float)
    (a, slope), *_ = np.linalg.lstsq(np.column_stack([np.ones(x.size), x]), lp[sel], rcond=None)
    succ = (lp[:-1] - lp[1:]) / np.diff(n)
    flags: list[str] = []
    drift = math.nan
    if succ.size >= 2:
        drift = float(succ[-1] - succ[-2])
    if succ.size >= 3:
        prev = float(succ[-2] - succ[-3])
        q = drift / prev if prev != 0.0 else math.nan
        if 0.0 < q < 1.0:
            aitken = float(succ[-1]) + drift * q / (1.0 - q)
        else:
            aitken = float(succ[-1]) + drift
            flags.append("aitken_fallback")
    elif succ.size == 2:
        aitken = float(succ[-1]) + drift
        flags.append("aitken_fallback")
    else:
        aitken = float(succ[-1])
        flags.append("no_drift")
    return RateFit(R=float(R), record=str(record), k=int(k), ns=tuple(int(v) for v in n),
                   p=tuple(float(v) for v in p), successive=tuple(float(v) for v in succ), a=float(a),
                   alpha=float(-slope), drift=drift, alpha_aitken=aitken, flags=tuple(flags))


def _log_overlap(fit: RateFit, n: np.ndarray, rate: str) -> np.ndarray:
    """ln p(n) of the F22 extrapolation, capped at 0."""
    lp = math.log(fit.p[-1]) - fit.rate(rate) * (np.asarray(n, dtype=float) - fit.ns[-1])
    return np.minimum(lp, 0.0)


def _extrapolate(fit: RateFit, n: np.ndarray, rate: str) -> np.ndarray:
    return np.exp(_log_overlap(fit, n, rate))


def extrapolate_overlap(fit: RateFit, n: float, *, rate: str = "last") -> float:
    """F22: p(n) = p(n_last) exp(-alpha (n - n_last)) with alpha = ``fit.rate(rate)``, capped at 1."""
    n = _finite("n", n)
    return float(_extrapolate(fit, np.array(n), rate))


def size_at_overlap(fit: RateFit, p_target: float, *, rate: str = "last") -> float:
    """The (real) chain length at which the F22 extrapolation reaches ``p_target`` (inf if it never does)."""
    pt = _finite("p_target", p_target)
    if not 0.0 < pt <= 1.0:
        raise ValueError(f"p_target must lie in (0, 1], got {p_target!r}")
    alpha = fit.rate(rate)
    lp_last, lt = math.log(fit.p[-1]), math.log(pt)
    if alpha > 0.0 or (alpha < 0.0 and lt > lp_last):
        return float(fit.ns[-1] + (lp_last - lt) / alpha)
    if lt == lp_last:
        return float(fit.ns[-1])
    return math.inf


# ------------------------------------------------------------------------- F21: gap fits and the window
#: Gap models of the F21 bracket: Delta_1 = A n^-gamma ("power") and A'/n ("inverse").
GAP_MODELS = ("power", "inverse")


@dataclass(frozen=True)
class GapFit:
    """F21: Delta_1(n) = A n^-gamma through the two largest sizes, and A'/n with A' = n_max Delta_1(n_max).

    Attributes:
        A, gamma: the power law (Hartree, dimensionless).
        A_inv: A' (Hartree).
        ns, gaps: the sizes and same-irrep gaps Delta_1 (Hartree), increasing n.
    """

    A: float
    gamma: float
    A_inv: float
    ns: tuple[int, ...]
    gaps: tuple[float, ...]

    def __post_init__(self) -> None:
        n, g = _points(self.ns, self.gaps, "gaps")
        if tuple(int(x) for x in n) != tuple(self.ns):
            raise ValueError(f"ns must be increasing, got {self.ns!r}")
        object.__setattr__(self, "ns", tuple(int(x) for x in n))
        object.__setattr__(self, "gaps", tuple(float(x) for x in g))
        for name in ("A", "gamma", "A_inv"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if not (self.A > 0.0 and self.A_inv > 0.0):
            raise ValueError(f"A and A_inv must be positive, got {self.A!r}, {self.A_inv!r}")
        n_max, g_max = self.ns[-1], self.gaps[-1]
        if not (_close(self.A_inv, n_max * g_max) and _close(self.A * n_max ** -self.gamma, g_max)):
            raise ValueError("A and A_inv must reproduce the gap at the largest size")

    def gap(self, n: float, model: str = "power") -> float:
        """Delta_1(n) of the model ("power": A n^-gamma; "inverse": A'/n)."""
        if model not in GAP_MODELS:
            raise ValueError(f"model must be one of {GAP_MODELS}, got {model!r}")
        n = _finite("n", n)
        if not n > 0.0:
            raise ValueError(f"n must be positive, got {n!r}")
        return self.A * n ** -self.gamma if model == "power" else self.A_inv / n

    def as_power_law(self, model: str = "power") -> "GapFit":
        """The fit whose power law is ``model``: itself, or A'/n written as A = A', gamma = 1."""
        if model not in GAP_MODELS:
            raise ValueError(f"model must be one of {GAP_MODELS}, got {model!r}")
        return self if model == "power" else dataclasses.replace(self, A=self.A_inv, gamma=1.0)


def fit_gap(ns: Sequence[int], gaps: Sequence[float]) -> GapFit:
    """F21: gamma = ln(Delta(n_prev)/Delta(n_max))/ln(n_max/n_prev) on the two largest sizes, A = Delta(n_max)
    n_max^gamma and A' = n_max Delta(n_max)."""
    n, g = _points(ns, gaps, "gaps")
    gamma = math.log(g[-2] / g[-1]) / math.log(n[-1] / n[-2])
    return GapFit(A=float(g[-1] * float(n[-1]) ** gamma), gamma=float(gamma), A_inv=float(n[-1] * g[-1]),
                  ns=tuple(int(v) for v in n), gaps=tuple(float(v) for v in g))


def acceptance_margins(de_qpe: float, sin_theta0: float, e1_margin: float, *, d_l: int = 5) -> dict:
    """F21: the smallest gap that admits a sound hand-off at QPE error ``de_qpe``, per rule (Hartree).

    "E": rule E, 3 dE; "B": rule B in the window model, 2 dE + (E1 - E1lb); "BL": rule B_L with the sine
    window at M = 2 n_iter (grid g = 2 dE sin(theta_0)), dE (1 + 2 D_L sin(theta_0)) + (E1 - E1lb), with
    ``e1_margin`` = E1 - E1lb and D_L = ``d_l`` grid units.
    """
    de = _finite("de_qpe", de_qpe)
    s = _finite("sin_theta0", sin_theta0)
    em = _finite("e1_margin", e1_margin)
    if not de > 0.0 or not 0.0 < s <= 1.0 or em < 0.0:
        raise ValueError(f"need de_qpe > 0, 0 < sin_theta0 <= 1 and e1_margin >= 0, got {de_qpe!r}, "
                         f"{sin_theta0!r}, {e1_margin!r}")
    if isinstance(d_l, (bool, np.bool_)) or int(d_l) != d_l or d_l < 0:
        raise ValueError(f"d_l must be a non-negative integer, got {d_l!r}")
    return {"E": 3.0 * de, "B": 2.0 * de + em, "BL": de * (1.0 + 2.0 * int(d_l) * s) + em}


def certification_window(fit: GapFit, margin: float) -> tuple[float, float]:
    """F21: the bracket [A'/m, (A/m)^(1/gamma)] of the largest n with Delta_1(n) >= m (inverse model first). A power
    law with gamma <= 0 does not close: its end is inf for a growing gap (gamma < 0) and for a constant one with
    A >= m, and 0 (no n) for a constant one below m."""
    m = _finite("margin", margin)
    if not m > 0.0:
        raise ValueError(f"margin must be positive, got {margin!r}")
    if fit.gamma > 0.0:
        upper = (fit.A / m) ** (1.0 / fit.gamma)
    else:
        upper = math.inf if fit.gamma < 0.0 or fit.A >= m else 0.0
    return float(fit.A_inv / m), float(upper)


def _window_end(fit: GapFit, margin: float, model: str) -> float:
    """The largest n with Delta_1(n) >= margin in one gap model ("power": (A/m)^(1/gamma); "inverse": A'/m)."""
    lo, hi = certification_window(fit, margin)
    return hi if model == "power" else lo


#: B_L's top margin in grid units with F40's verification run (two runs).
_D_L_TWO_RUNS = 5
# Grid distances searched for the one-run D_L(p) (the domain of the tail of ft.acceptance).
_D_L_MAX = 1000


def d_l_one_run(p: float, *, eps_id: float = EPS_ID) -> int:
    """F40: D_L(p) = min{D in N : tau_W(D) <= eps_id p/(1 - p)}, the B_L top margin (grid units of the sine law at
    M = 2 n_iter, ``ft.acceptance.tail_mass``) that makes one run sound at overlap p (0 at p = 1).

    [build check: 5, 6, 7, 9, 13, 27 at p = 0.25, 0.1, 0.078, 0.03, 0.01, 1e-3; tests/test_ft_scaling.py.]
    ValueError if no D <= 1000 suffices.
    """
    p = _finite("p", p)
    eps = _finite("eps_id", eps_id)
    if not 0.0 < p <= 1.0 or not 0.0 < eps < 1.0:
        raise ValueError(f"need 0 < p <= 1 and 0 < eps_id < 1, got {p!r}, {eps_id!r}")
    if p >= 1.0:
        return 0
    threshold = eps * p / (1.0 - p)
    for d in range(_D_L_MAX + 1):
        if tail_mass(float(d), window="sine") <= threshold:
            return d
    raise ValueError(f"no D_L <= {_D_L_MAX} grid units is sound with one run at p = {p!r}")


def two_run_overlap_floor(*, eps_id: float = EPS_ID, d_l: int = _D_L_TWO_RUNS) -> float:
    """F40: tau_W(D_L)^2/(eps_id + tau_W(D_L)^2), the smallest overlap for which B_L with its verification run
    (two runs at D_L grid units) is sound to eps_id (3.37e-5 at D_L = 5, eps_id = 1e-3)."""
    eps = _finite("eps_id", eps_id)
    if not 0.0 < eps < 1.0:
        raise ValueError(f"eps_id must lie in (0, 1), got {eps_id!r}")
    if isinstance(d_l, (bool, np.bool_)) or int(d_l) != d_l or d_l < 0:
        raise ValueError(f"d_l must be a non-negative integer, got {d_l!r}")
    tau = tail_mass(float(d_l), window="sine")
    return float(tau ** 2 / (eps + tau ** 2))


def certification_window_overlap(gap: GapFit, fit: RateFit, *, de_qpe: float, sin_theta0: float, e1_margin: float,
                                 eps_id: float = EPS_ID, rate: str = "last", model: str = "power") -> float:
    """F21's one-run B_L window, solved jointly with F22: the largest (real) n with Delta_1(n) >= M(n) =
    dE (1 + 2 D_L(p(n)) sin(theta_0)) + (E1 - E1lb), p(n) = ``extrapolate_overlap(fit, n, rate=rate)``, D_L(p) =
    :func:`d_l_one_run` (no verification run; its domain D_L <= 1000 grid units) and Delta_1 = ``gap.gap(n, model)``.

    D_L(p) <= d iff p >= q_d = min_{d' <= d} tau_W(d')/(eps_id + tau_W(d')), so with a decaying overlap (rate > 0)
    the level D_L = d holds on n in (u_{d-1}, u_d], u_d the last n with p(n) >= q_d, which moves outward with d while
    the margin M_d rises. A closing gap (A'/n, or the power law with gamma > 0) falls: the window ends where Delta_1
    meets the margin of one level, or where D_L steps up. A constant overlap (rate 0) has one level; a growing one
    (rate < 0) its first admitted level, scanning d upward. A power law with gamma <= 0 does not close: with a
    decaying overlap, level d holds from n = (M_d/A)^(1/|gamma|) on (all of it if gamma = 0 and A >= M_d, none if
    A < M_d), and the window ends at u_d of the largest such level (smaller n need not be admitted); tau_W does not
    increase with D (``ft.acceptance``), so u_d does not decrease and a bisection over blocks of levels finds it.
    Where the overlap does not decay, the margin does not grow with n: inf if the gap reaches its large-n value
    (always for gamma < 0), else no n. 0 if no n > 0 is admitted; ValueError if the window does not end within
    D_L <= 1000.
    """
    if not isinstance(gap, GapFit) or not isinstance(fit, RateFit):
        raise ValueError("gap must be a GapFit and fit a RateFit")
    if model not in GAP_MODELS:
        raise ValueError(f"model must be one of {GAP_MODELS}, got {model!r}")
    acceptance_margins(de_qpe, sin_theta0, e1_margin)               # validates the three inputs
    de, s, em = float(de_qpe), float(sin_theta0), float(e1_margin)
    eps = _finite("eps_id", eps_id)
    if not 0.0 < eps < 1.0:
        raise ValueError(f"eps_id must lie in (0, 1), got {eps_id!r}")
    alpha = fit.rate(rate)
    lp_last, n_last = math.log(fit.p[-1]), float(fit.ns[-1])
    if model == "power" and gap.gamma <= 0.0:
        return _one_run_window_open(gap, alpha, lp_last, n_last, de=de, s=s, em=em, eps=eps)
    edge, q = -math.inf, math.inf
    for d in range(_D_L_MAX + 1):
        tau = tail_mass(float(d), window="sine")
        q = min(q, tau / (eps + tau))                          # q < 1: the cap of p at 1 does not enter
        g = _window_end(gap, de * (1.0 + 2.0 * d * s) + em, model)
        if alpha > 0.0:                                        # level d: n in (edge, u], u the last n with p >= q
            if g <= edge:
                return float(edge)
            u = n_last + (lp_last - math.log(q)) / alpha
            if g < u:
                return float(g)
            edge = u
        elif alpha == 0.0:
            if min(lp_last, 0.0) >= math.log(q):
                return float(g)
        elif g >= n_last + (lp_last - math.log(q)) / alpha:    # level d: n in [u_d, u_{d-1}), u falling with d
            return float(g)
    raise ValueError(f"no one-run window within D_L <= {_D_L_MAX} grid units")


def _one_run_window_open(gap: GapFit, alpha: float, lp_last: float, n_last: float, *, de: float, s: float, em: float,
                         eps: float) -> float:
    """:func:`certification_window_overlap` for the power law with gamma <= 0, Delta_1 = A n^w with w = -gamma >= 0."""
    w = -gap.gamma

    def margin(d: int) -> float:
        return de * (1.0 + 2.0 * d * s) + em

    if alpha <= 0.0:                         # D_L(p(n)) does not grow with n; for large n it is D_L(lim p)
        if w > 0.0:
            return math.inf
        d_inf = 0 if alpha < 0.0 else d_l_one_run(math.exp(min(lp_last, 0.0)), eps_id=eps)
        return math.inf if gap.A >= margin(d_inf) else 0.0

    def u(d: int) -> float:                  # the last n with D_L(p(n)) <= d (q_d of the non-increasing tail)
        tau = tail_mass(float(d), window="sine")
        return n_last + (lp_last - math.log(tau / (eps + tau))) / alpha

    def reaches(d_margin: int, d_end: int) -> bool:              # Delta_1(u_{d_end}) >= M_{d_margin}
        n = u(d_end)
        return n > 0.0 and math.log(margin(d_margin) / gap.A) <= w * math.log(n)

    blocks = [(0, _D_L_MAX)]
    while blocks:                            # depth first, the upper half of a block before the lower one
        a, b = blocks.pop()
        if not reaches(a, b):                # M_d >= M_a and u_d <= u_b for every level d of [a, b]
            continue
        if a == b:
            if a == _D_L_MAX:
                raise ValueError(f"no one-run window within D_L <= {_D_L_MAX} grid units")
            return float(u(a))
        mid = (a + b) // 2
        blocks += [(a, mid), (mid + 1, b)]
    return 0.0


# ------------------------------------------------------------------------- F23: cost components against N
@dataclass(frozen=True)
class PowerFit:
    """F23: y(N) = y_last (N/N_last)^beta with the successive exponents of consecutive sizes.

    Attributes:
        x, y: sizes and values (positive), increasing x.
        successive: beta^(x_i) = ln(y_i/y_{i-1})/ln(x_i/x_{i-1}), i = 1..
        beta_last, beta_mean: the last successive exponent and their mean.
    """

    x: tuple[float, ...]
    y: tuple[float, ...]
    successive: tuple[float, ...]
    beta_last: float
    beta_mean: float

    def __post_init__(self) -> None:
        try:
            x = np.asarray(self.x, dtype=float)
            y = np.asarray(self.y, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("x and y must be real sequences") from None
        if x.ndim != 1 or y.shape != x.shape or x.size < 2:
            raise ValueError(f"need at least two points, got {x.shape} and {y.shape}")
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))) or np.any(x <= 0.0) or np.any(y <= 0.0) \
                or np.any(np.diff(x) <= 0.0):
            raise ValueError("x must be increasing and positive, y positive, both finite")
        object.__setattr__(self, "x", tuple(float(v) for v in x))
        object.__setattr__(self, "y", tuple(float(v) for v in y))
        ref = np.log(y[1:] / y[:-1]) / np.log(x[1:] / x[:-1])
        succ = np.asarray(self.successive, dtype=float)
        if succ.shape != ref.shape or not np.allclose(succ, ref, rtol=_CONSISTENCY_RTOL, atol=1e-14):
            raise ValueError("successive must be the successive exponents of (x, y)")
        object.__setattr__(self, "successive", tuple(float(v) for v in succ))
        bl, bm = _finite("beta_last", self.beta_last), _finite("beta_mean", self.beta_mean)
        if not (_close(bl, float(succ[-1])) and _close(bm, float(np.mean(succ)))):
            raise ValueError("beta_last and beta_mean must be the last and the mean successive exponent")
        object.__setattr__(self, "beta_last", bl)
        object.__setattr__(self, "beta_mean", bm)

    def predict(self, x: float, beta: float | None = None) -> float:
        """y_last (x/x_last)^beta (default beta_last)."""
        b = self.beta_last if beta is None else _finite("beta", beta)
        return self.y[-1] * (_finite("x", x) / self.x[-1]) ** b


def fit_power_law(xs: Sequence[float], ys: Sequence[float]) -> PowerFit:
    """F23: the successive exponents of ``ys`` against ``xs`` (sorted by x), their last value and mean."""
    try:
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("xs and ys must be real sequences") from None
    if x.ndim != 1 or y.shape != x.shape or x.size < 2:
        raise ValueError(f"need at least two points, got {x.shape} and {y.shape}")
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))) or np.any(x <= 0.0) or np.any(y <= 0.0) \
            or np.any(np.diff(x) <= 0.0):
        raise ValueError("xs must be distinct and positive, ys positive, both finite")
    succ = np.log(y[1:] / y[:-1]) / np.log(x[1:] / x[:-1])
    return PowerFit(x=tuple(x.tolist()), y=tuple(y.tolist()), successive=tuple(succ.tolist()),
                    beta_last=float(succ[-1]), beta_mean=float(np.mean(succ)))


# ------------------------------------------------------------------------- F44 under repetition
def _nonneg(name: str, x):
    v = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(v)) or np.any(v < 0.0):
        raise ValueError(f"{name} must be finite and >= 0, got {x!r}")
    return v


def _overlap_arg(name: str, x):
    v = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(v)) or np.any(v <= 0.0) or np.any(v > 1.0 + _P_TOL):
        raise ValueError(f"{name} must lie in (0, 1], got {x!r}")
    return v


def _scalar_or_array(v: np.ndarray):
    return float(v) if np.ndim(v) == 0 else v


def marginal_saving(p_k, p_k1, c_k=0.0, c_k1=0.0) -> float:
    """F44 under repetition (M3's rule E, F15), in C_QPE units: mu_k = (1 + c_k)/p_k - (1 + c_{k+1})/p_{k+1},
    c_k = C(Psi_k)/C_QPE. Iteration k pays iff c_iter(k) < mu_k. Accepts arrays (broadcast)."""
    pk, pk1 = _overlap_arg("p_k", p_k), _overlap_arg("p_k1", p_k1)
    ck, ck1 = _nonneg("c_k", c_k), _nonneg("c_k1", c_k1)
    return _scalar_or_array((1.0 + ck) / pk - (1.0 + ck1) / pk1)


def break_even_overlap(c_iter, gain, c_k=0.0, c_k1=0.0) -> float:
    """F44: p_k* = [(1 + c_k) - (1 + c_{k+1})/G_k]/c_iter(k) with the gain G_k = p_{k+1}/p_k (C_QPE units).

    Iteration k pays under repetition iff p_k < p_k*, which depends on the gain and the costs only. With
    c_iter = 0: inf if the bracket is positive (it always pays), else 0 (it never does)."""
    c = float(_nonneg("c_iter", c_iter))
    g = _finite("gain", gain)
    if not g > 0.0:
        raise ValueError(f"gain must be positive, got {gain!r}")
    num = (1.0 + float(_nonneg("c_k", c_k))) - (1.0 + float(_nonneg("c_k1", c_k1))) / g
    if c > 0.0:
        return num / c
    return math.inf if num > 0.0 else 0.0


def cost_ratio(c_iter, p_k, p_k1, c_k=0.0, c_k1=0.0, *, saving: Callable | None = None) -> float:
    """F44: rho_k = c_iter(k)/mu_k (inf if mu_k <= 0); iteration k pays iff rho_k < 1.

    ``saving(p_k, p_k1, c_k, c_k1) -> mu_k`` defaults to :func:`marginal_saving` (repetition); a QPE-side scheme
    plugs in its own (``ft.amplify.scheme_saving``). c_iter may be +inf (an infeasible route)."""
    c = float(np.asarray(c_iter, dtype=float))
    if math.isnan(c) or c < 0.0:
        raise ValueError(f"c_iter must be >= 0 (inf allowed), got {c_iter!r}")
    mu = float((marginal_saving if saving is None else saving)(p_k, p_k1, c_k, c_k1))
    if not mu > 0.0:
        return math.inf
    return c / mu


# ------------------------------------------------------------------------- F45: break-even size
_N_MAX = 3000
_N_STEP = 2                     # closed-shell chains: even n
_BETA_FIXED = 9.0               # the fixed exponent of the F23 cost bracket
_CROSSOVERS = (None, (20, 2.0), (20, 4.0))
#: Smallest extrapolated overlap of the F45 scan: the scan ends at the first n where p_k or p_k+1 falls below it.
P_FLOOR = 1e-300
_LN_P_FLOOR = math.log(P_FLOOR)
# Sizes per call of a saving hook: the scan stops after the first block with rho < 1.
_HOOK_BLOCK = 128


def _saving_values(saving: Callable | None, pk: np.ndarray, pk1: np.ndarray, ck: np.ndarray | None = None,
                   ck1: np.ndarray | None = None) -> np.ndarray:
    """mu_k = saving(p_k, p_k1, c_k, c_k1) on arrays (the array path of the hook, else elementwise; repetition for
    ``saving`` None); c_k = c_k1 = 0 where not given."""
    ck = np.zeros_like(pk) if ck is None else ck
    ck1 = np.zeros_like(pk) if ck1 is None else ck1
    if saving is None:
        return (1.0 + ck) / pk - (1.0 + ck1) / pk1
    try:
        out = np.asarray(saving(pk, pk1, ck, ck1), dtype=float)
        if out.shape == pk.shape:
            return out
    except (TypeError, ValueError):
        pass
    return np.array([float(saving(float(a), float(b), float(x), float(y))) for a, b, x, y in zip(pk, pk1, ck, ck1)])


def _cost_curve(c0: float, n0: int, beta: float, ns: np.ndarray, crossover) -> np.ndarray:
    if crossover is None:
        return c0 * (ns / n0) ** beta
    n_c, beta_loc = crossover
    n_c = max(float(n_c), float(n0))
    return np.where(ns <= n_c, c0 * (ns / n0) ** beta, c0 * (n_c / n0) ** beta * (ns / n_c) ** beta_loc)


def _check_crossover(crossover):
    if crossover is None:
        return None
    try:
        n_c, beta_loc = crossover
    except (TypeError, ValueError):
        raise ValueError(f"crossover must be (n_c, beta_loc) or None, got {crossover!r}") from None
    n_c = _finite("crossover n_c", n_c)
    if not n_c >= 1.0:
        raise ValueError(f"crossover n_c must be >= 1, got {n_c!r}")
    return n_c, _finite("crossover beta_loc", beta_loc)


def _check_prep(prep) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """None, or the power laws ((c_0, n_0, beta) of c_k, of c_k+1) of c_j(n) = c_0 (n/n_0)^beta (a PowerFit gives
    (y_last, x_last, beta_last))."""
    if prep is None:
        return None
    try:
        terms = tuple(prep)
    except TypeError:
        raise ValueError(f"prep must be a pair of power laws or None, got {prep!r}") from None
    if len(terms) != 2:
        raise ValueError(f"prep must be a pair (c_k, c_k1) of power laws, got {prep!r}")
    out = []
    for label, term in zip(("c_k", "c_k1"), terms):
        if isinstance(term, PowerFit):
            term = (term.y[-1], term.x[-1], term.beta_last)
        try:
            c, n, b = term
        except (TypeError, ValueError):
            raise ValueError(f"prep {label} must be (c_0, n_0, beta) or a PowerFit, got {term!r}") from None
        c = float(_nonneg(f"prep {label} c_0", c))
        n, b = _finite(f"prep {label} n_0", n), _finite(f"prep {label} beta", b)
        if not n > 0.0:
            raise ValueError(f"prep {label} n_0 must be positive, got {n!r}")
        out.append((c, n, b))
    return out[0], out[1]


def _prep_curve(term: tuple[float, float, float], ns) -> np.ndarray:
    c, n, b = term
    return c * (np.asarray(ns, dtype=float) / n) ** b


def _qpe_scaled(saving: Callable | None, qpe_scaled) -> bool:
    """Whether the hand-off cost of ``saving`` scales with C_QPE (the repetition default does)."""
    if saving is None:
        if qpe_scaled is not None and not qpe_scaled:
            raise ValueError("qpe_scaled=False needs a saving hook (the repetition default scales with C_QPE)")
        return True
    flag = getattr(saving, "qpe_scaled", True) if qpe_scaled is None else qpe_scaled
    if not isinstance(flag, (bool, np.bool_)):
        raise ValueError(f"qpe_scaled must be a bool, got {flag!r}")
    return bool(flag)


def _break_even_scan(fit_k: RateFit, fit_k1: RateFit, c0: float, n0: int, beta: float, *, crossover, gap, margin,
                     saving, rates: tuple[str, str], n_max: int, prep, qpe_scaled) -> int | None:
    """F45 with p_k(n) and p_k+1(n) each extrapolated at its own rate (``rates``)."""
    if not isinstance(fit_k, RateFit) or not isinstance(fit_k1, RateFit):
        raise ValueError("fit_k and fit_k1 must be RateFit")
    c0 = float(_nonneg("c0", c0))
    n0 = int(_sizes([n0], "n0")[0])
    if n0 < 1:
        raise ValueError(f"n0 must be >= 1, got {n0!r}")
    beta = _finite("beta", beta)
    crossover = _check_crossover(crossover)
    if (gap is None) != (margin is None):
        raise ValueError("gap and margin must be given together")
    if gap is not None and not _finite("margin", margin) > 0.0:
        raise ValueError(f"margin must be positive, got {margin!r}")
    for rate in rates:
        if rate not in RATES:
            raise ValueError(f"rate must be one of {RATES}, got {rate!r}")
    n_max = int(n_max)
    if n_max < n0:
        raise ValueError(f"n_max ({n_max}) must be >= n0 ({n0})")
    prep = _check_prep(prep)
    scaled = _qpe_scaled(saving, qpe_scaled)
    ns = np.arange(n0, n_max + 1, _N_STEP, dtype=float)
    lpk, lpk1 = _log_overlap(fit_k, ns, rates[0]), _log_overlap(fit_k1, ns, rates[1])
    keep = (lpk >= _LN_P_FLOOR) & (lpk1 >= _LN_P_FLOOR)
    if not keep.all():
        stop = int(np.argmin(keep))
        ns, lpk, lpk1 = ns[:stop], lpk[:stop], lpk1[:stop]
    if ns.size == 0:
        return None
    pk, pk1 = np.exp(lpk), np.exp(lpk1)
    c = _cost_curve(c0, n0, beta, ns, crossover)
    f = np.ones_like(ns) if gap is None else np.maximum(1.0, float(margin) / (gap.A * ns ** -gap.gamma))
    if prep is None:
        ck = ck1 = np.zeros_like(ns)
    else:
        ck, ck1 = _prep_curve(prep[0], ns), _prep_curve(prep[1], ns)
    step = ns.size if saving is None else _HOOK_BLOCK
    for i in range(0, ns.size, step):
        s = slice(i, i + step)
        if scaled:
            mu = _saving_values(saving, pk[s], pk1[s], ck[s] / f[s], ck1[s] / f[s]) * f[s]
        else:
            mu = _saving_values(saving, pk[s], pk1[s], ck[s], ck1[s])
        pos = mu > 0.0
        rho = np.full(mu.shape, np.inf)
        rho[pos] = c[s][pos] / mu[pos]
        hit = np.flatnonzero(rho < 1.0)
        if hit.size:
            return int(ns[s][hit[0]])
    return None


def break_even_size(fit_k: RateFit, fit_k1: RateFit, c0: float, n0: int, beta: float, *,
                    crossover: tuple[int, float] | None = None, gap: GapFit | None = None,
                    margin: float | None = None, saving: Callable | None = None, rate: str = "last",
                    n_max: int = _N_MAX, prep=None, qpe_scaled: bool | None = None) -> int | None:
    """F45: the smallest even n >= n0 (up to ``n_max``) with rho_k(n) = c_iter(k; n)/mu_k(n) < 1, or None.

    p_k(n) and p_{k+1}(n) are the F22 extrapolations of ``fit_k`` and ``fit_k1``, both at ``rate`` (matched corners of
    the two brackets), evaluated in logarithms; the scan ends at the first n where either falls below
    :data:`P_FLOOR` (no break-even from there on). c_iter(k; n) = c0 (n/n0)^beta, with ``crossover`` = (n_c, beta_loc)
    continued as (n/n_c)^beta_loc beyond n_c (F23). ``prep`` = (c_k, c_k1), each (c_0, n_0, beta) or a PowerFit:
    the preparation costs c_j(n) = C(Psi_j)(n)/C_QPE(n) = c_0 (n/n_0)^beta (None: c_k = c_k+1 = 0, F45's c_k << 1
    form). f(n) = max(1, m/Delta_1(n)) lengthens the hand-off QPE beyond the window (``gap``'s power law and the
    margin m = ``margin``; f = 1 without them), C_QPE(n) -> f(n) C_QPE(n). In C_QPE(n) units:

    * ``qpe_scaled`` True (the default for ``saving`` None, i.e. repetition, and for a hook without a
      ``qpe_scaled`` attribute): H_s(p, c) scales with C_QPE at fixed C(Psi) (rep, aa, bbht, fp and their minima,
      with F40's (n_v - 1) C_QPE), so mu_k(n) = f saving(p_k, p_k+1, c_k/f, c_k+1/f) exactly; under repetition
      mu_k = (f + c_k)/p_k - (f + c_k+1)/p_k+1, the preparation terms are not lengthened;
    * ``qpe_scaled`` False (the filtered schemes lt, lt_bbht, lt_bbht_L, 2s, 2s_L, whose H_s carries C_QPE once,
      so that it cancels in mu, and whose C_filt or C_c does not depend on dE): mu_k(n) = saving(p_k, p_k+1, c_k,
      c_k+1), independent of f, with the hook's C_filt in C_QPE(n) units.

    ``saving(p_k, p_k1, c_k, c_k1) -> mu_k`` defaults to repetition (:func:`marginal_saving`); a hook that mixes
    the two kinds (a minimum over rep and a filtered scheme) is exact under neither beyond the window.
    """
    return _break_even_scan(fit_k, fit_k1, c0, n0, beta, crossover=crossover, gap=gap, margin=margin, saving=saving,
                            rates=(rate, rate), n_max=n_max, prep=prep, qpe_scaled=qpe_scaled)


def break_even_bracket(fit_k: RateFit, fit_k1: RateFit, cost_fit: PowerFit, c0: float, n0: int, *,
                       gap: GapFit | None = None, margins: Mapping[str, float] | None = None,
                       saving: Callable | None = None, prep=None, qpe_scaled: bool | None = None) -> dict:
    """F45 over the brackets of the extrapolation protocol (module docstring).

    Cases: rate in {last, aitken} for both fits (F22; matched corners) x beta in {last, mean successive of
    ``cost_fit``, 9} x crossover in {none, (20, 2), (20, 4)} (F23) x window in {none} and, with ``gap`` and
    ``margins`` (label -> m), {each margin} x gap model in {power, inverse} (F21); ``prep``, ``saving`` and
    ``qpe_scaled`` as in :func:`break_even_size`.

    Returns {"n0", "c0", "betas", "prep" (the two power laws or None), "qpe_scaled", "cases", "overall",
    "overall_scope", "no_window", "by_margin" (with a gap), "crossed"}. A case holds rate (rate_k, rate_k1), beta,
    crossover, margin, gap_model, n_star, and at n*: p_k, p_k1, c_k, c_k1 (C_QPE(n) units) and f; with a window
    "n_cert" (the window end of its own gap model: (A/m)^(1/gamma) or A'/m) and "inside" (Delta_1(n*) >= m in that
    model, f = 1 at n*; for a closing gap n* <= n_cert); "n_star_c0" is n* with c_k = c_k+1 = 0 (F45's c_k << 1
    form, a check of the preparation terms). A summary holds "n_star" [min, max] over the cases with an n* (None if
    none has), p_k, p_k1, c_k, c_k1 at both ends, "n_none", "n_cases", "n_star_c0", "n_none_c0" and, over windowed
    cases, "n_inside" (cases with n* inside the window of their own gap model); "overall" summarizes every windowed
    case ("overall_scope" "windowed"; the no-window cases without a gap, "no_window"), "no_window" the cases with
    f = 1, and "by_margin" {label: summary with "margin", "n_cert" = sorted ``certification_window``,
    "n_cert_by_model", "n_star_min_by_model", "inside_any" (n_inside > 0) and "inside_all" (n_inside == n_cases)}.
    "crossed" is the sensitivity to the crossed corners, (p_k, p_k+1) at (last, Aitken) and (Aitken, last), never
    part of the bracket: {"cases", "overall", "no_window", "by_margin"} as above.
    """
    if not isinstance(cost_fit, PowerFit):
        raise ValueError(f"cost_fit must be a PowerFit, got {type(cost_fit).__name__}")
    if (gap is None) != (margins is None):
        raise ValueError("gap and margins must be given together")
    prep = _check_prep(prep)
    scaled = _qpe_scaled(saving, qpe_scaled)
    betas = (("last", cost_fit.beta_last), ("mean", cost_fit.beta_mean), ("fixed_9", _BETA_FIXED))
    windows: list[tuple[str | None, str | None, float | None]] = [(None, None, None)]
    if gap is not None:
        windows += [(str(label), model, float(m)) for label, m in margins.items() for model in GAP_MODELS]

    def case(rate_k: str, rate_k1: str, blabel: str, beta: float, cross, mlabel, model, m) -> dict:
        g = None if model is None else gap.as_power_law(model)
        kw = dict(crossover=cross, gap=g, margin=m, saving=saving, rates=(rate_k, rate_k1), n_max=_N_MAX,
                  qpe_scaled=scaled)
        n_star = _break_even_scan(fit_k, fit_k1, c0, n0, beta, prep=prep, **kw)
        n_c0 = n_star if prep is None else _break_even_scan(fit_k, fit_k1, c0, n0, beta, prep=None, **kw)
        out = {"rate": rate_k if rate_k == rate_k1 else f"{rate_k}/{rate_k1}", "rate_k": rate_k, "rate_k1": rate_k1,
               "beta": blabel, "beta_value": float(beta),
               "crossover": None if cross is None else [int(cross[0]), float(cross[1])],
               "margin": mlabel, "gap_model": model, "n_star": n_star, "p_k": None, "p_k1": None, "c_k": None,
               "c_k1": None, "f": None, "n_cert": None if model is None else _window_end(gap, m, model),
               "inside": None, "n_star_c0": n_c0}
        if n_star is not None:
            out.update(p_k=extrapolate_overlap(fit_k, n_star, rate=rate_k),
                       p_k1=extrapolate_overlap(fit_k1, n_star, rate=rate_k1),
                       c_k=0.0 if prep is None else float(_prep_curve(prep[0], n_star)),
                       c_k1=0.0 if prep is None else float(_prep_curve(prep[1], n_star)),
                       f=1.0 if g is None else max(1.0, float(m) / g.gap(n_star)))
        if model is not None:
            out["inside"] = bool(n_star is not None and g.gap(n_star) >= float(m))
        return out

    def summary(sub: list[dict]) -> dict:
        hits = [c for c in sub if c["n_star"] is not None]
        out = {"n_none": len(sub) - len(hits), "n_cases": len(sub), "n_star": None, "p_k": None, "p_k1": None,
               "c_k": None, "c_k1": None}
        if hits:
            lo = min(hits, key=lambda c: c["n_star"])
            hi = max(hits, key=lambda c: c["n_star"])
            out.update({key: [lo[key], hi[key]] for key in ("n_star", "p_k", "p_k1", "c_k", "c_k1")})
        c0_hits = [c["n_star_c0"] for c in sub if c["n_star_c0"] is not None]
        out["n_star_c0"] = [min(c0_hits), max(c0_hits)] if c0_hits else None
        out["n_none_c0"] = len(sub) - len(c0_hits)
        if any(c["margin"] is not None for c in sub):
            out["n_inside"] = sum(1 for c in sub if c["inside"])
        return out

    def by_margin(cases: list[dict]) -> dict:
        res = {}
        for label, m in margins.items():
            sub = [c for c in cases if c["margin"] == str(label)]
            s = summary(sub)
            s["margin"] = float(m)
            s["n_cert"] = sorted(certification_window(gap, m))
            s["n_cert_by_model"] = {model: _window_end(gap, m, model) for model in GAP_MODELS}
            s["n_star_min_by_model"] = {model: min((c["n_star"] for c in sub if c["gap_model"] == model
                                                    and c["n_star"] is not None), default=None)
                                        for model in GAP_MODELS}
            s["inside_any"] = s["n_inside"] > 0
            s["inside_all"] = s["n_inside"] == s["n_cases"]
            res[str(label)] = s
        return res

    def block(rate_pairs) -> dict:
        cases = [case(rk, rk1, blabel, beta, cross, mlabel, model, m)
                 for (rk, rk1), (blabel, beta), cross, (mlabel, model, m)
                 in itertools.product(rate_pairs, betas, _CROSSOVERS, windows)]
        windowed = [c for c in cases if c["margin"] is not None]
        res = {"cases": cases, "overall": summary(windowed if gap is not None else cases),
               "overall_scope": "windowed" if gap is not None else "no_window",
               "no_window": summary([c for c in cases if c["margin"] is None])}
        if gap is not None:
            res["by_margin"] = by_margin(cases)
        return res

    out = {"n0": int(n0), "c0": float(c0), "betas": {label: float(b) for label, b in betas},
           "prep": None if prep is None else {label: {"c0": t[0], "n0": t[1], "beta": t[2]}
                                              for label, t in zip(("c_k", "c_k1"), prep)},
           "qpe_scaled": scaled, **block([(rate, rate) for rate in RATES])}
    crossed = block([("last", "aitken"), ("aitken", "last")])
    crossed.pop("overall_scope")
    out["crossed"] = crossed
    return out


# ------------------------------------------------------------------------- the cheapest iteration (F24, F30)
ROUTE_SETS = ("all", "direct")
_SAMPLING_ROUTES = ("direct_mg", "aux_mg")
_FLAG_NO_LAYOUT = "no layout within the failure budget"


def _with_r(params: FTParameters, r) -> FTParameters:
    r = _finite("r", r)
    if not r > 0.0:
        raise ValueError(f"r must be positive, got {r!r}")
    return params if r == params.r else dataclasses.replace(params, r=r)


def _r_values(r_values) -> tuple[float, ...]:
    try:
        rs = tuple(float(r) for r in r_values)
    except (TypeError, ValueError):
        raise ValueError(f"r_values must be a sequence of positive numbers, got {r_values!r}") from None
    if not rs or not all(math.isfinite(r) and r > 0.0 for r in rs) or len(set(rs)) != len(rs):
        raise ValueError(f"r_values must be distinct positive numbers, got {r_values!r}")
    return rs


def _direct_closed_form(v_direct: np.ndarray, a: np.ndarray, m: np.ndarray, prep: PrepParams, r: float) -> RouteCost:
    """F10 without biases, T = V_m C(Psi_k)/B^2 with B = r ||A_k||_m: ``residual.sampling_direct`` evaluated on the
    arrays directly (same arithmetic: ``routes.shots_for_relative_error``, ``residual.variance_budget``), so it needs
    no LCU normalizations."""
    shots = measure_routes.shots_for_relative_error(v_direct, a, r, weights=m)
    info = {"r": r, "variance_m": float(np.sum(m * v_direct)), "bias_m": 0.0}
    if not math.isfinite(shots):
        return RouteCost("direct_mg", math.inf, math.inf, math.nan, math.inf, math.inf, int(prep.logical_qubits),
                         dict(info, budget=0.0), ("infeasible",))
    info["budget"] = variance_budget(math.sqrt(float(np.sum(m * a ** 2))), r)
    return RouteCost("direct_mg", shots, shots * prep.toffolis, prep.toffolis, prep.toffolis, shots,
                     prep.logical_qubits, info)


class _ResidualAt:
    """The residual inputs of one state, built once and costed at any r (F14 "all", or F24 (ii) "direct")."""

    def __init__(self, active: ActiveSpace, psi: np.ndarray, walk, inputs: Mapping, routes: str,
                 params: FTParameters, model_active: ActiveSpace | None = None) -> None:
        self.routes, self.params = routes, params
        el = np.asarray(inputs["elements"])
        var = inputs["variances"]
        self.m = targets.element_multiplicities(el, params.kind, active.norb).astype(float)
        self.v_direct = np.asarray(var["v_direct"], dtype=float)
        self.w_aux = np.asarray(var["w_aux"], dtype=float)
        if routes == "all":
            self.problem = ri.build_problem(active, psi, el, params.kind, variances=var,
                                            normalizations=inputs["normalizations"], deltas=params.aux_deltas)
            model_active = walk.model(active) if model_active is None else model_active
            self.bias = ri.factorization_bias(active, model_active, psi, el, params.kind)
            self.walk = walk.walk_params(self.bias)
            self.a = self.problem.A
            self.norm = self.problem.norm_a_m
        else:
            self.problem, self.walk, self.bias = None, None, math.nan
            self.a = np.asarray(measure_routes.exact_residual(active, psi, el, params.kind), dtype=float)
            self.norm = math.sqrt(float(np.sum(self.m * self.a ** 2)))

    def routes_at(self, prep: PrepParams, r: float) -> tuple[RouteCost, ...]:
        if self.routes == "all":
            return tuple(compare_routes(self.problem, prep, self.walk, params=_with_r(self.params, r)))
        return (_direct_closed_form(self.v_direct, self.a, self.m, prep, float(r)),)


def cheapest_iteration(active: ActiveSpace, record: TrajectoryRecord, prep: PrepCost, walk, inputs: Mapping, *,
                       k: int = 1, r_values=(0.3, 1.0, 3.0), routes: str = "all",
                       params: FTParameters = FTParameters()) -> dict:
    """The residual cost of iteration ``k`` of ``record`` at every r (module docstring; F14, F24, F30).

    Args:
        active: the active space; record and prep: the trajectory (states) and its elementwise preparation cost.
        walk: the selected block encoding (``ft.qpe.BlockEncoding``; C_QPE = ``qpe_cost(walk, params.de_qpe)``).
        inputs: {"elements", "variances"} (``ft.residual_inputs`` formats) and, for ``routes="all"``,
            "normalizations".
        k: the iterate (1: the cheapest measured iteration after the classical step).
        r_values: residual accuracies (F10's r).
        routes: "all" (every route of ``residual.compare_routes`` with the aux-bias curve and the factorization
            bias, the best at ``params.kappa``) or "direct" (the closed-form direct route of F24 (ii), F10 without
            biases).

    Returns {"k", "routes", "metric", "first_step", "c_qpe", "c_prep", "q_prep", "norm_a_m", "variance_m",
    "w_aux_m", "fact_bias_m" (nan for "direct"), "n_elements", "by_r" {f"{r:g}": {"best", "toffolis",
    "over_qpe", "shots" (preparations of the best route), "logical_qubits", "budget", "route_toffolis" (every
    route at ``params.kappa``; gradient estimation excluded)}}}.
    """
    if routes not in ROUTE_SETS:
        raise ValueError(f"routes must be one of {ROUTE_SETS}, got {routes!r}")
    if isinstance(k, (bool, np.bool_)) or int(k) != k or not 0 <= k <= record.n_steps:
        raise ValueError(f"k must be an iterate 0..{record.n_steps}, got {k!r}")
    k = int(k)
    if len(prep.cumulative) != len(record.states):
        raise ValueError(f"prep covers {len(prep.cumulative)} states, the record {len(record.states)}")
    rs = _r_values(r_values)
    prep_k = PrepParams(prep.cumulative[k], prep.logical_qubits[k])
    c_qpe = float(qpe_cost(walk, params.de_qpe).toffolis)
    res = _ResidualAt(active, record.states[k], walk, inputs, routes, params)
    by_r = {}
    for r in rs:
        costs = res.routes_at(prep_k, r)
        best = best_route(costs, kappa=params.kappa)
        tof = {c.route: float(c.toffolis) for c in costs if c.route != "gaw"
               and ("kappa" not in c.params or math.isclose(c.params["kappa"], params.kappa, rel_tol=1e-9))}
        by_r[f"{r:g}"] = {"best": best.route, "toffolis": float(best.toffolis),
                          "over_qpe": float(best.toffolis) / c_qpe, "shots": float(best.n_prep),
                          "logical_qubits": int(best.logical_qubits), "budget": float(best.params.get("budget", 0.0)),
                          "route_toffolis": tof}
    return {"k": k, "routes": routes, "metric": record.metric, "first_step": record.first_step, "c_qpe": c_qpe,
            "c_prep": float(prep_k.toffolis), "q_prep": int(prep_k.logical_qubits), "norm_a_m": float(res.norm),
            "variance_m": float(np.sum(res.m * res.v_direct)), "w_aux_m": float(np.sum(res.m * res.w_aux)),
            "fact_bias_m": float(res.bias), "n_elements": int(res.m.size), "by_r": by_r}


def _classical_iteration(prep_k: PrepParams) -> IterationCost:
    return IterationCost(k=0, residual=(), best="classical", line_search=None, toffolis=0.0, shots=0.0,
                         seconds_one_machine=0.0, machines_7d=0, best_practical="classical",
                         prep_toffolis=float(prep_k.toffolis))


def _iteration_costs_by_r(active: ActiveSpace, record: TrajectoryRecord, prep: PrepCost, walk, inputs: Mapping, *,
                          r_values: Sequence[float], k_max: int, line_search: bool, energy_split: str,
                          readout_toffolis: int, routes: str = "all", result=None,
                          params: FTParameters = FTParameters(),
                          model: SurfaceCodeModel = SurfaceCodeModel()) -> dict[float, list[IterationCost]]:
    """{r: C_iter(k), k = 0..k_max} with one residual problem per iterate for every r.

    ``line_search=True``: M3's F14 costs, ``pipeline.iteration_costs`` at ``params.r = r`` (classical iteration 0
    on a "second_order" record); ``line_search=False``: the unit rule's F30 costs,
    ``ft.steprules.step_rule_iteration_costs`` at r (``result``, if given, must charge one residual at Psi_k per
    iteration and nothing else). Both bit for bit with ``routes="all"`` (tests/test_ft_scaling.py); "direct"
    replaces the routes by the closed-form direct route (F24 (ii)).
    """
    rs = _r_values(r_values)
    classical = record.first_step == "second_order"
    q_rot = readout_rotation_qubits(active.norb, params=params)
    model_active = walk.model(active) if routes == "all" else None
    out: dict[float, list[IterationCost]] = {r: [] for r in rs}
    for k in range(int(k_max) + 1):
        prep_k = PrepParams(prep.cumulative[k], prep.logical_qubits[k])
        if k == 0 and classical:
            for r in rs:
                out[r].append(_classical_iteration(prep_k))
            continue
        if result is not None:
            events = result.iteration_events(k)
            if len(events) != 1 or events[0].kind != "residual" or events[0].eps != 0.0:
                raise ValueError(f"iteration {k}: the ledger must charge one residual at Psi_k and nothing else")
        ls = None
        if line_search:
            ls, flagged, ls_campaigns, ls_ae = _line_search(active, record, prep_k, k, energy_split, readout_toffolis,
                                                            q_rot, walk, params, model)
        res = _ResidualAt(active, record.states[k], walk, inputs, routes, params, model_active=model_active)
        for r in rs:
            p_r = _with_r(params, r)
            costs = res.routes_at(prep_k, r)
            best = best_route(costs, kappa=params.kappa)
            best_p = best_route(costs, kappa=params.kappa_practical)
            t_res, t_res_p = best.toffolis, best_p.toffolis
            shots = best.n_prep if best.route in _SAMPLING_ROUTES else 0.0
            if line_search:
                campaigns = tuple(route_campaigns(best, p_r, model)) + ls_campaigns
                flags = (_FLAG_NO_LAYOUT,) if any(c.per_run is None and 0 < c.n_runs < math.inf
                                                  for c in campaigns) else ()
                out[r].append(IterationCost(
                    k=k, residual=costs, best=best.route, line_search=ls, toffolis=float(t_res + ls.toffolis),
                    shots=float(shots + ls.n_prep), seconds_one_machine=float(sum(c.seconds(1) for c in campaigns)),
                    machines_7d=machines_for_campaigns(campaigns), best_practical=best_p.route,
                    toffolis_practical=float(t_res_p + ls.toffolis), residual_toffolis=float(t_res),
                    residual_toffolis_practical=float(t_res_p), line_search_ae=ls_ae, residual_norm_m=float(res.norm),
                    prep_toffolis=float(prep_k.toffolis), ls_flagged=int(flagged), campaigns=campaigns, flags=flags))
            else:
                campaigns = tuple(route_campaigns(best, p_r, model))
                flags = (_FLAG_NO_LAYOUT,) if any(c.per_run is None and 0 < c.n_runs < math.inf
                                                  for c in campaigns) else ()
                out[r].append(IterationCost(
                    k=k, residual=costs, best=best.route, line_search=None, toffolis=float(t_res + 0.0),
                    shots=float(shots + 0.0), seconds_one_machine=float(sum(c.seconds(1) for c in campaigns)),
                    machines_7d=machines_for_campaigns(campaigns), best_practical=best_p.route,
                    toffolis_practical=float(t_res_p + 0.0), residual_toffolis=float(t_res),
                    residual_toffolis_practical=float(t_res_p), line_search_ae=0.0, residual_norm_m=float(res.norm),
                    prep_toffolis=float(prep_k.toffolis), ls_flagged=0, campaigns=campaigns, flags=flags))
    return out


# ------------------------------------------------------------------------- F25: compiled overlaps
#: Slice choices and threshold of the F25 rule.
SLICE_RULE = (1, 2, 4, 8, 16)
ETA_SLICE = 0.05
_FOCK_MAX_NORB = 8


def _spin_square(active: ActiveSpace, ci: np.ndarray) -> float:
    from pyscf.fci import spin_op

    return float(spin_op.spin_square(np.asarray(ci), active.norb, active.nelec)[0])


def _sliced_step(active: ActiveSpace, state: np.ndarray, d, eps: float, s: int, irreps, params: FTParameters,
                 shape) -> tuple[np.ndarray, float]:
    """s ``circuits.compiled_step`` calls at eps/s along d from ``state``: (the normalized state, its norm before)."""
    psi = state
    for _ in range(s):
        psi = compiled_step(active, psi, d, eps / s if s > 1 else eps, irreps=irreps, rounded=True, params=params)
    psi = np.asarray(psi).reshape(shape)
    nrm = float(np.linalg.norm(psi))
    return psi / nrm, nrm


def compiled_overlaps(active: ActiveSpace, record: TrajectoryRecord, psi0: np.ndarray, *, k_max: int = 3,
                      slices=(1, 2, 4), params: FTParameters = FTParameters()) -> dict:
    """F25: the open-loop compiled iterates of ``record`` with every step in s first-order slices (module docstring).

    For each s of ``slices``: Psi~_0 = HF and Psi~_{k+1} = s successive ``circuits.compiled_step`` calls at
    eps_k/s along d_k from Psi~_k (= ``replay_elementwise(active, [d_0]*s + [d_1]*s + ..., [eps_0/s]*s + ...)``),
    k < k_max (at most the record's steps); N <= 8 (the Fock-space oracle). The F25 rule follows its own compiled
    trajectory: from its current state Psi~_k it takes the smallest s of :data:`SLICE_RULE` = (1, 2, 4, 8, 16) with
    ||Psi~^(s) - exp(eps_k D_k) Psi~_k|| <= eta (both normalized), and s = 16 with the flag "slice_cap" if none meets
    it, as ``ft.closedloop``'s compiled propagation; a trial whose state is that of a computed slice count (the rule has
    taken this s at every earlier step) is read from its rows. Returns {"k_max", "slices", "eta", "rows": [{"k", "s",
    "p" (exact record), "p_compiled" (normalized Psi~_k against psi0), "error" (||Psi~_k - Psi_k||, normalized states),
    "step_error" (||Psi~_k - exp(eps D) Psi~_{k-1}|| of the last step at the same s), "s2" (<S^2>), "norm_deficit",
    "c_step" (s C_step(d, eps/s)), "c_prep" (cumulative)}], "rule": [the same fields on the rule's trajectory, with
    "flags" ("slice_cap")]}.
    """
    if active.norb > _FOCK_MAX_NORB:
        raise ValueError(f"compiled overlaps need N <= {_FOCK_MAX_NORB} orbitals (Fock-space oracle), got "
                         f"{active.norb}")
    if isinstance(k_max, (bool, np.bool_)) or int(k_max) != k_max or k_max < 1:
        raise ValueError(f"k_max must be a positive integer, got {k_max!r}")
    ss = _sizes(slices, "slices")
    if ss.size == 0 or np.any(ss < 1) or len(set(ss.tolist())) != ss.size:
        raise ValueError(f"slices must be distinct positive integers, got {slices!r}")
    kk = min(int(k_max), record.n_steps)
    space = FCISpace.of(active)
    ham = Hamiltonian(active)
    psi0 = normalize(np.asarray(psi0).reshape(space.shape))
    hf = normalize(space.hf())

    def row(k: int, s: int, state: np.ndarray, nrm: float, step_error: float, c_step: int, c_prep: int) -> dict:
        return {"k": k, "s": s, "p": float(record.overlaps[k]),
                "p_compiled": float(abs(complex(inner(psi0, state))) ** 2),
                "error": float(np.linalg.norm(state - record.states[k])), "step_error": step_error,
                "s2": _spin_square(active, state), "norm_deficit": float(abs(1.0 - nrm)), "c_step": c_step,
                "c_prep": c_prep}

    def step_cost(k: int, s: int) -> tuple[int, int]:
        """(s C_step(d_k, eps_k/s), its increment of the cumulative cost)."""
        eps = float(record.eps[k])
        tof = elementwise_step_cost(record.directions[k], eps / s if s > 1 else eps, irreps=record.irreps,
                                    params=params).toffolis
        return int(s * tof), s * int(tof)

    rows: list[dict] = []
    known: dict[tuple[int, int], tuple[np.ndarray, float, float]] = {}    # (s, k + 1) -> (state, norm, step error)
    for s in (int(x) for x in ss):
        state, c_prep = hf, 0
        for k in range(kk):
            d, eps = record.directions[k], float(record.eps[k])
            exact = propagate(ham, d, state, eps)
            state, nrm = _sliced_step(active, state, d, eps, s, record.irreps, params, space.shape)
            err = float(np.linalg.norm(state - exact))
            known[(s, k + 1)] = (state, nrm, err)
            c, dc = step_cost(k, s)
            c_prep += dc
            rows.append(row(k + 1, s, state, nrm, err, c, int(c_prep)))
    rule: list[dict] = []
    state, c_prep, path = hf, 0, []
    for k in range(kk):
        d, eps = record.directions[k], float(record.eps[k])
        exact = None
        for s in SLICE_RULE:
            if (s, k + 1) in known and path == [s] * k:
                trial = known[(s, k + 1)]
            else:
                if exact is None:
                    exact = propagate(ham, d, state, eps)
                psi, nrm = _sliced_step(active, state, d, eps, s, record.irreps, params, space.shape)
                trial = (psi, nrm, float(np.linalg.norm(psi - exact)))
            if trial[2] <= ETA_SLICE:
                break
        flags = [] if trial[2] <= ETA_SLICE else ["slice_cap"]
        state, nrm, err = trial
        c, dc = step_cost(k, s)
        c_prep += dc
        path.append(s)
        rule.append({**row(k + 1, s, state, nrm, err, c, int(c_prep)), "flags": flags})
    return {"k_max": kk, "slices": [int(x) for x in ss], "eta": ETA_SLICE, "rows": rows, "rule": rule}


# ------------------------------------------------------------------------- the per-system scan
#: Records of a scan: the exact L-BFGS records (family/first step; ``ft.pipeline.FAMILIES``) and the R1 unit runs.
RECORD_KEYS = ("lbfgs/exact", "lbfgs/second_order", "lbfgs_so/exact", "lbfgs_so/second_order", "r1_so", "r1_sf")
#: Cached stages of :func:`scan_system` (at N <= 8 the first three are ``ft.closedloop.system_inputs``).
SCAN_STAGES = ("spectrum", "qpe", "residual_inputs", "records", "costs", "compiled")
#: Flags of :class:`ScalingRecord` beyond the spectrum's.
SCAN_FLAGS = ("roots_extended", "no_model_spectrum", "direct_route_only", "subset_variances", "no_compiled_overlaps")
_FAMILY_METRIC = {"lbfgs": "spin_free", "lbfgs_so": "spin_orbital"}
_R1_METRIC = {"r1_so": "spin_orbital", "r1_sf": "spin_free"}
_METRIC_KEYS = (("sf", "r1_sf", "lbfgs/second_order"), ("so", "r1_so", "lbfgs_so/second_order"))
_COMPILED_KEYS = ("r1_so", "lbfgs/second_order", "lbfgs_so/second_order")
_COMPILED_K = 3
_COMPILED_SLICES = (1, 2, 4)
_N_DENSE_MAX = 8                # N <= 8: dense inputs (all M3 routes, compiled overlaps); above: F24 (ii)
_SPECTRUM_ROOTS = 12
_SPECTRUM_ROOTS_LARGE = 2
_SPIN_TOLS = (1e-6, 1e-4)
_SUBSET_K = 100
_SUBSET_SEED = 11


@dataclass
class ScalingRecord:
    """Everything the scaling tables need of one system (:func:`scan_system`).

    Attributes:
        name, norb: the system and its active orbitals.
        spectrum: :class:`Spectrum` (12 singlet roots; 2 at N = 10 unless "roots_extended").
        qpe: the selected QPE (F9).
        records: :data:`RECORD_KEYS` -> ``TrajectoryRecord`` (with states; at N = 10 the second-order records and
            r1_so only).
        prep: record -> elementwise ``PrepCost``.
        costs: record -> r -> C_iter(k), k <= k_cost (``pipeline.IterationCost``; M3's F14 on the exact records, F30
            of the unit rule on the R1 records).
        cheapest: "sf"/"so" -> :func:`cheapest_iteration` at k = 1 (with its "record").
        compiled: record -> :func:`compiled_overlaps` (None at N > 8).
        timings: seconds per stage (computed, or loaded from the cache).
        flags: the spectrum's flags and :data:`SCAN_FLAGS`.
    """

    name: str
    norb: int
    spectrum: Spectrum
    qpe: QPECost
    records: dict
    prep: dict
    costs: dict
    cheapest: dict
    compiled: dict | None
    timings: dict
    flags: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.spectrum, Spectrum) or not isinstance(self.qpe, QPECost):
            raise ValueError("spectrum must be a Spectrum and qpe a QPECost")
        if self.spectrum.norb != self.norb:
            raise ValueError("the spectrum belongs to another active space")
        if set(self.records) - set(RECORD_KEYS) or set(self.prep) != set(self.records) \
                or set(self.costs) - set(self.records):
            raise ValueError(f"records, prep and costs must be keyed by {RECORD_KEYS}")
        if any(not isinstance(r, TrajectoryRecord) for r in self.records.values()):
            raise ValueError("records must be TrajectoryRecord objects")
        unknown = [f for f in self.flags if f not in SPECTRUM_FLAGS + SCAN_FLAGS]
        if unknown:
            raise ValueError(f"unknown flags {unknown}")
        self.flags = tuple(self.flags)

    @property
    def c_qpe(self) -> float:
        """C_QPE (Toffolis)."""
        return float(self.qpe.toffolis)


def _params_key(params: FTParameters) -> dict:
    return dataclasses.asdict(params)


class _Scan:
    """The stages of one system's scan, each through the cache (module docstring)."""

    def __init__(self, name: str, cache: "Cache | None", k_cost: int, n_steps: int, r_values, params: FTParameters):
        if cache is not None and not isinstance(cache, Cache):
            raise ValueError(f"cache must be an ft.scaling.Cache or None, got {type(cache).__name__}")
        if isinstance(k_cost, (bool, np.bool_)) or int(k_cost) != k_cost or k_cost < 0:
            raise ValueError(f"k_cost must be a non-negative integer, got {k_cost!r}")
        if isinstance(n_steps, (bool, np.bool_)) or int(n_steps) != n_steps or n_steps < 2:
            raise ValueError(f"n_steps must be an integer >= 2, got {n_steps!r}")
        if params.kind != "spin_free":
            raise ValueError(f"the scan measures spin-free elements; params.kind = {params.kind!r}")
        self.name, self.cache, self.params = name, cache, params
        self.k_cost, self.n_steps, self.r_values = int(k_cost), int(n_steps), _r_values(r_values)
        self.active = benchmark_system(name)
        self.irreps = orbital_irreps(self.active)
        self.norb = self.active.norb
        self.dense = self.norb <= _N_DENSE_MAX
        self.timings: dict[str, float] = {}
        self.flags: list[str] = []
        self._memo: dict[str, Any] = {}

    def _get(self, stage: str, compute: Callable[[], Any], key: Mapping):
        t = time.perf_counter()
        value = compute() if self.cache is None else self.cache.get(self.name, stage, compute, key=key)
        self.timings[stage] = self.timings.get(stage, 0.0) + time.perf_counter() - t
        return value

    def _once(self, label: str, compute: Callable[[], Any]):
        if label not in self._memo:
            self._memo[label] = compute()
        return self._memo[label]

    def inputs(self):
        """N <= 8: ``ft.closedloop.system_inputs`` (stages "spectrum", "qpe", "loop_inputs")."""
        from . import closedloop

        def compute():
            t = time.perf_counter()
            out = closedloop.system_inputs(self.name, cache=self.cache, params=self.params)
            self.timings["inputs"] = time.perf_counter() - t
            return out
        return self._once("inputs", compute)

    def spectrum(self) -> Spectrum:
        if self.dense:
            return self.inputs().spectrum

        def compute():
            def solve(n):
                return lambda: spectrum(self.active, self.irreps, n_roots=n, spin_tols=_SPIN_TOLS)
            sp = self._get("spectrum", solve(_SPECTRUM_ROOTS_LARGE),
                           {"n_roots": _SPECTRUM_ROOTS_LARGE, "spin_tols": list(_SPIN_TOLS)})
            if "e1_lower_bound" in sp.flags:         # root 1 outside the ground irrep: extend to 12 roots
                sp = self._get("spectrum", solve(_SPECTRUM_ROOTS), {"n_roots": _SPECTRUM_ROOTS,
                                                                    "spin_tols": list(_SPIN_TOLS)})
                self.flags.append("roots_extended")
            return sp
        return self._once("spectrum", compute)

    def qpe(self) -> QPECost:
        if self.dense:
            return self.inputs().qpe

        def compute():
            def select():
                from ..diagnostics.thc_floors import ExactReference
                encodings = scan_encodings(self.active, reference=ExactReference.build(self.active), params=self.params)
                return select_qpe(encodings, params=self.params)
            return self._get("qpe", select, {"params": _params_key(self.params)})
        return self._once("qpe", compute)

    def residual_inputs(self) -> dict:
        if self.dense:
            inp = self.inputs()
            return {"elements": inp.elements, "variances": inp.variances, "normalizations": inp.normalizations,
                    "energy_split": inp.energy_split, "variance_state": inp.variance_state}

        def compute():
            def subset():
                el, _ = ri.allowed_elements(self.active, "spin_free", self.irreps)
                hf = normalize(Hamiltonian(self.active).space.hf())
                return {"elements": el, "normalizations": None, "variance_state": "hf_subset",
                        "variances": ri.variance_totals_subset(self.active, hf, el, "spin_free", k_subset=_SUBSET_K,
                                                               seed=_SUBSET_SEED, params=self.params),
                        "energy_split": ri.energy_variances(self.active, hf)["best"]}
            return self._get("residual_inputs", subset, {"params": _params_key(self.params), "k_subset": _SUBSET_K,
                                                         "seed": _SUBSET_SEED})
        return self._once("residual_inputs", compute)

    def records(self) -> dict:
        from . import steprules as ftsr

        def compute():
            spec = self.spectrum()

            def run():
                recs, results = {}, {}
                firsts = ("exact", "second_order") if self.dense else ("second_order",)
                for fam, metric in _FAMILY_METRIC.items():
                    for first in firsts:
                        recs[f"{fam}/{first}"] = exact_trajectory(
                            self.active, optimizer="lbfgs", metric=metric, max_iter=self.n_steps, system=self.name,
                            irreps=self.irreps, n_roots=spec.n_roots, first_step=first, roots=spec.roots)
                for key in (("r1_so", "r1_sf") if self.dense else ("r1_so",)):
                    res = run_step_rule(self.active, StepRule("unit"), max_iter=self.n_steps, metric=_R1_METRIC[key],
                                        irreps=self.irreps, keep_states=True)
                    recs[key] = ftsr.to_trajectory_record(res, system=self.name, active=self.active,
                                                          irreps=self.irreps, spectrum=spec)
                    results[key] = dataclasses.replace(res, states=None)
                return {"records": recs, "results": results}
            return self._get("records", run, {"n_steps": self.n_steps, "unit_rule": dataclasses.asdict(StepRule()),
                                              "n_roots": spec.n_roots})
        return self._once("records", compute)

    def prep(self) -> dict:
        return self._once("prep", lambda: {key: prep_cost(rec.directions, rec.eps, method="elementwise",
                                                          irreps=self.irreps, params=self.params)
                                           for key, rec in self.records()["records"].items()})

    def costs(self) -> dict:
        def compute():
            stage = self.records()
            recs, results = stage["records"], stage["results"]
            prep, walk, inp = self.prep(), self.qpe().encoding, self.residual_inputs()
            routes = "all" if self.dense else "direct"

            def run():
                readout = readout_rotation_toffolis(self.norb, params=self.params)
                costs = {}
                for key, rec in recs.items():
                    k_max = min(self.k_cost, rec.n_steps - 1)
                    if k_max < 0:
                        continue
                    costs[key] = _iteration_costs_by_r(
                        self.active, rec, prep[key], walk, inp, r_values=self.r_values, k_max=k_max,
                        line_search=not key.startswith("r1_"), energy_split=inp["energy_split"],
                        readout_toffolis=readout, routes=routes, result=results.get(key), params=self.params)
                cheapest = {}
                for label, r1_key, exact_key in _METRIC_KEYS:
                    key = r1_key if r1_key in recs else exact_key
                    if recs[key].n_steps < 1:
                        continue
                    c = cheapest_iteration(self.active, recs[key], prep[key], walk, inp, k=1, r_values=self.r_values,
                                           routes=routes, params=self.params)
                    c["record"] = key
                    c["variance_method"] = inp["variances"].get("method")
                    c["variance_state"] = inp["variance_state"]
                    cheapest[label] = c
                return {"costs": costs, "cheapest": cheapest}
            return self._get("costs", run, {"params": _params_key(self.params), "k_cost": self.k_cost,
                                            "r_values": list(self.r_values), "n_steps": self.n_steps})
        return self._once("costs", compute)

    def compiled(self) -> dict | None:
        if self.norb > _FOCK_MAX_NORB:
            return None

        def compute():
            recs, spec = self.records()["records"], self.spectrum()

            def run():
                return {key: compiled_overlaps(self.active, recs[key], spec.psi0, k_max=_COMPILED_K,
                                               slices=_COMPILED_SLICES, params=self.params)
                        for key in _COMPILED_KEYS if key in recs and recs[key].n_steps >= 1}
            return self._get("compiled", run, {"params": _params_key(self.params), "k_max": _COMPILED_K,
                                               "slices": list(_COMPILED_SLICES), "n_steps": self.n_steps})
        return self._once("compiled", compute)

    def run_stage(self, stage: str) -> None:
        if stage not in SCAN_STAGES:
            raise ValueError(f"stage must be one of {SCAN_STAGES}, got {stage!r}")
        getattr(self, stage)()

    def record(self) -> ScalingRecord:
        spec, qpe = self.spectrum(), self.qpe()
        stage = self.costs()
        compiled = self.compiled()
        flags = list(spec.flags) + self.flags
        if not self.dense:
            flags += ["no_model_spectrum", "direct_route_only", "subset_variances"]
        if compiled is None:
            flags.append("no_compiled_overlaps")
        return ScalingRecord(name=self.name, norb=self.norb, spectrum=spec, qpe=qpe, records=self.records()["records"],
                             prep=self.prep(), costs=stage["costs"], cheapest=stage["cheapest"], compiled=compiled,
                             timings=dict(self.timings), flags=tuple(dict.fromkeys(flags)))


def scan_system(name: str, *, cache: Cache | None = None, k_cost: int = 3, n_steps: int = 6,
                r_values=(0.3, 1.0, 3.0), params: FTParameters = FTParameters()) -> ScalingRecord:
    """The per-system protocol of the scaling study (module docstring): spectrum, QPE, records, residual inputs,
    costs for k <= ``k_cost`` at every r, the cheapest iteration and the compiled overlaps; each stage cached."""
    return _Scan(name, cache, k_cost, n_steps, r_values, params).record()


def scan_stage(name: str, stage: str, *, cache: Cache, k_cost: int = 3, n_steps: int = 6, r_values=(0.3, 1.0, 3.0),
               params: FTParameters = FTParameters()) -> dict:
    """Compute one stage of :data:`SCAN_STAGES` (and, through ``cache``, the stages it needs) and store it in the
    cache, so that a later :func:`scan_system` with the same arguments only loads; returns the seconds per stage."""
    if not isinstance(cache, Cache):
        raise ValueError("scan_stage needs a Cache")
    scan = _Scan(name, cache, k_cost, n_steps, r_values, params)
    scan.run_stage(stage)
    return dict(scan.timings)


def _k_cert(record: TrajectoryRecord, de_qpe: float) -> int | None:
    hit = [k for k, e in enumerate(record.energies) if rule_e_certifiable(float(e), record.e1, de_qpe)]
    return hit[0] if hit else None


def _cost_rows(costs: Sequence[IterationCost], c_qpe: float) -> list[dict]:
    return [{"k": c.k, "c_iter": float(c.toffolis), "c_iter_over_qpe": float(c.toffolis) / c_qpe, "best": c.best,
             "residual_toffolis": float(c.residual_toffolis),
             "line_search_toffolis": None if c.line_search is None else float(c.line_search.toffolis),
             "line_search_ae": float(c.line_search_ae), "shots": float(c.shots), "machines_7d": c.machines_7d,
             "residual_norm_m": float(c.residual_norm_m), "prep_toffolis": float(c.prep_toffolis),
             "flags": list(c.flags)} for c in costs]


def record_summary(rec: ScalingRecord) -> dict:
    """The strict-JSON layout of ``scaling_<system>.json`` (:func:`to_json` applied): the spectrum, the QPE (kind,
    lambda, C_W, n_iter, C_QPE, Q), per record the energies, errors, overlaps, eps, prep cumulative, k*, rule-E
    k_cert and the costs per r (no states), the cheapest iteration, the compiled overlaps, timings and flags."""
    c_qpe = rec.c_qpe
    enc = rec.qpe.encoding
    out = {"system": rec.name, "norb": rec.norb, "flags": list(rec.flags), "spectrum": rec.spectrum,
           "qpe": {"kind": enc.kind, "params": enc.params, "lam": float(enc.lam), "c_w": int(enc.toffoli_per_step),
                   "n_iter": int(rec.qpe.iterations), "c_qpe": c_qpe, "logical_qubits": int(rec.qpe.logical_qubits),
                   "de_qpe": float(rec.qpe.de_qpe), "de_fact": float(enc.de_fact), "infidelity": float(enc.infidelity)},
           "records": {}, "cheapest": rec.cheapest, "compiled": rec.compiled, "timings": rec.timings}
    for key, r in rec.records.items():
        pc = rec.prep[key]
        out["records"][key] = {
            "metric": r.metric, "optimizer": r.optimizer, "first_step": r.first_step, "n_steps": r.n_steps,
            "converged": bool(r.converged), "flags": list(r.flags), "energies": r.energies, "e_err": r.errors,
            "overlaps": r.overlaps, "eps": r.eps, "variances": r.variances, "residual_norms": r.residual_norms,
            "prep_cumulative": pc.cumulative, "prep_qubits": pc.logical_qubits,
            "k_star": first_iterate_below(r, FTParameters().target), "k_cert": _k_cert(r, rec.qpe.de_qpe),
            "costs": {f"{rv:g}": _cost_rows(cs, c_qpe) for rv, cs in rec.costs.get(key, {}).items()}}
    return to_json(out)


# ------------------------------------------------------------------------- report: tables U1-U4
_CHAIN = re.compile(r"h(\d+)_r([0-9]+(?:\.[0-9]+)?)")
#: Overlap series of U2: (label, k, record; None = the spectrum's p_HF).
_SERIES = (("hf", 0, None), ("classical_sf", 1, "lbfgs/second_order"), ("classical_so", 1, "lbfgs_so/second_order"),
           ("r1_sf", 2, "r1_sf"), ("r1_so", 2, "r1_so"))
_FIT_WINDOW = (6, 8, 10)
_P_TARGET = 1e-3
_N_COST_MIN = 6                 # F23: exponents on N >= 6 (the selected kind switches between N = 4 and 6)
_U4_K = (0, 1, 2)


def _chain(name: str) -> tuple[int, str] | None:
    m = _CHAIN.fullmatch(name)
    return (int(m.group(1)), m.group(2)) if m else None


def _series_p(rec: ScalingRecord, k: int, key: str | None) -> float | None:
    if key is None:
        return float(rec.spectrum.p_hf)
    r = rec.records.get(key)
    if r is None or r.n_steps < k:
        return None
    return float(r.overlaps[k])


def _fit(ns, ps, R, label, k) -> RateFit:
    window = _FIT_WINDOW if int(np.isin(ns, _FIT_WINDOW).sum()) >= 2 else tuple(ns)
    return fit_log_overlap(ns, ps, window=window, R=R, record=label, k=k)


def _u1(records: Sequence[ScalingRecord]) -> list[dict]:
    rows = []
    for rec in records:
        sp, q = rec.spectrum, rec.qpe
        rows.append({"system": rec.name, "N": rec.norb, "dim": sp.dim, "p_hf": sp.p_hf, "e_hf_minus_e0": sp.e_hf - sp.e0,
                     "gap": sp.gap, "gap_any": sp.gap_any, "kind": q.encoding.kind, "lam": float(q.encoding.lam),
                     "c_w": int(q.encoding.toffoli_per_step), "c_qpe": rec.c_qpe, "q_qpe": int(q.logical_qubits),
                     "k_cert": {key: _k_cert(rec.records[key], q.de_qpe) for key in RECORD_KEYS[:4]
                                if key in rec.records},
                     "flags": list(rec.flags)})
    return rows


def _gap_fit(recs: Sequence[ScalingRecord]) -> GapFit | None:
    """F21's gap fit of one bond length (sizes with a same-irrep E_1); None below two sizes."""
    pts = [(_chain(r.name)[0], r.spectrum.gap) for r in recs
           if r.spectrum.e1 is not None and "e1_lower_bound" not in r.spectrum.flags]
    if len(pts) < 2:
        return None
    return fit_gap([n for n, _ in pts], [x for _, x in pts])


#: States of U2's one-run B_L side column (F21): HF and the classical Psi_1 of both metrics.
_ONE_RUN_STATES = ("hf", "classical_sf", "classical_so")


def _p_at(fit: RateFit, n: float, rate: str) -> float | None:
    return extrapolate_overlap(fit, n, rate=rate) if math.isfinite(n) else None


def _bl_windows(g: GapFit, fits: Mapping, rlabel: str, margins: Mapping[str, float], *, de_qpe: float,
                sin_theta0: float, e1_margin: float) -> dict:
    """F21's B_L windows of one bond length beyond the margin m_BL: the one-run window with D_L(p(n)) per state, gap
    model and rate (:func:`certification_window_overlap`), and the two-run window n_cert(m_BL) against F40's floor:
    p at both ends, the states below the floor there and the window capped where p falls to the floor."""
    floor = two_run_overlap_floor()
    lo, hi = certification_window(g, margins["BL"])
    one_run, edges, sound, below = {}, {}, {}, []
    for label in _ONE_RUN_STATES:
        fit = fits.get((rlabel, label))
        if fit is None:
            continue
        one_run[label] = {model: [certification_window_overlap(g, fit, de_qpe=de_qpe, sin_theta0=sin_theta0,
                                                               e1_margin=e1_margin, rate=rate, model=model)
                                  for rate in RATES] for model in GAP_MODELS}
        edges[label] = {"inverse": [_p_at(fit, lo, rate) for rate in RATES],
                        "power": [_p_at(fit, hi, rate) for rate in RATES]}
        n_floor = [size_at_overlap(fit, floor, rate=rate) for rate in RATES]
        sound[label] = {"inverse": [min(lo, n) for n in n_floor], "power": [min(hi, n) for n in n_floor]}
        if any(p is not None and p < floor for ps in edges[label].values() for p in ps):
            below.append(label)
    return {"one_run": one_run, "two_run": {"p_floor": floor, "n_cert": [lo, hi], "p_at_n_cert": edges,
                                            "below_floor": below, "n_cert_sound": sound}}


def _u2(chains: Mapping[str, list[ScalingRecord]], margins: Mapping[str, float], *, de_qpe: float,
        sin_theta0: float, e1_margin: float) -> dict:
    overlap, gaps, fits = [], [], {}
    for rlabel, recs in chains.items():
        R = float(rlabel)
        for label, k, key in _SERIES:
            pts = [(rec_n, p) for rec_n, p in ((_chain(r.name)[0], _series_p(r, k, key)) for r in recs) if p]
            if len(pts) < 2:
                continue
            ns, ps = [n for n, _ in pts], [p for _, p in pts]
            fit = _fit(ns, ps, R, label, k)
            fits[(rlabel, label)] = fit
            holdouts = []
            for i, n_h in enumerate(ns):
                if i >= 3:
                    sub = _fit(ns[:i], ps[:i], R, label, k)
                    pred = {rate: extrapolate_overlap(sub, n_h, rate=rate) for rate in RATES}
                    holdouts.append({"n": n_h, "computed": ps[i], "from": ns[:i], **{f"p_{r}": v for r, v in pred.items()},
                                     **{f"error_{r}": v / ps[i] - 1.0 for r, v in pred.items()}})
            overlap.append({"R": R, "record": label, "k": k, "ns": ns, "p": ps, "successive": list(fit.successive),
                            "a": fit.a, "alpha": fit.alpha, "drift": fit.drift, "alpha_aitken": fit.alpha_aitken,
                            "flags": list(fit.flags), "holdouts": holdouts,
                            "n_at_p": [size_at_overlap(fit, _P_TARGET, rate=rate) for rate in RATES]})
        for row in overlap:
            if row["R"] == R and row["record"] == "hf":
                row["p1_at_n"] = {}
                for m in ("sf", "so"):
                    f1 = fits.get((rlabel, f"classical_{m}"))
                    if f1 is not None:
                        row["p1_at_n"][m] = [extrapolate_overlap(f1, n, rate=rate) if math.isfinite(n) else None
                                             for n, rate in zip(row["n_at_p"], RATES)]
        g = _gap_fit(recs)
        if g is not None:
            fits[(rlabel, "gap")] = g
            row = {"R": R, "ns": list(g.ns), "gaps": list(g.gaps), "n_gap": [n * x for n, x in zip(g.ns, g.gaps)],
                   "A": g.A, "gamma": g.gamma, "A_inv": g.A_inv,
                   "n_cert": {label: list(certification_window(g, m)) for label, m in margins.items()}}
            if "BL" in margins:
                bl = _bl_windows(g, fits, rlabel, margins, de_qpe=de_qpe, sin_theta0=sin_theta0, e1_margin=e1_margin)
                row["bl_one_run"], row["bl_two_run"] = bl["one_run"], bl["two_run"]
            gaps.append(row)
    return {"overlap": overlap, "gap": gaps, "_fits": fits}


def _metric_record(rec: ScalingRecord, label: str) -> str | None:
    c = rec.cheapest.get(label)
    return None if c is None else c["record"]


def _u3(records: Sequence[ScalingRecord], chains: Mapping[str, list[ScalingRecord]]) -> dict:
    rows = []
    for rec in records:
        enc = rec.qpe.encoding
        row = {"system": rec.name, "N": rec.norb, "c_qpe": rec.c_qpe, "lam": float(enc.lam),
               "c_w": int(enc.toffoli_per_step), "kind": enc.kind}
        any_c = next(iter(rec.cheapest.values()), None)
        if any_c is not None:
            row.update(K=any_c["n_elements"], variance_m=any_c["variance_m"], variance_method=any_c["variance_method"],
                       variance_state=any_c["variance_state"], w_aux_m=any_c["w_aux_m"])
        for label in ("sf", "so"):
            c = rec.cheapest.get(label)
            if c is None:
                continue
            pc = rec.prep[c["record"]]
            row[label] = {"record": c["record"], "c_psi1": float(pc.cumulative[1]),
                          "c_psi2": float(pc.cumulative[2]) if len(pc.cumulative) > 2 else None,
                          "c_psi2_record": c["record"] if len(pc.cumulative) > 2 else None,
                          "norm_a1": c["norm_a_m"],
                          "t_res1": {r: {"over_qpe": v["over_qpe"], "route": v["best"]} for r, v in c["by_r"].items()}}
        rows.append(row)
    by_name = {row["system"]: row for row in rows}
    exponents = []
    for rlabel, recs in chains.items():
        sized = sorted((_chain(r.name)[0], by_name[r.name]) for r in recs if r.norb >= _N_COST_MIN)
        quantities: dict[str, list[tuple[int, float, str | None]]] = {}
        for n, row in sized:
            vals = {key: (row.get(key), None) for key in ("c_qpe", "lam", "c_w", "K", "variance_m", "w_aux_m")}
            for label in ("sf", "so"):
                m = row.get(label)
                if m is None:
                    continue
                vals[f"c_psi1_{label}"] = (m["c_psi1"], m["record"])
                vals[f"c_psi2_{label}"] = (m["c_psi2"], m["c_psi2_record"])
                vals[f"norm_a1_{label}"] = (m["norm_a1"], m["record"])
                for r, v in m["t_res1"].items():
                    vals[f"t_res1_over_qpe_{label}_r{r}"] = (v["over_qpe"], m["record"])
            for key, (v, src) in vals.items():
                if v is not None and math.isfinite(v) and v > 0.0:
                    quantities.setdefault(key, []).append((n, float(v), src))
        for key, pts in quantities.items():
            if len(pts) >= 2:
                f = fit_power_law([n for n, _, _ in pts], [v for _, v, _ in pts])
                exponents.append({"R": float(rlabel), "quantity": key, "N": list(f.x), "y": list(f.y),
                                  "records": [src for _, _, src in pts],
                                  "successive": list(f.successive), "beta_last": f.beta_last, "beta_mean": f.beta_mean})
    return {"rows": rows, "exponents": exponents}


def _u4(records: Sequence[ScalingRecord], chains: Mapping[str, list[ScalingRecord]], u2: dict,
        margins: Mapping[str, float]) -> dict:
    rows = []
    for rec in records:
        c_qpe = rec.c_qpe
        for label, r1_key, exact_key in _METRIC_KEYS:
            if r1_key not in rec.records or r1_key not in rec.costs:
                continue
            tr, pc = rec.records[r1_key], rec.prep[r1_key]
            er, epc = rec.records.get(exact_key), rec.prep.get(exact_key)
            for r, costs in rec.costs[r1_key].items():
                ecosts = rec.costs.get(exact_key, {}).get(r)
                for k in _U4_K:
                    if k + 1 > tr.n_steps or k >= len(costs):
                        continue
                    p, p1 = float(tr.overlaps[k]), float(tr.overlaps[k + 1])
                    ck, ck1 = pc.cumulative[k] / c_qpe, pc.cumulative[k + 1] / c_qpe
                    c_it = float(costs[k].toffolis) / c_qpe
                    row = {"system": rec.name, "N": rec.norb, "metric": label, "r": float(r), "k": k, "p": p, "p_next": p1,
                           "gain": p1 / p, "c_k": ck, "c_k1": ck1, "c_iter": c_it,
                           "mu": marginal_saving(p, p1, ck, ck1), "rho": cost_ratio(c_it, p, p1, ck, ck1),
                           "p_star": break_even_overlap(c_it, p1 / p, ck, ck1), "rho_ls": None}
                    row["pays"] = bool(row["rho"] < 1.0)
                    if er is not None and ecosts is not None and k + 1 <= er.n_steps and k < len(ecosts):
                        res = float(ecosts[k].residual_toffolis) / c_qpe
                        row["rho_ls"] = cost_ratio(res, float(er.overlaps[k]), float(er.overlaps[k + 1]),
                                                   epc.cumulative[k] / c_qpe, epc.cumulative[k + 1] / c_qpe)
                    rows.append(row)
    gaps = {rlabel: u2["_fits"].get((rlabel, "gap")) for rlabel in chains}
    return {"rows": rows, "n_star": _n_star(chains, gaps, margins)}


def _first_cost(rec: ScalingRecord, label: str, r1_key: str, r: float) -> tuple[float, str] | None:
    """T_res(1)/C_QPE of the first measured iteration (at the classical Psi_1) at r: the R1 record's k = 1 cost, else
    the cheapest iteration of the metric (the same residual at the same state); None without either."""
    costs = rec.costs.get(r1_key, {}).get(r) if r1_key in rec.records else None
    if costs is not None and len(costs) > 1:
        c, source = float(costs[1].toffolis) / rec.c_qpe, r1_key
    else:
        ch = rec.cheapest.get(label)
        v = None if ch is None else ch["by_r"].get(f"{float(r):g}")
        if v is None or ch.get("k") != 1 or ch.get("first_step") != "second_order":
            return None
        c, source = float(v["over_qpe"]), f"cheapest:{ch['record']}"
    return (c, source) if math.isfinite(c) and c > 0.0 else None


def _n_star(chains: Mapping[str, list[ScalingRecord]], gaps: Mapping[str, GapFit | None],
            margins: Mapping[str, float], *, saving: Callable | None = None, scheme: str = "rep",
            qpe_scaled: bool | None = None) -> list[dict]:
    """F45 brackets of the first measured iteration of the R1 loop (classical Psi_1 -> unit Psi_2) per bond length,
    metric and r (U4).

    alpha_1 is fitted on the classical-step series (overlaps[1] of the second-order record of the metric, every size:
    the R1 loop's Psi_1), alpha_2 on the R1 record's overlaps[2]; the cost points (N, T_res(1)/C_QPE) on N >= 6 come
    from the R1 record's k = 1 cost where it exists and from the cheapest iteration (the same residual at the same
    Psi_1) elsewhere, and the bracket is anchored at the last of them (n0, c0); the preparation terms c_1 = C(Psi_1)/
    C_QPE (classical-step record) and c_2 = C(Psi_2)/C_QPE (R1 record) are power laws on N >= 6 anchored at their last
    size (beta_last). Each overlap fit is extrapolated from its own last size.
    """
    n_star = []
    for rlabel, recs in chains.items():
        sized = sorted(((_chain(r.name)[0], r) for r in recs), key=lambda t: t[0])
        gap = gaps.get(rlabel)
        for label, r1_key, exact_key in _METRIC_KEYS:
            s1 = [(n, r) for n, r in sized if exact_key in r.records and r.records[exact_key].n_steps >= 1]
            s2 = [(n, r) for n, r in sized if r1_key in r.records and r1_key in r.costs
                  and r.records[r1_key].n_steps >= 2]
            if len(s1) < 2 or len(s2) < 2:
                continue
            ns1, ns2 = [n for n, _ in s1], [n for n, _ in s2]
            f1 = _fit(ns1, [float(r.records[exact_key].overlaps[1]) for _, r in s1], float(rlabel),
                      f"classical_{label}", 1)
            f2 = _fit(ns2, [float(r.records[r1_key].overlaps[2]) for _, r in s2], float(rlabel), r1_key, 2)
            prep_pts = {"c_k": (exact_key, [(n, float(r.prep[exact_key].cumulative[1]) / r.c_qpe) for n, r in s1]),
                        "c_k1": (r1_key, [(n, float(r.prep[r1_key].cumulative[2]) / r.c_qpe) for n, r in s2])}
            prep_fits, prep_info = {}, {}
            for key, (src, pts) in prep_pts.items():
                pts = [(n, c) for n, c in pts if n >= _N_COST_MIN and math.isfinite(c) and c > 0.0]
                if len(pts) >= 2:
                    pf = fit_power_law([n for n, _ in pts], [c for _, c in pts])
                    prep_fits[key] = pf
                    prep_info[key] = {"record": src, "ns": [int(n) for n, _ in pts], "c": [c for _, c in pts],
                                      "successive": list(pf.successive)}
            prep = (prep_fits["c_k"], prep_fits["c_k1"]) if len(prep_fits) == 2 else None
            for r in s2[-1][1].costs[r1_key]:
                pts = []
                for n, rec in sized:
                    if n >= _N_COST_MIN:
                        pt = _first_cost(rec, label, r1_key, r)
                        if pt is not None:
                            pts.append((n, *pt))
                if len(pts) < 2:
                    continue
                cost_fit = fit_power_law([n for n, _, _ in pts], [c for _, c, _ in pts])
                n0, c0 = pts[-1][0], pts[-1][1]
                br = break_even_bracket(f1, f2, cost_fit, c0, n0, gap=gap, margins=None if gap is None else margins,
                                        saving=saving, prep=prep, qpe_scaled=qpe_scaled)
                n_star.append({"R": float(rlabel), "metric": label, "r": float(r), "k": 1, "scheme": scheme,
                               "alpha_k": [f1.alpha_last, f1.alpha_aitken],
                               "alpha_k1": [f2.alpha_last, f2.alpha_aitken],
                               "cost_successive": list(cost_fit.successive),
                               "inputs": {"alpha_k": {"record": exact_key, "ns": ns1, "p": list(f1.p)},
                                          "alpha_k1": {"record": r1_key, "ns": ns2, "p": list(f2.p)},
                                          "cost": {"ns": [int(n) for n, _, _ in pts], "c_iter": [c for _, c, _ in pts],
                                                   "source": [s for _, _, s in pts]},
                                          "c_k": prep_info.get("c_k"), "c_k1": prep_info.get("c_k1")},
                               **br})
    return n_star


def _chains_of(records: Sequence[ScalingRecord]) -> dict[str, list[ScalingRecord]]:
    """The H chains of ``records`` by bond length (as a string), in increasing R and n."""
    chains: dict[str, list[ScalingRecord]] = {}
    for rec in records:
        c = _chain(rec.name)
        if c is not None:
            chains.setdefault(c[1], []).append(rec)
    return {k: sorted(v, key=lambda r: _chain(r.name)[0]) for k, v in sorted(chains.items(), key=lambda kv: float(kv[0]))}


def break_even_sizes(records: Sequence[ScalingRecord], *, saving: Callable | None = None, scheme: str = "rep",
                     sin_theta0: float = 1.0, e1_margin: float = 1e-3, qpe_scaled: bool | None = None) -> list[dict]:
    """F45 brackets of U4 under a QPE-side scheme: :func:`scaling_report`'s "n_star" entries with ``saving``.

    Per bond length of the H chains in ``records``, metric and r: the first measured iteration of the R1 loop
    (classical Psi_1 -> unit Psi_2), its overlap fits (F22), cost fit (F23), preparation terms, gap fit and the
    margins of ``acceptance_margins(dE, sin_theta0, e1_margin)`` (F21), bracketed by :func:`break_even_bracket` with
    mu_k(n) from ``saving``(p_k(n), p_{k+1}(n), c_k(n), c_{k+1}(n)) (``ft.amplify.scheme_saving`` or
    ``ft.amplify.best_realizable_saving``; None: repetition, the report's own column; ``qpe_scaled`` as in
    :func:`break_even_size`) and labelled ``scheme``. The same saving holds at every size: a scheme whose availability
    depends on n (the E1lb-referenced forms need Delta_1(n) >= w_L + (E_1 - E1lb)) is the caller's to restrict.
    Strict JSON (:func:`to_json`).
    """
    records = list(records)
    if not records or not all(isinstance(r, ScalingRecord) for r in records):
        raise ValueError("records must be a non-empty sequence of ScalingRecord")
    de = {float(r.qpe.de_qpe) for r in records}
    if len(de) != 1:
        raise ValueError(f"the records use different QPE errors {sorted(de)}")
    margins = acceptance_margins(de.pop(), sin_theta0, e1_margin)
    chains = _chains_of(records)
    return to_json(_n_star(chains, {rlabel: _gap_fit(recs) for rlabel, recs in chains.items()}, margins,
                           saving=saving, scheme=str(scheme), qpe_scaled=qpe_scaled))


def scaling_report(records: Sequence[ScalingRecord], *, sin_theta0: float = 1.0, e1_margin: float = 1e-3) -> dict:
    """Tables U1-U4 (repetition columns), the fits, windows and n* brackets of ``records`` (module docstring).

    ``sin_theta0`` (the walk phase of E_0 at M = 2 n_iter; 1 is its upper bound, the largest m_BL) and
    ``e1_margin`` (E1 - E1lb; 1 mHa beyond the model spectrum, F41) set the F21 margins of the windows. Returns
    {"systems", "settings", "U1", "U2" {"overlap", "gap"}, "U3" {"rows", "exponents"}, "U4" {"rows",
    "n_star"}} (JSON-able through :func:`to_json`).
    """
    records = list(records)
    if not records or not all(isinstance(r, ScalingRecord) for r in records):
        raise ValueError("records must be a non-empty sequence of ScalingRecord")
    de = {float(r.qpe.de_qpe) for r in records}
    if len(de) != 1:
        raise ValueError(f"the records use different QPE errors {sorted(de)}")
    de_qpe = de.pop()
    margins = acceptance_margins(de_qpe, sin_theta0, e1_margin)
    chains = _chains_of(records)
    u2 = _u2(chains, margins, de_qpe=de_qpe, sin_theta0=sin_theta0, e1_margin=e1_margin)
    out = {"systems": [r.name for r in records],
           "settings": {"de_qpe": de_qpe, "sin_theta0": float(sin_theta0), "e1_margin": float(e1_margin), "d_l": 5,
                        "margins": margins, "eps_id": EPS_ID, "p_floor_two_run": two_run_overlap_floor(),
                        "p_target": _P_TARGET, "fit_window": list(_FIT_WINDOW), "n_max": _N_MAX,
                        "p_floor_scan": P_FLOOR, "n_cost_min": _N_COST_MIN, "beta_fixed": _BETA_FIXED,
                        "crossovers": [None if c is None else list(c) for c in _CROSSOVERS]},
           "U1": _u1(records), "U2": {"overlap": u2["overlap"], "gap": u2["gap"]}, "U3": _u3(records, chains),
           "U4": _u4(records, chains, u2, margins)}
    return to_json(out)


def _f(x, spec: str = ".4g") -> str:
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    x = float(x)
    if math.isnan(x):
        return "-"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return format(x, spec)


def _rng(v, spec: str = ".4g") -> str:
    if v is None:
        return "none"
    lo, hi = v
    return _f(lo, spec) if lo == hi else f"{_f(lo, spec)}-{_f(hi, spec)}"


def _md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]


def _num(x) -> float:
    """A number of the strict JSON (:func:`to_json`): "inf"/"-inf" -> +-inf, None -> nan."""
    if x is None:
        return math.nan
    if isinstance(x, str):
        return {"inf": math.inf, "-inf": -math.inf}[x]
    return float(x)


def _cases(s: Mapping | None, key: str = "n_star", none_key: str = "n_none") -> str:
    """A bracket as 'lo-hi (k/N none)'."""
    if s is None:
        return "-"
    return f"{_rng(s[key], '.0f')} ({s[none_key]}/{s['n_cases']} none)"


def _pair(v, spec: str = ".4f") -> str:
    """[last, Aitken] as 'x / y'."""
    return " / ".join(_f(x, spec) for x in v) if v else "-"


_MODEL_LABEL = {"power": "power", "inverse": "A'/n"}
_MODEL_SCOPE = ("Beyond the computed sizes every value is a model statement, and its range a model bracket "
                "(F21-F23, F45: validated over Delta n = 2 by the holdouts, unquantified model risk beyond; never a "
                "confidence interval). The model covers H chains in canonical orbitals with elementwise compilation "
                "and the full allowed residual; localized or truncated residuals, other topologies and larger bases "
                "are outside it.")


def markdown_tables(report: Mapping) -> str:
    """U1-U4 of :func:`scaling_report` as markdown (energies in mHa, costs in C_QPE units)."""
    lines = ["## U1. Systems and spectra", ""]
    lines += _md_table(
        ["system", "N", "dets", "p_HF", "E_HF - E0 (mHa)", "Delta_1 same / any (mHa)", "QPE (kind, lambda Ha, C_W)",
         "C_QPE", "Q_QPE", "rule-E k_cert (sf ex / sf so / so ex / so so)", "flags"],
        [[r["system"], str(r["N"]), str(r["dim"]), _f(r["p_hf"]), _f(1e3 * r["e_hf_minus_e0"], ".1f"),
          f"{_f(None if r['gap'] is None else 1e3 * r['gap'], '.2f')} / "
          f"{_f(None if r['gap_any'] is None else 1e3 * r['gap_any'], '.2f')}",
          f"{r['kind']}, {_f(r['lam'], '.3f')}, {r['c_w']}", _f(r["c_qpe"], ".4g"), str(r["q_qpe"]),
          " / ".join(_f(r["k_cert"].get(k)) for k in RECORD_KEYS[:4]), ", ".join(r["flags"]) or "-"]
         for r in report["U1"]])
    lines += ["", "## U2. Overlap scaling (F22) and the hand-off window (F21)", "",
              "Rates per atom; fit ln p = a - alpha n on n in (6, 8, 10); bracket [alpha_last, alpha^A]; "
              "n(p = 1e-3) at both rates (last - Aitken; p_1 there: classical step sf / so, at the same rate). "
              + _MODEL_SCOPE, ""]
    rows = []
    for r in report["U2"]["overlap"]:
        hold = "; ".join(f"n={h['n']}: {100 * h['error_last']:+.2f}% / {100 * h['error_aitken']:+.2f}%"
                         for h in r["holdouts"]) or "-"
        p1 = r.get("p1_at_n")
        p1s = "-" if not p1 else "; ".join(f"{m} {_rng(v, '.3g')}" for m, v in p1.items())
        rows.append([_f(r["R"], "g"), r["record"], str(r["k"]), " / ".join(str(n) for n in r["ns"]),
                     " / ".join(_f(p) for p in r["p"]),
                     ", ".join(_f(a, ".4f") for a in r["successive"]), f"{_f(r['a'], '+.3f')}, {_f(r['alpha'], '.4f')}",
                     _f(r["alpha_aitken"], ".4f"), hold, _rng(r["n_at_p"], ".1f"), p1s])
    lines += _md_table(["R", "record", "k", "n", "p", "successive", "a, alpha", "alpha^A",
                        "holdout error (last / Aitken)", "n(p = 1e-3)", "p_1 there"], rows)
    m = report["settings"]["margins"]
    lines += ["", f"Margins (mHa): m_E = {1e3 * m['E']:.2f}, m_B = {1e3 * m['B']:.2f}, m_BL = {1e3 * m['BL']:.2f} "
                  f"(sin theta_0 = {report['settings']['sin_theta0']:g}, E1 - E1lb = "
                  f"{1e3 * report['settings']['e1_margin']:g} mHa; m_BL with F40's verification run, D_L = "
                  f"{report['settings']['d_l']}). n_cert(m) = [A'/m, (A/m)^(1/gamma)] is a model bracket.", ""]
    lines += _md_table(["R", "Delta_1 at n (mHa)", "n Delta_1 (mHa)", "A (mHa), gamma", "A' (mHa)",
                        "n_cert(m_E)", "n_cert(m_B)", "n_cert(m_BL)"],
                       [[_f(g["R"], "g"), " / ".join(_f(1e3 * x, ".2f") for x in g["gaps"]),
                         " / ".join(_f(1e3 * x, ".0f") for x in g["n_gap"]), f"{_f(1e3 * g['A'], '.0f')}, "
                         f"{_f(g['gamma'], '.3f')}", _f(1e3 * g["A_inv"], ".0f"),
                         *[_rng(g["n_cert"][k], ".0f") for k in ("E", "B", "BL")]] for g in report["U2"]["gap"]])
    rows = []
    for g in report["U2"]["gap"]:
        one, two = g.get("bl_one_run") or {}, g.get("bl_two_run")
        if two is None:
            continue
        for state in _ONE_RUN_STATES:
            if state not in one:
                continue
            o, e, s = one[state], two["p_at_n_cert"][state], two["n_cert_sound"][state]
            windows = [_num(x) for v in o.values() for x in v]
            corners = [_num(x) for v in s.values() for x in v]
            ends = {"inverse": _num(two["n_cert"][0]), "power": _num(two["n_cert"][1])}
            capped = [f"{_MODEL_LABEL[model]}, {rate}: {_f(_num(x), '.0f')}"
                      for model, v in s.items() for rate, x in zip(RATES, v) if _num(x) < ends[model]]
            rows.append([_f(g["R"], "g"), state, _pair(o["power"], ".1f"), _pair(o["inverse"], ".1f"),
                         _rng([min(windows), max(windows)], ".0f"), _pair(e["inverse"], ".3g"),
                         _pair(e["power"], ".3g"),
                         _rng([min(corners), max(corners)], ".0f")
                         + (f" (capped at {'; '.join(capped)})" if capped else "")])
    if rows:
        floor = report["settings"].get("p_floor_two_run")
        lines += ["", "B_L without the verification run (one run): D_L = D_L(p(n)) = min{D : tau_W(D) <= "
                      "eps_id p/(1 - p)} solved jointly with F22, the largest n with Delta_1(n) >= dE (1 + 2 D_L(p(n)) "
                      "sin theta_0) + (E1 - E1lb) per gap model and rate (last / Aitken; F21, F40). The two-run window "
                      f"n_cert(m_BL) above is sound while p >= {_f(floor, '.3g')} (F40): p at its ends A'/m_BL and "
                      "(A/m_BL)^(1/gamma) (last / Aitken), and the window over gap model x rate where p stays above "
                      "the floor.", ""]
        lines += _md_table(["R", "state", "one run, power (last / Aitken)", "one run, A'/n (last / Aitken)",
                            "one-run n_cert(m_BL(p))", "p at A'/m_BL", "p at (A/m_BL)^(1/gamma)",
                            "two-run n_cert(m_BL) with p >= floor"], rows)
    lines += ["", "## U3. Cost components (C_QPE units unless noted)", "",
              "C(Psi_2) carries the record of its Psi_2 (r1_*: unit step; */second_order: exact line search where the "
              "R1 record is not run).", ""]
    rows = []
    for r in report["U3"]["rows"]:
        for label in ("sf", "so"):
            x = r.get(label)
            if x is None:
                continue
            t = x["t_res1"]
            c2 = "-" if x["c_psi2"] is None else f"{_f(x['c_psi2'], '.0f')} ({x.get('c_psi2_record') or x['record']})"
            rows.append([r["system"], str(r["N"]), label, _f(r["c_qpe"], ".4g"), _f(r["lam"], ".3f"), str(r["c_w"]),
                         str(r.get("K", "-")), f"{_f(r.get('variance_m'), '.4g')} ({r.get('variance_method', '-')})",
                         _f(r.get("w_aux_m"), ".4g"), _f(x["c_psi1"], ".0f"), c2,
                         _f(x["norm_a1"], ".4f"),
                         " / ".join(f"{_f(t[k]['over_qpe'], '.3g')} {t[k]['route']}" for k in t)])
    lines += _md_table(["system", "N", "metric", "C_QPE", "lambda", "C_W", "K", "V_m (method)", "W_m", "C(Psi_1)",
                        "C(Psi_2) (record)", "||A_1||_m", "T_res(1)/C_QPE at r = " + " / ".join(
                            next((list(x[m]["t_res1"]) for x in report["U3"]["rows"] for m in ("sf", "so") if m in x),
                                 []))], rows)
    if report["U3"]["exponents"]:
        lines += ["", "Successive exponents in N at fixed R (chains, N >= 6; the records where they differ between "
                      "sizes):", ""]
        rows = []
        for e in report["U3"]["exponents"]:
            recs = [x for x in e.get("records", []) if x is not None]
            q = e["quantity"] + (f" [{', '.join(recs)}]" if len(set(recs)) > 1 else "")
            rows.append([_f(e["R"], "g"), q, ", ".join(_f(n, "g") for n in e["N"]),
                         ", ".join(_f(b, ".2f") for b in e["successive"])])
        lines += _md_table(["R", "quantity", "N", "successive"], rows)
    lines += ["", "## U4. Break-even map under repetition (R1 records; C_QPE units)", "",
              "mu_k = (1 + c_k)/p_k - (1 + c_k+1)/p_k+1, rho_k = c_iter(k)/mu_k (pays iff < 1), p_k* = "
              "[(1 + c_k) - (1 + c_k+1)/G_k]/c_iter(k); rho_LS: the exact-line-search Psi_k+1 (optimistic).", ""]
    lines += _md_table(["system", "metric", "r", "k", "p_k", "G_k", "c_iter(k)", "mu_k", "rho_k", "p_k*", "rho_LS"],
                       [[r["system"], r["metric"], _f(r["r"], "g"), str(r["k"]), _f(r["p"]), _f(r["gain"]),
                         _f(r["c_iter"], ".3g"), _f(r["mu"], ".3g"), _f(r["rho"], ".3g"), _f(r["p_star"], ".3g"),
                         _f(r["rho_ls"], ".3g")] for r in report["U4"]["rows"]])
    if report["U4"]["n_star"]:
        lines += ["", "Break-even size n* of the first measured iteration (F45, repetition), inputs: alpha_1 from the "
                      "classical-step series (overlaps[1] of the second-order record), alpha_2 from the R1 record, "
                      "each extrapolated from its own last size; c_iter(1) = T_res(1)/C_QPE on N >= 6 (the R1 record's "
                      "k = 1 cost, else the cheapest iteration at the same Psi_1), anchored at n_0; c_1 and c_2 = "
                      "C(Psi_j)/C_QPE as power laws on N >= 6, anchored at their last size.", ""]
        rows = []
        for b in report["U4"]["n_star"]:
            inp = b.get("inputs") or {}
            ak, ak1, cost = inp.get("alpha_k") or {}, inp.get("alpha_k1") or {}, inp.get("cost") or {}
            pr = b.get("prep") or {}
            prep = "-" if not pr else "; ".join(
                f"{label} {_f(pr[key]['c0'], '.3g')} at {_f(pr[key]['n0'], 'g')} (beta {_f(pr[key]['beta'], '.2f')})"
                for key, label in (("c_k", "c_1"), ("c_k1", "c_2")) if key in pr)
            rows.append([_f(b["R"], "g"), b["metric"], _f(b["r"], "g"), str(b["n0"]), _f(b["c0"], ".3g"),
                         f"{', '.join(_f(x, '.2f') for x in b['cost_successive'])} (N = "
                         f"{', '.join(str(n) for n in cost.get('ns', []))})",
                         f"{_rng(b['alpha_k'], '.4f')} (n = {', '.join(str(n) for n in ak.get('ns', []))})",
                         f"{_rng(b['alpha_k1'], '.4f')} (n = {', '.join(str(n) for n in ak1.get('ns', []))})", prep])
        lines += _md_table(["R", "metric", "r", "n_0", "c_0", "beta successive", "alpha_1 bracket", "alpha_2 bracket",
                            "preparation terms"], rows)
        lines += ["", "n* (F45, repetition): the range over rate x beta x crossover x gap model, 'none' below "
                      "n = 3000; p_1 and p_2 are extrapolated at the same rate (matched corners: both at the last "
                      "successive rate or both at the Aitken limit). 'inside k/N': cases whose n* lies within the "
                      "window of their own gap model; c_1(n*), c_2(n*) in C_QPE(n) units (the lengthened QPE divides "
                      "them by f); 'c_k = 0': F45's closed form without the preparation terms (a check); 'crossed': "
                      "(p_1, p_2) at (last, Aitken) and (Aitken, last), a sensitivity outside the bracket. "
                      + _MODEL_SCOPE, ""]
        rows = []
        for b in report["U4"]["n_star"]:
            bm, cr = b.get("by_margin") or {}, b.get("crossed") or {}
            cbm = cr.get("by_margin") or {}
            entries = [("none", b["no_window"], cr.get("no_window"))]
            entries += [(f"m_{k}", bm[k], cbm.get(k)) for k in ("E", "B", "BL") if k in bm]
            if b.get("overall_scope") == "windowed":
                entries.append(("any window", b["overall"], cr.get("overall")))
            for label, s, c in entries:
                n_cert = "-" if "n_cert" not in s else _rng(s["n_cert"], ".0f")
                inside = "-" if "n_inside" not in s else f"{s['n_inside']}/{s['n_cases']}"
                rows.append([_f(b["R"], "g"), b["metric"], _f(b["r"], "g"), label, _cases(s), n_cert, inside,
                             _rng(s["p_k"], ".2g"), f"{_rng(s.get('c_k'), '.3g')} / {_rng(s.get('c_k1'), '.3g')}",
                             _cases(s, "n_star_c0", "n_none_c0") if "n_star_c0" in s else "-", _cases(c)])
        lines += _md_table(["R", "metric", "r", "window", "n*", "n_cert", "inside", "p_1(n*)", "c_1(n*) / c_2(n*)",
                            "n* at c_k = 0", "crossed"], rows)
    return "\n".join(lines) + "\n"
